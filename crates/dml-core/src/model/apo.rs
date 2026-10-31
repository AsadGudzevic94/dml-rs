//! Average potential outcomes of a discrete (multi-valued) treatment, as
//! DoubleML's `DoubleMLAPO` and `DoubleMLAPOS`:
//!
//! ```text
//! θ(d) = E[Y(d)] = E[g(d, X)],   g(d, X) = E[Y | D = d, X],   m_d(X) = P(D = d | X)
//! ```
//!
//! estimated with the doubly robust score
//! `ψ = g(d, X) + 1{D = d}·(Y − g(d, X)) / m_d(X) − θ`. Contrasts between
//! levels (e.g. `θ(1) − θ(0)`, the average effect of level 1 against level 0)
//! come from [`AposFit::causal_contrast`].

use ndarray::{concatenate, Array1, Array2, Axis};

use super::irm::{normalize_propensity, ScoreWeights};
use super::{
    dml_model_api, nuisance_api, CateSignal, LinearScoreModel, NuisanceLearners, NuisanceTarget,
    ScoreEstimate, ScoreInput,
};
use crate::cluster::ClusterFold;
use crate::crossfit::{cross_fit_predict_proba, cross_fit_predict_subset, Fold, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};
use crate::propensity::{adjust_propensity, Calibration};
use crate::score::LinearScore;
use crate::sensitivity::{Elements, Nu2Parts, SensitivityParts};

/// Average potential outcome `E[Y(d)]` of one treatment level `d`.
///
/// `ml_g` learns `E[Y | D = d, X]` on the units at level `d`; `ml_m` learns the
/// generalised propensity `P(D = d | X)`. A second outcome model, trained on
/// `(D, X)` for the units at the other levels, enters only the sensitivity
/// analysis (as in DoubleML). Default sample splits are stratified by the
/// treatment.
pub struct Apo {
    nuisance: NuisanceLearners,
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    treatment_level: f64,
    clipping_threshold: f64,
    calibration: Calibration,
    normalize_ipw: bool,
    weights: Option<ScoreWeights>,
    splitting: SampleSplitting,
}

impl Apo {
    pub fn new(
        ml_g: impl Regressor + 'static,
        ml_m: impl Classifier + 'static,
        treatment_level: f64,
    ) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m), treatment_level)
    }

    pub fn from_boxed(
        ml_g: Box<dyn Regressor>,
        ml_m: Box<dyn Classifier>,
        treatment_level: f64,
    ) -> Self {
        Self {
            nuisance: NuisanceLearners::default(),
            ml_g,
            ml_m,
            treatment_level,
            clipping_threshold: 1e-2,
            calibration: Calibration::None,
            normalize_ipw: false,
            weights: None,
            splitting: SampleSplitting::default(),
        }
    }

    /// Propensity scores are clipped to `[t, 1 − t]` (default `t = 0.01`).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }
    /// Calibrate the propensity scores before clipping (default: none), as
    /// DoubleML's `ps_processor_config`.
    pub fn calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Normalise the inverse-probability weights, as DoubleML's `normalize_ipw=True`.
    pub fn normalize_ipw(mut self, normalize: bool) -> Self {
        self.normalize_ipw = normalize;
        self
    }

    /// Weighted average potential outcome, as DoubleML's `weights`:
    /// `E[w·g(d, X)] / E[w]` (e.g. `w = 1{X ∈ G}` for a subgroup).
    pub fn weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(ScoreWeights {
            weights,
            weights_bar: None,
        });
        self
    }

    /// Weights with separate `weights_bar` for the inverse-probability
    /// correction, one column per repetition (DoubleML's dict form).
    pub fn weights_with_bar(mut self, weights: Array1<f64>, weights_bar: Array2<f64>) -> Self {
        self.weights = Some(ScoreWeights {
            weights,
            weights_bar: Some(weights_bar),
        });
        self
    }

    dml_model_api!();
    nuisance_api!();
}

/// Index of each observation's treatment value among the sorted distinct
/// values (sample-splitting strata).
fn level_codes(d: &[f64]) -> Vec<usize> {
    let mut levels = d.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    d.iter()
        .map(|v| {
            levels
                .binary_search_by(|l| l.total_cmp(v))
                .expect("present")
        })
        .collect()
}

fn check_clipping(t: f64) -> Result<()> {
    if t > 0.0 && t < 0.5 {
        Ok(())
    } else {
        Err(DmlError::InvalidArgument(
            "clipping_threshold must be in (0, 0.5)".into(),
        ))
    }
}

impl LinearScoreModel for Apo {
    fn name(&self) -> String {
        format!("APO (treatment level {})", self.treatment_level)
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        self.nuisance
            .check("APO", &["ml_g_d_lvl1", "ml_g_d_lvl0"], &["ml_m"])?;
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "APO requires exactly one (discrete) treatment".into(),
            ));
        }
        check_clipping(self.clipping_threshold)?;
        if let Some(w) = &self.weights {
            w.check(data.n_obs())?;
        }
        let d = data.treatment(0);
        let at_level = d.iter().filter(|&&v| v == self.treatment_level).count();
        if at_level == 0 || at_level == d.len() {
            return Err(DmlError::InvalidArgument(format!(
                "treatment level {} must occur in the data, and other levels too",
                self.treatment_level
            )));
        }
        Ok(())
    }

    fn strata(&self, data: &DmlData) -> Option<Vec<usize>> {
        Some(level_codes(&data.treatment(0).to_vec()))
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, rep, .. } = *input;
        let level = self.treatment_level;
        let at_level: Vec<bool> = d.iter().map(|&v| v == level).collect();
        let others: Vec<bool> = at_level.iter().map(|t| !t).collect();
        let treated: Array1<f64> = at_level.iter().map(|&t| f64::from(t)).collect();

        // g(d, X) on the units at the level; the model for the other levels
        // also sees D (it only enters the sensitivity analysis).
        let nu = &self.nuisance;
        let g1 = input.predictions_or("ml_g_d_lvl1", || {
            let ml = nu.regressor("ml_g_d_lvl1", self.ml_g.as_ref());
            cross_fit_predict_subset(ml, x, y, folds, &at_level)
        })?;
        let dx: Array2<f64> = concatenate![Axis(1), d.insert_axis(Axis(1)), x];
        let g0 = input.predictions_or("ml_g_d_lvl0", || {
            let ml = nu.regressor("ml_g_d_lvl0", self.ml_g.as_ref());
            cross_fit_predict_subset(ml, dx.view(), y, folds, &others)
        })?;
        let m_raw = input.predictions_or("ml_m", || {
            let ml = nu.classifier("ml_m", self.ml_m.as_ref());
            cross_fit_predict_proba(ml, x, treated.view(), folds)
        })?;
        let m = adjust_propensity(
            m_raw,
            treated.view(),
            folds,
            self.clipping_threshold,
            self.calibration,
        )?;
        let m_adj = if self.normalize_ipw {
            normalize_propensity(&m, &treated)
        } else {
            m.clone()
        };

        let (w, w_bar) = ScoreWeights::ate(self.weights.as_ref(), d.len(), rep)?;
        let psi_b = &w * &g1 + &(&w_bar * &treated * &(&y - &g1) / &m_adj);
        let psi_a = -(&w / w.mean().expect("n > 0"));

        // Sensitivity: outcome residual y − ĝ(D, X), Riesz representer
        // 1{D = d} / m_d(X) and m(W, α) = 1 / m_d(X).
        let outcome_residual = &y - &(&treated * &g1) - &(&(1.0 - &treated) * &g0);
        let riesz = &w_bar * &treated / &m_adj;
        let nu2 = 2.0 * &w * &w_bar / &m_adj - &riesz.mapv(|a| a * a);
        Ok(ScoreEstimate::new(
            LinearScore::new(psi_a, psi_b.clone())?,
            vec![("ml_g_d_lvl0", g0), ("ml_g_d_lvl1", g1), ("ml_m", m)],
        )
        .with_sensitivity(SensitivityParts {
            outcome_residual,
            theta_factor: None,
            riesz,
            nu2: Nu2Parts::Elementwise(nu2),
        })
        .with_targets(vec![
            NuisanceTarget::regression("ml_g_d_lvl0", y.to_owned()).only_where(|i| !at_level[i]),
            NuisanceTarget::regression("ml_g_d_lvl1", y.to_owned()).only_where(|i| at_level[i]),
            NuisanceTarget::classification("ml_m", treated.clone()),
        ])
        // Conditional APOs (DoubleML's capo / gapo): the orthogonal signal ψ_b.
        .with_cate_signal(CateSignal {
            signal: psi_b,
            theta_factor: None,
            basis_scale: None,
            is_effect: false,
        }))
    }
}

/// Average potential outcomes of several treatment levels, fitted on shared
/// sample splits so that they (and their contrasts) have joint inference.
pub struct Apos {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    treatment_levels: Vec<f64>,
    clipping_threshold: f64,
    calibration: Calibration,
    normalize_ipw: bool,
    weights: Option<ScoreWeights>,
    splitting: SampleSplitting,
}

impl Apos {
    pub fn new(
        ml_g: impl Regressor + 'static,
        ml_m: impl Classifier + 'static,
        treatment_levels: Vec<f64>,
    ) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m), treatment_levels)
    }

    pub fn from_boxed(
        ml_g: Box<dyn Regressor>,
        ml_m: Box<dyn Classifier>,
        treatment_levels: Vec<f64>,
    ) -> Self {
        Self {
            ml_g,
            ml_m,
            treatment_levels,
            clipping_threshold: 1e-2,
            calibration: Calibration::None,
            normalize_ipw: false,
            weights: None,
            splitting: SampleSplitting::default(),
        }
    }

    /// Propensity scores are clipped to `[t, 1 − t]` (default `t = 0.01`).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }
    /// Calibrate the propensity scores before clipping (default: none), as
    /// DoubleML's `ps_processor_config`.
    pub fn calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Normalise the inverse-probability weights, as DoubleML's `normalize_ipw=True`.
    pub fn normalize_ipw(mut self, normalize: bool) -> Self {
        self.normalize_ipw = normalize;
        self
    }

    /// Weighted average potential outcome, as DoubleML's `weights`:
    /// `E[w·g(d, X)] / E[w]` (e.g. `w = 1{X ∈ G}` for a subgroup).
    pub fn weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(ScoreWeights {
            weights,
            weights_bar: None,
        });
        self
    }

    /// Weights with separate `weights_bar` for the inverse-probability
    /// correction, one column per repetition (DoubleML's dict form).
    pub fn weights_with_bar(mut self, weights: Array1<f64>, weights_bar: Array2<f64>) -> Self {
        self.weights = Some(ScoreWeights {
            weights,
            weights_bar: Some(weights_bar),
        });
        self
    }

    /// Number of cross-fitting folds (default 5).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Number of independent sample splits (default 1).
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed for the sample splits (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    fn level_model(&self, level: f64) -> Apo {
        let model = Apo::from_boxed(self.ml_g.fresh(), self.ml_m.fresh(), level)
            .clipping_threshold(self.clipping_threshold)
            .calibration(self.calibration)
            .normalize_ipw(self.normalize_ipw);
        match &self.weights {
            Some(w) => Apo {
                weights: Some(w.clone()),
                ..model
            },
            None => model,
        }
    }

    fn check_levels(&self, data: &DmlData) -> Result<()> {
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "APOS requires exactly one (discrete) treatment".into(),
            ));
        }
        if self.treatment_levels.is_empty() {
            return Err(DmlError::InvalidArgument(
                "at least one treatment level is required".into(),
            ));
        }
        let d = data.treatment(0);
        for (i, &level) in self.treatment_levels.iter().enumerate() {
            if self.treatment_levels[..i].contains(&level) {
                return Err(DmlError::InvalidArgument(format!(
                    "treatment level {level} is listed twice"
                )));
            }
            if !d.iter().any(|&v| v == level) {
                return Err(DmlError::InvalidArgument(format!(
                    "treatment level {level} does not occur in the data"
                )));
            }
        }
        Ok(())
    }

    /// Draw sample splits (stratified by the treatment, or by cluster for
    /// clustered data) and fit every level on them.
    pub fn fit(&self, data: &DmlData) -> Result<AposFit> {
        self.check_levels(data)?;
        match data.clusters() {
            Some(clusters) => {
                let splits = self.splitting.draw_clustered(clusters)?;
                self.fit_with_cluster_folds(data, &splits)
            }
            None => {
                let strata = level_codes(&data.treatment(0).to_vec());
                let splits = self.splitting.draw_stratified(&strata)?;
                self.fit_with_folds(data, &splits)
            }
        }
    }

    /// Fit every level on the given splits (one `Vec<Fold>` per repetition).
    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<AposFit> {
        self.check_levels(data)?;
        let fits = self
            .treatment_levels
            .iter()
            .map(|&level| self.level_model(level).fit_with_folds(data, splits))
            .collect::<Result<Vec<_>>>()?;
        self.assemble(fits)
    }

    /// Fit every level of clustered data on the given cluster splits.
    pub fn fit_with_cluster_folds(
        &self,
        data: &DmlData,
        splits: &[Vec<ClusterFold>],
    ) -> Result<AposFit> {
        self.check_levels(data)?;
        let fits = self
            .treatment_levels
            .iter()
            .map(|&level| self.level_model(level).fit_with_cluster_folds(data, splits))
            .collect::<Result<Vec<_>>>()?;
        self.assemble(fits)
    }

    fn assemble(&self, fits: Vec<DmlFit>) -> Result<AposFit> {
        let mut fit = DmlFit::concat(&fits)?;
        fit.model = "APOS".into();
        fit.treatment_names = self
            .treatment_levels
            .iter()
            .map(|l| l.to_string())
            .collect();
        Ok(AposFit {
            levels: self.treatment_levels.clone(),
            fit,
        })
    }
}

/// Fitted [`Apos`]: the average potential outcome of every level, with joint
/// inference, and contrasts between levels.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AposFit {
    levels: Vec<f64>,
    fit: DmlFit,
}

impl AposFit {
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// One estimate per treatment level (named by the level).
    pub fn apos(&self) -> &DmlFit {
        &self.fit
    }

    /// Average effects of each level against each reference level,
    /// `θ(d) − θ(d_ref)`, named `"d vs d_ref"`, as DoubleML's
    /// `causal_contrast`: for every reference level in turn, all levels not yet
    /// used as a reference. Inference accounts for the correlation between the
    /// estimates; sensitivity analysis combines the elements of both levels
    /// (average `σ²`, summed `ν²`), as DoubleML.
    pub fn causal_contrast(&self, reference_levels: &[f64]) -> Result<DmlFit> {
        let k = self.levels.len();
        if k < 2 {
            return Err(DmlError::InvalidArgument(
                "only one treatment level: no causal contrast can be computed".into(),
            ));
        }
        let mut rows: Vec<(usize, usize)> = Vec::new();
        let mut used = vec![false; k];
        for &reference in reference_levels {
            let r = self
                .levels
                .iter()
                .position(|&l| l == reference)
                .ok_or_else(|| {
                    DmlError::InvalidArgument(format!(
                        "reference level {reference} is not one of the fitted treatment levels"
                    ))
                })?;
            used[r] = true;
            rows.extend((0..k).filter(|&i| !used[i]).map(|i| (i, r)));
        }
        if rows.is_empty() {
            return Err(DmlError::InvalidArgument(
                "no contrasts: give at least one reference level".into(),
            ));
        }
        let mut weights = Array2::<f64>::zeros((rows.len(), k));
        for (c, &(i, r)) in rows.iter().enumerate() {
            weights[[c, i]] = 1.0;
            weights[[c, r]] = -1.0;
        }
        let names = rows
            .iter()
            .map(|&(i, r)| format!("{} vs {}", self.levels[i], self.levels[r]))
            .collect();
        let mut contrast = self.fit.combine("APOS contrasts".into(), names, &weights);
        if let Some(store) = &self.fit.sensitivity {
            let n_rep = self.fit.n_rep();
            let mut combined = crate::sensitivity::SensitivityStore::new(
                self.fit.n_obs,
                rows.len(),
                n_rep,
                store.var_y,
            );
            for (c, &(i, r)) in rows.iter().enumerate() {
                for rep in 0..n_rep {
                    let (a, b) = (store.elements(i, rep), store.elements(r, rep));
                    combined.set(
                        c,
                        rep,
                        &Elements {
                            sigma2: (a.sigma2 + b.sigma2) / 2.0,
                            nu2: a.nu2 + b.nu2,
                            psi_sigma2: (&a.psi_sigma2 + &b.psi_sigma2) / 2.0,
                            psi_nu2: &a.psi_nu2 + &b.psi_nu2,
                            riesz: None,
                            nu2_fallback: false,
                        },
                    );
                }
            }
            contrast.sensitivity = Some(combined);
        }
        Ok(contrast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use ndarray::array;

    #[test]
    fn level_codes_index_sorted_distinct_values() {
        assert_eq!(level_codes(&[2.0, 0.0, 1.0, 2.0, 0.0]), vec![2, 0, 1, 2, 0]);
    }

    #[test]
    fn rejects_missing_levels_and_single_level_contrasts() {
        let data = DmlData::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![0.0, 1.0, 0.0, 1.0],
            array![[1.0], [2.0], [3.0], [4.0]],
        )
        .unwrap();
        assert!(Apo::new(Ols::new(), LogisticRegression::default(), 5.0)
            .validate(&data)
            .is_err());
        let apos = Apos::new(Ols::new(), LogisticRegression::default(), vec![0.0, 0.0]);
        assert!(apos.check_levels(&data).is_err());
    }
}

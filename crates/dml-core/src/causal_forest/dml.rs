//! Forests on cross-fitted residuals, with doubly robust inference on averages.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use super::forests::{CausalForest, InstrumentalForest, MultiArmCausalForest};
use super::ForestSettings;
use crate::blp::Blp;
use crate::crossfit::{cross_fit_predict, cross_fit_predict_proba, Fold, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};
use crate::model::TreatmentLearner;
use crate::policy::PolicyTree;

fn cross_fit_treatment(
    learner: &TreatmentLearner,
    x: ArrayView2<'_, f64>,
    v: ArrayView1<'_, f64>,
    folds: &[Fold],
) -> Result<Array1<f64>> {
    match learner {
        TreatmentLearner::Regressor(r) => cross_fit_predict(r.as_ref(), x, v, folds),
        TreatmentLearner::Classifier(c) => cross_fit_predict_proba(c.as_ref(), x, v, folds),
    }
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

fn is_binary(v: ArrayView1<'_, f64>) -> bool {
    v.iter().all(|&v| v == 0.0 || v == 1.0)
}

fn check_scores(tau: &Array1<f64>) -> Result<()> {
    if tau.iter().any(|v| !v.is_finite()) {
        return Err(DmlError::InvalidArgument(
            "some rows have no out-of-bag estimate; grow more trees".into(),
        ));
    }
    Ok(())
}

/// Average of doubly robust scores (`n × k`) as a [`DmlFit`].
fn average(model: &str, names: Vec<String>, n_folds: usize, gamma: &Array2<f64>) -> Result<DmlFit> {
    let (n, k) = gamma.dim();
    let psi_b = gamma
        .clone()
        .into_shape_with_order((n, k, 1))
        .expect("shape");
    DmlFit::from_linear_scores(
        model,
        names,
        n_folds,
        Array3::from_elem((n, k, 1), -1.0),
        psi_b,
    )
}

/// Builder settings shared by the forests on residuals.
macro_rules! forest_builder {
    () => {
        /// Forest settings (default [`ForestSettings::default`], grf's defaults).
        pub fn forest(mut self, settings: ForestSettings) -> Self {
            self.forest = settings;
            self
        }

        /// Folds for the nuisance models (default 5).
        pub fn n_folds(mut self, n_folds: usize) -> Self {
            self.splitting.n_folds = n_folds;
            self
        }

        /// Seed of the nuisance folds (the forest has its own seed).
        pub fn seed(mut self, seed: u64) -> Self {
            self.splitting.seed = seed;
            self
        }

        /// Propensities are clipped to `[t, 1 − t]` in the doubly robust
        /// scores (default 0.01).
        pub fn clipping_threshold(mut self, t: f64) -> Self {
            self.clipping_threshold = t;
            self
        }
    };
}

// ---------------------------------------------------------------------------
// Causal forest
// ---------------------------------------------------------------------------

/// A causal forest on cross-fitted residuals: `ml_y` learns `E[Y | X]`,
/// `ml_d` learns `E[D | X]` (a classifier for a binary treatment), both
/// cross-fitted on `n_folds` folds, then the forest is grown on the residuals.
pub struct CausalForestDml {
    ml_y: Box<dyn Regressor>,
    ml_d: TreatmentLearner,
    forest: ForestSettings,
    splitting: SampleSplitting,
    clipping_threshold: f64,
}

impl CausalForestDml {
    pub fn new(ml_y: impl Regressor + 'static, ml_d: TreatmentLearner) -> Self {
        Self::from_boxed(Box::new(ml_y), ml_d)
    }

    pub fn from_boxed(ml_y: Box<dyn Regressor>, ml_d: TreatmentLearner) -> Self {
        Self {
            ml_y,
            ml_d,
            forest: ForestSettings::default(),
            splitting: SampleSplitting::default(),
            clipping_threshold: 1e-2,
        }
    }

    forest_builder!();

    pub fn fit(&self, data: &DmlData) -> Result<CausalForestFit> {
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "a causal forest needs exactly one treatment".into(),
            ));
        }
        check_clipping(self.clipping_threshold)?;
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let folds = self.splitting.draw(data.n_obs())?;
        let folds = &folds[0];
        let y_hat = cross_fit_predict(self.ml_y.as_ref(), x, y, folds)?;
        let d_hat = cross_fit_treatment(&self.ml_d, x, d, folds)?;
        let y_res = &y - &y_hat;
        let d_res = &d - &d_hat;
        let forest = CausalForest::fit(x, y_res.view(), d_res.view(), self.forest.clone())?;
        let (tau_oob, var_oob) = forest.predict_oob(x)?;
        Ok(CausalForestFit {
            forest,
            y: y.to_owned(),
            d: d.to_owned(),
            y_hat,
            d_hat,
            tau_oob,
            var_oob,
            binary: is_binary(d),
            clipping_threshold: self.clipping_threshold,
            treatment_name: data.treatment_names()[0].clone(),
            n_folds: self.splitting.n_folds,
        })
    }
}

/// A fitted [`CausalForestDml`].
#[derive(Debug, Clone)]
pub struct CausalForestFit {
    forest: CausalForest,
    y: Array1<f64>,
    d: Array1<f64>,
    y_hat: Array1<f64>,
    d_hat: Array1<f64>,
    tau_oob: Array1<f64>,
    var_oob: Array1<f64>,
    binary: bool,
    clipping_threshold: f64,
    treatment_name: String,
    n_folds: usize,
}

impl CausalForestFit {
    pub fn forest(&self) -> &CausalForest {
        &self.forest
    }

    /// `τ̂(x)` and its variance at new points.
    pub fn effect(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        self.forest.predict(x)
    }

    /// Out-of-bag `τ̂(Xᵢ)` of the training rows.
    pub fn oob_effect(&self) -> &Array1<f64> {
        &self.tau_oob
    }

    /// Variances of the out-of-bag estimates.
    pub fn oob_variance(&self) -> &Array1<f64> {
        &self.var_oob
    }

    /// Cross-fitted `E[Y | X]`.
    pub fn y_hat(&self) -> &Array1<f64> {
        &self.y_hat
    }

    /// Cross-fitted `E[D | X]`.
    pub fn d_hat(&self) -> &Array1<f64> {
        &self.d_hat
    }

    pub fn is_binary_treatment(&self) -> bool {
        self.binary
    }

    /// Doubly robust scores of a binary treatment,
    /// `τ̂(Xᵢ) + (Dᵢ − êᵢ) / (êᵢ(1 − êᵢ)) · (Yᵢ − ŷᵢ − (Dᵢ − êᵢ) τ̂(Xᵢ))`,
    /// with out-of-bag `τ̂` and clipped propensities `ê` (grf's AIPW scores).
    pub fn dr_scores(&self) -> Result<Array1<f64>> {
        if !self.binary {
            return Err(DmlError::InvalidArgument(
                "doubly robust scores (ATE, BLP, GATE) need a binary 0/1 treatment".into(),
            ));
        }
        check_scores(&self.tau_oob)?;
        let t = self.clipping_threshold;
        Ok(Array1::from_iter((0..self.y.len()).map(|i| {
            let e = self.d_hat[i].clamp(t, 1.0 - t);
            let dr = self.d[i] - e;
            let tau = self.tau_oob[i];
            tau + dr / (e * (1.0 - e)) * (self.y[i] - self.y_hat[i] - dr * tau)
        })))
    }

    /// Average treatment effect from the doubly robust scores, with the usual
    /// DML standard error.
    pub fn ate(&self) -> Result<DmlFit> {
        let gamma = self.dr_scores()?.insert_axis(Axis(1));
        average(
            "Causal forest (ATE)",
            vec![self.treatment_name.clone()],
            self.n_folds,
            &gamma,
        )
    }

    /// Best linear predictor of the CATE on `basis` (HC0 inference on the
    /// doubly robust scores).
    pub fn blp(&self, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        let gamma = self.dr_scores()?;
        Blp::fit(gamma.view().insert_axis(Axis(1)), basis)
    }

    /// Group average treatment effects for 0/1 group dummies.
    pub fn gate(&self, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        let gamma = self.dr_scores()?;
        Blp::fit_gate(gamma.view().insert_axis(Axis(1)), groups)
    }

    /// Policy tree on the doubly robust scores: a shallow tree of `features`
    /// recommending treatment where the effect is positive (the tree
    /// classifies the sign of the scores, weighted by their size). Defaults as
    /// [`DmlFit::policy_tree`]: `depth = 2`, `min_samples_leaf = 8`,
    /// `ccp_alpha = 0.01`.
    pub fn policy_tree(
        &self,
        features: ArrayView2<'_, f64>,
        depth: usize,
        min_samples_leaf: usize,
        ccp_alpha: f64,
    ) -> Result<PolicyTree> {
        let gamma = self.dr_scores()?;
        PolicyTree::fit(gamma.view(), features, depth, min_samples_leaf, ccp_alpha)
    }
}

// ---------------------------------------------------------------------------
// Instrumental forest
// ---------------------------------------------------------------------------

/// An instrumental forest on cross-fitted residuals: `ml_y`, `ml_d` and `ml_z`
/// learn `E[Y | X]`, `E[D | X]` and `E[Z | X]` (classifiers for binary
/// variables); the forest estimates the conditional local average treatment
/// effect `θ(x)` on the residuals (grf's `instrumental_forest`). The data need
/// one treatment and one instrument ([`DmlData::with_instrument`]).
pub struct InstrumentalForestDml {
    ml_y: Box<dyn Regressor>,
    ml_d: TreatmentLearner,
    ml_z: TreatmentLearner,
    forest: ForestSettings,
    splitting: SampleSplitting,
    clipping_threshold: f64,
    reduced_form_weight: f64,
}

impl InstrumentalForestDml {
    pub fn new(
        ml_y: impl Regressor + 'static,
        ml_d: TreatmentLearner,
        ml_z: TreatmentLearner,
    ) -> Self {
        Self::from_boxed(Box::new(ml_y), ml_d, ml_z)
    }

    pub fn from_boxed(
        ml_y: Box<dyn Regressor>,
        ml_d: TreatmentLearner,
        ml_z: TreatmentLearner,
    ) -> Self {
        Self {
            ml_y,
            ml_d,
            ml_z,
            forest: ForestSettings::default(),
            splitting: SampleSplitting::default(),
            clipping_threshold: 1e-2,
            reduced_form_weight: 0.0,
        }
    }

    forest_builder!();

    /// grf's `reduced.form.weight`: the splits target `(1 − r) Z + r W`
    /// (default 0, the instrument).
    pub fn reduced_form_weight(mut self, r: f64) -> Self {
        self.reduced_form_weight = r;
        self
    }

    pub fn fit(&self, data: &DmlData) -> Result<InstrumentalForestFit> {
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "an instrumental forest needs exactly one treatment".into(),
            ));
        }
        let z = data.z().ok_or_else(|| {
            DmlError::InvalidArgument("an instrumental forest needs exactly one instrument".into())
        })?;
        check_clipping(self.clipping_threshold)?;
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let folds = self.splitting.draw(data.n_obs())?;
        let folds = &folds[0];
        let y_hat = cross_fit_predict(self.ml_y.as_ref(), x, y, folds)?;
        let d_hat = cross_fit_treatment(&self.ml_d, x, d, folds)?;
        let z_hat = cross_fit_treatment(&self.ml_z, x, z, folds)?;
        let (y_res, d_res, z_res) = (&y - &y_hat, &d - &d_hat, &z - &z_hat);
        let forest = InstrumentalForest::fit(
            x,
            y_res.view(),
            d_res.view(),
            z_res.view(),
            self.forest.clone(),
            self.reduced_form_weight,
        )?;
        let (tau_oob, var_oob) = forest.predict_oob(x)?;
        // Compliance E[D | Z = 1, X] − E[D | Z = 0, X]: a causal forest of the
        // treatment on the instrument (grf: 500 trees).
        let binary_z = is_binary(z);
        let compliance = if binary_z {
            let l = self.forest.ci_group_size;
            let trees = (self.forest.num_trees.min(500) / l).max(1) * l;
            let settings = ForestSettings {
                num_trees: trees,
                seed: self.forest.seed.wrapping_add(1),
                ..self.forest.clone()
            };
            let cf = CausalForest::fit(x, d_res.view(), z_res.view(), settings)?;
            Some(cf.predict_oob(x)?.0)
        } else {
            None
        };
        Ok(InstrumentalForestFit {
            forest,
            y: y.to_owned(),
            d: d.to_owned(),
            z: z.to_owned(),
            y_hat,
            d_hat,
            z_hat,
            tau_oob,
            var_oob,
            compliance,
            clipping_threshold: self.clipping_threshold,
            treatment_name: data.treatment_names()[0].clone(),
            n_folds: self.splitting.n_folds,
        })
    }
}

/// A fitted [`InstrumentalForestDml`].
#[derive(Debug, Clone)]
pub struct InstrumentalForestFit {
    forest: InstrumentalForest,
    y: Array1<f64>,
    d: Array1<f64>,
    z: Array1<f64>,
    y_hat: Array1<f64>,
    d_hat: Array1<f64>,
    z_hat: Array1<f64>,
    tau_oob: Array1<f64>,
    var_oob: Array1<f64>,
    compliance: Option<Array1<f64>>,
    clipping_threshold: f64,
    treatment_name: String,
    n_folds: usize,
}

impl InstrumentalForestFit {
    pub fn forest(&self) -> &InstrumentalForest {
        &self.forest
    }

    /// `θ̂(x)` and its variance at new points.
    pub fn effect(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        self.forest.predict(x)
    }

    pub fn oob_effect(&self) -> &Array1<f64> {
        &self.tau_oob
    }

    pub fn oob_variance(&self) -> &Array1<f64> {
        &self.var_oob
    }

    pub fn y_hat(&self) -> &Array1<f64> {
        &self.y_hat
    }

    pub fn d_hat(&self) -> &Array1<f64> {
        &self.d_hat
    }

    pub fn z_hat(&self) -> &Array1<f64> {
        &self.z_hat
    }

    /// Out-of-bag compliance scores `E[D | Z = 1, X] − E[D | Z = 0, X]` (binary
    /// instrument).
    pub fn compliance(&self) -> Option<&Array1<f64>> {
        self.compliance.as_ref()
    }

    /// Doubly robust scores of a binary instrument (grf's
    /// `get_scores.instrumental_forest`):
    /// `θ̂(Xᵢ) + (Zᵢ − ẑᵢ) / (ẑᵢ(1 − ẑᵢ) cᵢ) · (Yᵢ − ŷᵢ − (Dᵢ − d̂ᵢ) θ̂(Xᵢ))`
    /// with the compliance score `cᵢ` and clipped `ẑ`.
    pub fn dr_scores(&self) -> Result<Array1<f64>> {
        let c = self.compliance.as_ref().ok_or_else(|| {
            DmlError::InvalidArgument(
                "doubly robust scores (average effect, BLP, GATE) need a binary 0/1 instrument"
                    .into(),
            )
        })?;
        check_scores(&self.tau_oob)?;
        if c.iter().any(|v| !v.is_finite() || *v == 0.0) {
            return Err(DmlError::InvalidArgument(
                "some compliance scores are zero or missing; the instrument is too weak there"
                    .into(),
            ));
        }
        let t = self.clipping_threshold;
        Ok(Array1::from_iter((0..self.y.len()).map(|i| {
            let e = self.z_hat[i].clamp(t, 1.0 - t);
            let tau = self.tau_oob[i];
            let res = self.y[i] - self.y_hat[i] - (self.d[i] - self.d_hat[i]) * tau;
            tau + (self.z[i] - e) / (e * (1.0 - e) * c[i]) * res
        })))
    }

    /// Average of the conditional local average treatment effects, from the
    /// doubly robust scores.
    pub fn ate(&self) -> Result<DmlFit> {
        let gamma = self.dr_scores()?.insert_axis(Axis(1));
        average(
            "Instrumental forest (average LATE)",
            vec![self.treatment_name.clone()],
            self.n_folds,
            &gamma,
        )
    }

    pub fn blp(&self, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        let gamma = self.dr_scores()?;
        Blp::fit(gamma.view().insert_axis(Axis(1)), basis)
    }

    pub fn gate(&self, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        let gamma = self.dr_scores()?;
        Blp::fit_gate(gamma.view().insert_axis(Axis(1)), groups)
    }
}

// ---------------------------------------------------------------------------
// Multi-arm causal forest
// ---------------------------------------------------------------------------

/// A multi-arm causal forest on cross-fitted residuals (grf's
/// `multi_arm_causal_forest`). The treatment is one column of discrete levels;
/// the smallest is the baseline. `ml_y` learns `E[Y | X]`, `ml_d` the
/// propensity of every level (one-vs-rest, normalised to sum to one); the
/// forest estimates the effects of the other levels against the baseline.
pub struct MultiArmCausalForestDml {
    ml_y: Box<dyn Regressor>,
    ml_d: Box<dyn Classifier>,
    forest: ForestSettings,
    splitting: SampleSplitting,
    clipping_threshold: f64,
}

impl MultiArmCausalForestDml {
    pub fn new(ml_y: impl Regressor + 'static, ml_d: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_y), Box::new(ml_d))
    }

    pub fn from_boxed(ml_y: Box<dyn Regressor>, ml_d: Box<dyn Classifier>) -> Self {
        Self {
            ml_y,
            ml_d,
            forest: ForestSettings::default(),
            splitting: SampleSplitting::default(),
            clipping_threshold: 1e-2,
        }
    }

    forest_builder!();

    pub fn fit(&self, data: &DmlData) -> Result<MultiArmCausalForestFit> {
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "a multi-arm causal forest takes the arms as one treatment column of levels".into(),
            ));
        }
        check_clipping(self.clipping_threshold)?;
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let mut levels: Vec<f64> = d.to_vec();
        levels.sort_by(f64::total_cmp);
        levels.dedup();
        if levels.len() < 2 {
            return Err(DmlError::InvalidArgument(
                "the treatment needs at least two levels".into(),
            ));
        }
        let n = data.n_obs();
        let folds = self.splitting.draw(n)?;
        let folds = &folds[0];
        let y_hat = cross_fit_predict(self.ml_y.as_ref(), x, y, folds)?;
        let mut propensity = Array2::zeros((n, levels.len()));
        for (j, &level) in levels.iter().enumerate() {
            let ind = d.mapv(|v| f64::from(v == level));
            let p = cross_fit_predict_proba(self.ml_d.as_ref(), x, ind.view(), folds)?;
            propensity.column_mut(j).assign(&p);
        }
        for mut row in propensity.rows_mut() {
            let s = row.sum();
            if s.is_nan() || s <= 0.0 {
                return Err(DmlError::InvalidArgument(
                    "the propensities of some row sum to zero".into(),
                ));
            }
            row /= s;
        }
        let k = levels.len() - 1;
        let w_res = Array2::from_shape_fn((n, k), |(i, a)| {
            f64::from(d[i] == levels[a + 1]) - propensity[[i, a + 1]]
        });
        let y_res = &y - &y_hat;
        let forest = MultiArmCausalForest::fit(x, y_res.view(), w_res.view(), self.forest.clone())?;
        let (tau_oob, var_oob) = forest.predict_oob(x)?;
        let name = &data.treatment_names()[0];
        let arm_names = levels[1..]
            .iter()
            .map(|l| format!("{name}={l} vs {}", levels[0]))
            .collect();
        Ok(MultiArmCausalForestFit {
            forest,
            y: y.to_owned(),
            d: d.to_owned(),
            levels,
            y_hat,
            propensity,
            tau_oob,
            var_oob,
            clipping_threshold: self.clipping_threshold,
            arm_names,
            n_folds: self.splitting.n_folds,
        })
    }
}

/// A fitted [`MultiArmCausalForestDml`]. Effects are `n × k` for the `k`
/// non-baseline levels, in increasing order.
#[derive(Debug, Clone)]
pub struct MultiArmCausalForestFit {
    forest: MultiArmCausalForest,
    y: Array1<f64>,
    d: Array1<f64>,
    levels: Vec<f64>,
    y_hat: Array1<f64>,
    propensity: Array2<f64>,
    tau_oob: Array2<f64>,
    var_oob: Array2<f64>,
    clipping_threshold: f64,
    arm_names: Vec<String>,
    n_folds: usize,
}

impl MultiArmCausalForestFit {
    pub fn forest(&self) -> &MultiArmCausalForest {
        &self.forest
    }

    /// Treatment levels; the first is the baseline.
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// Names of the contrasts, `"d=level vs baseline"`.
    pub fn arm_names(&self) -> &[String] {
        &self.arm_names
    }

    /// Effects against the baseline and their variances at new points (`n × k`).
    pub fn effect(&self, x: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        self.forest.predict(x)
    }

    pub fn oob_effect(&self) -> &Array2<f64> {
        &self.tau_oob
    }

    pub fn oob_variance(&self) -> &Array2<f64> {
        &self.var_oob
    }

    pub fn y_hat(&self) -> &Array1<f64> {
        &self.y_hat
    }

    /// Cross-fitted propensities of every level (`n × (k + 1)`, rows sum to one).
    pub fn propensity(&self) -> &Array2<f64> {
        &self.propensity
    }

    /// Doubly robust scores of every contrast (grf's
    /// `get_scores.multi_arm_causal_forest`): `Γ̂ₐ − Γ̂₀` with
    /// `Γ̂ⱼ = μ̂ⱼ(Xᵢ) + 1{Dᵢ = j} (Yᵢ − μ̂ⱼ(Xᵢ)) / êⱼ(Xᵢ)` and
    /// `μ̂ⱼ = ŷ − Σₐ êₐ τ̂ₐ + τ̂ⱼ` (clipped propensities). `n × k`.
    pub fn dr_scores(&self) -> Result<Array2<f64>> {
        if self.tau_oob.iter().any(|v| !v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "some rows have no out-of-bag estimate; grow more trees".into(),
            ));
        }
        let (n, k) = self.tau_oob.dim();
        let t = self.clipping_threshold;
        let mut out = Array2::zeros((n, k));
        for i in 0..n {
            let tau = self.tau_oob.row(i);
            let base = self.y_hat[i]
                - (0..k)
                    .map(|a| self.propensity[[i, a + 1]] * tau[a])
                    .sum::<f64>();
            let observed = self
                .levels
                .iter()
                .position(|&l| l == self.d[i])
                .expect("level");
            let mu = if observed == 0 {
                base
            } else {
                base + tau[observed - 1]
            };
            let e = self.propensity[[i, observed]].clamp(t, 1.0 - t);
            let ipw = (self.y[i] - mu) / e;
            for a in 0..k {
                let mut g = tau[a];
                if observed == a + 1 {
                    g += ipw;
                } else if observed == 0 {
                    g -= ipw;
                }
                out[[i, a]] = g;
            }
        }
        Ok(out)
    }

    /// Average effect of every arm against the baseline.
    pub fn ate(&self) -> Result<DmlFit> {
        let gamma = self.dr_scores()?;
        average(
            "Multi-arm causal forest (ATE)",
            self.arm_names.clone(),
            self.n_folds,
            &gamma,
        )
    }

    fn arm_scores(&self, arm: usize) -> Result<Array2<f64>> {
        let gamma = self.dr_scores()?;
        if arm >= gamma.ncols() {
            return Err(DmlError::InvalidArgument(format!(
                "arm {arm} out of range: there are {} non-baseline arms",
                gamma.ncols()
            )));
        }
        Ok(gamma.column(arm).to_owned().insert_axis(Axis(1)))
    }

    /// Best linear predictor of arm `arm`'s effect on `basis`.
    pub fn blp(&self, arm: usize, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        Blp::fit(self.arm_scores(arm)?.view(), basis)
    }

    /// Group average effects of arm `arm`.
    pub fn gate(&self, arm: usize, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        Blp::fit_gate(self.arm_scores(arm)?.view(), groups)
    }
}

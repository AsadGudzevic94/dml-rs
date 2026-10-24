//! Interactive regression model (IRM) for a binary treatment:
//!
//! ```text
//! Y = g(D, X) + U,   E[U | X, D] = 0
//! D = m(X) + V,      E[V | X] = 0,   m(X) = P(D = 1 | X)
//! ```
//!
//! Targets the average treatment effect (ATE) or the average treatment effect
//! on the treated (ATTE) with the doubly robust (AIPW) score.

use ndarray::{Array1, ArrayView1, ArrayView2};

use super::{dml_model_api, LinearScoreModel, NuisanceTarget, ScoreEstimate, ScoreInput};
use crate::crossfit::{cross_fit_predict_proba, cross_fit_predict_subset, Fold, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};
use crate::score::LinearScore;
use crate::sensitivity::{Nu2Parts, SensitivityParts};

/// Target parameter of [`Irm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IrmScore {
    /// `E[g(1, X) − g(0, X)]`
    #[default]
    Ate,
    /// `E[g(1, X) − g(0, X) | D = 1]`
    Atte,
}

/// IRM estimator: `ml_g` learns `E[Y | D = d, X]` (fitted separately on the
/// treated and control rows of each training fold), `ml_m` learns the
/// propensity score `P(D = 1 | X)`.
pub struct Irm {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    score: IrmScore,
    clipping_threshold: f64,
    normalize_ipw: bool,
    splitting: SampleSplitting,
}

impl Irm {
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Box<dyn Classifier>) -> Self {
        Self {
            ml_g,
            ml_m,
            score: IrmScore::default(),
            clipping_threshold: 1e-2,
            normalize_ipw: false,
            splitting: SampleSplitting::default(),
        }
    }

    pub fn score(mut self, score: IrmScore) -> Self {
        self.score = score;
        self
    }

    /// Propensity scores are clipped to `[t, 1 − t]` (default `t = 0.01`).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }

    /// Normalise the inverse-probability weights to mean one within each
    /// treatment arm (Hájek-style), as DoubleML's `normalize_ipw=True`.
    pub fn normalize_ipw(mut self, normalize: bool) -> Self {
        self.normalize_ipw = normalize;
        self
    }

    dml_model_api!();
}

/// DoubleML's `_normalize_ipw`: rescale `m̂` so that `mean(D/m̂)` and
/// `mean((1−D)/(1−m̂))` both become one.
pub(crate) fn normalize_propensity(m: &Array1<f64>, d: &Array1<f64>) -> Array1<f64> {
    let n = m.len() as f64;
    let mean1 = d.iter().zip(m).map(|(&di, &mi)| di / mi).sum::<f64>() / n;
    let mean0 = d
        .iter()
        .zip(m)
        .map(|(&di, &mi)| (1.0 - di) / (1.0 - mi))
        .sum::<f64>()
        / n;
    d.iter()
        .zip(m)
        .map(|(&di, &mi)| di * mi * mean1 + (1.0 - di) * (1.0 - (1.0 - mi) * mean0))
        .collect()
}

impl LinearScoreModel for Irm {
    fn name(&self) -> String {
        match self.score {
            IrmScore::Ate => "IRM (ATE)".into(),
            IrmScore::Atte => "IRM (ATTE)".into(),
        }
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return Err(DmlError::InvalidArgument(
                "IRM requires exactly one binary (0/1) treatment".into(),
            ));
        }
        if !(self.clipping_threshold > 0.0 && self.clipping_threshold < 0.5) {
            return Err(DmlError::InvalidArgument(
                "clipping_threshold must be in (0, 0.5)".into(),
            ));
        }
        Ok(())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, .. } = *input;
        let (g0, g1, m) = irm_nuisances(
            self.ml_g.as_ref(),
            self.ml_m.as_ref(),
            x,
            y,
            d,
            folds,
            self.clipping_threshold,
        )?;
        let d = d.to_owned();
        let m_adj = if self.normalize_ipw {
            normalize_propensity(&m, &d)
        } else {
            m.clone()
        };
        let ipw = ipw_residual(y, &d, &g0, &g1, &m_adj);

        let n = d.len();
        let (weights, weights_bar) = match self.score {
            IrmScore::Ate => (Array1::ones(n), Array1::ones(n)),
            IrmScore::Atte => {
                let p = d.mean().expect("n > 0");
                (&d / p, &m_adj / p)
            }
        };
        let psi_b = &weights * &(&g1 - &g0) + &weights_bar * &ipw;
        let psi_a = -(&weights / weights.mean().expect("n > 0"));

        // Sensitivity: outcome residual y − ĝ(D, X); Riesz representer and
        // m(W, α) of the (weighted) average treatment effect.
        let outcome_residual = &y - &(&d * &g1) - &(&(1.0 - &d) * &g0);
        let m_alpha = &weights * &weights_bar * &(1.0 / &m_adj + 1.0 / &(1.0 - &m_adj));
        let riesz = &weights_bar * &(&d / &m_adj - &(1.0 - &d) / &(1.0 - &m_adj));
        let nu2 = 2.0 * &m_alpha - &riesz.mapv(|a| a * a);
        Ok(ScoreEstimate::new(
            LinearScore::new(psi_a, psi_b)?,
            vec![("ml_g0", g0), ("ml_g1", g1), ("ml_m", m)],
        )
        .with_sensitivity(SensitivityParts {
            outcome_residual,
            theta_factor: None,
            riesz,
            nu2: Nu2Parts::Elementwise(nu2),
        })
        .with_targets(vec![
            NuisanceTarget::regression("ml_g0", y.to_owned()).only_where(|i| d[i] == 0.0),
            NuisanceTarget::regression("ml_g1", y.to_owned()).only_where(|i| d[i] == 1.0),
            NuisanceTarget::classification("ml_m", d.clone()),
        ]))
    }
}

/// Cross-fitted IRM nuisances: `ĝ₀` (trained on controls), `ĝ₁` (trained on
/// treated) and the clipped propensity `m̂`.
pub(crate) fn irm_nuisances(
    ml_g: &dyn Regressor,
    ml_m: &dyn Classifier,
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    d: ArrayView1<'_, f64>,
    folds: &[Fold],
    clipping_threshold: f64,
) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
    let treated: Vec<bool> = d.iter().map(|&v| v == 1.0).collect();
    let control: Vec<bool> = treated.iter().map(|t| !t).collect();
    let g0 = cross_fit_predict_subset(ml_g, x, y, folds, &control)?;
    let g1 = cross_fit_predict_subset(ml_g, x, y, folds, &treated)?;
    let t = clipping_threshold;
    let m = cross_fit_predict_proba(ml_m, x, d, folds)?.mapv(|p| p.clamp(t, 1.0 - t));
    Ok((g0, g1, m))
}

/// Inverse-probability-weighted residual correction
/// `D(Y − ĝ₁)/m̂ − (1 − D)(Y − ĝ₀)/(1 − m̂)`.
pub(crate) fn ipw_residual(
    y: ArrayView1<'_, f64>,
    d: &Array1<f64>,
    g0: &Array1<f64>,
    g1: &Array1<f64>,
    m: &Array1<f64>,
) -> Array1<f64> {
    let u0: Array1<f64> = &y - g0;
    let u1: Array1<f64> = &y - g1;
    d * &u1 / m - &(1.0 - d) * &u0 / &(1.0 - m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use ndarray::{array, Array2};

    #[test]
    fn rejects_continuous_treatment_and_bad_clipping() {
        let data = DmlData::new(
            array![1.0, 2.0, 3.0],
            array![0.0, 0.5, 1.0],
            array![[1.0], [2.0], [3.0]],
        )
        .unwrap();
        let irm = Irm::new(Ols::new(), LogisticRegression::default());
        assert!(irm.fit(&data).is_err());
        let binary =
            DmlData::new(array![1.0, 2.0], array![0.0, 1.0], array![[1.0], [2.0]]).unwrap();
        let bad = Irm::new(Ols::new(), LogisticRegression::default()).clipping_threshold(0.7);
        assert!(bad.validate(&binary).is_err());
    }

    #[test]
    fn normalized_weights_have_unit_means() {
        let m = array![0.2, 0.5, 0.7, 0.4];
        let d = array![1.0, 0.0, 1.0, 0.0];
        let adj = normalize_propensity(&m, &d);
        let n = 4.0;
        let mean1: f64 = d.iter().zip(&adj).map(|(&a, &b)| a / b).sum::<f64>() / n;
        let mean0: f64 = d
            .iter()
            .zip(&adj)
            .map(|(&a, &b)| (1.0 - a) / (1.0 - b))
            .sum::<f64>()
            / n;
        assert!((mean1 - 1.0).abs() < 1e-12 && (mean0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn randomized_experiment_recovers_difference_in_means() {
        // Constant propensity, no covariate signal: ATE ≈ treated − control mean.
        let n = 200;
        let x = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 37) % 17) as f64);
        let d: Array1<f64> = (0..n).map(|i| ((i * 7919) % 2) as f64).collect();
        let y: Array1<f64> = (0..n).map(|i| 2.0 * d[i] + ((i * 31) % 7) as f64).collect();
        let data = DmlData::new(y, d, x).unwrap();
        let fit = Irm::new(Ols::new(), LogisticRegression::default())
            .fit(&data)
            .unwrap();
        assert!((fit.coef()[0] - 2.0).abs() < 0.5, "ate = {}", fit.coef()[0]);
    }
}

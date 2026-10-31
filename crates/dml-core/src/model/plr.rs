//! Partially linear regression (PLR):
//!
//! ```text
//! Y = θ·D + g(X) + ε,   E[ε | D, X] = 0
//! D = m(X) + v,         E[v | X] = 0
//! ```

use ndarray::Array1;

use super::{
    dml_model_api, CateSignal, LinearScoreModel, NuisanceTarget, ScoreEstimate, ScoreInput,
};
use crate::crossfit::{cross_fit_predict, Fold, SampleSplitting};
use crate::error::{DmlError, Result};
use crate::learner::Regressor;
use crate::score::LinearScore;
use crate::sensitivity::{Nu2Parts, SensitivityParts};

/// Orthogonal score used by [`Plr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PlrScore {
    /// `ψ = (Y − ℓ(X) − θ(D − m(X)))·(D − m(X))` with `ℓ(X) = E[Y | X]`.
    #[default]
    PartiallingOut,
    /// `ψ = (Y − θD − g(X))·(D − m(X))`, with `g` fitted on `Y − θ̃D` for a
    /// preliminary partialling-out estimate `θ̃`.
    IvType,
}

/// PLR estimator: `ml_l` learns `E[Y | X]`, `ml_m` learns `E[D | X]`, and
/// the optional `ml_g` learns `g(X)` for the IV-type score.
pub struct Plr {
    ml_l: Box<dyn Regressor>,
    ml_m: Box<dyn Regressor>,
    ml_g: Option<Box<dyn Regressor>>,
    score: PlrScore,
    splitting: SampleSplitting,
}

impl Plr {
    pub fn new(ml_l: impl Regressor + 'static, ml_m: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_l), Box::new(ml_m))
    }

    /// Like [`Plr::new`] for learners that are already boxed trait objects.
    pub fn from_boxed(ml_l: Box<dyn Regressor>, ml_m: Box<dyn Regressor>) -> Self {
        Self {
            ml_l,
            ml_m,
            ml_g: None,
            score: PlrScore::default(),
            splitting: SampleSplitting::default(),
        }
    }

    pub fn score(mut self, score: PlrScore) -> Self {
        self.score = score;
        self
    }

    /// Learner for `g(X)` in the IV-type score (default: a fresh copy of `ml_l`).
    pub fn ml_g(mut self, ml_g: impl Regressor + 'static) -> Self {
        self.ml_g = Some(Box::new(ml_g));
        self
    }

    pub fn ml_g_boxed(mut self, ml_g: Box<dyn Regressor>) -> Self {
        self.ml_g = Some(ml_g);
        self
    }

    dml_model_api!();
}

impl LinearScoreModel for Plr {
    fn name(&self) -> String {
        match self.score {
            PlrScore::PartiallingOut => "PLR (partialling out)".into(),
            PlrScore::IvType => "PLR (IV-type)".into(),
        }
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, .. } = *input;
        let l_hat = input.predictions_or("ml_l", || {
            cross_fit_predict(self.ml_l.as_ref(), x, y, folds)
        })?;
        let m_hat = input.predictions_or("ml_m", || {
            cross_fit_predict(self.ml_m.as_ref(), x, d, folds)
        })?;
        let v: Array1<f64> = &d - &m_hat;
        if v.dot(&v) <= f64::EPSILON * d.dot(&d).max(1.0) {
            return Err(DmlError::InvalidArgument(
                "treatment residuals are zero: D is fully explained by X".into(),
            ));
        }
        let u: Array1<f64> = &y - &l_hat;

        match self.score {
            PlrScore::PartiallingOut => {
                // Outcome residual y − ℓ̂ − θ(d − m̂); ν² = 1 / mean((d − m̂)²).
                let parts = SensitivityParts {
                    outcome_residual: u.clone(),
                    theta_factor: Some(v.clone()),
                    riesz: Array1::zeros(v.len()),
                    nu2: Nu2Parts::InverseMeanSquare(v.clone()),
                };
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&v * &v), &v * &u)?,
                    vec![("ml_l", l_hat), ("ml_m", m_hat)],
                )
                .with_sensitivity(parts)
                .with_targets(vec![
                    NuisanceTarget::regression("ml_l", y.to_owned()),
                    NuisanceTarget::regression("ml_m", d.to_owned()),
                ])
                // CATE: regress y − ℓ̂ on the basis scaled by d − m̂ (as DoubleML).
                .with_cate_signal(CateSignal {
                    signal: u,
                    theta_factor: None,
                    basis_scale: Some(v),
                    is_effect: false,
                }))
            }
            PlrScore::IvType => {
                let theta_init = u.dot(&v) / v.dot(&v);
                let target: Array1<f64> = &y - &(theta_init * &d);
                let fresh_l;
                let ml_g = match &self.ml_g {
                    Some(g) => g.as_ref(),
                    None => {
                        fresh_l = self.ml_l.fresh();
                        fresh_l.as_ref()
                    }
                };
                let g_hat = input
                    .predictions_or("ml_g", || cross_fit_predict(ml_g, x, target.view(), folds))?;
                let resid_g: Array1<f64> = &y - &g_hat;
                // Outcome residual y − ĝ − θd.
                let parts = SensitivityParts {
                    outcome_residual: resid_g.clone(),
                    theta_factor: Some(d.to_owned()),
                    riesz: Array1::zeros(v.len()),
                    nu2: Nu2Parts::InverseMeanSquare(v.clone()),
                };
                // CATE: regress y − ĝ − θ·m̂ on the basis scaled by d − m̂ (as DoubleML).
                let cate = CateSignal {
                    signal: resid_g.clone(),
                    theta_factor: Some(m_hat.clone()),
                    basis_scale: Some(v.clone()),
                    is_effect: false,
                };
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&v * &d), &v * &resid_g)?,
                    vec![("ml_l", l_hat), ("ml_m", m_hat), ("ml_g", g_hat)],
                )
                .with_sensitivity(parts)
                .with_targets(vec![
                    NuisanceTarget::regression("ml_l", y.to_owned()),
                    NuisanceTarget::regression("ml_m", d.to_owned()),
                    NuisanceTarget::regression("ml_g", target),
                ])
                .with_cate_signal(cate))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DmlData;
    use crate::learner::Ols;
    use ndarray::{Array1, Array2};

    fn toy_data() -> DmlData {
        let n = 60;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
        let d: Array1<f64> = (0..n)
            .map(|i| 0.3 * x[[i, 0]] + ((i * 13) % 5) as f64 - 2.0)
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| 0.5 * d[i] + x[[i, 1]] + ((i * 17) % 3) as f64 - 1.0)
            .collect();
        DmlData::new(y, d, x).unwrap()
    }

    #[test]
    fn coef_matches_residual_formula() {
        let data = toy_data();
        let fit = Plr::new(Ols::new(), Ols::new()).seed(3).fit(&data).unwrap();
        let u = &data.y() - &fit.prediction("ml_l", 0, 0).unwrap();
        let v = &data.treatment(0) - &fit.prediction("ml_m", 0, 0).unwrap();
        assert!((fit.coef()[0] - u.dot(&v) / v.dot(&v)).abs() < 1e-12);
        // The score is zero on average at the DML2 solution.
        assert!(fit.psi().sum().abs() < 1e-10);
    }

    #[test]
    fn iv_type_with_linear_learners_is_close_to_partialling_out() {
        let data = toy_data();
        let po = Plr::new(Ols::new(), Ols::new()).fit(&data).unwrap();
        let iv = Plr::new(Ols::new(), Ols::new())
            .score(PlrScore::IvType)
            .fit(&data)
            .unwrap();
        assert!(iv.prediction("ml_g", 0, 0).is_some());
        assert!((po.coef()[0] - iv.coef()[0]).abs() < 0.05);
    }

    #[test]
    fn rejects_treatment_explained_by_x() {
        let x = Array2::from_shape_fn((20, 1), |(i, _)| i as f64);
        let d = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let y = d.mapv(|v| v * 0.5);
        let data = DmlData::new(y, d, x).unwrap();
        assert!(Plr::new(Ols::new(), Ols::new()).fit(&data).is_err());
    }
}

//! Automatic debiased machine learning (Chernozhukov, Newey and Singh, 2022):
//! `θ = E[m(W; γ)]` for any linear functional `m` of the regression
//! `γ(D, X) = E[Y | D, X]`, with the Riesz representer learned from the
//! functional (see [`riesz`](crate::riesz)).

use ndarray::{concatenate, Array1, Array2, Axis};

use super::{
    dml_model_api, CateSignal, LinearScoreModel, NuisanceTarget, ScoreEstimate, ScoreInput,
};
use crate::crossfit::{map_folds, Fold, SampleSplitting};
use crate::data::gather;
use crate::error::{DmlError, Result};
use crate::learner::Regressor;
use crate::riesz::{apply_terms, LinearFunctional, RieszLasso, RieszLearner, Term};
use crate::score::LinearScore;
use crate::sensitivity::{Nu2Parts, SensitivityParts};

/// Auto-DML estimator of `θ = E[m(W; γ)]`.
///
/// `ml_g` learns `γ(D, X) = E[Y | D, X]` on `W = (D, X)` (the treatment in the
/// first column), the Riesz learner (default [`RieszLasso`]) learns `α` from
/// the functional, and the score is
///
/// ```text
/// ψ = m(W; γ̂) + α̂(W)·(Y − γ̂(W)) − θ,
/// ```
///
/// cross-fitted like every other model. Each treatment in the data gets the
/// same functional applied to it (other treatments act as covariates).
/// Sensitivity analysis uses `σ² = E[(Y − γ)²]` and `ν² = E[2m(W; α) − α²]`.
///
/// ```
/// use dml_core::{AutoDml, AverageDerivative, DmlData, Ols, ShiftEffect};
/// use ndarray::{Array1, Array2};
///
/// let n = 400;
/// let x = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 7) % 11) as f64 / 5.0);
/// let d: Array1<f64> = (0..n).map(|i| x[[i, 0]] + ((i * 13) % 7) as f64 / 3.0).collect();
/// let y: Array1<f64> = (0..n).map(|i| 0.5 * d[i] + x[[i, 0]].sin() + ((i * 17) % 5) as f64 / 4.0).collect();
/// let data = DmlData::new(y, d, x)?;
///
/// // Average derivative E[∂γ/∂D]; here γ is linear in D with slope 0.5.
/// let fit = AutoDml::new(Ols::new(), AverageDerivative::new()).fit(&data)?;
/// assert!((fit.coef()[0] - 0.5).abs() < 0.1);
///
/// // Effect of raising every treatment by one unit.
/// let shift = AutoDml::new(Ols::new(), ShiftEffect::new(1.0)).fit(&data)?;
/// assert!((shift.coef()[0] - fit.coef()[0]).abs() < 1e-6);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
pub struct AutoDml {
    ml_g: Box<dyn Regressor>,
    riesz: Box<dyn RieszLearner>,
    functional: Box<dyn LinearFunctional>,
    splitting: SampleSplitting,
}

impl AutoDml {
    pub fn new(
        ml_g: impl Regressor + 'static,
        functional: impl LinearFunctional + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(functional))
    }

    /// Like [`AutoDml::new`] for boxed trait objects.
    pub fn from_boxed(ml_g: Box<dyn Regressor>, functional: Box<dyn LinearFunctional>) -> Self {
        Self {
            ml_g,
            riesz: Box::new(RieszLasso::default()),
            functional,
            splitting: SampleSplitting::default(),
        }
    }

    /// Learner for the Riesz representer (default [`RieszLasso::default`]).
    pub fn riesz(mut self, riesz: impl RieszLearner + 'static) -> Self {
        self.riesz = Box::new(riesz);
        self
    }

    pub fn riesz_boxed(mut self, riesz: Box<dyn RieszLearner>) -> Self {
        self.riesz = riesz;
        self
    }

    dml_model_api!();
}

/// Out-of-fold quantities of one fold, on its test rows.
struct FoldResult {
    g: Array1<f64>,
    m_g: Array1<f64>,
    alpha: Array1<f64>,
    m_alpha: Array1<f64>,
}

fn rows_of(terms: &[Term], idx: &[usize]) -> Vec<Term> {
    terms.iter().map(|t| t.rows(idx)).collect()
}

impl LinearScoreModel for AutoDml {
    fn name(&self) -> String {
        format!("Auto-DML ({})", self.functional.name())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, .. } = *input;
        let n = y.len();
        let w: Array2<f64> = concatenate![Axis(1), d.insert_axis(Axis(1)), x];
        let terms = self.functional.terms(w.view())?;
        if terms.is_empty() {
            return Err(DmlError::InvalidArgument(
                "the functional has no terms".into(),
            ));
        }
        for t in &terms {
            if t.w.dim() != w.dim() || t.weight.len() != n {
                return Err(DmlError::DimensionMismatch("functional terms".into()));
            }
        }

        let per_fold = map_folds(folds.len(), |k| -> Result<FoldResult> {
            let fold = &folds[k];
            let w_train = w.select(Axis(0), &fold.train);
            let w_test = w.select(Axis(0), &fold.test);
            let test_terms = rows_of(&terms, &fold.test);

            let mut g = self.ml_g.fresh();
            g.fit(w_train.view(), gather(y, &fold.train).view())?;
            let g_test = g.predict(w_test.view())?;
            let m_g = apply_terms(&test_terms, |wk| g.predict(wk))?;

            let mut a = self.riesz.fresh();
            a.fit(w_train.view(), &rows_of(&terms, &fold.train))?;
            let alpha = a.predict(w_test.view())?;
            let m_alpha = a.predict_functional(&test_terms)?;
            Ok(FoldResult {
                g: g_test,
                m_g,
                alpha,
                m_alpha,
            })
        })?;

        let mut g_hat = Array1::from_elem(n, f64::NAN);
        let mut m_hat = Array1::from_elem(n, f64::NAN);
        let mut alpha = Array1::from_elem(n, f64::NAN);
        let mut m_alpha = Array1::from_elem(n, f64::NAN);
        for (fold, r) in folds.iter().zip(per_fold) {
            if r.g.len() != fold.test.len() || r.alpha.len() != fold.test.len() {
                return Err(DmlError::DimensionMismatch(
                    "learner returned wrong number of predictions".into(),
                ));
            }
            for (j, &i) in fold.test.iter().enumerate() {
                g_hat[i] = r.g[j];
                m_hat[i] = r.m_g[j];
                alpha[i] = r.alpha[j];
                m_alpha[i] = r.m_alpha[j];
            }
        }
        if [&g_hat, &m_hat, &alpha, &m_alpha]
            .iter()
            .any(|v| v.iter().any(|x| !x.is_finite()))
        {
            return Err(DmlError::InvalidArgument(
                "folds do not cover every observation, or a learner returned a non-finite value"
                    .into(),
            ));
        }

        let resid: Array1<f64> = &y - &g_hat;
        let psi_b: Array1<f64> = &m_hat + &(&alpha * &resid);
        let parts = SensitivityParts {
            outcome_residual: resid,
            theta_factor: None,
            riesz: alpha.clone(),
            nu2: Nu2Parts::Elementwise(2.0 * &m_alpha - &alpha.mapv(|a| a * a)),
        };
        Ok(ScoreEstimate::new(
            LinearScore::new(Array1::from_elem(n, -1.0), psi_b.clone())?,
            vec![("ml_g", g_hat), ("riesz", alpha), ("functional", m_hat)],
        )
        .with_sensitivity(parts)
        .with_targets(vec![NuisanceTarget::regression("ml_g", y.to_owned())])
        // ψ_b is a per-unit signal of m(W; γ) (for the ATE the AIPW signal), so
        // BLP / GATE inference project it on a basis, as for IRM.
        .with_cate_signal(CateSignal {
            signal: psi_b,
            theta_factor: None,
            basis_scale: None,
            is_effect: true,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DmlData;
    use crate::learner::Ols;
    use crate::riesz::{Ate, AverageDerivative, PolicyEffect, RieszPenalty, ShiftEffect};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StandardNormal;

    /// `Y = θD + sin(X₁) + X₂ + ε`, `D = 0.5 X₁ + ν`: γ is linear in `D`, so the
    /// average derivative and the per-unit shift effect are `θ`.
    fn continuous(n: usize, seed: u64, theta: f64) -> DmlData {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut draw = || -> f64 { rng.sample(StandardNormal) };
        let x = Array2::from_shape_fn((n, 2), |_| draw());
        let d: Array1<f64> = (0..n).map(|i| 0.5 * x[[i, 0]] + draw()).collect();
        let y: Array1<f64> = (0..n)
            .map(|i| theta * d[i] + x[[i, 0]].sin() + x[[i, 1]] + draw())
            .collect();
        DmlData::new(y, d, x).unwrap()
    }

    #[test]
    fn average_derivative_and_shift_recover_theta() {
        let data = continuous(2000, 1, 0.7);
        let fit = AutoDml::new(Ols::new(), AverageDerivative::new())
            .fit(&data)
            .unwrap();
        assert!(
            (fit.coef()[0] - 0.7).abs() < 4.0 * fit.se()[0],
            "{}",
            fit.coef()[0]
        );
        let shift = AutoDml::new(Ols::new(), ShiftEffect::new(0.5).per_unit(true))
            .fit(&data)
            .unwrap();
        assert!((shift.coef()[0] - fit.coef()[0]).abs() < 1e-6);
        // A policy that shifts everyone by 0.5 is the (unscaled) shift effect.
        let policy = PolicyEffect::new(data.treatment(0).mapv(|d| d + 0.5));
        let p = AutoDml::new(Ols::new(), policy).fit(&data).unwrap();
        let s = AutoDml::new(Ols::new(), ShiftEffect::new(0.5))
            .fit(&data)
            .unwrap();
        assert!((p.coef()[0] - s.coef()[0]).abs() < 1e-10);
        assert!(fit.prediction("riesz", 0, 0).is_some());
    }

    #[test]
    fn riesz_correction_removes_the_bias_of_a_misspecified_regression() {
        // γ by OLS without the D·X₁ interaction is wrong; the learned Riesz
        // representer corrects it. Y = D(1 + X₁) + X₁ + ε has average derivative 1.
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let n = 4000;
        let mut draw = || -> f64 { rng.sample(StandardNormal) };
        let x = Array2::from_shape_fn((n, 1), |_| draw());
        let d: Array1<f64> = (0..n).map(|i| x[[i, 0]] + draw()).collect();
        let y: Array1<f64> = (0..n)
            .map(|i| d[i] * (1.0 + x[[i, 0]]) + x[[i, 0]] + draw())
            .collect();
        let data = DmlData::new(y, d, x).unwrap();
        let fit = AutoDml::new(Ols::new(), AverageDerivative::new())
            .fit(&data)
            .unwrap();
        assert!(
            (fit.coef()[0] - 1.0).abs() < 4.0 * fit.se()[0],
            "{} ± {}",
            fit.coef()[0],
            fit.se()[0]
        );
        let plug_in = fit.prediction("functional", 0, 0).unwrap().mean().unwrap();
        assert!((plug_in - 1.0).abs() > (fit.coef()[0] - 1.0).abs());
    }

    #[test]
    fn ate_is_close_to_irm() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 3000;
        let mut draw = || -> f64 { rng.sample(StandardNormal) };
        let x = Array2::from_shape_fn((n, 2), |_| draw());
        let mut u = ChaCha8Rng::seed_from_u64(4);
        let d: Array1<f64> = (0..n)
            .map(|i| {
                let p = 1.0 / (1.0 + (-0.8 * x[[i, 0]]).exp());
                f64::from(u.gen::<f64>() < p)
            })
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| 0.5 * d[i] + x[[i, 0]] + 0.5 * x[[i, 1]] + draw())
            .collect();
        let data = DmlData::new(y, d, x).unwrap();
        let auto = AutoDml::new(Ols::new(), Ate).fit(&data).unwrap();
        let irm = crate::Irm::new(Ols::new(), crate::LogisticRegression::default())
            .fit(&data)
            .unwrap();
        assert!((auto.coef()[0] - irm.coef()[0]).abs() < 0.5 * irm.se()[0]);
        assert!((auto.coef()[0] - 0.5).abs() < 4.0 * auto.se()[0]);
        let s = auto
            .sensitivity_analysis(&crate::SensitivityParams::default())
            .unwrap();
        assert!(s.theta_lower[0] < auto.coef()[0]);
    }

    #[test]
    fn fixed_penalty_and_errors() {
        let data = continuous(300, 2, 0.5);
        let fit = AutoDml::new(Ols::new(), AverageDerivative::new())
            .riesz(RieszLasso::new().penalty(RieszPenalty::Fixed(0.0)))
            .fit(&data)
            .unwrap();
        assert!(fit.coef()[0].is_finite());
        assert!(AutoDml::new(Ols::new(), Ate).fit(&data).is_err());
    }

    #[test]
    fn coverage_of_the_average_derivative() {
        let reps = 120;
        let mut hits = 0;
        for r in 0..reps {
            let data = continuous(400, 100 + r, 0.7);
            let fit = AutoDml::new(Ols::new(), AverageDerivative::new())
                .seed(r)
                .fit(&data)
                .unwrap();
            let ci = fit.confint(0.95).unwrap();
            if ci[[0, 0]] <= 0.7 && 0.7 <= ci[[0, 1]] {
                hits += 1;
            }
        }
        let cover = hits as f64 / reps as f64;
        assert!(cover > 0.88, "coverage {cover}");
    }
}

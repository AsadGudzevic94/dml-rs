//! Meta-learners for the conditional average treatment effect
//! `τ(x) = E[Y(1) − Y(0) | X = x]` of a binary treatment.
//!
//! | learner | idea | reference |
//! |---|---|---|
//! | [`SLearner`] | one model `μ(x, d)`; `τ̂ = μ(x,1) − μ(x,0)` | — |
//! | [`TLearner`] | separate `μ₀`, `μ₁` per arm; `τ̂ = μ₁ − μ₀` | — |
//! | [`XLearner`] | impute individual effects, model them per arm, blend by propensity | Künzel et al. (2019) |
//! | [`RLearner`] | residual-on-residual weighted regression (Robinson decomposition) | Nie & Wager (2021) |
//! | [`DrLearner`] | regress the cross-fitted AIPW pseudo-outcome on `x` | Kennedy (2023) |
//!
//! The R- and DR-learners cross-fit their nuisances. The DR-learner's
//! pseudo-outcome is a valid orthogonal signal, so it also supports
//! inference on the CATE through [`CateFit::blp`] / [`CateFit::gate`].

use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::blp::{as_column, Blp};
use crate::crossfit::{cross_fit_predict, cross_fit_predict_proba, Fold, KFold};
use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};
use crate::model::irm::{ipw_residual, irm_nuisances};

/// How a fitted CATE model turns covariates into effects.
enum Predictor {
    /// μ([x, 1]) − μ([x, 0])
    Single(Box<dyn Regressor>),
    /// μ₁(x) − μ₀(x)
    TwoModels {
        mu0: Box<dyn Regressor>,
        mu1: Box<dyn Regressor>,
    },
    /// e(x)·τ₀(x) + (1 − e(x))·τ₁(x)
    Blend {
        tau0: Box<dyn Regressor>,
        tau1: Box<dyn Regressor>,
        propensity: Box<dyn Classifier>,
    },
    /// τ̂(x) = f(x) from a final-stage regression
    Direct(Box<dyn Regressor>),
}

/// A fitted CATE model.
pub struct CateFit {
    method: &'static str,
    n_features: usize,
    predictor: Predictor,
    pseudo_outcomes: Option<Array1<f64>>,
}

impl std::fmt::Debug for CateFit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CateFit")
            .field("method", &self.method)
            .field("n_features", &self.n_features)
            .finish_non_exhaustive()
    }
}

impl CateFit {
    /// `"S-learner"`, `"T-learner"`, …
    pub fn method(&self) -> &str {
        self.method
    }

    /// Estimated effects `τ̂(x)` for each row of `x`.
    pub fn effect(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "x has {} columns, the model was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        match &self.predictor {
            Predictor::Single(model) => {
                let n = x.nrows();
                let x1 = concatenate![Axis(1), x, Array2::ones((n, 1))];
                let x0 = concatenate![Axis(1), x, Array2::zeros((n, 1))];
                Ok(model.predict(x1.view())? - model.predict(x0.view())?)
            }
            Predictor::TwoModels { mu0, mu1 } => Ok(mu1.predict(x)? - mu0.predict(x)?),
            Predictor::Blend {
                tau0,
                tau1,
                propensity,
            } => {
                let e = propensity.predict_proba(x)?;
                let (t0, t1) = (tau0.predict(x)?, tau1.predict(x)?);
                Ok(&e * &t0 + &(1.0 - &e) * &t1)
            }
            Predictor::Direct(model) => model.predict(x),
        }
    }

    /// Average of `τ̂(x)` over the rows of `x`.
    pub fn average_effect(&self, x: ArrayView2<'_, f64>) -> Result<f64> {
        Ok(self.effect(x)?.mean().unwrap_or(f64::NAN))
    }

    /// DR-learner only: the cross-fitted AIPW pseudo-outcomes `φᵢ`.
    pub fn pseudo_outcomes(&self) -> Option<ArrayView1<'_, f64>> {
        self.pseudo_outcomes.as_ref().map(|p| p.view())
    }

    /// DR-learner only: best linear predictor of the CATE on `basis` with
    /// HC0 inference. A basis of ones gives the ATE with its DML standard error.
    pub fn blp(&self, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        Blp::fit(as_column(self.signal()?).view(), basis)
    }

    /// DR-learner only: group average treatment effects for 0/1 group dummies.
    pub fn gate(&self, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        Blp::fit_gate(as_column(self.signal()?).view(), groups)
    }

    fn signal(&self) -> Result<ArrayView1<'_, f64>> {
        self.pseudo_outcomes().ok_or_else(|| {
            DmlError::InvalidArgument("BLP / GATE inference is available for the DR-learner".into())
        })
    }
}

/// Validate a single binary treatment and split the data by arm.
struct Arms<'a> {
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    d: ArrayView1<'a, f64>,
    treated: Vec<usize>,
    control: Vec<usize>,
}

impl<'a> Arms<'a> {
    fn new(data: &'a DmlData) -> Result<Self> {
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return Err(DmlError::InvalidArgument(
                "CATE learners need exactly one binary (0/1) treatment".into(),
            ));
        }
        let d = data.treatment(0);
        let treated: Vec<usize> = (0..d.len()).filter(|&i| d[i] == 1.0).collect();
        let control: Vec<usize> = (0..d.len()).filter(|&i| d[i] == 0.0).collect();
        if treated.is_empty() || control.is_empty() {
            return Err(DmlError::InvalidArgument(
                "need both treated and control observations".into(),
            ));
        }
        Ok(Self {
            x: data.x(),
            y: data.y(),
            d,
            treated,
            control,
        })
    }

    /// Fresh copy of `template` fitted on one arm.
    fn fit_arm(
        &self,
        template: &dyn Regressor,
        rows: &[usize],
        target: ArrayView1<'_, f64>,
    ) -> Result<Box<dyn Regressor>> {
        let mut model = template.fresh();
        model.fit(
            self.x.select(Axis(0), rows).view(),
            gather(target, rows).view(),
        )?;
        Ok(model)
    }
}

/// Folds for the cross-fitted learners: explicit, or seeded K-fold.
fn resolve_folds(n: usize, n_folds: usize, seed: u64, folds: Option<&[Fold]>) -> Result<Vec<Fold>> {
    match folds {
        Some(f) => Ok(f.to_vec()),
        None => KFold::new(n_folds, seed)?.split(n),
    }
}

/// S-learner: one outcome model on `[x, d]`.
pub struct SLearner {
    model: Box<dyn Regressor>,
}

impl SLearner {
    pub fn new(model: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(model))
    }

    pub fn from_boxed(model: Box<dyn Regressor>) -> Self {
        Self { model }
    }

    pub fn fit(&self, data: &DmlData) -> Result<CateFit> {
        let arms = Arms::new(data)?;
        let xd = concatenate![Axis(1), arms.x, arms.d.insert_axis(Axis(1))];
        let mut model = self.model.fresh();
        model.fit(xd.view(), arms.y)?;
        Ok(CateFit {
            method: "S-learner",
            n_features: data.n_features(),
            predictor: Predictor::Single(model),
            pseudo_outcomes: None,
        })
    }
}

/// T-learner: separate outcome models for treated and controls.
pub struct TLearner {
    model: Box<dyn Regressor>,
}

impl TLearner {
    pub fn new(model: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(model))
    }

    pub fn from_boxed(model: Box<dyn Regressor>) -> Self {
        Self { model }
    }

    pub fn fit(&self, data: &DmlData) -> Result<CateFit> {
        let arms = Arms::new(data)?;
        let mu0 = arms.fit_arm(self.model.as_ref(), &arms.control, arms.y)?;
        let mu1 = arms.fit_arm(self.model.as_ref(), &arms.treated, arms.y)?;
        Ok(CateFit {
            method: "T-learner",
            n_features: data.n_features(),
            predictor: Predictor::TwoModels { mu0, mu1 },
            pseudo_outcomes: None,
        })
    }
}

/// X-learner (Künzel et al., 2019).
///
/// 1. Fit `μ₀`, `μ₁` per arm (`outcome` learner).
/// 2. Impute effects: `Y − μ₀(X)` for treated, `μ₁(X) − Y` for controls.
/// 3. Model the imputed effects per arm (`effect` learner): `τ₁`, `τ₀`.
/// 4. Blend: `τ̂(x) = e(x)·τ₀(x) + (1 − e(x))·τ₁(x)` with propensity `e`.
pub struct XLearner {
    outcome: Box<dyn Regressor>,
    effect: Box<dyn Regressor>,
    propensity: Box<dyn Classifier>,
}

impl XLearner {
    pub fn new(
        outcome: impl Regressor + 'static,
        effect: impl Regressor + 'static,
        propensity: impl Classifier + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(outcome), Box::new(effect), Box::new(propensity))
    }

    pub fn from_boxed(
        outcome: Box<dyn Regressor>,
        effect: Box<dyn Regressor>,
        propensity: Box<dyn Classifier>,
    ) -> Self {
        Self {
            outcome,
            effect,
            propensity,
        }
    }

    pub fn fit(&self, data: &DmlData) -> Result<CateFit> {
        let arms = Arms::new(data)?;
        let mu0 = arms.fit_arm(self.outcome.as_ref(), &arms.control, arms.y)?;
        let mu1 = arms.fit_arm(self.outcome.as_ref(), &arms.treated, arms.y)?;
        // Imputed individual effects, laid out over all rows; each arm's
        // effect model only reads its own rows.
        let mut imputed = Array1::<f64>::zeros(arms.y.len());
        let x_t = arms.x.select(Axis(0), &arms.treated);
        let x_c = arms.x.select(Axis(0), &arms.control);
        for (&i, m0) in arms.treated.iter().zip(mu0.predict(x_t.view())?) {
            imputed[i] = arms.y[i] - m0;
        }
        for (&i, m1) in arms.control.iter().zip(mu1.predict(x_c.view())?) {
            imputed[i] = m1 - arms.y[i];
        }
        let tau1 = arms.fit_arm(self.effect.as_ref(), &arms.treated, imputed.view())?;
        let tau0 = arms.fit_arm(self.effect.as_ref(), &arms.control, imputed.view())?;
        let mut propensity = self.propensity.fresh();
        propensity.fit(arms.x, arms.d)?;
        Ok(CateFit {
            method: "X-learner",
            n_features: data.n_features(),
            predictor: Predictor::Blend {
                tau0,
                tau1,
                propensity,
            },
            pseudo_outcomes: None,
        })
    }
}

/// R-learner (Nie & Wager, 2021): with cross-fitted `m̂(x) = E[Y | X]` and
/// propensity `ê(x)`, minimise `Σ (Ỹ − τ(X)·D̃)²`, i.e. a weighted regression
/// of `Ỹ/D̃` on `X` with weights `D̃²`. The `effect` learner must support
/// [`Regressor::fit_weighted`].
pub struct RLearner {
    outcome: Box<dyn Regressor>,
    propensity: Box<dyn Classifier>,
    effect: Box<dyn Regressor>,
    n_folds: usize,
    seed: u64,
    clipping_threshold: f64,
}

impl RLearner {
    pub fn new(
        outcome: impl Regressor + 'static,
        propensity: impl Classifier + 'static,
        effect: impl Regressor + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(outcome), Box::new(propensity), Box::new(effect))
    }

    pub fn from_boxed(
        outcome: Box<dyn Regressor>,
        propensity: Box<dyn Classifier>,
        effect: Box<dyn Regressor>,
    ) -> Self {
        Self {
            outcome,
            propensity,
            effect,
            n_folds: 5,
            seed: 0,
            clipping_threshold: 1e-2,
        }
    }

    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.n_folds = n_folds;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Propensities are clipped to `[t, 1 − t]` (default 0.01).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }

    pub fn fit(&self, data: &DmlData) -> Result<CateFit> {
        self.fit_impl(data, None)
    }

    pub fn fit_with_folds(&self, data: &DmlData, folds: &[Fold]) -> Result<CateFit> {
        self.fit_impl(data, Some(folds))
    }

    fn fit_impl(&self, data: &DmlData, folds: Option<&[Fold]>) -> Result<CateFit> {
        let arms = Arms::new(data)?;
        check_clipping(self.clipping_threshold)?;
        let folds = resolve_folds(data.n_obs(), self.n_folds, self.seed, folds)?;
        let m_hat = cross_fit_predict(self.outcome.as_ref(), arms.x, arms.y, &folds)?;
        let t = self.clipping_threshold;
        let e_hat = cross_fit_predict_proba(self.propensity.as_ref(), arms.x, arms.d, &folds)?
            .mapv(|p| p.clamp(t, 1.0 - t));
        let y_res: Array1<f64> = &arms.y - &m_hat;
        let d_res: Array1<f64> = &arms.d - &e_hat;
        let target = &y_res / &d_res;
        let weights = &d_res * &d_res;
        let mut effect = self.effect.fresh();
        effect.fit_weighted(arms.x, target.view(), weights.view())?;
        Ok(CateFit {
            method: "R-learner",
            n_features: data.n_features(),
            predictor: Predictor::Direct(effect),
            pseudo_outcomes: None,
        })
    }
}

/// DR-learner (Kennedy, 2023): cross-fit `ĝ₀`, `ĝ₁` and the propensity `m̂`
/// exactly as [`Irm`](crate::Irm) does, form the AIPW pseudo-outcome
/// `φ = ĝ₁ − ĝ₀ + D(Y − ĝ₁)/m̂ − (1 − D)(Y − ĝ₀)/(1 − m̂)`, and regress it
/// on `X` with the `effect` learner.
pub struct DrLearner {
    outcome: Box<dyn Regressor>,
    propensity: Box<dyn Classifier>,
    effect: Box<dyn Regressor>,
    n_folds: usize,
    seed: u64,
    clipping_threshold: f64,
}

impl DrLearner {
    pub fn new(
        outcome: impl Regressor + 'static,
        propensity: impl Classifier + 'static,
        effect: impl Regressor + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(outcome), Box::new(propensity), Box::new(effect))
    }

    pub fn from_boxed(
        outcome: Box<dyn Regressor>,
        propensity: Box<dyn Classifier>,
        effect: Box<dyn Regressor>,
    ) -> Self {
        Self {
            outcome,
            propensity,
            effect,
            n_folds: 5,
            seed: 0,
            clipping_threshold: 1e-2,
        }
    }

    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.n_folds = n_folds;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Propensities are clipped to `[t, 1 − t]` (default 0.01).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }

    pub fn fit(&self, data: &DmlData) -> Result<CateFit> {
        self.fit_impl(data, None)
    }

    pub fn fit_with_folds(&self, data: &DmlData, folds: &[Fold]) -> Result<CateFit> {
        self.fit_impl(data, Some(folds))
    }

    fn fit_impl(&self, data: &DmlData, folds: Option<&[Fold]>) -> Result<CateFit> {
        let arms = Arms::new(data)?;
        check_clipping(self.clipping_threshold)?;
        let folds = resolve_folds(data.n_obs(), self.n_folds, self.seed, folds)?;
        let (g0, g1, m) = irm_nuisances(
            self.outcome.as_ref(),
            self.propensity.as_ref(),
            arms.x,
            arms.y,
            arms.d,
            &folds,
            self.clipping_threshold,
            crate::propensity::Calibration::None,
        )?;
        let d = arms.d.to_owned();
        let phi = &g1 - &g0 + &ipw_residual(arms.y, &d, &g0, &g1, &m);
        let mut effect = self.effect.fresh();
        effect.fit(arms.x, phi.view())?;
        Ok(CateFit {
            method: "DR-learner",
            n_features: data.n_features(),
            predictor: Predictor::Direct(effect),
            pseudo_outcomes: Some(phi),
        })
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

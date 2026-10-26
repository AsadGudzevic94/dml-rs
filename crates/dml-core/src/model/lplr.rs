//! Logistic partially linear regression (LPLR) for a binary outcome, as
//! DoubleML's `DoubleMLLPLR` (Liu, Zhang and Zhou, 2021):
//!
//! ```text
//! P(Y = 1 | D, X) = expit(θ·D + r(X))
//! ```
//!
//! θ is a log odds ratio. The nuisances are `M(D, X) = P(Y = 1 | D, X)`
//! (`ml_M`), `t(X) = E[logit M(D, X) | X]` (`ml_t`), `m(X) = E[D | X]`
//! (`ml_m`) and `a(X) = E[D | X]` (`ml_a`, by default a copy of `ml_m`).
//! `ml_M` and `ml_a` are double cross-fitted: within every outer training
//! fold, inner folds give out-of-sample predictions on the training fold
//! (needed to build the targets of `ml_t` and a preliminary θ), and the
//! average of the inner models predicts the outer test fold. The score is
//! solved with Newton's method from the preliminary estimate.
//!
//! - [`LplrScore::NuisanceSpace`]: `m` is learned on the observations with
//!   `Y = 0`.
//! - [`LplrScore::Instrument`]: `m` is learned on all observations, weighted
//!   by `M(1 − M)`.
//!
//! External predictions ([`DmlData::with_external_predictions`]) may replace
//! any learner. For the double cross-fitted `ml_M` and `ml_a`, the inner
//! predictions of every outer fold `k` are needed too (`ml_M_inner_k`,
//! `ml_a_inner_k`: the out-of-inner-fold predictions on fold `k`'s training
//! set), as in DoubleML; a fit reports them under the same names.

use ndarray::{concatenate, Array1, ArrayView1, ArrayView2, Axis};

use super::nonlinear::{assemble, RepFit};
use super::{external_get, external_or, ExternalRef, Name, NuisanceTarget};
use crate::compat::{newton, np_mean, np_sum};
use crate::crossfit::{Fold, KFold, SampleSplitting};
use crate::data::{gather, is_binary, DmlData};
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};

/// Score of [`Lplr`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LplrScore {
    /// `m(X)` estimated on the observations with `Y = 0`.
    #[default]
    NuisanceSpace,
    /// Instrument-type score; `m(X)` estimated with weights `M(1 − M)`.
    Instrument,
}

/// A learner for `E[D | X]`: a regressor, or for a binary treatment a
/// classifier whose class-1 probability is used.
pub enum TreatmentLearner {
    Regressor(Box<dyn Regressor>),
    Classifier(Box<dyn Classifier>),
}

impl TreatmentLearner {
    fn is_classifier(&self) -> bool {
        matches!(self, Self::Classifier(_))
    }

    fn as_ref(&self) -> Learner<'_> {
        match self {
            Self::Regressor(r) => Learner::Regressor(r.as_ref()),
            Self::Classifier(c) => Learner::Classifier(c.as_ref()),
        }
    }
}

/// A borrowed learner of either kind.
#[derive(Clone, Copy)]
enum Learner<'a> {
    Regressor(&'a dyn Regressor),
    Classifier(&'a dyn Classifier),
}

impl Learner<'_> {
    /// Fit a fresh copy on `rows` of `x` (optionally weighted) and predict
    /// every matrix in `x_pred`.
    fn fit_predict(
        self,
        x: ArrayView2<'_, f64>,
        target: ArrayView1<'_, f64>,
        rows: &[usize],
        weights: Option<ArrayView1<'_, f64>>,
        x_pred: &[ArrayView2<'_, f64>],
    ) -> Result<Vec<Array1<f64>>> {
        let (xt, yt) = (x.select(Axis(0), rows), gather(target, rows));
        let wt = weights.map(|w| gather(w, rows));
        match self {
            Learner::Regressor(r) => {
                let mut m = r.fresh();
                match &wt {
                    Some(w) => m.fit_weighted(xt.view(), yt.view(), w.view())?,
                    None => m.fit(xt.view(), yt.view())?,
                }
                x_pred.iter().map(|xp| m.predict(*xp)).collect()
            }
            Learner::Classifier(c) => {
                let mut m = c.fresh();
                match &wt {
                    Some(w) => m.fit_weighted(xt.view(), yt.view(), w.view())?,
                    None => m.fit(xt.view(), yt.view())?,
                }
                x_pred.iter().map(|xp| m.predict_proba(*xp)).collect()
            }
        }
    }
}

/// Logistic partially linear regression for a binary outcome,
/// `P(Y = 1 | D, X) = expit(θ·D + r(X))`, as DoubleML's `DoubleMLLPLR`; θ is
/// a log odds ratio.
///
/// Nuisances: `ml_outcome` (DoubleML's `ml_M`) for `P(Y = 1 | D, X)`, `ml_t`
/// for `E[logit M(D, X) | X]`, `ml_m` (and optionally `ml_a`) for `E[D | X]`.
/// `ml_M` and `ml_a` are double cross-fitted ([`Lplr::n_folds_inner`]) and
/// the score is solved with Newton's method; see [`LplrScore`] for the scores.
pub struct Lplr {
    ml_outcome: Box<dyn Classifier>,
    ml_t: Box<dyn Regressor>,
    ml_m: TreatmentLearner,
    ml_a: Option<TreatmentLearner>,
    score: LplrScore,
    n_folds_inner: usize,
    splitting: SampleSplitting,
}

impl Lplr {
    /// `ml_outcome` learns `M(D, X) = P(Y = 1 | D, X)` (DoubleML's `ml_M`),
    /// `ml_t` the log odds `E[logit M(D, X) | X]`, `ml_m` the treatment
    /// regression `E[D | X]`.
    pub fn new(
        ml_outcome: impl Classifier + 'static,
        ml_t: impl Regressor + 'static,
        ml_m: impl Regressor + 'static,
    ) -> Self {
        Self::from_boxed(
            Box::new(ml_outcome),
            Box::new(ml_t),
            TreatmentLearner::Regressor(Box::new(ml_m)),
            None,
        )
    }

    /// As [`Lplr::new`], with a classifier for a binary treatment's
    /// propensity `P(D = 1 | X)`.
    pub fn with_propensity(
        ml_outcome: impl Classifier + 'static,
        ml_t: impl Regressor + 'static,
        ml_m: impl Classifier + 'static,
    ) -> Self {
        Self::from_boxed(
            Box::new(ml_outcome),
            Box::new(ml_t),
            TreatmentLearner::Classifier(Box::new(ml_m)),
            None,
        )
    }

    /// Boxed learners; `ml_a` defaults to a copy of `ml_m`.
    pub fn from_boxed(
        ml_outcome: Box<dyn Classifier>,
        ml_t: Box<dyn Regressor>,
        ml_m: TreatmentLearner,
        ml_a: Option<TreatmentLearner>,
    ) -> Self {
        Self {
            ml_outcome,
            ml_t,
            ml_m,
            ml_a,
            score: LplrScore::default(),
            n_folds_inner: 5,
            splitting: SampleSplitting::default(),
        }
    }

    pub fn score(mut self, score: LplrScore) -> Self {
        self.score = score;
        self
    }

    /// A separate learner for `a(X) = E[D | X]`, the regression used to build
    /// the preliminary estimate and `r(X)` (default: a copy of `ml_m`).
    pub fn ml_a(mut self, ml_a: TreatmentLearner) -> Self {
        self.ml_a = Some(ml_a);
        self
    }

    /// Inner folds of the double cross-fitting (default 5).
    pub fn n_folds_inner(mut self, k: usize) -> Self {
        self.n_folds_inner = k;
        self
    }

    /// Number of outer cross-fitting folds (default 5).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Number of independent sample splits (default 1), aggregated by the median.
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed for the outer and inner sample splits (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Draw seeded outer and inner sample splits and fit.
    pub fn fit(&self, data: &DmlData) -> Result<DmlFit> {
        self.validate(data)?;
        let splits = self.splitting.draw(data.n_obs())?;
        self.fit_with_folds(data, &splits)
    }

    /// Fit on given outer splits (one `Vec<Fold>` per repetition); the inner
    /// splits are drawn from the seed.
    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<DmlFit> {
        let inner = splits
            .iter()
            .enumerate()
            .map(|(r, folds)| {
                folds
                    .iter()
                    .enumerate()
                    .map(|(k, fold)| self.inner_folds(&fold.train, r, k))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        self.fit_with_inner_folds(data, splits, &inner)
    }

    /// Fit on given outer and inner splits: `inner[r][k]` are the inner folds
    /// (data indices, partitioning the training set of outer fold `k`) of
    /// repetition `r`.
    pub fn fit_with_inner_folds(
        &self,
        data: &DmlData,
        splits: &[Vec<Fold>],
        inner: &[Vec<Vec<Fold>>],
    ) -> Result<DmlFit> {
        self.validate(data)?;
        if splits.is_empty() || inner.len() != splits.len() {
            return Err(DmlError::InvalidArgument(
                "need one set of inner folds per repetition".into(),
            ));
        }
        for (folds, inner_r) in splits.iter().zip(inner) {
            if inner_r.len() != folds.len() {
                return Err(DmlError::InvalidArgument(
                    "need one set of inner folds per outer fold".into(),
                ));
            }
        }
        let fits = (0..data.n_treat())
            .map(|j| {
                let x = data.covariates_for(j);
                let d = data.treatment(j);
                let reps = splits
                    .iter()
                    .zip(inner)
                    .enumerate()
                    .map(|(rep, (folds, inner_r))| {
                        let ext = data.external_predictions().map(|preds| ExternalRef {
                            preds,
                            treatment: j,
                            rep,
                        });
                        self.fit_rep(data.y(), d, x.view(), folds, inner_r, ext)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mut fit = assemble(self.name(), data, splits[0].len(), reps);
                fit.treatment_names = vec![data.treatment_names()[j].clone()];
                Ok(fit)
            })
            .collect::<Result<Vec<_>>>()?;
        if fits.len() == 1 {
            Ok(fits.into_iter().next().expect("one fit"))
        } else {
            DmlFit::concat(&fits)
        }
    }

    fn name(&self) -> String {
        match self.score {
            LplrScore::NuisanceSpace => "LPLR (nuisance space)".into(),
            LplrScore::Instrument => "LPLR (instrument)".into(),
        }
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        if !is_binary(data.y()) {
            return Err(DmlError::InvalidArgument(
                "LPLR needs a binary (0/1) outcome".into(),
            ));
        }
        if data.clusters().is_some() {
            return Err(DmlError::InvalidArgument(
                "LPLR does not support clustered data (as DoubleML's double sample splitting)"
                    .into(),
            ));
        }
        let classifier = self.ml_m.is_classifier()
            || self
                .ml_a
                .as_ref()
                .is_some_and(TreatmentLearner::is_classifier);
        if classifier && !(0..data.n_treat()).all(|j| data.is_binary_treatment(j)) {
            return Err(DmlError::InvalidArgument(
                "a classifier for ml_m / ml_a needs binary (0/1) treatments".into(),
            ));
        }
        if self.n_folds_inner < 2 {
            return Err(DmlError::InvalidArgument(
                "n_folds_inner must be at least 2".into(),
            ));
        }
        Ok(())
    }

    /// Seeded inner K-fold split of one outer training fold.
    fn inner_folds(&self, train: &[usize], rep: usize, fold: usize) -> Result<Vec<Fold>> {
        let seed = self
            .splitting
            .seed
            .wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((rep * 1009 + fold + 1) as u64));
        Ok(KFold::new(self.n_folds_inner, seed)?
            .split(train.len())?
            .into_iter()
            .map(|f| Fold {
                train: f.train.iter().map(|&p| train[p]).collect(),
                test: f.test.iter().map(|&p| train[p]).collect(),
            })
            .collect())
    }

    /// Double cross-fitting: for every outer fold, out-of-inner-fold
    /// predictions on its training set, and the mean of the inner models'
    /// predictions on its test set.
    fn double_cross_fit(
        learner: Learner<'_>,
        x: ArrayView2<'_, f64>,
        target: ArrayView1<'_, f64>,
        folds: &[Fold],
        inner: &[Vec<Fold>],
    ) -> Result<(Array1<f64>, Vec<Array1<f64>>)> {
        let n = x.nrows();
        let per_fold = |k: usize| -> Result<(Array1<f64>, Array1<f64>)> {
            let fold = &folds[k];
            let x_test = x.select(Axis(0), &fold.test);
            let mut inner_pred = Array1::from_elem(n, f64::NAN);
            let mut outer = Array1::zeros(fold.test.len());
            for f in &inner[k] {
                let x_in = x.select(Axis(0), &f.test);
                let preds = learner.fit_predict(
                    x,
                    target,
                    &f.train,
                    None,
                    &[x_in.view(), x_test.view()],
                )?;
                for (p, &i) in preds[0].iter().zip(&f.test) {
                    inner_pred[i] = *p;
                }
                outer += &preds[1];
            }
            Ok((outer / inner[k].len() as f64, inner_pred))
        };
        let results: Vec<_> = crate::crossfit::map_folds(folds.len(), per_fold)?;
        let mut outer = Array1::from_elem(n, f64::NAN);
        let mut inner_preds = Vec::with_capacity(folds.len());
        for (fold, (pred, inner_pred)) in folds.iter().zip(results) {
            for (p, &i) in pred.iter().zip(&fold.test) {
                outer[i] = *p;
            }
            inner_preds.push(inner_pred);
        }
        Ok((outer, inner_preds))
    }

    fn fit_rep<'a>(
        &self,
        y: ArrayView1<'a, f64>,
        d: ArrayView1<'a, f64>,
        x: ArrayView2<'_, f64>,
        folds: &[Fold],
        inner: &[Vec<Fold>],
        ext: Option<ExternalRef<'_>>,
    ) -> Result<RepFit> {
        let n = y.len();
        let is_external = |name: &str| ext.is_some_and(|e| e.preds.contains(e.treatment, name));
        // External outer predictions of a double cross-fitted learner need
        // the inner predictions of every outer fold too.
        type Double = (Array1<f64>, Vec<Array1<f64>>);
        let double = |name: &str, fit: &dyn Fn() -> Result<Double>| -> Result<Double> {
            let Some(outer) = external_get(ext, n, name)? else {
                return fit();
            };
            let inner = (0..folds.len())
                .map(|k| {
                    external_get(ext, n, &format!("{name}_inner_{k}"))?.ok_or_else(|| {
                        DmlError::InvalidArgument(format!(
                            "external predictions for {name} also need {name}_inner_k \
                             for every outer fold k (missing k = {k})"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((outer, inner))
        };
        let xd = concatenate![Axis(1), d.insert_axis(Axis(1)), x];
        let (big_m, big_m_inner) = double("ml_M", &|| {
            Self::double_cross_fit(
                Learner::Classifier(self.ml_outcome.as_ref()),
                xd.view(),
                y,
                folds,
                inner,
            )
        })?;
        let ml_a = self.ml_a.as_ref().unwrap_or(&self.ml_m);
        let (a, a_inner) = double("ml_a", &|| {
            Self::double_cross_fit(ml_a.as_ref(), x, d, folds, inner)
        })?;

        // m(X): on Y = 0 (nuisance space) or weighted by M(1 − M) (instrument).
        let m = external_or(ext, n, "ml_m", || {
            let weights = big_m.mapv(|p| p * (1.0 - p));
            let mut m = Array1::from_elem(n, f64::NAN);
            for fold in folds {
                let (rows, w) = match self.score {
                    LplrScore::NuisanceSpace => (
                        fold.train
                            .iter()
                            .copied()
                            .filter(|&i| y[i] == 0.0)
                            .collect(),
                        None,
                    ),
                    LplrScore::Instrument => (fold.train.clone(), Some(weights.view())),
                };
                let x_test = x.select(Axis(0), &fold.test);
                let pred = self
                    .ml_m
                    .as_ref()
                    .fit_predict(x, d, &rows, w, &[x_test.view()])?;
                for (p, &i) in pred[0].iter().zip(&fold.test) {
                    m[i] = *p;
                }
            }
            Ok(m)
        })?;

        // Preliminary θ per fold from the inner predictions, and the targets
        // logit M of ml_t on each training fold.
        let mut beta = Array1::zeros(n);
        let mut t = Array1::from_elem(n, f64::NAN);
        let t_external = external_get(ext, n, "ml_t")?;
        for (k, fold) in folds.iter().enumerate() {
            let w: Array1<f64> = fold
                .train
                .iter()
                .map(|&i| logit(big_m_inner[k][i].clamp(1e-8, 1.0 - 1e-8)))
                .collect();
            let d_tilde: Vec<f64> = fold.train.iter().map(|&i| d[i] - a_inner[k][i]).collect();
            let cross: Vec<f64> = d_tilde.iter().zip(&w).map(|(a, b)| a * b).collect();
            let square: Vec<f64> = d_tilde.iter().map(|v| v * v).collect();
            let b = np_sum(&cross) / np_sum(&square);
            for &i in &fold.test {
                beta[i] = b;
            }
            if t_external.is_none() {
                let mut model = self.ml_t.fresh();
                model.fit(x.select(Axis(0), &fold.train).view(), w.view())?;
                let pred = model.predict(x.select(Axis(0), &fold.test).view())?;
                for (p, &i) in pred.iter().zip(&fold.test) {
                    t[i] = *p;
                }
            }
        }
        if let Some(ext_t) = t_external {
            t = ext_t;
        }
        let start = np_mean(beta.as_slice().expect("contiguous"));
        let r = &t - &(&beta * &a);

        let score = LplrElements::new(self.score, y, d, &r, &m);
        let (theta, converged) = newton(
            &mut |theta| np_mean(&score.psi(theta)),
            &mut |theta| np_mean(&score.deriv(theta)),
            start,
        );
        if !converged || !theta.is_finite() {
            return Err(DmlError::Convergence(format!(
                "could not find a root of the LPLR score (last value θ = {theta})"
            )));
        }
        let psi = Array1::from(score.psi(theta));
        let jacobian = np_mean(&score.deriv(theta));
        let kind = |l: &TreatmentLearner, name: &'static str| {
            if l.is_classifier() {
                NuisanceTarget::classification(name, d.to_owned())
            } else {
                NuisanceTarget::regression(name, d.to_owned())
            }
        };
        // Externally predicted learners have no out-of-sample loss.
        let targets = [
            kind(&self.ml_m, "ml_m"),
            kind(ml_a, "ml_a"),
            NuisanceTarget::classification("ml_M", y.to_owned()),
        ]
        .into_iter()
        .filter(|t| !is_external(&t.name))
        .collect();
        let mut predictions: Vec<(Name, Array1<f64>)> = vec![
            ("ml_r".into(), r),
            ("ml_m".into(), m),
            ("ml_a".into(), a),
            ("ml_t".into(), t),
            ("ml_M".into(), big_m),
        ];
        for (name, inner_preds) in [("ml_M", big_m_inner), ("ml_a", a_inner)] {
            for (k, p) in inner_preds.into_iter().enumerate() {
                predictions.push((format!("{name}_inner_{k}").into(), p));
            }
        }
        Ok(RepFit {
            next_start: start,
            theta,
            psi,
            jacobian,
            predictions,
            targets,
        })
    }
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// SciPy's `expit`, `1 / (1 + e^{−x})`.
fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Score elements, evaluated in DoubleML's operation order.
struct LplrElements<'a> {
    score: LplrScore,
    y: ArrayView1<'a, f64>,
    d: ArrayView1<'a, f64>,
    r: Array1<f64>,
    d_tilde: Array1<f64>,
    psi_hat: Array1<f64>,
    score_const: Array1<f64>,
}

impl<'a> LplrElements<'a> {
    fn new(
        score: LplrScore,
        y: ArrayView1<'a, f64>,
        d: ArrayView1<'a, f64>,
        r: &Array1<f64>,
        m: &Array1<f64>,
    ) -> Self {
        let d_tilde = &d - m;
        let psi_hat = r.mapv(|v| expit(-v));
        let score_const: Array1<f64> = (0..y.len())
            .map(|i| d_tilde[i] * (1.0 - y[i]) * r[i].exp())
            .collect();
        Self {
            score,
            y,
            d,
            r: r.clone(),
            d_tilde,
            psi_hat,
            score_const,
        }
    }

    fn psi(&self, theta: f64) -> Vec<f64> {
        (0..self.y.len())
            .map(|i| match self.score {
                LplrScore::NuisanceSpace => {
                    let s1 = self.y[i] * (-theta * self.d[i]).exp() * self.d_tilde[i];
                    self.psi_hat[i] * (s1 - self.score_const[i])
                }
                LplrScore::Instrument => {
                    (self.y[i] - expit(theta * self.d[i] + self.r[i])) * self.d_tilde[i]
                }
            })
            .collect()
    }

    fn deriv(&self, theta: f64) -> Vec<f64> {
        (0..self.y.len())
            .map(|i| match self.score {
                LplrScore::NuisanceSpace => {
                    let d1 = -self.y[i] * (-theta * self.d[i]).exp() * self.d[i];
                    self.psi_hat[i] * self.d_tilde[i] * d1
                }
                LplrScore::Instrument => {
                    let e = expit(theta * self.d[i] + self.r[i]);
                    -self.d[i] * e * (1.0 - e) * self.d_tilde[i]
                }
            })
            .collect()
    }
}

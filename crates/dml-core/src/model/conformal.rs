//! Conformal inference for counterfactuals and individual treatment effects:
//! weighted split conformal prediction under covariate shift (Tibshirani,
//! Barber, Candès and Ramdas 2019) applied to potential outcomes, as in Lei
//! and Candès (2021), "Conformal inference of counterfactuals and individual
//! treatment effects", Journal of the Royal Statistical Society B 83(5).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};

/// Conformity score of [`ConformalIte`].
pub enum ConformalScore {
    /// `|y − μ(x)|` with a regressor for the conditional mean (intervals of
    /// constant width `μ(x) ± Q`).
    Residual(Box<dyn Regressor>),
    /// Conformalised quantile regression (Romano, Patterson and Candès 2019):
    /// `max(q_lo(x) − y, y − q_hi(x))` with regressors fitted to a lower and an
    /// upper conditional quantile (intervals `[q_lo(x) − Q, q_hi(x) + Q]` that
    /// adapt to heteroscedasticity).
    Quantile {
        lower: Box<dyn Regressor>,
        upper: Box<dyn Regressor>,
    },
}

impl ConformalScore {
    fn fresh(&self) -> FittedScore {
        match self {
            ConformalScore::Residual(m) => FittedScore::Residual(m.fresh()),
            ConformalScore::Quantile { lower, upper } => {
                FittedScore::Quantile(lower.fresh(), upper.fresh())
            }
        }
    }
}

enum FittedScore {
    Residual(Box<dyn Regressor>),
    Quantile(Box<dyn Regressor>, Box<dyn Regressor>),
}

impl FittedScore {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        match self {
            FittedScore::Residual(m) => m.fit(x, y),
            FittedScore::Quantile(lo, hi) => {
                lo.fit(x, y)?;
                hi.fit(x, y)
            }
        }
    }

    /// Lower and upper band before calibration (equal for the residual score).
    fn band(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        match self {
            FittedScore::Residual(m) => {
                let mu = m.predict(x)?;
                Ok((mu.clone(), mu))
            }
            FittedScore::Quantile(lo, hi) => Ok((lo.predict(x)?, hi.predict(x)?)),
        }
    }

    fn scores(&self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<Array1<f64>> {
        let (lo, hi) = self.band(x)?;
        if lo.len() != y.len() || hi.len() != y.len() {
            return Err(DmlError::DimensionMismatch(
                "learner returned wrong number of predictions".into(),
            ));
        }
        Ok(match self {
            FittedScore::Residual(_) => (&y - &lo).mapv(f64::abs),
            FittedScore::Quantile(..) => {
                Array1::from_iter((0..y.len()).map(|i| (lo[i] - y[i]).max(y[i] - hi[i])))
            }
        })
    }
}

/// Population whose counterfactual outcomes the interval should cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConformalTarget {
    /// All units (the default; the target of individual-effect intervals).
    #[default]
    All,
    /// Units like the treated ones (e.g. `Y(0)` of a treated unit).
    Treated,
    /// Units like the control ones (e.g. `Y(1)` of a control unit).
    Control,
}

/// Prediction intervals for potential outcomes and individual treatment
/// effects of a binary treatment under unconfoundedness, with finite-sample
/// marginal coverage (Lei and Candès 2021).
///
/// The sample is split once into a training part (`train_fraction`, default
/// 0.75) and a calibration part. On the training part the conformity-score
/// learners are fitted separately per arm, and `ml_m` learns the propensity
/// score `e(x)` (or a known `propensity` is used). The calibration units of arm
/// `w` give conformity scores; since they come from `X | W = w` rather than the
/// target population, they are reweighted by the likelihood ratio, e.g.
/// `1/e(x)` for `Y(1)` over all units or `(1 − e(x))/e(x)` for `Y(1)` of
/// control units (weighted conformal prediction, Tibshirani et al. 2019). With
/// the true propensity score the intervals cover with probability at least
/// `1 − α`; with an estimated one, approximately.
///
/// The interval for an individual effect `Y(1) − Y(0)` of a new unit combines
/// the two counterfactual intervals at level `α/2` each:
/// `[L₁ − U₀, U₁ − L₀]` (conservative).
///
/// Estimated propensity scores are clipped to `[t, 1 − t]` (`t = 0.01`).
///
/// ```
/// use dml_core::{ConformalIte, ConformalScore, ConformalTarget, DmlData, LogisticRegression, Ols};
/// use ndarray::{Array1, Array2};
/// use rand::{Rng, SeedableRng};
/// use rand_distr::StandardNormal;
///
/// let n = 2000;
/// let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
/// let x = Array2::from_shape_simple_fn((n, 1), || rng.sample::<f64, _>(StandardNormal));
/// let d: Array1<f64> = (0..n).map(|_| f64::from(u8::from(rng.gen::<bool>()))).collect();
/// let y: Array1<f64> = (0..n).map(|i| x[[i, 0]] + d[i] + rng.sample::<f64, _>(StandardNormal)).collect();
/// let data = DmlData::new(y, d, x.clone())?;
///
/// let fit = ConformalIte::new(ConformalScore::Residual(Box::new(Ols::new())))
///     .propensity(0.5)
///     .fit(&data)?;
/// let new_x = Array2::from_shape_vec((1, 1), vec![0.3])?;
/// let ite = fit.predict_ite(new_x.view(), 0.1)?;            // [L, U] for Y(1) − Y(0)
/// assert!(ite[[0, 0]] < 1.0 && 1.0 < ite[[0, 1]]);
/// let y0 = fit.predict_counterfactual(new_x.view(), 0, ConformalTarget::All, 0.1)?;
/// assert!(y0[[0, 1]] - y0[[0, 0]] > 2.0);                    // about ±1.645σ, σ = 1
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ConformalIte {
    score: ConformalScore,
    ml_m: Option<Box<dyn Classifier>>,
    propensity: Option<f64>,
    train_fraction: f64,
    trimming_threshold: f64,
    seed: u64,
}

impl ConformalIte {
    pub fn new(score: ConformalScore) -> Self {
        Self {
            score,
            ml_m: None,
            propensity: None,
            train_fraction: 0.75,
            trimming_threshold: 1e-2,
            seed: 0,
        }
    }

    /// Learner of the propensity score `P(D = 1 | X)`.
    pub fn ml_m(mut self, ml_m: Box<dyn Classifier>) -> Self {
        self.ml_m = Some(ml_m);
        self
    }

    /// Known treatment probability of a randomised experiment.
    pub fn propensity(mut self, p: f64) -> Self {
        self.propensity = Some(p);
        self
    }

    /// Share of the sample used to fit the learners (default 0.75).
    pub fn train_fraction(mut self, fraction: f64) -> Self {
        self.train_fraction = fraction;
        self
    }

    /// Clip estimated propensity scores to `[t, 1 − t]` (default 0.01).
    pub fn trimming_threshold(mut self, threshold: f64) -> Self {
        self.trimming_threshold = threshold;
        self
    }

    /// Seed of the random split (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Split at random and fit.
    pub fn fit(&self, data: &DmlData) -> Result<ConformalFit> {
        let n = data.n_obs();
        if !(self.train_fraction > 0.0 && self.train_fraction < 1.0) {
            return Err(DmlError::InvalidArgument(
                "train_fraction must be in (0, 1)".into(),
            ));
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rand_chacha::ChaCha8Rng::seed_from_u64(self.seed));
        let n_train = ((n as f64) * self.train_fraction).round() as usize;
        let (train, calib) = idx.split_at(n_train);
        self.fit_with_split(data, train, calib)
    }

    /// Fit on a given split into training and calibration units.
    pub fn fit_with_split(
        &self,
        data: &DmlData,
        train: &[usize],
        calib: &[usize],
    ) -> Result<ConformalFit> {
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return Err(DmlError::InvalidArgument(
                "ConformalIte needs exactly one binary (0/1) treatment".into(),
            ));
        }
        let t = self.trimming_threshold;
        if t.is_nan() || !(0.0..0.5).contains(&t) {
            return Err(DmlError::InvalidArgument(
                "trimming_threshold must be in [0, 0.5)".into(),
            ));
        }
        let propensity = match (self.propensity, &self.ml_m) {
            (Some(p), _) if p > 0.0 && p < 1.0 => Propensity::Known(p),
            (Some(_), _) => {
                return Err(DmlError::InvalidArgument(
                    "the propensity must be in (0, 1)".into(),
                ))
            }
            (None, Some(m)) => {
                let mut m = m.fresh();
                m.fit(
                    data.x().select(Axis(0), train).view(),
                    gather(data.treatment(0), train).view(),
                )?;
                Propensity::Fitted(m, t)
            }
            (None, None) => {
                return Err(DmlError::InvalidArgument(
                    "pass a propensity learner (ml_m) or a known propensity".into(),
                ))
            }
        };
        let (x, y, d) = (data.x(), data.y(), data.treatment(0));
        let mut arms = Vec::with_capacity(2);
        for arm in [0.0, 1.0] {
            let tr: Vec<usize> = train.iter().copied().filter(|&i| d[i] == arm).collect();
            let ca: Vec<usize> = calib.iter().copied().filter(|&i| d[i] == arm).collect();
            if tr.is_empty() || ca.is_empty() {
                return Err(DmlError::InvalidArgument(format!(
                    "the training and the calibration part need units with treatment {arm}"
                )));
            }
            let mut model = self.score.fresh();
            model.fit(x.select(Axis(0), &tr).view(), gather(y, &tr).view())?;
            let xc = x.select(Axis(0), &ca);
            let scores = model.scores(xc.view(), gather(y, &ca).view())?;
            let e = propensity.predict(xc.view())?;
            // Sort the calibration units by score once.
            let mut order: Vec<usize> = (0..ca.len()).collect();
            order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
            arms.push(Arm {
                model,
                scores: order.iter().map(|&i| scores[i]).collect(),
                propensity: order.iter().map(|&i| e[i]).collect(),
            });
        }
        Ok(ConformalFit {
            arms,
            propensity,
            n_features: x.ncols(),
        })
    }
}

enum Propensity {
    Known(f64),
    Fitted(Box<dyn Classifier>, f64),
}

impl Propensity {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        match self {
            Propensity::Known(p) => Ok(Array1::from_elem(x.nrows(), *p)),
            Propensity::Fitted(m, t) => Ok(m.predict_proba(x)?.mapv(|v| v.clamp(*t, 1.0 - *t))),
        }
    }
}

struct Arm {
    model: FittedScore,
    /// Calibration scores in increasing order and the propensity scores of
    /// the same units.
    scores: Vec<f64>,
    propensity: Vec<f64>,
}

/// Likelihood ratio of the target population to `X | W = arm`, up to a constant.
fn shift_weight(arm: usize, target: ConformalTarget, e: f64) -> f64 {
    match (arm, target) {
        (1, ConformalTarget::All) => 1.0 / e,
        (1, ConformalTarget::Treated) => 1.0,
        (1, ConformalTarget::Control) => (1.0 - e) / e,
        (_, ConformalTarget::All) => 1.0 / (1.0 - e),
        (_, ConformalTarget::Treated) => e / (1.0 - e),
        (_, ConformalTarget::Control) => 1.0,
    }
}

/// Result of [`ConformalIte::fit`]: fitted learners and calibration scores.
pub struct ConformalFit {
    arms: Vec<Arm>,
    propensity: Propensity,
    n_features: usize,
}

impl ConformalFit {
    /// Intervals `[lower, upper]` (shape `(m, 2)`) for the potential outcome
    /// `Y(arm)` of new units `x` from the `target` population, at miscoverage
    /// level `alpha`.
    pub fn predict_counterfactual(
        &self,
        x: ArrayView2<'_, f64>,
        arm: usize,
        target: ConformalTarget,
        alpha: f64,
    ) -> Result<Array2<f64>> {
        if arm > 1 {
            return Err(DmlError::InvalidArgument("arm must be 0 or 1".into()));
        }
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(DmlError::InvalidArgument("alpha must be in (0, 1)".into()));
        }
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "x has {} columns, the model was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        let a = &self.arms[arm];
        let (lo, hi) = a.model.band(x)?;
        let e = self.propensity.predict(x)?;
        let w_cal: Vec<f64> = a
            .propensity
            .iter()
            .map(|&p| shift_weight(arm, target, p))
            .collect();
        let mut cum = Vec::with_capacity(w_cal.len());
        let mut total = 0.0;
        for w in &w_cal {
            total += w;
            cum.push(total);
        }
        let mut out = Array2::<f64>::zeros((x.nrows(), 2));
        for i in 0..x.nrows() {
            // Weighted (1 − α) quantile of the scores with mass w(x)/(W + w(x)) at +∞.
            let w_x = shift_weight(arm, target, e[i]);
            let need = (1.0 - alpha) * (total + w_x);
            let k = cum.partition_point(|&c| c < need);
            let q = if k < cum.len() {
                a.scores[k]
            } else {
                f64::INFINITY
            };
            out[[i, 0]] = lo[i] - q;
            out[[i, 1]] = hi[i] + q;
        }
        Ok(out)
    }

    /// Intervals for the individual effect `Y(1) − Y(0)` of new units (from the
    /// whole population) at level `alpha`: counterfactual intervals at `α/2`
    /// each, combined as `[L₁ − U₀, U₁ − L₀]`.
    pub fn predict_ite(&self, x: ArrayView2<'_, f64>, alpha: f64) -> Result<Array2<f64>> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(DmlError::InvalidArgument("alpha must be in (0, 1)".into()));
        }
        let one = self.predict_counterfactual(x, 1, ConformalTarget::All, alpha / 2.0)?;
        let zero = self.predict_counterfactual(x, 0, ConformalTarget::All, alpha / 2.0)?;
        let mut out = Array2::<f64>::zeros((x.nrows(), 2));
        for i in 0..x.nrows() {
            out[[i, 0]] = one[[i, 0]] - zero[[i, 1]];
            out[[i, 1]] = one[[i, 1]] - zero[[i, 0]];
        }
        Ok(out)
    }

    /// Number of calibration units in each arm (`[controls, treated]`).
    pub fn n_calibration(&self) -> [usize; 2] {
        [self.arms[0].scores.len(), self.arms[1].scores.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    /// `(x, y, d, y0, y1)`: covariates, observed outcome, treatment and both
    /// potential outcomes.
    type Sample = (
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
        Array1<f64>,
    );

    fn draw(n: usize, rng: &mut rand_chacha::ChaCha8Rng) -> Sample {
        let x = Array2::from_shape_simple_fn((n, 2), || rng.sample::<f64, _>(StandardNormal));
        let mut d = Array1::zeros(n);
        let mut y0 = Array1::zeros(n);
        let mut y1 = Array1::zeros(n);
        for i in 0..n {
            let e = 1.0 / (1.0 + (-x[[i, 0]]).exp());
            d[i] = f64::from(u8::from(rng.gen::<f64>() < e));
            let sd = 0.5 + x[[i, 1]].abs();
            y0[i] = x[[i, 0]] + sd * rng.sample::<f64, _>(StandardNormal);
            y1[i] = x[[i, 0]] + 1.0 + x[[i, 1]] + sd * rng.sample::<f64, _>(StandardNormal);
        }
        let y = Array1::from_iter((0..n).map(|i| if d[i] == 1.0 { y1[i] } else { y0[i] }));
        (x, y, d, y0, y1)
    }

    /// Marginal coverage of counterfactual and effect intervals for new units,
    /// with confounded assignment and an estimated propensity score.
    #[test]
    fn covers_counterfactuals_and_effects() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let (mut c1, mut c_ite, mut total) = (0usize, 0usize, 0usize);
        for rep in 0..20 {
            let (x, y, d, _, _) = draw(1500, &mut rng);
            let data = DmlData::new(y, d, x).unwrap();
            let fit = ConformalIte::new(ConformalScore::Residual(Box::new(Ols::new())))
                .ml_m(Box::new(LogisticRegression::unpenalized()))
                .seed(rep)
                .fit(&data)
                .unwrap();
            let (xt, _, _, y0, y1) = draw(500, &mut rng);
            let cf = fit
                .predict_counterfactual(xt.view(), 1, ConformalTarget::All, 0.1)
                .unwrap();
            let ite = fit.predict_ite(xt.view(), 0.1).unwrap();
            for i in 0..500 {
                c1 += usize::from(cf[[i, 0]] <= y1[i] && y1[i] <= cf[[i, 1]]);
                let tau = y1[i] - y0[i];
                c_ite += usize::from(ite[[i, 0]] <= tau && tau <= ite[[i, 1]]);
                total += 1;
            }
        }
        let (a, b) = (c1 as f64 / total as f64, c_ite as f64 / total as f64);
        assert!(a > 0.88 && a < 0.93, "counterfactual coverage {a}");
        assert!(b > 0.88, "ITE coverage {b}");
    }

    #[test]
    fn rejects_bad_input() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(2);
        let (x, y, d, _, _) = draw(200, &mut rng);
        let data = DmlData::new(y, d, x.clone()).unwrap();
        let score = || ConformalScore::Residual(Box::new(Ols::new()));
        assert!(ConformalIte::new(score()).fit(&data).is_err());
        assert!(ConformalIte::new(score())
            .propensity(1.5)
            .fit(&data)
            .is_err());
        let fit = ConformalIte::new(score())
            .propensity(0.5)
            .fit(&data)
            .unwrap();
        assert!(fit.predict_ite(x.view(), 1.5).is_err());
        assert!(fit
            .predict_counterfactual(x.view(), 2, ConformalTarget::All, 0.1)
            .is_err());
    }
}

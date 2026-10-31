//! Causal mediation analysis with double machine learning: Farbmacher, Huber,
//! Lafférs, Langen and Spindler (2022), "Causal mediation analysis with double
//! machine learning", The Econometrics Journal 25, 277–300, with the debiased
//! nested regressions of Qi and Zhang (2026, arXiv 2601.22592). Matches R's
//! `causalweight::medDML` (`multmed = TRUE`).

use ndarray::{concatenate, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use crate::crossfit::{map_folds, Fold, SampleSplitting};
use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};

/// Names of the four potential-outcome means `E[Y(d, M(d'))]`.
const POTENTIAL: [&str; 4] = ["Y(1, M(1))", "Y(1, M(0))", "Y(0, M(1))", "Y(0, M(0))"];

/// Names of the effects and their weights on [`POTENTIAL`].
const EFFECTS: [(&str, [f64; 4]); 5] = [
    ("total", [1.0, 0.0, 0.0, -1.0]),
    ("direct (treated)", [1.0, 0.0, -1.0, 0.0]),
    ("direct (control)", [0.0, 1.0, 0.0, -1.0]),
    ("indirect (treated)", [1.0, -1.0, 0.0, 0.0]),
    ("indirect (control)", [0.0, 0.0, 1.0, -1.0]),
];

/// How [`Mediation`] guards against propensity scores near 0 or 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TrimmingRule {
    /// Clip `P(D = 1 | X)` and `P(D = 1 | M, X)` to `[t, 1 − t]` and keep every
    /// observation (DoubleML's IRM rule).
    #[default]
    Truncate,
    /// Drop observations whose propensity products in a denominator (`p`,
    /// `1 − p`, `q(1 − p)`, `(1 − q)p`) fall below `t`: R's `medDML` rule.
    /// `q(M, X)` depends on the post-treatment mediator, so this selects the
    /// sample on a treatment outcome and biases the estimates when the mediator
    /// predicts the treatment well (the effects then refer to an unknown
    /// subpopulation). Use it to reproduce `medDML`.
    Drop,
}

/// Natural direct and indirect effects of a binary treatment `D` that works
/// partly through one or several mediators `M` (binary, categorical or
/// continuous), under sequential ignorability given covariates `X`.
///
/// The effects are differences of the potential-outcome means `E[Y(d, M(d'))]`,
/// the mean outcome under treatment `d` with the mediator at its distribution
/// under treatment `d'`:
///
/// - total effect `E[Y(1, M(1))] − E[Y(0, M(0))]` (the ATE),
/// - direct effects `E[Y(1, M(d))] − E[Y(0, M(d))]` for `d` = 1 ("treated") and 0 ("control"),
/// - indirect effects `E[Y(d, M(1))] − E[Y(d, M(0))]` for `d` = 1 and 0.
///
/// `E[Y(d, M(d))]` has the usual doubly robust (AIPW) score. The cross-world
/// means `E[Y(d, M(1 − d))]` use the efficient score of Farbmacher et al.
/// (2022), written with the propensity scores `p(X) = P(D = 1 | X)` and
/// `q(M, X) = P(D = 1 | M, X)` instead of mediator densities, so that `M` can
/// be multivariate and continuous. For `E[Y(1, M(0))]`:
///
/// ```text
/// ψ = D(1 − q)/(q(1 − p))·(Y − μ₁(M, X)) + (1 − D)/(1 − p)·(μ₁(M, X) − ω₁(X))
///     + D/p·(ω₁(X) − τ₁(X)) + τ₁(X)
/// ```
///
/// with `μ₁(M, X) = E[Y | D = 1, M, X]` (`ml_g`), the nested mean
/// `ω₁(X) = E[μ₁(M, X) | D = 0, X]` (`ml_nested`) and its debiased version
/// `τ₁(X)` (Qi and Zhang 2026: the regression of
/// `ω₁(X) + p(1 − q)/((1 − p)q)·(Y − μ₁)` on `X` among the treated); without
/// [`debias_nested`](Self::debias_nested), `τ₁ = ω₁` and the third term
/// vanishes. `E[Y(0, M(1))]` is the mirror image.
///
/// The nested regressions are cross-fitted in two stages: in every fold the
/// training folds are split into one half for `μ` and one for `ω` and `τ`
/// (with three folds: the next fold for `μ`, the one after for `ω`), as in
/// `medDML`, unless [`nested_split`](Self::nested_split) is off.
///
/// Propensity scores are clipped to `[0.01, 0.99]` by default
/// ([`TrimmingRule::Truncate`], [`trimming_threshold`](Self::trimming_threshold));
/// `medDML` instead drops observations ([`TrimmingRule::Drop`] with threshold
/// 0.05). With [`normalize_ipw`](Self::normalize_ipw) (default) each inverse
/// probability weight is scaled to mean one over the (kept) sample.
/// Standard errors come from the estimated influence functions; all estimates
/// share the sample splits, so the bootstrap gives joint inference.
///
/// ```
/// use dml_core::{DmlData, LogisticRegression, Mediation, Ols};
/// use ndarray::{Array1, Array2};
/// use rand::{Rng, SeedableRng};
/// use rand_distr::StandardNormal;
///
/// let n = 3000;
/// let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
/// let x = Array2::from_shape_simple_fn((n, 1), || rng.gen::<f64>() * 2.0 - 1.0);
/// let d: Array1<f64> = (0..n).map(|i| f64::from(u8::from(rng.gen::<f64>() < 0.4 + 0.2 * x[[i, 0]]))).collect();
/// let m = Array2::from_shape_fn((n, 1), |(i, _)| d[i] + x[[i, 0]] + rng.sample::<f64, _>(StandardNormal));
/// let y: Array1<f64> = (0..n).map(|i| d[i] + 2.0 * m[[i, 0]] + x[[i, 0]] + rng.gen::<f64>()).collect();
/// let data = DmlData::new(y, d, x)?.with_mediators(m)?;
///
/// let fit = Mediation::new(Ols::new(), LogisticRegression::unpenalized()).fit(&data)?;
/// let effects = fit.effects();
/// // Direct effect 1, indirect effect 2 (one unit of M per unit of D).
/// assert!((effects.coef()[1] - 1.0).abs() < 0.15);
/// assert!((effects.coef()[3] - 2.0).abs() < 0.15);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
pub struct Mediation {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    ml_nested: Box<dyn Regressor>,
    trimming_rule: TrimmingRule,
    trimming_threshold: f64,
    normalize_ipw: bool,
    nested_split: bool,
    debias_nested: bool,
    splitting: SampleSplitting,
}

impl Mediation {
    /// `ml_g` learns the outcome means `E[Y | D = d, X]` and `E[Y | D = d, M, X]`
    /// (fitted separately per treatment arm), `ml_m` the propensity scores
    /// `P(D = 1 | X)` and `P(D = 1 | M, X)`. The nested means use a fresh copy
    /// of `ml_g` unless [`ml_nested`](Self::ml_nested) is set.
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    /// Like [`Mediation::new`] for boxed trait objects.
    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Box<dyn Classifier>) -> Self {
        let ml_nested = ml_g.fresh();
        Self {
            ml_g,
            ml_m,
            ml_nested,
            trimming_rule: TrimmingRule::default(),
            trimming_threshold: 1e-2,
            normalize_ipw: true,
            nested_split: true,
            debias_nested: true,
            splitting: SampleSplitting::default(),
        }
    }

    /// Learner of the nested means `ω_d(X)` and `τ_d(X)` (regressions of
    /// predicted outcomes on `X`).
    pub fn ml_nested(self, ml_nested: impl Regressor + 'static) -> Self {
        self.ml_nested_boxed(Box::new(ml_nested))
    }

    /// Like [`ml_nested`](Self::ml_nested) for a boxed trait object.
    pub fn ml_nested_boxed(mut self, ml_nested: Box<dyn Regressor>) -> Self {
        self.ml_nested = ml_nested;
        self
    }

    /// Clip the propensity scores (default) or drop observations, see
    /// [`TrimmingRule`]. `medDML` = `Drop` with threshold 0.05.
    pub fn trimming_rule(mut self, rule: TrimmingRule) -> Self {
        self.trimming_rule = rule;
        self
    }

    /// Threshold `t` of the [`TrimmingRule`] (default 0.01).
    pub fn trimming_threshold(mut self, threshold: f64) -> Self {
        self.trimming_threshold = threshold;
        self
    }

    /// Normalise each inverse probability weight to mean one over the kept
    /// observations (default `true`).
    pub fn normalize_ipw(mut self, yes: bool) -> Self {
        self.normalize_ipw = yes;
        self
    }

    /// Fit `μ` and the nested means on disjoint parts of the training folds
    /// (default `true`; `false` = `medDML(fewsplits = TRUE)`).
    pub fn nested_split(mut self, yes: bool) -> Self {
        self.nested_split = yes;
        self
    }

    /// Debias the nested means with reweighted residuals of `μ` (Qi and Zhang
    /// 2026; default `true`, as `medDML(debiasfirst = TRUE)`).
    pub fn debias_nested(mut self, yes: bool) -> Self {
        self.debias_nested = yes;
        self
    }

    /// Number of cross-fitting folds (default 5; `medDML` uses 3). With the
    /// nested split at least 3.
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Number of independent sample splits (default 1), aggregated by the median.
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed for the sample splits (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Draw seeded sample splits and fit.
    pub fn fit(&self, data: &DmlData) -> Result<MediationFit> {
        let splits = self.splitting.draw(data.n_obs())?;
        self.fit_with_folds(data, &splits)
    }

    /// Fit on given K-fold splits (one `Vec<Fold>` per repetition; the test
    /// sets must partition the sample). The nested split of fold `k` uses the
    /// folds `k + 1, …` for `μ` and the remaining ones for the nested means, so
    /// `medDML`'s three folds (in its order) reproduce its estimates.
    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<MediationFit> {
        if data.clusters().is_some() {
            return Err(DmlError::InvalidArgument(
                "Mediation does not support clustered data yet".into(),
            ));
        }
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return Err(DmlError::InvalidArgument(
                "Mediation needs exactly one binary (0/1) treatment".into(),
            ));
        }
        let m = data.mediators().ok_or_else(|| {
            DmlError::InvalidArgument("Mediation needs mediators (DmlData::with_mediators)".into())
        })?;
        let trim = self.trimming_threshold;
        if trim.is_nan() || !(0.0..0.5).contains(&trim) {
            return Err(DmlError::InvalidArgument(
                "trimming_threshold must be in [0, 0.5)".into(),
            ));
        }
        if splits.is_empty() {
            return Err(DmlError::InvalidArgument(
                "need at least one repetition of sample splits".into(),
            ));
        }
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let n = y.len();
        let xm = concatenate(Axis(1), &[x, m]).expect("same rows");
        let n_rep = splits.len();
        let n_folds = splits[0].len();

        let mut psi_a = Array3::<f64>::zeros((n, 4, n_rep));
        let mut psi_b = Array3::<f64>::zeros((n, 4, n_rep));
        let mut stored: Vec<(String, Array3<f64>)> = Vec::new();
        let mut n_trimmed = Vec::with_capacity(n_rep);
        for (r, folds) in splits.iter().enumerate() {
            let mut nu = self.nuisances(y, d, x, xm.view(), folds)?;
            let kept: Vec<bool> = match self.trimming_rule {
                TrimmingRule::Truncate => {
                    nu.p.mapv_inplace(|v| v.clamp(trim, 1.0 - trim));
                    nu.q.mapv_inplace(|v| v.clamp(trim, 1.0 - trim));
                    vec![true; n]
                }
                TrimmingRule::Drop => (0..n)
                    .map(|i| {
                        let (p, q) = (nu.p[i], nu.q[i]);
                        (1.0 - q) * p >= trim
                            && 1.0 - p >= trim
                            && p >= trim
                            && q * (1.0 - p) >= trim
                    })
                    .collect(),
            };
            let n_kept = kept.iter().filter(|&&k| k).count();
            if n_kept == 0 {
                return Err(DmlError::InvalidArgument(
                    "every observation is trimmed; lower trimming_threshold".into(),
                ));
            }
            n_trimmed.push(n - n_kept);
            // Inverse probability weights: a for Y(1, M(0)), e for Y(0, M(1)),
            // c = D/p and b = (1 − D)/(1 − p).
            let w = |i: usize| {
                let (di, p, q) = (d[i], nu.p[i], nu.q[i]);
                [
                    di * (1.0 - q) / (q * (1.0 - p)),
                    (1.0 - di) * q / ((1.0 - q) * p),
                    di / p,
                    (1.0 - di) / (1.0 - p),
                ]
            };
            let mut scale = [1.0; 4];
            if self.normalize_ipw {
                let mut sums = [0.0; 4];
                for i in (0..n).filter(|&i| kept[i]) {
                    for (s, v) in sums.iter_mut().zip(w(i)) {
                        *s += v;
                    }
                }
                for (f, s) in scale.iter_mut().zip(sums) {
                    if s <= 0.0 {
                        return Err(DmlError::InvalidArgument(
                            "a treatment arm has no kept observations; lower trimming_threshold"
                                .into(),
                        ));
                    }
                    *f = n_kept as f64 / s;
                }
            }
            for i in (0..n).filter(|&i| kept[i]) {
                let [a, e, c, b] = w(i);
                let [a, e, c, b] = [a * scale[0], e * scale[1], c * scale[2], b * scale[3]];
                let yi = y[i];
                let scores = [
                    nu.g1[i] + c * (yi - nu.g1[i]),
                    a * (yi - nu.mu1[i])
                        + b * (nu.mu1[i] - nu.om1[i])
                        + c * (nu.om1[i] - nu.tau1[i])
                        + nu.tau1[i],
                    e * (yi - nu.mu0[i])
                        + c * (nu.mu0[i] - nu.om0[i])
                        + b * (nu.om0[i] - nu.tau0[i])
                        + nu.tau0[i],
                    nu.g0[i] + b * (yi - nu.g0[i]),
                ];
                for (j, s) in scores.into_iter().enumerate() {
                    psi_a[[i, j, r]] = -1.0;
                    psi_b[[i, j, r]] = s;
                }
            }
            let mut named = vec![
                ("ml_g_d1", nu.g1),
                ("ml_g_d0", nu.g0),
                ("ml_mu_d1", nu.mu1),
                ("ml_mu_d0", nu.mu0),
                ("ml_nested_d1", nu.om1),
                ("ml_nested_d0", nu.om0),
            ];
            if self.debias_nested {
                named.push(("ml_nested_debiased_d1", nu.tau1));
                named.push(("ml_nested_debiased_d0", nu.tau0));
            }
            named.push(("ml_m", nu.p));
            named.push(("ml_m_mediator", nu.q));
            if r == 0 {
                stored = named
                    .iter()
                    .map(|(name, _)| (name.to_string(), Array3::zeros((n, n_rep, 1))))
                    .collect();
            }
            for ((_, all), (_, v)) in stored.iter_mut().zip(named) {
                all.slice_mut(ndarray::s![.., r, 0]).assign(&v);
            }
        }

        let mut potential = DmlFit::from_linear_scores(
            "Mediation",
            POTENTIAL.iter().map(|s| s.to_string()).collect(),
            n_folds,
            psi_a,
            psi_b,
        )?;
        potential.predictions = stored;
        let weights = Array2::from_shape_fn((EFFECTS.len(), 4), |(e, j)| EFFECTS[e].1[j]);
        let mut effects = potential.combine(
            "Mediation".into(),
            EFFECTS.iter().map(|(name, _)| name.to_string()).collect(),
            &weights,
        );
        effects.predictions = potential.predictions.clone();
        Ok(MediationFit {
            effects,
            potential,
            n_trimmed,
        })
    }

    /// Out-of-fold nuisance predictions for one repetition.
    fn nuisances(
        &self,
        y: ArrayView1<'_, f64>,
        d: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        xm: ArrayView2<'_, f64>,
        folds: &[Fold],
    ) -> Result<Nuisances> {
        let n = y.len();
        let roles = nested_roles(n, folds, self.nested_split)?;
        let clip = |v: f64| v.clamp(self.trimming_threshold, 1.0 - self.trimming_threshold);
        let per_fold = map_folds(folds.len(), |k| -> Result<[Array1<f64>; 10]> {
            let (train, test) = (&folds[k].train, &folds[k].test);
            let (mu_rows, delta_rows) = &roles[k];
            let arm = |rows: &[usize], v: f64| -> Vec<usize> {
                rows.iter().copied().filter(|&i| d[i] == v).collect()
            };

            let p = classify(&*self.ml_m, x, train, d, test)?;
            let q = classify(&*self.ml_m, xm, train, d, test)?;
            let g1 = regress(&*self.ml_g, x, &arm(train, 1.0), y, test)?;
            let g0 = regress(&*self.ml_g, x, &arm(train, 0.0), y, test)?;

            // μ_d on the first part, then its predictions on the second part are
            // regressed on X among the other arm (ω_d).
            let mut out = Vec::with_capacity(4);
            let (delta1, delta0) = (arm(delta_rows, 1.0), arm(delta_rows, 0.0));
            let odds = if self.debias_nested {
                // p/(1 − p)·(1 − q)/q from the μ part = f(M | D=0, X)/f(M | D=1, X).
                let pa = classify(&*self.ml_m, x, mu_rows, d, delta_rows)?;
                let qa = classify(&*self.ml_m, xm, mu_rows, d, delta_rows)?;
                let odds: std::collections::HashMap<usize, f64> = delta_rows
                    .iter()
                    .zip(pa.iter().zip(&qa))
                    .map(|(&i, (&pa, &qa))| {
                        let (pa, qa) = (clip(pa), clip(qa));
                        (i, pa / (1.0 - pa) * (1.0 - qa) / qa)
                    })
                    .collect();
                Some(odds)
            } else {
                None
            };
            for (arm_value, own, other) in [(1.0, &delta1, &delta0), (0.0, &delta0, &delta1)] {
                let mu_fit = fit_regressor(
                    &*self.ml_g,
                    xm,
                    &arm(mu_rows, arm_value),
                    gather(y, &arm(mu_rows, arm_value)).view(),
                    "outcome",
                )?;
                let mu_test = predict_rows(&*mu_fit, xm, test)?;
                // ω_d: E[μ_d(M, X) | D = 1 − d, X].
                let mu_other = predict_rows(&*mu_fit, xm, other)?;
                let om_fit = fit_regressor(&*self.ml_nested, x, other, mu_other.view(), "nested")?;
                let om_test = predict_rows(&*om_fit, x, test)?;
                let tau_test = match &odds {
                    Some(odds) => {
                        // τ_d: regress ω_d(X) + weight·(Y − μ_d(M, X)) on X in arm d,
                        // weight = odds (d = 1) or 1/odds (d = 0).
                        let om_own = predict_rows(&*om_fit, x, own)?;
                        let mu_own = predict_rows(&*mu_fit, xm, own)?;
                        let target: Array1<f64> = own
                            .iter()
                            .enumerate()
                            .map(|(j, &i)| {
                                let w = if arm_value == 1.0 {
                                    odds[&i]
                                } else {
                                    1.0 / odds[&i]
                                };
                                om_own[j] + w * (y[i] - mu_own[j])
                            })
                            .collect();
                        let tau_fit =
                            fit_regressor(&*self.ml_nested, x, own, target.view(), "nested")?;
                        predict_rows(&*tau_fit, x, test)?
                    }
                    None => om_test.clone(),
                };
                out.push((mu_test, om_test, tau_test));
            }
            let (mu0, om0, tau0) = out.pop().expect("two arms");
            let (mu1, om1, tau1) = out.pop().expect("two arms");
            Ok([p, q, g1, g0, mu1, mu0, om1, om0, tau1, tau0])
        })?;

        let mut all: [Array1<f64>; 10] = std::array::from_fn(|_| Array1::from_elem(n, f64::NAN));
        for (fold, values) in folds.iter().zip(per_fold) {
            for (target, v) in all.iter_mut().zip(values) {
                for (j, &i) in fold.test.iter().enumerate() {
                    target[i] = v[j];
                }
            }
        }
        if all.iter().any(|a| a.iter().any(|v| !v.is_finite())) {
            return Err(DmlError::InvalidArgument(
                "a learner returned a non-finite prediction".into(),
            ));
        }
        let [p, q, g1, g0, mu1, mu0, om1, om0, tau1, tau0] = all;
        Ok(Nuisances {
            p,
            q,
            g1,
            g0,
            mu1,
            mu0,
            om1,
            om0,
            tau1,
            tau0,
        })
    }
}

/// Out-of-fold nuisance predictions (length `n`).
struct Nuisances {
    /// `P(D = 1 | X)` and `P(D = 1 | M, X)`.
    p: Array1<f64>,
    q: Array1<f64>,
    /// `E[Y | D = d, X]`.
    g1: Array1<f64>,
    g0: Array1<f64>,
    /// `E[Y | D = d, M, X]`.
    mu1: Array1<f64>,
    mu0: Array1<f64>,
    /// Nested means `E[μ_d(M, X) | D = 1 − d, X]` and their debiased versions.
    om1: Array1<f64>,
    om0: Array1<f64>,
    tau1: Array1<f64>,
    tau0: Array1<f64>,
}

/// For every fold, the rows of the `μ` part and of the nested part of its
/// training sample. Checks that the test sets partition the sample.
fn nested_roles(n: usize, folds: &[Fold], split: bool) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    let k = folds.len();
    let not_kfold = || {
        DmlError::InvalidArgument(
            "Mediation needs K-fold splits: the test sets partition the sample and each \
             training set is the rest"
                .into(),
        )
    };
    let mut fold_of = vec![usize::MAX; n];
    for (f, fold) in folds.iter().enumerate() {
        for &i in &fold.test {
            if i >= n || fold_of[i] != usize::MAX {
                return Err(not_kfold());
            }
            fold_of[i] = f;
        }
    }
    if fold_of.contains(&usize::MAX) {
        return Err(not_kfold());
    }
    for (f, fold) in folds.iter().enumerate() {
        if fold.train.len() + fold.test.len() != n
            || fold.train.iter().any(|&i| i >= n || fold_of[i] == f)
        {
            return Err(not_kfold());
        }
    }
    if !split {
        return Ok(folds
            .iter()
            .map(|f| (f.train.clone(), f.train.clone()))
            .collect());
    }
    if k < 3 {
        return Err(DmlError::InvalidArgument(
            "the nested split needs at least 3 folds (or nested_split(false))".into(),
        ));
    }
    let n_mu = k / 2;
    Ok((0..k)
        .map(|f| {
            let rows = |offsets: std::ops::Range<usize>| -> Vec<usize> {
                offsets
                    .flat_map(|o| folds[(f + o) % k].test.iter().copied())
                    .collect()
            };
            (rows(1..1 + n_mu), rows(1 + n_mu..k))
        })
        .collect())
}

fn fit_regressor(
    proto: &dyn Regressor,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
    target: ArrayView1<'_, f64>,
    what: &str,
) -> Result<Box<dyn Regressor>> {
    if rows.is_empty() {
        return Err(DmlError::InvalidArgument(format!(
            "a training sample has no observations for an {what} regression (a treatment arm \
             is empty); use fewer folds"
        )));
    }
    let mut model = proto.fresh();
    model.fit(x.select(Axis(0), rows).view(), target)?;
    Ok(model)
}

/// Fit `proto` on `rows` (target `y[rows]`) and predict at `at`.
fn regress(
    proto: &dyn Regressor,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
    y: ArrayView1<'_, f64>,
    at: &[usize],
) -> Result<Array1<f64>> {
    let model = fit_regressor(proto, x, rows, gather(y, rows).view(), "outcome")?;
    predict_rows(&*model, x, at)
}

fn predict_rows(
    model: &dyn Regressor,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
) -> Result<Array1<f64>> {
    let pred = model.predict(x.select(Axis(0), rows).view())?;
    check_len(pred, rows.len())
}

/// Fit the classifier on `rows` (target `d[rows]`) and predict `P(D = 1)` at `at`.
fn classify(
    proto: &dyn Classifier,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
    d: ArrayView1<'_, f64>,
    at: &[usize],
) -> Result<Array1<f64>> {
    let mut model = proto.fresh();
    model.fit(x.select(Axis(0), rows).view(), gather(d, rows).view())?;
    check_len(model.predict_proba(x.select(Axis(0), at).view())?, at.len())
}

fn check_len(v: Array1<f64>, n: usize) -> Result<Array1<f64>> {
    if v.len() == n {
        Ok(v)
    } else {
        Err(DmlError::DimensionMismatch(
            "learner returned wrong number of predictions".into(),
        ))
    }
}

/// Result of [`Mediation::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MediationFit {
    effects: DmlFit,
    potential: DmlFit,
    n_trimmed: Vec<usize>,
}

impl MediationFit {
    /// Total, direct (treated, control) and indirect (treated, control)
    /// effects, in this order. Predictions of every nuisance are attached
    /// (`"ml_g_d1"`, `"ml_mu_d1"`, `"ml_nested_d1"`, `"ml_m"`,
    /// `"ml_m_mediator"`, ...).
    pub fn effects(&self) -> &DmlFit {
        &self.effects
    }

    /// The potential-outcome means `E[Y(1, M(1))]`, `E[Y(1, M(0))]`,
    /// `E[Y(0, M(1))]` and `E[Y(0, M(0))]`.
    pub fn potential_outcomes(&self) -> &DmlFit {
        &self.potential
    }

    /// Number of trimmed observations in each repetition.
    pub fn n_trimmed(&self) -> &[usize] {
        &self.n_trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    fn simulate(n: usize, seed: u64) -> DmlData {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut normal = || rng.sample::<f64, _>(StandardNormal);
        let x = Array2::from_shape_fn((n, 3), |_| normal());
        let mut d = Array1::zeros(n);
        let mut m = Array2::zeros((n, 1));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            d[i] = f64::from(u8::from(0.5 * x[[i, 0]] - 0.3 * x[[i, 1]] + normal() > 0.0));
            m[[i, 0]] = 0.8 * d[i] + 0.5 * x[[i, 0]] + normal();
            y[i] = 0.5 * d[i] + m[[i, 0]] + 0.4 * x[[i, 1]] - 0.2 * x[[i, 2]] + normal();
        }
        DmlData::new(y, d, x).unwrap().with_mediators(m).unwrap()
    }

    #[test]
    fn nested_roles_follow_med_dml() {
        let folds: Vec<Fold> = (0..3)
            .map(|k| Fold {
                train: (0..6).filter(|i| i % 3 != k).collect(),
                test: (0..6).filter(|i| i % 3 == k).collect(),
            })
            .collect();
        let roles = nested_roles(6, &folds, true).unwrap();
        // Fold 0: μ on fold 1, nested means on fold 2; fold 2: μ on fold 0.
        assert_eq!(roles[0], (vec![1, 4], vec![2, 5]));
        assert_eq!(roles[2], (vec![0, 3], vec![1, 4]));
        let bad = vec![folds[0].clone(), folds[1].clone()];
        assert!(nested_roles(6, &bad, true).is_err());
    }

    /// Linear design with direct effect 0.5 and indirect effect 0.8: the
    /// estimates should be within a few standard errors.
    #[test]
    fn recovers_direct_and_indirect_effects() {
        let data = simulate(4000, 3);
        for debias in [true, false] {
            let fit = Mediation::new(Ols::new(), LogisticRegression::unpenalized())
                .debias_nested(debias)
                .fit(&data)
                .unwrap();
            let e = fit.effects();
            let truth = [1.3, 0.5, 0.5, 0.8, 0.8];
            for (j, t) in truth.iter().enumerate() {
                let z = (e.coef()[j] - t) / e.se()[j];
                assert!(z.abs() < 4.0, "{} z = {z}", e.treatment_names()[j]);
            }
            // Effects are the differences of the potential outcomes.
            let po = fit.potential_outcomes().coef();
            assert!((e.coef()[0] - (po[0] - po[3])).abs() < 1e-12);
            assert!((e.coef()[3] - (po[0] - po[1])).abs() < 1e-12);
        }
    }

    /// A mediator that predicts the treatment well: dropping on `q(M, X)`
    /// selects on a post-treatment variable and biases the total effect
    /// (truth 3), clipping does not.
    #[test]
    fn dropping_on_the_mediator_propensity_biases() {
        let n = 20_000;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let x = Array2::from_shape_simple_fn((n, 1), || rng.gen::<f64>() * 2.0 - 1.0);
        let d: Array1<f64> = (0..n)
            .map(|i| f64::from(u8::from(rng.gen::<f64>() < 0.4 + 0.2 * x[[i, 0]])))
            .collect();
        let m = Array2::from_shape_fn((n, 1), |(i, _)| {
            d[i] + x[[i, 0]] + rng.sample::<f64, _>(StandardNormal)
        });
        let y: Array1<f64> = (0..n)
            .map(|i| d[i] + 2.0 * m[[i, 0]] + x[[i, 0]] + rng.sample::<f64, _>(StandardNormal))
            .collect();
        let data = DmlData::new(y, d, x).unwrap().with_mediators(m).unwrap();
        let z = |rule, t| {
            let fit = Mediation::new(Ols::new(), LogisticRegression::unpenalized())
                .trimming_rule(rule)
                .trimming_threshold(t)
                .fit(&data)
                .unwrap();
            (fit.effects().coef()[0] - 3.0) / fit.effects().se()[0]
        };
        assert!(z(TrimmingRule::Truncate, 0.01).abs() < 3.0);
        assert!(z(TrimmingRule::Drop, 0.05) < -5.0);
    }

    #[test]
    fn rejects_bad_input() {
        let data = simulate(200, 1);
        let model = Mediation::new(Ols::new(), LogisticRegression::unpenalized());
        let no_m = DmlData::new(
            data.y().to_owned(),
            data.treatment(0).to_owned(),
            data.x().to_owned(),
        )
        .unwrap();
        assert!(model.fit(&no_m).is_err());
        assert!(
            Mediation::new(Ols::new(), LogisticRegression::unpenalized())
                .n_folds(2)
                .fit(&data)
                .is_err()
        );
        assert!(
            Mediation::new(Ols::new(), LogisticRegression::unpenalized())
                .n_folds(2)
                .nested_split(false)
                .fit(&data)
                .is_ok()
        );
        assert!(
            Mediation::new(Ols::new(), LogisticRegression::unpenalized())
                .trimming_threshold(0.6)
                .fit(&data)
                .is_err()
        );
    }
}

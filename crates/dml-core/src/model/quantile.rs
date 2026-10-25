//! Quantile treatment effects of a binary treatment, as DoubleML's
//! `DoubleMLPQ`, `DoubleMLLPQ`, `DoubleMLCVAR` and `DoubleMLQTE`
//! (Kallus, Mao and Uehara, 2024):
//!
//! - [`Pq`]: the potential quantile `θ` with `P(Y(d) ≤ θ) = τ`;
//! - [`Lpq`]: the local (complier) potential quantile, with a binary instrument;
//! - [`Cvar`]: the conditional value at risk `E[Y(d) | Y(d) ≥ q_τ]`;
//! - [`Qte`]: treatment effects on quantiles (or CVaRs), `θ(1) − θ(0)` for a
//!   grid of `τ`, with joint inference.
//!
//! The nuisances are localised at a preliminary IPW estimate of the quantile,
//! computed by nested cross-fitting within each training fold: as in DoubleML,
//! the training fold is split in halves (stratified, scikit-learn's
//! `train_test_split` with `random_state=42`); the preliminary estimate comes
//! from the first half and the outcome model is fitted on the second. The
//! potential quantile solves a non-linear score with Brent's method; its
//! standard error uses a kernel density estimate. The numerical routines are
//! exact ports of NumPy, scikit-learn, SciPy and statsmodels (see `compat`),
//! so results match DoubleML on the same sample splits.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::irm::normalize_propensity;
use super::nonlinear::{assemble, RepFit};
use super::{
    dml_model_api, external_get, ExternalRef, LinearScoreModel, NuisanceTarget, ScoreEstimate,
    ScoreInput,
};
use crate::compat::{
    bracket_guess, brent_minimize, brentq, kde_at_zero, np_mean, np_quantile, stratified_kfold,
    stratified_train_test_split,
};
use crate::crossfit::{cross_fit_predict_proba, Fold, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};
use crate::propensity::{adjust_propensity, Calibration};
use crate::score::LinearScore;

/// Settings shared by the quantile models.
#[derive(Debug, Clone, Copy)]
struct Settings {
    treatment: f64,
    quantile: f64,
    clipping_threshold: f64,
    calibration: Calibration,
    normalize_ipw: bool,
}

impl Settings {
    fn new() -> Self {
        Self {
            treatment: 1.0,
            quantile: 0.5,
            clipping_threshold: 1e-2,
            calibration: Calibration::None,
            normalize_ipw: true,
        }
    }

    fn check(&self) -> Result<()> {
        if self.treatment != 0.0 && self.treatment != 1.0 {
            return Err(DmlError::InvalidArgument("treatment must be 0 or 1".into()));
        }
        if !(self.quantile > 0.0 && self.quantile < 1.0) {
            return Err(DmlError::InvalidArgument(
                "quantile must be in (0, 1)".into(),
            ));
        }
        if !(self.clipping_threshold > 0.0 && self.clipping_threshold < 0.5) {
            return Err(DmlError::InvalidArgument(
                "clipping_threshold must be in (0, 0.5)".into(),
            ));
        }
        Ok(())
    }

    /// Clip, optionally normalise with the indicator `ind`, and for treatment
    /// 0 flip to `P(D = 0 | X)` (DoubleML's propensity adjustment).
    fn adjust(
        &self,
        m: &Array1<f64>,
        ind: &Array1<f64>,
        flip: bool,
        folds: &[Fold],
    ) -> Result<Array1<f64>> {
        let clipped = self.clip(m, ind, folds)?;
        let normalized = if self.normalize_ipw {
            normalize_propensity(&clipped, ind)
        } else {
            clipped
        };
        Ok(if flip && self.treatment == 0.0 {
            1.0 - normalized
        } else {
            normalized
        })
    }

    /// Calibrate (optionally, against `ind`) and clip.
    fn clip(&self, m: &Array1<f64>, ind: &Array1<f64>, folds: &[Fold]) -> Result<Array1<f64>> {
        adjust_propensity(
            m.clone(),
            ind.view(),
            folds,
            self.clipping_threshold,
            self.calibration,
        )
    }
}

fn check_data(data: &DmlData, instrument: bool) -> Result<()> {
    if data.n_treat() != 1 || !data.is_binary_treatment(0) {
        return Err(DmlError::InvalidArgument(
            "quantile models need exactly one binary (0/1) treatment".into(),
        ));
    }
    if data.clusters().is_some() {
        return Err(DmlError::InvalidArgument(
            "quantile models do not support clustered data".into(),
        ));
    }
    if instrument && data.n_instruments() > 1 {
        return Err(DmlError::InvalidArgument(
            "LPQ needs exactly one instrument".into(),
        ));
    }
    match (instrument, data.z()) {
        (true, None) => Err(DmlError::InvalidArgument(
            "LPQ needs a binary instrument (DmlData::with_instrument)".into(),
        )),
        (true, Some(z)) if !z.iter().all(|&v| v == 0.0 || v == 1.0) => Err(
            DmlError::InvalidArgument("the instrument must be binary (0/1)".into()),
        ),
        _ => Ok(()),
    }
}

fn take(a: ArrayView1<'_, f64>, idx: &[usize]) -> Array1<f64> {
    idx.iter().map(|&i| a[i]).collect()
}

fn min_max(y: ArrayView1<'_, f64>) -> (f64, f64) {
    y.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        })
}

/// Split a training fold into halves stratified by `strata`, as DoubleML's
/// `train_test_split(train, test_size=0.5, random_state=42, stratify=…)`.
/// Returns data indices, in scikit-learn's (shuffled) order.
pub(super) fn half_split(train: &[usize], strata: &[f64]) -> Result<(Vec<usize>, Vec<usize>)> {
    let labels: Vec<f64> = train.iter().map(|&i| strata[i]).collect();
    let (p1, p2) = stratified_train_test_split(&labels, 42)?;
    Ok((
        p1.iter().map(|&p| train[p]).collect(),
        p2.iter().map(|&p| train[p]).collect(),
    ))
}

/// One fold's nested split: the two halves of the training fold (data
/// indices) and the preliminary folds on the first half (positions within it).
fn nested_split(
    train: &[usize],
    strata: &[f64],
    n_folds: usize,
) -> Result<(Vec<usize>, Vec<usize>, Vec<Fold>)> {
    let (train1, train2) = half_split(train, strata)?;
    let labels1: Vec<f64> = train1.iter().map(|&i| strata[i]).collect();
    let n1 = train1.len();
    let folds = stratified_kfold(&labels1, n_folds)?
        .into_iter()
        .map(|test| {
            let mut in_test = vec![false; n1];
            for &i in &test {
                in_test[i] = true;
            }
            Fold {
                train: (0..n1).filter(|&i| !in_test[i]).collect(),
                test,
            }
        })
        .collect();
    Ok((train1, train2, folds))
}

/// Fit a fresh copy of `learner` on `rows` with `labels`; predict `x_pred`.
fn fit_proba(
    learner: &dyn Classifier,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
    labels: &[f64],
    x_pred: ArrayView2<'_, f64>,
) -> Result<Array1<f64>> {
    let mut m = learner.fresh();
    m.fit(x.select(Axis(0), rows).view(), ArrayView1::from(labels))?;
    m.predict_proba(x_pred)
}

/// DoubleML's preliminary IPW estimate: minimise `|mean score|` with Brent's
/// method from a bracket around `start`.
fn ipw_estimate(score: &mut dyn FnMut(f64) -> f64, start: f64, bounds: (f64, f64)) -> f64 {
    let (guess, _) = bracket_guess(score, start, bounds);
    brent_minimize(&mut |theta| score(theta).abs(), guess.0, guess.1)
}

/// Solve the non-linear score: bracket around `start`, then `brentq`.
fn solve_score(score: &mut dyn FnMut(f64) -> f64, start: f64, bounds: (f64, f64)) -> Result<f64> {
    let (guess, different) = bracket_guess(score, start, bounds);
    if !different {
        return Err(DmlError::InvalidArgument(
            "could not find a root of the quantile score (no sign change within the outcome range)"
                .into(),
        ));
    }
    brentq(score, guess.0, guess.1)
}

/// Default sample splits, stratified by `strata` codes.
fn draw_splits(splitting: &SampleSplitting, strata: &[f64]) -> Result<Vec<Vec<Fold>>> {
    let mut levels = strata.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    let codes: Vec<usize> = strata
        .iter()
        .map(|v| {
            levels
                .binary_search_by(|l| l.total_cmp(v))
                .expect("present")
        })
        .collect();
    splitting.draw_stratified(&codes)
}

/// Builder methods shared by the quantile models (settings and splitting).
macro_rules! quantile_api {
    () => {
        /// Treatment level `d` of the potential outcome (0 or 1, default 1).
        pub fn treatment(mut self, treatment: u8) -> Self {
            self.settings.treatment = f64::from(treatment);
            self
        }

        /// Quantile level `τ` in (0, 1) (default 0.5).
        pub fn quantile(mut self, quantile: f64) -> Self {
            self.settings.quantile = quantile;
            self
        }

        /// Propensity scores are clipped to `[t, 1 − t]` (default `t = 0.01`).
        pub fn clipping_threshold(mut self, t: f64) -> Self {
            self.settings.clipping_threshold = t;
            self
        }

        /// Normalise the inverse-probability weights (default `true`, as DoubleML).
        pub fn normalize_ipw(mut self, normalize: bool) -> Self {
            self.settings.normalize_ipw = normalize;
            self
        }

        /// Calibrate the propensity scores before clipping (default: none), as
        /// DoubleML's `ps_processor_config`.
        pub fn calibration(mut self, calibration: Calibration) -> Self {
            self.settings.calibration = calibration;
            self
        }
    };
}

/// External predictions of the single treatment in repetition `rep`.
fn external_ref(data: &DmlData, rep: usize) -> Option<ExternalRef<'_>> {
    data.external_predictions().map(|preds| ExternalRef {
        preds,
        treatment: 0,
        rep,
    })
}

/// The learners of [`Lpq`], in the order of its external predictions.
const LPQ_LEARNERS: [&str; 5] = [
    "ml_m_z",
    "ml_m_d_z0",
    "ml_m_d_z1",
    "ml_g_du_z0",
    "ml_g_du_z1",
];

/// Builder methods for the sample splitting of the non-linear models.
macro_rules! splitting_api {
    () => {
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
    };
}

// ---------------------------------------------------------------------------
// Potential quantile (PQ)
// ---------------------------------------------------------------------------

/// Potential quantile `θ` with `P(Y(d) ≤ θ) = τ`, as DoubleML's `DoubleMLPQ`.
///
/// `ml_g` is a classifier for `P(Y ≤ θ̃ | D = d, X)` at the preliminary
/// estimate `θ̃`, `ml_m` a classifier for the propensity `P(D = 1 | X)`.
/// Default sample splits are stratified by the treatment.
pub struct Pq {
    ml_g: Box<dyn Classifier>,
    ml_m: Box<dyn Classifier>,
    settings: Settings,
    splitting: SampleSplitting,
}

impl Pq {
    pub fn new(ml_g: impl Classifier + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    pub fn from_boxed(ml_g: Box<dyn Classifier>, ml_m: Box<dyn Classifier>) -> Self {
        Self {
            ml_g,
            ml_m,
            settings: Settings::new(),
            splitting: SampleSplitting::default(),
        }
    }

    quantile_api!();
    splitting_api!();

    pub fn fit(&self, data: &DmlData) -> Result<DmlFit> {
        check_data(data, false)?;
        let splits = draw_splits(&self.splitting, &data.treatment(0).to_vec())?;
        self.fit_with_folds(data, &splits)
    }

    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<DmlFit> {
        check_data(data, false)?;
        self.settings.check()?;
        // DoubleML carries the start value across repetitions: each repetition's
        // preliminary search starts at the previous one's mean IPW estimate.
        let mut start = None;
        let mut reps = Vec::with_capacity(splits.len());
        for (r, folds) in splits.iter().enumerate() {
            let rep = self.fit_rep(data, folds, start, r)?;
            start = Some(rep.next_start);
            reps.push(rep);
        }
        let name = format!(
            "PQ (treatment {}, quantile {})",
            self.settings.treatment, self.settings.quantile
        );
        Ok(assemble(name, data, splits[0].len(), reps))
    }

    fn fit_rep(
        &self,
        data: &DmlData,
        folds: &[Fold],
        start: Option<f64>,
        rep: usize,
    ) -> Result<RepFit> {
        let s = self.settings;
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let n = y.len();
        let ext = external_ref(data, rep);
        let (g_ext, m_ext) = (external_get(ext, n, "ml_g")?, external_get(ext, n, "ml_m")?);
        let bounds = min_max(y);
        let y_treat: Vec<f64> = (0..n)
            .filter(|&i| d[i] == s.treatment)
            .map(|i| y[i])
            .collect();
        let start = start.unwrap_or_else(|| np_quantile(&y_treat, s.quantile));
        let strata = d.to_vec();
        let (mut g, mut g_target, mut m) = (
            Array1::from_elem(n, f64::NAN),
            Array1::from_elem(n, f64::NAN),
            Array1::from_elem(n, f64::NAN),
        );
        let mut ipw_vec = Vec::with_capacity(folds.len());
        // With both nuisances supplied there is no preliminary estimate (DoubleML).
        let all_external = g_ext.is_some() && m_ext.is_some();
        for fold in folds.iter().filter(|_| !all_external) {
            let (train1, train2, prelim) = nested_split(&fold.train, &strata, folds.len())?;
            let (d1, y1) = (take(d, &train1), take(y, &train1));
            let x1 = x.select(Axis(0), &train1);
            let m1 = match &m_ext {
                Some(m_ext) => take(m_ext.view(), &train1),
                None => cross_fit_predict_proba(self.ml_m.as_ref(), x1.view(), d1.view(), &prelim)?,
            };
            let m1 = s.adjust(&m1, &d1, true, &prelim)?;
            let mut ipw_score = |theta: f64| {
                let terms: Vec<f64> = (0..d1.len())
                    .map(|i| {
                        f64::from(d1[i] == s.treatment) / m1[i] * f64::from(y1[i] <= theta)
                            - s.quantile
                    })
                    .collect();
                np_mean(&terms)
            };
            let ipw = ipw_estimate(&mut ipw_score, start, bounds);
            ipw_vec.push(ipw);

            let x_test = x.select(Axis(0), &fold.test);
            if g_ext.is_none() {
                let rows: Vec<usize> = train2
                    .iter()
                    .copied()
                    .filter(|&i| d[i] == s.treatment)
                    .collect();
                let labels: Vec<f64> = rows.iter().map(|&i| f64::from(y[i] <= ipw)).collect();
                let g_test = fit_proba(self.ml_g.as_ref(), x, &rows, &labels, x_test.view())?;
                for (k, &i) in fold.test.iter().enumerate() {
                    g[i] = g_test[k];
                    g_target[i] = f64::from(y[i] <= ipw);
                }
            }
            if m_ext.is_none() {
                let d_train: Vec<f64> = fold.train.iter().map(|&i| d[i]).collect();
                let m_test =
                    fit_proba(self.ml_m.as_ref(), x, &fold.train, &d_train, x_test.view())?;
                for (k, &i) in fold.test.iter().enumerate() {
                    m[i] = m_test[k];
                }
            }
        }
        let (g_external, m_external) = (g_ext.is_some(), m_ext.is_some());
        let g = g_ext.unwrap_or(g);
        let m = m_ext.unwrap_or(m);
        let d_owned = d.to_owned();
        let m_clipped = s.clip(&m, &d_owned, folds)?;
        let m_adj = s.adjust(&m, &d_owned, true, folds)?;
        let ind: Array1<f64> = d.mapv(|v| f64::from(v == s.treatment));
        let terms = |theta: f64| -> Array1<f64> {
            (0..n)
                .map(|i| ind[i] * (f64::from(y[i] <= theta) - g[i]) / m_adj[i] + g[i] - s.quantile)
                .collect()
        };
        let next_start = if all_external {
            start
        } else {
            np_mean(&ipw_vec)
        };
        let theta = solve_score(
            &mut |theta| np_mean(terms(theta).as_slice().expect("contiguous")),
            next_start,
            bounds,
        )?;
        let psi = terms(theta);
        let u: Vec<f64> = y.iter().map(|&v| v - theta).collect();
        let weights: Vec<f64> = (0..n).map(|i| ind[i] / m_adj[i]).collect();
        // DoubleML: J is the mean of the (constant) score derivative.
        let jacobian = np_mean(&vec![kde_at_zero(&u, &weights); n]);
        Ok(RepFit {
            next_start,
            theta,
            psi,
            jacobian,
            // Externally predicted learners have no out-of-sample loss.
            targets: [
                (!g_external).then(|| {
                    NuisanceTarget::classification("ml_g", g_target).only_where(|i| ind[i] == 1.0)
                }),
                (!m_external).then(|| NuisanceTarget::classification("ml_m", d_owned)),
            ]
            .into_iter()
            .flatten()
            .collect(),
            predictions: vec![("ml_g".into(), g), ("ml_m".into(), m_clipped)],
        })
    }
}

// ---------------------------------------------------------------------------
// Local potential quantile (LPQ)
// ---------------------------------------------------------------------------

/// Local potential quantile of compliers, with a binary instrument, as
/// DoubleML's `DoubleMLLPQ`. `ml_g` is a classifier for
/// `P(1{D = d}·1{Y ≤ θ̃} = 1 | Z = z, X)`, `ml_m` a classifier used for the
/// instrument propensity `P(Z = 1 | X)` and for `P(D = 1 | Z = z, X)`.
/// Default sample splits are stratified by treatment and instrument.
pub struct Lpq {
    ml_g: Box<dyn Classifier>,
    ml_m: Box<dyn Classifier>,
    settings: Settings,
    splitting: SampleSplitting,
}

impl Lpq {
    pub fn new(ml_g: impl Classifier + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    pub fn from_boxed(ml_g: Box<dyn Classifier>, ml_m: Box<dyn Classifier>) -> Self {
        Self {
            ml_g,
            ml_m,
            settings: Settings::new(),
            splitting: SampleSplitting::default(),
        }
    }

    quantile_api!();
    splitting_api!();

    fn strata(data: &DmlData) -> Vec<f64> {
        let (d, z) = (data.treatment(0), data.z().expect("checked"));
        d.iter().zip(z.iter()).map(|(&a, &b)| a + 2.0 * b).collect()
    }

    pub fn fit(&self, data: &DmlData) -> Result<DmlFit> {
        check_data(data, true)?;
        let splits = draw_splits(&self.splitting, &Self::strata(data))?;
        self.fit_with_folds(data, &splits)
    }

    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<DmlFit> {
        check_data(data, true)?;
        self.settings.check()?;
        // DoubleML carries the start value across repetitions: each repetition's
        // preliminary search starts at the previous one's mean IPW estimate.
        let mut start = None;
        let mut reps = Vec::with_capacity(splits.len());
        for (r, folds) in splits.iter().enumerate() {
            let rep = self.fit_rep(data, folds, start, r)?;
            start = Some(rep.next_start);
            reps.push(rep);
        }
        let name = format!(
            "LPQ (treatment {}, quantile {})",
            self.settings.treatment, self.settings.quantile
        );
        Ok(assemble(name, data, splits[0].len(), reps))
    }

    fn fit_rep(
        &self,
        data: &DmlData,
        folds: &[Fold],
        start: Option<f64>,
        rep: usize,
    ) -> Result<RepFit> {
        let s = self.settings;
        let (y, d, x, z) = (
            data.y(),
            data.treatment(0),
            data.x(),
            data.z().expect("checked"),
        );
        let n = y.len();
        let sign = 2.0 * s.treatment - 1.0;
        let bounds = min_max(y);
        let y_treat: Vec<f64> = (0..n)
            .filter(|&i| d[i] == s.treatment)
            .map(|i| y[i])
            .collect();
        let start = start.unwrap_or_else(|| np_quantile(&y_treat, s.quantile));
        let strata = Self::strata(data);
        let nan = || Array1::from_elem(n, f64::NAN);
        let (mut m_z, mut m_d_z0, mut m_d_z1, mut g0, mut g1) = (nan(), nan(), nan(), nan(), nan());
        let (mut g0_target, mut g1_target) = (nan(), nan());
        // DoubleML: either all five nuisances are supplied or none.
        let ext = external_ref(data, rep);
        let external = LPQ_LEARNERS
            .iter()
            .map(|&name| external_get(ext, n, name))
            .collect::<Result<Vec<_>>>()?;
        let n_external = external.iter().filter(|p| p.is_some()).count();
        if n_external != 0 && n_external != LPQ_LEARNERS.len() {
            return Err(DmlError::InvalidArgument(format!(
                "LPQ needs external predictions for all of {} or none",
                LPQ_LEARNERS.join(", ")
            )));
        }
        let all_external = n_external == LPQ_LEARNERS.len();
        let mut ipw_vec = Vec::with_capacity(folds.len());
        for fold in folds.iter().filter(|_| !all_external) {
            let (train1, train2, prelim) = nested_split(&fold.train, &strata, folds.len())?;
            let (d1, y1, z1) = (take(d, &train1), take(y, &train1), take(z, &train1));
            let x1 = x.select(Axis(0), &train1);
            let mz1 = cross_fit_predict_proba(self.ml_m.as_ref(), x1.view(), z1.view(), &prelim)?;
            let mz1 = s.adjust(&mz1, &z1, false, &prelim)?;
            // P(D = 1 | Z = z, X) fitted and predicted on the first half.
            let pos =
                |zv: f64| -> Vec<usize> { (0..train1.len()).filter(|&p| z1[p] == zv).collect() };
            let (pos0, pos1) = (pos(0.0), pos(1.0));
            let md0 = fit_proba(
                self.ml_m.as_ref(),
                x1.view(),
                &pos0,
                &pos0.iter().map(|&p| d1[p]).collect::<Vec<_>>(),
                x1.view(),
            )?;
            let md1 = fit_proba(
                self.ml_m.as_ref(),
                x1.view(),
                &pos1,
                &pos1.iter().map(|&p| d1[p]).collect::<Vec<_>>(),
                x1.view(),
            )?;
            let comp_terms: Vec<f64> = (0..train1.len())
                .map(|p| {
                    md1[p] - md0[p] + z1[p] / mz1[p] * (d1[p] - md1[p])
                        - (1.0 - z1[p]) / (1.0 - mz1[p]) * (d1[p] - md0[p])
                })
                .collect();
            let comp_prelim = np_mean(&comp_terms);
            let mut ipw_score = |theta: f64| {
                let terms: Vec<f64> = (0..train1.len())
                    .map(|p| {
                        let w =
                            sign * (z1[p] / mz1[p] - (1.0 - z1[p]) / (1.0 - mz1[p])) / comp_prelim;
                        let u = f64::from(d1[p] == s.treatment) * f64::from(y1[p] <= theta);
                        w * u + -s.quantile
                    })
                    .collect();
                np_mean(&terms)
            };
            let ipw = ipw_estimate(&mut ipw_score, start, bounds);
            ipw_vec.push(ipw);

            let x_test = x.select(Axis(0), &fold.test);
            let du = |i: usize| f64::from(d[i] == s.treatment) * f64::from(y[i] <= ipw);
            let rows2 = |zv: f64| -> Vec<usize> {
                train2.iter().copied().filter(|&i| z[i] == zv).collect()
            };
            let (r0, r1) = (rows2(0.0), rows2(1.0));
            let g0_test = fit_proba(
                self.ml_g.as_ref(),
                x,
                &r0,
                &r0.iter().map(|&i| du(i)).collect::<Vec<_>>(),
                x_test.view(),
            )?;
            let g1_test = fit_proba(
                self.ml_g.as_ref(),
                x,
                &r1,
                &r1.iter().map(|&i| du(i)).collect::<Vec<_>>(),
                x_test.view(),
            )?;
            // Final propensities on the whole training fold.
            let train = &fold.train;
            let mz_test = fit_proba(
                self.ml_m.as_ref(),
                x,
                train,
                &train.iter().map(|&i| z[i]).collect::<Vec<_>>(),
                x_test.view(),
            )?;
            let t0: Vec<usize> = train.iter().copied().filter(|&i| z[i] == 0.0).collect();
            let t1: Vec<usize> = train.iter().copied().filter(|&i| z[i] == 1.0).collect();
            let md0_test = fit_proba(
                self.ml_m.as_ref(),
                x,
                &t0,
                &t0.iter().map(|&i| d[i]).collect::<Vec<_>>(),
                x_test.view(),
            )?;
            let md1_test = fit_proba(
                self.ml_m.as_ref(),
                x,
                &t1,
                &t1.iter().map(|&i| d[i]).collect::<Vec<_>>(),
                x_test.view(),
            )?;
            for (k, &i) in fold.test.iter().enumerate() {
                g0[i] = g0_test[k];
                g1[i] = g1_test[k];
                m_z[i] = mz_test[k];
                m_d_z0[i] = md0_test[k];
                m_d_z1[i] = md1_test[k];
                if z[i] == 0.0 {
                    g0_target[i] = du(i);
                } else {
                    g1_target[i] = du(i);
                }
            }
        }
        if all_external {
            let mut it = external.into_iter().flatten();
            let mut next = || it.next().expect("all five supplied");
            (m_z, m_d_z0, m_d_z1, g0, g1) = (next(), next(), next(), next(), next());
        }
        let z_owned = z.to_owned();
        let m_z_clipped = s.clip(&m_z, &z_owned, folds)?;
        let m_z_adj = s.adjust(&m_z, &z_owned, false, folds)?;
        let comp_terms: Vec<f64> = (0..n)
            .map(|i| {
                m_d_z1[i] - m_d_z0[i] + z[i] / m_z_adj[i] * (d[i] - m_d_z1[i])
                    - (1.0 - z[i]) / (1.0 - m_z_adj[i]) * (d[i] - m_d_z0[i])
            })
            .collect();
        let comp_prob = np_mean(&comp_terms);
        let ind: Array1<f64> = d.mapv(|v| f64::from(v == s.treatment));
        let terms = |theta: f64| -> Array1<f64> {
            (0..n)
                .map(|i| {
                    let yi = ind[i] * f64::from(y[i] <= theta);
                    let score1 = g1[i] - g0[i];
                    let score2 = (z[i] / m_z_adj[i]) * (yi - g1[i]);
                    let score3 = (1.0 - z[i]) / (1.0 - m_z_adj[i]) * (yi - g0[i]);
                    sign * (score1 + score2 - score3) / comp_prob - s.quantile
                })
                .collect()
        };
        let next_start = if all_external {
            start
        } else {
            np_mean(&ipw_vec)
        };
        let theta = solve_score(
            &mut |theta| np_mean(terms(theta).as_slice().expect("contiguous")),
            next_start,
            bounds,
        )?;
        let psi = terms(theta);
        let u: Vec<f64> = y.iter().map(|&v| v - theta).collect();
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                sign * ((z[i] / m_z_adj[i]) - (1.0 - z[i]) / (1.0 - m_z_adj[i])) * ind[i]
                    / comp_prob
            })
            .collect();
        // DoubleML: J is the mean of the (constant) score derivative.
        let jacobian = np_mean(&vec![kde_at_zero(&u, &weights); n]);
        let d_owned = d.to_owned();
        Ok(RepFit {
            next_start,
            theta,
            psi,
            jacobian,
            // Externally predicted learners have no out-of-sample loss.
            targets: if all_external {
                Vec::new()
            } else {
                vec![
                    NuisanceTarget::classification("ml_m_z", z_owned.clone()),
                    NuisanceTarget::classification("ml_m_d_z0", d_owned.clone())
                        .only_where(|i| z[i] == 0.0),
                    NuisanceTarget::classification("ml_m_d_z1", d_owned)
                        .only_where(|i| z[i] == 1.0),
                    NuisanceTarget::classification("ml_g_du_z0", g0_target)
                        .only_where(|i| z[i] == 0.0),
                    NuisanceTarget::classification("ml_g_du_z1", g1_target)
                        .only_where(|i| z[i] == 1.0),
                ]
            },
            predictions: vec![
                ("ml_m_z".into(), m_z_clipped),
                ("ml_m_d_z0".into(), m_d_z0),
                ("ml_m_d_z1".into(), m_d_z1),
                ("ml_g_du_z0".into(), g0),
                ("ml_g_du_z1".into(), g1),
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Conditional value at risk (CVaR)
// ---------------------------------------------------------------------------

/// Conditional value at risk of the potential outcome,
/// `E[Y(d) | Y(d) ≥ q_τ(Y(d))]`, as DoubleML's `DoubleMLCVAR`. `ml_g` is a
/// regressor, `ml_m` a classifier for `P(D = 1 | X)`. The score is linear.
pub struct Cvar {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    settings: Settings,
    splitting: SampleSplitting,
}

impl Cvar {
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Box<dyn Classifier>) -> Self {
        Self {
            ml_g,
            ml_m,
            settings: Settings::new(),
            splitting: SampleSplitting::default(),
        }
    }

    quantile_api!();
    dml_model_api!();
}

impl LinearScoreModel for Cvar {
    fn name(&self) -> String {
        format!(
            "CVaR (treatment {}, quantile {})",
            self.settings.treatment, self.settings.quantile
        )
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        check_data(data, false)?;
        self.settings.check()
    }

    fn strata(&self, data: &DmlData) -> Option<Vec<usize>> {
        Some(data.treatment(0).iter().map(|&v| v as usize).collect())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let s = self.settings;
        let ScoreInput { y, d, x, .. } = *input;
        let n = y.len();
        let bounds = min_max(y);
        let y_treat: Vec<f64> = (0..n)
            .filter(|&i| d[i] == s.treatment)
            .map(|i| y[i])
            .collect();
        let q_treat = np_quantile(&y_treat, s.quantile);
        let upper: Vec<f64> = y_treat.iter().copied().filter(|&v| v >= q_treat).collect();
        let start = np_mean(&upper);
        let strata = d.to_vec();
        let (mut g, mut g_target, mut m) = (
            Array1::from_elem(n, f64::NAN),
            Array1::from_elem(n, f64::NAN),
            Array1::from_elem(n, f64::NAN),
        );
        let cvar_target = |yi: f64, pq: f64| pq.max((yi - s.quantile * pq) / (1.0 - s.quantile));
        let mut ipw_vec = Vec::with_capacity(folds.len());
        for fold in folds {
            let (train1, train2, prelim) = nested_split(&fold.train, &strata, folds.len())?;
            let (d1, y1) = (take(d, &train1), take(y, &train1));
            let x1 = x.select(Axis(0), &train1);
            let m1 = cross_fit_predict_proba(self.ml_m.as_ref(), x1.view(), d1.view(), &prelim)?;
            let m1 = s.adjust(&m1, &d1, true, &prelim)?;
            let mut ipw_score = |theta: f64| {
                let terms: Vec<f64> = (0..d1.len())
                    .map(|i| {
                        f64::from(d1[i] == s.treatment) / m1[i] * f64::from(y1[i] <= theta)
                            - s.quantile
                    })
                    .collect();
                np_mean(&terms)
            };
            let ipw = ipw_estimate(&mut ipw_score, start, bounds);
            ipw_vec.push(ipw);

            let x_test = x.select(Axis(0), &fold.test);
            let rows: Vec<usize> = train2
                .iter()
                .copied()
                .filter(|&i| d[i] == s.treatment)
                .collect();
            let targets: Array1<f64> = rows.iter().map(|&i| cvar_target(y[i], ipw)).collect();
            let mut model = self.ml_g.fresh();
            model.fit(x.select(Axis(0), &rows).view(), targets.view())?;
            let g_test = model.predict(x_test.view())?;
            let d_train: Vec<f64> = fold.train.iter().map(|&i| d[i]).collect();
            let m_test = fit_proba(self.ml_m.as_ref(), x, &fold.train, &d_train, x_test.view())?;
            for (k, &i) in fold.test.iter().enumerate() {
                g[i] = g_test[k];
                g_target[i] = cvar_target(y[i], ipw);
                m[i] = m_test[k];
            }
        }
        let d_owned = d.to_owned();
        let m_clipped = s.clip(&m, &d_owned, folds)?;
        let m_adj = s.adjust(&m, &d_owned, true, folds)?;
        // The potential quantile is approximated by the mean of the IPW estimates.
        let pq = np_mean(&ipw_vec);
        let ind: Array1<f64> = d.mapv(|v| f64::from(v == s.treatment));
        let psi_b: Array1<f64> = (0..n)
            .map(|i| ind[i] * (cvar_target(y[i], pq) - g[i]) / m_adj[i] + g[i])
            .collect();
        let psi_a = Array1::from_elem(n, -1.0);
        Ok(ScoreEstimate::new(
            LinearScore::new(psi_a, psi_b)?,
            vec![("ml_g", g), ("ml_m", m_clipped)],
        )
        .with_targets(vec![
            NuisanceTarget::regression("ml_g", g_target).only_where(|i| ind[i] == 1.0),
            NuisanceTarget::classification("ml_m", d_owned),
        ]))
    }
}

// ---------------------------------------------------------------------------
// Quantile treatment effects (QTE)
// ---------------------------------------------------------------------------

/// Outcome learner of a [`Qte`]: classifiers for quantiles, a regressor for
/// CVaRs.
enum QteKind {
    Pq(Box<dyn Classifier>),
    Lpq(Box<dyn Classifier>),
    Cvar(Box<dyn Regressor>),
}

/// Treatment effects on quantiles (or CVaRs), as DoubleML's `DoubleMLQTE`: for
/// every `τ`, the potential quantiles of treatment 1 and 0 are fitted on the
/// same sample splits and their difference is the effect, with joint
/// inference across all quantiles.
pub struct Qte {
    kind: QteKind,
    ml_m: Box<dyn Classifier>,
    quantiles: Vec<f64>,
    clipping_threshold: f64,
    calibration: Calibration,
    normalize_ipw: bool,
    splitting: SampleSplitting,
}

impl Qte {
    fn with_kind(kind: QteKind, ml_m: Box<dyn Classifier>, quantiles: Vec<f64>) -> Self {
        Self {
            kind,
            ml_m,
            quantiles,
            clipping_threshold: 1e-2,
            calibration: Calibration::None,
            normalize_ipw: true,
            splitting: SampleSplitting::default(),
        }
    }

    /// Effects on quantiles (potential quantiles, [`Pq`]).
    pub fn pq(
        ml_g: impl Classifier + 'static,
        ml_m: impl Classifier + 'static,
        quantiles: Vec<f64>,
    ) -> Self {
        Self::with_kind(QteKind::Pq(Box::new(ml_g)), Box::new(ml_m), quantiles)
    }

    /// Effects on compliers' quantiles (local potential quantiles, [`Lpq`]).
    pub fn lpq(
        ml_g: impl Classifier + 'static,
        ml_m: impl Classifier + 'static,
        quantiles: Vec<f64>,
    ) -> Self {
        Self::with_kind(QteKind::Lpq(Box::new(ml_g)), Box::new(ml_m), quantiles)
    }

    /// Effects on the conditional value at risk ([`Cvar`]).
    pub fn cvar(
        ml_g: impl Regressor + 'static,
        ml_m: impl Classifier + 'static,
        quantiles: Vec<f64>,
    ) -> Self {
        Self::with_kind(QteKind::Cvar(Box::new(ml_g)), Box::new(ml_m), quantiles)
    }

    /// Boxed learners, e.g. from Python: `score` is `"PQ"`, `"LPQ"` or `"CVaR"`;
    /// for `"CVaR"` pass `ml_g_regressor`, otherwise `ml_g_classifier`.
    pub fn from_boxed(
        score: &str,
        ml_g_classifier: Option<Box<dyn Classifier>>,
        ml_g_regressor: Option<Box<dyn Regressor>>,
        ml_m: Box<dyn Classifier>,
        quantiles: Vec<f64>,
    ) -> Result<Self> {
        let missing =
            || DmlError::InvalidArgument(format!("score {score:?} needs a different ml_g"));
        let kind = match score {
            "PQ" => QteKind::Pq(ml_g_classifier.ok_or_else(missing)?),
            "LPQ" => QteKind::Lpq(ml_g_classifier.ok_or_else(missing)?),
            "CVaR" => QteKind::Cvar(ml_g_regressor.ok_or_else(missing)?),
            other => {
                return Err(DmlError::InvalidArgument(format!(
                    "unknown QTE score {other:?} (use 'PQ', 'LPQ' or 'CVaR')"
                )))
            }
        };
        Ok(Self::with_kind(kind, ml_m, quantiles))
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

    /// Normalise the inverse-probability weights (default `true`).
    pub fn normalize_ipw(mut self, normalize: bool) -> Self {
        self.normalize_ipw = normalize;
        self
    }

    splitting_api!();

    /// Draw sample splits (stratified by the treatment) and fit.
    pub fn fit(&self, data: &DmlData) -> Result<QteFit> {
        check_data(data, matches!(self.kind, QteKind::Lpq(_)))?;
        let splits = draw_splits(&self.splitting, &data.treatment(0).to_vec())?;
        self.fit_with_folds(data, &splits)
    }

    /// Fit on given sample splits (one `Vec<Fold>` per repetition).
    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<QteFit> {
        if self.quantiles.is_empty() {
            return Err(DmlError::InvalidArgument(
                "at least one quantile is required".into(),
            ));
        }
        let mut fits = Vec::with_capacity(2 * self.quantiles.len());
        let mut names = Vec::with_capacity(2 * self.quantiles.len());
        for &q in &self.quantiles {
            for t in [0u8, 1u8] {
                let fit = match &self.kind {
                    QteKind::Pq(g) => Pq::from_boxed(g.fresh(), self.ml_m.fresh())
                        .treatment(t)
                        .quantile(q)
                        .clipping_threshold(self.clipping_threshold)
                        .calibration(self.calibration)
                        .normalize_ipw(self.normalize_ipw)
                        .fit_with_folds(data, splits)?,
                    QteKind::Lpq(g) => Lpq::from_boxed(g.fresh(), self.ml_m.fresh())
                        .treatment(t)
                        .quantile(q)
                        .clipping_threshold(self.clipping_threshold)
                        .calibration(self.calibration)
                        .normalize_ipw(self.normalize_ipw)
                        .fit_with_folds(data, splits)?,
                    QteKind::Cvar(g) => Cvar::from_boxed(g.fresh(), self.ml_m.fresh())
                        .treatment(t)
                        .quantile(q)
                        .clipping_threshold(self.clipping_threshold)
                        .calibration(self.calibration)
                        .normalize_ipw(self.normalize_ipw)
                        .fit_with_folds(data, splits)?,
                };
                fits.push(fit);
                names.push(format!("{q} (d={t})"));
            }
        }
        let mut potential = DmlFit::concat(&fits)?;
        potential.treatment_names = names;
        potential.model = match self.kind {
            QteKind::Pq(_) => "potential quantiles".into(),
            QteKind::Lpq(_) => "local potential quantiles".into(),
            QteKind::Cvar(_) => "potential CVaRs".into(),
        };
        let k = self.quantiles.len();
        let mut weights = Array2::zeros((k, 2 * k));
        for i in 0..k {
            weights[[i, 2 * i]] = -1.0;
            weights[[i, 2 * i + 1]] = 1.0;
        }
        let effect_names = self.quantiles.iter().map(|q| q.to_string()).collect();
        let effects = potential.combine("QTE".into(), effect_names, &weights);
        Ok(QteFit {
            quantiles: self.quantiles.clone(),
            potential,
            effects,
        })
    }
}

/// Fitted [`Qte`].
#[derive(Debug, Clone)]
pub struct QteFit {
    quantiles: Vec<f64>,
    potential: DmlFit,
    effects: DmlFit,
}

impl QteFit {
    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    /// The effect at every quantile (named by the quantile), with joint inference.
    pub fn effects(&self) -> &DmlFit {
        &self.effects
    }

    /// The potential quantiles (or CVaRs) of treatment 0 and 1 at every
    /// quantile, in the order `τ₁ (d=0), τ₁ (d=1), τ₂ (d=0), …`.
    pub fn potential(&self) -> &DmlFit {
        &self.potential
    }
}

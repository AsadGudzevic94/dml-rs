//! Difference-in-differences with **staggered adoption**: units start
//! treatment in different periods (cohorts), following Callaway & Sant'Anna
//! (2021) with the doubly robust DML scores of DoubleML's `DoubleMLDIDMulti`.
//!
//! For every cohort `g` and evaluation period `t` the group-time effect
//!
//! ```text
//! ATT(g, t) = E[Y_t(g) − Y_t(∞) | G = g]
//! ```
//!
//! is estimated by a two-period panel DID comparing cohort `g` with a control
//! group (never treated, or not yet treated by `t`) between a base period
//! `t_pre` and `t`. The per-unit scores of all `ATT(g, t)` share one index of
//! units, so joint inference and aggregation (by cohort, calendar time or
//! event time) account for their correlation.
//!
//! Works with panel data ([`PanelData`]: outcome changes per unit) and with
//! repeated cross-sections ([`RepeatedCrossSections`]: each row observed once).

use std::fmt;

use ndarray::{Array1, Array2, Array3, Axis};

use crate::crossfit::{Fold, KFold};
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};
use crate::model::did::{cs_did_score, panel_did_score, DidSettings};
use crate::model::DidScore;

/// Panel data for staggered DID: one row per unit, one column per period.
///
/// - `y`: outcomes, shape `(n_units, n_periods)`; `NaN` marks a period in
///   which a unit is not observed (each `ATT(g, t)` then uses the units
///   observed in both of its periods)
/// - `periods`: the period values, strictly increasing (e.g. years)
/// - `cohort`: the first treated period of each unit, or `f64::INFINITY`
///   (or `NaN`) for never-treated units; must be one of `periods`
/// - `x`: time-invariant covariates, shape `(n_units, p)`
#[derive(Debug, Clone)]
pub struct PanelData {
    y: Array2<f64>,
    periods: Vec<f64>,
    cohort: Array1<f64>,
    x: Array2<f64>,
}

impl PanelData {
    pub fn new(
        y: Array2<f64>,
        periods: Vec<f64>,
        cohort: Array1<f64>,
        x: Array2<f64>,
    ) -> Result<Self> {
        let (n, t) = y.dim();
        if periods.len() != t || t < 2 {
            return Err(DmlError::DimensionMismatch(format!(
                "y has {t} period columns but {} period values were given (need ≥ 2)",
                periods.len()
            )));
        }
        // `partial_cmp` also rejects NaN periods, which `<` would let through.
        if periods
            .windows(2)
            .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
        {
            return Err(DmlError::InvalidArgument(
                "periods must be strictly increasing".into(),
            ));
        }
        if cohort.len() != n || x.nrows() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "y has {n} units, cohort has {}, x has {}",
                cohort.len(),
                x.nrows()
            )));
        }
        let cohort = cohort.mapv(|g| if g.is_nan() { f64::INFINITY } else { g });
        if let Some(bad) = cohort
            .iter()
            .find(|&&g| g.is_finite() && !periods.contains(&g))
        {
            return Err(DmlError::InvalidArgument(format!(
                "cohort {bad} is not one of the periods"
            )));
        }
        if !x.iter().all(|v| v.is_finite()) || !y.iter().all(|v| v.is_finite() || v.is_nan()) {
            return Err(DmlError::InvalidArgument(
                "covariates must be finite and outcomes finite or NaN".into(),
            ));
        }
        Ok(Self {
            y,
            periods,
            cohort,
            x,
        })
    }

    /// Build from long format (one row per unit and period), as DoubleML's
    /// `DoubleMLPanelData`. Units are ordered by id; cohort and covariates are
    /// taken from each unit's first row and must not vary within a unit.
    pub fn from_long(
        id: &[i64],
        time: &[f64],
        y: &[f64],
        cohort: &[f64],
        x: &Array2<f64>,
    ) -> Result<Self> {
        let rows = id.len();
        if time.len() != rows || y.len() != rows || cohort.len() != rows || x.nrows() != rows {
            return Err(DmlError::DimensionMismatch(
                "id, time, y, cohort and x must have the same number of rows".into(),
            ));
        }
        let mut ids: Vec<i64> = id.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let mut periods: Vec<f64> = time.to_vec();
        periods.sort_by(f64::total_cmp);
        periods.dedup();
        let (n, t, p) = (ids.len(), periods.len(), x.ncols());
        let mut wide = Array2::from_elem((n, t), f64::NAN);
        let mut g = Array1::from_elem(n, f64::NAN);
        let mut xs = Array2::zeros((n, p));
        let mut seen = vec![false; n];
        for r in 0..rows {
            let u = ids.binary_search(&id[r]).expect("id collected above");
            let c = periods
                .binary_search_by(|v| v.total_cmp(&time[r]))
                .expect("time collected above");
            if !wide[[u, c]].is_nan() {
                return Err(DmlError::InvalidArgument(format!(
                    "unit {} appears twice in period {}",
                    id[r], time[r]
                )));
            }
            wide[[u, c]] = y[r];
            let gr = if cohort[r].is_nan() {
                f64::INFINITY
            } else {
                cohort[r]
            };
            if !seen[u] {
                seen[u] = true;
                g[u] = gr;
                xs.row_mut(u).assign(&x.row(r));
            } else if g[u] != gr || xs.row(u) != x.row(r) {
                return Err(DmlError::InvalidArgument(format!(
                    "cohort and covariates must be constant within unit {}",
                    id[r]
                )));
            }
        }
        Self::new(wide, periods, g, xs)
    }

    pub fn n_units(&self) -> usize {
        self.y.nrows()
    }

    pub fn periods(&self) -> &[f64] {
        &self.periods
    }

    /// Distinct treated cohorts, ascending.
    pub fn cohorts(&self) -> Vec<f64> {
        let mut g: Vec<f64> = self
            .cohort
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        g.sort_by(f64::total_cmp);
        g.dedup();
        g
    }
}

impl PanelData {
    /// Number of observed outcomes per unit (non-`NaN` cells).
    fn observations(&self, u: usize) -> usize {
        self.y.row(u).iter().filter(|v| !v.is_nan()).count()
    }
}

/// Repeated cross-sections for staggered DID: every row is a separate
/// observation in one period (DoubleML's `DoubleMLDIDMulti(panel=False)`).
///
/// - `y`: outcome of each row
/// - `time`: the period of each row
/// - `cohort`: the first treated period of the row's group, or
///   `f64::INFINITY` / `NaN` for never treated; must be one of the periods
/// - `x`: covariates of each row, shape `(n_rows, p)`
#[derive(Debug, Clone)]
pub struct RepeatedCrossSections {
    y: Array1<f64>,
    time: Array1<f64>,
    cohort: Array1<f64>,
    x: Array2<f64>,
    periods: Vec<f64>,
}

impl RepeatedCrossSections {
    pub fn new(
        y: Array1<f64>,
        time: Array1<f64>,
        cohort: Array1<f64>,
        x: Array2<f64>,
    ) -> Result<Self> {
        let n = y.len();
        if time.len() != n || cohort.len() != n || x.nrows() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "y has {n} rows, time {}, cohort {}, x {}",
                time.len(),
                cohort.len(),
                x.nrows()
            )));
        }
        if !y
            .iter()
            .chain(time.iter())
            .chain(x.iter())
            .all(|v| v.is_finite())
        {
            return Err(DmlError::InvalidArgument(
                "outcomes, periods and covariates must be finite".into(),
            ));
        }
        let mut periods: Vec<f64> = time.to_vec();
        periods.sort_by(f64::total_cmp);
        periods.dedup();
        if periods.len() < 2 {
            return Err(DmlError::InvalidArgument(
                "need at least two periods".into(),
            ));
        }
        let cohort = cohort.mapv(|g| if g.is_nan() { f64::INFINITY } else { g });
        if let Some(bad) = cohort
            .iter()
            .find(|&&g| g.is_finite() && !periods.contains(&g))
        {
            return Err(DmlError::InvalidArgument(format!(
                "cohort {bad} is not one of the periods"
            )));
        }
        Ok(Self {
            y,
            time,
            cohort,
            x,
            periods,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.y.len()
    }

    pub fn periods(&self) -> &[f64] {
        &self.periods
    }
}

/// Data for [`StaggeredDid::fit`]: a panel or repeated cross-sections.
/// `&PanelData` and `&RepeatedCrossSections` convert into it automatically.
#[derive(Debug, Clone, Copy)]
pub enum StaggeredInput<'a> {
    Panel(&'a PanelData),
    CrossSections(&'a RepeatedCrossSections),
}

impl<'a> From<&'a PanelData> for StaggeredInput<'a> {
    fn from(data: &'a PanelData) -> Self {
        StaggeredInput::Panel(data)
    }
}

impl<'a> From<&'a RepeatedCrossSections> for StaggeredInput<'a> {
    fn from(data: &'a RepeatedCrossSections) -> Self {
        StaggeredInput::CrossSections(data)
    }
}

impl StaggeredInput<'_> {
    /// Observations: units for panels, rows for cross-sections.
    fn n_obs(&self) -> usize {
        match self {
            StaggeredInput::Panel(p) => p.n_units(),
            StaggeredInput::CrossSections(c) => c.n_rows(),
        }
    }

    fn periods(&self) -> &[f64] {
        match self {
            StaggeredInput::Panel(p) => p.periods(),
            StaggeredInput::CrossSections(c) => c.periods(),
        }
    }

    fn cohort_values(&self) -> &Array1<f64> {
        match self {
            StaggeredInput::Panel(p) => &p.cohort,
            StaggeredInput::CrossSections(c) => &c.cohort,
        }
    }

    /// Distinct treated cohorts, ascending.
    fn cohorts(&self) -> Vec<f64> {
        let mut g: Vec<f64> = self
            .cohort_values()
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        g.sort_by(f64::total_cmp);
        g.dedup();
        g
    }

    /// Share of all observed outcomes (panel cells or rows) in each cohort,
    /// as DoubleML's aggregation weights.
    fn cohort_shares(&self) -> Vec<f64> {
        let cohorts = self.cohorts();
        let mut counts = vec![0.0; cohorts.len()];
        let mut total = 0.0;
        for (u, &g) in self.cohort_values().iter().enumerate() {
            let w = match self {
                StaggeredInput::Panel(p) => p.observations(u) as f64,
                StaggeredInput::CrossSections(_) => 1.0,
            };
            total += w;
            if let Some(i) = cohorts.iter().position(|&c| c == g) {
                counts[i] += w;
            }
        }
        counts.into_iter().map(|c| c / total).collect()
    }
}

/// Which units serve as controls for cohort `g` in period `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ControlGroup {
    /// Units that are never treated.
    #[default]
    NeverTreated,
    /// Never-treated units plus cohorts not yet treated by `t` (allowing for
    /// anticipation).
    NotYetTreated,
}

/// Which `(g, t_pre, t_eval)` combinations to estimate (as DoubleML).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GtCombinations {
    /// Post-treatment periods against the last pre-treatment period; pre-treatment
    /// periods against the period before ("varying base", Callaway & Sant'Anna).
    #[default]
    Standard,
    /// Every valid base period for every evaluation period.
    All,
    /// Every period against the last pre-treatment period ("universal base").
    Universal,
}

/// One group-time combination: cohort, base period and evaluation period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupTime {
    pub cohort: f64,
    pub t_pre: f64,
    pub t_eval: f64,
}

impl GroupTime {
    /// `t_eval ≥ cohort`.
    pub fn is_post_treatment(&self) -> bool {
        self.t_eval >= self.cohort
    }

    fn label(&self) -> String {
        format!("ATT({},{},{})", self.cohort, self.t_pre, self.t_eval)
    }
}

/// Staggered-adoption DID estimator (DoubleML's `DoubleMLDIDMulti` for panels).
///
/// `ml_g` learns the outcome change of the control group given `X`, `ml_m` the
/// propensity of belonging to cohort `g` (observational score only).
pub struct StaggeredDid {
    ml_g: Box<dyn Regressor>,
    ml_m: Option<Box<dyn Classifier>>,
    score: DidScore,
    in_sample_normalization: bool,
    clipping_threshold: f64,
    control_group: ControlGroup,
    anticipation_periods: usize,
    gt_combinations: GtCombinations,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
}

impl StaggeredDid {
    /// Observational design (doubly robust score).
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Some(Box::new(ml_m)))
    }

    /// Randomised design: only the outcome model is needed.
    pub fn experimental(ml_g: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), None).score(DidScore::Experimental)
    }

    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Option<Box<dyn Classifier>>) -> Self {
        Self {
            ml_g,
            ml_m,
            score: DidScore::default(),
            in_sample_normalization: true,
            clipping_threshold: 1e-2,
            control_group: ControlGroup::default(),
            anticipation_periods: 0,
            gt_combinations: GtCombinations::default(),
            n_folds: 5,
            n_rep: 1,
            seed: 0,
        }
    }

    pub fn score(mut self, score: DidScore) -> Self {
        self.score = score;
        self
    }

    /// Normalise the weights by their sample means (default `true`).
    pub fn in_sample_normalization(mut self, normalize: bool) -> Self {
        self.in_sample_normalization = normalize;
        self
    }

    /// Propensities are clipped to `[t, 1 − t]` (default 0.01).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }

    pub fn control_group(mut self, control_group: ControlGroup) -> Self {
        self.control_group = control_group;
        self
    }

    /// Periods before treatment in which units may already react (default 0).
    pub fn anticipation_periods(mut self, periods: usize) -> Self {
        self.anticipation_periods = periods;
        self
    }

    pub fn gt_combinations(mut self, setting: GtCombinations) -> Self {
        self.gt_combinations = setting;
        self
    }

    /// Number of cross-fitting folds (default 5); folds are stratified by
    /// cohort membership within every group-time sample.
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.n_folds = n_folds;
        self
    }

    /// Repetitions of the sample split (default 1), aggregated by the median.
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.n_rep = n_rep;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The group-time combinations this estimator evaluates on `data`.
    pub fn group_times<'a>(&self, data: impl Into<StaggeredInput<'a>>) -> Result<Vec<GroupTime>> {
        let data = data.into();
        construct_group_times(
            self.gt_combinations,
            &data.cohorts(),
            data.periods(),
            self.anticipation_periods,
        )
    }

    /// Fit every `ATT(g, t)` with seeded, stratified folds, on a
    /// [`PanelData`] or [`RepeatedCrossSections`].
    pub fn fit<'a>(&self, data: impl Into<StaggeredInput<'a>>) -> Result<StaggeredDidFit> {
        self.fit_impl(data.into(), None)
    }

    /// Fit with caller-supplied folds: `folds[i][r]` are the splits of
    /// repetition `r` for the `i`-th group-time combination, indexing that
    /// combination's sample in data order (panels: its cohort and control
    /// units observed in both periods; cross-sections: its cohort and control
    /// rows from the two periods).
    pub fn fit_with_folds<'a>(
        &self,
        data: impl Into<StaggeredInput<'a>>,
        folds: &[Vec<Vec<Fold>>],
    ) -> Result<StaggeredDidFit> {
        self.fit_impl(data.into(), Some(folds))
    }

    fn fit_impl(
        &self,
        data: StaggeredInput<'_>,
        folds: Option<&[Vec<Vec<Fold>>]>,
    ) -> Result<StaggeredDidFit> {
        if self.score == DidScore::Observational && self.ml_m.is_none() {
            return Err(DmlError::InvalidArgument(
                "the observational DID score needs a propensity learner ml_m".into(),
            ));
        }
        if self.n_rep == 0 {
            return Err(DmlError::InvalidArgument("n_rep must be at least 1".into()));
        }
        let gts = self.group_times(data)?;
        if let Some(f) = folds {
            if f.len() != gts.len()
                || f.iter()
                    .any(|reps| reps.len() != f[0].len() || reps.is_empty())
            {
                return Err(DmlError::DimensionMismatch(format!(
                    "need folds for each of the {} group-time combinations, with the same number of repetitions",
                    gts.len()
                )));
            }
        }
        let n_rep = folds.map_or(self.n_rep, |f| f[0].len());
        let n = data.n_obs();
        let k = gts.len();
        let settings = DidSettings {
            ml_g: self.ml_g.as_ref(),
            ml_m: self.ml_m.as_deref(),
            score: self.score,
            in_sample_normalization: self.in_sample_normalization,
            clipping_threshold: self.clipping_threshold,
        };

        let mut all_coef = Array2::<f64>::zeros((k, n_rep));
        let mut psi = Array3::<f64>::zeros((n, k, n_rep));
        let mut psi_a = Array3::<f64>::zeros((n, k, n_rep));
        let mut psi_b = Array3::<f64>::zeros((n, k, n_rep));
        let mut predictions: Vec<(String, Array3<f64>)> = Vec::new();
        let mut n_folds_used = self.n_folds;

        for (i, gt) in gts.iter().enumerate() {
            let sample = self.group_time_sample(data, gt)?;
            for r in 0..n_rep {
                let rep_folds = match folds {
                    Some(f) => f[i][r].clone(),
                    None => {
                        let seed = self
                            .seed
                            .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            .wrapping_add((r as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
                        KFold::new(self.n_folds, seed)?
                            .split_stratified(&sample.strata)
                            .map_err(|e| context(gt, e))?
                    }
                };
                n_folds_used = rep_folds.len();
                let est = match &sample.t {
                    None => panel_did_score(
                        &settings,
                        sample.x.view(),
                        sample.y.view(),
                        sample.d.view(),
                        &rep_folds,
                    ),
                    Some(t) => cs_did_score(
                        &settings,
                        sample.x.view(),
                        sample.y.view(),
                        sample.d.view(),
                        t.view(),
                        &rep_folds,
                    ),
                }
                .map_err(|e| context(gt, e))?;
                let sol = est.score.solve().map_err(|e| context(gt, e))?;
                all_coef[[i, r]] = sol.theta;
                for (pos, &obs) in sample.positions.iter().enumerate() {
                    psi[[obs, i, r]] = sol.psi[pos];
                    psi_a[[obs, i, r]] = est.score.psi_a[pos];
                    psi_b[[obs, i, r]] = est.score.psi_b[pos];
                }
                for (name, pred) in est.predictions {
                    let slot = match predictions.iter().position(|(nm, _)| *nm == name) {
                        Some(s) => s,
                        None => {
                            predictions.push((
                                name.to_string(),
                                Array3::from_elem((n, n_rep, k), f64::NAN),
                            ));
                            predictions.len() - 1
                        }
                    };
                    for (pos, &obs) in sample.positions.iter().enumerate() {
                        predictions[slot].1[[obs, r, i]] = pred[pos];
                    }
                }
            }
        }

        let model = format!(
            "Staggered DID {} ({}, {} control group)",
            match data {
                StaggeredInput::Panel(_) => "panel",
                StaggeredInput::CrossSections(_) => "repeated cross-sections",
            },
            match self.score {
                DidScore::Observational => "observational",
                DidScore::Experimental => "experimental",
            },
            match self.control_group {
                ControlGroup::NeverTreated => "never-treated",
                ControlGroup::NotYetTreated => "not-yet-treated",
            }
        );
        let names = gts.iter().map(GroupTime::label).collect();
        let fit = DmlFit::from_scores(
            model,
            names,
            n_folds_used,
            all_coef,
            psi,
            psi_a,
            psi_b,
            predictions,
        );
        Ok(StaggeredDidFit {
            group_times: gts,
            cohorts: data.cohorts(),
            cohort_share: data.cohort_shares(),
            fit,
        })
    }

    /// Is a member of `g_obs`'s cohort a control for `gt`?
    fn is_control(&self, g_obs: f64, gt: &GroupTime, max_g: f64) -> bool {
        match self.control_group {
            ControlGroup::NeverTreated => g_obs.is_infinite(),
            ControlGroup::NotYetTreated => {
                g_obs.is_infinite() || (g_obs > max_g && g_obs != gt.cohort)
            }
        }
    }

    /// Observations used for one group-time combination.
    fn group_time_sample(
        &self,
        data: StaggeredInput<'_>,
        gt: &GroupTime,
    ) -> Result<GroupTimeSample> {
        let periods = data.periods();
        let index = |v: f64| {
            periods
                .iter()
                .position(|&p| p == v)
                .expect("period of this data")
        };
        let (i_pre, i_eval) = (index(gt.t_pre), index(gt.t_eval));
        // Controls must not be treated (or anticipating) by the evaluation period.
        let max_g = periods[(i_eval + self.anticipation_periods).min(periods.len() - 1)];
        let sample = match data {
            StaggeredInput::Panel(p) => {
                let mut positions = Vec::new();
                let mut d = Vec::new();
                for u in 0..p.n_units() {
                    let g = p.cohort[u];
                    let treated = g == gt.cohort;
                    let observed = !p.y[[u, i_pre]].is_nan() && !p.y[[u, i_eval]].is_nan();
                    if (treated || self.is_control(g, gt, max_g)) && observed {
                        positions.push(u);
                        d.push(f64::from(treated));
                    }
                }
                let y = positions
                    .iter()
                    .map(|&u| p.y[[u, i_eval]] - p.y[[u, i_pre]])
                    .collect();
                GroupTimeSample {
                    x: p.x.select(Axis(0), &positions),
                    strata: d.iter().map(|&v| v as usize).collect(),
                    y,
                    d: Array1::from(d),
                    t: None,
                    positions,
                }
            }
            StaggeredInput::CrossSections(c) => {
                let mut positions = Vec::new();
                let (mut d, mut t) = (Vec::new(), Vec::new());
                for r in 0..c.n_rows() {
                    let (g, time) = (c.cohort[r], c.time[r]);
                    let treated = g == gt.cohort;
                    let in_periods = time == gt.t_pre || time == gt.t_eval;
                    if in_periods && (treated || self.is_control(g, gt, max_g)) {
                        positions.push(r);
                        d.push(f64::from(treated));
                        t.push(f64::from(time == gt.t_eval));
                    }
                }
                // Four strata: group × period, so every fold covers every cell.
                let strata = d
                    .iter()
                    .zip(&t)
                    .map(|(&di, &ti)| di as usize + 2 * ti as usize)
                    .collect();
                GroupTimeSample {
                    x: c.x.select(Axis(0), &positions),
                    y: positions.iter().map(|&r| c.y[r]).collect(),
                    strata,
                    d: Array1::from(d),
                    t: Some(Array1::from(t)),
                    positions,
                }
            }
        };
        let n_treated = sample.d.iter().filter(|&&v| v == 1.0).count();
        if n_treated == 0 || n_treated == sample.d.len() {
            return Err(DmlError::InvalidArgument(format!(
                "{}: need both cohort and control observations in periods {} and {}",
                gt.label(),
                gt.t_pre,
                gt.t_eval
            )));
        }
        Ok(sample)
    }
}

/// The observations of one group-time combination: positions in the data,
/// outcome (panel: change; cross-section: level), cohort indicator, optional
/// post-period indicator (cross-sections), covariates and fold strata.
struct GroupTimeSample {
    positions: Vec<usize>,
    y: Array1<f64>,
    d: Array1<f64>,
    t: Option<Array1<f64>>,
    x: Array2<f64>,
    strata: Vec<usize>,
}

fn context(gt: &GroupTime, err: DmlError) -> DmlError {
    match err {
        DmlError::InvalidArgument(m) => DmlError::InvalidArgument(format!("{}: {m}", gt.label())),
        other => other,
    }
}

/// DoubleML's `_construct_gt_combinations`.
fn construct_group_times(
    setting: GtCombinations,
    cohorts: &[f64],
    periods: &[f64],
    anticipation: usize,
) -> Result<Vec<GroupTime>> {
    let first_eval = anticipation + 1;
    let mut out = Vec::new();
    for &g in cohorts {
        let before: Vec<f64> = periods.iter().copied().filter(|&t| t < g).collect();
        if before.len() <= anticipation {
            continue; // no usable pre-treatment period
        }
        let base = before[before.len() - first_eval];
        match setting {
            GtCombinations::Standard => {
                for (i, &t_eval) in periods.iter().enumerate().skip(first_eval) {
                    let t_pre = periods[i - first_eval].min(base);
                    out.push(GroupTime {
                        cohort: g,
                        t_pre,
                        t_eval,
                    });
                }
            }
            GtCombinations::All => {
                for &t_eval in &periods[first_eval..] {
                    let candidates: Vec<f64> = periods
                        .iter()
                        .copied()
                        .filter(|&t| t <= g.min(t_eval))
                        .collect();
                    let keep = candidates.len().saturating_sub(first_eval);
                    for &t_pre in &candidates[..keep] {
                        out.push(GroupTime {
                            cohort: g,
                            t_pre,
                            t_eval,
                        });
                    }
                }
            }
            GtCombinations::Universal => {
                for &t_eval in periods.iter().filter(|&&t| t != base) {
                    out.push(GroupTime {
                        cohort: g,
                        t_pre: base,
                        t_eval,
                    });
                }
            }
        }
    }
    if out.is_empty() {
        return Err(DmlError::InvalidArgument(
            "no valid group-time combinations: every cohort needs a pre-treatment period (after anticipation)"
                .into(),
        ));
    }
    Ok(out)
}

/// How [`StaggeredDidFit::aggregate`] combines the group-time effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// One effect per cohort (average of its post-treatment `ATT(g, t)`);
    /// overall = cohort-size weighted average.
    Group,
    /// One effect per calendar period (cohort-size weighted across cohorts
    /// treated by then); overall = simple average over periods.
    Time,
    /// One effect per event time `e = t − g`, pre-treatment included
    /// (cohort-size weighted); overall = simple average over `e ≥ 0`.
    EventStudy,
}

/// All group-time effects of a staggered DID, with joint inference.
#[derive(Debug, Clone)]
pub struct StaggeredDidFit {
    group_times: Vec<GroupTime>,
    cohorts: Vec<f64>,
    /// Share of all units in each treated cohort.
    cohort_share: Vec<f64>,
    fit: DmlFit,
}

impl StaggeredDidFit {
    /// The `(g, t_pre, t_eval)` combinations, in the order of the estimates.
    pub fn group_times(&self) -> &[GroupTime] {
        &self.group_times
    }

    /// The `ATT(g, t)` estimates as a [`DmlFit`]: coefficients, standard
    /// errors, confidence intervals, and the multiplier bootstrap for
    /// simultaneous bands over all group-time effects.
    pub fn att(&self) -> &DmlFit {
        &self.fit
    }

    fn share_of(&self, g: f64) -> f64 {
        let i = self
            .cohorts
            .iter()
            .position(|&c| c == g)
            .expect("cohort of this fit");
        self.cohort_share[i]
    }

    /// Aggregate the group-time effects (DoubleML's `aggregate`).
    pub fn aggregate(&self, how: Aggregation) -> Result<DidAggregation> {
        let k = self.group_times.len();
        let post: Vec<bool> = self
            .group_times
            .iter()
            .map(GroupTime::is_post_treatment)
            .collect();
        let mut names: Vec<String> = Vec::new();
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let unique = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.dedup();
            v
        };
        let overall: Vec<f64> = match how {
            Aggregation::Group => {
                let groups = unique(
                    (0..k)
                        .filter(|&i| post[i])
                        .map(|i| self.group_times[i].cohort)
                        .collect(),
                );
                for &g in &groups {
                    let members: Vec<usize> = (0..k)
                        .filter(|&i| post[i] && self.group_times[i].cohort == g)
                        .collect();
                    let mut w = vec![0.0; k];
                    for &i in &members {
                        w[i] = 1.0 / members.len() as f64;
                    }
                    names.push(format!("{g}"));
                    rows.push(w);
                }
                let shares: Vec<f64> = groups.iter().map(|&g| self.share_of(g)).collect();
                let total: f64 = shares.iter().sum();
                shares.iter().map(|s| s / total).collect()
            }
            Aggregation::Time => {
                let times = unique(
                    (0..k)
                        .filter(|&i| post[i])
                        .map(|i| self.group_times[i].t_eval)
                        .collect(),
                );
                for &t in &times {
                    rows.push(self.share_weights(|i| post[i] && self.group_times[i].t_eval == t));
                    names.push(format!("{t}"));
                }
                vec![1.0 / times.len() as f64; times.len()]
            }
            Aggregation::EventStudy => {
                let event = |i: usize| self.group_times[i].t_eval - self.group_times[i].cohort;
                let events = unique((0..k).map(event).collect());
                for &e in &events {
                    rows.push(self.share_weights(|i| event(i) == e));
                    names.push(format!("{e}"));
                }
                let n_post = events.iter().filter(|&&e| e >= 0.0).count();
                events
                    .iter()
                    .map(|&e| if e >= 0.0 { 1.0 / n_post as f64 } else { 0.0 })
                    .collect()
            }
        };
        if rows.is_empty() {
            return Err(DmlError::InvalidArgument(
                "no group-time effects to aggregate (no post-treatment periods?)".into(),
            ));
        }
        let weights =
            Array2::from_shape_vec((rows.len(), k), rows.concat()).expect("rectangular weights");
        let label = match how {
            Aggregation::Group => "group",
            Aggregation::Time => "time",
            Aggregation::EventStudy => "event study",
        };
        let effects = self.fit.linear_combination(
            format!("{} — {label} aggregation", self.fit.model_name()),
            names,
            &weights,
        );
        let overall_weights = Array1::from(overall);
        let overall_fit = effects.linear_combination(
            format!("{} — overall ({label})", self.fit.model_name()),
            vec!["overall".to_string()],
            &overall_weights.clone().insert_axis(Axis(0)),
        );
        Ok(DidAggregation {
            aggregation: how,
            weights,
            overall_weights,
            effects,
            overall: overall_fit,
        })
    }

    /// Cohort-share weights over the selected group-time effects, normalised to one.
    fn share_weights(&self, selected: impl Fn(usize) -> bool) -> Vec<f64> {
        let mut w: Vec<f64> = (0..self.group_times.len())
            .map(|i| {
                if selected(i) {
                    self.share_of(self.group_times[i].cohort)
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = w.iter().sum();
        for v in &mut w {
            *v /= total;
        }
        w
    }
}

impl fmt::Display for StaggeredDidFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fit.fmt(f)
    }
}

/// Aggregated staggered-DID effects and their overall summary.
#[derive(Debug, Clone)]
pub struct DidAggregation {
    aggregation: Aggregation,
    weights: Array2<f64>,
    overall_weights: Array1<f64>,
    effects: DmlFit,
    overall: DmlFit,
}

impl DidAggregation {
    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }

    /// Aggregated effects (one per cohort, period or event time) with standard
    /// errors, intervals and bootstrap, as a [`DmlFit`].
    pub fn effects(&self) -> &DmlFit {
        &self.effects
    }

    /// The single overall effect.
    pub fn overall(&self) -> &DmlFit {
        &self.overall
    }

    /// Weights on the group-time effects, shape `(n_effects, n_group_times)`.
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }

    /// Weights of the aggregated effects in the overall effect.
    pub fn overall_weights(&self) -> &Array1<f64> {
        &self.overall_weights
    }
}

impl fmt::Display for DidAggregation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.effects)?;
        write!(f, "{}", self.overall)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gt(g: f64, a: f64, b: f64) -> GroupTime {
        GroupTime {
            cohort: g,
            t_pre: a,
            t_eval: b,
        }
    }

    #[test]
    fn standard_combinations_match_doubleml() {
        // DoubleML: g = 2, t = 0..4 → (2,0,1), (2,1,2), (2,1,3), (2,1,4).
        let combos = construct_group_times(
            GtCombinations::Standard,
            &[2.0],
            &[0.0, 1.0, 2.0, 3.0, 4.0],
            0,
        )
        .unwrap();
        assert_eq!(
            combos,
            vec![
                gt(2.0, 0.0, 1.0),
                gt(2.0, 1.0, 2.0),
                gt(2.0, 1.0, 3.0),
                gt(2.0, 1.0, 4.0)
            ]
        );
        let universal =
            construct_group_times(GtCombinations::Universal, &[2.0], &[0.0, 1.0, 2.0, 3.0], 0)
                .unwrap();
        assert_eq!(
            universal,
            vec![gt(2.0, 1.0, 0.0), gt(2.0, 1.0, 2.0), gt(2.0, 1.0, 3.0)]
        );
        // A cohort treated in the first period has no pre-period and is skipped.
        assert!(construct_group_times(GtCombinations::Standard, &[0.0], &[0.0, 1.0], 0).is_err());
        // Anticipation of one period moves the base back by one.
        let antic =
            construct_group_times(GtCombinations::Standard, &[3.0], &[0.0, 1.0, 2.0, 3.0], 1)
                .unwrap();
        assert_eq!(antic, vec![gt(3.0, 0.0, 2.0), gt(3.0, 1.0, 3.0)]);
    }

    #[test]
    fn long_format_round_trip() {
        let id = [2, 1, 2, 1];
        let time = [0.0, 0.0, 1.0, 1.0];
        let y = [20.0, 10.0, 21.0, 11.0];
        let cohort = [1.0, f64::INFINITY, 1.0, f64::INFINITY];
        let x = Array2::from_shape_vec((4, 1), vec![0.2, 0.1, 0.2, 0.1]).unwrap();
        let data = PanelData::from_long(&id, &time, &y, &cohort, &x).unwrap();
        assert_eq!(data.y, ndarray::array![[10.0, 11.0], [20.0, 21.0]]); // unit 1 first
        assert_eq!(data.cohorts(), vec![1.0]);
        let bad_x = Array2::from_shape_vec((4, 1), vec![0.2, 0.1, 0.3, 0.1]).unwrap();
        assert!(PanelData::from_long(&id, &time, &y, &cohort, &bad_x).is_err());
    }
}

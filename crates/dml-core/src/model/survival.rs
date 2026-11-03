//! Causal effects on a right-censored time to event: the survival probability
//! and the restricted mean survival time of a binary treatment.
//!
//! Time is split into the slots of a grid and each unit becomes one row per
//! slot at risk, so that any [`Classifier`] can learn the discrete hazards of
//! the event and of censoring. These enter the efficient influence function of
//! the treatment-specific survival curve (Bai, Tsiatis and O'Brien, 2013;
//! Westling, Luedtke, Gilbert and Carone, 2024):
//!
//! ```text
//! φ_a(t_m) = S_m − 1(D = a)/π_a · Σ_{j≤m} R_j · (S_m/S_j) · (dN_j − h_j)/G_j
//! ```
//!
//! with `h_j` the event hazard of slot `j`, `S_m = Π_{j≤m}(1 − h_j)`, `G_j` the
//! probability of being uncensored at slot `j`, `R_j` the at-risk indicator and
//! `dN_j` the event indicator. The score is doubly robust: it is consistent if
//! the event hazards, or the censoring hazards and the propensity score, are.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use crate::compat::np_quantile;
use crate::crossfit::{cross_fit_predict_proba, map_folds, Fold, SampleSplitting};
use crate::data::{is_binary, DmlData};
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::Classifier;
use crate::propensity::{adjust_propensity, Calibration};

/// What is compared between treated and controls at a horizon `τ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SurvivalEstimand {
    /// The survival probability `P(T > τ)`.
    #[default]
    Survival,
    /// The restricted mean survival time `E[min(T, τ)]`.
    Rmst,
}

/// The slots time is split into.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SurvivalGrid {
    /// `k` quantiles of the event times, plus the horizons.
    Quantiles(usize),
    /// Every distinct event time, plus the horizons. Without covariates this
    /// gives the Kaplan-Meier estimator; the size of the data grows quickly.
    EventTimes,
    /// The given times, plus the horizons.
    Fixed(Vec<f64>),
}

/// How the slot enters the hazard learners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum TimeEncoding {
    /// The time of the slot as a number (suits trees and boosting).
    #[default]
    Numeric,
    /// One indicator per slot (gives a linear learner a free baseline hazard).
    Dummies,
}

/// Effect of a binary treatment on a right-censored time to event.
///
/// `data` holds the observed time (`y`), the treatment and the covariates, and
/// `event` is 1 for an observed event and 0 for censoring. Treated and controls
/// have separate hazard models. Each arm's hazards are cross-fitted, whole units
/// go to folds, and folds are stratified by treatment and event. `T` and the
/// censoring time must be independent given the treatment and covariates.
///
/// ```
/// use dml_core::{CausalSurvival, DmlData, LogisticRegression, SurvivalGrid};
/// use ndarray::{Array1, Array2};
///
/// let n = 400;
/// let x = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 7) % 11) as f64 / 5.0 - 1.0);
/// let d: Array1<f64> = (0..n).map(|i| ((i * 13) % 7 < 3) as u8 as f64).collect();
/// let time: Array1<f64> = (0..n)
///     .map(|i| 1.0 + d[i] + 0.2 * x[[i, 0]] + ((i * 31) % 17) as f64 / 8.0)
///     .collect();
/// let event = Array1::from_elem(n, 1.0);
/// let data = DmlData::new(time, d, x)?;
///
/// let fit = CausalSurvival::new(
///     LogisticRegression::new(1.0),
///     LogisticRegression::new(1.0),
///     LogisticRegression::new(1.0),
/// )
/// .horizons(vec![2.0])
/// .grid(SurvivalGrid::Quantiles(10))
/// .fit(&data, event.view())?;
/// assert!(fit.effect().coef()[0] > 0.0); // treatment delays the event
/// # Ok::<(), dml_core::DmlError>(())
/// ```
pub struct CausalSurvival {
    ml_event: Box<dyn Classifier>,
    ml_censor: Box<dyn Classifier>,
    ml_m: Box<dyn Classifier>,
    estimand: SurvivalEstimand,
    horizons: Option<Vec<f64>>,
    grid: SurvivalGrid,
    encoding: TimeEncoding,
    clipping_threshold: f64,
    censoring_floor: f64,
    splitting: SampleSplitting,
}

impl CausalSurvival {
    /// `ml_event` and `ml_censor` learn the hazards of the event and of
    /// censoring per slot, `ml_m` the propensity score.
    pub fn new(
        ml_event: impl Classifier + 'static,
        ml_censor: impl Classifier + 'static,
        ml_m: impl Classifier + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(ml_event), Box::new(ml_censor), Box::new(ml_m))
    }

    /// As [`new`](Self::new), with boxed learners.
    pub fn from_boxed(
        ml_event: Box<dyn Classifier>,
        ml_censor: Box<dyn Classifier>,
        ml_m: Box<dyn Classifier>,
    ) -> Self {
        Self {
            ml_event,
            ml_censor,
            ml_m,
            estimand: SurvivalEstimand::default(),
            horizons: None,
            grid: SurvivalGrid::Quantiles(30),
            encoding: TimeEncoding::default(),
            clipping_threshold: 1e-2,
            censoring_floor: 1e-2,
            splitting: SampleSplitting::default(),
        }
    }

    /// Survival probability (default) or restricted mean survival time.
    pub fn estimand(mut self, estimand: SurvivalEstimand) -> Self {
        self.estimand = estimand;
        self
    }

    /// Times `τ` of the effects (default: the quartiles of the event times).
    pub fn horizons(mut self, horizons: Vec<f64>) -> Self {
        self.horizons = Some(horizons);
        self
    }

    /// The slots time is split into (default: 30 quantiles of the event times).
    pub fn grid(mut self, grid: SurvivalGrid) -> Self {
        self.grid = grid;
        self
    }

    /// How the slot enters the hazard learners (default: its time).
    pub fn time_encoding(mut self, encoding: TimeEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Propensity scores are clipped to `[t, 1 − t]` (default 0.01).
    pub fn clipping_threshold(mut self, threshold: f64) -> Self {
        self.clipping_threshold = threshold;
        self
    }

    /// Probabilities of remaining uncensored are raised to this value (default 0.01).
    pub fn censoring_floor(mut self, floor: f64) -> Self {
        self.censoring_floor = floor;
        self
    }

    /// Number of cross-fitting folds (default 5).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Repetitions of the sample split (default 1).
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed of the sample splits.
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Draw folds, stratified by treatment and event, and fit.
    pub fn fit(&self, data: &DmlData, event: ArrayView1<'_, f64>) -> Result<CausalSurvivalFit> {
        self.check_data(data, event)?;
        let strata: Vec<usize> = data
            .treatment(0)
            .iter()
            .zip(event)
            .map(|(&d, &e)| 2 * (d as usize) + e as usize)
            .collect();
        let splits = self.splitting.draw_stratified(&strata)?;
        self.fit_with_folds(data, event, &splits)
    }

    fn check_data(&self, data: &DmlData, event: ArrayView1<'_, f64>) -> Result<()> {
        let invalid = |message: &str| Err(DmlError::InvalidArgument(message.into()));
        if data.clusters().is_some() {
            return invalid("CausalSurvival does not support clustered data yet");
        }
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return invalid("CausalSurvival needs exactly one binary (0/1) treatment");
        }
        if event.len() != data.n_obs() {
            return Err(DmlError::DimensionMismatch(format!(
                "event has {} rows, expected {}",
                event.len(),
                data.n_obs()
            )));
        }
        if !is_binary(event) {
            return invalid("event must be 1 for an event and 0 for censoring");
        }
        if data.y().iter().any(|&t| t < 0.0) {
            return invalid("observed times must not be negative");
        }
        if !event.iter().any(|&e| e == 1.0) {
            return invalid("there are no observed events");
        }
        if !(0.0..0.5).contains(&self.clipping_threshold) {
            return invalid("clipping_threshold must be in [0, 0.5)");
        }
        if !(self.censoring_floor > 0.0 && self.censoring_floor <= 1.0) {
            return invalid("censoring_floor must be in (0, 1]");
        }
        Ok(())
    }

    /// Fit on given sample splits (one `Vec<Fold>` per repetition).
    pub fn fit_with_folds(
        &self,
        data: &DmlData,
        event: ArrayView1<'_, f64>,
        splits: &[Vec<Fold>],
    ) -> Result<CausalSurvivalFit> {
        self.check_data(data, event)?;
        if splits.is_empty() {
            return Err(DmlError::InvalidArgument(
                "need at least one repetition of sample splits".into(),
            ));
        }
        let (time, d, x) = (data.y(), data.treatment(0), data.x());
        let n = time.len();
        let horizons = self.resolve_horizons(time, event)?;
        let grid = self.resolve_grid(time, event, &horizons)?;
        let schedule = Schedule::new(time, event, &grid);
        let k = grid.len();
        let n_rep = splits.len();
        let mut phi = [
            Array3::<f64>::zeros((n, k, n_rep)),
            Array3::<f64>::zeros((n, k, n_rep)),
        ];
        let mut propensity = Array2::<f64>::zeros((n, n_rep));
        for (r, folds) in splits.iter().enumerate() {
            let m = cross_fit_predict_proba(self.ml_m.as_ref(), x, d, folds)?;
            let m = adjust_propensity(m, d, folds, self.clipping_threshold, Calibration::None)?;
            let hazards = self.hazards(x, d, &schedule, folds)?;
            for (arm, scores) in phi.iter_mut().enumerate() {
                let arm_value = 1.0 - arm as f64;
                let rep = scores_of_arm(
                    arm_value,
                    d,
                    &m,
                    &hazards[arm],
                    &schedule,
                    self.censoring_floor,
                );
                scores.index_axis_mut(Axis(2), r).assign(&rep);
            }
            propensity.column_mut(r).assign(&m);
        }

        let n_folds = splits[0].len();
        let points: Vec<usize> = horizons
            .iter()
            .map(|h| {
                grid.iter()
                    .position(|t| t == h)
                    .expect("horizons are in the grid")
            })
            .collect();
        let [treated, control] = phi;
        let at_horizons = |phi: &Array3<f64>| match self.estimand {
            SurvivalEstimand::Survival => phi.select(Axis(1), &points),
            SurvivalEstimand::Rmst => restricted_means(phi, &grid, &points),
        };
        let (at_treated, at_control) = (at_horizons(&treated), at_horizons(&control));
        let label = match self.estimand {
            SurvivalEstimand::Survival => "survival",
            SurvivalEstimand::Rmst => "RMST",
        };
        let names: Vec<String> = horizons.iter().map(|t| format!("{label}({t})")).collect();
        let fit = |model: String, names: Vec<String>, psi_b: Array3<f64>| {
            DmlFit::from_linear_scores(
                model,
                names,
                n_folds,
                Array3::from_elem(psi_b.dim(), -1.0),
                psi_b,
            )
        };
        let grid_names: Vec<String> = grid.iter().map(|t| format!("survival({t})")).collect();
        Ok(CausalSurvivalFit {
            effect: fit(
                format!("Causal survival ({label}, treated - control)"),
                names.clone(),
                &at_treated - &at_control,
            )?,
            treated: fit(
                format!("Causal survival ({label}, treated)"),
                names.clone(),
                at_treated,
            )?,
            control: fit(
                format!("Causal survival ({label}, control)"),
                names,
                at_control,
            )?,
            curve_treated: fit(
                "Survival curve (treated)".into(),
                grid_names.clone(),
                treated,
            )?,
            curve_control: fit("Survival curve (control)".into(), grid_names, control)?,
            estimand: self.estimand,
            horizons,
            grid,
            propensity,
        })
    }

    fn resolve_horizons(
        &self,
        time: ArrayView1<'_, f64>,
        event: ArrayView1<'_, f64>,
    ) -> Result<Vec<f64>> {
        let events = event_times(time, event);
        let last = *events.last().expect("checked: there are events");
        let mut horizons = match &self.horizons {
            Some(h) => h.clone(),
            None => [0.25, 0.5, 0.75]
                .iter()
                .map(|&q| np_quantile(&events, q))
                .collect(),
        };
        if horizons.is_empty() || horizons.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return Err(DmlError::InvalidArgument(
                "horizons must be positive numbers".into(),
            ));
        }
        if horizons.iter().any(|&t| t > last) {
            return Err(DmlError::InvalidArgument(format!(
                "horizons must not exceed the last event time, {last}"
            )));
        }
        horizons.sort_by(f64::total_cmp);
        horizons.dedup();
        Ok(horizons)
    }

    fn resolve_grid(
        &self,
        time: ArrayView1<'_, f64>,
        event: ArrayView1<'_, f64>,
        horizons: &[f64],
    ) -> Result<Vec<f64>> {
        let last = *horizons.last().expect("checked: horizons are not empty");
        let events: Vec<f64> = event_times(time, event)
            .into_iter()
            .filter(|&t| t <= last)
            .collect();
        let mut grid = match &self.grid {
            SurvivalGrid::Quantiles(0) => {
                return Err(DmlError::InvalidArgument(
                    "the grid needs at least one quantile".into(),
                ))
            }
            SurvivalGrid::Quantiles(_) if events.is_empty() => Vec::new(),
            SurvivalGrid::Quantiles(k) => (1..=*k)
                .map(|j| np_quantile(&events, j as f64 / *k as f64))
                .collect(),
            SurvivalGrid::EventTimes => events,
            SurvivalGrid::Fixed(times) => {
                if times.iter().any(|t| !t.is_finite() || *t <= 0.0) {
                    return Err(DmlError::InvalidArgument(
                        "grid times must be positive numbers".into(),
                    ));
                }
                times.iter().copied().filter(|&t| t <= last).collect()
            }
        };
        grid.extend_from_slice(horizons);
        grid.sort_by(f64::total_cmp);
        grid.dedup();
        Ok(grid)
    }

    /// Cross-fitted hazards of every unit in every slot, for treated and controls.
    fn hazards<'a>(
        &self,
        x: ArrayView2<'a, f64>,
        d: ArrayView1<'_, f64>,
        schedule: &'a Schedule,
        folds: &[Fold],
    ) -> Result<[ArmHazards; 2]> {
        let (n, k) = (x.nrows(), schedule.grid.len());
        let per_fold = map_folds(folds.len(), |f| -> Result<[ArmHazards; 2]> {
            let (train, test) = (&folds[f].train, &folds[f].test);
            let x_test = x.select(Axis(0), test);
            let mut arms = Vec::with_capacity(2);
            for arm_value in [1.0, 0.0] {
                let units: Vec<usize> = train
                    .iter()
                    .copied()
                    .filter(|&i| d[i] == arm_value)
                    .collect();
                if units.is_empty() {
                    return Err(DmlError::InvalidArgument(
                        "a training fold has no units in one of the treatment arms".into(),
                    ));
                }
                let design = Design {
                    x,
                    schedule,
                    encoding: self.encoding,
                };
                let (rows, target) = design.event_rows(&units);
                let event = Hazard::fit(self.ml_event.as_ref(), rows, target)?;
                let (rows, target) = design.censoring_rows(&units);
                let censoring = Hazard::fit(self.ml_censor.as_ref(), rows, target)?;
                arms.push(ArmHazards {
                    event: design.predict(&event, x_test.view(), 1)?,
                    censoring: design.predict(&censoring, x_test.view(), 0)?,
                });
            }
            let control = arms.pop().expect("two arms");
            let treated = arms.pop().expect("two arms");
            Ok([treated, control])
        })?;
        let empty = || ArmHazards {
            event: Array2::zeros((n, k)),
            censoring: Array2::zeros((n, k)),
        };
        let mut all = [empty(), empty()];
        for (fold, hazards) in folds.iter().zip(&per_fold) {
            for (arm, values) in hazards.iter().enumerate() {
                for (row, &i) in fold.test.iter().enumerate() {
                    all[arm].event.row_mut(i).assign(&values.event.row(row));
                    all[arm]
                        .censoring
                        .row_mut(i)
                        .assign(&values.censoring.row(row));
                }
            }
        }
        Ok(all)
    }
}

/// Sorted distinct event times.
fn event_times(time: ArrayView1<'_, f64>, event: ArrayView1<'_, f64>) -> Vec<f64> {
    let mut times: Vec<f64> = time
        .iter()
        .zip(event)
        .filter(|(_, &e)| e == 1.0)
        .map(|(&t, _)| t)
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

/// When each unit is at risk. Slots are numbered from 1 and the grid times are
/// the end of each slot. A unit censored at `c` is at risk at the grid times up
/// to `c`, as in Kaplan-Meier (events come before censoring at a tie).
struct Schedule {
    grid: Vec<f64>,
    /// Last slot in which the unit is at risk of the event (0: none).
    last_at_risk: Vec<usize>,
    /// Slot of the event, if it falls within the grid.
    event_slot: Vec<Option<usize>>,
    /// Slot after which the unit is censored, if before the end of the grid.
    censored_after: Vec<Option<usize>>,
}

impl Schedule {
    fn new(time: ArrayView1<'_, f64>, event: ArrayView1<'_, f64>, grid: &[f64]) -> Self {
        let k = grid.len();
        let n = time.len();
        let (mut last_at_risk, mut event_slot, mut censored_after) =
            (vec![0; n], vec![None; n], vec![None; n]);
        for i in 0..n {
            if event[i] == 1.0 && time[i] <= grid[k - 1] {
                let slot = grid.partition_point(|&t| t < time[i]) + 1;
                last_at_risk[i] = slot;
                event_slot[i] = Some(slot);
            } else {
                let at_risk = grid.partition_point(|&t| t <= time[i]);
                last_at_risk[i] = at_risk;
                if event[i] == 0.0 && at_risk < k {
                    censored_after[i] = Some(at_risk);
                }
            }
        }
        Self {
            grid: grid.to_vec(),
            last_at_risk,
            event_slot,
            censored_after,
        }
    }

    fn time_of(&self, slot: usize) -> f64 {
        if slot == 0 {
            0.0
        } else {
            self.grid[slot - 1]
        }
    }
}

/// Person-period rows: the time features, then the covariates.
struct Design<'a> {
    x: ArrayView2<'a, f64>,
    schedule: &'a Schedule,
    encoding: TimeEncoding,
}

impl Design<'_> {
    fn time_columns(&self) -> usize {
        match self.encoding {
            TimeEncoding::Numeric => 1,
            TimeEncoding::Dummies => self.schedule.grid.len().saturating_sub(1),
        }
    }

    /// The row of `slot` for covariates `x`, where `first` is the first slot of
    /// the hazard (its indicator is dropped, the learner's intercept covers it).
    fn push_row(&self, out: &mut Vec<f64>, x: ArrayView1<'_, f64>, slot: usize, first: usize) {
        match self.encoding {
            TimeEncoding::Numeric => out.push(self.schedule.time_of(slot)),
            TimeEncoding::Dummies => {
                let columns = self.time_columns();
                out.extend((0..columns).map(|c| f64::from(slot == first + 1 + c)));
            }
        }
        out.extend(x.iter().copied());
    }

    fn matrix(&self, values: Vec<f64>) -> Array2<f64> {
        let p = self.time_columns() + self.x.ncols();
        let n = values.len() / p;
        Array2::from_shape_vec((n, p), values).expect("rows of equal length")
    }

    /// Slots at risk of the event: target 1 in the slot of the event.
    fn event_rows(&self, units: &[usize]) -> (Array2<f64>, Array1<f64>) {
        let s = self.schedule;
        let (mut rows, mut target) = (Vec::new(), Vec::new());
        for &i in units {
            for slot in 1..=s.last_at_risk[i] {
                self.push_row(&mut rows, self.x.row(i), slot, 1);
                target.push(f64::from(s.event_slot[i] == Some(slot)));
            }
        }
        (self.matrix(rows), Array1::from(target))
    }

    /// Slots after which a unit could be censored: target 1 in the slot of censoring.
    fn censoring_rows(&self, units: &[usize]) -> (Array2<f64>, Array1<f64>) {
        let s = self.schedule;
        let k = s.grid.len();
        let (mut rows, mut target) = (Vec::new(), Vec::new());
        for &i in units {
            for slot in 0..k.min(s.last_at_risk[i] + 1) {
                if s.event_slot[i] == Some(slot) {
                    break;
                }
                self.push_row(&mut rows, self.x.row(i), slot, 0);
                target.push(f64::from(s.censored_after[i] == Some(slot)));
            }
        }
        (self.matrix(rows), Array1::from(target))
    }

    /// Hazards of `x` in the `k` slots starting at `first`, shape `(n, k)`.
    fn predict(
        &self,
        hazard: &Hazard,
        x: ArrayView2<'_, f64>,
        first: usize,
    ) -> Result<Array2<f64>> {
        let k = self.schedule.grid.len();
        if let Hazard::Constant(p) = hazard {
            return Ok(Array2::from_elem((x.nrows(), k), *p));
        }
        let mut rows = Vec::new();
        for row in x.rows() {
            for slot in first..first + k {
                self.push_row(&mut rows, row, slot, first);
            }
        }
        let rows = self.matrix(rows);
        let values = match hazard {
            Hazard::Fitted(learner) => learner.predict_proba(rows.view())?,
            Hazard::Constant(_) => unreachable!("handled above"),
        };
        let hazards = values
            .into_shape_with_order((x.nrows(), k))
            .map_err(|e| DmlError::InvalidArgument(e.to_string()))?;
        Ok(hazards.mapv(|h| h.clamp(0.0, 1.0)))
    }
}

/// A learned hazard. A fold without events (or censoring) has nothing to learn.
enum Hazard {
    Constant(f64),
    Fitted(Box<dyn Classifier>),
}

impl Hazard {
    fn fit(proto: &dyn Classifier, rows: Array2<f64>, target: Array1<f64>) -> Result<Self> {
        let first = target.first().copied().unwrap_or(0.0);
        if target.iter().all(|&v| v == first) {
            return Ok(Hazard::Constant(first));
        }
        let mut learner = proto.fresh();
        learner.fit(rows.view(), target.view())?;
        Ok(Hazard::Fitted(learner))
    }
}

/// Hazards of one arm, shape `(n, slots)`. `event` starts at slot 1, `censoring`
/// at slot 0: column `j` of `censoring` is the hazard after slot `j`.
struct ArmHazards {
    event: Array2<f64>,
    censoring: Array2<f64>,
}

/// The influence function values `φ_a(t_m)` of one arm, shape `(n, slots)`.
fn scores_of_arm(
    arm: f64,
    d: ArrayView1<'_, f64>,
    m: &Array1<f64>,
    hazards: &ArmHazards,
    schedule: &Schedule,
    floor: f64,
) -> Array2<f64> {
    let (n, k) = hazards.event.dim();
    let mut phi = Array2::zeros((n, k));
    for i in 0..n {
        let weight = if d[i] != arm {
            0.0
        } else if arm == 1.0 {
            1.0 / m[i]
        } else {
            1.0 / (1.0 - m[i])
        };
        let (mut survival, mut uncensored, mut correction) = (1.0, 1.0, 0.0);
        for j in 0..k {
            let slot = j + 1;
            let h = hazards.event[[i, j]];
            uncensored *= 1.0 - hazards.censoring[[i, j]];
            survival *= 1.0 - h;
            // S_m/S_j updated in place: no division by a small survival probability.
            correction *= 1.0 - h;
            if weight != 0.0 && slot <= schedule.last_at_risk[i] {
                let dn = f64::from(schedule.event_slot[i] == Some(slot));
                correction += (dn - h) / f64::max(uncensored, floor);
            }
            phi[[i, j]] = survival - weight * correction;
        }
    }
    phi
}

/// `RMST(τ) = Σ_j S(t_j)·(t_{j+1} − t_j)` from `t_0 = 0`, for the scores of the
/// survival curve, shape `(n, slots, n_rep)`, at the grid positions `points`.
fn restricted_means(phi: &Array3<f64>, grid: &[f64], points: &[usize]) -> Array3<f64> {
    let (n, _, n_rep) = phi.dim();
    let mut out = Array3::zeros((n, points.len(), n_rep));
    for ((i, h, r), value) in out.indexed_iter_mut() {
        let end = points[h];
        let mut total = grid[0];
        for j in 0..end {
            total += phi[[i, j, r]] * (grid[j + 1] - grid[j]);
        }
        *value = total;
    }
    out
}

/// Result of [`CausalSurvival::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CausalSurvivalFit {
    estimand: SurvivalEstimand,
    horizons: Vec<f64>,
    grid: Vec<f64>,
    effect: DmlFit,
    treated: DmlFit,
    control: DmlFit,
    curve_treated: DmlFit,
    curve_control: DmlFit,
    propensity: Array2<f64>,
}

impl CausalSurvivalFit {
    /// The compared quantity.
    pub fn estimand(&self) -> SurvivalEstimand {
        self.estimand
    }

    /// Horizons of the effects.
    pub fn horizons(&self) -> &[f64] {
        &self.horizons
    }

    /// Time grid of the survival curves.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Treated minus control, one estimate per horizon.
    pub fn effect(&self) -> &DmlFit {
        &self.effect
    }

    /// The estimand if everyone were treated (`treated`) or untreated, per horizon.
    pub fn potential(&self, treated: bool) -> &DmlFit {
        if treated {
            &self.treated
        } else {
            &self.control
        }
    }

    /// Survival curve if everyone were treated (`treated`) or untreated, on the grid.
    pub fn curve(&self, treated: bool) -> &DmlFit {
        if treated {
            &self.curve_treated
        } else {
            &self.curve_control
        }
    }

    /// Cross-fitted, clipped propensity scores, shape `(n, n_rep)`.
    pub fn propensity(&self) -> &Array2<f64> {
        &self.propensity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::LogisticRegression;
    use ndarray::array;

    #[test]
    fn schedule_follows_kaplan_meier_ties() {
        let time = array![1.0, 2.0, 2.0, 3.0, 5.0, 0.5];
        let event = array![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let s = Schedule::new(time.view(), event.view(), &[1.0, 2.0, 4.0]);
        assert_eq!(s.last_at_risk, vec![1, 2, 2, 2, 3, 0]);
        assert_eq!(s.event_slot, vec![Some(1), Some(2), None, None, None, None]);
        assert_eq!(
            s.censored_after,
            vec![None, None, Some(2), Some(2), None, Some(0)]
        );
    }

    #[test]
    fn scores_equal_the_plug_in_without_censoring_errors() {
        // All unit weights zero (nobody in the arm): only the plug-in remains.
        let hazards = ArmHazards {
            event: array![[0.5, 0.5]],
            censoring: array![[0.0, 0.0]],
        };
        let s = Schedule::new(array![1.0].view(), array![1.0].view(), &[1.0, 2.0]);
        let phi = scores_of_arm(1.0, array![0.0].view(), &array![0.5], &hazards, &s, 0.01);
        assert_eq!(phi, array![[0.5, 0.25]]);
    }

    #[test]
    fn restricted_mean_integrates_the_step_function() {
        let phi = Array3::from_shape_vec((1, 3, 1), vec![0.8, 0.5, 0.2]).unwrap();
        let means = restricted_means(&phi, &[1.0, 2.0, 4.0], &[0, 2]);
        assert!((means[[0, 0, 0]] - 1.0).abs() < 1e-15);
        assert!((means[[0, 1, 0]] - (1.0 + 0.8 + 0.5 * 2.0)).abs() < 1e-15);
    }

    #[test]
    fn rejects_bad_input() {
        let x = array![[0.0], [1.0], [0.5], [0.2]];
        let time = array![1.0, 2.0, 3.0, 4.0];
        let d = array![0.0, 1.0, 0.0, 1.0];
        let data = DmlData::new(time, d, x).unwrap();
        let logit = || LogisticRegression::new(1.0);
        let model = CausalSurvival::new(logit(), logit(), logit()).n_folds(2);
        assert!(model.fit(&data, array![0.0, 0.0, 0.0, 0.0].view()).is_err());
        assert!(model.fit(&data, array![1.0, 0.5, 1.0, 1.0].view()).is_err());
        let late = CausalSurvival::new(logit(), logit(), logit()).horizons(vec![9.0]);
        assert!(late.fit(&data, array![1.0, 1.0, 1.0, 1.0].view()).is_err());
    }
}

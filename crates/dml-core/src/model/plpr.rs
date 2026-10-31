//! Partially linear panel regression (PLPR) for static panels, as DoubleML's
//! `DoubleMLPLPR` (Clarke and Polselli, 2025):
//!
//! ```text
//! Y_it = θ·D_it + g(X_it) + α_i + U_it,    D_it = m(X_it) + γ_i + V_it
//! ```
//!
//! with unit effects `α_i`, `γ_i` correlated with the covariates. The panel is
//! transformed to remove (or model) the unit effects, and a partially linear
//! regression is fitted on the transformed data with sample splits and
//! standard errors clustered by unit.

use std::collections::HashMap;

use ndarray::{concatenate, Array1, Array2, ArrayView1, Axis};

use super::lplr::TreatmentLearner;
use super::{LinearScoreModel, NuisanceTarget, ScoreEstimate, ScoreInput};
use crate::cluster::ClusterFold;
use crate::compat::np_mean;
use crate::crossfit::{cross_fit_predict, cross_fit_predict_proba, Fold, SampleSplitting};
use crate::data::{is_binary, DmlData};
use crate::error::{DmlError, Result};
use crate::fit::{fit_model, fit_model_clustered, DmlFit};
use crate::learner::Regressor;
use crate::model::PlrScore;
use crate::score::LinearScore;

/// How [`Plpr`] handles the unit effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PanelApproach {
    /// Correlated random effects: unit means of the covariates are added to
    /// them, and the treatment predictions are shifted so their unit means
    /// equal the unit means of the treatment.
    CreGeneral,
    /// Correlated random effects assuming normally distributed treatments:
    /// unit means of the covariates are added, and the treatment learner also
    /// sees the unit mean of the treatment.
    CreNormal,
    /// First differences of outcome and treatment, with the covariates and
    /// their lags (exact).
    #[default]
    FdExact,
    /// Within-unit demeaning of outcome, treatment and covariates (approximate).
    WgApprox,
}

/// A static panel in long format: one row per unit `id` and period `t`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StaticPanel {
    y: Array1<f64>,
    d: Array2<f64>,
    x: Array2<f64>,
    id: Array1<i64>,
    t: Array1<f64>,
    treatment_names: Vec<String>,
    external: Option<super::ExternalPredictions>,
}

impl StaticPanel {
    /// One treatment.
    pub fn new(
        y: Array1<f64>,
        d: Array1<f64>,
        x: Array2<f64>,
        id: Array1<i64>,
        t: Array1<f64>,
    ) -> Result<Self> {
        let n = d.len();
        Self::with_treatments(y, d.into_shape_with_order((n, 1)).expect("1-D"), x, id, t)
    }

    /// Several treatments, one per column of `d`.
    pub fn with_treatments(
        y: Array1<f64>,
        d: Array2<f64>,
        x: Array2<f64>,
        id: Array1<i64>,
        t: Array1<f64>,
    ) -> Result<Self> {
        // DmlData checks shapes and finiteness.
        let data = DmlData::with_treatments(y.clone(), d.clone(), x.clone())?;
        if id.len() != y.len() || t.len() != y.len() {
            return Err(DmlError::DimensionMismatch(
                "id and t need one entry per row".into(),
            ));
        }
        if !t.iter().all(|v| v.is_finite()) {
            return Err(DmlError::InvalidArgument("t must be finite".into()));
        }
        if x.ncols() == 0 {
            return Err(DmlError::InvalidArgument(
                "PLPR needs at least one covariate".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        if !id
            .iter()
            .zip(&t)
            .all(|(&i, &s)| seen.insert((i, s.to_bits())))
        {
            return Err(DmlError::InvalidArgument(
                "each (id, t) pair must appear once".into(),
            ));
        }
        Ok(Self {
            treatment_names: data.treatment_names().to_vec(),
            y,
            d,
            x,
            id,
            t,
            external: None,
        })
    }

    pub fn with_treatment_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.d.ncols() {
            return Err(DmlError::DimensionMismatch(format!(
                "{} treatment names for {} treatments",
                names.len(),
                self.d.ncols()
            )));
        }
        self.treatment_names = names;
        Ok(self)
    }

    /// External nuisance predictions for the rows of the transformed panel
    /// ([`Plpr::transform`]), as DoubleML.
    pub fn with_external_predictions(mut self, external: super::ExternalPredictions) -> Self {
        self.external = Some(external);
        self
    }

    pub fn n_obs(&self) -> usize {
        self.y.len()
    }
}

/// Partially linear regression on a transformed static panel, with folds and
/// standard errors clustered by unit. See [`PanelApproach`] for the
/// transformations and [`PlrScore`] for the scores.
pub struct Plpr {
    ml_l: Box<dyn Regressor>,
    ml_m: TreatmentLearner,
    ml_g: Option<Box<dyn Regressor>>,
    score: PlrScore,
    approach: PanelApproach,
    splitting: SampleSplitting,
}

impl Plpr {
    /// `ml_l` learns `E[Y | X]`, `ml_m` `E[D | X]` on the transformed panel.
    pub fn new(ml_l: impl Regressor + 'static, ml_m: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_l), TreatmentLearner::Regressor(Box::new(ml_m)))
    }

    /// `ml_m` may be a classifier for a binary treatment with the correlated
    /// random effects approaches (the other approaches transform the
    /// treatment, which is then no longer binary).
    pub fn from_boxed(ml_l: Box<dyn Regressor>, ml_m: TreatmentLearner) -> Self {
        Self {
            ml_l,
            ml_m,
            ml_g: None,
            score: PlrScore::PartiallingOut,
            approach: PanelApproach::default(),
            splitting: SampleSplitting::default(),
        }
    }

    pub fn score(mut self, score: PlrScore) -> Self {
        self.score = score;
        self
    }

    pub fn approach(mut self, approach: PanelApproach) -> Self {
        self.approach = approach;
        self
    }

    /// Learner for `E[Y − θD | X]` with the IV-type score (default: a copy of `ml_l`).
    pub fn ml_g(mut self, ml_g: impl Regressor + 'static) -> Self {
        self.ml_g = Some(Box::new(ml_g));
        self
    }

    pub fn ml_g_boxed(mut self, ml_g: Box<dyn Regressor>) -> Self {
        self.ml_g = Some(ml_g);
        self
    }

    /// Number of cross-fitting folds (default 5), over units.
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

    /// The transformed panel the partially linear regression is fitted on,
    /// clustered by unit.
    pub fn transform(&self, panel: &StaticPanel) -> Result<DmlData> {
        Ok(transform(panel, self.approach)?.0)
    }

    /// Draw folds of whole units and fit.
    pub fn fit(&self, panel: &StaticPanel) -> Result<DmlFit> {
        let (data, ids) = transform(panel, self.approach)?;
        let splits = self
            .splitting
            .draw_clustered(data.clusters().expect("clustered by unit"))?;
        fit_model_clustered(&self.model(&ids), &data, &splits)
    }

    /// Fit on given folds of the transformed rows ([`Plpr::transform`]); each
    /// test set should hold whole units.
    pub fn fit_with_folds(&self, panel: &StaticPanel, splits: &[Vec<Fold>]) -> Result<DmlFit> {
        let (data, ids) = transform(panel, self.approach)?;
        fit_model(&self.model(&ids), &data, splits)
    }

    /// Fit on given cluster folds of the transformed rows.
    pub fn fit_with_cluster_folds(
        &self,
        panel: &StaticPanel,
        splits: &[Vec<ClusterFold>],
    ) -> Result<DmlFit> {
        let (data, ids) = transform(panel, self.approach)?;
        fit_model_clustered(&self.model(&ids), &data, splits)
    }

    fn model<'a>(&'a self, ids: &'a [i64]) -> PlprModel<'a> {
        PlprModel { plpr: self, ids }
    }
}

/// Unit mean of `v` for every row, as pandas' `groupby(id).transform("mean")`
/// (Kahan-compensated sums).
fn unit_means(v: ArrayView1<'_, f64>, ids: &[i64]) -> Array1<f64> {
    let mut acc: HashMap<i64, (f64, f64, usize)> = HashMap::new();
    for (&value, id) in v.iter().zip(ids) {
        let (sum, comp, count) = acc.entry(*id).or_insert((0.0, 0.0, 0));
        let y = value - *comp;
        let t = *sum + y;
        *comp = t - *sum - y;
        *sum = t;
        *count += 1;
    }
    ids.iter()
        .map(|id| {
            let (sum, _, count) = acc[id];
            sum / count as f64
        })
        .collect()
}

/// Sort by unit and period, drop units observed once, and transform. Returns
/// the transformed data (clustered by unit) and the unit of every row.
fn transform(panel: &StaticPanel, approach: PanelApproach) -> Result<(DmlData, Vec<i64>)> {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &i in &panel.id {
        *counts.entry(i).or_default() += 1;
    }
    let mut order: Vec<usize> = (0..panel.n_obs())
        .filter(|&r| counts[&panel.id[r]] > 1)
        .collect();
    order.sort_by(|&a, &b| {
        panel.id[a]
            .cmp(&panel.id[b])
            .then(panel.t[a].total_cmp(&panel.t[b]))
    });
    if order.is_empty() {
        return Err(DmlError::InvalidArgument(
            "PLPR needs units observed in at least two periods".into(),
        ));
    }
    let y = panel.y.select(Axis(0), &order);
    let d = panel.d.select(Axis(0), &order);
    let x = panel.x.select(Axis(0), &order);
    let ids: Vec<i64> = order.iter().map(|&r| panel.id[r]).collect();
    let names = &panel.treatment_names;

    let (y, d, x, ids, names) = match approach {
        PanelApproach::CreGeneral | PanelApproach::CreNormal => {
            let mut means = Array2::zeros(x.dim());
            for (j, col) in x.columns().into_iter().enumerate() {
                means.column_mut(j).assign(&unit_means(col, &ids));
            }
            (y, d, concatenate![Axis(1), x, means], ids, names.clone())
        }
        PanelApproach::FdExact => {
            // Each row needs the same unit's previous period on the grid of
            // all periods (DoubleML reindexes to the full grid; rows after a
            // gap have no lag and are dropped).
            let t: Vec<f64> = order.iter().map(|&r| panel.t[r]).collect();
            let mut grid = t.clone();
            grid.sort_by(f64::total_cmp);
            grid.dedup();
            let pos = |v: f64| grid.binary_search_by(|g| g.total_cmp(&v)).expect("on grid");
            let keep: Vec<usize> = (1..ids.len())
                .filter(|&r| ids[r] == ids[r - 1] && pos(t[r]) == pos(t[r - 1]) + 1)
                .collect();
            let prev: Vec<usize> = keep.iter().map(|&r| r - 1).collect();
            let dy = &y.select(Axis(0), &keep) - &y.select(Axis(0), &prev);
            let dd = &d.select(Axis(0), &keep) - &d.select(Axis(0), &prev);
            let xx = concatenate![Axis(1), x.select(Axis(0), &keep), x.select(Axis(0), &prev)];
            let ids: Vec<i64> = keep.iter().map(|&r| ids[r]).collect();
            let names = names.iter().map(|n| format!("{n}_diff")).collect();
            (dy, dd, xx, ids, names)
        }
        PanelApproach::WgApprox => {
            let within = |v: ArrayView1<'_, f64>| -> Array1<f64> {
                let grand = np_mean(&v.to_vec());
                &v - &unit_means(v, &ids) + grand
            };
            let mut d_w = Array2::zeros(d.dim());
            for (j, col) in d.columns().into_iter().enumerate() {
                d_w.column_mut(j).assign(&within(col));
            }
            let mut x_w = Array2::zeros(x.dim());
            for (j, col) in x.columns().into_iter().enumerate() {
                x_w.column_mut(j).assign(&within(col));
            }
            let names = names.iter().map(|n| format!("{n}_demean")).collect();
            (within(y.view()), d_w, x_w, ids, names)
        }
    };
    if ids.is_empty() {
        return Err(DmlError::InvalidArgument(
            "no rows left after first-differencing (units need consecutive periods)".into(),
        ));
    }
    let mut data = DmlData::with_treatments(y, d, x)?
        .with_treatment_names(names)?
        .with_clusters(Array1::from(ids.clone()))?;
    if let Some(e) = &panel.external {
        data = data.with_external_predictions(e.clone());
    }
    Ok((data, ids))
}

/// [`Plpr`] on the transformed data, with the unit of every row.
struct PlprModel<'a> {
    plpr: &'a Plpr,
    ids: &'a [i64],
}

impl LinearScoreModel for PlprModel<'_> {
    fn name(&self) -> String {
        let approach = match self.plpr.approach {
            PanelApproach::CreGeneral => "CRE general",
            PanelApproach::CreNormal => "CRE normal",
            PanelApproach::FdExact => "first differences",
            PanelApproach::WgApprox => "within groups",
        };
        let score = match self.plpr.score {
            PlrScore::PartiallingOut => "partialling out",
            PlrScore::IvType => "IV-type",
        };
        format!("PLPR ({approach}, {score})")
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        if matches!(self.plpr.ml_m, TreatmentLearner::Classifier(_)) {
            let cre = matches!(
                self.plpr.approach,
                PanelApproach::CreGeneral | PanelApproach::CreNormal
            );
            if !cre || !(0..data.n_treat()).all(|j| data.is_binary_treatment(j)) {
                return Err(DmlError::InvalidArgument(
                    "a classifier for ml_m needs binary treatments and a correlated random effects approach"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, .. } = *input;
        let p = self.plpr;
        let l_hat =
            input.predictions_or("ml_l", || cross_fit_predict(p.ml_l.as_ref(), x, y, folds))?;
        let d_mean = unit_means(d, self.ids);
        let x_m = match p.approach {
            PanelApproach::CreNormal => {
                concatenate![Axis(1), x, d_mean.view().insert_axis(Axis(1))]
            }
            _ => x.to_owned(),
        };
        let external_m = input
            .external_ref()
            .is_some_and(|e| e.preds.contains(e.treatment, "ml_m"));
        let mut m_hat = input.predictions_or("ml_m", || match &p.ml_m {
            TreatmentLearner::Regressor(r) => cross_fit_predict(r.as_ref(), x_m.view(), d, folds),
            TreatmentLearner::Classifier(c) => {
                if !is_binary(d) {
                    return Err(DmlError::InvalidArgument(
                        "a classifier for ml_m needs a binary treatment".into(),
                    ));
                }
                cross_fit_predict_proba(c.as_ref(), x_m.view(), d, folds)
            }
        })?;
        // External predictions are used as given (DoubleML).
        if p.approach == PanelApproach::CreGeneral && !external_m {
            // Shift the predictions so their unit means match the treatment's.
            let m_mean = unit_means(m_hat.view(), self.ids);
            m_hat = &m_hat + &d_mean - &m_mean;
        }
        let v: Array1<f64> = &d - &m_hat;
        let m_target = match p.ml_m {
            TreatmentLearner::Regressor(_) => NuisanceTarget::regression("ml_m", d.to_owned()),
            TreatmentLearner::Classifier(_) => NuisanceTarget::classification("ml_m", d.to_owned()),
        };
        match p.score {
            PlrScore::PartiallingOut => {
                let u: Array1<f64> = &y - &l_hat;
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&v * &v), &v * &u)?,
                    vec![("ml_l", l_hat), ("ml_m", m_hat)],
                )
                .with_targets(vec![
                    NuisanceTarget::regression("ml_l", y.to_owned()),
                    m_target,
                ]))
            }
            PlrScore::IvType => {
                // Initial θ from the partialling-out score.
                let psi_a: Vec<f64> = v.iter().map(|v| -(v * v)).collect();
                let psi_b: Vec<f64> = v
                    .iter()
                    .zip(&y)
                    .zip(&l_hat)
                    .map(|((v, y), l)| v * (y - l))
                    .collect();
                let theta_init = -np_mean(&psi_b) / np_mean(&psi_a);
                let target: Array1<f64> = &y - &(theta_init * &d);
                let fresh_l;
                let ml_g = match &p.ml_g {
                    Some(g) => g.as_ref(),
                    None => {
                        fresh_l = p.ml_l.fresh();
                        fresh_l.as_ref()
                    }
                };
                let g_hat = input
                    .predictions_or("ml_g", || cross_fit_predict(ml_g, x, target.view(), folds))?;
                let resid: Array1<f64> = &y - &g_hat;
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&v * &d), &v * &resid)?,
                    vec![("ml_l", l_hat), ("ml_m", m_hat), ("ml_g", g_hat)],
                )
                .with_targets(vec![
                    NuisanceTarget::regression("ml_l", y.to_owned()),
                    m_target,
                    NuisanceTarget::regression("ml_g", target),
                ]))
            }
        }
    }
}

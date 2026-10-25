//! The shared DML driver and its result type, [`DmlFit`].

use std::fmt;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::blp::Blp;
use crate::cluster::{derive_test_clusters, ClusterFold, ClusterInfo};
use crate::crossfit::Fold;
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::model::{CateSignal, LinearScoreModel, LossKind, ScoreInput};
use crate::policy::PolicyTree;
use crate::sensitivity::SensitivityStore;
use crate::stats::{median, norm_ppf, norm_sf};

/// Fit any [`LinearScoreModel`] on the given sample splits (one `Vec<Fold>`
/// per repetition): for every repetition and treatment, cross-fit the
/// nuisances, solve the DML2 score and estimate the standard error; then
/// aggregate over repetitions.
///
/// (`?Sized` lets this accept trait objects such as `&dyn LinearScoreModel`
/// as well as concrete models.)
pub fn fit_model<M: LinearScoreModel + ?Sized>(
    model: &M,
    data: &DmlData,
    splits: &[Vec<Fold>],
) -> Result<DmlFit> {
    // With clustered data, the test clusters of each fold are read off its test rows.
    let cluster = match data.clusters() {
        Some(c) => Some(ClusterInfo::new(
            c.to_vec(),
            splits
                .iter()
                .map(|rep| {
                    rep.iter()
                        .map(|f| ClusterFold {
                            fold: f.clone(),
                            test_clusters: derive_test_clusters(c, f),
                        })
                        .collect()
                })
                .collect(),
        )?),
        None => None,
    };
    fit_model_impl(model, data, splits, cluster)
}

/// Fit on clustered data with explicit cluster folds (one `Vec<ClusterFold>`
/// per repetition), e.g. to reproduce another implementation exactly.
pub fn fit_model_clustered<M: LinearScoreModel + ?Sized>(
    model: &M,
    data: &DmlData,
    splits: &[Vec<ClusterFold>],
) -> Result<DmlFit> {
    let clusters = data.clusters().ok_or_else(|| {
        DmlError::InvalidArgument(
            "cluster folds need clustered data: use DmlData::with_clusters".into(),
        )
    })?;
    let folds: Vec<Vec<Fold>> = splits
        .iter()
        .map(|rep| rep.iter().map(|cf| cf.fold.clone()).collect())
        .collect();
    let cluster = ClusterInfo::new(clusters.to_vec(), splits.to_vec())?;
    fit_model_impl(model, data, &folds, Some(cluster))
}

fn fit_model_impl<M: LinearScoreModel + ?Sized>(
    model: &M,
    data: &DmlData,
    splits: &[Vec<Fold>],
    cluster: Option<ClusterInfo>,
) -> Result<DmlFit> {
    model.validate(data)?;
    let n = data.n_obs();
    check_splits(splits, n)?;
    let (k, n_rep) = (data.n_treat(), splits.len());

    let mut all_coef = Array2::<f64>::zeros((k, n_rep));
    let mut all_se = Array2::<f64>::zeros((k, n_rep));
    let mut psi = Array3::<f64>::zeros((n, k, n_rep));
    let mut psi_deriv = Array2::<f64>::zeros((k, n_rep));
    let mut psi_a = Array3::<f64>::zeros((n, k, n_rep));
    let mut psi_b = Array3::<f64>::zeros((n, k, n_rep));
    let mut predictions: Vec<(String, Array3<f64>)> = Vec::new();
    // Population variance of y, for sensitivity benchmarks.
    let y_mean = data.y().mean().expect("n > 0");
    let var_y = data
        .y()
        .mapv(|v| (v - y_mean).powi(2))
        .mean()
        .expect("n > 0");
    let mut sensitivity = Some(SensitivityStore::new(n, k, n_rep, var_y));
    let mut cate = Some(CateStore {
        signal: Array3::zeros((n, k, n_rep)),
        scale: None,
        is_effect: true,
    });
    let mut losses: Vec<NuisanceLoss> = Vec::new();

    for j in 0..k {
        let x_j = data.covariates_for(j);
        for (r, folds) in splits.iter().enumerate() {
            let input = ScoreInput {
                y: data.y(),
                d: data.treatment(j),
                x: x_j.view(),
                z: data.z(),
                instruments: data.instruments(),
                t: data.time(),
                s: data.selection(),
                rep: r,
            };
            let est = model.estimate_score(&input, folds)?;
            let mut sol = est.score.solve()?;
            if let Some(ci) = &cluster {
                // Fold-weighted estimate and cluster-robust variance; ψ is
                // scaled by the plain mean of ψ_a elsewhere, as in DoubleML.
                let theta = ci.solve_theta(&est.score, r)?;
                let psi_r = &est.score.psi_a * theta + &est.score.psi_b;
                let sigma2 = ci.variance(psi_r.view(), est.score.psi_a.view(), r);
                sol.theta = theta;
                sol.se = sigma2.sqrt();
                sol.psi = psi_r;
            }
            if !(sol.theta.is_finite() && sol.se.is_finite()) {
                return Err(DmlError::InvalidArgument(format!(
                    "non-finite estimate for treatment {j}, repetition {r}"
                )));
            }
            all_coef[[j, r]] = sol.theta;
            all_se[[j, r]] = sol.se;
            psi_deriv[[j, r]] = sol.j;
            psi.slice_mut(ndarray::s![.., j, r]).assign(&sol.psi);
            psi_a
                .slice_mut(ndarray::s![.., j, r])
                .assign(&est.score.psi_a);
            psi_b
                .slice_mut(ndarray::s![.., j, r])
                .assign(&est.score.psi_b);
            for target in &est.targets {
                let Some((_, pred)) = est.predictions.iter().find(|(nm, _)| *nm == target.name)
                else {
                    continue;
                };
                let slot = match losses.iter().position(|l| l.name == target.name) {
                    Some(i) => i,
                    None => {
                        losses.push(NuisanceLoss {
                            name: target.name.to_string(),
                            kind: target.kind,
                            loss: Array2::from_elem((k, n_rep), f64::NAN),
                            targets: Array3::from_elem((n, n_rep, k), f64::NAN),
                        });
                        losses.len() - 1
                    }
                };
                losses[slot].loss[[j, r]] = target.loss(pred);
                losses[slot]
                    .targets
                    .slice_mut(ndarray::s![.., r, j])
                    .assign(&target.target);
            }
            match (&est.sensitivity, &mut sensitivity) {
                (Some(parts), Some(store)) => store.set(j, r, &parts.elements(sol.theta)),
                _ => sensitivity = None,
            }
            match (&est.cate, &mut cate) {
                (Some(c), Some(store)) => store.set(j, r, c, sol.theta),
                _ => cate = None,
            }
            for (name, pred) in est.predictions {
                let slot = match predictions.iter().position(|(nm, _)| *nm == name) {
                    Some(i) => i,
                    None => {
                        predictions
                            .push((name.to_string(), Array3::from_elem((n, n_rep, k), f64::NAN)));
                        predictions.len() - 1
                    }
                };
                predictions[slot]
                    .1
                    .slice_mut(ndarray::s![.., r, j])
                    .assign(&pred);
            }
        }
    }

    if let Some(store) = &mut sensitivity {
        store.validate();
    }
    let (coef, se) = aggregate(&all_coef, &all_se);
    Ok(DmlFit {
        model: model.name(),
        treatment_names: data.treatment_names().to_vec(),
        n_obs: n,
        n_folds: splits[0].len(),
        coef,
        se,
        all_coef,
        all_se,
        psi,
        psi_a,
        psi_b,
        psi_deriv,
        predictions,
        sensitivity,
        cate,
        losses,
        cluster,
    })
}

/// Signal (and basis scale) for CATE / GATE inference, per treatment and
/// repetition, shape `(n, k, n_rep)`.
#[derive(Debug, Clone)]
pub(crate) struct CateStore {
    pub signal: Array3<f64>,
    pub scale: Option<Array3<f64>>,
    pub is_effect: bool,
}

impl CateStore {
    fn set(&mut self, j: usize, r: usize, c: &CateSignal, theta: f64) {
        let signal = match &c.theta_factor {
            Some(f) => &c.signal - &(theta * f),
            None => c.signal.clone(),
        };
        self.signal.slice_mut(ndarray::s![.., j, r]).assign(&signal);
        self.is_effect &= c.is_effect;
        if let Some(scale) = &c.basis_scale {
            let dim = self.signal.dim();
            self.scale
                .get_or_insert_with(|| Array3::zeros(dim))
                .slice_mut(ndarray::s![.., j, r])
                .assign(scale);
        }
    }
}

/// Out-of-sample loss of one nuisance learner, per treatment and repetition.
#[derive(Debug, Clone)]
pub(crate) struct NuisanceLoss {
    pub name: String,
    pub kind: LossKind,
    /// Shape `(k, n_rep)`.
    pub loss: Array2<f64>,
    /// Targets, shape `(n, n_rep, k)`; `NaN` where the nuisance does not apply.
    pub targets: Array3<f64>,
}

fn check_splits(splits: &[Vec<Fold>], n: usize) -> Result<()> {
    if splits.is_empty() {
        return Err(DmlError::InvalidArgument("no sample splits given".into()));
    }
    for (r, folds) in splits.iter().enumerate() {
        let mut seen = vec![false; n];
        for fold in folds {
            for &i in &fold.test {
                if i >= n || seen[i] {
                    return Err(DmlError::InvalidArgument(format!(
                        "repetition {r}: test sets must partition 0..{n}"
                    )));
                }
                seen[i] = true;
            }
            if fold.train.iter().any(|&i| i >= n) {
                return Err(DmlError::InvalidArgument(format!(
                    "repetition {r}: training index out of range"
                )));
            }
        }
        if seen.iter().any(|s| !s) {
            return Err(DmlError::InvalidArgument(format!(
                "repetition {r}: test sets must partition 0..{n}"
            )));
        }
    }
    Ok(())
}

/// DoubleML's aggregation over repetitions: median coefficient, and the
/// standard error implied by the median of the upper bounds `θ_r + 1.96·se_r`.
pub(crate) fn aggregate(
    all_coef: &Array2<f64>,
    all_se: &Array2<f64>,
) -> (Array1<f64>, Array1<f64>) {
    const CRIT: f64 = 1.96;
    let k = all_coef.nrows();
    let mut coef = Array1::zeros(k);
    let mut se = Array1::zeros(k);
    for j in 0..k {
        let c = all_coef.row(j);
        coef[j] = median(&c.to_vec());
        let upper: Vec<f64> = c
            .iter()
            .zip(all_se.row(j))
            .map(|(&t, &s)| t + CRIT * s)
            .collect();
        se[j] = (median(&upper) - coef[j]) / CRIT;
    }
    (coef, se)
}

/// Multiple-testing adjustment for [`DmlFit::p_adjust`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PAdjust {
    Bonferroni,
    /// Holm's step-down procedure.
    Holm,
}

/// A fitted DML model: estimates, standard errors and the per-repetition
/// scores and nuisance predictions behind them.
///
/// Arrays indexed by treatment have length `k = n_treat`; per-repetition
/// arrays have shape `(k, n_rep)`.
#[derive(Debug, Clone)]
pub struct DmlFit {
    pub(crate) model: String,
    pub(crate) treatment_names: Vec<String>,
    pub(crate) n_obs: usize,
    pub(crate) n_folds: usize,
    pub(crate) coef: Array1<f64>,
    pub(crate) se: Array1<f64>,
    pub(crate) all_coef: Array2<f64>,
    pub(crate) all_se: Array2<f64>,
    /// Shape `(n, k, n_rep)`.
    pub(crate) psi: Array3<f64>,
    /// Score elements, shape `(n, k, n_rep)`.
    pub(crate) psi_a: Array3<f64>,
    pub(crate) psi_b: Array3<f64>,
    /// `J = mean(ψ_a)`, shape `(k, n_rep)`.
    pub(crate) psi_deriv: Array2<f64>,
    /// Each array has shape `(n, n_rep, k)`.
    pub(crate) predictions: Vec<(String, Array3<f64>)>,
    /// Sensitivity elements, for models that support sensitivity analysis.
    pub(crate) sensitivity: Option<SensitivityStore>,
    /// Signal for CATE / GATE inference, for models that support it.
    pub(crate) cate: Option<CateStore>,
    /// Out-of-sample losses of the nuisance learners.
    pub(crate) losses: Vec<NuisanceLoss>,
    /// Cluster structure, for clustered data.
    pub(crate) cluster: Option<ClusterInfo>,
}

impl DmlFit {
    pub fn model_name(&self) -> &str {
        &self.model
    }

    pub fn treatment_names(&self) -> &[String] {
        &self.treatment_names
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn n_treat(&self) -> usize {
        self.coef.len()
    }

    pub fn n_rep(&self) -> usize {
        self.all_coef.ncols()
    }

    pub fn n_folds(&self) -> usize {
        self.n_folds
    }

    /// Number of clusters (two-way: the smaller dimension) for a clustered
    /// fit, `None` otherwise.
    pub fn n_clusters(&self) -> Option<usize> {
        self.cluster.as_ref().map(ClusterInfo::n_clusters)
    }

    /// Point estimates θ̂ (median over repetitions).
    pub fn coef(&self) -> ArrayView1<'_, f64> {
        self.coef.view()
    }

    /// Standard errors (aggregated over repetitions).
    pub fn se(&self) -> ArrayView1<'_, f64> {
        self.se.view()
    }

    pub fn t_stat(&self) -> Array1<f64> {
        &self.coef / &self.se
    }

    /// Two-sided p-values for `H0: θ = 0` from the aggregated t-statistic,
    /// as DoubleML's `pval`. (Multiple-testing adjustments in
    /// [`p_adjust`](Self::p_adjust) instead take the median over repetitions.)
    pub fn p_value(&self) -> Array1<f64> {
        self.t_stat().mapv(|t| 2.0 * norm_sf(t.abs()))
    }

    fn per_rep_pvals(&self) -> Array2<f64> {
        let mut p = &self.all_coef / &self.all_se;
        p.mapv_inplace(|t| 2.0 * norm_sf(t.abs()));
        p
    }

    /// Pointwise Wald confidence intervals at `level`, shape `(k, 2)`
    /// (median over repetitions of each repetition's interval).
    pub fn confint(&self, level: f64) -> Result<Array2<f64>> {
        check_level(level)?;
        let z = norm_ppf(0.5 + level / 2.0);
        Ok(confint_from(
            &self.all_coef,
            &self.all_se,
            &Array1::from_elem(self.n_rep(), z),
        ))
    }

    /// Multiple-testing adjusted p-values across treatments (applied per
    /// repetition, then the median is taken). For the bootstrap-based
    /// Romano–Wolf procedure see [`Bootstrap::romano_wolf`](crate::Bootstrap::romano_wolf).
    pub fn p_adjust(&self, method: PAdjust) -> Array1<f64> {
        let raw = self.per_rep_pvals();
        let mut adjusted = Array2::<f64>::zeros(raw.raw_dim());
        for (r, col) in raw.columns().into_iter().enumerate() {
            let adj = match method {
                PAdjust::Bonferroni => {
                    let m = col.len() as f64;
                    col.mapv(|p| (m * p).min(1.0))
                }
                PAdjust::Holm => holm(col),
            };
            adjusted.column_mut(r).assign(&adj);
        }
        adjusted
            .rows()
            .into_iter()
            .map(|row| median(&row.to_vec()))
            .collect()
    }

    /// Estimates for every repetition, shape `(k, n_rep)`.
    pub fn all_coef(&self) -> ArrayView2<'_, f64> {
        self.all_coef.view()
    }

    pub fn all_se(&self) -> ArrayView2<'_, f64> {
        self.all_se.view()
    }

    /// Score values ψ at θ̂, shape `(n, k, n_rep)`.
    pub fn psi(&self) -> ArrayView3<'_, f64> {
        self.psi.view()
    }

    /// Score elements `ψ_a` and `ψ_b` (ψ = ψ_a·θ + ψ_b), shape `(n, k, n_rep)`.
    pub fn psi_elements(&self) -> (ArrayView3<'_, f64>, ArrayView3<'_, f64>) {
        (self.psi_a.view(), self.psi_b.view())
    }

    /// Best linear predictor of the CATE on `basis` (shape `(n, p)`, include a
    /// constant column for an intercept), with HC0 inference (Semenova and
    /// Chernozhukov, 2021), as DoubleML's `cate`. Available for a single
    /// treatment of [`Plr`](crate::Plr) (outcome residual on the basis scaled by
    /// the treatment residual), [`Irm`](crate::Irm) with the ATE score and
    /// [`Apo`](crate::Apo) (orthogonal signal on the basis).
    pub fn cate(&self, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        let (signal, bases) = self.cate_inputs(basis)?;
        Blp::fit_bases(signal.view(), &bases)
    }

    /// Group average treatment effects for mutually exclusive groups given as
    /// 0/1 dummy columns (shape `(n, n_groups)`), as DoubleML's `gate`.
    pub fn gate(&self, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        let (signal, bases) = self.cate_inputs(groups)?;
        Blp::fit_gate_bases(signal.view(), groups, &bases)
    }

    /// Policy tree recommending treatment where the estimated effect is
    /// positive, as DoubleML's `policy_tree` (for [`Irm`](crate::Irm) with the
    /// ATE score, one treatment and one repetition). DoubleML's defaults:
    /// `depth = 2`, `min_samples_leaf = 8`, `ccp_alpha = 0.01`.
    pub fn policy_tree(
        &self,
        features: ArrayView2<'_, f64>,
        depth: usize,
        min_samples_leaf: usize,
        ccp_alpha: f64,
    ) -> Result<PolicyTree> {
        let store = self.cate.as_ref().filter(|c| c.is_effect).ok_or_else(|| {
            DmlError::InvalidArgument(format!(
                "policy trees need a treatment-effect signal (IRM with the ATE score), not {}",
                self.model
            ))
        })?;
        if self.n_treat() != 1 || self.n_rep() != 1 {
            return Err(DmlError::InvalidArgument(
                "policy trees need a single treatment and one repetition (as DoubleML)".into(),
            ));
        }
        if features.nrows() != self.n_obs {
            return Err(DmlError::DimensionMismatch(format!(
                "features have {} rows, the data {}",
                features.nrows(),
                self.n_obs
            )));
        }
        PolicyTree::fit(
            store.signal.slice(ndarray::s![.., 0, 0]),
            features,
            depth,
            min_samples_leaf,
            ccp_alpha,
        )
    }

    /// The signal of every repetition and the matching (scaled) bases.
    fn cate_inputs(&self, basis: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Vec<Array2<f64>>)> {
        let store = self.cate.as_ref().ok_or_else(|| {
            DmlError::InvalidArgument(format!(
                "CATE / GATE are not available for {} (PLR, IRM with the ATE score and APO support them)",
                self.model
            ))
        })?;
        if self.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "CATE / GATE need a single treatment".into(),
            ));
        }
        if basis.nrows() != self.n_obs {
            return Err(DmlError::DimensionMismatch(format!(
                "basis has {} rows, the data {}",
                basis.nrows(),
                self.n_obs
            )));
        }
        let signal = store.signal.index_axis(Axis(1), 0).to_owned();
        let bases = (0..self.n_rep())
            .map(|r| match &store.scale {
                Some(scale) => {
                    let s = scale.slice(ndarray::s![.., 0, r]);
                    &basis * &s.insert_axis(Axis(1))
                }
                None => basis.to_owned(),
            })
            .collect();
        Ok((signal, bases))
    }

    /// Out-of-sample loss of a nuisance learner, shape `(n_treat, n_rep)`:
    /// RMSE for regression learners, log loss for classifiers (DoubleML's
    /// `nuisance_loss`). Computed on the rows where the nuisance applies.
    pub fn nuisance_loss(&self, name: &str) -> Option<ArrayView2<'_, f64>> {
        self.losses
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.loss.view())
    }

    /// `(name, loss kind, losses)` for every evaluated nuisance learner.
    pub fn nuisance_losses(&self) -> impl Iterator<Item = (&str, LossKind, ArrayView2<'_, f64>)> {
        self.losses
            .iter()
            .map(|l| (l.name.as_str(), l.kind, l.loss.view()))
    }

    /// Targets of a nuisance learner, shape `(n, n_rep, n_treat)`, with `NaN`
    /// where the nuisance does not apply (to evaluate custom metrics).
    pub fn nuisance_targets(&self, name: &str) -> Option<ArrayView3<'_, f64>> {
        self.losses
            .iter()
            .find(|l| l.name == name)
            .map(|l| l.targets.view())
    }

    /// Names of the stored nuisance predictions (e.g. `ml_l`, `ml_m`).
    pub fn nuisance_names(&self) -> impl Iterator<Item = &str> {
        self.predictions.iter().map(|(n, _)| n.as_str())
    }

    /// All predictions of one nuisance, shape `(n, n_rep, k)`.
    pub fn predictions(&self, name: &str) -> Option<ArrayView3<'_, f64>> {
        self.predictions
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, p)| p.view())
    }

    /// Out-of-fold predictions of one nuisance for a repetition and treatment.
    pub fn prediction(&self, name: &str, rep: usize, treatment: usize) -> Option<Array1<f64>> {
        let p = self.predictions(name)?;
        (rep < self.n_rep() && treatment < self.n_treat())
            .then(|| p.index_axis(Axis(2), treatment).column(rep).to_owned())
    }
}

/// Per-repetition intervals `θ_r ± crit_r · se_r`, aggregated by the median
/// over repetitions (DoubleML's convention). Shape `(k, 2)`.
pub(crate) fn confint_from(
    all_coef: &Array2<f64>,
    all_se: &Array2<f64>,
    crit: &Array1<f64>,
) -> Array2<f64> {
    let (k, n_rep) = all_coef.dim();
    let mut ci = Array2::zeros((k, 2));
    for j in 0..k {
        let lo: Vec<f64> = (0..n_rep)
            .map(|r| all_coef[[j, r]] - crit[r] * all_se[[j, r]])
            .collect();
        let hi: Vec<f64> = (0..n_rep)
            .map(|r| all_coef[[j, r]] + crit[r] * all_se[[j, r]])
            .collect();
        ci[[j, 0]] = median(&lo);
        ci[[j, 1]] = median(&hi);
    }
    ci
}

impl DmlFit {
    /// Assemble a fit from per-repetition estimates and score elements that
    /// are already laid out over all `n` units (zero for units a sub-model did
    /// not use). Standard errors follow DoubleML: `J = mean(ψ_a)` and
    /// `se = √(mean(ψ²) / J² / n)` over all `n` units.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_scores(
        model: String,
        treatment_names: Vec<String>,
        n_folds: usize,
        all_coef: Array2<f64>,
        psi: Array3<f64>,
        psi_a: Array3<f64>,
        psi_b: Array3<f64>,
        predictions: Vec<(String, Array3<f64>)>,
    ) -> Self {
        let (n, k, n_rep) = psi.dim();
        let mut psi_deriv = Array2::<f64>::zeros((k, n_rep));
        let mut all_se = Array2::<f64>::zeros((k, n_rep));
        for j in 0..k {
            for r in 0..n_rep {
                let jac = psi_a.slice(ndarray::s![.., j, r]).sum() / n as f64;
                let col = psi.slice(ndarray::s![.., j, r]);
                psi_deriv[[j, r]] = jac;
                all_se[[j, r]] = (col.dot(&col) / n as f64 / (jac * jac) / n as f64).sqrt();
            }
        }
        let (coef, se) = aggregate(&all_coef, &all_se);
        Self {
            model,
            treatment_names,
            n_obs: n,
            n_folds,
            coef,
            se,
            all_coef,
            all_se,
            psi,
            psi_a,
            psi_b,
            psi_deriv,
            predictions,
            sensitivity: None,
            cate: None,
            losses: Vec::new(),
            cluster: None,
        }
    }

    /// Linear combinations `Σ_j w_cj θ_j` of this fit's estimates (one row of
    /// `weights` per combination), with influence functions combined the same
    /// way, so correlation between the estimates is accounted for.
    pub(crate) fn linear_combination(
        &self,
        model: String,
        names: Vec<String>,
        weights: &Array2<f64>,
    ) -> Self {
        let (n, k, n_rep) = self.psi.dim();
        assert_eq!(weights.ncols(), k, "one weight per estimate");
        let n_comb = weights.nrows();
        let mut scaled = Array3::<f64>::zeros((n, n_comb, n_rep));
        let mut all_coef = Array2::<f64>::zeros((n_comb, n_rep));
        let mut all_se = Array2::<f64>::zeros((n_comb, n_rep));
        for r in 0..n_rep {
            for c in 0..n_comb {
                let mut col = Array1::<f64>::zeros(n);
                let mut theta = 0.0;
                for j in 0..k {
                    let w = weights[[c, j]];
                    if w != 0.0 {
                        col.scaled_add(
                            w / self.psi_deriv[[j, r]],
                            &self.psi.slice(ndarray::s![.., j, r]),
                        );
                        theta += w * self.all_coef[[j, r]];
                    }
                }
                all_coef[[c, r]] = theta;
                all_se[[c, r]] = match &self.cluster {
                    // The combined influence function is already scaled (J = 1).
                    Some(ci) => ci.variance(col.view(), Array1::ones(n).view(), r).sqrt(),
                    None => (col.dot(&col) / n as f64 / n as f64).sqrt(),
                };
                scaled.slice_mut(ndarray::s![.., c, r]).assign(&col);
            }
        }
        let (coef, se) = aggregate(&all_coef, &all_se);
        Self {
            model,
            treatment_names: names,
            n_obs: n,
            n_folds: self.n_folds,
            coef,
            se,
            all_coef,
            all_se,
            // Influence functions are stored already scaled (J = 1); the score
            // elements are zero so CATE methods correctly refuse this fit.
            psi: scaled,
            psi_a: Array3::zeros((n, n_comb, n_rep)),
            psi_b: Array3::zeros((n, n_comb, n_rep)),
            psi_deriv: Array2::ones((n_comb, n_rep)),
            predictions: Vec::new(),
            sensitivity: None,
            cate: None,
            losses: Vec::new(),
            cluster: self.cluster.clone(),
        }
    }

    /// Stack fits of the same data and sample splits into one fit with all
    /// their estimates (DoubleML's `concat`), for joint inference: the
    /// bootstrap, simultaneous confidence bands and linear combinations then
    /// account for the correlation between the estimates.
    ///
    /// Predictions and learner losses are stacked by nuisance name (`NaN` where
    /// a fit has no such nuisance); sensitivity elements are kept if every fit
    /// has them.
    pub fn concat(fits: &[DmlFit]) -> Result<DmlFit> {
        let first = fits
            .first()
            .ok_or_else(|| DmlError::InvalidArgument("concat needs at least one fit".into()))?;
        let (n, n_rep) = (first.n_obs, first.n_rep());
        for f in fits {
            if f.n_obs != n || f.n_rep() != n_rep || f.n_folds != first.n_folds {
                return Err(DmlError::DimensionMismatch(
                    "concat needs fits with the same observations, folds and repetitions".into(),
                ));
            }
            let same_clusters = match (&f.cluster, &first.cluster) {
                (None, None) => true,
                (Some(a), Some(b)) => a.clusters == b.clusters && a.folds == b.folds,
                _ => false,
            };
            if !same_clusters {
                return Err(DmlError::InvalidArgument(
                    "concat needs the same clusters and cluster splits in every fit".into(),
                ));
            }
        }
        let cat1 = |f: fn(&DmlFit) -> &Array1<f64>| {
            let v: Vec<_> = fits.iter().map(|x| f(x).view()).collect();
            ndarray::concatenate(Axis(0), &v).expect("1-D")
        };
        let cat2 = |f: fn(&DmlFit) -> &Array2<f64>| {
            let v: Vec<_> = fits.iter().map(|x| f(x).view()).collect();
            ndarray::concatenate(Axis(0), &v).expect("same n_rep")
        };
        let cat3 = |f: fn(&DmlFit) -> &Array3<f64>| {
            let v: Vec<_> = fits.iter().map(|x| f(x).view()).collect();
            ndarray::concatenate(Axis(1), &v).expect("same n and n_rep")
        };
        let ks: Vec<usize> = fits.iter().map(DmlFit::n_treat).collect();
        let k_total: usize = ks.iter().sum();

        // Stack per-nuisance arrays (shape (n, n_rep, k)); NaN where absent.
        let mut predictions: Vec<(String, Array3<f64>)> = Vec::new();
        let mut losses: Vec<NuisanceLoss> = Vec::new();
        let mut offset = 0;
        for (f, &k) in fits.iter().zip(&ks) {
            for (name, pred) in &f.predictions {
                let slot = match predictions.iter().position(|(nm, _)| nm == name) {
                    Some(i) => i,
                    None => {
                        predictions.push((
                            name.clone(),
                            Array3::from_elem((n, n_rep, k_total), f64::NAN),
                        ));
                        predictions.len() - 1
                    }
                };
                predictions[slot]
                    .1
                    .slice_mut(ndarray::s![.., .., offset..offset + k])
                    .assign(pred);
            }
            for l in &f.losses {
                let slot = match losses.iter().position(|x| x.name == l.name) {
                    Some(i) => i,
                    None => {
                        losses.push(NuisanceLoss {
                            name: l.name.clone(),
                            kind: l.kind,
                            loss: Array2::from_elem((k_total, n_rep), f64::NAN),
                            targets: Array3::from_elem((n, n_rep, k_total), f64::NAN),
                        });
                        losses.len() - 1
                    }
                };
                losses[slot]
                    .loss
                    .slice_mut(ndarray::s![offset..offset + k, ..])
                    .assign(&l.loss);
                losses[slot]
                    .targets
                    .slice_mut(ndarray::s![.., .., offset..offset + k])
                    .assign(&l.targets);
            }
            offset += k;
        }
        let stores: Option<Vec<&SensitivityStore>> =
            fits.iter().map(|f| f.sensitivity.as_ref()).collect();
        // CATE signals stack if every fit has one and all or none scale the basis.
        let cates: Option<Vec<&CateStore>> = fits.iter().map(|f| f.cate.as_ref()).collect();
        let cate_store = cates.and_then(|c| {
            let views: Vec<_> = c.iter().map(|s| s.signal.view()).collect();
            let signal = ndarray::concatenate(Axis(1), &views).ok()?;
            let scales: Option<Vec<_>> = c
                .iter()
                .map(|s| s.scale.as_ref().map(|a| a.view()))
                .collect();
            let is_effect = c.iter().all(|s| s.is_effect);
            match scales {
                Some(v) => Some(CateStore {
                    signal,
                    scale: Some(ndarray::concatenate(Axis(1), &v).ok()?),
                    is_effect,
                }),
                None if c.iter().all(|s| s.scale.is_none()) => Some(CateStore {
                    signal,
                    scale: None,
                    is_effect,
                }),
                None => None,
            }
        });
        let names: Vec<&str> = fits.iter().map(|f| f.model.as_str()).collect();
        Ok(DmlFit {
            model: if names.iter().all(|m| *m == names[0]) {
                names[0].to_string()
            } else {
                names.join(" + ")
            },
            treatment_names: fits
                .iter()
                .flat_map(|f| f.treatment_names.iter().cloned())
                .collect(),
            n_obs: n,
            n_folds: first.n_folds,
            coef: cat1(|f| &f.coef),
            se: cat1(|f| &f.se),
            all_coef: cat2(|f| &f.all_coef),
            all_se: cat2(|f| &f.all_se),
            psi: cat3(|f| &f.psi),
            psi_a: cat3(|f| &f.psi_a),
            psi_b: cat3(|f| &f.psi_b),
            psi_deriv: cat2(|f| &f.psi_deriv),
            predictions,
            sensitivity: stores.map(|s| SensitivityStore::concat(&s)),
            cate: cate_store,
            losses,
            cluster: first.cluster.clone(),
        })
    }
}

pub(crate) fn check_level(level: f64) -> Result<()> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(DmlError::InvalidArgument(
            "confidence level must be in (0, 1)".into(),
        ))
    }
}

/// Holm step-down adjustment (as `statsmodels.multipletests(method="holm")`).
fn holm(p: ArrayView1<'_, f64>) -> Array1<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut out = Array1::zeros(m);
    let mut running = 0.0_f64;
    for (rank, &i) in order.iter().enumerate() {
        running = running.max(((m - rank) as f64 * p[i]).min(1.0));
        out[i] = running;
    }
    out
}

impl fmt::Display for DmlFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let clustering = match &self.cluster {
            Some(ci) if ci.n_ways() == 2 => {
                format!(", two-way clustered ({} clusters)", ci.n_clusters())
            }
            Some(ci) => format!(", clustered ({} clusters)", ci.n_clusters()),
            None => String::new(),
        };
        writeln!(
            f,
            "{}  |  n_obs = {}, n_folds = {}, n_rep = {}{clustering}",
            self.model,
            self.n_obs,
            self.n_folds,
            self.n_rep()
        )?;
        let ci = self.confint(0.95).map_err(|_| fmt::Error)?;
        let (t, p) = (self.t_stat(), self.p_value());
        let width = self
            .treatment_names
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(1)
            .max(4);
        write!(
            f,
            "{:<width$} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "", "coef", "std err", "t", "P>|t|", "2.5 %", "97.5 %"
        )?;
        for j in 0..self.n_treat() {
            write!(
                f,
                "\n{:<width$} {:>10.4} {:>10.4} {:>8.3} {:>8.4} {:>10.4} {:>10.4}",
                self.treatment_names[j],
                self.coef[j],
                self.se[j],
                t[j],
                p[j],
                ci[[j, 0]],
                ci[[j, 1]]
            )?;
        }
        if !self.losses.is_empty() {
            write!(
                f,
                "\nNuisance learners (out-of-sample, mean over repetitions):"
            )?;
            for l in &self.losses {
                let metric = match l.kind {
                    LossKind::Rmse => "RMSE",
                    LossKind::LogLoss => "log loss",
                };
                for j in 0..self.n_treat() {
                    let row: Vec<f64> = l
                        .loss
                        .row(j)
                        .iter()
                        .copied()
                        .filter(|v| v.is_finite())
                        .collect();
                    if row.is_empty() {
                        continue;
                    }
                    let mean = row.iter().sum::<f64>() / row.len() as f64;
                    let who = if self.n_treat() > 1 {
                        format!(" ({})", self.treatment_names[j])
                    } else {
                        String::new()
                    };
                    write!(f, "\n  {:<12} {metric:<8} {mean:.4}{who}", l.name)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn aggregation_matches_doubleml_definition() {
        let coef = array![[1.0, 2.0, 4.0]];
        let se = array![[0.5, 0.1, 0.2]];
        let (c, s) = aggregate(&coef, &se);
        assert_eq!(c[0], 2.0);
        // upper bounds: 1.98, 2.196, 4.392 → median 2.196
        assert!((s[0] - (2.196 - 2.0) / 1.96).abs() < 1e-12);
    }

    #[test]
    fn holm_matches_statsmodels() {
        // statsmodels.multipletests([0.01, 0.04, 0.03], method="holm")[1]
        let adj = holm(array![0.01, 0.04, 0.03].view());
        assert_eq!(adj, array![0.03, 0.06, 0.06]);
    }
}

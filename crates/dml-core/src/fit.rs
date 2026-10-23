//! The shared DML driver and its result type, [`DmlFit`].

use std::fmt;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::blp::Blp;
use crate::crossfit::Fold;
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::model::{LinearScoreModel, ScoreInput};
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

    for j in 0..k {
        let x_j = data.covariates_for(j);
        let input = ScoreInput {
            y: data.y(),
            d: data.treatment(j),
            x: x_j.view(),
            z: data.z(),
            t: data.time(),
        };
        for (r, folds) in splits.iter().enumerate() {
            let est = model.estimate_score(&input, folds)?;
            let sol = est.score.solve()?;
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
            match (&est.sensitivity, &mut sensitivity) {
                (Some(parts), Some(store)) => store.set(j, r, &parts.elements(sol.theta)),
                _ => sensitivity = None,
            }
            for (name, pred) in est.predictions {
                let slot = match predictions.iter().position(|(nm, _)| nm == name) {
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
    })
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
    /// constant column for an intercept), using the orthogonal signal `ψ_b`.
    /// Valid for scores with `ψ_a = −1`, i.e. [`Irm`](crate::Irm) with the ATE score.
    pub fn cate(&self, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        Blp::fit(self.orthogonal_signal()?.view(), basis)
    }

    /// Group average treatment effects for mutually exclusive groups given as
    /// 0/1 dummy columns (shape `(n, n_groups)`).
    pub fn gate(&self, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        Blp::fit_gate(self.orthogonal_signal()?.view(), groups)
    }

    fn orthogonal_signal(&self) -> Result<Array2<f64>> {
        if self.n_treat() != 1 || self.psi_a.iter().any(|&a| a != -1.0) {
            return Err(DmlError::InvalidArgument(
                "CATE / GATE need a single treatment and a score with psi_a = -1 (IRM with the ATE score)"
                    .into(),
            ));
        }
        Ok(self.psi_b.index_axis(Axis(1), 0).to_owned())
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
                all_se[[c, r]] = (col.dot(&col) / n as f64 / n as f64).sqrt();
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
        }
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
        writeln!(
            f,
            "{}  |  n_obs = {}, n_folds = {}, n_rep = {}",
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

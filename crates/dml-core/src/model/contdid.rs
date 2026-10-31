//! Difference-in-differences with a continuous treatment: Callaway,
//! Goodman-Bacon and Sant'Anna (2024), "Difference-in-differences with a
//! continuous treatment" (NBER WP 32117), two periods. Without covariates this
//! is the B-spline estimator of their R package `contdid`; with covariates a
//! doubly robust version under conditional parallel trends with cross-fitted
//! nuisances.

use ndarray::{s, Array1, Array2, Array3, Axis};

use super::mediation::{classify, fit_regressor, predict_rows};
use crate::compat::np_quantile;
use crate::crossfit::{map_folds, Fold, SampleSplitting};
use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};
use crate::linalg::spd_inverse;

/// B-spline basis of `splines2::bSpline` (no intercept column) with interior
/// `knots` and boundary knots `[lo, hi]`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct BSpline {
    degree: usize,
    /// Full knot vector: `lo` and `hi` repeated `degree + 1` times around the
    /// interior knots.
    knots: Vec<f64>,
}

impl BSpline {
    pub(crate) fn new(degree: usize, interior: &[f64], lo: f64, hi: f64) -> Self {
        let mut knots = vec![lo; degree + 1];
        knots.extend_from_slice(interior);
        knots.extend(std::iter::repeat(hi).take(degree + 1));
        Self { degree, knots }
    }

    /// Number of basis functions without the first one (`intercept = FALSE`).
    pub(crate) fn dim(&self) -> usize {
        self.knots.len() - self.degree - 2
    }

    /// All `dim + 1` basis functions of degree `p` at `x` (Cox–de Boor).
    fn all(&self, x: f64, p: usize) -> Vec<f64> {
        let t = &self.knots;
        let (lo, hi) = (t[0], t[t.len() - 1]);
        let m = t.len() - 1;
        // Degree 0: indicator of the knot span, the last span closed on the right.
        let mut b: Vec<f64> = (0..m)
            .map(|i| {
                let inside =
                    (t[i] <= x && x < t[i + 1]) || (x == hi && t[i] < hi && t[i + 1] == hi);
                f64::from(u8::from(inside && x >= lo))
            })
            .collect();
        for k in 1..=p {
            b = (0..m - k)
                .map(|i| {
                    let left = if t[i + k] > t[i] {
                        (x - t[i]) / (t[i + k] - t[i]) * b[i]
                    } else {
                        0.0
                    };
                    let right = if t[i + k + 1] > t[i + 1] {
                        (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * b[i + 1]
                    } else {
                        0.0
                    };
                    left + right
                })
                .collect();
        }
        b
    }

    /// `[1, B_2(x), …, B_K(x)]`: an intercept and the basis without its first
    /// function, as `lm(dy ~ bSpline(d))` in `contdid`.
    pub(crate) fn row(&self, x: f64) -> Array1<f64> {
        let b = self.all(x, self.degree);
        let mut out = Array1::ones(self.dim() + 1);
        for j in 0..self.dim() {
            out[j + 1] = b[j + 1];
        }
        out
    }

    /// Derivative of [`row`](Self::row) (`splines2::dbs`, 0 for the intercept).
    pub(crate) fn deriv_row(&self, x: f64) -> Array1<f64> {
        let p = self.degree;
        let t = &self.knots;
        let mut out = Array1::zeros(self.dim() + 1);
        if p == 0 {
            return out;
        }
        let lower = self.all(x, p - 1);
        for j in 0..self.dim() {
            let i = j + 1;
            let a = if t[i + p] > t[i] {
                p as f64 / (t[i + p] - t[i]) * lower[i]
            } else {
                0.0
            };
            let b = if t[i + p + 1] > t[i + 1] {
                p as f64 / (t[i + p + 1] - t[i + 1]) * lower[i + 1]
            } else {
                0.0
            };
            out[j + 1] = a - b;
        }
        out
    }
}

/// Dose-response of a continuous treatment (dose) in a two-period
/// difference-in-differences design (Callaway, Goodman-Bacon and Sant'Anna
/// 2024).
///
/// The data are one row per unit: the outcome change `y = Y_post − Y_pre`, the
/// dose `d ≥ 0` received between the periods (0 = untreated, the comparison
/// group) and pre-treatment covariates `x` (possibly none). Under (conditional)
/// parallel trends, `E[ΔY(0) | D = d, X] = E[ΔY | D = 0, X]`, the average
/// effect of dose `d` on the units that received it is
///
/// ```text
/// ATT(d | d) = E[ΔY − m₀(X) | D = d],   m₀(X) = E[ΔY | D = 0, X],
/// ```
///
/// estimated as a B-spline series in `d` (degree 3, no interior knots by
/// default, as `contdid`): `ATT(d) = B(d)'β` with
///
/// ```text
/// β = Q⁻¹ · 1/n Σ [T·B(D)·R − (1 − T)·p(X)/(1 − p(X))·b̄(X)·R],   Q = 1/n Σ T·B(D)B(D)',
/// ```
///
/// where `T = 1{D > 0}`, `R = ΔY − m₀(X)`, `p(X) = P(T = 1 | X)` and
/// `b̄(X) = E[B(D) | X, T = 1]`. The second term makes the estimate doubly
/// robust (insensitive to first-order errors in `m₀`, `p` and `b̄`). The
/// derivative `ACRT(d) = B'(d)'β` is the average causal response; the fit also
/// reports the overall `ATT = E[ΔY − m₀(X) | T = 1]` (the doubly robust DiD
/// estimate) and the overall `ACRT = E[B'(D)'β | T = 1]`. All have explicit
/// influence functions, so standard errors, the multiplier bootstrap and
/// uniform bands work as for the other models.
///
/// Without covariates the nuisances are sample means and the estimates equal
/// `contdid`'s (`cont_did(..., dose_est_method = "parametric")`). With
/// covariates `ml_g` learns `m₀` on the untreated units, `ml_m` the
/// probability `p(X)` (clipped to `[t, 1 − t]`, default `t = 0.01`) and
/// `ml_nested` (default: a copy of `ml_g`) each spline function `b̄_j(X)` on the
/// treated units, all cross-fitted.
///
/// ```
/// use dml_core::{ContinuousDid, DmlData};
/// use ndarray::{Array1, Array2};
///
/// let n = 2000;
/// let d: Array1<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { (i % 97) as f64 / 97.0 + 0.01 }).collect();
/// let y: Array1<f64> = (0..n).map(|i| 1.0 + 2.0 * d[i] * d[i] + ((i * 37) % 11) as f64 / 10.0).collect();
/// let data = DmlData::new(y, d, Array2::zeros((n, 0)))?;
///
/// let fit = ContinuousDid::new().grid(vec![0.25, 0.5, 0.75]).fit(&data)?;
/// // ATT(d) = 2d² (the noise pattern is unrelated to the dose).
/// assert!((fit.att().coef()[1] - 0.5).abs() < 0.15);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
pub struct ContinuousDid {
    ml_g: Option<Box<dyn Regressor>>,
    ml_m: Option<Box<dyn Classifier>>,
    ml_nested: Option<Box<dyn Regressor>>,
    degree: usize,
    num_knots: usize,
    grid: Option<Vec<f64>>,
    trimming_threshold: f64,
    splitting: SampleSplitting,
}

impl Default for ContinuousDid {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousDid {
    /// Without learners: for data without covariates.
    pub fn new() -> Self {
        Self {
            ml_g: None,
            ml_m: None,
            ml_nested: None,
            degree: 3,
            num_knots: 0,
            grid: None,
            trimming_threshold: 1e-2,
            splitting: SampleSplitting::default(),
        }
    }

    /// With covariates: `ml_g` learns `E[ΔY | D = 0, X]` (and, through a
    /// fresh copy, the spline means `E[B(D) | X, D > 0]` unless
    /// [`ml_nested`](Self::ml_nested) is set), `ml_m` the probability
    /// `P(D > 0 | X)`.
    pub fn with_learners(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    /// Like [`ContinuousDid::with_learners`] for boxed trait objects.
    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Box<dyn Classifier>) -> Self {
        Self {
            ml_g: Some(ml_g),
            ml_m: Some(ml_m),
            ..Self::new()
        }
    }

    /// Learner of the spline means `E[B_j(D) | X, D > 0]`.
    pub fn ml_nested(self, ml_nested: impl Regressor + 'static) -> Self {
        self.ml_nested_boxed(Box::new(ml_nested))
    }

    /// Like [`ml_nested`](Self::ml_nested) for a boxed trait object.
    pub fn ml_nested_boxed(mut self, ml_nested: Box<dyn Regressor>) -> Self {
        self.ml_nested = Some(ml_nested);
        self
    }

    /// Degree of the B-spline in the dose (default 3).
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Number of interior knots, at quantiles of the positive doses (default 0).
    pub fn num_knots(mut self, num_knots: usize) -> Self {
        self.num_knots = num_knots;
        self
    }

    /// Doses at which to report `ATT(d)` and `ACRT(d)` (default: the
    /// 5%, 10%, …, 95% quantiles of the positive doses). Must lie within the
    /// range of the positive doses.
    pub fn grid(mut self, grid: Vec<f64>) -> Self {
        self.grid = Some(grid);
        self
    }

    /// Clip `P(D > 0 | X)` to `[t, 1 − t]` (default 0.01; covariates only).
    pub fn trimming_threshold(mut self, threshold: f64) -> Self {
        self.trimming_threshold = threshold;
        self
    }

    /// Number of cross-fitting folds (default 5; covariates only).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Number of independent sample splits (default 1; covariates only).
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed for the sample splits (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Fit; with covariates on seeded splits stratified by treated and
    /// untreated units.
    pub fn fit(&self, data: &DmlData) -> Result<ContinuousDidFit> {
        if data.n_features() == 0 {
            return self.fit_with_folds(data, &[]);
        }
        let strata: Vec<usize> = data
            .treatment(0)
            .iter()
            .map(|&v| usize::from(v > 0.0))
            .collect();
        let splits = self.splitting.draw_stratified(&strata)?;
        self.fit_with_folds(data, &splits)
    }

    /// Fit on given sample splits (one `Vec<Fold>` per repetition; ignored and
    /// may be empty without covariates).
    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<ContinuousDidFit> {
        if data.clusters().is_some() {
            return Err(DmlError::InvalidArgument(
                "ContinuousDid does not support clustered data yet".into(),
            ));
        }
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "ContinuousDid needs exactly one dose variable".into(),
            ));
        }
        let (y, d) = (data.y(), data.treatment(0));
        let n = y.len();
        if d.iter().any(|&v| v < 0.0) {
            return Err(DmlError::InvalidArgument(
                "the dose must be non-negative (0 = untreated)".into(),
            ));
        }
        let treated: Vec<usize> = (0..n).filter(|&i| d[i] > 0.0).collect();
        let n1 = treated.len();
        if n1 == 0 || n1 == n {
            return Err(DmlError::InvalidArgument(
                "need both untreated (dose 0) and treated units".into(),
            ));
        }
        let t = self.trimming_threshold;
        if t.is_nan() || !(0.0..0.5).contains(&t) {
            return Err(DmlError::InvalidArgument(
                "trimming_threshold must be in [0, 0.5)".into(),
            ));
        }
        // Spline in the dose: knots at quantiles of the positive doses (contdid).
        let doses: Vec<f64> = treated.iter().map(|&i| d[i]).collect();
        let (lo, hi) = doses
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        if lo == hi {
            return Err(DmlError::InvalidArgument(
                "all treated units have the same dose".into(),
            ));
        }
        let interior: Vec<f64> = (1..=self.num_knots)
            .map(|k| np_quantile(&doses, k as f64 / (self.num_knots + 1) as f64))
            .collect();
        let spline = BSpline::new(self.degree, &interior, lo, hi);
        let k = spline.dim() + 1;
        if n1 <= k {
            return Err(DmlError::InvalidArgument(format!(
                "{n1} treated units for {k} spline coefficients"
            )));
        }
        let grid = match &self.grid {
            Some(g) => g.clone(),
            None => (1..20)
                .map(|j| np_quantile(&doses, j as f64 / 20.0))
                .collect(),
        };
        if grid.is_empty() || grid.iter().any(|&g| !(lo..=hi).contains(&g)) {
            return Err(DmlError::InvalidArgument(format!(
                "grid doses must lie within the range of the positive doses [{lo}, {hi}]"
            )));
        }

        // Rows B(D_i) of the treated units (zero for the untreated, never used).
        let mut basis = Array2::<f64>::zeros((n, k));
        for &i in &treated {
            basis.row_mut(i).assign(&spline.row(d[i]));
        }
        // Q = 1/n Σ T B B'.
        let mut q = Array2::<f64>::zeros((k, k));
        for &i in &treated {
            let b = basis.row(i);
            for a in 0..k {
                for c in 0..k {
                    q[[a, c]] += b[a] * b[c];
                }
            }
        }
        q /= n as f64;
        let q_inv = spd_inverse(&q)?;
        let grid_rows: Vec<Array1<f64>> = grid.iter().map(|&g| spline.row(g)).collect();
        let grid_derivs: Vec<Array1<f64>> = grid.iter().map(|&g| spline.deriv_row(g)).collect();
        let mean_deriv: Array1<f64> = treated
            .iter()
            .map(|&i| spline.deriv_row(d[i]))
            .fold(Array1::zeros(k), |a, b| a + b)
            / n1 as f64;

        let reps: Vec<Option<&Vec<Fold>>> = if data.n_features() == 0 {
            vec![None]
        } else {
            if self.ml_g.is_none() || self.ml_m.is_none() {
                return Err(DmlError::InvalidArgument(
                    "with covariates ContinuousDid needs learners (ContinuousDid::with_learners)"
                        .into(),
                ));
            }
            if splits.is_empty() {
                return Err(DmlError::InvalidArgument(
                    "need at least one repetition of sample splits".into(),
                ));
            }
            splits.iter().map(Some).collect()
        };
        let n_rep = reps.len();
        let n_grid = grid.len();
        // Influence-function scores: psi_a = −1, psi_b = θ + φ.
        let mut att = Array3::<f64>::zeros((n, n_grid, n_rep));
        let mut acrt = Array3::<f64>::zeros((n, n_grid, n_rep));
        let mut overall = Array3::<f64>::zeros((n, 2, n_rep));
        let mut stored: Vec<(String, Array3<f64>)> = ["ml_g", "ml_m"]
            .iter()
            .map(|name| (name.to_string(), Array3::zeros((n, n_rep, 1))))
            .collect();
        for (r, folds) in reps.iter().enumerate() {
            let (m0, p, bbar) = match folds {
                None => {
                    let controls: Vec<usize> = (0..n).filter(|&i| d[i] == 0.0).collect();
                    let c = controls.iter().map(|&i| y[i]).sum::<f64>() / controls.len() as f64;
                    let mean_b = treated
                        .iter()
                        .fold(Array1::<f64>::zeros(k), |a, &i| a + basis.row(i))
                        / n1 as f64;
                    let bbar = Array2::from_shape_fn((n, k), |(_, j)| mean_b[j]);
                    (
                        Array1::from_elem(n, c),
                        Array1::from_elem(n, n1 as f64 / n as f64),
                        bbar,
                    )
                }
                Some(folds) => self.nuisances(data, &basis, folds)?,
            };
            let w = p.mapv(|v| {
                let v = if data.n_features() == 0 {
                    v
                } else {
                    v.clamp(t, 1.0 - t)
                };
                v / (1.0 - v)
            });
            let resid = &y - &m0;
            // Σ [T B R − (1 − T) w b̄ R] and the per-unit terms.
            let mut terms = Array2::<f64>::zeros((n, k));
            for i in 0..n {
                if d[i] > 0.0 {
                    terms.row_mut(i).assign(&(&basis.row(i) * resid[i]));
                } else {
                    terms
                        .row_mut(i)
                        .assign(&(&bbar.row(i) * (-w[i] * resid[i])));
                }
            }
            let v = terms.sum_axis(Axis(0)) / n as f64;
            let beta = q_inv.dot(&v);
            // φ_i = Q⁻¹ [term_i − T_i B_i B_i'β].
            let mut phi = terms;
            for &i in &treated {
                let fitted = basis.row(i).dot(&beta);
                let row = &basis.row(i) * fitted;
                let mut target = phi.row_mut(i);
                target -= &row;
            }
            let phi = phi.dot(&q_inv.t());
            for (g, (row, deriv)) in grid_rows.iter().zip(&grid_derivs).enumerate() {
                let (a, s) = (row.dot(&beta), deriv.dot(&beta));
                att.slice_mut(s![.., g, r]).assign(&(phi.dot(row) + a));
                acrt.slice_mut(s![.., g, r]).assign(&(phi.dot(deriv) + s));
            }
            // Overall ATT = mean of B'β over the treated (= the doubly robust DiD
            // estimate); overall ACRT = mean of B'(D)'β over the treated.
            let fitted: Array1<f64> = (0..n).map(|i| basis.row(i).dot(&beta)).collect();
            let slopes: Array1<f64> = (0..n)
                .map(|i| {
                    if d[i] > 0.0 {
                        spline.deriv_row(d[i]).dot(&beta)
                    } else {
                        0.0
                    }
                })
                .collect();
            let att_o = treated.iter().map(|&i| fitted[i]).sum::<f64>() / n1 as f64;
            let acrt_o = treated.iter().map(|&i| slopes[i]).sum::<f64>() / n1 as f64;
            let scale = n as f64 / n1 as f64;
            for i in 0..n {
                let ti = f64::from(u8::from(d[i] > 0.0));
                // ATT: (n/n1)[T R − (1 − T) w R − T θ].
                let dr = if d[i] > 0.0 {
                    resid[i]
                } else {
                    -w[i] * resid[i]
                };
                overall[[i, 0, r]] = att_o + scale * (dr - ti * att_o);
                overall[[i, 1, r]] =
                    acrt_o + scale * ti * (slopes[i] - acrt_o) + phi.row(i).dot(&mean_deriv);
            }
            stored[0].1.slice_mut(s![.., r, 0]).assign(&m0);
            stored[1].1.slice_mut(s![.., r, 0]).assign(&p);
        }
        let n_folds = splits.first().map_or(1, |s| s.len());
        let names: Vec<String> = grid
            .iter()
            .map(|&g| {
                format!("{g:.6}")
                    .trim_end_matches('0')
                    .trim_end_matches('.')
                    .to_string()
            })
            .collect();
        let build = |model: &str, names: Vec<String>, psi_b: Array3<f64>| {
            let psi_a = Array3::from_elem(psi_b.dim(), -1.0);
            DmlFit::from_linear_scores(model, names, n_folds, psi_a, psi_b)
        };
        let mut att_fit = build(
            "Continuous DiD: ATT(d)",
            names.iter().map(|g| format!("ATT({g})")).collect(),
            att,
        )?;
        att_fit.predictions = stored.clone();
        let acrt_fit = build(
            "Continuous DiD: ACRT(d)",
            names.iter().map(|g| format!("ACRT({g})")).collect(),
            acrt,
        )?;
        let mut overall_fit = build("Continuous DiD", vec!["ATT".into(), "ACRT".into()], overall)?;
        overall_fit.predictions = stored;
        Ok(ContinuousDidFit {
            grid,
            spline,
            att: att_fit,
            acrt: acrt_fit,
            overall: overall_fit,
        })
    }

    /// Cross-fitted `m₀(X)`, `p(X)` and `b̄(X)` (rows `(n, k)`).
    fn nuisances(
        &self,
        data: &DmlData,
        basis: &Array2<f64>,
        folds: &[Fold],
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>)> {
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let n = y.len();
        let k = basis.ncols();
        let ml_g = self.ml_g.as_deref().expect("checked");
        let ml_m = self.ml_m.as_deref().expect("checked");
        let ml_nested = self.ml_nested.as_deref().unwrap_or(ml_g);
        let treated_ind: Array1<f64> = d.mapv(|v| f64::from(u8::from(v > 0.0)));
        let per_fold = map_folds(
            folds.len(),
            |f| -> Result<(Array1<f64>, Array1<f64>, Array2<f64>)> {
                let (train, test) = (&folds[f].train, &folds[f].test);
                let controls: Vec<usize> = train.iter().copied().filter(|&i| d[i] == 0.0).collect();
                let treated: Vec<usize> = train.iter().copied().filter(|&i| d[i] > 0.0).collect();
                let m0 = fit_regressor(ml_g, x, &controls, gather(y, &controls).view(), "outcome")?;
                let m0 = predict_rows(&*m0, x, test)?;
                let p = classify(ml_m, x, train, treated_ind.view(), test)?;
                let mut bbar = Array2::<f64>::ones((test.len(), k));
                for j in 1..k {
                    let target: Array1<f64> = treated.iter().map(|&i| basis[[i, j]]).collect();
                    let model = fit_regressor(ml_nested, x, &treated, target.view(), "spline")?;
                    bbar.column_mut(j).assign(&predict_rows(&*model, x, test)?);
                }
                Ok((m0, p, bbar))
            },
        )?;
        let mut m0 = Array1::from_elem(n, f64::NAN);
        let mut p = Array1::from_elem(n, f64::NAN);
        let mut bbar = Array2::from_elem((n, k), f64::NAN);
        for (fold, (m, pr, b)) in folds.iter().zip(per_fold) {
            for (j, &i) in fold.test.iter().enumerate() {
                m0[i] = m[j];
                p[i] = pr[j];
                bbar.row_mut(i).assign(&b.row(j));
            }
        }
        if m0
            .iter()
            .chain(p.iter())
            .chain(bbar.iter())
            .any(|v| !v.is_finite())
        {
            return Err(DmlError::InvalidArgument(
                "folds do not cover every observation, or a learner returned a non-finite value"
                    .into(),
            ));
        }
        Ok((m0, p, bbar))
    }
}

/// Result of [`ContinuousDid::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContinuousDidFit {
    grid: Vec<f64>,
    spline: BSpline,
    att: DmlFit,
    acrt: DmlFit,
    overall: DmlFit,
}

impl ContinuousDidFit {
    /// Doses of [`att`](Self::att) and [`acrt`](Self::acrt).
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// `ATT(d | d)` at the grid doses (predictions `"ml_g"` = `m₀(X)` and
    /// `"ml_m"` = `P(D > 0 | X)`, covariates only meaningful).
    pub fn att(&self) -> &DmlFit {
        &self.att
    }

    /// The average causal response `ACRT(d | d) = ∂ATT(d | d)/∂d` at the grid doses.
    pub fn acrt(&self) -> &DmlFit {
        &self.acrt
    }

    /// The overall `ATT` (average over the treated units' doses) and `ACRT`.
    pub fn overall(&self) -> &DmlFit {
        &self.overall
    }

    /// Number of spline coefficients (intercept included).
    pub fn n_coefficients(&self) -> usize {
        self.spline.dim() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    #[test]
    fn spline_partition_of_unity_and_derivative() {
        let s = BSpline::new(3, &[0.4, 0.7], 0.0, 1.0);
        assert_eq!(s.dim(), 5);
        for &x in &[0.0, 0.13, 0.4, 0.55, 0.99, 1.0] {
            let full = s.all(x, 3);
            assert!((full.iter().sum::<f64>() - 1.0).abs() < 1e-12, "x = {x}");
            // Central difference of the basis.
            if x > 0.01 && x < 0.99 {
                let h = 1e-6;
                let num = (&s.row(x + h) - &s.row(x - h)) / (2.0 * h);
                let ana = s.deriv_row(x);
                for j in 0..num.len() {
                    assert!((num[j] - ana[j]).abs() < 1e-5, "x = {x}, j = {j}");
                }
            }
        }
    }

    /// Covariates confound the dose and the trend; the doubly robust estimate
    /// with OLS / logit covers ATT(d) = d + d² and ACRT(d) = 1 + 2d.
    #[test]
    fn coverage_with_covariates() {
        let reps = 40;
        let (mut cover_att, mut cover_acrt) = (0, 0);
        for rep in 0..reps {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rep);
            let n = 2000;
            let x = Array2::from_shape_simple_fn((n, 2), || rng.sample::<f64, _>(StandardNormal));
            let mut d = Array1::zeros(n);
            let mut y = Array1::zeros(n);
            for i in 0..n {
                let treated = rng.gen::<f64>() < 1.0 / (1.0 + (-0.8 * x[[i, 0]]).exp());
                if treated {
                    d[i] = (0.2 + 0.6 * rng.gen::<f64>() + 0.1 * x[[i, 0]]).clamp(0.05, 1.0);
                }
                let trend = 1.0 + x[[i, 0]] - 0.5 * x[[i, 1]];
                y[i] = trend + d[i] + d[i] * d[i] + rng.sample::<f64, _>(StandardNormal);
            }
            let data = DmlData::new(y, d, x).unwrap();
            let fit = ContinuousDid::with_learners(Ols::new(), LogisticRegression::unpenalized())
                .grid(vec![0.5])
                .fit(&data)
                .unwrap();
            let ci = fit.att().confint(0.95).unwrap();
            cover_att += usize::from(ci[[0, 0]] <= 0.75 && 0.75 <= ci[[0, 1]]);
            let ci = fit.acrt().confint(0.95).unwrap();
            cover_acrt += usize::from(ci[[0, 0]] <= 2.0 && 2.0 <= ci[[0, 1]]);
        }
        let (a, b) = (
            cover_att as f64 / reps as f64,
            cover_acrt as f64 / reps as f64,
        );
        assert!(a > 0.85 && b > 0.85, "coverage ATT {a}, ACRT {b}");
    }
}

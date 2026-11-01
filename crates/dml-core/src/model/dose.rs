//! Dose-response curves of a continuous treatment: the kernel-localised DML
//! estimator of Colangelo and Lee (2023), "Double debiased machine learning
//! nonparametric inference with continuous treatments" (arXiv 2004.03036).

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use crate::compat::np_quantile;
use crate::crossfit::{map_folds, Fold, SampleSplitting};
use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::Regressor;

/// Kernel `k` of the localisation `K_h(D − t) = k((D − t)/h)/h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DoseKernel {
    /// `0.75·(1 − u²)` on `|u| ≤ 1` (the kernel of Colangelo and Lee's simulations).
    #[default]
    Epanechnikov,
    /// Standard normal density.
    Gaussian,
}

impl DoseKernel {
    fn eval(self, u: f64) -> f64 {
        match self {
            DoseKernel::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            DoseKernel::Gaussian => gaussian(u),
        }
    }
}

fn gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Bandwidth `h` of the kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Bandwidth {
    /// `h = c·σ_D·n^(−1/5)` with `σ_D` the sample standard deviation of the
    /// treatment (Colangelo and Lee's choice). Constants below the AMSE-optimal
    /// one undersmooth, so that the kernel bias is small next to the standard error.
    RuleOfThumb(f64),
    /// A fixed bandwidth in units of the treatment.
    Fixed(f64),
}

impl Default for Bandwidth {
    fn default() -> Self {
        Bandwidth::RuleOfThumb(1.0)
    }
}

/// Estimator of the generalised propensity score `f(t | X)`, the conditional
/// density of the treatment.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DensityModel {
    /// Colangelo and Lee's MultiGPS: `f(t | X) = E[φ((D − t)/h₁) | X]/h₁` with
    /// `φ` the normal density, learned by `ml_m` separately for every
    /// evaluation point. Needs no distributional assumption; `h₁` is the
    /// bandwidth `h` unless set with [`DoseResponse::density_bandwidth`].
    #[default]
    KernelRegression,
    /// Normal location model `D | X ~ N(m(X), σ²)`: `ml_m` learns `m(X)` once
    /// per fold, and `σ²` is the mean squared cross-fitted residual `D − m̂(X)`.
    Normal,
}

/// Average dose-response function `β(t) = E[Y(t)]` of a continuous treatment
/// at a grid of treatment values, with the doubly robust kernel score of
/// Colangelo and Lee (2023):
///
/// ```text
/// ψ(t) = γ(t, X) + K_h(D − t)/f(t | X) · (Y − γ(t, X)) − β(t),
/// ```
///
/// where `γ(t, X) = E[Y | D = t, X]` is learned by `ml_g` on `(D, X)` (the
/// treatment in the first column) and the generalised propensity score
/// `f(t | X)` by `ml_m` (see [`DensityModel`]). The nuisances are cross-fitted,
/// and the standard error is that of the estimated influence function, which
/// converges at the nonparametric rate `√(nh)`.
///
/// All evaluation points share the sample splits, so the curve has joint
/// inference: uniform confidence bands from the multiplier bootstrap
/// ([`Bootstrap`](crate::Bootstrap)) and contrasts with correct standard errors
/// ([`DoseResponseFit::contrast`]). With [`partial_effects`](Self::partial_effects)
/// the fit also estimates `∂β(t)/∂t` by `(β̂(t + η/2) − β̂(t − η/2))/η`.
///
/// Small density estimates are raised to `density_floor / σ_D` (default
/// `0.01 / σ_D`, i.e. 0.01 on the scale of the standardised treatment).
///
/// ```
/// use dml_core::{DmlData, DoseResponse, Ols};
/// use ndarray::{Array1, Array2};
///
/// let n = 2000;
/// let x = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 7) % 13) as f64 / 6.0 - 1.0);
/// let d: Array1<f64> = (0..n).map(|i| 0.5 * x[[i, 0]] + ((i * 31) % 17) as f64 / 8.0 - 1.0).collect();
/// let y: Array1<f64> = (0..n).map(|i| 2.0 * d[i] + x[[i, 0]] + ((i * 11) % 5) as f64 / 4.0).collect();
/// let data = DmlData::new(y, d, x)?;
///
/// let fit = DoseResponse::new(Ols::new(), Ols::new())
///     .grid(vec![-0.5, 0.0, 0.5])
///     .fit(&data)?;
/// let curve = fit.curve();
/// // β(t) is linear in t with slope 2.
/// assert!((curve.coef()[2] - curve.coef()[0] - 2.0).abs() < 0.3);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
pub struct DoseResponse {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Regressor>,
    grid: Option<Vec<f64>>,
    kernel: DoseKernel,
    bandwidth: Bandwidth,
    density: DensityModel,
    density_bandwidth: Option<f64>,
    density_floor: f64,
    partial_effects: bool,
    eta: Option<f64>,
    splitting: SampleSplitting,
}

impl DoseResponse {
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    /// Like [`DoseResponse::new`] for boxed trait objects.
    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Box<dyn Regressor>) -> Self {
        Self {
            ml_g,
            ml_m,
            grid: None,
            kernel: DoseKernel::default(),
            bandwidth: Bandwidth::default(),
            density: DensityModel::default(),
            density_bandwidth: None,
            density_floor: 1e-2,
            partial_effects: false,
            eta: None,
            splitting: SampleSplitting::default(),
        }
    }

    /// Treatment values at which to estimate `β(t)` (default: the deciles
    /// 0.1, …, 0.9 of the treatment).
    pub fn grid(mut self, grid: Vec<f64>) -> Self {
        self.grid = Some(grid);
        self
    }

    /// Kernel of the localisation (default Epanechnikov).
    pub fn kernel(mut self, kernel: DoseKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Bandwidth of the localisation (default `RuleOfThumb(1.0)`).
    pub fn bandwidth(mut self, bandwidth: Bandwidth) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Estimator of the generalised propensity score (default `KernelRegression`).
    pub fn density(mut self, density: DensityModel) -> Self {
        self.density = density;
        self
    }

    /// Bandwidth `h₁` of [`DensityModel::KernelRegression`] (default: `h`).
    pub fn density_bandwidth(mut self, h1: f64) -> Self {
        self.density_bandwidth = Some(h1);
        self
    }

    /// Lower bound on the density of the standardised treatment (default 0.01).
    pub fn density_floor(mut self, threshold: f64) -> Self {
        self.density_floor = threshold;
        self
    }

    /// Former name of [`density_floor`](Self::density_floor).
    #[deprecated(since = "0.32.0", note = "renamed to `density_floor`")]
    pub fn trimming_threshold(self, threshold: f64) -> Self {
        self.density_floor(threshold)
    }

    /// Also estimate the partial effects `∂β(t)/∂t` at the grid points.
    pub fn partial_effects(mut self, yes: bool) -> Self {
        self.partial_effects = yes;
        self
    }

    /// Step `η` of the partial effects (default `h·n^(−1/6)`, which shrinks
    /// faster than `h` as Colangelo and Lee require).
    pub fn eta(mut self, eta: f64) -> Self {
        self.eta = Some(eta);
        self.partial_effects = true;
        self
    }

    /// Number of cross-fitting folds (default 5).
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
    pub fn fit(&self, data: &DmlData) -> Result<DoseResponseFit> {
        let splits = self.splitting.draw(data.n_obs())?;
        self.fit_with_folds(data, &splits)
    }

    /// Fit on given sample splits (one `Vec<Fold>` per repetition).
    pub fn fit_with_folds(&self, data: &DmlData, splits: &[Vec<Fold>]) -> Result<DoseResponseFit> {
        if data.clusters().is_some() {
            return Err(DmlError::InvalidArgument(
                "DoseResponse does not support clustered data yet".into(),
            ));
        }
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "DoseResponse needs exactly one (continuous) treatment".into(),
            ));
        }
        if splits.is_empty() {
            return Err(DmlError::InvalidArgument(
                "need at least one repetition of sample splits".into(),
            ));
        }
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let n = y.len();
        let sd = sample_sd(d);
        if sd.is_nan() || sd <= 0.0 {
            return Err(DmlError::InvalidArgument(
                "the treatment is constant".into(),
            ));
        }
        let h = match self.bandwidth {
            Bandwidth::RuleOfThumb(c) => c * sd * (n as f64).powf(-0.2),
            Bandwidth::Fixed(h) => h,
        };
        positive("bandwidth", h)?;
        let h1 = self.density_bandwidth.unwrap_or(h);
        positive("density bandwidth", h1)?;
        if self.density_floor.is_nan() || self.density_floor < 0.0 {
            return Err(DmlError::InvalidArgument(
                "density_floor must be non-negative".into(),
            ));
        }
        let grid = match &self.grid {
            Some(g) => g.clone(),
            None => {
                let values = d.to_vec();
                (1..10)
                    .map(|k| np_quantile(&values, k as f64 / 10.0))
                    .collect()
            }
        };
        check_grid(&grid)?;
        let eta = if self.partial_effects {
            let eta = self.eta.unwrap_or(h * (n as f64).powf(-1.0 / 6.0));
            positive("eta", eta)?;
            Some(eta)
        } else {
            None
        };
        // Evaluation points: the grid, then t − η/2 and t + η/2 for the partial effects.
        let mut points = grid.clone();
        if let Some(eta) = eta {
            points.extend(grid.iter().map(|t| t - eta / 2.0));
            points.extend(grid.iter().map(|t| t + eta / 2.0));
        }

        let settings = Settings {
            h1,
            floor: self.density_floor / sd,
        };
        let n_points = points.len();
        let n_rep = splits.len();
        let mut psi_b = Array3::<f64>::zeros((n, n_points, n_rep));
        let mut g_all = Array3::<f64>::zeros((n, n_rep, n_points));
        let mut f_all = Array3::<f64>::zeros((n, n_rep, n_points));
        for (r, folds) in splits.iter().enumerate() {
            let (g, f) = self.nuisances(y, d, x, folds, &points, &settings)?;
            for (p, &t) in points.iter().enumerate() {
                for i in 0..n {
                    let (gi, fi) = (g[[i, p]], f[[i, p]]);
                    let k = self.kernel.eval((d[i] - t) / h) / h;
                    psi_b[[i, p, r]] = gi + k / fi * (y[i] - gi);
                }
            }
            g_all.index_axis_mut(Axis(1), r).assign(&g);
            f_all.index_axis_mut(Axis(1), r).assign(&f);
        }
        let n_folds = splits[0].len();
        let k = grid.len();
        let first = |a: &Array3<f64>| a.slice(ndarray::s![.., ..k, ..]).to_owned();
        let psi_a = Array3::<f64>::from_elem((n, n_points, n_rep), -1.0);
        let mut curve = DmlFit::from_linear_scores(
            "Dose response",
            grid.iter().map(|&t| point_name(t)).collect(),
            n_folds,
            first(&psi_a),
            first(&psi_b),
        )?;
        // Predictions are stored (n, n_rep, k), one column per grid point.
        let grid_cols = |a: &Array3<f64>| a.slice(ndarray::s![.., .., ..k]).to_owned();
        curve.predictions = vec![
            ("ml_g".into(), grid_cols(&g_all)),
            ("density".into(), grid_cols(&f_all)),
        ];
        // The partial effects combine the scores at t ± η/2 (points k..3k).
        let all = match eta {
            Some(_) => Some(DmlFit::from_linear_scores(
                "Dose response",
                points.iter().map(|&t| point_name(t)).collect(),
                n_folds,
                psi_a,
                psi_b,
            )?),
            None => None,
        };
        let partial = eta.zip(all).map(|(eta, all)| {
            let w = Array2::from_shape_fn((k, n_points), |(i, j)| {
                if j == k + i {
                    -1.0 / eta
                } else if j == 2 * k + i {
                    1.0 / eta
                } else {
                    0.0
                }
            });
            all.combine(
                "Dose response: partial effect".into(),
                grid.iter()
                    .map(|&t| format!("d'({})", point_name(t)))
                    .collect(),
                &w,
            )
        });
        Ok(DoseResponseFit {
            grid,
            bandwidth: h,
            eta,
            curve,
            partial,
        })
    }

    /// Out-of-fold `γ̂(t, X_i)` and floored `f̂(t | X_i)`, shape `(n, points)`.
    fn nuisances(
        &self,
        y: ArrayView1<'_, f64>,
        d: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
        folds: &[Fold],
        points: &[f64],
        s: &Settings,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let n = y.len();
        let p = x.ncols();
        let mut w = Array2::<f64>::zeros((n, p + 1));
        w.column_mut(0).assign(&d);
        w.slice_mut(ndarray::s![.., 1..]).assign(&x);

        // Normal model: m̂(X) out of fold, then σ² from the cross-fitted residuals.
        let normal_mean = match self.density {
            DensityModel::Normal => {
                let per_fold = map_folds(folds.len(), |k| -> Result<Array1<f64>> {
                    let fold = &folds[k];
                    let mut m = self.ml_m.fresh();
                    m.fit(
                        x.select(Axis(0), &fold.train).view(),
                        gather(d, &fold.train).view(),
                    )?;
                    m.predict(x.select(Axis(0), &fold.test).view())
                })?;
                let m_hat = scatter(n, folds, &per_fold)?;
                let resid = &d - &m_hat;
                let sigma = (resid.dot(&resid) / n as f64).sqrt();
                if sigma.is_nan() || sigma <= 0.0 {
                    return Err(DmlError::InvalidArgument(
                        "ml_m predicts the treatment exactly; the normal density is degenerate"
                            .into(),
                    ));
                }
                Some((m_hat, sigma))
            }
            DensityModel::KernelRegression => None,
        };

        let per_fold = map_folds(folds.len(), |k| -> Result<(Array2<f64>, Array2<f64>)> {
            let fold = &folds[k];
            let m_test = fold.test.len();
            let x_train = x.select(Axis(0), &fold.train);
            let x_test = x.select(Axis(0), &fold.test);
            let mut g = self.ml_g.fresh();
            g.fit(
                w.select(Axis(0), &fold.train).view(),
                gather(y, &fold.train).view(),
            )?;
            let mut w_test = w.select(Axis(0), &fold.test);
            let mut g_out = Array2::<f64>::zeros((m_test, points.len()));
            let mut f_out = Array2::<f64>::zeros((m_test, points.len()));
            for (j, &t) in points.iter().enumerate() {
                w_test.column_mut(0).fill(t);
                g_out
                    .column_mut(j)
                    .assign(&check_len(g.predict(w_test.view())?, m_test)?);
                let f: Array1<f64> = match &normal_mean {
                    Some((m_hat, sigma)) => fold
                        .test
                        .iter()
                        .map(|&i| gaussian((t - m_hat[i]) / sigma) / sigma)
                        .collect(),
                    None => {
                        let target: Array1<f64> = fold
                            .train
                            .iter()
                            .map(|&i| gaussian((d[i] - t) / s.h1))
                            .collect();
                        let mut m = self.ml_m.fresh();
                        m.fit(x_train.view(), target.view())?;
                        check_len(m.predict(x_test.view())?, m_test)? / s.h1
                    }
                };
                f_out.column_mut(j).assign(&f.mapv(|v: f64| v.max(s.floor)));
            }
            Ok((g_out, f_out))
        })?;

        let mut g = Array2::from_elem((n, points.len()), f64::NAN);
        let mut f = Array2::from_elem((n, points.len()), f64::NAN);
        for (fold, (g_k, f_k)) in folds.iter().zip(per_fold) {
            for (row, &i) in fold.test.iter().enumerate() {
                g.row_mut(i).assign(&g_k.row(row));
                f.row_mut(i).assign(&f_k.row(row));
            }
        }
        if g.iter().chain(f.iter()).any(|v| !v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "folds do not cover every observation, or a learner returned a non-finite value"
                    .into(),
            ));
        }
        Ok((g, f))
    }
}

/// Density settings: bandwidth `h₁` of the kernel regression, density floor.
struct Settings {
    h1: f64,
    floor: f64,
}

/// Result of [`DoseResponse::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoseResponseFit {
    grid: Vec<f64>,
    bandwidth: f64,
    eta: Option<f64>,
    curve: DmlFit,
    partial: Option<DmlFit>,
}

impl DoseResponseFit {
    /// Treatment values of the estimates.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Bandwidth `h` used.
    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }

    /// Step `η` of the partial effects, if estimated.
    pub fn eta(&self) -> Option<f64> {
        self.eta
    }

    /// `β̂(t)` at the grid points, one estimate per point (predictions
    /// `"ml_g"` = `γ̂(t, X_i)` and `"density"` = `f̂(t | X_i)`).
    pub fn curve(&self) -> &DmlFit {
        &self.curve
    }

    /// `∂β(t)/∂t` at the grid points, if requested with
    /// [`DoseResponse::partial_effects`].
    pub fn partial_effects(&self) -> Option<&DmlFit> {
        self.partial.as_ref()
    }

    /// `β(t) − β(t_ref)` for every other grid point `t`, where `t_ref` must be
    /// a grid point.
    pub fn contrast(&self, reference: f64) -> Result<DmlFit> {
        let r = self
            .grid
            .iter()
            .position(|&t| t == reference)
            .ok_or_else(|| DmlError::InvalidArgument(format!("{reference} is not a grid point")))?;
        let k = self.grid.len();
        let others: Vec<usize> = (0..k).filter(|&j| j != r).collect();
        if others.is_empty() {
            return Err(DmlError::InvalidArgument(
                "a contrast needs at least two grid points".into(),
            ));
        }
        let mut w = Array2::<f64>::zeros((others.len(), k));
        for (row, &j) in others.iter().enumerate() {
            w[[row, j]] = 1.0;
            w[[row, r]] = -1.0;
        }
        let names = others
            .iter()
            .map(|&j| format!("{} vs {}", point_name(self.grid[j]), point_name(reference)))
            .collect();
        Ok(self
            .curve
            .combine("Dose response: contrast".into(), names, &w))
    }
}

fn sample_sd(v: ArrayView1<'_, f64>) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let mean = v.sum() / n as f64;
    (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

fn positive(what: &str, v: f64) -> Result<()> {
    if v > 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(DmlError::InvalidArgument(format!(
            "{what} must be positive and finite, got {v}"
        )))
    }
}

fn check_grid(grid: &[f64]) -> Result<()> {
    if grid.is_empty() || grid.iter().any(|t| !t.is_finite()) {
        return Err(DmlError::InvalidArgument(
            "grid needs at least one finite treatment value".into(),
        ));
    }
    let names: std::collections::HashSet<String> = grid.iter().map(|&t| point_name(t)).collect();
    if names.len() != grid.len() {
        return Err(DmlError::InvalidArgument(
            "grid values must be distinct".into(),
        ));
    }
    Ok(())
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

fn scatter(n: usize, folds: &[Fold], per_fold: &[Array1<f64>]) -> Result<Array1<f64>> {
    let mut out = Array1::from_elem(n, f64::NAN);
    for (fold, v) in folds.iter().zip(per_fold) {
        let v = check_len(v.clone(), fold.test.len())?;
        for (j, &i) in fold.test.iter().enumerate() {
            out[i] = v[j];
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(DmlError::InvalidArgument(
            "folds do not cover every observation, or ml_m returned a non-finite value".into(),
        ));
    }
    Ok(out)
}

/// Label of a grid point: six significant digits, without trailing zeros.
fn point_name(t: f64) -> String {
    if t == 0.0 {
        return "0".into();
    }
    let digits = (6 - 1 - t.abs().log10().floor() as i32).max(0) as usize;
    let s = format!("{t:.digits$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Ols;
    use ndarray::Array1;
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    #[test]
    fn point_names() {
        assert_eq!(point_name(0.5), "0.5");
        assert_eq!(point_name(-1.25), "-1.25");
        assert_eq!(point_name(1.0 / 3.0), "0.333333");
        assert_eq!(point_name(12345.678), "12345.7");
        assert_eq!(point_name(0.0), "0");
    }

    /// `γ̂` known exactly (OLS on a linear outcome) and a constant density:
    /// the estimate is the sample mean of `γ(t, X)` plus the kernel-weighted
    /// residuals, computed here by hand.
    #[test]
    fn score_by_hand() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let n = 300;
        let x = Array2::from_shape_fn((n, 2), |_| rng.sample::<f64, _>(StandardNormal));
        let d: Array1<f64> = (0..n)
            .map(|i| x[[i, 0]] + rng.sample::<f64, _>(StandardNormal))
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| d[i] + x[[i, 1]] + rng.sample::<f64, _>(StandardNormal))
            .collect();
        let data = DmlData::new(y.clone(), d.clone(), x.clone()).unwrap();
        let model = DoseResponse::new(Ols::new(), Ols::new())
            .density(DensityModel::Normal)
            .grid(vec![0.0, 0.7])
            .bandwidth(Bandwidth::Fixed(0.4));
        let fit = model.fit(&data).unwrap();
        let g = fit.curve().predictions("ml_g").unwrap();
        let f = fit.curve().predictions("density").unwrap();
        for (j, &t) in [0.0, 0.7].iter().enumerate() {
            let mut sum = 0.0;
            for i in 0..n {
                let u: f64 = (d[i] - t) / 0.4;
                let k = if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u) / 0.4
                } else {
                    0.0
                };
                sum += g[[i, 0, j]] + k / f[[i, 0, j]] * (y[i] - g[[i, 0, j]]);
            }
            assert!((fit.curve().coef()[j] - sum / n as f64).abs() < 1e-12);
        }
        // Contrast = difference with matching standard error.
        let c = fit.contrast(0.0).unwrap();
        assert!((c.coef()[0] - (fit.curve().coef()[1] - fit.curve().coef()[0])).abs() < 1e-12);
        assert_eq!(c.treatment_names(), ["0.7 vs 0"]);
    }

    /// Monte Carlo: Colangelo and Lee's design with 10 covariates; the 95%
    /// interval for β(0) = 0 and the partial effect θ(0) = 1.2 should cover.
    #[test]
    fn coverage_colangelo_lee_design() {
        let (reps, n, p) = (60, 1000, 10);
        let (mut cover_b, mut cover_p) = (0, 0);
        for rep in 0..reps {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(100 + rep);
            let theta: Vec<f64> = (1..=p).map(|j| 1.0 / (j * j) as f64).collect();
            let x = Array2::from_shape_fn((n, p), |_| rng.sample::<f64, _>(StandardNormal));
            let xt: Array1<f64> = x.dot(&Array1::from(theta));
            let d: Array1<f64> = (0..n)
                .map(|i| {
                    crate::stats::norm_cdf(3.0 * xt[i])
                        + 0.75 * rng.sample::<f64, _>(StandardNormal)
                        - 0.5
                })
                .collect();
            let y: Array1<f64> = (0..n)
                .map(|i| {
                    let e: f64 = rng.sample(StandardNormal);
                    1.2 * d[i]
                        + 1.2 * xt[i]
                        + d[i] * d[i]
                        + d[i] * x[[i, 0]]
                        + e * (0.5 + crate::stats::norm_cdf(x[[i, 0]])).sqrt()
                })
                .collect();
            // OLS on (D, D², D·X1, X) is correctly specified for γ.
            let data = DmlData::new(y, d, x).unwrap();
            let fit = DoseResponse::new(QuadOls, Ols::new())
                .grid(vec![0.0])
                .partial_effects(true)
                .density(DensityModel::Normal)
                .fit(&data)
                .unwrap();
            let ci = fit.curve().confint(0.95).unwrap();
            cover_b += usize::from(ci[[0, 0]] <= 0.0 && 0.0 <= ci[[0, 1]]);
            let ci = fit.partial_effects().unwrap().confint(0.95).unwrap();
            cover_p += usize::from(ci[[0, 0]] <= 1.2 && 1.2 <= ci[[0, 1]]);
        }
        let (cb, cp) = (cover_b as f64 / reps as f64, cover_p as f64 / reps as f64);
        assert!(cb > 0.85, "beta coverage {cb}");
        assert!(cp > 0.85, "partial effect coverage {cp}");
    }

    /// OLS of `Y` on `[D, D², D·X1, X]` given `W = [D, X]`.
    struct QuadOls;

    impl QuadOls {
        fn features(w: ArrayView2<'_, f64>) -> Array2<f64> {
            let (n, k) = w.dim();
            let d = w.column(0).to_owned();
            let mut f = Array2::<f64>::zeros((n, k + 2));
            f.column_mut(0).assign(&d);
            f.column_mut(1).assign(&(&d * &d));
            f.column_mut(2).assign(&(&d * &w.column(1)));
            f.slice_mut(ndarray::s![.., 3..])
                .assign(&w.slice(ndarray::s![.., 1..]));
            f
        }
    }

    struct FittedQuad(Ols);

    impl Regressor for QuadOls {
        fn fit(&mut self, _x: ArrayView2<'_, f64>, _y: ArrayView1<'_, f64>) -> Result<()> {
            unreachable!("QuadOls is only used through fresh()")
        }
        fn predict(&self, _x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
            unreachable!()
        }
        fn fresh(&self) -> Box<dyn Regressor> {
            Box::new(FittedQuad(Ols::new()))
        }
    }

    impl Regressor for FittedQuad {
        fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
            self.0.fit(QuadOls::features(x).view(), y)
        }
        fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
            self.0.predict(QuadOls::features(x).view())
        }
        fn fresh(&self) -> Box<dyn Regressor> {
            Box::new(FittedQuad(Ols::new()))
        }
    }
}

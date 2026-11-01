//! Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg,
//! Imbens and Wager 2021, "Synthetic difference-in-differences", American
//! Economic Review 111(12), 4088–4118), with synthetic control and plain DiD
//! as special cases: a port of the authors' R package `synthdid` (0.0.9) that
//! reproduces its estimates, weights and standard errors.

use ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis};

use crate::compat::RRng;
use crate::crossfit::map_folds;
use crate::error::{DmlError, Result};

/// A balanced panel for [`SyntheticDid`]: outcomes `y` with one row per unit
/// and one column per period, the `n0` control units first and the `t0`
/// pre-treatment periods first (the layout of `synthdid::panel.matrices`);
/// the remaining units are treated in the remaining periods. Optional
/// time-varying covariates `x` have shape `(units, periods, covariates)`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SynthPanel {
    y: Array2<f64>,
    x: Array3<f64>,
    n0: usize,
    t0: usize,
}

impl SynthPanel {
    pub fn new(y: Array2<f64>, n0: usize, t0: usize) -> Result<Self> {
        let (n, t) = y.dim();
        if n0 == 0 || n0 >= n || t0 == 0 || t0 >= t {
            return Err(DmlError::InvalidArgument(format!(
                "need 0 < n0 < {n} control units and 0 < t0 < {t} pre-treatment periods, got n0 = {n0}, t0 = {t0}"
            )));
        }
        if !y.iter().all(|v| v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "outcomes contain NaN or infinite values".into(),
            ));
        }
        Ok(Self {
            x: Array3::zeros((n, t, 0)),
            y,
            n0,
            t0,
        })
    }

    /// Attach time-varying covariates, shape `(units, periods, covariates)`.
    pub fn with_covariates(mut self, x: Array3<f64>) -> Result<Self> {
        let (n, t) = self.y.dim();
        if x.dim().0 != n || x.dim().1 != t {
            return Err(DmlError::DimensionMismatch(format!(
                "covariates have shape {:?}, need ({n}, {t}, k)",
                x.dim()
            )));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "covariates contain NaN or infinite values".into(),
            ));
        }
        self.x = x;
        Ok(self)
    }

    pub fn n_units(&self) -> usize {
        self.y.nrows()
    }

    pub fn n_periods(&self) -> usize {
        self.y.ncols()
    }

    pub fn n_control(&self) -> usize {
        self.n0
    }

    pub fn n_pre(&self) -> usize {
        self.t0
    }

    fn rows(&self, rows: &[usize], n0: usize) -> Self {
        Self {
            y: self.y.select(Axis(0), rows),
            x: self.x.select(Axis(0), rows),
            n0,
            t0: self.t0,
        }
    }

    /// `Y − Σ_k β_k X_k`.
    fn adjusted(&self, beta: &[f64]) -> Array2<f64> {
        let mut out = self.y.clone();
        for (k, &b) in beta.iter().enumerate() {
            out.scaled_add(-b, &self.x.index_axis(Axis(2), k));
        }
        out
    }
}

/// Which estimator of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum SynthEstimator {
    /// Unit and time weights (`synthdid_estimate`).
    #[default]
    SyntheticDid,
    /// Unit weights without intercept, no time weights (`sc_estimate`).
    SyntheticControl,
    /// Uniform weights (`did_estimate`).
    Did,
}

/// Standard error method (`vcov.synthdid_estimate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SynthSe {
    /// Leave-one-unit-out jackknife with the weights held fixed (needs at least
    /// two treated units).
    Jackknife,
    /// Placebo: random control units are relabelled as treated (needs more
    /// controls than treated units). `seed` is R's `set.seed`.
    Placebo { replications: usize, seed: i32 },
    /// Unit bootstrap (needs at least two treated units). `seed` is R's
    /// `set.seed`.
    Bootstrap { replications: usize, seed: i32 },
}

/// Solver settings and regularisation (`synthdid_estimate`'s `opts`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Opts {
    zeta_omega: f64,
    zeta_lambda: f64,
    omega_intercept: bool,
    lambda_intercept: bool,
    update_omega: bool,
    update_lambda: bool,
    min_decrease: f64,
    max_iter: usize,
    sparsify: bool,
    max_iter_pre_sparsify: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Weights {
    omega: Option<Vec<f64>>,
    lambda: Option<Vec<f64>>,
    beta: Vec<f64>,
}

/// Synthetic difference-in-differences and its relatives for a block design:
/// `n1` units treated from period `t0 + 1` on, `n0` never-treated controls.
///
/// The estimate is a weighted DiD,
///
/// ```text
/// τ̂ = (ȳ_treated,post − Σ_t λ_t ȳ_treated,t) − Σ_i ω_i (ȳ_i,post − Σ_t λ_t y_i,t),
/// ```
///
/// with unit weights `ω` on the controls chosen so that their (intercept-
/// shifted) pre-treatment path matches the treated average, with an `L2`
/// penalty `ζ_ω² T0 ‖ω‖²` (`ζ_ω = (N1 T1)^{1/4} σ̂`, `σ̂` the standard
/// deviation of the controls' first differences), and time weights `λ` on the
/// pre-treatment periods chosen so that they predict the controls'
/// post-treatment average. Both are fitted by Frank–Wolfe on the simplex,
/// sparsified (weights below a quarter of the largest are dropped) and
/// refitted, exactly as `synthdid`. With covariates the outcomes are first
/// adjusted by `Σ_k β_k X_k`, with `β` fitted jointly with the weights.
///
/// [`SynthEstimator::SyntheticControl`] (no time weights, no intercept,
/// negligible penalty) and [`SynthEstimator::Did`] (uniform weights) use the
/// same machinery. Standard errors: jackknife, placebo or bootstrap
/// ([`SynthFit::se`]).
///
/// ```
/// use dml_core::{SynthPanel, SyntheticDid};
/// use ndarray::Array2;
///
/// // 20 controls and 2 treated units over 10 periods; effect 3 after period 7.
/// let y = Array2::from_shape_fn((22, 10), |(i, t)| {
///     let unit = (i % 5) as f64 + 0.1 * ((i * 7) % 3) as f64;
///     let time = 0.5 * t as f64 + ((i * 13 + t * 7) % 11) as f64 / 20.0;
///     unit + time + if i >= 20 && t >= 7 { 3.0 } else { 0.0 }
/// });
/// let fit = SyntheticDid::new().fit(&SynthPanel::new(y, 20, 7)?)?;
/// assert!((fit.estimate() - 3.0).abs() < 0.3);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct SyntheticDid {
    estimator: SynthEstimator,
    noise_level: Option<f64>,
    eta_omega: Option<f64>,
    eta_lambda: Option<f64>,
    min_decrease: Option<f64>,
    max_iter: Option<usize>,
    sparsify: Option<bool>,
}

impl SyntheticDid {
    pub fn new() -> Self {
        Self::default()
    }

    /// Synthetic DiD (default), synthetic control or plain DiD.
    pub fn estimator(mut self, estimator: SynthEstimator) -> Self {
        self.estimator = estimator;
        self
    }

    /// Noise level `σ̂` that scales the penalties and the stopping rule
    /// (default: standard deviation of the controls' pre-treatment first
    /// differences).
    pub fn noise_level(mut self, sigma: f64) -> Self {
        self.noise_level = Some(sigma);
        self
    }

    /// Unit-weight penalty `η_ω` (`ζ_ω = η_ω σ̂`; default `(N1 T1)^{1/4}`, for
    /// synthetic control 1e-6).
    pub fn eta_omega(mut self, eta: f64) -> Self {
        self.eta_omega = Some(eta);
        self
    }

    /// Time-weight penalty `η_λ` (`ζ_λ = η_λ σ̂`; default 1e-6).
    pub fn eta_lambda(mut self, eta: f64) -> Self {
        self.eta_lambda = Some(eta);
        self
    }

    /// Stop when the objective decreases by less than `min_decrease²`
    /// (default `1e-5 σ̂`).
    pub fn min_decrease(mut self, value: f64) -> Self {
        self.min_decrease = Some(value);
        self
    }

    /// Maximum Frank–Wolfe iterations (default 10000).
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    /// Sparsify the weights after 100 iterations and refit (default `true`).
    pub fn sparsify(mut self, yes: bool) -> Self {
        self.sparsify = Some(yes);
        self
    }

    pub fn fit(&self, panel: &SynthPanel) -> Result<SynthFit> {
        let (n, t) = panel.y.dim();
        let (n0, t0) = (panel.n0, panel.t0);
        let (n1, t1) = (n - n0, t - t0);
        let noise = match self.noise_level {
            Some(v) => v,
            None => {
                let diffs: Vec<f64> = (0..n0)
                    .flat_map(|i| (1..t0).map(move |s| (i, s)))
                    .map(|(i, s)| panel.y[[i, s]] - panel.y[[i, s - 1]])
                    .collect();
                sample_sd(&diffs)
            }
        };
        if !(noise.is_finite() && noise >= 0.0) {
            return Err(DmlError::InvalidArgument(
                "cannot estimate the noise level (need at least 2 pre-treatment periods), pass noise_level"
                    .into(),
            ));
        }
        let (default_eta_omega, omega_intercept) = match self.estimator {
            SynthEstimator::SyntheticControl => (1e-6, false),
            _ => (((n1 * t1) as f64).powf(0.25), true),
        };
        let eta_omega = self.eta_omega.unwrap_or(default_eta_omega);
        let eta_lambda = self.eta_lambda.unwrap_or(1e-6);
        let init = match self.estimator {
            SynthEstimator::SyntheticDid => Weights {
                omega: None,
                lambda: None,
                beta: Vec::new(),
            },
            SynthEstimator::SyntheticControl => Weights {
                omega: None,
                lambda: Some(vec![0.0; t0]),
                beta: Vec::new(),
            },
            SynthEstimator::Did => Weights {
                omega: Some(vec![1.0 / n0 as f64; n0]),
                lambda: Some(vec![1.0 / t0 as f64; t0]),
                beta: Vec::new(),
            },
        };
        let opts = Opts {
            zeta_omega: eta_omega * noise,
            zeta_lambda: eta_lambda * noise,
            omega_intercept,
            lambda_intercept: true,
            update_omega: init.omega.is_none(),
            update_lambda: init.lambda.is_none(),
            min_decrease: self.min_decrease.unwrap_or(1e-5 * noise),
            max_iter: self.max_iter.unwrap_or(10_000),
            sparsify: self.sparsify.unwrap_or(true),
            max_iter_pre_sparsify: 100,
        };
        let (estimate, weights) = estimate(panel, &opts, init);
        Ok(SynthFit {
            estimator: self.estimator,
            estimate,
            omega: weights.omega.clone().expect("set"),
            lambda: weights.lambda.clone().expect("set"),
            beta: weights.beta.clone(),
            noise_level: noise,
            panel: panel.clone(),
            opts,
        })
    }
}

/// `synthdid_estimate` given options and (initial or fixed) weights.
fn estimate(panel: &SynthPanel, opts: &Opts, mut w: Weights) -> (f64, Weights) {
    let (n, t) = panel.y.dim();
    let (n0, t0) = (panel.n0, panel.t0);
    let (n1, t1) = (n - n0, t - t0);
    let k = panel.x.dim().2;
    if k == 0 {
        let yc = collapsed_form(panel.y.view(), n0, t0);
        if opts.update_lambda {
            let a = yc.slice(s![..n0, ..]).to_owned();
            let mut lambda = sc_weight_fw(
                &a,
                opts.zeta_lambda,
                opts.lambda_intercept,
                w.lambda.take(),
                opts.min_decrease,
                pre_iter(opts),
            );
            if opts.sparsify {
                lambda = sc_weight_fw(
                    &a,
                    opts.zeta_lambda,
                    opts.lambda_intercept,
                    Some(sparsify(&lambda)),
                    opts.min_decrease,
                    opts.max_iter,
                );
            }
            w.lambda = Some(lambda);
        }
        if opts.update_omega {
            let a = yc.slice(s![.., ..t0]).t().to_owned();
            let mut omega = sc_weight_fw(
                &a,
                opts.zeta_omega,
                opts.omega_intercept,
                w.omega.take(),
                opts.min_decrease,
                pre_iter(opts),
            );
            if opts.sparsify {
                omega = sc_weight_fw(
                    &a,
                    opts.zeta_omega,
                    opts.omega_intercept,
                    Some(sparsify(&omega)),
                    opts.min_decrease,
                    opts.max_iter,
                );
            }
            w.omega = Some(omega);
        }
        w.beta = Vec::new();
    } else {
        let yc = collapsed_form(panel.y.view(), n0, t0);
        let xc: Vec<Array2<f64>> = (0..k)
            .map(|j| collapsed_form(panel.x.index_axis(Axis(2), j), n0, t0))
            .collect();
        // synthdid passes the weights but not β: every fit starts from β = 0.
        w.beta = Vec::new();
        w = sc_weight_fw_covariates(&yc, &xc, opts, w);
    }
    let omega = w.omega.as_ref().expect("set");
    let lambda = w.lambda.as_ref().expect("set");
    let y = panel.adjusted(&w.beta);
    let mut unit = Array1::<f64>::zeros(n);
    for i in 0..n0 {
        unit[i] = -omega[i];
    }
    for i in n0..n {
        unit[i] = 1.0 / n1 as f64;
    }
    let mut time = Array1::<f64>::zeros(t);
    for s in 0..t0 {
        time[s] = -lambda[s];
    }
    for s in t0..t {
        time[s] = 1.0 / t1 as f64;
    }
    (unit.dot(&y).dot(&time), w)
}

fn pre_iter(opts: &Opts) -> usize {
    if opts.sparsify {
        opts.max_iter_pre_sparsify
    } else {
        opts.max_iter
    }
}

/// Controls' pre-treatment outcomes with their post-treatment means as the last
/// column, and the treated units' averages as the last row.
fn collapsed_form(y: ArrayView2<'_, f64>, n0: usize, t0: usize) -> Array2<f64> {
    let (n, t) = y.dim();
    let mut out = Array2::<f64>::zeros((n0 + 1, t0 + 1));
    out.slice_mut(s![..n0, ..t0])
        .assign(&y.slice(s![..n0, ..t0]));
    for i in 0..n0 {
        out[[i, t0]] = y.slice(s![i, t0..]).sum() / (t - t0) as f64;
    }
    for s in 0..t0 {
        out[[n0, s]] = y.slice(s![n0.., s]).sum() / (n - n0) as f64;
    }
    out[[n0, t0]] = y.slice(s![n0.., t0..]).sum() / ((n - n0) * (t - t0)) as f64;
    out
}

/// One Frank–Wolfe step for `min ‖A x − b‖² + η ‖x‖²` on the simplex, with
/// exact line search.
fn fw_step(a: &Array2<f64>, x: &[f64], b: &Array1<f64>, eta: f64) -> Vec<f64> {
    let xv = Array1::from(x.to_vec());
    let ax = a.dot(&xv);
    let resid = &ax - b;
    let half_grad: Vec<f64> = (0..a.ncols())
        .map(|j| resid.dot(&a.column(j)) + eta * x[j])
        .collect();
    let mut i = 0;
    for (j, &g) in half_grad.iter().enumerate() {
        if g < half_grad[i] {
            i = j;
        }
    }
    let mut dx: Vec<f64> = x.iter().map(|v| -v).collect();
    dx[i] = 1.0 - x[i];
    if dx.iter().all(|&v| v == 0.0) {
        return x.to_vec();
    }
    let derr = &a.column(i) - &ax;
    let num: f64 = half_grad.iter().zip(&dx).map(|(g, d)| g * d).sum();
    let den = derr.dot(&derr) + eta * dx.iter().map(|d| d * d).sum::<f64>();
    let step = (-num / den).clamp(0.0, 1.0);
    x.iter().zip(&dx).map(|(v, d)| v + step * d).collect()
}

/// Center each column (over rows).
fn center_columns(y: &Array2<f64>) -> Array2<f64> {
    let mut out = y.clone();
    for mut col in out.columns_mut() {
        let mean = col.sum() / col.len() as f64;
        col.mapv_inplace(|v| v - mean);
    }
    out
}

/// `sc.weight.fw`: weights on the first `ncol − 1` columns that predict the
/// last one.
fn sc_weight_fw(
    y: &Array2<f64>,
    zeta: f64,
    intercept: bool,
    init: Option<Vec<f64>>,
    min_decrease: f64,
    max_iter: usize,
) -> Vec<f64> {
    let t0 = y.ncols() - 1;
    let n0 = y.nrows();
    let mut lambda = init.unwrap_or_else(|| vec![1.0 / t0 as f64; t0]);
    let y = if intercept {
        center_columns(y)
    } else {
        y.clone()
    };
    let a = y.slice(s![.., ..t0]).to_owned();
    let b = y.column(t0).to_owned();
    let eta = n0 as f64 * zeta * zeta;
    let mut vals: Vec<f64> = Vec::new();
    let mut t = 0;
    while t < max_iter && (t < 2 || vals[t - 2] - vals[t - 1] > min_decrease * min_decrease) {
        t += 1;
        lambda = fw_step(&a, &lambda, &b, eta);
        let mut coef = Array1::from(lambda.clone()).to_vec();
        coef.push(-1.0);
        let err = y.dot(&Array1::from(coef));
        vals.push(
            zeta * zeta * lambda.iter().map(|v| v * v).sum::<f64>() + err.dot(&err) / n0 as f64,
        );
    }
    lambda
}

/// `sc.weight.fw.covariates`: weights and covariate coefficients `β` jointly.
fn sc_weight_fw_covariates(
    yc: &Array2<f64>,
    xc: &[Array2<f64>],
    opts: &Opts,
    w: Weights,
) -> Weights {
    let t0 = yc.ncols() - 1;
    let n0 = yc.nrows() - 1;
    let mut lambda = w.lambda.unwrap_or_else(|| vec![1.0 / t0 as f64; t0]);
    let mut omega = w.omega.unwrap_or_else(|| vec![1.0 / n0 as f64; n0]);
    let mut beta = if w.beta.is_empty() {
        vec![0.0; xc.len()]
    } else {
        w.beta
    };
    let adjust = |beta: &[f64]| {
        let mut out = yc.clone();
        for (x, &b) in xc.iter().zip(beta) {
            out.scaled_add(-b, x);
        }
        out
    };
    struct Update {
        val: f64,
        err_lambda: Array1<f64>,
        err_omega: Array1<f64>,
    }
    let update = |y: &Array2<f64>, lambda: &mut Vec<f64>, omega: &mut Vec<f64>| -> Update {
        let ctrl = y.slice(s![..n0, ..]).to_owned();
        let y_lambda = if opts.lambda_intercept {
            center_columns(&ctrl)
        } else {
            ctrl
        };
        if opts.update_lambda {
            *lambda = fw_step(
                &y_lambda.slice(s![.., ..t0]).to_owned(),
                lambda,
                &y_lambda.column(t0).to_owned(),
                n0 as f64 * opts.zeta_lambda * opts.zeta_lambda,
            );
        }
        let mut c = lambda.clone();
        c.push(-1.0);
        let err_lambda = y_lambda.dot(&Array1::from(c));
        let pre = y.slice(s![.., ..t0]).t().to_owned();
        let y_omega = if opts.omega_intercept {
            center_columns(&pre)
        } else {
            pre
        };
        if opts.update_omega {
            *omega = fw_step(
                &y_omega.slice(s![.., ..n0]).to_owned(),
                omega,
                &y_omega.column(n0).to_owned(),
                t0 as f64 * opts.zeta_omega * opts.zeta_omega,
            );
        }
        let mut c = omega.clone();
        c.push(-1.0);
        let err_omega = y_omega.dot(&Array1::from(c));
        let val = opts.zeta_omega * opts.zeta_omega * omega.iter().map(|v| v * v).sum::<f64>()
            + opts.zeta_lambda * opts.zeta_lambda * lambda.iter().map(|v| v * v).sum::<f64>()
            + err_omega.dot(&err_omega) / t0 as f64
            + err_lambda.dot(&err_lambda) / n0 as f64;
        Update {
            val,
            err_lambda,
            err_omega,
        }
    };
    let mut current = update(&adjust(&beta), &mut lambda, &mut omega);
    let mut vals: Vec<f64> = Vec::new();
    let mut t = 0;
    let md2 = opts.min_decrease * opts.min_decrease;
    while t < opts.max_iter && (t < 2 || (vals[t - 2] - vals[t - 1]).abs() > md2) {
        t += 1;
        let mut cl = lambda.clone();
        cl.push(-1.0);
        let cl = Array1::from(cl);
        let mut co = omega.clone();
        co.push(-1.0);
        let co = Array1::from(co);
        let grad: Vec<f64> = xc
            .iter()
            .map(|x| {
                let a = current.err_lambda.dot(&x.slice(s![..n0, ..]).dot(&cl)) / n0 as f64;
                let b = current.err_omega.dot(&x.slice(s![.., ..t0]).t().dot(&co)) / t0 as f64;
                -(a + b)
            })
            .collect();
        let alpha = 1.0 / t as f64;
        for (b, g) in beta.iter_mut().zip(&grad) {
            *b -= alpha * g;
        }
        current = update(&adjust(&beta), &mut lambda, &mut omega);
        vals.push(current.val);
    }
    Weights {
        omega: Some(omega),
        lambda: Some(lambda),
        beta,
    }
}

/// Zero weights below a quarter of the largest, renormalise.
fn sparsify(v: &[f64]) -> Vec<f64> {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let kept: Vec<f64> = v
        .iter()
        .map(|&x| if x <= max / 4.0 { 0.0 } else { x })
        .collect();
    let sum: f64 = kept.iter().sum();
    kept.iter().map(|x| x / sum).collect()
}

fn sum_normalize(v: &[f64]) -> Vec<f64> {
    let sum: f64 = v.iter().sum();
    if sum != 0.0 {
        v.iter().map(|x| x / sum).collect()
    } else {
        vec![1.0 / v.len() as f64; v.len()]
    }
}

fn sample_sd(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

/// Result of [`SyntheticDid::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SynthFit {
    estimator: SynthEstimator,
    estimate: f64,
    omega: Vec<f64>,
    lambda: Vec<f64>,
    beta: Vec<f64>,
    noise_level: f64,
    panel: SynthPanel,
    opts: Opts,
}

impl SynthFit {
    /// The average treatment effect on the treated units in the post periods.
    pub fn estimate(&self) -> f64 {
        self.estimate
    }

    pub fn estimator(&self) -> SynthEstimator {
        self.estimator
    }

    /// Unit weights `ω` of the control units (sum to one).
    pub fn unit_weights(&self) -> &[f64] {
        &self.omega
    }

    /// Time weights `λ` of the pre-treatment periods (sum to one, all zero for
    /// synthetic control).
    pub fn time_weights(&self) -> &[f64] {
        &self.lambda
    }

    /// Covariate coefficients `β` (empty without covariates).
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }

    /// The noise level `σ̂` used.
    pub fn noise_level(&self) -> f64 {
        self.noise_level
    }

    pub fn panel(&self) -> &SynthPanel {
        &self.panel
    }

    /// Effect in each post-treatment period (`synthdid_effect_curve`); their
    /// mean is the estimate.
    pub fn effect_curve(&self) -> Array1<f64> {
        let p = &self.panel;
        let (n, t) = p.y.dim();
        let (n0, t0) = (p.n0, p.t0);
        let y = p.adjusted(&self.beta);
        let mut unit = Array1::<f64>::zeros(n);
        for i in 0..n0 {
            unit[i] = -self.omega[i];
        }
        for i in n0..n {
            unit[i] = 1.0 / (n - n0) as f64;
        }
        let tau = unit.dot(&y);
        let base: f64 = (0..t0).map(|s| tau[s] * self.lambda[s]).sum();
        (t0..t).map(|s| tau[s] - base).collect()
    }

    /// The synthetic control trajectory `Σ_i ω_i y_it` and the treated
    /// average for every period (covariate-adjusted outcomes).
    pub fn trajectories(&self) -> (Array1<f64>, Array1<f64>) {
        let p = &self.panel;
        let y = p.adjusted(&self.beta);
        let synthetic = Array1::from(self.omega.clone()).dot(&y.slice(s![..p.n0, ..]));
        let treated = y
            .slice(s![p.n0.., ..])
            .mean_axis(Axis(0))
            .expect("treated units");
        (synthetic, treated)
    }

    fn weights(&self) -> Weights {
        Weights {
            omega: Some(self.omega.clone()),
            lambda: Some(self.lambda.clone()),
            beta: self.beta.clone(),
        }
    }

    /// Standard error (`synthdid_se`). Jackknife and bootstrap need at least
    /// two treated units, the placebo more controls than treated units.
    pub fn se(&self, method: SynthSe) -> Result<f64> {
        let p = &self.panel;
        let n = p.n_units();
        let (n0, n1) = (p.n0, n - p.n0);
        match method {
            SynthSe::Jackknife => {
                let nonzero = self.omega.iter().filter(|&&w| w != 0.0).count();
                if n0 == n - 1 || nonzero == 1 {
                    return Err(DmlError::InvalidArgument(
                        "the jackknife needs at least two treated units and two control units with weight"
                            .into(),
                    ));
                }
                let mut opts = self.opts;
                opts.update_omega = false;
                opts.update_lambda = false;
                let estimates = map_folds(n, |drop| -> Result<f64> {
                    let rows: Vec<usize> = (0..n).filter(|&i| i != drop).collect();
                    let kept_n0 = rows.iter().filter(|&&i| i < n0).count();
                    let mut w = self.weights();
                    let om: Vec<f64> = rows
                        .iter()
                        .filter(|&&i| i < n0)
                        .map(|&i| self.omega[i])
                        .collect();
                    w.omega = Some(sum_normalize(&om));
                    Ok(estimate(&p.rows(&rows, kept_n0), &opts, w).0)
                })?;
                let mean = estimates.iter().sum::<f64>() / n as f64;
                let var =
                    estimates.iter().map(|u| (u - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
                Ok((((n - 1) as f64 / n as f64) * (n - 1) as f64 * var).sqrt())
            }
            SynthSe::Placebo { replications, seed } => {
                if n0 <= n1 {
                    return Err(DmlError::InvalidArgument(
                        "the placebo standard error needs more controls than treated units".into(),
                    ));
                }
                check_replications(replications)?;
                let mut rng = RRng::set_seed(seed);
                let draws: Vec<Vec<usize>> = (0..replications).map(|_| rng.sample(n0)).collect();
                let estimates = map_folds(replications, |r| -> Result<f64> {
                    let ind = &draws[r];
                    let placebo_n0 = n0 - n1;
                    let mut w = self.weights();
                    let om: Vec<f64> = ind[..placebo_n0].iter().map(|&i| self.omega[i]).collect();
                    w.omega = Some(sum_normalize(&om));
                    Ok(estimate(&p.rows(ind, placebo_n0), &self.opts, w).0)
                })?;
                Ok(scaled_sd(&estimates))
            }
            SynthSe::Bootstrap { replications, seed } => {
                if n0 == n - 1 {
                    return Err(DmlError::InvalidArgument(
                        "the bootstrap needs at least two treated units".into(),
                    ));
                }
                check_replications(replications)?;
                let mut rng = RRng::set_seed(seed);
                let mut draws: Vec<Vec<usize>> = Vec::with_capacity(replications);
                while draws.len() < replications {
                    let mut ind = rng.sample_replace(n, n);
                    if ind.iter().all(|&i| i < n0) || ind.iter().all(|&i| i >= n0) {
                        continue;
                    }
                    ind.sort_unstable();
                    draws.push(ind);
                }
                let estimates = map_folds(replications, |r| -> Result<f64> {
                    let ind = &draws[r];
                    let boot_n0 = ind.iter().filter(|&&i| i < n0).count();
                    let mut w = self.weights();
                    let om: Vec<f64> = ind[..boot_n0].iter().map(|&i| self.omega[i]).collect();
                    w.omega = Some(sum_normalize(&om));
                    Ok(estimate(&p.rows(ind, boot_n0), &self.opts, w).0)
                })?;
                Ok(scaled_sd(&estimates))
            }
        }
    }
}

fn check_replications(r: usize) -> Result<()> {
    if r < 2 {
        return Err(DmlError::InvalidArgument(
            "need at least 2 replications".into(),
        ));
    }
    Ok(())
}

/// `sqrt((r − 1)/r) · sd(estimates)`.
fn scaled_sd(v: &[f64]) -> f64 {
    let r = v.len() as f64;
    ((r - 1.0) / r).sqrt() * sample_sd(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn did_matches_closed_form() {
        let y = Array2::from_shape_fn((6, 5), |(i, t)| {
            (i * i) as f64 + 0.3 * (t * t) as f64 + if i >= 4 && t >= 3 { 2.0 } else { 0.0 }
        });
        let fit = SyntheticDid::new()
            .estimator(SynthEstimator::Did)
            .fit(&SynthPanel::new(y.clone(), 4, 3).unwrap())
            .unwrap();
        let mean = |r: std::ops::Range<usize>, c: std::ops::Range<usize>| {
            y.slice(s![r, c]).mean().unwrap()
        };
        let did = (mean(4..6, 3..5) - mean(4..6, 0..3)) - (mean(0..4, 3..5) - mean(0..4, 0..3));
        assert!((fit.estimate() - did).abs() < 1e-12);
        assert!((fit.effect_curve().mean().unwrap() - fit.estimate()).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_panels() {
        let y = Array2::zeros((4, 4));
        assert!(SynthPanel::new(y.clone(), 4, 2).is_err());
        assert!(SynthPanel::new(y.clone(), 2, 0).is_err());
        let fit = SyntheticDid::new()
            .noise_level(1.0)
            .fit(&SynthPanel::new(y, 3, 2).unwrap())
            .unwrap();
        assert!(fit.se(SynthSe::Jackknife).is_err());
    }
}

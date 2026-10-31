//! Sensitivity analysis for omitted confounders (Chernozhukov, Cinelli, Newey,
//! Sharma & Syrgkanis, 2022, "Long Story Short"), as in DoubleML.
//!
//! If an unobserved confounder `A` was omitted, the estimate is biased by at
//! most
//!
//! ```text
//! |bias| ≤ |ρ| · √(c_y · c_d / (1 − c_d)) · √(σ² · ν²)
//! ```
//!
//! where `σ²` is the residual variance of the outcome, `ν²` the variance of the
//! Riesz representer, `c_y` the share of the outcome's residual variation
//! explained by `A`, `c_d` the relative gain in the Riesz representer's
//! variation from `A`, and `ρ` the correlation between the two (adversarial at
//! 1). [`DmlFit::sensitivity_analysis`] turns a scenario `(c_y, c_d, ρ)` into
//! bounds on θ with confidence bounds, and computes robustness values: how
//! strong confounding (`c_y = c_d`) must be to move θ (RV) or its confidence
//! bound (RVa) to the null hypothesis.

use std::fmt;

use ndarray::{Array1, Array2, Array3, ArrayView2};

use crate::error::{DmlError, Result};
use crate::fit::{aggregate, DmlFit};
use crate::stats::{median, norm_ppf};

/// How a model's `ν²` is formed from per-observation elements.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Nu2Parts {
    /// `ν² = mean(e)` with per-observation elements `e = 2·m(W, α) − α²`.
    Elementwise(Array1<f64>),
    /// Partially linear models: `ν² = 1 / mean(v²)` for the treatment residual
    /// `v`, with Riesz representer `α = v · ν²`.
    InverseMeanSquare(Array1<f64>),
}

/// The per-observation ingredients of a model's sensitivity elements, produced
/// while estimating the score.
///
/// The outcome residual is `a − θ·b` (`b` only for partially linear models,
/// whose residual depends on θ).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SensitivityParts {
    pub outcome_residual: Array1<f64>,
    pub theta_factor: Option<Array1<f64>>,
    /// Riesz representer `α`; ignored for [`Nu2Parts::InverseMeanSquare`],
    /// where it follows from `v`.
    pub riesz: Array1<f64>,
    pub nu2: Nu2Parts,
}

/// Sensitivity elements of one treatment and repetition.
pub(crate) struct Elements {
    pub sigma2: f64,
    pub nu2: f64,
    pub psi_sigma2: Array1<f64>,
    pub psi_nu2: Array1<f64>,
    /// Riesz representer, for the fallback when `ν²` is not positive.
    pub riesz: Option<Array1<f64>>,
    /// The orthogonal `ν²` was not positive (and fell back to `mean(α²)`).
    pub nu2_fallback: bool,
}

impl Elements {
    /// Elements of a sub-sample laid out over all `n` units, as DoubleML's
    /// staggered DiD: `σ²` and `ν²` stay sub-sample means, while the influence
    /// functions and the Riesz representer are zero outside the sub-sample and
    /// scaled by `n / n_sub`, matching the scaling of the padded score.
    pub(crate) fn padded(self, positions: &[usize], n: usize) -> Self {
        let scale = n as f64 / positions.len() as f64;
        let pad = |v: &Array1<f64>| {
            let mut out = Array1::zeros(n);
            for (&p, &x) in positions.iter().zip(v) {
                out[p] = scale * x;
            }
            out
        };
        Self {
            psi_sigma2: pad(&self.psi_sigma2),
            psi_nu2: pad(&self.psi_nu2),
            riesz: self.riesz.as_ref().map(pad),
            ..self
        }
    }
}

impl SensitivityParts {
    /// `σ²`, `ν²` and their influence functions at the estimate θ. A non-positive
    /// `ν²` falls back to the (non-orthogonal) `mean(α²)`, as DoubleML does.
    pub(crate) fn elements(&self, theta: f64) -> Elements {
        let resid = match &self.theta_factor {
            Some(b) => &self.outcome_residual - &(theta * b),
            None => self.outcome_residual.clone(),
        };
        let sigma_el = resid.mapv(|e| e * e);
        let sigma2 = sigma_el.mean().expect("n > 0");
        let (nu2, psi_nu2, riesz) = match &self.nu2 {
            Nu2Parts::Elementwise(e) => {
                let nu2 = e.mean().expect("n > 0");
                (nu2, e - nu2, self.riesz.clone())
            }
            Nu2Parts::InverseMeanSquare(v) => {
                let v2 = v.mapv(|x| x * x);
                let nu2 = 1.0 / v2.mean().expect("n > 0");
                (nu2, v2.mapv(|s| nu2 - s * nu2 * nu2), v * nu2)
            }
        };
        let nu2_fallback = nu2 <= 0.0;
        let (nu2, psi_nu2) = if nu2_fallback {
            let alpha2 = riesz.mapv(|a| a * a);
            (alpha2.mean().expect("n > 0"), alpha2)
        } else {
            (nu2, psi_nu2)
        };
        Elements {
            sigma2,
            nu2,
            psi_sigma2: sigma_el - sigma2,
            psi_nu2,
            riesz: Some(riesz),
            nu2_fallback,
        }
    }
}

/// Stored sensitivity elements of a fit, per treatment `j` and repetition `r`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SensitivityStore {
    /// Shape `(k, n_rep)`.
    pub sigma2: Array2<f64>,
    pub nu2: Array2<f64>,
    pub max_bias: Array2<f64>,
    /// Shape `(n, k, n_rep)`.
    pub psi_max_bias: Array3<f64>,
    /// Influence functions of `σ²` and `ν²`, shape `(n, k, n_rep)` (to combine
    /// the elements of several estimates, e.g. APO contrasts).
    pub psi_sigma2: Array3<f64>,
    pub psi_nu2: Array3<f64>,
    /// Riesz representers, shape `(n, k, n_rep)`.
    pub riesz: Array3<f64>,
    /// Estimates whose orthogonal `ν²` was not positive in some repetition.
    pub nu2_fallback: Vec<bool>,
    /// Population variance of the outcome (for benchmarking).
    pub var_y: f64,
}

impl SensitivityStore {
    pub(crate) fn new(n: usize, k: usize, n_rep: usize, var_y: f64) -> Self {
        Self {
            sigma2: Array2::zeros((k, n_rep)),
            nu2: Array2::zeros((k, n_rep)),
            max_bias: Array2::zeros((k, n_rep)),
            psi_max_bias: Array3::zeros((n, k, n_rep)),
            psi_sigma2: Array3::zeros((n, k, n_rep)),
            psi_nu2: Array3::zeros((n, k, n_rep)),
            riesz: Array3::zeros((n, k, n_rep)),
            nu2_fallback: vec![false; k],
            var_y,
        }
    }

    /// The stored elements of estimate `j` in repetition `r`.
    pub(crate) fn elements(&self, j: usize, r: usize) -> Elements {
        Elements {
            sigma2: self.sigma2[[j, r]],
            nu2: self.nu2[[j, r]],
            psi_sigma2: self.psi_sigma2.slice(ndarray::s![.., j, r]).to_owned(),
            psi_nu2: self.psi_nu2.slice(ndarray::s![.., j, r]).to_owned(),
            riesz: Some(self.riesz.slice(ndarray::s![.., j, r]).to_owned()),
            nu2_fallback: self.nu2_fallback[j],
        }
    }

    /// Stack stores of fits on the same observations along the estimates.
    pub(crate) fn concat(stores: &[&SensitivityStore]) -> Self {
        let cat2 = |f: fn(&SensitivityStore) -> &Array2<f64>| {
            let views: Vec<_> = stores.iter().map(|s| f(s).view()).collect();
            ndarray::concatenate(ndarray::Axis(0), &views).expect("same n_rep")
        };
        let cat3 = |f: fn(&SensitivityStore) -> &Array3<f64>| {
            let views: Vec<_> = stores.iter().map(|s| f(s).view()).collect();
            ndarray::concatenate(ndarray::Axis(1), &views).expect("same n and n_rep")
        };
        Self {
            sigma2: cat2(|s| &s.sigma2),
            nu2: cat2(|s| &s.nu2),
            max_bias: cat2(|s| &s.max_bias),
            psi_max_bias: cat3(|s| &s.psi_max_bias),
            psi_sigma2: cat3(|s| &s.psi_sigma2),
            psi_nu2: cat3(|s| &s.psi_nu2),
            riesz: cat3(|s| &s.riesz),
            nu2_fallback: stores
                .iter()
                .flat_map(|s| s.nu2_fallback.iter().copied())
                .collect(),
            var_y: stores[0].var_y,
        }
    }

    /// Elements of linear combinations of the estimates (one row of `weights`
    /// per combination), by DoubleML's framework arithmetic: bias bounds add
    /// with absolute weights, `max_bias = Σ_j |w_j|·max_bias_j`, and so do
    /// their influence functions. `σ²` and `ν²` survive (as `σ²` and `w²·ν²`)
    /// only for a multiple of a single estimate; otherwise they are `NaN` and
    /// benchmarking is unavailable.
    pub(crate) fn combine(&self, weights: &Array2<f64>) -> Self {
        let (n, _, n_rep) = self.psi_max_bias.dim();
        let mut out = Self::new(n, weights.nrows(), n_rep, self.var_y);
        for (c, row) in weights.outer_iter().enumerate() {
            let used: Vec<usize> = (0..row.len()).filter(|&j| row[j] != 0.0).collect();
            for r in 0..n_rep {
                for &j in &used {
                    let w = row[j].abs();
                    out.max_bias[[c, r]] += w * self.max_bias[[j, r]];
                    out.psi_max_bias
                        .slice_mut(ndarray::s![.., c, r])
                        .scaled_add(w, &self.psi_max_bias.slice(ndarray::s![.., j, r]));
                }
                if let [j] = used[..] {
                    let w = row[j];
                    out.sigma2[[c, r]] = self.sigma2[[j, r]];
                    out.nu2[[c, r]] = w * w * self.nu2[[j, r]];
                    for (dst, src, scale) in [
                        (&mut out.psi_sigma2, &self.psi_sigma2, 1.0),
                        (&mut out.psi_nu2, &self.psi_nu2, w * w),
                        (&mut out.riesz, &self.riesz, w),
                    ] {
                        dst.slice_mut(ndarray::s![.., c, r])
                            .scaled_add(scale, &src.slice(ndarray::s![.., j, r]));
                    }
                } else {
                    out.sigma2[[c, r]] = f64::NAN;
                    out.nu2[[c, r]] = f64::NAN;
                }
            }
            if let [j] = used[..] {
                out.nu2_fallback[c] = self.nu2_fallback[j];
            }
        }
        out
    }

    /// Record one treatment / repetition: `max_bias = √(σ²ν²)` and its
    /// influence function `(σ²·ψ_ν² + ν²·ψ_σ²) / (2·max_bias)`.
    pub(crate) fn set(&mut self, j: usize, r: usize, e: &Elements) {
        if let Some(riesz) = &e.riesz {
            self.riesz.slice_mut(ndarray::s![.., j, r]).assign(riesz);
        }
        self.nu2_fallback[j] |= e.nu2_fallback;
        let max_bias = (e.sigma2 * e.nu2).sqrt();
        self.sigma2[[j, r]] = e.sigma2;
        self.nu2[[j, r]] = e.nu2;
        self.max_bias[[j, r]] = max_bias;
        let psi = (e.sigma2 * &e.psi_nu2 + e.nu2 * &e.psi_sigma2) / (2.0 * max_bias);
        self.psi_max_bias
            .slice_mut(ndarray::s![.., j, r])
            .assign(&psi);
        self.psi_sigma2
            .slice_mut(ndarray::s![.., j, r])
            .assign(&e.psi_sigma2);
        self.psi_nu2
            .slice_mut(ndarray::s![.., j, r])
            .assign(&e.psi_nu2);
    }

    /// DoubleML's rule after all repetitions: if `ν²` was not positive in any
    /// repetition of an estimate, every repetition of it uses the
    /// (non-orthogonal) `mean(α²)`, with `α²` as its influence function (as
    /// DoubleML, which leaves it uncentred, a conservative choice).
    pub(crate) fn validate(&mut self) {
        for j in 0..self.nu2_fallback.len() {
            if !self.nu2_fallback[j] {
                continue;
            }
            for r in 0..self.nu2.ncols() {
                let alpha2 = self.riesz.slice(ndarray::s![.., j, r]).mapv(|a| a * a);
                let e = Elements {
                    sigma2: self.sigma2[[j, r]],
                    nu2: alpha2.mean().expect("n > 0"),
                    psi_sigma2: self.psi_sigma2.slice(ndarray::s![.., j, r]).to_owned(),
                    psi_nu2: alpha2,
                    riesz: None,
                    nu2_fallback: true,
                };
                self.set(j, r, &e);
            }
        }
    }
}

/// A confounding scenario for [`DmlFit::sensitivity_analysis`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SensitivityParams {
    pub cf_y: f64,
    pub cf_d: f64,
    pub rho: f64,
    pub level: f64,
    /// One null hypothesis per treatment, or a single value for all.
    pub null_hypothesis: Vec<f64>,
}

impl Default for SensitivityParams {
    /// DoubleML's defaults: `c_y = c_d = 0.03`, `ρ = 1`, level 0.95, `H0: θ = 0`.
    fn default() -> Self {
        Self {
            cf_y: 0.03,
            cf_d: 0.03,
            rho: 1.0,
            level: 0.95,
            null_hypothesis: vec![0.0],
        }
    }
}

impl SensitivityParams {
    pub fn new() -> Self {
        Self::default()
    }

    /// Share of the outcome's residual variation explained by the confounder.
    pub fn cf_y(mut self, cf_y: f64) -> Self {
        self.cf_y = cf_y;
        self
    }

    /// Relative gain in the Riesz representer's variation from the confounder.
    pub fn cf_d(mut self, cf_d: f64) -> Self {
        self.cf_d = cf_d;
        self
    }

    /// Correlation of the two confounding channels, in `[−1, 1]`.
    pub fn rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    /// Level of the one-sided confidence bounds.
    pub fn level(mut self, level: f64) -> Self {
        self.level = level;
        self
    }

    pub fn null_hypothesis(mut self, value: f64) -> Self {
        self.null_hypothesis = vec![value];
        self
    }

    pub fn null_hypotheses(mut self, values: Vec<f64>) -> Self {
        self.null_hypothesis = values;
        self
    }

    fn validate(&self, k: usize) -> Result<()> {
        let open01 = |v: f64| (0.0..1.0).contains(&v);
        if !open01(self.cf_y) || !open01(self.cf_d) {
            return Err(DmlError::InvalidArgument(
                "cf_y and cf_d must be in [0, 1)".into(),
            ));
        }
        if self.rho.is_nan() || self.rho.abs() > 1.0 {
            return Err(DmlError::InvalidArgument("rho must be in [-1, 1]".into()));
        }
        if !(self.level > 0.0 && self.level < 1.0) {
            return Err(DmlError::InvalidArgument("level must be in (0, 1)".into()));
        }
        if self.null_hypothesis.len() != 1 && self.null_hypothesis.len() != k {
            return Err(DmlError::DimensionMismatch(format!(
                "give one null hypothesis or one per treatment ({k})"
            )));
        }
        Ok(())
    }
}

/// Bounds under one confounding scenario (per treatment).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Bounds {
    theta: (Array1<f64>, Array1<f64>),
    se: (Array1<f64>, Array1<f64>),
    ci: (Array1<f64>, Array1<f64>),
}

/// Result of [`DmlFit::sensitivity_analysis`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SensitivityResult {
    pub params: SensitivityParams,
    treatment_names: Vec<String>,
    theta: Array1<f64>,
    pub theta_lower: Array1<f64>,
    pub theta_upper: Array1<f64>,
    pub se_lower: Array1<f64>,
    pub se_upper: Array1<f64>,
    /// One-sided confidence bounds at `level`.
    pub ci_lower: Array1<f64>,
    pub ci_upper: Array1<f64>,
    /// Robustness value: confounding strength `c_y = c_d` that moves θ to the null.
    pub rv: Array1<f64>,
    /// Robustness value for the confidence bound.
    pub rva: Array1<f64>,
}

/// Largest confounding strength searched for robustness values (as DoubleML).
const RV_MAX: f64 = 0.9999;

impl DmlFit {
    /// Bounds on θ, their confidence bounds and robustness values under the
    /// confounding scenario `params` (PLR, IRM and DID models).
    pub fn sensitivity_analysis(&self, params: &SensitivityParams) -> Result<SensitivityResult> {
        let store = self.sensitivity_store()?;
        let k = self.n_treat();
        params.validate(k)?;
        let b = self.bounds(store, params.cf_y, params.cf_d, params.rho, params.level);
        let null = |j: usize| {
            params.null_hypothesis[if params.null_hypothesis.len() == 1 {
                0
            } else {
                j
            }]
        };
        let mut rv = Array1::zeros(k);
        let mut rva = Array1::zeros(k);
        for j in 0..k {
            let h0 = null(j);
            let upper = h0 > self.coef[j];
            let theta_at = |v: f64| {
                let b = self.bounds(store, v, v, params.rho, params.level);
                if upper {
                    b.theta.1[j]
                } else {
                    b.theta.0[j]
                }
            };
            let ci_at = |v: f64| {
                let b = self.bounds(store, v, v, params.rho, params.level);
                if upper {
                    b.ci.1[j]
                } else {
                    b.ci.0[j]
                }
            };
            rv[j] = robustness_value(theta_at, h0, upper);
            rva[j] = robustness_value(ci_at, h0, upper);
        }
        Ok(SensitivityResult {
            params: SensitivityParams {
                null_hypothesis: (0..k).map(null).collect(),
                ..params.clone()
            },
            treatment_names: self.treatment_names.clone(),
            theta: self.coef.clone(),
            theta_lower: b.theta.0,
            theta_upper: b.theta.1,
            se_lower: b.se.0,
            se_upper: b.se.1,
            ci_lower: b.ci.0,
            ci_upper: b.ci.1,
            rv,
            rva,
        })
    }

    /// Sensitivity elements `(σ², ν²)`, each of shape `(n_treat, n_rep)`:
    /// the residual variance of the outcome and the variance of the Riesz
    /// representer. `None` for models without sensitivity analysis.
    pub fn sensitivity_elements(&self) -> Option<(ArrayView2<'_, f64>, ArrayView2<'_, f64>)> {
        self.sensitivity
            .as_ref()
            .map(|s| (s.sigma2.view(), s.nu2.view()))
    }

    fn sensitivity_store(&self) -> Result<&SensitivityStore> {
        self.sensitivity.as_ref().ok_or_else(|| {
            DmlError::InvalidArgument(
                "sensitivity analysis is available for PLR, IRM and DID fits".into(),
            )
        })
    }

    /// DoubleML's `_calc_sensitivity_analysis`.
    fn bounds(
        &self,
        store: &SensitivityStore,
        cf_y: f64,
        cf_d: f64,
        rho: f64,
        level: f64,
    ) -> Bounds {
        let strength = rho.abs() * (cf_y * cf_d / (1.0 - cf_d)).sqrt();
        let (n, k, n_rep) = self.psi.dim();
        let mut theta_lo = Array2::zeros((k, n_rep));
        let mut theta_hi = Array2::zeros((k, n_rep));
        let mut se_lo = Array2::zeros((k, n_rep));
        let mut se_hi = Array2::zeros((k, n_rep));
        for j in 0..k {
            for r in 0..n_rep {
                let shift = strength * store.max_bias[[j, r]];
                theta_lo[[j, r]] = self.all_coef[[j, r]] - shift;
                theta_hi[[j, r]] = self.all_coef[[j, r]] + shift;
                let jac = self.psi_deriv[[j, r]];
                let scaled = self.psi.slice(ndarray::s![.., j, r]).mapv(|v| v / jac);
                let pmb = store
                    .psi_max_bias
                    .slice(ndarray::s![.., j, r])
                    .mapv(|v| strength * v);
                let (psi_lo, psi_hi) = (&scaled - &pmb, &scaled + &pmb);
                let variance = |psi: &Array1<f64>| match &self.cluster {
                    Some(ci) => ci.variance(psi.view(), Array1::ones(n).view(), r),
                    None => psi.dot(psi) / n as f64 / n as f64,
                };
                se_lo[[j, r]] = variance(&psi_lo).sqrt();
                se_hi[[j, r]] = variance(&psi_hi).sqrt();
            }
        }
        let quant = norm_ppf(level);
        let ci = |theta: &Array2<f64>, se: &Array2<f64>, sign: f64| -> Array1<f64> {
            (0..k)
                .map(|j| {
                    let v: Vec<f64> = (0..n_rep)
                        .map(|r| theta[[j, r]] + sign * quant * se[[j, r]])
                        .collect();
                    median(&v)
                })
                .collect()
        };
        let ci_lo = ci(&theta_lo, &se_lo, -1.0);
        let ci_hi = ci(&theta_hi, &se_hi, 1.0);
        let (t_lo, s_lo) = aggregate(&theta_lo, &se_lo);
        let (t_hi, s_hi) = aggregate(&theta_hi, &se_hi);
        Bounds {
            theta: (t_lo, t_hi),
            se: (s_lo, s_hi),
            ci: (ci_lo, ci_hi),
        }
    }

    /// Calibrate a confounding scenario against observed covariates: `short`
    /// must be the same model refitted **on the same sample splits** without
    /// the benchmarking covariates (see `sensitivity_benchmark` on the models).
    pub fn sensitivity_benchmark(&self, short: &DmlFit) -> Result<SensitivityBenchmark> {
        let long = self.sensitivity_store()?;
        let short_store = short.sensitivity_store()?;
        if long
            .sigma2
            .iter()
            .chain(&short_store.sigma2)
            .any(|v| v.is_nan())
        {
            return Err(DmlError::InvalidArgument(
                "benchmarking is not available for combinations of several estimates".into(),
            ));
        }
        if short.all_coef.dim() != self.all_coef.dim() {
            return Err(DmlError::DimensionMismatch(
                "long and short fits need the same treatments and repetitions".into(),
            ));
        }
        let (k, n_rep) = self.all_coef.dim();
        let (mut cf_y, mut cf_d, mut rho, mut delta) = (
            Array1::zeros(k),
            Array1::zeros(k),
            Array1::zeros(k),
            Array1::zeros(k),
        );
        for j in 0..k {
            let mut cy = Vec::with_capacity(n_rep);
            let mut cd = Vec::with_capacity(n_rep);
            let mut rh = Vec::with_capacity(n_rep);
            let mut dt = Vec::with_capacity(n_rep);
            for r in 0..n_rep {
                let (s_long, s_short) = (long.sigma2[[j, r]], short_store.sigma2[[j, r]]);
                let (n_long, n_short) = (long.nu2[[j, r]], short_store.nu2[[j, r]]);
                let r2_long = 1.0 - s_long / long.var_y;
                let r2_short = 1.0 - s_short / long.var_y;
                let r2_riesz = n_short / n_long;
                cy.push(((r2_long - r2_short) / (1.0 - r2_long)).clamp(0.0, 1.0));
                cd.push(((1.0 - r2_riesz) / r2_riesz).clamp(0.0, 1.0));
                let d_theta = short.all_coef[[j, r]] - self.all_coef[[j, r]];
                dt.push(d_theta);
                let (var_g, var_riesz) = (s_short - s_long, n_long - n_short);
                let denom = if var_g > 0.0 && var_riesz > 0.0 {
                    (var_g * var_riesz).sqrt()
                } else {
                    0.0
                };
                let magnitude = if denom != 0.0 {
                    (d_theta.abs() / denom).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                let sign = if d_theta > 0.0 {
                    1.0
                } else if d_theta < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                rh.push(magnitude * sign);
            }
            cf_y[j] = median(&cy);
            cf_d[j] = median(&cd);
            rho[j] = median(&rh);
            delta[j] = median(&dt);
        }
        Ok(SensitivityBenchmark {
            treatment_names: self.treatment_names.clone(),
            cf_y,
            cf_d,
            rho,
            delta_theta: delta,
        })
    }
}

/// Smallest `v ∈ [0, 0.9999]` at which `bound(v)` reaches `h0` (bisection on a
/// bound that moves monotonically towards the null as confounding grows).
/// Returns 0 if the bound already reaches the null without confounding and
/// 0.9999 if it never does, matching DoubleML's bounded search.
fn robustness_value(bound: impl Fn(f64) -> f64, h0: f64, upper: bool) -> f64 {
    // gap > 0 while the bound has not yet crossed the null.
    let gap = |v: f64| if upper { h0 - bound(v) } else { bound(v) - h0 };
    if gap(0.0) <= 0.0 {
        return 0.0;
    }
    if gap(RV_MAX) > 0.0 {
        return RV_MAX;
    }
    let (mut lo, mut hi) = (0.0, RV_MAX);
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if gap(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-13 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Benchmark values for `(c_y, c_d, ρ)` from omitting observed covariates.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SensitivityBenchmark {
    treatment_names: Vec<String>,
    pub cf_y: Array1<f64>,
    pub cf_d: Array1<f64>,
    pub rho: Array1<f64>,
    /// Change in the estimate when the covariates are dropped.
    pub delta_theta: Array1<f64>,
}

impl SensitivityBenchmark {
    /// The benchmarked scenario for treatment `j`, ready for
    /// [`DmlFit::sensitivity_analysis`].
    pub fn params(&self, j: usize) -> SensitivityParams {
        SensitivityParams::new()
            .cf_y(self.cf_y[j])
            .cf_d(self.cf_d[j])
            .rho(self.rho[j])
    }
}

impl fmt::Display for SensitivityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = &self.params;
        writeln!(
            f,
            "================== Sensitivity Analysis =================="
        )?;
        writeln!(f)?;
        writeln!(f, "------------------ Scenario          ------------------")?;
        writeln!(f, "Significance Level: level={}", p.level)?;
        writeln!(
            f,
            "Sensitivity parameters: cf_y={}; cf_d={}, rho={}",
            p.cf_y, p.cf_d, p.rho
        )?;
        writeln!(f)?;
        writeln!(f, "------------------ Bounds with CI    ------------------")?;
        let w = self
            .treatment_names
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(1)
            .max(2);
        writeln!(
            f,
            "{:<w$} {:>10} {:>12} {:>10} {:>12} {:>10}",
            "", "CI lower", "theta lower", "theta", "theta upper", "CI upper"
        )?;
        for (j, name) in self.treatment_names.iter().enumerate() {
            writeln!(
                f,
                "{:<w$} {:>10.6} {:>12.6} {:>10.6} {:>12.6} {:>10.6}",
                name,
                self.ci_lower[j],
                self.theta_lower[j],
                self.theta[j],
                self.theta_upper[j],
                self.ci_upper[j]
            )?;
        }
        writeln!(f)?;
        writeln!(f, "------------------ Robustness Values ------------------")?;
        writeln!(
            f,
            "{:<w$} {:>8} {:>10} {:>10}",
            "", "H_0", "RV (%)", "RVa (%)"
        )?;
        for (j, name) in self.treatment_names.iter().enumerate() {
            write!(
                f,
                "{:<w$} {:>8.1} {:>10.6} {:>10.6}",
                name,
                p.null_hypothesis[j],
                100.0 * self.rv[j],
                100.0 * self.rva[j]
            )?;
            if j + 1 < self.treatment_names.len() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for SensitivityBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = self
            .treatment_names
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(1)
            .max(2);
        write!(
            f,
            "{:<w$} {:>10} {:>10} {:>10} {:>12}",
            "", "cf_y", "cf_d", "rho", "delta_theta"
        )?;
        for (j, name) in self.treatment_names.iter().enumerate() {
            write!(
                f,
                "\n{:<w$} {:>10.6} {:>10.6} {:>10.6} {:>12.6}",
                name, self.cf_y[j], self.cf_d[j], self.rho[j], self.delta_theta[j]
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
    fn plr_elements_match_closed_form() {
        // a = y − ℓ̂, b = d − m̂ = v
        let parts = SensitivityParts {
            outcome_residual: array![1.0, -1.0, 2.0, 0.0],
            theta_factor: Some(array![0.5, -0.5, 1.0, -1.0]),
            riesz: Array1::zeros(4),
            nu2: Nu2Parts::InverseMeanSquare(array![0.5, -0.5, 1.0, -1.0]),
        };
        let e = parts.elements(2.0);
        // residuals 1−1, −1+1, 2−2, 0+2 → (0, 0, 0, 2): σ² = 1
        assert!((e.sigma2 - 1.0).abs() < 1e-15);
        // mean v² = (0.25+0.25+1+1)/4 = 0.625 → ν² = 1.6
        assert!((e.nu2 - 1.6).abs() < 1e-15);
        assert!(e.psi_nu2.sum().abs() < 1e-12 && e.psi_sigma2.sum().abs() < 1e-12);
    }

    #[test]
    fn negative_nu2_falls_back_to_riesz_square() {
        let parts = SensitivityParts {
            outcome_residual: array![1.0, 2.0],
            theta_factor: None,
            riesz: array![1.0, 3.0],
            nu2: Nu2Parts::Elementwise(array![-1.0, -2.0]),
        };
        let e = parts.elements(0.0);
        assert!((e.nu2 - 5.0).abs() < 1e-15); // mean(1, 9)
    }

    #[test]
    fn robustness_value_solves_the_crossing() {
        // bound(v) = 1 − v, null 0.25 (lower bound) → v = 0.75
        let rv = robustness_value(|v| 1.0 - v, 0.25, false);
        assert!((rv - 0.75).abs() < 1e-12);
        // Upper bound rising with confounding: bound(v) = 1 + v reaches 1.5 at v = 0.5.
        assert!((robustness_value(|v| 1.0 + v, 1.5, true) - 0.5).abs() < 1e-12);
        assert_eq!(robustness_value(|v| 1.0 - v, 1.5, false), 0.0); // null already beyond the bound
        assert_eq!(robustness_value(|_| 1.0, 0.0, false), RV_MAX); // never reached
    }

    #[test]
    fn params_validation() {
        assert!(SensitivityParams::new().cf_y(1.0).validate(1).is_err());
        assert!(SensitivityParams::new().rho(1.5).validate(1).is_err());
        assert!(SensitivityParams::new()
            .null_hypotheses(vec![0.0, 1.0])
            .validate(3)
            .is_err());
        assert!(SensitivityParams::new().validate(2).is_ok());
    }
}

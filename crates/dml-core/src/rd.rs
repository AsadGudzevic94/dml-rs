//! Local polynomial regression discontinuity (RD) estimation: a port of the
//! `rdrobust` package 2.1.1 (Calonico, Cattaneo and Titiunik, 2014;
//! Calonico, Cattaneo and Farrell, 2020), used by [`RdFlex`](crate::RdFlex).
//!
//! - [`rd_bandwidths`]: the MSE- or CER-optimal bandwidths of `rdbwselect`
//!   for the point estimate (`h`) and the bias correction (`b`).
//! - [`rd_estimate`]: the local polynomial estimate at the cutoff with its
//!   bias correction and the conventional, bias-corrected and robust
//!   confidence intervals.
//!
//! [`RdOptions`] holds `rdrobust`'s options (polynomial orders, kernel,
//! bandwidth selector, variance estimator, mass-point handling, ...) with its
//! defaults; [`RdData`] the outcome, running variable and optional treatment
//! (fuzzy design), covariates and cluster labels. With clusters the variance
//! is cluster-robust (`rdrobust`'s `cr1`). [`rdbwselect`] and [`rdrobust`]
//! are shortcuts with the default options.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::compat::{np_mean, np_sum};
use crate::error::{DmlError, Result};
use crate::linalg::{spd_inverse, sym_pinv};
use crate::stats::norm_ppf;

/// Kernel of the local polynomial fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kernel {
    #[default]
    Triangular,
    Epanechnikov,
    Uniform,
}

impl Kernel {
    /// `rdrobust_kweight`.
    fn weight(self, x: f64, c: f64, h: f64) -> f64 {
        let u = (x - c) / h;
        let inside = f64::from(u.abs() <= 1.0);
        match self {
            Kernel::Epanechnikov => (0.75 * (1.0 - u * u) * inside) / h,
            Kernel::Uniform => (0.5 * inside) / h,
            Kernel::Triangular => ((1.0 - u.abs()) * inside) / h,
        }
    }

    /// The constant `C_c` of the pilot bandwidth.
    fn c_c(self) -> f64 {
        match self {
            Kernel::Epanechnikov => 2.34,
            Kernel::Uniform => 1.843,
            Kernel::Triangular => 2.576,
        }
    }
}

/// Bandwidth selector, as `rdrobust`'s `bwselect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BwSelect {
    /// One common MSE-optimal bandwidth for the RD effect.
    #[default]
    Mserd,
    /// Separate MSE-optimal bandwidths below and above the cutoff.
    Msetwo,
    /// One common MSE-optimal bandwidth for the sum of the two regression
    /// estimates.
    Msesum,
    /// `min(mserd, msesum)`.
    Msecomb1,
    /// `median(msetwo, mserd, msesum)` on each side.
    Msecomb2,
    /// Coverage-error-rate optimal versions of the above.
    Cerrd,
    Certwo,
    Cersum,
    Cercomb1,
    Cercomb2,
}

impl BwSelect {
    fn needs_two(self) -> bool {
        matches!(
            self,
            BwSelect::Msetwo | BwSelect::Certwo | BwSelect::Msecomb2 | BwSelect::Cercomb2
        )
    }

    fn needs_sum(self) -> bool {
        matches!(
            self,
            BwSelect::Msesum
                | BwSelect::Cersum
                | BwSelect::Msecomb1
                | BwSelect::Msecomb2
                | BwSelect::Cercomb1
                | BwSelect::Cercomb2
        )
    }

    fn needs_rd(self) -> bool {
        matches!(
            self,
            BwSelect::Mserd
                | BwSelect::Cerrd
                | BwSelect::Msecomb1
                | BwSelect::Msecomb2
                | BwSelect::Cercomb1
                | BwSelect::Cercomb2
        )
    }

    fn is_cer(self) -> bool {
        matches!(
            self,
            BwSelect::Cerrd
                | BwSelect::Certwo
                | BwSelect::Cersum
                | BwSelect::Cercomb1
                | BwSelect::Cercomb2
        )
    }
}

/// Variance estimator, as `rdrobust`'s `vce`. With cluster labels, `Nn`,
/// `Hc0` and `Hc1` all give the cluster-robust `cr1`; the leverage-adjusted
/// cluster estimators (`rdrobust`'s `cr2` / `cr3`, from `Hc2` / `Hc3`) are
/// not available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Vce {
    /// Nearest-neighbour residuals (with at least `nnmatch` neighbours).
    #[default]
    Nn,
    Hc0,
    Hc1,
    Hc2,
    Hc3,
}

/// Treatment of repeated values of the running variable, as `rdrobust`'s
/// `masspoints`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassPoints {
    /// With many repeated values, enlarge the pilot bandwidths to contain at
    /// least `bwcheck` (default 10) distinct values, and count distinct
    /// values in the pilot rule.
    #[default]
    Adjust,
    /// Count distinct values only for detection (no adjustment).
    Check,
    /// Ignore repeated values.
    Off,
}

/// Options of [`rd_bandwidths`] and [`rd_estimate`], with `rdrobust`'s
/// defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct RdOptions {
    /// Polynomial order of the point estimate (default 1).
    pub p: usize,
    /// Polynomial order of the bias correction (default `p + 1`).
    pub q: Option<usize>,
    pub kernel: Kernel,
    pub bwselect: BwSelect,
    pub vce: Vce,
    /// Minimum number of neighbours for `Vce::Nn` (default 3).
    pub nnmatch: usize,
    /// Confidence level of the intervals (default 0.95).
    pub level: f64,
    pub masspoints: MassPoints,
    /// Minimum number of distinct running-variable values in the pilot
    /// bandwidths (default: none, or 10 with detected mass points).
    pub bwcheck: Option<usize>,
    /// Restrict bandwidths to the range of the running variable (default true).
    pub bwrestrict: bool,
    /// Standardise the outcome and running variable for bandwidth selection
    /// (default true).
    pub stdvars: bool,
    /// Scale of the regularisation term of the bandwidth selectors (default 1).
    pub scaleregul: f64,
    /// Fuzzy design with the bandwidth selector of the sharp design (also
    /// used when the treatment does not vary on one side).
    pub sharpbw: bool,
    /// Bias bandwidth `b = h / rho` (default: selected separately).
    pub rho: Option<f64>,
    /// Multiplier of the estimate (default 1).
    pub scalepar: f64,
    /// Drop collinear covariates (default true).
    pub covs_drop: bool,
}

impl Default for RdOptions {
    fn default() -> Self {
        Self {
            p: 1,
            q: None,
            kernel: Kernel::Triangular,
            bwselect: BwSelect::Mserd,
            vce: Vce::Nn,
            nnmatch: 3,
            level: 0.95,
            masspoints: MassPoints::Adjust,
            bwcheck: None,
            bwrestrict: true,
            stdvars: true,
            scaleregul: 1.0,
            sharpbw: false,
            rho: None,
            scalepar: 1.0,
            covs_drop: true,
        }
    }
}

impl RdOptions {
    fn q(&self) -> usize {
        self.q.unwrap_or(self.p + 1)
    }

    fn validate(&self) -> Result<()> {
        let q = self.q();
        let bad = |msg: &str| Err(DmlError::InvalidArgument(format!("rdrobust: {msg}")));
        if self.p > 20 || q > 20 || q <= self.p {
            return bad("polynomial orders need p <= 20 and p < q <= 20");
        }
        if self.nnmatch == 0 {
            return bad("nnmatch must be at least 1");
        }
        if !(self.level > 0.0 && self.level < 1.0) {
            return bad("level must be in (0, 1)");
        }
        if self.bwcheck == Some(0) {
            return bad("bwcheck must be a positive integer");
        }
        if self.rho.is_some_and(|r| !(r.is_finite() && r > 0.0)) {
            return bad("rho must be positive");
        }
        if !(self.scaleregul.is_finite() && self.scaleregul >= 0.0) || !self.scalepar.is_finite() {
            return bad("scaleregul and scalepar must be finite (scaleregul >= 0)");
        }
        Ok(())
    }
}

/// Data of an RD design: outcome `y`, running variable `x` and cutoff, and
/// optionally the treatment (fuzzy design), covariates and cluster labels.
#[derive(Debug, Clone, Copy)]
pub struct RdData<'a> {
    pub y: ArrayView1<'a, f64>,
    pub x: ArrayView1<'a, f64>,
    pub cutoff: f64,
    pub fuzzy: Option<ArrayView1<'a, f64>>,
    pub covs: Option<ArrayView2<'a, f64>>,
    pub cluster: Option<&'a [i64]>,
}

impl<'a> RdData<'a> {
    /// A sharp design with cutoff 0.
    pub fn new(y: ArrayView1<'a, f64>, x: ArrayView1<'a, f64>) -> Self {
        Self {
            y,
            x,
            cutoff: 0.0,
            fuzzy: None,
            covs: None,
            cluster: None,
        }
    }

    pub fn cutoff(mut self, c: f64) -> Self {
        self.cutoff = c;
        self
    }

    /// Fuzzy design with treatment `t`.
    pub fn fuzzy(mut self, t: ArrayView1<'a, f64>) -> Self {
        self.fuzzy = Some(t);
        self
    }

    /// Covariates for covariate-adjusted estimation (`rdrobust`'s `covs`).
    pub fn covariates(mut self, z: ArrayView2<'a, f64>) -> Self {
        self.covs = Some(z);
        self
    }

    /// Cluster labels for cluster-robust standard errors.
    pub fn clusters(mut self, labels: &'a [i64]) -> Self {
        self.cluster = Some(labels);
        self
    }
}

/// Bandwidths below (`.0`) and above (`.1`) the cutoff.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdBandwidths {
    /// Main bandwidth.
    pub h: (f64, f64),
    /// Bias bandwidth.
    pub b: (f64, f64),
}

impl RdBandwidths {
    /// The same `h` and `b` on both sides.
    pub fn common(h: f64, b: f64) -> Self {
        Self {
            h: (h, h),
            b: (b, b),
        }
    }
}

/// Result of [`rd_estimate`]: rows are the conventional, bias-corrected and
/// robust (bias-corrected with robust standard error) estimates.
#[derive(Debug, Clone)]
pub struct RdEstimate {
    pub coef: [f64; 3],
    pub se: [f64; 3],
    /// Confidence intervals `[lower, upper]` at the options' level.
    pub ci: [[f64; 2]; 3],
    /// Main bandwidth `(left, right)`.
    pub h: (f64, f64),
    /// Bias bandwidth `(left, right)`.
    pub b: (f64, f64),
    /// Observations within the main bandwidth `(left, right)`.
    pub n_h: (usize, usize),
    /// Observations on each side of the cutoff.
    pub n: (usize, usize),
}

// ---------------------------------------------------------------------------
// Data preparation
// ---------------------------------------------------------------------------

/// One side of the cutoff, sorted by the running variable.
#[derive(Clone)]
struct Side {
    x: Vec<f64>,
    y: Vec<f64>,
    t: Option<Vec<f64>>,
    z: Option<Array2<f64>>,
    cl: Option<Vec<i64>>,
    dups: Vec<usize>,
    dupsid: Vec<usize>,
}

impl Side {
    fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        t: Option<Vec<f64>>,
        z: Option<Array2<f64>>,
        cl: Option<Vec<i64>>,
    ) -> Self {
        let (dups, dupsid) = duplicates(&x);
        Self {
            x,
            y,
            t,
            z,
            cl,
            dups,
            dupsid,
        }
    }

    fn n(&self) -> usize {
        self.x.len()
    }

    /// The observations in `keep`, with duplicates counted within them.
    fn select(&self, keep: &[bool]) -> Side {
        let rows: Vec<usize> = (0..self.n()).filter(|&i| keep[i]).collect();
        let pick = |v: &[f64]| -> Vec<f64> { rows.iter().map(|&i| v[i]).collect() };
        Side::new(
            pick(&self.x),
            pick(&self.y),
            self.t.as_ref().map(|t| pick(t)),
            self.z.as_ref().map(|z| z.select(Axis(0), &rows)),
            self.cl
                .as_ref()
                .map(|c| rows.iter().map(|&i| c[i]).collect()),
        )
    }

    fn d_t(&self) -> usize {
        usize::from(self.t.is_some())
    }

    fn d_z(&self) -> usize {
        self.z.as_ref().map_or(0, |z| z.ncols())
    }

    /// Outcome, treatment and covariate columns.
    fn d(&self) -> Array2<f64> {
        let (dt, dz) = (self.d_t(), self.d_z());
        let mut d = Array2::zeros((self.n(), 1 + dt + dz));
        for i in 0..self.n() {
            d[[i, 0]] = self.y[i];
            if let Some(t) = &self.t {
                d[[i, 1]] = t[i];
            }
            if let Some(z) = &self.z {
                for j in 0..dz {
                    d[[i, 1 + dt + j]] = z[[i, j]];
                }
            }
        }
        d
    }
}

/// Size of each run of equal values and the position within it, as
/// pandas' `groupby(x).transform("sum" / "cumsum")` on sorted `x`.
fn duplicates(x: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = x.len();
    let (mut dups, mut dupsid) = (vec![0; n], vec![0; n]);
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && x[end] == x[start] {
            end += 1;
        }
        for (k, i) in (start..end).enumerate() {
            dups[i] = end - start;
            dupsid[i] = k + 1;
        }
        start = end;
    }
    (dups, dupsid)
}

/// `[1, u, u², …, u^p]` by successive multiplication.
fn vander(u: &[f64], p: usize) -> Array2<f64> {
    let mut out = Array2::zeros((u.len(), p + 1));
    for (i, &v) in u.iter().enumerate() {
        out[[i, 0]] = 1.0;
        for j in 1..=p {
            out[[i, j]] = out[[i, j - 1]] * v;
        }
    }
    out
}

/// NumPy's `x ** k` for a float base and integer exponent.
fn powk(x: f64, k: usize) -> f64 {
    match k {
        0 => 1.0,
        1 => x,
        2 => x * x,
        _ => x.powf(k as f64),
    }
}

fn scale_rows(m: &Array2<f64>, w: &[f64]) -> Array2<f64> {
    let mut out = m.clone();
    for (mut row, &wi) in out.rows_mut().into_iter().zip(w) {
        row *= wi;
    }
    out
}

fn crossprod(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.t().dot(b)
}

fn n_unique(x: &[f64]) -> usize {
    let mut v = x.to_vec();
    v.sort_by(f64::total_cmp);
    v.dedup();
    v.len()
}

fn factorial(k: usize) -> f64 {
    (1..=k).product::<usize>() as f64
}

/// NumPy's `std(ddof=1)`.
fn std_ddof1(v: &[f64]) -> f64 {
    let mean = np_sum(v) / v.len() as f64;
    let sq: Vec<f64> = v.iter().map(|x| (x - mean) * (x - mean)).collect();
    (np_sum(&sq) / (v.len() - 1) as f64).sqrt()
}

/// NumPy's `var` (population variance).
fn var(v: &[f64]) -> f64 {
    let mean = np_mean(v);
    let sq: Vec<f64> = v.iter().map(|x| (x - mean) * (x - mean)).collect();
    np_mean(&sq)
}

/// R's `quantile(type = 2)` of sorted `x`.
fn quantile_type2(x: &[f64], prob: f64) -> f64 {
    let n = x.len();
    let h = n as f64 * prob;
    let j = h.floor() as usize;
    if h == j as f64 {
        0.5 * (x[j - 1] + x[j])
    } else {
        x[j]
    }
}

/// Covariate columns to keep (`covs_drop_fun`): all of them if the QR
/// factorisation has no diagonal entry below `1e-5`, else the columns a
/// column-pivoted QR finds independent, in pivot order.
fn independent_columns(z: &Array2<f64>) -> Vec<usize> {
    const TOL: f64 = 1e-5;
    let k = z.ncols();
    let (diag, _) = householder_qr(z, false);
    if diag.iter().all(|d| d.abs() > TOL) {
        return (0..k).collect();
    }
    let (diag, pivot) = householder_qr(z, true);
    pivot
        .into_iter()
        .zip(diag)
        .filter(|(_, d)| d.abs() > TOL)
        .map(|(j, _)| j)
        .collect()
}

/// Diagonal of `R` in a Householder QR of `a` (with column pivoting by the
/// largest remaining norm, as LAPACK's `geqp3`) and the column order.
fn householder_qr(a: &Array2<f64>, pivoting: bool) -> (Vec<f64>, Vec<usize>) {
    let (n, k) = a.dim();
    let mut m = a.clone();
    let mut order: Vec<usize> = (0..k).collect();
    let mut diag = Vec::with_capacity(k.min(n));
    for j in 0..k.min(n) {
        if pivoting {
            let norm =
                |c: usize, m: &Array2<f64>| (j..n).map(|i| m[[i, c]] * m[[i, c]]).sum::<f64>();
            let best = (j..k)
                .max_by(|&a, &b| norm(a, &m).total_cmp(&norm(b, &m)))
                .expect("columns left");
            if best != j {
                for i in 0..n {
                    m.swap([i, j], [i, best]);
                }
                order.swap(j, best);
            }
        }
        let alpha = (j..n).map(|i| m[[i, j]] * m[[i, j]]).sum::<f64>().sqrt();
        let alpha = if m[[j, j]] > 0.0 { -alpha } else { alpha };
        diag.push(alpha);
        if alpha == 0.0 {
            continue;
        }
        let mut v: Vec<f64> = (j..n).map(|i| m[[i, j]]).collect();
        v[0] -= alpha;
        let vnorm2: f64 = v.iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        for c in j..k {
            let dot: f64 = (j..n).map(|i| v[i - j] * m[[i, c]]).sum();
            let f = 2.0 * dot / vnorm2;
            for i in j..n {
                m[[i, c]] -= f * v[i - j];
            }
        }
    }
    (diag, order)
}

/// Sorted, validated sides of the cutoff, with collinear covariates dropped.
fn prepare(data: &RdData<'_>, opts: &RdOptions) -> Result<(Side, Side)> {
    let (y, x, c) = (data.y, data.x, data.cutoff);
    let n = x.len();
    let lengths_ok = y.len() == n
        && data.fuzzy.is_none_or_len(n)
        && data.covs.map_or(true, |z| z.nrows() == n)
        && data.cluster.map_or(true, |cl| cl.len() == n);
    if !lengths_ok {
        return Err(DmlError::DimensionMismatch(
            "rdrobust: y, x, fuzzy, covs and cluster need the same number of observations".into(),
        ));
    }
    let finite = x.iter().chain(y.iter()).all(|v| v.is_finite())
        && data.fuzzy.map_or(true, |t| t.iter().all(|v| v.is_finite()))
        && data.covs.map_or(true, |z| z.iter().all(|v| v.is_finite()));
    if !finite {
        return Err(DmlError::InvalidArgument(
            "rdrobust: data contain NaN or infinite values".into(),
        ));
    }
    if !c.is_finite() {
        return Err(DmlError::InvalidArgument(
            "rdrobust: the cutoff must be finite".into(),
        ));
    }
    let covs = match data.covs {
        Some(z) if opts.covs_drop => {
            let z = z.to_owned();
            let keep = independent_columns(&z);
            Some(z.select(Axis(1), &keep))
        }
        Some(z) => Some(z.to_owned()),
        None => None,
    };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let (x_min, x_max) = (x[order[0]], x[order[n - 1]]);
    if c <= x_min || c >= x_max {
        return Err(DmlError::InvalidArgument(
            "the cutoff must lie within the range of the running variable".into(),
        ));
    }
    let side = |left: bool| -> Side {
        let rows: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&i| (x[i] < c) == left)
            .collect();
        Side::new(
            rows.iter().map(|&i| x[i]).collect(),
            rows.iter().map(|&i| y[i]).collect(),
            data.fuzzy.map(|t| rows.iter().map(|&i| t[i]).collect()),
            covs.as_ref().map(|z| z.select(Axis(0), &rows)),
            data.cluster.map(|cl| rows.iter().map(|&i| cl[i]).collect()),
        )
    };
    Ok((side(true), side(false)))
}

/// `Option<ArrayView1>` has the expected length (or is absent).
trait LenCheck {
    fn is_none_or_len(&self, n: usize) -> bool;
}

impl LenCheck for Option<ArrayView1<'_, f64>> {
    fn is_none_or_len(&self, n: usize) -> bool {
        self.map_or(true, |t| t.len() == n)
    }
}

// ---------------------------------------------------------------------------
// Residuals and variance
// ---------------------------------------------------------------------------

/// `rdrobust_res`: nearest-neighbour residuals, or plug-in residuals
/// `w · (D − fitted)` with the HC weight `w` (`d` coefficients in the fit).
fn residuals(
    e: &Side,
    vce: Vce,
    fitted: Option<&Array2<f64>>,
    hii: Option<&Array1<f64>>,
    nnmatch: usize,
    d: usize,
) -> Array2<f64> {
    let dmat = e.d();
    if vce == Vce::Nn {
        return nn_residuals(e, &dmat, nnmatch);
    }
    let n = e.n();
    let clustered = e.cl.is_some();
    let fitted = fitted.expect("plug-in residuals need fitted values");
    let mut res = &dmat - fitted;
    for i in 0..n {
        let w = match vce {
            Vce::Hc0 => 1.0,
            Vce::Hc1 if clustered => 1.0,
            Vce::Hc1 => (n as f64 / (n - d) as f64).sqrt(),
            Vce::Hc2 => (1.0 / (1.0 - hii.expect("leverages")[i]).max(1e-8)).sqrt(),
            Vce::Hc3 => 1.0 / (1.0 - hii.expect("leverages")[i]).max(1e-8),
            Vce::Nn => unreachable!(),
        };
        res.row_mut(i).mapv_inplace(|r| w * r);
    }
    res
}

/// Nearest-neighbour residuals (`vce = "nn"`): each observation minus the
/// mean of its (at least) `nnmatch` nearest neighbours, scaled by
/// `√(J / (J + 1))`. `x` is sorted.
fn nn_residuals(s: &Side, dmat: &Array2<f64>, nnmatch: usize) -> Array2<f64> {
    let n = s.n();
    let mut res = Array2::zeros(dmat.dim());
    let tol_eps = f64::EPSILON.sqrt();
    let (x, dups, dupsid) = (&s.x, &s.dups, &s.dupsid);
    for pos in 0..n {
        let p = pos as i64;
        let mut rpos = (dups[pos] - dupsid[pos]) as i64;
        let mut lpos = dupsid[pos] as i64 - 1;
        let cap = nnmatch.min(n.saturating_sub(1)) as i64;
        while lpos + rpos < cap {
            if p - lpos - 1 < 0 {
                rpos += dups[(p + rpos + 1) as usize] as i64;
            } else if p + rpos + 1 >= n as i64 {
                lpos += dups[(p - lpos - 1) as usize] as i64;
            } else {
                let dleft = x[pos] - x[(p - lpos - 1) as usize];
                let dright = x[(p + rpos + 1) as usize] - x[pos];
                let nn_tol = dleft.max(dright) * tol_eps;
                if dleft - dright > nn_tol {
                    rpos += dups[(p + rpos + 1) as usize] as i64;
                } else if dright - dleft > nn_tol {
                    lpos += dups[(p - lpos - 1) as usize] as i64;
                } else {
                    rpos += dups[(p + rpos + 1) as usize] as i64;
                    lpos += dups[(p - lpos - 1) as usize] as i64;
                }
            }
        }
        let (lo, hi) = ((p - lpos) as usize, (p + rpos + 1) as usize);
        let ji = (hi - lo - 1) as f64;
        let sf = (ji / (ji + 1.0)).powf(0.5);
        for col in 0..dmat.ncols() {
            let mut sum = 0.0;
            for j in lo..hi {
                sum += dmat[[j, col]];
            }
            sum -= dmat[[pos, col]];
            res[[pos, col]] = sf * (dmat[[pos, col]] - sum / ji);
        }
    }
    res
}

/// Meat of the sandwich (`rdrobust_vce`): `Σᵢ (sᵀrᵢ)² RXᵢ RXᵢᵀ`, or with
/// clusters the CR1 sum over clusters of `(Σᵢ∈g RXᵢ sᵀrᵢ)(…)ᵀ` times
/// `(n − 1)/(n − k) · G/(G − 1)` (`k_df` coefficients).
fn vce_meat(
    s: &[f64],
    rx: &Array2<f64>,
    res: &Array2<f64>,
    cl: Option<&[i64]>,
    k_df: usize,
) -> Array2<f64> {
    let k = rx.ncols();
    let combined: Vec<f64> = res
        .rows()
        .into_iter()
        .map(|row| row.iter().zip(s).map(|(r, si)| r * si).sum())
        .collect();
    match cl {
        None => {
            let scaled = scale_rows(rx, &combined);
            crossprod(&scaled, &scaled)
        }
        Some(cl) => {
            let n = cl.len();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| cl[i]);
            let mut meat = Array2::<f64>::zeros((k, k));
            let mut g = 0usize;
            let mut start = 0;
            while start < n {
                let mut end = start + 1;
                while end < n && cl[order[end]] == cl[order[start]] {
                    end += 1;
                }
                let mut sv = Array1::<f64>::zeros(k);
                for &i in &order[start..end] {
                    sv.scaled_add(combined[i], &rx.row(i));
                }
                for a in 0..k {
                    for b in 0..k {
                        meat[[a, b]] += sv[a] * sv[b];
                    }
                }
                g += 1;
                start = end;
            }
            let w = if g > 1 {
                ((n - 1) as f64 / (n - k_df) as f64) * (g as f64 / (g - 1) as f64)
            } else {
                f64::NAN
            };
            meat * w
        }
    }
}

/// Leverages `hᵢᵢ = Σⱼ (R G⁻¹)ᵢⱼ (RW)ᵢⱼ`.
fn leverages(r: &Array2<f64>, inv_g: &Array2<f64>, rw: &Array2<f64>) -> Array1<f64> {
    (r.dot(inv_g) * rw).sum_axis(Axis(1))
}

/// Partialling the covariates out of a weighted local fit: `γ` (rows:
/// covariates, columns: outcome and treatment).
fn covariate_gamma(rw: &Array2<f64>, inv_g: &Array2<f64>, e: &Side, w: &[f64]) -> Array2<f64> {
    let (dt, dz) = (e.d_t(), e.d_z());
    let d = e.d();
    let u = crossprod(rw, &d);
    let zw = scale_rows(e.z.as_ref().expect("covariates"), w);
    let zwd = crossprod(&zw, &d);
    let cols_z = s![.., 1 + dt..1 + dt + dz];
    let uigu = u.slice(cols_z).t().dot(&inv_g.dot(&u));
    let zwz = &zwd.slice(cols_z) - &uigu.slice(cols_z);
    let zwy = &zwd.slice(s![.., ..1 + dt]) - &uigu.slice(s![.., ..1 + dt]);
    sym_pinv(&zwz).dot(&zwy)
}

// ---------------------------------------------------------------------------
// Bandwidth selection (rdbwselect)
// ---------------------------------------------------------------------------

/// `(V, B, R, rate)` of one bandwidth step on one side, NaN when a pilot
/// fit is not identified.
type BwParts = (f64, f64, f64, f64);

/// The (standardised) data and settings of a bandwidth computation.
struct BwCtx<'o> {
    opts: &'o RdOptions,
    vce: Vce,
    c: f64,
}

impl BwCtx<'_> {
    /// Plug-in or NN residuals of a pilot fit.
    fn pilot_residuals(
        &self,
        e: &Side,
        r: &Array2<f64>,
        inv_g: &Array2<f64>,
        rw: &Array2<f64>,
        beta: &Array2<f64>,
        d: usize,
    ) -> Array2<f64> {
        match self.vce {
            Vce::Nn => residuals(e, Vce::Nn, None, None, self.opts.nnmatch, d),
            v => {
                let fitted = r.dot(beta);
                let hii = matches!(v, Vce::Hc2 | Vce::Hc3).then(|| leverages(r, inv_g, rw));
                residuals(e, v, Some(&fitted), hii.as_ref(), self.opts.nnmatch, d)
            }
        }
    }

    /// Pilot fit of `rdrobust_bw` (`_rdrobust_bw_Vfit`): `(V_V, BConst, s)`.
    fn vfit(&self, side: &Side, o: usize, nu: usize, h_v: f64) -> Option<(f64, f64, Vec<f64>)> {
        let (c, kernel) = (self.c, self.opts.kernel);
        let keep: Vec<bool> = side
            .x
            .iter()
            .map(|&x| kernel.weight(x, c, h_v) > 0.0)
            .collect();
        let e = side.select(&keep);
        if n_unique(&e.x) < o + 1 {
            return None;
        }
        let ew: Vec<f64> = e.x.iter().map(|&x| kernel.weight(x, c, h_v)).collect();
        let r = vander(&e.x.iter().map(|x| x - c).collect::<Vec<_>>(), o);
        let rws = scale_rows(&r, &ew.iter().map(|w| w.sqrt()).collect::<Vec<_>>());
        let inv_g = spd_inverse(&crossprod(&rws, &rws)).ok()?;
        let rw = scale_rows(&r, &ew);
        let beta = inv_g.dot(&crossprod(&rw, &e.d()));
        let fact = factorial(nu);
        let dt = e.d_t();
        let s = match (e.z.is_some(), e.t.is_some()) {
            (false, false) => vec![1.0],
            (false, true) => {
                let (tau_y, tau_t) = (fact * beta[[nu, 0]], fact * beta[[nu, 1]]);
                vec![1.0 / tau_t, -(tau_y / (tau_t * tau_t))]
            }
            (true, has_t) => {
                let gamma = covariate_gamma(&rw, &inv_g, &e, &ew);
                let dz = e.d_z();
                let s_y: Vec<f64> = std::iter::once(1.0)
                    .chain(gamma.column(0).iter().map(|g| -g))
                    .collect();
                if has_t {
                    let s_t: Vec<f64> = std::iter::once(1.0)
                        .chain(gamma.column(1).iter().map(|g| -g))
                        .collect();
                    let pick = |col: usize| -> Vec<f64> {
                        std::iter::once(beta[[nu, col]])
                            .chain((0..dz).map(|j| beta[[nu, 1 + dt + j]]))
                            .collect()
                    };
                    let dot =
                        |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
                    let tau_y = fact * dot(&s_y, &pick(0));
                    let tau_t = fact * dot(&s_t, &pick(1));
                    let mut s = vec![1.0 / tau_t, -(tau_y / (tau_t * tau_t))];
                    s.extend((0..dz).map(|j| {
                        -(1.0 / tau_t) * gamma[[j, 0]] + (tau_y / (tau_t * tau_t)) * gamma[[j, 1]]
                    }));
                    s
                } else {
                    s_y
                }
            }
        };
        let res = self.pilot_residuals(&e, &r, &inv_g, &rw, &beta, o + 1);
        let meat = vce_meat(&s, &rw, &res, e.cl.as_deref(), rw.ncols());
        let v_v = inv_g.dot(&meat).dot(&inv_g)[[nu, nu]];
        let pw: Array1<f64> = e.x.iter().map(|x| powk((x - c) / h_v, o + 1)).collect();
        let ig_v = inv_g.dot(&rw.t().dot(&pw));
        // BConst = (h^j · (G⁻¹v)_j)[nu]
        let b_const = powk(h_v, nu) * ig_v[nu];
        Some((v_v, b_const, s))
    }

    /// `rdrobust_bw`.
    #[allow(clippy::too_many_arguments)]
    fn bw(
        &self,
        side: &Side,
        o: usize,
        nu: usize,
        o_b: usize,
        h_v: f64,
        h_b: f64,
        scale: f64,
    ) -> BwParts {
        let (c, kernel) = (self.c, self.opts.kernel);
        let nan = (f64::NAN, f64::NAN, f64::NAN, 1.0 / (2 * o + 3) as f64);
        let Some((v_v, b_const, s)) = self.vfit(side, o, nu, h_v) else {
            return nan;
        };
        if !v_v.is_finite() || !(h_b.is_finite() && h_b > 0.0) {
            return nan;
        }
        let keep: Vec<bool> = side
            .x
            .iter()
            .map(|&x| kernel.weight(x, c, h_b) > 0.0)
            .collect();
        let e = side.select(&keep);
        if n_unique(&e.x) < o_b + 1 {
            return nan;
        }
        let ew: Vec<f64> = e.x.iter().map(|&x| kernel.weight(x, c, h_b)).collect();
        let r = vander(&e.x.iter().map(|x| x - c).collect::<Vec<_>>(), o_b);
        let rws = scale_rows(&r, &ew.iter().map(|w| w.sqrt()).collect::<Vec<_>>());
        let Ok(inv_g) = spd_inverse(&crossprod(&rws, &rws)) else {
            return nan;
        };
        let rw = scale_rows(&r, &ew);
        let beta = inv_g.dot(&crossprod(&rw, &e.d()));
        let mut bw_reg = 0.0;
        if scale > 0.0 {
            let res = self.pilot_residuals(&e, &r, &inv_g, &rw, &beta, o_b + 1);
            let meat = vce_meat(&s, &rw, &res, e.cl.as_deref(), rw.ncols());
            let v_b = inv_g.dot(&meat).dot(&inv_g)[[o_b, o_b]];
            bw_reg = 3.0 * b_const * b_const * v_b;
        }
        let last = beta.row(o_b);
        let s_beta: f64 = s.iter().zip(last.iter()).map(|(a, b)| a * b).sum();
        let b = (2.0 * (o + 1 - nu) as f64).sqrt() * b_const * s_beta;
        let v = (2 * nu + 1) as f64 * h_v.powf((2 * nu + 1) as f64) * v_v;
        let r = scale * (2 * (o + 1 - nu)) as f64 * bw_reg;
        (v, b, r, 1.0 / (2 * o + 3) as f64)
    }
}

/// The inline bandwidth selection of `rdrobust` (`_rdbwselect_compute`).
fn select_bandwidths(left: &Side, right: &Side, c: f64, opts: &RdOptions) -> Result<RdBandwidths> {
    let (p, q) = (opts.p, opts.q());
    let x_all: Vec<f64> = left.x.iter().chain(&right.x).copied().collect();
    let y_all: Vec<f64> = left.y.iter().chain(&right.y).copied().collect();
    let n = x_all.len();
    let x_iq = quantile_type2(&x_all, 0.75) - quantile_type2(&x_all, 0.25);
    let (x_sd, y_sd, bwp) = if opts.stdvars {
        let x_sd = std_ddof1(&x_all);
        (x_sd, std_ddof1(&y_all), 1.0_f64.min((x_iq / x_sd) / 1.349))
    } else {
        (1.0, 1.0, std_ddof1(&x_all).min(x_iq / 1.349))
    };
    // A first stage without variation on a side falls back to a sharp bandwidth.
    let no_variation = |t: &Option<Vec<f64>>| t.as_ref().is_some_and(|t| var(t) == 0.0);
    let sharp_bw = opts.sharpbw || no_variation(&left.t) || no_variation(&right.t);
    let standardise = |s: &Side| -> Side {
        Side::new(
            s.x.iter().map(|x| x / x_sd).collect(),
            s.y.iter().map(|y| y / y_sd).collect(),
            if sharp_bw { None } else { s.t.clone() },
            s.z.clone(),
            s.cl.clone(),
        )
    };
    let (l, r) = (standardise(left), standardise(right));
    let c = c / x_sd;
    let (x_min, x_max) = (l.x[0], r.x[r.n() - 1]);
    let range_l = (c - l.x[0]).abs();
    let range_r = (c - r.x[r.n() - 1]).abs();

    let mut uniq_l = l.x.clone();
    uniq_l.dedup();
    uniq_l.reverse();
    let mut uniq_r = r.x.clone();
    uniq_r.dedup();
    let inspect = opts.masspoints != MassPoints::Off;
    let (m_l, m_r) = if inspect || opts.bwcheck.is_some() {
        (uniq_l.len(), uniq_r.len())
    } else {
        (l.n(), r.n())
    };
    let mut bwcheck = opts.bwcheck;
    if inspect {
        let mass = |m: usize, n: usize| 1.0 - m as f64 / n as f64;
        let detected = mass(m_l, l.n()) >= 0.2 || mass(m_r, r.n()) >= 0.2;
        if detected && bwcheck.is_none() && opts.masspoints == MassPoints::Adjust {
            bwcheck = Some(10);
        }
    }
    let m_total = if inspect { m_l + m_r } else { n };

    let c_c = opts.kernel.c_c();
    let mut c_bw = c_c * bwp * (n as f64).powf(-1.0 / 5.0);
    if opts.masspoints == MassPoints::Adjust {
        c_bw = c_c * bwp * (m_total as f64).powf(-1.0 / 5.0);
    }
    let (bw_max_l, bw_max_r) = ((c - x_min).abs(), (c - x_max).abs());
    let bw_max = bw_max_l.max(bw_max_r);
    if opts.bwrestrict {
        c_bw = c_bw.min(bw_max);
    }
    let pad = 1.0 + f64::EPSILON.sqrt();
    let bw_min = bwcheck.map(|k| {
        (
            (uniq_l[k.min(m_l) - 1] - c).abs() * pad,
            (uniq_r[k.min(m_r) - 1] - c).abs() * pad,
        )
    });
    if let Some((lo_l, lo_r)) = bw_min {
        c_bw = c_bw.max(lo_l).max(lo_r);
    }
    let restrict = |v: f64, max: f64| if opts.bwrestrict { v.min(max) } else { v };
    let floor = |v: f64| match bw_min {
        Some((lo_l, lo_r)) => v.max(lo_l).max(lo_r),
        None => v,
    };

    let ctx = BwCtx {
        opts,
        vce: effective_vce(opts.vce, left.cl.is_some())?,
        c,
    };
    let reg = opts.scaleregul;
    let c_d_l = ctx.bw(&l, q + 1, q + 1, q + 2, c_bw, range_l * pad, 0.0);
    let c_d_r = ctx.bw(&r, q + 1, q + 1, q + 2, c_bw, range_r * pad, 0.0);

    let (mut two, mut sum, mut rd) = (None, None, None);
    let sel = opts.bwselect;
    if sel.needs_two() {
        let side = |s: &Side, cd: BwParts, max: f64, lo: Option<f64>| -> (f64, f64) {
            let mut d_bw = restrict((cd.0 / (cd.1 * cd.1)).powf(cd.3), max);
            if let Some(lo) = lo {
                d_bw = d_bw.max(lo);
            }
            let cb = ctx.bw(s, q, p + 1, q + 1, c_bw, d_bw, reg);
            let b_bw = restrict((cb.0 / (cb.1 * cb.1 + reg * cb.2)).powf(cb.3), max);
            let ch = ctx.bw(s, p, 0, q, c_bw, b_bw, reg);
            let h_bw = restrict((ch.0 / (ch.1 * ch.1 + reg * ch.2)).powf(ch.3), max);
            (x_sd * h_bw, x_sd * b_bw)
        };
        let (h_l, b_l) = side(&l, c_d_l, bw_max_l, bw_min.map(|m| m.0));
        let (h_r, b_r) = side(&r, c_d_r, bw_max_r, bw_min.map(|m| m.1));
        two = Some(RdBandwidths {
            h: (h_l, h_r),
            b: (b_l, b_r),
        });
    }
    // Common bandwidths for the difference (`sign = −1`) or the sum (`+1`)
    // of the two sides' estimates.
    let common = |sign: f64| -> (f64, f64) {
        let d_bw = floor(restrict(
            ((c_d_l.0 + c_d_r.0) / (c_d_r.1 + sign * c_d_l.1).powi(2)).powf(c_d_l.3),
            bw_max,
        ));
        let cb_l = ctx.bw(&l, q, p + 1, q + 1, c_bw, d_bw, reg);
        let cb_r = ctx.bw(&r, q, p + 1, q + 1, c_bw, d_bw, reg);
        let b_bw = restrict(
            ((cb_l.0 + cb_r.0) / ((cb_r.1 + sign * cb_l.1).powi(2) + reg * (cb_r.2 + cb_l.2)))
                .powf(cb_l.3),
            bw_max,
        );
        let ch_l = ctx.bw(&l, p, 0, q, c_bw, b_bw, reg);
        let ch_r = ctx.bw(&r, p, 0, q, c_bw, b_bw, reg);
        let h_bw = restrict(
            ((ch_l.0 + ch_r.0) / ((ch_r.1 + sign * ch_l.1).powi(2) + reg * (ch_r.2 + ch_l.2)))
                .powf(ch_l.3),
            bw_max,
        );
        (x_sd * h_bw, x_sd * b_bw)
    };
    if sel.needs_sum() {
        sum = Some(common(1.0));
    }
    if sel.needs_rd() {
        rd = Some(common(-1.0));
    }
    // np.min / np.median propagate NaN.
    let min_nan = |a: f64, b: f64| {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else {
            a.min(b)
        }
    };
    let median3 = |a: f64, b: f64, c: f64| {
        if a.is_nan() || b.is_nan() || c.is_nan() {
            return f64::NAN;
        }
        let mut v = [a, b, c];
        v.sort_by(f64::total_cmp);
        v[1]
    };
    let mse = match sel {
        BwSelect::Mserd | BwSelect::Cerrd => {
            let (h, b) = rd.expect("computed");
            RdBandwidths::common(h, b)
        }
        BwSelect::Msetwo | BwSelect::Certwo => two.expect("computed"),
        BwSelect::Msesum | BwSelect::Cersum => {
            let (h, b) = sum.expect("computed");
            RdBandwidths::common(h, b)
        }
        BwSelect::Msecomb1 | BwSelect::Cercomb1 => {
            let ((h_d, b_d), (h_s, b_s)) = (rd.expect("computed"), sum.expect("computed"));
            RdBandwidths::common(min_nan(h_d, h_s), min_nan(b_d, b_s))
        }
        BwSelect::Msecomb2 | BwSelect::Cercomb2 => {
            let ((h_d, b_d), (h_s, b_s)) = (rd.expect("computed"), sum.expect("computed"));
            let t = two.expect("computed");
            RdBandwidths {
                h: (median3(h_d, h_s, t.h.0), median3(h_d, h_s, t.h.1)),
                b: (median3(b_d, b_s, t.b.0), median3(b_d, b_s, t.b.1)),
            }
        }
    };
    let bws = if sel.is_cer() {
        let base = match (left.cl.as_ref(), right.cl.as_ref()) {
            (Some(a), Some(b)) => (n_unique_i64(a) + n_unique_i64(b)) as f64,
            _ => n as f64,
        };
        let pf = p as f64;
        let cer_h = base.powf(-(pf / ((3.0 + pf) * (3.0 + 2.0 * pf))));
        RdBandwidths {
            h: (mse.h.0 * cer_h, mse.h.1 * cer_h),
            b: mse.b,
        }
    } else {
        mse
    };
    let all = [bws.h.0, bws.h.1, bws.b.0, bws.b.1];
    if !all.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(DmlError::InvalidArgument(
            "not enough variability in the running variable to compute the bandwidth".into(),
        ));
    }
    Ok(match opts.rho {
        Some(rho) => RdBandwidths {
            h: bws.h,
            b: (bws.h.0 / rho, bws.h.1 / rho),
        },
        None => bws,
    })
}

fn n_unique_i64(v: &[i64]) -> usize {
    let mut v = v.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// The variance estimator actually used: with clusters `rdrobust` switches
/// `nn`, `hc0` and `hc1` to the cluster-robust `cr1` (internally `hc1`).
fn effective_vce(vce: Vce, clustered: bool) -> Result<Vce> {
    match (vce, clustered) {
        (Vce::Nn | Vce::Hc0 | Vce::Hc1, true) => Ok(Vce::Hc1),
        (Vce::Hc2 | Vce::Hc3, true) => Err(DmlError::InvalidArgument(
            "rdrobust: the leverage-adjusted cluster-robust variances (cr2, cr3) are not available; use Vce::Hc1 (cr1) with clusters".into(),
        )),
        (v, false) => Ok(v),
    }
}

fn check_support(left: &Side, right: &Side, order: usize) -> Result<()> {
    if n_unique(&left.x).min(n_unique(&right.x)) < order + 1 {
        return Err(DmlError::InvalidArgument(format!(
            "not enough distinct running-variable values on each side of the cutoff for a polynomial of order {order}"
        )));
    }
    Ok(())
}

/// Bandwidths of `rdrobust`'s `rdbwselect` for the RD effect at the cutoff,
/// with the selector and options in `opts`.
pub fn rd_bandwidths(data: &RdData<'_>, opts: &RdOptions) -> Result<RdBandwidths> {
    opts.validate()?;
    let (left, right) = prepare(data, opts)?;
    check_support(&left, &right, opts.p)?;
    select_bandwidths(&left, &right, data.cutoff, opts)
}

/// The MSE-optimal common bandwidths `(h, b)` of `rdrobust`'s `rdbwselect`
/// (default options) for the RD effect at cutoff `c`; pass the treatment as
/// `fuzzy` for a fuzzy design.
pub fn rdbwselect<'a>(
    y: ArrayView1<'a, f64>,
    x: ArrayView1<'a, f64>,
    fuzzy: Option<ArrayView1<'a, f64>>,
    c: f64,
) -> Result<(f64, f64)> {
    let mut data = RdData::new(y, x).cutoff(c);
    data.fuzzy = fuzzy;
    let bws = rd_bandwidths(&data, &RdOptions::default())?;
    Ok((bws.h.0, bws.b.0))
}

// ---------------------------------------------------------------------------
// Estimation (rdrobust)
// ---------------------------------------------------------------------------

/// One side's local polynomial fits at bandwidths `h` (order `p`) and `b`
/// (order `q`), and the residuals for the variance.
struct SideFit {
    beta_p: Array2<f64>,
    beta_bc: Array2<f64>,
    inv_g_p: Array2<f64>,
    inv_g_q: Array2<f64>,
    rw_p: Array2<f64>,
    rw_q_h: Array2<f64>,
    q_q: Array2<f64>,
    res_h: Array2<f64>,
    res_b: Array2<f64>,
    /// `Σ ZW D` and `Σ U` terms for covariate adjustment.
    zwd: Option<Array2<f64>>,
    u_p: Option<Array2<f64>>,
    cl: Option<Vec<i64>>,
    n_h: usize,
}

fn fit_side(side: &Side, c: f64, h: f64, b: f64, opts: &RdOptions, vce: Vce) -> Result<SideFit> {
    let (p, q, kernel) = (opts.p, opts.q(), opts.kernel);
    let w_h: Vec<f64> = side.x.iter().map(|&x| kernel.weight(x, c, h)).collect();
    let w_b: Vec<f64> = side.x.iter().map(|&x| kernel.weight(x, c, b)).collect();
    let in_h: Vec<bool> = w_h.iter().map(|&w| w > 0.0).collect();
    let in_b: Vec<bool> = w_b.iter().map(|&w| w > 0.0).collect();
    let n_h = in_h.iter().filter(|&&k| k).count();
    let uniq_in = |keep: &[bool]| {
        n_unique(
            &side
                .x
                .iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(x, _)| *x)
                .collect::<Vec<_>>(),
        )
    };
    if uniq_in(&in_h) < p + 1 || uniq_in(&in_b) < q + 1 {
        return Err(DmlError::InvalidArgument(
            "not enough distinct running-variable values with positive weight on each side of the cutoff"
                .into(),
        ));
    }
    let keep = if h > b { &in_h } else { &in_b };
    let pick = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .zip(keep.iter())
            .filter(|(_, &k)| k)
            .map(|(a, _)| *a)
            .collect()
    };
    let (ww_h, ww_b) = (pick(&w_h), pick(&w_b));
    // Nearest neighbours are counted within the estimation window.
    let e = side.select(keep);
    let u: Vec<f64> = e.x.iter().map(|x| (x - c) / h).collect();
    let r_q = vander(&e.x.iter().map(|x| x - c).collect::<Vec<_>>(), q);
    let r_p = r_q.slice(s![.., ..=p]).to_owned();
    let rw_p = scale_rows(&r_p, &ww_h);
    let rw_q_b = scale_rows(&r_q, &ww_b);
    let l: Array1<f64> = rw_p
        .t()
        .dot(&Array1::from_iter(u.iter().map(|&v| powk(v, p + 1))));
    let sq = |w: &[f64]| w.iter().map(|v| v.sqrt()).collect::<Vec<_>>();
    let inv_g_q = {
        let m = scale_rows(&r_q, &sq(&ww_b));
        spd_inverse(&crossprod(&m, &m))?
    };
    let inv_g_p = {
        let m = scale_rows(&r_p, &sq(&ww_h));
        spd_inverse(&crossprod(&m, &m))?
    };
    let m: Vec<f64> = r_q
        .dot(&inv_g_q.row(p + 1))
        .iter()
        .zip(&ww_b)
        .map(|(a, w)| a * w)
        .collect();
    let hp = powk(h, p + 1);
    let mut q_q = rw_p.clone();
    for i in 0..q_q.nrows() {
        for j in 0..q_q.ncols() {
            q_q[[i, j]] -= hp * m[i] * l[j];
        }
    }
    let d = e.d();
    let beta_p = inv_g_p.dot(&crossprod(&rw_p, &d));
    let beta_q = inv_g_q.dot(&crossprod(&rw_q_b, &d));
    let beta_bc = inv_g_p.dot(&crossprod(&q_q, &d));
    let (res_h, res_b) = if vce == Vce::Nn {
        let res = residuals(&e, Vce::Nn, None, None, opts.nnmatch, p + 1);
        (res.clone(), res)
    } else {
        let (fit_p, fit_q) = (r_p.dot(&beta_p), r_q.dot(&beta_q));
        let (hii_p, hii_q) = if matches!(vce, Vce::Hc2 | Vce::Hc3) {
            (
                Some(leverages(&r_p, &inv_g_p, &rw_p)),
                Some(leverages(&r_q, &inv_g_q, &rw_q_b)),
            )
        } else {
            (None, None)
        };
        (
            residuals(&e, vce, Some(&fit_p), hii_p.as_ref(), opts.nnmatch, p + 1),
            residuals(&e, vce, Some(&fit_q), hii_q.as_ref(), opts.nnmatch, q + 1),
        )
    };
    let (zwd, u_p) = match &e.z {
        Some(z) => (
            Some(crossprod(&scale_rows(z, &ww_h), &d)),
            Some(crossprod(&rw_p, &d)),
        ),
        None => (None, None),
    };
    Ok(SideFit {
        rw_q_h: scale_rows(&r_q, &ww_h),
        beta_p,
        beta_bc,
        inv_g_p,
        inv_g_q,
        rw_p,
        q_q,
        res_h,
        res_b,
        zwd,
        u_p,
        cl: e.cl,
        n_h,
    })
}

/// Local polynomial RD estimate at the cutoff with its bias correction, as
/// `rdrobust` with the options in `opts`. `bandwidths` fixes `h` and `b`
/// (with `opts.rho`, `b = h / rho`); `None` selects them with
/// [`rd_bandwidths`]. A fuzzy design estimates the ratio of the outcome and
/// treatment jumps; covariates adjust both.
pub fn rd_estimate(
    data: &RdData<'_>,
    opts: &RdOptions,
    bandwidths: Option<RdBandwidths>,
) -> Result<RdEstimate> {
    opts.validate()?;
    let q = opts.q();
    let c = data.cutoff;
    let (left, right) = prepare(data, opts)?;
    check_support(&left, &right, q)?;
    if let (Some(lt), Some(rt)) = (&left.t, &right.t) {
        if var(lt) == 0.0
            && var(rt) == 0.0
            && (np_mean(lt) - np_mean(rt)).abs() < f64::EPSILON.sqrt()
        {
            return Err(DmlError::InvalidArgument(
                "fuzzy RD: the treatment has no variation and no jump at the cutoff".into(),
            ));
        }
    }
    let vce = effective_vce(opts.vce, data.cluster.is_some())?;
    let n = left.n() + right.n();
    let bws = match bandwidths {
        Some(bws) => {
            let all = [bws.h.0, bws.h.1, bws.b.0, bws.b.1];
            if !all.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(DmlError::InvalidArgument(
                    "bandwidths must be positive and finite".into(),
                ));
            }
            match opts.rho {
                Some(rho) => RdBandwidths {
                    h: bws.h,
                    b: (bws.h.0 / rho, bws.h.1 / rho),
                },
                None => bws,
            }
        }
        None if n < 20 => {
            let h = (c - left.x[0])
                .abs()
                .max((right.x[right.n() - 1] - c).abs());
            RdBandwidths::common(h, opts.rho.map_or(h, |rho| h / rho))
        }
        None => select_bandwidths(&left, &right, c, opts)?,
    };
    let fl = fit_side(&left, c, bws.h.0, bws.b.0, opts, vce)?;
    let fr = fit_side(&right, c, bws.h.1, bws.b.1, opts, vce)?;

    let fuzzy = data.fuzzy.is_some();
    let dt = usize::from(fuzzy);
    let dz = left.d_z();
    let sp = opts.scalepar;
    let beta_p = &fr.beta_p - &fl.beta_p;
    let beta_bc = &fr.beta_bc - &fl.beta_bc;
    let (tau_cl, tau_bc, s_y) = if dz == 0 {
        let (tau_y_cl, tau_y_bc) = (sp * beta_p[[0, 0]], sp * beta_bc[[0, 0]]);
        if fuzzy {
            let (tau_t_cl, tau_t_bc) = (beta_p[[0, 1]], beta_bc[[0, 1]]);
            let tau_cl = tau_y_cl / tau_t_cl;
            let s = vec![1.0 / tau_t_cl, -(tau_y_cl / (tau_t_cl * tau_t_cl))];
            let tau_bc = tau_cl - (s[0] * (tau_y_cl - tau_y_bc) + s[1] * (tau_t_cl - tau_t_bc));
            (tau_cl, tau_bc, s)
        } else {
            (tau_y_cl, tau_y_bc, vec![1.0])
        }
    } else {
        // Partial the covariates out of the pooled local fits (pinv, as
        // rdrobust with covs_drop).
        let cols_z = s![.., 1 + dt..1 + dt + dz];
        let side_terms = |f: &SideFit| -> (Array2<f64>, Array2<f64>) {
            let (zwd, u) = (f.zwd.as_ref().expect("covs"), f.u_p.as_ref().expect("covs"));
            let uigu = u.slice(cols_z).t().dot(&f.inv_g_p.dot(u));
            (
                &zwd.slice(cols_z) - &uigu.slice(cols_z),
                &zwd.slice(s![.., ..1 + dt]) - &uigu.slice(s![.., ..1 + dt]),
            )
        };
        let ((zwz_l, zwy_l), (zwz_r, zwy_r)) = (side_terms(&fl), side_terms(&fr));
        let gamma = sym_pinv(&(&zwz_r + &zwz_l)).dot(&(&zwy_r + &zwy_l));
        let s_y: Vec<f64> = std::iter::once(1.0)
            .chain(gamma.column(0).iter().map(|g| -g))
            .collect();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let row = |beta: &Array2<f64>, col: usize| -> Vec<f64> {
            std::iter::once(beta[[0, col]])
                .chain((0..dz).map(|j| beta[[0, 1 + dt + j]]))
                .collect()
        };
        if fuzzy {
            let s_t: Vec<f64> = std::iter::once(1.0)
                .chain(gamma.column(1).iter().map(|g| -g))
                .collect();
            let tau_y_cl = sp * dot(&s_y, &row(&beta_p, 0));
            let tau_t_cl = dot(&s_t, &row(&beta_p, 1));
            let tau_y_bc = sp * dot(&s_y, &row(&beta_bc, 0));
            let tau_t_bc = dot(&s_t, &row(&beta_bc, 1));
            let tau_cl = tau_y_cl / tau_t_cl;
            let s2 = [1.0 / tau_t_cl, -(tau_y_cl / (tau_t_cl * tau_t_cl))];
            let tau_bc = tau_cl - (s2[0] * (tau_y_cl - tau_y_bc) + s2[1] * (tau_t_cl - tau_t_bc));
            let mut s = s2.to_vec();
            s.extend((0..dz).map(|j| {
                -(1.0 / tau_t_cl) * gamma[[j, 0]]
                    + (tau_y_cl / (tau_t_cl * tau_t_cl)) * gamma[[j, 1]]
            }));
            (tau_cl, tau_bc, s)
        } else {
            let tau_cl = sp * dot(&s_y, &beta_p.row(0).to_vec());
            let tau_bc = sp * dot(&s_y, &beta_bc.row(0).to_vec());
            (tau_cl, tau_bc, s_y)
        }
    };

    let hb_match = bws.h == bws.b;
    let clustered = data.cluster.is_some();
    let variance = |f: &SideFit, robust: bool| -> f64 {
        let cl = f.cl.as_deref();
        if !robust {
            let k = f.rw_p.ncols();
            return f
                .inv_g_p
                .dot(&vce_meat(&s_y, &f.rw_p, &f.res_h, cl, k))
                .dot(&f.inv_g_p)[[0, 0]];
        }
        if hb_match && clustered {
            // The q-order regression directly (rdrobust's h = b path).
            let k = f.rw_q_h.ncols();
            f.inv_g_q
                .dot(&vce_meat(&s_y, &f.rw_q_h, &f.res_b, cl, k))
                .dot(&f.inv_g_q)[[0, 0]]
        } else {
            f.inv_g_p
                .dot(&vce_meat(&s_y, &f.q_q, &f.res_b, cl, q + 1))
                .dot(&f.inv_g_p)[[0, 0]]
        }
    };
    let se_cl = (sp * sp * (variance(&fl, false) + variance(&fr, false))).sqrt();
    let se_rb = (sp * sp * (variance(&fl, true) + variance(&fr, true))).sqrt();
    let coef = [tau_cl, tau_bc, tau_bc];
    let se = [se_cl, se_cl, se_rb];
    let quant = -norm_ppf((1.0 - opts.level) / 2.0);
    let ci = std::array::from_fn(|k| [coef[k] - quant * se[k], coef[k] + quant * se[k]]);
    Ok(RdEstimate {
        coef,
        se,
        ci,
        h: bws.h,
        b: bws.b,
        n_h: (fl.n_h, fr.n_h),
        n: (left.n(), right.n()),
    })
}

/// Local linear RD estimate at cutoff `c` with a quadratic bias correction
/// and nearest-neighbour standard errors, as `rdrobust` with its default
/// options. `bandwidths` gives `(h, b)` (both sides); `None` selects them
/// with [`rdbwselect`]. Pass the treatment as `fuzzy` for a fuzzy design (the
/// effect is then the ratio of the outcome and treatment jumps).
pub fn rdrobust<'a>(
    y: ArrayView1<'a, f64>,
    x: ArrayView1<'a, f64>,
    fuzzy: Option<ArrayView1<'a, f64>>,
    c: f64,
    bandwidths: Option<(f64, f64)>,
) -> Result<RdEstimate> {
    let mut data = RdData::new(y, x).cutoff(c);
    data.fuzzy = fuzzy;
    rd_estimate(
        &data,
        &RdOptions::default(),
        bandwidths.map(|(h, b)| RdBandwidths::common(h, b)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn duplicates_count_runs() {
        let (d, id) = duplicates(&[1.0, 2.0, 2.0, 2.0, 3.0]);
        assert_eq!(d, vec![1, 3, 3, 3, 1]);
        assert_eq!(id, vec![1, 1, 2, 3, 1]);
    }

    #[test]
    fn quantile_type2_matches_r() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(quantile_type2(&x, 0.25), 1.5);
        assert_eq!(quantile_type2(&x, 0.75), 3.5);
        assert_eq!(quantile_type2(&[1.0, 2.0, 3.0], 0.25), 1.0);
    }

    #[test]
    fn collinear_covariates_are_dropped() {
        let z = array![
            [1.0, 2.0, 0.5],
            [2.0, 4.0, -1.0],
            [3.0, 6.0, 2.0],
            [4.0, 8.0, 0.0]
        ];
        let keep = independent_columns(&z);
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&2));
        let full = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        assert_eq!(independent_columns(&full), vec![0, 1]);
    }

    #[test]
    fn options_are_validated() {
        let bad = |o: RdOptions| o.validate().is_err();
        assert!(bad(RdOptions {
            q: Some(1),
            ..RdOptions::default()
        }));
        assert!(bad(RdOptions {
            level: 95.0,
            ..RdOptions::default()
        }));
        assert!(bad(RdOptions {
            nnmatch: 0,
            ..RdOptions::default()
        }));
        assert!(bad(RdOptions {
            rho: Some(0.0),
            ..RdOptions::default()
        }));
        assert!(RdOptions::default().validate().is_ok());
        assert!(effective_vce(Vce::Hc2, true).is_err());
        assert_eq!(effective_vce(Vce::Nn, true).unwrap(), Vce::Hc1);
    }
}

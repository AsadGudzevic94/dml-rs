//! Local polynomial regression discontinuity (RD) estimation: a port of the
//! default path of the `rdrobust` package (Calonico, Cattaneo and Titiunik,
//! 2014; Calonico, Cattaneo and Farrell, 2020), used by
//! [`RdFlex`](crate::RdFlex).
//!
//! - [`rdbwselect`]: the MSE-optimal common bandwidth (`bwselect = "mserd"`)
//!   for the point estimate (`h`) and the bias correction (`b`).
//! - [`rdrobust`]: the local linear estimate at the cutoff with a quadratic
//!   bias correction, nearest-neighbour standard errors and the
//!   conventional, bias-corrected and robust confidence intervals.
//!
//! Defaults as in `rdrobust`: triangular kernel, `p = 1`, `q = 2`,
//! `vce = "nn"` with 3 neighbours, standardised variables for bandwidth
//! selection and `masspoints = "adjust"`. Sharp and fuzzy designs are
//! supported; covariates, clusters and weights are not.

use ndarray::{Array1, Array2, ArrayView1};

use crate::compat::{np_mean, np_sum};
use crate::error::{DmlError, Result};
use crate::linalg::spd_inverse;
use crate::stats::norm_ppf;

const P: usize = 1;
const Q: usize = 2;
const NN_MATCHES: usize = 3;
/// `C_c` of the triangular kernel.
const C_C: f64 = 2.576;

/// Result of [`rdrobust`]: rows are the conventional, bias-corrected and
/// robust (bias-corrected with robust standard error) estimates.
#[derive(Debug, Clone)]
pub struct RdEstimate {
    pub coef: [f64; 3],
    pub se: [f64; 3],
    /// 95% confidence intervals `[lower, upper]`.
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

/// One side of the cutoff, sorted by the running variable.
struct Side {
    x: Vec<f64>,
    y: Vec<f64>,
    t: Option<Vec<f64>>,
    dups: Vec<usize>,
    dupsid: Vec<usize>,
}

impl Side {
    fn new(x: Vec<f64>, y: Vec<f64>, t: Option<Vec<f64>>) -> Self {
        let (dups, dupsid) = duplicates(&x);
        Self {
            x,
            y,
            t,
            dups,
            dupsid,
        }
    }

    fn n(&self) -> usize {
        self.x.len()
    }

    fn select(&self, keep: &[bool]) -> Side {
        let pick = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(a, _)| *a)
                .collect()
        };
        let pick_n = |v: &[usize]| -> Vec<usize> {
            v.iter()
                .zip(keep)
                .filter(|(_, &k)| k)
                .map(|(a, _)| *a)
                .collect()
        };
        Side {
            x: pick(&self.x),
            y: pick(&self.y),
            t: self.t.as_ref().map(|t| pick(t)),
            dups: pick_n(&self.dups),
            dupsid: pick_n(&self.dupsid),
        }
    }

    /// Outcome (and treatment) columns.
    fn d(&self) -> Array2<f64> {
        let k = 1 + usize::from(self.t.is_some());
        let mut d = Array2::zeros((self.n(), k));
        for i in 0..self.n() {
            d[[i, 0]] = self.y[i];
            if let Some(t) = &self.t {
                d[[i, 1]] = t[i];
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

/// Triangular kernel weight `(1 − |u|)₊ / h` with `u = (x − c) / h`.
fn kweight(x: f64, c: f64, h: f64) -> f64 {
    let u = (x - c) / h;
    ((1.0 - u.abs()) * f64::from(u.abs() <= 1.0)) / h
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

/// Nearest-neighbour residuals (`vce = "nn"`): each observation minus the
/// mean of its (at least) `NN_MATCHES` nearest neighbours, scaled by
/// `√(J / (J + 1))`. `x` is sorted.
fn nn_residuals(s: &Side) -> Array2<f64> {
    let n = s.n();
    let k = 1 + usize::from(s.t.is_some());
    let mut res = Array2::zeros((n, k));
    let tol_eps = f64::EPSILON.sqrt();
    let (x, dups, dupsid) = (&s.x, &s.dups, &s.dupsid);
    for pos in 0..n {
        let p = pos as i64;
        let mut rpos = (dups[pos] - dupsid[pos]) as i64;
        let mut lpos = dupsid[pos] as i64 - 1;
        let cap = NN_MATCHES.min(n.saturating_sub(1)) as i64;
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
        let resid = |v: &[f64]| -> f64 {
            let mut sum = 0.0;
            for &vj in &v[lo..hi] {
                sum += vj;
            }
            sum -= v[pos];
            sf * (v[pos] - sum / ji)
        };
        res[[pos, 0]] = resid(&s.y);
        if let Some(t) = &s.t {
            res[[pos, 1]] = resid(t);
        }
    }
    res
}

/// Meat of the sandwich: `Σᵢ (sᵀrᵢ)² RXᵢ RXᵢᵀ`.
fn vce(s: &[f64], rx: &Array2<f64>, res: &Array2<f64>) -> Array2<f64> {
    let r: Vec<f64> = if res.ncols() == 1 {
        res.column(0).to_vec()
    } else {
        res.rows()
            .into_iter()
            .map(|row| row.dot(&ArrayView1::from(s)))
            .collect()
    };
    let scaled = scale_rows(rx, &r);
    crossprod(&scaled, &scaled)
}

/// NumPy's `std(ddof=1)`.
fn std_ddof1(v: &[f64]) -> f64 {
    let mean = np_sum(v) / v.len() as f64;
    let sq: Vec<f64> = v.iter().map(|x| (x - mean) * (x - mean)).collect();
    (np_sum(&sq) / (v.len() - 1) as f64).sqrt()
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

/// Sort by the running variable (stably) and split at `c`.
fn split(
    y: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    t: Option<ArrayView1<'_, f64>>,
    c: f64,
) -> Result<(Side, Side)> {
    let n = x.len();
    if y.len() != n || t.is_some_and(|t| t.len() != n) {
        return Err(DmlError::DimensionMismatch(
            "rdrobust: y, x and fuzzy need equal lengths".into(),
        ));
    }
    if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
        return Err(DmlError::InvalidArgument(
            "rdrobust: data contain NaN or infinite values".into(),
        ));
    }
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
            t.map(|t| rows.iter().map(|&i| t[i]).collect()),
        )
    };
    Ok((side(true), side(false)))
}

// ---------------------------------------------------------------------------
// Bandwidth selection (rdbwselect, mserd)
// ---------------------------------------------------------------------------

/// `(V, B, R, rate)` of one bandwidth step on one side, NaN when a pilot
/// fit is not identified.
type BwParts = (f64, f64, f64, f64);

/// Pilot fit of `rdrobust_bw`: `(V_V, BConst, s)`.
fn bw_vfit(side: &Side, c: f64, o: usize, nu: usize, h_v: f64) -> Option<(f64, f64, Vec<f64>)> {
    let keep: Vec<bool> = side.x.iter().map(|&x| kweight(x, c, h_v) > 0.0).collect();
    let e = side.select(&keep);
    if n_unique(&e.x) < o + 1 {
        return None;
    }
    let ew: Vec<f64> = e.x.iter().map(|&x| kweight(x, c, h_v)).collect();
    let u: Vec<f64> = e.x.iter().map(|x| x - c).collect();
    let r = vander(&u, o);
    let sqrt_w: Vec<f64> = ew.iter().map(|w| w.sqrt()).collect();
    let rws = scale_rows(&r, &sqrt_w);
    let inv_g = spd_inverse(&crossprod(&rws, &rws)).ok()?;
    let rw = scale_rows(&r, &ew);
    let beta = inv_g.dot(&crossprod(&rw, &e.d()));
    let fact = (1..=nu).product::<usize>() as f64;
    let s = if e.t.is_some() {
        let (tau_y, tau_t) = (fact * beta[[nu, 0]], fact * beta[[nu, 1]]);
        vec![1.0 / tau_t, -(tau_y / (tau_t * tau_t))]
    } else {
        vec![1.0]
    };
    let res = nn_residuals(&e);
    let v_v = inv_g.dot(&vce(&s, &rw, &res)).dot(&inv_g)[[nu, nu]];
    let pw: Array1<f64> = e.x.iter().map(|x| powk((x - c) / h_v, o + 1)).collect();
    let v = rw.t().dot(&pw);
    let ig_v = inv_g.dot(&v);
    // BConst = (h^j · (G⁻¹v)_j)[nu]
    let b_const = powk(h_v, nu) * ig_v[nu];
    Some((v_v, b_const, s))
}

#[allow(clippy::too_many_arguments)]
fn rdrobust_bw(
    side: &Side,
    c: f64,
    o: usize,
    nu: usize,
    o_b: usize,
    h_v: f64,
    h_b: f64,
    scale: f64,
) -> BwParts {
    let nan = (f64::NAN, f64::NAN, f64::NAN, 1.0 / (2 * o + 3) as f64);
    let Some((v_v, b_const, s)) = bw_vfit(side, c, o, nu, h_v) else {
        return nan;
    };
    if !v_v.is_finite() || !(h_b.is_finite() && h_b > 0.0) {
        return nan;
    }
    let keep: Vec<bool> = side.x.iter().map(|&x| kweight(x, c, h_b) > 0.0).collect();
    let e = side.select(&keep);
    if n_unique(&e.x) < o_b + 1 {
        return nan;
    }
    let ew: Vec<f64> = e.x.iter().map(|&x| kweight(x, c, h_b)).collect();
    let u: Vec<f64> = e.x.iter().map(|x| x - c).collect();
    let r = vander(&u, o_b);
    let sqrt_w: Vec<f64> = ew.iter().map(|w| w.sqrt()).collect();
    let rws = scale_rows(&r, &sqrt_w);
    let Ok(inv_g) = spd_inverse(&crossprod(&rws, &rws)) else {
        return nan;
    };
    let rw = scale_rows(&r, &ew);
    let beta = inv_g.dot(&crossprod(&rw, &e.d()));
    let mut bw_reg = 0.0;
    if scale > 0.0 {
        let res = nn_residuals(&e);
        let v_b = inv_g.dot(&vce(&s, &rw, &res)).dot(&inv_g)[[o_b, o_b]];
        bw_reg = 3.0 * b_const * b_const * v_b;
    }
    let last = beta.row(o_b);
    let s_beta: f64 = s.iter().zip(last.iter()).map(|(a, b)| a * b).sum();
    let b = (2.0 * (o + 1 - nu) as f64).sqrt() * b_const * s_beta;
    let v = (2 * nu + 1) as f64 * h_v.powf((2 * nu + 1) as f64) * v_v;
    let r = scale * (2 * (o + 1 - nu)) as f64 * bw_reg;
    (v, b, r, 1.0 / (2 * o + 3) as f64)
}

/// MSE-optimal common bandwidths `(h, b)` on the original scale.
fn select_bandwidth(left: &Side, right: &Side, c: f64) -> Result<(f64, f64)> {
    let x_all: Vec<f64> = left.x.iter().chain(&right.x).copied().collect();
    let y_all: Vec<f64> = left.y.iter().chain(&right.y).copied().collect();
    let x_iq = quantile_type2(&x_all, 0.75) - quantile_type2(&x_all, 0.25);
    let (x_sd, y_sd) = (std_ddof1(&x_all), std_ddof1(&y_all));
    let bwp = 1.0_f64.min((x_iq / x_sd) / 1.349);
    // A first stage without variation on a side falls back to a sharp bandwidth.
    let constant = |t: &Option<Vec<f64>>| t.as_ref().is_some_and(|t| t.iter().all(|&v| v == t[0]));
    let sharp_bw = constant(&left.t) || constant(&right.t);
    let standardise = |s: &Side| -> Side {
        Side::new(
            s.x.iter().map(|x| x / x_sd).collect(),
            s.y.iter().map(|y| y / y_sd).collect(),
            if sharp_bw { None } else { s.t.clone() },
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
    let (m_l, m_r) = (uniq_l.len(), uniq_r.len());
    let mass = |m: usize, n: usize| 1.0 - m as f64 / n as f64;
    let bwcheck = (mass(m_l, l.n()) >= 0.2 || mass(m_r, r.n()) >= 0.2).then_some(10usize);

    let mut c_bw = C_C * bwp * ((m_l + m_r) as f64).powf(-1.0 / 5.0);
    let (bw_max_l, bw_max_r) = ((c - x_min).abs(), (c - x_max).abs());
    let bw_max = bw_max_l.max(bw_max_r);
    c_bw = c_bw.min(bw_max);
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

    let c_d_l = rdrobust_bw(&l, c, Q + 1, Q + 1, Q + 2, c_bw, range_l * pad, 0.0);
    let c_d_r = rdrobust_bw(&r, c, Q + 1, Q + 1, Q + 2, c_bw, range_r * pad, 0.0);
    let mut d_bw = ((c_d_l.0 + c_d_r.0) / (c_d_r.1 - c_d_l.1).powi(2)).powf(c_d_l.3);
    d_bw = d_bw.min(bw_max);
    if let Some((lo_l, lo_r)) = bw_min {
        d_bw = d_bw.max(lo_l).max(lo_r);
    }
    let c_b_l = rdrobust_bw(&l, c, Q, P + 1, Q + 1, c_bw, d_bw, 1.0);
    let c_b_r = rdrobust_bw(&r, c, Q, P + 1, Q + 1, c_bw, d_bw, 1.0);
    let b_bw = ((c_b_l.0 + c_b_r.0) / ((c_b_r.1 - c_b_l.1).powi(2) + (c_b_r.2 + c_b_l.2)))
        .powf(c_b_l.3)
        .min(bw_max);
    let c_h_l = rdrobust_bw(&l, c, P, 0, Q, c_bw, b_bw, 1.0);
    let c_h_r = rdrobust_bw(&r, c, P, 0, Q, c_bw, b_bw, 1.0);
    let h_bw = ((c_h_l.0 + c_h_r.0) / ((c_h_r.1 - c_h_l.1).powi(2) + (c_h_r.2 + c_h_l.2)))
        .powf(c_h_l.3)
        .min(bw_max);
    let (h, b) = (x_sd * h_bw, x_sd * b_bw);
    if !(h.is_finite() && h > 0.0 && b.is_finite() && b > 0.0) {
        return Err(DmlError::InvalidArgument(
            "not enough variability in the running variable to compute the bandwidth".into(),
        ));
    }
    Ok((h, b))
}

/// The MSE-optimal common bandwidths `(h, b)` of `rdrobust`'s `rdbwselect`
/// (`bwselect = "mserd"`) for the RD effect at cutoff `c`; pass the
/// treatment as `fuzzy` for a fuzzy design.
pub fn rdbwselect(
    y: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    fuzzy: Option<ArrayView1<'_, f64>>,
    c: f64,
) -> Result<(f64, f64)> {
    let (left, right) = split(y, x, fuzzy, c)?;
    check_support(&left, &right, P)?;
    select_bandwidth(&left, &right, c)
}

fn check_support(left: &Side, right: &Side, order: usize) -> Result<()> {
    if n_unique(&left.x).min(n_unique(&right.x)) < order + 1 {
        return Err(DmlError::InvalidArgument(format!(
            "not enough distinct running-variable values on each side of the cutoff for a polynomial of order {order}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Estimation (rdrobust)
// ---------------------------------------------------------------------------

/// One side's fit: `(β_p, β_bc, V_cl, V_rb factors)`.
struct SideFit {
    beta_p: Array2<f64>,
    beta_bc: Array2<f64>,
    inv_g_p: Array2<f64>,
    rw_p: Array2<f64>,
    q_q: Array2<f64>,
    res: Array2<f64>,
    n_h: usize,
}

fn fit_side(side: &Side, c: f64, h: f64, b: f64) -> Result<SideFit> {
    let w_h: Vec<f64> = side.x.iter().map(|&x| kweight(x, c, h)).collect();
    let w_b: Vec<f64> = side.x.iter().map(|&x| kweight(x, c, b)).collect();
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
    if uniq_in(&in_h) < P + 1 || uniq_in(&in_b) < Q + 1 {
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
    let e = {
        let s = side.select(keep);
        Side::new(s.x, s.y, s.t)
    };
    let u: Vec<f64> = e.x.iter().map(|x| (x - c) / h).collect();
    let r_q = vander(&e.x.iter().map(|x| x - c).collect::<Vec<_>>(), Q);
    let r_p = r_q.slice(ndarray::s![.., ..=P]).to_owned();
    let rw_p = scale_rows(&r_p, &ww_h);
    let l: Array1<f64> = rw_p
        .t()
        .dot(&Array1::from_iter(u.iter().map(|&v| powk(v, P + 1))));
    let sq = |w: &[f64]| w.iter().map(|v| v.sqrt()).collect::<Vec<_>>();
    let g_q = {
        let m = scale_rows(&r_q, &sq(&ww_b));
        crossprod(&m, &m)
    };
    let inv_g_q = spd_inverse(&g_q)?;
    let inv_g_p = {
        let m = scale_rows(&r_p, &sq(&ww_h));
        spd_inverse(&crossprod(&m, &m))?
    };
    let m: Vec<f64> = r_q
        .dot(&inv_g_q.row(P + 1))
        .iter()
        .zip(&ww_b)
        .map(|(a, w)| a * w)
        .collect();
    let hp = powk(h, P + 1);
    let mut q_q = rw_p.clone();
    for i in 0..q_q.nrows() {
        for j in 0..q_q.ncols() {
            q_q[[i, j]] -= hp * m[i] * l[j];
        }
    }
    let d = e.d();
    let beta_p = inv_g_p.dot(&crossprod(&rw_p, &d));
    let beta_bc = inv_g_p.dot(&crossprod(&q_q, &d));
    let res = nn_residuals(&e);
    Ok(SideFit {
        beta_p,
        beta_bc,
        inv_g_p,
        rw_p,
        q_q,
        res,
        n_h,
    })
}

/// Local linear RD estimate at cutoff `c` with a quadratic bias correction
/// and nearest-neighbour standard errors, as `rdrobust`. `bandwidths` gives
/// `(h, b)` (both sides); `None` selects them with [`rdbwselect`]. Pass the
/// treatment as `fuzzy` for a fuzzy design (the effect is then the ratio of
/// the outcome and treatment jumps).
pub fn rdrobust(
    y: ArrayView1<'_, f64>,
    x: ArrayView1<'_, f64>,
    fuzzy: Option<ArrayView1<'_, f64>>,
    c: f64,
    bandwidths: Option<(f64, f64)>,
) -> Result<RdEstimate> {
    let (left, right) = split(y, x, fuzzy, c)?;
    check_support(&left, &right, Q)?;
    if let (Some(lt), Some(rt)) = (&left.t, &right.t) {
        let constant = |t: &[f64]| t.iter().all(|&v| v == t[0]);
        if constant(lt) && constant(rt) && (np_mean(lt) - np_mean(rt)).abs() < f64::EPSILON.sqrt() {
            return Err(DmlError::InvalidArgument(
                "fuzzy RD: the treatment has no variation and no jump at the cutoff".into(),
            ));
        }
    }
    let n = left.n() + right.n();
    let (h, b) = match bandwidths {
        Some((h, b)) => {
            if !(h.is_finite() && h > 0.0 && b.is_finite() && b > 0.0) {
                return Err(DmlError::InvalidArgument(
                    "bandwidths must be positive and finite".into(),
                ));
            }
            (h, b)
        }
        None if n < 20 => {
            let h = (c - left.x[0])
                .abs()
                .max((right.x[right.n() - 1] - c).abs());
            (h, h)
        }
        None => select_bandwidth(&left, &right, c)?,
    };
    let fl = fit_side(&left, c, h, b)?;
    let fr = fit_side(&right, c, h, b)?;

    let (tau_y_cl, tau_y_bc) = (
        fr.beta_p[[0, 0]] - fl.beta_p[[0, 0]],
        fr.beta_bc[[0, 0]] - fl.beta_bc[[0, 0]],
    );
    let (tau_cl, tau_bc, s) = if fuzzy.is_some() {
        let tau_t_cl = fr.beta_p[[0, 1]] - fl.beta_p[[0, 1]];
        let tau_t_bc = fr.beta_bc[[0, 1]] - fl.beta_bc[[0, 1]];
        let tau_cl = tau_y_cl / tau_t_cl;
        let s = vec![1.0 / tau_t_cl, -(tau_y_cl / (tau_t_cl * tau_t_cl))];
        let tau_bc = tau_cl - (s[0] * (tau_y_cl - tau_y_bc) + s[1] * (tau_t_cl - tau_t_bc));
        (tau_cl, tau_bc, s)
    } else {
        (tau_y_cl, tau_y_bc, vec![1.0])
    };
    let variance = |f: &SideFit, robust: bool| -> f64 {
        let rx = if robust { &f.q_q } else { &f.rw_p };
        f.inv_g_p.dot(&vce(&s, rx, &f.res)).dot(&f.inv_g_p)[[0, 0]]
    };
    let se_cl = (variance(&fl, false) + variance(&fr, false)).sqrt();
    let se_rb = (variance(&fl, true) + variance(&fr, true)).sqrt();
    let coef = [tau_cl, tau_bc, tau_bc];
    let se = [se_cl, se_cl, se_rb];
    let quant = -norm_ppf(0.025);
    let ci = std::array::from_fn(|k| [coef[k] - quant * se[k], coef[k] + quant * se[k]]);
    Ok(RdEstimate {
        coef,
        se,
        ci,
        h: (h, h),
        b: (b, b),
        n_h: (fl.n_h, fr.n_h),
        n: (left.n(), right.n()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

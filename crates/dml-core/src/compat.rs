//! Bit-faithful ports of the numerical routines that DoubleML's quantile
//! models rely on, so that dml-rs reproduces them exactly on the same data:
//!
//! - NumPy's legacy `RandomState` (MT19937) and its `permutation`;
//! - scikit-learn's stratified `train_test_split` and `StratifiedKFold`
//!   (without shuffling);
//! - SciPy's `bracket`, Brent minimiser (`minimize_scalar(method="brent")`)
//!   and `brentq` root finder;
//! - statsmodels' weighted Gaussian KDE with Silverman's bandwidth;
//! - NumPy's linear `quantile` and SciPy's `scoreatpercentile`.
//!
//! Every operation follows the original's order of floating-point steps.

use crate::error::{DmlError, Result};

// ---------------------------------------------------------------------------
// NumPy legacy RandomState (MT19937)
// ---------------------------------------------------------------------------

/// The Mersenne Twister behind NumPy's legacy `np.random.RandomState(seed)`.
pub(crate) struct Mt19937 {
    mt: [u32; 624],
    index: usize,
}

impl Mt19937 {
    /// `np.random.RandomState(seed)` for an integer seed.
    pub(crate) fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(i + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[i] = v;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// NumPy's `random_interval(max)`: uniform on `0..=max` by masked rejection.
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for shift in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> shift;
        }
        assert!(
            max <= 0xffff_ffff,
            "intervals beyond 32 bits are not needed"
        );
        loop {
            let value = u64::from(self.next_u32()) & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// Legacy `RandomState.shuffle` of a 1-D sequence (Fisher–Yates from the end).
    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.interval(i as u64) as usize;
            values.swap(i, j);
        }
    }

    /// `RandomState.permutation(n)`.
    pub(crate) fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut out: Vec<usize> = (0..n).collect();
        self.shuffle(&mut out);
        out
    }
}

// ---------------------------------------------------------------------------
// scikit-learn sample splitting
// ---------------------------------------------------------------------------

/// Sorted distinct labels and each observation's index among them (`np.unique`
/// with `return_inverse`). The labels used here are small non-negative
/// numbers, for which numeric and scikit-learn's string order coincide.
fn unique_inverse(labels: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut classes = labels.to_vec();
    classes.sort_by(f64::total_cmp);
    classes.dedup();
    let inverse = labels
        .iter()
        .map(|v| {
            classes
                .binary_search_by(|c| c.total_cmp(v))
                .expect("present")
        })
        .collect();
    (classes, inverse)
}

/// scikit-learn's `_approximate_mode`: split `n_draws` over classes in
/// proportion to their counts, breaking ties at random.
fn approximate_mode(class_counts: &[usize], n_draws: usize, rng: &mut Mt19937) -> Vec<usize> {
    let total: usize = class_counts.iter().sum();
    let continuous: Vec<f64> = class_counts
        .iter()
        .map(|&c| c as f64 / total as f64 * n_draws as f64)
        .collect();
    let mut floored: Vec<f64> = continuous.iter().map(|c| c.floor()).collect();
    let mut need_to_add = (n_draws as f64 - floored.iter().sum::<f64>()) as i64;
    if need_to_add > 0 {
        let remainder: Vec<f64> = continuous
            .iter()
            .zip(&floored)
            .map(|(c, f)| c - f)
            .collect();
        let mut values = remainder.clone();
        values.sort_by(|a, b| b.total_cmp(a));
        values.dedup();
        for value in values {
            let inds: Vec<usize> = (0..remainder.len())
                .filter(|&i| remainder[i] == value)
                .collect();
            let add_now = inds.len().min(need_to_add as usize);
            // `rng.choice(inds, size=add_now, replace=False)` permutes all of `inds`.
            let perm = rng.permutation(inds.len());
            for &k in &perm[..add_now] {
                floored[inds[k]] += 1.0;
            }
            need_to_add -= add_now as i64;
            if need_to_add == 0 {
                break;
            }
        }
    }
    floored.iter().map(|&f| f as usize).collect()
}

/// `train_test_split(range(n), test_size=0.5, random_state=seed, stratify=labels)`:
/// positions of the training and test parts, in scikit-learn's order.
pub(crate) fn stratified_train_test_split(
    labels: &[f64],
    seed: u32,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let n = labels.len();
    let n_test = (0.5 * n as f64).ceil() as usize;
    let n_train = n - n_test;
    let (classes, y_indices) = unique_inverse(labels);
    let mut class_counts = vec![0usize; classes.len()];
    for &c in &y_indices {
        class_counts[c] += 1;
    }
    if class_counts.iter().any(|&c| c < 2) {
        return Err(DmlError::InvalidArgument(
            "every treatment / instrument group needs at least two observations per fold".into(),
        ));
    }
    if n_train < classes.len() || n_test < classes.len() {
        return Err(DmlError::InvalidArgument(
            "too few observations per fold for a stratified split".into(),
        ));
    }
    let mut rng = Mt19937::new(seed);
    let n_i = approximate_mode(&class_counts, n_train, &mut rng);
    let remaining: Vec<usize> = class_counts.iter().zip(&n_i).map(|(c, k)| c - k).collect();
    let t_i = approximate_mode(&remaining, n_test, &mut rng);
    let (mut train, mut test) = (Vec::with_capacity(n_train), Vec::with_capacity(n_test));
    for (c, &count) in class_counts.iter().enumerate() {
        let members: Vec<usize> = (0..n).filter(|&i| y_indices[i] == c).collect();
        let perm = rng.permutation(count);
        let permuted: Vec<usize> = perm.iter().map(|&k| members[k]).collect();
        train.extend_from_slice(&permuted[..n_i[c]]);
        test.extend_from_slice(&permuted[n_i[c]..n_i[c] + t_i[c]]);
    }
    rng.shuffle(&mut train);
    rng.shuffle(&mut test);
    Ok((train, test))
}

/// Test sets of scikit-learn's `StratifiedKFold(k)` without shuffling: classes
/// numbered by first appearance are dealt round-robin over the sorted labels,
/// then assigned to folds in data order.
pub(crate) fn stratified_kfold(labels: &[f64], k: usize) -> Result<Vec<Vec<usize>>> {
    let n = labels.len();
    // Classes by order of first appearance.
    let mut first: Vec<f64> = Vec::new();
    let encoded: Vec<usize> = labels
        .iter()
        .map(|&v| match first.iter().position(|&f| f == v) {
            Some(i) => i,
            None => {
                first.push(v);
                first.len() - 1
            }
        })
        .collect();
    let n_classes = first.len();
    let mut counts = vec![0usize; n_classes];
    for &c in &encoded {
        counts[c] += 1;
    }
    if counts.iter().all(|&c| c < k) {
        return Err(DmlError::InvalidArgument(format!(
            "{k} folds is more than the number of members in each class"
        )));
    }
    // Sorted encoded labels: class c occupies positions [start_c, start_c + count_c).
    let mut starts = vec![0usize; n_classes];
    for c in 1..n_classes {
        starts[c] = starts[c - 1] + counts[c - 1];
    }
    let dealt = |fold: usize, upto: usize| {
        if fold < upto {
            (upto - 1 - fold) / k + 1
        } else {
            0
        }
    };
    let mut fold_of = vec![0usize; n];
    for c in 0..n_classes {
        let (lo, hi) = (starts[c], starts[c] + counts[c]);
        let folds: Vec<usize> = (0..k)
            .flat_map(|f| std::iter::repeat(f).take(dealt(f, hi) - dealt(f, lo)))
            .collect();
        let members = (0..n).filter(|&i| encoded[i] == c);
        for (i, f) in members.zip(folds) {
            fold_of[i] = f;
        }
    }
    Ok((0..k)
        .map(|f| (0..n).filter(|&i| fold_of[i] == f).collect())
        .collect())
}

// ---------------------------------------------------------------------------
// SciPy optimisation
// ---------------------------------------------------------------------------

/// SciPy's `brentq` (root of `f` in `[a, b]`, with `f(a)`, `f(b)` of opposite
/// sign) with `root_scalar`'s defaults `xtol = 2e-12`, `rtol = 4·eps`,
/// `maxiter = 100`.
pub(crate) fn brentq(f: &mut dyn FnMut(f64) -> f64, a: f64, b: f64) -> Result<f64> {
    let (xtol, rtol, iter) = (2e-12, 4.0 * f64::EPSILON, 100);
    let (mut xpre, mut xcur) = (a, b);
    let (mut xblk, mut fblk, mut spre, mut scur) = (0.0, 0.0, 0.0, 0.0);
    let mut fpre = f(xpre);
    let mut fcur = f(xcur);
    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.is_sign_negative() == fcur.is_sign_negative() {
        return Err(DmlError::InvalidArgument(
            "the score does not change sign over the bracket".into(),
        ));
    }
    for _ in 0..iter {
        if fpre != 0.0 && fcur != 0.0 && fpre.is_sign_negative() != fcur.is_sign_negative() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }
        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // interpolate
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // extrapolate
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < spre.abs().min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }
    Ok(xcur)
}

/// SciPy's `root_scalar(method="newton")` with an analytic derivative:
/// Newton–Raphson from `x0` with `newton`'s defaults (`tol = 1.48e-8`
/// absolute on the step, at most 50 iterations). Returns the root and whether
/// it converged (SciPy returns the last iterate either way).
pub(crate) fn newton(
    f: &mut dyn FnMut(f64) -> f64,
    fprime: &mut dyn FnMut(f64) -> f64,
    x0: f64,
) -> (f64, bool) {
    const TOL: f64 = 1.48e-8;
    let mut p0 = x0;
    for _ in 0..50 {
        let fval = f(p0);
        if fval == 0.0 {
            return (p0, true);
        }
        let fder = fprime(p0);
        if fder == 0.0 {
            return (p0, false);
        }
        let p = p0 - fval / fder;
        // np.isclose(p, p0, rtol=0, atol=tol)
        if (p - p0).abs() <= TOL {
            return (p, true);
        }
        p0 = p;
    }
    (p0, false)
}

/// SciPy's `bracket`: from `(xa, xb)`, a triple with `f(xb)` below both ends,
/// or the last triple tried if none is valid (SciPy then raises `BracketError`).
fn bracket(f: &mut dyn FnMut(f64) -> f64, xa: f64, xb: f64) -> ([f64; 3], [f64; 3], bool) {
    const GOLD: f64 = 1.618034;
    const VERY_SMALL: f64 = 1e-21;
    let grow_limit = 110.0;
    let (mut xa, mut xb) = (xa, xb);
    let mut fa = f(xa);
    let mut fb = f(xb);
    if fa < fb {
        std::mem::swap(&mut xa, &mut xb);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut xc = xb + GOLD * (xb - xa);
    let mut fc = f(xc);
    let mut iter = 0;
    while fc < fb {
        let tmp1 = (xb - xa) * (fb - fc);
        let tmp2 = (xb - xc) * (fb - fa);
        let val = tmp2 - tmp1;
        let denom = if val.abs() < VERY_SMALL {
            2.0 * VERY_SMALL
        } else {
            2.0 * val
        };
        let mut w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
        let wlim = xb + grow_limit * (xc - xb);
        if iter > 1000 {
            break;
        }
        iter += 1;
        let mut fw;
        if (w - xc) * (xb - w) > 0.0 {
            fw = f(w);
            if fw < fc {
                xa = xb;
                xb = w;
                fa = fb;
                fb = fw;
                break;
            } else if fw > fb {
                xc = w;
                fc = fw;
                break;
            }
            w = xc + GOLD * (xc - xb);
            fw = f(w);
        } else if (w - wlim) * (wlim - xc) >= 0.0 {
            w = wlim;
            fw = f(w);
        } else if (w - wlim) * (xc - w) > 0.0 {
            fw = f(w);
            if fw < fc {
                xb = xc;
                xc = w;
                w = xc + GOLD * (xc - xb);
                fb = fc;
                fc = fw;
                fw = f(w);
            }
        } else {
            w = xc + GOLD * (xc - xb);
            fw = f(w);
        }
        xa = xb;
        xb = xc;
        xc = w;
        fa = fb;
        fb = fc;
        fc = fw;
    }
    let cond1 = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
    let cond2 = (xa < xb && xb < xc) || (xc < xb && xb < xa);
    let cond3 = xa.is_finite() && xb.is_finite() && xc.is_finite();
    ([xa, xb, xc], [fa, fb, fc], cond1 && cond2 && cond3)
}

/// `minimize_scalar(f, bracket=(a, b), method="brent").x` (SciPy defaults
/// `xtol = 1.48e-8`, `maxiter = 500`), including SciPy's recovery when no valid
/// bracket is found: the bracket point with the smallest value.
pub(crate) fn brent_minimize(f: &mut dyn FnMut(f64) -> f64, a: f64, b: f64) -> f64 {
    let (xs, fs, valid) = bracket(f, a, b);
    if !valid {
        if xs.iter().chain(&fs).any(|v| v.is_nan()) {
            return f64::NAN;
        }
        let imin = (0..3).fold(0, |m, i| if fs[i] < fs[m] { i } else { m });
        return xs[imin];
    }
    let (xa, xb, xc, fb) = (xs[0], xs[1], xs[2], fs[1]);
    let (tol, mintol, cg, maxiter) = (1.48e-8, 1.0e-11, 0.3819660, 500);
    let (mut x, mut w, mut v) = (xb, xb, xb);
    let (mut fw, mut fv, mut fx) = (fb, fb, fb);
    let (mut a, mut b) = if xa < xc { (xa, xc) } else { (xc, xa) };
    let mut deltax: f64 = 0.0;
    let mut rat: f64 = 0.0;
    for _ in 0..maxiter {
        let tol1 = tol * x.abs() + mintol;
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        if (x - xmid).abs() < (tol2 - 0.5 * (b - a)) {
            break;
        }
        if deltax.abs() <= tol1 {
            deltax = if x >= xmid { a - x } else { b - x };
            rat = cg * deltax;
        } else {
            let tmp1 = (x - w) * (fx - fv);
            let mut tmp2 = (x - v) * (fx - fw);
            let mut p = (x - v) * tmp2 - (x - w) * tmp1;
            tmp2 = 2.0 * (tmp2 - tmp1);
            if tmp2 > 0.0 {
                p = -p;
            }
            tmp2 = tmp2.abs();
            let dx_temp = deltax;
            deltax = rat;
            if p > tmp2 * (a - x) && p < tmp2 * (b - x) && p.abs() < (0.5 * tmp2 * dx_temp).abs() {
                rat = p * 1.0 / tmp2;
                let u = x + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                }
            } else {
                deltax = if x >= xmid { a - x } else { b - x };
                rat = cg * deltax;
            }
        }
        let u = if rat.abs() < tol1 {
            if rat >= 0.0 {
                x + tol1
            } else {
                x - tol1
            }
        } else {
            x + rat
        };
        let fu = f(u);
        if fu > fx {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        } else {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
    }
    x
}

/// DoubleML's `_get_bracket_guess`: widen a bracket around `start` (within
/// `bounds`) until the score changes sign; returns the bracket and whether
/// the signs differ.
pub(crate) fn bracket_guess(
    score: &mut dyn FnMut(f64) -> f64,
    start: f64,
    bounds: (f64, f64),
) -> ((f64, f64), bool) {
    let max_len = bounds.1 - bounds.0;
    let mut guess = bounds;
    let mut delta = 0.1;
    let mut different = false;
    while !different && delta <= 1.0 {
        let a = (start - delta * max_len / 2.0).max(bounds.0);
        let b = (start + delta * max_len / 2.0).min(bounds.1);
        guess = (a, b);
        let (fa, fb) = (score(a), score(b));
        different = sign(fa) != sign(fb);
        delta += 0.1;
    }
    (guess, different)
}

/// `np.sign` (0 for 0).
fn sign(v: f64) -> f64 {
    if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Quantiles and kernel density
// ---------------------------------------------------------------------------

/// `np.sum` of a contiguous float64 array: NumPy's pairwise summation (blocks
/// of 128 summed with eight accumulators, larger arrays split in halves).
pub(crate) fn np_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        np_sum(&a[..n2]) + np_sum(&a[n2..])
    }
}

/// `np.mean`.
pub(crate) fn np_mean(a: &[f64]) -> f64 {
    np_sum(a) / a.len() as f64
}

/// `np.quantile(values, q)` with the default linear method.
pub(crate) fn np_quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let virtual_index = (n - 1) as f64 * q;
    let prev = (virtual_index.floor() as usize).min(n - 1);
    let next = (prev + 1).min(n - 1);
    let gamma = virtual_index - prev as f64;
    let (a, b) = (sorted[prev], sorted[next]);
    let diff = b - a;
    // NumPy's `_lerp` switches to the upper end for t >= 0.5.
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// `scipy.stats.scoreatpercentile(values, per)` (fractional interpolation).
fn score_at_percentile(sorted: &[f64], per: f64) -> f64 {
    let idx = per / 100.0 * (sorted.len() - 1) as f64;
    let i = idx as usize;
    if i as f64 == idx {
        return sorted[i];
    }
    let j = i + 1;
    let (w0, w1) = (j as f64 - idx, idx - i as f64);
    (sorted[i] * w0 + sorted[j] * w1) / (w0 + w1)
}

/// DoubleML's default KDE: statsmodels' `KDEUnivariate(u).fit(kernel="gau",
/// bw="silverman", weights=w, fft=False).evaluate(0)`. The bandwidth comes from
/// the unweighted `u`; the weights are normalised to sum to one.
pub(crate) fn kde_at_zero(u: &[f64], weights: &[f64]) -> f64 {
    let n = u.len() as f64;
    let mut sorted = u.to_vec();
    sorted.sort_by(f64::total_cmp);
    let iqr = (score_at_percentile(&sorted, 75.0) - score_at_percentile(&sorted, 25.0)) / 1.349;
    let mean = np_sum(u) / n;
    let squares: Vec<f64> = u.iter().map(|v| (v - mean) * (v - mean)).collect();
    let std = (np_sum(&squares) / (n - 1.0)).sqrt();
    let a = if iqr > 0.0 { std.min(iqr) } else { std };
    let h = 0.9 * a * n.powf(-0.2);
    let total = np_sum(weights);
    let terms: Vec<f64> = u
        .iter()
        .zip(weights)
        .map(|(&ui, &wi)| {
            let z = ui / h;
            0.398_942_280_401_432_7 * (-(z * z) / 2.0).exp() * (wi / total)
        })
        .collect();
    1.0 / h * np_sum(&terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values computed with NumPy 2.5, scikit-learn 1.9, SciPy 1.18
    // and statsmodels (see parity/ for the generating code).

    #[test]
    fn permutation_matches_numpy_random_state() {
        let mut rng = Mt19937::new(42);
        assert_eq!(rng.permutation(10), vec![8, 1, 5, 0, 7, 2, 9, 4, 3, 6]);
        assert_eq!(rng.permutation(7), vec![4, 1, 0, 3, 5, 6, 2]);
    }

    #[test]
    fn train_test_split_matches_scikit_learn() {
        let lab = [
            1., 0., 0., 1., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0.,
            1.,
        ];
        let (tr, te) = stratified_train_test_split(&lab, 42).unwrap();
        let shift = |v: Vec<usize>| v.into_iter().map(|i| i + 20).collect::<Vec<_>>();
        assert_eq!(shift(tr), vec![39, 31, 23, 36, 34, 20, 29, 21, 40, 37, 22]);
        assert_eq!(
            shift(te),
            vec![41, 42, 35, 33, 25, 27, 26, 28, 30, 38, 24, 32]
        );
        let lab4 = [
            0., 1., 2., 3., 0., 1., 2., 3., 3., 2., 1., 0., 0., 0., 1., 1., 2., 2., 3., 3., 0., 1.,
            2.,
        ];
        let (tr4, te4) = stratified_train_test_split(&lab4, 42).unwrap();
        assert_eq!(shift(tr4), vec![22, 20, 29, 26, 23, 34, 28, 41, 24, 21, 40]);
        assert_eq!(
            shift(te4),
            vec![31, 25, 30, 37, 32, 36, 42, 35, 33, 38, 39, 27]
        );
        assert_eq!(
            stratified_kfold(&lab4, 3).unwrap(),
            vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![8, 9, 10, 11, 12, 14, 16, 18],
                vec![13, 15, 17, 19, 20, 21, 22]
            ]
        );
    }

    #[test]
    fn solvers_match_scipy() {
        let y = [0.3, -1.2, 2.5, 0.7, 1.1, -0.4, 0.9, 3.1, -2.2, 0.05];
        // np.mean((y <= t) * 1.7 - 0.6), with NumPy's pairwise summation.
        let mut f = |t: f64| {
            let terms: Vec<f64> = y.iter().map(|&v| f64::from(v <= t) * 1.7 - 0.6).collect();
            np_mean(&terms)
        };
        assert_eq!(brentq(&mut f, -2.0, 2.9).unwrap(), 0.050_000_000_001_808_75);
        let mut g = |t: f64| f(t).abs();
        assert_eq!(brent_minimize(&mut g, -1.0, 2.0), 0.064_570_603_034_054_2);
    }

    #[test]
    fn kde_and_quantiles_match_statsmodels_and_numpy() {
        let u = [0.4, -1.0, 2.2, 0.1, -0.3, 1.5, 0.0, -2.1];
        let w = [1.0, 2.0, 0.0, 0.5, 1.5, 0.0, 3.0, 1.0];
        assert!((kde_at_zero(&u, &w) - 0.504_832_554_253_779_5).abs() < 1e-15);
        let y = [0.3, -1.2, 2.5, 0.7, 1.1, -0.4, 0.9, 3.1, -2.2, 0.05];
        assert!((np_quantile(&y, 0.37) - 0.1325).abs() < 1e-15);
        assert_eq!(np_quantile(&y, 0.5), 0.5);
    }
}

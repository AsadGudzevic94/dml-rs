//! Partial identification under sample selection: Lee (2009), "Training,
//! wages, and sample selection: estimating sharp bounds on treatment effects",
//! Review of Economic Studies 76(3), with the covariate-tightened bounds of its
//! Section 4 and the confidence interval of Imbens and Manski (2004).

use rand::{Rng, SeedableRng};

use crate::error::{DmlError, Result};
use crate::stats::{norm_cdf, norm_ppf};

/// Sharp bounds on the effect of a randomised binary treatment `d` on the
/// units whose outcome is observed under both treatments (the "always
/// observed"), when the treatment also affects whether the outcome is
/// observed (`s = 1`: e.g. employment for wages, response for survey
/// outcomes, survival for test scores).
///
/// Under monotonicity (the treatment can only increase, or only decrease,
/// selection) the group with the higher selection rate contains a share
/// `q = |p₁ − p₀| / max(p₁, p₀)` of units who are observed only because of
/// their treatment. Trimming that share from the top or the bottom of its
/// outcome distribution gives the lower and the upper bound (Lee 2009).
/// Trimming is fractional (the boundary observation gets a partial weight),
/// so ties and small groups are handled exactly.
///
/// With discrete covariate cells ([`LeeBounds::fit_with_strata`]) the bounds
/// are computed per cell and averaged with the cells' shares of always-observed
/// units, which tightens them (Lee 2009, Section 4). The direction of
/// monotonicity is that of the whole sample; cells whose estimated selection
/// rates point the other way are not trimmed.
///
/// Standard errors come from the nonparametric bootstrap (units resampled
/// within treatment arms), and the confidence interval for the effect is
/// Imbens and Manski's (2004), which covers the true effect (not the whole
/// identified set) with the requested probability.
///
/// ```
/// use dml_core::LeeBounds;
///
/// // Treated units are observed more often (all of them vs. two thirds).
/// let y = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
/// let d = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let s = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
/// let fit = LeeBounds::new().n_boot(0).fit(&y, &d, &s)?;
/// assert!(fit.lower() <= fit.upper());
/// # Ok::<(), dml_core::DmlError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LeeBounds {
    n_boot: usize,
    level: f64,
    seed: u64,
}

impl Default for LeeBounds {
    fn default() -> Self {
        Self::new()
    }
}

impl LeeBounds {
    pub fn new() -> Self {
        Self {
            n_boot: 500,
            level: 0.95,
            seed: 0,
        }
    }

    /// Bootstrap replications for the standard errors (default 500; 0 skips
    /// inference).
    pub fn n_boot(mut self, n_boot: usize) -> Self {
        self.n_boot = n_boot;
        self
    }

    /// Confidence level of the Imbens–Manski interval (default 0.95).
    pub fn level(mut self, level: f64) -> Self {
        self.level = level;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Bounds without covariates. `y` may hold anything (e.g. NaN) where `s = 0`.
    pub fn fit(&self, y: &[f64], d: &[f64], s: &[f64]) -> Result<LeeBoundsFit> {
        self.fit_with_strata(y, d, s, &vec![0; y.len()])
    }

    /// Bounds tightened with discrete covariate cells `strata`.
    pub fn fit_with_strata(
        &self,
        y: &[f64],
        d: &[f64],
        s: &[f64],
        strata: &[i64],
    ) -> Result<LeeBoundsFit> {
        let n = y.len();
        if d.len() != n || s.len() != n || strata.len() != n {
            return Err(DmlError::DimensionMismatch(
                "y, d, s and strata need the same length".into(),
            ));
        }
        if d.iter().chain(s).any(|&v| v != 0.0 && v != 1.0) {
            return Err(DmlError::InvalidArgument(
                "d and s must contain only 0 and 1".into(),
            ));
        }
        if (0..n).any(|i| s[i] == 1.0 && !y[i].is_finite()) {
            return Err(DmlError::InvalidArgument(
                "observed outcomes (s = 1) must be finite".into(),
            ));
        }
        if !(self.level > 0.0 && self.level < 1.0) {
            return Err(DmlError::InvalidArgument("level must be in (0, 1)".into()));
        }
        let mut cells: Vec<i64> = strata.to_vec();
        cells.sort_unstable();
        cells.dedup();
        let cell_of: Vec<usize> = strata
            .iter()
            .map(|v| cells.binary_search(v).expect("cell"))
            .collect();
        let units: Vec<usize> = (0..n).collect();
        let est = estimate(y, d, s, &cell_of, cells.len(), &units)?;

        let (mut se_lower, mut se_upper) = (f64::NAN, f64::NAN);
        if self.n_boot > 0 {
            if self.n_boot < 2 {
                return Err(DmlError::InvalidArgument(
                    "need at least 2 bootstrap replications".into(),
                ));
            }
            let arms: [Vec<usize>; 2] = [
                (0..n).filter(|&i| d[i] == 0.0).collect(),
                (0..n).filter(|&i| d[i] == 1.0).collect(),
            ];
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(self.seed);
            let (mut lows, mut ups) = (Vec::new(), Vec::new());
            let mut attempts = 0;
            while lows.len() < self.n_boot {
                attempts += 1;
                if attempts > 20 * self.n_boot {
                    return Err(DmlError::InvalidArgument(
                        "too many bootstrap samples without observed outcomes in a cell".into(),
                    ));
                }
                let draw: Vec<usize> = arms
                    .iter()
                    .flat_map(|arm| {
                        (0..arm.len())
                            .map(|_| arm[rng.gen_range(0..arm.len())])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                if let Ok(b) = estimate(y, d, s, &cell_of, cells.len(), &draw) {
                    lows.push(b.lower);
                    ups.push(b.upper);
                }
            }
            se_lower = sample_sd(&lows);
            se_upper = sample_sd(&ups);
        }
        let ci = if se_lower.is_finite() && se_upper.is_finite() {
            imbens_manski(est.lower, est.upper, se_lower, se_upper, self.level)
        } else {
            [f64::NAN, f64::NAN]
        };
        Ok(LeeBoundsFit {
            lower: est.lower,
            upper: est.upper,
            trimming: est.trimming,
            trimmed_treated: est.trimmed_treated,
            se_lower,
            se_upper,
            ci,
            level: self.level,
            n_cells: cells.len(),
        })
    }
}

struct Estimate {
    lower: f64,
    upper: f64,
    trimming: f64,
    trimmed_treated: bool,
}

/// Bounds on the units `idx` (with repetitions for the bootstrap).
fn estimate(
    y: &[f64],
    d: &[f64],
    s: &[f64],
    cell_of: &[usize],
    n_cells: usize,
    idx: &[usize],
) -> Result<Estimate> {
    // Counts per cell: [arm][cell] units and observed units.
    let mut units = vec![vec![0.0; n_cells]; 2];
    let mut observed = vec![vec![0.0; n_cells]; 2];
    let mut outcomes: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n_cells]; 2];
    for &i in idx {
        let (a, c) = (d[i] as usize, cell_of[i]);
        units[a][c] += 1.0;
        if s[i] == 1.0 {
            observed[a][c] += 1.0;
            outcomes[a][c].push(y[i]);
        }
    }
    let total = |v: &[f64]| v.iter().sum::<f64>();
    let (p1, p0) = (
        total(&observed[1]) / total(&units[1]),
        total(&observed[0]) / total(&units[0]),
    );
    if !(p1 > 0.0 && p0 > 0.0) {
        return Err(DmlError::InvalidArgument(
            "both arms need observed outcomes".into(),
        ));
    }
    let trimmed_treated = p1 >= p0;
    let (big, small) = if trimmed_treated { (1, 0) } else { (0, 1) };
    let (mut lower, mut upper, mut weight_sum) = (0.0, 0.0, 0.0);
    for c in 0..n_cells {
        if units[0][c] == 0.0 || units[1][c] == 0.0 {
            continue;
        }
        let (pb, ps) = (
            observed[big][c] / units[big][c],
            observed[small][c] / units[small][c],
        );
        // Cell share of always-observed units: units observed in the low-selection arm.
        let w = ps * (units[0][c] + units[1][c]);
        if w == 0.0 {
            continue;
        }
        if outcomes[big][c].is_empty() {
            return Err(DmlError::InvalidArgument(
                "a cell has always-observed units but no observed outcomes in the trimmed arm"
                    .into(),
            ));
        }
        let q = if pb > ps { (pb - ps) / pb } else { 0.0 };
        let mut sorted = outcomes[big][c].clone();
        sorted.sort_by(f64::total_cmp);
        let keep = 1.0 - q;
        let (bottom, top) = (
            trimmed_mean(&sorted, keep, false),
            trimmed_mean(&sorted, keep, true),
        );
        let other = outcomes[small][c].iter().sum::<f64>() / outcomes[small][c].len() as f64;
        let (lo, hi) = if trimmed_treated {
            (bottom - other, top - other)
        } else {
            (other - top, other - bottom)
        };
        lower += w * lo;
        upper += w * hi;
        weight_sum += w;
    }
    if weight_sum == 0.0 {
        return Err(DmlError::InvalidArgument(
            "no cell has observed units in both arms".into(),
        ));
    }
    Ok(Estimate {
        lower: lower / weight_sum,
        upper: upper / weight_sum,
        trimming: if trimmed_treated {
            (p1 - p0) / p1
        } else {
            (p0 - p1) / p0
        },
        trimmed_treated,
    })
}

/// Mean of the lowest (or highest) share `keep` of sorted values, the boundary
/// value entering with a fractional weight.
fn trimmed_mean(sorted: &[f64], keep: f64, from_top: bool) -> f64 {
    let m = sorted.len() as f64;
    let mass = keep * m;
    let mut remaining = mass;
    let mut sum = 0.0;
    let iter: Box<dyn Iterator<Item = &f64>> = if from_top {
        Box::new(sorted.iter().rev())
    } else {
        Box::new(sorted.iter())
    };
    for &v in iter {
        if remaining <= 0.0 {
            break;
        }
        let w = remaining.min(1.0);
        sum += w * v;
        remaining -= w;
    }
    sum / mass
}

fn sample_sd(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Imbens and Manski (2004): `[L − c σ_L, U + c σ_U]` with `c` solving
/// `Φ(c + Δ / max(σ_L, σ_U)) − Φ(−c) = level`, `Δ = U − L`.
fn imbens_manski(lower: f64, upper: f64, se_lower: f64, se_upper: f64, level: f64) -> [f64; 2] {
    let sigma = se_lower.max(se_upper);
    let delta = (upper - lower).max(0.0);
    let c = if sigma == 0.0 {
        0.0
    } else {
        let f = |c: f64| norm_cdf(c + delta / sigma) - norm_cdf(-c) - level;
        let (mut lo, mut hi) = (0.0, norm_ppf(0.5 + level / 2.0));
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    [lower - c * se_lower, upper + c * se_upper]
}

/// Result of [`LeeBounds::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LeeBoundsFit {
    lower: f64,
    upper: f64,
    trimming: f64,
    trimmed_treated: bool,
    se_lower: f64,
    se_upper: f64,
    ci: [f64; 2],
    level: f64,
    n_cells: usize,
}

impl LeeBoundsFit {
    /// Lower bound on the effect for the always-observed units.
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Upper bound.
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Overall trimming share `|p₁ − p₀| / max(p₁, p₀)`.
    pub fn trimming(&self) -> f64 {
        self.trimming
    }

    /// Whether the treated arm (higher selection) was trimmed.
    pub fn trimmed_treated(&self) -> bool {
        self.trimmed_treated
    }

    /// Bootstrap standard errors of the bounds (NaN without bootstrap).
    pub fn se(&self) -> [f64; 2] {
        [self.se_lower, self.se_upper]
    }

    /// Imbens–Manski confidence interval for the effect.
    pub fn confint(&self) -> [f64; 2] {
        self.ci
    }

    pub fn level(&self) -> f64 {
        self.level
    }

    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::StandardNormal;

    #[test]
    fn trimmed_means() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((trimmed_mean(&v, 0.5, false) - 1.5).abs() < 1e-15);
        assert!((trimmed_mean(&v, 0.5, true) - 3.5).abs() < 1e-15);
        // Keep 62.5% = 2.5 values: 1 + 2 + 0.5·3 over 2.5.
        assert!((trimmed_mean(&v, 0.625, false) - 4.5 / 2.5).abs() < 1e-15);
        assert!((trimmed_mean(&v, 1.0, true) - 2.5).abs() < 1e-15);
    }

    /// Monotone selection: treatment raises the outcome by 1 and selection
    /// from 60% to 80%; units observed only under treatment have low
    /// outcomes. The bounds contain the effect 1 and the Imbens–Manski
    /// interval covers it.
    #[test]
    fn bounds_contain_the_effect() {
        let (mut contain, mut cover) = (0, 0);
        let reps = 40;
        for rep in 0..reps {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(rep);
            let n = 2000;
            let (mut y, mut d, mut s) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
            for i in 0..n {
                d[i] = f64::from(u8::from(rng.gen::<bool>()));
                let u: f64 = rng.gen();
                let y0 = 2.0 * u + rng.sample::<f64, _>(StandardNormal) * 0.3;
                // Always observed if u > 0.4; observed only when treated if 0.2 < u <= 0.4.
                s[i] = f64::from(u8::from(u > 0.4 || (d[i] == 1.0 && u > 0.2)));
                y[i] = y0 + d[i];
            }
            let fit = LeeBounds::new()
                .n_boot(100)
                .seed(rep)
                .fit(&y, &d, &s)
                .unwrap();
            assert!(fit.trimmed_treated());
            contain += usize::from(fit.lower() <= 1.0 && 1.0 <= fit.upper());
            let ci = fit.confint();
            cover += usize::from(ci[0] <= 1.0 && 1.0 <= ci[1]);
        }
        assert!(contain as f64 / reps as f64 > 0.85, "contain {contain}");
        assert!(cover as f64 / reps as f64 > 0.9, "cover {cover}");
    }

    #[test]
    fn rejects_bad_input() {
        let lb = LeeBounds::new().n_boot(0);
        assert!(lb.fit(&[1.0, 2.0], &[0.0, 1.0], &[1.0]).is_err());
        assert!(lb.fit(&[1.0, 2.0], &[0.0, 2.0], &[1.0, 1.0]).is_err());
        assert!(lb.fit(&[f64::NAN, 2.0], &[0.0, 1.0], &[1.0, 1.0]).is_err());
        assert!(lb.fit(&[1.0, 2.0], &[0.0, 1.0], &[0.0, 1.0]).is_err());
    }
}

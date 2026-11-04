//! Anytime-valid inference: confidence sequences for any [`DmlFit`].
//!
//! A confidence interval at level `1 − α` covers θ with probability `1 − α` at
//! *one* sample size fixed in advance. An experiment that is checked as data
//! come in and stopped once the interval excludes zero looks many times, and the
//! chance that some look misses θ is much larger than α. A *confidence sequence*
//! covers θ at all sample sizes at once, `P(θ ∈ C_t for all t) ≥ 1 − α`, so it
//! can be checked after every new observation and the experiment stopped at any
//! time (Howard, Ramdas, McAuliffe & Sekhon, 2021).
//!
//! This module uses the asymptotic confidence sequences of Waudby-Smith, Arbour,
//! Sinha, Kennedy & Ramdas (2024, *Time-uniform central limit theory and
//! asymptotic confidence sequences*, Annals of Statistics), which play the role
//! for confidence sequences that the central limit theorem plays for confidence
//! intervals. For an estimator with standard error `se_t` after `t`
//! observations the sequence is
//!
//! ```text
//! θ̂_t ± se_t · √( 2(u + 1)/u · log(√(u + 1) / α) ),   u = t ρ²,
//! ```
//!
//! the Robbins normal-mixture boundary. `ρ` is a tuning constant fixed in
//! advance; it makes the sequence tightest around the sample size `t*`
//! (`opt_time`) via `ρ² = (−2 log α + log(−2 log α + 1)) / t*`. At `t*` and
//! α = 0.05 the half-width is about 3.0 standard errors (1.96 for a confidence
//! interval at one fixed sample size); it grows like `√(log t)` away from `t*`.
//! Their Theorem 3 covers doubly robust (AIPW) estimators with cross-fitted
//! nuisances, i.e. the DML scores here.
//!
//! [`DmlFit::confidence_sequence`] gives the sequence at the fit's sample size.
//! To monitor a study, refit at each look on all data so far and feed the fits
//! to a [`ConfidenceSequence`], which keeps the running intersection of the
//! intervals and the anytime-valid p-values.

use ndarray::{Array1, Array2, Array3};

use crate::error::{DmlError, Result};
use crate::fit::{check_level, confint_from, DmlFit};

/// Half-width of the asymptotic confidence sequence in standard errors after
/// `t` observations, at confidence `level`, for the boundary tuned to be
/// tightest at `opt_time` observations.
///
/// ```
/// # use dml_core::anytime::cs_multiplier;
/// // At the tuning time and 95 %: about 3.04 standard errors.
/// let m = cs_multiplier(1000, 0.95, 1000).unwrap();
/// assert!((m - 3.035).abs() < 1e-3);
/// ```
pub fn cs_multiplier(t: usize, level: f64, opt_time: usize) -> Result<f64> {
    check_inputs(t, level, opt_time)?;
    let u = t as f64 * rho2(level, opt_time);
    Ok((2.0 * (u + 1.0) / u * ((u + 1.0).sqrt() / (1.0 - level)).ln()).sqrt())
}

/// `ρ²` of the normal mixture tuned for `opt_time` observations.
fn rho2(level: f64, opt_time: usize) -> f64 {
    let c = -2.0 * (1.0 - level).ln();
    (c + (c + 1.0).ln()) / opt_time as f64
}

fn check_inputs(t: usize, level: f64, opt_time: usize) -> Result<()> {
    check_level(level)?;
    if t == 0 || opt_time == 0 {
        return Err(DmlError::InvalidArgument(
            "the sample size and opt_time must be positive".into(),
        ));
    }
    Ok(())
}

/// Anytime-valid p-value at one look: the inverse of the normal-mixture
/// martingale, `min(1, √(1 + u) · exp(−u z² / (2(1 + u))))` with `z` the
/// t-statistic for `H0: θ = 0`. `θ = 0` lies outside the confidence sequence
/// exactly when this is below `1 − level`.
fn anytime_p(z: f64, u: f64) -> f64 {
    ((1.0 + u).sqrt() * (-u * z * z / (2.0 * (1.0 + u))).exp()).min(1.0)
}

impl DmlFit {
    /// Observations that count as the sample size of the sequence: units, or
    /// clusters for a clustered fit.
    fn sequence_time(&self) -> usize {
        self.n_clusters().unwrap_or(self.n_obs)
    }

    /// Asymptotic confidence sequence at the fit's sample size, shape
    /// `(n_treat, 2)`: an interval that, recomputed at every sample size, covers
    /// θ at all of them at once with probability `level` (see
    /// [`anytime`](crate::anytime)). `opt_time` is the sample size where the
    /// sequence is tightest; it must be fixed before looking at the data.
    /// Clustered fits count clusters. Repetitions are aggregated as in
    /// [`confint`](Self::confint).
    pub fn confidence_sequence(&self, level: f64, opt_time: usize) -> Result<Array2<f64>> {
        let m = cs_multiplier(self.sequence_time(), level, opt_time)?;
        Ok(confint_from(
            &self.all_coef,
            &self.all_se,
            &Array1::from_elem(self.n_rep(), m),
        ))
    }

    /// Anytime-valid p-values for `H0: θ = 0` at this look, for the
    /// confidence sequence of [`confidence_sequence`](Self::confidence_sequence)
    /// with the same `level` and `opt_time`. Taking the minimum over all looks
    /// so far keeps them valid ([`ConfidenceSequence::p_values`] does).
    pub fn anytime_p_value(&self, level: f64, opt_time: usize) -> Result<Array1<f64>> {
        let t = self.sequence_time();
        check_inputs(t, level, opt_time)?;
        let u = t as f64 * rho2(level, opt_time);
        Ok(self.t_stat().mapv(|z| anytime_p(z, u)))
    }
}

/// Confidence sequences over the looks of a study: refit the model on all
/// data so far at each look and pass the fits to [`update`](Self::update).
///
/// The intervals of all looks hold simultaneously, so their running
/// intersection does too ([`running_intervals`](Self::running_intervals)),
/// and the running minimum of the anytime p-values is a valid p-value at
/// any stopping time ([`p_values`](Self::p_values)).
///
/// ```
/// # use dml_core::{anytime::ConfidenceSequence, DmlFit};
/// # use ndarray::Array3;
/// let mut cs = ConfidenceSequence::new(0.95, 100).unwrap();
/// for t in [50, 100, 200] {
///     let psi_a = Array3::from_elem((t, 1, 1), -1.0);
///     let psi_b = Array3::from_shape_fn((t, 1, 1), |(i, _, _)| 0.3 + (i as f64 * 2.3).sin());
///     cs.update(&DmlFit::from_linear_scores("mean", vec!["y".into()], 2, psi_a, psi_b).unwrap())
///         .unwrap();
/// }
/// let running = cs.running_intervals(); // shape (looks, n_treat, 2)
/// assert!(running[[2, 0, 0]] >= cs.intervals()[[2, 0, 0]]);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConfidenceSequence {
    level: f64,
    opt_time: usize,
    names: Vec<String>,
    times: Vec<usize>,
    coef: Vec<Array1<f64>>,
    se: Vec<Array1<f64>>,
    intervals: Vec<Array2<f64>>,
    p_values: Vec<Array1<f64>>,
}

impl ConfidenceSequence {
    /// Sequences at confidence `level`, tightest around `opt_time`
    /// observations (clusters for clustered fits).
    pub fn new(level: f64, opt_time: usize) -> Result<Self> {
        check_inputs(1, level, opt_time)?;
        Ok(Self {
            level,
            opt_time,
            names: Vec::new(),
            times: Vec::new(),
            coef: Vec::new(),
            se: Vec::new(),
            intervals: Vec::new(),
            p_values: Vec::new(),
        })
    }

    /// Add a look: a fit on all data so far. Looks must come with growing
    /// sample sizes and the same estimates.
    pub fn update(&mut self, fit: &DmlFit) -> Result<&mut Self> {
        let t = fit.sequence_time();
        if let Some(&last) = self.times.last() {
            if t <= last {
                return Err(DmlError::InvalidArgument(format!(
                    "looks need growing sample sizes: {t} after {last}"
                )));
            }
            if fit.treatment_names() != self.names.as_slice() {
                return Err(DmlError::InvalidArgument(
                    "every look needs the same estimates".into(),
                ));
            }
        } else {
            self.names = fit.treatment_names().to_vec();
        }
        let intervals = fit.confidence_sequence(self.level, self.opt_time)?;
        let p = fit.anytime_p_value(self.level, self.opt_time)?;
        self.times.push(t);
        self.coef.push(fit.coef().to_owned());
        self.se.push(fit.se().to_owned());
        self.intervals.push(intervals);
        self.p_values.push(p);
        Ok(self)
    }

    pub fn level(&self) -> f64 {
        self.level
    }

    pub fn opt_time(&self) -> usize {
        self.opt_time
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Sample size at each look.
    pub fn times(&self) -> &[usize] {
        &self.times
    }

    /// Estimates at each look, shape `(looks, n_treat)`.
    pub fn coef(&self) -> Array2<f64> {
        stack(&self.coef, self.names.len())
    }

    /// Standard errors at each look, shape `(looks, n_treat)`.
    pub fn se(&self) -> Array2<f64> {
        stack(&self.se, self.names.len())
    }

    /// The confidence sequence at each look, shape `(looks, n_treat, 2)`.
    pub fn intervals(&self) -> Array3<f64> {
        let k = self.names.len();
        Array3::from_shape_fn((self.times.len(), k, 2), |(l, j, s)| {
            self.intervals[l][[j, s]]
        })
    }

    /// Running intersection of the intervals up to each look, shape
    /// `(looks, n_treat, 2)`. If it is ever empty (lower above upper), the
    /// assumptions behind the sequence are in doubt.
    pub fn running_intervals(&self) -> Array3<f64> {
        let mut out = self.intervals();
        for l in 1..out.dim().0 {
            for j in 0..out.dim().1 {
                out[[l, j, 0]] = out[[l, j, 0]].max(out[[l - 1, j, 0]]);
                out[[l, j, 1]] = out[[l, j, 1]].min(out[[l - 1, j, 1]]);
            }
        }
        out
    }

    /// Anytime-valid p-values for `H0: θ = 0`, the running minimum over looks,
    /// shape `(looks, n_treat)`.
    pub fn p_values(&self) -> Array2<f64> {
        let mut p = stack(&self.p_values, self.names.len());
        for l in 1..p.nrows() {
            for j in 0..p.ncols() {
                p[[l, j]] = p[[l, j]].min(p[[l - 1, j]]);
            }
        }
        p
    }

    /// For each estimate, the first look (index into [`times`](Self::times))
    /// whose running interval excludes `value`, i.e. when a study testing
    /// `θ = value` could stop; `None` if no look does yet.
    pub fn first_exclusion(&self, value: f64) -> Vec<Option<usize>> {
        let running = self.running_intervals();
        (0..self.names.len())
            .map(|j| {
                (0..self.times.len())
                    .find(|&l| running[[l, j, 0]] > value || running[[l, j, 1]] < value)
            })
            .collect()
    }
}

fn stack(rows: &[Array1<f64>], k: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows.len(), k), |(l, j)| rows[l][j])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn mean_fit(values: &[f64]) -> DmlFit {
        let n = values.len();
        let psi_a = Array3::from_elem((n, 1, 1), -1.0);
        let psi_b = Array3::from_shape_fn((n, 1, 1), |(i, _, _)| values[i]);
        DmlFit::from_linear_scores("mean", vec!["y".into()], 2, psi_a, psi_b).unwrap()
    }

    #[test]
    fn multiplier_matches_the_closed_form_and_is_smallest_near_opt_time() {
        let alpha: f64 = 0.05;
        let rho2 = (-2.0 * alpha.ln() + (-2.0 * alpha.ln() + 1.0).ln()) / 500.0;
        for t in [10, 500, 10_000] {
            let u = t as f64 * rho2;
            let direct = (2.0 * (u + 1.0) / u * ((u + 1.0).sqrt() / alpha).ln()).sqrt();
            assert!((cs_multiplier(t, 0.95, 500).unwrap() - direct).abs() < 1e-14);
        }
        let at = |opt| cs_multiplier(500, 0.95, opt).unwrap();
        assert!(at(500) < at(250) && at(500) < at(1000));
        // Always wider than the fixed-sample interval.
        assert!(at(500) > 1.96);
    }

    #[test]
    fn p_value_crosses_one_minus_level_at_the_boundary() {
        let fit = mean_fit(
            &(0..400)
                .map(|i| 0.2 + (i as f64 * 1.3).sin())
                .collect::<Vec<_>>(),
        );
        let cs = fit.confidence_sequence(0.9, 400).unwrap();
        let p = fit.anytime_p_value(0.9, 400).unwrap()[0];
        let excludes_zero = cs[[0, 0]] > 0.0 || cs[[0, 1]] < 0.0;
        assert_eq!(excludes_zero, p < 0.1);
        // At the boundary itself the p-value is exactly 1 − level.
        let m = cs_multiplier(400, 0.9, 400).unwrap();
        let u = 400.0 * super::rho2(0.9, 400);
        assert!((anytime_p(m, u) - 0.1).abs() < 1e-12);
    }

    /// Time-uniform coverage: over 500 streams of 2000 normal draws, looked at
    /// every 20 observations, the sequence misses the mean at *some* look far
    /// less often than repeated 95 % confidence intervals do.
    #[test]
    fn covers_the_mean_at_all_looks_at_once() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (reps, mut cs_miss, mut ci_miss) = (500, 0, 0);
        for _ in 0..reps {
            let x: Vec<f64> = (0..2000)
                .map(|_| {
                    1.0 + <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                })
                .collect();
            let (mut sum, mut sum2) = (0.0, 0.0);
            let (mut cs_out, mut ci_out) = (false, false);
            for (i, v) in x.iter().enumerate() {
                sum += v;
                sum2 += v * v;
                let t = i + 1;
                if t < 40 || t % 20 != 0 {
                    continue;
                }
                let mean = sum / t as f64;
                let se = ((sum2 / t as f64 - mean * mean) / t as f64).sqrt();
                let m = cs_multiplier(t, 0.95, 100).unwrap();
                cs_out |= (mean - 1.0).abs() > m * se;
                ci_out |= (mean - 1.0).abs() > 1.96 * se;
            }
            cs_miss += usize::from(cs_out);
            ci_miss += usize::from(ci_out);
        }
        let (cs_rate, ci_rate) = (cs_miss as f64 / reps as f64, ci_miss as f64 / reps as f64);
        assert!(cs_rate <= 0.06, "confidence sequence missed in {cs_rate}");
        assert!(
            ci_rate >= 0.2,
            "repeated intervals missed in only {ci_rate}"
        );
    }

    #[test]
    fn running_intersection_and_first_exclusion() {
        let mut cs = ConfidenceSequence::new(0.95, 200).unwrap();
        for t in [100, 200, 400, 800] {
            let values: Vec<f64> = (0..t).map(|i| 0.4 + (i as f64 * 2.1).sin()).collect();
            cs.update(&mean_fit(&values)).unwrap();
        }
        let raw = cs.intervals();
        let running = cs.running_intervals();
        for l in 0..4 {
            assert!(running[[l, 0, 0]] >= raw[[l, 0, 0]] && running[[l, 0, 1]] <= raw[[l, 0, 1]]);
        }
        let p = cs.p_values();
        assert!((1..4).all(|l| p[[l, 0]] <= p[[l - 1, 0]]));
        let first = cs.first_exclusion(0.0)[0].expect("mean 0.4 is detected");
        assert!(running[[first, 0, 0]] > 0.0);
        assert!(p[[first, 0]] < 0.05);
        assert!(first == 0 || running[[first - 1, 0, 0]] <= 0.0);
        // Shrinking samples and other estimates are rejected.
        assert!(cs.update(&mean_fit(&[1.0; 50])).is_err());
        assert!(ConfidenceSequence::new(1.5, 10).is_err());
        assert!(ConfidenceSequence::new(0.9, 0).is_err());
    }
}

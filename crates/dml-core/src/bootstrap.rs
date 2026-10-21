//! Multiplier bootstrap for joint confidence intervals and Romano–Wolf
//! multiple-testing adjustment.

use ndarray::{Array1, Array2, Array3, ArrayView3, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Exp1, StandardNormal};

use crate::error::{DmlError, Result};
use crate::fit::{check_level, confint_from, DmlFit};
use crate::stats::{median, quantile_mut};

/// Distribution of the multiplier weights `ξ_i` (all mean 0, variance 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BootstrapMethod {
    /// `ξ ~ N(0, 1)`
    #[default]
    Normal,
    /// `ξ ~ Exp(1) − 1`
    Bayes,
    /// Mammen-type: `ξ = a/√2 + (b² − 1)/2`, with `a, b ~ N(0, 1)`
    Wild,
}

/// Draw an `(n_boot, n)` matrix of multiplier weights.
pub fn draw_weights(
    method: BootstrapMethod,
    n_boot: usize,
    n: usize,
    rng: &mut impl Rng,
) -> Array2<f64> {
    let shape = (n_boot, n);
    match method {
        BootstrapMethod::Normal => {
            Array2::from_shape_simple_fn(shape, || rng.sample(StandardNormal))
        }
        BootstrapMethod::Bayes => {
            Array2::from_shape_simple_fn(shape, || rng.sample::<f64, _>(Exp1) - 1.0)
        }
        BootstrapMethod::Wild => Array2::from_shape_simple_fn(shape, || {
            let a: f64 = rng.sample(StandardNormal);
            let b: f64 = rng.sample(StandardNormal);
            a / std::f64::consts::SQRT_2 + (b * b - 1.0) / 2.0
        }),
    }
}

/// Bootstrap distribution of the t-statistics of a [`DmlFit`].
///
/// Keeps its own copy of the per-repetition estimates (small `(k, n_rep)`
/// arrays), so it is independent of the fit it came from.
#[derive(Debug, Clone)]
pub struct Bootstrap {
    all_coef: Array2<f64>,
    all_se: Array2<f64>,
    /// Shape `(n_boot, k, n_rep)`.
    t_stats: Array3<f64>,
}

impl DmlFit {
    /// Multiplier bootstrap with `n_boot` draws per repetition:
    /// `t*_b = Σ_i ξ_bi · ψ_i / (J · n · se)`.
    pub fn bootstrap(
        &self,
        method: BootstrapMethod,
        n_boot: usize,
        seed: u64,
    ) -> Result<Bootstrap> {
        if n_boot == 0 {
            return Err(DmlError::InvalidArgument("n_boot must be positive".into()));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let weights: Vec<Array2<f64>> = (0..self.n_rep())
            .map(|_| draw_weights(method, n_boot, self.n_obs, &mut rng))
            .collect();
        self.bootstrap_with_weights(&weights)
    }

    /// Bootstrap with caller-supplied weights, one `(n_boot, n)` matrix per
    /// repetition (e.g. to reproduce DoubleML's draws exactly).
    pub fn bootstrap_with_weights(&self, weights: &[Array2<f64>]) -> Result<Bootstrap> {
        if weights.len() != self.n_rep() {
            return Err(DmlError::DimensionMismatch(format!(
                "{} weight matrices for {} repetitions",
                weights.len(),
                self.n_rep()
            )));
        }
        let n_boot = weights[0].nrows();
        let (n, k) = (self.n_obs, self.n_treat());
        let mut t_stats = Array3::<f64>::zeros((n_boot, k, self.n_rep()));
        for (r, w) in weights.iter().enumerate() {
            if w.dim() != (n_boot, n) {
                return Err(DmlError::DimensionMismatch(
                    "every weight matrix must be (n_boot, n_obs)".into(),
                ));
            }
            // scaled_psi[i, j] = ψ_ij / (J_j · n · se_j)
            let mut scaled = self.psi.index_axis(Axis(2), r).to_owned();
            for j in 0..k {
                let denom = self.psi_deriv[[j, r]] * n as f64 * self.all_se[[j, r]];
                scaled.column_mut(j).mapv_inplace(|v| v / denom);
            }
            t_stats.index_axis_mut(Axis(2), r).assign(&w.dot(&scaled));
        }
        Ok(Bootstrap {
            all_coef: self.all_coef.clone(),
            all_se: self.all_se.clone(),
            t_stats,
        })
    }
}

impl Bootstrap {
    /// Bootstrapped t-statistics, shape `(n_boot, k, n_rep)`.
    pub fn t_stats(&self) -> ArrayView3<'_, f64> {
        self.t_stats.view()
    }

    /// Per-repetition critical values: the `level` quantile of `max_j |t*_j|`.
    pub fn critical_values(&self, level: f64) -> Result<Array1<f64>> {
        check_level(level)?;
        Ok(self
            .t_stats
            .axis_iter(Axis(2))
            .map(|rep| {
                let mut max_abs: Vec<f64> = rep
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().fold(0.0_f64, |m, v| m.max(v.abs())))
                    .collect();
                quantile_mut(&mut max_abs, level)
            })
            .collect())
    }

    /// Simultaneous confidence intervals covering all treatments jointly at
    /// `level`, shape `(k, 2)`.
    pub fn joint_confint(&self, level: f64) -> Result<Array2<f64>> {
        Ok(confint_from(
            &self.all_coef,
            &self.all_se,
            &self.critical_values(level)?,
        ))
    }

    /// Romano–Wolf step-down adjusted p-values (median over repetitions).
    pub fn romano_wolf(&self) -> Array1<f64> {
        let (k, n_rep) = self.all_coef.dim();
        let mut all = Array2::<f64>::zeros((k, n_rep));
        for r in 0..n_rep {
            let boot = self.t_stats.index_axis(Axis(2), r);
            let abs_t: Vec<f64> = (0..k)
                .map(|j| (self.all_coef[[j, r]] / self.all_se[[j, r]]).abs())
                .collect();
            // Treatments in decreasing order of |t|.
            let mut order: Vec<usize> = (0..k).collect();
            order.sort_by(|&a, &b| abs_t[b].total_cmp(&abs_t[a]));
            let mut running = 0.0_f64;
            for (step, &j) in order.iter().enumerate() {
                let remaining = &order[step..];
                let exceed = boot
                    .rows()
                    .into_iter()
                    .filter(|row| {
                        remaining.iter().fold(0.0_f64, |m, &i| m.max(row[i].abs())) >= abs_t[j]
                    })
                    .count();
                let p = (exceed as f64 / boot.nrows() as f64).min(1.0);
                running = running.max(p);
                all[[j, r]] = running;
            }
        }
        all.rows()
            .into_iter()
            .map(|row| median(&row.to_vec()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_have_mean_zero_unit_variance() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for method in [
            BootstrapMethod::Normal,
            BootstrapMethod::Bayes,
            BootstrapMethod::Wild,
        ] {
            let w = draw_weights(method, 200, 500, &mut rng);
            let mean = w.mean().unwrap();
            let var = w.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            assert!(mean.abs() < 0.02, "{method:?} mean {mean}");
            assert!((var - 1.0).abs() < 0.03, "{method:?} var {var}");
        }
    }
}

//! Best linear predictor (BLP) of a CATE and group average treatment
//! effects (GATE), following Semenova & Chernozhukov (2021) and DoubleML.
//!
//! Given an orthogonal signal `φᵢ` with `E[φ | X] = τ(X)` (e.g. the AIPW
//! pseudo-outcome), the BLP is the OLS fit of `φ` on a basis `b(X)`, with
//! heteroskedasticity-robust (HC0) covariance `Ω`. Pointwise inference for
//! `τ̂(x) = b(x)ᵀβ̂` uses `se(x) = √(b(x)ᵀ Ω b(x))`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::error::{DmlError, Result};
use crate::fit::{aggregate, check_level};
use crate::linalg::{cholesky_factor, spd_inverse};
use crate::stats::{median, norm_ppf, norm_sf, quantile_mut};

/// A fitted best linear predictor, one OLS fit per repetition of the signal.
#[derive(Debug, Clone)]
pub struct Blp {
    /// Shape `(p, n_rep)`.
    all_coef: Array2<f64>,
    all_se: Array2<f64>,
    /// HC0 covariance per repetition, each `(p, p)`.
    omega: Vec<Array2<f64>>,
    coef: Array1<f64>,
    se: Array1<f64>,
    is_gate: bool,
}

impl Blp {
    /// Regress each column of `signal` (shape `(n, n_rep)`) on `basis` (shape `(n, p)`).
    pub fn fit(signal: ArrayView2<'_, f64>, basis: ArrayView2<'_, f64>) -> Result<Self> {
        let (n, n_rep) = signal.dim();
        let p = basis.ncols();
        if basis.nrows() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "basis has {} rows, signal has {n}",
                basis.nrows()
            )));
        }
        if p == 0 || n_rep == 0 || n <= p {
            return Err(DmlError::InvalidArgument(
                "BLP needs at least one basis column, one repetition and n > p".into(),
            ));
        }
        let btb = basis.t().dot(&basis);
        let bread = spd_inverse(&btb).map_err(|_| {
            DmlError::InvalidArgument("basis columns are collinear (BᵀB is singular)".into())
        })?;
        let mut all_coef = Array2::zeros((p, n_rep));
        let mut all_se = Array2::zeros((p, n_rep));
        let mut omega = Vec::with_capacity(n_rep);
        for r in 0..n_rep {
            let phi = signal.column(r);
            let beta = bread.dot(&basis.t().dot(&phi));
            let resid = &phi - &basis.dot(&beta);
            // HC0 meat: Bᵀ diag(e²) B
            let weighted = &basis * &resid.mapv(|e| e * e).insert_axis(Axis(1));
            let meat = basis.t().dot(&weighted);
            let om = bread.dot(&meat).dot(&bread);
            all_se.column_mut(r).assign(&om.diag().mapv(f64::sqrt));
            all_coef.column_mut(r).assign(&beta);
            omega.push(om);
        }
        let (coef, se) = aggregate(&all_coef, &all_se);
        Ok(Self {
            all_coef,
            all_se,
            omega,
            coef,
            se,
            is_gate: false,
        })
    }

    /// GATEs: BLP on mutually exclusive 0/1 group dummies.
    pub fn fit_gate(signal: ArrayView2<'_, f64>, groups: ArrayView2<'_, f64>) -> Result<Self> {
        let valid = groups.iter().all(|&g| g == 0.0 || g == 1.0)
            && groups.rows().into_iter().all(|r| r.sum() <= 1.0);
        if !valid {
            return Err(DmlError::InvalidArgument(
                "groups must be mutually exclusive 0/1 dummy columns".into(),
            ));
        }
        let mut blp = Self::fit(signal, groups)?;
        blp.is_gate = true;
        Ok(blp)
    }

    pub fn is_gate(&self) -> bool {
        self.is_gate
    }

    /// Basis coefficients (for GATEs: the group effects).
    pub fn coef(&self) -> &Array1<f64> {
        &self.coef
    }

    /// HC0 standard errors of the coefficients.
    pub fn se(&self) -> &Array1<f64> {
        &self.se
    }

    pub fn t_stat(&self) -> Array1<f64> {
        &self.coef / &self.se
    }

    pub fn p_value(&self) -> Array1<f64> {
        self.t_stat().mapv(|t| 2.0 * norm_sf(t.abs()))
    }

    /// HC0 covariance of the coefficients for one repetition.
    pub fn covariance(&self, rep: usize) -> Option<&Array2<f64>> {
        self.omega.get(rep)
    }

    /// Pointwise intervals for the coefficients, shape `(p, 2)`.
    pub fn confint(&self, level: f64) -> Result<Array2<f64>> {
        check_level(level)?;
        let z = norm_ppf(0.5 + level / 2.0);
        Ok(per_rep_bounds(
            &self.all_coef,
            &self.all_se,
            &Array1::from_elem(self.all_coef.ncols(), z),
        ))
    }

    /// Predicted effects `b(x)ᵀβ̂` and standard errors at new basis rows.
    pub fn predict(&self, basis: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (g, s) = self.predict_all(basis)?;
        Ok(aggregate(&g, &s))
    }

    fn predict_all(&self, basis: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        if basis.ncols() != self.coef.len() {
            return Err(DmlError::DimensionMismatch(format!(
                "basis has {} columns, the BLP was fitted with {}",
                basis.ncols(),
                self.coef.len()
            )));
        }
        let n_rep = self.all_coef.ncols();
        let mut g = Array2::zeros((basis.nrows(), n_rep));
        let mut s = Array2::zeros((basis.nrows(), n_rep));
        for r in 0..n_rep {
            g.column_mut(r).assign(&basis.dot(&self.all_coef.column(r)));
            let bo = basis.dot(&self.omega[r]);
            let var = (&bo * &basis).sum_axis(Axis(1));
            s.column_mut(r).assign(&var.mapv(f64::sqrt));
        }
        Ok((g, s))
    }

    /// Pointwise intervals for `τ(x)` at new basis rows: columns
    /// `[lower, effect, upper]`.
    pub fn confint_at(&self, basis: ArrayView2<'_, f64>, level: f64) -> Result<Array2<f64>> {
        check_level(level)?;
        let z = norm_ppf(0.5 + level / 2.0);
        let crit = Array1::from_elem(self.all_coef.ncols(), z);
        self.intervals_at(basis, &crit)
    }

    /// Simultaneous intervals over all rows of `basis`, with critical values
    /// from a Gaussian multiplier bootstrap of `b(x)ᵀ Ω^{1/2} Z / se(x)`.
    pub fn joint_confint_at(
        &self,
        basis: ArrayView2<'_, f64>,
        level: f64,
        n_boot: usize,
        seed: u64,
    ) -> Result<Array2<f64>> {
        check_level(level)?;
        if n_boot == 0 {
            return Err(DmlError::InvalidArgument("n_boot must be positive".into()));
        }
        let (_, all_s) = self.predict_all(basis)?;
        let p = self.coef.len();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut crit = Array1::zeros(self.all_coef.ncols());
        for (r, om) in self.omega.iter().enumerate() {
            let l = cholesky_factor(om)?;
            let bl = basis.dot(&l);
            let mut max_abs = Vec::with_capacity(n_boot);
            for _ in 0..n_boot {
                let z: Array1<f64> = (0..p)
                    .map(|_| rng.sample::<f64, _>(StandardNormal))
                    .collect();
                let draw = bl.dot(&z);
                let m = draw
                    .iter()
                    .zip(all_s.column(r))
                    .fold(0.0_f64, |m, (&v, &s)| m.max((v / s).abs()));
                max_abs.push(m);
            }
            crit[r] = quantile_mut(&mut max_abs, level);
        }
        self.intervals_at(basis, &crit)
    }

    fn intervals_at(&self, basis: ArrayView2<'_, f64>, crit: &Array1<f64>) -> Result<Array2<f64>> {
        let (g, s) = self.predict_all(basis)?;
        let (effect, _) = aggregate(&g, &s);
        let bounds = per_rep_bounds(&g, &s, crit);
        let mut out = Array2::zeros((basis.nrows(), 3));
        out.column_mut(0).assign(&bounds.column(0));
        out.column_mut(1).assign(&effect);
        out.column_mut(2).assign(&bounds.column(1));
        Ok(out)
    }
}

/// Median over repetitions of `value ∓ crit·se`, shape `(rows, 2)`.
fn per_rep_bounds(values: &Array2<f64>, ses: &Array2<f64>, crit: &Array1<f64>) -> Array2<f64> {
    let rows = values.nrows();
    let mut out = Array2::zeros((rows, 2));
    for i in 0..rows {
        let lo: Vec<f64> = (0..values.ncols())
            .map(|r| values[[i, r]] - crit[r] * ses[[i, r]])
            .collect();
        let hi: Vec<f64> = (0..values.ncols())
            .map(|r| values[[i, r]] + crit[r] * ses[[i, r]])
            .collect();
        out[[i, 0]] = median(&lo);
        out[[i, 1]] = median(&hi);
    }
    out
}

/// Convenience: a single-repetition signal as an `(n, 1)` matrix.
pub(crate) fn as_column(signal: ArrayView1<'_, f64>) -> Array2<f64> {
    signal.to_owned().insert_axis(Axis(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn intercept_only_blp_is_mean_with_hc0_se() {
        let phi = array![1.0, 2.0, 4.0, 5.0];
        let basis = Array2::ones((4, 1));
        let blp = Blp::fit(as_column(phi.view()).view(), basis.view()).unwrap();
        assert!((blp.coef()[0] - 3.0).abs() < 1e-12);
        // HC0: Σe² / n² = (4 + 1 + 1 + 4) / 16
        assert!((blp.se()[0] - (10.0_f64 / 16.0).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn prediction_se_matches_coefficient_se_on_unit_basis() {
        let phi = array![1.0, 3.0, 2.0, 6.0, 5.0, 9.0];
        let basis = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, 2.0],
            [1.0, 1.5],
            [1.0, 3.0]
        ];
        let blp = Blp::fit(as_column(phi.view()).view(), basis.view()).unwrap();
        let (effect, se) = blp.predict(array![[1.0, 0.0], [0.0, 1.0]].view()).unwrap();
        assert!((effect[0] - blp.coef()[0]).abs() < 1e-12);
        assert!((se[1] - blp.se()[1]).abs() < 1e-12);
        let joint = blp.joint_confint_at(basis.view(), 0.95, 2000, 1).unwrap();
        let point = blp.confint_at(basis.view(), 0.95).unwrap();
        assert!((0..6).all(|i| joint[[i, 2]] - joint[[i, 0]] >= point[[i, 2]] - point[[i, 0]]));
    }

    #[test]
    fn gate_requires_exclusive_dummies() {
        let phi = array![1.0, 2.0, 3.0, 4.0];
        let overlapping = array![[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        assert!(Blp::fit_gate(as_column(phi.view()).view(), overlapping.view()).is_err());
        let groups = array![[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]];
        let gate = Blp::fit_gate(as_column(phi.view()).view(), groups.view()).unwrap();
        let diff = gate.coef() - &array![1.5, 3.5];
        assert!(diff.iter().all(|v| v.abs() < 1e-12));
    }
}

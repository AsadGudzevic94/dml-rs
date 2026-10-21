//! Partially linear regression (PLR) model:
//!
//! ```text
//! Y = θ·D + g(X) + ε,   E[ε | D, X] = 0
//! D = m(X) + v,         E[v | X] = 0
//! ```
//!
//! Estimated with the partialling-out score
//! `ψ = (Y − ℓ(X) − θ(D − m(X)))·(D − m(X))`, where `ℓ(X) = E[Y | X]` and
//! `m(X) = E[D | X]` are cross-fitted with the same folds (DML2).

use std::fmt;

use ndarray::Array1;

use crate::crossfit::{cross_fit_predict, Fold, KFold};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::learner::Regressor;
use crate::stats::{norm_cdf, norm_ppf};

/// PLR estimator configuration: two nuisance learners plus the fold setup.
pub struct Plr {
    ml_l: Box<dyn Regressor>,
    ml_m: Box<dyn Regressor>,
    n_folds: usize,
    seed: u64,
}

impl Plr {
    /// `ml_l` learns `E[Y | X]`, `ml_m` learns `E[D | X]`.
    /// Defaults: 5 folds, seed 0.
    pub fn new(ml_l: impl Regressor + 'static, ml_m: impl Regressor + 'static) -> Self {
        Self {
            ml_l: Box::new(ml_l),
            ml_m: Box::new(ml_m),
            n_folds: 5,
            seed: 0,
        }
    }

    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.n_folds = n_folds;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Draw seeded K-fold splits and fit.
    pub fn fit(&self, data: &DmlData) -> Result<PlrResult> {
        let folds = KFold::new(self.n_folds, self.seed)?.split(data.n_obs())?;
        self.fit_with_folds(data, &folds)
    }

    /// Fit with caller-supplied folds (e.g. to match another implementation).
    pub fn fit_with_folds(&self, data: &DmlData, folds: &[Fold]) -> Result<PlrResult> {
        let (x, y, d) = (data.x(), data.y(), data.d());
        let l_hat = cross_fit_predict(self.ml_l.as_ref(), x, y, folds)?;
        let m_hat = cross_fit_predict(self.ml_m.as_ref(), x, d, folds)?;

        let u: Array1<f64> = &y - &l_hat;
        let v: Array1<f64> = &d - &m_hat;
        let vv = v.dot(&v);
        if vv <= f64::EPSILON * d.dot(&d).max(1.0) {
            return Err(DmlError::InvalidArgument(
                "treatment residuals are zero: D is fully explained by X".into(),
            ));
        }
        let coef = u.dot(&v) / vv;

        let n = data.n_obs() as f64;
        let psi: Array1<f64> = (&u - &(coef * &v)) * &v;
        let j = -vv / n;
        let sigma2 = psi.dot(&psi) / n / (j * j);
        let se = (sigma2 / n).sqrt();

        Ok(PlrResult {
            coef,
            se,
            n_obs: data.n_obs(),
            l_hat,
            m_hat,
            psi,
        })
    }
}

/// Fitted PLR: point estimate, standard error and cross-fitted nuisances.
#[derive(Debug, Clone)]
pub struct PlrResult {
    pub coef: f64,
    pub se: f64,
    pub n_obs: usize,
    /// Out-of-fold predictions of `E[Y | X]`.
    pub l_hat: Array1<f64>,
    /// Out-of-fold predictions of `E[D | X]`.
    pub m_hat: Array1<f64>,
    /// Score `ψ_i` evaluated at the estimate.
    pub psi: Array1<f64>,
}

impl PlrResult {
    pub fn t_stat(&self) -> f64 {
        self.coef / self.se
    }

    /// Two-sided p-value for `H0: θ = 0`.
    pub fn p_value(&self) -> f64 {
        2.0 * norm_cdf(-self.t_stat().abs())
    }

    /// Wald confidence interval at `level` (e.g. 0.95).
    pub fn confint(&self, level: f64) -> Result<(f64, f64)> {
        if !(level > 0.0 && level < 1.0) {
            return Err(DmlError::InvalidArgument(
                "confidence level must be in (0, 1)".into(),
            ));
        }
        let z = norm_ppf(0.5 + level / 2.0);
        Ok((self.coef - z * self.se, self.coef + z * self.se))
    }
}

impl fmt::Display for PlrResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (lo, hi) = self.confint(0.95).map_err(|_| fmt::Error)?;
        writeln!(
            f,
            "{:>10} {:>10} {:>8} {:>8} {:>10} {:>10}",
            "coef", "std err", "t", "P>|t|", "2.5 %", "97.5 %"
        )?;
        write!(
            f,
            "{:>10.4} {:>10.4} {:>8.3} {:>8.4} {:>10.4} {:>10.4}",
            self.coef,
            self.se,
            self.t_stat(),
            self.p_value(),
            lo,
            hi
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Ols;
    use ndarray::{Array1, Array2};

    fn toy_data() -> DmlData {
        let n = 60;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
        let d: Array1<f64> = (0..n)
            .map(|i| 0.3 * x[[i, 0]] + ((i * 13) % 5) as f64 - 2.0)
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| 0.5 * d[i] + x[[i, 1]] + ((i * 17) % 3) as f64 - 1.0)
            .collect();
        DmlData::new(y, d, x).unwrap()
    }

    #[test]
    fn coef_matches_residual_formula() {
        let data = toy_data();
        let res = Plr::new(Ols::new(), Ols::new()).seed(3).fit(&data).unwrap();
        let u = &data.y() - &res.l_hat;
        let v = &data.d() - &res.m_hat;
        assert!((res.coef - u.dot(&v) / v.dot(&v)).abs() < 1e-12);
        // The score is zero on average at the DML2 solution.
        assert!(res.psi.mean().unwrap().abs() < 1e-10);
    }

    #[test]
    fn inference_is_consistent() {
        let res = Plr::new(Ols::new(), Ols::new()).fit(&toy_data()).unwrap();
        let (lo, hi) = res.confint(0.95).unwrap();
        assert!(lo < res.coef && res.coef < hi);
        assert!(((hi - lo) / 2.0 - 1.959_964 * res.se).abs() < 1e-5);
        assert!(res.p_value() > 0.0 && res.p_value() <= 1.0);
        assert!(res.confint(1.0).is_err());
    }

    #[test]
    fn rejects_treatment_explained_by_x() {
        let x = Array2::from_shape_fn((20, 1), |(i, _)| i as f64);
        let d = x.column(0).mapv(|v| 2.0 * v + 1.0);
        let y = d.mapv(|v| v * 0.5);
        let data = DmlData::new(y, d, x).unwrap();
        assert!(Plr::new(Ols::new(), Ols::new()).fit(&data).is_err());
    }
}

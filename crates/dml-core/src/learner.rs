use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{DmlError, Result};
use crate::linalg::cholesky_solve;

/// A regression learner for nuisance functions such as `E[Y | X]` or `E[D | X]`.
///
/// `fresh` returns an unfitted copy with the same hyperparameters, so
/// cross-fitting can train an independent model on each fold.
pub trait Regressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()>;
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
    fn fresh(&self) -> Box<dyn Regressor>;
}

/// A classification learner returning `P(label = 1 | X)`, used for
/// propensity scores in models with a binary treatment (IRM, later phases).
pub trait Classifier {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()>;
    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
    fn fresh(&self) -> Box<dyn Classifier>;
}

/// Ordinary least squares with an intercept and optional ridge penalty.
///
/// `ridge` (default 0) is added to the diagonal for slopes only, which keeps
/// the fit well defined when covariates are collinear.
#[derive(Debug, Clone, Default)]
pub struct Ols {
    ridge: f64,
    coef: Option<Array1<f64>>,
    intercept: f64,
}

impl Ols {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ridge(ridge: f64) -> Self {
        Self {
            ridge: ridge.max(0.0),
            ..Self::default()
        }
    }

    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.coef.as_ref()
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

impl Regressor for Ols {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        let (n, p) = x.dim();
        if y.len() != n {
            return Err(DmlError::DimensionMismatch("Ols::fit x/y rows".into()));
        }
        if n == 0 {
            return Err(DmlError::InvalidArgument("empty training set".into()));
        }
        // Center to absorb the intercept, then solve (Xc'Xc + λI) b = Xc'yc.
        let x_mean = x.mean_axis(ndarray::Axis(0)).expect("n > 0");
        let y_mean = y.mean().expect("n > 0");
        let xc: Array2<f64> = &x - &x_mean;
        let yc: Array1<f64> = &y - y_mean;
        let mut xtx = xc.t().dot(&xc);
        for j in 0..p {
            xtx[[j, j]] += self.ridge;
        }
        let xty = xc.t().dot(&yc);
        let coef = if p == 0 {
            Array1::zeros(0)
        } else {
            cholesky_solve(&xtx, &xty)?
        };
        self.intercept = y_mean - x_mean.dot(&coef);
        self.coef = Some(coef);
        Ok(())
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let coef = self.coef.as_ref().ok_or(DmlError::NotFitted)?;
        if x.ncols() != coef.len() {
            return Err(DmlError::DimensionMismatch(
                "Ols::predict feature count".into(),
            ));
        }
        Ok(x.dot(coef) + self.intercept)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Ols::with_ridge(self.ridge))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn recovers_exact_linear_fit() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [0.5, 3.0]];
        let y: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| 1.5 + 2.0 * r[0] - 0.5 * r[1])
            .collect();
        let mut m = Ols::new();
        m.fit(x.view(), y.view()).unwrap();
        let c = m.coef().unwrap();
        assert!((c[0] - 2.0).abs() < 1e-10 && (c[1] + 0.5).abs() < 1e-10);
        assert!((m.intercept() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn predict_before_fit_errors() {
        assert_eq!(
            Ols::new().predict(array![[1.0]].view()).unwrap_err(),
            DmlError::NotFitted
        );
    }

    #[test]
    fn ridge_handles_collinearity() {
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];
        assert!(Ols::new().fit(x.view(), y.view()).is_err());
        assert!(Ols::with_ridge(1e-6).fit(x.view(), y.view()).is_ok());
    }
}

use ndarray::{Array1, ArrayView1, ArrayView2};

use super::{center, Centered, Regressor};
use crate::error::{DmlError, Result};
use crate::linalg::cholesky_solve;

/// Ordinary least squares with an intercept and optional ridge penalty.
///
/// `ridge` (default 0) is added to the diagonal for slopes only, which keeps
/// the fit well defined when covariates are collinear.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

impl Ols {
    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        let p = x.ncols();
        // Centre to absorb the intercept, then solve (Xc'Xc + λI) b = Xc'yc.
        let Centered {
            x_mean,
            y_mean,
            xc,
            yc,
        } = center(x, y, w, false)?;
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
}

impl Regressor for Ols {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_impl(x, y, None)
    }

    /// Weighted least squares: minimises `Σ wᵢ (yᵢ − b − xᵢᵀβ)² + ridge·‖β‖²`.
    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.fit_impl(x, y, Some(w))
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
    fn integer_weights_equal_row_duplication() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![1.0, 0.5, 2.5, 2.0];
        let mut weighted = Ols::new();
        weighted
            .fit_weighted(x.view(), y.view(), array![1.0, 2.0, 1.0, 3.0].view())
            .unwrap();
        let xd = array![[0.0], [1.0], [1.0], [2.0], [3.0], [3.0], [3.0]];
        let yd = array![1.0, 0.5, 0.5, 2.5, 2.0, 2.0, 2.0];
        let mut dup = Ols::new();
        dup.fit(xd.view(), yd.view()).unwrap();
        assert!((weighted.coef().unwrap()[0] - dup.coef().unwrap()[0]).abs() < 1e-12);
        assert!((weighted.intercept() - dup.intercept()).abs() < 1e-12);
    }

    #[test]
    fn ridge_handles_collinearity() {
        let x = array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];
        assert!(Ols::new().fit(x.view(), y.view()).is_err());
        assert!(Ols::with_ridge(1e-6).fit(x.view(), y.view()).is_ok());
    }
}

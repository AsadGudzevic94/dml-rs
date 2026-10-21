use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};

use super::{center, Centered, Regressor};
use crate::error::{DmlError, Result};

/// Lasso regression with an unpenalised intercept, fitted by cyclic
/// coordinate descent.
///
/// Minimises `1/(2n)·‖y − b − Xw‖² + α‖w‖₁`, the same objective as
/// scikit-learn's `Lasso(alpha=α)`. The classic DML learner for
/// high-dimensional confounders.
#[derive(Debug, Clone)]
pub struct Lasso {
    alpha: f64,
    tol: f64,
    max_iter: usize,
    coef: Option<Array1<f64>>,
    intercept: f64,
}

impl Lasso {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.max(0.0),
            tol: 1e-12,
            max_iter: 100_000,
            coef: None,
            intercept: 0.0,
        }
    }

    /// Stop when no coefficient moves by more than `tol · max|w|` in a sweep.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.coef.as_ref()
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

impl Lasso {
    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        let (n, p) = x.dim();
        let Centered {
            x_mean,
            y_mean,
            xc: centered,
            yc: mut resid,
        } = center(x, y, w, true)?;
        // Column-major copy so each coordinate update reads a contiguous column.
        let mut xc = Array2::<f64>::zeros((n, p).f());
        xc.assign(&centered);
        let col_sq: Vec<f64> = xc.columns().into_iter().map(|c| c.dot(&c)).collect();
        let n_alpha = n as f64 * self.alpha;
        let mut w = Array1::<f64>::zeros(p);

        let mut converged = p == 0;
        for _ in 0..self.max_iter {
            if converged {
                break;
            }
            let mut max_change = 0.0_f64;
            let mut max_w = 0.0_f64;
            for j in 0..p {
                if col_sq[j] == 0.0 {
                    continue;
                }
                let col = xc.column(j);
                let old = w[j];
                let rho = col.dot(&resid) + old * col_sq[j];
                let new = soft_threshold(rho, n_alpha) / col_sq[j];
                if new != old {
                    resid.scaled_add(old - new, &col);
                    w[j] = new;
                }
                max_change = max_change.max((new - old).abs());
                max_w = max_w.max(new.abs());
            }
            converged = max_w == 0.0 || max_change <= self.tol * max_w;
        }
        if !converged {
            return Err(DmlError::Convergence(format!(
                "Lasso did not converge in {} sweeps",
                self.max_iter
            )));
        }
        self.intercept = y_mean - x_mean.dot(&w);
        self.coef = Some(w);
        Ok(())
    }
}

impl Regressor for Lasso {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_impl(x, y, None)
    }

    /// Minimises `1/(2n)·Σ wᵢ rᵢ² + α‖β‖₁` with weights rescaled to sum to
    /// `n`, as scikit-learn's `Lasso.fit(sample_weight=w)`.
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
                "Lasso::predict feature count".into(),
            ));
        }
        Ok(x.dot(coef) + self.intercept)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self::new(self.alpha).tol(self.tol).max_iter(self.max_iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Ols;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((40, 3), |(i, j)| {
            (((i + 1) * (j + 2) * 7) % 13) as f64 - 6.0
        });
        let y = x
            .rows()
            .into_iter()
            .map(|r| 2.0 + 1.5 * r[0] - 0.2 * r[2])
            .collect();
        (x, y)
    }

    #[test]
    fn zero_alpha_equals_ols() {
        let (x, y) = data();
        let mut lasso = Lasso::new(0.0);
        lasso.fit(x.view(), y.view()).unwrap();
        let mut ols = Ols::new();
        ols.fit(x.view(), y.view()).unwrap();
        let diff = lasso.coef().unwrap() - ols.coef().unwrap();
        assert!(diff.iter().all(|v| v.abs() < 1e-8));
    }

    #[test]
    fn satisfies_kkt_conditions() {
        let (x, y) = data();
        let alpha = 0.5;
        let mut m = Lasso::new(alpha);
        m.fit(x.view(), y.view()).unwrap();
        let resid = &y - &m.predict(x.view()).unwrap();
        let n = x.nrows() as f64;
        for (j, &wj) in m.coef().unwrap().iter().enumerate() {
            let g = x.column(j).dot(&resid) / n;
            if wj != 0.0 {
                assert!((g - alpha * wj.signum()).abs() < 1e-8, "active {j}: {g}");
            } else {
                assert!(g.abs() <= alpha + 1e-8, "inactive {j}: {g}");
            }
        }
    }

    #[test]
    fn large_alpha_gives_intercept_only() {
        let (x, y) = data();
        let mut m = Lasso::new(1e6);
        m.fit(x.view(), y.view()).unwrap();
        assert!(m.coef().unwrap().iter().all(|&v| v == 0.0));
        assert!((m.intercept() - y.mean().unwrap()).abs() < 1e-12);
        assert_eq!(
            m.predict(array![[0.0, 0.0, 0.0]].view()).unwrap()[0],
            m.intercept()
        );
    }
}

use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::Classifier;
use crate::error::{DmlError, Result};
use crate::linalg::cholesky_solve;

/// L2-penalised logistic regression with an unpenalised intercept, fitted by
/// Newton's method with backtracking line search.
///
/// Minimises `C · Σ logloss + ½‖w‖²`, the same objective as scikit-learn's
/// `LogisticRegression(C=c)`; the default `C = 1` matches scikit-learn's default.
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    c: f64,
    tol: f64,
    max_iter: usize,
    coef: Option<Array1<f64>>,
    intercept: f64,
    n_iter: usize,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LogisticRegression {
    /// Inverse regularisation strength `c` (larger = weaker penalty).
    pub fn new(c: f64) -> Self {
        Self {
            c,
            tol: 1e-10,
            max_iter: 100,
            coef: None,
            intercept: 0.0,
            n_iter: 0,
        }
    }

    /// No penalty. Fails to converge if the classes are perfectly separable.
    pub fn unpenalized() -> Self {
        Self::new(f64::INFINITY)
    }

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

/// `log(1 + e^η)` without overflow.
fn softplus(eta: f64) -> f64 {
    if eta > 0.0 {
        eta + (-eta).exp().ln_1p()
    } else {
        eta.exp().ln_1p()
    }
}

fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

impl LogisticRegression {
    /// Newton steps of the last fit (scikit-learn's `n_iter_`).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }

    /// `[intercept, coef…]` of the fitted model, to warm-start another fit.
    pub(super) fn params(&self) -> Option<Array1<f64>> {
        let coef = self.coef.as_ref()?;
        let mut beta = Array1::zeros(coef.len() + 1);
        beta[0] = self.intercept;
        beta.slice_mut(ndarray::s![1..]).assign(coef);
        Some(beta)
    }

    /// Newton's method (optionally weighted) started from
    /// `init = [intercept, coef…]` if given, else from the intercept-only fit.
    pub(super) fn fit_from(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        weights: Option<ArrayView1<'_, f64>>,
        init: Option<&Array1<f64>>,
    ) -> Result<()> {
        self.fit_impl(x, y, weights, init)
    }

    /// Newton's method on `Σ wᵢ·loglossᵢ + ‖coef‖² / (2C)` (unit weights if
    /// `weights` is `None`), the minimiser scikit-learn's `sample_weight` gives.
    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        weights: Option<ArrayView1<'_, f64>>,
        init: Option<&Array1<f64>>,
    ) -> Result<()> {
        let (n, p) = x.dim();
        let sw = match weights {
            Some(w) if w.len() != n => {
                return Err(DmlError::DimensionMismatch(
                    "LogisticRegression::fit_weighted weights length".into(),
                ))
            }
            Some(w) if !w.iter().all(|&v| v.is_finite() && v >= 0.0) || w.sum() <= 0.0 => {
                return Err(DmlError::InvalidArgument(
                    "sample weights must be finite, non-negative and not all zero".into(),
                ))
            }
            Some(w) => w.to_owned(),
            None => Array1::ones(n),
        };
        if y.len() != n {
            return Err(DmlError::DimensionMismatch(
                "LogisticRegression::fit x/y rows".into(),
            ));
        }
        if !y.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(DmlError::InvalidArgument(
                "LogisticRegression labels must be 0 or 1".into(),
            ));
        }
        let n1 = y.sum();
        if n1 == 0.0 || n1 == n as f64 {
            return Err(DmlError::InvalidArgument(
                "LogisticRegression needs both classes in the training data".into(),
            ));
        }
        let lambda = if self.c.is_finite() {
            1.0 / self.c
        } else {
            0.0
        };
        // Design with a leading column of ones; beta = [intercept, w].
        let xa = concatenate![Axis(1), Array2::ones((n, 1)), x];
        let objective = |beta: &Array1<f64>| -> f64 {
            let eta = xa.dot(beta);
            let loss: f64 = eta
                .iter()
                .zip(y.iter())
                .zip(sw.iter())
                .map(|((&e, &t), &w)| w * (softplus(e) - t * e))
                .sum();
            loss + 0.5
                * lambda
                * beta
                    .slice(ndarray::s![1..])
                    .dot(&beta.slice(ndarray::s![1..]))
        };

        let mut beta = match init {
            Some(b) if b.len() == p + 1 => b.clone(),
            _ => {
                let mean = sw.dot(&y) / sw.sum();
                let mut b = Array1::<f64>::zeros(p + 1);
                b[0] = (mean / (1.0 - mean)).ln();
                b
            }
        };
        let mut obj = objective(&beta);
        let mut converged = false;
        let mut steps = 0;
        for _ in 0..self.max_iter {
            steps += 1;
            let prob = xa.dot(&beta).mapv(sigmoid);
            let mut grad = xa.t().dot(&(&sw * &(&prob - &y)));
            let w = &sw * &prob.mapv(|q| q * (1.0 - q));
            let mut hess = xa.t().dot(&(&xa * &w.insert_axis(Axis(1))));
            for j in 1..=p {
                grad[j] += lambda * beta[j];
                hess[[j, j]] += lambda;
            }
            let step = cholesky_solve(&hess, &grad)?;
            // Backtracking: halve the Newton step until the objective decreases.
            let mut t = 1.0;
            let mut candidate = &beta - &step;
            let mut cand_obj = objective(&candidate);
            while cand_obj > obj && t > 1e-10 {
                t *= 0.5;
                candidate = &beta - &(t * &step);
                cand_obj = objective(&candidate);
            }
            let change = (t * &step).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            beta = candidate;
            obj = cand_obj;
            if change <= self.tol * beta.iter().fold(1.0_f64, |m, v| m.max(v.abs())) {
                converged = true;
                break;
            }
        }
        if !converged || !beta.iter().all(|v| v.is_finite()) {
            return Err(DmlError::Convergence(format!(
                "LogisticRegression did not converge in {} Newton steps (separable classes?)",
                self.max_iter
            )));
        }
        self.intercept = beta[0];
        self.coef = Some(beta.slice(ndarray::s![1..]).to_owned());
        self.n_iter = steps;
        Ok(())
    }
}

impl Classifier for LogisticRegression {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_from(x, y, None, None)
    }

    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.fit_impl(x, y, Some(w), None)
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let coef = self.coef.as_ref().ok_or(DmlError::NotFitted)?;
        if x.ncols() != coef.len() {
            return Err(DmlError::DimensionMismatch(
                "LogisticRegression::predict_proba feature count".into(),
            ));
        }
        Ok((x.dot(coef) + self.intercept).mapv(sigmoid))
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::new(Self::new(self.c).tol(self.tol).max_iter(self.max_iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn integer_weights_equal_repeated_rows() {
        let x = array![
            [0.0, 1.0],
            [1.0, 0.5],
            [2.0, -1.0],
            [3.0, 0.0],
            [1.5, 2.0],
            [0.5, -0.5]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let w = array![1.0, 2.0, 1.0, 3.0, 1.0, 2.0];
        let mut weighted = LogisticRegression::new(2.0).tol(1e-12);
        weighted.fit_weighted(x.view(), y.view(), w.view()).unwrap();
        let rows: Vec<usize> = (0..6)
            .flat_map(|i| std::iter::repeat(i).take(w[i] as usize))
            .collect();
        let mut repeated = LogisticRegression::new(2.0).tol(1e-12);
        repeated
            .fit(
                x.select(Axis(0), &rows).view(),
                y.select(Axis(0), &rows).view(),
            )
            .unwrap();
        let (a, b) = (
            weighted.predict_proba(x.view()).unwrap(),
            repeated.predict_proba(x.view()).unwrap(),
        );
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).abs() < 1e-10));
        assert!(weighted
            .fit_weighted(x.view(), y.view(), (-&w).view())
            .is_err());
    }

    #[test]
    fn gradient_is_zero_at_optimum() {
        let x = array![
            [0.0, 1.0],
            [1.0, 0.5],
            [2.0, -1.0],
            [3.0, 0.0],
            [1.5, 2.0],
            [0.5, -0.5]
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let mut m = LogisticRegression::new(2.0);
        m.fit(x.view(), y.view()).unwrap();
        let p = m.predict_proba(x.view()).unwrap();
        let r = &p - &y;
        // Intercept gradient: Σ (p − y) = 0; slopes: C·Xᵀ(p − y) + w = 0.
        assert!(r.sum().abs() < 1e-9);
        let g = 2.0 * x.t().dot(&r) + m.coef().unwrap();
        assert!(g.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn rejects_single_class_and_bad_labels() {
        let x = array![[0.0], [1.0]];
        let mut m = LogisticRegression::default();
        assert!(m.fit(x.view(), array![1.0, 1.0].view()).is_err());
        assert!(m.fit(x.view(), array![0.0, 2.0].view()).is_err());
    }

    #[test]
    fn separable_unpenalized_reports_non_convergence() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let err = LogisticRegression::unpenalized()
            .fit(x.view(), y.view())
            .unwrap_err();
        assert!(matches!(err, DmlError::Convergence(_) | DmlError::Singular));
    }
}

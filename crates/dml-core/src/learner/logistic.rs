use std::ops::Range;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::Classifier;
use crate::error::{DmlError, Result};
use crate::linalg::cholesky_solve;

/// L2-penalised logistic regression with an unpenalised intercept, fitted by
/// Newton's method with backtracking line search.
///
/// Minimises `C · Σ logloss + ½‖w‖²`, the same objective as scikit-learn's
/// `LogisticRegression(C=c)`; the default `C = 1` matches scikit-learn's default.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// `(log(1 + e^η), sigmoid(η))` from one exponential, without overflow.
fn softplus_sigmoid(eta: f64) -> (f64, f64) {
    let e = (-eta.abs()).exp();
    if eta >= 0.0 {
        (eta + e.ln_1p(), 1.0 / (1.0 + e))
    } else {
        (e.ln_1p(), e / (1.0 + e))
    }
}

/// Sums `block(rows, acc)` over consecutive blocks of `n` rows into a vector of
/// length `len`, the blocks in parallel when the `parallel` feature is on. The
/// blocks depend only on `n` and `len` and are added in order, so the result
/// does not depend on the number of threads.
fn sum_blocks(n: usize, len: usize, block: impl Fn(Range<usize>, &mut [f64]) + Sync) -> Vec<f64> {
    // 2048 rows per block, or more if the partial sums would exceed 32 MB.
    let size = 2048.max((n * len) >> 22);
    let run = |start: usize| {
        let mut acc = vec![0.0; len];
        block(start..(start + size).min(n), &mut acc);
        acc
    };
    let starts: Vec<usize> = (0..n).step_by(size).collect();
    #[cfg(feature = "parallel")]
    let parts: Vec<Vec<f64>> = {
        use rayon::prelude::*;
        starts.into_par_iter().map(run).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let parts: Vec<Vec<f64>> = starts.into_iter().map(run).collect();
    let mut total = vec![0.0; len];
    for part in parts {
        for (t, v) in total.iter_mut().zip(part) {
            *t += v;
        }
    }
    total
}

fn linear_predictor(xi: &[f64], beta: &[f64]) -> f64 {
    beta[0] + xi.iter().zip(&beta[1..]).map(|(a, b)| a * b).sum::<f64>()
}

/// Weighted log loss at `beta = [intercept, coef…]` (no penalty); `x` is
/// row-major.
fn loss_pass(x: &[f64], y: &[f64], sw: &[f64], beta: &[f64]) -> f64 {
    let p = beta.len() - 1;
    sum_blocks(y.len(), 1, |rows, acc| {
        for i in rows {
            let eta = linear_predictor(&x[i * p..(i + 1) * p], beta);
            acc[0] += sw[i] * (softplus_sigmoid(eta).0 - y[i] * eta);
        }
    })[0]
}

/// Weighted log loss and its gradient at `beta` (no penalty).
fn gradient_pass(x: &[f64], y: &[f64], sw: &[f64], beta: &[f64]) -> (f64, Array1<f64>) {
    let q = beta.len();
    let p = q - 1;
    let sums = sum_blocks(y.len(), 1 + q, |rows, acc| {
        let (loss, grad) = acc.split_first_mut().expect("non-empty");
        for i in rows {
            let xi = &x[i * p..(i + 1) * p];
            let eta = linear_predictor(xi, beta);
            let (softplus, prob) = softplus_sigmoid(eta);
            *loss += sw[i] * (softplus - y[i] * eta);
            let r = sw[i] * (prob - y[i]);
            grad[0] += r;
            for (g, xj) in grad[1..].iter_mut().zip(xi) {
                *g += r * xj;
            }
        }
    });
    (sums[0], Array1::from(sums[1..].to_vec()))
}

/// Weighted log loss, its gradient and its Hessian `Σ wᵢ pᵢ(1 − pᵢ) zᵢzᵢᵀ`
/// with `zᵢ = [1, xᵢ]` at `beta` (no penalty), in one pass over the rows. The
/// Hessian's upper triangle is updated from four rows at a time, so that the
/// accumulator is read and written a quarter as often.
fn newton_pass(x: &[f64], y: &[f64], sw: &[f64], beta: &[f64]) -> (f64, Array1<f64>, Array2<f64>) {
    const GROUP: usize = 4;
    let q = beta.len();
    let p = q - 1;
    let sums = sum_blocks(y.len(), 1 + q + q * q, |rows, acc| {
        let (loss, rest) = acc.split_first_mut().expect("non-empty");
        let (grad, hess) = rest.split_at_mut(q);
        let mut start = rows.start;
        while start < rows.end {
            let len = GROUP.min(rows.end - start);
            // Rows of the group, their residuals and Hessian weights (zero
            // for the padding of a short last group).
            let mut xs = [&x[start * p..(start + 1) * p]; GROUP];
            let (mut r, mut h) = ([0.0; GROUP], [0.0; GROUP]);
            for g in 0..len {
                let i = start + g;
                xs[g] = &x[i * p..(i + 1) * p];
                let eta = linear_predictor(xs[g], beta);
                let (softplus, prob) = softplus_sigmoid(eta);
                *loss += sw[i] * (softplus - y[i] * eta);
                r[g] = sw[i] * (prob - y[i]);
                h[g] = sw[i] * prob * (1.0 - prob);
            }
            grad[0] += r.iter().sum::<f64>();
            hess[0] += h.iter().sum::<f64>();
            for j in 0..p {
                let a = [
                    h[0] * xs[0][j],
                    h[1] * xs[1][j],
                    h[2] * xs[2][j],
                    h[3] * xs[3][j],
                ];
                grad[j + 1] +=
                    r[0] * xs[0][j] + r[1] * xs[1][j] + r[2] * xs[2][j] + r[3] * xs[3][j];
                hess[j + 1] += a[0] + a[1] + a[2] + a[3];
                let row = &mut hess[(j + 1) * q + j + 1..(j + 2) * q];
                for (k, hk) in row.iter_mut().enumerate() {
                    let k = j + k;
                    *hk += a[0] * xs[0][k] + a[1] * xs[1][k] + a[2] * xs[2][k] + a[3] * xs[3][k];
                }
            }
            start += len;
        }
    });
    let grad = Array1::from(sums[1..=q].to_vec());
    let mut hess = Array2::from_shape_vec((q, q), sums[1 + q..].to_vec()).expect("q x q");
    for j in 0..q {
        for k in 0..j {
            hess[[j, k]] = hess[[k, j]];
        }
    }
    (sums[0], grad, hess)
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
        let x = x.as_standard_layout();
        let x = x.as_slice().expect("standard layout");
        let (y, sw) = (y.to_vec(), sw.to_vec());
        let penalty = |beta: &Array1<f64>| {
            0.5 * lambda
                * beta
                    .slice(ndarray::s![1..])
                    .dot(&beta.slice(ndarray::s![1..]))
        };
        let objective =
            |beta: &Array1<f64>| loss_pass(x, &y, &sw, beta.as_slice().unwrap()) + penalty(beta);
        // Objective, gradient and Hessian at `beta`.
        let derivatives = |beta: &Array1<f64>| {
            let (loss, mut grad, mut hess) = newton_pass(x, &y, &sw, beta.as_slice().unwrap());
            for j in 1..=p {
                grad[j] += lambda * beta[j];
                hess[[j, j]] += lambda;
            }
            (loss + penalty(beta), grad, hess)
        };

        let mut beta = match init {
            Some(b) if b.len() == p + 1 => b.clone(),
            _ => {
                let mean =
                    sw.iter().zip(&y).map(|(w, t)| w * t).sum::<f64>() / sw.iter().sum::<f64>();
                let mut b = Array1::<f64>::zeros(p + 1);
                b[0] = (mean / (1.0 - mean)).ln();
                b
            }
        };
        // Objective and gradient at `beta`, keeping the Hessian of an earlier point.
        let gradient = |beta: &Array1<f64>, hess: &Array2<f64>| {
            let (loss, mut grad) = gradient_pass(x, &y, &sw, beta.as_slice().unwrap());
            for j in 1..=p {
                grad[j] += lambda * beta[j];
            }
            (loss + penalty(beta), grad, hess.clone())
        };
        let scale = |beta: &Array1<f64>| beta.iter().fold(1.0_f64, |m, v| m.max(v.abs()));

        let (mut obj, mut grad, mut hess) = derivatives(&beta);
        let mut converged = false;
        let mut steps = 0;
        for _ in 0..self.max_iter {
            steps += 1;
            let step = cholesky_solve(&hess, &grad)?;
            let size = step.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            // After a small step the Hessian hardly changes: the next steps reuse
            // it and need only the gradient, which costs a fraction of the
            // Hessian. Convergence stays fast (the error shrinks by about the
            // step size each time) and the solution, where the gradient vanishes,
            // is the same.
            let lazy = size <= 1e-3 * scale(&beta);
            let at = |candidate: &Array1<f64>| {
                if lazy {
                    gradient(candidate, &hess)
                } else {
                    derivatives(candidate)
                }
            };
            // Backtracking: halve the step until the objective decreases. The
            // full step is usually accepted, and its pass already has the
            // derivatives for the next step. Close to the optimum the predicted
            // decrease (half the Newton decrement) is below the rounding error of
            // the objective, which cannot judge the step any more; the full step
            // is reliable there and is taken.
            let decrement = grad.dot(&step);
            let mut t = 1.0;
            let mut candidate = &beta - &step;
            let mut next = at(&candidate);
            if next.0 > obj && decrement > 1e-10 * obj.abs() {
                let mut cand_obj = next.0;
                while cand_obj > obj && t > 1e-10 {
                    t *= 0.5;
                    candidate = &beta - &(t * &step);
                    cand_obj = objective(&candidate);
                }
                next = derivatives(&candidate);
            }
            let change = t * size;
            beta = candidate;
            (obj, grad, hess) = next;
            if change <= self.tol * scale(&beta) {
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
    use ndarray::{array, Axis};

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

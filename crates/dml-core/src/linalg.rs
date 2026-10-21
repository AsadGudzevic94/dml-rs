//! Minimal dense linear algebra so the core crate has no BLAS dependency.

use ndarray::{Array1, Array2};

use crate::error::{DmlError, Result};

/// Solve `A x = b` for symmetric positive-definite `A` via Cholesky.
pub(crate) fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(DmlError::DimensionMismatch("cholesky_solve".into()));
    }
    // Pivots below this are treated as zero: rounding turns an exactly
    // singular matrix into one with tiny positive pivots.
    let max_diag = (0..n).map(|i| a[[i, i]].abs()).fold(0.0_f64, f64::max);
    let tol = max_diag * n as f64 * f64::EPSILON * 16.0;
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= tol || !s.is_finite() {
                    return Err(DmlError::Singular);
                }
                l[[i, i]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Forward: L z = b
    let mut z = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * z[k];
        }
        z[i] = s / l[[i, i]];
    }
    // Backward: L^T x = z
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= l[[k, i]] * x[k];
        }
        x[i] = s / l[[i, i]];
    }
    Ok(x)
}

/// Inverse of a symmetric positive-definite matrix (column by column).
pub(crate) fn spd_inverse(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        inv.column_mut(j).assign(&cholesky_solve(a, &e)?);
    }
    Ok(inv)
}

/// Lower-triangular Cholesky factor `L` with `A = L Lᵀ`.
pub(crate) fn cholesky_factor(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return Err(DmlError::Singular);
                }
                l[[i, i]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn solves_spd_system() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let b = array![2.0, 1.0];
        let x = cholesky_solve(&a, &b).unwrap();
        let r = a.dot(&x) - &b;
        assert!(r.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn rejects_singular() {
        let a = array![[1.0, 1.0], [1.0, 1.0]];
        assert_eq!(
            cholesky_solve(&a, &array![1.0, 1.0]),
            Err(DmlError::Singular)
        );
    }
}

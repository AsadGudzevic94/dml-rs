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

/// Eigen-decomposition `A = V diag(λ) Vᵀ` of a small symmetric matrix by
/// cyclic Jacobi rotations; returns `(λ, V)` with eigenvectors in columns.
pub(crate) fn sym_eigen(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut m = a.clone();
    let mut v = Array2::<f64>::eye(n);
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[[i, j]] * m[[i, j]])
            .sum();
        let scale: f64 = m.iter().map(|x| x * x).sum();
        if off <= f64::EPSILON * f64::EPSILON * scale || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[[p, q]] == 0.0 {
                    continue;
                }
                // Rotation angle that zeroes m[p, q] (Golub & Van Loan 8.4).
                let theta = (m[[q, q]] - m[[p, p]]) / (2.0 * m[[p, q]]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[[k, p]], m[[k, q]]);
                    m[[k, p]] = c * mkp - s * mkq;
                    m[[k, q]] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[[p, k]], m[[q, k]]);
                    m[[p, k]] = c * mpk - s * mqk;
                    m[[q, k]] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[[k, p]], v[[k, q]]);
                    v[[k, p]] = c * vkp - s * vkq;
                    v[[k, q]] = s * vkp + c * vkq;
                }
            }
        }
    }
    (Array1::from_iter((0..n).map(|i| m[[i, i]])), v)
}

/// Moore–Penrose pseudo-inverse of a symmetric matrix, dropping eigenvalues
/// below `1e-15 · max|λ|` (NumPy's `pinv` default cutoff).
pub(crate) fn sym_pinv(a: &Array2<f64>) -> Array2<f64> {
    let (lambda, v) = sym_eigen(a);
    let cutoff = 1e-15 * lambda.iter().fold(0.0_f64, |m, l| m.max(l.abs()));
    let n = a.nrows();
    let mut out = Array2::<f64>::zeros((n, n));
    for (k, &l) in lambda.iter().enumerate() {
        if l.abs() > cutoff {
            for i in 0..n {
                for j in 0..n {
                    out[[i, j]] += v[[i, k]] * v[[j, k]] / l;
                }
            }
        }
    }
    out
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

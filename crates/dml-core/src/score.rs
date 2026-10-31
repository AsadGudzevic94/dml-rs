//! Linear Neyman-orthogonal scores and the DML2 estimator.

use ndarray::Array1;

use crate::error::{DmlError, Result};

/// Score elements of a model whose orthogonal score is linear in θ:
///
/// ```text
/// ψ(W; θ, η) = ψ_a(W; η) · θ + ψ_b(W; η)
/// ```
///
/// PLR, PLIV and IRM all have this form, so one solver serves every model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LinearScore {
    pub psi_a: Array1<f64>,
    pub psi_b: Array1<f64>,
}

/// θ̂ and its standard error for one treatment and one repetition.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct ScoreSolution {
    pub theta: f64,
    pub se: f64,
    /// ψ evaluated at θ̂.
    pub psi: Array1<f64>,
    /// `J = mean(ψ_a)`, the derivative of the mean score in θ.
    pub j: f64,
}

impl LinearScore {
    pub fn new(psi_a: Array1<f64>, psi_b: Array1<f64>) -> Result<Self> {
        if psi_a.len() != psi_b.len() || psi_a.is_empty() {
            return Err(DmlError::DimensionMismatch(
                "psi_a and psi_b must have the same non-zero length".into(),
            ));
        }
        Ok(Self { psi_a, psi_b })
    }

    /// DML2: solve `mean(ψ_a)·θ + mean(ψ_b) = 0` over all folds pooled, then
    /// `σ² = mean(ψ²) / J²` and `se = √(σ² / n)`.
    pub(crate) fn solve(&self) -> Result<ScoreSolution> {
        let n = self.psi_a.len() as f64;
        let j = self.psi_a.sum() / n;
        if j == 0.0 || !j.is_finite() {
            return Err(DmlError::InvalidArgument(
                "mean of psi_a is zero or not finite: theta is not identified".into(),
            ));
        }
        let theta = -(self.psi_b.sum() / n) / j;
        let psi = &self.psi_a * theta + &self.psi_b;
        let sigma2 = psi.dot(&psi) / n / (j * j);
        Ok(ScoreSolution {
            theta,
            se: (sigma2 / n).sqrt(),
            psi,
            j,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn solves_mean_score_equation() {
        let s = LinearScore::new(array![-1.0, -2.0, -3.0], array![2.0, 1.0, 3.0]).unwrap();
        let sol = s.solve().unwrap();
        assert!((sol.theta - 1.0).abs() < 1e-15);
        assert!(sol.psi.sum().abs() < 1e-12);
        assert_eq!(sol.j, -2.0);
        // ψ = [1, -1, 0]: σ² = (2/3) / 4, se = √(σ²/3)
        assert!((sol.se - (2.0_f64 / 3.0 / 4.0 / 3.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn unidentified_score_errors() {
        let s = LinearScore::new(array![0.0, 0.0], array![1.0, 2.0]).unwrap();
        assert!(s.solve().is_err());
        assert!(LinearScore::new(array![1.0], array![1.0, 2.0]).is_err());
    }
}

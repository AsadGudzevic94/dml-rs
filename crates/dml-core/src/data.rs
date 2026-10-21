use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{DmlError, Result};

/// Data for a single-treatment DML model.
///
/// - `y`: outcome, length `n`
/// - `d`: treatment, length `n` (continuous or 0/1)
/// - `x`: covariates / confounders, shape `(n, p)`
#[derive(Debug, Clone)]
pub struct DmlData {
    y: Array1<f64>,
    d: Array1<f64>,
    x: Array2<f64>,
}

impl DmlData {
    pub fn new(y: Array1<f64>, d: Array1<f64>, x: Array2<f64>) -> Result<Self> {
        let n = y.len();
        if n < 2 {
            return Err(DmlError::InvalidArgument(
                "need at least 2 observations".into(),
            ));
        }
        if d.len() != n || x.nrows() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "y has {n} rows, d has {}, x has {}",
                d.len(),
                x.nrows()
            )));
        }
        let finite = |v: f64| v.is_finite();
        if !y.iter().copied().all(finite)
            || !d.iter().copied().all(finite)
            || !x.iter().copied().all(finite)
        {
            return Err(DmlError::InvalidArgument(
                "data contains NaN or infinite values".into(),
            ));
        }
        Ok(Self { y, d, x })
    }

    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }

    pub fn d(&self) -> ArrayView1<'_, f64> {
        self.d.view()
    }

    pub fn x(&self) -> ArrayView2<'_, f64> {
        self.x.view()
    }

    /// True when every treatment value is exactly 0 or 1.
    pub fn is_binary_treatment(&self) -> bool {
        self.d.iter().all(|&v| v == 0.0 || v == 1.0)
    }

    /// Select rows by index (used when fitting on a training fold).
    pub fn x_rows(&self, idx: &[usize]) -> Array2<f64> {
        self.x.select(Axis(0), idx)
    }
}

/// Gather `v[idx]` into a new array.
pub(crate) fn gather(v: ArrayView1<'_, f64>, idx: &[usize]) -> Array1<f64> {
    idx.iter().map(|&i| v[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rejects_mismatched_lengths() {
        let err = DmlData::new(array![1.0, 2.0], array![0.0], array![[1.0], [2.0]]).unwrap_err();
        assert!(matches!(err, DmlError::DimensionMismatch(_)));
    }

    #[test]
    fn rejects_nan() {
        let err = DmlData::new(
            array![1.0, f64::NAN],
            array![0.0, 1.0],
            array![[1.0], [2.0]],
        )
        .unwrap_err();
        assert!(matches!(err, DmlError::InvalidArgument(_)));
    }

    #[test]
    fn detects_binary_treatment() {
        let data = DmlData::new(array![1.0, 2.0], array![0.0, 1.0], array![[1.0], [2.0]]).unwrap();
        assert!(data.is_binary_treatment());
    }
}

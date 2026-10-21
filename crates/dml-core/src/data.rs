use std::borrow::Cow;

use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{DmlError, Result};

/// Data for a DML model.
///
/// - `y`: outcome, length `n`
/// - `d`: one or more treatments, shape `(n, k)` (continuous or 0/1)
/// - `x`: covariates / confounders, shape `(n, p)`
/// - `z`: optional instrument, length `n` (for [`Pliv`](crate::Pliv))
///
/// With several treatments each one is estimated in turn, and the other
/// treatments are added to the covariates (as in DoubleML).
#[derive(Debug, Clone)]
pub struct DmlData {
    y: Array1<f64>,
    d: Array2<f64>,
    x: Array2<f64>,
    z: Option<Array1<f64>>,
    treatment_names: Vec<String>,
}

fn all_finite<'a>(mut values: impl Iterator<Item = &'a f64>) -> bool {
    values.all(|v| v.is_finite())
}

impl DmlData {
    /// Single treatment.
    pub fn new(y: Array1<f64>, d: Array1<f64>, x: Array2<f64>) -> Result<Self> {
        let n = d.len();
        let d = d.into_shape_with_order((n, 1)).expect("1-D to (n, 1)");
        Self::with_treatments(y, d, x)
    }

    /// Several treatments, one per column of `d`.
    pub fn with_treatments(y: Array1<f64>, d: Array2<f64>, x: Array2<f64>) -> Result<Self> {
        let n = y.len();
        if n < 2 {
            return Err(DmlError::InvalidArgument(
                "need at least 2 observations".into(),
            ));
        }
        if d.nrows() != n || x.nrows() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "y has {n} rows, d has {}, x has {}",
                d.nrows(),
                x.nrows()
            )));
        }
        if d.ncols() == 0 {
            return Err(DmlError::InvalidArgument(
                "need at least one treatment".into(),
            ));
        }
        if !all_finite(y.iter()) || !all_finite(d.iter()) || !all_finite(x.iter()) {
            return Err(DmlError::InvalidArgument(
                "data contains NaN or infinite values".into(),
            ));
        }
        let treatment_names = if d.ncols() == 1 {
            vec!["d".to_string()]
        } else {
            (0..d.ncols()).map(|j| format!("d{j}")).collect()
        };
        Ok(Self {
            y,
            d,
            x,
            z: None,
            treatment_names,
        })
    }

    /// Attach an instrument `z` (required by [`Pliv`](crate::Pliv)).
    pub fn with_instrument(mut self, z: Array1<f64>) -> Result<Self> {
        if z.len() != self.n_obs() {
            return Err(DmlError::DimensionMismatch(format!(
                "z has {} rows, expected {}",
                z.len(),
                self.n_obs()
            )));
        }
        if !all_finite(z.iter()) {
            return Err(DmlError::InvalidArgument(
                "instrument contains NaN or infinite values".into(),
            ));
        }
        self.z = Some(z);
        Ok(self)
    }

    /// Names shown in summaries (default `d`, or `d0`, `d1`, …).
    pub fn with_treatment_names<S: Into<String>>(
        mut self,
        names: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        if names.len() != self.n_treat() {
            return Err(DmlError::DimensionMismatch(format!(
                "{} names for {} treatments",
                names.len(),
                self.n_treat()
            )));
        }
        self.treatment_names = names;
        Ok(self)
    }

    pub fn n_obs(&self) -> usize {
        self.y.len()
    }

    pub fn n_features(&self) -> usize {
        self.x.ncols()
    }

    pub fn n_treat(&self) -> usize {
        self.d.ncols()
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }

    /// All treatments, shape `(n, k)`.
    pub fn d(&self) -> ArrayView2<'_, f64> {
        self.d.view()
    }

    /// Treatment `j`.
    pub fn treatment(&self, j: usize) -> ArrayView1<'_, f64> {
        self.d.column(j)
    }

    pub fn x(&self) -> ArrayView2<'_, f64> {
        self.x.view()
    }

    pub fn z(&self) -> Option<ArrayView1<'_, f64>> {
        self.z.as_ref().map(|z| z.view())
    }

    pub fn treatment_names(&self) -> &[String] {
        &self.treatment_names
    }

    /// Covariates used when estimating the effect of treatment `j`: `x` plus
    /// the other treatment columns. Borrowed when there is only one treatment.
    ///
    /// (`Cow` = "clone on write": either a borrowed view or an owned matrix.)
    pub fn covariates_for(&self, j: usize) -> Cow<'_, Array2<f64>> {
        if self.n_treat() == 1 {
            return Cow::Borrowed(&self.x);
        }
        let others: Vec<usize> = (0..self.n_treat()).filter(|&k| k != j).collect();
        let d_other = self.d.select(Axis(1), &others);
        Cow::Owned(concatenate![Axis(1), self.x, d_other])
    }

    /// True when every value of treatment `j` is exactly 0 or 1.
    pub fn is_binary_treatment(&self, j: usize) -> bool {
        self.d.column(j).iter().all(|&v| v == 0.0 || v == 1.0)
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
        assert!(data.is_binary_treatment(0));
    }

    #[test]
    fn other_treatments_become_covariates() {
        let d = array![[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]];
        let data = DmlData::with_treatments(array![0.0, 1.0], d, array![[5.0], [6.0]]).unwrap();
        assert_eq!(data.treatment_names(), ["d0", "d1", "d2"]);
        assert_eq!(
            *data.covariates_for(1),
            array![[5.0, 1.0, 100.0], [6.0, 2.0, 200.0]]
        );
        assert!(matches!(data.covariates_for(0), Cow::Owned(_)));
    }

    #[test]
    fn instrument_length_checked() {
        let data = DmlData::new(array![1.0, 2.0], array![0.0, 1.0], array![[1.0], [2.0]]).unwrap();
        assert!(data.clone().with_instrument(array![1.0]).is_err());
        assert!(data
            .with_instrument(array![1.0, 0.0])
            .unwrap()
            .z()
            .is_some());
    }
}

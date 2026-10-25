use std::borrow::Cow;

use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{DmlError, Result};

/// Data for a DML model.
///
/// - `y`: outcome, length `n`
/// - `d`: one or more treatments, shape `(n, k)` (continuous or 0/1)
/// - `x`: covariates / confounders, shape `(n, p)`
/// - `z`: optional instrument(s), length `n` (for [`Pliv`](crate::Pliv), which also takes several,
///   and, binary, [`Iivm`](crate::Iivm))
/// - `t`: optional 0/1 time-period indicator (for repeated cross-sections, [`DidCs`](crate::DidCs))
/// - `s`: optional 0/1 selection indicator, 1 where the outcome is observed (for [`Ssm`](crate::Ssm))
///
/// With several treatments each one is estimated in turn, and the other
/// treatments are added to the covariates (as in DoubleML).
#[derive(Debug, Clone)]
pub struct DmlData {
    y: Array1<f64>,
    d: Array2<f64>,
    x: Array2<f64>,
    z: Option<Array2<f64>>,
    t: Option<Array1<f64>>,
    s: Option<Array1<f64>>,
    external: Option<crate::model::ExternalPredictions>,
    clusters: Option<Vec<Array1<i64>>>,
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
            t: None,
            s: None,
            external: None,
            clusters: None,
            treatment_names,
        })
    }

    /// Attach an instrument `z` (required by [`Pliv`](crate::Pliv)).
    pub fn with_instrument(self, z: Array1<f64>) -> Result<Self> {
        let n = z.len();
        self.with_instruments(z.into_shape_with_order((n, 1)).expect("1-D to (n, 1)"))
    }

    /// Attach several instruments, one per column ([`Pliv`](crate::Pliv)
    /// combines them; the other instrumental models need exactly one).
    pub fn with_instruments(mut self, z: Array2<f64>) -> Result<Self> {
        if z.nrows() != self.n_obs() {
            return Err(DmlError::DimensionMismatch(format!(
                "z has {} rows, expected {}",
                z.nrows(),
                self.n_obs()
            )));
        }
        if z.ncols() == 0 {
            return Err(DmlError::InvalidArgument(
                "need at least one instrument".into(),
            ));
        }
        if !all_finite(z.iter()) {
            return Err(DmlError::InvalidArgument(
                "instrument contains NaN or infinite values".into(),
            ));
        }
        self.z = Some(z);
        Ok(self)
    }

    /// Attach a 0/1 time-period indicator (post-period = 1), required by
    /// [`DidCs`](crate::DidCs) for repeated cross-sections.
    pub fn with_time_indicator(mut self, t: Array1<f64>) -> Result<Self> {
        if t.len() != self.n_obs() {
            return Err(DmlError::DimensionMismatch(format!(
                "t has {} rows, expected {}",
                t.len(),
                self.n_obs()
            )));
        }
        if !is_binary(t.view()) {
            return Err(DmlError::InvalidArgument(
                "time indicator must contain only 0 and 1".into(),
            ));
        }
        self.t = Some(t);
        Ok(self)
    }

    /// Attach a 0/1 selection indicator (1 where the outcome is observed),
    /// required by the sample selection model [`Ssm`](crate::Ssm). The
    /// outcome of unselected observations is not used but must be finite
    /// (e.g. 0).
    pub fn with_selection_indicator(mut self, s: Array1<f64>) -> Result<Self> {
        if s.len() != self.n_obs() {
            return Err(DmlError::DimensionMismatch(format!(
                "s has {} rows, expected {}",
                s.len(),
                self.n_obs()
            )));
        }
        if !is_binary(s.view()) {
            return Err(DmlError::InvalidArgument(
                "selection indicator must contain only 0 and 1".into(),
            ));
        }
        self.s = Some(s);
        Ok(self)
    }

    /// Cluster the observations (e.g. by firm or school): sample splitting
    /// then keeps clusters together and standard errors are cluster-robust.
    pub fn with_clusters(self, cluster: Array1<i64>) -> Result<Self> {
        self.set_clusters(vec![cluster])
    }

    /// Two-way clustering (e.g. by firm and by year).
    pub fn with_two_way_clusters(self, first: Array1<i64>, second: Array1<i64>) -> Result<Self> {
        self.set_clusters(vec![first, second])
    }

    fn set_clusters(mut self, clusters: Vec<Array1<i64>>) -> Result<Self> {
        if clusters.iter().any(|c| c.len() != self.n_obs()) {
            return Err(DmlError::DimensionMismatch(
                "cluster variables need one entry per observation".into(),
            ));
        }
        self.clusters = Some(clusters);
        Ok(self)
    }

    /// The cluster variables (one or two), if any.
    pub fn clusters(&self) -> Option<&[Array1<i64>]> {
        self.clusters.as_deref()
    }

    /// The same data without the covariate columns `drop` (for sensitivity
    /// benchmarks).
    pub fn without_covariates(&self, drop: &[usize]) -> Result<Self> {
        let p = self.n_features();
        if drop.is_empty() || drop.iter().any(|&c| c >= p) {
            return Err(DmlError::InvalidArgument(format!(
                "benchmarking columns must be a non-empty subset of the {p} covariates"
            )));
        }
        let keep: Vec<usize> = (0..p).filter(|c| !drop.contains(c)).collect();
        // External predictions belong to the full covariate set.
        Ok(Self {
            x: self.x.select(Axis(1), &keep),
            external: None,
            ..self.clone()
        })
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

    /// The instrument, if exactly one is attached.
    pub fn z(&self) -> Option<ArrayView1<'_, f64>> {
        self.z
            .as_ref()
            .filter(|z| z.ncols() == 1)
            .map(|z| z.column(0))
    }

    /// All instruments, one per column.
    pub fn instruments(&self) -> Option<ArrayView2<'_, f64>> {
        self.z.as_ref().map(|z| z.view())
    }

    pub fn n_instruments(&self) -> usize {
        self.z.as_ref().map_or(0, |z| z.ncols())
    }

    /// The 0/1 time-period indicator, if attached.
    pub fn time(&self) -> Option<ArrayView1<'_, f64>> {
        self.t.as_ref().map(|t| t.view())
    }

    /// Use user-supplied out-of-fold predictions for some nuisance learners
    /// instead of fitting them (DoubleML's `external_predictions`).
    pub fn with_external_predictions(
        mut self,
        external: crate::model::ExternalPredictions,
    ) -> Self {
        self.external = Some(external);
        self
    }

    pub fn external_predictions(&self) -> Option<&crate::model::ExternalPredictions> {
        self.external.as_ref()
    }

    /// The 0/1 selection indicator, if attached.
    pub fn selection(&self) -> Option<ArrayView1<'_, f64>> {
        self.s.as_ref().map(|s| s.view())
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
        is_binary(self.d.column(j))
    }
}

/// True when every value is exactly 0 or 1.
pub(crate) fn is_binary(v: ArrayView1<'_, f64>) -> bool {
    v.iter().all(|&x| x == 0.0 || x == 1.0)
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
    fn several_instruments() {
        let data = DmlData::new(array![1.0, 2.0], array![0.0, 1.0], array![[1.0], [2.0]]).unwrap();
        let two = data
            .clone()
            .with_instruments(array![[1.0, 0.5], [0.0, 2.0]])
            .unwrap();
        assert_eq!(two.n_instruments(), 2);
        assert!(two.z().is_none());
        assert_eq!(two.instruments().unwrap().dim(), (2, 2));
        let one = data.with_instrument(array![1.0, 0.0]).unwrap();
        assert_eq!(one.z().unwrap().to_vec(), vec![1.0, 0.0]);
    }

    #[test]
    fn time_indicator_must_be_binary() {
        let data = DmlData::new(array![1.0, 2.0], array![0.0, 1.0], array![[1.0], [2.0]]).unwrap();
        assert!(data.clone().with_time_indicator(array![0.0, 2.0]).is_err());
        assert!(data.clone().with_time_indicator(array![1.0]).is_err());
        assert!(data
            .with_time_indicator(array![0.0, 1.0])
            .unwrap()
            .time()
            .is_some());
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

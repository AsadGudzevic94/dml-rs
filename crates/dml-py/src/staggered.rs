//! Staggered-adoption DID for Python.

use dml_core::{
    Aggregation, ControlGroup, GtCombinations, PanelData, StaggeredDid, StaggeredDidFit,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::learners::{to_classifier, to_regressor};
use crate::{did_score, PyDmlResult};

/// Fitted staggered DID: group-time effects and their aggregations.
#[pyclass(name = "StaggeredResult", module = "dml_rs", frozen)]
pub struct PyStaggeredResult {
    fit: StaggeredDidFit,
}

#[pymethods]
impl PyStaggeredResult {
    /// The ATT(g, t) estimates (coef, se, confint, bootstrap, ...).
    #[getter]
    fn att(&self) -> PyDmlResult {
        PyDmlResult::from_fit(self.fit.att().clone())
    }

    /// `(cohort, t_pre, t_eval)` for each ATT(g, t).
    #[getter]
    fn group_times(&self) -> Vec<(f64, f64, f64)> {
        self.fit
            .group_times()
            .iter()
            .map(|gt| (gt.cohort, gt.t_pre, gt.t_eval))
            .collect()
    }

    /// `"group"`, `"time"` or `"eventstudy"` →
    /// `(effects, overall, weights, overall_weights)`.
    #[allow(clippy::type_complexity)]
    fn aggregate<'py>(
        &self,
        py: Python<'py>,
        how: &str,
    ) -> PyResult<(
        PyDmlResult,
        PyDmlResult,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let how = match how {
            "group" => Aggregation::Group,
            "time" => Aggregation::Time,
            "eventstudy" => Aggregation::EventStudy,
            other => {
                return Err(PyValueError::new_err(format!(
                    "aggregation must be 'group', 'time' or 'eventstudy', got {other:?}"
                )))
            }
        };
        let agg = self.fit.aggregate(how).map_err(to_py_err)?;
        Ok((
            PyDmlResult::from_fit(agg.effects().clone()),
            PyDmlResult::from_fit(agg.overall().clone()),
            agg.weights().clone().into_pyarray(py),
            agg.overall_weights().clone().into_pyarray(py),
        ))
    }
}

/// Staggered DID from long-format data: one row per unit and period, with the
/// unit's first treated period `g` (`inf` or `nan` = never treated).
#[pyfunction]
#[pyo3(signature = (y, id, t, g, x, ml_g, ml_m = None, control_group = "never_treated",
                    anticipation_periods = 0, gt_combinations = "standard", score = "observational",
                    in_sample_normalization = true, clipping_threshold = 1e-2, n_folds = 5,
                    n_rep = 1, seed = 0))]
#[allow(clippy::too_many_arguments)]
pub fn fit_staggered_did(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    id: PyReadonlyArray1<'_, i64>,
    t: PyReadonlyArray1<'_, f64>,
    g: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: Option<&Bound<'_, PyAny>>,
    control_group: &str,
    anticipation_periods: usize,
    gt_combinations: &str,
    score: &str,
    in_sample_normalization: bool,
    clipping_threshold: f64,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
) -> PyResult<PyStaggeredResult> {
    let control_group = match control_group {
        "never_treated" => ControlGroup::NeverTreated,
        "not_yet_treated" => ControlGroup::NotYetTreated,
        other => {
            return Err(PyValueError::new_err(format!(
                "control_group must be 'never_treated' or 'not_yet_treated', got {other:?}"
            )))
        }
    };
    let setting = match gt_combinations {
        "standard" => GtCombinations::Standard,
        "all" => GtCombinations::All,
        "universal" => GtCombinations::Universal,
        other => {
            return Err(PyValueError::new_err(format!(
                "gt_combinations must be 'standard', 'all' or 'universal', got {other:?}"
            )))
        }
    };
    let (y, id, t, g) = (y.as_array(), id.as_array(), t.as_array(), g.as_array());
    let to_vec = |a: ndarray::ArrayView1<'_, f64>| a.iter().copied().collect::<Vec<f64>>();
    let ids: Vec<i64> = id.iter().copied().collect();
    let data = PanelData::from_long(
        &ids,
        &to_vec(t),
        &to_vec(y),
        &to_vec(g),
        &x.as_array().to_owned(),
    )
    .map_err(to_py_err)?;
    let model = StaggeredDid::from_boxed(to_regressor(ml_g)?, ml_m.map(to_classifier).transpose()?)
        .score(did_score(score)?)
        .control_group(control_group)
        .anticipation_periods(anticipation_periods)
        .gt_combinations(setting)
        .in_sample_normalization(in_sample_normalization)
        .clipping_threshold(clipping_threshold)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    let fit = py.detach(|| model.fit(&data)).map_err(to_py_err)?;
    Ok(PyStaggeredResult { fit })
}

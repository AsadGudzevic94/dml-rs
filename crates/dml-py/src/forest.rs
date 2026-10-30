//! Causal forests (`dml_rs.CausalForest`).

use dml_core::{CausalForestDml, CausalForestFit, ForestSettings, TreatmentLearner};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::cate::PyBlp;
use crate::errors::to_py_err;
use crate::learners::{to_classifier, to_regressor};
use crate::{build_data, PyDmlResult};

/// A fitted causal forest: effects with variances, out-of-bag estimates, and
/// doubly robust inference for a binary treatment.
#[pyclass(name = "CausalForestResult", module = "dml_rs", frozen)]
pub struct PyCausalForestResult {
    fit: CausalForestFit,
}

type Pair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

#[pymethods]
impl PyCausalForestResult {
    /// `(tau, variance)` at the rows of `x`.
    fn effect<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<Pair<'py>> {
        let x = x.as_array().to_owned();
        let (tau, var) = py.detach(|| self.fit.effect(x.view())).map_err(to_py_err)?;
        Ok((tau.into_pyarray(py), var.into_pyarray(py)))
    }

    #[getter]
    fn oob_effect<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.oob_effect().clone().into_pyarray(py)
    }

    #[getter]
    fn oob_variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.oob_variance().clone().into_pyarray(py)
    }

    #[getter]
    fn y_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.y_hat().clone().into_pyarray(py)
    }

    #[getter]
    fn d_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.d_hat().clone().into_pyarray(py)
    }

    #[getter]
    fn variable_importance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit
            .forest()
            .variable_importance()
            .clone()
            .into_pyarray(py)
    }

    #[getter]
    fn n_trees(&self) -> usize {
        self.fit.forest().n_trees()
    }

    #[getter]
    fn binary_treatment(&self) -> bool {
        self.fit.is_binary_treatment()
    }

    /// Doubly robust scores (binary treatment).
    fn dr_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.fit.dr_scores().map_err(to_py_err)?.into_pyarray(py))
    }

    /// Average treatment effect from the doubly robust scores.
    fn ate(&self) -> PyResult<PyDmlResult> {
        self.fit.ate().map(PyDmlResult::from_fit).map_err(to_py_err)
    }

    fn blp(&self, basis: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.blp(basis.as_array()).map_err(to_py_err)?,
        })
    }

    fn gate(&self, groups: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.gate(groups.as_array()).map_err(to_py_err)?,
        })
    }
}

/// Causal forest on cross-fitted residuals (grf's `causal_forest` with any
/// nuisance learners).
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_y, ml_d, num_trees = 2000, sample_fraction = 0.5, mtry = None,
                    min_node_size = 5, honesty = true, honesty_fraction = 0.5, prune_empty_leaves = true,
                    alpha = 0.05, imbalance_penalty = 0.0, stabilize_splits = true, ci_group_size = 2,
                    forest_seed = 0, n_folds = 5, seed = 0, clipping_threshold = 1e-2, treatment_names = None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_causal_forest(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_y: &Bound<'_, PyAny>,
    ml_d: &Bound<'_, PyAny>,
    num_trees: usize,
    sample_fraction: f64,
    mtry: Option<usize>,
    min_node_size: usize,
    honesty: bool,
    honesty_fraction: f64,
    prune_empty_leaves: bool,
    alpha: f64,
    imbalance_penalty: f64,
    stabilize_splits: bool,
    ci_group_size: usize,
    forest_seed: u64,
    n_folds: usize,
    seed: u64,
    clipping_threshold: f64,
    treatment_names: Option<Vec<String>>,
) -> PyResult<PyCausalForestResult> {
    let data = build_data(y, d, x, None, None, treatment_names, None)?;
    let ml_d = if ml_d.hasattr("predict_proba")? {
        TreatmentLearner::Classifier(to_classifier(ml_d)?)
    } else {
        TreatmentLearner::Regressor(to_regressor(ml_d)?)
    };
    let model = CausalForestDml::from_boxed(to_regressor(ml_y)?, ml_d)
        .forest(ForestSettings {
            num_trees,
            sample_fraction,
            mtry,
            min_node_size,
            honesty,
            honesty_fraction,
            prune_empty_leaves,
            alpha,
            imbalance_penalty,
            stabilize_splits,
            ci_group_size,
            seed: forest_seed,
        })
        .n_folds(n_folds)
        .seed(seed)
        .clipping_threshold(clipping_threshold);
    let fit = py.detach(|| model.fit(&data)).map_err(to_py_err)?;
    Ok(PyCausalForestResult { fit })
}

//! Generalized random forests (`dml_rs.CausalForest`, `InstrumentalForest`,
//! `MultiArmCausalForest`, `RegressionForest`, `QuantileForest`).

use dml_core::learner::Regressor;
use dml_core::{
    CausalForestDml, CausalForestFit, DmlData, ForestSettings, InstrumentalForestDml,
    InstrumentalForestFit, MultiArmCausalForestDml, MultiArmCausalForestFit, QuantileForest,
    RegressionForest, TreatmentLearner,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::cate::PyBlp;
use crate::errors::to_py_err;
use crate::learners::{to_classifier, to_regressor};
use crate::{build_data, PyDmlResult, PyPolicyTree};

type Pair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type Pair2<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// A classifier if the learner has `predict_proba`, otherwise a regressor.
fn treatment_learner(obj: &Bound<'_, PyAny>) -> PyResult<TreatmentLearner> {
    Ok(if obj.hasattr("predict_proba")? {
        TreatmentLearner::Classifier(to_classifier(obj)?)
    } else {
        TreatmentLearner::Regressor(to_regressor(obj)?)
    })
}

/// The forest arguments every Python entry point takes, as [`ForestSettings`].
#[allow(clippy::too_many_arguments)]
fn settings(
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
    seed: u64,
) -> ForestSettings {
    ForestSettings {
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
        seed,
    }
}

// ---------------------------------------------------------------------------
// Causal forest
// ---------------------------------------------------------------------------

/// A fitted causal forest: effects with variances, out-of-bag estimates, and
/// doubly robust inference for a binary treatment.
#[pyclass(name = "CausalForestResult", module = "dml_rs", frozen)]
pub struct PyCausalForestResult {
    fit: CausalForestFit,
}

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

    /// Policy tree on the doubly robust scores.
    fn policy_tree(
        &self,
        features: PyReadonlyArray2<'_, f64>,
        depth: usize,
        min_samples_leaf: usize,
        ccp_alpha: f64,
    ) -> PyResult<PyPolicyTree> {
        self.fit
            .policy_tree(features.as_array(), depth, min_samples_leaf, ccp_alpha)
            .map(|tree| PyPolicyTree { tree })
            .map_err(to_py_err)
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
    let model = CausalForestDml::from_boxed(to_regressor(ml_y)?, treatment_learner(ml_d)?)
        .forest(settings(
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
            forest_seed,
        ))
        .n_folds(n_folds)
        .seed(seed)
        .clipping_threshold(clipping_threshold);
    let fit = py.detach(|| model.fit(&data)).map_err(to_py_err)?;
    Ok(PyCausalForestResult { fit })
}

// ---------------------------------------------------------------------------
// Instrumental forest
// ---------------------------------------------------------------------------

/// A fitted instrumental forest.
#[pyclass(name = "InstrumentalForestResult", module = "dml_rs", frozen)]
pub struct PyInstrumentalForestResult {
    fit: InstrumentalForestFit,
}

#[pymethods]
impl PyInstrumentalForestResult {
    /// `(theta, variance)` at the rows of `x`.
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
    fn z_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.z_hat().clone().into_pyarray(py)
    }

    /// Out-of-bag compliance scores (binary instrument), else `None`.
    #[getter]
    fn compliance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.fit.compliance().map(|c| c.clone().into_pyarray(py))
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

    fn dr_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.fit.dr_scores().map_err(to_py_err)?.into_pyarray(py))
    }

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

/// Instrumental forest on cross-fitted residuals (grf's `instrumental_forest`).
#[pyfunction]
#[pyo3(signature = (y, d, z, x, ml_y, ml_d, ml_z, num_trees = 2000, sample_fraction = 0.5, mtry = None,
                    min_node_size = 5, honesty = true, honesty_fraction = 0.5, prune_empty_leaves = true,
                    alpha = 0.05, imbalance_penalty = 0.0, stabilize_splits = true, ci_group_size = 2,
                    reduced_form_weight = 0.0, forest_seed = 0, n_folds = 5, seed = 0,
                    clipping_threshold = 1e-2, treatment_names = None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_instrumental_forest(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_y: &Bound<'_, PyAny>,
    ml_d: &Bound<'_, PyAny>,
    ml_z: &Bound<'_, PyAny>,
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
    reduced_form_weight: f64,
    forest_seed: u64,
    n_folds: usize,
    seed: u64,
    clipping_threshold: f64,
    treatment_names: Option<Vec<String>>,
) -> PyResult<PyInstrumentalForestResult> {
    let data = build_data(y, d, x, Some(z), None, treatment_names, None)?;
    let model = InstrumentalForestDml::from_boxed(
        to_regressor(ml_y)?,
        treatment_learner(ml_d)?,
        treatment_learner(ml_z)?,
    )
    .forest(settings(
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
        forest_seed,
    ))
    .reduced_form_weight(reduced_form_weight)
    .n_folds(n_folds)
    .seed(seed)
    .clipping_threshold(clipping_threshold);
    let fit = py.detach(|| model.fit(&data)).map_err(to_py_err)?;
    Ok(PyInstrumentalForestResult { fit })
}

// ---------------------------------------------------------------------------
// Multi-arm causal forest
// ---------------------------------------------------------------------------

/// A fitted multi-arm causal forest (effects of each non-baseline level).
#[pyclass(name = "MultiArmCausalForestResult", module = "dml_rs", frozen)]
pub struct PyMultiArmCausalForestResult {
    fit: MultiArmCausalForestFit,
}

#[pymethods]
impl PyMultiArmCausalForestResult {
    /// `(tau, variance)` at the rows of `x`, each `(n, k)`.
    fn effect<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>) -> PyResult<Pair2<'py>> {
        let x = x.as_array().to_owned();
        let (tau, var) = py.detach(|| self.fit.effect(x.view())).map_err(to_py_err)?;
        Ok((tau.into_pyarray(py), var.into_pyarray(py)))
    }

    #[getter]
    fn oob_effect<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.oob_effect().clone().into_pyarray(py)
    }

    #[getter]
    fn oob_variance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.oob_variance().clone().into_pyarray(py)
    }

    #[getter]
    fn y_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.y_hat().clone().into_pyarray(py)
    }

    /// Propensities of every level, `(n, k + 1)`.
    #[getter]
    fn propensity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.propensity().clone().into_pyarray(py)
    }

    #[getter]
    fn levels(&self) -> Vec<f64> {
        self.fit.levels().to_vec()
    }

    #[getter]
    fn arm_names(&self) -> Vec<String> {
        self.fit.arm_names().to_vec()
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

    fn dr_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.fit.dr_scores().map_err(to_py_err)?.into_pyarray(py))
    }

    fn ate(&self) -> PyResult<PyDmlResult> {
        self.fit.ate().map(PyDmlResult::from_fit).map_err(to_py_err)
    }

    fn blp(&self, arm: usize, basis: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.blp(arm, basis.as_array()).map_err(to_py_err)?,
        })
    }

    fn gate(&self, arm: usize, groups: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.gate(arm, groups.as_array()).map_err(to_py_err)?,
        })
    }
}

/// Multi-arm causal forest on cross-fitted residuals (grf's
/// `multi_arm_causal_forest`).
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_y, ml_d, num_trees = 2000, sample_fraction = 0.5, mtry = None,
                    min_node_size = 5, honesty = true, honesty_fraction = 0.5, prune_empty_leaves = true,
                    alpha = 0.05, imbalance_penalty = 0.0, stabilize_splits = true, ci_group_size = 2,
                    forest_seed = 0, n_folds = 5, seed = 0, clipping_threshold = 1e-2, treatment_name = None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_multi_arm_causal_forest(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, f64>,
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
    treatment_name: Option<String>,
) -> PyResult<PyMultiArmCausalForestResult> {
    let mut data = DmlData::new(
        y.as_array().to_owned(),
        d.as_array().to_owned(),
        x.as_array().to_owned(),
    )
    .map_err(to_py_err)?;
    if let Some(name) = treatment_name {
        data = data.with_treatment_names(vec![name]).map_err(to_py_err)?;
    }
    let model = MultiArmCausalForestDml::from_boxed(to_regressor(ml_y)?, to_classifier(ml_d)?)
        .forest(settings(
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
            forest_seed,
        ))
        .n_folds(n_folds)
        .seed(seed)
        .clipping_threshold(clipping_threshold);
    let fit = py.detach(|| model.fit(&data)).map_err(to_py_err)?;
    Ok(PyMultiArmCausalForestResult { fit })
}

// ---------------------------------------------------------------------------
// Regression and quantile forests (learners)
// ---------------------------------------------------------------------------

/// Honest regression forest (grf's `regression_forest`), implemented in Rust:
/// a nuisance learner with variance estimates and out-of-bag predictions.
#[pyclass(name = "RegressionForest", module = "dml_rs")]
#[derive(Clone)]
pub struct PyRegressionForest {
    pub(crate) inner: RegressionForest,
    repr: String,
}

#[pymethods]
impl PyRegressionForest {
    #[new]
    #[pyo3(signature = (n_estimators = 2000, sample_fraction = 0.5, mtry = None, min_node_size = 5,
                        honesty = true, honesty_fraction = 0.5, honesty_prune_leaves = true, alpha = 0.05,
                        imbalance_penalty = 0.0, ci_group_size = 2, random_state = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_estimators: usize,
        sample_fraction: f64,
        mtry: Option<usize>,
        min_node_size: usize,
        honesty: bool,
        honesty_fraction: f64,
        honesty_prune_leaves: bool,
        alpha: f64,
        imbalance_penalty: f64,
        ci_group_size: usize,
        random_state: u64,
    ) -> Self {
        Self {
            inner: RegressionForest::new().settings(settings(
                n_estimators,
                sample_fraction,
                mtry,
                min_node_size,
                honesty,
                honesty_fraction,
                honesty_prune_leaves,
                alpha,
                imbalance_penalty,
                false,
                ci_group_size,
                random_state,
            )),
            repr: format!(
                "RegressionForest(n_estimators={n_estimators}, min_node_size={min_node_size}, random_state={random_state})"
            ),
        }
    }

    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (x, y) = (x.as_array().to_owned(), y.as_array().to_owned());
        let mut model = slf.inner.clone();
        slf.py()
            .detach(|| model.fit(x.view(), y.view()))
            .map_err(to_py_err)?;
        slf.inner = model;
        Ok(slf)
    }

    /// Predicted values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x = x.as_array().to_owned();
        Ok(py
            .detach(|| self.inner.predict(x.view()))
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    /// `(prediction, variance)` (bootstrap of little bags).
    fn predict_with_variance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Pair<'py>> {
        let x = x.as_array().to_owned();
        let (p, v) = py
            .detach(|| self.inner.predict_with_variance(x.view()))
            .map_err(to_py_err)?;
        Ok((p.into_pyarray(py), v.into_pyarray(py)))
    }

    /// Out-of-bag `(prediction, variance)` for the training rows.
    fn predict_oob<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Pair<'py>> {
        let x = x.as_array().to_owned();
        let (p, v) = py
            .detach(|| self.inner.predict_oob(x.view()))
            .map_err(to_py_err)?;
        Ok((p.into_pyarray(py), v.into_pyarray(py)))
    }

    /// grf's split-frequency variable importance, after `fit`.
    #[getter]
    fn variable_importance_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .variable_importance()
            .map(|a| a.clone().into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

/// Quantile forest (grf's `quantile_forest`), implemented in Rust.
#[pyclass(name = "QuantileForest", module = "dml_rs")]
#[derive(Clone)]
pub struct PyQuantileForest {
    inner: QuantileForest,
    quantiles: Vec<f64>,
    repr: String,
}

#[pymethods]
impl PyQuantileForest {
    #[new]
    #[pyo3(signature = (quantiles = vec![0.1, 0.5, 0.9], n_estimators = 2000, sample_fraction = 0.5,
                        mtry = None, min_node_size = 5, honesty = true, honesty_fraction = 0.5,
                        honesty_prune_leaves = true, alpha = 0.05, imbalance_penalty = 0.0, random_state = 0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        quantiles: Vec<f64>,
        n_estimators: usize,
        sample_fraction: f64,
        mtry: Option<usize>,
        min_node_size: usize,
        honesty: bool,
        honesty_fraction: f64,
        honesty_prune_leaves: bool,
        alpha: f64,
        imbalance_penalty: f64,
        random_state: u64,
    ) -> Self {
        let repr = format!(
            "QuantileForest(quantiles={quantiles:?}, n_estimators={n_estimators}, random_state={random_state})"
        );
        Self {
            inner: QuantileForest::new(quantiles.clone()).settings(settings(
                n_estimators,
                sample_fraction,
                mtry,
                min_node_size,
                honesty,
                honesty_fraction,
                honesty_prune_leaves,
                alpha,
                imbalance_penalty,
                false,
                1,
                random_state,
            )),
            quantiles,
            repr,
        }
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let (x, y) = (x.as_array().to_owned(), y.as_array().to_owned());
        let mut model = slf.inner.clone();
        slf.py()
            .detach(|| model.fit(x.view(), y.view()))
            .map_err(to_py_err)?;
        slf.inner = model;
        Ok(slf)
    }

    /// Conditional quantiles, `(n, len(quantiles))` (default: the fitted ones).
    #[pyo3(signature = (x, quantiles = None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        quantiles: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = x.as_array().to_owned();
        let q = quantiles.unwrap_or_else(|| self.quantiles.clone());
        Ok(py
            .detach(|| self.inner.predict(x.view(), &q))
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    /// Out-of-bag quantiles for the training rows.
    #[pyo3(signature = (x, quantiles = None))]
    fn predict_oob<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        quantiles: Option<Vec<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = x.as_array().to_owned();
        let q = quantiles.unwrap_or_else(|| self.quantiles.clone());
        Ok(py
            .detach(|| self.inner.predict_oob(x.view(), &q))
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        self.quantiles.clone()
    }

    #[getter]
    fn variable_importance_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .variable_importance()
            .map(|a| a.clone().into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

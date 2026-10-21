//! Learners exposed to Python, and adapters that let any scikit-learn style
//! estimator act as a Rust [`Regressor`] / [`Classifier`].

use dml_core::{Classifier, DmlError, Lasso, LogisticRegression, Ols, Regressor};
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::errors::to_py_err;

// ---------------------------------------------------------------------------
// Built-in Rust learners. Cross-fitting with these never touches Python, so
// it runs fully in parallel with the GIL released.
// ---------------------------------------------------------------------------

/// Ordinary least squares with intercept and optional ridge penalty
/// (`ridge=α` matches scikit-learn's `Ridge(alpha=α)`).
#[pyclass(name = "Ols", module = "dml_rs")]
#[derive(Clone)]
pub struct PyOls {
    pub(crate) inner: Ols,
    ridge: f64,
}

#[pymethods]
impl PyOls {
    #[new]
    #[pyo3(signature = (ridge = 0.0))]
    fn new(ridge: f64) -> Self {
        Self {
            inner: Ols::with_ridge(ridge),
            ridge,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .fit(x.as_array(), y.as_array())
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .predict(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Ols(ridge={})", self.ridge)
    }
}

/// Lasso by coordinate descent; same objective as scikit-learn's `Lasso(alpha=α)`.
#[pyclass(name = "Lasso", module = "dml_rs")]
#[derive(Clone)]
pub struct PyLasso {
    pub(crate) inner: Lasso,
    alpha: f64,
}

#[pymethods]
impl PyLasso {
    #[new]
    #[pyo3(signature = (alpha = 1.0, tol = 1e-12, max_iter = 100_000))]
    fn new(alpha: f64, tol: f64, max_iter: usize) -> Self {
        Self {
            inner: Lasso::new(alpha).tol(tol).max_iter(max_iter),
            alpha,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .fit(x.as_array(), y.as_array())
            .map_err(to_py_err)?;
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .predict(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Lasso(alpha={})", self.alpha)
    }
}

/// L2-penalised logistic regression (Newton); same objective as
/// scikit-learn's `LogisticRegression(C=C)`.
#[pyclass(name = "LogisticRegression", module = "dml_rs")]
#[derive(Clone)]
pub struct PyLogisticRegression {
    pub(crate) inner: LogisticRegression,
    c: f64,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
    #[pyo3(signature = (C = 1.0, tol = 1e-10, max_iter = 100))]
    #[allow(non_snake_case)] // keep scikit-learn's parameter name
    fn new(C: f64, tol: f64, max_iter: usize) -> Self {
        Self {
            inner: LogisticRegression::new(C).tol(tol).max_iter(max_iter),
            c: C,
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .fit(x.as_array(), y.as_array())
            .map_err(to_py_err)?;
        Ok(slf)
    }

    /// Probability of class 1 (a 1-D array, unlike scikit-learn's two columns).
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .inner
            .predict_proba(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("LogisticRegression(C={})", self.c)
    }
}

// ---------------------------------------------------------------------------
// Adapters for arbitrary Python estimators.
// ---------------------------------------------------------------------------

/// `sklearn.base.clone(est)` when scikit-learn is installed, else `copy.deepcopy`.
fn clone_estimator<'py>(est: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = est.py();
    match py.import("sklearn.base") {
        Ok(base) => base.getattr("clone")?.call1((est,)),
        Err(_) => py.import("copy")?.getattr("deepcopy")?.call1((est,)),
    }
}

fn as_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Array1<f64>> {
    let np = obj.py().import("numpy")?;
    let arr = np
        .getattr("ascontiguousarray")?
        .call1((obj, "float64"))?
        .call_method0("ravel")?;
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    Ok(arr.as_array().to_owned())
}

fn learner_err(err: PyErr) -> DmlError {
    DmlError::Learner(err.to_string())
}

/// A Python object with `fit(X, y)` / `predict(X)`.
///
/// Every call attaches to the interpreter (`Python::attach`), so it is safe
/// to use from the rayon worker threads that fit the folds.
pub struct PyRegressor {
    template: Py<PyAny>,
    fitted: Option<Py<PyAny>>,
}

impl PyRegressor {
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        for method in ["fit", "predict"] {
            if !obj.hasattr(method)? {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "regressor {} has no `{method}` method",
                    obj.repr()?
                )));
            }
        }
        Ok(Self {
            template: obj.clone().unbind(),
            fitted: None,
        })
    }
}

impl Regressor for PyRegressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> dml_core::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            let est = clone_estimator(self.template.bind(py))?;
            est.call_method1(
                "fit",
                (x.to_owned().into_pyarray(py), y.to_owned().into_pyarray(py)),
            )?;
            self.fitted = Some(est.unbind());
            Ok(())
        })
        .map_err(learner_err)
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> dml_core::Result<Array1<f64>> {
        let est = self.fitted.as_ref().ok_or(DmlError::NotFitted)?;
        Python::attach(|py| {
            let pred = est
                .bind(py)
                .call_method1("predict", (x.to_owned().into_pyarray(py),))?;
            as_f64_vec(&pred)
        })
        .map_err(learner_err)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Python::attach(|py| {
            Box::new(PyRegressor {
                template: self.template.clone_ref(py),
                fitted: None,
            })
        })
    }
}

/// A Python object with `fit(X, y)` / `predict_proba(X)` (column 1 is used).
pub struct PyClassifier {
    template: Py<PyAny>,
    fitted: Option<Py<PyAny>>,
}

impl PyClassifier {
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        for method in ["fit", "predict_proba"] {
            if !obj.hasattr(method)? {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "classifier {} has no `{method}` method",
                    obj.repr()?
                )));
            }
        }
        Ok(Self {
            template: obj.clone().unbind(),
            fitted: None,
        })
    }
}

impl Classifier for PyClassifier {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> dml_core::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            let est = clone_estimator(self.template.bind(py))?;
            est.call_method1(
                "fit",
                (x.to_owned().into_pyarray(py), y.to_owned().into_pyarray(py)),
            )?;
            self.fitted = Some(est.unbind());
            Ok(())
        })
        .map_err(learner_err)
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> dml_core::Result<Array1<f64>> {
        let est = self.fitted.as_ref().ok_or(DmlError::NotFitted)?;
        Python::attach(|py| -> PyResult<Array1<f64>> {
            let proba = est
                .bind(py)
                .call_method1("predict_proba", (x.to_owned().into_pyarray(py),))?;
            let np = py.import("numpy")?;
            let proba = np.getattr("asarray")?.call1((proba, "float64"))?;
            let proba: PyReadonlyArray2<'_, f64> = proba.extract()?;
            let proba = proba.as_array();
            if proba.ncols() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "predict_proba must return two columns (binary classification)",
                ));
            }
            Ok(proba.column(1).to_owned())
        })
        .map_err(learner_err)
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Python::attach(|py| {
            Box::new(PyClassifier {
                template: self.template.clone_ref(py),
                fitted: None,
            })
        })
    }
}

/// Built-in learner if `obj` is one, otherwise wrap the Python estimator.
pub fn to_regressor(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Regressor>> {
    if let Ok(o) = obj.cast::<PyOls>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyLasso>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    Ok(Box::new(PyRegressor::new(obj)?))
}

pub fn to_classifier(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Classifier>> {
    if let Ok(o) = obj.cast::<PyLogisticRegression>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    Ok(Box::new(PyClassifier::new(obj)?))
}

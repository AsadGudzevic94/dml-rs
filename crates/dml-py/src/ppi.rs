//! Prediction-powered inference for Python.

use dml_core::{PpiData, PpiEstimand, PpiFit, PpiLambda, PredictionPowered};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::learners::{to_classifier, to_regressor};
use crate::PyDmlResult;

/// Result of `fit_ppi`: the prediction-powered and the classical estimate.
#[pyclass(name = "PpiResult", module = "dml_rs", frozen)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PyPpiResult {
    fit: PpiFit,
    /// Cross-fitted predictions (labelled, unlabelled); `None` when they
    /// were passed in.
    predictions: Option<(Vec<f64>, Vec<f64>)>,
}

#[pymethods]
impl PyPpiResult {
    /// Pickle support (`pickle`, `copy.deepcopy`, `joblib`).
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::pickle::Reduced<'py>> {
        crate::pickle::reduce(py, "PpiResult", self)
    }

    #[getter]
    fn estimate(&self) -> PyDmlResult {
        PyDmlResult::from_fit(self.fit.estimate().clone())
    }

    #[getter]
    fn classical(&self) -> PyDmlResult {
        PyDmlResult::from_fit(self.fit.classical().clone())
    }

    #[getter]
    fn lam(&self) -> f64 {
        self.fit.lambda()
    }

    #[getter]
    fn n_labelled(&self) -> usize {
        self.fit.n_labelled()
    }

    #[getter]
    fn n_unlabelled(&self) -> usize {
        self.fit.n_unlabelled()
    }

    #[getter]
    fn effective_sample_size<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.effective_sample_size().into_pyarray(py)
    }

    /// Cross-fitted predictions on the labelled (out-of-fold) and the
    /// unlabelled sample (fold-averaged); `None` when they were passed in.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn predictions<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        self.predictions
            .as_ref()
            .map(|(l, u)| (PyArray1::from_slice(py, l), PyArray1::from_slice(py, u)))
    }
}

/// Prediction-powered inference. Predictions come either as `yhat` /
/// `yhat_unlabelled` or from cross-fitting `learner` on `features` (a
/// classifier's probabilities when `classifier`).
#[pyfunction]
#[pyo3(signature = (
    estimand, y, yhat = None, yhat_unlabelled = None, x = None, x_unlabelled = None, names = None,
    lam = None, learner = None, classifier = false, features = None, features_unlabelled = None,
    n_folds = 5, seed = 0
))]
#[allow(clippy::too_many_arguments)]
pub fn fit_ppi(
    py: Python<'_>,
    estimand: &str,
    y: PyReadonlyArray1<'_, f64>,
    yhat: Option<PyReadonlyArray1<'_, f64>>,
    yhat_unlabelled: Option<PyReadonlyArray1<'_, f64>>,
    x: Option<PyReadonlyArray2<'_, f64>>,
    x_unlabelled: Option<PyReadonlyArray2<'_, f64>>,
    names: Option<Vec<String>>,
    lam: Option<f64>,
    learner: Option<Bound<'_, PyAny>>,
    classifier: bool,
    features: Option<PyReadonlyArray2<'_, f64>>,
    features_unlabelled: Option<PyReadonlyArray2<'_, f64>>,
    n_folds: usize,
    seed: u64,
) -> PyResult<PyPpiResult> {
    let estimand = match estimand {
        "mean" => PpiEstimand::Mean,
        "ols" => PpiEstimand::LinearRegression,
        "logistic" => PpiEstimand::LogisticRegression,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown estimand {other:?}: use mean, ols or logistic"
            )))
        }
    };
    let y = y.as_array().to_owned();
    let cross_fitted = learner.is_some();
    let mut data = match (learner, yhat, yhat_unlabelled) {
        (Some(learner), None, None) => {
            let (Some(f), Some(fu)) = (features, features_unlabelled) else {
                return Err(PyValueError::new_err(
                    "cross-fitting needs features and features_unlabelled",
                ));
            };
            let (f, fu) = (f.as_array(), fu.as_array());
            if classifier {
                let learner = to_classifier(&learner)?;
                py.detach(|| PpiData::cross_fit_proba(learner.as_ref(), y, f, fu, n_folds, seed))
            } else {
                let learner = to_regressor(&learner)?;
                py.detach(|| PpiData::cross_fit(learner.as_ref(), y, f, fu, n_folds, seed))
            }
        }
        (None, Some(yhat), Some(yhat_u)) => {
            PpiData::new(y, yhat.as_array().to_owned(), yhat_u.as_array().to_owned())
        }
        _ => {
            return Err(PyValueError::new_err(
                "pass either yhat and yhat_unlabelled or a learner",
            ))
        }
    }
    .map_err(to_py_err)?;
    match (x, x_unlabelled) {
        (Some(x), Some(xu)) => {
            data = data
                .with_covariates(x.as_array().to_owned(), xu.as_array().to_owned())
                .map_err(to_py_err)?;
        }
        (None, None) => {}
        _ => {
            return Err(PyValueError::new_err(
                "pass covariates for both samples or for neither",
            ))
        }
    }
    if let Some(names) = names {
        data = data.with_covariate_names(names);
    }
    let lambda = lam.map_or(PpiLambda::Tuned, PpiLambda::Fixed);
    let model = PredictionPowered::new(estimand).lambda(lambda);
    let fit = py.detach(|| model.fit(&data)).map_err(to_py_err)?;
    let predictions = cross_fitted.then(|| (data.yhat().to_vec(), data.yhat_unlabelled().to_vec()));
    Ok(PyPpiResult { fit, predictions })
}

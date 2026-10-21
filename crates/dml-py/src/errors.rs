use dml_core::DmlError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyErr;

/// Learner failures (including exceptions raised by Python estimators) become
/// `RuntimeError`; invalid data or settings become `ValueError`.
pub fn to_py_err(err: DmlError) -> PyErr {
    match err {
        DmlError::Learner(_) | DmlError::Convergence(_) => PyRuntimeError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
    }
}

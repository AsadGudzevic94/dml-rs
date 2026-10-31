//! Pickling of fitted learners and results (`pickle`, `copy`, `joblib`).
//!
//! `__reduce__` serializes the Rust value with bincode and hands it to
//! `dml_rs._pickle.restore(kind, data)`, which calls [`restore`] here. The
//! format is tied to the dml-rs version, like scikit-learn's pickles.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{de::DeserializeOwned, Serialize};

/// Prefix of every pickled payload: format tag and the dml-rs version.
fn header() -> Vec<u8> {
    format!("dml-rs/{}\n", env!("CARGO_PKG_VERSION")).into_bytes()
}

/// What `__reduce__` returns: the restore function and its arguments.
pub type Reduced<'py> = (Bound<'py, PyAny>, (&'static str, Bound<'py, PyBytes>));

pub fn reduce<'py, T: Serialize>(
    py: Python<'py>,
    kind: &'static str,
    value: &T,
) -> PyResult<Reduced<'py>> {
    let mut data = header();
    bincode::serialize_into(&mut data, value)
        .map_err(|e| PyValueError::new_err(format!("cannot pickle {kind}: {e}")))?;
    let restore = py.import("dml_rs._pickle")?.getattr("restore")?;
    Ok((restore, (kind, PyBytes::new(py, &data))))
}

fn load<T: DeserializeOwned>(kind: &str, data: &[u8]) -> PyResult<T> {
    let header = header();
    if !data.starts_with(&header) {
        let found = data.split(|&b| b == b'\n').next().unwrap_or_default();
        return Err(PyValueError::new_err(format!(
            "this {kind} was pickled by {}; it can only be loaded with the same dml-rs version ({})",
            String::from_utf8_lossy(found),
            String::from_utf8_lossy(&header[..header.len() - 1]),
        )));
    }
    bincode::deserialize(&data[header.len()..])
        .map_err(|e| PyValueError::new_err(format!("cannot unpickle {kind}: {e}")))
}

/// `restore(kind, data)`: rebuild the object `__reduce__` pickled.
macro_rules! restorers {
    ($($kind:literal => $ty:ty),* $(,)?) => {
        #[pyfunction(name = "_restore")]
        pub fn restore(py: Python<'_>, kind: &str, data: &[u8]) -> PyResult<Py<PyAny>> {
            match kind {
                $($kind => Ok(Py::new(py, load::<$ty>(kind, data)?)?.into_any()),)*
                other => Err(PyValueError::new_err(format!("unknown pickled dml-rs type {other:?}"))),
            }
        }
    };
}

restorers! {
    "Ols" => crate::learners::PyOls,
    "Lasso" => crate::learners::PyLasso,
    "LassoCV" => crate::learners::PyLassoCV,
    "LogisticRegression" => crate::learners::PyLogisticRegression,
    "LogisticRegressionCV" => crate::learners::PyLogisticRegressionCV,
    "RandomForestRegressor" => crate::learners::PyRandomForestRegressor,
    "RandomForestClassifier" => crate::learners::PyRandomForestClassifier,
    "GradientBoostingRegressor" => crate::learners::PyGradientBoostingRegressor,
    "GradientBoostingClassifier" => crate::learners::PyGradientBoostingClassifier,
    "HistGradientBoostingRegressor" => crate::learners::PyHistGradientBoostingRegressor,
    "HistGradientBoostingClassifier" => crate::learners::PyHistGradientBoostingClassifier,
    "RegressionForest" => crate::forest::PyRegressionForest,
    "QuantileForest" => crate::forest::PyQuantileForest,
    "DmlResult" => crate::PyDmlResult,
    "SensitivityResult" => crate::PySensitivityResult,
    "SensitivityBenchmark" => crate::PySensitivityBenchmark,
    "Bootstrap" => crate::PyBootstrap,
    "PolicyTree" => crate::PyPolicyTree,
    "AposResult" => crate::PyAposResult,
    "RdFlexResult" => crate::PyRdFlexResult,
    "QteResult" => crate::PyQteResult,
    "Blp" => crate::cate::PyBlp,
    "CausalForestResult" => crate::forest::PyCausalForestResult,
    "InstrumentalForestResult" => crate::forest::PyInstrumentalForestResult,
    "MultiArmCausalForestResult" => crate::forest::PyMultiArmCausalForestResult,
    "StaggeredResult" => crate::staggered::PyStaggeredResult,
}

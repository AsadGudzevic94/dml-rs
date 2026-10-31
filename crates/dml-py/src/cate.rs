//! CATE meta-learners and BLP / GATE inference for Python.

use dml_core::{Blp, CateFit, DmlData, DrLearner, RLearner, SLearner, TLearner, XLearner};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::learners::{to_classifier, to_regressor};
use crate::{to_splits, PySplits};

/// Two NumPy vectors, e.g. `(effect, se)`.
type VecPair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Best linear predictor of a CATE (or GATEs) with HC0 inference.
#[pyclass(name = "Blp", module = "dml_rs", frozen)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PyBlp {
    pub(crate) blp: Blp,
}

#[pymethods]
impl PyBlp {
    /// Pickle support (`pickle`, `copy.deepcopy`, `joblib`).
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<crate::pickle::Reduced<'py>> {
        crate::pickle::reduce(py, "Blp", self)
    }

    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.blp.coef().clone().into_pyarray(py)
    }

    #[getter]
    fn se<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.blp.se().clone().into_pyarray(py)
    }

    #[getter]
    fn t_stat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.blp.t_stat().into_pyarray(py)
    }

    #[getter]
    fn pval<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.blp.p_value().into_pyarray(py)
    }

    #[getter]
    fn is_gate(&self) -> bool {
        self.blp.is_gate()
    }

    /// Coefficient intervals, shape `(p, 2)`.
    #[pyo3(signature = (level = 0.95))]
    fn confint<'py>(&self, py: Python<'py>, level: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.blp.confint(level).map_err(to_py_err)?.into_pyarray(py))
    }

    /// `(effect, se)` at new basis rows.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<VecPair<'py>> {
        let (effect, se) = self.blp.predict(basis.as_array()).map_err(to_py_err)?;
        Ok((effect.into_pyarray(py), se.into_pyarray(py)))
    }

    /// Pointwise intervals at basis rows, columns `[lower, effect, upper]`.
    #[pyo3(signature = (basis, level = 0.95))]
    fn confint_at<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, f64>,
        level: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .blp
            .confint_at(basis.as_array(), level)
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    /// Simultaneous intervals over all basis rows (Gaussian multiplier bootstrap).
    #[pyo3(signature = (basis, level = 0.95, n_boot = 500, seed = 0))]
    fn joint_confint_at<'py>(
        &self,
        py: Python<'py>,
        basis: PyReadonlyArray2<'py, f64>,
        level: f64,
        n_boot: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .blp
            .joint_confint_at(basis.as_array(), level, n_boot, seed)
            .map_err(to_py_err)?
            .into_pyarray(py))
    }
}

/// A fitted CATE meta-learner.
#[pyclass(name = "CateResult", module = "dml_rs", frozen)]
pub struct PyCateResult {
    fit: CateFit,
}

#[pymethods]
impl PyCateResult {
    #[getter]
    fn method(&self) -> &str {
        self.fit.method()
    }

    /// `τ̂(x)` for each row of `x`.
    fn effect<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .fit
            .effect(x.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    /// DR-learner only: cross-fitted AIPW pseudo-outcomes (else `None`).
    #[getter]
    fn pseudo_outcomes<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.fit
            .pseudo_outcomes()
            .map(|p| p.to_owned().into_pyarray(py))
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

    fn __repr__(&self) -> String {
        format!("<CateResult {}>", self.fit.method())
    }
}

fn binary_data(
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
) -> PyResult<DmlData> {
    DmlData::new(
        y.as_array().to_owned(),
        d.as_array().to_owned(),
        x.as_array().to_owned(),
    )
    .map_err(to_py_err)
}

/// The learner for a role, or an error naming the missing role.
fn need<'a, 'py>(
    obj: Option<&'a Bound<'py, PyAny>>,
    method: &str,
    role: &str,
) -> PyResult<&'a Bound<'py, PyAny>> {
    obj.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{method}-learner needs a `{role}` learner"
        ))
    })
}

/// `method`: `"S"`, `"T"`, `"X"`, `"R"` or `"DR"`. Learner roles:
/// S/T use `outcome`; X uses `outcome`, `effect`, `propensity`;
/// R and DR use `outcome`, `propensity`, `effect` and cross-fit on `splits`
/// (one repetition) or seeded `n_folds`.
#[pyfunction]
#[pyo3(signature = (method, y, d, x, outcome, effect = None, propensity = None,
                    n_folds = 5, seed = 0, clipping_threshold = 1e-2, splits = None))]
#[allow(clippy::too_many_arguments)]
pub fn fit_cate(
    py: Python<'_>,
    method: &str,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray1<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    outcome: &Bound<'_, PyAny>,
    effect: Option<&Bound<'_, PyAny>>,
    propensity: Option<&Bound<'_, PyAny>>,
    n_folds: usize,
    seed: u64,
    clipping_threshold: f64,
    splits: Option<PySplits>,
) -> PyResult<PyCateResult> {
    let data = binary_data(y, d, x)?;
    let folds = splits
        .map(to_splits)
        .and_then(|mut reps| (!reps.is_empty()).then(|| reps.swap_remove(0)));
    let outcome = to_regressor(outcome)?;
    let fit = match method {
        "S" => {
            let model = SLearner::from_boxed(outcome);
            py.detach(|| model.fit(&data))
        }
        "T" => {
            let model = TLearner::from_boxed(outcome);
            py.detach(|| model.fit(&data))
        }
        "X" => {
            let model = XLearner::from_boxed(
                outcome,
                to_regressor(need(effect, method, "effect")?)?,
                to_classifier(need(propensity, method, "propensity")?)?,
            );
            py.detach(|| model.fit(&data))
        }
        "R" => {
            let model = RLearner::from_boxed(
                outcome,
                to_classifier(need(propensity, method, "propensity")?)?,
                to_regressor(need(effect, method, "effect")?)?,
            )
            .n_folds(n_folds)
            .seed(seed)
            .clipping_threshold(clipping_threshold);
            py.detach(|| match &folds {
                Some(f) => model.fit_with_folds(&data, f),
                None => model.fit(&data),
            })
        }
        "DR" => {
            let model = DrLearner::from_boxed(
                outcome,
                to_classifier(need(propensity, method, "propensity")?)?,
                to_regressor(need(effect, method, "effect")?)?,
            )
            .n_folds(n_folds)
            .seed(seed)
            .clipping_threshold(clipping_threshold);
            py.detach(|| match &folds {
                Some(f) => model.fit_with_folds(&data, f),
                None => model.fit(&data),
            })
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown CATE method {other:?} (use 'S', 'T', 'X', 'R' or 'DR')"
            )))
        }
    };
    Ok(PyCateResult {
        fit: fit.map_err(to_py_err)?,
    })
}

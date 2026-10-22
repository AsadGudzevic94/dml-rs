//! Native extension `dml_rs._dml_rs`. The user-facing scikit-learn style
//! classes live in the Python package (`python/dml_rs`), which calls the
//! `fit_*` functions below.

use dml_core::{
    BootstrapMethod, DmlData, DmlFit, Fold, Irm, IrmScore, PAdjust, Pliv, Plr, PlrScore,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod cate;
mod errors;
mod learners;

use cate::{fit_cate, PyBlp, PyCateResult};
use errors::to_py_err;
use learners::{
    to_classifier, to_regressor, PyLasso, PyLogisticRegression, PyOls, PyRandomForestClassifier,
    PyRandomForestRegressor,
};

/// Splits from Python: one list per repetition of `(train, test)` index lists.
pub(crate) type PySplits = Vec<Vec<(Vec<usize>, Vec<usize>)>>;

pub(crate) fn to_splits(splits: PySplits) -> Vec<Vec<Fold>> {
    splits
        .into_iter()
        .map(|rep| {
            rep.into_iter()
                .map(|(train, test)| Fold { train, test })
                .collect()
        })
        .collect()
}

fn build_data(
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    z: Option<PyReadonlyArray1<'_, f64>>,
    treatment_names: Option<Vec<String>>,
) -> PyResult<DmlData> {
    let mut data = DmlData::with_treatments(
        y.as_array().to_owned(),
        d.as_array().to_owned(),
        x.as_array().to_owned(),
    )
    .map_err(to_py_err)?;
    if let Some(z) = z {
        data = data
            .with_instrument(z.as_array().to_owned())
            .map_err(to_py_err)?;
    }
    if let Some(names) = treatment_names {
        data = data.with_treatment_names(names).map_err(to_py_err)?;
    }
    Ok(data)
}

/// A fitted DML model (returned by the `fit_*` functions).
#[pyclass(name = "DmlResult", module = "dml_rs", frozen)]
pub struct PyDmlResult {
    fit: DmlFit,
}

#[pymethods]
impl PyDmlResult {
    #[getter]
    fn model_name(&self) -> &str {
        self.fit.model_name()
    }

    #[getter]
    fn treatment_names(&self) -> Vec<String> {
        self.fit.treatment_names().to_vec()
    }

    #[getter]
    fn n_obs(&self) -> usize {
        self.fit.n_obs()
    }

    #[getter]
    fn n_rep(&self) -> usize {
        self.fit.n_rep()
    }

    #[getter]
    fn n_folds(&self) -> usize {
        self.fit.n_folds()
    }

    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.coef().to_owned().into_pyarray(py)
    }

    #[getter]
    fn se<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.se().to_owned().into_pyarray(py)
    }

    #[getter]
    fn t_stat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.t_stat().into_pyarray(py)
    }

    #[getter]
    fn pval<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.p_value().into_pyarray(py)
    }

    /// Shape `(n_treat, n_rep)`.
    #[getter]
    fn all_coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.all_coef().to_owned().into_pyarray(py)
    }

    #[getter]
    fn all_se<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.all_se().to_owned().into_pyarray(py)
    }

    /// Score values at θ̂, shape `(n_obs, n_treat, n_rep)`.
    #[getter]
    fn psi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.fit.psi().to_owned().into_pyarray(py)
    }

    #[getter]
    fn nuisance_names(&self) -> Vec<String> {
        self.fit.nuisance_names().map(str::to_string).collect()
    }

    /// Cross-fitted predictions of one nuisance, shape `(n_obs, n_rep, n_treat)`.
    fn predictions<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.fit
            .predictions(name)
            .map(|p| p.to_owned().into_pyarray(py))
            .ok_or_else(|| PyValueError::new_err(format!("no nuisance named {name:?}")))
    }

    /// Pointwise confidence intervals, shape `(n_treat, 2)`.
    #[pyo3(signature = (level = 0.95))]
    fn confint<'py>(&self, py: Python<'py>, level: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.fit.confint(level).map_err(to_py_err)?.into_pyarray(py))
    }

    /// `"bonferroni"` or `"holm"`.
    fn p_adjust<'py>(&self, py: Python<'py>, method: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let method = match method.to_ascii_lowercase().as_str() {
            "bonferroni" => PAdjust::Bonferroni,
            "holm" => PAdjust::Holm,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown p_adjust method {other:?} (use 'bonferroni', 'holm', or bootstrap().romano_wolf())"
                )))
            }
        };
        Ok(self.fit.p_adjust(method).into_pyarray(py))
    }

    /// Multiplier bootstrap; `method` is `"normal"`, `"Bayes"` or `"wild"`.
    #[pyo3(signature = (method = "normal", n_boot = 500, seed = 0))]
    fn bootstrap(
        &self,
        py: Python<'_>,
        method: &str,
        n_boot: usize,
        seed: u64,
    ) -> PyResult<PyBootstrap> {
        let method = match method.to_ascii_lowercase().as_str() {
            "normal" => BootstrapMethod::Normal,
            "bayes" => BootstrapMethod::Bayes,
            "wild" => BootstrapMethod::Wild,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown bootstrap method {other:?}"
                )))
            }
        };
        let boot = py
            .detach(|| self.fit.bootstrap(method, n_boot, seed))
            .map_err(to_py_err)?;
        Ok(PyBootstrap { boot })
    }

    /// BLP of the CATE on `basis` (IRM with the ATE score).
    fn cate(&self, basis: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.cate(basis.as_array()).map_err(to_py_err)?,
        })
    }

    /// GATEs for mutually exclusive 0/1 group dummies (IRM with the ATE score).
    fn gate(&self, groups: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.gate(groups.as_array()).map_err(to_py_err)?,
        })
    }

    fn __str__(&self) -> String {
        self.fit.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "<DmlResult {} n_obs={} n_treat={} n_rep={}>",
            self.fit.model_name(),
            self.fit.n_obs(),
            self.fit.n_treat(),
            self.fit.n_rep()
        )
    }
}

/// Bootstrap distribution of the t-statistics.
#[pyclass(name = "Bootstrap", module = "dml_rs", frozen)]
pub struct PyBootstrap {
    boot: dml_core::Bootstrap,
}

#[pymethods]
impl PyBootstrap {
    /// Shape `(n_boot, n_treat, n_rep)`.
    #[getter]
    fn t_stats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.boot.t_stats().to_owned().into_pyarray(py)
    }

    /// Simultaneous confidence intervals, shape `(n_treat, 2)`.
    #[pyo3(signature = (level = 0.95))]
    fn joint_confint<'py>(
        &self,
        py: Python<'py>,
        level: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self
            .boot
            .joint_confint(level)
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    fn romano_wolf<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.boot.romano_wolf().into_pyarray(py)
    }
}

/// Run a configured model with the GIL released (so rayon worker threads can
/// call back into Python learners), on explicit or seeded splits.
macro_rules! run_model {
    ($py:expr, $model:expr, $data:expr, $splits:expr) => {{
        let model = $model;
        let data = $data;
        let result = match $splits {
            Some(s) => {
                let s = to_splits(s);
                $py.detach(|| model.fit_with_folds(&data, &s))
            }
            None => $py.detach(|| model.fit(&data)),
        };
        result.map(|fit| PyDmlResult { fit }).map_err(to_py_err)
    }};
}

#[pyfunction]
#[pyo3(signature = (y, d, x, ml_l, ml_m, ml_g = None, score = "partialling out",
                    n_folds = 5, n_rep = 1, seed = 0, splits = None, treatment_names = None))]
#[allow(clippy::too_many_arguments)]
fn fit_plr(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_l: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    ml_g: Option<&Bound<'_, PyAny>>,
    score: &str,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
) -> PyResult<PyDmlResult> {
    let score = match score {
        "partialling out" => PlrScore::PartiallingOut,
        "IV-type" => PlrScore::IvType,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown PLR score {other:?}"
            )))
        }
    };
    let data = build_data(y, d, x, None, treatment_names)?;
    let mut model = Plr::from_boxed(to_regressor(ml_l)?, to_regressor(ml_m)?)
        .score(score)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    if let Some(g) = ml_g {
        model = model.ml_g_boxed(to_regressor(g)?);
    }
    run_model!(py, model, data, splits)
}

#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m, score = "ATE", clipping_threshold = 1e-2,
                    normalize_ipw = false, n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    treatment_names = None))]
#[allow(clippy::too_many_arguments)]
fn fit_irm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    score: &str,
    clipping_threshold: f64,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
) -> PyResult<PyDmlResult> {
    let score = match score {
        "ATE" => IrmScore::Ate,
        "ATTE" => IrmScore::Atte,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown IRM score {other:?}"
            )))
        }
    };
    let data = build_data(y, d, x, None, treatment_names)?;
    let model = Irm::from_boxed(to_regressor(ml_g)?, to_classifier(ml_m)?)
        .score(score)
        .clipping_threshold(clipping_threshold)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    run_model!(py, model, data, splits)
}

#[pyfunction]
#[pyo3(signature = (y, d, x, z, ml_l, ml_m, ml_r, n_folds = 5, n_rep = 1, seed = 0,
                    splits = None, treatment_names = None))]
#[allow(clippy::too_many_arguments)]
fn fit_pliv(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    ml_l: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    ml_r: &Bound<'_, PyAny>,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
) -> PyResult<PyDmlResult> {
    let data = build_data(y, d, x, Some(z), treatment_names)?;
    let model = Pliv::from_boxed(
        to_regressor(ml_l)?,
        to_regressor(ml_m)?,
        to_regressor(ml_r)?,
    )
    .n_folds(n_folds)
    .n_rep(n_rep)
    .seed(seed);
    run_model!(py, model, data, splits)
}

#[pymodule(name = "_dml_rs")]
mod dml_rs_native {
    #[pymodule_export]
    use super::{
        fit_cate, fit_irm, fit_pliv, fit_plr, PyBlp, PyBootstrap, PyCateResult, PyDmlResult,
        PyLasso, PyLogisticRegression, PyOls, PyRandomForestClassifier, PyRandomForestRegressor,
    };

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        use pyo3::types::PyModuleMethods;
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}

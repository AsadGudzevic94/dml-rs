//! Native extension `dml_rs._dml_rs`. The user-facing scikit-learn style
//! classes live in the Python package (`python/dml_rs`), which calls the
//! `fit_*` functions below.

use dml_core::{
    Apo, Apos, AposFit, BootstrapMethod, Did, DidCs, DidScore, DmlData, DmlFit, Fold, Iivm, Irm,
    IrmScore, PAdjust, Pliv, Plr, PlrScore,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod cate;
mod errors;
mod learners;
mod staggered;

use cate::{fit_cate, PyBlp, PyCateResult};
use errors::to_py_err;
use learners::{
    to_classifier, to_regressor, PyGradientBoostingClassifier, PyGradientBoostingRegressor,
    PyLasso, PyLassoCV, PyLogisticRegression, PyLogisticRegressionCV, PyOls,
    PyRandomForestClassifier, PyRandomForestRegressor,
};
use staggered::{fit_staggered_did, PyStaggeredResult};

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
    t: Option<PyReadonlyArray1<'_, f64>>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
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
    if let Some(t) = t {
        data = data
            .with_time_indicator(t.as_array().to_owned())
            .map_err(to_py_err)?;
    }
    if let Some(names) = treatment_names {
        data = data.with_treatment_names(names).map_err(to_py_err)?;
    }
    // Integer cluster ids, one column per cluster variable (one- or two-way).
    if let Some(c) = cluster {
        let c = c.as_array();
        data = match c.ncols() {
            1 => data.with_clusters(c.column(0).to_owned()),
            2 => data.with_two_way_clusters(c.column(0).to_owned(), c.column(1).to_owned()),
            k => {
                return Err(PyValueError::new_err(format!(
                    "cluster must have one or two columns, got {k}"
                )))
            }
        }
        .map_err(to_py_err)?;
    }
    Ok(data)
}

/// A fitted DML model (returned by the `fit_*` functions).
#[pyclass(name = "DmlResult", module = "dml_rs", frozen)]
pub struct PyDmlResult {
    fit: DmlFit,
}

impl PyDmlResult {
    pub(crate) fn from_fit(fit: DmlFit) -> Self {
        Self { fit }
    }
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

    /// Number of clusters (two-way: the smaller dimension), `None` without clustering.
    #[getter]
    fn n_clusters(&self) -> Option<usize> {
        self.fit.n_clusters()
    }

    /// Out-of-sample loss of every nuisance learner (RMSE for regressions,
    /// log loss for classifications), each of shape `(n_treat, n_rep)`.
    fn nuisance_losses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, _, loss) in self.fit.nuisance_losses() {
            out.set_item(name, loss.to_owned().into_pyarray(py))?;
        }
        Ok(out)
    }

    /// Targets of one nuisance learner, shape `(n_obs, n_rep, n_treat)`, `NaN`
    /// where the nuisance does not apply.
    fn nuisance_targets<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.fit
            .nuisance_targets(name)
            .map(|t| t.to_owned().into_pyarray(py))
            .ok_or_else(|| PyValueError::new_err(format!("no nuisance named {name:?}")))
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

    /// Sensitivity analysis for omitted confounders (PLR, IRM, DID).
    #[pyo3(signature = (cf_y = 0.03, cf_d = 0.03, rho = 1.0, level = 0.95, null_hypothesis = None))]
    fn sensitivity_analysis(
        &self,
        cf_y: f64,
        cf_d: f64,
        rho: f64,
        level: f64,
        null_hypothesis: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PySensitivityResult> {
        let nulls: Vec<f64> = match null_hypothesis {
            None => vec![0.0],
            Some(obj) => match obj.extract::<f64>() {
                Ok(v) => vec![v],
                Err(_) => obj.extract::<Vec<f64>>()?,
            },
        };
        let params = dml_core::SensitivityParams::new()
            .cf_y(cf_y)
            .cf_d(cf_d)
            .rho(rho)
            .level(level)
            .null_hypotheses(nulls);
        Ok(PySensitivityResult {
            res: self.fit.sensitivity_analysis(&params).map_err(to_py_err)?,
            text: String::new(),
        }
        .with_text())
    }

    /// `(sigma2, nu2)`, each of shape `(n_treat, n_rep)`, or `None`.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn sensitivity_elements<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        self.fit
            .sensitivity_elements()
            .map(|(s, n)| (s.to_owned().into_pyarray(py), n.to_owned().into_pyarray(py)))
    }

    /// Benchmark against `short`, the same model refitted on the same splits
    /// without the benchmarking covariates.
    fn sensitivity_benchmark(&self, short: &PyDmlResult) -> PyResult<PySensitivityBenchmark> {
        Ok(PySensitivityBenchmark {
            bm: self
                .fit
                .sensitivity_benchmark(&short.fit)
                .map_err(to_py_err)?,
        })
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

/// Result of a sensitivity analysis.
#[pyclass(name = "SensitivityResult", module = "dml_rs", frozen)]
pub struct PySensitivityResult {
    res: dml_core::SensitivityResult,
    text: String,
}

impl PySensitivityResult {
    fn with_text(mut self) -> Self {
        self.text = self.res.to_string();
        self
    }
}

#[pymethods]
impl PySensitivityResult {
    #[getter]
    fn theta_lower<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.theta_lower.clone().into_pyarray(py)
    }
    #[getter]
    fn theta_upper<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.theta_upper.clone().into_pyarray(py)
    }
    #[getter]
    fn se_lower<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.se_lower.clone().into_pyarray(py)
    }
    #[getter]
    fn se_upper<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.se_upper.clone().into_pyarray(py)
    }
    #[getter]
    fn ci_lower<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.ci_lower.clone().into_pyarray(py)
    }
    #[getter]
    fn ci_upper<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.ci_upper.clone().into_pyarray(py)
    }
    #[getter]
    fn rv<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.rv.clone().into_pyarray(py)
    }
    #[getter]
    fn rva<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.res.rva.clone().into_pyarray(py)
    }
    /// `(cf_y, cf_d, rho, level, null_hypothesis)`.
    #[getter]
    fn params(&self) -> (f64, f64, f64, f64, Vec<f64>) {
        let p = &self.res.params;
        (p.cf_y, p.cf_d, p.rho, p.level, p.null_hypothesis.clone())
    }
    fn __str__(&self) -> String {
        self.text.clone()
    }
}

/// Benchmarked confounding scenario from omitting observed covariates.
#[pyclass(name = "SensitivityBenchmark", module = "dml_rs", frozen)]
pub struct PySensitivityBenchmark {
    bm: dml_core::SensitivityBenchmark,
}

#[pymethods]
impl PySensitivityBenchmark {
    #[getter]
    fn cf_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bm.cf_y.clone().into_pyarray(py)
    }
    #[getter]
    fn cf_d<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bm.cf_d.clone().into_pyarray(py)
    }
    #[getter]
    fn rho<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bm.rho.clone().into_pyarray(py)
    }
    #[getter]
    fn delta_theta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.bm.delta_theta.clone().into_pyarray(py)
    }
    fn __str__(&self) -> String {
        self.bm.to_string()
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
                    n_folds = 5, n_rep = 1, seed = 0, splits = None, treatment_names = None, cluster = None))]
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
    cluster: Option<PyReadonlyArray2<'_, i64>>,
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
    let data = build_data(y, d, x, None, None, treatment_names, cluster)?;
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
                    treatment_names = None, cluster = None))]
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
    cluster: Option<PyReadonlyArray2<'_, i64>>,
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
    let data = build_data(y, d, x, None, None, treatment_names, cluster)?;
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
                    splits = None, treatment_names = None, cluster = None))]
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
    cluster: Option<PyReadonlyArray2<'_, i64>>,
) -> PyResult<PyDmlResult> {
    let data = build_data(y, d, x, Some(z), None, treatment_names, cluster)?;
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

#[pyfunction]
#[pyo3(signature = (y, d, x, z, ml_g, ml_m, ml_r, always_takers = true, never_takers = true,
                    clipping_threshold = 1e-2, normalize_ipw = false, n_folds = 5, n_rep = 1,
                    seed = 0, splits = None, treatment_names = None, cluster = None))]
#[allow(clippy::too_many_arguments)]
fn fit_iivm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    z: PyReadonlyArray1<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    ml_r: &Bound<'_, PyAny>,
    always_takers: bool,
    never_takers: bool,
    clipping_threshold: f64,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
) -> PyResult<PyDmlResult> {
    let data = build_data(y, d, x, Some(z), None, treatment_names, cluster)?;
    let model = Iivm::from_boxed(
        to_regressor(ml_g)?,
        to_classifier(ml_m)?,
        to_classifier(ml_r)?,
    )
    .subgroups(always_takers, never_takers)
    .clipping_threshold(clipping_threshold)
    .normalize_ipw(normalize_ipw)
    .n_folds(n_folds)
    .n_rep(n_rep)
    .seed(seed);
    run_model!(py, model, data, splits)
}

/// Average potential outcome of one treatment level (discrete treatment).
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m, treatment_level, clipping_threshold = 1e-2,
                    normalize_ipw = false, n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    treatment_names = None, cluster = None))]
#[allow(clippy::too_many_arguments)]
fn fit_apo(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    treatment_level: f64,
    clipping_threshold: f64,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
) -> PyResult<PyDmlResult> {
    let data = build_data(y, d, x, None, None, treatment_names, cluster)?;
    let model = Apo::from_boxed(to_regressor(ml_g)?, to_classifier(ml_m)?, treatment_level)
        .clipping_threshold(clipping_threshold)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    run_model!(py, model, data, splits)
}

/// Average potential outcomes of several treatment levels, fitted on shared
/// sample splits (returned by `fit_apos`).
#[pyclass(name = "AposResult", module = "dml_rs", frozen)]
pub struct PyAposResult {
    inner: AposFit,
}

#[pymethods]
impl PyAposResult {
    #[getter]
    fn treatment_levels(&self) -> Vec<f64> {
        self.inner.levels().to_vec()
    }

    /// One estimate per treatment level.
    #[getter]
    fn levels(&self) -> PyDmlResult {
        PyDmlResult::from_fit(self.inner.apos().clone())
    }

    /// Effects of each level against each reference level (`"d vs d_ref"`).
    fn causal_contrast(&self, reference_levels: Vec<f64>) -> PyResult<PyDmlResult> {
        self.inner
            .causal_contrast(&reference_levels)
            .map(PyDmlResult::from_fit)
            .map_err(to_py_err)
    }
}

#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m, treatment_levels, clipping_threshold = 1e-2,
                    normalize_ipw = false, n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    cluster = None))]
#[allow(clippy::too_many_arguments)]
fn fit_apos(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    treatment_levels: Vec<f64>,
    clipping_threshold: f64,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
) -> PyResult<PyAposResult> {
    let data = build_data(y, d, x, None, None, None, cluster)?;
    let model = Apos::from_boxed(to_regressor(ml_g)?, to_classifier(ml_m)?, treatment_levels)
        .clipping_threshold(clipping_threshold)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    let result = match splits {
        Some(s) => {
            let s = to_splits(s);
            py.detach(|| model.fit_with_folds(&data, &s))
        }
        None => py.detach(|| model.fit(&data)),
    };
    result
        .map(|inner| PyAposResult { inner })
        .map_err(to_py_err)
}

pub(crate) fn did_score(score: &str) -> PyResult<DidScore> {
    match score {
        "observational" => Ok(DidScore::Observational),
        "experimental" => Ok(DidScore::Experimental),
        other => Err(PyValueError::new_err(format!(
            "unknown DID score {other:?} (use 'observational' or 'experimental')"
        ))),
    }
}

/// Two-period DID. Panel data when `t` is `None` (then `y` is the outcome
/// change), repeated cross-sections when the time indicator `t` is given.
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m = None, t = None, score = "observational",
                    in_sample_normalization = true, clipping_threshold = 1e-2, n_folds = 5,
                    n_rep = 1, seed = 0, splits = None, treatment_names = None, cluster = None))]
#[allow(clippy::too_many_arguments)]
fn fit_did(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: Option<&Bound<'_, PyAny>>,
    t: Option<PyReadonlyArray1<'_, f64>>,
    score: &str,
    in_sample_normalization: bool,
    clipping_threshold: f64,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
) -> PyResult<PyDmlResult> {
    let score = did_score(score)?;
    let ml_m = ml_m.map(to_classifier).transpose()?;
    let cross_section = t.is_some();
    let data = build_data(y, d, x, None, t, treatment_names, cluster)?;
    if cross_section {
        let model = DidCs::from_boxed(to_regressor(ml_g)?, ml_m)
            .score(score)
            .in_sample_normalization(in_sample_normalization)
            .clipping_threshold(clipping_threshold)
            .n_folds(n_folds)
            .n_rep(n_rep)
            .seed(seed);
        run_model!(py, model, data, splits)
    } else {
        let model = Did::from_boxed(to_regressor(ml_g)?, ml_m)
            .score(score)
            .in_sample_normalization(in_sample_normalization)
            .clipping_threshold(clipping_threshold)
            .n_folds(n_folds)
            .n_rep(n_rep)
            .seed(seed);
        run_model!(py, model, data, splits)
    }
}

#[pymodule(name = "_dml_rs")]
mod dml_rs_native {
    #[pymodule_export]
    use super::{
        fit_apo, fit_apos, fit_cate, fit_did, fit_iivm, fit_irm, fit_pliv, fit_plr,
        fit_staggered_did, PyAposResult, PyBlp, PyBootstrap, PyCateResult, PyDmlResult,
        PyGradientBoostingClassifier, PyGradientBoostingRegressor, PyLasso, PyLassoCV,
        PyLogisticRegression, PyLogisticRegressionCV, PyOls, PyRandomForestClassifier,
        PyRandomForestRegressor, PySensitivityBenchmark, PySensitivityResult, PyStaggeredResult,
    };

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        use pyo3::types::PyModuleMethods;
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}

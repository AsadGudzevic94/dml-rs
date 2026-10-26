//! Native extension `dml_rs._dml_rs`. The user-facing scikit-learn style
//! classes live in the Python package (`python/dml_rs`), which calls the
//! `fit_*` functions below.

use dml_core::{
    Apo, Apos, AposFit, BootstrapMethod, Cvar, Did, DidCs, DidScore, DmlData, DmlFit,
    ExternalPredictions, Fold, Iivm, Irm, IrmScore, Lplr, LplrScore, Lpq, PAdjust, PanelApproach,
    Pliv, PlivScore, PlivVariant, Plpr, Plr, PlrScore, Pq, Qte, QteFit, Ssm, SsmScore, StaticPanel,
    TreatmentLearner,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod cate;
mod datasets;
mod errors;
mod learners;
mod staggered;

use cate::{fit_cate, PyBlp, PyCateResult};
use datasets::simulate;
use errors::to_py_err;
use learners::{
    to_classifier, to_regressor, PyGradientBoostingClassifier, PyGradientBoostingRegressor,
    PyGridSearchClassifier, PyGridSearchRegressor, PyLasso, PyLassoCV, PyLogisticRegression,
    PyLogisticRegressionCV, PyOls, PyRandomForestClassifier, PyRandomForestRegressor,
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

/// External nuisance predictions from Python: `(treatment index, learner
/// name, predictions of shape (n_obs, n_rep))`.
pub(crate) type PyExternal<'py> = Vec<(usize, String, PyReadonlyArray2<'py, f64>)>;

fn external_from(external: Option<PyExternal<'_>>) -> Option<ExternalPredictions> {
    external.map(|items| {
        items
            .into_iter()
            .fold(ExternalPredictions::new(), |e, (j, name, p)| {
                e.with(j, name, p.as_array().to_owned())
            })
    })
}

fn attach_external(data: DmlData, external: Option<PyExternal<'_>>) -> DmlData {
    match external_from(external) {
        Some(e) => data.with_external_predictions(e),
        None => data,
    }
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

    /// Linear combinations of the estimates, one per row of `weights`
    /// (shape `(n_combinations, n_treat)`), with joint inference.
    fn linear_combination(
        &self,
        weights: PyReadonlyArray2<'_, f64>,
        names: Vec<String>,
    ) -> PyResult<PyDmlResult> {
        self.fit
            .linear_combination(weights.as_array(), names)
            .map(PyDmlResult::from_fit)
            .map_err(to_py_err)
    }

    // Arithmetic as DoubleML's `DoubleMLFramework`. An operand of another
    // type cannot be extracted, so PyO3 returns `NotImplemented` for it.
    fn __add__(&self, other: PyRef<'_, PyDmlResult>) -> PyResult<PyDmlResult> {
        (&self.fit + &other.fit)
            .map(PyDmlResult::from_fit)
            .map_err(to_py_err)
    }

    fn __sub__(&self, other: PyRef<'_, PyDmlResult>) -> PyResult<PyDmlResult> {
        (&self.fit - &other.fit)
            .map(PyDmlResult::from_fit)
            .map_err(to_py_err)
    }

    fn __mul__(&self, c: f64) -> PyDmlResult {
        PyDmlResult::from_fit(&self.fit * c)
    }

    fn __rmul__(&self, c: f64) -> PyDmlResult {
        PyDmlResult::from_fit(c * &self.fit)
    }

    fn __neg__(&self) -> PyDmlResult {
        PyDmlResult::from_fit(-&self.fit)
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

    /// Sensitivity bounds over a grid: entry `[i, j]` is the `value`
    /// (`"theta"` or `"ci"`) `bound` (`"lower"` or `"upper"`) of `treatment`
    /// at `cf_d = cf_d[i]`, `cf_y = cf_y[j]`.
    #[allow(clippy::too_many_arguments)]
    fn sensitivity_grid<'py>(
        &self,
        py: Python<'py>,
        cf_y: Vec<f64>,
        cf_d: Vec<f64>,
        rho: f64,
        level: f64,
        null_hypothesis: Vec<f64>,
        treatment: usize,
        value: &str,
        bound: &str,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let pick = match (value, bound) {
            ("theta", "lower") => 0,
            ("theta", "upper") => 1,
            ("ci", "lower") => 2,
            ("ci", "upper") => 3,
            _ => {
                return Err(PyValueError::new_err(
                    "value must be 'theta' or 'ci' and bound 'lower' or 'upper'",
                ))
            }
        };
        if treatment >= self.fit.n_treat() {
            return Err(PyValueError::new_err("treatment index out of range"));
        }
        let grid = py.detach(|| -> dml_core::Result<ndarray::Array2<f64>> {
            let mut out = ndarray::Array2::zeros((cf_d.len(), cf_y.len()));
            for (i, &d) in cf_d.iter().enumerate() {
                for (j, &y) in cf_y.iter().enumerate() {
                    let params = dml_core::SensitivityParams::new()
                        .cf_y(y)
                        .cf_d(d)
                        .rho(rho)
                        .level(level)
                        .null_hypotheses(null_hypothesis.clone());
                    let res = self.fit.sensitivity_analysis(&params)?;
                    out[[i, j]] = match pick {
                        0 => res.theta_lower[treatment],
                        1 => res.theta_upper[treatment],
                        2 => res.ci_lower[treatment],
                        _ => res.ci_upper[treatment],
                    };
                }
            }
            Ok(out)
        });
        Ok(grid.map_err(to_py_err)?.into_pyarray(py))
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

    /// GATEs for mutually exclusive 0/1 group dummies (PLR, IRM with the ATE score, APO).
    fn gate(&self, groups: PyReadonlyArray2<'_, f64>) -> PyResult<PyBlp> {
        Ok(PyBlp {
            blp: self.fit.gate(groups.as_array()).map_err(to_py_err)?,
        })
    }

    /// Policy tree on the treatment-effect signal (IRM with the ATE score).
    #[pyo3(signature = (features, depth = 2, min_samples_leaf = 8, ccp_alpha = 0.01))]
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

/// Apply DoubleML-style `weights` (and optional per-repetition `weights_bar`).
macro_rules! with_weights {
    ($model:expr, $weights:expr, $weights_bar:expr) => {
        match ($weights, $weights_bar) {
            (Some(w), Some(b)) => {
                $model.weights_with_bar(w.as_array().to_owned(), b.as_array().to_owned())
            }
            (Some(w), None) => $model.weights(w.as_array().to_owned()),
            (None, None) => $model,
            (None, Some(_)) => {
                return Err(PyValueError::new_err("weights_bar requires weights"));
            }
        }
    };
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
                    n_folds = 5, n_rep = 1, seed = 0, splits = None, treatment_names = None, cluster = None, external_predictions = None))]
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
    external_predictions: Option<PyExternal<'_>>,
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
    let data = attach_external(data, external_predictions);
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
#[pyo3(signature = (y, d, x, ml_g, ml_m, score = "ATE", clipping_threshold = 1e-2, calibration = None,
                    normalize_ipw = false, n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    treatment_names = None, cluster = None, weights = None, weights_bar = None, external_predictions = None))]
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
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    weights_bar: Option<PyReadonlyArray2<'_, f64>>,
    external_predictions: Option<PyExternal<'_>>,
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
    let data = attach_external(data, external_predictions);
    let model = Irm::from_boxed(to_regressor(ml_g)?, to_classifier(ml_m)?)
        .score(score)
        .clipping_threshold(clipping_threshold)
        .calibration(parse_calibration(calibration)?)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    let model = with_weights!(model, weights, weights_bar);
    run_model!(py, model, data, splits)
}

/// A learner the chosen PLIV variant needs.
fn required<'a, 'py>(
    learner: Option<&'a Bound<'py, PyAny>>,
    partial: &str,
    name: &str,
) -> PyResult<&'a Bound<'py, PyAny>> {
    learner
        .ok_or_else(|| PyValueError::new_err(format!("PLIV with partial={partial:?} needs {name}")))
}

/// Partially linear IV regression; `z` has one column per instrument.
/// `partial` is `"X"` (default), `"Z"` (only `ml_r`, learning `E[D | X, Z]`)
/// or `"XZ"`.
#[pyfunction]
#[pyo3(signature = (y, d, x, z, ml_l, ml_m, ml_r, ml_g = None, score = "partialling out",
                    partial = "X", n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    treatment_names = None, cluster = None, external_predictions = None))]
#[allow(clippy::too_many_arguments)]
fn fit_pliv(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    z: PyReadonlyArray2<'_, f64>,
    ml_l: Option<&Bound<'_, PyAny>>,
    ml_m: Option<&Bound<'_, PyAny>>,
    ml_r: &Bound<'_, PyAny>,
    ml_g: Option<&Bound<'_, PyAny>>,
    score: &str,
    partial: &str,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    let score = match score {
        "partialling out" => PlivScore::PartiallingOut,
        "IV-type" => PlivScore::IvType,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown PLIV score {other:?}"
            )))
        }
    };
    let data = build_data(y, d, x, None, None, treatment_names, cluster)?
        .with_instruments(z.as_array().to_owned())
        .map_err(to_py_err)?;
    let data = attach_external(data, external_predictions);
    let mut model = match partial {
        "X" | "XZ" => Pliv::from_boxed(
            to_regressor(required(ml_l, partial, "ml_l")?)?,
            to_regressor(required(ml_m, partial, "ml_m")?)?,
            to_regressor(ml_r)?,
        )
        .variant(if partial == "X" {
            PlivVariant::PartialX
        } else {
            PlivVariant::PartialXZ
        }),
        "Z" => Pliv::partial_z_boxed(to_regressor(ml_r)?),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown partial {other:?} (use 'X', 'Z' or 'XZ')"
            )))
        }
    }
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
#[pyo3(signature = (y, d, x, z, ml_g, ml_m, ml_r, always_takers = true, never_takers = true,
                    clipping_threshold = 1e-2, calibration = None, normalize_ipw = false, n_folds = 5, n_rep = 1,
                    seed = 0, splits = None, treatment_names = None, cluster = None, external_predictions = None))]
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
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    let data = build_data(y, d, x, Some(z), None, treatment_names, cluster)?;
    let data = attach_external(data, external_predictions);
    let model = Iivm::from_boxed(
        to_regressor(ml_g)?,
        to_classifier(ml_m)?,
        to_classifier(ml_r)?,
    )
    .subgroups(always_takers, never_takers)
    .clipping_threshold(clipping_threshold)
    .calibration(parse_calibration(calibration)?)
    .normalize_ipw(normalize_ipw)
    .n_folds(n_folds)
    .n_rep(n_rep)
    .seed(seed);
    run_model!(py, model, data, splits)
}

/// Average potential outcome of one treatment level (discrete treatment).
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m, treatment_level, clipping_threshold = 1e-2, calibration = None,
                    normalize_ipw = false, n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    treatment_names = None, cluster = None, weights = None, weights_bar = None, external_predictions = None))]
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
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    weights_bar: Option<PyReadonlyArray2<'_, f64>>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    let data = build_data(y, d, x, None, None, treatment_names, cluster)?;
    let data = attach_external(data, external_predictions);
    let model = Apo::from_boxed(to_regressor(ml_g)?, to_classifier(ml_m)?, treatment_level)
        .clipping_threshold(clipping_threshold)
        .calibration(parse_calibration(calibration)?)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    let model = with_weights!(model, weights, weights_bar);
    run_model!(py, model, data, splits)
}

/// A fitted policy tree (returned by `DmlResult.policy_tree`).
#[pyclass(name = "PolicyTree", module = "dml_rs", frozen)]
pub struct PyPolicyTree {
    tree: dml_core::PolicyTree,
}

#[pymethods]
impl PyPolicyTree {
    /// Recommended treatment (1) or not (0) for each row of `features`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self
            .tree
            .predict(features.as_array())
            .map_err(to_py_err)?
            .into_pyarray(py))
    }

    /// The decision rules as text.
    #[pyo3(signature = (feature_names = None))]
    fn export_text(&self, feature_names: Option<Vec<String>>) -> String {
        self.tree.export_text(feature_names.as_deref())
    }

    /// The nodes in preorder as `(feature, threshold, left, right, share)`;
    /// leaves have `feature = left = right = -1` and the share of positive
    /// effects, splits a NaN share.
    fn nodes(&self) -> Vec<(i64, f64, i64, i64, f64)> {
        self.tree
            .nodes()
            .into_iter()
            .map(|n| match n {
                dml_core::PolicyNode::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => (
                    feature as i64,
                    threshold,
                    left as i64,
                    right as i64,
                    f64::NAN,
                ),
                dml_core::PolicyNode::Leaf { share_positive } => {
                    (-1, f64::NAN, -1, -1, share_positive)
                }
            })
            .collect()
    }

    #[getter]
    fn n_leaves(&self) -> usize {
        self.tree.n_leaves()
    }

    #[getter]
    fn depth(&self) -> usize {
        self.tree.depth()
    }
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
#[pyo3(signature = (y, d, x, ml_g, ml_m, treatment_levels, clipping_threshold = 1e-2, calibration = None,
                    normalize_ipw = false, n_folds = 5, n_rep = 1, seed = 0, splits = None,
                    cluster = None, weights = None, weights_bar = None))]
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
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    weights_bar: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<PyAposResult> {
    let data = build_data(y, d, x, None, None, None, cluster)?;
    let model = Apos::from_boxed(to_regressor(ml_g)?, to_classifier(ml_m)?, treatment_levels)
        .clipping_threshold(clipping_threshold)
        .calibration(parse_calibration(calibration)?)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    let model = with_weights!(model, weights, weights_bar);
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

/// `E[D | X]` learner of LPLR: a classifier (probabilities) or a regressor.
fn to_treatment_learner(obj: &Bound<'_, PyAny>, classifier: bool) -> PyResult<TreatmentLearner> {
    Ok(if classifier {
        TreatmentLearner::Classifier(to_classifier(obj)?)
    } else {
        TreatmentLearner::Regressor(to_regressor(obj)?)
    })
}

/// Result of `fit_rdflex`: conventional, bias-corrected and robust RD estimates.
#[pyclass(name = "RdFlexResult", module = "dml_rs", frozen)]
pub struct PyRdFlexResult {
    fit: dml_core::RdFlexFit,
}

#[pymethods]
impl PyRdFlexResult {
    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.coef().to_vec().into_pyarray(py)
    }

    #[getter]
    fn se<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.se().to_vec().into_pyarray(py)
    }

    #[getter]
    fn t_stat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.t_stat().to_vec().into_pyarray(py)
    }

    #[getter]
    fn pval<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.p_value().to_vec().into_pyarray(py)
    }

    /// Median of the repetitions' 95% `rdrobust` intervals, shape `(3, 2)`.
    #[getter]
    fn ci<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let ci = self.fit.ci();
        ndarray::Array2::from_shape_fn((3, 2), |(i, j)| ci[i][j]).into_pyarray(py)
    }

    /// Shape `(3, n_rep)`.
    #[getter]
    fn all_coef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.all_coef().into_pyarray(py)
    }

    #[getter]
    fn all_se<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.fit.all_se().into_pyarray(py)
    }

    /// Final RD bandwidth of each repetition.
    #[getter]
    fn h<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.h().into_pyarray(py)
    }

    /// Bias bandwidth of each repetition.
    #[getter]
    fn b<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.b().into_pyarray(py)
    }

    #[getter]
    fn h_fs(&self) -> f64 {
        self.fit.h_fs()
    }

    /// Intervals at `level` from the repetitions' estimates and standard errors.
    #[pyo3(signature = (level = 0.95))]
    fn confint<'py>(&self, py: Python<'py>, level: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ci = self.fit.confint(level).map_err(to_py_err)?;
        Ok(ndarray::Array2::from_shape_fn((3, 2), |(i, j)| ci[i][j]).into_pyarray(py))
    }

    fn __str__(&self) -> String {
        self.fit.to_string()
    }
}

/// `rdrobust` keyword arguments (as in DoubleML's `RDFlex(**kwargs)`) as
/// [`RdOptions`](dml_core::rd::RdOptions). `level` is in percent, as in
/// `rdrobust`.
fn parse_rd_options(options: Option<&Bound<'_, PyDict>>) -> PyResult<dml_core::rd::RdOptions> {
    use dml_core::rd::{BwSelect, Kernel, MassPoints, RdOptions, Vce};
    let mut opts = RdOptions::default();
    let Some(options) = options else {
        return Ok(opts);
    };
    let bad = |key: &str, value: &str| {
        PyValueError::new_err(format!(
            "rdrobust option {key} = {value:?} is not supported"
        ))
    };
    for (key, value) in options.iter() {
        let key: String = key.extract()?;
        let text =
            || -> PyResult<String> { Ok(value.extract::<String>()?.trim().to_ascii_lowercase()) };
        match key.as_str() {
            "p" => opts.p = value.extract()?,
            "q" => opts.q = value.extract()?,
            "kernel" => {
                let v = text()?;
                opts.kernel = match v.as_str() {
                    "tri" | "triangular" => Kernel::Triangular,
                    "epa" | "epanechnikov" => Kernel::Epanechnikov,
                    "uni" | "uniform" => Kernel::Uniform,
                    _ => return Err(bad(&key, &v)),
                }
            }
            "bwselect" => {
                let v = text()?;
                opts.bwselect = match v.as_str() {
                    "mserd" => BwSelect::Mserd,
                    "msetwo" => BwSelect::Msetwo,
                    "msesum" => BwSelect::Msesum,
                    "msecomb1" => BwSelect::Msecomb1,
                    "msecomb2" => BwSelect::Msecomb2,
                    "cerrd" => BwSelect::Cerrd,
                    "certwo" => BwSelect::Certwo,
                    "cersum" => BwSelect::Cersum,
                    "cercomb1" => BwSelect::Cercomb1,
                    "cercomb2" => BwSelect::Cercomb2,
                    _ => return Err(bad(&key, &v)),
                }
            }
            "vce" => {
                let v = text()?;
                opts.vce = match v.as_str() {
                    "nn" => Vce::Nn,
                    "hc0" => Vce::Hc0,
                    "hc1" | "cr1" => Vce::Hc1,
                    "hc2" => Vce::Hc2,
                    "hc3" => Vce::Hc3,
                    _ => return Err(bad(&key, &v)),
                }
            }
            "masspoints" => {
                let v = text()?;
                opts.masspoints = match v.as_str() {
                    "adjust" => MassPoints::Adjust,
                    "check" => MassPoints::Check,
                    "off" => MassPoints::Off,
                    _ => return Err(bad(&key, &v)),
                }
            }
            "nnmatch" => opts.nnmatch = value.extract()?,
            "level" => opts.level = value.extract::<f64>()? / 100.0,
            "bwcheck" => opts.bwcheck = value.extract()?,
            "bwrestrict" => opts.bwrestrict = value.extract()?,
            "stdvars" => opts.stdvars = value.extract()?,
            "scaleregul" => opts.scaleregul = value.extract()?,
            "sharpbw" => opts.sharpbw = value.extract()?,
            "rho" => opts.rho = value.extract()?,
            "scalepar" => opts.scalepar = value.extract()?,
            "covs_drop" => opts.covs_drop = value.extract()?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown or unsupported rdrobust option {other:?}"
                )))
            }
        }
    }
    Ok(opts)
}

/// Regression discontinuity with flexible covariate adjustment (sharp, or
/// fuzzy when `ml_m` is given). `score` is the running variable; `ml_g` is a
/// classifier for a binary outcome when `g_classifier`. `rd_options` holds
/// `rdrobust` keyword arguments, `rd_covs` and `rd_cluster` the RD stage's
/// covariates and cluster labels.
#[pyfunction]
#[pyo3(signature = (y, d, x, score, ml_g, ml_m = None, cutoff = 0.0, h_fs = None,
                    fs_specification = "cutoff", fs_kernel = "triangular", n_iterations = 2,
                    n_folds = 5, n_rep = 1, seed = 0, splits = None, g_classifier = false,
                    rd_options = None, rd_covs = None, rd_cluster = None))]
#[allow(clippy::too_many_arguments)]
fn fit_rdflex(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    score: PyReadonlyArray1<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: Option<&Bound<'_, PyAny>>,
    cutoff: f64,
    h_fs: Option<f64>,
    fs_specification: &str,
    fs_kernel: &str,
    n_iterations: usize,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    g_classifier: bool,
    rd_options: Option<&Bound<'_, PyDict>>,
    rd_covs: Option<PyReadonlyArray2<'_, f64>>,
    rd_cluster: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<PyRdFlexResult> {
    let first_stage = match fs_specification {
        "cutoff" => dml_core::FirstStage::Cutoff,
        "cutoff and score" => dml_core::FirstStage::CutoffAndScore,
        "interacted cutoff and score" => dml_core::FirstStage::InteractedCutoffAndScore,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown fs_specification {other:?} (use 'cutoff', 'cutoff and score' or 'interacted cutoff and score')"
            )))
        }
    };
    let kernel = match fs_kernel.to_ascii_lowercase().as_str() {
        "uniform" => dml_core::RdKernel::Uniform,
        "triangular" => dml_core::RdKernel::Triangular,
        "epanechnikov" => dml_core::RdKernel::Epanechnikov,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown fs_kernel {other:?} (use 'uniform', 'triangular' or 'epanechnikov')"
            )))
        }
    };
    let data = build_data(y, d, x, None, None, None, None)?;
    let ml_m = ml_m.map(to_classifier).transpose()?;
    let model = if g_classifier {
        dml_core::RdFlex::from_boxed_binary(to_classifier(ml_g)?, ml_m)
    } else {
        dml_core::RdFlex::from_boxed(to_regressor(ml_g)?, ml_m)
    };
    let mut model = model.rd_options(parse_rd_options(rd_options)?);
    if let Some(z) = rd_covs {
        model = model.rd_covariates(z.as_array().to_owned());
    }
    if let Some(cl) = rd_cluster {
        model = model.rd_clusters(cl.as_array().to_vec());
    }
    let mut model = model
        .cutoff(cutoff)
        .first_stage(first_stage)
        .kernel(kernel)
        .n_iterations(n_iterations)
        .n_folds(n_folds)
        .n_rep(n_rep)
        .seed(seed);
    if let Some(h) = h_fs {
        model = model.h_fs(h);
    }
    let score = score.as_array().to_owned();
    let result = match splits {
        Some(s) => {
            let s = to_splits(s);
            py.detach(|| model.fit_with_folds(&data, score.view(), &s))
        }
        None => py.detach(|| model.fit(&data, score.view())),
    };
    result.map(|fit| PyRdFlexResult { fit }).map_err(to_py_err)
}

/// Partially linear regression on a static panel (long format: unit `id` and
/// period `t` per row), clustered by unit. `splits` refer to the rows of the
/// transformed panel.
#[pyfunction]
#[pyo3(signature = (y, d, x, id, t, ml_l, ml_m, m_classifier = false, ml_g = None,
                    score = "partialling out", approach = "fd_exact", n_folds = 5, n_rep = 1, seed = 0,
                    splits = None, treatment_names = None, external_predictions = None))]
#[allow(clippy::too_many_arguments)]
fn fit_plpr(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    id: PyReadonlyArray1<'_, i64>,
    t: PyReadonlyArray1<'_, f64>,
    ml_l: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    m_classifier: bool,
    ml_g: Option<&Bound<'_, PyAny>>,
    score: &str,
    approach: &str,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    let score = match score {
        "partialling out" => PlrScore::PartiallingOut,
        "IV-type" => PlrScore::IvType,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown PLPR score {other:?}"
            )))
        }
    };
    let approach = match approach {
        "cre_general" => PanelApproach::CreGeneral,
        "cre_normal" => PanelApproach::CreNormal,
        "fd_exact" => PanelApproach::FdExact,
        "wg_approx" => PanelApproach::WgApprox,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown PLPR approach {other:?} (use 'cre_general', 'cre_normal', 'fd_exact' or 'wg_approx')"
            )))
        }
    };
    let mut panel = StaticPanel::with_treatments(
        y.as_array().to_owned(),
        d.as_array().to_owned(),
        x.as_array().to_owned(),
        id.as_array().to_owned(),
        t.as_array().to_owned(),
    )
    .map_err(to_py_err)?;
    if let Some(names) = treatment_names {
        panel = panel.with_treatment_names(names).map_err(to_py_err)?;
    }
    if let Some(e) = external_from(external_predictions) {
        panel = panel.with_external_predictions(e);
    }
    let mut model = Plpr::from_boxed(
        to_regressor(ml_l)?,
        to_treatment_learner(ml_m, m_classifier)?,
    )
    .score(score)
    .approach(approach)
    .n_folds(n_folds)
    .n_rep(n_rep)
    .seed(seed);
    if let Some(g) = ml_g {
        model = model.ml_g_boxed(to_regressor(g)?);
    }
    let result = match splits {
        Some(s) => {
            let s = to_splits(s);
            py.detach(|| model.fit_with_folds(&panel, &s))
        }
        None => py.detach(|| model.fit(&panel)),
    };
    result.map(PyDmlResult::from_fit).map_err(to_py_err)
}

/// Stack results fitted on the same observations and sample splits into one,
/// for joint inference (DoubleML's `concat`).
#[pyfunction]
fn concat(results: Vec<PyRef<'_, PyDmlResult>>) -> PyResult<PyDmlResult> {
    let fits: Vec<DmlFit> = results.iter().map(|r| r.fit.clone()).collect();
    DmlFit::concat(&fits)
        .map(PyDmlResult::from_fit)
        .map_err(to_py_err)
}

/// Logistic partially linear regression for a binary outcome. `inner_splits`
/// (per repetition, per outer fold: `(train, test)` index lists) fixes the
/// inner folds of the double cross-fitting; otherwise they are drawn.
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_M, ml_t, ml_m, m_classifier, ml_a = None, a_classifier = None,
                    score = "nuisance_space", n_folds = 5, n_folds_inner = 5, n_rep = 1, seed = 0,
                    splits = None, inner_splits = None, treatment_names = None,
                    external_predictions = None))]
#[allow(clippy::too_many_arguments, non_snake_case)]
fn fit_lplr(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_M: &Bound<'_, PyAny>,
    ml_t: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    m_classifier: bool,
    ml_a: Option<&Bound<'_, PyAny>>,
    a_classifier: Option<bool>,
    score: &str,
    n_folds: usize,
    n_folds_inner: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    inner_splits: Option<Vec<PySplits>>,
    treatment_names: Option<Vec<String>>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    let score = match score {
        "nuisance_space" => LplrScore::NuisanceSpace,
        "instrument" => LplrScore::Instrument,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown LPLR score {other:?} (use 'nuisance_space' or 'instrument')"
            )))
        }
    };
    let data = attach_external(
        build_data(y, d, x, None, None, treatment_names, None)?,
        external_predictions,
    );
    let ml_a = match ml_a {
        Some(a) => Some(to_treatment_learner(
            a,
            a_classifier.unwrap_or(m_classifier),
        )?),
        None => None,
    };
    let model = Lplr::from_boxed(
        to_classifier(ml_M)?,
        to_regressor(ml_t)?,
        to_treatment_learner(ml_m, m_classifier)?,
        ml_a,
    )
    .score(score)
    .n_folds(n_folds)
    .n_folds_inner(n_folds_inner)
    .n_rep(n_rep)
    .seed(seed);
    let result = match (splits, inner_splits) {
        (Some(s), Some(inner)) => {
            let s = to_splits(s);
            let inner: Vec<Vec<Vec<Fold>>> = inner.into_iter().map(to_splits).collect();
            py.detach(|| model.fit_with_inner_folds(&data, &s, &inner))
        }
        (Some(s), None) => {
            let s = to_splits(s);
            py.detach(|| model.fit_with_folds(&data, &s))
        }
        (None, None) => py.detach(|| model.fit(&data)),
        (None, Some(_)) => return Err(PyValueError::new_err("inner_splits requires splits")),
    };
    result.map(PyDmlResult::from_fit).map_err(to_py_err)
}

/// Average treatment effect under sample selection: the outcome is observed
/// where the 0/1 indicator `s` is 1. The nonignorable score needs `z`.
#[pyfunction]
#[pyo3(signature = (y, d, x, s, ml_g, ml_pi, ml_m, z = None, score = "missing-at-random",
                    clipping_threshold = 1e-2, calibration = None, normalize_ipw = false, n_folds = 5, n_rep = 1,
                    seed = 0, splits = None, treatment_names = None, cluster = None))]
#[allow(clippy::too_many_arguments)]
fn fit_ssm(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    s: PyReadonlyArray1<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_pi: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    z: Option<PyReadonlyArray1<'_, f64>>,
    score: &str,
    clipping_threshold: f64,
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
) -> PyResult<PyDmlResult> {
    let score = match score {
        "missing-at-random" => SsmScore::MissingAtRandom,
        "nonignorable" => SsmScore::Nonignorable,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown SSM score {other:?} (use 'missing-at-random' or 'nonignorable')"
            )))
        }
    };
    let data = build_data(y, d, x, z, None, treatment_names, cluster)?
        .with_selection_indicator(s.as_array().to_owned())
        .map_err(to_py_err)?;
    let model = Ssm::from_boxed(
        to_regressor(ml_g)?,
        to_classifier(ml_pi)?,
        to_classifier(ml_m)?,
    )
    .score(score)
    .clipping_threshold(clipping_threshold)
    .calibration(parse_calibration(calibration)?)
    .normalize_ipw(normalize_ipw)
    .n_folds(n_folds)
    .n_rep(n_rep)
    .seed(seed);
    run_model!(py, model, data, splits)
}

/// Potential quantile (`"PQ"`), local potential quantile (`"LPQ"`, needs `z`)
/// or conditional value at risk (`"CVaR"`) of one treatment level.
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m, score = "PQ", treatment = 1, quantile = 0.5, z = None,
                    clipping_threshold = 1e-2, calibration = None, normalize_ipw = true, n_folds = 5, n_rep = 1,
                    seed = 0, splits = None, external_predictions = None))]
#[allow(clippy::too_many_arguments)]
fn fit_quantile(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    score: &str,
    treatment: u8,
    quantile: f64,
    z: Option<PyReadonlyArray1<'_, f64>>,
    clipping_threshold: f64,
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    if score == "CVaR" && external_predictions.is_some() {
        return Err(PyValueError::new_err(
            "external predictions are not available for CVaR (as in DoubleML)",
        ));
    }
    let data = attach_external(
        build_data(y, d, x, z, None, None, None)?,
        external_predictions,
    );
    let ml_m = to_classifier(ml_m)?;
    // A macro rather than a function: `Pq`, `Lpq` and `Cvar` are different
    // types that share these builder methods by name only.
    macro_rules! configure {
        ($model:expr) => {
            $model
                .treatment(treatment)
                .quantile(quantile)
                .clipping_threshold(clipping_threshold)
                .calibration(parse_calibration(calibration)?)
                .normalize_ipw(normalize_ipw)
                .n_folds(n_folds)
                .n_rep(n_rep)
                .seed(seed)
        };
    }
    match score {
        "PQ" => run_model!(
            py,
            configure!(Pq::from_boxed(to_classifier(ml_g)?, ml_m)),
            data,
            splits
        ),
        "LPQ" => run_model!(
            py,
            configure!(Lpq::from_boxed(to_classifier(ml_g)?, ml_m)),
            data,
            splits
        ),
        "CVaR" => run_model!(
            py,
            configure!(Cvar::from_boxed(to_regressor(ml_g)?, ml_m)),
            data,
            splits
        ),
        other => Err(PyValueError::new_err(format!(
            "unknown quantile score {other:?} (use 'PQ', 'LPQ' or 'CVaR')"
        ))),
    }
}

/// Treatment effects on quantiles or CVaRs (returned by `fit_qte`).
#[pyclass(name = "QteResult", module = "dml_rs", frozen)]
pub struct PyQteResult {
    inner: QteFit,
}

#[pymethods]
impl PyQteResult {
    #[getter]
    fn quantiles(&self) -> Vec<f64> {
        self.inner.quantiles().to_vec()
    }

    /// The effect at every quantile, with joint inference.
    #[getter]
    fn effects(&self) -> PyDmlResult {
        PyDmlResult::from_fit(self.inner.effects().clone())
    }

    /// Potential quantiles of treatment 0 and 1, ordered `q1 (d=0), q1 (d=1), q2 (d=0), ...`.
    #[getter]
    fn potential(&self) -> PyDmlResult {
        PyDmlResult::from_fit(self.inner.potential().clone())
    }
}

/// Quantile (`"PQ"`), local quantile (`"LPQ"`, needs `z`) or CVaR (`"CVaR"`)
/// treatment effects at several quantiles, on shared sample splits.
#[pyfunction]
#[pyo3(signature = (y, d, x, ml_g, ml_m, quantiles, score = "PQ", z = None,
                    clipping_threshold = 1e-2, calibration = None, normalize_ipw = true, n_folds = 5, n_rep = 1,
                    seed = 0, splits = None))]
#[allow(clippy::too_many_arguments)]
fn fit_qte(
    py: Python<'_>,
    y: PyReadonlyArray1<'_, f64>,
    d: PyReadonlyArray2<'_, f64>,
    x: PyReadonlyArray2<'_, f64>,
    ml_g: &Bound<'_, PyAny>,
    ml_m: &Bound<'_, PyAny>,
    quantiles: Vec<f64>,
    score: &str,
    z: Option<PyReadonlyArray1<'_, f64>>,
    clipping_threshold: f64,
    calibration: Option<&str>,
    normalize_ipw: bool,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
) -> PyResult<PyQteResult> {
    let data = build_data(y, d, x, z, None, None, None)?;
    let (g_classifier, g_regressor) = if score == "CVaR" {
        (None, Some(to_regressor(ml_g)?))
    } else {
        (Some(to_classifier(ml_g)?), None)
    };
    let model = Qte::from_boxed(
        score,
        g_classifier,
        g_regressor,
        to_classifier(ml_m)?,
        quantiles,
    )
    .map_err(to_py_err)?
    .clipping_threshold(clipping_threshold)
    .calibration(parse_calibration(calibration)?)
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
    result.map(|inner| PyQteResult { inner }).map_err(to_py_err)
}

/// Propensity calibration: `None`, `"isotonic"` or `"isotonic_cv"` (cross-fitted).
pub(crate) fn parse_calibration(calibration: Option<&str>) -> PyResult<dml_core::Calibration> {
    match calibration {
        None => Ok(dml_core::Calibration::None),
        Some("isotonic") => Ok(dml_core::Calibration::Isotonic),
        Some("isotonic_cv") => Ok(dml_core::Calibration::IsotonicCrossFitted),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown calibration {other:?} (use None, 'isotonic' or 'isotonic_cv')"
        ))),
    }
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
                    in_sample_normalization = true, clipping_threshold = 1e-2, calibration = None, n_folds = 5,
                    n_rep = 1, seed = 0, splits = None, treatment_names = None, cluster = None, external_predictions = None))]
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
    calibration: Option<&str>,
    n_folds: usize,
    n_rep: usize,
    seed: u64,
    splits: Option<PySplits>,
    treatment_names: Option<Vec<String>>,
    cluster: Option<PyReadonlyArray2<'_, i64>>,
    external_predictions: Option<PyExternal<'_>>,
) -> PyResult<PyDmlResult> {
    let score = did_score(score)?;
    let ml_m = ml_m.map(to_classifier).transpose()?;
    let cross_section = t.is_some();
    let data = build_data(y, d, x, None, t, treatment_names, cluster)?;
    let data = attach_external(data, external_predictions);
    if cross_section {
        let model = DidCs::from_boxed(to_regressor(ml_g)?, ml_m)
            .score(score)
            .in_sample_normalization(in_sample_normalization)
            .clipping_threshold(clipping_threshold)
            .calibration(parse_calibration(calibration)?)
            .n_folds(n_folds)
            .n_rep(n_rep)
            .seed(seed);
        run_model!(py, model, data, splits)
    } else {
        let model = Did::from_boxed(to_regressor(ml_g)?, ml_m)
            .score(score)
            .in_sample_normalization(in_sample_normalization)
            .clipping_threshold(clipping_threshold)
            .calibration(parse_calibration(calibration)?)
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
        concat, fit_apo, fit_apos, fit_cate, fit_did, fit_iivm, fit_irm, fit_lplr, fit_pliv,
        fit_plpr, fit_plr, fit_qte, fit_quantile, fit_rdflex, fit_ssm, fit_staggered_did, simulate,
        PyAposResult, PyBlp, PyBootstrap, PyCateResult, PyDmlResult, PyGradientBoostingClassifier,
        PyGradientBoostingRegressor, PyGridSearchClassifier, PyGridSearchRegressor, PyLasso,
        PyLassoCV, PyLogisticRegression, PyLogisticRegressionCV, PyOls, PyPolicyTree, PyQteResult,
        PyRandomForestClassifier, PyRandomForestRegressor, PyRdFlexResult, PySensitivityBenchmark,
        PySensitivityResult, PyStaggeredResult,
    };

    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        use pyo3::types::PyModuleMethods;
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}

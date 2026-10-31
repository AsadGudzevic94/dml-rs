//! Learners exposed to Python, and adapters that let any scikit-learn style
//! estimator act as a Rust [`Regressor`] / [`Classifier`].

use dml_core::{
    Classifier, CvFolds, CvScoring, DmlError, EarlyStopping, GradientBoostingClassifier,
    GradientBoostingRegressor, GridSearchClassifier, GridSearchRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, Lasso, LassoCV,
    LogisticRegression, LogisticRegressionCV, MaxFeatures, Ols, RandomForestClassifier,
    RandomForestRegressor, Regressor,
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;

use crate::errors::to_py_err;

// ---------------------------------------------------------------------------
// Built-in Rust learners. Cross-fitting with these never touches Python, so
// it runs fully in parallel with the GIL released.
//
// These are the native classes (`dml_rs._dml_rs.Ols`, ...). The public
// `dml_rs.Ols` etc. are scikit-learn estimators (`python/dml_rs/_learners.py`)
// that build one of these from their parameters.
// ---------------------------------------------------------------------------

type Weights<'py> = Option<PyReadonlyArray1<'py, f64>>;

/// `fit`, or `fit_weighted` with `sample_weight`, with the GIL released.
fn fit_regressor<R: Regressor>(
    py: Python<'_>,
    model: &mut R,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    sample_weight: Weights<'_>,
) -> PyResult<()> {
    let (x, y) = (x.as_array().to_owned(), y.as_array().to_owned());
    let w = sample_weight.map(|w| w.as_array().to_owned());
    py.detach(|| match &w {
        Some(w) => model.fit_weighted(x.view(), y.view(), w.view()),
        None => model.fit(x.view(), y.view()),
    })
    .map_err(to_py_err)
}

/// [`fit_regressor`] for classifiers.
fn fit_classifier<C: Classifier>(
    py: Python<'_>,
    model: &mut C,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    sample_weight: Weights<'_>,
) -> PyResult<()> {
    let (x, y) = (x.as_array().to_owned(), y.as_array().to_owned());
    let w = sample_weight.map(|w| w.as_array().to_owned());
    py.detach(|| match &w {
        Some(w) => model.fit_weighted(x.view(), y.view(), w.view()),
        None => model.fit(x.view(), y.view()),
    })
    .map_err(to_py_err)
}

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
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (ridge = 0.0))]
    fn new(ridge: f64) -> Self {
        Self {
            inner: Ols::with_ridge(ridge),
            ridge,
        }
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_regressor(py, &mut slf.inner, x, y, sample_weight)?;
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

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self.inner.coef().ok_or_else(|| not_fitted("Ols"))?;
        Ok(c.clone().into_pyarray(py))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.coef().ok_or_else(|| not_fitted("Ols"))?;
        Ok(self.inner.intercept())
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
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (alpha = 1.0, tol = 1e-4, max_iter = 100_000))]
    fn new(alpha: f64, tol: f64, max_iter: usize) -> Self {
        Self {
            inner: Lasso::new(alpha).tol(tol).max_iter(max_iter),
            alpha,
        }
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_regressor(py, &mut slf.inner, x, y, sample_weight)?;
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

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self.inner.coef().ok_or_else(|| not_fitted("Lasso"))?;
        Ok(c.clone().into_pyarray(py))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.coef().ok_or_else(|| not_fitted("Lasso"))?;
        Ok(self.inner.intercept())
    }

    /// Coordinate-descent sweeps of the last fit.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner.coef().ok_or_else(|| not_fitted("Lasso"))?;
        Ok(self.inner.n_iter())
    }

    fn __repr__(&self) -> String {
        format!("Lasso(alpha={})", self.alpha)
    }
}

/// `cv` from Python: a number of folds or a list of `(train, test)` index
/// arrays (what a scikit-learn splitter's `split` yields).
fn parse_cv(cv: Option<&Bound<'_, PyAny>>) -> PyResult<(CvFolds, String)> {
    let Some(cv) = cv else {
        return Ok((CvFolds::K(5), "5".into()));
    };
    if let Ok(k) = cv.extract::<usize>() {
        return Ok((CvFolds::K(k), k.to_string()));
    }
    let folds: Vec<(Vec<usize>, Vec<usize>)> = cv.extract().map_err(|_| {
        PyValueError::new_err("cv must be an int or a list of (train, test) index arrays")
    })?;
    let repr = format!("<{} splits>", folds.len());
    Ok((CvFolds::Explicit(folds), repr))
}

/// Apply parsed [`CvFolds`] to a learner builder with `cv` / `folds` methods.
macro_rules! with_cv {
    ($builder:expr, $folds:expr) => {
        match $folds {
            CvFolds::K(k) => $builder.cv(k),
            CvFolds::Explicit(f) => $builder.folds(f),
        }
    };
}

fn not_fitted(name: &str) -> PyErr {
    PyAttributeError::new_err(format!("{name} is not fitted yet; call fit() first"))
}

/// Lasso with the penalty chosen by cross-validation, as scikit-learn's
/// `LassoCV`: `alphas` is the number of penalties on the automatic grid
/// (from `alpha_max` down to `eps * alpha_max`) or a list of penalties.
#[pyclass(name = "LassoCV", module = "dml_rs")]
#[derive(Clone)]
pub struct PyLassoCV {
    pub(crate) inner: LassoCV,
    repr: String,
}

#[pymethods]
impl PyLassoCV {
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (alphas = None, eps = 1e-3, cv = None, tol = 1e-4, max_iter = 100_000))]
    fn new(
        alphas: Option<&Bound<'_, PyAny>>,
        eps: f64,
        cv: Option<&Bound<'_, PyAny>>,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        let (folds, cv) = parse_cv(cv)?;
        let mut inner = with_cv!(LassoCV::new(), folds)
            .eps(eps)
            .tol(tol)
            .max_iter(max_iter);
        let mut grid = "100".to_string();
        if let Some(a) = alphas {
            if let Ok(n) = a.extract::<usize>() {
                inner = inner.n_alphas(n);
                grid = n.to_string();
            } else {
                let list: Vec<f64> = a.extract().map_err(|_| {
                    PyValueError::new_err("alphas must be an int or a sequence of floats")
                })?;
                grid = format!("{list:?}");
                inner = inner.alphas(list);
            }
        }
        Ok(Self {
            inner,
            repr: format!("LassoCV(alphas={grid}, eps={eps}, cv={cv})"),
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_regressor(py, &mut slf.inner, x, y, sample_weight)?;
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

    /// The chosen penalty.
    #[getter]
    fn alpha_(&self) -> PyResult<f64> {
        self.inner.alpha().ok_or_else(|| not_fitted("LassoCV"))
    }

    /// The candidate penalties, in decreasing order.
    #[getter]
    fn alphas_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let a = self
            .inner
            .alphas_path()
            .ok_or_else(|| not_fitted("LassoCV"))?;
        Ok(Array1::from(a.to_vec()).into_pyarray(py))
    }

    /// Test MSE per penalty and fold, shape `(n_alphas, cv)`.
    #[getter]
    fn mse_path_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = self.inner.mse_path().ok_or_else(|| not_fitted("LassoCV"))?;
        Ok(m.clone().into_pyarray(py))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self.inner.coef().ok_or_else(|| not_fitted("LassoCV"))?;
        Ok(c.clone().into_pyarray(py))
    }

    /// Coordinate-descent sweeps of the refit at `alpha_`.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner.n_iter().ok_or_else(|| not_fitted("LassoCV"))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner.alpha().ok_or_else(|| not_fitted("LassoCV"))?;
        Ok(self.inner.intercept())
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

/// L2-penalised logistic regression with `C` chosen by stratified
/// cross-validation, as scikit-learn's `LogisticRegressionCV`: `Cs` is the
/// number of values between 1e-4 and 1e4 or a list of values; `scoring` is
/// `"neg_log_loss"` (default, scikit-learn's default from 1.11) or `"accuracy"`.
#[pyclass(name = "LogisticRegressionCV", module = "dml_rs")]
#[derive(Clone)]
pub struct PyLogisticRegressionCV {
    pub(crate) inner: LogisticRegressionCV,
    repr: String,
}

#[pymethods]
impl PyLogisticRegressionCV {
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (Cs = None, cv = None, scoring = "neg_log_loss", tol = 1e-10, max_iter = 100))]
    #[allow(non_snake_case)] // keep scikit-learn's parameter name
    fn new(
        Cs: Option<&Bound<'_, PyAny>>,
        cv: Option<&Bound<'_, PyAny>>,
        scoring: &str,
        tol: f64,
        max_iter: usize,
    ) -> PyResult<Self> {
        let score = match scoring {
            "neg_log_loss" | "log_loss" => CvScoring::LogLoss,
            "accuracy" => CvScoring::Accuracy,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown scoring {other:?} (use 'neg_log_loss' or 'accuracy')"
                )))
            }
        };
        let (folds, cv) = parse_cv(cv)?;
        let mut inner = with_cv!(LogisticRegressionCV::new(), folds)
            .scoring(score)
            .tol(tol)
            .max_iter(max_iter);
        let mut grid = "10".to_string();
        if let Some(c) = Cs {
            if let Ok(n) = c.extract::<usize>() {
                inner = inner.n_cs(n);
                grid = n.to_string();
            } else {
                let list: Vec<f64> = c.extract().map_err(|_| {
                    PyValueError::new_err("Cs must be an int or a sequence of floats")
                })?;
                grid = format!("{list:?}");
                inner = inner.cs(list);
            }
        }
        Ok(Self {
            inner,
            repr: format!("LogisticRegressionCV(Cs={grid}, cv={cv}, scoring={scoring:?})"),
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_classifier(py, &mut slf.inner, x, y, sample_weight)?;
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

    /// The chosen `C`.
    #[getter]
    #[allow(non_snake_case)]
    fn C_(&self) -> PyResult<f64> {
        self.inner
            .c()
            .ok_or_else(|| not_fitted("LogisticRegressionCV"))
    }

    /// The candidate values of `C`.
    #[getter]
    #[allow(non_snake_case)]
    fn Cs_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        Array1::from(self.inner.cs_path().to_vec()).into_pyarray(py)
    }

    /// Score per fold and `C` (higher is better), shape `(cv, n_Cs)`.
    #[getter]
    fn scores_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let s = self
            .inner
            .scores()
            .ok_or_else(|| not_fitted("LogisticRegressionCV"))?;
        Ok(s.clone().into_pyarray(py))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self
            .inner
            .coef()
            .ok_or_else(|| not_fitted("LogisticRegressionCV"))?;
        Ok(c.clone().into_pyarray(py))
    }

    /// Newton steps of the refit at `C_`.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner
            .n_iter()
            .ok_or_else(|| not_fitted("LogisticRegressionCV"))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner
            .c()
            .ok_or_else(|| not_fitted("LogisticRegressionCV"))?;
        Ok(self.inner.intercept())
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
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
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (C = 1.0, tol = 1e-10, max_iter = 100))]
    #[allow(non_snake_case)] // keep scikit-learn's parameter name
    fn new(C: f64, tol: f64, max_iter: usize) -> Self {
        Self {
            inner: LogisticRegression::new(C).tol(tol).max_iter(max_iter),
            c: C,
        }
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_classifier(py, &mut slf.inner, x, y, sample_weight)?;
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

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = self
            .inner
            .coef()
            .ok_or_else(|| not_fitted("LogisticRegression"))?;
        Ok(c.clone().into_pyarray(py))
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        self.inner
            .coef()
            .ok_or_else(|| not_fitted("LogisticRegression"))?;
        Ok(self.inner.intercept())
    }

    /// Newton steps of the last fit.
    #[getter]
    fn n_iter_(&self) -> PyResult<usize> {
        self.inner
            .coef()
            .ok_or_else(|| not_fitted("LogisticRegression"))?;
        Ok(self.inner.n_iter())
    }

    fn __repr__(&self) -> String {
        format!("LogisticRegression(C={})", self.c)
    }
}

/// scikit-learn style `max_features`: `None` (all), `"sqrt"`, `"log2"`, a
/// fraction in `(0, 1]` or a number of features.
fn parse_max_features(obj: &Bound<'_, PyAny>, default: MaxFeatures) -> PyResult<MaxFeatures> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
    if obj.is_none() {
        return Ok(default);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return match s.extract::<String>()?.as_str() {
            "sqrt" => Ok(MaxFeatures::Sqrt),
            "log2" => Ok(MaxFeatures::Log2),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "max_features must be None, 'sqrt', 'log2', a float or an int, got {other:?}"
            ))),
        };
    }
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        return Ok(MaxFeatures::Count(obj.extract()?));
    }
    if obj.is_instance_of::<PyFloat>() {
        return Ok(MaxFeatures::Fraction(obj.extract()?));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "max_features must be None, 'sqrt', 'log2', a float or an int",
    ))
}

/// Forest settings as given from Python, shared by both forest classes.
#[derive(Clone, Copy)]
struct ForestArgs {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    bootstrap: bool,
    max_samples: Option<f64>,
    random_state: u64,
}

/// Apply [`ForestArgs`] to either forest type (they share builder methods).
macro_rules! configure_forest {
    ($forest:expr, $args:expr) => {
        $forest
            .n_estimators($args.n_estimators)
            .max_depth($args.max_depth)
            .min_samples_split($args.min_samples_split)
            .min_samples_leaf($args.min_samples_leaf)
            .max_features($args.max_features)
            .bootstrap($args.bootstrap)
            .max_samples($args.max_samples)
            .seed($args.random_state)
    };
}

/// Generate a forest pyclass: constructor with scikit-learn's parameter names,
/// `fit` (GIL released while the trees grow), a prediction method and
/// `feature_importances_`.
macro_rules! forest_pyclass {
    ($py_ty:ident, $rust_ty:ty, $name:literal, $fit:ident, $default_features:expr, $predict:ident, $predict_doc:literal) => {
        #[doc = concat!("Random forest implemented in Rust; trees are grown in parallel without the GIL. ",
                        "Parameters follow scikit-learn's `", $name, "`.")]
        #[pyclass(name = $name, module = "dml_rs")]
        #[derive(Clone)]
        pub struct $py_ty {
            pub(crate) inner: $rust_ty,
            args: ForestArgs,
        }

        #[pymethods]
        impl $py_ty {
            /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            #[new]
            #[pyo3(signature = (n_estimators = 100, max_depth = None, min_samples_split = 2,
                                min_samples_leaf = 1, max_features = None, bootstrap = true,
                                max_samples = None, random_state = 0))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                n_estimators: usize,
                max_depth: Option<usize>,
                min_samples_split: usize,
                min_samples_leaf: usize,
                max_features: Option<&Bound<'_, PyAny>>,
                bootstrap: bool,
                max_samples: Option<f64>,
                random_state: u64,
            ) -> PyResult<Self> {
                let max_features = match max_features {
                    Some(obj) => parse_max_features(obj, $default_features)?,
                    None => $default_features,
                };
                let args = ForestArgs {
                    n_estimators,
                    max_depth,
                    min_samples_split,
                    min_samples_leaf,
                    max_features,
                    bootstrap,
                    max_samples,
                    random_state,
                };
                Ok(Self {
                    inner: configure_forest!(<$rust_ty>::new(), args),
                    args,
                })
            }

            #[pyo3(signature = (x, y, sample_weight = None))]
            fn fit<'py>(
                mut slf: PyRefMut<'py, Self>,
                x: PyReadonlyArray2<'py, f64>,
                y: PyReadonlyArray1<'py, f64>,
                sample_weight: Weights<'py>,
            ) -> PyResult<PyRefMut<'py, Self>> {
                let py = slf.py();
                $fit(py, &mut slf.inner, x, y, sample_weight)?;
                Ok(slf)
            }

            #[doc = $predict_doc]
            fn $predict<'py>(
                &self,
                py: Python<'py>,
                x: PyReadonlyArray2<'py, f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                Ok(self.inner.$predict(x.as_array()).map_err(to_py_err)?.into_pyarray(py))
            }

            /// Mean decrease in impurity per feature (sums to one), after `fit`.
            #[getter]
            fn feature_importances_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
                self.inner.feature_importances().map(|a| a.clone().into_pyarray(py))
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(n_estimators={}, min_samples_leaf={}, random_state={})",
                    $name, self.args.n_estimators, self.args.min_samples_leaf, self.args.random_state
                )
            }
        }
    };
}

forest_pyclass!(
    PyRandomForestRegressor,
    RandomForestRegressor,
    "RandomForestRegressor",
    fit_regressor,
    MaxFeatures::All,
    predict,
    "Predicted values."
);

forest_pyclass!(
    PyRandomForestClassifier,
    RandomForestClassifier,
    "RandomForestClassifier",
    fit_classifier,
    MaxFeatures::Sqrt,
    predict_proba,
    "Probability of class 1 (a 1-D array, unlike scikit-learn's two columns)."
);

/// Generate a gradient-boosting pyclass with scikit-learn's parameter names.
macro_rules! boosting_pyclass {
    ($py_ty:ident, $rust_ty:ty, $name:literal, $fit:ident, $predict:ident, $predict_doc:literal) => {
        #[doc = concat!("Gradient boosting implemented in Rust (`fit` releases the GIL). ",
                        "Parameters follow scikit-learn's `", $name, "`.")]
        #[pyclass(name = $name, module = "dml_rs")]
        #[derive(Clone)]
        pub struct $py_ty {
            pub(crate) inner: $rust_ty,
            n_estimators: usize,
            learning_rate: f64,
            max_depth: Option<usize>,
        }

        #[pymethods]
        impl $py_ty {
            /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            #[new]
            #[pyo3(signature = (n_estimators = 100, learning_rate = 0.1, max_depth = Some(3),
                                min_samples_split = 2, min_samples_leaf = 1, max_features = None,
                                subsample = 1.0, random_state = 0))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                n_estimators: usize,
                learning_rate: f64,
                max_depth: Option<usize>,
                min_samples_split: usize,
                min_samples_leaf: usize,
                max_features: Option<&Bound<'_, PyAny>>,
                subsample: f64,
                random_state: u64,
            ) -> PyResult<Self> {
                let max_features = match max_features {
                    Some(obj) => parse_max_features(obj, MaxFeatures::All)?,
                    None => MaxFeatures::All,
                };
                Ok(Self {
                    inner: <$rust_ty>::new()
                        .n_estimators(n_estimators)
                        .learning_rate(learning_rate)
                        .max_depth(max_depth)
                        .min_samples_split(min_samples_split)
                        .min_samples_leaf(min_samples_leaf)
                        .max_features(max_features)
                        .subsample(subsample)
                        .seed(random_state),
                    n_estimators,
                    learning_rate,
                    max_depth,
                })
            }

            #[pyo3(signature = (x, y, sample_weight = None))]
            fn fit<'py>(
                mut slf: PyRefMut<'py, Self>,
                x: PyReadonlyArray2<'py, f64>,
                y: PyReadonlyArray1<'py, f64>,
                sample_weight: Weights<'py>,
            ) -> PyResult<PyRefMut<'py, Self>> {
                let py = slf.py();
                $fit(py, &mut slf.inner, x, y, sample_weight)?;
                Ok(slf)
            }

            #[doc = $predict_doc]
            fn $predict<'py>(
                &self,
                py: Python<'py>,
                x: PyReadonlyArray2<'py, f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                Ok(self.inner.$predict(x.as_array()).map_err(to_py_err)?.into_pyarray(py))
            }

            /// Total impurity reduction per feature over all trees (sums to one).
            #[getter]
            fn feature_importances_<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
                self.inner.feature_importances().map(|a| a.clone().into_pyarray(py))
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(n_estimators={}, learning_rate={}, max_depth={:?})",
                    $name, self.n_estimators, self.learning_rate, self.max_depth
                )
            }
        }
    };
}

boosting_pyclass!(
    PyGradientBoostingRegressor,
    GradientBoostingRegressor,
    "GradientBoostingRegressor",
    fit_regressor,
    predict,
    "Predicted values."
);

boosting_pyclass!(
    PyGradientBoostingClassifier,
    GradientBoostingClassifier,
    "GradientBoostingClassifier",
    fit_classifier,
    predict_proba,
    "Probability of class 1 (a 1-D array, unlike scikit-learn's two columns)."
);

/// `early_stopping` from Python: `"auto"`, `True` or `False`.
fn parse_early_stopping(obj: &Bound<'_, PyAny>) -> PyResult<EarlyStopping> {
    if let Ok(flag) = obj.extract::<bool>() {
        return Ok(if flag {
            EarlyStopping::Enabled
        } else {
            EarlyStopping::Disabled
        });
    }
    match obj.extract::<String>().as_deref() {
        Ok("auto") => Ok(EarlyStopping::Auto),
        _ => Err(PyValueError::new_err(
            "early_stopping must be 'auto', True or False",
        )),
    }
}

/// Generate a histogram gradient boosting pyclass with scikit-learn's
/// parameters, `fit` (GIL released, optional `sample_weight`), a prediction
/// method and `n_iter_`.
macro_rules! hist_pyclass {
    ($py_ty:ident, $rust_ty:ty, $name:literal, $fit:ident, $predict:ident, $predict_doc:literal) => {
        #[doc = concat!("Histogram-based gradient boosting implemented in Rust (`fit` releases the GIL). ",
                        "Parameters and results follow scikit-learn's `", $name, "`.")]
        #[pyclass(name = $name, module = "dml_rs")]
        #[derive(Clone)]
        pub struct $py_ty {
            pub(crate) inner: $rust_ty,
            learning_rate: f64,
            max_iter: usize,
            max_leaf_nodes: Option<usize>,
        }

        #[pymethods]
        impl $py_ty {
            /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
            fn __copy__(&self) -> Self {
                self.clone()
            }

            fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
                self.clone()
            }

            #[new]
            #[pyo3(signature = (learning_rate = 0.1, max_iter = 100, max_leaf_nodes = Some(31),
                                max_depth = None, min_samples_leaf = 20, l2_regularization = 0.0,
                                max_features = 1.0, max_bins = 255, early_stopping = None,
                                validation_fraction = Some(0.1), n_iter_no_change = 10, tol = 1e-7,
                                random_state = 0))]
            #[allow(clippy::too_many_arguments)]
            fn new(
                learning_rate: f64,
                max_iter: usize,
                max_leaf_nodes: Option<usize>,
                max_depth: Option<usize>,
                min_samples_leaf: usize,
                l2_regularization: f64,
                max_features: f64,
                max_bins: usize,
                early_stopping: Option<&Bound<'_, PyAny>>,
                validation_fraction: Option<f64>,
                n_iter_no_change: usize,
                tol: f64,
                random_state: u32,
            ) -> PyResult<Self> {
                let early_stopping = match early_stopping {
                    Some(obj) => parse_early_stopping(obj)?,
                    None => EarlyStopping::Auto,
                };
                Ok(Self {
                    inner: <$rust_ty>::new()
                        .learning_rate(learning_rate)
                        .max_iter(max_iter)
                        .max_leaf_nodes(max_leaf_nodes)
                        .max_depth(max_depth)
                        .min_samples_leaf(min_samples_leaf)
                        .l2_regularization(l2_regularization)
                        .max_features(max_features)
                        .max_bins(max_bins)
                        .early_stopping(early_stopping)
                        .validation_fraction(validation_fraction)
                        .n_iter_no_change(n_iter_no_change)
                        .tol(tol)
                        .random_state(random_state),
                    learning_rate,
                    max_iter,
                    max_leaf_nodes,
                })
            }

            #[pyo3(signature = (x, y, sample_weight = None))]
            fn fit<'py>(
                mut slf: PyRefMut<'py, Self>,
                x: PyReadonlyArray2<'py, f64>,
                y: PyReadonlyArray1<'py, f64>,
                sample_weight: Weights<'py>,
            ) -> PyResult<PyRefMut<'py, Self>> {
                let py = slf.py();
                $fit(py, &mut slf.inner, x, y, sample_weight)?;
                Ok(slf)
            }

            #[doc = $predict_doc]
            fn $predict<'py>(
                &self,
                py: Python<'py>,
                x: PyReadonlyArray2<'py, f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                Ok(self.inner.$predict(x.as_array()).map_err(to_py_err)?.into_pyarray(py))
            }

            /// Number of fitted trees (fewer than `max_iter` after early stopping).
            #[getter]
            fn n_iter_(&self) -> usize {
                self.inner.n_iter()
            }

            /// Early-stopping scores (negative loss) from the baseline on, as
            /// scikit-learn's `validation_score_` (or `train_score_` without a
            /// validation split); empty without early stopping.
            #[getter]
            fn validation_score_<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                self.inner.validation_scores().to_vec().into_pyarray(py)
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(learning_rate={}, max_iter={}, max_leaf_nodes={:?})",
                    $name, self.learning_rate, self.max_iter, self.max_leaf_nodes
                )
            }
        }
    };
}

hist_pyclass!(
    PyHistGradientBoostingRegressor,
    HistGradientBoostingRegressor,
    "HistGradientBoostingRegressor",
    fit_regressor,
    predict,
    "Predicted values."
);

hist_pyclass!(
    PyHistGradientBoostingClassifier,
    HistGradientBoostingClassifier,
    "HistGradientBoostingClassifier",
    fit_classifier,
    predict_proba,
    "Probability of class 1 (a 1-D array, unlike scikit-learn's two columns)."
);

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

    /// Calls `fit(X, y, sample_weight=w)`.
    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> dml_core::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            let est = clone_estimator(self.template.bind(py))?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("sample_weight", w.to_owned().into_pyarray(py))?;
            est.call_method(
                "fit",
                (x.to_owned().into_pyarray(py), y.to_owned().into_pyarray(py)),
                Some(&kwargs),
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

    /// Calls `fit(X, y, sample_weight=w)`.
    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> dml_core::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            let est = clone_estimator(self.template.bind(py))?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("sample_weight", w.to_owned().into_pyarray(py))?;
            est.call_method(
                "fit",
                (x.to_owned().into_pyarray(py), y.to_owned().into_pyarray(py)),
                Some(&kwargs),
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

/// Grid search over regression learners: the candidate with the smallest
/// cross-validated mean squared error is refitted on all data.
#[pyclass(name = "GridSearchRegressor", module = "dml_rs")]
#[derive(Clone)]
pub struct PyGridSearchRegressor {
    pub(crate) inner: GridSearchRegressor,
    repr: String,
}

#[pymethods]
impl PyGridSearchRegressor {
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (candidates, cv = None))]
    fn new(candidates: Vec<Bound<'_, PyAny>>, cv: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (folds, cv) = parse_cv(cv)?;
        let boxed = candidates
            .iter()
            .map(to_regressor)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: with_cv!(GridSearchRegressor::new(boxed), folds),
            repr: format!(
                "GridSearchRegressor({} candidates, cv={cv})",
                candidates.len()
            ),
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_regressor(py, &mut slf.inner, x, y, sample_weight)?;
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

    /// Index of the chosen candidate.
    #[getter]
    fn best_index_(&self) -> PyResult<usize> {
        self.inner
            .best_index()
            .ok_or_else(|| PyValueError::new_err("not fitted"))
    }

    /// Cross-validated mean squared error of every candidate.
    #[getter]
    fn cv_scores_(&self) -> Vec<f64> {
        self.inner.cv_scores().to_vec()
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

/// Grid search over classifiers: the candidate with the smallest
/// cross-validated log loss is refitted on all data.
#[pyclass(name = "GridSearchClassifier", module = "dml_rs")]
#[derive(Clone)]
pub struct PyGridSearchClassifier {
    pub(crate) inner: GridSearchClassifier,
    repr: String,
}

#[pymethods]
impl PyGridSearchClassifier {
    /// A copy (`copy.copy` / `copy.deepcopy`; fitted state included).
    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    #[new]
    #[pyo3(signature = (candidates, cv = None))]
    fn new(candidates: Vec<Bound<'_, PyAny>>, cv: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (folds, cv) = parse_cv(cv)?;
        let boxed = candidates
            .iter()
            .map(to_classifier)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: with_cv!(GridSearchClassifier::new(boxed), folds),
            repr: format!(
                "GridSearchClassifier({} candidates, cv={cv})",
                candidates.len()
            ),
        })
    }

    #[pyo3(signature = (x, y, sample_weight = None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        sample_weight: Weights<'py>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        fit_classifier(py, &mut slf.inner, x, y, sample_weight)?;
        Ok(slf)
    }

    /// Class-1 probabilities.
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

    #[getter]
    fn best_index_(&self) -> PyResult<usize> {
        self.inner
            .best_index()
            .ok_or_else(|| PyValueError::new_err("not fitted"))
    }

    /// Cross-validated log loss of every candidate.
    #[getter]
    fn cv_scores_(&self) -> Vec<f64> {
        self.inner.cv_scores().to_vec()
    }

    fn __repr__(&self) -> String {
        self.repr.clone()
    }
}

/// The native learner behind a `dml_rs` scikit-learn estimator: its
/// `_dml_native()` builds one from the estimator's parameters, or returns
/// `None` when it cannot (e.g. `cv` is a splitter that needs the data), and
/// the estimator is then fitted through Python like any other.
fn native_learner<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = obj.py();
    if !obj.hasattr(pyo3::intern!(py, "_dml_native"))? {
        return Ok(None);
    }
    let native = obj.call_method0(pyo3::intern!(py, "_dml_native"))?;
    Ok((!native.is_none()).then_some(native))
}

/// Built-in learner if `obj` is one, otherwise wrap the Python estimator.
pub fn to_regressor(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Regressor>> {
    if let Some(native) = native_learner(obj)? {
        return to_regressor(&native);
    }
    if let Ok(o) = obj.cast::<PyGridSearchRegressor>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyOls>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyLasso>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyLassoCV>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyRandomForestRegressor>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyGradientBoostingRegressor>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyHistGradientBoostingRegressor>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<crate::forest::PyRegressionForest>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    Ok(Box::new(PyRegressor::new(obj)?))
}

pub fn to_classifier(obj: &Bound<'_, PyAny>) -> PyResult<Box<dyn Classifier>> {
    if let Some(native) = native_learner(obj)? {
        return to_classifier(&native);
    }
    if let Ok(o) = obj.cast::<PyGridSearchClassifier>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyLogisticRegression>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyLogisticRegressionCV>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyRandomForestClassifier>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyGradientBoostingClassifier>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    if let Ok(o) = obj.cast::<PyHistGradientBoostingClassifier>() {
        return Ok(Box::new(o.borrow().inner.clone()));
    }
    Ok(Box::new(PyClassifier::new(obj)?))
}

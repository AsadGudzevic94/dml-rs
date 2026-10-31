//! R bindings for dml-core (double / debiased machine learning).
//!
//! The R functions in `R/` validate their input and call the functions below
//! through `.Call` (see `R/extendr-wrappers.R`). A fitted model lives in R as
//! an external pointer to a [`DmlFit`]; the accessors read results from it.

use std::collections::HashMap;

use dml_core::learner::{Classifier, Regressor};
use dml_core::{
    BootstrapMethod, DidScore, DmlData, DmlFit, ForestSettings, GradientBoostingClassifier,
    GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, Iivm,
    Irm, IrmScore, Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, MaxFeatures, Ols,
    Pliv, Plr, PlrScore, RandomForestClassifier, RandomForestRegressor, RegressionForest,
    SensitivityParams,
};
use extendr_api::prelude::*;
use extendr_api::Result;
use ndarray::{Array1, Array2, ShapeBuilder};

fn err(e: impl std::fmt::Display) -> Error {
    Error::Other(e.to_string())
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/// An R numeric matrix (column-major) as an ndarray in standard layout.
fn matrix(m: &RMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_vec((m.nrows(), m.ncols()).f(), m.data().to_vec())
        .expect("R matrix dimensions")
        .as_standard_layout()
        .to_owned()
}

fn data(y: &[f64], d: &RMatrix<f64>, x: &RMatrix<f64>, names: Vec<String>) -> Result<DmlData> {
    DmlData::with_treatments(Array1::from(y.to_vec()), matrix(d), matrix(x))
        .and_then(|data| data.with_treatment_names(names))
        .map_err(err)
}

// ---------------------------------------------------------------------------
// Learners: `list(type = "lasso", alpha = 0.1, ...)` from `R/learners.R`
// ---------------------------------------------------------------------------

struct Spec {
    kind: String,
    params: HashMap<String, Robj>,
}

impl Spec {
    fn new(spec: &List) -> Result<Self> {
        let params: HashMap<String, Robj> = spec
            .iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect();
        let kind = params
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| err("a learner needs a `type`"))?
            .to_string();
        Ok(Self { kind, params })
    }

    fn get(&self, key: &str) -> Option<&Robj> {
        self.params.get(key).filter(|v| !v.is_null())
    }

    fn num(&self, key: &str, default: f64) -> Result<f64> {
        match self.get(key) {
            None => Ok(default),
            Some(v) => v
                .as_real()
                .or_else(|| v.as_integer().map(f64::from))
                .ok_or_else(|| err(format!("learner parameter `{key}` must be a number"))),
        }
    }

    fn count(&self, key: &str, default: usize) -> Result<usize> {
        let v = self.num(key, default as f64)?;
        if v < 0.0 || v.fract() != 0.0 {
            return Err(err(format!(
                "learner parameter `{key}` must be a whole number"
            )));
        }
        Ok(v as usize)
    }

    /// A count, or `None` for `NULL` (no limit).
    fn opt_count(&self, key: &str, default: Option<usize>) -> Result<Option<usize>> {
        if !self.params.contains_key(key) {
            return Ok(default);
        }
        match self.get(key) {
            None => Ok(None),
            Some(_) => self.count(key, 0).map(Some),
        }
    }

    fn seed(&self) -> Result<u64> {
        self.count("random_state", 0).map(|s| s as u64)
    }

    fn max_features(&self, default: MaxFeatures) -> Result<MaxFeatures> {
        match self.get("max_features") {
            None => Ok(default),
            Some(v) => match (v.as_str(), v.as_real().or_else(|| v.as_integer().map(f64::from))) {
                (Some("sqrt"), _) => Ok(MaxFeatures::Sqrt),
                (Some("log2"), _) => Ok(MaxFeatures::Log2),
                (_, Some(f)) if f > 0.0 && f <= 1.0 && v.as_integer().is_none() => {
                    Ok(MaxFeatures::Fraction(f))
                }
                (_, Some(n)) if n >= 1.0 && n.fract() == 0.0 => Ok(MaxFeatures::Count(n as usize)),
                _ => Err(err(
                    "max_features must be NULL, \"sqrt\", \"log2\", a fraction or a number of features",
                )),
            },
        }
    }
}

fn regressor(spec: &List) -> Result<Box<dyn Regressor>> {
    let s = Spec::new(spec)?;
    Ok(match s.kind.as_str() {
        "ols" => Box::new(Ols::with_ridge(s.num("ridge", 0.0)?)),
        "lasso" => Box::new(
            Lasso::new(s.num("alpha", 1.0)?)
                .tol(s.num("tol", 1e-4)?)
                .max_iter(s.count("max_iter", 100_000)?),
        ),
        "lasso_cv" => Box::new(
            LassoCV::new()
                .n_alphas(s.count("n_alphas", 100)?)
                .eps(s.num("eps", 1e-3)?)
                .cv(s.count("cv", 5)?),
        ),
        "random_forest" => Box::new(
            RandomForestRegressor::new()
                .n_estimators(s.count("n_estimators", 100)?)
                .max_depth(s.opt_count("max_depth", None)?)
                .min_samples_leaf(s.count("min_samples_leaf", 1)?)
                .max_features(s.max_features(MaxFeatures::All)?)
                .seed(s.seed()?),
        ),
        "gradient_boosting" => Box::new(
            GradientBoostingRegressor::new()
                .n_estimators(s.count("n_estimators", 100)?)
                .learning_rate(s.num("learning_rate", 0.1)?)
                .max_depth(s.opt_count("max_depth", Some(3))?)
                .min_samples_leaf(s.count("min_samples_leaf", 1)?)
                .subsample(s.num("subsample", 1.0)?)
                .seed(s.seed()?),
        ),
        "hist_gradient_boosting" => Box::new(
            HistGradientBoostingRegressor::new()
                .max_iter(s.count("max_iter", 100)?)
                .learning_rate(s.num("learning_rate", 0.1)?)
                .max_leaf_nodes(s.opt_count("max_leaf_nodes", Some(31))?)
                .min_samples_leaf(s.count("min_samples_leaf", 20)?)
                .l2_regularization(s.num("l2_regularization", 0.0)?)
                .random_state(s.seed()? as u32),
        ),
        "regression_forest" => Box::new(RegressionForest::new().settings(ForestSettings {
            num_trees: s.count("n_estimators", 2000)?,
            min_node_size: s.count("min_node_size", 5)?,
            seed: s.seed()?,
            ..ForestSettings::default()
        })),
        other => {
            return Err(err(format!(
                "learner {other:?} cannot be used as a regressor"
            )))
        }
    })
}

fn classifier(spec: &List) -> Result<Box<dyn Classifier>> {
    let s = Spec::new(spec)?;
    Ok(match s.kind.as_str() {
        "logistic" => Box::new(LogisticRegression::new(s.num("C", 1.0)?)),
        "logistic_cv" => Box::new(
            LogisticRegressionCV::new()
                .n_cs(s.count("n_cs", 10)?)
                .cv(s.count("cv", 5)?),
        ),
        "random_forest" => Box::new(
            RandomForestClassifier::new()
                .n_estimators(s.count("n_estimators", 100)?)
                .max_depth(s.opt_count("max_depth", None)?)
                .min_samples_leaf(s.count("min_samples_leaf", 1)?)
                .max_features(s.max_features(MaxFeatures::Sqrt)?)
                .seed(s.seed()?),
        ),
        "gradient_boosting" => Box::new(
            GradientBoostingClassifier::new()
                .n_estimators(s.count("n_estimators", 100)?)
                .learning_rate(s.num("learning_rate", 0.1)?)
                .max_depth(s.opt_count("max_depth", Some(3))?)
                .min_samples_leaf(s.count("min_samples_leaf", 1)?)
                .subsample(s.num("subsample", 1.0)?)
                .seed(s.seed()?),
        ),
        "hist_gradient_boosting" => Box::new(
            HistGradientBoostingClassifier::new()
                .max_iter(s.count("max_iter", 100)?)
                .learning_rate(s.num("learning_rate", 0.1)?)
                .max_leaf_nodes(s.opt_count("max_leaf_nodes", Some(31))?)
                .min_samples_leaf(s.count("min_samples_leaf", 20)?)
                .l2_regularization(s.num("l2_regularization", 0.0)?)
                .random_state(s.seed()? as u32),
        ),
        other => {
            return Err(err(format!(
                "learner {other:?} cannot be used as a classifier"
            )))
        }
    })
}

fn optional_list(robj: &Robj) -> Result<Option<List>> {
    if robj.is_null() {
        Ok(None)
    } else {
        List::try_from(robj.clone()).map(Some)
    }
}

fn seed(seed: f64) -> u64 {
    seed as u64
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// Partially linear regression.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_plr(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    names: Vec<String>,
    ml_l: List,
    ml_m: List,
    ml_g: Robj,
    score: &str,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<ExternalPtr<DmlFit>> {
    let score = match score {
        "partialling out" => PlrScore::PartiallingOut,
        "IV-type" => PlrScore::IvType,
        other => return Err(err(format!("unknown PLR score {other:?}"))),
    };
    let mut model = Plr::from_boxed(regressor(&ml_l)?, regressor(&ml_m)?)
        .score(score)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_));
    if let Some(g) = optional_list(&ml_g)? {
        model = model.ml_g_boxed(regressor(&g)?);
    }
    let fit = model.fit(&data(y, &d, &x, names)?).map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// Interactive regression model (binary treatment).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_irm(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    names: Vec<String>,
    ml_g: List,
    ml_m: List,
    score: &str,
    clipping_threshold: f64,
    normalize_ipw: bool,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<ExternalPtr<DmlFit>> {
    let score = match score {
        "ATE" => IrmScore::Ate,
        "ATTE" => IrmScore::Atte,
        other => return Err(err(format!("unknown IRM score {other:?}"))),
    };
    let fit = Irm::from_boxed(regressor(&ml_g)?, classifier(&ml_m)?)
        .score(score)
        .clipping_threshold(clipping_threshold)
        .normalize_ipw(normalize_ipw)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_))
        .fit(&data(y, &d, &x, names)?)
        .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// Partially linear IV regression with one instrument.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_pliv(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    z: &[f64],
    names: Vec<String>,
    ml_l: List,
    ml_m: List,
    ml_r: List,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<ExternalPtr<DmlFit>> {
    let data = data(y, &d, &x, names)?
        .with_instrument(Array1::from(z.to_vec()))
        .map_err(err)?;
    let fit = Pliv::from_boxed(regressor(&ml_l)?, regressor(&ml_m)?, regressor(&ml_r)?)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_))
        .fit(&data)
        .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// Interactive IV model: LATE with a binary instrument.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_iivm(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    z: &[f64],
    names: Vec<String>,
    ml_g: List,
    ml_m: List,
    ml_r: List,
    always_takers: bool,
    never_takers: bool,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<ExternalPtr<DmlFit>> {
    let data = data(y, &d, &x, names)?
        .with_instrument(Array1::from(z.to_vec()))
        .map_err(err)?;
    let fit = Iivm::from_boxed(regressor(&ml_g)?, classifier(&ml_m)?, classifier(&ml_r)?)
        .subgroups(always_takers, never_takers)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_))
        .fit(&data)
        .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// Difference in differences: panel data (`y` = outcome change) or, with a
/// time indicator `t`, repeated cross-sections.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_did(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    t: Robj,
    names: Vec<String>,
    ml_g: List,
    ml_m: Robj,
    score: &str,
    in_sample_normalization: bool,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<ExternalPtr<DmlFit>> {
    let score = match score {
        "observational" => DidScore::Observational,
        "experimental" => DidScore::Experimental,
        other => return Err(err(format!("unknown DID score {other:?}"))),
    };
    let ml_m = optional_list(&ml_m)?.map(|m| classifier(&m)).transpose()?;
    let mut data = data(y, &d, &x, names)?;
    let fit = if t.is_null() {
        dml_core::Did::from_boxed(regressor(&ml_g)?, ml_m)
            .score(score)
            .in_sample_normalization(in_sample_normalization)
            .n_folds(n_folds as usize)
            .n_rep(n_rep as usize)
            .seed(seed(seed_))
            .fit(&data)
    } else {
        let t: Vec<f64> = t
            .as_real_vector()
            .ok_or_else(|| err("t must be a numeric vector"))?;
        data = data.with_time_indicator(Array1::from(t)).map_err(err)?;
        dml_core::DidCs::from_boxed(regressor(&ml_g)?, ml_m)
            .score(score)
            .in_sample_normalization(in_sample_normalization)
            .n_folds(n_folds as usize)
            .n_rep(n_rep as usize)
            .seed(seed(seed_))
            .fit(&data)
    }
    .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

fn to_rmatrix(a: &Array2<f64>) -> RMatrix<f64> {
    RMatrix::new_matrix(a.nrows(), a.ncols(), |i, j| a[[i, j]])
}

/// Estimates, standard errors, t statistics and p-values.
#[extendr]
fn rs_summary(fit: ExternalPtr<DmlFit>) -> List {
    list!(
        model = fit.model_name(),
        treatments = fit.treatment_names().to_vec(),
        coef = fit.coef().to_vec(),
        se = fit.se().to_vec(),
        t = fit.t_stat().to_vec(),
        p = fit.p_value().to_vec(),
        n_obs = fit.n_obs() as i32,
        n_folds = fit.n_folds() as i32,
        n_rep = fit.n_rep() as i32,
        all_coef = to_rmatrix(&fit.all_coef().to_owned()),
        all_se = to_rmatrix(&fit.all_se().to_owned())
    )
}

/// The results table as printed by Rust.
#[extendr]
fn rs_format(fit: ExternalPtr<DmlFit>) -> String {
    fit.to_string()
}

/// Pointwise confidence intervals, one row per treatment.
#[extendr]
fn rs_confint(fit: ExternalPtr<DmlFit>, level: f64) -> Result<RMatrix<f64>> {
    Ok(to_rmatrix(&fit.confint(level).map_err(err)?))
}

/// Simultaneous confidence intervals from the multiplier bootstrap.
#[extendr]
fn rs_joint_confint(
    fit: ExternalPtr<DmlFit>,
    method: &str,
    n_boot: i32,
    seed_: f64,
    level: f64,
) -> Result<RMatrix<f64>> {
    let method = match method {
        "normal" => BootstrapMethod::Normal,
        "Bayes" => BootstrapMethod::Bayes,
        "wild" => BootstrapMethod::Wild,
        other => return Err(err(format!("unknown bootstrap method {other:?}"))),
    };
    let boot = fit
        .bootstrap(method, n_boot as usize, seed(seed_))
        .map_err(err)?;
    Ok(to_rmatrix(&boot.joint_confint(level).map_err(err)?))
}

/// Bounds and robustness values under omitted confounding (PLR, IRM, DID).
#[extendr]
fn rs_sensitivity(
    fit: ExternalPtr<DmlFit>,
    cf_y: f64,
    cf_d: f64,
    rho: f64,
    level: f64,
    null_hypothesis: &[f64],
) -> Result<List> {
    let params = SensitivityParams::new()
        .cf_y(cf_y)
        .cf_d(cf_d)
        .rho(rho)
        .level(level)
        .null_hypotheses(null_hypothesis.to_vec());
    let res = fit.sensitivity_analysis(&params).map_err(err)?;
    Ok(list!(
        theta_lower = res.theta_lower.to_vec(),
        theta_upper = res.theta_upper.to_vec(),
        ci_lower = res.ci_lower.to_vec(),
        ci_upper = res.ci_upper.to_vec(),
        rv = res.rv.to_vec(),
        rva = res.rva.to_vec(),
        text = res.to_string()
    ))
}

/// Called from `.onLoad`. Errors reach R as R errors carrying the message
/// (extendr raises them by unwinding), so the default panic hook's console
/// output would only repeat it.
#[extendr]
fn rs_init() {
    std::panic::set_hook(Box::new(|_| {}));
}

/// Whether `ptr` still points to a fit (it does not after `readRDS`).
#[extendr]
fn rs_is_valid(ptr: Robj) -> bool {
    ExternalPtr::<DmlFit>::try_from(&ptr).is_ok()
}

extendr_module! {
    mod dmlrs;
    fn rs_fit_plr;
    fn rs_fit_irm;
    fn rs_fit_pliv;
    fn rs_fit_iivm;
    fn rs_fit_did;
    fn rs_summary;
    fn rs_format;
    fn rs_confint;
    fn rs_joint_confint;
    fn rs_sensitivity;
    fn rs_is_valid;
    fn rs_init;
}

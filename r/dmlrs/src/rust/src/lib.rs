//! R bindings for dml-core (double / debiased machine learning).
//!
//! The R functions in `R/` validate their input and call the functions below
//! through `.Call` (see `R/extendr-wrappers.R`). A fitted model lives in R as
//! an external pointer to a [`DmlFit`]; the accessors read results from it.

use std::collections::HashMap;

use dml_core::learner::{Classifier, Regressor};
use dml_core::{
    policy_value, Apos, BootstrapMethod, ContinuousDid, DidScore, DmlData, DmlFit, ForestSettings,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, Iivm, Irm, IrmScore, Lasso, LassoCV, LeeBounds,
    LogisticRegression, LogisticRegressionCV, MaxFeatures, Mediation, Ols, OptimalPolicyTree, Pliv,
    Plr, PlrScore, RandomForestClassifier, RandomForestRegressor, RegressionForest,
    SensitivityParams, SynthEstimator, SynthFit, SynthPanel, SynthSe, SyntheticDid, TrimmingRule,
};
use extendr_api::prelude::*;
use extendr_api::Result;
use ndarray::{Array1, Array2, Array3, ShapeBuilder};

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

/// Average potential outcomes of the levels of a discrete treatment.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_apos(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    names: Vec<String>,
    ml_g: List,
    ml_m: List,
    treatment_levels: &[f64],
    clipping_threshold: f64,
    normalize_ipw: bool,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<ExternalPtr<DmlFit>> {
    let fit = Apos::from_boxed(
        regressor(&ml_g)?,
        classifier(&ml_m)?,
        treatment_levels.to_vec(),
    )
    .clipping_threshold(clipping_threshold)
    .normalize_ipw(normalize_ipw)
    .n_folds(n_folds as usize)
    .n_rep(n_rep as usize)
    .seed(seed(seed_))
    .fit(&data(y, &d, &x, names)?)
    .map_err(err)?;
    Ok(ExternalPtr::new(fit.apos().clone()))
}

/// Natural direct and indirect effects: the effects and the four potential
/// outcome means as two fits.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_mediation(
    y: &[f64],
    d: RMatrix<f64>,
    m: RMatrix<f64>,
    x: RMatrix<f64>,
    names: Vec<String>,
    ml_g: List,
    ml_m: List,
    ml_nested: Robj,
    trimming_rule: &str,
    trimming_threshold: f64,
    normalize_ipw: bool,
    nested_split: bool,
    debias_nested: bool,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<List> {
    let rule = match trimming_rule {
        "truncate" => TrimmingRule::Truncate,
        "drop" => TrimmingRule::Drop,
        other => return Err(err(format!("unknown trimming rule {other:?}"))),
    };
    let data = data(y, &d, &x, names)?
        .with_mediators(matrix(&m))
        .map_err(err)?;
    let mut model = Mediation::from_boxed(regressor(&ml_g)?, classifier(&ml_m)?)
        .trimming_rule(rule)
        .trimming_threshold(trimming_threshold)
        .normalize_ipw(normalize_ipw)
        .nested_split(nested_split)
        .debias_nested(debias_nested)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_));
    if let Some(nested) = optional_list(&ml_nested)? {
        model = model.ml_nested_boxed(regressor(&nested)?);
    }
    let fit = model.fit(&data).map_err(err)?;
    let n_trimmed: Vec<i32> = fit.n_trimmed().iter().map(|&v| v as i32).collect();
    Ok(list!(
        effects = ExternalPtr::new(fit.effects().clone()),
        potential_outcomes = ExternalPtr::new(fit.potential_outcomes().clone()),
        n_trimmed = n_trimmed
    ))
}

/// Two-period difference in differences with a continuous dose: ATT(d|d),
/// ACRT(d|d) on the grid and the overall summaries, as three fits.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_contdid(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    ml_g: Robj,
    ml_m: Robj,
    ml_nested: Robj,
    degree: i32,
    num_knots: i32,
    grid: Robj,
    trimming_threshold: f64,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<List> {
    let mut model = match (optional_list(&ml_g)?, optional_list(&ml_m)?) {
        (Some(g), Some(m)) => ContinuousDid::from_boxed(regressor(&g)?, classifier(&m)?),
        (None, None) => ContinuousDid::new(),
        _ => return Err(err("pass both ml_g and ml_m (with covariates) or neither")),
    }
    .degree(degree as usize)
    .num_knots(num_knots as usize)
    .trimming_threshold(trimming_threshold)
    .n_folds(n_folds as usize)
    .n_rep(n_rep as usize)
    .seed(seed(seed_));
    if let Some(nested) = optional_list(&ml_nested)? {
        model = model.ml_nested_boxed(regressor(&nested)?);
    }
    if !grid.is_null() {
        let grid = grid
            .as_real_vector()
            .ok_or_else(|| err("grid must be a numeric vector"))?;
        model = model.grid(grid);
    }
    let fit = model
        .fit(&data(y, &d, &x, vec!["dose".into()])?)
        .map_err(err)?;
    Ok(list!(
        att = ExternalPtr::new(fit.att().clone()),
        acrt = ExternalPtr::new(fit.acrt().clone()),
        overall = ExternalPtr::new(fit.overall().clone()),
        grid = fit.grid().to_vec()
    ))
}

// ---------------------------------------------------------------------------
// Synthetic difference in differences
// ---------------------------------------------------------------------------

fn opt_num(v: &Robj, name: &str) -> Result<Option<f64>> {
    if v.is_null() {
        return Ok(None);
    }
    v.as_real()
        .or_else(|| v.as_integer().map(f64::from))
        .map(Some)
        .ok_or_else(|| err(format!("`{name}` must be a number or NULL")))
}

/// A units x periods x covariates R array as an ndarray.
fn array3(x: &Robj) -> Result<Array3<f64>> {
    let dims: Vec<usize> = x
        .get_attrib("dim")
        .and_then(|d| d.as_integer_vector())
        .ok_or_else(|| err("X must be a 3-dimensional array"))?
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let values = x.as_real_vector().ok_or_else(|| err("X must be numeric"))?;
    if dims.len() != 3 {
        return Err(err("X must be a 3-dimensional array"));
    }
    Ok(
        Array3::from_shape_vec((dims[0], dims[1], dims[2]).f(), values)
            .map_err(err)?
            .as_standard_layout()
            .to_owned(),
    )
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_synthdid(
    y: RMatrix<f64>,
    n0: i32,
    t0: i32,
    x: Robj,
    estimator: &str,
    noise_level: Robj,
    eta_omega: Robj,
    eta_lambda: Robj,
    min_decrease: Robj,
    max_iter: Robj,
    sparsify: bool,
) -> Result<ExternalPtr<SynthFit>> {
    let mut panel = SynthPanel::new(matrix(&y), n0 as usize, t0 as usize).map_err(err)?;
    if !x.is_null() {
        panel = panel.with_covariates(array3(&x)?).map_err(err)?;
    }
    let estimator = match estimator {
        "sdid" => SynthEstimator::SyntheticDid,
        "sc" => SynthEstimator::SyntheticControl,
        "did" => SynthEstimator::Did,
        other => return Err(err(format!("unknown estimator {other:?}"))),
    };
    let mut model = SyntheticDid::new().estimator(estimator).sparsify(sparsify);
    if let Some(v) = opt_num(&noise_level, "noise_level")? {
        model = model.noise_level(v);
    }
    if let Some(v) = opt_num(&eta_omega, "eta_omega")? {
        model = model.eta_omega(v);
    }
    if let Some(v) = opt_num(&eta_lambda, "eta_lambda")? {
        model = model.eta_lambda(v);
    }
    if let Some(v) = opt_num(&min_decrease, "min_decrease")? {
        model = model.min_decrease(v);
    }
    if let Some(v) = opt_num(&max_iter, "max_iter")? {
        model = model.max_iter(v as usize);
    }
    Ok(ExternalPtr::new(model.fit(&panel).map_err(err)?))
}

/// Estimate, weights, effect curve and trajectories of a synthetic DiD fit.
#[extendr]
fn rs_synth_info(fit: ExternalPtr<SynthFit>) -> List {
    let (synthetic, treated) = fit.trajectories();
    list!(
        estimate = fit.estimate(),
        unit_weights = fit.unit_weights().to_vec(),
        time_weights = fit.time_weights().to_vec(),
        beta = fit.beta().to_vec(),
        effect_curve = fit.effect_curve().to_vec(),
        noise_level = fit.noise_level(),
        synthetic = synthetic.to_vec(),
        treated = treated.to_vec()
    )
}

/// Standard error by `"placebo"`, `"jackknife"` or `"bootstrap"`.
#[extendr]
fn rs_synth_se(
    fit: ExternalPtr<SynthFit>,
    method: &str,
    replications: i32,
    seed_: i32,
) -> Result<f64> {
    let method = match method {
        "placebo" => SynthSe::Placebo {
            replications: replications as usize,
            seed: seed_,
        },
        "bootstrap" => SynthSe::Bootstrap {
            replications: replications as usize,
            seed: seed_,
        },
        "jackknife" => SynthSe::Jackknife,
        other => return Err(err(format!("unknown standard error method {other:?}"))),
    };
    fit.se(method).map_err(err)
}

// ---------------------------------------------------------------------------
// Policy learning and bounds
// ---------------------------------------------------------------------------

/// Exact optimal policy tree; the nodes as columns (0-based positions and actions).
#[extendr]
fn rs_policy_tree(
    x: RMatrix<f64>,
    gamma: RMatrix<f64>,
    depth: i32,
    split_step: i32,
    min_node_size: i32,
) -> Result<List> {
    let (x, gamma) = (matrix(&x), matrix(&gamma));
    let fit = OptimalPolicyTree::new(depth as usize)
        .split_step(split_step as usize)
        .min_node_size(min_node_size as usize)
        .fit(x.view(), gamma.view())
        .map_err(err)?;
    let nodes = fit.nodes();
    let col = |f: &dyn Fn(&dml_core::OptimalPolicyNode) -> f64| -> Vec<f64> {
        nodes.iter().map(f).collect()
    };
    Ok(list!(
        is_leaf = nodes.iter().map(|n| n.is_leaf).collect::<Vec<bool>>(),
        action = col(&|n| n.action as f64),
        split_variable = col(&|n| n.split_variable as f64),
        split_value = col(&|n| n.split_value),
        left = col(&|n| n.left as f64),
        right = col(&|n| n.right as f64),
        reward = col(&|n| n.reward)
    ))
}

/// Doubly robust values of policies; `actions` has one column of 0-based
/// actions per policy.
#[extendr]
fn rs_policy_value(
    gamma: RMatrix<f64>,
    actions: RMatrix<f64>,
    names: Vec<String>,
) -> Result<ExternalPtr<DmlFit>> {
    let actions = matrix(&actions);
    let columns: Vec<Vec<usize>> = actions
        .columns()
        .into_iter()
        .map(|c| c.iter().map(|&a| a as usize).collect())
        .collect();
    let policies: Vec<(&str, &[usize])> = names
        .iter()
        .zip(&columns)
        .map(|(n, a)| (n.as_str(), a.as_slice()))
        .collect();
    let fit = policy_value(matrix(&gamma).view(), &policies).map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// Lee (2009) bounds with bootstrap standard errors.
#[extendr]
fn rs_lee_bounds(
    y: &[f64],
    d: &[f64],
    s: &[f64],
    strata: Robj,
    n_boot: i32,
    level: f64,
    seed_: f64,
) -> Result<List> {
    let model = LeeBounds::new()
        .n_boot(n_boot as usize)
        .level(level)
        .seed(seed(seed_));
    let fit = if strata.is_null() {
        model.fit(y, d, s)
    } else {
        let cells: Vec<i64> = strata
            .as_integer_vector()
            .ok_or_else(|| err("strata must be an integer vector"))?
            .into_iter()
            .map(i64::from)
            .collect();
        model.fit_with_strata(y, d, s, &cells)
    }
    .map_err(err)?;
    Ok(list!(
        lower = fit.lower(),
        upper = fit.upper(),
        se = fit.se().to_vec(),
        confint = fit.confint().to_vec(),
        trimming = fit.trimming(),
        trimmed_treated = fit.trimmed_treated(),
        n_cells = fit.n_cells() as i32
    ))
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

/// Linear combinations `weights %*% theta` with standard errors from the scores.
#[extendr]
fn rs_linear_combination(
    fit: ExternalPtr<DmlFit>,
    weights: RMatrix<f64>,
    names: Vec<String>,
) -> Result<ExternalPtr<DmlFit>> {
    let combined = fit
        .linear_combination(matrix(&weights).view(), names)
        .map_err(err)?;
    Ok(ExternalPtr::new(combined))
}

/// Doubly robust scores of an APOS fit (first repetition): units x levels.
#[extendr]
fn rs_dr_scores(fit: ExternalPtr<DmlFit>) -> Result<RMatrix<f64>> {
    if fit.model_name() != "APOS" {
        return Err(err("doubly robust scores need a fit from dml_apos()"));
    }
    let psi = fit.psi();
    let theta = fit.all_coef();
    Ok(RMatrix::new_matrix(psi.dim().0, psi.dim().1, |i, j| {
        // The APO score has psi_a = -1, so psi = psi_b - theta.
        psi[[i, j, 0]] + theta[[j, 0]]
    }))
}

/// Called from `.onLoad`. Errors reach R as R errors carrying the message
/// (extendr raises them by unwinding), so the default panic hook's console
/// output would only repeat it. `threads` (or `NULL`: rayon's default, all
/// cores or `RAYON_NUM_THREADS`) sizes the global thread pool, which can be
/// set only before its first use.
#[extendr]
fn rs_init(threads: Option<i32>) {
    std::panic::set_hook(Box::new(|_| {}));
    if let Some(n) = threads.filter(|&n| n > 0) {
        // Fails only when the pool already exists (the package was loaded before).
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n as usize)
            .build_global();
    }
}

/// Whether `ptr` still points to a fit (it does not after `readRDS`).
#[extendr]
fn rs_is_valid(ptr: Robj) -> bool {
    ExternalPtr::<DmlFit>::try_from(&ptr).is_ok()
}

/// Whether `ptr` still points to a synthetic DiD fit.
#[extendr]
fn rs_is_valid_synth(ptr: Robj) -> bool {
    ExternalPtr::<SynthFit>::try_from(&ptr).is_ok()
}

extendr_module! {
    mod dmlrs;
    fn rs_fit_plr;
    fn rs_fit_irm;
    fn rs_fit_pliv;
    fn rs_fit_iivm;
    fn rs_fit_did;
    fn rs_fit_apos;
    fn rs_fit_mediation;
    fn rs_fit_contdid;
    fn rs_fit_synthdid;
    fn rs_synth_info;
    fn rs_synth_se;
    fn rs_policy_tree;
    fn rs_policy_value;
    fn rs_lee_bounds;
    fn rs_linear_combination;
    fn rs_dr_scores;
    fn rs_summary;
    fn rs_format;
    fn rs_confint;
    fn rs_joint_confint;
    fn rs_sensitivity;
    fn rs_is_valid;
    fn rs_is_valid_synth;
    fn rs_init;
}

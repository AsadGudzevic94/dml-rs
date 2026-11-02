//! Heterogeneous effects: BLP / GATE, CATE meta-learners, causal and
//! instrumental forests, conformal intervals and dose-response curves.
//!
//! Each fit lives in R as an external pointer; `R/heterogeneity.R` and
//! `R/dose.R` wrap the functions below.

use dml_core::learner::Regressor;
use dml_core::{
    Bandwidth, Blp, CateFit, CausalForestDml, CausalForestFit, ConformalFit, ConformalIte,
    ConformalScore, ConformalTarget, DensityModel, DmlFit, DoseKernel, DoseResponse,
    DoseResponseFit, DrLearner, ForestSettings, InstrumentalForestDml, InstrumentalForestFit,
    QuantileForest, RLearner, SLearner, TLearner, TreatmentLearner, XLearner,
};
use extendr_api::prelude::*;
use extendr_api::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{classifier, data, err, matrix, optional_list, regressor, seed, to_rmatrix, Spec};

// ---------------------------------------------------------------------------
// Best linear predictor and group effects
// ---------------------------------------------------------------------------

/// BLP of the CATE of a PLR or IRM (ATE) fit on `basis`.
#[extendr]
fn rs_cate(fit: ExternalPtr<DmlFit>, basis: RMatrix<f64>) -> Result<ExternalPtr<Blp>> {
    Ok(ExternalPtr::new(
        fit.cate(matrix(&basis).view()).map_err(err)?,
    ))
}

/// Group average treatment effects of a PLR or IRM (ATE) fit.
#[extendr]
fn rs_gate(fit: ExternalPtr<DmlFit>, groups: RMatrix<f64>) -> Result<ExternalPtr<Blp>> {
    Ok(ExternalPtr::new(
        fit.gate(matrix(&groups).view()).map_err(err)?,
    ))
}

#[extendr]
fn rs_blp_info(blp: ExternalPtr<Blp>) -> List {
    list!(
        coef = blp.coef().to_vec(),
        se = blp.se().to_vec(),
        t = blp.t_stat().to_vec(),
        p = blp.p_value().to_vec(),
        is_gate = blp.is_gate()
    )
}

#[extendr]
fn rs_blp_confint(blp: ExternalPtr<Blp>, level: f64) -> Result<RMatrix<f64>> {
    Ok(to_rmatrix(&blp.confint(level).map_err(err)?))
}

/// Effects at new basis rows: columns `[effect, se]` (`interval = "none"`) or
/// `[lower, effect, upper]`.
#[extendr]
fn rs_blp_predict(
    blp: ExternalPtr<Blp>,
    basis: RMatrix<f64>,
    interval: &str,
    level: f64,
    n_boot: i32,
    seed_: f64,
) -> Result<RMatrix<f64>> {
    let basis = matrix(&basis);
    let out = match interval {
        "none" => {
            let (effect, se) = blp.predict(basis.view()).map_err(err)?;
            columns(&[effect.view(), se.view()])
        }
        "pointwise" => blp.confint_at(basis.view(), level).map_err(err)?,
        "joint" => blp
            .joint_confint_at(basis.view(), level, n_boot as usize, seed(seed_))
            .map_err(err)?,
        other => return Err(err(format!("unknown interval {other:?}"))),
    };
    Ok(to_rmatrix(&out))
}

fn columns(cols: &[ArrayView1<'_, f64>]) -> Array2<f64> {
    let n = cols.first().map_or(0, |c| c.len());
    Array2::from_shape_fn((n, cols.len()), |(i, j)| cols[j][i])
}

// ---------------------------------------------------------------------------
// CATE meta-learners
// ---------------------------------------------------------------------------

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_cate_learner(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    method: &str,
    ml_outcome: List,
    ml_propensity: Robj,
    ml_effect: Robj,
    n_folds: i32,
    seed_: f64,
    clipping_threshold: f64,
) -> Result<ExternalPtr<CateFit>> {
    let data = data(y, &d, &x, vec!["d".into()])?;
    let outcome = regressor(&ml_outcome)?;
    let need = |robj: &Robj, arg: &str| -> Result<List> {
        optional_list(robj)?.ok_or_else(|| err(format!("the {method}-learner needs `{arg}`")))
    };
    let fit = match method {
        "S" => SLearner::from_boxed(outcome).fit(&data),
        "T" => TLearner::from_boxed(outcome).fit(&data),
        "X" => XLearner::from_boxed(
            outcome,
            regressor(&need(&ml_effect, "ml_effect")?)?,
            classifier(&need(&ml_propensity, "ml_propensity")?)?,
        )
        .fit(&data),
        "R" => RLearner::from_boxed(
            outcome,
            classifier(&need(&ml_propensity, "ml_propensity")?)?,
            regressor(&need(&ml_effect, "ml_effect")?)?,
        )
        .n_folds(n_folds as usize)
        .seed(seed(seed_))
        .clipping_threshold(clipping_threshold)
        .fit(&data),
        "DR" => DrLearner::from_boxed(
            outcome,
            classifier(&need(&ml_propensity, "ml_propensity")?)?,
            regressor(&need(&ml_effect, "ml_effect")?)?,
        )
        .n_folds(n_folds as usize)
        .seed(seed(seed_))
        .clipping_threshold(clipping_threshold)
        .fit(&data),
        other => return Err(err(format!("unknown CATE learner {other:?}"))),
    }
    .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

#[extendr]
fn rs_cate_effect(fit: ExternalPtr<CateFit>, x: RMatrix<f64>) -> Result<Vec<f64>> {
    Ok(fit.effect(matrix(&x).view()).map_err(err)?.to_vec())
}

/// The DR-learner's pseudo-outcomes, or `NULL` for the other learners.
#[extendr]
fn rs_cate_pseudo_outcomes(fit: ExternalPtr<CateFit>) -> Robj {
    match fit.pseudo_outcomes() {
        Some(p) => p.to_vec().into(),
        None => ().into(),
    }
}

#[extendr]
fn rs_cate_learner_blp(fit: ExternalPtr<CateFit>, basis: RMatrix<f64>) -> Result<ExternalPtr<Blp>> {
    Ok(ExternalPtr::new(
        fit.blp(matrix(&basis).view()).map_err(err)?,
    ))
}

#[extendr]
fn rs_cate_learner_gate(
    fit: ExternalPtr<CateFit>,
    groups: RMatrix<f64>,
) -> Result<ExternalPtr<Blp>> {
    Ok(ExternalPtr::new(
        fit.gate(matrix(&groups).view()).map_err(err)?,
    ))
}

// ---------------------------------------------------------------------------
// Causal and instrumental forests
// ---------------------------------------------------------------------------

/// A fitted causal or instrumental forest, behind one external pointer type.
enum ForestFit {
    Causal(CausalForestFit),
    Instrumental(InstrumentalForestFit),
}

impl ForestFit {
    fn effect(&self, x: ArrayView2<'_, f64>) -> dml_core::Result<(Array1<f64>, Array1<f64>)> {
        match self {
            Self::Causal(f) => f.effect(x),
            Self::Instrumental(f) => f.effect(x),
        }
    }

    fn dr_scores(&self) -> dml_core::Result<Array1<f64>> {
        match self {
            Self::Causal(f) => f.dr_scores(),
            Self::Instrumental(f) => f.dr_scores(),
        }
    }

    fn ate(&self) -> dml_core::Result<DmlFit> {
        match self {
            Self::Causal(f) => f.ate(),
            Self::Instrumental(f) => f.ate(),
        }
    }

    fn blp(&self, basis: ArrayView2<'_, f64>) -> dml_core::Result<Blp> {
        match self {
            Self::Causal(f) => f.blp(basis),
            Self::Instrumental(f) => f.blp(basis),
        }
    }

    fn gate(&self, groups: ArrayView2<'_, f64>) -> dml_core::Result<Blp> {
        match self {
            Self::Causal(f) => f.gate(groups),
            Self::Instrumental(f) => f.gate(groups),
        }
    }
}

/// grf's forest settings from `list(num_trees = 2000, ...)` (`R/heterogeneity.R`).
fn forest_settings(spec: &List) -> Result<ForestSettings> {
    let s = Spec::new(spec)?;
    let d = ForestSettings::default();
    Ok(ForestSettings::default()
        .num_trees(s.count("num_trees", d.num_trees)?)
        .sample_fraction(s.num("sample_fraction", d.sample_fraction)?)
        .mtry(s.opt_count("mtry", d.mtry)?)
        .min_node_size(s.count("min_node_size", d.min_node_size)?)
        .honesty(s.flag("honesty", d.honesty)?)
        .honesty_fraction(s.num("honesty_fraction", d.honesty_fraction)?)
        .prune_empty_leaves(s.flag("honesty_prune_leaves", d.prune_empty_leaves)?)
        .alpha(s.num("alpha", d.alpha)?)
        .imbalance_penalty(s.num("imbalance_penalty", d.imbalance_penalty)?)
        .stabilize_splits(s.flag("stabilize_splits", d.stabilize_splits)?)
        .ci_group_size(s.count("ci_group_size", d.ci_group_size)?)
        .seed(s.seed()?))
}

/// A classifier for a binary variable, a regressor otherwise (`R` decides).
fn treatment_learner(spec: &List, binary: bool) -> Result<TreatmentLearner> {
    Ok(if binary {
        TreatmentLearner::Classifier(classifier(spec)?)
    } else {
        TreatmentLearner::Regressor(regressor(spec)?)
    })
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_causal_forest(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    ml_y: List,
    ml_d: List,
    d_classifier: bool,
    forest: List,
    clipping_threshold: f64,
    n_folds: i32,
    seed_: f64,
) -> Result<ExternalPtr<ForestFit>> {
    let fit =
        CausalForestDml::from_boxed(regressor(&ml_y)?, treatment_learner(&ml_d, d_classifier)?)
            .forest(forest_settings(&forest)?)
            .clipping_threshold(clipping_threshold)
            .n_folds(n_folds as usize)
            .seed(seed(seed_))
            .fit(&data(y, &d, &x, vec!["d".into()])?)
            .map_err(err)?;
    Ok(ExternalPtr::new(ForestFit::Causal(fit)))
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_instrumental_forest(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    z: &[f64],
    ml_y: List,
    ml_d: List,
    ml_z: List,
    d_classifier: bool,
    z_classifier: bool,
    forest: List,
    reduced_form_weight: f64,
    clipping_threshold: f64,
    n_folds: i32,
    seed_: f64,
) -> Result<ExternalPtr<ForestFit>> {
    let data = data(y, &d, &x, vec!["d".into()])?
        .with_instrument(Array1::from(z.to_vec()))
        .map_err(err)?;
    let fit = InstrumentalForestDml::from_boxed(
        regressor(&ml_y)?,
        treatment_learner(&ml_d, d_classifier)?,
        treatment_learner(&ml_z, z_classifier)?,
    )
    .forest(forest_settings(&forest)?)
    .reduced_form_weight(reduced_form_weight)
    .clipping_threshold(clipping_threshold)
    .n_folds(n_folds as usize)
    .seed(seed(seed_))
    .fit(&data)
    .map_err(err)?;
    Ok(ExternalPtr::new(ForestFit::Instrumental(fit)))
}

/// Out-of-bag estimates, nuisances and variable importance.
#[extendr]
fn rs_forest_info(fit: ExternalPtr<ForestFit>) -> List {
    match &*fit {
        ForestFit::Causal(f) => list!(
            effect = f.oob_effect().to_vec(),
            variance = f.oob_variance().to_vec(),
            y_hat = f.y_hat().to_vec(),
            d_hat = f.d_hat().to_vec(),
            importance = f.forest().variable_importance().to_vec()
        ),
        ForestFit::Instrumental(f) => list!(
            effect = f.oob_effect().to_vec(),
            variance = f.oob_variance().to_vec(),
            y_hat = f.y_hat().to_vec(),
            d_hat = f.d_hat().to_vec(),
            z_hat = f.z_hat().to_vec(),
            importance = f.forest().variable_importance().to_vec()
        ),
    }
}

/// Effects and their variances at new points: columns `[effect, variance]`.
#[extendr]
fn rs_forest_effect(fit: ExternalPtr<ForestFit>, x: RMatrix<f64>) -> Result<RMatrix<f64>> {
    let (effect, variance) = fit.effect(matrix(&x).view()).map_err(err)?;
    Ok(to_rmatrix(&columns(&[effect.view(), variance.view()])))
}

#[extendr]
fn rs_forest_dr_scores(fit: ExternalPtr<ForestFit>) -> Result<Vec<f64>> {
    Ok(fit.dr_scores().map_err(err)?.to_vec())
}

#[extendr]
fn rs_forest_ate(fit: ExternalPtr<ForestFit>) -> Result<ExternalPtr<DmlFit>> {
    Ok(ExternalPtr::new(fit.ate().map_err(err)?))
}

#[extendr]
fn rs_forest_blp(fit: ExternalPtr<ForestFit>, basis: RMatrix<f64>) -> Result<ExternalPtr<Blp>> {
    Ok(ExternalPtr::new(
        fit.blp(matrix(&basis).view()).map_err(err)?,
    ))
}

#[extendr]
fn rs_forest_gate(fit: ExternalPtr<ForestFit>, groups: RMatrix<f64>) -> Result<ExternalPtr<Blp>> {
    Ok(ExternalPtr::new(
        fit.gate(matrix(&groups).view()).map_err(err)?,
    ))
}

// ---------------------------------------------------------------------------
// Conformal intervals
// ---------------------------------------------------------------------------

/// A quantile forest predicting one quantile, so that it can serve as the
/// lower or upper learner of the conformalised quantile score.
struct QuantileForestRegressor {
    quantile: f64,
    forest: QuantileForest,
}

impl Regressor for QuantileForestRegressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> dml_core::Result<()> {
        self.forest.fit(x, y)
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> dml_core::Result<Array1<f64>> {
        Ok(self
            .forest
            .predict(x, &[self.quantile])?
            .column(0)
            .to_owned())
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(QuantileForestRegressor {
            quantile: self.quantile,
            forest: self.forest.clone(),
        })
    }
}

/// A regressor for the conformal scores: any regressor, or a quantile forest.
fn score_learner(spec: &List) -> Result<Box<dyn Regressor>> {
    let s = Spec::new(spec)?;
    if s.kind != "quantile_forest" {
        return regressor(spec);
    }
    let quantile = s.num("quantile", 0.5)?;
    let settings = ForestSettings::default()
        .num_trees(s.count("n_estimators", 2000)?)
        .min_node_size(s.count("min_node_size", 5)?)
        .ci_group_size(1)
        .seed(s.seed()?);
    Ok(Box::new(QuantileForestRegressor {
        quantile,
        forest: QuantileForest::new(vec![quantile]).settings(settings),
    }))
}

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_conformal(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    ml_y: Robj,
    ml_lower: Robj,
    ml_upper: Robj,
    ml_m: Robj,
    propensity: Robj,
    train_fraction: f64,
    clipping_threshold: f64,
    seed_: f64,
) -> Result<ExternalPtr<ConformalFit>> {
    let score =
        match (
            optional_list(&ml_y)?,
            optional_list(&ml_lower)?,
            optional_list(&ml_upper)?,
        ) {
            (Some(m), None, None) => ConformalScore::Residual(score_learner(&m)?),
            (None, Some(lo), Some(hi)) => ConformalScore::Quantile {
                lower: score_learner(&lo)?,
                upper: score_learner(&hi)?,
            },
            _ => return Err(err(
                "pass either ml_y (residual score) or both ml_lower and ml_upper (quantile score)",
            )),
        };
    let mut model = ConformalIte::new(score)
        .train_fraction(train_fraction)
        .clipping_threshold(clipping_threshold)
        .seed(seed(seed_));
    if let Some(m) = optional_list(&ml_m)? {
        model = model.ml_m(classifier(&m)?);
    }
    if !propensity.is_null() {
        let p = propensity
            .as_real()
            .ok_or_else(|| err("propensity must be a number"))?;
        model = model.propensity(p);
    }
    let fit = model
        .fit(&data(y, &d, &x, vec!["d".into()])?)
        .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// Intervals `[lower, upper]` for `"ite"`, `"y1"` or `"y0"`.
#[extendr]
fn rs_conformal_predict(
    fit: ExternalPtr<ConformalFit>,
    x: RMatrix<f64>,
    what: &str,
    target: &str,
    alpha: f64,
) -> Result<RMatrix<f64>> {
    let x = matrix(&x);
    let target = match target {
        "all" => ConformalTarget::All,
        "treated" => ConformalTarget::Treated,
        "control" => ConformalTarget::Control,
        other => return Err(err(format!("unknown target {other:?}"))),
    };
    let out = match what {
        "ite" => fit.predict_ite(x.view(), alpha),
        "y1" => fit.predict_counterfactual(x.view(), 1, target, alpha),
        "y0" => fit.predict_counterfactual(x.view(), 0, target, alpha),
        other => return Err(err(format!("unknown prediction {other:?}"))),
    }
    .map_err(err)?;
    Ok(to_rmatrix(&out))
}

#[extendr]
fn rs_conformal_n_calibration(fit: ExternalPtr<ConformalFit>) -> Vec<i32> {
    fit.n_calibration().iter().map(|&n| n as i32).collect()
}

// ---------------------------------------------------------------------------
// Dose-response curves
// ---------------------------------------------------------------------------

#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_dose_response(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    ml_g: List,
    ml_m: List,
    grid: Robj,
    kernel: &str,
    bandwidth: Robj,
    bandwidth_scale: f64,
    density: &str,
    density_bandwidth: Robj,
    density_floor: f64,
    partial_effects: bool,
    eta: Robj,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<List> {
    let kernel = match kernel {
        "epanechnikov" => DoseKernel::Epanechnikov,
        "gaussian" => DoseKernel::Gaussian,
        other => return Err(err(format!("unknown kernel {other:?}"))),
    };
    let density = match density {
        "kernel" => DensityModel::KernelRegression,
        "normal" => DensityModel::Normal,
        other => return Err(err(format!("unknown density model {other:?}"))),
    };
    let bandwidth = match crate::opt_num(&bandwidth, "bandwidth")? {
        Some(h) => Bandwidth::Fixed(h),
        None => Bandwidth::RuleOfThumb(bandwidth_scale),
    };
    let mut model = DoseResponse::from_boxed(regressor(&ml_g)?, regressor(&ml_m)?)
        .kernel(kernel)
        .bandwidth(bandwidth)
        .density(density)
        .density_floor(density_floor)
        .partial_effects(partial_effects)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_));
    if !grid.is_null() {
        let grid = grid
            .as_real_vector()
            .ok_or_else(|| err("grid must be a numeric vector"))?;
        model = model.grid(grid);
    }
    if let Some(h1) = crate::opt_num(&density_bandwidth, "density_bandwidth")? {
        model = model.density_bandwidth(h1);
    }
    if let Some(e) = crate::opt_num(&eta, "eta")? {
        model = model.eta(e);
    }
    let fit = model
        .fit(&data(y, &d, &x, vec!["dose".into()])?)
        .map_err(err)?;
    let partial: Robj = match fit.partial_effects() {
        Some(p) => ExternalPtr::new(p.clone()).into(),
        None => ().into(),
    };
    Ok(list!(
        curve = ExternalPtr::new(fit.curve().clone()),
        partial = partial,
        grid = fit.grid().to_vec(),
        bandwidth = fit.bandwidth(),
        eta = fit.eta(),
        fit = ExternalPtr::new(fit)
    ))
}

/// Differences `θ(t) − θ(reference)` for the other grid points.
#[extendr]
fn rs_dose_contrast(
    fit: ExternalPtr<DoseResponseFit>,
    reference: f64,
) -> Result<ExternalPtr<DmlFit>> {
    Ok(ExternalPtr::new(fit.contrast(reference).map_err(err)?))
}

// ---------------------------------------------------------------------------

/// Whether `ptr` still points to an object of `kind` (it does not after `readRDS`).
#[extendr]
fn rs_is_valid_kind(ptr: Robj, kind: &str) -> bool {
    match kind {
        "blp" => ExternalPtr::<Blp>::try_from(&ptr).is_ok(),
        "cate" => ExternalPtr::<CateFit>::try_from(&ptr).is_ok(),
        "forest" => ExternalPtr::<ForestFit>::try_from(&ptr).is_ok(),
        "conformal" => ExternalPtr::<ConformalFit>::try_from(&ptr).is_ok(),
        "dose" => ExternalPtr::<DoseResponseFit>::try_from(&ptr).is_ok(),
        "staggered" => ExternalPtr::<dml_core::StaggeredDidFit>::try_from(&ptr).is_ok(),
        "rdflex" => ExternalPtr::<dml_core::RdFlexFit>::try_from(&ptr).is_ok(),
        _ => false,
    }
}

extendr_module! {
    mod heterogeneity;
    fn rs_cate;
    fn rs_gate;
    fn rs_blp_info;
    fn rs_blp_confint;
    fn rs_blp_predict;
    fn rs_fit_cate_learner;
    fn rs_cate_effect;
    fn rs_cate_pseudo_outcomes;
    fn rs_cate_learner_blp;
    fn rs_cate_learner_gate;
    fn rs_fit_causal_forest;
    fn rs_fit_instrumental_forest;
    fn rs_forest_info;
    fn rs_forest_effect;
    fn rs_forest_dr_scores;
    fn rs_forest_ate;
    fn rs_forest_blp;
    fn rs_forest_gate;
    fn rs_fit_conformal;
    fn rs_conformal_predict;
    fn rs_conformal_n_calibration;
    fn rs_fit_dose_response;
    fn rs_dose_contrast;
    fn rs_is_valid_kind;
}

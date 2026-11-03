//! Staggered difference in differences, regression discontinuity with
//! flexible covariate adjustment, and effects on a censored time to event.
//!
//! Each fit lives in R as an external pointer; `R/staggered.R`, `R/rdflex.R` and
//! `R/survival.R` wrap the functions below.

use dml_core::rd::{BwSelect, Kernel, MassPoints, RdOptions, Vce};
use dml_core::{
    Aggregation, CausalSurvival, ControlGroup, DidScore, DmlFit, FirstStage, GtCombinations,
    PanelData, RdFlex, RdFlexFit, RepeatedCrossSections, StaggeredDid, StaggeredDidFit,
    SurvivalEstimand, SurvivalGrid, TimeEncoding,
};
use extendr_api::prelude::*;
use extendr_api::Result;
use ndarray::Array1;
use std::collections::BTreeMap;

use crate::{classifier, data, err, matrix, optional_list, regressor, seed, to_rmatrix};

// ---------------------------------------------------------------------------
// Staggered difference in differences
// ---------------------------------------------------------------------------

/// One cluster label per unit, in the order of `PanelData::from_long` (units
/// sorted by id), from labels per row that must be constant within a unit.
fn unit_labels(ids: &[i64], labels: &[i64]) -> Result<Array1<i64>> {
    let mut by_unit = BTreeMap::new();
    for (&id, &label) in ids.iter().zip(labels) {
        if *by_unit.entry(id).or_insert(label) != label {
            return Err(err(format!(
                "the cluster of unit {id} changes over time; clusters must be constant within units"
            )));
        }
    }
    Ok(by_unit.into_values().collect())
}

/// Group-time effects from long data: `id` are integer codes (ignored for
/// repeated cross-sections), `g` the first treated period (`Inf` = never).
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_staggered(
    y: &[f64],
    id: &[i32],
    t: &[f64],
    g: &[f64],
    x: RMatrix<f64>,
    ml_g: List,
    ml_m: Robj,
    control_group: &str,
    anticipation_periods: i32,
    gt_combinations: &str,
    score: &str,
    in_sample_normalization: bool,
    clipping_threshold: f64,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
    panel: bool,
    cluster: Robj,
) -> Result<ExternalPtr<StaggeredDidFit>> {
    let control_group = match control_group {
        "never_treated" => ControlGroup::NeverTreated,
        "not_yet_treated" => ControlGroup::NotYetTreated,
        other => return Err(err(format!("unknown control group {other:?}"))),
    };
    let setting = match gt_combinations {
        "standard" => GtCombinations::Standard,
        "all" => GtCombinations::All,
        "universal" => GtCombinations::Universal,
        other => return Err(err(format!("unknown gt_combinations {other:?}"))),
    };
    let score = match score {
        "observational" => DidScore::Observational,
        "experimental" => DidScore::Experimental,
        other => return Err(err(format!("unknown DID score {other:?}"))),
    };
    let ml_m = optional_list(&ml_m)?.map(|m| classifier(&m)).transpose()?;
    let model = StaggeredDid::from_boxed(regressor(&ml_g)?, ml_m)
        .score(score)
        .control_group(control_group)
        .anticipation_periods(anticipation_periods as usize)
        .gt_combinations(setting)
        .in_sample_normalization(in_sample_normalization)
        .clipping_threshold(clipping_threshold)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_));
    let x = matrix(&x);
    let cluster: Option<Vec<i64>> = if cluster.is_null() {
        None
    } else {
        let labels = cluster
            .as_integer_vector()
            .ok_or_else(|| err("cluster must hold integer codes"))?;
        Some(labels.into_iter().map(i64::from).collect())
    };
    let fit = if panel {
        let ids: Vec<i64> = id.iter().map(|&i| i64::from(i)).collect();
        let mut data = PanelData::from_long(&ids, t, y, g, &x).map_err(err)?;
        if let Some(labels) = &cluster {
            data = data
                .with_clusters(unit_labels(&ids, labels)?)
                .map_err(err)?;
        }
        model.fit(&data)
    } else {
        let mut data = RepeatedCrossSections::new(
            Array1::from(y.to_vec()),
            Array1::from(t.to_vec()),
            Array1::from(g.to_vec()),
            x,
        )
        .map_err(err)?;
        if let Some(labels) = cluster {
            data = data.with_clusters(Array1::from(labels)).map_err(err)?;
        }
        model.fit(&data)
    }
    .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

/// The ATT(g, t) as a fit.
#[extendr]
fn rs_staggered_att(fit: ExternalPtr<StaggeredDidFit>) -> ExternalPtr<DmlFit> {
    ExternalPtr::new(fit.att().clone())
}

/// `(cohort, pre-period, evaluation period)` of each ATT(g, t), one row each.
#[extendr]
fn rs_staggered_group_times(fit: ExternalPtr<StaggeredDidFit>) -> RMatrix<f64> {
    let gts = fit.group_times();
    RMatrix::new_matrix(gts.len(), 3, |i, j| match j {
        0 => gts[i].cohort,
        1 => gts[i].t_pre,
        _ => gts[i].t_eval,
    })
}

/// Group, calendar-time or event-study aggregation.
#[extendr]
fn rs_staggered_aggregate(fit: ExternalPtr<StaggeredDidFit>, how: &str) -> Result<List> {
    let how = match how {
        "group" => Aggregation::Group,
        "time" => Aggregation::Time,
        "eventstudy" => Aggregation::EventStudy,
        other => return Err(err(format!("unknown aggregation {other:?}"))),
    };
    let agg = fit.aggregate(how).map_err(err)?;
    Ok(list!(
        effects = ExternalPtr::new(agg.effects().clone()),
        overall = ExternalPtr::new(agg.overall().clone()),
        weights = to_rmatrix(agg.weights()),
        overall_weights = agg.overall_weights().to_vec()
    ))
}

// ---------------------------------------------------------------------------
// Regression discontinuity
// ---------------------------------------------------------------------------

/// `rdrobust` options from a named list.
fn rd_options(options: &List) -> Result<RdOptions> {
    let mut opts = RdOptions::default();
    for (key, value) in options.iter() {
        let bad = |shown: &str| {
            err(format!(
                "rdrobust option {key} = {shown:?} is not supported"
            ))
        };
        let text = || {
            value
                .as_str()
                .map(|s| s.trim().to_ascii_lowercase())
                .ok_or_else(|| err(format!("rdrobust option {key} must be a string")))
        };
        let number = || {
            value
                .as_real()
                .or_else(|| value.as_integer().map(f64::from))
                .ok_or_else(|| err(format!("rdrobust option {key} must be a number")))
        };
        let flag = || {
            value
                .as_bool()
                .ok_or_else(|| err(format!("rdrobust option {key} must be TRUE or FALSE")))
        };
        match key {
            "p" => opts.p = number()? as usize,
            "q" => opts.q = Some(number()? as usize),
            "kernel" => {
                let v = text()?;
                opts.kernel = match v.as_str() {
                    "tri" | "triangular" => Kernel::Triangular,
                    "epa" | "epanechnikov" => Kernel::Epanechnikov,
                    "uni" | "uniform" => Kernel::Uniform,
                    _ => return Err(bad(&v)),
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
                    _ => return Err(bad(&v)),
                }
            }
            "vce" => {
                let v = text()?;
                opts.vce = match v.as_str() {
                    "nn" => Vce::Nn,
                    "hc0" => Vce::Hc0,
                    "hc1" => Vce::Hc1,
                    "hc2" => Vce::Hc2,
                    "hc3" => Vce::Hc3,
                    "cr1" => Vce::Cr1,
                    "cr2" => Vce::Cr2,
                    "cr3" => Vce::Cr3,
                    _ => return Err(bad(&v)),
                }
            }
            "masspoints" => {
                let v = text()?;
                opts.masspoints = match v.as_str() {
                    "adjust" => MassPoints::Adjust,
                    "check" => MassPoints::Check,
                    "off" => MassPoints::Off,
                    _ => return Err(bad(&v)),
                }
            }
            "nnmatch" => opts.nnmatch = number()? as usize,
            "level" => opts.level = number()? / 100.0,
            "bwcheck" => opts.bwcheck = Some(number()? as usize),
            "bwrestrict" => opts.bwrestrict = flag()?,
            "stdvars" => opts.stdvars = flag()?,
            "scaleregul" => opts.scaleregul = number()?,
            "sharpbw" => opts.sharpbw = flag()?,
            "rho" => opts.rho = Some(number()?),
            "scalepar" => opts.scalepar = number()?,
            "covs_drop" => opts.covs_drop = flag()?,
            other => return Err(err(format!("unknown rdrobust option {other:?}"))),
        }
    }
    Ok(opts)
}

/// Sharp, or fuzzy with `ml_m`; `ml_g` is a classifier when `binary_outcome`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_rdflex(
    y: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    score: &[f64],
    names: Vec<String>,
    ml_g: List,
    ml_m: Robj,
    binary_outcome: bool,
    cutoff: f64,
    h_fs: Robj,
    fs_specification: &str,
    fs_kernel: &str,
    n_iterations: i32,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
    options: List,
    rd_covs: Robj,
    rd_cluster: Robj,
) -> Result<ExternalPtr<RdFlexFit>> {
    let first_stage = match fs_specification {
        "cutoff" => FirstStage::Cutoff,
        "cutoff and score" => FirstStage::CutoffAndScore,
        "interacted cutoff and score" => FirstStage::InteractedCutoffAndScore,
        other => return Err(err(format!("unknown fs_specification {other:?}"))),
    };
    let kernel = match fs_kernel {
        "uniform" => Kernel::Uniform,
        "triangular" => Kernel::Triangular,
        "epanechnikov" => Kernel::Epanechnikov,
        other => return Err(err(format!("unknown fs_kernel {other:?}"))),
    };
    let ml_m = optional_list(&ml_m)?.map(|m| classifier(&m)).transpose()?;
    let model = if binary_outcome {
        RdFlex::from_boxed_binary(classifier(&ml_g)?, ml_m)
    } else {
        RdFlex::from_boxed(regressor(&ml_g)?, ml_m)
    };
    let mut model = model.rd_options(rd_options(&options)?);
    if !rd_covs.is_null() {
        let covs = RMatrix::<f64>::try_from(rd_covs)?;
        model = model.rd_covariates(matrix(&covs));
    }
    if !rd_cluster.is_null() {
        let labels = rd_cluster
            .as_integer_vector()
            .ok_or_else(|| err("cluster must hold integer codes"))?;
        model = model.rd_clusters(labels.into_iter().map(i64::from).collect());
    }
    let mut model = model
        .cutoff(cutoff)
        .first_stage(first_stage)
        .kernel(kernel)
        .n_iterations(n_iterations as usize)
        .n_folds(n_folds as usize)
        .n_rep(n_rep as usize)
        .seed(seed(seed_));
    if !h_fs.is_null() {
        let h = h_fs.as_real().ok_or_else(|| err("h_fs must be a number"))?;
        model = model.h_fs(h);
    }
    let data = data(y, &d, &x, names)?;
    let fit = model
        .fit(&data, Array1::from(score.to_vec()).view())
        .map_err(err)?;
    Ok(ExternalPtr::new(fit))
}

fn intervals(ci: [[f64; 2]; 3]) -> RMatrix<f64> {
    RMatrix::new_matrix(3, 2, |i, j| ci[i][j])
}

/// Estimates of the conventional, bias-corrected and robust rows.
#[extendr]
fn rs_rdflex_summary(fit: ExternalPtr<RdFlexFit>) -> List {
    list!(
        coef = fit.coef().to_vec(),
        se = fit.se().to_vec(),
        t = fit.t_stat().to_vec(),
        p = fit.p_value().to_vec(),
        all_coef = to_rmatrix(&fit.all_coef()),
        all_se = to_rmatrix(&fit.all_se()),
        h = fit.h(),
        b = fit.b(),
        h_fs = fit.h_fs()
    )
}

/// Intervals at `level`, one row per estimator.
#[extendr]
fn rs_rdflex_confint(fit: ExternalPtr<RdFlexFit>, level: f64) -> Result<RMatrix<f64>> {
    Ok(intervals(fit.confint(level).map_err(err)?))
}

// ---------------------------------------------------------------------------
// Censored time to event
// ---------------------------------------------------------------------------

/// `time` is the observed time; `grid` a number of quantiles, `"event_times"` or
/// a vector of times.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_survival(
    time: &[f64],
    d: RMatrix<f64>,
    x: RMatrix<f64>,
    event: &[f64],
    ml_event: List,
    ml_censor: List,
    ml_m: List,
    horizons: Robj,
    estimand: &str,
    grid: Robj,
    time_encoding: &str,
    clipping_threshold: f64,
    censoring_floor: f64,
    n_folds: i32,
    n_rep: i32,
    seed_: f64,
) -> Result<List> {
    let estimand = match estimand {
        "survival" => SurvivalEstimand::Survival,
        "rmst" => SurvivalEstimand::Rmst,
        other => return Err(err(format!("unknown estimand {other:?}"))),
    };
    let encoding = match time_encoding {
        "numeric" => TimeEncoding::Numeric,
        "dummies" => TimeEncoding::Dummies,
        other => return Err(err(format!("unknown time_encoding {other:?}"))),
    };
    let grid = match (grid.as_str(), grid.as_real_vector()) {
        (Some("event_times"), _) => SurvivalGrid::EventTimes,
        (None, Some(times)) if times.len() == 1 => SurvivalGrid::Quantiles(times[0] as usize),
        (None, Some(times)) => SurvivalGrid::Fixed(times),
        _ => {
            return Err(err(
                "grid must be a number, \"event_times\" or a vector of times",
            ))
        }
    };
    let mut model = CausalSurvival::from_boxed(
        classifier(&ml_event)?,
        classifier(&ml_censor)?,
        classifier(&ml_m)?,
    )
    .estimand(estimand)
    .grid(grid)
    .time_encoding(encoding)
    .clipping_threshold(clipping_threshold)
    .censoring_floor(censoring_floor)
    .n_folds(n_folds as usize)
    .n_rep(n_rep as usize)
    .seed(seed(seed_));
    if !horizons.is_null() {
        let horizons = horizons
            .as_real_vector()
            .ok_or_else(|| err("horizons must be numbers"))?;
        model = model.horizons(horizons);
    }
    let data = data(time, &d, &x, vec!["d".to_string()])?;
    let fit = model
        .fit(&data, Array1::from(event.to_vec()).view())
        .map_err(err)?;
    let pointer = |f: &DmlFit| ExternalPtr::new(f.clone());
    Ok(list!(
        effect = pointer(fit.effect()),
        treated = pointer(fit.potential(true)),
        control = pointer(fit.potential(false)),
        curve_treated = pointer(fit.curve(true)),
        curve_control = pointer(fit.curve(false)),
        horizons = fit.horizons().to_vec(),
        grid = fit.grid().to_vec(),
        propensity = to_rmatrix(fit.propensity())
    ))
}

extendr_module! {
    mod panel;
    fn rs_fit_survival;
    fn rs_fit_staggered;
    fn rs_staggered_att;
    fn rs_staggered_group_times;
    fn rs_staggered_aggregate;
    fn rs_fit_rdflex;
    fn rs_rdflex_summary;
    fn rs_rdflex_confint;
}

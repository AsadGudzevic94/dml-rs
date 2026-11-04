//! Anytime-valid confidence sequences and prediction-powered inference.
//!
//! `R/sequential.R` wraps the functions below.

use dml_core::anytime::ConfidenceSequence;
use dml_core::{DmlFit, PpiData, PpiEstimand, PpiLambda, PredictionPowered};
use extendr_api::prelude::*;
use extendr_api::Result;
use ndarray::Array2;

use crate::{classifier, err, matrix, optional_list, regressor, seed, to_rmatrix};

/// Units, or clusters for a clustered fit: the sample size of a sequence.
fn sequence_time(fit: &DmlFit) -> usize {
    fit.n_clusters().unwrap_or(fit.n_obs())
}

fn opt_time(value: Option<f64>, default: usize) -> Result<usize> {
    match value {
        None => Ok(default),
        Some(v) if v >= 1.0 && v.fract() == 0.0 => Ok(v as usize),
        Some(v) => Err(err(format!("opt_time must be a positive integer, got {v}"))),
    }
}

/// Confidence sequence and anytime p-values of one fit.
#[extendr]
fn rs_confidence_sequence(
    fit: ExternalPtr<DmlFit>,
    level: f64,
    opt_time_: Option<f64>,
) -> Result<List> {
    let t = sequence_time(&fit);
    let opt = opt_time(opt_time_, t)?;
    Ok(list!(
        interval = to_rmatrix(&fit.confidence_sequence(level, opt).map_err(err)?),
        p = fit.anytime_p_value(level, opt).map_err(err)?.to_vec(),
        n = t as i32,
        opt_time = opt as i32
    ))
}

/// A confidence sequence over looks: `fits` are fits on all data so far, in
/// order. `opt_time` defaults to the sample size of the first look.
#[extendr]
fn rs_monitor(fits: List, level: f64, opt_time_: Option<f64>) -> Result<List> {
    let ptrs: Vec<ExternalPtr<DmlFit>> = fits
        .values()
        .map(|f| ExternalPtr::<DmlFit>::try_from(&f).map_err(|_| err("every look must be a fit")))
        .collect::<Result<_>>()?;
    let first = ptrs.first().ok_or_else(|| err("need at least one look"))?;
    let mut cs =
        ConfidenceSequence::new(level, opt_time(opt_time_, sequence_time(first))?).map_err(err)?;
    for fit in &ptrs {
        cs.update(fit).map_err(err)?;
    }
    let running = cs.running_intervals();
    let side = |s: usize| -> RMatrix<f64> {
        let (looks, k, _) = running.dim();
        RMatrix::new_matrix(looks, k, |l, j| running[[l, j, s]])
    };
    let raw = cs.intervals();
    let raw_side = |s: usize| -> RMatrix<f64> {
        let (looks, k, _) = raw.dim();
        RMatrix::new_matrix(looks, k, |l, j| raw[[l, j, s]])
    };
    // 1-based look of the first exclusion of 0, 0 for none.
    let first_exclusion: Vec<i32> = cs
        .first_exclusion(0.0)
        .into_iter()
        .map(|l| l.map_or(0, |l| l as i32 + 1))
        .collect();
    Ok(list!(
        n = cs.times().iter().map(|&t| t as i32).collect::<Vec<_>>(),
        terms = cs.names().to_vec(),
        coef = to_rmatrix(&cs.coef()),
        se = to_rmatrix(&cs.se()),
        lower = side(0),
        upper = side(1),
        lower_look = raw_side(0),
        upper_look = raw_side(1),
        p = to_rmatrix(&cs.p_values()),
        first_exclusion = first_exclusion,
        opt_time = cs.opt_time() as i32
    ))
}

fn real_matrix(robj: &Robj, arg: &str) -> Result<Option<Array2<f64>>> {
    if robj.is_null() {
        return Ok(None);
    }
    let m = RMatrix::<f64>::try_from(robj.clone())
        .map_err(|_| err(format!("`{arg}` must be a numeric matrix")))?;
    Ok(Some(matrix(&m)))
}

fn real_vector(robj: &Robj, arg: &str) -> Result<Option<Vec<f64>>> {
    if robj.is_null() {
        return Ok(None);
    }
    robj.as_real_vector()
        .map(Some)
        .ok_or_else(|| err(format!("`{arg}` must be a numeric vector")))
}

/// Prediction-powered inference; predictions passed in or cross-fitted
/// from `learner` on `features`.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_fit_ppi(
    estimand: &str,
    y: &[f64],
    yhat: Robj,
    yhat_unlabeled: Robj,
    x: Robj,
    x_unlabeled: Robj,
    names: Robj,
    lambda: Option<f64>,
    learner: Robj,
    use_classifier: bool,
    features: Robj,
    features_unlabeled: Robj,
    n_folds: i32,
    seed_: f64,
) -> Result<List> {
    let estimand = match estimand {
        "mean" => PpiEstimand::Mean,
        "ols" => PpiEstimand::LinearRegression,
        "logistic" => PpiEstimand::LogisticRegression,
        other => return Err(err(format!("unknown estimand {other:?}"))),
    };
    let y = ndarray::Array1::from(y.to_vec());
    let cross_fitted = !learner.is_null();
    let mut data = match optional_list(&learner)? {
        Some(spec) => {
            let f = real_matrix(&features, "features")?.ok_or_else(|| err("need features"))?;
            let fu = real_matrix(&features_unlabeled, "features_unlabeled")?
                .ok_or_else(|| err("need features_unlabeled"))?;
            let folds = n_folds as usize;
            if use_classifier {
                PpiData::cross_fit_proba(
                    classifier(&spec)?.as_ref(),
                    y,
                    f.view(),
                    fu.view(),
                    folds,
                    seed(seed_),
                )
            } else {
                PpiData::cross_fit(
                    regressor(&spec)?.as_ref(),
                    y,
                    f.view(),
                    fu.view(),
                    folds,
                    seed(seed_),
                )
            }
        }
        None => {
            let yh = real_vector(&yhat, "yhat")?.ok_or_else(|| err("need yhat"))?;
            let yu =
                real_vector(&yhat_unlabeled, "yhat_unlabeled")?.ok_or_else(|| err("need yhat_unlabeled"))?;
            PpiData::new(y, yh.into(), yu.into())
        }
    }
    .map_err(err)?;
    if let (Some(x), Some(xu)) = (real_matrix(&x, "x")?, real_matrix(&x_unlabeled, "x_unlabeled")?) {
        data = data.with_covariates(x, xu).map_err(err)?;
    }
    if let Some(names) = names.as_string_vector() {
        data = data.with_covariate_names(names);
    }
    let lambda = lambda.map_or(PpiLambda::Tuned, PpiLambda::Fixed);
    let fit = PredictionPowered::new(estimand)
        .lambda(lambda)
        .fit(&data)
        .map_err(err)?;
    let (yhat_out, yhat_u_out): (Robj, Robj) = if cross_fitted {
        (data.yhat().to_vec().into(), data.yhat_unlabelled().to_vec().into())
    } else {
        (().into(), ().into())
    };
    Ok(list!(
        estimate = ExternalPtr::new(fit.estimate().clone()),
        classical = ExternalPtr::new(fit.classical().clone()),
        lambda = fit.lambda(),
        effective_n = fit.effective_sample_size().to_vec(),
        n_labeled = fit.n_labelled() as i32,
        n_unlabeled = fit.n_unlabelled() as i32,
        yhat = yhat_out,
        yhat_unlabeled = yhat_u_out
    ))
}

extendr_module! {
    mod sequential;
    fn rs_confidence_sequence;
    fn rs_monitor;
    fn rs_fit_ppi;
}

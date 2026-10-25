//! # dml-core
//!
//! Double / debiased machine learning (Chernozhukov et al., 2018) in Rust.
//!
//! ## Models
//!
//! - [`Plr`]: partially linear regression (partialling-out or IV-type score)
//! - [`Irm`]: interactive regression model for a binary treatment (ATE / ATTE)
//! - [`Pliv`]: partially linear IV regression
//! - [`Iivm`]: interactive IV model, local average treatment effect (LATE) with a binary instrument
//! - [`Did`] / [`DidCs`]: doubly robust difference-in-differences for panel data and
//!   repeated cross-sections (Sant'Anna & Zhao, 2020)
//! - [`StaggeredDid`]: DID with staggered adoption, group-time effects `ATT(g, t)` and their
//!   aggregation by cohort, calendar time or event time (Callaway & Sant'Anna, 2021)
//!
//! All of them have scores that are linear in θ and share one engine
//! ([`fit_model`]): repeated cross-fitting, DML2 estimation, standard errors,
//! confidence intervals, a multiplier [`Bootstrap`] for joint inference and
//! multiple-testing adjustment. New scores plug in via [`LinearScoreModel`].
//!
//! ## Clustered data
//!
//! [`DmlData::with_clusters`] / [`DmlData::with_two_way_clusters`] make sample splitting
//! cluster-aware and standard errors cluster-robust (Chiang et al., 2022).
//!
//! ## Sensitivity analysis
//!
//! [`DmlFit::sensitivity_analysis`] bounds the effect of omitted confounders (PLR, IRM, DID)
//! and reports robustness values; `sensitivity_benchmark` calibrates the scenario against
//! observed covariates.
//!
//! ## Heterogeneous effects
//!
//! - [`SLearner`], [`TLearner`], [`XLearner`], [`RLearner`], [`DrLearner`]: CATE meta-learners
//! - [`Blp`]: best linear predictor of the CATE and group average effects (GATE) with
//!   robust inference, from [`DrLearner`] fits or [`DmlFit::cate`] / [`DmlFit::gate`]
//!
//! ## Building blocks
//!
//! - [`DmlData`]: outcome `y`, one or more treatments `d`, covariates `x`, optional instrument `z`
//! - [`Regressor`] / [`Classifier`]: plug-in nuisance learners
//!   ([`Ols`], [`Lasso`], [`LassoCV`], [`LogisticRegression`], [`LogisticRegressionCV`],
//!   [`RandomForestRegressor`],
//!   [`RandomForestClassifier`], [`GradientBoostingRegressor`], [`GradientBoostingClassifier`]
//!   built in)
//! - [`KFold`] / [`SampleSplitting`]: reproducible sample splitting
//!
//! ```
//! use dml_core::{DmlData, Ols, Plr};
//! use ndarray::{Array1, Array2};
//!
//! let n = 200;
//! let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
//! let d: Array1<f64> = (0..n).map(|i| x[[i, 0]] + ((i * 13) % 5) as f64).collect();
//! let y: Array1<f64> = (0..n).map(|i| 0.5 * d[i] - x[[i, 1]] + ((i * 17) % 3) as f64).collect();
//!
//! let fit = Plr::new(Ols::new(), Ols::new()).n_folds(5).seed(42).fit(&DmlData::new(y, d, x)?)?;
//! assert!((fit.coef()[0] - 0.5).abs() < 0.1);
//! let ci = fit.confint(0.95)?; // shape (n_treat, 2)
//! # let _ = ci;
//! # Ok::<(), dml_core::DmlError>(())
//! ```

pub mod blp;
pub mod bootstrap;
pub mod cate;
pub mod cluster;
mod compat;
pub mod crossfit;
pub mod data;
pub mod error;
pub mod fit;
pub mod learner;
pub(crate) mod linalg;
pub mod model;
pub mod policy;
pub mod propensity;
pub mod rd;
pub mod score;
pub mod sensitivity;
pub mod staggered;
pub mod stats;

pub use blp::Blp;
pub use bootstrap::{Bootstrap, BootstrapMethod};
pub use cate::{CateFit, DrLearner, RLearner, SLearner, TLearner, XLearner};
pub use cluster::ClusterFold;
pub use crossfit::{
    cross_fit_predict, cross_fit_predict_proba, cross_fit_predict_proba_subset,
    cross_fit_predict_subset, Fold, KFold, SampleSplitting,
};
pub use data::DmlData;
pub use error::{DmlError, Result};
pub use fit::{fit_model, fit_model_clustered, DmlFit, PAdjust};
pub use learner::{
    Classifier, CvScoring, GradientBoostingClassifier, GradientBoostingRegressor, Lasso, LassoCV,
    LogisticRegression, LogisticRegressionCV, MaxFeatures, Ols, RandomForestClassifier,
    RandomForestRegressor, Regressor,
};
pub use model::{
    Apo, Apos, AposFit, CateSignal, Cvar, Did, DidCs, DidScore, FirstStage, Iivm, Irm, IrmScore,
    LinearScoreModel, LossKind, Lplr, LplrScore, Lpq, Name, NuisanceTarget, PanelApproach, Pliv,
    PlivScore, PlivVariant, Plpr, Plr, PlrScore, Pq, Qte, QteFit, RdFlex, RdFlexFit, RdKernel,
    ScoreEstimate, ScoreInput, Ssm, SsmScore, StaticPanel, TreatmentLearner,
};
pub use policy::PolicyTree;
pub use propensity::{Calibration, IsotonicRegression};
pub use score::LinearScore;
pub use sensitivity::{
    Nu2Parts, SensitivityBenchmark, SensitivityParams, SensitivityParts, SensitivityResult,
};
pub use staggered::{
    Aggregation, ControlGroup, DidAggregation, GroupTime, GtCombinations, PanelData,
    RepeatedCrossSections, StaggeredDid, StaggeredDidFit, StaggeredInput,
};

//! # dml-core
//!
//! Double / debiased machine learning (Chernozhukov et al., 2018) in Rust.
//!
//! ## Models
//!
//! - [`Plr`]: partially linear regression (partialling-out or IV-type score)
//! - [`Irm`]: interactive regression model for a binary treatment (ATE / ATTE)
//! - [`Pliv`]: partially linear IV regression
//!
//! All three have scores that are linear in θ and share one engine
//! ([`fit_model`]): repeated cross-fitting, DML2 estimation, standard errors,
//! confidence intervals, a multiplier [`Bootstrap`] for joint inference and
//! multiple-testing adjustment. New scores plug in via [`LinearScoreModel`].
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
//!   ([`Ols`], [`Lasso`], [`LogisticRegression`], [`RandomForestRegressor`],
//!   [`RandomForestClassifier`] built in)
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
pub mod crossfit;
pub mod data;
pub mod error;
pub mod fit;
pub mod learner;
pub(crate) mod linalg;
pub mod model;
pub mod score;
pub mod stats;

pub use blp::Blp;
pub use bootstrap::{Bootstrap, BootstrapMethod};
pub use cate::{CateFit, DrLearner, RLearner, SLearner, TLearner, XLearner};
pub use crossfit::{
    cross_fit_predict, cross_fit_predict_proba, cross_fit_predict_subset, Fold, KFold,
    SampleSplitting,
};
pub use data::DmlData;
pub use error::{DmlError, Result};
pub use fit::{fit_model, DmlFit, PAdjust};
pub use learner::{
    Classifier, Lasso, LogisticRegression, MaxFeatures, Ols, RandomForestClassifier,
    RandomForestRegressor, Regressor,
};
pub use model::{Irm, IrmScore, LinearScoreModel, Pliv, Plr, PlrScore, ScoreEstimate, ScoreInput};
pub use score::LinearScore;

//! # dml-core
//!
//! Double / debiased machine learning (Chernozhukov et al., 2018) in Rust.
//!
//! Phase 1 provides the building blocks every DML model needs:
//!
//! - [`DmlData`]: outcome `y`, treatment `d`, covariates `x`
//! - [`Regressor`] / [`Classifier`]: plug-in nuisance learners
//! - [`KFold`]: reproducible sample splitting for cross-fitting
//! - [`Ols`]: a built-in baseline learner

pub mod crossfit;
pub mod data;
pub mod error;
pub mod learner;
pub(crate) mod linalg;

pub use crossfit::{cross_fit_predict, Fold, KFold};
pub use data::DmlData;
pub use error::{DmlError, Result};
pub use learner::{Classifier, Ols, Regressor};

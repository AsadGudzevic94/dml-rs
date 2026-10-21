//! Nuisance learners: the [`Regressor`] / [`Classifier`] traits and the
//! built-in implementations.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::error::Result;

mod lasso;
mod logistic;
mod ols;

pub use lasso::Lasso;
pub use logistic::LogisticRegression;
pub use ols::Ols;

/// A regression learner for nuisance functions such as `E[Y | X]` or `E[D | X]`.
///
/// `fresh` returns an unfitted copy with the same hyperparameters, so
/// cross-fitting can train an independent model on each fold.
///
/// `Send + Sync` lets folds be fitted on parallel threads.
pub trait Regressor: Send + Sync {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()>;
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
    fn fresh(&self) -> Box<dyn Regressor>;
}

/// A classification learner returning `P(label = 1 | X)`, used for
/// propensity scores in models with a binary treatment such as [`Irm`](crate::Irm).
pub trait Classifier: Send + Sync {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()>;
    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
    fn fresh(&self) -> Box<dyn Classifier>;
}

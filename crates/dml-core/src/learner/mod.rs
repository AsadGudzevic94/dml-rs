//! Nuisance learners: the [`Regressor`] / [`Classifier`] traits and the
//! built-in implementations.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{DmlError, Result};

mod boosting;
mod cv;
mod forest;
mod lasso;
mod logistic;
mod ols;

pub use boosting::{GradientBoostingClassifier, GradientBoostingRegressor};
pub use cv::{CvScoring, LassoCV, LogisticRegressionCV};
pub use forest::{MaxFeatures, RandomForestClassifier, RandomForestRegressor};
pub(crate) use forest::{Node, Tree, TreeBuilder, TreeParams};
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

    /// Fit minimising a `w`-weighted loss (needed by the [`RLearner`](crate::RLearner)).
    /// The default reports that weights are unsupported.
    fn fit_weighted(
        &mut self,
        _x: ArrayView2<'_, f64>,
        _y: ArrayView1<'_, f64>,
        _w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        Err(DmlError::InvalidArgument(
            "this learner does not support sample weights".into(),
        ))
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
    fn fresh(&self) -> Box<dyn Regressor>;
}

/// A classification learner returning `P(label = 1 | X)`, used for
/// propensity scores in models with a binary treatment such as [`Irm`](crate::Irm).
pub trait Classifier: Send + Sync {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()>;

    /// Fit minimising a `w`-weighted loss (needed by [`Lplr`](crate::Lplr)'s
    /// instrument score). The default reports that weights are unsupported.
    fn fit_weighted(
        &mut self,
        _x: ArrayView2<'_, f64>,
        _y: ArrayView1<'_, f64>,
        _w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        Err(DmlError::InvalidArgument(
            "this learner does not support sample weights".into(),
        ))
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
    fn fresh(&self) -> Box<dyn Classifier>;
}

/// Centred (and, with weights, `√w`-scaled) data for the linear learners, so
/// the intercept drops out: `b = ȳ − x̄ᵀβ`.
pub(crate) struct Centered {
    pub x_mean: Array1<f64>,
    pub y_mean: f64,
    pub xc: Array2<f64>,
    pub yc: Array1<f64>,
}

/// With `w`, rows are scaled by `√w` after weighted centring. If
/// `normalize_weights`, weights are first rescaled to sum to `n` (as
/// scikit-learn's `Lasso` does; `Ridge` uses them as given).
pub(crate) fn center(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    w: Option<ArrayView1<'_, f64>>,
    normalize_weights: bool,
) -> Result<Centered> {
    let n = x.nrows();
    if y.len() != n {
        return Err(DmlError::DimensionMismatch("fit x/y rows".into()));
    }
    if n == 0 {
        return Err(DmlError::InvalidArgument("empty training set".into()));
    }
    let Some(w) = w else {
        let x_mean = x.mean_axis(Axis(0)).expect("n > 0");
        let y_mean = y.mean().expect("n > 0");
        return Ok(Centered {
            xc: &x - &x_mean,
            yc: &y - y_mean,
            x_mean,
            y_mean,
        });
    };
    if w.len() != n {
        return Err(DmlError::DimensionMismatch("sample weight length".into()));
    }
    let total = w.sum();
    if !w.iter().all(|&v| v >= 0.0 && v.is_finite()) || total <= 0.0 {
        return Err(DmlError::InvalidArgument(
            "sample weights must be non-negative, finite and not all zero".into(),
        ));
    }
    let w: Array1<f64> = if normalize_weights {
        &w * (n as f64 / total)
    } else {
        w.to_owned()
    };
    let total = w.sum();
    let x_mean = x.t().dot(&w) / total;
    let y_mean = y.dot(&w) / total;
    let sqrt_w = w.mapv(f64::sqrt);
    let xc = (&x - &x_mean) * sqrt_w.view().insert_axis(Axis(1));
    let yc = (&y - y_mean) * &sqrt_w;
    Ok(Centered {
        x_mean,
        y_mean,
        xc,
        yc,
    })
}

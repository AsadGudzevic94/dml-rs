//! DML models with linear scores: [`Plr`], [`Irm`], [`Pliv`], and the
//! [`LinearScoreModel`] trait for adding your own.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::crossfit::Fold;
use crate::data::DmlData;
use crate::error::Result;
use crate::score::LinearScore;

pub(crate) mod did;
mod iivm;
pub(crate) mod irm;
mod pliv;
mod plr;

pub use did::{Did, DidCs, DidScore};
pub use iivm::Iivm;
pub use irm::{Irm, IrmScore};
pub use pliv::Pliv;
pub use plr::{Plr, PlrScore};

/// The data a model sees when estimating the effect of one treatment.
///
/// `#[non_exhaustive]`: fields may be added in future versions, so code outside
/// this crate reads the fields but cannot construct the struct.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ScoreInput<'a> {
    pub y: ArrayView1<'a, f64>,
    /// The treatment being estimated.
    pub d: ArrayView1<'a, f64>,
    /// Covariates, including any other treatments.
    pub x: ArrayView2<'a, f64>,
    pub z: Option<ArrayView1<'a, f64>>,
    /// 0/1 time-period indicator (repeated cross-sections).
    pub t: Option<ArrayView1<'a, f64>>,
}

/// Score elements plus the cross-fitted nuisance predictions behind them.
#[derive(Debug, Clone)]
pub struct ScoreEstimate {
    pub score: LinearScore,
    /// `(name, out-of-fold predictions)` — e.g. `("ml_l", ℓ̂)`.
    pub predictions: Vec<(&'static str, Array1<f64>)>,
}

/// A DML model whose orthogonal score is linear in θ.
///
/// Implement this to plug a new score into the shared machinery (repeated
/// cross-fitting, aggregation, inference, bootstrap) via [`fit_model`](crate::fit_model).
pub trait LinearScoreModel {
    /// Label for summaries, e.g. `"PLR (partialling out)"`.
    fn name(&self) -> String;

    /// Reject data the model cannot handle (called once before fitting).
    fn validate(&self, _data: &DmlData) -> Result<()> {
        Ok(())
    }

    /// Cross-fit the nuisances on `folds` and return the score elements.
    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate>;
}

/// Builder methods shared by every model: sample-splitting settings and `fit`.
///
/// (`macro_rules!` generates code at compile time; here it pastes the same
/// methods into each model's `impl` block instead of repeating them.)
macro_rules! dml_model_api {
    () => {
        /// Number of cross-fitting folds (default 5).
        pub fn n_folds(mut self, n_folds: usize) -> Self {
            self.splitting.n_folds = n_folds;
            self
        }

        /// Number of independent sample splits (default 1). Estimates are
        /// aggregated across repetitions by the median, as in DoubleML.
        pub fn n_rep(mut self, n_rep: usize) -> Self {
            self.splitting.n_rep = n_rep;
            self
        }

        /// Seed for the sample splits (default 0).
        pub fn seed(mut self, seed: u64) -> Self {
            self.splitting.seed = seed;
            self
        }

        /// Draw seeded sample splits and fit.
        pub fn fit(&self, data: &$crate::DmlData) -> $crate::Result<$crate::DmlFit> {
            let splits = self.splitting.draw(data.n_obs())?;
            $crate::fit_model(self, data, &splits)
        }

        /// Fit with caller-supplied splits, one `Vec<Fold>` per repetition
        /// (e.g. to reproduce another implementation exactly).
        pub fn fit_with_folds(
            &self,
            data: &$crate::DmlData,
            splits: &[Vec<$crate::Fold>],
        ) -> $crate::Result<$crate::DmlFit> {
            $crate::fit_model(self, data, splits)
        }
    };
}
pub(crate) use dml_model_api;

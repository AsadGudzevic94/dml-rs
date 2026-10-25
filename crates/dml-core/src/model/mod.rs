//! DML models with linear scores: [`Plr`], [`Irm`], [`Pliv`], and the
//! [`LinearScoreModel`] trait for adding your own.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::crossfit::Fold;
use crate::data::DmlData;
use crate::error::Result;
use crate::score::LinearScore;
use crate::sensitivity::SensitivityParts;

mod apo;
pub(crate) mod did;
mod iivm;
pub(crate) mod irm;
mod lplr;
mod nonlinear;
mod pliv;
mod plr;
mod quantile;
mod ssm;

pub use apo::{Apo, Apos, AposFit};
pub use did::{Did, DidCs, DidScore};
pub use iivm::Iivm;
pub use irm::{Irm, IrmScore};
pub use lplr::{Lplr, LplrScore, TreatmentLearner};
pub use pliv::Pliv;
pub use plr::{Plr, PlrScore};
pub use quantile::{Cvar, Lpq, Pq, Qte, QteFit};
pub use ssm::{Ssm, SsmScore};

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
    /// 0/1 selection indicator (sample selection models).
    pub s: Option<ArrayView1<'a, f64>>,
    /// Index of the sample-splitting repetition being estimated.
    pub rep: usize,
}

/// Ingredients for best-linear-predictor inference on effect heterogeneity
/// ([`DmlFit::cate`](crate::DmlFit::cate), [`DmlFit::gate`](crate::DmlFit::gate)):
/// the BLP regresses `signal − θ·theta_factor` on the basis, with every row of
/// the basis multiplied by `basis_scale` if given (partially linear models,
/// where the signal is the outcome residual and the scale the treatment
/// residual).
#[derive(Debug, Clone)]
pub struct CateSignal {
    pub signal: Array1<f64>,
    pub theta_factor: Option<Array1<f64>>,
    pub basis_scale: Option<Array1<f64>>,
    /// The signal is a (doubly robust) signal of each unit's treatment effect,
    /// so a policy tree may be fitted on it ([`DmlFit::policy_tree`](crate::DmlFit::policy_tree)).
    pub is_effect: bool,
}

/// Score elements plus the cross-fitted nuisance predictions behind them.
///
/// Build it with [`ScoreEstimate::new`] (the struct is `#[non_exhaustive]`, so
/// fields can be added without breaking custom models).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ScoreEstimate {
    pub score: LinearScore,
    /// `(name, out-of-fold predictions)` — e.g. `("ml_l", ℓ̂)`.
    pub predictions: Vec<(&'static str, Array1<f64>)>,
    /// Ingredients for [sensitivity analysis](crate::sensitivity), if the
    /// model supports it.
    pub sensitivity: Option<SensitivityParts>,
    /// What each nuisance learner predicts, for out-of-sample diagnostics.
    pub targets: Vec<NuisanceTarget>,
    /// Signal for CATE / GATE inference, if the model supports it.
    pub cate: Option<CateSignal>,
}

/// Loss used to evaluate a nuisance learner out of sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossKind {
    /// Root mean squared error (regression learners).
    Rmse,
    /// Log loss with predictions clipped to `[1e-15, 1 − 1e-15]` (classifiers).
    LogLoss,
}

/// The target a nuisance learner predicts; `NaN` marks rows where the
/// nuisance does not apply (e.g. `g₀` on treated rows).
#[derive(Debug, Clone)]
pub struct NuisanceTarget {
    pub name: &'static str,
    pub target: Array1<f64>,
    pub kind: LossKind,
}

impl NuisanceTarget {
    pub fn regression(name: &'static str, target: Array1<f64>) -> Self {
        Self {
            name,
            target,
            kind: LossKind::Rmse,
        }
    }

    pub fn classification(name: &'static str, target: Array1<f64>) -> Self {
        Self {
            name,
            target,
            kind: LossKind::LogLoss,
        }
    }

    /// Keep the target only where `keep` is true (`NaN` elsewhere).
    pub fn only_where(mut self, keep: impl Fn(usize) -> bool) -> Self {
        for (i, v) in self.target.iter_mut().enumerate() {
            if !keep(i) {
                *v = f64::NAN;
            }
        }
        self
    }

    /// Out-of-sample loss of `prediction` over the rows with a target.
    pub fn loss(&self, prediction: &Array1<f64>) -> f64 {
        let mut total = 0.0;
        let mut count = 0usize;
        for (&t, &p) in self.target.iter().zip(prediction) {
            if t.is_nan() {
                continue;
            }
            total += match self.kind {
                LossKind::Rmse => (t - p).powi(2),
                LossKind::LogLoss => {
                    let p = p.clamp(1e-15, 1.0 - 1e-15);
                    -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                }
            };
            count += 1;
        }
        if count == 0 {
            return f64::NAN;
        }
        let mean = total / count as f64;
        match self.kind {
            LossKind::Rmse => mean.sqrt(),
            LossKind::LogLoss => mean,
        }
    }
}

impl ScoreEstimate {
    pub fn new(score: LinearScore, predictions: Vec<(&'static str, Array1<f64>)>) -> Self {
        Self {
            score,
            predictions,
            sensitivity: None,
            targets: Vec::new(),
            cate: None,
        }
    }

    /// Attach the signal for CATE / GATE inference.
    pub fn with_cate_signal(mut self, cate: CateSignal) -> Self {
        self.cate = Some(cate);
        self
    }

    /// Attach the nuisance targets used for learner diagnostics.
    pub fn with_targets(mut self, targets: Vec<NuisanceTarget>) -> Self {
        self.targets = targets;
        self
    }

    /// Attach the model's sensitivity ingredients.
    pub fn with_sensitivity(mut self, parts: SensitivityParts) -> Self {
        self.sensitivity = Some(parts);
        self
    }
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

    /// Strata for the default sample splits (e.g. treatment levels), so each
    /// fold holds about the same share of every stratum. `None` (the default)
    /// draws plain K-fold splits.
    fn strata(&self, _data: &DmlData) -> Option<Vec<usize>> {
        None
    }
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

        /// Draw seeded sample splits and fit. With clustered data
        /// ([`DmlData::with_clusters`](crate::DmlData::with_clusters)) whole clusters are assigned to folds
        /// and standard errors are cluster-robust.
        pub fn fit(&self, data: &$crate::DmlData) -> $crate::Result<$crate::DmlFit> {
            match data.clusters() {
                Some(clusters) => {
                    let splits = self.splitting.draw_clustered(clusters)?;
                    $crate::fit_model_clustered(self, data, &splits)
                }
                None => {
                    let splits = match $crate::LinearScoreModel::strata(self, data) {
                        Some(strata) => self.splitting.draw_stratified(&strata)?,
                        None => self.splitting.draw(data.n_obs())?,
                    };
                    $crate::fit_model(self, data, &splits)
                }
            }
        }

        /// Fit clustered data with explicit cluster folds, one
        /// `Vec<ClusterFold>` per repetition.
        pub fn fit_with_cluster_folds(
            &self,
            data: &$crate::DmlData,
            splits: &[Vec<$crate::ClusterFold>],
        ) -> $crate::Result<$crate::DmlFit> {
            $crate::fit_model_clustered(self, data, splits)
        }

        /// Sensitivity benchmark for the covariate columns `benchmarking_set`:
        /// refits this model without them (with the same sample splitting, so
        /// `long` must come from [`fit`](Self::fit) with the current settings)
        /// and compares it with `long`.
        pub fn sensitivity_benchmark(
            &self,
            data: &$crate::DmlData,
            long: &$crate::DmlFit,
            benchmarking_set: &[usize],
        ) -> $crate::Result<$crate::SensitivityBenchmark> {
            let short = self.fit(&data.without_covariates(benchmarking_set)?)?;
            long.sensitivity_benchmark(&short)
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

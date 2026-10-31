//! Generalized random forests (Athey, Tibshirani and Wager, 2019; Wager and
//! Athey, 2018), as R's `grf`: causal, instrumental, multi-arm causal,
//! regression and quantile forests.
//!
//! Every forest solves a local estimating equation at the target point `x`,
//! weighting the training rows by how often they share a leaf with `x`. The
//! trees are *honest*: each takes a subsample, half of which chooses the
//! splits and the other half estimates the leaves. Splits come from grf's
//! gradient approximation: the parent node's estimate is computed once, every
//! row gets a pseudo-outcome `ρᵢ` (its influence on that estimate), and a
//! regression split on `ρ` follows.
//!
//! - [`CausalForest`]: `τ(x)` in `Y = τ(X)·W + m(X) + ε` on residuals,
//!   `ρᵢ = (W̃ᵢ − W̄)(Ỹᵢ − Ȳ − θ̂ (W̃ᵢ − W̄))`; [`CausalForestDml`] adds the
//!   cross-fitted residuals and doubly robust inference.
//! - [`InstrumentalForest`]: the conditional local average treatment effect
//!   with an instrument `Z`, `θ(x) = Cov(Z, Y | x) / Cov(Z, W | x)`
//!   ([`InstrumentalForestDml`]).
//! - [`MultiArmCausalForest`]: effects of several treatment arms against a
//!   baseline, `θ(x) = Var(W | x)⁻¹ Cov(W, Y | x)` with a one-hot `W`
//!   ([`MultiArmCausalForestDml`]).
//! - [`RegressionForest`]: honest regression forest (a [`Regressor`](crate::learner::Regressor)
//!   with variance estimates).
//! - [`QuantileForest`]: conditional quantiles from the forest weights.
//!
//! Trees are grown in little bags (groups of `ci_group_size` trees on one
//! half-sample), which gives variance estimates (the bootstrap of little
//! bags) for the causal, instrumental, multi-arm and regression forests.

mod dml;
mod engine;
mod forests;
#[cfg(test)]
mod tests;

pub use dml::{
    CausalForestDml, CausalForestFit, InstrumentalForestDml, InstrumentalForestFit,
    MultiArmCausalForestDml, MultiArmCausalForestFit,
};
pub use forests::{
    CausalForest, InstrumentalForest, MultiArmCausalForest, QuantileForest, RegressionForest,
};

use crate::error::{DmlError, Result};

/// Settings of a generalized random forest (grf's defaults).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForestSettings {
    pub num_trees: usize,
    pub sample_fraction: f64,
    /// Variables tried per split; `None`: `min(⌈√p + 20⌉, p)`.
    pub mtry: Option<usize>,
    pub min_node_size: usize,
    pub honesty: bool,
    pub honesty_fraction: f64,
    pub prune_empty_leaves: bool,
    pub alpha: f64,
    pub imbalance_penalty: f64,
    pub stabilize_splits: bool,
    pub ci_group_size: usize,
    pub seed: u64,
}

impl Default for ForestSettings {
    fn default() -> Self {
        Self {
            num_trees: 2000,
            sample_fraction: 0.5,
            mtry: None,
            min_node_size: 5,
            honesty: true,
            honesty_fraction: 0.5,
            prune_empty_leaves: true,
            alpha: 0.05,
            imbalance_penalty: 0.0,
            stabilize_splits: true,
            ci_group_size: 2,
            seed: 0,
        }
    }
}

impl ForestSettings {
    fn validate(&self, n: usize) -> Result<()> {
        let bad = |m: &str| Err(DmlError::InvalidArgument(m.into()));
        if self.ci_group_size == 0 || self.num_trees < self.ci_group_size {
            return bad("num_trees must be at least ci_group_size (and ci_group_size at least 1)");
        }
        if self.sample_fraction.is_nan()
            || self.sample_fraction <= 0.0
            || self.sample_fraction > 1.0
        {
            return bad("sample_fraction must be in (0, 1]");
        }
        if self.ci_group_size > 1 && self.sample_fraction > 0.5 {
            return bad("sample_fraction must be at most 0.5 when ci_group_size > 1");
        }
        if self.honesty_fraction.is_nan()
            || self.honesty_fraction <= 0.0
            || self.honesty_fraction >= 1.0
        {
            return bad("honesty_fraction must be in (0, 1)");
        }
        if self.alpha.is_nan() || !(0.0..0.25).contains(&self.alpha) {
            return bad("alpha must be in [0, 0.25)");
        }
        if self.imbalance_penalty.is_nan() || self.imbalance_penalty < 0.0 {
            return bad("imbalance_penalty must be non-negative");
        }
        if self.mtry == Some(0) {
            return bad("mtry must be at least 1");
        }
        if n < 4 {
            return bad("a forest needs at least 4 observations");
        }
        Ok(())
    }
}

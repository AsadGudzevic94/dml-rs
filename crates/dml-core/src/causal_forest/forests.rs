//! The forests on given data: causal, instrumental, multi-arm causal,
//! regression and quantile.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::engine::{Grf, Target};
use super::ForestSettings;
use crate::error::{DmlError, Result};
use crate::learner::Regressor;

fn column(a: Array2<f64>) -> Array1<f64> {
    a.column(0).to_owned()
}

/// A causal forest grown on residuals; see the [module docs](super).
#[derive(Debug, Clone)]
pub struct CausalForest {
    grf: Grf,
}

impl CausalForest {
    /// Grow a forest of `settings.num_trees` honest trees on covariates `x` and
    /// residuals `y` (outcome) and `w` (treatment).
    pub fn fit<'a>(
        x: ArrayView2<'a, f64>,
        y: ArrayView1<'a, f64>,
        w: ArrayView1<'a, f64>,
        settings: ForestSettings,
    ) -> Result<Self> {
        if w.len() != x.nrows() {
            return Err(DmlError::DimensionMismatch(
                "causal forest x / y / w rows".into(),
            ));
        }
        let w2 = w.insert_axis(Axis(1));
        let target = Target::Moment {
            y,
            w: w2,
            z: w2,
            reduced_form_weight: 0.0,
            iv: false,
        };
        Ok(Self {
            grf: Grf::fit(x, target, settings)?,
        })
    }

    pub fn settings(&self) -> &ForestSettings {
        self.grf.settings()
    }

    /// Number of trees grown.
    pub fn n_trees(&self) -> usize {
        self.grf.n_trees()
    }

    /// Split-frequency variable importance (sums to one when there are splits).
    pub fn variable_importance(&self) -> &Array1<f64> {
        self.grf.importance()
    }

    /// `τ̂(x)` and its variance for the rows of `x` (all trees). Variances are
    /// `NaN` with `ci_group_size = 1`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (t, v) = self.grf.predict_moments(x, false)?;
        Ok((column(t), column(v)))
    }

    /// Out-of-bag `τ̂(Xᵢ)` and variance for the training rows `x` (in training
    /// order): each estimate uses only the trees whose subsample left row `i`
    /// out.
    pub fn predict_oob(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (t, v) = self.grf.predict_moments(x, true)?;
        Ok((column(t), column(v)))
    }
}

/// An instrumental forest grown on residuals: the conditional local average
/// treatment effect `θ(x) = Cov(Z, Y | x) / Cov(Z, W | x)`, as grf's
/// `instrumental_forest`.
#[derive(Debug, Clone)]
pub struct InstrumentalForest {
    grf: Grf,
}

impl InstrumentalForest {
    /// Grow the forest on covariates `x` and residuals `y` (outcome), `w`
    /// (treatment) and `z` (instrument). The splits use the pseudo-outcomes of
    /// the instrument `(1 − r) Z + r W` with `r = reduced_form_weight` (grf's
    /// default 0); the children must keep a share `alpha` of the parent's
    /// instrument variation.
    pub fn fit<'a>(
        x: ArrayView2<'a, f64>,
        y: ArrayView1<'a, f64>,
        w: ArrayView1<'a, f64>,
        z: ArrayView1<'a, f64>,
        settings: ForestSettings,
        reduced_form_weight: f64,
    ) -> Result<Self> {
        if w.len() != x.nrows() || z.len() != x.nrows() {
            return Err(DmlError::DimensionMismatch(
                "instrumental forest x / y / w / z rows".into(),
            ));
        }
        if !(0.0..=1.0).contains(&reduced_form_weight) {
            return Err(DmlError::InvalidArgument(
                "reduced_form_weight must be in [0, 1]".into(),
            ));
        }
        let target = Target::Moment {
            y,
            w: w.insert_axis(Axis(1)),
            z: z.insert_axis(Axis(1)),
            reduced_form_weight,
            iv: true,
        };
        Ok(Self {
            grf: Grf::fit(x, target, settings)?,
        })
    }

    pub fn settings(&self) -> &ForestSettings {
        self.grf.settings()
    }

    pub fn n_trees(&self) -> usize {
        self.grf.n_trees()
    }

    pub fn variable_importance(&self) -> &Array1<f64> {
        self.grf.importance()
    }

    /// `θ̂(x)` and its variance.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (t, v) = self.grf.predict_moments(x, false)?;
        Ok((column(t), column(v)))
    }

    /// Out-of-bag `θ̂(Xᵢ)` and variance for the training rows.
    pub fn predict_oob(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let (t, v) = self.grf.predict_moments(x, true)?;
        Ok((column(t), column(v)))
    }
}

/// A multi-arm causal forest grown on residuals: the effects of `k` treatment
/// arms against a baseline, `θ(x) = Var(W | x)⁻¹ Cov(W, Y | x)` for the
/// residualised one-hot arm indicators `W` (`n × k`), as grf's
/// `multi_arm_causal_forest`. The splits use the vector pseudo-outcomes
/// `A⁻¹ (Wᵢ − W̄) rᵢ` of all arms; each child keeps a share `alpha` of the
/// parent's variation of every arm.
#[derive(Debug, Clone)]
pub struct MultiArmCausalForest {
    grf: Grf,
}

impl MultiArmCausalForest {
    pub fn fit<'a>(
        x: ArrayView2<'a, f64>,
        y: ArrayView1<'a, f64>,
        w: ArrayView2<'a, f64>,
        settings: ForestSettings,
    ) -> Result<Self> {
        if w.nrows() != x.nrows() || w.ncols() == 0 {
            return Err(DmlError::DimensionMismatch(
                "multi-arm forest needs one row of arm residuals per observation".into(),
            ));
        }
        let target = Target::Moment {
            y,
            w,
            z: w,
            reduced_form_weight: 0.0,
            iv: false,
        };
        Ok(Self {
            grf: Grf::fit(x, target, settings)?,
        })
    }

    pub fn settings(&self) -> &ForestSettings {
        self.grf.settings()
    }

    pub fn n_trees(&self) -> usize {
        self.grf.n_trees()
    }

    pub fn variable_importance(&self) -> &Array1<f64> {
        self.grf.importance()
    }

    /// `θ̂(x)` and the variances of its components, each `n × k`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        self.grf.predict_moments(x, false)
    }

    /// Out-of-bag estimates for the training rows.
    pub fn predict_oob(&self, x: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        self.grf.predict_moments(x, true)
    }
}

/// An honest regression forest (grf's `regression_forest`), usable as a
/// nuisance [`Regressor`]. Besides predictions it gives their variance
/// ([`predict_with_variance`](Self::predict_with_variance)) and out-of-bag
/// predictions.
#[derive(Debug, Clone, Default)]
pub struct RegressionForest {
    settings: ForestSettings,
    grf: Option<Grf>,
}

impl RegressionForest {
    /// grf's defaults ([`ForestSettings::default`]).
    pub fn new() -> Self {
        Self::default()
    }

    pub fn settings(mut self, settings: ForestSettings) -> Self {
        self.settings = settings;
        self
    }

    fn fitted(&self) -> Result<&Grf> {
        self.grf
            .as_ref()
            .ok_or_else(|| DmlError::InvalidArgument("the forest is not fitted".into()))
    }

    /// Predictions and their variances (bootstrap of little bags).
    pub fn predict_with_variance(
        &self,
        x: ArrayView2<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        self.fitted()?.predict_regression(x, false)
    }

    /// Out-of-bag predictions and variances for the training rows.
    pub fn predict_oob(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        self.fitted()?.predict_regression(x, true)
    }

    /// Split-frequency variable importance, after fitting.
    pub fn variable_importance(&self) -> Option<&Array1<f64>> {
        self.grf.as_ref().map(Grf::importance)
    }
}

impl Regressor for RegressionForest {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.grf = Some(Grf::fit(
            x,
            Target::Regression { y },
            self.settings.clone(),
        )?);
        Ok(())
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        Ok(self.fitted()?.predict_regression(x, false)?.0)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self {
            settings: self.settings.clone(),
            grf: None,
        })
    }
}

/// A quantile forest (grf's `quantile_forest`; Meinshausen, 2006, with grf's
/// splitting rule): conditional quantiles of `Y` from the forest weights.
/// Splits separate the node's `quantiles` (default 0.1, 0.5, 0.9); any
/// quantiles can be predicted.
#[derive(Debug, Clone)]
pub struct QuantileForest {
    settings: ForestSettings,
    quantiles: Vec<f64>,
    grf: Option<Grf>,
}

impl Default for QuantileForest {
    fn default() -> Self {
        Self::new(vec![0.1, 0.5, 0.9])
    }
}

impl QuantileForest {
    /// A forest splitting on `quantiles`, with grf's defaults (no little bags:
    /// `ci_group_size = 1`).
    pub fn new(quantiles: Vec<f64>) -> Self {
        Self {
            settings: ForestSettings {
                ci_group_size: 1,
                ..Default::default()
            },
            quantiles,
            grf: None,
        }
    }

    pub fn settings(mut self, settings: ForestSettings) -> Self {
        self.settings = settings;
        self
    }

    pub fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        // Views are invariant in their lifetime: re-borrow `y` so it shares
        // the (shorter) lifetime of the local quantiles.
        let quantiles = self.quantiles.clone();
        let target = Target::Quantile {
            y: y.view(),
            quantiles: &quantiles,
        };
        self.grf = Some(Grf::fit(x, target, self.settings.clone())?);
        Ok(())
    }

    fn fitted(&self) -> Result<&Grf> {
        self.grf
            .as_ref()
            .ok_or_else(|| DmlError::InvalidArgument("the forest is not fitted".into()))
    }

    /// Conditional quantiles at the rows of `x`, one column per quantile.
    pub fn predict(&self, x: ArrayView2<'_, f64>, quantiles: &[f64]) -> Result<Array2<f64>> {
        self.fitted()?.predict_quantiles(x, quantiles, false)
    }

    /// Out-of-bag quantiles for the training rows.
    pub fn predict_oob(&self, x: ArrayView2<'_, f64>, quantiles: &[f64]) -> Result<Array2<f64>> {
        self.fitted()?.predict_quantiles(x, quantiles, true)
    }

    pub fn variable_importance(&self) -> Option<&Array1<f64>> {
        self.grf.as_ref().map(Grf::importance)
    }
}

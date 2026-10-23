//! Doubly robust difference-in-differences (Sant'Anna & Zhao, 2020) for the
//! average treatment effect on the treated (ATT) in a two-period design.
//!
//! - [`Did`]: **panel data**. The outcome passed to [`DmlData`] is the change
//!   `Y = Y₁ − Y₀` of each unit between the two periods.
//! - [`DidCs`]: **repeated cross-sections**. Each row is observed once, in the
//!   pre- (`t = 0`) or post-period (`t = 1`), given by
//!   [`DmlData::with_time_indicator`].
//!
//! Identification rests on conditional parallel trends: absent treatment,
//! treated and control units with the same covariates would have evolved alike.

use ndarray::{Array1, ArrayView1};

use super::{dml_model_api, LinearScoreModel, ScoreEstimate, ScoreInput};
use crate::crossfit::{cross_fit_predict_proba, cross_fit_predict_subset, Fold, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};
use crate::score::LinearScore;

/// Assignment mechanism assumed by [`Did`] / [`DidCs`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DidScore {
    /// Treatment depends on covariates: the propensity `P(D = 1 | X)` is
    /// estimated with `ml_m` (doubly robust score).
    #[default]
    Observational,
    /// Treatment is randomised: no propensity model is needed.
    Experimental,
}

fn mean(v: &Array1<f64>) -> f64 {
    v.mean().expect("n > 0")
}

/// Shared validation for both DID estimators.
fn validate_common(
    data: &DmlData,
    score: DidScore,
    has_ml_m: bool,
    clipping_threshold: f64,
) -> Result<()> {
    if data.n_treat() != 1 || !data.is_binary_treatment(0) {
        return Err(DmlError::InvalidArgument(
            "DID requires exactly one binary (0/1) treatment group indicator".into(),
        ));
    }
    if score == DidScore::Observational && !has_ml_m {
        return Err(DmlError::InvalidArgument(
            "the observational DID score needs a propensity learner ml_m".into(),
        ));
    }
    if !(clipping_threshold > 0.0 && clipping_threshold < 0.5) {
        return Err(DmlError::InvalidArgument(
            "clipping_threshold must be in (0, 0.5)".into(),
        ));
    }
    Ok(())
}

/// Builder methods shared by [`Did`] and [`DidCs`].
macro_rules! did_builders {
    () => {
        /// Observational (default) or experimental score.
        pub fn score(mut self, score: DidScore) -> Self {
            self.score = score;
            self
        }

        /// Normalise the weights by their sample means (default `true`, as
        /// DoubleML); `false` uses the plug-in probabilities.
        pub fn in_sample_normalization(mut self, normalize: bool) -> Self {
            self.in_sample_normalization = normalize;
            self
        }

        /// Propensities are clipped to `[t, 1 − t]` (default 0.01).
        pub fn clipping_threshold(mut self, t: f64) -> Self {
            self.clipping_threshold = t;
            self
        }
    };
}

/// Difference-in-differences with **panel data** (two periods): `y` must be
/// the outcome change `Y₁ − Y₀`. `ml_g` learns `E[ΔY | D = d, X]` per group,
/// `ml_m` the propensity `P(D = 1 | X)` (observational score only).
pub struct Did {
    ml_g: Box<dyn Regressor>,
    ml_m: Option<Box<dyn Classifier>>,
    score: DidScore,
    in_sample_normalization: bool,
    clipping_threshold: f64,
    splitting: SampleSplitting,
}

impl Did {
    /// Observational design (doubly robust score).
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Some(Box::new(ml_m)))
    }

    /// Randomised design: only the outcome model is needed.
    pub fn experimental(ml_g: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), None).score(DidScore::Experimental)
    }

    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Option<Box<dyn Classifier>>) -> Self {
        Self {
            ml_g,
            ml_m,
            score: DidScore::default(),
            in_sample_normalization: true,
            clipping_threshold: 1e-2,
            splitting: SampleSplitting::default(),
        }
    }

    did_builders!();
    dml_model_api!();
}

impl LinearScoreModel for Did {
    fn name(&self) -> String {
        match self.score {
            DidScore::Observational => "DID panel (observational)".into(),
            DidScore::Experimental => "DID panel (experimental)".into(),
        }
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        validate_common(
            data,
            self.score,
            self.ml_m.is_some(),
            self.clipping_threshold,
        )
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, .. } = *input;
        let treated: Vec<bool> = d.iter().map(|&v| v == 1.0).collect();
        let control: Vec<bool> = treated.iter().map(|b| !b).collect();
        let g0 = cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &control)?;
        let g1 = cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &treated)?;
        let d = d.to_owned();
        let n = d.len();
        let p = mean(&d);
        let resid_d0: Array1<f64> = &y - &g0;
        let mut predictions = vec![("ml_g0", g0.clone()), ("ml_g1", g1.clone())];

        let (weight_a, psi_b) = match self.score {
            DidScore::Observational => {
                let ml_m = self.ml_m.as_ref().expect("checked in validate");
                let t = self.clipping_threshold;
                let m = cross_fit_predict_proba(ml_m.as_ref(), x, d.view(), folds)?
                    .mapv(|q| q.clamp(t, 1.0 - t));
                let (weight_a, weight_resid) = if self.in_sample_normalization {
                    let weight_a = &d / mean(&d);
                    let propensity_weight = (1.0 - &d) * &m / &(1.0 - &m);
                    let weight_resid = &weight_a - &(&propensity_weight / mean(&propensity_weight));
                    (weight_a, weight_resid)
                } else {
                    (&d / p, (&d - &m) / &(p * &(1.0 - &m)))
                };
                predictions.push(("ml_m", m));
                let psi_b = &weight_resid * &resid_d0;
                (weight_a, psi_b)
            }
            DidScore::Experimental => {
                let (weight_g0, weight_g1, weight_resid) = if self.in_sample_normalization {
                    let share = &d / mean(&d);
                    let control_share = (1.0 - &d) / mean(&(1.0 - &d));
                    (&share - 1.0, 1.0 - &share, &share - &control_share)
                } else {
                    (&d / p - 1.0, 1.0 - &d / p, (&d - p) / (p * (1.0 - p)))
                };
                let psi_b = &weight_g0 * &g0 + &weight_g1 * &g1 + &weight_resid * &resid_d0;
                (Array1::ones(n), psi_b)
            }
        };
        Ok(ScoreEstimate {
            score: LinearScore::new(-weight_a, psi_b)?,
            predictions,
        })
    }
}

/// Difference-in-differences with **repeated cross-sections**: the data need
/// a 0/1 time indicator ([`DmlData::with_time_indicator`]). `ml_g` learns
/// `E[Y | D = d, T = t, X]` separately in each of the four group × period
/// cells, `ml_m` the propensity `P(D = 1 | X)` (observational score only).
pub struct DidCs {
    ml_g: Box<dyn Regressor>,
    ml_m: Option<Box<dyn Classifier>>,
    score: DidScore,
    in_sample_normalization: bool,
    clipping_threshold: f64,
    splitting: SampleSplitting,
}

impl DidCs {
    /// Observational design (doubly robust score).
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Some(Box::new(ml_m)))
    }

    /// Randomised design: only the outcome model is needed.
    pub fn experimental(ml_g: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), None).score(DidScore::Experimental)
    }

    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Option<Box<dyn Classifier>>) -> Self {
        Self {
            ml_g,
            ml_m,
            score: DidScore::default(),
            in_sample_normalization: true,
            clipping_threshold: 1e-2,
            splitting: SampleSplitting::default(),
        }
    }

    did_builders!();
    dml_model_api!();
}

/// Cell indicator `1{D = a, T = b}`.
fn cell(d: &Array1<f64>, t: ArrayView1<'_, f64>, a: f64, b: f64) -> Array1<f64> {
    d.iter()
        .zip(t)
        .map(|(&di, &ti)| f64::from(di == a && ti == b))
        .collect()
}

impl LinearScoreModel for DidCs {
    fn name(&self) -> String {
        match self.score {
            DidScore::Observational => "DID repeated cross-sections (observational)".into(),
            DidScore::Experimental => "DID repeated cross-sections (experimental)".into(),
        }
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        validate_common(
            data,
            self.score,
            self.ml_m.is_some(),
            self.clipping_threshold,
        )?;
        if data.time().is_none() {
            return Err(DmlError::InvalidArgument(
                "repeated cross-section DID needs a time indicator: use DmlData::with_time_indicator"
                    .into(),
            ));
        }
        Ok(())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, t, .. } = *input;
        let t = t.ok_or_else(|| DmlError::InvalidArgument("missing time indicator".into()))?;
        let d = d.to_owned();
        let (d1t1, d1t0) = (cell(&d, t, 1.0, 1.0), cell(&d, t, 1.0, 0.0));
        let (d0t1, d0t0) = (cell(&d, t, 0.0, 1.0), cell(&d, t, 0.0, 0.0));
        let fit_cell = |c: &Array1<f64>| -> Result<Array1<f64>> {
            let keep: Vec<bool> = c.iter().map(|&v| v == 1.0).collect();
            cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &keep)
        };
        let (g00, g01) = (fit_cell(&d0t0)?, fit_cell(&d0t1)?);
        let (g10, g11) = (fit_cell(&d1t0)?, fit_cell(&d1t1)?);
        let n = d.len();
        let p = mean(&d);
        let lambda = t.mean().expect("n > 0");

        let m = match self.score {
            DidScore::Observational => {
                let ml_m = self.ml_m.as_ref().expect("checked in validate");
                let c = self.clipping_threshold;
                Some(
                    cross_fit_predict_proba(ml_m.as_ref(), x, d.view(), folds)?
                        .mapv(|q| q.clamp(c, 1.0 - c)),
                )
            }
            DidScore::Experimental => None,
        };

        // Weights on the four residuals (y − ĝ_cell) and on the ĝ contrast.
        let (weight_a, w11, w10, w01, w00) = match (&m, self.in_sample_normalization) {
            (Some(m), true) => {
                let odds = m / &(1.0 - m);
                let u01 = &d0t1 * &odds;
                let u00 = &d0t0 * &odds;
                (
                    &d / mean(&d),
                    &d1t1 / mean(&d1t1),
                    -(&d1t0 / mean(&d1t0)),
                    -(&u01 / mean(&u01)),
                    &u00 / mean(&u00),
                )
            }
            (Some(m), false) => {
                let odds = m / &(1.0 - m);
                (
                    &d / p,
                    &d1t1 / (p * lambda),
                    -(&d1t0 / (p * (1.0 - lambda))),
                    -(&d0t1 / (p * lambda) * &odds),
                    &d0t0 / (p * (1.0 - lambda)) * &odds,
                )
            }
            (None, true) => (
                Array1::ones(n),
                &d1t1 / mean(&d1t1),
                -(&d1t0 / mean(&d1t0)),
                -(&d0t1 / mean(&d0t1)),
                &d0t0 / mean(&d0t0),
            ),
            (None, false) => (
                Array1::ones(n),
                &d1t1 / (p * lambda),
                -(&d1t0 / (p * (1.0 - lambda))),
                -(&d0t1 / ((1.0 - p) * lambda)),
                &d0t0 / ((1.0 - p) * (1.0 - lambda)),
            ),
        };
        let contrast = &g11 - &g10 - &g01 + &g00;
        let psi_b = &weight_a * &contrast
            + &w11 * &(&y - &g11)
            + &w10 * &(&y - &g10)
            + &w00 * &(&y - &g00)
            + &w01 * &(&y - &g01);

        let mut predictions = vec![
            ("ml_g_d0_t0", g00),
            ("ml_g_d0_t1", g01),
            ("ml_g_d1_t0", g10),
            ("ml_g_d1_t1", g11),
        ];
        if let Some(m) = m {
            predictions.push(("ml_m", m));
        }
        Ok(ScoreEstimate {
            score: LinearScore::new(-weight_a, psi_b)?,
            predictions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use ndarray::array;

    #[test]
    fn validation_rules() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let data = DmlData::new(array![1.0, 2.0, 3.0, 4.0], array![0.0, 1.0, 0.0, 1.0], x).unwrap();
        // Observational score needs ml_m.
        let no_m = Did::from_boxed(Box::new(Ols::new()), None);
        assert!(no_m.validate(&data).is_err());
        assert!(Did::experimental(Ols::new()).validate(&data).is_ok());
        assert!(Did::new(Ols::new(), LogisticRegression::default())
            .validate(&data)
            .is_ok());
        // Repeated cross-sections need a time indicator.
        assert!(DidCs::experimental(Ols::new()).validate(&data).is_err());
        let with_t = data
            .with_time_indicator(array![0.0, 0.0, 1.0, 1.0])
            .unwrap();
        assert!(DidCs::experimental(Ols::new()).validate(&with_t).is_ok());
    }

    #[test]
    fn cell_indicators_partition_rows() {
        let d = array![0.0, 0.0, 1.0, 1.0];
        let t = array![0.0, 1.0, 0.0, 1.0];
        let total = cell(&d, t.view(), 0.0, 0.0)
            + cell(&d, t.view(), 0.0, 1.0)
            + cell(&d, t.view(), 1.0, 0.0)
            + cell(&d, t.view(), 1.0, 1.0);
        assert_eq!(total, Array1::ones(4));
        assert_eq!(cell(&d, t.view(), 1.0, 0.0), array![0.0, 0.0, 1.0, 0.0]);
    }
}

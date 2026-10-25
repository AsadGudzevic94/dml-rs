//! Sample selection model (SSM): the average treatment effect of a binary
//! treatment when the outcome is observed only for a selected subsample
//! (attrition, non-response), as DoubleML's `DoubleMLSSM` (Bia, Huber and
//! Lafférs, 2024).
//!
//! ```text
//! Y = g(D, X) + U,   observed only where S = 1
//! m(X) = P(D = 1 | X),   π(D, X) = P(S = 1 | D, X)
//! ```
//!
//! - [`SsmScore::MissingAtRandom`]: selection depends on the treatment and the
//!   covariates only. `π` is learned from `(X, D)`, `m` from `X`, and the
//!   outcome models on the selected treated / controls.
//! - [`SsmScore::Nonignorable`]: selection may depend on unobservables, with
//!   an instrument `Z` for selection. Each training fold is split in halves
//!   (stratified by `D + 2S`): `π` is learned from `(X, D, Z)` on the first
//!   half, and `m` and the outcome models on the second, with the predicted
//!   selection probability as an extra covariate.

use ndarray::{concatenate, Array1, ArrayView1, ArrayView2, Axis};

use super::quantile::half_split;
use super::{dml_model_api, LinearScoreModel, NuisanceTarget, ScoreEstimate, ScoreInput};
use crate::crossfit::{cross_fit_predict_proba, cross_fit_predict_subset, Fold, SampleSplitting};
use crate::data::{gather, is_binary, DmlData};
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};
use crate::propensity::{adjust_propensity, Calibration};
use crate::score::LinearScore;

/// Selection mechanism of [`Ssm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SsmScore {
    /// Selection depends on `D` and `X` only.
    #[default]
    MissingAtRandom,
    /// Selection may depend on unobservables; needs an instrument for selection.
    Nonignorable,
}

/// Average treatment effect of a binary treatment when the outcome is
/// observed only where the selection indicator `S` is 1 (attrition,
/// non-response), as DoubleML's `DoubleMLSSM`. Requires a selection indicator
/// ([`DmlData::with_selection_indicator`]) and, for
/// [`SsmScore::Nonignorable`], an instrument for selection
/// ([`DmlData::with_instrument`]).
///
/// The score is doubly robust in the outcome models of the selected treated
/// and controls, the propensity `P(D = 1 | X)` and the selection probability
/// `P(S = 1 | D, X)`. Under nonignorable selection each training fold is
/// split in halves: the selection probability is learned on the first, and
/// the other nuisances on the second with the predicted selection
/// probability as a covariate.
pub struct Ssm {
    ml_g: Box<dyn Regressor>,
    ml_pi: Box<dyn Classifier>,
    ml_m: Box<dyn Classifier>,
    score: SsmScore,
    clipping_threshold: f64,
    calibration: Calibration,
    normalize_ipw: bool,
    splitting: SampleSplitting,
}

impl Ssm {
    /// `ml_g` learns `E[Y | D, X]` on the selected, `ml_pi` the selection
    /// probability `P(S = 1 | D, X)`, `ml_m` the propensity `P(D = 1 | X)`.
    pub fn new(
        ml_g: impl Regressor + 'static,
        ml_pi: impl Classifier + 'static,
        ml_m: impl Classifier + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_pi), Box::new(ml_m))
    }

    pub fn from_boxed(
        ml_g: Box<dyn Regressor>,
        ml_pi: Box<dyn Classifier>,
        ml_m: Box<dyn Classifier>,
    ) -> Self {
        Self {
            ml_g,
            ml_pi,
            ml_m,
            score: SsmScore::default(),
            clipping_threshold: 1e-2,
            calibration: Calibration::None,
            normalize_ipw: false,
            splitting: SampleSplitting::default(),
        }
    }

    pub fn score(mut self, score: SsmScore) -> Self {
        self.score = score;
        self
    }

    /// Treatment propensities are clipped to `[t, 1 − t]` (default `t = 0.01`).
    /// Selection probabilities are not clipped, as in DoubleML.
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }
    /// Calibrate the propensity scores before clipping (default: none), as
    /// DoubleML's `ps_processor_config`.
    pub fn calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Normalise the inverse-probability weights within treated and controls.
    pub fn normalize_ipw(mut self, normalize: bool) -> Self {
        self.normalize_ipw = normalize;
        self
    }

    dml_model_api!();

    /// Nonignorable selection: nested cross-fitting within each fold.
    /// Returns `(g0, g1, π, m)` out of fold.
    fn nonignorable_nuisances(
        &self,
        input: &ScoreInput<'_>,
        s: ArrayView1<'_, f64>,
        z: ArrayView1<'_, f64>,
        folds: &[Fold],
    ) -> Result<[Array1<f64>; 4]> {
        let ScoreInput { y, d, x, .. } = *input;
        let n = y.len();
        let strata: Vec<f64> = (0..n).map(|i| d[i] + 2.0 * s[i]).collect();
        let dxz = concatenate![Axis(1), x, d.insert_axis(Axis(1)), z.insert_axis(Axis(1))];
        let fit_fold = |fold: &Fold| -> Result<[Array1<f64>; 4]> {
            let (train1, train2) = half_split(&fold.train, &strata)?;
            // Selection probability from the first half, predicted for every
            // observation and added to the covariates.
            let pi_all = fit_proba(self.ml_pi.as_ref(), dxz.view(), &train1, s, dxz.view())?;
            let xpi = concatenate![Axis(1), x, pi_all.view().insert_axis(Axis(1))];
            let xpi_test = xpi.select(Axis(0), &fold.test);
            let m = fit_proba(self.ml_m.as_ref(), xpi.view(), &train2, d, xpi_test.view())?;
            // Outcome models on the selected of the second half, in index order
            // (DoubleML's `np.intersect1d`).
            let mut sorted2 = train2.clone();
            sorted2.sort_unstable();
            let outcome = |level: f64| -> Result<Array1<f64>> {
                let rows: Vec<usize> = sorted2
                    .iter()
                    .copied()
                    .filter(|&i| d[i] == level && s[i] == 1.0)
                    .collect();
                if rows.is_empty() {
                    return Err(DmlError::InvalidArgument(
                        "a training fold has no selected observations in a treatment arm".into(),
                    ));
                }
                let mut model = self.ml_g.fresh();
                model.fit(xpi.select(Axis(0), &rows).view(), gather(y, &rows).view())?;
                model.predict(xpi_test.view())
            };
            let pi_test = gather(pi_all.view(), &fold.test);
            Ok([outcome(0.0)?, outcome(1.0)?, pi_test, m])
        };
        #[cfg(feature = "parallel")]
        let per_fold: Vec<[Array1<f64>; 4]> = {
            use rayon::prelude::*;
            folds.par_iter().map(fit_fold).collect::<Result<_>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let per_fold: Vec<[Array1<f64>; 4]> = folds.iter().map(fit_fold).collect::<Result<_>>()?;

        let mut out: [Array1<f64>; 4] = std::array::from_fn(|_| Array1::from_elem(n, f64::NAN));
        for (fold, preds) in folds.iter().zip(per_fold) {
            for (dst, src) in out.iter_mut().zip(preds) {
                for (k, &i) in fold.test.iter().enumerate() {
                    dst[i] = src[k];
                }
            }
        }
        Ok(out)
    }
}

/// Fit a fresh copy of `learner` on `rows` (in that order); predict `x_pred`.
fn fit_proba(
    learner: &dyn Classifier,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
    labels: ArrayView1<'_, f64>,
    x_pred: ArrayView2<'_, f64>,
) -> Result<Array1<f64>> {
    let mut model = learner.fresh();
    model.fit(x.select(Axis(0), rows).view(), gather(labels, rows).view())?;
    model.predict_proba(x_pred)
}

impl LinearScoreModel for Ssm {
    fn name(&self) -> String {
        match self.score {
            SsmScore::MissingAtRandom => "SSM (missing at random)".into(),
            SsmScore::Nonignorable => "SSM (nonignorable)".into(),
        }
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        if !(0..data.n_treat()).all(|j| data.is_binary_treatment(j)) {
            return Err(DmlError::InvalidArgument(
                "SSM needs binary (0/1) treatments".into(),
            ));
        }
        if data.selection().is_none() {
            return Err(DmlError::InvalidArgument(
                "SSM needs a selection indicator (DmlData::with_selection_indicator)".into(),
            ));
        }
        if data.n_instruments() > 1 {
            return Err(DmlError::InvalidArgument(
                "SSM needs exactly one instrument".into(),
            ));
        }
        if self.score == SsmScore::Nonignorable && data.z().is_none() {
            return Err(DmlError::InvalidArgument(
                "the nonignorable score needs an instrument for selection (DmlData::with_instrument)"
                    .into(),
            ));
        }
        if !(self.clipping_threshold > 0.0 && self.clipping_threshold < 0.5) {
            return Err(DmlError::InvalidArgument(
                "clipping_threshold must be in (0, 0.5)".into(),
            ));
        }
        Ok(())
    }

    /// Stratify by treatment and selection (`D + 2S`), as DoubleML.
    fn strata(&self, data: &DmlData) -> Option<Vec<usize>> {
        let (d, s) = (data.treatment(0), data.selection()?);
        Some(
            d.iter()
                .zip(s)
                .map(|(&d, &s)| (d + 2.0 * s) as usize)
                .collect(),
        )
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, z, s, .. } = *input;
        let s =
            s.ok_or_else(|| DmlError::InvalidArgument("SSM needs a selection indicator".into()))?;
        if !is_binary(d) {
            return Err(DmlError::InvalidArgument(
                "SSM needs a binary treatment".into(),
            ));
        }
        let n = y.len();
        let [g0, g1, pi, m] = match self.score {
            SsmScore::MissingAtRandom => {
                let dx = concatenate![Axis(1), x, d.insert_axis(Axis(1))];
                let pi = cross_fit_predict_proba(self.ml_pi.as_ref(), dx.view(), s, folds)?;
                let m = cross_fit_predict_proba(self.ml_m.as_ref(), x, d, folds)?;
                let selected = |level: f64| -> Vec<bool> {
                    (0..n).map(|i| d[i] == level && s[i] == 1.0).collect()
                };
                let g1 = cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &selected(1.0))?;
                let g0 = cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &selected(0.0))?;
                [g0, g1, pi, m]
            }
            SsmScore::Nonignorable => {
                let z = z.ok_or_else(|| {
                    DmlError::InvalidArgument("the nonignorable score needs an instrument".into())
                })?;
                self.nonignorable_nuisances(input, s, z, folds)?
            }
        };
        let t = self.clipping_threshold;
        let m = adjust_propensity(m, d, folds, t, self.calibration)?;

        // ψ_b = ψ_b1 − ψ_b0, each an IPW-corrected outcome prediction; written
        // in DoubleML's operation order (and its sequential sums) so results
        // agree to the last bit.
        let treat: Array1<f64> = d.mapv(|v| f64::from(v == 1.0));
        let control: Array1<f64> = d.mapv(|v| f64::from(v == 0.0));
        let (w1, w0) = if self.normalize_ipw {
            let sum = |a: Array1<f64>| a.iter().fold(0.0, |acc, &v| acc + v);
            let w1 = treat.sum() / sum(&treat * &s / &(&pi * &m));
            let w0 = control.sum() / sum(&control * &s / &(&pi * &(1.0 - &m)));
            (Some(w1), Some(w0))
        } else {
            (None, None)
        };
        let psi_b: Array1<f64> = (0..n)
            .map(|i| {
                let c1 = treat[i] * s[i] * (y[i] - g1[i]) / (m[i] * pi[i]);
                let c0 = control[i] * s[i] * (y[i] - g0[i]) / ((1.0 - m[i]) * pi[i]);
                let b1 = w1.map_or(c1, |w| w * c1) + g1[i];
                let b0 = w0.map_or(c0, |w| w * c0) + g0[i];
                b1 - b0
            })
            .collect();
        let psi_a = Array1::from_elem(n, -1.0);
        Ok(ScoreEstimate::new(
            LinearScore::new(psi_a, psi_b)?,
            vec![("ml_g_d0", g0), ("ml_g_d1", g1), ("ml_pi", pi), ("ml_m", m)],
        )
        // As DoubleML, the outcome models are evaluated on every observation.
        .with_targets(vec![
            NuisanceTarget::regression("ml_g_d0", y.to_owned()),
            NuisanceTarget::regression("ml_g_d1", y.to_owned()),
            NuisanceTarget::classification("ml_pi", s.to_owned()),
            NuisanceTarget::classification("ml_m", d.to_owned()),
        ]))
    }
}

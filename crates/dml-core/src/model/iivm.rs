//! Interactive IV model (IIVM) for a binary treatment `D` and a binary
//! instrument `Z`, identifying the local average treatment effect (LATE) for
//! compliers:
//!
//! ```text
//! Y = ℓ(D, X) + ζ,       E[ζ | Z, X] = 0
//! D = r(Z, X) + ν,       E[ν | Z, X] = 0
//! Z = m(X) + V,          E[V | X] = 0
//! θ = (E[g(1,X) − g(0,X)]) / (E[r(1,X) − r(0,X)])
//! ```
//!
//! where `g(z, X) = E[Y | Z = z, X]` and `r(z, X) = P(D = 1 | Z = z, X)`.

use ndarray::Array1;

use super::irm::{ipw_residual, normalize_propensity};
use super::{dml_model_api, LinearScoreModel, ScoreEstimate, ScoreInput};
use crate::crossfit::{
    cross_fit_predict_proba, cross_fit_predict_proba_subset, cross_fit_predict_subset, Fold,
    SampleSplitting,
};
use crate::data::{is_binary, DmlData};
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};
use crate::score::LinearScore;

/// IIVM estimator of the LATE: `ml_g` learns `E[Y | Z, X]` (fitted separately
/// on the `Z = 0` and `Z = 1` rows), `ml_m` the instrument propensity
/// `P(Z = 1 | X)`, and `ml_r` the treatment probability `P(D = 1 | Z, X)`
/// (again per instrument arm).
pub struct Iivm {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    ml_r: Box<dyn Classifier>,
    always_takers: bool,
    never_takers: bool,
    clipping_threshold: f64,
    normalize_ipw: bool,
    splitting: SampleSplitting,
}

impl Iivm {
    pub fn new(
        ml_g: impl Regressor + 'static,
        ml_m: impl Classifier + 'static,
        ml_r: impl Classifier + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m), Box::new(ml_r))
    }

    pub fn from_boxed(
        ml_g: Box<dyn Regressor>,
        ml_m: Box<dyn Classifier>,
        ml_r: Box<dyn Classifier>,
    ) -> Self {
        Self {
            ml_g,
            ml_m,
            ml_r,
            always_takers: true,
            never_takers: true,
            clipping_threshold: 1e-2,
            normalize_ipw: false,
            splitting: SampleSplitting::default(),
        }
    }

    /// Whether always-takers (`D = 1` despite `Z = 0`) and never-takers
    /// (`D = 0` despite `Z = 1`) can exist. Without always-takers
    /// `r(0, X) = 0` is imposed instead of estimated; without never-takers
    /// `r(1, X) = 1`. With one-sided non-compliance, switch the impossible
    /// group off: its classifier would otherwise see only one class.
    pub fn subgroups(mut self, always_takers: bool, never_takers: bool) -> Self {
        self.always_takers = always_takers;
        self.never_takers = never_takers;
        self
    }

    /// Instrument propensities are clipped to `[t, 1 − t]` (default 0.01).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }

    /// Normalise the inverse-probability weights within each instrument arm.
    ///
    /// Note: DoubleML 0.11 normalises with the treatment `D` here; this
    /// implementation uses the instrument `Z`, whose propensity the weights
    /// invert. Results therefore differ from DoubleML when this is enabled.
    /// Reported upstream as
    /// [DoubleML#411](https://github.com/DoubleML/doubleml-for-py/issues/411).
    pub fn normalize_ipw(mut self, normalize: bool) -> Self {
        self.normalize_ipw = normalize;
        self
    }

    dml_model_api!();
}

impl LinearScoreModel for Iivm {
    fn name(&self) -> String {
        "IIVM (LATE)".into()
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return Err(DmlError::InvalidArgument(
                "IIVM requires exactly one binary (0/1) treatment".into(),
            ));
        }
        match data.z() {
            Some(z) if is_binary(z) => {}
            _ => {
                return Err(DmlError::InvalidArgument(
                    "IIVM requires a binary (0/1) instrument: use DmlData::with_instrument".into(),
                ))
            }
        }
        if !(self.clipping_threshold > 0.0 && self.clipping_threshold < 0.5) {
            return Err(DmlError::InvalidArgument(
                "clipping_threshold must be in (0, 0.5)".into(),
            ));
        }
        Ok(())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, z, .. } = *input;
        let z = z.ok_or_else(|| DmlError::InvalidArgument("IIVM requires an instrument".into()))?;
        let z1: Vec<bool> = z.iter().map(|&v| v == 1.0).collect();
        let z0: Vec<bool> = z1.iter().map(|b| !b).collect();
        let n = y.len();

        let g0 = cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &z0)?;
        let g1 = cross_fit_predict_subset(self.ml_g.as_ref(), x, y, folds, &z1)?;
        let t = self.clipping_threshold;
        let m =
            cross_fit_predict_proba(self.ml_m.as_ref(), x, z, folds)?.mapv(|p| p.clamp(t, 1.0 - t));
        let r0 = if self.always_takers {
            cross_fit_predict_proba_subset(self.ml_r.as_ref(), x, d, folds, &z0)?
        } else {
            Array1::zeros(n)
        };
        let r1 = if self.never_takers {
            cross_fit_predict_proba_subset(self.ml_r.as_ref(), x, d, folds, &z1)?
        } else {
            Array1::ones(n)
        };

        let z = z.to_owned();
        let m_adj = if self.normalize_ipw {
            normalize_propensity(&m, &z)
        } else {
            m.clone()
        };
        // Reduced form (effect of Z on Y) over first stage (effect of Z on D).
        let psi_b = &g1 - &g0 + &ipw_residual(y, &z, &g0, &g1, &m_adj);
        let first_stage = &r1 - &r0 + &ipw_residual(d, &z, &r0, &r1, &m_adj);
        Ok(ScoreEstimate::new(
            LinearScore::new(-first_stage, psi_b)?,
            vec![
                ("ml_g0", g0),
                ("ml_g1", g1),
                ("ml_m", m),
                ("ml_r0", r0),
                ("ml_r1", r1),
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use ndarray::array;

    #[test]
    fn requires_binary_treatment_and_instrument() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let base = DmlData::new(
            array![1.0, 2.0, 3.0, 4.0],
            array![0.0, 1.0, 0.0, 1.0],
            x.clone(),
        )
        .unwrap();
        let iivm = Iivm::new(
            Ols::new(),
            LogisticRegression::default(),
            LogisticRegression::default(),
        );
        assert!(iivm.validate(&base).is_err()); // no instrument
        let continuous_z = base
            .clone()
            .with_instrument(array![0.5, 1.0, 0.0, 1.0])
            .unwrap();
        assert!(iivm.validate(&continuous_z).is_err());
        let ok = base.with_instrument(array![0.0, 1.0, 0.0, 1.0]).unwrap();
        assert!(iivm.validate(&ok).is_ok());
    }
}

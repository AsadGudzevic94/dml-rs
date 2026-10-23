//! Partially linear IV regression (PLIV) with one instrument `Z`:
//!
//! ```text
//! Y = θ·D + g(X) + ε,   E[ε | Z, X] = 0
//! Z = m(X) + v,         E[v | X] = 0
//! ```

use ndarray::Array1;

use super::{dml_model_api, LinearScoreModel, ScoreEstimate, ScoreInput};
use crate::crossfit::{cross_fit_predict, Fold, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::learner::Regressor;
use crate::score::LinearScore;

/// PLIV estimator with the partialling-out score
/// `ψ = (Y − ℓ(X) − θ(D − r(X)))·(Z − m(X))`.
///
/// `ml_l` learns `E[Y | X]`, `ml_m` learns `E[Z | X]`, `ml_r` learns `E[D | X]`.
pub struct Pliv {
    ml_l: Box<dyn Regressor>,
    ml_m: Box<dyn Regressor>,
    ml_r: Box<dyn Regressor>,
    splitting: SampleSplitting,
}

impl Pliv {
    pub fn new(
        ml_l: impl Regressor + 'static,
        ml_m: impl Regressor + 'static,
        ml_r: impl Regressor + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(ml_l), Box::new(ml_m), Box::new(ml_r))
    }

    pub fn from_boxed(
        ml_l: Box<dyn Regressor>,
        ml_m: Box<dyn Regressor>,
        ml_r: Box<dyn Regressor>,
    ) -> Self {
        Self {
            ml_l,
            ml_m,
            ml_r,
            splitting: SampleSplitting::default(),
        }
    }

    dml_model_api!();
}

impl LinearScoreModel for Pliv {
    fn name(&self) -> String {
        "PLIV (partialling out)".into()
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        if data.z().is_none() {
            return Err(DmlError::InvalidArgument(
                "PLIV requires an instrument: use DmlData::with_instrument".into(),
            ));
        }
        Ok(())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, z, .. } = *input;
        let z = z.ok_or_else(|| DmlError::InvalidArgument("PLIV requires an instrument".into()))?;
        let l_hat = cross_fit_predict(self.ml_l.as_ref(), x, y, folds)?;
        let m_hat = cross_fit_predict(self.ml_m.as_ref(), x, z, folds)?;
        let r_hat = cross_fit_predict(self.ml_r.as_ref(), x, d, folds)?;
        let u: Array1<f64> = &y - &l_hat;
        let v: Array1<f64> = &z - &m_hat;
        let w: Array1<f64> = &d - &r_hat;
        Ok(ScoreEstimate::new(
            LinearScore::new(-(&w * &v), &v * &u)?,
            vec![("ml_l", l_hat), ("ml_m", m_hat), ("ml_r", r_hat)],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Ols;
    use ndarray::array;

    #[test]
    fn requires_instrument() {
        let data = DmlData::new(array![1.0, 2.0], array![0.0, 1.0], array![[1.0], [2.0]]).unwrap();
        assert!(Pliv::new(Ols::new(), Ols::new(), Ols::new())
            .fit(&data)
            .is_err());
    }
}

//! Partially linear IV regression (PLIV) with one or several instruments `Z`:
//!
//! ```text
//! Y = θ·D + g(X) + ε,   E[ε | Z, X] = 0
//! ```
//!
//! As DoubleML's `DoubleMLPLIV` (Chernozhukov et al., 2018), in three
//! variants ([`PlivVariant`]): partialling `X` out of `Y`, `D` and `Z`
//! (default), partialling `Z` out of `D` (`E[D | X, Z]` as optimal
//! instrument), or both.

use ndarray::{concatenate, Array1, Array2, Axis};

use super::{dml_model_api, LinearScoreModel, Name, NuisanceTarget, ScoreEstimate, ScoreInput};
use crate::compat::np_mean;
use crate::crossfit::{cross_fit_predict, Fold, SampleSplitting};
use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::learner::{Ols, Regressor};
use crate::score::LinearScore;

/// Score of [`Pliv`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlivScore {
    /// `ψ = (Y − ℓ(X) − θ(D − r(X)))·(Z − m(X))`.
    #[default]
    PartiallingOut,
    /// `ψ = (Y − θD − g(X))·(Z − m(X))`; one instrument, [`PlivVariant::PartialX`] only.
    IvType,
}

/// What [`Pliv`] partials out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlivVariant {
    /// Partial `X` out of `Y`, `D` and every instrument. With several
    /// instruments the treatment residual is projected on the instrument
    /// residuals (two-stage least squares). Learners `ml_l` (`E[Y | X]`),
    /// `ml_m` (`E[Z | X]`, one fit per instrument), `ml_r` (`E[D | X]`).
    #[default]
    PartialX,
    /// Use `E[D | X, Z]` (`ml_r`) as instrument; no other learner.
    PartialZ,
    /// Partial out both: `ml_m` learns `E[D | X, Z]`, `ml_r` the projection of
    /// its in-sample predictions on `X`, `ml_l` learns `E[Y | X]`.
    PartialXZ,
}

/// Partially linear IV regression. See [`PlivVariant`] and [`PlivScore`].
pub struct Pliv {
    ml_l: Option<Box<dyn Regressor>>,
    ml_m: Option<Box<dyn Regressor>>,
    ml_r: Box<dyn Regressor>,
    ml_g: Option<Box<dyn Regressor>>,
    score: PlivScore,
    variant: PlivVariant,
    splitting: SampleSplitting,
}

impl Pliv {
    /// Partial `X` out ([`PlivVariant::PartialX`]): `ml_l` learns `E[Y | X]`,
    /// `ml_m` `E[Z | X]`, `ml_r` `E[D | X]`.
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
            ml_l: Some(ml_l),
            ml_m: Some(ml_m),
            ml_r,
            ml_g: None,
            score: PlivScore::default(),
            variant: PlivVariant::PartialX,
            splitting: SampleSplitting::default(),
        }
    }

    /// Partial `Z` out ([`PlivVariant::PartialZ`]): `ml_r` learns `E[D | X, Z]`.
    pub fn partial_z(ml_r: impl Regressor + 'static) -> Self {
        Self::partial_z_boxed(Box::new(ml_r))
    }

    pub fn partial_z_boxed(ml_r: Box<dyn Regressor>) -> Self {
        Self {
            ml_l: None,
            ml_m: None,
            ml_r,
            ml_g: None,
            score: PlivScore::default(),
            variant: PlivVariant::PartialZ,
            splitting: SampleSplitting::default(),
        }
    }

    /// Partial out `X` and `Z` ([`PlivVariant::PartialXZ`]): `ml_l` learns
    /// `E[Y | X]`, `ml_m` `E[D | X, Z]`, `ml_r` the projection of `ml_m`'s
    /// in-sample predictions on `X`.
    pub fn partial_xz(
        ml_l: impl Regressor + 'static,
        ml_m: impl Regressor + 'static,
        ml_r: impl Regressor + 'static,
    ) -> Self {
        Self::from_boxed(Box::new(ml_l), Box::new(ml_m), Box::new(ml_r))
            .variant(PlivVariant::PartialXZ)
    }

    /// Choose the variant (learners as for [`Pliv::new`]; [`PlivVariant::PartialZ`]
    /// uses only `ml_r`, learning `E[D | X, Z]`).
    pub fn variant(mut self, variant: PlivVariant) -> Self {
        self.variant = variant;
        self
    }

    pub fn score(mut self, score: PlivScore) -> Self {
        self.score = score;
        self
    }

    /// Learner for `E[Y − θD | X]` with the IV-type score (default: a copy of `ml_l`).
    pub fn ml_g(self, ml_g: impl Regressor + 'static) -> Self {
        self.ml_g_boxed(Box::new(ml_g))
    }

    pub fn ml_g_boxed(mut self, ml_g: Box<dyn Regressor>) -> Self {
        self.ml_g = Some(ml_g);
        self
    }

    dml_model_api!();

    fn learner(&self, l: &Option<Box<dyn Regressor>>, name: &str) -> Result<Box<dyn Regressor>> {
        l.as_ref().map(|l| l.fresh()).ok_or_else(|| {
            DmlError::InvalidArgument(format!("this PLIV variant needs a learner {name}"))
        })
    }
}

impl LinearScoreModel for Pliv {
    fn name(&self) -> String {
        let variant = match self.variant {
            PlivVariant::PartialX => "",
            PlivVariant::PartialZ => ", partial Z",
            PlivVariant::PartialXZ => ", partial XZ",
        };
        let score = match self.score {
            PlivScore::PartiallingOut => "partialling out",
            PlivScore::IvType => "IV-type",
        };
        format!("PLIV ({score}{variant})")
    }

    fn validate(&self, data: &DmlData) -> Result<()> {
        let k = data.n_instruments();
        if k == 0 {
            return Err(DmlError::InvalidArgument(
                "PLIV requires an instrument: use DmlData::with_instrument".into(),
            ));
        }
        if self.score == PlivScore::IvType && (k > 1 || self.variant != PlivVariant::PartialX) {
            return Err(DmlError::InvalidArgument(
                "the IV-type score needs one instrument and the partial X variant".into(),
            ));
        }
        Ok(())
    }

    fn estimate_score(&self, input: &ScoreInput<'_>, folds: &[Fold]) -> Result<ScoreEstimate> {
        let ScoreInput { y, d, x, .. } = *input;
        let z = input
            .instruments
            .ok_or_else(|| DmlError::InvalidArgument("PLIV requires an instrument".into()))?;
        match self.variant {
            PlivVariant::PartialX => {
                let ml_l = self.learner(&self.ml_l, "ml_l")?;
                let ml_m = self.learner(&self.ml_m, "ml_m")?;
                let l_hat = input
                    .predictions_or("ml_l", || cross_fit_predict(ml_l.as_ref(), x, y, folds))?;
                let r_hat = input.predictions_or("ml_r", || {
                    cross_fit_predict(self.ml_r.as_ref(), x, d, folds)
                })?;
                let u: Array1<f64> = &y - &l_hat;
                let w: Array1<f64> = &d - &r_hat;
                let mut predictions: Vec<(Name, Array1<f64>)> = vec![("ml_l".into(), l_hat)];
                let mut targets = vec![NuisanceTarget::regression("ml_l", y.to_owned())];
                if z.ncols() == 1 {
                    let z = z.column(0);
                    let m_hat = input
                        .predictions_or("ml_m", || cross_fit_predict(ml_m.as_ref(), x, z, folds))?;
                    let v: Array1<f64> = &z - &m_hat;
                    predictions.push(("ml_m".into(), m_hat));
                    targets.push(NuisanceTarget::regression("ml_m", z.to_owned()));
                    let score = match self.score {
                        PlivScore::PartiallingOut => LinearScore::new(-(&w * &v), &v * &u)?,
                        PlivScore::IvType => {
                            // Initial θ from the partialling-out score.
                            let psi_a: Vec<f64> = w.iter().zip(&v).map(|(w, v)| -(w * v)).collect();
                            let psi_b: Vec<f64> = v.iter().zip(&u).map(|(v, u)| v * u).collect();
                            let theta_init = -np_mean(&psi_b) / np_mean(&psi_a);
                            let target: Array1<f64> = &y - &(theta_init * &d);
                            let ml_g = match &self.ml_g {
                                Some(g) => g.fresh(),
                                None => ml_l.fresh(),
                            };
                            let g_hat = input.predictions_or("ml_g", || {
                                cross_fit_predict(ml_g.as_ref(), x, target.view(), folds)
                            })?;
                            let score = LinearScore::new(-(&v * &d), &v * &(&y - &g_hat))?;
                            predictions.push(("ml_g".into(), g_hat));
                            targets.push(NuisanceTarget::regression("ml_g", target));
                            score
                        }
                    };
                    predictions.push(("ml_r".into(), r_hat.clone()));
                    targets.push(NuisanceTarget::regression("ml_r", d.to_owned()));
                    return Ok(ScoreEstimate::new(score, predictions).with_targets(targets));
                }
                // Several instruments: residualise each, then project the
                // treatment residual on them (OLS with intercept).
                let mut v = Array2::zeros(z.dim());
                for (j, col) in z.columns().into_iter().enumerate() {
                    let name = format!("ml_m_z{}", j + 1);
                    let m_hat = input.predictions_or(&name, || {
                        cross_fit_predict(ml_m.fresh().as_ref(), x, col, folds)
                    })?;
                    v.column_mut(j).assign(&(&col - &m_hat));
                    targets.push(NuisanceTarget::regression(name.clone(), col.to_owned()));
                    predictions.push((name.into(), m_hat));
                }
                let mut proj = Ols::new();
                proj.fit(v.view(), w.view())?;
                let r_tilde = proj.predict(v.view())?;
                predictions.push(("ml_r".into(), r_hat.clone()));
                targets.push(NuisanceTarget::regression("ml_r", d.to_owned()));
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&w * &r_tilde), &r_tilde * &u)?,
                    predictions,
                )
                .with_targets(targets))
            }
            PlivVariant::PartialZ => {
                let xz = concatenate![Axis(1), x, z];
                let r_hat = cross_fit_predict(self.ml_r.as_ref(), xz.view(), d, folds)?;
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&r_hat * &d), &r_hat * &y)?,
                    vec![("ml_r", r_hat)],
                )
                .with_targets(vec![NuisanceTarget::regression("ml_r", d.to_owned())]))
            }
            PlivVariant::PartialXZ => {
                let ml_l = self.learner(&self.ml_l, "ml_l")?;
                let ml_m = self.learner(&self.ml_m, "ml_m")?;
                let xz = concatenate![Axis(1), x, z];
                let l_hat = cross_fit_predict(ml_l.as_ref(), x, y, folds)?;
                // ml_m on (X, Z), predicting both its training rows (the
                // targets of ml_r) and the test rows; ml_r on X.
                let n = y.len();
                let fold_fit = |fold: &Fold| -> Result<(Array1<f64>, Array1<f64>)> {
                    let mut m = ml_m.fresh();
                    let x_train = xz.select(Axis(0), &fold.train);
                    m.fit(x_train.view(), gather(d, &fold.train).view())?;
                    let train_pred = m.predict(x_train.view())?;
                    let test_pred = m.predict(xz.select(Axis(0), &fold.test).view())?;
                    let mut r = self.ml_r.fresh();
                    r.fit(x.select(Axis(0), &fold.train).view(), train_pred.view())?;
                    let tilde = r.predict(x.select(Axis(0), &fold.test).view())?;
                    Ok((test_pred, tilde))
                };
                let per_fold: Vec<(Array1<f64>, Array1<f64>)> =
                    folds.iter().map(fold_fit).collect::<Result<_>>()?;
                let mut m_hat = Array1::from_elem(n, f64::NAN);
                let mut m_tilde = Array1::from_elem(n, f64::NAN);
                for (fold, (pred, tilde)) in folds.iter().zip(&per_fold) {
                    for (k, &i) in fold.test.iter().enumerate() {
                        m_hat[i] = pred[k];
                        m_tilde[i] = tilde[k];
                    }
                }
                let u: Array1<f64> = &y - &l_hat;
                let w: Array1<f64> = &d - &m_tilde;
                let instrument: Array1<f64> = &m_hat - &m_tilde;
                Ok(ScoreEstimate::new(
                    LinearScore::new(-(&w * &instrument), &instrument * &u)?,
                    vec![("ml_l", l_hat), ("ml_m", m_hat), ("ml_r", m_tilde)],
                )
                .with_targets(vec![
                    NuisanceTarget::regression("ml_l", y.to_owned()),
                    NuisanceTarget::regression("ml_m", d.to_owned()),
                ]))
            }
        }
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

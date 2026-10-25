//! Shared machinery for models whose score is non-linear in θ (quantile
//! models, the logistic partially linear model): one repetition's estimate
//! and score, assembled into a [`DmlFit`] with DoubleML's variance estimate.

use ndarray::{Array1, Array2, Array3};

use super::NuisanceTarget;
use crate::compat::np_mean;
use crate::data::DmlData;
use crate::fit::{aggregate, DmlFit, NuisanceLoss};

/// Result of one repetition of a model with a non-linear score.
pub(super) struct RepFit {
    /// Mean preliminary estimate, the next repetition's start value.
    pub next_start: f64,
    pub theta: f64,
    pub psi: Array1<f64>,
    /// Mean score derivative `J`.
    pub jacobian: f64,
    pub predictions: Vec<(&'static str, Array1<f64>)>,
    pub targets: Vec<NuisanceTarget>,
}

/// Assemble a [`DmlFit`] from the repetitions of a non-linear model. The score
/// is stored linearised at the estimate (`ψ_a = J`, `ψ_b = ψ − J·θ̂`), so the
/// bootstrap and linear combinations work as for linear scores.
pub(super) fn assemble(name: String, data: &DmlData, n_folds: usize, reps: Vec<RepFit>) -> DmlFit {
    let n = data.n_obs();
    let n_rep = reps.len();
    let mut all_coef = Array2::zeros((1, n_rep));
    let mut all_se = Array2::zeros((1, n_rep));
    let mut psi = Array3::zeros((n, 1, n_rep));
    let mut psi_a = Array3::zeros((n, 1, n_rep));
    let mut psi_b = Array3::zeros((n, 1, n_rep));
    let mut psi_deriv = Array2::zeros((1, n_rep));
    let mut predictions: Vec<(String, Array3<f64>)> = Vec::new();
    let mut losses: Vec<NuisanceLoss> = Vec::new();
    for (r, rep) in reps.into_iter().enumerate() {
        // DoubleML: σ² = γ / (n·J²) with γ the mean squared score.
        let jac = rep.jacobian;
        let squares: Vec<f64> = rep.psi.iter().map(|v| v * v).collect();
        let gamma = np_mean(&squares);
        let sigma2 = 1.0 / (n as f64 * (jac * jac)) * gamma;
        all_coef[[0, r]] = rep.theta;
        all_se[[0, r]] = sigma2.sqrt();
        psi_deriv[[0, r]] = jac;
        psi.slice_mut(ndarray::s![.., 0, r]).assign(&rep.psi);
        psi_a.slice_mut(ndarray::s![.., 0, r]).fill(jac);
        psi_b
            .slice_mut(ndarray::s![.., 0, r])
            .assign(&(&rep.psi - jac * rep.theta));
        for target in &rep.targets {
            let Some((_, pred)) = rep.predictions.iter().find(|(nm, _)| *nm == target.name) else {
                continue;
            };
            let slot = match losses.iter().position(|l| l.name == target.name) {
                Some(i) => i,
                None => {
                    losses.push(NuisanceLoss {
                        name: target.name.to_string(),
                        kind: target.kind,
                        loss: Array2::from_elem((1, n_rep), f64::NAN),
                        targets: Array3::from_elem((n, n_rep, 1), f64::NAN),
                    });
                    losses.len() - 1
                }
            };
            losses[slot].loss[[0, r]] = target.loss(pred);
            losses[slot]
                .targets
                .slice_mut(ndarray::s![.., r, 0])
                .assign(&target.target);
        }
        for (name, pred) in rep.predictions {
            let slot = match predictions.iter().position(|(nm, _)| nm == name) {
                Some(i) => i,
                None => {
                    predictions
                        .push((name.to_string(), Array3::from_elem((n, n_rep, 1), f64::NAN)));
                    predictions.len() - 1
                }
            };
            predictions[slot]
                .1
                .slice_mut(ndarray::s![.., r, 0])
                .assign(&pred);
        }
    }
    let (coef, se) = aggregate(&all_coef, &all_se);
    DmlFit {
        model: name,
        treatment_names: data.treatment_names().to_vec(),
        n_obs: n,
        n_folds,
        coef,
        se,
        all_coef,
        all_se,
        psi,
        psi_a,
        psi_b,
        psi_deriv,
        predictions,
        sensitivity: None,
        cate: None,
        losses,
        cluster: None,
    }
}

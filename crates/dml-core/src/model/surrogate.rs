//! Long-term treatment effects from short-term surrogates: the surrogate index
//! of Athey, Chetty, Imbens and Kang (2019, "The surrogate index: combining
//! short-term proxies to estimate long-term treatment effects", NBER WP 26463),
//! estimated with the efficient influence function of Chen and Ritzwoller
//! (2023, "Semiparametric estimation of long-term treatment effects", Journal of
//! Econometrics 237).

use ndarray::{concatenate, s, Array1, Array2, Array3, ArrayView2, Axis};

use super::mediation::{classify, fit_regressor, predict_rows};
use crate::crossfit::{map_folds, Fold, SampleSplitting};
use crate::data::gather;
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};

/// An experiment without the long-term outcome and an observational sample
/// with it. Both have the covariates `x` and the surrogates `s` (short-term
/// outcomes, one per column); only the experiment has the binary treatment
/// `w`, only the observational sample the long-term outcome `y`.
///
/// Internally the experimental rows come first, so sample splits (for
/// [`SurrogateIndex::fit_with_folds`]) index the experimental units as
/// `0..n_exp` and the observational ones as `n_exp..n_exp + n_obs`.
#[derive(Debug, Clone)]
pub struct SurrogateData {
    x: Array2<f64>,
    s: Array2<f64>,
    w: Array1<f64>,
    y: Array1<f64>,
    n_exp: usize,
}

impl SurrogateData {
    pub fn new(
        x_exp: Array2<f64>,
        s_exp: Array2<f64>,
        w: Array1<f64>,
        x_obs: Array2<f64>,
        s_obs: Array2<f64>,
        y: Array1<f64>,
    ) -> Result<Self> {
        let (n_exp, n_obs) = (w.len(), y.len());
        if x_exp.nrows() != n_exp || s_exp.nrows() != n_exp {
            return Err(DmlError::DimensionMismatch(format!(
                "experiment: w has {n_exp} rows, x {}, s {}",
                x_exp.nrows(),
                s_exp.nrows()
            )));
        }
        if x_obs.nrows() != n_obs || s_obs.nrows() != n_obs {
            return Err(DmlError::DimensionMismatch(format!(
                "observational sample: y has {n_obs} rows, x {}, s {}",
                x_obs.nrows(),
                s_obs.nrows()
            )));
        }
        if x_exp.ncols() != x_obs.ncols() || s_exp.ncols() != s_obs.ncols() {
            return Err(DmlError::DimensionMismatch(
                "both samples need the same covariate and surrogate columns".into(),
            ));
        }
        if s_exp.ncols() == 0 {
            return Err(DmlError::InvalidArgument(
                "need at least one surrogate".into(),
            ));
        }
        if n_exp < 2 || n_obs < 2 {
            return Err(DmlError::InvalidArgument(
                "both samples need at least 2 observations".into(),
            ));
        }
        if w.iter().any(|&v| v != 0.0 && v != 1.0) {
            return Err(DmlError::InvalidArgument(
                "the treatment w must contain only 0 and 1".into(),
            ));
        }
        let finite = x_exp.iter().chain(&x_obs).all(|v| v.is_finite())
            && s_exp.iter().chain(&s_obs).all(|v| v.is_finite())
            && y.iter().all(|v| v.is_finite());
        if !finite {
            return Err(DmlError::InvalidArgument(
                "data contains NaN or infinite values".into(),
            ));
        }
        let x = concatenate(Axis(0), &[x_exp.view(), x_obs.view()]).expect("same columns");
        let s = concatenate(Axis(0), &[s_exp.view(), s_obs.view()]).expect("same columns");
        let mut w_all = Array1::zeros(n_exp + n_obs);
        w_all.slice_mut(s![..n_exp]).assign(&w);
        let mut y_all = Array1::zeros(n_exp + n_obs);
        y_all.slice_mut(s![n_exp..]).assign(&y);
        Ok(Self {
            x,
            s,
            w: w_all,
            y: y_all,
            n_exp,
        })
    }

    /// Number of experimental units (the first rows).
    pub fn n_experimental(&self) -> usize {
        self.n_exp
    }

    /// Number of observational units (the last rows).
    pub fn n_observational(&self) -> usize {
        self.y.len() - self.n_exp
    }

    /// Strata for the sample splits: treated, control, observational.
    fn strata(&self) -> Vec<usize> {
        (0..self.y.len())
            .map(|i| match (i < self.n_exp, self.w[i] == 1.0) {
                (true, true) => 0,
                (true, false) => 1,
                (false, _) => 2,
            })
            .collect()
    }
}

/// Average effect of a binary treatment on a long-term outcome that the
/// experiment does not observe, through short-term surrogates `S`.
///
/// The experiment (`G = 1`) randomises `W` (given covariates `X`) and records
/// `S`; an observational sample (`G = 0`) records `S` and the long-term outcome
/// `Y`. Under *surrogacy* (`Y ⊥ W | S, X` in the experiment) and *comparability*
/// (`Y | S, X` is distributed alike in both samples), the effect in the
/// experimental population is
///
/// ```text
/// τ = E[μ̄₁(X) − μ̄₀(X) | G = 1],   μ̄_w(X) = E[ν̄(S, X) | W = w, X, G = 1],
/// ```
///
/// where `ν̄(S, X) = E[Y | S, X, G = 0]` is the *surrogate index*. Each
/// potential-outcome mean `E[Y(w) | G = 1]` has the efficient score of Chen and
/// Ritzwoller (2023); for `w = 1`:
///
/// ```text
/// ψ = G·(μ̄₁(X) + W/π(X)·(ν̄ − μ̄₁(X)) − θ)
///     + (1 − G)·ρ(S, X)/(1 − ρ(S, X))·γ(S, X)/π(X)·(Y − ν̄(S, X)),
/// ```
///
/// with the experiment's propensity score `π(X) = P(W = 1 | X, G = 1)` (learned,
/// or known with [`propensity`](Self::propensity)), `γ(S, X) = P(W = 1 | S, X, G = 1)`
/// and the sample-membership probability `ρ(S, X) = P(G = 1 | S, X)`. The last
/// term corrects the surrogate index with the observational outcomes, weighted
/// to the experimental population. `ml_g` learns `ν̄` (on the observational
/// sample), `ml_nested` the `μ̄_w` (regressions of the predicted index on `X`
/// in each arm of the experiment), `ml_m` the three probabilities, which are
/// clipped to `[t, 1 − t]` (`clipping_threshold`, default 0.01). Sample splits
/// are stratified by treated, control and observational units.
///
/// ```
/// use dml_core::{LogisticRegression, Ols, SurrogateData, SurrogateIndex};
/// use ndarray::{Array1, Array2};
/// use rand::{Rng, SeedableRng};
/// use rand_distr::StandardNormal;
///
/// let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
/// let mut sample = |n: usize, experiment: bool| {
///     let x = Array2::from_shape_simple_fn((n, 1), || rng.sample::<f64, _>(StandardNormal));
///     let w: Array1<f64> = (0..n).map(|_| f64::from(u8::from(experiment && rng.gen::<bool>()))).collect();
///     let s = Array2::from_shape_fn((n, 1), |(i, _)| w[i] + x[[i, 0]] + rng.sample::<f64, _>(StandardNormal));
///     // The long-term outcome depends on the treatment only through S.
///     let y: Array1<f64> = (0..n).map(|i| 2.0 * s[[i, 0]] + x[[i, 0]] + rng.sample::<f64, _>(StandardNormal)).collect();
///     (x, s, w, y)
/// };
/// let (x_e, s_e, w, _) = sample(2000, true);
/// let (x_o, s_o, _, y) = sample(3000, false);
/// let data = SurrogateData::new(x_e, s_e, w, x_o, s_o, y)?;
///
/// let fit = SurrogateIndex::new(Ols::new(), LogisticRegression::unpenalized())
///     .propensity(0.5)
///     .fit(&data)?;
/// // Effect on S is 1, so the long-term effect is 2.
/// assert!((fit.effect().coef()[0] - 2.0).abs() < 0.25);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
pub struct SurrogateIndex {
    ml_g: Box<dyn Regressor>,
    ml_m: Box<dyn Classifier>,
    ml_nested: Box<dyn Regressor>,
    propensity: Option<f64>,
    clipping_threshold: f64,
    splitting: SampleSplitting,
}

impl SurrogateIndex {
    /// `ml_g` learns the surrogate index `E[Y | S, X]` (and, through a fresh
    /// copy, the nested means unless [`ml_nested`](Self::ml_nested) is set);
    /// `ml_m` the probabilities `π`, `γ` and `ρ`.
    pub fn new(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Box::new(ml_m))
    }

    /// Like [`SurrogateIndex::new`] for boxed trait objects.
    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Box<dyn Classifier>) -> Self {
        let ml_nested = ml_g.fresh();
        Self {
            ml_g,
            ml_m,
            ml_nested,
            propensity: None,
            clipping_threshold: 1e-2,
            splitting: SampleSplitting::default(),
        }
    }

    /// Learner of the nested means `μ̄_w(X)`.
    pub fn ml_nested(self, ml_nested: impl Regressor + 'static) -> Self {
        self.ml_nested_boxed(Box::new(ml_nested))
    }

    /// Like [`ml_nested`](Self::ml_nested) for a boxed trait object.
    pub fn ml_nested_boxed(mut self, ml_nested: Box<dyn Regressor>) -> Self {
        self.ml_nested = ml_nested;
        self
    }

    /// Known treatment probability of a completely randomised experiment
    /// (otherwise `π(X)` is learned by `ml_m`).
    pub fn propensity(mut self, p: f64) -> Self {
        self.propensity = Some(p);
        self
    }

    /// Clip the estimated probabilities to `[t, 1 − t]` (default 0.01).
    pub fn clipping_threshold(mut self, threshold: f64) -> Self {
        self.clipping_threshold = threshold;
        self
    }

    /// Former name of [`clipping_threshold`](Self::clipping_threshold).
    #[deprecated(since = "0.32.0", note = "renamed to `clipping_threshold`")]
    pub fn trimming_threshold(self, threshold: f64) -> Self {
        self.clipping_threshold(threshold)
    }

    /// Number of cross-fitting folds (default 5).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Number of independent sample splits (default 1), aggregated by the median.
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed for the sample splits (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Draw seeded sample splits, stratified by treated, control and
    /// observational units, and fit.
    pub fn fit(&self, data: &SurrogateData) -> Result<SurrogateFit> {
        let splits = self.splitting.draw_stratified(&data.strata())?;
        self.fit_with_folds(data, &splits)
    }

    /// Fit on given sample splits (one `Vec<Fold>` per repetition) of the
    /// stacked data: experimental units `0..n_exp`, then observational ones.
    pub fn fit_with_folds(
        &self,
        data: &SurrogateData,
        splits: &[Vec<Fold>],
    ) -> Result<SurrogateFit> {
        let t = self.clipping_threshold;
        if t.is_nan() || !(0.0..0.5).contains(&t) {
            return Err(DmlError::InvalidArgument(
                "clipping_threshold must be in [0, 0.5)".into(),
            ));
        }
        if let Some(p) = self.propensity {
            if !(p > 0.0 && p < 1.0) {
                return Err(DmlError::InvalidArgument(
                    "the propensity must be in (0, 1)".into(),
                ));
            }
        }
        if splits.is_empty() {
            return Err(DmlError::InvalidArgument(
                "need at least one repetition of sample splits".into(),
            ));
        }
        let n = data.y.len();
        let n_exp = data.n_exp;
        let sx = concatenate(Axis(1), &[data.s.view(), data.x.view()]).expect("same rows");
        let n_rep = splits.len();
        let mut psi_a = Array3::<f64>::zeros((n, 2, n_rep));
        let mut psi_b = Array3::<f64>::zeros((n, 2, n_rep));
        let names = [
            "ml_g",
            "ml_nested_d1",
            "ml_nested_d0",
            "ml_m",
            "ml_m_surrogate",
            "ml_sample",
        ];
        let mut stored: Vec<(String, Array3<f64>)> = names
            .iter()
            .map(|name| (name.to_string(), Array3::zeros((n, n_rep, 1))))
            .collect();
        for (r, folds) in splits.iter().enumerate() {
            let nu = self.nuisances(data, sx.view(), folds)?;
            let clip = |v: f64| v.clamp(t, 1.0 - t);
            for i in 0..n {
                let (pi, gamma, rho) = (clip(nu[3][i]), clip(nu[4][i]), clip(nu[5][i]));
                let index = nu[0][i];
                if i < n_exp {
                    let w = data.w[i];
                    let (mu1, mu0) = (nu[1][i], nu[2][i]);
                    psi_a[[i, 0, r]] = -1.0;
                    psi_a[[i, 1, r]] = -1.0;
                    psi_b[[i, 0, r]] = mu1 + w / pi * (index - mu1);
                    psi_b[[i, 1, r]] = mu0 + (1.0 - w) / (1.0 - pi) * (index - mu0);
                } else {
                    let odds = rho / (1.0 - rho);
                    let resid = data.y[i] - index;
                    psi_b[[i, 0, r]] = odds * gamma / pi * resid;
                    psi_b[[i, 1, r]] = odds * (1.0 - gamma) / (1.0 - pi) * resid;
                }
            }
            for ((_, all), v) in stored.iter_mut().zip(nu) {
                all.slice_mut(s![.., r, 0]).assign(&v);
            }
        }
        let mut potential = DmlFit::from_linear_scores(
            "Surrogate index",
            vec!["Y(1)".into(), "Y(0)".into()],
            splits[0].len(),
            psi_a,
            psi_b,
        )?;
        potential.predictions = stored;
        let mut effect = potential.combine(
            "Surrogate index".into(),
            vec!["ATE".into()],
            &Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).expect("1 x 2"),
        );
        effect.predictions = potential.predictions.clone();
        Ok(SurrogateFit { effect, potential })
    }

    /// Out-of-fold `[ν̄, μ̄₁, μ̄₀, π, γ, ρ]` for every row (μ̄ and π only
    /// meaningful on experimental rows).
    fn nuisances(
        &self,
        data: &SurrogateData,
        sx: ArrayView2<'_, f64>,
        folds: &[Fold],
    ) -> Result<[Array1<f64>; 6]> {
        let n = data.y.len();
        let n_exp = data.n_exp;
        let x = data.x.view();
        let (y, w) = (data.y.view(), data.w.view());
        let g: Array1<f64> = (0..n).map(|i| f64::from(u8::from(i < n_exp))).collect();
        let per_fold = map_folds(folds.len(), |k| -> Result<[Array1<f64>; 6]> {
            let (train, test) = (&folds[k].train, &folds[k].test);
            let exp: Vec<usize> = train.iter().copied().filter(|&i| i < n_exp).collect();
            let obs: Vec<usize> = train.iter().copied().filter(|&i| i >= n_exp).collect();
            if exp.is_empty() || obs.is_empty() {
                return Err(DmlError::InvalidArgument(
                    "a training fold lacks experimental or observational units".into(),
                ));
            }
            // Surrogate index on the observational sample; its predictions on the
            // experimental training units are out of sample.
            let index = fit_regressor(&*self.ml_g, sx, &obs, gather(y, &obs).view(), "outcome")?;
            let index_test = predict_rows(&*index, sx, test)?;
            let index_exp = predict_rows(&*index, sx, &exp)?;
            let mut nested = Vec::with_capacity(2);
            for arm in [1.0, 0.0] {
                let pos: Vec<usize> = (0..exp.len()).filter(|&j| w[exp[j]] == arm).collect();
                let rows: Vec<usize> = pos.iter().map(|&j| exp[j]).collect();
                let target: Array1<f64> = pos.iter().map(|&j| index_exp[j]).collect();
                let model = fit_regressor(&*self.ml_nested, x, &rows, target.view(), "nested")?;
                nested.push(predict_rows(&*model, x, test)?);
            }
            let mu0 = nested.pop().expect("two arms");
            let mu1 = nested.pop().expect("two arms");
            let pi = match self.propensity {
                Some(p) => Array1::from_elem(test.len(), p),
                None => classify(&*self.ml_m, x, &exp, w, test)?,
            };
            let gamma = classify(&*self.ml_m, sx, &exp, w, test)?;
            let rho = classify(&*self.ml_m, sx, train, g.view(), test)?;
            Ok([index_test, mu1, mu0, pi, gamma, rho])
        })?;
        let mut all: [Array1<f64>; 6] = std::array::from_fn(|_| Array1::from_elem(n, f64::NAN));
        for (fold, values) in folds.iter().zip(per_fold) {
            for (target, v) in all.iter_mut().zip(values) {
                for (j, &i) in fold.test.iter().enumerate() {
                    target[i] = v[j];
                }
            }
        }
        if all.iter().any(|a| a.iter().any(|v| !v.is_finite())) {
            return Err(DmlError::InvalidArgument(
                "folds do not cover every observation, or a learner returned a non-finite value"
                    .into(),
            ));
        }
        Ok(all)
    }
}

/// Result of [`SurrogateIndex::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SurrogateFit {
    effect: DmlFit,
    potential: DmlFit,
}

impl SurrogateFit {
    /// The long-term average treatment effect in the experimental population.
    /// Predictions: `"ml_g"` (the surrogate index `ν̄(S, X)`), `"ml_nested_d1"`,
    /// `"ml_nested_d0"`, `"ml_m"` (`π`), `"ml_m_surrogate"` (`γ`) and
    /// `"ml_sample"` (`ρ`), unclipped.
    pub fn effect(&self) -> &DmlFit {
        &self.effect
    }

    /// `E[Y(1) | G = 1]` and `E[Y(0) | G = 1]`.
    pub fn potential_outcomes(&self) -> &DmlFit {
        &self.potential
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use rand::{Rng, SeedableRng};
    use rand_distr::StandardNormal;

    /// Experiment with confounded assignment (π depends on X), effect on the
    /// surrogates (1, 0.5), long-term effect 2·1 + 1·0.5 = 2.5.
    fn simulate(seed: u64, n_exp: usize, n_obs: usize) -> SurrogateData {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut draw = |n: usize, experiment: bool| {
            let mut normal = || rng.sample::<f64, _>(StandardNormal);
            let x = Array2::from_shape_simple_fn((n, 2), &mut normal);
            let mut w = Array1::zeros(n);
            let mut s = Array2::zeros((n, 2));
            let mut y = Array1::zeros(n);
            for i in 0..n {
                // Observational units: no treatment, surrogates shifted by X.
                if experiment {
                    w[i] = f64::from(u8::from(0.6 * x[[i, 0]] + normal() > 0.0));
                }
                s[[i, 0]] = w[i] + 0.5 * x[[i, 0]] + normal();
                s[[i, 1]] = 0.5 * w[i] - 0.3 * x[[i, 1]] + normal();
                y[i] = 2.0 * s[[i, 0]] + s[[i, 1]] + x[[i, 1]] + normal();
            }
            (x, s, w, y)
        };
        let (xe, se, w, _) = draw(n_exp, true);
        let (xo, so, _, y) = draw(n_obs, false);
        SurrogateData::new(xe, se, w, xo, so, y).unwrap()
    }

    #[test]
    fn coverage() {
        let reps = 40;
        let mut cover = 0;
        for rep in 0..reps {
            let data = simulate(rep, 1500, 2000);
            let fit = SurrogateIndex::new(Ols::new(), LogisticRegression::unpenalized())
                .fit(&data)
                .unwrap();
            let ci = fit.effect().confint(0.95).unwrap();
            cover += usize::from(ci[[0, 0]] <= 2.5 && 2.5 <= ci[[0, 1]]);
            let po = fit.potential_outcomes().coef();
            assert!((fit.effect().coef()[0] - (po[0] - po[1])).abs() < 1e-12);
        }
        assert!(cover as f64 / reps as f64 > 0.85, "coverage {cover}/{reps}");
    }

    #[test]
    fn rejects_bad_input() {
        let x = Array2::zeros((4, 1));
        let s = Array2::zeros((4, 1));
        let bad_w = Array1::from(vec![0.0, 1.0, 2.0, 0.0]);
        let y = Array1::zeros(4);
        assert!(
            SurrogateData::new(x.clone(), s.clone(), bad_w, x.clone(), s.clone(), y.clone())
                .is_err()
        );
        let wide = Array2::zeros((4, 2));
        let w = Array1::from(vec![0.0, 1.0, 1.0, 0.0]);
        assert!(
            SurrogateData::new(x.clone(), s.clone(), w.clone(), wide, s.clone(), y.clone())
                .is_err()
        );
        let data = simulate(1, 200, 200);
        assert!(
            SurrogateIndex::new(Ols::new(), LogisticRegression::unpenalized())
                .propensity(1.0)
                .fit(&data)
                .is_err()
        );
    }
}

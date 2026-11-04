//! Prediction-powered inference (Angelopoulos, Bates, Fannjiang, Jordan &
//! Zrnic, 2023, *Science*; PPI++: Angelopoulos, Duchi & Zrnic, 2023).
//!
//! A small *labelled* sample has the outcome `y`; a large *unlabelled* sample
//! has only predictions `ŷ` of it, from any model (a machine-learning model, an
//! LLM, a simulator). Treating the predictions as data is biased; using only
//! the labelled sample wastes the unlabelled one. Prediction-powered inference
//! corrects the estimate from the predictions with the prediction errors on the
//! labelled sample (the *rectifier*), so the estimate is unbiased whatever the
//! quality of the predictions, and more precise the better they are. For a mean,
//!
//! ```text
//! θ̂_λ = λ · mean_unlabelled(ŷ) + mean_labelled(y − λ ŷ),
//! ```
//!
//! and for a linear or logistic regression of `y` on covariates `x` the loss
//! is rectified the same way. PPI++ ([`PpiLambda::Tuned`], the default) picks
//! the weight `λ ∈ [0, 1]` that minimises the variance, so the estimate is
//! never less precise than the labelled-only one (asymptotically): `λ = 1` is
//! the original PPI, `λ = 0` the classical estimate.
//!
//! The computations follow the reference implementation `ppi_py` (0.2.3:
//! `ppi_mean_ci`, `ppi_ols_ci`, `ppi_logistic_ci`, including its two-step
//! choice of `λ` for regressions). Results are [`DmlFit`]s, so confidence
//! intervals, p-values, linear combinations and confidence sequences
//! ([`anytime`](crate::anytime)) work on them.
//!
//! [`PpiData::cross_fit`] fits the predictor on the labelled sample itself by
//! cross-fitting (cross-prediction-powered inference, Zrnic & Candès, 2024):
//! labelled rows get out-of-fold predictions, unlabelled rows the average of
//! the fold models' predictions.

use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use crate::crossfit::{map_folds, KFold};
use crate::data::gather;
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::learner::{Classifier, Regressor};
use crate::linalg::{cholesky_solve, spd_inverse};

/// The parameter estimated by [`PredictionPowered`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PpiEstimand {
    /// The mean of `y`.
    #[default]
    Mean,
    /// Coefficients of the least-squares regression of `y` on the
    /// covariates (no intercept is added: include a column of ones).
    LinearRegression,
    /// Coefficients of the logistic regression of a `y` in `[0, 1]` on the
    /// covariates (no intercept is added; predictions are probabilities).
    LogisticRegression,
}

/// The weight `λ` of the predictions.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PpiLambda {
    /// PPI++: the variance-minimising `λ`, estimated and clipped to `[0, 1]`.
    #[default]
    Tuned,
    /// A fixed `λ` in `[0, 1]`: 1 is the original PPI, 0 the classical
    /// labelled-only estimate.
    Fixed(f64),
}

/// Labelled outcomes and predictions, unlabelled predictions and, for
/// regressions, the covariates of both samples.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PpiData {
    y: Array1<f64>,
    yhat: Array1<f64>,
    yhat_unlabelled: Array1<f64>,
    covariates: Option<(Array2<f64>, Array2<f64>)>,
    names: Option<Vec<String>>,
    n_folds: usize,
}

impl PpiData {
    /// Outcomes `y` and predictions `yhat` on the labelled sample,
    /// predictions `yhat_unlabelled` on the unlabelled one.
    pub fn new(y: Array1<f64>, yhat: Array1<f64>, yhat_unlabelled: Array1<f64>) -> Result<Self> {
        if y.len() != yhat.len() {
            return Err(DmlError::DimensionMismatch(format!(
                "{} labelled outcomes but {} predictions",
                y.len(),
                yhat.len()
            )));
        }
        if y.len() < 2 || yhat_unlabelled.len() < 2 {
            return Err(DmlError::InvalidArgument(
                "both samples need at least two observations".into(),
            ));
        }
        if !y
            .iter()
            .chain(&yhat)
            .chain(&yhat_unlabelled)
            .all(|v| v.is_finite())
        {
            return Err(DmlError::InvalidArgument(
                "outcomes and predictions must be finite".into(),
            ));
        }
        Ok(Self {
            y,
            yhat,
            yhat_unlabelled,
            covariates: None,
            names: None,
            n_folds: 1,
        })
    }

    /// Predictions by cross-fitting `learner` on the labelled sample
    /// (cross-prediction-powered inference): the model is trained on
    /// `n_folds − 1` folds of `(features, y)` and predicts the held-out fold;
    /// each unlabelled row gets the mean of the `n_folds` models' predictions
    /// from `features_unlabelled`.
    pub fn cross_fit(
        learner: &dyn Regressor,
        y: Array1<f64>,
        features: ArrayView2<'_, f64>,
        features_unlabelled: ArrayView2<'_, f64>,
        n_folds: usize,
        seed: u64,
    ) -> Result<Self> {
        let (yhat, yhat_unlabelled) = cross_predict(
            y.view(),
            features,
            features_unlabelled,
            n_folds,
            seed,
            |x, t, tests| {
                let mut model = learner.fresh();
                model.fit(x, t)?;
                tests.iter().map(|x| model.predict(*x)).collect()
            },
        )?;
        let mut data = Self::new(y, yhat, yhat_unlabelled)?;
        data.n_folds = n_folds;
        Ok(data)
    }

    /// [`cross_fit`](Self::cross_fit) with a classifier: the predictions are
    /// probabilities of `y = 1` (for a binary `y`, e.g. with
    /// [`PpiEstimand::LogisticRegression`]).
    pub fn cross_fit_proba(
        learner: &dyn Classifier,
        y: Array1<f64>,
        features: ArrayView2<'_, f64>,
        features_unlabelled: ArrayView2<'_, f64>,
        n_folds: usize,
        seed: u64,
    ) -> Result<Self> {
        let (yhat, yhat_unlabelled) = cross_predict(
            y.view(),
            features,
            features_unlabelled,
            n_folds,
            seed,
            |x, t, tests| {
                let mut model = learner.fresh();
                model.fit(x, t)?;
                tests.iter().map(|x| model.predict_proba(*x)).collect()
            },
        )?;
        let mut data = Self::new(y, yhat, yhat_unlabelled)?;
        data.n_folds = n_folds;
        Ok(data)
    }

    /// Covariates of the regression, `x` for the labelled and
    /// `x_unlabelled` for the unlabelled sample (same columns; no intercept
    /// is added).
    pub fn with_covariates(mut self, x: Array2<f64>, x_unlabelled: Array2<f64>) -> Result<Self> {
        if x.nrows() != self.y.len() || x_unlabelled.nrows() != self.yhat_unlabelled.len() {
            return Err(DmlError::DimensionMismatch(format!(
                "covariates have {} and {} rows for samples of {} and {}",
                x.nrows(),
                x_unlabelled.nrows(),
                self.y.len(),
                self.yhat_unlabelled.len()
            )));
        }
        if x.ncols() == 0 || x.ncols() != x_unlabelled.ncols() {
            return Err(DmlError::DimensionMismatch(
                "both samples need the same, non-empty set of covariates".into(),
            ));
        }
        if !x.iter().chain(&x_unlabelled).all(|v| v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "covariates must be finite".into(),
            ));
        }
        // Row-major, so the blocked passes read each row contiguously.
        let standard = |a: Array2<f64>| {
            if a.is_standard_layout() {
                a
            } else {
                a.as_standard_layout().into_owned()
            }
        };
        self.covariates = Some((standard(x), standard(x_unlabelled)));
        Ok(self)
    }

    /// Names of the regression coefficients (default `x0, x1, …`).
    pub fn with_covariate_names(mut self, names: Vec<String>) -> Self {
        self.names = Some(names);
        self
    }

    pub fn y(&self) -> ArrayView1<'_, f64> {
        self.y.view()
    }

    /// Predictions on the labelled sample.
    pub fn yhat(&self) -> ArrayView1<'_, f64> {
        self.yhat.view()
    }

    /// Predictions on the unlabelled sample.
    pub fn yhat_unlabelled(&self) -> ArrayView1<'_, f64> {
        self.yhat_unlabelled.view()
    }

    pub fn n_labelled(&self) -> usize {
        self.y.len()
    }

    pub fn n_unlabelled(&self) -> usize {
        self.yhat_unlabelled.len()
    }
}

/// Out-of-fold predictions on the labelled rows and fold-averaged ones on
/// the unlabelled rows; `fit_predict(train_x, train_y, [test_x, unlabelled_x])`.
fn cross_predict<F>(
    y: ArrayView1<'_, f64>,
    features: ArrayView2<'_, f64>,
    features_unlabelled: ArrayView2<'_, f64>,
    n_folds: usize,
    seed: u64,
    fit_predict: F,
) -> Result<(Array1<f64>, Array1<f64>)>
where
    F: Fn(
            ArrayView2<'_, f64>,
            ArrayView1<'_, f64>,
            &[ArrayView2<'_, f64>],
        ) -> Result<Vec<Array1<f64>>>
        + Sync,
{
    let n = y.len();
    if features.nrows() != n || features.ncols() != features_unlabelled.ncols() {
        return Err(DmlError::DimensionMismatch(
            "features need one row per labelled outcome and the same columns in both samples"
                .into(),
        ));
    }
    let folds = KFold::new(n_folds, seed)?.split(n)?;
    let preds = map_folds(folds.len(), |k| {
        let fold = &folds[k];
        let train_x = features.select(Axis(0), &fold.train);
        let test_x = features.select(Axis(0), &fold.test);
        fit_predict(
            train_x.view(),
            gather(y, &fold.train).view(),
            &[test_x.view(), features_unlabelled.view()],
        )
    })?;
    let mut yhat = Array1::<f64>::zeros(n);
    let mut yhat_unlabelled = Array1::<f64>::zeros(features_unlabelled.nrows());
    for (fold, p) in folds.iter().zip(preds) {
        if p[0].len() != fold.test.len() || p[1].len() != yhat_unlabelled.len() {
            return Err(DmlError::DimensionMismatch(
                "learner returned wrong number of predictions".into(),
            ));
        }
        for (&i, &v) in fold.test.iter().zip(&p[0]) {
            yhat[i] = v;
        }
        yhat_unlabelled += &p[1];
    }
    yhat_unlabelled /= n_folds as f64;
    Ok((yhat, yhat_unlabelled))
}

/// Prediction-powered estimate of a mean or of regression coefficients.
///
/// ```
/// # use dml_core::ppi::{PpiData, PredictionPowered};
/// # use ndarray::Array1;
/// // Labelled: 200 outcomes with predictions; unlabelled: 5000 predictions.
/// let f = |i: usize| (i as f64 * 0.7).sin();
/// let y: Array1<f64> = (0..200).map(|i| 1.0 + f(i) + 0.3 * (i as f64 * 2.9).cos()).collect();
/// let yhat: Array1<f64> = (0..200).map(|i| 1.1 + f(i)).collect();
/// let yhat_u: Array1<f64> = (200..5200).map(|i| 1.1 + f(i)).collect();
/// let fit = PredictionPowered::mean().fit(&PpiData::new(y, yhat, yhat_u)?)?;
/// // Unbiased despite the predictions' offset of 0.1, and more precise than
/// // the labelled mean alone.
/// assert!(fit.estimate().se()[0] < fit.classical().se()[0]);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct PredictionPowered {
    estimand: PpiEstimand,
    lambda: PpiLambda,
}

impl PredictionPowered {
    pub fn new(estimand: PpiEstimand) -> Self {
        Self {
            estimand,
            lambda: PpiLambda::Tuned,
        }
    }

    /// The mean of `y`.
    pub fn mean() -> Self {
        Self::new(PpiEstimand::Mean)
    }

    /// Least-squares coefficients of `y` on the covariates.
    pub fn linear_regression() -> Self {
        Self::new(PpiEstimand::LinearRegression)
    }

    /// Logistic-regression coefficients of `y` on the covariates.
    pub fn logistic_regression() -> Self {
        Self::new(PpiEstimand::LogisticRegression)
    }

    /// Weight of the predictions (default [`PpiLambda::Tuned`]).
    pub fn lambda(mut self, lambda: PpiLambda) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn fit(&self, data: &PpiData) -> Result<PpiFit> {
        if let PpiLambda::Fixed(l) = self.lambda {
            if !(0.0..=1.0).contains(&l) {
                return Err(DmlError::InvalidArgument(format!(
                    "lambda must be in [0, 1], got {l}"
                )));
            }
        }
        match self.estimand {
            PpiEstimand::Mean => {
                if data.covariates.is_some() {
                    return Err(DmlError::InvalidArgument(
                        "the mean takes no covariates".into(),
                    ));
                }
                fit_mean(data, self.lambda)
            }
            PpiEstimand::LinearRegression => fit_glm(data, Glm::Linear, self.lambda),
            PpiEstimand::LogisticRegression => {
                if !data
                    .y
                    .iter()
                    .chain(&data.yhat)
                    .chain(&data.yhat_unlabelled)
                    .all(|v| (0.0..=1.0).contains(v))
                {
                    return Err(DmlError::InvalidArgument(
                        "logistic regression needs outcomes and predictions in [0, 1]".into(),
                    ));
                }
                fit_glm(data, Glm::Logistic, self.lambda)
            }
        }
    }
}

/// A prediction-powered estimate with the classical (labelled-only)
/// estimate for comparison.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PpiFit {
    estimate: DmlFit,
    classical: DmlFit,
    lambda: f64,
    n_labelled: usize,
    n_unlabelled: usize,
}

impl PpiFit {
    /// The prediction-powered estimate. Its scores span both samples
    /// (labelled rows first), so `n_obs` is `n_labelled + n_unlabelled`.
    pub fn estimate(&self) -> &DmlFit {
        &self.estimate
    }

    /// The classical estimate from the labelled sample alone (`λ = 0`).
    pub fn classical(&self) -> &DmlFit {
        &self.classical
    }

    /// The weight `λ` of the predictions that was used.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn n_labelled(&self) -> usize {
        self.n_labelled
    }

    pub fn n_unlabelled(&self) -> usize {
        self.n_unlabelled
    }

    /// Labelled sample size the classical estimate would need for the
    /// precision of the prediction-powered one: `n · (se_classical / se)²`,
    /// per estimate.
    pub fn effective_sample_size(&self) -> Array1<f64> {
        let ratio = &self.classical.se / &self.estimate.se;
        ratio.mapv(|r| self.n_labelled as f64 * r * r)
    }
}

/// A fit whose scores are `θ̂ + IF`, labelled rows then (when given)
/// unlabelled rows, so that its standard error is `√(Σ IF² ) / n_rows`.
fn assemble(
    model: &str,
    names: Vec<String>,
    n_folds: usize,
    theta: &Array1<f64>,
    influence: Array2<f64>,
) -> Result<DmlFit> {
    let (rows, k) = influence.dim();
    if !influence.iter().all(|v| v.is_finite()) || !theta.iter().all(|v| v.is_finite()) {
        return Err(DmlError::InvalidArgument(
            "the estimate is not finite".into(),
        ));
    }
    let psi = influence
        .into_shape_with_order((rows, k, 1))
        .expect("(rows, k) to (rows, k, 1)");
    let mut psi_b = psi.clone();
    for (j, mut col) in psi_b.axis_iter_mut(Axis(1)).enumerate() {
        col += theta[j];
    }
    let psi_a = Array3::from_elem((rows, k, 1), -1.0);
    let all_coef = theta.clone().insert_axis(Axis(1));
    Ok(DmlFit::from_scores(
        model.into(),
        names,
        n_folds.max(1),
        all_coef,
        psi,
        psi_a,
        psi_b,
        Vec::new(),
    ))
}

/// Centred rows of `a` times `scale` (and, if given, times the symmetric
/// `h_inv`), written into `out`.
fn centred_into(
    mut out: ndarray::ArrayViewMut2<'_, f64>,
    a: &Array2<f64>,
    h_inv: Option<&Array2<f64>>,
    scale: f64,
) {
    let mean = column_sums(a.view()) / a.nrows() as f64;
    let centred = a - &mean;
    match h_inv {
        Some(h) => out.assign(&(centred.dot(h) * scale)),
        None => out.assign(&(centred * scale)),
    }
}

fn fit_mean(data: &PpiData, lambda: PpiLambda) -> Result<PpiFit> {
    let (n, nu) = (data.y.len(), data.yhat_unlabelled.len());
    let lam = match lambda {
        PpiLambda::Fixed(l) => l,
        _ => {
            // ppi_py: gradients y − θ, ŷ − θ at the λ = 1 estimate with an
            // identity Hessian; λ only uses them centred, so θ drops out.
            tuned_lambda(
                column(&data.y),
                column(&data.yhat),
                column(&data.yhat_unlabelled),
                &Array2::eye(1),
            )
        }
    };
    let estimate = mean_fit(data, lam, data.n_folds, true)?;
    let classical = mean_fit(data, 0.0, data.n_folds, false)?;
    Ok(PpiFit {
        estimate,
        classical,
        lambda: lam,
        n_labelled: n,
        n_unlabelled: nu,
    })
}

/// Sum of each column (`sum_axis(Axis(0))` adds a tall array row by row,
/// which is slow).
fn column_sums(a: ArrayView2<'_, f64>) -> Array1<f64> {
    a.columns().into_iter().map(|c| c.sum()).collect()
}

fn column(v: &Array1<f64>) -> ArrayView2<'_, f64> {
    v.view().insert_axis(Axis(1))
}

/// Mean estimate with weight `lam`; variances with `ddof = 0` (ppi_py's
/// `std`). Without the unlabelled rows it is the labelled mean.
fn mean_fit(data: &PpiData, lam: f64, n_folds: usize, unlabelled: bool) -> Result<DmlFit> {
    let (n, nu) = (data.y.len(), data.yhat_unlabelled.len());
    let rows = if unlabelled { n + nu } else { n };
    let rect = |i: usize| data.y[i] - lam * data.yhat[i];
    let mean_rect = (0..n).map(rect).sum::<f64>() / n as f64;
    let mut influence = Array2::<f64>::zeros((rows, 1));
    let scale = rows as f64 / n as f64;
    for i in 0..n {
        influence[[i, 0]] = (rect(i) - mean_rect) * scale;
    }
    let mut theta = mean_rect;
    if unlabelled {
        let mean_imputed = lam * data.yhat_unlabelled.mean().expect("N > 0");
        let scale = rows as f64 / nu as f64;
        let out = influence.slice_mut(s![n.., 0]);
        ndarray::Zip::from(out)
            .and(&data.yhat_unlabelled)
            .for_each(|o, &v| *o = (lam * v - mean_imputed) * scale);
        theta += mean_imputed;
    }
    let model = if unlabelled {
        "PPI mean"
    } else {
        "classical mean"
    };
    assemble(
        model,
        vec!["mean".into()],
        n_folds,
        &Array1::from_elem(1, theta),
        influence,
    )
}

#[derive(Debug, Clone, Copy)]
enum Glm {
    Linear,
    Logistic,
}

impl Glm {
    fn mean(self, eta: f64) -> f64 {
        match self {
            Glm::Linear => eta,
            Glm::Logistic => expit(eta),
        }
    }

    fn weight(self, eta: f64) -> f64 {
        match self {
            Glm::Linear => 1.0,
            Glm::Logistic => {
                let p = expit(eta);
                p * (1.0 - p)
            }
        }
    }
}

/// `1 / (1 + e^{−x})` without overflow (ppi_py's `safe_expit`).
fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + e^x)` without overflow.
fn log1pexp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Gradients of the loss at θ on the labelled rows (with `y` and with
/// `ŷ`), on the unlabelled rows (with `ŷ`), and the inverse Hessian
/// (ppi_py's `_ols_get_stats` / `_logistic_get_stats`).
struct Stats {
    grads: Array2<f64>,
    grads_hat: Array2<f64>,
    grads_hat_unlabelled: Array2<f64>,
    inv_hessian: Array2<f64>,
}

fn glm_stats(
    glm: Glm,
    theta: &Array1<f64>,
    x: &Array2<f64>,
    xu: &Array2<f64>,
    data: &PpiData,
    use_unlabelled: bool,
) -> Result<Stats> {
    let (n, nu, d) = (x.nrows(), xu.nrows(), x.ncols());
    let denom = if use_unlabelled {
        (n + nu) as f64
    } else {
        n as f64
    };
    let eta = x.dot(theta);
    let mu = eta.mapv(|e| glm.mean(e));
    let mut hessian = weighted_gram(x, &eta.mapv(|e| glm.weight(e) / denom))?;
    let grads = x * &(&mu - &data.y).insert_axis(Axis(1));
    let grads_hat = x * &(&mu - &data.yhat).insert_axis(Axis(1));
    let grads_hat_unlabelled = if use_unlabelled {
        let eta_u = xu.dot(theta);
        hessian += &weighted_gram(xu, &eta_u.mapv(|e| glm.weight(e) / denom))?;
        let r = eta_u.mapv(|e| glm.mean(e)) - &data.yhat_unlabelled;
        xu * &r.insert_axis(Axis(1))
    } else {
        Array2::zeros((nu, d))
    };
    Ok(Stats {
        grads,
        grads_hat,
        grads_hat_unlabelled,
        inv_hessian: spd_inverse(&hessian)?,
    })
}

/// Rows per block of the parallel passes over the samples.
const BLOCK: usize = 4096;

/// `f` on blocks of rows `0..n` (in parallel), results in block order.
fn map_blocks<T, F>(n: usize, f: F) -> Result<Vec<T>>
where
    T: Send,
    F: Fn(std::ops::Range<usize>) -> Result<T> + Sync,
{
    map_folds(
        n.div_ceil(BLOCK),
        |b| f(b * BLOCK..((b + 1) * BLOCK).min(n)),
    )
}

/// Upper triangle (row-major `d × d`) to a full symmetric matrix.
fn symmetric(lower: &[f64], d: usize) -> Array2<f64> {
    Array2::from_shape_fn((d, d), |(a, b)| lower[a.max(b) * d + a.min(b)])
}

/// `Xᵀ diag(w) X` for row-major `x`, summed over row blocks in block order
/// (so the result does not depend on the number of threads).
fn weighted_gram(x: &Array2<f64>, w: &Array1<f64>) -> Result<Array2<f64>> {
    let d = x.ncols();
    let xs = x.as_slice().expect("row-major covariates");
    let parts = map_blocks(x.nrows(), |rows| {
        let mut m = vec![0.0; d * d];
        for i in rows {
            add_outer(&mut m, &xs[i * d..(i + 1) * d], w[i]);
        }
        Ok(m)
    })?;
    Ok(symmetric(
        &sum_parts(parts.iter().map(Vec::as_slice), d * d),
        d,
    ))
}

/// Lower triangle of `w · row rowᵀ` added to `m` (row-major `d × d`).
fn add_outer(m: &mut [f64], row: &[f64], w: f64) {
    let d = row.len();
    for a in 0..d {
        let wa = w * row[a];
        for b in 0..=a {
            m[a * d + b] += wa * row[b];
        }
    }
}

fn sum_parts<'a>(parts: impl Iterator<Item = &'a [f64]>, len: usize) -> Vec<f64> {
    let mut total = vec![0.0; len];
    for part in parts {
        for (t, v) in total.iter_mut().zip(part) {
            *t += v;
        }
    }
    total
}

/// `Σ_i [−t_i η_i + c · log(1 + e^{η_i})]` with `η = Xθ`, and with
/// `derivatives` its gradient `Σ (c p_i − t_i) x_i` and Hessian
/// `Σ c p_i (1 − p_i) x_i x_iᵀ`, in one blocked pass.
fn logistic_moments(
    x: &Array2<f64>,
    t: ArrayView1<'_, f64>,
    c: f64,
    theta: &Array1<f64>,
    derivatives: bool,
) -> Result<(f64, Array1<f64>, Array2<f64>)> {
    let d = x.ncols();
    let xs = x.as_slice().expect("row-major covariates");
    let parts = map_blocks(x.nrows(), |rows| {
        let mut value = 0.0;
        let mut out = vec![0.0; if derivatives { d + d * d } else { 0 }];
        for i in rows {
            let row = &xs[i * d..(i + 1) * d];
            let eta: f64 = row.iter().zip(theta).map(|(a, b)| a * b).sum();
            value += -t[i] * eta + c * log1pexp(eta);
            if derivatives {
                let p = expit(eta);
                let r = c * p - t[i];
                let (grad, hess) = out.split_at_mut(d);
                for (g, v) in grad.iter_mut().zip(row) {
                    *g += r * v;
                }
                add_outer(hess, row, c * p * (1.0 - p));
            }
        }
        Ok((value, out))
    })?;
    let value = parts.iter().map(|p| p.0).sum();
    if !derivatives {
        return Ok((value, Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let total = sum_parts(parts.iter().map(|p| p.1.as_slice()), d + d * d);
    Ok((
        value,
        Array1::from(total[..d].to_vec()),
        symmetric(&total[d..], d),
    ))
}

/// PPI++'s variance-minimising λ (ppi_py's `_calc_lam_glm`, `coord=None`,
/// `clip=True`): `tr(H⁻¹ C H⁻¹) / (2 (1 + n/N) tr(H⁻¹ V H⁻¹))`, with `C` the
/// symmetrised covariance of the gradients with `y` and with `ŷ` and `V` the
/// covariance (`ddof = 1`) of the `ŷ` gradients of both samples.
fn tuned_lambda(
    grads: ArrayView2<'_, f64>,
    grads_hat: ArrayView2<'_, f64>,
    grads_hat_unlabelled: ArrayView2<'_, f64>,
    inv_hessian: &Array2<f64>,
) -> f64 {
    let (n, nu, d) = (
        grads.nrows() as f64,
        grads_hat_unlabelled.nrows() as f64,
        grads.ncols(),
    );
    let gc = &grads - &(column_sums(grads) / n);
    let ghc = &grads_hat - &(column_sums(grads_hat) / n);
    let cross = gc.t().dot(&ghc);
    let cov = (&cross + &cross.t()) / n;
    // Covariance of the stacked ŷ gradients of both samples, in two passes
    // without stacking them.
    let mean = (column_sums(grads_hat) + column_sums(grads_hat_unlabelled)) / (n + nu);
    let mut var = Array2::<f64>::zeros((d, d));
    for a in 0..d {
        for b in a..d {
            let (ma, mb) = (mean[a], mean[b]);
            let sum = |g: &ArrayView2<'_, f64>| {
                ndarray::Zip::from(g.column(a))
                    .and(g.column(b))
                    .fold(0.0, |acc, &u, &v| acc + (u - ma) * (v - mb))
            };
            var[[a, b]] = sum(&grads_hat) + sum(&grads_hat_unlabelled);
            var[[b, a]] = var[[a, b]];
        }
    }
    var /= n + nu - 1.0;
    let trace = |m: &Array2<f64>| inv_hessian.dot(m).dot(inv_hessian).diag().sum();
    let lam = trace(&cov) / (2.0 * (1.0 + n / nu) * trace(&var));
    if lam.is_finite() {
        lam.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn fit_glm(data: &PpiData, glm: Glm, lambda: PpiLambda) -> Result<PpiFit> {
    let (x, xu) = data.covariates.as_ref().ok_or_else(|| {
        DmlError::InvalidArgument("regressions need covariates (PpiData::with_covariates)".into())
    })?;
    let d = x.ncols();
    let names = match &data.names {
        Some(names) if names.len() == d => names.clone(),
        Some(names) => {
            return Err(DmlError::DimensionMismatch(format!(
                "{} names for {d} covariates",
                names.len()
            )))
        }
        None => (0..d).map(|j| format!("x{j}")).collect(),
    };
    let lam = match lambda {
        PpiLambda::Fixed(l) => l,
        _ => {
            // ppi_py's ppi_*_ci: λ₁ from the gradients at the λ = 1 estimate,
            // then λ₂ from the gradients at the λ₁ estimate.
            let mut lam = 1.0;
            for _ in 0..2 {
                let theta = glm_estimate(glm, lam, x, xu, data)?;
                let st = glm_stats(glm, &theta, x, xu, data, true)?;
                lam = tuned_lambda(
                    st.grads.view(),
                    st.grads_hat.view(),
                    st.grads_hat_unlabelled.view(),
                    &st.inv_hessian,
                );
            }
            lam
        }
    };
    let model = match glm {
        Glm::Linear => "PPI linear regression",
        Glm::Logistic => "PPI logistic regression",
    };
    let estimate = glm_fit(glm, lam, x, xu, data, true, model, names.clone())?;
    let classical_model = match glm {
        Glm::Linear => "classical linear regression",
        Glm::Logistic => "classical logistic regression",
    };
    let classical = glm_fit(glm, 0.0, x, xu, data, false, classical_model, names)?;
    Ok(PpiFit {
        estimate,
        classical,
        lambda: lam,
        n_labelled: x.nrows(),
        n_unlabelled: xu.nrows(),
    })
}

/// Influence functions `−H⁻¹ (g_i − ḡ)` scaled so that the standard errors
/// are ppi_py's (`np.cov`, `ddof = 1`, in each sample).
#[allow(clippy::too_many_arguments)]
fn glm_fit(
    glm: Glm,
    lam: f64,
    x: &Array2<f64>,
    xu: &Array2<f64>,
    data: &PpiData,
    unlabelled: bool,
    model: &str,
    names: Vec<String>,
) -> Result<DmlFit> {
    let (n, nu, d) = (x.nrows(), xu.nrows(), x.ncols());
    let theta = glm_estimate(glm, lam, x, xu, data)?;
    let st = glm_stats(glm, &theta, x, xu, data, unlabelled && lam != 0.0)?;
    let rows = if unlabelled { n + nu } else { n };
    let neg_inv = -&st.inv_hessian;
    let mut influence = Array2::<f64>::zeros((rows, d));
    let rect = &st.grads - &(&st.grads_hat * lam);
    let scale = |m: usize| rows as f64 / ((m * (m - 1)) as f64).sqrt();
    centred_into(
        influence.slice_mut(s![..n, ..]),
        &rect,
        Some(&neg_inv),
        scale(n),
    );
    if unlabelled {
        let imputed = &st.grads_hat_unlabelled * lam;
        centred_into(
            influence.slice_mut(s![n.., ..]),
            &imputed,
            Some(&neg_inv),
            scale(nu),
        );
    }
    assemble(model, names, data.n_folds, &theta, influence)
}

/// Point estimate with weight `lam`. Linear: ppi_py's
/// `OLS(x_u, λ ŷ_u) + OLS(x, y − λ ŷ)`; logistic: the minimiser of the
/// rectified loss `λ L_u(ŷ) − λ L(ŷ) + L(y)` (convex for λ ≤ 1), by Newton.
fn glm_estimate(
    glm: Glm,
    lam: f64,
    x: &Array2<f64>,
    xu: &Array2<f64>,
    data: &PpiData,
) -> Result<Array1<f64>> {
    match glm {
        Glm::Linear => {
            let mut theta = least_squares(x, &(&data.y - &(&data.yhat * lam)))?;
            if lam != 0.0 {
                theta += &least_squares(xu, &(&data.yhat_unlabelled * lam))?;
            }
            Ok(theta)
        }
        Glm::Logistic => logistic_estimate(lam, x, xu, data),
    }
}

fn least_squares(x: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
    cholesky_solve(&x.t().dot(x), &x.t().dot(y))
}

fn logistic_estimate(
    lam: f64,
    x: &Array2<f64>,
    xu: &Array2<f64>,
    data: &PpiData,
) -> Result<Array1<f64>> {
    let (n, nu, d) = (x.nrows() as f64, xu.nrows() as f64, x.ncols());
    // λ/N Σ ℓ(ŷ_u) + 1/n Σ [ℓ(y) − λ ℓ(ŷ)] with ℓ(t) = −t η + log(1 + e^η),
    // so a labelled row contributes −(y − λ ŷ) η + (1 − λ) log(1 + e^η).
    let target = &data.y - &(&data.yhat * lam);
    let moments =
        |theta: &Array1<f64>, derivatives: bool| -> Result<(f64, Array1<f64>, Array2<f64>)> {
            let (v, g, h) = logistic_moments(x, target.view(), 1.0 - lam, theta, derivatives)?;
            let (mut v, mut g, mut h) = (v / n, g / n, h / n);
            if lam != 0.0 {
                let (vu, gu, hu) =
                    logistic_moments(xu, data.yhat_unlabelled.view(), 1.0, theta, derivatives)?;
                v += lam * vu / nu;
                if derivatives {
                    g.scaled_add(lam / nu, &gu);
                    h.scaled_add(lam / nu, &hu);
                }
            }
            Ok((v, g, h))
        };
    let mut theta = Array1::<f64>::zeros(d);
    for _ in 0..200 {
        let (current, grad, hess) = moments(&theta, true)?;
        let step = cholesky_solve(&hess, &grad)?;
        // Newton decrement: near the optimum the loss changes by less than
        // its rounding error, so a line search would reject good steps; take
        // full Newton steps there (they converge quadratically).
        let small = grad.dot(&step) <= 1e-10 * current.abs().max(1.0);
        let mut t = 1.0;
        let mut next = &theta - &step;
        if !small {
            while t > 1e-10 {
                let (value, _, _) = moments(&next, false)?;
                if value <= current {
                    break;
                }
                t *= 0.5;
                next = &theta - &(&step * t);
            }
        }
        let change = step.iter().fold(0.0_f64, |m, v| m.max(v.abs())) * t;
        theta = next;
        if change <= 1e-13 * (1.0 + theta.iter().fold(0.0_f64, |m, v| m.max(v.abs()))) {
            return Ok(theta);
        }
    }
    Err(DmlError::InvalidArgument(
        "the logistic regression did not converge (separable data?)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn normal(rng: &mut ChaCha8Rng) -> f64 {
        <StandardNormal as Distribution<f64>>::sample(&StandardNormal, rng)
    }

    #[test]
    fn mean_with_fixed_lambda_is_the_closed_form() {
        let y = Array1::from(vec![1.0, 2.0, 4.0, 3.0]);
        let yhat = Array1::from(vec![1.5, 2.0, 3.0, 3.5]);
        let yu = Array1::from(vec![2.0, 3.0, 2.5, 1.0, 4.0]);
        let data = PpiData::new(y.clone(), yhat.clone(), yu.clone()).unwrap();
        let fit = PredictionPowered::mean()
            .lambda(PpiLambda::Fixed(0.5))
            .fit(&data)
            .unwrap();
        let rect = &y - &(&yhat * 0.5);
        let imp = &yu * 0.5;
        let theta = rect.mean().unwrap() + imp.mean().unwrap();
        let var = rect.var(0.0) / 4.0 + imp.var(0.0) / 5.0;
        assert!((fit.estimate().coef()[0] - theta).abs() < 1e-14);
        assert!((fit.estimate().se()[0] - var.sqrt()).abs() < 1e-14);
        assert!((fit.classical().coef()[0] - 2.5).abs() < 1e-14);
        assert!((fit.classical().se()[0] - (y.var(0.0) / 4.0).sqrt()).abs() < 1e-14);
        assert_eq!(fit.lambda(), 0.5);
    }

    #[test]
    fn lambda_zero_is_ordinary_least_squares_with_sandwich_errors() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let n = 300;
        let x = Array2::from_shape_fn((n, 2), |(_, j)| if j == 0 { 1.0 } else { normal(&mut rng) });
        let y: Array1<f64> = (0..n)
            .map(|i| 0.5 + 2.0 * x[[i, 1]] + normal(&mut rng) * (1.0 + x[[i, 1]].abs()))
            .collect();
        let xu = Array2::from_shape_fn(
            (50, 2),
            |(_, j)| if j == 0 { 1.0 } else { normal(&mut rng) },
        );
        let data = PpiData::new(y.clone(), y.mapv(|v| v * 0.9), Array1::zeros(50))
            .unwrap()
            .with_covariates(x.clone(), xu)
            .unwrap();
        let fit = PredictionPowered::linear_regression()
            .lambda(PpiLambda::Fixed(0.0))
            .fit(&data)
            .unwrap();
        let xtx_inv = spd_inverse(&x.t().dot(&x)).unwrap();
        let beta = xtx_inv.dot(&x.t().dot(&y));
        let resid = &y - &x.dot(&beta);
        let meat = weighted_gram(&x, &resid.mapv(|r| r * r)).unwrap();
        // HC0 times n / (n − 1) (ppi_py's np.cov).
        let cov = xtx_inv.dot(&meat).dot(&xtx_inv) * (n as f64 / (n as f64 - 1.0));
        for j in 0..2 {
            assert!((fit.estimate().coef()[j] - beta[j]).abs() < 1e-12);
            assert!((fit.estimate().se()[j] - cov[[j, j]].sqrt()).abs() < 1e-12);
            assert!((fit.classical().se()[j] - cov[[j, j]].sqrt()).abs() < 1e-12);
        }
    }

    /// Good predictions: PPI++ is unbiased, covers the truth and is much more
    /// precise than the labelled-only estimate; useless predictions: λ goes
    /// to about 0 and nothing is lost.
    #[test]
    fn tuned_lambda_gains_with_good_predictions_and_loses_nothing_with_bad_ones() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (n, nu) = (300, 20_000);
        let draw = |rng: &mut ChaCha8Rng, m: usize| -> (Array1<f64>, Array1<f64>) {
            let z: Array1<f64> = (0..m).map(|_| normal(rng)).collect();
            let y =
                z.mapv(|v| 1.0 + v) + (0..m).map(|_| 0.3 * normal(rng)).collect::<Array1<f64>>();
            (y, z)
        };
        let (y, z) = draw(&mut rng, n);
        let (_, zu) = draw(&mut rng, nu);
        let good = PpiData::new(y.clone(), z.mapv(|v| 1.2 + v), zu.mapv(|v| 1.2 + v)).unwrap();
        let fit = PredictionPowered::mean().fit(&good).unwrap();
        assert!(fit.lambda() > 0.8);
        assert!((fit.estimate().coef()[0] - 1.0).abs() < 3.0 * fit.estimate().se()[0]);
        assert!(fit.effective_sample_size()[0] > 5.0 * n as f64);
        let noise: Array1<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let noise_u: Array1<f64> = (0..nu).map(|_| normal(&mut rng)).collect();
        let bad = PpiData::new(y, noise, noise_u).unwrap();
        let fit = PredictionPowered::mean().fit(&bad).unwrap();
        assert!(fit.lambda() < 0.2);
        assert!(fit.estimate().se()[0] < 1.02 * fit.classical().se()[0]);
    }

    #[test]
    fn logistic_regression_solves_the_rectified_score() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        // The predictions also see a feature z outside the regression, so
        // they explain part of y given x.
        let make = |rng: &mut ChaCha8Rng, m: usize| {
            let x = Array2::from_shape_fn((m, 2), |(_, j)| if j == 0 { 1.0 } else { normal(rng) });
            let p: Array1<f64> = x
                .rows()
                .into_iter()
                .map(|r| expit(-0.5 + 1.5 * r[1] + 2.0 * normal(rng)))
                .collect();
            (x, p)
        };
        let (x, p) = make(&mut rng, 400);
        let y = p.mapv(|v| f64::from(rand::Rng::gen::<f64>(&mut rng) < v));
        let (xu, pu) = make(&mut rng, 4000);
        let data = PpiData::new(y.clone(), p.clone(), pu.clone())
            .unwrap()
            .with_covariates(x.clone(), xu.clone())
            .unwrap();
        let fit = PredictionPowered::logistic_regression()
            .lambda(PpiLambda::Fixed(0.7))
            .fit(&data)
            .unwrap();
        let theta = fit.estimate().coef().to_owned();
        // Rectified score: λ/N X_uᵀ(μ_u − ŷ_u) − λ/n Xᵀ(μ − ŷ) + 1/n Xᵀ(μ − y) = 0.
        let mu = x.dot(&theta).mapv(expit);
        let mu_u = xu.dot(&theta).mapv(expit);
        let score = xu.t().dot(&(&mu_u - &pu)) * (0.7 / 4000.0)
            - x.t().dot(&(&mu - &p)) * (0.7 / 400.0)
            + x.t().dot(&(&mu - &y)) / 400.0;
        assert!(score.iter().all(|v| v.abs() < 1e-12), "{score}");
        let classical = fit.classical();
        assert!((theta[1] - classical.coef()[1]).abs() < 3.0 * classical.se()[1]);
        assert!(fit.estimate().se()[1] < 0.9 * classical.se()[1]);
    }

    #[test]
    fn cross_fitting_predicts_out_of_fold_and_averages_the_fold_models() {
        use crate::learner::Ols;
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let f = Array2::from_shape_fn((120, 2), |_| normal(&mut rng));
        let fu = Array2::from_shape_fn((30, 2), |_| normal(&mut rng));
        let y: Array1<f64> = f
            .rows()
            .into_iter()
            .map(|r| 1.0 + r[0] - r[1] + 0.5 * normal(&mut rng))
            .collect();
        let data = PpiData::cross_fit(&Ols::new(), y.clone(), f.view(), fu.view(), 4, 3).unwrap();
        let folds = KFold::new(4, 3).unwrap().split(120).unwrap();
        let mut expected_u = Array1::<f64>::zeros(30);
        for fold in &folds {
            let mut m = Ols::new();
            m.fit(
                f.select(Axis(0), &fold.train).view(),
                gather(y.view(), &fold.train).view(),
            )
            .unwrap();
            let p = m.predict(f.select(Axis(0), &fold.test).view()).unwrap();
            for (&i, v) in fold.test.iter().zip(&p) {
                assert!((data.yhat()[i] - v).abs() < 1e-12);
            }
            expected_u += &(m.predict(fu.view()).unwrap() / 4.0);
        }
        for (a, b) in data.yhat_unlabelled().iter().zip(&expected_u) {
            assert!((a - b).abs() < 1e-12);
        }
        let fit = PredictionPowered::mean().fit(&data).unwrap();
        assert_eq!(fit.estimate().n_folds(), 4);
        assert_eq!(fit.estimate().n_obs(), 150);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let v = Array1::from(vec![0.2, 0.4, 0.9]);
        let data = PpiData::new(v.clone(), v.clone(), v.clone()).unwrap();
        assert!(PredictionPowered::mean()
            .lambda(PpiLambda::Fixed(1.5))
            .fit(&data)
            .is_err());
        assert!(PredictionPowered::linear_regression().fit(&data).is_err());
        assert!(PpiData::new(v.clone(), Array1::zeros(2), v.clone()).is_err());
        assert!(PpiData::new(v.clone(), v.clone(), Array1::from(vec![f64::NAN, 1.0])).is_err());
        let with_x = data
            .clone()
            .with_covariates(Array2::ones((3, 1)), Array2::ones((3, 1)))
            .unwrap();
        assert!(PredictionPowered::mean().fit(&with_x).is_err());
        let big = PpiData::new(Array1::from(vec![2.0, 0.0, 1.0]), v.clone(), v.clone())
            .unwrap()
            .with_covariates(Array2::ones((3, 1)), Array2::ones((3, 1)))
            .unwrap();
        assert!(PredictionPowered::logistic_regression().fit(&big).is_err());
    }
}

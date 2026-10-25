//! Cross-validated linear learners, as scikit-learn's `LassoCV` and
//! `LogisticRegressionCV`: the penalty is chosen by K-fold cross-validation
//! on the training data, then the model is refitted on all of it. Used as
//! nuisance learners, the tuning happens inside every cross-fitting fold.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::lasso::{lasso_path, Lasso};
use super::logistic::LogisticRegression;
use super::{Classifier, Regressor};
use crate::error::{DmlError, Result};

/// numpy's `finfo(float64).resolution`, scikit-learn's floor for the alpha grid.
const RESOLUTION: f64 = 1e-15;

pub(super) fn check_cv(n: usize, cv: usize) -> Result<()> {
    if cv < 2 {
        return Err(DmlError::InvalidArgument("cv must be at least 2".into()));
    }
    if n < cv {
        return Err(DmlError::InvalidArgument(format!(
            "cannot cross-validate {n} observations with {cv} folds"
        )));
    }
    Ok(())
}

/// Test sets of scikit-learn's `KFold(cv)` without shuffling: contiguous
/// blocks, the first `n % cv` of them one observation larger.
pub(super) fn kfold_test_sets(n: usize, cv: usize) -> Vec<Vec<usize>> {
    let mut start = 0;
    (0..cv)
        .map(|k| {
            let size = n / cv + usize::from(k < n % cv);
            let block = (start..start + size).collect();
            start += size;
            block
        })
        .collect()
}

/// Test sets of scikit-learn's `StratifiedKFold(cv)` without shuffling, for
/// 0/1 labels: each class (numbered by order of first appearance) is dealt
/// round-robin over the sorted labels, then assigned to folds in data order.
pub(super) fn stratified_test_sets(y: ArrayView1<'_, f64>, cv: usize) -> Result<Vec<Vec<usize>>> {
    let n = y.len();
    let first = y[0];
    let n_first = y.iter().filter(|&&v| v == first).count();
    let counts = [n_first, n - n_first];
    if counts.iter().all(|&c| c < cv) {
        return Err(DmlError::InvalidArgument(format!(
            "cv = {cv} is larger than the number of members in each class"
        )));
    }
    // Positions i, i + cv, i + 2cv, … of the sorted labels go to fold i; the
    // first `n_first` sorted labels belong to the first class.
    let dealt = |i: usize, upto: usize| if i < upto { (upto - 1 - i) / cv + 1 } else { 0 };
    let mut fold_of = vec![0usize; n];
    for (class, &count) in counts.iter().enumerate() {
        let folds: Vec<usize> = (0..cv)
            .flat_map(|k| {
                let in_first = dealt(k, n_first);
                let m = if class == 0 {
                    in_first
                } else {
                    dealt(k, n) - in_first
                };
                std::iter::repeat(k).take(m)
            })
            .collect();
        debug_assert_eq!(folds.len(), count);
        let members = (0..n).filter(|&i| (y[i] == first) == (class == 0));
        for (i, k) in members.zip(folds) {
            fold_of[i] = k;
        }
    }
    Ok((0..cv)
        .map(|k| (0..n).filter(|&i| fold_of[i] == k).collect())
        .collect())
}

/// `fit_fold(test set)` for every fold, in parallel with the `parallel`
/// feature; results are in fold order either way.
pub(super) fn map_folds<T: Send>(
    folds: &[Vec<usize>],
    fit_fold: impl Fn(&[usize]) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        folds.par_iter().map(|test| fit_fold(test)).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        folds.iter().map(|test| fit_fold(test)).collect()
    }
}

pub(super) fn complement(n: usize, test: &[usize]) -> Vec<usize> {
    let mut in_test = vec![false; n];
    for &i in test {
        in_test[i] = true;
    }
    (0..n).filter(|&i| !in_test[i]).collect()
}

/// `n` points from `start` to `stop` evenly spaced on a log scale, computed
/// as numpy's `geomspace`.
fn geomspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let (a, b) = (start.log10(), stop.log10());
    let step = (b - a) / (n - 1) as f64;
    let mut out: Vec<f64> = (0..n).map(|i| 10f64.powf(a + i as f64 * step)).collect();
    out[0] = start;
    out[n - 1] = stop;
    out
}

/// Weighted mean (plain mean without weights).
fn average(values: impl Iterator<Item = f64>, w: Option<ArrayView1<'_, f64>>) -> f64 {
    match w {
        None => {
            let (sum, count) = values.fold((0.0, 0usize), |(s, c), v| (s + v, c + 1));
            sum / count as f64
        }
        Some(w) => {
            let sum: f64 = values.zip(w.iter()).map(|(v, &wi)| v * wi).sum();
            sum / w.sum()
        }
    }
}

/// Lasso with the penalty chosen by K-fold cross-validation, as
/// scikit-learn's `LassoCV`.
///
/// The candidate penalties are `n_alphas` values evenly spaced on a log scale
/// from `α_max = max_j |x̃_jᵀ ỹ| / n` (the smallest penalty giving an
/// all-zero solution) down to `eps · α_max`. For each fold the whole path is
/// fitted with warm starts, the penalty with the lowest mean test MSE is
/// chosen, and the Lasso is refitted on all the data. Folds are contiguous
/// blocks (scikit-learn's default `KFold` without shuffling), so results
/// reproduce `LassoCV(cv=5)`.
#[derive(Debug, Clone)]
pub struct LassoCV {
    n_alphas: usize,
    eps: f64,
    alphas: Option<Vec<f64>>,
    cv: usize,
    tol: f64,
    max_iter: usize,
    fitted: Option<LassoCvFit>,
}

#[derive(Debug, Clone)]
struct LassoCvFit {
    alpha: f64,
    alphas: Vec<f64>,
    mse_path: Array2<f64>,
    model: Lasso,
}

impl Default for LassoCV {
    fn default() -> Self {
        Self::new()
    }
}

impl LassoCV {
    /// 100 penalties down to `1e-3 · α_max`, 5 folds (scikit-learn's defaults).
    pub fn new() -> Self {
        Self {
            n_alphas: 100,
            eps: 1e-3,
            alphas: None,
            cv: 5,
            tol: 1e-4,
            max_iter: 100_000,
            fitted: None,
        }
    }

    /// Number of penalties on the automatic grid.
    pub fn n_alphas(mut self, n_alphas: usize) -> Self {
        self.n_alphas = n_alphas.max(1);
        self
    }

    /// Ratio of the smallest to the largest penalty on the automatic grid.
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Explicit candidate penalties instead of the automatic grid.
    pub fn alphas(mut self, alphas: Vec<f64>) -> Self {
        self.alphas = Some(alphas);
        self
    }

    /// Number of cross-validation folds.
    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Coordinate-descent tolerance, as [`Lasso::tol`].
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The chosen penalty.
    pub fn alpha(&self) -> Option<f64> {
        self.fitted.as_ref().map(|f| f.alpha)
    }

    /// The candidate penalties, in decreasing order.
    pub fn alphas_path(&self) -> Option<&[f64]> {
        self.fitted.as_ref().map(|f| f.alphas.as_slice())
    }

    /// Test MSE of every penalty on every fold, shape `(n_alphas, cv)`.
    pub fn mse_path(&self) -> Option<&Array2<f64>> {
        self.fitted.as_ref().map(|f| &f.mse_path)
    }

    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.fitted.as_ref().and_then(|f| f.model.coef())
    }

    pub fn intercept(&self) -> f64 {
        self.fitted.as_ref().map_or(0.0, |f| f.model.intercept())
    }

    fn grid(
        &self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<Vec<f64>> {
        if let Some(alphas) = &self.alphas {
            if alphas.is_empty() || alphas.iter().any(|a| !(a.is_finite() && *a >= 0.0)) {
                return Err(DmlError::InvalidArgument(
                    "alphas must be non-empty, finite and non-negative".into(),
                ));
            }
            let mut a = alphas.clone();
            a.sort_by(|p, q| q.total_cmp(p));
            return Ok(a);
        }
        let c = super::center(x, y, w, true)?;
        let alpha_max =
            c.xc.t()
                .dot(&c.yc)
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                / x.nrows() as f64;
        if alpha_max <= RESOLUTION {
            return Ok(vec![RESOLUTION; self.n_alphas]);
        }
        Ok(geomspace(alpha_max, alpha_max * self.eps, self.n_alphas))
    }

    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        let n = x.nrows();
        check_cv(n, self.cv)?;
        let alphas = self.grid(x, y, w)?;
        // Test MSE of every penalty, per fold.
        let fold_mse = map_folds(&kfold_test_sets(n, self.cv), |test| {
            let train = complement(n, test);
            let w_train = w.map(|w| w.select(Axis(0), &train));
            let w_test = w.map(|w| w.select(Axis(0), test));
            let path = lasso_path(
                x.select(Axis(0), &train).view(),
                y.select(Axis(0), &train).view(),
                w_train.as_ref().map(|w| w.view()),
                &alphas,
                self.tol,
                self.max_iter,
            )?;
            let (x_test, y_test) = (x.select(Axis(0), test), y.select(Axis(0), test));
            Ok(path
                .iter()
                .map(|(coef, intercept)| {
                    let pred = x_test.dot(coef) + *intercept;
                    let sq = pred.iter().zip(&y_test).map(|(p, t)| (p - t).powi(2));
                    average(sq, w_test.as_ref().map(|w| w.view()))
                })
                .collect::<Vec<f64>>())
        })?;
        let mse_path = Array2::from_shape_fn((alphas.len(), self.cv), |(a, k)| fold_mse[k][a]);
        // First penalty with the lowest mean MSE (numpy's argmin).
        let mean_mse = mse_path.mean_axis(Axis(1)).expect("cv >= 2");
        let best = (0..alphas.len()).fold(0, |b, a| if mean_mse[a] < mean_mse[b] { a } else { b });
        let mut model = Lasso::new(alphas[best])
            .tol(self.tol)
            .max_iter(self.max_iter);
        match w {
            Some(w) => model.fit_weighted(x, y, w)?,
            None => model.fit(x, y)?,
        }
        self.fitted = Some(LassoCvFit {
            alpha: alphas[best],
            alphas,
            mse_path,
            model,
        });
        Ok(())
    }
}

impl Regressor for LassoCV {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_impl(x, y, None)
    }

    /// Weighted fits and weighted test MSEs, as `LassoCV.fit(sample_weight=w)`.
    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.fit_impl(x, y, Some(w))
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.fitted
            .as_ref()
            .ok_or(DmlError::NotFitted)?
            .model
            .predict(x)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self {
            fitted: None,
            ..self.clone()
        })
    }
}

/// Criterion for choosing `C` in [`LogisticRegressionCV`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CvScoring {
    /// Mean log loss of the predicted probabilities (scikit-learn's
    /// `"neg_log_loss"`, its default from version 1.11). The natural choice for
    /// propensity scores, which enter DML scores as `1 / m(x)`.
    #[default]
    LogLoss,
    /// Share of correctly classified observations (scikit-learn's default
    /// before 1.11).
    Accuracy,
}

/// L2-penalised logistic regression with `C` chosen by stratified K-fold
/// cross-validation, as scikit-learn's `LogisticRegressionCV`.
///
/// The candidates are `n_cs` values of `C` evenly spaced on a log scale
/// between `1e-4` and `1e4` (or given explicitly). Each is fitted on every
/// fold (warm-started along the path), the `C` with the best total score is
/// chosen and the model is refitted on all the data. Folds follow
/// scikit-learn's `StratifiedKFold` without shuffling.
#[derive(Debug, Clone)]
pub struct LogisticRegressionCV {
    cs: Vec<f64>,
    cv: usize,
    scoring: CvScoring,
    tol: f64,
    max_iter: usize,
    fitted: Option<LogitCvFit>,
}

#[derive(Debug, Clone)]
struct LogitCvFit {
    c: f64,
    scores: Array2<f64>,
    model: LogisticRegression,
}

impl Default for LogisticRegressionCV {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegressionCV {
    /// Ten values of `C` from `1e-4` to `1e4`, 5 folds, log-loss scoring.
    pub fn new() -> Self {
        Self {
            cs: geomspace(1e-4, 1e4, 10),
            cv: 5,
            scoring: CvScoring::LogLoss,
            tol: 1e-10,
            max_iter: 100,
            fitted: None,
        }
    }

    /// `n` values of `C` evenly spaced on a log scale from `1e-4` to `1e4`.
    pub fn n_cs(mut self, n: usize) -> Self {
        self.cs = geomspace(1e-4, 1e4, n.max(1));
        self
    }

    /// Explicit candidate values of `C`, tried in the given order.
    pub fn cs(mut self, cs: Vec<f64>) -> Self {
        self.cs = cs;
        self
    }

    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    pub fn scoring(mut self, scoring: CvScoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Newton tolerance, as [`LogisticRegression::tol`].
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The candidate values of `C`.
    pub fn cs_path(&self) -> &[f64] {
        &self.cs
    }

    /// The chosen `C`.
    pub fn c(&self) -> Option<f64> {
        self.fitted.as_ref().map(|f| f.c)
    }

    /// Score of every `C` on every fold (higher is better; the log loss enters
    /// negated), shape `(cv, n_cs)`.
    pub fn scores(&self) -> Option<&Array2<f64>> {
        self.fitted.as_ref().map(|f| &f.scores)
    }

    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.fitted.as_ref().and_then(|f| f.model.coef())
    }

    pub fn intercept(&self) -> f64 {
        self.fitted.as_ref().map_or(0.0, |f| f.model.intercept())
    }

    fn score(&self, prob: &Array1<f64>, y: &Array1<f64>) -> f64 {
        match self.scoring {
            CvScoring::LogLoss => {
                let eps = f64::EPSILON;
                -average(
                    prob.iter().zip(y).map(|(&p, &t)| {
                        let p = p.clamp(eps, 1.0 - eps);
                        -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                    }),
                    None,
                )
            }
            CvScoring::Accuracy => average(
                prob.iter()
                    .zip(y)
                    .map(|(&p, &t)| f64::from((p > 0.5) == (t == 1.0))),
                None,
            ),
        }
    }
}

impl Classifier for LogisticRegressionCV {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        let n = x.nrows();
        check_cv(n, self.cv)?;
        if y.len() != n {
            return Err(DmlError::DimensionMismatch(
                "LogisticRegressionCV::fit x/y rows".into(),
            ));
        }
        if self.cs.is_empty() || self.cs.iter().any(|c| c.is_nan() || *c <= 0.0) {
            return Err(DmlError::InvalidArgument(
                "the candidate values of C must be positive".into(),
            ));
        }
        if !y.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(DmlError::InvalidArgument(
                "LogisticRegressionCV labels must be 0 or 1".into(),
            ));
        }
        let n_cs = self.cs.len();
        // Per fold: the score and solution of every C, warm-started along the path.
        let per_fold = map_folds(&stratified_test_sets(y, self.cv)?, |test| {
            let train = complement(n, test);
            let (x_train, y_train) = (x.select(Axis(0), &train), y.select(Axis(0), &train));
            let (x_test, y_test) = (x.select(Axis(0), test), y.select(Axis(0), test));
            let mut init: Option<Array1<f64>> = None;
            let mut fold = Vec::with_capacity(n_cs);
            for &c in &self.cs {
                let mut m = LogisticRegression::new(c)
                    .tol(self.tol)
                    .max_iter(self.max_iter);
                m.fit_from(x_train.view(), y_train.view(), init.as_ref())?;
                let score = self.score(&m.predict_proba(x_test.view())?, &y_test);
                init = m.params();
                fold.push((score, init.clone().expect("fitted")));
            }
            Ok(fold)
        })?;
        let scores = Array2::from_shape_fn((self.cv, n_cs), |(k, j)| per_fold[k][j].0);
        let params: Vec<Vec<Array1<f64>>> = per_fold
            .into_iter()
            .map(|fold| fold.into_iter().map(|(_, b)| b).collect())
            .collect();
        // First C with the highest total score (numpy's argmax).
        let total = scores.sum_axis(Axis(0));
        let best = (0..n_cs).fold(0, |b, j| if total[j] > total[b] { j } else { b });
        // Refit on all data, started from the folds' mean solution (as scikit-learn).
        let init = params
            .iter()
            .map(|p| p[best].clone())
            .reduce(|a, b| a + b)
            .map(|s| s / self.cv as f64);
        let mut model = LogisticRegression::new(self.cs[best])
            .tol(self.tol)
            .max_iter(self.max_iter);
        model.fit_from(x, y, init.as_ref())?;
        self.fitted = Some(LogitCvFit {
            c: self.cs[best],
            scores,
            model,
        });
        Ok(())
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.fitted
            .as_ref()
            .ok_or(DmlError::NotFitted)?
            .model
            .predict_proba(x)
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::new(Self {
            fitted: None,
            ..self.clone()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn kfold_blocks_match_scikit_learn() {
        // KFold(3).split(range(8)) -> [0, 1, 2], [3, 4, 5], [6, 7]
        assert_eq!(
            kfold_test_sets(8, 3),
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7]]
        );
    }

    #[test]
    fn stratified_folds_match_scikit_learn() {
        // StratifiedKFold(3).split(X, y) test sets for this y, from scikit-learn 1.9.1.
        let y = array![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        assert_eq!(
            stratified_test_sets(y.view(), 3).unwrap(),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]
        );
    }

    #[test]
    fn geomspace_hits_both_ends() {
        let g = geomspace(1e-4, 1e4, 10);
        assert_eq!((g[0], g[9]), (1e-4, 1e4));
        assert!((g[5] / g[4] - 10f64.powf(8.0 / 9.0)).abs() < 1e-12);
    }

    #[test]
    fn lasso_cv_picks_a_grid_penalty_and_refits() {
        let x = Array2::from_shape_fn((60, 4), |(i, j)| {
            (((i * 7 + j * 5) % 13) as f64 - 6.0) / 4.0
        });
        let y: Array1<f64> = x
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, r)| 1.0 + 2.0 * r[0] - r[2] + ((i % 5) as f64 - 2.0) * 0.1)
            .collect();
        let mut m = LassoCV::new().n_alphas(20);
        m.fit(x.view(), y.view()).unwrap();
        let alpha = m.alpha().unwrap();
        assert!(m.alphas_path().unwrap().contains(&alpha));
        assert_eq!(m.mse_path().unwrap().dim(), (20, 5));
        let mut direct = Lasso::new(alpha);
        direct.fit(x.view(), y.view()).unwrap();
        assert_eq!(direct.coef(), m.coef());
        assert!(m.fresh().predict(x.view()).is_err()); // fresh copies are unfitted
    }
}

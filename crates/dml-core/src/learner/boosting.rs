//! Gradient boosting (Friedman, 2001) with CART regression trees.
//!
//! Each stage fits a shallow tree to the negative gradient of the loss and
//! adds it, scaled by the learning rate, to the current prediction. This
//! follows scikit-learn's `GradientBoostingRegressor` / `Classifier`:
//!
//! - **Regression** (squared error): start from the (weighted) mean; trees
//!   are fitted to the residuals and their leaves predict mean residuals.
//! - **Classification** (log loss): start from the log-odds of the class-1
//!   share; trees are fitted to `y − p`, then every leaf takes one Newton
//!   step `Σ w(y − p) / Σ w·p(1 − p)`.
//!
//! Optional row subsampling (stochastic gradient boosting) and random feature
//! subsets are seeded per stage, so results are reproducible.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::seq::index::sample;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::forest::{MaxFeatures, Node, Tree, TreeBuilder, TreeParams};
use super::{Classifier, Regressor};
use crate::error::{DmlError, Result};

/// Settings shared by the regressor and the classifier.
#[derive(Debug, Clone)]
struct Booster {
    n_estimators: usize,
    learning_rate: f64,
    params: TreeParams,
    subsample: f64,
    seed: u64,
    trees: Vec<Tree>,
    init: f64,
    n_features: usize,
    importances: Option<Array1<f64>>,
}

/// The loss being boosted.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Loss {
    SquaredError,
    LogLoss,
}

fn sigmoid(v: f64) -> f64 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let e = v.exp();
        e / (1.0 + e)
    }
}

impl Booster {
    fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            params: TreeParams {
                max_depth: Some(3),
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::All,
            },
            subsample: 1.0,
            seed: 0,
            trees: Vec::new(),
            init: 0.0,
            n_features: 0,
            importances: None,
        }
    }

    fn unfitted(&self) -> Self {
        Self {
            trees: Vec::new(),
            importances: None,
            ..self.clone()
        }
    }

    fn validate(&self) -> Result<()> {
        if self.n_estimators == 0 {
            return Err(DmlError::InvalidArgument(
                "n_estimators must be at least 1".into(),
            ));
        }
        if !(self.learning_rate > 0.0 && self.learning_rate.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "learning_rate must be positive".into(),
            ));
        }
        if !(self.subsample > 0.0 && self.subsample <= 1.0) {
            return Err(DmlError::InvalidArgument(
                "subsample must be in (0, 1]".into(),
            ));
        }
        if self.params.min_samples_leaf == 0 || self.params.min_samples_split < 2 {
            return Err(DmlError::InvalidArgument(
                "min_samples_leaf must be ≥ 1 and min_samples_split ≥ 2".into(),
            ));
        }
        if let MaxFeatures::Fraction(f) = self.params.max_features {
            if !(f > 0.0 && f <= 1.0) {
                return Err(DmlError::InvalidArgument(
                    "max_features fraction must be in (0, 1]".into(),
                ));
            }
        }
        Ok(())
    }

    fn fit(
        &mut self,
        loss: Loss,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        self.validate()?;
        let (n, p) = x.dim();
        if y.len() != n {
            return Err(DmlError::DimensionMismatch("boosting fit x/y rows".into()));
        }
        if n == 0 {
            return Err(DmlError::InvalidArgument("empty training set".into()));
        }
        let w: Vec<f64> = match w {
            Some(w) => {
                if w.len() != n {
                    return Err(DmlError::DimensionMismatch("sample weight length".into()));
                }
                if !w.iter().all(|&v| v >= 0.0 && v.is_finite()) || w.sum() <= 0.0 {
                    return Err(DmlError::InvalidArgument(
                        "sample weights must be non-negative, finite and not all zero".into(),
                    ));
                }
                w.to_vec()
            }
            None => vec![1.0; n],
        };
        let total_w: f64 = w.iter().sum();
        let mean = y.iter().zip(&w).map(|(&a, &b)| a * b).sum::<f64>() / total_w;
        self.init = match loss {
            Loss::SquaredError => mean,
            Loss::LogLoss => {
                if !(mean > 0.0 && mean < 1.0) {
                    return Err(DmlError::InvalidArgument(
                        "GradientBoostingClassifier needs both classes in the training data".into(),
                    ));
                }
                (mean / (1.0 - mean)).ln()
            }
        };
        let n_try = self.params.max_features.resolve(p);
        let n_sub = ((self.subsample * n as f64) as usize).max(1);
        let mut raw = vec![self.init; n];
        let mut grad = vec![0.0; n];
        let mut importance = vec![0.0; p];
        self.trees = Vec::with_capacity(self.n_estimators);

        for m in 0..self.n_estimators {
            let stage_seed = self
                .seed
                .wrapping_add((m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = ChaCha8Rng::seed_from_u64(stage_seed);
            // Negative gradient: residual (squared error) or y − p (log loss).
            for i in 0..n {
                grad[i] = match loss {
                    Loss::SquaredError => y[i] - raw[i],
                    Loss::LogLoss => y[i] - sigmoid(raw[i]),
                };
            }
            let mut rows: Vec<usize> = if n_sub < n {
                sample(&mut rng, n, n_sub).into_vec()
            } else {
                (0..n).collect()
            };
            rows.sort_unstable();
            rows.retain(|&i| w[i] > 0.0);
            let (mut tree, gains) = TreeBuilder {
                x,
                y: &grad,
                w: &w,
                params: self.params,
                n_try,
                rng,
                importance: vec![0.0; p],
            }
            .grow(rows.clone());

            if loss == Loss::LogLoss {
                newton_leaf_values(&mut tree, x, &rows, &grad, &raw, &w);
            }
            for (acc, g) in importance.iter_mut().zip(&gains) {
                *acc += g;
            }
            for (i, r) in raw.iter_mut().enumerate() {
                *r += self.learning_rate * tree.predict_row(x.row(i));
            }
            self.trees.push(tree);
        }
        let total: f64 = importance.iter().sum();
        self.importances = Some(if total > 0.0 {
            Array1::from(importance) / total
        } else {
            Array1::zeros(p)
        });
        self.n_features = p;
        Ok(())
    }

    /// Raw additive score `init + lr·Σ tree(x)`.
    fn decision(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(DmlError::NotFitted);
        }
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "x has {} columns, the model was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        Ok(x.rows()
            .into_iter()
            .map(|row| {
                self.init
                    + self.learning_rate
                        * self.trees.iter().map(|t| t.predict_row(row)).sum::<f64>()
            })
            .collect())
    }
}

/// Log-loss Newton step per leaf over the in-bag `rows` (as scikit-learn's
/// `_update_terminal_regions`): `Σ w·g / Σ w·p(1 − p)`, or 0 if the
/// denominator vanishes.
fn newton_leaf_values(
    tree: &mut Tree,
    x: ArrayView2<'_, f64>,
    rows: &[usize],
    grad: &[f64],
    raw: &[f64],
    w: &[f64],
) {
    let mut num = vec![0.0; tree.nodes.len()];
    let mut den = vec![0.0; tree.nodes.len()];
    for &i in rows {
        let leaf = tree.leaf_index(x.row(i));
        let p = sigmoid(raw[i]);
        num[leaf] += w[i] * grad[i];
        den[leaf] += w[i] * p * (1.0 - p);
    }
    for (k, node) in tree.nodes.iter_mut().enumerate() {
        if let Node::Leaf { value } = node {
            *value = if den[k].abs() < 1e-150 {
                0.0
            } else {
                num[k] / den[k]
            };
        }
    }
}

/// Builder methods shared by both boosting models.
macro_rules! boosting_builders {
    () => {
        /// Number of boosting stages (default 100).
        pub fn n_estimators(mut self, n: usize) -> Self {
            self.booster.n_estimators = n;
            self
        }

        /// Shrinkage applied to every tree (default 0.1). Smaller values need
        /// more stages but usually generalise better.
        pub fn learning_rate(mut self, rate: f64) -> Self {
            self.booster.learning_rate = rate;
            self
        }

        /// Maximum depth of each tree (default 3); `None` for unlimited.
        pub fn max_depth(mut self, depth: Option<usize>) -> Self {
            self.booster.params.max_depth = depth;
            self
        }

        /// Minimum number of rows to split a node (default 2).
        pub fn min_samples_split(mut self, n: usize) -> Self {
            self.booster.params.min_samples_split = n;
            self
        }

        /// Minimum number of rows in each leaf (default 1).
        pub fn min_samples_leaf(mut self, n: usize) -> Self {
            self.booster.params.min_samples_leaf = n;
            self
        }

        /// Features considered per split (default: all).
        pub fn max_features(mut self, m: MaxFeatures) -> Self {
            self.booster.params.max_features = m;
            self
        }

        /// Fraction of rows drawn (without replacement) for each stage
        /// (default 1.0); below 1 gives stochastic gradient boosting.
        pub fn subsample(mut self, fraction: f64) -> Self {
            self.booster.subsample = fraction;
            self
        }

        /// Seed for row subsampling and feature subsets (default 0).
        pub fn seed(mut self, seed: u64) -> Self {
            self.booster.seed = seed;
            self
        }

        /// Total impurity reduction per feature over all trees (sums to one).
        pub fn feature_importances(&self) -> Option<&Array1<f64>> {
            self.booster.importances.as_ref()
        }

        /// Number of fitted trees.
        pub fn n_trees(&self) -> usize {
            self.booster.trees.len()
        }
    };
}

/// Gradient boosting for regression (squared error). Defaults match
/// scikit-learn's `GradientBoostingRegressor`: 100 stages, learning rate 0.1,
/// depth-3 trees.
///
/// ```
/// use dml_core::GradientBoostingRegressor;
/// let gb = GradientBoostingRegressor::new()
///     .n_estimators(300)
///     .learning_rate(0.05)
///     .max_depth(Some(4))
///     .subsample(0.8)
///     .seed(1);
/// # let _ = gb;
/// ```
#[derive(Debug, Clone)]
pub struct GradientBoostingRegressor {
    booster: Booster,
}

impl Default for GradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientBoostingRegressor {
    pub fn new() -> Self {
        Self {
            booster: Booster::new(),
        }
    }

    boosting_builders!();
}

impl Regressor for GradientBoostingRegressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.booster.fit(Loss::SquaredError, x, y, None)
    }

    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.booster.fit(Loss::SquaredError, x, y, Some(w))
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.booster.decision(x)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self {
            booster: self.booster.unfitted(),
        })
    }
}

/// Gradient boosting for 0/1 classification (log loss). Defaults match
/// scikit-learn's `GradientBoostingClassifier`.
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    booster: Booster,
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientBoostingClassifier {
    pub fn new() -> Self {
        Self {
            booster: Booster::new(),
        }
    }

    boosting_builders!();
}

impl Classifier for GradientBoostingClassifier {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        if !y.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(DmlError::InvalidArgument(
                "GradientBoostingClassifier labels must be 0 or 1".into(),
            ));
        }
        self.booster.fit(Loss::LogLoss, x, y, None)
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        Ok(self.booster.decision(x)?.mapv(sigmoid))
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::new(Self {
            booster: self.booster.unfitted(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use rand::Rng;

    fn nonlinear(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x = Array2::from_shape_simple_fn((n, 3), || rng.gen::<f64>() * 4.0 - 2.0);
        let y = x
            .rows()
            .into_iter()
            .map(|r| (2.0 * r[0]).sin() + r[1] * r[1] - 0.5 * r[2] + 0.1 * (rng.gen::<f64>() - 0.5))
            .collect();
        (x, y)
    }

    fn r2(y: &Array1<f64>, pred: &Array1<f64>) -> f64 {
        let mean = y.mean().unwrap();
        1.0 - (y - pred).mapv(|e| e * e).sum() / y.mapv(|v| (v - mean).powi(2)).sum()
    }

    #[test]
    fn learns_a_nonlinear_function() {
        let (x, y) = nonlinear(2000, 1);
        let (x_test, y_test) = nonlinear(500, 2);
        let mut gb = GradientBoostingRegressor::new().n_estimators(300);
        gb.fit(x.view(), y.view()).unwrap();
        let score = r2(&y_test, &gb.predict(x_test.view()).unwrap());
        assert!(score > 0.95, "R² = {score}");
        assert_eq!(gb.n_trees(), 300);
        assert!((gb.feature_importances().unwrap().sum() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn more_stages_reduce_training_error() {
        let (x, y) = nonlinear(500, 3);
        let train_r2 = |n| {
            let mut gb = GradientBoostingRegressor::new().n_estimators(n);
            gb.fit(x.view(), y.view()).unwrap();
            r2(&y, &gb.predict(x.view()).unwrap())
        };
        let (a, b, c) = (train_r2(10), train_r2(50), train_r2(200));
        assert!(a < b && b < c, "{a} {b} {c}");
    }

    #[test]
    fn single_stump_with_unit_rate_fits_a_step() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![1.0, 1.0, 5.0, 5.0];
        let mut gb = GradientBoostingRegressor::new()
            .n_estimators(1)
            .learning_rate(1.0)
            .max_depth(Some(1));
        gb.fit(x.view(), y.view()).unwrap();
        assert_eq!(gb.predict(x.view()).unwrap(), y);
    }

    #[test]
    fn deterministic_and_subsampling_changes_the_fit() {
        let (x, y) = nonlinear(400, 4);
        let fit = |sub: f64, seed: u64| {
            let mut gb = GradientBoostingRegressor::new()
                .n_estimators(30)
                .subsample(sub)
                .seed(seed);
            gb.fit(x.view(), y.view()).unwrap();
            gb.predict(x.view()).unwrap()
        };
        assert_eq!(fit(0.7, 1), fit(0.7, 1));
        assert_ne!(fit(0.7, 1), fit(0.7, 2));
        assert_eq!(fit(1.0, 1), fit(1.0, 2)); // no randomness without subsampling or feature subsets
    }

    #[test]
    fn weights_equal_row_duplication() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = array![0.0, 1.0, 3.0, 2.0, 5.0];
        let mut weighted = GradientBoostingRegressor::new().n_estimators(5);
        weighted
            .fit_weighted(x.view(), y.view(), array![2.0, 1.0, 1.0, 3.0, 1.0].view())
            .unwrap();
        let xd = array![[0.0], [0.0], [1.0], [2.0], [3.0], [3.0], [3.0], [4.0]];
        let yd = array![0.0, 0.0, 1.0, 3.0, 2.0, 2.0, 2.0, 5.0];
        let mut dup = GradientBoostingRegressor::new()
            .n_estimators(5)
            .min_samples_leaf(1);
        dup.fit(xd.view(), yd.view()).unwrap();
        let diff = weighted.predict(x.view()).unwrap() - dup.predict(x.view()).unwrap();
        assert!(diff.iter().all(|d| d.abs() < 1e-12), "{diff}");
    }

    #[test]
    fn classifier_tracks_true_probabilities() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 4000;
        let x = Array2::from_shape_simple_fn((n, 2), || rng.gen::<f64>() * 4.0 - 2.0);
        let truth: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| sigmoid(2.0 * r[0] - r[1] * r[1] + 0.5))
            .collect();
        let labels = truth.mapv(|p| f64::from(rng.gen::<f64>() < p));
        let mut gb = GradientBoostingClassifier::new()
            .n_estimators(150)
            .learning_rate(0.05);
        gb.fit(x.view(), labels.view()).unwrap();
        let p = gb.predict_proba(x.view()).unwrap();
        assert!(p.iter().all(|&v| v > 0.0 && v < 1.0));
        let mae = (&p - &truth).mapv(f64::abs).mean().unwrap();
        assert!(mae < 0.06, "mean |p̂ − p| = {mae}");
        assert!(gb.fit(x.view(), (&labels * 0.0).view()).is_err()); // one class only
    }

    #[test]
    fn invalid_settings_and_unfitted_use_error() {
        let x = array![[0.0], [1.0]];
        let y = array![0.0, 1.0];
        assert!(GradientBoostingRegressor::new()
            .learning_rate(0.0)
            .fit(x.view(), y.view())
            .is_err());
        assert!(GradientBoostingRegressor::new()
            .subsample(1.5)
            .fit(x.view(), y.view())
            .is_err());
        assert_eq!(
            GradientBoostingRegressor::new()
                .predict(x.view())
                .unwrap_err(),
            DmlError::NotFitted
        );
    }
}

//! Random forests (Breiman, 2001) built from CART regression trees.
//!
//! Every tree is grown on a bootstrap sample, choosing each split among a
//! random subset of features by the largest reduction in (weighted) squared
//! error. Bootstrap draws are represented as integer sample weights, so
//! user-supplied weights ([`Regressor::fit_weighted`]) come for free.
//!
//! For 0/1 labels the squared-error criterion selects exactly the same splits
//! as the Gini criterion (node variance `p(1−p)` is half the Gini impurity),
//! so [`RandomForestClassifier`] reuses the same trees and averages the leaf
//! class-1 frequencies, as scikit-learn does.
//!
//! Trees are independent and each has its own seed derived from the forest
//! seed, so fitting is parallel (with the `parallel` feature) yet results are
//! identical whatever the number of threads.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::{Classifier, Regressor};
use crate::error::{DmlError, Result};

/// Number of features considered at each split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxFeatures {
    /// All `p` features (scikit-learn's regressor default, `max_features=1.0`).
    All,
    /// `⌊√p⌋` (scikit-learn's classifier default).
    Sqrt,
    /// `⌊log₂ p⌋`
    Log2,
    /// `⌊f·p⌋` for `0 < f ≤ 1`.
    Fraction(f64),
    /// A fixed number of features.
    Count(usize),
}

impl MaxFeatures {
    pub(super) fn resolve(self, p: usize) -> usize {
        let k = match self {
            MaxFeatures::All => p,
            MaxFeatures::Sqrt => (p as f64).sqrt() as usize,
            MaxFeatures::Log2 => (p as f64).log2() as usize,
            MaxFeatures::Fraction(f) => (f * p as f64) as usize,
            MaxFeatures::Count(k) => k,
        };
        k.clamp(1, p.max(1))
    }
}

/// Growth limits of one CART tree (shared with gradient boosting).
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeParams {
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: MaxFeatures,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Node {
    Leaf {
        value: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: u32,
        right: u32,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub(super) fn predict_row(&self, row: ArrayView1<'_, f64>) -> f64 {
        match self.nodes[self.leaf_index(row)] {
            Node::Leaf { value } => value,
            Node::Split { .. } => unreachable!("leaf_index returns a leaf"),
        }
    }

    /// Index of the leaf node that `row` falls into.
    pub(crate) fn leaf_index(&self, row: ArrayView1<'_, f64>) -> usize {
        let mut i = 0;
        loop {
            match self.nodes[i] {
                Node::Leaf { .. } => return i,
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => {
                    i = if row[feature] <= threshold {
                        left
                    } else {
                        right
                    } as usize
                }
            }
        }
    }
}

/// Grows one tree. `w` holds the effective weight of every row (user weight
/// times bootstrap count; zero for rows not drawn) and `y` is centred.
///
/// Two lifetimes: `'x` for the caller's feature matrix and `'b` for the
/// per-tree buffers, which live much shorter.
pub(crate) struct TreeBuilder<'x, 'b> {
    pub x: ArrayView2<'x, f64>,
    pub y: &'b [f64],
    pub w: &'b [f64],
    pub params: TreeParams,
    pub n_try: usize,
    pub rng: ChaCha8Rng,
    /// Total squared-error reduction per feature.
    pub importance: Vec<f64>,
}

struct Best {
    feature: usize,
    threshold: f64,
    gain: f64,
}

impl TreeBuilder<'_, '_> {
    pub(crate) fn grow(mut self, rows: Vec<usize>) -> (Tree, Vec<f64>) {
        let mut nodes = vec![Node::Leaf { value: 0.0 }];
        // Explicit stack instead of recursion: deep trees cannot overflow.
        let mut stack = vec![(0usize, rows, 0usize)];
        while let Some((id, rows, depth)) = stack.pop() {
            let (sw, swy, swy2) = rows.iter().fold((0.0, 0.0, 0.0), |(a, b, c), &i| {
                let (wi, yi) = (self.w[i], self.y[i]);
                (a + wi, b + wi * yi, c + wi * yi * yi)
            });
            let value = swy / sw;
            let sse = swy2 - swy * swy / sw;
            let splittable = rows.len() >= self.params.min_samples_split
                && rows.len() >= 2 * self.params.min_samples_leaf
                && self.params.max_depth.map_or(true, |d| depth < d)
                && sse > 1e-12 * swy2.max(f64::MIN_POSITIVE);
            let best = if splittable {
                self.best_split(&rows, sw, swy)
            } else {
                None
            };
            let Some(best) = best else {
                nodes[id] = Node::Leaf { value };
                continue;
            };
            self.importance[best.feature] += best.gain;
            let (left_rows, right_rows): (Vec<usize>, Vec<usize>) = rows
                .into_iter()
                .partition(|&i| self.x[[i, best.feature]] <= best.threshold);
            let left = nodes.len();
            nodes.push(Node::Leaf { value: 0.0 });
            nodes.push(Node::Leaf { value: 0.0 });
            nodes[id] = Node::Split {
                feature: best.feature,
                threshold: best.threshold,
                left: left as u32,
                right: (left + 1) as u32,
            };
            stack.push((left + 1, right_rows, depth + 1));
            stack.push((left, left_rows, depth + 1));
        }
        (Tree { nodes }, self.importance)
    }

    /// Best split among `n_try` random features: maximise the reduction in
    /// weighted SSE, `S_L²/W_L + S_R²/W_R − S²/W`.
    fn best_split(&mut self, rows: &[usize], sw: f64, swy: f64) -> Option<Best> {
        let p = self.x.ncols();
        let parent = swy * swy / sw;
        let min_leaf = self.params.min_samples_leaf;
        let mut best: Option<Best> = None;
        let mut sorted: Vec<(f64, usize)> = Vec::with_capacity(rows.len());
        for feature in sample(&mut self.rng, p, self.n_try).into_iter() {
            sorted.clear();
            sorted.extend(rows.iter().map(|&i| (self.x[[i, feature]], i)));
            sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            if sorted[0].0 == sorted[sorted.len() - 1].0 {
                continue; // constant in this node
            }
            let (mut wl, mut sl) = (0.0, 0.0);
            for k in 0..sorted.len() - 1 {
                let i = sorted[k].1;
                wl += self.w[i];
                sl += self.w[i] * self.y[i];
                let (v, v_next) = (sorted[k].0, sorted[k + 1].0);
                let n_left = k + 1;
                if v == v_next || n_left < min_leaf || sorted.len() - n_left < min_leaf {
                    continue;
                }
                let (wr, sr) = (sw - wl, swy - sl);
                if wl <= 0.0 || wr <= 0.0 {
                    continue;
                }
                let gain = sl * sl / wl + sr * sr / wr - parent;
                if best.as_ref().map_or(true, |b| gain > b.gain) {
                    // Midpoint, unless rounding makes it equal the upper value.
                    let mid = v + (v_next - v) / 2.0;
                    let threshold = if mid < v_next { mid } else { v };
                    best = Some(Best {
                        feature,
                        threshold,
                        gain,
                    });
                }
            }
        }
        best.filter(|b| b.gain > 1e-12 * parent.abs().max(f64::MIN_POSITIVE))
    }
}

/// The shared forest engine behind the regressor and the classifier.
#[derive(Debug, Clone)]
struct Forest {
    n_estimators: usize,
    params: TreeParams,
    bootstrap: bool,
    max_samples: Option<f64>,
    seed: u64,
    trees: Vec<Tree>,
    y_offset: f64,
    n_features: usize,
    importances: Option<Array1<f64>>,
}

impl Forest {
    fn new(max_features: MaxFeatures) -> Self {
        Self {
            n_estimators: 100,
            params: TreeParams {
                max_depth: None,
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features,
            },
            bootstrap: true,
            max_samples: None,
            seed: 0,
            trees: Vec::new(),
            y_offset: 0.0,
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
        if let Some(m) = self.max_samples {
            if !(m > 0.0 && m <= 1.0) {
                return Err(DmlError::InvalidArgument(
                    "max_samples must be in (0, 1]".into(),
                ));
            }
        }
        Ok(())
    }

    fn fit(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        self.validate()?;
        let (n, p) = x.dim();
        if y.len() != n {
            return Err(DmlError::DimensionMismatch("forest fit x/y rows".into()));
        }
        if n == 0 {
            return Err(DmlError::InvalidArgument("empty training set".into()));
        }
        let weights: Vec<f64> = match w {
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
        // Centre y so the split arithmetic does not lose precision to a large mean.
        let total_w: f64 = weights.iter().sum();
        let offset = y.iter().zip(&weights).map(|(&a, &b)| a * b).sum::<f64>() / total_w;
        let yc: Vec<f64> = y.iter().map(|&v| v - offset).collect();
        let n_draw = self
            .max_samples
            .map_or(n, |m| ((m * n as f64).round() as usize).max(1));
        let n_try = self.params.max_features.resolve(p);

        let grow_tree = |t: usize| -> (Tree, Vec<f64>) {
            let tree_seed = self
                .seed
                .wrapping_add((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut rng = ChaCha8Rng::seed_from_u64(tree_seed);
            let mut eff = vec![0.0; n];
            if self.bootstrap {
                for _ in 0..n_draw {
                    eff[rng.gen_range(0..n)] += 1.0;
                }
                for (e, &wi) in eff.iter_mut().zip(&weights) {
                    *e *= wi;
                }
            } else {
                eff.clone_from(&weights);
            }
            let rows: Vec<usize> = (0..n).filter(|&i| eff[i] > 0.0).collect();
            let rows = if rows.is_empty() {
                (0..n).collect()
            } else {
                rows
            };
            TreeBuilder {
                x,
                y: &yc,
                w: &eff,
                params: self.params,
                n_try,
                rng,
                importance: vec![0.0; p],
            }
            .grow(rows)
        };

        #[cfg(feature = "parallel")]
        let grown: Vec<(Tree, Vec<f64>)> = {
            use rayon::prelude::*;
            (0..self.n_estimators)
                .into_par_iter()
                .map(grow_tree)
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let grown: Vec<(Tree, Vec<f64>)> = (0..self.n_estimators).map(grow_tree).collect();

        // Mean decrease in impurity, each tree normalised to sum to one.
        let mut importances = Array1::<f64>::zeros(p);
        let mut counted = 0.0;
        for (_, imp) in &grown {
            let total: f64 = imp.iter().sum();
            if total > 0.0 {
                importances += &(Array1::from(imp.clone()) / total);
                counted += 1.0;
            }
        }
        if counted > 0.0 {
            importances /= counted;
        }
        self.trees = grown.into_iter().map(|(t, _)| t).collect();
        self.y_offset = offset;
        self.n_features = p;
        self.importances = Some(importances);
        Ok(())
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(DmlError::NotFitted);
        }
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "x has {} columns, the forest was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        let per_row = |row: ArrayView1<'_, f64>| -> f64 {
            self.trees.iter().map(|t| t.predict_row(row)).sum::<f64>() / self.trees.len() as f64
                + self.y_offset
        };
        Ok(x.rows().into_iter().map(per_row).collect())
    }
}

/// Builder methods shared by the regressor and the classifier.
macro_rules! forest_builders {
    () => {
        /// Number of trees (default 100).
        pub fn n_estimators(mut self, n: usize) -> Self {
            self.forest.n_estimators = n;
            self
        }

        /// Maximum tree depth (default: unlimited).
        pub fn max_depth(mut self, depth: Option<usize>) -> Self {
            self.forest.params.max_depth = depth;
            self
        }

        /// Minimum number of rows required to split a node (default 2).
        pub fn min_samples_split(mut self, n: usize) -> Self {
            self.forest.params.min_samples_split = n;
            self
        }

        /// Minimum number of distinct rows in each leaf (default 1). Values
        /// around 5–20 give smoother nuisance estimates for DML.
        pub fn min_samples_leaf(mut self, n: usize) -> Self {
            self.forest.params.min_samples_leaf = n;
            self
        }

        /// Features considered per split.
        pub fn max_features(mut self, m: MaxFeatures) -> Self {
            self.forest.params.max_features = m;
            self
        }

        /// Grow each tree on a bootstrap sample (default `true`).
        pub fn bootstrap(mut self, bootstrap: bool) -> Self {
            self.forest.bootstrap = bootstrap;
            self
        }

        /// Bootstrap sample size as a fraction of `n` (default 1.0).
        pub fn max_samples(mut self, fraction: Option<f64>) -> Self {
            self.forest.max_samples = fraction;
            self
        }

        /// Seed for bootstrap draws and feature sampling (default 0).
        pub fn seed(mut self, seed: u64) -> Self {
            self.forest.seed = seed;
            self
        }

        /// Mean decrease in impurity per feature (sums to one), after fitting.
        pub fn feature_importances(&self) -> Option<&Array1<f64>> {
            self.forest.importances.as_ref()
        }
    };
}

/// Random forest regressor (squared-error CART trees on bootstrap samples).
///
/// Defaults match scikit-learn's `RandomForestRegressor`: 100 trees, all
/// features per split, unlimited depth, `min_samples_leaf = 1`.
///
/// ```
/// use dml_core::{MaxFeatures, RandomForestRegressor};
/// let rf = RandomForestRegressor::new()
///     .n_estimators(200)
///     .min_samples_leaf(5)
///     .max_features(MaxFeatures::Fraction(0.5))
///     .seed(42);
/// # let _ = rf;
/// ```
#[derive(Debug, Clone)]
pub struct RandomForestRegressor {
    forest: Forest,
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomForestRegressor {
    pub fn new() -> Self {
        Self {
            forest: Forest::new(MaxFeatures::All),
        }
    }

    forest_builders!();
}

impl Regressor for RandomForestRegressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.forest.fit(x, y, None)
    }

    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.forest.fit(x, y, Some(w))
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.forest.predict(x)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self {
            forest: self.forest.unfitted(),
        })
    }
}

/// Random forest classifier for 0/1 labels; `predict_proba` averages the
/// class-1 frequency of the leaves.
///
/// Defaults match scikit-learn's `RandomForestClassifier`: 100 trees,
/// `√p` features per split, unlimited depth, `min_samples_leaf = 1`.
#[derive(Debug, Clone)]
pub struct RandomForestClassifier {
    forest: Forest,
}

impl Default for RandomForestClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomForestClassifier {
    pub fn new() -> Self {
        Self {
            forest: Forest::new(MaxFeatures::Sqrt),
        }
    }

    forest_builders!();
}

impl Classifier for RandomForestClassifier {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        if !y.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(DmlError::InvalidArgument(
                "RandomForestClassifier labels must be 0 or 1".into(),
            ));
        }
        self.forest.fit(x, y, None)
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        // Leaf means of 0/1 labels are already in [0, 1]; clamp rounding noise.
        Ok(self.forest.predict(x)?.mapv(|p| p.clamp(0.0, 1.0)))
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::new(Self {
            forest: self.forest.unfitted(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn nonlinear(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x = Array2::from_shape_simple_fn((n, 3), || rng.gen::<f64>() * 4.0 - 2.0);
        let y = x
            .rows()
            .into_iter()
            .map(|r| r[0].sin() * 2.0 + (r[1] > 0.0) as u8 as f64)
            .collect();
        (x, y)
    }

    #[test]
    fn single_tree_without_bootstrap_fits_a_step_exactly() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![1.0, 1.0, 1.0, 7.0, 7.0, 7.0];
        let mut rf = RandomForestRegressor::new()
            .n_estimators(1)
            .bootstrap(false);
        rf.fit(x.view(), y.view()).unwrap();
        let pred = rf
            .predict(array![[2.4], [2.6], [-10.0], [10.0]].view())
            .unwrap();
        assert_eq!(pred, array![1.0, 7.0, 1.0, 7.0]);
        // The only split is at the midpoint 2.5 between the two groups.
        assert!(
            matches!(rf.forest.trees[0].nodes[0], Node::Split { threshold, .. } if threshold == 2.5)
        );
    }

    #[test]
    fn forest_learns_a_nonlinear_function() {
        let (x, y) = nonlinear(2000, 1);
        let (x_test, y_test) = nonlinear(500, 2);
        let mut rf = RandomForestRegressor::new()
            .n_estimators(100)
            .min_samples_leaf(3)
            .seed(7);
        rf.fit(x.view(), y.view()).unwrap();
        let resid = &y_test - &rf.predict(x_test.view()).unwrap();
        let r2 = 1.0
            - resid.mapv(|e| e * e).sum()
                / y_test.mapv(|v| (v - y_test.mean().unwrap()).powi(2)).sum();
        assert!(r2 > 0.9, "out-of-sample R² = {r2}");
        // x₂ is pure noise: it should get the least importance.
        let imp = rf.feature_importances().unwrap();
        assert!((imp.sum() - 1.0).abs() < 1e-9);
        assert!(imp[2] < imp[0] && imp[2] < imp[1], "{imp}");
    }

    #[test]
    fn same_seed_same_forest_and_seeds_matter() {
        let (x, y) = nonlinear(300, 3);
        let fit = |seed| {
            let mut rf = RandomForestRegressor::new()
                .n_estimators(20)
                .max_features(MaxFeatures::Count(2))
                .seed(seed);
            rf.fit(x.view(), y.view()).unwrap();
            rf.predict(x.view()).unwrap()
        };
        assert_eq!(fit(1), fit(1));
        assert_ne!(fit(1), fit(2));
    }

    #[test]
    fn weights_equal_row_duplication_without_bootstrap() {
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![0.0, 1.0, 5.0, 6.0];
        let mut weighted = RandomForestRegressor::new()
            .n_estimators(1)
            .bootstrap(false)
            .max_depth(Some(1));
        weighted
            .fit_weighted(x.view(), y.view(), array![3.0, 1.0, 1.0, 1.0].view())
            .unwrap();
        let mut dup = RandomForestRegressor::new()
            .n_estimators(1)
            .bootstrap(false)
            .max_depth(Some(1));
        let xd = array![[0.0], [0.0], [0.0], [1.0], [2.0], [3.0]];
        let yd = array![0.0, 0.0, 0.0, 1.0, 5.0, 6.0];
        dup.fit(xd.view(), yd.view()).unwrap();
        let grid = array![[0.0], [1.5], [3.0]];
        let diff = weighted.predict(grid.view()).unwrap() - dup.predict(grid.view()).unwrap();
        assert!(diff.iter().all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn min_samples_leaf_and_depth_are_respected() {
        let (x, y) = nonlinear(200, 4);
        let mut stump = RandomForestRegressor::new()
            .n_estimators(3)
            .max_depth(Some(1));
        stump.fit(x.view(), y.view()).unwrap();
        assert!(stump.forest.trees.iter().all(|t| t.nodes.len() <= 3));
        let mut big_leaves = RandomForestRegressor::new()
            .n_estimators(1)
            .bootstrap(false)
            .min_samples_leaf(50);
        big_leaves.fit(x.view(), y.view()).unwrap();
        assert!(big_leaves.forest.trees[0].nodes.len() <= 7); // at most 4 leaves of ≥ 50 rows
    }

    #[test]
    fn classifier_probabilities_track_the_truth() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 3000;
        let x = Array2::from_shape_simple_fn((n, 2), || rng.gen::<f64>() * 4.0 - 2.0);
        let truth: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| 1.0 / (1.0 + (-2.0 * r[0]).exp()))
            .collect();
        let labels = truth.mapv(|p| f64::from(rng.gen::<f64>() < p));
        let mut rf = RandomForestClassifier::new().min_samples_leaf(20).seed(1);
        rf.fit(x.view(), labels.view()).unwrap();
        let p = rf.predict_proba(x.view()).unwrap();
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
        let mae = (&p - &truth).mapv(f64::abs).mean().unwrap();
        assert!(mae < 0.08, "mean |p̂ − p| = {mae}");
        assert!(rf.fit(x.view(), (&labels * 2.0).view()).is_err());
    }

    #[test]
    fn invalid_settings_and_unfitted_use_error() {
        let x = array![[0.0], [1.0]];
        let y = array![0.0, 1.0];
        assert!(RandomForestRegressor::new()
            .n_estimators(0)
            .fit(x.view(), y.view())
            .is_err());
        assert!(RandomForestRegressor::new()
            .max_features(MaxFeatures::Fraction(1.5))
            .fit(x.view(), y.view())
            .is_err());
        assert_eq!(
            RandomForestRegressor::new().predict(x.view()).unwrap_err(),
            DmlError::NotFitted
        );
        assert_eq!(MaxFeatures::Sqrt.resolve(10), 3);
        assert_eq!(MaxFeatures::Log2.resolve(1), 1);
        assert_eq!(MaxFeatures::Count(50).resolve(4), 4);
    }
}

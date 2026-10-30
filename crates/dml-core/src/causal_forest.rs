//! Causal forests: generalized random forests for heterogeneous treatment
//! effects (Athey, Tibshirani and Wager, 2019; Wager and Athey, 2018), as R's
//! `grf::causal_forest`.
//!
//! The model is `Y = τ(X)·D + m(X) + ε`. After orthogonalisation (cross-fitted
//! `ŷ(X) = E[Y | X]` and `d̂(X) = E[D | X]`, any learners), a forest of *honest*
//! trees is grown on the residuals `Ỹ = Y − ŷ`, `D̃ = D − d̂`:
//!
//! - every tree uses a subsample, half of which chooses the splits and the
//!   other half estimates the leaves (honesty);
//! - splits maximise the heterogeneity of the treatment effect, through a
//!   regression split on the pseudo-outcomes
//!   `ρᵢ = (D̃ᵢ − D̄)(Ỹᵢ − Ȳ − θ̂_P (D̃ᵢ − D̄))` of the parent node (the gradient
//!   approximation of grf), subject to the children keeping a share `alpha` of
//!   the parent's treatment variation;
//! - `τ̂(x)` is a residual-on-residual regression weighted by how often each
//!   training point shares a leaf with `x`.
//!
//! Trees are grown in little bags (groups of `ci_group_size` trees on one
//! half-sample), which gives variance estimates for `τ̂(x)` (the bootstrap of
//! little bags). For a binary treatment, [`CausalForestFit::ate`] combines the
//! out-of-bag `τ̂(Xᵢ)` into doubly robust scores for the average effect, best
//! linear predictors and group effects.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::blp::Blp;
use crate::crossfit::{cross_fit_predict, cross_fit_predict_proba, SampleSplitting};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::fit::DmlFit;
use crate::model::TreatmentLearner;
use crate::stats::norm_cdf;

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/// Settings of a causal forest (grf's defaults).
#[derive(Debug, Clone)]
pub struct ForestSettings {
    pub num_trees: usize,
    pub sample_fraction: f64,
    /// Variables tried per split; `None`: `min(⌈√p + 20⌉, p)`.
    pub mtry: Option<usize>,
    pub min_node_size: usize,
    pub honesty: bool,
    pub honesty_fraction: f64,
    pub prune_empty_leaves: bool,
    pub alpha: f64,
    pub imbalance_penalty: f64,
    pub stabilize_splits: bool,
    pub ci_group_size: usize,
    pub seed: u64,
}

impl Default for ForestSettings {
    fn default() -> Self {
        Self {
            num_trees: 2000,
            sample_fraction: 0.5,
            mtry: None,
            min_node_size: 5,
            honesty: true,
            honesty_fraction: 0.5,
            prune_empty_leaves: true,
            alpha: 0.05,
            imbalance_penalty: 0.0,
            stabilize_splits: true,
            ci_group_size: 2,
            seed: 0,
        }
    }
}

impl ForestSettings {
    fn validate(&self, n: usize) -> Result<()> {
        let bad = |m: &str| Err(DmlError::InvalidArgument(m.into()));
        if self.ci_group_size == 0 || self.num_trees < self.ci_group_size {
            return bad("num_trees must be at least ci_group_size (and ci_group_size at least 1)");
        }
        if self.sample_fraction.is_nan()
            || self.sample_fraction <= 0.0
            || self.sample_fraction > 1.0
        {
            return bad("sample_fraction must be in (0, 1]");
        }
        if self.ci_group_size > 1 && self.sample_fraction > 0.5 {
            return bad("sample_fraction must be at most 0.5 when ci_group_size > 1");
        }
        if self.honesty_fraction.is_nan()
            || self.honesty_fraction <= 0.0
            || self.honesty_fraction >= 1.0
        {
            return bad("honesty_fraction must be in (0, 1)");
        }
        if self.alpha.is_nan() || !(0.0..0.25).contains(&self.alpha) {
            return bad("alpha must be in [0, 0.25)");
        }
        if self.imbalance_penalty.is_nan() || self.imbalance_penalty < 0.0 {
            return bad("imbalance_penalty must be non-negative");
        }
        if self.mtry == Some(0) {
            return bad("mtry must be at least 1");
        }
        if n < 4 {
            return bad("a causal forest needs at least 4 observations");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

/// Sums over a leaf's estimation sample.
#[derive(Debug, Clone, Copy, Default)]
struct LeafStats {
    n: f64,
    sy: f64,
    sw: f64,
    syw: f64,
    sww: f64,
}

impl LeafStats {
    fn add(&mut self, y: f64, w: f64) {
        self.n += 1.0;
        self.sy += y;
        self.sw += w;
        self.syw += y * w;
        self.sww += w * w;
    }

    fn merge(a: Self, b: Self) -> Self {
        Self {
            n: a.n + b.n,
            sy: a.sy + b.sy,
            sw: a.sw + b.sw,
            syw: a.syw + b.syw,
            sww: a.sww + b.sww,
        }
    }

    /// Leaf means `(ȳ, w̄, mean yw, mean ww)`.
    fn means(&self) -> [f64; 4] {
        [
            self.sy / self.n,
            self.sw / self.n,
            self.syw / self.n,
            self.sww / self.n,
        ]
    }
}

#[derive(Debug, Clone)]
struct Node {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
    is_leaf: bool,
    depth: usize,
    stats: LeafStats,
}

#[derive(Debug, Clone)]
struct CfTree {
    nodes: Vec<Node>,
    /// Bitset over the training rows: in this tree's subsample.
    inbag: Vec<u64>,
}

impl CfTree {
    fn leaf(&self, row: ArrayView1<'_, f64>) -> &LeafStats {
        let mut k = 0;
        loop {
            let node = &self.nodes[k];
            if node.is_leaf {
                return &node.stats;
            }
            k = if row[node.feature] <= node.threshold {
                node.left
            } else {
                node.right
            };
        }
    }

    fn contains(&self, i: usize) -> bool {
        self.inbag[i / 64] >> (i % 64) & 1 == 1
    }
}

/// Training data of the forest: covariates and residuals.
struct Train<'a> {
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    w: ArrayView1<'a, f64>,
}

/// Grow one honest tree on `subsample` (training rows).
fn grow_tree(
    t: &Train<'_>,
    s: &ForestSettings,
    mtry: usize,
    mut subsample: Vec<usize>,
    rng: &mut ChaCha8Rng,
) -> CfTree {
    let n = t.x.nrows();
    let mut inbag = vec![0u64; n.div_ceil(64)];
    for &i in &subsample {
        inbag[i / 64] |= 1 << (i % 64);
    }
    subsample.shuffle(rng);
    let (split_rows, est_rows): (Vec<usize>, Vec<usize>) = if s.honesty {
        let k = ((subsample.len() as f64 * s.honesty_fraction).floor() as usize)
            .clamp(1, subsample.len() - 1);
        (subsample[..k].to_vec(), subsample[k..].to_vec())
    } else {
        (subsample.clone(), subsample)
    };

    let p = t.x.ncols();
    let mut nodes: Vec<Node> = vec![Node {
        feature: 0,
        threshold: 0.0,
        left: 0,
        right: 0,
        is_leaf: true,
        depth: 0,
        stats: LeafStats::default(),
    }];
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, split_rows)];
    let mut features: Vec<usize> = (0..p).collect();
    while let Some((id, rows)) = stack.pop() {
        let Some((feature, threshold)) = best_split(t, s, &rows, mtry, &mut features, rng) else {
            continue;
        };
        let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
            rows.iter().partition(|&&i| t.x[[i, feature]] <= threshold);
        let depth = nodes[id].depth + 1;
        let (l, r) = (nodes.len(), nodes.len() + 1);
        for _ in 0..2 {
            nodes.push(Node {
                feature: 0,
                threshold: 0.0,
                left: 0,
                right: 0,
                is_leaf: true,
                depth,
                stats: LeafStats::default(),
            });
        }
        let node = &mut nodes[id];
        node.feature = feature;
        node.threshold = threshold;
        node.left = l;
        node.right = r;
        node.is_leaf = false;
        stack.push((l, left_rows));
        stack.push((r, right_rows));
    }

    // Honest leaves: the estimation half, dropped down the tree.
    let mut tree = CfTree { nodes, inbag };
    for &i in &est_rows {
        let row = t.x.row(i);
        let mut k = 0;
        while !tree.nodes[k].is_leaf {
            let node = &tree.nodes[k];
            k = if row[node.feature] <= node.threshold {
                node.left
            } else {
                node.right
            };
        }
        tree.nodes[k].stats.add(t.y[i], t.w[i]);
    }
    if s.prune_empty_leaves {
        prune(&mut tree.nodes, 0);
    } else {
        fill_inner(&mut tree.nodes, 0);
    }
    tree
}

/// Collapse inner nodes with an empty child into leaves (grf's
/// `honesty.prune.leaves`); returns the node's sums.
fn prune(nodes: &mut [Node], id: usize) -> LeafStats {
    if nodes[id].is_leaf {
        return nodes[id].stats;
    }
    let (l, r) = (nodes[id].left, nodes[id].right);
    let (sl, sr) = (prune(nodes, l), prune(nodes, r));
    let merged = LeafStats::merge(sl, sr);
    nodes[id].stats = merged;
    if sl.n == 0.0 || sr.n == 0.0 {
        nodes[id].is_leaf = true;
    }
    merged
}

/// Sums of inner nodes (without pruning, empty leaves fall back to their parent).
fn fill_inner(nodes: &mut [Node], id: usize) -> LeafStats {
    if nodes[id].is_leaf {
        return nodes[id].stats;
    }
    let (l, r) = (nodes[id].left, nodes[id].right);
    let merged = LeafStats::merge(fill_inner(nodes, l), fill_inner(nodes, r));
    nodes[id].stats = merged;
    for c in [l, r] {
        if nodes[c].is_leaf && nodes[c].stats.n == 0.0 {
            nodes[c].stats = merged;
        }
    }
    merged
}

/// The best causal split of a node, or `None` for a leaf.
fn best_split(
    t: &Train<'_>,
    s: &ForestSettings,
    rows: &[usize],
    mtry: usize,
    features: &mut [usize],
    rng: &mut ChaCha8Rng,
) -> Option<(usize, f64)> {
    let n = rows.len();
    if n <= s.min_node_size || n < 2 {
        return None;
    }
    // Relabel: pseudo-outcomes of the parent's local treatment effect.
    let nf = n as f64;
    let (mut sy, mut sw) = (0.0, 0.0);
    for &i in rows {
        sy += t.y[i];
        sw += t.w[i];
    }
    let (ybar, wbar) = (sy / nf, sw / nf);
    let (mut num, mut den) = (0.0, 0.0);
    for &i in rows {
        let dw = t.w[i] - wbar;
        num += dw * (t.y[i] - ybar);
        den += dw * dw;
    }
    if den <= 1e-12 * nf.max(1.0) {
        return None;
    }
    let theta = num / den;
    // Treatment variation of the node (for the stabilised imbalance limit).
    let size_node = den;
    let min_count = ((s.alpha * nf).ceil() as usize).max(1);
    let min_size = s.alpha * size_node;

    let p = features.len();
    let k = mtry.min(p);
    features.shuffle(rng);
    let mut best: Option<(f64, usize, f64)> = None;
    let mut vals: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for &f in &features[..k] {
        vals.clear();
        for &i in rows {
            let dw = t.w[i] - wbar;
            let rho = dw * ((t.y[i] - ybar) - theta * dw);
            vals.push((t.x[[i, f]], rho, t.w[i]));
        }
        vals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut nl, mut rl, mut wl, mut wwl) = (0usize, 0.0, 0.0, 0.0);
        let (rho_total, w_total, ww_total): (f64, f64, f64) =
            vals.iter().fold((0.0, 0.0, 0.0), |(a, b, c), v| {
                (a + v.1, b + v.2, c + v.2 * v.2)
            });
        for j in 0..n - 1 {
            let (xv, rho, w) = vals[j];
            nl += 1;
            rl += rho;
            wl += w;
            wwl += w * w;
            if vals[j + 1].0 <= xv {
                continue; // not a boundary between distinct values
            }
            let nr = n - nl;
            if nl < min_count || nr < min_count {
                continue;
            }
            let (nlf, nrf) = (nl as f64, nr as f64);
            let size_l = wwl - wl * wl / nlf;
            let wr = w_total - wl;
            let size_r = (ww_total - wwl) - wr * wr / nrf;
            if s.stabilize_splits && (size_l < min_size || size_r < min_size) {
                continue;
            }
            let rr = rho_total - rl;
            let mut decrease = rl * rl / nlf + rr * rr / nrf;
            if s.imbalance_penalty > 0.0 {
                if size_l <= 0.0 || size_r <= 0.0 {
                    continue;
                }
                decrease -= s.imbalance_penalty * (1.0 / size_l + 1.0 / size_r);
            }
            if best.map_or(true, |b| decrease > b.0) {
                best = Some((decrease, f, 0.5 * (xv + vals[j + 1].0)));
            }
        }
    }
    best.map(|(_, f, thr)| (f, thr))
}

// ---------------------------------------------------------------------------
// Forest
// ---------------------------------------------------------------------------

/// A causal forest grown on residuals; see the [module docs](self).
#[derive(Debug, Clone)]
pub struct CausalForest {
    settings: ForestSettings,
    trees: Vec<CfTree>,
    n_features: usize,
    n_train: usize,
    importance: Array1<f64>,
}

/// grf's `ObjectiveBayesDebiaser`: posterior mean of a variance under a flat
/// prior on positive values.
fn debias(var_between: f64, group_noise: f64, num_good_groups: f64) -> f64 {
    let initial = var_between - group_noise;
    let initial_se = var_between.max(group_noise) * (2.0 / num_good_groups).sqrt();
    if initial_se.is_nan() || initial_se <= 0.0 {
        return initial.max(0.0);
    }
    let ratio = initial / initial_se;
    let density = (-0.5 * ratio * ratio).exp() / (2.0 * std::f64::consts::PI).sqrt();
    initial + initial_se * density / norm_cdf(ratio).max(1e-300)
}

impl CausalForest {
    /// Grow a forest of `settings.num_trees` honest trees on covariates `x` and
    /// residuals `y` (outcome) and `w` (treatment).
    pub fn fit<'a>(
        x: ArrayView2<'a, f64>,
        y: ArrayView1<'a, f64>,
        w: ArrayView1<'a, f64>,
        settings: ForestSettings,
    ) -> Result<Self> {
        let n = x.nrows();
        if y.len() != n || w.len() != n {
            return Err(DmlError::DimensionMismatch(
                "causal forest x / y / w rows".into(),
            ));
        }
        if x.iter()
            .chain(y.iter())
            .chain(w.iter())
            .any(|v| !v.is_finite())
        {
            return Err(DmlError::InvalidArgument(
                "the causal forest needs finite data".into(),
            ));
        }
        settings.validate(n)?;
        let p = x.ncols();
        let mtry = settings
            .mtry
            .unwrap_or_else(|| ((p as f64).sqrt() + 20.0).ceil() as usize)
            .min(p)
            .max(1);
        let l = settings.ci_group_size;
        let n_groups = settings.num_trees / l;
        let t = Train { x, y, w };
        let s = &settings;
        let group = |g: usize| -> Vec<CfTree> {
            let mut rng = ChaCha8Rng::seed_from_u64(
                s.seed
                    .wrapping_add((g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let pool: Vec<usize> = if l > 1 {
                let mut all: Vec<usize> = (0..n).collect();
                all.shuffle(&mut rng);
                all.truncate(n / 2);
                all
            } else {
                (0..n).collect()
            };
            let size = ((s.sample_fraction * n as f64) as usize).clamp(2, pool.len());
            (0..l)
                .map(|_| {
                    let mut sub = pool.clone();
                    sub.shuffle(&mut rng);
                    sub.truncate(size);
                    grow_tree(&t, s, mtry, sub, &mut rng)
                })
                .collect()
        };
        #[cfg(feature = "parallel")]
        let groups: Vec<Vec<CfTree>> = {
            use rayon::prelude::*;
            (0..n_groups).into_par_iter().map(group).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let groups: Vec<Vec<CfTree>> = (0..n_groups).map(group).collect();
        let trees: Vec<CfTree> = groups.into_iter().flatten().collect();
        let importance = Self::split_importance(&trees, p);
        Ok(Self {
            settings,
            trees,
            n_features: p,
            n_train: n,
            importance,
        })
    }

    /// grf's `variable_importance`: split frequencies by depth (up to depth 4),
    /// weighted by `depth⁻²`.
    fn split_importance(trees: &[CfTree], p: usize) -> Array1<f64> {
        const MAX_DEPTH: usize = 4;
        let mut freq = Array2::<f64>::zeros((MAX_DEPTH, p));
        for tree in trees {
            for node in &tree.nodes {
                if !node.is_leaf && node.depth < MAX_DEPTH {
                    freq[[node.depth, node.feature]] += 1.0;
                }
            }
        }
        let mut out = Array1::zeros(p);
        let mut total_weight = 0.0;
        for d in 0..MAX_DEPTH {
            let row_sum = freq.row(d).sum().max(1.0);
            let weight = ((d + 1) as f64).powi(-2);
            total_weight += weight;
            out.scaled_add(weight / row_sum, &freq.row(d));
        }
        out / total_weight
    }

    pub fn settings(&self) -> &ForestSettings {
        &self.settings
    }

    /// Number of trees grown.
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }

    /// Split-frequency variable importance (sums to one when there are splits).
    pub fn variable_importance(&self) -> &Array1<f64> {
        &self.importance
    }

    /// Estimate and variance at one point, from the trees `valid` accepts.
    fn estimate(&self, row: ArrayView1<'_, f64>, valid: impl Fn(&CfTree) -> bool) -> (f64, f64) {
        let l = self.settings.ci_group_size;
        let mut means: Vec<Option<[f64; 4]>> = Vec::with_capacity(self.trees.len());
        let mut total = [0.0; 4];
        let mut count = 0.0;
        for tree in &self.trees {
            if valid(tree) {
                let m = tree.leaf(row).means();
                for k in 0..4 {
                    total[k] += m[k];
                }
                count += 1.0;
                means.push(Some(m));
            } else {
                means.push(None);
            }
        }
        if count == 0.0 {
            return (f64::NAN, f64::NAN);
        }
        let [my, mw, myw, mww] = total.map(|v| v / count);
        let den = mww - mw * mw;
        if den.is_nan() || den <= 0.0 {
            return (f64::NAN, f64::NAN);
        }
        let theta = (myw - my * mw) / den;
        if l < 2 {
            return (theta, f64::NAN);
        }
        // Delta method: the influence of each tree's leaf means on θ̂.
        let psi =
            |m: &[f64; 4]| (m[2] - mw * m[0] + (2.0 * theta * mw - my) * m[1] - theta * m[3]) / den;
        let mut group_psi: Vec<(f64, Vec<f64>)> = Vec::new();
        for chunk in means.chunks(l) {
            if chunk.iter().all(Option::is_some) {
                let values: Vec<f64> = chunk
                    .iter()
                    .map(|m| psi(m.as_ref().expect("valid")))
                    .collect();
                group_psi.push((values.iter().sum::<f64>() / l as f64, values));
            }
        }
        let g = group_psi.len() as f64;
        if g < 1.0 {
            return (theta, f64::NAN);
        }
        let center = group_psi.iter().map(|(m, _)| m).sum::<f64>() / g;
        let var_between = group_psi
            .iter()
            .map(|(m, _)| (m - center).powi(2))
            .sum::<f64>()
            / g;
        let var_total = group_psi
            .iter()
            .flat_map(|(_, v)| v.iter())
            .map(|v| (v - center).powi(2))
            .sum::<f64>()
            / (g * l as f64);
        let group_noise = (var_total - var_between) / (l as f64 - 1.0);
        (theta, debias(var_between, group_noise, g))
    }

    fn check_features(&self, x: ArrayView2<'_, f64>) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "fitted on {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        Ok(())
    }

    /// `τ̂(x)` and its variance for the rows of `x` (all trees). Variances are
    /// `NaN` with `ci_group_size = 1`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        self.check_features(x)?;
        let one = |i: usize| self.estimate(x.row(i), |_| true);
        Ok(collect_pairs(x.nrows(), one))
    }

    /// Out-of-bag `τ̂(Xᵢ)` and variance for the training rows `x` (in training
    /// order): each estimate uses only the trees whose subsample left row `i`
    /// out.
    pub fn predict_oob(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        self.check_features(x)?;
        if x.nrows() != self.n_train {
            return Err(DmlError::DimensionMismatch(
                "out-of-bag predictions need the training rows".into(),
            ));
        }
        let one = |i: usize| self.estimate(x.row(i), |t| !t.contains(i));
        Ok(collect_pairs(x.nrows(), one))
    }
}

fn collect_pairs(
    n: usize,
    one: impl Fn(usize) -> (f64, f64) + Sync + Send,
) -> (Array1<f64>, Array1<f64>) {
    #[cfg(feature = "parallel")]
    let pairs: Vec<(f64, f64)> = {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(one).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let pairs: Vec<(f64, f64)> = (0..n).map(one).collect();
    let (a, b): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    (Array1::from(a), Array1::from(b))
}

// ---------------------------------------------------------------------------
// Orthogonalisation and inference
// ---------------------------------------------------------------------------

/// A causal forest on cross-fitted residuals: `ml_y` learns `E[Y | X]`,
/// `ml_d` learns `E[D | X]` (a classifier for a binary treatment), both
/// cross-fitted on `n_folds` folds, then the forest is grown on the residuals.
pub struct CausalForestDml {
    ml_y: Box<dyn crate::learner::Regressor>,
    ml_d: TreatmentLearner,
    forest: ForestSettings,
    splitting: SampleSplitting,
    clipping_threshold: f64,
}

impl CausalForestDml {
    pub fn new(ml_y: impl crate::learner::Regressor + 'static, ml_d: TreatmentLearner) -> Self {
        Self::from_boxed(Box::new(ml_y), ml_d)
    }

    pub fn from_boxed(ml_y: Box<dyn crate::learner::Regressor>, ml_d: TreatmentLearner) -> Self {
        Self {
            ml_y,
            ml_d,
            forest: ForestSettings::default(),
            splitting: SampleSplitting::default(),
            clipping_threshold: 1e-2,
        }
    }

    /// Forest settings (default [`ForestSettings::default`], grf's defaults).
    pub fn forest(mut self, settings: ForestSettings) -> Self {
        self.forest = settings;
        self
    }

    /// Folds for the nuisance models (default 5).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Seed of the nuisance folds (the forest has its own seed).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Propensities are clipped to `[t, 1 − t]` in the doubly robust scores
    /// of a binary treatment (default 0.01).
    pub fn clipping_threshold(mut self, t: f64) -> Self {
        self.clipping_threshold = t;
        self
    }

    pub fn fit(&self, data: &DmlData) -> Result<CausalForestFit> {
        if data.n_treat() != 1 {
            return Err(DmlError::InvalidArgument(
                "a causal forest needs exactly one treatment".into(),
            ));
        }
        if !(self.clipping_threshold > 0.0 && self.clipping_threshold < 0.5) {
            return Err(DmlError::InvalidArgument(
                "clipping_threshold must be in (0, 0.5)".into(),
            ));
        }
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let folds = self.splitting.draw(data.n_obs())?;
        let folds = &folds[0];
        let y_hat = cross_fit_predict(self.ml_y.as_ref(), x, y, folds)?;
        let d_hat = match &self.ml_d {
            TreatmentLearner::Regressor(r) => cross_fit_predict(r.as_ref(), x, d, folds)?,
            TreatmentLearner::Classifier(c) => cross_fit_predict_proba(c.as_ref(), x, d, folds)?,
        };
        let y_res = &y - &y_hat;
        let d_res = &d - &d_hat;
        let forest = CausalForest::fit(x, y_res.view(), d_res.view(), self.forest.clone())?;
        let (tau_oob, var_oob) = forest.predict_oob(x)?;
        let binary = d.iter().all(|&v| v == 0.0 || v == 1.0);
        Ok(CausalForestFit {
            forest,
            y: y.to_owned(),
            d: d.to_owned(),
            y_hat,
            d_hat,
            tau_oob,
            var_oob,
            binary,
            clipping_threshold: self.clipping_threshold,
            treatment_name: data.treatment_names()[0].clone(),
            n_folds: self.splitting.n_folds,
        })
    }
}

/// A fitted [`CausalForestDml`].
#[derive(Debug, Clone)]
pub struct CausalForestFit {
    forest: CausalForest,
    y: Array1<f64>,
    d: Array1<f64>,
    y_hat: Array1<f64>,
    d_hat: Array1<f64>,
    tau_oob: Array1<f64>,
    var_oob: Array1<f64>,
    binary: bool,
    clipping_threshold: f64,
    treatment_name: String,
    n_folds: usize,
}

impl CausalForestFit {
    pub fn forest(&self) -> &CausalForest {
        &self.forest
    }

    /// `τ̂(x)` and its variance at new points.
    pub fn effect(&self, x: ArrayView2<'_, f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        self.forest.predict(x)
    }

    /// Out-of-bag `τ̂(Xᵢ)` of the training rows.
    pub fn oob_effect(&self) -> &Array1<f64> {
        &self.tau_oob
    }

    /// Variances of the out-of-bag estimates.
    pub fn oob_variance(&self) -> &Array1<f64> {
        &self.var_oob
    }

    /// Cross-fitted `E[Y | X]`.
    pub fn y_hat(&self) -> &Array1<f64> {
        &self.y_hat
    }

    /// Cross-fitted `E[D | X]`.
    pub fn d_hat(&self) -> &Array1<f64> {
        &self.d_hat
    }

    pub fn is_binary_treatment(&self) -> bool {
        self.binary
    }

    /// Doubly robust scores of a binary treatment,
    /// `τ̂(Xᵢ) + (Dᵢ − êᵢ) / (êᵢ(1 − êᵢ)) · (Yᵢ − ŷᵢ − (Dᵢ − êᵢ) τ̂(Xᵢ))`,
    /// with out-of-bag `τ̂` and clipped propensities `ê` (grf's AIPW scores).
    pub fn dr_scores(&self) -> Result<Array1<f64>> {
        if !self.binary {
            return Err(DmlError::InvalidArgument(
                "doubly robust scores (ATE, BLP, GATE) need a binary 0/1 treatment".into(),
            ));
        }
        if self.tau_oob.iter().any(|v| !v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "some rows have no out-of-bag estimate; grow more trees".into(),
            ));
        }
        let t = self.clipping_threshold;
        Ok(Array1::from_iter((0..self.y.len()).map(|i| {
            let e = self.d_hat[i].clamp(t, 1.0 - t);
            let dr = self.d[i] - e;
            let tau = self.tau_oob[i];
            tau + dr / (e * (1.0 - e)) * (self.y[i] - self.y_hat[i] - dr * tau)
        })))
    }

    /// Average treatment effect from the doubly robust scores, with the usual
    /// DML standard error.
    pub fn ate(&self) -> Result<DmlFit> {
        let gamma = self.dr_scores()?;
        let n = gamma.len();
        let psi_b = Array3::from_shape_vec((n, 1, 1), gamma.to_vec()).expect("shape");
        DmlFit::from_linear_scores(
            "Causal forest (ATE)",
            vec![self.treatment_name.clone()],
            self.n_folds,
            Array3::from_elem((n, 1, 1), -1.0),
            psi_b,
        )
    }

    /// Best linear predictor of the CATE on `basis` (HC0 inference on the
    /// doubly robust scores).
    pub fn blp(&self, basis: ArrayView2<'_, f64>) -> Result<Blp> {
        let gamma = self.dr_scores()?;
        Blp::fit(gamma.view().insert_axis(ndarray::Axis(1)), basis)
    }

    /// Group average treatment effects for 0/1 group dummies.
    pub fn gate(&self, groups: ArrayView2<'_, f64>) -> Result<Blp> {
        let gamma = self.dr_scores()?;
        Blp::fit_gate(gamma.view().insert_axis(ndarray::Axis(1)), groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{LogisticRegression, Ols};
    use rand::Rng;
    use rand_distr::StandardNormal;

    /// Wager and Athey (2018)-style design: τ(x) = 1 + 1 / (1 + e^{−20(x₁ − 1/3)}),
    /// randomised binary treatment, x uniform.
    fn design(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x = Array2::from_shape_fn((n, 4), |_| rng.gen::<f64>());
        let d: Array1<f64> = (0..n).map(|_| f64::from(rng.gen::<f64>() < 0.5)).collect();
        let y: Array1<f64> = (0..n)
            .map(|i| {
                let e: f64 = rng.sample(StandardNormal);
                tau(x[[i, 0]]) * d[i] + x[[i, 1]] + e
            })
            .collect();
        (x, y, d)
    }

    fn tau(x1: f64) -> f64 {
        1.0 + 1.0 / (1.0 + (-20.0 * (x1 - 1.0 / 3.0)).exp())
    }

    #[test]
    fn recovers_the_effect_function_and_covers() {
        let (x, y, d) = design(3000, 1);
        let data = DmlData::new(y, d, x).unwrap();
        let fit = CausalForestDml::new(
            Ols::new(),
            TreatmentLearner::Classifier(Box::new(LogisticRegression::default())),
        )
        .forest(ForestSettings {
            num_trees: 1000,
            ..Default::default()
        })
        .fit(&data)
        .unwrap();
        let grid =
            Array2::from_shape_fn(
                (50, 4),
                |(i, j)| if j == 0 { (i as f64 + 0.5) / 50.0 } else { 0.5 },
            );
        let (est, var) = fit.effect(grid.view()).unwrap();
        let mut covered = 0;
        let mut sq = 0.0;
        for i in 0..50 {
            let truth = tau(grid[[i, 0]]);
            sq += (est[i] - truth).powi(2);
            if (est[i] - truth).abs() <= 1.96 * var[i].sqrt() {
                covered += 1;
            }
        }
        let rmse = (sq / 50.0).sqrt();
        assert!(rmse < 0.25, "rmse {rmse}");
        assert!(covered >= 35, "covered {covered} of 50");
        // The splits concentrate on x₁.
        let imp = fit.forest().variable_importance();
        assert!(imp[0] > 0.5, "{imp}");
        // ATE of the design: E[τ(X)] ≈ 1.66.
        let ate = fit.ate().unwrap();
        assert!((ate.coef()[0] - 1.664).abs() < 4.0 * ate.se()[0]);
        let gate = fit
            .gate(
                Array2::from_shape_fn((3000, 2), |(i, g)| {
                    f64::from((data.x()[[i, 0]] > 0.5) == (g == 1))
                })
                .view(),
            )
            .unwrap();
        assert!(gate.coef()[1] > gate.coef()[0]);
    }

    #[test]
    fn homogeneous_effect_and_continuous_treatment() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let n = 1500;
        let x = Array2::from_shape_fn((n, 3), |_| rng.gen::<f64>());
        let d: Array1<f64> = (0..n)
            .map(|i| x[[i, 0]] + rng.sample::<f64, _>(StandardNormal))
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| 0.7 * d[i] + x[[i, 1]] + rng.sample::<f64, _>(StandardNormal))
            .collect();
        let data = DmlData::new(y, d, x).unwrap();
        let fit = CausalForestDml::new(
            Ols::new(),
            TreatmentLearner::Regressor(Box::new(Ols::new())),
        )
        .forest(ForestSettings {
            num_trees: 400,
            seed: 3,
            ..Default::default()
        })
        .fit(&data)
        .unwrap();
        let mean = fit.oob_effect().mean().unwrap();
        assert!((mean - 0.7).abs() < 0.1, "{mean}");
        assert!(fit.ate().is_err());
        assert!(fit
            .oob_variance()
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn settings_are_validated_and_seeded() {
        let (x, y, d) = design(300, 2);
        let s = ForestSettings {
            num_trees: 50,
            seed: 9,
            ..Default::default()
        };
        let a = CausalForest::fit(x.view(), y.view(), d.view(), s.clone()).unwrap();
        let b = CausalForest::fit(x.view(), y.view(), d.view(), s).unwrap();
        assert_eq!(
            a.predict(x.view()).unwrap().0,
            b.predict(x.view()).unwrap().0
        );
        let bad = ForestSettings {
            sample_fraction: 0.8,
            ..Default::default()
        };
        assert!(CausalForest::fit(x.view(), y.view(), d.view(), bad).is_err());
        let one = ForestSettings {
            num_trees: 20,
            ci_group_size: 1,
            sample_fraction: 0.7,
            ..Default::default()
        };
        let f = CausalForest::fit(x.view(), y.view(), d.view(), one).unwrap();
        assert!(f.predict(x.view()).unwrap().1.iter().all(|v| v.is_nan()));
    }
}

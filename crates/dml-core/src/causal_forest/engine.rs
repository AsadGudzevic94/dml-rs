//! The tree-growing engine shared by every forest.

// Loops index several flat arrays (`row * k + column`) in step.
#![allow(clippy::needless_range_loop)]

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::ForestSettings;
use crate::error::{DmlError, Result};
use crate::stats::norm_cdf;

/// What the trees estimate, and so how they split.
#[derive(Clone, Copy)]
pub(super) enum Target<'a> {
    /// `θ(x)` solving `E[(Z − Z̄)(Y − Ȳ − (W − W̄)ᵀθ) | x] = 0` with `k`
    /// treatment columns (`z = w` for causal forests). The pseudo-outcomes use
    /// the instrument `(1 − r) Z + r W` (grf's `reduced.form.weight` `r`).
    Moment {
        y: ArrayView1<'a, f64>,
        w: ArrayView2<'a, f64>,
        z: ArrayView2<'a, f64>,
        reduced_form_weight: f64,
        iv: bool,
    },
    /// `E[Y | x]`.
    Regression { y: ArrayView1<'a, f64> },
    /// Conditional quantiles; splits separate the node's `quantiles`.
    Quantile {
        y: ArrayView1<'a, f64>,
        quantiles: &'a [f64],
    },
}

/// The fitted counterpart of [`Target`].
#[derive(Debug, Clone)]
enum Kind {
    Moment { k: usize, iv: bool },
    Regression,
    Quantile { y: Array1<f64> },
}

impl Kind {
    /// Length of a leaf's sums (count first).
    fn stat_len(&self) -> usize {
        match self {
            Kind::Moment { k, .. } => 2 + 3 * k + k * k,
            Kind::Regression => 2,
            Kind::Quantile { .. } => 1,
        }
    }
}

impl Target<'_> {
    fn kind(&self) -> Kind {
        match *self {
            Target::Moment { w, iv, .. } => Kind::Moment { k: w.ncols(), iv },
            Target::Regression { .. } => Kind::Regression,
            Target::Quantile { y, .. } => Kind::Quantile { y: y.to_owned() },
        }
    }

    fn n(&self) -> usize {
        match self {
            Target::Moment { y, .. } | Target::Regression { y } | Target::Quantile { y, .. } => {
                y.len()
            }
        }
    }

    /// Add row `i` to a leaf's sums: `[1, y, w, z, z·y, z·wᵀ]` (moments) or
    /// `[1, y]` (regression) or `[1]` (quantiles).
    fn add_stats(&self, i: usize, out: &mut [f64]) {
        out[0] += 1.0;
        match *self {
            Target::Moment { y, w, z, .. } => {
                let k = w.ncols();
                let yi = y[i];
                out[1] += yi;
                for b in 0..k {
                    out[2 + b] += w[[i, b]];
                }
                for a in 0..k {
                    let za = z[[i, a]];
                    out[2 + k + a] += za;
                    out[2 + 2 * k + a] += za * yi;
                    for b in 0..k {
                        out[2 + 3 * k + a * k + b] += za * w[[i, b]];
                    }
                }
            }
            Target::Regression { y } => out[1] += y[i],
            Target::Quantile { .. } => {}
        }
    }

    /// grf's relabelling of a node: the pseudo-outcomes `ρ` (one row of
    /// `k_rho` values per node row), and the treatment values whose variation
    /// the children must keep (`None`: plain counts). `None` when the node
    /// cannot be split (no treatment variation, a single quantile class).
    fn relabel(&self, rows: &[usize]) -> Option<Relabeled> {
        let n = rows.len();
        let nf = n as f64;
        match *self {
            Target::Moment {
                y,
                w,
                z,
                reduced_form_weight: r,
                ..
            } => {
                let k = w.ncols();
                let mut sy = 0.0;
                let (mut sw, mut sz) = (vec![0.0; k], vec![0.0; k]);
                for &i in rows {
                    sy += y[i];
                    for b in 0..k {
                        sw[b] += w[[i, b]];
                        sz[b] += z[[i, b]];
                    }
                }
                let ybar = sy / nf;
                let wbar: Vec<f64> = sw.iter().map(|v| v / nf).collect();
                let zbar: Vec<f64> = sz.iter().map(|v| v / nf).collect();
                let mut a_mat = vec![0.0; k * k];
                let mut bvec = vec![0.0; k];
                for &i in rows {
                    let dy = y[i] - ybar;
                    for a in 0..k {
                        let dz = z[[i, a]] - zbar[a];
                        bvec[a] += dz * dy;
                        for b in 0..k {
                            a_mat[a * k + b] += dz * (w[[i, b]] - wbar[b]);
                        }
                    }
                }
                // The instrument of the pseudo-outcomes.
                let zr = |i: usize, a: usize| {
                    if r == 0.0 {
                        z[[i, a]] - zbar[a]
                    } else {
                        (1.0 - r) * (z[[i, a]] - zbar[a]) + r * (w[[i, a]] - wbar[a])
                    }
                };
                let mut rho = Vec::with_capacity(n * k);
                if k == 1 {
                    let den = a_mat[0];
                    if den.is_nan() || den.abs() <= 1e-12 * nf.max(1.0) {
                        return None;
                    }
                    let theta = bvec[0] / den;
                    for &i in rows {
                        let dw = w[[i, 0]] - wbar[0];
                        rho.push(zr(i, 0) * ((y[i] - ybar) - theta * dw));
                    }
                } else {
                    let inv = invert(&a_mat, k)?;
                    let theta: Vec<f64> = (0..k)
                        .map(|a| (0..k).map(|b| inv[a * k + b] * bvec[b]).sum())
                        .collect();
                    let mut dz = vec![0.0; k];
                    for &i in rows {
                        let mut res = y[i] - ybar;
                        for b in 0..k {
                            res -= (w[[i, b]] - wbar[b]) * theta[b];
                            dz[b] = zr(i, b);
                        }
                        for a in 0..k {
                            let g: f64 = (0..k).map(|c| inv[a * k + c] * dz[c]).sum();
                            rho.push(g * res);
                        }
                    }
                }
                let mut size = Vec::with_capacity(n * k);
                for &i in rows {
                    for a in 0..k {
                        size.push(z[[i, a]]);
                    }
                }
                Some(Relabeled {
                    k_rho: k,
                    rho,
                    size: Some((k, size)),
                })
            }
            Target::Regression { y } => Some(Relabeled {
                k_rho: 1,
                rho: rows.iter().map(|&i| y[i]).collect(),
                size: None,
            }),
            Target::Quantile { y, quantiles } => {
                let mut sorted: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
                sorted.sort_by(f64::total_cmp);
                let cutoffs: Vec<f64> = quantiles
                    .iter()
                    .map(|&q| sorted[((nf * q).ceil() as usize).saturating_sub(1).min(n - 1)])
                    .collect();
                let classes: Vec<usize> = rows
                    .iter()
                    .map(|&i| cutoffs.partition_point(|&c| c < y[i]))
                    .collect();
                if classes.iter().all(|&c| c == classes[0]) {
                    return None;
                }
                let k_rho = quantiles.len() + 1;
                let mut rho = vec![0.0; n * k_rho];
                for (j, &c) in classes.iter().enumerate() {
                    rho[j * k_rho + c] = 1.0;
                }
                Some(Relabeled {
                    k_rho,
                    rho,
                    size: None,
                })
            }
        }
    }
}

/// Inverse of a small `k × k` matrix (Gauss–Jordan with partial pivoting);
/// `None` if it is singular.
pub(super) fn invert(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    let scale = a.iter().fold(0.0f64, |s, v| s.max(v.abs()));
    if scale.is_nan() || scale <= 0.0 {
        return None;
    }
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&r, &s| m[r * k + col].abs().total_cmp(&m[s * k + col].abs()))
            .expect("non-empty");
        let pv = m[pivot * k + col].abs();
        if pv.is_nan() || pv <= 1e-12 * scale {
            return None;
        }
        for c in 0..k {
            m.swap(col * k + c, pivot * k + c);
            inv.swap(col * k + c, pivot * k + c);
        }
        let p = m[col * k + col];
        for c in 0..k {
            m[col * k + c] /= p;
            inv[col * k + c] /= p;
        }
        for r in 0..k {
            if r != col {
                let f = m[r * k + col];
                if f != 0.0 {
                    for c in 0..k {
                        m[r * k + c] -= f * m[col * k + c];
                        inv[r * k + c] -= f * inv[col * k + c];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// Pseudo-outcomes of a node (see [`Target::relabel`]).
struct Relabeled {
    k_rho: usize,
    rho: Vec<f64>,
    size: Option<(usize, Vec<f64>)>,
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

const INNER: usize = usize::MAX;

#[derive(Debug, Clone)]
struct Node {
    feature: usize,
    threshold: f64,
    left: usize,
    right: usize,
    depth: usize,
    /// Leaf index, or `INNER`.
    leaf: usize,
}

impl Node {
    fn is_leaf(&self) -> bool {
        self.leaf != INNER
    }
}

#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
    /// Bitset over the training rows: in this tree's subsample.
    inbag: Vec<u64>,
    /// Sums of every leaf's estimation rows, `stat_len` values per leaf.
    stats: Vec<f64>,
    /// Estimation rows by leaf (quantile forests): leaf `l` holds
    /// `members[offsets[l]..offsets[l + 1]]`.
    members: Vec<u32>,
    offsets: Vec<u32>,
}

impl Tree {
    fn leaf(&self, row: ArrayView1<'_, f64>) -> usize {
        let mut k = 0;
        loop {
            let node = &self.nodes[k];
            if node.is_leaf() {
                return node.leaf;
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

/// A node while growing: split or not.
struct RawNode {
    feature: usize,
    threshold: f64,
    children: Option<(usize, usize)>,
    depth: usize,
}

fn descend(nodes: &[RawNode], row: ArrayView1<'_, f64>) -> usize {
    let mut k = 0;
    while let Some((l, r)) = nodes[k].children {
        k = if row[nodes[k].feature] <= nodes[k].threshold {
            l
        } else {
            r
        };
    }
    k
}

/// Grow one honest tree on `subsample` (training rows).
fn grow_tree(
    x: ArrayView2<'_, f64>,
    t: &Target<'_>,
    s: &ForestSettings,
    mtry: usize,
    mut subsample: Vec<usize>,
    rng: &mut ChaCha8Rng,
) -> Tree {
    let n = x.nrows();
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

    let p = x.ncols();
    let mut raw = vec![RawNode {
        feature: 0,
        threshold: 0.0,
        children: None,
        depth: 0,
    }];
    let mut stack: Vec<(usize, Vec<usize>)> = vec![(0, split_rows)];
    let mut features: Vec<usize> = (0..p).collect();
    while let Some((id, rows)) = stack.pop() {
        if rows.len() <= s.min_node_size || rows.len() < 2 {
            continue;
        }
        let Some(relabeled) = t.relabel(&rows) else {
            continue;
        };
        let Some((feature, threshold)) =
            best_split(x, s, &rows, &relabeled, mtry, &mut features, rng)
        else {
            continue;
        };
        let (left_rows, right_rows): (Vec<usize>, Vec<usize>) =
            rows.iter().partition(|&&i| x[[i, feature]] <= threshold);
        let depth = raw[id].depth + 1;
        let (l, r) = (raw.len(), raw.len() + 1);
        for _ in 0..2 {
            raw.push(RawNode {
                feature: 0,
                threshold: 0.0,
                children: None,
                depth,
            });
        }
        raw[id].feature = feature;
        raw[id].threshold = threshold;
        raw[id].children = Some((l, r));
        stack.push((l, left_rows));
        stack.push((r, right_rows));
    }

    // Honest leaves: the estimation rows, dropped down the tree; leaves left
    // empty collapse into their parent (grf's `honesty.prune.leaves`).
    let mut counts = vec![0usize; raw.len()];
    for &i in &est_rows {
        counts[descend(&raw, x.row(i))] += 1;
    }
    if s.prune_empty_leaves {
        prune(&mut raw, &mut counts, 0);
    }
    let mut nodes = Vec::with_capacity(raw.len());
    let mut n_leaves = 0;
    compact(&raw, 0, &mut nodes, &mut n_leaves);

    let mut tree = Tree {
        nodes,
        inbag,
        stats: Vec::new(),
        members: Vec::new(),
        offsets: Vec::new(),
    };
    let kind = t.kind();
    let len = kind.stat_len();
    tree.stats = vec![0.0; n_leaves * len];
    let mut by_leaf: Vec<(usize, u32)> = Vec::new();
    for &i in &est_rows {
        let leaf = tree.leaf(x.row(i));
        t.add_stats(i, &mut tree.stats[leaf * len..(leaf + 1) * len]);
        if matches!(kind, Kind::Quantile { .. }) {
            by_leaf.push((leaf, i as u32));
        }
    }
    if matches!(kind, Kind::Quantile { .. }) {
        by_leaf.sort_by_key(|&(leaf, _)| leaf); // stable: keeps the row order
        tree.offsets = vec![0; n_leaves + 1];
        for &(leaf, _) in &by_leaf {
            tree.offsets[leaf + 1] += 1;
        }
        for l in 0..n_leaves {
            tree.offsets[l + 1] += tree.offsets[l];
        }
        tree.members = by_leaf.into_iter().map(|(_, i)| i).collect();
    }
    tree
}

/// Collapse split nodes with an empty child; returns the node's count.
fn prune(raw: &mut [RawNode], counts: &mut [usize], id: usize) -> usize {
    let Some((l, r)) = raw[id].children else {
        return counts[id];
    };
    let (cl, cr) = (prune(raw, counts, l), prune(raw, counts, r));
    counts[id] = cl + cr;
    if cl == 0 || cr == 0 {
        raw[id].children = None;
    }
    counts[id]
}

/// Copy the reachable nodes, numbering the leaves.
fn compact(raw: &[RawNode], id: usize, out: &mut Vec<Node>, n_leaves: &mut usize) -> usize {
    let me = out.len();
    out.push(Node {
        feature: raw[id].feature,
        threshold: raw[id].threshold,
        left: 0,
        right: 0,
        depth: raw[id].depth,
        leaf: INNER,
    });
    match raw[id].children {
        None => {
            out[me].leaf = *n_leaves;
            *n_leaves += 1;
        }
        Some((l, r)) => {
            let left = compact(raw, l, out, n_leaves);
            let right = compact(raw, r, out, n_leaves);
            out[me].left = left;
            out[me].right = right;
        }
    }
    me
}

/// The best split of a node on its pseudo-outcomes, or `None` for a leaf.
fn best_split(
    x: ArrayView2<'_, f64>,
    s: &ForestSettings,
    rows: &[usize],
    rel: &Relabeled,
    mtry: usize,
    features: &mut [usize],
    rng: &mut ChaCha8Rng,
) -> Option<(usize, f64)> {
    let n = rows.len();
    let nf = n as f64;
    let kr = rel.k_rho;
    let min_count = ((s.alpha * nf).ceil() as usize).max(1);
    // Treatment variation of the node (for the stabilised imbalance limit).
    let (kz, zvals) = match &rel.size {
        Some((kz, v)) => (*kz, v.as_slice()),
        None => (0, &[][..]),
    };
    let mut min_size = vec![0.0; kz];
    for c in 0..kz {
        let mean = (0..n).map(|j| zvals[j * kz + c]).sum::<f64>() / nf;
        let size_node: f64 = (0..n).map(|j| (zvals[j * kz + c] - mean).powi(2)).sum();
        min_size[c] = s.alpha * size_node;
    }

    let p = features.len();
    let k = mtry.min(p);
    features.shuffle(rng);
    let mut best: Option<(f64, usize, f64)> = None;
    let mut order: Vec<(f64, usize)> = Vec::with_capacity(n);
    let (mut rl, mut rho_total) = (vec![0.0; kr], vec![0.0; kr]);
    let (mut zl, mut zzl) = (vec![0.0; kz], vec![0.0; kz]);
    let (mut z_total, mut zz_total) = (vec![0.0; kz], vec![0.0; kz]);
    let (mut size_l, mut size_r) = (vec![0.0; kz], vec![0.0; kz]);
    for &f in &features[..k] {
        order.clear();
        order.extend(rows.iter().enumerate().map(|(j, &i)| (x[[i, f]], j)));
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        rl.fill(0.0);
        rho_total.fill(0.0);
        zl.fill(0.0);
        zzl.fill(0.0);
        z_total.fill(0.0);
        zz_total.fill(0.0);
        for &(_, j) in &order {
            for c in 0..kr {
                rho_total[c] += rel.rho[j * kr + c];
            }
            for c in 0..kz {
                let z = zvals[j * kz + c];
                z_total[c] += z;
                zz_total[c] += z * z;
            }
        }
        let mut nl = 0usize;
        for pos in 0..n - 1 {
            let (xv, j) = order[pos];
            nl += 1;
            for c in 0..kr {
                rl[c] += rel.rho[j * kr + c];
            }
            for c in 0..kz {
                let z = zvals[j * kz + c];
                zl[c] += z;
                zzl[c] += z * z;
            }
            if order[pos + 1].0 <= xv {
                continue; // not a boundary between distinct values
            }
            let nr = n - nl;
            if nl < min_count || nr < min_count {
                continue;
            }
            let (nlf, nrf) = (nl as f64, nr as f64);
            let mut stable = true;
            for c in 0..kz {
                size_l[c] = zzl[c] - zl[c] * zl[c] / nlf;
                let zr = z_total[c] - zl[c];
                size_r[c] = (zz_total[c] - zzl[c]) - zr * zr / nrf;
                if s.stabilize_splits && (size_l[c] < min_size[c] || size_r[c] < min_size[c]) {
                    stable = false;
                }
            }
            if !stable {
                continue;
            }
            let mut decrease = 0.0;
            for c in 0..kr {
                let rr = rho_total[c] - rl[c];
                decrease += rl[c] * rl[c] / nlf + rr * rr / nrf;
            }
            if s.imbalance_penalty > 0.0 {
                let mut penalty = 0.0;
                if kz == 0 {
                    penalty = 1.0 / nlf + 1.0 / nrf;
                } else {
                    if size_l.iter().chain(&size_r).any(|&v| v <= 0.0) {
                        continue;
                    }
                    for c in 0..kz {
                        penalty += 1.0 / size_l[c] + 1.0 / size_r[c];
                    }
                }
                decrease -= s.imbalance_penalty * penalty;
            }
            if best.map_or(true, |b| decrease > b.0) {
                best = Some((decrease, f, 0.5 * (xv + order[pos + 1].0)));
            }
        }
    }
    best.map(|(_, f, thr)| (f, thr))
}

// ---------------------------------------------------------------------------
// Forest
// ---------------------------------------------------------------------------

/// A grown forest; the public forests wrap it.
#[derive(Debug, Clone)]
pub(super) struct Grf {
    settings: ForestSettings,
    trees: Vec<Tree>,
    n_features: usize,
    n_train: usize,
    importance: Array1<f64>,
    kind: Kind,
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

/// Variance of the forest average from per-tree influence values (`NaN` for
/// trees left out), by the bootstrap of little bags: groups of `l` trees share
/// a half-sample.
fn little_bag_variance(psi: &[f64], l: usize) -> f64 {
    if l < 2 {
        return f64::NAN;
    }
    let mut group_psi: Vec<(f64, &[f64])> = Vec::new();
    for chunk in psi.chunks(l) {
        if chunk.len() == l && chunk.iter().all(|v| !v.is_nan()) {
            group_psi.push((chunk.iter().sum::<f64>() / l as f64, chunk));
        }
    }
    let g = group_psi.len() as f64;
    if g < 1.0 {
        return f64::NAN;
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
    debias(var_between, group_noise, g)
}

impl Grf {
    /// Grow `settings.num_trees` honest trees on covariates `x` for `target`.
    pub(super) fn fit(
        x: ArrayView2<'_, f64>,
        target: Target<'_>,
        settings: ForestSettings,
    ) -> Result<Self> {
        let n = x.nrows();
        if target.n() != n {
            return Err(DmlError::DimensionMismatch(
                "forest covariates and outcomes have different rows".into(),
            ));
        }
        let finite = match target {
            Target::Moment { y, w, z, .. } => {
                if w.nrows() != n || z.dim() != w.dim() {
                    return Err(DmlError::DimensionMismatch(
                        "forest treatment / instrument rows".into(),
                    ));
                }
                y.iter()
                    .chain(w.iter())
                    .chain(z.iter())
                    .all(|v| v.is_finite())
            }
            Target::Regression { y } => y.iter().all(|v| v.is_finite()),
            Target::Quantile { y, quantiles } => {
                if quantiles.is_empty() || quantiles.iter().any(|q| !(*q > 0.0 && *q < 1.0)) {
                    return Err(DmlError::InvalidArgument(
                        "quantiles must be in (0, 1)".into(),
                    ));
                }
                y.iter().all(|v| v.is_finite())
            }
        };
        if !finite || x.iter().any(|v| !v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "the forest needs finite data".into(),
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
        let s = &settings;
        let t = &target;
        let group = |g: usize| -> Vec<Tree> {
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
                    grow_tree(x, t, s, mtry, sub, &mut rng)
                })
                .collect()
        };
        #[cfg(feature = "parallel")]
        let groups: Vec<Vec<Tree>> = {
            use rayon::prelude::*;
            (0..n_groups).into_par_iter().map(group).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let groups: Vec<Vec<Tree>> = (0..n_groups).map(group).collect();
        let trees: Vec<Tree> = groups.into_iter().flatten().collect();
        let importance = split_importance(&trees, p);
        Ok(Self {
            settings,
            trees,
            n_features: p,
            n_train: n,
            importance,
            kind: target.kind(),
        })
    }

    pub(super) fn settings(&self) -> &ForestSettings {
        &self.settings
    }

    pub(super) fn n_trees(&self) -> usize {
        self.trees.len()
    }

    pub(super) fn importance(&self) -> &Array1<f64> {
        &self.importance
    }

    fn check(&self, x: ArrayView2<'_, f64>, oob: bool) -> Result<()> {
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "fitted on {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        if oob && x.nrows() != self.n_train {
            return Err(DmlError::DimensionMismatch(
                "out-of-bag predictions need the training rows".into(),
            ));
        }
        Ok(())
    }

    /// The trees used at row `i` of `x`: all of them, or (`oob`) those whose
    /// subsample left training row `i` out.
    fn valid(&self, oob: bool, i: usize) -> impl Fn(&Tree) -> bool {
        move |t: &Tree| !oob || !t.contains(i)
    }

    /// Leaf means of every tree at `row` (`NaN` rows for trees left out or
    /// empty leaves), flattened; and the number of trees used.
    fn leaf_means(
        &self,
        row: ArrayView1<'_, f64>,
        valid: impl Fn(&Tree) -> bool,
    ) -> (Vec<f64>, usize) {
        let len = self.kind.stat_len();
        let m = len - 1;
        let mut out = vec![f64::NAN; self.trees.len() * m];
        let mut used = 0;
        for (b, tree) in self.trees.iter().enumerate() {
            if !valid(tree) {
                continue;
            }
            let leaf = tree.leaf(row);
            let st = &tree.stats[leaf * len..(leaf + 1) * len];
            if st[0] == 0.0 {
                continue;
            }
            for c in 0..m {
                out[b * m + c] = st[1 + c] / st[0];
            }
            used += 1;
        }
        (out, used)
    }

    /// Moment forests: `θ̂` and the variance of each component at one point.
    fn moment_estimate(&self, row: ArrayView1<'_, f64>, valid: impl Fn(&Tree) -> bool) -> Vec<f64> {
        let Kind::Moment { k, iv } = self.kind else {
            unreachable!("moment forest");
        };
        let nan = vec![f64::NAN; 2 * k];
        let (means, used) = self.leaf_means(row, valid);
        if used == 0 {
            return nan;
        }
        let m = 1 + 3 * k + k * k;
        let mut avg = vec![0.0; m];
        for tree in means.chunks(m) {
            if !tree[0].is_nan() {
                for c in 0..m {
                    avg[c] += tree[c];
                }
            }
        }
        for v in &mut avg {
            *v /= used as f64;
        }
        // Layout: [y, w (k), z (k), z·y (k), z·wᵀ (k×k)].
        let (my, mw, mz) = (avg[0], &avg[1..1 + k], &avg[1 + k..1 + 2 * k]);
        let mzy = &avg[1 + 2 * k..1 + 3 * k];
        let mzw = &avg[1 + 3 * k..];
        let mut a_mat = vec![0.0; k * k];
        let mut bvec = vec![0.0; k];
        for a in 0..k {
            bvec[a] = mzy[a] - mz[a] * my;
            for b in 0..k {
                a_mat[a * k + b] = mzw[a * k + b] - mz[a] * mw[b];
            }
        }
        let inv = if k == 1 {
            let den = a_mat[0];
            if den.is_nan() || (if iv { den == 0.0 } else { den <= 0.0 }) {
                return nan;
            }
            vec![1.0 / den]
        } else {
            match invert(&a_mat, k) {
                Some(inv) => inv,
                None => return nan,
            }
        };
        let theta: Vec<f64> = if k == 1 {
            vec![bvec[0] / a_mat[0]]
        } else {
            (0..k)
                .map(|a| (0..k).map(|b| inv[a * k + b] * bvec[b]).sum())
                .collect()
        };
        let mut out = theta.clone();
        let l = self.settings.ci_group_size;
        if l < 2 {
            out.extend(std::iter::repeat(f64::NAN).take(k));
            return out;
        }
        // Delta method: each tree's influence on θ̂ through its leaf means,
        // A⁻¹ [(m_zy − M_y m_z − M_z m_y) − (m_zw − m_z M_wᵀ − M_z m_wᵀ) θ̂].
        let n_trees = self.trees.len();
        let mut psi = vec![f64::NAN; k * n_trees];
        let mut v = vec![0.0; k];
        for (b, tree) in means.chunks(m).enumerate() {
            if tree[0].is_nan() {
                continue;
            }
            let (ty, tw, tz) = (tree[0], &tree[1..1 + k], &tree[1 + k..1 + 2 * k]);
            let (tzy, tzw) = (&tree[1 + 2 * k..1 + 3 * k], &tree[1 + 3 * k..]);
            for a in 0..k {
                let mut va = tzy[a] - my * tz[a] - mz[a] * ty;
                for c in 0..k {
                    va -= (tzw[a * k + c] - tz[a] * mw[c] - mz[a] * tw[c]) * theta[c];
                }
                v[a] = va;
            }
            for a in 0..k {
                psi[a * n_trees + b] = (0..k).map(|c| inv[a * k + c] * v[c]).sum();
            }
        }
        for a in 0..k {
            out.push(little_bag_variance(&psi[a * n_trees..(a + 1) * n_trees], l));
        }
        out
    }

    /// Moment forests: `(θ̂, variances)`, each `n × k`.
    pub(super) fn predict_moments(
        &self,
        x: ArrayView2<'_, f64>,
        oob: bool,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        self.check(x, oob)?;
        let Kind::Moment { k, .. } = self.kind else {
            unreachable!("moment forest");
        };
        let rows = map_rows(x.nrows(), |i| {
            self.moment_estimate(x.row(i), self.valid(oob, i))
        });
        let mut theta = Array2::zeros((x.nrows(), k));
        let mut var = Array2::zeros((x.nrows(), k));
        for (i, r) in rows.iter().enumerate() {
            for a in 0..k {
                theta[[i, a]] = r[a];
                var[[i, a]] = r[k + a];
            }
        }
        Ok((theta, var))
    }

    /// Regression forests: mean of the trees' leaf means and its variance.
    pub(super) fn predict_regression(
        &self,
        x: ArrayView2<'_, f64>,
        oob: bool,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        self.check(x, oob)?;
        let l = self.settings.ci_group_size;
        let rows = map_rows(x.nrows(), |i| {
            let (means, used) = self.leaf_means(x.row(i), self.valid(oob, i));
            if used == 0 {
                return vec![f64::NAN, f64::NAN];
            }
            let mean = means.iter().filter(|v| !v.is_nan()).sum::<f64>() / used as f64;
            vec![mean, little_bag_variance(&means, l)]
        });
        Ok((
            rows.iter().map(|r| r[0]).collect(),
            rows.iter().map(|r| r[1]).collect(),
        ))
    }

    /// Quantile forests: weighted quantiles of the training outcomes with the
    /// forest weights `αᵢ(x)` (grf's rule: the first outcome, in sorted order,
    /// at which the cumulative weight reaches `q`). Returns `n × quantiles`.
    pub(super) fn predict_quantiles(
        &self,
        x: ArrayView2<'_, f64>,
        quantiles: &[f64],
        oob: bool,
    ) -> Result<Array2<f64>> {
        self.check(x, oob)?;
        let Kind::Quantile { y } = &self.kind else {
            unreachable!("quantile forest");
        };
        if quantiles.iter().any(|q| !(*q > 0.0 && *q < 1.0)) {
            return Err(DmlError::InvalidArgument(
                "quantiles must be in (0, 1)".into(),
            ));
        }
        let mut q_order: Vec<usize> = (0..quantiles.len()).collect();
        q_order.sort_by(|&a, &b| quantiles[a].total_cmp(&quantiles[b]));
        let rows = map_rows(x.nrows(), |i| {
            let valid = self.valid(oob, i);
            let row = x.row(i);
            let mut weights: Vec<(u32, f64)> = Vec::new();
            let mut used = 0usize;
            for tree in &self.trees {
                if !valid(tree) {
                    continue;
                }
                let leaf = tree.leaf(row);
                let members =
                    &tree.members[tree.offsets[leaf] as usize..tree.offsets[leaf + 1] as usize];
                if members.is_empty() {
                    continue;
                }
                used += 1;
                let w = 1.0 / members.len() as f64;
                weights.extend(members.iter().map(|&m| (m, w)));
            }
            let mut out = vec![f64::NAN; quantiles.len()];
            if used == 0 {
                return out;
            }
            // Merge the weights of each training row, then sort by outcome.
            weights.sort_unstable_by_key(|&(m, _)| m);
            let mut merged: Vec<(f64, f64)> = Vec::with_capacity(weights.len());
            let mut last = u32::MAX;
            for (m, w) in weights {
                if m == last {
                    merged.last_mut().expect("merged").1 += w / used as f64;
                } else {
                    merged.push((y[m as usize], w / used as f64));
                    last = m;
                }
            }
            merged.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (mut cum, mut j) = (0.0, 0);
            for &qi in &q_order {
                while j < merged.len() && cum < quantiles[qi] {
                    cum += merged[j].1;
                    j += 1;
                }
                out[qi] = merged[j.max(1) - 1].0;
            }
            out
        });
        let mut out = Array2::zeros((x.nrows(), quantiles.len()));
        for (i, r) in rows.iter().enumerate() {
            for (c, v) in r.iter().enumerate() {
                out[[i, c]] = *v;
            }
        }
        Ok(out)
    }
}

/// grf's `variable_importance`: split frequencies by depth (up to depth 4),
/// weighted by `depth⁻²`.
fn split_importance(trees: &[Tree], p: usize) -> Array1<f64> {
    const MAX_DEPTH: usize = 4;
    let mut freq = Array2::<f64>::zeros((MAX_DEPTH, p));
    for tree in trees {
        for node in &tree.nodes {
            if !node.is_leaf() && node.depth < MAX_DEPTH {
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

/// `one(i)` for every row, in parallel with the `parallel` feature.
fn map_rows(n: usize, one: impl Fn(usize) -> Vec<f64> + Sync + Send) -> Vec<Vec<f64>> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(one).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(one).collect()
    }
}

//! Histogram-based gradient boosting, as scikit-learn's
//! `HistGradientBoostingRegressor` / `HistGradientBoostingClassifier` (in the
//! spirit of LightGBM).
//!
//! Every feature is binned once into at most `max_bins` quantile bins (plus a
//! bin for missing values). Trees are then grown best-first on per-node
//! histograms of the gradients: the gain of every split is read off cumulative
//! bin sums, so a split costs `O(bins)` instead of `O(n log n)`, and the
//! larger child's histogram is the parent's minus the smaller child's. This
//! makes boosting practical for millions of rows.
//!
//! The implementation follows scikit-learn step by step (binning by
//! `averaged_inverted_cdf` percentiles, `float32` gradients summed in `f64`,
//! best-first growth with the same heap and tie-breaking, early stopping on a
//! validation split drawn with NumPy's `RandomState`), so with the same
//! `random_state` the fitted trees and predictions match scikit-learn's.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::{Classifier, Regressor};
use crate::compat::{np_sum, shuffle_split, stratified_split, Mt19937};
use crate::error::{DmlError, Result};

const ALMOST_INF: f64 = 1e300;
const MIN_HESSIAN_TO_SPLIT: f64 = 1e-3;
/// Rows used to find the bin thresholds (scikit-learn's `subsample=2e5`).
const BINNING_SUBSAMPLE: usize = 200_000;

/// When to stop adding trees early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EarlyStopping {
    /// On for more than 10 000 samples (scikit-learn's `"auto"`).
    #[default]
    Auto,
    Enabled,
    Disabled,
}

// ---------------------------------------------------------------------------
// Binning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BinMapper {
    thresholds: Vec<Vec<f64>>,
    n_bins_non_missing: Vec<usize>,
    missing_bin: u8,
}

/// `np.percentile(sorted, p, method="averaged_inverted_cdf")` on sorted data.
fn averaged_inverted_cdf(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let q = p / 100.0;
    let vi = n as f64 * q - 1.0;
    let (prev, next) = if vi >= (n - 1) as f64 {
        (n - 1, n - 1)
    } else if vi < 0.0 {
        (0, 0)
    } else {
        let f = vi.floor() as usize;
        (f, f + 1)
    };
    let gamma = if vi - vi.floor() == 0.0 { 0.5 } else { 1.0 };
    let (a, b) = (sorted[prev], sorted[next]);
    let diff = b - a;
    // NumPy's `_lerp` for t ≥ 0.5.
    b - diff * (1.0 - gamma)
}

/// scikit-learn's `_weighted_percentile(..., average=True)` on sorted data
/// with positive weights.
fn weighted_percentile(sorted: &[f64], weights: &[f64], p: f64) -> f64 {
    let mut cdf = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        cdf.push(acc);
    }
    let total = *cdf.last().expect("non-empty");
    let mut adjusted = p / 100.0 * total;
    if adjusted == 0.0 {
        adjusted = f64::from_bits(adjusted.to_bits() + 1);
    }
    let max_idx = sorted.len() - 1;
    let idx = cdf.partition_point(|&c| c < adjusted).min(max_idx);
    if cdf[idx] - adjusted > f64::EPSILON {
        sorted[idx]
    } else {
        (sorted[idx] + sorted[(idx + 1).min(max_idx)]) / 2.0
    }
}

/// Bin thresholds of one feature (scikit-learn's `_find_binning_thresholds`).
fn binning_thresholds(col: &[f64], weights: Option<&[f64]>, max_bins: usize) -> Vec<f64> {
    let mut pairs: Vec<(f64, f64)> = match weights {
        Some(w) => col
            .iter()
            .zip(w)
            .filter(|(v, &w)| !v.is_nan() && w != 0.0)
            .map(|(&v, &w)| (v, w))
            .collect(),
        None => col
            .iter()
            .filter(|v| !v.is_nan())
            .map(|&v| (v, 1.0))
            .collect(),
    };
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let sorted: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let mut distinct = sorted.clone();
    distinct.dedup_by(|a, b| a == b);
    if distinct.len() <= 1 {
        return Vec::new();
    }
    let mut thresholds: Vec<f64> = if distinct.len() <= max_bins {
        distinct.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect()
    } else {
        let step = 100.0 / max_bins as f64;
        let ps = (1..max_bins).map(|i| i as f64 * step);
        match weights {
            None => ps.map(|p| averaged_inverted_cdf(&sorted, p)).collect(),
            Some(_) => {
                let w: Vec<f64> = pairs.iter().map(|p| p.1).collect();
                ps.map(|p| weighted_percentile(&sorted, &w, p)).collect()
            }
        }
    };
    thresholds.dedup_by(|a, b| a == b);
    for t in thresholds.iter_mut() {
        if *t > ALMOST_INF {
            *t = ALMOST_INF;
        }
    }
    thresholds
}

impl BinMapper {
    fn fit(
        x: ArrayView2<'_, f64>,
        rows: &[usize],
        weights: Option<&[f64]>,
        max_bins: usize,
        rng: &mut Mt19937,
    ) -> Self {
        // Large data: thresholds from a sample drawn with replacement, as
        // `rng.choice(n, subsample, p=weights / weights.sum())`.
        let (sample, weights): (Vec<usize>, Option<Vec<f64>>) = if rows.len() > BINNING_SUBSAMPLE {
            let n = rows.len();
            let picks: Vec<usize> = match weights {
                None => (0..BINNING_SUBSAMPLE)
                    .map(|_| rng.interval((n - 1) as u64) as usize)
                    .collect(),
                Some(w) => {
                    let total = np_sum(w);
                    let mut cdf = Vec::with_capacity(n);
                    let mut acc = 0.0;
                    for &v in w {
                        acc += v / total;
                        cdf.push(acc);
                    }
                    let last = *cdf.last().expect("non-empty");
                    for c in cdf.iter_mut() {
                        *c /= last;
                    }
                    (0..BINNING_SUBSAMPLE)
                        .map(|_| {
                            let u = rng.random_sample();
                            cdf.partition_point(|&c| c <= u).min(n - 1)
                        })
                        .collect()
                }
            };
            (picks.iter().map(|&k| rows[k]).collect(), None)
        } else {
            (rows.to_vec(), weights.map(<[f64]>::to_vec))
        };
        let one = |f: usize| {
            let col: Vec<f64> = sample.iter().map(|&i| x[[i, f]]).collect();
            binning_thresholds(&col, weights.as_deref(), max_bins)
        };
        #[cfg(feature = "parallel")]
        let thresholds: Vec<Vec<f64>> = {
            use rayon::prelude::*;
            (0..x.ncols()).into_par_iter().map(one).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let thresholds: Vec<Vec<f64>> = (0..x.ncols()).map(one).collect();
        let n_bins_non_missing = thresholds.iter().map(|t| t.len() + 1).collect();
        Self {
            thresholds,
            n_bins_non_missing,
            missing_bin: max_bins as u8,
        }
    }

    /// Column-major bins of the given rows.
    fn transform(&self, x: ArrayView2<'_, f64>, rows: &[usize]) -> Vec<u8> {
        let n = rows.len();
        let mut out = vec![0u8; n * x.ncols()];
        if n == 0 {
            return out;
        }
        let fill = |(f, col): (usize, &mut [u8])| {
            let thr = &self.thresholds[f];
            for (k, &i) in rows.iter().enumerate() {
                let v = x[[i, f]];
                col[k] = if v.is_nan() {
                    self.missing_bin
                } else {
                    // First threshold ≥ v: thresholds[b − 1] < v ≤ thresholds[b].
                    thr.partition_point(|&t| t < v) as u8
                };
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out.par_chunks_mut(n).enumerate().for_each(fill);
        }
        #[cfg(not(feature = "parallel"))]
        out.chunks_mut(n).enumerate().for_each(fill);
        out
    }
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Node {
    value: f64,
    is_leaf: bool,
    feature: usize,
    bin_threshold: u8,
    num_threshold: f64,
    missing_go_to_left: bool,
    left: usize,
    right: usize,
}

#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn predict_row(&self, row: ArrayView1<'_, f64>) -> f64 {
        let mut k = 0;
        loop {
            let node = &self.nodes[k];
            if node.is_leaf {
                return node.value;
            }
            let v = row[node.feature];
            let left = if v.is_nan() {
                node.missing_go_to_left
            } else {
                v <= node.num_threshold
            };
            k = if left { node.left } else { node.right };
        }
    }

    fn predict_binned(&self, binned: &[u8], n: usize, i: usize, missing_bin: u8) -> f64 {
        let mut k = 0;
        loop {
            let node = &self.nodes[k];
            if node.is_leaf {
                return node.value;
            }
            let b = binned[node.feature * n + i];
            let left = (node.missing_go_to_left && b == missing_bin) || b <= node.bin_threshold;
            k = if left { node.left } else { node.right };
        }
    }
}

/// Gradient and hessian histograms of one node (`n_features × n_bins`).
#[derive(Debug, Clone)]
struct Histogram {
    g: Vec<f64>,
    h: Vec<f64>,
    count: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct SplitInfo {
    gain: f64,
    feature: usize,
    bin: u8,
    missing_go_to_left: bool,
    sum_g_left: f64,
    sum_h_left: f64,
    sum_g_right: f64,
    sum_h_right: f64,
    n_left: usize,
    value_left: f64,
    value_right: f64,
}

impl SplitInfo {
    /// No valid split of feature `f`.
    fn none(f: usize) -> Self {
        Self {
            gain: -1.0,
            feature: f,
            bin: 0,
            missing_go_to_left: false,
            sum_g_left: 0.0,
            sum_h_left: 0.0,
            sum_g_right: 0.0,
            sum_h_right: 0.0,
            n_left: 0,
            value_left: 0.0,
            value_right: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct GrowNode {
    depth: usize,
    start: usize,
    stop: usize,
    sum_g: f64,
    sum_h: f64,
    value: f64,
    split: Option<SplitInfo>,
    is_leaf: bool,
    children: Option<(usize, usize)>,
    hist: Option<Histogram>,
}

impl GrowNode {
    fn n(&self) -> usize {
        self.stop - self.start
    }
}

fn node_value(sum_g: f64, sum_h: f64, l2: f64) -> f64 {
    -sum_g / (sum_h + l2 + 1e-15)
}

struct GrowSettings {
    max_leaf_nodes: Option<usize>,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2: f64,
    shrinkage: f64,
    max_features: f64,
}

/// Best-first tree growth on histograms (scikit-learn's `TreeGrower`).
struct Grower<'a> {
    binned: &'a [u8],
    n: usize,
    n_features: usize,
    n_bins: usize,
    gradients: &'a [f32],
    /// One value when the hessians are constant.
    hessians: &'a [f32],
    n_bins_non_missing: &'a [usize],
    has_missing: &'a [bool],
    missing_bin: u8,
    s: &'a GrowSettings,
    partition: Vec<u32>,
    /// Scratch space for partitioning, as long as `partition`.
    scratch: Vec<u32>,
    nodes: Vec<GrowNode>,
    heap: Vec<usize>,
    n_finalized: usize,
    rng: Option<&'a mut ChaCha8Rng>,
}

impl<'a> Grower<'a> {
    fn constant_hessian(&self) -> bool {
        self.hessians.len() == 1
    }

    fn histogram(&self, start: usize, stop: usize) -> Histogram {
        let nb = self.n_bins;
        let idx = &self.partition[start..stop];
        let constant = self.constant_hessian();
        // The root holds every sample in order: read the gradients directly.
        let root = start == 0 && stop == self.n;
        let gather = |src: &[f32]| -> Vec<f32> {
            #[cfg(feature = "parallel")]
            if idx.len() >= 100_000 {
                use rayon::prelude::*;
                return idx.par_iter().map(|&i| src[i as usize]).collect();
            }
            idx.iter().map(|&i| src[i as usize]).collect()
        };
        let ordered_g: Vec<f32> = if root {
            Vec::new()
        } else {
            gather(self.gradients)
        };
        let ordered_h: Vec<f32> = if constant || root {
            Vec::new()
        } else {
            gather(self.hessians)
        };
        let (grads, hess): (&[f32], &[f32]) = if root {
            (self.gradients, if constant { &[] } else { self.hessians })
        } else {
            (&ordered_g, &ordered_h)
        };
        let nf = self.n_features;
        let mut hist = Histogram {
            g: vec![0.0; nb * nf],
            h: vec![0.0; if constant { 0 } else { nb * nf }],
            count: vec![0; nb * nf],
        };
        // One feature: sums over the node's samples in partition order, in
        // 256-entry arrays that a u8 bin indexes without bounds checks.
        let fill = |f: usize, g: &mut [f64], h: &mut [f64], c: &mut [u32]| {
            let col = &self.binned[f * self.n..(f + 1) * self.n];
            let mut lg = [0.0f64; 256];
            let mut lc = [0u32; 256];
            let mut lh = [0.0f64; 256];
            match (root, constant) {
                (true, true) => {
                    for (&b, &gi) in col.iter().zip(grads) {
                        lg[b as usize] += f64::from(gi);
                        lc[b as usize] += 1;
                    }
                }
                (true, false) => {
                    for ((&b, &gi), &hi) in col.iter().zip(grads).zip(hess) {
                        lg[b as usize] += f64::from(gi);
                        lh[b as usize] += f64::from(hi);
                        lc[b as usize] += 1;
                    }
                }
                (false, true) => {
                    for (&i, &gi) in idx.iter().zip(grads) {
                        let b = col[i as usize];
                        lg[b as usize] += f64::from(gi);
                        lc[b as usize] += 1;
                    }
                }
                (false, false) => {
                    for ((&i, &gi), &hi) in idx.iter().zip(grads).zip(hess) {
                        let b = col[i as usize];
                        lg[b as usize] += f64::from(gi);
                        lh[b as usize] += f64::from(hi);
                        lc[b as usize] += 1;
                    }
                }
            }
            g.copy_from_slice(&lg[..nb]);
            c.copy_from_slice(&lc[..nb]);
            if !constant {
                h.copy_from_slice(&lh[..nb]);
            }
        };
        let parallel = cfg!(feature = "parallel") && idx.len() * nf >= 20_000;
        if parallel {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if constant {
                    hist.g
                        .par_chunks_mut(nb)
                        .zip(hist.count.par_chunks_mut(nb))
                        .enumerate()
                        .for_each(|(f, (g, c))| fill(f, g, &mut [], c));
                } else {
                    hist.g
                        .par_chunks_mut(nb)
                        .zip(hist.h.par_chunks_mut(nb))
                        .zip(hist.count.par_chunks_mut(nb))
                        .enumerate()
                        .for_each(|(f, ((g, h), c))| fill(f, g, h, c));
                }
            }
        } else {
            for f in 0..nf {
                let (g, c) = (
                    &mut hist.g[f * nb..(f + 1) * nb],
                    &mut hist.count[f * nb..(f + 1) * nb],
                );
                if constant {
                    fill(f, g, &mut [], c);
                } else {
                    fill(f, g, &mut hist.h[f * nb..(f + 1) * nb], c);
                }
            }
        }
        hist
    }

    fn subtract(parent: &Histogram, small: &Histogram) -> Histogram {
        Histogram {
            g: parent.g.iter().zip(&small.g).map(|(a, b)| a - b).collect(),
            h: parent.h.iter().zip(&small.h).map(|(a, b)| a - b).collect(),
            count: parent
                .count
                .iter()
                .zip(&small.count)
                .map(|(a, b)| a - b)
                .collect(),
        }
    }

    /// Best split of one feature (both scan directions), or gain −1.
    fn feature_split(&self, node: &GrowNode, hist: &Histogram, f: usize) -> SplitInfo {
        let nb = self.n_bins;
        let (g, c) = (
            &hist.g[f * nb..(f + 1) * nb],
            &hist.count[f * nb..(f + 1) * nb],
        );
        let h = if self.constant_hessian() {
            &[][..]
        } else {
            &hist.h[f * nb..(f + 1) * nb]
        };
        let n = node.n();
        let msl = self.s.min_samples_leaf;
        let l2 = self.s.l2;
        let loss_current = node.value * node.sum_g;
        let gain_of = |gl: f64, hl: f64, gr: f64, hr: f64| {
            let (vl, vr) = (node_value(gl, hl, l2), node_value(gr, hr, l2));
            loss_current - gl * vl - gr * vr
        };
        let mut best = SplitInfo::none(f);
        let finish = |best: &mut SplitInfo,
                      bin: usize,
                      gl: f64,
                      hl: f64,
                      nl: usize,
                      gain: f64,
                      mgl: bool| {
            best.gain = gain;
            best.bin = bin as u8;
            best.missing_go_to_left = mgl;
            best.sum_g_left = gl;
            best.sum_g_right = node.sum_g - gl;
            best.sum_h_left = hl;
            best.sum_h_right = node.sum_h - hl;
            best.n_left = nl;
            best.value_left = node_value(best.sum_g_left, best.sum_h_left, l2);
            best.value_right = node_value(best.sum_g_right, best.sum_h_right, l2);
        };
        let has_missing = self.has_missing[f];
        let nbnm = self.n_bins_non_missing[f];
        // Left to right; missing values (if any) go right.
        let end = nbnm - 1 + usize::from(has_missing);
        let (mut gl, mut hl, mut nl) = (0.0f64, 0.0f64, 0usize);
        let mut found: Option<(usize, f64, f64, usize, f64)> = None;
        let mut best_gain = -1.0;
        for b in 0..end {
            nl += c[b] as usize;
            let nr = n - nl;
            hl += if h.is_empty() { f64::from(c[b]) } else { h[b] };
            let hr = node.sum_h - hl;
            gl += g[b];
            let gr = node.sum_g - gl;
            if nl < msl {
                continue;
            }
            if nr < msl {
                break;
            }
            if hl < MIN_HESSIAN_TO_SPLIT {
                continue;
            }
            if hr < MIN_HESSIAN_TO_SPLIT {
                break;
            }
            let gain = gain_of(gl, hl, gr, hr);
            if gain > best_gain && gain > 0.0 {
                best_gain = gain;
                found = Some((b, gl, hl, nl, gain));
            }
        }
        if let Some((b, gl, hl, nl, gain)) = found {
            finish(&mut best, b, gl, hl, nl, gain, false);
        }
        // Right to left; missing values go left.
        if has_missing && nbnm >= 2 {
            let (mut gr, mut hr, mut nr) = (0.0f64, 0.0f64, 0usize);
            let mut best_gain = best.gain;
            let mut found: Option<(usize, f64, f64, usize, f64)> = None;
            for b in (0..=nbnm - 2).rev() {
                let k = b + 1;
                nr += c[k] as usize;
                let nl = n - nr;
                hr += if h.is_empty() { f64::from(c[k]) } else { h[k] };
                let hl = node.sum_h - hr;
                gr += g[k];
                let gl = node.sum_g - gr;
                if nr < msl {
                    continue;
                }
                if nl < msl {
                    break;
                }
                if hr < MIN_HESSIAN_TO_SPLIT {
                    continue;
                }
                if hl < MIN_HESSIAN_TO_SPLIT {
                    break;
                }
                let gain = gain_of(gl, hl, gr, hr);
                if gain > best_gain && gain > 0.0 {
                    best_gain = gain;
                    found = Some((b, gl, hl, nl, gain));
                }
            }
            if let Some((b, gl, hl, nl, gain)) = found {
                finish(&mut best, b, gl, hl, nl, gain, true);
            }
        }
        best
    }

    fn find_split(&mut self, id: usize) -> SplitInfo {
        let allowed: Option<Vec<bool>> = if self.s.max_features < 1.0 {
            let k = ((self.s.max_features * self.n_features as f64).ceil() as usize).max(1);
            let mut mask: Vec<bool> = (0..self.n_features).map(|f| f < k).collect();
            if let Some(rng) = self.rng.as_deref_mut() {
                mask.shuffle(rng);
            }
            Some(mask)
        } else {
            None
        };
        let node = &self.nodes[id];
        let hist = node.hist.as_ref().expect("histogram");
        let mut best: Option<SplitInfo> = None;
        for f in 0..self.n_features {
            let info = if allowed.as_ref().is_some_and(|m| !m[f]) {
                SplitInfo::none(f)
            } else {
                self.feature_split(node, hist, f)
            };
            match &best {
                Some(b) if info.gain <= b.gain => {}
                _ => best = Some(info),
            }
        }
        best.expect("at least one feature")
    }

    fn finalize(&mut self, id: usize) {
        self.nodes[id].is_leaf = true;
        self.nodes[id].hist = None;
        self.n_finalized += 1;
    }

    fn gain(&self, id: usize) -> f64 {
        self.nodes[id].split.expect("split").gain
    }

    /// `a < b` in the heap: larger gain first (scikit-learn's `TreeNode.__lt__`).
    fn less(&self, a: usize, b: usize) -> bool {
        self.gain(a) > self.gain(b)
    }

    // Python's heapq.
    fn sift_down(&mut self, startpos: usize, mut pos: usize) {
        let newitem = self.heap[pos];
        while pos > startpos {
            let parentpos = (pos - 1) >> 1;
            let parent = self.heap[parentpos];
            if self.less(newitem, parent) {
                self.heap[pos] = parent;
                pos = parentpos;
                continue;
            }
            break;
        }
        self.heap[pos] = newitem;
    }

    fn sift_up(&mut self, mut pos: usize) {
        let endpos = self.heap.len();
        let startpos = pos;
        let newitem = self.heap[pos];
        let mut childpos = 2 * pos + 1;
        while childpos < endpos {
            let rightpos = childpos + 1;
            if rightpos < endpos && !self.less(self.heap[childpos], self.heap[rightpos]) {
                childpos = rightpos;
            }
            self.heap[pos] = self.heap[childpos];
            pos = childpos;
            childpos = 2 * pos + 1;
        }
        self.heap[pos] = newitem;
        self.sift_down(startpos, pos);
    }

    fn heap_push(&mut self, id: usize) {
        self.heap.push(id);
        let last = self.heap.len() - 1;
        self.sift_down(0, last);
    }

    fn heap_pop(&mut self) -> usize {
        let last = self.heap.pop().expect("non-empty heap");
        if self.heap.is_empty() {
            return last;
        }
        let top = self.heap[0];
        self.heap[0] = last;
        self.sift_up(0);
        top
    }

    fn best_split_and_push(&mut self, id: usize) {
        let split = self.find_split(id);
        self.nodes[id].split = Some(split);
        if split.gain <= 0.0 {
            self.finalize(id);
        } else {
            self.heap_push(id);
        }
    }

    fn grow(mut self) -> (Vec<GrowNode>, Vec<u32>) {
        let n = self.n;
        let root_hist = self.histogram(0, n);
        let nb = self.n_bins;
        let sum_g = np_sum(&root_hist.g[..nb]);
        let sum_h = if self.constant_hessian() {
            f64::from(self.hessians[0]) * n as f64
        } else {
            np_sum(&root_hist.h[..nb])
        };
        self.nodes.push(GrowNode {
            depth: 0,
            start: 0,
            stop: n,
            sum_g,
            sum_h,
            value: 0.0,
            split: None,
            is_leaf: false,
            children: None,
            hist: Some(root_hist),
        });
        if n < 2 * self.s.min_samples_leaf || sum_h < MIN_HESSIAN_TO_SPLIT {
            self.finalize(0);
        } else {
            self.best_split_and_push(0);
        }
        while !self.heap.is_empty() {
            self.split_next();
        }
        for node in self.nodes.iter_mut().filter(|n| n.is_leaf) {
            node.value *= self.s.shrinkage;
        }
        (self.nodes, self.partition)
    }

    fn split_next(&mut self) {
        let id = self.heap_pop();
        let split = self.nodes[id].split.expect("split");
        let (start, stop) = (self.nodes[id].start, self.nodes[id].stop);
        // Stable partition of the node's samples.
        let (f, bin, mgl, missing) = (
            split.feature,
            split.bin,
            split.missing_go_to_left,
            self.missing_bin,
        );
        let binned: &'a [u8] = self.binned;
        let col = &binned[f * self.n..(f + 1) * self.n];
        let goes_left = |i: u32| {
            let b = col[i as usize];
            (mgl && b == missing) || b <= bin
        };
        // Stable partition: each chunk's left samples go, in order, after the
        // left samples of the chunks before it (same for the right ones).
        let len = stop - start;
        let n_chunks = if cfg!(feature = "parallel") && len >= 50_000 {
            #[cfg(feature = "parallel")]
            {
                rayon::current_num_threads().max(1)
            }
            #[cfg(not(feature = "parallel"))]
            1
        } else {
            1
        };
        let chunk = len.div_ceil(n_chunks).max(1);
        let node_idx = &self.partition[start..stop];
        let count_left = |c: &[u32]| c.iter().filter(|&&i| goes_left(i)).count();
        #[cfg(feature = "parallel")]
        let lefts: Vec<usize> = {
            use rayon::prelude::*;
            node_idx.par_chunks(chunk).map(count_left).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let lefts: Vec<usize> = node_idx.chunks(chunk).map(count_left).collect();
        let n_left: usize = lefts.iter().sum();
        let mid = start + n_left;
        // Carve the scratch space into each chunk's left and right destination.
        let (left_dst, right_dst) = self.scratch[..len].split_at_mut(n_left);
        let mut jobs: Vec<(&[u32], &mut [u32], &mut [u32])> = Vec::with_capacity(lefts.len());
        let (mut l_rest, mut r_rest) = (left_dst, right_dst);
        for (c, &nl) in node_idx.chunks(chunk).zip(&lefts) {
            let (l, lr) = std::mem::take(&mut l_rest).split_at_mut(nl);
            let (r, rr) = std::mem::take(&mut r_rest).split_at_mut(c.len() - nl);
            l_rest = lr;
            r_rest = rr;
            jobs.push((c, l, r));
        }
        let write = |(c, l, r): (&[u32], &mut [u32], &mut [u32])| {
            let (mut a, mut b) = (0, 0);
            for &i in c {
                if goes_left(i) {
                    l[a] = i;
                    a += 1;
                } else {
                    r[b] = i;
                    b += 1;
                }
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            jobs.into_par_iter().for_each(write);
        }
        #[cfg(not(feature = "parallel"))]
        jobs.into_iter().for_each(write);
        self.partition[start..stop].copy_from_slice(&self.scratch[..len]);

        let depth = self.nodes[id].depth + 1;
        let n_leaf_nodes = self.n_finalized + self.heap.len() + 2;
        let l = self.nodes.len();
        let child = |s: usize, e: usize, g: f64, h: f64, v: f64| GrowNode {
            depth,
            start: s,
            stop: e,
            sum_g: g,
            sum_h: h,
            value: v,
            split: None,
            is_leaf: false,
            children: None,
            hist: None,
        };
        self.nodes.push(child(
            start,
            mid,
            split.sum_g_left,
            split.sum_h_left,
            split.value_left,
        ));
        self.nodes.push(child(
            mid,
            stop,
            split.sum_g_right,
            split.sum_h_right,
            split.value_right,
        ));
        let (li, ri) = (l, l + 1);
        self.nodes[id].children = Some((li, ri));
        if !self.has_missing[f] {
            let mut s = split;
            s.missing_go_to_left = self.nodes[li].n() > self.nodes[ri].n();
            self.nodes[id].split = Some(s);
        }

        if self.s.max_leaf_nodes == Some(n_leaf_nodes) {
            self.finalize(li);
            self.finalize(ri);
            while let Some(k) = self.heap.pop() {
                self.finalize(k);
            }
            self.nodes[id].hist = None;
            return;
        }
        if self.s.max_depth == Some(depth) {
            self.finalize(li);
            self.finalize(ri);
            self.nodes[id].hist = None;
            return;
        }
        if self.nodes[li].n() < 2 * self.s.min_samples_leaf {
            self.finalize(li);
        }
        if self.nodes[ri].n() < 2 * self.s.min_samples_leaf {
            self.finalize(ri);
        }
        let (split_left, split_right) = (!self.nodes[li].is_leaf, !self.nodes[ri].is_leaf);
        if split_left || split_right {
            let (small, large) = if self.nodes[li].n() < self.nodes[ri].n() {
                (li, ri)
            } else {
                (ri, li)
            };
            let small_hist = self.histogram(self.nodes[small].start, self.nodes[small].stop);
            let parent = self.nodes[id].hist.take().expect("parent histogram");
            let large_hist = Self::subtract(&parent, &small_hist);
            if !self.nodes[small].is_leaf {
                self.nodes[small].hist = Some(small_hist);
            }
            if !self.nodes[large].is_leaf {
                self.nodes[large].hist = Some(large_hist);
            }
            if split_left {
                self.best_split_and_push(li);
            }
            if split_right {
                self.best_split_and_push(ri);
            }
        }
        self.nodes[id].hist = None;
    }
}

/// Predictor nodes in preorder, as scikit-learn's `_fill_predictor_arrays`.
fn make_tree(nodes: &[GrowNode], mapper: &BinMapper) -> Tree {
    fn fill(id: usize, nodes: &[GrowNode], mapper: &BinMapper, out: &mut Vec<Node>) {
        let gn = &nodes[id];
        let k = out.len();
        out.push(Node {
            value: gn.value,
            is_leaf: gn.is_leaf,
            feature: 0,
            bin_threshold: 0,
            num_threshold: 0.0,
            missing_go_to_left: false,
            left: 0,
            right: 0,
        });
        if gn.is_leaf {
            return;
        }
        let s = gn.split.expect("split of an inner node");
        let (li, ri) = gn.children.expect("children");
        out[k].feature = s.feature;
        out[k].bin_threshold = s.bin;
        out[k].missing_go_to_left = s.missing_go_to_left;
        out[k].num_threshold = if s.bin as usize == mapper.n_bins_non_missing[s.feature] - 1 {
            f64::INFINITY
        } else {
            mapper.thresholds[s.feature][s.bin as usize]
        };
        out[k].left = out.len();
        fill(li, nodes, mapper, out);
        out[k].right = out.len();
        fill(ri, nodes, mapper, out);
    }
    let mut out = Vec::new();
    fill(0, nodes, mapper, &mut out);
    Tree { nodes: out }
}

// ---------------------------------------------------------------------------
// Boosting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Loss {
    SquaredError,
    LogLoss,
}

/// scipy's `logit` (precise near 0.5).
fn logit(p: f64) -> f64 {
    if !(0.3..=0.65).contains(&p) {
        (p / (1.0 - p)).ln()
    } else {
        let s = 2.0 * (p - 0.5);
        s.ln_1p() - (-s).ln_1p()
    }
}

/// scipy's `expit`.
fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn log1pexp(x: f64) -> f64 {
    if x <= -37.0 {
        x.exp()
    } else if x <= -2.0 {
        x.exp().ln_1p()
    } else if x <= 18.0 {
        (1.0 + x.exp()).ln()
    } else if x <= 33.3 {
        x + (-x).exp()
    } else {
        x
    }
}

impl Loss {
    fn sample_loss(self, y: f64, raw: f64) -> f64 {
        match self {
            Loss::SquaredError => 0.5 * (raw - y) * (raw - y),
            Loss::LogLoss => log1pexp(raw) - y * raw,
        }
    }

    /// Gradient and hessian at one sample (scikit-learn's Cython formulas).
    fn grad_hess(self, y: f64, raw: f64) -> (f64, f64) {
        match self {
            Loss::SquaredError => (raw - y, 1.0),
            Loss::LogLoss => {
                if raw > -37.0 {
                    let e = (-raw).exp();
                    (((1.0 - y) - y * e) / (1.0 + e), e / ((1.0 + e) * (1.0 + e)))
                } else {
                    let e = raw.exp();
                    (e - y, e)
                }
            }
        }
    }

    /// `np.average(loss, weights)`.
    fn mean_loss(self, y: &[f64], raw: &[f64], w: Option<&[f64]>) -> f64 {
        let per: Vec<f64> = y
            .iter()
            .zip(raw)
            .map(|(&y, &r)| self.sample_loss(y, r))
            .collect();
        match w {
            None => np_sum(&per) / per.len() as f64,
            Some(w) => {
                let prod: Vec<f64> = per.iter().zip(w).map(|(a, b)| a * b).collect();
                np_sum(&prod) / np_sum(w)
            }
        }
    }
}

/// Settings and fitted state shared by the regressor and the classifier.
#[derive(Debug, Clone)]
struct HistBooster {
    learning_rate: f64,
    max_iter: usize,
    max_leaf_nodes: Option<usize>,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    l2_regularization: f64,
    max_features: f64,
    max_bins: usize,
    early_stopping: EarlyStopping,
    validation_fraction: Option<f64>,
    n_iter_no_change: usize,
    tol: f64,
    random_state: u32,
    // Fitted.
    baseline: f64,
    trees: Vec<Tree>,
    n_features: usize,
    validation_scores: Vec<f64>,
    fitted: bool,
}

impl HistBooster {
    fn new() -> Self {
        Self {
            learning_rate: 0.1,
            max_iter: 100,
            max_leaf_nodes: Some(31),
            max_depth: None,
            min_samples_leaf: 20,
            l2_regularization: 0.0,
            max_features: 1.0,
            max_bins: 255,
            early_stopping: EarlyStopping::Auto,
            validation_fraction: Some(0.1),
            n_iter_no_change: 10,
            tol: 1e-7,
            random_state: 0,
            baseline: 0.0,
            trees: Vec::new(),
            n_features: 0,
            validation_scores: Vec::new(),
            fitted: false,
        }
    }

    fn unfitted(&self) -> Self {
        Self {
            trees: Vec::new(),
            validation_scores: Vec::new(),
            fitted: false,
            ..self.clone()
        }
    }

    fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(DmlError::InvalidArgument(m.into()));
        if self.learning_rate.is_nan() || self.learning_rate <= 0.0 {
            return bad("learning_rate must be positive");
        }
        if self.max_iter == 0 {
            return bad("max_iter must be at least 1");
        }
        if self.max_leaf_nodes.is_some_and(|m| m < 2) {
            return bad("max_leaf_nodes must be at least 2");
        }
        if self.max_depth.is_some_and(|d| d < 1) {
            return bad("max_depth must be at least 1");
        }
        if self.min_samples_leaf == 0 {
            return bad("min_samples_leaf must be at least 1");
        }
        if self.l2_regularization.is_nan() || self.l2_regularization < 0.0 {
            return bad("l2_regularization must be non-negative");
        }
        if self.max_features.is_nan() || self.max_features <= 0.0 || self.max_features > 1.0 {
            return bad("max_features must be in (0, 1]");
        }
        if !(2..=255).contains(&self.max_bins) {
            return bad("max_bins must be between 2 and 255");
        }
        if self
            .validation_fraction
            .is_some_and(|v| v.is_nan() || v <= 0.0 || v >= 1.0)
        {
            return bad("validation_fraction must be in (0, 1)");
        }
        if self.n_iter_no_change == 0 {
            return bad("n_iter_no_change must be at least 1");
        }
        Ok(())
    }

    fn fit(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
        loss: Loss,
    ) -> Result<()> {
        self.validate()?;
        let n = x.nrows();
        if n == 0 || y.len() != n || w.is_some_and(|w| w.len() != n) {
            return Err(DmlError::DimensionMismatch(
                "histogram boosting x / y / weights rows".into(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "the target must be finite".into(),
            ));
        }
        if let Some(w) = w {
            if w.iter().any(|&v| !(v.is_finite() && v >= 0.0)) || w.sum() <= 0.0 {
                return Err(DmlError::InvalidArgument(
                    "sample weights must be non-negative with a positive sum".into(),
                ));
            }
        }
        let mut rng = Mt19937::new(self.random_state);
        let random_seed = rng.interval(u64::from(u32::MAX) - 1) as u32;
        let feature_seed = rng.interval(u64::from(u32::MAX) - 1);
        let do_es = match self.early_stopping {
            EarlyStopping::Auto => n > 10_000,
            EarlyStopping::Enabled => true,
            EarlyStopping::Disabled => false,
        };
        let use_val = do_es && self.validation_fraction.is_some();
        let (train, val): (Vec<usize>, Vec<usize>) = if use_val {
            let vf = self.validation_fraction.expect("validation fraction");
            match loss {
                Loss::SquaredError => shuffle_split(n, vf, random_seed),
                Loss::LogLoss => stratified_split(&y.to_vec(), random_seed, vf)?,
            }
        } else {
            ((0..n).collect(), Vec::new())
        };
        let y_tr: Vec<f64> = train.iter().map(|&i| y[i]).collect();
        let w_tr: Option<Vec<f64>> = w.map(|w| train.iter().map(|&i| w[i]).collect());
        let y_val: Vec<f64> = val.iter().map(|&i| y[i]).collect();
        let w_val: Option<Vec<f64>> = w.map(|w| val.iter().map(|&i| w[i]).collect());

        let mut bin_rng = Mt19937::new(random_seed);
        let mapper = BinMapper::fit(x, &train, w_tr.as_deref(), self.max_bins, &mut bin_rng);
        let n_tr = train.len();
        let binned = mapper.transform(x, &train);
        let binned_val = mapper.transform(x, &val);
        let n_features = x.ncols();
        let has_missing: Vec<bool> = (0..n_features)
            .map(|f| binned[f * n_tr..(f + 1) * n_tr].contains(&mapper.missing_bin))
            .collect();

        let mean = match &w_tr {
            None => np_sum(&y_tr) / n_tr as f64,
            Some(w) => {
                let prod: Vec<f64> = y_tr.iter().zip(w).map(|(a, b)| a * b).collect();
                np_sum(&prod) / np_sum(w)
            }
        };
        let baseline = match loss {
            Loss::SquaredError => mean,
            Loss::LogLoss => {
                let eps = 10.0 * f64::EPSILON;
                logit(mean.clamp(eps, 1.0 - eps))
            }
        };
        let mut raw = vec![baseline; n_tr];
        let mut raw_val = vec![baseline; val.len()];
        let mut scores: Vec<f64> = Vec::new();
        let score = |raw: &[f64], raw_val: &[f64]| -> f64 {
            if use_val {
                -loss.mean_loss(&y_val, raw_val, w_val.as_deref())
            } else {
                -loss.mean_loss(&y_tr, raw, w_tr.as_deref())
            }
        };
        if do_es {
            scores.push(score(&raw, &raw_val));
        }
        let constant_hessian = loss == Loss::SquaredError && w_tr.is_none();
        let mut gradients = vec![0f32; n_tr];
        let mut hessians = vec![1f32; if constant_hessian { 1 } else { n_tr }];
        let settings = GrowSettings {
            max_leaf_nodes: self.max_leaf_nodes,
            max_depth: self.max_depth,
            min_samples_leaf: self.min_samples_leaf,
            l2: self.l2_regularization,
            shrinkage: self.learning_rate,
            max_features: self.max_features,
        };
        let mut feature_rng = ChaCha8Rng::seed_from_u64(feature_seed);
        let mut trees = Vec::new();
        for _ in 0..self.max_iter {
            let gh = |i: usize| -> (f32, f32) {
                let (g, h) = loss.grad_hess(y_tr[i], raw[i]);
                match &w_tr {
                    None => (g as f32, h as f32),
                    Some(w) => ((w[i] * g) as f32, (w[i] * h) as f32),
                }
            };
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if constant_hessian {
                    gradients
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, g)| *g = gh(i).0);
                } else {
                    gradients
                        .par_iter_mut()
                        .zip(hessians.par_iter_mut())
                        .enumerate()
                        .for_each(|(i, (g, h))| (*g, *h) = gh(i));
                }
            }
            #[cfg(not(feature = "parallel"))]
            for i in 0..n_tr {
                let (g, h) = gh(i);
                gradients[i] = g;
                if !constant_hessian {
                    hessians[i] = h;
                }
            }
            let grower = Grower {
                binned: &binned,
                n: n_tr,
                n_features,
                n_bins: self.max_bins + 1,
                gradients: &gradients,
                hessians: &hessians,
                n_bins_non_missing: &mapper.n_bins_non_missing,
                has_missing: &has_missing,
                missing_bin: mapper.missing_bin,
                s: &settings,
                partition: (0..n_tr as u32).collect(),
                scratch: vec![0; n_tr],
                nodes: Vec::new(),
                heap: Vec::new(),
                n_finalized: 0,
                rng: Some(&mut feature_rng),
            };
            let (nodes, _) = grower.grow();
            let tree = make_tree(&nodes, &mapper);
            // Each training sample gets the value of its leaf (as scikit-learn's
            // per-leaf update; the tree routes binned samples to the same leaves).
            let leaf_value = |i: usize| tree.predict_binned(&binned, n_tr, i, mapper.missing_bin);
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                raw.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, r)| *r += leaf_value(i));
            }
            #[cfg(not(feature = "parallel"))]
            for (i, r) in raw.iter_mut().enumerate() {
                *r += leaf_value(i);
            }
            if do_es {
                if use_val {
                    for (k, r) in raw_val.iter_mut().enumerate() {
                        *r += tree.predict_binned(&binned_val, val.len(), k, mapper.missing_bin);
                    }
                }
                trees.push(tree);
                scores.push(score(&raw, &raw_val));
                if self.should_stop(&scores) {
                    break;
                }
            } else {
                trees.push(tree);
            }
        }
        self.baseline = baseline;
        self.trees = trees;
        self.n_features = n_features;
        self.validation_scores = scores;
        self.fitted = true;
        Ok(())
    }

    fn should_stop(&self, scores: &[f64]) -> bool {
        let reference_position = self.n_iter_no_change + 1;
        if scores.len() < reference_position {
            return false;
        }
        let reference = scores[scores.len() - reference_position] + self.tol;
        !scores[scores.len() - reference_position + 1..]
            .iter()
            .any(|&s| s > reference)
    }

    fn raw_predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(DmlError::InvalidArgument("the model is not fitted".into()));
        }
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "fitted on {} features, got {}",
                self.n_features,
                x.ncols()
            )));
        }
        let one = |i: usize| {
            let row = x.row(i);
            let mut r = self.baseline;
            for t in &self.trees {
                r += t.predict_row(row);
            }
            r
        };
        #[cfg(feature = "parallel")]
        let out: Vec<f64> = {
            use rayon::prelude::*;
            (0..x.nrows()).into_par_iter().map(one).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let out: Vec<f64> = (0..x.nrows()).map(one).collect();
        Ok(Array1::from(out))
    }
}

macro_rules! hist_api {
    () => {
        /// Shrinkage of every tree's contribution (default 0.1).
        pub fn learning_rate(mut self, rate: f64) -> Self {
            self.inner.learning_rate = rate;
            self
        }

        /// Maximum number of boosting iterations (trees; default 100).
        pub fn max_iter(mut self, n: usize) -> Self {
            self.inner.max_iter = n;
            self
        }

        /// Maximum number of leaves per tree (default `Some(31)`).
        pub fn max_leaf_nodes(mut self, n: Option<usize>) -> Self {
            self.inner.max_leaf_nodes = n;
            self
        }

        /// Maximum depth of each tree (default unlimited).
        pub fn max_depth(mut self, depth: Option<usize>) -> Self {
            self.inner.max_depth = depth;
            self
        }

        /// Minimum number of samples per leaf (default 20).
        pub fn min_samples_leaf(mut self, n: usize) -> Self {
            self.inner.min_samples_leaf = n;
            self
        }

        /// L2 penalty on the leaf values (default 0).
        pub fn l2_regularization(mut self, l2: f64) -> Self {
            self.inner.l2_regularization = l2;
            self
        }

        /// Share of features considered at each split (default 1; random subsets
        /// are drawn with this crate's generator, not NumPy's).
        pub fn max_features(mut self, share: f64) -> Self {
            self.inner.max_features = share;
            self
        }

        /// Bins per feature for non-missing values (2 to 255, default 255).
        pub fn max_bins(mut self, bins: usize) -> Self {
            self.inner.max_bins = bins;
            self
        }

        /// Early stopping (default [`EarlyStopping::Auto`]: on above 10 000 samples).
        pub fn early_stopping(mut self, mode: EarlyStopping) -> Self {
            self.inner.early_stopping = mode;
            self
        }

        /// Share of the training data held out to score early stopping
        /// (default `Some(0.1)`; `None` scores on the training data).
        pub fn validation_fraction(mut self, fraction: Option<f64>) -> Self {
            self.inner.validation_fraction = fraction;
            self
        }

        /// Stop when none of the last `n` iterations improved the score by more
        /// than `tol` (defaults 10 and `1e-7`).
        pub fn n_iter_no_change(mut self, n: usize) -> Self {
            self.inner.n_iter_no_change = n;
            self
        }

        pub fn tol(mut self, tol: f64) -> Self {
            self.inner.tol = tol;
            self
        }

        /// Seed of the validation split and binning subsample (NumPy
        /// `RandomState` semantics, as scikit-learn's integer `random_state`).
        pub fn random_state(mut self, seed: u32) -> Self {
            self.inner.random_state = seed;
            self
        }

        /// Number of fitted trees (fewer than `max_iter` after early stopping).
        pub fn n_iter(&self) -> usize {
            self.inner.trees.len()
        }

        /// Early-stopping scores (negative loss), starting with the baseline;
        /// empty without early stopping.
        pub fn validation_scores(&self) -> &[f64] {
            &self.inner.validation_scores
        }
    };
}

/// Histogram-based gradient boosting for regression (squared error), as
/// scikit-learn's `HistGradientBoostingRegressor`.
#[derive(Debug, Clone)]
pub struct HistGradientBoostingRegressor {
    inner: HistBooster,
}

impl Default for HistGradientBoostingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl HistGradientBoostingRegressor {
    pub fn new() -> Self {
        Self {
            inner: HistBooster::new(),
        }
    }

    hist_api!();
}

impl Regressor for HistGradientBoostingRegressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.inner.fit(x, y, None, Loss::SquaredError)
    }

    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.inner.fit(x, y, Some(w), Loss::SquaredError)
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.inner.raw_predict(x)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self {
            inner: self.inner.unfitted(),
        })
    }
}

/// Histogram-based gradient boosting for a binary outcome (log loss), as
/// scikit-learn's `HistGradientBoostingClassifier`. Labels must be 0 / 1.
#[derive(Debug, Clone)]
pub struct HistGradientBoostingClassifier {
    inner: HistBooster,
}

impl Default for HistGradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl HistGradientBoostingClassifier {
    pub fn new() -> Self {
        Self {
            inner: HistBooster::new(),
        }
    }

    hist_api!();

    fn check_labels(y: ArrayView1<'_, f64>) -> Result<()> {
        if y.iter().any(|&v| v != 0.0 && v != 1.0) {
            return Err(DmlError::InvalidArgument(
                "HistGradientBoostingClassifier needs 0/1 labels".into(),
            ));
        }
        if y.iter().all(|&v| v == y[0]) {
            return Err(DmlError::InvalidArgument(
                "HistGradientBoostingClassifier needs both classes".into(),
            ));
        }
        Ok(())
    }
}

impl Classifier for HistGradientBoostingClassifier {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        Self::check_labels(y)?;
        self.inner.fit(x, y, None, Loss::LogLoss)
    }

    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        Self::check_labels(y)?;
        self.inner.fit(x, y, Some(w), Loss::LogLoss)
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        Ok(self.inner.raw_predict(x)?.mapv(expit))
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::new(Self {
            inner: self.inner.unfitted(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn data(n: usize) -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((n, 3), |(i, j)| {
            (((i + 1) * (j + 7) * 7919) % 1009) as f64 / 100.0
        });
        let y = x
            .rows()
            .into_iter()
            .map(|r| (r[0] - 5.0).abs() + 0.3 * r[1])
            .collect();
        (x, y)
    }

    #[test]
    fn percentile_matches_numpy_examples() {
        // np.percentile([1, 2, 3, 4], [25, 50, 75], method="averaged_inverted_cdf") = [1.5, 2.5, 3.5]
        let s = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(averaged_inverted_cdf(&s, 25.0), 1.5);
        assert_eq!(averaged_inverted_cdf(&s, 50.0), 2.5);
        assert_eq!(averaged_inverted_cdf(&s, 75.0), 3.5);
        // ... and [1, 2, 3] at 50 is 2.
        assert_eq!(averaged_inverted_cdf(&[1.0, 2.0, 3.0], 50.0), 2.0);
    }

    #[test]
    fn regressor_fits_and_beats_the_mean() {
        let (x, y) = data(2000);
        let mut m = HistGradientBoostingRegressor::new().early_stopping(EarlyStopping::Disabled);
        m.fit(x.view(), y.view()).unwrap();
        assert_eq!(m.n_iter(), 100);
        let p = m.predict(x.view()).unwrap();
        let mean = y.mean().unwrap();
        let sse: f64 = (&p - &y).mapv(|v| v * v).sum();
        let sst: f64 = y.mapv(|v| (v - mean).powi(2)).sum();
        assert!(sse < 0.05 * sst, "R² too low: {}", 1.0 - sse / sst);
        // Fresh copies are unfitted.
        assert!(m.fresh().predict(x.view()).is_err());
    }

    #[test]
    fn early_stopping_stops() {
        let (x, y) = data(3000);
        let mut m = HistGradientBoostingRegressor::new()
            .early_stopping(EarlyStopping::Enabled)
            .max_iter(500)
            .learning_rate(0.5)
            .tol(1e-3);
        m.fit(x.view(), y.view()).unwrap();
        assert!(m.n_iter() < 500);
        assert_eq!(m.validation_scores().len(), m.n_iter() + 1);
    }

    #[test]
    fn classifier_probabilities() {
        let (x, y) = data(1500);
        let labels = y.mapv(|v| f64::from(v > 3.0));
        let mut m = HistGradientBoostingClassifier::new();
        m.fit(x.view(), labels.view()).unwrap();
        let p = m.predict_proba(x.view()).unwrap();
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
        let acc = p
            .iter()
            .zip(&labels)
            .filter(|(p, l)| (**p > 0.5) == (**l == 1.0))
            .count();
        assert!(acc as f64 / 1500.0 > 0.95);
        assert!(m.fit(x.view(), Array1::zeros(1500).view()).is_err());
    }

    #[test]
    fn missing_values_are_routed() {
        let (mut x, y) = data(800);
        for i in (0..800).step_by(7) {
            x[[i, 0]] = f64::NAN;
        }
        let mut m = HistGradientBoostingRegressor::new();
        m.fit(x.view(), y.view()).unwrap();
        assert!(m.predict(x.view()).unwrap().iter().all(|v| v.is_finite()));
    }
}

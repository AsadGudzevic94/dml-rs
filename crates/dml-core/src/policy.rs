//! Policy trees: a shallow decision tree that recommends treatment where the
//! estimated effect is positive, as DoubleML's `DoubleMLPolicyTree` (Athey and
//! Wager, 2021).
//!
//! The tree classifies the sign of the orthogonal signal `ψ_b` (labels
//! `(sign + 1) / 2`) with sample weights `|ψ_b|`, so misclassifying units with
//! large effects costs more. It is grown like scikit-learn's
//! `DecisionTreeClassifier` (Gini criterion, depth and leaf-size limits) and
//! then pruned by minimal cost-complexity pruning (`ccp_alpha`), with the same
//! pruning order as scikit-learn.

use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{DmlError, Result};
use crate::learner::{MaxFeatures, Node, Tree, TreeBuilder, TreeParams};

/// One node of a fitted [`PolicyTree`] (see [`PolicyTree::nodes`]).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PolicyNode {
    /// Rows with `feature <= threshold` go to node `left`, the others to `right`.
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    /// A leaf: the weighted share of positive effects; treatment is
    /// recommended when it exceeds 0.5.
    Leaf { share_positive: f64 },
}

/// A fitted policy tree.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PolicyTree {
    depth: usize,
    min_samples_leaf: usize,
    ccp_alpha: f64,
    tree: Tree,
    n_features: usize,
}

/// Weighted size and Gini impurity of one node.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct NodeStats {
    weight: f64,
    weight1: f64,
}

impl NodeStats {
    fn share1(&self) -> f64 {
        self.weight1 / self.weight
    }

    fn gini(&self) -> f64 {
        let p1 = self.share1();
        1.0 - p1 * p1 - (1.0 - p1) * (1.0 - p1)
    }
}

impl PolicyTree {
    /// Fit on the orthogonal signal (one value per unit) and the features the
    /// policy may depend on. DoubleML's defaults: `min_samples_leaf = 8`,
    /// `ccp_alpha = 0.01`.
    pub fn fit(
        signal: ArrayView1<'_, f64>,
        features: ArrayView2<'_, f64>,
        depth: usize,
        min_samples_leaf: usize,
        ccp_alpha: f64,
    ) -> Result<Self> {
        let (n, p) = features.dim();
        if signal.len() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "signal has {} values, features {n} rows",
                signal.len()
            )));
        }
        if p == 0 || min_samples_leaf == 0 || ccp_alpha.is_nan() || ccp_alpha < 0.0 {
            return Err(DmlError::InvalidArgument(
                "need at least one feature, min_samples_leaf >= 1 and ccp_alpha >= 0".into(),
            ));
        }
        let labels: Vec<f64> = signal
            .iter()
            .map(|&s| (s.signum_or_zero() + 1.0) / 2.0)
            .collect();
        let weights: Vec<f64> = signal.iter().map(|s| s.abs()).collect();
        // scikit-learn ignores samples with zero weight.
        let rows: Vec<usize> = (0..n).filter(|&i| weights[i] > 0.0).collect();
        if rows.is_empty() {
            return Err(DmlError::InvalidArgument(
                "the signal is zero everywhere".into(),
            ));
        }
        let (grown, _) = TreeBuilder {
            x: features,
            y: &labels,
            w: &weights,
            params: TreeParams {
                max_depth: Some(depth),
                min_samples_split: 2,
                min_samples_leaf,
                max_features: MaxFeatures::All,
            },
            n_try: p,
            rng: ChaCha8Rng::seed_from_u64(0),
            importance: vec![0.0; p],
        }
        .grow(rows.clone());

        // Weighted class counts of every node, from the training rows.
        let mut stats = vec![NodeStats::default(); grown.nodes.len()];
        for &i in &rows {
            let mut node = 0usize;
            loop {
                stats[node].weight += weights[i];
                stats[node].weight1 += weights[i] * labels[i];
                match grown.nodes[node] {
                    Node::Leaf { .. } => break,
                    Node::Split {
                        feature,
                        threshold,
                        left,
                        right,
                    } => {
                        node = if features[[i, feature]] <= threshold {
                            left
                        } else {
                            right
                        } as usize
                    }
                }
            }
        }
        let tree = prune(&grown, &stats, ccp_alpha);
        Ok(Self {
            depth,
            min_samples_leaf,
            ccp_alpha,
            tree,
            n_features: p,
        })
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn min_samples_leaf(&self) -> usize {
        self.min_samples_leaf
    }

    pub fn ccp_alpha(&self) -> f64 {
        self.ccp_alpha
    }

    /// Number of leaves (distinct recommendations groups).
    pub fn n_leaves(&self) -> usize {
        self.tree
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Leaf { .. }))
            .count()
    }

    /// Recommended treatment (1) or not (0) for each row of `features`: the
    /// class with the larger weighted share in the row's leaf (no treatment on
    /// ties, as scikit-learn's argmax).
    pub fn predict(&self, features: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        if features.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "the tree was fitted on {} features, got {}",
                self.n_features,
                features.ncols()
            )));
        }
        Ok(features
            .rows()
            .into_iter()
            .map(|row| f64::from(self.leaf_share(row) > 0.5))
            .collect())
    }

    fn leaf_share(&self, row: ArrayView1<'_, f64>) -> f64 {
        match self.tree.nodes[self.tree.leaf_index(row)] {
            Node::Leaf { value } => value,
            Node::Split { .. } => unreachable!("leaf_index returns a leaf"),
        }
    }

    /// The nodes in preorder; node 0 is the root (for plotting the tree).
    pub fn nodes(&self) -> Vec<PolicyNode> {
        self.tree
            .nodes
            .iter()
            .map(|n| match *n {
                Node::Leaf { value } => PolicyNode::Leaf {
                    share_positive: value,
                },
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                } => PolicyNode::Split {
                    feature,
                    threshold,
                    left: left as usize,
                    right: right as usize,
                },
            })
            .collect()
    }

    /// The decision rules as text, in the style of scikit-learn's
    /// `export_text`; `feature_names` defaults to `x0, x1, …`.
    pub fn export_text(&self, feature_names: Option<&[String]>) -> String {
        let name = |j: usize| match feature_names {
            Some(names) if j < names.len() => names[j].clone(),
            _ => format!("x{j}"),
        };
        let mut lines: Vec<(usize, String)> = Vec::new();
        self.write_rules(0, 0, &name, &mut lines);
        lines
            .into_iter()
            .map(|(depth, text)| format!("{}|--- {text}\n", "|   ".repeat(depth)))
            .collect()
    }

    fn write_rules(
        &self,
        node: usize,
        depth: usize,
        name: &dyn Fn(usize) -> String,
        lines: &mut Vec<(usize, String)>,
    ) {
        match self.tree.nodes[node] {
            Node::Leaf { value } => {
                let class = if value > 0.5 { "treat" } else { "do not treat" };
                lines.push((
                    depth,
                    format!("recommend: {class} (share of positive effects {value:.2})"),
                ));
            }
            Node::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                lines.push((depth, format!("{} <= {threshold:.4}", name(feature))));
                self.write_rules(left as usize, depth + 1, name, lines);
                lines.push((depth, format!("{} >  {threshold:.4}", name(feature))));
                self.write_rules(right as usize, depth + 1, name, lines);
            }
        }
    }
}

/// `f64::signum` maps 0 to 1; numpy's `sign` maps it to 0.
trait SignumOrZero {
    fn signum_or_zero(self) -> f64;
}

impl SignumOrZero for f64 {
    fn signum_or_zero(self) -> f64 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

/// Minimal cost-complexity pruning, as scikit-learn's `_cost_complexity_prune`:
/// repeatedly collapse the internal node with the smallest effective alpha
/// `(r(t) − r(T_t)) / (|leaves(T_t)| − 1)` while it is at most `ccp_alpha`,
/// where `r(t) = W_t · Gini_t / W_root`. Nodes are visited in scikit-learn's
/// depth-first (preorder) numbering, so ties are broken the same way.
fn prune(tree: &Tree, stats: &[NodeStats], ccp_alpha: f64) -> Tree {
    // Preorder numbering: node, left subtree, right subtree.
    let mut order: Vec<usize> = Vec::with_capacity(tree.nodes.len());
    let mut stack = vec![0usize];
    while let Some(node) = stack.pop() {
        order.push(node);
        if let Node::Split { left, right, .. } = tree.nodes[node] {
            stack.push(right as usize);
            stack.push(left as usize);
        }
    }
    let m = order.len();
    let mut pre = vec![0usize; tree.nodes.len()]; // node id -> preorder index
    for (k, &node) in order.iter().enumerate() {
        pre[node] = k;
    }
    let children = |k: usize| match tree.nodes[order[k]] {
        Node::Split { left, right, .. } => Some((pre[left as usize], pre[right as usize])),
        Node::Leaf { .. } => None,
    };
    let total = stats[0].weight;
    let r_node: Vec<f64> = order
        .iter()
        .map(|&node| stats[node].weight * stats[node].gini() / total)
        .collect();
    let mut parent = vec![usize::MAX; m];
    for k in 0..m {
        if let Some((l, r)) = children(k) {
            parent[l] = k;
            parent[r] = k;
        }
    }
    let mut is_leaf: Vec<bool> = (0..m).map(|k| children(k).is_none()).collect();
    let mut r_branch = vec![0.0; m];
    let mut n_leaves = vec![0usize; m];
    for leaf in (0..m).filter(|&k| is_leaf[k]) {
        r_branch[leaf] = r_node[leaf];
        let mut k = leaf;
        while k != 0 {
            let up = parent[k];
            r_branch[up] += r_node[leaf];
            n_leaves[up] += 1;
            k = up;
        }
    }
    let mut candidate: Vec<bool> = is_leaf.iter().map(|l| !l).collect();
    let mut in_subtree = vec![true; m];
    while candidate[0] {
        let (mut alpha, mut pruned) = (f64::MAX, 0usize);
        for k in (0..m).filter(|&k| candidate[k]) {
            let a = (r_node[k] - r_branch[k]) / (n_leaves[k] as f64 - 1.0);
            if a < alpha {
                alpha = a;
                pruned = k;
            }
        }
        if ccp_alpha < alpha {
            break;
        }
        // Remove the subtree below `pruned`; it becomes a leaf.
        let mut stack = vec![pruned];
        while let Some(k) = stack.pop() {
            if !in_subtree[k] {
                continue;
            }
            candidate[k] = false;
            is_leaf[k] = false;
            in_subtree[k] = false;
            if let Some((l, r)) = children(k) {
                stack.push(l);
                stack.push(r);
            }
        }
        is_leaf[pruned] = true;
        in_subtree[pruned] = true;
        let n_pruned = n_leaves[pruned] - 1;
        n_leaves[pruned] = 0;
        let r_diff = r_node[pruned] - r_branch[pruned];
        r_branch[pruned] = r_node[pruned];
        let mut k = parent[pruned];
        while k != usize::MAX {
            n_leaves[k] -= n_pruned;
            r_branch[k] += r_diff;
            k = parent[k];
        }
    }
    // Rebuild the kept nodes, with leaf values = weighted share of class 1.
    let mut nodes = Vec::new();
    let mut stack = vec![(0usize, usize::MAX, false)]; // (preorder index, parent slot, is right child)
    while let Some((k, parent_slot, is_right)) = stack.pop() {
        let slot = nodes.len();
        if is_leaf[k] {
            nodes.push(Node::Leaf {
                value: stats[order[k]].share1(),
            });
        } else {
            let Node::Split {
                feature, threshold, ..
            } = tree.nodes[order[k]]
            else {
                unreachable!("internal nodes are splits")
            };
            nodes.push(Node::Split {
                feature,
                threshold,
                left: 0,
                right: 0,
            });
            let (l, r) = children(k).expect("split");
            stack.push((r, slot, true));
            stack.push((l, slot, false));
        }
        if parent_slot != usize::MAX {
            if let Node::Split { left, right, .. } = &mut nodes[parent_slot] {
                if is_right {
                    *right = slot as u32;
                } else {
                    *left = slot as u32;
                }
            }
        }
    }
    Tree { nodes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn splits_on_the_feature_that_separates_positive_from_negative_effects() {
        // Effect positive when x0 > 0, negative otherwise; x1 is noise.
        let n = 200;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                i as f64 / n as f64 - 0.5
            } else {
                ((i * 37) % 11) as f64
            }
        });
        let signal: Array1<f64> = (0..n)
            .map(|i| if x[[i, 0]] > 0.0 { 2.0 } else { -1.0 })
            .collect();
        let tree = PolicyTree::fit(signal.view(), x.view(), 2, 8, 0.01).unwrap();
        assert_eq!(tree.n_leaves(), 2);
        let pred = tree
            .predict(array![[-0.3, 1.0], [0.3, 1.0]].view())
            .unwrap();
        assert_eq!(pred.to_vec(), vec![0.0, 1.0]);
        assert!(tree.export_text(None).contains("x0 <="));
    }

    #[test]
    fn large_ccp_alpha_prunes_to_the_root() {
        let x = Array2::from_shape_fn((40, 1), |(i, _)| i as f64);
        let signal: Array1<f64> = (0..40).map(|i| if i < 20 { -1.0 } else { 3.0 }).collect();
        let tree = PolicyTree::fit(signal.view(), x.view(), 2, 8, 10.0).unwrap();
        assert_eq!(tree.n_leaves(), 1);
        // Weighted share of positive effects is 60/80 > 0.5: treat everyone.
        assert_eq!(tree.predict(x.view()).unwrap().sum(), 40.0);
    }
}

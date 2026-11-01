//! Policy learning by exact tree search (Athey and Wager 2021, "Policy learning
//! with observational data", Econometrica 89(1); Sverdrup, Kanodia, Zhou, Athey
//! and Wager 2020, "policytree: Policy learning via doubly robust empirical
//! welfare maximization over trees", JOSS): a port of the R package
//! `policytree` (1.2.5) that finds the same trees, and the doubly robust value
//! of a policy.

use ndarray::{Array3, ArrayView2};

use crate::error::{DmlError, Result};
use crate::fit::DmlFit;

/// Exhaustive search for the depth-`k` decision tree that maximises the summed
/// rewards `Σ_i Γ_i(π(X_i))`, where `Γ` (`n × d`) holds a doubly robust score
/// of every unit under every action (e.g. from [`Apos`](crate::Apos)). Splits
/// are axis-aligned (`x_j ≤ v` goes left), every candidate split is evaluated
/// (`split_step = 1`), and leaves have at least `min_node_size` units. Two
/// sibling leaves with the same action are merged. The search is exact, so
/// its cost grows like `(p n)^k`: depth 2 is practical for thousands of units
/// and tens of features, depth 3 only for small problems. Ties are broken as
/// in `policytree` (first feature, first threshold, first action).
///
/// ```
/// use dml_core::OptimalPolicyTree;
/// use ndarray::Array2;
///
/// // Action 1 is better when x0 > 0.5.
/// let x = Array2::from_shape_fn((100, 2), |(i, j)| if j == 0 { i as f64 / 100.0 } else { (i % 7) as f64 });
/// let gamma = Array2::from_shape_fn((100, 2), |(i, a)| if (a == 1) == (i >= 50) { 1.0 } else { 0.0 });
/// let tree = OptimalPolicyTree::new(1).fit(x.view(), gamma.view())?;
/// assert_eq!(tree.predict(x.view())?[10], 0);
/// assert_eq!(tree.predict(x.view())?[90], 1);
/// # Ok::<(), dml_core::DmlError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct OptimalPolicyTree {
    depth: usize,
    split_step: usize,
    min_node_size: usize,
}

impl OptimalPolicyTree {
    pub fn new(depth: usize) -> Self {
        Self {
            depth,
            split_step: 1,
            min_node_size: 1,
        }
    }

    /// Consider only every `split_step`-th candidate split (an approximation
    /// that speeds up the search on dense features; default 1).
    pub fn split_step(mut self, split_step: usize) -> Self {
        self.split_step = split_step;
        self
    }

    /// Smallest number of units in a leaf (default 1).
    pub fn min_node_size(mut self, min_node_size: usize) -> Self {
        self.min_node_size = min_node_size;
        self
    }

    pub fn fit(
        &self,
        x: ArrayView2<'_, f64>,
        gamma: ArrayView2<'_, f64>,
    ) -> Result<OptimalPolicyTreeFit> {
        let (n, p) = x.dim();
        if gamma.nrows() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "x has {n} rows, gamma {}",
                gamma.nrows()
            )));
        }
        if n == 0 || p == 0 || gamma.ncols() == 0 {
            return Err(DmlError::InvalidArgument(
                "need at least one unit, feature and action".into(),
            ));
        }
        if self.split_step == 0 || self.min_node_size == 0 {
            return Err(DmlError::InvalidArgument(
                "split_step and min_node_size must be at least 1".into(),
            ));
        }
        if !x.iter().chain(gamma.iter()).all(|v| v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "x and gamma must not contain NaN or infinite values".into(),
            ));
        }
        // Points sorted along every feature, ties by sample index.
        let sets: Vec<Vec<usize>> = (0..p)
            .map(|j| {
                let mut idx: Vec<usize> = (0..n).collect();
                idx.sort_by(|&a, &b| x[[a, j]].total_cmp(&x[[b, j]]).then(a.cmp(&b)));
                idx
            })
            .collect();
        let search = Search {
            x,
            gamma,
            split_step: self.split_step,
            min_node_size: self.min_node_size,
        };
        let root = search.find_best_split(&sets, self.depth);
        Ok(OptimalPolicyTreeFit {
            nodes: flatten(root),
            n_features: p,
            n_actions: gamma.ncols(),
        })
    }
}

/// A node of the search (children owned).
struct SearchNode {
    index: usize,
    value: f64,
    reward: f64,
    action: usize,
    children: Option<Box<(SearchNode, SearchNode)>>,
}

impl SearchNode {
    fn leaf(reward: f64, action: usize) -> Self {
        Self {
            index: 0,
            value: 0.0,
            reward,
            action,
            children: None,
        }
    }
}

/// The search state; `'x` and `'g` are the lifetimes of the two borrowed
/// arrays (they may differ, so each gets its own parameter).
struct Search<'x, 'g> {
    x: ArrayView2<'x, f64>,
    gamma: ArrayView2<'g, f64>,
    split_step: usize,
    min_node_size: usize,
}

impl Search<'_, '_> {
    /// Best single action (rewards summed in the order of the first feature).
    fn level_zero(&self, sets: &[Vec<usize>]) -> SearchNode {
        let mut best = (f64::NEG_INFINITY, 0);
        for d in 0..self.gamma.ncols() {
            let mut sum = 0.0;
            for &i in &sets[0] {
                sum += self.gamma[[i, d]];
            }
            if sum > best.0 {
                best = (sum, d);
            }
        }
        SearchNode::leaf(best.0, best.1)
    }

    /// Best split with the best action in each leaf.
    fn level_one(&self, sets: &[Vec<usize>]) -> SearchNode {
        let num_points = sets[0].len();
        let num_rewards = self.gamma.ncols();
        let mut best_reward = f64::NEG_INFINITY;
        let (mut best_left, mut best_right) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        let (mut action_left, mut action_right) = (0, 0);
        let (mut split_var, mut split_val) = (0, 0.0);
        let mut sums = vec![vec![0.0; num_points + 1]; num_rewards];
        for (p, set) in sets.iter().enumerate() {
            for (d, sum) in sums.iter_mut().enumerate() {
                for (k, &i) in set.iter().enumerate() {
                    sum[k + 1] = sum[k] + self.gamma[[i, d]];
                }
            }
            let (mut split_counter, mut samples_counter) = (0usize, 0usize);
            for k in 0..num_points.saturating_sub(1) {
                let n = k + 1;
                let value = self.x[[set[k], p]];
                let next_value = self.x[[set[k + 1], p]];
                split_counter += 1;
                samples_counter += 1;
                if value == next_value {
                    continue;
                }
                if samples_counter < self.min_node_size
                    || num_points - samples_counter < self.min_node_size
                {
                    continue;
                }
                if split_counter >= self.split_step {
                    split_counter = 0;
                } else {
                    continue;
                }
                let (mut left_best, mut right_best) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
                let (mut left_action, mut right_action) = (0, 0);
                for (d, sum) in sums.iter().enumerate() {
                    let left = sum[n];
                    let right = sum[num_points] - left;
                    if left_best < left {
                        left_best = left;
                        left_action = d;
                    }
                    if right_best < right {
                        right_best = right;
                        right_action = d;
                    }
                }
                if best_reward < left_best + right_best {
                    best_reward = left_best + right_best;
                    best_left = left_best;
                    best_right = right_best;
                    action_left = left_action;
                    action_right = right_action;
                    split_var = p;
                    split_val = value;
                }
            }
        }
        if best_reward > f64::NEG_INFINITY {
            if action_left == action_right {
                SearchNode::leaf(best_reward, action_left)
            } else {
                SearchNode {
                    index: split_var,
                    value: split_val,
                    reward: best_reward,
                    action: 0,
                    children: Some(Box::new((
                        SearchNode::leaf(best_left, action_left),
                        SearchNode::leaf(best_right, action_right),
                    ))),
                }
            }
        } else {
            self.level_zero(sets)
        }
    }

    fn find_best_split(&self, sets: &[Vec<usize>], level: usize) -> SearchNode {
        match level {
            0 => return self.level_zero(sets),
            1 => return self.level_one(sets),
            _ => {}
        }
        let num_points = sets[0].len();
        let n_total = self.x.nrows();
        let mut best: Option<(SearchNode, SearchNode, usize, f64)> = None;
        let mut in_left = vec![false; n_total];
        for p in 0..sets.len() {
            in_left.iter_mut().for_each(|v| *v = false);
            let (mut split_counter, mut samples_counter) = (0usize, 0usize);
            for k in 0..num_points.saturating_sub(1) {
                // Move the k-th point along feature p from the right to the left.
                let point = sets[p][k];
                in_left[point] = true;
                let next = sets[p][k + 1];
                split_counter += 1;
                samples_counter += 1;
                if self.x[[point, p]] >= self.x[[next, p]] {
                    continue;
                }
                if samples_counter < self.min_node_size
                    || num_points - samples_counter < self.min_node_size
                {
                    continue;
                }
                if split_counter >= self.split_step {
                    split_counter = 0;
                } else {
                    continue;
                }
                let left_sets: Vec<Vec<usize>> = sets
                    .iter()
                    .map(|s| s.iter().copied().filter(|&i| in_left[i]).collect())
                    .collect();
                let right_sets: Vec<Vec<usize>> = sets
                    .iter()
                    .map(|s| s.iter().copied().filter(|&i| !in_left[i]).collect())
                    .collect();
                let left = self.find_best_split(&left_sets, level - 1);
                let right = self.find_best_split(&right_sets, level - 1);
                let better = match &best {
                    None => true,
                    Some((bl, br, _, _)) => left.reward + right.reward > bl.reward + br.reward,
                };
                if better {
                    best = Some((left, right, p, self.x[[point, p]]));
                }
            }
        }
        match best {
            None => self.level_zero(sets),
            Some((left, right, var, val)) => {
                if left.children.is_none()
                    && right.children.is_none()
                    && left.action == right.action
                {
                    SearchNode::leaf(left.reward + right.reward, left.action)
                } else {
                    SearchNode {
                        index: var,
                        value: val,
                        reward: left.reward + right.reward,
                        action: 0,
                        children: Some(Box::new((left, right))),
                    }
                }
            }
        }
    }
}

/// Breadth-first node list (the layout of `policytree`'s `nodes`).
fn flatten(root: SearchNode) -> Vec<OptimalPolicyNode> {
    let mut out = Vec::new();
    let mut queue = std::collections::VecDeque::from([root]);
    let mut next_child = 1;
    while let Some(node) = queue.pop_front() {
        match node.children {
            None => out.push(OptimalPolicyNode {
                is_leaf: true,
                action: node.action,
                split_variable: 0,
                split_value: 0.0,
                left: 0,
                right: 0,
                reward: node.reward,
            }),
            Some(children) => {
                out.push(OptimalPolicyNode {
                    is_leaf: false,
                    action: 0,
                    split_variable: node.index,
                    split_value: node.value,
                    left: next_child,
                    right: next_child + 1,
                    reward: node.reward,
                });
                next_child += 2;
                let (l, r) = *children;
                queue.push_back(l);
                queue.push_back(r);
            }
        }
    }
    out
}

/// A node of an [`OptimalPolicyTreeFit`] (0-based indices).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct OptimalPolicyNode {
    pub is_leaf: bool,
    /// Action (column of `Γ`) of a leaf.
    pub action: usize,
    /// Split of an inner node: `x[split_variable] <= split_value` goes left.
    pub split_variable: usize,
    pub split_value: f64,
    /// Positions of the children in the node list.
    pub left: usize,
    pub right: usize,
    /// Summed reward of the units in the node under the tree's actions.
    pub reward: f64,
}

/// Result of [`OptimalPolicyTree::fit`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptimalPolicyTreeFit {
    nodes: Vec<OptimalPolicyNode>,
    n_features: usize,
    n_actions: usize,
}

impl OptimalPolicyTreeFit {
    /// Nodes in breadth-first order, the root first.
    pub fn nodes(&self) -> &[OptimalPolicyNode] {
        &self.nodes
    }

    pub fn n_actions(&self) -> usize {
        self.n_actions
    }

    /// Recommended action (column of `Γ`) for every row of `x`.
    pub fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Vec<usize>> {
        if x.ncols() != self.n_features {
            return Err(DmlError::DimensionMismatch(format!(
                "x has {} columns, the tree was fitted with {}",
                x.ncols(),
                self.n_features
            )));
        }
        Ok((0..x.nrows())
            .map(|i| {
                let mut node = &self.nodes[0];
                while !node.is_leaf {
                    node = if x[[i, node.split_variable]] <= node.split_value {
                        &self.nodes[node.left]
                    } else {
                        &self.nodes[node.right]
                    };
                }
                node.action
            })
            .collect())
    }
}

/// Doubly robust value `E[Γ(π(X))]` of one or several policies, given the
/// scores `Γ` (`n × d`, e.g. from [`Apos`](crate::Apos)) and each policy's
/// action per unit. The result has one estimate per policy with the standard
/// error of the mean score; all share the units, so differences of policies
/// ([`DmlFit::linear_combination`], or `a - b`) have correct standard errors
/// and the bootstrap gives joint intervals. Evaluate a policy on units that
/// were not used to learn it.
pub fn policy_value(gamma: ArrayView2<'_, f64>, policies: &[(&str, &[usize])]) -> Result<DmlFit> {
    let (n, d) = gamma.dim();
    if policies.is_empty() {
        return Err(DmlError::InvalidArgument("need at least one policy".into()));
    }
    let mut psi_b = Array3::<f64>::zeros((n, policies.len(), 1));
    for (k, (name, actions)) in policies.iter().enumerate() {
        if actions.len() != n {
            return Err(DmlError::DimensionMismatch(format!(
                "policy {name} has {} actions for {n} units",
                actions.len()
            )));
        }
        for (i, &a) in actions.iter().enumerate() {
            if a >= d {
                return Err(DmlError::InvalidArgument(format!(
                    "policy {name} uses action {a}, but gamma has {d} columns"
                )));
            }
            psi_b[[i, k, 0]] = gamma[[i, a]];
        }
    }
    let psi_a = Array3::from_elem(psi_b.dim(), -1.0);
    DmlFit::from_linear_scores(
        "Policy value",
        policies.iter().map(|(name, _)| name.to_string()).collect(),
        1,
        psi_a,
        psi_b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn depth_two_recovers_quadrants() {
        // Best action depends on both features: 0 0 / 1 2 pattern.
        let x = Array2::from_shape_fn((200, 3), |(i, j)| match j {
            0 => (i % 20) as f64,
            1 => (i / 20) as f64,
            _ => ((i * 7) % 13) as f64,
        });
        let best = |i: usize| match (i % 20 < 10, i / 20 < 5) {
            (true, _) => 0,
            (false, true) => 1,
            (false, false) => 2,
        };
        let gamma = Array2::from_shape_fn((200, 3), |(i, a)| if a == best(i) { 1.0 } else { 0.0 });
        let tree = OptimalPolicyTree::new(2)
            .fit(x.view(), gamma.view())
            .unwrap();
        let pred = tree.predict(x.view()).unwrap();
        assert!((0..200).all(|i| pred[i] == best(i)));
        // The left subtree (x0 <= 9) is a single leaf: three nodes pruned to one.
        assert_eq!(tree.nodes().len(), 5);
        assert!((tree.nodes()[0].reward - 200.0).abs() < 1e-12);
    }

    #[test]
    fn policy_value_and_contrast() {
        let gamma = Array2::from_shape_fn((4, 2), |(i, a)| (i + 3 * a) as f64);
        let fit = policy_value(
            gamma.view(),
            &[("treat all", &[1, 1, 1, 1]), ("treat none", &[0, 0, 0, 0])],
        )
        .unwrap();
        assert!((fit.coef()[0] - 4.5).abs() < 1e-12);
        assert!((fit.coef()[1] - 1.5).abs() < 1e-12);
        let diff = fit
            .linear_combination(ndarray::array![[1.0, -1.0]].view(), vec!["gain".into()])
            .unwrap();
        assert!((diff.coef()[0] - 3.0).abs() < 1e-12);
        assert!(diff.se()[0] < 1e-12);
        assert!(policy_value(gamma.view(), &[("bad", &[2, 0, 0, 0])]).is_err());
    }
}

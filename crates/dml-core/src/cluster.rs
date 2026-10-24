//! Cluster-robust DML (Chiang, Kato, Ma & Sasaki, 2022), as DoubleML:
//! one-way or two-way clustered observations.
//!
//! With clustered data, sample splitting assigns whole clusters to folds (for
//! two-way clustering the folds are the `K²` combinations of the two cluster
//! dimensions, so no cluster appears in both the training and the test set),
//! θ solves a fold-weighted score equation, and the variance sums the score
//! within clusters.

use std::collections::{BTreeMap, HashSet};

use ndarray::{Array1, ArrayView1};

use crate::crossfit::{Fold, KFold};
use crate::error::{DmlError, Result};
use crate::score::LinearScore;

/// One fold of a cluster split: the observation-level fold and, for each
/// cluster variable, the clusters in its test set.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterFold {
    pub fold: Fold,
    pub test_clusters: Vec<Vec<i64>>,
}

/// Cluster-level sample splitting for one repetition (DoubleML's
/// `DoubleMLClusterResampling`): K-fold over the distinct clusters of each
/// variable (seeded per variable), then the Cartesian product of the folds,
/// first variable outermost.
pub(crate) fn split_clusters(
    clusters: &[Array1<i64>],
    n_folds: usize,
    seed: u64,
) -> Result<Vec<ClusterFold>> {
    let n = clusters[0].len();
    let mut per_var: Vec<(Vec<i64>, Vec<usize>)> = Vec::new(); // (distinct clusters, fold of each cluster)
    for (v, c) in clusters.iter().enumerate() {
        let mut distinct: Vec<i64> = c.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        let var_seed = seed.wrapping_add((v as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
        let folds = KFold::new(n_folds, var_seed)?
            .split(distinct.len())
            .map_err(|_| {
                DmlError::InvalidArgument(format!(
                    "cluster variable {v} has {} clusters, fewer than {n_folds} folds",
                    distinct.len()
                ))
            })?;
        let mut fold_of = vec![0usize; distinct.len()];
        for (k, f) in folds.iter().enumerate() {
            for &i in &f.test {
                fold_of[i] = k;
            }
        }
        per_var.push((distinct, fold_of));
    }
    // Fold index of every observation in every cluster variable.
    let obs_fold: Vec<Vec<usize>> = clusters
        .iter()
        .zip(&per_var)
        .map(|(c, (distinct, fold_of))| {
            c.iter()
                .map(|v| fold_of[distinct.binary_search(v).expect("collected above")])
                .collect()
        })
        .collect();
    let m = clusters.len();
    let combos = n_folds.pow(m as u32);
    let mut out = Vec::with_capacity(combos);
    for combo in 0..combos {
        // Mixed-radix digits, first variable most significant.
        let mut digits = vec![0usize; m];
        let mut rest = combo;
        for v in (0..m).rev() {
            digits[v] = rest % n_folds;
            rest /= n_folds;
        }
        let test: Vec<usize> = (0..n)
            .filter(|&i| (0..m).all(|v| obs_fold[v][i] == digits[v]))
            .collect();
        let train: Vec<usize> = (0..n)
            .filter(|&i| (0..m).all(|v| obs_fold[v][i] != digits[v]))
            .collect();
        let test_clusters = per_var
            .iter()
            .zip(&digits)
            .map(|((distinct, fold_of), &k)| {
                distinct
                    .iter()
                    .zip(fold_of)
                    .filter(|(_, &f)| f == k)
                    .map(|(&c, _)| c)
                    .collect()
            })
            .collect();
        out.push(ClusterFold {
            fold: Fold { train, test },
            test_clusters,
        });
    }
    Ok(out)
}

/// Test clusters derived from a fold's test rows (exact for one-way
/// clustering and for two-way designs where every cluster pair is observed).
pub(crate) fn derive_test_clusters(clusters: &[Array1<i64>], fold: &Fold) -> Vec<Vec<i64>> {
    clusters
        .iter()
        .map(|c| {
            let mut v: Vec<i64> = fold.test.iter().map(|&i| c[i]).collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .collect()
}

/// Cluster structure of a fit: the cluster variables and, per repetition,
/// the cluster folds used.
#[derive(Debug, Clone)]
pub(crate) struct ClusterInfo {
    pub clusters: Vec<Array1<i64>>,
    pub n_folds_per_cluster: usize,
    /// `[rep][fold]`.
    pub folds: Vec<Vec<ClusterFold>>,
    /// Observations of each cluster, per variable.
    members: Vec<BTreeMap<i64, Vec<usize>>>,
}

impl ClusterInfo {
    pub(crate) fn new(clusters: Vec<Array1<i64>>, folds: Vec<Vec<ClusterFold>>) -> Result<Self> {
        if clusters.is_empty() || clusters.len() > 2 {
            return Err(DmlError::InvalidArgument(
                "one or two cluster variables are supported".into(),
            ));
        }
        let m = clusters.len() as u32;
        let n_folds = folds.first().map_or(0, |f| f.len());
        let k = (n_folds as f64).powf(1.0 / m as f64).round() as usize;
        if k.pow(m) != n_folds || folds.iter().any(|f| f.len() != n_folds) {
            return Err(DmlError::InvalidArgument(
                "cluster folds must be the product of the per-variable folds".into(),
            ));
        }
        let members = clusters
            .iter()
            .map(|c| {
                let mut map: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
                for (i, &v) in c.iter().enumerate() {
                    map.entry(v).or_default().push(i);
                }
                map
            })
            .collect();
        Ok(Self {
            clusters,
            n_folds_per_cluster: k,
            folds,
            members,
        })
    }

    pub(crate) fn n_ways(&self) -> usize {
        self.clusters.len()
    }

    /// Number of clusters (two-way: the smaller dimension), DoubleML's
    /// variance scaling factor.
    pub(crate) fn n_clusters(&self) -> usize {
        self.members.iter().map(BTreeMap::len).min().unwrap_or(0)
    }

    /// θ as the root of the fold-weighted score: each fold's sums are scaled by
    /// `1 / Π |test clusters|`.
    pub(crate) fn solve_theta(&self, score: &LinearScore, rep: usize) -> Result<f64> {
        let (mut sa, mut sb) = (0.0, 0.0);
        for cf in &self.folds[rep] {
            let scale = 1.0
                / cf.test_clusters
                    .iter()
                    .map(|c| c.len() as f64)
                    .product::<f64>();
            sa += scale * cf.fold.test.iter().map(|&i| score.psi_a[i]).sum::<f64>();
            sb += scale * cf.fold.test.iter().map(|&i| score.psi_b[i]).sum::<f64>();
        }
        if sa == 0.0 || !sa.is_finite() {
            return Err(DmlError::InvalidArgument(
                "weighted mean of psi_a is zero or not finite: theta is not identified".into(),
            ));
        }
        Ok(-sb / sa)
    }

    /// Cluster-robust variance `σ̂² = Γ̂ / (C · Ĵ²)` (DoubleML's `_var_est`),
    /// so that `se = √σ̂²`.
    pub(crate) fn variance(
        &self,
        psi: ArrayView1<'_, f64>,
        psi_deriv: ArrayView1<'_, f64>,
        rep: usize,
    ) -> f64 {
        let k = self.n_folds_per_cluster as f64;
        let (mut gamma, mut jac) = (0.0, 0.0);
        let sum_over = |idx: &[usize]| idx.iter().map(|&i| psi[i]).sum::<f64>();
        for cf in &self.folds[rep] {
            let test_deriv: f64 = cf.fold.test.iter().map(|&i| psi_deriv[i]).sum();
            if self.n_ways() == 1 {
                let i_k = &cf.test_clusters[0];
                let c = 1.0 / i_k.len() as f64;
                for cl in i_k {
                    let s = self.members[0].get(cl).map_or(0.0, |m| sum_over(m));
                    gamma += c * s * s;
                }
                jac += test_deriv / i_k.len() as f64;
            } else {
                let (i_k, j_l) = (&cf.test_clusters[0], &cf.test_clusters[1]);
                let (ni, nj) = (i_k.len() as f64, j_l.len() as f64);
                let c = ni.min(nj) / (ni * nj).powi(2);
                let in_j: HashSet<i64> = j_l.iter().copied().collect();
                let in_i: HashSet<i64> = i_k.iter().copied().collect();
                for cl in i_k {
                    if let Some(m) = self.members[0].get(cl) {
                        let s: f64 = m
                            .iter()
                            .filter(|&&i| in_j.contains(&self.clusters[1][i]))
                            .map(|&i| psi[i])
                            .sum();
                        gamma += c * s * s;
                    }
                }
                for cl in j_l {
                    if let Some(m) = self.members[1].get(cl) {
                        let s: f64 = m
                            .iter()
                            .filter(|&&i| in_i.contains(&self.clusters[0][i]))
                            .map(|&i| psi[i])
                            .sum();
                        gamma += c * s * s;
                    }
                }
                jac += test_deriv / (ni * nj);
            }
        }
        let norm = k.powi(self.n_ways() as i32);
        let (gamma, jac) = (gamma / norm, jac / norm);
        gamma / (self.n_clusters() as f64 * jac * jac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn one_way_split_keeps_clusters_together() {
        let c = Array1::from_shape_fn(40, |i| (i / 4) as i64); // 10 clusters of 4
        let folds = split_clusters(std::slice::from_ref(&c), 3, 7).unwrap();
        assert_eq!(folds.len(), 3);
        let mut seen = [false; 40];
        for cf in &folds {
            for &i in &cf.fold.test {
                assert!(!seen[i]);
                seen[i] = true;
                assert!(cf.test_clusters[0].contains(&c[i]));
            }
            // No cluster is in both training and test rows.
            assert!(cf
                .fold
                .train
                .iter()
                .all(|&i| !cf.test_clusters[0].contains(&c[i])));
            assert_eq!(cf.fold.train.len() + cf.fold.test.len(), 40);
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn two_way_split_is_the_fold_product() {
        let a = Array1::from_shape_fn(36, |i| (i / 6) as i64);
        let b = Array1::from_shape_fn(36, |i| (i % 6) as i64);
        let folds = split_clusters(&[a.clone(), b.clone()], 2, 1).unwrap();
        assert_eq!(folds.len(), 4);
        let total: usize = folds.iter().map(|cf| cf.fold.test.len()).sum();
        assert_eq!(total, 36);
        for cf in &folds {
            for &i in &cf.fold.train {
                assert!(
                    !cf.test_clusters[0].contains(&a[i]) && !cf.test_clusters[1].contains(&b[i])
                );
            }
        }
    }

    #[test]
    fn one_way_variance_with_singleton_clusters_is_the_iid_variance() {
        // With one observation per cluster and one fold assignment, the
        // clustered variance reduces to mean(ψ²) / (n · J²).
        let n = 6;
        let c = Array1::from_shape_fn(n, |i| i as i64);
        let folds = split_clusters(std::slice::from_ref(&c), 2, 3).unwrap();
        let info = ClusterInfo::new(vec![c], vec![folds]).unwrap();
        let psi = array![1.0, -2.0, 0.5, 0.3, -1.0, 1.2];
        let deriv = Array1::from_elem(n, -2.0);
        let iid = psi.dot(&psi) / n as f64 / 4.0 / n as f64;
        assert!((info.variance(psi.view(), deriv.view(), 0) - iid).abs() < 1e-12);
    }
}

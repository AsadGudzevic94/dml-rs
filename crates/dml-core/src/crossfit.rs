use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::data::gather;
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};

/// One train/test split.
#[derive(Debug, Clone, PartialEq)]
pub struct Fold {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
}

/// Shuffled K-fold splitter with a fixed seed, so results are reproducible.
#[derive(Debug, Clone)]
pub struct KFold {
    n_folds: usize,
    seed: u64,
}

impl KFold {
    pub fn new(n_folds: usize, seed: u64) -> Result<Self> {
        if n_folds < 2 {
            return Err(DmlError::InvalidArgument(
                "n_folds must be at least 2".into(),
            ));
        }
        Ok(Self { n_folds, seed })
    }

    pub fn n_folds(&self) -> usize {
        self.n_folds
    }

    /// Split `0..n` into `n_folds` disjoint test sets covering every index once.
    pub fn split(&self, n: usize) -> Result<Vec<Fold>> {
        if n < self.n_folds {
            return Err(DmlError::InvalidArgument(format!(
                "{n} observations is fewer than {} folds",
                self.n_folds
            )));
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut ChaCha8Rng::seed_from_u64(self.seed));

        let base = n / self.n_folds;
        let extra = n % self.n_folds;
        let mut folds = Vec::with_capacity(self.n_folds);
        let mut start = 0;
        for k in 0..self.n_folds {
            let size = base + usize::from(k < extra);
            let mut test = idx[start..start + size].to_vec();
            let mut train: Vec<usize> = idx[..start]
                .iter()
                .chain(&idx[start + size..])
                .copied()
                .collect();
            test.sort_unstable();
            train.sort_unstable();
            folds.push(Fold { train, test });
            start += size;
        }
        Ok(folds)
    }

    /// Stratified split: the rows of each stratum (e.g. treated / control) are
    /// shuffled and dealt round-robin over the folds, so every fold holds about
    /// the same share of each stratum. Fails if a stratum has fewer rows than
    /// there are folds.
    pub fn split_stratified(&self, strata: &[usize]) -> Result<Vec<Fold>> {
        let n = strata.len();
        if n < self.n_folds {
            return Err(DmlError::InvalidArgument(format!(
                "{n} observations is fewer than {} folds",
                self.n_folds
            )));
        }
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, &s) in strata.iter().enumerate() {
            groups.entry(s).or_default().push(i);
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut tests: Vec<Vec<usize>> = vec![Vec::new(); self.n_folds];
        let mut next = 0;
        for (stratum, mut idx) in groups {
            if idx.len() < self.n_folds {
                return Err(DmlError::InvalidArgument(format!(
                    "stratum {stratum} has {} observations, fewer than {} folds",
                    idx.len(),
                    self.n_folds
                )));
            }
            idx.shuffle(&mut rng);
            for i in idx {
                tests[next].push(i);
                next = (next + 1) % self.n_folds;
            }
        }
        Ok(tests
            .into_iter()
            .map(|mut test| {
                test.sort_unstable();
                let mut in_test = vec![false; n];
                for &i in &test {
                    in_test[i] = true;
                }
                let train = (0..n).filter(|&i| !in_test[i]).collect();
                Fold { train, test }
            })
            .collect())
    }
}

/// How observations are split for cross-fitting: `n_rep` independent
/// repetitions of seeded K-fold splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleSplitting {
    pub n_folds: usize,
    pub n_rep: usize,
    pub seed: u64,
}

impl Default for SampleSplitting {
    fn default() -> Self {
        Self {
            n_folds: 5,
            n_rep: 1,
            seed: 0,
        }
    }
}

impl SampleSplitting {
    /// Draw the folds for every repetition. Repetition `r` uses its own seed
    /// derived from `seed`, so repetition 0 equals `KFold::new(n_folds, seed)`.
    pub fn draw(&self, n: usize) -> Result<Vec<Vec<Fold>>> {
        if self.n_rep == 0 {
            return Err(DmlError::InvalidArgument("n_rep must be at least 1".into()));
        }
        (0..self.n_rep as u64)
            .map(|r| {
                // Golden-ratio increment keeps repetition seeds far apart.
                let seed = self
                    .seed
                    .wrapping_add(r.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                KFold::new(self.n_folds, seed)?.split(n)
            })
            .collect()
    }

    /// Cluster-level splits for every repetition (see [`crate::cluster`]).
    pub fn draw_clustered(&self, clusters: &[Array1<i64>]) -> Result<Vec<Vec<crate::ClusterFold>>> {
        if self.n_rep == 0 {
            return Err(DmlError::InvalidArgument("n_rep must be at least 1".into()));
        }
        (0..self.n_rep as u64)
            .map(|r| {
                let seed = self
                    .seed
                    .wrapping_add(r.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                crate::cluster::split_clusters(clusters, self.n_folds, seed)
            })
            .collect()
    }
}

/// Out-of-fold predictions: for each fold, fit a fresh copy of `learner` on the
/// training rows and predict the held-out rows. Every row is predicted by a
/// model that never saw it — the core of DML cross-fitting.
pub fn cross_fit_predict(
    learner: &dyn Regressor,
    x: ArrayView2<'_, f64>,
    target: ArrayView1<'_, f64>,
    folds: &[Fold],
) -> Result<Array1<f64>> {
    if target.len() != x.nrows() {
        return Err(DmlError::DimensionMismatch(
            "cross_fit_predict x/target rows".into(),
        ));
    }
    cross_fit_with(x.nrows(), folds, |fold| {
        let mut model = learner.fresh();
        model.fit(
            x.select(Axis(0), &fold.train).view(),
            gather(target, &fold.train).view(),
        )?;
        model.predict(x.select(Axis(0), &fold.test).view())
    })
}

/// Like [`cross_fit_predict`], but each model is trained only on the training
/// rows where `keep[i]` is true (e.g. controls only); predictions still cover
/// every test row.
pub fn cross_fit_predict_subset(
    learner: &dyn Regressor,
    x: ArrayView2<'_, f64>,
    target: ArrayView1<'_, f64>,
    folds: &[Fold],
    keep: &[bool],
) -> Result<Array1<f64>> {
    cross_fit_predict(learner, x, target, &subset_folds(folds, keep, x.nrows())?)
}

/// Like [`cross_fit_predict_proba`], but each model is trained only on the
/// training rows where `keep[i]` is true; predictions cover every test row.
pub fn cross_fit_predict_proba_subset(
    learner: &dyn Classifier,
    x: ArrayView2<'_, f64>,
    labels: ArrayView1<'_, f64>,
    folds: &[Fold],
    keep: &[bool],
) -> Result<Array1<f64>> {
    cross_fit_predict_proba(learner, x, labels, &subset_folds(folds, keep, x.nrows())?)
}

/// Restrict every training set to the rows where `keep` is true.
fn subset_folds(folds: &[Fold], keep: &[bool], n: usize) -> Result<Vec<Fold>> {
    if keep.len() != n {
        return Err(DmlError::DimensionMismatch("subgroup mask length".into()));
    }
    let subset: Vec<Fold> = folds
        .iter()
        .map(|f| Fold {
            train: f.train.iter().copied().filter(|&i| keep[i]).collect(),
            test: f.test.clone(),
        })
        .collect();
    if subset.iter().any(|f| f.train.is_empty()) {
        return Err(DmlError::InvalidArgument(
            "a training fold has no observations in the requested subgroup".into(),
        ));
    }
    Ok(subset)
}

/// Out-of-fold class-1 probabilities from a [`Classifier`].
pub fn cross_fit_predict_proba(
    learner: &dyn Classifier,
    x: ArrayView2<'_, f64>,
    labels: ArrayView1<'_, f64>,
    folds: &[Fold],
) -> Result<Array1<f64>> {
    if labels.len() != x.nrows() {
        return Err(DmlError::DimensionMismatch(
            "cross_fit_predict_proba x/labels rows".into(),
        ));
    }
    cross_fit_with(x.nrows(), folds, |fold| {
        let mut model = learner.fresh();
        model.fit(
            x.select(Axis(0), &fold.train).view(),
            gather(labels, &fold.train).view(),
        )?;
        model.predict_proba(x.select(Axis(0), &fold.test).view())
    })
}

/// Run `fit_predict` on every fold and scatter each fold's test-set
/// predictions into one length-`n` vector.
///
/// With the `parallel` feature (on by default) folds run on rayon's thread
/// pool; results are identical to the sequential order because every fold
/// writes to disjoint rows.
pub(crate) fn cross_fit_with<F>(n: usize, folds: &[Fold], fit_predict: F) -> Result<Array1<f64>>
where
    F: Fn(&Fold) -> Result<Array1<f64>> + Sync,
{
    #[cfg(feature = "parallel")]
    let preds: Vec<Array1<f64>> = {
        use rayon::prelude::*;
        folds.par_iter().map(&fit_predict).collect::<Result<_>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let preds: Vec<Array1<f64>> = folds.iter().map(&fit_predict).collect::<Result<_>>()?;

    let mut out = Array1::<f64>::from_elem(n, f64::NAN);
    for (fold, pred) in folds.iter().zip(preds) {
        if pred.len() != fold.test.len() {
            return Err(DmlError::DimensionMismatch(
                "learner returned wrong number of predictions".into(),
            ));
        }
        for (&i, &p) in fold.test.iter().zip(pred.iter()) {
            out[i] = p;
        }
    }
    if out.iter().any(|v| v.is_nan()) {
        return Err(DmlError::InvalidArgument(
            "folds do not cover every observation, or a learner returned NaN".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Ols;
    use ndarray::Array2;

    #[test]
    fn folds_partition_all_indices() {
        let folds = KFold::new(3, 7).unwrap().split(10).unwrap();
        let mut seen: Vec<usize> = folds.iter().flat_map(|f| f.test.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<_>>());
        for f in &folds {
            assert!(f.test.iter().all(|i| !f.train.contains(i)));
            assert_eq!(f.train.len() + f.test.len(), 10);
        }
        let sizes: Vec<usize> = folds.iter().map(|f| f.test.len()).collect();
        assert_eq!(sizes, vec![4, 3, 3]);
    }

    #[test]
    fn same_seed_same_split() {
        let a = KFold::new(5, 42).unwrap().split(50).unwrap();
        let b = KFold::new(5, 42).unwrap().split(50).unwrap();
        let c = KFold::new(5, 43).unwrap().split(50).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn repetitions_differ_and_first_matches_kfold() {
        let reps = SampleSplitting {
            n_folds: 4,
            n_rep: 3,
            seed: 9,
        }
        .draw(40)
        .unwrap();
        assert_eq!(reps.len(), 3);
        assert_eq!(reps[0], KFold::new(4, 9).unwrap().split(40).unwrap());
        assert_ne!(reps[0], reps[1]);
        assert_ne!(reps[1], reps[2]);
    }

    #[test]
    fn subset_training_uses_only_kept_rows() {
        // Target is 0 on kept rows and 100 elsewhere: a mean model trained
        // only on kept rows predicts ~0 everywhere.
        let x = Array2::<f64>::zeros((20, 0));
        let keep: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
        let y: Array1<f64> = keep.iter().map(|&k| if k { 0.0 } else { 100.0 }).collect();
        let folds = KFold::new(4, 1).unwrap().split(20).unwrap();
        let pred =
            cross_fit_predict_subset(&Ols::new(), x.view(), y.view(), &folds, &keep).unwrap();
        assert!(pred.iter().all(|p| p.abs() < 1e-12));
    }

    #[test]
    fn stratified_folds_balance_each_stratum() {
        let strata: Vec<usize> = (0..53).map(|i| usize::from(i % 5 == 0)).collect(); // 11 treated
        let folds = KFold::new(5, 3).unwrap().split_stratified(&strata).unwrap();
        let mut seen: Vec<usize> = folds.iter().flat_map(|f| f.test.clone()).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..53).collect::<Vec<_>>());
        for f in &folds {
            let treated = f.test.iter().filter(|&&i| strata[i] == 1).count();
            assert!((2..=3).contains(&treated), "treated per fold: {treated}");
            assert_eq!(f.train.len() + f.test.len(), 53);
        }
        assert!(KFold::new(5, 0)
            .unwrap()
            .split_stratified(&[0, 0, 0, 0, 0, 1])
            .is_err());
    }

    #[test]
    fn rejects_bad_settings() {
        assert!(KFold::new(1, 0).is_err());
        assert!(KFold::new(5, 0).unwrap().split(3).is_err());
    }

    #[test]
    fn cross_fit_recovers_linear_signal() {
        let n = 100;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
        let y: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| 1.0 + 0.5 * r[0] - 2.0 * r[1])
            .collect();
        let folds = KFold::new(5, 1).unwrap().split(n).unwrap();
        let pred = cross_fit_predict(&Ols::new(), x.view(), y.view(), &folds).unwrap();
        assert!((&pred - &y).iter().all(|e| e.abs() < 1e-8));
    }
}

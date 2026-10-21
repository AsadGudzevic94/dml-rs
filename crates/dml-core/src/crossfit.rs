use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::data::gather;
use crate::error::{DmlError, Result};
use crate::learner::Regressor;

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
    let n = x.nrows();
    if target.len() != n {
        return Err(DmlError::DimensionMismatch(
            "cross_fit_predict x/target rows".into(),
        ));
    }
    let mut out = Array1::<f64>::from_elem(n, f64::NAN);
    for fold in folds {
        let mut model = learner.fresh();
        let x_train = x.select(Axis(0), &fold.train);
        let t_train = gather(target, &fold.train);
        model.fit(x_train.view(), t_train.view())?;
        let pred = model.predict(x.select(Axis(0), &fold.test).view())?;
        for (&i, &p) in fold.test.iter().zip(pred.iter()) {
            out[i] = p;
        }
    }
    if out.iter().any(|v| v.is_nan()) {
        return Err(DmlError::InvalidArgument(
            "folds do not cover every observation".into(),
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

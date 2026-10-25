//! Hyperparameter tuning by grid search: choose the best of several
//! candidate learners by K-fold cross-validation (scikit-learn's `KFold` /
//! `StratifiedKFold` folds without shuffling), then refit it on all data.
//!
//! Used as a nuisance learner, the search runs inside every cross-fitting
//! fold on that fold's training data, which corresponds to DoubleML's
//! `tune(..., tune_on_folds=True)`.

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use super::cv::{check_cv, complement, kfold_test_sets, map_folds, stratified_test_sets};
use super::{Classifier, Regressor};
use crate::data::gather;
use crate::error::{DmlError, Result};

/// Index of the smallest score (the first one on ties).
fn argmin(scores: &[f64]) -> usize {
    scores
        .iter()
        .enumerate()
        .fold(
            (0, f64::INFINITY),
            |(bi, bv), (i, &v)| if v < bv { (i, v) } else { (bi, bv) },
        )
        .0
}

fn check_candidates(n: usize) -> Result<()> {
    if n == 0 {
        return Err(DmlError::InvalidArgument(
            "a grid search needs at least one candidate".into(),
        ));
    }
    Ok(())
}

/// Grid search over regression learners, scored by the mean squared error
/// of the held-out folds.
pub struct GridSearchRegressor {
    candidates: Vec<Box<dyn Regressor>>,
    cv: usize,
    best: Option<usize>,
    scores: Vec<f64>,
    fitted: Option<Box<dyn Regressor>>,
}

impl GridSearchRegressor {
    /// `candidates`, e.g. the same learner with different hyperparameters.
    pub fn new(candidates: Vec<Box<dyn Regressor>>) -> Self {
        Self {
            candidates,
            cv: 5,
            best: None,
            scores: Vec::new(),
            fitted: None,
        }
    }

    /// Number of cross-validation folds (default 5).
    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Index of the chosen candidate after fitting.
    pub fn best_index(&self) -> Option<usize> {
        self.best
    }

    /// Cross-validated mean squared error of every candidate.
    pub fn cv_scores(&self) -> &[f64] {
        &self.scores
    }

    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        check_candidates(self.candidates.len())?;
        let n = x.nrows();
        check_cv(n, self.cv)?;
        let tests = kfold_test_sets(n, self.cv);
        let fit = |c: &dyn Regressor, rows: &[usize]| -> Result<Box<dyn Regressor>> {
            let mut m = c.fresh();
            let (xt, yt) = (x.select(Axis(0), rows), gather(y, rows));
            match w {
                Some(w) => m.fit_weighted(xt.view(), yt.view(), gather(w, rows).view())?,
                None => m.fit(xt.view(), yt.view())?,
            }
            Ok(m)
        };
        let mut scores = Vec::with_capacity(self.candidates.len());
        for c in &self.candidates {
            let fold_mse = map_folds(&tests, |test| {
                let m = fit(c.as_ref(), &complement(n, test))?;
                let pred = m.predict(x.select(Axis(0), test).view())?;
                Ok(test
                    .iter()
                    .zip(&pred)
                    .map(|(&i, p)| (y[i] - p).powi(2))
                    .sum::<f64>()
                    / test.len() as f64)
            })?;
            scores.push(fold_mse.iter().sum::<f64>() / fold_mse.len() as f64);
        }
        let best = argmin(&scores);
        let all: Vec<usize> = (0..n).collect();
        self.fitted = Some(fit(self.candidates[best].as_ref(), &all)?);
        self.best = Some(best);
        self.scores = scores;
        Ok(())
    }
}

impl Regressor for GridSearchRegressor {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_impl(x, y, None)
    }

    /// Candidates are fitted with the weights; folds are scored unweighted.
    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.fit_impl(x, y, Some(w))
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.fitted.as_ref().ok_or(DmlError::NotFitted)?.predict(x)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self::new(self.candidates.iter().map(|c| c.fresh()).collect()).cv(self.cv))
    }
}

impl Clone for GridSearchRegressor {
    /// An unfitted copy with the same candidates.
    fn clone(&self) -> Self {
        Self::new(self.candidates.iter().map(|c| c.fresh()).collect()).cv(self.cv)
    }
}

/// Grid search over classifiers, scored by the log loss of the held-out
/// folds (probabilities clipped to `[1e-15, 1 − 1e-15]`).
pub struct GridSearchClassifier {
    candidates: Vec<Box<dyn Classifier>>,
    cv: usize,
    best: Option<usize>,
    scores: Vec<f64>,
    fitted: Option<Box<dyn Classifier>>,
}

impl GridSearchClassifier {
    pub fn new(candidates: Vec<Box<dyn Classifier>>) -> Self {
        Self {
            candidates,
            cv: 5,
            best: None,
            scores: Vec::new(),
            fitted: None,
        }
    }

    /// Number of (stratified) cross-validation folds (default 5).
    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    pub fn best_index(&self) -> Option<usize> {
        self.best
    }

    /// Cross-validated log loss of every candidate.
    pub fn cv_scores(&self) -> &[f64] {
        &self.scores
    }

    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        check_candidates(self.candidates.len())?;
        let n = x.nrows();
        check_cv(n, self.cv)?;
        let tests = stratified_test_sets(y, self.cv)?;
        let fit = |c: &dyn Classifier, rows: &[usize]| -> Result<Box<dyn Classifier>> {
            let mut m = c.fresh();
            let (xt, yt) = (x.select(Axis(0), rows), gather(y, rows));
            match w {
                Some(w) => m.fit_weighted(xt.view(), yt.view(), gather(w, rows).view())?,
                None => m.fit(xt.view(), yt.view())?,
            }
            Ok(m)
        };
        let mut scores = Vec::with_capacity(self.candidates.len());
        for c in &self.candidates {
            let fold_loss = map_folds(&tests, |test| {
                let m = fit(c.as_ref(), &complement(n, test))?;
                let p = m.predict_proba(x.select(Axis(0), test).view())?;
                let loss: f64 = test
                    .iter()
                    .zip(&p)
                    .map(|(&i, &q)| {
                        let q = q.clamp(1e-15, 1.0 - 1e-15);
                        -(y[i] * q.ln() + (1.0 - y[i]) * (1.0 - q).ln())
                    })
                    .sum();
                Ok(loss / test.len() as f64)
            })?;
            scores.push(fold_loss.iter().sum::<f64>() / fold_loss.len() as f64);
        }
        let best = argmin(&scores);
        let all: Vec<usize> = (0..n).collect();
        self.fitted = Some(fit(self.candidates[best].as_ref(), &all)?);
        self.best = Some(best);
        self.scores = scores;
        Ok(())
    }
}

impl Classifier for GridSearchClassifier {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_impl(x, y, None)
    }

    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.fit_impl(x, y, Some(w))
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.fitted
            .as_ref()
            .ok_or(DmlError::NotFitted)?
            .predict_proba(x)
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::new(Self::new(self.candidates.iter().map(|c| c.fresh()).collect()).cv(self.cv))
    }
}

impl Clone for GridSearchClassifier {
    /// An unfitted copy with the same candidates.
    fn clone(&self) -> Self {
        Self::new(self.candidates.iter().map(|c| c.fresh()).collect()).cv(self.cv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::{Lasso, LogisticRegression, Ols};
    use ndarray::Array2;

    #[test]
    fn picks_the_candidate_with_the_smallest_cv_error() {
        // y depends on x0 only; a heavy Lasso penalty underfits.
        let n = 200;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.5);
        let y: Array1<f64> = (0..n)
            .map(|i| 3.0 * x[[i, 0]] + 0.01 * ((i % 5) as f64))
            .collect();
        let mut gs = GridSearchRegressor::new(vec![
            Box::new(Lasso::new(1.0)),
            Box::new(Ols::new()),
            Box::new(Lasso::new(0.5)),
        ]);
        gs.fit(x.view(), y.view()).unwrap();
        assert_eq!(gs.best_index(), Some(1));
        assert_eq!(gs.cv_scores().len(), 3);
        let p = gs.predict(x.view()).unwrap();
        assert!(p.iter().zip(&y).all(|(a, b)| (a - b).abs() < 0.1));
    }

    #[test]
    fn classifier_search_and_errors() {
        let n = 200;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 3 + j * 11) % 23) as f64 / 23.0 - 0.5);
        let y: Array1<f64> = (0..n)
            .map(|i| f64::from(x[[i, 0]] + 0.1 * x[[i, 1]] > 0.0))
            .collect();
        let mut gs = GridSearchClassifier::new(vec![
            Box::new(LogisticRegression::new(0.001)),
            Box::new(LogisticRegression::new(10.0)),
        ]);
        gs.fit(x.view(), y.view()).unwrap();
        assert_eq!(gs.best_index(), Some(1));
        assert!(GridSearchClassifier::new(vec![])
            .fit(x.view(), y.view())
            .is_err());
        assert!(gs.fresh().predict_proba(x.view()).is_err());
    }
}

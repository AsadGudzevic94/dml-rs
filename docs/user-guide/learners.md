# Learners

The nuisance functions are ordinary prediction problems. dml-rs accepts two kinds of learners:

- **Built-in learners**, implemented in Rust. Cross-fitting with them never touches Python, so
  folds run in parallel and the GIL is released.
- **Any scikit-learn compatible estimator** (`fit` / `predict`, and `predict_proba` for
  classifiers). It is cloned for each fold and called from Rust.

## Built-in learners

| Regression | Classification | Matches scikit-learn's |
|---|---|---|
| `Ols(ridge=0.0)` | | `LinearRegression` / `Ridge(alpha=ridge)` |
| `Lasso(alpha)` | | `Lasso(alpha)` |
| `LassoCV()` | | `LassoCV()` |
| | `LogisticRegression(C)` | `LogisticRegression(C)` (L2) |
| | `LogisticRegressionCV()` | `LogisticRegressionCV(scoring="neg_log_loss")` |
| `RandomForestRegressor(...)` | `RandomForestClassifier(...)` | `RandomForest*` (same parameters) |
| `GradientBoostingRegressor(...)` | `GradientBoostingClassifier(...)` | `GradientBoosting*` (same parameters) |

Parameter names follow scikit-learn, so switching between the two is a matter of changing the
import.

- **Linear learners** minimise the same objectives as scikit-learn with the same algorithms and
  stopping rules (coordinate descent for the Lasso, Newton's method for logistic regression), and
  agree with it to numerical precision.
- **Cross-validated learners** use scikit-learn's grids, fold schemes and selection rules:
  `LassoCV` tries 100 penalties on a log scale from the smallest penalty giving an all-zero
  solution down to 1/1000 of it, on 5 contiguous folds, and keeps the one with the lowest mean
  test error; `LogisticRegressionCV` tries 10 values of `C` from `1e-4` to `1e4` on stratified
  folds, scored by log loss (scikit-learn's default from version 1.11; `scoring="accuracy"` is
  available). Both refit on all the data with the chosen penalty, and the fitted attributes
  (`alpha_`, `mse_path_`, `C_`, `scores_`, ...) use scikit-learn's names.
- **Gradient boosting** grows the same trees as scikit-learn. When several splits fit a small node
  equally well, scikit-learn breaks the tie by a random feature order set by its `random_state`;
  dml-rs breaks it deterministically. Predictions on the training data are identical; on new data
  they agree up to the variation scikit-learn itself shows across seeds.
- **Random forests** follow scikit-learn's parameters and bootstrap scheme with their own random
  number generator, so they are statistically equivalent rather than identical.

## scikit-learn estimators

```python
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

model = dml_rs.IRM(HistGradientBoostingRegressor(), HistGradientBoostingClassifier()).fit(X, y, d)
```

Pipelines and tuned search objects (`GridSearchCV`, ...) work too. Python learners are slower than
built-in ones because each fold calls back into Python, but the DML algebra still runs in Rust.

## Comparing learners

Each fit reports the out-of-fold loss of every nuisance learner, on data the learner did not see:
RMSE for regressions and log loss for classifications, in DoubleML's layout (`n_rep` by
`n_treat`). Comparing learners on these losses is a simple, honest way to choose them.

```python
model.nuisance_loss_
# {'ml_g0': array([[1.05]]), 'ml_g1': array([[1.07]]), 'ml_m': array([[0.65]])}

from sklearn.metrics import mean_absolute_error

def mae(y_true, y_pred):
    keep = ~np.isnan(y_true)  # some learners apply only to part of the sample
    return mean_absolute_error(y_true[keep], y_pred[keep])

model.evaluate_learners(metric=mae)
```

See the [clustered data and diagnostics example](../examples/05_clusters_and_diagnostics.ipynb).

## Your own learner in Rust

Implement the `Regressor` (or `Classifier`) trait: `fit`, `predict` (or `predict_proba`) and
`fresh`, which returns an unfitted copy with the same settings for the next fold.
`fit_weighted` is optional and needed only by the R-learner. Learners must be `Send + Sync` so
that folds can be fitted on parallel threads.

```rust
use dml_core::{Regressor, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};

#[derive(Clone)]
struct Mean(f64);

impl Regressor for Mean {
    fn fit(&mut self, _x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.0 = y.mean().unwrap_or(0.0);
        Ok(())
    }
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_elem(x.nrows(), self.0))
    }
    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Mean(0.0))
    }
}
```

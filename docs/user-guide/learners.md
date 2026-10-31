# Learners

The nuisance functions are ordinary prediction problems. dml-rs accepts two kinds of learners:

- **Built-in learners**, implemented in Rust. In Python they are scikit-learn estimators, but
  cross-fitting with them never touches Python, so folds run in parallel and the GIL is released.
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
| `HistGradientBoostingRegressor(...)` | `HistGradientBoostingClassifier(...)` | `HistGradientBoosting*` (same parameters) |
| `RegressionForest(...)` | | R's `grf::regression_forest` (honest, with variances) |

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
- **Histogram gradient boosting** (LightGBM-style) is the learner for large data: every feature is
  binned once into at most 255 quantile bins, and trees are grown best-first on histograms of the
  gradients, so a split costs a pass over 255 bins instead of a sort. It follows scikit-learn's
  `HistGradientBoosting*` step by step (binning, leaf-wise growth with `max_leaf_nodes=31`,
  `min_samples_leaf=20`, missing values sent to the better side, early stopping on a 10%
  validation split above 10 000 samples), and with the same `random_state` its predictions are
  identical to scikit-learn's, bit for bit, also with sample weights and missing values. As a
  nuisance learner it runs without Python and fits the folds in parallel; see
  [Benchmarks](../benchmarks.md).
- **Random forests** follow scikit-learn's parameters and bootstrap scheme with their own random
  number generator, so they are statistically equivalent rather than identical.

### scikit-learn compatibility

In Python the built-in learners are scikit-learn estimators (they derive from scikit-learn's
`BaseEstimator` when it is installed; scikit-learn stays optional). `get_params`, `set_params`
and `clone` work, and so do scikit-learn's tools:

```python
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

search = GridSearchCV(dml_rs.Lasso(), {"alpha": [0.01, 0.1, 1.0]}, cv=5).fit(X, y)
cross_val_score(make_pipeline(StandardScaler(), dml_rs.Ols()), X, y)
# cv: a number of folds, any splitter, or (train, test) pairs
dml_rs.LassoCV(cv=KFold(5, shuffle=True, random_state=0)).fit(X, y)
dml_rs.LassoCV(cv=list(GroupKFold(5).split(X, y, groups))).fit(X, y)
```

- Classifiers are binary and take any two labels (`classes_`); `predict_proba` returns two
  columns and `predict` the more probable class.
- Every learner except `RegressionForest` and `QuantileForest` accepts `sample_weight` in
  `fit`.
- `cv` of `LassoCV`, `LogisticRegressionCV` and the grid searches is a number of folds, a
  scikit-learn splitter (`KFold(shuffle=True)`, `GroupKFold`, ...) or a list of `(train, test)`
  index arrays.
- Inputs are validated as in scikit-learn (2-D numeric arrays or DataFrames, whose column
  names are kept in `feature_names_in_`). Only the histogram gradient boosting learners accept
  missing values (NaN).
- Parameters are checked when `fit` runs, not in the constructor.

They pass scikit-learn's estimator checks (`sklearn.utils.estimator_checks`) with two
exceptions: a fitted grid search cannot be pickled (every other fitted learner can), and where
scikit-learn's own forests and boosting also fail, a sample weight of 2 is not the same as a
duplicated row.

Inside a DML model a built-in learner is still fitted in Rust: the model takes its parameters
and cross-fits the Rust learner directly. The exception is a cross-validated learner whose `cv` is
a splitter, which needs the data; it is fitted through Python, fold by fold.

## scikit-learn estimators

```python
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

model = dml_rs.IRM(HistGradientBoostingRegressor(), HistGradientBoostingClassifier()).fit(X, y, d)
```

Pipelines and tuned search objects (`GridSearchCV`, ...) work too. Python learners are slower than
built-in ones because each fold calls back into Python, but the DML algebra still runs in Rust.

## Tuning

### DoubleML's `tune` and `tune_ml_models`

The models with named nuisance learners (PLR, LPLR, PLPR, IRM, APO, SSM, PLIV, IIVM, DID and
DIDCS) have DoubleML's tuning methods. They take the arguments of `fit` (the data, or a
`DmlData`), tune every nuisance learner on its own training data (for IRM, `ml_g` is tuned
separately for `ml_g0` on the untreated and `ml_g1` on the treated), and set the best parameters
for the next `fit`. The learners can be built-in or scikit-learn estimators; with built-in ones
every fit of the search runs in Rust, and the final `fit` cross-fits fully in Rust. `tune` and
`tune_ml_models` are implemented for PLR, IRM, APO, IIVM, DID, DIDCS, SSM (missing at random) and
PLIV (partial `X` with one instrument, or partial `Z`).

`tune` is a grid search (`search_mode="grid_search"`, scikit-learn's `GridSearchCV`) or a
randomized search (`"randomized_search"`), with `param_grids` keyed by learner name. As in
DoubleML, the search folds are shuffled with NumPy's global random state, so
`np.random.seed(...)` makes it reproducible, and the same seed gives the same parameters as
DoubleML. With `tune_on_folds=True` the search runs on each cross-fitting training fold and the
parameters are fold-specific.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

model = dml_rs.IRM(RandomForestRegressor(n_estimators=200), LogisticRegression())
np.random.seed(42)
model.tune(X, y, d, param_grids={"ml_g": {"max_depth": [3, 5, 8]}, "ml_m": {"C": [0.1, 1, 10]}})
model.params["ml_g1"]   # {'d': [[{'max_depth': 5}, ...]]}: per treatment, repetition and fold
model.fit(X, y, d)
```

`tune_ml_models` uses [Optuna](https://optuna.org) (`pip install optuna`). Each parameter space
is a function of an Optuna trial; a space given for a learner name (`ml_g`) applies to all its
nuisance models, one given for a nuisance name (`ml_g1`) takes precedence. Each nuisance learner
is tuned once on its training data in the whole sample, by cross-validation (`cv`, default five
shuffled folds with `random_state=42`), maximising the negative RMSE (regressors) or negative log
loss (classifiers) unless `scoring_methods` says otherwise. `optuna_settings` takes DoubleML's
settings (`n_trials`, default 100, `sampler`, `timeout`, `study_kwargs`, ..., also per learner or
nuisance name). With `return_tune_res=True` both methods return the search results instead of the
model.

```python
def forest(trial):
    return {"max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20)}

def logit(trial):
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}

res = model.tune_ml_models(X, y, d, ml_param_space={"ml_g": forest, "ml_m": logit},
                           optuna_settings={"n_trials": 50, "sampler": optuna.samplers.TPESampler(seed=1)},
                           return_tune_res=True)
res[0]["ml_g1"].best_params   # OptunaResult per treatment and nuisance model
```

### Parameters per nuisance model

`set_ml_nuisance_params(name, treatment, params)` sets parameters of one nuisance model, as
DoubleML: `name` is one of `model.params_names` (for IRM `ml_g0`, `ml_g1`, `ml_m`), `treatment` a
treatment name (`"d"` for a single unnamed treatment; `None` for all treatments), and `params` a
dict for every fold or, for fold-specific parameters, one list of `n_folds` dicts per repetition.
Later calls update earlier ones. `model.params` shows what is set.

```python
model.set_ml_nuisance_params("ml_g1", "d", {"max_depth": 3})
model.set_ml_nuisance_params("ml_g0", "d", [[{"max_depth": 2}, {"max_depth": 4}, {"max_depth": 3},
                                             {"max_depth": 3}, {"max_depth": 5}]])  # one repetition
```

In Rust, any nuisance model can get its own learner with `nuisance_regressor(name, learner)` or
`nuisance_classifier(name, learner)` (IRM, APO, IIVM, DID, DIDCS and SSM, whose nuisance models
share a learner argument):

```rust
let model = Irm::new(RandomForestRegressor::new(), LogisticRegression::default())
    .nuisance_regressor("ml_g1", RandomForestRegressor::new().max_depth(Some(3)));
```

### Searching inside every fold

`GridSearchRegressor` and `GridSearchClassifier` choose the best of several candidate learners by
5-fold cross-validation (mean squared error, or log loss with stratified folds) and refit it.
Candidates can be built-in or scikit-learn learners. As a nuisance learner the search runs inside
every cross-fitting fold on that fold's training data, which is DoubleML's
`tune(..., tune_on_folds=True)`; with scikit-learn candidates it reproduces `GridSearchCV` with
`scoring="neg_mean_squared_error"` exactly. `LassoCV` and `LogisticRegressionCV` tune their
penalty the same way, faster.

=== "Python"

    ```python
    ml = dml_rs.GridSearchRegressor([dml_rs.RandomForestRegressor(max_depth=d) for d in (3, 5, 8)])
    model = dml_rs.PLR(ml, ml).fit(X, y, d)
    ```

=== "Rust"

    ```rust
    let grid = || {
        let candidates: Vec<Box<dyn Regressor>> = [3, 5, 8]
            .iter()
            .map(|&depth| Box::new(RandomForestRegressor::new().max_depth(Some(depth))) as Box<dyn Regressor>)
            .collect();
        GridSearchRegressor::new(candidates)
    };
    let fit = Plr::new(grid(), grid()).fit(&data)?;
    ```

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

## Stored learners

`fit(..., store_models=True)` keeps every fitted nuisance learner in `models_`, in DoubleML's
layout `models_[name][treatment][repetition][fold]` (models that fit a learner several times per
fold, such as LPLR's double cross-fitting, store a list per fold). The folds are then fitted one
after the other. Available for the models with named nuisance learners.

```python
model.fit(X, y, d, store_models=True)
model.models_["ml_g1"]["d"][0][2].feature_importances_   # repetition 0, fold 2
```

## External predictions

Nuisance predictions computed elsewhere (a model trained on another system, or an expensive model
fitted once and reused) can replace any learner, as DoubleML's `external_predictions`. They must
be out-of-fold predictions for the same sample splits, one column per repetition. The learners
they replace are not fitted and have no out-of-sample loss. Supported by PLR, IRM, APO, PLIV,
IIVM, DID, DIDCS, PLPR, PQ, LPQ and LPLR, with DoubleML's learner names (`ml_l`, `ml_m`, `ml_g0`,
...). LPQ takes all five of its learners or none. LPLR's double cross-fitted `ml_M` and `ml_a` also
need their inner predictions, `ml_M_inner_k` and `ml_a_inner_k` for every outer fold `k` (a fit
reports them in `predictions_`).

=== "Python"

    ```python
    # {treatment: {learner: array (n_obs, n_rep)}}, or {learner: array} for one treatment
    model = dml_rs.PLR(ml_l, ml_m).fit(X, y, d, folds=folds, external_predictions={"ml_m": m_hat})
    ```

=== "Rust"

    ```rust
    let external = ExternalPredictions::new().with(0, "ml_m", m_hat); // (n_obs, n_rep)
    let fit = Plr::new(ml_l, ml_m).fit_with_folds(&data.with_external_predictions(external), &folds)?;
    ```

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

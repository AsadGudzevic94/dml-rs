# Migrating from DoubleML

dml-rs follows DoubleML's models, scores and estimators, so most code moves over by changing how
the data and the learners are passed. On the same sample splits with the same scikit-learn
learners, the estimates, standard errors and intervals agree to `1e-9`; see
[Validation](../validation.md) for what is compared and for the known differences.

Your scikit-learn learners work unchanged, and you can keep them to start with. The speed gain
comes from the learners and the cross-fitting that run in Rust, so switch to `dml_rs.Lasso`,
`dml_rs.RandomForestRegressor` and the like when you want it. See the
[benchmarks](../benchmarks.md).

```sh
pip install dml-rs
```

## The main differences

| | DoubleML | dml-rs |
|---|---|---|
| Data | `DoubleMLData` is required | Arrays, or an optional `DmlData` |
| Where the data goes | The constructor | `fit()`, as in scikit-learn |
| Arguments | `DoubleMLPLR(data, ml_l, ml_m)` | `PLR(ml_l, ml_m).fit(X, y, d)` |
| Argument order | `from_arrays(x, y, d)` | `fit(X, y, d)`, the same order |
| Results | `model.coef`, `model.se` | `model.coef_`, `model.se_` (trailing underscore) |
| Sample splits | Drawn from NumPy's global state | Set by `random_state`, default `0` |
| Your own splits | `set_sample_splitting(smpls)` | `fit(..., folds=smpls)` |

## A first model

=== "DoubleML"

    ```python
    import doubleml as dml
    import numpy as np
    from doubleml.plm.datasets import make_plr_CCDDHNR2018
    from sklearn.linear_model import LassoCV

    np.random.seed(1)
    X, y, d = make_plr_CCDDHNR2018(n_obs=500, dim_x=10, return_type="array")

    data = dml.DoubleMLData.from_arrays(X, y, d)
    ref = dml.DoubleMLPLR(data, LassoCV(), LassoCV(), n_folds=5).fit()
    print(ref.summary)
    ```

=== "dml-rs"

    ```python
    import dml_rs

    model = dml_rs.PLR(LassoCV(), LassoCV(), n_folds=5, random_state=0).fit(X, y, d)
    print(model.summary)
    ```

The two estimates differ a little here because the sample splits differ. To compare them exactly,
reuse DoubleML's splits. This also makes a good first check of your own migration:

```python
same_splits = dml_rs.PLR(LassoCV(), LassoCV()).fit(X, y, d, folds=ref.smpls)
assert np.allclose(same_splits.coef_, ref.coef, rtol=1e-9)
assert np.allclose(same_splits.se_, ref.se, rtol=1e-9)
```

Named columns work as in DoubleML, with `dml_rs.DmlData` in place of `DoubleMLData`:

```python
import pandas as pd

frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])]).assign(y=y, d=d)
data = dml_rs.DmlData(frame, y_col="y", d_cols="d")
named = dml_rs.PLR(LassoCV(), LassoCV()).fit(data)
```

## Model names

| DoubleML | dml-rs |
|---|---|
| `DoubleMLPLR` | `PLR` |
| `DoubleMLIRM` | `IRM` |
| `DoubleMLPLIV` | `PLIV` |
| `DoubleMLIIVM` | `IIVM` |
| `DoubleMLDID`, `DoubleMLDIDCS` | `DID`, `DIDCS` |
| `DoubleMLDIDMulti` | `StaggeredDID` |
| `DoubleMLAPO`, `DoubleMLAPOS` | `APO`, `APOS` |
| `DoubleMLPQ`, `DoubleMLLPQ`, `DoubleMLCVAR`, `DoubleMLQTE` | `PQ`, `LPQ`, `CVAR`, `QTE` |
| `DoubleMLSSM` | `SSM` |
| `DoubleMLLPLR` | `LPLR` |
| `DoubleMLPLPR` | `PLPR` |
| `RDFlex` | `RDFlex` |

The options keep their names: `score`, `n_folds`, `n_rep`, `normalize_ipw` and so on. Two
differences come up often. Propensity clipping is set with `clipping_threshold` rather than
`ps_processor_config`, and calibration with `calibration`. Binary-treatment models take a
classifier for `ml_m`, as in DoubleML.

```python
from doubleml.irm.datasets import make_irm_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

np.random.seed(2)
X_irm, y_irm, d_irm = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")

forest = RandomForestRegressor(n_estimators=50, random_state=0)
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
irm = dml_rs.IRM(forest, classifier, score="ATE", clipping_threshold=0.05, n_rep=2).fit(X_irm, y_irm, d_irm)
```

## After fitting

| Task | DoubleML | dml-rs |
|---|---|---|
| Estimates | `coef`, `se`, `t_stat`, `pval` | `coef_`, `se_`, `t_stat_`, `pval_` |
| Per repetition | `all_coef`, `all_se` | `all_coef_`, `all_se_` |
| Table | `summary` | `summary` |
| Intervals | `confint(level=0.95)` | `confint(level=0.95)` |
| Simultaneous intervals | `bootstrap()`, then `confint(joint=True)` | The same |
| Multiple testing | `p_adjust("romano-wolf")` | The same |
| Nuisance predictions | `predictions` | `predictions_` |
| Nuisance losses | `nuisance_loss` | `nuisance_loss_` |
| Learner evaluation | `evaluate_learners()` | The same |
| Fitted learners | `fit(store_models=True)`, `models` | `fit(store_models=True)`, `models_` |
| Framework | `framework` | `framework` |

Two points need care. `confint()` returns a NumPy array, one row per treatment, rather than a
DataFrame. And `pval_` comes from the aggregated t statistic, so with `n_rep > 1` it can differ
slightly from `framework.pvals` in DoubleML, which takes the median over repetitions.

```python
irm.bootstrap(method="normal", n_boot=500)
intervals = irm.confint(joint=True)
```

## Sensitivity analysis

The names are the same, and `sensitivity_summary` is a property on both sides:

```python
irm.sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0)
print(irm.sensitivity_summary)
```

## Tuning

`tune` and `tune_ml_models` keep their arguments, and they take the fit inputs first because the
data is not in the constructor:

=== "DoubleML"

    ```python
    from sklearn.linear_model import Lasso

    grids = {"ml_l": {"alpha": [0.01, 0.1]}, "ml_m": {"alpha": [0.01, 0.1]}}
    tuned_ref = dml.DoubleMLPLR(dml.DoubleMLData.from_arrays(X, y, d), Lasso(), Lasso())
    tuned_ref.tune(grids)
    tuned_ref.fit()
    ```

=== "dml-rs"

    ```python
    tuned = dml_rs.PLR(Lasso(), Lasso())
    tuned.tune(X, y, d, param_grids=grids)
    tuned.fit(X, y, d)
    ```

`set_ml_nuisance_params(learner, treat_var, params)` works the same way. Treatment names come from
your data, so with plain arrays they are `d`, or `d0`, `d1` and so on, where DoubleML's
`from_arrays` uses `d` or `d1`, `d2`. Tuning needs learners with the scikit-learn API. The built-in
`dml_rs` learners have it too.

## Using the built-in learners

These are drop-in estimators with `fit`, `predict` and `get_params`, so `GridSearchCV`, `Pipeline`
and `clone` work with them:

```python
fast = dml_rs.PLR(dml_rs.LassoCV(), dml_rs.LassoCV()).fit(X, y, d)
```

| scikit-learn | dml-rs |
|---|---|
| `LinearRegression`, `Ridge` | `Ols` |
| `Lasso`, `LassoCV` | `Lasso`, `LassoCV` |
| `LogisticRegression`, `LogisticRegressionCV` | `LogisticRegression`, `LogisticRegressionCV` |
| `RandomForestRegressor`, `RandomForestClassifier` | Same names |
| `GradientBoostingRegressor`, `GradientBoostingClassifier` | Same names |
| `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` | Same names |

`Lasso`, `LassoCV` and histogram boosting reproduce scikit-learn's results. The random forests use a
different random number generator, so their results agree in distribution rather than exactly.

## What behaves differently

- **DoubleML bugs fixed here.** dml-rs differs on purpose where DoubleML 0.11.4 has a known bug, for
  example `normalize_ipw` in SSM. [Validation](../validation.md) lists each case with its upstream
  report.
- **Reproducibility.** Results depend on `random_state`, not on `np.random.seed`. The exception is
  `tune`, which draws its folds from NumPy's global state as DoubleML does.
- **Not-yet-treated controls.** `StaggeredDID` uses DoubleML 0.11.4's rule for pre-treatment periods;
  DoubleML's main branch has changed it.

## Beyond DoubleML

After migrating, these are available on the same fits or data:
[causal forests](../examples/06_causal_forests.ipynb),
[synthetic DiD](../examples/07_synthetic_did.ipynb),
[mediation](../examples/08_mediation.ipynb),
[policy learning](../examples/09_policy_learning.ipynb),
[prediction intervals for individual effects](../examples/10_prediction_intervals.ipynb) and
[Lee bounds](../examples/11_lee_bounds.ipynb).

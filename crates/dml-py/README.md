# dml-rs

[![PyPI](https://img.shields.io/pypi/v/dml-rs.svg?label=PyPI)](https://pypi.org/project/dml-rs/)
[![Python versions](https://img.shields.io/pypi/pyversions/dml-rs.svg)](https://pypi.org/project/dml-rs/)
[![CI](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/AsadGudzevic94/dml-rs#license)

**Fast double / debiased machine learning for causal inference, implemented in Rust.**

dml-rs implements the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097) with a scikit-learn style API.
Every model is tested for numerical agreement with the reference
[DoubleML](https://docs.doubleml.org) package, and the Rust core makes estimation typically
3–24× faster.

## Installation

```sh
pip install dml-rs
```

Prebuilt wheels are available for Linux, macOS and Windows (Python 3.9 or newer).

## Quick start

```python
import numpy as np
import dml_rs

# Simulated data: the true effect of d on y is 0.5; X confounds both
rng = np.random.default_rng(42)
n = 2000
X = rng.normal(size=(n, 5))
d = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(size=n)
y = 0.5 * d + np.cos(X[:, 0]) + X[:, 1] ** 2 + rng.normal(size=n)

model = dml_rs.PLR(
    ml_l=dml_rs.RandomForestRegressor(min_samples_leaf=5),  # learner for E[y | X]
    ml_m=dml_rs.RandomForestRegressor(min_samples_leaf=5),  # learner for E[d | X]
    n_folds=5,
)
model.fit(X, y, d)
print(model.summary)
```

```
       coef   std err          t         P>|t|     2.5 %    97.5 %
d  0.516919  0.029474  17.538215  7.319031e-69  0.459151  0.574687
```

Any scikit-learn estimator works as a learner. The built-in learners
(`GradientBoostingRegressor`, `RandomForestRegressor`, `Lasso`, `Ols`, `LogisticRegression`, and
the corresponding classifiers) run the whole estimation in Rust, in parallel and without the GIL.

## Models

| Class | Target parameter |
|---|---|
| `PLR` | Partially linear regression: effect of a continuous or binary treatment |
| `IRM` | Interactive regression model: ATE / ATTE of a binary treatment; `cate` and `gate` |
| `PLIV` | Partially linear IV regression |
| `IIVM` | Interactive IV model: LATE with a binary instrument |
| `DID`, `DIDCS` | Difference-in-differences for panels and repeated cross-sections |
| `StaggeredDID` | Staggered adoption: ATT(g, t) with group, time and event-study aggregation |
| `SLearner`, `TLearner`, `XLearner`, `RLearner`, `DRLearner` | Heterogeneous (conditional average) treatment effects |

All estimators support repeated sample splitting, multiplier bootstrap with simultaneous
confidence bands (`bootstrap`, `confint(joint=True)`), multiple-testing adjustment (`p_adjust`),
pandas summaries, and the scikit-learn parameter protocol (`get_params`, `set_params`, `clone`).
PLR, IRM and DID models also offer sensitivity analysis for omitted confounders
(`sensitivity_analysis`, `sensitivity_summary`, `sensitivity_benchmark`). Pass `cluster=` to
`fit` for cluster-robust inference with one- or two-way clustered data, and inspect the nuisance
learners with `nuisance_loss_` and `evaluate_learners`.

## Documentation

Full feature overview, benchmarks and validation details:
<https://github.com/AsadGudzevic94/dml-rs>

## License

Licensed under either of the Apache License 2.0 or the MIT license, at your option.

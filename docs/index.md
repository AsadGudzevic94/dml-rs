# dml-rs

**Double / debiased machine learning for causal inference, in Rust with Python bindings.**

dml-rs implements the double / debiased machine learning (DML) framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097) for estimating causal effects when
many confounders must be controlled for. It combines Neyman-orthogonal scores, flexible
machine-learning models for the nuisance functions and cross-fitting, and delivers valid standard
errors, confidence intervals and hypothesis tests.

The core is written in Rust and exposed to Python with a scikit-learn style API. Every model is
tested for numerical agreement with the reference [DoubleML](https://docs.doubleml.org) package.

<div class="grid cards" markdown>

- **Validated**

    Estimates, standard errors, confidence intervals, bootstrap inference and aggregations agree
    with DoubleML to `1e-9` on identical data and sample splits. [How it is tested](validation.md)

- **Fast**

    A pure-Rust core with parallel cross-fitting and built-in learners: typically 3–25× faster than
    DoubleML with scikit-learn learners.

- **Complete**

    Partially linear and interactive models, instrumental variables, difference-in-differences
    with staggered adoption, heterogeneous effects, sensitivity analysis and clustered data.

- **Rust and Python**

    Use it as the Rust crate `dml-core` or as the Python package `dml-rs`, which accepts any
    scikit-learn estimator.

</div>

## Install

=== "Python"

    ```sh
    pip install dml-rs
    ```

=== "Rust"

    ```sh
    cargo add dml-core
    ```

## A first estimate

```python
import numpy as np
import dml_rs

rng = np.random.default_rng(42)
n = 2000
X = rng.normal(size=(n, 5))
d = np.sin(X[:, 0]) + X[:, 1] ** 2 + rng.normal(size=n)              # treatment
y = 0.5 * d + np.cos(X[:, 0]) + X[:, 1] ** 2 + rng.normal(size=n)    # true effect 0.5

model = dml_rs.PLR(
    ml_l=dml_rs.RandomForestRegressor(min_samples_leaf=5),   # learner for E[y | X]
    ml_m=dml_rs.RandomForestRegressor(min_samples_leaf=5),   # learner for E[d | X]
).fit(X, y, d)
print(model.summary)
```

Continue with [Getting started](getting-started.md), or browse the [examples](examples/01_getting_started.ipynb).

## What is included

| Model | Target parameter | Python | Rust |
|---|---|---|---|
| Partially linear regression | Effect of a continuous or binary treatment | `PLR` | `Plr` |
| Logistic partially linear regression | Log odds ratio of a treatment on a binary outcome | `LPLR` | `Lplr` |
| Partially linear panel regression | Effect in static panels with correlated unit effects | `PLPR` | `Plpr` |
| Interactive regression model | ATE and ATTE of a binary treatment | `IRM` | `Irm` |
| Multi-valued treatments | Average potential outcomes and contrasts between levels | `APO`, `APOS` | `Apo`, `Apos` |
| Quantile effects | Quantile and CVaR treatment effects, potential and compliers' quantiles | `QTE`, `PQ`, `LPQ`, `CVAR` | `Qte`, `Pq`, `Lpq`, `Cvar` |
| Sample selection | ATE when outcomes are missing at random or non-ignorably | `SSM` | `Ssm` |
| Partially linear IV regression | Effect of an endogenous treatment with an instrument | `PLIV` | `Pliv` |
| Interactive IV model | Local average treatment effect (LATE) | `IIVM` | `Iivm` |
| Difference-in-differences | ATT for panels and repeated cross-sections | `DID`, `DIDCS` | `Did`, `DidCs` |
| Staggered difference-in-differences | Group-time effects ATT(g, t) and aggregations | `StaggeredDID` | `StaggeredDid` |
| Regression discontinuity | Sharp and fuzzy RD effects with flexible covariate adjustment | `RDFlex` | `RdFlex` |
| Heterogeneous effects | CATE meta-learners, best linear predictor, group effects, policy trees | `DRLearner`, `PLR.cate`, `IRM.policy_tree`, … | `DrLearner`, `Blp`, `PolicyTree`, … |

All models share the same inference toolkit: repeated sample splitting, multiplier bootstrap with
simultaneous confidence bands, multiple-testing adjustment, cluster-robust standard errors,
sensitivity analysis for omitted confounders and out-of-sample learner diagnostics. See the
[user guide](user-guide/models.md).

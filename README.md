<h1 align="center">dml-rs</h1>

<p align="center">
  <b>Double / debiased machine learning for causal inference, in Rust with Python bindings</b>
</p>

<p align="center">
  <a href="https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml"><img src="https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/dml-rs/"><img src="https://img.shields.io/pypi/v/dml-rs.svg?label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/dml-rs/"><img src="https://img.shields.io/pypi/pyversions/dml-rs.svg" alt="Python versions"></a>
  <a href="https://crates.io/crates/dml-core"><img src="https://img.shields.io/crates/v/dml-core.svg?label=crates.io" alt="crates.io"></a>
  <a href="https://docs.rs/dml-core"><img src="https://img.shields.io/docsrs/dml-core?label=docs.rs" alt="docs.rs"></a>
  <a href="#license"><img src="https://img.shields.io/crates/l/dml-core.svg" alt="License: MIT OR Apache-2.0"></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/dml-rs/">Python package</a> ·
  <a href="https://crates.io/crates/dml-core">Rust crate</a> ·
  <a href="https://docs.rs/dml-core">API documentation</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="https://github.com/AsadGudzevic94/dml-rs/releases">Releases</a>
</p>

---

**dml-rs** implements the double / debiased machine learning (DML) framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097) for estimating causal effects
when many confounders must be controlled for. It combines Neyman-orthogonal scores, flexible
machine-learning models for the nuisance functions and cross-fitting, and delivers valid
standard errors, confidence intervals and hypothesis tests.

The core is written in Rust and exposed to Python with a scikit-learn style API. Every model is
tested for numerical agreement with the reference [DoubleML](https://docs.doubleml.org) package.

## Why dml-rs

- **Validated.** Estimates, standard errors, confidence intervals, bootstrap inference and
  aggregations agree with DoubleML to `1e-9` on identical data and sample splits.
- **Fast.** A pure-Rust core with parallel cross-fitting and built-in learners: typically **3–24×
  faster** than DoubleML with scikit-learn learners ([benchmarks](#performance)).
- **Complete.** Partially linear and interactive models, instrumental variables,
  difference-in-differences (including staggered adoption) and heterogeneous effects.
- **Rust and Python.** Use it as a Rust crate or as a Python package that accepts any
  scikit-learn estimator.

## Main features

| Model | Target parameter | Rust | Python |
|---|---|---|---|
| Partially linear regression | Effect of a continuous or binary treatment (partialling-out or IV-type score) | `Plr` | `PLR` |
| Interactive regression model | ATE and ATTE of a binary treatment | `Irm` | `IRM` |
| Partially linear IV regression | Effect of an endogenous treatment with an instrument | `Pliv` | `PLIV` |
| Interactive IV model | Local average treatment effect (LATE) with a binary instrument | `Iivm` | `IIVM` |
| Difference-in-differences | ATT for panel data and repeated cross-sections | `Did`, `DidCs` | `DID`, `DIDCS` |
| Staggered difference-in-differences | Group-time effects ATT(g, t), aggregated by cohort, period or event time | `StaggeredDid` | `StaggeredDID` |
| Heterogeneous effects | CATE meta-learners (S, T, X, R, DR); best linear predictor and group effects (GATE) | `DrLearner`, `Blp`, … | `DRLearner`, `IRM.cate`, … |

**Inference:** standard errors and confidence intervals, multiple treatments, repeated sample
splitting, multiplier bootstrap (normal, Bayes, wild) with simultaneous confidence bands, and
Romano–Wolf, Holm and Bonferroni multiple-testing adjustment.

**Learners:** gradient boosting and random forests (both implemented in Rust; the boosting
reproduces scikit-learn's predictions), OLS / ridge, Lasso and logistic regression, or any
scikit-learn estimator from Python. New learners and new orthogonal scores plug in through the
`Regressor`, `Classifier` and `LinearScoreModel` traits.

## Installation

**Python** (3.9 or newer; prebuilt wheels for Linux, macOS and Windows):

```sh
pip install dml-rs
```

**Rust:**

```sh
cargo add dml-core
```

## Quick start

### Python

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

Any scikit-learn estimator can be used instead, for example
`ml_l=sklearn.ensemble.HistGradientBoostingRegressor()`. The built-in learners
(`dml_rs.GradientBoostingRegressor`, `dml_rs.RandomForestRegressor`, `dml_rs.Lasso`, …) run the
whole estimation in Rust, in parallel and without the GIL.

### Rust

```rust
use dml_core::{DmlData, GradientBoostingRegressor, Plr};

let data = DmlData::new(y, d, x)?;
let learner = GradientBoostingRegressor::new().n_estimators(200).learning_rate(0.05);
let fit = Plr::new(learner.clone(), learner)
    .n_folds(5)
    .n_rep(10) // repeated sample splitting, aggregated by the median
    .seed(42)
    .fit(&data)?;

println!("{fit}");
let ci = fit.confint(0.95)?;
```

### More examples

<details>
<summary><b>Average treatment effect of a binary treatment (IRM)</b></summary>

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

irm = dml_rs.IRM(ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier(), score="ATE")
irm.fit(X, y, d)
irm.summary
```
</details>

<details>
<summary><b>Instrumental variables (PLIV, IIVM)</b></summary>

```python
pliv = dml_rs.PLIV(dml_rs.Lasso(alpha=0.01), dml_rs.Lasso(alpha=0.01), dml_rs.Lasso(alpha=0.01))
pliv.fit(X, y, d, z)

# Local average treatment effect with a binary instrument
late = dml_rs.IIVM(dml_rs.GradientBoostingRegressor(),
                   dml_rs.GradientBoostingClassifier(),
                   dml_rs.GradientBoostingClassifier())
late.fit(X, y, d, z)
```
</details>

<details>
<summary><b>Difference-in-differences and staggered adoption (event study)</b></summary>

```python
# Two periods, panel data: y is each unit's change between the periods
did = dml_rs.DID(dml_rs.Ols(), dml_rs.LogisticRegression()).fit(X, y_post - y_pre, d)

# Staggered adoption from a long panel (g = first treated period, np.inf = never treated)
stag = dml_rs.StaggeredDID(dml_rs.Ols(), dml_rs.LogisticRegression(), control_group="not_yet_treated")
stag.fit_frame(df, y_col="y", id_col="id", t_col="t", g_col="g", x_cols=["x1", "x2"])
stag.summary                              # every group-time effect ATT(g, t)
stag.aggregate("eventstudy").summary      # dynamic effects; also "group" and "time"

# Repeated cross-sections instead of a panel
dml_rs.StaggeredDID(dml_rs.Ols(), dml_rs.LogisticRegression(), panel=False)
```
</details>

<details>
<summary><b>Heterogeneous treatment effects (CATE)</b></summary>

```python
dr = dml_rs.DRLearner(dml_rs.GradientBoostingRegressor(),
                      dml_rs.GradientBoostingClassifier(),
                      dml_rs.Ols()).fit(X, y, d)
dr.effect(X_new)                    # estimated CATE for new covariates
dr.blp(basis).summary               # best linear predictor with robust standard errors

irm.cate(basis).confint(basis_grid) # CATE inference from an IRM fit, as in DoubleML
irm.gate(groups).confint()          # group average treatment effects
```
</details>

<details>
<summary><b>Several treatments and joint inference</b></summary>

```python
model = dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()).fit(X, y, D)  # D with one column per treatment
model.bootstrap(method="normal", n_boot=1000)
model.confint(joint=True)           # simultaneous confidence bands
model.p_adjust("romano-wolf")       # also "holm" and "bonferroni"
```
</details>

## Performance

Partially linear regression with 5-fold cross-fitting, run on the same machine (16 cores).
DoubleML uses scikit-learn learners with `n_jobs=-1` where available.

| Nuisance learner | n × p | dml-rs | DoubleML | Speed-up |
|---|---|---|---|---|
| Linear regression | 10 000 × 20 | 7.5 ms | 54 ms | ~7× |
| Random forest (100 trees) | 10 000 × 20 | 4.4 s | 12.4 s | ~3× |
| Gradient boosting (200 stages) | 10 000 × 20 | 10.0 s | 84.9 s | ~8.5× |

<details>
<summary>All benchmark sizes</summary>

| Nuisance learner | n × p | dml-rs | DoubleML |
|---|---|---|---|
| Linear regression | 1 000 × 10 | 0.52 ms | 12.4 ms |
| Linear regression | 100 000 × 50 | 241 ms | 1 244 ms |
| Random forest | 2 000 × 10 | 0.30 s | 1.82 s |
| Gradient boosting | 2 000 × 10 | 0.87 s | 8.1 s |

Scripts: [`parity/bench_doubleml.py`](parity/bench_doubleml.py),
[`parity/bench_forest.py`](parity/bench_forest.py),
[`parity/bench_boosting.py`](parity/bench_boosting.py) and `cargo bench -p dml-core`.
In every comparison the estimates agree.
</details>

## Validation

[`parity/generate_fixtures.py`](parity/generate_fixtures.py) runs DoubleML 0.11.4 on fixed data
and fixed sample splits and stores the results; the Rust test suite refits every case on the same
folds. Nuisance predictions, coefficients, standard errors, p-values (raw and adjusted),
confidence intervals, bootstrap bands, BLP / GATE inference and staggered-DiD aggregations must
agree to `1e-9` (`1e-7` for iterative learners). Monte-Carlo tests additionally check that
confidence intervals reach their nominal coverage.

<details>
<summary>Validated configurations</summary>

| Fixtures | Coverage |
|---|---|
| `plr_ols`, `plr_ridge`, `plr_lasso`, `plr_ivtype_rep3` | PLR with partialling-out and IV-type scores, several learners, repeated splits |
| `plr_boot_bayes`, `plr_boot_wild`, `plr_multi_rep3_boot` | Bootstrap methods, multiple treatments, joint intervals, Romano–Wolf |
| `irm_ate`, `irm_atte`, `irm_ate_normalized`, `irm_cate_gate` | IRM (ATE / ATTE, normalised IPW), BLP and GATE |
| `pliv_ols`, `iivm_late_rep2`, `iivm_no_always_takers` | PLIV and IIVM, including one-sided non-compliance |
| `did_panel_*`, `did_cs_*` | Two-period DiD for panels and cross-sections, all score / normalisation variants |
| `staggered_*`, `staggered_cs_*` | Staggered DiD: every ATT(g, t) and all aggregations, panels and cross-sections |

The Python test suite additionally checks the bindings against DoubleML and scikit-learn
(including random forests and gradient boosting).
</details>

## Citation

If you use dml-rs in your work, please cite the package and the underlying method:

```bibtex
@software{gudzevic_dml_rs,
  author  = {Gudzevic, Asad},
  title   = {dml-rs: Double / debiased machine learning in Rust and Python},
  url     = {https://github.com/AsadGudzevic94/dml-rs},
  year    = {2026}
}

@article{chernozhukov2018double,
  author  = {Chernozhukov, Victor and Chetverikov, Denis and Demirer, Mert and Duflo, Esther and
             Hansen, Christian and Newey, Whitney and Robins, James},
  title   = {Double/debiased machine learning for treatment and structural parameters},
  journal = {The Econometrics Journal},
  volume  = {21},
  number  = {1},
  pages   = {C1--C68},
  year    = {2018},
  doi     = {10.1111/ectj.12097}
}
```

## Contributing

Bug reports and feature requests are welcome in the
[issue tracker](https://github.com/AsadGudzevic94/dml-rs/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the test and parity suites, and
the release process.

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An object-oriented
  implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6.
- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods.
  *Journal of Econometrics*, 225(2), 200–230.
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences estimators.
  *Journal of Econometrics*, 219(1), 101–122.
- Semenova, V., & Chernozhukov, V. (2021). Debiased machine learning of conditional average treatment
  effects and other causal functions. *The Econometrics Journal*, 24(2), 264–289.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous
  treatment effects using machine learning. *PNAS*, 116(10), 4156–4165.
- Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects.
  *Biometrika*, 108(2), 299–319.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects.
  *Electronic Journal of Statistics*, 17(2), 3008–3049.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at
your option.

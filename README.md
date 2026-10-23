# dml-rs

[![crates.io](https://img.shields.io/crates/v/dml-core.svg)](https://crates.io/crates/dml-core)
[![PyPI](https://img.shields.io/pypi/v/dml-rs.svg)](https://pypi.org/project/dml-rs/)
[![Python versions](https://img.shields.io/pypi/pyversions/dml-rs.svg)](https://pypi.org/project/dml-rs/)
[![docs.rs](https://img.shields.io/docsrs/dml-core)](https://docs.rs/dml-core)
[![CI](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/dml-core.svg)](#license)

Double / debiased machine learning (DML) for causal inference, in Rust and Python.

DML (Chernozhukov et al., 2018) estimates causal effects when many confounders
must be controlled for. It combines three ingredients:

1. **Neyman-orthogonal scores**, so small errors in the nuisance models do not bias the effect estimate
2. **Flexible ML learners** for the nuisance functions (e.g. `E[Y|X]`, `E[D|X]`)
3. **Cross-fitting**, so every observation is predicted by a model that never saw it

`dml-rs` is a fast, dependency-light Rust implementation modelled on the reference
[DoubleML](https://docs.doubleml.org) package and tested for numerical parity with it.

> **Status:** 0.1.0 released. The API may still change in 0.x releases.

## Installation

**Rust** ([crates.io](https://crates.io/crates/dml-core), API docs on [docs.rs](https://docs.rs/dml-core)):

```sh
cargo add dml-core
```

**Python** ([PyPI](https://pypi.org/project/dml-rs/); prebuilt wheels for Windows, macOS and
Linux, Python 3.9+):

```sh
pip install dml-rs
```

## Features

- **Models:** partially linear regression (`Plr`, partialling-out and IV-type scores),
  interactive regression model (`Irm`, ATE and ATTE, propensity clipping, normalised IPW),
  partially linear IV (`Pliv`), interactive IV model (`Iivm`, LATE with a binary instrument),
  doubly robust difference-in-differences for panel data (`Did`) and repeated cross-sections
  (`DidCs`), and **staggered adoption** (`StaggeredDid`: group-time effects ATT(g, t) aggregated by
  cohort, calendar time or event time, for panels and repeated cross-sections)
- **Inference:** DML2 estimation, standard errors, t-statistics, p-values, Wald confidence
  intervals; multiple treatments; repeated sample splitting with median aggregation
- **Bootstrap:** normal, Bayes and wild multiplier bootstrap; simultaneous (joint) confidence
  intervals; Romano–Wolf, Holm and Bonferroni multiple-testing adjustment
- **Heterogeneous effects:** S-, T-, X-, R- and DR-learners for the CATE; best linear predictor
  (BLP) of the CATE and group average treatment effects (GATE) with HC0 inference, pointwise and
  simultaneous confidence bands
- **Learners:** `GradientBoostingRegressor` / `GradientBoostingClassifier` (reproduce scikit-learn's
  predictions), `RandomForestRegressor` / `RandomForestClassifier` (parallel, seeded CART forests),
  `Ols` (with optional ridge), `Lasso` (coordinate descent), `LogisticRegression` (Newton), or any
  type implementing the `Regressor` / `Classifier` traits (with optional weighted fitting)
- **Extensible:** implement `LinearScoreModel` to add a new orthogonal score and reuse the
  whole cross-fitting and inference engine
- **Fast:** folds fitted in parallel with rayon; no BLAS or C dependencies
- **Python:** `pip install`-able bindings (`dml_rs`) with a scikit-learn style API that accepts
  any scikit-learn estimator as a nuisance learner

## Roadmap

- [x] **Phase 1 – Foundation:** data container, `Regressor` / `Classifier` traits, seeded K-fold cross-fitting, OLS/ridge learner
- [x] **Phase 2 – PLR:** partially linear regression (DML2), partialling-out score, standard errors, confidence intervals
- [x] **Phase 3 – Validation:** numerical parity tests against Python DoubleML, benchmarks
- [x] **Phase 4 – More models:** IRM (ATE/ATTE), PLIV, multiplier bootstrap, repeated sample splitting
- [x] **Phase 5 – Python:** PyO3 + maturin bindings with a scikit-learn-style API
- [x] **Phase 6 – CATE:** meta-learners (S/T/X/R/DR), best linear predictor and GATE inference

## Layout

```
crates/
  dml-core/   # Rust library: data, learners, cross-fitting, models, inference
  dml-py/     # Python bindings (PyO3 + maturin), package `dml_rs`
parity/       # DoubleML fixture generator and benchmark script
```

## Examples

### Partially linear regression

```rust
use dml_core::{DmlData, Lasso, Plr};

// y: outcome, d: treatment, x: confounders (ndarray Array1 / Array2)
let data = DmlData::new(y, d, x)?;
let fit = Plr::new(Lasso::new(0.01), Lasso::new(0.01)) // learners for E[Y|X] and E[D|X]
    .n_folds(5)
    .n_rep(10)          // repeated sample splitting, median-aggregated
    .seed(42)
    .fit(&data)?;

println!("{fit}");
// PLR (partialling out)  |  n_obs = 2000, n_folds = 5, n_rep = 10
//            coef    std err        t    P>|t|      2.5 %     97.5 %
// d        0.5007     0.0236   21.259   0.0000     0.4545     0.5468
let ci = fit.confint(0.95)?;            // shape (n_treat, 2)
let l_hat = fit.prediction("ml_l", 0, 0); // cross-fitted E[Y|X], repetition 0
```

### Average treatment effect with a binary treatment

```rust
use dml_core::{Irm, IrmScore, RandomForestClassifier, RandomForestRegressor};

let fit = Irm::new(
    RandomForestRegressor::new().min_samples_leaf(5),  // E[Y|D,X]
    RandomForestClassifier::new().min_samples_leaf(5), // P(D=1|X)
)
.score(IrmScore::Ate) // or IrmScore::Atte
.clipping_threshold(0.01)
.fit(&data)?;
```

### Instrumental variables

```rust
use dml_core::{DmlData, Ols, Pliv};

let data = DmlData::new(y, d, x)?.with_instrument(z)?;
let fit = Pliv::new(Ols::new(), Ols::new(), Ols::new()).fit(&data)?; // E[Y|X], E[Z|X], E[D|X]
```

### Local average treatment effect (binary instrument)

```rust
use dml_core::{DmlData, Iivm, LogisticRegression, Ols};

let data = DmlData::new(y, d, x)?.with_instrument(z)?; // z: 0/1 instrument
let fit = Iivm::new(Ols::new(), LogisticRegression::default(), LogisticRegression::default())
    .subgroups(false, true) // one-sided non-compliance: no always-takers
    .fit(&data)?;
```

### Difference-in-differences

```rust
use dml_core::{Did, DidCs, DmlData, LogisticRegression, Ols};

// Panel: y is each unit's outcome change between the two periods.
let att = Did::new(Ols::new(), LogisticRegression::default()).fit(&DmlData::new(dy, d, x)?)?;

// Repeated cross-sections: each row observed once, t = 1 in the post-period.
let data = DmlData::new(y, d, x)?.with_time_indicator(t)?;
let att = DidCs::new(Ols::new(), LogisticRegression::default()).fit(&data)?;
```

### Staggered adoption (event study)

```rust
use dml_core::{Aggregation, ControlGroup, LogisticRegression, Ols, PanelData, StaggeredDid};

// Long format: one row per unit and period; g = first treated period (f64::INFINITY = never).
let data = PanelData::from_long(&id, &t, &y, &g, &x)?;
let fit = StaggeredDid::new(Ols::new(), LogisticRegression::default())
    .control_group(ControlGroup::NotYetTreated)
    .fit(&data)?;
println!("{fit}");                                  // every ATT(g, t)
let es = fit.aggregate(Aggregation::EventStudy)?;  // dynamic effects by event time
let bands = fit.att().bootstrap(dml_core::BootstrapMethod::Normal, 999, 0)?.joint_confint(0.95)?;

// Repeated cross-sections: every row is its own observation.
let rcs = dml_core::RepeatedCrossSections::new(y_rows, t_rows, g_rows, x_rows)?;
let fit_cs = StaggeredDid::new(Ols::new(), LogisticRegression::default()).fit(&rcs)?;
```

### Several treatments, joint inference

```rust
use dml_core::{BootstrapMethod, DmlData, Ols, PAdjust, Plr};

let data = DmlData::with_treatments(y, d_matrix, x)?.with_treatment_names(["price", "ads"])?;
let fit = Plr::new(Ols::new(), Ols::new()).fit(&data)?;
let boot = fit.bootstrap(BootstrapMethod::Normal, 1000, 7)?;
let joint_ci = boot.joint_confint(0.95)?;   // simultaneous coverage for all treatments
let p_rw = boot.romano_wolf();              // Romano–Wolf adjusted p-values
let p_holm = fit.p_adjust(PAdjust::Holm);
```

### Heterogeneous effects (CATE)

```rust
use dml_core::{DrLearner, Irm, LogisticRegression, Ols, XLearner};

// DR-learner: cross-fitted AIPW pseudo-outcomes regressed on X
let dr = DrLearner::new(Ols::new(), LogisticRegression::default(), Ols::new()).fit(&data)?;
let tau_hat = dr.effect(x_new.view())?;               // τ̂(x) for new covariates
let blp = dr.blp(basis.view())?;                      // best linear predictor, HC0 inference
let bands = blp.joint_confint_at(grid.view(), 0.95, 1000, 0)?; // simultaneous bands

// X-learner (Künzel et al.)
let x = XLearner::new(Ols::new(), Ols::new(), LogisticRegression::default()).fit(&data)?;

// From an IRM fit, as in DoubleML: BLP and group effects
let irm = Irm::new(Ols::new(), LogisticRegression::default()).n_rep(5).fit(&data)?;
let gate = irm.gate(group_dummies.view())?;
```

## Python

```sh
pip install dml-rs
```

```python
import dml_rs
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

plr = dml_rs.PLR(ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor(), n_folds=5, n_rep=3)
plr.fit(X, y, d)              # X, d may be NumPy arrays or pandas objects
plr.summary                   # pandas DataFrame: coef, std err, t, P>|t|, CI
plr.confint(level=0.95)
plr.predictions_["ml_l"]      # cross-fitted nuisance predictions (n, n_rep, n_treat)

irm = dml_rs.IRM(ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier(), score="ATTE").fit(X, y, d)
pliv = dml_rs.PLIV(dml_rs.Ols(), dml_rs.Ols(), dml_rs.Ols()).fit(X, y, d, z)
late = dml_rs.IIVM(RandomForestRegressor(), RandomForestClassifier(), RandomForestClassifier()).fit(X, y, d, z)
att = dml_rs.DID(dml_rs.Ols(), dml_rs.LogisticRegression()).fit(X, y_post - y_pre, d)
att_cs = dml_rs.DIDCS(dml_rs.Ols(), dml_rs.LogisticRegression()).fit(X, y, d, t)

# Staggered adoption from a long panel (id, t, g = first treated period, np.inf = never)
stag = dml_rs.StaggeredDID(dml_rs.Ols(), dml_rs.LogisticRegression(), control_group="not_yet_treated")
stag.fit_frame(df, y_col="y", id_col="id", t_col="t", g_col="g", x_cols=["x1", "x2"])
stag.summary                              # every ATT(g, t)
stag.aggregate("eventstudy").summary      # or "group" / "time"
dml_rs.StaggeredDID(dml_rs.Ols(), dml_rs.LogisticRegression(), panel=False)  # repeated cross-sections

plr.bootstrap(method="normal", n_boot=1000)
plr.confint(joint=True); plr.p_adjust("romano-wolf")

# Heterogeneous effects
dr = dml_rs.DRLearner(RandomForestRegressor(), RandomForestClassifier(), dml_rs.Ols()).fit(X, y, d)
dr.effect(X_new)                       # τ̂(x)
dr.blp(basis).summary                  # BLP coefficients with robust standard errors
irm.cate(basis).confint(basis_grid)    # DoubleML-compatible CATE / GATE inference
irm.gate(groups).confint()
```

Nuisance learners can be the built-in Rust learners (`dml_rs.GradientBoostingRegressor`,
`dml_rs.GradientBoostingClassifier`, `dml_rs.RandomForestRegressor`, `dml_rs.RandomForestClassifier`, `dml_rs.Ols`, `dml_rs.Lasso`, `dml_rs.LogisticRegression`) — the
whole fit then runs in Rust, in parallel, without the GIL — or any object with scikit-learn's `fit` / `predict` (`predict_proba` for
classifiers). Python learners are cloned per fold with `sklearn.base.clone` and called
from Rust's worker threads. Estimators follow scikit-learn conventions (`get_params`,
`set_params`, `sklearn.base.clone`, fitted attributes ending in `_`), and `fit(..., folds=...)`
accepts explicit sample splits.

The Python test suite (`pytest crates/dml-py/tests`) checks the bindings against
DoubleML with scikit-learn learners (including random forests) on identical folds.

## Validation against DoubleML

`parity/generate_fixtures.py` runs Python [DoubleML](https://docs.doubleml.org) 0.11.4
on fixed data and fixed sample splits and stores the results as JSON fixtures in
`crates/dml-core/tests/fixtures/`. `cargo test` refits each case in Rust on the same folds
and requires the nuisance predictions, per-repetition and aggregated coefficients and
standard errors, t-statistics, p-values (raw, Bonferroni, Holm), confidence intervals,
bootstrap joint intervals and Romano–Wolf p-values to agree to within `1e-9`
(`1e-7` for the iterative Lasso / logistic learners).

| Fixture | Covers |
|---|---|
| `plr_ols`, `plr_ridge`, `plr_lasso` | PLR partialling-out with OLS, ridge and Lasso nuisances |
| `plr_ivtype_rep3` | PLR IV-type score, 3 repeated sample splits |
| `plr_boot_bayes`, `plr_boot_wild` | Bayes and wild multiplier bootstrap |
| `plr_multi_rep3_boot` | 3 treatments, 3 repetitions, normal bootstrap, joint CI, Romano–Wolf |
| `irm_ate`, `irm_atte`, `irm_ate_normalized` | IRM ATE / ATTE with logistic propensity, normalised IPW |
| `pliv_ols` | PLIV partialling-out, 2 repetitions |
| `irm_cate_gate` | BLP of the CATE (coefficients, HC0 errors, pointwise bands) and GATEs from IRM |
| `iivm_late_rep2`, `iivm_no_always_takers` | IIVM (LATE), 2 repetitions; one-sided non-compliance |
| `did_panel_*`, `did_cs_*` | DID for panels and repeated cross-sections, observational / experimental scores, with and without in-sample normalisation |
| `staggered_*` | Staggered DID (`DoubleMLDIDMulti`): every ATT(g, t), group / time / event-study aggregation and overall effects; never- and not-yet-treated controls, repeated splits, universal base period with anticipation |
| `staggered_cs_*` | Staggered DID for repeated cross-sections (`DoubleMLDIDMulti(panel=False)`), observational and experimental scores |

To regenerate the fixtures:

```sh
python -m venv .venv
.venv/Scripts/pip install -r parity/requirements.txt   # .venv/bin/pip on Linux/macOS
.venv/Scripts/python parity/generate_fixtures.py
```

## Performance

PLR with OLS nuisances and 5-fold cross-fitting (`cargo bench -p dml-core`, versus
`parity/bench_doubleml.py` with scikit-learn `LinearRegression`, same machine):

| n × p        | dml-rs  | DoubleML (Python) | speed-up |
|--------------|---------|-------------------|----------|
| 1 000 × 10   | 0.52 ms | 12.4 ms           | ~24×     |
| 10 000 × 20  | 7.5 ms  | 54 ms             | ~7×      |
| 100 000 × 50 | 241 ms  | 1 244 ms          | ~5×      |

PLR with **random-forest** nuisances (100 trees, `min_samples_leaf=5`, all features per split,
5 folds; `parity/bench_forest.py`, 16 cores, scikit-learn with `n_jobs=-1`):

| n × p       | dml-rs native forest | dml-rs + scikit-learn forest | DoubleML + scikit-learn forest |
|-------------|----------------------|------------------------------|--------------------------------|
| 2 000 × 10  | 0.30 s               | 1.68 s                       | 1.82 s (~6× slower)            |
| 10 000 × 20 | 4.4 s                | 10.5 s                       | 12.4 s (~3× slower)            |

All three estimate θ within 0.02 of the true 0.5 on this nonlinear design.

PLR with **gradient-boosting** nuisances (200 stages, learning rate 0.05, depth 3, 5 folds;
`parity/bench_boosting.py`). dml-rs boosting reproduces scikit-learn's `GradientBoostingRegressor`
predictions to ~1e-14, so the difference is purely speed:

| n × p       | dml-rs native boosting | dml-rs + scikit-learn boosting | DoubleML + scikit-learn boosting |
|-------------|------------------------|--------------------------------|----------------------------------|
| 2 000 × 10  | 0.87 s                 | 2.8 s                          | 8.1 s (~9× slower)               |
| 10 000 × 20 | 10.0 s                 | 28.0 s                         | 84.9 s (~8.5× slower)            |

Folds (and trees) are fitted in parallel with [rayon](https://docs.rs/rayon) (the default
`parallel` feature; disable with `default-features = false`). Results do not depend on the
number of threads.

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo bench -p dml-core

# Python bindings (inside a virtualenv with parity/requirements.txt + pytest + maturin)
cd crates/dml-py && maturin develop --release && pytest tests
```

## Releasing

1. Bump `version` in the root `Cargo.toml` and move the `Unreleased` notes in `CHANGELOG.md`
   under the new version.
2. Commit, then tag and push: `git tag -a v0.2.0 -m "dml-rs 0.2.0" && git push origin v0.2.0`.
3. Approve the two deployments in the Actions tab. `release-crate.yml` publishes `dml-core` to
   crates.io and `wheels.yml` publishes `dml-rs` to PyPI, both through Trusted Publishing
   (no API tokens). The tag must match the `Cargo.toml` version or both workflows stop.

## References

- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods.
  *Journal of Econometrics*, 225(2), 200–230.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
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
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An object-oriented
  implementation of double machine learning in Python. *JMLR*, 23(53), 1–6.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

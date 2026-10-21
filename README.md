# dml-rs

Double / debiased machine learning (DML) for causal inference, in Rust.

DML (Chernozhukov et al., 2018) estimates causal effects when many confounders
must be controlled for. It combines three ingredients:

1. **Neyman-orthogonal scores**, so small errors in the nuisance models do not bias the effect estimate
2. **Flexible ML learners** for the nuisance functions (e.g. `E[Y|X]`, `E[D|X]`)
3. **Cross-fitting**, so every observation is predicted by a model that never saw it

`dml-rs` is a fast, dependency-light Rust implementation modelled on the reference
[DoubleML](https://docs.doubleml.org) package and tested for numerical parity with it.

> **Status: pre-release.** The API may still change before 0.1.0 is published.

## Features

- **Models:** partially linear regression (`Plr`, partialling-out and IV-type scores),
  interactive regression model (`Irm`, ATE and ATTE, propensity clipping, normalised IPW),
  partially linear IV (`Pliv`)
- **Inference:** DML2 estimation, standard errors, t-statistics, p-values, Wald confidence
  intervals; multiple treatments; repeated sample splitting with median aggregation
- **Bootstrap:** normal, Bayes and wild multiplier bootstrap; simultaneous (joint) confidence
  intervals; Romano–Wolf, Holm and Bonferroni multiple-testing adjustment
- **Learners:** `Ols` (with optional ridge), `Lasso` (coordinate descent), `LogisticRegression`
  (Newton), or any type implementing the `Regressor` / `Classifier` traits
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
- [ ] **Phase 6 – CATE:** meta-learners (S/T/X/R/DR)

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
use dml_core::{Irm, IrmScore, LogisticRegression, Ols};

let fit = Irm::new(Ols::new(), LogisticRegression::default()) // E[Y|D,X], P(D=1|X)
    .score(IrmScore::Ate)          // or IrmScore::Atte
    .clipping_threshold(0.01)
    .fit(&data)?;
```

### Instrumental variables

```rust
use dml_core::{DmlData, Ols, Pliv};

let data = DmlData::new(y, d, x)?.with_instrument(z)?;
let fit = Pliv::new(Ols::new(), Ols::new(), Ols::new()).fit(&data)?; // E[Y|X], E[Z|X], E[D|X]
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

## Python

```sh
pip install ./crates/dml-py        # builds the Rust extension with maturin
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

plr.bootstrap(method="normal", n_boot=1000)
plr.confint(joint=True); plr.p_adjust("romano-wolf")
```

Nuisance learners can be the built-in Rust learners (`dml_rs.Ols`, `dml_rs.Lasso`,
`dml_rs.LogisticRegression`) — the whole fit then runs in Rust, in parallel, without the
GIL — or any object with scikit-learn's `fit` / `predict` (`predict_proba` for
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

Folds are fitted in parallel with [rayon](https://docs.rs/rayon) (the default
`parallel` feature; disable with `default-features = false`).

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
cargo bench -p dml-core

# Python bindings (inside a virtualenv with parity/requirements.txt + pytest + maturin)
cd crates/dml-py && maturin develop --release && pytest tests
```

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An object-oriented
  implementation of double machine learning in Python. *JMLR*, 23(53), 1–6.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

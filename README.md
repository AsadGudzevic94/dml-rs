# dml-rs

Double / debiased machine learning (DML) for causal inference, in Rust.

DML (Chernozhukov et al., 2018) estimates causal effects when many confounders
must be controlled for. It combines three ingredients:

1. **Neyman-orthogonal scores**, so small errors in the nuisance models do not bias the effect estimate
2. **Flexible ML learners** for the nuisance functions (e.g. `E[Y|X]`, `E[D|X]`)
3. **Cross-fitting**, so every observation is predicted by a model that never saw it

The goal is a fast, dependency-light Rust core with Python bindings, validated
against the reference [DoubleML](https://docs.doubleml.org) package.

> **Status: early development.** The API will change before 0.1.0 is published.

## Roadmap

- [x] **Phase 1 – Foundation:** data container, `Regressor` / `Classifier` traits, seeded K-fold cross-fitting, OLS/ridge learner
- [x] **Phase 2 – PLR:** partially linear regression (DML2), partialling-out score, standard errors, confidence intervals
- [x] **Phase 3 – Validation:** numerical parity tests against Python DoubleML, benchmarks
- [ ] **Phase 4 – More models:** IRM (ATE/ATTE), PLIV, multiplier bootstrap, repeated sample splitting
- [ ] **Phase 5 – Python:** PyO3 + maturin bindings with a scikit-learn-style API
- [ ] **Later:** CATE meta-learners (S/T/X/R/DR)

## Layout

```
crates/
  dml-core/   # Rust library: data, learners, cross-fitting, models
```

## Example: partially linear regression

```rust
use dml_core::{DmlData, Ols, Plr};

// y: outcome, d: treatment, x: confounders (ndarray Array1 / Array2)
let data = DmlData::new(y, d, x)?;
let res = Plr::new(Ols::new(), Ols::new()) // learners for E[Y|X] and E[D|X]
    .n_folds(5)
    .seed(42)
    .fit(&data)?;

println!("{res}");
//       coef    std err        t    P>|t|      2.5 %     97.5 %
//     0.5007     0.0236   21.259   0.0000     0.4545     0.5468
let (lo, hi) = res.confint(0.95)?;
```

## Validation against DoubleML

`parity/generate_fixtures.py` runs Python [DoubleML](https://docs.doubleml.org) 0.11.4
on fixed data and fixed sample splits and stores the results as JSON fixtures in
`crates/dml-core/tests/fixtures/`. `cargo test` refits each case in Rust on the same folds
and requires the nuisance predictions, coefficient, standard error, t-statistic,
p-value and confidence interval to agree to within `1e-9`.

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
```

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An object-oriented
  implementation of double machine learning in Python. *JMLR*, 23(53), 1–6.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

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
- [ ] **Phase 2 – PLR:** partially linear regression (DML2), partialling-out score, standard errors, confidence intervals
- [ ] **Phase 3 – Validation:** numerical parity tests against Python DoubleML, benchmarks
- [ ] **Phase 4 – More models:** IRM (ATE/ATTE), PLIV, multiplier bootstrap, repeated sample splitting
- [ ] **Phase 5 – Python:** PyO3 + maturin bindings with a scikit-learn-style API
- [ ] **Later:** CATE meta-learners (S/T/X/R/DR)

## Layout

```
crates/
  dml-core/   # Rust library: data, learners, cross-fitting, models
```

## Example (Phase 1 building blocks)

```rust
use dml_core::{cross_fit_predict, KFold, Ols};
use ndarray::{Array1, Array2};

let x = Array2::from_shape_fn((100, 2), |(i, j)| ((i * 7 + j * 3) % 11) as f64);
let y: Array1<f64> = x.rows().into_iter().map(|r| 1.0 + 0.5 * r[0] - 2.0 * r[1]).collect();

let folds = KFold::new(5, 42)?.split(x.nrows())?;
let y_hat = cross_fit_predict(&Ols::new(), x.view(), y.view(), &folds)?;
# Ok::<(), dml_core::DmlError>(())
```

## Development

```sh
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018).
  Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML – An object-oriented
  implementation of double machine learning in Python. *JMLR*, 23(53), 1–6.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

# dml-core

[![crates.io](https://img.shields.io/crates/v/dml-core.svg)](https://crates.io/crates/dml-core)
[![docs.rs](https://img.shields.io/docsrs/dml-core)](https://docs.rs/dml-core)
[![CI](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml)

Double / debiased machine learning (Chernozhukov et al., 2018) for causal inference in Rust,
modelled on and numerically validated against the [DoubleML](https://docs.doubleml.org) package.

- **Models:** partially linear regression (`Plr`, partialling-out and IV-type scores),
  interactive regression model (`Irm`, ATE / ATTE), partially linear IV (`Pliv`)
- **Inference:** DML2 estimation, standard errors, confidence intervals, multiple treatments,
  repeated sample splitting, multiplier bootstrap, joint intervals, Romano–Wolf / Holm / Bonferroni
- **Heterogeneous effects:** S-, T-, X-, R- and DR-learners; best linear predictor of the CATE and
  group average treatment effects with robust inference
- **Learners:** `RandomForestRegressor` / `RandomForestClassifier`, `Ols` (ridge), `Lasso`,
  `LogisticRegression`, or your own via the `Regressor` / `Classifier` traits
- **Fast:** pure Rust, no BLAS; folds fitted in parallel with rayon (`parallel` feature, default)

```sh
cargo add dml-core
```

```rust
use dml_core::{DmlData, Lasso, Plr};

let data = DmlData::new(y, d, x)?;
let fit = Plr::new(Lasso::new(0.01), Lasso::new(0.01)) // learners for E[Y|X] and E[D|X]
    .n_folds(5)
    .n_rep(10)
    .seed(42)
    .fit(&data)?;
println!("{fit}");
```

Every model is tested for parity with Python DoubleML on identical data and sample splits
(agreement to 1e-9). Python bindings, benchmarks and the full documentation live in the
[repository](https://github.com/AsadGudzevic94/dml-rs).

## License

Licensed under either of [Apache License 2.0](https://github.com/AsadGudzevic94/dml-rs/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/AsadGudzevic94/dml-rs/blob/main/LICENSE-MIT), at your option.

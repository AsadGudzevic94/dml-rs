# dml-core

[![crates.io](https://img.shields.io/crates/v/dml-core.svg)](https://crates.io/crates/dml-core)
[![docs.rs](https://img.shields.io/docsrs/dml-core)](https://docs.rs/dml-core)
[![CI](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/AsadGudzevic94/dml-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/dml-core.svg)](https://github.com/AsadGudzevic94/dml-rs#license)

**Double / debiased machine learning for causal inference in Rust.**

`dml-core` implements the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097): Neyman-orthogonal scores,
flexible nuisance learners and cross-fitting, with valid standard errors, confidence intervals
and tests. Every model is tested for numerical agreement with the reference
[DoubleML](https://docs.doubleml.org) package. It is the core of the
[`dml-rs`](https://pypi.org/project/dml-rs/) Python package.

```sh
cargo add dml-core
```

## Example

```rust
use dml_core::{DmlData, GradientBoostingRegressor, Plr};

let data = DmlData::new(y, d, x)?;
let learner = GradientBoostingRegressor::new().n_estimators(200).learning_rate(0.05);
let fit = Plr::new(learner.clone(), learner) // learners for E[Y|X] and E[D|X]
    .n_folds(5)
    .n_rep(10) // repeated sample splitting, aggregated by the median
    .seed(42)
    .fit(&data)?;

println!("{fit}");
let ci = fit.confint(0.95)?;
```

## Features

- **Models:** partially linear regression (`Plr`), interactive regression (`Irm`, ATE / ATTE),
  partially linear IV (`Pliv`), interactive IV (`Iivm`, LATE), difference-in-differences for
  panels and repeated cross-sections (`Did`, `DidCs`) and staggered adoption (`StaggeredDid`, with
  group, time and event-study aggregation)
- **Heterogeneous effects:** S-, T-, X-, R- and DR-learners; best linear predictor of the CATE and
  group average treatment effects (`Blp`)
- **Inference:** multiple treatments, repeated sample splitting, multiplier bootstrap with
  simultaneous confidence bands, Romano–Wolf / Holm / Bonferroni adjustment
- **Learners:** `GradientBoostingRegressor` / `GradientBoostingClassifier`,
  `RandomForestRegressor` / `RandomForestClassifier`, `Ols` (ridge), `Lasso`,
  `LogisticRegression`, or your own through the `Regressor` / `Classifier` traits
- **Extensible:** implement `LinearScoreModel` to add a new orthogonal score and reuse the whole
  cross-fitting and inference engine
- **Fast and lightweight:** pure Rust without BLAS; folds and trees are fitted in parallel with
  rayon (`parallel` feature, on by default); minimum supported Rust version 1.80

See the [repository](https://github.com/AsadGudzevic94/dml-rs) for benchmarks, validation
details and Python usage.

## License

Licensed under either of [Apache License 2.0](https://github.com/AsadGudzevic94/dml-rs/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/AsadGudzevic94/dml-rs/blob/main/LICENSE-MIT), at your option.

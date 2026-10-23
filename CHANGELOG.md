# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-10-23

### Added
- `Iivm`: interactive IV model for the local average treatment effect (LATE) with a binary
  instrument; always-/never-taker subgroup switches for one-sided non-compliance.
- `Did` (panel data) and `DidCs` (repeated cross-sections): doubly robust difference-in-differences
  for the ATT (Sant'Anna & Zhao, 2020) with observational and experimental scores and optional
  in-sample normalisation. `DmlData::with_time_indicator` for the period indicator.
- `cross_fit_predict_proba_subset`.
- Python: `dml_rs.IIVM`, `dml_rs.DID`, `dml_rs.DIDCS`.
- DoubleML parity fixtures for IIVM and all DID score / normalisation variants.

### Changed
- `ScoreInput` gains a time-indicator field `t` and is now `#[non_exhaustive]`; custom
  `LinearScoreModel` implementations that destructure it need a trailing `..`.

### Notes
- `Iivm::normalize_ipw` normalises with the instrument, whereas DoubleML 0.11 normalises with
  the treatment; results differ from DoubleML only when this option is enabled (reported upstream
  as [DoubleML#411](https://github.com/DoubleML/doubleml-for-py/issues/411)).

## [0.2.0] - 2026-10-23

### Added
- `RandomForestRegressor` and `RandomForestClassifier`: CART forests in pure Rust with bootstrap,
  random feature subsets (`MaxFeatures`), depth and leaf-size limits, sample weights and
  feature importances. Trees are grown in parallel and are reproducible for a given seed
  regardless of the number of threads. Also exposed in Python as `dml_rs.RandomForestRegressor`
  and `dml_rs.RandomForestClassifier` (about 3–6× faster than DoubleML with scikit-learn
  forests for PLR).
- CI job checking the minimum supported Rust version (1.80).

## [0.1.0] - 2026-10-23

First release.

### Added
- `DmlData` with one or more treatments and an optional instrument.
- Seeded K-fold cross-fitting with repeated sample splitting, fitted in parallel (rayon).
- Models: `Plr` (partialling-out and IV-type scores), `Irm` (ATE / ATTE, propensity clipping,
  normalised IPW), `Pliv`; extensible through the `LinearScoreModel` trait.
- Inference: standard errors, t-statistics, p-values, confidence intervals; multiplier bootstrap
  (normal, Bayes, wild), joint confidence intervals, Romano–Wolf, Holm and Bonferroni adjustment.
- CATE meta-learners (S, T, X, R, DR) and best-linear-predictor / GATE inference.
- Learners: `Ols` (with ridge), `Lasso`, `LogisticRegression`; weighted fitting.
- Python package `dml-rs` with a scikit-learn style API accepting any scikit-learn estimator.
- Numerical parity tests against Python DoubleML 0.11.4.

[Unreleased]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AsadGudzevic94/dml-rs/releases/tag/v0.1.0

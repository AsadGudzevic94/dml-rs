# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `HistGradientBoostingRegressor` and `HistGradientBoostingClassifier`: histogram-based gradient
  boosting (LightGBM-style binning and leaf-wise growth) for large data, with scikit-learn's
  parameters, missing values, sample weights and early stopping. With the same `random_state` the
  predictions are identical to scikit-learn's `HistGradientBoosting*`. Rust: `EarlyStopping`.
- Benchmarks page in the documentation and `parity/bench_all.py`, timing dml-rs against DoubleML
  with the matching scikit-learn learners from 10 000 to 1 000 000 observations.

## [0.20.0] - 2026-10-27

### Added
- Automatic debiased machine learning (Chernozhukov, Newey and Singh, 2022): `AutoDml` (Python
  `AutoDML`) estimates any mean of a linear functional of the regression `E[Y | D, X]`, with the
  Riesz representer learned from the functional by the Lasso on a dictionary (`RieszLasso`,
  penalty by cross-validation of the Riesz loss). Built-in functionals: ATE, average derivative,
  shift effect, policy effect; your own through `CustomFunctional` (Python: a function returning
  the evaluation points and weights) or the `LinearFunctional` trait; your own dictionary or Riesz
  learner through `Dictionary::Custom` and the `RieszLearner` trait. Inference, clustering,
  sensitivity analysis and `cate` / `gate` work as for the other models.

## [0.19.0] - 2026-10-26

### Added
- Hyperparameter tuning as DoubleML (Python): `tune` (grid or randomized search, on the whole
  sample or on every cross-fitting training fold) and `tune_ml_models` (Optuna), for PLR, IRM,
  APO, IIVM, DID, DIDCS, SSM (missing at random) and PLIV. With the same random state both choose
  the same parameters as DoubleML. `OptunaResult` holds the Optuna studies.
- `set_ml_nuisance_params`, `params`, `params_names` and `learner_names` on PLR, LPLR, PLPR, IRM,
  APO, SSM, PLIV, IIVM, DID and DIDCS: parameters per nuisance model, treatment and (optionally)
  fold, as DoubleML.
- `fit(..., store_models=True)` keeps the fitted nuisance learners in `models_`, in DoubleML's
  layout.
- Callable `score` functions for PLR, IRM, PLIV and IIVM, as DoubleML.
- Rust: learners per nuisance model (`nuisance_regressor` / `nuisance_classifier` on `Irm`, `Apo`,
  `Iivm`, `Did`, `DidCs` and `Ssm`), `NuisanceLearners`; `set_parallel_folds` to fit folds one
  after the other; `DmlFit::from_linear_scores` for score elements computed elsewhere. Python:
  `copy.copy` / `copy.deepcopy` of the built-in learners.

## [0.18.0] - 2026-10-26

### Added
- The leverage-adjusted cluster-robust RD variances of `rdrobust`: CRV2 (Bell and McCaffrey, 2002)
  and CRV3 (Pustejovsky and Tipton, 2018), in estimation (for `h = b` and `h ≠ b`) and bandwidth
  selection. `rd::Vce` gains `Cr1`, `Cr2` and `Cr3`; with cluster labels `Hc2` / `Hc3` switch to
  `Cr2` / `Cr3` as in `rdrobust` (previously an error). Python `RDFlex(vce="cr2", cluster=...)`.
  Twelve new cases in the `rd_options` fixture match `rdrobust` to `1e-9`; RDFlex matches DoubleML.
- Clustered staggered DiD: `PanelData::with_clusters` (per unit) and
  `RepeatedCrossSections::with_clusters` (per row); Python `StaggeredDID.fit(..., cluster=...)`,
  `fit_frame(..., cluster_col=...)` and `DmlData(cluster_cols=...)`. Each ATT(g, t) assigns whole
  clusters to folds; standard errors, aggregations and sensitivity bounds are cluster-robust. An
  extension (DoubleML's `DoubleMLDIDMulti` has no clustered version), checked by simulation.

## [0.17.0] - 2026-10-26

### Added
- `dml_rs.DmlData`, a data container naming the roles of a DataFrame's columns (outcome,
  treatments, covariates, instruments, clusters, time, selection indicator, unit identifier,
  running variable), as DoubleML's data classes; every estimator's `fit` accepts it
  (`model.fit(data)`), and `DmlData.from_arrays` uses DoubleML's column names. The simulated data
  sets take `return_type="DmlData"`.
- `fetch_401K` and `fetch_bonus` (`dml_rs.datasets`): the 401(k) and Pennsylvania bonus data sets
  of DoubleML's tutorials, downloaded and prepared as DoubleML (including the bonus data's
  polynomial features).
- Plots (matplotlib): `StaggeredDID.plot_effects` (the ATT(g, t) by cohort, with pre-treatment,
  anticipation and post-treatment periods, or sensitivity bounds and robustness values),
  `DIDAggregation.plot_effects` and `PolicyTree.plot_tree`, as DoubleML's. Rust:
  `PolicyTree::nodes` (`PolicyNode`).

## [0.16.0] - 2026-10-26

### Added
- Simulated data sets: the data generating processes of DoubleML's 17 `make_*` functions. Rust:
  `dml_core::datasets` (`PlrCcddhnr2018`, `PlrTurrell2018`, `ConfoundedPlr`, `PlivChs2015`,
  `PlivMultiwayClusterCkms2021`, `LplrLzz2020`, `PlprCp2025`, `IrmData`, `ConfoundedIrm`,
  `IrmDiscreteTreatments`, `IivmData`, `SsmData`, `HeterogeneousData`, `DidSz2020`, `DidCs2021`,
  `SimpleRdd`): structs with DoubleML's parameters and defaults and `generate(seed)`, with
  conversions to `DmlData`, `StaticPanel`, `PanelData` and `RepeatedCrossSections`. Python:
  `dml_rs.datasets` with DoubleML's function names and signatures (`return_type="array"` or
  `"DataFrame"`, `random_state`). Draws come from a seeded ChaCha generator; the tests compare the
  distributions with DoubleML's generators.

## [0.15.0] - 2026-10-26

### Added
- The RD back end now ports `rdrobust` 2.1.1's options: `dml_core::rd::rd_estimate` and
  `rd_bandwidths` with `RdOptions` (polynomial orders `p` / `q`, `Kernel`, all ten `BwSelect`
  bandwidth selectors, `Vce` nearest-neighbour or HC0–HC3, `nnmatch`, `level`, `MassPoints`,
  `bwcheck`, `bwrestrict`, `stdvars`, `scaleregul`, `sharpbw`, `rho`, `scalepar`, `covs_drop`) and
  `RdData` with covariates and cluster labels (cluster-robust `cr1`). `rdrobust` and `rdbwselect`
  remain as shortcuts with the defaults. A fixture with 42 option sets matches `rdrobust` to `1e-9`.
- RDFlex passes these options to the RD stage, as DoubleML's `RDFlex(**kwargs)`:
  `RdFlex::rd_options`, `rd_covariates`, `rd_clusters`, and a classifier for a binary outcome
  (`RdFlex::sharp_binary`, `fuzzy_binary`, `from_boxed_binary`). Python: `RDFlex(..., p=2,
  kernel="epa", covs=Z, cluster=g, ...)`, and a classifier `ml_g` for a binary outcome. Four parity
  fixtures; the Python tests match DoubleML to `1e-9`.

## [0.14.0] - 2026-10-26

### Added
- Framework arithmetic, as DoubleML's `DoubleMLFramework`: `&a + &b`, `&a - &b`, `c * &fit` and
  `-&fit` on `DmlFit` (the operator traits `Add`, `Sub`, `Mul<f64>` and `Neg`), and the public
  `DmlFit::linear_combination(weights, names)`, all with combined scores so inference accounts for
  the correlation between the estimates; sensitivity bounds add up with absolute weights.
  `DmlFit::concat` no longer requires equal fold counts. Python: `Framework` (`model.framework`,
  `+`, `-`, `*`, `linear_combination`), `dml_rs.concat`, and `CausalContrast` and
  `PotentialQuantiles` are frameworks. A parity fixture and Python tests match DoubleML to `1e-9`.
- Sensitivity analysis for staggered DiD (panels and repeated cross-sections), as DoubleML's
  `DoubleMLDIDMulti`: every ATT(g, t), the aggregations (Python
  `aggregate(...).aggregated_frameworks` / `overall_aggregated_framework`) and benchmarks
  (`StaggeredDid::sensitivity_benchmark`, `PanelData::without_covariates`,
  `RepeatedCrossSections::without_covariates`; Python `sensitivity_benchmark`). The six staggered
  fixtures now check the bounds, robustness values and benchmarks.
- External predictions for PQ, LPQ and LPLR. LPLR fits report the inner predictions of `ml_M` and
  `ml_a` (`ml_M_inner_k`, `ml_a_inner_k`), which external `ml_M` / `ml_a` need too, as DoubleML.

## [0.13.0] - 2026-10-25

### Added
- PLIV with several instruments (`DmlData::with_instruments`; residualised instruments combined by
  two-stage least squares), the IV-type score for one instrument (`PlivScore`, `ml_g`), and
  DoubleML's partial Z and partial XZ variants (`PlivVariant`, `Pliv::partial_z`,
  `Pliv::partial_xz`; Python `PLIV(partial=...)`). Five parity fixtures; the Python tests match
  DoubleML to `1e-9`.
- Propensity calibration, as DoubleML's `ps_processor_config`: `Calibration::Isotonic` /
  `IsotonicCrossFitted` (Python `calibration="isotonic"` / `"isotonic_cv"`) calibrates the
  cross-fitted propensity scores with an isotonic regression before clipping, for IRM, APO, APOS,
  IIVM, SSM, PQ, LPQ, CVaR, QTE, DID, DIDCS and staggered DID. New `propensity` module with
  `IsotonicRegression`, a port of scikit-learn's. Six parity fixtures; the Python tests match
  DoubleML to `1e-9`.

- External nuisance predictions, as DoubleML's `external_predictions`
  (`DmlData::with_external_predictions(ExternalPredictions)`, `StaticPanel::with_external_predictions`;
  Python `fit(..., external_predictions=...)`) for PLR, IRM, APO, PLIV, IIVM, DID, DIDCS and PLPR:
  supplied learners are not fitted and report no loss. Models read them through
  `ScoreInput::predictions_or`; `ScoreInput` gains `treatment` and `external`.

- Tuning by grid search: `GridSearchRegressor` (cross-validated mean squared error) and
  `GridSearchClassifier` (log loss, stratified folds) choose the best of several candidate learners
  and refit it; as nuisance learners they tune inside every cross-fitting fold, as DoubleML's
  `tune(..., tune_on_folds=True)`. Python classes of the same names accept built-in and
  scikit-learn candidates and reproduce `GridSearchCV(scoring="neg_mean_squared_error")`.

- `sensitivity_plot` and `sensitivity_contours` (Python), as DoubleML's `sensitivity_plot`: contours
  of the effect's bound (or confidence bound) over `cf_d` and `cf_y` with the scenario and
  benchmarks marked (matplotlib), and the grid as data; the grid comes from the new native
  `DmlResult.sensitivity_grid` and matches DoubleML's to `1e-9`.

### Fixed
- Sensitivity analysis: when `ν²` is not positive in some repetition of an estimate, every
  repetition now falls back to the non-orthogonal `mean(α²)`, as DoubleML (previously only the
  affected repetition did).

### Changed
- Nuisance names in `ScoreEstimate::predictions` and `NuisanceTarget::name` are `Name`
  (`Cow<'static, str>`), so models can name learners at run time (e.g. `ml_m_z2`). The
  constructors accept string literals as before.
- `DmlData::z` returns the instrument only when exactly one is attached; `instruments` returns all.
  IIVM, LPQ and SSM reject several instruments with a clear error.

## [0.12.0] - 2026-10-25

### Added
- Sample selection model (`Ssm` / `SSM`), as DoubleML's `DoubleMLSSM`: the average treatment effect
  when the outcome is observed only for a selected subsample, with selection missing at random or
  nonignorable (with an instrument for selection; nested cross-fitting within each fold).
  `DmlData::with_selection_indicator`. Four parity fixtures; the Python tests match DoubleML to
  `1e-9` with scikit-learn learners.
- Logistic partially linear regression (`Lplr` / `LPLR`), as DoubleML's `DoubleMLLPLR`: the log
  odds ratio of a treatment on a binary outcome, with the nuisance-space and instrument scores,
  double cross-fitting (`n_folds_inner`; `fit_with_inner_folds` replays given inner splits) and
  Newton's method on the non-linear score. `ml_m` may be a regressor or, for a binary treatment, a
  classifier (`TreatmentLearner`). Four parity fixtures; the Python tests match DoubleML to `1e-9`.
- Partially linear panel regression (`Plpr` / `PLPR`), as DoubleML's `DoubleMLPLPR`: static
  panels (`StaticPanel`, long format with unit and period) transformed by first differences,
  within-group demeaning or correlated random effects (`PanelApproach`), then a partially linear
  regression with unit-clustered folds and standard errors. Five parity fixtures (including
  unbalanced panels); the Python tests match DoubleML to `1e-9`. Python accepts any sortable unit
  and period labels (e.g. strings and dates).
- Regression discontinuity with flexible covariate adjustment (`RdFlex` / `RDFlex`), as
  DoubleML's `RDFlex`: sharp and fuzzy designs, cutoff, first-stage specifications and kernels
  (`FirstStage`, `RdKernel`), iterative bandwidth updates. Its back end, the new `rd` module, ports
  the defaults of the `rdrobust` package (`rdbwselect` MSE-optimal bandwidths with the mass-point
  floor; `rdrobust` local linear estimates with bias correction, nearest-neighbour standard errors
  and conventional / bias-corrected / robust intervals) and matches it to `1e-9`. Seven parity
  fixtures; the Python tests match DoubleML to `1e-9`. `parity/requirements.txt` pins `rdrobust`.
- `Classifier::fit_weighted` (default: unsupported) and sample weights for `LogisticRegression`;
  Python classifiers receive `sample_weight`.

## [0.11.0] - 2026-10-25

### Added
- Quantile models, as DoubleML: potential quantiles (`Pq` / `PQ`), local potential quantiles of
  the compliers with a binary instrument (`Lpq` / `LPQ`), conditional value at risk (`Cvar` /
  `CVAR`), and quantile treatment effects at several quantiles with joint inference (`Qte` /
  `QTE`, scores PQ, LPQ and CVaR; `QTE.potential` gives the potential quantiles). Includes
  DoubleML's nested preliminary estimate, the start value carried across repetitions, and the
  kernel density standard errors. Default sample splits are stratified by treatment (and
  instrument). Seven parity fixtures; the Python tests match DoubleML to `1e-9` with scikit-learn
  learners.

### Changed
- IRM, APO and IIVM with `normalize_ipw=True` compute the normalising means with NumPy's pairwise
  summation, so they agree with DoubleML to the last bit (previously within about `1e-15`).

## [0.10.0] - 2026-10-25

### Added
- CATE / GATE for PLR, as DoubleML's `DoubleMLPLR.cate` / `gate` (both scores: the outcome
  residual on the basis scaled by each repetition's treatment residual), and conditional average
  potential outcomes for APO (`capo` / `gapo`). Models declare their signal through the new
  `CateSignal` (`ScoreEstimate::with_cate_signal`); `Blp::fit_bases` takes one basis per
  repetition.
- `weights` for IRM and APO, as DoubleML: weighted average effects (e.g. the effect in a subgroup),
  0/1 subgroup weights with the ATTE score, and separate per-repetition `weights_bar`. Python:
  `IRM(weights=...)` / `APO(weights=...)` with an array or DoubleML's dict form.
- Policy trees, as DoubleML's `DoubleMLPolicyTree`: `DmlFit::policy_tree` / `IRM.policy_tree` fit a
  shallow decision tree to the sign of the doubly robust effect signal, weighted by its size, with
  scikit-learn's growth rules and minimal cost-complexity pruning (same pruning order), and give
  treatment recommendations and the rules as text. Python: `dml_rs.PolicyTree`.
- `ScoreInput::rep`, the index of the repetition being estimated.
- Multi-valued treatments, as DoubleML's `DoubleMLAPO` and `DoubleMLAPOS`: `Apo` estimates the
  average potential outcome `E[Y(d)]` of one treatment level with the doubly robust score;
  `Apos` fits several levels on shared sample splits, and `AposFit::causal_contrast` gives the
  contrasts between levels with joint inference and DoubleML's combined sensitivity elements.
  Default splits are stratified by the treatment. Python: `dml_rs.APO`, `dml_rs.APOS` and
  `APOS.causal_contrast`, which returns a `CausalContrast` result (summary, confidence intervals,
  bootstrap, sensitivity analysis).
- `DmlFit::concat` stacks fits of the same data and sample splits for joint inference (DoubleML's
  `concat`).
- DoubleML parity fixtures for APO (repeated splits, normalised IPW, sensitivity benchmark) and
  APOS (levels, contrasts against one and two reference levels, sensitivity), and Monte-Carlo tests
  of contrast coverage, including clustered data.
- Documentation site at https://asadgudzevic94.github.io/dml-rs/: getting started, a user guide
  for models, learners and inference, the Python API reference, validation details (including the
  known, deliberate differences from DoubleML and scikit-learn), and five worked notebooks
  (partially linear model, binary treatments and heterogeneity, difference-in-differences,
  sensitivity analysis, clustered data and learner diagnostics), each checked against the true
  effects or DoubleML.
- Python type information: stubs for the native extension (`_dml_rs.pyi`), annotations on the
  estimator classes and a `py.typed` marker, so editors autocomplete `dml_rs` and type checkers
  (mypy, pyright) check calls, e.g. invalid `score` or `scoring` values. CI checks the stubs
  against the compiled extension with `mypy.stubtest` and type-checks the package.

### Changed
- Linear combinations of clustered fits (APOS contrasts) keep the cluster structure and use the
  cluster-robust variance of the combined score; DoubleML's framework arithmetic ignores the
  correlation within clusters there.
- `LinearScoreModel` has an optional `strata` method: models can stratify their default sample
  splits (used by `Apo`).

## [0.9.0] - 2026-10-24

### Added
- `LassoCV` and `LogisticRegressionCV`, the cross-validated learners most DoubleML analyses use
  for their nuisances, with scikit-learn's grids, folds and selection rules: `LassoCV` fits a
  warm-started path of 100 penalties from `α_max` down to `1e-3 · α_max` on each of 5 contiguous
  folds and keeps the one with the lowest mean test MSE (sample weights supported);
  `LogisticRegressionCV` tries 10 values of `C` from `1e-4` to `1e4` on scikit-learn's stratified
  folds. Both refit on all the data. `CvScoring::LogLoss` (default; scikit-learn's default from
  version 1.11) or `CvScoring::Accuracy`. With scikit-learn's default settings and `n > p`,
  `LassoCV` reproduces scikit-learn's chosen penalty and coefficients to ~1e-15 and is 2–40×
  faster; PLR
  with `LassoCV` nuisances is ~10× faster than DoubleML. Python: `dml_rs.LassoCV` and
  `dml_rs.LogisticRegressionCV` with scikit-learn's parameter and attribute names (`alpha_`,
  `alphas_`, `mse_path_`, `C_`, `Cs_`, `scores_`, `coef_`, `intercept_`).
- DoubleML parity fixtures with cross-validated nuisance learners (`plr_lasso_cv`,
  `irm_ate_lasso_cv_logit_cv`) and `parity/bench_cv.py`.

### Changed
- `Lasso` now follows scikit-learn's coordinate descent exactly: the stopping rule is
  scikit-learn's (after a sweep in which no coefficient moves by more than `tol · max|w|`, the
  duality gap must be at most `tol · ‖y − ȳ‖² / n`), the default `tol` is scikit-learn's `1e-4`
  (was a step-size rule with `1e-12`), and gap safe screening skips coefficients that are provably
  zero. With `n > p` the solver works on the Gram matrix. Results at default settings match
  scikit-learn's `Lasso()` to ~1e-15 when `n > p`; with `p > n`, where the loose default `tol`
  can stop at slightly different points depending on floating-point rounding, both agree to
  within the tolerance. Pass a smaller `tol` for a more precise optimum.

## [0.8.0] - 2026-10-24

### Added
- Cluster-robust DML for one-way and two-way clustered data (Chiang, Kato, Ma & Sasaki, 2022), as
  DoubleML's `cluster_vars`: `DmlData::with_clusters` / `with_two_way_clusters` make every model
  assign whole clusters to folds (two-way: the product of the per-variable folds), solve the
  fold-weighted score and report cluster-robust standard errors; sensitivity bounds use the
  clustered variance. `fit_with_cluster_folds` replays given cluster splits, `DmlFit::n_clusters`
  reports the number of clusters. The multiplier bootstrap returns an error for clustered fits,
  as in DoubleML.
- Learner diagnostics: `DmlFit::nuisance_loss` / `nuisance_losses` give the out-of-sample RMSE
  (regressions) or log loss (classifications) of every nuisance learner on its own target, as
  DoubleML's `nuisance_loss`; `DmlFit::nuisance_targets` returns the targets (`NaN` where a
  learner does not apply) for custom metrics. Shown in the fit summary.
- `LossKind` and `NuisanceTarget`, with which custom `LinearScoreModel`s declare their nuisance
  targets (`ScoreEstimate::with_targets`).
- Python: `cluster=` (one or two columns of any label type) in the `fit` of PLR, IRM, PLIV,
  IIVM, DID and DIDCS, `n_clusters_`, `nuisance_loss_` (DoubleML's layout) and
  `evaluate_learners(learners=None, metric=None)`.
- DoubleML parity fixtures for one-way (PLR with repeated splits, IRM) and two-way (PLR, PLIV)
  clustering, nuisance losses in every fixture, and a Monte-Carlo test in which iid standard
  errors under-cover with clustered errors while cluster-robust ones do not.

### Changed
- `ScoreEstimate` has a new `targets` field (empty by default, so existing custom models keep
  working; without targets no losses are reported for them).

## [0.7.0] - 2026-10-24

### Added
- Sensitivity analysis for omitted confounders (Chernozhukov et al., 2022), as DoubleML's
  `sensitivity_analysis` / `sensitivity_benchmark`, for PLR (both scores), IRM (ATE / ATTE) and
  DiD for panels and repeated cross-sections: `DmlFit::sensitivity_analysis` gives bounds on θ,
  one-sided confidence bounds and robustness values (RV, RVa); `DmlFit::sensitivity_benchmark`
  and the models' `sensitivity_benchmark` calibrate `cf_y`, `cf_d` and `rho` against observed
  covariates; `DmlFit::sensitivity_elements` exposes σ² and ν². Robustness values are solved
  exactly (DoubleML uses a bounded numerical minimiser).
- `DmlData::without_covariates`.
- Python: `sensitivity_analysis`, `sensitivity_summary`, `sensitivity_params_` and
  `sensitivity_benchmark` on PLR, IRM, DID and DIDCS.
- DoubleML parity checks for sensitivity analysis in all PLR, IRM and DiD fixtures.

### Changed
- `ScoreEstimate` is now `#[non_exhaustive]` with an optional `sensitivity` field; custom
  `LinearScoreModel` implementations build it with `ScoreEstimate::new(score, predictions)`.

## [0.6.0] - 2026-10-24

### Added
- `GradientBoostingRegressor` (squared error) and `GradientBoostingClassifier` (log loss with
  Newton leaf updates): shallow CART trees on the negative gradient with learning rate, depth and
  leaf-size limits, stochastic row subsampling, random feature subsets, sample weights and
  feature importances. Predictions reproduce scikit-learn's `GradientBoostingRegressor` /
  `GradientBoostingClassifier` to ~1e-14 (the classifier can differ only where two splits have
  exactly equal gain, which scikit-learn itself resolves differently per `random_state`).
  About 9× faster than DoubleML with scikit-learn boosting for PLR. Python:
  `dml_rs.GradientBoostingRegressor` / `dml_rs.GradientBoostingClassifier`.
- `CONTRIBUTING.md` and `CITATION.cff` (GitHub's "Cite this repository").

### Changed
- README rewritten: model overview, reproducible quick start, benchmarks, validation and
  citation; matching crates.io and PyPI pages.

## [0.5.0] - 2026-10-24

### Added
- Staggered DID for **repeated cross-sections**: `RepeatedCrossSections` data, accepted by
  `StaggeredDid::fit` alongside `PanelData` (both convert into the new `StaggeredInput`), as
  DoubleML's `DoubleMLDIDMulti(panel=False)`; folds are stratified by group × period.
- Python: `StaggeredDID(..., panel=False)`.
- DoubleML parity fixtures for staggered DID with repeated cross-sections.

### Changed
- Staggered aggregation weights use each cohort's share of *observations* (DoubleML's
  convention). Identical to unit shares for balanced panels; for unbalanced panels results now
  match DoubleML.
- `StaggeredDid::fit`, `fit_with_folds` and `group_times` take `impl Into<StaggeredInput>`;
  existing calls with `&PanelData` keep working.

## [0.4.0] - 2026-10-23

### Added
- `StaggeredDid`: difference-in-differences with staggered adoption (Callaway & Sant'Anna, 2021;
  DoubleML's `DoubleMLDIDMulti` for panels). Estimates every group-time effect ATT(g, t) with the
  doubly robust panel DID score; never-treated or not-yet-treated controls, anticipation periods,
  standard / all / universal base periods. Results are a `DmlFit`, so joint bootstrap bands and
  Romano–Wolf adjustment work across all ATT(g, t).
- `StaggeredDidFit::aggregate` by cohort, calendar time or event time, plus the overall effect,
  with influence-function based standard errors.
- `PanelData` (wide, with `NaN` for missing periods, or `from_long`), unbalanced panels supported.
- `KFold::split_stratified`.
- Python: `dml_rs.StaggeredDID` (`fit` on long arrays or `fit_frame` on a DataFrame, `aggregate`).
- DoubleML parity fixtures for three staggered settings, including all aggregations.

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

[Unreleased]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.20.0...HEAD
[0.20.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.19.0...v0.20.0
[0.19.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.18.0...v0.19.0
[0.18.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AsadGudzevic94/dml-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AsadGudzevic94/dml-rs/releases/tag/v0.1.0

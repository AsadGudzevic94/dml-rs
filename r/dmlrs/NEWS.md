# dmlrs (development version)

* Heterogeneous effects:
  * `cate()` and `gate()`: best linear predictor of the CATE and group average
    effects of `dml_plr()` and `dml_irm()` fits, with `confint()` (also
    simultaneous) and `predict()`.
  * `cate_learner()`: S-, T-, X-, R- and DR-learners.
  * `causal_forest()` and `instrumental_forest()`: ports of the grf forests on
    cross-fitted nuisances, with `predict()`, `average_treatment_effect()`,
    `test_calibration()`, `cate()` and `gate()`.
  * `conformal_ite()`: conformal prediction intervals for individual effects,
    with residual or quantile scores (`lrn_quantile_forest()`).
* `dml_dose_response()`: dose-response curves of a continuous treatment, with
  `dose_contrast()` and `plot()`.
* All of these give the same estimates as the Python package (new parity tests).
* Reference site and an article on heterogeneous effects:
  <https://asadgudzevic94.github.io/dml-rs/r/>.
* Built on dml-core 1.0.
* The learners' `random_state` argument is now `seed`, as in the model
  functions; `random_state` still works with a deprecation warning.
* `dml_contdid()`: `trimming_threshold` is now `clipping_threshold`, as in the
  other models (`dml_mediation()` keeps `trimming_threshold`, the threshold of
  its trimming rule); the old name still works with a deprecation warning.

# dmlrs 0.2.1

* Prepared for CRAN: the package tarball bundles the Rust crates and builds
  offline, `configure` checks for and reports the Rust toolchain, and the
  Rust library is built with link-time optimisation (smaller).
* The number of threads can be set with `options(dmlrs.threads = n)` or
  `RAYON_NUM_THREADS` before the package is loaded; under `R CMD check` with
  `_R_CHECK_LIMIT_CORES_` set the package uses two.
* `inst/AUTHORS` lists the authors and licences of the bundled Rust crates.

# dmlrs 0.2.0

* New: `dml_apos()`, `linear_combination()`, `dml_mediation()`,
  `dml_contdid()`, `synth_did()`, `synth_did_panel()`, `synth_se()`,
  `policy_tree()`, `policy_value()`, `double_robust_scores()` and
  `lee_bounds()`.

# dmlrs 0.1.0

* First release: `dml_plr()`, `dml_irm()`, `dml_pliv()`, `dml_iivm()`,
  `dml_did()`, the built-in learners, multiplier bootstrap and sensitivity
  analysis.

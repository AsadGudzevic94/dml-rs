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

#' dmlrs: fast double / debiased machine learning
#'
#' Double / debiased machine learning (Chernozhukov et al., 2018) with the
#' estimation, including the nuisance learners, implemented in Rust. The
#' models follow the DoubleML package: [dml_plr()], [dml_irm()], [dml_pliv()],
#' [dml_iivm()], [dml_did()] and [dml_apos()]. Nuisance learners are created
#' with the [learners] constructors such as [lrn_lasso_cv()] or
#' [lrn_random_forest()].
#'
#' Beyond DoubleML: causal mediation ([dml_mediation()]), difference in
#' differences with a continuous treatment ([dml_contdid()]), synthetic
#' difference in differences ([synth_did()], a port of `synthdid`), optimal
#' policy trees ([policy_tree()], a port of `policytree`) with
#' [policy_value()], and Lee bounds under sample selection ([lee_bounds()]).
#'
#' The same Rust library powers the Python package dml-rs, whose results are
#' validated against DoubleML; see <https://asadgudzevic94.github.io/dml-rs/>.
#'
#' @keywords internal
#' @importFrom stats coef confint nobs predict
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  rs_init()
}

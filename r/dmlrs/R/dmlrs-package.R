#' dmlrs: fast double / debiased machine learning
#'
#' Double / debiased machine learning (Chernozhukov et al., 2018) with the
#' estimation, including the nuisance learners, implemented in Rust. The
#' models follow the DoubleML package: [dml_plr()], [dml_irm()], [dml_pliv()],
#' [dml_iivm()] and [dml_did()]. Nuisance learners are created with the
#' [learners] constructors such as [lrn_lasso_cv()] or [lrn_random_forest()].
#'
#' The same Rust library powers the Python package dml-rs, whose results are
#' validated against DoubleML; see <https://asadgudzevic94.github.io/dml-rs/>.
#'
#' @keywords internal
#' @importFrom stats confint nobs
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  rs_init()
}

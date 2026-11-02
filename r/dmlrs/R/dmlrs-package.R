#' dmlrs: fast double / debiased machine learning
#'
#' Double / debiased machine learning (Chernozhukov et al., 2018) with the
#' estimation, including the nuisance learners, implemented in Rust. The
#' models follow the DoubleML package: [dml_plr()], [dml_irm()], [dml_pliv()],
#' [dml_iivm()], [dml_did()] and [dml_apos()]. Nuisance learners are created
#' with the [learners] constructors such as [lrn_lasso_cv()] or
#' [lrn_random_forest()].
#'
#' Heterogeneous effects: the best linear predictor of the CATE and group
#' effects of a fit ([cate()], [gate()]), CATE meta-learners
#' ([cate_learner()]), causal forests ([causal_forest()], a port of `grf`) and
#' conformal intervals for individual effects ([conformal_ite()]).
#'
#' Beyond DoubleML: dose-response curves ([dml_dose_response()]), causal
#' mediation ([dml_mediation()]), difference in
#' differences with a continuous treatment ([dml_contdid()]), synthetic
#' difference in differences ([synth_did()], a port of `synthdid`), optimal
#' policy trees ([policy_tree()], a port of `policytree`) with
#' [policy_value()], and Lee bounds under sample selection ([lee_bounds()]).
#'
#' The same Rust library powers the Python package dml-rs, whose results are
#' validated against DoubleML; see <https://asadgudzevic94.github.io/dml-rs/>.
#' The R reference and articles are at <https://asadgudzevic94.github.io/dml-rs/r/>.
#'
#' @section Threads:
#' Cross-fitting and the forest and boosting learners run in parallel on all
#' cores. To use fewer, set `options(dmlrs.threads = n)` or the environment
#' variable `RAYON_NUM_THREADS` before the package is loaded. Under
#' `R CMD check` with `_R_CHECK_LIMIT_CORES_` set (as on CRAN), the package
#' uses two threads.
#'
#' @keywords internal
#' @importFrom stats coef confint nobs predict
"_PACKAGE"

.onLoad <- function(libname, pkgname) {
  rs_init(default_threads())
}

# The size of the thread pool, or NULL for rayon's default (all cores, or
# RAYON_NUM_THREADS).
default_threads <- function() {
  threads <- getOption("dmlrs.threads")
  if (!is.null(threads)) return(as.integer(threads))
  if (nzchar(Sys.getenv("RAYON_NUM_THREADS"))) return(NULL)
  limit <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_"))
  if (nzchar(limit) && limit != "false") return(2L)
  NULL
}

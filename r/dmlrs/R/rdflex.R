# Regression discontinuity with flexible covariate adjustment.

rdflex_options <- c("p", "q", "kernel", "bwselect", "vce", "nnmatch", "level", "masspoints", "bwcheck",
                    "bwrestrict", "stdvars", "scaleregul", "sharpbw", "rho", "scalepar", "covs_drop")

#' Regression discontinuity with flexible covariate adjustment
#'
#' Sharp and fuzzy regression discontinuity designs with covariates adjusted by
#' cross-fitted machine learning (Noack, Olma and Rothe, 2024), as DoubleML's
#' `RDFlex`. The outcome (and in a fuzzy design the treatment) is adjusted by a
#' prediction from `x` at the cutoff, learned with kernel weights around it, and
#' a local polynomial RD estimator runs on the adjusted outcome. That stage is a
#' port of `rdrobust` with its bias correction and robust inference. The
#' first-stage bandwidth is updated from the RD bandwidth over `n_iterations`
#' rounds.
#'
#' @param y Outcome.
#' @param d Treatment: the indicator of `score >= cutoff` in a sharp design, or
#'   the treatment received in a fuzzy one.
#' @param x Covariates: a numeric matrix or data frame.
#' @param score Running variable.
#' @param ml_g Learner for the outcome; needs case weights (`lrn_ols()`,
#'   `lrn_lasso()`, the forests and boosting learners).
#' @param ml_m Classifier for the treatment, used in a fuzzy design.
#' @param fuzzy Fuzzy design (the treatment may differ from
#'   `score >= cutoff`).
#' @param cutoff The cutoff in the score.
#' @param h_fs Initial first-stage bandwidth (default: `rdbwselect` without
#'   covariates).
#' @param fs_specification `"cutoff"`, `"cutoff and score"` or
#'   `"interacted cutoff and score"`.
#' @param fs_kernel Kernel of the first stage: `"triangular"`, `"uniform"` or
#'   `"epanechnikov"`.
#' @param n_iterations Rounds of first-stage fitting and bandwidth updates.
#' @param binary_outcome Learn a binary `y` with a classifier: `ml_g` then
#'   needs to be a classifier too.
#' @param n_folds,n_rep Cross-fitting folds and repetitions of the sample split.
#' @param seed Seed of the sample splits.
#' @param ... Options of the RD estimator, as in `rdrobust::rdrobust()`: `p`,
#'   `q`, `kernel`, `bwselect`, `vce`, `nnmatch`, `level` (in percent),
#'   `masspoints`, `bwcheck`, `bwrestrict`, `stdvars`, `scaleregul`, `sharpbw`,
#'   `rho`, `scalepar`, `covs_drop`, `covs` (covariates of the RD stage) and
#'   `cluster` (labels for cluster-robust standard errors). The bandwidths `h`
#'   and `b` are selected by the method.
#' @return A `dmlrs_rdflex` object: `coefficients`, `se`, `t` and `p` with three
#'   entries (conventional, bias-corrected and robust), `all_coef` and `all_se`
#'   per repetition, the bandwidths `h` and `b` per repetition and the first-stage
#'   bandwidth `h_fs`. The robust row is the one to report: conventional intervals
#'   ignore the bias of the local polynomial.
#' @references Noack, C., Olma, T. and Rothe, C. (2024). Flexible covariate
#'   adjustments in regression discontinuity designs. arXiv:2107.07942.
#' @examples
#' set.seed(2)
#' n <- 1000
#' score <- runif(n, -1, 1)
#' x <- matrix(rnorm(n * 3), n)
#' d <- as.numeric(score >= 0)
#' y <- 0.5 * d + score + x[, 1] + rnorm(n)
#' fit <- dml_rdflex(y, d, x, score, ml_g = lrn_ols())
#' fit
#' confint(fit)
#' @export
dml_rdflex <- function(y, d, x, score, ml_g = lrn_lasso_cv(), ml_m = NULL, fuzzy = FALSE, cutoff = 0, ...,
                       h_fs = NULL, fs_specification = c("cutoff", "cutoff and score",
                                                         "interacted cutoff and score"),
                       fs_kernel = c("triangular", "uniform", "epanechnikov"), n_iterations = 2,
                       binary_outcome = FALSE, n_folds = 5, n_rep = 1, seed = 0) {
  fs_specification <- match.arg(fs_specification)
  fs_kernel <- match.arg(fs_kernel)
  rd <- list(...)
  if (any(c("h", "b") %in% names(rd))) {
    stop("dml_rdflex() selects the bandwidths h and b itself (use rho for b = h / rho)", call. = FALSE)
  }
  unknown <- setdiff(names(rd), c(rdflex_options, "covs", "cluster"))
  if (length(unknown) > 0 || (length(rd) > 0 && is.null(names(rd)))) {
    stop(sprintf("unknown rdrobust options: %s", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  n <- length(data$y)
  score <- as_numeric_vector(score, "score", n)
  if (isTRUE(fuzzy) && is.null(ml_m)) {
    stop("a fuzzy design needs a classifier `ml_m` for the treatment", call. = FALSE)
  }
  covs <- if (!is.null(rd$covs)) as_numeric_matrix(rd$covs, "covs", n)
  cluster <- rd$cluster
  if (!is.null(cluster)) {
    if (length(cluster) != n) stop("`cluster` must have one value per observation", call. = FALSE)
    cluster <- match(cluster, sort(unique(cluster)))
  }
  options <- rd[setdiff(names(rd), c("covs", "cluster"))]
  ptr <- rs_fit_rdflex(data$y, data$d, data$x, score, data$names,
                       check_learner(ml_g, if (isTRUE(binary_outcome)) "classifier" else "regressor", "ml_g"),
                       if (isTRUE(fuzzy)) check_learner(ml_m, "classifier", "ml_m"),
                       isTRUE(binary_outcome), as.numeric(cutoff),
                       if (!is.null(h_fs)) as.numeric(h_fs), fs_specification, fs_kernel,
                       check_count(n_iterations, "n_iterations"), check_count(n_folds, "n_folds", 2),
                       check_count(n_rep, "n_rep"), seed, as.list(options), covs, cluster)
  s <- rs_rdflex_summary(ptr)
  rows <- c("Conventional", "Bias-Corrected", "Robust")
  structure(list(ptr = ptr, call = match.call(), coefficients = stats::setNames(s$coef, rows),
                 se = stats::setNames(s$se, rows), t = stats::setNames(s$t, rows),
                 p = stats::setNames(s$p, rows), all_coef = s$all_coef, all_se = s$all_se, h = s$h,
                 b = s$b, h_fs = s$h_fs, fuzzy = isTRUE(fuzzy), n_obs = n, n_folds = as.integer(n_folds),
                 n_rep = as.integer(n_rep)),
            class = "dmlrs_rdflex")
}

#' @export
print.dmlrs_rdflex <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("Flexible regression discontinuity: %s design\n", if (x$fuzzy) "fuzzy" else "sharp"))
  cat(sprintf("%d observations, %d folds, %d repetition%s, median bandwidth %.4g\n\n", x$n_obs,
              x$n_folds, x$n_rep, if (x$n_rep == 1) "" else "s", stats::median(x$h)))
  stats::printCoefmat(coef_table(x), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  invisible(x)
}

#' @export
summary.dmlrs_rdflex <- function(object, ...) {
  print(object$call)
  cat("\n")
  print(object)
  invisible(object)
}

#' @export
nobs.dmlrs_rdflex <- function(object, ...) object$n_obs

#' @export
confint.dmlrs_rdflex <- function(object, parm, level = 0.95, ...) {
  if (!rs_is_valid_kind(object$ptr, "rdflex")) {
    stop("this fit was restored from a saved session and has no Rust state; refit the model",
         call. = FALSE)
  }
  ci <- rs_rdflex_confint(object$ptr, level)
  pct <- paste(format(100 * c((1 - level) / 2, (1 + level) / 2), trim = TRUE, digits = 3), "%")
  dimnames(ci) <- list(names(object$coefficients), pct)
  if (missing(parm)) ci else ci[parm, , drop = FALSE]
}

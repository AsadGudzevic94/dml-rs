# Models beyond the DoubleML core: discrete treatments, mediation, continuous DiD.

#' Average potential outcomes of a discrete treatment
#'
#' The average potential outcome \eqn{E[Y(a)]} of each treatment level
#' \eqn{a}, as `DoubleMLAPOS`. Contrasts between levels come from
#' [linear_combination()]; [double_robust_scores()] gives the scores for
#' [policy_tree()] and [policy_value()].
#'
#' @inheritParams dml_irm
#' @param d Discrete treatment (numeric levels).
#' @param ml_g Learner for \eqn{E[Y \mid D, X]}.
#' @param ml_m Classifier for \eqn{P(D = a \mid X)} (fitted one level against the rest).
#' @param treatment_levels Levels to estimate (default: all levels of `d`).
#' @inherit dml_plr return
#' @examples
#' set.seed(5)
#' n <- 1000
#' x <- matrix(rnorm(n * 3), n)
#' d <- sample(0:2, n, replace = TRUE)
#' y <- d * x[, 1] + x[, 2] + rnorm(n)
#' fit <- dml_apos(y, d, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
#' fit
#' # Each level against level 0:
#' linear_combination(fit, rbind("1 vs 0" = c(-1, 1, 0), "2 vs 0" = c(-1, 0, 1)))
#' @export
dml_apos <- function(y, d, x, ml_g = lrn_lasso_cv(), ml_m = lrn_logistic_cv(),
                     treatment_levels = sort(unique(d)), clipping_threshold = 0.01,
                     normalize_ipw = FALSE, n_folds = 5, n_rep = 1, seed = 0) {
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  ptr <- rs_fit_apos(data$y, data$d, data$x, data$names, check_learner(ml_g, "regressor", "ml_g"),
                     check_learner(ml_m, "classifier", "ml_m"), as.numeric(treatment_levels),
                     clipping_threshold, isTRUE(normalize_ipw), check_count(n_folds, "n_folds", 2),
                     check_count(n_rep, "n_rep"), seed)
  new_fit(ptr, match.call())
}

#' Causal mediation analysis
#'
#' Natural direct and indirect effects of a binary treatment `d` that acts
#' partly through one or several mediators `m`, with the double machine
#' learning estimator of Farbmacher, Huber, Laffers, Langen and Spindler
#' (2022), the estimator of `causalweight::medDML()`. The effects are
#' differences of the potential-outcome means \eqn{E[Y(d, M(d'))]}:
#' `total`, `direct (treated)`, `direct (control)`, `indirect (treated)` and
#' `indirect (control)`.
#'
#' The effects assume that `x` contains all confounders of the treatment,
#' the mediator and the outcome, and that none of them is affected by the
#' treatment.
#'
#' @inheritParams dml_irm
#' @param m Mediators: a numeric vector, or a matrix with one column per
#'   mediator (binary, categorical or continuous).
#' @param ml_g Learner for the outcome means \eqn{E[Y \mid D, X]} and
#'   \eqn{E[Y \mid D, M, X]}.
#' @param ml_m Classifier for \eqn{P(D = 1 \mid X)} and \eqn{P(D = 1 \mid M, X)}.
#' @param ml_nested Learner for the nested means (default: `ml_g`).
#' @param trimming_rule `"truncate"` clips the propensity scores to
#'   \eqn{[t, 1 - t]}; `"drop"` (medDML's rule) drops observations with small
#'   propensity products. Dropping selects on \eqn{P(D = 1 \mid M, X)}, which
#'   depends on the post-treatment mediator, and can bias the effects; use it
#'   to reproduce medDML (with `trimming_threshold = 0.05`).
#' @param trimming_threshold The threshold \eqn{t}.
#' @param normalize_ipw Normalise each inverse probability weight to mean one.
#' @param nested_split Fit the outcome means and the nested means on disjoint
#'   parts of the training folds (needs `n_folds >= 3`).
#' @param debias_nested Debias the nested means (Qi and Zhang, 2026).
#' @return A `dmlrs_fit` with the five effects, plus `potential_outcomes`
#'   (a `dmlrs_fit` with the four means) and `n_trimmed` (trimmed
#'   observations per repetition).
#' @examples
#' set.seed(6)
#' n <- 1000
#' x <- matrix(rnorm(n * 3), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' m <- d + x[, 1] + rnorm(n)
#' y <- 0.5 * d + m + x[, 2] + rnorm(n)
#' fit <- dml_mediation(y, d, m, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
#' fit
#' fit$potential_outcomes
#' @export
dml_mediation <- function(y, d, m, x, ml_g = lrn_lasso_cv(), ml_m = lrn_logistic_cv(),
                          ml_nested = NULL, trimming_rule = c("truncate", "drop"),
                          trimming_threshold = 0.01, normalize_ipw = TRUE, nested_split = TRUE,
                          debias_nested = TRUE, n_folds = 5, n_rep = 1, seed = 0) {
  trimming_rule <- match.arg(trimming_rule)
  data <- prepare(y, d, x)
  check_binary(data$d, "d")
  m <- as_numeric_matrix(m, "m", length(data$y))
  ml_nested <- if (is.null(ml_nested)) NULL else check_learner(ml_nested, "regressor", "ml_nested")
  res <- rs_fit_mediation(data$y, data$d, m, data$x, data$names,
                          check_learner(ml_g, "regressor", "ml_g"),
                          check_learner(ml_m, "classifier", "ml_m"), ml_nested, trimming_rule,
                          trimming_threshold, isTRUE(normalize_ipw), isTRUE(nested_split),
                          isTRUE(debias_nested), check_count(n_folds, "n_folds", 2),
                          check_count(n_rep, "n_rep"), seed)
  call <- match.call()
  fit <- new_fit(res$effects, call)
  fit$potential_outcomes <- new_fit(res$potential_outcomes, call)
  fit$n_trimmed <- res$n_trimmed
  fit
}

#' Difference in differences with a continuous treatment
#'
#' Two-period difference in differences with a treatment dose (Callaway,
#' Goodman-Bacon and Sant'Anna, 2024): the average effect of dose \eqn{d} on
#' the units that received it, \eqn{ATT(d \mid d)}, as a B-spline series in
#' the dose, its derivative \eqn{ACRT(d \mid d)}, and their averages over the
#' treated units. Without covariates the estimates equal those of R's
#' `contdid` package; with covariates the comparison is doubly robust, with
#' cross-fitted nuisance functions.
#'
#' @inheritParams dml_plr
#' @param y Outcome change between the two periods.
#' @param d Dose received between the periods (0 = untreated comparison units).
#' @param x Pre-treatment covariates, or `NULL` for none.
#' @param ml_g Learner for the outcome change of untreated units given `x`
#'   (with covariates).
#' @param ml_m Classifier for the probability of being treated given `x`.
#' @param ml_nested Learner for the dose-specific nuisance regressions
#'   (default: `ml_g`).
#' @param degree,num_knots Degree and number of interior knots of the B-spline.
#' @param grid Doses at which to report the effects (default: 5% to 95%
#'   quantiles of the positive doses).
#' @param clipping_threshold Propensity scores are clipped to \eqn{[t, 1 - t]}.
#' @param trimming_threshold Deprecated name of `clipping_threshold`.
#' @return A `dmlrs_contdid` object: `att` and `acrt` (`dmlrs_fit` objects with
#'   one estimate per grid dose), `overall` (the overall ATT and ACRT) and
#'   `grid`. `plot()` draws a curve with pointwise intervals.
#' @examples
#' set.seed(7)
#' n <- 1000
#' dose <- ifelse(runif(n) < 0.3, 0, runif(n, 0.5, 2))
#' dy <- dose * (1 + 0.5 * dose) + rnorm(n)
#' fit <- dml_contdid(dy, dose)
#' fit
#' plot(fit)
#' @export
dml_contdid <- function(y, d, x = NULL, ml_g = NULL, ml_m = NULL, ml_nested = NULL, degree = 3,
                        num_knots = 0, grid = NULL, clipping_threshold = 0.01, n_folds = 5,
                        n_rep = 1, seed = 0, trimming_threshold = NULL) {
  clipping_threshold <- renamed_arg(clipping_threshold, trimming_threshold, "clipping_threshold",
                                    "trimming_threshold")
  y <- as_numeric_vector(y, "y")
  n <- length(y)
  d <- as_numeric_matrix(d, "d", n)
  if (ncol(d) != 1) stop("`d` must be a single dose", call. = FALSE)
  x <- if (is.null(x)) matrix(numeric(0), n, 0) else as_numeric_matrix(x, "x", n)
  if (ncol(x) > 0 && (is.null(ml_g) || is.null(ml_m))) {
    stop("with covariates, pass `ml_g` and `ml_m`", call. = FALSE)
  }
  if (ncol(x) == 0 && !(is.null(ml_g) && is.null(ml_m))) {
    stop("`ml_g` and `ml_m` need covariates `x`", call. = FALSE)
  }
  ml_g <- if (is.null(ml_g)) NULL else check_learner(ml_g, "regressor", "ml_g")
  ml_m <- if (is.null(ml_m)) NULL else check_learner(ml_m, "classifier", "ml_m")
  ml_nested <- if (is.null(ml_nested)) NULL else check_learner(ml_nested, "regressor", "ml_nested")
  res <- rs_fit_contdid(y, d, x, ml_g, ml_m, ml_nested, check_count(degree, "degree"),
                        check_count(num_knots, "num_knots", 0),
                        if (is.null(grid)) NULL else as.numeric(grid), clipping_threshold,
                        check_count(n_folds, "n_folds", 2), check_count(n_rep, "n_rep"), seed)
  call <- match.call()
  structure(list(att = new_fit(res$att, call), acrt = new_fit(res$acrt, call),
                 overall = new_fit(res$overall, call), grid = res$grid, call = call),
            class = "dmlrs_contdid")
}

#' @export
print.dmlrs_contdid <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Difference in differences with a continuous treatment\n\n")
  cat("Overall effects:\n")
  stats::printCoefmat(coef_table(x$overall), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  cat(sprintf("\nATT(d | d) and ACRT(d | d) at %d doses from %.3g to %.3g: see $att and $acrt\n",
              length(x$grid), min(x$grid), max(x$grid)))
  invisible(x)
}

#' Plot a dose-response curve
#'
#' @param x A `dmlrs_contdid` fit.
#' @param type `"att"` or `"acrt"`.
#' @param level Level of the pointwise confidence band.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.dmlrs_contdid <- function(x, type = c("att", "acrt"), level = 0.95, ...) {
  type <- match.arg(type)
  fit <- x[[type]]
  ci <- confint(fit, level = level)
  est <- unname(stats::coef(fit))
  graphics::plot(x$grid, est, type = "n", ylim = range(ci, 0), xlab = "dose",
                 ylab = if (type == "att") "ATT(d | d)" else "ACRT(d | d)", ...)
  graphics::polygon(c(x$grid, rev(x$grid)), c(ci[, 1], rev(ci[, 2])),
                    col = grDevices::adjustcolor("#0173B2", 0.25), border = NA)
  graphics::lines(x$grid, est, col = "#0173B2", lwd = 2)
  graphics::abline(h = 0, col = "grey50", lty = 2)
  invisible(x)
}

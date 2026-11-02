# Staggered difference in differences: group-time effects and their aggregations.

#' Difference in differences with staggered adoption
#'
#' Group-time average treatment effects \eqn{ATT(g, t)} of Callaway and
#' Sant'Anna (2021) with cross-fitted nuisance functions, as DoubleML's
#' `DoubleMLDIDMulti`. Each cohort \eqn{g} (units first treated in period
#' \eqn{g}) is compared with never-treated or not-yet-treated units between a
#' pre-treatment period and an evaluation period. Use
#' [aggregate()][aggregate.dmlrs_staggered] for effects by cohort, by calendar
#' time or by time since treatment.
#'
#' The result is also a `dmlrs_fit`, so [confint()][confint.dmlrs_fit] gives
#' simultaneous bands with `joint = TRUE` (unavailable with `cluster`) and
#' [sensitivity_analysis()] applies. Controls of pre-treatment periods follow
#' DoubleML 0.11.4 with `control_group = "not_yet_treated"`: cohorts treated
#' after the evaluation period, also those before \eqn{g}.
#'
#' @param y Outcome, one value per row of the long data.
#' @param x Covariates: a numeric matrix or data frame, one row per row of `y`.
#'   For panels they must not change over time within a unit.
#' @param id Unit of each row (any type; ignored with `panel = FALSE`).
#' @param t Period of each row (numeric or `Date`).
#' @param g First treated period of each row's unit; `NA` or `Inf` for
#'   never-treated units.
#' @param ml_g Learner for the outcome regression.
#' @param ml_m Classifier for the propensity score (`NULL` with
#'   `score = "experimental"`).
#' @param control_group `"never_treated"` or `"not_yet_treated"`.
#' @param anticipation_periods Number of periods before \eqn{g} in which units
#'   may already react.
#' @param gt_combinations `"standard"` (post-treatment periods against the last
#'   pre-treatment period, pre-treatment periods against the period before),
#'   `"universal"` (every period against the last pre-treatment period) or
#'   `"all"` (every valid base period).
#' @param score `"observational"` (doubly robust) or `"experimental"` (treatment
#'   randomised).
#' @param in_sample_normalization Normalise the weights in the sample.
#' @param clipping_threshold Propensity scores are clipped to
#'   \eqn{[t, 1 - t]}.
#' @param panel `TRUE` for panel data (rows linked by `id`), `FALSE` for
#'   repeated cross-sections.
#' @param cluster Optional cluster of each row (any type; constant within a
#'   panel unit). Standard errors, aggregations and sensitivity bounds are then
#'   cluster-robust and whole clusters go to folds. DoubleML has no clustered
#'   version of this model.
#' @param n_folds,n_rep Cross-fitting folds and repetitions of the sample split.
#' @param seed Seed of the sample splits.
#' @return A `dmlrs_staggered` object, which is a `dmlrs_fit` of the
#'   \eqn{ATT(g, t)} plus `group_times`, a data frame with one row per estimate
#'   (`group`, `t_pre`, `t_eval`).
#' @references Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
#'   with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.
#' @examples
#' set.seed(1)
#' n <- 200
#' cohort <- sample(c(2, 3, NA), n, replace = TRUE)
#' covariate <- rnorm(n)
#' panel <- expand.grid(id = seq_len(n), t = 1:4)
#' panel$g <- cohort[panel$id]
#' panel$x <- covariate[panel$id]
#' treated <- !is.na(panel$g) & panel$t >= panel$g
#' panel$y <- panel$x + 0.5 * panel$t + ifelse(treated, 1, 0) + rnorm(nrow(panel))
#' fit <- dml_staggered_did(panel$y, panel["x"], panel$id, panel$t, panel$g,
#'                          ml_g = lrn_ols(), ml_m = lrn_logistic())
#' fit
#' aggregate(fit, "eventstudy")
#' @export
dml_staggered_did <- function(y, x, id, t, g, ml_g = lrn_lasso_cv(), ml_m = lrn_logistic_cv(),
                              control_group = c("never_treated", "not_yet_treated"),
                              anticipation_periods = 0,
                              gt_combinations = c("standard", "universal", "all"),
                              score = c("observational", "experimental"),
                              in_sample_normalization = TRUE, clipping_threshold = 0.01,
                              panel = TRUE, cluster = NULL, n_folds = 5, n_rep = 1, seed = 0) {
  control_group <- match.arg(control_group)
  gt_combinations <- match.arg(gt_combinations)
  score <- match.arg(score)
  y <- as_numeric_vector(y, "y")
  n <- length(y)
  x <- as_numeric_matrix(x, "x", n)
  if (length(t) != n) stop("`t` must have one value per observation", call. = FALSE)
  if (length(g) != n) stop("`g` must have one value per observation", call. = FALSE)
  if (anyNA(t)) stop("`t` contains missing values", call. = FALSE)
  t <- as.numeric(t)
  g <- as.numeric(g)
  g[is.na(g)] <- Inf
  if (isTRUE(panel)) {
    if (length(id) != n) stop("`id` must have one value per observation", call. = FALSE)
    if (anyNA(id)) stop("`id` contains missing values", call. = FALSE)
    id <- match(id, sort(unique(id)))
  } else {
    id <- rep(0L, n)
  }
  if (!is.null(cluster)) {
    if (length(cluster) != n) stop("`cluster` must have one value per observation", call. = FALSE)
    if (anyNA(cluster)) stop("`cluster` contains missing values", call. = FALSE)
    cluster <- match(cluster, sort(unique(cluster)))
  }
  if (score == "observational") {
    ml_m <- check_learner(ml_m, "classifier", "ml_m")
  } else {
    ml_m <- NULL
  }
  ptr <- rs_fit_staggered(y, as.integer(id), t, g, x, check_learner(ml_g, "regressor", "ml_g"),
                          ml_m, control_group, check_count(anticipation_periods, "anticipation_periods", 0),
                          gt_combinations, score, isTRUE(in_sample_normalization),
                          clipping_threshold, check_count(n_folds, "n_folds", 2),
                          check_count(n_rep, "n_rep"), seed, isTRUE(panel), cluster)
  out <- new_fit(rs_staggered_att(ptr), match.call())
  times <- rs_staggered_group_times(ptr)
  out$group_times <- data.frame(group = times[, 1], t_pre = times[, 2], t_eval = times[, 3])
  out$staggered <- ptr
  class(out) <- c("dmlrs_staggered", class(out))
  out
}

#' Aggregate group-time effects
#'
#' Average the \eqn{ATT(g, t)} of [dml_staggered_did()] over the post-treatment
#' periods by cohort, by calendar time or by time since treatment, as DoubleML's
#' `aggregate()`. Weights are the cohort shares, and the standard errors take the
#' correlation between the group-time effects into account.
#'
#' @param x A fit from [dml_staggered_did()].
#' @param by `"group"`, `"time"` or `"eventstudy"`.
#' @param ... Not used.
#' @return A `dmlrs_did_aggregation` object: `effects` and `overall` are
#'   `dmlrs_fit` objects (for [coef()], [confint()][confint.dmlrs_fit]), `weights`
#'   has one row per effect and one column per \eqn{ATT(g, t)}, and
#'   `overall_weights` weighs the effects into the overall one.
#' @examples
#' set.seed(1)
#' n <- 200
#' cohort <- sample(c(2, 3, NA), n, replace = TRUE)
#' panel <- expand.grid(id = seq_len(n), t = 1:4)
#' panel$g <- cohort[panel$id]
#' panel$x <- rnorm(n)[panel$id]
#' treated <- !is.na(panel$g) & panel$t >= panel$g
#' panel$y <- panel$x + 0.5 * panel$t + ifelse(treated, 1, 0) + rnorm(nrow(panel))
#' fit <- dml_staggered_did(panel$y, panel["x"], panel$id, panel$t, panel$g,
#'                          ml_g = lrn_ols(), ml_m = lrn_logistic())
#' effects <- aggregate(fit, "group")
#' effects
#' confint(effects$overall)
#' @importFrom stats aggregate
#' @export
aggregate.dmlrs_staggered <- function(x, by = c("group", "time", "eventstudy"), ...) {
  by <- match.arg(by)
  if (!rs_is_valid_kind(x$staggered, "staggered")) {
    stop("this fit was restored from a saved session and has no Rust state; refit the model",
         call. = FALSE)
  }
  res <- rs_staggered_aggregate(x$staggered, by)
  dimnames(res$weights) <- list(NULL, names(x$coefficients))
  structure(list(by = by, effects = new_fit(res$effects, x$call), overall = new_fit(res$overall, x$call),
                 weights = res$weights, overall_weights = res$overall_weights),
            class = "dmlrs_did_aggregation")
}

#' @export
print.dmlrs_did_aggregation <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  label <- c(group = "group", time = "calendar time", eventstudy = "time since treatment")
  cat(sprintf("Staggered DiD: effects by %s\n\n", label[[x$by]]))
  stats::printCoefmat(rbind(coef_table(x$effects), overall = coef_table(x$overall)), digits = digits,
                      has.Pvalue = TRUE, P.values = TRUE)
  invisible(x)
}

# One point range per effect: estimates, intervals and the colour of each.
effect_ranges <- function(at, est, ci, colours, ...) {
  graphics::abline(h = 0, col = "grey60", lty = 2)
  graphics::segments(at, ci[, 1], at, ci[, 2], col = colours, lwd = 2)
  graphics::points(at, est, col = colours, pch = 16)
}

#' Plot staggered DiD effects
#'
#' `plot()` of a fit shows the \eqn{ATT(g, t)} over evaluation periods, one
#' panel per cohort, in the style of DoubleML's `plot_effects()`; of an
#' aggregation, the aggregated effects with the overall effect as a solid line.
#' Pre-treatment estimates are orange and post-treatment ones blue. Set
#' `joint = TRUE` for simultaneous bands (not with clusters).
#'
#' @param x A fit from [dml_staggered_did()] or an aggregation from
#'   [aggregate()][aggregate.dmlrs_staggered].
#' @param level Confidence level.
#' @param joint Simultaneous bands from the multiplier bootstrap.
#' @param seed Seed of the bootstrap.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @examples
#' set.seed(1)
#' n <- 200
#' cohort <- sample(c(2, 3, NA), n, replace = TRUE)
#' panel <- expand.grid(id = seq_len(n), t = 1:4)
#' panel$g <- cohort[panel$id]
#' panel$x <- rnorm(n)[panel$id]
#' treated <- !is.na(panel$g) & panel$t >= panel$g
#' panel$y <- panel$x + 0.5 * panel$t + ifelse(treated, 1, 0) + rnorm(nrow(panel))
#' fit <- dml_staggered_did(panel$y, panel["x"], panel$id, panel$t, panel$g,
#'                          ml_g = lrn_ols(), ml_m = lrn_logistic())
#' plot(fit)
#' plot(aggregate(fit, "eventstudy"))
#' @export
plot.dmlrs_staggered <- function(x, level = 0.95, joint = FALSE, seed = 0, ...) {
  ci <- confint(x, level = level, joint = joint, seed = seed)
  est <- unname(x$coefficients)
  times <- x$group_times
  cohorts <- sort(unique(times$group))
  old <- graphics::par(mfrow = grDevices::n2mfrow(length(cohorts)), mar = c(4, 4, 2.5, 1))
  on.exit(graphics::par(old))
  for (cohort in cohorts) {
    rows <- which(times$group == cohort)
    pre <- times$t_eval[rows] < cohort
    graphics::plot(times$t_eval[rows], est[rows], type = "n", ylim = range(ci, 0), xaxt = "n",
                   xlab = "evaluation period", ylab = "ATT",
                   main = sprintf("cohort %g", cohort), ...)
    graphics::axis(1, at = sort(unique(times$t_eval)))
    effect_ranges(times$t_eval[rows], est[rows], ci[rows, , drop = FALSE],
                  ifelse(pre, "#DE8F05", "#0173B2"))
  }
  invisible(x)
}

#' @rdname plot.dmlrs_staggered
#' @export
plot.dmlrs_did_aggregation <- function(x, level = 0.95, joint = FALSE, seed = 0, ...) {
  ci <- confint(x$effects, level = level, joint = joint, seed = seed)
  est <- unname(stats::coef(x$effects))
  at <- suppressWarnings(as.numeric(names(stats::coef(x$effects))))
  if (anyNA(at)) at <- seq_along(est)
  pre <- x$by == "eventstudy" & at < 0
  label <- c(group = "group", time = "calendar time", eventstudy = "time since treatment")
  graphics::plot(at, est, type = "n", ylim = range(ci, 0), xaxt = "n", xlab = label[[x$by]],
                 ylab = "effect", ...)
  graphics::axis(1, at = at)
  graphics::abline(h = unname(stats::coef(x$overall)), col = "#0173B2")
  effect_ranges(at, est, ci, ifelse(pre, "#DE8F05", "#0173B2"))
  invisible(x)
}

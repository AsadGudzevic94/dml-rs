# Causal effects on a right-censored time to event.

#' Effects on a censored time to event
#'
#' Effect of a binary treatment on a right-censored time to event, as the
#' difference in survival probabilities or in restricted mean survival time at
#' chosen horizons. Treatment and censoring may depend on the covariates. Time
#' is split into slots and each unit becomes one row per slot at risk, so any
#' classifier can learn the hazards of the event and of censoring, separately
#' for treated and controls. The cross-fitted hazards and the propensity score
#' enter the efficient influence function of the survival curve, so the estimate
#' is doubly robust: it is consistent if the event hazards, or the censoring
#' hazards and the propensity score, are.
#'
#' Besides unconfoundedness, the event and censoring times must be independent
#' given the treatment and covariates. Choose horizons where units are still
#' observed in both arms: standard errors grow as few remain uncensored. Without
#' covariates and with empirical hazards, the estimator equals Kaplan-Meier.
#'
#' With competing events, code `event` as 1, 2, ... for the cause. Survival and
#' RMST are then freedom from every cause, and `estimand = "cumulative_incidence"`
#' estimates the cumulative incidence of `cause` (the Aalen-Johansen estimator
#' without covariates). With `cluster`, whole clusters go to the folds and the
#' standard errors are cluster-robust.
#'
#' @param time Observed time: of the event, or of censoring.
#' @param event 0 for censoring, 1 for an observed event, or 1, 2, ... for the
#'   cause of competing events.
#' @param d Binary treatment.
#' @param x Covariates: a numeric matrix or data frame.
#' @param ml_event Classifier for the hazard of the event in a slot.
#' @param ml_censor Classifier for the hazard of censoring (default: `ml_event`).
#' @param ml_m Classifier for the propensity score.
#' @param horizons Times of the effects (default: the quartiles of the event times).
#' @param estimand `"survival"` for \eqn{P(T > \tau)}, `"rmst"` for
#'   \eqn{E[\min(T, \tau)]} or `"cumulative_incidence"` for
#'   \eqn{P(T \le \tau, \textrm{cause} = c)} or `"rmtl"` for the restricted mean
#'   time lost to cause \eqn{c}, \eqn{\int_0^\tau F_c(t)\,dt}.
#' @param cause The cause \eqn{c} of the cumulative incidence (an event code).
#' @param cluster Cluster labels (one per observation) for cluster-robust
#'   standard errors, or `NULL`.
#' @param grid The slots: a whole number for that many quantiles of the event
#'   times, `"event_times"` for every distinct event time (slow on large data),
#'   or a vector of at least two times. The horizons are always added. Use 20
#'   slots or more.
#' @param time_encoding How the slot enters the learners: `"numeric"` (its time,
#'   for trees and boosting) or `"dummies"` (one indicator per slot, which gives a
#'   linear learner a free baseline hazard).
#' @param clipping_threshold Propensity scores are clipped to \eqn{[t, 1 - t]}.
#' @param censoring_floor Probabilities of remaining uncensored are raised to this
#'   value.
#' @param n_folds,n_rep Cross-fitting folds and repetitions of the sample split.
#' @param seed Seed of the sample splits, stratified by treatment and event.
#' @return A `dmlrs_survival` object, which is a `dmlrs_fit` of treated minus
#'   control at each horizon, plus `treated` and `control` (`dmlrs_fit` objects of
#'   the two arms at the horizons), `curve_treated` and `curve_control` (the
#'   survival curves on `grid`), `horizons`, `grid` and `propensity` (one column
#'   per repetition). Cumulative-incidence fits also hold `incidence_treated` and
#'   `incidence_control`, the cumulative incidence curves of the cause.
#' @references Bai, X., Tsiatis, A. A. and O'Brien, S. M. (2013). Doubly-robust
#'   estimators of treatment-specific survival distributions in observational
#'   studies with stratified sampling. *Biometrics*, 69(4), 830-839.
#' @examples
#' set.seed(3)
#' n <- 800
#' x <- matrix(rnorm(n * 2), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' event_time <- rexp(n, exp(0.5 * x[, 1]) * ifelse(d == 1, 0.5, 1))
#' censoring <- rexp(n, 0.3)
#' fit <- dml_survival(pmin(event_time, censoring), as.numeric(event_time <= censoring), d, x,
#'                     ml_event = lrn_logistic(), ml_m = lrn_logistic(), horizons = c(0.5, 1),
#'                     grid = 12, time_encoding = "dummies")
#' fit
#' fit$control
#' plot(fit)
#' @export
dml_survival <- function(time, event, d, x, ml_event = lrn_random_forest(min_samples_leaf = 20),
                         ml_censor = ml_event, ml_m = lrn_logistic_cv(), horizons = NULL,
                         estimand = c("survival", "rmst", "cumulative_incidence", "rmtl"), grid = 30,
                         time_encoding = c("numeric", "dummies"), clipping_threshold = 0.01,
                         censoring_floor = 0.01, n_folds = 5, n_rep = 1, seed = 0, cause = 1,
                         cluster = NULL) {
  estimand <- match.arg(estimand)
  time_encoding <- match.arg(time_encoding)
  data <- prepare(time, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  check_binary(data$d, "d")
  event <- as_numeric_vector(event, "event", length(data$y))
  if (any(event < 0 | event != round(event))) {
    stop("`event` must be 0 for censoring and 1, 2, ... for the cause of an event", call. = FALSE)
  }
  if (!is.null(cluster)) {
    if (length(cluster) != length(data$y)) {
      stop("`cluster` must have one value per observation", call. = FALSE)
    }
    cluster <- match(cluster, sort(unique(cluster)))
  }
  if (!(is.numeric(grid) || identical(grid, "event_times"))) {
    stop("`grid` must be a number, \"event_times\" or a vector of times", call. = FALSE)
  }
  res <- rs_fit_survival(data$y, data$d, data$x, event, check_learner(ml_event, "classifier", "ml_event"),
                         check_learner(ml_censor, "classifier", "ml_censor"),
                         check_learner(ml_m, "classifier", "ml_m"),
                         if (!is.null(horizons)) as.numeric(horizons), estimand,
                         if (is.numeric(grid)) as.numeric(grid) else grid, time_encoding,
                         clipping_threshold, censoring_floor, check_count(n_folds, "n_folds", 2),
                         check_count(n_rep, "n_rep"), seed, check_count(cause, "cause"), cluster)
  out <- new_fit(res$effect, match.call())
  arm <- function(ptr) new_fit(ptr, out$call)
  out$treated <- arm(res$treated)
  out$control <- arm(res$control)
  out$curve_treated <- arm(res$curve_treated)
  out$curve_control <- arm(res$curve_control)
  if (!is.null(res$incidence_treated)) {
    out$incidence_treated <- arm(res$incidence_treated)
    out$incidence_control <- arm(res$incidence_control)
  }
  out$horizons <- res$horizons
  out$grid <- res$grid
  out$propensity <- res$propensity
  class(out) <- c("dmlrs_survival", class(out))
  out
}

# A band around a step function: the value at `times[i]` holds until `times[i + 1]`.
step_band <- function(times, lower, upper) {
  n <- length(times)
  at <- rep(times, each = 2)[-c(1, 2 * n)]
  steps <- rep(seq_len(n - 1), each = 2)
  list(x = c(at, rev(at)), y = c(lower[steps], rev(upper[steps])))
}

#' Plot estimated survival curves
#'
#' Both survival curves, or for a cumulative-incidence fit both cumulative
#' incidence curves, with pointwise confidence bands.
#'
#' @param x A fit from [dml_survival()].
#' @param level Confidence level.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @examples
#' set.seed(3)
#' n <- 800
#' x <- matrix(rnorm(n * 2), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' event_time <- rexp(n, exp(0.5 * x[, 1]) * ifelse(d == 1, 0.5, 1))
#' censoring <- rexp(n, 0.3)
#' fit <- dml_survival(pmin(event_time, censoring), as.numeric(event_time <= censoring), d, x,
#'                     ml_event = lrn_logistic(), ml_m = lrn_logistic(), horizons = 1,
#'                     grid = 12, time_encoding = "dummies")
#' plot(fit)
#' @export
plot.dmlrs_survival <- function(x, level = 0.95, ...) {
  times <- c(0, x$grid)
  incidence <- !is.null(x$incidence_treated)
  start <- if (incidence) 0 else 1
  graphics::plot(range(times), c(0, 1), type = "n", xlab = "time",
                 ylab = if (incidence) "cumulative incidence" else "survival probability", ...)
  draw <- function(fit, colour) {
    ci <- pmin(pmax(confint(fit, level = level), 0), 1)
    lower <- c(start, ci[, 1])
    upper <- c(start, ci[, 2])
    band <- step_band(times, lower, upper)
    graphics::polygon(band$x, band$y, col = grDevices::adjustcolor(colour, 0.2), border = NA)
    graphics::lines(times, c(start, unname(stats::coef(fit))), type = "s", col = colour, lwd = 2)
  }
  draw(if (incidence) x$incidence_treated else x$curve_treated, "#0173B2")
  draw(if (incidence) x$incidence_control else x$curve_control, "#DE8F05")
  graphics::legend(if (incidence) "topleft" else "topright", c("treated", "control"),
                   col = c("#0173B2", "#DE8F05"), lwd = 2, bty = "n")
  invisible(x)
}

# Synthetic difference in differences (class `dmlrs_synthdid`).

#' Synthetic difference in differences
#'
#' Synthetic difference in differences (Arkhangelsky, Athey, Hirshberg,
#' Imbens and Wager, 2021) for a block design, with synthetic control and
#' plain DiD as special cases: a port of the authors' package `synthdid` that
#' reproduces its estimates, weights and standard errors. `n_control` units
#' are never treated, the others are treated from period `n_pre + 1` on.
#'
#' Unit weights make the controls' (intercept-shifted) pre-treatment path
#' match the treated average; time weights make the pre-treatment periods
#' predict the controls' post-treatment average. With `estimator = "sc"` the
#' estimate is the synthetic control estimator, with `"did"` the unweighted
#' DiD. Time-varying covariates enter as \eqn{Y - \sum_k \beta_k X_k} with
#' \eqn{\beta} fitted jointly.
#'
#' @param Y Outcome matrix (units x periods): the control units in the first
#'   `n_control` rows, the pre-treatment periods in the first `n_pre` columns.
#'   Row and column names label the weights.
#' @param n_control Number of control units.
#' @param n_pre Number of pre-treatment periods.
#' @param X Covariates: an array of dimension units x periods x covariates
#'   (or a units x periods matrix for one covariate), or `NULL`.
#' @param estimator `"sdid"`, `"sc"` or `"did"`.
#' @param se_method Standard error computed by the fit: `"placebo"` (needs
#'   more controls than treated units), `"jackknife"` or `"bootstrap"` (both
#'   need at least two treated units), or `"none"`. Others come from
#'   [synth_se()].
#' @param replications Replications of the placebo or bootstrap.
#' @param seed Seed of the placebo or bootstrap draws: they equal `synthdid`'s
#'   after `set.seed(seed)`.
#' @param noise_level,eta_omega,eta_lambda,min_decrease,max_iter,sparsify
#'   Regularisation and solver settings, as `synthdid::synthdid_estimate()`
#'   (`NULL`: its defaults).
#' @return A `dmlrs_synthdid` object with `estimate`, `se`, `unit_weights`,
#'   `time_weights`, `beta`, `effect_curve` (the effect in each post-treatment
#'   period), and methods `coef()`, `confint()`, `print()` and `plot()`.
#' @seealso [synth_did_panel()] for a long panel.
#' @examples
#' set.seed(8)
#' Y <- matrix(rnorm(30 * 12), 30, 12) + outer(rnorm(30), 1:12 / 4)
#' Y[28:30, 9:12] <- Y[28:30, 9:12] + 1
#' fit <- synth_did(Y, n_control = 27, n_pre = 8)
#' fit
#' synth_se(fit, "jackknife")
#' plot(fit)
#' @export
synth_did <- function(Y, n_control, n_pre, X = NULL, estimator = c("sdid", "sc", "did"),
                      se_method = c("placebo", "jackknife", "bootstrap", "none"),
                      replications = 200, seed = 0, noise_level = NULL, eta_omega = NULL,
                      eta_lambda = NULL, min_decrease = NULL, max_iter = NULL, sparsify = TRUE) {
  estimator <- match.arg(estimator)
  se_method <- match.arg(se_method)
  if (!is.matrix(Y)) stop("`Y` must be a units x periods matrix", call. = FALSE)
  y <- as_numeric_matrix(Y, "Y")
  n_control <- check_count(n_control, "n_control")
  n_pre <- check_count(n_pre, "n_pre")
  if (!is.null(X)) {
    if (is.matrix(X)) X <- array(X, c(dim(X), 1))
    if (length(dim(X)) != 3 || any(dim(X)[1:2] != dim(y))) {
      stop("`X` must be an array of dimension units x periods x covariates", call. = FALSE)
    }
    storage.mode(X) <- "double"
    if (anyNA(X)) stop("`X` contains missing values", call. = FALSE)
  }
  ptr <- rs_fit_synthdid(y, n_control, n_pre, X, estimator, noise_level, eta_omega, eta_lambda,
                         min_decrease, max_iter, isTRUE(sparsify))
  units <- if (is.null(rownames(Y))) seq_len(nrow(Y)) else rownames(Y)
  periods <- if (is.null(colnames(Y))) seq_len(ncol(Y)) else colnames(Y)
  info <- rs_synth_info(ptr)
  fit <- structure(list(
    ptr = ptr,
    call = match.call(),
    estimator = estimator,
    estimate = info$estimate,
    se = NA_real_,
    se_method = se_method,
    unit_weights = stats::setNames(info$unit_weights, units[seq_len(n_control)]),
    time_weights = stats::setNames(info$time_weights, periods[seq_len(n_pre)]),
    beta = info$beta,
    effect_curve = stats::setNames(info$effect_curve, periods[-seq_len(n_pre)]),
    noise_level = info$noise_level,
    trajectories = cbind(synthetic = info$synthetic, treated = info$treated),
    periods = periods,
    n_control = n_control,
    n_pre = n_pre,
    dim = dim(y)
  ), class = "dmlrs_synthdid")
  if (se_method != "none") {
    fit$se <- tryCatch(synth_se(fit, se_method, replications, seed), error = function(e) {
      warning(sprintf("no %s standard error: %s", se_method, conditionMessage(e)), call. = FALSE)
      NA_real_
    })
  }
  fit
}

#' Synthetic difference in differences on a long panel
#'
#' Arranges a balanced long panel as `synthdid::panel.matrices()` does
#' (control units first, then treated units, each sorted by unit label;
#' periods sorted) and calls [synth_did()].
#'
#' @param data A data frame with one row per unit and period.
#' @param unit,time,outcome,treatment Column names; `treatment` is 0/1 and
#'   switches on in the same period for all treated units.
#' @param covariates Names of time-varying covariate columns, or `NULL`.
#' @param ... Passed to [synth_did()].
#' @inherit synth_did return
#' @export
synth_did_panel <- function(data, unit, time, outcome, treatment, covariates = NULL, ...) {
  cols <- c(unit, time, outcome, treatment, covariates)
  missing_cols <- setdiff(cols, names(data))
  if (length(missing_cols)) {
    stop(sprintf("columns not in `data`: %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }
  df <- as.data.frame(data)[cols]
  if (anyNA(df)) stop("missing values in the panel", call. = FALSE)
  if (!all(df[[treatment]] %in% c(0, 1))) stop("the treatment must be 0 or 1", call. = FALSE)
  units <- sort(unique(df[[unit]]))
  periods <- sort(unique(df[[time]]))
  if (nrow(df) != length(units) * length(periods) || anyDuplicated(df[c(unit, time)])) {
    stop("the panel must be balanced: one observation per unit and period", call. = FALSE)
  }
  wide <- function(col) {
    m <- matrix(NA_real_, length(units), length(periods), dimnames = list(units, periods))
    m[cbind(match(df[[unit]], units), match(df[[time]], periods))] <- df[[col]]
    m
  }
  w <- wide(treatment)
  treated <- rowSums(w) > 0
  if (!any(treated) || all(treated)) stop("there is no variation in treatment status", call. = FALSE)
  first <- which(colSums(w) > 0)[1]
  ok <- all(w[!treated, ] == 0) && all(w[, seq_len(first - 1)] == 0) &&
    all(w[treated, first:ncol(w)] == 1)
  if (!ok) stop("treatment adoption is not simultaneous", call. = FALSE)
  order <- c(which(!treated)[order(as.character(units[!treated]))],
             which(treated)[order(as.character(units[treated]))])
  y <- wide(outcome)[order, , drop = FALSE]
  x <- NULL
  if (length(covariates)) {
    x <- array(vapply(covariates, function(c) wide(c)[order, , drop = FALSE],
                      matrix(0, length(units), length(periods))),
               c(length(units), length(periods), length(covariates)))
  }
  fit <- synth_did(y, sum(!treated), first - 1, X = x, ...)
  fit$call <- match.call()
  fit
}

#' Standard errors of synthetic difference in differences
#'
#' @param object A [synth_did()] fit.
#' @param method `"placebo"`, `"jackknife"` or `"bootstrap"`.
#' @param replications Replications of the placebo or bootstrap.
#' @param seed Seed: the draws equal `synthdid`'s after `set.seed(seed)`.
#' @return The standard error.
#' @export
synth_se <- function(object, method = c("placebo", "jackknife", "bootstrap"),
                     replications = 200, seed = 0) {
  method <- match.arg(method)
  if (!inherits(object, "dmlrs_synthdid")) stop("`object` must be a synth_did() fit", call. = FALSE)
  if (!rs_is_valid_synth(object$ptr)) {
    stop("this fit was restored from a saved session and has no Rust state; refit it",
         call. = FALSE)
  }
  rs_synth_se(object$ptr, method, check_count(replications, "replications"), as.integer(seed))
}

synth_name <- function(estimator) {
  c(sdid = "synthetic DiD", sc = "synthetic control", did = "DiD")[[estimator]]
}

#' @export
coef.dmlrs_synthdid <- function(object, ...) {
  stats::setNames(object$estimate, synth_name(object$estimator))
}

#' @export
confint.dmlrs_synthdid <- function(object, parm, level = 0.95, ...) {
  z <- stats::qnorm(0.5 + level / 2)
  ci <- matrix(object$estimate + c(-1, 1) * z * object$se, 1)
  pct <- paste(format(100 * c((1 - level) / 2, (1 + level) / 2), trim = TRUE, digits = 3), "%")
  dimnames(ci) <- list(synth_name(object$estimator), pct)
  ci
}

#' @export
print.dmlrs_synthdid <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  n_units <- x$dim[1]
  n_periods <- x$dim[2]
  cat(sprintf("%s: %d control and %d treated units, %d pre- and %d post-treatment periods\n\n",
              synth_name(x$estimator), x$n_control, n_units - x$n_control, x$n_pre,
              n_periods - x$n_pre))
  t <- x$estimate / x$se
  table <- cbind(x$estimate, x$se, t, 2 * stats::pnorm(-abs(t)))
  dimnames(table) <- list(synth_name(x$estimator), c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  stats::printCoefmat(table, digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  if (x$se_method != "none") cat(sprintf("Standard error: %s\n", x$se_method))
  invisible(x)
}

#' Plot synthetic difference in differences
#'
#' The treated average and the synthetic control over time, with the
#' treatment start marked and the time weights as bars.
#'
#' @param x A [synth_did()] fit.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.dmlrs_synthdid <- function(x, ...) {
  tr <- x$trajectories
  t <- seq_len(nrow(tr))
  graphics::matplot(t, tr, type = "l", lty = 1, lwd = 2, col = c("#DE8F05", "#0173B2"),
                    xaxt = "n", xlab = "period", ylab = "outcome", ...)
  graphics::axis(1, at = t, labels = x$periods)
  graphics::abline(v = x$n_pre + 0.5, col = "grey50", lty = 2)
  lam <- x$time_weights
  if (sum(lam) > 0) {
    usr <- graphics::par("usr")
    graphics::rect(seq_len(x$n_pre) - 0.4, usr[3], seq_len(x$n_pre) + 0.4,
                   usr[3] + lam * 0.15 * (usr[4] - usr[3]), col = "grey80", border = NA)
  }
  graphics::legend("topright", c("treated", paste(synth_name(x$estimator), "control")),
                   col = c("#0173B2", "#DE8F05"), lwd = 2, bty = "n")
  invisible(x)
}

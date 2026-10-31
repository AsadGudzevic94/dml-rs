# Methods for fitted models (class `dmlrs_fit`).

# The Rust state of a fit. An object restored with readRDS() keeps its
# estimates (coef(), summary()) but no longer has the Rust state.
fit_ptr <- function(object) {
  if (!rs_is_valid(object$ptr)) {
    stop("this fit was restored from a saved session and has no Rust state; refit the model ",
         "to compute intervals, the bootstrap or a sensitivity analysis", call. = FALSE)
  }
  object$ptr
}

coef_table <- function(object) {
  table <- cbind(object$coefficients, object$se, object$t, object$p)
  dimnames(table) <- list(names(object$coefficients),
                          c("Estimate", "Std. Error", "t value", "Pr(>|t|)"))
  table
}

#' @export
print.dmlrs_fit <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("Double machine learning: %s\n", x$model))
  cat(sprintf("%d observations, %d folds, %d repetition%s\n\n", x$n_obs, x$n_folds, x$n_rep,
              if (x$n_rep == 1) "" else "s"))
  stats::printCoefmat(coef_table(x), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  invisible(x)
}

#' @export
summary.dmlrs_fit <- function(object, ...) {
  structure(list(model = object$model, coefficients = coef_table(object), n_obs = object$n_obs,
                 n_folds = object$n_folds, n_rep = object$n_rep, call = object$call),
            class = "summary.dmlrs_fit")
}

#' @export
print.summary.dmlrs_fit <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Call:\n")
  print(x$call)
  cat(sprintf("\nDouble machine learning: %s (%d observations, %d folds, %d repetition%s)\n\n",
              x$model, x$n_obs, x$n_folds, x$n_rep, if (x$n_rep == 1) "" else "s"))
  stats::printCoefmat(x$coefficients, digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  invisible(x)
}

#' @export
nobs.dmlrs_fit <- function(object, ...) object$n_obs

#' Confidence intervals for DML estimates
#'
#' Pointwise intervals, or simultaneous intervals for all treatments from the
#' multiplier bootstrap (`joint = TRUE`), as DoubleML's `confint()`.
#'
#' @param object A fitted model.
#' @param parm Treatments (names or indices); all by default.
#' @param level Confidence level.
#' @param joint Simultaneous intervals from the multiplier bootstrap.
#' @param method Bootstrap weights: `"normal"`, `"Bayes"` or `"wild"`.
#' @param n_boot Number of bootstrap draws.
#' @param seed Seed of the bootstrap.
#' @param ... Not used.
#' @return A matrix with one row per treatment.
#' @examples
#' set.seed(3)
#' n <- 400
#' x <- matrix(rnorm(n * 3), n)
#' d <- cbind(d1 = x[, 1] + rnorm(n), d2 = x[, 2] + rnorm(n))
#' y <- 0.5 * d[, 1] - 0.2 * d[, 2] + x[, 3] + rnorm(n)
#' fit <- dml_plr(y, d, x, ml_l = lrn_ols())
#' confint(fit, joint = TRUE, n_boot = 1000)
#' @export
confint.dmlrs_fit <- function(object, parm, level = 0.95, joint = FALSE,
                              method = c("normal", "Bayes", "wild"), n_boot = 500, seed = 0, ...) {
  method <- match.arg(method)
  ci <- if (joint) {
    rs_joint_confint(fit_ptr(object), method, check_count(n_boot, "n_boot"), seed, level)
  } else {
    rs_confint(fit_ptr(object), level)
  }
  pct <- paste(format(100 * c((1 - level) / 2, (1 + level) / 2), trim = TRUE, digits = 3), "%")
  dimnames(ci) <- list(names(object$coefficients), pct)
  if (missing(parm)) ci else ci[parm, , drop = FALSE]
}

#' Sensitivity analysis for omitted confounding
#'
#' Bounds on the effect and robustness values when the confounding is as
#' strong as `cf_y` (share of the residual outcome variance) and `cf_d`
#' (share of the residual Riesz representer variance), with correlation
#' `rho` (Chernozhukov et al., 2022), as DoubleML's `sensitivity_analysis()`.
#' Available for PLR, IRM and DID fits.
#'
#' @param object A fitted model.
#' @param cf_y,cf_d Confounding strengths.
#' @param rho Correlation of the confounding in the outcome and treatment.
#' @param level Confidence level of the bounds.
#' @param null_hypothesis Value of the effect under the null hypothesis.
#' @return A data frame with one row per treatment: the bounds on the effect
#'   (`theta_lower`, `theta_upper`), their confidence bounds (`ci_lower`,
#'   `ci_upper`) and the robustness values `rv` and `rva` (in percent).
#' @examples
#' set.seed(4)
#' n <- 500
#' x <- matrix(rnorm(n * 3), n)
#' d <- x[, 1] + rnorm(n)
#' y <- 0.5 * d + x[, 2] + rnorm(n)
#' sensitivity_analysis(dml_plr(y, d, x, ml_l = lrn_ols()))
#' @export
sensitivity_analysis <- function(object, cf_y = 0.03, cf_d = 0.03, rho = 1, level = 0.95,
                                 null_hypothesis = 0) {
  if (!inherits(object, "dmlrs_fit")) stop("`object` must be a fitted dmlrs model", call. = FALSE)
  res <- rs_sensitivity(fit_ptr(object), cf_y, cf_d, rho, level, as.numeric(null_hypothesis))
  out <- data.frame(ci_lower = res$ci_lower, theta_lower = res$theta_lower,
                    theta = unname(object$coefficients), theta_upper = res$theta_upper,
                    ci_upper = res$ci_upper, rv = 100 * res$rv, rva = 100 * res$rva,
                    row.names = names(object$coefficients))
  attr(out, "params") <- c(cf_y = cf_y, cf_d = cf_d, rho = rho, level = level)
  structure(out, class = c("dmlrs_sensitivity", "data.frame"))
}

#' @export
print.dmlrs_sensitivity <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  p <- attr(x, "params")
  cat(sprintf("Sensitivity analysis: cf_y = %g, cf_d = %g, rho = %g, level = %g\n\n",
              p[["cf_y"]], p[["cf_d"]], p[["rho"]], p[["level"]]))
  table <- x
  attr(table, "params") <- NULL
  class(table) <- "data.frame"
  print(table, digits = digits)
  invisible(x)
}

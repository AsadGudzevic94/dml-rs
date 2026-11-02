# Dose-response curves of a continuous treatment.

#' Dose-response curves of a continuous treatment
#'
#' The average dose-response function \eqn{\beta(t) = E[Y(t)]} with the
#' kernel-localised double machine learning estimator of Colangelo and Lee
#' (2023). At every dose \eqn{t} of `grid` the estimate is the mean of the
#' doubly robust score
#' \deqn{\gamma(t, X) + K_h(D - t) / f(t \mid X) \cdot (Y - \gamma(t, X))}
#' with cross-fitted \eqn{\gamma(t, X) = E[Y \mid D = t, X]} and generalised
#' propensity score \eqn{f(t \mid X)}, the conditional density of the dose.
#' The estimates converge at the nonparametric rate \eqn{\sqrt{nh}}. All grid
#' points share the sample splits, so `confint(fit$curve, joint = TRUE)` gives
#' a uniform confidence band and [dose_contrast()] has correct joint
#' inference.
#'
#' @inheritParams dml_plr
#' @param d The dose (continuous treatment).
#' @param ml_g Learner for \eqn{E[Y \mid D, X]}, fitted on `cbind(d, x)`.
#' @param ml_m Learner for the density model: with `density = "kernel"`
#'   (Colangelo and Lee's MultiGPS) it learns \eqn{E[\phi((D - t)/h_1) \mid X]}
#'   at every grid point; with `density = "normal"`, the mean \eqn{E[D \mid X]}
#'   of a normal model with constant variance.
#' @param grid Doses at which to estimate \eqn{\beta(t)} (default: the deciles
#'   of `d`).
#' @param kernel `"epanechnikov"` or `"gaussian"`.
#' @param bandwidth Bandwidth \eqn{h} in units of the dose. Default:
#'   `bandwidth_scale * sd(d) * n^(-1/5)`; scales below the AMSE-optimal one
#'   undersmooth, which keeps the kernel bias small next to the standard error.
#' @param bandwidth_scale Constant of the default bandwidth.
#' @param density `"kernel"` or `"normal"`.
#' @param density_bandwidth Bandwidth \eqn{h_1} of the kernel density model
#'   (default \eqn{h}).
#' @param density_floor Densities of the standardised dose `d / sd(d)` below
#'   this value are raised to it.
#' @param partial_effects Also estimate \eqn{\partial\beta(t)/\partial t} by
#'   \eqn{(\hat\beta(t + \eta/2) - \hat\beta(t - \eta/2))/\eta}.
#' @param eta The step \eqn{\eta} (default \eqn{h n^{-1/6}}); setting it turns
#'   on `partial_effects`.
#' @return A `dmlrs_dose` object: `curve` (a `dmlrs_fit` with one estimate per
#'   grid point), `partial` (a `dmlrs_fit` of the partial effects, or `NULL`),
#'   `grid`, `bandwidth` and `eta`. `plot()` draws the curve with a confidence
#'   band.
#' @examples
#' set.seed(10)
#' n <- 2000
#' x <- matrix(rnorm(n * 2), n)
#' dose <- x[, 1] + rnorm(n)
#' y <- sin(dose) + x[, 2] + rnorm(n)
#' fit <- dml_dose_response(y, dose, x, ml_g = lrn_random_forest(n_estimators = 100),
#'                          ml_m = lrn_ols(), grid = c(-1, 0, 1))
#' fit
#' dose_contrast(fit, reference = 0)
#' @export
dml_dose_response <- function(y, d, x, ml_g = lrn_random_forest(), ml_m = lrn_lasso_cv(),
                              grid = NULL, kernel = c("epanechnikov", "gaussian"),
                              bandwidth = NULL, bandwidth_scale = 1,
                              density = c("kernel", "normal"), density_bandwidth = NULL,
                              density_floor = 0.01, partial_effects = FALSE, eta = NULL,
                              n_folds = 5, n_rep = 1, seed = 0) {
  kernel <- match.arg(kernel)
  density <- match.arg(density)
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single dose", call. = FALSE)
  res <- rs_fit_dose_response(data$y, data$d, data$x, check_learner(ml_g, "regressor", "ml_g"),
                              check_learner(ml_m, "regressor", "ml_m"),
                              if (is.null(grid)) NULL else as.numeric(grid), kernel, bandwidth,
                              bandwidth_scale, density, density_bandwidth, density_floor,
                              isTRUE(partial_effects), eta, check_count(n_folds, "n_folds", 2),
                              check_count(n_rep, "n_rep"), seed)
  call <- match.call()
  structure(list(ptr = res$fit, call = call, curve = new_fit(res$curve, call),
                 partial = if (is.null(res$partial)) NULL else new_fit(res$partial, call),
                 grid = res$grid, bandwidth = res$bandwidth,
                 eta = if (is.null(res$eta) || is.na(res$eta)) NULL else res$eta),
            class = "dmlrs_dose")
}

#' @export
print.dmlrs_dose <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("Dose-response curve: %d grid points, bandwidth %s\n\n", length(x$grid),
              format(x$bandwidth, digits = digits)))
  stats::printCoefmat(coef_table(x$curve), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  if (!is.null(x$partial)) cat("\nPartial effects: see $partial\n")
  invisible(x)
}

#' Contrasts of a dose-response curve
#'
#' \eqn{\beta(t) - \beta(t_0)} for every grid point \eqn{t} other than the
#' reference \eqn{t_0}, with standard errors from the joint scores.
#'
#' @param object A [dml_dose_response()] fit.
#' @param reference A grid point.
#' @return A `dmlrs_fit` object.
#' @export
dose_contrast <- function(object, reference) {
  if (!inherits(object, "dmlrs_dose")) stop("`object` must be a dml_dose_response() fit",
                                            call. = FALSE)
  new_fit(rs_dose_contrast(live_ptr(object$ptr, "dose"), as.numeric(reference)), match.call())
}

#' Plot a dose-response curve
#'
#' @param x A [dml_dose_response()] fit.
#' @param type `"curve"` or `"partial"` (fit with `partial_effects = TRUE`).
#' @param level Level of the confidence band.
#' @param joint Uniform band from the multiplier bootstrap instead of pointwise
#'   intervals.
#' @param ... Passed to [graphics::plot()].
#' @return `x`, invisibly.
#' @export
plot.dmlrs_dose <- function(x, type = c("curve", "partial"), level = 0.95, joint = FALSE, ...) {
  type <- match.arg(type)
  fit <- if (type == "curve") x$curve else x$partial
  if (is.null(fit)) stop("fit with partial_effects = TRUE to plot partial effects", call. = FALSE)
  ci <- confint(fit, level = level, joint = joint)
  est <- unname(stats::coef(fit))
  graphics::plot(x$grid, est, type = "n", ylim = range(ci), xlab = "dose t",
                 ylab = if (type == "curve") "E[Y(t)]" else "dE[Y(t)]/dt", ...)
  graphics::polygon(c(x$grid, rev(x$grid)), c(ci[, 1], rev(ci[, 2])),
                    col = grDevices::adjustcolor("#0173B2", 0.25), border = NA)
  graphics::lines(x$grid, est, col = "#0173B2", lwd = 2, type = "o", pch = 16)
  invisible(x)
}

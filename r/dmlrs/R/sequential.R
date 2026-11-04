# Anytime-valid confidence sequences and prediction-powered inference
# (src/rust/src/sequential.rs).

#' Anytime-valid confidence sequences
#'
#' An interval at this fit's sample size that, recomputed at every sample size,
#' covers the effect at all of them at once with probability `level`, so a
#' study can be checked as often as wanted and stopped at any time (asymptotic
#' confidence sequences of Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas,
#' 2024). The half-width after \eqn{t} observations is
#' \eqn{\widehat{se}_t \sqrt{2(u+1)/u \log(\sqrt{u+1}/\alpha)}} with
#' \eqn{u = t\rho^2}; `opt_time` sets \eqn{\rho} so that the sequence is
#' tightest at that sample size (about 3.0 standard errors on each side at
#' 95%). Fix `opt_time` before looking at the data. Clustered fits count
#' clusters. To follow a study over several looks use [monitor_study()].
#'
#' @param object A fitted model.
#' @param level Confidence level.
#' @param opt_time Sample size at which the sequence is tightest (default:
#'   this fit's sample size).
#' @return `confidence_sequence()`: a matrix with one row per treatment.
#'   `anytime_pvalue()`: anytime-valid p-values for a zero effect, below
#'   `1 - level` exactly when the sequence excludes zero.
#' @examples
#' set.seed(5)
#' n <- 1000
#' x <- matrix(rnorm(n * 3), n)
#' d <- x[, 1] + rnorm(n)
#' y <- 0.3 * d + x[, 2] + rnorm(n)
#' fit <- dml_plr(y, d, x, ml_l = lrn_ols())
#' confidence_sequence(fit, opt_time = 500)
#' anytime_pvalue(fit, opt_time = 500)
#' @export
confidence_sequence <- function(object, level = 0.95, opt_time = NULL) {
  res <- sequence_of(object, level, opt_time)
  ci <- res$interval
  pct <- paste(format(100 * c((1 - level) / 2, (1 + level) / 2), trim = TRUE, digits = 3), "%")
  dimnames(ci) <- list(names(object$coefficients), pct)
  ci
}

#' @rdname confidence_sequence
#' @export
anytime_pvalue <- function(object, level = 0.95, opt_time = NULL) {
  stats::setNames(sequence_of(object, level, opt_time)$p, names(object$coefficients))
}

sequence_of <- function(object, level, opt_time) {
  if (!inherits(object, "dmlrs_fit")) stop("`object` must be a fitted dmlrs model", call. = FALSE)
  rs_confidence_sequence(fit_ptr(object), level, check_opt_time(opt_time))
}

check_opt_time <- function(opt_time) {
  if (is.null(opt_time)) NULL else as.numeric(check_count(opt_time, "opt_time"))
}

#' Monitor a study with confidence sequences
#'
#' Confidence sequences over the looks of a study: refit the model on all data
#' so far at each look and pass the fits in order. The intervals of all looks
#' hold together, so their running intersection (`cs_lower`, `cs_upper`) is a
#' confidence sequence too, and the running minimum of the anytime p-values
#' (`p_anytime`) is a valid p-value whenever the study stops. See
#' [confidence_sequence()].
#'
#' @param fits A list of fitted models, one per look, on growing samples (rows
#'   in arrival order).
#' @param level Confidence level.
#' @param opt_time Sample size at which the sequence is tightest; default the
#'   sample size of the first look. Fix it before looking at the data.
#' @return A `dmlrs_monitor` object: `table`, a data frame with one row per
#'   look and treatment (`n`, `term`, `estimate`, `std_error`, `cs_lower`,
#'   `cs_upper`, `p_anytime`), `first_exclusion` (per treatment, the sample size
#'   of the first look whose running interval excludes zero, or `NA`), `level`
#'   and `opt_time`. `plot()` draws the estimates and the running sequence.
#' @examples
#' set.seed(6)
#' n <- 3000
#' x <- matrix(rnorm(n * 3), n)
#' d <- x[, 1] + rnorm(n)
#' y <- 0.15 * d + x[, 2] + rnorm(n)
#' looks <- c(250, 500, 1000, 2000, 3000)
#' fits <- lapply(looks, function(t) dml_plr(y[1:t], d[1:t], x[1:t, ], ml_l = lrn_ols()))
#' mon <- monitor_study(fits, opt_time = 500)
#' mon
#' plot(mon)
#' @export
monitor_study <- function(fits, level = 0.95, opt_time = NULL) {
  if (inherits(fits, "dmlrs_fit") || !is.list(fits) || length(fits) == 0 ||
      !all(vapply(fits, inherits, logical(1), "dmlrs_fit"))) {
    stop("`fits` must be a non-empty list of fitted dmlrs models", call. = FALSE)
  }
  res <- rs_monitor(lapply(fits, fit_ptr), level, check_opt_time(opt_time))
  looks <- length(res$n)
  k <- length(res$terms)
  table <- data.frame(
    n = rep(res$n, each = k), term = rep(res$terms, times = looks),
    estimate = as.vector(t(res$coef)), std_error = as.vector(t(res$se)),
    cs_lower = as.vector(t(res$lower)), cs_upper = as.vector(t(res$upper)),
    p_anytime = as.vector(t(res$p)), stringsAsFactors = FALSE
  )
  first <- ifelse(res$first_exclusion > 0, res$n[pmax(res$first_exclusion, 1)], NA_integer_)
  structure(list(table = table, first_exclusion = stats::setNames(first, res$terms),
                 level = level, opt_time = res$opt_time),
            class = "dmlrs_monitor")
}

#' @export
print.dmlrs_monitor <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("Confidence sequences (%g%%, tightest at n = %d) over %d looks\n\n",
              100 * x$level, x$opt_time, length(unique(x$table$n))))
  print(x$table, digits = digits, row.names = FALSE)
  for (term in names(x$first_exclusion)) {
    n <- x$first_exclusion[[term]]
    cat(sprintf("\n%s: %s", term, if (is.na(n)) "zero not excluded yet"
                else sprintf("zero excluded from n = %d on", n)))
  }
  cat("\n")
  invisible(x)
}

#' @export
plot.dmlrs_monitor <- function(x, term = NULL, value = 0, ...) {
  if (is.null(term)) term <- x$table$term[1]
  rows <- x$table[x$table$term == term, ]
  if (nrow(rows) == 0) stop(sprintf("no term %s", term), call. = FALSE)
  ylim <- range(c(rows$cs_lower, rows$cs_upper, rows$estimate, value))
  graphics::plot(rows$n, rows$estimate, type = "n", ylim = ylim, xlab = "sample size",
                 ylab = term, ...)
  n <- rows$n
  steps <- c(n[-1], n[length(n)])
  graphics::rect(n, rows$cs_lower, steps, rows$cs_upper,
                 col = grDevices::adjustcolor("#0173B2", 0.25), border = NA)
  graphics::lines(n, rows$estimate, col = "#0173B2", lwd = 2, type = "o", pch = 16)
  if (!is.null(value)) graphics::abline(h = value, lty = 2, col = "grey40")
  invisible(x)
}

#' Prediction-powered inference
#'
#' Means and linear or logistic regression coefficients from a small labelled
#' sample and many predictions of the outcome (from a machine-learning model,
#' an LLM, a simulator), unbiased whatever the quality of the predictions
#' (Angelopoulos, Bates, Fannjiang, Jordan and Zrnic, 2023). The estimate from
#' the predictions is corrected by the prediction errors on the labelled
#' sample; for a mean,
#' \eqn{\hat\theta = \lambda\,\overline{\hat y^u} + \overline{y - \lambda\hat y}}.
#' `lambda = NULL` (PPI++, Angelopoulos, Duchi and Zrnic, 2023) estimates the
#' variance-minimising weight in \eqn{[0, 1]}, so the estimate is
#' (asymptotically) never less precise than the labelled-only one; `lambda = 1`
#' is the original PPI and `lambda = 0` the classical estimate. Matches the
#' Python package `ppi_py`.
#'
#' With `learner`, the predictions are cross-fitted on the labelled sample
#' from `features` (cross-prediction-powered inference, Zrnic and Candès,
#' 2024): each labelled unit is predicted by a model that did not see it, each
#' unlabelled unit gets the average of the fold models' predictions.
#'
#' @param y Labelled outcomes (0/1 for `estimand = "logistic"`).
#' @param yhat,yhat_unlabeled Predictions for the labelled and the unlabelled
#'   sample (probabilities for `"logistic"`); not used with `learner`.
#' @param x,x_unlabeled Covariates of the regression in both samples
#'   (`"ols"`, `"logistic"`).
#' @param estimand `"mean"`, `"ols"` or `"logistic"`.
#' @param lambda Weight of the predictions in \eqn{[0, 1]}, or `NULL` to tune it.
#' @param fit_intercept Add an intercept to the regressions.
#' @param learner Optional dmlrs learner to cross-fit the predictions (a
#'   classifier's probabilities for `"logistic"` or a classification learner).
#' @param features,features_unlabeled Predictors of the learner in both
#'   samples (default `x`, `x_unlabeled`).
#' @param n_folds,seed Folds and seed of the cross-fitting.
#' @return A `dmlrs_ppi` object, which is also a `dmlrs_fit` (`coef()`,
#'   `confint()`, `summary()`, `confidence_sequence()` work on it), with
#'   `lambda`, `classical` (the labelled-only estimate as a `dmlrs_fit`),
#'   `effective_n` (the labelled sample size the classical estimate would need
#'   for the same precision), `n_labeled`, `n_unlabeled` and, when
#'   cross-fitted, `yhat` and `yhat_unlabeled`.
#' @examples
#' set.seed(7)
#' n <- 200
#' N <- 5000
#' z <- rnorm(n + N)
#' y <- 1 + z + rnorm(n + N, sd = 0.5)
#' yhat <- 1.2 + 0.9 * z
#' fit <- ppi(y[1:n], yhat[1:n], yhat[-(1:n)])
#' fit
#' confint(fit$classical)
#' @export
ppi <- function(y, yhat = NULL, yhat_unlabeled = NULL, x = NULL, x_unlabeled = NULL,
                estimand = c("mean", "ols", "logistic"), lambda = NULL, fit_intercept = TRUE,
                learner = NULL, features = NULL, features_unlabeled = NULL, n_folds = 5,
                seed = 0) {
  estimand <- match.arg(estimand)
  y <- as_numeric_vector(y, "y")
  n <- length(y)
  names <- NULL
  if (estimand == "mean") {
    if (!is.null(x) || !is.null(x_unlabeled)) stop("the mean takes no covariates `x`", call. = FALSE)
  } else {
    if (is.null(x) || is.null(x_unlabeled)) {
      stop(sprintf("estimand = \"%s\" needs `x` and `x_unlabeled`", estimand), call. = FALSE)
    }
    x <- as_numeric_matrix(x, "x", n)
    x_unlabeled <- as_numeric_matrix(x_unlabeled, "x_unlabeled")
    names <- colnames(x)
    if (is.null(names)) names <- paste0("x", seq_len(ncol(x)))
    if (fit_intercept) {
      x <- cbind(1, x)
      x_unlabeled <- cbind(1, x_unlabeled)
      names <- c("(Intercept)", names)
    }
  }
  if (!is.null(lambda) && (length(lambda) != 1 || !is.numeric(lambda))) {
    stop("`lambda` must be NULL or a number in [0, 1]", call. = FALSE)
  }
  classifier <- FALSE
  if (!is.null(learner)) {
    if (!is.null(yhat) || !is.null(yhat_unlabeled)) {
      stop("pass either predictions or a learner, not both", call. = FALSE)
    }
    classifier <- estimand == "logistic" || learner$type %in% c("logistic", "logistic_cv")
    learner <- check_learner(learner, if (classifier) "classifier" else "regressor", "learner")
    if (is.null(features)) features <- x
    if (is.null(features_unlabeled)) features_unlabeled <- x_unlabeled
    if (is.null(features) || is.null(features_unlabeled)) {
      stop("a learner needs `features` and `features_unlabeled` to predict from", call. = FALSE)
    }
    features <- as_numeric_matrix(features, "features", n)
    features_unlabeled <- as_numeric_matrix(features_unlabeled, "features_unlabeled")
  } else {
    if (is.null(yhat) || is.null(yhat_unlabeled)) {
      stop("pass `yhat` and `yhat_unlabeled`, or a learner", call. = FALSE)
    }
    yhat <- as_numeric_vector(yhat, "yhat", n)
    yhat_unlabeled <- as_numeric_vector(yhat_unlabeled, "yhat_unlabeled")
  }
  res <- rs_fit_ppi(estimand, y, yhat, yhat_unlabeled, x, x_unlabeled, names,
                    if (is.null(lambda)) NULL else as.numeric(lambda), learner, classifier,
                    features, features_unlabeled, check_count(n_folds, "n_folds", 2), seed)
  call <- match.call()
  fit <- new_fit(res$estimate, call)
  fit$lambda <- res$lambda
  fit$classical <- new_fit(res$classical, call)
  fit$effective_n <- stats::setNames(res$effective_n, names(fit$coefficients))
  fit$n_labeled <- res$n_labeled
  fit$n_unlabeled <- res$n_unlabeled
  fit$yhat <- res$yhat
  fit$yhat_unlabeled <- res$yhat_unlabeled
  class(fit) <- c("dmlrs_ppi", class(fit))
  fit
}

#' @export
print.dmlrs_ppi <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("Prediction-powered inference: %d labelled, %d unlabelled, lambda = %s\n\n",
              x$n_labeled, x$n_unlabeled, format(x$lambda, digits = digits)))
  stats::printCoefmat(coef_table(x), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  cat("
Against the labelled sample alone:
")
  compare <- cbind("Classical estimate" = x$classical$coefficients,
                   "Classical SE" = x$classical$se, "Effective n" = x$effective_n)
  print(compare, digits = digits)
  invisible(x)
}

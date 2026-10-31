# DML models. Each validates its input, fits in Rust and returns a `dmlrs_fit`.

as_numeric_matrix <- function(v, arg, n = NULL) {
  if (is.data.frame(v)) {
    numeric <- vapply(v, is.numeric, logical(1))
    if (!all(numeric)) {
      stop(sprintf("`%s` has non-numeric columns (%s); encode them, e.g. with model.matrix()",
                   arg, paste(names(v)[!numeric], collapse = ", ")), call. = FALSE)
    }
    v <- as.matrix(v)
  }
  if (is.null(dim(v))) v <- matrix(as.numeric(v), ncol = 1)
  if (!is.numeric(v)) stop(sprintf("`%s` must be numeric", arg), call. = FALSE)
  storage.mode(v) <- "double"
  if (anyNA(v) || any(!is.finite(v))) stop(sprintf("`%s` contains missing or infinite values", arg),
                                           call. = FALSE)
  if (!is.null(n) && nrow(v) != n) {
    stop(sprintf("`%s` has %d rows but y has %d observations", arg, nrow(v), n), call. = FALSE)
  }
  v
}

as_numeric_vector <- function(v, arg, n = NULL) {
  m <- as_numeric_matrix(v, arg, n)
  if (ncol(m) != 1) stop(sprintf("`%s` must be a vector", arg), call. = FALSE)
  as.vector(m)
}

check_binary <- function(v, arg) {
  if (!all(v %in% c(0, 1))) stop(sprintf("`%s` must be binary (0/1)", arg), call. = FALSE)
}

prepare <- function(y, d, x) {
  y <- as_numeric_vector(y, "y")
  n <- length(y)
  d <- as_numeric_matrix(d, "d", n)
  x <- as_numeric_matrix(x, "x", n)
  names <- colnames(d)
  if (is.null(names)) names <- if (ncol(d) == 1) "d" else paste0("d", seq_len(ncol(d)))
  list(y = y, d = d, x = x, names = as.character(names))
}

check_count <- function(v, arg, min = 1) {
  if (length(v) != 1 || !is.numeric(v) || v < min || v != round(v)) {
    stop(sprintf("`%s` must be a whole number >= %d", arg, min), call. = FALSE)
  }
  as.integer(v)
}

new_fit <- function(ptr, call) {
  s <- rs_summary(ptr)
  structure(list(
    ptr = ptr,
    call = call,
    model = s$model,
    coefficients = stats::setNames(s$coef, s$treatments),
    se = stats::setNames(s$se, s$treatments),
    t = stats::setNames(s$t, s$treatments),
    p = stats::setNames(s$p, s$treatments),
    all_coef = s$all_coef,
    all_se = s$all_se,
    n_obs = s$n_obs,
    n_folds = s$n_folds,
    n_rep = s$n_rep
  ), class = "dmlrs_fit")
}

#' Partially linear regression
#'
#' Double / debiased machine learning for the partially linear model
#' \eqn{Y = D\theta + g(X) + \varepsilon}, as `DoubleMLPLR`. The nuisance
#' functions are cross-fitted in Rust.
#'
#' @param y Outcome (numeric vector).
#' @param d Treatment: a numeric vector, or a matrix with one column per
#'   treatment (column names become the treatment names).
#' @param x Covariates: a numeric matrix or data frame.
#' @param ml_l Learner for \eqn{E[Y \mid X]} (see [learners]).
#' @param ml_m Learner for \eqn{E[D \mid X]}.
#' @param ml_g Learner for \eqn{g(X)}, used by the `"IV-type"` score
#'   (default: `ml_l`).
#' @param score `"partialling out"` or `"IV-type"`.
#' @param n_folds Number of cross-fitting folds.
#' @param n_rep Number of repetitions of the sample split; estimates are
#'   aggregated with the median, as in DoubleML.
#' @param seed Seed of the sample splits.
#' @return A `dmlrs_fit` object; see [stats::coef()], [confint.dmlrs_fit()], [base::summary()] and
#'   [sensitivity_analysis()].
#' @examples
#' set.seed(1)
#' n <- 500
#' x <- matrix(rnorm(n * 5), n)
#' d <- x[, 1] + rnorm(n)
#' y <- 0.5 * d + x[, 2] + rnorm(n)
#' fit <- dml_plr(y, d, x, ml_l = lrn_lasso_cv())
#' fit
#' confint(fit)
#' @export
dml_plr <- function(y, d, x, ml_l = lrn_lasso_cv(), ml_m = ml_l, ml_g = NULL,
                    score = c("partialling out", "IV-type"), n_folds = 5, n_rep = 1, seed = 0) {
  score <- match.arg(score)
  data <- prepare(y, d, x)
  ml_g <- if (is.null(ml_g)) NULL else check_learner(ml_g, "regressor", "ml_g")
  ptr <- rs_fit_plr(data$y, data$d, data$x, data$names, check_learner(ml_l, "regressor", "ml_l"),
                    check_learner(ml_m, "regressor", "ml_m"), ml_g, score,
                    check_count(n_folds, "n_folds", 2), check_count(n_rep, "n_rep"), seed)
  new_fit(ptr, match.call())
}

#' Interactive regression model
#'
#' Average treatment effect (`"ATE"`) or average treatment effect on the
#' treated (`"ATTE"`) of a binary treatment, as `DoubleMLIRM`.
#'
#' @inheritParams dml_plr
#' @param d Binary treatment (0/1).
#' @param ml_g Learner for \eqn{E[Y \mid D, X]} (fitted on each treatment arm).
#' @param ml_m Classifier for the propensity score \eqn{P(D = 1 \mid X)}.
#' @param score `"ATE"` or `"ATTE"`.
#' @param clipping_threshold Propensity scores are clipped to
#'   \eqn{[c, 1 - c]}.
#' @param normalize_ipw Normalise the inverse-probability weights.
#' @inherit dml_plr return
#' @examples
#' set.seed(2)
#' n <- 1000
#' x <- matrix(rnorm(n * 4), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' y <- d + x[, 1] + rnorm(n)
#' dml_irm(y, d, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
#' @export
dml_irm <- function(y, d, x, ml_g = lrn_lasso_cv(), ml_m = lrn_logistic_cv(),
                    score = c("ATE", "ATTE"), clipping_threshold = 0.01, normalize_ipw = FALSE,
                    n_folds = 5, n_rep = 1, seed = 0) {
  score <- match.arg(score)
  data <- prepare(y, d, x)
  check_binary(data$d, "d")
  ptr <- rs_fit_irm(data$y, data$d, data$x, data$names, check_learner(ml_g, "regressor", "ml_g"),
                    check_learner(ml_m, "classifier", "ml_m"), score, clipping_threshold,
                    isTRUE(normalize_ipw), check_count(n_folds, "n_folds", 2),
                    check_count(n_rep, "n_rep"), seed)
  new_fit(ptr, match.call())
}

#' Partially linear IV regression
#'
#' The partially linear model with an endogenous treatment and one
#' instrument \eqn{Z}, as `DoubleMLPLIV` (`"partialling out"` score).
#'
#' @inheritParams dml_plr
#' @param z Instrument (numeric vector).
#' @param ml_l Learner for \eqn{E[Y \mid X]}.
#' @param ml_m Learner for \eqn{E[Z \mid X]}.
#' @param ml_r Learner for \eqn{E[D \mid X]}.
#' @inherit dml_plr return
#' @export
dml_pliv <- function(y, d, z, x, ml_l = lrn_lasso_cv(), ml_m = ml_l, ml_r = ml_l,
                     n_folds = 5, n_rep = 1, seed = 0) {
  data <- prepare(y, d, x)
  z <- as_numeric_vector(z, "z", length(data$y))
  ptr <- rs_fit_pliv(data$y, data$d, data$x, z, data$names,
                     check_learner(ml_l, "regressor", "ml_l"),
                     check_learner(ml_m, "regressor", "ml_m"),
                     check_learner(ml_r, "regressor", "ml_r"),
                     check_count(n_folds, "n_folds", 2), check_count(n_rep, "n_rep"), seed)
  new_fit(ptr, match.call())
}

#' Interactive IV model
#'
#' Local average treatment effect of a binary treatment with a binary
#' instrument, as `DoubleMLIIVM`.
#'
#' @inheritParams dml_plr
#' @param d Binary treatment (0/1).
#' @param z Binary instrument (0/1).
#' @param ml_g Learner for \eqn{E[Y \mid Z, X]}.
#' @param ml_m Classifier for \eqn{P(Z = 1 \mid X)}.
#' @param ml_r Classifier for \eqn{P(D = 1 \mid Z, X)}.
#' @param subgroups Named logical vector: whether there can be
#'   `always_takers` and `never_takers` (as DoubleML's `subgroups`).
#' @inherit dml_plr return
#' @export
dml_iivm <- function(y, d, z, x, ml_g = lrn_lasso_cv(), ml_m = lrn_logistic_cv(), ml_r = ml_m,
                     subgroups = c(always_takers = TRUE, never_takers = TRUE),
                     n_folds = 5, n_rep = 1, seed = 0) {
  data <- prepare(y, d, x)
  check_binary(data$d, "d")
  z <- as_numeric_vector(z, "z", length(data$y))
  check_binary(z, "z")
  ptr <- rs_fit_iivm(data$y, data$d, data$x, z, data$names,
                     check_learner(ml_g, "regressor", "ml_g"),
                     check_learner(ml_m, "classifier", "ml_m"),
                     check_learner(ml_r, "classifier", "ml_r"),
                     isTRUE(subgroups[["always_takers"]]), isTRUE(subgroups[["never_takers"]]),
                     check_count(n_folds, "n_folds", 2), check_count(n_rep, "n_rep"), seed)
  new_fit(ptr, match.call())
}

#' Difference in differences
#'
#' The average treatment effect on the treated with two periods (Sant'Anna
#' and Zhao, 2020), as `DoubleMLDID` and `DoubleMLDIDCS`. With panel data,
#' `y` is the change of the outcome between the periods; with repeated
#' cross-sections, `y` is the outcome and `t` the period (0/1).
#'
#' @inheritParams dml_plr
#' @param y Outcome change (panel) or outcome (repeated cross-sections).
#' @param d Binary treatment group indicator (0/1).
#' @param t `NULL` for panel data, or the period indicator (0/1) of repeated
#'   cross-sections.
#' @param ml_g Learner for the outcome regressions.
#' @param ml_m Classifier for the propensity score (not used by the
#'   `"experimental"` score).
#' @param score `"observational"` or `"experimental"`.
#' @param in_sample_normalization Normalise the weights in sample, as DoubleML.
#' @inherit dml_plr return
#' @export
dml_did <- function(y, d, x, t = NULL, ml_g = lrn_lasso_cv(), ml_m = lrn_logistic_cv(),
                    score = c("observational", "experimental"), in_sample_normalization = TRUE,
                    n_folds = 5, n_rep = 1, seed = 0) {
  score <- match.arg(score)
  data <- prepare(y, d, x)
  check_binary(data$d, "d")
  if (!is.null(t)) {
    t <- as_numeric_vector(t, "t", length(data$y))
    check_binary(t, "t")
  }
  ml_m <- if (score == "experimental") NULL else check_learner(ml_m, "classifier", "ml_m")
  ptr <- rs_fit_did(data$y, data$d, data$x, t, data$names, check_learner(ml_g, "regressor", "ml_g"),
                    ml_m, score, isTRUE(in_sample_normalization),
                    check_count(n_folds, "n_folds", 2), check_count(n_rep, "n_rep"), seed)
  new_fit(ptr, match.call())
}

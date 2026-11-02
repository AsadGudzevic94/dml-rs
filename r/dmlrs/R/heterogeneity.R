# Heterogeneous effects: BLP / GATE, CATE meta-learners, causal and
# instrumental forests, and conformal intervals for individual effects.

# The Rust state of an object; restored objects (readRDS) have none.
live_ptr <- function(ptr, kind) {
  if (!rs_is_valid_kind(ptr, kind)) {
    stop("this object was restored from a saved session and has no Rust state; refit the model",
         call. = FALSE)
  }
  ptr
}

basis_matrix <- function(basis, n, prefix) {
  basis <- as_numeric_matrix(basis, "basis", n)
  if (is.null(colnames(basis))) colnames(basis) <- paste0(prefix, seq_len(ncol(basis)))
  basis
}

# Groups as 0/1 dummy columns: a matrix or data frame of dummies, or a vector
# (factor, character, numbers) with one value per group.
group_matrix <- function(groups, n) {
  if (is.null(dim(groups)) && !is.data.frame(groups)) {
    groups <- factor(groups)
    if (length(groups) != n) {
      stop(sprintf("`groups` has %d values but the model has %d observations", length(groups), n),
           call. = FALSE)
    }
    out <- vapply(levels(groups), function(l) as.numeric(groups == l), numeric(n))
    return(matrix(out, n, dimnames = list(NULL, levels(groups))))
  }
  groups <- as_numeric_matrix(groups, "groups", n)
  if (!all(groups %in% c(0, 1))) stop("`groups` must be 0/1 dummies or a factor", call. = FALSE)
  if (is.null(colnames(groups))) colnames(groups) <- paste0("group_", seq_len(ncol(groups)))
  groups
}

#' Best linear predictor of the CATE and group average effects
#'
#' `cate()` regresses an orthogonal signal of the treatment effect on the
#' columns of `basis` (Semenova and Chernozhukov, 2021): the best linear
#' predictor of the conditional average treatment effect
#' \eqn{\tau(x) \approx b(x)^\top \beta}, with heteroscedasticity-robust (HC0)
#' inference. `gate()` estimates the average effect in each of several
#' mutually exclusive groups. These are DoubleML's `cate()` and `gate()`.
#'
#' They work on
#' * [dml_plr()] fits (one treatment) and [dml_irm()] fits with the `"ATE"`
#'   score,
#' * DR-learner fits from [cate_learner()],
#' * [causal_forest()] and [instrumental_forest()] fits (binary treatment or
#'   instrument), on their doubly robust scores.
#'
#' @param object A fitted model (see above).
#' @param basis Basis functions of the covariates, one row per observation of
#'   the fit: a numeric matrix or data frame. Include a column of ones for an
#'   intercept.
#' @param groups Group membership: a vector (factor, character or numbers)
#'   with one value per observation and one group per distinct value, or a
#'   matrix of 0/1 dummy columns.
#' @param ... Not used.
#' @return A `dmlrs_blp` object with [stats::coef()], [confint()] (pointwise or,
#'   with `joint = TRUE`, simultaneous intervals) and [predict()] (effects at
#'   new basis rows).
#' @examples
#' set.seed(5)
#' n <- 1000
#' x <- matrix(rnorm(n * 3), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' y <- d * (1 + x[, 2]) + x[, 1] + rnorm(n)
#' fit <- dml_irm(y, d, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
#' blp <- cate(fit, cbind(intercept = 1, x2 = x[, 2]))
#' blp
#' predict(blp, cbind(1, c(-1, 0, 1)), interval = "pointwise")
#' gate(fit, ifelse(x[, 2] > 0, "high", "low"))
#' @export
cate <- function(object, basis, ...) UseMethod("cate")

#' @rdname cate
#' @export
gate <- function(object, groups, ...) UseMethod("gate")

#' @export
cate.dmlrs_fit <- function(object, basis, ...) {
  basis <- basis_matrix(basis, object$n_obs, "b")
  new_blp(rs_cate(fit_ptr(object), basis), colnames(basis), match.call())
}

#' @export
gate.dmlrs_fit <- function(object, groups, ...) {
  groups <- group_matrix(groups, object$n_obs)
  new_blp(rs_gate(fit_ptr(object), groups), colnames(groups), match.call())
}

new_blp <- function(ptr, names, call) {
  info <- rs_blp_info(ptr)
  structure(list(ptr = ptr, call = call, is_gate = info$is_gate,
                 coefficients = stats::setNames(info$coef, names),
                 se = stats::setNames(info$se, names), t = stats::setNames(info$t, names),
                 p = stats::setNames(info$p, names)),
            class = "dmlrs_blp")
}

#' @export
print.dmlrs_blp <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(if (x$is_gate) "Group average treatment effects\n\n"
      else "Best linear predictor of the CATE\n\n")
  stats::printCoefmat(coef_table(x), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  invisible(x)
}

#' Intervals and predictions of a best linear predictor
#'
#' @param object A `dmlrs_blp` object from [cate()] or [gate()].
#' @param parm Coefficients (names or indices); all by default.
#' @param level Confidence level.
#' @param joint Simultaneous intervals for all coefficients (for
#'   `confint()`) or all rows of `newdata` (for `predict(interval = "joint")`),
#'   from a Gaussian multiplier bootstrap.
#' @param n_boot Number of bootstrap draws.
#' @param seed Seed of the bootstrap.
#' @param newdata Basis rows at which to predict the effect (same columns as
#'   the basis of the fit).
#' @param interval `"none"` (effect and standard error), `"pointwise"` or
#'   `"joint"`.
#' @param ... Not used.
#' @return `confint()`: a matrix with one row per coefficient. `predict()`: a
#'   data frame with columns `estimate` and `se`, or `estimate`, `lower` and
#'   `upper`.
#' @name blp-methods
NULL

#' @rdname blp-methods
#' @export
confint.dmlrs_blp <- function(object, parm, level = 0.95, joint = FALSE, n_boot = 500,
                              seed = 0, ...) {
  ptr <- live_ptr(object$ptr, "blp")
  ci <- if (joint) {
    k <- length(object$coefficients)
    rs_blp_predict(ptr, diag(k), "joint", level, check_count(n_boot, "n_boot"), seed)[, c(1, 3),
                                                                                    drop = FALSE]
  } else {
    rs_blp_confint(ptr, level)
  }
  pct <- paste(format(100 * c((1 - level) / 2, (1 + level) / 2), trim = TRUE, digits = 3), "%")
  dimnames(ci) <- list(names(object$coefficients), pct)
  if (missing(parm)) ci else ci[parm, , drop = FALSE]
}

#' @rdname blp-methods
#' @export
predict.dmlrs_blp <- function(object, newdata, interval = c("none", "pointwise", "joint"),
                              level = 0.95, n_boot = 500, seed = 0, ...) {
  interval <- match.arg(interval)
  newdata <- as_numeric_matrix(newdata, "newdata")
  out <- rs_blp_predict(live_ptr(object$ptr, "blp"), newdata, interval, level,
                        check_count(n_boot, "n_boot"), seed)
  if (interval == "none") {
    data.frame(estimate = out[, 1], se = out[, 2])
  } else {
    data.frame(estimate = out[, 2], lower = out[, 1], upper = out[, 3])
  }
}

# ---------------------------------------------------------------------------
# CATE meta-learners
# ---------------------------------------------------------------------------

#' Meta-learners for conditional average treatment effects
#'
#' Estimate \eqn{\tau(x) = E[Y(1) - Y(0) \mid X = x]} of a binary treatment
#' with a meta-learner:
#' * `"S"`: one outcome model on \eqn{(X, D)}; \eqn{\hat\tau(x) =
#'   \hat\mu(x, 1) - \hat\mu(x, 0)}.
#' * `"T"`: one outcome model per arm.
#' * `"X"`: imputed effects per arm, blended with the propensity score
#'   (Kunzel et al., 2019).
#' * `"R"`: residual-on-residual regression with cross-fitted nuisances (Nie
#'   and Wager, 2021); `ml_effect` must support sample weights (all built-in
#'   regressors except the regression forest do).
#' * `"DR"`: regression of the cross-fitted AIPW pseudo-outcome on \eqn{X}
#'   (Kennedy, 2023). Supports inference on the heterogeneity with [cate()]
#'   and [gate()].
#'
#' @inheritParams dml_irm
#' @param method `"DR"`, `"R"`, `"X"`, `"T"` or `"S"`.
#' @param ml_outcome Learner for the outcome regressions.
#' @param ml_propensity Classifier for the propensity score (`"X"`, `"R"`,
#'   `"DR"`).
#' @param ml_effect Learner for the effect regression (`"X"`, `"R"`, `"DR"`).
#' @param n_folds,seed Cross-fitting folds and their seed (`"R"`, `"DR"`).
#' @param clipping_threshold Propensity scores are clipped to \eqn{[t, 1 - t]}
#'   (`"R"`, `"DR"`).
#' @return A `dmlrs_cate_learner` object: `effect` (in-sample effects),
#'   `pseudo_outcomes` (DR-learner) and a [predict()] method returning the
#'   effects at new covariates.
#' @examples
#' set.seed(6)
#' n <- 1000
#' x <- matrix(rnorm(n * 3), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' y <- d * (1 + x[, 2]) + x[, 1] + rnorm(n)
#' fit <- cate_learner(y, d, x, ml_outcome = lrn_ols(), ml_propensity = lrn_logistic(),
#'                     ml_effect = lrn_ols())
#' fit
#' head(predict(fit, x))
#' cate(fit, cbind(1, x[, 2]))
#' @export
cate_learner <- function(y, d, x, method = c("DR", "R", "X", "T", "S"),
                         ml_outcome = lrn_random_forest(), ml_propensity = lrn_logistic_cv(),
                         ml_effect = lrn_random_forest(), n_folds = 5, seed = 0,
                         clipping_threshold = 0.01) {
  method <- match.arg(method)
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  check_binary(data$d, "d")
  uses <- c(propensity = method %in% c("X", "R", "DR"), effect = method %in% c("X", "R", "DR"))
  ml_propensity <- if (uses[["propensity"]]) {
    check_learner(ml_propensity, "classifier", "ml_propensity")
  }
  ml_effect <- if (uses[["effect"]]) check_learner(ml_effect, "regressor", "ml_effect")
  ptr <- rs_fit_cate_learner(data$y, data$d, data$x, method,
                             check_learner(ml_outcome, "regressor", "ml_outcome"), ml_propensity,
                             ml_effect, check_count(n_folds, "n_folds", 2), seed,
                             clipping_threshold)
  structure(list(ptr = ptr, call = match.call(), method = method, n_obs = length(data$y),
                 n_features = ncol(data$x), effect = rs_cate_effect(ptr, data$x),
                 pseudo_outcomes = rs_cate_pseudo_outcomes(ptr)),
            class = "dmlrs_cate_learner")
}

#' @export
print.dmlrs_cate_learner <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("%s-learner for the conditional average treatment effect\n", x$method))
  cat(sprintf("%d observations; in-sample effects: mean %s, sd %s\n", x$n_obs,
              format(mean(x$effect), digits = digits), format(stats::sd(x$effect), digits = digits)))
  if (x$method == "DR") cat("Inference on the heterogeneity: cate(), gate()\n")
  invisible(x)
}

#' @export
predict.dmlrs_cate_learner <- function(object, newdata, ...) {
  rs_cate_effect(live_ptr(object$ptr, "cate"), as_numeric_matrix(newdata, "newdata"))
}

#' @export
cate.dmlrs_cate_learner <- function(object, basis, ...) {
  basis <- basis_matrix(basis, object$n_obs, "b")
  new_blp(rs_cate_learner_blp(live_ptr(object$ptr, "cate"), basis), colnames(basis), match.call())
}

#' @export
gate.dmlrs_cate_learner <- function(object, groups, ...) {
  groups <- group_matrix(groups, object$n_obs)
  new_blp(rs_cate_learner_gate(live_ptr(object$ptr, "cate"), groups), colnames(groups),
          match.call())
}

# ---------------------------------------------------------------------------
# Causal and instrumental forests
# ---------------------------------------------------------------------------

forest_spec <- function(num_trees, sample_fraction, mtry, min_node_size, honesty,
                        honesty_fraction, honesty_prune_leaves, alpha, imbalance_penalty,
                        stabilize_splits, ci_group_size, seed) {
  list(type = "forest", num_trees = check_count(num_trees, "num_trees"),
       sample_fraction = sample_fraction,
       mtry = if (is.null(mtry)) NULL else check_count(mtry, "mtry"),
       min_node_size = check_count(min_node_size, "min_node_size"), honesty = isTRUE(honesty),
       honesty_fraction = honesty_fraction, honesty_prune_leaves = isTRUE(honesty_prune_leaves),
       alpha = alpha, imbalance_penalty = imbalance_penalty,
       stabilize_splits = isTRUE(stabilize_splits),
       ci_group_size = check_count(ci_group_size, "ci_group_size"), seed = seed)
}

# A learner for a treatment or instrument: a classifier when the variable is
# binary and the learner can classify, a regressor otherwise.
nuisance_learner <- function(learner, v, arg) {
  classify <- all(v %in% c(0, 1)) && inherits(learner, "dmlrs_learner") &&
    learner$type %in% c("logistic", "logistic_cv", "random_forest", "gradient_boosting",
                        "hist_gradient_boosting")
  list(spec = check_learner(learner, if (classify) "classifier" else "regressor", arg),
       classifier = classify)
}

#' Causal forests
#'
#' Generalized random forests (Athey, Tibshirani and Wager, 2019) for
#' heterogeneous effects, ports of `grf::causal_forest()` and
#' `grf::instrumental_forest()`. The nuisance functions \eqn{E[Y \mid X]},
#' \eqn{E[D \mid X]} (and \eqn{E[Z \mid X]}) are cross-fitted with any
#' learners; honest trees grown on the residuals estimate the conditional
#' average treatment effect \eqn{\tau(x)} (or, with an instrument, the
#' conditional local average treatment effect).
#'
#' The estimates agree with grf in distribution, not draw by draw: the random
#' number generators differ.
#'
#' @inheritParams dml_plr
#' @param d Treatment: binary or continuous.
#' @param z Instrument: binary or continuous.
#' @param ml_y Learner for \eqn{E[Y \mid X]}.
#' @param ml_d,ml_z Learners for \eqn{E[D \mid X]} and \eqn{E[Z \mid X]}: used as
#'   classifiers when the variable is binary and the learner can classify (e.g.
#'   [lrn_random_forest()]), as regressors otherwise.
#' @param num_trees Number of trees.
#' @param sample_fraction Share of the data in each tree's subsample (at most
#'   0.5 with `ci_group_size > 1`).
#' @param mtry Variables tried per split (default
#'   \eqn{\min(\lceil\sqrt{p} + 20\rceil, p)}).
#' @param min_node_size Nodes with at most this many observations are not split.
#' @param honesty,honesty_fraction,honesty_prune_leaves Honest trees, the
#'   share of the subsample used for splitting, and pruning of leaves without
#'   estimation observations.
#' @param alpha Each child of a split keeps at least this share of the parent's
#'   observations and treatment variation.
#' @param imbalance_penalty Penalty on splits with little treatment variation
#'   in a child.
#' @param stabilize_splits Apply `alpha` to the treatment variation.
#' @param ci_group_size Trees per half-sample (2 gives variance estimates, 1
#'   disables them).
#' @param reduced_form_weight grf's `reduced.form.weight`: splits target
#'   \eqn{(1 - r) Z + r D}.
#' @param clipping_threshold Propensities are clipped to \eqn{[t, 1 - t]} in the
#'   doubly robust scores.
#' @param n_folds Folds of the nuisance models.
#' @param seed Seed of the folds and the forest.
#' @return A `dmlrs_causal_forest` object: `predictions` and `variance`
#'   (out-of-bag estimates for the training rows), `y_hat`, `d_hat` (and
#'   `z_hat`), `variable_importance`. Methods: [predict()] at new covariates,
#'   [average_treatment_effect()], [test_calibration()], [cate()], [gate()].
#' @examples
#' set.seed(8)
#' n <- 1000
#' x <- matrix(rnorm(n * 4), n)
#' d <- rbinom(n, 1, plogis(x[, 1]))
#' y <- d * (1 + x[, 2]) + x[, 1] + rnorm(n)
#' cf <- causal_forest(y, d, x, ml_y = lrn_ols(), ml_d = lrn_logistic(), num_trees = 500)
#' cf
#' average_treatment_effect(cf)
#' head(predict(cf, x))
#' test_calibration(cf)
#' @export
causal_forest <- function(y, d, x, ml_y = lrn_regression_forest(), ml_d = lrn_regression_forest(),
                          num_trees = 2000, sample_fraction = 0.5, mtry = NULL,
                          min_node_size = 5, honesty = TRUE, honesty_fraction = 0.5,
                          honesty_prune_leaves = TRUE, alpha = 0.05, imbalance_penalty = 0,
                          stabilize_splits = TRUE, ci_group_size = 2,
                          clipping_threshold = 0.01, n_folds = 5, seed = 0) {
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  ml_d <- nuisance_learner(ml_d, data$d, "ml_d")
  spec <- forest_spec(num_trees, sample_fraction, mtry, min_node_size, honesty, honesty_fraction,
                      honesty_prune_leaves, alpha, imbalance_penalty, stabilize_splits,
                      ci_group_size, seed)
  ptr <- rs_fit_causal_forest(data$y, data$d, data$x, check_learner(ml_y, "regressor", "ml_y"),
                              ml_d$spec, ml_d$classifier, spec, clipping_threshold,
                              check_count(n_folds, "n_folds", 2), seed)
  new_forest(ptr, "causal", data, num_trees, match.call())
}

#' @rdname causal_forest
#' @export
instrumental_forest <- function(y, d, z, x, ml_y = lrn_regression_forest(),
                                ml_d = lrn_regression_forest(), ml_z = lrn_regression_forest(),
                                num_trees = 2000, sample_fraction = 0.5, mtry = NULL,
                                min_node_size = 5, honesty = TRUE, honesty_fraction = 0.5,
                                honesty_prune_leaves = TRUE, alpha = 0.05, imbalance_penalty = 0,
                                stabilize_splits = TRUE, ci_group_size = 2,
                                reduced_form_weight = 0, clipping_threshold = 0.01, n_folds = 5,
                                seed = 0) {
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  z <- as_numeric_vector(z, "z", length(data$y))
  ml_d <- nuisance_learner(ml_d, data$d, "ml_d")
  ml_z <- nuisance_learner(ml_z, z, "ml_z")
  spec <- forest_spec(num_trees, sample_fraction, mtry, min_node_size, honesty, honesty_fraction,
                      honesty_prune_leaves, alpha, imbalance_penalty, stabilize_splits,
                      ci_group_size, seed)
  ptr <- rs_fit_instrumental_forest(data$y, data$d, data$x, z,
                                    check_learner(ml_y, "regressor", "ml_y"), ml_d$spec,
                                    ml_z$spec, ml_d$classifier, ml_z$classifier, spec,
                                    reduced_form_weight, clipping_threshold,
                                    check_count(n_folds, "n_folds", 2), seed)
  new_forest(ptr, "instrumental", data, num_trees, match.call())
}

new_forest <- function(ptr, kind, data, num_trees, call) {
  info <- rs_forest_info(ptr)
  names <- colnames(data$x)
  if (is.null(names)) names <- paste0("x", seq_len(ncol(data$x)))
  structure(list(ptr = ptr, call = call, kind = kind, n_obs = length(data$y),
                 num_trees = num_trees, predictions = info$effect, variance = info$variance,
                 y_hat = info$y_hat, d_hat = info$d_hat, z_hat = info$z_hat,
                 variable_importance = stats::setNames(info$importance, names),
                 y = data$y, d = as.vector(data$d)),
            class = "dmlrs_causal_forest")
}

#' @export
print.dmlrs_causal_forest <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("%s forest: %d trees, %d observations\n",
              if (x$kind == "causal") "Causal" else "Instrumental", x$num_trees, x$n_obs))
  cat(sprintf("Out-of-bag effects: mean %s, sd %s\n", format(mean(x$predictions), digits = digits),
              format(stats::sd(x$predictions), digits = digits)))
  imp <- sort(x$variable_importance, decreasing = TRUE)
  cat("Variable importance:\n")
  print(round(imp[seq_len(min(10, length(imp)))], 3))
  invisible(x)
}

#' Predictions of a causal forest
#'
#' @param object A [causal_forest()] or [instrumental_forest()] fit.
#' @param newdata Covariates at which to estimate the effect, or `NULL` for the
#'   out-of-bag estimates of the training rows.
#' @param ... Not used.
#' @return A data frame with columns `estimate` and `se` (from the little-bag
#'   variance estimates; `NA` with `ci_group_size = 1`).
#' @export
predict.dmlrs_causal_forest <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    est <- object$predictions
    var <- object$variance
  } else {
    out <- rs_forest_effect(live_ptr(object$ptr, "forest"), as_numeric_matrix(newdata, "newdata"))
    est <- out[, 1]
    var <- out[, 2]
  }
  data.frame(estimate = est, se = sqrt(ifelse(var >= 0, var, NA)))
}

#' Average treatment effect of a causal forest
#'
#' The average of the doubly robust scores, with the usual DML standard error,
#' as `grf::average_treatment_effect()` (binary treatment: the average
#' treatment effect; binary instrument: the local average treatment effect).
#'
#' @param object A [causal_forest()] or [instrumental_forest()] fit.
#' @return A `dmlrs_fit` object.
#' @export
average_treatment_effect <- function(object) {
  if (!inherits(object, "dmlrs_causal_forest")) {
    stop("`object` must be a causal_forest() or instrumental_forest() fit", call. = FALSE)
  }
  new_fit(rs_forest_ate(live_ptr(object$ptr, "forest")), match.call())
}

#' Calibration test of a causal forest
#'
#' `grf::test_calibration()`: regress \eqn{Y - \hat y} on
#' \eqn{(D - \hat d)\bar\tau} and \eqn{(D - \hat d)(\hat\tau(X) - \bar\tau)}
#' (out-of-bag \eqn{\hat\tau}, mean \eqn{\bar\tau}) with HC3 standard errors.
#' A `mean.forest.prediction` coefficient near 1 says the average prediction
#' is right; a `differential.forest.prediction` coefficient significantly
#' above 0 says the forest captures real heterogeneity. p-values are
#' one-sided and use the normal distribution.
#'
#' @param object A [causal_forest()] fit.
#' @return A coefficient matrix.
#' @export
test_calibration <- function(object) {
  if (!inherits(object, "dmlrs_causal_forest") || object$kind != "causal") {
    stop("`object` must be a causal_forest() fit", call. = FALSE)
  }
  tau <- object$predictions
  dres <- object$d - object$d_hat
  z <- cbind(dres * mean(tau), dres * (tau - mean(tau)))
  target <- object$y - object$y_hat
  bread <- solve(crossprod(z))
  coef <- drop(bread %*% crossprod(z, target))
  resid <- drop(target - z %*% coef)
  leverage <- rowSums((z %*% bread) * z)
  meat <- crossprod(z * (resid / (1 - leverage)))
  se <- sqrt(diag(bread %*% meat %*% bread))
  t <- coef / se
  table <- cbind(coef, se, t, stats::pnorm(t, lower.tail = FALSE))
  dimnames(table) <- list(c("mean.forest.prediction", "differential.forest.prediction"),
                          c("Estimate", "Std. Error", "t value", "Pr(>t)"))
  structure(table, class = c("dmlrs_calibration", "matrix"))
}

#' @export
print.dmlrs_calibration <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Best linear fit using forest predictions (on held-out data)\n")
  cat("as well as the mean forest prediction as regressors, along\n")
  cat("with one-sided heteroskedasticity-robust (HC3) SEs:\n\n")
  stats::printCoefmat(unclass(x), digits = digits, has.Pvalue = TRUE, P.values = TRUE)
  invisible(x)
}

#' @export
cate.dmlrs_causal_forest <- function(object, basis, ...) {
  basis <- basis_matrix(basis, object$n_obs, "b")
  new_blp(rs_forest_blp(live_ptr(object$ptr, "forest"), basis), colnames(basis), match.call())
}

#' @export
gate.dmlrs_causal_forest <- function(object, groups, ...) {
  groups <- group_matrix(groups, object$n_obs)
  new_blp(rs_forest_gate(live_ptr(object$ptr, "forest"), groups), colnames(groups), match.call())
}

# ---------------------------------------------------------------------------
# Conformal intervals
# ---------------------------------------------------------------------------

#' Conformal prediction intervals for individual treatment effects
#'
#' Intervals for the potential outcomes and the individual effect
#' \eqn{Y(1) - Y(0)} of new units with finite-sample marginal coverage under
#' unconfoundedness: weighted split conformal prediction (Lei and Candes,
#' 2021). The sample is split once into a training part, on which the score
#' learners are fitted per arm and `ml_m` learns the propensity score, and a
#' calibration part, whose conformity scores are reweighted to the target
#' population.
#'
#' Pass `ml_y` for the residual score \eqn{|y - \mu(x)|} (intervals of constant
#' width), or `ml_lower` and `ml_upper` (e.g. [lrn_quantile_forest()] at 0.05
#' and 0.95) for conformalised quantile regression (intervals that adapt to
#' the noise level). Pass `ml_m` or, in a randomised experiment, the known
#' `propensity`.
#'
#' @inheritParams dml_irm
#' @param ml_y Learner for \eqn{E[Y \mid X]} in each arm (residual score).
#' @param ml_lower,ml_upper Learners for a lower and an upper conditional
#'   quantile (quantile score).
#' @param ml_m Classifier for the propensity score.
#' @param propensity Known treatment probability (instead of `ml_m`).
#' @param train_fraction Share of the sample used to fit the learners.
#' @param clipping_threshold Estimated propensities are clipped to
#'   \eqn{[t, 1 - t]}.
#' @param seed Seed of the split.
#' @return A `dmlrs_conformal` object with a [predict()] method.
#' @examples
#' set.seed(9)
#' n <- 2000
#' x <- matrix(rnorm(n * 2), n)
#' d <- rbinom(n, 1, 0.5)
#' y <- d * (1 + x[, 1]) + x[, 2] + rnorm(n)
#' fit <- conformal_ite(y, d, x, ml_y = lrn_ols(), propensity = 0.5)
#' predict(fit, x[1:3, ], alpha = 0.1)
#' @export
conformal_ite <- function(y, d, x, ml_y = NULL, ml_lower = NULL, ml_upper = NULL, ml_m = NULL,
                          propensity = NULL, train_fraction = 0.75, clipping_threshold = 0.01,
                          seed = 0) {
  data <- prepare(y, d, x)
  if (ncol(data$d) != 1) stop("`d` must be a single treatment", call. = FALSE)
  check_binary(data$d, "d")
  if (is.null(ml_y) == (is.null(ml_lower) || is.null(ml_upper))) {
    stop("pass either `ml_y` (residual score) or both `ml_lower` and `ml_upper` (quantile score)",
         call. = FALSE)
  }
  if (is.null(ml_m) == is.null(propensity)) {
    stop("pass either `ml_m` or a known `propensity`", call. = FALSE)
  }
  learner <- function(l, arg) if (is.null(l)) NULL else check_learner(l, "quantile", arg)
  ptr <- rs_fit_conformal(data$y, data$d, data$x, learner(ml_y, "ml_y"),
                          learner(ml_lower, "ml_lower"), learner(ml_upper, "ml_upper"),
                          if (is.null(ml_m)) NULL else check_learner(ml_m, "classifier", "ml_m"),
                          if (is.null(propensity)) NULL else as.numeric(propensity),
                          train_fraction, clipping_threshold, seed)
  n_cal <- rs_conformal_n_calibration(ptr)
  structure(list(ptr = ptr, call = match.call(), n_obs = length(data$y),
                 score = if (is.null(ml_y)) "quantile" else "residual",
                 n_calibration = c(control = n_cal[1], treated = n_cal[2])),
            class = "dmlrs_conformal")
}

#' @export
print.dmlrs_conformal <- function(x, ...) {
  cat(sprintf("Conformal intervals for individual effects (%s score)\n", x$score))
  cat(sprintf("%d observations; calibration units: %d control, %d treated\n", x$n_obs,
              x$n_calibration[["control"]], x$n_calibration[["treated"]]))
  invisible(x)
}

#' Conformal intervals at new units
#'
#' @param object A [conformal_ite()] fit.
#' @param newdata Covariates of the new units.
#' @param type `"ite"` for the individual effect \eqn{Y(1) - Y(0)} (the two
#'   potential-outcome intervals at level \eqn{\alpha/2} each, combined), or
#'   `"y1"` / `"y0"` for one potential outcome.
#' @param alpha Miscoverage level: intervals cover with probability at least
#'   \eqn{1 - \alpha}.
#' @param target For `"y1"` / `"y0"`: the population the new units come from,
#'   `"all"`, `"treated"` or `"control"` (e.g. \eqn{Y(0)} of treated units).
#' @param ... Not used.
#' @return A matrix with columns `lower` and `upper`.
#' @export
predict.dmlrs_conformal <- function(object, newdata, type = c("ite", "y1", "y0"), alpha = 0.1,
                                    target = c("all", "treated", "control"), ...) {
  type <- match.arg(type)
  target <- match.arg(target)
  out <- rs_conformal_predict(live_ptr(object$ptr, "conformal"),
                              as_numeric_matrix(newdata, "newdata"), type, target, alpha)
  colnames(out) <- c("lower", "upper")
  out
}

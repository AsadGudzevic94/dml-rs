# Nuisance learners. Each constructor returns a specification; the model
# builds the Rust learner from it, so cross-fitting never calls back into R.

new_learner <- function(type, ...) {
  structure(list(type = type, ...), class = "dmlrs_learner")
}

# The value of a renamed argument: `old` (passed under the former name, with a
# deprecation warning) or else `new`.
renamed_arg <- function(new, old, new_name, old_name) {
  if (is.null(old)) return(new)
  warning(sprintf("`%s` is deprecated; use `%s`", old_name, new_name), call. = FALSE)
  old
}

#' Built-in nuisance learners
#'
#' Learners for the nuisance functions of the DML models, implemented in Rust.
#' Parameters follow scikit-learn, and results match the learners of the
#' Python package dml-rs. Regressors learn conditional means; classifiers
#' (`lrn_logistic()`, `lrn_logistic_cv()`, and the tree ensembles) learn
#' probabilities of a binary variable.
#'
#' * `lrn_ols()`: least squares with intercept, optionally ridge-penalised
#'   (`ridge` is scikit-learn's `Ridge(alpha)`).
#' * `lrn_lasso()`: lasso with penalty `alpha` (scikit-learn's `Lasso`).
#' * `lrn_lasso_cv()`: lasso with the penalty chosen by `cv`-fold
#'   cross-validation over `n_alphas` values (scikit-learn's `LassoCV`).
#' * `lrn_logistic()`: L2-penalised logistic regression with inverse penalty `C`.
#' * `lrn_logistic_cv()`: logistic regression with `C` chosen by stratified
#'   cross-validation (log loss) over `n_cs` values from 1e-4 to 1e4.
#' * `lrn_random_forest()`: random forest (regressor or classifier).
#' * `lrn_gradient_boosting()`: gradient boosting with the trees of
#'   scikit-learn's `GradientBoosting*`.
#' * `lrn_hist_gradient_boosting()`: histogram gradient boosting, identical to
#'   scikit-learn's `HistGradientBoosting*`; the learner for large data.
#' * `lrn_regression_forest()`: honest regression forest (R's
#'   `grf::regression_forest`), regression only.
#' * `lrn_quantile_forest()`: quantile forest (R's `grf::quantile_forest`)
#'   predicting the conditional `quantile` of the outcome; the lower and
#'   upper learners of [conformal_ite()].
#'
#' @param ridge,alpha,C Penalties (see above).
#' @param tol,max_iter Convergence tolerance and maximum iterations (sweeps of
#'   coordinate descent, or boosting iterations for `lrn_hist_gradient_boosting()`).
#' @param n_alphas,eps,n_cs,cv Grid size, smallest-to-largest penalty ratio and
#'   number of folds of the cross-validated learners.
#' @param n_estimators Number of trees.
#' @param max_depth Maximum tree depth (`NULL`: unlimited).
#' @param min_samples_leaf Minimum observations per leaf.
#' @param max_features Features tried per split: `NULL` (all for regression,
#'   `"sqrt"` for classification), `"sqrt"`, `"log2"`, a fraction or a number.
#' @param max_bins `NULL` (exact splits) or 2 to 255: search splits on that many
#'   quantile bins per feature, much faster on large data with practically the
#'   same accuracy.
#' @param learning_rate Shrinkage of each tree.
#' @param subsample Share of observations used for each tree.
#' @param max_leaf_nodes Maximum leaves per tree (`NULL`: unlimited).
#' @param l2_regularization L2 penalty on leaf values.
#' @param min_node_size Minimum observations per leaf (grf's `min.node.size`).
#' @param quantile Quantile predicted by `lrn_quantile_forest()`.
#' @param seed Seed of the random number generator.
#' @param random_state Deprecated name of `seed`.
#' @return A learner specification (class `dmlrs_learner`).
#' @examples
#' lrn_lasso_cv()
#' lrn_random_forest(n_estimators = 200, min_samples_leaf = 5)
#' @name learners
NULL

#' @rdname learners
#' @export
lrn_ols <- function(ridge = 0) new_learner("ols", ridge = ridge)

#' @rdname learners
#' @export
lrn_lasso <- function(alpha = 1, tol = 1e-4, max_iter = 100000) {
  new_learner("lasso", alpha = alpha, tol = tol, max_iter = max_iter)
}

#' @rdname learners
#' @export
lrn_lasso_cv <- function(n_alphas = 100, eps = 1e-3, cv = 5) {
  new_learner("lasso_cv", n_alphas = n_alphas, eps = eps, cv = cv)
}

#' @rdname learners
#' @export
lrn_logistic <- function(C = 1) new_learner("logistic", C = C)

#' @rdname learners
#' @export
lrn_logistic_cv <- function(n_cs = 10, cv = 5) new_learner("logistic_cv", n_cs = n_cs, cv = cv)

#' @rdname learners
#' @export
lrn_random_forest <- function(n_estimators = 100, max_depth = NULL, min_samples_leaf = 1,
                              max_features = NULL, seed = 0, random_state = NULL, max_bins = NULL) {
  new_learner("random_forest", n_estimators = n_estimators, max_depth = max_depth,
              min_samples_leaf = min_samples_leaf, max_features = max_features,
              max_bins = max_bins, seed = renamed_arg(seed, random_state, "seed", "random_state"))
}

#' @rdname learners
#' @export
lrn_gradient_boosting <- function(n_estimators = 100, learning_rate = 0.1, max_depth = 3,
                                  min_samples_leaf = 1, subsample = 1, seed = 0, random_state = NULL) {
  new_learner("gradient_boosting", n_estimators = n_estimators, learning_rate = learning_rate,
              max_depth = max_depth, min_samples_leaf = min_samples_leaf, subsample = subsample,
              seed = renamed_arg(seed, random_state, "seed", "random_state"))
}

#' @rdname learners
#' @export
lrn_hist_gradient_boosting <- function(max_iter = 100, learning_rate = 0.1, max_leaf_nodes = 31,
                                       min_samples_leaf = 20, l2_regularization = 0,
                                       seed = 0, random_state = NULL) {
  new_learner("hist_gradient_boosting", max_iter = max_iter, learning_rate = learning_rate,
              max_leaf_nodes = max_leaf_nodes, min_samples_leaf = min_samples_leaf,
              l2_regularization = l2_regularization, seed = renamed_arg(seed, random_state, "seed", "random_state"))
}

#' @rdname learners
#' @export
lrn_regression_forest <- function(n_estimators = 2000, min_node_size = 5, seed = 0, random_state = NULL) {
  new_learner("regression_forest", n_estimators = n_estimators, min_node_size = min_node_size,
              seed = renamed_arg(seed, random_state, "seed", "random_state"))
}

#' @rdname learners
#' @export
lrn_quantile_forest <- function(quantile, n_estimators = 2000, min_node_size = 5, seed = 0) {
  if (!is.numeric(quantile) || length(quantile) != 1 || quantile <= 0 || quantile >= 1) {
    stop("`quantile` must be a number in (0, 1)", call. = FALSE)
  }
  new_learner("quantile_forest", quantile = quantile, n_estimators = n_estimators,
              min_node_size = min_node_size, seed = seed)
}

#' @export
print.dmlrs_learner <- function(x, ...) {
  params <- x[names(x) != "type"]
  shown <- vapply(params, function(v) if (is.null(v)) "NULL" else format(v), character(1))
  cat(sprintf("<dmlrs learner: %s(%s)>\n", x$type,
              paste(names(shown), shown, sep = " = ", collapse = ", ")))
  invisible(x)
}

check_learner <- function(learner, role, arg) {
  if (!inherits(learner, "dmlrs_learner")) {
    stop(sprintf("`%s` must be a dmlrs learner such as lrn_lasso_cv()", arg), call. = FALSE)
  }
  classifiers <- c("logistic", "logistic_cv", "random_forest", "gradient_boosting",
                   "hist_gradient_boosting")
  regressors <- c("ols", "lasso", "lasso_cv", "random_forest", "gradient_boosting",
                  "hist_gradient_boosting", "regression_forest")
  allowed <- switch(role, classifier = classifiers, regressor = regressors,
                    quantile = c(regressors, "quantile_forest"))
  if (role == "quantile") role <- "regressor"
  if (!learner$type %in% allowed) {
    stop(sprintf("`%s` needs a %s; %s is not one", arg, role, learner$type), call. = FALSE)
  }
  unclass(learner)
}

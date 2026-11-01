# Policy learning and evaluation, and Lee bounds.

#' Doubly robust scores of a discrete treatment
#'
#' The doubly robust score \eqn{\Gamma_i(a)} of every unit under every
#' treatment level of a [dml_apos()] fit (first repetition of the sample
#' splits): the input of [policy_tree()] and [policy_value()]. Column means
#' are the average potential outcomes.
#'
#' @param object A [dml_apos()] fit.
#' @return A matrix with one row per unit and one column per level.
#' @export
double_robust_scores <- function(object) {
  if (!inherits(object, "dmlrs_fit")) stop("`object` must be a dml_apos() fit", call. = FALSE)
  scores <- rs_dr_scores(fit_ptr(object))
  colnames(scores) <- names(object$coefficients)
  scores
}

#' Optimal policy trees
#'
#' The decision tree of depth `depth` that maximises the summed rewards
#' `gamma[i, action]`, found by exhaustive search (Athey and Wager, 2021): a
#' port of `policytree::policy_tree()` that finds the same trees. Splits are
#' `x[, j] <= value` (left). The search is exact, so its cost grows like
#' \eqn{(p n)^{depth}}: depth 2 suits thousands of units and tens of
#' features. `split_step > 1` considers only every `split_step`-th split.
#'
#' Learn the tree on some units and estimate its value with [policy_value()]
#' on others.
#'
#' @param x Features: a numeric matrix or data frame.
#' @param gamma Rewards: one row per unit and one column per action, e.g.
#'   from [double_robust_scores()].
#' @param depth Depth of the tree.
#' @param split_step Consider every `split_step`-th split point.
#' @param min_node_size Smallest number of units in a leaf.
#' @return A `dmlrs_policy_tree` object: `nodes` (a data frame; actions,
#'   features and child positions are 1-based) and a [predict()] method that
#'   returns the action (column of `gamma`) of each row.
#' @examples
#' set.seed(9)
#' n <- 500
#' x <- matrix(rnorm(n * 2), n, dimnames = list(NULL, c("age", "income")))
#' gamma <- cbind(control = 0, treat = x[, 2] + rnorm(n))
#' tree <- policy_tree(x, gamma, depth = 1)
#' tree
#' table(predict(tree, x))
#' @export
policy_tree <- function(x, gamma, depth = 2, split_step = 1, min_node_size = 1) {
  x <- as_numeric_matrix(x, "x")
  gamma <- as_numeric_matrix(gamma, "gamma", nrow(x))
  res <- rs_policy_tree(x, gamma, check_count(depth, "depth", 0),
                        check_count(split_step, "split_step"),
                        check_count(min_node_size, "min_node_size"))
  leaf <- res$is_leaf
  nodes <- data.frame(
    is_leaf = leaf,
    action = ifelse(leaf, res$action + 1, NA),
    split_variable = ifelse(leaf, NA, res$split_variable + 1),
    split_value = ifelse(leaf, NA, res$split_value),
    left = ifelse(leaf, NA, res$left + 1),
    right = ifelse(leaf, NA, res$right + 1),
    reward = res$reward
  )
  features <- if (is.null(colnames(x))) paste0("x", seq_len(ncol(x))) else colnames(x)
  actions <- if (is.null(colnames(gamma))) paste0("action ", seq_len(ncol(gamma))) else colnames(gamma)
  structure(list(nodes = nodes, features = features, actions = actions, depth = depth,
                 call = match.call()), class = "dmlrs_policy_tree")
}

#' @export
predict.dmlrs_policy_tree <- function(object, newdata, ...) {
  x <- as_numeric_matrix(newdata, "newdata")
  if (ncol(x) != length(object$features)) {
    stop(sprintf("`newdata` needs %d columns", length(object$features)), call. = FALSE)
  }
  nodes <- object$nodes
  at <- rep(1L, nrow(x))
  repeat {
    inner <- which(!nodes$is_leaf[at])
    if (!length(inner)) break
    node <- at[inner]
    left <- x[cbind(inner, nodes$split_variable[node])] <= nodes$split_value[node]
    at[inner] <- ifelse(left, nodes$left[node], nodes$right[node])
  }
  as.integer(nodes$action[at])
}

#' @export
print.dmlrs_policy_tree <- function(x, ...) {
  cat(sprintf("Optimal policy tree of depth %d\n", x$depth))
  walk <- function(i, indent) {
    node <- x$nodes[i, ]
    pad <- strrep("  ", indent)
    if (node$is_leaf) {
      cat(sprintf("%s* action %d: %s\n", pad, node$action, x$actions[node$action]))
    } else {
      feature <- x$features[node$split_variable]
      cat(sprintf("%s%s <= %g\n", pad, feature, node$split_value))
      walk(node$left, indent + 1)
      cat(sprintf("%s%s > %g\n", pad, feature, node$split_value))
      walk(node$right, indent + 1)
    }
  }
  walk(1, 1)
  invisible(x)
}

#' Doubly robust values of treatment rules
#'
#' The value \eqn{E[\Gamma(\pi(X))]} of each treatment rule, with standard
#' errors. All rules are evaluated on the same units, so their differences
#' ([linear_combination()]) have correct standard errors.
#'
#' @param gamma Rewards: one row per unit and one column per action, e.g.
#'   from [double_robust_scores()] on units not used to learn the rules.
#' @param ... Named rules: integer vectors with the action (column of
#'   `gamma`, 1-based) of every unit, e.g. `tree = predict(tree, x)`.
#' @return A `dmlrs_fit` with one estimate per rule.
#' @examples
#' set.seed(10)
#' n <- 500
#' gamma <- cbind(control = rnorm(n), treat = 0.2 + rnorm(n))
#' values <- policy_value(gamma, treat_all = rep(2, n), treat_none = rep(1, n))
#' values
#' linear_combination(values, c(1, -1))
#' @export
policy_value <- function(gamma, ...) {
  policies <- list(...)
  if (!length(policies) || is.null(names(policies)) || any(names(policies) == "")) {
    stop("pass the rules as named arguments, e.g. tree = predict(tree, x)", call. = FALSE)
  }
  gamma <- as_numeric_matrix(gamma, "gamma")
  actions <- vapply(policies, function(a) {
    if (length(a) != nrow(gamma)) stop("each rule needs one action per row of `gamma`", call. = FALSE)
    if (any(a < 1 | a > ncol(gamma) | a != round(a))) {
      stop(sprintf("actions must be column numbers of `gamma` (1 to %d)", ncol(gamma)), call. = FALSE)
    }
    as.numeric(a) - 1
  }, numeric(nrow(gamma)))
  actions <- matrix(actions, nrow(gamma))
  new_fit(rs_policy_value(gamma, actions, names(policies)), match.call())
}

#' Linear combinations of estimates
#'
#' Estimates of `weights %*% theta` with standard errors from the scores of
#' the fit, e.g. contrasts between treatment levels of [dml_apos()] or
#' between rules of [policy_value()].
#'
#' @param object A `dmlrs_fit`.
#' @param weights A matrix with one row per combination and one column per
#'   estimate, or a vector for a single combination. Row names name the
#'   combinations.
#' @return A `dmlrs_fit` with one estimate per combination.
#' @export
linear_combination <- function(object, weights) {
  if (!inherits(object, "dmlrs_fit")) stop("`object` must be a fitted dmlrs model", call. = FALSE)
  if (is.null(dim(weights))) weights <- matrix(weights, 1)
  storage.mode(weights) <- "double"
  names <- rownames(weights)
  if (is.null(names)) names <- paste("combination", seq_len(nrow(weights)))
  new_fit(rs_linear_combination(fit_ptr(object), weights, names), match.call())
}

#' Lee bounds under sample selection
#'
#' Sharp bounds on the effect of a randomised binary treatment for the units
#' whose outcome is observed whichever treatment they get, when the treatment
#' also changes whether the outcome is observed (Lee, 2009): wages when
#' training changes employment, test scores under attrition. Under
#' monotonicity, the arm with the higher selection rate contains a share
#' \eqn{q = |p_1 - p_0| / \max(p_1, p_0)} of units observed only because of
#' their treatment; trimming that share from the top or the bottom of its
#' outcomes gives the bounds. Discrete covariate cells (`strata`) tighten them.
#'
#' @param y Outcome (any value, e.g. `NA`, where it is not observed).
#' @param d Binary treatment (0/1).
#' @param s Selection indicator (1 = outcome observed).
#' @param strata Discrete covariate cells (any labels), or `NULL`.
#' @param n_boot Bootstrap replications for the standard errors (units are
#'   resampled within arms; 0 skips inference).
#' @param level Confidence level of the Imbens-Manski interval.
#' @param seed Seed of the bootstrap.
#' @return A `dmlrs_lee_bounds` object with `lower`, `upper`, `se` (of the
#'   two bounds), `confint` (the Imbens-Manski interval for the effect),
#'   `trimming` (share trimmed), `trimmed_treated` and `n_cells`.
#' @examples
#' set.seed(11)
#' n <- 2000
#' d <- rbinom(n, 1, 0.5)
#' ability <- rnorm(n)
#' s <- as.numeric(ability + 0.5 * d + rnorm(n) > 0)
#' y <- ifelse(s == 1, 1 + 0.2 * d + ability + rnorm(n), NA)
#' lee_bounds(y, d, s)
#' @export
lee_bounds <- function(y, d, s, strata = NULL, n_boot = 500, level = 0.95, seed = 0) {
  y <- as.numeric(y)
  n <- length(y)
  d <- as_numeric_vector(d, "d", n)
  s <- as_numeric_vector(s, "s", n)
  check_binary(d, "d")
  check_binary(s, "s")
  if (any(!is.finite(y[s == 1]))) {
    stop("`y` must be finite where the outcome is observed (s = 1)", call. = FALSE)
  }
  y[s == 0] <- NaN
  if (!is.null(strata)) {
    if (length(strata) != n || anyNA(strata)) {
      stop("`strata` needs one non-missing label per observation", call. = FALSE)
    }
    strata <- as.integer(factor(strata)) - 1L
  }
  res <- rs_lee_bounds(y, d, s, strata, check_count(n_boot, "n_boot", 0), level, seed)
  structure(c(res, list(level = level, call = match.call())), class = "dmlrs_lee_bounds")
}

#' @export
print.dmlrs_lee_bounds <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(sprintf("Lee bounds (%s arm trimmed by %.1f%%%s)\n\n",
              if (x$trimmed_treated) "treated" else "control", 100 * x$trimming,
              if (x$n_cells > 1) sprintf(", %d covariate cells", x$n_cells) else ""))
  table <- rbind(c(x$lower, x$se[1]), c(x$upper, x$se[2]))
  dimnames(table) <- list(c("lower bound", "upper bound"), c("Estimate", "Std. Error"))
  print(table, digits = digits)
  cat(sprintf("\n%g%% Imbens-Manski interval for the effect: [%s, %s]\n", 100 * x$level,
              format(x$confint[1], digits = digits), format(x$confint[2], digits = digits)))
  invisible(x)
}

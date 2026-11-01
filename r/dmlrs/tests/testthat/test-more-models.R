test_that("mediation returns effects and potential outcomes", {
  set.seed(21)
  n <- 600
  x <- matrix(rnorm(n * 3), n)
  d <- rbinom(n, 1, plogis(x[, 1]))
  m <- d + x[, 1] + rnorm(n)
  y <- 0.5 * d + m + x[, 2] + rnorm(n)
  fit <- dml_mediation(y, d, m, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
  expect_named(coef(fit), c("total", "direct (treated)", "direct (control)",
                            "indirect (treated)", "indirect (control)"))
  po <- coef(fit$potential_outcomes)
  expect_equal(unname(coef(fit)[1]), unname(po[1] - po[4]), tolerance = 1e-10)
  expect_output(print(fit), "indirect")
  expect_error(dml_mediation(y, x[, 1], m, x), "binary")
  drop <- dml_mediation(y, d, m, x, ml_g = lrn_ols(), ml_m = lrn_logistic(),
                        trimming_rule = "drop", trimming_threshold = 0.05)
  expect_true(drop$n_trimmed >= 0)
})

test_that("continuous DiD prints and plots", {
  set.seed(22)
  n <- 800
  dose <- ifelse(runif(n) < 0.3, 0, runif(n, 0.5, 2))
  dy <- dose * (1 + 0.5 * dose) + rnorm(n)
  fit <- dml_contdid(dy, dose, grid = c(0.75, 1, 1.5))
  expect_s3_class(fit, "dmlrs_contdid")
  expect_equal(fit$grid, c(0.75, 1, 1.5))
  expect_length(coef(fit$att), 3)
  expect_output(print(fit), "Overall")
  pdf(NULL)
  on.exit(dev.off())
  expect_invisible(plot(fit, "acrt"))
  expect_error(dml_contdid(dy, dose, ml_g = lrn_ols(), ml_m = lrn_logistic()), "covariates")
})

test_that("APOS contrasts equal differences of the levels", {
  set.seed(23)
  n <- 900
  x <- matrix(rnorm(n * 2), n)
  d <- sample(c(0, 1, 2), n, replace = TRUE)
  y <- d * x[, 1] + x[, 2] + rnorm(n)
  fit <- dml_apos(y, d, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
  expect_length(coef(fit), 3)
  contrast <- linear_combination(fit, rbind("2 vs 0" = c(-1, 0, 1)))
  expect_named(coef(contrast), "2 vs 0")
  expect_equal(unname(coef(contrast)), unname(coef(fit)[3] - coef(fit)[1]), tolerance = 1e-10)
  gamma <- double_robust_scores(fit)
  expect_equal(dim(gamma), c(n, 3))
  expect_equal(unname(colMeans(gamma)), unname(coef(fit)), tolerance = 1e-10)
  expect_error(double_robust_scores(dml_plr(y, x[, 1], x[, 2, drop = FALSE], ml_l = lrn_ols())),
               "dml_apos")
})

test_that("policy trees predict, print and survive saveRDS", {
  set.seed(24)
  n <- 300
  x <- matrix(rnorm(n * 2), n, dimnames = list(NULL, c("age", "income")))
  gamma <- cbind(control = rnorm(n), treat = 2 * (x[, 2] > 0.3) - 1 + rnorm(n, sd = 0.1))
  tree <- policy_tree(x, gamma, depth = 1)
  expect_equal(tree$nodes$split_variable[1], 2)
  expect_equal(predict(tree, x), ifelse(x[, 2] <= tree$nodes$split_value[1],
                                        tree$nodes$action[2], tree$nodes$action[3]))
  expect_output(print(tree), "income <=")
  path <- tempfile(fileext = ".rds")
  saveRDS(tree, path)
  expect_identical(predict(readRDS(path), x), predict(tree, x))
  values <- policy_value(gamma, tree = predict(tree, x), all = rep(2L, n))
  expect_named(coef(values), c("tree", "all"))
  expect_error(policy_value(gamma, rep(1, n)), "named")
  expect_error(policy_value(gamma, bad = rep(3, n)), "column numbers")
})

test_that("policy trees equal policytree's", {
  skip_if_not_installed("policytree")
  set.seed(25)
  n <- 200
  x <- matrix(round(rnorm(n * 3), 1), n)
  gamma <- cbind(rnorm(n), rnorm(n) + x[, 1], rnorm(n) - x[, 2])
  for (depth in 1:2) {
    ours <- policy_tree(x, gamma, depth = depth)
    theirs <- policytree::policy_tree(x, gamma, depth = depth)
    expect_equal(predict(ours, x), as.integer(predict(theirs, x)), label = paste("depth", depth))
  }
})

test_that("synthetic DiD works on matrices and long panels", {
  set.seed(26)
  units <- 20
  periods <- 10
  y <- matrix(rnorm(units * periods), units, periods,
              dimnames = list(paste0("u", 1:units), 2001:2010)) + outer(rnorm(units), 1:10 / 3)
  y[19:20, 7:10] <- y[19:20, 7:10] + 2
  fit <- synth_did(y, 18, 6)
  expect_s3_class(fit, "dmlrs_synthdid")
  expect_equal(sum(fit$unit_weights), 1, tolerance = 1e-10)
  expect_named(fit$effect_curve, as.character(2007:2010))
  expect_true(is.finite(fit$se))
  expect_output(print(fit), "synthetic DiD")
  expect_equal(dim(confint(fit)), c(1, 2))
  pdf(NULL)
  on.exit(dev.off())
  expect_invisible(plot(fit))
  long <- data.frame(unit = rep(rownames(y), periods), year = rep(2001:2010, each = units),
                     outcome = as.vector(y))
  long$treated <- as.numeric(long$unit %in% c("u19", "u20") & long$year >= 2007)
  long <- long[sample(nrow(long)), ]
  panel <- synth_did_panel(long, "unit", "year", "outcome", "treated", se_method = "none")
  # synthdid's ordering sorts the unit labels as strings (u1, u10, u11, ...).
  order <- c(sort(rownames(y)[1:18]), "u19", "u20")
  direct <- synth_did(y[order, ], 18, 6, se_method = "none")
  expect_equal(panel$estimate, direct$estimate, tolerance = 1e-12)
  expect_equal(panel$unit_weights, direct$unit_weights, tolerance = 1e-12)
  expect_warning(synth_did(y, 18, 6, se_method = "jackknife")$se, NA)
  expect_warning(one <- synth_did(y[1:19, ], 18, 6, se_method = "jackknife"), "jackknife")
  expect_true(is.na(one$se))
  expect_error(synth_did_panel(long, "unit", "year", "outcome", "missing"), "columns")
})

test_that("Lee bounds contain the naive estimate's direction and print", {
  set.seed(27)
  n <- 3000
  d <- rbinom(n, 1, 0.5)
  ability <- rnorm(n)
  s <- as.numeric(ability + 0.5 * d + rnorm(n) > 0)
  y <- ifelse(s == 1, 1 + 0.2 * d + ability + rnorm(n), NA)
  lee <- lee_bounds(y, d, s, n_boot = 100)
  expect_true(lee$lower < 0.2 && 0.2 < lee$upper)
  expect_true(lee$trimmed_treated)
  expect_true(lee$confint[1] <= lee$lower && lee$upper <= lee$confint[2])
  expect_output(print(lee), "Imbens-Manski")
  cells <- lee_bounds(y, d, s, strata = ifelse(ability > 0, "high", "low"), n_boot = 0)
  expect_equal(cells$n_cells, 2)
  expect_error(lee_bounds(ifelse(s == 1, NA, y), d, s), "finite")
})

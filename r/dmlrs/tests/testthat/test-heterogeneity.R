# Heterogeneous effects and dose response: the same estimates as the Python package (fixtures
# from parity/generate_r_fixtures.py), plus checks of the R-side code.

p <- read_parity()
tol <- 1e-8
x <- p$x
y <- p$data[, "y"]
d <- p$data[, "d"]
basis <- cbind(1, x[, 2])
groups <- cbind(as.numeric(x[, 2] <= 0), as.numeric(x[, 2] > 0))

test_that("cate() and gate() of PLR and IRM fits match Python", {
  fits <- list(irm = dml_irm(y, d, x, ml_g = lrn_ols(), ml_m = lrn_logistic()),
               plr = dml_plr(y, p$data[, "dcont"], x, ml_l = lrn_ols()))
  for (name in names(fits)) {
    blp <- cate(fits[[name]], basis)
    at <- predict(blp, basis[1:3, ])
    expect_equal(c(unname(coef(blp)), unname(blp$se), at$estimate, at$se),
                 p$expected[[paste0(name, "_cate")]], tolerance = tol, label = name)
    g <- gate(fits[[name]], groups)
    expect_equal(c(unname(coef(g)), unname(g$se)), p$expected[[paste0(name, "_gate")]],
                 tolerance = tol, label = name)
  }
})

test_that("gate() takes a factor and BLP intervals are consistent", {
  fit <- dml_irm(y, d, x, ml_g = lrn_ols(), ml_m = lrn_logistic())
  by_factor <- gate(fit, factor(ifelse(x[, 2] > 0, "high", "low"), levels = c("low", "high")))
  by_dummies <- gate(fit, groups)
  expect_equal(unname(coef(by_factor)), unname(coef(by_dummies)))
  expect_named(coef(by_factor), c("low", "high"))
  blp <- cate(fit, basis)
  ci <- confint(blp)
  expect_equal(unname(rowMeans(ci)), unname(coef(blp)))
  pointwise <- predict(blp, basis[1:5, ], interval = "pointwise")
  joint <- predict(blp, basis[1:5, ], interval = "joint", n_boot = 2000)
  expect_true(all(joint$upper - joint$lower > pointwise$upper - pointwise$lower))
  joint_coef <- confint(by_dummies, joint = TRUE)
  expect_true(all(joint_coef[, 2] - joint_coef[, 1] > confint(by_dummies)[, 2] -
                    confint(by_dummies)[, 1]))
})

test_that("the CATE meta-learners match Python", {
  ols <- lrn_ols()
  for (method in c("S", "T", "X", "R", "DR")) {
    fit <- cate_learner(y, d, x, method = method, ml_outcome = ols, ml_propensity = lrn_logistic(),
                        ml_effect = ols)
    expect_equal(predict(fit, x[1:5, ]), p$expected[[paste0("cate_", method)]], tolerance = tol,
                 label = method)
    expect_equal(fit$effect, predict(fit, x))
  }
  dr <- cate_learner(y, d, x, ml_outcome = ols, ml_propensity = lrn_logistic(), ml_effect = ols)
  blp <- cate(dr, basis)
  expect_equal(c(unname(coef(blp)), unname(blp$se)), p$expected$cate_DR_blp, tolerance = tol)
  expect_length(dr$pseudo_outcomes, nrow(x))
  t_learner <- cate_learner(y, d, x, method = "T", ml_outcome = ols)
  expect_null(t_learner$pseudo_outcomes)
  expect_error(cate(t_learner, basis), "DR-learner")
})

test_that("causal and instrumental forests match Python", {
  # OLS nuisances (see parity/generate_r_fixtures.py).
  cf <- causal_forest(y, d, x, ml_y = lrn_ols(), ml_d = lrn_ols(), num_trees = 200)
  at <- predict(cf, x[1:3, ])
  ate <- average_treatment_effect(cf)
  expect_equal(c(cf$predictions[1:5], at$estimate, at$se, unname(coef(ate)), unname(ate$se),
                 unname(coef(cate(cf, basis)))), p$expected$causal_forest, tolerance = tol)
  expect_equal(predict(cf)$estimate, cf$predictions)
  calibration <- test_calibration(cf)
  expect_equal(dim(calibration), c(2L, 4L))

  ivf <- instrumental_forest(y, p$data[, "d_iv"], p$data[, "z"], x, ml_y = lrn_ols(),
                             ml_d = lrn_ols(), ml_z = lrn_ols(), num_trees = 200)
  at <- predict(ivf, x[1:3, ])
  ate <- average_treatment_effect(ivf)
  expect_equal(c(ivf$predictions[1:5], at$estimate, unname(coef(ate)), unname(ate$se)),
               p$expected$instrumental_forest, tolerance = tol)
  expect_error(test_calibration(ivf), "causal_forest")
})

test_that("conformal intervals match Python and the quantile score covers", {
  fit <- conformal_ite(y, d, x, ml_y = lrn_ols(), ml_m = lrn_logistic())
  ite <- predict(fit, x[1:3, ], alpha = 0.1)
  y0 <- predict(fit, x[1:3, ], type = "y0", alpha = 0.2, target = "treated")
  expect_equal(c(t(ite), t(y0)), p$expected$conformal, tolerance = tol)
  expect_equal(colnames(ite), c("lower", "upper"))

  set.seed(11)
  n <- 3000
  xs <- matrix(runif(n * 2, -1, 1), n)
  ds <- rbinom(n, 1, 0.5)
  noise <- (0.5 + abs(xs[, 1])) * rnorm(n)
  ys <- ds + xs[, 2] + noise
  cqr <- conformal_ite(ys, ds, xs, ml_lower = lrn_quantile_forest(0.05, n_estimators = 200),
                       ml_upper = lrn_quantile_forest(0.95, n_estimators = 200), propensity = 0.5)
  expect_equal(cqr$score, "quantile")
  new_x <- matrix(runif(1000 * 2, -1, 1), 1000)
  y1 <- 1 + new_x[, 2] + (0.5 + abs(new_x[, 1])) * rnorm(1000)
  band <- predict(cqr, new_x, type = "y1", alpha = 0.1)
  expect_gt(mean(band[, 1] <= y1 & y1 <= band[, 2]), 0.85)
  # Adaptive: wider where the noise is larger.
  width <- band[, 2] - band[, 1]
  expect_gt(mean(width[abs(new_x[, 1]) > 0.7]), mean(width[abs(new_x[, 1]) < 0.3]))
  expect_error(conformal_ite(ys, ds, xs, ml_y = lrn_ols()), "propensity")
  expect_error(conformal_ite(ys, ds, xs, propensity = 0.5), "ml_y")
  expect_error(dml_plr(ys, ds, xs, ml_l = lrn_quantile_forest(0.5)), "regressor")
})

test_that("dose-response curves match Python", {
  fit <- dml_dose_response(y, p$data[, "dcont"], x, ml_g = lrn_ols(), ml_m = lrn_ols(),
                           grid = c(-1, 0, 1), partial_effects = TRUE)
  contrast <- dose_contrast(fit, 0)
  expect_equal(c(unname(coef(fit$curve)), unname(fit$curve$se), unname(coef(fit$partial)),
                 unname(coef(contrast)), unname(contrast$se)), p$expected$dose_response,
               tolerance = tol)
  expect_equal(fit$grid, c(-1, 0, 1))
  expect_false(is.null(fit$eta))
  no_partial <- dml_dose_response(y, p$data[, "dcont"], x, ml_g = lrn_ols(), ml_m = lrn_ols(),
                                  grid = c(-1, 0, 1))
  expect_null(no_partial$partial)
  expect_null(no_partial$eta)
  expect_error(dose_contrast(fit, 0.5), "grid point")
  pdf(NULL)
  on.exit(dev.off())
  expect_invisible(plot(fit))
  expect_invisible(plot(fit, type = "partial"))
  expect_error(plot(no_partial, type = "partial"), "partial_effects")
})

test_that("restored objects say that they need a refit", {
  cf <- causal_forest(y, d, x, ml_y = lrn_ols(), ml_d = lrn_logistic(), num_trees = 50)
  path <- tempfile(fileext = ".rds")
  saveRDS(cf, path)
  restored <- readRDS(path)
  expect_equal(predict(restored)$estimate, cf$predictions)
  expect_error(predict(restored, x[1:2, ]), "refit")
})

test_that("binned causal forests", {
  set.seed(5)
  n <- 1500
  xb <- matrix(runif(n * 3), n)
  db <- rbinom(n, 1, 0.5)
  yb <- (1 + (xb[, 1] > 0.5)) * db + xb[, 2] + rnorm(n)
  exact <- causal_forest(yb, db, xb, ml_y = lrn_ols(), ml_d = lrn_logistic(), num_trees = 200)
  binned <- causal_forest(yb, db, xb, ml_y = lrn_ols(), ml_d = lrn_logistic(), num_trees = 200,
                          max_bins = 64)
  expect_lt(abs(mean(binned$predictions) - mean(exact$predictions)), 0.1)
  expect_error(causal_forest(yb, db, xb, ml_y = lrn_ols(), ml_d = lrn_logistic(), max_bins = 1),
               "max_bins")
})

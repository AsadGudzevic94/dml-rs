simulate <- function(n = 500, seed = 1) {
  set.seed(seed)
  x <- matrix(rnorm(n * 4), n, dimnames = list(NULL, paste0("x", 1:4)))
  d <- x[, 1] + rnorm(n)
  list(x = x, d = d, y = 0.5 * d + x[, 2] + rnorm(n), db = rbinom(n, 1, plogis(x[, 1])))
}

test_that("a fitted model has the usual methods", {
  s <- simulate()
  fit <- dml_plr(s$y, s$d, s$x, ml_l = lrn_ols())
  expect_s3_class(fit, "dmlrs_fit")
  expect_named(coef(fit), "d")
  expect_equal(nobs(fit), 500)
  expect_lt(abs(coef(fit) - 0.5), 4 * fit$se)
  ci <- confint(fit, level = 0.9)
  expect_equal(colnames(ci), c("5 %", "95 %"))
  expect_true(ci[1, 1] < coef(fit) && coef(fit) < ci[1, 2])
  expect_output(print(fit), "PLR")
  expect_output(print(summary(fit)), "Estimate")
  expect_output(print(sensitivity_analysis(fit)), "rv")
})

test_that("several treatments and data frames work", {
  s <- simulate()
  d <- cbind(first = s$d, second = s$x[, 3] + rnorm(500))
  fit <- dml_plr(s$y, d, as.data.frame(s$x), ml_l = lrn_ols())
  expect_named(coef(fit), c("first", "second"))
  joint <- confint(fit, joint = TRUE)
  pointwise <- confint(fit)
  expect_true(all(joint[, 1] <= pointwise[, 1]) && all(joint[, 2] >= pointwise[, 2]))
  expect_equal(rownames(confint(fit, "second")), "second")
})

test_that("results are reproducible and repetitions aggregate", {
  s <- simulate()
  a <- dml_irm(s$y, s$db, s$x, ml_g = lrn_random_forest(n_estimators = 20), ml_m = lrn_logistic())
  b <- dml_irm(s$y, s$db, s$x, ml_g = lrn_random_forest(n_estimators = 20), ml_m = lrn_logistic())
  expect_identical(coef(a), coef(b))
  rep <- dml_plr(s$y, s$d, s$x, ml_l = lrn_ols(), n_rep = 3, seed = 4)
  expect_equal(dim(rep$all_coef), c(1, 3))
  expect_equal(unname(coef(rep)), stats::median(rep$all_coef[1, ]))
})

test_that("input errors are clear", {
  s <- simulate()
  expect_error(dml_plr(s$y, s$d, s$x[1:10, ]), "rows")
  expect_error(dml_irm(s$y, s$d, s$x), "binary")
  expect_error(dml_irm(s$y, s$db, s$x, ml_m = lrn_lasso()), "classifier")
  expect_error(dml_plr(s$y, s$d, s$x, ml_l = "lasso"), "dmlrs learner")
  expect_error(dml_plr(s$y, s$d, data.frame(a = letters[1:500])), "non-numeric")
  bad <- s$x
  bad[1, 1] <- NA
  expect_error(dml_plr(s$y, s$d, bad), "missing")
  expect_error(dml_plr(s$y, s$d, s$x, n_folds = 1), "n_folds")
  expect_error(dml_plr(s$y, s$d, s$x, ml_l = lrn_random_forest(max_features = "half")),
               "max_features")
})

test_that("a saved fit keeps its estimates", {
  s <- simulate()
  fit <- dml_plr(s$y, s$d, s$x, ml_l = lrn_ols())
  path <- tempfile(fileext = ".rds")
  saveRDS(fit, path)
  back <- readRDS(path)
  expect_identical(coef(back), coef(fit))
  expect_output(print(back), "PLR")
  expect_error(confint(back), "refit")
})

test_that("all learners fit", {
  s <- simulate(300)
  regressors <- list(lrn_ols(ridge = 0.1), lrn_lasso(alpha = 0.05), lrn_lasso_cv(n_alphas = 20),
                     lrn_random_forest(n_estimators = 20, max_features = 0.5),
                     lrn_gradient_boosting(n_estimators = 20),
                     lrn_hist_gradient_boosting(max_iter = 20),
                     lrn_regression_forest(n_estimators = 100))
  for (learner in regressors) {
    expect_true(is.finite(coef(dml_plr(s$y, s$d, s$x, ml_l = learner, n_folds = 2))), label = learner$type)
  }
  classifiers <- list(lrn_logistic(C = 0.5), lrn_logistic_cv(n_cs = 5),
                      lrn_random_forest(n_estimators = 20), lrn_gradient_boosting(n_estimators = 20),
                      lrn_hist_gradient_boosting(max_iter = 20))
  for (learner in classifiers) {
    fit <- dml_irm(s$y, s$db, s$x, ml_g = lrn_ols(), ml_m = learner, n_folds = 2)
    expect_true(is.finite(coef(fit)), label = learner$type)
  }
  expect_output(print(lrn_lasso_cv()), "lasso_cv")
})

test_that("DID works for panels and repeated cross-sections", {
  s <- simulate()
  t <- rbinom(500, 1, 0.5)
  panel <- dml_did(s$y, s$db, s$x, ml_g = lrn_ols(), ml_m = lrn_logistic())
  cs <- dml_did(s$y + s$db * t, s$db, s$x, t = t, ml_g = lrn_ols(), ml_m = lrn_logistic())
  experimental <- dml_did(s$y, s$db, s$x, ml_g = lrn_ols(), score = "experimental")
  for (fit in list(panel, cs, experimental)) expect_true(is.finite(coef(fit)))
  expect_match(cs$model, "DID")
})

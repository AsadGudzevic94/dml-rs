# Confidence sequences and prediction-powered inference against the Python
# package (fixtures from parity/generate_r_fixtures.py).

p <- read_parity()
size <- file.info(test_path("fixtures", "parity_ppi.bin"))$size
raw <- readBin(test_path("fixtures", "parity_ppi.bin"), "double", n = size / 8, size = 8,
               endian = "little")
ppi_data <- matrix(raw, ncol = 7)
colnames(ppi_data) <- c("y", "yhat", "b", "phat", "z1", "z2", "z3")
lab <- 1:300
unl <- 301:nrow(ppi_data)
xp <- ppi_data[, c("z1", "z2")]
zp <- ppi_data[, c("z1", "z2", "z3")]

ppi_values <- function(fit) {
  unname(c(fit$coefficients, fit$se, fit$lambda, fit$classical$se, fit$effective_n))
}

test_that("confidence sequences match the Python package", {
  fit <- dml_plr(p$data[, "y"], p$data[, "dcont"], p$x, ml_l = lrn_lasso_cv(), n_rep = 2)
  cs <- confidence_sequence(fit, level = 0.9, opt_time = 150)
  expect_equal(c(unname(cs[1, ]), unname(anytime_pvalue(fit, 0.9, 150))),
               p$expected$plr_confidence_sequence, tolerance = 1e-8)
  expect_true(cs[1, 1] < confint(fit, level = 0.9)[1, 1])
  # The default opt_time is the fit's own sample size.
  expect_equal(confidence_sequence(fit), confidence_sequence(fit, opt_time = 400))

  looks <- c(100, 200, 400)
  fits <- lapply(looks, function(t) {
    dml_plr(p$data[1:t, "y"], p$data[1:t, "dcont"], p$x[1:t, ], ml_l = lrn_ols())
  })
  mon <- monitor_study(fits, opt_time = 150)
  expect_equal(c(mon$table$cs_lower, mon$table$cs_upper, mon$table$p_anytime), p$expected$monitor,
               tolerance = 1e-8)
  expect_equal(mon$table$n, looks)
  expect_true(all(diff(mon$table$cs_lower) >= 0) && all(diff(mon$table$cs_upper) <= 0))
  expect_equal(unname(mon$first_exclusion), 100L)
  expect_output(print(mon), "zero excluded from n = 100")
  pdf(NULL)
  plot(mon)
  dev.off()
  expect_error(monitor_study(rev(fits)), "growing sample sizes")
  expect_error(monitor_study(fits[[1]]), "list of fitted")
  expect_error(confidence_sequence(fit, opt_time = 0), "opt_time")
})

test_that("prediction-powered inference matches the Python package", {
  y <- ppi_data[, "y"]
  yhat <- ppi_data[, "yhat"]
  expect_equal(ppi_values(ppi(y[lab], yhat[lab], yhat[unl])), p$expected$ppi_mean, tolerance = 1e-10)
  expect_equal(ppi_values(ppi(y[lab], yhat[lab], yhat[unl], lambda = 1)), p$expected$ppi_mean_lam1,
               tolerance = 1e-10)
  ols <- ppi(y[lab], yhat[lab], yhat[unl], xp[lab, ], xp[unl, ], estimand = "ols")
  expect_equal(ppi_values(ols), p$expected$ppi_ols, tolerance = 1e-10)
  expect_equal(names(coef(ols)), c("(Intercept)", "z1", "z2"))
  b <- ppi_data[, "b"]
  phat <- ppi_data[, "phat"]
  logit <- ppi(b[lab], phat[lab], phat[unl], xp[lab, ], xp[unl, ], estimand = "logistic")
  expect_equal(ppi_values(logit), p$expected$ppi_logistic, tolerance = 1e-10)
  cross <- ppi(y[lab], learner = lrn_ols(), features = zp[lab, ], features_unlabeled = zp[unl, ],
               n_folds = 4, seed = 3)
  expect_equal(ppi_values(cross), p$expected$ppi_cross, tolerance = 1e-10)
  expect_length(cross$yhat, length(lab))
  cross_logit <- ppi(b[lab], x = xp[lab, ], x_unlabeled = xp[unl, ], estimand = "logistic",
                     learner = lrn_logistic(), features = zp[lab, ], features_unlabeled = zp[unl, ])
  expect_equal(ppi_values(cross_logit), p$expected$ppi_cross_logistic, tolerance = 1e-8)
})

test_that("prediction-powered fits behave like other fits", {
  y <- ppi_data[, "y"]
  yhat <- ppi_data[, "yhat"]
  fit <- ppi(y[lab], yhat[lab], yhat[unl])
  expect_s3_class(fit, "dmlrs_fit")
  expect_true(fit$se < fit$classical$se)
  expect_equal(dim(confint(fit)), c(1L, 2L))
  expect_equal(dim(confidence_sequence(fit)), c(1L, 2L))
  expect_output(print(fit), "Prediction-powered inference")
  expect_error(ppi(y[lab], yhat[lab], yhat[unl], lambda = 2), "lambda")
  expect_error(ppi(y[lab], yhat[lab], yhat[unl], estimand = "ols"), "needs `x`")
  expect_error(ppi(y[lab], yhat[lab], yhat[unl], x = xp[lab, ], x_unlabeled = xp[unl, ]),
               "no covariates")
  expect_error(ppi(y[lab]), "yhat")
  expect_error(ppi(y[lab], yhat[lab], yhat[unl], learner = lrn_ols(), features = zp[lab, ],
                   features_unlabeled = zp[unl, ]), "not both")
})

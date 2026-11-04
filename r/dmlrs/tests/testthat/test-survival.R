# Censored time to event: the same estimates as the Python package (fixtures from
# parity/generate_r_fixtures.py), plus checks of the R-side code.

size <- file.info(test_path("fixtures", "parity_survival.bin"))$size
raw <- readBin(test_path("fixtures", "parity_survival.bin"), "double", n = size / 8, size = 8,
               endian = "little")
data <- matrix(raw, ncol = 5)
time <- data[, 1]
event <- data[, 2]
d <- data[, 3]
x <- data[, 4:5]
expected <- read_parity()$expected

fit_survival <- function(...) {
  dml_survival(time, event, d, x, ml_event = lrn_logistic(), ml_m = lrn_logistic(), grid = 10,
               time_encoding = "dummies", ...)
}

test_that("estimates match Python", {
  for (estimand in c("survival", "rmst")) {
    fit <- fit_survival(horizons = c(0.5, 1), estimand = estimand)
    expect_equal(c(unname(coef(fit)), unname(fit$se), unname(coef(fit$control)), unname(fit$treated$se),
                   unname(coef(fit$curve_treated)), fit$grid),
                 expected[[paste0("survival_", estimand)]], tolerance = 1e-8, label = estimand)
  }
})

test_that("fits work with the other dmlrs methods", {
  fit <- fit_survival(horizons = c(0.5, 1), n_rep = 2)
  expect_s3_class(fit, "dmlrs_fit")
  expect_equal(dim(confint(fit)), c(2L, 2L))
  expect_equal(dim(fit$propensity), c(length(time), 2L))
  expect_equal(fit$horizons, c(0.5, 1))
  expect_length(coef(fit$curve_control), length(fit$grid))
  expect_true(all(diff(unname(coef(fit$curve_control))) < 0.05))
  pdf(NULL)
  on.exit(dev.off())
  expect_silent(plot(fit))
})

test_that("the band of a step function holds each value until the next time", {
  band <- dmlrs:::step_band(c(0, 1, 3), c(1, 0.5, 0.2), c(1, 0.7, 0.4))
  expect_equal(band$x, c(0, 1, 1, 3, 3, 1, 1, 0))
  expect_equal(band$y, c(1, 1, 0.5, 0.5, 0.7, 0.7, 1, 1))
})

test_that("every form of the grid works", {
  fixed <- dml_survival(time, event, d, x, ml_event = lrn_logistic(), ml_m = lrn_logistic(),
                        grid = c(0.25, 0.5, 1), horizons = 1, n_folds = 2)
  expect_equal(fixed$grid, c(0.25, 0.5, 1))
  everything <- dml_survival(time[1:150], event[1:150], d[1:150], x[1:150, ], ml_event = lrn_logistic(),
                             ml_m = lrn_logistic(), grid = "event_times", horizons = 0.5, n_folds = 2)
  expect_gt(length(everything$grid), 10)
})

test_that("inputs are checked", {
  expect_error(fit_survival(horizons = 1e6), "last event time")
  expect_error(dml_survival(time, event + 0.5, d, x), "`event` must be 0 for censoring")
  expect_error(dml_survival(time, event, d + 1, x), "`d` must be binary")
  expect_error(dml_survival(time, event[-1], d, x), "`event`")
  expect_error(dml_survival(time, event, d, x, grid = "weekly"), "`grid`")
  expect_error(dml_survival(time, event, d, x, estimand = "hazard"), "should be one of")
  path <- tempfile(fileext = ".rds")
  saveRDS(fit_survival(horizons = 1), path)
  restored <- readRDS(path)
  expect_length(coef(restored), 1)
  expect_error(confint(restored), "saved session")
})

test_that("competing events: the incidences of all causes and survival nearly sum to one", {
  set.seed(4)
  n <- 600
  xs <- matrix(runif(n * 2, -1, 1), n)
  ds <- rbinom(n, 1, plogis(1.5 * xs[, 1]))
  l1 <- 0.6 * exp(0.8 * xs[, 1]) * ifelse(ds == 1, 0.5, 1)
  l2 <- 0.4 * exp(-0.5 * xs[, 1])
  tt <- rexp(n, l1 + l2)
  cause <- ifelse(runif(n) < l1 / (l1 + l2), 1, 2)
  cc <- rexp(n, 0.3)
  ts <- pmin(tt, cc)
  es <- ifelse(tt <= cc, cause, 0)
  fit <- function(...) {
    dml_survival(ts, es, ds, xs, ml_event = lrn_logistic(), ml_m = lrn_logistic(), grid = 10,
                 horizons = 1, time_encoding = "dummies", ...)
  }
  surv <- fit()
  first <- fit(estimand = "cumulative_incidence", cause = 1)
  second <- fit(estimand = "cumulative_incidence", cause = 2)
  # Exactly one only when the learned cause hazards add up to that of any event (empirical hazards
  # do); three separate logistic fits come close.
  total <- coef(surv$curve_treated) + coef(first$incidence_treated) + coef(second$incidence_treated)
  expect_equal(unname(total), rep(1, length(surv$grid)), tolerance = 0.01)
  expect_lt(unname(coef(first)), 0)
  expect_null(surv$incidence_treated)
  pdf(NULL)
  on.exit(dev.off())
  expect_silent(plot(first))
  expect_error(fit(estimand = "cumulative_incidence", cause = 3), "cause")
  expect_error(fit(cluster = 1:3), "one value per observation")
})

test_that("clustered standard errors and binned forests", {
  rep3 <- rep(seq_len(200), each = 3)
  naive <- dml_survival(time[rep3], event[rep3], d[rep3], x[rep3, ], ml_event = lrn_logistic(),
                        ml_m = lrn_logistic(), grid = 10, horizons = 1, time_encoding = "dummies")
  clustered <- dml_survival(time[rep3], event[rep3], d[rep3], x[rep3, ], ml_event = lrn_logistic(),
                            ml_m = lrn_logistic(), grid = 10, horizons = 1,
                            time_encoding = "dummies", cluster = rep3)
  expect_gt(clustered$se, 1.4 * naive$se)
  forest <- lrn_random_forest(n_estimators = 30, min_samples_leaf = 20, max_bins = 64)
  binned <- dml_survival(time, event, d, x, ml_event = forest, ml_m = lrn_logistic(), grid = 10,
                         horizons = 1)
  expect_true(is.finite(coef(binned)) && binned$se > 0)
})

# The R package and the Python package dml-rs run the same Rust code: on the same data
# (exact doubles from parity/generate_r_fixtures.py) they give the same estimates.

read_parity <- function() {
  n <- 400
  raw <- readBin(test_path("fixtures", "parity_data.bin"), "double", n = 10 * n, size = 8,
                 endian = "little")
  cols <- matrix(raw, nrow = n)
  colnames(cols) <- c(paste0("x", 1:4), "y", "d", "dcont", "z", "d_iv", "ydid")
  expected <- utils::read.csv(test_path("fixtures", "parity_expected.csv"))
  list(data = cols, x = cols[, 1:4], expected = split(expected$value, expected$name))
}

p <- read_parity()
tol <- 1e-10

test_that("PLR, its intervals and its sensitivity analysis match Python", {
  fit <- dml_plr(p$data[, "y"], p$data[, "dcont"], p$x, ml_l = lrn_lasso_cv(), n_rep = 2)
  ref <- p$expected$plr
  expect_equal(unname(coef(fit)), ref[1], tolerance = tol)
  expect_equal(unname(fit$se), ref[2], tolerance = tol)
  expect_equal(unname(confint(fit)[1, ]), ref[3:4], tolerance = tol)
  joint <- confint(fit, joint = TRUE, n_boot = 500, seed = 0)
  expect_equal(unname(joint[1, ]), ref[5:6], tolerance = tol)
  sens <- sensitivity_analysis(fit, cf_y = 0.04, cf_d = 0.03)
  expect_equal(c(sens$theta_lower, sens$theta_upper, sens$ci_lower, sens$ci_upper,
                 sens$rv / 100, sens$rva / 100), p$expected$plr_sensitivity, tolerance = 1e-8)
})

test_that("IRM, PLIV, IIVM and DID match Python", {
  ols <- lrn_ols()
  logit <- lrn_logistic()
  fits <- list(
    irm = dml_irm(p$data[, "y"], p$data[, "d"], p$x, ml_g = lrn_random_forest(n_estimators = 30),
                  ml_m = logit),
    pliv = dml_pliv(p$data[, "y"], p$data[, "dcont"], p$data[, "z"], p$x, ml_l = ols),
    iivm = dml_iivm(p$data[, "y"], p$data[, "d_iv"], p$data[, "z"], p$x, ml_g = ols, ml_m = logit),
    did = dml_did(p$data[, "ydid"], p$data[, "d"], p$x, ml_g = ols, ml_m = logit),
    plr_boosting = dml_plr(p$data[, "y"], p$data[, "dcont"], p$x,
                           ml_l = lrn_hist_gradient_boosting(max_iter = 30), ml_m = ols,
                           ml_g = lrn_gradient_boosting(n_estimators = 30), score = "IV-type")
  )
  for (name in names(fits)) {
    expect_equal(c(unname(coef(fits[[name]])), unname(fits[[name]]$se)), p$expected[[name]],
                 tolerance = tol, label = name)
  }
})

# The R package and the Python package dml-rs run the same Rust code: on the same data
# (exact doubles from parity/generate_r_fixtures.py) they give the same estimates.

read_parity <- function() {
  n <- 400
  names <- c(paste0("x", 1:4), "y", "d", "dcont", "z", "d_iv", "ydid", "m", "ymed", "dose", "ycd",
             "d3", "y3", "s", "ylee", "strata")
  raw <- readBin(test_path("fixtures", "parity_data.bin"), "double", n = length(names) * n,
                 size = 8, endian = "little")
  cols <- matrix(raw, nrow = n)
  colnames(cols) <- names
  expected <- utils::read.csv(test_path("fixtures", "parity_expected.csv"))
  list(data = cols, x = cols[, 1:4], expected = split(expected$value, expected$name))
}

p <- read_parity()
# The fixtures come from Python on one machine. Within a platform the two packages agree to
# machine precision; across platforms, last-bit differences in the logistic regression fits (and
# a forest split that depends on them) move the estimates by ~1e-9 relative.
tol <- 1e-8

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

test_that("APOS, contrasts, policy trees and policy values match Python", {
  apos <- dml_apos(p$data[, "y3"], p$data[, "d3"], p$x, ml_g = lrn_ols(), ml_m = lrn_logistic())
  contrast <- linear_combination(apos, rbind(c(-1, 1, 0), c(-1, 0, 1)))
  expect_equal(c(coef(apos), apos$se, coef(contrast), contrast$se), p$expected$apos,
               tolerance = tol, ignore_attr = TRUE)
  gamma <- double_robust_scores(apos)
  tree <- policy_tree(p$x, gamma, depth = 2, min_node_size = 20)
  actions <- predict(tree, p$x)
  expect_equal(actions, as.integer(p$expected$policy_tree))
  values <- policy_value(gamma, tree = actions, treat_all = rep(3, nrow(gamma)))
  expect_equal(c(coef(values), values$se), p$expected$policy_value, tolerance = tol,
               ignore_attr = TRUE)
})

test_that("mediation and continuous DiD match Python", {
  med <- dml_mediation(p$data[, "ymed"], p$data[, "d"], p$data[, "m"], p$x, ml_g = lrn_ols(),
                       ml_m = lrn_logistic())
  expect_equal(c(coef(med), med$se, coef(med$potential_outcomes)), p$expected$mediation,
               tolerance = tol, ignore_attr = TRUE)
  cd <- dml_contdid(p$data[, "ycd"], p$data[, "dose"])
  expect_equal(c(coef(cd$att), cd$att$se, coef(cd$overall), cd$overall$se), p$expected$contdid,
               tolerance = tol, ignore_attr = TRUE)
  cd_x <- dml_contdid(p$data[, "ycd"], p$data[, "dose"], p$x, ml_g = lrn_ols(),
                      ml_m = lrn_logistic())
  expect_equal(c(coef(cd_x$overall), cd_x$overall$se), p$expected$contdid_covariates,
               tolerance = tol, ignore_attr = TRUE)
})

test_that("Lee bounds match Python", {
  for (case in c("lee", "lee_strata")) {
    strata <- if (case == "lee") NULL else p$data[, "strata"]
    lee <- lee_bounds(p$data[, "ylee"], p$data[, "d"], p$data[, "s"], strata = strata,
                      n_boot = 200, seed = 3)
    expect_equal(c(lee$lower, lee$upper, lee$se, lee$confint), p$expected[[case]],
                 tolerance = tol, label = case)
  }
})

test_that("synthetic DiD matches Python", {
  y <- matrix(readBin(test_path("fixtures", "parity_synth.bin"), "double", n = 30 * 15, size = 8,
                      endian = "little"), 30, 15)
  fit <- synth_did(y, 27, 10, se_method = "none")
  ref <- p$expected$synthdid
  expect_equal(fit$estimate, ref[1], tolerance = tol)
  expect_equal(synth_se(fit, "placebo", replications = 50, seed = 1), ref[2], tolerance = tol)
  expect_equal(synth_se(fit, "jackknife"), ref[3], tolerance = tol)
  expect_equal(synth_se(fit, "bootstrap", replications = 50, seed = 2), ref[4], tolerance = tol)
  expect_equal(unname(fit$effect_curve), ref[-(1:4)], tolerance = tol)
  sc <- synth_did(y, 27, 10, estimator = "sc", se_method = "none")
  expect_equal(c(sc$estimate, unname(sc$unit_weights)), p$expected$synthdid_sc, tolerance = tol)
})

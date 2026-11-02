# Staggered DiD and flexible RD: the same estimates as the Python package (fixtures from
# parity/generate_r_fixtures.py), plus checks of the R-side code.

read_bin <- function(file, columns) {
  size <- file.info(test_path("fixtures", file))$size
  raw <- readBin(test_path("fixtures", file), "double", n = size / 8, size = 8, endian = "little")
  m <- matrix(raw, ncol = length(columns))
  colnames(m) <- columns
  as.data.frame(m)
}

expected <- read_parity()$expected
tol <- 1e-8
panel <- read_bin("parity_panel.bin", c("id", "t", "g", "x1", "x2", "y", "cluster", "ycs", "tcs", "gcs"))
rd <- read_bin("parity_rd.bin", c("score", "x1", "x2", "y", "d_sharp", "d_fuzzy", "y_binary", "cl"))
xs <- as.matrix(panel[c("x1", "x2")])
staggered <- function(...) {
  dml_staggered_did(panel$y, xs, panel$id, panel$t, panel$g, ml_g = lrn_ols(), ml_m = lrn_logistic(), ...)
}

test_that("group-time effects and aggregations match Python", {
  for (control in c("never_treated", "not_yet_treated")) {
    fit <- staggered(control_group = control)
    expect_equal(c(unname(coef(fit)), unname(fit$se)), expected[[paste0("staggered_", control)]],
                 tolerance = tol, label = control)
    expect_equal(c(t(as.matrix(fit$group_times))), expected[[paste0("staggered_gt_", control)]], tolerance = 0)
    for (by in c("group", "time", "eventstudy")) {
      agg <- aggregate(fit, by)
      name <- paste("staggered", control, by, sep = "_")
      expect_equal(c(unname(coef(agg$effects)), unname(agg$effects$se), unname(coef(agg$overall)),
                     unname(agg$overall$se)),
                   expected[[name]], tolerance = tol, label = name)
    }
  }
})

test_that("repeated cross-sections, clusters and the experimental score match Python", {
  cs <- dml_staggered_did(panel$ycs, xs, NULL, panel$tcs, panel$gcs, ml_g = lrn_ols(), ml_m = lrn_logistic(),
                          panel = FALSE, n_folds = 3)
  expect_equal(c(unname(coef(cs)), unname(cs$se)), expected$staggered_cs, tolerance = tol)
  clustered <- staggered(cluster = panel$cluster)
  expect_equal(c(unname(coef(clustered)), unname(clustered$se)), expected$staggered_cluster, tolerance = tol)
  experimental <- dml_staggered_did(panel$y, xs, panel$id, panel$t, panel$g, ml_g = lrn_ols(),
                                    score = "experimental", anticipation_periods = 0)
  expect_equal(c(unname(coef(experimental)), unname(experimental$se)), expected$staggered_experimental,
               tolerance = tol)
})

test_that("staggered fits work with the other dmlrs methods", {
  fit <- staggered()
  expect_s3_class(fit, "dmlrs_fit")
  ci <- confint(fit, joint = TRUE, n_boot = 200)
  expect_true(all(ci[, 2] - ci[, 1] > confint(fit)[, 2] - confint(fit)[, 1]))
  expect_equal(nrow(sensitivity_analysis(fit)), length(coef(fit)))
  expect_output(print(aggregate(fit, "time")), "calendar time")
  agg <- aggregate(fit, "eventstudy")
  expect_equal(dim(agg$weights), c(length(coef(agg$effects)), length(coef(fit))))
  expect_equal(unname(drop(agg$weights %*% coef(fit))), unname(coef(agg$effects)), tolerance = 1e-10)
  pdf(NULL)
  on.exit(dev.off())
  expect_silent(plot(fit))
  expect_silent(plot(agg))
  expect_silent(plot(aggregate(fit, "group"), joint = TRUE))
})

test_that("staggered inputs are checked", {
  expect_error(staggered(cluster = seq_len(nrow(panel))), "changes over time")
  expect_error(dml_staggered_did(panel$y, xs, panel$id[-1], panel$t, panel$g), "`id`")
  expect_error(dml_staggered_did(panel$y, xs, panel$id, panel$t, panel$g[-1]), "`g`")
  expect_equal(unname(coef(staggered())),
               unname(coef(dml_staggered_did(panel$y, xs, panel$id, panel$t,
                                             ifelse(is.finite(panel$g), panel$g, NA),
                                             ml_g = lrn_ols(), ml_m = lrn_logistic()))))
  path <- tempfile(fileext = ".rds")
  saveRDS(staggered(), path)
  restored <- readRDS(path)
  expect_length(coef(restored), length(coef(staggered())))
  expect_error(aggregate(restored, "group"), "saved session")
  expect_error(confint(restored), "saved session")
})

rd_x <- as.matrix(rd[c("x1", "x2")])
rd_values <- function(fit) c(unname(coef(fit)), unname(fit$se), c(confint(fit, level = 0.9)), fit$h, fit$b, fit$h_fs)

test_that("flexible RD matches Python", {
  sharp <- dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols())
  expect_equal(rd_values(sharp), expected$rdflex_sharp, tolerance = tol)
  fuzzy <- dml_rdflex(rd$y, rd$d_fuzzy, rd_x, rd$score, ml_g = lrn_ols(), ml_m = lrn_logistic(), fuzzy = TRUE)
  expect_equal(rd_values(fuzzy), expected$rdflex_fuzzy, tolerance = tol)
  options <- dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols(), fs_specification = "cutoff and score",
                        fs_kernel = "uniform", n_iterations = 3, cutoff = 0.1, n_rep = 2, p = 2, kernel = "uniform",
                        vce = "hc1", bwselect = "msetwo", level = 90, covs = rd_x[, 1], cluster = rd$cl)
  expect_equal(rd_values(options), expected$rdflex_options, tolerance = tol)
  binary <- dml_rdflex(rd$y_binary, rd$d_sharp, rd_x, rd$score, ml_g = lrn_logistic(), binary_outcome = TRUE)
  expect_equal(rd_values(binary), expected$rdflex_binary, tolerance = tol)
})

test_that("flexible RD inputs and methods", {
  fit <- dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols())
  expect_output(print(fit), "sharp design")
  expect_output(summary(fit), "Robust")
  expect_equal(nobs(fit), nrow(rd))
  expect_equal(rownames(confint(fit, "Robust")), "Robust")
  expect_error(dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols(), fuzzy = TRUE), "classifier")
  expect_error(dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols(), h = 0.2), "bandwidths")
  expect_error(dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols(), nonsense = 1), "unknown")
  expect_error(dml_rdflex(rd$y, rd$d_sharp, rd_x, rd$score, ml_g = lrn_ols(), kernel = "nope"), "not supported")
  path <- tempfile(fileext = ".rds")
  saveRDS(fit, path)
  expect_error(confint(readRDS(path)), "saved session")
})

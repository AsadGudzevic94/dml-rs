# Reference results of R's contdid (Callaway, Goodman-Bacon and Sant'Anna) for
# dml-rs ContinuousDid without covariates: two periods, never-treated
# comparison group, B-spline dose-response ("parametric").
#
# contdid evaluates the spline on the grid with boundary knots taken from the
# grid's own range (splines2::bSpline(dvals)), while the fit uses the range of
# the treated doses. The two agree only if the grid spans exactly the treated
# doses, so the grids here run from the smallest to the largest dose.
#
# Standard errors: contdid reports multiplier-bootstrap standard errors; the
# fixture stores the analytic ones from its influence functions,
# sqrt(sum(IF^2)) / n. The influence function of the overall ACRT
# (cont_did_acrt) is on the scale of the n1 treated units (the regression's
# bread is computed on them and the controls contribute nothing), so its
# standard error is sqrt(sum(IF^2)) / n1; contdid divides by n and reports a
# standard error that is too small by the factor n1 / n.
#
#   Rscript parity/generate_contdid_fixtures.R
suppressPackageStartupMessages({
  library(contdid)
  library(jsonlite)
})
stopifnot(packageVersion("contdid") == "0.1.2")

simulate <- function(n, seed) {
  set.seed(seed)
  G <- sample(c(0, 2), n, replace = TRUE, prob = c(0.35, 0.65))
  D <- ifelse(G == 2, runif(n, 0.1, 1), 0)
  y1 <- rnorm(n)
  dy <- 0.5 + (G == 2) * (D + D^2) + rnorm(n)
  data.frame(id = rep(1:n, each = 2), period = rep(1:2, n), G = rep(G, each = 2),
             D = rep(D, each = 2), Y = as.numeric(rbind(y1, y1 + dy)))
}

cases <- list(
  list(name = "cubic", degree = 3, num_knots = 0, seed = 1),
  list(name = "quadratic_two_knots", degree = 2, num_knots = 2, seed = 2),
  list(name = "linear_three_knots", degree = 1, num_knots = 3, seed = 3)
)
n <- 1500
out <- list()
for (case in cases) {
  df <- simulate(n, case$seed)
  post <- df[df$period == 2, ]
  pre <- df[df$period == 1, ]
  dy <- post$Y - pre$Y
  dose <- post$D
  dvals <- seq(min(dose[dose > 0]), max(dose[dose > 0]), length.out = 15)
  res <- suppressWarnings(suppressMessages(cont_did(
    yname = "Y", tname = "period", idname = "id", dname = "D", gname = "G", data = df,
    target_parameter = "slope", aggregation = "dose", treatment_type = "continuous",
    dose_est_method = "parametric", dvals = dvals, degree = case$degree,
    num_knots = case$num_knots, control_group = "nevertreated", biters = 100)))
  if_se <- function(m) as.numeric(sqrt(colSums(as.matrix(m)^2)) / n)
  out[[case$name]] <- list(
    y = dy, d = dose, grid = dvals, degree = case$degree, num_knots = case$num_knots,
    att = as.numeric(res$att.d), acrt = as.numeric(res$acrt.d),
    att_se = if_se(res$att.d_inffunc), acrt_se = if_se(res$acrt.d_inffunc),
    overall_att = res$overall_att, overall_acrt = res$overall_acrt,
    overall_att_se = if_se(res$overall_att_inffunc),
    overall_acrt_se = if_se(res$overall_acrt_inffunc) * n / sum(dose > 0),
    overall_acrt_se_contdid = if_se(res$overall_acrt_inffunc)
  )
  cat(sprintf("%-22s overall ATT %.4f (se %.4f)  ACRT %.4f (se %.4f)\n", case$name,
              res$overall_att, if_se(res$overall_att_inffunc), res$overall_acrt,
              if_se(res$overall_acrt_inffunc) * n / sum(dose > 0)))
}
out$`_meta` <- list(contdid = as.character(packageVersion("contdid")),
                    ptetools = as.character(packageVersion("ptetools")), R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE),
           file.path("crates", "dml-core", "tests", "fixtures", "contdid_r.json"))

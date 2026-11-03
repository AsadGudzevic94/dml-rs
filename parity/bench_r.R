# R side of parity/bench_r.py: times grf, policytree, synthdid, causalweight and contdid on the
# data that bench_r.py wrote to `folder`, and writes the Prop 99 matrix and the simulated data
# of the last three for bench_r.py to time dml-rs on.
#   Rscript parity/bench_r.R <folder>
args <- commandArgs(trailingOnly = TRUE)
folder <- args[1]
suppressMessages({
  library(grf)
  library(policytree)
  library(synthdid)
  library(causalweight)
  library(contdid)
})
timeit <- function(expr, reps = 2) {
  best <- Inf
  for (i in seq_len(reps)) best <- min(best, system.time(force(eval.parent(substitute(expr))))[["elapsed"]])
  best
}
out <- c()
for (n in c(2000, 10000, 50000)) {
  d <- as.matrix(read.csv(file.path(folder, sprintf("cf_%d.csv", n)), header = FALSE))
  y <- d[, 1]; w <- d[, 2]; x <- d[, -(1:2)]
  out <- c(out, sprintf("causal_forest n=%d %.3f", n, timeit(causal_forest(x, y, w, num.trees = 2000, seed = 1))))
}
for (n in c(1000, 5000)) {
  d <- as.matrix(read.csv(file.path(folder, sprintf("pt_%d.csv", n)), header = FALSE))
  out <- c(out, sprintf("policy_tree n=%d %.3f", n, timeit(policy_tree(d[, -(1:3)], d[, 1:3], depth = 2))))
}

data("california_prop99")
setup <- panel.matrices(california_prop99)
write.csv(setup$Y, file.path(folder, "prop99_Y.csv"), row.names = FALSE)
writeLines(as.character(c(setup$N0, setup$T0)), file.path(folder, "prop99_dims.txt"))
out <- c(out, sprintf("synthdid estimate %.3f", timeit(synthdid_estimate(setup$Y, setup$N0, setup$T0))))
est <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
out <- c(out, sprintf("synthdid placebo %.3f", timeit(vcov(est, method = "placebo", replications = 200), reps = 1)))

for (n in c(5000, 20000)) {
  set.seed(1)
  x <- matrix(rnorm(n * 4), n, 4)
  d <- as.numeric(x %*% c(0.5, -0.4, 0.3, 0) + rnorm(n) > 0)
  m <- 0.6 * d + 0.5 * x[, 1] + rnorm(n)
  y <- 0.5 * d + m + 0.3 * d * m + x %*% c(0.4, 0.2, -0.3, 0.1) + rnorm(n)
  write.csv(cbind(y = as.numeric(y), d = d, m = m, x), file.path(folder, sprintf("med_%d.csv", n)),
            row.names = FALSE)
  out <- c(out, sprintf("medDML n=%d %.3f", n,
                        timeit(medDML(y = as.numeric(y), d = d, m = m, x = x, MLmethod = "parametric"))))

  G <- sample(c(0, 2), n, replace = TRUE, prob = c(0.35, 0.65))
  D <- ifelse(G == 2, runif(n, 0.1, 1), 0)
  y1 <- rnorm(n)
  dy <- 0.5 + (G == 2) * (D + D^2) + rnorm(n)
  df <- data.frame(id = rep(1:n, each = 2), period = rep(1:2, n), G = rep(G, each = 2),
                   D = rep(D, each = 2), Y = as.numeric(rbind(y1, y1 + dy)))
  write.csv(data.frame(dy = dy, d = D), file.path(folder, sprintf("contdid_%d.csv", n)), row.names = FALSE)
  dvals <- seq(min(D[D > 0]), max(D[D > 0]), length.out = 50)
  out <- c(out, sprintf("cont_did n=%d %.3f", n, timeit(suppressWarnings(suppressMessages(cont_did(
    yname = "Y", tname = "period", idname = "id", dname = "D", gname = "G", data = df,
    target_parameter = "slope", aggregation = "dose", treatment_type = "continuous",
    dose_est_method = "parametric", dvals = dvals, degree = 3, num_knots = 0,
    control_group = "nevertreated", biters = 100))))))
}
writeLines(out, file.path(folder, "r_times.txt"))

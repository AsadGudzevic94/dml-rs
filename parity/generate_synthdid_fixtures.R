# Reference results of R's synthdid (Arkhangelsky, Athey, Hirshberg, Imbens
# and Wager) for dml-rs SyntheticDid: estimates, unit and time weights,
# covariate coefficients, effect curves and standard errors (jackknife;
# placebo and bootstrap after set.seed, which dml-rs reproduces with its port
# of R's generator and sample()).
#
#   Rscript parity/generate_synthdid_fixtures.R
#
# Needs synthdid 0.0.9 (remotes::install_github("synth-inference/synthdid")).
suppressPackageStartupMessages({
  library(synthdid)
  library(jsonlite)
})
stopifnot(packageVersion("synthdid") == "0.0.9")

fns <- list(sdid = synthdid_estimate, sc = sc_estimate, did = did_estimate)
se_or_null <- function(est, ...) {
  v <- tryCatch(synthdid_se(est, ...), error = function(e) NA)
  if (is.na(v)) NULL else as.numeric(v)
}

record <- function(Y, N0, T0, estimator, X = NULL, placebo_seed = NULL, boot_seed = NULL, reps = 50) {
  args <- list(Y = Y, N0 = N0, T0 = T0)
  if (!is.null(X)) args$X <- X
  est <- do.call(fns[[estimator]], args)
  w <- attr(est, "weights")
  out <- list(y = unname(Y), n0 = N0, t0 = T0, estimator = estimator,
              estimate = as.numeric(est), omega = as.numeric(w$omega), lambda = as.numeric(w$lambda),
              beta = if (is.null(w$beta)) numeric(0) else as.numeric(w$beta),
              effect_curve = as.numeric(synthdid_effect_curve(est)),
              se_jackknife = se_or_null(est, method = "jackknife"))
  if (!is.null(X)) out$x <- lapply(seq_len(dim(X)[3]), function(k) unname(X[, , k]))
  if (!is.null(placebo_seed)) {
    set.seed(placebo_seed)
    out$placebo <- list(seed = placebo_seed, replications = reps,
                        se = se_or_null(est, method = "placebo", replications = reps))
  }
  if (!is.null(boot_seed)) {
    set.seed(boot_seed)
    out$bootstrap <- list(seed = boot_seed, replications = reps,
                          se = se_or_null(est, method = "bootstrap", replications = reps))
  }
  out
}

simulate <- function(N0, N1, T0, T1, K = 0, seed = 1) {
  set.seed(seed)
  N <- N0 + N1; T <- T0 + T1
  unit <- rnorm(N); time <- cumsum(rnorm(T, 0.2))
  load <- rnorm(N); factor <- rnorm(T)
  X <- if (K > 0) array(rnorm(N * T * K), c(N, T, K)) else NULL
  Y <- outer(unit, rep(1, T)) + outer(rep(1, N), time) + outer(load, factor) + matrix(rnorm(N * T, sd = 0.5), N, T)
  if (K > 0) for (k in 1:K) Y <- Y + c(0.8, -0.5, 0.3)[k] * X[, , k]
  Y[(N0 + 1):N, (T0 + 1):T] <- Y[(N0 + 1):N, (T0 + 1):T] + 2
  list(Y = Y, X = X, N0 = N0, T0 = T0)
}

data("california_prop99")
prop99 <- panel.matrices(california_prop99)
out <- list()
for (e in names(fns)) {
  out[[paste0("prop99_", e)]] <- record(prop99$Y, prop99$N0, prop99$T0, e, placebo_seed = 12345)
}
sim <- simulate(30, 5, 12, 4, seed = 1)
for (e in names(fns)) {
  out[[paste0("sim_", e)]] <- record(sim$Y, sim$N0, sim$T0, e, placebo_seed = 7, boot_seed = 8)
}
simx <- simulate(25, 4, 10, 3, K = 2, seed = 2)
for (e in names(fns)) {
  out[[paste0("covariates_", e)]] <- record(simx$Y, simx$N0, simx$T0, e, X = simx$X,
                                            placebo_seed = 9, boot_seed = 10, reps = 30)
}
out$`_meta` <- list(synthdid = as.character(packageVersion("synthdid")), R = R.version.string)
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, matrix = "rowmajor", null = "null"),
           file.path("crates", "dml-core", "tests", "fixtures", "synthdid_r.json"))
for (nm in setdiff(names(out), "_meta")) {
  o <- out[[nm]]
  cat(sprintf("%-18s tau %8.4f  jk %s  placebo %s  boot %s\n", nm, o$estimate,
              format(if (is.null(o$se_jackknife)) NA else o$se_jackknife, digits = 4),
              format(if (is.null(o$placebo$se)) NA else o$placebo$se, digits = 4),
              format(if (is.null(o$bootstrap$se)) NA else o$bootstrap$se, digits = 4)))
}

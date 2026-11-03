# The grf side of parity/survival_mc.py: causal_survival_forest on the same datasets.

args <- commandArgs(TRUE)
folder <- args[1]
reps <- as.integer(args[2])
horizon <- as.numeric(args[3])

fit <- function(data, target) {
  forest <- grf::causal_survival_forest(as.matrix(data[, 4:7]), data[, 1], data[, 3], data[, 2],
                                        target = target, horizon = horizon, num.trees = 1000,
                                        seed = 1)
  grf::average_treatment_effect(forest)
}

out <- t(sapply(seq_len(reps) - 1, function(seed) {
  data <- read.csv(file.path(folder, paste0(seed, ".csv")), header = FALSE)
  start <- Sys.time()
  survival <- fit(data, "survival.probability")
  rmst <- fit(data, "RMST")
  seconds <- as.numeric(difftime(Sys.time(), start, units = "secs")) / 2
  c(survival, rmst, seconds)
}))
colnames(out) <- c("survival", "survival_se", "rmst", "rmst_se", "seconds")
write.csv(out, file.path(folder, "grf.csv"), row.names = FALSE)

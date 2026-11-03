# Kaplan-Meier, restricted mean survival time and Aalen-Johansen cumulative incidences from R,
# for crates/dml-core/tests/survival.rs: without covariates and with empirical hazards,
# CausalSurvival must reproduce them.
#
#   Rscript parity/generate_survival_fixtures.R   (needs survival and survRM2)

library(survival)

fmt <- function(v) paste0("[", paste(sprintf("%.17g", v), collapse = ","), "]")

# lung: survival of patients with advanced lung cancer, by sex.
data <- na.omit(survival::lung[c("time", "status", "sex")])
time <- as.numeric(data$time)
event <- as.numeric(data$status == 2)
d <- as.numeric(data$sex == 2)
horizons <- c(100, 200, 365, 500)

km <- function(arm) {
  summary(survfit(Surv(time, event) ~ 1, subset = d == arm), times = horizons, extend = TRUE)
}
treated <- km(1)
control <- km(0)
rmst <- survRM2::rmst2(time, event, d, tau = 500)

lines <- c(
  sprintf("\"versions\":{\"survival\":\"%s\",\"survRM2\":\"%s\"}",
          packageVersion("survival"), packageVersion("survRM2")),
  sprintf("\"time\":%s", fmt(time)),
  sprintf("\"event\":%s", fmt(event)),
  sprintf("\"d\":%s", fmt(d)),
  sprintf("\"horizons\":%s", fmt(horizons)),
  sprintf("\"km_treated\":%s", fmt(treated$surv)),
  sprintf("\"km_control\":%s", fmt(control$surv)),
  sprintf("\"km_treated_se\":%s", fmt(treated$std.err)),
  sprintf("\"km_control_se\":%s", fmt(control$std.err)),
  sprintf("\"rmst_tau\":%s", fmt(500)),
  sprintf("\"rmst_treated\":%s", fmt(rmst$RMST.arm1$rmst[["Est."]])),
  sprintf("\"rmst_control\":%s", fmt(rmst$RMST.arm0$rmst[["Est."]])),
  sprintf("\"rmst_treated_se\":%s", fmt(rmst$RMST.arm1$rmst[["se"]])),
  sprintf("\"rmst_control_se\":%s", fmt(rmst$RMST.arm0$rmst[["se"]]))
)
writeLines(paste0("{", paste(lines, collapse = ","), "}"),
           "crates/dml-core/tests/fixtures/survival_kaplan_meier.json")

# mgus2: progression to a plasma cell malignancy (cause 1) competing with death (cause 2), by sex.
mg <- survival::mgus2
mg_time <- as.numeric(with(mg, ifelse(pstat == 0, futime, ptime)))
mg_event <- as.numeric(with(mg, ifelse(pstat == 0, 2 * death, 1)))
mg_d <- as.numeric(mg$sex == "M")
mg_horizons <- c(60, 120, 240)
aj <- function(arm) {
  fit <- survfit(Surv(mg_time, factor(mg_event)) ~ 1, subset = mg_d == arm)
  summary(fit, times = mg_horizons, extend = TRUE)$pstate
}
aj_treated <- aj(1)
aj_control <- aj(0)
lines <- c(
  sprintf("\"versions\":{\"survival\":\"%s\"}", packageVersion("survival")),
  sprintf("\"time\":%s", fmt(mg_time)),
  sprintf("\"event\":%s", fmt(mg_event)),
  sprintf("\"d\":%s", fmt(mg_d)),
  sprintf("\"horizons\":%s", fmt(mg_horizons)),
  sprintf("\"survival_treated\":%s", fmt(aj_treated[, 1])),
  sprintf("\"cif1_treated\":%s", fmt(aj_treated[, 2])),
  sprintf("\"cif2_treated\":%s", fmt(aj_treated[, 3])),
  sprintf("\"survival_control\":%s", fmt(aj_control[, 1])),
  sprintf("\"cif1_control\":%s", fmt(aj_control[, 2])),
  sprintf("\"cif2_control\":%s", fmt(aj_control[, 3]))
)
writeLines(paste0("{", paste(lines, collapse = ","), "}"),
           "crates/dml-core/tests/fixtures/survival_aalen_johansen.json")

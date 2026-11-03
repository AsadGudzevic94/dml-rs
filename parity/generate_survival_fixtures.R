# Kaplan-Meier and restricted mean survival time from R, for crates/dml-core/tests/survival.rs:
# without covariates and with empirical hazards, CausalSurvival must reproduce them.
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

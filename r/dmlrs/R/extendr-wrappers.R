# Calls into the Rust library (src/rust/src/lib.rs). Internal: the exported
# functions in models.R and methods.R validate their arguments first.

#' @useDynLib dmlrs, .registration = TRUE
NULL

rs_fit_plr <- function(y, d, x, names, ml_l, ml_m, ml_g, score, n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_plr, y, d, x, names, ml_l, ml_m, ml_g, score, n_folds, n_rep, seed_)
}

rs_fit_irm <- function(y, d, x, names, ml_g, ml_m, score, clipping_threshold, normalize_ipw,
                       n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_irm, y, d, x, names, ml_g, ml_m, score, clipping_threshold, normalize_ipw,
        n_folds, n_rep, seed_)
}

rs_fit_pliv <- function(y, d, x, z, names, ml_l, ml_m, ml_r, n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_pliv, y, d, x, z, names, ml_l, ml_m, ml_r, n_folds, n_rep, seed_)
}

rs_fit_iivm <- function(y, d, x, z, names, ml_g, ml_m, ml_r, always_takers, never_takers,
                        n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_iivm, y, d, x, z, names, ml_g, ml_m, ml_r, always_takers, never_takers,
        n_folds, n_rep, seed_)
}

rs_fit_did <- function(y, d, x, t, names, ml_g, ml_m, score, in_sample_normalization,
                       n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_did, y, d, x, t, names, ml_g, ml_m, score, in_sample_normalization,
        n_folds, n_rep, seed_)
}

rs_summary <- function(fit) .Call(wrap__rs_summary, fit)

rs_format <- function(fit) .Call(wrap__rs_format, fit)

rs_confint <- function(fit, level) .Call(wrap__rs_confint, fit, level)

rs_joint_confint <- function(fit, method, n_boot, seed_, level) {
  .Call(wrap__rs_joint_confint, fit, method, n_boot, seed_, level)
}

rs_sensitivity <- function(fit, cf_y, cf_d, rho, level, null_hypothesis) {
  .Call(wrap__rs_sensitivity, fit, cf_y, cf_d, rho, level, null_hypothesis)
}

rs_is_valid <- function(ptr) .Call(wrap__rs_is_valid, ptr)

rs_init <- function() .Call(wrap__rs_init)

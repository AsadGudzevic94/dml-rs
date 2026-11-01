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

rs_fit_apos <- function(y, d, x, names, ml_g, ml_m, treatment_levels, clipping_threshold,
                        normalize_ipw, n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_apos, y, d, x, names, ml_g, ml_m, treatment_levels, clipping_threshold,
        normalize_ipw, n_folds, n_rep, seed_)
}

rs_fit_mediation <- function(y, d, m, x, names, ml_g, ml_m, ml_nested, trimming_rule,
                             trimming_threshold, normalize_ipw, nested_split, debias_nested,
                             n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_mediation, y, d, m, x, names, ml_g, ml_m, ml_nested, trimming_rule,
        trimming_threshold, normalize_ipw, nested_split, debias_nested, n_folds, n_rep, seed_)
}

rs_fit_contdid <- function(y, d, x, ml_g, ml_m, ml_nested, degree, num_knots, grid,
                           trimming_threshold, n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_contdid, y, d, x, ml_g, ml_m, ml_nested, degree, num_knots, grid,
        trimming_threshold, n_folds, n_rep, seed_)
}

rs_fit_synthdid <- function(y, n0, t0, x, estimator, noise_level, eta_omega, eta_lambda,
                            min_decrease, max_iter, sparsify) {
  .Call(wrap__rs_fit_synthdid, y, n0, t0, x, estimator, noise_level, eta_omega, eta_lambda,
        min_decrease, max_iter, sparsify)
}

rs_synth_info <- function(fit) .Call(wrap__rs_synth_info, fit)

rs_synth_se <- function(fit, method, replications, seed_) {
  .Call(wrap__rs_synth_se, fit, method, replications, seed_)
}

rs_policy_tree <- function(x, gamma, depth, split_step, min_node_size) {
  .Call(wrap__rs_policy_tree, x, gamma, depth, split_step, min_node_size)
}

rs_policy_value <- function(gamma, actions, names) .Call(wrap__rs_policy_value, gamma, actions, names)

rs_lee_bounds <- function(y, d, s, strata, n_boot, level, seed_) {
  .Call(wrap__rs_lee_bounds, y, d, s, strata, n_boot, level, seed_)
}

rs_linear_combination <- function(fit, weights, names) {
  .Call(wrap__rs_linear_combination, fit, weights, names)
}

rs_dr_scores <- function(fit) .Call(wrap__rs_dr_scores, fit)

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

rs_is_valid_synth <- function(ptr) .Call(wrap__rs_is_valid_synth, ptr)

rs_init <- function() .Call(wrap__rs_init)

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
                           clipping_threshold, n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_contdid, y, d, x, ml_g, ml_m, ml_nested, degree, num_knots, grid,
        clipping_threshold, n_folds, n_rep, seed_)
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

rs_init <- function(threads) .Call(wrap__rs_init, threads)

# Heterogeneous effects (src/rust/src/heterogeneity.rs)

rs_cate <- function(fit, basis) .Call(wrap__rs_cate, fit, basis)

rs_gate <- function(fit, groups) .Call(wrap__rs_gate, fit, groups)

rs_blp_info <- function(blp) .Call(wrap__rs_blp_info, blp)

rs_blp_confint <- function(blp, level) .Call(wrap__rs_blp_confint, blp, level)

rs_blp_predict <- function(blp, basis, interval, level, n_boot, seed_) {
  .Call(wrap__rs_blp_predict, blp, basis, interval, level, n_boot, seed_)
}

rs_fit_cate_learner <- function(y, d, x, method, ml_outcome, ml_propensity, ml_effect, n_folds,
                                seed_, clipping_threshold) {
  .Call(wrap__rs_fit_cate_learner, y, d, x, method, ml_outcome, ml_propensity, ml_effect, n_folds,
        seed_, clipping_threshold)
}

rs_cate_effect <- function(fit, x) .Call(wrap__rs_cate_effect, fit, x)

rs_cate_pseudo_outcomes <- function(fit) .Call(wrap__rs_cate_pseudo_outcomes, fit)

rs_cate_learner_blp <- function(fit, basis) .Call(wrap__rs_cate_learner_blp, fit, basis)

rs_cate_learner_gate <- function(fit, groups) .Call(wrap__rs_cate_learner_gate, fit, groups)

rs_fit_causal_forest <- function(y, d, x, ml_y, ml_d, d_classifier, forest, clipping_threshold,
                                 n_folds, seed_) {
  .Call(wrap__rs_fit_causal_forest, y, d, x, ml_y, ml_d, d_classifier, forest, clipping_threshold,
        n_folds, seed_)
}

rs_fit_instrumental_forest <- function(y, d, x, z, ml_y, ml_d, ml_z, d_classifier, z_classifier,
                                       forest, reduced_form_weight, clipping_threshold, n_folds,
                                       seed_) {
  .Call(wrap__rs_fit_instrumental_forest, y, d, x, z, ml_y, ml_d, ml_z, d_classifier, z_classifier,
        forest, reduced_form_weight, clipping_threshold, n_folds, seed_)
}

rs_forest_info <- function(fit) .Call(wrap__rs_forest_info, fit)

rs_forest_effect <- function(fit, x) .Call(wrap__rs_forest_effect, fit, x)

rs_forest_dr_scores <- function(fit) .Call(wrap__rs_forest_dr_scores, fit)

rs_forest_ate <- function(fit) .Call(wrap__rs_forest_ate, fit)

rs_forest_blp <- function(fit, basis) .Call(wrap__rs_forest_blp, fit, basis)

rs_forest_gate <- function(fit, groups) .Call(wrap__rs_forest_gate, fit, groups)

rs_fit_conformal <- function(y, d, x, ml_y, ml_lower, ml_upper, ml_m, propensity, train_fraction,
                             clipping_threshold, seed_) {
  .Call(wrap__rs_fit_conformal, y, d, x, ml_y, ml_lower, ml_upper, ml_m, propensity,
        train_fraction, clipping_threshold, seed_)
}

rs_conformal_predict <- function(fit, x, what, target, alpha) {
  .Call(wrap__rs_conformal_predict, fit, x, what, target, alpha)
}

rs_conformal_n_calibration <- function(fit) .Call(wrap__rs_conformal_n_calibration, fit)

rs_fit_dose_response <- function(y, d, x, ml_g, ml_m, grid, kernel, bandwidth, bandwidth_scale,
                                 density, density_bandwidth, density_floor, partial_effects, eta,
                                 n_folds, n_rep, seed_) {
  .Call(wrap__rs_fit_dose_response, y, d, x, ml_g, ml_m, grid, kernel, bandwidth, bandwidth_scale,
        density, density_bandwidth, density_floor, partial_effects, eta, n_folds, n_rep, seed_)
}

rs_dose_contrast <- function(fit, reference) .Call(wrap__rs_dose_contrast, fit, reference)

rs_is_valid_kind <- function(ptr, kind) .Call(wrap__rs_is_valid_kind, ptr, kind)

rs_fit_staggered <- function(y, id, t, g, x, ml_g, ml_m, control_group, anticipation_periods,
                             gt_combinations, score, in_sample_normalization, clipping_threshold,
                             n_folds, n_rep, seed_, panel, cluster) {
  .Call(wrap__rs_fit_staggered, y, id, t, g, x, ml_g, ml_m, control_group, anticipation_periods,
        gt_combinations, score, in_sample_normalization, clipping_threshold, n_folds, n_rep, seed_,
        panel, cluster)
}

rs_staggered_att <- function(fit) .Call(wrap__rs_staggered_att, fit)

rs_staggered_group_times <- function(fit) .Call(wrap__rs_staggered_group_times, fit)

rs_staggered_aggregate <- function(fit, how) .Call(wrap__rs_staggered_aggregate, fit, how)

rs_fit_rdflex <- function(y, d, x, score, names, ml_g, ml_m, binary_outcome, cutoff, h_fs,
                          fs_specification, fs_kernel, n_iterations, n_folds, n_rep, seed_, options,
                          rd_covs, rd_cluster) {
  .Call(wrap__rs_fit_rdflex, y, d, x, score, names, ml_g, ml_m, binary_outcome, cutoff, h_fs,
        fs_specification, fs_kernel, n_iterations, n_folds, n_rep, seed_, options, rd_covs, rd_cluster)
}

rs_fit_survival <- function(time, d, x, event, ml_event, ml_censor, ml_m, horizons, estimand, grid,
                            time_encoding, clipping_threshold, censoring_floor, n_folds, n_rep,
                            seed_, cause, cluster) {
  .Call(wrap__rs_fit_survival, time, d, x, event, ml_event, ml_censor, ml_m, horizons, estimand,
        grid, time_encoding, clipping_threshold, censoring_floor, n_folds, n_rep, seed_, cause,
        cluster)
}

rs_rdflex_summary <- function(fit) .Call(wrap__rs_rdflex_summary, fit)

rs_rdflex_confint <- function(fit, level) .Call(wrap__rs_rdflex_confint, fit, level)

# Confidence sequences and prediction-powered inference (src/rust/src/sequential.rs)

rs_confidence_sequence <- function(fit, level, opt_time_) {
  .Call(wrap__rs_confidence_sequence, fit, level, opt_time_)
}

rs_monitor <- function(fits, level, opt_time_) .Call(wrap__rs_monitor, fits, level, opt_time_)

rs_fit_ppi <- function(estimand, y, yhat, yhat_unlabeled, x, x_unlabeled, names, lambda, learner,
                       use_classifier, features, features_unlabeled, n_folds, seed_) {
  .Call(wrap__rs_fit_ppi, estimand, y, yhat, yhat_unlabeled, x, x_unlabeled, names, lambda, learner,
        use_classifier, features, features_unlabeled, n_folds, seed_)
}

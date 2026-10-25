# Validation

dml-rs is tested against the reference implementation, [DoubleML](https://docs.doubleml.org)
0.11.4, on identical data and identical sample splits.

## Parity tests

[`parity/generate_fixtures.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_fixtures.py)
runs DoubleML on fixed data and fixed folds and stores data, folds and results as JSON. The Rust
test suite refits every case on the same folds and requires agreement to `1e-9` for closed-form
learners (OLS, ridge) and `1e-7` for iterative ones (Lasso, logistic regression, cross-validated
learners). Compared are the nuisance predictions, coefficients, standard errors, t-statistics,
p-values (raw and multiplicity-adjusted), confidence intervals, bootstrap bands, BLP and GATE
inference, staggered-DiD aggregations, sensitivity bounds and robustness values, and learner
losses.

| Fixtures | Coverage |
|---|---|
| `plr_ols`, `plr_ridge`, `plr_lasso`, `plr_ivtype_rep3` | PLR with partialling-out and IV-type scores, several learners, repeated splits |
| `plr_lasso_cv`, `irm_ate_lasso_cv_logit_cv` | scikit-learn's `LassoCV` and `LogisticRegressionCV` as nuisance learners |
| `plr_boot_bayes`, `plr_boot_wild`, `plr_multi_rep3_boot` | Bootstrap methods, multiple treatments, joint intervals, Romano–Wolf |
| `lplr_*` | Logistic partially linear regression: both scores, continuous and binary treatments, replaying DoubleML's outer and inner splits |
| `irm_ate`, `irm_atte`, `irm_ate_normalized`, `irm_cate_gate` | IRM (ATE / ATTE, normalised IPW), BLP and GATE |
| `apo_*`, `apos_rep2` | Average potential outcomes, causal contrasts and their sensitivity bounds |
| `pq_*`, `lpq_t1`, `cvar_*`, `qte_rep2` | Potential, local potential and CVaR quantiles for both treatment levels, with and without normalised weights, and quantile treatment effects |
| `ssm_*` | Sample selection, missing at random and nonignorable, with and without normalised weights |
| `pliv_ols`, `iivm_late_rep2`, `iivm_no_always_takers` | PLIV and IIVM, including one-sided non-compliance |
| `did_panel_*`, `did_cs_*` | Two-period DiD for panels and cross-sections, all score and normalisation variants |
| `staggered_*`, `staggered_cs_*` | Staggered DiD: every ATT(g, t) and all aggregations |
| `plr_cluster_*`, `irm_cluster_one_way`, `pliv_cluster_two_way` | One-way and two-way clustering |
| all PLR, IRM and DiD fixtures | Sensitivity analysis in two scenarios, and benchmarks |

The Python test suite checks the bindings against DoubleML and scikit-learn (including the
learners themselves), and the type stubs are checked against the compiled extension with
`mypy.stubtest`.

## Statistical tests

Monte-Carlo tests simulate data with a known effect and check that confidence intervals reach
their nominal coverage: PLR, LPLR (both scores), IRM (ATE and ATTE), PLIV, IIVM, DiD, sample selection (both scores), quantile and CVaR effects, joint bootstrap bands, and
clustered data (where iid intervals cover about 75% of the time and cluster-robust ones about 95%).

## Known differences

These are deliberate, documented, and covered by tests:

- **IIVM with `normalize_ipw=True`.** dml-rs normalises the inverse-probability weights with the
  instrument; DoubleML 0.11 used the treatment. The DoubleML maintainers confirmed the instrument
  is correct ([issue #411](https://github.com/DoubleML/doubleml-for-py/issues/411)).
- **Contrasts of clustered fits** (`APOS.causal_contrast` and staggered-DiD aggregations) use the
  cluster-robust variance of the combined score. DoubleML's framework arithmetic scales the score
  by the number of clusters but ignores the correlation within clusters. In a simulation with
  treatment assigned per cluster, cluster-robust contrast intervals cover the true effect 95% of
  the time and iid intervals 62%.
- **Robustness values** are solved exactly; DoubleML uses a bounded numerical minimiser, so the two
  agree to about `1e-5`.
- **Gradient boosting** breaks ties between equally good splits deterministically, scikit-learn by
  a random feature order. Training-set predictions are identical; held-out predictions and
  estimates agree up to scikit-learn's own seed-to-seed variation.
- **Lasso with more covariates than observations.** At scikit-learn's default tolerance
  (`tol=1e-4`) the coordinate descent may stop at slightly different points, depending on
  floating-point rounding (as scikit-learn does across platforms); both are within the tolerance.
  With a tighter tolerance the two agree to numerical precision.
- **Quantile models need bit-identical propensities for exact agreement.** The preliminary
  estimate minimises the absolute value of a step function with Brent's method (as DoubleML), and
  that search reacts to differences in the last bit of the propensity scores. With identical
  learners (for example the same scikit-learn estimator in Python) dml-rs and DoubleML agree to
  `1e-9`. With learners that differ by rounding, such as the built-in `LogisticRegression` against
  scikit-learn's, the search occasionally ends at a neighbouring point: in 57 potential-quantile
  fits on simulated data, 54 agreed to `1e-9` and the other three differed by at most 0.08
  standard errors. The Rust parity fixtures therefore use class-share propensities, which are exact
  on both sides.
- **LPLR with `n_folds_inner` different from `n_folds`.** The outer predictions of `ml_M` and
  `ml_a` average the inner models. DoubleML 0.11 sums the inner models' predictions and divides by
  the number of outer folds, which scales them by `n_folds_inner / n_folds` when the two differ
  (with 4 outer and 2 inner folds its `ml_M` predictions average half the outcome mean) and changes
  the estimate through `r̂ = t̂ − β·â`. With the default, equal fold counts the two agree to `1e-9`.
- **`LogisticRegressionCV`** solves each fit to the exact optimum with Newton's method;
  scikit-learn's default L-BFGS solver stops earlier, so its probabilities can differ by about
  `1e-3` at default settings.

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
| `rd_*`, `rdflex_*` | The RD back end against the `rdrobust` package (bandwidths and estimates, sharp, fuzzy and with mass points); RDFlex in sharp and fuzzy designs with every first-stage specification and kernel, a non-zero cutoff, one to three bandwidth rounds and repeated splits |
| `rd_options`, `rdflex_*_rd_options`, `rdflex_fuzzy_covs_cluster`, `rdflex_*binary*` | 54 `rdrobust` option sets (polynomial orders, kernels, all ten bandwidth selectors, HC variances, mass points, regularisation, `rho`, covariates with a collinear column, clusters with the `cr1`, `cr2` and `cr3` variances for `h = b` and `h ≠ b`, manual bandwidths) and RDFlex with `rdrobust` options, RD covariates, clusters and a classifier for a binary outcome |
| `*_isotonic*` | Isotonic propensity calibration, on all observations and cross-fitted: IRM (ATE, ATTE with sensitivity), APO, IIVM, SSM and staggered DiD |
| `plpr_*` | Panel regression: all four approaches, unbalanced panels with gaps, IV-type score, DoubleML's unit-clustered splits |
| `lplr_*` | Logistic partially linear regression: both scores, continuous and binary treatments, replaying DoubleML's outer and inner splits |
| `irm_ate`, `irm_atte`, `irm_ate_normalized`, `irm_cate_gate` | IRM (ATE / ATTE, normalised IPW), BLP and GATE |
| `apo_*`, `apos_rep2` | Average potential outcomes, causal contrasts and their sensitivity bounds |
| `pq_*`, `lpq_t1`, `cvar_*`, `qte_rep2` | Potential, local potential and CVaR quantiles for both treatment levels, with and without normalised weights, and quantile treatment effects |
| `ssm_*` | Sample selection, missing at random and nonignorable, with and without normalised weights |
| `pliv_ols`, `pliv_ivtype`, `pliv_*_instruments*`, `pliv_partial_z`, `pliv_partial_xz`, `iivm_late_rep2`, `iivm_no_always_takers` | PLIV and IIVM, including one-sided non-compliance |
| `did_panel_*`, `did_cs_*` | Two-period DiD for panels and cross-sections, all score and normalisation variants |
| `staggered_*`, `staggered_cs_*` | Staggered DiD: every ATT(g, t) and all aggregations, with their sensitivity bounds and (panels) benchmarks |
| `framework_arithmetic_rep2` | Framework arithmetic: sums, differences, multiples and `concat` of a PLR and an IRM fit, with sensitivity bounds |
| `plr_cluster_*`, `irm_cluster_one_way`, `pliv_cluster_two_way` | One-way and two-way clustering |
| all PLR, IRM and DiD fixtures | Sensitivity analysis in two scenarios, and benchmarks |

The Python tests also compare DoubleML's `tune` (grid and randomized search, on the whole sample
and on the folds) and `tune_ml_models` (Optuna) with the same random state: the chosen parameters
and the resulting estimates agree for PLR (both scores), IRM, APO, IIVM, DID, DIDCS, SSM and PLIV.
So do `set_ml_nuisance_params` (including fold- and treatment-specific parameters), the learners
stored with `store_models`, and callable scores for PLR, IRM, PLIV and IIVM.

The simulated data sets (`dml_rs.datasets`) draw from another random generator than DoubleML, so
their tests compare output formats and, on samples of 20,000, the means and standard deviations of
the outcomes, treatments, instruments and covariates with DoubleML's generators, as well as the
oracle confounding strengths; the deterministic effect functions agree exactly. The Rust tests
check that every design's effect is recovered by the matching estimator.

`fetch_401K` and `fetch_bonus` return the same DataFrames as DoubleML's (the bonus preparation is
compared on a synthetic raw file, the 401(k) data with a download).

The histogram gradient boosting learners reproduce scikit-learn's `HistGradientBoosting*`
exactly: with the same `random_state` the number of trees, the early-stopping scores and every
prediction are identical (also with missing values, sample weights, more than 200 000 rows, where
the bins come from a random subsample, and early stopping on a validation split), and DML fits
with them equal fits with scikit-learn's learners to `1e-12`.

The Python test suite checks the bindings against DoubleML and scikit-learn (including the
learners themselves), and the type stubs are checked against the compiled extension with
`mypy.stubtest`.

## Statistical tests

Clustered staggered DiD has no DoubleML counterpart: its tests check that singleton clusters
reproduce the iid standard errors exactly and that cluster-robust intervals reach their coverage
when shocks are shared within clusters.

`AutoDML` has no DoubleML counterpart. Its tests compare it with independent computations of
the same estimator: with an unpenalised Riesz representer against the closed form
$\hat\rho = \hat G^{-1}\hat M$, and with a penalty against scikit-learn's `Lasso` (the Riesz
loss equals a least-squares loss on a transformed outcome), both to `1e-8`; a saturated dictionary
reproduces the inverse propensity weights exactly. Simulations check that it recovers average
derivatives, shift and policy effects and the ATE (where it agrees with IRM), that the Riesz
correction removes the bias of a misspecified regression, and that its confidence intervals cover.

Monte-Carlo tests simulate data with a known effect and check that confidence intervals reach
their nominal coverage: PLR, LPLR (both scores), PLPR (three approaches, with correlated unit effects), RDFlex (sharp and fuzzy), IRM (ATE and ATTE), PLIV, IIVM, DiD, sample selection (both scores), quantile and CVaR effects, joint bootstrap bands, and
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
  the estimate through `r̂ = t̂ − β·â`. This was reported in
  [DoubleML #401](https://github.com/DoubleML/doubleml-for-py/issues/401) and is fixed on its main
  branch ([#402](https://github.com/DoubleML/doubleml-for-py/pull/402)), not yet in a release;
  dml-rs follows the fixed behaviour. With the default, equal fold counts the two agree to `1e-9`.
- **Sensitivity contour plots.** `sensitivity_contours` gives the same grid of bounds as DoubleML's
  `sensitivity_plot` (to `1e-9`), with rows indexed by `cf_d`. DoubleML 0.11 hands that grid to
  plotly as `z` with `x = cf_d` and `y = cf_y`, but plotly reads the rows of `z` along `y`, so its
  contours are drawn transposed; `sensitivity_plot` draws them with `cf_d` on the x axis as labelled.
  The bound is not symmetric in `cf_y` and `cf_d`, so the two pictures differ slightly (reported as
  [DoubleML #414](https://github.com/DoubleML/doubleml-for-py/issues/414)).
- **Negative multiples of a framework.** dml-rs reports `|c|·se` for `c · framework`. DoubleML 0.11
  multiplies the standard errors by `c` itself, so a negative multiple gets a negative standard
  error, a t-statistic with the wrong sign and a confidence interval with its bounds swapped. The
  estimates and sensitivity bounds agree.
- **Sensitivity benchmarks of a staggered DiD.** dml-rs refits every ATT(g, t) without the
  benchmarking covariates. DoubleML 0.11's `DoubleMLDIDMulti.sensitivity_benchmark` changes the
  covariate list of a copy whose group-time models have already read their covariates, so its
  short model equals the long one and every benchmark is 0. The fixtures compare with DoubleML's
  benchmark statistics computed from a short model fitted on data without the covariates, on the
  same splits; they agree to `1e-7`.
- **PQ with external propensities only.** A supplied `ml_m` also serves the preliminary estimate,
  on the rows of the first half of each training fold. DoubleML 0.11 indexes the external
  propensities by position within that half instead of by row, so it pairs them with the wrong
  observations. With external `ml_g` (with or without `ml_m`) the two agree to `1e-9`.
- **σ² and ν² of combinations.** For a multiple of a single estimate dml-rs keeps the
  sensitivity elements (with `ν²` scaled by the squared weight), so benchmarks remain possible;
  DoubleML drops them once frameworks are added, even when all but one weight is zero. The bounds
  are the same.
- **Staggered DiD data with anticipation.** `make_did_CS2021(anticipation_periods > 0,
  time_type="float")` in DoubleML 0.11 also drops the never-treated units (its first filter tests
  `pd.isna`, which is false for the `inf` cohort; the datetime version keeps them). dml-rs keeps
  them in both versions.
- **Fold-specific tuning with repeated splits.** DoubleML 0.11's `tune(tune_on_folds=True)` stores
  its results in a list indexed `[repetition][treatment]` that has one entry per treatment, so it
  fails whenever there are more repetitions than treatments. dml-rs tunes every repetition; the
  tests compare with DoubleML for one repetition.
- **Optuna tuning of IV-type PLIV.** DoubleML 0.11's `tune_ml_models` for `DoubleMLPLIV` with
  `score="IV-type"` calls `cross_val_predict` for `ml_r` without the treatment as target and fails;
  dml-rs uses the treatment, as its grid search does.
- **Initial estimate for tuning `ml_g`** (PLR and PLIV, IV-type). With a classifier for `ml_m`,
  dml-rs uses predicted probabilities in the partialling-out estimate that defines `ml_g`'s target;
  DoubleML's grid search uses predicted classes. With regressors the two agree.
- **`LogisticRegressionCV`** solves each fit to the exact optimum with Newton's method;
  scikit-learn's default L-BFGS solver stops earlier, so its probabilities can differ by about
  `1e-3` at default settings.

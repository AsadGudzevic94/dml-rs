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

Prediction-powered inference is compared with the reference implementation `ppi_py` 0.2.3
([`parity/generate_ppi_fixtures.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_ppi_fixtures.py)):
14 cases (means, OLS and logistic regression with tuned, fixed and zero $\lambda$, and the
cross-prediction versions) agree in the estimate, standard error and tuned $\lambda$ to `1e-9`,
the logistic ones to `1e-7`: `ppi_py`'s L-BFGS-B stops with gradients of about `5e-10`, while
Newton's method reaches `1e-17`.

The Python test suite checks the bindings against DoubleML and scikit-learn (including the
learners themselves), and the type stubs are checked against the compiled extension with
`mypy.stubtest`.

## Statistical tests

The confidence sequences ([Inference](user-guide/inference.md#anytime-valid-inference-confidence-sequences))
have no DoubleML counterpart. Their tests check the boundary against its closed form and the
anytime p-values against it, and simulate streams looked at many times: for means (500 streams of
2000 draws, 99 looks) and for PLR fits (150 streams, 20 looks) the sequence misses at some look in
at most 6% and 8% of the streams, the repeated 95% intervals in at least 20% and 15%. A larger
study for a doubly robust ATE ([`parity/anytime_mc.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/anytime_mc.py);
IRM, 300 streams, 20 looks from 250 to 5000 units):

| True ATE | Misses the ATE at some look: sequence | repeated intervals | Excludes 0 by n = 5000: sequence |
|--:|--:|--:|--:|
| 0 | 0.7% | 22.3% | |
| 0.15 | 0.7% | 24.0% | 96.0% |

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

`DoseResponse` (continuous treatments) has no DoubleML counterpart. Its tests compare it with an
independent NumPy / scikit-learn implementation of Colangelo and Lee's estimator on the same folds
(both density models, linear models and random forests) to `1e-10`, and check the partial effects
against the difference quotient of the curve. A Monte Carlo replication of Colangelo and Lee's
simulation design (2023, Table 1: 100 correlated covariates, $n = 1000$, five-fold cross-fitting,
Epanechnikov kernel, kernel-regression density, bandwidth $h = c\,\sigma_T n^{-1/5}$) with
`LassoCV` for both nuisances
([`parity/dose_response_mc.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/dose_response_mc.py),
500 replications) gives, next to their Lasso results (1000 replications):

| Target | $c$ | Bias: dml-rs / paper | RMSE: dml-rs / paper | Coverage: dml-rs / paper |
|---|--:|--:|--:|--:|
| $\beta(0) = 0$ | 0.75 | −0.019 / 0.001 | 0.122 / 0.111 | 0.976 / 0.936 |
|  | 1.00 | −0.018 / 0.008 | 0.105 / 0.100 | 0.978 / 0.936 |
|  | 1.25 | −0.020 / 0.016 | 0.097 / 0.094 | 0.974 / 0.936 |
|  | 1.50 | −0.024 / 0.025 | 0.093 / 0.092 | 0.970 / 0.925 |
| $\theta(0) = 1.2$ | 0.75 | −0.010 / 0.098 | 1.109 / 0.892 | 0.976 / 0.977 |
|  | 1.00 | 0.039 / 0.084 | 0.719 / 0.588 | 0.966 / 0.980 |
|  | 1.25 | 0.057 / 0.077 | 0.530 / 0.434 | 0.964 / 0.979 |
|  | 1.50 | 0.051 / 0.077 | 0.398 / 0.335 | 0.972 / 0.982 |

The paper does not specify its Lasso tuning or the step $\eta$ of the partial effect, so the
numbers are not expected to agree exactly: the dose-response estimates have about the same RMSE,
the partial effects (with dml-rs's default $\eta = h\,n^{-1/6}$) somewhat larger, and the
intervals of both cover at or above the nominal 95%.

`Mediation` reproduces R's `causalweight::medDML` (version 1.1.6, `multmed = TRUE`), the
reference implementation of Farbmacher et al. (2022) by two of its authors. With
`MLmethod = "parametric"` every nuisance is an `lm` or logit `glm` fit, which dml-rs reproduces
with `Ols` and an unpenalised `LogisticRegression`, and medDML's trimming with
`trimming_rule="drop"`; medDML's folds are recomputed from its seed and passed to
`fit_with_folds`. Five configurations (one and two mediators, with and without
weight normalisation, nested split and debiased nested means, and a trimming case that drops 262
of 900 observations), generated by
[`parity/generate_mediation_fixtures.R`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_mediation_fixtures.R),
match in all six estimates and standard errors to `1e-9` (the differences, about `2e-10`, come
from glm's IRLS stopping rule), with the same number of trimmed observations. An independent
NumPy implementation with random forests on the same folds agrees to `1e-10` for both trimming
rules.

One deliberate difference: dml-rs clips the propensity scores by default instead of dropping
observations, because medDML's rule drops by $P(D = 1 \mid M, X)$, a function of the
post-treatment mediator, which biases all effects when the mediator predicts the treatment (see
the warning in the [user guide](user-guide/models.md#causal-mediation-direct-and-indirect-effects)).

`SurrogateIndex` has no reference implementation in DoubleML, R or Python packages. Its tests
compare it with an independent NumPy / scikit-learn implementation of the Chen and Ritzwoller
(2023) estimator on the same folds (random forests, learned and known propensity) to `1e-10`, and
with Athey et al.'s plug-in estimator in a randomised experiment, where both are consistent and
agree within half a standard error.

A simulation with experimental assignment confounded by a covariate (two surrogates, true
long-term effect 2.5, 1000 experimental and 2000 observational units,
[`parity/surrogate_mc.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/surrogate_mc.py))
gives:

| Estimator | Replications | Bias | RMSE | Coverage of 95% intervals |
|---|--:|--:|--:|--:|
| `SurrogateIndex`, OLS / logit | 500 | −0.009 | 0.156 | 0.970 |
| `SurrogateIndex`, random forests | 100 | −0.034 | 0.218 | 0.980 |
| Plug-in surrogate index (difference of arm means) | 500 | 0.823 | 0.838 | — |

The plug-in ignores that assignment depends on the covariate; the efficient score adjusts for
it.

`ContinuousDID` reproduces R's `contdid` (version 0.1.2, Callaway, Goodman-Bacon and Sant'Anna's
package) without covariates: in three spline configurations (cubic; quadratic with two interior
knots; linear with three) the dose-response $ATT(d)$, the average causal response $ACRT(d)$, the
overall ATT and ACRT and the analytic standard errors from contdid's influence functions agree to
`1e-9`
([`parity/generate_contdid_fixtures.R`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_contdid_fixtures.R)).
contdid has no covariates; the doubly robust estimator with covariates is checked against a NumPy
implementation on the same folds (random forests) to `1e-10`, its overall ATT equals the doubly
robust DiD estimate, and simulations with confounding covariates check coverage. Two contdid
issues are avoided (see the known differences below); over 300 simulations dml-rs's standard
error of the overall ACRT matches the spread of the estimates.

`SyntheticDID` reproduces R's `synthdid` (version 0.0.9, the authors' package) in nine
configurations: the California Proposition 99 data shipped with the package and simulated panels
with several treated units, with and without covariates, each for synthetic DiD, synthetic
control and DiD
([`parity/generate_synthdid_fixtures.R`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_synthdid_fixtures.R)).
Estimates, unit and time weights, covariate coefficients, effect curves and the jackknife,
placebo and bootstrap standard errors all agree to `1e-9`. The placebo and bootstrap draws match
because dml-rs ports R's random number generator (`set.seed` and `sample()`, checked against R's
own draws).

`CausalSurvival` has no counterpart in DoubleML. Three checks cover it. Without covariates and
with empirical hazards on a grid of all event times it reduces to Kaplan-Meier, and on R's `lung`
data it reproduces `survival::survfit` (version 3.8.6) to `1e-12` and the restricted mean of
`survRM2::rmst2` (version 1.0.4) to `1e-10`; the standard errors agree with Greenwood's formula and
with `rmst2` to 2% ([`parity/generate_survival_fixtures.R`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_survival_fixtures.R)).
A NumPy implementation of the score from scratch matches to `1e-6`. On simulated data with a known
truth, it is compared with `grf::causal_survival_forest` (version 2.6.1, 1000 trees) on the same
datasets ([`parity/survival_mc.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/survival_mc.py)):
`n = 2000`, 60 datasets, horizon 1, nonlinear hazards, and treatment and censoring that depend on
the covariates. dml-rs uses random forest hazards on 20 slots and a logistic propensity model.

| Estimator | Bias | Sd | Mean se | RMSE | Coverage |
|---|---|---|---|---|---|
| dml-rs, survival probability | -0.0016 | 0.0255 | 0.0268 | 0.0254 | 0.98 |
| grf, survival probability | -0.0023 | 0.0234 | 0.0238 | 0.0233 | 0.98 |
| dml-rs, RMST | -0.0051 | 0.0164 | 0.0162 | 0.0170 | 0.97 |
| grf, RMST | -0.0008 | 0.0166 | 0.0156 | 0.0165 | 0.95 |

The two are comparable. grf is somewhat more precise for the survival probability, and the small
RMST bias of dml-rs is about 2.4 Monte Carlo standard errors. It may come from the step function on
a coarse grid, which a finer grid reduces at the cost of more rows. With 60 datasets, the coverage
figures are accurate to about 0.03. In a rerun with 10 datasets after dml-rs 1.2.1 made forest
predictions parallel, a fit took 1.7 seconds with dml-rs and 1.6 with grf (1.2.0 took 2.0). With
binned forests (`max_bins=255`, dml-rs 1.3.0) it takes 0.9 seconds, with the same bias and spread
over those 10 datasets (-0.0033 and 0.025 against -0.0041 and 0.026).

Competing events and clusters (dml-rs 1.3.0) have three more checks. On R's `mgus2` data
(progression to a plasma cell malignancy competing with death, by sex), the cumulative incidences
of both causes reproduce the Aalen-Johansen estimates of `survival::survfit` to `1e-12`, and the
scores of survival and of both causes add up to one, unit by unit. A NumPy implementation of the
cumulative-incidence score in its direct form (with $S_j$ in the denominator, where dml-rs uses a
recursion) matches to `1e-6`. With 50 clusters of 20 units that share a gamma frailty (variance 2)
and a cluster-level treatment, the 95% intervals covered the truth in 56% of 150 datasets without
clustering and in 93% with `cluster=` (standard errors about four times larger).

The restricted mean time lost to a cause (dml-rs 1.4.0) is the integral of its incidence curve:
on `mgus2` with empirical hazards, RMST plus the time lost to both causes equals the horizon to
`1e-12` in both arms, and the scores cancel unit by unit.

`OptimalPolicyTree` reproduces R's `policytree` (version 1.2.5) exact tree search
([`parity/generate_policytree_fixtures.R`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/generate_policytree_fixtures.R)):
in six configurations (depth 0 to 3, continuous and discrete features with ties, `split_step` and
`min_node_size`) the predicted actions are identical, and the trees are identical node for node
except in one depth-3 case with an exact tie between two thresholds (the unit between them gets
the same action either way, so the choice rests on the last bit of the reward sums; policytree's
own depth-2 search on that node picks dml-rs's threshold). `policy_value` equals the mean of the
selected scores with the standard error of the mean.

`ConformalITE` and `LeeBounds` have no installable reference implementation here (`cfcausal` and
`leebounds` are not on CRAN). `ConformalITE` matches a NumPy implementation of weighted split
conformal prediction on the same split (residual and quantile scores, all target populations) to
`1e-10`, and its intervals reach the nominal coverage in simulations (finite-sample coverage with
a known propensity score). `LeeBounds` matches a NumPy implementation of fractional trimming with
and without covariate cells to `1e-12`; in simulations with monotone selection the bounds contain
the effect, covariate cells narrow them, and the Imbens–Manski interval covers.

Causal forests are compared with EconML's port of grf (`econml.grf.CausalForest`, 2000 trees),
grown on exactly the same residuals, in the design of Wager and Athey (2018) with 4000
observations and 20 replications
([`parity/compare_causal_forest.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/compare_causal_forest.py)).
Their random number generators differ, so the forests are compared statistically:

| | RMSE of $\hat\tau(x)$ | Coverage of 95% intervals | Mean interval width | Time per forest |
|---|--:|--:|--:|--:|
| dml-rs | 0.182 | 0.93 | 0.78 | 0.9 s |
| EconML | 0.174 | 0.88 | 0.60 | 3.1 s |

The two forests' estimates correlate at 0.998, and the average effect from dml-rs's doubly
robust scores covers the true average effect in 95% of the replications.

The instrumental, multi-arm causal and regression forests are compared the same way with
`econml.grf.CausalIVForest`, `CausalForest` (one treatment column per arm) and `RegressionForest`
(2000 trees, 4000 observations, 20 replications;
[`parity/compare_grf_forests.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/compare_grf_forests.py)).
The dml-rs times include cross-fitting the nuisance models; EconML's grow only the forest.

| Forest | | RMSE | Coverage of 95% intervals | Mean interval width | Time |
|---|---|--:|--:|--:|--:|
| Instrumental | dml-rs | 0.259 | 0.98 | 1.41 | 3.7 s |
| | EconML | 0.218 | 0.95 | 0.94 | 6.9 s |
| Multi-arm (two arms) | dml-rs | 0.157 | 0.96 | 0.78 | 2.5 s |
| | EconML | 0.148 | 0.93 | 0.60 | 7.4 s |
| Regression | dml-rs | 0.124 | 0.86 | 0.41 | 1.7 s |
| | EconML | 0.124 | 0.79 | 0.33 | 3.8 s |

The estimates of the two libraries correlate at 0.978 (instrumental), 0.992 (multi-arm) and 0.998
(regression). dml-rs's intervals are wider and cover more often. The average conditional LATE of
the instrumental forest has a mean bias of 0.013 and covers in 90% of the replications.

Monte-Carlo tests simulate data with a known effect and check that confidence intervals reach
their nominal coverage: PLR, LPLR (both scores), PLPR (three approaches, with correlated unit effects), RDFlex (sharp and fuzzy), IRM (ATE and ATTE), PLIV, IIVM, DiD, sample selection (both scores), quantile and CVaR effects, joint bootstrap bands, and
clustered data (where iid intervals cover about 75% of the time and cluster-robust ones about 95%).

## Known differences

These are deliberate, documented, and covered by tests:

- **Continuous-treatment DiD (contdid 0.1.2).** contdid evaluates its spline on the grid with
  `splines2::bSpline(dvals)`, whose boundary knots come from the grid's range rather than from the
  treated doses used in the fit, so with its default grid (10%–99% quantiles of the dose) the
  dose-response curve is off (by up to 0.08 in a simple design with effect $2d^2$); dml-rs uses
  the fitted basis everywhere, and the two agree when the grid spans the treated doses. contdid
  also divides the influence function of the overall ACRT, which is on the scale of the $n_1$
  treated units, by the full sample size $n$, so its standard error is too small by the factor
  $n_1/n$; dml-rs's matches the sampling spread in simulations. Both are reported
  ([contdid #16](https://github.com/bcallaway11/contdid/issues/16),
  [contdid #13](https://github.com/bcallaway11/contdid/issues/13)), with fixes proposed in
  contdid [#17](https://github.com/bcallaway11/contdid/pull/17) and
  [#18](https://github.com/bcallaway11/contdid/pull/18) and
  [ptetools #7](https://github.com/bcallaway11/ptetools/pull/7).

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
  estimates and sensitivity bounds agree. This is fixed on DoubleML's main branch, not yet in a
  release.
- **Sensitivity benchmarks of a staggered DiD.** dml-rs refits every ATT(g, t) without the
  benchmarking covariates. DoubleML 0.11's `DoubleMLDIDMulti.sensitivity_benchmark` changes the
  covariate list of a copy whose group-time models have already read their covariates, so its
  short model equals the long one and every benchmark is 0. The fixtures compare with DoubleML's
  benchmark statistics computed from a short model fitted on data without the covariates, on the
  same splits; they agree to `1e-7`. Reported as [DoubleML #430](https://github.com/DoubleML/doubleml-for-py/issues/430) (the same happens in
  `DoubleMLDIDBinary` and `DoubleMLDIDCSBinary`).
- **PQ with external propensities only.** A supplied `ml_m` also serves the preliminary estimate,
  on the rows of the first half of each training fold. DoubleML 0.11 indexes the external
  propensities by position within that half instead of by row, so it pairs them with the wrong
  observations ([DoubleML #431](https://github.com/DoubleML/doubleml-for-py/issues/431)). With external `ml_g` (with or without `ml_m`) the two
  agree to `1e-9`.
- **σ² and ν² of combinations.** For a multiple of a single estimate dml-rs keeps the
  sensitivity elements (with `ν²` scaled by the squared weight), so benchmarks remain possible;
  DoubleML drops them once frameworks are added, even when all but one weight is zero. The bounds
  are the same.
- **Staggered DiD data with anticipation.** `make_did_CS2021(anticipation_periods > 0,
  time_type="float")` in DoubleML 0.11 also drops the never-treated units (its first filter tests
  `pd.isna`, which is false for the `inf` cohort; the datetime version keeps them). dml-rs keeps
  them in both versions ([DoubleML #432](https://github.com/DoubleML/doubleml-for-py/issues/432)).
- **Not-yet-treated controls before treatment.** DoubleML's main branch (#393, not in 0.11.4) drops
  cohorts treated between the evaluation period and `g` from the controls of pre-treatment
  (placebo) periods. dml-rs keeps them, as 0.11.4 does, so post-treatment group-time effects agree
  and pre-treatment ones can differ. Both are valid comparisons under parallel trends.
- **Normalised SSM weights.** DoubleML 0.11.4 scales the weights to about `P(D = d)`, which biases
  the estimate; its main branch normalises them to mean one (#427), and so does dml-rs. These two
  fixtures come from DoubleML's main branch.
- **Fold-specific tuning with repeated splits.** DoubleML 0.11's `tune(tune_on_folds=True)` stores
  its results in a list indexed `[repetition][treatment]` that has one entry per treatment, so it
  fails whenever the numbers of repetitions and treatments differ
  ([DoubleML #433](https://github.com/DoubleML/doubleml-for-py/issues/433)). dml-rs tunes every repetition; the tests compare with DoubleML for
  one repetition.
- **Optuna tuning of IV-type PLIV.** DoubleML 0.11's `tune_ml_models` for `DoubleMLPLIV` with
  `score="IV-type"` calls `cross_val_predict` for `ml_r` without the treatment as target and fails;
  dml-rs uses the treatment, as its grid search does ([DoubleML #434](https://github.com/DoubleML/doubleml-for-py/issues/434)).
- **Initial estimate for tuning `ml_g`** (PLR and PLIV, IV-type). With a classifier for `ml_m`,
  dml-rs uses predicted probabilities in the partialling-out estimate that defines `ml_g`'s target;
  DoubleML's grid search uses predicted classes. With regressors the two agree.
- **`LogisticRegressionCV`** solves each fit to the exact optimum with Newton's method;
  scikit-learn's default L-BFGS solver stops earlier, so its probabilities can differ by about
  `1e-3` at default settings.

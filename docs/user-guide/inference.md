# Inference

## Standard errors and confidence intervals

Standard errors come from the orthogonal score (see [Models](models.md)); intervals and p-values
use the normal approximation. With several treatments, each gets its own estimate, standard error
and interval.

```python
model.summary          # coef, std err, t, P>|t|, 2.5 %, 97.5 %
model.confint(0.90)
```

## Repeated sample splitting

The estimate depends on the random split into folds. With `n_rep` repetitions the reported
coefficient is the median over repetitions, and the standard error accounts for the variation
between them: it is computed from the median of $\hat\theta_r + 1.96\,\widehat{\text{se}}_r$, as in
DoubleML. Per-repetition results are in `all_coef_` and `all_se_`.

```python
dml_rs.PLR(ml_l, ml_m, n_rep=10).fit(X, y, d)
```

## Multiplier bootstrap and simultaneous inference

With several treatments (or the many effects of a staggered DiD), pointwise intervals do not
cover all effects jointly. The multiplier bootstrap gives simultaneous confidence bands and
multiple-testing adjustment:

```python
model.bootstrap(method="normal", n_boot=1000)   # also "Bayes" and "wild"
model.confint(joint=True)                        # simultaneous bands
model.p_adjust("romano-wolf")                    # also "holm" and "bonferroni"
```

## Combining estimates

Estimates from fits on the same observations and sample splits can be added, subtracted, scaled
and stacked, as DoubleML's `DoubleMLFramework`. The scores are combined the same way, so standard
errors and intervals account for the correlation between the estimates.

=== "Python"

    ```python
    plr = dml_rs.PLR(ml_l, ml_m).fit(X, y, d, folds=folds)
    irm = dml_rs.IRM(ml_g, ml_m_clf).fit(X, y, d, folds=folds)
    diff = plr.framework - irm.framework   # also +, c * fw, -fw
    diff.summary
    both = dml_rs.concat([plr, irm])       # both estimates
    both.bootstrap().confint(joint=True)   # simultaneous bands
    both.linear_combination([[0.5, 0.5]], names=["average"])
    ```

=== "Rust"

    ```rust
    let diff = (&plr - &irm)?;                      // also &a + &b, c * &fit, -&fit
    let both = DmlFit::concat(&[plr, irm])?;
    let average = both.linear_combination(array![[0.5, 0.5]].view(), vec!["average".into()])?;
    ```

A `Framework` has the fitted-model API (`summary`, `confint`, `bootstrap`, `p_adjust`,
`sensitivity_analysis`). For the sensitivity analysis the bias bounds add up with absolute weights,
as in DoubleML. `framework.pval` is computed from the aggregated t-statistic, as a DoubleML
model's `pval`. DoubleML's `DoubleMLFramework.pvals` instead takes the median of the
per-repetition p-values; the two agree with a single repetition.

## Clustered data

When observations are grouped (pupils in schools, workers in firms) and share unobserved shocks,
pass the cluster labels. Whole clusters are assigned to folds, $\theta$ solves a fold-weighted
score and the variance sums the score within clusters (Chiang, Kato, Ma and Sasaki, 2022), as in
DoubleML. Two columns give two-way clustering (e.g. firm and year).

```python
model.fit(X, y, d, cluster=df["school"])                # one-way
model.fit(X, y, d, cluster=df[["firm", "year"]])        # two-way
model.n_clusters_
```

Labels can be of any type. With clustered data, the multiplier bootstrap is not available (as in
DoubleML). In Rust: `DmlData::with_clusters` and `with_two_way_clusters`.

## Sensitivity analysis

How strong would an omitted confounder have to be to change the conclusion? Following Chernozhukov,
Cinelli, Newey, Sharma and Syrgkanis (2022), a confounder is described by `cf_y` (the share of the
outcome's residual variation it explains), `cf_d` (its effect on the Riesz representer) and their
correlation `rho`.

```python
model.sensitivity_analysis(cf_y=0.04, cf_d=0.04, rho=1.0)
print(model.sensitivity_summary)     # bounds, confidence bounds, robustness values
model.sensitivity_benchmark(benchmarking_set=["age"])   # calibrate against an observed covariate
model.sensitivity_plot(benchmarks={"cf_y": [0.03], "cf_d": [0.02], "name": ["age"]})
```

The robustness value `RV` is the confounding strength (with `cf_y = cf_d`) that would move the
estimate to the null hypothesis; `RVa` the strength that would make it insignificant. Available for
PLR, IRM, APO and DiD models, including every ATT(g, t) of a staggered DiD and its aggregations
(`aggregate(...).aggregated_frameworks.sensitivity_analysis()`), and for combinations of these. See the [sensitivity example](../examples/04_sensitivity_analysis.ipynb).

`sensitivity_plot` draws the bound of the effect (`value="theta"`) or of its confidence interval
(`"ci"`) as contours over a grid of `cf_d` and `cf_y` (matplotlib, installed separately), marking
the unadjusted estimate, the scenario of the last `sensitivity_analysis` and any benchmarks, as
DoubleML's `sensitivity_plot`. `sensitivity_contours` returns the grid for other plotting libraries.

## Learner diagnostics

`nuisance_loss_` and `evaluate_learners` report how well the nuisances were learned, on held-out
data; see [Learners](learners.md#comparing-learners).

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

## Anytime-valid inference: confidence sequences

A 95% confidence interval covers θ with probability 0.95 at one sample size fixed in advance. An
experiment or rollout that is checked as data come in, and stopped once the interval excludes
zero, looks many times, and the chance that *some* look is wrong is much larger than 5%: in our
simulation below, 23% with 20 looks. A **confidence sequence** covers θ at all sample sizes at
once, $P(\theta \in C_t \text{ for every } t) \ge 1 - \alpha$, so the data can be checked as
often as wanted and the study stopped at any time.

dml-rs uses the asymptotic confidence sequences of Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas
(2024, *Annals of Statistics*), which do for sequential monitoring what the central limit theorem
does for a fixed sample. After $t$ observations the sequence is

$$
\hat\theta_t \pm \widehat{\text{se}}_t \sqrt{\frac{2(u+1)}{u}\log\frac{\sqrt{u+1}}{\alpha}},
\qquad u = t\rho^2,
$$

the Robbins normal-mixture boundary. Their theory covers doubly robust estimators with
cross-fitted nuisance models, i.e. the DML scores here, so it applies to every model in the
library. `opt_time` sets $\rho$ so that the sequence is tightest around that sample size,
$\rho^2 = (-2\log\alpha + \log(-2\log\alpha + 1)) / t^*$. At $t^*$ and 95% it is about 3.0
standard errors wide on each side (1.96 for a fixed-sample interval), and it grows like
$\sqrt{\log t}$ away from $t^*$: the price of being allowed to look at any time. Fix
`opt_time` before seeing the data, for example at the expected sample size of the first look that
matters.

=== "Python"

    ```python
    model.confidence_sequence(0.95, opt_time=2000)  # (n_coefs, 2)
    model.anytime_pval(0.95, opt_time=2000)  # < 0.05 iff 0 is excluded

    # Monitor a study: refit on all data so far at each look.
    cs = dml_rs.ConfidenceSequence(level=0.95, opt_time=2000)
    cs.run(dml_rs.IRM(ml_g, ml_m), looks=range(500, 20_001, 500), X=X, y=y, d=d)
    cs.summary             # per look: estimate, se, interval, p-value
    cs.first_exclusion(0)  # first sample size that excludes 0
    cs.plot()
    # or one look at a time: cs.update(model_fitted_on_data_so_far)
    ```

=== "Rust"

    ```rust
    use dml_core::anytime::ConfidenceSequence;

    let mut cs = ConfidenceSequence::new(0.95, 2000)?;
    for &t in &looks {
        let fit = Irm::new(ml_g.clone(), ml_m.clone()).fit(&data_up_to(t))?;
        cs.update(&fit)?;
    }
    let running = cs.running_intervals(); // (looks, n_coefs, 2)
    let stop = cs.first_exclusion(0.0);
    ```

The intervals of all looks hold together, so their running intersection (`lower_`, `upper_`) is
also a confidence sequence, and the running minimum of the anytime p-values is a valid p-value at
whatever time the study stops. Clustered fits count clusters rather than observations as the
sample size. The rows must be in the order in which the data arrived.

In 300 simulated observational studies (IRM with cross-fitted nuisance models, looks every 250
units up to 5000), the 95% sequence missed the true ATE at some look in 0.7% of the studies and the
repeated 95% intervals in 23% ([`parity/anytime_mc.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/anytime_mc.py)).
With a true effect of 0.15, the sequence excluded zero by 5000 units in 96% of the studies.
See the [monitoring example](../examples/12_confidence_sequences.ipynb).

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

Staggered DiD takes one-way cluster labels too (`StaggeredDID.fit(..., cluster=...)`, constant
within a panel unit; Rust `PanelData::with_clusters`, `RepeatedCrossSections::with_clusters`), an
extension, since DoubleML's `DoubleMLDIDMulti` does not support clustered data. Each ATT(g, t)
assigns whole clusters to folds, and the standard errors of the ATT(g, t), their aggregations and
the sensitivity bounds use the cluster-robust variance of the score over all units,
$\sum_c \big(\sum_{i \in c} \psi_i\big)^2 / \big(\sum_i \psi_{a,i}\big)^2$. In a simulation
with a common shock per cluster and period, clustered intervals for the group effects cover 90%
and 96% of the time, iid intervals about a third.

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

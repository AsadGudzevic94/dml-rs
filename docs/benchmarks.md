# Benchmarks

dml-rs runs the same estimators as DoubleML and gives the same results (see
[Validation](validation.md)), so the comparison is purely about time. Each row fits the same
model on the same simulated data with 5-fold cross-fitting, once with dml-rs and its built-in
learners and once with DoubleML 0.11.4 and the matching scikit-learn 1.9.1 learners.

**Machine:** a laptop with an Intel Core i5-1240P (12 cores, 16 threads) and 8 GB of memory,
Windows 11, Python 3.13. Times are wall-clock seconds for one complete fit (the faster of two
runs; one run for DoubleML at 1 000 000 observations). Both libraries use all cores where they
can: dml-rs fits the folds in parallel and its forests and histogram boosting are multi-threaded;
scikit-learn's forests run with `n_jobs=-1` and its histogram boosting uses OpenMP, while its
`LassoCV`, `LogisticRegressionCV` and `GradientBoosting*` are single-threaded, as is DoubleML's
loop over folds.

**Data:** 20 covariates, a nonlinear confounder, a continuous treatment (PLR) or a binary one
(IRM). The estimates of the two libraries agree in every row (exactly for the learners that match
scikit-learn bit for bit, within sampling noise for the random forests).

## Results

| Model | Learners | n | dml-rs | DoubleML + scikit-learn | Speed-up |
|---|---|--:|--:|--:|--:|
| PLR | Lasso, CV penalty | 10 000 | 0.05 s | 0.47 s | **9.3×** |
| PLR | Lasso, CV penalty | 100 000 | 0.60 s | 2.78 s | **4.7×** |
| PLR | Random forest (100 trees) | 10 000 | 3.23 s | 11.18 s | **3.5×** |
| PLR | Random forest (100 trees) | 50 000 | 30.56 s | 73.48 s | **2.4×** |
| PLR | Gradient boosting (100 trees) | 10 000 | 3.76 s | 32.81 s | **8.7×** |
| PLR | Gradient boosting (100 trees) | 50 000 | 23.08 s | 195.83 s | **8.5×** |
| PLR | Histogram gradient boosting (100 trees) | 10 000 | 0.44 s | 2.92 s | **6.6×** |
| PLR | Histogram gradient boosting (100 trees) | 100 000 | 2.00 s | 5.17 s | **2.6×** |
| PLR | Histogram gradient boosting (100 trees) | 1 000 000 | 18.37 s | 30.69 s | **1.7×** |
| IRM | Lasso + logistic regression, CV penalty | 10 000 | 0.13 s | 1.40 s | **10.9×** |
| IRM | Lasso + logistic regression, CV penalty | 100 000 | 1.85 s | 6.34 s | **3.4×** |
| IRM | Random forest (100 trees) | 10 000 | 2.19 s | 7.14 s | **3.3×** |
| IRM | Random forest (100 trees) | 50 000 | 17.44 s | 43.16 s | **2.5×** |
| IRM | Histogram gradient boosting (100 trees) | 10 000 | 0.77 s | 4.78 s | **6.2×** |
| IRM | Histogram gradient boosting (100 trees) | 100 000 | 2.52 s | 7.71 s | **3.1×** |
| IRM | Histogram gradient boosting (100 trees) | 1 000 000 | 20.68 s | 43.24 s | **2.1×** |

## Where the time goes

- **Single-threaded learners** (gradient boosting, cross-validated Lasso and logistic
  regression) gain the most: dml-rs fits the five folds in parallel, and its implementations are
  faster than scikit-learn's to begin with.
- **Multi-threaded learners** (forests, histogram boosting) already use all cores in
  scikit-learn, so the gain comes from running without Python between the folds and from
  scheduling folds and trees on one thread pool. It shrinks as the data grow and the learners
  dominate the time.
- **Binned random forests:** `max_bins=255` makes a random forest at 100 000 observations and 20
  features 3.4 times faster than the exact one (5.4 s against 18.5 s for 100 trees; scikit-learn
  takes 39 s) at practically the same held-out R² (0.640 against 0.642).
- **The learner alone:** on its own, dml-rs's histogram boosting is faster than scikit-learn's up
  to about 100 000 observations (1.6× at 10 000) and about 10% slower at 1 000 000 on this
  machine, where both are limited by memory bandwidth. It does the same arithmetic in the same
  order as scikit-learn, which is what makes the predictions identical, so the remaining
  difference is in low-level tuning.

## Every model

The table above uses flexible learners, which dominate the time. With linear learners
(scikit-learn's `LinearRegression` and `LogisticRegression` for DoubleML, the built-in `Ols` and
`LogisticRegression` for dml-rs) the framework itself is what is timed. Speed-up of dml-rs over
DoubleML 0.11.4 for every model both libraries have:

| Model | n = 5 000 | n = 100 000 |
|---|--:|--:|
| IRM | 9.1× | 6.3× |
| IRM sensitivity analysis (bounds and robustness values) | 6.3× | 91× |
| IIVM | 11.0× | 7.9× |
| APOS (4 treatment levels) | 12.4× | 11.1× |
| PQ (potential quantile) | 38× | DoubleML runs out of memory |
| LPQ (local potential quantile) | 23× | DoubleML runs out of memory |
| CVaR | 5.7× | 3.6× |
| QTE (3 quantiles) | 29× | DoubleML runs out of memory |
| SSM, missing at random | 10.7× | 7.2× |
| SSM, nonignorable | 26× | 21× |
| DID (panel) | 10.1× | 6.7× |
| DID (repeated cross-sections) | 9.8× | 7.0× |
| Staggered DID (n / 4 units) | 23× | 7.3× |
| LPLR | 7.6× | 3.1× |
| PLPR (`cre_general`, 10 periods) | 5.8× | 5.3× |
| RDFlex (sharp) | 36× | 25× |

At 100 000 observations DoubleML's PQ, LPQ and QTE ask for a 75 GB array: the density in the
score derivative comes from statsmodels' kernel density estimate, which builds an n × n matrix,
while only its value at the estimate is needed; dml-rs fits PQ in 0.3 s and the three-quantile QTE
in 2.2 s. The script is
[`parity/bench_models.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/bench_models.py).

## Against the R packages

The causal forest and the exact policy tree are ports of R packages written in C++, so the
comparison there is with compiled code. Same data, same settings; each row is the less favourable
of two runs of the script (timings on this laptop vary between runs):

| Task | n | dml-rs | R | Speed-up |
|---|--:|--:|--:|--:|
| Causal forest, 2000 trees (`grf::causal_forest`) | 2 000 | 0.63 s | 1.76 s | **2.8×** |
| Causal forest, 2000 trees | 10 000 | 8.5 s | 13.6 s | **1.6×** |
| Causal forest, 2000 trees | 50 000 | 76 s | 104 s | **1.4×** |
| Policy tree, depth 2, 10 features (`policytree::policy_tree`) | 1 000 | 0.41 s | 0.88 s | **2.1×** |
| Policy tree, depth 2, 10 features | 5 000 | 13.0 s | 21.1 s | **1.6×** |
| Synthetic DiD on Prop 99 (`synthdid::synthdid_estimate`) | 39 × 31 | 0.013 s | 0.19 s | **15×** |
| Synthetic DiD, placebo standard error (200 replications) | 39 × 31 | 0.45 s | 29.0 s | **65×** |
| Causal mediation (`causalweight::medDML`, parametric) | 5 000 | 0.016 s | 1.14 s | **70×** |
| Causal mediation | 20 000 | 0.047 s | 3.65 s | **78×** |
| Continuous DiD (`contdid::cont_did`, cubic B-spline) | 5 000 | 0.036 s | 3.02 s | **85×** |
| Continuous DiD | 20 000 | 0.125 s | 11.7 s | **93×** |

Both forests get the same linear nuisance estimates, so only the forest is timed. The policy
trees are identical. `contdid` computes its standard errors with 100 bootstrap draws, dml-rs from
the influence function; synthdid, causalweight and contdid are written in R itself, which
explains most of their gap. The script is
[`parity/bench_r.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/bench_r.py).

Effects on a censored outcome (`CausalSurvival`) take about as long as
`grf::causal_survival_forest` with exact random forests (1.7 s against 1.6 s per fit in the
[validation study](validation.md), where the two are different estimators of the same effect) and
about 0.9 s with binned forests (`max_bins=255`), at the same accuracy.

## Reproduce

```sh
python parity/bench_all.py
```

[`parity/bench_all.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/bench_all.py)
prints one JSON line per configuration and the table above. Timings depend on the machine; the
ratios are what carries over.

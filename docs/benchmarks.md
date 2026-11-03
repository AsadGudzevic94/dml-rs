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
- **The learner alone:** on its own, dml-rs's histogram boosting is faster than scikit-learn's up
  to about 100 000 observations (1.6× at 10 000) and about 10% slower at 1 000 000 on this
  machine, where both are limited by memory bandwidth. It does the same arithmetic in the same
  order as scikit-learn, which is what makes the predictions identical, so the remaining
  difference is in low-level tuning.

## Against the R packages

The causal forest and the exact policy tree are ports of R packages written in C++, so the
comparison there is with compiled code. Same data, same settings, the faster of two runs:

| Task | n | dml-rs | R | Speed-up |
|---|--:|--:|--:|--:|
| Causal forest, 2000 trees (`grf::causal_forest`) | 2 000 | 0.63 s | 1.76 s | **2.8×** |
| Causal forest, 2000 trees | 10 000 | 5.3 s | 12.5 s | **2.4×** |
| Causal forest, 2000 trees | 50 000 | 70 s | 123 s | **1.8×** |
| Policy tree, depth 2, 10 features (`policytree::policy_tree`) | 1 000 | 0.36 s | 0.99 s | **2.8×** |
| Policy tree, depth 2, 10 features | 5 000 | 13.0 s | 24.1 s | **1.9×** |

Both forests get the same linear nuisance estimates, so only the forest is timed. The policy
trees are identical. Effects on a censored outcome (`CausalSurvival`) are **not** faster than
`grf::causal_survival_forest` (about 2.0 s against 1.4 s per fit in the
[validation study](validation.md)); the two are different estimators.

## Reproduce

```sh
python parity/bench_all.py
```

[`parity/bench_all.py`](https://github.com/AsadGudzevic94/dml-rs/blob/main/parity/bench_all.py)
prints one JSON line per configuration and the table above. Timings depend on the machine; the
ratios are what carries over.

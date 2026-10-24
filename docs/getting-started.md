# Getting started

## Installation

dml-rs is published as the Python package [`dml-rs`](https://pypi.org/project/dml-rs/) (Python
3.9 or newer, prebuilt wheels for Linux, macOS and Windows) and as the Rust crate
[`dml-core`](https://crates.io/crates/dml-core) (Rust 1.80 or newer).

=== "Python"

    ```sh
    pip install dml-rs
    ```

=== "Rust"

    ```sh
    cargo add dml-core
    ```

The Python package only requires NumPy. pandas is optional (summaries are returned as DataFrames
when it is installed), and scikit-learn is needed only if you use its estimators as learners.

## The workflow

Every DML estimation has the same three ingredients:

1. **A model**, which defines the causal parameter and its orthogonal score, for example the
   partially linear model `PLR` for a continuous treatment or the interactive model `IRM` for the
   average effect of a binary treatment.
2. **Nuisance learners**, machine-learning models for the conditional expectations the score needs
   (e.g. $E[Y \mid X]$ and $E[D \mid X]$). Any learner works; the better it predicts, the more
   precise the estimate.
3. **Sample splitting.** Each nuisance is fitted on $K-1$ folds and evaluated on the held-out fold
   (cross-fitting), optionally repeated over several random splits.

=== "Python"

    ```python
    import dml_rs

    model = dml_rs.PLR(
        ml_l=dml_rs.GradientBoostingRegressor(),   # E[Y | X]
        ml_m=dml_rs.GradientBoostingRegressor(),   # E[D | X]
        n_folds=5,
        n_rep=1,
        random_state=0,
    )
    model.fit(X, y, d)          # X: (n, p), y: (n,), d: (n,) or (n, k) for k treatments
    ```

=== "Rust"

    ```rust
    use dml_core::{DmlData, GradientBoostingRegressor, Plr};

    let data = DmlData::new(y, d, x)?;
    let fit = Plr::new(GradientBoostingRegressor::new(), GradientBoostingRegressor::new())
        .n_folds(5)
        .n_rep(1)
        .seed(0)
        .fit(&data)?;
    ```

## Results

=== "Python"

    ```python
    model.summary            # coefficient table (pandas DataFrame)
    print(model)             # full report, including learner losses
    model.coef_, model.se_   # estimates and standard errors, one per treatment
    model.confint(0.95)      # confidence intervals, shape (n_treat, 2)
    model.pval_              # p-values
    model.nuisance_loss_     # out-of-sample RMSE / log loss of each learner
    model.predictions_       # cross-fitted nuisance predictions
    ```

=== "Rust"

    ```rust
    println!("{fit}");                 // full report
    let (coef, se) = (fit.coef(), fit.se());
    let ci = fit.confint(0.95)?;
    let p = fit.p_value();
    let loss = fit.nuisance_loss("ml_l");
    ```

```
PLR (partialling out)  |  n_obs = 2000, n_folds = 5, n_rep = 1
           coef    std err        t    P>|t|      2.5 %     97.5 %
d        0.4706     0.0222   21.157   0.0000     0.4270     0.5142
Nuisance learners (out-of-sample, mean over repetitions):
  ml_l         RMSE     1.1579
  ml_m         RMSE     1.0413
```

## Choosing learners

The nuisance functions are prediction problems, so choose learners as you would for prediction and
compare them on the out-of-fold losses in `nuisance_loss_`. The built-in learners run entirely in
Rust; scikit-learn estimators are called from Rust for each fold. See [Learners](user-guide/learners.md).

## Reproducibility

Sample splits are drawn from `random_state` (Python) or `seed` (Rust), so results are reproducible.
To reproduce a DoubleML analysis exactly, pass its splits: `fit(..., folds=model.smpls)` in Python
or `fit_with_folds` in Rust.

## Next steps

- The [examples](examples/01_getting_started.ipynb) walk through complete analyses.
- The [user guide](user-guide/models.md) describes every model and the inference options.
- The [API reference](api/python.md) lists every class and parameter.

# R package

The R package **dmlrs** runs the same Rust engine as the Python package. On the same data the two
give the same estimates (the R tests check this against Python results to 1e-10). It is a first
version: the partially linear, interactive, IV and difference-in-differences models with the
built-in learners, multiplier bootstrap and sensitivity analysis.

## Installation

dmlrs builds from source and needs a Rust toolchain (<https://rustup.rs>, Rust 1.80 or later). On
Windows, also install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and the GNU target
with `rustup target add x86_64-pc-windows-gnu`.

```r
# install.packages("remotes")
remotes::install_github("AsadGudzevic94/dml-rs", subdir = "r/dmlrs")
```

## Usage

```r
library(dmlrs)

set.seed(1)
n <- 1000
x <- matrix(rnorm(n * 10), n)
d <- x[, 1] + rnorm(n)
y <- 0.5 * d + x[, 2] + rnorm(n)

fit <- dml_plr(y, d, x, ml_l = lrn_lasso_cv(), n_rep = 3)
fit
#> Double machine learning: PLR (partialling out)
#> 1000 observations, 5 folds, 3 repetitions
#>
#>   Estimate Std. Error t value Pr(>|t|)
#> d  0.51048    0.03322   15.37   <2e-16 ***

confint(fit)
confint(fit, joint = TRUE)                 # simultaneous intervals, multiplier bootstrap
sensitivity_analysis(fit, cf_y = 0.03, cf_d = 0.03)
```

| Function | Model | Python |
|---|---|---|
| `dml_plr()` | Partially linear regression, one or several treatments | `PLR` |
| `dml_irm()` | Interactive regression model: ATE / ATTE | `IRM` |
| `dml_pliv()` | Partially linear IV regression, one instrument | `PLIV` |
| `dml_iivm()` | Interactive IV model: LATE | `IIVM` |
| `dml_did()` | Difference in differences, panels (`t = NULL`) or repeated cross-sections | `DID`, `DIDCS` |

Learners are specifications that the Rust code turns into learners, so cross-fitting never calls
back into R: `lrn_ols()`, `lrn_lasso()`, `lrn_lasso_cv()`, `lrn_logistic()`, `lrn_logistic_cv()`,
`lrn_random_forest()`, `lrn_gradient_boosting()`, `lrn_hist_gradient_boosting()` and
`lrn_regression_forest()`, with the parameters of the Python learners.

Fitted models have `print()`, `summary()`, `coef()`, `confint()` and `nobs()` methods. A model saved
with `saveRDS()` keeps its estimates; intervals, the bootstrap and the sensitivity analysis need a
refit.

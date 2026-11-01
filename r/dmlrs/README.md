# dmlrs

Fast double / debiased machine learning for R, implemented in Rust.

dmlrs fits the DoubleML models (partially linear, interactive, instrumental variable,
difference-in-differences and multi-valued treatments) with cross-fitted nuisance learners,
multiplier bootstrap and sensitivity analysis, and goes beyond them with causal mediation,
difference in differences with a continuous treatment, synthetic difference in differences,
optimal policy trees and Lee bounds. The estimation, including the learners (lasso, logistic
regression, random forests, gradient boosting, histogram gradient boosting), runs in Rust through
the [dml-core](https://crates.io/crates/dml-core) crate, the engine of the Python package
[dml-rs](https://pypi.org/project/dml-rs/). On the same data the two packages give the same
estimates; the Python package is validated against DoubleML.

## Installation

Binaries for Windows, macOS and Linux are on [r-universe](https://asadgudzevic94.r-universe.dev),
so no Rust toolchain is needed:

```r
install.packages("dmlrs", repos = c("https://asadgudzevic94.r-universe.dev", "https://cloud.r-project.org"))
```

To build from source instead, install a Rust toolchain (<https://rustup.rs>, Rust 1.80 or later)
and, on Windows, [Rtools](https://cran.r-project.org/bin/windows/Rtools/) and the GNU target
(`rustup target add x86_64-pc-windows-gnu`), then:

```r
# install.packages("remotes")
remotes::install_github("AsadGudzevic94/dml-rs", subdir = "r/dmlrs")
```

## Example

```r
library(dmlrs)

set.seed(1)
n <- 1000
x <- matrix(rnorm(n * 10), n)
d <- x[, 1] + rnorm(n)
y <- 0.5 * d + x[, 2] + rnorm(n)

fit <- dml_plr(y, d, x, ml_l = lrn_lasso_cv(), n_rep = 3)
fit
confint(fit)
confint(fit, joint = TRUE)       # multiplier bootstrap
sensitivity_analysis(fit, cf_y = 0.03, cf_d = 0.03)

db <- rbinom(n, 1, plogis(x[, 1]))
dml_irm(y + db, db, x, ml_g = lrn_random_forest(), ml_m = lrn_logistic_cv())
```

| Function | Model |
|---|---|
| `dml_plr()` | Partially linear regression (one or several treatments) |
| `dml_irm()` | Interactive regression model: ATE / ATTE of a binary treatment |
| `dml_pliv()` | Partially linear IV regression |
| `dml_iivm()` | Interactive IV model: LATE with a binary instrument |
| `dml_did()` | Difference in differences (panels and repeated cross-sections) |
| `dml_apos()` | Average potential outcomes of a discrete treatment (contrasts via `linear_combination()`) |
| `dml_mediation()` | Natural direct and indirect effects (the estimator of `causalweight::medDML()`) |
| `dml_contdid()` | Difference in differences with a continuous dose (equals `contdid` without covariates) |
| `synth_did()`, `synth_did_panel()` | Synthetic DiD, synthetic control and DiD (a port of `synthdid`) |
| `policy_tree()`, `policy_value()` | Exact optimal policy trees (a port of `policytree`) and doubly robust policy values |
| `lee_bounds()` | Lee bounds on effects under sample selection |

Learners: `lrn_ols()`, `lrn_lasso()`, `lrn_lasso_cv()`, `lrn_logistic()`, `lrn_logistic_cv()`,
`lrn_random_forest()`, `lrn_gradient_boosting()`, `lrn_hist_gradient_boosting()`,
`lrn_regression_forest()`.

The Python package has more models (quantile effects, staggered DID, selection, RD, causal forests,
CATE learners, ...) and accepts any scikit-learn estimator; see the
[documentation](https://asadgudzevic94.github.io/dml-rs/).

## License

MIT

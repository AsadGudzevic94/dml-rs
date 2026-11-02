# R package

The R package **dmlrs** runs the same Rust engine as the Python package. On the same data the two
give the same estimates (the R tests check this against Python results). It covers the partially
linear, interactive, IV, difference-in-differences and multi-valued treatment models with the
built-in learners, multiplier bootstrap and sensitivity analysis, plus heterogeneous effects (best
linear predictor, CATE meta-learners, causal forests, conformal intervals), dose-response curves,
causal mediation, difference in differences with a continuous treatment, synthetic difference in
differences, optimal policy trees and Lee bounds.

The full reference of the R functions, and an article on heterogeneous effects, are on the
[R package site](https://asadgudzevic94.github.io/dml-rs/r/).

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
| `dml_apos()` | Average potential outcomes of a discrete treatment | `APOS` |
| `dml_mediation()` | Natural direct and indirect effects | `Mediation` |
| `dml_contdid()` | Difference in differences with a continuous dose | `ContinuousDID` |
| `synth_did()`, `synth_did_panel()` | Synthetic DiD, synthetic control and DiD | `SyntheticDID` |
| `policy_tree()`, `policy_value()` | Exact optimal policy trees and policy values | `OptimalPolicyTree`, `policy_value` |
| `lee_bounds()` | Lee bounds under sample selection | `LeeBounds` |
| `dml_dose_response()` | Dose-response curve of a continuous treatment | `DoseResponse` |
| `cate()`, `gate()` | Best linear predictor of the CATE, group average effects | `PLR.cate`, `IRM.gate`, ... |
| `cate_learner()` | CATE meta-learners (S, T, X, R, DR) | `SLearner`, ..., `DRLearner` |
| `causal_forest()`, `instrumental_forest()` | Causal and instrumental forests | `CausalForest`, `InstrumentalForest` |
| `conformal_ite()` | Conformal intervals for individual effects | `ConformalITE` |

Learners are specifications that the Rust code turns into learners, so cross-fitting never calls
back into R: `lrn_ols()`, `lrn_lasso()`, `lrn_lasso_cv()`, `lrn_logistic()`, `lrn_logistic_cv()`,
`lrn_random_forest()`, `lrn_gradient_boosting()`, `lrn_hist_gradient_boosting()`,
`lrn_regression_forest()` and `lrn_quantile_forest()`, with the parameters of the Python learners.

Fitted models have `print()`, `summary()`, `coef()`, `confint()` and `nobs()` methods, and
`linear_combination()` estimates contrasts such as differences between treatment levels or between
policies. Synthetic DiD fits have `plot()` and `synth_se()` for further standard errors. A model
saved with `saveRDS()` keeps its estimates; intervals, the bootstrap and the sensitivity analysis
need a refit.

```r
# Synthetic DiD on a long panel (Proposition 99 data, as in synthdid)
url <- "https://raw.githubusercontent.com/synth-inference/synthdid/master/data/california_prop99.csv"
prop99 <- read.csv(url, sep = ";")
fit <- synth_did_panel(prop99, "State", "Year", "PacksPerCapita", "treated")
fit
plot(fit)

# Learn a policy on half of the units, evaluate it on the other half
apos <- dml_apos(y, d, x, ml_g = lrn_random_forest(), ml_m = lrn_random_forest())
gamma <- double_robust_scores(apos)
tree <- policy_tree(x[learn, ], gamma[learn, ], depth = 2)
values <- policy_value(gamma[test, ], tree = predict(tree, x[test, ]),
                       treat_all = rep(2, length(test)))
linear_combination(values, c(1, -1))     # gain of the tree over treating everyone
```

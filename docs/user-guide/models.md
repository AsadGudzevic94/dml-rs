# Models

Each model defines a causal parameter $\theta$ through a Neyman-orthogonal score
$\psi(W; \theta, \eta)$ that is linear in $\theta$, $\psi = \psi_a \theta + \psi_b$, where $\eta$
are the nuisance functions. dml-rs cross-fits $\eta$ and solves $\frac{1}{n}\sum_i \psi(W_i;
\hat\theta, \hat\eta) = 0$ (the DML2 estimator). The variance follows from the score:
$\hat\sigma^2 = \hat J^{-2}\,\frac1n\sum_i \psi_i^2$ with $\hat J = \frac1n \sum_i \psi_{a,i}$.

The models and their scores follow [DoubleML](https://docs.doubleml.org), and each is tested
against it.

## Partially linear regression (PLR)

$$
Y = \theta D + g_0(X) + \varepsilon, \qquad D = m_0(X) + V .
$$

For continuous or binary treatments with a constant effect. Several treatments can be estimated
at once (`d` with one column each).

| Score | Nuisances | Option |
|---|---|---|
| Partialling out (default) | `ml_l` for $E[Y \mid X]$, `ml_m` for $E[D \mid X]$ | `score="partialling out"` |
| IV-type | additionally `ml_g` for $E[Y - \theta D \mid X]$ | `score="IV-type"` |

=== "Python"

    ```python
    dml_rs.PLR(ml_l, ml_m).fit(X, y, d)
    dml_rs.PLR(ml_l, ml_m, ml_g=ml_g, score="IV-type").fit(X, y, d)
    ```

=== "Rust"

    ```rust
    Plr::new(ml_l, ml_m).fit(&data)?;
    Plr::new(ml_l, ml_m).score(PlrScore::IvType).ml_g(ml_g).fit(&data)?;
    ```

## Logistic partially linear regression (LPLR)

For a binary outcome (Liu, Zhang and Zhou, 2021):

$$
P(Y = 1 \mid D, X) = \operatorname{expit}\big(\theta D + r_0(X)\big),
$$

so $\theta$ is a log odds ratio, constant across $X$. The nuisances are `ml_M` for
$M(D, X) = P(Y = 1 \mid D, X)$, `ml_t` for the log odds $E[\operatorname{logit} M(D, X) \mid X]$,
`ml_m` for $E[D \mid X]$ (a regressor, or a classifier for a binary treatment) and optionally
`ml_a`, a second learner for $E[D \mid X]$. `ml_M` and `ml_a` are double cross-fitted: inner folds
within each training fold provide the out-of-sample predictions from which the targets of `ml_t`
and a preliminary estimate are built. The score, non-linear in $\theta$, is solved with Newton's
method.

| Score | `ml_m` is fitted | Option |
|---|---|---|
| Nuisance space (default) | on the observations with $Y = 0$ | `score="nuisance_space"` |
| Instrument | on all observations, weighted by $M(1 - M)$ | `score="instrument"` |

=== "Python"

    ```python
    from sklearn.linear_model import LinearRegression, LogisticRegression

    model = dml_rs.LPLR(LogisticRegression(), LinearRegression(), LinearRegression()).fit(X, y, d)
    model.summary                        # log odds ratio of D
    ```

=== "Rust"

    ```rust
    let fit = Lplr::new(LogisticRegression::default(), Ols::new(), Ols::new())
        .score(LplrScore::Instrument)
        .fit(&data)?;
    ```

## Partially linear panel regression (PLPR)

For static panels with unit effects correlated with the covariates and the treatment (Clarke and
Polselli, 2025):

$$
\begin{aligned}
Y_{it} &= \theta D_{it} + g_0(X_{it}) + \alpha_i + U_{it}, \\
D_{it} &= m_0(X_{it}) + \gamma_i + V_{it} .
\end{aligned}
$$

The panel, in long format with a unit `id` and a period `t` per row, is transformed to deal with
the unit effects, and a partially linear regression (either score) is fitted on the transformed
rows. Folds hold whole units and standard errors are clustered by unit.

| Approach | Transformation | Option |
|---|---|---|
| First differences (default) | $\Delta Y_{it}$ and $\Delta D_{it}$ on $X_{it}$ and $X_{i,t-1}$; rows after a gap are dropped | `approach="fd_exact"` |
| Within groups | outcome, treatment and covariates demeaned within units | `approach="wg_approx"` |
| Correlated random effects | unit means of the covariates added; treatment predictions shifted to the unit means of the treatment | `approach="cre_general"` |
| Correlated random effects, normal | unit means added; `ml_m` also sees the unit mean of the treatment | `approach="cre_normal"` |

Units observed in a single period are dropped. With the correlated random effects approaches a
binary treatment may use a classifier for `ml_m`.

=== "Python"

    ```python
    model = dml_rs.PLPR(ml_l, ml_m, approach="fd_exact").fit(X, y, d, id=df["id"], t=df["year"])
    model.summary                        # treatment "d_diff"
    ```

=== "Rust"

    ```rust
    let panel = StaticPanel::new(y, d, x, id, t)?;
    let fit = Plpr::new(ml_l, ml_m).approach(PanelApproach::WgApprox).fit(&panel)?;
    ```

## Interactive regression model (IRM)

$$
Y = g_0(D, X) + \varepsilon, \qquad D \in \{0, 1\}, \qquad P(D = 1 \mid X) = m_0(X).
$$

The effect may vary arbitrarily with $X$. The target is the average treatment effect
($\text{ATE}$) or the average effect on the treated ($\text{ATTE}$), estimated with the doubly
robust (AIPW) score. `ml_g` is a regressor (fitted separately for treated and controls), `ml_m` a
classifier for the propensity score. Propensities are clipped to
`[clipping_threshold, 1 - clipping_threshold]` (default `0.01`); `normalize_ipw=True` normalises the
inverse-probability weights.

!!! tip "Propensity calibration"
    Flexible classifiers often give poorly calibrated probabilities, and the inverse-probability
    weights amplify the errors. `calibration="isotonic"` maps the cross-fitted propensities through
    an isotonic regression of the treatment on them before clipping; `"isotonic_cv"` cross-fits
    that regression on the same folds (DoubleML's `ps_processor_config`). Available for IRM, APO,
    APOS, IIVM (on the instrument propensity), SSM, the quantile models, DID, DIDCS and
    staggered DID; in Rust `.calibration(Calibration::IsotonicCrossFitted)`.

`weights` target a weighted average effect, as in DoubleML: with the ATE score
$E[w\,(g_0(1, X) - g_0(0, X))] / E[w]$, for example $w = 1\{X \in G\}$ for the effect in a
subgroup; with the ATTE score, 0/1 weights select a subgroup of the treated. A dict with
`weights` and `weights_bar` (one column per repetition) sets the inverse-probability weights
separately.

=== "Python"

    ```python
    model = dml_rs.IRM(ml_g, ml_m, score="ATE").fit(X, y, d)
    model.gate(groups)              # group average treatment effects
    model.cate(basis)               # best linear predictor of the CATE
    model.policy_tree(features)     # who should be treated?

    subgroup = dml_rs.IRM(ml_g, ml_m, weights=(age > 50).astype(float)).fit(X, y, d)
    ```

=== "Rust"

    ```rust
    let fit = Irm::new(ml_g, ml_m).score(IrmScore::Ate).fit(&data)?;
    let gate = fit.gate(groups.view())?;
    let policy = fit.policy_tree(features.view(), 2, 8, 0.01)?;
    ```

## Multi-valued treatments (APO, APOS)

For a discrete treatment with several levels $d$ (e.g. no programme, short programme, long
programme), the average potential outcome

$$
\theta(d) = E[Y(d)] = E[g_0(d, X)], \qquad g_0(d, X) = E[Y \mid D = d, X], \qquad
m_d(X) = P(D = d \mid X),
$$

is estimated with the doubly robust score
$g(d, X) + 1\{D = d\}\,(Y - g(d, X))/m_d(X) - \theta$. `APO` estimates one level; `APOS`
estimates several levels on the same sample splits and gives the contrasts
$\theta(d) - \theta(d_{\text{ref}})$, the average effect of level $d$ against a reference level,
with joint inference. Sensitivity analysis is available for both, and for contrasts it combines
the elements of the two levels (as DoubleML). Default sample splits are stratified by the treatment.

=== "Python"

    ```python
    model = dml_rs.APOS(ml_g, ml_m, treatment_levels=[0, 1, 2]).fit(X, y, d)
    model.summary                       # E[Y(0)], E[Y(1)], E[Y(2)]
    effects = model.causal_contrast(0)  # "1 vs 0", "2 vs 0"
    effects.summary
    effects.bootstrap().confint(joint=True)
    ```

=== "Rust"

    ```rust
    let fit = Apos::new(ml_g, ml_m, vec![0.0, 1.0, 2.0]).fit(&data)?;
    let levels = fit.apos();
    let effects = fit.causal_contrast(&[0.0])?;
    ```

## Quantile effects (PQ, LPQ, CVAR, QTE)

Effects on the whole outcome distribution of a binary treatment, not only its mean. The potential
quantile $\theta_\tau(d)$ of $Y(d)$ at level $\tau$ solves

$$
E\left[ g_0(\theta, X) + \frac{1\{D = d\}}{m_d(X)} \big( 1\{Y \le \theta\} - g_0(\theta, X) \big) \right] = \tau,
$$

with $g_0(\theta, X) = P(Y \le \theta \mid D = d, X)$ and the propensity
$m_d(X) = P(D = d \mid X)$. `ml_g` is a classifier for the event
$Y \le \theta$, `ml_m` a propensity classifier. As in DoubleML (Kallus, Mao and Uehara, 2024), each
fold first computes a preliminary inverse-probability-weighted estimate on a nested split, learns
$g_0$ at that point, and then solves the score; the standard error uses a kernel density estimate
of the score's slope.

- `PQ`: potential quantiles $\theta_\tau(d)$.
- `LPQ`: local potential quantiles of the compliers, with a binary instrument $Z$
  (`ml_m` is then the instrument propensity).
- `CVAR`: the conditional value at risk $E[Y(d) \mid Y(d) \ge \theta_\tau(d)]$, the mean of the
  upper tail; `ml_g` is a regressor.
- `QTE`: the effects $\theta_\tau(1) - \theta_\tau(0)$ at several quantiles (`score="PQ"`, `"LPQ"`
  or `"CVaR"`), with both potential quantiles fitted on the same sample splits and joint
  inference across quantiles.

Inverse-probability weights are normalised by default (`normalize_ipw=True`, as DoubleML), and
default sample splits are stratified by the treatment (and instrument).

=== "Python"

    ```python
    from sklearn.linear_model import LogisticRegression

    model = dml_rs.QTE(LogisticRegression(), quantiles=[0.25, 0.5, 0.75]).fit(X, y, d)
    model.summary                     # effect at each quantile
    model.bootstrap().confint(joint=True)
    model.potential.summary           # "0.25 (d=0)", "0.25 (d=1)", ...

    dml_rs.PQ(ml_g, ml_m, treatment=1, quantile=0.9).fit(X, y, d)
    dml_rs.CVAR(ml_g_regressor, ml_m, quantile=0.9).fit(X, y, d)
    ```

=== "Rust"

    ```rust
    let fit = Qte::pq(ml_g, ml_m, vec![0.25, 0.5, 0.75]).fit(&data)?;
    let effects = fit.effects();      // a DmlFit: confint, bootstrap, ...
    let q90 = Pq::new(ml_g, ml_m).treatment(1).quantile(0.9).fit(&data)?;
    ```

## Sample selection (SSM)

The average treatment effect of a binary treatment when the outcome is observed only for part
of the sample: survey non-response, attrition from a panel, or wages observed only for the
employed (Bia, Huber and Lafférs, 2024). The selection indicator $S$ is 1 where $Y$ is observed.
The score combines the outcome models of the selected treated and controls with
inverse-probability weights for both treatment and selection:

$$
\begin{aligned}
\theta = {} & E\left[ g(1, X) + \frac{D\, S\, (Y - g(1, X))}{m(X)\, \pi(1, X)} \right] \\
& - E\left[ g(0, X) + \frac{(1 - D)\, S\, (Y - g(0, X))}{(1 - m(X))\, \pi(0, X)} \right],
\end{aligned}
$$

with $m(X) = P(D = 1 \mid X)$ and the selection probability $\pi(D, X) = P(S = 1 \mid D, X)$.

- `score="missing-at-random"`: selection depends on the treatment and the covariates only.
- `score="nonignorable"`: selection may also depend on unobservables that affect the outcome. This
  needs an instrument $Z$ that shifts selection but not the outcome. Each training fold is split
  in halves: $\pi$ is learned from $(X, D, Z)$ on the first, and $m$ and $g$ on the second, with
  the predicted selection probability as an additional covariate.

Default sample splits are stratified by treatment and selection. The outcome of unselected
observations is ignored but must be finite (e.g. 0).

=== "Python"

    ```python
    model = dml_rs.SSM(ml_g, ml_pi, ml_m, score="nonignorable").fit(X, y, d, s, z=z)
    model.summary
    ```

=== "Rust"

    ```rust
    let data = DmlData::new(y, d, x)?.with_selection_indicator(s)?.with_instrument(z)?;
    let fit = Ssm::new(ml_g, ml_pi, ml_m).score(SsmScore::Nonignorable).fit(&data)?;
    ```

## Partially linear IV regression (PLIV)

$$
Y = \theta D + g_0(X) + \varepsilon, \qquad E[\varepsilon \mid Z, X] = 0 .
$$

For an endogenous treatment with one or several instruments $Z$ (one column each). Three variants,
as DoubleML:

| Variant | Nuisances | Option |
|---|---|---|
| Partial X (default) | `ml_l` for $E[Y \mid X]$, `ml_m` for $E[Z \mid X]$ (one fit per instrument), `ml_r` for $E[D \mid X]$; several instruments are combined by projecting the treatment residual on the instrument residuals (two-stage least squares) | `partial="X"` |
| Partial Z | `ml_r` for $E[D \mid X, Z]$, used as the instrument | `partial="Z"` |
| Partial XZ | `ml_l` for $E[Y \mid X]$, `ml_m` for $E[D \mid X, Z]$, `ml_r` for the projection of its in-sample predictions on $X$ | `partial="XZ"` |

With one instrument and the partial X variant, `score="IV-type"` is available too (`ml_g` learns
$E[Y - \theta D \mid X]$; default: a copy of `ml_l`).

=== "Python"

    ```python
    dml_rs.PLIV(ml_l, ml_m, ml_r).fit(X, y, d, z)        # z may have several columns
    dml_rs.PLIV(ml_l, ml_m, ml_r, ml_g=ml_g, score="IV-type").fit(X, y, d, z)
    dml_rs.PLIV(None, None, ml_r, partial="Z").fit(X, y, d, z)
    ```

=== "Rust"

    ```rust
    Pliv::new(ml_l, ml_m, ml_r).fit(&data.with_instruments(z)?)?;
    Pliv::new(ml_l, ml_m, ml_r).score(PlivScore::IvType).ml_g(ml_g).fit(&data.with_instrument(z1)?)?;
    Pliv::partial_z(ml_r).fit(&data.with_instruments(z)?)?;
    ```

## Interactive IV model (IIVM)

The local average treatment effect (LATE) of a binary treatment for compliers, with a binary
instrument. `ml_g` models $E[Y \mid Z, X]$, `ml_m` the instrument propensity
$P(Z = 1 \mid X)$ and `ml_r` $P(D = 1 \mid Z, X)$. Under one-sided non-compliance, switch off the
corresponding subgroup.

=== "Python"

    ```python
    dml_rs.IIVM(ml_g, ml_m, ml_r, subgroups={"always_takers": False}).fit(X, y, d, z)
    ```

=== "Rust"

    ```rust
    Iivm::new(ml_g, ml_m, ml_r).subgroups(false, true).fit(&data.with_instrument(z)?)?;
    ```

!!! note "normalize_ipw"
    With `normalize_ipw=True`, dml-rs normalises the weights $Z/m(X)$ and
    $(1-Z)/(1-m(X))$ with the instrument $Z$. DoubleML 0.11 used the treatment $D$; the DoubleML
    maintainers confirmed $Z$ is correct ([issue #411](https://github.com/DoubleML/doubleml-for-py/issues/411)).

## Difference-in-differences (DID, DIDCS)

The average treatment effect on the treated with two periods, under parallel trends conditional on
$X$ (Sant'Anna and Zhao, 2020).

- **Panel data** (`DID`): `y` is each unit's outcome change $Y_{\text{post}} - Y_{\text{pre}}$.
- **Repeated cross-sections** (`DIDCS`): `y` is the outcome level and `t` indicates the
  post-period.

`score="observational"` (doubly robust, needs the propensity learner `ml_m`) or
`"experimental"` (randomised treatment, no `ml_m`).

=== "Python"

    ```python
    dml_rs.DID(ml_g, ml_m).fit(X, delta_y, d)
    dml_rs.DIDCS(ml_g, ml_m).fit(X, y, d, t)
    ```

=== "Rust"

    ```rust
    Did::new(ml_g, ml_m).fit(&data)?;
    DidCs::new(ml_g, ml_m).fit(&data.with_time_indicator(t)?)?;
    ```

## Staggered difference-in-differences

With several cohorts first treated at different times, `StaggeredDID` estimates each group-time
effect $\text{ATT}(g, t)$ (Callaway and Sant'Anna, 2021) and aggregates them by cohort
(`"group"`), calendar time (`"time"`) or event time (`"eventstudy"`), with simultaneous confidence
bands. Controls are the never-treated (`control_group="never_treated"`) or the not-yet-treated
units; `anticipation_periods` shifts the reference period. Works with panels and, with
`panel=False`, repeated cross-sections.

=== "Python"

    ```python
    model = dml_rs.StaggeredDID(ml_g, ml_m, control_group="not_yet_treated")
    model.fit_frame(df, y_col="y", id_col="id", t_col="t", g_col="first_treated", x_cols=x_cols)
    model.aggregate("eventstudy").summary
    model.plot_effects()                                     # ATT(g, t) by cohort (matplotlib)
    model.fit_frame(df, "y", "id", "t", "first_treated", x_cols, cluster_col="state")   # clustered
    model.aggregate("eventstudy").plot_effects()
    model.sensitivity_analysis(cf_y=0.03, cf_d=0.03)        # every ATT(g, t)
    model.aggregate("group").aggregated_frameworks.sensitivity_analysis()
    model.sensitivity_benchmark(benchmarking_set=["x1"])
    ```

=== "Rust"

    ```rust
    let panel = PanelData::from_long(&ids, &periods, &y, &first_treated, &x)?;
    let fit = StaggeredDid::new(ml_g, ml_m).control_group(ControlGroup::NotYetTreated).fit(&panel)?;
    let event_study = fit.aggregate(Aggregation::EventStudy)?;
    ```

## Regression discontinuity (RDFlex)

The effect of a treatment assigned by a score crossing a cutoff, estimated at the cutoff with
flexible covariate adjustment (Noack, Olma and Rothe, 2024). Covariates $X$ do not change what is
estimated, but they can make the estimate much more precise. The outcome (and in a fuzzy design
the treatment) is adjusted by a cross-fitted prediction at the cutoff,

$$
\eta(X) = \tfrac{1}{2}\big(\mu(X;\ \text{left}) + \mu(X;\ \text{right})\big),
$$

learned with kernel weights around the cutoff. A local polynomial (by default linear) RD estimator is then run on
$Y - \eta(X)$, with a quadratic bias correction and robust confidence intervals (Calonico,
Cattaneo and Titiunik, 2014). The first-stage bandwidth is updated from the RD bandwidth over
`n_iterations` rounds.

- **Sharp design**: the treatment is $1\{\text{score} \ge \text{cutoff}\}$; only `ml_g` is needed.
- **Fuzzy design** (`fuzzy=True`): crossing the cutoff changes the treatment probability; `ml_m`
  adjusts the treatment, and the effect is the ratio of the two jumps.

Results have three rows: the conventional estimate, the bias-corrected estimate, and the
bias-corrected estimate with its robust standard error, which is the one to report. Learners must
accept sample weights; a binary outcome can be learned with a classifier.

The RD estimator is a port of the `rdrobust` package (version 2.1.1). Its options are passed as
keyword arguments, as in DoubleML:

- `p` and `q`: polynomial orders of the estimate and of the bias correction
- `kernel`
- `bwselect`: all ten bandwidth selectors (MSE- and CER-optimal; common, two-sided, sum and
  combinations)
- `vce`: `"nn"`, `"hc0"` to `"hc3"`, or the cluster-robust `"cr1"` to `"cr3"`
- `nnmatch`, `level`, `masspoints`, `bwcheck`, `bwrestrict`, `stdvars`, `scaleregul`, `sharpbw`,
  `rho`, `scalepar`
- `covs`: covariates of the RD stage
- `cluster`: labels for cluster-robust standard errors

With clusters the variance is cluster-robust, as in `rdrobust`: `cr1` by default, the
leverage-adjusted `cr2` (Bell and McCaffrey, 2002) with `vce="hc2"` or `"cr2"`, and `cr3`
(Pustejovsky and Tipton, 2018) with `vce="hc3"` or `"cr3"`. In Rust the back end is also
available on its own (`dml_core::rd::rd_estimate` and `rd_bandwidths` with `RdOptions`).

=== "Python"

    ```python
    model = dml_rs.RDFlex(ml_g, ml_m, fuzzy=True, cutoff=0.0).fit(X, y, d, score)
    model.summary                        # Conventional, Bias-Corrected, Robust
    dml_rs.RDFlex(ml_g, p=2, kernel="epa", bwselect="msetwo", covs=Z, cluster=school)
    dml_rs.RDFlex(LogisticRegression()).fit(X, y01, d, score)   # binary y
    ```

=== "Rust"

    ```rust
    let fit = RdFlex::fuzzy(ml_g, ml_m).cutoff(0.0).fit(&data, score.view())?;
    let options = RdOptions { p: 2, bwselect: BwSelect::Msetwo, ..RdOptions::default() };
    let fit = RdFlex::sharp(ml_g).rd_options(options).rd_clusters(school).fit(&data, score.view())?;
    ```

## Automatic debiased ML (AutoDML)

Many effects are a mean of a *linear functional* of the regression
$\gamma_0(D, X) = E[Y \mid D, X]$:

$$
\theta_0 = E[m(W; \gamma_0)], \qquad W = (D, X).
$$

The ATE is $m = \gamma(1, X) - \gamma(0, X)$, the average derivative $m = \partial\gamma(D, X) /
\partial D$, the effect of shifting everyone's treatment by $\delta$ is $m = \gamma(D + \delta, X) -
\gamma(D, X)$. The debiased score is

$$
\psi = m(W; \gamma) + \alpha(W)\,(Y - \gamma(W)) - \theta,
$$

where $\alpha$ is the *Riesz representer* of the functional: $E[m(W; f)] = E[\alpha(W) f(W)]$ for
every $f$. For the ATE, $\alpha$ is the familiar inverse propensity weight, but for most functionals
it has no simple formula. Automatic debiased machine learning (Chernozhukov, Newey and Singh,
2022) learns it from the functional alone, by minimising the Riesz loss
$E[\alpha(W)^2 - 2\, m(W; \alpha)]$, so any linear functional can be estimated without deriving
anything by hand.

`AutoDML` takes a regressor `ml_g` for $\gamma_0$ (fitted on $W$, the treatment first) and a
functional:

| Functional | $m(W; \gamma)$ | Python | Rust |
|---|---|---|---|
| Average treatment effect | $\gamma(1, X) - \gamma(0, X)$ | `"ate"`, `ATE()` | `Ate` |
| Average derivative | $\partial \gamma(D, X) / \partial D$ (central difference) | `"average_derivative"`, `AverageDerivative()` | `AverageDerivative` |
| Shift effect | $\gamma(D + \delta, X) - \gamma(D, X)$, optionally divided by $\delta$ | `ShiftEffect(delta, per_unit)` | `ShiftEffect` |
| Policy effect | $\gamma(d_{\text{new}}, X) - \gamma(D, X)$ | `PolicyEffect(new_d or function)` | `PolicyEffect` |
| Your own | $\sum_k c_k(W)\, \gamma(W_k)$ | `CustomFunctional(terms)` | `CustomFunctional`, or implement `LinearFunctional` |

The Riesz representer is learned by the Lasso of Chernozhukov, Newey and Singh (`RieszLasso`):
$\alpha(w) = b(w)'\rho$ for a dictionary $b$, with
$\hat\rho = \arg\min_\rho\ \rho'\hat G\rho - 2\hat M'\rho + 2r\sum_j \hat\psi_j|\rho_j|$, where
$\hat G = E_n[b(W)b(W)']$, $\hat M = E_n[m(W; b)]$ and $\hat\psi_j = \hat G_{jj}^{1/2}$ (the
constant is not penalised). The default dictionary holds all monomials of the covariates up to
degree 2, each also multiplied by the treatment (`degree`, `treatment_degree`); any basis can be
passed as `dictionary(W)`. The penalty $r$ is chosen by cross-validation of the Riesz loss, or
fixed with `penalty` (`0` gives the unpenalised minimum-distance solution). Both nuisances are
cross-fitted, and repeated splits, clustered data, the bootstrap, sensitivity analysis (with
$\sigma^2 = E[(Y - \gamma)^2]$ and $\nu^2 = E[2m(W; \alpha) - \alpha^2]$) and `cate` / `gate`
work as for the other models.

=== "Python"

    ```python
    # Average derivative of a continuous treatment (a smooth regression in D)
    dml_rs.AutoDML(LassoCV(), "average_derivative").fit(X, y, d)

    # Effect of raising everyone's treatment by 10%, with a forest
    model = dml_rs.AutoDML(RandomForestRegressor(), dml_rs.PolicyEffect(lambda W: 1.1 * W[:, 0]))
    model.fit(X, y, d)

    # A functional of your own: the ATE among units whose first covariate is positive
    def subgroup_ate(W):
        w1, w0 = W.copy(), W.copy()
        w1[:, 0], w0[:, 0] = 1, 0
        weight = (W[:, 1] > 0) / np.mean(W[:, 1] > 0)
        return [(w1, weight), (w0, -weight)]

    dml_rs.AutoDML(ml_g, dml_rs.CustomFunctional(subgroup_ate),
                   riesz=dml_rs.RieszLasso(degree=3)).fit(X, y, d)
    ```

=== "Rust"

    ```rust
    use dml_core::{AutoDml, AverageDerivative, Dictionary, RieszLasso, ShiftEffect};

    let fit = AutoDml::new(Lasso::new(0.01), AverageDerivative::new()).fit(&data)?;
    let shift = AutoDml::new(RandomForestRegressor::new(), ShiftEffect::new(0.5).per_unit(true))
        .riesz(RieszLasso::new().dictionary(Dictionary::Polynomial { degree: 3, treatment_degree: 2 }))
        .fit(&data)?;
    ```

!!! note "When to use it"
    For the ATE of a binary treatment, `IRM` (with a propensity model) and `AutoDML` give similar
    estimates in simulations; `AutoDML` is meant for functionals without a built-in model. The
    average derivative is a central difference, so it needs a regression that is smooth in the
    treatment (linear or polynomial models, neural networks); with
    forests use a shift effect instead. As for every DML estimator, valid confidence intervals
    need both the regression and the Riesz representer to be estimated well; with a misspecified
    regression the estimate stays consistent if the dictionary is rich enough, but the standard
    errors can be conservative.

## Causal forests

$$
Y = \tau(X)\,D + m(X) + \varepsilon,
$$

with an effect $\tau(x)$ that may vary arbitrarily with the covariates. `CausalForest`
estimates $\tau(x)$ at any point, with a confidence interval, following the generalized random
forests of Athey, Tibshirani and Wager (2019) as implemented in R's `grf::causal_forest`:

1. **Orthogonalisation.** `ml_y` and `ml_d` learn $E[Y \mid X]$ and $E[D \mid X]$ (a classifier
   for a binary treatment), cross-fitted with any learners; the forest works on the residuals
   $\tilde Y$, $\tilde D$.
2. **Honest trees.** Each tree takes a subsample; half of it chooses the splits and the other half
   estimates the leaves. Splits maximise the heterogeneity of the effect (grf's gradient-based
   split on the pseudo-outcomes of the parent's local effect), and every child keeps a share
   `alpha` of the parent's treatment variation.
3. **Estimation.** $\hat\tau(x)$ is a residual-on-residual regression weighted by how often each
   training point shares a leaf with $x$. Trees are grown in groups of `ci_group_size` on one
   half-sample, which gives the variance of $\hat\tau(x)$ (the bootstrap of little bags).

For the training rows, `effect_` holds out-of-bag estimates (each from the trees that did not see
the row). With a binary treatment, these feed doubly robust (AIPW) scores for the average effect
(`ate()`), best linear predictors (`blp`), group effects (`gate`) and grf's `test_calibration`,
which checks whether the forest's heterogeneity is real. `variable_importance_` is grf's
split-frequency importance. The defaults are grf's (2000 trees, subsamples of half the data,
`min_node_size=5`, `alpha=0.05`).

=== "Python"

    ```python
    cf = dml_rs.CausalForest(dml_rs.HistGradientBoostingRegressor(),
                             dml_rs.HistGradientBoostingClassifier()).fit(X, y, d)
    cf.effect(X_new)                  # τ̂(x)
    cf.effect_interval(X_new)         # 95% intervals
    cf.ate().summary                  # average effect from the doubly robust scores
    cf.gate(groups).summary           # group average effects
    cf.test_calibration()             # is the heterogeneity real?
    cf.variable_importance_
    ```

=== "Rust"

    ```rust
    use dml_core::{CausalForestDml, ForestSettings, TreatmentLearner};

    let fit = CausalForestDml::new(ml_y, TreatmentLearner::Classifier(Box::new(ml_d)))
        .forest(ForestSettings { num_trees: 2000, ..Default::default() })
        .fit(&data)?;
    let (tau, variance) = fit.effect(x_new.view())?;
    let ate = fit.ate()?;
    ```

`policy_tree(features, depth=2)` turns the doubly robust scores into a treatment rule: a shallow
tree of `features` that recommends treatment where the effect is positive (as `IRM.policy_tree`).

### Instrumental forests

When the treatment is endogenous but an instrument $Z$ moves it, `InstrumentalForest` (R's
`grf::instrumental_forest`) estimates the conditional local average treatment effect

$$
\theta(x) = \frac{\operatorname{Cov}(Z, Y \mid X = x)}{\operatorname{Cov}(Z, D \mid X = x)},
$$

the effect among compliers with covariates $x$ for a binary instrument. `ml_z` learns
$E[Z \mid X]$; the trees split on the heterogeneity of $\theta$, and each child keeps a share
`alpha` of the parent's instrument variation. With a binary instrument, `compliance_` holds
out-of-bag estimates of $E[D \mid Z = 1, X] - E[D \mid Z = 0, X]$, and `ate()`, `blp` and `gate`
use grf's doubly robust scores for the average of $\theta(X)$.

```python
iv = dml_rs.InstrumentalForest(ml_y, ml_d, ml_z).fit(X, y, d, z)
iv.effect(X_new), iv.effect_interval(X_new)
iv.ate().summary                      # average conditional LATE
```

### Multi-arm causal forests

`MultiArmCausalForest` (R's `grf::multi_arm_causal_forest`) takes a treatment of several discrete
levels and estimates the effect of every level against the smallest (the baseline),
$\tau_k(x) = E[Y(k) - Y(0) \mid X = x]$, in one forest: $\theta(x) = \operatorname{Var}(W \mid
x)^{-1} \operatorname{Cov}(W, Y \mid x)$ for the residualised arm indicators $W$. `ml_d` is a
classifier fitted for each level against the rest. `effect` returns one column per non-baseline
level, `ate()` one row per contrast, and `blp(basis, level)` / `gate(groups, level)` describe one
arm.

```python
ma = dml_rs.MultiArmCausalForest(ml_y, ml_d).fit(X, y, d)   # d in {0, 1, 2}
ma.effect(X_new)                      # (n, 2): effects of levels 1 and 2 against 0
ma.ate().summary
```

### Regression and quantile forests

The same engine gives two learners. `RegressionForest` (grf's `regression_forest`) is an honest
random forest for $E[Y \mid X]$ with variance estimates (`predict_with_variance`) and out-of-bag
predictions; it works as a nuisance learner in any model. `QuantileForest` (grf's
`quantile_forest`) predicts conditional quantiles from the forest weights, with splits chosen to
separate the quantiles given at construction (default 0.1, 0.5, 0.9):

```python
qf = dml_rs.QuantileForest(quantiles=[0.1, 0.5, 0.9]).fit(X, y)
qf.predict(X_new)                     # (n, 3)
qf.predict(X_new, quantiles=[0.25])   # any quantile
```

## Heterogeneous effects

- **Best linear predictor** (`cate`) and **group average treatment effects** (`gate`) give
  inference on how the effect varies (Semenova and Chernozhukov, 2021), for `PLR`, `IRM` (ATE
  score) and `DRLearner`; `APO.capo` and `APO.gapo` do the same for average potential outcomes.
  For PLR, the outcome residual is regressed on the basis scaled by the treatment residual, as in
  DoubleML.
- **Policy trees** (`IRM.policy_tree`) turn the estimated effects into a treatment rule: a shallow
  decision tree classifies the sign of each unit's doubly robust effect signal, weighted by its
  size, and is pruned by cost-complexity pruning (as DoubleML's `DoubleMLPolicyTree`; Athey and
  Wager, 2021). `print(tree)` shows the rules, `tree.predict(features)` the recommendations and
  `tree.plot_tree()` draws the tree (matplotlib).
- **Meta-learners** estimate the conditional average treatment effect $\tau(x)$ itself:
  `SLearner`, `TLearner`, `XLearner` (Künzel et al., 2019), `RLearner` (Nie and Wager, 2021) and
  `DRLearner` (Kennedy, 2023), the last with BLP and GATE inference.

=== "Python"

    ```python
    dr = dml_rs.DRLearner(outcome_model, propensity_model, effect_model).fit(X, y, d)
    dr.effect(X_new)      # estimated CATE
    dr.blp(basis).summary
    ```

=== "Rust"

    ```rust
    let fit = DrLearner::new(outcome, propensity, effect).fit(&data)?;
    let tau = fit.effect(x_new.view())?;
    ```

## Your own model

In Python, PLR, IRM, PLIV and IIVM take a callable `score` as DoubleML: a function of the data,
the cross-fitted nuisance predictions and the sample splits that returns the elements `psi_a`,
`psi_b` of a linear score $\psi = \psi_a \theta + \psi_b$. The nuisance models are fitted as for
the built-in score (PLR and PLIV fit `ml_g` only when it is given), and $\theta$, its standard
error and the aggregation over repetitions follow from the score as for the built-in ones.
Sensitivity analysis needs a built-in score.

```python
def ate_trimmed(y, d, g_hat0, g_hat1, m_hat, smpls):
    keep = (m_hat > 0.05) & (m_hat < 0.95)
    psi_b = keep * (g_hat1 - g_hat0 + d * (y - g_hat1) / m_hat - (1 - d) * (y - g_hat0) / (1 - m_hat))
    return -keep.astype(float), psi_b

dml_rs.IRM(ml_g, ml_m, score=ate_trimmed).fit(X, y, d)
```

The arguments are `(y, d, l_hat, m_hat, g_hat, smpls)` for PLR, `(y, d, g_hat0, g_hat1, m_hat,
smpls)` for IRM (`m_hat` clipped, and normalised with `normalize_ipw`), `(y, z, d, l_hat, m_hat,
r_hat, g_hat, smpls)` for PLIV (partial `X`, one instrument) and `(y, z, d, g_hat0, g_hat1, m_hat,
r_hat0, r_hat1, smpls)` for IIVM.

In Rust, any score that is linear in $\theta$ plugs into the same cross-fitting and inference
engine by implementing the `LinearScoreModel` trait; see the
[API documentation](https://docs.rs/dml-core/latest/dml_core/trait.LinearScoreModel.html).
Score elements computed elsewhere become a fit with `DmlFit::from_linear_scores`.

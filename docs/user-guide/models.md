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
Y = \theta D + g_0(X) + \varepsilon, \qquad Z = m_0(X) + V, \qquad E[\varepsilon \mid Z, X] = 0 .
$$

For an endogenous treatment with an instrument $Z$. Nuisances: `ml_l` for $E[Y \mid X]$, `ml_m` for
$E[Z \mid X]$, `ml_r` for $E[D \mid X]$.

=== "Python"

    ```python
    dml_rs.PLIV(ml_l, ml_m, ml_r).fit(X, y, d, z)
    ```

=== "Rust"

    ```rust
    Pliv::new(ml_l, ml_m, ml_r).fit(&data.with_instrument(z)?)?;
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
    ```

=== "Rust"

    ```rust
    let panel = PanelData::from_long(&ids, &periods, &y, &first_treated, &x)?;
    let fit = StaggeredDid::new(ml_g, ml_m).control_group(ControlGroup::NotYetTreated).fit(&panel)?;
    let event_study = fit.aggregate(Aggregation::EventStudy)?;
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
  Wager, 2021). `print(tree)` shows the rules, `tree.predict(features)` the recommendations.
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

In Rust, any score that is linear in $\theta$ plugs into the same cross-fitting and inference
engine by implementing the `LinearScoreModel` trait; see the
[API documentation](https://docs.rs/dml-core/latest/dml_core/trait.LinearScoreModel.html).

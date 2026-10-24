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

=== "Python"

    ```python
    model = dml_rs.IRM(ml_g, ml_m, score="ATE").fit(X, y, d)
    model.gate(groups)    # group average treatment effects
    model.cate(basis)     # best linear predictor of the CATE
    ```

=== "Rust"

    ```rust
    let fit = Irm::new(ml_g, ml_m).score(IrmScore::Ate).fit(&data)?;
    let gate = fit.gate(groups.view())?;
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

- **Best linear predictor** (`IRM.cate`) and **group average treatment effects** (`IRM.gate`,
  `DRLearner.gate`) give inference on how the effect varies (Semenova and Chernozhukov, 2021).
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

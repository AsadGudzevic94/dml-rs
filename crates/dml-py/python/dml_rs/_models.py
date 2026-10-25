"""scikit-learn style estimators on top of the Rust core."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

    # Folds: one repetition ``[(train, test), ...]`` or a list of repetitions.
    Folds = Sequence[Any]

from . import _dml_rs
from ._blp import BLP, blp_from
from ._policy import PolicyTree
from ._validation import _as_1d, _as_2d, _as_clusters


def _treatment_names(d):
    columns = getattr(d, "columns", None)
    if columns is not None:
        return [str(c) for c in columns]
    name = getattr(d, "name", None)  # pandas Series
    return [str(name)] if name is not None else None


def _weights_args(weights: Any) -> dict[str, Any]:
    """DoubleML's ``weights``: an array, or a dict with ``weights`` and ``weights_bar`` (n, n_rep)."""
    if weights is None:
        return {"weights": None, "weights_bar": None}
    if isinstance(weights, dict):
        if set(weights) != {"weights", "weights_bar"}:
            raise ValueError("a weights dict must have the keys 'weights' and 'weights_bar'")
        bar = np.ascontiguousarray(weights["weights_bar"], dtype=np.float64)
        if bar.ndim == 1:
            bar = bar.reshape(-1, 1)
        return {"weights": _as_1d(weights["weights"], "weights"), "weights_bar": bar}
    return {"weights": _as_1d(weights, "weights"), "weights_bar": None}


class _HeterogeneityMixin:
    """``cate`` and ``gate`` (best linear predictor inference on effect heterogeneity)."""

    def cate(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the CATE on ``basis``, as DoubleML's ``cate``."""
        return blp_from(self._check_fitted().cate(_as_2d(basis, "basis")), basis, "b")  # type: ignore[attr-defined]

    def gate(self, groups: ArrayLike) -> BLP:
        """Group average treatment effects for mutually exclusive 0/1 group dummies, as DoubleML's ``gate``."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")  # type: ignore[attr-defined]


def _rmse(y_true, y_pred):
    keep = ~np.isnan(y_true)
    return float(np.sqrt(np.mean((y_true[keep] - y_pred[keep]) ** 2)))


def _splits(folds):
    """Accept one repetition ``[(train, test), ...]`` or a list of repetitions."""
    if folds is None:
        return None
    folds = list(folds)
    first = folds[0]
    one_rep = len(first) == 2 and np.ndim(first[0]) == 1
    reps = [folds] if one_rep else folds
    return [[(np.asarray(tr, dtype=np.int64).tolist(), np.asarray(te, dtype=np.int64).tolist()) for tr, te in rep]
            for rep in reps]


class _Params:
    """scikit-learn parameter protocol (``get_params`` / ``set_params`` / ``clone``)."""

    @classmethod
    def _get_param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> Self:
        for key, value in params.items():
            if key not in self._get_param_names():
                raise ValueError(f"invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _check_fitted(self):
        if not hasattr(self, "result_"):
            raise AttributeError(f"{type(self).__name__} is not fitted yet; call fit() first")
        return self.result_


class _DMLEstimator(_Params):
    """Shared fitted-model API of the DML estimators."""

    _boot = None

    # -- fitted attributes ---------------------------------------------------
    def _store(self, result):
        self.result_ = result
        self.coef_ = result.coef
        self.se_ = result.se
        self.t_stat_ = result.t_stat
        self.pval_ = result.pval
        self.all_coef_ = result.all_coef
        self.all_se_ = result.all_se
        self.psi_ = result.psi
        self.treatment_names_ = result.treatment_names
        self.predictions_ = {name: result.predictions(name) for name in result.nuisance_names}
        # DoubleML's layout: one (n_rep, n_coefs) array per learner.
        self.nuisance_loss_ = {name: loss.T for name, loss in result.nuisance_losses().items()}
        self.n_clusters_ = result.n_clusters
        self._boot = None
        return self

    def evaluate_learners(
        self,
        learners: Optional[Sequence[str]] = None,
        metric: Optional[Callable[..., float]] = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Evaluate the cross-fitted nuisance predictions with ``metric(y_true, y_pred)``.

        As DoubleML: both arguments have shape ``(1, n)`` and ``y_true`` is ``NaN`` where a
        learner does not apply (e.g. ``ml_g0`` on treated units). The default metric is the
        RMSE over the non-missing targets. Returns ``{learner: array (n_rep, n_coefs)}``.
        """
        result = self._check_fitted()
        metric = _rmse if metric is None else metric
        if not callable(metric):
            raise TypeError(f"metric should be a callable, got {metric!r}")
        names = list(result.nuisance_losses()) if learners is None else list(learners)
        out = {}
        for name in names:
            targets = result.nuisance_targets(name)
            preds = self.predictions_[name]
            n_rep, k = targets.shape[1], targets.shape[2]
            res = np.full((n_rep, k), np.nan)
            for r in range(n_rep):
                for j in range(k):
                    y_true = targets[:, r, j].reshape(1, -1)
                    if np.all(np.isnan(y_true)):
                        continue
                    res[r, j] = metric(y_true=y_true, y_pred=preds[:, r, j].reshape(1, -1))
                    if not np.isfinite(res[r, j]):
                        raise ValueError(f"evaluation of learner {name!r} is not finite")
            out[name] = res
        return out

    def confint(self, level: float = 0.95, joint: bool = False) -> NDArray[np.float64]:
        """Confidence intervals, shape ``(n_treat, 2)``. ``joint=True`` needs `bootstrap`."""
        result = self._check_fitted()
        if joint:
            if self._boot is None:
                raise ValueError("call bootstrap() before confint(joint=True)")
            return self._boot.joint_confint(level)
        return result.confint(level)

    def bootstrap(self, method: str = "normal", n_boot: int = 500, random_state: int = 0) -> Self:
        """Multiplier bootstrap (``'normal'``, ``'Bayes'`` or ``'wild'``) for joint inference."""
        self._boot = self._check_fitted().bootstrap(method, n_boot, random_state)
        self.boot_t_stat_ = self._boot.t_stats
        return self

    def p_adjust(self, method: str = "romano-wolf") -> NDArray[np.float64]:
        """Adjusted p-values: ``'romano-wolf'`` (needs `bootstrap`), ``'holm'`` or ``'bonferroni'``."""
        result = self._check_fitted()
        if method.lower() in ("romano-wolf", "rw"):
            if self._boot is None:
                raise ValueError("call bootstrap() before p_adjust('romano-wolf')")
            return self._boot.romano_wolf()
        return result.p_adjust(method)

    # -- sensitivity analysis -------------------------------------------------
    def sensitivity_analysis(
        self,
        cf_y: float = 0.03,
        cf_d: float = 0.03,
        rho: float = 1.0,
        level: float = 0.95,
        null_hypothesis: Union[float, Sequence[float]] = 0.0,
    ) -> Self:
        """Bounds on the effect under omitted confounding of strength ``(cf_y, cf_d, rho)``.

        ``cf_y`` is the share of the outcome's residual variation explained by the confounder,
        ``cf_d`` the relative gain in the Riesz representer's variation, ``rho`` their correlation
        (adversarial at 1). Results are stored in ``sensitivity_params_`` (DoubleML's layout)
        and summarised by `sensitivity_summary`. Available for PLR, IRM and DID models.
        """
        null = np.atleast_1d(np.asarray(null_hypothesis, dtype=np.float64)).tolist()
        res = self._check_fitted().sensitivity_analysis(cf_y, cf_d, rho, level, null)
        self.sensitivity_result_ = res
        self.sensitivity_params_ = {
            "theta": {"lower": res.theta_lower, "upper": res.theta_upper},
            "se": {"lower": res.se_lower, "upper": res.se_upper},
            "ci": {"lower": res.ci_lower, "upper": res.ci_upper},
            "rv": res.rv,
            "rva": res.rva,
            "input": {"cf_y": cf_y, "cf_d": cf_d, "rho": rho, "level": level,
                      "null_hypothesis": np.asarray(res.params[4])},
        }
        return self

    @property
    def sensitivity_summary(self) -> str:
        """Text summary of the last `sensitivity_analysis`."""
        if not hasattr(self, "sensitivity_result_"):
            raise ValueError("call sensitivity_analysis() first")
        return str(self.sensitivity_result_)

    def sensitivity_benchmark(self, benchmarking_set: Sequence[Union[str, int]]) -> pd.DataFrame:
        """Calibrate ``cf_y``, ``cf_d`` and ``rho`` against observed covariates.

        Refits the model without the covariates in ``benchmarking_set`` (column names if ``X``
        was a DataFrame, else column indices) on the same sample splits and returns a table with
        ``cf_y``, ``cf_d``, ``rho`` and ``delta_theta`` per treatment, as DoubleML.
        """
        result = self._check_fitted()
        if not getattr(self, "_fit_inputs", None):
            raise ValueError("refit is not available for this estimator")
        X, rest, kwargs = self._fit_inputs
        columns = getattr(X, "columns", None)
        drop = [list(columns).index(c) if columns is not None and not isinstance(c, int) else int(c)
                for c in benchmarking_set]
        if not drop:
            raise ValueError("benchmarking_set must not be empty")
        X_arr = np.asarray(X, dtype=np.float64)
        keep = [i for i in range(X_arr.shape[1]) if i not in drop]
        short: Any = type(self)(**self.get_params())
        short.fit(X_arr[:, keep], *rest, **kwargs)
        bm = result.sensitivity_benchmark(short.result_)
        table = {"cf_y": bm.cf_y, "cf_d": bm.cf_d, "rho": bm.rho, "delta_theta": bm.delta_theta}
        try:
            import pandas as pd
        except ImportError:
            return {"index": result.treatment_names, **table}  # type: ignore[return-value]  # no pandas
        return pd.DataFrame(table, index=result.treatment_names)

    def _remember(self, X, *rest, **kwargs):
        """Keep the fit inputs so sensitivity_benchmark can refit without covariates."""
        self._fit_inputs = (X, rest, kwargs)

    @property
    def summary(self) -> pd.DataFrame:
        """Coefficient table as a pandas DataFrame (or a dict without pandas)."""
        result = self._check_fitted()
        ci = result.confint(0.95)
        table = {
            "coef": result.coef,
            "std err": result.se,
            "t": result.t_stat,
            "P>|t|": result.pval,
            "2.5 %": ci[:, 0],
            "97.5 %": ci[:, 1],
        }
        try:
            import pandas as pd
        except ImportError:
            return {"index": result.treatment_names, **table}  # type: ignore[return-value]  # no pandas
        return pd.DataFrame(table, index=result.treatment_names)

    def __str__(self):
        return str(self.result_) if hasattr(self, "result_") else repr(self)


class PLR(_HeterogeneityMixin, _DMLEstimator):
    """Partially linear regression ``Y = θ·D + g(X) + ε``.

    Parameters
    ----------
    ml_l : regressor for ``E[Y | X]`` — a ``dml_rs`` learner or any object with
        ``fit`` / ``predict`` (e.g. a scikit-learn estimator).
    ml_m : regressor for ``E[D | X]``.
    ml_g : regressor for ``g(X)``, used by ``score="IV-type"`` (default: a clone of ``ml_l``).
    score : ``"partialling out"`` or ``"IV-type"``.
    n_folds, n_rep : cross-fitting folds and repetitions of the sample split.
    random_state : seed for the sample splits.
    """

    def __init__(
        self,
        ml_l: Any,
        ml_m: Any,
        ml_g: Any = None,
        score: Literal["partialling out", "IV-type"] = "partialling out",
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_g = ml_g
        self.score = score
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        """Fit on covariates ``X``, outcome ``y`` and treatment(s) ``d``.

        ``folds`` optionally fixes the sample splits: a list of ``(train, test)``
        index pairs, or one such list per repetition.

        ``cluster`` gives cluster labels, one column (one-way) or two columns (two-way
        clustering, as Chiang et al. 2022): whole clusters are then assigned to folds and
        the standard errors are cluster-robust. Accepted by PLR, IRM, PLIV, IIVM, DID and
        DIDCS; the multiplier bootstrap is not available with clusters (as in DoubleML).
        """
        self._remember(X, y, d, folds=folds, cluster=cluster)
        result = _dml_rs.fit_plr(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_l, self.ml_m, self.ml_g,
            score=self.score, n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
        )
        return self._store(result)


class IRM(_HeterogeneityMixin, _DMLEstimator):
    """Interactive regression model for a binary treatment (ATE or ATTE).

    Parameters
    ----------
    ml_g : regressor for ``E[Y | D, X]`` (fitted separately on treated and controls).
    ml_m : classifier for the propensity ``P(D = 1 | X)`` with ``predict_proba``.
    score : ``"ATE"`` or ``"ATTE"``.
    clipping_threshold : propensities are clipped to ``[t, 1 - t]``.
    normalize_ipw : normalise the inverse-probability weights.
    weights : weighted average effect, as DoubleML: an array ``w`` (ATE score: ``E[w (g1 - g0)] /
        E[w]``; ATTE score: 0/1 weights selecting a subgroup of the treated) or a dict with
        ``weights`` and ``weights_bar`` of shape ``(n, n_rep)`` (ATE score).
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        score: Literal["ATE", "ATTE"] = "ATE",
        clipping_threshold: float = 1e-2,
        normalize_ipw: bool = False,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
        weights: Any = None,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.score = score
        self.clipping_threshold = clipping_threshold
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state
        self.weights = weights

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        self._remember(X, y, d, folds=folds, cluster=cluster)
        result = _dml_rs.fit_irm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            score=self.score, clipping_threshold=self.clipping_threshold, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            **_weights_args(self.weights),
        )
        return self._store(result)

    def policy_tree(
        self, features: ArrayLike, depth: int = 2, min_samples_leaf: int = 8, ccp_alpha: float = 0.01
    ) -> PolicyTree:
        """Policy tree recommending treatment where the estimated effect is positive (ATE score, one
        repetition), as DoubleML's ``policy_tree``: a decision tree of the sign of the orthogonal
        signal weighted by its size, pruned with cost-complexity ``ccp_alpha``."""
        native = self._check_fitted().policy_tree(_as_2d(features, "features"), depth, min_samples_leaf, ccp_alpha)
        return PolicyTree(native, features)


class IIVM(_DMLEstimator):
    """Interactive IV model: local average treatment effect (LATE) of a binary
    treatment ``d`` using a binary instrument ``z``.

    Parameters
    ----------
    ml_g : regressor for ``E[Y | Z, X]`` (fitted separately for ``Z = 0`` and ``Z = 1``).
    ml_m : classifier for the instrument propensity ``P(Z = 1 | X)``.
    ml_r : classifier for ``P(D = 1 | Z, X)`` (fitted separately for ``Z = 0`` and ``Z = 1``).
    subgroups : dict with ``always_takers`` / ``never_takers`` (default both True). Set one
        to False under one-sided non-compliance.
    clipping_threshold : instrument propensities are clipped to ``[t, 1 - t]``.
    normalize_ipw : normalise the inverse-probability weights within instrument arms.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        ml_r: Any,
        subgroups: Optional[dict[str, bool]] = None,
        clipping_threshold: float = 1e-2,
        normalize_ipw: bool = False,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_r = ml_r
        self.subgroups = subgroups
        self.clipping_threshold = clipping_threshold
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        z: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        subgroups = {"always_takers": True, "never_takers": True, **(self.subgroups or {})}
        result = _dml_rs.fit_iivm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_1d(z, "z"), self.ml_g, self.ml_m, self.ml_r,
            always_takers=subgroups["always_takers"], never_takers=subgroups["never_takers"],
            clipping_threshold=self.clipping_threshold, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
        )
        return self._store(result)


class _DIDBase(_DMLEstimator):
    def __init__(
        self,
        ml_g: Any,
        ml_m: Any = None,
        score: Literal["observational", "experimental"] = "observational",
        in_sample_normalization: bool = True,
        clipping_threshold: float = 1e-2,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.score = score
        self.in_sample_normalization = in_sample_normalization
        self.clipping_threshold = clipping_threshold
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def _fit(self, X, y, d, t, folds, cluster):
        if t is None:
            self._remember(X, y, d, folds=folds, cluster=cluster)
        else:
            self._remember(X, y, d, t, folds=folds, cluster=cluster)
        result = _dml_rs.fit_did(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            t=None if t is None else _as_1d(t, "t"), score=self.score,
            in_sample_normalization=self.in_sample_normalization, clipping_threshold=self.clipping_threshold,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
        )
        return self._store(result)


class DID(_DIDBase):
    """Doubly robust difference-in-differences (ATT) with **panel data** over two periods.

    ``y`` passed to `fit` is each unit's outcome change ``Y_post - Y_pre``.

    Parameters
    ----------
    ml_g : regressor for ``E[ΔY | D = d, X]`` (fitted per treatment group).
    ml_m : classifier for the propensity ``P(D = 1 | X)``; required for ``score="observational"``.
    score : ``"observational"`` (doubly robust) or ``"experimental"`` (randomised treatment).
    in_sample_normalization : normalise the weights by their sample means (DoubleML default).
    """

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        return self._fit(X, y, d, None, folds, cluster)


class DIDCS(_DIDBase):
    """Doubly robust difference-in-differences (ATT) with **repeated cross-sections**.

    Each row is observed once; ``t`` passed to `fit` is 1 in the post-period, 0 before.
    Parameters as `DID`, with ``ml_g`` fitted in each treatment-group x period cell.
    """

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        t: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        return self._fit(X, y, d, t, folds, cluster)


class DIDAggregation:
    """Aggregated staggered-DID effects (by cohort, calendar time or event time)."""

    def __init__(self, how: str, effects: Any, overall: Any, weights: Any, overall_weights: Any) -> None:
        self.aggregation = how
        self._effects = effects
        self._overall = overall
        self.weights = weights
        self.overall_weights = overall_weights

    coef = property(lambda self: self._effects.coef)
    se = property(lambda self: self._effects.se)
    names = property(lambda self: self._effects.treatment_names)
    overall_coef = property(lambda self: float(self._overall.coef[0]))
    overall_se = property(lambda self: float(self._overall.se[0]))

    def confint(
        self, level: float = 0.95, joint: bool = False, n_boot: int = 500, random_state: int = 0
    ) -> NDArray[np.float64]:
        """Intervals for the aggregated effects; ``joint=True`` gives simultaneous bands."""
        if joint:
            return self._effects.bootstrap("normal", n_boot, random_state).joint_confint(level)
        return self._effects.confint(level)

    @property
    def summary(self) -> pd.DataFrame:
        return _summary_table(self._effects)

    @property
    def overall_summary(self) -> pd.DataFrame:
        return _summary_table(self._overall)

    def __str__(self):
        return f"{self._effects}\n{self._overall}"


def _summary_table(result):
    ci = result.confint(0.95)
    table = {"coef": result.coef, "std err": result.se, "t": result.t_stat, "P>|t|": result.pval,
             "2.5 %": ci[:, 0], "97.5 %": ci[:, 1]}
    try:
        import pandas as pd
    except ImportError:
        return {"index": result.treatment_names, **table}
    return pd.DataFrame(table, index=result.treatment_names)


class StaggeredDID(_DMLEstimator):
    """Difference-in-differences with staggered adoption (Callaway & Sant'Anna, 2021),
    as DoubleML's ``DoubleMLDIDMulti`` for panel data.

    Estimates every group-time effect ATT(g, t) by a doubly robust two-period DID and
    aggregates them with `aggregate`.

    Parameters
    ----------
    ml_g : regressor for the control group's outcome change given ``X``.
    ml_m : classifier for the cohort propensity; required for ``score="observational"``.
    control_group : ``"never_treated"`` or ``"not_yet_treated"``.
    anticipation_periods : periods before treatment in which units may already react.
    gt_combinations : ``"standard"`` (varying base), ``"all"`` or ``"universal"`` (fixed base).
    score, in_sample_normalization, clipping_threshold : as `DID`.
    panel : ``True`` for panel data (rows linked by ``id``); ``False`` for repeated
        cross-sections, where every row is a separate observation.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any = None,
        control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",
        anticipation_periods: int = 0,
        gt_combinations: Literal["standard", "all"] = "standard",
        score: Literal["observational", "experimental"] = "observational",
        in_sample_normalization: bool = True,
        clipping_threshold: float = 1e-2,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
        panel: bool = True,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.control_group = control_group
        self.anticipation_periods = anticipation_periods
        self.gt_combinations = gt_combinations
        self.score = score
        self.in_sample_normalization = in_sample_normalization
        self.clipping_threshold = clipping_threshold
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state
        self.panel = panel

    def fit(self, X: ArrayLike, y: ArrayLike, id: ArrayLike, t: ArrayLike, g: ArrayLike) -> Self:
        """Fit on long-format data: one row per observation with period ``t`` and cohort
        ``g`` (first treated period; ``np.inf`` or NaN for never treated). For panels, rows
        are linked by ``id`` and ``X`` must be time-invariant; for repeated cross-sections
        (``panel=False``) ``id`` is ignored."""
        self.staggered_result_ = _dml_rs.fit_staggered_did(
            _as_1d(y, "y"), np.ascontiguousarray(id, dtype=np.int64), _as_1d(t, "t"), _as_1d(g, "g"),
            _as_2d(X, "X"), self.ml_g, self.ml_m, control_group=self.control_group,
            anticipation_periods=self.anticipation_periods, gt_combinations=self.gt_combinations,
            score=self.score, in_sample_normalization=self.in_sample_normalization,
            clipping_threshold=self.clipping_threshold, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, panel=self.panel,
        )
        self.group_times_ = self.staggered_result_.group_times
        return self._store(self.staggered_result_.att)

    def fit_frame(
        self, df: pd.DataFrame, y_col: str, id_col: str, t_col: str, g_col: str, x_cols: Sequence[str]
    ) -> Self:
        """Convenience wrapper for a pandas DataFrame in long format."""
        return self.fit(df[list(x_cols)].to_numpy(), df[y_col].to_numpy(), df[id_col].to_numpy(),
                        df[t_col].to_numpy(), df[g_col].to_numpy())

    def aggregate(self, aggregation: Literal["group", "time", "eventstudy"] = "group") -> DIDAggregation:
        """Aggregate the ATT(g, t): ``"group"``, ``"time"`` or ``"eventstudy"``."""
        if not hasattr(self, "staggered_result_"):
            raise AttributeError("StaggeredDID is not fitted yet; call fit() first")
        return DIDAggregation(aggregation, *self.staggered_result_.aggregate(aggregation))


class PLIV(_DMLEstimator):
    """Partially linear IV regression with one instrument ``z``.

    Parameters
    ----------
    ml_l : regressor for ``E[Y | X]``.
    ml_m : regressor for ``E[Z | X]``.
    ml_r : regressor for ``E[D | X]``.
    """

    def __init__(
        self,
        ml_l: Any,
        ml_m: Any,
        ml_r: Any,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_r = ml_r
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        z: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        result = _dml_rs.fit_pliv(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_1d(z, "z"), self.ml_l, self.ml_m, self.ml_r,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
        )
        return self._store(result)


class APO(_DMLEstimator):
    """Average potential outcome ``E[Y(d)]`` of one level ``d`` of a discrete treatment.

    Default sample splits are stratified by the treatment, as in DoubleML.

    Parameters
    ----------
    ml_g : regressor for ``E[Y | D = d, X]``.
    ml_m : classifier for the generalised propensity ``P(D = d | X)``.
    treatment_level : the level ``d``.
    clipping_threshold : propensities are clipped to ``[t, 1 - t]``.
    normalize_ipw : normalise the inverse-probability weights.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        treatment_level: float,
        clipping_threshold: float = 1e-2,
        normalize_ipw: bool = False,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
        weights: Any = None,
    ) -> None:
        self.weights = weights
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.treatment_level = treatment_level
        self.clipping_threshold = clipping_threshold
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        self._remember(X, y, d, folds=folds, cluster=cluster)
        result = _dml_rs.fit_apo(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            float(self.treatment_level), clipping_threshold=self.clipping_threshold,
            normalize_ipw=self.normalize_ipw, n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            **_weights_args(self.weights),
        )
        return self._store(result)

    def capo(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the conditional average potential outcome, as DoubleML's ``capo``."""
        return blp_from(self._check_fitted().cate(_as_2d(basis, "basis")), basis, "b")

    def gapo(self, groups: ArrayLike) -> BLP:
        """Group average potential outcomes for 0/1 group dummies, as DoubleML's ``gapo``."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")


class CausalContrast(_DMLEstimator):
    """Contrasts between average potential outcomes, from `APOS.causal_contrast`.

    Offers the fitted-model API (``summary``, ``confint``, ``bootstrap``, ``p_adjust``,
    ``sensitivity_analysis``); there is nothing to refit.
    """

    def __init__(self, result: Any) -> None:
        self._store(result)

    @classmethod
    def _get_param_names(cls) -> list[str]:
        return []


class APOS(_DMLEstimator):
    """Average potential outcomes of several levels of a discrete treatment, and contrasts.

    All levels are fitted on the same sample splits, so their estimates and contrasts have joint
    inference (`bootstrap`, ``confint(joint=True)``). Parameters as `APO`, with
    ``treatment_levels`` the levels to estimate.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        treatment_levels: Sequence[float],
        clipping_threshold: float = 1e-2,
        normalize_ipw: bool = False,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
        weights: Any = None,
    ) -> None:
        self.weights = weights
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.treatment_levels = treatment_levels
        self.clipping_threshold = clipping_threshold
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        self._remember(X, y, d, folds=folds, cluster=cluster)
        levels = [float(v) for v in np.atleast_1d(np.asarray(self.treatment_levels, dtype=np.float64))]
        self._apos = _dml_rs.fit_apos(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), self.ml_g, self.ml_m, levels,
            clipping_threshold=self.clipping_threshold, normalize_ipw=self.normalize_ipw, n_folds=self.n_folds,
            n_rep=self.n_rep, seed=self.random_state, splits=_splits(folds), cluster=_as_clusters(cluster),
            **_weights_args(self.weights),
        )
        return self._store(self._apos.levels)

    def causal_contrast(self, reference_levels: Union[float, Sequence[float]]) -> CausalContrast:
        """Average effects of each level against each reference level, as DoubleML's ``causal_contrast``.

        For every reference level in turn, all levels not yet used as a reference are compared with
        it (``"d vs d_ref"``). Inference accounts for the correlation between the estimates; with
        clustered data the standard errors are cluster-robust.
        """
        self._check_fitted()
        refs = [float(v) for v in np.atleast_1d(np.asarray(reference_levels, dtype=np.float64))]
        return CausalContrast(self._apos.causal_contrast(refs))

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


def _external_args(external: Any, d: Any) -> Optional[list[tuple[int, str, Any]]]:
    """DoubleML's ``external_predictions``: ``{treatment: {learner: array (n_obs, n_rep)}}``, keyed by
    treatment name or index (a single treatment may also be given as ``{learner: array}``)."""
    if external is None:
        return None
    names = _treatment_names(d)
    n_treat = 1 if np.ndim(np.asarray(d)) == 1 else np.asarray(d).shape[1]
    names = names if names is not None else (["d"] if n_treat == 1 else [f"d{j}" for j in range(n_treat)])
    if all(not isinstance(v, dict) for v in external.values()):
        external = {0: external}
    out = []
    for key, learners in external.items():
        j = key if isinstance(key, (int, np.integer)) else names.index(str(key))
        for name, preds in learners.items():
            arr = np.ascontiguousarray(preds, dtype=np.float64)
            out.append((int(j), str(name), arr.reshape(-1, 1) if arr.ndim == 1 else arr))
    return out


class _HeterogeneityMixin:
    """``cate`` and ``gate`` (best linear predictor inference on effect heterogeneity)."""

    def cate(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the CATE on ``basis``, as DoubleML's ``cate``."""
        return blp_from(self._check_fitted().cate(_as_2d(basis, "basis")), basis, "b")  # type: ignore[attr-defined]

    def gate(self, groups: ArrayLike) -> BLP:
        """Group average treatment effects for mutually exclusive 0/1 group dummies, as DoubleML's ``gate``."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")  # type: ignore[attr-defined]


def _is_classifier(learner: Any) -> bool:
    """A learner predicting class probabilities (scikit-learn's ``predict_proba``)."""
    return hasattr(learner, "predict_proba")


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

    def sensitivity_contours(
        self,
        idx_treatment: int = 0,
        value: Literal["theta", "ci"] = "theta",
        rho: float = 1.0,
        level: float = 0.95,
        grid_bounds: tuple[float, float] = (0.15, 0.15),
        grid_size: int = 100,
    ) -> dict[str, Any]:
        """The data of `sensitivity_plot`, as DoubleML's ``sensitivity_plot``: the lower (or, when the
        null hypothesis exceeds the estimate, upper) bound of the effect (``value="theta"``) or of its
        confidence interval (``"ci"``) on a grid of ``cf_d`` in ``[0, grid_bounds[0]]`` and ``cf_y`` in
        ``[0, grid_bounds[1]]``. Call `sensitivity_analysis` first; its null hypothesis decides the
        side. ``values[i, j]`` belongs to ``cf_d[i]``, ``cf_y[j]``."""
        result = self._check_fitted()
        if not hasattr(self, "sensitivity_result_"):
            raise ValueError("call sensitivity_analysis() before sensitivity_contours()")
        if value not in ("theta", "ci"):
            raise ValueError(f"value must be 'theta' or 'ci', got {value!r}")
        if not all(0 < b < 1 for b in grid_bounds) or grid_size < 10:
            raise ValueError("grid_bounds must lie in (0, 1) and grid_size be at least 10")
        params = self.sensitivity_params_["input"]
        nulls = np.atleast_1d(params["null_hypothesis"]).astype(float).tolist()
        null = nulls[idx_treatment] if len(nulls) > 1 else nulls[0]
        theta = float(self.coef_[idx_treatment])
        bound = "upper" if null > theta else "lower"
        cf_d = np.linspace(0, grid_bounds[0], grid_size)
        cf_y = np.linspace(0, grid_bounds[1], grid_size)
        values = result.sensitivity_grid(cf_y.tolist(), cf_d.tolist(), rho, level, nulls, idx_treatment, value, bound)
        if value == "theta":
            unadjusted = theta
        else:
            ci = result.confint(params["level"])
            unadjusted = float(ci[idx_treatment, 1 if bound == "upper" else 0])
        return {"cf_d": cf_d, "cf_y": cf_y, "values": values, "bound": bound, "unadjusted": unadjusted}

    def sensitivity_plot(
        self,
        idx_treatment: int = 0,
        value: Literal["theta", "ci"] = "theta",
        rho: float = 1.0,
        level: float = 0.95,
        include_scenario: bool = True,
        benchmarks: Optional[dict[str, Sequence[Any]]] = None,
        fill: bool = True,
        grid_bounds: tuple[float, float] = (0.15, 0.15),
        grid_size: int = 100,
        ax: Any = None,
    ) -> Any:
        """Contour plot of the effect's bound over confounding strengths ``cf_d`` (x) and ``cf_y``
        (y), as DoubleML's ``sensitivity_plot``, with the scenario of `sensitivity_analysis` and
        optional ``benchmarks`` (``{"cf_y": [...], "cf_d": [...], "name": [...]}``) marked. Draws with
        matplotlib (installed separately) and returns the axes; `sensitivity_contours` gives the
        data for other plotting libraries."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError("sensitivity_plot needs matplotlib: pip install matplotlib") from err
        data = self.sensitivity_contours(idx_treatment, value, rho, level, grid_bounds, grid_size)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        draw = ax.contourf if fill else ax.contour
        cs = draw(data["cf_d"], data["cf_y"], data["values"].T, levels=20, cmap="RdBu_r" if fill else None)
        ax.clabel(ax.contour(data["cf_d"], data["cf_y"], data["values"].T, levels=8, colors="k",
                             linewidths=0.5), fontsize=7)
        if fill:
            ax.figure.colorbar(cs, ax=ax, label=f"{data['bound']} bound ({value})")
        ax.plot(0, 0, "o", color="k")
        ax.annotate(f"unadjusted: {data['unadjusted']:.3f}", (0, 0), xytext=(4, 4), textcoords="offset points")
        if include_scenario:
            inp = self.sensitivity_params_["input"]
            point = self.sensitivity_params_[value][data["bound"]][idx_treatment]
            ax.plot(inp["cf_d"], inp["cf_y"], "^", color="k")
            ax.annotate(f"scenario: {point:.3f}", (inp["cf_d"], inp["cf_y"]), xytext=(4, 4),
                        textcoords="offset points")
        if benchmarks is not None:
            result = self._check_fitted()
            inp = self.sensitivity_params_["input"]
            nulls = np.atleast_1d(inp["null_hypothesis"]).astype(float).tolist()
            for y_b, d_b, name in zip(benchmarks["cf_y"], benchmarks["cf_d"], benchmarks["name"]):
                v = result.sensitivity_grid([float(y_b)], [float(d_b)], inp["rho"], inp["level"], nulls,
                                            idx_treatment, value, data["bound"])[0, 0]
                ax.plot(d_b, y_b, "D", color="k")
                ax.annotate(f"{name}: {v:.3f}", (d_b, y_b), xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("cf_d")
        ax.set_ylabel("cf_y")
        return ax

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
        external_predictions: Optional[dict[Any, Any]] = None,
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
            external_predictions=_external_args(external_predictions, d),
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
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
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
        self.calibration = calibration
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
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        self._remember(X, y, d, folds=folds, cluster=cluster)
        result = _dml_rs.fit_irm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            score=self.score, clipping_threshold=self.clipping_threshold, calibration=self.calibration, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            **_weights_args(self.weights),
            external_predictions=_external_args(external_predictions, d),
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
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
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
        self.calibration = calibration
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
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        subgroups = {"always_takers": True, "never_takers": True, **(self.subgroups or {})}
        result = _dml_rs.fit_iivm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_1d(z, "z"), self.ml_g, self.ml_m, self.ml_r,
            always_takers=subgroups["always_takers"], never_takers=subgroups["never_takers"],
            clipping_threshold=self.clipping_threshold, calibration=self.calibration, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            external_predictions=_external_args(external_predictions, d),
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
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.score = score
        self.in_sample_normalization = in_sample_normalization
        self.clipping_threshold = clipping_threshold
        self.calibration = calibration
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def _fit(self, X, y, d, t, folds, cluster, external_predictions=None):
        if t is None:
            self._remember(X, y, d, folds=folds, cluster=cluster)
        else:
            self._remember(X, y, d, t, folds=folds, cluster=cluster)
        result = _dml_rs.fit_did(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            t=None if t is None else _as_1d(t, "t"), score=self.score,
            in_sample_normalization=self.in_sample_normalization, clipping_threshold=self.clipping_threshold, calibration=self.calibration,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            external_predictions=_external_args(external_predictions, d),
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
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        return self._fit(X, y, d, None, folds, cluster, external_predictions)


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
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        return self._fit(X, y, d, t, folds, cluster, external_predictions)


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
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
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
        self.calibration = calibration
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
            clipping_threshold=self.clipping_threshold, calibration=self.calibration, n_folds=self.n_folds, n_rep=self.n_rep,
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
    """Partially linear IV regression with one or several instruments ``z`` (one column each), as
    DoubleML's ``DoubleMLPLIV``.

    With ``partial="X"`` (default) ``X`` is partialled out of ``Y``, ``D`` and every instrument;
    several instruments are combined by projecting the treatment residual on the instrument
    residuals (two-stage least squares). ``partial="Z"`` uses ``E[D | X, Z]`` (``ml_r``) as
    instrument, and ``partial="XZ"`` partials out both (DoubleML's ``_partialZ`` / ``_partialXZ``).

    Parameters
    ----------
    ml_l : regressor for ``E[Y | X]`` (not used with ``partial="Z"``).
    ml_m : regressor for ``E[Z | X]``; with ``partial="XZ"`` for ``E[D | X, Z]`` (not used with ``"Z"``).
    ml_r : regressor for ``E[D | X]``; with ``partial="Z"`` for ``E[D | X, Z]``; with ``"XZ"`` for
        the projection of ``ml_m``'s in-sample predictions on ``X``.
    ml_g : regressor for ``E[Y - θD | X]``, used by ``score="IV-type"`` (default: a copy of ``ml_l``).
    score : ``"partialling out"`` or ``"IV-type"`` (one instrument, ``partial="X"``).
    partial : ``"X"``, ``"Z"`` or ``"XZ"``.
    """

    def __init__(
        self,
        ml_l: Any,
        ml_m: Any,
        ml_r: Any,
        ml_g: Any = None,
        score: Literal["partialling out", "IV-type"] = "partialling out",
        partial: Literal["X", "Z", "XZ"] = "X",
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_r = ml_r
        self.ml_g = ml_g
        self.score = score
        self.partial = partial
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
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        result = _dml_rs.fit_pliv(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_2d(z, "z"), self.ml_l, self.ml_m, self.ml_r,
            ml_g=self.ml_g, score=self.score, partial=self.partial,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            external_predictions=_external_args(external_predictions, d),
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
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
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
        self.calibration = calibration
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
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        self._remember(X, y, d, folds=folds, cluster=cluster)
        result = _dml_rs.fit_apo(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            float(self.treatment_level), clipping_threshold=self.clipping_threshold, calibration=self.calibration,
            normalize_ipw=self.normalize_ipw, n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
            **_weights_args(self.weights),
            external_predictions=_external_args(external_predictions, d),
        )
        return self._store(result)

    def capo(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the conditional average potential outcome, as DoubleML's ``capo``."""
        return blp_from(self._check_fitted().cate(_as_2d(basis, "basis")), basis, "b")

    def gapo(self, groups: ArrayLike) -> BLP:
        """Group average potential outcomes for 0/1 group dummies, as DoubleML's ``gapo``."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")


class _ResultView(_DMLEstimator):
    """The fitted-model API over an existing result; there is nothing to refit."""

    def __init__(self, result: Any) -> None:
        self._store(result)

    @classmethod
    def _get_param_names(cls) -> list[str]:
        return []


class CausalContrast(_ResultView):
    """Contrasts between average potential outcomes, from `APOS.causal_contrast`.

    Offers the fitted-model API (``summary``, ``confint``, ``bootstrap``, ``p_adjust``,
    ``sensitivity_analysis``); there is nothing to refit.
    """


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
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
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
        self.calibration = calibration
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
            clipping_threshold=self.clipping_threshold, calibration=self.calibration, normalize_ipw=self.normalize_ipw, n_folds=self.n_folds,
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


def _codes(values: ArrayLike, name: str) -> NDArray[np.int64]:
    """Integer codes in sorted order of the distinct values (numbers, strings or dates)."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return np.unique(arr, return_inverse=True)[1].astype(np.int64)


class PLPR(_DMLEstimator):
    """Partially linear regression for static panels, as DoubleML's ``DoubleMLPLPR`` (Clarke and
    Polselli, 2025): ``Y_it = θ·D_it + g(X_it) + α_i + U_it`` with unit effects correlated with the
    covariates and the treatment.

    The panel (long format, one row per unit ``id`` and period ``t``) is transformed according to
    ``approach``, and a partially linear regression is fitted on the transformed rows with folds of
    whole units and unit-clustered standard errors.

    Parameters
    ----------
    ml_l : regressor for ``E[Y | X]`` on the transformed panel.
    ml_m : regressor for ``E[D | X]``, or a classifier for a binary treatment with the correlated
        random effects approaches.
    ml_g : regressor for ``E[Y - θD | X]``, used by ``score="IV-type"`` (default: a copy of ``ml_l``).
    score : ``"partialling out"`` or ``"IV-type"``.
    approach : ``"fd_exact"`` (first differences, covariates and their lags), ``"wg_approx"``
        (within-unit demeaning), ``"cre_general"`` or ``"cre_normal"`` (correlated random effects:
        unit means of the covariates added).
    """

    def __init__(
        self,
        ml_l: Any,
        ml_m: Any,
        ml_g: Any = None,
        score: Literal["partialling out", "IV-type"] = "partialling out",
        approach: Literal["cre_general", "cre_normal", "fd_exact", "wg_approx"] = "fd_exact",
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_g = ml_g
        self.score = score
        self.approach = approach
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        id: ArrayLike,
        t: ArrayLike,
        folds: Optional[Folds] = None,
        external_predictions: Optional[dict[Any, Any]] = None,
    ) -> Self:
        """``folds`` (optional) index the rows of the transformed panel, sorted by unit and period,
        as DoubleML's ``smpls``; each test set should hold whole units."""
        result = _dml_rs.fit_plpr(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _codes(id, "id"),
            _codes(t, "t").astype(np.float64), self.ml_l, self.ml_m, _is_classifier(self.ml_m), self.ml_g,
            score=self.score, approach=self.approach, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, splits=_splits(folds), treatment_names=_treatment_names(d),
            external_predictions=_external_args(external_predictions, d),
        )
        return self._store(result)


class RDFlex(_Params):
    """Regression discontinuity design with flexible covariate adjustment, as DoubleML's ``RDFlex``
    (Noack, Olma and Rothe, 2024).

    The outcome (and in a fuzzy design the treatment) is adjusted by a cross-fitted prediction from
    the covariates at the cutoff, learned with kernel weights around it; a local linear RD estimator
    (a port of ``rdrobust``: bias correction, robust intervals) is then run on the adjusted outcome.
    The first-stage bandwidth is updated from the RD bandwidth over ``n_iterations`` rounds. Fitted
    attributes hold three rows: conventional, bias-corrected and robust estimates.

    Parameters
    ----------
    ml_g : regressor for the outcome; must accept ``sample_weight``.
    ml_m : classifier for the treatment (fuzzy design only); must accept ``sample_weight``.
    fuzzy : fuzzy design (the treatment may differ from ``score >= cutoff``).
    cutoff : the cutoff in the score.
    h_fs : initial first-stage bandwidth (default: ``rdbwselect`` without covariates).
    fs_specification : ``"cutoff"``, ``"cutoff and score"`` or ``"interacted cutoff and score"``.
    fs_kernel : ``"triangular"``, ``"uniform"`` or ``"epanechnikov"``.
    n_iterations : rounds of first-stage fitting and bandwidth updates.
    """

    _ROWS = ["Conventional", "Bias-Corrected", "Robust"]

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any = None,
        fuzzy: bool = False,
        cutoff: float = 0.0,
        h_fs: Optional[float] = None,
        fs_specification: Literal["cutoff", "cutoff and score", "interacted cutoff and score"] = "cutoff",
        fs_kernel: Literal["triangular", "uniform", "epanechnikov"] = "triangular",
        n_iterations: int = 2,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.fuzzy = fuzzy
        self.cutoff = cutoff
        self.h_fs = h_fs
        self.fs_specification = fs_specification
        self.fs_kernel = fs_kernel
        self.n_iterations = n_iterations
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, score: ArrayLike, folds: Optional[Folds] = None) -> Self:
        if _is_classifier(self.ml_g):
            raise TypeError("ml_g must be a regressor")
        if self.fuzzy and self.ml_m is None:
            raise ValueError("a fuzzy design needs a classifier ml_m for the treatment")
        result = _dml_rs.fit_rdflex(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), _as_1d(score, "score"), self.ml_g,
            self.ml_m if self.fuzzy else None, cutoff=float(self.cutoff),
            h_fs=None if self.h_fs is None else float(self.h_fs), fs_specification=self.fs_specification,
            fs_kernel=self.fs_kernel, n_iterations=self.n_iterations, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, splits=_splits(folds),
        )
        self.result_ = result
        self.coef_ = result.coef
        self.se_ = result.se
        self.t_stat_ = result.t_stat
        self.pval_ = result.pval
        self.ci_ = result.ci
        self.all_coef_ = result.all_coef
        self.all_se_ = result.all_se
        self.h_ = result.h
        self.b_ = result.b
        self.h_fs_ = result.h_fs
        return self

    def confint(self, level: float = 0.95) -> NDArray[np.float64]:
        """Intervals at ``level`` (rows: conventional, bias-corrected, robust), as DoubleML's ``confint``."""
        return self._check_fitted().confint(level)

    @property
    def summary(self) -> pd.DataFrame:
        """Coefficient table (rows: conventional, bias-corrected, robust)."""
        result = self._check_fitted()
        ci = result.ci
        table = {"coef": result.coef, "std err": result.se, "t": result.t_stat, "P>|t|": result.pval,
                 "2.5 %": ci[:, 0], "97.5 %": ci[:, 1]}
        try:
            import pandas as pd
        except ImportError:
            return {"index": self._ROWS, **table}  # type: ignore[return-value]  # no pandas
        return pd.DataFrame(table, index=self._ROWS)

    def __str__(self):
        return str(self.result_) if hasattr(self, "result_") else repr(self)


class LPLR(_DMLEstimator):
    """Logistic partially linear regression for a binary outcome, as DoubleML's ``DoubleMLLPLR``
    (Liu, Zhang and Zhou, 2021): ``P(Y = 1 | D, X) = expit(θ·D + r(X))``, so ``θ`` is a log odds
    ratio.

    ``ml_M`` and ``ml_a`` are double cross-fitted (inner folds within every outer training fold),
    and the score is solved with Newton's method from a preliminary estimate.

    Parameters
    ----------
    ml_M : classifier for ``P(Y = 1 | D, X)``.
    ml_t : regressor for the log odds ``E[logit M(D, X) | X]``.
    ml_m : regressor for ``E[D | X]``, or a classifier for a binary treatment.
    ml_a : learner for ``E[D | X]`` used for the preliminary estimate (default: a copy of ``ml_m``).
    score : ``"nuisance_space"`` (``ml_m`` fitted on ``Y = 0``) or ``"instrument"`` (``ml_m`` fitted
        with weights ``M(1 - M)``; the learner must accept ``sample_weight``).
    n_folds_inner : inner folds of the double cross-fitting.
    """

    def __init__(
        self,
        ml_M: Any,
        ml_t: Any,
        ml_m: Any,
        ml_a: Any = None,
        score: Literal["nuisance_space", "instrument"] = "nuisance_space",
        n_folds: int = 5,
        n_folds_inner: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_M = ml_M
        self.ml_t = ml_t
        self.ml_m = ml_m
        self.ml_a = ml_a
        self.score = score
        self.n_folds = n_folds
        self.n_folds_inner = n_folds_inner
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        inner_folds: Optional[Sequence[Any]] = None,
    ) -> Self:
        """``inner_folds`` (with ``folds``) fixes the inner splits in DoubleML's ``smpls_inner``
        layout: per repetition, per outer fold, a list of ``(train, test)`` index pairs within that
        fold's training set."""
        splits = _splits(folds)
        inner = None
        if inner_folds is not None:
            if splits is None:
                raise ValueError("inner_folds requires folds")
            inner = [[_splits(fold)[0] for fold in rep] for rep in inner_folds]  # type: ignore[index]
        result = _dml_rs.fit_lplr(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_M, self.ml_t, self.ml_m,
            _is_classifier(self.ml_m), self.ml_a, None if self.ml_a is None else _is_classifier(self.ml_a),
            score=self.score, n_folds=self.n_folds, n_folds_inner=self.n_folds_inner, n_rep=self.n_rep,
            seed=self.random_state, splits=splits, inner_splits=inner, treatment_names=_treatment_names(d),
        )
        return self._store(result)


class SSM(_DMLEstimator):
    """Average treatment effect of a binary treatment under sample selection (attrition or
    non-response), as DoubleML's ``DoubleMLSSM`` (Bia, Huber and Lafférs, 2024). The outcome is
    observed only where the selection indicator ``s`` is 1; other values of ``y`` are ignored but
    must be finite.

    ``score="missing-at-random"``: selection depends on the treatment and the covariates only.
    ``score="nonignorable"``: selection may also depend on unobservables; needs an instrument ``z``
    that shifts selection but not the outcome. Each training fold is then split in halves: the
    selection probability is learned on the first, the propensity and outcome models (with the
    predicted selection probability as a covariate) on the second.

    Parameters
    ----------
    ml_g : regressor for ``E[Y | D, X]`` among the selected.
    ml_pi : classifier for the selection probability ``P(S = 1 | D, X)`` (and ``Z``).
    ml_m : classifier for the propensity ``P(D = 1 | X)``.
    score : ``"missing-at-random"`` or ``"nonignorable"``.
    clipping_threshold : propensities are clipped to ``[t, 1 - t]`` (selection probabilities are not,
        as in DoubleML).
    normalize_ipw : normalise the inverse-probability weights.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_pi: Any,
        ml_m: Any,
        score: Literal["missing-at-random", "nonignorable"] = "missing-at-random",
        clipping_threshold: float = 1e-2,
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
        normalize_ipw: bool = False,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_pi = ml_pi
        self.ml_m = ml_m
        self.score = score
        self.clipping_threshold = clipping_threshold
        self.calibration = calibration
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        s: ArrayLike,
        z: Optional[ArrayLike] = None,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        result = _dml_rs.fit_ssm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_1d(s, "s"), self.ml_g, self.ml_pi, self.ml_m,
            z=None if z is None else _as_1d(z, "z"), score=self.score,
            clipping_threshold=self.clipping_threshold, calibration=self.calibration, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state, splits=_splits(folds),
            treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
        )
        return self._store(result)


class _QuantileBase(_DMLEstimator):
    _score: str = "PQ"

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        treatment: int = 1,
        quantile: float = 0.5,
        clipping_threshold: float = 1e-2,
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
        normalize_ipw: bool = True,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.treatment = treatment
        self.quantile = quantile
        self.clipping_threshold = clipping_threshold
        self.calibration = calibration
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def _fit(self, X, y, d, z, folds):
        result = _dml_rs.fit_quantile(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            score=self._score, treatment=int(self.treatment), quantile=float(self.quantile),
            z=None if z is None else _as_1d(z, "z"), clipping_threshold=self.clipping_threshold, calibration=self.calibration,
            normalize_ipw=self.normalize_ipw, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, splits=_splits(folds),
        )
        return self._store(result)


class PQ(_QuantileBase):
    """Potential quantile ``q`` with ``P(Y(d) <= q) = quantile`` for a binary treatment, as
    DoubleML's ``DoubleMLPQ``.

    Default sample splits are stratified by the treatment. Each fold runs a nested split for a
    preliminary inverse-probability-weighted estimate, at which the conditional distribution
    function is learned; the score is then solved with Brent's method and the standard error
    uses a kernel density estimate.

    Parameters
    ----------
    ml_g : classifier for ``P(Y <= q | D = d, X)`` with ``predict_proba``.
    ml_m : classifier for the propensity ``P(D = 1 | X)``.
    treatment : the treatment level ``d`` (0 or 1).
    quantile : the quantile level in ``(0, 1)``.
    clipping_threshold : propensities are clipped to ``[t, 1 - t]``.
    normalize_ipw : normalise the inverse-probability weights (default True, as DoubleML).
    """

    _score = "PQ"

    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, folds: Optional[Folds] = None) -> Self:
        return self._fit(X, y, d, None, folds)


class LPQ(_QuantileBase):
    """Local potential quantile of the compliers, with a binary instrument ``z``, as DoubleML's
    ``DoubleMLLPQ``. Default sample splits are stratified by treatment and instrument. Other
    parameters as `PQ`.

    Parameters
    ----------
    ml_g : classifier for ``P(Y <= q, D = d | Z = z, X)``.
    ml_m : classifier for the instrument propensity ``P(Z = 1 | X)``.
    """

    _score = "LPQ"

    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, z: ArrayLike, folds: Optional[Folds] = None) -> Self:
        return self._fit(X, y, d, z, folds)


class CVAR(_QuantileBase):
    """Conditional value at risk ``E[Y(d) | Y(d) >= q]`` above the potential quantile ``q``, as
    DoubleML's ``DoubleMLCVAR``. Other parameters as `PQ`.

    Parameters
    ----------
    ml_g : regressor for ``E[max(q, (Y - quantile * q) / (1 - quantile)) | D = d, X]``.
    ml_m : classifier for the propensity ``P(D = 1 | X)``.
    """

    _score = "CVaR"

    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, folds: Optional[Folds] = None) -> Self:
        return self._fit(X, y, d, None, folds)


class PotentialQuantiles(_ResultView):
    """Potential quantiles (or CVaRs) of both treatment levels, from `QTE.potential`, ordered
    ``q1 (d=0), q1 (d=1), q2 (d=0), ...`` and with joint inference."""


class QTE(_DMLEstimator):
    """Quantile treatment effects ``q(Y(1)) - q(Y(0))`` at several quantiles, as DoubleML's
    ``DoubleMLQTE``. ``score`` is ``"PQ"`` (quantiles), ``"LPQ"`` (compliers' quantiles, needs
    ``z``) or ``"CVaR"`` (effects on the conditional value at risk).

    Both potential quantiles are fitted on the same sample splits at every quantile, so the effects
    have joint inference (`bootstrap`, ``confint(joint=True)``). Other parameters as `PQ`.

    Parameters
    ----------
    ml_g : classifier (``"PQ"``, ``"LPQ"``) or regressor (``"CVaR"``), as in `PQ`, `LPQ`, `CVAR`.
    ml_m : propensity classifier (default: ``ml_g``, which must then be a classifier).
    quantiles : the quantile levels.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any = None,
        quantiles: Union[float, Sequence[float]] = 0.5,
        score: Literal["PQ", "LPQ", "CVaR"] = "PQ",
        clipping_threshold: float = 1e-2,
        calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
        normalize_ipw: bool = True,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.quantiles = quantiles
        self.score = score
        self.clipping_threshold = clipping_threshold
        self.calibration = calibration
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        z: Optional[ArrayLike] = None,
        folds: Optional[Folds] = None,
    ) -> Self:
        if self.ml_m is None and self.score == "CVaR":
            raise ValueError("score 'CVaR' needs a propensity classifier ml_m")
        quantiles = [float(q) for q in np.atleast_1d(np.asarray(self.quantiles, dtype=np.float64))]
        self._qte = _dml_rs.fit_qte(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), self.ml_g,
            self.ml_g if self.ml_m is None else self.ml_m, quantiles, score=self.score,
            z=None if z is None else _as_1d(z, "z"), clipping_threshold=self.clipping_threshold, calibration=self.calibration,
            normalize_ipw=self.normalize_ipw, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, splits=_splits(folds),
        )
        self.quantiles_ = np.asarray(self._qte.quantiles)
        return self._store(self._qte.effects)

    @property
    def potential(self) -> PotentialQuantiles:
        """The potential quantiles (or CVaRs) of treatment 0 and 1 at every quantile."""
        self._check_fitted()
        return PotentialQuantiles(self._qte.potential)

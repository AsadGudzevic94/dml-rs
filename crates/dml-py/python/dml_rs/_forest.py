"""Generalized random forests (Athey, Tibshirani and Wager, 2019): causal, instrumental and
multi-arm causal forests."""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from . import _dml_rs
from ._blp import BLP, blp_from
from ._models import Framework, _Params, _treatment_names
from ._policy import PolicyTree
from ._validation import _as_1d, _as_2d
from .data import accepts_data

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


class _Forest(_Params):
    """Shared settings and effect methods of the forests on residuals."""

    def _set_forest(self, n_estimators: int, sample_fraction: float, mtry: Optional[int], min_node_size: int,
                    honesty: bool, honesty_fraction: float, honesty_prune_leaves: bool, alpha: float,
                    imbalance_penalty: float, stabilize_splits: bool, ci_group_size: int,
                    clipping_threshold: float, n_folds: int, random_state: int) -> None:
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.stabilize_splits = stabilize_splits
        self.ci_group_size = ci_group_size
        self.clipping_threshold = clipping_threshold
        self.n_folds = n_folds
        self.random_state = random_state

    def _forest_kwargs(self) -> dict[str, Any]:
        return dict(
            num_trees=self.n_estimators, sample_fraction=self.sample_fraction, mtry=self.mtry,
            min_node_size=self.min_node_size, honesty=self.honesty, honesty_fraction=self.honesty_fraction,
            prune_empty_leaves=self.honesty_prune_leaves, alpha=self.alpha,
            imbalance_penalty=self.imbalance_penalty, stabilize_splits=self.stabilize_splits,
            ci_group_size=self.ci_group_size, forest_seed=self.random_state, n_folds=self.n_folds,
            seed=self.random_state, clipping_threshold=self.clipping_threshold,
        )

    def _store_common(self, result: Any) -> None:
        self.result_ = result
        self.effect_ = result.oob_effect
        self.effect_se_ = np.sqrt(result.oob_variance)
        self.y_hat_ = result.y_hat
        self.variable_importance_ = result.variable_importance
        self.n_trees_ = result.n_trees

    def _check_fitted(self) -> Any:
        if not hasattr(self, "result_"):
            raise AttributeError(f"{type(self).__name__} is not fitted yet; call fit() first")
        return self.result_

    def effect(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimated effects at the rows of ``X``."""
        return self._check_fitted().effect(_as_2d(X, "X"))[0]  # type: ignore[no-any-return]

    def effect_se(self, X: ArrayLike) -> NDArray[np.float64]:
        """Standard errors of the effects (bootstrap of little bags; needs ``ci_group_size > 1``)."""
        return np.sqrt(self._check_fitted().effect(_as_2d(X, "X"))[1])  # type: ignore[no-any-return]

    def effect_interval(self, X: ArrayLike, level: float = 0.95) -> NDArray[np.float64]:
        """Pointwise confidence intervals; the last axis holds the lower and upper bounds."""
        tau, var = self._check_fitted().effect(_as_2d(X, "X"))
        half = NormalDist().inv_cdf(0.5 + level / 2) * np.sqrt(var)
        return np.stack([tau - half, tau + half], axis=-1)  # type: ignore[no-any-return]

    def ate(self) -> Framework:
        """Average effect from the doubly robust scores built on the out-of-bag effects, with its
        standard error (grf's ``average_treatment_effect``)."""
        return Framework(self._check_fitted().ate())


class CausalForest(_Forest):
    """Causal forest for heterogeneous treatment effects ``τ(x)`` with confidence intervals, as R's
    ``grf::causal_forest`` (Athey, Tibshirani and Wager, 2019), on cross-fitted nuisances.

    The model is ``Y = τ(X)·D + m(X) + ε``. ``ml_y`` and ``ml_d`` learn ``E[Y | X]`` and ``E[D | X]``
    (a classifier for a binary treatment), cross-fitted on ``n_folds`` folds; a forest of honest
    trees is then grown on the residuals. Each tree takes a subsample (``sample_fraction``), half
    of which (``honesty_fraction``) chooses splits that maximise the heterogeneity of the effect,
    and the other half estimates the leaves. ``τ̂(x)`` is a residual-on-residual regression weighted
    by how often each training point shares a leaf with ``x``. Trees grown in groups of
    ``ci_group_size`` on one half-sample give variance estimates (the bootstrap of little bags).

    Parameters
    ----------
    ml_y : regressor for ``E[Y | X]``.
    ml_d : regressor for ``E[D | X]``, or a classifier (with ``predict_proba``) for a binary
        treatment.
    n_estimators : number of trees (grf's ``num.trees``, default 2000).
    sample_fraction : share of the data in each tree's subsample (at most 0.5 with
        ``ci_group_size > 1``).
    mtry : variables tried per split (default ``min(ceil(sqrt(p) + 20), p)``).
    min_node_size : nodes with at most this many splitting observations are not split.
    honesty, honesty_fraction, honesty_prune_leaves : honest trees, the share of the subsample used
        for splitting, and pruning of leaves left without estimation observations.
    alpha : each child of a split keeps at least this share of the parent's observations and
        treatment (instrument) variation (``stabilize_splits``).
    imbalance_penalty : penalty on splits with little treatment variation in a child.
    ci_group_size : trees per half-sample (2 gives variance estimates; 1 disables them).
    clipping_threshold : propensities are clipped to ``[t, 1 - t]`` in the doubly robust scores.
    n_folds : folds for the nuisance models.
    random_state : seed of the folds and the forest.
    """

    def __init__(
        self,
        ml_y: Any,
        ml_d: Any,
        n_estimators: int = 2000,
        sample_fraction: float = 0.5,
        mtry: Optional[int] = None,
        min_node_size: int = 5,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        honesty_prune_leaves: bool = True,
        alpha: float = 0.05,
        imbalance_penalty: float = 0.0,
        stabilize_splits: bool = True,
        ci_group_size: int = 2,
        clipping_threshold: float = 1e-2,
        n_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.ml_y = ml_y
        self.ml_d = ml_d
        self._set_forest(n_estimators, sample_fraction, mtry, min_node_size, honesty, honesty_fraction,
                         honesty_prune_leaves, alpha, imbalance_penalty, stabilize_splits, ci_group_size,
                         clipping_threshold, n_folds, random_state)

    @accepts_data("x", "y", "d")
    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike) -> Self:
        """Fit on covariates ``X``, outcome ``y`` and one treatment ``d``.

        Sets ``effect_`` and ``effect_se_`` (out-of-bag estimates for the training rows, each from
        the trees that did not see the row), ``y_hat_`` / ``d_hat_`` (cross-fitted nuisances) and
        ``variable_importance_`` (grf's split-frequency importance)."""
        d_arr = np.asarray(d, dtype=np.float64)
        if d_arr.ndim == 2 and d_arr.shape[1] != 1:
            raise ValueError("a causal forest takes one treatment")
        result = _dml_rs.fit_causal_forest(
            _as_1d(y, "y"), _as_2d(d_arr.reshape(-1), "d"), _as_2d(X, "X"), self.ml_y, self.ml_d,
            treatment_names=_treatment_names(d), **self._forest_kwargs(),
        )
        self._store_common(result)
        self.d_hat_ = result.d_hat
        self._y, self._d = _as_1d(y, "y"), d_arr.reshape(-1)
        return self

    # -- inference on averages (binary treatment) ----------------------------
    def blp(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the CATE on ``basis`` (HC0 inference on the doubly robust
        scores; binary treatment)."""
        return blp_from(self._check_fitted().blp(_as_2d(basis, "basis")), basis, "b")

    def gate(self, groups: ArrayLike) -> BLP:
        """Group average treatment effects for mutually exclusive 0/1 group dummies."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")

    def policy_tree(
        self, features: ArrayLike, depth: int = 2, min_samples_leaf: int = 8, ccp_alpha: float = 0.01
    ) -> PolicyTree:
        """Policy tree on the doubly robust scores: a shallow decision tree of ``features``
        recommending treatment where the effect is positive (it classifies the sign of the scores,
        weighted by their size, and is pruned with cost-complexity ``ccp_alpha``)."""
        native = self._check_fitted().policy_tree(_as_2d(features, "features"), depth, min_samples_leaf, ccp_alpha)
        return PolicyTree(native, features)

    def test_calibration(self) -> pd.DataFrame:
        """grf's ``test_calibration``: regress ``Y - ŷ`` on ``(D - d̂)·τ̄`` and
        ``(D - d̂)(τ̂(X) - τ̄)`` (out-of-bag ``τ̂``, mean ``τ̄``) with HC3 standard errors.

        A ``mean.forest.prediction`` coefficient near 1 says the average prediction is right; a
        ``differential.forest.prediction`` coefficient significantly above 0 (near 1) says the
        forest captures real heterogeneity. p-values are one-sided (``H1: coefficient > 0``) and use
        the normal distribution (grf: Student's t with ``n - 2`` degrees of freedom, the same for the
        sample sizes forests are fitted on)."""
        self._check_fitted()
        tau = np.asarray(self.effect_)
        mean = tau.mean()
        dres = self._d - self.d_hat_
        Z = np.column_stack([dres * mean, dres * (tau - mean)])
        target = self._y - self.y_hat_
        bread = np.linalg.inv(Z.T @ Z)
        coef = bread @ Z.T @ target
        resid = target - Z @ coef
        leverage = np.einsum("ij,jk,ik->i", Z, bread, Z)
        meat = (Z * (resid / (1 - leverage))[:, None] ** 2).T @ Z
        se = np.sqrt(np.diag(bread @ meat @ bread))
        tstat = coef / se
        pval = np.array([1 - NormalDist().cdf(v) for v in tstat])
        table = {"estimate": coef, "std. error": se, "t value": tstat, "Pr(>t)": pval}
        index = ["mean.forest.prediction", "differential.forest.prediction"]
        try:
            import pandas as pd
        except ImportError:
            return {"index": index, **table}  # type: ignore[return-value]  # no pandas
        return pd.DataFrame(table, index=index)


class InstrumentalForest(_Forest):
    """Instrumental forest for the conditional local average treatment effect ``θ(x)``, as R's
    ``grf::instrumental_forest``, on cross-fitted nuisances.

    With an instrument ``Z`` that moves the treatment ``D``, ``θ(x) = Cov(Z, Y | x) / Cov(Z, D | x)``:
    for a binary instrument, the effect among the units whose treatment the instrument changes
    (compliers) with covariates ``x``. ``ml_y``, ``ml_d`` and ``ml_z`` learn ``E[Y | X]``,
    ``E[D | X]`` and ``E[Z | X]`` (classifiers for binary variables); honest trees on the residuals
    split on the heterogeneity of ``θ``, and each child keeps a share ``alpha`` of the parent's
    instrument variation.

    Parameters
    ----------
    ml_y : regressor for ``E[Y | X]``.
    ml_d : regressor or classifier for ``E[D | X]``.
    ml_z : regressor or classifier for ``E[Z | X]``.
    reduced_form_weight : grf's ``reduced.form.weight``: splits target the instrument
        ``(1 - r) Z + r D`` (default 0).
    n_estimators : number of trees (grf's ``num.trees``, default 2000).
    sample_fraction : share of the data in each tree's subsample (at most 0.5 with
        ``ci_group_size > 1``).
    mtry : variables tried per split (default ``min(ceil(sqrt(p) + 20), p)``).
    min_node_size : nodes with at most this many splitting observations are not split.
    honesty, honesty_fraction, honesty_prune_leaves : honest trees, the share of the subsample used
        for splitting, and pruning of leaves left without estimation observations.
    alpha : each child of a split keeps at least this share of the parent's observations and
        treatment (instrument) variation (``stabilize_splits``).
    imbalance_penalty : penalty on splits with little treatment variation in a child.
    ci_group_size : trees per half-sample (2 gives variance estimates; 1 disables them).
    clipping_threshold : propensities are clipped to ``[t, 1 - t]`` in the doubly robust scores.
    n_folds : folds for the nuisance models.
    random_state : seed of the folds and the forest.
    """

    def __init__(
        self,
        ml_y: Any,
        ml_d: Any,
        ml_z: Any,
        n_estimators: int = 2000,
        sample_fraction: float = 0.5,
        mtry: Optional[int] = None,
        min_node_size: int = 5,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        honesty_prune_leaves: bool = True,
        alpha: float = 0.05,
        imbalance_penalty: float = 0.0,
        stabilize_splits: bool = True,
        ci_group_size: int = 2,
        reduced_form_weight: float = 0.0,
        clipping_threshold: float = 1e-2,
        n_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.ml_y = ml_y
        self.ml_d = ml_d
        self.ml_z = ml_z
        self.reduced_form_weight = reduced_form_weight
        self._set_forest(n_estimators, sample_fraction, mtry, min_node_size, honesty, honesty_fraction,
                         honesty_prune_leaves, alpha, imbalance_penalty, stabilize_splits, ci_group_size,
                         clipping_threshold, n_folds, random_state)

    @accepts_data("x", "y", "d", "z")
    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, z: ArrayLike) -> Self:
        """Fit on covariates ``X``, outcome ``y``, one treatment ``d`` and one instrument ``z``.

        Sets ``effect_`` / ``effect_se_`` (out-of-bag), ``y_hat_``, ``d_hat_``, ``z_hat_``,
        ``compliance_`` (out-of-bag ``E[D | Z=1, X] - E[D | Z=0, X]`` for a binary instrument,
        else None) and ``variable_importance_``."""
        d_arr = np.asarray(d, dtype=np.float64)
        if d_arr.ndim == 2 and d_arr.shape[1] != 1:
            raise ValueError("an instrumental forest takes one treatment")
        z_arr = np.asarray(z, dtype=np.float64)
        if z_arr.ndim == 2 and z_arr.shape[1] != 1:
            raise ValueError("an instrumental forest takes one instrument")
        result = _dml_rs.fit_instrumental_forest(
            _as_1d(y, "y"), _as_2d(d_arr.reshape(-1), "d"), _as_1d(z_arr.reshape(-1), "z"), _as_2d(X, "X"),
            self.ml_y, self.ml_d, self.ml_z, reduced_form_weight=self.reduced_form_weight,
            treatment_names=_treatment_names(d), **self._forest_kwargs(),
        )
        self._store_common(result)
        self.d_hat_ = result.d_hat
        self.z_hat_ = result.z_hat
        self.compliance_ = result.compliance
        return self

    def blp(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of ``θ(x)`` on ``basis`` (binary instrument)."""
        return blp_from(self._check_fitted().blp(_as_2d(basis, "basis")), basis, "b")

    def gate(self, groups: ArrayLike) -> BLP:
        """Group averages of ``θ(x)`` for mutually exclusive 0/1 group dummies (binary instrument)."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")


class MultiArmCausalForest(_Forest):
    """Multi-arm causal forest: effects of several treatment levels against a baseline, as R's
    ``grf::multi_arm_causal_forest``, on cross-fitted nuisances.

    ``d`` holds discrete treatment levels; the smallest is the baseline and the effects
    ``τₖ(x) = E[Y(k) - Y(baseline) | X = x]`` of the other levels are estimated jointly:
    ``θ(x) = Var(W | x)⁻¹ Cov(W, Y | x)`` for the residualised one-hot arm indicators ``W``. Splits
    use the pseudo-outcomes of all arms, and each child keeps a share ``alpha`` of the variation of
    every arm.

    Parameters
    ----------
    ml_y : regressor for ``E[Y | X]``.
    ml_d : classifier for the propensity of each level (fitted one level against the rest, then
        normalised to sum to one).
    n_estimators : number of trees (grf's ``num.trees``, default 2000).
    sample_fraction : share of the data in each tree's subsample (at most 0.5 with
        ``ci_group_size > 1``).
    mtry : variables tried per split (default ``min(ceil(sqrt(p) + 20), p)``).
    min_node_size : nodes with at most this many splitting observations are not split.
    honesty, honesty_fraction, honesty_prune_leaves : honest trees, the share of the subsample used
        for splitting, and pruning of leaves left without estimation observations.
    alpha : each child of a split keeps at least this share of the parent's observations and
        treatment (instrument) variation (``stabilize_splits``).
    imbalance_penalty : penalty on splits with little treatment variation in a child.
    ci_group_size : trees per half-sample (2 gives variance estimates; 1 disables them).
    clipping_threshold : propensities are clipped to ``[t, 1 - t]`` in the doubly robust scores.
    n_folds : folds for the nuisance models.
    random_state : seed of the folds and the forest.
    """

    def __init__(
        self,
        ml_y: Any,
        ml_d: Any,
        n_estimators: int = 2000,
        sample_fraction: float = 0.5,
        mtry: Optional[int] = None,
        min_node_size: int = 5,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        honesty_prune_leaves: bool = True,
        alpha: float = 0.05,
        imbalance_penalty: float = 0.0,
        stabilize_splits: bool = True,
        ci_group_size: int = 2,
        clipping_threshold: float = 1e-2,
        n_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.ml_y = ml_y
        self.ml_d = ml_d
        self._set_forest(n_estimators, sample_fraction, mtry, min_node_size, honesty, honesty_fraction,
                         honesty_prune_leaves, alpha, imbalance_penalty, stabilize_splits, ci_group_size,
                         clipping_threshold, n_folds, random_state)

    @accepts_data("x", "y", "d")
    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike) -> Self:
        """Fit on covariates ``X``, outcome ``y`` and a treatment ``d`` of discrete levels.

        Sets ``treatment_levels_`` (baseline first), ``effect_`` / ``effect_se_`` (out-of-bag,
        ``(n, k)`` for the ``k`` non-baseline levels), ``y_hat_``, ``propensity_`` (``(n, k + 1)``)
        and ``variable_importance_``."""
        d_arr = np.asarray(d, dtype=np.float64)
        if d_arr.ndim == 2 and d_arr.shape[1] != 1:
            raise ValueError("a multi-arm causal forest takes one column of treatment levels")
        names = _treatment_names(d)
        result = _dml_rs.fit_multi_arm_causal_forest(
            _as_1d(y, "y"), _as_1d(d_arr.reshape(-1), "d"), _as_2d(X, "X"), self.ml_y, self.ml_d,
            treatment_name=names[0] if names else None, **self._forest_kwargs(),
        )
        self._store_common(result)
        self.propensity_ = result.propensity
        self.treatment_levels_ = np.asarray(result.levels)
        self.arm_names_ = list(result.arm_names)
        return self

    def _arm(self, level: Any) -> int:
        levels = list(self._check_fitted().levels[1:])
        if level not in levels:
            raise ValueError(f"level {level!r} is not a non-baseline treatment level {levels}")
        return levels.index(level)

    def blp(self, basis: ArrayLike, level: Any) -> BLP:
        """Best linear predictor of the effect of ``level`` (against the baseline) on ``basis``."""
        native = self._check_fitted().blp(self._arm(level), _as_2d(basis, "basis"))
        return blp_from(native, basis, "b")

    def gate(self, groups: ArrayLike, level: Any) -> BLP:
        """Group average effects of ``level`` for mutually exclusive 0/1 group dummies."""
        native = self._check_fitted().gate(self._arm(level), _as_2d(groups, "groups"))
        return blp_from(native, groups, "group_")

"""Causal forests (generalized random forests; Athey, Tibshirani and Wager, 2019)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from . import _dml_rs
from ._blp import BLP, blp_from
from ._models import Framework, _Params, _treatment_names
from ._validation import _as_1d, _as_2d
from .data import accepts_data

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


class CausalForest(_Params):
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
        treatment variation (``stabilize_splits``).
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
            num_trees=self.n_estimators, sample_fraction=self.sample_fraction, mtry=self.mtry,
            min_node_size=self.min_node_size, honesty=self.honesty, honesty_fraction=self.honesty_fraction,
            prune_empty_leaves=self.honesty_prune_leaves, alpha=self.alpha,
            imbalance_penalty=self.imbalance_penalty, stabilize_splits=self.stabilize_splits,
            ci_group_size=self.ci_group_size, forest_seed=self.random_state, n_folds=self.n_folds,
            seed=self.random_state, clipping_threshold=self.clipping_threshold,
            treatment_names=_treatment_names(d),
        )
        self.result_ = result
        self.effect_ = result.oob_effect
        self.effect_se_ = np.sqrt(result.oob_variance)
        self.y_hat_ = result.y_hat
        self.d_hat_ = result.d_hat
        self.variable_importance_ = result.variable_importance
        self.n_trees_ = result.n_trees
        self._y, self._d = _as_1d(y, "y"), d_arr.reshape(-1)
        return self

    # -- effects ------------------------------------------------------------
    def effect(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimated effects ``τ̂(x)`` at the rows of ``X``."""
        return self._check_fitted().effect(_as_2d(X, "X"))[0]  # type: ignore[no-any-return]

    def effect_se(self, X: ArrayLike) -> NDArray[np.float64]:
        """Standard errors of ``τ̂(x)`` (bootstrap of little bags; needs ``ci_group_size > 1``)."""
        return np.sqrt(self._check_fitted().effect(_as_2d(X, "X"))[1])  # type: ignore[no-any-return]

    def effect_interval(self, X: ArrayLike, level: float = 0.95) -> NDArray[np.float64]:
        """Pointwise confidence intervals for ``τ(x)``, shape ``(n, 2)``."""
        from statistics import NormalDist

        tau, var = self._check_fitted().effect(_as_2d(X, "X"))
        half = NormalDist().inv_cdf(0.5 + level / 2) * np.sqrt(var)
        return np.column_stack([tau - half, tau + half])  # type: ignore[no-any-return]

    # -- inference on averages (binary treatment) ----------------------------
    def ate(self) -> Framework:
        """Average treatment effect from the doubly robust (AIPW) scores built on the out-of-bag
        ``τ̂(Xᵢ)``, with its standard error (binary treatment; grf's ``average_treatment_effect``)."""
        return Framework(self._check_fitted().ate())

    def blp(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the CATE on ``basis`` (HC0 inference on the doubly robust
        scores; binary treatment)."""
        return blp_from(self._check_fitted().blp(_as_2d(basis, "basis")), basis, "b")

    def gate(self, groups: ArrayLike) -> BLP:
        """Group average treatment effects for mutually exclusive 0/1 group dummies."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")

    def test_calibration(self) -> pd.DataFrame:
        """grf's ``test_calibration``: regress ``Y - ŷ`` on ``(D - d̂)·τ̄`` and
        ``(D - d̂)(τ̂(X) - τ̄)`` (out-of-bag ``τ̂``, mean ``τ̄``) with HC3 standard errors.

        A ``mean.forest.prediction`` coefficient near 1 says the average prediction is right; a
        ``differential.forest.prediction`` coefficient significantly above 0 (near 1) says the
        forest captures real heterogeneity. p-values are one-sided (``H1: coefficient > 0``) and use
        the normal distribution (grf: Student's t with ``n - 2`` degrees of freedom, the same for the
        sample sizes forests are fitted on)."""
        from statistics import NormalDist

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

    def _check_fitted(self) -> Any:
        if not hasattr(self, "result_"):
            raise AttributeError("CausalForest is not fitted yet; call fit() first")
        return self.result_

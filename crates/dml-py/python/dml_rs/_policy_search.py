"""Policy learning by exact tree search (policytree) and doubly robust policy values."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from . import _dml_rs
from ._models import Framework

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


def _matrix(a: Any, name: str) -> NDArray[np.float64]:
    arr = np.ascontiguousarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr


class OptimalPolicyTree:
    """The decision tree of depth ``depth`` that maximises the summed doubly robust rewards,
    found by exhaustive search (Athey and Wager, 2021): a port of R's ``policytree`` that finds
    the same trees.

    ``Gamma`` (units × actions) holds a doubly robust score of every unit under every action, for
    example from `double_robust_scores` of a fitted `APOS`. The tree recommends one action per
    leaf; splits are ``x_j <= value`` (left). The search is exact, so its cost grows like
    ``(p · n)^depth``: depth 2 suits thousands of units and tens of features, depth 3 small
    problems. ``split_step`` > 1 considers only every ``split_step``-th candidate split.

    After ``fit``: ``nodes_`` (breadth-first list of dicts), `predict`, and the fitted
    ``n_actions_``. Evaluate the learned policy with `policy_value` on units not used to learn it.
    """

    def __init__(self, depth: int = 2, split_step: int = 1, min_node_size: int = 1) -> None:
        self.depth = depth
        self.split_step = split_step
        self.min_node_size = min_node_size

    def fit(self, X: ArrayLike, Gamma: ArrayLike) -> Self:
        x = _matrix(X, "X")
        gamma = _matrix(Gamma, "Gamma")
        self._tree = _dml_rs.fit_optimal_policy_tree(
            x, gamma, depth=int(self.depth), split_step=int(self.split_step),
            min_node_size=int(self.min_node_size),
        )
        self.nodes_ = self._tree.nodes()
        self.n_actions_ = gamma.shape[1]
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Recommended action (column of ``Gamma``) for every row of ``X``."""
        if not hasattr(self, "_tree"):
            raise RuntimeError("call fit first")
        return np.asarray(self._tree.predict(_matrix(X, "X")), dtype=np.int64)


def policy_value(Gamma: ArrayLike, **policies: ArrayLike) -> Framework:
    """Doubly robust value ``E[Γ(π(X))]`` of each named policy (an array with the action, i.e. the
    column of ``Gamma``, of every unit), e.g. ``policy_value(Gamma, tree=tree.predict(X),
    treat_all=np.ones(n, int))``.

    Returns a `Framework` with one estimate per policy; all share the units, so differences
    (``fw.linear_combination(...)``) have correct standard errors and ``bootstrap`` gives joint
    intervals.
    """
    if not policies:
        raise ValueError("pass at least one policy as name=actions")
    gamma = _matrix(Gamma, "Gamma")
    actions = []
    for name, a in policies.items():
        arr = np.asarray(a)
        if arr.ndim != 1 or arr.shape[0] != gamma.shape[0]:
            raise ValueError(f"policy {name!r} needs one action per row of Gamma")
        if np.any(arr < 0) or np.any(arr != np.round(arr)):
            raise ValueError(f"policy {name!r} must contain non-negative integer actions")
        actions.append([int(v) for v in arr])
    return Framework(_dml_rs.policy_value(gamma, list(policies), actions))


def double_robust_scores(model: Any) -> NDArray[np.float64]:
    """Doubly robust scores ``Γ`` (units × levels) of a fitted `APOS`: the estimated potential
    outcome of every unit under every treatment level (the levels in the order of
    ``treatment_levels``), from the first repetition of the sample splits. Their column means are
    the average potential outcomes."""
    from ._models import APOS

    if not isinstance(model, APOS):
        raise TypeError("double_robust_scores needs a fitted APOS model")
    model._check_fitted()
    psi = np.asarray(model.psi_)[:, :, 0]
    theta = np.asarray(model.all_coef_)[:, 0]
    # The APO score has psi_a = -1, so psi = psi_b - theta.
    return psi + theta

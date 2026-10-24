"""CATE meta-learners: S, T, X, R and DR."""

from __future__ import annotations

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
from ._models import _Params, _splits
from ._validation import _as_1d, _as_2d


class _MetaLearner(_Params):
    _method: Literal["S", "T", "X", "R", "DR"]

    def _learners(self):
        raise NotImplementedError

    def _options(self):
        return {}

    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, folds: Optional[Folds] = None) -> Self:
        """Fit on covariates ``X``, outcome ``y`` and binary treatment ``d``."""
        x = _as_2d(X, "X")
        self.result_ = _dml_rs.fit_cate(
            self._method, _as_1d(y, "y"), _as_1d(d, "d"), x, **self._learners(), **self._options(),
            splits=_splits(folds),
        )
        self.ate_ = float(self.result_.effect(x).mean())
        return self

    def effect(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimated CATE ``τ̂(x)`` for each row of ``X``."""
        return self._check_fitted().effect(_as_2d(X, "X"))


class SLearner(_MetaLearner):
    """One model ``μ(x, d)`` on ``[X, d]``; ``τ̂(x) = μ(x, 1) − μ(x, 0)``."""

    _method = "S"

    def __init__(self, model: Any) -> None:
        self.model = model

    def _learners(self):
        return {"outcome": self.model}


class TLearner(_MetaLearner):
    """Separate outcome models for treated and controls; ``τ̂ = μ₁ − μ₀``."""

    _method = "T"

    def __init__(self, model: Any) -> None:
        self.model = model

    def _learners(self):
        return {"outcome": self.model}


class XLearner(_MetaLearner):
    """X-learner (Künzel et al., 2019)."""

    _method = "X"

    def __init__(self, outcome_model: Any, effect_model: Any, propensity_model: Any) -> None:
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity_model = propensity_model

    def _learners(self):
        return {"outcome": self.outcome_model, "effect": self.effect_model, "propensity": self.propensity_model}


class _CrossFitted(_MetaLearner):
    def __init__(
        self,
        outcome_model: Any,
        propensity_model: Any,
        effect_model: Any,
        n_folds: int = 5,
        clipping_threshold: float = 1e-2,
        random_state: int = 0,
    ) -> None:
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.effect_model = effect_model
        self.n_folds = n_folds
        self.clipping_threshold = clipping_threshold
        self.random_state = random_state

    def _learners(self):
        return {"outcome": self.outcome_model, "effect": self.effect_model, "propensity": self.propensity_model}

    def _options(self):
        return {"n_folds": self.n_folds, "seed": self.random_state, "clipping_threshold": self.clipping_threshold}


class RLearner(_CrossFitted):
    """R-learner (Nie & Wager, 2021). ``effect_model`` must accept ``sample_weight``."""

    _method = "R"


class DRLearner(_CrossFitted):
    """DR-learner (Kennedy, 2023): regress cross-fitted AIPW pseudo-outcomes on ``X``.

    Supports inference on effect heterogeneity via `blp` and `gate`.
    """

    _method = "DR"

    @property
    def pseudo_outcomes_(self) -> Optional[NDArray[np.float64]]:
        return self._check_fitted().pseudo_outcomes

    def blp(self, basis: ArrayLike) -> BLP:
        """Best linear predictor of the CATE on ``basis`` with HC0 inference."""
        return blp_from(self._check_fitted().blp(_as_2d(basis, "basis")), basis, "b")

    def gate(self, groups: ArrayLike) -> BLP:
        """Group average treatment effects for mutually exclusive 0/1 group dummies."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")

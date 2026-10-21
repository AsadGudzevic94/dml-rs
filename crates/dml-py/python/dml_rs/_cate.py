"""CATE meta-learners: S, T, X, R and DR."""

from __future__ import annotations

import numpy as np

from . import _dml_rs
from ._blp import blp_from
from ._models import _Params, _splits
from ._validation import _as_1d, _as_2d


class _MetaLearner(_Params):
    _method = None

    def _learners(self):
        raise NotImplementedError

    def _options(self):
        return {}

    def fit(self, X, y, d, folds=None):
        """Fit on covariates ``X``, outcome ``y`` and binary treatment ``d``."""
        X = _as_2d(X, "X")
        self.result_ = _dml_rs.fit_cate(
            self._method, _as_1d(y, "y"), _as_1d(d, "d"), X, **self._learners(), **self._options(),
            splits=_splits(folds),
        )
        self.ate_ = float(np.mean(self.result_.effect(X)))
        return self

    def effect(self, X):
        """Estimated CATE ``τ̂(x)`` for each row of ``X``."""
        return self._check_fitted().effect(_as_2d(X, "X"))


class SLearner(_MetaLearner):
    """One model ``μ(x, d)`` on ``[X, d]``; ``τ̂(x) = μ(x, 1) − μ(x, 0)``."""

    _method = "S"

    def __init__(self, model):
        self.model = model

    def _learners(self):
        return {"outcome": self.model}


class TLearner(_MetaLearner):
    """Separate outcome models for treated and controls; ``τ̂ = μ₁ − μ₀``."""

    _method = "T"

    def __init__(self, model):
        self.model = model

    def _learners(self):
        return {"outcome": self.model}


class XLearner(_MetaLearner):
    """X-learner (Künzel et al., 2019)."""

    _method = "X"

    def __init__(self, outcome_model, effect_model, propensity_model):
        self.outcome_model = outcome_model
        self.effect_model = effect_model
        self.propensity_model = propensity_model

    def _learners(self):
        return {"outcome": self.outcome_model, "effect": self.effect_model, "propensity": self.propensity_model}


class _CrossFitted(_MetaLearner):
    def __init__(self, outcome_model, propensity_model, effect_model, n_folds=5, clipping_threshold=1e-2,
                 random_state=0):
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

    Supports inference on effect heterogeneity via :meth:`blp` and :meth:`gate`.
    """

    _method = "DR"

    @property
    def pseudo_outcomes_(self):
        return self._check_fitted().pseudo_outcomes

    def blp(self, basis):
        """Best linear predictor of the CATE on ``basis`` with HC0 inference."""
        return blp_from(self._check_fitted().blp(_as_2d(basis, "basis")), basis, "b")

    def gate(self, groups):
        """Group average treatment effects for mutually exclusive 0/1 group dummies."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")

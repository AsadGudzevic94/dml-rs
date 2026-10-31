"""Causal mediation analysis with double machine learning (Farbmacher et al., 2022)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

from . import _dml_rs
from ._models import Framework, _DMLEstimator, _splits
from ._validation import _as_1d, _as_2d
from .data import accepts_data

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from typing_extensions import Self

    from ._models import Folds


class Mediation(_DMLEstimator):
    """Natural direct and indirect effects of a binary treatment ``d`` that acts partly through
    one or several mediators ``m`` (binary, categorical or continuous), with the double machine
    learning estimator of Farbmacher, Huber, Lafférs, Langen and Spindler (2022) — the estimator
    of R's ``causalweight::medDML`` (``multmed=TRUE``), including its debiased nested
    regressions (Qi and Zhang, 2026). By default the propensity scores are clipped rather than
    observations dropped (see ``trimming_rule``).

    The effects are differences of the potential-outcome means ``E[Y(d, M(d'))]``:

    - ``total``: ``E[Y(1, M(1))] − E[Y(0, M(0))]`` (the ATE),
    - ``direct (treated)`` / ``direct (control)``: ``E[Y(1, M(d))] − E[Y(0, M(d))]`` for
      ``d = 1`` / ``0``,
    - ``indirect (treated)`` / ``indirect (control)``: ``E[Y(d, M(1))] − E[Y(d, M(0))]`` for
      ``d = 1`` / ``0``.

    They assume that ``X`` contains all confounders of the treatment, the mediator and the
    outcome, and that none of them is affected by the treatment. The scores use the propensity
    scores ``P(D=1 | X)`` and ``P(D=1 | M, X)`` instead of mediator densities. The nested means
    ``E[E[Y | D=d, M, X] | D=1−d, X]`` are cross-fitted on a separate part of each training
    sample (with 3 folds, exactly medDML's split).

    After ``fit``: the fitted-model API of the five effects (``coef_``, ``se_``, ``summary``,
    ``confint``, ``bootstrap``, ``predictions_``), ``n_trimmed_`` and `potential_outcomes`.

    Parameters
    ----------
    ml_g : regressor for the outcome means ``E[Y | D=d, X]`` and ``E[Y | D=d, M, X]`` (fitted
        per treatment arm, on ``X`` and on ``[X, m]``).
    ml_m : classifier for the propensity scores ``P(D=1 | X)`` and ``P(D=1 | M, X)``.
    ml_nested : regressor for the nested means (default: a copy of ``ml_g``).
    trimming_rule : ``"truncate"`` (default) clips ``P(D=1 | X)`` and ``P(D=1 | M, X)`` to
        ``[t, 1 − t]``; ``"drop"`` (medDML's rule) drops observations whose propensity products
        in a denominator are below ``t``. Dropping selects on ``P(D=1 | M, X)``, which depends
        on the post-treatment mediator, and biases the effects when the mediator predicts the
        treatment well; use it to reproduce medDML (with ``trimming_threshold=0.05``).
    trimming_threshold : the threshold ``t`` (default 0.01).
    normalize_ipw : normalise each inverse probability weight to mean one (default True).
    nested_split : fit the outcome means and the nested means on disjoint parts of the training
        folds (default True; False is medDML's ``fewsplits=TRUE``). Needs ``n_folds >= 3``.
    debias_nested : debias the nested means with reweighted outcome residuals (Qi and Zhang,
        2026; default True, medDML's ``debiasfirst=TRUE``).
    n_folds, n_rep : cross-fitting folds (default 5; medDML uses 3) and repetitions of the split.
    random_state : seed for the sample splits.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        ml_nested: Any = None,
        trimming_rule: Literal["truncate", "drop"] = "truncate",
        trimming_threshold: float = 1e-2,
        normalize_ipw: bool = True,
        nested_split: bool = True,
        debias_nested: bool = True,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_nested = ml_nested
        self.trimming_rule = trimming_rule
        self.trimming_threshold = trimming_threshold
        self.normalize_ipw = normalize_ipw
        self.nested_split = nested_split
        self.debias_nested = debias_nested
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    @accepts_data("x", "y", "d", "m")
    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, m: ArrayLike,
            folds: Optional[Folds] = None) -> Self:
        """Fit on covariates ``X``, outcome ``y``, binary treatment ``d`` and mediators ``m``
        (1-D or one column per mediator). ``folds`` are K-fold splits (``(train, test)`` pairs
        whose test sets partition the sample); fold ``k`` fits the outcome means on the next
        ``K // 2`` folds and the nested means on the rest."""
        self._mediation = _dml_rs.fit_mediation(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(m, "m"), _as_2d(X, "X"),
            self.ml_g, self.ml_m, ml_nested=self.ml_nested, trimming_rule=self.trimming_rule,
            trimming_threshold=self.trimming_threshold, normalize_ipw=self.normalize_ipw,
            nested_split=self.nested_split, debias_nested=self.debias_nested,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state, splits=_splits(folds),
        )
        self.n_trimmed_ = list(self._mediation.n_trimmed)
        return self._store(self._mediation.effects)

    def potential_outcomes(self) -> Framework:
        """``E[Y(1, M(1))]``, ``E[Y(1, M(0))]``, ``E[Y(0, M(1))]`` and ``E[Y(0, M(0))]``."""
        self._check_fitted()
        return Framework(self._mediation.potential_outcomes)

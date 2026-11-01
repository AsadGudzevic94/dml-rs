"""Conformal intervals for counterfactuals and individual treatment effects (Lei and Candès, 2021)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np

from . import _dml_rs
from ._validation import _as_1d, _as_2d
from .data import accepts_data

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


class ConformalITE:
    """Prediction intervals for the potential outcomes and the individual treatment effect
    ``Y(1) − Y(0)`` of new units, with finite-sample coverage under unconfoundedness (Lei and
    Candès, 2021).

    Unlike confidence intervals for an average or conditional average effect, these intervals
    cover the *individual* outcome or effect of a new unit with probability ``1 − alpha``. They
    come from weighted split conformal prediction: the learners are fitted on a training part,
    and conformity scores of the calibration units of each arm are reweighted by the propensity
    score to the target population (covariate shift). With a known propensity score (a randomised
    experiment) coverage holds exactly in finite samples; with an estimated one, approximately.

    Two conformity scores:

    - ``ml_y``: a regressor for the mean; intervals ``μ(x) ± Q``.
    - ``ml_lower`` and ``ml_upper``: regressors for a lower and an upper conditional quantile
      (conformalised quantile regression, e.g. scikit-learn's
      ``GradientBoostingRegressor(loss="quantile", alpha=...)``); intervals adapt to
      heteroscedasticity.

    The effect interval combines the two counterfactual intervals at ``alpha / 2`` each,
    ``[L₁ − U₀, U₁ − L₀]``, so it is conservative.

    Parameters
    ----------
    ml_m : classifier for the propensity score (or pass ``propensity``).
    ml_y : regressor for the residual score.
    ml_lower, ml_upper : quantile regressors for the quantile score.
    propensity : known treatment probability of a randomised experiment.
    train_fraction : share of the sample used to fit the learners (default 0.75).
    trimming_threshold : clip estimated propensity scores to ``[t, 1 − t]`` (default 0.01).
    random_state : seed of the split.

    Fitted models hold the learners and are not picklable.
    """

    def __init__(
        self,
        ml_m: Any = None,
        ml_y: Any = None,
        ml_lower: Any = None,
        ml_upper: Any = None,
        propensity: Optional[float] = None,
        train_fraction: float = 0.75,
        trimming_threshold: float = 1e-2,
        random_state: int = 0,
    ) -> None:
        self.ml_m = ml_m
        self.ml_y = ml_y
        self.ml_lower = ml_lower
        self.ml_upper = ml_upper
        self.propensity = propensity
        self.train_fraction = train_fraction
        self.trimming_threshold = trimming_threshold
        self.random_state = random_state

    @accepts_data("x", "y", "d")
    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike,
            split: Optional[tuple[Sequence[int], Sequence[int]]] = None) -> Self:
        """Fit on covariates ``X``, outcome ``y`` and binary treatment ``d``; ``split`` =
        ``(train, calibration)`` indices replaces the random split."""
        x = _as_2d(X, "X")
        self._result = _dml_rs.fit_conformal(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), x, ml_m=self.ml_m, ml_y=self.ml_y,
            ml_lower=self.ml_lower, ml_upper=self.ml_upper, propensity=self.propensity,
            train_fraction=self.train_fraction, trimming_threshold=self.trimming_threshold,
            seed=self.random_state,
            split=None if split is None else ([int(i) for i in split[0]], [int(i) for i in split[1]]),
        )
        self.n_features_in_ = x.shape[1]
        self.n_calibration_ = list(self._result.n_calibration)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "_result"):
            raise RuntimeError("call fit first")

    def predict_ite(self, X: ArrayLike, alpha: float = 0.1) -> NDArray[np.float64]:
        """Intervals ``[lower, upper]`` (shape (m, 2)) for the individual effects of new units
        ``X`` at miscoverage level ``alpha``."""
        self._check_fitted()
        return np.asarray(self._result.predict_ite(_as_2d(X, "X"), alpha))

    def predict_counterfactual(
        self,
        X: ArrayLike,
        treatment: int = 1,
        target: Literal["all", "treated", "control"] = "all",
        alpha: float = 0.1,
    ) -> NDArray[np.float64]:
        """Intervals (shape (m, 2)) for the potential outcome ``Y(treatment)`` of new units
        ``X`` drawn like ``target`` (all units, or units like the treated / control ones, e.g.
        ``Y(0)`` of treated units with ``treatment=0, target="treated"``)."""
        self._check_fitted()
        return np.asarray(self._result.predict_counterfactual(_as_2d(X, "X"), int(treatment), target, alpha))

    def __reduce__(self) -> Any:
        if hasattr(self, "_result"):
            raise TypeError("fitted ConformalITE models are not picklable")
        return super().__reduce__()

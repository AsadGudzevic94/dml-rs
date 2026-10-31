"""Long-term treatment effects from short-term surrogates (the surrogate index)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from . import _dml_rs
from ._models import Framework, _DMLEstimator, _splits
from ._validation import _as_1d, _as_2d

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from typing_extensions import Self

    from ._models import Folds


class SurrogateIndex(_DMLEstimator):
    """Average effect of a binary treatment on a long-term outcome that the experiment does not
    observe, from short-term surrogates: the surrogate index of Athey, Chetty, Imbens and Kang
    (2019) with the efficient (doubly robust) score of Chen and Ritzwoller (2023).

    The experiment records the treatment ``w``, the surrogates ``S`` (short-term outcomes) and
    covariates ``X``, but not the long-term outcome. An observational sample records ``S``,
    ``X`` and the long-term outcome ``y``. Under *surrogacy* (in the experiment the treatment
    affects ``y`` only through ``S``, given ``X``) and *comparability* (``y`` given ``S, X`` is
    distributed alike in both samples), the long-term effect in the experimental population is
    identified. The surrogate index ``E[y | S, X]`` is learned on the observational sample; the
    efficient score adds the observational residuals ``y − E[y | S, X]``, reweighted to the
    experiment, so that errors in the index do not bias the estimate to first order.

    After ``fit``: the fitted-model API of the effect (``coef_``, ``se_``, ``summary``,
    ``confint``, ``bootstrap``, ``predictions_`` with ``"ml_g"`` = the surrogate index,
    ``"ml_nested_d1"`` / ``"ml_nested_d0"`` = its mean given ``X`` in each arm, ``"ml_m"`` =
    ``P(w=1 | X)``, ``"ml_m_surrogate"`` = ``P(w=1 | S, X)``, ``"ml_sample"`` = ``P(experiment
    | S, X)``) and `potential_outcomes`.

    Parameters
    ----------
    ml_g : regressor for the surrogate index ``E[y | S, X]`` (fitted on ``[S, X]``).
    ml_m : classifier for ``P(w=1 | X)``, ``P(w=1 | S, X)`` and ``P(experiment | S, X)``.
    ml_nested : regressor for the mean of the index given ``X`` in each arm (default: a copy of
        ``ml_g``).
    propensity : known treatment probability of a completely randomised experiment (otherwise
        ``P(w=1 | X)`` is learned).
    trimming_threshold : clip the estimated probabilities to ``[t, 1 − t]`` (default 0.01).
    n_folds, n_rep : cross-fitting folds and repetitions; splits are stratified by treated,
        control and observational units.
    random_state : seed for the sample splits.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        ml_nested: Any = None,
        propensity: Optional[float] = None,
        trimming_threshold: float = 1e-2,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_nested = ml_nested
        self.propensity = propensity
        self.trimming_threshold = trimming_threshold
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(self, X_exp: ArrayLike, S_exp: ArrayLike, w: ArrayLike, X_obs: ArrayLike,
            S_obs: ArrayLike, y: ArrayLike, folds: Optional[Folds] = None) -> Self:
        """Fit on the experiment (``X_exp``, surrogates ``S_exp``, treatment ``w``) and the
        observational sample (``X_obs``, ``S_obs``, long-term outcome ``y``). ``folds`` index
        the stacked data: experimental units ``0 .. n_exp − 1``, then observational ones."""
        self._surrogate = _dml_rs.fit_surrogate_index(
            _as_2d(X_exp, "X_exp"), _as_2d(S_exp, "S_exp"), _as_1d(w, "w"),
            _as_2d(X_obs, "X_obs"), _as_2d(S_obs, "S_obs"), _as_1d(y, "y"),
            self.ml_g, self.ml_m, ml_nested=self.ml_nested, propensity=self.propensity,
            trimming_threshold=self.trimming_threshold, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, splits=_splits(folds),
        )
        return self._store(self._surrogate.effect)

    def potential_outcomes(self) -> Framework:
        """``E[Y(1)]`` and ``E[Y(0)]`` of the long-term outcome in the experimental population."""
        self._check_fitted()
        return Framework(self._surrogate.potential_outcomes)

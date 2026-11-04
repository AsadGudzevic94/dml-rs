"""Causal effects on a right-censored time to event."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union

import numpy as np

from . import _dml_rs
from ._models import Framework, _COLORBLIND, _DMLEstimator, _codes, _pyplot, _splits
from ._validation import _as_1d, _as_2d

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

    from ._models import Folds


class CausalSurvival(_DMLEstimator):
    """Effect of a binary treatment on a right-censored time to event.

    Compares the survival probability ``P(T(d) > τ)``, the restricted mean survival time
    ``E[min(T(d), τ)]`` or, with competing events, the cumulative incidence
    ``P(T(d) ≤ τ, cause = c)`` if everyone were treated with the same quantity if nobody were.
    Both event and censoring may depend on the treatment and the covariates. They must be
    independent of each other given these, and the covariates must contain all confounders of
    the treatment.

    With competing events (``event`` = 1, 2, ... for the cause), survival and RMST are freedom
    from every cause, and ``estimand="cumulative_incidence"`` targets cause ``cause``: the
    hazards of any event and of that cause are learned, and without covariates the estimate is
    the Aalen-Johansen estimator. ``estimand="rmtl"`` gives the restricted mean time lost to that
    cause, ``∫₀^τ F_c(t) dt``, which with RMST and the other causes adds up to ``τ``.

    Time is split into the slots of a grid, and each unit becomes one row per slot in which it is
    at risk, so any classifier can learn the discrete hazards of the event and of censoring. The
    cross-fitted hazards and the propensity score enter the efficient influence function of the
    survival curve. The estimate is doubly robust: it is consistent if the event hazards, or the
    censoring hazards and the propensity score, are. Standard errors come from the influence
    function. Without covariates and with empirical hazards the estimator is Kaplan-Meier.

    After ``fit``: the fitted-model API of the effects at ``horizons_`` (treated minus control;
    ``coef_``, ``se_``, ``summary``, ``confint``, ``bootstrap``), plus ``grid_`` and
    ``propensity_``. `potential` gives the two arms, `curve` the survival curves and
    `incidence_curve` the cumulative incidence curves of the cause.

    Parameters
    ----------
    ml_event : classifier for the hazard of the event in a slot, given the slot and ``X``, fitted
        separately on treated and controls.
    ml_censor : classifier for the hazard of censoring (default: a copy of ``ml_event``).
    ml_m : classifier for the propensity ``P(D = 1 | X)``.
    horizons : times ``τ`` of the effects (default: the quartiles of the event times).
    estimand : ``"survival"``, ``"rmst"``, ``"cumulative_incidence"`` or ``"rmtl"``.
    cause : the cause of the cumulative incidence or of the time lost (a code of ``event``,
        default 1).
    grid : the slots: an int for that many quantiles of the event times, ``"event_times"`` for
        every distinct event time (slow on large data), or the times themselves. The horizons are
        always added. A coarse grid drops some information on the order of events and censoring
        within a slot, so use 20 or more slots.
    time_encoding : how the slot enters the learners: ``"numeric"`` (its time, for trees and
        boosting) or ``"dummies"`` (one indicator per slot, which gives a linear learner a free
        baseline hazard).
    clipping_threshold : propensities are clipped to ``[t, 1 - t]``.
    censoring_floor : probabilities of remaining uncensored are raised to this value. Horizons
        where few units remain uncensored give large standard errors.
    n_folds, n_rep : cross-fitting folds and repetitions of the sample split.
    random_state : seed for the sample splits, which are stratified by treatment and event
        (without clusters).
    """

    def __init__(
        self,
        ml_event: Any,
        ml_censor: Any = None,
        ml_m: Any = None,
        horizons: Optional[Sequence[float]] = None,
        estimand: Literal["survival", "rmst", "cumulative_incidence", "rmtl"] = "survival",
        cause: int = 1,
        grid: Union[int, str, Sequence[float]] = 30,
        time_encoding: Literal["numeric", "dummies"] = "numeric",
        clipping_threshold: float = 1e-2,
        censoring_floor: float = 1e-2,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_event = ml_event
        self.ml_censor = ml_censor
        self.ml_m = ml_m
        self.horizons = horizons
        self.estimand = estimand
        self.cause = cause
        self.grid = grid
        self.time_encoding = time_encoding
        self.clipping_threshold = clipping_threshold
        self.censoring_floor = censoring_floor
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        time: ArrayLike,
        event: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
    ) -> Self:
        """Fit on covariates ``X``, observed time ``time``, ``event`` (0 for censoring, 1 for an
        event, or 1, 2, ... for the cause of competing events) and binary treatment ``d``. With
        ``cluster`` (one label per row), whole clusters go to folds and the standard errors are
        cluster-robust."""
        if self.ml_m is None:
            raise ValueError("ml_m (a classifier for the propensity score) is required")
        horizons = None if self.horizons is None else [float(t) for t in np.atleast_1d(self.horizons)]
        grid: Any = self.grid
        if not isinstance(grid, (str, int, np.integer)):
            grid = [float(t) for t in np.atleast_1d(grid)]
        self._survival = _dml_rs.fit_causal_survival(
            _as_1d(time, "time"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), _as_1d(event, "event"),
            self.ml_event, self.ml_event if self.ml_censor is None else self.ml_censor, self.ml_m,
            horizons=horizons, estimand=self.estimand, grid=grid, time_encoding=self.time_encoding,
            clipping_threshold=self.clipping_threshold, censoring_floor=self.censoring_floor,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state, splits=_splits(folds),
            cause=int(self.cause),
            cluster=None if cluster is None else _codes(cluster, "cluster").reshape(-1, 1),
        )
        self.n_clusters_ = None if cluster is None else int(len(np.unique(np.asarray(cluster))))
        self.horizons_ = np.asarray(self._survival.horizons)
        self.grid_ = np.asarray(self._survival.grid)
        self.propensity_ = self._survival.propensity
        return self._store(self._survival.effect)

    def potential(self, treated: bool = True) -> Framework:
        """The estimand at the horizons if everyone were treated, or if nobody were."""
        self._check_fitted()
        return Framework(self._survival.potential(bool(treated)))

    def curve(self, treated: bool = True) -> Framework:
        """The survival curve on ``grid_`` if everyone were treated, or if nobody were."""
        self._check_fitted()
        return Framework(self._survival.curve(bool(treated)))

    def incidence_curve(self, treated: bool = True) -> Framework:
        """The cumulative incidence of ``cause`` on ``grid_`` if everyone were treated, or if
        nobody were (``estimand="cumulative_incidence"`` or ``"rmtl"`` only)."""
        self._check_fitted()
        result = self._survival.incidence_curve(bool(treated))
        if result is None:
            raise ValueError('incidence curves need estimand="cumulative_incidence" or "rmtl"')
        return Framework(result)

    def plot(self, level: float = 0.95, ax: Any = None) -> Any:
        """Plot both survival curves, or with ``estimand="cumulative_incidence"`` both cumulative
        incidence curves, with pointwise confidence bands. Needs matplotlib; returns the axes."""
        self._check_fitted()
        if ax is None:
            ax = _pyplot().subplots()[1]
        incidence = self.estimand in ("cumulative_incidence", "rmtl")
        start = 0.0 if incidence else 1.0
        times = np.concatenate(([0.0], self.grid_))
        for treated, name, color in ((True, "Treated", _COLORBLIND[0]), (False, "Control", _COLORBLIND[1])):
            curve = self.incidence_curve(treated) if incidence else self.curve(treated)
            ci: NDArray[np.float64] = np.clip(curve.confint(level), 0.0, 1.0)
            ax.step(times, np.concatenate(([start], curve.coef_)), where="post", color=color, label=name)
            ax.fill_between(times, np.concatenate(([start], ci[:, 0])), np.concatenate(([start], ci[:, 1])),
                            step="post", color=color, alpha=0.2)
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Cumulative incidence of cause {self.cause}" if incidence else "Survival probability")
        ax.set_ylim(0, 1.02)
        ax.legend()
        return ax

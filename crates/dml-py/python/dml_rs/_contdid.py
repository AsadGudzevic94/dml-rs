"""Difference-in-differences with a continuous treatment (Callaway, Goodman-Bacon and
Sant'Anna, 2024)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np

from . import _dml_rs
from ._models import Framework, _DMLEstimator, _pyplot, _splits, _COLORBLIND
from ._validation import _as_1d, _as_2d
from .data import accepts_data

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from typing_extensions import Self

    from ._models import Folds


class ContinuousDID(_DMLEstimator):
    """Dose-response of a continuous treatment in a two-period difference-in-differences design
    (Callaway, Goodman-Bacon and Sant'Anna, 2024).

    One row per unit: the outcome change ``y = Y_post − Y_pre``, the dose ``d ≥ 0`` received
    between the periods (0 = untreated comparison units) and optional pre-treatment covariates
    ``X``. Under (conditional) parallel trends the average effect of dose ``d`` on the units
    that received it,

    ``ATT(d | d) = E[ΔY − E[ΔY | D = 0, X] | D = d]``,

    is estimated as a B-spline series in the dose; its derivative ``ACRT(d | d)`` is the
    average causal response. The fit also reports the overall ``ATT`` (over the treated units'
    doses) and ``ACRT``.

    Without covariates this is the estimator of R's ``contdid`` package and reproduces it.
    With covariates the estimate is doubly robust: ``ml_g`` learns ``E[ΔY | D = 0, X]`` on the
    untreated units, ``ml_m`` the probability of being treated, ``ml_nested`` the spline
    functions' means given ``X`` among the treated, all cross-fitted. Note that a
    heterogeneous ``ATT(d | d)`` compares different units at different doses; reading
    differences across doses as causal needs the stronger parallel trends assumption of the
    paper.

    After ``fit``: the fitted-model API of ``ATT(d)`` at the grid doses (``coef_``, ``se_``,
    ``summary``, ``confint``, ``bootstrap``), ``grid_``, `acrt`, `overall` and `plot`.

    Parameters
    ----------
    ml_g, ml_m : learners for data with covariates (omit both without covariates).
    ml_nested : regressor for the spline means (default: a copy of ``ml_g``).
    degree : degree of the B-spline in the dose (default 3).
    num_knots : interior knots at quantiles of the positive doses (default 0).
    grid : doses at which to report ``ATT(d)`` and ``ACRT(d)`` (default: the 5%, …, 95%
        quantiles of the positive doses); within the range of the positive doses.
    trimming_threshold : clip ``P(D > 0 | X)`` to ``[t, 1 − t]`` (default 0.01).
    n_folds, n_rep : cross-fitting folds and repetitions (with covariates).
    random_state : seed for the sample splits.
    """

    def __init__(
        self,
        ml_g: Any = None,
        ml_m: Any = None,
        ml_nested: Any = None,
        degree: int = 3,
        num_knots: int = 0,
        grid: Optional[Sequence[float]] = None,
        trimming_threshold: float = 1e-2,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_nested = ml_nested
        self.degree = degree
        self.num_knots = num_knots
        self.grid = grid
        self.trimming_threshold = trimming_threshold
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    @accepts_data("x", "y", "d")
    def fit(self, X: Optional[ArrayLike], y: ArrayLike, d: ArrayLike,
            folds: Optional[Folds] = None) -> Self:
        """Fit on covariates ``X`` (``None`` for none), outcome changes ``y`` and doses ``d``."""
        y_arr = _as_1d(y, "y")
        x_arr = np.zeros((len(y_arr), 0)) if X is None else _as_2d(X, "X")
        grid = None if self.grid is None else [float(v) for v in np.atleast_1d(self.grid)]
        self._contdid = _dml_rs.fit_continuous_did(
            y_arr, _as_2d(_as_1d(d, "d"), "d"), x_arr, ml_g=self.ml_g, ml_m=self.ml_m,
            ml_nested=self.ml_nested, degree=self.degree, num_knots=self.num_knots, grid=grid,
            trimming_threshold=self.trimming_threshold, n_folds=self.n_folds, n_rep=self.n_rep,
            seed=self.random_state, splits=_splits(folds),
        )
        self.grid_ = np.asarray(self._contdid.grid)
        return self._store(self._contdid.att)

    def acrt(self) -> Framework:
        """The average causal response ``ACRT(d | d)`` at the grid doses."""
        self._check_fitted()
        return Framework(self._contdid.acrt)

    def overall(self) -> Framework:
        """The overall ``ATT`` and ``ACRT``, averaged over the treated units' doses."""
        self._check_fitted()
        return Framework(self._contdid.overall)

    def plot(
        self,
        kind: Literal["att", "acrt"] = "att",
        level: float = 0.95,
        joint: bool = False,
        ax: Any = None,
        color: str = _COLORBLIND[0],
    ) -> Any:
        """Plot ``ATT(d)`` or ``ACRT(d)`` with pointwise (or, ``joint=True``, uniform
        bootstrap) confidence bands. Needs matplotlib; returns the axes."""
        self._check_fitted()
        if kind not in ("att", "acrt"):
            raise ValueError(f"kind must be 'att' or 'acrt', got {kind!r}")
        target = self if kind == "att" else self.acrt()
        if joint and kind == "acrt":
            target.bootstrap()
        ci = target.confint(level, joint=joint)
        if ax is None:
            ax = _pyplot().subplots()[1]
        label = f"{level:.0%} {'uniform' if joint else 'pointwise'} band"
        ax.fill_between(self.grid_, ci[:, 0], ci[:, 1], color=color, alpha=0.2, label=label)
        ax.plot(self.grid_, target.coef_, color=color, label=f"{kind.upper()}(d)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("dose d")
        ax.set_ylabel(kind.upper())
        ax.legend()
        return ax

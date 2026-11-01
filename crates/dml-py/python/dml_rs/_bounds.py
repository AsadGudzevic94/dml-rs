"""Partial identification under sample selection: Lee (2009) bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from . import _dml_rs

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self


class LeeBounds:
    """Sharp bounds on the effect of a randomised binary treatment for the units whose outcome is
    observed whichever treatment they get, when the treatment also changes *whether* the outcome
    is observed (Lee, 2009): wages when training changes employment, test scores when a programme
    changes attrition, survey outcomes when an incentive changes response.

    Under monotonicity (the treatment only raises, or only lowers, selection), the arm with the
    higher selection rate contains a share ``q = |p₁ − p₀| / max(p₁, p₀)`` of units observed only
    because of their treatment. Trimming that share from the top or the bottom of its observed
    outcomes gives the lower and the upper bound. With discrete covariate cells (``strata``) the
    bounds are computed per cell and averaged, which tightens them (Lee 2009, Section 4).

    Standard errors come from the bootstrap (units resampled within arms); ``confint_`` is the
    Imbens–Manski (2004) interval, which covers the true effect with probability ``level``.

    After ``fit``: ``lower_``, ``upper_``, ``bounds_``, ``trimming_`` (share trimmed),
    ``trimmed_treated_``, ``se_`` (bootstrap standard errors of the two bounds), ``confint_`` and
    ``summary``.

    Parameters
    ----------
    n_boot : bootstrap replications (default 500; 0 skips inference).
    level : confidence level of the interval (default 0.95).
    random_state : seed of the bootstrap.
    """

    def __init__(self, n_boot: int = 500, level: float = 0.95, random_state: int = 0) -> None:
        self.n_boot = n_boot
        self.level = level
        self.random_state = random_state

    def fit(self, y: ArrayLike, d: ArrayLike, s: ArrayLike, strata: Optional[ArrayLike] = None) -> Self:
        """Fit on outcomes ``y`` (any value, e.g. NaN, where unobserved), the 0/1 treatment ``d``,
        the 0/1 selection indicator ``s`` (1 = outcome observed) and optional discrete covariate
        cells ``strata`` (any labels)."""
        y_arr = np.ascontiguousarray(y, dtype=np.float64).ravel()
        cells = None
        if strata is not None:
            cells = np.unique(np.asarray(strata), return_inverse=True)[1].astype(np.int64).ravel()
        self._result = _dml_rs.fit_lee_bounds(
            y_arr, np.ascontiguousarray(d, dtype=np.float64).ravel(),
            np.ascontiguousarray(s, dtype=np.float64).ravel(), cells, n_boot=int(self.n_boot),
            level=float(self.level), seed=int(self.random_state),
        )
        r = self._result
        self.lower_, self.upper_ = r.lower, r.upper
        self.bounds_ = np.array([r.lower, r.upper])
        self.trimming_ = r.trimming
        self.trimmed_treated_ = r.trimmed_treated
        self.se_ = np.asarray(r.se)
        self.confint_ = np.asarray(r.confint)
        return self

    @property
    def summary(self) -> Any:
        """Bounds, their standard errors and the Imbens–Manski interval."""
        if not hasattr(self, "_result"):
            raise RuntimeError("call fit first")
        row = {"lower bound": self.lower_, "upper bound": self.upper_, "se lower": self.se_[0],
               "se upper": self.se_[1], f"{self.level:.0%} CI lower": self.confint_[0],
               f"{self.level:.0%} CI upper": self.confint_[1], "trimmed share": self.trimming_}
        try:
            import pandas as pd
        except ImportError:
            return row
        return pd.DataFrame([row], index=["Lee bounds"])

"""Dose-response curves of a continuous treatment (Colangelo and Lee, 2023)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np

from . import _dml_rs
from ._models import Framework, _DMLEstimator, _pyplot, _splits, _COLORBLIND
from ._validation import _as_1d, _as_2d
from .data import accepts_data

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

    from ._models import Folds


class DoseResponse(_DMLEstimator):
    """Average dose-response function ``β(t) = E[Y(t)]`` of a continuous treatment, with the
    kernel-localised double machine learning estimator of Colangelo and Lee (2023).

    At every treatment value ``t`` of ``grid`` the estimate is the mean of the doubly robust score

    ``γ(t, X) + K_h(D − t) / f(t | X) · (Y − γ(t, X))``

    with cross-fitted ``γ(t, X) = E[Y | D = t, X]`` (``ml_g``, trained on ``[d, X]``) and
    generalised propensity score ``f(t | X)``, the conditional density of the treatment
    (``ml_m``). The estimates converge at the nonparametric rate ``√(nh)``; their standard errors
    come from the estimated influence function. All grid points share the sample splits, so
    ``bootstrap`` followed by ``confint(joint=True)`` gives a uniform confidence band for the
    curve, and `contrast` and `partial_effect` have correct joint inference.

    After ``fit``: ``grid_``, ``bandwidth_``, ``eta_`` and the fitted-model API of the curve
    (``coef_``, ``se_``, ``summary``, ``confint``, ``bootstrap``, ``predictions_`` with
    ``"ml_g"`` = ``γ̂(t, X_i)`` and ``"density"`` = ``f̂(t | X_i)`` per grid point).

    Parameters
    ----------
    ml_g : regressor for ``E[Y | D, X]``, fitted on ``[d, X]`` (treatment in the first column).
    ml_m : regressor for the density model: with ``density="kernel"`` (Colangelo and Lee's
        MultiGPS) it learns ``E[φ((D − t)/h₁) | X]`` at every evaluation point, with
        ``density="normal"`` the mean ``E[D | X]`` of a normal model with constant variance.
    grid : treatment values at which to estimate ``β(t)`` (default: the deciles of ``d``).
    kernel : ``"epanechnikov"`` (default) or ``"gaussian"``.
    bandwidth : bandwidth ``h`` in units of the treatment. Default:
        ``bandwidth_scale · std(d) · n^(-1/5)``; values of ``bandwidth_scale`` below the
        AMSE-optimal one undersmooth, which keeps the kernel bias small next to the standard error.
    bandwidth_scale : the constant of the default bandwidth (default 1.0).
    density : ``"kernel"`` (default) or ``"normal"``.
    density_bandwidth : bandwidth ``h₁`` of the kernel density model (default ``h``).
    trimming_threshold : densities of the standardised treatment ``d / std(d)`` below this value
        are raised to it (default 0.01).
    partial_effects : also estimate ``∂β(t)/∂t`` by ``(β̂(t + η/2) − β̂(t − η/2))/η``.
    eta : the step ``η`` (default ``h · n^(-1/6)``); setting it turns on ``partial_effects``.
    n_folds, n_rep : cross-fitting folds and repetitions of the sample split.
    random_state : seed for the sample splits.
    """

    def __init__(
        self,
        ml_g: Any,
        ml_m: Any,
        grid: Optional[Sequence[float]] = None,
        kernel: Literal["epanechnikov", "gaussian"] = "epanechnikov",
        bandwidth: Optional[float] = None,
        bandwidth_scale: float = 1.0,
        density: Literal["kernel", "normal"] = "kernel",
        density_bandwidth: Optional[float] = None,
        trimming_threshold: float = 1e-2,
        partial_effects: bool = False,
        eta: Optional[float] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.grid = grid
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_scale = bandwidth_scale
        self.density = density
        self.density_bandwidth = density_bandwidth
        self.trimming_threshold = trimming_threshold
        self.partial_effects = partial_effects
        self.eta = eta
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    @accepts_data("x", "y", "d")
    def fit(self, X: ArrayLike, y: ArrayLike, d: ArrayLike, folds: Optional[Folds] = None) -> Self:
        grid = None if self.grid is None else [float(t) for t in np.atleast_1d(np.asarray(self.grid, dtype=np.float64))]
        self._dose = _dml_rs.fit_dose_response(
            _as_1d(y, "y"), _as_2d(_as_1d(d, "d"), "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            grid=grid, kernel=self.kernel, bandwidth=self.bandwidth, bandwidth_scale=self.bandwidth_scale,
            density=self.density, density_bandwidth=self.density_bandwidth,
            trimming_threshold=self.trimming_threshold, partial_effects=self.partial_effects, eta=self.eta,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state, splits=_splits(folds),
        )
        self.grid_ = np.asarray(self._dose.grid)
        self.bandwidth_ = self._dose.bandwidth
        self.eta_ = self._dose.eta
        return self._store(self._dose.curve)

    def partial_effect(self) -> Framework:
        """The partial effects ``∂β(t)/∂t`` at the grid points (fit with ``partial_effects=True``)."""
        self._check_fitted()
        effects = self._dose.partial_effects
        if effects is None:
            raise ValueError("fit with partial_effects=True (or eta=...) to estimate partial effects")
        return Framework(effects)

    def contrast(self, reference: float) -> Framework:
        """``β(t) − β(reference)`` for every other grid point; ``reference`` must be in ``grid_``."""
        self._check_fitted()
        return Framework(self._dose.contrast(float(reference)))

    def plot(
        self,
        level: float = 0.95,
        joint: bool = False,
        ax: Any = None,
        color: str = _COLORBLIND[0],
    ) -> Any:
        """Plot the estimated curve with pointwise (or, after ``bootstrap``, ``joint=True``
        uniform) confidence bands. Needs matplotlib; returns the axes."""
        self._check_fitted()
        ci = self.confint(level, joint=joint)
        if ax is None:
            ax = _pyplot().subplots()[1]
        label = f"{level:.0%} {'uniform' if joint else 'pointwise'} band"
        ax.fill_between(self.grid_, ci[:, 0], ci[:, 1], color=color, alpha=0.2, label=label)
        ax.plot(self.grid_, self.coef_, marker="o", color=color, label="β̂(t)")
        ax.set_xlabel("treatment t")
        ax.set_ylabel("E[Y(t)]")
        ax.legend()
        return ax

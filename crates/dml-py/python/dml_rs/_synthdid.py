"""Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens and Wager, 2021)."""

from __future__ import annotations

import math
import warnings
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np

from . import _dml_rs
from ._models import _pyplot, _COLORBLIND

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

_NAMES = {"sdid": "synthetic DiD", "sc": "synthetic control", "did": "DiD"}


class SyntheticDID:
    """Synthetic difference-in-differences for a block design, with synthetic control and plain
    DiD as special cases: a port of the authors' R package ``synthdid`` that reproduces its
    estimates, weights and standard errors.

    ``N1`` units are treated from period ``T0 + 1`` on, ``N0`` units never. The estimate is a
    weighted DiD: unit weights make the controls' (intercept-shifted) pre-treatment path match
    the treated average, time weights make the pre-treatment periods predict the controls'
    post-treatment average, both fitted on the simplex with ``synthdid``'s penalties,
    sparsified and refitted. With ``estimator="sc"`` it is the synthetic control estimator
    (unit weights without intercept, no time weights), with ``"did"`` the unweighted DiD.
    Time-varying covariates enter as ``Y − Σ β_k X_k`` with ``β`` fitted jointly.

    After ``fit``: ``coef_`` (one entry, the average effect on the treated in the post periods),
    ``se_``, ``summary``, ``confint``, ``unit_weights_``, ``time_weights_``, ``beta_``,
    ``effect_curve_`` (effect in each post period), `se` for other methods and `plot`.

    Parameters
    ----------
    estimator : ``"sdid"`` (default), ``"sc"`` or ``"did"``.
    se_method : standard error computed by ``fit``: ``"placebo"`` (default; needs more controls
        than treated units), ``"jackknife"`` or ``"bootstrap"`` (both need at least two treated
        units), or ``None``.
    replications : replications of the placebo or bootstrap (default 200).
    random_state : seed of the placebo or bootstrap draws; the draws equal R's after
        ``set.seed(random_state)``.
    noise_level, eta_omega, eta_lambda, min_decrease, max_iter, sparsify : ``synthdid``'s
        regularisation and solver settings (defaults as ``synthdid_estimate``).
    """

    def __init__(
        self,
        estimator: Literal["sdid", "sc", "did"] = "sdid",
        se_method: Optional[Literal["placebo", "jackknife", "bootstrap"]] = "placebo",
        replications: int = 200,
        random_state: int = 0,
        noise_level: Optional[float] = None,
        eta_omega: Optional[float] = None,
        eta_lambda: Optional[float] = None,
        min_decrease: Optional[float] = None,
        max_iter: Optional[int] = None,
        sparsify: bool = True,
    ) -> None:
        self.estimator = estimator
        self.se_method = se_method
        self.replications = replications
        self.random_state = random_state
        self.noise_level = noise_level
        self.eta_omega = eta_omega
        self.eta_lambda = eta_lambda
        self.min_decrease = min_decrease
        self.max_iter = max_iter
        self.sparsify = sparsify

    def fit(self, Y: ArrayLike, n_control: int, n_pre: int, X: Optional[ArrayLike] = None,
            units: Optional[Sequence[Any]] = None, periods: Optional[Sequence[Any]] = None) -> Self:
        """Fit on the outcome matrix ``Y`` (units × periods) with the ``n_control`` control units
        in the first rows and the ``n_pre`` pre-treatment periods in the first columns; ``X``
        are covariates of shape (units, periods, covariates). ``units`` and ``periods`` label
        the weights."""
        y = np.ascontiguousarray(Y, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"Y must be a units × periods matrix, got shape {y.shape}")
        x = None
        if X is not None:
            x = np.ascontiguousarray(X, dtype=np.float64)
            if x.ndim == 2:
                x = x[:, :, None]
        self._result = _dml_rs.fit_synthdid(
            y, int(n_control), int(n_pre), x, estimator=self.estimator,
            noise_level=self.noise_level, eta_omega=self.eta_omega, eta_lambda=self.eta_lambda,
            min_decrease=self.min_decrease, max_iter=self.max_iter, sparsify=self.sparsify,
        )
        self.n_control_, self.n_pre_ = int(n_control), int(n_pre)
        self.units_ = list(range(y.shape[0])) if units is None else list(units)
        self.periods_ = list(range(y.shape[1])) if periods is None else list(periods)
        self.coef_ = np.array([self._result.estimate])
        self.unit_weights_ = self._labelled(self._result.unit_weights, self.units_[: self.n_control_])
        self.time_weights_ = self._labelled(self._result.time_weights, self.periods_[: self.n_pre_])
        self.beta_ = np.asarray(self._result.beta)
        self.effect_curve_ = self._labelled(self._result.effect_curve, self.periods_[self.n_pre_:])
        self.noise_level_ = self._result.noise_level
        self.se_ = np.array([np.nan])
        if self.se_method is not None:
            try:
                self.se_ = np.array([self.se(self.se_method)])
            except ValueError as err:
                warnings.warn(f"no {self.se_method} standard error: {err}", stacklevel=2)
        return self

    def fit_panel(self, data: pd.DataFrame, unit: str, time: str, outcome: str, treatment: str,
                  covariates: Optional[Sequence[str]] = None) -> Self:
        """Fit on a balanced long panel with a 0/1 ``treatment`` column (simultaneous adoption),
        as ``synthdid::panel.matrices``: controls first, then treated units, each sorted by
        unit label."""
        cols = [unit, time, outcome, treatment, *(covariates or [])]
        df = data[cols]
        if df.isna().any().any():
            raise ValueError("missing values in the panel")
        if not df[treatment].isin([0, 1]).all():
            raise ValueError("the treatment must be 0 or 1")
        if df[treatment].nunique() == 1:
            raise ValueError("there is no variation in treatment status")
        if df.groupby([unit, time]).size().max() != 1 or \
                len(df) != df[unit].nunique() * df[time].nunique():
            raise ValueError("the panel must be balanced: one observation per unit and period")
        wide = df.pivot(index=unit, columns=time, values=outcome).sort_index().sort_index(axis=1)
        w = df.pivot(index=unit, columns=time, values=treatment).loc[wide.index, wide.columns]
        treated = w.any(axis=1).to_numpy()
        first = int(np.argmax(w.any(axis=0).to_numpy()))
        wv = w.to_numpy()
        if not ((wv[~treated] == 0).all() and (wv[:, :first] == 0).all() and (wv[treated, first:] == 1).all()):
            raise ValueError("treatment adoption is not simultaneous")
        order = sorted(range(len(wide.index)), key=lambda i: (bool(treated[i]), str(wide.index[i])))
        y = wide.to_numpy()[order]
        x = None
        if covariates:
            x = np.stack([df.pivot(index=unit, columns=time, values=c).loc[wide.index, wide.columns]
                          .to_numpy()[order] for c in covariates], axis=2)
        return self.fit(y, int((~treated).sum()), first, x,
                        units=[wide.index[i] for i in order], periods=list(wide.columns))

    @staticmethod
    def _labelled(values: Any, labels: Sequence[Any]) -> Any:
        try:
            import pandas as pd
        except ImportError:
            return np.asarray(values)
        return pd.Series(np.asarray(values), index=list(labels))

    def _check_fitted(self) -> None:
        if not hasattr(self, "_result"):
            raise RuntimeError("call fit first")

    def se(self, method: Literal["placebo", "jackknife", "bootstrap"] = "placebo",
           replications: Optional[int] = None, random_state: Optional[int] = None) -> float:
        """Standard error by ``method`` (the placebo and bootstrap draws equal R's after
        ``set.seed(random_state)``)."""
        self._check_fitted()
        reps = self.replications if replications is None else replications
        seed = self.random_state if random_state is None else random_state
        return float(self._result.se(method, int(reps), int(seed)))

    def confint(self, level: float = 0.95) -> NDArray[np.float64]:
        """Normal confidence interval from ``se_``, shape (1, 2)."""
        self._check_fitted()
        z = NormalDist().inv_cdf(0.5 + level / 2)
        return np.array([[self.coef_[0] - z * self.se_[0], self.coef_[0] + z * self.se_[0]]])

    @property
    def summary(self) -> Any:
        """The estimate with standard error, t statistic, p-value and 95% interval."""
        self._check_fitted()
        est, se = float(self.coef_[0]), float(self.se_[0])
        t = est / se if se > 0 else math.nan
        p = 2 * (1 - NormalDist().cdf(abs(t))) if not math.isnan(t) else math.nan
        lo, hi = self.confint()[0]
        row = {"coef": est, "std err": se, "t": t, "P>|t|": p, "2.5 %": lo, "97.5 %": hi}
        try:
            import pandas as pd
        except ImportError:
            return {_NAMES[self.estimator]: row}
        return pd.DataFrame([row], index=[_NAMES[self.estimator]])

    def plot(self, ax: Any = None, colors: Sequence[str] = (_COLORBLIND[0], _COLORBLIND[1])) -> Any:
        """Treated average and synthetic control trajectories, with the time weights as bars and
        the treatment start marked. Needs matplotlib; returns the axes."""
        self._check_fitted()
        if ax is None:
            ax = _pyplot().subplots()[1]
        synthetic, treated = self._result.trajectories()
        x = np.arange(len(self.periods_))
        ax.plot(x, treated, color=colors[0], label="treated")
        ax.plot(x, synthetic, color=colors[1], label=f"{_NAMES[self.estimator]} control")
        ax.axvline(self.n_pre_ - 0.5, color="grey", ls="--", lw=0.8)
        lam = np.asarray(self._result.time_weights)
        if lam.sum() > 0:
            lo, hi = ax.get_ylim()
            ax.bar(x[: self.n_pre_], lam * 0.15 * (hi - lo), bottom=lo, color="grey", alpha=0.4,
                   width=0.8, label="time weights")
        ticks = x[:: max(1, len(x) // 10)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(self.periods_[i]) for i in ticks])
        ax.set_ylabel("outcome")
        ax.legend()
        return ax

"""Anytime-valid inference: confidence sequences over the looks of a study."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from ._models import _DMLEstimator, _pyplot

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from typing_extensions import Self


class ConfidenceSequence:
    """Confidence sequences for monitoring a study as data come in (Waudby-Smith, Arbour,
    Sinha, Kennedy & Ramdas, 2024, *Time-uniform central limit theory and asymptotic
    confidence sequences*).

    A 95 % confidence interval covers θ with probability 0.95 at one sample size fixed in
    advance. Checking it after every batch of data and stopping once it excludes zero inflates
    the error rate far beyond 5 %. A confidence sequence covers θ at *all* sample sizes at once,
    ``P(θ ∈ C_t for every t) ≥ level``, so the study can be checked as often as wanted and
    stopped at any time. The asymptotic sequence works for any estimator here: at each look,
    ``θ̂_t ± se_t · √(2(u+1)/u · log(√(u+1)/α))`` with ``u = t ρ²`` (the Robbins normal
    mixture), which covers doubly robust estimators with cross-fitted nuisances (their
    Theorem 3).

    Refit the model on all data so far at each look and pass it to `update`, or let `run` do
    the refitting. The intervals of all looks hold together, so their running intersection does
    too (``lower_``, ``upper_``), and the running minimum of the anytime p-values (``pval_``) is a
    valid p-value whenever the study stops.

    After updates: ``n_`` (sample size per look; clusters for clustered fits), ``coef_``,
    ``se_``, ``intervals_`` (per look, shape ``(looks, n_coefs, 2)``), ``lower_`` / ``upper_``
    (running intersection, shape ``(looks, n_coefs)``), ``pval_``, `first_exclusion`, `summary`
    and `plot`.

    Parameters
    ----------
    level : confidence level (default 0.95).
    opt_time : sample size where the sequence is tightest, to fix before seeing the data
        (default: the sample size of the first look). Around ``opt_time`` the half-width at 95 %
        is about 3.0 standard errors, against 1.96 for a fixed-sample interval; it grows like
        ``√log t`` away from it.
    """

    def __init__(self, level: float = 0.95, opt_time: Optional[int] = None) -> None:
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        if opt_time is not None and int(opt_time) < 1:
            raise ValueError("opt_time must be a positive integer")
        self.level = level
        self.opt_time = opt_time
        self._looks: list[tuple[int, Any, Any, Any, Any]] = []
        self._names: Optional[list[str]] = None

    def update(self, model: Any) -> Self:
        """Add a look: a fitted estimator (or `Framework`) on all data so far."""
        if not isinstance(model, _DMLEstimator):
            raise TypeError(f"expected a fitted dml_rs estimator, got {type(model).__name__}")
        result = model._check_fitted()
        t = result.sequence_time
        if self._looks and t <= self._looks[-1][0]:
            raise ValueError(f"looks need growing sample sizes: {t} after {self._looks[-1][0]}")
        if self._names is not None and list(result.treatment_names) != self._names:
            raise ValueError("every look needs the same estimates")
        if not self._looks:
            # Fixed at the first look: changing it later would void the guarantee.
            self.opt_time_ = t if self.opt_time is None else int(self.opt_time)
        self._names = list(result.treatment_names)
        self._looks.append((t, np.asarray(result.coef), np.asarray(result.se),
                            np.asarray(result.confidence_sequence(self.level, self.opt_time_)),
                            np.asarray(result.anytime_pval(self.level, self.opt_time_))))
        return self

    def run(self, estimator: Any, looks: Sequence[int], **data: Any) -> Self:
        """Refit a copy of ``estimator`` on the first ``t`` rows of the data for every ``t`` in
        ``looks`` and add each fit. ``data`` are the keyword arguments of ``estimator.fit``
        (e.g. ``X=X, y=y, d=d``); arrays, DataFrames and Series with one row per observation are
        cut, other arguments passed on unchanged. The rows must be in arrival order."""
        lengths = [len(v) for v in data.values() if _is_rows(v)]
        if not lengths:
            raise ValueError("pass the data as keyword arguments of estimator.fit")
        n = max(lengths)
        looks = [int(t) for t in looks]
        if any(t < 1 or t > n for t in looks) or any(b <= a for a, b in zip(looks, looks[1:])):
            raise ValueError(f"looks must increase and lie in 1..{n}")
        for t in looks:
            model = type(estimator)(**estimator.get_params(deep=False))
            model.fit(**{k: _head(v, t) if _is_rows(v) and len(v) == n else v for k, v in data.items()})
            self.update(model)
        return self

    def _check(self) -> None:
        if not self._looks:
            raise RuntimeError("no looks yet: call update or run first")

    @property
    def n_(self) -> NDArray[np.int64]:
        self._check()
        return np.array([look[0] for look in self._looks], dtype=np.int64)

    @property
    def coef_(self) -> NDArray[np.float64]:
        self._check()
        return np.array([look[1] for look in self._looks])

    @property
    def se_(self) -> NDArray[np.float64]:
        self._check()
        return np.array([look[2] for look in self._looks])

    @property
    def intervals_(self) -> NDArray[np.float64]:
        """The sequence at each look, shape ``(looks, n_coefs, 2)``."""
        self._check()
        return np.array([look[3] for look in self._looks])

    @property
    def lower_(self) -> NDArray[np.float64]:
        """Running maximum of the lower ends, shape ``(looks, n_coefs)``."""
        return np.maximum.accumulate(self.intervals_[:, :, 0], axis=0)

    @property
    def upper_(self) -> NDArray[np.float64]:
        """Running minimum of the upper ends, shape ``(looks, n_coefs)``. Below ``lower_`` (an
        empty intersection) the assumptions behind the sequence are in doubt."""
        return np.minimum.accumulate(self.intervals_[:, :, 1], axis=0)

    @property
    def pval_(self) -> NDArray[np.float64]:
        """Anytime-valid p-values for ``θ = 0``: the running minimum over looks."""
        self._check()
        return np.minimum.accumulate(np.array([look[4] for look in self._looks]), axis=0)

    def first_exclusion(self, value: float = 0.0) -> list[Optional[int]]:
        """For each estimate, the sample size of the first look whose running interval excludes
        ``value`` (when a study testing ``θ = value`` could stop), or ``None``."""
        lower, upper, n = self.lower_, self.upper_, self.n_
        out: list[Optional[int]] = []
        for j in range(lower.shape[1]):
            hit = np.flatnonzero((lower[:, j] > value) | (upper[:, j] < value))
            out.append(int(n[hit[0]]) if hit.size else None)
        return out

    @property
    def summary(self) -> pd.DataFrame:
        """One row per look and estimate: estimate, standard error, running interval and
        anytime p-value."""
        n, coef, se, lower, upper, p = self.n_, self.coef_, self.se_, self.lower_, self.upper_, self.pval_
        names = self._names or []
        rows = [(int(n[l]), names[j], coef[l, j], se[l, j], lower[l, j], upper[l, j], p[l, j])
                for l in range(len(n)) for j in range(len(names))]
        pct = f"{self.level:.0%}"
        columns = ["n", "coef", "std err", f"{pct} CS lower", f"{pct} CS upper", "anytime P"]
        try:
            import pandas as pd
        except ImportError:
            return {"index": [(r[0], r[1]) for r in rows],  # type: ignore[return-value]  # no pandas
                    **{c: [r[i + 2] for r in rows] for i, c in enumerate(columns[1:])}}
        frame = pd.DataFrame([[r[0], *r[2:]] for r in rows], columns=columns,
                             index=pd.Index([r[1] for r in rows], name="estimate"))
        return frame

    def plot(self, treatment: int = 0, value: Optional[float] = 0.0, ax: Any = None) -> Any:
        """Estimates and the running confidence sequence against the sample size, with a
        reference line at ``value`` (``None`` for none). Returns the axes."""
        plt = _pyplot()
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        n = self.n_
        ax.fill_between(n, self.lower_[:, treatment], self.upper_[:, treatment], color="#0173B2",
                        alpha=0.2, step="post", label=f"{self.level:.0%} confidence sequence")
        ax.plot(n, self.coef_[:, treatment], color="#0173B2", marker="o", markersize=3, label="estimate")
        if value is not None:
            ax.axhline(value, color="#666666", linewidth=1, linestyle="--")
        ax.set_xlabel("sample size")
        ax.set_ylabel((self._names or ["θ"])[treatment])
        ax.legend(frameon=False)
        return ax


def _is_rows(v: Any) -> bool:
    return hasattr(v, "__len__") and not isinstance(v, (str, bytes, dict)) and hasattr(v, "__getitem__")


def _head(v: Any, t: int) -> Any:
    if hasattr(v, "iloc"):
        return v.iloc[:t]
    return np.asarray(v)[:t]

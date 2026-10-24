"""Best-linear-predictor inference on CATEs and GATEs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import Self

    # Folds: one repetition ``[(train, test), ...]`` or a list of repetitions.
    Folds = Sequence[Any]

from ._validation import _as_2d


def _labels(obj, n, prefix):
    columns = getattr(obj, "columns", None)
    return [str(c) for c in columns] if columns is not None else [f"{prefix}{i}" for i in range(n)]


class BLP:
    """Best linear predictor of the CATE (or GATEs) with HC0 inference.

    Mirrors DoubleML's ``DoubleMLBLP``: ``confint()`` gives coefficient (or, for
    GATEs, group) intervals; ``confint(basis)`` gives intervals for the CATE at
    the rows of a new basis, optionally simultaneous with ``joint=True``.
    """

    def __init__(self, native: Any, names: list[str]) -> None:
        self._native = native
        self.names = names

    coef = property(lambda self: self._native.coef)
    se = property(lambda self: self._native.se)
    t_stat = property(lambda self: self._native.t_stat)
    pval = property(lambda self: self._native.pval)

    def predict(self, basis: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(effect, se)`` at the rows of ``basis``."""
        return self._native.predict(_as_2d(basis, "basis"))

    def confint(
        self,
        basis: Optional[ArrayLike] = None,
        joint: bool = False,
        level: float = 0.95,
        n_boot: int = 500,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Intervals with columns ``[lower, effect, upper]``."""
        if basis is None:
            if self._native.is_gate:
                basis = np.eye(len(self.names))
            else:
                ci = self._native.confint(level)
                table = np.column_stack([ci[:, 0], self.coef, ci[:, 1]])
                return self._frame(table, self.names, level)
            index: Optional[Sequence[Any]] = self.names
        else:
            index = getattr(basis, "index", None)
        b = _as_2d(basis, "basis")
        if joint:
            table = self._native.joint_confint_at(b, level, n_boot, random_state)
        else:
            table = self._native.confint_at(b, level)
        return self._frame(table, index, level)

    @staticmethod
    def _frame(table, index, level):
        alpha = 1 - level
        columns = [f"{alpha / 2 * 100:.1f} %", "effect", f"{(1 - alpha / 2) * 100:.1f} %"]
        try:
            import pandas as pd
        except ImportError:
            return table
        return pd.DataFrame(table, columns=columns, index=None if index is None else list(index))

    @property
    def summary(self) -> pd.DataFrame:
        ci = self._native.confint(0.95)
        table = {"coef": self.coef, "std err": self.se, "t": self.t_stat, "P>|t|": self.pval,
                 "[0.025": ci[:, 0], "0.975]": ci[:, 1]}
        try:
            import pandas as pd
        except ImportError:
            return {"index": self.names, **table}  # type: ignore[return-value]  # no pandas
        return pd.DataFrame(table, index=self.names)


def blp_from(native, basis, prefix):
    """Wrap a native ``Blp`` with labels taken from a pandas basis, if any."""
    return BLP(native, _labels(basis, _as_2d(basis, "basis").shape[1], prefix))

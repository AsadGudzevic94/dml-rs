"""Best-linear-predictor inference on CATEs and GATEs."""

from __future__ import annotations

import numpy as np

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

    def __init__(self, native, names):
        self._native = native
        self.names = names

    coef = property(lambda self: self._native.coef)
    se = property(lambda self: self._native.se)
    t_stat = property(lambda self: self._native.t_stat)
    pval = property(lambda self: self._native.pval)

    def predict(self, basis):
        """``(effect, se)`` at the rows of ``basis``."""
        return self._native.predict(_as_2d(basis, "basis"))

    def confint(self, basis=None, joint=False, level=0.95, n_boot=500, random_state=0):
        """Intervals with columns ``[lower, effect, upper]``."""
        if basis is None:
            if self._native.is_gate:
                basis = np.eye(len(self.names))
            else:
                ci = self._native.confint(level)
                table = np.column_stack([ci[:, 0], self.coef, ci[:, 1]])
                return self._frame(table, self.names, level)
            index = self.names
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
    def summary(self):
        ci = self._native.confint(0.95)
        table = {"coef": self.coef, "std err": self.se, "t": self.t_stat, "P>|t|": self.pval,
                 "[0.025": ci[:, 0], "0.975]": ci[:, 1]}
        try:
            import pandas as pd
        except ImportError:
            return {"index": self.names, **table}
        return pd.DataFrame(table, index=self.names)


def blp_from(native, basis, prefix):
    """Wrap a native ``Blp`` with labels taken from a pandas basis, if any."""
    return BLP(native, _labels(basis, _as_2d(basis, "basis").shape[1], prefix))

"""Policy trees: treatment recommendations from the estimated effects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._validation import _as_2d

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class PolicyTree:
    """A fitted policy tree, as DoubleML's ``DoubleMLPolicyTree``.

    ``predict`` recommends treatment (1) or not (0); ``str(tree)`` shows the decision rules.
    """

    def __init__(self, native: Any, features: Any) -> None:
        self._native = native
        columns = getattr(features, "columns", None)
        self.feature_names = [str(c) for c in columns] if columns is not None else None

    @property
    def depth(self) -> int:
        return int(self._native.depth)

    @property
    def n_leaves(self) -> int:
        return int(self._native.n_leaves)

    def predict(self, features: ArrayLike) -> Any:
        """Recommended treatment for each row; for a DataFrame, the frame with a ``pred_treatment``
        column (as DoubleML), otherwise an integer array."""
        pred = np.asarray(self._native.predict(_as_2d(features, "features")), dtype=int)
        if hasattr(features, "assign"):
            return features.assign(pred_treatment=pred)
        return pred

    def __str__(self) -> str:
        return str(self._native.export_text(self.feature_names))

    def __repr__(self) -> str:
        return f"PolicyTree(depth={self.depth}, n_leaves={self.n_leaves})"

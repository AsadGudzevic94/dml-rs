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

    def plot_tree(self, ax: Any = None, fontsize: int = 9) -> Any:
        """Draw the tree with matplotlib (installed separately), as DoubleML's ``plot_tree``:
        splits show their rule, leaves the recommendation and the share of positive effects
        (blue: treat, orange: do not treat); left branches are the rule's "True" side. Returns the
        axes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError("plot_tree needs matplotlib: pip install matplotlib") from err
        nodes = self._native.nodes()
        names = self.feature_names
        # Leaves get consecutive x positions from left to right; splits sit above their children.
        pos: dict[int, tuple[float, int]] = {}
        next_leaf = [0]

        def place(i: int, depth: int) -> float:
            feature, _, left, right, _ = nodes[i]
            if feature < 0:
                x = float(next_leaf[0])
                next_leaf[0] += 1
            else:
                x = (place(left, depth + 1) + place(right, depth + 1)) / 2
            pos[i] = (x, depth)
            return x

        place(0, 0)
        n_leaves = next_leaf[0]
        depth_max = max(d for _, d in pos.values())
        if ax is None:
            _, ax = plt.subplots(figsize=(max(5.0, 2.8 * n_leaves), 1.5 * (depth_max + 1) + 0.3))
        treat, no_treat = "#0173b2", "#de8f05"  # colorblind palette
        for i, (feature, threshold, left, right, share) in enumerate(nodes):
            x, depth = pos[i]
            if feature >= 0:
                name = names[feature] if names is not None else f"x{feature}"
                text, color = f"{name} <= {threshold:.3f}", "white"
                for child, label in ((left, "True"), (right, "False")):
                    cx, cdepth = pos[child]
                    ax.annotate("", xy=(cx, -cdepth + 0.25), xytext=(x, -depth - 0.25),
                                arrowprops={"arrowstyle": "->", "color": "0.4"})
                    ax.text((x + cx) / 2, -(depth + cdepth) / 2, label, ha="center", va="center",
                            fontsize=fontsize - 1, color="0.4", backgroundcolor="white")
            else:
                recommend = share > 0.5
                label = "Treatment" if recommend else "No Treatment"
                text = f"{label}\nshare positive {share:.2f}"
                color = treat if recommend else no_treat
            ax.text(x, -depth, text, ha="center", va="center", fontsize=fontsize,
                    color="white" if feature < 0 else "black",
                    bbox={"boxstyle": "round,pad=0.4", "facecolor": color, "edgecolor": "0.3"})
        ax.set_xlim(-0.6, n_leaves - 0.4)
        ax.set_ylim(-depth_max - 0.5, 0.5)
        ax.axis("off")
        return ax

    def __str__(self) -> str:
        return str(self._native.export_text(self.feature_names))

    def __repr__(self) -> str:
        return f"PolicyTree(depth={self.depth}, n_leaves={self.n_leaves})"

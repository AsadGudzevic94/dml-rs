"""A data container for the estimators, as DoubleML's ``DoubleMLData`` family.

`DmlData` names the roles of the columns of a DataFrame (outcome, treatments, covariates and,
depending on the design, instruments, clusters, time, selection indicator, unit identifier or
running variable). Every estimator's ``fit`` accepts it in place of the arrays::

    data = dml_rs.DmlData(df, y_col="y", d_cols="d", x_cols=["x1", "x2"])
    dml_rs.PLR(ml_l, ml_m).fit(data)

One class covers DoubleML's ``DoubleMLData``, ``DoubleMLClusterData``, ``DoubleMLDIDData``,
``DoubleMLPanelData``, ``DoubleMLSSMData`` and ``DoubleMLRDDData``: pass the columns the design
needs.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import ArrayLike

__all__ = ["DmlData"]

Columns = Union[str, Sequence[str], None]


def _as_list(cols: Columns) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return [str(c) for c in cols]


class DmlData:
    """Columns of a DataFrame in their roles, as DoubleML's ``DoubleMLData``.

    Parameters
    ----------
    data : pandas.DataFrame
        The data.
    y_col : str
        Outcome column.
    d_cols : str or list of str
        Treatment column(s); for `StaggeredDID` the first treated period (``inf`` or missing for
        never treated), as DoubleML's ``DoubleMLPanelData``.
    x_cols : str or list of str, optional
        Covariates; by default all columns not used in another role.
    z_cols : str or list of str, optional
        Instrument column(s).
    cluster_cols : str or list of str, optional
        One or two cluster columns.
    t_col : str, optional
        Time: the period of a panel or staggered DiD, or the 0/1 post-period indicator of
        repeated cross-sections (`DIDCS`).
    s_col : str, optional
        Selection indicator (`SSM`).
    id_col : str, optional
        Unit identifier of a panel (`PLPR`, `StaggeredDID`).
    score_col : str, optional
        Running variable of a regression discontinuity design (`RDFlex`).
    """


    def __init__(
        self,
        data: pd.DataFrame,
        y_col: str,
        d_cols: Columns,
        x_cols: Columns = None,
        z_cols: Columns = None,
        cluster_cols: Columns = None,
        t_col: Optional[str] = None,
        s_col: Optional[str] = None,
        id_col: Optional[str] = None,
        score_col: Optional[str] = None,
    ) -> None:
        if not hasattr(data, "columns"):
            raise TypeError("data must be a pandas DataFrame (use DmlData.from_arrays for arrays)")
        self.data = data
        self.y_col = str(y_col)
        self.d_cols = _as_list(d_cols)
        self.z_cols = _as_list(z_cols)
        self.cluster_cols = _as_list(cluster_cols)
        self.t_col = t_col
        self.s_col = s_col
        self.id_col = id_col
        self.score_col = score_col
        if not self.d_cols:
            raise ValueError("d_cols must name at least one treatment column")
        if len(self.cluster_cols) > 2:
            raise ValueError("at most two cluster columns are supported")
        used = [self.y_col, *self.d_cols, *self.z_cols, *self.cluster_cols,
                *[c for c in (t_col, s_col, id_col, score_col) if c is not None]]
        missing = [c for c in used + _as_list(x_cols) if c not in data.columns]
        if missing:
            raise ValueError(f"columns not in the data: {missing}")
        if len(set(used)) != len(used):
            raise ValueError("a column cannot have two roles")
        if x_cols is None:
            self.x_cols = [str(c) for c in data.columns if c not in used]
        else:
            self.x_cols = _as_list(x_cols)
            overlap = set(self.x_cols) & set(used)
            if overlap:
                raise ValueError(f"covariates cannot have another role: {sorted(overlap)}")
        if not self.x_cols:
            raise ValueError("no covariates: pass x_cols or leave other columns in the data")

    @classmethod
    def from_arrays(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        z: Optional[ArrayLike] = None,
        cluster_vars: Optional[ArrayLike] = None,
        t: Optional[ArrayLike] = None,
        s: Optional[ArrayLike] = None,
        id: Optional[ArrayLike] = None,
        score: Optional[ArrayLike] = None,
    ) -> DmlData:
        """Build from arrays with DoubleML's default column names (``X1, ...``, ``y``, ``d`` or
        ``d1, ...``, ``z`` or ``z1, ...``, ``cluster_var`` or ``cluster_var1, cluster_var2``,
        ``t``, ``s``, ``id``, ``score``)."""
        import pandas as pd

        def columns(values: Any, name: str) -> dict[str, Any]:
            arr = np.asarray(values)
            if arr.ndim == 1:
                return {name: arr}
            if arr.shape[1] == 1:
                return {name: arr[:, 0]}
            return {f"{name}{i + 1}": arr[:, i] for i in range(arr.shape[1])}

        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(-1, 1)
        cols: dict[str, Any] = {f"X{i + 1}": x_arr[:, i] for i in range(x_arr.shape[1])}
        cols["y"] = np.asarray(y)
        d_part = columns(d, "d")
        cols.update(d_part)
        roles: dict[str, Any] = {}
        if z is not None:
            part = columns(z, "z")
            cols.update(part)
            roles["z_cols"] = list(part)
        if cluster_vars is not None:
            part = columns(cluster_vars, "cluster_var")
            cols.update(part)
            roles["cluster_cols"] = list(part)
        for name, values in (("t", t), ("s", s), ("id", id), ("score", score)):
            if values is not None:
                cols[name] = np.asarray(values)
                roles[f"{name}_col"] = name
        return cls(pd.DataFrame(cols), "y", list(d_part), list(cols)[: x_arr.shape[1]], **roles)

    # -- arrays in their roles -------------------------------------------------
    @property
    def n_obs(self) -> int:
        return len(self.data)

    @property
    def n_treat(self) -> int:
        return len(self.d_cols)

    @property
    def x(self) -> pd.DataFrame:
        """Covariates (a DataFrame, so that column names reach the estimators)."""
        return self.data[self.x_cols]

    @property
    def y(self) -> pd.Series:
        return self.data[self.y_col]

    @property
    def d(self) -> Any:
        """The treatment (a Series), or a DataFrame for several treatments."""
        return self.data[self.d_cols[0]] if self.n_treat == 1 else self.data[self.d_cols]

    @property
    def z(self) -> Any:
        if not self.z_cols:
            return None
        return self.data[self.z_cols[0]] if len(self.z_cols) == 1 else self.data[self.z_cols]

    @property
    def cluster(self) -> Any:
        if not self.cluster_cols:
            return None
        return self.data[self.cluster_cols[0]] if len(self.cluster_cols) == 1 else self.data[self.cluster_cols]

    @property
    def t(self) -> Any:
        return None if self.t_col is None else self.data[self.t_col]

    @property
    def s(self) -> Any:
        return None if self.s_col is None else self.data[self.s_col]

    @property
    def id(self) -> Any:
        return None if self.id_col is None else self.data[self.id_col]

    @property
    def score(self) -> Any:
        return None if self.score_col is None else self.data[self.score_col]

    def _role(self, role: str, estimator: str) -> Any:
        optional = role.endswith("?")
        name = role.rstrip("?")
        value = getattr(self, name)
        if value is None and not optional:
            col = {"z": "z_cols", "cluster": "cluster_cols"}.get(name, f"{name}_col")
            raise ValueError(f"{estimator} needs the DmlData to have {col}")
        return value

    def __repr__(self) -> str:
        roles = [f"y_col={self.y_col!r}", f"d_cols={self.d_cols}", f"x_cols={self.x_cols}"]
        for name in ("z_cols", "cluster_cols"):
            if getattr(self, name):
                roles.append(f"{name}={getattr(self, name)}")
        for name in ("t_col", "s_col", "id_col", "score_col"):
            if getattr(self, name) is not None:
                roles.append(f"{name}={getattr(self, name)!r}")
        return f"DmlData(n_obs={self.n_obs}, {', '.join(roles)})"


F = TypeVar("F", bound=Callable[..., Any])


def accepts_data(*roles: str, cluster: bool = False) -> Callable[[F], F]:
    """Let ``fit`` take a `DmlData` in place of its leading arrays: ``roles`` name them in order
    (``"z?"`` is optional); with ``cluster``, the data's cluster columns are passed as
    ``cluster=`` unless given."""

    def wrap(fit: F) -> F:
        @functools.wraps(fit)
        def inner(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not (args and isinstance(args[0], DmlData)):
                return fit(self, *args, **kwargs)
            if len(args) > 1:
                raise TypeError("with a DmlData, pass the other arguments of fit by keyword")
            data = args[0]
            name = type(self).__name__
            values = [data._role(role, name) for role in roles]
            if cluster and data.cluster is not None:
                kwargs.setdefault("cluster", data.cluster)
            elif data.cluster is not None:
                raise ValueError(f"{name} does not support clustered data")
            return fit(self, *values, **kwargs)

        inner.__dml_roles__ = (roles, cluster)  # type: ignore[attr-defined]
        return inner  # type: ignore[return-value]

    return wrap

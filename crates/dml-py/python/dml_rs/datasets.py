"""Simulated data sets: the data generating processes of DoubleML's ``make_*`` functions.

The functions keep DoubleML's names, parameters and defaults. The data are drawn in Rust from a
seeded generator: pass ``random_state`` for reproducible data (without it, the seed is drawn from
NumPy's global generator, so ``np.random.seed`` makes the data reproducible as well). The draws
are not identical to DoubleML's, but the distributions are the same.

``return_type`` is ``"array"`` (a tuple of arrays, the default) or ``"DataFrame"`` (pandas, with
DoubleML's column names); DoubleML's data classes are not available. Generators that DoubleML
returns as a dict or a DataFrame return the same here.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import numpy as np

from . import _dml_rs

__all__ = [
    "make_plr_CCDDHNR2018",
    "make_plr_turrell2018",
    "make_confounded_plr_data",
    "make_pliv_CHS2015",
    "make_pliv_multiway_cluster_CKMS2021",
    "make_lplr_LZZ2020",
    "make_plpr_CP2025",
    "make_irm_data",
    "make_confounded_irm_data",
    "make_irm_data_discrete_treatments",
    "make_iivm_data",
    "make_ssm_data",
    "make_heterogeneous_data",
    "make_did_SZ2020",
    "make_did_CS2021",
    "make_did_cs_CS2021",
    "make_simple_rdd_data",
]



def _seed(random_state: Optional[int]) -> int:
    if random_state is None:
        return int(np.random.randint(0, 2**62, dtype=np.int64))
    if random_state < 0:
        raise ValueError("random_state must be non-negative")
    return int(random_state)


def _draw(name: str, random_state: Optional[int], **params: Any) -> dict[str, Any]:
    return _dml_rs.simulate(name, params, _seed(random_state))


def _pandas():
    try:
        import pandas as pd
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("this return_type needs pandas: pip install pandas") from err
    return pd


def _frame(columns: dict[str, Any]):
    return _pandas().DataFrame(columns)


def _xyd_frame(x, y, d, extra: Optional[dict[str, Any]] = None, x_prefix: str = "X"):
    cols = {f"{x_prefix}{i + 1}": x[:, i] for i in range(x.shape[1])}
    cols.update({"y": y, "d": d})
    cols.update(extra or {})
    return _frame(cols)


def _check_return_type(return_type: str) -> None:
    if return_type not in ("array", "DataFrame"):
        raise ValueError(f"return_type must be 'array' or 'DataFrame', got {return_type!r}")


def make_plr_CCDDHNR2018(
    n_obs: int = 500,
    dim_x: int = 20,
    alpha: float = 0.5,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
    **kwargs: float,
):
    """Partially linear regression of Chernozhukov et al. (2018):
    ``d = a_0 x_0 + a_1 expit(x_2) + s_1 e``, ``y = alpha d + b_0 expit(x_0) + b_1 x_2 + s_2 u``
    with Toeplitz-correlated covariates. ``kwargs``: ``a_0``, ``a_1``, ``s_1``, ``b_0``, ``b_1``,
    ``s_2``. Returns ``(x, y, d)``."""
    _check_return_type(return_type)
    r = _draw("plr_CCDDHNR2018", random_state, n_obs=n_obs, dim_x=dim_x, alpha=alpha, **kwargs)
    if return_type == "array":
        return r["x"], r["y"], r["d"]
    return _xyd_frame(r["x"], r["y"], r["d"])


def make_plr_turrell2018(
    n_obs: int = 100,
    dim_x: int = 20,
    theta: float = 0.5,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
    **kwargs: float,
):
    """Partially linear regression of Turrell (2018) with a random covariance matrix.
    ``kwargs``: ``nu``, ``gamma``. Returns ``(x, y, d)``."""
    _check_return_type(return_type)
    r = _draw("plr_turrell2018", random_state, n_obs=n_obs, dim_x=dim_x, theta=theta, **kwargs)
    if return_type == "array":
        return r["x"], r["y"], r["d"]
    return _xyd_frame(r["x"], r["y"], r["d"])


def make_confounded_plr_data(
    n_obs: int = 500,
    theta: float = 5.0,
    cf_y: float = 0.04,
    cf_d: float = 0.04,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Partially linear model with an unobserved confounder of strength ``(cf_y, cf_d)``, for
    sensitivity analysis. ``kwargs``: ``c``, ``dim_x``. Returns ``{"x", "y", "d",
    "oracle_values"}``."""
    r = _draw("confounded_plr_data", random_state, n_obs=n_obs, theta=theta, cf_y=cf_y, cf_d=cf_d, **kwargs)
    return {"x": r["x"], "y": r["y"], "d": r["d"], "oracle_values": r["oracle_values"]}


def make_pliv_CHS2015(
    n_obs: int,
    alpha: float = 1.0,
    dim_x: int = 200,
    dim_z: int = 150,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
):
    """Partially linear IV of Chernozhukov, Hansen and Spindler (2015) with ``dim_z``
    instruments. Returns ``(x, y, d, z)``."""
    _check_return_type(return_type)
    r = _draw("pliv_CHS2015", random_state, n_obs=n_obs, alpha=alpha, dim_x=dim_x, dim_z=dim_z)
    z = r["z"][:, 0] if dim_z == 1 else r["z"]
    if return_type == "array":
        return r["x"], r["y"], r["d"], z
    return _xyd_frame(r["x"], r["y"], r["d"], {f"Z{i + 1}": r["z"][:, i] for i in range(dim_z)})


def make_pliv_multiway_cluster_CKMS2021(
    N: int = 25,
    M: int = 25,
    dim_X: int = 100,
    theta: float = 1.0,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
    **kwargs: Any,
):
    """Partially linear IV with two-way clustering (Chiang, Kato, Ma and Sasaki, 2021) on an
    ``N x M`` grid. Returns ``(x, y, d, cluster_vars, z)``."""
    _check_return_type(return_type)
    kwargs = {k: (list(np.asarray(v, dtype=float)) if np.ndim(v) else v) for k, v in kwargs.items()}
    r = _draw("pliv_multiway_cluster_CKMS2021", random_state, N=N, M=M, dim_X=dim_X, theta=theta, **kwargs)
    z = r["z"][:, 0]
    if return_type == "array":
        return r["x"], r["y"], r["d"], r["clusters"], z
    cols = {"cluster_var_i": r["clusters"][:, 0], "cluster_var_j": r["clusters"][:, 1]}
    cols.update({f"X{i + 1}": r["x"][:, i] for i in range(dim_X)})
    cols.update({"Y": r["y"], "D": r["d"], "Z": z})
    return _frame(cols)


def make_lplr_LZZ2020(
    n_obs: int = 500,
    dim_x: int = 20,
    alpha: float = 0.5,
    return_type: Literal["array", "DataFrame"] = "array",
    balanced_r0: bool = True,
    treatment: Literal["continuous", "binary", "binary_unbalanced"] = "continuous",
    random_state: Optional[int] = None,
):
    """Logistic partially linear model of Liu, Zhang and Zhou (2020): a binary outcome with
    ``P(y = 1) = expit(alpha d + r_0(x))``. Returns ``(x, y, d, p)`` with ``p`` the outcome
    probabilities."""
    _check_return_type(return_type)
    r = _draw("lplr_LZZ2020", random_state, n_obs=n_obs, dim_x=dim_x, alpha=alpha, balanced_r0=balanced_r0,
              treatment=treatment)
    if return_type == "array":
        return r["x"], r["y"], r["d"], r["p"]
    return _xyd_frame(r["x"], r["y"], r["d"], {"p": r["p"]})


def make_plpr_CP2025(
    num_id: int = 250,
    num_t: int = 10,
    dim_x: int = 30,
    theta: float = 0.5,
    dgp_type: Literal["dgp1", "dgp2", "dgp3"] = "dgp1",
    time_type: Literal["int", "float", "datetime"] = "int",
    random_state: Optional[int] = None,
):
    """Static panel of Clarke and Polselli (2025) with correlated unit effects, as a long-format
    DataFrame with ``id``, ``time``, ``y``, ``d`` and ``x1, x2, ...``."""
    if time_type not in ("int", "float", "datetime"):
        raise ValueError(f"time_type must be 'int', 'float' or 'datetime', got {time_type!r}")
    r = _draw("plpr_CP2025", random_state, num_id=num_id, num_t=num_t, dim_x=dim_x, theta=theta, dgp_type=dgp_type)
    pd = _pandas()
    time: Any = r["t"]
    if time_type == "int":
        time = time.astype(np.int64)
    elif time_type == "datetime":
        time = pd.to_datetime([f"2020-{int(m):02d}-01" for m in time])
    cols = {"id": r["id"], "time": time, "y": r["y"], "d": r["d"]}
    cols.update({f"x{i + 1}": r["x"][:, i] for i in range(dim_x)})
    return pd.DataFrame(cols)


def make_irm_data(
    n_obs: int = 500,
    dim_x: int = 20,
    theta: float = 0.0,
    R2_d: float = 0.5,
    R2_y: float = 0.5,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
):
    """Interactive regression with a binary treatment (Belloni et al., 2017), with the strength of
    the propensity and outcome models set by ``R2_d`` and ``R2_y``. Returns ``(x, y, d)``."""
    _check_return_type(return_type)
    r = _draw("irm_data", random_state, n_obs=n_obs, dim_x=dim_x, theta=theta, R2_d=R2_d, R2_y=R2_y)
    if return_type == "array":
        return r["x"], r["y"], r["d"]
    return _xyd_frame(r["x"], r["y"], r["d"])


def make_confounded_irm_data(
    n_obs: int = 500,
    theta: float = 0.0,
    gamma_a: float = 0.127,
    beta_a: float = 0.58,
    linear: bool = False,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Binary treatment with an unobserved confounder, with oracle confounding strengths for
    sensitivity analysis. ``kwargs``: ``dim_x``, ``trimming_threshold``, ``var_eps_y``. Returns
    ``{"x", "y", "d", "oracle_values"}``."""
    r = _draw("confounded_irm_data", random_state, n_obs=n_obs, theta=theta, gamma_a=gamma_a, beta_a=beta_a,
              linear=linear, **kwargs)
    return {"x": r["x"], "y": r["y"], "d": r["d"], "oracle_values": r["oracle_values"]}


def make_irm_data_discrete_treatments(
    n_obs: int = 200,
    n_levels: int = 3,
    linear: bool = False,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Multi-valued treatment ``d`` in ``{0, 1, ..., n_levels}``. ``kwargs``: ``xi``, ``c``,
    ``dim_x``. Returns ``{"x", "y", "d", "oracle_values"}``."""
    if not isinstance(n_levels, int):
        raise ValueError("n_levels must be an integer.")
    r = _draw("irm_data_discrete_treatments", random_state, n_obs=n_obs, n_levels=n_levels, linear=linear,
              **kwargs)
    return {"x": r["x"], "y": r["y"], "d": r["d"], "oracle_values": r["oracle_values"]}


def make_iivm_data(
    n_obs: int = 500,
    dim_x: int = 20,
    theta: float = 1.0,
    alpha_x: float = 0.2,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
):
    """Binary treatment shifted by a binary instrument (LATE). Returns ``(x, y, d, z)``."""
    _check_return_type(return_type)
    r = _draw("iivm_data", random_state, n_obs=n_obs, dim_x=dim_x, theta=theta, alpha_x=alpha_x)
    z = r["z"][:, 0]
    if return_type == "array":
        return r["x"], r["y"], r["d"], z
    return _xyd_frame(r["x"], r["y"], r["d"], {"z": z})


def make_ssm_data(
    n_obs: int = 8000,
    dim_x: int = 100,
    theta: float = 1.0,
    mar: bool = True,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
):
    """Sample selection: the outcome is observed where ``s = 1`` (0 elsewhere); missing at random
    (``mar``) or nonignorable with an instrument ``z`` for selection. Returns
    ``(x, y, d, z, s)``."""
    _check_return_type(return_type)
    r = _draw("ssm_data", random_state, n_obs=n_obs, dim_x=dim_x, theta=theta, mar=mar)
    z = r["z"][:, 0]
    if return_type == "array":
        return r["x"], r["y"], r["d"], z, r["s"]
    extra = {"s": r["s"]} if mar else {"z": z, "s": r["s"]}
    return _xyd_frame(r["x"], r["y"], r["d"], extra)


def make_heterogeneous_data(
    n_obs: int = 200,
    p: int = 30,
    support_size: int = 5,
    n_x: int = 1,
    binary_treatment: bool = False,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Heterogeneous effects ``exp(2 x_0) + 3 sin(4 x_0)`` (``n_x=2``: ``3 sin(4 x_1)``) on
    uniform covariates. Returns ``{"data", "effects", "treatment_effect"}``: the data as a
    DataFrame (columns ``y``, ``d``, ``X_0``, ...; a dict without pandas), the individual effects
    and the effect function."""
    if n_x not in (1, 2):
        raise ValueError("n_x must be either 1 or 2.")
    r = _draw("heterogeneous_data", random_state, n_obs=n_obs, p=p, support_size=support_size, n_x=n_x,
              binary_treatment=binary_treatment)
    k = 1 if n_x == 2 else 0

    def treatment_effect(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return np.exp(2 * x[:, 0]) + 3 * np.sin(4 * x[:, k])

    cols = {"y": r["y"], "d": r["d"], **{f"X_{i}": r["x"][:, i] for i in range(p)}}
    try:
        data: Any = _frame(cols)
    except ImportError:
        data = cols
    effect: Callable[[np.ndarray], np.ndarray] = treatment_effect
    return {"data": data, "effects": r["effects"], "treatment_effect": effect}


def make_did_SZ2020(
    n_obs: int = 500,
    dgp_type: int = 1,
    cross_sectional_data: bool = False,
    return_type: Literal["array", "DataFrame"] = "array",
    random_state: Optional[int] = None,
    **kwargs: float,
):
    """Two-period difference-in-differences of Sant'Anna and Zhao (2020), ``dgp_type`` 1 to 6.
    Panel data give the outcome change; ``cross_sectional_data=True`` repeated cross-sections
    with a time indicator. ``kwargs``: ``xi``, ``c``, ``lambda_t``. Returns ``(z, y, d, t)``
    (``t`` is None for panels)."""
    _check_return_type(return_type)
    r = _draw("did_SZ2020", random_state, n_obs=n_obs, dgp_type=dgp_type,
              cross_sectional_data=cross_sectional_data, **kwargs)
    t = r.get("t")
    if return_type == "array":
        return r["x"], r["y"], r["d"], t
    return _xyd_frame(r["x"], r["y"], r["d"], {} if t is None else {"t": t}, x_prefix="Z")


def _staggered_frame(r: dict[str, Any], time_type: str, start_date: str):
    pd = _pandas()
    t: Any = r["t"]
    d: Any = r["d"]
    if time_type == "datetime":
        start = np.datetime64(start_date, "M")
        t = start + t.astype(np.int64).astype("timedelta64[M]")
        finite = np.isfinite(d)
        dd = np.full(d.shape, np.datetime64("NaT"), dtype="datetime64[M]")
        dd[finite] = start + d[finite].astype(np.int64).astype("timedelta64[M]")
        t, d = pd.to_datetime(t), pd.to_datetime(dd)
    cols = {"id": r["id"], "y": r["y"], "y0": r["y0"], "y1": r["y1"], "d": d, "t": t}
    cols.update({f"Z{i + 1}": r["x"][:, i] for i in range(r["x"].shape[1])})
    return pd.DataFrame(cols)


def make_did_CS2021(
    n_obs: int = 1000,
    dgp_type: int = 1,
    include_never_treated: bool = True,
    time_type: Literal["float", "datetime"] = "float",
    random_state: Optional[int] = None,
    **kwargs: Any,
):
    """Staggered adoption of Callaway and Sant'Anna (2021): a long-format panel DataFrame with
    ``id``, ``y``, ``y0``, ``y1``, ``d`` (first treated period; ``inf`` or ``NaT`` never),
    ``t`` and ``Z1``–``Z4``. ``kwargs``: ``c``, ``xi``, ``n_periods``, ``anticipation_periods``,
    ``n_pre_treat_periods``, ``start_date`` (for ``time_type="datetime"``)."""
    if time_type not in ("float", "datetime"):
        raise ValueError(f"time_type must be one of ('datetime', 'float'). Got {time_type}.")
    start_date = kwargs.pop("start_date", "2025-01")
    r = _draw("did_CS2021", random_state, n_obs=n_obs, dgp_type=dgp_type,
              include_never_treated=include_never_treated, **kwargs)
    return _staggered_frame(r, time_type, start_date)


def make_did_cs_CS2021(
    n_obs: int = 1000,
    dgp_type: int = 1,
    include_never_treated: bool = True,
    lambda_t: float = 0.5,
    time_type: Literal["float", "datetime"] = "float",
    random_state: Optional[int] = None,
    **kwargs: Any,
):
    """Repeated cross-sections of the Callaway and Sant'Anna (2021) design: rows of a panel of
    ``ceil(n_obs / lambda_t)`` units, each kept with probability ``lambda_t``. Columns as
    `make_did_CS2021`."""
    if time_type not in ("float", "datetime"):
        raise ValueError(f"time_type must be one of ('datetime', 'float'). Got {time_type}.")
    start_date = kwargs.pop("start_date", "2025-01")
    r = _draw("did_cs_CS2021", random_state, n_obs=n_obs, dgp_type=dgp_type,
              include_never_treated=include_never_treated, lambda_t=lambda_t, **kwargs)
    return _staggered_frame(r, time_type, start_date)


def make_simple_rdd_data(
    n_obs: int = 5000,
    p: int = 4,
    fuzzy: bool = True,
    binary_outcome: bool = False,
    random_state: Optional[int] = None,
    **kwargs: float,
) -> dict[str, Any]:
    """Regression discontinuity with a normal score and polynomial covariate effects. ``kwargs``:
    ``cutoff``, ``dim_x``, ``a``, ``tau``. Returns ``{"score", "Y", "D", "X",
    "oracle_values": {"Y0", "Y1"}}``."""
    r = _draw("simple_rdd_data", random_state, n_obs=n_obs, p=p, fuzzy=fuzzy, binary_outcome=binary_outcome,
              **kwargs)
    return {"score": r["score"], "Y": r["y"], "D": r["d"].astype(np.int64), "X": r["x"],
            "oracle_values": {"Y0": r["y0"], "Y1": r["y1"]}}

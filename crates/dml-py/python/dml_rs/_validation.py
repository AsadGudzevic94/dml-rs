"""Input coercion shared by the estimators."""

import numpy as np


def _as_2d(a, name):
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {a.shape}")
    return a


def _as_1d(a, name):
    a = np.ascontiguousarray(a, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0].copy()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


def _as_clusters(cluster):
    """Cluster labels (any dtype, one or two columns) as int64 ids, shape ``(n, k)``."""
    if cluster is None:
        return None
    arr = np.asarray(cluster)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] not in (1, 2):
        raise ValueError(f"cluster must have one or two columns, got shape {arr.shape}")
    ids = [np.unique(arr[:, j], return_inverse=True)[1].reshape(-1) for j in range(arr.shape[1])]
    return np.ascontiguousarray(np.column_stack(ids), dtype=np.int64)


def _renamed_kwarg(owner, kwargs, old, new, value):
    """``value``, or the value passed under the former name ``old`` (with a DeprecationWarning)."""
    if old in kwargs:
        import warnings

        warnings.warn(
            f"{owner}: `{old}` was renamed to `{new}` in dml-rs 0.32.0; the old name will be removed in 2.0",
            DeprecationWarning,
            stacklevel=3,
        )
        value = kwargs.pop(old)
    if kwargs:
        raise TypeError(f"{owner}() got an unexpected keyword argument {next(iter(kwargs))!r}")
    return value

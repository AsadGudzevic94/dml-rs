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

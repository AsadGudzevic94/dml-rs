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

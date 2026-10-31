"""Unpickling of the native learners and results (see ``__reduce__`` in ``src/pickle.rs``)."""

from typing import Any

from . import _dml_rs


def restore(kind: str, data: bytes) -> Any:
    """Rebuild a pickled native object; the data must come from the same dml-rs version."""
    return _dml_rs._restore(kind, data)

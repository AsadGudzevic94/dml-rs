"""dml_rs — fast double / debiased machine learning, implemented in Rust.

>>> import numpy as np
>>> from sklearn.ensemble import RandomForestRegressor
>>> import dml_rs
>>> model = dml_rs.PLR(ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor())  # doctest: +SKIP
>>> model.fit(X, y, d).summary  # doctest: +SKIP

Nuisance learners can be the built-in Rust learners (:class:`Ols`,
:class:`Lasso`, :class:`LogisticRegression`), which run fully in parallel,
or any scikit-learn compatible estimator.
"""

from ._dml_rs import (
    Bootstrap,
    DmlResult,
    Lasso,
    LogisticRegression,
    Ols,
    __version__,
    fit_irm,
    fit_pliv,
    fit_plr,
)
from ._models import IRM, PLIV, PLR

__all__ = [
    "PLR",
    "IRM",
    "PLIV",
    "Ols",
    "Lasso",
    "LogisticRegression",
    "DmlResult",
    "Bootstrap",
    "fit_plr",
    "fit_irm",
    "fit_pliv",
    "__version__",
]

"""dml_rs — fast double / debiased machine learning, implemented in Rust.

>>> import numpy as np
>>> from sklearn.ensemble import RandomForestRegressor
>>> import dml_rs
>>> model = dml_rs.PLR(ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor())  # doctest: +SKIP
>>> model.fit(X, y, d).summary  # doctest: +SKIP

CATE meta-learners (:class:`SLearner`, :class:`TLearner`, :class:`XLearner`,
:class:`RLearner`, :class:`DRLearner`) estimate heterogeneous effects; the
DR-learner and :meth:`IRM.cate` / :meth:`IRM.gate` provide inference via
:class:`BLP`.

Nuisance learners can be the built-in Rust learners (:class:`Ols`,
:class:`Lasso`, :class:`LogisticRegression`, :class:`RandomForestRegressor`,
:class:`RandomForestClassifier`), which run fully in parallel without the GIL,
or any scikit-learn compatible estimator.
"""

from ._dml_rs import (
    Blp,
    Bootstrap,
    CateResult,
    DmlResult,
    Lasso,
    LogisticRegression,
    Ols,
    RandomForestClassifier,
    RandomForestRegressor,
    __version__,
    fit_cate,
    fit_did,
    fit_iivm,
    fit_irm,
    fit_pliv,
    fit_plr,
    fit_staggered_did,
)
from ._blp import BLP
from ._cate import DRLearner, RLearner, SLearner, TLearner, XLearner
from ._models import DID, DIDCS, IIVM, IRM, PLIV, PLR, DIDAggregation, StaggeredDID

__all__ = [
    "PLR",
    "IRM",
    "PLIV",
    "IIVM",
    "DID",
    "DIDCS",
    "StaggeredDID",
    "DIDAggregation",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "DRLearner",
    "BLP",
    "Ols",
    "Lasso",
    "LogisticRegression",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "DmlResult",
    "Bootstrap",
    "Blp",
    "CateResult",
    "fit_cate",
    "fit_iivm",
    "fit_did",
    "fit_staggered_did",
    "fit_plr",
    "fit_irm",
    "fit_pliv",
    "__version__",
]

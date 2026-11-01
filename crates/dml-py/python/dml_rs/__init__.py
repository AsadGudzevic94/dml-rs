"""dml_rs — fast double / debiased machine learning, implemented in Rust.

>>> import numpy as np
>>> from sklearn.ensemble import RandomForestRegressor
>>> import dml_rs
>>> model = dml_rs.PLR(ml_l=RandomForestRegressor(), ml_m=RandomForestRegressor())  # doctest: +SKIP
>>> model.fit(X, y, d).summary  # doctest: +SKIP

CATE meta-learners (`SLearner`, `TLearner`, `XLearner`,
`RLearner`, `DRLearner`) estimate heterogeneous effects; the
DR-learner and `IRM.cate` / `IRM.gate` provide inference via
`BLP`.

Nuisance learners can be the built-in Rust learners (`Ols`, `Lasso`,
`LassoCV`, `LogisticRegression`, `LogisticRegressionCV`, random forests,
gradient boosting, histogram gradient boosting, ...), which are
scikit-learn estimators but cross-fit fully in Rust, in parallel and
without the GIL, or any scikit-learn compatible estimator.
"""

from ._dml_rs import __version__
from . import native
from ._learners import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    GridSearchClassifier,
    GridSearchRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    Lasso,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
    Ols,
    QuantileForest,
    RandomForestClassifier,
    RandomForestRegressor,
    RegressionForest,
)
from . import datasets
from .data import DmlData
from ._blp import BLP
from ._cate import DRLearner, RLearner, SLearner, TLearner, XLearner
from ._policy import PolicyTree
from ._tuning import OptunaResult
from ._forest import CausalForest, InstrumentalForest, MultiArmCausalForest
from ._autodml import (
    ATE,
    AutoDML,
    AverageDerivative,
    CustomFunctional,
    Functional,
    PolicyEffect,
    RieszLasso,
    ShiftEffect,
)
from ._conformal import ConformalITE
from ._bounds import LeeBounds
from ._contdid import ContinuousDID
from ._dose import DoseResponse
from ._mediation import Mediation
from ._policy_search import OptimalPolicyTree, double_robust_scores, policy_value
from ._surrogate import SurrogateIndex
from ._synthdid import SyntheticDID
from ._models import (
    APO,
    APOS,
    CVAR,
    DID,
    DIDCS,
    IIVM,
    IRM,
    LPLR,
    LPQ,
    PLIV,
    PLPR,
    PLR,
    PQ,
    QTE,
    SSM,
    RDFlex,
    CausalContrast,
    DIDAggregation,
    Framework,
    concat,
    PotentialQuantiles,
    StaggeredDID,
)

__all__ = [
    "PLR",
    "IRM",
    "PLIV",
    "IIVM",
    "DID",
    "DIDCS",
    "StaggeredDID",
    "DIDAggregation",
    "APO",
    "APOS",
    "CausalContrast",
    "ConformalITE",
    "ContinuousDID",
    "LeeBounds",
    "DoseResponse",
    "Mediation",
    "OptimalPolicyTree",
    "double_robust_scores",
    "policy_value",
    "SurrogateIndex",
    "SyntheticDID",
    "Framework",
    "concat",
    "SSM",
    "AutoDML",
    "CausalForest",
    "InstrumentalForest",
    "MultiArmCausalForest",
    "ATE",
    "AverageDerivative",
    "ShiftEffect",
    "PolicyEffect",
    "CustomFunctional",
    "Functional",
    "RieszLasso",
    "LPLR",
    "PLPR",
    "RDFlex",
    "PQ",
    "LPQ",
    "CVAR",
    "QTE",
    "PotentialQuantiles",
    "PolicyTree",
    "SLearner",
    "TLearner",
    "XLearner",
    "RLearner",
    "DRLearner",
    "BLP",
    "OptunaResult",
    "Ols",
    "Lasso",
    "LassoCV",
    "LogisticRegression",
    "LogisticRegressionCV",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "RegressionForest",
    "QuantileForest",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "GradientBoostingClassifier",
    "GridSearchRegressor",
    "GridSearchClassifier",
    "datasets",
    "native",
    "DmlData",
    "__version__",
]


# Moved to dml_rs.native in 0.32.0; the top-level names still work (with a DeprecationWarning)
# until 2.0.
_MOVED_TO_NATIVE = frozenset(native.__all__)


def __getattr__(name):
    if name in _MOVED_TO_NATIVE:
        import warnings

        warnings.warn(
            f"dml_rs.{name} moved to dml_rs.native.{name} in dml-rs 0.32.0; "
            "the top-level name will be removed in 2.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(native, name)
    raise AttributeError(f"module 'dml_rs' has no attribute {name!r}")

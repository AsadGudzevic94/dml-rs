"""Type stubs for the native extension ``dml_rs._dml_rs`` (implemented in Rust).

Array arguments are converted to float64 NumPy arrays by the Python layer; the
estimator classes in ``dml_rs`` are the documented entry points.
"""

from typing import Any, Literal, Optional, Sequence, Union, final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

__version__: str
__all__ = [
    "fit_cate",
    "fit_did",
    "fit_iivm",
    "fit_irm",
    "fit_pliv",
    "fit_plr",
    "fit_staggered_did",
    "Blp",
    "Bootstrap",
    "CateResult",
    "DmlResult",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "Lasso",
    "LassoCV",
    "LogisticRegression",
    "LogisticRegressionCV",
    "Ols",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "SensitivityBenchmark",
    "SensitivityResult",
    "StaggeredResult",
    "__version__",
]

_F1 = NDArray[np.float64]  # 1-D
_F2 = NDArray[np.float64]  # 2-D
_F3 = NDArray[np.float64]  # 3-D
_Splits = Sequence[Sequence[tuple[Sequence[int], Sequence[int]]]]
_MaxFeatures = Union[None, Literal["sqrt", "log2"], float, int]

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@final
class DmlResult:
    """A fitted DML model."""

    @property
    def model_name(self) -> str: ...
    @property
    def treatment_names(self) -> list[str]: ...
    @property
    def n_obs(self) -> int: ...
    @property
    def n_rep(self) -> int: ...
    @property
    def n_folds(self) -> int: ...
    @property
    def n_clusters(self) -> Optional[int]: ...
    @property
    def coef(self) -> _F1: ...
    @property
    def se(self) -> _F1: ...
    @property
    def t_stat(self) -> _F1: ...
    @property
    def pval(self) -> _F1: ...
    @property
    def all_coef(self) -> _F2:
        """Shape ``(n_treat, n_rep)``."""
    @property
    def all_se(self) -> _F2:
        """Shape ``(n_treat, n_rep)``."""
    @property
    def psi(self) -> _F3:
        """Score values at the estimate, shape ``(n_obs, n_treat, n_rep)``."""
    @property
    def nuisance_names(self) -> list[str]: ...
    @property
    def sensitivity_elements(self) -> Optional[tuple[_F2, _F2]]:
        """``(sigma2, nu2)``, each of shape ``(n_treat, n_rep)``."""
    def predictions(self, name: str) -> _F3:
        """Cross-fitted predictions, shape ``(n_obs, n_rep, n_treat)``."""
    def nuisance_losses(self) -> dict[str, _F2]:
        """Out-of-sample loss per learner, each of shape ``(n_treat, n_rep)``."""
    def nuisance_targets(self, name: str) -> _F3:
        """Targets of a learner, shape ``(n_obs, n_rep, n_treat)``, NaN where it does not apply."""
    def confint(self, level: float = 0.95) -> _F2: ...
    def p_adjust(self, method: str) -> _F1: ...
    def bootstrap(
        self, method: Literal["normal", "Bayes", "bayes", "wild"] = "normal", n_boot: int = 500, seed: int = 0
    ) -> Bootstrap: ...
    def sensitivity_analysis(
        self,
        cf_y: float = 0.03,
        cf_d: float = 0.03,
        rho: float = 1.0,
        level: float = 0.95,
        null_hypothesis: Optional[Sequence[float]] = None,
    ) -> SensitivityResult: ...
    def sensitivity_benchmark(self, short: DmlResult) -> SensitivityBenchmark: ...
    def cate(self, basis: _F2) -> Blp: ...
    def gate(self, groups: _F2) -> Blp: ...

@final
class Bootstrap:
    @property
    def t_stats(self) -> _F3:
        """Shape ``(n_boot, n_treat, n_rep)``."""
    def joint_confint(self, level: float = 0.95) -> _F2: ...
    def romano_wolf(self) -> _F1: ...

@final
class SensitivityResult:
    @property
    def theta_lower(self) -> _F1: ...
    @property
    def theta_upper(self) -> _F1: ...
    @property
    def se_lower(self) -> _F1: ...
    @property
    def se_upper(self) -> _F1: ...
    @property
    def ci_lower(self) -> _F1: ...
    @property
    def ci_upper(self) -> _F1: ...
    @property
    def rv(self) -> _F1: ...
    @property
    def rva(self) -> _F1: ...
    @property
    def params(self) -> tuple[float, float, float, float, list[float]]:
        """``(cf_y, cf_d, rho, level, null_hypothesis)``."""

@final
class SensitivityBenchmark:
    @property
    def cf_y(self) -> _F1: ...
    @property
    def cf_d(self) -> _F1: ...
    @property
    def rho(self) -> _F1: ...
    @property
    def delta_theta(self) -> _F1: ...

@final
class Blp:
    """Best linear predictor of the CATE (or group effects) with HC0 inference."""

    @property
    def coef(self) -> _F1: ...
    @property
    def se(self) -> _F1: ...
    @property
    def t_stat(self) -> _F1: ...
    @property
    def pval(self) -> _F1: ...
    @property
    def is_gate(self) -> bool: ...
    def confint(self, level: float = 0.95) -> _F2: ...
    def predict(self, basis: _F2) -> tuple[_F1, _F1]:
        """``(effect, se)`` at each row of ``basis``."""
    def confint_at(self, basis: _F2, level: float = 0.95) -> _F2: ...
    def joint_confint_at(self, basis: _F2, level: float = 0.95, n_boot: int = 500, seed: int = 0) -> _F2: ...

@final
class CateResult:
    @property
    def method(self) -> str: ...
    @property
    def pseudo_outcomes(self) -> Optional[_F1]: ...
    def effect(self, x: _F2) -> _F1: ...
    def blp(self, basis: _F2) -> Blp: ...
    def gate(self, groups: _F2) -> Blp: ...

@final
class StaggeredResult:
    @property
    def att(self) -> DmlResult:
        """All group-time effects ATT(g, t) as one fit."""
    @property
    def group_times(self) -> list[tuple[float, float, float]]:
        """``(g, t_pre, t_eval)`` of each effect."""
    def aggregate(self, how: Literal["group", "time", "eventstudy"]) -> tuple[DmlResult, DmlResult, _F2, _F1]:
        """``(effects, overall, weights, overall_weights)``."""

# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

@final
class Ols:
    def __new__(cls, ridge: float = 0.0) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict(self, x: _F2) -> _F1: ...

@final
class Lasso:
    def __new__(cls, alpha: float = 1.0, tol: float = 1e-4, max_iter: int = 100000) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict(self, x: _F2) -> _F1: ...

@final
class LassoCV:
    def __new__(
        cls,
        alphas: Union[None, int, Sequence[float]] = None,
        eps: float = 1e-3,
        cv: int = 5,
        tol: float = 1e-4,
        max_iter: int = 100000,
    ) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict(self, x: _F2) -> _F1: ...
    @property
    def alpha_(self) -> float: ...
    @property
    def alphas_(self) -> _F1: ...
    @property
    def mse_path_(self) -> _F2:
        """Shape ``(n_alphas, cv)``."""
    @property
    def coef_(self) -> _F1: ...
    @property
    def intercept_(self) -> float: ...

@final
class LogisticRegression:
    def __new__(cls, C: float = 1.0, tol: float = 1e-10, max_iter: int = 100) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict_proba(self, x: _F2) -> _F1:
        """Probability of class 1 (1-D, unlike scikit-learn's two columns)."""

@final
class LogisticRegressionCV:
    def __new__(
        cls,
        Cs: Union[None, int, Sequence[float]] = None,
        cv: int = 5,
        scoring: Literal["neg_log_loss", "log_loss", "accuracy"] = "neg_log_loss",
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict_proba(self, x: _F2) -> _F1: ...
    @property
    def C_(self) -> float: ...
    @property
    def Cs_(self) -> _F1: ...
    @property
    def scores_(self) -> _F2:
        """Shape ``(cv, n_Cs)``; higher is better."""
    @property
    def coef_(self) -> _F1: ...
    @property
    def intercept_(self) -> float: ...

@final
class RandomForestRegressor:
    def __new__(
        cls,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: _MaxFeatures = None,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        random_state: int = 0,
    ) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict(self, x: _F2) -> _F1: ...
    @property
    def feature_importances_(self) -> Optional[_F1]: ...

@final
class RandomForestClassifier:
    def __new__(
        cls,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: _MaxFeatures = None,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        random_state: int = 0,
    ) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict_proba(self, x: _F2) -> _F1: ...
    @property
    def feature_importances_(self) -> Optional[_F1]: ...

@final
class GradientBoostingRegressor:
    def __new__(
        cls,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = ...,  # default 3
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: _MaxFeatures = None,
        subsample: float = 1.0,
        random_state: int = 0,
    ) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict(self, x: _F2) -> _F1: ...
    @property
    def feature_importances_(self) -> Optional[_F1]: ...

@final
class GradientBoostingClassifier:
    def __new__(
        cls,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = ...,  # default 3
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: _MaxFeatures = None,
        subsample: float = 1.0,
        random_state: int = 0,
    ) -> Self: ...
    def fit(self, x: _F2, y: _F1) -> Self: ...
    def predict_proba(self, x: _F2) -> _F1: ...
    @property
    def feature_importances_(self) -> Optional[_F1]: ...

# ---------------------------------------------------------------------------
# Fitting functions (called by the estimator classes)
# ---------------------------------------------------------------------------

def fit_plr(
    y: _F1,
    d: _F2,
    x: _F2,
    ml_l: Any,
    ml_m: Any,
    ml_g: Any = None,
    score: Literal["partialling out", "IV-type"] = "partialling out",
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
) -> DmlResult: ...
def fit_irm(
    y: _F1,
    d: _F2,
    x: _F2,
    ml_g: Any,
    ml_m: Any,
    score: Literal["ATE", "ATTE"] = "ATE",
    clipping_threshold: float = 1e-2,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
) -> DmlResult: ...
def fit_pliv(
    y: _F1,
    d: _F2,
    x: _F2,
    z: _F1,
    ml_l: Any,
    ml_m: Any,
    ml_r: Any,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
) -> DmlResult: ...
def fit_iivm(
    y: _F1,
    d: _F2,
    x: _F2,
    z: _F1,
    ml_g: Any,
    ml_m: Any,
    ml_r: Any,
    always_takers: bool = True,
    never_takers: bool = True,
    clipping_threshold: float = 1e-2,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
) -> DmlResult: ...
def fit_did(
    y: _F1,
    d: _F2,
    x: _F2,
    ml_g: Any,
    ml_m: Any = None,
    t: Optional[_F1] = None,
    score: Literal["observational", "experimental"] = "observational",
    in_sample_normalization: bool = True,
    clipping_threshold: float = 1e-2,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
) -> DmlResult: ...
def fit_staggered_did(
    y: _F1,
    id: _F1,
    t: _F1,
    g: _F1,
    x: _F2,
    ml_g: Any,
    ml_m: Any = None,
    control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",
    anticipation_periods: int = 0,
    gt_combinations: Literal["standard", "all"] = "standard",
    score: Literal["observational", "experimental"] = "observational",
    in_sample_normalization: bool = True,
    clipping_threshold: float = 1e-2,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    panel: bool = True,
) -> StaggeredResult: ...
def fit_cate(
    method: Literal["S", "T", "X", "R", "DR"],
    y: _F1,
    d: _F1,
    x: _F2,
    outcome: Any,
    effect: Any = None,
    propensity: Any = None,
    n_folds: int = 5,
    seed: int = 0,
    clipping_threshold: float = 1e-2,
    splits: Optional[_Splits] = None,
) -> CateResult: ...

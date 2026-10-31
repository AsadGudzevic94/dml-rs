"""Type stubs for the native extension ``dml_rs._dml_rs`` (implemented in Rust).

Array arguments are converted to float64 NumPy arrays by the Python layer; the
estimator classes in ``dml_rs`` are the documented entry points.
"""

from typing import Any, Callable, Literal, Optional, Sequence, Union, final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

__version__: str

# `cv` of the cross-validated learners: a number of folds or explicit (train, test) rows.
_Cv = Union[None, int, Sequence[tuple[Sequence[int], Sequence[int]]]]
__all__ = [
    "_restore",
    "concat",
    "draw_folds",
    "result_from_scores",
    "set_parallel_folds",
    "simulate",
    "fit_apo",
    "fit_apos",
    "fit_autodml",
    "fit_causal_forest",
    "fit_cate",
    "fit_did",
    "fit_iivm",
    "fit_instrumental_forest",
    "fit_irm",
    "fit_lplr",
    "fit_multi_arm_causal_forest",
    "fit_pliv",
    "fit_plpr",
    "fit_plr",
    "fit_qte",
    "fit_quantile",
    "fit_rdflex",
    "fit_ssm",
    "fit_staggered_did",
    "AposResult",
    "Blp",
    "Bootstrap",
    "CateResult",
    "CausalForestResult",
    "DmlResult",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "GridSearchClassifier",
    "GridSearchRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "InstrumentalForestResult",
    "Lasso",
    "LassoCV",
    "LogisticRegression",
    "LogisticRegressionCV",
    "MultiArmCausalForestResult",
    "Ols",
    "PolicyTree",
    "QteResult",
    "QuantileForest",
    "RdFlexResult",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "RegressionForest",
    "SensitivityBenchmark",
    "SensitivityResult",
    "StaggeredResult",
    "__version__",
]

_Splits = Sequence[Sequence[tuple[Sequence[int], Sequence[int]]]]
_MaxFeatures = Union[None, Literal["sqrt", "log2"], float, int]

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@final
class DmlResult:
    """A fitted DML model (returned by the `fit_*` functions)."""

    @property
    def model_name(self) -> str: ...
    def linear_combination(self, weights: NDArray[np.float64], names: Sequence[str]) -> DmlResult: ...
    def __add__(self, other: DmlResult, /) -> DmlResult: ...
    def __radd__(self, other: DmlResult, /) -> DmlResult: ...
    def __sub__(self, other: DmlResult, /) -> DmlResult: ...
    def __rsub__(self, other: DmlResult, /) -> DmlResult: ...
    def __mul__(self, c: float, /) -> DmlResult: ...
    def __rmul__(self, c: float, /) -> DmlResult: ...
    def __neg__(self) -> DmlResult: ...
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
    def coef(self) -> NDArray[np.float64]: ...
    @property
    def se(self) -> NDArray[np.float64]: ...
    @property
    def t_stat(self) -> NDArray[np.float64]: ...
    @property
    def pval(self) -> NDArray[np.float64]: ...
    @property
    def all_coef(self) -> NDArray[np.float64]:
        """Shape ``(n_treat, n_rep)``."""
    @property
    def all_se(self) -> NDArray[np.float64]:
        """Shape ``(n_treat, n_rep)``."""
    @property
    def psi(self) -> NDArray[np.float64]:
        """Score values at the estimate, shape ``(n_obs, n_treat, n_rep)``."""
    @property
    def nuisance_names(self) -> list[str]: ...
    def sensitivity_grid(
        self,
        cf_y: Sequence[float],
        cf_d: Sequence[float],
        rho: float,
        level: float,
        null_hypothesis: Sequence[float],
        treatment: int,
        value: Literal["theta", "ci"],
        bound: Literal["lower", "upper"],
    ) -> NDArray[np.float64]:
        """Bounds over a grid, entry ``[i, j]`` at ``cf_d[i]``, ``cf_y[j]``."""
    @property
    def sensitivity_elements(self) -> Optional[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """``(sigma2, nu2)``, each of shape ``(n_treat, n_rep)``."""
    def predictions(self, name: str) -> NDArray[np.float64]:
        """Cross-fitted predictions, shape ``(n_obs, n_rep, n_treat)``."""
    def nuisance_losses(self) -> dict[str, NDArray[np.float64]]:
        """Out-of-sample loss per learner, each of shape ``(n_treat, n_rep)``."""
    def nuisance_targets(self, name: str) -> NDArray[np.float64]:
        """Targets of a learner, shape ``(n_obs, n_rep, n_treat)``, NaN where it does not apply."""
    def confint(self, level: float = 0.95) -> NDArray[np.float64]: ...
    def p_adjust(self, method: str) -> NDArray[np.float64]: ...
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
    def cate(self, basis: NDArray[np.float64]) -> Blp: ...
    def gate(self, groups: NDArray[np.float64]) -> Blp: ...
    def policy_tree(
        self, features: NDArray[np.float64], depth: int = 2, min_samples_leaf: int = 8, ccp_alpha: float = 0.01
    ) -> PolicyTree: ...

@final
class Bootstrap:
    """Multiplier bootstrap draws of the t-statistics."""

    @property
    def t_stats(self) -> NDArray[np.float64]:
        """Shape ``(n_boot, n_treat, n_rep)``."""
    def joint_confint(self, level: float = 0.95) -> NDArray[np.float64]: ...
    def romano_wolf(self) -> NDArray[np.float64]: ...

@final
class SensitivityResult:
    """Bounds, confidence bounds and robustness values of a sensitivity analysis."""

    @property
    def theta_lower(self) -> NDArray[np.float64]: ...
    @property
    def theta_upper(self) -> NDArray[np.float64]: ...
    @property
    def se_lower(self) -> NDArray[np.float64]: ...
    @property
    def se_upper(self) -> NDArray[np.float64]: ...
    @property
    def ci_lower(self) -> NDArray[np.float64]: ...
    @property
    def ci_upper(self) -> NDArray[np.float64]: ...
    @property
    def rv(self) -> NDArray[np.float64]: ...
    @property
    def rva(self) -> NDArray[np.float64]: ...
    @property
    def params(self) -> tuple[float, float, float, float, list[float]]:
        """``(cf_y, cf_d, rho, level, null_hypothesis)``."""

@final
class SensitivityBenchmark:
    """Confounding strength represented by observed covariates."""

    @property
    def cf_y(self) -> NDArray[np.float64]: ...
    @property
    def cf_d(self) -> NDArray[np.float64]: ...
    @property
    def rho(self) -> NDArray[np.float64]: ...
    @property
    def delta_theta(self) -> NDArray[np.float64]: ...

@final
class Blp:
    """Best linear predictor of the CATE (or group effects) with HC0 inference."""

    @property
    def coef(self) -> NDArray[np.float64]: ...
    @property
    def se(self) -> NDArray[np.float64]: ...
    @property
    def t_stat(self) -> NDArray[np.float64]: ...
    @property
    def pval(self) -> NDArray[np.float64]: ...
    @property
    def is_gate(self) -> bool: ...
    def confint(self, level: float = 0.95) -> NDArray[np.float64]: ...
    def predict(self, basis: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(effect, se)`` at each row of ``basis``."""
    def confint_at(self, basis: NDArray[np.float64], level: float = 0.95) -> NDArray[np.float64]: ...
    def joint_confint_at(self, basis: NDArray[np.float64], level: float = 0.95, n_boot: int = 500, seed: int = 0) -> NDArray[np.float64]: ...

@final
class CateResult:
    """A fitted CATE meta-learner."""

    @property
    def method(self) -> str: ...
    @property
    def pseudo_outcomes(self) -> Optional[NDArray[np.float64]]: ...
    def effect(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def blp(self, basis: NDArray[np.float64]) -> Blp: ...
    def gate(self, groups: NDArray[np.float64]) -> Blp: ...

@final
class PolicyTree:
    """A fitted policy tree: treatment recommendations from the estimated effects."""

    def predict(self, features: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def nodes(self) -> list[tuple[int, float, int, int, float]]: ...
    def export_text(self, feature_names: Optional[list[str]] = None) -> str: ...
    @property
    def n_leaves(self) -> int: ...
    @property
    def depth(self) -> int: ...

@final
class AposResult:
    """Average potential outcomes of several treatment levels, fitted on shared sample splits."""

    @property
    def treatment_levels(self) -> list[float]: ...
    @property
    def levels(self) -> DmlResult:
        """One estimate per treatment level."""
    def causal_contrast(self, reference_levels: Sequence[float]) -> DmlResult:
        """Effects of each level against each reference level (``"d vs d_ref"``)."""

@final
class QteResult:
    """Treatment effects on quantiles or CVaRs at several quantiles."""

    @property
    def quantiles(self) -> list[float]: ...
    @property
    def effects(self) -> DmlResult:
        """The effect at every quantile, with joint inference."""
    @property
    def potential(self) -> DmlResult:
        """Potential quantiles of treatment 0 and 1, ordered ``q1 (d=0), q1 (d=1), q2 (d=0), ...``."""

@final
class RdFlexResult:
    """Conventional, bias-corrected and robust RD estimates of an RDFlex fit."""

    @property
    def coef(self) -> NDArray[np.float64]: ...
    @property
    def se(self) -> NDArray[np.float64]: ...
    @property
    def t_stat(self) -> NDArray[np.float64]: ...
    @property
    def pval(self) -> NDArray[np.float64]: ...
    @property
    def ci(self) -> NDArray[np.float64]:
        """Median of the repetitions' 95% intervals, shape ``(3, 2)``."""
    @property
    def all_coef(self) -> NDArray[np.float64]:
        """Shape ``(3, n_rep)``."""
    @property
    def all_se(self) -> NDArray[np.float64]: ...
    @property
    def h(self) -> NDArray[np.float64]:
        """Final RD bandwidth of each repetition."""
    @property
    def b(self) -> NDArray[np.float64]:
        """Bias bandwidth of each repetition."""
    @property
    def h_fs(self) -> float: ...
    def confint(self, level: float = 0.95) -> NDArray[np.float64]: ...

@final
class CausalForestResult:
    """A fitted causal forest (returned by `fit_causal_forest`)."""

    def effect(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @property
    def oob_effect(self) -> NDArray[np.float64]: ...
    @property
    def oob_variance(self) -> NDArray[np.float64]: ...
    @property
    def y_hat(self) -> NDArray[np.float64]: ...
    @property
    def d_hat(self) -> NDArray[np.float64]: ...
    @property
    def variable_importance(self) -> NDArray[np.float64]: ...
    @property
    def n_trees(self) -> int: ...
    @property
    def binary_treatment(self) -> bool: ...
    def dr_scores(self) -> NDArray[np.float64]: ...
    def ate(self) -> DmlResult: ...
    def blp(self, basis: NDArray[np.float64]) -> Blp: ...
    def gate(self, groups: NDArray[np.float64]) -> Blp: ...
    def policy_tree(self, features: NDArray[np.float64], depth: int, min_samples_leaf: int, ccp_alpha: float) -> PolicyTree: ...

@final
class InstrumentalForestResult:
    """A fitted instrumental forest (returned by `fit_instrumental_forest`)."""

    def effect(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @property
    def oob_effect(self) -> NDArray[np.float64]: ...
    @property
    def oob_variance(self) -> NDArray[np.float64]: ...
    @property
    def y_hat(self) -> NDArray[np.float64]: ...
    @property
    def d_hat(self) -> NDArray[np.float64]: ...
    @property
    def z_hat(self) -> NDArray[np.float64]: ...
    @property
    def compliance(self) -> Optional[NDArray[np.float64]]: ...
    @property
    def variable_importance(self) -> NDArray[np.float64]: ...
    @property
    def n_trees(self) -> int: ...
    def dr_scores(self) -> NDArray[np.float64]: ...
    def ate(self) -> DmlResult: ...
    def blp(self, basis: NDArray[np.float64]) -> Blp: ...
    def gate(self, groups: NDArray[np.float64]) -> Blp: ...

@final
class MultiArmCausalForestResult:
    """A fitted multi-arm causal forest (returned by `fit_multi_arm_causal_forest`)."""

    def effect(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @property
    def oob_effect(self) -> NDArray[np.float64]: ...
    @property
    def oob_variance(self) -> NDArray[np.float64]: ...
    @property
    def y_hat(self) -> NDArray[np.float64]: ...
    @property
    def propensity(self) -> NDArray[np.float64]: ...
    @property
    def levels(self) -> list[float]: ...
    @property
    def arm_names(self) -> list[str]: ...
    @property
    def variable_importance(self) -> NDArray[np.float64]: ...
    @property
    def n_trees(self) -> int: ...
    def dr_scores(self) -> NDArray[np.float64]: ...
    def ate(self) -> DmlResult: ...
    def blp(self, arm: int, basis: NDArray[np.float64]) -> Blp: ...
    def gate(self, arm: int, groups: NDArray[np.float64]) -> Blp: ...

@final
class StaggeredResult:
    """Group-time effects of a staggered difference-in-differences fit."""

    @property
    def att(self) -> DmlResult:
        """All group-time effects ATT(g, t) as one fit."""
    @property
    def group_times(self) -> list[tuple[float, float, float]]:
        """``(g, t_pre, t_eval)`` of each effect."""
    def aggregate(self, how: Literal["group", "time", "eventstudy"]) -> tuple[DmlResult, DmlResult, NDArray[np.float64], NDArray[np.float64]]:
        """``(effects, overall, weights, overall_weights)``."""

# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------

@final
class GridSearchRegressor:
    """Grid search over regression learners: the candidate with the smallest cross-validated mean
    squared error (``cv``-fold, no shuffling, or explicit folds) is refitted on all data. As a
    nuisance learner the search runs inside every cross-fitting fold."""

    def __new__(cls, candidates: Sequence[Any], cv: _Cv = None) -> GridSearchRegressor: ...
    def __copy__(self) -> GridSearchRegressor: ...
    def __deepcopy__(self, memo: Any, /) -> GridSearchRegressor: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> GridSearchRegressor: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def best_index_(self) -> int: ...
    @property
    def cv_scores_(self) -> list[float]: ...

@final
class GridSearchClassifier:
    """Grid search over classifiers by cross-validated log loss (stratified folds)."""

    def __new__(cls, candidates: Sequence[Any], cv: _Cv = None) -> GridSearchClassifier: ...
    def __copy__(self) -> GridSearchClassifier: ...
    def __deepcopy__(self, memo: Any, /) -> GridSearchClassifier: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> GridSearchClassifier: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def best_index_(self) -> int: ...
    @property
    def cv_scores_(self) -> list[float]: ...

@final
class Ols:
    """Ordinary least squares with intercept; `ridge=α` adds an L2 penalty as scikit-learn's `Ridge(alpha=α)`."""

    def __new__(cls, ridge: float = 0.0) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def coef_(self) -> NDArray[np.float64]: ...
    @property
    def intercept_(self) -> float: ...

@final
class Lasso:
    """Lasso by coordinate descent: the objective, stopping rule (`tol`) and defaults of scikit-learn's `Lasso`."""

    def __new__(cls, alpha: float = 1.0, tol: float = 1e-4, max_iter: int = 100000) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def coef_(self) -> NDArray[np.float64]: ...
    @property
    def intercept_(self) -> float: ...
    @property
    def n_iter_(self) -> int: ...

@final
class LassoCV:
    """Lasso with the penalty chosen by cross-validation, as scikit-learn's `LassoCV`.

    `alphas` is the number of penalties on the automatic grid (from `alpha_max` down to
    `eps * alpha_max`) or a list of penalties; `cv` is the number of contiguous folds or a list of
    `(train, test)` rows."""

    def __new__(
        cls,
        alphas: Union[None, int, Sequence[float]] = None,
        eps: float = 1e-3,
        cv: _Cv = None,
        tol: float = 1e-4,
        max_iter: int = 100000,
    ) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def alpha_(self) -> float: ...
    @property
    def alphas_(self) -> NDArray[np.float64]: ...
    @property
    def mse_path_(self) -> NDArray[np.float64]:
        """Shape ``(n_alphas, n_folds)``."""
    @property
    def coef_(self) -> NDArray[np.float64]: ...
    @property
    def intercept_(self) -> float: ...
    @property
    def n_iter_(self) -> int: ...

@final
class LogisticRegression:
    """L2-penalised logistic regression (Newton's method); scikit-learn's `LogisticRegression(C=C)`."""

    def __new__(cls, C: float = 1.0, tol: float = 1e-10, max_iter: int = 100) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability of class 1 (1-D; `dml_rs.LogisticRegression` returns two columns)."""
    @property
    def coef_(self) -> NDArray[np.float64]: ...
    @property
    def intercept_(self) -> float: ...
    @property
    def n_iter_(self) -> int: ...

@final
class LogisticRegressionCV:
    """Logistic regression with `C` chosen by stratified cross-validation, as scikit-learn's
    `LogisticRegressionCV`.

    `Cs` is the number of values between 1e-4 and 1e4 or a list of values; `scoring` is
    `"neg_log_loss"` (default) or `"accuracy"`."""

    def __new__(
        cls,
        Cs: Union[None, int, Sequence[float]] = None,
        cv: _Cv = None,
        scoring: Literal["neg_log_loss", "log_loss", "accuracy"] = "neg_log_loss",
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def C_(self) -> float: ...
    @property
    def Cs_(self) -> NDArray[np.float64]: ...
    @property
    def scores_(self) -> NDArray[np.float64]:
        """Shape ``(n_folds, n_Cs)``; higher is better."""
    @property
    def coef_(self) -> NDArray[np.float64]: ...
    @property
    def intercept_(self) -> float: ...
    @property
    def n_iter_(self) -> int: ...

@final
class RandomForestRegressor:
    """Random forest regressor with scikit-learn's parameters, fitted in parallel."""

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
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def feature_importances_(self) -> Optional[NDArray[np.float64]]: ...

@final
class RegressionForest:
    """Honest regression forest (grf's `regression_forest`), implemented in Rust."""

    def __new__(
        cls,
        n_estimators: int = 2000,
        sample_fraction: float = 0.5,
        mtry: Optional[int] = None,
        min_node_size: int = 5,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        honesty_prune_leaves: bool = True,
        alpha: float = 0.05,
        imbalance_penalty: float = 0.0,
        ci_group_size: int = 2,
        random_state: int = 0,
    ) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def predict_with_variance(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def predict_oob(self, x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @property
    def variable_importance_(self) -> Optional[NDArray[np.float64]]: ...

@final
class QuantileForest:
    """Quantile forest (grf's `quantile_forest`), implemented in Rust."""

    def __new__(
        cls,
        quantiles: Sequence[float] = ...,  # default [0.1, 0.5, 0.9]
        n_estimators: int = 2000,
        sample_fraction: float = 0.5,
        mtry: Optional[int] = None,
        min_node_size: int = 5,
        honesty: bool = True,
        honesty_fraction: float = 0.5,
        honesty_prune_leaves: bool = True,
        alpha: float = 0.05,
        imbalance_penalty: float = 0.0,
        random_state: int = 0,
    ) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> Self: ...
    def predict(self, x: NDArray[np.float64], quantiles: Optional[Sequence[float]] = None) -> NDArray[np.float64]: ...
    def predict_oob(self, x: NDArray[np.float64], quantiles: Optional[Sequence[float]] = None) -> NDArray[np.float64]: ...
    @property
    def quantiles(self) -> list[float]: ...
    @property
    def variable_importance_(self) -> Optional[NDArray[np.float64]]: ...

@final
class RandomForestClassifier:
    """Random forest classifier with scikit-learn's parameters, fitted in parallel."""

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
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def feature_importances_(self) -> Optional[NDArray[np.float64]]: ...

@final
class GradientBoostingRegressor:
    """Gradient boosting (squared error) that grows the same trees as scikit-learn's
    `GradientBoostingRegressor`. `max_depth` defaults to 3."""

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
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def feature_importances_(self) -> Optional[NDArray[np.float64]]: ...

@final
class GradientBoostingClassifier:
    """Gradient boosting (log loss) that grows the same trees as scikit-learn's
    `GradientBoostingClassifier`. `max_depth` defaults to 3."""

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
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None) -> Self: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def feature_importances_(self) -> Optional[NDArray[np.float64]]: ...

@final
class HistGradientBoostingRegressor:
    """Histogram-based gradient boosting (squared error); matches scikit-learn's
    `HistGradientBoostingRegressor` with the same `random_state`. `early_stopping=None` means "auto"."""

    def __new__(
        cls,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = ...,  # default 31
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_features: float = 1.0,
        max_bins: int = 255,
        early_stopping: Union[Literal["auto"], bool, None] = None,
        validation_fraction: Optional[float] = ...,  # default 0.1
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        random_state: int = 0,
    ) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None
    ) -> Self: ...
    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def n_iter_(self) -> int: ...
    @property
    def validation_score_(self) -> NDArray[np.float64]: ...

@final
class HistGradientBoostingClassifier:
    """Histogram-based gradient boosting (log loss, 0/1 labels); matches scikit-learn's
    `HistGradientBoostingClassifier`. `predict_proba` returns P(Y = 1) only."""

    def __new__(
        cls,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = ...,  # default 31
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_features: float = 1.0,
        max_bins: int = 255,
        early_stopping: Union[Literal["auto"], bool, None] = None,
        validation_fraction: Optional[float] = ...,  # default 0.1
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        random_state: int = 0,
    ) -> Self: ...
    def __copy__(self) -> Self: ...
    def __deepcopy__(self, memo: Any, /) -> Self: ...
    def fit(
        self, x: NDArray[np.float64], y: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None
    ) -> Self: ...
    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...
    @property
    def n_iter_(self) -> int: ...
    @property
    def validation_score_(self) -> NDArray[np.float64]: ...

def _restore(kind: str, data: bytes) -> Any:
    """Rebuild a pickled native object (used by ``dml_rs._pickle.restore``)."""

# ---------------------------------------------------------------------------
# Fitting functions (called by the estimator classes)
# ---------------------------------------------------------------------------

def fit_plr(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
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
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
) -> DmlResult: ...
def fit_irm(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any,
    score: Literal["ATE", "ATTE"] = "ATE",
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    weights_bar: Optional[NDArray[np.float64]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
    nuisance: Optional[dict[str, Any]] = None,
) -> DmlResult: ...
def fit_pliv(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    ml_l: Any,
    ml_m: Any,
    ml_r: Any,
    ml_g: Any = None,
    score: Literal["partialling out", "IV-type"] = "partialling out",
    partial: Literal["X", "Z", "XZ"] = "X",
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
) -> DmlResult: ...
def fit_iivm(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any,
    ml_r: Any,
    always_takers: bool = True,
    never_takers: bool = True,
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
    nuisance: Optional[dict[str, Any]] = None,
) -> DmlResult: ...
def fit_apo(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any,
    treatment_level: float,
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    weights_bar: Optional[NDArray[np.float64]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
    nuisance: Optional[dict[str, Any]] = None,
) -> DmlResult: ...
def fit_apos(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any,
    treatment_levels: Sequence[float],
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    weights: Optional[NDArray[np.float64]] = None,
    weights_bar: Optional[NDArray[np.float64]] = None,
) -> AposResult: ...
def fit_plpr(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    id: NDArray[np.int64],
    t: NDArray[np.float64],
    ml_l: Any,
    ml_m: Any,
    m_classifier: bool = False,
    ml_g: Any = None,
    score: Literal["partialling out", "IV-type"] = "partialling out",
    approach: Literal["cre_general", "cre_normal", "fd_exact", "wg_approx"] = "fd_exact",
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
) -> DmlResult: ...
def fit_lplr(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_M: Any,
    ml_t: Any,
    ml_m: Any,
    m_classifier: bool,
    ml_a: Any = None,
    a_classifier: Optional[bool] = None,
    score: Literal["nuisance_space", "instrument"] = "nuisance_space",
    n_folds: int = 5,
    n_folds_inner: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    inner_splits: Optional[Sequence[_Splits]] = None,
    treatment_names: Optional[list[str]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
) -> DmlResult: ...
def fit_rdflex(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    score: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any = None,
    cutoff: float = 0.0,
    h_fs: Optional[float] = None,
    fs_specification: Literal["cutoff", "cutoff and score", "interacted cutoff and score"] = "cutoff",
    fs_kernel: Literal["triangular", "uniform", "epanechnikov"] = "triangular",
    n_iterations: int = 2,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    g_classifier: bool = False,
    rd_options: Optional[dict[str, Any]] = None,
    rd_covs: Optional[NDArray[np.float64]] = None,
    rd_cluster: Optional[NDArray[np.int64]] = None,
) -> RdFlexResult: ...
def fit_ssm(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    s: NDArray[np.float64],
    ml_g: Any,
    ml_pi: Any,
    ml_m: Any,
    z: Optional[NDArray[np.float64]] = None,
    score: Literal["missing-at-random", "nonignorable"] = "missing-at-random",
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = False,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    nuisance: Optional[dict[str, Any]] = None,
) -> DmlResult: ...
def fit_quantile(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any,
    score: Literal["PQ", "LPQ", "CVaR"] = "PQ",
    treatment: int = 1,
    quantile: float = 0.5,
    z: Optional[NDArray[np.float64]] = None,
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = True,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
) -> DmlResult: ...
def concat(results: Sequence[DmlResult]) -> DmlResult: ...
def simulate(name: str, params: dict[str, Any], seed: int) -> dict[str, Any]: ...
def set_parallel_folds(on: bool) -> None: ...
def fit_causal_forest(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_y: Any,
    ml_d: Any,
    num_trees: int = 2000,
    sample_fraction: float = 0.5,
    mtry: Optional[int] = None,
    min_node_size: int = 5,
    honesty: bool = True,
    honesty_fraction: float = 0.5,
    prune_empty_leaves: bool = True,
    alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    stabilize_splits: bool = True,
    ci_group_size: int = 2,
    forest_seed: int = 0,
    n_folds: int = 5,
    seed: int = 0,
    clipping_threshold: float = 1e-2,
    treatment_names: Optional[list[str]] = None,
) -> CausalForestResult: ...
def fit_instrumental_forest(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    z: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_y: Any,
    ml_d: Any,
    ml_z: Any,
    num_trees: int = 2000,
    sample_fraction: float = 0.5,
    mtry: Optional[int] = None,
    min_node_size: int = 5,
    honesty: bool = True,
    honesty_fraction: float = 0.5,
    prune_empty_leaves: bool = True,
    alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    stabilize_splits: bool = True,
    ci_group_size: int = 2,
    reduced_form_weight: float = 0.0,
    forest_seed: int = 0,
    n_folds: int = 5,
    seed: int = 0,
    clipping_threshold: float = 1e-2,
    treatment_names: Optional[list[str]] = None,
) -> InstrumentalForestResult: ...
def fit_multi_arm_causal_forest(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_y: Any,
    ml_d: Any,
    num_trees: int = 2000,
    sample_fraction: float = 0.5,
    mtry: Optional[int] = None,
    min_node_size: int = 5,
    honesty: bool = True,
    honesty_fraction: float = 0.5,
    prune_empty_leaves: bool = True,
    alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    stabilize_splits: bool = True,
    ci_group_size: int = 2,
    forest_seed: int = 0,
    n_folds: int = 5,
    seed: int = 0,
    clipping_threshold: float = 1e-2,
    treatment_name: Optional[str] = None,
) -> MultiArmCausalForestResult: ...
def fit_autodml(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    functional: Literal["ate", "average_derivative", "shift", "policy", "custom"] = "ate",
    step: Optional[float] = None,
    delta: Optional[float] = None,
    per_unit: bool = False,
    policy: Optional[NDArray[np.float64]] = None,
    terms: Optional[Callable[[NDArray[np.float64]], Any]] = None,
    functional_name: Optional[str] = None,
    degree: int = 2,
    treatment_degree: int = 1,
    penalty: Optional[float] = None,
    n_alphas: int = 30,
    eps: float = 1e-3,
    cv: int = 5,
    dictionary: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
) -> DmlResult: ...
def result_from_scores(
    psi_a: NDArray[np.float64],
    psi_b: NDArray[np.float64],
    treatment_names: list[str],
    n_folds: int,
    model: str = "custom",
) -> DmlResult: ...
def draw_folds(
    n: int,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    strata: Optional[Sequence[int]] = None,
) -> list[list[tuple[NDArray[np.int64], NDArray[np.int64]]]]: ...
def fit_qte(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any,
    quantiles: Sequence[float],
    score: Literal["PQ", "LPQ", "CVaR"] = "PQ",
    z: Optional[NDArray[np.float64]] = None,
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    normalize_ipw: bool = True,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
) -> QteResult: ...
def fit_did(
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any = None,
    t: Optional[NDArray[np.float64]] = None,
    score: Literal["observational", "experimental"] = "observational",
    in_sample_normalization: bool = True,
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    splits: Optional[_Splits] = None,
    treatment_names: Optional[list[str]] = None,
    cluster: Optional[NDArray[np.int64]] = None,
    external_predictions: Optional[Sequence[tuple[int, str, NDArray[np.float64]]]] = None,
    nuisance: Optional[dict[str, Any]] = None,
) -> DmlResult: ...
def fit_staggered_did(
    y: NDArray[np.float64],
    id: NDArray[np.float64],
    t: NDArray[np.float64],
    g: NDArray[np.float64],
    x: NDArray[np.float64],
    ml_g: Any,
    ml_m: Any = None,
    control_group: Literal["never_treated", "not_yet_treated"] = "never_treated",
    anticipation_periods: int = 0,
    gt_combinations: Literal["standard", "all"] = "standard",
    score: Literal["observational", "experimental"] = "observational",
    in_sample_normalization: bool = True,
    clipping_threshold: float = 1e-2,
    calibration: Optional[Literal["isotonic", "isotonic_cv"]] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    seed: int = 0,
    panel: bool = True,
    cluster: Optional[NDArray[np.int64]] = None,
) -> StaggeredResult: ...
def fit_cate(
    method: Literal["S", "T", "X", "R", "DR"],
    y: NDArray[np.float64],
    d: NDArray[np.float64],
    x: NDArray[np.float64],
    outcome: Any,
    effect: Any = None,
    propensity: Any = None,
    n_folds: int = 5,
    seed: int = 0,
    clipping_threshold: float = 1e-2,
    splits: Optional[_Splits] = None,
) -> CateResult: ...

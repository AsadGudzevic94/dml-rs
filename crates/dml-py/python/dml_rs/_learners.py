"""The built-in Rust learners as scikit-learn estimators.

Every class here holds its parameters as plain attributes (scikit-learn's convention), so
``get_params`` / ``set_params``, ``sklearn.base.clone``, ``GridSearchCV``, ``Pipeline`` and
``cross_val_score`` work, and so do ``tune``, ``tune_ml_models`` and ``set_ml_nuisance_params``
of the DML models. ``fit`` builds the native learner (``dml_rs._dml_rs``) from the parameters.

Inside a DML model nothing goes through Python: the model asks the estimator for its native
learner (``_dml_native``) and cross-fits it in Rust, in parallel and without the GIL.

scikit-learn is optional. When it is installed the classes derive from its ``BaseEstimator``
and mixins; without it a small stand-in provides ``get_params``, ``set_params`` and ``score``.
"""

from __future__ import annotations

import inspect
import numbers
import warnings
from typing import Any, ClassVar, Optional, Sequence, Union

import numpy as np

from . import _dml_rs

try:
    from sklearn.base import (  # type: ignore[import-untyped]
        BaseEstimator,
        ClassifierMixin,
        RegressorMixin,
    )
    from sklearn.exceptions import (  # type: ignore[import-untyped]
        DataConversionWarning,
        NotFittedError,
    )
    from sklearn.utils.multiclass import type_of_target  # type: ignore[import-untyped]
except ImportError:  # scikit-learn is optional

    class NotFittedError(ValueError, AttributeError):  # type: ignore[no-redef]
        """The estimator is used before ``fit``."""

    class DataConversionWarning(UserWarning):  # type: ignore[no-redef]
        """The input was converted (a column vector ``y`` was flattened)."""

    type_of_target = None

    class BaseEstimator:  # type: ignore[no-redef]
        """``get_params`` / ``set_params`` as scikit-learn's ``BaseEstimator``."""

        @classmethod
        def _get_param_names(cls) -> list[str]:
            params = inspect.signature(cls.__init__).parameters
            return sorted(p for p in params if p != "self")

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key in self._get_param_names():
                value = getattr(self, key)
                if deep and hasattr(value, "get_params") and not isinstance(value, type):
                    out.update((f"{key}__{k}", v) for k, v in value.get_params().items())
                out[key] = value
            return out

        def set_params(self, **params: Any) -> Any:
            valid = self._get_param_names()
            nested: dict[str, dict[str, Any]] = {}
            for full_key, value in params.items():
                key, sep, sub = full_key.partition("__")
                if key not in valid:
                    raise ValueError(
                        f"Invalid parameter {key!r} for estimator {self}. Valid parameters are: {valid!r}."
                    )
                if sep:
                    nested.setdefault(key, {})[sub] = value
                else:
                    setattr(self, key, value)
            for key, sub_params in nested.items():
                getattr(self, key).set_params(**sub_params)
            return self

        def __repr__(self) -> str:
            defaults = inspect.signature(type(self).__init__).parameters
            changed = [
                f"{k}={v!r}"
                for k, v in self.get_params(deep=False).items()
                if not _same_param(v, defaults[k].default)
            ]
            return f"{type(self).__name__}({', '.join(changed)})"

    class RegressorMixin:  # type: ignore[no-redef]
        _estimator_type = "regressor"

        def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
            """Coefficient of determination R² of ``predict(X)``."""
            y = np.asarray(y, dtype=np.float64)
            pred = self.predict(X)  # type: ignore[attr-defined]
            mean = np.average(y, weights=sample_weight)
            res = np.average((y - pred) ** 2, weights=sample_weight)
            tot = np.average((y - mean) ** 2, weights=sample_weight)
            return float(1.0 - res / tot)

    class ClassifierMixin:  # type: ignore[no-redef]
        _estimator_type = "classifier"

        def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
            """Accuracy of ``predict(X)``."""
            hits = self.predict(X) == np.asarray(y)  # type: ignore[attr-defined]
            return float(np.average(hits, weights=sample_weight))


def _same_param(a: Any, b: Any) -> bool:
    try:
        return bool(a is b or a == b)
    except Exception:  # e.g. arrays
        return False


# ---------------------------------------------------------------------------
# Input validation (scikit-learn's conventions and messages)
# ---------------------------------------------------------------------------


def _feature_names(X: Any) -> Optional[np.ndarray]:
    columns = getattr(X, "columns", None)
    if columns is None or len(columns) == 0 or not all(isinstance(c, str) for c in columns):
        return None
    return np.asarray(columns, dtype=object)


def _check_x(est: Any, X: Any, reset: bool) -> np.ndarray:
    name = type(est).__name__
    if hasattr(X, "toarray") or type(X).__module__.startswith("scipy.sparse"):
        raise TypeError(f"{name} does not support sparse input; convert it with X.toarray()")
    names = _feature_names(X)
    arr = np.asarray(X)
    if np.iscomplexobj(arr):
        raise ValueError("Complex data not supported")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        got = "scalar" if arr.ndim == 0 else f"{arr.ndim}D"
        raise ValueError(
            f"Expected 2D array, got {got} array instead. Reshape your data either using "
            "array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it "
            "contains a single sample."
        )
    if arr.shape[0] == 0:
        raise ValueError(f"Found array with 0 sample(s) (shape={arr.shape}) while a minimum of 1 is required by {name}.")
    if arr.shape[1] == 0:
        raise ValueError(f"Found array with 0 feature(s) (shape={arr.shape}) while a minimum of 1 is required by {name}.")
    if getattr(est, "_allow_nan", False):
        if np.isinf(arr).any():
            raise ValueError(f"Input X contains infinity; {name} accepts NaN (missing values) but not infinity.")
    elif not np.isfinite(arr).all():
        raise ValueError(f"Input X contains NaN or infinity; {name} needs finite values.")
    if reset:
        if arr.shape[0] < 2:
            raise ValueError(f"{name} needs at least 2 samples to fit, got 1 sample")
        est.n_features_in_ = arr.shape[1]
        if names is not None:
            est.feature_names_in_ = names
        else:
            est.__dict__.pop("feature_names_in_", None)
        return arr
    fitted = est.__dict__.get("feature_names_in_")
    if fitted is not None and names is None:
        warnings.warn(f"X does not have valid feature names, but {name} was fitted with feature names", UserWarning)
    elif fitted is None and names is not None:
        warnings.warn(f"X has feature names, but {name} was fitted without feature names", UserWarning)
    elif fitted is not None and names is not None and not np.array_equal(fitted, names):
        raise ValueError("The feature names should match those that were passed during fit.")
    if arr.shape[1] != est.n_features_in_:
        raise ValueError(
            f"X has {arr.shape[1]} features, but {name} is expecting {est.n_features_in_} features as input."
        )
    return arr


def _check_y(y: Any, n: int) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. Please change the shape of y to "
            "(n_samples, ), for example using ravel().",
            DataConversionWarning,
            stacklevel=3,
        )
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"y should be a 1d array, got an array of shape {arr.shape} instead.")
    if arr.shape[0] != n:
        raise ValueError(f"Found input variables with inconsistent numbers of samples: [{n}, {arr.shape[0]}]")
    return arr


def _check_weight(sample_weight: Any, n: int) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(n, float(w))
    if w.shape != (n,):
        raise ValueError(f"sample_weight.shape == {w.shape}, expected {(n,)}!")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("sample_weight must be finite and non-negative")
    if not w.any():
        raise ValueError("Sample weights must contain at least one non-zero number.")
    return w


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class _Builtin(BaseEstimator):
    """A scikit-learn estimator around a native learner class ``_native``."""

    _native: ClassVar[Any]

    def _native_params(self, X: Any = None, y: Any = None) -> Optional[dict[str, Any]]:
        """Arguments of the native learner; ``None`` if they need the training data and none
        is given."""
        return dict(self.get_params(deep=False))

    def _dml_native(self) -> Any:
        """The native learner configured with these parameters, which the DML models cross-fit
        in Rust; ``None`` makes them fit this estimator through Python instead."""
        params = self._native_params()
        return None if params is None else self._native(**params)

    def _make_native(self, X: Any, y: Any) -> Any:
        return self._native(**self._native_params(X, y))  # type: ignore[arg-type]

    def _fitted(self) -> Any:
        native = self.__dict__.get("native_")
        if native is None:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call 'fit' with appropriate "
                "arguments before using this estimator."
            )
        return native

    def __getattr__(self, name: str) -> Any:
        # Fitted attributes (``coef_``, ``feature_importances_``, ...) live on the native learner.
        native = self.__dict__.get("native_")
        if not name.endswith("_") or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        if native is None:
            # NotFittedError is an AttributeError, so hasattr() is False before fit.
            raise NotFittedError(f"{type(self).__name__} is not fitted yet; call fit() before reading {name}")
        return getattr(native, name)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        tags.non_deterministic = False
        return tags


class _BuiltinRegressor(RegressorMixin, _Builtin):
    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        """Fit on covariates ``X`` and outcome ``y`` (optionally weighted)."""
        x = _check_x(self, X, reset=True)
        target = np.asarray(_check_y(y, x.shape[0]), dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("Input y contains NaN or infinity.")
        native = self._make_native(x, target)
        w = _check_weight(sample_weight, x.shape[0])
        native.fit(x, target) if w is None else native.fit(x, target, sample_weight=w)
        self.native_ = native
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicted values."""
        native = self._fitted()
        return np.asarray(native.predict(_check_x(self, X, reset=False)))


class _BuiltinClassifier(ClassifierMixin, _Builtin):
    """A binary classifier: ``classes_`` holds the two labels, ``predict_proba`` two columns."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        """Fit on covariates ``X`` and binary labels ``y`` (optionally weighted)."""
        x = _check_x(self, X, reset=True)
        labels = _check_y(y, x.shape[0])
        kind = type_of_target(labels) if type_of_target is not None else _target_kind(labels)
        if kind not in ("binary", "multiclass"):
            raise ValueError(f"Unknown label type: {kind}. {type(self).__name__} needs two classes.")
        classes = np.unique(labels)
        if len(classes) > 2:
            raise ValueError(
                f"Only binary classification is supported. {type(self).__name__} got {len(classes)} classes."
            )
        w = _check_weight(sample_weight, x.shape[0])
        weighted = classes if w is None else np.unique(labels[w > 0])
        if len(classes) < 2 or len(weighted) < 2:
            raise ValueError(
                f"{type(self).__name__} needs two classes (with positive sample weight) but y has one class"
            )
        native = self._make_native(x, labels)
        target = (labels == classes[1]).astype(np.float64)
        native.fit(x, target) if w is None else native.fit(x, target, sample_weight=w)
        self.classes_ = classes
        self.native_ = native
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities, shape ``(n, 2)``: column 1 is ``P(y = classes_[1] | X)``."""
        native = self._fitted()
        p = np.asarray(native.predict_proba(_check_x(self, X, reset=False)))
        return np.column_stack((1.0 - p, p))

    def predict(self, X: Any) -> np.ndarray:
        """The more probable class."""
        p = self.predict_proba(X)[:, 1]
        return self.classes_[(p > 0.5).astype(np.intp)]

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _more_tags(self) -> dict[str, Any]:  # scikit-learn < 1.6
        return {"binary_only": True}


class _CrossValidated:
    """``cv`` as scikit-learn: a number of folds, a splitter with ``split(X, y)`` (e.g.
    ``GroupKFold``, ``KFold(shuffle=True)``) or an iterable of ``(train, test)`` index arrays."""

    cv: Any

    def _native_params(self, X: Any = None, y: Any = None) -> Optional[dict[str, Any]]:
        params = super()._native_params(X, y)  # type: ignore[misc]
        cv: Any = 5 if self.cv is None else self.cv
        if isinstance(cv, numbers.Integral) and not isinstance(cv, bool):
            params["cv"] = int(cv)
        elif X is None:
            return None  # the splits depend on the data
        else:
            splits = cv.split(X, y) if hasattr(cv, "split") else cv
            params["cv"] = [
                (np.asarray(train, dtype=np.intp).tolist(), np.asarray(test, dtype=np.intp).tolist())
                for train, test in splits
            ]
        return params  # type: ignore[no-any-return]


def _target_kind(y: np.ndarray) -> str:
    """``type_of_target`` without scikit-learn (binary, multiclass or continuous)."""
    if y.dtype.kind == "f" and np.any(y != np.round(y)):
        return "continuous"
    return "binary" if len(np.unique(y)) <= 2 else "multiclass"


# ---------------------------------------------------------------------------
# Linear models
# ---------------------------------------------------------------------------


class Ols(_BuiltinRegressor):
    """Ordinary least squares with intercept and an optional ridge penalty (``ridge=α`` matches
    scikit-learn's ``Ridge(alpha=α)``)."""

    _native = _dml_rs.Ols

    def __init__(self, ridge: float = 0.0) -> None:
        self.ridge = ridge


class Lasso(_BuiltinRegressor):
    """Lasso by coordinate descent; the same objective, stopping rule and results as
    scikit-learn's ``Lasso(alpha=α)``."""

    _native = _dml_rs.Lasso

    def __init__(self, alpha: float = 1.0, tol: float = 1e-4, max_iter: int = 100_000) -> None:
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter


class LassoCV(_CrossValidated, _BuiltinRegressor):
    """Lasso with the penalty chosen by cross-validation, as scikit-learn's ``LassoCV``. After
    ``fit``: ``alpha_``, ``alphas_``, ``mse_path_`` (penalties × folds), ``coef_``,
    ``intercept_``, ``n_iter_``.

    Parameters
    ----------
    alphas : number of penalties on the automatic grid (from ``alpha_max`` down to
        ``eps * alpha_max``; ``None`` = 100) or a list of penalties.
    cv : number of unshuffled folds, a scikit-learn splitter or a list of ``(train, test)``
        index arrays.
    eps, tol, max_iter : as scikit-learn.
    """

    _native = _dml_rs.LassoCV

    def __init__(
        self,
        alphas: Union[int, Sequence[float], None] = None,
        eps: float = 1e-3,
        cv: Any = 5,
        tol: float = 1e-4,
        max_iter: int = 100_000,
    ) -> None:
        self.alphas = alphas
        self.eps = eps
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter


class LogisticRegression(_BuiltinClassifier):
    """L2-penalised logistic regression (Newton); the same objective as scikit-learn's
    ``LogisticRegression(C=C)``."""

    _native = _dml_rs.LogisticRegression

    def __init__(self, C: float = 1.0, tol: float = 1e-10, max_iter: int = 100) -> None:
        self.C = C
        self.tol = tol
        self.max_iter = max_iter


class LogisticRegressionCV(_CrossValidated, _BuiltinClassifier):
    """L2-penalised logistic regression with ``C`` chosen by stratified cross-validation, as
    scikit-learn's ``LogisticRegressionCV``. After ``fit``: ``C_``, ``Cs_``, ``scores_`` (folds ×
    Cs), ``classes_``, ``coef_``, ``intercept_``, ``n_iter_``.

    Parameters
    ----------
    Cs : number of values between 1e-4 and 1e4 (``None`` = 10) or a list of values.
    cv : number of stratified folds, a scikit-learn splitter or a list of ``(train, test)``
        index arrays.
    scoring : ``"neg_log_loss"`` (scikit-learn's default from 1.11) or ``"accuracy"``.
    """

    _native = _dml_rs.LogisticRegressionCV

    def __init__(
        self,
        Cs: Union[int, Sequence[float], None] = None,
        cv: Any = 5,
        scoring: str = "neg_log_loss",
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> None:
        self.Cs = Cs
        self.cv = cv
        self.scoring = scoring
        self.tol = tol
        self.max_iter = max_iter


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------


class _Forest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = None,
        bootstrap: bool = True,
        max_samples: Optional[float] = None,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state


class RandomForestRegressor(_Forest, _BuiltinRegressor):
    """Random forest implemented in Rust; the trees are grown in parallel. Parameters follow
    scikit-learn's ``RandomForestRegressor`` (``max_features=None`` means all features);
    ``random_state`` must be an integer. After ``fit``: ``feature_importances_``."""

    _native = _dml_rs.RandomForestRegressor


class RandomForestClassifier(_Forest, _BuiltinClassifier):
    """Random forest implemented in Rust; the trees are grown in parallel. Parameters follow
    scikit-learn's ``RandomForestClassifier`` (``max_features=None`` means ``"sqrt"``);
    ``random_state`` must be an integer. After ``fit``: ``classes_``, ``feature_importances_``."""

    _native = _dml_rs.RandomForestClassifier


class _Boosting:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float, None] = None,
        subsample: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.random_state = random_state


class GradientBoostingRegressor(_Boosting, _BuiltinRegressor):
    """Gradient boosting (squared error) implemented in Rust, growing the same trees as
    scikit-learn's ``GradientBoostingRegressor`` with the same parameters; ``random_state`` must be
    an integer. After ``fit``: ``feature_importances_``."""

    _native = _dml_rs.GradientBoostingRegressor


class GradientBoostingClassifier(_Boosting, _BuiltinClassifier):
    """Gradient boosting (log loss) implemented in Rust, growing the same trees as scikit-learn's
    ``GradientBoostingClassifier`` with the same parameters; ``random_state`` must be an integer.
    After ``fit``: ``classes_``, ``feature_importances_``."""

    _native = _dml_rs.GradientBoostingClassifier


class _HistBoosting:
    # Missing values (NaN) are sent down the learned side of every split, as scikit-learn.
    _allow_nan = True

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        tags.input_tags.allow_nan = True
        return tags

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: Optional[int] = 31,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_features: float = 1.0,
        max_bins: int = 255,
        early_stopping: Union[str, bool] = "auto",
        validation_fraction: Optional[float] = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        random_state: int = 0,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state


class HistGradientBoostingRegressor(_HistBoosting, _BuiltinRegressor):
    """Histogram-based gradient boosting implemented in Rust: the same results as scikit-learn's
    ``HistGradientBoostingRegressor`` with the same parameters, also with missing values (NaN) and
    sample weights; ``random_state`` must be an integer. After ``fit``: ``n_iter_``,
    ``validation_score_``."""

    _native = _dml_rs.HistGradientBoostingRegressor


class HistGradientBoostingClassifier(_HistBoosting, _BuiltinClassifier):
    """Histogram-based gradient boosting implemented in Rust: the same results as scikit-learn's
    ``HistGradientBoostingClassifier`` with the same parameters, also with missing values (NaN)
    and sample weights; ``random_state`` must be an integer. After ``fit``: ``classes_``,
    ``n_iter_``, ``validation_score_``."""

    _native = _dml_rs.HistGradientBoostingClassifier


# ---------------------------------------------------------------------------
# Grid search over built-in (or any) learners
# ---------------------------------------------------------------------------


class GridSearchRegressor(_CrossValidated, _BuiltinRegressor):
    """Grid search over regressors: the candidate with the smallest cross-validated mean squared
    error is refitted on all data. ``cv`` is a number of unshuffled folds, a scikit-learn splitter
    or a list of ``(train, test)`` index arrays. With only built-in candidates the whole search
    runs in Rust. After ``fit``: ``best_index_``, ``cv_scores_``."""

    _native = _dml_rs.GridSearchRegressor

    def __init__(self, candidates: Sequence[Any], cv: Any = 5) -> None:
        self.candidates = candidates
        self.cv = cv


class GridSearchClassifier(_CrossValidated, _BuiltinClassifier):
    """Grid search over classifiers by cross-validated log loss; see `GridSearchRegressor`."""

    _native = _dml_rs.GridSearchClassifier

    def __init__(self, candidates: Sequence[Any], cv: Any = 5) -> None:
        self.candidates = candidates
        self.cv = cv


# ---------------------------------------------------------------------------
# Generalized random forests
# ---------------------------------------------------------------------------


class RegressionForest(_BuiltinRegressor):
    """Honest regression forest (R's ``grf::regression_forest``), implemented in Rust: a nuisance
    learner with variance estimates and out-of-bag predictions. Parameters follow grf
    (``n_estimators`` = ``num.trees``, ``mtry``, ``min_node_size``, ``honesty``, ...).
    After ``fit``: ``variable_importance_``."""

    _native = _dml_rs.RegressionForest

    def __init__(
        self,
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
    ) -> None:
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.ci_group_size = ci_group_size
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> Any:  # type: ignore[override]
        """Fit on covariates ``X`` and outcome ``y`` (sample weights are not supported)."""
        return super().fit(X, y)

    def predict_with_variance(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(prediction, variance)`` (bootstrap of little bags)."""
        return self._fitted().predict_with_variance(_check_x(self, X, reset=False))  # type: ignore[no-any-return]

    def predict_oob(self, X: Any) -> tuple[np.ndarray, np.ndarray]:
        """Out-of-bag ``(prediction, variance)`` for the training rows."""
        return self._fitted().predict_oob(_check_x(self, X, reset=False))  # type: ignore[no-any-return]


class QuantileForest(_Builtin):
    """Quantile forest (R's ``grf::quantile_forest``), implemented in Rust: ``predict`` gives the
    conditional ``quantiles``, shape ``(n, len(quantiles))``. Other parameters as
    `RegressionForest`."""

    _native = _dml_rs.QuantileForest

    def __init__(
        self,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
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
    ) -> None:
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.random_state = random_state

    def _native_params(self, X: Any = None, y: Any = None) -> Optional[dict[str, Any]]:
        params = dict(self.get_params(deep=False))
        params["quantiles"] = [float(q) for q in self.quantiles]
        return params

    def fit(self, X: Any, y: Any) -> Any:
        """Fit on covariates ``X`` and outcome ``y``."""
        x = _check_x(self, X, reset=True)
        target = np.asarray(_check_y(y, x.shape[0]), dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("Input y contains NaN or infinity.")
        native = self._make_native(x, target)
        native.fit(x, target)
        self.native_ = native
        return self

    def predict(self, X: Any, quantiles: Optional[Sequence[float]] = None) -> np.ndarray:
        """Conditional quantiles, ``(n, len(quantiles))`` (default: the fitted ``quantiles``)."""
        q = None if quantiles is None else [float(v) for v in quantiles]
        return np.asarray(self._fitted().predict(_check_x(self, X, reset=False), q))

    def predict_oob(self, X: Any, quantiles: Optional[Sequence[float]] = None) -> np.ndarray:
        """Out-of-bag quantiles for the training rows."""
        q = None if quantiles is None else [float(v) for v in quantiles]
        return np.asarray(self._fitted().predict_oob(_check_x(self, X, reset=False), q))


__all__ = [
    "Ols",
    "Lasso",
    "LassoCV",
    "LogisticRegression",
    "LogisticRegressionCV",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "GridSearchRegressor",
    "GridSearchClassifier",
    "RegressionForest",
    "QuantileForest",
]

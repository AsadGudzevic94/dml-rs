"""scikit-learn style estimators on top of the Rust core."""

from __future__ import annotations

import inspect

import numpy as np

from . import _dml_rs
from ._blp import blp_from
from ._validation import _as_1d, _as_2d


def _treatment_names(d):
    columns = getattr(d, "columns", None)
    if columns is not None:
        return [str(c) for c in columns]
    name = getattr(d, "name", None)  # pandas Series
    return [str(name)] if name is not None else None


def _splits(folds):
    """Accept one repetition ``[(train, test), ...]`` or a list of repetitions."""
    if folds is None:
        return None
    folds = list(folds)
    first = folds[0]
    one_rep = len(first) == 2 and np.ndim(first[0]) == 1
    reps = [folds] if one_rep else folds
    return [[(np.asarray(tr, dtype=np.int64).tolist(), np.asarray(te, dtype=np.int64).tolist()) for tr, te in rep]
            for rep in reps]


class _Params:
    """scikit-learn parameter protocol (``get_params`` / ``set_params`` / ``clone``)."""

    @classmethod
    def _get_param_names(cls):
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self._get_param_names():
                raise ValueError(f"invalid parameter {key!r} for {type(self).__name__}")
            setattr(self, key, value)
        return self

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    def _check_fitted(self):
        if not hasattr(self, "result_"):
            raise AttributeError(f"{type(self).__name__} is not fitted yet; call fit() first")
        return self.result_


class _DMLEstimator(_Params):
    """Shared fitted-model API of the DML estimators."""

    _boot = None

    # -- fitted attributes ---------------------------------------------------
    def _store(self, result):
        self.result_ = result
        self.coef_ = result.coef
        self.se_ = result.se
        self.t_stat_ = result.t_stat
        self.pval_ = result.pval
        self.all_coef_ = result.all_coef
        self.all_se_ = result.all_se
        self.psi_ = result.psi
        self.treatment_names_ = result.treatment_names
        self.predictions_ = {name: result.predictions(name) for name in result.nuisance_names}
        self._boot = None
        return self

    def confint(self, level=0.95, joint=False):
        """Confidence intervals, shape ``(n_treat, 2)``. ``joint=True`` needs :meth:`bootstrap`."""
        result = self._check_fitted()
        if joint:
            if self._boot is None:
                raise ValueError("call bootstrap() before confint(joint=True)")
            return self._boot.joint_confint(level)
        return result.confint(level)

    def bootstrap(self, method="normal", n_boot=500, random_state=0):
        """Multiplier bootstrap (``'normal'``, ``'Bayes'`` or ``'wild'``) for joint inference."""
        self._boot = self._check_fitted().bootstrap(method, n_boot, random_state)
        self.boot_t_stat_ = self._boot.t_stats
        return self

    def p_adjust(self, method="romano-wolf"):
        """Adjusted p-values: ``'romano-wolf'`` (needs :meth:`bootstrap`), ``'holm'`` or ``'bonferroni'``."""
        result = self._check_fitted()
        if method.lower() in ("romano-wolf", "rw"):
            if self._boot is None:
                raise ValueError("call bootstrap() before p_adjust('romano-wolf')")
            return self._boot.romano_wolf()
        return result.p_adjust(method)

    @property
    def summary(self):
        """Coefficient table as a pandas DataFrame (or a dict without pandas)."""
        result = self._check_fitted()
        ci = result.confint(0.95)
        table = {
            "coef": result.coef,
            "std err": result.se,
            "t": result.t_stat,
            "P>|t|": result.pval,
            "2.5 %": ci[:, 0],
            "97.5 %": ci[:, 1],
        }
        try:
            import pandas as pd
        except ImportError:
            return {"index": result.treatment_names, **table}
        return pd.DataFrame(table, index=result.treatment_names)

    def __str__(self):
        return str(self.result_) if hasattr(self, "result_") else repr(self)


class PLR(_DMLEstimator):
    """Partially linear regression ``Y = θ·D + g(X) + ε``.

    Parameters
    ----------
    ml_l : regressor for ``E[Y | X]`` — a ``dml_rs`` learner or any object with
        ``fit`` / ``predict`` (e.g. a scikit-learn estimator).
    ml_m : regressor for ``E[D | X]``.
    ml_g : regressor for ``g(X)``, used by ``score="IV-type"`` (default: a clone of ``ml_l``).
    score : ``"partialling out"`` or ``"IV-type"``.
    n_folds, n_rep : cross-fitting folds and repetitions of the sample split.
    random_state : seed for the sample splits.
    """

    def __init__(self, ml_l, ml_m, ml_g=None, score="partialling out", n_folds=5, n_rep=1, random_state=0):
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_g = ml_g
        self.score = score
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(self, X, y, d, folds=None):
        """Fit on covariates ``X``, outcome ``y`` and treatment(s) ``d``.

        ``folds`` optionally fixes the sample splits: a list of ``(train, test)``
        index pairs, or one such list per repetition.
        """
        result = _dml_rs.fit_plr(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_l, self.ml_m, self.ml_g,
            score=self.score, n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d),
        )
        return self._store(result)


class IRM(_DMLEstimator):
    """Interactive regression model for a binary treatment (ATE or ATTE).

    Parameters
    ----------
    ml_g : regressor for ``E[Y | D, X]`` (fitted separately on treated and controls).
    ml_m : classifier for the propensity ``P(D = 1 | X)`` with ``predict_proba``.
    score : ``"ATE"`` or ``"ATTE"``.
    clipping_threshold : propensities are clipped to ``[t, 1 - t]``.
    normalize_ipw : normalise the inverse-probability weights.
    """

    def __init__(self, ml_g, ml_m, score="ATE", clipping_threshold=1e-2, normalize_ipw=False, n_folds=5,
                 n_rep=1, random_state=0):
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.score = score
        self.clipping_threshold = clipping_threshold
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(self, X, y, d, folds=None):
        result = _dml_rs.fit_irm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            score=self.score, clipping_threshold=self.clipping_threshold, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d),
        )
        return self._store(result)

    def cate(self, basis):
        """Best linear predictor of the CATE on ``basis`` (ATE score), as DoubleML's ``cate``."""
        return blp_from(self._check_fitted().cate(_as_2d(basis, "basis")), basis, "b")

    def gate(self, groups):
        """Group average treatment effects for 0/1 group dummies (ATE score), as DoubleML's ``gate``."""
        return blp_from(self._check_fitted().gate(_as_2d(groups, "groups")), groups, "group_")


class IIVM(_DMLEstimator):
    """Interactive IV model: local average treatment effect (LATE) of a binary
    treatment ``d`` using a binary instrument ``z``.

    Parameters
    ----------
    ml_g : regressor for ``E[Y | Z, X]`` (fitted separately for ``Z = 0`` and ``Z = 1``).
    ml_m : classifier for the instrument propensity ``P(Z = 1 | X)``.
    ml_r : classifier for ``P(D = 1 | Z, X)`` (fitted separately for ``Z = 0`` and ``Z = 1``).
    subgroups : dict with ``always_takers`` / ``never_takers`` (default both True). Set one
        to False under one-sided non-compliance.
    clipping_threshold : instrument propensities are clipped to ``[t, 1 - t]``.
    normalize_ipw : normalise the inverse-probability weights within instrument arms.
    """

    def __init__(self, ml_g, ml_m, ml_r, subgroups=None, clipping_threshold=1e-2, normalize_ipw=False,
                 n_folds=5, n_rep=1, random_state=0):
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.ml_r = ml_r
        self.subgroups = subgroups
        self.clipping_threshold = clipping_threshold
        self.normalize_ipw = normalize_ipw
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(self, X, y, d, z, folds=None):
        subgroups = {"always_takers": True, "never_takers": True, **(self.subgroups or {})}
        result = _dml_rs.fit_iivm(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_1d(z, "z"), self.ml_g, self.ml_m, self.ml_r,
            always_takers=subgroups["always_takers"], never_takers=subgroups["never_takers"],
            clipping_threshold=self.clipping_threshold, normalize_ipw=self.normalize_ipw,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d),
        )
        return self._store(result)


class _DIDBase(_DMLEstimator):
    def __init__(self, ml_g, ml_m=None, score="observational", in_sample_normalization=True,
                 clipping_threshold=1e-2, n_folds=5, n_rep=1, random_state=0):
        self.ml_g = ml_g
        self.ml_m = ml_m
        self.score = score
        self.in_sample_normalization = in_sample_normalization
        self.clipping_threshold = clipping_threshold
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def _fit(self, X, y, d, t, folds):
        result = _dml_rs.fit_did(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, self.ml_m,
            t=None if t is None else _as_1d(t, "t"), score=self.score,
            in_sample_normalization=self.in_sample_normalization, clipping_threshold=self.clipping_threshold,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d),
        )
        return self._store(result)


class DID(_DIDBase):
    """Doubly robust difference-in-differences (ATT) with **panel data** over two periods.

    ``y`` passed to :meth:`fit` is each unit's outcome change ``Y_post - Y_pre``.

    Parameters
    ----------
    ml_g : regressor for ``E[ΔY | D = d, X]`` (fitted per treatment group).
    ml_m : classifier for the propensity ``P(D = 1 | X)``; required for ``score="observational"``.
    score : ``"observational"`` (doubly robust) or ``"experimental"`` (randomised treatment).
    in_sample_normalization : normalise the weights by their sample means (DoubleML default).
    """

    def fit(self, X, y, d, folds=None):
        return self._fit(X, y, d, None, folds)


class DIDCS(_DIDBase):
    """Doubly robust difference-in-differences (ATT) with **repeated cross-sections**.

    Each row is observed once; ``t`` passed to :meth:`fit` is 1 in the post-period, 0 before.
    Parameters as :class:`DID`, with ``ml_g`` fitted in each treatment-group x period cell.
    """

    def fit(self, X, y, d, t, folds=None):
        return self._fit(X, y, d, t, folds)


class PLIV(_DMLEstimator):
    """Partially linear IV regression with one instrument ``z``.

    Parameters
    ----------
    ml_l : regressor for ``E[Y | X]``.
    ml_m : regressor for ``E[Z | X]``.
    ml_r : regressor for ``E[D | X]``.
    """

    def __init__(self, ml_l, ml_m, ml_r, n_folds=5, n_rep=1, random_state=0):
        self.ml_l = ml_l
        self.ml_m = ml_m
        self.ml_r = ml_r
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def fit(self, X, y, d, z, folds=None):
        result = _dml_rs.fit_pliv(
            _as_1d(y, "y"), _as_2d(d, "d"), _as_2d(X, "X"), _as_1d(z, "z"), self.ml_l, self.ml_m, self.ml_r,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state,
            splits=_splits(folds), treatment_names=_treatment_names(d),
        )
        return self._store(result)

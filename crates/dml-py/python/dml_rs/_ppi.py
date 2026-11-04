"""Prediction-powered inference: a few labels and many predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from . import _dml_rs
from ._models import Framework, _DMLEstimator, _is_classifier
from ._validation import _as_1d, _as_2d

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from typing_extensions import Self


class PPI(_DMLEstimator):
    """Prediction-powered inference (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, 2023) with
    the power tuning of PPI++ (Angelopoulos, Duchi & Zrnic, 2023): valid inference on a mean or
    on linear / logistic regression coefficients from a small labelled sample and a large sample
    that only has predictions of the outcome (from any model: an ML model, an LLM, a
    simulator).

    The estimate from the predictions is corrected with the prediction errors on the labelled
    sample, so it is unbiased whatever the quality of the predictions; for a mean,
    ``θ̂ = λ·mean(ŷ_unlabeled) + mean(y − λ·ŷ)``. ``lam=None`` (PPI++) picks the weight ``λ``
    in ``[0, 1]`` that minimises the variance, so the estimate is (asymptotically) never less
    precise than the labelled-only one; ``lam=1`` is the original PPI, ``lam=0`` the classical
    estimate. The computations follow ``ppi_py`` (``ppi_mean_ci``, ``ppi_ols_ci``,
    ``ppi_logistic_ci``) and match it to 1e-9 (logistic: 1e-7, where ``ppi_py``'s optimiser
    stops short).

    With ``learner``, the predictions are made here by cross-fitting the learner on the
    labelled sample (cross-prediction-powered inference, Zrnic & Candès, 2024): out-of-fold
    predictions for the labelled rows, the average of the fold models for the unlabelled ones.

    After ``fit``: the fitted-model API of the estimate (``coef_``, ``se_``, ``summary``,
    ``confint``, ``confidence_sequence``, ``framework``), ``lam_``, ``classical_`` (the
    labelled-only estimate as a `Framework`), ``effective_n_`` (the labelled sample size the
    classical estimate would need for the same precision), ``n_labeled_``, ``n_unlabeled_``
    and ``yhat_`` / ``yhat_unlabeled_`` (the predictions used).

    Parameters
    ----------
    estimand : ``"mean"``, ``"ols"`` (least-squares coefficients of ``y`` on ``X``) or
        ``"logistic"`` (logistic regression of a 0/1 ``y`` on ``X``; predictions are
        probabilities).
    lam : weight of the predictions in ``[0, 1]``; ``None`` (default) tunes it (PPI++).
    learner : optional regressor or classifier to cross-fit the predictions from
        ``features``; a classifier gives probabilities.
    fit_intercept : add an intercept column to ``X`` for the regressions (default True;
        ``ppi_py`` adds none).
    n_folds : folds for cross-fitting the learner.
    random_state : seed of the folds.
    """

    def __init__(
        self,
        estimand: Literal["mean", "ols", "logistic"] = "mean",
        lam: Optional[float] = None,
        learner: Any = None,
        fit_intercept: bool = True,
        n_folds: int = 5,
        random_state: int = 0,
    ) -> None:
        self.estimand = estimand
        self.lam = lam
        self.learner = learner
        self.fit_intercept = fit_intercept
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(
        self,
        y: ArrayLike,
        yhat: Optional[ArrayLike] = None,
        yhat_unlabeled: Optional[ArrayLike] = None,
        X: Optional[ArrayLike] = None,
        X_unlabeled: Optional[ArrayLike] = None,
        features: Optional[ArrayLike] = None,
        features_unlabeled: Optional[ArrayLike] = None,
    ) -> Self:
        """Fit on the labelled outcomes ``y`` with predictions ``yhat`` and the unlabelled
        predictions ``yhat_unlabeled`` (or, with ``learner``, the ``features`` of both samples
        to predict from; they default to ``X`` / ``X_unlabeled``). The regressions need the
        covariates ``X`` and ``X_unlabeled``."""
        if self.estimand not in ("mean", "ols", "logistic"):
            raise ValueError(f"estimand must be 'mean', 'ols' or 'logistic', got {self.estimand!r}")
        y_arr = _as_1d(y, "y")
        x = xu = None
        names = None
        if self.estimand == "mean":
            if X is not None or X_unlabeled is not None:
                raise ValueError("the mean takes no covariates X")
        else:
            if X is None or X_unlabeled is None:
                raise ValueError(f"estimand={self.estimand!r} needs X and X_unlabeled")
            names = [str(c) for c in X.columns] if hasattr(X, "columns") else None
            x, xu = _as_2d(X, "X"), _as_2d(X_unlabeled, "X_unlabeled")
            if names is None:
                names = [f"x{j}" for j in range(x.shape[1])]
            if self.fit_intercept:
                x = np.column_stack([np.ones(x.shape[0]), x])
                xu = np.column_stack([np.ones(xu.shape[0]), xu])
                names = ["intercept", *names]
        kwargs: dict[str, Any] = dict(x=x, x_unlabelled=xu, names=names,
                                      lam=None if self.lam is None else float(self.lam))
        if self.learner is not None:
            if yhat is not None or yhat_unlabeled is not None:
                raise ValueError("pass either predictions or a learner, not both")
            f = X if features is None else features
            fu = X_unlabeled if features_unlabeled is None else features_unlabeled
            if f is None or fu is None:
                raise ValueError("a learner needs features and features_unlabeled to predict from")
            kwargs.update(learner=self.learner, classifier=_is_classifier(self.learner),
                          features=_as_2d(f, "features"), features_unlabelled=_as_2d(fu, "features_unlabeled"),
                          n_folds=int(self.n_folds), seed=int(self.random_state))
        else:
            if yhat is None or yhat_unlabeled is None:
                raise ValueError("pass yhat and yhat_unlabeled, or set a learner")
            kwargs.update(yhat=_as_1d(yhat, "yhat"), yhat_unlabelled=_as_1d(yhat_unlabeled, "yhat_unlabeled"))
        r = _dml_rs.fit_ppi(self.estimand, y_arr, **kwargs)
        self._ppi = r
        self._store(r.estimate)
        self.lam_ = r.lam
        self.classical_ = Framework(r.classical)
        self.effective_n_ = np.asarray(r.effective_sample_size)
        self.n_labeled_, self.n_unlabeled_ = r.n_labelled, r.n_unlabelled
        if r.predictions is None:
            self.yhat_, self.yhat_unlabeled_ = kwargs["yhat"], kwargs["yhat_unlabelled"]
        else:
            self.yhat_, self.yhat_unlabeled_ = (np.asarray(p) for p in r.predictions)
        return self

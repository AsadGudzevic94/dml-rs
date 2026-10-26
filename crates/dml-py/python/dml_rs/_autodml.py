"""Automatic debiased machine learning (Chernozhukov, Newey and Singh, 2022): effects that are
means of a linear functional of the regression, with the Riesz representer learned from the
functional."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import numpy as np

from . import _dml_rs
from ._models import _DMLEstimator, _HeterogeneityMixin, _splits, _treatment_names
from ._tuning import nuisance_fit
from ._validation import _as_1d, _as_2d, _as_clusters
from .data import accepts_data

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from typing_extensions import Self

    from ._models import Folds


class Functional:
    """A linear functional ``m(W; f)`` of a function of ``W = (D, X)`` (treatment first)."""

    def _native(self, n: int) -> dict[str, Any]:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class ATE(Functional):
    """Average treatment effect of a binary treatment: ``m = f(1, X) - f(0, X)``."""

    def _native(self, n: int) -> dict[str, Any]:
        return {"functional": "ate"}


class AverageDerivative(Functional):
    """Average derivative ``E[dγ(D, X)/dD]`` of a continuous treatment, by a central difference
    with step ``step`` (default ``1e-4`` times the standard deviation of the treatment). Needs a
    regression that is smooth in the treatment (linear models, polynomials, neural networks); for
    trees and forests use `ShiftEffect`."""

    def __init__(self, step: Optional[float] = None) -> None:
        self.step = step

    def _native(self, n: int) -> dict[str, Any]:
        return {"functional": "average_derivative", "step": self.step}


class ShiftEffect(Functional):
    """Average effect of shifting every unit's treatment by ``delta``: ``m = f(D + δ, X) - f(D, X)``,
    divided by ``δ`` with ``per_unit=True``."""

    def __init__(self, delta: float, per_unit: bool = False) -> None:
        self.delta = delta
        self.per_unit = per_unit

    def _native(self, n: int) -> dict[str, Any]:
        return {"functional": "shift", "delta": float(self.delta), "per_unit": bool(self.per_unit)}


class PolicyEffect(Functional):
    """Average effect of a policy that assigns each unit a treatment: ``m = f(d_new, X) - f(D, X)``.

    ``policy`` is the array of new treatments (one per observation) or a function ``policy(W)``
    of the data matrix ``W = (D, X)`` returning them."""

    def __init__(self, policy: Union[ArrayLike, Callable[[Any], Any]]) -> None:
        self.policy = policy

    def _native(self, n: int) -> dict[str, Any]:
        if callable(self.policy):
            policy = self.policy

            def terms(W: Any) -> list[tuple[Any, Any]]:
                new = W.copy()
                new[:, 0] = np.asarray(policy(W), dtype=np.float64).reshape(-1)
                return [(new, 1.0), (W, -1.0)]

            return CustomFunctional(terms, name="policy effect")._native(n)
        return {"functional": "policy", "policy": _as_1d(self.policy, "policy")}


class CustomFunctional(Functional):
    """Any linear functional, given by ``terms(W)``: a list of ``(W_k, weight_k)`` pairs with
    ``m(W; f) = sum_k weight_k * f(W_k)``. ``W`` is the data matrix (treatment in column 0,
    covariates after it); each ``W_k`` has its shape, and a weight is a number or one value per
    row. For example the ATE is ``[(W with D = 1, 1.0), (W with D = 0, -1.0)]``."""

    def __init__(self, terms: Callable[[Any], Sequence[tuple[Any, Any]]], name: str = "custom functional") -> None:
        self.terms = terms
        self.name = name

    def _native(self, n: int) -> dict[str, Any]:
        user = self.terms

        def terms(W: Any) -> list[tuple[Any, Any]]:
            out = []
            for wk, c in user(W):
                wk = np.ascontiguousarray(wk, dtype=np.float64)
                if wk.shape != W.shape:
                    raise ValueError(f"a functional term has shape {wk.shape}; expected {W.shape}")
                weight = np.ascontiguousarray(np.broadcast_to(np.asarray(c, dtype=np.float64), (W.shape[0],)))
                out.append((wk, weight))
            return out

        return {"functional": "custom", "terms": terms, "functional_name": self.name}


_FUNCTIONALS = {"ate": ATE, "average_derivative": AverageDerivative}


class RieszLasso:
    """The Lasso Riesz representer of Chernozhukov, Newey and Singh (2022): ``α(w) = b(w)'ρ`` for a
    dictionary ``b``, minimising the sample Riesz loss ``ρ'Gρ - 2M'ρ`` plus a weighted l1 penalty.

    Parameters
    ----------
    degree : the dictionary holds all monomials of the covariates up to this total degree ...
    treatment_degree : ... each times the powers ``1, d, ..., d^treatment_degree`` of the treatment.
    penalty : fixed penalty ``r`` (0: unpenalised); ``None`` chooses it by ``cv``-fold
        cross-validation of the Riesz loss over ``n_alphas`` penalties down to ``eps`` times the
        largest useful one.
    dictionary : a function ``dictionary(W)`` returning the basis matrix (rows of ``W``: treatment
        first, then covariates), replacing the polynomial dictionary.
    """

    def __init__(
        self,
        degree: int = 2,
        treatment_degree: int = 1,
        penalty: Optional[float] = None,
        n_alphas: int = 30,
        eps: float = 1e-3,
        cv: int = 5,
        dictionary: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.degree = degree
        self.treatment_degree = treatment_degree
        self.penalty = penalty
        self.n_alphas = n_alphas
        self.eps = eps
        self.cv = cv
        self.dictionary = dictionary

    def _native(self) -> dict[str, Any]:
        dictionary = None
        if self.dictionary is not None:
            user = self.dictionary

            def dictionary(W: Any) -> Any:
                b = np.ascontiguousarray(user(W), dtype=np.float64)
                if b.ndim != 2:
                    raise ValueError("the dictionary must return a 2-D array")
                return b

        return {"degree": int(self.degree), "treatment_degree": int(self.treatment_degree),
                "penalty": None if self.penalty is None else float(self.penalty), "n_alphas": int(self.n_alphas),
                "eps": float(self.eps), "cv": int(self.cv), "dictionary": dictionary}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"RieszLasso({args})"


class AutoDML(_HeterogeneityMixin, _DMLEstimator):
    """Automatic debiased machine learning (Chernozhukov, Newey and Singh, 2022) of
    ``θ = E[m(W; γ)]``, a mean of a linear functional ``m`` of the regression ``γ(D, X) = E[Y | D, X]``:
    the average treatment effect, average derivatives, effects of shifting the treatment or of a
    policy, or any functional you define.

    The score ``m(W; γ) + α(W)(Y - γ(W)) - θ`` needs the Riesz representer ``α`` of the
    functional (for the ATE the inverse propensity weights); here it is learned from the functional
    itself, so no formula has to be derived. Inference, repeated sample splitting, clustering,
    sensitivity analysis and ``cate`` / ``gate`` work as for the other models.

    Parameters
    ----------
    ml_g : regressor for ``E[Y | D, X]``, fitted on ``W = (D, X)``.
    functional : ``"ate"``, ``"average_derivative"`` or a `Functional` (`ATE`,
        `AverageDerivative`, `ShiftEffect`, `PolicyEffect`, `CustomFunctional`).
    riesz : learner of the Riesz representer (default `RieszLasso()`).
    n_folds, n_rep, random_state : sample splitting.
    """

    _tuning_kind = None

    def _nuisance_spec(self) -> list[tuple[str, str]]:
        return [("ml_g", "ml_g")]

    def __init__(
        self,
        ml_g: Any,
        functional: Union[str, Functional] = "ate",
        riesz: Optional[RieszLasso] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        random_state: int = 0,
    ) -> None:
        self.ml_g = ml_g
        self.functional = functional
        self.riesz = riesz
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.random_state = random_state

    def _functional(self) -> Functional:
        if isinstance(self.functional, Functional):
            return self.functional
        if isinstance(self.functional, str) and self.functional in _FUNCTIONALS:
            return _FUNCTIONALS[self.functional]()  # type: ignore[no-any-return]
        raise ValueError(f"unknown functional {self.functional!r}: use 'ate', 'average_derivative' or a Functional")

    @accepts_data("x", "y", "d", cluster=True)
    @nuisance_fit
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        d: ArrayLike,
        folds: Optional[Folds] = None,
        cluster: Optional[ArrayLike] = None,
        store_models: bool = False,
    ) -> Self:
        """Fit on covariates ``X``, outcome ``y`` and treatment(s) ``d`` (each treatment gets the
        functional; other treatments act as covariates). ``predictions_`` holds ``ml_g`` (the
        regression at the data), ``riesz`` (the Riesz representer) and ``functional``
        (``m(W; γ̂)``)."""
        self._remember(X, y, d, folds=folds, cluster=cluster)
        y_arr = _as_1d(y, "y")
        riesz = (self.riesz if self.riesz is not None else RieszLasso())._native()
        result = _dml_rs.fit_autodml(
            y_arr, _as_2d(d, "d"), _as_2d(X, "X"), self.ml_g, **self._functional()._native(len(y_arr)), **riesz,
            n_folds=self.n_folds, n_rep=self.n_rep, seed=self.random_state, splits=_splits(folds),
            treatment_names=_treatment_names(d), cluster=_as_clusters(cluster),
        )
        return self._store(result)


__all__ = ["ATE", "AutoDML", "AverageDerivative", "CustomFunctional", "Functional", "PolicyEffect", "RieszLasso",
           "ShiftEffect"]

"""Learner hyperparameters per nuisance model, hyperparameter tuning, stored learners and callable
scores, as DoubleML's ``set_ml_nuisance_params``, ``tune``, ``tune_ml_models``,
``fit(store_models=True)`` and ``score=callable``."""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from . import _dml_rs

# ---------------------------------------------------------------------------
# Learners that change from fit to fit
# ---------------------------------------------------------------------------


def _clone(learner: Any) -> Any:
    """``sklearn.base.clone`` for scikit-learn style learners, a copy for the built-in ones."""
    if hasattr(learner, "get_params"):
        from sklearn.base import clone  # type: ignore[import-untyped]

        return clone(learner)
    import copy

    return copy.deepcopy(learner)


def _with_params(learner: Any, params: dict[str, Any]) -> Any:
    if not hasattr(learner, "set_params"):
        raise TypeError(
            f"learner {learner!r} has no set_params method: per-learner parameters and tuning need the "
            "scikit-learn estimator API (construct built-in dml_rs learners with the parameters instead)"
        )
    return _clone(learner).set_params(**params)


class _Calls:
    """The learner for each successive fit of one nuisance model, and the fitted learners.

    Shared by every clone of its `_Slot`: scikit-learn's ``clone`` deep-copies parameters, and
    this object's copy is itself."""

    def __init__(self, learners: list[Any]) -> None:
        self.learners = learners
        self.fitted: list[Any] = []

    def __copy__(self) -> _Calls:
        return self

    def __deepcopy__(self, memo: Any) -> _Calls:
        return self

    def next_learner(self) -> Any:
        if len(self.learners) == 1:
            return self.learners[0]
        i = len(self.fitted)
        if i >= len(self.learners):
            raise RuntimeError(
                "the learner was fitted more often than there are folds: fold-specific parameters need "
                "a model that fits each nuisance learner once per fold"
            )
        return self.learners[i]


class _Slot:
    """A regressor that fits the next learner of its `_Calls` and records it."""

    def __init__(self, calls: _Calls) -> None:
        self.calls = calls

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"calls": self.calls}

    def set_params(self, **params: Any) -> _Slot:
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Slot:
        est = _clone(self.calls.next_learner())
        if sample_weight is None:
            est.fit(X, y)
        else:
            est.fit(X, y, sample_weight=sample_weight)
        self.calls.fitted.append(est)
        self.estimator_ = est
        return self

    def predict(self, X: Any) -> Any:
        return self.estimator_.predict(X)


class _ClassifierSlot(_Slot):
    def predict_proba(self, X: Any) -> Any:
        p = np.asarray(self.estimator_.predict_proba(X))
        # Built-in classifiers return P(Y = 1) only.
        return np.column_stack((1.0 - p, p)) if p.ndim == 1 else p


def _is_classifier(learner: Any) -> bool:
    return hasattr(learner, "predict_proba")


def _treatment_names(d: Any) -> list[str]:
    columns = getattr(d, "columns", None)
    if columns is not None:
        return [str(c) for c in columns]
    name = getattr(d, "name", None)
    if name is not None:
        return [str(name)]
    k = 1 if np.ndim(np.asarray(d)) == 1 else np.asarray(d).shape[1]
    return ["d"] if k == 1 else [f"d{j}" for j in range(k)]


def _split_shape(folds: Any) -> Optional[tuple[int, int]]:
    """``(n_rep, n_folds)`` of explicit folds (one repetition or a list of them)."""
    if folds is None:
        return None
    folds = list(folds)
    one_rep = len(folds[0]) == 2 and np.ndim(folds[0][0]) == 1
    reps = [folds] if one_rep else folds
    return len(reps), len(reps[0])


class _LearnerPlan:
    """Learners for one fit: tuned parameters and recorders (``store_models``)."""

    def __init__(self, model: Any, treatments: list[str], n_rep: int, n_folds: int, store: bool) -> None:
        self.model = model
        self.treatments = treatments
        self.n_rep = n_rep
        self.n_folds = n_folds
        self.store = store
        self.calls: dict[str, _Calls] = {}
        self.fold_specific: set[str] = set()
        self.saved: dict[str, Any] = {}

    def _learners(self, name: str, base: Any) -> list[Any]:
        """One learner per fit slot (treatment, repetition, fold), sharing equal parameters."""
        stored = self.model._ml_params.get(name, {})
        unknown = [t for t in stored if t is not None and t not in self.treatments]
        if unknown:
            raise ValueError(f"parameters set for unknown treatment(s) {unknown}; treatments are {self.treatments}")
        seen: list[tuple[dict[str, Any], Any]] = []
        out = []
        for treat in self.treatments:
            for r in range(self.n_rep):
                for k in range(self.n_folds):
                    params: dict[str, Any] = {}
                    for key in (None, treat):
                        if key in stored:
                            nested = stored[key]
                            if len(nested) != self.n_rep or any(len(rep) != self.n_folds for rep in nested):
                                raise ValueError(
                                    f"fold-specific parameters of {name} have shape ({len(nested)}, "
                                    f"{len(nested[0])}) but the fit has n_rep={self.n_rep}, n_folds={self.n_folds}"
                                )
                            params = {**params, **nested[r][k]}
                    learner = next((lrn for p, lrn in seen if p == params), None)
                    if learner is None:
                        learner = base if not params else _with_params(base, params)
                        seen.append((params, learner))
                    out.append(learner)
        return out

    def __enter__(self) -> _LearnerPlan:
        model = self.model
        bases = {}
        for name, attr in model._nuisance_spec():
            base = getattr(model, attr)
            if base is None:
                base = getattr(model, model._learner_fallback.get(attr, attr))
            if base is not None:
                bases[name] = (attr, base)
        overrides: dict[str, Any] = {}
        for name, (attr, base) in bases.items():
            learners = self._learners(name, base)
            constant = all(lrn is learners[0] for lrn in learners)
            if self.store or not constant:
                calls = _Calls([learners[0]] if constant else learners)
                learner: Any = (_ClassifierSlot if _is_classifier(base) else _Slot)(calls)
                self.calls[name] = calls
                if not constant:
                    self.fold_specific.add(name)
            else:
                learner = learners[0]
                if learner is base:
                    continue
            if model._override_learners:
                overrides[name] = learner
            else:
                self.saved.setdefault(attr, getattr(model, attr))
                setattr(model, attr, learner)
        model._nuisance_overrides = overrides or None
        if self.calls:
            _dml_rs.set_parallel_folds(False)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for attr, value in self.saved.items():
            setattr(self.model, attr, value)
        self.model._nuisance_overrides = None
        if self.calls:
            _dml_rs.set_parallel_folds(True)
        if exc_type is None:
            slots = len(self.treatments) * self.n_rep * self.n_folds
            for name in self.fold_specific:
                n_fits = len(self.calls[name].fitted)
                if n_fits not in (0, slots):
                    raise ValueError(
                        f"fold-specific parameters for {name} need one fit per fold, but "
                        f"{type(self.model).__name__} fitted it {n_fits} times for {slots} folds"
                    )

    def models(self) -> dict[str, dict[str, list[list[Any]]]]:
        """DoubleML's ``models``: ``{learner: {treatment: [[fold model] per repetition]}}``; a fold
        holds a list when the model fits the learner several times per fold."""
        slots = len(self.treatments) * self.n_rep * self.n_folds
        out: dict[str, dict[str, list[list[Any]]]] = {}
        for name, calls in self.calls.items():
            fitted = calls.fitted
            if not fitted:
                continue
            if len(fitted) % slots:
                raise RuntimeError(f"{name} was fitted {len(fitted)} times, not a multiple of {slots} folds")
            per = len(fitted) // slots
            chunks = [fitted[i * per:(i + 1) * per] for i in range(slots)]
            it = iter(c[0] if per == 1 else c for c in chunks)
            out[name] = {t: [[next(it) for _ in range(self.n_folds)] for _ in range(self.n_rep)]
                         for t in self.treatments}
        return out


def nuisance_fit(fit: Callable[..., Any]) -> Callable[..., Any]:
    """Apply the learner parameters set with ``set_ml_nuisance_params`` (or ``tune``) and record
    the fitted learners when ``store_models=True``."""
    sig = inspect.signature(fit)

    @functools.wraps(fit)
    def inner(self: Any, *args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(self, *args, **kwargs)
        store = bool(bound.arguments.get("store_models", False))
        self.__dict__.pop("models_", None)
        # A callable score refits with a built-in score, which applies the learners.
        if callable(getattr(self, "score", None)) or (not store and not self._ml_params):
            return fit(self, *args, **kwargs)
        shape = _split_shape(bound.arguments.get("folds"))
        n_rep, n_folds = shape if shape is not None else (self.n_rep, self.n_folds)
        plan = _LearnerPlan(self, _treatment_names(bound.arguments["d"]), n_rep, n_folds, store)
        with plan:
            out = fit(self, *args, **kwargs)
        if store:
            self.models_ = plan.models()
        return out

    return inner


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


@dataclass
class OptunaResult:
    """Result of tuning one nuisance learner with Optuna (DoubleML's ``DMLOptunaResult``)."""

    learner_name: str
    params_name: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_score: float
    scoring_method: Any
    study: Any
    tuned: bool


_OPTUNA_DEFAULTS: dict[str, Any] = {
    "n_trials": 100,
    "timeout": None,
    "direction": "maximize",
    "study_kwargs": {},
    "optimize_kwargs": {},
    "sampler": None,
    "callbacks": None,
    "catch": (),
    "show_progress_bar": False,
    "gc_after_trial": False,
    "study": None,
    "n_jobs_optuna": None,
    "verbosity": None,
}


class _GridSearch:
    """DoubleML's ``tune``: GridSearchCV / RandomizedSearchCV on shuffled K folds, once per
    training set (the whole sample, or each cross-fitting training fold)."""

    def __init__(self, param_grids: dict[str, Any], scoring: dict[str, Any], n_folds_tune: int,
                 n_jobs_cv: Optional[int], search_mode: str, n_iter: int) -> None:
        self.param_grids = param_grids
        self.scoring = scoring
        self.n_folds_tune = n_folds_tune
        self.n_jobs_cv = n_jobs_cv
        self.search_mode = search_mode
        self.n_iter = n_iter

    def search(self, name: str, attr: str, learner: Any, target: Any, x: Any, rows: list[Any]) -> list[Any]:
        from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV  # type: ignore[import-untyped]

        out = []
        for idx in rows:
            cv = KFold(n_splits=self.n_folds_tune, shuffle=True)
            if self.search_mode == "grid_search":
                search: Any = GridSearchCV(learner, self.param_grids[attr], scoring=self.scoring.get(attr),
                                           cv=cv, n_jobs=self.n_jobs_cv)
            else:
                search = RandomizedSearchCV(learner, self.param_grids[attr], scoring=self.scoring.get(attr),
                                            cv=cv, n_jobs=self.n_jobs_cv, n_iter=self.n_iter)
            out.append(search.fit(x[idx], target[idx]))
        return out

    @staticmethod
    def best_params(res: Any) -> dict[str, Any]:
        return res.best_params_

    def initial_predictions(self, results: list[Any], x: Any, target: Any, rows: list[Any]) -> Any:
        """In-sample predictions of the tuned learners on their training sets (DoubleML)."""
        out = np.full(len(target), np.nan)
        for res, idx in zip(results, rows):
            out[idx] = _predict(res, x[idx])
        return out


class _OptunaSearch:
    """DoubleML's ``tune_ml_models``: one Optuna study per nuisance learner on the whole sample,
    scored by cross-validation."""

    def __init__(self, spaces: dict[str, Any], scoring: Optional[dict[str, Any]], cv: Any,
                 settings: Optional[dict[str, Any]]) -> None:
        self.spaces = spaces
        self.scoring = scoring or {}
        self.cv = cv
        self.settings = settings

    def _settings(self, name: str) -> dict[str, Any]:
        user = self.settings or {}
        base = {k: v for k, v in user.items() if k in _OPTUNA_DEFAULTS}
        learner = {}
        for key in set(user) - set(base):
            if key in name and key != name:
                learner = user[key]
        specific = user.get(name, {}) if name in user else {}
        resolved = {**_OPTUNA_DEFAULTS, **base, **learner, **specific}
        if not isinstance(resolved["study_kwargs"], dict) or not isinstance(resolved["optimize_kwargs"], dict):
            raise TypeError("study_kwargs and optimize_kwargs must be dicts")
        return resolved

    def search(self, name: str, attr: str, learner: Any, target: Any, x: Any, rows: list[Any]) -> list[Any]:
        import optuna
        from sklearn.base import clone, is_classifier, is_regressor  # type: ignore[import-untyped]
        from sklearn.model_selection import cross_val_score  # type: ignore[import-untyped]

        (idx,) = rows
        x, y = x[idx], target[idx]
        scoring = self.scoring.get(name)
        if scoring is None:
            if is_regressor(learner):
                scoring = "neg_root_mean_squared_error"
            elif is_classifier(learner):
                scoring = "neg_log_loss"
            else:
                raise ValueError(f"no scoring method for {name}: pass scoring_methods={{{name!r}: ...}}")
        space = self.spaces.get(name)
        if space is None:
            est = clone(learner).fit(x, y)
            return [OptunaResult(attr, name, est, {}, float("nan"), scoring, None, False)]
        settings = self._settings(name)
        if settings["verbosity"] is not None:
            optuna.logging.set_verbosity(settings["verbosity"])
        study = settings["study"]
        if study is None:
            kwargs = dict(settings["study_kwargs"])
            kwargs.setdefault("direction", settings["direction"])
            if settings["sampler"] is not None:
                kwargs["sampler"] = settings["sampler"]
            kwargs.setdefault("study_name", f"tune_{name}")
            study = optuna.create_study(**kwargs)

        def objective(trial: Any) -> float:
            params = space(trial)
            if not isinstance(params, dict):
                raise TypeError(f"the parameter space of {name} must return a dict, got {type(params).__name__}")
            est = clone(learner).set_params(**params)
            return float(np.nanmean(cross_val_score(est, x, y, cv=self.cv, scoring=scoring, error_score="raise")))

        optimize = {
            "n_trials": settings["n_trials"], "timeout": settings["timeout"], "callbacks": settings["callbacks"],
            "catch": settings["catch"], "show_progress_bar": settings["show_progress_bar"],
            "gc_after_trial": settings["gc_after_trial"], "n_jobs": settings["n_jobs_optuna"],
            **settings["optimize_kwargs"],
        }
        optimize = {k: v for k, v in optimize.items() if v is not None or k in ("show_progress_bar", "gc_after_trial")}
        study.optimize(objective, **optimize)
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
            raise RuntimeError(f"Optuna produced no complete trial for {name}")
        best = dict(study.best_trial.params)
        return [OptunaResult(attr, name, clone(learner).set_params(**best), best, study.best_value, scoring,
                             study, True)]

    @staticmethod
    def best_params(res: Any) -> Optional[dict[str, Any]]:
        return res.best_params if res.tuned else None

    def initial_predictions(self, results: list[Any], x: Any, target: Any, rows: list[Any]) -> Any:
        """Cross-validated predictions of the tuned learners (DoubleML)."""
        from sklearn.model_selection import cross_val_predict  # type: ignore[import-untyped]

        est = results[0].best_estimator
        if _is_classifier(est):
            return cross_val_predict(est, x, target, cv=self.cv, method="predict_proba")[:, 1]
        return cross_val_predict(est, x, target, cv=self.cv)


def _predict(est: Any, x: Any) -> Any:
    if _is_classifier(est):
        return est.predict_proba(x)[:, 1]
    return est.predict(x)


def _optuna_cv(cv: Any) -> Any:
    from sklearn.model_selection import KFold  # type: ignore[import-untyped]

    if cv is None:
        cv = 5
    if isinstance(cv, (int, np.integer)):
        if cv < 2:
            raise ValueError(f"cv must be at least 2, got {cv}")
        return KFold(n_splits=int(cv), shuffle=True, random_state=42)
    if isinstance(cv, str):
        raise TypeError("cv must be an int, a scikit-learn splitter or (train, test) pairs")
    if callable(getattr(cv, "split", None)):
        return cv
    pairs = list(cv)
    if not pairs or any(len(p) != 2 for p in pairs):
        raise TypeError("cv must yield (train, test) index pairs")
    return pairs


class _Rows:
    """Training rows of the tuning step: the whole sample or the folds of one repetition."""

    def __init__(self, train: list[Any]) -> None:
        self.train = train

    def where(self, mask: Any) -> list[Any]:
        return [idx[mask[idx]] for idx in self.train]


def _plan(model: Any, a: dict[str, Any], j: int, rows: _Rows, tuner: Any) -> dict[str, list[Any]]:
    """Tune the nuisance learners of ``model`` for treatment ``j`` in DoubleML's order; returns
    ``{params name: [search result per training set]}``."""
    kind = type(model)._tuning_kind
    names = dict(model._nuisance_spec())
    x, y = a["x"][j], a["y"]
    d = a["d"][:, j]
    res: dict[str, list[Any]] = {}

    def run(name: str, target: Any, features: Any, sets: list[Any]) -> list[Any]:
        attr = names[name]
        learner = getattr(model, attr)
        if learner is None:
            learner = getattr(model, model._learner_fallback.get(attr, attr))
        res[name] = tuner.search(name, attr, learner, np.asarray(target, dtype=np.float64), features, sets)
        return res[name]

    if kind == "PLR":
        lr = run("ml_l", y, x, rows.train)
        mr = run("ml_m", d, x, rows.train)
        if "ml_g" in names:
            l_hat = tuner.initial_predictions(lr, x, y, rows.train)
            m_hat = tuner.initial_predictions(mr, x, d, rows.train)
            theta = -np.nanmean((d - m_hat) * (y - l_hat)) / np.nanmean(-(d - m_hat) ** 2)
            run("ml_g", y - theta * d, x, rows.train)
    elif kind in ("IRM", "DID"):
        run("ml_g0", y, x, rows.where(d == 0))
        run("ml_g1", y, x, rows.where(d == 1))
        if "ml_m" in names:
            run("ml_m", d, x, rows.train)
    elif kind == "DIDCS":
        t = a["t"]
        for dv in (0, 1):
            for tv in (0, 1):
                run(f"ml_g_d{dv}_t{tv}", y, x, rows.where((d == dv) & (t == tv)))
        if "ml_m" in names:
            run("ml_m", d, x, rows.train)
    elif kind == "APO":
        treated = (d == float(model.treatment_level)).astype(np.float64)
        run("ml_g_d_lvl0", y, np.column_stack((d, x)), rows.where(treated == 0))
        run("ml_g_d_lvl1", y, x, rows.where(treated == 1))
        run("ml_m", treated, x, rows.train)
    elif kind == "IIVM":
        z = a["z"].ravel()
        run("ml_g0", y, x, rows.where(z == 0))
        run("ml_g1", y, x, rows.where(z == 1))
        run("ml_m", z, x, rows.train)
        subgroups = {"always_takers": True, "never_takers": True, **(model.subgroups or {})}
        if subgroups["always_takers"]:
            run("ml_r0", d, x, rows.where(z == 0))
        if subgroups["never_takers"]:
            run("ml_r1", d, x, rows.where(z == 1))
    elif kind == "SSM":
        if model.score != "missing-at-random":
            raise NotImplementedError("tuning SSM is implemented for score='missing-at-random' (as DoubleML's Optuna tuning)")
        s = a["s"]
        run("ml_g_d0", y, x, rows.where((d == 0) & (s == 1)))
        run("ml_g_d1", y, x, rows.where((d == 1) & (s == 1)))
        run("ml_pi", s, np.column_stack((x, d)), rows.train)
        run("ml_m", d, x, rows.train)
    elif kind == "PLIV":
        z = a["z"]
        if model.partial == "Z":
            run("ml_r", d, np.hstack((x, z)), rows.train)
        elif model.partial == "X" and z.shape[1] == 1:
            z = z.ravel()
            lr = run("ml_l", y, x, rows.train)
            mr = run("ml_m", z, x, rows.train)
            rr = run("ml_r", d, x, rows.train)
            if "ml_g" in names:
                l_hat = tuner.initial_predictions(lr, x, y, rows.train)
                m_hat = tuner.initial_predictions(mr, x, z, rows.train)
                r_hat = tuner.initial_predictions(rr, x, d, rows.train)
                theta = -np.nanmean((z - m_hat) * (y - l_hat)) / np.nanmean(-(d - r_hat) * (z - m_hat))
                run("ml_g", y - theta * d, x, rows.train)
        else:
            raise NotImplementedError("tuning PLIV is implemented for partial='X' with one instrument and partial='Z'")
    else:
        raise NotImplementedError(f"tuning is not implemented for {type(model).__name__}")
    return res


class TuningMixin:
    """``set_ml_nuisance_params``, ``tune``, ``tune_ml_models`` and ``models_`` for the models with
    named nuisance learners (DoubleML's ``params_names``)."""

    #: Nuisance names other than the learner attributes reach the native fit as overrides.
    _override_learners = False
    #: Learner used when an optional learner attribute is None (e.g. PLR's ``ml_g`` -> ``ml_l``).
    _learner_fallback: dict[str, str] = {}
    _tuning_kind: Optional[str] = None
    _nuisance_overrides: Optional[dict[str, Any]] = None

    def _nuisance_spec(self) -> list[tuple[str, str]]:
        raise NotImplementedError(f"{type(self).__name__} does not support per-learner parameters")

    @property
    def _ml_params(self) -> dict[str, dict[Optional[str], list[list[dict[str, Any]]]]]:
        if "_ml_params_store" not in self.__dict__:
            self.__dict__["_ml_params_store"] = {}
        return self.__dict__["_ml_params_store"]  # type: ignore[no-any-return]

    @property
    def params_names(self) -> list[str]:
        """Names of the nuisance models (DoubleML's ``params_names``)."""
        return [name for name, _ in self._nuisance_spec()]

    @property
    def learner_names(self) -> list[str]:
        """Names of the learner arguments the nuisance models use (DoubleML's ``learner_names``)."""
        return list(dict.fromkeys(attr for _, attr in self._nuisance_spec()))

    @property
    def params(self) -> dict[str, dict[Optional[str], list[list[dict[str, Any]]]]]:
        """Parameters set per nuisance model and treatment: ``{name: {treatment: [[params per fold]
        per repetition]}}`` (treatment ``None``: every treatment)."""
        return {name: dict(self._ml_params.get(name, {})) for name in self.params_names}

    def set_ml_nuisance_params(self, learner: str, treat_var: Optional[str], params: Any) -> Any:
        """Set hyperparameters of one nuisance model, as DoubleML.

        ``learner`` is a name from `params_names` (e.g. ``"ml_g1"``), ``treat_var`` a treatment name
        (``None``: every treatment), and ``params`` a dict used in every fold, or a nested list with
        one dict per fold for each repetition (``n_rep`` x ``n_folds``). New values update earlier
        ones. Learners need the scikit-learn API (``set_params``). Fold-specific parameters need a
        model that fits the learner once per fold; folds are then fitted one after the other.
        """
        if learner not in self.params_names:
            raise ValueError(f"invalid nuisance learner {learner!r}; valid: {', '.join(self.params_names)}")
        if treat_var is not None and not isinstance(treat_var, str):
            raise TypeError("treat_var must be a treatment name or None")
        if params is None:
            return self
        n_rep, n_folds = self.n_rep, self.n_folds  # type: ignore[attr-defined]
        if isinstance(params, dict):
            new = [[dict(params) for _ in range(n_folds)] for _ in range(n_rep)]
        else:
            params = list(params)
            if len(params) != n_rep or any(len(list(rep)) != n_folds for rep in params):
                raise ValueError(f"fold-specific params need n_rep={n_rep} lists of n_folds={n_folds} dicts")
            new = [[dict(p or {}) for p in rep] for rep in params]
        store = self._ml_params.setdefault(learner, {})
        old = store.get(treat_var)
        store[treat_var] = new if old is None else [[{**o, **n} for o, n in zip(orep, nrep)]
                                                   for orep, nrep in zip(old, new)]
        return self

    # -- tuning -------------------------------------------------------------
    def _tuning_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[dict[str, Any], Any, Any]:
        from .data import DmlData

        fit = type(self).fit  # type: ignore[attr-defined]
        if args and isinstance(args[0], DmlData):
            roles, cluster = fit.__dml_roles__
            data = args[0]
            args = tuple(data._role(role, type(self).__name__) for role in roles)
            if cluster and data.cluster is not None:
                kwargs.setdefault("cluster", data.cluster)
        bound = inspect.signature(fit).bind(self, *args, **kwargs)
        arg = bound.arguments
        if arg.get("cluster") is not None:
            raise NotImplementedError("tuning with clustered data is not implemented")
        d = np.asarray(arg["d"], dtype=np.float64)
        d = d.reshape(-1, 1) if d.ndim == 1 else d
        X = np.asarray(arg["X"], dtype=np.float64)
        k = d.shape[1]
        xs = [X if k == 1 else np.hstack((X, d[:, [o for o in range(k) if o != j]])) for j in range(k)]
        a: dict[str, Any] = {"x": xs, "y": np.asarray(arg["y"], dtype=np.float64), "d": d}
        for key in ("z", "t", "s"):
            if arg.get(key) is not None:
                v = np.asarray(arg[key], dtype=np.float64)
                a[key] = v.reshape(-1, 1) if key == "z" and v.ndim == 1 else v
        return a, arg["d"], arg.get("folds")

    def _strata(self, a: dict[str, Any]) -> Optional[list[int]]:
        return None

    def _apply_tuning(self, results: list[dict[str, list[Any]]], treat: str, tuner: Any, n_rep: int,
                      per_fold: bool) -> None:
        for name in results[0]:
            if per_fold:
                nested = [[tuner.best_params(r) for r in res[name]] for res in results]
                if all(p is None for rep in nested for p in rep):
                    continue
                self.set_ml_nuisance_params(name, treat, [[p or {} for p in rep] for rep in nested])
            else:
                best = tuner.best_params(results[0][name][0])
                if best is not None:
                    self.set_ml_nuisance_params(name, treat, best)

    def tune(
        self,
        *args: Any,
        param_grids: dict[str, Any],
        tune_on_folds: bool = False,
        scoring_methods: Optional[dict[str, Any]] = None,
        n_folds_tune: int = 5,
        search_mode: str = "grid_search",
        n_iter_randomized_search: int = 100,
        n_jobs_cv: Optional[int] = None,
        set_as_params: bool = True,
        return_tune_res: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Tune the nuisance learners by grid or randomized search, as DoubleML's ``tune``.

        Takes the arguments of `fit` (the data, and optionally ``folds``) plus DoubleML's tuning
        arguments: ``param_grids`` maps every name in `learner_names` (e.g. ``ml_g``, ``ml_m``) to a
        scikit-learn parameter grid; ``scoring_methods`` optionally maps learner names to scorers.
        Each learner is tuned on the training data of its nuisance model (e.g. ``ml_g`` for
        ``ml_g0`` on the untreated) with scikit-learn's ``GridSearchCV`` (or ``RandomizedSearchCV``)
        on ``n_folds_tune`` shuffled folds drawn from NumPy's global random state, as DoubleML.
        With ``tune_on_folds=True`` the search runs on every cross-fitting training fold (the
        folds `fit` draws, or ``folds``) and the parameters are fold-specific.

        The best parameters are set with `set_ml_nuisance_params` (``set_as_params``); returns
        ``self``, or the search results per treatment with ``return_tune_res=True``.
        """
        if not isinstance(param_grids, dict) or not all(k in param_grids for k in self.learner_names):
            raise ValueError(f"param_grids must be a dict with the keys {', '.join(self.learner_names)}")
        if scoring_methods is not None and (not isinstance(scoring_methods, dict)
                                            or not all(k in self.learner_names for k in scoring_methods)):
            raise ValueError(f"scoring_methods must be a dict with keys among {', '.join(self.learner_names)}")
        if not isinstance(n_folds_tune, int) or n_folds_tune < 2:
            raise ValueError("n_folds_tune must be an int of at least 2")
        if search_mode not in ("grid_search", "randomized_search"):
            raise ValueError('search_mode must be "grid_search" or "randomized_search"')
        if not isinstance(n_iter_randomized_search, int) or n_iter_randomized_search < 2:
            raise ValueError("n_iter_randomized_search must be an int of at least 2")
        a, d, folds = self._tuning_inputs(args, kwargs)
        tuner = _GridSearch(param_grids, scoring_methods or {}, n_folds_tune, n_jobs_cv, search_mode,
                            n_iter_randomized_search)
        n = len(a["y"])
        if tune_on_folds:
            if folds is None:
                reps: Any = _dml_rs.draw_folds(n, self.n_folds, self.n_rep, self.random_state,  # type: ignore[attr-defined]
                                               self._strata(a))
            else:
                from ._models import _splits

                reps = [[(np.asarray(tr), np.asarray(te)) for tr, te in rep] for rep in _splits(folds)]  # type: ignore[union-attr]
        out = []
        for j, treat in enumerate(_treatment_names(d)):
            if tune_on_folds:
                results = [_plan(self, a, j, _Rows([np.asarray(tr, dtype=np.int64) for tr, _ in rep]), tuner)
                           for rep in reps]
            else:
                results = [_plan(self, a, j, _Rows([np.arange(n)]), tuner)]
            if set_as_params:
                self._apply_tuning(results, treat, tuner, len(results), tune_on_folds)
            out.append([{"params": {k: [tuner.best_params(r) for r in v] for k, v in res.items()},
                         "tune_res": res} for res in results] if tune_on_folds else
                       {"params": {k: [tuner.best_params(r) for r in v] for k, v in results[0].items()},
                        "tune_res": results[0]})
        return out if return_tune_res else self

    def tune_ml_models(
        self,
        *args: Any,
        ml_param_space: dict[str, Callable[[Any], dict[str, Any]]],
        scoring_methods: Optional[dict[str, Any]] = None,
        cv: Any = 5,
        set_as_params: bool = True,
        return_tune_res: bool = False,
        optuna_settings: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Tune the nuisance learners with Optuna, as DoubleML's ``tune_ml_models`` (needs ``optuna``).

        Takes the arguments of `fit` plus DoubleML's tuning arguments. ``ml_param_space`` maps
        learner names (`learner_names`, applied to every nuisance model of that learner) or
        nuisance names (`params_names`, taking precedence) to functions ``trial -> dict`` of
        parameters. Each nuisance learner is tuned once on its training data in the whole sample
        by cross-validation (``cv``: folds, a splitter or ``(train, test)`` pairs; an int means
        shuffled K-fold with ``random_state=42``), maximising ``scoring_methods[name]`` (default:
        negative RMSE for regressors, negative log loss for classifiers). ``optuna_settings`` as
        DoubleML (``n_trials`` default 100, ``sampler``, ``study_kwargs``, ``optimize_kwargs``, ...;
        also per learner or nuisance name). Returns ``self``, or per treatment ``{name:
        OptunaResult}`` with ``return_tune_res=True``.
        """
        try:
            import optuna  # noqa: F401
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError("tune_ml_models needs optuna: pip install optuna") from err
        if not isinstance(ml_param_space, dict) or not ml_param_space:
            raise ValueError("ml_param_space must be a non-empty dict")
        allowed = set(self.params_names) | set(self.learner_names)
        bad = sorted(k for k in ml_param_space if k not in allowed)
        if bad:
            raise ValueError(f"invalid ml_param_space keys {bad}; valid keys: {', '.join(sorted(allowed))}")
        for key, fn in ml_param_space.items():
            if fn is not None and not callable(fn):
                raise TypeError(f"the parameter space for {key!r} must be a function trial -> dict")
        spaces: dict[str, Any] = {name: None for name in self.params_names}
        for learner in [ln for ln in self.learner_names if ln in ml_param_space]:
            for name in [pn for pn in self.params_names if learner in pn]:
                spaces[name] = ml_param_space[learner]
        for name in [pn for pn in self.params_names if pn in ml_param_space]:
            spaces[name] = ml_param_space[name]
        if scoring_methods is not None:
            if not isinstance(scoring_methods, dict):
                raise TypeError("scoring_methods must be a dict keyed by nuisance name")
            bad = sorted(k for k in scoring_methods if k not in self.params_names)
            if bad:
                raise ValueError(f"invalid scoring_methods keys {bad}; valid: {', '.join(self.params_names)}")
        if optuna_settings is not None:
            if not isinstance(optuna_settings, dict):
                raise TypeError("optuna_settings must be a dict or None")
            bad = sorted(k for k in optuna_settings if k not in _OPTUNA_DEFAULTS and k not in allowed)
            if bad:
                raise ValueError(f"invalid optuna_settings keys {bad}")
            for key in allowed & set(optuna_settings):
                if not isinstance(optuna_settings[key], dict):
                    raise TypeError(f"Optuna settings for {key!r} must be a dict")
        a, d, _ = self._tuning_inputs(args, kwargs)
        tuner = _OptunaSearch(spaces, scoring_methods, _optuna_cv(cv), optuna_settings)
        out = []
        for j, treat in enumerate(_treatment_names(d)):
            res = _plan(self, a, j, _Rows([np.arange(len(a["y"]))]), tuner)
            if set_as_params:
                self._apply_tuning([res], treat, tuner, 1, False)
            out.append({name: r[0] for name, r in res.items()})
        return out if return_tune_res else self


# ---------------------------------------------------------------------------
# Callable scores
# ---------------------------------------------------------------------------


def fit_callable_score(
    model: Any,
    builtin_score: str,
    call: Callable[[int, dict[str, Any], list[tuple[Any, Any]]], tuple[Any, Any]],
    refit: Callable[[Any], Any],
    folds: Any,
    cluster: Any,
    n: int,
) -> Any:
    """Fit with a callable ``score(...) -> (psi_a, psi_b)``, as DoubleML: the nuisance learners are
    cross-fitted as for ``builtin_score`` and ``call(j, predictions, smpls)`` gives the linear score
    elements of treatment ``j`` for each repetition."""
    if cluster is not None:
        raise NotImplementedError("callable scores with clustered data are not implemented")
    from ._models import _splits

    if folds is None:
        reps = [[(np.asarray(tr), np.asarray(te)) for tr, te in rep]
                for rep in _dml_rs.draw_folds(n, model.n_folds, model.n_rep, model.random_state)]
    else:
        reps = [[(np.asarray(tr), np.asarray(te)) for tr, te in rep] for rep in _splits(folds)]  # type: ignore[union-attr]
    score = model.score
    model.score = builtin_score
    try:
        refit(reps)
    finally:
        model.score = score
    builtin = model.result_
    preds = model.predictions_
    names = builtin.treatment_names
    psi_a = np.zeros((n, len(names), len(reps)))
    psi_b = np.zeros_like(psi_a)
    for j in range(len(names)):
        for r, smpls in enumerate(reps):
            p = {name: v[:, r, j] for name, v in preds.items()}
            a_, b_ = call(j, p, smpls)
            psi_a[:, j, r] = np.asarray(a_, dtype=np.float64).reshape(-1)
            psi_b[:, j, r] = np.asarray(b_, dtype=np.float64).reshape(-1)
    loss = model.nuisance_loss_
    model._store(_dml_rs.result_from_scores(psi_a, psi_b, list(names), len(reps[0]), type(model).__name__))
    model.predictions_ = preds
    model.nuisance_loss_ = loss
    return model


def normalize_ipw(m_hat: Any, d: Any) -> Any:
    """DoubleML's normalised propensity (``_normalize_ipw``)."""
    mean1 = np.mean(d / m_hat)
    mean0 = np.mean((1.0 - d) / (1.0 - m_hat))
    return d * (m_hat * mean1) + (1.0 - d) * (1.0 - (1.0 - m_hat) * mean0)


__all__ = ["OptunaResult", "TuningMixin", "fit_callable_score", "normalize_ipw", "nuisance_fit"]

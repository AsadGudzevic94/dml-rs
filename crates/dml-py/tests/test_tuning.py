"""Per-learner parameters, tuning, stored learners and callable scores against DoubleML."""

import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.did.datasets import make_did_SZ2020
from doubleml.irm.datasets import make_iivm_data, make_irm_data, make_irm_data_discrete_treatments, make_ssm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plr_CCDDHNR2018
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
GRID_G = {"alpha": [0.005, 0.05, 0.3]}
GRID_M = {"C": [0.05, 1.0]}


def folds(n, n_folds=3, n_rep=1):
    return [list(KFold(n_folds, shuffle=True, random_state=r).split(np.zeros(n))) for r in range(n_rep)]


def doubleml(cls, data, reps, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(data, n_folds=len(reps[0]), n_rep=len(reps), draw_sample_splitting=False, **kwargs)
        model.set_sample_splitting(reps)
        return model


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def assert_same_fit(ours, ref):
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL, atol=1e-12)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL, atol=1e-12)


def case(name, n=500, n_folds=3, n_rep=1):
    """(DoubleML model, our model, fit arguments, regressor and classifier learner names)."""
    np.random.seed(11)
    if name in ("PLR", "PLR-IV"):
        x, y, d = make_plr_CCDDHNR2018(n_obs=n, dim_x=6, return_type="array")
        kw = {"score": "IV-type"} if name == "PLR-IV" else {}
        extra = {"ml_g": Lasso()} if name == "PLR-IV" else {}
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d), reps,
                       ml_l=Lasso(), ml_m=Lasso(), **extra, **kw)
        ours = dml_rs.PLR(Lasso(), Lasso(), n_folds=n_folds, n_rep=n_rep, **extra, **kw)
        grids = {"ml_l": GRID_G, "ml_m": GRID_G, **({"ml_g": GRID_G} if extra else {})}
        return ref, ours, (x, y, d), reps, grids
    if name == "IRM":
        x, y, d = make_irm_data(n_obs=n, dim_x=5, theta=0.5, return_type="array")
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLIRM, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_g=Lasso(), ml_m=LOGIT)
        return ref, dml_rs.IRM(Lasso(), LOGIT, n_folds=n_folds, n_rep=n_rep), (x, y, d), reps, \
            {"ml_g": GRID_G, "ml_m": GRID_M}
    if name == "APO":
        r = make_irm_data_discrete_treatments(n_obs=n, n_levels=3, linear=True, random_state=41)
        x, y, d = r["x"], r["y"], r["d"]
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLAPO, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_g=Lasso(), ml_m=LOGIT,
                       treatment_level=1)
        return ref, dml_rs.APO(Lasso(), LOGIT, treatment_level=1, n_folds=n_folds, n_rep=n_rep), (x, y, d), reps, \
            {"ml_g": GRID_G, "ml_m": GRID_M}
    if name == "IIVM":
        x, y, d, z = make_iivm_data(n_obs=n, dim_x=5, theta=0.5, alpha_x=1.0, return_type="array")
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLIIVM, dml.DoubleMLData.from_arrays(x, y, d, z=z), reps, ml_g=Lasso(), ml_m=LOGIT,
                       ml_r=LOGIT)
        return ref, dml_rs.IIVM(Lasso(), LOGIT, LOGIT, n_folds=n_folds, n_rep=n_rep), (x, y, d, z), reps, \
            {"ml_g": GRID_G, "ml_m": GRID_M, "ml_r": GRID_M}
    if name == "DID":
        x, y, d, _ = make_did_SZ2020(n_obs=n, cross_sectional_data=False, return_type="array")
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLDID, dml.DoubleMLDIDData.from_arrays(x, y, d), reps, ml_g=Lasso(), ml_m=LOGIT)
        return ref, dml_rs.DID(Lasso(), LOGIT, n_folds=n_folds, n_rep=n_rep), (x, y, d), reps, \
            {"ml_g": GRID_G, "ml_m": GRID_M}
    if name == "DIDCS":
        x, y, d, t = make_did_SZ2020(n_obs=n, cross_sectional_data=True, return_type="array")
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLDIDCS, dml.DoubleMLDIDData.from_arrays(x, y, d, t=t), reps, ml_g=Lasso(),
                       ml_m=LOGIT)
        return ref, dml_rs.DIDCS(Lasso(), LOGIT, n_folds=n_folds, n_rep=n_rep), (x, y, d, t), reps, \
            {"ml_g": GRID_G, "ml_m": GRID_M}
    if name == "SSM":
        x, y, d, _, s = make_ssm_data(n_obs=n, dim_x=5, theta=1.0, mar=True, return_type="array")
        reps = folds(n, n_folds, n_rep)
        ref = doubleml(dml.DoubleMLSSM, dml.DoubleMLSSMData.from_arrays(x, y, d, s=s), reps, ml_g=Lasso(),
                       ml_pi=LOGIT, ml_m=LOGIT)
        return ref, dml_rs.SSM(Lasso(), LOGIT, LOGIT, n_folds=n_folds, n_rep=n_rep), (x, y, d, s), reps, \
            {"ml_g": GRID_G, "ml_pi": GRID_M, "ml_m": GRID_M}
    if name in ("PLIV", "PLIV-IV", "PLIV-Z"):
        x, y, d, z = make_pliv_CHS2015(n_obs=n, alpha=1.0, dim_x=8, dim_z=1, return_type="array")
        z = z.ravel()
        reps = folds(n, n_folds, n_rep)
        kw = {"score": "IV-type", "ml_g": Lasso()} if name == "PLIV-IV" else {}
        part = {"partialX": False, "partialZ": True} if name == "PLIV-Z" else {}
        ref = quiet(doubleml, dml.DoubleMLPLIV._partialZ if part else dml.DoubleMLPLIV,
                    dml.DoubleMLData.from_arrays(x, y, d, z=z), reps,
                    **({"ml_r": Lasso()} if part else {"ml_l": Lasso(), "ml_m": Lasso(), "ml_r": Lasso(), **kw}))
        ours = dml_rs.PLIV(Lasso(), Lasso(), Lasso(), partial="Z" if part else "X", n_folds=n_folds, n_rep=n_rep,
                           **kw)
        grids = {"ml_r": GRID_G} if part else {"ml_l": GRID_G, "ml_m": GRID_G, "ml_r": GRID_G,
                                               **({"ml_g": GRID_G} if kw else {})}
        return ref, ours, (x, y, d, z), reps, grids
    raise ValueError(name)


def tuned_names(ref):
    return [k for k, v in ref.params.items() if any(p is not None for p in v.values())]


MODELS = ["PLR", "PLR-IV", "IRM", "APO", "IIVM", "DID", "DIDCS", "SSM", "PLIV", "PLIV-IV", "PLIV-Z"]


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("tune_on_folds", [False, True])
def test_tune_matches_doubleml(name, tune_on_folds):
    # n_rep=1: DoubleML 0.11.4 tune(tune_on_folds=True) fails for n_rep > n_treat (results indexed [rep][treat]).
    ref, ours, args, reps, grids = case(name)
    np.random.seed(5)
    quiet(ref.tune, grids, tune_on_folds=tune_on_folds, n_folds_tune=3)
    quiet(ref.fit)
    np.random.seed(5)
    ours.tune(*args, param_grids=grids, tune_on_folds=tune_on_folds, n_folds_tune=3, folds=reps)
    for k in tuned_names(ref):
        assert ours.params[k] == ref.params[k], k
    assert set(tuned_names(ref)) == {k for k, v in ours.params.items() if v}
    ours.fit(*args, folds=reps)
    assert_same_fit(ours, ref)


def test_randomized_search_matches_doubleml():
    ref, ours, args, reps, _ = case("IRM")
    grids = {"ml_g": {"alpha": list(np.linspace(0.001, 0.5, 20))}, "ml_m": {"C": list(np.logspace(-2, 1, 10))}}
    np.random.seed(8)
    quiet(ref.tune, grids, search_mode="randomized_search", n_iter_randomized_search=4, n_folds_tune=3)
    quiet(ref.fit)
    np.random.seed(8)
    ours.tune(*args, param_grids=grids, search_mode="randomized_search", n_iter_randomized_search=4, n_folds_tune=3)
    for k in tuned_names(ref):
        assert ours.params[k] == ref.params[k]
    assert_same_fit(ours.fit(*args, folds=reps), ref)


def test_tune_returns_search_results():
    _, ours, args, _, grids = case("IRM")
    res = ours.tune(*args, param_grids=grids, return_tune_res=True, set_as_params=False)
    assert set(res[0]["params"]) == {"ml_g0", "ml_g1", "ml_m"}
    assert res[0]["tune_res"]["ml_m"][0].best_params_ == res[0]["params"]["ml_m"][0]
    assert ours.params == {"ml_g0": {}, "ml_g1": {}, "ml_m": {}}
    with pytest.raises(ValueError, match="param_grids"):
        ours.tune(*args, param_grids={"ml_g": GRID_G})


@pytest.mark.parametrize("name", ["PLR-IV", "IRM", "IIVM", "DIDCS"])
def test_tune_ml_models_matches_doubleml(name):
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ref, ours, args, reps, _ = case(name)

    def reg(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 0.5, log=True)}

    def clf(trial):
        return {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}

    space = {n: (clf if n in ("ml_m", "ml_r") and name != "PLR-IV" else reg) for n in ours.learner_names}
    space["ml_g1"] = reg  # a nuisance-specific space wins over the learner's
    if "ml_g1" not in ours.params_names:
        del space["ml_g1"]
    res_ref = quiet(ref.tune_ml_models, space, optuna_settings={"n_trials": 6, "sampler": optuna.samplers.TPESampler(seed=2)},
                    return_tune_res=True)
    quiet(ref.fit)
    res = ours.tune_ml_models(*args, ml_param_space=space, return_tune_res=True,
                              optuna_settings={"n_trials": 6, "sampler": optuna.samplers.TPESampler(seed=2)})
    for k, r in res_ref[0].items():
        if r is None:
            continue
        assert res[0][k].best_params == r.best_params, k
        np.testing.assert_allclose(res[0][k].best_score, r.best_score, rtol=1e-12)
    assert_same_fit(ours.fit(*args, folds=reps), ref)


@pytest.mark.parametrize("name", ["PLR", "IRM", "APO", "IIVM", "DID", "DIDCS", "SSM", "PLIV-IV"])
def test_set_ml_nuisance_params_matches_doubleml(name):
    ref, ours, args, reps, _ = case(name, n_rep=2)
    nuisance = ours.params_names[-2]
    learner = dict(ours._nuisance_spec())[nuisance]
    params = {"C": 0.2} if learner in ("ml_m", "ml_r", "ml_pi") and name not in ("PLR", "PLIV-IV") else {"alpha": 0.07}
    ref.set_ml_nuisance_params(nuisance, "d", params)
    quiet(ref.fit)
    ours.set_ml_nuisance_params(nuisance, "d", params).fit(*args, folds=reps)
    assert_same_fit(ours, ref)


def test_fold_specific_and_treatment_specific_params():
    np.random.seed(3)
    n = 400
    x = np.random.normal(size=(n, 4))
    d = x[:, :2] + np.random.normal(size=(n, 2))
    y = d @ np.array([0.5, -1.0]) + x[:, 2] + np.random.normal(size=n)
    reps = folds(n, 3, 2)
    fold_params = [[{"alpha": 0.01}, {"alpha": 0.2}, {"alpha": 0.05}], [{"alpha": 0.1}] * 3]
    ref = doubleml(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_l=Lasso(), ml_m=Lasso())
    ref.set_ml_nuisance_params("ml_l", "d2", fold_params)
    ref.set_ml_nuisance_params("ml_m", "d1", {"alpha": 0.3})
    quiet(ref.fit)
    ours = dml_rs.PLR(Lasso(), Lasso(), n_folds=3, n_rep=2)
    ours.set_ml_nuisance_params("ml_l", "d2", fold_params)
    ours.set_ml_nuisance_params("ml_m", "d1", {"alpha": 0.3})
    import pandas as pd

    ours.fit(x, y, pd.DataFrame(d, columns=["d1", "d2"]), folds=reps)
    assert_same_fit(ours, ref)
    # Later values update earlier ones fold by fold, as DoubleML.
    ours.set_ml_nuisance_params("ml_l", "d2", {"max_iter": 500})
    assert ours.params["ml_l"]["d2"][0][1] == {"alpha": 0.2, "max_iter": 500}
    with pytest.raises(ValueError, match="unknown treatment"):
        ours.fit(x, y, d, folds=reps)


def test_params_errors():
    model = dml_rs.IRM(Lasso(), LOGIT, n_folds=3)
    with pytest.raises(ValueError, match="invalid nuisance learner"):
        model.set_ml_nuisance_params("ml_g", "d", {"alpha": 1.0})
    with pytest.raises(ValueError, match="n_rep=1 lists"):
        model.set_ml_nuisance_params("ml_g0", "d", [[{}], [{}]])
    x, y, d = make_irm_data(n_obs=200, dim_x=3, return_type="array")
    native = dml_rs.IRM(dml_rs.Lasso(), LOGIT, n_folds=3).set_ml_nuisance_params("ml_g0", None, {"alpha": 0.1})
    with pytest.raises(TypeError, match="set_params"):
        native.fit(x, y, d)
    lplr = dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression(), n_folds=3, n_folds_inner=2)
    lplr.set_ml_nuisance_params("ml_M", None, [[{"C": 1.0}, {"C": 2.0}, {"C": 3.0}]])
    yb = (y > np.median(y)).astype(float)
    with pytest.raises(Exception, match="once per fold"):
        lplr.fit(x, yb, d)


@pytest.mark.parametrize("name", ["PLR-IV", "IRM", "IIVM", "DIDCS", "SSM", "PLIV"])
def test_store_models_matches_doubleml(name):
    ref, ours, args, reps, _ = case(name, n_rep=2)
    quiet(ref.fit, store_models=True)
    ours.fit(*args, folds=reps, store_models=True)
    x = args[0]
    for learner, by_treat in ref.models.items():
        mine = ours.models_[learner]["d"]
        for r, rep in enumerate(reps):
            for k, (_, test) in enumerate(rep):
                theirs = by_treat["d"][r][k]
                assert type(mine[r][k]) is type(theirs)
                np.testing.assert_allclose(getattr(mine[r][k], "coef_"), theirs.coef_, rtol=1e-12, atol=1e-14)
    assert_same_fit(ours, ref)
    assert set(ours.models_) == set(ref.models)


def test_store_models_with_native_learners_and_lplr():
    x, y, d = make_irm_data(n_obs=300, dim_x=4, return_type="array")
    model = dml_rs.IRM(dml_rs.Lasso(alpha=0.05), dml_rs.LogisticRegression(), n_folds=3).fit(x, y, d, store_models=True)
    plain = dml_rs.IRM(dml_rs.Lasso(alpha=0.05), dml_rs.LogisticRegression(), n_folds=3).fit(x, y, d)
    np.testing.assert_allclose(model.coef_, plain.coef_, rtol=1e-12)
    splits = dml_rs._dml_rs.draw_folds(len(y), 3, 1, 0)
    for k, (_, test) in enumerate(splits[0]):
        np.testing.assert_allclose(model.models_["ml_g1"]["d"][0][k].predict(x[test]),
                                   model.predictions_["ml_g1"][test, 0, 0], rtol=1e-12)
    fitted = model.models_["ml_m"]["d"][0]
    assert len({id(m) for m in fitted}) == 3
    yb = (y > np.median(y)).astype(float)
    lplr = dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression(), n_folds=3, n_folds_inner=2)
    lplr.fit(x, yb, d, store_models=True)
    assert len(lplr.models_["ml_t"]["d"][0]) == 3
    assert isinstance(lplr.models_["ml_M"]["d"][0][0], list)


def irm_score(y, d, g_hat0, g_hat1, m_hat, smpls):
    psi_b = g_hat1 - g_hat0 + d * (y - g_hat1) / m_hat - (1 - d) * (y - g_hat0) / (1 - m_hat)
    return -np.ones_like(y), psi_b


def plr_score(y, d, l_hat, m_hat, g_hat, smpls):
    v = d - m_hat
    return -v * d, v * (y - g_hat)


def iivm_score(y, z, d, g_hat0, g_hat1, m_hat, r_hat0, r_hat1, smpls):
    psi_b = g_hat1 - g_hat0 + z * (y - g_hat1) / m_hat - (1 - z) * (y - g_hat0) / (1 - m_hat)
    psi_a = -(r_hat1 - r_hat0 + z * (d - r_hat1) / m_hat - (1 - z) * (d - r_hat0) / (1 - m_hat))
    return psi_a, psi_b


def pliv_score(y, z, d, l_hat, m_hat, r_hat, g_hat, smpls):
    return -(d - r_hat) * (z - m_hat), (z - m_hat) * (y - l_hat)


@pytest.mark.parametrize("name, score, normalize", [
    ("PLR-IV", plr_score, None), ("IRM", irm_score, False), ("IRM", irm_score, True),
    ("IIVM", iivm_score, False), ("PLIV", pliv_score, None)])
def test_callable_scores_match_doubleml(name, score, normalize):
    _, ours, args, reps, _ = case(name, n_rep=2)
    ref_cls, data_args = {
        "PLR-IV": (dml.DoubleMLPLR, {"ml_l": Lasso(), "ml_m": Lasso(), "ml_g": Lasso()}),
        "IRM": (dml.DoubleMLIRM, {"ml_g": Lasso(), "ml_m": LOGIT}),
        "IIVM": (dml.DoubleMLIIVM, {"ml_g": Lasso(), "ml_m": LOGIT, "ml_r": LOGIT}),
        "PLIV": (dml.DoubleMLPLIV, {"ml_l": Lasso(), "ml_m": Lasso(), "ml_r": Lasso()}),
    }[name]
    x, y, d, *rest = args
    data = dml.DoubleMLData.from_arrays(x, y, d, z=rest[0] if rest else None)
    extra = {} if normalize is None else {"normalize_ipw": normalize}
    ref = doubleml(ref_cls, data, reps, score=score, **data_args, **extra)
    quiet(ref.fit)
    ours.set_params(score=score, **extra)
    ours.fit(*args, folds=reps)
    assert_same_fit(ours, ref)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    assert set(ours.predictions_) >= {"ml_m"} | ({"ml_g"} if name == "PLR-IV" else set())
    # Default folds are the model's own.
    ours.fit(*args)
    assert ours.all_coef_.shape == (1, 2)

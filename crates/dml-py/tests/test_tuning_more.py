"""Tuning of LPLR, PLPR, nonignorable SSM and PLIV with several instruments against DoubleML."""

import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_ssm_data
from doubleml.plm.datasets import make_lplr_LZZ2020, make_pliv_CHS2015, make_plpr_CP2025
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
GRID_LASSO = {"alpha": [0.005, 0.05, 0.3]}
GRID_RIDGE = {"alpha": [0.1, 10.0, 1000.0]}
GRID_LOGIT = {"C": [0.05, 1.0]}


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def folds(n, n_folds=3):
    return [list(KFold(n_folds, shuffle=True, random_state=0).split(np.zeros(n)))]


def with_folds(cls, data, reps, **kwargs):
    model = quiet(cls, data, n_folds=len(reps[0]), n_rep=len(reps), draw_sample_splitting=False, **kwargs)
    model.set_sample_splitting(reps)
    return model


def compare_params(ours, ref, names):
    for name in names:
        assert ours.params[name] == ref.params[name], name


def assert_same_fit(ours, ref):
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


def pliv_case(n=400, n_instruments=3):
    np.random.seed(11)
    x, y, d, z = make_pliv_CHS2015(n_obs=n, alpha=1.0, dim_x=8, dim_z=n_instruments, return_type="array")
    reps = folds(n)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    ref = with_folds(dml.DoubleMLPLIV, data, reps, ml_l=Lasso(), ml_m=Lasso(), ml_r=Lasso())
    return ref, dml_rs.PLIV(Lasso(), Lasso(), Lasso(), n_folds=3), (x, y, d, z), reps


@pytest.mark.parametrize("tune_on_folds", [False, True])
def test_tune_pliv_with_several_instruments(tune_on_folds):
    ref, ours, args, reps = pliv_case()
    grids = {"ml_l": GRID_LASSO, "ml_m": GRID_LASSO, "ml_r": GRID_LASSO}
    np.random.seed(5)
    quiet(ref.tune, grids, tune_on_folds=tune_on_folds, n_folds_tune=3)
    quiet(ref.fit)
    np.random.seed(5)
    ours.tune(*args, param_grids=grids, tune_on_folds=tune_on_folds, n_folds_tune=3, folds=reps)
    assert set(ours.params_names) == set(ref.params_names)
    compare_params(ours, ref, ref.params_names)
    assert_same_fit(ours.fit(*args, folds=reps), ref)


def test_tune_ml_models_pliv_with_several_instruments():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ref, ours, args, reps = pliv_case()

    def space(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 0.5, log=True)}

    spaces = {"ml_l": space, "ml_m": space, "ml_r": space}
    settings = {"n_trials": 5, "sampler": optuna.samplers.TPESampler(seed=2)}
    quiet(ref.tune_ml_models, spaces, optuna_settings=settings)
    quiet(ref.fit)
    settings = {"n_trials": 5, "sampler": optuna.samplers.TPESampler(seed=2)}
    ours.tune_ml_models(*args, ml_param_space=spaces, optuna_settings=settings)
    assert_same_fit(ours.fit(*args, folds=reps), ref)


def test_pliv_instrument_models_are_stored_and_ml_m_covers_them():
    ref, ours, args, reps = pliv_case(n_instruments=2)
    ours.fit(*args, folds=reps, store_models=True)
    assert set(ours.models_) >= {"ml_l", "ml_m_z1", "ml_m_z2", "ml_r"}
    fresh = dml_rs.PLIV(Lasso(), Lasso(), Lasso(), n_folds=3)
    fresh.set_ml_nuisance_params("ml_m", None, {"alpha": 0.01})
    other = dml_rs.PLIV(Lasso(), Lasso(alpha=0.01), Lasso(), n_folds=3)
    np.testing.assert_allclose(fresh.fit(*args, folds=reps).coef_, other.fit(*args, folds=reps).coef_, rtol=TOL)


def lplr_case(score, n=500):
    np.random.seed(3)
    df = make_lplr_LZZ2020(n_obs=n, dim_x=20, alpha=0.5, return_type="DataFrame", treatment="binary")
    x_cols = [c for c in df.columns if c.startswith("X")]
    x, y, d = df[x_cols].to_numpy(), df["y"].to_numpy(), df["d"].to_numpy()
    data = dml.DoubleMLData(df, "y", "d", x_cols=x_cols)
    logit = LogisticRegression(solver="newton-cholesky", tol=1e-12, max_iter=1000)
    return data, x, y, d, logit


@pytest.mark.parametrize("score", ["nuisance_space", "instrument"])
def test_tune_lplr(score):
    data, x, y, d, logit = lplr_case(score)
    ref = quiet(dml.DoubleMLLPLR, data, logit, Ridge(), logit, score=score, n_folds=3)
    grids = {"ml_M": GRID_LOGIT, "ml_t": GRID_RIDGE, "ml_m": GRID_LOGIT, "ml_a": GRID_LOGIT}
    np.random.seed(5)
    quiet(ref.tune, grids, n_folds_tune=3)
    ours = dml_rs.LPLR(logit, Ridge(), logit, logit, score=score, n_folds=3)
    np.random.seed(5)
    ours.tune(x, y, d, param_grids=grids, n_folds_tune=3)
    for name in ("ml_M", "ml_t", "ml_m", "ml_a"):
        assert ours.params[name]["d"][0][0] == ref.params[name]["d"][0][0], name
    assert np.isfinite(ours.fit(x, y, d).coef_).all()
    with pytest.raises(NotImplementedError, match="tune_on_folds"):
        ours.tune(x, y, d, param_grids=grids, tune_on_folds=True)


def test_tune_ml_models_lplr():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data, x, y, d, logit = lplr_case("nuisance_space")
    ref = quiet(dml.DoubleMLLPLR, data, logit, Ridge(), logit, n_folds=3)

    def reg(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}

    def clf(trial):
        return {"C": trial.suggest_float("C", 1e-2, 10.0, log=True)}

    spaces = {"ml_M": clf, "ml_t": reg, "ml_m": clf, "ml_a": clf}
    settings = {"n_trials": 4, "sampler": optuna.samplers.TPESampler(seed=2)}
    res_ref = quiet(ref.tune_ml_models, spaces, optuna_settings=settings, return_tune_res=True)
    ours = dml_rs.LPLR(logit, Ridge(), logit, logit, n_folds=3)
    settings = {"n_trials": 4, "sampler": optuna.samplers.TPESampler(seed=2)}
    res = ours.tune_ml_models(x, y, d, ml_param_space=spaces, optuna_settings=settings, return_tune_res=True)
    for name, r in res_ref[0].items():
        assert res[0][name].best_params == r.best_params, name


def plpr_case(approach, score):
    np.random.seed(4)
    df = make_plpr_CP2025(num_id=60, num_t=5, dim_x=4, theta=0.5)
    x_cols = [c for c in df.columns if c.startswith("x")]
    data = quiet(dml.data.DoubleMLPanelData, df, y_col="y", d_cols="d", t_col="time", id_col="id", x_cols=x_cols,
                 static_panel=True)
    extra = {"ml_g": Lasso()} if score == "IV-type" else {}
    ref = quiet(dml.DoubleMLPLPR, data, Lasso(), Lasso(), approach=approach, score=score, n_folds=3, **extra)
    ours = dml_rs.PLPR(Lasso(), Lasso(), approach=approach, score=score, n_folds=3, **extra)
    args = (df[x_cols].to_numpy(), df["y"].to_numpy(), df["d"].to_numpy(), df["id"].to_numpy(), df["time"].to_numpy())
    return ref, ours, args


@pytest.mark.parametrize("approach", ["fd_exact", "wg_approx", "cre_general", "cre_normal"])
@pytest.mark.parametrize("score", ["partialling out", "IV-type"])
def test_tune_plpr(approach, score):
    ref, ours, args = plpr_case(approach, score)
    grids = {name: GRID_LASSO for name in ref.params_names}
    np.random.seed(5)
    quiet(ref.tune, grids, n_folds_tune=3)
    quiet(ref.fit)
    np.random.seed(5)
    ours.tune(*args, param_grids=grids, n_folds_tune=3)
    for name in ref.params_names:
        assert ours.params[name]["d"] == next(iter(ref.params[name].values())), name
    assert_same_fit(ours.fit(*args, folds=ref.smpls), ref)


def test_tune_plpr_on_folds_needs_folds():
    ref, ours, args = plpr_case("fd_exact", "partialling out")
    grids = {name: GRID_LASSO for name in ref.params_names}
    with pytest.raises(ValueError, match="whole units"):
        ours.tune(*args, param_grids=grids, tune_on_folds=True)
    np.random.seed(5)
    quiet(ref.tune, grids, tune_on_folds=True, n_folds_tune=3)
    quiet(ref.fit)
    np.random.seed(5)
    ours.tune(*args, param_grids=grids, tune_on_folds=True, n_folds_tune=3, folds=ref.smpls)
    assert ours.params["ml_l"]["d"] == next(iter(ref.params["ml_l"].values()))
    assert_same_fit(ours.fit(*args, folds=ref.smpls), ref)


def ssm_case():
    np.random.seed(11)
    x, y, d, z, s = make_ssm_data(n_obs=700, dim_x=5, theta=1.0, mar=False, return_type="array")
    reps = folds(700)
    data = dml.DoubleMLSSMData.from_arrays(x, y, d, z=z, s=s)
    ref = with_folds(dml.DoubleMLSSM, data, reps, ml_g=Lasso(), ml_pi=LOGIT, ml_m=LOGIT, score="nonignorable")
    ours = dml_rs.SSM(Lasso(), LOGIT, LOGIT, score="nonignorable", n_folds=3)
    return ref, ours, (x, y, d, s), z, reps


def test_tune_ssm_nonignorable():
    ref, ours, args, z, reps = ssm_case()
    grids = {"ml_g": GRID_LASSO, "ml_pi": GRID_LOGIT, "ml_m": GRID_LOGIT}
    np.random.seed(5)
    quiet(ref.tune, grids, n_folds_tune=3)
    np.random.seed(5)
    ours.tune(*args, z=z, param_grids=grids, n_folds_tune=3, folds=reps)
    compare_params(ours, ref, ref.params_names)
    quiet(ref.fit)
    assert_same_fit(ours.fit(*args, z=z, folds=reps), ref)


def test_tune_ml_models_ssm_nonignorable_is_not_implemented():
    pytest.importorskip("optuna")
    _, ours, args, z, _ = ssm_case()
    with pytest.raises(NotImplementedError, match="nonignorable"):
        ours.tune_ml_models(*args, z=z, ml_param_space={"ml_g": lambda trial: {}})

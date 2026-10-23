import warnings

import doubleml as dml
import numpy as np
import pandas as pd
import pytest
from doubleml.did.datasets import make_did_SZ2020
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-7
TOL_RV = 1e-4  # DoubleML finds robustness values with a bounded minimiser (xatol ~1e-5)
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
SCENARIO = dict(cf_y=0.05, cf_d=0.04, rho=0.7, level=0.9)


def folds(n, n_rep=1):
    return [list(KFold(5, shuffle=True, random_state=r).split(np.zeros(n))) for r in range(n_rep)]


def doubleml(cls, data, reps, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(data, n_folds=5, n_rep=len(reps), draw_sample_splitting=False, **kwargs)
        model.set_sample_splitting(reps)
        return model.fit()


def assert_same_sensitivity(ours, ref):
    for q in ("theta", "se", "ci"):
        for b in ("lower", "upper"):
            np.testing.assert_allclose(ours.sensitivity_params_[q][b], ref.sensitivity_params[q][b], rtol=TOL)
    for key in ("rv", "rva"):
        np.testing.assert_allclose(ours.sensitivity_params_[key], ref.sensitivity_params[key], atol=TOL_RV)


def test_plr_sensitivity_and_benchmark_match_doubleml():
    np.random.seed(1)
    x, y, d = make_plr_CCDDHNR2018(n_obs=500, dim_x=6, alpha=0.5, return_type="array")
    reps = folds(len(y), n_rep=2)
    ref = doubleml(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d), reps,
                   ml_l=LinearRegression(), ml_m=LinearRegression())
    ours = dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d, folds=reps)
    ref.sensitivity_analysis(**SCENARIO, null_hypothesis=0.2)
    ours.sensitivity_analysis(**SCENARIO, null_hypothesis=0.2)
    assert_same_sensitivity(ours, ref)
    assert "Robustness Values" in ours.sensitivity_summary

    bm_ref = ref.sensitivity_benchmark(benchmarking_set=["X1", "X2"])
    bm = ours.sensitivity_benchmark(benchmarking_set=[0, 1])
    np.testing.assert_allclose(bm.to_numpy(), bm_ref[["cf_y", "cf_d", "rho", "delta_theta"]].to_numpy(), rtol=TOL)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_irm_sensitivity_matches_doubleml(score):
    np.random.seed(2)
    x, y, d = make_irm_data(n_obs=600, dim_x=5, theta=0.5, return_type="array")
    reps = folds(len(y))
    ref = doubleml(dml.DoubleMLIRM, dml.DoubleMLData.from_arrays(x, y, d), reps,
                   ml_g=LinearRegression(), ml_m=LOGIT, score=score)
    ours = dml_rs.IRM(LinearRegression(), LOGIT, score=score).fit(x, y, d, folds=reps)
    ref.sensitivity_analysis(**SCENARIO)
    ours.sensitivity_analysis(**SCENARIO)
    assert_same_sensitivity(ours, ref)


@pytest.mark.parametrize("score", ["observational", "experimental"])
def test_did_sensitivity_matches_doubleml(score):
    np.random.seed(3)
    x, y, d, _ = make_did_SZ2020(n_obs=600, cross_sectional_data=False, return_type="array")
    reps = folds(len(y))
    ml_m = LOGIT if score == "observational" else None
    ref = doubleml(dml.DoubleMLDID, dml.DoubleMLDIDData.from_arrays(x, y, d), reps,
                   ml_g=LinearRegression(), ml_m=ml_m, score=score)
    ours = dml_rs.DID(LinearRegression(), ml_m, score=score).fit(x, y, d, folds=reps)
    ref.sensitivity_analysis(**SCENARIO)
    ours.sensitivity_analysis(**SCENARIO)
    assert_same_sensitivity(ours, ref)


def test_dataframe_benchmark_names_and_errors():
    rng = np.random.default_rng(4)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["age", "income", "region"])
    d = (X["age"] + rng.normal(size=n)).rename("training")  # the treatment's name is kept
    y = 0.5 * d + X["age"] + 0.5 * X["income"] + rng.normal(size=n)
    model = dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()).fit(X, y, d)
    bm = model.sensitivity_benchmark(benchmarking_set=["age"])
    assert list(bm.columns) == ["cf_y", "cf_d", "rho", "delta_theta"]
    assert list(bm.index) == ["training"]
    assert bm.loc["training", "cf_d"] > 0.1  # age drives the treatment
    with pytest.raises(ValueError, match="sensitivity_analysis"):
        dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()).fit(X, y, d).sensitivity_summary
    with pytest.raises(ValueError, match="cf_y"):
        model.sensitivity_analysis(cf_y=1.5)
    z = X["income"].to_numpy()
    with pytest.raises(ValueError, match="sensitivity"):
        dml_rs.PLIV(dml_rs.Ols(), dml_rs.Ols(), dml_rs.Ols()).fit(X, y, d, z).sensitivity_analysis()

"""Prediction-powered inference: parity with ppi_py 0.2.3 (fixture from
parity/generate_ppi_fixtures.py) and cross-fitting with scikit-learn learners."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import dml_rs

FIXTURE = Path(__file__).parents[2] / "dml-core" / "tests" / "fixtures" / "ppi_ppi_py.json"


def load():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("case", load()["cases"], ids=lambda c: c["name"])
def test_matches_ppi_py(case):
    data = load()["data"]
    estimand = {"mean": "mean", "linear": "ols", "logistic": "logistic"}[case["estimand"]]
    y, yhat = (data["y"], data["yhat"]) if case["outcome"] == "y" else (data["b"], data["phat"])
    kwargs = {}
    if estimand != "mean":
        kwargs = dict(X=np.array(data["X"]), X_unlabeled=np.array(data["Xu"]))
    model = dml_rs.PPI(estimand, lam=case["lam"], fit_intercept=False).fit(
        y, yhat, data[case["unlabelled"]], **kwargs
    )
    tol = 1e-7 if estimand == "logistic" else 1e-9
    np.testing.assert_allclose(model.coef_, case["coef"], rtol=0, atol=tol)
    np.testing.assert_allclose(model.se_, case["se"], rtol=0, atol=tol)
    assert abs(model.lam_ - case["lam_used"]) < tol
    # ppi_py's 90 % interval.
    z = 1.6448536269514722
    ci = model.confint(0.9)
    np.testing.assert_allclose(ci[:, 0], np.array(case["coef"]) - z * np.array(case["se"]), atol=tol)


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 2.0 + x[:, 0] - 0.5 * x[:, 1] + np.sin(2 * x[:, 2]) + 0.5 * rng.normal(size=n)
    return x, y


def test_cross_fitting_matches_a_manual_scikit_learn_loop():
    x, y = simulate(300, 0)
    xu, _ = simulate(3000, 1)
    model = dml_rs.PPI(learner=LinearRegression(), n_folds=4, random_state=7).fit(
        y, features=x, features_unlabeled=xu
    )
    # Out-of-fold predictions on the labelled rows, fold-averaged on the unlabelled ones,
    # with the folds the model drew (recovered from its own predictions).
    yhat_u = np.zeros(len(xu))
    covered = np.zeros(len(y), dtype=bool)
    for train in _training_sets(model, x, y):
        m = LinearRegression().fit(x[train], y[train])
        test = np.setdiff1d(np.arange(len(y)), train)
        np.testing.assert_allclose(model.yhat_[test], m.predict(x[test]), atol=1e-10)
        covered[test] = True
        yhat_u += m.predict(xu) / 4
    assert covered.all()
    np.testing.assert_allclose(model.yhat_unlabeled_, yhat_u, atol=1e-10)
    # Same as PPI on those predictions.
    same = dml_rs.PPI().fit(y, model.yhat_, model.yhat_unlabeled_)
    np.testing.assert_allclose(same.coef_, model.coef_, atol=1e-12)
    np.testing.assert_allclose(same.se_, model.se_, atol=1e-12)


def _training_sets(model, x, y):
    """The fold structure of dml-rs' KFold(4, seed 7) via the native folds."""
    folds = dml_rs._dml_rs.draw_folds(len(y), 4, 1, 7)[0]
    return [np.asarray(train) for train, _ in folds]


def test_more_precise_with_good_predictions_and_unbiased_with_bad_ones():
    x, y = simulate(400, 2)
    xu, yu_true = simulate(20000, 3)
    model = dml_rs.PPI(learner=RandomForestRegressor(n_estimators=100, random_state=0)).fit(
        y, features=x, features_unlabeled=xu
    )
    assert model.se_[0] < 0.7 * model.classical_.se_[0]
    assert model.effective_n_[0] > 2 * len(y)
    assert abs(model.coef_[0] - yu_true.mean()) < 4 * model.se_[0]
    # Biased predictions: still centred on the truth.
    off = dml_rs.PPI(lam=1.0).fit(y, model.yhat_ + 0.5, model.yhat_unlabeled_ + 0.5)
    assert abs(off.coef_[0] - model.coef_[0]) < 1e-12


def test_regression_with_names_and_intercept():
    import pandas as pd

    x, y = simulate(300, 4)
    xu, _ = simulate(2000, 5)
    X = pd.DataFrame(x[:, :2], columns=["a", "b"])
    Xu = pd.DataFrame(xu[:, :2], columns=["a", "b"])
    model = dml_rs.PPI("ols", learner=dml_rs.RandomForestRegressor(n_estimators=50, random_state=0)).fit(
        y, X=X, X_unlabeled=Xu, features=x, features_unlabeled=xu
    )
    assert list(model.summary.index) == ["intercept", "a", "b"]
    assert abs(model.coef_[1] - 1.0) < 4 * model.se_[1]
    assert np.all(model.se_ < model.classical_.se_)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.coef_, model.coef_)


def test_logistic_with_a_classifier():
    rng = np.random.default_rng(6)
    n, nu = 500, 5000
    f = rng.normal(size=(n + nu, 2))
    p = 1 / (1 + np.exp(-(0.3 + f[:, 0] + 1.5 * f[:, 1])))
    yb = (rng.random(n + nu) < p).astype(float)
    model = dml_rs.PPI("logistic", learner=dml_rs.LogisticRegression()).fit(
        yb[:n], X=f[:n, :1], X_unlabeled=f[n:, :1], features=f[:n], features_unlabeled=f[n:]
    )
    assert model.yhat_.min() >= 0 and model.yhat_.max() <= 1
    assert np.all(model.se_ < model.classical_.se_)


def test_errors():
    y = np.arange(5.0)
    with pytest.raises(ValueError, match="estimand"):
        dml_rs.PPI("median").fit(y, y, y)
    with pytest.raises(ValueError, match="needs X"):
        dml_rs.PPI("ols").fit(y, y, y)
    with pytest.raises(ValueError, match="no covariates"):
        dml_rs.PPI().fit(y, y, y, X=y, X_unlabeled=y)
    with pytest.raises(ValueError, match="yhat"):
        dml_rs.PPI().fit(y)
    with pytest.raises(ValueError, match="lambda"):
        dml_rs.PPI(lam=2.0).fit(y, y, y)
    with pytest.raises(ValueError, match="not both"):
        dml_rs.PPI(learner=LinearRegression()).fit(y, y, y)

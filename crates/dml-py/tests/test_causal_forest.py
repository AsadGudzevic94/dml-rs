"""Causal forests: recovery of heterogeneous effects, inference and the model API."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs


def sig(u):
    return 1 + 1 / (1 + np.exp(-20 * (u - 1 / 3)))


def tau(X):
    """Wager and Athey (2018): heterogeneity in the first two covariates."""
    return sig(X[:, 0]) * sig(X[:, 1])


def data(n=3000, p=5, seed=0, confounded=False):
    rng = np.random.default_rng(seed)
    X = rng.random((n, p))
    e = 1 / (1 + np.exp(-(X[:, 2] - 0.5) * 3)) if confounded else np.full(n, 0.5)
    d = (rng.random(n) < e).astype(float)
    y = X[:, 2] + tau(X) * d + rng.normal(size=n)
    return X, y, d


@pytest.fixture(scope="module")
def fitted():
    X, y, d = data()
    model = dml_rs.CausalForest(LinearRegression(), LogisticRegression(), n_estimators=1000).fit(X, y, d)
    return model, X, y, d


def test_effects_and_intervals(fitted):
    model, X, _, _ = fitted
    grid = np.random.default_rng(1).random((200, X.shape[1]))
    est = model.effect(grid)
    rmse = np.sqrt(np.mean((est - tau(grid)) ** 2))
    assert rmse < 0.3
    ci = model.effect_interval(grid)
    assert ci.shape == (200, 2) and np.all(ci[:, 0] < ci[:, 1])
    cover = np.mean((ci[:, 0] <= tau(grid)) & (tau(grid) <= ci[:, 1]))
    assert cover > 0.8
    np.testing.assert_allclose((ci[:, 1] - ci[:, 0]) / 2, 1.959963984540054 * model.effect_se(grid), rtol=1e-10)
    # Out-of-bag estimates of the training rows.
    assert model.effect_.shape == (len(X),) and np.all(np.isfinite(model.effect_se_))
    assert np.corrcoef(model.effect_, tau(X))[0, 1] > 0.8
    imp = model.variable_importance_
    assert imp[:2].sum() > 0.7 and np.isclose(imp.sum(), 1.0)


def test_average_effects_and_calibration(fitted):
    model, X, _, _ = fitted
    ate = model.ate()
    assert abs(ate.coef_[0] - tau(X).mean()) < 4 * ate.se_[0]
    assert "Causal forest (ATE)" in str(ate)
    groups = np.column_stack([X[:, 0] <= 0.3, X[:, 0] > 0.3]).astype(float)
    gate = model.gate(groups)
    assert gate.coef[1] > gate.coef[0] + 3 * gate.se[1]
    blp = model.blp(np.column_stack([np.ones(len(X)), X[:, 0]]))
    assert blp.coef[1] > 0
    cal = model.test_calibration()
    assert abs(cal["estimate"].iloc[0] - 1) < 0.2
    assert cal["Pr(>t)"].iloc[1] < 0.01  # real heterogeneity is detected


def test_no_heterogeneity_is_not_detected():
    rng = np.random.default_rng(3)
    X = rng.random((2000, 4))
    d = (rng.random(2000) < 0.5).astype(float)
    y = X[:, 0] + 1.0 * d + rng.normal(size=2000)
    model = dml_rs.CausalForest(LinearRegression(), LogisticRegression(), n_estimators=500).fit(X, y, d)
    assert abs(model.effect_.mean() - 1.0) < 0.15
    assert model.test_calibration()["Pr(>t)"].iloc[1] > 0.01


def test_confounded_treatment_with_flexible_learners():
    X, y, d = data(n=2000, seed=4, confounded=True)
    model = dml_rs.CausalForest(dml_rs.HistGradientBoostingRegressor(), dml_rs.HistGradientBoostingClassifier(),
                                n_estimators=500).fit(X, y, d)
    ate = model.ate()
    assert abs(ate.coef_[0] - tau(X).mean()) < 4 * ate.se_[0]


def test_continuous_treatment_reproducibility_and_errors():
    rng = np.random.default_rng(5)
    X = rng.random((1500, 3))
    d = X[:, 0] + rng.normal(size=1500)
    y = (1 + X[:, 1]) * d + rng.normal(size=1500)
    kw = dict(n_estimators=400, random_state=7)
    a = dml_rs.CausalForest(dml_rs.Ols(), dml_rs.Ols(), **kw).fit(X, y, d)
    b = dml_rs.CausalForest(dml_rs.Ols(), dml_rs.Ols(), **kw).fit(dml_rs.DmlData.from_arrays(X, y, d))
    np.testing.assert_array_equal(a.effect(X[:50]), b.effect(X[:50]))
    assert np.corrcoef(a.effect_, 1 + X[:, 1])[0, 1] > 0.8
    with pytest.raises(Exception, match="binary"):
        a.ate()
    with pytest.raises(ValueError, match="one treatment"):
        dml_rs.CausalForest(dml_rs.Ols(), dml_rs.Ols()).fit(X, y, np.column_stack([d, d]))
    with pytest.raises(Exception, match="sample_fraction"):
        dml_rs.CausalForest(dml_rs.Ols(), dml_rs.Ols(), sample_fraction=0.8).fit(X, y, d)
    no_ci = dml_rs.CausalForest(dml_rs.Ols(), dml_rs.Ols(), n_estimators=50, ci_group_size=1,
                                sample_fraction=0.7).fit(X, y, d)
    assert np.all(np.isnan(no_ci.effect_se(X[:5])))
    with pytest.raises(AttributeError, match="not fitted"):
        dml_rs.CausalForest(dml_rs.Ols(), dml_rs.Ols()).effect(X)

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


def test_policy_tree_on_doubly_robust_scores():
    rng = np.random.default_rng(6)
    X = rng.random((2000, 3))
    d = (rng.random(2000) < 0.5).astype(float)
    y = 4 * (X[:, 0] - 0.5) * d + rng.normal(size=2000)
    model = dml_rs.CausalForest(LinearRegression(), LogisticRegression(), n_estimators=400).fit(X, y, d)
    tree = model.policy_tree(X, depth=1)
    np.testing.assert_array_equal(tree.predict(np.array([[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])), [0, 1])
    assert "X0" in str(tree) or "x0" in str(tree).lower()


def iv_data(n=4000, seed=7):
    """Binary instrument; compliers 60%, always-takers 20% (higher outcomes), never-takers 20%."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 4))
    z = (rng.random(n) < 0.5).astype(float)
    u = rng.random(n)
    d = np.where(u < 0.6, z, np.where(u < 0.8, 1.0, 0.0))
    y = (1 + 2 * X[:, 0]) * d + X[:, 1] + np.where((u >= 0.6) & (u < 0.8), 1.5, np.where(u >= 0.8, -0.5, 0.0))
    return X, y + rng.normal(size=n), d, z


def test_instrumental_forest():
    X, y, d, z = iv_data()
    model = dml_rs.InstrumentalForest(LinearRegression(), LogisticRegression(), LogisticRegression(),
                                      n_estimators=1000).fit(X, y, d, z)
    truth = 1 + 2 * X[:, 0]
    assert np.corrcoef(model.effect_, truth)[0, 1] > 0.8
    assert abs(model.compliance_.mean() - 0.6) < 0.05
    ate = model.ate()
    assert abs(ate.coef_[0] - truth.mean()) < 4 * ate.se_[0]
    assert "Instrumental forest (average LATE)" in str(ate)
    grid = np.random.default_rng(1).random((200, 4))
    ci = model.effect_interval(grid)
    cover = np.mean((ci[:, 0] <= 1 + 2 * grid[:, 0]) & (1 + 2 * grid[:, 0] <= ci[:, 1]))
    assert cover > 0.75
    groups = np.column_stack([X[:, 0] <= 0.5, X[:, 0] > 0.5]).astype(float)
    assert model.gate(groups).coef[1] > model.gate(groups).coef[0]
    # A naive causal forest that ignores the instrument is biased by the always-takers.
    naive = dml_rs.CausalForest(LinearRegression(), LogisticRegression(), n_estimators=300).fit(X, y, d)
    assert naive.ate().coef_[0] > ate.coef_[0] + 0.3
    # DmlData with an instrument; continuous instruments have no doubly robust scores.
    data = dml_rs.DmlData.from_arrays(X, y, d, z=z)
    same = dml_rs.InstrumentalForest(LinearRegression(), LogisticRegression(), LogisticRegression(),
                                     n_estimators=1000).fit(data)
    np.testing.assert_array_equal(same.effect_, model.effect_)
    cont = dml_rs.InstrumentalForest(dml_rs.Ols(), dml_rs.Ols(), dml_rs.Ols(), n_estimators=100).fit(X, y, d, z + 0.1)
    assert cont.compliance_ is None
    with pytest.raises(Exception, match="binary"):
        cont.ate()
    with pytest.raises(Exception, match="reduced_form_weight"):
        dml_rs.InstrumentalForest(dml_rs.Ols(), dml_rs.Ols(), dml_rs.Ols(), reduced_form_weight=2).fit(X, y, d, z)


def test_multi_arm_causal_forest():
    rng = np.random.default_rng(8)
    n = 4000
    X = rng.random((n, 4))
    d = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.4, 0.3, 0.3])
    tau1, tau2 = 2 * X[:, 0], 1 - X[:, 0]
    y = X[:, 1] + np.where(d == 1, tau1, np.where(d == 2, tau2, 0.0)) + rng.normal(size=n)
    model = dml_rs.MultiArmCausalForest(LinearRegression(), LogisticRegression(), n_estimators=1000).fit(X, y, d)
    np.testing.assert_array_equal(model.treatment_levels_, [0, 1, 2])
    assert model.arm_names_ == ["d=1 vs 0", "d=2 vs 0"]
    assert model.effect_.shape == (n, 2) and model.effect_interval(X[:5]).shape == (5, 2, 2)
    assert np.corrcoef(model.effect_[:, 0], tau1)[0, 1] > 0.8
    assert np.corrcoef(model.effect_[:, 1], tau2)[0, 1] > 0.6
    np.testing.assert_allclose(model.propensity_.sum(axis=1), 1.0)
    ate = model.ate()
    assert np.all(np.abs(ate.coef_ - [1.0, 0.5]) < 4 * ate.se_)
    groups = np.column_stack([X[:, 0] <= 0.5, X[:, 0] > 0.5]).astype(float)
    gate = model.gate(groups, level=2.0)
    assert gate.coef[0] > gate.coef[1]
    with pytest.raises(ValueError, match="non-baseline"):
        model.blp(X, level=0.0)


def test_regression_and_quantile_forests():
    rng = np.random.default_rng(9)
    n = 3000
    X = rng.random((n, 3))
    y = np.sin(3 * X[:, 0]) + (0.5 + X[:, 1]) * rng.normal(size=n)
    rf = dml_rs.RegressionForest(n_estimators=500).fit(X, y)
    grid = rng.random((200, 3))
    pred, var = rf.predict_with_variance(grid)
    np.testing.assert_array_equal(pred, rf.predict(grid))
    assert np.sqrt(np.mean((pred - np.sin(3 * grid[:, 0])) ** 2)) < 0.2 and np.all(var >= 0)
    assert rf.predict_oob(X)[0].shape == (n,)
    # Usable as a nuisance learner (and copied per fold).
    d = X[:, 0] + rng.normal(size=n)
    plr = dml_rs.PLR(dml_rs.RegressionForest(n_estimators=200), dml_rs.RegressionForest(n_estimators=200)).fit(
        X, 0.5 * d + y, d)
    assert abs(plr.coef_[0] - 0.5) < 4 * plr.se_[0]
    qf = dml_rs.QuantileForest(n_estimators=500).fit(X, y)
    assert list(qf.quantiles) == [0.1, 0.5, 0.9]
    q = qf.predict(grid)
    assert q.shape == (200, 3) and np.all(q[:, 0] <= q[:, 1]) and np.all(q[:, 1] <= q[:, 2])
    true90 = np.sin(3 * grid[:, 0]) + 1.2815515655446004 * (0.5 + grid[:, 1])
    assert np.mean(np.abs(q[:, 2] - true90)) < 0.3
    assert qf.predict(grid, quantiles=[0.25]).shape == (200, 1)
    assert qf.variable_importance_ is not None
    import copy
    np.testing.assert_array_equal(copy.deepcopy(qf).predict(grid[:3]), q[:3])

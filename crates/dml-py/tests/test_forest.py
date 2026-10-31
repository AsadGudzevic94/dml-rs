import os

import numpy as np
import pytest

import dml_rs


def nonlinear(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, 4))
    confounder = 2 * np.sin(x[:, 0]) + 1.5 * (x[:, 1] > 0)
    d = confounder + rng.normal(size=n)
    y = 0.5 * d + 2 * confounder + rng.normal(size=n)
    return x, y, d


def test_regressor_learns_and_is_deterministic():
    x, y, _ = nonlinear(2000, 0)
    x_test, y_test, _ = nonlinear(500, 1)
    rf = dml_rs.RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=3)
    pred = rf.fit(x, y).predict(x_test)
    r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    assert r2 > 0.6  # y also contains d's own noise, so R² is bounded well below 1
    again = dml_rs.RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=3).fit(x, y)
    np.testing.assert_array_equal(again.predict(x_test), pred)
    imp = rf.feature_importances_
    assert imp.shape == (4,) and imp.sum() == pytest.approx(1.0)
    assert imp[2] < imp[0] and imp[3] < imp[0]  # x2, x3 are noise


def test_classifier_and_max_features_parsing():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 5))
    labels = (x[:, 0] + 0.5 * rng.normal(size=1000) > 0).astype(float)
    for max_features in [None, "sqrt", "log2", 0.5, 2]:
        p = dml_rs.RandomForestClassifier(n_estimators=30, max_features=max_features).fit(x, labels).predict_proba(x)[:, 1]
        assert p.shape == (1000,) and np.all((p >= 0) & (p <= 1))
    # Parameters are validated by fit, as scikit-learn.
    with pytest.raises(ValueError, match="max_features"):
        dml_rs.RandomForestClassifier(max_features="half").fit(x, labels)
    with pytest.raises(TypeError, match="max_features"):
        dml_rs.RandomForestRegressor(max_features=[1]).fit(x, labels)
    # Any two labels work (classes_ holds them); more than two do not.
    relabelled = dml_rs.RandomForestClassifier(n_estimators=30).fit(x, labels * 2)
    np.testing.assert_array_equal(relabelled.classes_, [0.0, 2.0])
    np.testing.assert_allclose(
        relabelled.predict_proba(x), dml_rs.RandomForestClassifier(n_estimators=30).fit(x, labels).predict_proba(x)
    )
    with pytest.raises(ValueError, match="Only binary classification"):
        dml_rs.RandomForestClassifier().fit(x, labels + (x[:, 1] > 1))


def test_native_forests_inside_dml_models():
    x, y, d = nonlinear(2000, 4)
    rf = dml_rs.RandomForestRegressor(n_estimators=100, min_samples_leaf=5)
    plr = dml_rs.PLR(rf, rf).fit(x, y, d)
    lo, hi = plr.confint()[0]
    assert lo < 0.5 < hi
    ols = dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()).fit(x, y, d)
    assert abs(ols.coef_[0] - 0.5) > 0.2  # linear nuisances are biased here

    d_bin = (d > 0).astype(float)
    irm = dml_rs.IRM(
        dml_rs.RandomForestRegressor(n_estimators=50, min_samples_leaf=10),
        dml_rs.RandomForestClassifier(n_estimators=50, min_samples_leaf=10),
    ).fit(x, y + d_bin, d_bin)
    assert np.isfinite(irm.coef_[0]) and irm.se_[0] > 0


@pytest.mark.skipif(bool(os.environ.get("CI")), reason="timing comparison is unreliable on shared CI runners")
def test_native_forest_is_faster_than_sklearn_inside_plr():
    """Same forest settings; the native forest avoids per-fold Python calls."""
    import time

    from sklearn.ensemble import RandomForestRegressor

    x, y, d = nonlinear(3000, 5)
    settings = dict(n_estimators=100, min_samples_leaf=5, max_features=1.0)
    start = time.perf_counter()
    native = dml_rs.PLR(dml_rs.RandomForestRegressor(**settings), dml_rs.RandomForestRegressor(**settings))
    native.fit(x, y, d)
    t_native = time.perf_counter() - start
    start = time.perf_counter()
    sk = RandomForestRegressor(**settings, n_jobs=-1, random_state=0)
    dml_rs.PLR(sk, sk).fit(x, y, d)
    t_sklearn = time.perf_counter() - start
    assert t_native < t_sklearn, f"native {t_native:.2f}s vs sklearn {t_sklearn:.2f}s"

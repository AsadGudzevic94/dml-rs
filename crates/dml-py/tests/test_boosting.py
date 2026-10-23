import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import dml_rs


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(-2, 2, size=(n, 5))
    y = np.sin(2 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2] + 0.3 * rng.normal(size=n)
    labels = (rng.uniform(size=n) < 1 / (1 + np.exp(-(2 * x[:, 0] - x[:, 1] ** 2)))).astype(float)
    return x, y, labels


SETTINGS = [
    {},
    {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 5},
    {"n_estimators": 50, "learning_rate": 0.2, "max_depth": 2, "min_samples_split": 10},
]


@pytest.mark.parametrize("kwargs", SETTINGS)
def test_regressor_matches_sklearn_exactly(data, kwargs):
    """Same trees as scikit-learn: squared-error splits, midpoint thresholds, mean leaves."""
    x, y, _ = data
    ours = dml_rs.GradientBoostingRegressor(**kwargs).fit(x, y).predict(x)
    ref = GradientBoostingRegressor(random_state=0, **kwargs).fit(x, y).predict(x)
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-10)


@pytest.mark.parametrize("kwargs", [SETTINGS[0], SETTINGS[2]])
def test_classifier_matches_sklearn_exactly(data, kwargs):
    """Log-loss boosting with Newton leaf updates, as scikit-learn."""
    x, _, labels = data
    ours = dml_rs.GradientBoostingClassifier(**kwargs).fit(x, labels).predict_proba(x)
    ref = GradientBoostingClassifier(random_state=0, **kwargs).fit(x, labels).predict_proba(x)[:, 1]
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-10)


def test_classifier_tie_breaking_stays_within_sklearns_own_seed_variation(data):
    """With 0/1 labels the first stage fits a two-valued gradient, so splits on different
    features can have exactly equal gain in small nodes. scikit-learn breaks such ties by a
    random feature order (seeded by random_state, even with max_features=None), dml-rs by its
    own seeded order. Results then differ by no more than scikit-learn differs from itself."""
    x, _, labels = data
    kwargs = SETTINGS[1]
    ours = dml_rs.GradientBoostingClassifier(**kwargs).fit(x, labels).predict_proba(x)
    sk0, sk1 = (GradientBoostingClassifier(random_state=s, **kwargs).fit(x, labels).predict_proba(x)[:, 1]
                for s in (0, 1))
    seed_spread = np.max(np.abs(sk0 - sk1))
    assert seed_spread > 1e-6  # scikit-learn itself depends on the tie-break here
    assert np.max(np.abs(ours - sk0)) <= seed_spread


def test_boosting_nuisances_in_dml():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.uniform(-3, 3, size=(n, 4))
    confounder = 2 * np.sin(x[:, 0]) + 1.5 * (x[:, 1] > 0)
    d = confounder + rng.normal(size=n)
    y = 0.5 * d + 2 * confounder + rng.normal(size=n)
    gb = dml_rs.GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
    lo, hi = dml_rs.PLR(gb, gb).fit(x, y, d).confint()[0]
    assert lo < 0.5 < hi

    d_bin = (d > 0).astype(float)
    irm = dml_rs.IRM(gb, dml_rs.GradientBoostingClassifier(n_estimators=100)).fit(x, y + d_bin, d_bin)
    assert np.isfinite(irm.coef_[0]) and irm.se_[0] > 0


def test_options_and_errors(data):
    x, y, labels = data
    gb = dml_rs.GradientBoostingRegressor(subsample=0.7, max_features="sqrt", random_state=3).fit(x, y)
    again = dml_rs.GradientBoostingRegressor(subsample=0.7, max_features="sqrt", random_state=3).fit(x, y)
    np.testing.assert_array_equal(gb.predict(x), again.predict(x))
    assert gb.feature_importances_.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="learning_rate"):
        dml_rs.GradientBoostingRegressor(learning_rate=0).fit(x, y)
    with pytest.raises(ValueError, match="labels"):
        dml_rs.GradientBoostingClassifier().fit(x, labels * 2)

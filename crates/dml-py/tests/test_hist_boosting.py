"""Histogram gradient boosting against scikit-learn's HistGradientBoosting* (same random_state)."""

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier as SkClassifier
from sklearn.ensemble import HistGradientBoostingRegressor as SkRegressor

import dml_rs


def data(n, p=5, seed=0, missing=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, -1] = rng.integers(0, 4, size=n)  # few distinct values
    y = np.sin(2 * X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, -1] + rng.normal(size=n)
    if missing:
        X[rng.random((n, p)) < missing] = np.nan
    return X, y, rng.normal(size=(500, p))


SETTINGS = [
    {},
    {"max_depth": 3, "l2_regularization": 1.0},
    {"max_leaf_nodes": None, "max_depth": 4, "min_samples_leaf": 5},
    {"max_bins": 16, "learning_rate": 0.3, "max_iter": 50},
    {"early_stopping": True, "validation_fraction": None, "n_iter_no_change": 3, "max_iter": 300},
    {"early_stopping": True, "max_iter": 300, "learning_rate": 0.5},
]


@pytest.mark.parametrize("kwargs", SETTINGS)
@pytest.mark.parametrize("missing", [0.0, 0.05])
def test_regressor_matches_sklearn(kwargs, missing):
    X, y, Xt = data(1500, missing=missing)
    ours = dml_rs.HistGradientBoostingRegressor(random_state=3, **kwargs).fit(X, y)
    ref = SkRegressor(random_state=3, **kwargs).fit(X, y)
    assert ours.n_iter_ == ref.n_iter_
    np.testing.assert_array_equal(ours.predict(Xt), ref.predict(Xt))
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("kwargs", SETTINGS)
def test_classifier_matches_sklearn(kwargs):
    X, y, Xt = data(1500, seed=1, missing=0.02)
    yb = (y > np.median(y)).astype(float)
    ours = dml_rs.HistGradientBoostingClassifier(random_state=5, **kwargs).fit(X, yb)
    ref = SkClassifier(random_state=5, **kwargs).fit(X, yb)
    assert ours.n_iter_ == ref.n_iter_
    np.testing.assert_allclose(ours.predict_proba(Xt)[:, 1], ref.predict_proba(Xt)[:, 1], rtol=1e-14, atol=1e-15)


def test_early_stopping_scores_match():
    X, y, _ = data(12_000, seed=2)  # "auto": on above 10 000 samples
    ours = dml_rs.HistGradientBoostingRegressor(random_state=7).fit(X, y)
    ref = SkRegressor(random_state=7).fit(X, y)
    assert ref.do_early_stopping_ and ours.n_iter_ == ref.n_iter_ < 100
    np.testing.assert_array_equal(ours.validation_score_, ref.validation_score_)


@pytest.mark.parametrize("kind", ["regressor", "classifier"])
def test_sample_weights_match_sklearn(kind):
    X, y, Xt = data(2000, seed=3)
    w = np.random.default_rng(4).exponential(size=len(y))
    w[::17] = 0.0
    if kind == "regressor":
        ours = dml_rs.HistGradientBoostingRegressor(random_state=1).fit(X, y, sample_weight=w)
        ref = SkRegressor(random_state=1).fit(X, y, sample_weight=w)
        np.testing.assert_allclose(ours.predict(Xt), ref.predict(Xt), rtol=1e-13, atol=1e-13)
    else:
        yb = (y > np.median(y)).astype(float)
        ours = dml_rs.HistGradientBoostingClassifier(random_state=1).fit(X, yb, sample_weight=w)
        ref = SkClassifier(random_state=1).fit(X, yb, sample_weight=w)
        np.testing.assert_allclose(ours.predict_proba(Xt)[:, 1], ref.predict_proba(Xt)[:, 1], rtol=1e-13, atol=1e-14)


def test_large_data_bins_a_subsample_like_sklearn():
    X, y, Xt = data(210_000, p=3, seed=5)
    kwargs = {"max_iter": 20, "early_stopping": False, "random_state": 11}
    ours = dml_rs.HistGradientBoostingRegressor(**kwargs).fit(X, y)
    ref = SkRegressor(**kwargs).fit(X, y)
    np.testing.assert_array_equal(ours.predict(Xt), ref.predict(Xt))


def test_as_nuisance_learner_matches_sklearn_inside_dml():
    X, y, _ = data(1200, seed=6)
    d = (X[:, 0] + np.random.default_rng(7).normal(size=len(y)) > 0).astype(float)
    y = y + 0.5 * d
    ours = dml_rs.IRM(dml_rs.HistGradientBoostingRegressor(), dml_rs.HistGradientBoostingClassifier()).fit(X, y, d)
    ref = dml_rs.IRM(SkRegressor(random_state=0), SkClassifier(random_state=0)).fit(X, y, d)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-12)
    np.testing.assert_allclose(ours.se_, ref.se_, rtol=1e-12)


def test_errors_and_api():
    X, y, _ = data(300)
    with pytest.raises(ValueError, match="early_stopping"):
        dml_rs.HistGradientBoostingRegressor(early_stopping="sometimes").fit(X, y)
    with pytest.raises(Exception, match="max_bins"):
        dml_rs.HistGradientBoostingRegressor(max_bins=300).fit(X, y)
    with pytest.raises(ValueError, match="Unknown label type"):
        dml_rs.HistGradientBoostingClassifier().fit(X, y)
    model = dml_rs.HistGradientBoostingRegressor(max_iter=10).fit(X, y)
    assert "HistGradientBoostingRegressor" in repr(model) and model.n_iter_ == 10
    import copy

    np.testing.assert_array_equal(copy.deepcopy(model).predict(X), model.predict(X))

"""Dose-response curves of a continuous treatment (Colangelo and Lee, 2023), checked against an
independent NumPy / scikit-learn computation of the same estimator on the same folds."""

import pickle

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-10


def data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    d = 0.5 * X[:, 0] + rng.normal(size=n)
    y = d + 0.5 * d**2 + X[:, 1] + d * X[:, 0] + rng.normal(size=n)
    return X, y, d


def folds(n, n_folds=5, seed=0):
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n)))


def epanechnikov(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)


def reference(X, y, d, ml_g, ml_m, grid, h, splits, density="kernel", h1=None, floor=0.01):
    """The estimator written out: cross-fitted γ(t, X) and f(t | X), then the mean score."""
    n = len(y)
    h1 = h if h1 is None else h1
    floor = floor / np.std(d, ddof=1)
    W = np.column_stack([d, X])
    if density == "normal":
        m_hat = np.empty(n)
        for train, test in splits:
            m_hat[test] = ml_m.fit(X[train], d[train]).predict(X[test])
        sigma = np.sqrt(np.mean((d - m_hat) ** 2))
    psi = np.empty((n, len(grid)))
    for train, test in splits:
        g = LinearRegression() if ml_g is None else ml_g
        g.fit(W[train], y[train])
        for j, t in enumerate(grid):
            Wt = W[test].copy()
            Wt[:, 0] = t
            g_t = g.predict(Wt)
            if density == "normal":
                f = norm.pdf(t, loc=m_hat[test], scale=sigma)
            else:
                f = ml_m.fit(X[train], norm.pdf((d[train] - t) / h1)).predict(X[test]) / h1
            f = np.maximum(f, floor)
            k = epanechnikov((d[test] - t) / h) / h
            psi[test, j] = g_t + k / f * (y[test] - g_t)
    coef = psi.mean(axis=0)
    se = np.sqrt(((psi - coef) ** 2).mean(axis=0) / n)
    return coef, se, psi - coef


@pytest.mark.parametrize("density", ["kernel", "normal"])
def test_matches_reference(density):
    X, y, d = data()
    splits = folds(len(y))
    grid = [-1.0, 0.0, 0.5, 1.5]
    h = 0.4
    model = dml_rs.DoseResponse(LinearRegression(), Ridge(alpha=1.0), grid=grid, bandwidth=h, density=density)
    model.fit(X, y, d, folds=splits)
    coef, se, psi = reference(X, y, d, LinearRegression(), Ridge(alpha=1.0), grid, h, splits, density)
    np.testing.assert_allclose(model.coef_, coef, rtol=0, atol=TOL)
    np.testing.assert_allclose(model.se_, se, rtol=0, atol=TOL)
    np.testing.assert_allclose(model.psi_[:, :, 0], psi, rtol=0, atol=TOL)
    assert model.treatment_names_ == ["-1", "0", "0.5", "1.5"]
    assert model.predictions_["ml_g"].shape == (len(y), 1, len(grid))


def test_random_forest_and_density_bandwidth():
    X, y, d = data(500, seed=1)
    splits = folds(len(y), 3, seed=2)
    grid = [-0.5, 0.5]
    rf = dict(n_estimators=30, min_samples_leaf=10, random_state=0)
    model = dml_rs.DoseResponse(
        RandomForestRegressor(**rf), RandomForestRegressor(**rf), grid=grid, bandwidth=0.5,
        density_bandwidth=0.3, trimming_threshold=0.05,
    ).fit(X, y, d, folds=splits)
    coef, se, _ = reference(X, y, d, RandomForestRegressor(**rf), RandomForestRegressor(**rf), grid, 0.5,
                            splits, h1=0.3, floor=0.05)
    np.testing.assert_allclose(model.coef_, coef, rtol=0, atol=TOL)
    np.testing.assert_allclose(model.se_, se, rtol=0, atol=TOL)


def test_default_grid_bandwidth_and_partial_effects():
    X, y, d = data(1000, seed=3)
    model = dml_rs.DoseResponse(LinearRegression(), LinearRegression(), density="normal", partial_effects=True)
    model.fit(X, y, d)
    np.testing.assert_allclose(model.grid_, np.quantile(d, np.arange(1, 10) / 10), rtol=0, atol=1e-12)
    h = np.std(d, ddof=1) * len(d) ** -0.2
    assert model.bandwidth_ == pytest.approx(h, rel=1e-12)
    assert model.eta_ == pytest.approx(h * len(d) ** (-1 / 6), rel=1e-12)
    # The partial effect is the difference quotient of the curve at t ± η/2.
    eta = model.eta_
    grid = list(model.grid_)
    shifted = dml_rs.DoseResponse(
        LinearRegression(), LinearRegression(), density="normal", bandwidth=h,
        grid=[t - eta / 2 for t in grid] + [t + eta / 2 for t in grid],
    ).fit(X, y, d)
    k = len(grid)
    np.testing.assert_allclose(model.partial_effect().coef_, (shifted.coef_[k:] - shifted.coef_[:k]) / eta,
                               rtol=0, atol=1e-9)
    # True derivative of E[Y(t)] = t + t²/2 (+ E[X1] = 0, E[d·X0] term: t·E[X0] = 0) is 1 + t.
    pe = model.partial_effect()
    assert np.mean(np.abs(pe.coef_ - (1 + model.grid_)) / pe.se_) < 2.5


def test_contrast_bootstrap_and_plot():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    X, y, d = data()
    model = dml_rs.DoseResponse(LinearRegression(), LinearRegression(), grid=[-1, 0, 1], density="normal").fit(X, y, d)
    c = model.contrast(0)
    np.testing.assert_allclose(c.coef_, [model.coef_[0] - model.coef_[1], model.coef_[2] - model.coef_[1]],
                               rtol=0, atol=1e-12)
    assert list(c.summary.index) == ["-1 vs 0", "1 vs 0"]
    model.bootstrap(n_boot=200)
    uniform = model.confint(joint=True)
    pointwise = model.confint()
    assert np.all(uniform[:, 0] < pointwise[:, 0]) and np.all(uniform[:, 1] > pointwise[:, 1])
    ax = model.plot(joint=True)
    assert ax.get_xlabel() == "treatment t"
    with pytest.raises(ValueError, match="not a grid point"):
        model.contrast(0.25)
    with pytest.raises(ValueError, match="partial_effects=True"):
        model.partial_effect()


def test_builtin_learners_and_pickle():
    X, y, d = data()
    model = dml_rs.DoseResponse(dml_rs.LassoCV(), dml_rs.LassoCV(), grid=[0.0, 1.0]).fit(X, y, d)
    again = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(again.coef_, model.coef_)
    np.testing.assert_array_equal(again.contrast(0.0).coef_, model.contrast(0.0).coef_)


def test_invalid_input():
    X, y, d = data(200)
    with pytest.raises(ValueError, match="kernel"):
        dml_rs.DoseResponse(LinearRegression(), LinearRegression(), kernel="box").fit(X, y, d)
    with pytest.raises(ValueError, match="density"):
        dml_rs.DoseResponse(LinearRegression(), LinearRegression(), density="beta").fit(X, y, d)
    with pytest.raises(ValueError, match="distinct"):
        dml_rs.DoseResponse(LinearRegression(), LinearRegression(), grid=[0, 0]).fit(X, y, d)
    with pytest.raises(ValueError, match="bandwidth"):
        dml_rs.DoseResponse(LinearRegression(), LinearRegression(), bandwidth=-1.0).fit(X, y, d)
    with pytest.raises(ValueError, match="constant"):
        dml_rs.DoseResponse(LinearRegression(), LinearRegression()).fit(X, y, np.ones(200))

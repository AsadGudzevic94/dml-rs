"""Surrogate index (Athey, Chetty, Imbens and Kang; efficient score of Chen and Ritzwoller),
checked against a NumPy / scikit-learn computation of the same estimator on the same folds and
in simulations with a known long-term effect."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold

import dml_rs

EFFECT = 2.5  # 2 · (effect 1 on s1) + 1 · (effect 0.5 on s2)


def draw(n, rng, experiment):
    x = rng.normal(size=(n, 2))
    w = (0.6 * x[:, 0] + rng.normal(size=n) > 0).astype(float) if experiment else np.zeros(n)
    s = np.column_stack([w + 0.5 * x[:, 0] + rng.normal(size=n),
                         0.5 * w - 0.3 * x[:, 1] + rng.normal(size=n)])
    y = 2 * s[:, 0] + s[:, 1] + x[:, 1] + rng.normal(size=n)
    return x, s, w, y


def simulate(n_exp=600, n_obs=800, seed=0):
    rng = np.random.default_rng(seed)
    xe, se, w, _ = draw(n_exp, rng, True)
    xo, so, _, y = draw(n_obs, rng, False)
    return xe, se, w, xo, so, y


def reference(xe, se, w, xo, so, y, ml_g, ml_m, folds, propensity=None, t=0.01):
    """The estimator written out on the stacked data (experiment first)."""
    n_exp = len(w)
    x = np.vstack([xe, xo])
    sx = np.column_stack([np.vstack([se, so]), x])
    n = len(x)
    g = (np.arange(n) < n_exp).astype(float)
    ww = np.concatenate([w, np.zeros(len(y))])
    yy = np.concatenate([np.zeros(n_exp), y])
    psi_a = np.zeros((n, 2))
    psi_b = np.zeros((n, 2))
    for train, test in folds:
        exp, obs = train[train < n_exp], train[train >= n_exp]
        index = clone(ml_g).fit(sx[obs], yy[obs])
        nu, nu_exp = index.predict(sx[test]), index.predict(sx[exp])
        mu = {}
        for arm in (1, 0):
            sel = ww[exp] == arm
            mu[arm] = clone(ml_g).fit(x[exp[sel]], nu_exp[sel]).predict(x[test])
        if propensity is None:
            pi = clone(ml_m).fit(x[exp], ww[exp]).predict_proba(x[test])[:, 1]
        else:
            pi = np.full(len(test), propensity)
        gamma = clone(ml_m).fit(sx[exp], ww[exp]).predict_proba(sx[test])[:, 1]
        rho = clone(ml_m).fit(sx[train], g[train]).predict_proba(sx[test])[:, 1]
        pi, gamma, rho = (np.clip(v, t, 1 - t) for v in (pi, gamma, rho))
        is_exp = test < n_exp
        wt = ww[test]
        psi_a[test[is_exp]] = -1
        psi_b[test, 0] = np.where(is_exp, mu[1] + wt / pi * (nu - mu[1]),
                                  rho / (1 - rho) * gamma / pi * (yy[test] - nu))
        psi_b[test, 1] = np.where(is_exp, mu[0] + (1 - wt) / (1 - pi) * (nu - mu[0]),
                                  rho / (1 - rho) * (1 - gamma) / (1 - pi) * (yy[test] - nu))
    theta = -psi_b.sum(axis=0) / psi_a.sum(axis=0)
    psi = psi_a * theta + psi_b
    j = psi_a.mean(axis=0)
    comb = psi[:, 0] / j[0] - psi[:, 1] / j[1]
    return theta[0] - theta[1], np.sqrt(comb @ comb / n**2), theta


def stacked_folds(w, n_obs, k=4, seed=3):
    strata = np.concatenate([np.where(w == 1, 0, 1), np.full(n_obs, 2)])
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(strata, strata))


@pytest.mark.parametrize("propensity", [None, 0.5])
def test_matches_reference_with_random_forests(propensity):
    data = simulate()
    xe, se, w, xo, so, y = data
    folds = stacked_folds(w, len(y))
    rf = dict(n_estimators=25, min_samples_leaf=10, random_state=0)
    ml_g, ml_m = RandomForestRegressor(**rf), RandomForestClassifier(**rf)
    model = dml_rs.SurrogateIndex(ml_g, ml_m, propensity=propensity).fit(*data, folds=folds)
    effect, se_ref, theta = reference(*data, ml_g, ml_m, folds, propensity)
    np.testing.assert_allclose(model.coef_, [effect], rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.se_, [se_ref], rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.potential_outcomes().coef_, theta, rtol=0, atol=1e-10)
    assert model.predictions_["ml_g"].shape == (len(w) + len(y), 1, 1)


def test_close_to_plug_in_surrogate_index():
    # Athey et al.'s plug-in: predict the index in the experiment, compare arm means
    # (randomised experiment, so no covariate adjustment is needed).
    rng = np.random.default_rng(5)
    n_exp, n_obs = 3000, 4000
    xe = rng.normal(size=(n_exp, 2))
    w = rng.integers(0, 2, n_exp).astype(float)
    xo = rng.normal(size=(n_obs, 2))
    def surrogates(x, w):
        return np.column_stack([w + 0.5 * x[:, 0] + rng.normal(size=len(x)),
                                0.5 * w - 0.3 * x[:, 1] + rng.normal(size=len(x))])
    se, so = surrogates(xe, w), surrogates(xo, np.zeros(n_obs))
    y = 2 * so[:, 0] + so[:, 1] + xo[:, 1] + rng.normal(size=n_obs)
    index = LinearRegression().fit(np.column_stack([so, xo]), y).predict(np.column_stack([se, xe]))
    plug_in = index[w == 1].mean() - index[w == 0].mean()
    model = dml_rs.SurrogateIndex(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf),
                                  propensity=0.5).fit(xe, se, w, xo, so, y)
    assert abs(model.coef_[0] - plug_in) < 0.5 * model.se_[0]
    assert abs(model.coef_[0] - EFFECT) < 4 * model.se_[0]


def test_coverage():
    cover = 0
    reps = 60
    for rep in range(reps):
        model = dml_rs.SurrogateIndex(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf),
                                      random_state=rep).fit(*simulate(seed=100 + rep))
        lo, hi = model.confint()[0]
        cover += lo <= EFFECT <= hi
    assert cover / reps > 0.85


def test_bootstrap_and_pickle():
    model = dml_rs.SurrogateIndex(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf), n_rep=2)
    model.fit(*simulate(seed=1))
    assert list(model.summary.index) == ["ATE"]
    assert list(model.potential_outcomes().summary.index) == ["Y(1)", "Y(0)"]
    model.bootstrap(n_boot=200)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.coef_, model.coef_)
    np.testing.assert_array_equal(restored.potential_outcomes().coef_,
                                  model.potential_outcomes().coef_)


def test_invalid_input():
    xe, se, w, xo, so, y = simulate(100, 100)
    model = dml_rs.SurrogateIndex(dml_rs.Ols(), dml_rs.LogisticRegression())
    with pytest.raises(ValueError, match="only 0 and 1"):
        model.fit(xe, se, w + 0.5, xo, so, y)
    with pytest.raises(ValueError, match="same covariate"):
        model.fit(xe, se, w, xo[:, :1], so, y)
    with pytest.raises(ValueError, match="propensity"):
        dml_rs.SurrogateIndex(dml_rs.Ols(), dml_rs.LogisticRegression(), propensity=1.2).fit(
            xe, se, w, xo, so, y)
    with pytest.raises(ValueError, match="NaN"):
        model.fit(xe, se, w, xo, so, np.full(len(y), np.nan))

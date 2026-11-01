"""Conformal intervals for counterfactuals and individual effects (Lei and Candès), checked
against a NumPy implementation of weighted split conformal prediction on the same split and in
simulations of the coverage guarantee."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs


def draw(n, rng):
    x = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-x[:, 0]))
    d = (rng.random(n) < e).astype(float)
    sd = 0.5 + np.abs(x[:, 1])
    y0 = x[:, 0] + sd * rng.normal(size=n)
    y1 = x[:, 0] + 1 + x[:, 1] + sd * rng.normal(size=n)
    return x, np.where(d == 1, y1, y0), d, y0, y1


def weight(arm, target, e):
    if arm == 1:
        return {"all": 1 / e, "treated": np.ones_like(e), "control": (1 - e) / e}[target]
    return {"all": 1 / (1 - e), "treated": e / (1 - e), "control": np.ones_like(e)}[target]


def reference(x, y, d, train, calib, xt, arm, target, alpha, make_band, ml_m, t=0.01):
    e_model = clone(ml_m).fit(x[train], d[train])
    tr, ca = train[d[train] == arm], calib[d[calib] == arm]
    band = make_band(x[tr], y[tr])
    lo_c, hi_c = band(x[ca])
    scores = np.maximum(lo_c - y[ca], y[ca] - hi_c)
    w_cal = weight(arm, target, np.clip(e_model.predict_proba(x[ca])[:, 1], t, 1 - t))
    order = np.argsort(scores, kind="stable")
    s, cum = scores[order], np.cumsum(w_cal[order])
    w_x = weight(arm, target, np.clip(e_model.predict_proba(xt)[:, 1], t, 1 - t))
    k = np.searchsorted(cum, (1 - alpha) * (cum[-1] + w_x), side="left")
    q = np.where(k < len(s), s[np.minimum(k, len(s) - 1)], np.inf)
    lo, hi = band(xt)
    return np.column_stack([lo - q, hi + q])


def residual_band(x, y):
    m = LinearRegression().fit(x, y)
    return lambda z: (m.predict(z), m.predict(z))


def quantile_band(x, y):
    lo = GradientBoostingRegressor(loss="quantile", alpha=0.05, n_estimators=50, random_state=0).fit(x, y)
    hi = GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=50, random_state=0).fit(x, y)
    return lambda z: (lo.predict(z), hi.predict(z))


@pytest.mark.parametrize("score", ["residual", "quantile"])
@pytest.mark.parametrize("target", ["all", "treated", "control"])
def test_matches_reference(score, target):
    rng = np.random.default_rng(0)
    x, y, d, _, _ = draw(800, rng)
    xt = rng.normal(size=(50, 2))
    perm = rng.permutation(len(y))
    train, calib = np.sort(perm[:600]), np.sort(perm[600:])
    ml_m = LogisticRegression()
    if score == "residual":
        model = dml_rs.ConformalITE(ml_m, ml_y=LinearRegression())
        band = residual_band
    else:
        gb = dict(loss="quantile", n_estimators=50, random_state=0)
        model = dml_rs.ConformalITE(ml_m, ml_lower=GradientBoostingRegressor(alpha=0.05, **gb),
                                    ml_upper=GradientBoostingRegressor(alpha=0.95, **gb))
        band = quantile_band
    model.fit(x, y, d, split=(train, calib))
    for arm in (0, 1):
        ours = model.predict_counterfactual(xt, arm, target, alpha=0.1)
        ref = reference(x, y, d, train, calib, xt, arm, target, 0.1, band, ml_m)
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-10)
    if target == "all":
        one = reference(x, y, d, train, calib, xt, 1, "all", 0.05, band, ml_m)
        zero = reference(x, y, d, train, calib, xt, 0, "all", 0.05, band, ml_m)
        expected = np.column_stack([one[:, 0] - zero[:, 1], one[:, 1] - zero[:, 0]])
        np.testing.assert_allclose(model.predict_ite(xt, alpha=0.1), expected, rtol=0, atol=1e-10)


def test_coverage_with_known_propensity():
    # Randomised experiment: coverage holds in finite samples.
    rng = np.random.default_rng(1)
    hits_y1, hits_ite, total = 0, 0, 0
    for rep in range(30):
        n = 1000
        x = rng.normal(size=(n, 2))
        d = (rng.random(n) < 0.5).astype(float)
        y0 = x[:, 0] + rng.normal(size=n)
        y1 = x[:, 0] + 1 + x[:, 1] + rng.normal(size=n)
        model = dml_rs.ConformalITE(ml_y=dml_rs.Ols(), propensity=0.5, random_state=rep)
        model.fit(x, np.where(d == 1, y1, y0), d)
        xt = rng.normal(size=(400, 2))
        t0 = xt[:, 0] + rng.normal(size=400)
        t1 = xt[:, 0] + 1 + xt[:, 1] + rng.normal(size=400)
        cf = model.predict_counterfactual(xt, 1, alpha=0.1)
        ite = model.predict_ite(xt, alpha=0.1)
        hits_y1 += np.sum((cf[:, 0] <= t1) & (t1 <= cf[:, 1]))
        hits_ite += np.sum((ite[:, 0] <= t1 - t0) & (t1 - t0 <= ite[:, 1]))
        total += 400
    assert 0.885 < hits_y1 / total < 0.92
    assert hits_ite / total > 0.9


def test_data_input_pickle_and_errors():
    rng = np.random.default_rng(2)
    x, y, d, _, _ = draw(400, rng)
    data = dml_rs.DmlData.from_arrays(x, y, d)
    model = dml_rs.ConformalITE(dml_rs.LogisticRegression(), ml_y=dml_rs.Ols()).fit(data)
    assert sum(model.n_calibration_) == 100
    assert model.predict_ite(x[:3]).shape == (3, 2)
    with pytest.raises(TypeError, match="not picklable"):
        pickle.dumps(model)
    pickle.loads(pickle.dumps(dml_rs.ConformalITE(ml_y=dml_rs.Ols(), propensity=0.5)))
    with pytest.raises(ValueError, match="ml_y"):
        dml_rs.ConformalITE(dml_rs.LogisticRegression()).fit(x, y, d)
    with pytest.raises(ValueError, match="propensity learner"):
        dml_rs.ConformalITE(ml_y=dml_rs.Ols()).fit(x, y, d)
    with pytest.raises(ValueError, match="target"):
        model.predict_counterfactual(x[:2], 1, target="everyone")
    with pytest.raises(ValueError, match="columns"):
        model.predict_ite(x[:2, :1])

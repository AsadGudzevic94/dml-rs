"""Difference-in-differences with a continuous treatment: parity with R's contdid without
covariates (fixture from parity/generate_contdid_fixtures.R) and with a NumPy computation of the
doubly robust estimator with covariates on the same folds."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold

import dml_rs

FIXTURE = Path(__file__).parents[2] / "dml-core" / "tests" / "fixtures" / "contdid_r.json"
CASES = {k: v for k, v in json.loads(FIXTURE.read_text()).items() if not k.startswith("_")}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_contdid(name):
    case = CASES[name]
    model = dml_rs.ContinuousDID(degree=case["degree"], num_knots=case["num_knots"],
                                 grid=case["grid"]).fit(None, case["y"], case["d"])
    np.testing.assert_allclose(model.coef_, case["att"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(model.se_, case["att_se"], rtol=0, atol=1e-9)
    acrt = model.acrt()
    np.testing.assert_allclose(acrt.coef_, case["acrt"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(acrt.se_, case["acrt_se"], rtol=0, atol=1e-9)
    overall = model.overall()
    np.testing.assert_allclose(overall.coef_, [case["overall_att"], case["overall_acrt"]],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(overall.se_, [case["overall_att_se"], case["overall_acrt_se"]],
                               rtol=0, atol=1e-9)


def basis(d, degree, lo, hi):
    """[1, B_2(d), ..., B_K(d)]: cubic B-splines without interior knots, first one dropped."""
    knots = np.r_[[lo] * (degree + 1), [hi] * (degree + 1)]
    k = len(knots) - degree - 1
    cols = [BSpline(knots, np.eye(k)[j], degree, extrapolate=False)(d) for j in range(1, k)]
    return np.column_stack([np.ones(len(d))] + [np.nan_to_num(c) for c in cols])


def simulate(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    treated = rng.random(n) < 1 / (1 + np.exp(-0.8 * x[:, 0]))
    d = np.where(treated, np.clip(0.2 + 0.6 * rng.random(n) + 0.1 * x[:, 0], 0.05, 1), 0.0)
    y = 1 + x[:, 0] - 0.5 * x[:, 1] + d + d**2 + rng.normal(size=n)
    return x, y, d


def test_matches_reference_with_covariates():
    x, y, d = simulate()
    n = len(y)
    t = (d > 0).astype(float)
    folds = list(StratifiedKFold(4, shuffle=True, random_state=2).split(x, t))
    rf = dict(n_estimators=25, min_samples_leaf=10, random_state=0)
    ml_g, ml_m = RandomForestRegressor(**rf), RandomForestClassifier(**rf)
    grid = [0.3, 0.5, 0.7]
    model = dml_rs.ContinuousDID(ml_g, ml_m, grid=grid).fit(x, y, d, folds=folds)

    lo, hi = d[d > 0].min(), d[d > 0].max()
    B = np.zeros((n, 4))
    B[d > 0] = basis(d[d > 0], 3, lo, hi)
    m0, p, bbar = np.empty(n), np.empty(n), np.ones((n, 4))
    for train, test in folds:
        ctrl, trt = train[d[train] == 0], train[d[train] > 0]
        m0[test] = clone(ml_g).fit(x[ctrl], y[ctrl]).predict(x[test])
        p[test] = clone(ml_m).fit(x[train], t[train]).predict_proba(x[test])[:, 1]
        for j in range(1, 4):
            bbar[test, j] = clone(ml_g).fit(x[trt], B[trt, j]).predict(x[test])
    w = np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99))
    r = y - m0
    terms = np.where(t[:, None] == 1, B * r[:, None], -bbar * (w * r)[:, None])
    Q = (B * t[:, None]).T @ B / n
    beta = np.linalg.solve(Q, terms.mean(axis=0))
    phi = (terms - t[:, None] * B * (B @ beta)[:, None]) @ np.linalg.inv(Q).T
    G = basis(np.array(grid), 3, lo, hi)
    np.testing.assert_allclose(model.coef_, G @ beta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.se_, np.sqrt(((phi @ G.T) ** 2).sum(axis=0)) / n,
                               rtol=0, atol=1e-10)
    # The overall ATT is the doubly robust DiD estimate.
    att_dr = (t * r - (1 - t) * w * r).sum() / t.sum()
    np.testing.assert_allclose(model.overall().coef_[0], att_dr, rtol=0, atol=1e-10)


def test_overall_acrt_standard_error_is_calibrated():
    # Without covariates the estimates equal contdid's; the standard error of the overall
    # ACRT should match the sampling spread (contdid's own is smaller by n1 / n).
    est, se = [], []
    for rep in range(300):
        rng = np.random.default_rng(rep)
        n = 1000
        treated = rng.random(n) < 0.6
        d = np.where(treated, rng.uniform(0.1, 1, n), 0.0)
        y = 0.5 + d + d**2 + rng.normal(size=n)
        overall = dml_rs.ContinuousDID().fit(None, y, d).overall()
        est.append(overall.coef_[1])
        se.append(overall.se_[1])
    ratio = np.mean(se) / np.std(est)
    assert 0.85 < ratio < 1.15, ratio


def test_coverage_with_covariates():
    cover = np.zeros(2)
    reps = 50
    for rep in range(reps):
        x, y, d = simulate(seed=100 + rep)
        model = dml_rs.ContinuousDID(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf),
                                     grid=[0.5], random_state=rep).fit(x, y, d)
        lo, hi = model.confint()[0]
        cover[0] += lo <= 0.75 <= hi
        lo, hi = model.acrt().confint()[0]
        cover[1] += lo <= 2.0 <= hi
    assert np.all(cover / reps > 0.85), cover / reps


def test_plot_bootstrap_pickle_and_data():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    x, y, d = simulate(800, seed=4)
    model = dml_rs.ContinuousDID(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf)).fit(x, y, d)
    assert len(model.grid_) == 19
    assert model.summary.index[0].startswith("ATT(")
    assert list(model.overall().summary.index) == ["ATT", "ACRT"]
    model.bootstrap(n_boot=200)
    joint, pointwise = model.confint(joint=True), model.confint()
    assert np.all(joint[:, 0] < pointwise[:, 0])
    assert model.plot(joint=True).get_xlabel() == "dose d"
    model.plot(kind="acrt")
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.acrt().coef_, model.acrt().coef_)
    data = dml_rs.DmlData.from_arrays(x, y, d)
    again = dml_rs.ContinuousDID(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf)).fit(data)
    np.testing.assert_array_equal(again.coef_, model.coef_)


def test_invalid_input():
    x, y, d = simulate(300)
    with pytest.raises(ValueError, match="non-negative"):
        dml_rs.ContinuousDID().fit(None, y, d - 0.5)
    with pytest.raises(ValueError, match="needs learners"):
        dml_rs.ContinuousDID().fit(x, y, d)
    with pytest.raises(ValueError, match="both ml_g and ml_m"):
        dml_rs.ContinuousDID(ml_g=dml_rs.Ols()).fit(x, y, d)
    with pytest.raises(ValueError, match="range of the positive doses"):
        dml_rs.ContinuousDID(grid=[5.0]).fit(None, y, d)
    with pytest.raises(ValueError, match="untreated"):
        dml_rs.ContinuousDID().fit(None, y, np.ones(len(y)))

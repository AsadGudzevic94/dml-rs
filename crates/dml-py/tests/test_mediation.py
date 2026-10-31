"""Causal mediation analysis (Farbmacher et al., 2022): parity with R's causalweight::medDML
(fixture from parity/generate_mediation_fixtures.R) and with a NumPy / scikit-learn
computation of the same estimator on the same folds."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold

import dml_rs

FIXTURE = Path(__file__).parents[2] / "dml-core" / "tests" / "fixtures" / "mediation_causalweight.json"
CASES = {k: v for k, v in json.loads(FIXTURE.read_text()).items() if not k.startswith("_")}


def kfold(test_sets, n):
    return [(np.setdiff1d(np.arange(n), t), np.asarray(t)) for t in test_sets]


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_med_dml(name):
    case = CASES[name]
    y, d, m, x = (np.asarray(case[k], dtype=float) for k in ("y", "d", "m", "x"))
    opt = case["options"]
    model = dml_rs.Mediation(
        dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf), trimming_rule="drop",
        trimming_threshold=opt["trim"],
        normalize_ipw=opt["normalized"], nested_split=not opt["fewsplits"],
        debias_nested=opt["debiasfirst"],
    ).fit(x, y, d, m, folds=kfold(case["folds"], len(y)))
    # medDML columns: total, dir.treat, dir.control, indir.treat, indir.control, Y(0,M(0)).
    po = model.potential_outcomes()
    np.testing.assert_allclose(np.append(model.coef_, po.coef_[3]), case["effect"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.append(model.se_, po.se_[3]), case["se"], rtol=0, atol=1e-9)
    assert model.n_trimmed_ == [case["ntrimmed"]]


def simulate(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    d = (0.5 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(size=n) > 0).astype(float)
    m = np.column_stack([0.7 * d + 0.5 * x[:, 0] + rng.normal(size=n),
                         (0.5 * d + x[:, 2] + rng.normal(size=n) > 0).astype(float)])
    y = 0.5 * d + m[:, 0] + 0.5 * m[:, 1] + 0.3 * x[:, 1] + rng.normal(size=n)
    return x, y, d, m


def reference(x, y, d, m, ml_g, ml_m, folds, rule, trim):
    """The estimator written out (normalised weights, nested split, debiased nested means)."""
    n = len(y)
    xm = np.column_stack([x, m])
    k = len(folds)
    test_sets = [t for _, t in folds]
    cols = {c: np.empty(n) for c in ("p", "q", "g1", "g0", "mu1", "mu0", "om1", "om0", "tau1", "tau0")}

    def reg(X, rows, target, at):
        return clone(ml_g).fit(X[rows], target).predict(X[at])

    def prob(X, rows, at):
        return clone(ml_m).fit(X[rows], d[rows]).predict_proba(X[at])[:, 1]

    for f, (train, test) in enumerate(folds):
        mu_rows = np.concatenate([test_sets[(f + o) % k] for o in range(1, 1 + k // 2)])
        delta = np.concatenate([test_sets[(f + o) % k] for o in range(1 + k // 2, k)])
        cols["p"][test] = prob(x, train, test)
        cols["q"][test] = prob(xm, train, test)
        cols["g1"][test] = reg(x, train[d[train] == 1], y[train[d[train] == 1]], test)
        cols["g0"][test] = reg(x, train[d[train] == 0], y[train[d[train] == 0]], test)
        pa = np.clip(prob(x, mu_rows, delta), trim, 1 - trim)
        qa = np.clip(prob(xm, mu_rows, delta), trim, 1 - trim)
        odds = pa / (1 - pa) * (1 - qa) / qa
        for arm in (1, 0):
            rows = mu_rows[d[mu_rows] == arm]
            mu = clone(ml_g).fit(xm[rows], y[rows])
            own, other = d[delta] == arm, d[delta] != arm
            om = clone(ml_g).fit(x[delta[other]], mu.predict(xm[delta[other]]))
            w = odds[own] if arm == 1 else 1 / odds[own]
            target = om.predict(x[delta[own]]) + w * (y[delta[own]] - mu.predict(xm[delta[own]]))
            cols[f"mu{arm}"][test] = mu.predict(xm[test])
            cols[f"om{arm}"][test] = om.predict(x[test])
            cols[f"tau{arm}"][test] = reg(x, delta[own], target, test)
    c = cols
    p, q = c["p"], c["q"]
    if rule == "drop":
        keep = ((1 - q) * p >= trim) & (1 - p >= trim) & (p >= trim) & (q * (1 - p) >= trim)
    else:
        p, q = np.clip(p, trim, 1 - trim), np.clip(q, trim, 1 - trim)
        keep = np.ones(n, dtype=bool)
    a = d * (1 - q) / (q * (1 - p))
    e = (1 - d) * q / ((1 - q) * p)
    wc = d / p
    wb = (1 - d) / (1 - p)
    a, e, wc, wb = (w / w[keep].mean() for w in (a, e, wc, wb))
    y11 = c["g1"] + wc * (y - c["g1"])
    y10 = a * (y - c["mu1"]) + wb * (c["mu1"] - c["om1"]) + wc * (c["om1"] - c["tau1"]) + c["tau1"]
    y01 = e * (y - c["mu0"]) + wc * (c["mu0"] - c["om0"]) + wb * (c["om0"] - c["tau0"]) + c["tau0"]
    y00 = c["g0"] + wb * (y - c["g0"])
    po = np.column_stack([y11, y10, y01, y00])[keep]
    effects = po @ np.array([[1, 1, 0, 1, 0], [0, 0, 1, -1, 0], [0, -1, 0, 0, 1], [-1, 0, -1, 0, -1]])
    coef = effects.mean(axis=0)
    se = np.sqrt(((effects - coef) ** 2).mean(axis=0) / keep.sum())
    return coef, se, (~keep).sum()


@pytest.mark.parametrize("rule, trim", [("truncate", 0.01), ("drop", 0.05)])
def test_matches_reference_with_random_forests(rule, trim):
    x, y, d, m = simulate()
    folds = list(KFold(4, shuffle=True, random_state=1).split(x))
    rf = dict(n_estimators=25, min_samples_leaf=10, random_state=0)
    ml_g, ml_m = RandomForestRegressor(**rf), RandomForestClassifier(**rf)
    model = dml_rs.Mediation(ml_g, ml_m, trimming_rule=rule, trimming_threshold=trim).fit(
        x, y, d, m, folds=folds)
    coef, se, trimmed = reference(x, y, d, m, ml_g, ml_m, folds, rule, trim)
    np.testing.assert_allclose(model.coef_, coef, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.se_, se, rtol=0, atol=1e-10)
    assert model.n_trimmed_ == [trimmed]
    assert list(model.summary.index) == ["total", "direct (treated)", "direct (control)",
                                         "indirect (treated)", "indirect (control)"]
    assert model.predictions_["ml_nested_debiased_d1"].shape == (len(y), 1, 1)


def test_data_bootstrap_and_pickle():
    pd = pytest.importorskip("pandas")
    x, y, d, m = simulate(800, seed=2)
    data = dml_rs.DmlData.from_arrays(x, y, d, m=m)
    assert data.m_cols == ["m1", "m2"]
    model = dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf), n_rep=2).fit(data)
    again = dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf), n_rep=2).fit(
        pd.DataFrame(x), y, d, m)
    np.testing.assert_array_equal(model.coef_, again.coef_)
    po = model.potential_outcomes()
    assert list(po.summary.index) == ["Y(1, M(1))", "Y(1, M(0))", "Y(0, M(1))", "Y(0, M(0))"]
    model.bootstrap(n_boot=300)
    joint, pointwise = model.confint(joint=True), model.confint()
    assert np.all(joint[:, 0] < pointwise[:, 0])
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.coef_, model.coef_)
    np.testing.assert_array_equal(restored.potential_outcomes().coef_, po.coef_)
    with pytest.raises(ValueError, match="m_cols"):
        dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression()).fit(dml_rs.DmlData.from_arrays(x, y, d))


def test_invalid_input():
    x, y, d, m = simulate(200)
    model = dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression())
    with pytest.raises(ValueError, match="binary"):
        model.fit(x, y, y, m)
    with pytest.raises(ValueError, match="at least 3 folds"):
        dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression(), n_folds=2).fit(x, y, d, m)
    with pytest.raises(ValueError, match="partition"):
        model.fit(x, y, d, m, folds=[(np.arange(100), np.arange(100, 200))])
    with pytest.raises(ValueError, match="trimming_rule"):
        dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression(), trimming_rule="cut").fit(x, y, d, m)
    with pytest.raises(ValueError, match="trimming_threshold"):
        dml_rs.Mediation(dml_rs.Ols(), dml_rs.LogisticRegression(), trimming_threshold=0.7).fit(x, y, d, m)

"""Synthetic difference-in-differences: parity with R's synthdid (fixture from
parity/generate_synthdid_fixtures.R), long-panel input and the standard errors."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import dml_rs

FIXTURE = Path(__file__).parents[2] / "dml-core" / "tests" / "fixtures" / "synthdid_r.json"
CASES = {k: v for k, v in json.loads(FIXTURE.read_text()).items() if not k.startswith("_")}


def covariates(case):
    if "x" not in case:
        return None
    return np.stack([np.asarray(xk) for xk in case["x"]], axis=2)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_synthdid(name):
    case = CASES[name]
    model = dml_rs.SyntheticDID(estimator=case["estimator"], se_method=None)
    model.fit(np.asarray(case["y"]), case["n0"], case["t0"], covariates(case))
    np.testing.assert_allclose(model.coef_, [case["estimate"]], rtol=0, atol=1e-9)
    np.testing.assert_allclose(model.unit_weights_, case["omega"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(model.time_weights_, case["lambda"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(model.beta_, case["beta"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(model.effect_curve_, case["effect_curve"], rtol=0, atol=1e-9)
    if case["se_jackknife"] is None:
        with pytest.raises(ValueError, match="jackknife"):
            model.se("jackknife")
    else:
        assert model.se("jackknife") == pytest.approx(case["se_jackknife"], abs=1e-9)
    placebo = case["placebo"]
    if placebo["se"] is not None:
        se = model.se("placebo", placebo["replications"], placebo["seed"])
        assert se == pytest.approx(placebo["se"], abs=1e-9)
    if "bootstrap" in case and case["bootstrap"]["se"] is not None:
        boot = case["bootstrap"]
        assert model.se("bootstrap", boot["replications"], boot["seed"]) == pytest.approx(boot["se"], abs=1e-9)


def test_fit_panel_matches_matrix_input():
    pd = pytest.importorskip("pandas")
    case = CASES["sim_sdid"]
    y = np.asarray(case["y"])
    n, t = y.shape
    units = [f"u{i:02d}" for i in range(n)]
    rows = [(units[i], 2000 + s, y[i, s], int(i >= case["n0"] and s >= case["t0"]))
            for i in range(n) for s in range(t)]
    df = pd.DataFrame(rows, columns=["unit", "year", "y", "treated"]).sample(frac=1, random_state=0)
    model = dml_rs.SyntheticDID(se_method="jackknife").fit_panel(df, "unit", "year", "y", "treated")
    assert model.coef_[0] == pytest.approx(case["estimate"], abs=1e-9)
    assert model.se_[0] == pytest.approx(case["se_jackknife"], abs=1e-9)
    assert list(model.unit_weights_.index) == units[: case["n0"]]
    assert list(model.time_weights_.index) == list(range(2000, 2000 + case["t0"]))
    assert list(model.summary.index) == ["synthetic DiD"]
    with pytest.raises(ValueError, match="simultaneous"):
        bad = df.copy()
        bad.loc[(bad.unit == units[-1]) & (bad.year == 2000 + case["t0"]), "treated"] = 0
        dml_rs.SyntheticDID().fit_panel(bad, "unit", "year", "y", "treated")
    with pytest.raises(ValueError, match="balanced"):
        dml_rs.SyntheticDID().fit_panel(df.iloc[1:], "unit", "year", "y", "treated")


def test_single_treated_unit_and_pickle():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    case = CASES["prop99_sdid"]
    model = dml_rs.SyntheticDID(replications=50, random_state=12345)
    model.fit(np.asarray(case["y"]), case["n0"], case["t0"])
    assert model.se_[0] == pytest.approx(case["placebo"]["se"], abs=1e-9)
    lo, hi = model.confint()[0]
    assert lo < model.coef_[0] < hi
    with pytest.warns(UserWarning, match="bootstrap"):
        dml_rs.SyntheticDID(se_method="bootstrap").fit(np.asarray(case["y"]), case["n0"], case["t0"])
    restored = pickle.loads(pickle.dumps(model))
    assert restored.coef_[0] == model.coef_[0]
    assert restored.se("placebo", 20, 1) == model.se("placebo", 20, 1)
    assert model.plot().get_ylabel() == "outcome"


def test_invalid_input():
    y = np.zeros((5, 4))
    with pytest.raises(ValueError, match="estimator"):
        dml_rs.SyntheticDID(estimator="ols").fit(y, 3, 2)
    with pytest.raises(ValueError, match="control units"):
        dml_rs.SyntheticDID().fit(y, 5, 2)
    model = dml_rs.SyntheticDID(se_method=None, noise_level=1.0).fit(y + np.arange(4), 3, 2)
    with pytest.raises(ValueError, match="method"):
        model.se("delta")

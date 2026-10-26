import urllib.error
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs
from dml_rs import datasets as ds

LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


# -- DmlData ---------------------------------------------------------------------

def test_dml_data_roles_defaults_and_errors():
    df = ds.make_plr_CCDDHNR2018(n_obs=100, dim_x=4, return_type="DataFrame", random_state=1)
    data = dml_rs.DmlData(df, "y", "d")
    assert data.x_cols == ["X1", "X2", "X3", "X4"] and data.n_obs == 100 and data.n_treat == 1
    assert list(data.x.columns) == data.x_cols and data.y.name == "y" and data.z is None
    assert "x_cols" in repr(data)
    with pytest.raises(ValueError, match="not in the data"):
        dml_rs.DmlData(df, "y", "dd")
    with pytest.raises(ValueError, match="two roles"):
        dml_rs.DmlData(df, "y", "y")
    with pytest.raises(ValueError, match="another role"):
        dml_rs.DmlData(df, "y", "d", x_cols=["X1", "d"])
    with pytest.raises(TypeError, match="DataFrame"):
        dml_rs.DmlData(df.to_numpy(), "y", "d")  # type: ignore[arg-type]
    x, y, d, z = ds.make_pliv_CHS2015(n_obs=50, dim_x=6, dim_z=2, random_state=2)
    arrays = dml_rs.DmlData.from_arrays(x, y, d, z=z, cluster_vars=np.arange(50) % 5)
    assert arrays.z_cols == ["z1", "z2"] and arrays.cluster_cols == ["cluster_var"]
    assert arrays.x_cols == [f"X{i}" for i in range(1, 7)]


def assert_same_fit(a, b):
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-12)
    np.testing.assert_allclose(a.se_, b.se_, rtol=1e-12)


def test_estimators_accept_dml_data():
    x, y, d = ds.make_irm_data(n_obs=400, dim_x=4, theta=0.5, random_state=3)
    cluster = np.arange(400) % 40
    data = dml_rs.DmlData.from_arrays(x, y, d, cluster_vars=cluster)
    ols, logit = dml_rs.Ols(), dml_rs.LogisticRegression()
    for model in [dml_rs.PLR(ols, ols), dml_rs.IRM(ols, logit)]:
        a = type(model)(**model.get_params()).fit(data)
        b = type(model)(**model.get_params()).fit(x, y, d, cluster=cluster)
        assert_same_fit(a, b)
        assert a.n_clusters_ == 40
    plain = dml_rs.DmlData.from_arrays(x, y, d)
    for model in [dml_rs.SLearner(ols), dml_rs.PQ(LOGIT, LOGIT), dml_rs.LPLR(LOGIT, ols, ols)]:
        if isinstance(model, dml_rs.LPLR):
            yb = (y > np.median(y)).astype(float)
            a = model.fit(dml_rs.DmlData.from_arrays(x, yb, d))
            b = type(model)(**model.get_params()).fit(x, yb, d)
            assert_same_fit(a, b)
        elif isinstance(model, dml_rs.SLearner):
            np.testing.assert_allclose(model.fit(plain).effect(x), dml_rs.SLearner(ols).fit(x, y, d).effect(x))
        else:
            assert_same_fit(model.fit(plain), type(model)(**model.get_params()).fit(x, y, d))
    with pytest.raises(ValueError, match="does not support clustered"):
        dml_rs.PQ(LOGIT, LOGIT).fit(data)
    with pytest.raises(ValueError, match="needs the DmlData to have z_cols"):
        dml_rs.IIVM(ols, logit, logit).fit(plain)
    with pytest.raises(TypeError, match="by keyword"):
        dml_rs.PLR(ols, ols).fit(plain, None)


def test_dml_data_for_instruments_panels_and_rd():
    ols, logit = dml_rs.Ols(), dml_rs.LogisticRegression()
    data = ds.make_iivm_data(n_obs=400, dim_x=4, return_type="DmlData", random_state=4)
    assert data.z_cols == ["z"]
    x, y, d, z = ds.make_iivm_data(n_obs=400, dim_x=4, random_state=4)
    assert_same_fit(dml_rs.IIVM(ols, logit, logit).fit(data), dml_rs.IIVM(ols, logit, logit).fit(x, y, d, z))
    ssm = ds.make_ssm_data(n_obs=500, dim_x=5, mar=False, return_type="DmlData", random_state=5)
    assert ssm.s_col == "s" and ssm.z_cols == ["z"]
    fit = dml_rs.SSM(ols, logit, logit, score="nonignorable").fit(ssm)
    assert np.isfinite(fit.coef_[0])
    cs = ds.make_did_SZ2020(n_obs=400, cross_sectional_data=True, return_type="DmlData", random_state=6)
    assert cs.t_col == "t"
    assert np.isfinite(dml_rs.DIDCS(ols, logit).fit(cs).coef_[0])
    panel = ds.make_plpr_CP2025(num_id=50, num_t=4, dim_x=4, random_state=7)
    pdata = dml_rs.DmlData(panel, "y", "d", id_col="id", t_col="time")
    a = dml_rs.PLPR(ols, ols).fit(pdata)
    b = dml_rs.PLPR(ols, ols).fit(panel[[f"x{i}" for i in range(1, 5)]], panel["y"], panel["d"], panel["id"],
                                  panel["time"])
    assert_same_fit(a, b)
    stag = ds.make_did_CS2021(n_obs=400, random_state=8)
    sdata = dml_rs.DmlData(stag, "y", "d", x_cols=["Z1", "Z2", "Z3", "Z4"], id_col="id", t_col="t")
    a = dml_rs.StaggeredDID(ols, logit).fit(sdata)
    b = dml_rs.StaggeredDID(ols, logit).fit_frame(stag, "y", "id", "t", "d", ["Z1", "Z2", "Z3", "Z4"])
    assert_same_fit(a, b)
    rdd = ds.make_simple_rdd_data(n_obs=1000, fuzzy=False, random_state=9)
    rdata = dml_rs.DmlData.from_arrays(rdd["X"], rdd["Y"], rdd["D"], score=rdd["score"])
    a = dml_rs.RDFlex(LinearRegression()).fit(rdata)
    b = dml_rs.RDFlex(LinearRegression()).fit(rdd["X"], rdd["Y"], rdd["D"], rdd["score"])
    np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-12)
    multi = ds.make_pliv_multiway_cluster_CKMS2021(N=8, M=9, dim_X=5, return_type="DmlData", random_state=10)
    assert multi.cluster_cols == ["cluster_var_i", "cluster_var_j"] and multi.y_col == "Y"
    assert dml_rs.PLIV(ols, ols, ols).fit(multi).n_clusters_ is not None


# -- plots -----------------------------------------------------------------------

def test_staggered_and_aggregation_plots():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    df = ds.make_did_CS2021(n_obs=600, anticipation_periods=1, random_state=11)
    model = dml_rs.StaggeredDID(dml_rs.Ols(), dml_rs.LogisticRegression(), anticipation_periods=1).fit_frame(
        df, "y", "id", "t", "d", ["Z1", "Z2", "Z3", "Z4"])
    with pytest.warns(UserWarning, match="bootstrap"):
        fig, axes = model.plot_effects()
    assert len(axes) == len({g for g, _, _ in model.group_times_})
    assert fig.texts[0].get_text() == "Estimated ATTs by Group"
    with pytest.raises(ValueError, match="sensitivity"):
        model.plot_effects(result_type="rv")
    model.sensitivity_analysis()
    for kind in ("rv", "est_bounds", "ci_bounds"):
        model.plot_effects(result_type=kind)
    fig, ax = model.aggregate("eventstudy").plot_effects(sort_by="estimate", joint=False)
    assert ax.get_title() == "Aggregated Treatment Effects"
    with pytest.raises(ValueError, match="sort_by"):
        model.aggregate("group").plot_effects(sort_by="size")  # type: ignore[arg-type]
    matplotlib.pyplot.close("all")


def test_policy_tree_nodes_and_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    x, y, d = ds.make_irm_data(n_obs=1500, dim_x=3, theta=0.5, random_state=12)
    features = pd.DataFrame(x[:, :2], columns=["age", "income"])
    tree = dml_rs.IRM(dml_rs.Ols(), dml_rs.LogisticRegression()).fit(x, y, d).policy_tree(features, depth=2,
                                                                                            ccp_alpha=0.0)
    nodes = tree._native.nodes()
    leaves = [n for n in nodes if n[0] < 0]
    assert len(leaves) == tree.n_leaves
    # Walking the nodes reproduces predict.
    for row in x[:20, :2]:
        i = 0
        while nodes[i][0] >= 0:
            feature, threshold, left, right, _ = nodes[i]
            i = left if row[feature] <= threshold else right
        assert float(nodes[i][4] > 0.5) == tree.predict(row.reshape(1, -1))[0]
    ax = tree.plot_tree()
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("age <=") or t.startswith("income <=") for t in texts)
    assert sum("share positive" in t for t in texts) == tree.n_leaves
    matplotlib.pyplot.close("all")


# -- real data sets ----------------------------------------------------------------

def test_polynomial_features_match_scikit_learn():
    from sklearn.preprocessing import PolynomialFeatures

    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 4)), columns=["a", "b", "c", "dd"])
    ours = ds._polynomial_features(frame, list(frame.columns))
    poly = PolynomialFeatures(2, include_bias=False)
    theirs = poly.fit_transform(frame)
    assert list(ours.columns) == list(poly.get_feature_names_out(frame.columns))
    np.testing.assert_allclose(ours.to_numpy(), theirs, rtol=1e-15)


def test_bonus_preparation_matches_doubleml(monkeypatch):
    """DoubleML's preparation of the bonus data, on a synthetic raw file (no download)."""
    import importlib

    fb = importlib.import_module("doubleml.datasets.fetch_bonus")

    rng = np.random.default_rng(1)
    n = 60
    raw = pd.DataFrame({c: rng.integers(0, 2, n) for c in ds._BONUS_X if c not in ("dep1", "dep2")})
    raw["tg"] = rng.integers(0, 6, n)
    raw["dep"] = rng.integers(0, 3, n)
    raw["inuidur1"] = rng.integers(1, 30, n).astype(float)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: raw.copy())
    for poly in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = fb.fetch_bonus(return_type="DataFrame", polynomial_features=poly)
        ours = ds.fetch_bonus(return_type="DataFrame", polynomial_features=poly)
        pd.testing.assert_frame_equal(ours, theirs, check_dtype=False)
    data = ds.fetch_bonus()
    assert data.y_col == "inuidur1" and data.d_cols == ["tg"] and len(data.x_cols) == 15
    with pytest.raises(ValueError, match="return_type"):
        ds.fetch_bonus(return_type="array")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        ds.fetch_401K(polynomial_features=True)


def test_fetch_401k_matches_doubleml():
    """Downloads the data (skipped without internet access)."""
    from doubleml.datasets import fetch_401K

    try:
        ours = ds.fetch_401K(return_type="DataFrame")
        theirs = fetch_401K(return_type="DataFrame")
    except (urllib.error.URLError, OSError) as err:  # pragma: no cover - network
        pytest.skip(f"no network: {err}")
    pd.testing.assert_frame_equal(ours, theirs)
    data = ds.fetch_401K()
    assert data.y_col == "net_tfa" and data.d_cols == ["e401"] and data.n_obs == 9915

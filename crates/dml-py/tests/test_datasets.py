"""dml_rs.datasets against DoubleML's generators: the draws differ (another random generator),
so the tests compare the output format and the distributions on large samples."""

import warnings

import doubleml as dml
import numpy as np
import pandas as pd
import pytest
from doubleml.did.datasets import make_did_CS2021, make_did_cs_CS2021, make_did_SZ2020
from doubleml.irm.datasets import (
    make_confounded_irm_data,
    make_heterogeneous_data,
    make_iivm_data,
    make_irm_data,
    make_irm_data_discrete_treatments,
    make_ssm_data,
)
from doubleml.plm.datasets import (
    make_confounded_plr_data,
    make_lplr_LZZ2020,
    make_pliv_CHS2015,
    make_pliv_multiway_cluster_CKMS2021,
    make_plpr_CP2025,
    make_plr_CCDDHNR2018,
    make_plr_turrell2018,
)
from doubleml.rdd.datasets import make_simple_rdd_data
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs
from dml_rs import datasets as ds


def doubleml(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(0)
        return fn(*args, **kwargs)


def assert_similar(ours, theirs, name, k=6.0, rel_sd=0.06):
    """Means agree within ``k`` standard errors and standard deviations within ``rel_sd``."""
    ours, theirs = np.asarray(ours, dtype=float).ravel(), np.asarray(theirs, dtype=float).ravel()
    se = np.sqrt(ours.var() / ours.size + theirs.var() / theirs.size)
    assert abs(ours.mean() - theirs.mean()) <= k * se + 1e-9, f"{name}: mean {ours.mean()} vs {theirs.mean()}"
    assert abs(ours.std() - theirs.std()) <= rel_sd * theirs.std() + 1e-9, \
        f"{name}: sd {ours.std()} vs {theirs.std()}"


def assert_columns_similar(ours, theirs, prefix):
    for j in range(theirs.shape[1]):
        assert_similar(ours[:, j], theirs[:, j], f"{prefix}{j}")


N = 20_000


@pytest.mark.parametrize("ours_fn,theirs_fn,kwargs", [
    (ds.make_plr_CCDDHNR2018, make_plr_CCDDHNR2018, {"n_obs": N, "dim_x": 5}),
    (ds.make_irm_data, make_irm_data, {"n_obs": N, "dim_x": 5, "theta": 0.5}),
])
def test_partially_linear_and_irm_distributions(ours_fn, theirs_fn, kwargs):
    x, y, d = ours_fn(**kwargs, random_state=1)
    tx, ty, td = doubleml(theirs_fn, **kwargs, return_type="array")
    assert x.shape == tx.shape and y.shape == ty.shape
    assert_columns_similar(x, tx, "x")
    assert_similar(y, ty, "y")
    assert_similar(d, td, "d")
    assert_similar(y * d, ty * td, "y*d")
    frame = ours_fn(**{**kwargs, "n_obs": 50}, return_type="DataFrame", random_state=1)
    reference = doubleml(theirs_fn, **{**kwargs, "n_obs": 50}, return_type="DataFrame")
    assert list(frame.columns) == list(reference.columns)


def test_turrell_draws_its_own_covariance():
    x, y, d = ds.make_plr_turrell2018(n_obs=N, dim_x=5, random_state=2)
    tx, ty, td = doubleml(make_plr_turrell2018, n_obs=N, dim_x=5, return_type="array")
    # The covariance matrix itself is random, so only the treatment, whose regression is bounded
    # in [0.07, 0.43], has a common distribution.
    assert abs(d.mean() - td.mean()) < 0.05
    assert abs(d.std() - td.std()) < 0.05
    assert not np.allclose(np.cov(x.T), np.eye(5), atol=0.3)


def test_pliv_and_iivm_distributions():
    x, y, d, z = ds.make_pliv_CHS2015(n_obs=N, dim_x=10, dim_z=4, random_state=3)
    tx, ty, td, tz = doubleml(make_pliv_CHS2015, n_obs=N, dim_x=10, dim_z=4, return_type="array")
    assert z.shape == tz.shape
    for a, b, name in [(y, ty, "y"), (d, td, "d"), (z[:, 0], tz[:, 0], "z0"), (y * z[:, 1], ty * tz[:, 1], "yz")]:
        assert_similar(a, b, name)
    x, y, d, z = ds.make_iivm_data(n_obs=N, dim_x=5, random_state=4)
    tx, ty, td, tz = doubleml(make_iivm_data, n_obs=N, dim_x=5, return_type="array")
    for a, b, name in [(y, ty, "y"), (d, td, "d"), (z, tz, "z"), (d * z, td * tz, "dz")]:
        assert_similar(a, b, name)


def test_multiway_cluster_layout_and_distribution():
    x, y, d, cl, z = ds.make_pliv_multiway_cluster_CKMS2021(N=40, M=50, dim_X=10, random_state=5)
    tx, ty, td, tcl, tz = doubleml(make_pliv_multiway_cluster_CKMS2021, N=40, M=50, dim_X=10, return_type="array")
    np.testing.assert_array_equal(cl, tcl)
    # Means share cluster effects (only 40 + 50 independent draws): the iid standard error
    # understates their noise by about sqrt(0.25^2 * 50), so the tolerance is wider.
    for a, b, name in [(y, ty, "y"), (d, td, "d"), (z, tz, "z")]:
        assert_similar(a, b, name, k=30.0, rel_sd=0.15)
    frame = ds.make_pliv_multiway_cluster_CKMS2021(N=3, M=4, dim_X=5, return_type="DataFrame", random_state=5)
    assert list(frame.columns) == list(doubleml(make_pliv_multiway_cluster_CKMS2021, N=3, M=4, dim_X=5,
                                                return_type="DataFrame").columns)


@pytest.mark.parametrize("treatment", ["continuous", "binary"])
def test_lplr_distribution(treatment):
    x, y, d, p = ds.make_lplr_LZZ2020(n_obs=N, treatment=treatment, random_state=6)
    tx, ty, td, tp = doubleml(make_lplr_LZZ2020, n_obs=N, treatment=treatment, return_type="array")
    assert x.min() >= -2 and x.max() <= 2
    for a, b, name in [(y, ty, "y"), (d, td, "d"), (p, tp, "p"), (x[:, 0], tx[:, 0], "x0")]:
        assert_similar(a, b, name)


def test_ssm_distribution():
    for mar in (True, False):
        x, y, d, z, s = ds.make_ssm_data(n_obs=N, dim_x=10, mar=mar, random_state=7)
        tx, ty, td, tz, ts = doubleml(make_ssm_data, n_obs=N, dim_x=10, mar=mar, return_type="array")
        for a, b, name in [(y, ty, "y"), (d, td, "d"), (s, ts, "s"), (s * z, ts * tz, "sz")]:
            assert_similar(a, b, f"{name} (mar={mar})")
        assert np.all(y[s == 0] == 0)
    frame = ds.make_ssm_data(n_obs=20, dim_x=3, mar=False, return_type="DataFrame", random_state=7)
    assert list(frame.columns) == list(doubleml(make_ssm_data, n_obs=20, dim_x=3, mar=False,
                                                return_type="DataFrame").columns)


def test_confounded_designs_match_doubleml_oracles():
    ours = ds.make_confounded_plr_data(n_obs=N, random_state=8)
    theirs = doubleml(make_confounded_plr_data, n_obs=N)
    assert set(ours["oracle_values"]) == set(theirs["oracle_values"])
    for key in ("gamma_a", "beta_a"):
        assert ours["oracle_values"][key] == pytest.approx(theirs["oracle_values"][key], rel=0.1)
    assert_similar(ours["y"], theirs["y"], "y")
    assert_similar(ours["d"], theirs["d"], "d")

    ours = ds.make_confounded_irm_data(n_obs=N, theta=5.0, random_state=9)
    theirs = doubleml(make_confounded_irm_data, n_obs=N, theta=5.0)
    assert set(ours["oracle_values"]) == set(theirs["oracle_values"])
    for key in ("cf_y", "cf_d_ate", "cf_d_atte", "rho_ate", "rho_atte"):
        assert ours["oracle_values"][key] == pytest.approx(theirs["oracle_values"][key], rel=0.15, abs=0.01), key
    assert_similar(ours["y"], theirs["y"], "y")
    assert_similar(ours["d"], theirs["d"], "d")


def test_discrete_treatments_and_heterogeneous_effects():
    ours = ds.make_irm_data_discrete_treatments(n_obs=N, random_state=10)
    theirs = doubleml(make_irm_data_discrete_treatments, n_obs=N)
    np.testing.assert_allclose(np.bincount(ours["d"].astype(int)) / N,
                               np.bincount(theirs["d"].astype(int)) / N, atol=0.02)
    assert_similar(ours["y"], theirs["y"], "y")
    assert len(ours["oracle_values"]["level_bounds"]) == 4

    for n_x in (1, 2):
        ours = ds.make_heterogeneous_data(n_obs=N, n_x=n_x, binary_treatment=True, random_state=11)
        theirs = doubleml(make_heterogeneous_data, n_obs=N, n_x=n_x, binary_treatment=True)
        assert list(ours["data"].columns) == list(theirs["data"].columns)
        x = ours["data"].filter(like="X_").to_numpy()
        # The effect function is deterministic: identical to DoubleML's on the same covariates.
        np.testing.assert_allclose(ours["effects"], theirs["treatment_effect"](x), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ours["treatment_effect"](x), ours["effects"], rtol=1e-12, atol=1e-12)
        # The coefficients are drawn too, so the treatment share varies from draw to draw.
        assert abs(ours["data"]["d"].mean() - theirs["data"]["d"].mean()) < 0.15


@pytest.mark.parametrize("dgp_type", [1, 4, 5])
def test_did_sz2020_distribution(dgp_type):
    z, y, d, t = ds.make_did_SZ2020(n_obs=N, dgp_type=dgp_type, random_state=12)
    tz, ty, td, tt = doubleml(make_did_SZ2020, n_obs=N, dgp_type=dgp_type, return_type="array")
    assert t is None and tt is None
    for a, b, name in [(y, ty, "dy"), (d, td, "d"), (z[:, 0], tz[:, 0], "z0"), (y * d, ty * td, "dy*d")]:
        assert_similar(a, b, name)
    z, y, d, t = ds.make_did_SZ2020(n_obs=N, dgp_type=dgp_type, cross_sectional_data=True, random_state=12)
    tz, ty, td, tt = doubleml(make_did_SZ2020, n_obs=N, dgp_type=dgp_type, cross_sectional_data=True,
                              return_type="array")
    for a, b, name in [(y, ty, "y"), (t, tt, "t"), (y * t, ty * tt, "y*t")]:
        assert_similar(a, b, name)


def test_did_cs2021_panels_and_cross_sections():
    ours = ds.make_did_CS2021(n_obs=4000, random_state=13)
    theirs = doubleml(make_did_CS2021, n_obs=4000, time_type="float")
    assert list(ours.columns) == list(theirs.columns)
    assert len(ours) == len(theirs)
    np.testing.assert_allclose(ours["d"].value_counts(normalize=True).sort_index(),
                               theirs["d"].value_counts(normalize=True).sort_index(), atol=0.03)
    for period in range(5):
        assert_similar(ours.loc[ours.t == period, "y"], theirs.loc[theirs.t == period, "y"], f"y in {period}")
    # Anticipation shifts time and drops the latest cohort, as DoubleML.
    ours = ds.make_did_CS2021(n_obs=500, anticipation_periods=1, random_state=13)
    theirs = doubleml(make_did_CS2021, n_obs=500, anticipation_periods=1, time_type="float")
    assert sorted(ours["t"].unique()) == sorted(theirs["t"].unique())
    # DoubleML 0.11 also drops the never treated here (its first filter tests pd.isna, which is
    # false for inf; the datetime version keeps them); dml-rs keeps them.
    assert sorted(ours["d"].unique()) == sorted(theirs["d"].unique()) + [np.inf]
    dated = doubleml(make_did_CS2021, n_obs=500, anticipation_periods=1)
    assert dated["d"].isna().any()
    dated = ds.make_did_CS2021(n_obs=50, time_type="datetime", random_state=13)
    reference = doubleml(make_did_CS2021, n_obs=50)
    assert dated["t"].dtype == reference["t"].dtype
    assert sorted(dated["t"].unique()) == sorted(reference["t"].unique())
    cs = ds.make_did_cs_CS2021(n_obs=1000, random_state=14)
    reference = doubleml(make_did_cs_CS2021, n_obs=1000, time_type="float")
    assert abs(len(cs) - len(reference)) < 0.1 * len(reference)


def test_plpr_and_rdd_layout_and_distribution():
    ours = ds.make_plpr_CP2025(num_id=1000, num_t=5, dim_x=5, dgp_type="dgp2", random_state=15)
    theirs = doubleml(make_plpr_CP2025, num_id=1000, num_t=5, dim_x=5, dgp_type="dgp2")
    assert list(ours.columns) == list(theirs.columns) and ours.dtypes.equals(theirs.dtypes)
    for col in ("y", "d", "x1"):
        assert_similar(ours[col], theirs[col], col)
    ours = ds.make_simple_rdd_data(n_obs=N, fuzzy=True, random_state=16)
    theirs = doubleml(make_simple_rdd_data, n_obs=N, fuzzy=True)
    assert set(ours) == set(theirs) and set(ours["oracle_values"]) == {"Y0", "Y1"}
    for key in ("score", "Y", "D"):
        assert_similar(ours[key], theirs[key], key)
    ours = ds.make_simple_rdd_data(n_obs=N, binary_outcome=True, fuzzy=False, random_state=17)
    theirs = doubleml(make_simple_rdd_data, n_obs=N, binary_outcome=True, fuzzy=False)
    assert set(np.unique(ours["Y"])) <= {0.0, 1.0}
    assert_similar(ours["Y"], theirs["Y"], "binary Y")


def test_reproducibility_and_estimation():
    a = ds.make_irm_data(n_obs=100, random_state=3)
    b = ds.make_irm_data(n_obs=100, random_state=3)
    np.testing.assert_array_equal(a[1], b[1])
    np.random.seed(5)
    c = ds.make_irm_data(n_obs=100)
    np.random.seed(5)
    np.testing.assert_array_equal(c[1], ds.make_irm_data(n_obs=100)[1])
    with pytest.raises(ValueError, match="return_type"):
        ds.make_irm_data(return_type="DoubleMLData")  # type: ignore[arg-type]
    # The simulated effect is recovered.
    x, y, d = ds.make_plr_CCDDHNR2018(n_obs=2000, dim_x=10, alpha=0.5, random_state=4)
    fit = dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d)
    assert abs(fit.coef_[0] - 0.5) < 4 * fit.se_[0]
    x, y, d = ds.make_irm_data(n_obs=3000, dim_x=10, theta=0.5, random_state=5)
    fit = dml_rs.IRM(LinearRegression(), LogisticRegression()).fit(x, y, d)
    assert abs(fit.coef_[0] - 0.5) < 4 * fit.se_[0]

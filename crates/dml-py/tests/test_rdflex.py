import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.rdd import RDFlex as DoubleMLRDFlex
from doubleml.rdd.datasets import make_simple_rdd_data
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def rdd(seed, fuzzy, cutoff=0.0):
    np.random.seed(seed)
    data = make_simple_rdd_data(n_obs=1000, fuzzy=fuzzy, cutoff=cutoff)
    return data["X"], data["Y"], data["D"].astype(float), data["score"]


def fit_doubleml(x, y, d, score, n_iterations=2, **kwargs):
    obj = dml.DoubleMLRDDData.from_arrays(x=x, y=y, d=d, score=score)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return DoubleMLRDFlex(obj, **kwargs).fit(n_iterations=n_iterations)


def assert_same(ours, ref, tol=TOL):
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=tol)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=tol)
    # Interval bounds near zero lose relative precision to cancellation.
    np.testing.assert_allclose(ours.ci_, ref.ci, rtol=tol, atol=1e-10)
    np.testing.assert_allclose(ours.all_coef_, ref.all_coef, rtol=tol)
    np.testing.assert_allclose(ours.confint(0.9), ref.confint(level=0.9).to_numpy(), rtol=tol, atol=1e-10)
    np.testing.assert_allclose(ours.h_fs_, ref.h_fs, rtol=tol)


@pytest.mark.parametrize("spec,kernel", [("cutoff", "triangular"), ("cutoff and score", "uniform"),
                                         ("interacted cutoff and score", "epanechnikov")])
def test_rdflex_sharp_matches_doubleml(spec, kernel):
    x, y, d, score = rdd(21, fuzzy=False)
    ref = fit_doubleml(x, y, d, score, ml_g=LinearRegression(), fs_specification=spec, fs_kernel=kernel, n_rep=2)
    ours = dml_rs.RDFlex(LinearRegression(), fs_specification=spec, fs_kernel=kernel, n_rep=2).fit(
        x, y, d, score, folds=ref._smpls)
    assert_same(ours, ref)
    np.testing.assert_allclose(ours.h_, ref.h, rtol=TOL)


@pytest.mark.parametrize("n_iterations", [1, 3])
def test_rdflex_fuzzy_matches_doubleml(n_iterations):
    x, y, d, score = rdd(22, fuzzy=True, cutoff=0.3)
    ref = fit_doubleml(x, y, d, score, n_iterations=n_iterations, ml_g=LinearRegression(), ml_m=LOGIT,
                       fuzzy=True, cutoff=0.3)
    ours = dml_rs.RDFlex(LinearRegression(), LOGIT, fuzzy=True, cutoff=0.3, n_iterations=n_iterations).fit(
        x, y, d, score, folds=ref._smpls)
    assert_same(ours, ref)


def test_rdflex_defaults_and_errors():
    x, y, d, score = rdd(23, fuzzy=True)
    fit = dml_rs.RDFlex(dml_rs.Ols(), dml_rs.LogisticRegression(), fuzzy=True).fit(x, y, d, score)
    assert list(fit.summary.index) == ["Conventional", "Bias-Corrected", "Robust"]
    assert "Robust" in str(fit)
    with pytest.raises(ValueError, match="fuzzy design needs"):
        dml_rs.RDFlex(LinearRegression(), fuzzy=True).fit(x, y, d, score)
    with pytest.raises(ValueError, match="binary"):
        dml_rs.RDFlex(LOGIT).fit(x, y, d, score)
    with pytest.raises(ValueError, match="cutoff"):
        dml_rs.RDFlex(LinearRegression(), cutoff=50.0).fit(x, y, d, score)
    with pytest.raises(ValueError, match="fs_kernel"):
        dml_rs.RDFlex(LinearRegression(), fs_kernel="gaussian").fit(x, y, d, score)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"p": 2, "kernel": "epa", "bwselect": "msetwo", "vce": "hc1"},
                                    {"bwselect": "cercomb2", "vce": "hc3", "level": 90},
                                    {"rho": 1.5, "masspoints": "off", "q": 3}])
def test_rdflex_rdrobust_options_match_doubleml(kwargs):
    x, y, d, score = rdd(24, fuzzy=False)
    ref = fit_doubleml(x, y, d, score, ml_g=LinearRegression(), **kwargs)
    ours = dml_rs.RDFlex(LinearRegression(), **kwargs).fit(x, y, d, score, folds=ref._smpls)
    assert_same(ours, ref)
    np.testing.assert_allclose(ours.h_, ref.h, rtol=TOL)


def test_rdflex_covariates_clusters_and_binary_outcome_match_doubleml():
    x, y, d, score = rdd(25, fuzzy=True)
    cluster = np.random.default_rng(3).integers(0, 40, len(y))
    kwargs = {"covs": x[:, :2], "cluster": cluster}
    ref = fit_doubleml(x, y, d, score, ml_g=LinearRegression(), ml_m=LOGIT, fuzzy=True, **kwargs)
    ours = dml_rs.RDFlex(LinearRegression(), LOGIT, fuzzy=True, **kwargs).fit(x, y, d, score, folds=ref._smpls)
    assert_same(ours, ref)
    # Cluster labels of any type.
    labels = np.array([f"g{c}" for c in cluster])
    again = dml_rs.RDFlex(LinearRegression(), LOGIT, fuzzy=True, covs=x[:, :2], cluster=labels).fit(
        x, y, d, score, folds=ref._smpls)
    np.testing.assert_allclose(again.se_, ours.se_, rtol=1e-12)

    y_bin = (y > np.median(y)).astype(float)
    ref = fit_doubleml(x, y_bin, d, score, ml_g=LOGIT, ml_m=LOGIT, fuzzy=True)
    ours = dml_rs.RDFlex(LOGIT, LOGIT, fuzzy=True).fit(x, y_bin, d, score, folds=ref._smpls)
    assert_same(ours, ref)


def test_rdflex_rdrobust_options_are_parameters():
    from sklearn.base import clone
    model = dml_rs.RDFlex(LinearRegression(), p=2, kernel="uni")
    assert model.get_params()["p"] == 2
    copy = clone(model)
    assert copy.get_params()["kernel"] == "uni"
    model.set_params(p=3, n_folds=4)
    assert model.get_params()["p"] == 3 and model.n_folds == 4
    with pytest.raises(TypeError, match="unknown rdrobust options"):
        dml_rs.RDFlex(LinearRegression(), bandwith=0.3)
    with pytest.raises(ValueError, match="selects the bandwidths"):
        dml_rs.RDFlex(LinearRegression(), h=0.3)
    x, y, d, score = rdd(26, fuzzy=False)
    with pytest.raises(ValueError, match="kernel"):
        dml_rs.RDFlex(LinearRegression(), kernel="gauss").fit(x, y, d, score)


@pytest.mark.parametrize("vce", ["cr2", "cr3", "hc2"])
def test_rdflex_leverage_adjusted_cluster_variances_match_doubleml(vce):
    x, y, d, score = rdd(27, fuzzy=True)
    cluster = np.random.default_rng(4).integers(0, 30, len(y))
    kwargs = {"vce": vce, "cluster": cluster}
    ref = fit_doubleml(x, y, d, score, ml_g=LinearRegression(), ml_m=LOGIT, fuzzy=True, **kwargs)
    ours = dml_rs.RDFlex(LinearRegression(), LOGIT, fuzzy=True, **kwargs).fit(x, y, d, score, folds=ref._smpls)
    assert_same(ours, ref)

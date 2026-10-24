import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_irm_data_discrete_treatments
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-7
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
SCENARIO = dict(cf_y=0.04, cf_d=0.03, rho=0.8, level=0.95)


@pytest.fixture(scope="module")
def discrete():
    r = make_irm_data_discrete_treatments(n_obs=800, n_levels=3, linear=True, random_state=41)
    return r["x"], r["y"], r["d"]


def fit_doubleml(cls, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return cls(data, n_folds=5, **kwargs).fit()


def assert_same_sensitivity(ours_params, ref_params):
    for q in ("theta", "se", "ci"):
        for b in ("lower", "upper"):
            np.testing.assert_allclose(ours_params[q][b], ref_params[q][b], rtol=TOL)
    for key in ("rv", "rva"):
        np.testing.assert_allclose(ours_params[key], ref_params[key], atol=1e-4)


@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_apo_matches_doubleml(discrete, normalize_ipw):
    x, y, d = discrete
    ref = fit_doubleml(dml.DoubleMLAPO, dml.DoubleMLData.from_arrays(x, y, d), ml_g=LinearRegression(),
                       ml_m=LOGIT, treatment_level=2, normalize_ipw=normalize_ipw, n_rep=2)
    ours = dml_rs.APO(LinearRegression(), LOGIT, treatment_level=2, normalize_ipw=normalize_ipw).fit(
        x, y, d, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    for name, loss in ref.nuisance_loss.items():
        np.testing.assert_allclose(ours.nuisance_loss_[name], loss, rtol=TOL)
    ref.sensitivity_analysis(**SCENARIO)
    ours.sensitivity_analysis(**SCENARIO)
    assert_same_sensitivity(ours.sensitivity_params_, ref.sensitivity_params)


def test_apos_levels_and_contrasts_match_doubleml(discrete):
    x, y, d = discrete
    ref = fit_doubleml(dml.DoubleMLAPOS, dml.DoubleMLData.from_arrays(x, y, d), ml_g=LinearRegression(),
                       ml_m=LOGIT, treatment_levels=[0, 1, 2, 3], n_rep=2)
    ours = dml_rs.APOS(LinearRegression(), LOGIT, treatment_levels=[0, 1, 2, 3]).fit(x, y, d, folds=ref.smpls)
    assert list(ours.summary.index) == ["0", "1", "2", "3"]
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)

    for refs in ([0], [0, 1]):
        ref_c = ref.causal_contrast(refs)
        ours_c = ours.causal_contrast(refs)
        assert list(ours_c.summary.index) == list(ref_c.treatment_names)
        np.testing.assert_allclose(ours_c.coef_, ref_c.thetas, rtol=TOL)
        np.testing.assert_allclose(ours_c.se_, ref_c.ses, rtol=TOL)
        np.testing.assert_allclose(ours_c.confint(), ref_c.confint().to_numpy(), rtol=TOL)
        ref_c.sensitivity_analysis(**SCENARIO)
        ours_c.sensitivity_analysis(**SCENARIO)
        assert_same_sensitivity(ours_c.sensitivity_params_, ref_c.sensitivity_params)

    bm_ref = ref.sensitivity_benchmark(benchmarking_set=["X1"])
    bm = ours.sensitivity_benchmark(benchmarking_set=[0])
    np.testing.assert_allclose(bm.to_numpy(), bm_ref[["cf_y", "cf_d", "rho", "delta_theta"]].to_numpy(), rtol=1e-6)


def test_apos_joint_inference_and_errors(discrete):
    x, y, d = discrete
    model = dml_rs.APOS(dml_rs.Ols(), dml_rs.LogisticRegression(), treatment_levels=[0, 1, 2]).fit(x, y, d)
    contrast = model.causal_contrast(0)
    contrast.bootstrap(n_boot=500)
    joint, pointwise = contrast.confint(joint=True), contrast.confint()
    assert np.all(joint[:, 1] - joint[:, 0] > pointwise[:, 1] - pointwise[:, 0])
    with pytest.raises(ValueError, match="reference level"):
        model.causal_contrast(7)
    with pytest.raises(ValueError, match="does not occur"):
        dml_rs.APOS(dml_rs.Ols(), dml_rs.LogisticRegression(), treatment_levels=[0, 9]).fit(x, y, d)
    with pytest.raises(ValueError, match="only one treatment level"):
        dml_rs.APOS(dml_rs.Ols(), dml_rs.LogisticRegression(), treatment_levels=[1]).fit(x, y, d).causal_contrast(1)

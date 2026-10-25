import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_iivm_data, make_irm_data
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


@pytest.fixture(scope="module")
def irm():
    np.random.seed(3)
    return make_irm_data(n_obs=600, dim_x=5, theta=0.5, return_type="array")


@pytest.fixture(scope="module")
def iivm():
    np.random.seed(4)
    return make_iivm_data(n_obs=800, dim_x=5, theta=0.5, alpha_x=1.0, return_type="array")


def fit_doubleml(cls, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return cls(data, n_folds=5, **kwargs).fit()


def assert_same(ours, ref):
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.all_coef_, ref.all_coef, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)


# Both sides call the same scikit-learn learners on the same rows, so even the preliminary
# estimate (Brent's method on a step function, sensitive to the last bit) agrees exactly.
@pytest.mark.parametrize("treatment,quantile,normalize_ipw", [(1, 0.5, True), (0, 0.25, True), (1, 0.75, False)])
def test_pq_matches_doubleml(irm, treatment, quantile, normalize_ipw):
    x, y, d = irm
    ref = fit_doubleml(dml.DoubleMLPQ, dml.DoubleMLData.from_arrays(x, y, d), ml_g=LOGIT, ml_m=LOGIT,
                       treatment=treatment, quantile=quantile, normalize_ipw=normalize_ipw, n_rep=2)
    ours = dml_rs.PQ(LOGIT, LOGIT, treatment=treatment, quantile=quantile, normalize_ipw=normalize_ipw).fit(
        x, y, d, folds=ref.smpls)
    assert_same(ours, ref)
    for name, loss in ref.nuisance_loss.items():
        np.testing.assert_allclose(ours.nuisance_loss_[name], loss, rtol=TOL)


@pytest.mark.parametrize("treatment", [0, 1])
def test_lpq_matches_doubleml(iivm, treatment):
    x, y, d, z = iivm
    ref = fit_doubleml(dml.DoubleMLLPQ, dml.DoubleMLData.from_arrays(x, y, d, z=z), ml_g=LOGIT, ml_m=LOGIT,
                       treatment=treatment, quantile=0.5)
    ours = dml_rs.LPQ(LOGIT, LOGIT, treatment=treatment).fit(x, y, d, z, folds=ref.smpls)
    assert_same(ours, ref)


@pytest.mark.parametrize("treatment,quantile", [(1, 0.5), (0, 0.25)])
def test_cvar_matches_doubleml(irm, treatment, quantile):
    x, y, d = irm
    ref = fit_doubleml(dml.DoubleMLCVAR, dml.DoubleMLData.from_arrays(x, y, d), ml_g=LinearRegression(),
                       ml_m=LOGIT, treatment=treatment, quantile=quantile, n_rep=2)
    ours = dml_rs.CVAR(LinearRegression(), LOGIT, treatment=treatment, quantile=quantile).fit(
        x, y, d, folds=ref.smpls)
    assert_same(ours, ref)


@pytest.mark.parametrize("score", ["PQ", "CVaR"])
def test_qte_matches_doubleml(irm, score):
    x, y, d = irm
    quantiles = [0.25, 0.5, 0.75]
    ml_g = LinearRegression() if score == "CVaR" else LOGIT
    ref = fit_doubleml(dml.DoubleMLQTE, dml.DoubleMLData.from_arrays(x, y, d), ml_g=ml_g, ml_m=LOGIT,
                       quantiles=quantiles, score=score, n_rep=2)
    ours = dml_rs.QTE(ml_g, LOGIT, quantiles=quantiles, score=score).fit(x, y, d, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    potential = ours.potential
    np.testing.assert_allclose(potential.coef_[0::2], [m.coef[0] for m in ref.modellist_0], rtol=TOL)
    np.testing.assert_allclose(potential.coef_[1::2], [m.coef[0] for m in ref.modellist_1], rtol=TOL)
    np.testing.assert_allclose(potential.se_[1::2], [m.se[0] for m in ref.modellist_1], rtol=TOL)
    # Joint inference over the quantiles.
    ref.bootstrap(n_rep_boot=2000)
    ours.bootstrap(n_boot=2000, random_state=3)
    np.testing.assert_allclose(ours.confint(joint=True), ref.confint(joint=True).to_numpy(), atol=0.05)


def test_qte_defaults_and_errors(irm, iivm):
    x, y, d = irm
    fit = dml_rs.QTE(LOGIT, quantiles=[0.3, 0.7]).fit(x, y, d)  # ml_m defaults to ml_g, splits stratified
    assert list(fit.summary.index) == ["0.3", "0.7"]
    assert fit.potential.treatment_names_ == ["0.3 (d=0)", "0.3 (d=1)", "0.7 (d=0)", "0.7 (d=1)"]
    assert fit.quantiles_.tolist() == [0.3, 0.7]
    with pytest.raises(ValueError, match="ml_m"):
        dml_rs.QTE(LinearRegression(), score="CVaR").fit(x, y, d)
    with pytest.raises(ValueError, match="quantile"):
        dml_rs.PQ(LOGIT, LOGIT, quantile=1.5).fit(x, y, d)
    with pytest.raises(ValueError):
        dml_rs.PQ(LOGIT, LOGIT, treatment=2).fit(x, y, d)
    with pytest.raises(ValueError):
        dml_rs.QTE(LOGIT, score="LPQ").fit(x, y, d)  # needs an instrument
    xi, yi, di, zi = iivm
    lpq = dml_rs.QTE(LOGIT, quantiles=0.5, score="LPQ").fit(xi, yi, di, zi)
    assert np.isfinite(lpq.coef_).all()

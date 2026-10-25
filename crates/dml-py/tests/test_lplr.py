import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.plm.datasets import make_lplr_LZZ2020
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def lplr_arrays(treatment, seed):
    np.random.seed(seed)
    x, y, d, _ = make_lplr_LZZ2020(n_obs=800, dim_x=13, alpha=0.5, return_type="array", treatment=treatment)
    return x, y, d.astype(float)


def fit_doubleml(x, y, d, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return dml.DoubleMLLPLR(dml.DoubleMLData.from_arrays(x, y, d), **kwargs).fit()


@pytest.mark.parametrize("treatment", ["continuous", "binary"])
@pytest.mark.parametrize("score", ["nuisance_space", "instrument"])
def test_lplr_matches_doubleml(treatment, score):
    x, y, d = lplr_arrays(treatment, 8)
    ml_m = LinearRegression() if treatment == "continuous" else LOGIT
    ref = fit_doubleml(x, y, d, ml_M=LOGIT, ml_t=LinearRegression(), ml_m=ml_m, score=score, n_rep=2)
    ours = dml_rs.LPLR(LOGIT, LinearRegression(), ml_m, score=score).fit(
        x, y, d, folds=ref.smpls, inner_folds=ref.smpls_inner)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.all_coef_, ref.all_coef, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    for name in ("ml_m", "ml_a", "ml_t", "ml_M"):
        np.testing.assert_allclose(ours.predictions_[name], ref.predictions[name], rtol=TOL, atol=1e-12)
    for name in ("ml_m", "ml_a", "ml_M"):
        np.testing.assert_allclose(ours.nuisance_loss_[name], ref.nuisance_loss[name], rtol=TOL)


def test_lplr_separate_ml_a():
    x, y, d = lplr_arrays("continuous", 9)
    ref = fit_doubleml(x, y, d, ml_M=LOGIT, ml_t=LinearRegression(), ml_m=LinearRegression(), ml_a=Ridge(alpha=5.0))
    ours = dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression(), ml_a=Ridge(alpha=5.0)).fit(
        x, y, d, folds=ref.smpls, inner_folds=ref.smpls_inner)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


def test_lplr_defaults_and_errors():
    x, y, d = lplr_arrays("binary", 10)
    fit = dml_rs.LPLR(dml_rs.LogisticRegression(), dml_rs.Ols(), dml_rs.LogisticRegression(), n_rep=2).fit(x, y, d)
    assert fit.all_coef_.shape == (1, 2)
    assert abs(fit.coef_[0] - 0.5) < 4 * fit.se_[0]
    with pytest.raises(ValueError, match="binary"):
        dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression()).fit(x, y + 0.5, d)
    with pytest.raises(ValueError, match="inner_folds requires folds"):
        dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression()).fit(x, y, d, inner_folds=[[]])
    with pytest.raises(ValueError, match="score"):
        dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression(), score="iv").fit(x, y, d)  # type: ignore[arg-type]

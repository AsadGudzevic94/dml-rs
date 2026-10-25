import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_ssm_data
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def ssm_arrays(mar, seed):
    np.random.seed(seed)
    return make_ssm_data(n_obs=1000, dim_x=5, theta=1.0, mar=mar, return_type="array")


@pytest.mark.parametrize("score", ["missing-at-random", "nonignorable"])
@pytest.mark.parametrize("normalize_ipw", [False, True])
def test_ssm_matches_doubleml(score, normalize_ipw):
    mar = score == "missing-at-random"
    x, y, d, z, s = ssm_arrays(mar, 5)
    data = dml.DoubleMLSSMData.from_arrays(x, y, d, z=None if mar else z, s=s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        ref = dml.DoubleMLSSM(data, LinearRegression(), LOGIT, LOGIT, score=score, normalize_ipw=normalize_ipw,
                              n_folds=5, n_rep=2).fit()
    ours = dml_rs.SSM(LinearRegression(), LOGIT, LOGIT, score=score, normalize_ipw=normalize_ipw).fit(
        x, y, d, s, z=None if mar else z, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.all_coef_, ref.all_coef, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    for name, loss in ref.nuisance_loss.items():
        np.testing.assert_allclose(ours.nuisance_loss_[name], loss, rtol=TOL)
    for name in ("ml_g_d0", "ml_g_d1", "ml_pi", "ml_m"):
        np.testing.assert_allclose(ours.predictions_[name][:, :, 0], ref.predictions[name][:, :, 0], rtol=TOL)


def test_ssm_defaults_and_errors():
    x, y, d, z, s = ssm_arrays(False, 6)
    fit = dml_rs.SSM(LinearRegression(), LOGIT, LOGIT, score="nonignorable").fit(x, y, d, s, z=z)
    assert abs(fit.coef_[0] - 1.0) < 4 * fit.se_[0]
    assert fit.summary["coef"].shape == (1,)
    with pytest.raises(ValueError, match="instrument"):
        dml_rs.SSM(LinearRegression(), LOGIT, LOGIT, score="nonignorable").fit(x, y, d, s)
    with pytest.raises(ValueError, match="selection indicator"):
        dml_rs.SSM(LinearRegression(), LOGIT, LOGIT).fit(x, y, d, s + 1)
    with pytest.raises(ValueError, match="score"):
        dml_rs.SSM(LinearRegression(), LOGIT, LOGIT, score="mnar").fit(x, y, d, s)  # type: ignore[arg-type]

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


def normalized_estimates(ref, y, d, s):
    estimates = []
    for rep in range(ref.n_rep):
        g0, g1, pi, m = (ref.predictions[name][:, rep, 0] for name in ("ml_g_d0", "ml_g_d1", "ml_pi", "ml_m"))
        w1, w0 = d * s / (m * pi), (1 - d) * s / ((1 - m) * pi)
        w1, w0 = w1 / w1.mean(), w0 / w0.mean()
        estimates.append(np.mean(w1 * (y - g1) + g1 - w0 * (y - g0) - g0))
    return estimates


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
    if normalize_ipw:
        # The pinned DoubleML 0.11.4 scales the weights wrongly (fixed on its main in #427), so
        # compare with the fixed score on DoubleML's predictions. The fixtures come from main.
        np.testing.assert_allclose(ours.all_coef_[0], normalized_estimates(ref, y, d, s), rtol=TOL)
    else:
        np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
        np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
        np.testing.assert_allclose(ours.all_coef_, ref.all_coef, rtol=TOL)
        np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    for name, loss in ref.nuisance_loss.items():
        np.testing.assert_allclose(ours.nuisance_loss_[name], loss, rtol=TOL)
    for name in ("ml_g_d0", "ml_g_d1", "ml_pi", "ml_m"):
        np.testing.assert_allclose(ours.predictions_[name][:, :, 0], ref.predictions[name][:, :, 0], rtol=TOL)


def test_ssm_normalized_weights_are_unbiased():
    # The outcome model is misspecified, so the weights have to remove the selection bias.
    estimates = {False: [], True: []}
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(4000, 3))
        d = rng.binomial(1, 1 / (1 + np.exp(0.8 - 0.5 * x[:, 0])))
        s = rng.binomial(1, 1 / (1 + np.exp(-1 - 0.5 * x[:, 1] - 0.5 * d)))
        y = d + 3 * x[:, 0] ** 2 + x[:, 1] + rng.normal(size=4000)
        for normalize_ipw in estimates:
            fit = dml_rs.SSM(LinearRegression(), LOGIT, LOGIT, normalize_ipw=normalize_ipw, random_state=seed)
            estimates[normalize_ipw].append(fit.fit(x, y, d, s).coef_[0])
    bias = np.mean(estimates[True]) - 1.0
    assert abs(bias) < 4 * np.std(estimates[True], ddof=1) / np.sqrt(20)
    assert abs(np.mean(estimates[True]) - np.mean(estimates[False])) < 0.05


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

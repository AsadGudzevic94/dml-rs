import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_irm_data
from doubleml.utils.propensity_score_processing import PSProcessorConfig
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def config(calibration):
    return PSProcessorConfig(calibration_method="isotonic", cv_calibration=calibration == "isotonic_cv")


@pytest.fixture(scope="module")
def irm():
    np.random.seed(7)
    return make_irm_data(n_obs=600, dim_x=5, theta=0.5, return_type="array")


def fit_doubleml(cls, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return cls(data, n_folds=5, **kwargs).fit()


@pytest.mark.parametrize("calibration", ["isotonic", "isotonic_cv"])
@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_irm_calibration_matches_doubleml(irm, calibration, score):
    x, y, d = irm
    ref = fit_doubleml(dml.DoubleMLIRM, dml.DoubleMLData.from_arrays(x, y, d), ml_g=LinearRegression(), ml_m=LOGIT,
                       score=score, ps_processor_config=config(calibration), n_rep=2)
    ours = dml_rs.IRM(LinearRegression(), LOGIT, score=score, calibration=calibration).fit(x, y, d, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.predictions_["ml_m"], ref.predictions["ml_m"], rtol=TOL)
    ref.sensitivity_analysis()
    ours.sensitivity_analysis()
    np.testing.assert_allclose(ours.sensitivity_params_["theta"]["lower"], ref.sensitivity_params["theta"]["lower"],
                               rtol=1e-7)


@pytest.mark.parametrize("cls,kind", [(dml.DoubleMLCVAR, "cvar"), (dml.DoubleMLPQ, "pq")])
def test_quantile_calibration_matches_doubleml(irm, cls, kind):
    x, y, d = irm
    ml_g = LinearRegression() if kind == "cvar" else LOGIT
    ref = fit_doubleml(cls, dml.DoubleMLData.from_arrays(x, y, d), ml_g=ml_g, ml_m=LOGIT, quantile=0.5,
                       ps_processor_config=config("isotonic_cv"))
    model = dml_rs.CVAR if kind == "cvar" else dml_rs.PQ
    ours = model(ml_g, LOGIT, calibration="isotonic_cv").fit(x, y, d, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


def test_unknown_calibration_is_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="calibration"):
        dml_rs.IRM(LinearRegression(), LOGIT, calibration="platt").fit(  # type: ignore[arg-type]
            rng.normal(size=(50, 2)), rng.normal(size=50), (rng.random(50) < 0.5).astype(float))

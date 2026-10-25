import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plpr_CP2025, make_plr_CCDDHNR2018
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def fitted(model, external):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(external_predictions=external)


def test_plr_partial_external_matches_doubleml():
    np.random.seed(3)
    x, y, d = make_plr_CCDDHNR2018(n_obs=400, dim_x=5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(1)
    first = dml.DoubleMLPLR(data, Lasso(alpha=0.05), Lasso(alpha=0.05), n_rep=2).fit()
    ext = {"d": {"ml_m": first.predictions["ml_m"][:, :, 0]}}
    np.random.seed(2)
    ref = dml.DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_rep=2)
    fitted(ref, ext)
    ours = dml_rs.PLR(LinearRegression(), LinearRegression(), n_rep=2).fit(
        x, y, d, folds=ref.smpls, external_predictions=ext)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    assert "ml_m" not in ours.nuisance_loss_ and "ml_l" in ours.nuisance_loss_
    np.testing.assert_allclose(ours.predictions_["ml_m"][:, :, 0], ext["d"]["ml_m"])


def test_irm_external_matches_doubleml():
    np.random.seed(4)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    rng = np.random.default_rng(0)
    ext = {"d": {"ml_m": rng.uniform(0.2, 0.8, size=(500, 1)), "ml_g0": rng.normal(size=(500, 1))}}
    np.random.seed(5)
    ref = dml.DoubleMLIRM(data, LinearRegression(), LOGIT)
    fitted(ref, ext)
    ours = dml_rs.IRM(LinearRegression(), LOGIT).fit(x, y, d, folds=ref.smpls, external_predictions=ext["d"])
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


def test_plpr_external_m_skips_cre_adjustment():
    np.random.seed(6)
    df = make_plpr_CP2025(num_id=50, num_t=5, dim_x=3, theta=0.5)
    x_cols = [c for c in df.columns if c.startswith("x")]
    data = dml.DoubleMLPanelData(df, "y", "d", "time", "id", x_cols=x_cols, static_panel=True)
    np.random.seed(7)
    ref = dml.DoubleMLPLPR(data, LinearRegression(), LinearRegression(), approach="cre_general")
    ext = {"d": {"ml_m": np.random.default_rng(1).normal(size=(len(df), 1))}}
    fitted(ref, ext)
    ours = dml_rs.PLPR(LinearRegression(), LinearRegression(), approach="cre_general").fit(
        df[x_cols], df["y"], df["d"], df["id"], df["time"], folds=ref.smpls, external_predictions=ext)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


def test_external_predictions_errors():
    np.random.seed(8)
    x, y, d = make_plr_CCDDHNR2018(n_obs=100, dim_x=3, return_type="array")
    with pytest.raises(ValueError, match="external predictions"):
        dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d, external_predictions={"ml_l": np.zeros(99)})
    with pytest.raises(ValueError):
        dml_rs.PLR(LinearRegression(), LinearRegression()).fit(
            x, y, d, external_predictions={"other": {"ml_l": np.zeros(100)}})

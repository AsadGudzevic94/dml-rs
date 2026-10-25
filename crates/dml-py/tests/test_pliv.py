import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.plm.datasets import make_pliv_CHS2015
from sklearn.linear_model import Lasso, LinearRegression

import dml_rs

TOL = 1e-9


def pliv_data(seed, dim_z):
    np.random.seed(seed)
    x, y, d, z = make_pliv_CHS2015(n_obs=500, alpha=1.0, dim_x=20, dim_z=dim_z, return_type="array")
    return x, y, d, np.asarray(z).reshape(len(y), -1)


def fit_doubleml(factory, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return factory(data, n_folds=5, **kwargs).fit()


def assert_same(ours, ref):
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    for name, loss in ref.nuisance_loss.items():
        if not np.all(np.isnan(loss)):
            np.testing.assert_allclose(ours.nuisance_loss_[name], loss, rtol=TOL)


def test_pliv_iv_type_matches_doubleml():
    x, y, d, z = pliv_data(41, 1)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z.ravel())
    ref = fit_doubleml(dml.DoubleMLPLIV, data, ml_l=LinearRegression(), ml_m=LinearRegression(),
                       ml_r=LinearRegression(), ml_g=Lasso(alpha=0.05), score="IV-type", n_rep=2)
    ours = dml_rs.PLIV(LinearRegression(), LinearRegression(), LinearRegression(), ml_g=Lasso(alpha=0.05),
                       score="IV-type").fit(x, y, d, z, folds=ref.smpls)
    assert_same(ours, ref)


@pytest.mark.parametrize("dim_z", [2, 4])
def test_pliv_several_instruments_matches_doubleml(dim_z):
    x, y, d, z = pliv_data(42, dim_z)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    ref = fit_doubleml(dml.DoubleMLPLIV, data, ml_l=LinearRegression(), ml_m=LinearRegression(),
                       ml_r=LinearRegression(), n_rep=2)
    ours = dml_rs.PLIV(LinearRegression(), LinearRegression(), LinearRegression()).fit(x, y, d, z, folds=ref.smpls)
    assert_same(ours, ref)
    assert sorted(k for k in ours.predictions_ if k.startswith("ml_m_")) == [f"ml_m_z{i + 1}" for i in range(dim_z)]


def test_pliv_partial_z_and_xz_match_doubleml():
    x, y, d, z = pliv_data(43, 3)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    ref = fit_doubleml(dml.DoubleMLPLIV._partialZ, data, ml_r=LinearRegression())
    ours = dml_rs.PLIV(None, None, LinearRegression(), partial="Z").fit(x, y, d, z, folds=ref.smpls)
    assert_same(ours, ref)
    ref = fit_doubleml(dml.DoubleMLPLIV._partialXZ, data, ml_l=LinearRegression(), ml_m=LinearRegression(),
                       ml_r=LinearRegression())
    ours = dml_rs.PLIV(LinearRegression(), LinearRegression(), LinearRegression(), partial="XZ").fit(
        x, y, d, z, folds=ref.smpls)
    assert_same(ours, ref)


def test_pliv_errors():
    x, y, d, z = pliv_data(44, 2)
    with pytest.raises(ValueError, match="IV-type"):
        dml_rs.PLIV(LinearRegression(), LinearRegression(), LinearRegression(), score="IV-type").fit(x, y, d, z)
    with pytest.raises(ValueError, match="needs ml_l"):
        dml_rs.PLIV(None, LinearRegression(), LinearRegression()).fit(x, y, d, z)
    with pytest.raises(ValueError, match="1-D|exactly one instrument"):
        dml_rs.IIVM(LinearRegression(), dml_rs.LogisticRegression(), dml_rs.LogisticRegression()).fit(
            x, y, (d > 0).astype(float), z)

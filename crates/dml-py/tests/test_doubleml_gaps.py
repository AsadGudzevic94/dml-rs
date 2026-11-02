import warnings

import doubleml as dml
import matplotlib
import numpy as np
import pytest
from doubleml.irm.datasets import make_irm_data
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

import dml_rs

matplotlib.use("Agg")
TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def two_treatments(seed=4, n=500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    d = np.column_stack((x[:, 0] + rng.normal(size=n), x[:, 1] + rng.normal(size=n)))
    y = d[:, 0] + 0.5 * d[:, 1] + x[:, 0] + rng.normal(size=n)
    return x, y, d


@pytest.mark.parametrize("use_others", [True, False])
def test_plr_other_treatments_as_covariates(use_others):
    x, y, d = two_treatments()
    data = dml.DoubleMLData.from_arrays(x, y, d, use_other_treat_as_covariate=use_others)
    np.random.seed(1)
    ref = quiet(dml.DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=4).fit)
    ours = dml_rs.PLR(LinearRegression(), LinearRegression(), n_folds=4, use_other_treat_as_covariate=use_others)
    ours.fit(x, y, d, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


@pytest.mark.parametrize("use_others", [True, False])
def test_pliv_other_treatments_as_covariates(use_others):
    x, y, d = two_treatments()
    z = (d + np.random.default_rng(9).normal(size=d.shape))[:, :1]
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z, use_other_treat_as_covariate=use_others)
    np.random.seed(1)
    ref = quiet(dml.DoubleMLPLIV(data, LinearRegression(), LinearRegression(), LinearRegression(), n_folds=4).fit)
    ours = dml_rs.PLIV(LinearRegression(), LinearRegression(), LinearRegression(), n_folds=4,
                       use_other_treat_as_covariate=use_others)
    ours.fit(x, y, d, z, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)


def test_tuning_follows_the_covariate_choice():
    x, y, d = two_treatments()
    grid = {"ml_l": {"alpha": [0.1, 10.0, 1000.0]}, "ml_m": {"alpha": [0.1, 10.0, 1000.0]}}
    data = dml.DoubleMLData.from_arrays(x, y, d, use_other_treat_as_covariate=False)
    ref = dml.DoubleMLPLR(data, Ridge(), Ridge(), n_folds=4)
    np.random.seed(3)
    quiet(ref.tune, grid)
    np.random.seed(3)
    ours = dml_rs.PLR(Ridge(), Ridge(), n_folds=4, use_other_treat_as_covariate=False)
    ours.tune(x, y, d, param_grids=grid)
    for name in grid:
        assert [ours.params[name][t] for t in ours.params[name]] == [ref.params[name][t] for t in ref.params[name]]


def irm_arrays():
    np.random.seed(5)
    return make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")


def test_raw_propensity_is_the_cross_fitted_prediction():
    x, y, d = irm_arrays()
    fit = dml_rs.IRM(LinearRegression(), LOGIT, clipping_threshold=0.2, n_rep=2).fit(x, y, d)
    raw, clipped = fit.raw_propensity_, fit.predictions_["ml_m"]
    assert raw.shape == (500, 2, 1)
    np.testing.assert_array_equal(np.clip(raw, 0.2, 0.8), clipped)
    assert raw.min() < 0.2 or raw.max() > 0.8


def test_overlap_plot():
    x, y, d = irm_arrays()
    fit = dml_rs.IRM(LinearRegression(), LOGIT).fit(x, y, d)
    fig, axes = fit.plot_overlap_common_support(bins=8)
    counts = [sum(p.get_height() for p in axes[0, k].patches) for k in (0, 1)]
    assert counts == [d.sum(), (1 - d).sum()]
    assert axes.shape == (2, 2)
    fit.plot_overlap_common_support(density=True, bins=[0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="i_rep"):
        fit.plot_overlap_common_support(i_rep=1)
    with pytest.raises(ValueError, match="bins"):
        fit.plot_overlap_common_support(bins=1)

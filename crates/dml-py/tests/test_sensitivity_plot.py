import warnings

import doubleml as dml
import matplotlib
import numpy as np
import pytest
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.linear_model import LinearRegression

import dml_rs

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def fits():
    np.random.seed(1)
    x, y, d = make_plr_CCDDHNR2018(n_obs=500, dim_x=5, return_type="array")
    np.random.seed(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = dml.DoubleMLPLR(dml.DoubleMLData.from_arrays(x, y, d), LinearRegression(), LinearRegression()).fit()
    ours = dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d, folds=ref.smpls)
    return ref, ours


@pytest.mark.parametrize("value,null", [("theta", 0.0), ("ci", 0.0), ("theta", 2.0)])
def test_contours_match_doubleml(fits, value, null):
    ref, ours = fits
    ref.sensitivity_analysis(cf_y=0.04, cf_d=0.03, null_hypothesis=null)
    ours.sensitivity_analysis(cf_y=0.04, cf_d=0.03, null_hypothesis=null)
    fig = ref.sensitivity_plot(value=value, grid_size=15, grid_bounds=(0.1, 0.2))
    data = ours.sensitivity_contours(value=value, grid_size=15, grid_bounds=(0.1, 0.2))
    # DoubleML's z holds its grid with rows indexed by cf_d (see docs/validation.md).
    np.testing.assert_allclose(data["values"], np.asarray(fig.data[0].z), rtol=1e-9)
    np.testing.assert_allclose(data["cf_d"], fig.data[0].x)
    assert data["bound"] == ("upper" if null > ours.coef_[0] else "lower")


def test_plot_draws_with_scenario_and_benchmarks(fits):
    _, ours = fits
    ours.sensitivity_analysis(cf_y=0.04, cf_d=0.03)
    ax = ours.sensitivity_plot(benchmarks={"cf_y": [0.02], "cf_d": [0.05], "name": ["X1"]}, grid_size=12)
    assert ax.get_xlabel() == "cf_d" and ax.get_ylabel() == "cf_y"
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("scenario") for t in texts) and any(t.startswith("X1") for t in texts)
    ours.sensitivity_plot(value="ci", fill=False, grid_size=10)


def test_contours_need_a_sensitivity_analysis():
    np.random.seed(3)
    x, y, d = make_plr_CCDDHNR2018(n_obs=200, dim_x=3, return_type="array")
    fit = dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d)
    with pytest.raises(ValueError, match="sensitivity_analysis"):
        fit.sensitivity_contours()
    fit.sensitivity_analysis()
    with pytest.raises(ValueError, match="value"):
        fit.sensitivity_contours(value="bias")  # type: ignore[arg-type]

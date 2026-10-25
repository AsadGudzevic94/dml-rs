import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.did.datasets import make_did_CS2021
from doubleml.irm.datasets import make_iivm_data, make_irm_data
from doubleml.plm.datasets import make_lplr_LZZ2020
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


@pytest.fixture(scope="module")
def fits():
    """PLR and IRM on the same data and sample splits, in dml-rs and DoubleML."""
    np.random.seed(7)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(1)
    ref_plr = quiet(lambda: dml.DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_rep=2).fit())
    ref_irm = dml.DoubleMLIRM(data, LinearRegression(), LOGIT, n_rep=2, draw_sample_splitting=False)
    ref_irm.set_sample_splitting(ref_plr.smpls)
    quiet(ref_irm.fit)
    plr = dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d, folds=ref_plr.smpls)
    irm = dml_rs.IRM(LinearRegression(), LOGIT).fit(x, y, d, folds=ref_plr.smpls)
    return plr.framework, irm.framework, ref_plr.framework, ref_irm.framework


def assert_framework(ours, ref, se_sign=1.0):
    np.testing.assert_allclose(ours.coef_, ref.thetas, rtol=TOL)
    np.testing.assert_allclose(ours.all_coef_, ref.all_thetas, rtol=TOL)
    np.testing.assert_allclose(ours.se_, se_sign * ref.ses, rtol=TOL)
    np.testing.assert_allclose(ours.all_se_, se_sign * ref.all_ses, rtol=TOL)
    if se_sign > 0:
        np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)


def assert_sensitivity(ours, ref):
    ours.sensitivity_analysis(cf_y=0.04, cf_d=0.03, rho=0.8)
    ref.sensitivity_analysis(cf_y=0.04, cf_d=0.03, rho=0.8)
    for key in ("theta", "ci"):
        for bound in ("lower", "upper"):
            np.testing.assert_allclose(ours.sensitivity_params_[key][bound], ref.sensitivity_params[key][bound],
                                       rtol=TOL)
    np.testing.assert_allclose(ours.sensitivity_params_["rv"], ref.sensitivity_params["rv"], atol=1e-4)


def test_arithmetic_matches_doubleml(fits):
    plr, irm, ref_plr, ref_irm = fits
    for ours, ref in [(plr + irm, ref_plr + ref_irm), (plr - irm, ref_plr - ref_irm), (2.5 * plr, 2.5 * ref_plr),
                      (irm * 3, ref_irm * 3.0)]:
        assert isinstance(ours, dml_rs.Framework)
        assert_framework(ours, ref)
        assert_sensitivity(ours, ref)
    # DoubleML 0.11 also multiplies the standard error by a negative scalar (negative se); dml-rs
    # reports |c| * se. The sensitivity bounds come from the scores and agree.
    neg = -0.5 * irm
    assert_framework(neg, -0.5 * ref_irm, se_sign=-1.0)
    assert_sensitivity(neg, -0.5 * ref_irm)
    np.testing.assert_allclose((-irm).coef_, -irm.coef_, rtol=0)


def test_concat_and_linear_combination(fits):
    plr, irm, ref_plr, ref_irm = fits
    both = dml_rs.concat([plr, irm])
    ref = dml.double_ml_framework.concat([ref_plr, ref_irm])
    assert_framework(both, ref)
    assert_sensitivity(both, ref)
    diff = both.linear_combination([1.0, -1.0], names=["PLR - IRM"])
    assert_framework(diff, ref_plr - ref_irm)
    assert list(diff.summary.index) == ["PLR - IRM"]
    # Joint inference over the stacked estimates.
    both.bootstrap(n_boot=200)
    assert both.confint(joint=True).shape == (2, 2)
    # Estimators can be stacked directly, and invalid operands are rejected.
    assert dml_rs.concat([dml_rs.PLR(LinearRegression(), LinearRegression(), n_rep=2).fit(*_irm_arrays()),
                          plr]).coef_.shape == (2,)
    with pytest.raises(TypeError):
        plr + 1.0
    with pytest.raises(TypeError):
        dml_rs.concat([1.0])
    with pytest.raises(ValueError):
        both + plr  # two estimates and one


def _irm_arrays():
    np.random.seed(7)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    return x, y, d


def test_staggered_sensitivity_and_aggregations():
    np.random.seed(5)
    df = make_did_CS2021(n_obs=500, dgp_type=1, include_never_treated=True, time_type="float")
    x_cols = [c for c in df.columns if c.startswith("Z")]
    model = dml_rs.StaggeredDID(LinearRegression(), LOGIT).fit_frame(df, "y", "id", "t", "d", x_cols)
    model.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
    k = len(model.coef_)
    lower = model.sensitivity_params_["theta"]["lower"]
    assert lower.shape == (k,) and np.all(lower < model.coef_)
    assert "Robustness" in model.sensitivity_summary
    agg = model.aggregate("eventstudy")
    fw = agg.aggregated_frameworks.sensitivity_analysis()
    assert fw.sensitivity_params_["rv"].shape == (len(agg.names),)
    overall = agg.overall_aggregated_framework.sensitivity_analysis()
    assert overall.coef_[0] == pytest.approx(agg.overall_coef)
    # Benchmarks refit without the covariate on the same seeded splits.
    bm = model.sensitivity_benchmark(benchmarking_set=[0])
    assert bm.shape == (k, 4)
    assert np.all((bm["cf_d"] >= 0) & (bm["cf_d"] <= 1))


def test_pq_external_predictions_match_doubleml():
    np.random.seed(3)
    x, y, d = make_irm_data(n_obs=600, dim_x=5, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(1)
    first = quiet(lambda: dml.DoubleMLPQ(data, ml_g=LOGIT, ml_m=LOGIT, n_rep=2).fit())
    preds = first.predictions
    for names in (["ml_g"], ["ml_g", "ml_m"]):
        ext = {"d": {name: preds[name][:, :, 0] for name in names}}
        ref = dml.DoubleMLPQ(data, ml_g=LOGIT, ml_m=LOGIT, n_rep=2, draw_sample_splitting=False)
        ref.set_sample_splitting(first.smpls)
        quiet(lambda: ref.fit(external_predictions=ext))
        ours = dml_rs.PQ(LOGIT, LOGIT, n_rep=2).fit(x, y, d, folds=first.smpls, external_predictions=ext)
        np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
        np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
        assert set(ours.nuisance_loss_) == {"ml_g", "ml_m"} - set(names)


def test_lpq_external_predictions_match_doubleml():
    np.random.seed(4)
    x, y, d, z = make_iivm_data(n_obs=800, dim_x=5, theta=0.5, alpha_x=1.0, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    np.random.seed(1)
    first = quiet(lambda: dml.DoubleMLLPQ(data, ml_g=LOGIT, ml_m=LOGIT).fit())
    ext = {"d": {name: p[:, :, 0] for name, p in first.predictions.items()}}
    ref = dml.DoubleMLLPQ(data, ml_g=LOGIT, ml_m=LOGIT, draw_sample_splitting=False)
    ref.set_sample_splitting(first.smpls)
    quiet(lambda: ref.fit(external_predictions=ext))
    ours = dml_rs.LPQ(LOGIT, LOGIT).fit(x, y, d, z, folds=first.smpls, external_predictions=ext)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    with pytest.raises(ValueError, match="all of"):
        dml_rs.LPQ(LOGIT, LOGIT).fit(x, y, d, z, folds=first.smpls,
                                     external_predictions={"d": {"ml_m_z": ext["d"]["ml_m_z"]}})


def test_lplr_external_predictions_match_doubleml():
    np.random.seed(8)
    x, y, d, _ = make_lplr_LZZ2020(n_obs=800, dim_x=13, alpha=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(1)
    first = quiet(lambda: dml.DoubleMLLPLR(data, ml_M=LOGIT, ml_t=LinearRegression(), ml_m=LinearRegression()).fit())
    preds = first.predictions
    names = ["ml_M", "ml_m", "ml_a", "ml_t"] + [f"ml_{a}_inner_{k}" for a in ("M", "a") for k in range(5)]
    ext = {"d": {name: preds[name][:, :, 0] for name in names}}
    # DoubleMLLPLR has no set_sample_splitting: the same seed draws the same outer and inner splits.
    np.random.seed(1)
    ref = quiet(lambda: dml.DoubleMLLPLR(data, ml_M=LOGIT, ml_t=LinearRegression(), ml_m=LinearRegression()))
    quiet(lambda: ref.fit(external_predictions=ext))
    ours = dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression()).fit(
        x, y, d, folds=first.smpls, inner_folds=first.smpls_inner, external_predictions=ext)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    # Our own predictions round-trip, including the inner ones.
    again = dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression()).fit(
        x, y, d, folds=first.smpls, inner_folds=first.smpls_inner,
        external_predictions={"d": {name: ours.predictions_[name][:, :, 0] for name in names}})
    np.testing.assert_allclose(again.coef_, ours.coef_, rtol=1e-12)
    with pytest.raises(ValueError, match="ml_M_inner"):
        dml_rs.LPLR(LOGIT, LinearRegression(), LinearRegression()).fit(
            x, y, d, external_predictions={"d": {"ml_M": preds["ml_M"][:, :, 0]}})

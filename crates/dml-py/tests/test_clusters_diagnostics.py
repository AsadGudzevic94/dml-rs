import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_pliv_multiway_cluster_CKMS2021, make_plr_CCDDHNR2018
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error

import dml_rs

TOL = 1e-7
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
SCENARIO = dict(cf_y=0.05, cf_d=0.04, rho=0.7, level=0.9)


def fit_doubleml(cls, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(data, n_folds=3, **kwargs).fit()


def assert_same_inference(ours, ref):
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.all_coef_, ref.all_coef, rtol=TOL)
    np.testing.assert_allclose(ours.all_se_, ref.all_se, rtol=TOL)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)


def test_plr_one_way_clusters_match_doubleml():
    np.random.seed(11)
    x, y, d = make_plr_CCDDHNR2018(n_obs=400, dim_x=5, alpha=0.5, return_type="array")
    cluster = np.random.choice(40, size=400)
    np.random.seed(5)
    ref = fit_doubleml(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d, cluster_vars=cluster),
                       ml_l=LinearRegression(), ml_m=LinearRegression(), n_rep=2)
    labels = np.array([f"school-{c}" for c in cluster])  # any label type works
    ours = dml_rs.PLR(LinearRegression(), LinearRegression(), n_folds=3, n_rep=2).fit(
        x, y, d, folds=ref.smpls, cluster=labels)
    assert ours.n_clusters_ == 40
    assert_same_inference(ours, ref)
    ref.sensitivity_analysis(**SCENARIO)
    ours.sensitivity_analysis(**SCENARIO)
    for q in ("theta", "se", "ci"):
        for b in ("lower", "upper"):
            np.testing.assert_allclose(ours.sensitivity_params_[q][b], ref.sensitivity_params[q][b], rtol=TOL)
    assert "cluster" in str(ours).lower()


def test_irm_one_way_clusters_match_doubleml():
    np.random.seed(12)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    cluster = np.repeat(np.arange(100), 5)
    np.random.seed(6)
    ref = fit_doubleml(dml.DoubleMLIRM, dml.DoubleMLData.from_arrays(x, y, d, cluster_vars=cluster),
                       ml_g=LinearRegression(), ml_m=LOGIT)
    ours = dml_rs.IRM(LinearRegression(), LOGIT, n_folds=3).fit(x, y, d, folds=ref.smpls, cluster=cluster)
    assert_same_inference(ours, ref)


def test_pliv_two_way_clusters_match_doubleml():
    np.random.seed(13)
    df = make_pliv_multiway_cluster_CKMS2021(N=20, M=15, dim_X=6, theta=1.0, return_type="DataFrame")
    x_cols = [c for c in df.columns if c.startswith("X")]
    data = dml.DoubleMLData(df, "Y", "D", x_cols=x_cols, z_cols="Z",
                            cluster_cols=["cluster_var_i", "cluster_var_j"])
    np.random.seed(7)
    ref = fit_doubleml(dml.DoubleMLPLIV, data, ml_l=LinearRegression(), ml_m=LinearRegression(),
                       ml_r=LinearRegression())
    ours = dml_rs.PLIV(LinearRegression(), LinearRegression(), LinearRegression(), n_folds=3).fit(
        df[x_cols], df["Y"], df["D"], df["Z"], folds=ref.smpls,
        cluster=df[["cluster_var_i", "cluster_var_j"]])
    assert ours.n_clusters_ == 15
    assert_same_inference(ours, ref)


def test_bootstrap_is_rejected_with_clusters():
    np.random.seed(14)
    x, y, d = make_plr_CCDDHNR2018(n_obs=200, dim_x=3, return_type="array")
    model = dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d, cluster=np.arange(200) // 4)
    with pytest.raises(ValueError, match="clustered"):
        model.bootstrap()
    with pytest.raises(ValueError, match="one or two columns"):
        dml_rs.PLR(LinearRegression(), LinearRegression()).fit(x, y, d, cluster=np.zeros((200, 3)))


@pytest.mark.parametrize("kind", ["plr", "irm"])
def test_nuisance_loss_and_evaluate_learners_match_doubleml(kind):
    np.random.seed(15)
    if kind == "plr":
        x, y, d = make_plr_CCDDHNR2018(n_obs=400, dim_x=5, return_type="array")
        ref_cls, learners = dml.DoubleMLPLR, dict(ml_l=LinearRegression(), ml_m=LinearRegression())
        ours_model = dml_rs.PLR(LinearRegression(), LinearRegression(), n_folds=3, n_rep=2)
    else:
        x, y, d = make_irm_data(n_obs=400, dim_x=5, return_type="array")
        ref_cls, learners = dml.DoubleMLIRM, dict(ml_g=LinearRegression(), ml_m=LOGIT)
        ours_model = dml_rs.IRM(LinearRegression(), LOGIT, n_folds=3, n_rep=2)
    ref = fit_doubleml(ref_cls, dml.DoubleMLData.from_arrays(x, y, d), n_rep=2, **learners)
    ours = ours_model.fit(x, y, d, folds=ref.smpls)

    assert set(ours.nuisance_loss_) == set(ref.nuisance_loss)
    for name, loss in ref.nuisance_loss.items():
        assert ours.nuisance_loss_[name].shape == loss.shape == (2, 1)
        np.testing.assert_allclose(ours.nuisance_loss_[name], loss, rtol=TOL)

    def mae(y_true, y_pred):
        keep = ~np.isnan(y_true)
        return mean_absolute_error(y_true[keep], y_pred[keep])

    for metric in (None, mae):
        theirs = ref.evaluate_learners() if metric is None else ref.evaluate_learners(metric=metric)
        mine = ours.evaluate_learners(metric=metric)
        for name, value in theirs.items():
            np.testing.assert_allclose(mine[name], value, rtol=TOL)
    only_m = ours.evaluate_learners(learners=["ml_m"])
    assert list(only_m) == ["ml_m"]

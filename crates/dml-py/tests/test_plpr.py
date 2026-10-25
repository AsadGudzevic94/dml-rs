import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.plm.datasets import make_plpr_CP2025
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

TOL = 1e-9
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def panel(seed, drop=0, time_type="int"):
    np.random.seed(seed)
    df = make_plpr_CP2025(num_id=60, num_t=6, dim_x=4, theta=0.5, dgp_type="dgp1", time_type=time_type)
    if drop:
        df = df.drop(index=np.random.choice(len(df), size=drop, replace=False)).reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def fit_doubleml(df, **kwargs):
    x_cols = [c for c in df.columns if c.startswith("x")]
    data = dml.DoubleMLPanelData(df, "y", "d", "time", "id", x_cols=x_cols, static_panel=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(1)
        return dml.DoubleMLPLPR(data, n_folds=5, **kwargs).fit()


def ours(df, folds, **kwargs):
    x = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    return dml_rs.PLPR(**kwargs).fit(x, df["y"], df[["d"]], df["id"], df["time"], folds=folds)


@pytest.mark.parametrize("approach", ["fd_exact", "wg_approx", "cre_general", "cre_normal"])
@pytest.mark.parametrize("score", ["partialling out", "IV-type"])
def test_plpr_matches_doubleml(approach, score):
    df = panel(11, drop=20)
    kwargs = dict(ml_l=LinearRegression(), ml_m=LinearRegression(), score=score, approach=approach)
    if score == "IV-type":
        kwargs["ml_g"] = LinearRegression()
    ref = fit_doubleml(df, n_rep=2, **kwargs)
    fit = ours(df, ref.smpls, **kwargs)
    np.testing.assert_allclose(fit.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(fit.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(fit.all_coef_, ref.all_coef, rtol=TOL)
    np.testing.assert_allclose(fit.confint(), ref.confint().to_numpy(), rtol=TOL)
    assert fit.treatment_names_ == list(ref.data_transform.d_cols)
    for name, loss in ref.nuisance_loss.items():
        np.testing.assert_allclose(fit.nuisance_loss_[name], loss, rtol=TOL)
    assert fit.n_clusters_ == len(np.unique(ref.data_transform.cluster_vars))


def test_plpr_binary_treatment_with_classifier():
    df = panel(12)
    df["d"] = (df["d"] > df["d"].median()).astype(float)
    kwargs = dict(ml_l=LinearRegression(), ml_m=LOGIT, approach="cre_general")
    ref = fit_doubleml(df, **kwargs)
    fit = ours(df, ref.smpls, **kwargs)
    np.testing.assert_allclose(fit.coef_, ref.coef, rtol=1e-7)
    np.testing.assert_allclose(fit.se_, ref.se, rtol=1e-7)


def test_plpr_dates_and_string_ids():
    df = panel(13, time_type="datetime")
    ref = fit_doubleml(df, ml_l=LinearRegression(), ml_m=LinearRegression())
    df["id"] = "unit_" + df["id"].astype(str).str.zfill(3)
    fit = ours(df, ref.smpls, ml_l=LinearRegression(), ml_m=LinearRegression())
    np.testing.assert_allclose(fit.coef_, ref.coef, rtol=TOL)


def test_plpr_defaults_and_errors():
    df = panel(14)
    x = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    fit = dml_rs.PLPR(dml_rs.Ols(), dml_rs.Ols()).fit(x, df["y"], df["d"], df["id"], df["time"])
    assert abs(fit.coef_[0] - 0.5) < 4 * fit.se_[0]
    with pytest.raises(ValueError, match="approach"):
        dml_rs.PLPR(dml_rs.Ols(), dml_rs.Ols(), approach="fe").fit(  # type: ignore[arg-type]
            x, df["y"], df["d"], df["id"], df["time"])
    with pytest.raises(ValueError, match="classifier"):
        dml_rs.PLPR(dml_rs.Ols(), LOGIT).fit(x, df["y"], df["d"] > 0, df["id"], df["time"])
    dup = df.copy()
    dup.loc[1, ["id", "time"]] = dup.loc[0, ["id", "time"]].to_numpy()
    with pytest.raises(ValueError, match="once"):
        dml_rs.PLPR(dml_rs.Ols(), dml_rs.Ols()).fit(x, dup["y"], dup["d"], dup["id"], dup["time"])

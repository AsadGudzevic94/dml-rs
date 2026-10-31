import doubleml as dml
import numpy as np
import pandas as pd
import pytest
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plr_CCDDHNR2018
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-9


def folds(n, n_folds=5, n_rep=1, seed=0):
    return [list(KFold(n_folds, shuffle=True, random_state=seed + r).split(np.zeros(n))) for r in range(n_rep)]


def doubleml_fit(cls, data, reps, **kwargs):
    model = cls(data, n_folds=len(reps[0]), n_rep=len(reps), draw_sample_splitting=False, **kwargs)
    model.set_sample_splitting(reps)
    return model.fit(store_predictions=True)


@pytest.fixture(scope="module")
def plr_data():
    np.random.seed(3)
    return make_plr_CCDDHNR2018(n_obs=400, dim_x=8, alpha=0.5, return_type="array")


@pytest.mark.parametrize("learner", [LinearRegression(), dml_rs.Ols()], ids=["sklearn", "native"])
@pytest.mark.parametrize("score", ["partialling out", "IV-type"])
def test_plr_matches_doubleml(plr_data, learner, score):
    x, y, d = plr_data
    reps = folds(len(y), n_rep=2)
    ref = doubleml_fit(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d), reps,
                       ml_l=LinearRegression(), ml_m=LinearRegression(),
                       ml_g=LinearRegression() if score == "IV-type" else None, score=score)
    ours = dml_rs.PLR(learner, learner, ml_g=learner if score == "IV-type" else None, score=score).fit(
        x, y, d, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    np.testing.assert_allclose(ours.pval_, ref.pval, rtol=1e-7, atol=1e-300)
    np.testing.assert_allclose(ours.confint(), ref.confint().to_numpy(), rtol=TOL)
    np.testing.assert_allclose(ours.predictions_["ml_l"], ref.predictions["ml_l"], rtol=TOL, atol=1e-12)


@pytest.mark.parametrize("score", ["ATE", "ATTE"])
def test_irm_matches_doubleml_with_sklearn_learners(score):
    np.random.seed(5)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    reps = folds(len(y))
    logit = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
    ref = doubleml_fit(dml.DoubleMLIRM, dml.DoubleMLData.from_arrays(x, y, d), reps,
                       ml_g=LinearRegression(), ml_m=logit, score=score)
    ours = dml_rs.IRM(LinearRegression(), logit, score=score).fit(x, y, d, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    # The native Rust logistic regression solves the same objective.
    native = dml_rs.IRM(dml_rs.Ols(), dml_rs.LogisticRegression(), score=score).fit(x, y, d, folds=reps)
    np.testing.assert_allclose(native.coef_, ref.coef, rtol=1e-7)


def test_pliv_matches_doubleml():
    np.random.seed(7)
    x, y, d, z = make_pliv_CHS2015(n_obs=400, dim_x=10, dim_z=1, return_type="array")
    z = np.asarray(z).ravel()
    reps = folds(len(y))
    ref = doubleml_fit(dml.DoubleMLPLIV, dml.DoubleMLData.from_arrays(x, y, d, z=z), reps,
                       ml_l=LinearRegression(), ml_m=LinearRegression(), ml_r=LinearRegression())
    ours = dml_rs.PLIV(dml_rs.Ols(), dml_rs.Ols(), dml_rs.Ols()).fit(x, y, d, z, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)


def test_multiple_treatments_dataframe_summary_and_bootstrap():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 4))
    d = pd.DataFrame({"price": x[:, 0] + rng.normal(size=n), "ads": x[:, 1] + rng.normal(size=n)})
    y = 0.5 * d["price"] - 0.2 * d["ads"] + x[:, 0] + rng.normal(size=n)
    model = dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols(), n_rep=3).fit(x, y, d)
    assert model.treatment_names_ == ["price", "ads"]
    summary = model.summary
    assert list(summary.index) == ["price", "ads"]
    assert summary.loc["price", "coef"] == pytest.approx(0.5, abs=0.15)
    with pytest.raises(ValueError, match="bootstrap"):
        model.confint(joint=True)
    model.bootstrap(n_boot=200, random_state=1)
    joint, pointwise = model.confint(joint=True), model.confint()
    assert joint.shape == (2, 2)
    assert np.all(joint[:, 1] - joint[:, 0] > pointwise[:, 1] - pointwise[:, 0])
    assert model.boot_t_stat_.shape == (200, 2, 3)
    assert model.p_adjust("romano-wolf").shape == (2,)
    assert np.all(model.p_adjust("holm") >= model.pval_ - 1e-12)


def test_random_forest_learners_run_on_worker_threads():
    """Python learners are called from Rust's rayon threads with the GIL released."""
    rng = np.random.default_rng(1)
    n = 600
    x = rng.uniform(-2, 2, size=(n, 3))
    d = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 + rng.normal(size=n)
    y = 0.7 * d + np.cos(x[:, 0]) + x[:, 2] + rng.normal(size=n)
    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=0)
    reps = folds(n, n_folds=4)
    model = dml_rs.PLR(rf, rf).fit(x, y, d, folds=reps)
    # A seeded forest is deterministic, so DoubleML on the same folds must agree exactly.
    ref = doubleml_fit(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_l=clone(rf), ml_m=clone(rf))
    np.testing.assert_allclose(model.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(model.se_, ref.se, rtol=TOL)
    assert abs(model.coef_[0] - 0.7) < 0.2


class Exploding(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        raise ZeroDivisionError("boom")

    def predict(self, X):
        return np.zeros(len(X))


def test_errors_are_propagated(plr_data):
    x, y, d = plr_data
    with pytest.raises(RuntimeError, match="boom"):
        dml_rs.PLR(Exploding(), dml_rs.Ols()).fit(x, y, d)
    with pytest.raises(ValueError, match="score"):
        dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols(), score="nope").fit(x, y, d)
    with pytest.raises(ValueError, match="binary"):
        dml_rs.IRM(dml_rs.Ols(), dml_rs.LogisticRegression()).fit(x, y, d)
    with pytest.raises(TypeError, match="has no"):
        dml_rs.PLR(object(), dml_rs.Ols()).fit(x, y, d)
    with pytest.raises(AttributeError, match="not fitted"):
        dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()).confint()


def test_sklearn_parameter_protocol():
    model = dml_rs.PLR(LinearRegression(), LinearRegression(), n_folds=3)
    assert model.get_params()["n_folds"] == 3
    model.set_params(n_rep=4)
    assert model.n_rep == 4
    copy = clone(model)
    assert copy is not model and copy.n_rep == 4
    with pytest.raises(ValueError):
        model.set_params(bogus=1)


def test_native_learners_standalone():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 3))
    y = 1.0 + x @ np.array([2.0, 0.0, -1.0]) + 0.01 * rng.normal(size=200)
    np.testing.assert_allclose(dml_rs.Ols().fit(x, y).predict(x), LinearRegression().fit(x, y).predict(x), atol=1e-10)
    labels = (x[:, 0] + rng.normal(size=200) > 0).astype(float)
    p = dml_rs.LogisticRegression().fit(x, labels).predict_proba(x)[:, 1]
    assert p.shape == (200,) and np.all((p > 0) & (p < 1))

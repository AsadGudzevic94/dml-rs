import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn import linear_model as sk
from sklearn.model_selection import KFold

import dml_rs


def logit_cv_sklearn(**kwargs):
    # Exact solver on scikit-learn's side, so both reach the same optimum.
    return sk.LogisticRegressionCV(solver="newton-cholesky", tol=1e-14, max_iter=1000, l1_ratios=(0,),
                                   use_legacy_attributes=False, **kwargs)


@pytest.mark.parametrize("n, p", [(200, 10), (400, 60), (100, 150)])
def test_lasso_cv_matches_scikit_learn(n, p):
    np.random.seed(n + p)
    x, y, _ = make_plr_CCDDHNR2018(n_obs=n, dim_x=p, return_type="array")
    # With n > p, scikit-learn's defaults. With p > n a path near interpolation makes the default
    # tol = 1e-4 stop at points that depend on BLAS rounding (both within tolerance, but not the
    # same point on every platform), so both sides solve to a tight tolerance on a shorter path.
    kwargs = {} if n > p else dict(alphas=30, eps=1e-2, tol=1e-8, max_iter=1_000_000)
    ref = sk.LassoCV(**kwargs).fit(x, y)
    ours = dml_rs.LassoCV(**kwargs).fit(x, y)
    assert ours.alpha_ == pytest.approx(ref.alpha_, rel=1e-12)
    np.testing.assert_allclose(ours.alphas_, ref.alphas_, rtol=1e-12)
    np.testing.assert_allclose(ours.mse_path_, ref.mse_path_, rtol=1e-9)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-9, atol=1e-12)
    assert ours.intercept_ == pytest.approx(ref.intercept_, rel=1e-9)
    np.testing.assert_allclose(ours.predict(x), ref.predict(x), rtol=1e-9)


def test_lasso_cv_grids_and_folds():
    np.random.seed(3)
    x, y, _ = make_plr_CCDDHNR2018(n_obs=300, dim_x=20, return_type="array")
    for kwargs in [dict(alphas=30, eps=1e-2, cv=3), dict(alphas=[0.5, 0.01, 0.1, 0.05])]:
        ref = sk.LassoCV(**kwargs).fit(x, y)
        ours = dml_rs.LassoCV(**kwargs).fit(x, y)
        assert ours.alpha_ == pytest.approx(ref.alpha_, rel=1e-12)
        np.testing.assert_allclose(ours.alphas_, ref.alphas_, rtol=1e-12)
        np.testing.assert_allclose(ours.mse_path_, ref.mse_path_, rtol=1e-9)
    # Lasso's default stopping rule is scikit-learn's as well.
    np.testing.assert_allclose(dml_rs.Lasso(alpha=0.05).fit(x, y).predict(x),
                               sk.Lasso(alpha=0.05).fit(x, y).predict(x), rtol=1e-9)
    with pytest.raises(AttributeError, match="not fitted"):
        dml_rs.LassoCV().alpha_


@pytest.mark.parametrize("scoring", ["neg_log_loss", "accuracy"])
@pytest.mark.parametrize("n, p", [(300, 5), (600, 20)])
def test_logistic_regression_cv_matches_scikit_learn(scoring, n, p):
    np.random.seed(n + p)
    x, _, d = make_irm_data(n_obs=n, dim_x=p, return_type="array")
    ref = logit_cv_sklearn(scoring=scoring).fit(x, d)
    ours = dml_rs.LogisticRegressionCV(scoring=scoring).fit(x, d)
    assert ours.C_ == pytest.approx(float(np.ravel(ref.C_)[0]), rel=1e-12)
    np.testing.assert_allclose(ours.Cs_, ref.Cs_, rtol=1e-12)
    np.testing.assert_allclose(ours.scores_, np.asarray(ref.scores_).reshape(5, -1), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(ours.coef_, np.ravel(ref.coef_), rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(ours.predict_proba(x), ref.predict_proba(x)[:, 1], rtol=1e-8, atol=1e-12)


def test_logistic_regression_cv_options():
    np.random.seed(4)
    x, _, d = make_irm_data(n_obs=400, dim_x=5, return_type="array")
    ref = logit_cv_sklearn(Cs=[10.0, 0.01, 1.0], cv=3, scoring="neg_log_loss").fit(x, d)
    ours = dml_rs.LogisticRegressionCV(Cs=[10.0, 0.01, 1.0], cv=3).fit(x, d)
    assert ours.C_ == float(np.ravel(ref.C_)[0])
    np.testing.assert_allclose(ours.scores_, np.asarray(ref.scores_).reshape(3, -1), rtol=1e-7)
    with pytest.raises(ValueError, match="scoring"):
        dml_rs.LogisticRegressionCV(scoring="roc_auc")


def test_dml_with_cv_learners_matches_doubleml():
    np.random.seed(7)
    x, y, d = make_irm_data(n_obs=500, dim_x=15, theta=0.5, return_type="array")
    folds = list(KFold(5, shuffle=True, random_state=1).split(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = dml.DoubleMLIRM(dml.DoubleMLData.from_arrays(x, y, d), sk.LassoCV(),
                              logit_cv_sklearn(scoring="neg_log_loss"), n_folds=5, draw_sample_splitting=False)
        ref.set_sample_splitting([folds]).fit()
        ref_plr = dml.DoubleMLPLR(dml.DoubleMLData.from_arrays(x, y, d), sk.LassoCV(), sk.LassoCV(),
                                  n_folds=5, draw_sample_splitting=False)
        ref_plr.set_sample_splitting([folds]).fit()
    ours = dml_rs.IRM(dml_rs.LassoCV(), dml_rs.LogisticRegressionCV()).fit(x, y, d, folds=folds)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=1e-7)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=1e-7)
    # Default scikit-learn LassoCV on DoubleML's side, default dml_rs.LassoCV on ours.
    ours_plr = dml_rs.PLR(dml_rs.LassoCV(), dml_rs.LassoCV()).fit(x, y, d, folds=folds)
    np.testing.assert_allclose(ours_plr.coef_, ref_plr.coef, rtol=1e-8)
    np.testing.assert_allclose(ours_plr.se_, ref_plr.se, rtol=1e-8)

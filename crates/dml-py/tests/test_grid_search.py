import doubleml as dml
import numpy as np
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV

import dml_rs

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5]


def test_grid_search_matches_scikit_learn():
    np.random.seed(1)
    x, y, _ = make_plr_CCDDHNR2018(n_obs=300, dim_x=10, return_type="array")
    ref = GridSearchCV(Lasso(), {"alpha": ALPHAS}, scoring="neg_mean_squared_error", cv=5).fit(x, y)
    ours = dml_rs.GridSearchRegressor([Lasso(alpha=a) for a in ALPHAS], cv=5).fit(x, y)
    assert ours.best_index_ == ref.best_index_
    np.testing.assert_allclose(ours.cv_scores_, -ref.cv_results_["mean_test_score"], rtol=1e-12)
    np.testing.assert_allclose(ours.predict(x), ref.predict(x), rtol=1e-12)
    # Built-in candidates choose the same penalty (their solver matches scikit-learn's).
    builtin = dml_rs.GridSearchRegressor([dml_rs.Lasso(alpha=a) for a in ALPHAS]).fit(x, y)
    assert builtin.best_index_ == ref.best_index_


def test_grid_search_as_nuisance_learner_matches_grid_search_cv():
    np.random.seed(2)
    x, y, d = make_plr_CCDDHNR2018(n_obs=300, dim_x=10, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    search = GridSearchCV(Lasso(), {"alpha": ALPHAS}, scoring="neg_mean_squared_error", cv=5)
    np.random.seed(3)
    ref = dml.DoubleMLPLR(data, search, search).fit()
    ours_search = dml_rs.GridSearchRegressor([Lasso(alpha=a) for a in ALPHAS])
    ours = dml_rs.PLR(ours_search, ours_search).fit(x, y, d, folds=ref.smpls)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=1e-9)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=1e-9)


def test_classifier_grid_search_in_irm():
    np.random.seed(4)
    x, y, d = make_irm_data(n_obs=400, dim_x=5, return_type="array")
    cands = [dml_rs.LogisticRegression(c) for c in (0.01, 0.1, 1.0)]
    fit = dml_rs.GridSearchClassifier(cands).fit(x, d)
    assert fit.best_index_ in (0, 1, 2) and len(fit.cv_scores_) == 3
    model = dml_rs.IRM(dml_rs.Ols(), dml_rs.GridSearchClassifier(cands)).fit(x, y, d)
    assert np.isfinite(model.coef_).all()
    p = fit.predict_proba(x)
    assert np.all((p > 0) & (p < 1))
    np.testing.assert_allclose(
        p, LogisticRegression(C=[0.01, 0.1, 1.0][fit.best_index_], solver="newton-cholesky", tol=1e-12)
        .fit(x, d).predict_proba(x)[:, 1], atol=1e-6)

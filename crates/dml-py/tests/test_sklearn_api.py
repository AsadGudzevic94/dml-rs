"""The built-in learners are scikit-learn estimators: scikit-learn's own estimator checks, its
tools (clone, GridSearchCV, Pipeline, splitters) and the DML models' tuning all work with them,
while the DML models still cross-fit them in Rust."""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn import linear_model as sk
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import dml_rs
from doubleml.plm.datasets import make_plr_CCDDHNR2018

SKLEARN_1_6 = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 6)

ESTIMATORS = [
    dml_rs.Ols(),
    dml_rs.Lasso(alpha=0.1),
    dml_rs.LassoCV(cv=3),
    dml_rs.LogisticRegression(),
    dml_rs.LogisticRegressionCV(cv=3),
    dml_rs.RandomForestRegressor(n_estimators=10),
    dml_rs.RandomForestClassifier(n_estimators=10),
    dml_rs.GradientBoostingRegressor(n_estimators=10),
    dml_rs.GradientBoostingClassifier(n_estimators=10),
    dml_rs.HistGradientBoostingRegressor(max_iter=10),
    dml_rs.HistGradientBoostingClassifier(max_iter=10),
    dml_rs.GridSearchRegressor([dml_rs.Ols(), dml_rs.Lasso(alpha=0.1)], cv=3),
    dml_rs.GridSearchClassifier([dml_rs.LogisticRegression(C=0.1), dml_rs.LogisticRegression()], cv=3),
    dml_rs.RegressionForest(n_estimators=20),
    dml_rs.QuantileForest(n_estimators=20),
]


def expected_failures(est):
    fails = {"check_estimators_pickle": "fitted built-in learners cannot be pickled yet"}
    if isinstance(est, (dml_rs.Ols, dml_rs.GridSearchRegressor)):
        fails["check_sample_weight_equivalence_on_dense_data"] = (
            "plain least squares has no unique solution on the check's rank-deficient data")
    if isinstance(est, (dml_rs.RandomForestRegressor, dml_rs.RandomForestClassifier,
                        dml_rs.GradientBoostingRegressor)):
        fails["check_sample_weight_equivalence_on_dense_data"] = (
            "sample counts constrain the trees, as in scikit-learn's own forests and boosting")
    return fails


if SKLEARN_1_6:
    from sklearn.utils.estimator_checks import parametrize_with_checks

    @parametrize_with_checks(ESTIMATORS, expected_failed_checks=expected_failures)
    def test_scikit_learn_estimator_checks(estimator, check):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check(estimator)


def data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(size=n)
    labels = np.where(x[:, 1] + rng.normal(size=n) > 0, "yes", "no")
    return x, y, labels


def test_parameters_clone_and_tools():
    x, y, labels = data()
    est = dml_rs.RandomForestRegressor(n_estimators=20, max_depth=4)
    assert est.get_params()["max_depth"] == 4 and repr(est) == "RandomForestRegressor(max_depth=4, n_estimators=20)"
    copy = clone(est.set_params(min_samples_leaf=5))
    assert copy is not est and copy.get_params() == est.get_params()
    assert is_regressor(dml_rs.Lasso()) and is_classifier(dml_rs.HistGradientBoostingClassifier())

    search = GridSearchCV(dml_rs.Lasso(), {"alpha": [0.01, 0.1, 1.0]}, cv=3).fit(x, y)
    ref = GridSearchCV(sk.Lasso(), {"alpha": [0.01, 0.1, 1.0]}, cv=3).fit(x, y)
    assert search.best_params_ == ref.best_params_
    np.testing.assert_allclose(search.cv_results_["mean_test_score"], ref.cv_results_["mean_test_score"], rtol=1e-10)
    # Unfitted learners pickle, so joblib can run the search in worker processes.
    parallel = GridSearchCV(dml_rs.Lasso(), {"alpha": [0.01, 0.1]}, cv=3, n_jobs=2).fit(x, y)
    assert parallel.best_params_ == GridSearchCV(dml_rs.Lasso(), {"alpha": [0.01, 0.1]}, cv=3).fit(x, y).best_params_

    pipe = make_pipeline(StandardScaler(), dml_rs.Ols())
    np.testing.assert_allclose(cross_val_score(pipe, x, y, cv=3),
                               cross_val_score(make_pipeline(StandardScaler(), sk.LinearRegression()), x, y, cv=3))


def test_classifiers_take_any_two_labels():
    x, _, labels = data()
    frame = pd.DataFrame(x, columns=list("abcd"))
    clf = dml_rs.LogisticRegression().fit(frame, labels)
    ref = sk.LogisticRegression(solver="newton-cholesky", tol=1e-12).fit(frame, labels)
    np.testing.assert_array_equal(clf.classes_, ["no", "yes"])
    np.testing.assert_array_equal(clf.feature_names_in_, list("abcd"))
    np.testing.assert_allclose(clf.predict_proba(frame), ref.predict_proba(frame), atol=1e-8)
    np.testing.assert_array_equal(clf.predict(frame), ref.predict(frame))
    assert clf.score(frame, labels) == pytest.approx(ref.score(frame, labels))
    assert clf.n_iter_ >= 1 and clf.coef_.shape == (4,)
    with pytest.raises(ValueError, match="feature names should match"):
        clf.predict(frame.rename(columns={"a": "z"}))
    with pytest.raises(ValueError, match="Only binary classification"):
        dml_rs.LogisticRegression().fit(x, np.arange(len(x)) % 3)


def test_fitted_attributes():
    x, y, labels = data()
    with pytest.raises(AttributeError, match="not fitted"):
        dml_rs.Lasso().coef_
    lasso = dml_rs.Lasso(alpha=0.05).fit(x, y)
    ref = sk.Lasso(alpha=0.05).fit(x, y)
    np.testing.assert_allclose(lasso.coef_, ref.coef_, rtol=1e-12, atol=1e-14)
    assert lasso.intercept_ == pytest.approx(ref.intercept_, rel=1e-12)
    assert lasso.n_iter_ == ref.n_iter_
    assert dml_rs.LassoCV(cv=3).fit(x, y).n_iter_ >= 1
    assert dml_rs.RandomForestClassifier(n_estimators=5).fit(x, labels).feature_importances_.shape == (4,)


def test_dml_models_cross_fit_built_ins_in_rust(monkeypatch):
    np.random.seed(1)
    x, y, d = make_plr_CCDDHNR2018(n_obs=300, dim_x=5, return_type="array")
    expected = dml_rs.PLR(dml_rs._dml_rs.Lasso(alpha=0.05), dml_rs._dml_rs.Ols()).fit(x, y, d)

    def python_fit(*args, **kwargs):
        raise AssertionError("the DML model fitted the learner through Python")

    monkeypatch.setattr(dml_rs.Lasso, "fit", python_fit)
    monkeypatch.setattr(dml_rs.Ols, "fit", python_fit)
    got = dml_rs.PLR(dml_rs.Lasso(alpha=0.05), dml_rs.Ols()).fit(x, y, d)
    np.testing.assert_array_equal(got.coef_, expected.coef_)
    np.testing.assert_array_equal(got.se_, expected.se_)


def test_tuning_built_in_learners_matches_scikit_learn():
    np.random.seed(2)
    x, y, d = make_plr_CCDDHNR2018(n_obs=300, dim_x=8, return_type="array")
    grids = {"ml_l": {"alpha": [0.01, 0.05, 0.2]}, "ml_m": {"alpha": [0.01, 0.05, 0.2]}}
    fits = []
    for learner in (dml_rs.Lasso(), sk.Lasso()):
        model = dml_rs.PLR(clone(learner), clone(learner), n_folds=3)
        np.random.seed(3)
        model.tune(x, y, d, param_grids=grids, n_folds_tune=3)
        fits.append((model.params, model.fit(x, y, d).coef_))
    assert fits[0][0] == fits[1][0]
    np.testing.assert_allclose(fits[0][1], fits[1][1], rtol=1e-9)

    # Fold-specific parameters on a built-in learner.
    model = dml_rs.PLR(dml_rs.Lasso(), dml_rs.Lasso(), n_folds=2)
    model.set_ml_nuisance_params("ml_l", "d", [[{"alpha": 0.01}, {"alpha": 0.5}]])
    model.fit(x, y, d, store_models=True)
    assert [m.alpha for m in model.models_["ml_l"]["d"][0]] == [0.01, 0.5]


def test_optuna_tuning_of_built_in_learners():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(4)
    x, y, d = make_plr_CCDDHNR2018(n_obs=300, dim_x=5, return_type="array")

    def space(trial):
        return {"alpha": trial.suggest_float("alpha", 1e-3, 0.5, log=True)}

    best = []
    for learner in (dml_rs.Lasso(), sk.Lasso()):
        model = dml_rs.PLR(clone(learner), clone(learner), n_folds=3)
        res = model.tune_ml_models(x, y, d, ml_param_space={"ml_l": space, "ml_m": space}, return_tune_res=True,
                                   optuna_settings={"n_trials": 5, "sampler": optuna.samplers.TPESampler(seed=1)})
        best.append({k: r.best_params for k, r in res[0].items() if r is not None})
    assert best[0] == best[1]


def test_cv_splitters():
    x, y, _ = data(300)
    groups = np.arange(300) % 7
    assert dml_rs.LassoCV(cv=KFold(5)).fit(x, y).alpha_ == dml_rs.LassoCV(cv=5).fit(x, y).alpha_
    splits = list(GroupKFold(3).split(x, y, groups))
    ours = dml_rs.LassoCV(cv=splits).fit(x, y)
    ref = sk.LassoCV(cv=splits).fit(x, y)
    assert ours.alpha_ == pytest.approx(ref.alpha_, rel=1e-12)
    np.testing.assert_allclose(ours.mse_path_, ref.mse_path_, rtol=1e-9)

    labels = (x[:, 0] > 0).astype(float)
    a = dml_rs.LogisticRegressionCV(cv=StratifiedKFold(4)).fit(x, labels)
    b = dml_rs.LogisticRegressionCV(cv=4).fit(x, labels)
    assert a.C_ == b.C_
    np.testing.assert_allclose(a.scores_, b.scores_)

    # A splitter needs the data, so DML models fit such a learner through Python, fold by fold.
    shuffled = dml_rs.LassoCV(cv=KFold(3, shuffle=True, random_state=0))
    assert shuffled._dml_native() is None
    np.random.seed(5)
    xx, yy, dd = make_plr_CCDDHNR2018(n_obs=200, dim_x=5, return_type="array")
    assert np.isfinite(dml_rs.PLR(shuffled, dml_rs.Ols(), n_folds=2).fit(xx, yy, dd).coef_).all()


def test_sample_weights_match_scikit_learn():
    x, y, _ = data(300)
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 3.0, size=300)
    labels = (x[:, 0] + rng.normal(size=300) > 0).astype(float)
    ours = dml_rs.LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3).fit(x, labels, sample_weight=w)
    ref = sk.LogisticRegressionCV(Cs=[0.01, 0.1, 1.0, 10.0], cv=3, solver="newton-cholesky", tol=1e-14,
                                  max_iter=1000, l1_ratios=(0,), use_legacy_attributes=False,
                                  scoring="neg_log_loss").fit(x, labels, sample_weight=w)
    assert ours.C_ == float(np.ravel(ref.C_)[0])
    np.testing.assert_allclose(ours.predict_proba(x), ref.predict_proba(x), rtol=1e-7)

    from sklearn.ensemble import GradientBoostingClassifier
    gb = dml_rs.GradientBoostingClassifier(n_estimators=30).fit(x, labels, sample_weight=w)
    gb_ref = GradientBoostingClassifier(n_estimators=30, random_state=0).fit(x, labels, sample_weight=w)
    np.testing.assert_allclose(gb.predict_proba(x), gb_ref.predict_proba(x), atol=1e-10)
    rf = dml_rs.RandomForestClassifier(n_estimators=30).fit(x, labels, sample_weight=w)
    assert rf.predict_proba(x).shape == (300, 2)


def test_without_scikit_learn():
    """scikit-learn is optional: the stand-in base class gives the parameter API and score."""
    code = textwrap.dedent("""
        import sys
        sys.modules["sklearn"] = None  # any import of scikit-learn now fails
        import numpy as np
        import dml_rs
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3)); y = x[:, 0] + rng.normal(size=200); d = (x[:, 1] > 0).astype(float)
        est = dml_rs.Lasso(alpha=0.1).set_params(tol=1e-6)
        assert est.get_params() == {"alpha": 0.1, "max_iter": 100000, "tol": 1e-6}, est.get_params()
        assert repr(est) == "Lasso(alpha=0.1, tol=1e-06)", repr(est)
        assert 0 < est.fit(x, y).score(x, y) < 1
        clf = dml_rs.LogisticRegression().fit(x, d)
        assert clf.predict_proba(x).shape == (200, 2) and 0.5 < clf.score(x, d) <= 1
        try:
            dml_rs.Lasso().set_params(beta=1)
        except ValueError as e:
            assert "Invalid parameter" in str(e)
        else:
            raise AssertionError("set_params accepted an unknown parameter")
        plr = dml_rs.PLR(dml_rs.Lasso(alpha=0.05), dml_rs.Ols()).fit(x, y, d)
        assert np.isfinite(plr.coef_).all()
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr

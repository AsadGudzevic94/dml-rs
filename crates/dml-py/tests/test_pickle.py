"""Fitted learners and models survive pickle / joblib round trips unchanged."""

import io
import pickle

import joblib
import numpy as np
import pandas as pd
import pytest

import dml_rs
from dml_rs import datasets as ds

OLS, LOGIT = dml_rs.Ols(), dml_rs.LogisticRegression()
# A constant treatment column inside a treatment arm needs a small penalty.
RIDGE = dml_rs.Ols(ridge=0.1)


def round_trip(obj):
    return pickle.loads(pickle.dumps(obj))


def fitted_arrays(obj):
    """Public fitted attributes (``name_``) that are numbers, arrays or frames."""
    out = {}
    for key, value in vars(obj).items():
        if key.endswith("_") and not key.startswith("_"):
            if isinstance(value, (pd.DataFrame, pd.Series)):
                out[key] = value.to_numpy()
            elif isinstance(value, (np.ndarray, float, int, np.floating)):
                out[key] = np.asarray(value)
    return out


def assert_same(a, b):
    fa, fb = fitted_arrays(a), fitted_arrays(b)
    assert fa.keys() == fb.keys() and fa
    for key in fa:
        np.testing.assert_array_equal(fa[key], fb[key], err_msg=key)


LEARNERS = [
    dml_rs.Ols(ridge=0.1), dml_rs.Lasso(alpha=0.05), dml_rs.LassoCV(cv=3), dml_rs.RandomForestRegressor(n_estimators=20),
    dml_rs.GradientBoostingRegressor(n_estimators=20), dml_rs.HistGradientBoostingRegressor(max_iter=20),
    dml_rs.RegressionForest(n_estimators=50), dml_rs.QuantileForest(n_estimators=50),
    dml_rs.LogisticRegression(), dml_rs.LogisticRegressionCV(cv=3), dml_rs.RandomForestClassifier(n_estimators=20),
    dml_rs.GradientBoostingClassifier(n_estimators=20), dml_rs.HistGradientBoostingClassifier(max_iter=20),
]


@pytest.mark.parametrize("learner", LEARNERS, ids=lambda est: type(est).__name__)
def test_fitted_learners(learner):
    x, y, d = ds.make_irm_data(n_obs=300, dim_x=4, random_state=1)
    target = np.where(d > 0, "treated", "control") if hasattr(learner, "predict_proba") else y
    fitted = learner.fit(pd.DataFrame(x, columns=list("abcd")), target)
    frame = pd.DataFrame(x, columns=list("abcd"))
    for back in (round_trip(fitted), joblib_round_trip(fitted)):
        np.testing.assert_array_equal(back.predict(frame), fitted.predict(frame))
        if hasattr(fitted, "predict_proba"):
            np.testing.assert_array_equal(back.predict_proba(frame), fitted.predict_proba(frame))
            np.testing.assert_array_equal(back.classes_, fitted.classes_)
        np.testing.assert_array_equal(back.feature_names_in_, fitted.feature_names_in_)


def joblib_round_trip(obj):
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    return joblib.load(buffer)


def _panel():
    panel = ds.make_plpr_CP2025(num_id=40, num_t=4, dim_x=4, random_state=7)
    return dml_rs.DmlData(panel, "y", "d", id_col="id", t_col="time")


def _staggered():
    stag = ds.make_did_CS2021(n_obs=300, random_state=8)
    return dml_rs.DmlData(stag, "y", "d", x_cols=["Z1", "Z2", "Z3", "Z4"], id_col="id", t_col="t")


def _binary_outcome():
    x, y, d = ds.make_irm_data(n_obs=300, dim_x=4, random_state=2)
    return dml_rs.DmlData.from_arrays(x, (y > np.median(y)).astype(float), d)


def _rdd():
    rdd = ds.make_simple_rdd_data(n_obs=600, fuzzy=False, random_state=9)
    return dml_rs.DmlData.from_arrays(rdd["X"], rdd["Y"], rdd["D"], score=rdd["score"])


def _discrete():
    r = ds.make_irm_data_discrete_treatments(n_obs=400, random_state=1)
    return dml_rs.DmlData.from_arrays(r["x"], r["y"], r["d"])


IRM = lambda: ds.make_irm_data(n_obs=300, dim_x=4, return_type="DmlData", random_state=3)  # noqa: E731
IIVM = lambda: ds.make_iivm_data(n_obs=300, dim_x=4, return_type="DmlData", random_state=4)  # noqa: E731

MODELS = {
    "PLR": (lambda: dml_rs.PLR(OLS, OLS), lambda: ds.make_plr_CCDDHNR2018(n_obs=300, return_type="DmlData",
                                                                         random_state=1)),
    "IRM": (lambda: dml_rs.IRM(OLS, LOGIT), IRM),
    "PLIV": (lambda: dml_rs.PLIV(OLS, OLS, OLS), lambda: ds.make_pliv_CHS2015(n_obs=300, dim_z=1, return_type="DmlData",
                                                                             random_state=5)),
    "IIVM": (lambda: dml_rs.IIVM(OLS, LOGIT, LOGIT), IIVM),
    "DID": (lambda: dml_rs.DID(OLS, LOGIT), lambda: ds.make_did_SZ2020(n_obs=300, return_type="DmlData", random_state=6)),
    "DIDCS": (lambda: dml_rs.DIDCS(OLS, LOGIT),
              lambda: ds.make_did_SZ2020(n_obs=300, cross_sectional_data=True, return_type="DmlData", random_state=6)),
    "SSM": (lambda: dml_rs.SSM(RIDGE, LOGIT, LOGIT), lambda: ds.make_ssm_data(n_obs=400, return_type="DmlData",
                                                                           random_state=5)),
    "LPLR": (lambda: dml_rs.LPLR(LOGIT, OLS, OLS, n_folds_inner=3), _binary_outcome),
    "PLPR": (lambda: dml_rs.PLPR(OLS, OLS), _panel),
    "APO": (lambda: dml_rs.APO(RIDGE, LOGIT, treatment_level=1), IRM),
    "APOS": (lambda: dml_rs.APOS(RIDGE, LOGIT, treatment_levels=[0, 1]), IRM),
    "PQ": (lambda: dml_rs.PQ(LOGIT, LOGIT), IRM),
    "CVAR": (lambda: dml_rs.CVAR(OLS, LOGIT), IRM),
    "LPQ": (lambda: dml_rs.LPQ(LOGIT, LOGIT), IIVM),
    "QTE": (lambda: dml_rs.QTE(LOGIT, quantiles=[0.25, 0.5]), IRM),
    "RDFlex": (lambda: dml_rs.RDFlex(OLS), _rdd),
    "StaggeredDID": (lambda: dml_rs.StaggeredDID(OLS, LOGIT), _staggered),
    "AutoDML": (lambda: dml_rs.AutoDML(OLS), IRM),
    "CausalForest": (lambda: dml_rs.CausalForest(OLS, LOGIT, n_estimators=100), IRM),
    "InstrumentalForest": (lambda: dml_rs.InstrumentalForest(OLS, LOGIT, LOGIT, n_estimators=100), IIVM),
    "MultiArmCausalForest": (lambda: dml_rs.MultiArmCausalForest(OLS, LOGIT, n_estimators=100), _discrete),
}


@pytest.mark.parametrize("name", MODELS)
def test_fitted_models(name):
    make, data = MODELS[name]
    model = make().fit(data())
    back = round_trip(model)
    assert_same(back, model)
    if hasattr(model, "confint"):
        np.testing.assert_array_equal(np.asarray(back.confint()), np.asarray(model.confint()))


def test_follow_up_analyses_after_loading():
    data = IRM()
    model = round_trip(dml_rs.IRM(OLS, LOGIT).fit(data))
    reference = dml_rs.IRM(OLS, LOGIT).fit(data)
    for m in (model, reference):
        m.bootstrap(n_boot=200).sensitivity_analysis()
    np.testing.assert_array_equal(model.confint(joint=True), reference.confint(joint=True))
    assert model.sensitivity_summary == reference.sensitivity_summary
    x = data.data[data.x_cols].to_numpy()
    groups = pd.get_dummies(x[:, 0] > 0).astype(float)
    np.testing.assert_array_equal(model.gate(groups).summary.to_numpy(), reference.gate(groups).summary.to_numpy())
    # Results computed after loading pickle too.
    blp = round_trip(model.cate(np.column_stack([np.ones(len(x)), x[:, 0]])))
    assert blp.summary.shape[0] == 2
    tree = model.policy_tree(x[:, :2], depth=2)
    np.testing.assert_array_equal(round_trip(tree).predict(x[:, :2]), tree.predict(x[:, :2]))
    bench = round_trip(dml_rs.PLR(OLS, OLS).fit(data).sensitivity_benchmark(["X1"]))
    assert bench.shape[0] == 1


def test_learners_inside_unfitted_and_stored_models():
    data = IRM()
    model = dml_rs.IRM(dml_rs.HistGradientBoostingRegressor(max_iter=10), LOGIT, n_folds=2)
    model.fit(data, store_models=True)
    back = round_trip(model)
    x = data.data[data.x_cols].to_numpy()
    first, again = model.models_["ml_m"]["d"][0][0], back.models_["ml_m"]["d"][0][0]
    np.testing.assert_array_equal(first.predict_proba(x), again.predict_proba(x))


def test_not_yet_picklable_objects_say_so():
    x, y, d = ds.make_irm_data(n_obs=200, dim_x=3, random_state=1)
    with pytest.raises(TypeError, match="pickle"):
        pickle.dumps(dml_rs.DRLearner(OLS, LOGIT, OLS).fit(x, y, d))
    with pytest.raises(TypeError, match="pickle"):
        pickle.dumps(dml_rs.GridSearchRegressor([OLS, dml_rs.Lasso()], cv=3).fit(x, y))


def test_pickles_are_tied_to_the_version():
    rng = np.random.default_rng(0)
    fitted = dml_rs._dml_rs.Ols().fit(rng.normal(size=(20, 2)), rng.normal(size=20))
    restore, (kind, data) = fitted.__reduce__()
    assert data.startswith(f"dml-rs/{dml_rs.__version__}\n".encode())
    with pytest.raises(ValueError, match="same dml-rs version"):
        restore(kind, b"dml-rs/0.0.1\n" + data.split(b"\n", 1)[1])

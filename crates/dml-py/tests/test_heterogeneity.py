import warnings

import doubleml as dml
import numpy as np
import pandas as pd
import pytest
from doubleml.irm.datasets import make_irm_data, make_irm_data_discrete_treatments
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-7
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)
SCENARIO = dict(cf_y=0.04, cf_d=0.03, rho=0.8, level=0.95)


def folds(n, n_rep=1):
    return [list(KFold(5, shuffle=True, random_state=r).split(np.zeros(n))) for r in range(n_rep)]


def doubleml(cls, data, reps, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(data, n_folds=5, n_rep=len(reps), draw_sample_splitting=False, **kwargs)
        model.set_sample_splitting(reps)
        return model.fit()


def assert_same_blp(ours, ref):
    np.testing.assert_allclose(ours.summary["coef"], ref.summary["coef"], rtol=TOL)
    np.testing.assert_allclose(ours.summary["std err"], ref.summary["std err"], rtol=TOL)


@pytest.mark.parametrize("score", ["partialling out", "IV-type"])
def test_plr_cate_and_gate_match_doubleml(score):
    np.random.seed(21)
    x, y, d = make_plr_CCDDHNR2018(n_obs=600, dim_x=6, return_type="array")
    reps = folds(len(y), n_rep=2)
    learners = dict(ml_l=LinearRegression(), ml_m=LinearRegression())
    if score == "IV-type":
        learners["ml_g"] = LinearRegression()
    ref = doubleml(dml.DoubleMLPLR, dml.DoubleMLData.from_arrays(x, y, d), reps, score=score, **learners)
    ours = dml_rs.PLR(**learners, score=score).fit(x, y, d, folds=reps)

    basis = pd.DataFrame({"const": 1.0, "x1": x[:, 0], "x2": x[:, 1]})
    ref_cate, ours_cate = ref.cate(basis), ours.cate(basis)
    assert_same_blp(ours_cate, ref_cate)
    new = basis.iloc[:7]
    np.testing.assert_allclose(ours_cate.confint(new).to_numpy(), ref_cate.confint(new).to_numpy(), rtol=1e-6)

    groups = pd.DataFrame({"low": x[:, 0] <= 0, "high": x[:, 0] > 0})
    assert_same_blp(ours.gate(groups.astype(float)), ref.gate(groups))


def test_irm_weights_match_doubleml():
    np.random.seed(22)
    x, y, d = make_irm_data(n_obs=800, dim_x=5, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    reps = folds(len(y), n_rep=2)
    w = (x[:, 0] > 0).astype(float)
    cases = [
        ("ATE", w),
        ("ATE", np.exp(0.3 * x[:, 1])),
        ("ATTE", w),
        ("ATE", {"weights": w, "weights_bar": np.column_stack([w, 0.5 + w])}),
    ]
    for score, weights in cases:
        ref = doubleml(dml.DoubleMLIRM, data, reps, ml_g=LinearRegression(), ml_m=LOGIT, score=score, weights=weights)
        ours = dml_rs.IRM(LinearRegression(), LOGIT, score=score, weights=weights).fit(x, y, d, folds=reps)
        np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
        np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
        ref.sensitivity_analysis(**SCENARIO)
        ours.sensitivity_analysis(**SCENARIO)
        for q in ("theta", "ci"):
            for b in ("lower", "upper"):
                np.testing.assert_allclose(ours.sensitivity_params_[q][b], ref.sensitivity_params[q][b], rtol=TOL)
    with pytest.raises(ValueError, match="0/1"):
        dml_rs.IRM(LinearRegression(), LOGIT, score="ATTE", weights=np.exp(x[:, 1])).fit(x, y, d)


def test_apo_weights_and_capo_gapo_match_doubleml():
    r = make_irm_data_discrete_treatments(n_obs=800, n_levels=3, linear=True, random_state=23)
    x, y, d = r["x"], r["y"], r["d"]
    reps = folds(len(y))
    w = np.exp(0.2 * x[:, 0])
    ref = doubleml(dml.DoubleMLAPO, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_g=LinearRegression(), ml_m=LOGIT,
                   treatment_level=1, weights=w)
    ours = dml_rs.APO(LinearRegression(), LOGIT, treatment_level=1, weights=w).fit(x, y, d, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)

    plain = doubleml(dml.DoubleMLAPO, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_g=LinearRegression(),
                     ml_m=LOGIT, treatment_level=1)
    ours_plain = dml_rs.APO(LinearRegression(), LOGIT, treatment_level=1).fit(x, y, d, folds=reps)
    basis = pd.DataFrame({"const": 1.0, "x1": x[:, 0]})
    assert_same_blp(ours_plain.capo(basis), plain.capo(basis))
    groups = pd.DataFrame({"a": x[:, 1] <= 0, "b": x[:, 1] > 0})
    assert_same_blp(ours_plain.gapo(groups.astype(float)), plain.gapo(groups))


@pytest.mark.parametrize("depth, ccp_alpha", [(2, 0.01), (3, 0.01), (3, 0.0005), (1, 0.01)])
def test_policy_tree_matches_doubleml(depth, ccp_alpha):
    np.random.seed(24)
    n = 1500
    x = np.random.normal(size=(n, 4))
    d = np.random.binomial(1, 1 / (1 + np.exp(-0.5 * x[:, 0]))).astype(float)
    tau = np.where(x[:, 1] > 0.3, 1.5, -0.8) + 0.5 * x[:, 2]
    y = tau * d + x[:, 0] + np.random.normal(size=n)
    reps = folds(n)
    ref = doubleml(dml.DoubleMLIRM, dml.DoubleMLData.from_arrays(x, y, d), reps, ml_g=LinearRegression(), ml_m=LOGIT)
    ours = dml_rs.IRM(LinearRegression(), LOGIT).fit(x, y, d, folds=reps)

    features = pd.DataFrame(x[:, 1:3], columns=["x2", "x3"])
    ref_tree = ref.policy_tree(features, depth=depth, ccp_alpha=ccp_alpha, random_state=0)
    ours_tree = ours.policy_tree(features, depth=depth, ccp_alpha=ccp_alpha)
    assert ours_tree.n_leaves == ref_tree.policy_tree.get_n_leaves()
    new = pd.DataFrame(np.random.normal(size=(500, 2)), columns=["x2", "x3"])
    for f in (features, new):
        np.testing.assert_array_equal(ours_tree.predict(f)["pred_treatment"], ref_tree.predict(f)["pred_treatment"])
    # Same splits (scikit-learn computes thresholds in float32).
    t = ref_tree.policy_tree.tree_
    ref_splits = sorted((int(f), round(float(th), 4)) for f, th in zip(t.feature, t.threshold) if f >= 0)
    rules = str(ours_tree)
    for f, th in ref_splits:
        assert f"{features.columns[f]} <= {th:.4f}" in rules
    assert ours_tree.predict(x[:, 1:3]).dtype.kind == "i"


def test_policy_tree_requires_an_effect_signal():
    np.random.seed(25)
    x, y, d = make_plr_CCDDHNR2018(n_obs=200, dim_x=3, return_type="array")
    plr = dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()).fit(x, y, d)
    with pytest.raises(ValueError, match="treatment-effect signal"):
        plr.result_.policy_tree(x)

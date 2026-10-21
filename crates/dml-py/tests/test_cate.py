import doubleml as dml
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-7
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def simulate(n, seed):
    """τ(x) = 1 + x₀, logistic propensity, outcome linear in X per arm."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    e = 1 / (1 + np.exp(-(0.5 * x[:, 0] - 0.3 * x[:, 1])))
    d = (rng.uniform(size=n) < e).astype(float)
    y = x[:, 1] + 0.5 * x[:, 2] + d * (1 + x[:, 0]) + rng.normal(size=n)
    return x, y, d


@pytest.fixture(scope="module")
def doubleml_irm():
    x, y, d = simulate(800, 0)
    reps = [list(KFold(5, shuffle=True, random_state=r).split(x)) for r in range(2)]
    model = dml.DoubleMLIRM(dml.DoubleMLData.from_arrays(x, y, d), LinearRegression(), LOGIT, n_folds=5, n_rep=2,
                            draw_sample_splitting=False)
    model.set_sample_splitting(reps)
    model.fit()
    return x, y, d, reps, model


def test_irm_cate_and_gate_match_doubleml(doubleml_irm):
    x, y, d, reps, ref = doubleml_irm
    ours = dml_rs.IRM(LinearRegression(), LOGIT).fit(x, y, d, folds=reps)
    basis = pd.DataFrame({"const": 1.0, "x0": x[:, 0]})
    cate, ref_cate = ours.cate(basis), ref.cate(basis)
    np.testing.assert_allclose(cate.coef, ref_cate.coef, rtol=TOL)
    np.testing.assert_allclose(cate.se, ref_cate.se, rtol=TOL)
    np.testing.assert_allclose(cate.confint().to_numpy(), ref_cate.confint().to_numpy(), rtol=TOL)
    grid = basis.iloc[:15]
    np.testing.assert_allclose(cate.confint(grid).to_numpy(), ref_cate.confint(grid).to_numpy(), rtol=TOL)
    assert list(cate.summary.index) == ["const", "x0"]

    groups = pd.DataFrame({"low": x[:, 0] <= 0, "high": x[:, 0] > 0})
    gate, ref_gate = ours.gate(groups), ref.gate(groups)
    np.testing.assert_allclose(gate.coef, ref_gate.coef, rtol=TOL)
    np.testing.assert_allclose(gate.confint().to_numpy(), ref_gate.confint().to_numpy(), rtol=TOL)
    assert list(gate.confint().index) == ["low", "high"]


def test_dr_learner_pseudo_outcomes_are_doubleml_orthogonal_signal(doubleml_irm):
    x, y, d, reps, ref = doubleml_irm
    dr = dml_rs.DRLearner(LinearRegression(), LOGIT, LinearRegression()).fit(x, y, d, folds=reps[0])
    np.testing.assert_allclose(dr.pseudo_outcomes_, ref.psi_elements["psi_b"][:, 0, 0], rtol=TOL, atol=1e-10)


@pytest.mark.parametrize(
    "make",
    [
        lambda: dml_rs.TLearner(LinearRegression()),
        lambda: dml_rs.XLearner(LinearRegression(), LinearRegression(), LogisticRegression()),
        lambda: dml_rs.RLearner(LinearRegression(), LogisticRegression(), LinearRegression()),
        lambda: dml_rs.RLearner(dml_rs.Ols(), dml_rs.LogisticRegression(), dml_rs.Ols()),
        lambda: dml_rs.DRLearner(dml_rs.Ols(), dml_rs.LogisticRegression(), dml_rs.Ols()),
    ],
    ids=["T-sklearn", "X-sklearn", "R-sklearn-weights", "R-native", "DR-native"],
)
def test_meta_learners_recover_heterogeneous_effect(make):
    x, y, d = simulate(4000, 1)
    model = make().fit(x, y, d)
    grid = np.column_stack([np.linspace(-2, 2, 41), np.zeros(41), np.zeros(41)])
    rmse = np.sqrt(np.mean((model.effect(grid) - (1 + grid[:, 0])) ** 2))
    assert rmse < 0.15
    assert model.ate_ == pytest.approx(1.0, abs=0.15)


def test_x_learner_with_forests_and_dr_blp_inference():
    x, y, d = simulate(1500, 2)
    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=10, random_state=0)
    x_model = dml_rs.XLearner(rf, rf, RandomForestClassifier(n_estimators=50, min_samples_leaf=10, random_state=0))
    effect = x_model.fit(x, y, d).effect(x)
    assert np.corrcoef(effect, 1 + x[:, 0])[0, 1] > 0.8

    dr = dml_rs.DRLearner(dml_rs.Ols(), dml_rs.LogisticRegression(), dml_rs.Ols()).fit(x, y, d)
    blp = dr.blp(np.column_stack([np.ones(len(y)), x[:, 0]]))
    lo, _, hi = blp.confint().to_numpy()[1]
    assert lo < 1.0 < hi
    joint = blp.confint(np.column_stack([np.ones(5), np.linspace(-1, 1, 5)]), joint=True, n_boot=300)
    assert joint.shape == (5, 3)


def test_meta_learner_errors():
    x, y, d = simulate(200, 3)
    with pytest.raises(ValueError, match="binary"):
        dml_rs.TLearner(dml_rs.Ols()).fit(x, y, x[:, 0])
    with pytest.raises(AttributeError, match="not fitted"):
        dml_rs.SLearner(dml_rs.Ols()).effect(x)
    t = dml_rs.TLearner(dml_rs.Ols()).fit(x, y, d)
    with pytest.raises(ValueError, match="columns"):
        t.effect(x[:, :2])

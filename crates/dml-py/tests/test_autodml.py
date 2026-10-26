"""Automatic debiased machine learning, checked against independent NumPy / scikit-learn
computations of the same estimator."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-8


def folds(n, n_folds=4, seed=0):
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(n)))


def binary_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    d = (rng.random(n) < p).astype(float)
    y = d * (1 + 0.5 * X[:, 0]) + np.sin(X[:, 0]) + X[:, 2] + rng.normal(size=n)
    return X, y, d


def continuous_data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    d = 0.5 * X[:, 0] + rng.normal(size=n)
    y = 0.7 * d + np.sin(X[:, 0]) + X[:, 1] + rng.normal(size=n)
    return X, y, d


def dictionary(W):
    """The degree-1 polynomial dictionary in dml-rs order: [1, x, d, d x]."""
    base = np.column_stack([np.ones(len(W)), W[:, 1:]])
    return np.column_stack([base, W[:, :1] * base])


def ate_terms(W):
    w1, w0 = W.copy(), W.copy()
    w1[:, 0], w0[:, 0] = 1.0, 0.0
    return [(w1, 1.0), (w0, -1.0)]


def shift_terms(delta):
    def terms(W):
        w = W.copy()
        w[:, 0] += delta
        return [(w, 1.0), (W, -1.0)]
    return terms


def reference(X, y, d, terms, smpls, riesz):
    """θ and se of Auto-DML computed directly: OLS regression on W = (D, X) and the Riesz
    representer from ``riesz(B, MB) -> rho`` on each training fold."""
    W = np.column_stack([d, X])
    n = len(y)
    psi_b = np.empty(n)
    for train, test in smpls:
        g = LinearRegression().fit(W[train], y[train])
        tr_terms, te_terms = [(wk[train], c) for wk, c in terms(W)], [(wk[test], c) for wk, c in terms(W)]
        B = dictionary(W[train])
        MB = sum(c * dictionary(wk) for wk, c in tr_terms)
        rho = riesz(B, MB)
        alpha = dictionary(W[test]) @ rho
        m_g = sum(c * g.predict(wk) for wk, c in te_terms)
        psi_b[test] = m_g + alpha * (y[test] - g.predict(W[test]))
    theta = psi_b.mean()
    return theta, np.sqrt(np.mean((psi_b - theta) ** 2) / n)


def unpenalised(B, MB):
    G, M = B.T @ B / len(B), MB.mean(axis=0)
    return np.linalg.solve(G, M)


@pytest.mark.parametrize("kind", ["ate", "shift"])
def test_unpenalised_riesz_matches_closed_form(kind):
    X, y, d = binary_data() if kind == "ate" else continuous_data()
    smpls = folds(len(y))
    functional = dml_rs.ATE() if kind == "ate" else dml_rs.ShiftEffect(0.5)
    terms = ate_terms if kind == "ate" else shift_terms(0.5)
    model = dml_rs.AutoDML(LinearRegression(), functional, riesz=dml_rs.RieszLasso(degree=1, penalty=0.0))
    model.fit(X, y, d, folds=smpls)
    theta, se = reference(X, y, d, terms, smpls, unpenalised)
    np.testing.assert_allclose(model.coef_[0], theta, rtol=TOL)
    np.testing.assert_allclose(model.se_[0], se, rtol=TOL)


@pytest.mark.parametrize("r", [0.01, 0.1])
def test_penalised_riesz_matches_sklearn_lasso(r):
    X, y, d = binary_data(seed=3)
    smpls = folds(len(y))
    model = dml_rs.AutoDML(LinearRegression(), "ate", riesz=dml_rs.RieszLasso(degree=1, penalty=r))
    model.fit(X, y, d, folds=smpls)

    def lasso(B, MB):
        G, M = B.T @ B / len(B), MB.mean(axis=0)
        u = B @ np.linalg.solve(G, M)
        psi = np.sqrt(np.diag(G))[1:]
        fit = Lasso(alpha=r, tol=1e-15, max_iter=1_000_000).fit(B[:, 1:] / psi, u)
        return np.concatenate([[fit.intercept_], fit.coef_ / psi])

    theta, se = reference(X, y, d, ate_terms, smpls, lasso)
    np.testing.assert_allclose(model.coef_[0], theta, rtol=1e-7)
    np.testing.assert_allclose(model.se_[0], se, rtol=1e-6)


def test_custom_functional_and_dictionary_equal_builtins():
    X, y, d = binary_data(seed=1)
    smpls = folds(len(y))
    builtin = dml_rs.AutoDML(LinearRegression(), "ate", riesz=dml_rs.RieszLasso(degree=1, penalty=0.02)).fit(
        X, y, d, folds=smpls)
    custom = dml_rs.AutoDML(LinearRegression(), dml_rs.CustomFunctional(ate_terms, name="my ATE"),
                            riesz=dml_rs.RieszLasso(dictionary=dictionary, penalty=0.02)).fit(X, y, d, folds=smpls)
    np.testing.assert_allclose(custom.coef_, builtin.coef_, rtol=1e-12)
    np.testing.assert_allclose(custom.se_, builtin.se_, rtol=1e-12)
    assert "my ATE" in str(custom)

    Xc, yc, dc = continuous_data(seed=2)
    arr = dml_rs.AutoDML(dml_rs.Ols(), dml_rs.PolicyEffect(dc + 0.3)).fit(Xc, yc, dc)
    fun = dml_rs.AutoDML(dml_rs.Ols(), dml_rs.PolicyEffect(lambda W: W[:, 0] + 0.3)).fit(Xc, yc, dc)
    shift = dml_rs.AutoDML(dml_rs.Ols(), dml_rs.ShiftEffect(0.3)).fit(Xc, yc, dc)
    np.testing.assert_allclose(arr.coef_, fun.coef_, rtol=1e-12)
    np.testing.assert_allclose(arr.coef_, shift.coef_, rtol=1e-12)


def test_average_derivative_recovers_the_effect():
    X, y, d = continuous_data(n=3000, seed=4)
    model = dml_rs.AutoDML(dml_rs.Ols(), "average_derivative").fit(X, y, d)
    assert abs(model.coef_[0] - 0.7) < 4 * model.se_[0]
    per_unit = dml_rs.AutoDML(dml_rs.Ols(), dml_rs.ShiftEffect(0.2, per_unit=True)).fit(X, y, d)
    np.testing.assert_allclose(per_unit.coef_, model.coef_, rtol=1e-6)
    assert set(model.predictions_) == {"ml_g", "riesz", "functional"}


def test_ate_agrees_with_irm_and_supports_the_model_api():
    X, y, d = binary_data(n=3000, seed=5)
    auto = dml_rs.AutoDML(LinearRegression(), n_rep=2).fit(X, y, d)
    irm = dml_rs.IRM(LinearRegression(), LogisticRegression(), n_rep=2).fit(X, y, d)
    assert abs(auto.coef_[0] - irm.coef_[0]) < irm.se_[0]
    assert "Auto-DML (ATE)" in str(auto)
    auto.sensitivity_analysis(cf_y=0.03, cf_d=0.03)
    assert auto.sensitivity_params_["theta"]["lower"][0] < auto.coef_[0]
    blp = auto.gate((X[:, :1] > 0).astype(float) * [1, 0] + (X[:, :1] <= 0) * [0, 1])
    assert blp.summary.shape[0] == 2
    auto.bootstrap(n_boot=200)
    assert auto.confint(joint=True).shape == (1, 2)


def test_learner_parameters_stored_models_and_data():
    X, y, d = binary_data(seed=6)
    model = dml_rs.AutoDML(Lasso(alpha=0.1), n_folds=3)
    model.set_ml_nuisance_params("ml_g", "d", {"alpha": 0.01}).fit(X, y, d, store_models=True)
    fitted = model.models_["ml_g"]["d"][0]
    assert len(fitted) == 3 and all(m.alpha == 0.01 for m in fitted)
    data = dml_rs.DmlData.from_arrays(X, y, d)
    again = dml_rs.AutoDML(Lasso(alpha=0.01), n_folds=3).fit(data)
    np.testing.assert_allclose(again.coef_, model.coef_, rtol=1e-12)


def test_errors():
    X, y, d = continuous_data(n=200)
    with pytest.raises(Exception, match="0/1 treatment"):
        dml_rs.AutoDML(dml_rs.Ols(), "ate").fit(X, y, d)
    with pytest.raises(ValueError, match="unknown functional"):
        dml_rs.AutoDML(dml_rs.Ols(), "att").fit(X, y, d)
    bad = dml_rs.CustomFunctional(lambda W: [(W[:, :1], 1.0)])
    with pytest.raises(Exception, match="shape"):
        dml_rs.AutoDML(dml_rs.Ols(), bad).fit(X, y, d)
    with pytest.raises(NotImplementedError):
        dml_rs.AutoDML(dml_rs.Ols()).tune(X, y, d, param_grids={"ml_g": {}})

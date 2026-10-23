import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.did.datasets import make_did_SZ2020
from doubleml.irm.datasets import make_iivm_data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold

import dml_rs

TOL = 1e-7
LOGIT = LogisticRegression(solver="newton-cholesky", tol=1e-14, max_iter=1000)


def folds(n, n_rep=1):
    return [list(KFold(5, shuffle=True, random_state=r).split(np.zeros(n))) for r in range(n_rep)]


def doubleml_fit(cls, data, reps, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(data, n_folds=5, n_rep=len(reps), draw_sample_splitting=False, **kwargs)
        model.set_sample_splitting(reps)
        return model.fit()


@pytest.mark.parametrize("subgroups", [None, {"always_takers": False, "never_takers": True}])
def test_iivm_matches_doubleml(subgroups):
    np.random.seed(1)
    x, y, d, z = make_iivm_data(n_obs=600, dim_x=5, theta=0.5, alpha_x=1.0, return_type="array")
    z = np.asarray(z).ravel()
    reps = folds(len(y), n_rep=2)
    extra = {} if subgroups is None else {"subgroups": subgroups}
    ref = doubleml_fit(dml.DoubleMLIIVM, dml.DoubleMLData.from_arrays(x, y, d, z=z), reps,
                       ml_g=LinearRegression(), ml_m=LOGIT, ml_r=LOGIT, **extra)
    ours = dml_rs.IIVM(LinearRegression(), LOGIT, LOGIT, subgroups=subgroups).fit(x, y, d, z, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    # Native Rust learners solve the same problems.
    native = dml_rs.IIVM(dml_rs.Ols(), dml_rs.LogisticRegression(), dml_rs.LogisticRegression(),
                         subgroups=subgroups).fit(x, y, d, z, folds=reps)
    np.testing.assert_allclose(native.coef_, ref.coef, rtol=1e-6)


@pytest.mark.parametrize("score", ["observational", "experimental"])
@pytest.mark.parametrize("normalize", [True, False])
def test_did_panel_and_cross_section_match_doubleml(score, normalize):
    ml_m = LOGIT if score == "observational" else None
    kwargs = dict(score=score, in_sample_normalization=normalize)

    np.random.seed(2)
    x, y, d, _ = make_did_SZ2020(n_obs=600, cross_sectional_data=False, return_type="array")
    reps = folds(len(y))
    ref = doubleml_fit(dml.DoubleMLDID, dml.DoubleMLDIDData.from_arrays(x, y, d), reps,
                       ml_g=LinearRegression(), ml_m=ml_m, **kwargs)
    ours = dml_rs.DID(LinearRegression(), ml_m, **kwargs).fit(x, y, d, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)

    np.random.seed(3)
    x, y, d, t = make_did_SZ2020(n_obs=800, cross_sectional_data=True, return_type="array")
    reps = folds(len(y))
    ref = doubleml_fit(dml.DoubleMLDIDCS, dml.DoubleMLDIDData.from_arrays(x, y, d, t=t), reps,
                       ml_g=LinearRegression(), ml_m=ml_m, **kwargs)
    ours = dml_rs.DIDCS(LinearRegression(), ml_m, **kwargs).fit(x, y, d, t, folds=reps)
    np.testing.assert_allclose(ours.coef_, ref.coef, rtol=TOL)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=TOL)
    assert set(ours.predictions_) >= {"ml_g_d0_t0", "ml_g_d1_t1"}


def test_did_errors():
    np.random.seed(4)
    x, y, d, t = make_did_SZ2020(n_obs=200, cross_sectional_data=True, return_type="array")
    with pytest.raises(ValueError, match="ml_m"):
        dml_rs.DID(dml_rs.Ols()).fit(x, y, d)  # observational needs a propensity learner
    with pytest.raises(ValueError, match="score"):
        dml_rs.DID(dml_rs.Ols(), score="quasi").fit(x, y, d)
    with pytest.raises(ValueError, match="time indicator"):
        dml_rs.DIDCS(dml_rs.Ols(), score="experimental").fit(x, y, d, t * 2)
    with pytest.raises(ValueError, match="binary"):
        dml_rs.IIVM(dml_rs.Ols(), dml_rs.LogisticRegression(), dml_rs.LogisticRegression()).fit(x, y, d, x[:, 0])

"""Names renamed or moved in 0.32.0 keep working with a DeprecationWarning until 2.0."""

import numpy as np
import pytest

import dml_rs


LEARNERS = {"ml_g": dml_rs.Ols(), "ml_m": dml_rs.LogisticRegression()}


@pytest.mark.parametrize(
    "cls, new, required",
    [
        (dml_rs.ConformalITE, "clipping_threshold", {}),
        (dml_rs.ContinuousDID, "clipping_threshold", {}),
        (dml_rs.SurrogateIndex, "clipping_threshold", LEARNERS),
        (dml_rs.DoseResponse, "density_floor", {"ml_g": dml_rs.Ols(), "ml_m": dml_rs.Ols()}),
    ],
)
def test_trimming_threshold_is_a_deprecated_alias(cls, new, required):
    with pytest.warns(DeprecationWarning, match=new):
        est = cls(trimming_threshold=0.05, **required)
    assert getattr(est, new) == 0.05
    assert not hasattr(est, "trimming_threshold")
    with pytest.raises(TypeError, match="unexpected keyword argument 'nonsense'"):
        cls(nonsense=1, **required)


def test_rdflex_ci_is_a_deprecated_alias():
    rng = np.random.default_rng(0)
    n = 600
    score = rng.uniform(-1, 1, n)
    X = rng.normal(size=(n, 2))
    d = (score >= 0).astype(float)
    y = 0.5 * d + score + X[:, 0] + rng.normal(size=n)
    est = dml_rs.RDFlex(ml_g=dml_rs.Ols()).fit(X, y, d, score)
    with pytest.warns(DeprecationWarning, match="confint_"):
        np.testing.assert_array_equal(est.ci_, est.confint_)


def test_native_layer_moved_to_dml_rs_native():
    assert "fit_plr" not in dml_rs.__all__
    assert "DmlResult" not in dml_rs.__all__
    with pytest.warns(DeprecationWarning, match="dml_rs.native.fit_plr"):
        assert dml_rs.fit_plr is dml_rs.native.fit_plr
    with pytest.warns(DeprecationWarning, match="dml_rs.native.Bootstrap"):
        assert dml_rs.Bootstrap is dml_rs.native.Bootstrap
    with pytest.raises(AttributeError, match="no attribute 'nonsense'"):
        dml_rs.nonsense  # noqa: B018

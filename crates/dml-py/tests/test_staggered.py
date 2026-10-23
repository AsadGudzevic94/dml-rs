import warnings

import doubleml as dml
import numpy as np
import pytest
from doubleml.did.datasets import make_did_CS2021
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs


@pytest.fixture(scope="module")
def panel():
    np.random.seed(5)
    df = make_did_CS2021(n_obs=800, dgp_type=1, include_never_treated=True, time_type="float")
    x_cols = [c for c in df.columns if c.startswith("Z")]
    return df, x_cols


def test_staggered_did_agrees_with_doubleml(panel):
    df, x_cols = panel
    ours = dml_rs.StaggeredDID(LinearRegression(), LogisticRegression(), control_group="not_yet_treated")
    ours.fit_frame(df, "y", "id", "t", "d", x_cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=x_cols)
        ref = dml.did.DoubleMLDIDMulti(data, LinearRegression(), LogisticRegression(),
                                       control_group="not_yet_treated").fit()

    assert [tuple(map(float, gt)) for gt in ref.gt_combinations] == ours.group_times_
    assert list(ours.summary.index) == [f"ATT({g:g},{a:g},{b:g})" for g, a, b in ours.group_times_]
    # Different (random) folds, so agreement is up to cross-fitting noise, well within one se.
    assert np.all(np.abs(ours.coef_ - ref.coef) < 0.5 * ref.se)
    np.testing.assert_allclose(ours.se_, ref.se, rtol=0.15)

    for how in ["group", "time", "eventstudy"]:
        agg = ours.aggregate(how)
        ref_agg = ref.aggregate(how)
        assert [float(v) for v in agg.names] == [float(v) for v in ref_agg.aggregation_names]
        np.testing.assert_allclose(agg.weights, ref_agg.aggregation_weights, atol=1e-12)
        # Aggregates are exact weighted averages of the group-time effects.
        np.testing.assert_allclose(agg.coef, agg.weights @ ours.coef_, rtol=1e-12)
        assert agg.overall_coef == pytest.approx(float(agg.overall_weights @ agg.coef), rel=1e-12)
        assert abs(agg.overall_coef - ref_agg.overall_aggregated_framework.thetas[0]) < \
            0.5 * ref_agg.overall_aggregated_framework.ses[0]
        assert agg.summary.shape == (len(agg.names), 6)
        assert agg.overall_summary.shape == (1, 6)


def test_joint_bands_and_native_learners(panel):
    df, x_cols = panel
    model = dml_rs.StaggeredDID(dml_rs.Ols(), dml_rs.LogisticRegression(), n_rep=2)
    model.fit(df[x_cols].to_numpy(), df["y"].to_numpy(), df["id"].to_numpy(), df["t"].to_numpy(), df["d"].to_numpy())
    assert model.all_coef_.shape == (len(model.group_times_), 2)
    model.bootstrap(n_boot=300)
    joint, point = model.confint(joint=True), model.confint()
    assert np.all(joint[:, 1] - joint[:, 0] > point[:, 1] - point[:, 0])
    es = model.aggregate("eventstudy")
    es_joint = es.confint(joint=True, n_boot=300)
    assert es_joint.shape == (len(es.names), 2)


def test_staggered_errors(panel):
    df, x_cols = panel
    args = (df[x_cols].to_numpy(), df["y"].to_numpy(), df["id"].to_numpy(), df["t"].to_numpy(), df["d"].to_numpy())
    with pytest.raises(ValueError, match="ml_m"):
        dml_rs.StaggeredDID(dml_rs.Ols()).fit(*args)
    with pytest.raises(ValueError, match="control_group"):
        dml_rs.StaggeredDID(dml_rs.Ols(), score="experimental", control_group="everyone").fit(*args)
    with pytest.raises(ValueError, match="aggregation"):
        dml_rs.StaggeredDID(dml_rs.Ols(), score="experimental").fit(*args).aggregate("weekly")
    with pytest.raises(AttributeError, match="not fitted"):
        dml_rs.StaggeredDID(dml_rs.Ols()).aggregate()

"""Exact policy trees (parity with R's policytree, fixture from
parity/generate_policytree_fixtures.R), doubly robust scores and policy values."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import dml_rs

FIXTURE = Path(__file__).parents[2] / "dml-core" / "tests" / "fixtures" / "policytree_r.json"
CASES = {k: v for k, v in json.loads(FIXTURE.read_text()).items() if not k.startswith("_")}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_policytree(name):
    case = CASES[name]
    tree = dml_rs.OptimalPolicyTree(case["depth"], case["split_step"], case["min_node_size"])
    tree.fit(np.asarray(case["x"]), np.asarray(case["gamma"]))
    np.testing.assert_array_equal(tree.predict(np.asarray(case["x"])), case["predictions"])
    assert len(tree.nodes_) == len(case["nodes"])
    if name != "depth3_small":  # see tests/policytree.rs: an exact tie between two thresholds
        for ours, theirs in zip(tree.nodes_, case["nodes"]):
            for key, value in theirs.items():
                assert ours[key] == value, (name, key)


def simulate(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    d = (rng.random(n) < 1 / (1 + np.exp(-0.5 * x[:, 1]))).astype(float)
    tau = np.where(x[:, 0] > 0, 1.0, -0.5)
    y = x[:, 1] + d * tau + rng.normal(size=n)
    return x, y, d


def test_scores_tree_and_value_end_to_end():
    x, y, d = simulate(4000, 0)
    learn, evaluate = np.arange(2000), np.arange(2000, 4000)
    rf = dict(n_estimators=100, min_samples_leaf=20, random_state=0)
    def scores(idx):
        apos = dml_rs.APOS(RandomForestRegressor(**rf), RandomForestClassifier(**rf),
                           treatment_levels=[0, 1]).fit(x[idx], y[idx], d[idx])
        gamma = dml_rs.double_robust_scores(apos)
        np.testing.assert_allclose(gamma.mean(axis=0), apos.coef_, rtol=0, atol=1e-10)
        return gamma
    tree = dml_rs.OptimalPolicyTree(depth=1).fit(x[learn], scores(learn))
    assert tree.nodes_[0]["split_variable"] == 0 and abs(tree.nodes_[0]["split_value"]) < 0.3
    gamma = scores(evaluate)
    pi = tree.predict(x[evaluate])
    fw = dml_rs.policy_value(gamma, tree=pi, treat_all=np.ones(2000, int))
    rows = np.arange(2000)
    np.testing.assert_allclose(fw.coef_, [gamma[rows, pi].mean(), gamma[:, 1].mean()], rtol=0, atol=1e-12)
    np.testing.assert_allclose(fw.se_[0], gamma[rows, pi].std() / np.sqrt(2000), rtol=1e-10)
    gain = fw.linear_combination(np.array([[1.0, -1.0]]))
    diff = gamma[rows, pi] - gamma[:, 1]
    np.testing.assert_allclose(gain.coef_, [diff.mean()], rtol=0, atol=1e-12)
    np.testing.assert_allclose(gain.se_, [diff.std() / np.sqrt(2000)], rtol=1e-10)
    # Treating only x0 > 0 gains about 0.25 (half the units lose 0.5 under treat-all).
    assert gain.coef_[0] > 0.1 and gain.confint()[0, 0] > 0


def test_pickle_and_errors():
    case = CASES["depth2"]
    tree = dml_rs.OptimalPolicyTree(2).fit(np.asarray(case["x"]), np.asarray(case["gamma"]))
    again = pickle.loads(pickle.dumps(tree))
    np.testing.assert_array_equal(again.predict(np.asarray(case["x"])), case["predictions"])
    with pytest.raises(ValueError, match="same number of rows|rows"):
        dml_rs.OptimalPolicyTree(1).fit(np.zeros((5, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="columns"):
        tree.predict(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="action"):
        dml_rs.policy_value(np.zeros((3, 2)), bad=[0, 2, 1])
    with pytest.raises(TypeError, match="APOS"):
        dml_rs.double_robust_scores(dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols()))

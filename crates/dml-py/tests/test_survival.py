import pickle

import matplotlib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import dml_rs

matplotlib.use("Agg")
LOGIT = LogisticRegression(C=1e4, solver="newton-cholesky", tol=1e-12, max_iter=1000)


def simulate(n, seed):
    """Treatment and censoring depend on x; treatment halves the event rate."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 2))
    d = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x[:, 0]))).astype(float)
    t = rng.exponential(1 / (np.exp(0.8 * x[:, 0]) * np.where(d == 1, 0.5, 1.0)))
    c = rng.exponential(1 / (0.3 * np.exp(-0.8 * x[:, 0])))
    return x, np.minimum(t, c), (t <= c).astype(float), d


def truth(hazard_ratio, t):
    grid = (np.arange(200000) + 0.5) / 100000 - 1
    return np.exp(-np.exp(0.8 * grid) * hazard_ratio * t).mean()


def logistic_model(**kwargs):
    return dml_rs.CausalSurvival(LOGIT, LOGIT, LOGIT, time_encoding="dummies", grid=12, **kwargs)


def hazard(rows, target, new_rows):
    if len(set(target)) == 1:
        return np.full(len(new_rows), target[0])
    return LOGIT.fit(rows, target).predict_proba(new_rows)[:, 1]


def reference_scores(x, time, event, d, grid, train, test, arm, m):
    """The score from scratch on one split, with logistic hazards fitted by scikit-learn."""
    k = len(grid)
    at_risk = np.where((event == 1) & (time <= grid[-1]), np.searchsorted(grid, time, side="left") + 1,
                       np.searchsorted(grid, time, side="right"))
    event_slot = np.where((event == 1) & (time <= grid[-1]), at_risk, -1)
    censored_after = np.where((event == 0) & (at_risk < k), at_risk, -1)

    def rows(units, slots, first):
        out = []
        for i, slot in zip(units, slots):
            indicator = np.zeros(k - 1)
            if slot > first:
                indicator[slot - first - 1] = 1.0
            out.append(np.concatenate((indicator, x[i])))
        return np.array(out)

    arm_units = train[d[train] == arm]
    units, slots, target = [], [], []
    for i in arm_units:
        for slot in range(1, at_risk[i] + 1):
            units.append(i), slots.append(slot), target.append(float(event_slot[i] == slot))
    every = np.repeat(test, k)
    h = hazard(rows(units, slots, 1), target, rows(every, np.tile(np.arange(1, k + 1), len(test)), 1))
    h = h.reshape(len(test), k)

    units, slots, target = [], [], []
    for i in arm_units:
        for slot in range(0, min(k, at_risk[i] + 1)):
            if event_slot[i] == slot:
                break
            units.append(i), slots.append(slot), target.append(float(censored_after[i] == slot))
    c = hazard(rows(units, slots, 0), target, rows(every, np.tile(np.arange(k), len(test)), 0)).reshape(len(test), k)

    survival = np.cumprod(1 - h, axis=1)
    uncensored = np.maximum(np.cumprod(1 - c, axis=1), 0.01)
    phi = np.zeros((len(test), k))
    for row, i in enumerate(test):
        weight = (d[i] == arm) / (m[i] if arm == 1 else 1 - m[i])
        for last in range(k):
            total = 0.0
            for j in range(last + 1):
                if j + 1 <= at_risk[i]:
                    dn = float(event_slot[i] == j + 1)
                    total += survival[row, last] / survival[row, j] * (dn - h[row, j]) / uncensored[row, j]
            phi[row, last] = survival[row, last] - weight * total
    return phi


def test_score_matches_a_reference_from_scratch():
    x, time, event, d = simulate(800, 1)
    grid = np.array([0.2, 0.5, 1.0, 1.5])
    n = len(time)
    a, b = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    folds = [(np.where(a)[0], np.where(b)[0]), (np.where(b)[0], np.where(a)[0])]
    model = dml_rs.CausalSurvival(LOGIT, LOGIT, LOGIT, time_encoding="dummies", grid=list(grid), horizons=[1.5])
    fit = model.fit(x, time, event, d, folds=folds)
    m = fit.propensity_[:, 0]
    expected = {}
    for arm, treated in ((1.0, True), (0.0, False)):
        phi = np.zeros((n, len(grid)))
        for train, test in folds:
            phi[test] = reference_scores(x, time, event, d, grid, train, test, arm, m)
        expected[treated] = phi.mean(axis=0)
        np.testing.assert_allclose(fit.curve(treated).coef_, expected[treated], rtol=1e-6)
    np.testing.assert_allclose(fit.coef_[0], expected[True][-1] - expected[False][-1], rtol=1e-6)


def test_restricted_mean_is_the_integral_of_the_curve():
    x, time, event, d = simulate(500, 2)
    fit = logistic_model(horizons=[1.0], estimand="rmst").fit(x, time, event, d)
    grid = fit.grid_
    for treated in (True, False):
        curve = fit.curve(treated).coef_
        steps = np.diff(np.concatenate(([0.0], grid)))
        integral = steps[0] + (curve[:-1] * steps[1:]).sum()
        assert fit.potential(treated).coef_[0] == pytest.approx(integral, rel=1e-12)
    assert fit.coef_[0] == pytest.approx(fit.potential(True).coef_[0] - fit.potential(False).coef_[0], rel=1e-12)


def test_covers_the_truth_with_a_flexible_learner():
    expected = truth(0.5, 1.0) - truth(1.0, 1.0)
    hits, estimates = 0, []
    for seed in range(12):
        x, time, event, d = simulate(1200, seed)
        forest = dml_rs.RandomForestClassifier(n_estimators=60, min_samples_leaf=20, random_state=seed)
        fit = dml_rs.CausalSurvival(forest, forest, LOGIT, horizons=[1.0], grid=15, random_state=seed).fit(
            x, time, event, d)
        hits += abs(fit.coef_[0] - expected) <= 1.96 * fit.se_[0]
        estimates.append(fit.coef_[0])
    assert abs(np.mean(estimates) - expected) < 0.03
    assert hits >= 9


def test_doubly_robust_when_the_censoring_model_is_wrong():
    x, time, event, d = simulate(4000, 3)
    wrong = dml_rs.CausalSurvival(LOGIT, RandomForestClassifier(max_depth=1, n_estimators=1, random_state=0), LOGIT,
                                  time_encoding="dummies", grid=12, horizons=[1.0]).fit(x, time, event, d)
    right = logistic_model(horizons=[1.0]).fit(x, time, event, d)
    assert wrong.coef_[0] == pytest.approx(right.coef_[0], abs=0.03)


def test_fitted_api():
    x, time, event, d = simulate(400, 4)
    fit = logistic_model(horizons=[0.5, 1.0], n_rep=2).fit(x, time, event, d)
    assert list(fit.summary.index) == ["survival(0.5)", "survival(1)"]
    assert fit.confint().shape == (2, 2)
    assert fit.propensity_.shape == (400, 2)
    assert fit.potential(False).coef_.shape == (2,)
    fit.bootstrap(n_boot=100)
    assert fit.confint(joint=True).shape == (2, 2)
    ax = fit.plot()
    assert len(ax.get_lines()) == 2
    copied = pickle.loads(pickle.dumps(fit))
    np.testing.assert_array_equal(copied.coef_, fit.coef_)
    np.testing.assert_array_equal(copied.curve(True).coef_, fit.curve(True).coef_)
    # ml_censor defaults to ml_event, and every grid form works.
    for grid in (10, "event_times", [0.5, 1.0, 2.0]):
        small = dml_rs.CausalSurvival(LOGIT, ml_m=LOGIT, grid=grid, horizons=[1.0], n_folds=2)
        assert small.fit(x, time, event, d).coef_.shape == (1,)


def test_errors():
    x, time, event, d = simulate(100, 5)
    with pytest.raises(ValueError, match="ml_m"):
        dml_rs.CausalSurvival(LOGIT).fit(x, time, event, d)
    with pytest.raises(ValueError, match="last event time"):
        logistic_model(horizons=[1e6]).fit(x, time, event, d)
    with pytest.raises(ValueError, match="estimand"):
        logistic_model(estimand="hazard").fit(x, time, event, d)
    with pytest.raises(ValueError, match="grid"):
        dml_rs.CausalSurvival(LOGIT, LOGIT, LOGIT, grid="weekly").fit(x, time, event, d)
    with pytest.raises(ValueError, match="time_encoding"):
        dml_rs.CausalSurvival(LOGIT, LOGIT, LOGIT, time_encoding="spline").fit(x, time, event, d)
    with pytest.raises(ValueError, match="binary"):
        logistic_model().fit(x, time, event, d + 1)
    with pytest.raises(ValueError, match="event"):
        logistic_model().fit(x, time, np.zeros(100), d)
    with pytest.raises(ValueError, match="negative"):
        logistic_model().fit(x, -time, event, d)


# -- competing events ------------------------------------------------------------------------

def simulate_competing(n, seed):
    """Cause 1 (halved by treatment) competes with cause 2; treatment and censoring depend on x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 2))
    d = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x[:, 0]))).astype(float)
    l1 = 0.6 * np.exp(0.8 * x[:, 0]) * np.where(d == 1, 0.5, 1.0)
    l2 = 0.4 * np.exp(-0.5 * x[:, 0])
    t = rng.exponential(1 / (l1 + l2))
    cause = np.where(rng.random(n) < l1 / (l1 + l2), 1.0, 2.0)
    c = rng.exponential(1 / (0.3 * np.exp(-0.8 * x[:, 0])))
    return x, np.minimum(t, c), np.where(t <= c, cause, 0.0), d


def reference_incidence(x, time, event, d, grid, train, test, arm, m, cause):
    """Cumulative incidence score from scratch in its direct form (with S_j in the denominator),
    from logistic hazards of any event and of the cause fitted by scikit-learn."""
    k = len(grid)
    any_event = event > 0
    at_risk = np.where(any_event & (time <= grid[-1]), np.searchsorted(grid, time, side="left") + 1,
                       np.searchsorted(grid, time, side="right"))
    event_slot = np.where(any_event & (time <= grid[-1]), at_risk, -1)
    cause_slot = np.where(event == cause, event_slot, -1)
    censored_after = np.where(~any_event & (at_risk < k), at_risk, -1)

    def rows(units, slots, first):
        out = []
        for i, slot in zip(units, slots):
            indicator = np.zeros(k - 1)
            if slot > first:
                indicator[slot - first - 1] = 1.0
            out.append(np.concatenate((indicator, x[i])))
        return np.array(out)

    arm_units = train[d[train] == arm]
    every = np.repeat(test, k)
    units, slots = [], []
    for i in arm_units:
        for slot in range(1, at_risk[i] + 1):
            units.append(i), slots.append(slot)
    new = rows(every, np.tile(np.arange(1, k + 1), len(test)), 1)
    fitted = rows(units, slots, 1)
    h = hazard(fitted, [float(event_slot[i] == s) for i, s in zip(units, slots)], new).reshape(len(test), k)
    hc = hazard(fitted, [float(cause_slot[i] == s) for i, s in zip(units, slots)], new).reshape(len(test), k)
    hc = np.minimum(hc, h)
    units, slots, target = [], [], []
    for i in arm_units:
        for slot in range(0, min(k, at_risk[i] + 1)):
            if event_slot[i] == slot:
                break
            units.append(i), slots.append(slot), target.append(float(censored_after[i] == slot))
    c = hazard(rows(units, slots, 0), target, rows(every, np.tile(np.arange(k), len(test)), 0)).reshape(len(test), k)

    survival = np.cumprod(1 - h, axis=1)
    before = np.hstack((np.ones((len(test), 1)), survival[:, :-1]))
    incidence = np.cumsum(before * hc, axis=1)
    uncensored = np.maximum(np.cumprod(1 - c, axis=1), 0.01)
    phi = np.zeros((len(test), k))
    for row, i in enumerate(test):
        weight = (d[i] == arm) / (m[i] if arm == 1 else 1 - m[i])
        for last in range(k):
            total = 0.0
            for j in range(last + 1):
                if j + 1 <= at_risk[i]:
                    dn, dnc = float(event_slot[i] == j + 1), float(cause_slot[i] == j + 1)
                    rest = (incidence[row, last] - incidence[row, j]) / survival[row, j]
                    total += ((dnc - hc[row, j]) - rest * (dn - h[row, j])) / uncensored[row, j]
            phi[row, last] = incidence[row, last] + weight * total
    return phi


def test_cumulative_incidence_matches_a_reference_from_scratch():
    x, time, event, d = simulate_competing(800, 11)
    grid = np.array([0.2, 0.5, 1.0, 1.5])
    n = len(time)
    a, b = np.arange(n) % 2 == 0, np.arange(n) % 2 == 1
    folds = [(np.where(a)[0], np.where(b)[0]), (np.where(b)[0], np.where(a)[0])]
    for cause in (1, 2):
        fit = dml_rs.CausalSurvival(LOGIT, LOGIT, LOGIT, time_encoding="dummies", grid=list(grid), horizons=[1.5],
                                    estimand="cumulative_incidence", cause=cause).fit(x, time, event, d, folds=folds)
        m = fit.propensity_[:, 0]
        expected = {}
        for arm, treated in ((1.0, True), (0.0, False)):
            phi = np.zeros((n, len(grid)))
            for train, test in folds:
                phi[test] = reference_incidence(x, time, event, d, grid, train, test, arm, m, cause)
            expected[treated] = phi.mean(axis=0)
            np.testing.assert_allclose(fit.incidence_curve(treated).coef_, expected[treated], rtol=1e-6)
        np.testing.assert_allclose(fit.coef_[0], expected[True][-1] - expected[False][-1], rtol=1e-6)
    assert list(fit.summary.index) == ["incidence[2](1.5)"]


def test_competing_events_api_and_errors():
    x, time, event, d = simulate_competing(600, 12)
    fit = logistic_model(horizons=[1.0], estimand="cumulative_incidence", cause=1).fit(x, time, event, d)
    # Treatment halves the hazard of cause 1, so fewer treated units have it by t = 1.
    assert fit.coef_[0] < 0
    assert fit.incidence_curve(True).coef_.shape == fit.grid_.shape
    ax = fit.plot()
    assert "incidence" in ax.get_ylabel()
    # Survival is freedom from either cause.
    both = logistic_model(horizons=[1.0]).fit(x, time, event, d)
    binary = logistic_model(horizons=[1.0]).fit(x, time, (event > 0).astype(float), d)
    np.testing.assert_allclose(both.coef_, binary.coef_, rtol=1e-12)
    with pytest.raises(ValueError, match="cumulative_incidence"):
        both.incidence_curve(True)
    with pytest.raises(ValueError, match="cause"):
        logistic_model(estimand="cumulative_incidence", cause=3).fit(x, time, event, d)
    with pytest.raises(ValueError, match="event"):
        logistic_model().fit(x, time, event + 0.5, d)
    copied = pickle.loads(pickle.dumps(fit))
    np.testing.assert_array_equal(copied.incidence_curve(False).coef_, fit.incidence_curve(False).coef_)


def test_cluster_robust_standard_errors():
    x, time, event, d = simulate(1200, 13)
    folds = [(np.where(np.arange(1200) % 5 != k)[0], np.where(np.arange(1200) % 5 == k)[0]) for k in range(5)]
    plain = logistic_model(horizons=[1.0]).fit(x, time, event, d, folds=folds)
    singletons = logistic_model(horizons=[1.0]).fit(x, time, event, d, folds=folds, cluster=np.arange(1200))
    np.testing.assert_allclose(singletons.se_, plain.se_, rtol=1e-12)
    assert singletons.n_clusters_ == 1200 and plain.n_clusters_ is None
    # Duplicating every unit three times in one cluster: the unclustered standard error shrinks as if
    # the sample were three times larger, the clustered one does not.
    rep = np.repeat(np.arange(400), 3)
    xs, ts, es, ds = x[rep], time[rep], event[rep], d[rep]
    naive = logistic_model(horizons=[1.0]).fit(xs, ts, es, ds)
    clustered = logistic_model(horizons=[1.0]).fit(xs, ts, es, ds, cluster=rep)
    assert clustered.se_[0] > 1.4 * naive.se_[0]
    assert clustered.n_clusters_ == 400


def test_tuned_hazard_learners():
    # Grid searches are classifiers like any other: tuned on each training fold's person-period rows.
    from sklearn.model_selection import GridSearchCV

    x, time, event, d = simulate(600, 21)
    forest = dml_rs.RandomForestClassifier(n_estimators=30, max_bins=64)
    sklearn_search = GridSearchCV(forest, {"min_samples_leaf": [5, 40]}, cv=3, scoring="neg_log_loss")
    native_search = dml_rs.GridSearchClassifier(
        [dml_rs.RandomForestClassifier(n_estimators=30, min_samples_leaf=leaf, max_bins=64) for leaf in (5, 40)],
        cv=3)
    for search in (sklearn_search, native_search):
        fit = dml_rs.CausalSurvival(search, search, LOGIT, horizons=[1.0], grid=10).fit(x, time, event, d)
        assert np.isfinite(fit.coef_[0]) and fit.se_[0] > 0


def test_restricted_mean_time_lost():
    x, time, event, d = simulate_competing(800, 31)
    fits = {}
    for estimand, cause in (("rmst", 1), ("rmtl", 1), ("rmtl", 2)):
        fits[(estimand, cause)] = logistic_model(horizons=[1.0], estimand=estimand, cause=cause).fit(
            x, time, event, d)
    lost = fits[("rmtl", 1)]
    assert list(lost.summary.index) == ["RMTL[1](1)"]
    # Time lost is the integral of the incidence curve, which starts at zero.
    for treated in (True, False):
        curve, grid = lost.incidence_curve(treated).coef_, lost.grid_
        end = int(np.searchsorted(grid, 1.0))
        assert lost.potential(treated).coef_[0] == pytest.approx((curve[:end] * np.diff(grid[:end + 1])).sum(),
                                                                 rel=1e-12)
        # With RMST and the other cause it nearly adds up to the horizon (exactly with empirical hazards).
        total = sum(f.potential(treated).coef_[0] for f in fits.values())
        assert total == pytest.approx(1.0, abs=0.01)
    assert lost.plot().get_ylabel().startswith("Cumulative incidence")

"""Anytime-valid confidence sequences (Waudby-Smith et al., 2024)."""

import matplotlib
import numpy as np
import pytest

import dml_rs

matplotlib.use("Agg")


def multiplier(t, level, opt_time):
    alpha = 1 - level
    rho2 = (-2 * np.log(alpha) + np.log(-2 * np.log(alpha) + 1)) / opt_time
    u = t * rho2
    return np.sqrt(2 * (u + 1) / u * np.log(np.sqrt(u + 1) / alpha))


def plr_data(n, seed, theta=0.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    d = x[:, 0] + rng.normal(size=n)
    y = theta * d + x[:, 0] - x[:, 1] + rng.normal(size=n)
    return x, y, d


def plr():
    return dml_rs.PLR(dml_rs.Ols(), dml_rs.Ols(), n_folds=3)


def test_sequence_is_the_closed_form_and_wider_than_the_interval():
    x, y, d = plr_data(800, 0)
    model = plr().fit(x, y, d)
    for opt_time in (None, 200, 5000):
        cs = model.confidence_sequence(0.9, opt_time)
        m = multiplier(800, 0.9, 800 if opt_time is None else opt_time)
        np.testing.assert_allclose(cs[:, 0], model.coef_ - m * model.se_, rtol=1e-12)
        np.testing.assert_allclose(cs[:, 1], model.coef_ + m * model.se_, rtol=1e-12)
        assert np.all(cs[:, 0] < model.confint(0.9)[:, 0])
        p = model.anytime_pval(0.9, opt_time)
        assert bool((cs[:, 0] > 0) | (cs[:, 1] < 0)) == bool(p[0] < 0.1)
    assert abs(multiplier(1000, 0.95, 1000) - 3.035) < 1e-3


def test_monitoring_a_study():
    x, y, d = plr_data(3000, 1, theta=0.15)
    looks = [200, 400, 800, 1600, 3000]
    cs = dml_rs.ConfidenceSequence(level=0.95, opt_time=500).run(plr(), looks, X=x, y=y, d=d)
    np.testing.assert_array_equal(cs.n_, looks)
    for i, t in enumerate(looks):
        fit = plr().fit(x[:t], y[:t], d[:t])
        np.testing.assert_allclose(cs.coef_[i], fit.coef_)
        np.testing.assert_allclose(cs.intervals_[i], fit.confidence_sequence(0.95, 500))
    assert np.all(np.diff(cs.lower_[:, 0]) >= 0) and np.all(np.diff(cs.upper_[:, 0]) <= 0)
    assert np.all(np.diff(cs.pval_[:, 0]) <= 0)
    stop = cs.first_exclusion(0.0)[0]
    assert stop is not None and stop in looks
    i = looks.index(stop)
    assert cs.lower_[i, 0] > 0 and cs.pval_[i, 0] < 0.05
    assert cs.summary.shape == (5, 6)
    cs.plot()
    # Updates by hand give the same sequence; opt_time defaults to the first look.
    by_hand = dml_rs.ConfidenceSequence()
    for t in looks[:2]:
        by_hand.update(plr().fit(x[:t], y[:t], d[:t]))
    assert by_hand.opt_time_ == 200
    with pytest.raises(ValueError, match="growing"):
        by_hand.update(plr().fit(x[:100], y[:100], d[:100]))
    with pytest.raises(ValueError, match="same estimates"):
        by_hand.update(plr().fit(x[:500], y[:500], np.column_stack([d[:500], np.sin(np.arange(500.0))])))
    with pytest.raises(ValueError, match="looks"):
        dml_rs.ConfidenceSequence().run(plr(), [400, 200], X=x, y=y, d=d)
    with pytest.raises(TypeError):
        dml_rs.ConfidenceSequence().update("model")


def test_covers_at_all_looks_where_repeated_intervals_do_not():
    looks = list(range(100, 2001, 100))
    reps, cs_miss, ci_miss = 150, 0, 0
    for rep in range(reps):
        x, y, d = plr_data(looks[-1], 100 + rep)
        cs_out = ci_out = False
        for t in looks:
            fit = plr().fit(x[:t], y[:t], d[:t])
            lo, hi = fit.confidence_sequence(0.95, 300)[0]
            cs_out |= not lo <= 0.5 <= hi
            lo, hi = fit.confint(0.95)[0]
            ci_out |= not lo <= 0.5 <= hi
        cs_miss += cs_out
        ci_miss += ci_out
    assert cs_miss / reps <= 0.08
    assert ci_miss / reps >= 0.15


def test_clustered_fits_count_clusters():
    x, y, d = plr_data(600, 3)
    cluster = np.repeat(np.arange(60), 10)
    model = plr().fit(x, y, d, cluster=cluster)
    assert model.result_.sequence_time == 60
    cs = model.confidence_sequence(0.95)
    m = multiplier(60, 0.95, 60)
    np.testing.assert_allclose(cs[:, 1] - cs[:, 0], 2 * m * model.se_, rtol=1e-12)

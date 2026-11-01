"""Lee (2009) bounds: checked against a NumPy implementation of fractional trimming (with and
without covariate cells) and in simulations with monotone selection."""

import pickle

import numpy as np
import pytest
from scipy.stats import norm

import dml_rs


def simulate(n, seed, cells=False):
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n).astype(float)
    x = rng.integers(0, 3, n)
    u = rng.random(n)
    y0 = 2 * u + 0.5 * x + 0.3 * rng.normal(size=n)
    shift = 0.1 * x if cells else 0.0
    s = ((u > 0.4 + shift) | ((d == 1) & (u > 0.2 + shift))).astype(float)
    y = np.where(s == 1, y0 + d, np.nan)
    return y, d, s, x


def trimmed(values, keep, top):
    v = np.sort(values)[::-1] if top else np.sort(values)
    mass = keep * len(v)
    w = np.clip(mass - np.arange(len(v)), 0, 1)
    return (w * v).sum() / mass


def reference(y, d, s, x):
    p1, p0 = s[d == 1].mean(), s[d == 0].mean()
    big, small = (1, 0) if p1 >= p0 else (0, 1)
    lo = hi = wsum = 0.0
    for c in np.unique(x):
        m = x == c
        pb, ps = s[m & (d == big)].mean(), s[m & (d == small)].mean()
        w = ps * m.sum()
        q = (pb - ps) / pb if pb > ps else 0.0
        ob = y[m & (d == big) & (s == 1)]
        other = y[m & (d == small) & (s == 1)].mean()
        bottom, top = trimmed(ob, 1 - q, False), trimmed(ob, 1 - q, True)
        l, h = (bottom - other, top - other) if big == 1 else (other - top, other - bottom)
        lo, hi, wsum = lo + w * l, hi + w * h, wsum + w
    return lo / wsum, hi / wsum, (abs(p1 - p0) / max(p1, p0))


@pytest.mark.parametrize("cells", [False, True])
def test_matches_reference(cells):
    y, d, s, x = simulate(3000, 0, cells)
    strata = x if cells else np.zeros(len(y), int)
    fit = dml_rs.LeeBounds(n_boot=0).fit(y, d, s, strata=x if cells else None)
    lo, hi, q = reference(y, d, s, strata)
    np.testing.assert_allclose([fit.lower_, fit.upper_, fit.trimming_], [lo, hi, q], rtol=0, atol=1e-12)
    assert fit.trimmed_treated_
    # Swapping the arms swaps and negates the bounds.
    swapped = dml_rs.LeeBounds(n_boot=0).fit(np.where(s == 1, -y, np.nan), 1 - d, s, strata=x if cells else None)
    np.testing.assert_allclose([swapped.lower_, swapped.upper_], [fit.lower_, fit.upper_], rtol=0, atol=1e-12)
    assert not swapped.trimmed_treated_


def test_covariates_tighten_and_bounds_cover():
    widths, covered, cover = [], 0, 0
    for rep in range(40):
        y, d, s, x = simulate(2000, 100 + rep, cells=True)
        plain = dml_rs.LeeBounds(n_boot=0).fit(y, d, s)
        tight = dml_rs.LeeBounds(n_boot=100, random_state=rep).fit(y, d, s, strata=x)
        widths.append((plain.upper_ - plain.lower_, tight.upper_ - tight.lower_))
        covered += tight.lower_ <= 1.0 <= tight.upper_
        cover += tight.confint_[0] <= 1.0 <= tight.confint_[1]
    widths = np.array(widths)
    assert widths[:, 1].mean() < widths[:, 0].mean()
    assert covered / 40 > 0.85 and cover / 40 > 0.9


def test_imbens_manski_and_summary():
    y, d, s, x = simulate(1500, 3)
    fit = dml_rs.LeeBounds(n_boot=200, level=0.9).fit(y, d, s)
    sigma = fit.se_.max()
    delta = fit.upper_ - fit.lower_
    c = (fit.lower_ - fit.confint_[0]) / fit.se_[0]
    assert norm.cdf(c + delta / sigma) - norm.cdf(-c) == pytest.approx(0.9, abs=1e-9)
    assert fit.upper_ + c * fit.se_[1] == pytest.approx(fit.confint_[1], abs=1e-12)
    assert list(fit.summary.index) == ["Lee bounds"]
    assert pickle.loads(pickle.dumps(fit._result)).lower == fit.lower_


def test_invalid_input():
    y, d, s, _ = simulate(200, 4)
    with pytest.raises(ValueError, match="only 0 and 1"):
        dml_rs.LeeBounds(n_boot=0).fit(y, d + 1, s)
    with pytest.raises(ValueError, match="finite"):
        dml_rs.LeeBounds(n_boot=0).fit(np.full(len(y), np.nan), d, s)
    with pytest.raises(ValueError, match="same length"):
        dml_rs.LeeBounds(n_boot=0).fit(y[:-1], d, s)

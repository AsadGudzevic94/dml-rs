"""Reference results of ppi_py (prediction-powered inference, Angelopoulos et al.) for
dml_core::ppi.

ppi_py pulls in numba, so run this with a separate environment:

    python -m venv ppienv && ppienv/Scripts/python -m pip install ppi_python==0.2.3
    ppienv/Scripts/python parity/generate_ppi_fixtures.py

ppi_py returns intervals, not standard errors; both are recovered from the 90% interval
(midpoint and half-width / z). The tuned lambda is recomputed with ppi_py's own internals,
following the steps of ppi_*_ci, and checked to give the same interval as lam=None.
"""

import json
import pathlib

import numpy as np
import ppi_py
from ppi_py import ppi as P
from ppi_py.cross_ppi import crossppi_logistic_ci, crossppi_mean_ci, crossppi_ols_ci
from scipy.stats import norm

OUT = pathlib.Path(__file__).resolve().parents[1] / "crates/dml-core/tests/fixtures/ppi_ppi_py.json"
ALPHA = 0.1
Z = norm.ppf(1 - ALPHA / 2)


def coef_se(ci):
    lo, hi = (np.atleast_1d(np.asarray(v, dtype=float)) for v in ci)
    return ((lo + hi) / 2).tolist(), ((hi - lo) / (2 * Z)).tolist()


def simulate(rng, m):
    """Covariates x (with intercept), an extra feature z and outcomes; the predictions
    use z, which the regression does not."""
    x1 = rng.normal(size=m)
    x2 = rng.binomial(1, 0.4, size=m).astype(float)
    z = rng.normal(size=m)
    X = np.column_stack([np.ones(m), x1, x2])
    y = 1.0 + 0.8 * x1 - 0.5 * x2 + 1.5 * z + rng.normal(size=m) * (1 + 0.5 * np.abs(x1))
    eta = -0.3 + 1.0 * x1 + 0.7 * x2 + 1.8 * z
    p = 1 / (1 + np.exp(-eta))
    b = rng.binomial(1, p).astype(float)
    # Imperfect predictions: shrunk, shifted, noisy.
    yhat = 0.9 + 0.7 * x1 - 0.4 * x2 + 1.2 * z + 2.0 * rng.normal(size=m)
    phat = 1 / (1 + np.exp(-(0.6 * eta + 0.2 + 1.5 * rng.normal(size=m))))
    return X, y, yhat, b, phat


def tuned_mean(Y, Yhat, Yu):
    theta = P.ppi_mean_pointestimate(Y, Yhat, Yu, lam=1)
    g = (Y - theta).reshape(-1, 1)
    gh = (Yhat - theta).reshape(-1, 1)
    gu = (Yu - theta).reshape(-1, 1)
    return float(P._calc_lam_glm(g, gh, gu, np.eye(1), coord=None, clip=True))


def tuned_glm(est, stats, X, Y, Yhat, Xu, Yu):
    theta = est(X, Y, Yhat, Xu, Yu, lam=1)
    lam = P._calc_lam_glm(*stats(theta, X, Y, Yhat, Xu, Yu), clip=True)
    theta = est(X, Y, Yhat, Xu, Yu, lam=lam)
    return float(P._calc_lam_glm(*stats(theta, X, Y, Yhat, Xu, Yu), clip=True))


def ols_stats(theta, X, Y, Yhat, Xu, Yu):
    return P._ols_get_stats(theta, X, Y, Yhat, Xu, Yu)


def logistic_stats(theta, X, Y, Yhat, Xu, Yu):
    return P._logistic_get_stats(theta, X, Y, Yhat, Xu, Yu)


def main():
    rng = np.random.default_rng(20261104)
    n, N, K = 200, 2000, 5
    X, y, yhat, b, phat = simulate(rng, n)
    Xu, _, yhat_u, _, phat_u = simulate(rng, N)
    # Cross-prediction: K fold models' predictions on the unlabelled sample.
    yhat_u_k = yhat_u[:, None] + 0.1 * rng.normal(size=(N, K))
    phat_u_k = np.clip(phat_u[:, None] + 0.02 * rng.normal(size=(N, K)), 0.001, 0.999)

    cases = []

    def add(name, estimand, lam, ci, lam_used, outcome="y", unlabelled="yhat_u"):
        coef, se = coef_se(ci)
        cases.append(dict(name=name, estimand=estimand, lam=lam, lam_used=lam_used, coef=coef, se=se,
                          outcome=outcome, unlabelled=unlabelled))

    # Mean.
    lam = tuned_mean(y, yhat, yhat_u)
    ci = P.ppi_mean_ci(y, yhat, yhat_u, alpha=ALPHA)
    assert np.allclose(ci, P.ppi_mean_ci(y, yhat, yhat_u, alpha=ALPHA, lam=lam), rtol=0, atol=1e-14)
    add("mean_tuned", "mean", None, ci, lam)
    for fixed in (1.0, 0.35, 0.0):
        add(f"mean_lam{fixed}", "mean", fixed, P.ppi_mean_ci(y, yhat, yhat_u, alpha=ALPHA, lam=fixed), fixed)

    # OLS.
    lam = tuned_glm(P.ppi_ols_pointestimate, ols_stats, X, y, yhat, Xu, yhat_u)
    ci = P.ppi_ols_ci(X, y, yhat, Xu, yhat_u, alpha=ALPHA)
    assert np.allclose(ci, P.ppi_ols_ci(X, y, yhat, Xu, yhat_u, alpha=ALPHA, lam=lam), rtol=0, atol=1e-13)
    add("ols_tuned", "linear", None, ci, lam)
    for fixed in (1.0, 0.5, 0.0):
        add(f"ols_lam{fixed}", "linear", fixed, P.ppi_ols_ci(X, y, yhat, Xu, yhat_u, alpha=ALPHA, lam=fixed), fixed)

    # Logistic regression.
    lam = tuned_glm(P.ppi_logistic_pointestimate, logistic_stats, X, b, phat, Xu, phat_u)
    ci = P.ppi_logistic_ci(X, b, phat, Xu, phat_u, alpha=ALPHA)
    add("logistic_tuned", "logistic", None, ci, lam, outcome="b", unlabelled="phat_u")
    for fixed in (1.0, 0.0):
        add(f"logistic_lam{fixed}", "logistic", fixed,
            P.ppi_logistic_ci(X, b, phat, Xu, phat_u, alpha=ALPHA, lam=fixed), fixed,
            outcome="b", unlabelled="phat_u")

    # Cross-prediction-powered inference: lambda = 1 on the fold-averaged predictions.
    add("crossppi_mean", "mean", 1.0, crossppi_mean_ci(y, yhat, yhat_u_k, alpha=ALPHA), 1.0,
        unlabelled="yhat_u_k")
    add("crossppi_ols", "linear", 1.0, crossppi_ols_ci(X, y, yhat, Xu, yhat_u_k, alpha=ALPHA), 1.0,
        unlabelled="yhat_u_k")
    add("crossppi_logistic", "logistic", 1.0,
        crossppi_logistic_ci(X, b, phat, Xu, phat_u_k, alpha=ALPHA, optimizer_options={"ftol": 1e-15}),
        1.0, outcome="b", unlabelled="phat_u_k")

    data = dict(X=X.tolist(), Xu=Xu.tolist(), y=y.tolist(), yhat=yhat.tolist(), yhat_u=yhat_u.tolist(),
                b=b.tolist(), phat=phat.tolist(), phat_u=phat_u.tolist(),
                yhat_u_k=yhat_u_k.mean(axis=1).tolist(), phat_u_k=phat_u_k.mean(axis=1).tolist())
    OUT.write_text(json.dumps(dict(ppi_py=ppi_py.__version__ if hasattr(ppi_py, "__version__") else "0.2.3",
                                   alpha=ALPHA, data=data, cases=cases)))
    for c in cases:
        print(f"{c['name']:<20} lam={c['lam_used']:.6f} coef={np.round(c['coef'], 5)} se={np.round(c['se'], 5)}")


if __name__ == "__main__":
    main()

"""Monte Carlo replication of Colangelo and Lee (2023, arXiv 2004.03036v8), Table 1, Lasso columns.

Design: X ~ N(0, Σ) with 100 covariates (Σ tridiagonal, 0.5 off the diagonal), θ_j = 1/j²,
T = Φ(3X'θ) + 0.75ν − 0.5 and Y = 1.2T + 1.2X'θ + T² + TX₁ + ε√(0.5 + Φ(X₁)). Targets at t = 0:
β(0) = E[Y(0)] = 0 and θ(0) = ∂E[Y(t)]/∂t = 1.2. Five-fold cross-fitting, Epanechnikov kernel,
MultiGPS density with h₁ = h, h = c·σ_T·n^(−1/5).

    python parity/dose_response_mc.py --reps 500 --n 1000
"""

import argparse
import json

import numpy as np
from scipy.stats import norm

import dml_rs


def simulate(n, rng, p=100):
    cov = np.eye(p) + 0.5 * (np.eye(p, k=1) + np.eye(p, k=-1))
    X = rng.multivariate_normal(np.zeros(p), cov, size=n, method="cholesky")
    xt = X @ (1.0 / np.arange(1, p + 1) ** 2)
    t = norm.cdf(3 * xt) + 0.75 * rng.normal(size=n) - 0.5
    y = 1.2 * t + 1.2 * xt + t**2 + t * X[:, 0] + rng.normal(size=n) * np.sqrt(0.5 + norm.cdf(X[:, 0]))
    return X, y, t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5])
    args = parser.parse_args()
    out = {c: {"beta": [], "beta_se": [], "theta": [], "theta_se": []} for c in args.scales}
    for rep in range(args.reps):
        X, y, t = simulate(args.n, np.random.default_rng(rep))
        for c in args.scales:
            model = dml_rs.DoseResponse(
                dml_rs.LassoCV(), dml_rs.LassoCV(), grid=[0.0], bandwidth_scale=c, partial_effects=True,
                random_state=rep,
            ).fit(X, y, t)
            pe = model.partial_effect()
            out[c]["beta"].append(model.coef_[0])
            out[c]["beta_se"].append(model.se_[0])
            out[c]["theta"].append(pe.coef_[0])
            out[c]["theta_se"].append(pe.se_[0])
    print(f"n = {args.n}, {args.reps} replications")
    print(f"{'target':8} {'c':>5} {'bias':>8} {'RMSE':>8} {'cov.':>6}")
    rows = []
    for target, truth in [("beta", 0.0), ("theta", 1.2)]:
        for c in args.scales:
            est = np.array(out[c][target])
            se = np.array(out[c][target + "_se"])
            bias = est.mean() - truth
            rmse = np.sqrt(np.mean((est - truth) ** 2))
            cover = np.mean(np.abs(est - truth) <= 1.959964 * se)
            rows.append({"target": target, "c": c, "bias": bias, "rmse": rmse, "coverage": cover})
            print(f"{target:8} {c:5.2f} {bias:8.3f} {rmse:8.3f} {cover:6.3f}")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()

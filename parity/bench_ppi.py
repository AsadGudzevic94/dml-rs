"""Timing of dml_rs.PPI against ppi_py (mean, OLS and logistic regression, PPI++ with the
tuned lambda) on the same data.

ppi_py needs numba, so it runs in its own environment (see generate_ppi_fixtures.py):

    python parity/bench_ppi.py --ppi-python ppienv/Scripts/python
"""

import argparse
import json
import pathlib
import subprocess
import tempfile
import time

import numpy as np

import dml_rs

PPI_PY = r"""
import json, sys, time
import numpy as np
from ppi_py import ppi_logistic_ci, ppi_mean_ci, ppi_ols_ci
d = np.load(sys.argv[1])
X, Xu, y, yhat, yhat_u, b, p, pu = (d[k] for k in ["X", "Xu", "y", "yhat", "yhat_u", "b", "p", "pu"])
calls = {
    "mean": lambda: ppi_mean_ci(y, yhat, yhat_u, alpha=0.05),
    "ols": lambda: ppi_ols_ci(X, y, yhat, Xu, yhat_u, alpha=0.05),
    "logistic": lambda: ppi_logistic_ci(X, b, p, Xu, pu, alpha=0.05),
}
calls["ols"]()  # numba compiles its loops on the first call
calls["logistic"]()
out = {}
for name, f in calls.items():
    times = []
    for _ in range(3):
        t = time.perf_counter(); f(); times.append(time.perf_counter() - t)
    out[name] = min(times)
print(json.dumps(out))
"""


def simulate(rng, m, d):
    x = np.column_stack([np.ones(m), rng.normal(size=(m, d - 1))])
    z = rng.normal(size=m)
    y = x @ np.linspace(1, -1, d) + 1.5 * z + rng.normal(size=m)
    yhat = x @ np.linspace(0.8, -0.8, d) + 1.2 * z + rng.normal(size=m)
    eta = x @ np.linspace(0.5, -0.5, d) + 1.5 * z
    b = (rng.random(m) < 1 / (1 + np.exp(-eta))).astype(float)
    p = 1 / (1 + np.exp(-(0.8 * eta + 0.5 * rng.normal(size=m))))
    return x, y, yhat, b, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi-python", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--N", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--d", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    X, y, yhat, b, p = simulate(rng, args.n, args.d)
    print(f"n = {args.n} labelled, d = {args.d}")
    print(f"{'N':>9} {'estimand':<9} {'dml-rs':>9} {'ppi_py':>9} {'speed-up':>9}")
    for N in args.N:
        Xu, _, yhat_u, _, pu = simulate(rng, N, args.d)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "data.npz"
            np.savez(path, X=X, Xu=Xu, y=y, yhat=yhat, yhat_u=yhat_u, b=b, p=p, pu=pu)
            script = pathlib.Path(tmp) / "bench.py"
            script.write_text(PPI_PY)
            theirs = json.loads(subprocess.run([args.ppi_python, str(script), str(path)], check=True,
                                               capture_output=True, text=True).stdout)
        calls = {
            "mean": lambda: dml_rs.PPI().fit(y, yhat, yhat_u),
            "ols": lambda: dml_rs.PPI("ols", fit_intercept=False).fit(y, yhat, yhat_u, X=X, X_unlabeled=Xu),
            "logistic": lambda: dml_rs.PPI("logistic", fit_intercept=False).fit(b, p, pu, X=X, X_unlabeled=Xu),
        }
        for name, f in calls.items():
            times = []
            for _ in range(3):
                t = time.perf_counter()
                f()
                times.append(time.perf_counter() - t)
            ours = min(times)
            print(f"{N:>9} {name:<9} {ours:>8.3f}s {theirs[name]:>8.3f}s {theirs[name] / ours:>8.1f}x")


if __name__ == "__main__":
    main()

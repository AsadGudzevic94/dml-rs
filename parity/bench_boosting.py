"""PLR with gradient-boosting nuisances: dml-rs native boosting vs scikit-learn.

Same data, 5-fold cross-fitting, 200 stages, learning rate 0.05, depth-3 trees.
dml-rs boosting reproduces scikit-learn's GradientBoostingRegressor predictions
(squared error, same splits), so the estimates agree; the difference is speed.

    .venv/Scripts/python parity/bench_boosting.py
"""

import os
import time

import doubleml as dml
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

import dml_rs

SETTINGS = dict(n_estimators=200, learning_rate=0.05, max_depth=3)


def simulate(n, p, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, p))
    confounder = 2 * np.sin(x[:, 0]) + 1.5 * (x[:, 1] > 0)
    d = confounder + rng.normal(size=n)
    y = 0.5 * d + 2 * confounder + rng.normal(size=n)
    return x, y, d


def timed(fn):
    start = time.perf_counter()
    coef = fn()
    return time.perf_counter() - start, coef


def main():
    print(f"cores: {os.cpu_count()}")
    print(f"{'n x p':>12} {'dml-rs native':>15} {'dml-rs + sklearn':>17} {'DoubleML + sklearn':>19}")
    for n, p in [(2_000, 10), (10_000, 20)]:
        x, y, d = simulate(n, p)
        native = dml_rs.GradientBoostingRegressor(**SETTINGS)
        sk = GradientBoostingRegressor(**SETTINGS, random_state=0)
        t1, c1 = timed(lambda: dml_rs.PLR(native, native).fit(x, y, d).coef_[0])
        t2, c2 = timed(lambda: dml_rs.PLR(sk, sk).fit(x, y, d).coef_[0])
        data = dml.DoubleMLData.from_arrays(x, y, d)
        t3, c3 = timed(lambda: dml.DoubleMLPLR(data, sk, sk, n_folds=5).fit().coef[0])
        print(f"{n:>6} x {p:<4} {t1:>9.2f} s     {t2:>11.2f} s     {t3:>13.2f} s")
        print(f"{'theta-hat':>12} {c1:>15.3f} {c2:>17.3f} {c3:>19.3f}   (true 0.5)")


if __name__ == "__main__":
    main()

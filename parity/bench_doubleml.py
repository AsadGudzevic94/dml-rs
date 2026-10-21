"""Time Python DoubleML PLR (LinearRegression nuisances, 5 folds) for comparison
with `cargo bench -p dml-core`. Same problem sizes as benches/plr.rs."""

import time

import doubleml as dml
import numpy as np
from sklearn.linear_model import LinearRegression

for n, p in [(1_000, 10), (10_000, 20), (100_000, 50)]:
    rng = np.random.default_rng(0)
    x = rng.random((n, p)) - 0.5
    d = x[:, 0] + 0.5 * x[:, 1] + rng.random(n)
    y = 0.5 * d + x[:, 0] - x[:, 2] + rng.random(n)
    data = dml.DoubleMLData.from_arrays(x, y, d)
    reps = 20 if n < 100_000 else 5
    start = time.perf_counter()
    for _ in range(reps):
        dml.DoubleMLPLR(data, LinearRegression(), LinearRegression(), n_folds=5).fit()
    print(f"n{n}_p{p}: {(time.perf_counter() - start) / reps * 1e3:.2f} ms")

"""Instrumental, multi-arm causal and regression forests of dml-rs against EconML's port of grf.

For every design, dml-rs fits the forest on cross-fitted residuals; EconML's forest
(``econml.grf.CausalIVForest``, ``CausalForest`` with one column per arm, ``RegressionForest``)
is grown on exactly the same residuals. Both are compared with the true function on a grid: RMSE,
coverage and width of 95% intervals, the correlation of the two forests' estimates, and fit time.
Their random number generators differ, so the comparison is statistical.

EconML pins its own scikit-learn, so it runs in a separate environment; this script calls itself
there for part B:

    python -m venv econvenv && econvenv/bin/pip install econml
    .venv/bin/python parity/compare_grf_forests.py --econml-python econvenv/bin/python
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPS, N, P = 20, 4000, 6


def sig(u):
    return 1 + 1 / (1 + np.exp(-20 * (u - 1 / 3)))


def part_a(out):
    """Project venv: simulate, fit dml-rs, save residuals and estimates."""
    import dml_rs
    from sklearn.linear_model import LinearRegression, LogisticRegression

    grid = np.random.default_rng(123).random((100, P))
    store = {"grid": grid}
    times = {"iv": [], "multi": [], "reg": []}
    for r in range(REPS):
        rng = np.random.default_rng(r)
        X = rng.random((N, P))
        # Instrumental: compliers 60%, always-takers 20% (+1.5), never-takers 20% (-0.5).
        z = (rng.random(N) < 0.5).astype(float)
        u = rng.random(N)
        d = np.where(u < 0.6, z, np.where(u < 0.8, 1.0, 0.0))
        shift = np.where(u < 0.6, 0.0, np.where(u < 0.8, 1.5, -0.5))
        y = sig(X[:, 0]) * d + X[:, 2] + shift + rng.normal(size=N)
        t = time.perf_counter()
        iv = dml_rs.InstrumentalForest(LinearRegression(), LogisticRegression(), LogisticRegression(),
                                       random_state=r).fit(X, y, d, z)
        times["iv"].append(time.perf_counter() - t)
        est, var = iv.result_.effect(grid)
        store.update({f"iv_X{r}": X, f"iv_y{r}": y - iv.y_hat_, f"iv_d{r}": d - iv.d_hat_,
                      f"iv_z{r}": z - iv.z_hat_, f"iv_est{r}": est, f"iv_var{r}": var})
        ate = iv.ate()
        store[f"iv_ate{r}"] = np.array([ate.coef_[0], ate.se_[0], sig(X[:, 0]).mean()])

        # Multi-arm: levels 0, 1, 2 with tau1 = s(x1), tau2 = 2 - s(x2).
        dm = rng.choice([0.0, 1.0, 2.0], size=N, p=[0.4, 0.3, 0.3])
        ym = X[:, 2] + np.where(dm == 1, sig(X[:, 0]), np.where(dm == 2, 2 - sig(X[:, 1]), 0.0)) + rng.normal(size=N)
        t = time.perf_counter()
        ma = dml_rs.MultiArmCausalForest(LinearRegression(), LogisticRegression(), random_state=r).fit(X, ym, dm)
        times["multi"].append(time.perf_counter() - t)
        est, var = ma.result_.effect(grid)
        onehot = np.column_stack([dm == 1, dm == 2]).astype(float)
        store.update({f"ma_X{r}": X, f"ma_y{r}": ym - ma.y_hat_, f"ma_w{r}": onehot - ma.propensity_[:, 1:],
                      f"ma_est{r}": est, f"ma_var{r}": var})

        # Regression.
        yr = np.sin(3 * X[:, 0]) + X[:, 1] + rng.normal(size=N)
        t = time.perf_counter()
        rf = dml_rs.RegressionForest(random_state=r).fit(X, yr)
        times["reg"].append(time.perf_counter() - t)
        est, var = rf.predict_with_variance(grid)
        store.update({f"rf_X{r}": X, f"rf_y{r}": yr, f"rf_est{r}": est, f"rf_var{r}": var})
    for k, v in times.items():
        store[f"time_{k}"] = np.array(v)
    np.savez(out, **store)


def part_b(npz):
    """EconML venv: grow EconML's forests on the same residuals and compare."""
    from econml.grf import CausalForest, CausalIVForest, RegressionForest

    z = np.load(npz)
    grid = z["grid"]
    truths = {"iv": [sig(grid[:, 0])], "ma": [sig(grid[:, 0]), 2 - sig(grid[:, 1])],
              "rf": [np.sin(3 * grid[:, 0]) + grid[:, 1]]}
    names = {"iv": "Instrumental forest", "ma": "Multi-arm causal forest", "rf": "Regression forest"}
    ours_time = {"iv": z["time_iv"], "ma": z["time_multi"], "rf": z["time_reg"]}
    for key in ["iv", "ma", "rf"]:
        rows = {"dml-rs": [], "EconML": []}
        corr, times = [], []
        for r in range(REPS):
            X = z[f"{key}_X{r}"]
            t = time.perf_counter()
            if key == "iv":
                f = CausalIVForest(n_estimators=2000, min_samples_leaf=5, random_state=r)
                f.fit(X, z[f"iv_d{r}"], z[f"iv_y{r}"], Z=z[f"iv_z{r}"])
            elif key == "ma":
                f = CausalForest(n_estimators=2000, min_samples_leaf=5, random_state=r)
                f.fit(X, z[f"ma_w{r}"], z[f"ma_y{r}"])
            else:
                f = RegressionForest(n_estimators=2000, min_samples_leaf=5, random_state=r)
                f.fit(X, z[f"rf_y{r}"])
            times.append(time.perf_counter() - t)
            pe, ve = f.predict_and_var(grid)
            k = len(truths[key])
            pe = pe.reshape(len(grid), k)
            ve = ve.reshape(len(grid), k, k).diagonal(axis1=1, axis2=2) if k > 1 else ve.reshape(len(grid), 1)
            po = z[f"{key}_est{r}"].reshape(len(grid), k)
            vo = z[f"{key}_var{r}"].reshape(len(grid), k)
            for name, p, v in [("dml-rs", po, vo), ("EconML", pe, ve)]:
                se = np.sqrt(np.maximum(v, 0))
                for a, truth in enumerate(truths[key]):
                    rows[name].append((np.sqrt(np.mean((p[:, a] - truth) ** 2)),
                                       np.mean(np.abs(p[:, a] - truth) <= 1.96 * se[:, a]),
                                       np.mean(2 * 1.96 * se[:, a])))
            corr += [np.corrcoef(po[:, a], pe[:, a])[0, 1] for a in range(k)]
        print(f"\n{names[key]} (n={N}, {REPS} replications, 2000 trees)")
        print(f"{'':8s} {'RMSE':>7s} {'coverage':>9s} {'CI width':>9s} {'time (s)':>9s}")
        for name, v in rows.items():
            v = np.array(v)
            secs = np.mean(ours_time[key]) if name == "dml-rs" else np.mean(times)
            print(f"{name:8s} {v[:, 0].mean():7.3f} {v[:, 1].mean():9.3f} {v[:, 2].mean():9.3f} {secs:9.2f}")
        print("correlation of the two forests' estimates: %.3f" % np.mean(corr))
        if key == "iv":
            ates = np.array([z[f"iv_ate{r}"] for r in range(REPS)])
            print("dml-rs average LATE: coverage %.2f, mean bias %.3f"
                  % (np.mean(np.abs(ates[:, 0] - ates[:, 2]) <= 1.96 * ates[:, 1]), np.mean(ates[:, 0] - ates[:, 2])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--econml-python")
    parser.add_argument("--part-b")
    args = parser.parse_args()
    if args.part_b:
        part_b(args.part_b)
        return
    if not args.econml_python:
        parser.error("--econml-python is required")
    with tempfile.TemporaryDirectory() as tmp:
        npz = str(Path(tmp) / "grf.npz")
        part_a(npz)
        subprocess.run([args.econml_python, __file__, "--part-b", npz], check=True)


if __name__ == "__main__":
    main()

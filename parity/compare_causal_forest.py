"""Causal forests of dml-rs against EconML's port of grf (``econml.grf.CausalForest``).

Simulates the Wager and Athey (2018) design (tau(x) = s(x1) s(x2), randomised treatment), fits
``dml_rs.CausalForest`` and grows EconML's forest on exactly the same residuals, then compares the
error against the true tau(x), the coverage and width of 95% intervals, and agreement of the two.
EconML pins its own scikit-learn, so it runs in a separate environment:

    python -m venv econvenv && econvenv/bin/pip install econml
    .venv/bin/python parity/compare_causal_forest.py --econml-python econvenv/bin/python
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

PART_A = '"""Part A (project venv): simulate, fit dml-rs causal forests, save residuals and estimates."""\nimport sys, time\nimport numpy as np\nimport dml_rs\nfrom sklearn.linear_model import LinearRegression, LogisticRegression\n\nout = sys.argv[1]\nREPS, N, P = 20, 4000, 6\ngrid = np.full((100, P), 0.5)\nrng_grid = np.random.default_rng(123)\ngrid = rng_grid.random((100, P))\n\n\ndef sig(u):\n    return 1 + 1 / (1 + np.exp(-20 * (u - 1 / 3)))\n\n\ndef tau(X):\n    return sig(X[:, 0]) * sig(X[:, 1])\n\n\nstore = {"grid": grid, "tau_grid": tau(grid)}\ntimes = []\nfor r in range(REPS):\n    rng = np.random.default_rng(r)\n    X = rng.random((N, P))\n    d = (rng.random(N) < 0.5).astype(float)\n    y = X[:, 2] + tau(X) * d + rng.normal(size=N)\n    t = time.perf_counter()\n    cf = dml_rs.CausalForest(LinearRegression(), LogisticRegression(), random_state=r).fit(X, y, d)\n    times.append(time.perf_counter() - t)\n    tau_hat, var = cf.result_.effect(grid)\n    store[f"X{r}"] = X\n    store[f"yres{r}"] = y - cf.y_hat_\n    store[f"dres{r}"] = d - cf.d_hat_\n    store[f"ours_tau{r}"] = tau_hat\n    store[f"ours_var{r}"] = var\n    ate = cf.ate()\n    store[f"ours_ate{r}"] = np.array([ate.coef_[0], ate.se_[0], tau(X).mean()])\nnp.savez(out, **store)\nprint("dml-rs fit time per forest (2000 trees, n=4000): %.2f s" % np.mean(times))\n'

PART_B = '"""Part B (econml venv): grow EconML\'s grf causal forest on the same residuals and compare."""\nimport sys, time\nimport numpy as np\nfrom econml.grf import CausalForest\n\nz = np.load(sys.argv[1])\ngrid, truth = z["grid"], z["tau_grid"]\nREPS = sum(1 for k in z.files if k.startswith("ours_tau"))\nrows = {"dml-rs": [], "econml": []}\ncorr, times = [], []\nfor r in range(REPS):\n    X, yres, dres = z[f"X{r}"], z[f"yres{r}"], z[f"dres{r}"]\n    t = time.perf_counter()\n    ef = CausalForest(n_estimators=2000, min_samples_leaf=5, random_state=r).fit(X, dres, yres)\n    times.append(time.perf_counter() - t)\n    pe, var_e = ef.predict_and_var(grid)\n    pe, var_e = pe.ravel(), var_e.reshape(len(grid))\n    po, var_o = z[f"ours_tau{r}"], z[f"ours_var{r}"]\n    for name, p, v in [("dml-rs", po, var_o), ("econml", pe, var_e)]:\n        se = np.sqrt(np.maximum(v, 0))\n        rows[name].append((np.sqrt(np.mean((p - truth) ** 2)), np.mean(np.abs(p - truth) <= 1.96 * se),\n                           np.mean(2 * 1.96 * se)))\n    corr.append(np.corrcoef(po, pe)[0, 1])\nprint("econml fit time per forest: %.2f s" % np.mean(times))\nprint(f"{\'\':8s} {\'RMSE\':>8s} {\'coverage\':>9s} {\'CI width\':>9s}")\nfor name, v in rows.items():\n    v = np.array(v)\n    print(f"{name:8s} {v[:, 0].mean():8.3f} {v[:, 1].mean():9.3f} {v[:, 2].mean():9.3f}")\nprint("correlation of the two forests\' estimates: %.3f" % np.mean(corr))\nates = np.array([z[f"ours_ate{r}"] for r in range(REPS)])\nprint("dml-rs ATE: coverage %.2f, mean bias %.3f" % (np.mean(np.abs(ates[:, 0] - ates[:, 2]) <= 1.96 * ates[:, 1]),\n                                                    np.mean(ates[:, 0] - ates[:, 2])))\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--econml-python", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        npz = str(Path(tmp) / "cf.npz")
        subprocess.run([sys.executable, "-c", PART_A, npz], check=True)
        subprocess.run([args.econml_python, "-c", PART_B, npz], check=True)


if __name__ == "__main__":
    main()

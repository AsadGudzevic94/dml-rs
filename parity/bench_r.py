"""dml-rs against the R packages it ports: grf::causal_forest, policytree, synthdid,
causalweight::medDML and contdid, on the same data. The table is in docs/benchmarks.md.

    python parity/bench_r.py [--rscript Rscript]

Writes the forest and policy-tree data, runs parity/bench_r.R (which times R and writes the
other inputs), then times dml-rs on the same inputs.
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import dml_rs

HERE = Path(__file__).resolve().parent


def best(f, reps=2):
    out = []
    for _ in range(reps):
        start = time.perf_counter()
        f()
        out.append(time.perf_counter() - start)
    return min(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    folder = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    ours = {}
    for n in (2000, 10000, 50000):
        x = rng.normal(size=(n, 10))
        w = (rng.random(n) < 1 / (1 + np.exp(-x[:, 0]))).astype(float)
        y = x[:, 1] + w * (1 + (x[:, 2] > 0)) + rng.normal(size=n)
        np.savetxt(folder / f"cf_{n}.csv", np.column_stack([y, w, x]), delimiter=",")
        # Linear nuisances on both sides (grf gets Y.hat / W.hat from its own regression
        # forests otherwise), so that the forest itself is compared.
        ours[f"causal_forest n={n}"] = best(lambda: dml_rs.CausalForest(
            dml_rs.Ols(), dml_rs.Ols(), n_estimators=2000, random_state=1).fit(x, y, w))
    for n in (1000, 5000):
        x = rng.normal(size=(n, 10))
        gamma = np.column_stack([x[:, 0], -x[:, 0] + x[:, 1], 0.5 * x[:, 2]]) + rng.normal(size=(n, 3))
        np.savetxt(folder / f"pt_{n}.csv", np.column_stack([gamma, x]), delimiter=",")
        ours[f"policy_tree n={n}"] = best(lambda: dml_rs.OptimalPolicyTree(depth=2).fit(x, gamma))

    subprocess.run([args.rscript, str(HERE / "bench_r.R"), str(folder)], check=True)

    Y = np.loadtxt(folder / "prop99_Y.csv", delimiter=",", skiprows=1)
    n0, t0 = (int(v) for v in (folder / "prop99_dims.txt").read_text().split())
    ours["synthdid estimate"] = best(lambda: dml_rs.SyntheticDID(se_method=None).fit(Y, n0, t0))
    ours["synthdid placebo"] = best(lambda: dml_rs.SyntheticDID(se_method="placebo").fit(Y, n0, t0))
    for n in (5000, 20000):
        a = np.loadtxt(folder / f"med_{n}.csv", delimiter=",", skiprows=1)
        y, d, m, x = a[:, 0], a[:, 1], a[:, 2:3], a[:, 3:]
        # medDML's defaults: unpenalised logit, 3 folds, dropping propensities beyond 0.05.
        ours[f"medDML n={n}"] = best(lambda: dml_rs.Mediation(
            dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf), trimming_rule="drop",
            trimming_threshold=0.05, n_folds=3).fit(x, y, d, m))
        c = np.loadtxt(folder / f"contdid_{n}.csv", delimiter=",", skiprows=1)
        dy, dose = c[:, 0], c[:, 1]
        grid = np.linspace(dose[dose > 0].min(), dose[dose > 0].max(), 50)
        ours[f"cont_did n={n}"] = best(lambda: dml_rs.ContinuousDID(degree=3, num_knots=0, grid=grid).fit(
            None, dy, dose))

    theirs = dict(line.rsplit(" ", 1) for line in (folder / "r_times.txt").read_text().splitlines())
    shutil.rmtree(folder, ignore_errors=True)
    print("| Task | dml-rs | R | Speed-up |")
    print("|---|--:|--:|--:|")
    for task, t_ours in ours.items():
        t_r = float(theirs[task])
        print(f"| {task} | {t_ours:.3f} s | {t_r:.3f} s | **{t_r / t_ours:.1f}x** |")


if __name__ == "__main__":
    main()

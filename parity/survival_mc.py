"""Monte Carlo for CausalSurvival against grf::causal_survival_forest on the same data.

    python parity/survival_mc.py [--reps 60] [--n 2000] [--rscript Rscript]

Treatment and censoring depend on the covariates, and the event and censoring hazards are
nonlinear in them, so Kaplan-Meier by arm is biased. The table is in docs/validation.md.
"""

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

import dml_rs

HORIZON = 1.0
HERE = Path(__file__).resolve().parent


def rates(x, treated):
    event = np.exp(0.8 * x[:, 0] + 0.5 * x[:, 1] ** 2 - 0.5) * (0.5 if treated else 1.0)
    censoring = 0.3 * np.exp(-0.8 * x[:, 0])
    return event, censoring


def simulate(n, rng):
    x = rng.uniform(-1, 1, size=(n, 4))
    d = (rng.random(n) < 1 / (1 + np.exp(-1.2 * x[:, 0]))).astype(float)
    event_rate = rates(x, True)[0] * d + rates(x, False)[0] * (1 - d)
    t = rng.exponential(1 / event_rate)
    c = rng.exponential(1 / rates(x, False)[1])
    return x, np.minimum(t, c), (t <= c).astype(float), d


def truth():
    x = np.random.default_rng(0).uniform(-1, 1, size=(5_000_000, 4))
    treated, control = rates(x, True)[0], rates(x, False)[0]
    survival = np.exp(-treated * HORIZON).mean() - np.exp(-control * HORIZON).mean()
    rmst = ((1 - np.exp(-treated * HORIZON)) / treated).mean() - ((1 - np.exp(-control * HORIZON)) / control).mean()
    return {"survival": survival, "rmst": rmst}


def fit_dml(x, time_, event, d, estimand, seed):
    forest = dml_rs.RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=seed)
    logit = LogisticRegression(max_iter=1000)
    fit = dml_rs.CausalSurvival(forest, forest, logit, horizons=[HORIZON], estimand=estimand, grid=20,
                                random_state=seed).fit(x, time_, event, d)
    return fit.coef_[0], fit.se_[0]


def table(name, rows, true_value):
    rows = np.asarray(rows)
    est, se = rows[:, 0], rows[:, 1]
    cover = np.mean(np.abs(est - true_value) <= 1.96 * se)
    return (f"| {name} | {est.mean() - true_value:+.4f} | {est.std(ddof=1):.4f} | {se.mean():.4f} | "
            f"{np.sqrt(np.mean((est - true_value) ** 2)):.4f} | {cover:.2f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=60)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()

    values = truth()
    results = {key: [] for key in ("dml survival", "dml rmst")}
    seconds = 0.0
    folder = Path(tempfile.mkdtemp())
    for seed in range(args.reps):
        x, time_, event, d = simulate(args.n, np.random.default_rng(seed))
        np.savetxt(folder / f"{seed}.csv", np.column_stack((time_, event, d, x)), delimiter=",")
        start = time.perf_counter()
        for estimand in ("survival", "rmst"):
            results[f"dml {estimand}"].append(fit_dml(x, time_, event, d, estimand, seed))
        seconds += time.perf_counter() - start

    subprocess.run([args.rscript, str(HERE / "survival_mc_grf.R"), str(folder), str(args.reps), str(HORIZON)], check=True)
    grf = np.loadtxt(folder / "grf.csv", delimiter=",", skiprows=1)
    shutil.rmtree(folder, ignore_errors=True)

    print(f"n = {args.n}, {args.reps} datasets, truth: survival {values['survival']:.4f}, rmst {values['rmst']:.4f}")
    print(f"dml-rs seconds per dataset and estimand: {seconds / args.reps / 2:.2f}; grf: {grf[:, 4].mean():.2f}")
    print("| Estimator | Bias | Sd | Mean se | RMSE | Coverage |")
    print("|---|---|---|---|---|---|")
    for estimand, column in (("survival", 0), ("rmst", 2)):
        print(table(f"dml-rs, {estimand}", results[f"dml {estimand}"], values[estimand]))
        print(table(f"grf, {estimand}", grf[:, column:column + 2], values[estimand]))


if __name__ == "__main__":
    main()

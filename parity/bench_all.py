"""Benchmarks for the documentation: dml-rs with its built-in learners against DoubleML with the
matching scikit-learn learners, on the same data with 5-fold cross-fitting.

Each cell is the wall-clock time of one complete fit (the faster of two runs). Both libraries use
all cores: dml-rs fits folds in parallel and its forests / histogram boosting are multi-threaded;
scikit-learn forests use ``n_jobs=-1`` and its histogram boosting uses OpenMP. scikit-learn's
``LassoCV`` and ``GradientBoosting*`` are single-threaded, as DoubleML's fold loop.

    .venv/Scripts/python parity/bench_all.py            # prints a Markdown table
"""

import json
import os
import platform
import sys
import time
import warnings

import doubleml as dml
import numpy as np
import sklearn
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV

import dml_rs

warnings.simplefilter("ignore")
P = 20


def simulate(n, binary, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=(n, P))
    confounder = 2 * np.sin(x[:, 0]) + 1.5 * (x[:, 1] > 0) + 0.5 * x[:, 2]
    if binary:
        d = (rng.random(n) < 1 / (1 + np.exp(-0.5 * confounder))).astype(float)
    else:
        d = confounder + rng.normal(size=n)
    y = 0.5 * d + 2 * confounder + rng.normal(size=n)
    return x, y, d


def learners(kind):
    """(dml-rs regressor, dml-rs classifier, scikit-learn regressor, scikit-learn classifier)."""
    if kind == "Lasso / logistic, CV penalty":
        return (dml_rs.LassoCV(), dml_rs.LogisticRegressionCV(), LassoCV(), LogisticRegressionCV(scoring="neg_log_loss"))
    if kind == "Random forest (100 trees)":
        kw = dict(n_estimators=100, min_samples_leaf=5, random_state=0)
        return (dml_rs.RandomForestRegressor(max_features=1.0, **kw), dml_rs.RandomForestClassifier(**kw),
                RandomForestRegressor(max_features=1.0, n_jobs=-1, **kw), RandomForestClassifier(n_jobs=-1, **kw))
    if kind == "Gradient boosting (100 trees)":
        kw = dict(n_estimators=100, random_state=0)
        return (dml_rs.GradientBoostingRegressor(**kw), dml_rs.GradientBoostingClassifier(**kw),
                GradientBoostingRegressor(**kw), GradientBoostingClassifier(**kw))
    if kind == "Histogram gradient boosting (100 trees)":
        kw = dict(max_iter=100, early_stopping=False, random_state=0)
        return (dml_rs.HistGradientBoostingRegressor(**kw), dml_rs.HistGradientBoostingClassifier(**kw),
                HistGradientBoostingRegressor(**kw), HistGradientBoostingClassifier(**kw))
    raise ValueError(kind)


CASES = [
    ("PLR", "Lasso / logistic, CV penalty", [10_000, 100_000]),
    ("PLR", "Random forest (100 trees)", [10_000, 50_000]),
    ("PLR", "Gradient boosting (100 trees)", [10_000, 50_000]),
    ("PLR", "Histogram gradient boosting (100 trees)", [10_000, 100_000, 1_000_000]),
    ("IRM", "Lasso / logistic, CV penalty", [10_000, 100_000]),
    ("IRM", "Random forest (100 trees)", [10_000, 50_000]),
    ("IRM", "Histogram gradient boosting (100 trees)", [10_000, 100_000, 1_000_000]),
]


def best_of(fn, runs=2):
    times, out = [], None
    for _ in range(runs):
        start = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - start)
    return min(times), out


def run_case(model, kind, n):
    x, y, d = simulate(n, binary=model == "IRM")
    reg, clf, sk_reg, sk_clf = learners(kind)
    data = dml.DoubleMLData.from_arrays(x, y, d)
    if model == "PLR":
        ours = lambda: dml_rs.PLR(reg, reg).fit(x, y, d).coef_[0]  # noqa: E731
        theirs = lambda: dml.DoubleMLPLR(data, sk_reg, sk_reg, n_folds=5).fit().coef[0]  # noqa: E731
    else:
        ours = lambda: dml_rs.IRM(reg, clf).fit(x, y, d).coef_[0]  # noqa: E731
        theirs = lambda: dml.DoubleMLIRM(data, sk_reg, sk_clf, n_folds=5).fit().coef[0]  # noqa: E731
    t_ours, c_ours = best_of(ours)
    t_theirs, c_theirs = best_of(theirs, runs=1 if n >= 500_000 else 2)
    return {"model": model, "learner": kind, "n": n, "dml_rs": t_ours, "doubleml": t_theirs,
            "speedup": t_theirs / t_ours, "coef_dml_rs": float(c_ours), "coef_doubleml": float(c_theirs)}


def main():
    meta = {"cpus": os.cpu_count(), "platform": platform.platform(), "python": sys.version.split()[0],
            "dml_rs": dml_rs.__version__, "doubleml": dml.__version__, "sklearn": sklearn.__version__}
    print(json.dumps(meta))
    rows = []
    for model, kind, sizes in CASES:
        for n in sizes:
            r = run_case(model, kind, n)
            rows.append(r)
            print(json.dumps(r), flush=True)
    print()
    print("| Model | Learners | n | dml-rs | DoubleML + scikit-learn | Speed-up |")
    print("|---|---|--:|--:|--:|--:|")
    for r in rows:
        print(f"| {r['model']} | {r['learner']} | {r['n']:,} | {r['dml_rs']:.2f} s | {r['doubleml']:.2f} s | "
              f"**{r['speedup']:.1f}x** |")


if __name__ == "__main__":
    main()

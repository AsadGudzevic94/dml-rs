"""Monte Carlo for SurrogateIndex: bias, RMSE and coverage of the long-term effect.

Experiment with assignment confounded by X (P(w=1 | X) depends on x1), two surrogates with
effects 1 and 0.5, long-term outcome y = 2 s1 + s2 + x2 + noise, so the long-term effect is 2.5.
The naive estimator compares the mean predicted surrogate index between the arms (Athey et al.'s
plug-in without adjustment for the confounded assignment).

    python parity/surrogate_mc.py --reps 500
"""

import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import dml_rs

EFFECT = 2.5


def draw(n, rng, experiment):
    x = rng.normal(size=(n, 2))
    w = (0.6 * x[:, 0] + rng.normal(size=n) > 0).astype(float) if experiment else np.zeros(n)
    s = np.column_stack([w + 0.5 * x[:, 0] + rng.normal(size=n),
                         0.5 * w - 0.3 * x[:, 1] + rng.normal(size=n)])
    y = 2 * s[:, 0] + s[:, 1] + x[:, 1] + rng.normal(size=n)
    return x, s, w, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--forest-reps", type=int, default=100)
    parser.add_argument("--n-exp", type=int, default=1000)
    parser.add_argument("--n-obs", type=int, default=2000)
    args = parser.parse_args()
    learners = {
        "OLS / logit": (lambda: (dml_rs.Ols(), dml_rs.LogisticRegression(C=np.inf)), args.reps),
        "random forests": (lambda: (RandomForestRegressor(200, min_samples_leaf=5, random_state=0),
                                    RandomForestClassifier(200, min_samples_leaf=5, random_state=0)),
                           args.forest_reps),
    }
    rows = []
    naive = []
    for name, (make, reps) in learners.items():
        est, cover = [], 0
        for rep in range(reps):
            rng = np.random.default_rng(rep)
            xe, se, w, _ = draw(args.n_exp, rng, True)
            xo, so, _, y = draw(args.n_obs, rng, False)
            ml_g, ml_m = make()
            model = dml_rs.SurrogateIndex(ml_g, ml_m, random_state=rep).fit(xe, se, w, xo, so, y)
            est.append(model.coef_[0])
            lo, hi = model.confint()[0]
            cover += lo <= EFFECT <= hi
            if name == "OLS / logit":
                index = LinearRegression().fit(np.column_stack([so, xo]), y).predict(
                    np.column_stack([se, xe]))
                naive.append(index[w == 1].mean() - index[w == 0].mean())
        est = np.array(est)
        rows.append({"estimator": f"SurrogateIndex, {name}", "reps": reps,
                     "bias": est.mean() - EFFECT, "rmse": np.sqrt(np.mean((est - EFFECT) ** 2)),
                     "coverage": cover / reps})
    naive = np.array(naive)
    rows.append({"estimator": "naive plug-in", "reps": args.reps, "bias": naive.mean() - EFFECT,
                 "rmse": np.sqrt(np.mean((naive - EFFECT) ** 2)), "coverage": None})
    for r in rows:
        cov = "" if r["coverage"] is None else f"{r['coverage']:.3f}"
        print(f"{r['estimator']:36} {r['reps']:4d} {r['bias']:8.3f} {r['rmse']:8.3f} {cov}")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()

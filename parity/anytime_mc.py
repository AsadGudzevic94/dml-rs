"""Monte Carlo study of the anytime-valid confidence sequences for a doubly robust ATE (IRM).

An observational stream with confounding is looked at every 250 units up to 5000 (20 looks).
For each repetition we record whether the 95 % confidence sequence or the repeated 95 %
confidence interval misses the true ATE at *any* look (time-uniform error), and, under a true
effect, at which sample size the running sequence first excludes zero.

    python parity/anytime_mc.py --reps 300
"""

import argparse
import time

import numpy as np

import dml_rs


def simulate(n, theta, rng):
    x = rng.normal(size=(n, 5))
    p = 1 / (1 + np.exp(-(0.6 * x[:, 0] - 0.6 * x[:, 1])))
    d = (rng.random(n) < p).astype(float)
    y = theta * d + x[:, 0] + 0.5 * x[:, 1] + 0.5 * x[:, 2] * d + rng.normal(size=n)
    return x, y, d


def irm():
    return dml_rs.IRM(dml_rs.Ols(), dml_rs.LogisticRegression(), n_folds=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--opt-time", type=int, default=1000)
    args = parser.parse_args()
    looks = list(range(250, 5001, 250))
    # The ATE of y = θ d + 0.5 x₂ d + ... is θ (E[x₂] = 0).
    start = time.time()
    for theta in (0.0, 0.15):
        cs_miss = ci_miss = 0
        stops_cs, stops_ci = [], []
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + int(theta * 100))
            x, y, d = simulate(looks[-1], theta, rng)
            seq = dml_rs.ConfidenceSequence(0.95, opt_time=args.opt_time).run(irm(), looks, X=x, y=y, d=d)
            cs_out = bool(np.any((seq.intervals_[:, 0, 0] > theta) | (seq.intervals_[:, 0, 1] < theta)))
            ci = np.array([irm().fit(x[:t], y[:t], d[:t]).confint(0.95)[0] for t in looks])
            ci_out = bool(np.any((ci[:, 0] > theta) | (ci[:, 1] < theta)))
            cs_miss += cs_out
            ci_miss += ci_out
            if theta != 0:
                stops_cs.append(seq.first_exclusion(0.0)[0] or np.inf)
                hit = np.flatnonzero(ci[:, 0] > 0)
                stops_ci.append(looks[hit[0]] if hit.size else np.inf)
        print(f"theta={theta}: misses theta at some look: sequence {cs_miss / args.reps:.3f}, "
              f"repeated intervals {ci_miss / args.reps:.3f}  ({args.reps} reps, {len(looks)} looks)")
        if theta != 0:
            stops_cs, stops_ci = np.array(stops_cs), np.array(stops_ci)
            print(f"  excludes 0 by n=5000: sequence {np.mean(np.isfinite(stops_cs)):.3f}, "
                  f"repeated intervals {np.mean(np.isfinite(stops_ci)):.3f}; median stopping n: "
                  f"sequence {np.median(stops_cs):.0f}, intervals {np.median(stops_ci):.0f}")
    print(f"{time.time() - start:.0f} s")


if __name__ == "__main__":
    main()

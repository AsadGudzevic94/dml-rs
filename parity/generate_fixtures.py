"""Generate parity fixtures from Python DoubleML for the dml-rs test suite.

Each fixture stores the data, the exact sample splits, the learner spec and
DoubleML's estimates. The Rust tests (crates/dml-core/tests/parity.rs) refit
on the same folds and compare.

    python -m venv .venv
    .venv/Scripts/pip install -r parity/requirements.txt   # or .venv/bin/pip
    .venv/Scripts/python parity/generate_fixtures.py
"""

import json
from pathlib import Path

import doubleml as dml
import numpy as np
import sklearn
from doubleml.plm.datasets import make_plr_CCDDHNR2018
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

OUT = Path(__file__).resolve().parents[1] / "crates" / "dml-core" / "tests" / "fixtures"


def learner(spec):
    kind = spec["kind"]
    if kind == "ols":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=spec["alpha"])
    raise ValueError(kind)


def draw_folds(n, n_folds, n_rep, seed):
    reps = []
    for r in range(n_rep):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        reps.append([(train, test) for train, test in kf.split(np.zeros(n))])
    return reps


def folds_json(reps):
    return [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in reps]


def results_json(model):
    ci = model.confint(level=0.95).to_numpy()
    preds = {k: v[:, :, 0].T.tolist() for k, v in model.predictions.items() if v is not None}
    return {
        "coef": model.coef.tolist(),
        "se": model.se.tolist(),
        "t_stat": model.t_stat.tolist(),
        "pval": model.pval.tolist(),
        "ci95": ci.tolist(),
        "all_coef": model.all_coef.tolist(),
        "all_se": model.all_se.tolist(),
        # predictions[name][rep] -> length-n vector (first treatment)
        "predictions": {k: [p for p in v] for k, v in preds.items()},
    }


def plr_case(name, seed, n_obs, dim_x, learner_spec, score="partialling out", n_folds=5, n_rep=1):
    np.random.seed(seed)
    x, y, d = make_plr_CCDDHNR2018(n_obs=n_obs, dim_x=dim_x, alpha=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    reps = draw_folds(n_obs, n_folds, n_rep, seed)
    model = dml.DoubleMLPLR(
        data,
        ml_l=learner(learner_spec),
        ml_m=learner(learner_spec),
        n_folds=n_folds,
        n_rep=n_rep,
        score=score,
        draw_sample_splitting=False,
    )
    model.set_sample_splitting(reps)
    model.fit(store_predictions=True)
    return {
        "name": name,
        "model": "plr",
        "score": score,
        "learner": learner_spec,
        "x": x.tolist(),
        "y": y.tolist(),
        "d": d.tolist(),
        "folds": folds_json(reps),
        "expected": results_json(model),
    }


CASES = [
    lambda: plr_case("plr_ols", seed=11, n_obs=500, dim_x=10, learner_spec={"kind": "ols"}),
    lambda: plr_case("plr_ridge", seed=12, n_obs=400, dim_x=20, learner_spec={"kind": "ridge", "alpha": 5.0}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for make in CASES:
        case = make()
        case["versions"] = {"doubleml": dml.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}
        path = OUT / f"{case['name']}.json"
        path.write_text(json.dumps(case))
        print(f"wrote {path.relative_to(OUT.parents[2])}: coef={case['expected']['coef']}")


if __name__ == "__main__":
    main()

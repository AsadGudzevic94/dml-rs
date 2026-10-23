"""Generate parity fixtures from Python DoubleML for the dml-rs test suite.

Each fixture stores the data, the exact sample splits, the learner specs and
DoubleML's estimates. The Rust tests (crates/dml-core/tests/parity.rs) refit
on the same folds and compare. For bootstrap cases the multiplier weights
DoubleML draws are recorded too, so Rust can replay them.

    python -m venv .venv
    .venv/Scripts/pip install -r parity/requirements.txt   # or .venv/bin/pip
    .venv/Scripts/python parity/generate_fixtures.py
"""

import json
import warnings
from pathlib import Path

import doubleml as dml
import numpy as np
import pandas as pd
import sklearn
from doubleml.did.datasets import make_did_SZ2020
from doubleml.irm.datasets import make_iivm_data, make_irm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plr_CCDDHNR2018
from doubleml.utils._estimation import _draw_weights
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "crates" / "dml-core" / "tests" / "fixtures"

OLS = {"kind": "ols"}
LOGIT = {"kind": "logit", "c": 1.0}


def learner(spec):
    kind = spec["kind"]
    if kind == "ols":
        return LinearRegression()
    if kind == "ridge":
        return Ridge(alpha=spec["alpha"])
    if kind == "lasso":
        return Lasso(alpha=spec["alpha"], tol=1e-14, max_iter=1_000_000)
    if kind == "logit":
        return LogisticRegression(C=spec["c"], solver="newton-cholesky", tol=1e-14, max_iter=1000)
    raise ValueError(kind)


def draw_folds(n, n_folds, n_rep, seed):
    reps = []
    for r in range(n_rep):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        reps.append([(train, test) for train, test in kf.split(np.zeros(n))])
    return reps


def fit_case(name, model_kind, data, arrays, learners, options, n_folds=5, n_rep=1, seed=0, boot=None):
    """Fit a DoubleML model on fixed folds and serialise everything needed to replay it."""
    n = len(arrays["y"])
    reps = draw_folds(n, n_folds, n_rep, seed)
    kwargs = {k: learner(v) for k, v in learners.items()}
    cls = {
        "plr": dml.DoubleMLPLR,
        "irm": dml.DoubleMLIRM,
        "pliv": dml.DoubleMLPLIV,
        "iivm": dml.DoubleMLIIVM,
        "did": dml.DoubleMLDID,
        "did_cs": dml.DoubleMLDIDCS,
    }[model_kind]
    model = cls(data, **kwargs, n_folds=n_folds, n_rep=n_rep, draw_sample_splitting=False, **options)
    model.set_sample_splitting(reps)
    model.fit(store_predictions=True)

    ci = model.confint(level=0.95).to_numpy()
    expected = {
        "coef": model.coef.tolist(),
        "se": model.se.tolist(),
        "t_stat": model.t_stat.tolist(),
        "pval": model.pval.tolist(),
        "ci95": ci.tolist(),
        "all_coef": model.all_coef.tolist(),
        "all_se": model.all_se.tolist(),
        # predictions[name][treatment][rep] -> length-n vector
        "predictions": {
            k: np.transpose(v, (2, 1, 0)).tolist() for k, v in model.predictions.items() if v is not None
        },
        "p_bonferroni": model.p_adjust("bonferroni")["pval"].tolist(),
        "p_holm": model.p_adjust("holm")["pval"].tolist(),
    }
    case = {
        "name": name,
        "model": model_kind,
        "options": options,
        "learners": learners,
        **{k: np.asarray(v).tolist() for k, v in arrays.items()},
        "folds": [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in reps],
        "expected": expected,
    }
    case["_model"] = model
    if boot is not None:
        method, n_boot, boot_seed = boot
        np.random.seed(boot_seed)
        weights = [_draw_weights(method, n_boot, n) for _ in range(n_rep)]
        np.random.seed(boot_seed)
        model.bootstrap(method=method, n_rep_boot=n_boot)
        expected["boot"] = {
            "method": method,
            "weights": [w.tolist() for w in weights],
            "joint_ci95": model.confint(joint=True, level=0.95).to_numpy().tolist(),
            "romano_wolf": model.p_adjust("romano-wolf")["pval"].tolist(),
        }
    return case


def plr(name, seed, n_obs, dim_x, spec, score="partialling out", n_rep=1, boot=None):
    np.random.seed(seed)
    x, y, d = make_plr_CCDDHNR2018(n_obs=n_obs, dim_x=dim_x, alpha=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    learners = {"ml_l": spec, "ml_m": spec}
    if score == "IV-type":
        learners["ml_g"] = spec
    return fit_case(name, "plr", data, {"x": x, "y": y, "d": d}, learners, {"score": score},
                    n_rep=n_rep, seed=seed, boot=boot)


def plr_multi(name, seed, n_obs, n_rep, boot):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_obs, 5))
    d = np.column_stack([
        x[:, 0] + 0.5 * x[:, 1] + rng.normal(size=n_obs),
        -0.5 * x[:, 0] + x[:, 2] + rng.normal(size=n_obs),
        x[:, 3] + rng.normal(size=n_obs),
    ])
    y = 0.5 * d[:, 0] - 0.3 * d[:, 1] + 0.0 * d[:, 2] + x[:, 0] - x[:, 4] + rng.normal(size=n_obs)
    data = dml.DoubleMLData.from_arrays(x, y, d)
    return fit_case(name, "plr", data, {"x": x, "y": y, "d": d}, {"ml_l": OLS, "ml_m": OLS},
                    {"score": "partialling out"}, n_rep=n_rep, seed=seed, boot=boot)


def irm(name, seed, score, normalize_ipw=False, n_rep=1, boot=None):
    np.random.seed(seed)
    x, y, d = make_irm_data(n_obs=500, dim_x=8, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    return fit_case(name, "irm", data, {"x": x, "y": y, "d": d}, {"ml_g": OLS, "ml_m": LOGIT},
                    {"score": score, "normalize_ipw": normalize_ipw}, n_rep=n_rep, seed=seed, boot=boot)


def iivm(name, seed, subgroups=None, n_rep=1):
    np.random.seed(seed)
    x, y, d, z = make_iivm_data(n_obs=600, dim_x=6, theta=0.5, alpha_x=1.0, return_type="array")
    z = np.asarray(z).reshape(-1)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    options = {"score": "LATE"}
    if subgroups is not None:
        options["subgroups"] = subgroups
    return fit_case(name, "iivm", data, {"x": x, "y": y, "d": d, "z": z},
                    {"ml_g": OLS, "ml_m": LOGIT, "ml_r": LOGIT}, options, n_rep=n_rep, seed=seed)


def did(name, seed, score, normalize, cross_section=False):
    np.random.seed(seed)
    if cross_section:
        x, y, d, t = make_did_SZ2020(n_obs=800, dgp_type=1, cross_sectional_data=True, return_type="array")
        data = dml.DoubleMLDIDData.from_arrays(x, y, d, t=t)
        arrays = {"x": x, "y": y, "d": d, "t": t}
    else:
        x, y, d, _ = make_did_SZ2020(n_obs=600, dgp_type=1, cross_sectional_data=False, return_type="array")
        data = dml.DoubleMLDIDData.from_arrays(x, y, d)
        arrays = {"x": x, "y": y, "d": d}
    learners = {"ml_g": OLS}
    if score == "observational":
        learners["ml_m"] = LOGIT
    return fit_case(name, "did_cs" if cross_section else "did", data, arrays, learners,
                    {"score": score, "in_sample_normalization": normalize}, seed=seed)


def irm_cate(name, seed):
    case = irm(name, seed, "ATE", n_rep=2)
    model = case["_model"]
    x = np.asarray(case["x"])
    basis = pd.DataFrame({"const": 1.0, "x0": x[:, 0], "x1": x[:, 1]})
    cate = model.cate(basis)
    groups = pd.DataFrame({"low": x[:, 0] <= 0, "high": x[:, 0] > 0})
    gate = model.gate(groups)
    case["expected"]["blp"] = {
        "basis": basis.to_numpy().tolist(),
        "coef": np.asarray(cate.coef).tolist(),
        "se": np.asarray(cate.se).tolist(),
        "ci_coef": cate.confint(level=0.95).to_numpy().tolist(),
        "ci_at_first10": cate.confint(basis.iloc[:10], level=0.95).to_numpy().tolist(),
    }
    case["expected"]["gate"] = {
        "groups": groups.to_numpy(dtype=float).tolist(),
        "coef": np.asarray(gate.coef).tolist(),
        "se": np.asarray(gate.se).tolist(),
        "ci": gate.confint(level=0.95).to_numpy().tolist(),
    }
    return case


def pliv(name, seed, n_rep=1):
    np.random.seed(seed)
    x, y, d, z = make_pliv_CHS2015(n_obs=500, alpha=1.0, dim_x=20, dim_z=1, return_type="array")
    z = np.asarray(z).reshape(-1)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    return fit_case(name, "pliv", data, {"x": x, "y": y, "d": d, "z": z},
                    {"ml_l": OLS, "ml_m": OLS, "ml_r": OLS}, {"score": "partialling out"}, n_rep=n_rep, seed=seed)


CASES = [
    lambda: plr("plr_ols", 11, 500, 10, OLS),
    lambda: plr("plr_ridge", 12, 400, 20, {"kind": "ridge", "alpha": 5.0}),
    lambda: plr("plr_lasso", 13, 400, 30, {"kind": "lasso", "alpha": 0.05}),
    lambda: plr("plr_ivtype_rep3", 14, 400, 10, OLS, score="IV-type", n_rep=3),
    lambda: plr("plr_boot_bayes", 15, 200, 5, OLS, boot=("Bayes", 50, 1)),
    lambda: plr("plr_boot_wild", 16, 200, 5, OLS, boot=("wild", 50, 2)),
    lambda: plr_multi("plr_multi_rep3_boot", 17, 200, n_rep=3, boot=("normal", 40, 3)),
    lambda: irm("irm_ate", 21, "ATE", n_rep=2),
    lambda: irm("irm_atte", 22, "ATTE"),
    lambda: irm("irm_ate_normalized", 23, "ATE", normalize_ipw=True),
    lambda: pliv("pliv_ols", 31, n_rep=2),
    lambda: irm_cate("irm_cate_gate", 41),
    lambda: iivm("iivm_late_rep2", 51, n_rep=2),
    lambda: iivm("iivm_no_always_takers", 52, subgroups={"always_takers": False, "never_takers": True}),
    lambda: did("did_panel_observational", 61, "observational", True),
    lambda: did("did_panel_experimental_nonorm", 62, "experimental", False),
    lambda: did("did_cs_observational_nonorm", 63, "observational", False, cross_section=True),
    lambda: did("did_cs_experimental", 64, "experimental", True, cross_section=True),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for make in CASES:
        case = make()
        case.pop("_model")
        case["versions"] = {"doubleml": dml.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}
        path = OUT / f"{case['name']}.json"
        path.write_text(json.dumps(case))
        print(f"wrote {path.name:28s} coef={np.round(case['expected']['coef'], 6).tolist()}")


if __name__ == "__main__":
    main()

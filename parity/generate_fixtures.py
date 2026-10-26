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
from doubleml.utils.propensity_score_processing import PSProcessorConfig
from doubleml.did.datasets import make_did_CS2021, make_did_cs_CS2021, make_did_SZ2020
from doubleml.irm.datasets import make_iivm_data, make_irm_data, make_irm_data_discrete_treatments, make_ssm_data
from doubleml.plm.datasets import make_lplr_LZZ2020, make_pliv_CHS2015, make_plpr_CP2025, make_pliv_multiway_cluster_CKMS2021, make_plr_CCDDHNR2018
from doubleml.utils._estimation import _draw_weights
from doubleml.utils.gain_statistics import gain_statistics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, LogisticRegression, LogisticRegressionCV, Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "crates" / "dml-core" / "tests" / "fixtures"

OLS = {"kind": "ols"}
PRIOR = {"kind": "prior"}
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
    if kind == "prior":
        # Class shares (count / n): exact on both sides, for routines that are sensitive to the last bit.
        return DummyClassifier(strategy="prior")
    if kind == "lasso_cv":
        return LassoCV(tol=1e-14, max_iter=1_000_000)
    if kind == "logit_cv":
        return LogisticRegressionCV(solver="newton-cholesky", tol=1e-14, max_iter=1000, scoring=spec["scoring"],
                                    l1_ratios=(0,), use_legacy_attributes=False)
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
        "pliv_partial_z": dml.DoubleMLPLIV._partialZ,
        "pliv_partial_xz": dml.DoubleMLPLIV._partialXZ,
        "iivm": dml.DoubleMLIIVM,
        "did": dml.DoubleMLDID,
        "did_cs": dml.DoubleMLDIDCS,
        "apo": dml.DoubleMLAPO,
        "pq": dml.DoubleMLPQ,
        "lpq": dml.DoubleMLLPQ,
        "cvar": dml.DoubleMLCVAR,
        "ssm": dml.DoubleMLSSM,
    }[model_kind]
    opts = dict(options)
    calibration = opts.pop("calibration", None)
    if calibration is not None:
        from doubleml.utils.propensity_score_processing import PSProcessorConfig
        opts["ps_processor_config"] = PSProcessorConfig(calibration_method="isotonic",
                                                        cv_calibration=calibration == "isotonic_cv")
    model = cls(data, **kwargs, n_folds=n_folds, n_rep=n_rep, draw_sample_splitting=False, **opts)
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
        # nuisance_loss[learner] has shape (n_rep, n_coefs): RMSE or log loss
        # Learners without targets (e.g. fold-specific ones) have no loss.
        "nuisance_loss": {k: np.asarray(v).tolist() for k, v in model.nuisance_loss.items()
                          if not np.all(np.isnan(v))},
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
    if model_kind in ("plr", "irm", "did", "did_cs", "apo"):
        expected["sensitivity"] = sensitivity_json(model, benchmark=model_kind in ("plr", "irm", "apo"))
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


def sensitivity_json(model, benchmark):
    """Sensitivity elements, two scenarios (lower and upper bounds) and a benchmark for X1."""
    se = model.sensitivity_elements
    out = {
        "sigma2": np.transpose(se["sigma2"][0]).tolist(),  # (n_coefs, n_rep)
        "nu2": np.transpose(se["nu2"][0]).tolist(),
        "scenarios": [],
    }
    scenarios = [
        {"cf_y": 0.04, "cf_d": 0.03, "rho": 0.8, "level": 0.95, "null_hypothesis": np.zeros(len(model.coef))},
        {"cf_y": 0.03, "cf_d": 0.03, "rho": 1.0, "level": 0.95, "null_hypothesis": 1.5 * np.asarray(model.coef)},
    ]
    for sc in scenarios:
        model.sensitivity_analysis(**sc)
        sp = model.sensitivity_params
        out["scenarios"].append({
            "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sc.items()},
            **{f"{q}_{b}": np.asarray(sp[q][b]).tolist() for q in ("theta", "se", "ci") for b in ("lower", "upper")},
            "rv": np.asarray(sp["rv"]).tolist(),
            "rva": np.asarray(sp["rva"]).tolist(),
        })
    if benchmark:
        bm = model.sensitivity_benchmark(benchmarking_set=["X1"])
        out["benchmark"] = {"drop": [0], **{c: bm[c].tolist() for c in ("cf_y", "cf_d", "rho", "delta_theta")}}
    return out


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


def cal(calibration):
    """Propensity calibration option: None, "isotonic" or "isotonic_cv"."""
    return {} if calibration is None else {"calibration": calibration}


def irm(name, seed, score, normalize_ipw=False, n_rep=1, boot=None, ml_g=OLS, ml_m=LOGIT, dim_x=8, calibration=None):
    np.random.seed(seed)
    x, y, d = make_irm_data(n_obs=500, dim_x=dim_x, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    return fit_case(name, "irm", data, {"x": x, "y": y, "d": d}, {"ml_g": ml_g, "ml_m": ml_m},
                    {"score": score, "normalize_ipw": normalize_ipw, **cal(calibration)}, n_rep=n_rep, seed=seed,
                    boot=boot)


def iivm(name, seed, subgroups=None, n_rep=1, calibration=None):
    np.random.seed(seed)
    x, y, d, z = make_iivm_data(n_obs=600, dim_x=6, theta=0.5, alpha_x=1.0, return_type="array")
    z = np.asarray(z).reshape(-1)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    options = {"score": "LATE", **cal(calibration)}
    if subgroups is not None:
        options["subgroups"] = subgroups
    return fit_case(name, "iivm", data, {"x": x, "y": y, "d": d, "z": z},
                    {"ml_g": OLS, "ml_m": LOGIT, "ml_r": LOGIT}, options, n_rep=n_rep, seed=seed)


def did(name, seed, score, normalize, cross_section=False, calibration=None):
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
                    {"score": score, "in_sample_normalization": normalize, **cal(calibration)}, seed=seed)


def staggered(name, seed, control_group, score, normalize, gt_setting="standard", anticipation=0, n_rep=1,
              calibration=None):
    """DoubleMLDIDMulti on a staggered panel; records each group-time model's own folds."""
    np.random.seed(seed)
    df = make_did_CS2021(n_obs=400, dgp_type=1, include_never_treated=True, time_type="float")
    x_cols = [c for c in df.columns if c.startswith("Z")]
    data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=x_cols)
    learners = {"ml_g": OLS}
    if score == "observational":
        learners["ml_m"] = LOGIT
    model = dml.did.DoubleMLDIDMulti(
        data, ml_g=learner(OLS), ml_m=learner(LOGIT) if score == "observational" else None,
        gt_combinations=gt_setting, control_group=control_group, anticipation_periods=anticipation,
        score=score, in_sample_normalization=normalize, n_folds=5, n_rep=n_rep,
        **({} if calibration is None else {"ps_processor_config": PSProcessorConfig(
            calibration_method="isotonic", cv_calibration=calibration == "isotonic_cv")}),
    )
    model.fit()

    ids = np.sort(df["id"].unique())
    periods = np.sort(df["t"].unique()).astype(float)
    wide = df.pivot(index="id", columns="t", values="y").loc[ids, periods].to_numpy()
    first = df.sort_values(["id", "t"]).groupby("id").first().loc[ids]
    cohort = [None if np.isinf(g) else float(g) for g in first["d"]]

    expected = {
        "gt": [[float(g), float(a), float(b)] for g, a, b in model.gt_combinations],
        "coef": model.coef.tolist(),
        "se": model.se.tolist(),
        "all_coef": model.all_coef.tolist(),
        "all_se": model.all_se.tolist(),
        "aggregations": {},
    }
    for how in ["group", "time", "eventstudy"]:
        agg = model.aggregate(how)
        expected["aggregations"][how] = {
            "names": list(agg.aggregation_names),
            "weights": np.asarray(agg.aggregation_weights).tolist(),
            "overall_weights": np.asarray(agg.overall_aggregation_weights).tolist(),
            "coef": agg.aggregated_frameworks.thetas.tolist(),
            "se": agg.aggregated_frameworks.ses.tolist(),
            "overall_coef": agg.overall_aggregated_framework.thetas.tolist(),
            "overall_se": agg.overall_aggregated_framework.ses.tolist(),
            "sensitivity": framework_sensitivity_json(agg.aggregated_frameworks),
            "overall_sensitivity": framework_sensitivity_json(agg.overall_aggregated_framework),
        }
    expected["sensitivity"] = framework_sensitivity_json(model.framework)
    gt_folds = [
        [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in m.smpls]
        for m in model.modellist
    ]
    # DoubleMLDIDMulti.sensitivity_benchmark refits a deep copy with fewer x_cols, but the group-time
    # models keep the covariates they read at construction, so its benchmark is always 0. Build the
    # short model on data without the covariate instead, on the long model's splits.
    data_short = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=x_cols[1:])
    short = dml.did.DoubleMLDIDMulti(
        data_short, ml_g=learner(OLS), ml_m=learner(LOGIT) if score == "observational" else None,
        gt_combinations=gt_setting, control_group=control_group, anticipation_periods=anticipation,
        score=score, in_sample_normalization=normalize, n_folds=5, n_rep=n_rep,
        **({} if calibration is None else {"ps_processor_config": PSProcessorConfig(
            calibration_method="isotonic", cv_calibration=calibration == "isotonic_cv")}),
    )
    for m_short, m_long in zip(short.modellist, model.modellist):
        m_short.set_sample_splitting(m_long.smpls)
    short.fit()
    bm = gain_statistics(dml_long=model, dml_short=short)
    expected["sensitivity"]["benchmark"] = {"drop": [0], **{c: np.asarray(bm[c]).tolist()
                                                            for c in ("cf_y", "cf_d", "rho", "delta_theta")}}
    return {
        "name": name,
        "model": "staggered",
        "options": {"control_group": control_group, "score": score, "in_sample_normalization": normalize,
                    "gt_combinations": gt_setting, "anticipation_periods": anticipation, **cal(calibration)},
        "learners": learners,
        "y_wide": wide.tolist(),
        "periods": periods.tolist(),
        "cohort": cohort,
        "x": first[x_cols].to_numpy().tolist(),
        "gt_folds": gt_folds,
        "expected": expected,
        "_model": model,
    }


def staggered_cs(name, seed, control_group, score, normalize, n_rep=1):
    """DoubleMLDIDMulti(panel=False) on repeated cross-sections; rows kept in data order."""
    np.random.seed(seed)
    df = make_did_cs_CS2021(n_obs=500, dgp_type=1, include_never_treated=True, time_type="float")
    df = df.sort_values(["id", "t"]).reset_index(drop=True)
    x_cols = [c for c in df.columns if c.startswith("Z")]
    data = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=x_cols)
    learners = {"ml_g": OLS}
    if score == "observational":
        learners["ml_m"] = LOGIT
    model = dml.did.DoubleMLDIDMulti(
        data, ml_g=learner(OLS), ml_m=learner(LOGIT) if score == "observational" else None,
        control_group=control_group, score=score, in_sample_normalization=normalize,
        panel=False, n_folds=5, n_rep=n_rep,
    )
    model.fit()
    expected = {
        "gt": [[float(g), float(a), float(b)] for g, a, b in model.gt_combinations],
        "coef": model.coef.tolist(),
        "se": model.se.tolist(),
        "all_coef": model.all_coef.tolist(),
        "all_se": model.all_se.tolist(),
        "aggregations": {},
    }
    for how in ["group", "time", "eventstudy"]:
        agg = model.aggregate(how)
        expected["aggregations"][how] = {
            "names": list(agg.aggregation_names),
            "weights": np.asarray(agg.aggregation_weights).tolist(),
            "overall_weights": np.asarray(agg.overall_aggregation_weights).tolist(),
            "coef": agg.aggregated_frameworks.thetas.tolist(),
            "se": agg.aggregated_frameworks.ses.tolist(),
            "overall_coef": agg.overall_aggregated_framework.thetas.tolist(),
            "overall_se": agg.overall_aggregated_framework.ses.tolist(),
            "sensitivity": framework_sensitivity_json(agg.aggregated_frameworks),
            "overall_sensitivity": framework_sensitivity_json(agg.overall_aggregated_framework),
        }
    expected["sensitivity"] = framework_sensitivity_json(model.framework)
    gt_folds = [
        [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in m.smpls]
        for m in model.modellist
    ]
    return {
        "name": name,
        "model": "staggered_cs",
        "options": {"control_group": control_group, "score": score, "in_sample_normalization": normalize,
                    "gt_combinations": "standard", "anticipation_periods": 0},
        "learners": learners,
        "y": df["y"].tolist(),
        "time": df["t"].astype(float).tolist(),
        "cohort": [None if np.isinf(g) else float(g) for g in df["d"]],
        "x": df[x_cols].to_numpy().tolist(),
        "gt_folds": gt_folds,
        "expected": expected,
        "_model": model,
    }


def cluster_case(name, model_kind, arrays, clusters, learners, options, n_rep=1, seed=0, sensitivity=False):
    """Clustered data: DoubleML draws its own cluster splits (external splits are not supported
    with clusters), and the fixture records them together with each fold's test clusters."""
    cl = np.column_stack(clusters) if len(clusters) > 1 else clusters[0]
    data = dml.DoubleMLData.from_arrays(arrays["x"], arrays["y"], arrays["d"], z=arrays.get("z"), cluster_vars=cl)
    cls = {"plr": dml.DoubleMLPLR, "irm": dml.DoubleMLIRM, "pliv": dml.DoubleMLPLIV}[model_kind]
    np.random.seed(seed)
    model = cls(data, **{k: learner(v) for k, v in learners.items()}, n_folds=3, n_rep=n_rep, **options)
    model.fit(store_predictions=True)
    expected = {
        "coef": model.coef.tolist(),
        "se": model.se.tolist(),
        "t_stat": model.t_stat.tolist(),
        "pval": model.pval.tolist(),
        "ci95": model.confint(level=0.95).to_numpy().tolist(),
        "all_coef": model.all_coef.tolist(),
        "all_se": model.all_se.tolist(),
        "predictions": {
            k: np.transpose(v, (2, 1, 0)).tolist() for k, v in model.predictions.items() if v is not None
        },
        "nuisance_loss": {k: np.asarray(v).tolist() for k, v in model.nuisance_loss.items()},
        "p_bonferroni": model.p_adjust("bonferroni")["pval"].tolist(),
        "p_holm": model.p_adjust("holm")["pval"].tolist(),
    }
    if sensitivity:
        expected["sensitivity"] = sensitivity_json(model, benchmark=False)
    cluster_folds = [
        [{"train": tr.tolist(), "test": te.tolist(),
          "test_clusters": [np.asarray(c).tolist() for c in sc[1]]}
         for (tr, te), sc in zip(smpls, smpls_cl)]
        for smpls, smpls_cl in zip(model.smpls, model.smpls_cluster)
    ]
    return {
        "name": name,
        "model": model_kind,
        "options": options,
        "learners": learners,
        **{k: np.asarray(v).tolist() for k, v in arrays.items()},
        "clusters": [np.asarray(c).astype(int).tolist() for c in clusters],
        "folds": [[{"train": f["train"], "test": f["test"]} for f in rep] for rep in cluster_folds],
        "cluster_folds": cluster_folds,
        "expected": expected,
        "_model": model,
    }


def plr_cluster_one_way(name, seed):
    np.random.seed(seed)
    x, y, d = make_plr_CCDDHNR2018(n_obs=400, dim_x=5, alpha=0.5, return_type="array")
    cluster = np.random.choice(50, size=400)  # unequal cluster sizes
    return cluster_case(name, "plr", {"x": x, "y": y, "d": d}, [cluster], {"ml_l": OLS, "ml_m": OLS},
                        {"score": "partialling out"}, n_rep=2, seed=seed, sensitivity=True)


def irm_cluster_one_way(name, seed):
    np.random.seed(seed)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    cluster = np.repeat(np.arange(100), 5)
    return cluster_case(name, "irm", {"x": x, "y": y, "d": d}, [cluster], {"ml_g": OLS, "ml_m": LOGIT},
                        {"score": "ATE"}, seed=seed, sensitivity=True)


def multiway(name, seed, model_kind):
    np.random.seed(seed)
    df = make_pliv_multiway_cluster_CKMS2021(N=18, M=15, dim_X=6, theta=1.0, return_type="DataFrame")
    x = df[[c for c in df.columns if c.startswith("X")]].to_numpy()
    arrays = {"x": x, "y": df["Y"].to_numpy(), "d": df["D"].to_numpy()}
    clusters = [df["cluster_var_i"].to_numpy(), df["cluster_var_j"].to_numpy()]
    if model_kind == "pliv":
        arrays["z"] = df["Z"].to_numpy()
        return cluster_case(name, "pliv", arrays, clusters, {"ml_l": OLS, "ml_m": OLS, "ml_r": OLS},
                            {"score": "partialling out"}, seed=seed)
    return cluster_case(name, "plr", arrays, clusters, {"ml_l": OLS, "ml_m": OLS},
                        {"score": "partialling out"}, seed=seed, sensitivity=True)


def discrete_treatment_data(seed):
    r = make_irm_data_discrete_treatments(n_obs=600, n_levels=3, linear=True, random_state=seed)
    return r["x"], r["y"], r["d"]


def apo(name, seed, level, normalize_ipw=False, n_rep=1, calibration=None):
    x, y, d = discrete_treatment_data(seed)
    data = dml.DoubleMLData.from_arrays(x, y, d)
    return fit_case(name, "apo", data, {"x": x, "y": y, "d": d}, {"ml_g": OLS, "ml_m": LOGIT},
                    {"treatment_level": level, "normalize_ipw": normalize_ipw, **cal(calibration)}, n_rep=n_rep,
                    seed=seed)


def framework_json(fw):
    return {
        "names": list(fw.treatment_names) if fw.treatment_names is not None else None,
        "coef": np.asarray(fw.thetas).tolist(),
        "se": np.asarray(fw.ses).tolist(),
        "pval": np.asarray(fw.pvals).tolist(),
        "all_coef": np.asarray(fw.all_thetas).tolist(),
        "all_se": np.asarray(fw.all_ses).tolist(),
        "ci95": fw.confint(level=0.95).to_numpy().tolist(),
    }


def framework_sensitivity_json(fw):
    """Sensitivity scenarios of a framework; sigma2 / nu2 are null where DoubleML drops them (sums)."""
    se = fw.sensitivity_elements
    out = {key: np.asarray(se[key])[0].tolist() if key in se else None for key in ("sigma2", "nu2")}
    out["scenarios"] = []
    k = len(fw.thetas)
    for sc in [{"cf_y": 0.04, "cf_d": 0.03, "rho": 0.8, "level": 0.95, "null_hypothesis": np.zeros(k)},
               {"cf_y": 0.03, "cf_d": 0.03, "rho": 1.0, "level": 0.95, "null_hypothesis": 0.5 * np.asarray(fw.thetas)}]:
        fw.sensitivity_analysis(**sc)
        sp = fw.sensitivity_params
        out["scenarios"].append({
            "params": {k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in sc.items()},
            **{f"{q}_{b}": np.asarray(sp[q][b]).tolist() for q in ("theta", "se", "ci") for b in ("lower", "upper")},
            "rv": np.asarray(sp["rv"]).tolist(),
            "rva": np.asarray(sp["rva"]).tolist(),
        })
    return out


def framework_arithmetic(name, seed):
    """DoubleMLFramework arithmetic on PLR and IRM (ATE) fitted on the same data and sample splits."""
    np.random.seed(seed)
    x, y, d = make_irm_data(n_obs=500, dim_x=5, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    reps = draw_folds(len(y), 5, 2, seed)
    plr = dml.DoubleMLPLR(data, learner(OLS), learner(OLS), n_folds=5, n_rep=2, draw_sample_splitting=False)
    irm = dml.DoubleMLIRM(data, learner(OLS), learner(LOGIT), n_folds=5, n_rep=2, draw_sample_splitting=False)
    for m in (plr, irm):
        m.set_sample_splitting(reps)
        m.fit()
    a, b = plr.framework, irm.framework
    combos = {"sum": a + b, "difference": a - b, "scaled": 2.5 * a,
              "concat": dml.double_ml_framework.concat([a, b])}
    expected = {key: {**framework_json(fw), "sensitivity": framework_sensitivity_json(fw)} for key, fw in combos.items()}
    expected["coef"] = expected["sum"]["coef"]
    # DoubleML 0.11 multiplies the standard errors by a negative scalar too (negative se): only the
    # coefficients and the sensitivity bounds (computed from the scores) are comparable.
    neg = -0.5 * b
    expected["negated"] = {"coef": np.asarray(neg.thetas).tolist(), "abs_se": np.abs(neg.ses).tolist(),
                           "sensitivity": framework_sensitivity_json(neg)}
    return {
        "name": name, "model": "framework", "options": {},
        "learners": {"ml_l": OLS, "ml_m": OLS, "ml_g": OLS, "ml_m_irm": LOGIT},
        "x": x.tolist(), "y": y.tolist(), "d": d.tolist(),
        "folds": [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in reps],
        "expected": expected, "_model": plr,
    }


def apos(name, seed, levels, n_rep=2, normalize_ipw=False):
    x, y, d = discrete_treatment_data(seed)
    data = dml.DoubleMLData.from_arrays(x, y, d)
    reps = draw_folds(len(y), 5, n_rep, seed)
    model = dml.DoubleMLAPOS(data, learner(OLS), learner(LOGIT), treatment_levels=levels, n_folds=5, n_rep=n_rep,
                             normalize_ipw=normalize_ipw, draw_sample_splitting=False)
    model.set_sample_splitting(reps)
    model.fit()
    expected = {
        "coef": np.asarray(model.framework.thetas).tolist(),
        "levels": framework_json(model.framework),
        "levels_sensitivity": framework_sensitivity_json(model.framework),
        "predictions": [
            {k: np.transpose(v, (2, 1, 0))[0].tolist() for k, v in m.predictions.items()} for m in model.modellist
        ],
        "nuisance_loss": [{k: np.asarray(v).tolist() for k, v in m.nuisance_loss.items()} for m in model.modellist],
        "contrasts": [],
    }
    for refs in ([levels[0]], [levels[0], levels[1]]):
        fw = model.causal_contrast(refs)
        expected["contrasts"].append({"reference_levels": refs, **framework_json(fw),
                                      "sensitivity": framework_sensitivity_json(fw)})
    return {
        "name": name, "model": "apos", "options": {"treatment_levels": levels, "normalize_ipw": normalize_ipw},
        "learners": {"ml_g": OLS, "ml_m": LOGIT},
        "x": x.tolist(), "y": y.tolist(), "d": d.tolist(),
        "folds": [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in reps],
        "expected": expected,
        "_model": model,
    }


def quantile_case(name, seed, kind, treatment, quantile, normalize_ipw=True, n_rep=1):
    if kind == "lpq":
        np.random.seed(seed)
        x, y, d, z = make_iivm_data(n_obs=800, dim_x=5, theta=0.5, alpha_x=1.0, return_type="array")
        arrays = {"x": x, "y": y, "d": d, "z": z}
        data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    else:
        np.random.seed(seed)
        x, y, d = make_irm_data(n_obs=600, dim_x=5, theta=0.5, return_type="array")
        arrays = {"x": x, "y": y, "d": d}
        data = dml.DoubleMLData.from_arrays(x, y, d)
    learners = {"ml_g": OLS if kind == "cvar" else LOGIT, "ml_m": PRIOR}
    options = {"treatment": treatment, "quantile": quantile, "normalize_ipw": normalize_ipw}
    return fit_case(name, kind, data, arrays, learners, options, n_rep=n_rep, seed=seed)


def ssm_case(name, seed, score, normalize_ipw=False, n_rep=1, calibration=None):
    mar = score == "missing-at-random"
    np.random.seed(seed)
    x, y, d, z, s = make_ssm_data(n_obs=800, dim_x=5, theta=1.0, mar=mar, return_type="array")
    arrays = {"x": x, "y": y, "d": d, "s": s}
    if mar:
        data = dml.DoubleMLSSMData.from_arrays(x, y, d, s=s)
    else:
        data = dml.DoubleMLSSMData.from_arrays(x, y, d, z=z, s=s)
        arrays["z"] = z
    learners = {"ml_g": OLS, "ml_pi": LOGIT, "ml_m": LOGIT}
    options = {"score": score, "normalize_ipw": normalize_ipw, **cal(calibration)}
    return fit_case(name, "ssm", data, arrays, learners, options, n_rep=n_rep, seed=seed)


def lplr_case(name, seed, score, treatment="continuous", n_rep=1):
    """LPLR draws its own outer and inner (double) sample splits; the fixture stores both."""
    np.random.seed(seed)
    x, y, d, _ = make_lplr_LZZ2020(n_obs=800, dim_x=13, alpha=0.5, return_type="array", treatment=treatment)
    data = dml.DoubleMLData.from_arrays(x, y, d)
    ml_m = OLS if treatment == "continuous" else LOGIT
    learners = {"ml_M": LOGIT, "ml_t": OLS, "ml_m": ml_m}
    np.random.seed(seed + 1)
    model = dml.DoubleMLLPLR(data, learner(LOGIT), learner(OLS), learner(ml_m), n_folds=5, n_folds_inner=5,
                             n_rep=n_rep, score=score)
    model.fit(store_predictions=True)
    expected = {
        "coef": model.coef.tolist(), "se": model.se.tolist(), "t_stat": model.t_stat.tolist(),
        "pval": model.pval.tolist(), "ci95": model.confint(level=0.95).to_numpy().tolist(),
        "all_coef": model.all_coef.tolist(), "all_se": model.all_se.tolist(),
        "predictions": {k: np.transpose(model.predictions[k], (2, 1, 0)).tolist()
                        for k in ("ml_m", "ml_a", "ml_t", "ml_M")},
        "nuisance_loss": {k: np.asarray(model.nuisance_loss[k]).tolist() for k in ("ml_m", "ml_a", "ml_M")},
        "p_bonferroni": model.p_adjust("bonferroni")["pval"].tolist(),
        "p_holm": model.p_adjust("holm")["pval"].tolist(),
    }
    return {
        "name": name, "model": "lplr", "options": {"score": score}, "learners": learners,
        "x": x.tolist(), "y": y.tolist(), "d": np.asarray(d, dtype=float).tolist(),
        "folds": [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in model.smpls],
        "inner_folds": [[[{"train": tr.tolist(), "test": te.tolist()} for tr, te in fold] for fold in rep]
                        for rep in model.smpls_inner],
        "expected": expected, "_model": model,
    }


def plpr_case(name, seed, approach, score="partialling out", n_rep=1, drop=0):
    """Static panel: DoubleML transforms the panel and clusters by unit; the fixture stores the raw
    panel and DoubleML's cluster splits of the transformed rows. `drop` removes random rows
    (gaps and units observed once)."""
    np.random.seed(seed)
    df = make_plpr_CP2025(num_id=80, num_t=5, dim_x=4, theta=0.5, dgp_type="dgp1")
    if drop:
        df = df.drop(index=np.random.choice(len(df), size=drop, replace=False)).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # input order must not matter
    x_cols = [c for c in df.columns if c.startswith("x")]
    data = dml.DoubleMLPanelData(df, "y", "d", "time", "id", x_cols=x_cols, static_panel=True)
    learners = {"ml_l": OLS, "ml_m": OLS}
    kwargs = {"ml_l": learner(OLS), "ml_m": learner(OLS)}
    if score == "IV-type":
        learners["ml_g"] = OLS
        kwargs["ml_g"] = learner(OLS)
    np.random.seed(seed + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = dml.DoubleMLPLPR(data, **kwargs, n_folds=5, n_rep=n_rep, score=score, approach=approach)
        model.fit(store_predictions=True)
    expected = {
        "coef": model.coef.tolist(), "se": model.se.tolist(), "t_stat": model.t_stat.tolist(),
        "pval": model.pval.tolist(), "ci95": model.confint(level=0.95).to_numpy().tolist(),
        "all_coef": model.all_coef.tolist(), "all_se": model.all_se.tolist(),
        "predictions": {k: np.transpose(v, (2, 1, 0)).tolist() for k, v in model.predictions.items() if v is not None},
        "nuisance_loss": {k: np.asarray(v).tolist() for k, v in model.nuisance_loss.items()},
        "p_bonferroni": model.p_adjust("bonferroni")["pval"].tolist(),
        "p_holm": model.p_adjust("holm")["pval"].tolist(),
        "treatment_names": list(model.data_transform.d_cols),
    }
    cluster_folds = [
        [{"train": tr.tolist(), "test": te.tolist(), "test_clusters": [np.asarray(c).tolist() for c in sc[1]]}
         for (tr, te), sc in zip(smpls, smpls_cl)]
        for smpls, smpls_cl in zip(model.smpls, model.smpls_cluster)
    ]
    return {
        "name": name, "model": "plpr", "options": {"score": score, "approach": approach}, "learners": learners,
        "x": df[x_cols].to_numpy().tolist(), "y": df["y"].tolist(), "d": df["d"].tolist(),
        "panel_id": df["id"].astype(int).tolist(), "panel_t": df["time"].astype(float).tolist(),
        "folds": [[{"train": f["train"], "test": f["test"]} for f in rep] for rep in cluster_folds],
        "cluster_folds": cluster_folds, "expected": expected, "_model": model,
    }


def rd_case(name, seed, fuzzy=False, grid=None, manual=(0.3, 0.5)):
    """rdrobust itself (the RD back end of RDFlex): bandwidth selection and estimation."""
    import rdrobust
    rng = np.random.default_rng(seed)
    n = 1000
    x = rng.uniform(-1, 1, n)
    if grid:
        x = np.round(x / grid) * grid
        x[x == 0] = grid  # keep the cutoff inside, not on, the support's mass points
    d = (x >= 0).astype(float)
    if fuzzy:
        flip = rng.random(n) < 0.25
        d = np.where(flip, 1 - d, d)
    y = 1 + 0.8 * x - 0.5 * x**2 + 1.5 * d + rng.normal(0, 0.5, n)
    t = d if fuzzy else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bw = rdrobust.rdbwselect(y=y, x=x, fuzzy=t).bws.values.flatten()
        auto = rdrobust.rdrobust(y=y, x=x, fuzzy=t)
        man = rdrobust.rdrobust(y=y, x=x, fuzzy=t, h=manual[0], b=manual[1])

    def res(r):
        return {"coef": r.coef.values.flatten().tolist(), "se": r.se.values.flatten().tolist(),
                "ci": r.ci.values.tolist(), "h": r.bws.loc["h"].tolist(), "b": r.bws.loc["b"].tolist(),
                "n_h": [int(v) for v in r.N_h]}

    return {"name": name, "model": "rd", "x": x.tolist(), "y": y.tolist(),
            "fuzzy": None if t is None else t.tolist(), "manual": list(manual),
            "expected": {"bws": bw.tolist(), "auto": res(auto), "manual": res(man)}, "_model": None}


def rd_options_case(name, seed):
    """rdrobust's options: polynomial orders, kernels, every bandwidth selector, HC variances,
    mass points, covariates (one collinear) and clusters, in sharp and fuzzy designs."""
    import rdrobust
    rng = np.random.default_rng(seed)
    n = 1000
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=(n, 2))
    cluster = rng.integers(0, 60, n)
    d_sharp = (x >= 0).astype(float)
    flip = rng.random(n) < 0.25
    d_fuzzy = np.where(flip, 1 - d_sharp, d_sharp)
    noise = rng.normal(0, 0.5, n) + 0.3 * rng.normal(size=60)[cluster]
    y_sharp = 1 + 0.8 * x - 0.5 * x**2 + 1.5 * d_sharp + z @ [0.6, -0.4] + noise
    y_fuzzy = 1 + 0.8 * x - 0.5 * x**2 + 1.5 * d_fuzzy + z @ [0.6, -0.4] + noise
    x_grid = np.round(x / 0.02) * 0.02
    x_grid[x_grid == 0] = 0.02
    data = {"sharp": (y_sharp, x, None), "fuzzy": (y_fuzzy, x, d_fuzzy), "grid": (y_sharp, x_grid, None)}
    z_collinear = np.column_stack([z, 2 * z[:, 0] - z[:, 1]])
    cases = [{"kwargs": {"p": 2}}, {"kwargs": {"q": 3}}, {"kwargs": {"kernel": "epa"}},
             {"kwargs": {"kernel": "uni"}, "data": "fuzzy"}]
    cases += [{"kwargs": {"bwselect": sel}} for sel in
              ["msetwo", "msesum", "msecomb1", "msecomb2", "cerrd", "certwo", "cersum", "cercomb1", "cercomb2"]]
    cases += [{"kwargs": {"bwselect": "msetwo", "kernel": "epa"}, "data": "fuzzy"}]
    cases += [{"kwargs": {"vce": v}} for v in ["hc0", "hc1", "hc2", "hc3"]]
    cases += [{"kwargs": {"vce": "hc3"}, "data": "fuzzy"}, {"kwargs": {"nnmatch": 5}},
              {"kwargs": {"level": 90}}, {"kwargs": {"stdvars": False}}, {"kwargs": {"bwrestrict": False}},
              {"kwargs": {"scaleregul": 0}}, {"kwargs": {"rho": 1.5}}, {"kwargs": {"scalepar": 2.0}},
              {"kwargs": {"bwcheck": 25}}, {"kwargs": {"sharpbw": True}, "data": "fuzzy"},
              {"kwargs": {"masspoints": "off"}, "data": "grid"}, {"kwargs": {"masspoints": "check"}, "data": "grid"},
              {"kwargs": {"vce": "hc2"}, "data": "grid"}]
    cases += [{"kwargs": {}, "covs": "z"}, {"kwargs": {}, "covs": "z", "data": "fuzzy"},
              {"kwargs": {"vce": "hc2"}, "covs": "z_collinear"}, {"kwargs": {"bwselect": "msetwo"}, "covs": "z"},
              {"kwargs": {}, "cluster": True}, {"kwargs": {}, "cluster": True, "data": "fuzzy"},
              {"kwargs": {"bwselect": "cerrd"}, "cluster": True}, {"kwargs": {"vce": "hc1"}, "cluster": True, "covs": "z"},
              {"kwargs": {"h": 0.4}, "cluster": True}, {"kwargs": {"h": 0.3, "b": 0.5}, "cluster": True},
              {"kwargs": {"h": 0.3, "rho": 0.6}},
              {"kwargs": {"vce": "cr2"}, "cluster": True}, {"kwargs": {"vce": "cr3"}, "cluster": True},
              {"kwargs": {"vce": "hc2"}, "cluster": True}, {"kwargs": {"vce": "hc3"}, "cluster": True, "data": "fuzzy"},
              {"kwargs": {"vce": "cr2"}, "cluster": True, "covs": "z"},
              {"kwargs": {"vce": "cr3", "bwselect": "cerrd"}, "cluster": True, "covs": "z", "data": "fuzzy"},
              {"kwargs": {"vce": "cr2", "h": 0.4}, "cluster": True}, {"kwargs": {"vce": "cr3", "h": 0.4}, "cluster": True},
              {"kwargs": {"vce": "cr2", "h": 0.3, "b": 0.5}, "cluster": True},
              {"kwargs": {"vce": "cr3", "h": 0.3, "b": 0.5}, "cluster": True, "data": "fuzzy"},
              {"kwargs": {"vce": "cr3", "bwselect": "msetwo", "kernel": "epa"}, "cluster": True},
              {"kwargs": {"vce": "cr2"}}]
    covs_map = {"z": z, "z_collinear": z_collinear}
    out = []
    for case in cases:
        y, xv, t = data[case.get("data", "sharp")]
        kwargs = dict(case["kwargs"])
        if "covs" in case:
            kwargs["covs"] = covs_map[case["covs"]]
        if case.get("cluster"):
            kwargs["cluster"] = cluster
        with warnings.catch_warnings(), __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
            warnings.simplefilter("ignore")
            r = rdrobust.rdrobust(y=y, x=xv, fuzzy=t, **kwargs)
        out.append({**{k: v for k, v in case.items() if k != "kwargs"}, "kwargs": case["kwargs"],
                    "coef": r.coef.values.flatten().tolist(), "se": r.se.values.flatten().tolist(),
                    "ci": r.ci.values.tolist(), "h": r.bws.loc["h"].tolist(), "b": r.bws.loc["b"].tolist()})
    return {"name": name, "model": "rd_options", "x": x.tolist(), "x_grid": x_grid.tolist(),
            "y_sharp": y_sharp.tolist(), "y_fuzzy": y_fuzzy.tolist(), "d_fuzzy": d_fuzzy.tolist(),
            "z": z.tolist(), "z_collinear": z_collinear.tolist(), "cluster": cluster.tolist(),
            "expected": {"coef": out[0]["coef"], "cases": out}, "_model": None}


def rdflex_case(name, seed, fuzzy, n_rep=1, cutoff=0.0, h_fs=None, fs_specification="cutoff",
                fs_kernel="triangular", n_iterations=2, rd_kwargs=None, rd_covs=False, rd_cluster=False,
                binary_outcome=False):
    """RDFlex draws its own (treatment-stratified) splits; the fixture records them. ``rd_kwargs`` are
    passed on to rdrobust, with optional RD covariates (two of the covariates) and cluster labels; a
    binary outcome is learned with a classifier."""
    from doubleml.rdd import RDFlex
    from doubleml.rdd.datasets import make_simple_rdd_data
    np.random.seed(seed)
    data = make_simple_rdd_data(n_obs=1000, fuzzy=fuzzy, cutoff=cutoff)
    x, y, d, score = data["X"], data["Y"], data["D"].astype(float), data["score"]
    if binary_outcome:
        y = (y > np.median(y)).astype(float)
    obj = dml.DoubleMLRDDData.from_arrays(x=x, y=y, d=d, score=score)
    ml_g = LOGIT if binary_outcome else OLS
    learners = {"ml_g": ml_g, **({"ml_m": LOGIT} if fuzzy else {})}
    kwargs = dict(rd_kwargs or {})
    covs = x[:, :2] if rd_covs else None
    cluster = np.random.default_rng(seed).integers(0, 50, len(y)) if rd_cluster else None
    if covs is not None:
        kwargs["covs"] = covs
    if cluster is not None:
        kwargs["cluster"] = cluster
    np.random.seed(seed + 1)
    with warnings.catch_warnings(), __import__("contextlib").redirect_stdout(__import__("io").StringIO()):
        warnings.simplefilter("ignore")
        model = RDFlex(obj, learner(ml_g), learner(LOGIT) if fuzzy else None, fuzzy=fuzzy, cutoff=cutoff, n_rep=n_rep,
                       h_fs=h_fs, fs_specification=fs_specification, fs_kernel=fs_kernel, **kwargs)
        model.fit(n_iterations=n_iterations)
    expected = {
        "coef": model.coef.tolist(), "se": model.se.tolist(), "ci": np.asarray(model.ci).tolist(),
        "all_coef": model.all_coef.tolist(), "all_se": model.all_se.tolist(),
        # DoubleML records no final bandwidth with a single round (it lets rdrobust choose).
        "h": None if np.isnan(model.h).any() else model.h.tolist(),
        "h_fs": float(model.h_fs), "t_stat": model.t_stat.tolist(), "pval": model.pval.tolist(),
        "confint90": model.confint(level=0.9).to_numpy().tolist(),
    }
    options = {"fuzzy": fuzzy, "cutoff": cutoff, "h_fs": h_fs, "fs_specification": fs_specification,
               "fs_kernel": fs_kernel, "n_iterations": n_iterations, "rd_kwargs": rd_kwargs or {},
               "binary_outcome": binary_outcome}
    extra = {}
    if covs is not None:
        extra["rd_covs"] = covs.tolist()
    if cluster is not None:
        extra["rd_cluster"] = cluster.tolist()
    return {
        "name": name, "model": "rdflex", "options": options, "learners": learners, **extra,
        "x": x.tolist(), "y": y.tolist(), "d": d.tolist(), "score": score.tolist(),
        "folds": [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in model._smpls],
        "expected": expected, "_model": model,
    }


def qte_case(name, seed, quantiles, n_rep=2):
    np.random.seed(seed)
    x, y, d = make_irm_data(n_obs=600, dim_x=5, theta=0.5, return_type="array")
    data = dml.DoubleMLData.from_arrays(x, y, d)
    reps = draw_folds(len(y), 5, n_rep, seed)
    model = dml.DoubleMLQTE(data, learner(LOGIT), learner(PRIOR), quantiles=quantiles, n_folds=5, n_rep=n_rep,
                            draw_sample_splitting=False)
    model.set_sample_splitting(reps)
    model.fit()
    fw = model.framework
    expected = {
        "coef": np.asarray(model.coef).tolist(),
        "effects": {
            "coef": np.asarray(fw.thetas).tolist(), "se": np.asarray(fw.ses).tolist(),
            "all_coef": np.asarray(fw.all_thetas).tolist(), "all_se": np.asarray(fw.all_ses).tolist(),
            "ci95": model.confint(level=0.95).to_numpy().tolist(),
        },
        "potential": [
            {"coef": [float(m0.coef[0]), float(m1.coef[0])], "se": [float(m0.se[0]), float(m1.se[0])]}
            for m0, m1 in zip(model.modellist_0, model.modellist_1)
        ],
    }
    return {
        "name": name, "model": "qte", "options": {"quantiles": list(quantiles)},
        "learners": {"ml_g": LOGIT, "ml_m": PRIOR},
        "x": x.tolist(), "y": y.tolist(), "d": d.tolist(),
        "folds": [[{"train": tr.tolist(), "test": te.tolist()} for tr, te in rep] for rep in reps],
        "expected": expected, "_model": model,
    }


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


def pliv(name, seed, n_rep=1, score="partialling out", dim_z=1, kind="pliv"):
    np.random.seed(seed)
    x, y, d, z = make_pliv_CHS2015(n_obs=500, alpha=1.0, dim_x=20, dim_z=dim_z, return_type="array")
    z = np.asarray(z).reshape(-1) if dim_z == 1 else np.asarray(z)
    data = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    learners = {"pliv": {"ml_l": OLS, "ml_m": OLS, "ml_r": OLS}, "pliv_partial_z": {"ml_r": OLS},
                "pliv_partial_xz": {"ml_l": OLS, "ml_m": OLS, "ml_r": OLS}}[kind]
    if score == "IV-type":
        learners["ml_g"] = OLS
    return fit_case(name, kind, data, {"x": x, "y": y, "d": d, "z": z}, learners, {"score": score},
                    n_rep=n_rep, seed=seed)


CASES = [
    lambda: plr("plr_ols", 11, 500, 10, OLS),
    lambda: plr("plr_ridge", 12, 400, 20, {"kind": "ridge", "alpha": 5.0}),
    lambda: plr("plr_lasso", 13, 400, 30, {"kind": "lasso", "alpha": 0.05}),
    lambda: plr("plr_lasso_cv", 14, 300, 40, {"kind": "lasso_cv"}),
    lambda: irm("irm_ate_lasso_cv_logit_cv", 24, "ATE", ml_g={"kind": "lasso_cv"},
                ml_m={"kind": "logit_cv", "scoring": "neg_log_loss"}, dim_x=20),
    lambda: plr("plr_ivtype_rep3", 14, 400, 10, OLS, score="IV-type", n_rep=3),
    lambda: plr("plr_boot_bayes", 15, 200, 5, OLS, boot=("Bayes", 50, 1)),
    lambda: plr("plr_boot_wild", 16, 200, 5, OLS, boot=("wild", 50, 2)),
    lambda: plr_multi("plr_multi_rep3_boot", 17, 200, n_rep=3, boot=("normal", 40, 3)),
    lambda: irm("irm_ate", 21, "ATE", n_rep=2),
    lambda: irm("irm_ate_isotonic_normalized", 25, "ATE", normalize_ipw=True, calibration="isotonic"),
    lambda: irm("irm_atte_isotonic_cv_rep2", 26, "ATTE", n_rep=2, calibration="isotonic_cv"),
    lambda: apo("apo_isotonic_cv", 34, 2, calibration="isotonic_cv"),
    lambda: iivm("iivm_isotonic", 53, calibration="isotonic"),
    lambda: staggered("staggered_isotonic_cv", 74, "never_treated", "observational", True, calibration="isotonic_cv"),
    lambda: ssm_case("ssm_mar_isotonic_cv", 77, "missing-at-random", calibration="isotonic_cv"),
    lambda: irm("irm_atte", 22, "ATTE"),
    lambda: irm("irm_ate_normalized", 23, "ATE", normalize_ipw=True),
    lambda: pliv("pliv_ols", 31, n_rep=2),
    lambda: pliv("pliv_ivtype", 32, score="IV-type"),
    lambda: pliv("pliv_two_instruments_rep2", 33, dim_z=2, n_rep=2),
    lambda: pliv("pliv_three_instruments", 34, dim_z=3),
    lambda: pliv("pliv_partial_z", 35, dim_z=2, kind="pliv_partial_z"),
    lambda: pliv("pliv_partial_xz", 36, dim_z=2, kind="pliv_partial_xz"),
    lambda: irm_cate("irm_cate_gate", 41),
    lambda: iivm("iivm_late_rep2", 51, n_rep=2),
    lambda: iivm("iivm_no_always_takers", 52, subgroups={"always_takers": False, "never_takers": True}),
    lambda: did("did_panel_observational", 61, "observational", True),
    lambda: did("did_panel_experimental_nonorm", 62, "experimental", False),
    lambda: did("did_cs_observational_nonorm", 63, "observational", False, cross_section=True),
    lambda: did("did_cs_experimental", 64, "experimental", True, cross_section=True),
    lambda: staggered("staggered_never_observational", 71, "never_treated", "observational", True),
    lambda: staggered("staggered_notyet_experimental_rep2", 72, "not_yet_treated", "experimental", False, n_rep=2),
    lambda: staggered("staggered_universal_anticipation", 73, "never_treated", "observational", True,
                      gt_setting="universal", anticipation=1),
    lambda: quantile_case("pq_t1_rep2", 61, "pq", 1, 0.5, n_rep=2),
    lambda: quantile_case("pq_t0_q25", 62, "pq", 0, 0.25),
    lambda: quantile_case("pq_t1_q75_nonorm", 63, "pq", 1, 0.75, normalize_ipw=False),
    lambda: quantile_case("cvar_t1", 64, "cvar", 1, 0.5),
    lambda: quantile_case("cvar_t0_q25", 65, "cvar", 0, 0.25),
    lambda: quantile_case("lpq_t1", 66, "lpq", 1, 0.5),
    lambda: qte_case("qte_rep2", 67, [0.25, 0.5, 0.75]),
    lambda: rd_case("rd_sharp", 91),
    lambda: rdflex_case("rdflex_sharp", 94, fuzzy=False),
    lambda: rdflex_case("rdflex_fuzzy_rep2", 95, fuzzy=True, n_rep=2),
    lambda: rdflex_case("rdflex_fuzzy_interacted_epa", 96, fuzzy=True, cutoff=0.2,
                        fs_specification="interacted cutoff and score", fs_kernel="epanechnikov", n_iterations=3),
    lambda: rdflex_case("rdflex_sharp_score_uniform", 97, fuzzy=False, h_fs=0.6,
                        fs_specification="cutoff and score", fs_kernel="uniform", n_iterations=1),
    lambda: rd_case("rd_fuzzy", 92, fuzzy=True),
    lambda: rd_case("rd_masspoints", 93, grid=0.02),
    lambda: rd_options_case("rd_options", 98),
    lambda: rdflex_case("rdflex_sharp_rd_options", 99, fuzzy=False,
                        rd_kwargs={"p": 2, "kernel": "epa", "bwselect": "msetwo", "vce": "hc1"}),
    lambda: rdflex_case("rdflex_fuzzy_covs_cluster", 100, fuzzy=True, rd_covs=True, rd_cluster=True,
                        rd_kwargs={"bwselect": "cerrd"}),
    lambda: rdflex_case("rdflex_binary_outcome", 101, fuzzy=False, binary_outcome=True),
    lambda: rdflex_case("rdflex_fuzzy_binary_rho", 102, fuzzy=True, binary_outcome=True, n_rep=2,
                        rd_kwargs={"rho": 1.25, "vce": "hc3", "masspoints": "off"}),
    lambda: plpr_case("plpr_fd_exact_rep2", 79, "fd_exact", n_rep=2),
    lambda: plpr_case("plpr_fd_exact_unbalanced", 80, "fd_exact", drop=25),
    lambda: plpr_case("plpr_wg_approx_unbalanced", 83, "wg_approx", drop=25),
    lambda: plpr_case("plpr_cre_general", 84, "cre_general"),
    lambda: plpr_case("plpr_cre_normal_ivtype", 85, "cre_normal", score="IV-type"),
    lambda: lplr_case("lplr_nuisance_space_rep2", 75, "nuisance_space", n_rep=2),
    lambda: lplr_case("lplr_instrument", 76, "instrument"),
    lambda: lplr_case("lplr_binary_propensity", 77, "nuisance_space", treatment="binary"),
    lambda: lplr_case("lplr_binary_instrument", 78, "instrument", treatment="binary"),
    lambda: ssm_case("ssm_mar_rep2", 71, "missing-at-random", n_rep=2),
    lambda: ssm_case("ssm_mar_normalized", 72, "missing-at-random", normalize_ipw=True),
    lambda: ssm_case("ssm_nonignorable", 73, "nonignorable"),
    lambda: ssm_case("ssm_nonignorable_normalized_rep2", 74, "nonignorable", normalize_ipw=True, n_rep=2),
    lambda: apo("apo_level1_rep2", 31, 1, n_rep=2),
    lambda: apo("apo_level0_normalized", 32, 0, normalize_ipw=True),
    lambda: apos("apos_rep2", 33, [0, 1, 2]),
    lambda: framework_arithmetic("framework_arithmetic_rep2", 35),
    lambda: plr_cluster_one_way("plr_cluster_one_way_rep2", 91),
    lambda: irm_cluster_one_way("irm_cluster_one_way", 92),
    lambda: multiway("pliv_cluster_two_way", 93, "pliv"),
    lambda: multiway("plr_cluster_two_way", 94, "plr"),
    lambda: staggered_cs("staggered_cs_never_observational", 81, "never_treated", "observational", True),
    lambda: staggered_cs("staggered_cs_notyet_experimental_rep2", 82, "not_yet_treated", "experimental", False,
                         n_rep=2),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for make in CASES:
        case = make()
        case.pop("_model")
        case["versions"] = {"doubleml": dml.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__}
        path = OUT / f"{case['name']}.json"
        path.write_text(json.dumps(case))
        coef = case["expected"].get("coef", case["expected"].get("auto", {}).get("coef"))
        print(f"wrote {path.name:28s} coef={np.round(coef, 6).tolist()}")


if __name__ == "__main__":
    main()

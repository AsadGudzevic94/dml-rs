"""Every DoubleML model against dml-rs: DoubleML 0.11.4 with scikit-learn's linear learners,
dml-rs with its built-in Ols / LogisticRegression, same data, 5 folds. Prints one line per case;
the table is in docs/benchmarks.md.

    python parity/bench_models.py 5000,100000 [name filter ...]
"""
import sys
import time
import warnings

import doubleml as dml
import numpy as np
from doubleml.did.datasets import make_did_CS2021, make_did_SZ2020
from doubleml.irm.datasets import make_iivm_data, make_irm_data, make_irm_data_discrete_treatments, make_ssm_data
from doubleml.plm.datasets import make_lplr_LZZ2020, make_plpr_CP2025
from doubleml.rdd import RDFlex as DoubleMLRDFlex
from doubleml.rdd.datasets import make_simple_rdd_data
from sklearn.linear_model import LinearRegression, LogisticRegression

import dml_rs

warnings.simplefilter("ignore")
OLS, LOGIT = dml_rs.Ols(), dml_rs.LogisticRegression()
SKO, SKL = LinearRegression(), LogisticRegression()


def best(f, reps=2):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        out.append(time.perf_counter() - t)
    return min(out)


def cases(n):
    np.random.seed(0)
    x, y, d = make_irm_data(n_obs=n, dim_x=10, theta=0.5, return_type="array")
    irm = dml.DoubleMLData.from_arrays(x, y, d)
    yield "IRM", lambda: dml.DoubleMLIRM(irm, SKO, SKL).fit(), lambda: dml_rs.IRM(OLS, LOGIT).fit(x, y, d)
    fit_r, fit_o = dml.DoubleMLIRM(irm, SKO, SKL).fit(), dml_rs.IRM(OLS, LOGIT).fit(x, y, d)
    yield ("IRM sensitivity", lambda: fit_r.sensitivity_analysis(cf_y=0.03, cf_d=0.03),
           lambda: fit_o.sensitivity_analysis(cf_y=0.03, cf_d=0.03))
    yield ("PQ", lambda: dml.DoubleMLPQ(irm, SKL, SKL, quantile=0.5).fit(),
           lambda: dml_rs.PQ(LOGIT, LOGIT, quantile=0.5).fit(x, y, d))
    yield ("CVaR", lambda: dml.DoubleMLCVAR(irm, SKO, SKL, quantile=0.5).fit(),
           lambda: dml_rs.CVAR(OLS, LOGIT, quantile=0.5).fit(x, y, d))
    q = [0.25, 0.5, 0.75]
    yield ("QTE (3 quantiles)", lambda: dml.DoubleMLQTE(irm, SKL, SKL, quantiles=q).fit(),
           lambda: dml_rs.QTE(LOGIT, LOGIT, quantiles=q).fit(x, y, d))

    x, y, d, z = make_iivm_data(n_obs=n, dim_x=10, theta=0.5, alpha_x=1.0, return_type="array")
    z = np.asarray(z).ravel()
    iv = dml.DoubleMLData.from_arrays(x, y, d, z=z)
    yield ("IIVM", lambda: dml.DoubleMLIIVM(iv, SKO, SKL, SKL).fit(),
           lambda: dml_rs.IIVM(OLS, LOGIT, LOGIT).fit(x, y, d, z))
    yield ("LPQ", lambda: dml.DoubleMLLPQ(iv, SKL, SKL, quantile=0.5).fit(),
           lambda: dml_rs.LPQ(LOGIT, LOGIT, quantile=0.5).fit(x, y, d, z))

    r = make_irm_data_discrete_treatments(n_obs=n, n_levels=3, linear=True, random_state=1)
    x, y, d = r["x"], r["y"], r["d"]
    apos = dml.DoubleMLData.from_arrays(x, y, d)
    yield ("APOS (4 levels)", lambda: dml.DoubleMLAPOS(apos, SKO, SKL, treatment_levels=[0, 1, 2, 3]).fit(),
           lambda: dml_rs.APOS(OLS, LOGIT, treatment_levels=[0, 1, 2, 3]).fit(x, y, d))

    for mar in (True, False):
        x, y, d, z, s = make_ssm_data(n_obs=n, dim_x=10, theta=1.0, mar=mar, return_type="array")
        score = "missing-at-random" if mar else "nonignorable"
        data = dml.DoubleMLSSMData.from_arrays(x, y, d, z=None if mar else z, s=s)
        yield (f"SSM {score}", lambda: dml.DoubleMLSSM(data, SKO, SKL, SKL, score=score).fit(),
               lambda: dml_rs.SSM(OLS, LOGIT, LOGIT, score=score).fit(x, y, d, s, z=None if mar else z))

    x, y, d, _ = make_did_SZ2020(n_obs=n, cross_sectional_data=False, return_type="array")
    did = dml.DoubleMLDIDData.from_arrays(x, y, d)
    yield "DID", lambda: dml.DoubleMLDID(did, SKO, SKL).fit(), lambda: dml_rs.DID(OLS, LOGIT).fit(x, y, d)
    x, y, d, t = make_did_SZ2020(n_obs=n, cross_sectional_data=True, return_type="array")
    cs = dml.DoubleMLDIDData.from_arrays(x, y, d, t=t)
    yield "DIDCS", lambda: dml.DoubleMLDIDCS(cs, SKO, SKL).fit(), lambda: dml_rs.DIDCS(OLS, LOGIT).fit(x, y, d, t)

    df = make_did_CS2021(n_obs=n // 4, dgp_type=1, include_never_treated=True, time_type="float")
    x_cols = [c for c in df.columns if c.startswith("Z")]
    panel = dml.data.DoubleMLPanelData(df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=x_cols)
    yield (f"Staggered DID ({n // 4} units)",
           lambda: dml.did.DoubleMLDIDMulti(panel, SKO, SKL, control_group="not_yet_treated").fit(),
           lambda: dml_rs.StaggeredDID(OLS, LOGIT, control_group="not_yet_treated").fit_frame(
               df, "y", "id", "t", "d", x_cols))

    x, y, d, _ = make_lplr_LZZ2020(n_obs=n, dim_x=20, alpha=0.5, return_type="array", treatment="continuous")
    lp = dml.DoubleMLData.from_arrays(x, y, d.astype(float))
    yield ("LPLR", lambda: dml.DoubleMLLPLR(lp, ml_M=SKL, ml_t=SKO, ml_m=SKO).fit(),
           lambda: dml_rs.LPLR(LOGIT, OLS, OLS).fit(x, y, d.astype(float)))

    df = make_plpr_CP2025(num_id=n // 10, num_t=10, dim_x=10, theta=0.5, dgp_type="dgp1")
    xc = [c for c in df.columns if c.startswith("x")]
    pp = dml.DoubleMLPanelData(df, "y", "d", "time", "id", x_cols=xc, static_panel=True)
    yield ("PLPR (cre_general)", lambda: dml.DoubleMLPLPR(pp, SKO, SKO, n_folds=5).fit(),
           lambda: dml_rs.PLPR(OLS, OLS).fit(df[xc].to_numpy(), df["y"], df[["d"]], df["id"], df["time"]))

    np.random.seed(0)
    rd = make_simple_rdd_data(n_obs=n, fuzzy=False)
    rdo = dml.DoubleMLRDDData.from_arrays(x=rd["X"], y=rd["Y"], d=rd["D"].astype(float), score=rd["score"])
    yield ("RDFlex (sharp)", lambda: DoubleMLRDFlex(rdo, SKO).fit(n_iterations=2),
           lambda: dml_rs.RDFlex(OLS).fit(rd["X"], rd["Y"], rd["D"].astype(float), rd["score"]))


if __name__ == "__main__":
    only = sys.argv[2:]
    for n in [int(v) for v in sys.argv[1].split(",")]:
        for name, theirs, ours in cases(n):
            if only and not any(o in name for o in only):
                continue
            try:
                a = best(ours)
                b = best(theirs, reps=1 if n > 20_000 else 2)
                print(f"n={n:>6} {name:30s} dml-rs {a:8.3f}s  DoubleML {b:8.3f}s  {b / a:6.1f}x", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"n={n:>6} {name:30s} ERROR {type(e).__name__}: {e}", flush=True)

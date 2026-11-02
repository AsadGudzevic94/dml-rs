"""Fixtures for the R package's parity tests (r/dmlrs/tests/testthat/test-parity.R).

The R package and the Python package dml-rs call the same Rust code, so on the same data they
must give the same estimates. The data are written as raw little-endian doubles (exact), the
Python results as text with 17 significant digits.

    .venv/bin/python parity/generate_r_fixtures.py
"""

from pathlib import Path

import numpy as np

import dml_rs

OUT = Path(__file__).resolve().parent.parent / "r" / "dmlrs" / "tests" / "testthat" / "fixtures"


def main():
    rng = np.random.default_rng(2026)
    n = 400
    x = rng.normal(size=(n, 4))
    d = (rng.random(n) < 1 / (1 + np.exp(-x[:, 0]))).astype(float)
    z = (rng.random(n) < 0.5).astype(float)
    dcont = x[:, 0] + 0.5 * x[:, 1] + z + rng.normal(size=n)
    d_iv = np.where(rng.random(n) < 0.7, z, (rng.random(n) < 0.5).astype(float))
    y = 0.5 * dcont + d + d_iv + x[:, 1] - 0.5 * x[:, 2] + rng.normal(size=n)
    ydid = d + 0.5 * x[:, 0] + rng.normal(size=n)
    # Drawn after the columns above, so that their values do not change.
    m = 0.5 * d + x[:, 0] + rng.normal(size=n)
    ymed = d + m + x[:, 1] + rng.normal(size=n)
    dose = np.where(rng.random(n) < 0.4, 0.0, rng.uniform(0.5, 2.0, size=n))
    ycd = dose * (1 + 0.5 * dose) + 0.3 * x[:, 0] + rng.normal(size=n)
    u = rng.random(n)
    p0, p1 = 1 / (1 + np.exp(x[:, 0])), 0.5 * np.ones(n)
    d3 = np.where(u < p0 * 0.6, 0.0, np.where(u < p0 * 0.6 + (1 - p0 * 0.6) * 0.5, 1.0, 2.0))
    y3 = d3 * x[:, 1] + x[:, 0] + rng.normal(size=n)
    s = (x[:, 0] + d + rng.normal(size=n) > 0).astype(float)
    ylee = np.where(s == 1, y, np.nan)
    strata = np.digitize(x[:, 1], [-0.5, 0.5]).astype(float)
    columns = [*x.T, y, d, dcont, z, d_iv, ydid, m, ymed, dose, ycd, d3, y3, s, ylee, strata]
    OUT.mkdir(parents=True, exist_ok=True)
    np.concatenate(columns).astype("<f8").tofile(OUT / "parity_data.bin")  # column by column

    ols, logit = dml_rs.Ols(), dml_rs.LogisticRegression()
    expected = {}

    plr = dml_rs.PLR(dml_rs.LassoCV(), dml_rs.LassoCV(), n_rep=2).fit(x, y, dcont)
    pointwise = np.asarray(plr.confint())[0]
    plr.bootstrap(method="normal", n_boot=500, random_state=0)
    joint = np.asarray(plr.confint(joint=True))[0]
    expected["plr"] = [plr.coef_[0], plr.se_[0], *pointwise, *joint]
    sens = plr.sensitivity_analysis(cf_y=0.04, cf_d=0.03).sensitivity_params_
    expected["plr_sensitivity"] = [sens["theta"]["lower"][0], sens["theta"]["upper"][0],
                                   sens["ci"]["lower"][0], sens["ci"]["upper"][0], sens["rv"][0], sens["rva"][0]]

    irm = dml_rs.IRM(dml_rs.RandomForestRegressor(n_estimators=30), logit).fit(x, y, d)
    expected["irm"] = [irm.coef_[0], irm.se_[0]]
    pliv = dml_rs.PLIV(ols, ols, ols).fit(x, y, dcont, z)
    expected["pliv"] = [pliv.coef_[0], pliv.se_[0]]
    iivm = dml_rs.IIVM(ols, logit, logit).fit(x, y, d_iv, z)
    expected["iivm"] = [iivm.coef_[0], iivm.se_[0]]
    did = dml_rs.DID(ols, logit).fit(x, ydid, d)
    expected["did"] = [did.coef_[0], did.se_[0]]
    boosting = dml_rs.PLR(dml_rs.HistGradientBoostingRegressor(max_iter=30), ols, score="IV-type",
                          ml_g=dml_rs.GradientBoostingRegressor(n_estimators=30)).fit(x, y, dcont)
    expected["plr_boosting"] = [boosting.coef_[0], boosting.se_[0]]

    apos = dml_rs.APOS(ols, logit, treatment_levels=[0, 1, 2]).fit(x, y3, d3)
    contrast = apos.causal_contrast(0)
    expected["apos"] = [*apos.coef_, *apos.se_, *contrast.coef_, *contrast.se_]
    gamma = dml_rs.double_robust_scores(apos)
    tree = dml_rs.OptimalPolicyTree(depth=2, min_node_size=20).fit(x, gamma)
    actions = tree.predict(x)
    value = dml_rs.policy_value(gamma, tree=actions, treat_all=np.full(n, 2))
    expected["policy_tree"] = [float(a + 1) for a in actions]  # R's actions are 1-based
    expected["policy_value"] = [*value.coef_, *value.se_]

    med = dml_rs.Mediation(ols, logit).fit(x, ymed, d, m)
    expected["mediation"] = [*med.coef_, *med.se_, *med.potential_outcomes().coef_]

    cd = dml_rs.ContinuousDID().fit(None, ycd, dose)
    expected["contdid"] = [*cd.coef_, *cd.se_, *cd.overall().coef_, *cd.overall().se_]
    cd_x = dml_rs.ContinuousDID(ols, logit).fit(x, ycd, dose)
    expected["contdid_covariates"] = [*cd_x.overall().coef_, *cd_x.overall().se_]

    for name, cells in [("lee", None), ("lee_strata", strata)]:
        lee = dml_rs.LeeBounds(n_boot=200, random_state=3).fit(ylee, d, s, strata=cells)
        expected[name] = [lee.lower_, lee.upper_, *lee.se_, *lee.confint_]

    # Synthetic DiD: 30 units (the last 3 treated) x 15 periods (the last 5 treated).
    units, periods, n0, t0 = 30, 15, 27, 10
    unit_fe = rng.normal(size=(units, 1))
    trend = np.cumsum(rng.normal(size=(1, periods)), axis=1)
    loading = rng.normal(size=(units, 1))
    panel = unit_fe + loading * trend + 0.3 * rng.normal(size=(units, periods))
    panel[n0:, t0:] += 1.0
    panel.T.astype("<f8").tofile(OUT / "parity_synth.bin")  # column-major, as R reads it
    sdid = dml_rs.SyntheticDID(se_method=None).fit(panel, n0, t0)
    expected["synthdid"] = [sdid.coef_[0], sdid.se("placebo", replications=50, random_state=1),
                            sdid.se("jackknife"), sdid.se("bootstrap", replications=50, random_state=2),
                            *sdid.effect_curve_]
    sc = dml_rs.SyntheticDID(estimator="sc", se_method=None).fit(panel, n0, t0)
    expected["synthdid_sc"] = [sc.coef_[0], *sc.unit_weights_]

    # Heterogeneous effects and dose response, on the columns above (no new draws).
    basis = np.column_stack([np.ones(n), x[:, 1]])
    groups = np.column_stack([x[:, 1] <= 0, x[:, 1] > 0]).astype(float)
    irm_ols = dml_rs.IRM(ols, logit).fit(x, y, d)
    plr_ols = dml_rs.PLR(ols, ols).fit(x, y, dcont)
    for name, model in [("irm", irm_ols), ("plr", plr_ols)]:
        blp, gate = model.cate(basis), model.gate(groups)
        effect, se = blp.predict(basis[:3])
        expected[f"{name}_cate"] = [*blp.coef, *blp.se, *effect, *se]
        expected[f"{name}_gate"] = [*gate.coef, *gate.se]
    for method, learner in [("S", dml_rs.SLearner(ols)), ("T", dml_rs.TLearner(ols)),
                            ("X", dml_rs.XLearner(ols, ols, logit)),
                            ("R", dml_rs.RLearner(ols, logit, ols)),
                            ("DR", dml_rs.DRLearner(ols, logit, ols))]:
        expected[f"cate_{method}"] = list(learner.fit(x, y, d).effect(x[:5]))
    dr = dml_rs.DRLearner(ols, logit, ols).fit(x, y, d)
    expected["cate_DR_blp"] = [*dr.blp(basis).coef, *dr.blp(basis).se]
    # OLS nuisances: forest splits flip on last-bit differences, which the logistic regression
    # has between compilers (the R package on Windows is built with GCC, the wheels with MSVC).
    cf = dml_rs.CausalForest(ols, ols, n_estimators=200).fit(x, y, d)
    ate = cf.ate()
    expected["causal_forest"] = [*cf.effect_[:5], *cf.effect(x[:3]), *cf.effect_se(x[:3]),
                                 ate.coef_[0], ate.se_[0], *cf.blp(basis).coef]
    ivf = dml_rs.InstrumentalForest(ols, ols, ols, n_estimators=200).fit(x, y, d_iv, z)
    expected["instrumental_forest"] = [*ivf.effect_[:5], *ivf.effect(x[:3]), ivf.ate().coef_[0],
                                       ivf.ate().se_[0]]
    conf = dml_rs.ConformalITE(ml_m=logit, ml_y=ols).fit(x, y, d)
    expected["conformal"] = [*conf.predict_ite(x[:3], 0.1).ravel(),
                             *conf.predict_counterfactual(x[:3], 0, alpha=0.2, target="treated").ravel()]
    dose_fit = dml_rs.DoseResponse(ols, ols, grid=[-1.0, 0.0, 1.0], partial_effects=True).fit(x, y, dcont)
    contrast_dose = dose_fit.contrast(0.0)
    expected["dose_response"] = [*dose_fit.coef_, *dose_fit.se_, *dose_fit.partial_effect().coef_,
                                 *contrast_dose.coef_, *contrast_dose.se_]

    with open(OUT / "parity_expected.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("name,index,value\n")
        for name, values in expected.items():
            for i, v in enumerate(values):
                f.write(f"{name},{i + 1},{float(v)!r}\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

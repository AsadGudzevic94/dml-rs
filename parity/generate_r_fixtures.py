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
    columns = [*x.T, y, d, dcont, z, d_iv, ydid]
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

    with open(OUT / "parity_expected.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write("name,index,value\n")
        for name, values in expected.items():
            for i, v in enumerate(values):
                f.write(f"{name},{i + 1},{float(v)!r}\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

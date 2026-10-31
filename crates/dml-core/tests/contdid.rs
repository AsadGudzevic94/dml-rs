//! Parity with R's `contdid` (two periods, no covariates, B-spline dose
//! response). The fixture comes from `parity/generate_contdid_fixtures.R`;
//! standard errors there are the analytic ones from contdid's influence
//! functions (for the overall ACRT rescaled from n to the n1 treated units,
//! the scale of that influence function; contdid itself reports a standard
//! error too small by n1 / n).

use dml_core::{ContinuousDid, DmlData};
use ndarray::{Array1, Array2};
use serde_json::Value;

const TOL: f64 = 1e-9;

fn vector(v: &Value) -> Array1<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_f64().unwrap())
        .collect()
}

fn close(name: &str, what: &str, ours: &[f64], theirs: &Array1<f64>) {
    assert_eq!(ours.len(), theirs.len(), "{name} {what}");
    for (j, (a, b)) in ours.iter().zip(theirs).enumerate() {
        assert!((a - b).abs() < TOL, "{name} {what}[{j}]: {a} vs {b}");
    }
}

#[test]
fn matches_contdid() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/contdid_r.json");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut checked = 0;
    for (name, case) in fixture.as_object().unwrap() {
        if name.starts_with('_') {
            continue;
        }
        let y = vector(&case["y"]);
        let n = y.len();
        let data = DmlData::new(y, vector(&case["d"]), Array2::zeros((n, 0))).unwrap();
        let fit = ContinuousDid::new()
            .degree(case["degree"].as_u64().unwrap() as usize)
            .num_knots(case["num_knots"].as_u64().unwrap() as usize)
            .grid(vector(&case["grid"]).to_vec())
            .fit(&data)
            .unwrap();
        close(
            name,
            "att",
            fit.att().coef().as_slice().unwrap(),
            &vector(&case["att"]),
        );
        close(
            name,
            "att se",
            fit.att().se().as_slice().unwrap(),
            &vector(&case["att_se"]),
        );
        close(
            name,
            "acrt",
            fit.acrt().coef().as_slice().unwrap(),
            &vector(&case["acrt"]),
        );
        close(
            name,
            "acrt se",
            fit.acrt().se().as_slice().unwrap(),
            &vector(&case["acrt_se"]),
        );
        let overall = fit.overall();
        let theirs = Array1::from(vec![
            case["overall_att"].as_f64().unwrap(),
            case["overall_acrt"].as_f64().unwrap(),
        ]);
        close(name, "overall", overall.coef().as_slice().unwrap(), &theirs);
        let theirs_se = Array1::from(vec![
            case["overall_att_se"].as_f64().unwrap(),
            case["overall_acrt_se"].as_f64().unwrap(),
        ]);
        close(
            name,
            "overall se",
            overall.se().as_slice().unwrap(),
            &theirs_se,
        );
        checked += 1;
    }
    assert_eq!(checked, 3);
}

//! Parity with R's `synthdid` (estimates, weights, covariate coefficients,
//! effect curves, jackknife, placebo and bootstrap standard errors). The
//! fixture comes from `parity/generate_synthdid_fixtures.R`.

use dml_core::{SynthEstimator, SynthPanel, SynthSe, SyntheticDid};
use ndarray::{Array1, Array2, Array3};
use serde_json::Value;

const TOL: f64 = 1e-9;

fn vector(v: &Value) -> Array1<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_f64().unwrap())
        .collect()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Array1<f64>> = v.as_array().unwrap().iter().map(vector).collect();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn close(name: &str, what: &str, ours: &[f64], theirs: &[f64]) {
    assert_eq!(ours.len(), theirs.len(), "{name} {what} length");
    for (j, (a, b)) in ours.iter().zip(theirs).enumerate() {
        assert!(
            (a - b).abs() < TOL,
            "{name} {what}[{j}]: {a} vs {b} (diff {:e})",
            a - b
        );
    }
}

#[test]
fn matches_synthdid() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synthdid_r.json"
    );
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut checked = 0;
    for (name, case) in fixture.as_object().unwrap() {
        if name.starts_with('_') {
            continue;
        }
        let y = matrix(&case["y"]);
        let (n, t) = y.dim();
        let mut panel = SynthPanel::new(
            y,
            case["n0"].as_u64().unwrap() as usize,
            case["t0"].as_u64().unwrap() as usize,
        )
        .unwrap();
        if let Some(xs) = case.get("x").and_then(Value::as_array) {
            let mut x = Array3::<f64>::zeros((n, t, xs.len()));
            for (k, xk) in xs.iter().enumerate() {
                x.index_axis_mut(ndarray::Axis(2), k).assign(&matrix(xk));
            }
            panel = panel.with_covariates(x).unwrap();
        }
        let estimator = match case["estimator"].as_str().unwrap() {
            "sdid" => SynthEstimator::SyntheticDid,
            "sc" => SynthEstimator::SyntheticControl,
            "did" => SynthEstimator::Did,
            e => panic!("unknown estimator {e}"),
        };
        let fit = SyntheticDid::new()
            .estimator(estimator)
            .fit(&panel)
            .unwrap();
        close(
            name,
            "estimate",
            &[fit.estimate()],
            &[case["estimate"].as_f64().unwrap()],
        );
        close(
            name,
            "omega",
            fit.unit_weights(),
            vector(&case["omega"]).as_slice().unwrap(),
        );
        close(
            name,
            "lambda",
            fit.time_weights(),
            vector(&case["lambda"]).as_slice().unwrap(),
        );
        close(
            name,
            "beta",
            fit.beta(),
            vector(&case["beta"]).as_slice().unwrap(),
        );
        close(
            name,
            "curve",
            fit.effect_curve().as_slice().unwrap(),
            vector(&case["effect_curve"]).as_slice().unwrap(),
        );
        match case["se_jackknife"].as_f64() {
            Some(se) => close(
                name,
                "jackknife",
                &[fit.se(SynthSe::Jackknife).unwrap()],
                &[se],
            ),
            None => assert!(
                fit.se(SynthSe::Jackknife).is_err(),
                "{name}: jackknife should be unavailable"
            ),
        }
        for (key, kind) in [("placebo", 0), ("bootstrap", 1)] {
            let Some(spec) = case.get(key) else { continue };
            let seed = spec["seed"].as_i64().unwrap() as i32;
            let replications = spec["replications"].as_u64().unwrap() as usize;
            let method = if kind == 0 {
                SynthSe::Placebo { replications, seed }
            } else {
                SynthSe::Bootstrap { replications, seed }
            };
            match spec["se"].as_f64() {
                Some(se) => close(name, key, &[fit.se(method).unwrap()], &[se]),
                None => assert!(
                    fit.se(method).is_err(),
                    "{name}: {key} should be unavailable"
                ),
            }
        }
        checked += 1;
    }
    assert_eq!(checked, 9);
}

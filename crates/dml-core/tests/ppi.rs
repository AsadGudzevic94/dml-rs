//! Parity with `ppi_py` 0.2.3 (`ppi_mean_ci`, `ppi_ols_ci`, `ppi_logistic_ci`
//! and the cross-prediction versions). The fixture comes from
//! `parity/generate_ppi_fixtures.py`.

use dml_core::{PpiData, PpiLambda, PredictionPowered};
use ndarray::{Array1, Array2};
use serde_json::Value;

/// ppi_py minimises the logistic loss with L-BFGS-B (`ftol = 1e-15`), which
/// stops with gradients of ~5e-10 (Newton's method reaches ~1e-17), so its
/// logistic estimates differ from the optimum by up to ~2e-8.
const TOL: f64 = 1e-9;
const TOL_LOGISTIC: f64 = 1e-7;

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

#[test]
fn matches_ppi_py() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ppi_ppi_py.json"
    );
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let data = &fixture["data"];
    let cases = fixture["cases"].as_array().unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let estimand = case["estimand"].as_str().unwrap();
        let (y, yhat) = match case["outcome"].as_str().unwrap() {
            "y" => (vector(&data["y"]), vector(&data["yhat"])),
            _ => (vector(&data["b"]), vector(&data["phat"])),
        };
        let unlabelled = vector(&data[case["unlabelled"].as_str().unwrap()]);
        let mut ppi_data = PpiData::new(y, yhat, unlabelled).unwrap();
        let model = match estimand {
            "mean" => PredictionPowered::mean(),
            "linear" => PredictionPowered::linear_regression(),
            _ => PredictionPowered::logistic_regression(),
        };
        if estimand != "mean" {
            ppi_data = ppi_data
                .with_covariates(matrix(&data["X"]), matrix(&data["Xu"]))
                .unwrap();
        }
        let model = match case["lam"].as_f64() {
            Some(l) => model.lambda(PpiLambda::Fixed(l)),
            None => model,
        };
        let fit = model.fit(&ppi_data).unwrap();
        let tol = if estimand == "logistic" {
            TOL_LOGISTIC
        } else {
            TOL
        };
        let lam = case["lam_used"].as_f64().unwrap();
        assert!(
            (fit.lambda() - lam).abs() < tol,
            "{name}: lambda {} vs {lam}",
            fit.lambda()
        );
        let (coef, se) = (vector(&case["coef"]), vector(&case["se"]));
        for j in 0..coef.len() {
            let (c, s) = (fit.estimate().coef()[j], fit.estimate().se()[j]);
            assert!(
                (c - coef[j]).abs() < tol,
                "{name}[{j}]: coef {c} vs {}",
                coef[j]
            );
            assert!((s - se[j]).abs() < tol, "{name}[{j}]: se {s} vs {}", se[j]);
        }
        if lam == 0.0 {
            // The classical estimate is the λ = 0 one.
            for j in 0..coef.len() {
                assert!((fit.classical().coef()[j] - coef[j]).abs() < tol, "{name}");
                assert!((fit.classical().se()[j] - se[j]).abs() < tol, "{name}");
            }
        }
    }
    assert_eq!(cases.len(), 14);
}

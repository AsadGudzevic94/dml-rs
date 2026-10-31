//! Parity with R's `causalweight::medDML(MLmethod = "parametric")` (lm and
//! logit glm nuisances) on medDML's own folds. The fixture comes from
//! `parity/generate_mediation_fixtures.R`.

use dml_core::{DmlData, Fold, LogisticRegression, Mediation, Ols, TrimmingRule};
use ndarray::{Array1, Array2};
use serde_json::Value;

/// glm stops its IRLS iterations at a relative deviance change of 1e-8, which
/// leaves differences of ~2e-10 in the estimates.
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

#[test]
fn matches_causalweight_med_dml() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/mediation_causalweight.json"
    );
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut checked = 0;
    for (name, case) in fixture.as_object().unwrap() {
        if name.starts_with('_') {
            continue;
        }
        let data = DmlData::new(vector(&case["y"]), vector(&case["d"]), matrix(&case["x"]))
            .unwrap()
            .with_mediators(matrix(&case["m"]))
            .unwrap();
        let n = data.n_obs();
        let tests: Vec<Vec<usize>> = case["folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                f.as_array()
                    .unwrap()
                    .iter()
                    .map(|i| i.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();
        let folds: Vec<Fold> = tests
            .iter()
            .map(|test| Fold {
                train: (0..n).filter(|i| !test.contains(i)).collect(),
                test: test.clone(),
            })
            .collect();
        let opt = &case["options"];
        let fit = Mediation::new(Ols::new(), LogisticRegression::unpenalized())
            .trimming_rule(TrimmingRule::Drop)
            .trimming_threshold(opt["trim"].as_f64().unwrap())
            .normalize_ipw(opt["normalized"].as_bool().unwrap())
            .nested_split(!opt["fewsplits"].as_bool().unwrap())
            .debias_nested(opt["debiasfirst"].as_bool().unwrap())
            .fit_with_folds(&data, &[folds])
            .unwrap();
        assert_eq!(
            fit.n_trimmed()[0] as u64,
            case["ntrimmed"].as_u64().unwrap(),
            "{name}"
        );
        // medDML: total, dir.treat, dir.control, indir.treat, indir.control, Y(0,M(0)).
        let effect = vector(&case["effect"]);
        let se = vector(&case["se"]);
        let e = fit.effects();
        let po = fit.potential_outcomes();
        let ours_coef: Vec<f64> = e.coef().iter().copied().chain([po.coef()[3]]).collect();
        let ours_se: Vec<f64> = e.se().iter().copied().chain([po.se()[3]]).collect();
        for j in 0..6 {
            assert!(
                (ours_coef[j] - effect[j]).abs() < TOL,
                "{name} effect {j}: {} vs {}",
                ours_coef[j],
                effect[j]
            );
            assert!(
                (ours_se[j] - se[j]).abs() < TOL,
                "{name} se {j}: {} vs {}",
                ours_se[j],
                se[j]
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 5);
}

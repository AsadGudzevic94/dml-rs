//! Shared helpers for loading the Python DoubleML parity fixtures.
#![allow(dead_code)] // each test binary uses a different subset

use std::path::PathBuf;

use dml_core::{Classifier, ClusterFold, DmlData, Fold, Lasso, LogisticRegression, Ols, Regressor};
use ndarray::{Array1, Array2};
use serde_json::Value;

pub struct Fixture(pub Value);

impl Fixture {
    pub fn load(name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(format!("{name}.json"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        Self(serde_json::from_str(&text).expect("valid fixture JSON"))
    }

    pub fn vec(&self, key: &str) -> Array1<f64> {
        to_vec(&self.0[key])
    }

    pub fn matrix(&self, key: &str) -> Array2<f64> {
        to_matrix(&self.0[key])
    }

    /// Sample splits per repetition.
    pub fn folds(&self) -> Vec<Vec<Fold>> {
        self.0["folds"]
            .as_array()
            .expect("folds")
            .iter()
            .map(|rep| {
                rep.as_array()
                    .expect("rep")
                    .iter()
                    .map(|f| Fold {
                        train: indices(&f["train"]),
                        test: indices(&f["test"]),
                    })
                    .collect()
            })
            .collect()
    }

    /// DoubleML's cluster splits per repetition, for clustered fixtures.
    pub fn cluster_folds(&self) -> Option<Vec<Vec<ClusterFold>>> {
        let reps = self
            .0
            .get("cluster_folds")?
            .as_array()
            .expect("cluster folds");
        Some(
            reps.iter()
                .map(|rep| {
                    rep.as_array()
                        .expect("rep")
                        .iter()
                        .map(|f| ClusterFold {
                            fold: Fold {
                                train: indices(&f["train"]),
                                test: indices(&f["test"]),
                            },
                            test_clusters: f["test_clusters"]
                                .as_array()
                                .expect("test clusters")
                                .iter()
                                .map(|c| {
                                    c.as_array()
                                        .expect("cluster ids")
                                        .iter()
                                        .map(|v| v.as_i64().expect("cluster id"))
                                        .collect()
                                })
                                .collect(),
                        })
                        .collect()
                })
                .collect(),
        )
    }

    pub fn expected(&self, key: &str) -> &Value {
        &self.0["expected"][key]
    }

    /// First entry of an expected per-coefficient list.
    pub fn expected_scalar(&self, key: &str) -> f64 {
        self.expected(key)[0].as_f64().expect("number")
    }

    /// Expected predictions of a nuisance for one treatment and repetition.
    pub fn prediction(&self, name: &str, treatment: usize, rep: usize) -> Array1<f64> {
        to_vec(&self.expected("predictions")[name][treatment][rep])
    }

    pub fn option_str(&self, key: &str) -> &str {
        self.0["options"][key].as_str().expect("string option")
    }

    pub fn option_bool(&self, key: &str) -> bool {
        self.0["options"][key].as_bool().unwrap_or(false)
    }

    /// A nuisance learner that may be absent (e.g. `ml_m` for experimental DID).
    pub fn has_learner(&self, nuisance: &str) -> bool {
        !self.0["learners"][nuisance].is_null()
    }

    pub fn regressor(&self, nuisance: &str) -> Box<dyn Regressor> {
        regressor_from_spec(&self.0["learners"][nuisance])
    }

    pub fn classifier(&self, nuisance: &str) -> Box<dyn Classifier> {
        let spec = &self.0["learners"][nuisance];
        match spec["kind"].as_str().expect("learner kind") {
            "logit" => Box::new(LogisticRegression::new(spec["c"].as_f64().expect("c"))),
            other => panic!("unknown classifier {other}"),
        }
    }

    /// `DmlData` from the stored arrays (1-D or 2-D `d`, optional `z`).
    pub fn data(&self) -> DmlData {
        let (y, x) = (self.vec("y"), self.matrix("x"));
        let d = &self.0["d"];
        let data = if d[0].is_array() {
            DmlData::with_treatments(y, to_matrix(d), x)
        } else {
            DmlData::new(y, to_vec(d), x)
        }
        .unwrap();
        let data = match self.0.get("z") {
            Some(z) => data.with_instrument(to_vec(z)).unwrap(),
            None => data,
        };
        let data = match self.0.get("t") {
            Some(t) => data.with_time_indicator(to_vec(t)).unwrap(),
            None => data,
        };
        let ids = |v: &Value| -> Array1<i64> {
            v.as_array()
                .expect("cluster ids")
                .iter()
                .map(|c| c.as_i64().expect("cluster id"))
                .collect()
        };
        match self.0.get("clusters").and_then(Value::as_array) {
            Some(c) if c.len() == 1 => data.with_clusters(ids(&c[0])).unwrap(),
            Some(c) => data.with_two_way_clusters(ids(&c[0]), ids(&c[1])).unwrap(),
            None => data,
        }
    }
}

pub fn regressor_from_spec(spec: &Value) -> Box<dyn Regressor> {
    match spec["kind"].as_str().expect("learner kind") {
        "ols" => Box::new(Ols::new()),
        "ridge" => Box::new(Ols::with_ridge(spec["alpha"].as_f64().expect("alpha"))),
        "lasso" => Box::new(Lasso::new(spec["alpha"].as_f64().expect("alpha"))),
        other => panic!("unknown regressor {other}"),
    }
}

pub fn to_vec(v: &Value) -> Array1<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

pub fn to_matrix(v: &Value) -> Array2<f64> {
    let rows = v.as_array().expect("matrix");
    let p = rows.first().map_or(0, |r| r.as_array().expect("row").len());
    let flat: Vec<f64> = rows.iter().flat_map(|r| to_vec(r).to_vec()).collect();
    Array2::from_shape_vec((rows.len(), p), flat).expect("rectangular matrix")
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("indices")
        .iter()
        .map(|i| i.as_u64().expect("index") as usize)
        .collect()
}

/// Assert `|a − b| ≤ tol · max(1, |b|)`.
#[track_caller]
pub fn assert_close(what: &str, a: f64, b: f64, tol: f64) {
    let scale = b.abs().max(1.0);
    assert!(
        (a - b).abs() <= tol * scale,
        "{what}: rust = {a:.15e}, doubleml = {b:.15e}, diff = {:.3e}",
        (a - b).abs()
    );
}

#[track_caller]
pub fn assert_all_close(what: &str, a: &Array1<f64>, b: &Array1<f64>, tol: f64) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_close(&format!("{what}[{i}]"), x, y, tol);
    }
}

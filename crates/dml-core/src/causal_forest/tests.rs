//! Tests of the forests.

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use super::engine::invert;
use super::*;
use crate::data::DmlData;
use crate::learner::{LogisticRegression, Ols, Regressor};
use crate::model::TreatmentLearner;

/// Wager and Athey (2018)-style design: τ(x) = 1 + 1 / (1 + e^{−20(x₁ − 1/3)}),
/// randomised binary treatment, x uniform.
fn design(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, 4), |_| rng.gen::<f64>());
    let d: Array1<f64> = (0..n).map(|_| f64::from(rng.gen::<f64>() < 0.5)).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| {
            let e: f64 = rng.sample(StandardNormal);
            tau(x[[i, 0]]) * d[i] + x[[i, 1]] + e
        })
        .collect();
    (x, y, d)
}

fn tau(x1: f64) -> f64 {
    1.0 + 1.0 / (1.0 + (-20.0 * (x1 - 1.0 / 3.0)).exp())
}

#[test]
fn recovers_the_effect_function_and_covers() {
    let (x, y, d) = design(3000, 1);
    let data = DmlData::new(y, d, x).unwrap();
    let fit = CausalForestDml::new(
        Ols::new(),
        TreatmentLearner::Classifier(Box::new(LogisticRegression::default())),
    )
    .forest(ForestSettings {
        num_trees: 1000,
        ..Default::default()
    })
    .fit(&data)
    .unwrap();
    let grid = Array2::from_shape_fn(
        (50, 4),
        |(i, j)| if j == 0 { (i as f64 + 0.5) / 50.0 } else { 0.5 },
    );
    let (est, var) = fit.effect(grid.view()).unwrap();
    let mut covered = 0;
    let mut sq = 0.0;
    for i in 0..50 {
        let truth = tau(grid[[i, 0]]);
        sq += (est[i] - truth).powi(2);
        if (est[i] - truth).abs() <= 1.96 * var[i].sqrt() {
            covered += 1;
        }
    }
    let rmse = (sq / 50.0).sqrt();
    assert!(rmse < 0.25, "rmse {rmse}");
    assert!(covered >= 35, "covered {covered} of 50");
    // The splits concentrate on x₁.
    let imp = fit.forest().variable_importance();
    assert!(imp[0] > 0.5, "{imp}");
    // ATE of the design: E[τ(X)] ≈ 1.66.
    let ate = fit.ate().unwrap();
    assert!((ate.coef()[0] - 1.664).abs() < 4.0 * ate.se()[0]);
    let gate = fit
        .gate(
            Array2::from_shape_fn((3000, 2), |(i, g)| {
                f64::from((data.x()[[i, 0]] > 0.5) == (g == 1))
            })
            .view(),
        )
        .unwrap();
    assert!(gate.coef()[1] > gate.coef()[0]);
}

#[test]
fn homogeneous_effect_and_continuous_treatment() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let n = 1500;
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen::<f64>());
    let d: Array1<f64> = (0..n)
        .map(|i| x[[i, 0]] + rng.sample::<f64, _>(StandardNormal))
        .collect();
    let y: Array1<f64> = (0..n)
        .map(|i| 0.7 * d[i] + x[[i, 1]] + rng.sample::<f64, _>(StandardNormal))
        .collect();
    let data = DmlData::new(y, d, x).unwrap();
    let fit = CausalForestDml::new(
        Ols::new(),
        TreatmentLearner::Regressor(Box::new(Ols::new())),
    )
    .forest(ForestSettings {
        num_trees: 400,
        seed: 3,
        ..Default::default()
    })
    .fit(&data)
    .unwrap();
    let mean = fit.oob_effect().mean().unwrap();
    assert!((mean - 0.7).abs() < 0.1, "{mean}");
    assert!(fit.ate().is_err());
    assert!(fit
        .oob_variance()
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0));
}

#[test]
fn settings_are_validated_and_seeded() {
    let (x, y, d) = design(300, 2);
    let s = ForestSettings {
        num_trees: 50,
        seed: 9,
        ..Default::default()
    };
    let a = CausalForest::fit(x.view(), y.view(), d.view(), s.clone()).unwrap();
    let b = CausalForest::fit(x.view(), y.view(), d.view(), s).unwrap();
    assert_eq!(
        a.predict(x.view()).unwrap().0,
        b.predict(x.view()).unwrap().0
    );
    let bad = ForestSettings {
        sample_fraction: 0.8,
        ..Default::default()
    };
    assert!(CausalForest::fit(x.view(), y.view(), d.view(), bad).is_err());
    let one = ForestSettings {
        num_trees: 20,
        ci_group_size: 1,
        sample_fraction: 0.7,
        ..Default::default()
    };
    let f = CausalForest::fit(x.view(), y.view(), d.view(), one).unwrap();
    assert!(f.predict(x.view()).unwrap().1.iter().all(|v| v.is_nan()));
}

fn corr(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let (ma, mb) = (a.mean().unwrap(), b.mean().unwrap());
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

fn logit() -> TreatmentLearner {
    TreatmentLearner::Classifier(Box::new(LogisticRegression::default()))
}

#[test]
fn small_inverse() {
    let a = [4.0, 1.0, 2.0, 3.0];
    let inv = invert(&a, 2).unwrap();
    let expect = [0.3, -0.1, -0.2, 0.4];
    for (u, v) in inv.iter().zip(expect) {
        assert!((u - v).abs() < 1e-15);
    }
    assert!(invert(&[1.0, 2.0, 2.0, 4.0], 2).is_none());
}

#[test]
fn policy_tree_treats_where_the_effect_is_large() {
    let n = 2000;
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen::<f64>());
    let d: Array1<f64> = (0..n).map(|_| f64::from(rng.gen::<f64>() < 0.5)).collect();
    let y: Array1<f64> = (0..n)
        .map(|i| (x[[i, 0]] - 0.5) * 4.0 * d[i] + rng.sample::<f64, _>(StandardNormal))
        .collect();
    let data = DmlData::new(y, d, x.clone()).unwrap();
    let fit = CausalForestDml::new(Ols::new(), logit())
        .forest(ForestSettings {
            num_trees: 400,
            ..Default::default()
        })
        .fit(&data)
        .unwrap();
    let tree = fit.policy_tree(x.view(), 1, 8, 0.01).unwrap();
    let grid = Array2::from_shape_fn((2, 3), |(i, _)| if i == 0 { 0.1 } else { 0.9 });
    assert_eq!(tree.predict(grid.view()).unwrap().to_vec(), vec![0.0, 1.0]);
}

/// Binary instrument with confounded take-up: compliers 60%, always-takers
/// 20% (higher outcomes), never-takers 20% (lower outcomes).
#[test]
fn instrumental_forest_recovers_the_local_effect() {
    let n = 4000;
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    let x = Array2::from_shape_fn((n, 4), |_| rng.gen::<f64>());
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    let mut z = Array1::zeros(n);
    let theta = |x0: f64| 1.0 + 2.0 * x0;
    for i in 0..n {
        z[i] = f64::from(rng.gen::<f64>() < 0.5);
        let u: f64 = rng.gen();
        let (di, shift) = if u < 0.6 {
            (z[i], 0.0)
        } else if u < 0.8 {
            (1.0, 1.5)
        } else {
            (0.0, -0.5)
        };
        d[i] = di;
        y[i] = theta(x[[i, 0]]) * di + x[[i, 1]] + shift + rng.sample::<f64, _>(StandardNormal);
    }
    let truth = x.column(0).mapv(theta);
    let data = DmlData::new(y, d, x.clone())
        .unwrap()
        .with_instrument(z)
        .unwrap();
    let fit = InstrumentalForestDml::new(Ols::new(), logit(), logit())
        .forest(ForestSettings {
            num_trees: 1000,
            ..Default::default()
        })
        .fit(&data)
        .unwrap();
    let r = corr(fit.oob_effect(), &truth);
    assert!(r > 0.8, "corr {r}");
    let mc = fit.compliance().unwrap().mean().unwrap();
    assert!((mc - 0.6).abs() < 0.05, "compliance {mc}");
    let ate = fit.ate().unwrap();
    let mean = truth.mean().unwrap();
    assert!(
        (ate.coef()[0] - mean).abs() < 4.0 * ate.se()[0],
        "{} vs {mean}",
        ate.coef()[0]
    );
    assert!(fit.oob_variance().iter().all(|v| v.is_finite()));
    // A continuous instrument has no doubly robust scores.
    let cont = DmlData::new(data.y().to_owned(), data.treatment(0).to_owned(), x)
        .unwrap()
        .with_instrument(data.z().unwrap().mapv(|v| v + 0.1))
        .unwrap();
    let f2 = InstrumentalForestDml::new(
        Ols::new(),
        logit(),
        TreatmentLearner::Regressor(Box::new(Ols::new())),
    )
    .forest(ForestSettings {
        num_trees: 50,
        ..Default::default()
    })
    .fit(&cont)
    .unwrap();
    assert!(f2.ate().is_err() && f2.compliance().is_none());
}

#[test]
fn multi_arm_forest_recovers_each_arm() {
    let n = 4000;
    let mut rng = ChaCha8Rng::seed_from_u64(31);
    let x = Array2::from_shape_fn((n, 4), |_| rng.gen::<f64>());
    let tau1 = |x0: f64| 2.0 * x0;
    let tau2 = |x0: f64| 1.0 - x0;
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let u: f64 = rng.gen();
        d[i] = if u < 0.4 {
            0.0
        } else if u < 0.7 {
            1.0
        } else {
            2.0
        };
        let effect = match d[i] as usize {
            0 => 0.0,
            1 => tau1(x[[i, 0]]),
            _ => tau2(x[[i, 0]]),
        };
        y[i] = x[[i, 1]] + effect + rng.sample::<f64, _>(StandardNormal);
    }
    let data = DmlData::new(y, d, x.clone()).unwrap();
    let fit = MultiArmCausalForestDml::new(Ols::new(), LogisticRegression::default())
        .forest(ForestSettings {
            num_trees: 1000,
            ..Default::default()
        })
        .fit(&data)
        .unwrap();
    assert_eq!(fit.levels(), &[0.0, 1.0, 2.0]);
    assert_eq!(fit.arm_names()[1], "d=2 vs 0");
    let t1 = x.column(0).mapv(tau1);
    let t2 = x.column(0).mapv(tau2);
    let oob = fit.oob_effect();
    let r1 = corr(&oob.column(0).to_owned(), &t1);
    let r2 = corr(&oob.column(1).to_owned(), &t2);
    assert!(r1 > 0.8 && r2 > 0.6, "corr {r1} {r2}");
    let ate = fit.ate().unwrap();
    for (a, truth) in [(0, 1.0), (1, 0.5)] {
        assert!(
            (ate.coef()[a] - truth).abs() < 4.0 * ate.se()[a],
            "arm {a}: {}",
            ate.coef()[a]
        );
    }
    let p = fit.propensity();
    assert!(p.rows().into_iter().all(|r| (r.sum() - 1.0).abs() < 1e-12));
    let groups = Array2::from_shape_fn((n, 2), |(i, g)| f64::from((x[[i, 0]] > 0.5) == (g == 1)));
    let gate = fit.gate(0, groups.view()).unwrap();
    assert!(gate.coef()[1] > gate.coef()[0]);
    assert!(fit.blp(2, x.view()).is_err());
}

#[test]
fn regression_forest_predicts_with_variance() {
    let n = 2000;
    let mut rng = ChaCha8Rng::seed_from_u64(41);
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen::<f64>());
    let f = |r: ndarray::ArrayView1<'_, f64>| (3.0 * r[0]).sin() + r[1];
    let y: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| f(r) + 0.3 * rng.sample::<f64, _>(StandardNormal))
        .collect();
    let mut forest = RegressionForest::new().settings(ForestSettings {
        num_trees: 500,
        ..Default::default()
    });
    assert!(forest.predict(x.view()).is_err());
    forest.fit(x.view(), y.view()).unwrap();
    let grid = Array2::from_shape_fn((100, 3), |_| rng.gen::<f64>());
    let (pred, var) = forest.predict_with_variance(grid.view()).unwrap();
    let truth: Array1<f64> = grid.rows().into_iter().map(f).collect();
    let rmse = (&pred - &truth).mapv(|v| v * v).mean().unwrap().sqrt();
    assert!(rmse < 0.15, "rmse {rmse}");
    assert!(var.iter().all(|v| v.is_finite() && *v >= 0.0));
    let (oob, _) = forest.predict_oob(x.view()).unwrap();
    let truth_train: Array1<f64> = x.rows().into_iter().map(f).collect();
    let r = corr(&oob, &truth_train);
    assert!(r > 0.9, "corr {r}");
    assert!(forest.fresh().predict(x.view()).is_err());
}

#[test]
fn quantile_forest_tracks_conditional_quantiles() {
    let n = 4000;
    let mut rng = ChaCha8Rng::seed_from_u64(51);
    let x = Array2::from_shape_fn((n, 2), |_| rng.gen::<f64>());
    let y: Array1<f64> = (0..n)
        .map(|i| x[[i, 0]] + (0.5 + 2.0 * x[[i, 1]]) * rng.sample::<f64, _>(StandardNormal))
        .collect();
    let mut forest = QuantileForest::default().settings(ForestSettings {
        num_trees: 500,
        ci_group_size: 1,
        ..Default::default()
    });
    forest.fit(x.view(), y.view()).unwrap();
    let grid = Array2::from_shape_fn(
        (50, 2),
        |(i, j)| {
            if j == 0 {
                0.5
            } else {
                (i as f64 + 0.5) / 50.0
            }
        },
    );
    let q = forest.predict(grid.view(), &[0.9, 0.1, 0.5]).unwrap();
    let z90 = 1.281_551_565_545;
    let mut err = 0.0;
    for i in 0..50 {
        let s = 0.5 + 2.0 * grid[[i, 1]];
        err += (q[[i, 0]] - (0.5 + z90 * s)).abs() + (q[[i, 1]] - (0.5 - z90 * s)).abs();
        assert!(q[[i, 1]] <= q[[i, 2]] && q[[i, 2]] <= q[[i, 0]]);
    }
    let mae = err / 100.0;
    assert!(mae < 0.3, "mae {mae}");
    // The spread grows with the second variable, which the splits use.
    let imp = forest.variable_importance().unwrap();
    assert!(imp[1] > imp[0], "{imp}");
    assert!(forest.predict(grid.view(), &[1.5]).is_err());
    assert_eq!(forest.predict_oob(x.view(), &[0.5]).unwrap().dim(), (n, 1));
}

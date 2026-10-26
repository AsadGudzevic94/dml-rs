//! Staggered DID: parity with DoubleML's `DoubleMLDIDMulti` and Monte-Carlo checks.

mod common;

use common::{
    assert_all_close, assert_close, check_framework_sensitivity, to_matrix, to_vec, Fixture,
};
use dml_core::{
    Aggregation, ControlGroup, DidScore, Fold, GtCombinations, LogisticRegression, Ols, PanelData,
    RepeatedCrossSections, StaggeredDid, StaggeredInput,
};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::Value;

const TOL: f64 = 1e-7;

fn folds_of(v: &Value) -> Vec<Vec<Vec<Fold>>> {
    let idx = |a: &Value| -> Vec<usize> {
        a.as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect()
    };
    v.as_array()
        .unwrap()
        .iter()
        .map(|gt| {
            gt.as_array()
                .unwrap()
                .iter()
                .map(|rep| {
                    rep.as_array()
                        .unwrap()
                        .iter()
                        .map(|f| Fold {
                            train: idx(&f["train"]),
                            test: idx(&f["test"]),
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

fn check_staggered(name: &str) {
    let fx = Fixture::load(name);
    let cohort: Array1<f64> = fx.0["cohort"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g.as_f64().unwrap_or(f64::INFINITY))
        .collect();
    // Either a wide panel or repeated cross-sections (rows in data order).
    let (panel, cross);
    let data: StaggeredInput = if fx.0["model"] == "staggered_cs" {
        cross = RepeatedCrossSections::new(fx.vec("y"), fx.vec("time"), cohort, fx.matrix("x"))
            .unwrap();
        (&cross).into()
    } else {
        let periods: Vec<f64> = to_vec(&fx.0["periods"]).to_vec();
        panel = PanelData::new(fx.matrix("y_wide"), periods, cohort, fx.matrix("x")).unwrap();
        (&panel).into()
    };

    let opt = &fx.0["options"];
    let score = match opt["score"].as_str().unwrap() {
        "observational" => DidScore::Observational,
        _ => DidScore::Experimental,
    };
    let control = match opt["control_group"].as_str().unwrap() {
        "never_treated" => ControlGroup::NeverTreated,
        _ => ControlGroup::NotYetTreated,
    };
    let setting = match opt["gt_combinations"].as_str().unwrap() {
        "standard" => GtCombinations::Standard,
        "all" => GtCombinations::All,
        _ => GtCombinations::Universal,
    };
    let ml_m = fx.has_learner("ml_m").then(|| fx.classifier("ml_m"));
    let model = StaggeredDid::from_boxed(fx.regressor("ml_g"), ml_m)
        .score(score)
        .control_group(control)
        .gt_combinations(setting)
        .anticipation_periods(opt["anticipation_periods"].as_u64().unwrap() as usize)
        .in_sample_normalization(opt["in_sample_normalization"].as_bool().unwrap())
        .calibration(fx.calibration());

    // Same group-time combinations, in the same order.
    let expected_gt = to_matrix(fx.expected("gt"));
    let gts = model.group_times(data).unwrap();
    assert_eq!(
        gts.len(),
        expected_gt.nrows(),
        "{name}: number of group-time effects"
    );
    for (i, gt) in gts.iter().enumerate() {
        assert_eq!(
            [gt.cohort, gt.t_pre, gt.t_eval],
            [
                expected_gt[[i, 0]],
                expected_gt[[i, 1]],
                expected_gt[[i, 2]]
            ],
            "{name}: combination {i}"
        );
    }

    let fit = model
        .fit_with_folds(data, &folds_of(&fx.0["gt_folds"]))
        .unwrap();
    let att = fit.att();
    assert_all_close(
        &format!("{name} ATT coef"),
        &att.coef().to_owned(),
        &to_vec(fx.expected("coef")),
        TOL,
    );
    assert_all_close(
        &format!("{name} ATT se"),
        &att.se().to_owned(),
        &to_vec(fx.expected("se")),
        TOL,
    );
    let (all_coef, all_se) = (
        to_matrix(fx.expected("all_coef")),
        to_matrix(fx.expected("all_se")),
    );
    for ((idx, &c), &s) in att.all_coef().indexed_iter().zip(att.all_se().iter()) {
        assert_close(&format!("{name} all_coef{idx:?}"), c, all_coef[idx], TOL);
        assert_close(&format!("{name} all_se{idx:?}"), s, all_se[idx], TOL);
    }

    for (key, how) in [
        ("group", Aggregation::Group),
        ("time", Aggregation::Time),
        ("eventstudy", Aggregation::EventStudy),
    ] {
        let e = &fx.expected("aggregations")[key];
        let agg = fit.aggregate(how).unwrap();
        let w = to_matrix(&e["weights"]);
        assert_eq!(agg.weights().dim(), w.dim(), "{name} {key}: weight shape");
        for ((idx, &a), &b) in agg.weights().indexed_iter().zip(w.iter()) {
            assert_close(&format!("{name} {key} weight{idx:?}"), a, b, 1e-12);
        }
        assert_all_close(
            &format!("{name} {key} overall weights"),
            agg.overall_weights(),
            &to_vec(&e["overall_weights"]),
            1e-12,
        );
        let names: Vec<String> = e["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let ours: Vec<f64> = agg
            .effects()
            .treatment_names()
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        let theirs: Vec<f64> = names.iter().map(|s| s.parse().unwrap()).collect();
        assert_eq!(ours, theirs, "{name} {key}: effect labels");
        assert_all_close(
            &format!("{name} {key} coef"),
            &agg.effects().coef().to_owned(),
            &to_vec(&e["coef"]),
            TOL,
        );
        assert_all_close(
            &format!("{name} {key} se"),
            &agg.effects().se().to_owned(),
            &to_vec(&e["se"]),
            TOL,
        );
        assert_all_close(
            &format!("{name} {key} overall coef"),
            &agg.overall().coef().to_owned(),
            &to_vec(&e["overall_coef"]),
            TOL,
        );
        assert_all_close(
            &format!("{name} {key} overall se"),
            &agg.overall().se().to_owned(),
            &to_vec(&e["overall_se"]),
            TOL,
        );
        // Sensitivity of the aggregated effects: bounds add with |weights|.
        let what = format!("{name} {key}");
        check_framework_sensitivity(&what, agg.effects(), &e["sensitivity"], TOL);
        check_framework_sensitivity(&what, agg.overall(), &e["overall_sensitivity"], TOL);
    }

    // Sensitivity of every ATT(g, t), and a benchmark without the first covariate.
    let e = fx.expected("sensitivity");
    check_framework_sensitivity(&format!("{name} ATT"), att, e, TOL);
    if let Some(bm) = e.get("benchmark") {
        let StaggeredInput::Panel(panel) = data else {
            panic!("benchmarks are recorded for panels")
        };
        let drop: Vec<usize> = to_vec(&bm["drop"]).iter().map(|&v| v as usize).collect();
        let short = model
            .fit_with_folds(
                &panel.without_covariates(&drop).unwrap(),
                &folds_of(&fx.0["gt_folds"]),
            )
            .unwrap();
        let ours = att.sensitivity_benchmark(short.att()).unwrap();
        for (key, v) in [
            ("cf_y", &ours.cf_y),
            ("cf_d", &ours.cf_d),
            ("rho", &ours.rho),
            ("delta_theta", &ours.delta_theta),
        ] {
            assert_all_close(
                &format!("{name} benchmark {key}"),
                v,
                &to_vec(&bm[key]),
                TOL,
            );
        }
    }
}

#[test]
fn parity_isotonic_calibration() {
    check_staggered("staggered_isotonic_cv");
}

#[test]
fn parity_never_treated_observational() {
    check_staggered("staggered_never_observational");
}

#[test]
fn parity_not_yet_treated_experimental_repeated_splits() {
    check_staggered("staggered_notyet_experimental_rep2");
}

#[test]
fn parity_cross_sections_never_treated_observational() {
    check_staggered("staggered_cs_never_observational");
}

#[test]
fn parity_cross_sections_not_yet_treated_experimental_repeated_splits() {
    check_staggered("staggered_cs_notyet_experimental_rep2");
}

#[test]
fn parity_universal_base_with_anticipation() {
    check_staggered("staggered_universal_anticipation");
}

fn std_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Cohorts treated in periods 2 and 3 (plus never treated) over periods 0..=3.
/// Selection into cohorts depends on x₀, which also drives the trend. The
/// effect grows with exposure: 1 + 0.5·(t − g) once treated, 0 before.
fn simulate_parts(n: usize, seed: u64) -> (Array2<f64>, Vec<f64>, Array1<f64>, Array2<f64>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let periods = vec![0.0, 1.0, 2.0, 3.0];
    let x = Array2::from_shape_simple_fn((n, 2), || std_normal(&mut rng));
    let mut y = Array2::zeros((n, 4));
    let mut cohort = Array1::zeros(n);
    for i in 0..n {
        let s = 1.0 / (1.0 + (-x[[i, 0]]).exp());
        let u: f64 = rng.gen();
        cohort[i] = if u < 0.3 * s + 0.1 {
            2.0
        } else if u < 0.6 * s + 0.2 {
            3.0
        } else {
            f64::INFINITY
        };
        let level = x[[i, 1]] + std_normal(&mut rng);
        for (j, &t) in periods.iter().enumerate() {
            let effect = if t >= cohort[i] {
                1.0 + 0.5 * (t - cohort[i])
            } else {
                0.0
            };
            y[[i, j]] = level + t * (0.5 + 0.4 * x[[i, 0]]) + effect + std_normal(&mut rng);
        }
    }
    (y, periods, cohort, x)
}

fn simulate(n: usize, seed: u64) -> PanelData {
    let (y, periods, cohort, x) = simulate_parts(n, seed);
    PanelData::new(y, periods, cohort, x).unwrap()
}

#[test]
fn recovers_dynamic_effects_and_flat_pre_trends() {
    let data = simulate(4000, 1);
    let fit = StaggeredDid::new(Ols::new(), LogisticRegression::default())
        .control_group(ControlGroup::NotYetTreated)
        .fit(&data)
        .unwrap();
    let es = fit.aggregate(Aggregation::EventStudy).unwrap();
    let names: Vec<f64> = es
        .effects()
        .treatment_names()
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    for (j, &e) in names.iter().enumerate() {
        let truth = if e >= 0.0 { 1.0 + 0.5 * e } else { 0.0 };
        let ci = es.effects().confint(0.99).unwrap();
        assert!(
            ci[[j, 0]] < truth && truth < ci[[j, 1]],
            "event time {e}: CI {:?}, truth {truth}",
            ci.row(j)
        );
    }
    // Overall event-study effect: mean of 1.0 (e = 0) and 1.5 (e = 1).
    let overall = es.overall().coef()[0];
    assert!((overall - 1.25).abs() < 0.15, "overall = {overall}");

    // Joint bands over all group-time effects are wider than pointwise ones.
    let boot = fit
        .att()
        .bootstrap(dml_core::BootstrapMethod::Normal, 500, 1)
        .unwrap();
    let joint = boot.joint_confint(0.95).unwrap();
    let point = fit.att().confint(0.95).unwrap();
    assert!(
        (0..joint.nrows()).all(|i| joint[[i, 1]] - joint[[i, 0]] > point[[i, 1]] - point[[i, 0]])
    );
}

#[test]
fn aggregated_group_effects_have_nominal_coverage() {
    // Group effects: cohort 2 averages e = 0, 1 → 1.25; cohort 3 has e = 0 → 1.0.
    let reps = 100;
    let mut covered = [0usize; 2];
    for r in 0..reps {
        let data = simulate(1500, 500 + r);
        let fit = StaggeredDid::new(Ols::new(), LogisticRegression::default())
            .seed(r)
            .fit(&data)
            .unwrap();
        let ci = fit
            .aggregate(Aggregation::Group)
            .unwrap()
            .effects()
            .confint(0.95)
            .unwrap();
        for (j, truth) in [1.25, 1.0].into_iter().enumerate() {
            covered[j] += usize::from(ci[[j, 0]] <= truth && truth <= ci[[j, 1]]);
        }
    }
    for (j, &c) in covered.iter().enumerate() {
        let coverage = c as f64 / reps as f64;
        assert!(coverage >= 0.87, "cohort {j}: coverage {coverage}");
    }
}

#[test]
fn unbalanced_panels_and_errors() {
    // Drop the last period for every 20th unit: ATT(g, t) involving period 3
    // then use the remaining units, and estimates stay close to the balanced ones.
    let (y, periods, cohort, x) = simulate_parts(3000, 9);
    let balanced = PanelData::new(y.clone(), periods.clone(), cohort.clone(), x.clone()).unwrap();
    let mut y_missing = y;
    for i in (0..3000).step_by(20) {
        y_missing[[i, 3]] = f64::NAN;
    }
    let unbalanced = PanelData::new(y_missing, periods.clone(), cohort.clone(), x.clone()).unwrap();
    let model = StaggeredDid::new(Ols::new(), LogisticRegression::default());
    let (full, part) = (
        model.fit(&balanced).unwrap(),
        model.fit(&unbalanced).unwrap(),
    );
    let diff = &full.att().coef().to_owned() - &part.att().coef().to_owned();
    assert!(diff.iter().all(|d| d.abs() < 0.1), "{diff}");
    // Effects that never touch period 3 use exactly the same sample, hence
    // the same folds and identical estimates.
    for (i, gt) in full.group_times().iter().enumerate() {
        if gt.t_pre != 3.0 && gt.t_eval != 3.0 {
            assert_eq!(full.att().coef()[i], part.att().coef()[i], "{gt:?}");
        }
    }

    // Observational score without a propensity learner, and a cohort that is not a period.
    let no_m = StaggeredDid::from_boxed(Box::new(Ols::new()), None);
    assert!(no_m.fit(&balanced).is_err());
    assert!(PanelData::new(
        Array2::zeros((3000, 4)),
        periods,
        Array1::from_elem(3000, 7.0),
        x
    )
    .is_err());
}

#[test]
fn cross_sections_recover_dynamic_effects() {
    // Observe every simulated unit in one random period only.
    let (y, periods, cohort, x) = simulate_parts(8000, 21);
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    let n = y.nrows();
    let (mut ys, mut ts) = (Array1::zeros(n), Array1::zeros(n));
    for i in 0..n {
        let j = rng.gen_range(0..periods.len());
        ys[i] = y[[i, j]];
        ts[i] = periods[j];
    }
    let data = RepeatedCrossSections::new(ys, ts, cohort, x).unwrap();
    let fit = StaggeredDid::new(Ols::new(), LogisticRegression::default())
        .control_group(ControlGroup::NotYetTreated)
        .fit(&data)
        .unwrap();
    assert!(fit.att().model_name().contains("repeated cross-sections"));
    let es = fit.aggregate(Aggregation::EventStudy).unwrap();
    let ci = es.effects().confint(0.99).unwrap();
    for (j, name) in es.effects().treatment_names().iter().enumerate() {
        let e: f64 = name.parse().unwrap();
        let truth = if e >= 0.0 { 1.0 + 0.5 * e } else { 0.0 };
        assert!(
            ci[[j, 0]] < truth && truth < ci[[j, 1]],
            "event time {e}: CI {:?}",
            ci.row(j)
        );
    }
}

/// Units in `n_clusters` clusters (cohort assigned per cluster) with a common
/// shock per cluster and period, so scores are correlated within clusters.
fn simulate_clustered(n_clusters: usize, per: usize, seed: u64) -> PanelData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let periods = vec![0.0, 1.0, 2.0, 3.0];
    let n = n_clusters * per;
    let x = Array2::from_shape_simple_fn((n, 2), || std_normal(&mut rng));
    let mut y = Array2::zeros((n, 4));
    let mut cohort = Array1::zeros(n);
    let mut labels = Array1::zeros(n);
    for c in 0..n_clusters {
        let u: f64 = rng.gen();
        let g = if u < 0.3 {
            2.0
        } else if u < 0.6 {
            3.0
        } else {
            f64::INFINITY
        };
        let shocks: Vec<f64> = (0..4).map(|_| 1.5 * std_normal(&mut rng)).collect();
        for k in 0..per {
            let i = c * per + k;
            cohort[i] = g;
            labels[i] = c as i64;
            let level = x[[i, 1]] + std_normal(&mut rng);
            for (j, &t) in periods.iter().enumerate() {
                let effect = if t >= g { 1.0 + 0.5 * (t - g) } else { 0.0 };
                y[[i, j]] =
                    level + t * (0.5 + 0.4 * x[[i, 0]]) + effect + shocks[j] + std_normal(&mut rng);
            }
        }
    }
    PanelData::new(y, periods, cohort, x)
        .unwrap()
        .with_clusters(labels)
        .unwrap()
}

#[test]
fn singleton_clusters_reproduce_the_iid_standard_errors() {
    let (y, periods, cohort, x) = simulate_parts(1200, 21);
    let plain = PanelData::new(y, periods, cohort.clone(), x).unwrap();
    let clustered = plain
        .clone()
        .with_clusters((0..1200).map(|i| i as i64).collect())
        .unwrap();
    let model = StaggeredDid::new(Ols::new(), LogisticRegression::default());
    // The same folds for both fits: K-fold over each ATT(g, t)'s sample
    // (its cohort and the never treated).
    let folds: Vec<Vec<Vec<Fold>>> = model
        .group_times(&plain)
        .unwrap()
        .iter()
        .map(|gt| {
            let size = cohort
                .iter()
                .filter(|&&g| g == gt.cohort || g.is_infinite())
                .count();
            vec![dml_core::KFold::new(5, 3).unwrap().split(size).unwrap()]
        })
        .collect();
    let (a, b) = (
        model.fit_with_folds(&plain, &folds).unwrap(),
        model.fit_with_folds(&clustered, &folds).unwrap(),
    );
    assert_all_close(
        "coef",
        &a.att().coef().to_owned(),
        &b.att().coef().to_owned(),
        1e-12,
    );
    assert_all_close(
        "se",
        &a.att().se().to_owned(),
        &b.att().se().to_owned(),
        1e-10,
    );
    assert_eq!(b.att().n_clusters(), Some(1200));
    let (ga, gb) = (
        a.aggregate(Aggregation::Group).unwrap(),
        b.aggregate(Aggregation::Group).unwrap(),
    );
    assert_all_close(
        "group se",
        &ga.effects().se().to_owned(),
        &gb.effects().se().to_owned(),
        1e-10,
    );
}

#[test]
fn cluster_robust_standard_errors_cover_with_cluster_shocks() {
    // Group effects: cohort 2 averages e = 0, 1 → 1.25; cohort 3 has e = 0 → 1.0.
    let reps = 80;
    let (mut clustered, mut iid) = ([0usize; 2], [0usize; 2]);
    for r in 0..reps {
        let data = simulate_clustered(60, 25, 900 + r);
        let model = StaggeredDid::new(Ols::new(), LogisticRegression::default()).seed(r);
        let fit = model.fit(&data).unwrap();
        // The same estimates with iid standard errors: singleton clusters.
        let single = data
            .clone()
            .with_clusters((0..1500).map(|i| i as i64).collect())
            .unwrap();
        let fit_iid = model.fit(&single).unwrap();
        for (counts, fit) in [(&mut clustered, &fit), (&mut iid, &fit_iid)] {
            let ci = fit
                .aggregate(Aggregation::Group)
                .unwrap()
                .effects()
                .confint(0.95)
                .unwrap();
            for (j, truth) in [1.25, 1.0].into_iter().enumerate() {
                counts[j] += usize::from(ci[[j, 0]] <= truth && truth <= ci[[j, 1]]);
            }
        }
    }
    for j in 0..2 {
        let (c, i) = (
            clustered[j] as f64 / reps as f64,
            iid[j] as f64 / reps as f64,
        );
        assert!(c >= 0.85, "cohort {j}: clustered coverage {c}");
        assert!(
            i < c - 0.15,
            "cohort {j}: iid coverage {i} vs clustered {c}"
        );
    }
}

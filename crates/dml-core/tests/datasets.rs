//! Simulated data sets: each design's effect is recovered by the matching
//! estimator, and the data have the documented structure.

use dml_core::datasets::{
    ConfoundedIrm, DidCs2021, DidSz2020, IivmData, IrmData, IrmDiscreteTreatments, LplrLzz2020,
    PlivChs2015, PlivMultiwayClusterCkms2021, PlprCp2025, PlrCcddhnr2018, SimpleRdd, SsmData,
};
use dml_core::{
    learner::{LogisticRegression, Ols},
    Aggregation, Apos, Did, Iivm, Irm, Lplr, Pliv, Plpr, Plr, RdFlex, Ssm, SsmScore, StaggeredDid,
};

/// `|estimate − truth| < 4 se`.
fn assert_recovers(what: &str, estimate: f64, se: f64, truth: f64) {
    assert!(
        (estimate - truth).abs() < 4.0 * se,
        "{what}: estimate {estimate} (se {se}), truth {truth}"
    );
}

#[test]
fn partially_linear_designs() {
    let sim = PlrCcddhnr2018 {
        n_obs: 2000,
        dim_x: 10,
        ..Default::default()
    }
    .generate(1)
    .unwrap();
    let fit = Plr::new(Ols::new(), Ols::new())
        .fit(&sim.to_data().unwrap())
        .unwrap();
    assert_recovers("PLR", fit.coef()[0], fit.se()[0], 0.5);

    let sim = PlivChs2015 {
        n_obs: 2000,
        dim_x: 20,
        dim_z: 1,
        ..Default::default()
    }
    .generate(2)
    .unwrap();
    let fit = Pliv::new(Ols::new(), Ols::new(), Ols::new())
        .fit(&sim.to_data().unwrap())
        .unwrap();
    assert_recovers("PLIV", fit.coef()[0], fit.se()[0], 1.0);

    let sim = PlivMultiwayClusterCkms2021 {
        n: 30,
        m: 30,
        dim_x: 10,
        ..Default::default()
    }
    .generate(3)
    .unwrap();
    assert_eq!(sim.clusters.as_ref().unwrap().row(31).to_vec(), vec![1, 1]);
    let fit = Pliv::new(Ols::new(), Ols::new(), Ols::new())
        .fit(&sim.to_data().unwrap())
        .unwrap();
    assert!(fit.n_clusters().is_some());
    assert_recovers("PLIV two-way clustered", fit.coef()[0], fit.se()[0], 1.0);

    let (sim, p) = LplrLzz2020 {
        n_obs: 3000,
        ..Default::default()
    }
    .generate(4)
    .unwrap();
    assert!(p.iter().all(|&q| q > 0.0 && q < 1.0));
    let fit = Lplr::new(LogisticRegression::default(), Ols::new(), Ols::new())
        .fit(&sim.to_data().unwrap())
        .unwrap();
    // Linear nuisances approximate r₀ only roughly: a loose check.
    assert!((fit.coef()[0] - 0.5).abs() < 0.5, "LPLR {}", fit.coef()[0]);

    let panel = PlprCp2025 {
        num_id: 300,
        num_t: 6,
        dim_x: 5,
        ..Default::default()
    }
    .generate(5)
    .unwrap();
    let fit = Plpr::new(Ols::new(), Ols::new())
        .fit(&panel.to_static_panel().unwrap())
        .unwrap();
    assert_recovers("PLPR", fit.coef()[0], fit.se()[0], 0.5);
}

#[test]
fn interactive_designs() {
    let sim = IrmData {
        n_obs: 3000,
        dim_x: 10,
        theta: 0.5,
        ..Default::default()
    }
    .generate(6)
    .unwrap();
    let fit = Irm::new(Ols::new(), LogisticRegression::default())
        .fit(&sim.to_data().unwrap())
        .unwrap();
    assert_recovers("IRM", fit.coef()[0], fit.se()[0], 0.5);

    let sim = IivmData {
        n_obs: 4000,
        dim_x: 10,
        alpha_x: 1.0,
        ..Default::default()
    }
    .generate(7)
    .unwrap();
    let fit = Iivm::new(
        Ols::new(),
        LogisticRegression::default(),
        LogisticRegression::default(),
    )
    .fit(&sim.to_data().unwrap())
    .unwrap();
    assert_recovers("IIVM", fit.coef()[0], fit.se()[0], 1.0);

    let sim = SsmData {
        n_obs: 3000,
        dim_x: 10,
        mar: false,
        ..Default::default()
    }
    .generate(8)
    .unwrap();
    let fit = Ssm::new(
        Ols::new(),
        LogisticRegression::default(),
        LogisticRegression::default(),
    )
    .score(SsmScore::Nonignorable)
    .fit(&sim.to_data().unwrap())
    .unwrap();
    assert_recovers("SSM", fit.coef()[0], fit.se()[0], 1.0);

    let conf = ConfoundedIrm {
        n_obs: 2000,
        theta: 5.0,
        ..Default::default()
    }
    .generate(9)
    .unwrap();
    let o = &conf.oracle;
    assert!(o.cf_y > 0.0 && o.cf_y < 1.0 && o.cf_d_ate > 0.0 && o.rho_ate.abs() <= 1.0);
    let d = &conf.data;
    let y = &d.d * &o.y_1 + &(&d.d.mapv(|v| 1.0 - v) * &o.y_0);
    assert!(y.iter().zip(&d.y).all(|(a, b)| (a - b).abs() < 1e-12));

    let disc = IrmDiscreteTreatments {
        n_obs: 3000,
        ..Default::default()
    }
    .generate(10)
    .unwrap();
    let levels = [0.0, 1.0, 2.0, 3.0];
    assert!(disc.data.d.iter().all(|v| levels.contains(v)));
    let apos = Apos::new(Ols::new(), LogisticRegression::default(), levels.to_vec())
        .fit(&disc.data.to_data().unwrap())
        .unwrap();
    let contrast = apos.causal_contrast(&[0.0]).unwrap();
    assert_eq!(contrast.n_treat(), 3);
}

#[test]
fn difference_in_differences_designs() {
    let sim = DidSz2020 {
        n_obs: 2000,
        dgp_type: 5,
        ..Default::default()
    }
    .generate(11)
    .unwrap();
    let fit = Did::new(Ols::new(), LogisticRegression::default())
        .fit(&sim.to_data().unwrap())
        .unwrap();
    // The ATT of the Sant'Anna–Zhao design is 0.
    assert_recovers("DID", fit.coef()[0], fit.se()[0], 0.0);
    let cs = DidSz2020 {
        n_obs: 2000,
        ..Default::default()
    }
    .generate_cross_sections(11)
    .unwrap();
    assert!(cs.t.as_ref().unwrap().iter().all(|&t| t == 0.0 || t == 1.0));

    let staggered = DidCs2021 {
        n_obs: 1500,
        ..Default::default()
    };
    let panel = staggered.generate(12).unwrap();
    let fit = StaggeredDid::new(Ols::new(), LogisticRegression::default())
        .fit(&panel.to_panel().unwrap())
        .unwrap();
    // Event-time effects grow with exposure (1, 2, 3 at event times 0, 1, 2, plus 0.1·x₀·e).
    let es = fit.aggregate(Aggregation::EventStudy).unwrap();
    let names = es.effects().treatment_names();
    let at = |e: &str| names.iter().position(|n| n == e).unwrap();
    for (e, truth) in [("0", 1.0), ("1", 2.0)] {
        let j = at(e);
        assert_recovers(
            &format!("event time {e}"),
            es.effects().coef()[j],
            es.effects().se()[j],
            truth,
        );
    }
    let cross = staggered.generate_cross_sections(13, 0.5).unwrap();
    assert!(cross.to_cross_sections().is_ok());
    let anticipating = DidCs2021 {
        n_obs: 300,
        anticipation_periods: 1,
        ..Default::default()
    }
    .generate(14)
    .unwrap();
    assert!(anticipating.cohort.iter().any(|g| g.is_infinite()));
    assert_eq!(anticipating.t.iter().fold(0.0_f64, |m, &t| m.max(t)), 4.0);
}

#[test]
fn regression_discontinuity_design() {
    let sim = SimpleRdd {
        n_obs: 4000,
        fuzzy: false,
        ..Default::default()
    }
    .generate(15)
    .unwrap();
    let fit = RdFlex::sharp(Ols::new())
        .fit(&sim.to_data().unwrap(), sim.score.view())
        .unwrap();
    assert_recovers("RDFlex", fit.coef()[2], fit.se()[2], 1.0);
}

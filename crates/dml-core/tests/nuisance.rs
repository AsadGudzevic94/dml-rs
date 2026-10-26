//! Per-nuisance learners (`nuisance_regressor` / `nuisance_classifier`),
//! user-defined linear scores and sequential cross-fitting.

use dml_core::datasets::{IivmData, IrmData};
use dml_core::{DmlFit, Iivm, Irm, Lasso, LogisticRegression, Ols};

#[test]
fn per_nuisance_learners_replace_only_their_nuisance() {
    let data = IrmData {
        n_obs: 600,
        dim_x: 5,
        theta: 0.5,
        ..Default::default()
    }
    .generate(1)
    .unwrap()
    .to_data()
    .unwrap();
    let base = Irm::new(Ols::new(), LogisticRegression::default())
        .fit(&data)
        .unwrap();
    // The same learners as overrides reproduce the fit exactly.
    let same = Irm::new(Ols::new(), LogisticRegression::default())
        .nuisance_regressor("ml_g0", Ols::new())
        .nuisance_regressor("ml_g1", Ols::new())
        .nuisance_classifier("ml_m", LogisticRegression::default())
        .fit(&data)
        .unwrap();
    assert_eq!(base.coef(), same.coef());
    // Another learner for ml_g1 changes ml_g1 only.
    let other = Irm::new(Ols::new(), LogisticRegression::default())
        .nuisance_regressor("ml_g1", Lasso::new(0.5))
        .fit(&data)
        .unwrap();
    assert_eq!(base.predictions("ml_g0"), other.predictions("ml_g0"));
    assert_eq!(base.predictions("ml_m"), other.predictions("ml_m"));
    assert_ne!(base.predictions("ml_g1"), other.predictions("ml_g1"));
    // Unknown names and wrong learner types are rejected.
    assert!(Irm::new(Ols::new(), LogisticRegression::default())
        .nuisance_regressor("ml_l", Ols::new())
        .fit(&data)
        .is_err());
    assert!(Irm::new(Ols::new(), LogisticRegression::default())
        .nuisance_classifier("ml_g0", LogisticRegression::default())
        .fit(&data)
        .is_err());
    let iv = IivmData {
        n_obs: 600,
        dim_x: 5,
        alpha_x: 1.0,
        ..Default::default()
    }
    .generate(2)
    .unwrap()
    .to_data()
    .unwrap();
    let a = Iivm::new(
        Ols::new(),
        LogisticRegression::default(),
        LogisticRegression::default(),
    )
    .fit(&iv)
    .unwrap();
    let b = Iivm::new(
        Ols::new(),
        LogisticRegression::default(),
        LogisticRegression::default(),
    )
    .nuisance_classifier("ml_r1", LogisticRegression::new(0.01))
    .fit(&iv)
    .unwrap();
    assert_eq!(a.predictions("ml_r0"), b.predictions("ml_r0"));
    assert_ne!(a.predictions("ml_r1"), b.predictions("ml_r1"));
}

#[test]
fn user_scores_and_sequential_folds() {
    let data = IrmData {
        n_obs: 500,
        dim_x: 4,
        theta: 0.5,
        ..Default::default()
    }
    .generate(3)
    .unwrap()
    .to_data()
    .unwrap();
    let fit = Irm::new(Ols::new(), LogisticRegression::default())
        .fit(&data)
        .unwrap();
    // The fit's own score elements give back the same estimate.
    let (psi_a, psi_b) = fit.psi_elements();
    let again = DmlFit::from_linear_scores(
        "IRM",
        vec!["d".into()],
        5,
        psi_a.to_owned(),
        psi_b.to_owned(),
    )
    .unwrap();
    assert!((again.coef()[0] - fit.coef()[0]).abs() < 1e-12);
    assert!((again.se()[0] - fit.se()[0]).abs() < 1e-12);
    assert!(
        DmlFit::from_linear_scores("x", vec![], 5, psi_a.to_owned(), psi_b.to_owned()).is_err()
    );
    // Sequential folds give identical results.
    dml_core::set_parallel_folds(false);
    let seq = Irm::new(Ols::new(), LogisticRegression::default()).fit(&data);
    dml_core::set_parallel_folds(true);
    assert_eq!(seq.unwrap().coef(), fit.coef());
}

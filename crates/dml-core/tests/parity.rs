//! Numerical parity with Python DoubleML on identical data and folds.
//! Fixtures come from `parity/generate_fixtures.py`.

mod common;

use common::{assert_all_close, assert_close, Fixture};
use dml_core::{DmlData, Plr};

const TOL: f64 = 1e-9;

fn check_plr(name: &str) {
    let fx = Fixture::load(name);
    let data = DmlData::new(fx.vec("y"), fx.vec("d"), fx.matrix("x")).unwrap();
    let folds = fx.folds();
    let learner = fx.regressor();
    let res = Plr::new(learner.clone(), learner)
        .fit_with_folds(&data, &folds[0])
        .unwrap();

    assert_all_close("ml_l", &res.l_hat, &fx.prediction("ml_l", 0), TOL);
    assert_all_close("ml_m", &res.m_hat, &fx.prediction("ml_m", 0), TOL);
    assert_close("coef", res.coef, fx.expected_scalar("coef"), TOL);
    assert_close("se", res.se, fx.expected_scalar("se"), TOL);
    assert_close("t", res.t_stat(), fx.expected_scalar("t_stat"), TOL);
    assert_close("pval", res.p_value(), fx.expected_scalar("pval"), TOL);
    let ci = &fx.expected("ci95")[0];
    let (lo, hi) = res.confint(0.95).unwrap();
    assert_close("ci lower", lo, ci[0].as_f64().unwrap(), TOL);
    assert_close("ci upper", hi, ci[1].as_f64().unwrap(), TOL);
}

#[test]
fn plr_partialling_out_ols_matches_doubleml() {
    check_plr("plr_ols");
}

#[test]
fn plr_partialling_out_ridge_matches_doubleml() {
    check_plr("plr_ridge");
}

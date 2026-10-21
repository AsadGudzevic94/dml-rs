//! Numerical parity with Python DoubleML on identical data and folds.
//! Fixtures come from `parity/generate_fixtures.py`.

mod common;

use common::{assert_all_close, assert_close, to_matrix, to_vec, Fixture};
use dml_core::{DmlFit, Irm, IrmScore, PAdjust, Pliv, Plr, PlrScore};
use ndarray::Array2;

/// Closed-form learners (OLS, ridge) and the DML algebra agree to ~1e-12;
/// iterative learners (Lasso, logistic) to their solver tolerance.
const TOL: f64 = 1e-9;
const TOL_ITERATIVE: f64 = 1e-7;

fn fit_fixture(fx: &Fixture) -> DmlFit {
    let data = fx.data();
    let folds = fx.folds();
    match fx.0["model"].as_str().unwrap() {
        "plr" => {
            let score = match fx.option_str("score") {
                "partialling out" => PlrScore::PartiallingOut,
                "IV-type" => PlrScore::IvType,
                s => panic!("unknown PLR score {s}"),
            };
            let mut plr = Plr::from_boxed(fx.regressor("ml_l"), fx.regressor("ml_m")).score(score);
            if score == PlrScore::IvType {
                plr = plr.ml_g_boxed(fx.regressor("ml_g"));
            }
            plr.fit_with_folds(&data, &folds)
        }
        "irm" => {
            let score = match fx.option_str("score") {
                "ATE" => IrmScore::Ate,
                "ATTE" => IrmScore::Atte,
                s => panic!("unknown IRM score {s}"),
            };
            Irm::from_boxed(fx.regressor("ml_g"), fx.classifier("ml_m"))
                .score(score)
                .normalize_ipw(fx.option_bool("normalize_ipw"))
                .fit_with_folds(&data, &folds)
        }
        "pliv" => Pliv::from_boxed(
            fx.regressor("ml_l"),
            fx.regressor("ml_m"),
            fx.regressor("ml_r"),
        )
        .fit_with_folds(&data, &folds),
        m => panic!("unknown model {m}"),
    }
    .unwrap()
}

fn check(name: &str, tol: f64) {
    let fx = Fixture::load(name);
    let fit = fit_fixture(&fx);
    let (k, n_rep) = (fit.n_treat(), fit.n_rep());

    let preds = fx.expected("predictions").as_object().unwrap();
    for nuisance in preds.keys() {
        for j in 0..k {
            for r in 0..n_rep {
                let ours = fit
                    .prediction(nuisance, r, j)
                    .unwrap_or_else(|| panic!("missing prediction {nuisance}"));
                assert_all_close(
                    &format!("{name} {nuisance} d{j} rep{r}"),
                    &ours,
                    &fx.prediction(nuisance, j, r),
                    tol,
                );
            }
        }
    }

    let all_coef = to_matrix(fx.expected("all_coef"));
    let all_se = to_matrix(fx.expected("all_se"));
    for j in 0..k {
        for r in 0..n_rep {
            assert_close(
                &format!("{name} all_coef[{j},{r}]"),
                fit.all_coef()[[j, r]],
                all_coef[[j, r]],
                tol,
            );
            assert_close(
                &format!("{name} all_se[{j},{r}]"),
                fit.all_se()[[j, r]],
                all_se[[j, r]],
                tol,
            );
        }
    }
    let vec_checks = [
        ("coef", fit.coef().to_owned()),
        ("se", fit.se().to_owned()),
        ("t_stat", fit.t_stat()),
        ("pval", fit.p_value()),
        ("p_bonferroni", fit.p_adjust(PAdjust::Bonferroni)),
        ("p_holm", fit.p_adjust(PAdjust::Holm)),
    ];
    for (key, ours) in vec_checks {
        assert_all_close(
            &format!("{name} {key}"),
            &ours,
            &to_vec(fx.expected(key)),
            tol,
        );
    }
    assert_matrix_close(
        &format!("{name} ci95"),
        &fit.confint(0.95).unwrap(),
        &to_matrix(fx.expected("ci95")),
        tol,
    );

    let boot = fx.expected("boot");
    if !boot.is_null() {
        let weights: Vec<Array2<f64>> = boot["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(to_matrix)
            .collect();
        let b = fit.bootstrap_with_weights(&weights).unwrap();
        assert_matrix_close(
            &format!("{name} joint ci95"),
            &b.joint_confint(0.95).unwrap(),
            &to_matrix(&boot["joint_ci95"]),
            tol,
        );
        assert_all_close(
            &format!("{name} romano-wolf"),
            &b.romano_wolf(),
            &to_vec(&boot["romano_wolf"]),
            tol,
        );
    }
}

fn assert_matrix_close(what: &str, a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.dim(), b.dim(), "{what}: shape");
    for ((idx, &x), &y) in a.indexed_iter().zip(b.iter()) {
        assert_close(&format!("{what}{idx:?}"), x, y, tol);
    }
}

#[test]
fn plr_partialling_out_ols() {
    check("plr_ols", TOL);
}

#[test]
fn plr_partialling_out_ridge() {
    check("plr_ridge", TOL);
}

#[test]
fn plr_partialling_out_lasso() {
    check("plr_lasso", TOL_ITERATIVE);
}

#[test]
fn plr_iv_type_repeated_splits() {
    check("plr_ivtype_rep3", TOL);
}

#[test]
fn plr_bayes_bootstrap() {
    check("plr_boot_bayes", TOL);
}

#[test]
fn plr_wild_bootstrap() {
    check("plr_boot_wild", TOL);
}

#[test]
fn plr_multiple_treatments_repeated_splits_bootstrap() {
    check("plr_multi_rep3_boot", TOL);
}

#[test]
fn irm_ate_repeated_splits() {
    check("irm_ate", TOL_ITERATIVE);
}

#[test]
fn irm_atte() {
    check("irm_atte", TOL_ITERATIVE);
}

#[test]
fn irm_ate_normalized_ipw() {
    check("irm_ate_normalized", TOL_ITERATIVE);
}

#[test]
fn pliv_partialling_out_repeated_splits() {
    check("pliv_ols", TOL);
}

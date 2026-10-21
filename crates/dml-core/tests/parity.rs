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

#[test]
fn irm_cate_blp_and_gate() {
    check("irm_cate_gate", TOL_ITERATIVE);
    let fx = Fixture::load("irm_cate_gate");
    let fit = fit_fixture(&fx);

    let blp_fx = &fx.expected("blp");
    let basis = to_matrix(&blp_fx["basis"]);
    let blp = fit.cate(basis.view()).unwrap();
    assert_all_close(
        "blp coef",
        blp.coef(),
        &to_vec(&blp_fx["coef"]),
        TOL_ITERATIVE,
    );
    assert_all_close("blp se", blp.se(), &to_vec(&blp_fx["se"]), TOL_ITERATIVE);
    // DoubleML's coefficient table is [lower, effect, upper].
    let ci_coef = to_matrix(&blp_fx["ci_coef"]);
    let ours = blp.confint(0.95).unwrap();
    for j in 0..ours.nrows() {
        assert_close("blp ci lower", ours[[j, 0]], ci_coef[[j, 0]], TOL_ITERATIVE);
        assert_close("blp ci upper", ours[[j, 1]], ci_coef[[j, 2]], TOL_ITERATIVE);
    }
    let first10 = basis.slice(ndarray::s![..10, ..]);
    assert_matrix_close(
        "blp ci at basis rows",
        &blp.confint_at(first10, 0.95).unwrap(),
        &to_matrix(&blp_fx["ci_at_first10"]),
        TOL_ITERATIVE,
    );

    let gate_fx = &fx.expected("gate");
    let gate = fit.gate(to_matrix(&gate_fx["groups"]).view()).unwrap();
    assert_all_close(
        "gate coef",
        gate.coef(),
        &to_vec(&gate_fx["coef"]),
        TOL_ITERATIVE,
    );
    assert_all_close("gate se", gate.se(), &to_vec(&gate_fx["se"]), TOL_ITERATIVE);
    let identity = Array2::eye(2);
    assert_matrix_close(
        "gate ci",
        &gate.confint_at(identity.view(), 0.95).unwrap(),
        &to_matrix(&gate_fx["ci"]),
        TOL_ITERATIVE,
    );
}

#[test]
fn dr_learner_pseudo_outcomes_equal_irm_orthogonal_signal() {
    use dml_core::{DrLearner, Ols};
    let fx = Fixture::load("irm_ate");
    let fit = fit_fixture(&fx);
    let folds = fx.folds();
    let dr = DrLearner::from_boxed(
        fx.regressor("ml_g"),
        fx.classifier("ml_m"),
        Box::new(Ols::new()),
    )
    .fit_with_folds(&fx.data(), &folds[0])
    .unwrap();
    let psi_b = fit.psi_elements().1.slice(ndarray::s![.., 0, 0]).to_owned();
    assert_all_close(
        "pseudo-outcome",
        &dr.pseudo_outcomes().unwrap().to_owned(),
        &psi_b,
        1e-12,
    );
}

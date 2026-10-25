//! Numerical parity with Python DoubleML on identical data and folds.
//! Fixtures come from `parity/generate_fixtures.py`.

mod common;

use common::{assert_all_close, assert_close, to_matrix, to_vec, Fixture};
use dml_core::{
    Apo, Apos, Cvar, Did, DidCs, DidScore, DmlFit, Iivm, Irm, IrmScore, Lpq, PAdjust, Pliv, Plr,
    PlrScore, Pq, Qte, Ssm, SsmScore,
};
use ndarray::Array2;

/// Closed-form learners (OLS, ridge) and the DML algebra agree to ~1e-12;
/// iterative learners (Lasso, logistic) to their solver tolerance.
const TOL: f64 = 1e-9;
const TOL_ITERATIVE: f64 = 1e-7;

fn fit_fixture(fx: &Fixture) -> DmlFit {
    fit_fixture_on(fx, &fx.data())
}

fn fit_fixture_on(fx: &Fixture, data: &dml_core::DmlData) -> DmlFit {
    let data = data.clone();
    let folds = fx.folds();
    let cluster_folds = fx.cluster_folds();
    // Clustered fixtures replay DoubleML's cluster splits (with their test clusters).
    macro_rules! fit {
        ($model:expr) => {
            match &cluster_folds {
                Some(cf) => $model.fit_with_cluster_folds(&data, cf),
                None => $model.fit_with_folds(&data, &folds),
            }
        };
    }
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
            fit!(plr)
        }
        "irm" => {
            let score = match fx.option_str("score") {
                "ATE" => IrmScore::Ate,
                "ATTE" => IrmScore::Atte,
                s => panic!("unknown IRM score {s}"),
            };
            let irm = Irm::from_boxed(fx.regressor("ml_g"), fx.classifier("ml_m"))
                .score(score)
                .normalize_ipw(fx.option_bool("normalize_ipw"));
            fit!(irm)
        }
        "pliv" => {
            let pliv = Pliv::from_boxed(
                fx.regressor("ml_l"),
                fx.regressor("ml_m"),
                fx.regressor("ml_r"),
            );
            fit!(pliv)
        }
        "iivm" => {
            let sub = &fx.0["options"]["subgroups"];
            let flag = |k: &str| sub[k].as_bool().unwrap_or(true);
            Iivm::from_boxed(
                fx.regressor("ml_g"),
                fx.classifier("ml_m"),
                fx.classifier("ml_r"),
            )
            .subgroups(flag("always_takers"), flag("never_takers"))
            .fit_with_folds(&data, &folds)
        }
        kind @ ("pq" | "lpq" | "cvar") => {
            let t = fx.0["options"]["treatment"].as_u64().unwrap() as u8;
            let q = fx.0["options"]["quantile"].as_f64().unwrap();
            let norm = fx.option_bool("normalize_ipw");
            match kind {
                "pq" => Pq::from_boxed(fx.classifier("ml_g"), fx.classifier("ml_m"))
                    .treatment(t)
                    .quantile(q)
                    .normalize_ipw(norm)
                    .fit_with_folds(&data, &folds),
                "lpq" => Lpq::from_boxed(fx.classifier("ml_g"), fx.classifier("ml_m"))
                    .treatment(t)
                    .quantile(q)
                    .normalize_ipw(norm)
                    .fit_with_folds(&data, &folds),
                _ => Cvar::from_boxed(fx.regressor("ml_g"), fx.classifier("ml_m"))
                    .treatment(t)
                    .quantile(q)
                    .normalize_ipw(norm)
                    .fit_with_folds(&data, &folds),
            }
        }
        "ssm" => {
            let score = match fx.option_str("score") {
                "missing-at-random" => SsmScore::MissingAtRandom,
                "nonignorable" => SsmScore::Nonignorable,
                s => panic!("unknown SSM score {s}"),
            };
            let ssm = Ssm::from_boxed(
                fx.regressor("ml_g"),
                fx.classifier("ml_pi"),
                fx.classifier("ml_m"),
            )
            .score(score)
            .normalize_ipw(fx.option_bool("normalize_ipw"));
            fit!(ssm)
        }
        "apo" => {
            let apo = Apo::from_boxed(
                fx.regressor("ml_g"),
                fx.classifier("ml_m"),
                fx.0["options"]["treatment_level"].as_f64().unwrap(),
            )
            .normalize_ipw(fx.option_bool("normalize_ipw"));
            fit!(apo)
        }
        kind @ ("did" | "did_cs") => {
            let score = match fx.option_str("score") {
                "observational" => DidScore::Observational,
                "experimental" => DidScore::Experimental,
                s => panic!("unknown DID score {s}"),
            };
            let ml_m = fx.has_learner("ml_m").then(|| fx.classifier("ml_m"));
            let normalize = fx.option_bool("in_sample_normalization");
            if kind == "did" {
                Did::from_boxed(fx.regressor("ml_g"), ml_m)
                    .score(score)
                    .in_sample_normalization(normalize)
                    .fit_with_folds(&data, &folds)
            } else {
                DidCs::from_boxed(fx.regressor("ml_g"), ml_m)
                    .score(score)
                    .in_sample_normalization(normalize)
                    .fit_with_folds(&data, &folds)
            }
        }
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

    // Out-of-sample learner losses (DoubleML's nuisance_loss, shape (n_rep, n_coefs)).
    let losses = fx.expected("nuisance_loss").as_object().unwrap();
    assert!(
        !losses.is_empty(),
        "{name}: fixture without nuisance losses"
    );
    for (learner, expected) in losses {
        let expected = to_matrix(expected);
        let ours = fit
            .nuisance_loss(learner)
            .unwrap_or_else(|| panic!("{name}: missing nuisance loss {learner}"));
        for j in 0..k {
            for r in 0..n_rep {
                assert_close(
                    &format!("{name} loss {learner}[{j},{r}]"),
                    ours[[j, r]],
                    expected[[r, j]],
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

    // In `irm_ate_normalized` the propensities hit the clipping bounds (0.01 / 0.99), and with
    // normalised weights nu2 amplifies a 1e-8 relative change in the propensity by ~1e6: the
    // logistic solvers' tiny differences then show up at ~1e-5. Applied to DoubleML's own
    // propensities, the nu2 formula reproduces DoubleML exactly.
    let sensitivity_tol = if name == "irm_ate_normalized" {
        1e-4
    } else {
        tol
    };
    check_sensitivity(name, &fx, &fit, sensitivity_tol);

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
fn plr_with_cross_validated_lasso() {
    check("plr_lasso_cv", TOL_ITERATIVE);
}

#[test]
fn irm_with_cross_validated_lasso_and_logistic_regression() {
    check("irm_ate_lasso_cv_logit_cv", TOL_ITERATIVE);
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

#[test]
fn iivm_late_repeated_splits() {
    check("iivm_late_rep2", TOL_ITERATIVE);
}

#[test]
fn iivm_without_always_takers() {
    check("iivm_no_always_takers", TOL_ITERATIVE);
}

#[test]
fn plr_one_way_clusters_repeated_splits() {
    check("plr_cluster_one_way_rep2", TOL);
}

#[test]
fn irm_one_way_clusters() {
    check("irm_cluster_one_way", TOL_ITERATIVE);
}

#[test]
fn pliv_two_way_clusters() {
    check("pliv_cluster_two_way", TOL);
}

#[test]
fn plr_two_way_clusters() {
    check("plr_cluster_two_way", TOL);
}

#[test]
fn derived_test_clusters_match_doubleml_splits() {
    // Passing only the observation folds of clustered data derives each
    // fold's test clusters from its rows: the same fit as DoubleML's splits.
    for name in ["plr_cluster_one_way_rep2", "pliv_cluster_two_way"] {
        let fx = Fixture::load(name);
        let data = fx.data();
        let folds = fx.folds();
        let derived = if name.starts_with("plr") {
            Plr::from_boxed(fx.regressor("ml_l"), fx.regressor("ml_m"))
                .fit_with_folds(&data, &folds)
        } else {
            Pliv::from_boxed(
                fx.regressor("ml_l"),
                fx.regressor("ml_m"),
                fx.regressor("ml_r"),
            )
            .fit_with_folds(&data, &folds)
        }
        .unwrap();
        assert_all_close(
            &format!("{name} coef"),
            &derived.coef().to_owned(),
            &to_vec(fx.expected("coef")),
            TOL,
        );
        assert_all_close(
            &format!("{name} se"),
            &derived.se().to_owned(),
            &to_vec(fx.expected("se")),
            TOL,
        );
    }
}

#[test]
fn bootstrap_is_rejected_for_clustered_fits() {
    let fx = Fixture::load("plr_cluster_one_way_rep2");
    let fit = fit_fixture(&fx);
    assert!(fit
        .bootstrap(dml_core::BootstrapMethod::Normal, 10, 1)
        .is_err());
}

#[test]
fn pq_repeated_splits() {
    check("pq_t1_rep2", TOL_ITERATIVE);
}

#[test]
fn pq_treatment_zero() {
    check("pq_t0_q25", TOL_ITERATIVE);
}

#[test]
fn pq_without_normalization() {
    check("pq_t1_q75_nonorm", TOL_ITERATIVE);
}

#[test]
fn cvar_treated() {
    check("cvar_t1", TOL_ITERATIVE);
}

#[test]
fn cvar_treatment_zero() {
    check("cvar_t0_q25", TOL_ITERATIVE);
}

#[test]
fn lpq_with_instrument() {
    check("lpq_t1", TOL_ITERATIVE);
}

#[test]
fn qte_repeated_splits() {
    let fx = Fixture::load("qte_rep2");
    let quantiles = to_vec(&fx.0["options"]["quantiles"]).to_vec();
    let fit = Qte::from_boxed(
        "PQ",
        Some(fx.classifier("ml_g")),
        None,
        fx.classifier("ml_m"),
        quantiles,
    )
    .unwrap()
    .fit_with_folds(&fx.data(), &fx.folds())
    .unwrap();
    let e = fx.expected("effects");
    let effects = fit.effects();
    let (all_coef, all_se) = (to_matrix(&e["all_coef"]), to_matrix(&e["all_se"]));
    for j in 0..effects.n_treat() {
        for r in 0..effects.n_rep() {
            assert_close(
                &format!("qte all_coef[{j},{r}]"),
                effects.all_coef()[[j, r]],
                all_coef[[j, r]],
                TOL_ITERATIVE,
            );
            assert_close(
                &format!("qte all_se[{j},{r}]"),
                effects.all_se()[[j, r]],
                all_se[[j, r]],
                TOL_ITERATIVE,
            );
        }
    }
    assert_all_close(
        "qte coef",
        &effects.coef().to_owned(),
        &to_vec(&e["coef"]),
        TOL_ITERATIVE,
    );
    assert_all_close(
        "qte se",
        &effects.se().to_owned(),
        &to_vec(&e["se"]),
        TOL_ITERATIVE,
    );
    assert_matrix_close(
        "qte ci95",
        &effects.confint(0.95).unwrap(),
        &to_matrix(&e["ci95"]),
        TOL_ITERATIVE,
    );
    let potential = fit.potential();
    for (i, p) in fx
        .expected("potential")
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let (coef, se) = (to_vec(&p["coef"]), to_vec(&p["se"]));
        for t in 0..2 {
            assert_close(
                &format!("potential q{i} d{t} coef"),
                potential.coef()[2 * i + t],
                coef[t],
                TOL_ITERATIVE,
            );
            assert_close(
                &format!("potential q{i} d{t} se"),
                potential.se()[2 * i + t],
                se[t],
                TOL_ITERATIVE,
            );
        }
    }
}

#[test]
fn ssm_missing_at_random() {
    check("ssm_mar_rep2", TOL_ITERATIVE);
    check("ssm_mar_normalized", TOL_ITERATIVE);
}

#[test]
fn ssm_nonignorable() {
    check("ssm_nonignorable", TOL_ITERATIVE);
    check("ssm_nonignorable_normalized_rep2", TOL_ITERATIVE);
}

#[test]
fn apo_repeated_splits() {
    check("apo_level1_rep2", TOL_ITERATIVE);
}

#[test]
fn apo_normalized_ipw() {
    check("apo_level0_normalized", TOL_ITERATIVE);
}

/// Compare a fit with a DoubleML framework (coefficients, inference) and its
/// sensitivity scenarios.
fn check_framework(what: &str, fit: &DmlFit, e: &serde_json::Value, tol: f64) {
    let (k, n_rep) = (fit.n_treat(), fit.n_rep());
    if let Some(names) = e["names"].as_array() {
        let names: Vec<&str> = names.iter().map(|v| v.as_str().unwrap()).collect();
        assert_eq!(fit.treatment_names(), names.as_slice(), "{what} names");
    }
    let (all_coef, all_se) = (to_matrix(&e["all_coef"]), to_matrix(&e["all_se"]));
    for j in 0..k {
        for r in 0..n_rep {
            assert_close(
                &format!("{what} all_coef[{j},{r}]"),
                fit.all_coef()[[j, r]],
                all_coef[[j, r]],
                tol,
            );
            assert_close(
                &format!("{what} all_se[{j},{r}]"),
                fit.all_se()[[j, r]],
                all_se[[j, r]],
                tol,
            );
        }
    }
    assert_all_close(
        &format!("{what} coef"),
        &fit.coef().to_owned(),
        &to_vec(&e["coef"]),
        tol,
    );
    assert_all_close(
        &format!("{what} se"),
        &fit.se().to_owned(),
        &to_vec(&e["se"]),
        tol,
    );
    assert_all_close(
        &format!("{what} pval"),
        &fit.p_value(),
        &to_vec(&e["pval"]),
        tol,
    );
    assert_matrix_close(
        &format!("{what} ci95"),
        &fit.confint(0.95).unwrap(),
        &to_matrix(&e["ci95"]),
        tol,
    );
}

fn check_framework_sensitivity(what: &str, fit: &DmlFit, e: &serde_json::Value, tol: f64) {
    let (sigma2, nu2) = fit.sensitivity_elements().expect("sensitivity elements");
    assert_matrix_close(
        &format!("{what} sigma2"),
        &sigma2.to_owned(),
        &to_matrix(&e["sigma2"]),
        tol,
    );
    assert_matrix_close(
        &format!("{what} nu2"),
        &nu2.to_owned(),
        &to_matrix(&e["nu2"]),
        tol,
    );
    for (i, sc) in e["scenarios"].as_array().unwrap().iter().enumerate() {
        let p = &sc["params"];
        let params = dml_core::SensitivityParams::new()
            .cf_y(p["cf_y"].as_f64().unwrap())
            .cf_d(p["cf_d"].as_f64().unwrap())
            .rho(p["rho"].as_f64().unwrap())
            .level(p["level"].as_f64().unwrap())
            .null_hypotheses(to_vec(&p["null_hypothesis"]).to_vec());
        let res = fit.sensitivity_analysis(&params).unwrap();
        for (key, ours) in [
            ("theta_lower", &res.theta_lower),
            ("theta_upper", &res.theta_upper),
            ("se_lower", &res.se_lower),
            ("se_upper", &res.se_upper),
            ("ci_lower", &res.ci_lower),
            ("ci_upper", &res.ci_upper),
        ] {
            assert_all_close(
                &format!("{what} scenario {i} {key}"),
                ours,
                &to_vec(&sc[key]),
                tol,
            );
        }
        for (key, ours) in [("rv", &res.rv), ("rva", &res.rva)] {
            for (j, (&a, &b)) in ours.iter().zip(to_vec(&sc[key]).iter()).enumerate() {
                assert!(
                    (a - b).abs() < TOL_RV,
                    "{what} scenario {i} {key}[{j}]: rust {a}, doubleml {b}"
                );
            }
        }
    }
}

#[test]
fn apos_levels_and_causal_contrasts() {
    let fx = Fixture::load("apos_rep2");
    let levels = to_vec(&fx.0["options"]["treatment_levels"]).to_vec();
    let fit = Apos::from_boxed(fx.regressor("ml_g"), fx.classifier("ml_m"), levels.clone())
        .normalize_ipw(fx.option_bool("normalize_ipw"))
        .fit_with_folds(&fx.data(), &fx.folds())
        .unwrap();
    let tol = TOL_ITERATIVE;
    let apos = fit.apos();
    let e = fx.expected("levels");
    check_framework("apos levels", apos, e, tol);
    check_framework_sensitivity("apos levels", apos, fx.expected("levels_sensitivity"), tol);

    // Each level's nuisance predictions and learner losses.
    for (j, (preds, losses)) in fx
        .expected("predictions")
        .as_array()
        .unwrap()
        .iter()
        .zip(fx.expected("nuisance_loss").as_array().unwrap())
        .enumerate()
    {
        for (name, by_rep) in preds.as_object().unwrap() {
            for (r, theirs) in by_rep.as_array().unwrap().iter().enumerate() {
                let ours = apos.prediction(name, r, j).unwrap();
                assert_all_close(
                    &format!("apos level {j} {name} rep{r}"),
                    &ours,
                    &to_vec(theirs),
                    tol,
                );
            }
        }
        for (name, theirs) in losses.as_object().unwrap() {
            let theirs = to_matrix(theirs);
            for r in 0..apos.n_rep() {
                assert_close(
                    &format!("apos level {j} loss {name}"),
                    apos.nuisance_loss(name).unwrap()[[j, r]],
                    theirs[[r, 0]],
                    tol,
                );
            }
        }
    }

    for c in fx.expected("contrasts").as_array().unwrap() {
        let refs = to_vec(&c["reference_levels"]).to_vec();
        let contrast = fit.causal_contrast(&refs).unwrap();
        let what = format!("contrast vs {refs:?}");
        check_framework(&what, &contrast, c, tol);
        check_framework_sensitivity(&what, &contrast, &c["sensitivity"], tol);
    }
    assert!(fit.causal_contrast(&[7.0]).is_err());
}

#[test]
fn did_panel_observational() {
    check("did_panel_observational", TOL_ITERATIVE);
}

#[test]
fn did_panel_experimental_without_normalization() {
    check("did_panel_experimental_nonorm", TOL);
}

#[test]
fn did_cross_section_observational_without_normalization() {
    check("did_cs_observational_nonorm", TOL_ITERATIVE);
}

#[test]
fn did_cross_section_experimental() {
    check("did_cs_experimental", TOL);
}

/// Robustness values come from scipy's bounded minimiser in DoubleML
/// (tolerance ~1e-5); dml-rs solves the crossing exactly.
const TOL_RV: f64 = 1e-4;

fn check_sensitivity(name: &str, fx: &Fixture, fit: &DmlFit, tol: f64) {
    let e = fx.expected("sensitivity");
    if e.is_null() {
        assert!(
            fit.sensitivity_elements().is_none()
                || fx.0["model"] == "plr"
                || fx.0["model"] == "irm"
        );
        return;
    }
    let (sigma2, nu2) = fit.sensitivity_elements().expect("sensitivity elements");
    assert_matrix_close(
        &format!("{name} sigma2"),
        &sigma2.to_owned(),
        &to_matrix(&e["sigma2"]),
        tol,
    );
    assert_matrix_close(
        &format!("{name} nu2"),
        &nu2.to_owned(),
        &to_matrix(&e["nu2"]),
        tol,
    );
    for (i, sc) in e["scenarios"].as_array().unwrap().iter().enumerate() {
        let p = &sc["params"];
        let params = dml_core::SensitivityParams::new()
            .cf_y(p["cf_y"].as_f64().unwrap())
            .cf_d(p["cf_d"].as_f64().unwrap())
            .rho(p["rho"].as_f64().unwrap())
            .level(p["level"].as_f64().unwrap())
            .null_hypotheses(to_vec(&p["null_hypothesis"]).to_vec());
        let res = fit.sensitivity_analysis(&params).unwrap();
        let pairs = [
            ("theta_lower", &res.theta_lower),
            ("theta_upper", &res.theta_upper),
            ("se_lower", &res.se_lower),
            ("se_upper", &res.se_upper),
            ("ci_lower", &res.ci_lower),
            ("ci_upper", &res.ci_upper),
        ];
        for (key, ours) in pairs {
            assert_all_close(
                &format!("{name} scenario {i} {key}"),
                ours,
                &to_vec(&sc[key]),
                tol,
            );
        }
        for (key, ours) in [("rv", &res.rv), ("rva", &res.rva)] {
            let theirs = to_vec(&sc[key]);
            for (j, (&a, &b)) in ours.iter().zip(theirs.iter()).enumerate() {
                assert!(
                    (a - b).abs() < TOL_RV,
                    "{name} scenario {i} {key}[{j}]: rust {a}, doubleml {b}"
                );
            }
        }
    }
    let bm = &e["benchmark"];
    if !bm.is_null() {
        let drop: Vec<usize> = bm["drop"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let short = fit_fixture_on(fx, &fx.data().without_covariates(&drop).unwrap());
        let ours = fit.sensitivity_benchmark(&short).unwrap();
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
                tol,
            );
        }
    }
}

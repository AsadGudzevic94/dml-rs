//! Numerical parity with Python DoubleML on identical data and folds.
//! Fixtures come from `parity/generate_fixtures.py`.

mod common;

use common::{
    assert_all_close, assert_close, assert_matrix_close, check_framework_sensitivity, to_matrix,
    to_vec, Fixture,
};
use dml_core::{
    Apo, Apos, Cvar, Did, DidCs, DidScore, DmlFit, Iivm, Irm, IrmScore, Lplr, LplrScore, Lpq,
    PAdjust, PanelApproach, Pliv, PlivScore, PlivVariant, Plpr, Plr, PlrScore, Pq, Qte, Ssm,
    SsmScore, StaticPanel, TreatmentLearner,
};
use ndarray::{Array1, Array2};

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
                .normalize_ipw(fx.option_bool("normalize_ipw"))
                .calibration(fx.calibration());
            fit!(irm)
        }
        kind @ ("pliv" | "pliv_partial_xz") => {
            let mut pliv = Pliv::from_boxed(
                fx.regressor("ml_l"),
                fx.regressor("ml_m"),
                fx.regressor("ml_r"),
            );
            if kind == "pliv_partial_xz" {
                pliv = pliv.variant(PlivVariant::PartialXZ);
            }
            if fx.option_str("score") == "IV-type" {
                pliv = pliv
                    .score(PlivScore::IvType)
                    .ml_g_boxed(fx.regressor("ml_g"));
            }
            fit!(pliv)
        }
        "pliv_partial_z" => fit!(Pliv::partial_z_boxed(fx.regressor("ml_r"))),
        "iivm" => {
            let sub = &fx.0["options"]["subgroups"];
            let flag = |k: &str| sub[k].as_bool().unwrap_or(true);
            Iivm::from_boxed(
                fx.regressor("ml_g"),
                fx.classifier("ml_m"),
                fx.classifier("ml_r"),
            )
            .subgroups(flag("always_takers"), flag("never_takers"))
            .calibration(fx.calibration())
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
                    .calibration(fx.calibration())
                    .fit_with_folds(&data, &folds),
                "lpq" => Lpq::from_boxed(fx.classifier("ml_g"), fx.classifier("ml_m"))
                    .treatment(t)
                    .quantile(q)
                    .normalize_ipw(norm)
                    .calibration(fx.calibration())
                    .fit_with_folds(&data, &folds),
                _ => Cvar::from_boxed(fx.regressor("ml_g"), fx.classifier("ml_m"))
                    .treatment(t)
                    .quantile(q)
                    .normalize_ipw(norm)
                    .calibration(fx.calibration())
                    .fit_with_folds(&data, &folds),
            }
        }
        "plpr" => {
            let score = match fx.option_str("score") {
                "partialling out" => PlrScore::PartiallingOut,
                "IV-type" => PlrScore::IvType,
                s => panic!("unknown PLPR score {s}"),
            };
            let approach = match fx.option_str("approach") {
                "cre_general" => PanelApproach::CreGeneral,
                "cre_normal" => PanelApproach::CreNormal,
                "fd_exact" => PanelApproach::FdExact,
                "wg_approx" => PanelApproach::WgApprox,
                a => panic!("unknown PLPR approach {a}"),
            };
            let ids = fx.0["panel_id"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let panel = StaticPanel::new(
                fx.vec("y"),
                fx.vec("d"),
                fx.matrix("x"),
                ids,
                fx.vec("panel_t"),
            )
            .unwrap();
            let mut plpr = Plpr::from_boxed(
                fx.regressor("ml_l"),
                TreatmentLearner::Regressor(fx.regressor("ml_m")),
            )
            .score(score)
            .approach(approach);
            if score == PlrScore::IvType {
                plpr = plpr.ml_g_boxed(fx.regressor("ml_g"));
            }
            let fit = plpr.fit_with_cluster_folds(&panel, &cluster_folds.expect("cluster folds"));
            if let Ok(fit) = &fit {
                let names: Vec<String> = fx
                    .expected("treatment_names")
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                assert_eq!(fit.treatment_names(), names.as_slice());
            }
            fit
        }
        "lplr" => {
            let score = match fx.option_str("score") {
                "nuisance_space" => LplrScore::NuisanceSpace,
                "instrument" => LplrScore::Instrument,
                s => panic!("unknown LPLR score {s}"),
            };
            let ml_m = match fx.0["learners"]["ml_m"]["kind"].as_str().unwrap() {
                "logit" => TreatmentLearner::Classifier(fx.classifier("ml_m")),
                _ => TreatmentLearner::Regressor(fx.regressor("ml_m")),
            };
            Lplr::from_boxed(fx.classifier("ml_M"), fx.regressor("ml_t"), ml_m, None)
                .score(score)
                .fit_with_inner_folds(&data, &folds, &fx.inner_folds())
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
            .normalize_ipw(fx.option_bool("normalize_ipw"))
            .calibration(fx.calibration());
            fit!(ssm)
        }
        "apo" => {
            let apo = Apo::from_boxed(
                fx.regressor("ml_g"),
                fx.classifier("ml_m"),
                fx.0["options"]["treatment_level"].as_f64().unwrap(),
            )
            .normalize_ipw(fx.option_bool("normalize_ipw"))
            .calibration(fx.calibration());
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
                    .calibration(fx.calibration())
                    .fit_with_folds(&data, &folds)
            } else {
                DidCs::from_boxed(fx.regressor("ml_g"), ml_m)
                    .score(score)
                    .in_sample_normalization(normalize)
                    .calibration(fx.calibration())
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
fn pliv_ivtype_and_several_instruments() {
    check("pliv_ivtype", TOL);
    check("pliv_two_instruments_rep2", TOL);
    check("pliv_three_instruments", TOL);
}

#[test]
fn pliv_partial_z_and_xz() {
    check("pliv_partial_z", TOL);
    check("pliv_partial_xz", TOL);
}

#[test]
fn isotonic_propensity_calibration() {
    check("irm_ate_isotonic_normalized", TOL_ITERATIVE);
    check("irm_atte_isotonic_cv_rep2", TOL_ITERATIVE);
    check("apo_isotonic_cv", TOL_ITERATIVE);
    check("iivm_isotonic", TOL_ITERATIVE);
    check("ssm_mar_isotonic_cv", TOL_ITERATIVE);
}

#[test]
fn plpr_first_differences() {
    check("plpr_fd_exact_rep2", TOL);
    check("plpr_fd_exact_unbalanced", TOL);
}

#[test]
fn plpr_within_groups_and_correlated_random_effects() {
    check("plpr_wg_approx_unbalanced", TOL);
    check("plpr_cre_general", TOL);
    check("plpr_cre_normal_ivtype", TOL);
}

#[test]
fn lplr_continuous_treatment() {
    check("lplr_nuisance_space_rep2", TOL_ITERATIVE);
    check("lplr_instrument", TOL_ITERATIVE);
}

#[test]
fn lplr_binary_treatment() {
    check("lplr_binary_propensity", TOL_ITERATIVE);
    check("lplr_binary_instrument", TOL_ITERATIVE);
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
    // A framework's p-value is the median of the per-repetition p-values
    // (DoubleML's `DoubleMLFramework.pvals`; models use the aggregated t).
    let pval: ndarray::Array1<f64> = (0..k)
        .map(|j| {
            let mut p: Vec<f64> = (0..n_rep)
                .map(|r| {
                    2.0 * dml_core::stats::norm_sf(
                        (fit.all_coef()[[j, r]] / fit.all_se()[[j, r]]).abs(),
                    )
                })
                .collect();
            p.sort_by(f64::total_cmp);
            let m = p.len() / 2;
            if p.len() % 2 == 1 {
                p[m]
            } else {
                (p[m - 1] + p[m]) / 2.0
            }
        })
        .collect();
    assert_all_close(&format!("{what} pval"), &pval, &to_vec(&e["pval"]), tol);
    assert_matrix_close(
        &format!("{what} ci95"),
        &fit.confint(0.95).unwrap(),
        &to_matrix(&e["ci95"]),
        tol,
    );
}

/// DoubleMLFramework arithmetic: `+`, `−`, scalar `×` and `concat` of a PLR
/// and an IRM fit on the same sample splits.
#[test]
fn framework_arithmetic() {
    let fx = Fixture::load("framework_arithmetic_rep2");
    let (data, folds) = (fx.data(), fx.folds());
    let plr = Plr::from_boxed(fx.regressor("ml_l"), fx.regressor("ml_m"))
        .fit_with_folds(&data, &folds)
        .unwrap();
    let irm = Irm::from_boxed(fx.regressor("ml_g"), fx.classifier("ml_m_irm"))
        .fit_with_folds(&data, &folds)
        .unwrap();
    let tol = TOL_ITERATIVE;
    let combos = [
        ("sum", (&plr + &irm).unwrap()),
        ("difference", (&plr - &irm).unwrap()),
        ("scaled", 2.5 * &plr),
        (
            "concat",
            DmlFit::concat(&[plr.clone(), irm.clone()]).unwrap(),
        ),
    ];
    for (key, fit) in &combos {
        let e = fx.expected(key);
        check_framework(key, fit, e, tol);
        check_framework_sensitivity(key, fit, &e["sensitivity"], tol);
    }
    // DoubleML 0.11 also multiplies the standard error by a negative scalar.
    let negated = &irm * -0.5;
    let e = fx.expected("negated");
    assert_all_close(
        "negated coef",
        &negated.coef().to_owned(),
        &to_vec(&e["coef"]),
        tol,
    );
    assert_all_close(
        "negated se",
        &negated.se().to_owned(),
        &to_vec(&e["abs_se"]),
        tol,
    );
    check_framework_sensitivity("negated", &negated, &e["sensitivity"], tol);
    assert_all_close(
        "neg",
        &(-&irm).coef().to_owned(),
        &(-irm.coef().to_owned()),
        0.0,
    );
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

/// The RD back end of RDFlex against the `rdrobust` package: MSE-optimal
/// bandwidths, and estimates with selected and with manual bandwidths.
fn check_rd(name: &str, tol: f64) {
    let fx = Fixture::load(name);
    let (y, x) = (fx.vec("y"), fx.vec("x"));
    let fuzzy = fx.0["fuzzy"].as_array().map(|_| to_vec(&fx.0["fuzzy"]));
    let e = &fx.0["expected"];
    let (h, b) =
        dml_core::rd::rdbwselect(y.view(), x.view(), fuzzy.as_ref().map(|t| t.view()), 0.0)
            .unwrap();
    let bws = to_vec(&e["bws"]);
    assert_close(&format!("{name} h"), h, bws[0], tol);
    assert_close(&format!("{name} b"), b, bws[2], tol);
    let manual = to_vec(&fx.0["manual"]);
    for (key, bw) in [("auto", None), ("manual", Some((manual[0], manual[1])))] {
        let est = dml_core::rd::rdrobust(
            y.view(),
            x.view(),
            fuzzy.as_ref().map(|t| t.view()),
            0.0,
            bw,
        )
        .unwrap();
        let ex = &e[key];
        assert_all_close(
            &format!("{name} {key} coef"),
            &Array1::from(est.coef.to_vec()),
            &to_vec(&ex["coef"]),
            tol,
        );
        assert_all_close(
            &format!("{name} {key} se"),
            &Array1::from(est.se.to_vec()),
            &to_vec(&ex["se"]),
            tol,
        );
        let ci = to_matrix(&ex["ci"]);
        for k in 0..3 {
            assert_close(
                &format!("{name} {key} ci[{k}]"),
                est.ci[k][0],
                ci[[k, 0]],
                tol,
            );
            assert_close(
                &format!("{name} {key} ci[{k}]"),
                est.ci[k][1],
                ci[[k, 1]],
                tol,
            );
        }
        assert_close(
            &format!("{name} {key} h"),
            est.h.0,
            to_vec(&ex["h"])[0],
            tol,
        );
        assert_close(
            &format!("{name} {key} b"),
            est.b.0,
            to_vec(&ex["b"])[0],
            tol,
        );
        let n_h = to_vec(&ex["n_h"]);
        assert_eq!(
            (est.n_h.0 as f64, est.n_h.1 as f64),
            (n_h[0], n_h[1]),
            "{name} {key} N_h"
        );
    }
}

#[test]
fn rd_matches_rdrobust() {
    check_rd("rd_sharp", TOL);
    check_rd("rd_fuzzy", TOL);
    check_rd("rd_masspoints", TOL);
}

/// `rdrobust` keyword arguments as [`RdOptions`] and manual bandwidths.
fn rd_options_from(kw: &serde_json::Value) -> (dml_core::rd::RdOptions, Option<f64>, Option<f64>) {
    use dml_core::rd::{BwSelect, Kernel, MassPoints, RdOptions, Vce};
    let mut opts = RdOptions::default();
    let (mut h, mut b) = (None, None);
    for (key, v) in kw.as_object().unwrap() {
        match key.as_str() {
            "p" => opts.p = v.as_u64().unwrap() as usize,
            "q" => opts.q = Some(v.as_u64().unwrap() as usize),
            "kernel" => {
                opts.kernel = match v.as_str().unwrap() {
                    "epa" => Kernel::Epanechnikov,
                    "uni" => Kernel::Uniform,
                    _ => Kernel::Triangular,
                }
            }
            "bwselect" => {
                opts.bwselect = match v.as_str().unwrap() {
                    "msetwo" => BwSelect::Msetwo,
                    "msesum" => BwSelect::Msesum,
                    "msecomb1" => BwSelect::Msecomb1,
                    "msecomb2" => BwSelect::Msecomb2,
                    "cerrd" => BwSelect::Cerrd,
                    "certwo" => BwSelect::Certwo,
                    "cersum" => BwSelect::Cersum,
                    "cercomb1" => BwSelect::Cercomb1,
                    "cercomb2" => BwSelect::Cercomb2,
                    _ => BwSelect::Mserd,
                }
            }
            "vce" => {
                opts.vce = match v.as_str().unwrap() {
                    "hc0" => Vce::Hc0,
                    "hc1" => Vce::Hc1,
                    "hc2" => Vce::Hc2,
                    "hc3" => Vce::Hc3,
                    "cr1" => Vce::Cr1,
                    "cr2" => Vce::Cr2,
                    "cr3" => Vce::Cr3,
                    _ => Vce::Nn,
                }
            }
            "nnmatch" => opts.nnmatch = v.as_u64().unwrap() as usize,
            "level" => opts.level = v.as_f64().unwrap() / 100.0,
            "stdvars" => opts.stdvars = v.as_bool().unwrap(),
            "bwrestrict" => opts.bwrestrict = v.as_bool().unwrap(),
            "scaleregul" => opts.scaleregul = v.as_f64().unwrap(),
            "rho" => opts.rho = Some(v.as_f64().unwrap()),
            "scalepar" => opts.scalepar = v.as_f64().unwrap(),
            "bwcheck" => opts.bwcheck = Some(v.as_u64().unwrap() as usize),
            "sharpbw" => opts.sharpbw = v.as_bool().unwrap(),
            "masspoints" => {
                opts.masspoints = match v.as_str().unwrap() {
                    "off" => MassPoints::Off,
                    "check" => MassPoints::Check,
                    _ => MassPoints::Adjust,
                }
            }
            "h" => h = v.as_f64(),
            "b" => b = v.as_f64(),
            other => panic!("unknown option {other}"),
        }
    }
    (opts, h, b)
}

/// `rdrobust`'s options, covariates and clusters, case by case.
#[test]
fn rd_options_match_rdrobust() {
    use dml_core::rd::{rd_estimate, RdBandwidths, RdData};
    let fx = Fixture::load("rd_options");
    let cluster: Vec<i64> = fx.0["cluster"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    let (x, x_grid) = (fx.vec("x"), fx.vec("x_grid"));
    let (y_sharp, y_fuzzy, d_fuzzy) = (fx.vec("y_sharp"), fx.vec("y_fuzzy"), fx.vec("d_fuzzy"));
    let (z, z_collinear) = (fx.matrix("z"), fx.matrix("z_collinear"));
    let cases = fx.expected("cases").as_array().unwrap();
    assert_eq!(cases.len(), 54);
    let mut checked = 0;
    for case in cases {
        checked += 1;
        let kw = &case["kwargs"];
        let what = format!("rd_options {case}");
        let (opts, h, b) = rd_options_from(kw);
        let (y, xv, t) = match case["data"].as_str().unwrap_or("sharp") {
            "fuzzy" => (&y_fuzzy, &x, Some(&d_fuzzy)),
            "grid" => (&y_sharp, &x_grid, None),
            _ => (&y_sharp, &x, None),
        };
        let mut data = RdData::new(y.view(), xv.view());
        if let Some(t) = t {
            data = data.fuzzy(t.view());
        }
        match case["covs"].as_str() {
            Some("z") => data = data.covariates(z.view()),
            Some(_) => data = data.covariates(z_collinear.view()),
            None => {}
        }
        if case["cluster"].as_bool() == Some(true) {
            data = data.clusters(&cluster);
        }
        // rdrobust: a manual h without b sets b = h.
        let bws = h.map(|h| RdBandwidths::common(h, b.unwrap_or(h)));
        let est = rd_estimate(&data, &opts, bws).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_all_close(
            &format!("{what} coef"),
            &Array1::from(est.coef.to_vec()),
            &to_vec(&case["coef"]),
            TOL,
        );
        assert_all_close(
            &format!("{what} se"),
            &Array1::from(est.se.to_vec()),
            &to_vec(&case["se"]),
            TOL,
        );
        let ci = to_matrix(&case["ci"]);
        for k in 0..3 {
            assert_close(
                &format!("{what} ci lower {k}"),
                est.ci[k][0],
                ci[[k, 0]],
                TOL,
            );
            assert_close(
                &format!("{what} ci upper {k}"),
                est.ci[k][1],
                ci[[k, 1]],
                TOL,
            );
        }
        let (eh, eb) = (to_vec(&case["h"]), to_vec(&case["b"]));
        assert_close(&format!("{what} h left"), est.h.0, eh[0], TOL);
        assert_close(&format!("{what} h right"), est.h.1, eh[1], TOL);
        assert_close(&format!("{what} b left"), est.b.0, eb[0], TOL);
        assert_close(&format!("{what} b right"), est.b.1, eb[1], TOL);
    }
    assert_eq!(checked, cases.len());
}

fn check_rdflex(name: &str, tol: f64) {
    let fx = Fixture::load(name);
    let o = &fx.0["options"];
    let first_stage = match o["fs_specification"].as_str().unwrap() {
        "cutoff" => dml_core::FirstStage::Cutoff,
        "cutoff and score" => dml_core::FirstStage::CutoffAndScore,
        _ => dml_core::FirstStage::InteractedCutoffAndScore,
    };
    let kernel = match o["fs_kernel"].as_str().unwrap() {
        "uniform" => dml_core::RdKernel::Uniform,
        "triangular" => dml_core::RdKernel::Triangular,
        _ => dml_core::RdKernel::Epanechnikov,
    };
    let ml_m = o["fuzzy"].as_bool().unwrap().then(|| fx.classifier("ml_m"));
    let model = if o["binary_outcome"].as_bool() == Some(true) {
        dml_core::RdFlex::from_boxed_binary(fx.classifier("ml_g"), ml_m)
    } else {
        dml_core::RdFlex::from_boxed(fx.regressor("ml_g"), ml_m)
    };
    let (rd_options, _, _) = rd_options_from(&o["rd_kwargs"]);
    let mut model = model.rd_options(rd_options);
    if fx.0.get("rd_covs").is_some() {
        model = model.rd_covariates(fx.matrix("rd_covs"));
    }
    if let Some(cl) = fx.0["rd_cluster"].as_array() {
        model = model.rd_clusters(cl.iter().map(|v| v.as_i64().unwrap()).collect());
    }
    let mut model = model
        .cutoff(o["cutoff"].as_f64().unwrap())
        .first_stage(first_stage)
        .kernel(kernel)
        .n_iterations(o["n_iterations"].as_u64().unwrap() as usize);
    if let Some(h) = o["h_fs"].as_f64() {
        model = model.h_fs(h);
    }
    let fit = model
        .fit_with_folds(&fx.data(), fx.vec("score").view(), &fx.folds())
        .unwrap();
    let e = fx.expected("coef");
    assert_all_close(
        &format!("{name} coef"),
        &Array1::from(fit.coef().to_vec()),
        &to_vec(e),
        tol,
    );
    assert_all_close(
        &format!("{name} se"),
        &Array1::from(fit.se().to_vec()),
        &to_vec(fx.expected("se")),
        tol,
    );
    assert_all_close(
        &format!("{name} t"),
        &Array1::from(fit.t_stat().to_vec()),
        &to_vec(fx.expected("t_stat")),
        tol,
    );
    assert_all_close(
        &format!("{name} p"),
        &Array1::from(fit.p_value().to_vec()),
        &to_vec(fx.expected("pval")),
        tol,
    );
    assert_matrix_close(
        &format!("{name} all_coef"),
        &fit.all_coef(),
        &to_matrix(fx.expected("all_coef")),
        tol,
    );
    assert_matrix_close(
        &format!("{name} all_se"),
        &fit.all_se(),
        &to_matrix(fx.expected("all_se")),
        tol,
    );
    let as_matrix = |m: [[f64; 2]; 3]| Array2::from_shape_fn((3, 2), |(i, j)| m[i][j]);
    assert_matrix_close(
        &format!("{name} ci"),
        &as_matrix(fit.ci()),
        &to_matrix(fx.expected("ci")),
        tol,
    );
    assert_matrix_close(
        &format!("{name} confint90"),
        &as_matrix(fit.confint(0.9).unwrap()),
        &to_matrix(fx.expected("confint90")),
        tol,
    );
    assert_close(
        &format!("{name} h_fs"),
        fit.h_fs(),
        fx.expected("h_fs").as_f64().unwrap(),
        tol,
    );
    // DoubleML records no final bandwidth with a single round.
    if !fx.expected("h").is_null() {
        assert_all_close(
            &format!("{name} h"),
            &Array1::from(fit.h()),
            &to_vec(fx.expected("h")),
            tol,
        );
    }
}

#[test]
fn rdflex_sharp_designs() {
    check_rdflex("rdflex_sharp", TOL_ITERATIVE);
    check_rdflex("rdflex_sharp_score_uniform", TOL_ITERATIVE);
}

#[test]
fn rdflex_fuzzy_designs() {
    check_rdflex("rdflex_fuzzy_rep2", TOL_ITERATIVE);
    check_rdflex("rdflex_fuzzy_interacted_epa", TOL_ITERATIVE);
}

/// rdrobust options passed through RDFlex (DoubleML's `**kwargs`), RD
/// covariates and clusters, and a classifier for a binary outcome.
#[test]
fn rdflex_with_rd_options_and_binary_outcomes() {
    check_rdflex("rdflex_sharp_rd_options", TOL_ITERATIVE);
    check_rdflex("rdflex_fuzzy_covs_cluster", TOL_ITERATIVE);
    check_rdflex("rdflex_binary_outcome", TOL_ITERATIVE);
    check_rdflex("rdflex_fuzzy_binary_rho", TOL_ITERATIVE);
}

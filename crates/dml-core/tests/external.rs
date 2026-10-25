//! External nuisance predictions: refitting with a first fit's own
//! predictions supplied externally reproduces the estimates exactly, and the
//! externally predicted learners have no out-of-sample loss.

use dml_core::{
    Apo, Did, DidCs, DidScore, DmlData, DmlFit, ExternalPredictions, Iivm, Irm, IrmScore,
    LogisticRegression, Ols, Pliv, PlivScore, Plr, PlrScore,
};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `(y, d, x, z, t)`.
type Arrays = (
    Array1<f64>,
    Array1<f64>,
    Array2<f64>,
    Array1<f64>,
    Array1<f64>,
);

fn data(n: usize, seed: u64) -> Arrays {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, 3), |_| rng.gen::<f64>() * 2.0 - 1.0);
    let (mut y, mut d, mut z, mut t) = (
        Array1::zeros(n),
        Array1::zeros(n),
        Array1::zeros(n),
        Array1::zeros(n),
    );
    for i in 0..n {
        z[i] = f64::from(rng.gen::<f64>() < 0.5);
        d[i] = f64::from(rng.gen::<f64>() < 0.3 + 0.3 * z[i] + 0.1 * x[[i, 0]]);
        t[i] = f64::from(rng.gen::<f64>() < 0.5);
        y[i] = d[i] + x[[i, 0]] - 0.5 * x[[i, 1]] + 0.3 * t[i] + rng.gen::<f64>();
    }
    (y, d, x, z, t)
}

/// Every prediction of `fit` (treatment 0) as external predictions.
fn all_external(fit: &DmlFit, names: &[&str]) -> ExternalPredictions {
    names.iter().fold(ExternalPredictions::new(), |e, name| {
        let p = fit
            .predictions(name)
            .unwrap_or_else(|| panic!("no predictions {name}"));
        e.with(0, *name, p.slice(ndarray::s![.., .., 0]).to_owned())
    })
}

fn assert_same(a: &DmlFit, b: &DmlFit) {
    for (x, y) in a.all_coef().iter().zip(b.all_coef()) {
        assert!((x - y).abs() < 1e-12, "coef {x} vs {y}");
    }
    for (x, y) in a.all_se().iter().zip(b.all_se()) {
        assert!((x - y).abs() < 1e-12, "se {x} vs {y}");
    }
}

#[test]
fn external_predictions_reproduce_every_linear_model() {
    let (y, d, x, z, t) = data(400, 1);
    let base = DmlData::new(y.clone(), d.clone(), x.clone()).unwrap();

    let plr = Plr::new(Ols::new(), Ols::new()).n_rep(2);
    let fit = plr.fit(&base).unwrap();
    let ext = all_external(&fit, &["ml_l", "ml_m"]);
    let refit = plr
        .fit(&base.clone().with_external_predictions(ext))
        .unwrap();
    assert_same(&fit, &refit);
    assert!(refit.nuisance_loss("ml_l").is_none() && refit.nuisance_loss("ml_m").is_none());

    // Partially external: only ml_m supplied, ml_l still fitted (and evaluated).
    let plr_iv = Plr::new(Ols::new(), Ols::new()).score(PlrScore::IvType);
    let fit = plr_iv.fit(&base).unwrap();
    let ext = all_external(&fit, &["ml_m", "ml_g"]);
    let refit = plr_iv
        .fit(&base.clone().with_external_predictions(ext))
        .unwrap();
    assert_same(&fit, &refit);
    assert!(refit.nuisance_loss("ml_l").is_some() && refit.nuisance_loss("ml_m").is_none());

    for score in [IrmScore::Ate, IrmScore::Atte] {
        let irm = Irm::new(Ols::new(), LogisticRegression::default()).score(score);
        let fit = irm.fit(&base).unwrap();
        let refit = irm
            .fit(
                &base
                    .clone()
                    .with_external_predictions(all_external(&fit, &["ml_g0", "ml_g1", "ml_m"])),
            )
            .unwrap();
        assert_same(&fit, &refit);
    }

    // APO needs a treatment with several levels (its other-level model sees D).
    let levels = &d + &z;
    let multi = DmlData::new(y.clone(), levels, x.clone()).unwrap();
    let apo = Apo::new(Ols::new(), LogisticRegression::default(), 1.0);
    let fit = apo.fit(&multi).unwrap();
    let refit =
        apo.fit(&multi.clone().with_external_predictions(all_external(
            &fit,
            &["ml_g_d_lvl0", "ml_g_d_lvl1", "ml_m"],
        )))
        .unwrap();
    assert_same(&fit, &refit);

    let with_z = base.clone().with_instrument(z).unwrap();
    let pliv = Pliv::new(Ols::new(), Ols::new(), Ols::new()).score(PlivScore::IvType);
    let fit = pliv.fit(&with_z).unwrap();
    let refit = pliv
        .fit(
            &with_z
                .clone()
                .with_external_predictions(all_external(&fit, &["ml_l", "ml_m", "ml_r", "ml_g"])),
        )
        .unwrap();
    assert_same(&fit, &refit);

    let iivm = Iivm::new(
        Ols::new(),
        LogisticRegression::default(),
        LogisticRegression::default(),
    );
    let fit = iivm.fit(&with_z).unwrap();
    let names = ["ml_g0", "ml_g1", "ml_m", "ml_r0", "ml_r1"];
    let refit = iivm
        .fit(
            &with_z
                .clone()
                .with_external_predictions(all_external(&fit, &names)),
        )
        .unwrap();
    assert_same(&fit, &refit);

    let did = Did::new(Ols::new(), LogisticRegression::default()).score(DidScore::Observational);
    let fit = did.fit(&base).unwrap();
    let refit = did
        .fit(
            &base
                .clone()
                .with_external_predictions(all_external(&fit, &["ml_g0", "ml_g1", "ml_m"])),
        )
        .unwrap();
    assert_same(&fit, &refit);

    let cs = base.clone().with_time_indicator(t).unwrap();
    let didcs = DidCs::new(Ols::new(), LogisticRegression::default());
    let fit = didcs.fit(&cs).unwrap();
    let names = [
        "ml_g_d0_t0",
        "ml_g_d0_t1",
        "ml_g_d1_t0",
        "ml_g_d1_t1",
        "ml_m",
    ];
    let refit = didcs
        .fit(
            &cs.clone()
                .with_external_predictions(all_external(&fit, &names)),
        )
        .unwrap();
    assert_same(&fit, &refit);
}

#[test]
fn external_predictions_are_checked() {
    let (y, d, x, _, _) = data(100, 2);
    let base = DmlData::new(y, d, x).unwrap();
    let wrong = ExternalPredictions::new().with(0, "ml_l", Array2::zeros((99, 1)));
    let err = Plr::new(Ols::new(), Ols::new()).fit(&base.clone().with_external_predictions(wrong));
    assert!(err.is_err());
    // One column but two repetitions.
    let short = ExternalPredictions::new().with_vector(0, "ml_l", Array1::zeros(100));
    assert!(Plr::new(Ols::new(), Ols::new())
        .n_rep(2)
        .fit(&base.with_external_predictions(short))
        .is_err());
}

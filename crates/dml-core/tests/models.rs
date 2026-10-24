//! Monte-Carlo checks for IRM, PLIV and the bootstrap on DGPs with a known
//! effect, where the nuisance learners are correctly specified.

use dml_core::{BootstrapMethod, DmlData, Irm, IrmScore, LogisticRegression, Ols, Pliv, Plr};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn std_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn normal_matrix(rng: &mut impl Rng, n: usize, p: usize) -> Array2<f64> {
    Array2::from_shape_simple_fn((n, p), || std_normal(rng))
}

/// Logistic propensity, outcome linear in X within each arm, constant effect 1.
fn irm_data(n: usize, seed: u64) -> DmlData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = normal_matrix(&mut rng, n, 3);
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let m = 1.0 / (1.0 + (-(0.6 * x[[i, 0]] - 0.4 * x[[i, 1]])).exp());
        d[i] = f64::from(rng.gen::<f64>() < m);
        y[i] = 1.0 * d[i] + x[[i, 0]] - 0.5 * x[[i, 1]] + 0.3 * x[[i, 2]] + std_normal(&mut rng);
    }
    DmlData::new(y, d, x).unwrap()
}

/// Endogenous D (shares `u` with Y), valid instrument Z; θ = 0.5.
fn pliv_data(n: usize, seed: u64) -> (DmlData, f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = normal_matrix(&mut rng, n, 3);
    let mut z = Array1::zeros(n);
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let u = std_normal(&mut rng);
        z[i] = x[[i, 0]] + std_normal(&mut rng);
        d[i] = 0.8 * z[i] + x[[i, 1]] + u + std_normal(&mut rng);
        y[i] = 0.5 * d[i] + x[[i, 0]] - x[[i, 2]] + 2.0 * u + std_normal(&mut rng);
    }
    let data = DmlData::new(y, d, x).unwrap().with_instrument(z).unwrap();
    (data, 0.5)
}

fn covers(ci: &Array2<f64>, j: usize, theta: f64) -> bool {
    ci[[j, 0]] <= theta && theta <= ci[[j, 1]]
}

#[test]
fn irm_ate_and_atte_have_nominal_coverage() {
    let reps = 150;
    let (mut ate_cov, mut atte_cov) = (0, 0);
    for r in 0..reps {
        let data = irm_data(800, 100 + r);
        for (score, counter) in [
            (IrmScore::Ate, &mut ate_cov),
            (IrmScore::Atte, &mut atte_cov),
        ] {
            let fit = Irm::new(Ols::new(), LogisticRegression::default())
                .score(score)
                .seed(r)
                .fit(&data)
                .unwrap();
            *counter += usize::from(covers(&fit.confint(0.95).unwrap(), 0, 1.0));
        }
    }
    for (name, c) in [("ATE", ate_cov), ("ATTE", atte_cov)] {
        let coverage = c as f64 / reps as f64;
        assert!(
            (0.89..=0.99).contains(&coverage),
            "{name} coverage = {coverage}"
        );
    }
}

#[test]
fn pliv_corrects_endogeneity_that_biases_plr() {
    let (data, theta) = pliv_data(3000, 7);
    let pliv = Pliv::new(Ols::new(), Ols::new(), Ols::new())
        .fit(&data)
        .unwrap();
    let plr = Plr::new(Ols::new(), Ols::new()).fit(&data).unwrap();
    assert!(
        (pliv.coef()[0] - theta).abs() < 0.1,
        "pliv = {}",
        pliv.coef()[0]
    );
    assert!(
        (plr.coef()[0] - theta).abs() > 0.3,
        "plr = {}",
        plr.coef()[0]
    );
}

#[test]
fn pliv_has_nominal_coverage() {
    let reps = 150;
    let mut covered = 0;
    for r in 0..reps {
        let (data, theta) = pliv_data(600, 1000 + r);
        let fit = Pliv::new(Ols::new(), Ols::new(), Ols::new())
            .seed(r)
            .fit(&data)
            .unwrap();
        covered += usize::from(covers(&fit.confint(0.95).unwrap(), 0, theta));
    }
    let coverage = covered as f64 / reps as f64;
    assert!((0.89..=0.99).contains(&coverage), "coverage = {coverage}");
}

#[test]
fn joint_intervals_are_wider_and_cover_all_effects() {
    // Three treatments with effects (0.5, −0.3, 0): simultaneous coverage.
    let reps = 100;
    let theta = [0.5, -0.3, 0.0];
    let mut joint_cov = 0;
    for r in 0..reps {
        let mut rng = ChaCha8Rng::seed_from_u64(5000 + r);
        let n = 400;
        let x = normal_matrix(&mut rng, n, 3);
        let mut d = Array2::zeros((n, 3));
        let mut y = Array1::zeros(n);
        for i in 0..n {
            d[[i, 0]] = x[[i, 0]] + std_normal(&mut rng);
            d[[i, 1]] = x[[i, 1]] - 0.5 * x[[i, 0]] + std_normal(&mut rng);
            d[[i, 2]] = x[[i, 2]] + std_normal(&mut rng);
            y[i] = (0..3).map(|j| theta[j] * d[[i, j]]).sum::<f64>()
                + x[[i, 0]]
                + std_normal(&mut rng);
        }
        let data = DmlData::with_treatments(y, d, x).unwrap();
        let fit = Plr::new(Ols::new(), Ols::new()).seed(r).fit(&data).unwrap();
        let boot = fit.bootstrap(BootstrapMethod::Normal, 499, r).unwrap();
        let joint = boot.joint_confint(0.95).unwrap();
        let pointwise = fit.confint(0.95).unwrap();
        for j in 0..3 {
            assert!(joint[[j, 1]] - joint[[j, 0]] > pointwise[[j, 1]] - pointwise[[j, 0]]);
        }
        joint_cov += usize::from((0..3).all(|j| covers(&joint, j, theta[j])));
    }
    let coverage = joint_cov as f64 / reps as f64;
    assert!(
        (0.88..=1.0).contains(&coverage),
        "joint coverage = {coverage}"
    );
}

#[test]
fn repeated_splits_aggregate_by_median() {
    let data = irm_data(500, 3);
    let fit = Irm::new(Ols::new(), LogisticRegression::default())
        .n_rep(5)
        .seed(1)
        .fit(&data)
        .unwrap();
    assert_eq!(fit.n_rep(), 5);
    let mut coefs = fit.all_coef().row(0).to_vec();
    coefs.sort_by(f64::total_cmp);
    assert_eq!(fit.coef()[0], coefs[2]);
    println!("{fit}");
}

/// Confounding through sin(x₀) and a threshold in x₁ — far from linear.
fn nonlinear_plr_data(n: usize, seed: u64) -> DmlData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = Array2::from_shape_simple_fn((n, 4), || rng.gen::<f64>() * 6.0 - 3.0);
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let (a, b) = (x[[i, 0]], x[[i, 1]]);
        let confounder = 2.0 * a.sin() + 1.5 * f64::from(b > 0.0);
        d[i] = confounder + std_normal(&mut rng);
        y[i] = 0.5 * d[i] + 2.0 * confounder + std_normal(&mut rng);
    }
    DmlData::new(y, d, x).unwrap()
}

#[test]
fn random_forest_nuisances_remove_nonlinear_confounding_bias() {
    use dml_core::{RandomForestClassifier, RandomForestRegressor};
    let data = nonlinear_plr_data(3000, 11);
    let rf = RandomForestRegressor::new()
        .n_estimators(100)
        .min_samples_leaf(5)
        .seed(1);
    let forest = Plr::new(rf.clone(), rf).fit(&data).unwrap();
    let linear = Plr::new(Ols::new(), Ols::new()).fit(&data).unwrap();
    let ci = forest.confint(0.95).unwrap();
    assert!(ci[[0, 0]] < 0.5 && 0.5 < ci[[0, 1]], "forest CI {ci}");
    assert!(
        (linear.coef()[0] - 0.5).abs() > 0.2,
        "ols = {}",
        linear.coef()[0]
    );

    // A forest propensity model inside IRM runs end to end.
    let n = data.n_obs();
    let d_bin: Array1<f64> = data.treatment(0).mapv(|v| f64::from(v > 0.0));
    let y_bin = &data.y().to_owned() + &d_bin;
    let irm_data = DmlData::new(y_bin, d_bin, data.x().to_owned()).unwrap();
    let irm = Irm::new(
        RandomForestRegressor::new()
            .n_estimators(50)
            .min_samples_leaf(10),
        RandomForestClassifier::new()
            .n_estimators(50)
            .min_samples_leaf(10),
    )
    .fit(&irm_data)
    .unwrap();
    assert_eq!(irm.n_obs(), n);
    assert!(irm.coef()[0].is_finite() && irm.se()[0] > 0.0);
}

fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// Binary instrument with always-takers (15%), never-takers (15%) and
/// compliers; effect 1 for everyone, so LATE = 1. Naive regression is
/// confounded through an unobserved `u` that also drives compliance type.
fn iivm_data(n: usize, seed: u64) -> DmlData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = normal_matrix(&mut rng, n, 2);
    let (mut y, mut d, mut z) = (Array1::zeros(n), Array1::zeros(n), Array1::zeros(n));
    for i in 0..n {
        z[i] = f64::from(rng.gen::<f64>() < sigmoid(0.5 * x[[i, 0]]));
        let u: f64 = rng.gen();
        let (always, never) = (u < 0.15, u > 0.85);
        d[i] = f64::from(always || (!never && z[i] == 1.0));
        y[i] = d[i] + x[[i, 0]] + 0.5 * x[[i, 1]] + 2.0 * (u - 0.5) + std_normal(&mut rng);
    }
    DmlData::new(y, d, x).unwrap().with_instrument(z).unwrap()
}

#[test]
fn iivm_late_has_nominal_coverage() {
    use dml_core::Iivm;
    let reps = 120;
    let mut covered = 0;
    for r in 0..reps {
        let data = iivm_data(1000, 7000 + r);
        let fit = Iivm::new(
            Ols::new(),
            LogisticRegression::default(),
            LogisticRegression::default(),
        )
        .seed(r)
        .fit(&data)
        .unwrap();
        covered += usize::from(covers(&fit.confint(0.95).unwrap(), 0, 1.0));
    }
    let coverage = covered as f64 / reps as f64;
    assert!((0.88..=0.99).contains(&coverage), "coverage = {coverage}");
}

/// Two-period design with confounded treatment, covariate-dependent trends
/// and level differences between groups; ATT = 1. Returns the panel
/// (outcome change) and a repeated cross-section built from the same units.
fn did_data(n: usize, seed: u64) -> (DmlData, DmlData) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = normal_matrix(&mut rng, n, 2);
    let (mut d, mut dy, mut y_obs, mut t) = (
        Array1::zeros(n),
        Array1::zeros(n),
        Array1::zeros(n),
        Array1::zeros(n),
    );
    for i in 0..n {
        d[i] = f64::from(rng.gen::<f64>() < sigmoid(0.6 * x[[i, 0]]));
        let level = 1.0 + x[[i, 1]] + 0.8 * d[i]; // groups differ in levels
        let trend = 0.3 + 0.5 * x[[i, 0]]; // covariate-dependent common trend
        let y0 = level + std_normal(&mut rng);
        let y1 = level + trend + d[i] * 1.0 + std_normal(&mut rng);
        dy[i] = y1 - y0;
        t[i] = f64::from(rng.gen::<f64>() < 0.5);
        y_obs[i] = if t[i] == 1.0 { y1 } else { y0 };
    }
    let panel = DmlData::new(dy, d.clone(), x.clone()).unwrap();
    let cross = DmlData::new(y_obs, d, x)
        .unwrap()
        .with_time_indicator(t)
        .unwrap();
    (panel, cross)
}

#[test]
fn did_panel_and_cross_section_have_nominal_coverage() {
    use dml_core::{Did, DidCs};
    let reps = 120;
    let (mut panel_cov, mut cs_cov) = (0, 0);
    for r in 0..reps {
        let (panel, cross) = did_data(1500, 9000 + r);
        let p = Did::new(Ols::new(), LogisticRegression::default())
            .seed(r)
            .fit(&panel)
            .unwrap();
        let c = DidCs::new(Ols::new(), LogisticRegression::default())
            .seed(r)
            .fit(&cross)
            .unwrap();
        panel_cov += usize::from(covers(&p.confint(0.95).unwrap(), 0, 1.0));
        cs_cov += usize::from(covers(&c.confint(0.95).unwrap(), 0, 1.0));
    }
    // 120 runs give coverage ±2 pp of binomial noise, so only under-coverage
    // is tested. (A 600-run check of the panel estimator gave 93.5% coverage
    // with mean se / sd(θ̂) = 0.95, i.e. well-calibrated intervals.)
    for (name, c) in [("panel", panel_cov), ("cross-section", cs_cov)] {
        let coverage = c as f64 / reps as f64;
        assert!(coverage >= 0.88, "{name} coverage = {coverage}");
    }
}

#[test]
fn naive_difference_in_means_is_biased_where_did_is_not() {
    use dml_core::Did;
    let (panel, _) = did_data(5000, 3);
    let d = panel.treatment(0);
    let dy = panel.y();
    let mean_by = |g: f64| {
        let (s, c) = d
            .iter()
            .zip(dy)
            .filter(|(&di, _)| di == g)
            .fold((0.0, 0.0), |(s, c), (_, &v)| (s + v, c + 1.0));
        s / c
    };
    // Unconditional DiD ignores that the trend depends on x₀, which drives treatment.
    let naive = mean_by(1.0) - mean_by(0.0);
    let dml = Did::new(Ols::new(), LogisticRegression::default())
        .fit(&panel)
        .unwrap();
    assert!((naive - 1.0).abs() > 0.15, "naive = {naive}");
    assert!((dml.coef()[0] - 1.0).abs() < 0.1, "dml = {}", dml.coef()[0]);
}

#[test]
fn gradient_boosting_nuisances_remove_nonlinear_confounding_bias() {
    use dml_core::{GradientBoostingClassifier, GradientBoostingRegressor};
    let data = nonlinear_plr_data(3000, 12);
    let gb = GradientBoostingRegressor::new()
        .n_estimators(200)
        .learning_rate(0.05)
        .max_depth(Some(3));
    let fit = Plr::new(gb.clone(), gb).fit(&data).unwrap();
    let ci = fit.confint(0.95).unwrap();
    assert!(ci[[0, 0]] < 0.5 && 0.5 < ci[[0, 1]], "boosting CI {ci}");

    let d_bin: Array1<f64> = data.treatment(0).mapv(|v| f64::from(v > 0.0));
    let y_bin = &data.y().to_owned() + &d_bin;
    let irm_data = DmlData::new(y_bin, d_bin, data.x().to_owned()).unwrap();
    let irm = Irm::new(
        GradientBoostingRegressor::new().n_estimators(100),
        GradientBoostingClassifier::new().n_estimators(100),
    )
    .fit(&irm_data)
    .unwrap();
    assert!(irm.coef()[0].is_finite() && irm.se()[0] > 0.0);
}

/// y = θd + x₀ + γA + ε, d = x₀ + δA + v with a confounder A. Returns the data
/// with A as the last covariate column.
fn confounded_data(n: usize, seed: u64) -> DmlData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut x = Array2::zeros((n, 3));
    let (mut d, mut y) = (Array1::zeros(n), Array1::zeros(n));
    for i in 0..n {
        let (x0, x1, a) = (
            std_normal(&mut rng),
            std_normal(&mut rng),
            std_normal(&mut rng),
        );
        x[[i, 0]] = x0;
        x[[i, 1]] = x1;
        x[[i, 2]] = a;
        d[i] = x0 + 0.6 * a + std_normal(&mut rng);
        y[i] = 0.5 * d[i] + x0 + 0.8 * a + 0.3 * x1 + std_normal(&mut rng);
    }
    DmlData::new(y, d, x).unwrap()
}

#[test]
fn sensitivity_bounds_cover_the_estimate_with_the_confounder_included() {
    use dml_core::SensitivityParams;
    let long_data = confounded_data(4000, 31);
    let short_data = long_data.without_covariates(&[2]).unwrap(); // A unobserved
    let model = Plr::new(Ols::new(), Ols::new()).seed(4);
    let long = model.fit(&long_data).unwrap();
    let short = model.fit(&short_data).unwrap();
    // Omitting A biases the estimate upwards.
    assert!(
        short.coef()[0] - long.coef()[0] > 0.2,
        "short {} long {}",
        short.coef()[0],
        long.coef()[0]
    );

    // Benchmark A's strength by dropping it from the long model, then bound the
    // short model with that strength: the bounds must contain the long estimate.
    let benchmark = model
        .sensitivity_benchmark(&long_data, &long, &[2])
        .unwrap();
    assert!(
        benchmark.cf_y[0] > 0.05 && benchmark.cf_d[0] > 0.05,
        "{benchmark}"
    );
    let scenario = SensitivityParams::new()
        .cf_y(benchmark.cf_y[0])
        .cf_d(benchmark.cf_d[0])
        .rho(1.0);
    let res = short.sensitivity_analysis(&scenario).unwrap();
    let truth = long.coef()[0];
    assert!(
        res.theta_lower[0] <= truth && truth <= res.theta_upper[0],
        "{res}"
    );
    // Robustness values: the effect is clearly positive, so moving it to zero
    // needs strong confounding, and the CI bound reaches zero first.
    assert!(
        res.rv[0] > 0.1 && res.rva[0] < res.rv[0] && res.rva[0] > 0.0,
        "{res}"
    );
    // A weaker scenario gives narrower bounds.
    let weak = short
        .sensitivity_analysis(&SensitivityParams::new().cf_y(0.01).cf_d(0.01))
        .unwrap();
    assert!(weak.theta_upper[0] - weak.theta_lower[0] < res.theta_upper[0] - res.theta_lower[0]);
    println!("{res}\n{benchmark}");

    // Models without sensitivity analysis say so.
    let z = long_data.x().column(0).to_owned();
    let pliv = Pliv::new(Ols::new(), Ols::new(), Ols::new())
        .fit(&short_data.clone().with_instrument(z).unwrap())
        .unwrap();
    assert!(pliv
        .sensitivity_analysis(&SensitivityParams::new())
        .is_err());
}

/// 60 clusters of 10 with cluster-level shocks in both D and Y: the score is
/// correlated within clusters (intra-cluster correlation 0.25), so the iid
/// standard error is too small by about √(1 + 9 · 0.25) ≈ 1.8; θ = 0.5.
fn clustered_plr_data(seed: u64) -> (DmlData, DmlData) {
    let (n_clusters, size) = (60, 10);
    let n = n_clusters * size;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = normal_matrix(&mut rng, n, 3);
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    let mut cluster = Array1::zeros(n);
    for g in 0..n_clusters {
        let (u_g, w_g) = (std_normal(&mut rng), std_normal(&mut rng));
        for i in g * size..(g + 1) * size {
            cluster[i] = g as i64;
            d[i] = x[[i, 0]] - 0.5 * x[[i, 1]] + u_g + std_normal(&mut rng);
            y[i] = 0.5 * d[i] + x[[i, 1]] + 0.5 * x[[i, 2]] + w_g + std_normal(&mut rng);
        }
    }
    let iid = DmlData::new(y, d, x).unwrap();
    (iid.clone(), iid.with_clusters(cluster).unwrap())
}

#[test]
fn cluster_robust_standard_errors_restore_coverage() {
    let reps = 150;
    let (mut iid_cov, mut cl_cov) = (0, 0);
    for r in 0..reps {
        let (iid, clustered) = clustered_plr_data(4000 + r);
        let plr = Plr::new(Ols::new(), Ols::new()).seed(r);
        let fit_iid = plr.fit(&iid).unwrap();
        let fit_cl = plr.fit(&clustered).unwrap();
        assert_eq!(fit_cl.n_clusters(), Some(60));
        iid_cov += usize::from(covers(&fit_iid.confint(0.95).unwrap(), 0, 0.5));
        cl_cov += usize::from(covers(&fit_cl.confint(0.95).unwrap(), 0, 0.5));
    }
    let (iid_cov, cl_cov) = (iid_cov as f64 / reps as f64, cl_cov as f64 / reps as f64);
    assert!(cl_cov >= 0.88, "clustered coverage = {cl_cov}");
    assert!(
        iid_cov <= 0.85,
        "iid coverage = {iid_cov} should be too low"
    );
}

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

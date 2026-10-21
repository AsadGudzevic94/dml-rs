//! PLR on simulated data with a known effect.

use dml_core::{DmlData, Ols, Plr};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const THETA: f64 = 0.5;

fn std_normal(rng: &mut impl Rng) -> f64 {
    // Box–Muller transform
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Nonlinear confounding through x1²: it drives both D and Y.
/// Returns data whose covariates are the basis [x1, x2, x3, x1², x2², x3²],
/// so OLS nuisances are correctly specified.
fn simulate(n: usize, seed: u64) -> (DmlData, f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut basis = Array2::<f64>::zeros((n, 6));
    let mut d = Array1::<f64>::zeros(n);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let x: [f64; 3] = [
            std_normal(&mut rng),
            std_normal(&mut rng),
            std_normal(&mut rng),
        ];
        let m = 0.8 * x[0] * x[0] + 0.5 * x[1];
        let g = 1.0 + 1.5 * x[0] * x[0] - 0.7 * x[1] + 0.4 * x[2] * x[2];
        d[i] = m + std_normal(&mut rng);
        y[i] = THETA * d[i] + g + std_normal(&mut rng);
        for (j, &xj) in x.iter().enumerate() {
            basis[[i, j]] = xj;
            basis[[i, 3 + j]] = xj * xj;
        }
    }
    // Naive slope of y on d (ignores confounding) for comparison.
    let (dm, ym) = (d.mean().unwrap(), y.mean().unwrap());
    let naive = (&d - dm).dot(&(&y - ym)) / (&d - dm).dot(&(&d - dm));
    (DmlData::new(y, d, basis).unwrap(), naive)
}

#[test]
fn recovers_theta_where_naive_regression_is_biased() {
    let (data, naive) = simulate(2000, 1);
    let fit = Plr::new(Ols::new(), Ols::new())
        .seed(42)
        .fit(&data)
        .unwrap();
    assert!(
        (fit.coef()[0] - THETA).abs() < 0.1,
        "coef = {}",
        fit.coef()[0]
    );
    assert!((naive - THETA).abs() > 0.3, "naive = {naive}");
    let ci = fit.confint(0.95).unwrap();
    assert!(ci[[0, 0]] < THETA && THETA < ci[[0, 1]]);
    println!("{fit}");
}

#[test]
fn ci_coverage_is_close_to_nominal() {
    let reps = 300;
    let mut covered = 0;
    for r in 0..reps {
        let (data, _) = simulate(500, 1000 + r);
        let fit = Plr::new(Ols::new(), Ols::new()).seed(r).fit(&data).unwrap();
        let ci = fit.confint(0.95).unwrap();
        covered += usize::from(ci[[0, 0]] <= THETA && THETA <= ci[[0, 1]]);
    }
    let coverage = covered as f64 / reps as f64;
    // Monte-Carlo s.e. of the coverage is about 1.3% at 300 reps.
    assert!((0.91..=0.99).contains(&coverage), "coverage = {coverage}");
}

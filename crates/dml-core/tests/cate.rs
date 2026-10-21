//! CATE meta-learners on a DGP with a known heterogeneous effect
//! τ(x) = 1 + x₁, logistic propensity and outcomes linear in X per arm.

use dml_core::{
    DmlData, DrLearner, LogisticRegression, Ols, RLearner, Regressor, Result, SLearner, TLearner,
    XLearner,
};
use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn std_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>();
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn tau(x: ArrayView1<'_, f64>) -> f64 {
    1.0 + x[0]
}

fn simulate(n: usize, seed: u64) -> DmlData {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = Array2::from_shape_simple_fn((n, 3), || std_normal(&mut rng));
    let mut d = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let row = x.row(i);
        let e = 1.0 / (1.0 + (-(0.5 * row[0] - 0.3 * row[1])).exp());
        d[i] = f64::from(rng.gen::<f64>() < e);
        y[i] = row[1] + 0.5 * row[2] + d[i] * tau(row) + std_normal(&mut rng);
    }
    DmlData::new(y, d, x).unwrap()
}

fn test_grid() -> Array2<f64> {
    Array2::from_shape_fn(
        (41, 3),
        |(i, j)| if j == 0 { -2.0 + 0.1 * i as f64 } else { 0.0 },
    )
}

fn rmse(effect: &Array1<f64>, grid: &Array2<f64>) -> f64 {
    let err: f64 = grid
        .rows()
        .into_iter()
        .zip(effect)
        .map(|(r, &e)| (e - tau(r)).powi(2))
        .sum();
    (err / grid.nrows() as f64).sqrt()
}

/// OLS on `[x, d, d·x]` — lets an S-learner represent effect heterogeneity.
/// Assumes the treatment is the last column, as the S-learner arranges it.
#[derive(Clone, Default)]
struct InteractedOls(Ols);

fn interact(xd: ArrayView2<'_, f64>) -> Array2<f64> {
    let p = xd.ncols() - 1;
    let d = xd.column(p).to_owned().insert_axis(Axis(1));
    let x = xd.slice(ndarray::s![.., ..p]);
    concatenate![Axis(1), xd, &x * &d]
}

impl Regressor for InteractedOls {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.0.fit(interact(x).view(), y)
    }
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.0.predict(interact(x).view())
    }
    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self::default())
    }
}

#[test]
fn all_meta_learners_recover_the_effect_function() {
    let data = simulate(4000, 1);
    let grid = test_grid();
    let fits = [
        SLearner::new(InteractedOls::default()).fit(&data).unwrap(),
        TLearner::new(Ols::new()).fit(&data).unwrap(),
        XLearner::new(Ols::new(), Ols::new(), LogisticRegression::default())
            .fit(&data)
            .unwrap(),
        RLearner::new(Ols::new(), LogisticRegression::default(), Ols::new())
            .seed(3)
            .fit(&data)
            .unwrap(),
        DrLearner::new(Ols::new(), LogisticRegression::default(), Ols::new())
            .seed(3)
            .fit(&data)
            .unwrap(),
    ];
    for fit in &fits {
        let effect = fit.effect(grid.view()).unwrap();
        let err = rmse(&effect, &grid);
        assert!(err < 0.15, "{}: rmse {err}", fit.method());
        let ate = fit.average_effect(data.x()).unwrap();
        assert!((ate - 1.0).abs() < 0.15, "{}: ate {ate}", fit.method());
    }
}

#[test]
fn plain_s_learner_with_linear_model_is_constant() {
    // Without interactions the S-learner cannot express heterogeneity.
    let data = simulate(2000, 2);
    let fit = SLearner::new(Ols::new()).fit(&data).unwrap();
    let effect = fit.effect(test_grid().view()).unwrap();
    let spread = effect.iter().fold(f64::MIN, |m, &v| m.max(v))
        - effect.iter().fold(f64::MAX, |m, &v| m.min(v));
    assert!(spread < 1e-9);
}

#[test]
fn dr_blp_inference_covers_true_coefficients() {
    // Basis [1, x₁]: the BLP of τ is exactly (1, 1).
    let reps = 100;
    let mut covered = [0usize; 2];
    for r in 0..reps {
        let data = simulate(1500, 100 + r);
        let fit = DrLearner::new(Ols::new(), LogisticRegression::default(), Ols::new())
            .seed(r)
            .fit(&data)
            .unwrap();
        let basis = concatenate![
            Axis(1),
            Array2::ones((data.n_obs(), 1)),
            data.x().slice(ndarray::s![.., ..1])
        ];
        let ci = fit.blp(basis.view()).unwrap().confint(0.95).unwrap();
        for j in 0..2 {
            covered[j] += usize::from(ci[[j, 0]] <= 1.0 && 1.0 <= ci[[j, 1]]);
        }
    }
    for (j, &c) in covered.iter().enumerate() {
        let coverage = c as f64 / reps as f64;
        assert!(
            (0.88..=1.0).contains(&coverage),
            "coef {j}: coverage {coverage}"
        );
    }
}

#[test]
fn gate_and_intercept_blp_agree_with_ate() {
    let data = simulate(3000, 9);
    let fit = DrLearner::new(Ols::new(), LogisticRegression::default(), Ols::new())
        .fit(&data)
        .unwrap();
    let n = data.n_obs();
    let ate = fit.blp(Array2::ones((n, 1)).view()).unwrap();
    let phi = fit.pseudo_outcomes().unwrap();
    assert!((ate.coef()[0] - phi.mean().unwrap()).abs() < 1e-12);
    let groups = Array2::from_shape_fn((n, 2), |(i, g)| {
        let high = data.x()[[i, 0]] > 0.0;
        f64::from(if g == 0 { !high } else { high })
    });
    let gate = fit.gate(groups.view()).unwrap();
    // E[1 + x₁ | x₁ ≤ 0] ≈ 1 − 0.798, E[1 + x₁ | x₁ > 0] ≈ 1 + 0.798
    assert!((gate.coef()[0] - 0.202).abs() < 0.2, "{}", gate.coef()[0]);
    assert!((gate.coef()[1] - 1.798).abs() < 0.2, "{}", gate.coef()[1]);
}

#[test]
fn r_learner_rejects_effect_learner_without_weights() {
    struct NoWeights;
    impl Regressor for NoWeights {
        fn fit(&mut self, _: ArrayView2<'_, f64>, _: ArrayView1<'_, f64>) -> Result<()> {
            Ok(())
        }
        fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
            Ok(Array1::zeros(x.nrows()))
        }
        fn fresh(&self) -> Box<dyn Regressor> {
            Box::new(NoWeights)
        }
    }
    let data = simulate(300, 4);
    let err = RLearner::new(Ols::new(), LogisticRegression::default(), NoWeights)
        .fit(&data)
        .unwrap_err();
    assert!(err.to_string().contains("sample weights"));
    assert!(TLearner::new(Ols::new())
        .fit(
            &DmlData::new(
                data.y().to_owned(),
                data.x().column(0).to_owned(),
                data.x().to_owned()
            )
            .unwrap()
        )
        .is_err());
}

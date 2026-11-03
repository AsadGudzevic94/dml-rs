//! `CausalSurvival` against R (`parity/generate_survival_fixtures.R`) and on simulated data with a
//! known truth.

use std::collections::HashMap;

use dml_core::learner::Classifier;
use dml_core::{
    CausalSurvival, DmlData, Fold, LogisticRegression, Result, SurvivalEstimand, SurvivalGrid,
    TimeEncoding,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::Value;

/// The empirical mean of the target for each distinct row: a saturated model.
#[derive(Default)]
struct Saturated {
    means: HashMap<Vec<u64>, f64>,
}

fn key(row: ArrayView1<'_, f64>) -> Vec<u64> {
    row.iter().map(|v| v.to_bits()).collect()
}

impl Classifier for Saturated {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        let mut totals: HashMap<Vec<u64>, (f64, f64)> = HashMap::new();
        for (row, &target) in x.rows().into_iter().zip(y) {
            let entry = totals.entry(key(row)).or_default();
            entry.0 += target;
            entry.1 += 1.0;
        }
        self.means = totals.into_iter().map(|(k, (s, n))| (k, s / n)).collect();
        Ok(())
    }

    fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        Ok(x.rows()
            .into_iter()
            .map(|row| self.means.get(&key(row)).copied().unwrap_or(0.0))
            .collect())
    }

    fn fresh(&self) -> Box<dyn Classifier> {
        Box::<Saturated>::default()
    }
}

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/survival_kaplan_meier.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn numbers(fx: &Value, key: &str) -> Vec<f64> {
    fx[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

/// One covariate that is always zero, and training on the test rows: the hazards are then the
/// empirical hazards of each arm.
fn lung(fx: &Value) -> (DmlData, Array1<f64>, Vec<Vec<Fold>>) {
    let (time, event, d) = (numbers(fx, "time"), numbers(fx, "event"), numbers(fx, "d"));
    let n = time.len();
    let data = DmlData::new(Array1::from(time), Array1::from(d), Array2::zeros((n, 1))).unwrap();
    let all: Vec<usize> = (0..n).collect();
    let fold = Fold {
        train: all.clone(),
        test: all,
    };
    (data, Array1::from(event), vec![vec![fold]])
}

fn saturated() -> CausalSurvival {
    CausalSurvival::new(
        Saturated::default(),
        Saturated::default(),
        Saturated::default(),
    )
    .grid(SurvivalGrid::EventTimes)
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol * e.abs().max(1.0), "{a} vs {e}");
    }
}

#[test]
fn reduces_to_kaplan_meier() {
    let fx = fixture();
    let (data, event, splits) = lung(&fx);
    let fit = saturated()
        .horizons(numbers(&fx, "horizons"))
        .fit_with_folds(&data, event.view(), &splits)
        .unwrap();
    assert_close(
        fit.potential(true).coef().as_slice().unwrap(),
        &numbers(&fx, "km_treated"),
        1e-12,
    );
    assert_close(
        fit.potential(false).coef().as_slice().unwrap(),
        &numbers(&fx, "km_control"),
        1e-12,
    );
    // The influence function gives Greenwood's variance up to the censoring estimate's noise.
    assert_close(
        fit.potential(true).se().as_slice().unwrap(),
        &numbers(&fx, "km_treated_se"),
        0.02,
    );
    assert_close(
        fit.potential(false).se().as_slice().unwrap(),
        &numbers(&fx, "km_control_se"),
        0.02,
    );
}

#[test]
fn reduces_to_the_restricted_mean_of_survrm2() {
    let fx = fixture();
    let (data, event, splits) = lung(&fx);
    let fit = saturated()
        .estimand(SurvivalEstimand::Rmst)
        .horizons(numbers(&fx, "rmst_tau"))
        .fit_with_folds(&data, event.view(), &splits)
        .unwrap();
    let (treated, control) = (fit.potential(true), fit.potential(false));
    assert_close(
        treated.coef().as_slice().unwrap(),
        &numbers(&fx, "rmst_treated"),
        1e-10,
    );
    assert_close(
        control.coef().as_slice().unwrap(),
        &numbers(&fx, "rmst_control"),
        1e-10,
    );
    assert_close(
        treated.se().as_slice().unwrap(),
        &numbers(&fx, "rmst_treated_se"),
        0.02,
    );
    assert_close(
        control.se().as_slice().unwrap(),
        &numbers(&fx, "rmst_control_se"),
        0.02,
    );
}

/// Exponential event and censoring times that depend on `x` and the treatment, with treatment
/// assigned on `x`: Kaplan-Meier by arm is confounded. Treatment multiplies the event rate by 0.5.
fn simulate(n: usize, seed: u64) -> (DmlData, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut uniform = || rng.gen::<f64>();
    let x: Vec<f64> = (0..n).map(|_| uniform() * 2.0 - 1.0).collect();
    let mut d = Vec::with_capacity(n);
    let (mut time, mut event) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for &xi in &x {
        let treated = uniform() < 1.0 / (1.0 + (-1.5 * xi).exp());
        let rate = (0.8 * xi).exp() * if treated { 0.5 } else { 1.0 };
        let t = -(1.0_f64 - uniform()).ln() / rate;
        let c = -(1.0_f64 - uniform()).ln() / (0.3 * (-0.8 * xi).exp());
        d.push(f64::from(treated));
        time.push(t.min(c));
        event.push(f64::from(t <= c));
    }
    let x = Array2::from_shape_fn((n, 1), |(i, _)| x[i]);
    (
        DmlData::new(Array1::from(time), Array1::from(d), x).unwrap(),
        Array1::from(event),
    )
}

/// `E[exp(−rate(x)·t)]` for `x` uniform on (−1, 1), by the midpoint rule.
fn true_survival(hazard_ratio: f64, t: f64) -> f64 {
    let m = 100_000;
    (0..m)
        .map(|j| {
            let x = (j as f64 + 0.5) / m as f64 * 2.0 - 1.0;
            (-(0.8 * x).exp() * hazard_ratio * t).exp()
        })
        .sum::<f64>()
        / m as f64
}

fn logistic_model() -> CausalSurvival {
    let logit = || LogisticRegression::new(1e4);
    CausalSurvival::new(logit(), logit(), logit())
        .time_encoding(TimeEncoding::Dummies)
        .grid(SurvivalGrid::Quantiles(12))
}

#[test]
fn removes_confounding_and_informative_censoring() {
    let horizon = 1.0;
    let truth = true_survival(0.5, horizon) - true_survival(1.0, horizon);
    let reps = 60;
    let mut estimates = Vec::new();
    let mut naive = Vec::new();
    let mut covered = 0;
    for seed in 0..reps {
        let (data, event) = simulate(1500, seed);
        let fit = logistic_model()
            .horizons(vec![horizon])
            .fit(&data, event.view())
            .unwrap();
        let (estimate, se) = (fit.effect().coef()[0], fit.effect().se()[0]);
        covered += usize::from((estimate - truth).abs() <= 1.96 * se);
        estimates.push(estimate);
        let flat = Array2::zeros((data.n_obs(), 1));
        let plain = DmlData::new(data.y().to_owned(), data.treatment(0).to_owned(), flat).unwrap();
        naive.push(
            logistic_model()
                .horizons(vec![horizon])
                .fit(&plain, event.view())
                .unwrap()
                .effect()
                .coef()[0],
        );
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let spread = {
        let m = mean(&estimates);
        (estimates.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (reps - 1) as f64).sqrt()
    };
    let mc_se = spread / (reps as f64).sqrt();
    assert!(
        (mean(&estimates) - truth).abs() < 0.015,
        "{} vs {truth}",
        mean(&estimates)
    );
    assert!((mean(&estimates) - truth).abs() < 4.0 * mc_se);
    assert!(covered >= 50, "covered {covered} of {reps}");
    // Without the covariate the estimate is clearly off.
    assert!(
        (mean(&naive) - truth).abs() > 0.03,
        "{} vs {truth}",
        mean(&naive)
    );
}

#[test]
fn repetitions_and_the_curves_are_consistent() {
    let (data, event) = simulate(600, 3);
    let fit = logistic_model()
        .horizons(vec![0.5, 1.0])
        .estimand(SurvivalEstimand::Rmst)
        .n_rep(3)
        .fit(&data, event.view())
        .unwrap();
    assert_eq!(fit.effect().coef().len(), 2);
    assert_eq!(fit.effect().n_rep(), 3);
    assert_eq!(fit.propensity().dim(), (600, 3));
    // RMST is at most the horizon, and a survival curve starts high and falls.
    assert!(fit.potential(true).coef()[0] <= 0.5);
    let curve = fit.curve(false).coef();
    assert!(curve[0] > curve[curve.len() - 1]);
    assert_eq!(curve.len(), fit.grid().len());
    assert_eq!(fit.horizons(), &[0.5, 1.0]);
}

#[test]
fn works_without_censoring() {
    let (data, _) = simulate(400, 5);
    let event = Array1::from_elem(400, 1.0);
    let fit = logistic_model()
        .horizons(vec![0.5])
        .fit(&data, event.view())
        .unwrap();
    assert!(fit.effect().se()[0] > 0.0);
}

fn aalen_johansen_fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/survival_aalen_johansen.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn reduces_to_aalen_johansen_with_competing_events() {
    let fx = aalen_johansen_fixture();
    let (time, event, d) = (
        numbers(&fx, "time"),
        numbers(&fx, "event"),
        numbers(&fx, "d"),
    );
    let n = time.len();
    let data = DmlData::new(Array1::from(time), Array1::from(d), Array2::zeros((n, 1))).unwrap();
    let event = Array1::from(event);
    let all: Vec<usize> = (0..n).collect();
    let splits = vec![vec![Fold {
        train: all.clone(),
        test: all,
    }]];
    let fit = |estimand| {
        saturated()
            .estimand(estimand)
            .horizons(numbers(&fx, "horizons"))
            .fit_with_folds(&data, event.view(), &splits)
            .unwrap()
    };
    let survival = fit(SurvivalEstimand::Survival);
    let first = fit(SurvivalEstimand::CumulativeIncidence(1));
    let second = fit(SurvivalEstimand::CumulativeIncidence(2));
    for (treated, arm) in [(true, "treated"), (false, "control")] {
        // Freedom from either cause, and each cause's cumulative incidence.
        assert_close(
            survival.potential(treated).coef().as_slice().unwrap(),
            &numbers(&fx, &format!("survival_{arm}")),
            1e-12,
        );
        assert_close(
            first.potential(treated).coef().as_slice().unwrap(),
            &numbers(&fx, &format!("cif1_{arm}")),
            1e-12,
        );
        assert_close(
            second.potential(treated).coef().as_slice().unwrap(),
            &numbers(&fx, &format!("cif2_{arm}")),
            1e-12,
        );
        // The scores of survival and of all causes sum to one, unit by unit.
        let fits = [
            survival.curve(treated),
            first.incidence_curve(treated).unwrap(),
            second.incidence_curve(treated).unwrap(),
        ];
        let mut coef = Array1::<f64>::zeros(fits[0].coef().len());
        let mut total = Array2::<f64>::zeros(fits[0].psi().slice(ndarray::s![.., .., 0]).dim());
        for fit in fits {
            coef += &fit.coef();
            total += &fit.psi().slice(ndarray::s![.., .., 0]);
        }
        assert!(coef.iter().all(|v| (v - 1.0).abs() < 1e-12));
        assert!(total.iter().all(|v| v.abs() < 1e-12));
    }
    assert!(survival.incidence_curve(true).is_none());
}

/// Two causes with exponential hazards `λ₁ = 0.6 e^{0.8x}·hr` and `λ₂ = 0.4 e^{−0.5x}`, confounded
/// treatment, censoring depending on `x`. Treatment halves the hazard of cause 1 only.
fn simulate_competing(n: usize, seed: u64) -> (DmlData, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut uniform = || rng.gen::<f64>();
    let (mut xs, mut d, mut time, mut event) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let x = uniform() * 2.0 - 1.0;
        let treated = uniform() < 1.0 / (1.0 + (-1.5 * x).exp());
        let (l1, l2) = (
            0.6 * (0.8 * x).exp() * if treated { 0.5 } else { 1.0 },
            0.4 * (-0.5 * x).exp(),
        );
        let t = -(1.0_f64 - uniform()).ln() / (l1 + l2);
        let cause = if uniform() < l1 / (l1 + l2) { 1.0 } else { 2.0 };
        let c = -(1.0_f64 - uniform()).ln() / (0.3 * (-0.8 * x).exp());
        xs.push(x);
        d.push(f64::from(treated));
        time.push(t.min(c));
        event.push(if t <= c { cause } else { 0.0 });
    }
    let x = Array2::from_shape_fn((n, 1), |(i, _)| xs[i]);
    (
        DmlData::new(Array1::from(time), Array1::from(d), x).unwrap(),
        Array1::from(event),
    )
}

/// `E[λ₁/(λ₁+λ₂)·(1 − e^{−(λ₁+λ₂)t})]` for `x` uniform on (−1, 1).
fn true_incidence(hazard_ratio: f64, t: f64) -> f64 {
    let m = 100_000;
    (0..m)
        .map(|j| {
            let x = (j as f64 + 0.5) / m as f64 * 2.0 - 1.0;
            let (l1, l2) = (0.6 * (0.8 * x).exp() * hazard_ratio, 0.4 * (-0.5 * x).exp());
            l1 / (l1 + l2) * (1.0 - (-(l1 + l2) * t).exp())
        })
        .sum::<f64>()
        / m as f64
}

#[test]
fn estimates_the_cumulative_incidence_of_a_competing_cause() {
    let horizon = 1.0;
    let truth = true_incidence(0.5, horizon) - true_incidence(1.0, horizon);
    let reps = 40;
    let (mut estimates, mut covered) = (Vec::new(), 0);
    for seed in 0..reps {
        let (data, event) = simulate_competing(1500, 100 + seed);
        let fit = logistic_model()
            .estimand(SurvivalEstimand::CumulativeIncidence(1))
            .horizons(vec![horizon])
            .fit(&data, event.view())
            .unwrap();
        let (estimate, se) = (fit.effect().coef()[0], fit.effect().se()[0]);
        covered += usize::from((estimate - truth).abs() <= 1.96 * se);
        estimates.push(estimate);
    }
    let mean = estimates.iter().sum::<f64>() / reps as f64;
    let spread =
        (estimates.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (reps - 1) as f64).sqrt();
    assert!(truth < -0.05, "{truth}");
    assert!(
        (mean - truth).abs() < 4.0 * spread / (reps as f64).sqrt(),
        "{mean} vs {truth}"
    );
    assert!(covered >= 34, "covered {covered} of {reps}");
}

#[test]
fn singleton_clusters_reproduce_the_unclustered_standard_errors() {
    let (data, event) = simulate(500, 7);
    let model = logistic_model().horizons(vec![0.5, 1.0]);
    let plain = model.fit(&data, event.view()).unwrap();
    let n = data.n_obs();
    let clustered_data = data
        .clone()
        .with_clusters(Array1::from_iter(0..n as i64))
        .unwrap();
    // The same folds: the clustered standard errors are then those without clusters.
    let folds = {
        let strata: Vec<usize> = data
            .treatment(0)
            .iter()
            .zip(&event)
            .map(|(&d, &e)| 2 * d as usize + e as usize)
            .collect();
        dml_core::SampleSplitting::new(5, 1, 0)
            .draw_stratified(&strata)
            .unwrap()
    };
    let a = model.fit_with_folds(&data, event.view(), &folds).unwrap();
    let b = model
        .fit_with_folds(&clustered_data, event.view(), &folds)
        .unwrap();
    assert_close(
        b.effect().se().as_slice().unwrap(),
        a.effect().se().as_slice().unwrap(),
        1e-12,
    );
    assert_eq!(a.effect().coef(), b.effect().coef());
    assert_eq!(plain.effect().coef(), a.effect().coef());
    // Real clusters: whole clusters go to folds and the fit runs.
    let grouped = data
        .with_clusters(Array1::from_iter((0..n as i64).map(|i| i / 5)))
        .unwrap();
    let fit = model.fit(&grouped, event.view()).unwrap();
    assert!(fit.effect().se().iter().all(|s| s.is_finite() && *s > 0.0));
}

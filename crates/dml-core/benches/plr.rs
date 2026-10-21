//! `cargo bench -p dml-core` — PLR fit time for a few problem sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dml_core::{DmlData, Ols, Plr};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn simulate(n: usize, p: usize) -> DmlData {
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let x = Array2::from_shape_fn((n, p), |_| rng.gen::<f64>() - 0.5);
    let d: Array1<f64> = x
        .rows()
        .into_iter()
        .map(|r| r[0] + 0.5 * r[1] + rng.gen::<f64>())
        .collect();
    let y: Array1<f64> = x
        .rows()
        .into_iter()
        .zip(d.iter())
        .map(|(r, &di)| 0.5 * di + r[0] - r[2] + rng.gen::<f64>())
        .collect();
    DmlData::new(y, d, x).unwrap()
}

fn bench_plr(c: &mut Criterion) {
    let mut group = c.benchmark_group("plr_fit_ols_5fold");
    group.sample_size(20);
    for &(n, p) in &[(1_000, 10), (10_000, 20), (100_000, 50)] {
        let data = simulate(n, p);
        let model = Plr::new(Ols::new(), Ols::new()).seed(1);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n{n}_p{p}")),
            &data,
            |b, data| b.iter(|| model.fit(data).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_plr);
criterion_main!(benches);

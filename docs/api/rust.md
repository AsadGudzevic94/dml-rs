# Rust API

The Rust crate [`dml-core`](https://crates.io/crates/dml-core) is documented on
[docs.rs/dml-core](https://docs.rs/dml-core).

Main entry points:

| Item | Purpose |
|---|---|
| [`DmlData`](https://docs.rs/dml-core/latest/dml_core/struct.DmlData.html) | Outcome, treatment(s), covariates, optional instrument, time indicator and clusters |
| [`Plr`](https://docs.rs/dml-core/latest/dml_core/struct.Plr.html), [`Irm`](https://docs.rs/dml-core/latest/dml_core/struct.Irm.html), [`Pliv`](https://docs.rs/dml-core/latest/dml_core/struct.Pliv.html), [`Iivm`](https://docs.rs/dml-core/latest/dml_core/struct.Iivm.html), [`Did`](https://docs.rs/dml-core/latest/dml_core/struct.Did.html), [`DidCs`](https://docs.rs/dml-core/latest/dml_core/struct.DidCs.html) | Models (builders with `n_folds`, `n_rep`, `seed`, `fit`, `fit_with_folds`) |
| [`StaggeredDid`](https://docs.rs/dml-core/latest/dml_core/struct.StaggeredDid.html) | Staggered difference-in-differences |
| [`DmlFit`](https://docs.rs/dml-core/latest/dml_core/struct.DmlFit.html) | Results: estimates, inference, bootstrap, sensitivity analysis, learner losses |
| [`Regressor`](https://docs.rs/dml-core/latest/dml_core/trait.Regressor.html), [`Classifier`](https://docs.rs/dml-core/latest/dml_core/trait.Classifier.html) | Learner traits, implemented by the built-in learners |
| [`LinearScoreModel`](https://docs.rs/dml-core/latest/dml_core/trait.LinearScoreModel.html) | Trait for adding a new orthogonal score |

The crate builds with Rust 1.80 or newer. Parallel cross-fitting is enabled by the default
`parallel` feature (rayon).

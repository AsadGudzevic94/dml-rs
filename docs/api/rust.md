# Rust API

The Rust crate [`dml-core`](https://crates.io/crates/dml-core) is documented on
[docs.rs/dml-core](https://docs.rs/dml-core).

Main entry points:

| Item | Purpose |
|---|---|
| [`DmlData`](https://docs.rs/dml-core/latest/dml_core/struct.DmlData.html) | Outcome, treatment(s), covariates, optional instrument, time indicator and clusters |
| [`Plr`](https://docs.rs/dml-core/latest/dml_core/struct.Plr.html), [`Lplr`](https://docs.rs/dml-core/latest/dml_core/struct.Lplr.html), [`Plpr`](https://docs.rs/dml-core/latest/dml_core/struct.Plpr.html), [`Irm`](https://docs.rs/dml-core/latest/dml_core/struct.Irm.html), [`Pliv`](https://docs.rs/dml-core/latest/dml_core/struct.Pliv.html), [`Iivm`](https://docs.rs/dml-core/latest/dml_core/struct.Iivm.html), [`Did`](https://docs.rs/dml-core/latest/dml_core/struct.Did.html), [`DidCs`](https://docs.rs/dml-core/latest/dml_core/struct.DidCs.html), [`Ssm`](https://docs.rs/dml-core/latest/dml_core/struct.Ssm.html) | Models (builders with `n_folds`, `n_rep`, `seed`, `fit`, `fit_with_folds`) |
| [`Apo`](https://docs.rs/dml-core/latest/dml_core/struct.Apo.html), [`Apos`](https://docs.rs/dml-core/latest/dml_core/struct.Apos.html), [`Pq`](https://docs.rs/dml-core/latest/dml_core/struct.Pq.html), [`Lpq`](https://docs.rs/dml-core/latest/dml_core/struct.Lpq.html), [`Cvar`](https://docs.rs/dml-core/latest/dml_core/struct.Cvar.html), [`Qte`](https://docs.rs/dml-core/latest/dml_core/struct.Qte.html) | Average potential outcomes and quantile models |
| [`RdFlex`](https://docs.rs/dml-core/latest/dml_core/struct.RdFlex.html), [`rd`](https://docs.rs/dml-core/latest/dml_core/rd/index.html) | Regression discontinuity with covariate adjustment, and the `rdrobust` back end |
| [`StaggeredDid`](https://docs.rs/dml-core/latest/dml_core/struct.StaggeredDid.html) | Staggered difference-in-differences |
| [`datasets`](https://docs.rs/dml-core/latest/dml_core/datasets/index.html) | Simulated data sets (DoubleML's data generating processes) |
| [`DmlFit`](https://docs.rs/dml-core/latest/dml_core/struct.DmlFit.html) | Results: estimates, inference, bootstrap, sensitivity analysis, learner losses |
| [`Regressor`](https://docs.rs/dml-core/latest/dml_core/trait.Regressor.html), [`Classifier`](https://docs.rs/dml-core/latest/dml_core/trait.Classifier.html) | Learner traits, implemented by the built-in learners |
| [`LinearScoreModel`](https://docs.rs/dml-core/latest/dml_core/trait.LinearScoreModel.html) | Trait for adding a new orthogonal score |

The crate builds with Rust 1.80 or newer. Parallel cross-fitting is enabled by the default
`parallel` feature (rayon).

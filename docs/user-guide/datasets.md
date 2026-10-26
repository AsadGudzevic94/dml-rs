# Simulated data

`dml_rs.datasets` (Rust: `dml_core::datasets`) provides the data generating processes of
DoubleML's `make_*` functions, for examples, tests and simulation studies. The functions keep
DoubleML's names, parameters and defaults.

=== "Python"

    ```python
    from dml_rs.datasets import make_plr_CCDDHNR2018, make_did_CS2021

    x, y, d = make_plr_CCDDHNR2018(n_obs=500, dim_x=20, alpha=0.5, random_state=42)
    dml_rs.PLR(ml_l, ml_m).fit(x, y, d).summary

    df = make_did_CS2021(n_obs=1000, random_state=42)    # long-format panel
    x_cols = ["Z1", "Z2", "Z3", "Z4"]
    dml_rs.StaggeredDID(ml_g, ml_m).fit_frame(df, "y", "id", "t", "d", x_cols)
    ```

=== "Rust"

    ```rust
    use dml_core::datasets::{DidCs2021, PlrCcddhnr2018};

    let sim = PlrCcddhnr2018 { n_obs: 500, ..Default::default() }.generate(42)?;
    let fit = Plr::new(Ols::new(), Ols::new()).fit(&sim.to_data()?)?;

    let panel = DidCs2021::default().generate(42)?;
    let fit = StaggeredDid::new(Ols::new(), LogisticRegression::default()).fit(&panel.to_panel()?)?;
    ```

The data are drawn in Rust from a seeded generator. `random_state` makes them reproducible;
without it the seed is drawn from NumPy's global generator, so `np.random.seed` works as well.
The draws are not identical to DoubleML's, but the distributions are the same. The tests compare
moments and output formats with DoubleML's generators on large samples.

`return_type` is `"array"` (a tuple of arrays, the default) or `"DataFrame"` (pandas, with
DoubleML's column names). DoubleML's data classes have no counterpart, since dml-rs models take
arrays. Generators that DoubleML returns as a dict (with oracle values) or a DataFrame return the
same here.

| Function | Design | Returns |
|---|---|---|
| `make_plr_CCDDHNR2018` | partially linear regression (Chernozhukov et al., 2018) | `x, y, d` |
| `make_plr_turrell2018` | partially linear regression with a random covariance (Turrell, 2018) | `x, y, d` |
| `make_confounded_plr_data` | PLR with an unobserved confounder of given strength | dict with oracle values |
| `make_pliv_CHS2015` | partially linear IV with many instruments | `x, y, d, z` |
| `make_pliv_multiway_cluster_CKMS2021` | PLIV with two-way clustering | `x, y, d, clusters, z` |
| `make_lplr_LZZ2020` | logistic partially linear model, binary outcome | `x, y, d, p` |
| `make_plpr_CP2025` | static panel with correlated unit effects | DataFrame |
| `make_irm_data` | binary treatment | `x, y, d` |
| `make_confounded_irm_data` | binary treatment with an unobserved confounder | dict with oracle values |
| `make_irm_data_discrete_treatments` | multi-valued treatment | dict with oracle values |
| `make_iivm_data` | binary instrument (LATE) | `x, y, d, z` |
| `make_ssm_data` | sample selection, missing at random or nonignorable | `x, y, d, z, s` |
| `make_heterogeneous_data` | heterogeneous treatment effects | dict with data, effects and effect function |
| `make_did_SZ2020` | two-period DiD (Sant'Anna and Zhao, 2020), panel or cross-sections | `z, y, d, t` |
| `make_did_CS2021`, `make_did_cs_CS2021` | staggered adoption (Callaway and Sant'Anna, 2021) | DataFrame |
| `make_simple_rdd_data` | regression discontinuity | dict |

In Rust, each design is a struct with public fields (DoubleML's parameters, with
`Default`) and a `generate(seed)` method. The results convert to the library's data types:
`Simulated::to_data`, `SimulatedPanel::to_static_panel`, `SimulatedStaggered::to_panel` /
`to_cross_sections` and `SimulatedRdd::to_data`.

//! Simulated data sets for Python (`dml_rs.datasets`).

use dml_core::datasets::{
    ConfoundedIrm, ConfoundedPlr, DidCs2021, DidSz2020, HeterogeneousData, IivmData, IrmData,
    IrmDiscreteTreatments, LplrLzz2020, LplrTreatment, PlivChs2015, PlivMultiwayClusterCkms2021,
    PlprCp2025, PlprDesign, PlrCcddhnr2018, PlrTurrell2018, SimpleRdd, Simulated, SsmData,
};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::errors::to_py_err;

/// Parameter `key` of `params`, or `default`.
fn get<'py, T>(params: &Bound<'py, PyDict>, key: &str, default: T) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    match params.get_item(key)? {
        Some(v) if !v.is_none() => v.extract(),
        _ => Ok(default),
    }
}

fn put_simulated<'py>(py: Python<'py>, out: &Bound<'py, PyDict>, sim: Simulated) -> PyResult<()> {
    out.set_item("x", sim.x.into_pyarray(py))?;
    out.set_item("y", sim.y.into_pyarray(py))?;
    out.set_item("d", sim.d.into_pyarray(py))?;
    if let Some(z) = sim.z {
        out.set_item("z", z.into_pyarray(py))?;
    }
    if let Some(s) = sim.s {
        out.set_item("s", s.into_pyarray(py))?;
    }
    if let Some(t) = sim.t {
        out.set_item("t", t.into_pyarray(py))?;
    }
    if let Some(c) = sim.clusters {
        out.set_item("clusters", c.into_pyarray(py))?;
    }
    Ok(())
}

/// Draw the simulated data set `name` (a DoubleML `make_*` design) with the
/// parameters in `params` (DoubleML's names; missing ones take its defaults).
/// Returns the arrays (and oracle values) in a dict.
#[pyfunction]
pub(crate) fn simulate<'py>(
    py: Python<'py>,
    name: &str,
    params: &Bound<'py, PyDict>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    match name {
        "plr_CCDDHNR2018" => {
            let d = PlrCcddhnr2018::default();
            let g = PlrCcddhnr2018 {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                alpha: get(params, "alpha", d.alpha)?,
                a_0: get(params, "a_0", d.a_0)?,
                a_1: get(params, "a_1", d.a_1)?,
                s_1: get(params, "s_1", d.s_1)?,
                b_0: get(params, "b_0", d.b_0)?,
                b_1: get(params, "b_1", d.b_1)?,
                s_2: get(params, "s_2", d.s_2)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "plr_turrell2018" => {
            let d = PlrTurrell2018::default();
            let g = PlrTurrell2018 {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                theta: get(params, "theta", d.theta)?,
                nu: get(params, "nu", d.nu)?,
                gamma: get(params, "gamma", d.gamma)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "confounded_plr_data" => {
            let d = ConfoundedPlr::default();
            let g = ConfoundedPlr {
                n_obs: get(params, "n_obs", d.n_obs)?,
                theta: get(params, "theta", d.theta)?,
                cf_y: get(params, "cf_y", d.cf_y)?,
                cf_d: get(params, "cf_d", d.cf_d)?,
                c: get(params, "c", d.c)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
            };
            let sim = g.generate(seed).map_err(to_py_err)?;
            put_simulated(py, &out, sim.data)?;
            let o = sim.oracle;
            let oracle = PyDict::new(py);
            oracle.set_item("g_long", o.g_long.into_pyarray(py))?;
            oracle.set_item("g_short", o.g_short.into_pyarray(py))?;
            oracle.set_item("m_long", o.m_long.into_pyarray(py))?;
            oracle.set_item("m_short", o.m_short.into_pyarray(py))?;
            oracle.set_item("theta", o.theta)?;
            oracle.set_item("gamma_a", o.gamma_a)?;
            oracle.set_item("beta_a", o.beta_a)?;
            oracle.set_item("a", o.a.into_pyarray(py))?;
            oracle.set_item("z", o.z.into_pyarray(py))?;
            out.set_item("oracle_values", oracle)?;
        }
        "pliv_CHS2015" => {
            let d = PlivChs2015::default();
            let g = PlivChs2015 {
                n_obs: get(params, "n_obs", d.n_obs)?,
                alpha: get(params, "alpha", d.alpha)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                dim_z: get(params, "dim_z", d.dim_z)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "pliv_multiway_cluster_CKMS2021" => {
            let d = PlivMultiwayClusterCkms2021::default();
            let pair = |key: &str, default: [f64; 2]| -> PyResult<[f64; 2]> {
                let v: Vec<f64> = get(params, key, default.to_vec())?;
                v.try_into()
                    .map_err(|_| PyValueError::new_err(format!("{key} needs two values")))
            };
            let g = PlivMultiwayClusterCkms2021 {
                n: get(params, "N", d.n)?,
                m: get(params, "M", d.m)?,
                dim_x: get(params, "dim_X", d.dim_x)?,
                theta: get(params, "theta", d.theta)?,
                pi_10: get(params, "pi_10", d.pi_10)?,
                zeta_0: get(params, "zeta_0", Vec::new())?,
                pi_20: get(params, "pi_20", Vec::new())?,
                xi_0: get(params, "xi_0", Vec::new())?,
                omega_x: pair("omega_X", d.omega_x)?,
                omega_epsilon: pair("omega_epsilon", d.omega_epsilon)?,
                omega_v: pair("omega_v", d.omega_v)?,
                omega_vv: pair("omega_V", d.omega_vv)?,
                s_x: get(params, "s_X", d.s_x)?,
                s_epsilon_v: get(params, "s_epsilon_v", d.s_epsilon_v)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "lplr_LZZ2020" => {
            let d = LplrLzz2020::default();
            let treatment = match get(params, "treatment", "continuous".to_string())?.as_str() {
                "continuous" => LplrTreatment::Continuous,
                "binary" => LplrTreatment::Binary,
                "binary_unbalanced" => LplrTreatment::BinaryUnbalanced,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "invalid treatment type {other:?}"
                    )))
                }
            };
            let g = LplrLzz2020 {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                alpha: get(params, "alpha", d.alpha)?,
                balanced_r0: get(params, "balanced_r0", d.balanced_r0)?,
                treatment,
            };
            let (sim, p) = g.generate(seed).map_err(to_py_err)?;
            put_simulated(py, &out, sim)?;
            out.set_item("p", p.into_pyarray(py))?;
        }
        "plpr_CP2025" => {
            let d = PlprCp2025::default();
            let design = match get(params, "dgp_type", "dgp1".to_string())?.as_str() {
                "dgp1" => PlprDesign::Dgp1,
                "dgp2" => PlprDesign::Dgp2,
                "dgp3" => PlprDesign::Dgp3,
                other => return Err(PyValueError::new_err(format!("invalid dgp type {other:?}"))),
            };
            let g = PlprCp2025 {
                num_id: get(params, "num_id", d.num_id)?,
                num_t: get(params, "num_t", d.num_t)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                theta: get(params, "theta", d.theta)?,
                design,
            };
            let panel = g.generate(seed).map_err(to_py_err)?;
            out.set_item("id", panel.id.into_pyarray(py))?;
            out.set_item("t", panel.t.into_pyarray(py))?;
            out.set_item("y", panel.y.into_pyarray(py))?;
            out.set_item("d", panel.d.into_pyarray(py))?;
            out.set_item("x", panel.x.into_pyarray(py))?;
        }
        "irm_data" => {
            let d = IrmData::default();
            let g = IrmData {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                theta: get(params, "theta", d.theta)?,
                r2_d: get(params, "R2_d", d.r2_d)?,
                r2_y: get(params, "R2_y", d.r2_y)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "confounded_irm_data" => {
            let d = ConfoundedIrm::default();
            let g = ConfoundedIrm {
                n_obs: get(params, "n_obs", d.n_obs)?,
                theta: get(params, "theta", d.theta)?,
                gamma_a: get(params, "gamma_a", d.gamma_a)?,
                beta_a: get(params, "beta_a", d.beta_a)?,
                linear: get(params, "linear", d.linear)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                trimming_threshold: get(params, "trimming_threshold", d.trimming_threshold)?,
                var_eps_y: get(params, "var_eps_y", d.var_eps_y)?,
            };
            let sim = g.generate(seed).map_err(to_py_err)?;
            put_simulated(py, &out, sim.data)?;
            let o = sim.oracle;
            let oracle = PyDict::new(py);
            oracle.set_item("g_long", o.g_long.into_pyarray(py))?;
            oracle.set_item("g_short", o.g_short.into_pyarray(py))?;
            oracle.set_item("m_long", o.m_long.into_pyarray(py))?;
            oracle.set_item("m_short", o.m_short.into_pyarray(py))?;
            oracle.set_item("gamma_a", o.gamma_a)?;
            oracle.set_item("beta_a", o.beta_a)?;
            oracle.set_item("a", o.a.into_pyarray(py))?;
            oracle.set_item("y_0", o.y_0.into_pyarray(py))?;
            oracle.set_item("y_1", o.y_1.into_pyarray(py))?;
            oracle.set_item("z", o.z.into_pyarray(py))?;
            oracle.set_item("cf_y", o.cf_y)?;
            oracle.set_item("cf_d_ate", o.cf_d_ate)?;
            oracle.set_item("cf_d_atte", o.cf_d_atte)?;
            oracle.set_item("rho_ate", o.rho_ate)?;
            oracle.set_item("rho_atte", o.rho_atte)?;
            out.set_item("oracle_values", oracle)?;
        }
        "irm_data_discrete_treatments" => {
            let d = IrmDiscreteTreatments::default();
            let g = IrmDiscreteTreatments {
                n_obs: get(params, "n_obs", d.n_obs)?,
                n_levels: get(params, "n_levels", d.n_levels)?,
                linear: get(params, "linear", d.linear)?,
                xi: get(params, "xi", d.xi)?,
                c: get(params, "c", d.c)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
            };
            let sim = g.generate(seed).map_err(to_py_err)?;
            put_simulated(py, &out, sim.data)?;
            let o = sim.oracle;
            let oracle = PyDict::new(py);
            oracle.set_item("cont_d", o.cont_d.into_pyarray(py))?;
            oracle.set_item("level_bounds", o.level_bounds.into_pyarray(py))?;
            oracle.set_item("potential_level", o.potential_level.into_pyarray(py))?;
            oracle.set_item("ite", o.ite.into_pyarray(py))?;
            oracle.set_item("y0", o.y0.into_pyarray(py))?;
            out.set_item("oracle_values", oracle)?;
        }
        "iivm_data" => {
            let d = IivmData::default();
            let g = IivmData {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                theta: get(params, "theta", d.theta)?,
                alpha_x: get(params, "alpha_x", d.alpha_x)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "ssm_data" => {
            let d = SsmData::default();
            let g = SsmData {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                theta: get(params, "theta", d.theta)?,
                mar: get(params, "mar", d.mar)?,
            };
            put_simulated(py, &out, g.generate(seed).map_err(to_py_err)?)?;
        }
        "heterogeneous_data" => {
            let d = HeterogeneousData::default();
            let g = HeterogeneousData {
                n_obs: get(params, "n_obs", d.n_obs)?,
                p: get(params, "p", d.p)?,
                support_size: get(params, "support_size", d.support_size)?,
                n_x: get(params, "n_x", d.n_x)?,
                binary_treatment: get(params, "binary_treatment", d.binary_treatment)?,
            };
            let sim = g.generate(seed).map_err(to_py_err)?;
            put_simulated(py, &out, sim.data)?;
            out.set_item("effects", sim.effects.into_pyarray(py))?;
        }
        "did_SZ2020" => {
            let d = DidSz2020::default();
            let g = DidSz2020 {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dgp_type: get(params, "dgp_type", d.dgp_type)?,
                xi: get(params, "xi", d.xi)?,
                c: get(params, "c", d.c)?,
                lambda_t: get(params, "lambda_t", d.lambda_t)?,
            };
            let sim = if get(params, "cross_sectional_data", false)? {
                g.generate_cross_sections(seed)
            } else {
                g.generate(seed)
            };
            put_simulated(py, &out, sim.map_err(to_py_err)?)?;
        }
        "did_CS2021" | "did_cs_CS2021" => {
            let d = DidCs2021::default();
            let g = DidCs2021 {
                n_obs: get(params, "n_obs", d.n_obs)?,
                dgp_type: get(params, "dgp_type", d.dgp_type)?,
                include_never_treated: get(
                    params,
                    "include_never_treated",
                    d.include_never_treated,
                )?,
                c: get(params, "c", d.c)?,
                xi: get(params, "xi", d.xi)?,
                n_periods: get(params, "n_periods", d.n_periods)?,
                anticipation_periods: get(params, "anticipation_periods", d.anticipation_periods)?,
                n_pre_treat_periods: get(params, "n_pre_treat_periods", d.n_pre_treat_periods)?,
            };
            let panel = if name == "did_cs_CS2021" {
                g.generate_cross_sections(seed, get(params, "lambda_t", 0.5)?)
            } else {
                g.generate(seed)
            }
            .map_err(to_py_err)?;
            out.set_item("id", panel.id.into_pyarray(py))?;
            out.set_item("t", panel.t.into_pyarray(py))?;
            out.set_item("y", panel.y.into_pyarray(py))?;
            out.set_item("y0", panel.y0.into_pyarray(py))?;
            out.set_item("y1", panel.y1.into_pyarray(py))?;
            out.set_item("d", panel.cohort.into_pyarray(py))?;
            out.set_item("x", panel.x.into_pyarray(py))?;
        }
        "simple_rdd_data" => {
            let d = SimpleRdd::default();
            let g = SimpleRdd {
                n_obs: get(params, "n_obs", d.n_obs)?,
                p: get(params, "p", d.p)?,
                fuzzy: get(params, "fuzzy", d.fuzzy)?,
                binary_outcome: get(params, "binary_outcome", d.binary_outcome)?,
                cutoff: get(params, "cutoff", d.cutoff)?,
                dim_x: get(params, "dim_x", d.dim_x)?,
                a: get(params, "a", d.a)?,
                tau: get(params, "tau", d.tau)?,
            };
            let sim = g.generate(seed).map_err(to_py_err)?;
            out.set_item("score", sim.score.into_pyarray(py))?;
            out.set_item("x", sim.x.into_pyarray(py))?;
            out.set_item("y", sim.y.into_pyarray(py))?;
            out.set_item("d", sim.d.into_pyarray(py))?;
            out.set_item("y0", sim.y0.into_pyarray(py))?;
            out.set_item("y1", sim.y1.into_pyarray(py))?;
        }
        other => return Err(PyValueError::new_err(format!("unknown data set {other:?}"))),
    }
    Ok(out)
}

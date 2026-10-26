//! Simulated data sets: the data generating processes of DoubleML's
//! `make_*` functions, for examples, tests and simulation studies.
//!
//! Every generator is a struct whose fields are the parameters, with
//! DoubleML's defaults, and `generate(seed)` draws a data set. The draws come
//! from a seeded ChaCha generator, so they are reproducible but not identical
//! to DoubleML's NumPy draws; the distributions are the same.
//!
//! ```
//! use dml_core::datasets::PlrCcddhnr2018;
//! use dml_core::{learner::Ols, Plr};
//!
//! let sim = PlrCcddhnr2018 { n_obs: 300, dim_x: 5, ..Default::default() }.generate(42).unwrap();
//! let fit = Plr::new(Ols::new(), Ols::new()).fit(&sim.to_data().unwrap()).unwrap();
//! assert!((fit.coef()[0] - 0.5).abs() < 0.3);
//! ```
//!
//! | Generator | DoubleML | Design |
//! |---|---|---|
//! | [`PlrCcddhnr2018`] | `make_plr_CCDDHNR2018` | partially linear regression |
//! | [`PlrTurrell2018`] | `make_plr_turrell2018` | partially linear regression |
//! | [`ConfoundedPlr`] | `make_confounded_plr_data` | PLR with an unobserved confounder |
//! | [`PlivChs2015`] | `make_pliv_CHS2015` | partially linear IV, many instruments |
//! | [`PlivMultiwayClusterCkms2021`] | `make_pliv_multiway_cluster_CKMS2021` | PLIV with two-way clustering |
//! | [`LplrLzz2020`] | `make_lplr_LZZ2020` | logistic partially linear regression |
//! | [`PlprCp2025`] | `make_plpr_CP2025` | partially linear panel regression |
//! | [`IrmData`] | `make_irm_data` | interactive regression, binary treatment |
//! | [`ConfoundedIrm`] | `make_confounded_irm_data` | IRM with an unobserved confounder |
//! | [`IrmDiscreteTreatments`] | `make_irm_data_discrete_treatments` | multi-valued treatment |
//! | [`IivmData`] | `make_iivm_data` | binary instrument (LATE) |
//! | [`SsmData`] | `make_ssm_data` | sample selection |
//! | [`HeterogeneousData`] | `make_heterogeneous_data` | heterogeneous treatment effects |
//! | [`DidSz2020`] | `make_did_SZ2020` | two-period difference-in-differences |
//! | [`DidCs2021`] | `make_did_CS2021`, `make_did_cs_CS2021` | staggered adoption |
//! | [`SimpleRdd`] | `make_simple_rdd_data` | regression discontinuity |

use ndarray::{s, Array1, Array2, Axis};
use rand::seq::index::sample;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::compat::{brent_minimize, np_quantile};
use crate::data::DmlData;
use crate::error::{DmlError, Result};
use crate::linalg::{cholesky_factor, sym_eigen};
use crate::model::StaticPanel;
use crate::staggered::{PanelData, RepeatedCrossSections};

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

/// A simulated cross-section: covariates `x`, outcome `y`, treatment `d` and,
/// depending on the design, instruments `z`, a selection indicator `s`, a
/// time indicator `t` or cluster labels (one column per clustering).
#[derive(Debug, Clone)]
pub struct Simulated {
    pub x: Array2<f64>,
    pub y: Array1<f64>,
    pub d: Array1<f64>,
    pub z: Option<Array2<f64>>,
    pub s: Option<Array1<f64>>,
    pub t: Option<Array1<f64>>,
    pub clusters: Option<Array2<i64>>,
}

impl Simulated {
    fn new(x: Array2<f64>, y: Array1<f64>, d: Array1<f64>) -> Self {
        Self {
            x,
            y,
            d,
            z: None,
            s: None,
            t: None,
            clusters: None,
        }
    }

    /// The data set as [`DmlData`], with the instruments, selection or time
    /// indicator and clusters attached.
    pub fn to_data(&self) -> Result<DmlData> {
        let mut data = DmlData::new(self.y.clone(), self.d.clone(), self.x.clone())?;
        if let Some(z) = &self.z {
            data = data.with_instruments(z.clone())?;
        }
        if let Some(s) = &self.s {
            data = data.with_selection_indicator(s.clone())?;
        }
        if let Some(t) = &self.t {
            data = data.with_time_indicator(t.clone())?;
        }
        if let Some(c) = &self.clusters {
            data = match c.ncols() {
                1 => data.with_clusters(c.column(0).to_owned())?,
                _ => data.with_two_way_clusters(c.column(0).to_owned(), c.column(1).to_owned())?,
            };
        }
        Ok(data)
    }
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn normal(rng: &mut ChaCha8Rng) -> f64 {
    rng.sample(StandardNormal)
}

fn normals(rng: &mut ChaCha8Rng, n: usize) -> Array1<f64> {
    (0..n).map(|_| normal(rng)).collect()
}

fn uniforms(rng: &mut ChaCha8Rng, n: usize, lo: f64, hi: f64) -> Array1<f64> {
    (0..n).map(|_| rng.gen_range(lo..hi)).collect()
}

fn bernoulli(rng: &mut ChaCha8Rng, p: f64) -> f64 {
    f64::from(rng.gen::<f64>() < p)
}

fn expit(v: f64) -> f64 {
    1.0 / (1.0 + (-v).exp())
}

/// `c^|i−j|` (with `0⁰ = 1`).
fn toeplitz(c: f64, p: usize) -> Array2<f64> {
    Array2::from_shape_fn((p, p), |(i, j)| c.powi(i.abs_diff(j) as i32))
}

/// `n` draws from `N(0, cov)` (rows).
fn mvn(rng: &mut ChaCha8Rng, n: usize, cov: &Array2<f64>) -> Result<Array2<f64>> {
    let l = cholesky_factor(cov)?;
    let p = cov.nrows();
    let e = Array2::from_shape_fn((n, p), |_| normal(rng));
    Ok(e.dot(&l.t()))
}

/// Pairs from a bivariate normal with unit variances and correlation `rho`.
fn correlated_pair(rng: &mut ChaCha8Rng, n: usize, rho: f64) -> (Array1<f64>, Array1<f64>) {
    let a = normals(rng, n);
    let b = normals(rng, n);
    let second = &a * rho + &b * (1.0 - rho * rho).sqrt();
    (a, second)
}

/// Columns centred and scaled by their population standard deviation.
fn standardize(z: &Array2<f64>) -> Array2<f64> {
    let mean = z.mean_axis(Axis(0)).expect("rows");
    let sd = z.std_axis(Axis(0), 0.0);
    (z - &mean) / &sd
}

/// The transformed covariates of Sant'Anna and Zhao (2020), standardised;
/// columns of `x` from `extra_from` on are appended unchanged first.
fn sz_features(x: &Array2<f64>, extra_from: Option<usize>) -> Array2<f64> {
    let n = x.nrows();
    let extra = extra_from.map_or(0, |k| x.ncols().saturating_sub(k));
    let mut z = Array2::zeros((n, 4 + extra));
    for i in 0..n {
        let (x0, x1, x2, x3) = (x[[i, 0]], x[[i, 1]], x[[i, 2]], x[[i, 3]]);
        z[[i, 0]] = (0.5 * x0).exp();
        z[[i, 1]] = 10.0 + x1 / (1.0 + x0.exp());
        z[[i, 2]] = (0.6 + x0 * x2 / 25.0).powi(3);
        z[[i, 3]] = (20.0 + x1 + x3).powi(2);
        for j in 0..extra {
            z[[i, 4 + j]] = x[[i, extra_from.expect("extra") + j]];
        }
    }
    standardize(&z)
}

fn check(cond: bool, msg: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(DmlError::InvalidArgument(msg.into()))
    }
}

fn mean(v: &Array1<f64>) -> f64 {
    v.mean().expect("non-empty")
}

fn var(v: &Array1<f64>) -> f64 {
    let m = mean(v);
    v.mapv(|x| (x - m) * (x - m)).mean().expect("non-empty")
}

fn corr(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let cov = (a - ma).dot(&(b - mb));
    cov / ((a - ma).dot(&(a - ma)) * (b - mb).dot(&(b - mb))).sqrt()
}

// ---------------------------------------------------------------------------
// Partially linear models
// ---------------------------------------------------------------------------

/// Chernozhukov et al. (2018): `d = a₀x₀ + a₁·expit(x₂) + s₁ε`,
/// `y = αd + b₀·expit(x₀) + b₁x₂ + s₂ζ` with Toeplitz-correlated
/// covariates (`0.7^|i−j|`).
#[derive(Debug, Clone, PartialEq)]
pub struct PlrCcddhnr2018 {
    pub n_obs: usize,
    pub dim_x: usize,
    pub alpha: f64,
    pub a_0: f64,
    pub a_1: f64,
    pub s_1: f64,
    pub b_0: f64,
    pub b_1: f64,
    pub s_2: f64,
}

impl Default for PlrCcddhnr2018 {
    fn default() -> Self {
        Self {
            n_obs: 500,
            dim_x: 20,
            alpha: 0.5,
            a_0: 1.0,
            a_1: 0.25,
            s_1: 1.0,
            b_0: 1.0,
            b_1: 0.25,
            s_2: 1.0,
        }
    }
}

impl PlrCcddhnr2018 {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        check(self.dim_x >= 3, "dim_x must be at least 3")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let x = mvn(&mut rng, n, &toeplitz(0.7, self.dim_x))?;
        let d: Array1<f64> = (0..n)
            .map(|i| {
                self.a_0 * x[[i, 0]] + self.a_1 * expit(x[[i, 2]]) + self.s_1 * normal(&mut rng)
            })
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| {
                self.alpha * d[i]
                    + self.b_0 * expit(x[[i, 0]])
                    + self.b_1 * x[[i, 2]]
                    + self.s_2 * normal(&mut rng)
            })
            .collect();
        Ok(Simulated::new(x, y, d))
    }
}

/// Turrell (2018): `d = m(x'b) + ε`, `y = θd + g(x'b) + ζ` with
/// `g(v) = sin²(v)`, `m(v) = sinh(γ) / (2π(cosh(γ) − cos(v − ν)))`,
/// `b_k = 1/k` and a random positive definite covariance (scikit-learn's
/// `make_spd_matrix`).
#[derive(Debug, Clone, PartialEq)]
pub struct PlrTurrell2018 {
    pub n_obs: usize,
    pub dim_x: usize,
    pub theta: f64,
    pub nu: f64,
    pub gamma: f64,
}

impl Default for PlrTurrell2018 {
    fn default() -> Self {
        Self {
            n_obs: 100,
            dim_x: 20,
            theta: 0.5,
            nu: 0.0,
            gamma: 1.0,
        }
    }
}

/// scikit-learn's `make_spd_matrix`: `V (J + diag(u)) Vᵀ` with `V` the
/// eigenvectors of `AᵀA` for uniform `A` and `J` the all-ones matrix.
fn spd_matrix(rng: &mut ChaCha8Rng, p: usize) -> Array2<f64> {
    let a = Array2::from_shape_fn((p, p), |_| rng.gen::<f64>());
    let (_, v) = sym_eigen(&a.t().dot(&a));
    let mut middle = Array2::<f64>::ones((p, p));
    for i in 0..p {
        middle[[i, i]] += rng.gen::<f64>();
    }
    let out = v.dot(&middle).dot(&v.t());
    // Symmetrise rounding.
    (&out + &out.t()) / 2.0
}

impl PlrTurrell2018 {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        check(self.dim_x >= 1, "dim_x must be at least 1")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let sigma = spd_matrix(&mut rng, self.dim_x);
        let x = mvn(&mut rng, n, &sigma)?;
        let b: Array1<f64> = (1..=self.dim_x).map(|k| 1.0 / k as f64).collect();
        let xb = x.dot(&b);
        let m = |v: f64| {
            0.5 / std::f64::consts::PI * self.gamma.sinh()
                / (self.gamma.cosh() - (v - self.nu).cos())
        };
        let d: Array1<f64> = xb.iter().map(|&v| m(v) + normal(&mut rng)).collect();
        let y: Array1<f64> = (0..n)
            .map(|i| self.theta * d[i] + xb[i].sin().powi(2) + normal(&mut rng))
            .collect();
        Ok(Simulated::new(x, y, d))
    }
}

/// A partially linear model with an unobserved confounder `A` of given
/// strength (`cf_y`, `cf_d`), as `make_confounded_plr_data`, for sensitivity
/// analysis. `ConfoundedPlrSimulated::oracle` holds the long and short
/// regressions and the confounder.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfoundedPlr {
    pub n_obs: usize,
    pub theta: f64,
    pub cf_y: f64,
    pub cf_d: f64,
    /// Correlation parameter of the covariates (`c^|i−j|`).
    pub c: f64,
    pub dim_x: usize,
}

impl Default for ConfoundedPlr {
    fn default() -> Self {
        Self {
            n_obs: 500,
            theta: 5.0,
            cf_y: 0.04,
            cf_d: 0.04,
            c: 0.0,
            dim_x: 4,
        }
    }
}

/// Oracle values of [`ConfoundedPlr`].
#[derive(Debug, Clone)]
pub struct ConfoundedPlrOracle {
    pub g_long: Array1<f64>,
    pub g_short: Array1<f64>,
    pub m_long: Array1<f64>,
    pub m_short: Array1<f64>,
    pub theta: f64,
    pub gamma_a: f64,
    pub beta_a: f64,
    pub a: Array1<f64>,
    pub z: Array2<f64>,
}

/// A data set of [`ConfoundedPlr`].
#[derive(Debug, Clone)]
pub struct ConfoundedPlrSimulated {
    pub data: Simulated,
    pub oracle: ConfoundedPlrOracle,
}

impl ConfoundedPlr {
    pub fn generate(&self, seed: u64) -> Result<ConfoundedPlrSimulated> {
        check(self.dim_x >= 4, "dim_x must be at least 4")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let x = mvn(&mut rng, n, &toeplitz(self.c, self.dim_x))?;
        let z = sz_features(&x, Some(4));
        let (var_eps_y, var_eps_d) = (5.0, 1.0);
        let eps_y = normals(&mut rng, n) * f64::sqrt(var_eps_y);
        let eps_d = normals(&mut rng, n) * f64::sqrt(var_eps_d);
        let a = uniforms(&mut rng, n, -1.0, 1.0);
        let var_a = 4.0 / 12.0;
        let m_short: Array1<f64> = (0..n)
            .map(|i| -z[[i, 0]] + 0.5 * z[[i, 1]] - 0.25 * z[[i, 2]] - 0.1 * z[[i, 3]])
            .collect();
        let cf_d = self.cf_d;
        let mut f_m = |gamma_a: f64| {
            let rr_long = &eps_d / var_eps_d;
            let rr_short = (&a * gamma_a + &eps_d) / (gamma_a * gamma_a * var_a + var_eps_d);
            let (l2, s2) = (
                rr_long.mapv(|v| v * v).mean().unwrap(),
                rr_short.mapv(|v| v * v).mean().unwrap(),
            );
            let c2 = (l2 - s2) / s2;
            (c2 / (1.0 + c2) - cf_d).powi(2)
        };
        let gamma_a = brent_minimize(&mut f_m, 0.0, 1.0);
        let m_long = &m_short + &(&a * gamma_a);
        let d = &m_long + &eps_d;
        let g_partial: Array1<f64> = (0..n)
            .map(|i| 210.0 + 27.4 * z[[i, 0]] + 13.7 * (z[[i, 1]] + z[[i, 2]] + z[[i, 3]]))
            .collect();
        let var_d = var(&d);
        let cf_y = self.cf_y;
        let mut f_g = |beta_a: f64| {
            let g_diff = (&a - &(&d * (gamma_a * var_a / var_d))) * beta_a;
            let y_diff = &eps_y + &g_diff;
            let ratio =
                g_diff.mapv(|v| v * v).mean().unwrap() / y_diff.mapv(|v| v * v).mean().unwrap();
            (ratio - cf_y).powi(2)
        };
        let beta_a = brent_minimize(&mut f_g, 0.0, 1.0);
        let g_long = &d * self.theta + &g_partial + &(&a * beta_a);
        let g_short = &d * (self.theta + gamma_a * beta_a * var_a / var_d) + &g_partial;
        let y = &g_long + &eps_y;
        Ok(ConfoundedPlrSimulated {
            data: Simulated::new(x, y, d),
            oracle: ConfoundedPlrOracle {
                g_long,
                g_short,
                m_long,
                m_short,
                theta: self.theta,
                gamma_a,
                beta_a,
                a,
                z,
            },
        })
    }
}

/// Chernozhukov, Hansen and Spindler (2015): partially linear IV with
/// `dim_z` instruments `z = x[:, :dim_z] + ξ`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlivChs2015 {
    pub n_obs: usize,
    pub alpha: f64,
    pub dim_x: usize,
    pub dim_z: usize,
}

impl Default for PlivChs2015 {
    fn default() -> Self {
        Self {
            n_obs: 500,
            alpha: 1.0,
            dim_x: 200,
            dim_z: 150,
        }
    }
}

impl PlivChs2015 {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        check(
            self.dim_z >= 1 && self.dim_x >= self.dim_z,
            "need 1 <= dim_z <= dim_x",
        )?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let (epsilon, u) = correlated_pair(&mut rng, n, 0.6);
        let x = mvn(&mut rng, n, &toeplitz(0.5, self.dim_x))?;
        let xi = Array2::from_shape_fn((n, self.dim_z), |_| 0.5 * normal(&mut rng));
        let beta: Array1<f64> = (1..=self.dim_x).map(|k| 1.0 / (k * k) as f64).collect();
        let delta: Array1<f64> = (1..=self.dim_z).map(|k| 1.0 / (k * k) as f64).collect();
        let z = &x.slice(s![.., ..self.dim_z]) + &xi;
        let xb = x.dot(&beta);
        let d = &xb + &z.dot(&delta) + &u;
        let y = &d * self.alpha + &xb + &epsilon;
        let mut sim = Simulated::new(x, y, d);
        sim.z = Some(z);
        Ok(sim)
    }
}

/// Chiang, Kato, Ma and Sasaki (2021): partially linear IV on an `n × m`
/// grid of two-way clusters, with cluster-correlated covariates and errors.
#[derive(Debug, Clone, PartialEq)]
pub struct PlivMultiwayClusterCkms2021 {
    /// Clusters of the first dimension (DoubleML's `N`).
    pub n: usize,
    /// Clusters of the second dimension (DoubleML's `M`).
    pub m: usize,
    pub dim_x: usize,
    pub theta: f64,
    pub pi_10: f64,
    /// Coefficients (default `0.5^k`, `k = 1..dim_x`); empty for the default.
    pub zeta_0: Vec<f64>,
    pub pi_20: Vec<f64>,
    pub xi_0: Vec<f64>,
    /// Weights of the two cluster dimensions in each component.
    pub omega_x: [f64; 2],
    pub omega_epsilon: [f64; 2],
    pub omega_v: [f64; 2],
    pub omega_vv: [f64; 2],
    pub s_x: f64,
    pub s_epsilon_v: f64,
}

impl Default for PlivMultiwayClusterCkms2021 {
    fn default() -> Self {
        Self {
            n: 25,
            m: 25,
            dim_x: 100,
            theta: 1.0,
            pi_10: 1.0,
            zeta_0: Vec::new(),
            pi_20: Vec::new(),
            xi_0: Vec::new(),
            omega_x: [0.25, 0.25],
            omega_epsilon: [0.25, 0.25],
            omega_v: [0.25, 0.25],
            omega_vv: [0.25, 0.25],
            s_x: 0.25,
            s_epsilon_v: 0.25,
        }
    }
}

impl PlivMultiwayClusterCkms2021 {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        let (nn, mm, p) = (self.n, self.m, self.dim_x);
        check(
            nn >= 1 && mm >= 1 && p >= 1,
            "n, m and dim_x must be positive",
        )?;
        let default = |v: &Vec<f64>| -> Result<Array1<f64>> {
            if v.is_empty() {
                Ok((1..=p).map(|k| 0.5_f64.powi(k as i32)).collect())
            } else if v.len() == p {
                Ok(Array1::from(v.clone()))
            } else {
                Err(DmlError::InvalidArgument(
                    "coefficient vectors need dim_x entries".into(),
                ))
            }
        };
        let (zeta_0, pi_20, xi_0) = (
            default(&self.zeta_0)?,
            default(&self.pi_20)?,
            default(&self.xi_0)?,
        );
        let mut rng = rng(seed);
        let rows = nn * mm;
        // Row r is the pair (i, j) = (r / m, r % m).
        let ij = |r: usize| (r / mm, r % mm);
        let alpha_vv = normals(&mut rng, rows);
        let alpha_vv_i = normals(&mut rng, nn);
        let alpha_vv_j = normals(&mut rng, mm);
        let pair = |rng: &mut ChaCha8Rng, k: usize| correlated_pair(rng, k, self.s_epsilon_v);
        let (alpha_eps, alpha_v) = pair(&mut rng, rows);
        let (alpha_eps_i, alpha_v_i) = pair(&mut rng, nn);
        let (alpha_eps_j, alpha_v_j) = pair(&mut rng, mm);
        let cov = toeplitz(self.s_x, p);
        let alpha_x = mvn(&mut rng, rows, &cov)?;
        let alpha_x_i = mvn(&mut rng, nn, &cov)?;
        let alpha_x_j = mvn(&mut rng, mm, &cov)?;
        let mix =
            |w: [f64; 2], own: f64, a: f64, b: f64| (1.0 - w[0] - w[1]) * own + w[0] * a + w[1] * b;
        let x = Array2::from_shape_fn((rows, p), |(r, k)| {
            let (i, j) = ij(r);
            mix(
                self.omega_x,
                alpha_x[[r, k]],
                alpha_x_i[[i, k]],
                alpha_x_j[[j, k]],
            )
        });
        let comp =
            |w: [f64; 2], own: &Array1<f64>, a: &Array1<f64>, b: &Array1<f64>| -> Array1<f64> {
                (0..rows)
                    .map(|r| {
                        let (i, j) = ij(r);
                        mix(w, own[r], a[i], b[j])
                    })
                    .collect()
            };
        let eps = comp(self.omega_epsilon, &alpha_eps, &alpha_eps_i, &alpha_eps_j);
        let v = comp(self.omega_v, &alpha_v, &alpha_v_i, &alpha_v_j);
        let vv = comp(self.omega_vv, &alpha_vv, &alpha_vv_i, &alpha_vv_j);
        let z = x.dot(&xi_0) + &vv;
        let d = &z * self.pi_10 + &x.dot(&pi_20) + &v;
        let y = &d * self.theta + &x.dot(&zeta_0) + &eps;
        let clusters = Array2::from_shape_fn((rows, 2), |(r, c)| {
            let (i, j) = ij(r);
            if c == 0 {
                i as i64
            } else {
                j as i64
            }
        });
        let mut sim = Simulated::new(x, y, d);
        sim.z = Some(z.insert_axis(Axis(1)));
        sim.clusters = Some(clusters);
        Ok(sim)
    }
}

/// Treatment of [`LplrLzz2020`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LplrTreatment {
    /// `d = a₀(x)`.
    #[default]
    Continuous,
    /// `d ~ Bernoulli(expit(a₀(x) − mean))`.
    Binary,
    /// `d ~ Bernoulli(expit(a₀(x)))`.
    BinaryUnbalanced,
}

/// Liu, Zhang and Zhou (2020): a binary outcome with
/// `P(y = 1) = expit(αd + r₀(x))`, equicorrelated covariates (0.2) clipped
/// to `[−2, 2]`. `generate` also returns the outcome probabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct LplrLzz2020 {
    pub n_obs: usize,
    pub dim_x: usize,
    pub alpha: f64,
    pub balanced_r0: bool,
    pub treatment: LplrTreatment,
}

impl Default for LplrLzz2020 {
    fn default() -> Self {
        Self {
            n_obs: 500,
            dim_x: 20,
            alpha: 0.5,
            balanced_r0: true,
            treatment: LplrTreatment::Continuous,
        }
    }
}

impl LplrLzz2020 {
    /// The data set and the outcome probabilities `p`.
    pub fn generate(&self, seed: u64) -> Result<(Simulated, Array1<f64>)> {
        check(self.dim_x >= 13, "dim_x must be at least 13")?;
        let mut rng = rng(seed);
        let (n, p) = (self.n_obs, self.dim_x);
        let sigma = Array2::from_shape_fn((p, p), |(i, j)| if i == j { 1.0 } else { 0.2 });
        let x = mvn(&mut rng, n, &sigma)?.mapv(|v| v.clamp(-2.0, 2.0));
        let ind = |v: f64| f64::from(v > 0.0);
        let (big, step) = if self.balanced_r0 {
            (1.0, 0.25)
        } else {
            (4.0, 1.5)
        };
        let r0 = |i: usize| {
            let r = x.row(i);
            0.1 * r[0] * r[1] * r[2] + 0.1 * r[3] * r[4] + 0.1 * r[5].powi(3)
                - 0.5 * r[6].sin().powi(2)
                + 0.5 * r[7].cos()
                + big / (1.0 + r[8] * r[8])
                - 1.0 / (1.0 + r[9].exp())
                + step * ind(r[10])
                - 0.25 * ind(r[12])
        };
        let a0: Array1<f64> = (0..n)
            .map(|i| {
                let r = x.row(i);
                2.0 / (1.0 + r[0].exp()) - 2.0 / (1.0 + r[1].exp())
                    + r[2].sin()
                    + r[3].cos()
                    + 0.5 * ind(r[4])
                    - 0.5 * ind(r[5])
                    + 0.2 * r[6] * r[7]
                    - 0.2 * r[8] * r[9]
            })
            .collect();
        let d = match self.treatment {
            LplrTreatment::Continuous => a0,
            LplrTreatment::Binary => {
                let m = mean(&a0);
                a0.mapv(|v| bernoulli(&mut rng, expit(v - m)))
            }
            LplrTreatment::BinaryUnbalanced => a0.mapv(|v| bernoulli(&mut rng, expit(v))),
        };
        let prob: Array1<f64> = (0..n).map(|i| expit(self.alpha * d[i] + r0(i))).collect();
        let y = prob.mapv(|q| bernoulli(&mut rng, q));
        Ok((Simulated::new(x, y, d), prob))
    }
}

/// Functional form of [`PlprCp2025`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlprDesign {
    /// Linear nuisances.
    #[default]
    Dgp1,
    /// Smooth non-linear nuisances.
    Dgp2,
    /// Interactions and kinks.
    Dgp3,
}

/// Clarke and Polselli (2025): a static panel with correlated unit effects,
/// `d = m₀(x) + cᵢ + v`, `y = θd + g₀(x) + αᵢ + u`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlprCp2025 {
    pub num_id: usize,
    pub num_t: usize,
    pub dim_x: usize,
    pub theta: f64,
    pub design: PlprDesign,
}

impl Default for PlprCp2025 {
    fn default() -> Self {
        Self {
            num_id: 250,
            num_t: 10,
            dim_x: 30,
            theta: 0.5,
            design: PlprDesign::Dgp1,
        }
    }
}

/// A simulated panel in long format (one row per unit and period).
#[derive(Debug, Clone)]
pub struct SimulatedPanel {
    pub id: Array1<i64>,
    pub t: Array1<f64>,
    pub y: Array1<f64>,
    pub d: Array1<f64>,
    pub x: Array2<f64>,
}

impl SimulatedPanel {
    /// The panel as [`StaticPanel`] for [`Plpr`](crate::Plpr).
    pub fn to_static_panel(&self) -> Result<StaticPanel> {
        StaticPanel::new(
            self.y.clone(),
            self.d.clone(),
            self.x.clone(),
            self.id.clone(),
            self.t.clone(),
        )
    }
}

impl PlprCp2025 {
    pub fn generate(&self, seed: u64) -> Result<SimulatedPanel> {
        check(self.dim_x >= 3, "dim_x must be at least 3")?;
        check(
            self.num_id >= 1 && self.num_t >= 1,
            "num_id and num_t must be positive",
        )?;
        let mut rng = rng(seed);
        let (a, b, sigma2_a, sigma2_x) = (0.25, 0.5, 0.95_f64, 5.0_f64);
        let (ni, nt) = (self.num_id, self.num_t);
        let rows = ni * nt;
        let a_i: Vec<f64> = (0..ni)
            .map(|_| sigma2_a.sqrt() * normal(&mut rng))
            .collect();
        let c_i: Vec<f64> = (0..ni).map(|_| normal(&mut rng)).collect();
        let x = Array2::from_shape_fn((rows, self.dim_x), |_| sigma2_x.sqrt() * normal(&mut rng));
        let u = normals(&mut rng, rows);
        let v = normals(&mut rng, rows);
        let logistic = |v: f64| v.exp() / (1.0 + v.exp());
        let pos = |v: f64| f64::from(v > 0.0);
        let (g0, m0): (Vec<f64>, Vec<f64>) = (0..rows)
            .map(|r| {
                let (x0, x2) = (x[[r, 0]], x[[r, 2]]);
                match self.design {
                    PlprDesign::Dgp1 => (a * x0 + x2, a * x0 + x2),
                    PlprDesign::Dgp2 => (logistic(x0) + a * x2.cos(), x0.cos() + a * logistic(x2)),
                    PlprDesign::Dgp3 => (
                        b * (x0 * x2) + a * (x2 * pos(x2)),
                        a * (x0 * pos(x0)) + b * (x0 * x2),
                    ),
                }
            })
            .unzip();
        let d: Array1<f64> = (0..rows).map(|r| m0[r] + c_i[r / nt] + v[r]).collect();
        let d_mean = mean(&d);
        let unit_mean = |f: &dyn Fn(usize) -> f64, i: usize| {
            (0..nt).map(|k| f(i * nt + k)).sum::<f64>() / nt as f64
        };
        let alpha: Vec<f64> = (0..ni)
            .map(|i| {
                let d_term = unit_mean(&|r| d[r], i) - d_mean;
                let x_term = unit_mean(&|r| x[[r, 0]] + x[[r, 2]], i);
                0.25 * d_term + 0.25 * x_term + a_i[i]
            })
            .collect();
        let y: Array1<f64> = (0..rows)
            .map(|r| d[r] * self.theta + g0[r] + alpha[r / nt] + u[r])
            .collect();
        Ok(SimulatedPanel {
            id: (0..rows).map(|r| (r / nt + 1) as i64).collect(),
            t: (0..rows).map(|r| (r % nt + 1) as f64).collect(),
            y,
            d,
            x,
        })
    }
}

// ---------------------------------------------------------------------------
// Interactive models
// ---------------------------------------------------------------------------

/// Belloni, Chernozhukov, Fernández-Val and Hansen (2017): binary treatment
/// with `P(d = 1 | x) = expit(c_d x'β)` and `y = θd + d·c_y x'β + ζ`, with
/// `c_d`, `c_y` set by the target `R²` values.
#[derive(Debug, Clone, PartialEq)]
pub struct IrmData {
    pub n_obs: usize,
    pub dim_x: usize,
    pub theta: f64,
    pub r2_d: f64,
    pub r2_y: f64,
}

impl Default for IrmData {
    fn default() -> Self {
        Self {
            n_obs: 500,
            dim_x: 20,
            theta: 0.0,
            r2_d: 0.5,
            r2_y: 0.5,
        }
    }
}

impl IrmData {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        check(self.dim_x >= 1, "dim_x must be positive")?;
        check(
            self.r2_d > 0.0 && self.r2_d < 1.0 && self.r2_y > 0.0 && self.r2_y < 1.0,
            "R2_d and R2_y must lie in (0, 1)",
        )?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let v = uniforms(&mut rng, n, 0.0, 1.0);
        let zeta = normals(&mut rng, n);
        let cov = toeplitz(0.5, self.dim_x);
        let x = mvn(&mut rng, n, &cov)?;
        let beta: Array1<f64> = (1..=self.dim_x).map(|k| 1.0 / (k * k) as f64).collect();
        let b_sigma_b = cov.dot(&beta).dot(&beta);
        let c_y = (self.r2_y / ((1.0 - self.r2_y) * b_sigma_b)).sqrt();
        let c_d = (std::f64::consts::PI.powi(2) / 3.0 * self.r2_d
            / ((1.0 - self.r2_d) * b_sigma_b))
            .sqrt();
        let xb = x.dot(&beta);
        let d: Array1<f64> = (0..n)
            .map(|i| f64::from(expit(c_d * xb[i]) > v[i]))
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| d[i] * self.theta + d[i] * c_y * xb[i] + zeta[i])
            .collect();
        Ok(Simulated::new(x, y, d))
    }
}

/// A binary treatment with an unobserved confounder `A`, as
/// `make_confounded_irm_data` (Sant'Anna and Zhao covariates), with oracle
/// confounding strengths for sensitivity analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct ConfoundedIrm {
    pub n_obs: usize,
    pub theta: f64,
    pub gamma_a: f64,
    pub beta_a: f64,
    pub linear: bool,
    pub dim_x: usize,
    pub trimming_threshold: f64,
    pub var_eps_y: f64,
}

impl Default for ConfoundedIrm {
    fn default() -> Self {
        Self {
            n_obs: 500,
            theta: 0.0,
            gamma_a: 0.127,
            beta_a: 0.58,
            linear: false,
            dim_x: 5,
            trimming_threshold: 0.01,
            var_eps_y: 1.0,
        }
    }
}

/// Oracle values of [`ConfoundedIrm`].
#[derive(Debug, Clone)]
pub struct ConfoundedIrmOracle {
    pub g_long: Array1<f64>,
    pub g_short: Array1<f64>,
    pub m_long: Array1<f64>,
    pub m_short: Array1<f64>,
    pub gamma_a: f64,
    pub beta_a: f64,
    pub a: Array1<f64>,
    pub y_0: Array1<f64>,
    pub y_1: Array1<f64>,
    pub z: Array2<f64>,
    pub cf_y: f64,
    pub cf_d_ate: f64,
    pub cf_d_atte: f64,
    pub rho_ate: f64,
    pub rho_atte: f64,
}

/// A data set of [`ConfoundedIrm`].
#[derive(Debug, Clone)]
pub struct ConfoundedIrmSimulated {
    pub data: Simulated,
    pub oracle: ConfoundedIrmOracle,
}

impl ConfoundedIrm {
    pub fn generate(&self, seed: u64) -> Result<ConfoundedIrmSimulated> {
        check(self.dim_x >= 5, "dim_x must be at least 5")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let xi = 0.75;
        let x = mvn(&mut rng, n, &toeplitz(0.0, self.dim_x))?;
        // Four transformed covariates and x₅.
        let mut z = sz_features(&x, None);
        z = ndarray::concatenate![Axis(1), z, standardize(&x.slice(s![.., 4..5]).to_owned())];
        let eps_y = normals(&mut rng, n) * self.var_eps_y.sqrt();
        let a = uniforms(&mut rng, n, -1.0, 1.0);
        let var_a = 4.0 / 12.0;
        let features = if self.linear { &x } else { &z };
        let f_reg = |i: usize| {
            let w = features.row(i);
            2.5 + 0.74 * w[0] + 0.25 * w[1] + 0.137 * (w[2] + w[3])
        };
        let f_ps = |i: usize| {
            let w = features.row(i);
            xi * (-w[0] + 0.1 * w[1] - 0.25 * w[2] - 0.1 * w[3])
        };
        let p: Array1<f64> = (0..n).map(|i| expit(f_ps(i))).collect();
        let mut m_long = &p + &(&a * self.gamma_a);
        let mut m_short = p;
        let t = self.trimming_threshold;
        if m_long.iter().any(|&m| m < t || m > 1.0 - t) {
            m_long.mapv_inplace(|m| m.clamp(t, 1.0 - t));
            m_short.mapv_inplace(|m| m.clamp(t, 1.0 - t));
        }
        let u = uniforms(&mut rng, n, 0.0, 1.0);
        let d: Array1<f64> = (0..n).map(|i| f64::from(m_long[i] >= u[i])).collect();
        let d1x = z.column(4).mapv(|v| v + 1.0);
        let var_dx = var(&(&d * &d1x));
        let cov_adx = self.gamma_a * var_a;
        let g_partial: Array1<f64> = (0..n).map(f_reg).collect();
        let g_short_d1 = &d1x * (self.theta + self.beta_a * cov_adx / var_dx) + &g_partial;
        let g_short = &d * &g_short_d1 + &(&d.mapv(|v| 1.0 - v) * &g_partial);
        let g_long_d0 = &g_partial + &(&a * self.beta_a);
        let g_long_d1 = &d1x * self.theta + &g_long_d0;
        let g_long = &d * &g_long_d1 + &(&d.mapv(|v| 1.0 - v) * &g_long_d0);
        let y_0 = &g_long_d0 + &eps_y;
        let y_1 = &g_long_d1 + &eps_y;
        let y = &d * &y_1 + &(&d.mapv(|v| 1.0 - v) * &y_0);
        let cf_y = (&g_long - &g_short).mapv(|v| v * v).mean().unwrap()
            / (&y - &g_short).mapv(|v| v * v).mean().unwrap();
        let mean_d = mean(&d);
        let rr = |m: &Array1<f64>| -> (Array1<f64>, Array1<f64>) {
            let ate = (0..n)
                .map(|i| d[i] / m[i] - (1.0 - d[i]) / (1.0 - m[i]))
                .collect();
            let atte = (0..n)
                .map(|i| d[i] / mean_d - (1.0 - d[i]) / mean_d * (m[i] / (1.0 - m[i])))
                .collect();
            (ate, atte)
        };
        let ((ate_long, atte_long), (ate_short, atte_short)) = (rr(&m_long), rr(&m_short));
        let inv_var = |m: &Array1<f64>| m.mapv(|v| 1.0 / (v * (1.0 - v))).mean().unwrap();
        let odds = |m: &Array1<f64>| m.mapv(|v| v / (1.0 - v)).mean().unwrap();
        let cf_d_ate = (inv_var(&m_long) - inv_var(&m_short)) / inv_var(&m_long);
        let cf_d_atte = (odds(&m_long) - odds(&m_short)) / odds(&m_long);
        let (rho_ate, rho_atte) = if self.beta_a == 0.0 || self.gamma_a == 0.0 {
            (0.0, 0.0)
        } else {
            let g_diff = &g_long - &g_short;
            (
                corr(&g_diff, &(&ate_long - &ate_short)),
                corr(&g_diff, &(&atte_long - &atte_short)),
            )
        };
        Ok(ConfoundedIrmSimulated {
            data: Simulated::new(x, y, d),
            oracle: ConfoundedIrmOracle {
                g_long,
                g_short,
                m_long,
                m_short,
                gamma_a: self.gamma_a,
                beta_a: self.beta_a,
                a,
                y_0,
                y_1,
                z,
                cf_y,
                cf_d_ate,
                cf_d_atte,
                rho_ate,
                rho_atte,
            },
        })
    }
}

/// A multi-valued treatment `d ∈ {0, 1, …, n_levels}`, as
/// `make_irm_data_discrete_treatments`: a latent continuous treatment cut at
/// its quantiles, with a share `1 / n_levels` of untreated units.
#[derive(Debug, Clone, PartialEq)]
pub struct IrmDiscreteTreatments {
    pub n_obs: usize,
    pub n_levels: usize,
    pub linear: bool,
    pub xi: f64,
    pub c: f64,
    pub dim_x: usize,
}

impl Default for IrmDiscreteTreatments {
    fn default() -> Self {
        Self {
            n_obs: 200,
            n_levels: 3,
            linear: false,
            xi: 0.3,
            c: 0.0,
            dim_x: 5,
        }
    }
}

/// Oracle values of [`IrmDiscreteTreatments`].
#[derive(Debug, Clone)]
pub struct DiscreteTreatmentOracle {
    pub cont_d: Array1<f64>,
    pub level_bounds: Array1<f64>,
    pub potential_level: Array1<f64>,
    pub ite: Array1<f64>,
    pub y0: Array1<f64>,
}

/// A data set of [`IrmDiscreteTreatments`].
#[derive(Debug, Clone)]
pub struct DiscreteTreatmentSimulated {
    pub data: Simulated,
    pub oracle: DiscreteTreatmentOracle,
}

impl IrmDiscreteTreatments {
    pub fn generate(&self, seed: u64) -> Result<DiscreteTreatmentSimulated> {
        check(self.n_levels >= 2, "n_levels must be at least 2")?;
        check(self.dim_x >= 4, "dim_x must be at least 4")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let x = mvn(&mut rng, n, &toeplitz(self.c, self.dim_x))?;
        let z = sz_features(&x, Some(4));
        let eps_y = normals(&mut rng, n) * 5.0_f64.sqrt();
        let eps_d = normals(&mut rng, n);
        let features = if self.linear { &x } else { &z };
        let g: Array1<f64> = (0..n)
            .map(|i| {
                let w = features.row(i);
                210.0 + 27.4 * w[0] + 13.7 * (w[1] + w[2] + w[3])
            })
            .collect();
        let cont_d: Array1<f64> = (0..n)
            .map(|i| {
                let w = features.row(i);
                self.xi * (-w[0] + 0.5 * w[1] - 0.25 * w[2] - 0.1 * w[3]) + eps_d[i]
            })
            .collect();
        let k = self.n_levels;
        let values = cont_d.to_vec();
        let level_bounds: Array1<f64> = (0..=k)
            .map(|j| np_quantile(&values, j as f64 / k as f64))
            .collect();
        let potential_level: Array1<f64> = cont_d.mapv(|c| {
            1.0 + level_bounds
                .slice(s![1..k])
                .iter()
                .filter(|&&b| c >= b)
                .count() as f64
        });
        let eta = uniforms(&mut rng, n, 0.0, 1.0);
        let d: Array1<f64> = (0..n)
            .map(|i| f64::from(eta[i] >= 1.0 / k as f64) * potential_level[i])
            .collect();
        let ite = cont_d.mapv(|c| 15.0 * expit(c + 1.2 * c.cos()) - 2.0);
        let y0 = &g + &eps_y;
        let y: Array1<f64> = (0..n)
            .map(|i| ite[i] * f64::from(d[i] > 0.0) + y0[i])
            .collect();
        Ok(DiscreteTreatmentSimulated {
            data: Simulated::new(x, y, d),
            oracle: DiscreteTreatmentOracle {
                cont_d,
                level_bounds,
                potential_level,
                ite,
                y0,
            },
        })
    }
}

/// A binary instrument `z` shifting a binary treatment, `d = 1{α_x z + v > 0}`,
/// `y = θd + x'β + u` with `corr(u, v) = 0.3`.
#[derive(Debug, Clone, PartialEq)]
pub struct IivmData {
    pub n_obs: usize,
    pub dim_x: usize,
    pub theta: f64,
    pub alpha_x: f64,
}

impl Default for IivmData {
    fn default() -> Self {
        Self {
            n_obs: 500,
            dim_x: 20,
            theta: 1.0,
            alpha_x: 0.2,
        }
    }
}

impl IivmData {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        check(self.dim_x >= 1, "dim_x must be positive")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let (u, v) = correlated_pair(&mut rng, n, 0.3);
        let x = mvn(&mut rng, n, &toeplitz(0.5, self.dim_x))?;
        let beta: Array1<f64> = (1..=self.dim_x).map(|k| 1.0 / (k * k) as f64).collect();
        let z: Array1<f64> = (0..n).map(|_| bernoulli(&mut rng, 0.5)).collect();
        let d: Array1<f64> = (0..n)
            .map(|i| f64::from(self.alpha_x * z[i] + v[i] > 0.0))
            .collect();
        let y = &d * self.theta + &x.dot(&beta) + &u;
        let mut sim = Simulated::new(x, y, d);
        sim.z = Some(z.insert_axis(Axis(1)));
        Ok(sim)
    }
}

/// Sample selection: the outcome is observed when `s = 1`, with
/// `s = 1{x'β + d + γz + e₀ > 0}`; missing at random (`mar`, errors
/// independent and `γ = 0`) or nonignorable (`corr(e₀, e₁) = 0.8`, `γ = 1`).
/// Unobserved outcomes are 0.
#[derive(Debug, Clone, PartialEq)]
pub struct SsmData {
    pub n_obs: usize,
    pub dim_x: usize,
    pub theta: f64,
    pub mar: bool,
}

impl Default for SsmData {
    fn default() -> Self {
        Self {
            n_obs: 8000,
            dim_x: 100,
            theta: 1.0,
            mar: true,
        }
    }
}

impl SsmData {
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        check(self.dim_x >= 1, "dim_x must be positive")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let (rho, gamma) = if self.mar { (0.0, 0.0) } else { (0.8, 1.0) };
        let (e0, e1) = correlated_pair(&mut rng, n, rho);
        let x = mvn(&mut rng, n, &toeplitz(0.5, self.dim_x))?;
        let beta: Array1<f64> = (1..=self.dim_x).map(|k| 0.4 / (k * k) as f64).collect();
        let xb = x.dot(&beta);
        let d: Array1<f64> = (0..n)
            .map(|i| f64::from(xb[i] + normal(&mut rng) > 0.0))
            .collect();
        let z = normals(&mut rng, n);
        let s: Array1<f64> = (0..n)
            .map(|i| f64::from(xb[i] + d[i] + gamma * z[i] + e0[i] > 0.0))
            .collect();
        let y: Array1<f64> = (0..n)
            .map(|i| {
                if s[i] == 1.0 {
                    xb[i] + self.theta * d[i] + e1[i]
                } else {
                    0.0
                }
            })
            .collect();
        let mut sim = Simulated::new(x, y, d);
        sim.z = Some(z.insert_axis(Axis(1)));
        sim.s = Some(s);
        Ok(sim)
    }
}

/// Heterogeneous treatment effects `τ(x) = exp(2x₀) + 3 sin(4x₀)` (or
/// `3 sin(4x₁)` with `n_x = 2`) on uniform covariates, as
/// `make_heterogeneous_data`.
#[derive(Debug, Clone, PartialEq)]
pub struct HeterogeneousData {
    pub n_obs: usize,
    /// Number of covariates.
    pub p: usize,
    /// Covariates with non-zero coefficients.
    pub support_size: usize,
    /// Covariates the effect depends on (1 or 2).
    pub n_x: usize,
    pub binary_treatment: bool,
}

impl Default for HeterogeneousData {
    fn default() -> Self {
        Self {
            n_obs: 200,
            p: 30,
            support_size: 5,
            n_x: 1,
            binary_treatment: false,
        }
    }
}

/// A data set of [`HeterogeneousData`] with the individual effects.
#[derive(Debug, Clone)]
pub struct HeterogeneousSimulated {
    pub data: Simulated,
    pub effects: Array1<f64>,
}

impl HeterogeneousData {
    /// The treatment effect at covariates `x`.
    pub fn treatment_effect(&self, x: ndarray::ArrayView2<'_, f64>) -> Array1<f64> {
        let k = if self.n_x == 2 { 1 } else { 0 };
        (0..x.nrows())
            .map(|i| (2.0 * x[[i, 0]]).exp() + 3.0 * (4.0 * x[[i, k]]).sin())
            .collect()
    }

    pub fn generate(&self, seed: u64) -> Result<HeterogeneousSimulated> {
        check(self.n_x == 1 || self.n_x == 2, "n_x must be 1 or 2")?;
        check(
            self.support_size <= self.p && self.p >= self.n_x,
            "support_size must not exceed p",
        )?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let support: Vec<usize> = sample(&mut rng, self.p, self.support_size).into_vec();
        let coefs_y = uniforms(&mut rng, self.support_size, 0.0, 1.0);
        let coefs_d = uniforms(&mut rng, self.support_size, 0.0, 0.3);
        let epsilon = uniforms(&mut rng, n, -1.0, 1.0);
        let eta = uniforms(&mut rng, n, -1.0, 1.0);
        let x = Array2::from_shape_fn((n, self.p), |_| rng.gen::<f64>());
        let te = self.treatment_effect(x.view());
        let xs = x.select(Axis(1), &support);
        let index_d = xs.dot(&coefs_d);
        let d: Array1<f64> = if self.binary_treatment {
            (0..n).map(|i| f64::from(index_d[i] >= eta[i])).collect()
        } else {
            &index_d + &eta
        };
        let y = &te * &d + &xs.dot(&coefs_y) + &epsilon;
        Ok(HeterogeneousSimulated {
            data: Simulated::new(x, y, d),
            effects: te,
        })
    }
}

// ---------------------------------------------------------------------------
// Difference-in-differences
// ---------------------------------------------------------------------------

/// Which covariates enter the propensity and outcome models of the DiD
/// designs (Sant'Anna and Zhao, 2020): `1` both transformed, `2` propensity
/// raw, `3` outcome raw, `4` both raw, `5`/`6` experimental (no
/// confounding) with transformed / raw outcome covariates.
fn did_features<'a>(
    dgp_type: u8,
    x: &'a Array2<f64>,
    z: &'a Array2<f64>,
) -> Result<(Option<&'a Array2<f64>>, &'a Array2<f64>)> {
    Ok(match dgp_type {
        1 => (Some(z), z),
        2 => (Some(x), z),
        3 => (Some(z), x),
        4 => (Some(x), x),
        5 => (None, z),
        6 => (None, x),
        _ => {
            return Err(DmlError::InvalidArgument(
                "dgp_type must be between 1 and 6".into(),
            ))
        }
    })
}

/// Sant'Anna and Zhao (2020): two periods, a binary treatment and the
/// transformed covariates `Z`. `generate` gives a panel (`y = Y₁ − Y₀`);
/// `generate_cross_sections` repeated cross-sections with a time indicator.
#[derive(Debug, Clone, PartialEq)]
pub struct DidSz2020 {
    pub n_obs: usize,
    pub dgp_type: u8,
    pub xi: f64,
    pub c: f64,
    /// Share of post-period observations (cross-sections).
    pub lambda_t: f64,
}

impl Default for DidSz2020 {
    fn default() -> Self {
        Self {
            n_obs: 500,
            dgp_type: 1,
            xi: 0.75,
            c: 0.0,
            lambda_t: 0.5,
        }
    }
}

/// `(z, d, y0, y1)`: covariates, treatment and the outcomes of both periods.
type TwoPeriods = (Array2<f64>, Array1<f64>, Array1<f64>, Array1<f64>);

impl DidSz2020 {
    /// `(z, d, y0, y1)` of the two periods.
    fn draw(&self, rng: &mut ChaCha8Rng) -> Result<TwoPeriods> {
        let n = self.n_obs;
        let x = mvn(rng, n, &toeplitz(self.c, 4))?;
        let z = sz_features(&x, None);
        let eps_0 = normals(rng, n);
        let eps_1a = normals(rng, n);
        let eps_1b = normals(rng, n);
        let (ps, reg) = did_features(self.dgp_type, &x, &z)?;
        let f_reg = |i: usize| {
            let w = reg.row(i);
            210.0 + 27.4 * w[0] + 13.7 * (w[1] + w[2] + w[3])
        };
        let p: Array1<f64> = (0..n)
            .map(|i| match ps {
                None => 0.5,
                Some(w) => {
                    let w = w.row(i);
                    expit(self.xi * (-w[0] + 0.5 * w[1] - 0.25 * w[2] - 0.1 * w[3]))
                }
            })
            .collect();
        let u = uniforms(rng, n, 0.0, 1.0);
        let d: Array1<f64> = (0..n).map(|i| f64::from(p[i] >= u[i])).collect();
        let fr: Array1<f64> = (0..n).map(f_reg).collect();
        let nu: Array1<f64> = (0..n).map(|i| d[i] * fr[i] + normal(rng)).collect();
        let y0 = &fr + &nu + &eps_0;
        let y1: Array1<f64> = (0..n)
            .map(|i| 2.0 * fr[i] + nu[i] + if d[i] == 1.0 { eps_1b[i] } else { eps_1a[i] })
            .collect();
        Ok((z, d, y0, y1))
    }

    /// A two-period panel: covariates `z`, treatment and the outcome change.
    pub fn generate(&self, seed: u64) -> Result<Simulated> {
        let (z, d, y0, y1) = self.draw(&mut rng(seed))?;
        Ok(Simulated::new(z, &y1 - &y0, d))
    }

    /// Repeated cross-sections: each unit is observed once, in the post
    /// period (`t = 1`) with probability `lambda_t`.
    pub fn generate_cross_sections(&self, seed: u64) -> Result<Simulated> {
        check(
            self.lambda_t > 0.0 && self.lambda_t < 1.0,
            "lambda_t must lie in (0, 1)",
        )?;
        let mut rng = rng(seed);
        let (z, d, y0, y1) = self.draw(&mut rng)?;
        let t: Array1<f64> = (0..self.n_obs)
            .map(|_| f64::from(rng.gen::<f64>() <= self.lambda_t))
            .collect();
        let y = &t * &y1 + &(&t.mapv(|v| 1.0 - v) * &y0);
        let mut sim = Simulated::new(z, y, d);
        sim.t = Some(t);
        Ok(sim)
    }
}

/// Callaway and Sant'Anna (2021): staggered adoption over `n_periods`
/// periods, with cohorts first treated from period `n_pre_treat_periods`
/// on, dynamic and heterogeneous effects, unit and time effects.
#[derive(Debug, Clone, PartialEq)]
pub struct DidCs2021 {
    pub n_obs: usize,
    pub dgp_type: u8,
    pub include_never_treated: bool,
    pub c: f64,
    pub xi: f64,
    pub n_periods: usize,
    pub anticipation_periods: usize,
    pub n_pre_treat_periods: usize,
}

impl Default for DidCs2021 {
    fn default() -> Self {
        Self {
            n_obs: 1000,
            dgp_type: 1,
            include_never_treated: true,
            c: 0.0,
            xi: 0.75,
            n_periods: 5,
            anticipation_periods: 0,
            n_pre_treat_periods: 2,
        }
    }
}

/// A simulated staggered-adoption panel in long format (periods
/// `0, 1, …`); `cohort` is the first treated period, `f64::INFINITY` for
/// never treated units.
#[derive(Debug, Clone)]
pub struct SimulatedStaggered {
    pub id: Array1<i64>,
    pub t: Array1<f64>,
    pub y: Array1<f64>,
    pub y0: Array1<f64>,
    pub y1: Array1<f64>,
    pub cohort: Array1<f64>,
    pub x: Array2<f64>,
}

impl SimulatedStaggered {
    fn select(&self, keep: &[bool]) -> Self {
        let rows: Vec<usize> = (0..keep.len()).filter(|&r| keep[r]).collect();
        let pick = |v: &Array1<f64>| rows.iter().map(|&r| v[r]).collect::<Array1<f64>>();
        Self {
            id: rows.iter().map(|&r| self.id[r]).collect(),
            t: pick(&self.t),
            y: pick(&self.y),
            y0: pick(&self.y0),
            y1: pick(&self.y1),
            cohort: pick(&self.cohort),
            x: self.x.select(Axis(0), &rows),
        }
    }

    /// The data as a [`PanelData`] for [`StaggeredDid`](crate::StaggeredDid).
    pub fn to_panel(&self) -> Result<PanelData> {
        PanelData::from_long(
            self.id.as_slice().expect("contiguous"),
            self.t.as_slice().expect("contiguous"),
            self.y.as_slice().expect("contiguous"),
            self.cohort.as_slice().expect("contiguous"),
            &self.x,
        )
    }

    /// The rows as independent observations ([`RepeatedCrossSections`]).
    pub fn to_cross_sections(&self) -> Result<RepeatedCrossSections> {
        RepeatedCrossSections::new(
            self.y.clone(),
            self.t.clone(),
            self.cohort.clone(),
            self.x.clone(),
        )
    }
}

impl DidCs2021 {
    /// The panel (`make_did_CS2021` with `time_type="float"`).
    pub fn generate(&self, seed: u64) -> Result<SimulatedStaggered> {
        self.panel(&mut rng(seed))
    }

    /// Repeated cross-sections (`make_did_cs_CS2021`): a panel of
    /// `⌈n_obs / lambda_t⌉` units of which each row is kept with
    /// probability `lambda_t`.
    pub fn generate_cross_sections(&self, seed: u64, lambda_t: f64) -> Result<SimulatedStaggered> {
        check(
            lambda_t > 0.0 && lambda_t <= 1.0,
            "lambda_t must lie in (0, 1]",
        )?;
        let mut rng = rng(seed);
        let panel = DidCs2021 {
            n_obs: (self.n_obs as f64 / lambda_t).ceil() as usize,
            ..self.clone()
        }
        .panel(&mut rng)?;
        let keep: Vec<bool> = (0..panel.y.len())
            .map(|_| rng.gen::<f64>() < lambda_t)
            .collect();
        Ok(panel.select(&keep))
    }

    fn panel(&self, rng: &mut ChaCha8Rng) -> Result<SimulatedStaggered> {
        check(
            self.n_pre_treat_periods >= 1,
            "n_pre_treat_periods must be at least 1",
        )?;
        let n_periods = self.n_periods + self.anticipation_periods;
        check(
            n_periods > self.n_pre_treat_periods,
            "need more periods than pre-treatment periods",
        )?;
        let n = self.n_obs;
        let x = mvn(rng, n, &toeplitz(self.c, 4))?;
        let z = sz_features(&x, None);
        let (ps, reg) = did_features(self.dgp_type, &x, &z)?;
        let periods: Vec<f64> = (0..n_periods).map(|t| t as f64).collect();
        let max_exposure = n_periods - self.n_pre_treat_periods;
        let mut cohorts: Vec<f64> = periods[self.n_pre_treat_periods..].to_vec();
        if self.include_never_treated {
            cohorts.push(f64::INFINITY);
        }
        let n_groups = cohorts.len();
        // Group of each unit.
        let group: Vec<usize> = (0..n)
            .map(|i| match ps {
                None => rng.gen_range(0..n_groups),
                Some(w) => {
                    let coef = [-1.0, 0.5, -0.25, -0.2];
                    let logits: Vec<f64> = (0..n_groups)
                        .map(|g| {
                            let scale = 1.0 - g as f64 / n_groups as f64;
                            let v: f64 = (0..4).map(|k| w[[i, k]] * coef[k] * scale).sum();
                            (self.xi * v).clamp(-2.5, 2.5)
                        })
                        .collect();
                    let e: Vec<f64> = logits.iter().map(|l| l.exp()).collect();
                    let total: f64 = e.iter().sum();
                    let clipped: Vec<f64> =
                        e.iter().map(|v| (v / total).clamp(0.05, 0.95)).collect();
                    let total: f64 = clipped.iter().sum();
                    let u = rng.gen::<f64>() * total;
                    let mut acc = 0.0;
                    let mut pick = n_groups - 1;
                    for (g, p) in clipped.iter().enumerate() {
                        acc += p;
                        if u < acc {
                            pick = g;
                            break;
                        }
                    }
                    pick
                }
            })
            .collect();
        let eta: Vec<f64> = group.iter().map(|&g| g as f64 + normal(rng)).collect();
        let rows = n * n_periods;
        let (mut y, mut y0, mut y1) = (
            Array1::zeros(rows),
            Array1::zeros(rows),
            Array1::zeros(rows),
        );
        let coef_reg = [27.4, 13.7, 13.7, 13.7];
        for i in 0..n {
            for t in 0..n_periods {
                let r = i * n_periods + t;
                let scale = (t + 1) as f64 / n_periods as f64;
                let f_reg = 210.0
                    + (0..4)
                        .map(|k| reg[[i, k]] * coef_reg[k] * scale)
                        .sum::<f64>();
                let exposure = if t < self.n_pre_treat_periods {
                    0.0
                } else {
                    let e = (t - self.n_pre_treat_periods) as f64 - group[i] as f64 + 1.0;
                    if group[i] >= max_exposure {
                        0.0
                    } else {
                        e.max(0.0)
                    }
                };
                let delta_e = exposure + 0.1 * x[[i, 0]] * exposure;
                let (e0, e1) = (normal(rng), normal(rng));
                y0[r] = f_reg + t as f64 + eta[i] + e0;
                y1[r] = y0[r] + delta_e + (e1 - e0);
                y[r] = if exposure > 0.0 { y1[r] } else { y0[r] };
            }
        }
        let panel = SimulatedStaggered {
            id: (0..rows).map(|r| (r / n_periods) as i64).collect(),
            t: (0..rows).map(|r| (r % n_periods) as f64).collect(),
            y,
            y0,
            y1,
            cohort: (0..rows).map(|r| cohorts[group[r / n_periods]]).collect(),
            x: Array2::from_shape_fn((rows, 4), |(r, k)| z[[r / n_periods, k]]),
        };
        if self.anticipation_periods == 0 {
            return Ok(panel);
        }
        // Drop the first periods and the cohorts treated too late, and shift
        // time back (DoubleML's anticipation handling).
        let a = self.anticipation_periods;
        let last_cohort = periods[n_periods - 1 - a];
        let keep: Vec<bool> = (0..rows)
            .map(|r| {
                panel.t[r] >= periods[a]
                    && (panel.cohort[r] <= last_cohort || panel.cohort[r].is_infinite())
            })
            .collect();
        let mut kept = panel.select(&keep);
        kept.t.mapv_inplace(|t| t - a as f64);
        Ok(kept)
    }
}

// ---------------------------------------------------------------------------
// Regression discontinuity
// ---------------------------------------------------------------------------

/// A simple RD design (`make_simple_rdd_data`): a normal score, uniform
/// covariates entering through polynomials of order `p`, effect `tau` at the
/// cutoff; sharp or fuzzy, with a continuous or binary outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct SimpleRdd {
    pub n_obs: usize,
    pub p: usize,
    pub fuzzy: bool,
    pub binary_outcome: bool,
    pub cutoff: f64,
    pub dim_x: usize,
    pub a: f64,
    pub tau: f64,
}

impl Default for SimpleRdd {
    fn default() -> Self {
        Self {
            n_obs: 5000,
            p: 4,
            fuzzy: true,
            binary_outcome: false,
            cutoff: 0.0,
            dim_x: 3,
            a: 0.0,
            tau: 1.0,
        }
    }
}

/// A data set of [`SimpleRdd`]: running variable, covariates, outcome,
/// treatment and both potential outcomes.
#[derive(Debug, Clone)]
pub struct SimulatedRdd {
    pub score: Array1<f64>,
    pub x: Array2<f64>,
    pub y: Array1<f64>,
    pub d: Array1<f64>,
    pub y0: Array1<f64>,
    pub y1: Array1<f64>,
}

impl SimulatedRdd {
    /// Outcome, treatment and covariates as [`DmlData`] (pass `score` to
    /// [`RdFlex::fit`](crate::RdFlex::fit)).
    pub fn to_data(&self) -> Result<DmlData> {
        DmlData::new(self.y.clone(), self.d.clone(), self.x.clone())
    }
}

impl SimpleRdd {
    pub fn generate(&self, seed: u64) -> Result<SimulatedRdd> {
        check(self.dim_x >= 1, "dim_x must be positive")?;
        let mut rng = rng(seed);
        let n = self.n_obs;
        let score = normals(&mut rng, n);
        let x: Array2<f64> = Array2::from_shape_fn((n, self.dim_x), |_| rng.gen_range(-1.0..1.0));
        // Σ_j Σ_k k·x_jᵏ (numpy's Polynomial(arange(p + 1))).
        let g_cov: Array1<f64> = (0..n)
            .map(|i| {
                (0..self.dim_x)
                    .map(|j| {
                        (1..=self.p)
                            .map(|k| k as f64 * x[[i, j]].powi(k as i32))
                            .sum::<f64>()
                    })
                    .sum()
            })
            .collect();
        let g0 = score.mapv(|s| 0.1 * s * s);
        let g1: Array1<f64> = (0..n)
            .map(|i| {
                let s = score[i];
                self.tau + 0.1 * s * s - 0.5 * s * s + self.a * x.row(i).sum() * s
            })
            .collect();
        let (y0, y1): (Array1<f64>, Array1<f64>) = if self.binary_outcome {
            (
                (0..n)
                    .map(|i| bernoulli(&mut rng, expit(g0[i] + g_cov[i])))
                    .collect(),
                (0..n)
                    .map(|i| bernoulli(&mut rng, expit(g1[i] + g_cov[i])))
                    .collect(),
            )
        } else {
            (
                (0..n)
                    .map(|i| g0[i] + g_cov[i] + 0.2 * normal(&mut rng))
                    .collect(),
                (0..n)
                    .map(|i| g1[i] + g_cov[i] + 0.2 * normal(&mut rng))
                    .collect(),
            )
        };
        let intended = score.mapv(|s| f64::from(s >= self.cutoff));
        let d: Array1<f64> = if self.fuzzy {
            (0..n)
                .map(|i| {
                    let (s, z) = (score[i], intended[i]);
                    let p = (0.3 + 0.4 * z + 0.01 * s * s - 0.02 * s * s * z + 0.2 * g_cov[i])
                        .clamp(0.0, 1.0);
                    bernoulli(&mut rng, p)
                })
                .collect()
        } else {
            intended
        };
        let y: Array1<f64> = (0..n)
            .map(|i| if d[i] == 1.0 { y1[i] } else { y0[i] })
            .collect();
        Ok(SimulatedRdd {
            score,
            x,
            y,
            d,
            y0,
            y1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvn_has_the_requested_covariance() {
        let cov = toeplitz(0.7, 3);
        let x = mvn(&mut rng(1), 40_000, &cov).unwrap();
        let emp = x.t().dot(&x) / x.nrows() as f64;
        for ((i, j), &c) in cov.indexed_iter() {
            assert!((emp[[i, j]] - c).abs() < 0.03, "cov[{i},{j}]");
        }
    }

    #[test]
    fn spd_matrix_is_positive_definite() {
        let m = spd_matrix(&mut rng(3), 6);
        assert!(cholesky_factor(&m).is_ok());
        assert!((&m - &m.t()).iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn generators_are_reproducible_and_validated() {
        let a = PlrCcddhnr2018::default().generate(7).unwrap();
        let b = PlrCcddhnr2018::default().generate(7).unwrap();
        assert_eq!(a.y, b.y);
        assert!(PlrCcddhnr2018 {
            dim_x: 2,
            ..Default::default()
        }
        .generate(0)
        .is_err());
        assert!(LplrLzz2020 {
            dim_x: 10,
            ..Default::default()
        }
        .generate(0)
        .is_err());
        assert!(DidSz2020 {
            dgp_type: 7,
            ..Default::default()
        }
        .generate(0)
        .is_err());
    }
}

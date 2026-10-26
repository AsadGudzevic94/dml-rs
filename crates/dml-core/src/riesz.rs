//! Linear functionals of the regression and their Riesz representers, for
//! automatic debiased machine learning ([`AutoDml`](crate::AutoDml); Chernozhukov,
//! Newey and Singh, 2022).
//!
//! The target is `θ = E[m(W; γ)]`, a mean of a linear functional `m` of the
//! regression `γ(w) = E[Y | W = w]` with `W = (D, X)`. Its Riesz representer
//! `α` satisfies `E[m(W; f)] = E[α(W) f(W)]` for all `f`, and the debiased
//! score is `m(W; γ) + α(W)(Y − γ(W)) − θ`. Instead of deriving `α` by hand
//! (as the propensity weights of the ATE), it is learned from the functional
//! alone by minimising the Riesz loss `E[α(W)² − 2 m(W; α)]`.
//!
//! A functional is written as evaluations of a function at transformed data
//! points: `m(W; f) = Σ_k c_k(W) · f(W_k)` ([`Term`]s), e.g. for the ATE
//! `f(1, X) − f(0, X)`.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::error::{DmlError, Result};

/// One evaluation in a functional: `weight ⊙ f(w)`, with `w` the data with
/// transformed columns (column 0 is the treatment).
#[derive(Debug, Clone)]
pub struct Term {
    pub w: Array2<f64>,
    pub weight: Array1<f64>,
}

impl Term {
    pub fn new(w: Array2<f64>, weight: Array1<f64>) -> Self {
        Self { w, weight }
    }

    /// `f` at the transformed points, with constant weight `c`.
    pub fn constant(w: Array2<f64>, c: f64) -> Self {
        let n = w.nrows();
        Self::new(w, Array1::from_elem(n, c))
    }

    /// The rows `idx` of this term.
    pub fn rows(&self, idx: &[usize]) -> Term {
        Term {
            w: self.w.select(Axis(0), idx),
            weight: idx.iter().map(|&i| self.weight[i]).collect(),
        }
    }
}

/// `m(W; f) = Σ_k weight_k ⊙ f(w_k)` for the terms of a functional.
pub fn apply_terms(
    terms: &[Term],
    mut f: impl FnMut(ArrayView2<'_, f64>) -> Result<Array1<f64>>,
) -> Result<Array1<f64>> {
    let n = terms.first().map_or(0, |t| t.w.nrows());
    let mut out = Array1::zeros(n);
    for t in terms {
        let v = f(t.w.view())?;
        if v.len() != n {
            return Err(DmlError::DimensionMismatch(
                "functional term predictions".into(),
            ));
        }
        out += &(&t.weight * &v);
    }
    Ok(out)
}

/// A linear functional `m(W; f)` of a function of `W = (D, X)`, given by the
/// transformed data points at which `f` is evaluated (column 0 of `w` is the
/// treatment).
pub trait LinearFunctional: Send + Sync {
    /// Label for summaries, e.g. `"ATE"`.
    fn name(&self) -> String;

    /// The terms of `m` at the rows of `w` (all observations; `w` has the
    /// treatment in column 0 and the covariates after it).
    fn terms(&self, w: ArrayView2<'_, f64>) -> Result<Vec<Term>>;
}

fn with_treatment(w: ArrayView2<'_, f64>, d: impl Fn(usize, f64) -> f64) -> Array2<f64> {
    let mut out = w.to_owned();
    for (i, v) in out.column_mut(0).iter_mut().enumerate() {
        *v = d(i, *v);
    }
    out
}

fn std_dev(v: ArrayView1<'_, f64>) -> f64 {
    let mean = v.mean().unwrap_or(0.0);
    (v.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0)).sqrt()
}

/// Average treatment effect of a binary treatment: `m = f(1, X) − f(0, X)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Ate;

impl LinearFunctional for Ate {
    fn name(&self) -> String {
        "ATE".into()
    }

    fn terms(&self, w: ArrayView2<'_, f64>) -> Result<Vec<Term>> {
        if w.column(0).iter().any(|&d| d != 0.0 && d != 1.0) {
            return Err(DmlError::InvalidArgument(
                "the ATE needs a 0/1 treatment".into(),
            ));
        }
        Ok(vec![
            Term::constant(with_treatment(w, |_, _| 1.0), 1.0),
            Term::constant(with_treatment(w, |_, _| 0.0), -1.0),
        ])
    }
}

/// Average derivative `E[∂γ(D, X)/∂D]` of a continuous treatment, by a central
/// difference with step `h` (default `10⁻⁴` times the standard deviation of
/// the treatment). Needs a regression that is smooth in the treatment (linear
/// models, polynomials, neural networks); for trees use [`ShiftEffect`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AverageDerivative {
    pub step: Option<f64>,
}

impl AverageDerivative {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step(mut self, h: f64) -> Self {
        self.step = Some(h);
        self
    }
}

impl LinearFunctional for AverageDerivative {
    fn name(&self) -> String {
        "average derivative".into()
    }

    fn terms(&self, w: ArrayView2<'_, f64>) -> Result<Vec<Term>> {
        let h = match self.step {
            Some(h) => h,
            None => 1e-4 * std_dev(w.column(0)),
        };
        if !(h.is_finite() && h > 0.0) {
            return Err(DmlError::InvalidArgument(
                "the derivative step must be positive (is the treatment constant?)".into(),
            ));
        }
        Ok(vec![
            Term::constant(with_treatment(w, |_, d| d + h), 0.5 / h),
            Term::constant(with_treatment(w, |_, d| d - h), -0.5 / h),
        ])
    }
}

/// Average effect of shifting the treatment by `delta`: `m = f(D + δ, X) −
/// f(D, X)`, divided by `δ` with [`per_unit`](Self::per_unit).
#[derive(Debug, Clone, Copy)]
pub struct ShiftEffect {
    pub delta: f64,
    pub per_unit: bool,
}

impl ShiftEffect {
    pub fn new(delta: f64) -> Self {
        Self {
            delta,
            per_unit: false,
        }
    }

    /// Report the effect per unit of treatment (divided by `δ`).
    pub fn per_unit(mut self, per_unit: bool) -> Self {
        self.per_unit = per_unit;
        self
    }
}

impl LinearFunctional for ShiftEffect {
    fn name(&self) -> String {
        format!("shift effect (δ = {})", self.delta)
    }

    fn terms(&self, w: ArrayView2<'_, f64>) -> Result<Vec<Term>> {
        if !(self.delta.is_finite() && self.delta != 0.0) {
            return Err(DmlError::InvalidArgument(
                "the shift must be non-zero".into(),
            ));
        }
        let c = if self.per_unit { 1.0 / self.delta } else { 1.0 };
        let delta = self.delta;
        Ok(vec![
            Term::constant(with_treatment(w, |_, d| d + delta), c),
            Term::constant(w.to_owned(), -c),
        ])
    }
}

/// Average effect of a policy that sets each unit's treatment to a given value:
/// `m = f(d_new, X) − f(D, X)`.
#[derive(Debug, Clone)]
pub struct PolicyEffect {
    pub treatment: Array1<f64>,
}

impl PolicyEffect {
    /// The treatment each unit would receive under the policy.
    pub fn new(treatment: Array1<f64>) -> Self {
        Self { treatment }
    }
}

impl LinearFunctional for PolicyEffect {
    fn name(&self) -> String {
        "policy effect".into()
    }

    fn terms(&self, w: ArrayView2<'_, f64>) -> Result<Vec<Term>> {
        if self.treatment.len() != w.nrows() {
            return Err(DmlError::DimensionMismatch(
                "the policy needs one treatment value per observation".into(),
            ));
        }
        Ok(vec![
            Term::constant(with_treatment(w, |i, _| self.treatment[i]), 1.0),
            Term::constant(w.to_owned(), -1.0),
        ])
    }
}

/// A functional given by a closure returning its terms.
#[derive(Clone)]
pub struct CustomFunctional {
    name: String,
    #[allow(clippy::type_complexity)]
    terms: Arc<dyn Fn(ArrayView2<'_, f64>) -> Result<Vec<Term>> + Send + Sync>,
}

impl CustomFunctional {
    pub fn new(
        name: impl Into<String>,
        terms: impl Fn(ArrayView2<'_, f64>) -> Result<Vec<Term>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            terms: Arc::new(terms),
        }
    }
}

impl LinearFunctional for CustomFunctional {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn terms(&self, w: ArrayView2<'_, f64>) -> Result<Vec<Term>> {
        let terms = (self.terms)(w)?;
        for t in &terms {
            if t.w.dim() != w.dim() || t.weight.len() != w.nrows() {
                return Err(DmlError::DimensionMismatch(format!(
                    "a functional term has shape {:?} and {} weights; expected {:?}",
                    t.w.dim(),
                    t.weight.len(),
                    w.dim()
                )));
            }
        }
        Ok(terms)
    }
}

// ---------------------------------------------------------------------------
// Riesz representers
// ---------------------------------------------------------------------------

/// A learner for the Riesz representer `α` of a functional.
pub trait RieszLearner: Send + Sync {
    /// Fit on the rows `w` (treatment in column 0) and the functional's terms
    /// at those rows.
    fn fit(&mut self, w: ArrayView2<'_, f64>, terms: &[Term]) -> Result<()>;

    /// `α` at the rows `w`.
    fn predict(&self, w: ArrayView2<'_, f64>) -> Result<Array1<f64>>;

    /// `m(W; α)` for the terms at some rows.
    fn predict_functional(&self, terms: &[Term]) -> Result<Array1<f64>> {
        apply_terms(terms, |w| self.predict(w))
    }

    /// An unfitted copy with the same settings.
    fn fresh(&self) -> Box<dyn RieszLearner>;
}

/// Basis functions `b(w)` in which the Riesz representer is approximated.
#[derive(Clone)]
pub enum Dictionary {
    /// `d^a · x^e` for treatment powers `a = 0, …, treatment_degree` and all
    /// monomials `x^e` of the covariates up to total degree `degree`
    /// (including the constant).
    Polynomial {
        degree: usize,
        treatment_degree: usize,
    },
    /// Any basis: a closure from `w` (treatment in column 0) to the matrix of
    /// basis functions.
    #[allow(clippy::type_complexity)]
    Custom(Arc<dyn Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>> + Send + Sync>),
}

impl std::fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dictionary::Polynomial {
                degree,
                treatment_degree,
            } => write!(
                f,
                "Polynomial {{ degree: {degree}, treatment_degree: {treatment_degree} }}"
            ),
            Dictionary::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Dictionary::Polynomial {
            degree: 2,
            treatment_degree: 1,
        }
    }
}

/// Exponent vectors of all monomials in `p` variables of total degree ≤ `degree`,
/// in graded order (constant first).
fn monomials(p: usize, degree: usize) -> Vec<Vec<usize>> {
    let mut out = vec![vec![0; p]];
    let mut last = vec![vec![0; p]];
    for _ in 0..degree {
        let mut next = Vec::new();
        for e in &last {
            // Increase variables at or after the last non-zero one (no duplicates).
            let start = e.iter().rposition(|&k| k > 0).unwrap_or(0);
            for j in start..p {
                let mut f = e.clone();
                f[j] += 1;
                next.push(f);
            }
        }
        out.extend(next.iter().cloned());
        last = next;
    }
    out
}

impl Dictionary {
    pub fn custom(
        basis: impl Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>> + Send + Sync + 'static,
    ) -> Self {
        Dictionary::Custom(Arc::new(basis))
    }

    /// The basis functions at the rows of `w`.
    pub fn apply(&self, w: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        match self {
            Dictionary::Polynomial {
                degree,
                treatment_degree,
            } => {
                let (n, cols) = w.dim();
                if cols == 0 {
                    return Err(DmlError::InvalidArgument("no treatment column".into()));
                }
                let monos = monomials(cols - 1, *degree);
                let k = monos.len() * (treatment_degree + 1);
                let mut b = Array2::zeros((n, k));
                for i in 0..n {
                    let row = w.row(i);
                    let mut c = 0;
                    for a in 0..=*treatment_degree {
                        let da = row[0].powi(a as i32);
                        for e in &monos {
                            let mut v = da;
                            for (j, &p) in e.iter().enumerate() {
                                if p > 0 {
                                    v *= row[j + 1].powi(p as i32);
                                }
                            }
                            b[[i, c]] = v;
                            c += 1;
                        }
                    }
                }
                Ok(b)
            }
            Dictionary::Custom(f) => {
                let b = f(w)?;
                if b.nrows() != w.nrows() {
                    return Err(DmlError::DimensionMismatch(
                        "the dictionary must return one row per observation".into(),
                    ));
                }
                Ok(b)
            }
        }
    }
}

/// How [`RieszLasso`] chooses its penalty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RieszPenalty {
    /// Cross-validation on the Riesz loss over a grid of penalties.
    CrossValidated,
    /// A fixed penalty `r` (0: unpenalised minimum distance).
    Fixed(f64),
}

/// The Lasso Riesz representer of Chernozhukov, Newey and Singh (2022):
/// `α(w) = b(w)'ρ` with
///
/// ```text
/// ρ = argmin  ρ'Ĝρ − 2M̂'ρ + 2r Σ_j ψ_j |ρ_j|,   Ĝ = E_n[b b'],  M̂ = E_n[m(W; b)],
/// ```
///
/// the sample Riesz loss with a weighted ℓ1 penalty (`ψ_j = √Ĝ_jj`, constant
/// basis functions unpenalised). The penalty is chosen by `cv`-fold
/// cross-validation of the Riesz loss over `n_alphas` values from the smallest
/// penalty that sets every coefficient to zero down to `eps` times it.
#[derive(Debug, Clone)]
pub struct RieszLasso {
    dictionary: Dictionary,
    penalty: RieszPenalty,
    n_alphas: usize,
    eps: f64,
    cv: usize,
    tol: f64,
    max_iter: usize,
    rho: Option<Array1<f64>>,
    chosen: Option<f64>,
}

impl Default for RieszLasso {
    fn default() -> Self {
        Self {
            dictionary: Dictionary::default(),
            penalty: RieszPenalty::CrossValidated,
            n_alphas: 30,
            eps: 1e-3,
            cv: 5,
            tol: 1e-10,
            max_iter: 100_000,
            rho: None,
            chosen: None,
        }
    }
}

/// Sample moments of the Riesz loss: `G = B'B / n`, `M = mean of m(W; b)`.
struct Moments {
    g: Array2<f64>,
    m: Array1<f64>,
}

impl Moments {
    fn new(b: ArrayView2<'_, f64>, mb: ArrayView2<'_, f64>) -> Self {
        let n = b.nrows().max(1) as f64;
        Self {
            g: b.t().dot(&b) / n,
            m: mb
                .mean_axis(Axis(0))
                .unwrap_or_else(|| Array1::zeros(b.ncols())),
        }
    }

    /// Riesz loss `ρ'Gρ − 2M'ρ` (the sample mean of `α² − 2m(W; α)`).
    fn loss(&self, rho: &Array1<f64>) -> f64 {
        rho.dot(&self.g.dot(rho)) - 2.0 * self.m.dot(rho)
    }
}

impl RieszLasso {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dictionary(mut self, dictionary: Dictionary) -> Self {
        self.dictionary = dictionary;
        self
    }

    pub fn penalty(mut self, penalty: RieszPenalty) -> Self {
        self.penalty = penalty;
        self
    }

    /// Number of penalties on the cross-validation grid (default 30).
    pub fn n_alphas(mut self, n_alphas: usize) -> Self {
        self.n_alphas = n_alphas;
        self
    }

    /// Smallest penalty on the grid relative to the largest (default `1e-3`).
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Cross-validation folds (default 5; observations are assigned in turn).
    pub fn cv(mut self, cv: usize) -> Self {
        self.cv = cv;
        self
    }

    /// Coefficients `ρ` after fitting.
    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.rho.as_ref()
    }

    /// The penalty used in the last fit.
    pub fn alpha(&self) -> Option<f64> {
        self.chosen
    }

    /// `B = b(w)` and `MB = Σ_k c_k ⊙ b(w_k)` for some rows.
    fn design(&self, w: ArrayView2<'_, f64>, terms: &[Term]) -> Result<(Array2<f64>, Array2<f64>)> {
        let b = self.dictionary.apply(w)?;
        let mut mb = Array2::zeros(b.dim());
        for t in terms {
            let bk = self.dictionary.apply(t.w.view())?;
            if bk.dim() != b.dim() {
                return Err(DmlError::DimensionMismatch(
                    "dictionary at transformed points".into(),
                ));
            }
            mb += &(&bk * &t.weight.view().insert_axis(Axis(1)));
        }
        Ok((b, mb))
    }

    /// Penalty loadings `√G_jj`, zero for constant basis functions.
    fn loadings(b: ArrayView2<'_, f64>, g: &Array2<f64>) -> Array1<f64> {
        (0..b.ncols())
            .map(|j| {
                let col = b.column(j);
                let first = col.first().copied().unwrap_or(0.0);
                if col.iter().all(|&v| v == first) {
                    0.0
                } else {
                    g[[j, j]].sqrt()
                }
            })
            .collect()
    }

    /// Coordinate descent for `ρ'Gρ − 2M'ρ + 2r Σ ψ_j|ρ_j|` from `rho`.
    fn solve(&self, mom: &Moments, psi: &Array1<f64>, r: f64, rho: &mut Array1<f64>) -> Result<()> {
        let p = rho.len();
        let mut grho = mom.g.dot(rho);
        let scale = mom
            .g
            .diag()
            .iter()
            .fold(0.0_f64, |a, &v| a.max(v))
            .max(1e-300);
        for _ in 0..self.max_iter {
            let mut max_step = 0.0_f64;
            for j in 0..p {
                let gjj = mom.g[[j, j]];
                if gjj <= 0.0 {
                    continue;
                }
                let z = mom.m[j] - (grho[j] - gjj * rho[j]);
                // Unpenalised coordinates (ψ_j = 0) have no threshold, also for r = ∞.
                let thr = if psi[j] > 0.0 { r * psi[j] } else { 0.0 };
                let new = if z > thr {
                    (z - thr) / gjj
                } else if z < -thr {
                    (z + thr) / gjj
                } else {
                    0.0
                };
                let delta = new - rho[j];
                if delta != 0.0 {
                    grho.scaled_add(delta, &mom.g.column(j));
                    rho[j] = new;
                    max_step = max_step.max(delta.abs() * gjj.sqrt());
                }
            }
            if max_step <= self.tol * scale.sqrt().max(1.0) {
                return Ok(());
            }
        }
        Err(DmlError::InvalidArgument(
            "the Riesz Lasso did not converge; increase the penalty or simplify the dictionary"
                .into(),
        ))
    }

    /// Unpenalised coefficients alone (penalty above `r_max`), and `r_max`.
    fn null_fit(&self, mom: &Moments, psi: &Array1<f64>) -> Result<(Array1<f64>, f64)> {
        let mut rho = Array1::zeros(psi.len());
        self.solve(mom, psi, f64::INFINITY, &mut rho)?;
        let grad = &mom.m - &mom.g.dot(&rho);
        let r_max = (0..psi.len())
            .filter(|&j| psi[j] > 0.0)
            .map(|j| grad[j].abs() / psi[j])
            .fold(0.0_f64, f64::max);
        Ok((rho, r_max))
    }

    fn grid(&self, r_max: f64) -> Vec<f64> {
        if r_max <= 0.0 {
            return vec![0.0];
        }
        let k = self.n_alphas.max(1);
        (0..k)
            .map(|i| {
                let t = if k == 1 {
                    0.0
                } else {
                    i as f64 / (k - 1) as f64
                };
                r_max * self.eps.powf(t)
            })
            .collect()
    }
}

impl RieszLearner for RieszLasso {
    fn fit(&mut self, w: ArrayView2<'_, f64>, terms: &[Term]) -> Result<()> {
        let (b, mb) = self.design(w, terms)?;
        let n = b.nrows();
        let mom = Moments::new(b.view(), mb.view());
        let psi = Self::loadings(b.view(), &mom.g);
        let r = match self.penalty {
            RieszPenalty::Fixed(r) => {
                if r.is_nan() || r < 0.0 {
                    return Err(DmlError::InvalidArgument(
                        "the penalty must be non-negative".into(),
                    ));
                }
                r
            }
            RieszPenalty::CrossValidated => {
                let (_, r_max) = self.null_fit(&mom, &psi)?;
                let grid = self.grid(r_max);
                let k = self.cv.max(2).min(n);
                let mut cv_loss = vec![0.0; grid.len()];
                for fold in 0..k {
                    let train: Vec<usize> = (0..n).filter(|i| i % k != fold).collect();
                    let test: Vec<usize> = (0..n).filter(|i| i % k == fold).collect();
                    let tr = Moments::new(
                        b.select(Axis(0), &train).view(),
                        mb.select(Axis(0), &train).view(),
                    );
                    let te = Moments::new(
                        b.select(Axis(0), &test).view(),
                        mb.select(Axis(0), &test).view(),
                    );
                    let psi_tr = Self::loadings(b.select(Axis(0), &train).view(), &tr.g);
                    let mut rho = Array1::zeros(psi.len());
                    for (loss, &r) in cv_loss.iter_mut().zip(&grid) {
                        self.solve(&tr, &psi_tr, r, &mut rho)?;
                        *loss += te.loss(&rho) * test.len() as f64 / n as f64;
                    }
                }
                let best = cv_loss
                    .iter()
                    .enumerate()
                    .fold(
                        (0, f64::INFINITY),
                        |acc, (i, &l)| if l < acc.1 { (i, l) } else { acc },
                    )
                    .0;
                grid[best]
            }
        };
        // Warm path down to the chosen penalty.
        let (mut rho, r_max) = self.null_fit(&mom, &psi)?;
        for &rr in self.grid(r_max).iter().filter(|&&g| g > r) {
            self.solve(&mom, &psi, rr, &mut rho)?;
        }
        self.solve(&mom, &psi, r, &mut rho)?;
        self.rho = Some(rho);
        self.chosen = Some(r);
        Ok(())
    }

    fn predict(&self, w: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let rho = self
            .rho
            .as_ref()
            .ok_or_else(|| DmlError::InvalidArgument("RieszLasso is not fitted".into()))?;
        let b = self.dictionary.apply(w)?;
        if b.ncols() != rho.len() {
            return Err(DmlError::DimensionMismatch(
                "dictionary size changed".into(),
            ));
        }
        Ok(b.dot(rho))
    }

    fn fresh(&self) -> Box<dyn RieszLearner> {
        Box::new(Self {
            rho: None,
            chosen: None,
            ..self.clone()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn monomials_count_and_order() {
        let m = monomials(3, 2);
        assert_eq!(m.len(), 10); // C(3 + 2, 2)
        assert_eq!(m[0], vec![0, 0, 0]);
        assert_eq!(m[1], vec![1, 0, 0]);
        assert!(m.contains(&vec![0, 1, 1]));
        let d = Dictionary::Polynomial {
            degree: 1,
            treatment_degree: 1,
        };
        let b = d.apply(array![[2.0, 3.0, 5.0]].view()).unwrap();
        assert_eq!(b, array![[1.0, 3.0, 5.0, 2.0, 6.0, 10.0]]);
    }

    #[test]
    fn unpenalised_fit_solves_the_normal_equations() {
        // r = 0: ρ = G⁻¹ M.
        let n = 200;
        let w = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                ((i * 7) % 5) as f64 / 2.0
            } else {
                ((i * 3) % 11) as f64 / 5.0
            }
        });
        let functional = ShiftEffect::new(0.5);
        let terms = functional.terms(w.view()).unwrap();
        let mut lasso = RieszLasso::new()
            .dictionary(Dictionary::Polynomial {
                degree: 1,
                treatment_degree: 1,
            })
            .penalty(RieszPenalty::Fixed(0.0));
        lasso.fit(w.view(), &terms).unwrap();
        let (b, mb) = lasso.design(w.view(), &terms).unwrap();
        let mom = Moments::new(b.view(), mb.view());
        let rho = lasso.coef().unwrap();
        let resid = mom.g.dot(rho) - &mom.m;
        assert!(resid.iter().all(|v| v.abs() < 1e-8), "{resid}");
    }

    #[test]
    fn saturated_ate_representer_is_the_inverse_propensity_weight() {
        // Discrete X with cell dummies interacted with D: the unpenalised Riesz
        // representer is D / π(x) − (1 − D) / (1 − π(x)) with cell shares π.
        let n = 300;
        let w = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                f64::from(((i * 7 + i / 3) % 5) < 2 + (i % 3) % 2)
            } else {
                (i % 3) as f64
            }
        });
        let dict = Dictionary::custom(|w: ArrayView2<'_, f64>| {
            Ok(Array2::from_shape_fn((w.nrows(), 6), |(i, k)| {
                let (d, x) = (w[[i, 0]], w[[i, 1]]);
                let arm = if k < 3 { d } else { 1.0 - d };
                if x == (k % 3) as f64 {
                    arm
                } else {
                    0.0
                }
            }))
        });
        let terms = Ate.terms(w.view()).unwrap();
        let mut lasso = RieszLasso::new()
            .dictionary(dict)
            .penalty(RieszPenalty::Fixed(0.0));
        lasso.fit(w.view(), &terms).unwrap();
        let alpha = lasso.predict(w.view()).unwrap();
        for c in 0..3 {
            let rows: Vec<usize> = (0..n).filter(|&i| w[[i, 1]] == c as f64).collect();
            let pi = rows.iter().map(|&i| w[[i, 0]]).sum::<f64>() / rows.len() as f64;
            for &i in &rows {
                let d = w[[i, 0]];
                let expected = d / pi - (1.0 - d) / (1.0 - pi);
                assert!(
                    (alpha[i] - expected).abs() < 1e-8,
                    "{} vs {expected}",
                    alpha[i]
                );
            }
        }
    }

    #[test]
    fn cross_validated_penalty_is_on_the_grid_and_predicts_m() {
        let n = 400;
        let w = Array2::from_shape_fn((n, 3), |(i, j)| {
            (((i + 1) * (j + 3) * 7919) % 1000) as f64 / 500.0 - 1.0
        });
        let terms = AverageDerivative::new().terms(w.view()).unwrap();
        let mut lasso = RieszLasso::new();
        lasso.fit(w.view(), &terms).unwrap();
        assert!(lasso.alpha().unwrap() >= 0.0);
        let m_alpha = lasso.predict_functional(&terms).unwrap();
        assert_eq!(m_alpha.len(), n);
        assert!(m_alpha.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn functional_terms() {
        let w = array![[0.0, 1.0], [1.0, 2.0]];
        let f = |w: ArrayView2<'_, f64>| Ok(w.column(0).mapv(|d| 3.0 * d) + w.column(1));
        assert_eq!(
            apply_terms(&Ate.terms(w.view()).unwrap(), f).unwrap(),
            array![3.0, 3.0]
        );
        let shift = ShiftEffect::new(2.0).per_unit(true);
        assert_eq!(
            apply_terms(&shift.terms(w.view()).unwrap(), f).unwrap(),
            array![3.0, 3.0]
        );
        let deriv = AverageDerivative::new().step(0.1);
        let v = apply_terms(&deriv.terms(w.view()).unwrap(), f).unwrap();
        assert!((v[0] - 3.0).abs() < 1e-12);
        let policy = PolicyEffect::new(array![1.0, 1.0]);
        assert_eq!(
            apply_terms(&policy.terms(w.view()).unwrap(), f).unwrap(),
            array![3.0, 0.0]
        );
        assert!(Ate.terms(array![[0.5, 1.0]].view()).is_err());
    }
}

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ShapeBuilder};

use super::{center, Centered, Regressor};
use crate::error::{DmlError, Result};

/// Lasso regression with an unpenalised intercept, fitted by cyclic
/// coordinate descent.
///
/// Minimises `1/(2n)·‖y − b − Xw‖² + α‖w‖₁`, the same objective as
/// scikit-learn's `Lasso(alpha=α)`. The classic DML learner for
/// high-dimensional confounders.
#[derive(Debug, Clone)]
pub struct Lasso {
    alpha: f64,
    tol: f64,
    max_iter: usize,
    coef: Option<Array1<f64>>,
    intercept: f64,
    n_iter: usize,
}

impl Lasso {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.max(0.0),
            tol: 1e-4,
            max_iter: 100_000,
            coef: None,
            intercept: 0.0,
            n_iter: 0,
        }
    }

    /// Stopping rule of scikit-learn's `Lasso(tol=tol)` (default `1e-4`): once
    /// no coefficient moves by more than `tol · max|w|` in a sweep, stop if the
    /// duality gap, which bounds the distance to the optimal objective, is at
    /// most `tol · ‖y − ȳ‖² / n`. With `α = 0` only the first condition applies.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn coef(&self) -> Option<&Array1<f64>> {
        self.coef.as_ref()
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Coordinate-descent sweeps of the last fit (scikit-learn's `n_iter_`;
    /// 0 if the starting point already met the tolerance).
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// A duality gap with the quantities screening reuses.
struct Gap {
    gap: f64,
    xt_r: Array1<f64>,
    dual_norm: f64,
}

/// Centred design in column-major order (each coordinate update reads a
/// contiguous column) with the column norms, shared by all fits on a path.
struct CdDesign {
    x_mean: Array1<f64>,
    y_mean: f64,
    xc: Array2<f64>,
    yc: Array1<f64>,
    col_sq: Vec<f64>,
    y_norm2: f64,
    /// `xcᵀxc` when `n > p` (as scikit-learn's `precompute="auto"`): each
    /// coordinate update then costs `O(p)` instead of `O(n)`.
    gram: Option<Array2<f64>>,
    /// `xcᵀyc`, kept with the Gram matrix for the duality gap.
    xty: Array1<f64>,
}

impl CdDesign {
    fn new(
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<Self> {
        let Centered {
            x_mean,
            y_mean,
            xc: centered,
            yc,
        } = center(x, y, w, true)?;
        let mut xc = Array2::<f64>::zeros(centered.dim().f());
        xc.assign(&centered);
        let col_sq = xc.columns().into_iter().map(|c| c.dot(&c)).collect();
        let (n, p) = xc.dim();
        let gram = (n > p).then(|| xc.t().dot(&xc));
        let xty = xc.t().dot(&yc);
        Ok(Self {
            x_mean,
            y_mean,
            y_norm2: yc.dot(&yc),
            xc,
            yc,
            col_sq,
            gram,
            xty,
        })
    }

    /// Solver state at `w = 0`: the residual `yc − xc·w`, or with the Gram
    /// matrix the correlations `xcᵀ(yc − xc·w)`.
    fn initial_state(&self) -> Array1<f64> {
        match self.gram {
            Some(_) => self.xty.clone(),
            None => self.yc.clone(),
        }
    }

    /// Duality gap of `½‖r‖² + nα‖w‖₁` at `w`, with `xcᵀr` and its largest
    /// absolute entry (scikit-learn's `gap_enet`): the dual point is the
    /// residual rescaled to be feasible. Without a penalty there is no useful
    /// gap, and `‖xcᵀr‖²` (the squared gradient) takes its place.
    fn duality_gap(&self, n_alpha: f64, w: &Array1<f64>, state: &Array1<f64>) -> Gap {
        let (xt_r, r_norm2, r_y) = match self.gram {
            Some(_) => {
                let w_xty = w.dot(&self.xty);
                // ‖r‖² = ‖y‖² − 2wᵀxᵀy + wᵀGw, with Gw = xᵀy − xᵀr.
                let w_gw = w_xty - w.dot(state);
                (
                    state.clone(),
                    self.y_norm2 - 2.0 * w_xty + w_gw,
                    self.y_norm2 - w_xty,
                )
            }
            None => (
                self.xc.t().dot(state),
                state.dot(state),
                state.dot(&self.yc),
            ),
        };
        if n_alpha == 0.0 {
            return Gap {
                gap: xt_r.dot(&xt_r),
                xt_r,
                dual_norm: 0.0,
            };
        }
        let dual_norm = xt_r.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let scale = if dual_norm > n_alpha {
            n_alpha / dual_norm
        } else {
            1.0
        };
        let primal = 0.5 * r_norm2 + n_alpha * w.iter().map(|v| v.abs()).sum::<f64>();
        let dual = -0.5 * scale * scale * r_norm2 + scale * r_y;
        Gap {
            gap: primal - dual,
            xt_r,
            dual_norm,
        }
    }

    /// Gap safe screening (Fercoq, Gramfort & Salmon, 2015), as scikit-learn:
    /// coordinate `j` is provably zero at the optimum when
    /// `(1 − |x_jᵀθ|) / ‖x_j‖ > √(2·gap) / (nα)` for the dual point `θ`; it is
    /// set to zero and left out of further sweeps.
    fn screen(
        &self,
        n_alpha: f64,
        g: &Gap,
        active: &mut [bool],
        w: &mut Array1<f64>,
        state: &mut Array1<f64>,
    ) {
        let radius = (2.0 * g.gap.abs()).sqrt() / n_alpha;
        for j in 0..active.len() {
            if !active[j] {
                continue;
            }
            let xj_theta = g.xt_r[j] / n_alpha.max(g.dual_norm);
            let d_j = (1.0 - xj_theta.abs()) / self.col_sq[j].sqrt();
            if self.col_sq[j] == 0.0 || d_j > radius {
                active[j] = false;
                if w[j] != 0.0 {
                    self.update(j, w[j], state);
                    w[j] = 0.0;
                }
            }
        }
    }

    /// Keep `state` in step with `w[j]` decreasing by `delta`.
    fn update(&self, j: usize, delta: f64, state: &mut Array1<f64>) {
        match &self.gram {
            // The Gram matrix is symmetric: row j is column j, contiguous.
            Some(g) => state.scaled_add(delta, &g.row(j)),
            None => state.scaled_add(delta, &self.xc.column(j)),
        }
    }

    /// Cyclic coordinate descent for penalty `alpha`, warm-started from `w`;
    /// `state` (see [`initial_state`](Self::initial_state)) must match `w` on
    /// entry and is kept up to date. Follows scikit-learn's `cd_fast`: stop at
    /// once if the warm start is already within tolerance, screen coordinates
    /// after every gap evaluation. Returns the number of sweeps.
    fn solve(
        &self,
        alpha: f64,
        w: &mut Array1<f64>,
        state: &mut Array1<f64>,
        tol: f64,
        max_iter: usize,
    ) -> Result<usize> {
        let p = self.xc.ncols();
        let n_alpha = self.xc.nrows() as f64 * alpha;
        let tol_gap = tol * self.y_norm2;
        let screening = n_alpha > 0.0;
        let g = self.duality_gap(n_alpha, w, state);
        if g.gap <= tol_gap && (self.gram.is_none() || g.gap >= 0.0) {
            return Ok(0);
        }
        let mut active = vec![true; p];
        if screening {
            self.screen(n_alpha, &g, &mut active, w, state);
        }
        for iter in 0..max_iter {
            let mut max_change = 0.0_f64;
            let mut max_w = 0.0_f64;
            for j in 0..p {
                if !active[j] || self.col_sq[j] == 0.0 {
                    continue;
                }
                let old = w[j];
                let corr = match self.gram {
                    Some(_) => state[j],
                    None => self.xc.column(j).dot(&*state),
                };
                let new = soft_threshold(corr + old * self.col_sq[j], n_alpha) / self.col_sq[j];
                if new != old {
                    self.update(j, old - new, state);
                    w[j] = new;
                }
                max_change = max_change.max((new - old).abs());
                max_w = max_w.max(new.abs());
            }
            if max_w == 0.0 || max_change <= tol * max_w || iter + 1 == max_iter {
                let g = self.duality_gap(n_alpha, w, state);
                if g.gap <= tol_gap {
                    return Ok(iter + 1);
                }
                if screening {
                    self.screen(n_alpha, &g, &mut active, w, state);
                }
            }
        }
        Err(DmlError::Convergence(format!(
            "Lasso did not converge in {max_iter} sweeps"
        )))
    }

    fn intercept(&self, w: &Array1<f64>) -> f64 {
        self.y_mean - self.x_mean.dot(w)
    }
}

/// Lasso solutions `(coef, intercept)` along `alphas` (best in decreasing
/// order), each fit warm-started from the previous one.
pub(super) fn lasso_path(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    w: Option<ArrayView1<'_, f64>>,
    alphas: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<(Array1<f64>, f64)>> {
    let design = CdDesign::new(x, y, w)?;
    let mut coef = Array1::<f64>::zeros(x.ncols());
    let mut state = design.initial_state();
    alphas
        .iter()
        .map(|&alpha| {
            design.solve(alpha, &mut coef, &mut state, tol, max_iter)?;
            Ok((coef.clone(), design.intercept(&coef)))
        })
        .collect()
}

impl Lasso {
    fn fit_impl(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: Option<ArrayView1<'_, f64>>,
    ) -> Result<()> {
        let design = CdDesign::new(x, y, w)?;
        let mut coef = Array1::<f64>::zeros(x.ncols());
        let mut state = design.initial_state();
        self.n_iter = design.solve(self.alpha, &mut coef, &mut state, self.tol, self.max_iter)?;
        self.intercept = design.intercept(&coef);
        self.coef = Some(coef);
        Ok(())
    }
}

impl Regressor for Lasso {
    fn fit(&mut self, x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Result<()> {
        self.fit_impl(x, y, None)
    }

    /// Minimises `1/(2n)·Σ wᵢ rᵢ² + α‖β‖₁` with weights rescaled to sum to
    /// `n`, as scikit-learn's `Lasso.fit(sample_weight=w)`.
    fn fit_weighted(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        w: ArrayView1<'_, f64>,
    ) -> Result<()> {
        self.fit_impl(x, y, Some(w))
    }

    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let coef = self.coef.as_ref().ok_or(DmlError::NotFitted)?;
        if x.ncols() != coef.len() {
            return Err(DmlError::DimensionMismatch(
                "Lasso::predict feature count".into(),
            ));
        }
        Ok(x.dot(coef) + self.intercept)
    }

    fn fresh(&self) -> Box<dyn Regressor> {
        Box::new(Self::new(self.alpha).tol(self.tol).max_iter(self.max_iter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::Ols;
    use ndarray::array;

    fn data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_fn((40, 3), |(i, j)| {
            (((i + 1) * (j + 2) * 7) % 13) as f64 - 6.0
        });
        let y = x
            .rows()
            .into_iter()
            .map(|r| 2.0 + 1.5 * r[0] - 0.2 * r[2])
            .collect();
        (x, y)
    }

    #[test]
    fn zero_alpha_equals_ols() {
        let (x, y) = data();
        let mut lasso = Lasso::new(0.0).tol(1e-12);
        lasso.fit(x.view(), y.view()).unwrap();
        let mut ols = Ols::new();
        ols.fit(x.view(), y.view()).unwrap();
        let diff = lasso.coef().unwrap() - ols.coef().unwrap();
        assert!(diff.iter().all(|v| v.abs() < 1e-8));
    }

    #[test]
    fn satisfies_kkt_conditions() {
        let (x, y) = data();
        let alpha = 0.5;
        let mut m = Lasso::new(alpha).tol(1e-12);
        m.fit(x.view(), y.view()).unwrap();
        let resid = &y - &m.predict(x.view()).unwrap();
        let n = x.nrows() as f64;
        for (j, &wj) in m.coef().unwrap().iter().enumerate() {
            let g = x.column(j).dot(&resid) / n;
            if wj != 0.0 {
                assert!((g - alpha * wj.signum()).abs() < 1e-8, "active {j}: {g}");
            } else {
                assert!(g.abs() <= alpha + 1e-8, "inactive {j}: {g}");
            }
        }
    }

    #[test]
    fn large_alpha_gives_intercept_only() {
        let (x, y) = data();
        let mut m = Lasso::new(1e6);
        m.fit(x.view(), y.view()).unwrap();
        assert!(m.coef().unwrap().iter().all(|&v| v == 0.0));
        assert!((m.intercept() - y.mean().unwrap()).abs() < 1e-12);
        assert_eq!(
            m.predict(array![[0.0, 0.0, 0.0]].view()).unwrap()[0],
            m.intercept()
        );
    }

    #[test]
    fn warm_started_path_equals_separate_fits() {
        let x = Array2::from_shape_fn((50, 4), |(i, j)| {
            (((i * 7 + j * 3) % 11) as f64 - 5.0) / 3.0
        });
        let y: Array1<f64> = x
            .rows()
            .into_iter()
            .map(|r| 1.0 + r[0] - 0.5 * r[3])
            .collect();
        let alphas = [1.0, 0.3, 0.1, 0.01];
        let path = lasso_path(x.view(), y.view(), None, &alphas, 1e-13, 100_000).unwrap();
        for (&alpha, (coef, intercept)) in alphas.iter().zip(&path) {
            let mut m = Lasso::new(alpha).tol(1e-13);
            m.fit(x.view(), y.view()).unwrap();
            let diff = (m.coef().unwrap() - coef).mapv(f64::abs).sum();
            assert!(
                diff < 1e-9 && (m.intercept() - intercept).abs() < 1e-9,
                "alpha {alpha}"
            );
        }
    }
}

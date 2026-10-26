//! Regression discontinuity designs with flexible covariate adjustment, as
//! DoubleML's `RDFlex` (Noack, Olma and Rothe, 2024).
//!
//! The outcome (and, in a fuzzy design, the treatment) is adjusted by a
//! cross-fitted prediction from the covariates `X` evaluated at the cutoff,
//! `η(X) = (μ(X; left) + μ(X; right)) / 2`, learned with kernel weights
//! around the cutoff. A local polynomial RD estimator ([`rd_estimate`], with
//! `rdrobust`'s options in [`RdOptions`]) is then run on the adjusted
//! outcome. The kernel bandwidth of the first stage is updated from the RD
//! bandwidth over `n_iterations` rounds. A binary outcome can be learned with
//! a classifier ([`RdFlex::sharp_binary`], [`RdFlex::fuzzy_binary`]).

use std::fmt;

use ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::crossfit::{Fold, SampleSplitting};
use crate::data::{gather, DmlData};
use crate::error::{DmlError, Result};
use crate::learner::{Classifier, Regressor};
use crate::rd::{rd_estimate, rdbwselect, RdBandwidths, RdData, RdEstimate, RdOptions};
use crate::stats::{median, norm_cdf, norm_ppf};

/// Regressors of the first stage besides the covariates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FirstStage {
    /// The cutoff indicator `1{score ≥ cutoff}`.
    #[default]
    Cutoff,
    /// The cutoff indicator and the score.
    CutoffAndScore,
    /// The cutoff indicator, the score and their interaction.
    InteractedCutoffAndScore,
}

/// Kernel of the first-stage weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RdKernel {
    Uniform,
    #[default]
    Triangular,
    Epanechnikov,
}

impl RdKernel {
    /// DoubleML's first-stage kernel weight at distance `x` from the cutoff.
    fn weight(self, x: f64, h: f64) -> f64 {
        match self {
            RdKernel::Uniform => f64::from(x.abs() <= h),
            RdKernel::Triangular => 0.0_f64.max((h - x.abs()) / h),
            RdKernel::Epanechnikov => {
                if x.abs() < h {
                    0.75 * (1.0 - (x / h) * (x / h))
                } else {
                    0.0
                }
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            RdKernel::Uniform => "uniform",
            RdKernel::Triangular => "triangular",
            RdKernel::Epanechnikov => "epanechnikov",
        }
    }
}

/// Learner of the outcome: a regressor, or a classifier for a binary outcome.
enum OutcomeLearner {
    Regressor(Box<dyn Regressor>),
    Classifier(Box<dyn Classifier>),
}

/// RD with flexible covariate adjustment. A fuzzy design ([`RdFlex::fuzzy`])
/// also adjusts the treatment with a classifier.
pub struct RdFlex {
    ml_g: OutcomeLearner,
    ml_m: Option<Box<dyn Classifier>>,
    cutoff: f64,
    h_fs: Option<f64>,
    first_stage: FirstStage,
    kernel: RdKernel,
    n_iterations: usize,
    rd_options: RdOptions,
    rd_covariates: Option<Array2<f64>>,
    rd_clusters: Option<Vec<i64>>,
    splitting: SampleSplitting,
}

impl RdFlex {
    /// Sharp design: treatment is `1{score ≥ cutoff}`. `ml_g` learns the
    /// outcome and must support sample weights.
    pub fn sharp(ml_g: impl Regressor + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), None)
    }

    /// Fuzzy design: `ml_m` (supporting sample weights) learns the treatment.
    pub fn fuzzy(ml_g: impl Regressor + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed(Box::new(ml_g), Some(Box::new(ml_m)))
    }

    /// `ml_m = Some(_)` fits a fuzzy design.
    pub fn from_boxed(ml_g: Box<dyn Regressor>, ml_m: Option<Box<dyn Classifier>>) -> Self {
        Self::with_outcome(OutcomeLearner::Regressor(ml_g), ml_m)
    }

    /// Sharp design with a binary (0/1) outcome learned by the classifier
    /// `ml_g` (its class-1 probability), as DoubleML with a classifier.
    pub fn sharp_binary(ml_g: impl Classifier + 'static) -> Self {
        Self::from_boxed_binary(Box::new(ml_g), None)
    }

    /// Fuzzy design with a binary outcome learned by the classifier `ml_g`.
    pub fn fuzzy_binary(ml_g: impl Classifier + 'static, ml_m: impl Classifier + 'static) -> Self {
        Self::from_boxed_binary(Box::new(ml_g), Some(Box::new(ml_m)))
    }

    /// Binary outcome with a boxed classifier; `ml_m = Some(_)` fits a fuzzy
    /// design.
    pub fn from_boxed_binary(ml_g: Box<dyn Classifier>, ml_m: Option<Box<dyn Classifier>>) -> Self {
        Self::with_outcome(OutcomeLearner::Classifier(ml_g), ml_m)
    }

    fn with_outcome(ml_g: OutcomeLearner, ml_m: Option<Box<dyn Classifier>>) -> Self {
        Self {
            ml_g,
            ml_m,
            cutoff: 0.0,
            h_fs: None,
            first_stage: FirstStage::default(),
            kernel: RdKernel::default(),
            n_iterations: 2,
            rd_options: RdOptions::default(),
            rd_covariates: None,
            rd_clusters: None,
            splitting: SampleSplitting::default(),
        }
    }

    /// Options of the RD estimator (`rdrobust`'s keyword arguments in
    /// DoubleML: polynomial order, kernel, bandwidth selector, variance, ...).
    /// The initial first-stage bandwidth is always selected with the defaults.
    pub fn rd_options(mut self, options: RdOptions) -> Self {
        self.rd_options = options;
        self
    }

    /// Covariates for the RD estimator itself (`rdrobust`'s `covs`), one row
    /// per observation; the first stage uses the data's covariates.
    pub fn rd_covariates(mut self, covs: Array2<f64>) -> Self {
        self.rd_covariates = Some(covs);
        self
    }

    /// Cluster labels for cluster-robust RD standard errors (`rdrobust`'s
    /// `cluster`); the sample splits are not clustered, as in DoubleML.
    pub fn rd_clusters(mut self, labels: Vec<i64>) -> Self {
        self.rd_clusters = Some(labels);
        self
    }

    pub fn cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Initial first-stage bandwidth (default: `rdbwselect` without covariates).
    pub fn h_fs(mut self, h: f64) -> Self {
        self.h_fs = Some(h);
        self
    }

    pub fn first_stage(mut self, spec: FirstStage) -> Self {
        self.first_stage = spec;
        self
    }

    pub fn kernel(mut self, kernel: RdKernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Rounds of first-stage fitting and bandwidth updating (default 2).
    pub fn n_iterations(mut self, n: usize) -> Self {
        self.n_iterations = n;
        self
    }

    /// Number of cross-fitting folds (default 5).
    pub fn n_folds(mut self, n_folds: usize) -> Self {
        self.splitting.n_folds = n_folds;
        self
    }

    /// Number of independent sample splits (default 1), aggregated by the median.
    pub fn n_rep(mut self, n_rep: usize) -> Self {
        self.splitting.n_rep = n_rep;
        self
    }

    /// Seed for the sample splits (default 0).
    pub fn seed(mut self, seed: u64) -> Self {
        self.splitting.seed = seed;
        self
    }

    /// Fit with sample splits stratified by the treatment; `score` is the
    /// running variable.
    pub fn fit(&self, data: &DmlData, score: ArrayView1<'_, f64>) -> Result<RdFlexFit> {
        self.validate(data, score)?;
        let codes: Vec<usize> = data.treatment(0).iter().map(|&v| v as usize).collect();
        let splits = self.splitting.draw_stratified(&codes)?;
        self.fit_with_folds(data, score, &splits)
    }

    /// Fit on given sample splits (one `Vec<Fold>` per repetition).
    pub fn fit_with_folds(
        &self,
        data: &DmlData,
        score: ArrayView1<'_, f64>,
        splits: &[Vec<Fold>],
    ) -> Result<RdFlexFit> {
        self.validate(data, score)?;
        if splits.is_empty() {
            return Err(DmlError::InvalidArgument(
                "need at least one repetition".into(),
            ));
        }
        let (y, d, x) = (data.y(), data.treatment(0), data.x());
        let score: Array1<f64> = score.mapv(|s| s - self.cutoff);
        let fuzzy = self.ml_m.is_some();
        let h_fs = match self.h_fs {
            Some(h) => h,
            None => {
                let (h, b) = rdbwselect(y, score.view(), fuzzy.then_some(d), 0.0)?;
                h.max(b)
            }
        };
        let (zx, zx_left, zx_right) = self.design(score.view(), x);
        let w0: Array1<f64> = score.mapv(|s| self.kernel.weight(s, h_fs));

        let mut reps = Vec::with_capacity(splits.len());
        let outcome = match &self.ml_g {
            OutcomeLearner::Regressor(r) => Nuisance::Regressor(r.as_ref()),
            OutcomeLearner::Classifier(c) => Nuisance::Classifier(c.as_ref()),
        };
        for folds in splits {
            let mut weights = w0.clone();
            let mut bandwidth = None;
            let mut last = None;
            for iteration in 0..self.n_iterations {
                let eta_y = adjust(outcome, &zx, &zx_left, &zx_right, y, &weights, folds)?;
                let m_y = &y - &eta_y;
                let m_d = match &self.ml_m {
                    Some(m) => {
                        let eta_d = adjust(
                            Nuisance::Classifier(m.as_ref()),
                            &zx,
                            &zx_left,
                            &zx_right,
                            d,
                            &weights,
                            folds,
                        )?;
                        Some(&d - &eta_d)
                    }
                    None => None,
                };
                let mut rd_data = RdData::new(m_y.view(), score.view());
                rd_data.fuzzy = m_d.as_ref().map(|t| t.view());
                rd_data.covs = self.rd_covariates.as_ref().map(|z| z.view());
                rd_data.cluster = self.rd_clusters.as_deref();
                if iteration + 1 < self.n_iterations {
                    // Intermediate rounds select the RD bandwidth afresh and
                    // update the first-stage weights with it (DoubleML).
                    let est = rd_estimate(&rd_data, &self.rd_options, None)?;
                    let (h, b) = (est.h.0.max(est.h.1), est.b.0.max(est.b.1));
                    weights = score.mapv(|s| self.kernel.weight(s, h));
                    bandwidth = Some(RdBandwidths::common(h, b));
                } else {
                    last = Some(rd_estimate(&rd_data, &self.rd_options, bandwidth)?);
                }
            }
            reps.push(last.expect("n_iterations >= 1"));
        }
        Ok(RdFlexFit::new(reps, h_fs, fuzzy, self.cutoff, self.kernel))
    }

    fn validate(&self, data: &DmlData, score: ArrayView1<'_, f64>) -> Result<()> {
        if data.n_treat() != 1 || !data.is_binary_treatment(0) {
            return Err(DmlError::InvalidArgument(
                "RDFlex needs exactly one binary (0/1) treatment".into(),
            ));
        }
        if data.clusters().is_some() || data.z().is_some() {
            return Err(DmlError::InvalidArgument(
                "RDFlex does not support clusters or instruments".into(),
            ));
        }
        if score.len() != data.n_obs() || !score.iter().all(|s| s.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "the score needs one finite value per observation".into(),
            ));
        }
        let (lo, hi) = score
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &s| {
                (a.min(s), b.max(s))
            });
        if !(lo <= self.cutoff && self.cutoff <= hi) {
            return Err(DmlError::InvalidArgument(
                "the cutoff is not within the range of the score".into(),
            ));
        }
        if self.n_iterations == 0 {
            return Err(DmlError::InvalidArgument(
                "n_iterations must be at least 1".into(),
            ));
        }
        if self.h_fs.is_some_and(|h| !(h.is_finite() && h > 0.0)) {
            return Err(DmlError::InvalidArgument("h_fs must be positive".into()));
        }
        if matches!(self.ml_g, OutcomeLearner::Classifier(_))
            && !data.y().iter().all(|&v| v == 0.0 || v == 1.0)
        {
            return Err(DmlError::InvalidArgument(
                "a classifier for the outcome needs a binary (0/1) outcome".into(),
            ));
        }
        let n = data.n_obs();
        if self.rd_covariates.as_ref().is_some_and(|z| z.nrows() != n)
            || self.rd_clusters.as_ref().is_some_and(|c| c.len() != n)
        {
            return Err(DmlError::DimensionMismatch(
                "RD covariates and clusters need one row per observation".into(),
            ));
        }
        Ok(())
    }

    /// First-stage design `[Z, X]` and its values at the cutoff from the left
    /// (`Z = 0`) and from the right (cutoff indicator 1, score 0).
    fn design(
        &self,
        score: ArrayView1<'_, f64>,
        x: ArrayView2<'_, f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = score.len();
        let z_ind = score.mapv(|s| f64::from(s >= 0.0));
        let z = match self.first_stage {
            FirstStage::Cutoff => z_ind.insert_axis(Axis(1)),
            FirstStage::CutoffAndScore => ndarray::stack![Axis(1), z_ind, score],
            FirstStage::InteractedCutoffAndScore => {
                ndarray::stack![Axis(1), z_ind, &z_ind * &score, score]
            }
        };
        let k = z.ncols();
        let mut z_right = Array2::zeros((n, k));
        z_right.column_mut(0).fill(1.0);
        let zx = concatenate![Axis(1), z, x];
        let zx_left = concatenate![Axis(1), Array2::zeros((n, k)), x];
        let zx_right = concatenate![Axis(1), z_right, x];
        (zx, zx_left, zx_right)
    }
}

#[derive(Clone, Copy)]
enum Nuisance<'a> {
    Regressor(&'a dyn Regressor),
    Classifier(&'a dyn Classifier),
}

/// Cross-fitted `(μ(left) + μ(right)) / 2` at the cutoff, fitted with
/// `weights`.
fn adjust(
    learner: Nuisance<'_>,
    zx: &Array2<f64>,
    zx_left: &Array2<f64>,
    zx_right: &Array2<f64>,
    target: ArrayView1<'_, f64>,
    weights: &Array1<f64>,
    folds: &[Fold],
) -> Result<Array1<f64>> {
    let n = target.len();
    let fit_fold = |fold: &Fold| -> Result<(Array1<f64>, Array1<f64>)> {
        let xt = zx.select(Axis(0), &fold.train);
        let (yt, wt) = (
            gather(target, &fold.train),
            gather(weights.view(), &fold.train),
        );
        let (xl, xr) = (
            zx_left.select(Axis(0), &fold.test),
            zx_right.select(Axis(0), &fold.test),
        );
        match learner {
            Nuisance::Regressor(r) => {
                let mut m = r.fresh();
                m.fit_weighted(xt.view(), yt.view(), wt.view())?;
                Ok((m.predict(xl.view())?, m.predict(xr.view())?))
            }
            Nuisance::Classifier(c) => {
                let mut m = c.fresh();
                m.fit_weighted(xt.view(), yt.view(), wt.view())?;
                Ok((m.predict_proba(xl.view())?, m.predict_proba(xr.view())?))
            }
        }
    };
    #[cfg(feature = "parallel")]
    let per_fold: Vec<_> = {
        use rayon::prelude::*;
        folds.par_iter().map(fit_fold).collect::<Result<_>>()?
    };
    #[cfg(not(feature = "parallel"))]
    let per_fold: Vec<_> = folds.iter().map(fit_fold).collect::<Result<_>>()?;
    let mut eta = Array1::from_elem(n, f64::NAN);
    for (fold, (left, right)) in folds.iter().zip(per_fold) {
        for (k, &i) in fold.test.iter().enumerate() {
            eta[i] = (left[k] + right[k]) / 2.0;
        }
    }
    Ok(eta)
}

/// Fitted [`RdFlex`]: the conventional, bias-corrected and robust RD
/// estimates (rows), aggregated over repetitions as DoubleML.
#[derive(Debug, Clone)]
pub struct RdFlexFit {
    reps: Vec<RdEstimate>,
    coef: [f64; 3],
    se: [f64; 3],
    ci: [[f64; 2]; 3],
    h_fs: f64,
    fuzzy: bool,
    cutoff: f64,
    kernel: RdKernel,
}

const LABELS: [&str; 3] = ["Conventional", "Bias-Corrected", "Robust"];

impl RdFlexFit {
    fn new(reps: Vec<RdEstimate>, h_fs: f64, fuzzy: bool, cutoff: f64, kernel: RdKernel) -> Self {
        // DoubleML: median estimate; se² = median(se_r² + (θ_r − θ)² / N_h,r).
        let coef: [f64; 3] =
            std::array::from_fn(|k| median(&reps.iter().map(|r| r.coef[k]).collect::<Vec<_>>()));
        let se = std::array::from_fn(|k| {
            let v: Vec<f64> = reps
                .iter()
                .map(|r| {
                    let n_h = (r.n_h.0 + r.n_h.1) as f64;
                    r.se[k] * r.se[k] + (r.coef[k] - coef[k]).powi(2) / n_h
                })
                .collect();
            median(&v).sqrt()
        });
        let ci = std::array::from_fn(|k| {
            std::array::from_fn(|j| median(&reps.iter().map(|r| r.ci[k][j]).collect::<Vec<_>>()))
        });
        Self {
            reps,
            coef,
            se,
            ci,
            h_fs,
            fuzzy,
            cutoff,
            kernel,
        }
    }

    /// Conventional, bias-corrected and robust estimates.
    pub fn coef(&self) -> [f64; 3] {
        self.coef
    }

    pub fn se(&self) -> [f64; 3] {
        self.se
    }

    pub fn t_stat(&self) -> [f64; 3] {
        std::array::from_fn(|k| self.coef[k] / self.se[k])
    }

    pub fn p_value(&self) -> [f64; 3] {
        std::array::from_fn(|k| 2.0 * norm_cdf(-(self.coef[k] / self.se[k]).abs()))
    }

    /// 95% intervals: the median of each repetition's `rdrobust` interval.
    pub fn ci(&self) -> [[f64; 2]; 3] {
        self.ci
    }

    /// Intervals at `level` from each repetition's estimate and standard
    /// error, aggregated by the median (DoubleML's `confint`).
    pub fn confint(&self, level: f64) -> Result<[[f64; 2]; 3]> {
        crate::fit::check_level(level)?;
        let z = norm_ppf(1.0 - (1.0 - level) / 2.0);
        Ok(std::array::from_fn(|k| {
            let lo: Vec<f64> = self.reps.iter().map(|r| r.coef[k] - r.se[k] * z).collect();
            let hi: Vec<f64> = self.reps.iter().map(|r| r.coef[k] + r.se[k] * z).collect();
            [median(&lo), median(&hi)]
        }))
    }

    /// Estimates per repetition, shape `(3, n_rep)`.
    pub fn all_coef(&self) -> Array2<f64> {
        Array2::from_shape_fn((3, self.reps.len()), |(k, r)| self.reps[r].coef[k])
    }

    pub fn all_se(&self) -> Array2<f64> {
        Array2::from_shape_fn((3, self.reps.len()), |(k, r)| self.reps[r].se[k])
    }

    /// Final RD bandwidth `h` of each repetition.
    pub fn h(&self) -> Vec<f64> {
        self.reps.iter().map(|r| r.h.0).collect()
    }

    /// Bias bandwidth `b` of each repetition.
    pub fn b(&self) -> Vec<f64> {
        self.reps.iter().map(|r| r.b.0).collect()
    }

    /// Initial first-stage bandwidth.
    pub fn h_fs(&self) -> f64 {
        self.h_fs
    }

    /// The `rdrobust` result of every repetition.
    pub fn repetitions(&self) -> &[RdEstimate] {
        &self.reps
    }

    pub fn is_fuzzy(&self) -> bool {
        self.fuzzy
    }
}

impl fmt::Display for RdFlexFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (t, p, ci) = (self.t_stat(), self.p_value(), self.ci);
        writeln!(
            f,
            "{:<16}{:>10}{:>10}{:>9}{:>11}{:>24}",
            "Method", "Coef.", "S.E.", "t-stat", "P>|t|", "95% CI"
        )?;
        writeln!(f, "{}", "-".repeat(80))?;
        for k in 0..3 {
            writeln!(
                f,
                "{:<16}{:>10.4}{:>10.4}{:>9.3}{:>11.3e}   [{:.4}, {:.4}]",
                LABELS[k], self.coef[k], self.se[k], t[k], p[k], ci[k][0], ci[k][1]
            )?;
        }
        writeln!(
            f,
            "Design:             {}",
            if self.fuzzy { "fuzzy" } else { "sharp" }
        )?;
        writeln!(f, "Cutoff:             {}", self.cutoff)?;
        writeln!(f, "First-stage kernel: {}", self.kernel.name())?;
        write!(f, "Final bandwidth:    {:?}", self.h())
    }
}

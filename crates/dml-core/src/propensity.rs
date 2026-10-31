//! Propensity score processing, as DoubleML's `PSProcessor`: optional
//! isotonic calibration of the cross-fitted propensity scores against the
//! treatment (or instrument), followed by clipping to `[t, 1 − t]`.
//!
//! [`IsotonicRegression`] is a port of scikit-learn's
//! `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)`.

use ndarray::{Array1, ArrayView1};

use crate::crossfit::Fold;
use crate::data::gather;
use crate::error::{DmlError, Result};

/// Calibration of propensity scores before clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Calibration {
    /// Clip only (default).
    #[default]
    None,
    /// Isotonic regression of the treatment on the propensity scores, fitted
    /// on all observations.
    Isotonic,
    /// Isotonic regression cross-fitted on the model's sample splits.
    IsotonicCrossFitted,
}

/// Monotone (non-decreasing) regression of `y` on a scalar `x`, predicting
/// by linear interpolation and clipping inputs to the training range, as
/// scikit-learn's `IsotonicRegression(out_of_bounds="clip")`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IsotonicRegression {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl IsotonicRegression {
    /// Fit with fitted values bounded to `[y_min, y_max]`.
    pub fn fit(
        x: ArrayView1<'_, f64>,
        y: ArrayView1<'_, f64>,
        y_min: f64,
        y_max: f64,
    ) -> Result<Self> {
        if x.len() != y.len() || x.is_empty() {
            return Err(DmlError::DimensionMismatch(
                "isotonic regression needs equally long, non-empty x and y".into(),
            ));
        }
        if !x.iter().chain(y.iter()).all(|v| v.is_finite()) {
            return Err(DmlError::InvalidArgument(
                "isotonic regression: data contain NaN or infinite values".into(),
            ));
        }
        // Sort by x, then y (np.lexsort((y, X))).
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(y[a].total_cmp(&y[b])));
        // Merge x values closer than float64 resolution, averaging y.
        let eps = 1e-15;
        let (mut ux, mut uy, mut uw) = (Vec::new(), Vec::new(), Vec::new());
        let (mut cur_x, mut cur_y, mut cur_w) = (x[order[0]], 0.0, 0.0);
        for &i in &order {
            if x[i] - cur_x >= eps {
                ux.push(cur_x);
                uw.push(cur_w);
                uy.push(cur_y / cur_w);
                cur_x = x[i];
                cur_w = 1.0;
                cur_y = y[i];
            } else {
                cur_w += 1.0;
                cur_y += y[i];
            }
        }
        ux.push(cur_x);
        uw.push(cur_w);
        uy.push(cur_y / cur_w);
        let mut fitted = pava(&uy, &uw);
        for v in &mut fitted {
            *v = v.clamp(y_min, y_max);
        }
        // Keep the first and last point and every point where the fit changes.
        let n = fitted.len();
        let keep: Vec<usize> = (0..n)
            .filter(|&i| {
                i == 0 || i == n - 1 || fitted[i] != fitted[i - 1] || fitted[i] != fitted[i + 1]
            })
            .collect();
        Ok(Self {
            x: keep.iter().map(|&i| ux[i]).collect(),
            y: keep.iter().map(|&i| fitted[i]).collect(),
        })
    }

    /// Predictions by linear interpolation (SciPy's `interp1d`), with inputs
    /// clipped to the training range.
    pub fn predict(&self, t: ArrayView1<'_, f64>) -> Array1<f64> {
        let (x, y) = (&self.x, &self.y);
        if x.len() == 1 {
            return Array1::from_elem(t.len(), y[0]);
        }
        let (lo_x, hi_x) = (x[0], x[x.len() - 1]);
        t.mapv(|v| {
            let v = v.clamp(lo_x, hi_x);
            // searchsorted(side="left"), clipped to [1, n − 1].
            let idx = x.partition_point(|&xi| xi < v).clamp(1, x.len() - 1);
            let (x_lo, x_hi, y_lo, y_hi) = (x[idx - 1], x[idx], y[idx - 1], y[idx]);
            ((v - x_lo) / (x_hi - x_lo)) * y_hi + ((x_hi - v) / (x_hi - x_lo)) * y_lo
        })
    }
}

/// Pool-adjacent-violators: the weighted non-decreasing fit of `y`.
fn pava(y: &[f64], w: &[f64]) -> Vec<f64> {
    // Blocks of (weighted sum, weight, length), merged while they violate
    // monotonicity.
    let mut blocks: Vec<(f64, f64, usize)> = Vec::with_capacity(y.len());
    for (&yi, &wi) in y.iter().zip(w) {
        let mut block = (wi * yi, wi, 1usize);
        while let Some(&(s, ww, len)) = blocks.last() {
            if s / ww >= block.0 / block.1 {
                blocks.pop();
                block = (s + block.0, ww + block.1, len + block.2);
            } else {
                break;
            }
        }
        blocks.push(block);
    }
    let mut out = Vec::with_capacity(y.len());
    for (s, ww, len) in blocks {
        out.extend(std::iter::repeat(s / ww).take(len));
    }
    out
}

/// Calibrate (optionally) and clip propensity scores `m` of the 0/1
/// `labels` (DoubleML's `PSProcessor.adjust_ps`).
pub(crate) fn adjust_propensity(
    m: Array1<f64>,
    labels: ArrayView1<'_, f64>,
    folds: &[Fold],
    clipping_threshold: f64,
    calibration: Calibration,
) -> Result<Array1<f64>> {
    let calibrated = match calibration {
        Calibration::None => m,
        Calibration::Isotonic => {
            IsotonicRegression::fit(m.view(), labels, 0.0, 1.0)?.predict(m.view())
        }
        Calibration::IsotonicCrossFitted => {
            let mut out = Array1::from_elem(m.len(), f64::NAN);
            for fold in folds {
                let iso = IsotonicRegression::fit(
                    gather(m.view(), &fold.train).view(),
                    gather(labels, &fold.train).view(),
                    0.0,
                    1.0,
                )?;
                let pred = iso.predict(gather(m.view(), &fold.test).view());
                for (p, &i) in pred.iter().zip(&fold.test) {
                    out[i] = *p;
                }
            }
            out
        }
    };
    let t = clipping_threshold;
    Ok(calibrated.mapv(|p| p.clamp(t, 1.0 - t)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn pava_matches_scikit_learn_example() {
        // sklearn.isotonic.isotonic_regression([5, 3, 1, 2, 8, 10, 7, 9, 6, 4])
        let fit = pava(
            &[5.0, 3.0, 1.0, 2.0, 8.0, 10.0, 7.0, 9.0, 6.0, 4.0],
            &[1.0; 10],
        );
        let expected = [
            2.75,
            2.75,
            2.75,
            2.75,
            44.0 / 6.0,
            44.0 / 6.0,
            44.0 / 6.0,
            44.0 / 6.0,
            44.0 / 6.0,
            44.0 / 6.0,
        ];
        for (a, b) in fit.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn matches_scikit_learn_with_ties() {
        // sklearn IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y).predict(t)
        let x = array![
            0.6, 0.3, 0.0, 0.0, 0.8, 0.9, 0.6, 0.7, 0.5, 0.9, 0.8, 0.0, 0.9, 0.0, 0.7, 0.2, 0.9,
            0.5, 0.3, 0.4, 0.0, 0.1, 0.7, 0.6
        ];
        let y = array![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0
        ];
        let t = array![-0.1, 0.0, 0.05, 0.33, 0.5, 0.71, 0.95, 1.2];
        let expected = [0.0, 0.0, 0.0, 0.15000000000000008, 0.5, 0.7, 1.0, 1.0];
        let p = IsotonicRegression::fit(x.view(), y.view(), 0.0, 1.0)
            .unwrap()
            .predict(t.view());
        for (a, b) in p.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn interpolates_and_clips() {
        let iso = IsotonicRegression::fit(
            array![0.1, 0.2, 0.3, 0.4].view(),
            array![0.0, 1.0, 0.0, 1.0].view(),
            0.0,
            1.0,
        )
        .unwrap();
        // Fit: [0, 0.5, 0.5, 1]; predictions interpolate and clip to [0.1, 0.4].
        let p = iso.predict(array![0.0, 0.15, 0.25, 0.35, 1.0].view());
        let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
        for (a, b) in p.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}

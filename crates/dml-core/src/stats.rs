//! Standard normal distribution helpers for Wald inference (no external deps).
//!
//! Accurate to near machine precision, so p-values and critical values match
//! SciPy (and therefore Python DoubleML) to ~1e-14.

use std::f64::consts::{PI, SQRT_2};

const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;

/// Complementary error function `erfc(x) = 1 − erf(x)`.
///
/// Uses the everywhere-positive series
/// `erf(x) = 2/√π · e^{−x²} · Σ 2ⁿ x^{2n+1} / (1·3·…·(2n+1))` for `|x| < 2`
/// and the Laplace continued fraction (modified Lentz) for `|x| ≥ 2`,
/// where the series would lose relative precision in the tail.
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    if x < 2.0 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0.0;
        while term > sum * f64::EPSILON {
            n += 1.0;
            term *= 2.0 * x2 / (2.0 * n + 1.0);
            sum += term;
        }
        return 1.0 - 2.0 * FRAC_1_SQRT_PI * (-x2).exp() * sum;
    }
    if x > 27.3 {
        return 0.0; // below the smallest subnormal
    }
    // erfc(x) = e^{−x²}/√π · 1 / (x + (1/2)/(x + 1/(x + (3/2)/(x + …))))
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for k in 1..1000 {
        let a = k as f64 / 2.0;
        d = x + a * d;
        d = if d.abs() < tiny { 1.0 / tiny } else { 1.0 / d };
        c = x + a / c;
        if c.abs() < tiny {
            c = tiny;
        }
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }
    FRAC_1_SQRT_PI * (-x * x).exp() / f
}

/// Standard normal CDF `Φ(x)`.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / SQRT_2)
}

/// Standard normal survival function `1 − Φ(x)`, accurate in the upper tail.
pub fn norm_sf(x: f64) -> f64 {
    0.5 * erfc(x / SQRT_2)
}

/// Standard normal quantile `Φ⁻¹(p)` for `0 < p < 1`: Acklam's rational
/// approximation followed by one Halley refinement step.
// Coefficients are kept exactly as published.
#[allow(clippy::excessive_precision)]
pub fn norm_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;

    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let x = if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - P_LOW {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    // Halley step on Φ(x) − p; work in the smaller tail to keep precision.
    let e = if p < 0.5 {
        norm_cdf(x) - p
    } else {
        (1.0 - p) - norm_sf(x)
    };
    let u = e * (2.0 * PI).sqrt() * (0.5 * x * x).exp();
    x - u / (1.0 + 0.5 * x * u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    // Reference values from scipy.stats.norm.
    #[test]
    fn cdf_matches_scipy() {
        assert_eq!(norm_cdf(0.0), 0.5);
        assert!(rel(norm_cdf(-1.0), 0.158_655_253_931_457_07) < 1e-14);
        assert!(rel(norm_cdf(1.96), 0.975_002_104_851_779_5) < 1e-14);
        assert!(rel(norm_cdf(-2.5), 0.006_209_665_325_776_132) < 1e-13);
        assert!(rel(norm_sf(5.0), 2.866_515_718_791_933e-7) < 1e-13);
        assert!(rel(norm_sf(10.0), 7.619_853_024_160_47e-24) < 1e-12);
    }

    #[test]
    fn ppf_matches_scipy_and_inverts_cdf() {
        assert!(rel(norm_ppf(0.975), 1.959_963_984_540_054) < 1e-14);
        assert!(rel(norm_ppf(0.995), 2.575_829_303_548_900_4) < 1e-14);
        assert!(rel(norm_ppf(1e-10), -6.361_340_902_404_056) < 1e-13);
        assert_eq!(norm_ppf(0.5), 0.0);
        for &p in &[1e-6, 0.001, 0.02, 0.3, 0.7, 0.99, 0.999_999] {
            assert!(rel(norm_cdf(norm_ppf(p)), p) < 1e-13, "p = {p}");
        }
    }
}

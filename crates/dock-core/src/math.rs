// SPDX-License-Identifier: GPL-3.0-or-later
//! Geometry primitives used by the whole kernel.
//!
//! The quaternion conventions are bit-for-bit compatible with those used by
//! AutoDock Vina (`quaternion.cpp`, Apache-2.0): a quaternion is stored as
//! `(x, y, z, w)` with `w` the scalar part, and the rotation matrix produced by
//! [`quaternion_to_r3`] is
//!
//! ```text
//!       | w²+x²-y²-z²   2(xy-wz)     2(xz+wy)  |
//!   R = | 2(wz+xy)     w²-x²+y²-z²  2(yz-wx)  |
//!       | 2(xz-wy)     2(yz+wx)     w²-x²-y²+z² |
//! ```
//!
//! which is exactly `glam::DMat3::from_quat`. A unit test pins this identity.

pub use glam::{DMat3, DQuat, DVec3};

/// π at full double precision (Vina uses the same literal).
pub const PI: f64 = 3.141_592_653_589_793;
/// Machine epsilon of `f64`; used as Vina's `epsilon_fl` "is this ~zero?" guard.
pub const EPSILON: f64 = f64::EPSILON;
/// The largest finite `f64`; Vina's `max_fl`.
pub const MAX_F: f64 = f64::MAX;

/// Square of a scalar.
#[inline(always)]
pub fn sqr(x: f64) -> f64 {
    x * x
}

/// True when `x` is comfortably below the "infinity" sentinel.
#[inline(always)]
pub fn not_max(x: f64) -> bool {
    x < 0.1 * MAX_F
}

/// Wrap an angle into `[-π, π]` (Vina's `normalize_angle`).
///
/// The C++ original asserts `-π <= x <= π` on entry which is too strict for a
/// general-purpose helper; the Rust version normalises unconditionally and also
/// survives `NaN`/`inf` by returning `0.0`.
/// Wrap an angle into the half-open interval `(-π, π]`.
///
/// Vina's `normalize_angle` uses exactly that range (see `common.h`), which
/// matters for torsion bookkeeping: `-π` is always reported as `+π` so that two
/// conformations that differ by a full turn compare equal.
#[inline]
pub fn normalize_angle(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    if x > -PI && x <= PI {
        return x;
    }
    let mut a = x % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

/// Squared distance between two points.
#[inline(always)]
pub fn distance_sqr(a: DVec3, b: DVec3) -> f64 {
    (a - b).length_squared()
}

/// Rotation matrix of a (unit) quaternion.
#[inline]
pub fn quaternion_to_r3(q: DQuat) -> DMat3 {
    DMat3::from_quat(q)
}

/// Unit quaternion for a rotation of `angle` radians about `axis`.
///
/// `axis` is assumed to be normalised, mirroring Vina's `angle_to_quaternion`.
#[inline]
pub fn angle_to_quaternion(axis: DVec3, angle: f64) -> DQuat {
    let a = normalize_angle(angle);
    DQuat::from_axis_angle(axis, a)
}

/// Quaternion encoding an axis–angle *increment* given as a rotation vector.
///
/// Vina's `angle_to_quaternion(const vec& rotation)`.
#[inline]
pub fn rotation_vector_to_quaternion(rotation: DVec3) -> DQuat {
    let angle = rotation.length();
    if angle > EPSILON {
        DQuat::from_axis_angle(rotation / angle, normalize_angle(angle))
    } else {
        DQuat::IDENTITY
    }
}

/// Left-multiply `q` by the rotation described by `rotation`, then renormalise.
///
/// Vina (`quaternion.cpp`, `quaternion_increment`) uses an approximate
/// renormalisation; we use the exact one, which is strictly more accurate and
/// removes the "rounding errors growing very slowly" caveat in the original.
#[inline]
pub fn quaternion_increment(q: DQuat, rotation: DVec3) -> DQuat {
    let r = rotation_vector_to_quaternion(rotation) * q;
    let len = r.length();
    if len > 0.0 {
        DQuat::from_xyzw(r.x / len, r.y / len, r.z / len, r.w / len)
    } else {
        DQuat::IDENTITY
    }
}

/// Rotation vector taking quaternion `a` to quaternion `b`.
#[inline]
pub fn quaternion_difference(b: DQuat, a: DQuat) -> DVec3 {
    let tmp = b * a.conjugate();
    let c = tmp.w.clamp(-1.0, 1.0);
    let angle = 2.0 * c.acos();
    let angle = if angle > PI { angle - 2.0 * PI } else { angle };
    let s = (angle / 2.0).sin();
    if s.abs() < EPSILON {
        DVec3::ZERO
    } else {
        DVec3::new(tmp.x, tmp.y, tmp.z) * (angle / s)
    }
}

/// `slope_step` from Vina's `potentials.h` — a clamped linear ramp.
///
/// Returns `0` at `x_bad`, `1` at `x_good`, and interpolates linearly between.
#[inline]
pub fn slope_step(x_bad: f64, x_good: f64, x: f64) -> f64 {
    if x_bad < x_good {
        if x <= x_bad {
            return 0.0;
        }
        if x >= x_good {
            return 1.0;
        }
    } else {
        if x >= x_bad {
            return 0.0;
        }
        if x <= x_good {
            return 1.0;
        }
    }
    (x - x_bad) / (x_good - x_bad)
}

/// Vina's `smooth_div`: a division that degrades gracefully near zero.
#[inline]
pub fn smooth_div(x: f64, y: f64) -> f64 {
    if x.abs() < EPSILON {
        return 0.0;
    }
    if y.abs() < EPSILON {
        return if x * y > 0.0 { MAX_F } else { -MAX_F };
    }
    x / y
}

/// Vina's `curl` (`curl.h`): soft-saturate a positive energy.
///
/// Applied both to the energy and to its derivative so that the pair stays
/// consistent, which is what keeps the analytic gradient usable inside BFGS.
#[inline]
pub fn curl(e: &mut f64, deriv: &mut DVec3, v: f64) {
    if *e > 0.0 && not_max(v) {
        let tmp = if v < EPSILON { 0.0 } else { v / (v + *e) };
        *e *= tmp;
        *deriv *= sqr(tmp);
    }
}

/// Scalar-only variant of [`curl`].
#[inline]
pub fn curl_scalar(e: &mut f64, v: f64) {
    if *e > 0.0 && not_max(v) {
        let tmp = if v < EPSILON { 0.0 } else { v / (v + *e) };
        *e *= tmp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_normalisation_wraps_into_range() {
        assert!((normalize_angle(0.0) - 0.0).abs() < 1e-15);
        assert!((normalize_angle(3.0 * PI) - PI).abs() < 1e-12);
        assert!((normalize_angle(-3.0 * PI) - PI).abs() < 1e-12);
        assert!((normalize_angle(4.0 * PI)).abs() < 1e-12);
        assert_eq!(normalize_angle(f64::NAN), 0.0);
    }

    #[test]
    fn quaternion_increment_is_normalised_and_composes() {
        let q = DQuat::from_axis_angle(DVec3::Z, 0.3);
        let q2 = quaternion_increment(q, DVec3::new(0.0, 0.0, 0.9));
        assert!((q2.length() - 1.0).abs() < 1e-14);
        // Rotation matrix must stay orthonormal.
        let m = quaternion_to_r3(q2);
        let id = m * m.transpose();
        assert!((id - DMat3::IDENTITY).abs_diff_eq(DMat3::ZERO, 1e-12));
    }

    #[test]
    fn rotation_matrix_matches_vina_convention() {
        // Vina's quaternion_to_r3 for q = (x=0.2, y=-0.3, z=0.5, w=normalised).
        let mut q = DQuat::from_xyzw(0.2, -0.3, 0.5, 0.7);
        q = q.normalize();
        let m = quaternion_to_r3(q);
        let (a, b, c, d) = (q.w, q.x, q.y, q.z);
        let expect = DMat3::from_cols_array(&[
            a * a + b * b - c * c - d * d,
            2.0 * (a * d + b * c),
            2.0 * (-a * c + b * d),
            2.0 * (-a * d + b * c),
            a * a - b * b + c * c - d * d,
            2.0 * (a * b + c * d),
            2.0 * (a * c + b * d),
            2.0 * (-a * b + c * d),
            a * a - b * b - c * c + d * d,
        ]);
        assert!(m.abs_diff_eq(expect, 1e-14), "mismatch\n{m:?}\n{expect:?}");
    }

    #[test]
    fn quaternion_difference_round_trips() {
        let a = DQuat::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
        let rot = DVec3::new(0.1, -0.4, 0.25);
        let b = quaternion_increment(a, rot);
        let back = quaternion_difference(b, a);
        assert!((back - rot).length() < 1e-12, "{back:?} vs {rot:?}");
    }

    #[test]
    fn slope_step_endpoints() {
        assert_eq!(slope_step(1.5, 0.5, 2.0), 0.0);
        assert_eq!(slope_step(1.5, 0.5, 0.0), 1.0);
        assert!((slope_step(1.5, 0.5, 1.0) - 0.5).abs() < 1e-15);
    }
}

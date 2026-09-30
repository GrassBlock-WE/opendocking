// SPDX-License-Identifier: GPL-3.0-or-later
//! Deterministic pseudo-random numbers.
//!
//! Docking is a stochastic search, so the generator must be:
//!
//! 1. **reproducible** — the same seed must give the same pose list, forever;
//! 2. **self-contained** — no dependency on a `rand` version bump silently
//!    changing the stream;
//! 3. **cheap** — millions of variates are drawn per run;
//! 4. **`Send`** — the island model evaluates millions of conformations in
//!    parallel with `rayon`.
//!
//! The generator is a 64-bit PCG-XSH-RR (O'Neill 2014), a well-tested small
//! family with excellent statistical quality for this purpose.

use crate::math::{DQuat, DVec3, PI};

/// PCG-XSH-RR 64/32 generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// Create a generator from a 64-bit seed.
    pub fn new(seed: u64) -> Self {
        let mut r = Rng {
            state: 0,
            // Any odd stream selector works; this one is the reference default.
            inc: 0xda3e_39cb_94b9_5bdb,
        };
        r.state = 0;
        let _ = r.next_u32();
        r.state = r.state.wrapping_add(seed);
        let _ = r.next_u32();
        r
    }

    /// Derive an independent generator stream, e.g. for one island of the GA.
    pub fn spawn(&self, salt: u64) -> Rng {
        Rng::new(self.state ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// Raw 32-bit output.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // 53 bits of mantissa.
        ((self.next_u32() as u64) << 21 | (self.next_u32() as u64) >> 11) as f64
            * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform `f64` in `[a, b)`; `a` must be `< b`.
    #[inline]
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        debug_assert!(a < b);
        a + (b - a) * self.next_f64()
    }

    /// Uniform integer in the inclusive range `[a, b]`.
    #[inline]
    pub fn int(&mut self, a: i64, b: i64) -> i64 {
        debug_assert!(a <= b);
        let span = (b - a) as u64 + 1;
        a + (self.next_u32() as u64 % span) as i64
    }

    /// Standard normal deviate via the Marsaglia polar method.
    #[inline]
    pub fn normal(&mut self, mean: f64, sigma: f64) -> f64 {
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let scale = (-2.0 * s.ln() / s).sqrt();
                return mean + sigma * u * scale;
            }
        }
    }

    /// Uniformly distributed point inside the unit sphere (Vina's
    /// `random_inside_sphere`).
    #[inline]
    pub fn inside_unit_sphere(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(
                self.uniform(-1.0, 1.0),
                self.uniform(-1.0, 1.0),
                self.uniform(-1.0, 1.0),
            );
            if v.length_squared() < 1.0 {
                return v;
            }
        }
    }

    /// Uniformly distributed point inside the axis-aligned box.
    #[inline]
    pub fn in_box(&mut self, corner1: DVec3, corner2: DVec3) -> DVec3 {
        DVec3::new(
            self.uniform(corner1.x, corner2.x),
            self.uniform(corner1.y, corner2.y),
            self.uniform(corner1.z, corner2.z),
        )
    }

    /// Uniformly distributed rotation (Shoemake's method).
    #[inline]
    pub fn orientation(&mut self) -> DQuat {
        loop {
            let q = DQuat::from_xyzw(
                self.normal(0.0, 1.0),
                self.normal(0.0, 1.0),
                self.normal(0.0, 1.0),
                self.normal(0.0, 1.0),
            );
            let n = q.length();
            if n > 1e-12 {
                return DQuat::from_xyzw(q.x / n, q.y / n, q.z / n, q.w / n);
            }
        }
    }

    /// Uniform angle in `[-π, π)`.
    #[inline]
    pub fn angle(&mut self) -> f64 {
        self.uniform(-PI, PI)
    }
}

/// Entropy-derived seed used when the caller passes `seed = 0`.
pub fn auto_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x2545_F491_4F6C_DD1D);
    let pid = std::process::id() as u64;
    // Mix with a splitmix64 finaliser so consecutive calls differ wildly.
    let mut z = nanos ^ (pid << 32) ^ 0x9E37_79B9_7F4A_7C15;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_reproducible() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let sa: Vec<u32> = (0..64).map(|_| a.next_u32()).collect();
        let sb: Vec<u32> = (0..64).map(|_| b.next_u32()).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn uniforms_are_in_range_and_roughly_uniform() {
        let mut r = Rng::new(7);
        let n = 50_000;
        let mut mean = 0.0;
        for _ in 0..n {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
            mean += x;
        }
        mean /= n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }

    #[test]
    fn orientations_are_unit_and_isotropic() {
        let mut r = Rng::new(11);
        let mut sum = DVec3::ZERO;
        let n = 20_000;
        for _ in 0..n {
            let q = r.orientation();
            assert!((q.length() - 1.0).abs() < 1e-12);
            // The rotation of +Z should average to zero over the sphere.
            sum += q * DVec3::Z;
        }
        assert!((sum / n as f64).length() < 0.02, "bias {:?}", sum / n as f64);
    }

    #[test]
    fn sphere_points_are_inside() {
        let mut r = Rng::new(3);
        for _ in 0..5000 {
            assert!(r.inside_unit_sphere().length() < 1.0);
        }
    }

    #[test]
    fn integer_range_is_inclusive() {
        let mut r = Rng::new(5);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..10_000 {
            let v = r.int(2, 5);
            assert!((2..=5).contains(&v));
            seen_lo |= v == 2;
            seen_hi |= v == 5;
        }
        assert!(seen_lo && seen_hi);
    }

    #[test]
    fn spawned_streams_are_independent() {
        let root = Rng::new(99);
        let mut a = root.spawn(0);
        let mut b = root.spawn(1);
        let sa: Vec<u32> = (0..32).map(|_| a.next_u32()).collect();
        let sb: Vec<u32> = (0..32).map(|_| b.next_u32()).collect();
        assert_ne!(sa, sb);
    }
}

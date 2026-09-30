// SPDX-License-Identifier: GPL-3.0-or-later
//! Solis-Wets adaptive local search — the local optimiser of AutoDock 4's
//! Lamarckian genetic algorithm.
//!
//! # Why not BFGS
//!
//! The reference AutoDock 4 implementation does **not** minimise the
//! Lamarckian GA's offspring with a quasi-Newton method: `solis.c` drives an
//! adaptive random-search procedure after Solis & Wets, *"Minimization by
//! Random Search Techniques"*, Math. Oper. Res. **6**(1), 19–30 (1981). The
//! method is derivative-free, which matters because the AD4 force field's
//! derivative is piecewise (the 12-6/12-10 terms are smoothed over a 0.5 Å
//! plateau, see `Term::Ad4Vdw`) and a quasi-Newton step built on a
//! discontinuous gradient regularly leaves the basin it started in.
//!
//! # The algorithm
//!
//! Each degree of freedom `i` owns a signed step `step[i]`. One **sweep**
//! visits every degree of freedom once, and for each one:
//!
//! 1. trial the point `x + step[i]·e_i`;
//! 2. if it lowers the energy, accept it and grow the step:
//!    `step[i] ← min(α · step[i], max_step)` with `α = 1.2` (a run of
//!    successes accelerates away from the start);
//! 3. if it does not lower the energy, reject it and shrink **and reverse**
//!    the step: `step[i] ← -β · step[i]` with `β = 0.8` (the next sweep probes
//!    the opposite direction with a shorter stride);
//! 4. after `max_fail = 4` consecutive rejections the whole step vector is
//!    **restarted** from its initial values, which is the diversification
//!    mechanism that lets the search escape a shallow local minimum.
//!
//! The search is monotone by construction — a trial is only ever accepted when
//! it is strictly better — so the result can never be worse than the point the
//! caller handed in.
//!
//! # Two bounds that keep the classic algorithm usable
//!
//! The restart of step 4 is what lets the search leave a shallow minimum, but
//! an *unbounded* restart also destroys the refinement phase: every time the
//! four failures accumulate the strides jump back to `initial_step`, so the
//! search can never resolve a minimum better than ~`initial_step · β^max_fail`
//! (≈ 0.2 Å with the AutoDock 4 constants). AutoDock 4 sidesteps that by
//! running a fixed, modest iteration budget per local search; OpenDocking makes
//! it explicit with two documented bounds:
//!
//! * [`SolisWetsParams::max_restarts`] caps the number of step restarts. Once
//!   they are used up the search keeps shrinking the strides, so the tail of
//!   every run is a genuine refinement and the method converges to the
//!   requested `min_step` precision (this is what the quadratic test pins).
//! * [`SolisWetsParams::max_iterations`] caps the number of sweeps, exactly as
//!   AutoDock 4's `sw_max_iter`.
//!
//! The two constants `α = 1.2`, `β = 0.8` and `max_fail = 4` are AutoDock 4's
//! (`solis.c`); the parameterisation in normalised degree-of-freedom units is
//! described below.
//!
//! # Normalised degrees of freedom
//!
//! The degrees of freedom of a docking system do not share a unit: three are
//! translations (Å), three are rotation-vector components (rad) and the rest
//! are torsion angles (rad). AutoDock 4 minimises in *normalised* variables
//! (`[-1, 1]` per axis) for exactly this reason. OpenDocking does the same
//! with an explicit per-DOF scale vector: a normalised step of `s` becomes
//!
//! ```text
//! translation : Δx = s            Å
//! rotation    : Δω = s / Rg       rad      (Rg = ligand gyration radius)
//! torsion     : Δτ = s            rad
//! ```
//!
//! so one scalar `step` means "move the ligand by roughly `step` Å" whichever
//! kind of degree of freedom it drives. This mirrors the rotational mutation
//! amplitude of [`crate::kinematics::mutate_conf`], which is scaled by `1/Rg`
//! for the same reason.

use crate::cancel::{stopped, CancelToken};
use crate::kinematics::{Conf, DofLayout};
use crate::search::OptimizeResult;

/// Tunable constants of the Solis-Wets adaptive search.
///
/// The defaults are AutoDock 4's (`solis.c`: `alpha = 1.2`, `beta = 0.8`,
/// `max_fail = 4`) expressed in normalised degree-of-freedom units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolisWetsParams {
    /// Initial step, in normalised degree-of-freedom units (≈ Å of travel).
    pub initial_step: f64,
    /// Largest magnitude a step may grow to (`alpha`-successes are capped).
    pub max_step: f64,
    /// Steps below this magnitude are not tried any more (rad / Å).
    pub min_step: f64,
    /// Success multiplier (AutoDock 4's `alpha`).
    pub alpha: f64,
    /// Failure multiplier; the step is also negated (AutoDock 4's `beta`).
    pub beta: f64,
    /// Consecutive failures that restart the step vector (`MAXFAIL`).
    pub max_fail: usize,
    /// Hard cap on the number of sweeps.
    pub max_iterations: usize,
    /// Hard cap on the number of step restarts.
    ///
    /// Bounding the restarts is what gives the search a refinement phase: with
    /// the budget exhausted the strides only ever shrink (on failure) or grow
    /// (on success), so the method converges to `min_step` instead of
    /// re-exploring the same basin at `initial_step` until the iteration cap.
    pub max_restarts: usize,
}

impl Default for SolisWetsParams {
    fn default() -> Self {
        SolisWetsParams {
            initial_step: 0.5,
            max_step: 1.0,
            min_step: 1e-6,
            alpha: 1.2,
            beta: 0.8,
            max_fail: 4,
            max_iterations: 300,
            max_restarts: 4,
        }
    }
}

impl SolisWetsParams {
    /// The parameters AutoDock 4 uses for one LGA local search.
    pub fn ad4() -> SolisWetsParams {
        SolisWetsParams::default()
    }

    /// The same constants with a different sweep budget.
    pub fn with_iterations(mut self, iterations: usize) -> SolisWetsParams {
        self.max_iterations = iterations.max(1);
        self
    }

    /// The same constants with a different restart budget.
    pub fn with_restarts(mut self, restarts: usize) -> SolisWetsParams {
        self.max_restarts = restarts;
        self
    }
}

/// The mutable state of one Solis-Wets minimisation.
///
/// The public fields are the algorithm's own bookkeeping — the step vector and
/// the success / failure counters — and are exposed because AutoDock's
/// diagnostics and the GUI's run monitor report exactly these numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct SolisWets {
    /// Current (signed, normalised) step per degree of freedom.
    pub step: Vec<f64>,
    /// Step vector a restart restores.
    pub initial_step: Vec<f64>,
    /// Normalised step → native unit conversion, per degree of freedom.
    pub scale: Vec<f64>,
    /// Accepted moves since the last restart.
    pub n_success: usize,
    /// Consecutive rejected moves.
    pub n_failure: usize,
    /// Accepted moves over the whole run.
    pub total_success: u64,
    /// Rejected moves over the whole run.
    pub total_failure: u64,
    /// Number of step restarts performed.
    pub restarts: usize,
    /// Success multiplier.
    pub alpha: f64,
    /// Failure multiplier.
    pub beta: f64,
    /// Consecutive failures that trigger a restart.
    pub max_fail: usize,
    /// Step magnitude cap.
    pub max_step: f64,
    /// Step magnitude below which a degree of freedom is skipped.
    pub min_step: f64,
}

impl SolisWets {
    /// Build the optimiser for a degree-of-freedom layout.
    ///
    /// `gyration_radius` scales the rotational degrees of freedom; pass `0.0`
    /// (or any non-finite value) and the rotational scale falls back to `1.0`,
    /// which simply means "the rotation vector is used in radians".
    pub fn new(layout: &DofLayout, gyration_radius: f64, params: &SolisWetsParams) -> SolisWets {
        let n = layout.num_floats();
        let mut scale = vec![1.0; n];
        let rg = if gyration_radius.is_finite() && gyration_radius > 1e-3 {
            gyration_radius
        } else {
            1.0
        };
        // Degrees of freedom are laid out as
        // [ligand 0: x y z | ω (3) | torsions ...] [ligand 1: ...] [flex ...]
        let mut off = 0usize;
        for &n_tors in &layout.ligands {
            // translation: 1 Å per unit step, rotation: 1 rad per Rg units.
            scale[off + 3] = 1.0 / rg;
            scale[off + 4] = 1.0 / rg;
            scale[off + 5] = 1.0 / rg;
            off += 6 + n_tors;
        }
        debug_assert_eq!(off + layout.flex.iter().sum::<usize>(), n);

        let step = vec![params.initial_step; n];
        SolisWets {
            initial_step: step.clone(),
            step,
            scale,
            n_success: 0,
            n_failure: 0,
            total_success: 0,
            total_failure: 0,
            restarts: 0,
            alpha: params.alpha,
            beta: params.beta,
            max_fail: params.max_fail.max(1),
            max_step: params.max_step,
            min_step: params.min_step,
        }
    }

    /// Largest step magnitude currently in use.
    pub fn max_step_len(&self) -> f64 {
        self.step.iter().fold(0.0_f64, |m, s| m.max(s.abs()))
    }

    /// Restore every step to its initial value (AutoDock 4's restart).
    pub fn restart(&mut self) {
        self.step.copy_from_slice(&self.initial_step);
        self.n_success = 0;
        self.n_failure = 0;
        self.restarts += 1;
    }

    /// Minimise `f` starting from `conf`, in place.
    ///
    /// `f` is called with a candidate conformation and returns its energy; the
    /// conformation is only ever replaced by a strictly better one, so the
    /// final energy is guaranteed to be `<=` the energy of the starting point
    /// (and the starting point is restored verbatim if it is not, which also
    /// covers a `NaN` from a pathological force field).
    ///
    /// `cancel` is polled before every single trial: a paused run freezes at the
    /// trial boundary (sleeping, not spinning) and continues when it is
    /// resumed, while a cancelled run unwinds within one energy evaluation.
    pub fn minimize<F>(
        &mut self,
        mut f: F,
        conf: &mut Conf,
        layout: &DofLayout,
        params: &SolisWetsParams,
        cancel: Option<&CancelToken>,
    ) -> OptimizeResult
    where
        F: FnMut(&Conf) -> f64,
    {
        let n = layout.num_floats();
        if n == 0 {
            let energy = f(conf);
            return OptimizeResult {
                energy,
                evals: 1,
                steps: 0,
            };
        }

        let x_orig = conf.clone();
        let mut best = x_orig.clone();
        let mut f_best = f(&best);
        let f_orig = f_best;
        let mut evals = 1u64;
        let mut sweeps = 0usize;
        let mut trial = best.clone();
        let mut delta = vec![0.0f64; n];

        'sweeps: for iteration in 0..params.max_iterations.max(1) {
            sweeps = iteration + 1;
            // A sweep that finds every stride below `min_step` means the search
            // has converged: nothing left to try at any scale.
            let mut all_converged = true;

            for i in 0..n {
                // Blocks while the token is paused (a paused search must not
                // burn CPU) and unwinds as soon as it is cancelled.
                if stopped(cancel) {
                    break 'sweeps;
                }
                if self.step[i].abs() < params.min_step {
                    continue;
                }
                all_converged = false;

                // Trial point: the current best with one degree of freedom
                // displaced by its (signed, normalised) step.
                trial.clone_from(&best);
                delta[i] = 1.0;
                trial.increment_flat(layout, &delta, self.step[i] * self.scale[i]);
                delta[i] = 0.0;

                let f_trial = f(&trial);
                evals += 1;

                if f_trial < f_best {
                    best.clone_from(&trial);
                    f_best = f_trial;
                    // Success: accelerate, but never past the cap.
                    self.step[i] = (self.step[i] * self.alpha).clamp(-self.max_step, self.max_step);
                    self.n_success += 1;
                    self.total_success += 1;
                    self.n_failure = 0;
                } else {
                    // Failure: shrink and reverse — the next sweep probes the
                    // opposite direction with a 20 % shorter stride.
                    self.step[i] = -self.step[i] * self.beta;
                    self.n_failure += 1;
                    self.total_failure += 1;
                    if self.n_failure >= self.max_fail {
                        self.n_failure = 0;
                        // The restart budget is finite; once it is spent the
                        // strides keep shrinking, which is the refinement phase.
                        if self.restarts < params.max_restarts {
                            self.restart();
                        }
                    }
                }
            }

            if all_converged {
                break;
            }
        }

        // Monotonicity guard: never hand back something worse than the start
        // (the `is_nan` test covers a pathological force field, for which every
        // comparison would be false).
        if f_best.is_nan() || f_best > f_orig {
            best = x_orig;
            f_best = f_orig;
        }
        *conf = best;

        OptimizeResult {
            energy: f_best,
            evals,
            steps: sweeps,
        }
    }
}

/// One-shot Solis-Wets minimisation of `conf` in place.
///
/// Convenience wrapper around [`SolisWets::new`] + [`SolisWets::minimize`] for
/// callers that do not need the step bookkeeping.
pub fn solis_wets<F>(
    f: F,
    conf: &mut Conf,
    layout: &DofLayout,
    gyration_radius: f64,
    params: &SolisWetsParams,
    cancel: Option<&CancelToken>,
) -> OptimizeResult
where
    F: FnMut(&Conf) -> f64,
{
    let mut sw = SolisWets::new(layout, gyration_radius, params);
    sw.minimize(f, conf, layout, params, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::LigandConf;
    use crate::math::DVec3;

    /// A coupled quadratic over the 6 rigid-body degrees of freedom and two
    /// torsions whose **unique** minimum is `(1.5, -2.0, 0.75)` with an
    /// identity rotation and torsions `[0.4, -1.1]`.
    ///
    /// The `x·y` coupling term is written relative to the target so that it is
    /// zero there: that keeps the surface non-separable without moving the
    /// minimum away from the values the test asserts.
    fn quadratic(conf: &Conf, target: DVec3, tors_target: [f64; 2]) -> f64 {
        let lig = &conf.ligands[0];
        let d = lig.position - target;
        let mut e = d.length_squared();
        e += 0.25 * (d.x - d.y).powi(2);
        e += 0.5 * crate::math::quaternion_difference(lig.orientation, crate::DQuat::IDENTITY)
            .length_squared();
        for (k, t) in lig.torsions.iter().enumerate() {
            let dt = *t - tors_target[k];
            e += 0.5 * dt * dt;
        }
        e
    }

    fn layout() -> DofLayout {
        DofLayout {
            ligands: vec![2],
            flex: vec![],
        }
    }

    fn start() -> Conf {
        Conf {
            ligands: vec![LigandConf::null(DVec3::new(-3.0, 4.0, 2.0), 2)],
            flex: vec![],
        }
    }

    #[test]
    fn converges_on_a_quadratic() {
        let layout = layout();
        let mut conf = start();
        let params = SolisWetsParams {
            max_iterations: 300,
            ..Default::default()
        };
        let res = solis_wets(
            |c| quadratic(c, DVec3::new(1.5, -2.0, 0.75), [0.4, -1.1]),
            &mut conf,
            &layout,
            2.0,
            &params,
            None,
        );
        assert!(
            res.energy < 1e-6,
            "Solis-Wets stalled at {} after {} sweeps",
            res.energy,
            res.steps
        );
        assert!((conf.ligands[0].position - DVec3::new(1.5, -2.0, 0.75)).length() < 5e-3);
        assert!((conf.ligands[0].torsions[0] - 0.4).abs() < 5e-3);
        assert!((conf.ligands[0].torsions[1] + 1.1).abs() < 5e-3);
        assert!(res.evals > 1);
    }

    #[test]
    fn never_worsens_the_starting_energy() {
        let layout = layout();
        // A surface whose minimum is unreachable in one step: the search must
        // still never report an energy above the starting one.
        let target = DVec3::new(50.0, -80.0, 40.0);
        let mut conf = start();
        let f0 = quadratic(&conf, target, [0.4, -1.1]);
        let params = SolisWetsParams {
            max_iterations: 5,
            initial_step: 0.2,
            ..Default::default()
        };
        let res = solis_wets(
            |c| quadratic(c, target, [0.4, -1.1]),
            &mut conf,
            &layout,
            2.0,
            &params,
            None,
        );
        assert!(res.energy <= f0, "{} > {}", res.energy, f0);
    }

    #[test]
    fn an_already_optimal_start_is_returned_untouched() {
        let layout = layout();
        let target = DVec3::new(1.5, -2.0, 0.75);
        let mut conf = Conf {
            ligands: vec![LigandConf {
                position: target,
                orientation: crate::DQuat::IDENTITY,
                torsions: vec![0.4, -1.1],
            }],
            flex: vec![],
        };
        let before = conf.clone();
        let params = SolisWetsParams {
            max_iterations: 20,
            ..Default::default()
        };
        let res = solis_wets(
            |c| quadratic(c, target, [0.4, -1.1]),
            &mut conf,
            &layout,
            2.0,
            &params,
            None,
        );
        assert!(res.energy < 1e-12, "{}", res.energy);
        // Every trial failed, so nothing may have moved.
        assert_eq!(conf, before);
    }

    #[test]
    fn step_adaptation_follows_the_autodock_constants() {
        let layout = layout();
        let params = SolisWetsParams::default();
        let sw = SolisWets::new(&layout, 2.0, &params);
        assert_eq!(sw.step.len(), 8);
        assert!((sw.step[0] - 0.5).abs() < 1e-15);
        // The rotational block is scaled by 1/Rg.
        assert!((sw.scale[3] - 0.5).abs() < 1e-15);
        assert!((sw.scale[0] - 1.0).abs() < 1e-15);

        // f(x) = x has its minimum at -inf, so the search walks in one
        // direction for ever: the stride must grow by alpha on every success
        // and must saturate at `max_step`, never beyond.
        let mut sw2 = SolisWets::new(&layout, 2.0, &params);
        let mut conf = Conf {
            ligands: vec![LigandConf::null(DVec3::new(10.0, 0.0, 0.0), 2)],
            flex: vec![],
        };
        let res = sw2.minimize(
            |c| c.ligands[0].position.x,
            &mut conf,
            &layout,
            &params,
            None,
        );
        assert!(res.energy < 10.0, "the search did not walk downhill");
        assert!(sw2.total_success > 0);
        for s in &sw2.step {
            assert!(
                s.abs() <= params.max_step + 1e-12,
                "a stride escaped the cap: {s}"
            );
        }
        assert!(
            (sw2.step[0].abs() - params.max_step).abs() < 1e-12,
            "the driven degree of freedom should have saturated at the cap, got {}",
            sw2.step[0]
        );
    }

    #[test]
    fn failures_shrink_and_flip_until_the_step_restarts() {
        // f is constant, so every trial fails: the step must shrink by beta and
        // change sign each time, and the step vector must restart after
        // max_fail = 4 consecutive failures.
        let layout = layout();
        let params = SolisWetsParams {
            max_iterations: 1,
            max_fail: 4,
            ..Default::default()
        };
        let mut sw = SolisWets::new(&layout, 2.0, &params);
        let mut conf = start();
        sw.minimize(|_| 1.0, &mut conf, &layout, &params, None);
        // 8 degrees of freedom: the restart fires on the 4th and 8th failure.
        assert_eq!(sw.restarts, 2, "restart bookkeeping: {sw:?}");
        assert_eq!(sw.n_failure, 0, "a restart resets the failure counter");
        assert!(sw.total_failure >= 8);
        // The step was restored by the last restart.
        assert!((sw.step[0] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn the_restart_budget_is_finite() {
        // With the restart budget exhausted the strides keep shrinking, which
        // is what lets the search refine instead of re-exploring for ever.
        let layout = layout();
        let params = SolisWetsParams {
            max_iterations: 6,
            max_fail: 4,
            max_restarts: 1,
            ..Default::default()
        };
        let mut sw = SolisWets::new(&layout, 2.0, &params);
        let mut conf = start();
        sw.minimize(|_| 1.0, &mut conf, &layout, &params, None);
        assert_eq!(sw.restarts, 1);
        assert!(
            sw.max_step_len() < 0.5 * 0.8f64.powi(4),
            "strides did not shrink after the restart budget ran out: {}",
            sw.max_step_len()
        );
    }

    #[test]
    fn a_failure_reverses_the_probed_direction() {
        // f has its minimum at the start: the first trial (+step) fails, so
        // the step must come back negated and shrunk by beta.
        let layout = DofLayout {
            ligands: vec![0],
            flex: vec![],
        };
        let params = SolisWetsParams {
            max_iterations: 1,
            max_restarts: 0,
            ..Default::default()
        };
        let mut sw = SolisWets::new(&layout, 1.0, &params);
        let mut conf = Conf {
            ligands: vec![LigandConf::null(DVec3::ZERO, 0)],
            flex: vec![],
        };
        sw.minimize(
            |c| c.ligands[0].position.length_squared(),
            &mut conf,
            &layout,
            &params,
            None,
        );
        assert!((sw.step[0] - -0.4).abs() < 1e-12, "step {}", sw.step[0]);
    }

    #[test]
    fn is_reproducible_for_a_fixed_start() {
        let layout = layout();
        let params = SolisWetsParams::default();
        let mut a = start();
        let mut b = start();
        let ra = solis_wets(
            |c| quadratic(c, DVec3::new(1.5, -2.0, 0.75), [0.4, -1.1]),
            &mut a,
            &layout,
            2.0,
            &params,
            None,
        );
        let rb = solis_wets(
            |c| quadratic(c, DVec3::new(1.5, -2.0, 0.75), [0.4, -1.1]),
            &mut b,
            &layout,
            2.0,
            &params,
            None,
        );
        assert_eq!(ra, rb);
        assert_eq!(a, b);
    }

    #[test]
    fn cancellation_unwinds_promptly() {
        use crate::cancel::CancelToken;
        use std::time::{Duration, Instant};

        let layout = layout();
        let token = CancelToken::new();
        token.cancel();
        let params = SolisWetsParams::default();
        let mut conf = start();
        let started = Instant::now();
        let res = solis_wets(
            |c| quadratic(c, DVec3::new(1.5, -2.0, 0.75), [0.4, -1.1]),
            &mut conf,
            &layout,
            2.0,
            &params,
            Some(&token),
        );
        assert!(started.elapsed() < Duration::from_millis(100));
        // One evaluation of the starting point only.
        assert_eq!(res.evals, 1);
    }

    #[test]
    fn zero_degrees_of_freedom_is_a_single_evaluation() {
        let layout = DofLayout {
            ligands: vec![],
            flex: vec![],
        };
        let mut conf = Conf::default();
        let res = solis_wets(|_| 3.5, &mut conf, &layout, 1.0, &SolisWetsParams::default(), None);
        assert_eq!(res.evals, 1);
        assert!((res.energy - 3.5).abs() < 1e-15);
    }
}

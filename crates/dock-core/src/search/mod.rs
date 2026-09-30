// SPDX-License-Identifier: GPL-3.0-or-later
//! Conformational search: BFGS and Solis-Wets local optimisation, Monte-Carlo
//! iterated local search and an island-model Lamarckian genetic algorithm.
//!
//! # The protocol
//!
//! OpenDocking follows the AutoDock Vina search protocol
//! (`monte_carlo.cpp` / `bfgs.h`, Apache-2.0) and adds an island-model genetic
//! algorithm for difficult, highly flexible ligands:
//!
//! 1. **Randomise** the ligand inside the box (translation, rotation, torsions).
//! 2. **Mutate** exactly one degree of freedom (Vina's `mutate_conf`).
//! 3. **Minimise** locally with BFGS using the analytic gradient.
//! 4. **Accept** the result with the Metropolis criterion at `T = 1.2`
//!    (600 K with `R = 2 cal/mol/K`, as in the reference implementation).
//! 5. Keep every accepted minimum in a bounded, RMSD-deduplicated container.
//! 6. Repeat from `exhaustiveness` independent random starts in parallel.
//!
//! The island-model Lamarckian GA of AutoDock 4 uses the same shape but
//! minimises every offspring with the derivative-free adaptive
//! [Solis-Wets](solis) search instead of BFGS, which is what AutoDock 4 itself
//! does; [`LocalSearch`] selects between the two.
//!
//! # Cancellation
//!
//! Every loop polls an optional [`CancelToken`] at each generation and each
//! local-search step, so a run can be paused, resumed and aborted from another
//! thread and still returns the best pose found so far. The original entry
//! points keep their signatures and simply pass `None`.
//!
//! Everything is deterministic given the seed, so any run can be reproduced
//! bit for bit.

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::cancel::{abort_requested, stopped, CancelToken};
use crate::kinematics::{mutate_conf, Conf, DofLayout, LigandConf};
use crate::math::{DVec3, EPSILON, MAX_F};
use crate::molecule::{Atom, Pair};
use crate::rng::Rng;
use crate::scoring::ScoreComponents;

pub mod solis;

pub use solis::{solis_wets, SolisWets, SolisWetsParams};

/// Default number of sweeps for one Solis-Wets local search.
///
/// AutoDock 4 runs a fixed budget per local search and relies on the step
/// restarts to converge; the search itself converges once every stride has
/// shrunk below `min_step` (see [`SolisWetsParams`]), so this is a safety cap
/// rather than a typical cost.
pub const SOLIS_LOCAL_ITERATIONS: usize = 300;

/// Which local optimiser relaxes a candidate conformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LocalSearch {
    /// BFGS with the analytic gradient — the AutoDock Vina protocol.
    #[default]
    Bfgs,
    /// AutoDock 4's derivative-free adaptive Solis-Wets search.
    SolisWets,
}

impl LocalSearch {
    /// Lower-case identifier used by the CLI / Python layer.
    pub fn name(self) -> &'static str {
        match self {
            LocalSearch::Bfgs => "bfgs",
            LocalSearch::SolisWets => "solis_wets",
        }
    }

    /// Parse the identifier.
    pub fn parse(s: &str) -> Option<LocalSearch> {
        match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "bfgs" | "lbfgs" | "l_bfgs" | "quasi_newton" => Some(LocalSearch::Bfgs),
            "solis" | "solis_wets" | "soliswets" | "sw" => Some(LocalSearch::SolisWets),
            _ => None,
        }
    }
}

impl std::fmt::Display for LocalSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Energy caps applied during the search (Vina's `hunt_cap`) and during final
/// scoring (Vina's `authentic_v`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Caps {
    /// Cap for the intra-ligand pair terms.
    pub intra: f64,
    /// Cap for the grid (ligand–receptor) term.
    pub grid: f64,
    /// Cap for the explicit intermolecular / flex pair terms.
    pub pairs: f64,
}

impl Caps {
    /// The loose caps used for reported energies (Vina's `authentic_v`).
    pub fn authentic() -> Caps {
        Caps {
            intra: 1000.0,
            grid: 1000.0,
            pairs: 1000.0,
        }
    }

    /// The tight caps used during the search (Vina's `hunt_cap`, i.e.
    /// `vec(10, 1.5, 10)`).
    pub fn hunt() -> Caps {
        Caps {
            intra: 10.0,
            grid: 1.5,
            pairs: 10.0,
        }
    }

    /// Replace the grid cap (used while re-refining out-of-box poses).
    pub fn with_grid_cap(mut self, cap: f64) -> Caps {
        self.grid = cap;
        self
    }
}

/// A docked pose.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    /// The conformation.
    pub conf: Conf,
    /// The energy the search minimised (the reported affinity).
    pub energy: f64,
    /// RMSD (lower bound) to the best pose.
    pub lower_bound: f64,
    /// RMSD (upper bound, atom-to-atom) to the best pose.
    pub upper_bound: f64,
    /// Full energy decomposition.
    pub components: ScoreComponents,
    /// Heavy-atom coordinates of the ligand.
    pub coords: Vec<DVec3>,
    /// Whether the ligand lies inside the search box.
    pub in_box: bool,
}

impl Pose {
    /// An empty, maximally bad pose.
    pub fn empty() -> Pose {
        Pose {
            conf: Conf::default(),
            energy: MAX_F,
            lower_bound: 0.0,
            upper_bound: 0.0,
            components: ScoreComponents::default(),
            coords: Vec::new(),
            in_box: false,
        }
    }
}

/// Everything the search needs from a docking system.
///
/// Implemented by [`crate::docking::System`]. The trait exists so that the
/// search code can be unit-tested against cheap analytic energy surfaces.
pub trait EnergyModel: Send {
    /// The degree-of-freedom layout.
    fn layout(&self) -> &DofLayout;

    /// Number of movable atoms (ligand + flexible residues).
    fn num_movable_atoms(&self) -> usize;

    /// Gyration radius of the ligand, used to scale rotational mutations.
    fn gyration_radius(&self) -> f64;

    /// Pose the system at `conf`, evaluate the energy and fill `grad` with the
    /// flat degree-of-freedom gradient. Returns the total energy.
    fn eval_deriv(&mut self, conf: &Conf, caps: Caps, grad: &mut [f64]) -> f64;

    /// Pose the system at `conf` and evaluate the energy only.
    fn eval(&mut self, conf: &Conf, caps: Caps) -> f64;

    /// Full energy decomposition at the current coordinates.
    fn score(&mut self, conf: &Conf, caps: Caps) -> ScoreComponents;

    /// Ligand heavy-atom coordinates after the most recent evaluation.
    fn ligand_coords(&self) -> Vec<DVec3>;

    /// Ligand atoms with their current coordinates.
    fn ligand_atoms(&self) -> Vec<Atom>;

    /// `true` when every heavy movable atom is inside the search box.
    fn within_box(&self, margin: f64) -> bool;
}

/// Tunable search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchParams {
    /// Number of independent Monte-Carlo runs (Vina's `exhaustiveness`).
    pub exhaustiveness: usize,
    /// Number of poses to retain.
    pub num_poses: usize,
    /// RMSD below which two poses are considered identical (Å).
    pub min_rmsd: f64,
    /// Optional hard cap on the number of energy evaluations per run.
    pub max_evals: u64,
    /// Metropolis temperature (kcal/mol); Vina uses 1.2.
    pub temperature: f64,
    /// Mutation amplitude in Å (Vina uses 2.0).
    pub mutation_amplitude: f64,
    /// Maximum number of MC steps per run; `None` selects Vina's heuristic.
    pub global_steps: Option<usize>,
    /// Number of local-search steps; `None` selects `(25 + n_atoms)/3`.
    pub local_steps: Option<usize>,
    /// Random seed; `0` means "draw one from the OS entropy pool".
    pub seed: u64,
    /// Use the island-model genetic algorithm instead of parallel MC.
    pub use_island_ga: bool,
    /// Local optimiser used by the island-model GA.
    pub local_search: LocalSearch,
    /// Number of GA islands (only for `use_island_ga`).
    pub islands: usize,
    /// Island population size.
    pub population: usize,
    /// Generations per island.
    pub generations: usize,
    /// Elites carried over per generation.
    pub elites: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            exhaustiveness: 8,
            num_poses: 9,
            min_rmsd: 1.0,
            max_evals: 0,
            temperature: 1.2,
            mutation_amplitude: 2.0,
            global_steps: None,
            local_steps: None,
            seed: 0,
            use_island_ga: false,
            local_search: LocalSearch::Bfgs,
            islands: 4,
            population: 32,
            generations: 20,
            elites: 2,
        }
    }
}

impl SearchParams {
    /// Vina's heuristic for the number of Monte-Carlo steps:
    /// `70 · 3 · (50 + n_atoms + 10·n_dof) / 2`.
    pub fn global_steps_for(&self, num_movable_atoms: usize, num_dof: usize) -> usize {
        match self.global_steps {
            Some(n) => n.max(1),
            None => {
                let heuristic = num_movable_atoms + 10 * num_dof;
                (70 * 3 * (50 + heuristic) / 2).max(1)
            }
        }
    }

    /// Vina's heuristic for the number of BFGS steps: `(25 + n_atoms)/3`.
    pub fn local_steps_for(&self, num_movable_atoms: usize) -> usize {
        match self.local_steps {
            Some(n) => n.max(1),
            None => ((25 + num_movable_atoms) / 3).max(1),
        }
    }

    /// Number of iterations for the local optimiser selected by `method`.
    ///
    /// The Vina heuristic (`(25 + n_atoms)/3` BFGS steps) is far too small for
    /// a derivative-free search, which needs one evaluation per degree of
    /// freedom and sweep; AutoDock 4 instead runs a fixed budget per local
    /// search. An explicit `local_steps` always wins, so callers stay in
    /// control.
    pub fn local_steps_for_method(&self, num_movable_atoms: usize, method: LocalSearch) -> usize {
        match method {
            LocalSearch::Bfgs => self.local_steps_for(num_movable_atoms),
            LocalSearch::SolisWets => match self.local_steps {
                Some(n) => n.max(1),
                None => SOLIS_LOCAL_ITERATIONS,
            },
        }
    }

    /// Resolve a zero seed into an entropy-derived one.
    pub fn effective_seed(&self) -> u64 {
        if self.seed == 0 {
            crate::rng::auto_seed()
        } else {
            self.seed
        }
    }
}

// ---------------------------------------------------------------------------
// BFGS
// ---------------------------------------------------------------------------

/// Symmetric matrix stored as a packed upper triangle.
#[derive(Debug, Clone)]
struct SymMatrix {
    n: usize,
    data: Vec<f64>,
}

impl SymMatrix {
    fn identity(n: usize) -> SymMatrix {
        let mut m = SymMatrix {
            n,
            data: vec![0.0; n * (n + 1) / 2],
        };
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i <= j { (i, j) } else { (j, i) };
        i * (2 * self.n - i + 1) / 2 + (j - i)
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.index(i, j)]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: f64) {
        let k = self.index(i, j);
        self.data[k] = v;
    }

    #[inline]
    fn add(&mut self, i: usize, j: usize, v: f64) {
        let k = self.index(i, j);
        self.data[k] += v;
    }

    /// `out = -H · v`.
    fn minus_mat_vec(&self, v: &[f64], out: &mut [f64]) {
        for i in 0..self.n {
            let mut sum = 0.0;
            for j in 0..self.n {
                sum += self.get(i, j) * v[j];
            }
            out[i] = -sum;
        }
    }

    fn set_diagonal(&mut self, x: f64) {
        for i in 0..self.n {
            self.set(i, i, x);
        }
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The BFGS inverse-Hessian update, ported from Vina's `bfgs.h` (Apache-2.0).
fn bfgs_update(h: &mut SymMatrix, p: &[f64], y: &[f64], alpha: f64) -> bool {
    let n = h.n;
    let yp = dot(y, p);
    if alpha * yp < EPSILON {
        return false;
    }
    let mut minus_hy = vec![0.0; n];
    h.minus_mat_vec(y, &mut minus_hy);
    let yhy = -dot(y, &minus_hy);
    let r = 1.0 / (alpha * yp);
    for i in 0..n {
        for j in i..n {
            let v = alpha * r * (minus_hy[i] * p[j] + minus_hy[j] * p[i])
                + alpha * alpha * (r * r * yhy + r) * p[i] * p[j];
            h.add(i, j, v);
        }
    }
    true
}

/// Result of a local optimisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizeResult {
    /// Final energy.
    pub energy: f64,
    /// Number of energy evaluations consumed.
    pub evals: u64,
    /// Number of BFGS iterations performed.
    pub steps: usize,
}

/// BFGS minimisation of `f` over the degree-of-freedom vector.
///
/// `conf` is updated in place; on failure to improve, the original
/// conformation is restored (Vina's `f0 <= f_orig` guard).
pub fn bfgs<F>(f: F, conf: &mut Conf, layout: &DofLayout, max_steps: usize) -> OptimizeResult
where
    F: FnMut(&Conf, &mut [f64]) -> f64,
{
    bfgs_with_cancel(f, conf, layout, max_steps, None)
}

/// [`bfgs`] with cooperative cancellation.
///
/// The token is polled once per BFGS iteration; an abort keeps the best point
/// found so far (the method is monotone), so a cancelled refinement still
/// returns a usable conformation.
pub fn bfgs_with_cancel<F>(
    mut f: F,
    conf: &mut Conf,
    layout: &DofLayout,
    max_steps: usize,
    cancel: Option<&CancelToken>,
) -> OptimizeResult
where
    F: FnMut(&Conf, &mut [f64]) -> f64,
{
    let n = layout.num_floats();
    if n == 0 {
        let mut g: Vec<f64> = Vec::new();
        let e = f(conf, &mut g);
        return OptimizeResult {
            energy: e,
            evals: 1,
            steps: 0,
        };
    }

    let mut h = SymMatrix::identity(n);
    let mut g = vec![0.0; n];
    let mut g_new = vec![0.0; n];
    let mut p = vec![0.0; n];

    let f0 = f(conf, &mut g);
    let mut evals = 1u64;

    let f_orig = f0;
    let x_orig = conf.clone();

    let mut f_cur = f0;
    let mut x_new = conf.clone();
    let mut steps = 0usize;

    'optimize: for step in 0..max_steps {
        // Blocks while the token is paused and returns `true` on cancellation,
        // so a pause freezes the refinement at an iteration boundary instead of
        // letting the current line search run to completion.
        if stopped(cancel) {
            break;
        }
        steps = step + 1;
        h.minus_mat_vec(&g, &mut p);

        // --- Armijo backtracking line search (Vina's `line_search`) ---
        const C0: f64 = 0.0001;
        const MAX_TRIALS: usize = 10;
        const MULTIPLIER: f64 = 0.5;
        let pg = dot(&p, &g);
        let mut alpha = 1.0;
        let mut f1 = f_cur;
        for _ in 0..MAX_TRIALS {
            // `x_new` is only valid after a completed trial, so an abort must
            // leave the whole iteration: breaking out of the line search alone
            // and then running the update below would write a stale (worse)
            // point back into `conf` and break the monotonicity guarantee.
            if abort_requested(cancel) {
                break 'optimize;
            }
            x_new = conf.clone();
            x_new.increment_flat(layout, &p, alpha);
            f1 = f(&x_new, &mut g_new);
            evals += 1;
            if f1 - f_cur < C0 * alpha * pg {
                break;
            }
            alpha *= MULTIPLIER;
        }

        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = g_new[i] - g[i];
        }
        f_cur = f1;
        *conf = std::mem::replace(&mut x_new, conf.clone());

        if !(dot(&g, &g).sqrt() >= 1e-5) {
            break;
        }
        g.copy_from_slice(&g_new);

        if step == 0 {
            let yy = dot(&y, &y);
            if yy.abs() > EPSILON {
                h.set_diagonal(alpha * dot(&y, &p) / yy);
            }
        }
        let _ = bfgs_update(&mut h, &p, &y, alpha);
    }

    if !(f_cur <= f_orig) {
        *conf = x_orig;
        f_cur = f_orig;
    }

    OptimizeResult {
        energy: f_cur,
        evals,
        steps,
    }
}

// ---------------------------------------------------------------------------
// Pose container
// ---------------------------------------------------------------------------

/// Root-mean-square deviation between two heavy-atom coordinate sets that
/// share an atom ordering.
pub fn rmsd_same_order(a: &[DVec3], b: &[DVec3]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = (0..n).map(|i| (a[i] - b[i]).length_squared()).sum();
    (sum / n as f64).sqrt()
}

/// Index and RMSD of the pose closest to `coords`.
fn find_closest(coords: &[DVec3], out: &[Pose]) -> (usize, f64) {
    let mut best = (usize::MAX, MAX_F);
    for (i, p) in out.iter().enumerate() {
        let r = rmsd_same_order(coords, &p.coords);
        if r < best.1 {
            best = (i, r);
        }
    }
    best
}

/// Insert a pose into the bounded, RMSD-deduplicated output container.
///
/// Ported from Vina's `add_to_output_container`: a pose within `min_rmsd` of an
/// existing one replaces it when it is better; otherwise it is appended while
/// there is room; otherwise the current worst entry is replaced when the new
/// pose is better than it.
pub fn add_to_output_container(out: &mut Vec<Pose>, pose: Pose, min_rmsd: f64, max_size: usize) {
    let (idx, rmsd) = find_closest(&pose.coords, out);
    if idx < out.len() && rmsd < min_rmsd {
        if pose.energy < out[idx].energy {
            out[idx] = pose;
        }
    } else if out.len() < max_size {
        out.push(pose);
    } else if let Some(last) = out.last_mut() {
        if pose.energy < last.energy {
            *last = pose;
        }
    }
    out.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(Ordering::Equal));
}

/// Merge `src` into `out`.
pub fn merge_containers(out: &mut Vec<Pose>, src: Vec<Pose>, min_rmsd: f64, max_size: usize) {
    for p in src {
        add_to_output_container(out, p, min_rmsd, max_size);
    }
    out.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(Ordering::Equal));
    out.truncate(max_size);
}

// ---------------------------------------------------------------------------
// Monte-Carlo iterated local search
// ---------------------------------------------------------------------------

/// Metropolis acceptance test.
#[inline]
pub fn metropolis_accept(old_f: f64, new_f: f64, temperature: f64, rng: &mut Rng) -> bool {
    if new_f < old_f {
        return true;
    }
    let p = ((old_f - new_f) / temperature).exp();
    rng.next_f64() < p
}

/// Build a [`Pose`] from the model's current state.
fn make_pose<M: EnergyModel>(model: &mut M, conf: &Conf) -> Pose {
    let components = model.score(conf, Caps::authentic());
    Pose {
        conf: conf.clone(),
        energy: components.total,
        lower_bound: 0.0,
        upper_bound: 0.0,
        components,
        coords: model.ligand_coords(),
        in_box: model.within_box(0.0),
    }
}

/// A starting conformation for a fresh MC run.
fn random_start(layout: &DofLayout, corner1: DVec3, corner2: DVec3, rng: &mut Rng) -> Conf {
    let mut c = Conf {
        ligands: layout
            .ligands
            .iter()
            .map(|&n| LigandConf::null(DVec3::ZERO, n))
            .collect(),
        flex: layout.flex.iter().map(|&n| vec![0.0; n]).collect(),
    };
    c.randomize(corner1, corner2, rng);
    c
}

/// One Monte-Carlo / iterated-local-search run.
///
/// Returns the local minima it discovered, sorted by energy.
pub fn run_monte_carlo<M: EnergyModel>(
    model: &mut M,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    rng: &mut Rng,
) -> Vec<Pose> {
    run_monte_carlo_with_cancel(model, corner1, corner2, params, rng, None)
}

/// [`run_monte_carlo`] with cooperative cancellation.
///
/// The token is polled at every global step (and inside every local search), so
/// an abort returns the best poses found so far within milliseconds. To honour
/// "a cancelled run always yields a valid pose", a run that is stopped before
/// its container filled up is finalised with the best point it had reached.
pub fn run_monte_carlo_with_cancel<M: EnergyModel>(
    model: &mut M,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    rng: &mut Rng,
    cancel: Option<&CancelToken>,
) -> Vec<Pose> {
    let layout = model.layout().clone();
    let n_atoms = model.num_movable_atoms();
    let n_dof = layout.num_floats();
    let global_steps = params.global_steps_for(n_atoms, n_dof);
    let local_steps = params.local_steps_for(n_atoms);
    let gyration = vec![model.gyration_radius()];

    let hunt = Caps::hunt();
    let authentic = Caps::authentic();
    let mut out: Vec<Pose> = Vec::with_capacity(params.num_poses);
    let mut evals: u64 = 0;
    let mut best_e = MAX_F;

    let mut current = random_start(&layout, corner1, corner2, rng);
    let res0 = bfgs_with_cancel(
        |conf, g| model.eval_deriv(conf, hunt, g),
        &mut current,
        &layout,
        local_steps,
        cancel,
    );
    evals += res0.evals;
    let mut current_e = res0.energy;
    let mut best_conf = current.clone();
    let mut best_conf_e = current_e;

    for step in 0..global_steps {
        // Pausing here is what makes `pause()` mean "does not advance": the
        // worker blocks on the token instead of burning a core.
        if stopped(cancel) {
            break;
        }
        if params.max_evals > 0 && evals > params.max_evals {
            break;
        }
        let mut cand = current.clone();
        mutate_conf(&mut cand, params.mutation_amplitude, &gyration, rng);

        let res = bfgs_with_cancel(
            |conf, g| model.eval_deriv(conf, hunt, g),
            &mut cand,
            &layout,
            local_steps,
            cancel,
        );
        evals += res.evals;

        if step == 0 || metropolis_accept(current_e, res.energy, params.temperature, rng) {
            current = cand;
            current_e = res.energy;
            if current_e < best_conf_e {
                best_conf_e = current_e;
                best_conf = current.clone();
            }

            if current_e < best_e || out.len() < params.num_poses {
                // Re-minimise with the loose caps so the stored pose is
                // directly comparable with the reported energies.
                let res2 = bfgs_with_cancel(
                    |conf, g| model.eval_deriv(conf, authentic, g),
                    &mut current,
                    &layout,
                    local_steps,
                    cancel,
                );
                evals += res2.evals;
                let pose = make_pose(model, &current);
                if pose.energy < best_e {
                    best_e = pose.energy;
                }
                add_to_output_container(&mut out, pose, params.min_rmsd, params.num_poses);
            }
        }
    }

    if out.is_empty() {
        // Cancelled (or budget-exhausted) before the container ever filled:
        // report the best point reached so that the caller always gets a pose.
        let mut fallback = best_conf;
        let _ = bfgs_with_cancel(
            |conf, g| model.eval_deriv(conf, authentic, g),
            &mut fallback,
            &layout,
            local_steps,
            None,
        );
        let pose = make_pose(model, &fallback);
        add_to_output_container(&mut out, pose, params.min_rmsd, params.num_poses);
    }

    out.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(Ordering::Equal));
    out
}

/// Run `exhaustiveness` independent Monte-Carlo searches in parallel.
pub fn parallel_monte_carlo<M, F>(
    make_model: F,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    seed: u64,
) -> Vec<Pose>
where
    M: EnergyModel,
    F: Fn(usize) -> M + Sync + Send,
{
    parallel_monte_carlo_with_cancel(make_model, corner1, corner2, params, seed, None)
}

/// [`parallel_monte_carlo`] with cooperative cancellation.
///
/// Every worker polls the same token, so one `cancel()` stops the whole pool;
/// each worker returns the poses it had found by then, which are merged exactly
/// as in the uncancelled case.
pub fn parallel_monte_carlo_with_cancel<M, F>(
    make_model: F,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    seed: u64,
    cancel: Option<&CancelToken>,
) -> Vec<Pose>
where
    M: EnergyModel,
    F: Fn(usize) -> M + Sync + Send,
{
    let root = Rng::new(seed);
    let tasks = params.exhaustiveness.max(1);

    let results: Vec<Vec<Pose>> = (0..tasks)
        .into_par_iter()
        .map(|i| {
            let mut model = make_model(i);
            let mut rng = root.spawn(i as u64 + 1);
            run_monte_carlo_with_cancel(&mut model, corner1, corner2, params, &mut rng, cancel)
        })
        .collect();

    // Vina merges the per-task containers with a relaxed 2 Å dedup cutoff so
    // that near-duplicate minima from different starts are not all reported.
    let mut out: Vec<Pose> = Vec::new();
    for r in results {
        merge_containers(&mut out, r, params.min_rmsd.max(2.0), params.num_poses);
    }
    out.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(Ordering::Equal));
    out.truncate(params.num_poses);
    out
}

// ---------------------------------------------------------------------------
// Island-model Lamarckian genetic algorithm
// ---------------------------------------------------------------------------

/// Island-model Lamarckian genetic algorithm.
///
/// Each island evolves a population of conformations with binary-tournament
/// selection, uniform crossover in torsion space and mutation, followed by a
/// **Lamarckian** local search: the optimised phenotype is written back
/// into the chromosome, exactly as in AutoDock 4's LGA. The best individuals
/// from every island are collected each generation, so the algorithm reports
/// the same kind of pose container as the Monte-Carlo search.
///
/// [`SearchParams::local_search`] selects the local optimiser: [`LocalSearch::Bfgs`]
/// (the Vina-flavoured variant) or [`LocalSearch::SolisWets`], which is what
/// AutoDock 4 actually runs.
pub fn island_lga<M, F>(
    make_model: F,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    seed: u64,
) -> Vec<Pose>
where
    M: EnergyModel,
    F: Fn(usize) -> M + Sync + Send,
{
    island_lga_with_cancel(make_model, corner1, corner2, params, seed, None)
}

/// Relax one candidate with the configured local optimiser.
#[allow(clippy::too_many_arguments)]
fn local_optimize<M: EnergyModel>(
    model: &mut M,
    conf: &mut Conf,
    layout: &DofLayout,
    caps: Caps,
    steps: usize,
    method: LocalSearch,
    gyration_radius: f64,
    cancel: Option<&CancelToken>,
) -> f64 {
    match method {
        LocalSearch::Bfgs => {
            bfgs_with_cancel(|c, g| model.eval_deriv(c, caps, g), conf, layout, steps, cancel).energy
        }
        LocalSearch::SolisWets => {
            let sw_params = SolisWetsParams::default().with_iterations(steps);
            solis_wets(
                |c| model.eval(c, caps),
                conf,
                layout,
                gyration_radius,
                &sw_params,
                cancel,
            )
            .energy
        }
    }
}

/// [`island_lga`] with cooperative cancellation.
///
/// Every island polls the token at each generation, and every local search
/// polls it at each step, so `cancel()` stops the whole pool promptly. Islands
/// that were interrupted still contribute the poses they had already found.
pub fn island_lga_with_cancel<M, F>(
    make_model: F,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    seed: u64,
    cancel: Option<&CancelToken>,
) -> Vec<Pose>
where
    M: EnergyModel,
    F: Fn(usize) -> M + Sync + Send,
{
    let n_islands = params.islands.max(1);
    let root = Rng::new(seed);
    let method = params.local_search;

    let island_results: Vec<Vec<Pose>> = (0..n_islands)
        .into_par_iter()
        .map(|island| {
            let mut model = make_model(island);
            let mut rng = root.spawn(0x9E37_79B9 ^ (island as u64 + 1));
            let layout = model.layout().clone();
            let local_steps = params
                .local_steps_for_method(model.num_movable_atoms(), method);
            let gyration_scalar = model.gyration_radius();
            let gyration = vec![gyration_scalar];
            let authentic = Caps::authentic();
            let hunt = Caps::hunt();

            let pop_size = params.population.max(4);
            let mut population: Vec<(Conf, f64)> = Vec::with_capacity(pop_size);
            for _ in 0..pop_size {
                if abort_requested(cancel) {
                    break;
                }
                let c = random_start(&layout, corner1, corner2, &mut rng);
                let e = model.eval(&c, hunt);
                population.push((c, e));
            }
            if population.is_empty() {
                let c = random_start(&layout, corner1, corner2, &mut rng);
                let e = model.eval(&c, hunt);
                population.push((c, e));
            }
            population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            let mut out: Vec<Pose> = Vec::new();

            'generations: for gen in 0..params.generations.max(1) {
                // Pausing blocks here, so a paused GA really stops evolving.
                if stopped(cancel) {
                    break;
                }
                // Lamarckian local search: the whole population on the first
                // generation, the better half afterwards.
                let improve = if gen == 0 {
                    pop_size
                } else {
                    (pop_size / 2).max(1)
                };
                for ind in population.iter_mut().take(improve) {
                    if abort_requested(cancel) {
                        break 'generations;
                    }
                    ind.1 = local_optimize(
                        &mut model,
                        &mut ind.0,
                        &layout,
                        hunt,
                        local_steps,
                        method,
                        gyration_scalar,
                        cancel,
                    );
                }
                population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                // Record the island's best individual with the final caps.
                let mut best = population[0].0.clone();
                let _ = local_optimize(
                    &mut model,
                    &mut best,
                    &layout,
                    authentic,
                    local_steps,
                    method,
                    gyration_scalar,
                    cancel,
                );
                let pose = make_pose(&mut model, &best);
                add_to_output_container(&mut out, pose, params.min_rmsd, params.num_poses);

                // --- next generation ---
                let mut next: Vec<(Conf, f64)> = Vec::with_capacity(pop_size);
                for elite in population.iter().take(params.elites.clamp(1, pop_size)) {
                    next.push(elite.clone());
                }
                while next.len() < pop_size {
                    if abort_requested(cancel) {
                        break;
                    }
                    let a = tournament(&population, &mut rng);
                    let b = tournament(&population, &mut rng);
                    let mut child = crossover(&population[a].0, &population[b].0, &mut rng);
                    mutate_conf(
                        &mut child,
                        params.mutation_amplitude * 0.5,
                        &gyration,
                        &mut rng,
                    );
                    let e = model.eval(&child, hunt);
                    next.push((child, e));
                }
                while next.len() < pop_size {
                    next.push(population[next.len() % population.len()].clone());
                }
                next.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                population = next;
            }

            if out.is_empty() {
                // Interrupted before the first generation completed: report the
                // best member of the initial population so the caller still
                // gets a valid pose.
                population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                let mut best = population[0].0.clone();
                let _ = local_optimize(
                    &mut model,
                    &mut best,
                    &layout,
                    authentic,
                    local_steps,
                    method,
                    gyration_scalar,
                    None,
                );
                let pose = make_pose(&mut model, &best);
                add_to_output_container(&mut out, pose, params.min_rmsd, params.num_poses);
            }

            out
        })
        .collect();

    let mut out: Vec<Pose> = Vec::new();
    for r in island_results {
        merge_containers(&mut out, r, params.min_rmsd.max(2.0), params.num_poses);
    }
    out.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(Ordering::Equal));
    out.truncate(params.num_poses);
    out
}

/// Binary tournament selection; returns the winning index.
fn tournament(population: &[(Conf, f64)], rng: &mut Rng) -> usize {
    let n = population.len();
    let a = rng.int(0, n as i64 - 1) as usize;
    let b = rng.int(0, n as i64 - 1) as usize;
    if population[a].1 <= population[b].1 {
        a
    } else {
        b
    }
}

/// Uniform crossover: each degree of freedom is inherited from either parent.
fn crossover(a: &Conf, b: &Conf, rng: &mut Rng) -> Conf {
    let mut child = a.clone();
    for (li, lig) in child.ligands.iter_mut().enumerate() {
        if rng.next_f64() < 0.5 {
            lig.position = b.ligands[li].position;
            lig.orientation = b.ligands[li].orientation;
        }
        for (k, t) in lig.torsions.iter_mut().enumerate() {
            if rng.next_f64() < 0.5 {
                *t = b.ligands[li].torsions[k];
            }
        }
    }
    for (fi, flex) in child.flex.iter_mut().enumerate() {
        for (k, t) in flex.iter_mut().enumerate() {
            if rng.next_f64() < 0.5 {
                *t = b.flex[fi][k];
            }
        }
    }
    child
}

/// Run whichever global search the parameters select.
pub fn run_search<M, F>(
    make_model: F,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    seed: u64,
) -> Vec<Pose>
where
    M: EnergyModel,
    F: Fn(usize) -> M + Sync + Send,
{
    run_search_with_cancel(make_model, corner1, corner2, params, seed, None)
}

/// [`run_search`] with cooperative cancellation.
pub fn run_search_with_cancel<M, F>(
    make_model: F,
    corner1: DVec3,
    corner2: DVec3,
    params: &SearchParams,
    seed: u64,
    cancel: Option<&CancelToken>,
) -> Vec<Pose>
where
    M: EnergyModel,
    F: Fn(usize) -> M + Sync + Send,
{
    if params.use_island_ga {
        island_lga_with_cancel(make_model, corner1, corner2, params, seed, cancel)
    } else {
        parallel_monte_carlo_with_cancel(make_model, corner1, corner2, params, seed, cancel)
    }
}

/// A pair list, re-exported for callers that build their own.
pub type PairList = Vec<Pair>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial quadratic energy surface over the rigid-body + torsion
    /// degrees of freedom, used to validate the optimiser in isolation.
    ///
    /// The optional evaluation counter and artificial delay turn it into a
    /// "slow" surface, which is what makes the cancellation tests deterministic:
    /// the search is guaranteed to still be running when the token is flipped.
    #[derive(Debug, Clone)]
    struct Toy {
        layout: DofLayout,
        target: DVec3,
        tors_target: Vec<f64>,
        evals: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
        delay: std::time::Duration,
    }

    impl EnergyModel for Toy {
        fn layout(&self) -> &DofLayout {
            &self.layout
        }
        fn num_movable_atoms(&self) -> usize {
            5
        }
        fn gyration_radius(&self) -> f64 {
            2.0
        }
        fn eval_deriv(&mut self, conf: &Conf, _caps: Caps, grad: &mut [f64]) -> f64 {
            let (e, g) = self.value(conf);
            grad.copy_from_slice(&g);
            e
        }
        fn eval(&mut self, conf: &Conf, _caps: Caps) -> f64 {
            self.value(conf).0
        }
        fn score(&mut self, conf: &Conf, _caps: Caps) -> ScoreComponents {
            let e = self.value(conf).0;
            ScoreComponents {
                total: e,
                inter: e,
                ..Default::default()
            }
        }
        fn ligand_coords(&self) -> Vec<DVec3> {
            Vec::new()
        }
        fn ligand_atoms(&self) -> Vec<Atom> {
            Vec::new()
        }
        fn within_box(&self, _margin: f64) -> bool {
            true
        }
    }

    impl Toy {
        fn value(&self, conf: &Conf) -> (f64, Vec<f64>) {
            if let Some(c) = &self.evals {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            let n = self.layout.num_floats();
            let mut g = vec![0.0; n];
            let lig = &conf.ligands[0];
            let d = lig.position - self.target;
            let mut e = d.length_squared();
            g[0] = 2.0 * d.x;
            g[1] = 2.0 * d.y;
            g[2] = 2.0 * d.z;
            for (k, t) in lig.torsions.iter().enumerate() {
                let dt = *t - self.tors_target[k];
                e += 0.5 * dt * dt;
                g[6 + k] = dt;
            }
            (e, g)
        }
    }

    fn toy() -> Toy {
        Toy {
            layout: DofLayout {
                ligands: vec![2],
                flex: vec![],
            },
            target: DVec3::new(1.5, -2.0, 0.75),
            tors_target: vec![0.4, -1.1],
            evals: None,
            delay: std::time::Duration::ZERO,
        }
    }

    /// The same surface, instrumented so that cancellation can be observed.
    fn slow_toy(
        evals: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        delay: std::time::Duration,
    ) -> Toy {
        Toy {
            evals: Some(evals),
            delay,
            ..toy()
        }
    }

    fn eval_counter() -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
        std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0))
    }

    fn counter_value(c: &std::sync::Arc<std::sync::atomic::AtomicUsize>) -> usize {
        c.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    fn bfgs_converges_on_a_quadratic() {
        let layout = DofLayout {
            ligands: vec![2],
            flex: vec![],
        };
        let mut m = toy();
        let mut conf = Conf {
            ligands: vec![LigandConf::null(DVec3::ZERO, 2)],
            flex: vec![],
        };
        let res = bfgs(
            |c, g| m.eval_deriv(c, Caps::hunt(), g),
            &mut conf,
            &layout,
            50,
        );
        assert!(res.energy < 1e-8, "energy {}", res.energy);
        assert!((conf.ligands[0].position - DVec3::new(1.5, -2.0, 0.75)).length() < 1e-4);
        assert!((conf.ligands[0].torsions[0] - 0.4).abs() < 1e-4);
        assert!((conf.ligands[0].torsions[1] + 1.1).abs() < 1e-4);
    }

    #[test]
    fn bfgs_restores_the_start_when_no_step_improves() {
        let layout = DofLayout {
            ligands: vec![0],
            flex: vec![],
        };
        let start = Conf {
            ligands: vec![LigandConf::null(DVec3::new(5.0, 5.0, 5.0), 0)],
            flex: vec![],
        };
        let before = start.clone();
        let mut conf = start;
        let res = bfgs(
            |_c, g| {
                for v in g.iter_mut() {
                    *v = 0.0;
                }
                1.0
            },
            &mut conf,
            &layout,
            10,
        );
        assert_eq!(conf, before);
        assert!((res.energy - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pose_container_deduplicates_and_sorts() {
        let mut out = Vec::new();
        let mk = |e: f64, x: f64| Pose {
            energy: e,
            coords: vec![DVec3::new(x, 0.0, 0.0)],
            ..Pose::empty()
        };
        add_to_output_container(&mut out, mk(-5.0, 0.0), 1.0, 3);
        add_to_output_container(&mut out, mk(-5.5, 0.1), 1.0, 3); // duplicate, better
        assert_eq!(out.len(), 1);
        assert!((out[0].energy - -5.5).abs() < 1e-12);
        add_to_output_container(&mut out, mk(-3.0, 5.0), 1.0, 3);
        add_to_output_container(&mut out, mk(-4.0, 10.0), 1.0, 3);
        assert_eq!(out.len(), 3);
        assert!((out[0].energy - -5.5).abs() < 1e-12);
        assert!((out[1].energy - -4.0).abs() < 1e-12);
        assert!((out[2].energy - -3.0).abs() < 1e-12);
        add_to_output_container(&mut out, mk(1.0, 20.0), 1.0, 3);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|p| p.energy < 0.0));
    }

    #[test]
    fn metropolis_prefers_lower_energy_and_rejects_huge_penalties() {
        let mut rng = Rng::new(3);
        assert!(metropolis_accept(0.0, -1.0, 1.2, &mut rng));
        let mut accepted_high = 0;
        for _ in 0..10_000 {
            if metropolis_accept(0.0, 100.0, 1.2, &mut rng) {
                accepted_high += 1;
            }
        }
        assert_eq!(accepted_high, 0);
    }

    #[test]
    fn rmsd_helper_is_symmetric() {
        let a = vec![DVec3::ZERO, DVec3::X];
        let b = vec![DVec3::new(0.0, 3.0, 0.0), DVec3::new(1.0, 3.0, 0.0)];
        assert!((rmsd_same_order(&a, &b) - 3.0).abs() < 1e-12);
        assert!((rmsd_same_order(&b, &a) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn monte_carlo_finds_the_minimum_of_a_quadratic() {
        let params = SearchParams {
            exhaustiveness: 2,
            num_poses: 3,
            global_steps: Some(60),
            local_steps: Some(40),
            seed: 12345,
            min_rmsd: 0.1,
            ..Default::default()
        };
        let models = || toy();
        let poses = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            12345,
        );
        assert!(!poses.is_empty());
        assert!(poses[0].energy < 1e-3, "best energy {}", poses[0].energy);
    }

    // -----------------------------------------------------------------------
    // Local optimiser selection
    // -----------------------------------------------------------------------

    #[test]
    fn local_search_identifiers_round_trip() {
        assert_eq!(LocalSearch::parse("bfgs"), Some(LocalSearch::Bfgs));
        assert_eq!(LocalSearch::parse("L-BFGS"), Some(LocalSearch::Bfgs));
        assert_eq!(
            LocalSearch::parse("solis_wets"),
            Some(LocalSearch::SolisWets)
        );
        assert_eq!(
            LocalSearch::parse("Solis-Wets"),
            Some(LocalSearch::SolisWets)
        );
        assert_eq!(LocalSearch::parse("nonsense"), None);
        assert_eq!(LocalSearch::default(), LocalSearch::Bfgs);
        assert_eq!(LocalSearch::SolisWets.name(), "solis_wets");
    }

    #[test]
    fn solis_wets_gets_a_larger_default_step_budget_than_bfgs() {
        let params = SearchParams::default();
        let bfgs = params.local_steps_for_method(20, LocalSearch::Bfgs);
        let sw = params.local_steps_for_method(20, LocalSearch::SolisWets);
        assert_eq!(bfgs, (25 + 20) / 3);
        assert_eq!(sw, SOLIS_LOCAL_ITERATIONS);
        // An explicit budget always wins.
        let explicit = SearchParams {
            local_steps: Some(7),
            ..Default::default()
        };
        assert_eq!(explicit.local_steps_for_method(20, LocalSearch::Bfgs), 7);
        assert_eq!(
            explicit.local_steps_for_method(20, LocalSearch::SolisWets),
            7
        );
    }

    /// The AutoDock 4 flavoured GA: island model + Solis-Wets local search.
    #[test]
    fn island_lga_with_solis_wets_converges() {
        let params = SearchParams {
            use_island_ga: true,
            local_search: LocalSearch::SolisWets,
            islands: 2,
            population: 8,
            generations: 6,
            elites: 2,
            num_poses: 3,
            min_rmsd: 0.1,
            seed: 4242,
            ..Default::default()
        };
        let models = || toy();
        let poses = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            4242,
        );
        assert!(!poses.is_empty());
        assert!(
            poses[0].energy < 1e-2,
            "Solis-Wets LGA stalled at {}",
            poses[0].energy
        );
    }

    #[test]
    fn island_lga_with_solis_wets_is_reproducible_for_a_fixed_seed() {
        let params = SearchParams {
            use_island_ga: true,
            local_search: LocalSearch::SolisWets,
            islands: 2,
            population: 6,
            generations: 3,
            num_poses: 2,
            min_rmsd: 0.1,
            seed: 99,
            ..Default::default()
        };
        let models = || toy();
        let a = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            params.effective_seed(),
        );
        let b = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            params.effective_seed(),
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.energy, y.energy);
            assert_eq!(x.conf, y.conf);
        }
    }

    /// Both local optimisers must reach the same minimum of the toy surface;
    /// that is the regression test that keeps Solis-Wets from silently
    /// degrading into "a worse BFGS" (or the other way round).
    #[test]
    fn both_local_optimisers_find_the_same_minimum() {
        let make = |method: LocalSearch| SearchParams {
            use_island_ga: true,
            local_search: method,
            islands: 2,
            population: 8,
            generations: 5,
            num_poses: 3,
            min_rmsd: 0.1,
            seed: 777,
            ..Default::default()
        };
        let models = || toy();
        let a = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &make(LocalSearch::Bfgs),
            777,
        );
        let b = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &make(LocalSearch::SolisWets),
            777,
        );
        assert!(a[0].energy < 1e-3, "bfgs: {}", a[0].energy);
        assert!(b[0].energy < 1e-3, "solis-wets: {}", b[0].energy);
    }

    // -----------------------------------------------------------------------
    // Cancellation
    // -----------------------------------------------------------------------

    fn long_params(seed: u64) -> SearchParams {
        SearchParams {
            exhaustiveness: 2,
            num_poses: 3,
            global_steps: Some(400),
            local_steps: Some(20),
            min_rmsd: 0.1,
            seed,
            ..Default::default()
        }
    }

    /// A cancelled run must come back within the promised 100 ms *and* still
    /// report a valid pose.
    #[test]
    fn a_cancelled_run_returns_promptly_with_a_valid_pose() {
        use crate::cancel::CancelToken;
        use std::time::{Duration, Instant};

        let token = CancelToken::new();
        let counter = eval_counter();
        let models = {
            let counter = std::sync::Arc::clone(&counter);
            move || slow_toy(std::sync::Arc::clone(&counter), Duration::from_micros(200))
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let canceller = {
            let token = token.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(40));
                token.cancel();
                let _ = tx.send(Instant::now());
            })
        };

        let poses = run_search_with_cancel(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &long_params(1),
            1,
            Some(&token),
        );
        let cancel_at = rx.recv().expect("the canceller must report back");
        let latency = cancel_at.elapsed();
        canceller.join().unwrap();

        assert!(!poses.is_empty(), "a cancelled run must still report a pose");
        assert!(
            poses[0].energy.is_finite(),
            "the fallback pose has a bogus energy"
        );
        assert!(
            latency < Duration::from_millis(100),
            "cancellation took {latency:?}"
        );
    }

    /// A paused run must stop advancing and pick up again on resume.
    #[test]
    fn a_paused_run_does_not_advance_and_resumes() {
        use crate::cancel::CancelToken;
        use std::time::Duration;

        let token = CancelToken::new();
        let counter = eval_counter();
        let models = {
            let counter = std::sync::Arc::clone(&counter);
            move || slow_toy(std::sync::Arc::clone(&counter), Duration::from_micros(200))
        };

        let runner = {
            let token = token.clone();
            std::thread::spawn(move || {
                run_search_with_cancel(
                    |_| models(),
                    DVec3::splat(-4.0),
                    DVec3::splat(4.0),
                    &long_params(2),
                    2,
                    Some(&token),
                )
            })
        };

        // Let it make some progress, then freeze it.
        std::thread::sleep(Duration::from_millis(30));
        token.pause();

        // A local-search iteration that was already in flight when `pause()`
        // was called is allowed to finish, so settle first and only then
        // require the counter to be frozen. `SLACK` bounds that in-flight work:
        // the same window advances by ~300 evaluations when the search is not
        // paused, so this is still a strict test of the pause.
        const SLACK: usize = 24;
        std::thread::sleep(Duration::from_millis(50));
        let before = counter_value(&counter);
        std::thread::sleep(Duration::from_millis(60));
        let during = counter_value(&counter);
        assert!(
            during - before <= SLACK,
            "a paused search kept advancing: {} evaluations in 60 ms",
            during - before
        );

        token.resume();
        std::thread::sleep(Duration::from_millis(40));
        let after = counter_value(&counter);
        assert!(
            after > during,
            "the resumed search did not advance ({during} -> {after})"
        );

        token.cancel();
        let poses = runner.join().expect("the worker must not panic");
        assert!(!poses.is_empty());
    }

    #[test]
    fn an_already_cancelled_token_still_yields_a_pose() {
        let token = CancelToken::new();
        token.cancel();
        let models = || toy();
        let poses = run_search_with_cancel(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &long_params(3),
            3,
            Some(&token),
        );
        assert_eq!(poses.len(), 1, "the fallback pose must be reported");
        assert!(poses[0].energy.is_finite());
    }

    #[test]
    fn an_already_cancelled_island_ga_still_yields_a_pose() {
        let token = CancelToken::new();
        token.cancel();
        let params = SearchParams {
            use_island_ga: true,
            local_search: LocalSearch::SolisWets,
            islands: 2,
            population: 6,
            generations: 5,
            num_poses: 2,
            seed: 5,
            ..Default::default()
        };
        let models = || toy();
        let poses = run_search_with_cancel(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            5,
            Some(&token),
        );
        assert!(!poses.is_empty());
        assert!(poses[0].energy.is_finite());
    }

    /// The cancellable entry point must not change the uncancelled result.
    #[test]
    fn passing_a_running_token_does_not_change_the_result() {
        let token = CancelToken::new();
        let params = SearchParams {
            exhaustiveness: 2,
            num_poses: 3,
            global_steps: Some(40),
            local_steps: Some(30),
            min_rmsd: 0.1,
            seed: 31337,
            ..Default::default()
        };
        let models = || toy();
        let a = run_search(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            31337,
        );
        let b = run_search_with_cancel(
            |_| models(),
            DVec3::splat(-4.0),
            DVec3::splat(4.0),
            &params,
            31337,
            Some(&token),
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.energy, y.energy);
            assert_eq!(x.conf, y.conf);
        }
    }
}

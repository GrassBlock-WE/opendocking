// SPDX-License-Identifier: GPL-3.0-or-later
//! The high-level docking orchestrator: the OpenDocking equivalent of
//! AutoDock Vina's `Vina` object.
//!
//! A [`System`] binds together
//!
//! * the rigid receptor,
//! * the ligand's kinematic tree ([`crate::kinematics::MovableModel`]),
//! * the force field ([`crate::scoring::ScoringFunction`]),
//! * the affinity grid used during the search, and
//! * the exact (non-grid) scorer used for refinement and final scoring.
//!
//! [`Docking`] wraps a [`System`] with the file-level workflow: read a receptor
//! and a ligand PDBQT, set up the search box, run the search, re-score and
//! write poses.
//!
//! # Energy bookkeeping
//!
//! Following the reference implementation, every evaluation splits the energy
//! into
//!
//! ```text
//! E_inter    ligand-receptor
//! E_intra    ligand internal (pairs at least four bonds apart)
//! E_unbound  the same ligand internal energy in isolation
//! E_tors     conformation-independent torsional penalty
//!
//! affinity = E_tors + (E_inter + E_intra - E_unbound)
//! ```
//!
//! For the Vina and Vinardo force fields the whole sum is divided by
//! `1 + rot * N_tors` (so `E_tors` is negative); for AD4 the torsional term is
//! additive. With a rigid receptor `E_intra == E_unbound`, so the ligand's
//! internal strain cancels exactly, which is the reference behaviour.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::atom::XsType;
use crate::cancel::{CancelState, CancelToken};
use crate::error::{DockError, Result};
use crate::io::pdbqt::{parse_ligand_pdbqt, parse_receptor_pdbqt, RecordAtom};
use crate::kinematics::{build_ligand_tree, Conf, DofLayout, LigandConf, MovableModel, TopologyNode};
use crate::math::{DVec3, MAX_F};
use crate::molecule::{assign_types, perceive_bonds, Atom, Pair};
use crate::scoring::grid::{grid_type, AffinityGrid, GridBox, GridDim, GRID_TYPES};
use crate::scoring::noncache::{eval_pairs, NonCache};
use crate::scoring::{
    is_closure_clash, is_glue_pair, is_unmatched_closure_dummy, make_scoring_function, num_tors,
    ScoreComponents, ScoringFunction, SfChoice, Weights,
};
use crate::search::{
    bfgs_with_cancel, run_search_with_cancel, Caps, EnergyModel, Pose, SearchParams,
};

/// Out-of-box penalty slope. AutoDock Vina uses `1e6`; it only has to be large
/// compared with any physical interaction energy.
pub const DEFAULT_SLOPE: f64 = 1e6;

/// Options controlling a docking run.
#[derive(Clone)]
pub struct DockOptions {
    /// Force field.
    pub sf_choice: SfChoice,
    /// Override the default term weights.
    pub weights: Option<Weights>,
    /// The search box.
    pub box_: GridBox,
    /// Use the pre-computed affinity grid during the search (much faster).
    /// Forced off for force fields that cannot be gridded, i.e. AD4.
    pub use_grid: bool,
    /// Re-minimise every reported pose with the exact pairwise scorer.
    pub refine: bool,
    /// Search parameters.
    pub search: SearchParams,
    /// Out-of-box penalty slope.
    pub slope: f64,
    /// Poses within this energy window of the best one are reported.
    pub energy_range: f64,
}

impl std::fmt::Debug for DockOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DockOptions")
            .field("sf_choice", &self.sf_choice)
            .field("box", &self.box_)
            .field("use_grid", &self.use_grid)
            .field("refine", &self.refine)
            .field("search", &self.search)
            .finish()
    }
}

impl Default for DockOptions {
    fn default() -> Self {
        DockOptions {
            sf_choice: SfChoice::Vina,
            weights: None,
            box_: GridBox::default(),
            use_grid: true,
            refine: true,
            search: SearchParams::default(),
            slope: DEFAULT_SLOPE,
            energy_range: 3.0,
        }
    }
}

impl DockOptions {
    /// A box centred on `center` with the given edge lengths.
    pub fn with_box(center: DVec3, size: DVec3, spacing: f64) -> DockOptions {
        DockOptions {
            box_: GridBox::new(center, size, spacing),
            ..Default::default()
        }
    }
}

/// Immutable scoring context shared by every search thread.
#[derive(Debug, Clone)]
pub struct Shared {
    /// The rigid receptor atoms.
    pub receptor: Vec<Atom>,
    /// Exact scorer over the receptor.
    pub noncache: NonCache,
    /// Optional affinity grid.
    pub grid: Option<AffinityGrid>,
    /// The force field.
    pub sf: Arc<dyn ScoringFunction>,
    /// Ligand atom templates (typing, names and *initial* coordinates).
    pub ligand_atoms: Vec<Atom>,
    /// Ligand atom line templates for PDBQT output.
    pub ligand_records: Vec<RecordAtom>,
    /// Ligand topology, for PDBQT output.
    pub top: TopologyNode,
    /// Rotatable bonds, for the torsional penalty.
    pub rotors: Vec<(Option<usize>, usize)>,
    /// Vina's `num_tors` value.
    pub num_tors: f64,
    /// Interaction pairs inside the ligand.
    pub intra_pairs: Vec<Pair>,
    /// Macrocycle closure pairs (long-range linear attraction).
    pub glue_pairs: Vec<Pair>,
    /// Search box, as requested.
    pub box_: GridBox,
    /// Search box, as rounded by the grid dimensions.
    pub dims: [GridDim; 3],
    /// Out-of-box penalty slope.
    pub slope: f64,
    /// Use the grid during the search.
    pub use_grid: bool,
}

impl Shared {
    /// Number of torsion segments in the ligand topology.
    pub fn num_torsions(&self) -> usize {
        fn count(n: &TopologyNode) -> usize {
            n.children.iter().map(|c| 1 + count(c)).sum()
        }
        count(&self.top)
    }
}

/// A docking system: shared scoring context plus per-thread mutable state.
#[derive(Clone)]
pub struct System {
    /// Shared, immutable context.
    pub shared: Arc<Shared>,
    /// Movable atoms, coordinates and kinematic trees.
    pub movable: MovableModel,
    /// Force the exact (non-grid) interaction energy. The search runs with the
    /// grid for speed; refinement and final scoring set this to `true`.
    pub force_exact: bool,
}

impl std::fmt::Debug for System {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("System")
            .field("receptor_atoms", &self.shared.receptor.len())
            .field("movable_atoms", &self.movable.num_atoms())
            .field("num_tors", &self.shared.num_tors)
            .field("intra_pairs", &self.shared.intra_pairs.len())
            .field("force_exact", &self.force_exact)
            .finish()
    }
}

impl System {
    /// Build an independent clone for a search thread.
    pub fn fork(template: &System) -> System {
        System {
            shared: Arc::clone(&template.shared),
            movable: template.movable.clone(),
            force_exact: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Assign each movable atom the id of the rigid frame that owns it.
fn frame_ids(tree: &crate::kinematics::KinematicTree) -> Vec<usize> {
    fn walk(node: &crate::kinematics::Node, id: &mut usize, out: &mut [usize], cur: usize) {
        for i in node.atoms.0..node.atoms.1 {
            out[i] = cur;
        }
        for c in &node.children {
            *id += 1;
            walk(c, id, out, *id);
        }
    }
    let mut out = vec![usize::MAX; tree.local.len()];
    let mut id = 0usize;
    walk(&tree.root, &mut id, &mut out, 0);
    out
}

/// Enumerate the intra-ligand interaction pairs.
///
/// Mirrors Vina's `model::initialize_pairs`: two atoms interact when they live
/// in *different* rigid frames (so their relative geometry can change) and are
/// not connected within three bonds. Macrocycle closure pairs are diverted to
/// their own list, exactly as the reference implementation does.
pub fn ligand_pairs(
    atoms: &[Atom],
    frames: &[usize],
    ligand_range: (usize, usize),
    xs_typed: bool,
) -> (Vec<Pair>, Vec<Pair>) {
    let mut intra = Vec::new();
    let mut glue = Vec::new();
    for i in ligand_range.0..ligand_range.1 {
        let bonded: Vec<usize> = crate::molecule::bonded_within(atoms, i, 3);
        for j in (i + 1)..ligand_range.1 {
            if frames[i] == frames[j] {
                continue; // rigid relative to each other
            }
            if bonded.contains(&j) {
                continue; // 1-2, 1-3 or 1-4
            }
            if is_closure_clash(atoms, i, j) || is_unmatched_closure_dummy(&atoms[i], &atoms[j]) {
                continue;
            }
            let t1 = atoms[i].xs;
            let t2 = atoms[j].xs;
            if xs_typed && (t1 == XsType::W || t2 == XsType::W) {
                // Untyped hydrogen: invisible to the X-Score force field.
                continue;
            }
            if is_glue_pair(&atoms[i], &atoms[j]) {
                glue.push(Pair::new(i, j));
            } else {
                intra.push(Pair::new(i, j));
            }
        }
    }
    (intra, glue)
}

/// Build a [`System`] from receptor and ligand PDBQT text.
pub fn build_system(receptor_text: &str, ligand_text: &str, options: &DockOptions) -> Result<System> {
    let receptor_rec = parse_receptor_pdbqt(receptor_text)?;
    let ligand = parse_ligand_pdbqt(ligand_text)?;

    let sf = make_scoring_function(options.sf_choice, options.weights.clone());

    // --- receptor typing ---------------------------------------------------
    // A PDBQT file carries the AutoDock type but *not* the connectivity, and the
    // X-Score typing that the Vina force field needs is derived from the bond
    // graph: a backbone amide nitrogen is only a donor because it carries a
    // polar hydrogen, and a hydroxyl oxygen is only a donor for the same reason.
    // Skipping this step silently removes every receptor-side hydrogen bond,
    // which is why it happens before anything reads `atom.xs`.
    let receptor_atoms = {
        let mut atoms = receptor_rec.atoms.clone();
        perceive_bonds(&mut atoms);
        assign_types(&mut atoms);
        atoms
    };

    // --- movable model -----------------------------------------------------
    let n_movable = ligand.atoms.len();
    let coords: Vec<DVec3> = ligand.atoms.iter().map(|a| a.coords).collect();
    let (tree, _root_idx) = build_ligand_tree(&ligand.top, &coords, n_movable);

    let mut movable_atoms = ligand.atoms.clone();
    perceive_bonds(&mut movable_atoms);
    assign_types(&mut movable_atoms);

    let layout = DofLayout {
        ligands: vec![tree.num_torsions],
        flex: Vec::new(),
    };
    let ligand_range = tree.atoms;
    let frames = frame_ids(&tree);
    let (intra_pairs, glue_pairs) =
        ligand_pairs(&movable_atoms, &frames, ligand_range, sf.is_xs_typed());
    let num_tors_value = num_tors(&movable_atoms, &ligand.rotors);

    let mut movable = MovableModel {
        atoms: movable_atoms,
        coords: coords.clone(),
        minus_forces: vec![DVec3::ZERO; n_movable],
        ligand: tree,
        ligand_pairs: intra_pairs.clone(),
        flex: Vec::new(),
        flex_pairs: Vec::new(),
        layout,
    };
    // Refresh the coordinates through the kinematics so the starting
    // conformation is exactly the input structure.
    let start = movable.initial_conf();
    movable.apply(&start);

    // --- box and grid ------------------------------------------------------
    let dims = options.box_.dims();
    let use_grid = options.use_grid && sf.is_grid_capable();

    let noncache = NonCache::new(receptor_atoms.clone(), sf.as_ref(), dims, options.slope);

    let grid = if use_grid {
        // Only the atom types the ligand actually presents need a map; that is
        // typically 6-10 instead of all 19.
        let mut wanted: Vec<XsType> = Vec::new();
        for a in &movable.atoms {
            if let Some(t) = grid_type(a.xs) {
                if !wanted.contains(&t) {
                    wanted.push(t);
                }
            }
        }
        if wanted.is_empty() {
            wanted.extend_from_slice(&GRID_TYPES);
        }
        let mut g = AffinityGrid::from_dims(dims, options.box_.spacing, options.slope);
        g.populate(&receptor_atoms, sf.as_ref(), &wanted);
        Some(g)
    } else {
        None
    };

    let shared = Shared {
        receptor: receptor_atoms,
        noncache,
        grid,
        sf,
        ligand_atoms: movable.atoms.clone(),
        ligand_records: ligand.records.clone(),
        top: ligand.top.clone(),
        rotors: ligand.rotors.clone(),
        num_tors: num_tors_value,
        intra_pairs,
        glue_pairs,
        box_: options.box_,
        dims,
        slope: options.slope,
        use_grid,
    };

    Ok(System {
        shared: Arc::new(shared),
        movable,
        force_exact: false,
    })
}

// ---------------------------------------------------------------------------
// Energy evaluation
// ---------------------------------------------------------------------------

impl System {
    /// `true` when the ligand lies inside the search box.
    pub fn is_inside_box(&self, margin: f64) -> bool {
        match (&self.shared.grid, self.force_exact) {
            (Some(g), false) => g.is_in_grid(&self.movable.atoms, &self.movable.coords, margin),
            _ => self
                .shared
                .noncache
                .within(&self.movable.atoms, &self.movable.coords, margin),
        }
    }

    /// The ligand-receptor energy (grid or exact, depending on the flags).
    fn inter_energy(&self, caps: Caps, forces: Option<&mut [DVec3]>) -> f64 {
        if self.shared.use_grid && !self.force_exact {
            if let Some(g) = &self.shared.grid {
                return match forces {
                    Some(f) => g.eval_deriv(&self.movable.atoms, &self.movable.coords, caps.grid, f),
                    None => g.eval(&self.movable.atoms, &self.movable.coords, caps.grid),
                };
            }
        }
        let n_lig = self.movable.atoms.len();
        self.shared.noncache.eval(
            self.shared.sf.as_ref(),
            &self.movable.atoms,
            &self.movable.coords,
            caps.grid,
            false,
            n_lig,
            forces,
        )
    }

    /// Full evaluation used by both the search and the final scoring.
    fn evaluate(
        &mut self,
        conf: &Conf,
        caps: Caps,
        mut gradient: Option<&mut [f64]>,
    ) -> ScoreComponents {
        self.movable.apply(conf);
        let want_grad = gradient.is_some();

        // The per-atom gradient buffer is moved out of the model so that the
        // scoring helpers can borrow the coordinates and the buffer at the same
        // time; it is put back before the DOF projection.
        let mut forces_buf = std::mem::take(&mut self.movable.minus_forces);
        if want_grad {
            for f in forces_buf.iter_mut() {
                *f = DVec3::ZERO;
            }
        }

        // 1. ligand-receptor: this *sets* the per-atom forces.
        let inter = {
            let forces: Option<&mut [DVec3]> = if want_grad {
                Some(&mut forces_buf[..])
            } else {
                None
            };
            self.inter_energy(caps, forces)
        };

        // 2. intra-ligand pairs: these *add* to the per-atom forces.
        let intra = eval_pairs(
            self.shared.sf.as_ref(),
            &self.movable.atoms,
            &self.movable.coords,
            &self.shared.intra_pairs,
            caps.intra,
            self.shared.sf.cutoff(),
            if want_grad {
                Some(&mut forces_buf[..])
            } else {
                None
            },
        );

        // 3. macrocycle closure pairs use the longer (maximum) cutoff.
        let glue = if self.shared.glue_pairs.is_empty() {
            0.0
        } else {
            eval_pairs(
                self.shared.sf.as_ref(),
                &self.movable.atoms,
                &self.movable.coords,
                &self.shared.glue_pairs,
                caps.pairs,
                self.shared.sf.max_cutoff(),
                if want_grad {
                    Some(&mut forces_buf[..])
                } else {
                    None
                },
            )
        };
        let intra = intra + glue;

        self.movable.minus_forces = forces_buf;
        if let Some(g) = gradient.as_deref_mut() {
            let dof = self.movable.dof_gradient();
            g.copy_from_slice(&dof);
        }

        let unbound = match self.shared.sf.choice() {
            SfChoice::Ad4 => 0.0,
            _ => intra,
        };
        let base = inter + intra - unbound;
        let total = match self.shared.sf.choice() {
            SfChoice::Ad4 => inter + self.shared.sf.conf_independent(0.0, self.shared.num_tors),
            _ => self.shared.sf.conf_independent(base, self.shared.num_tors),
        };
        ScoreComponents {
            total,
            inter,
            intra,
            conf_independent: total - base,
            unbound,
        }
    }

    /// The energy the search minimises: the undivided conformational energy.
    ///
    /// The torsional divisor depends only on the topology, so it cannot move the
    /// minimum; leaving it out of the objective matches the reference
    /// implementation exactly.
    fn objective(&self, c: &ScoreComponents) -> f64 {
        match self.shared.sf.choice() {
            SfChoice::Ad4 => c.inter + c.conf_independent,
            _ => c.inter + c.intra - c.unbound,
        }
    }

    /// Reported properties of a conformation.
    pub fn current_components(&mut self, conf: &Conf) -> ScoreComponents {
        self.evaluate(conf, Caps::authentic(), None)
    }

    /// The ligand's atoms with their current coordinates.
    pub fn current_ligand_atoms(&self) -> Vec<Atom> {
        self.movable.ligand_atoms()
    }

    /// Heavy-atom ligand coordinates (used for RMSD).
    pub fn current_ligand_coords(&self) -> Vec<DVec3> {
        let range = self.movable.ligand.atoms;
        (range.0..range.1)
            .filter(|&i| !self.movable.atoms[i].is_hydrogen())
            .map(|i| self.movable.coords[i])
            .collect()
    }
}

impl EnergyModel for System {
    fn layout(&self) -> &DofLayout {
        &self.movable.layout
    }

    fn num_movable_atoms(&self) -> usize {
        self.movable.atoms.len()
    }

    fn gyration_radius(&self) -> f64 {
        self.movable.ligand_gyration_radius()
    }

    fn eval_deriv(&mut self, conf: &Conf, caps: Caps, grad: &mut [f64]) -> f64 {
        let c = self.evaluate(conf, caps, Some(grad));
        self.objective(&c)
    }

    fn eval(&mut self, conf: &Conf, caps: Caps) -> f64 {
        let c = self.evaluate(conf, caps, None);
        self.objective(&c)
    }

    fn score(&mut self, conf: &Conf, caps: Caps) -> ScoreComponents {
        self.evaluate(conf, caps, None)
    }

    fn ligand_coords(&self) -> Vec<DVec3> {
        self.current_ligand_coords()
    }

    fn ligand_atoms(&self) -> Vec<Atom> {
        self.current_ligand_atoms()
    }

    fn within_box(&self, margin: f64) -> bool {
        self.is_inside_box(margin)
    }
}

// ---------------------------------------------------------------------------
// The user-facing object
// ---------------------------------------------------------------------------

/// The result of a docking run.
#[derive(Debug, Clone)]
pub struct DockResult {
    /// The reported poses, best first.
    pub poses: Vec<Pose>,
    /// The random seed actually used (never `0`).
    pub seed: u64,
    /// Approximate affinity-grid memory in megabytes (0 when unused).
    pub grid_mb: usize,
    /// Number of grid points (0 when unused).
    pub grid_points: usize,
    /// The `num_tors` value used by the torsional penalty.
    pub num_tors: f64,
    /// Number of movable atoms.
    pub num_movable_atoms: usize,
    /// Number of torsion degrees of freedom (excluding the rigid body).
    pub num_dof: usize,
    /// Whether the reported energies come from the exact pairwise scorer.
    pub exact: bool,
    /// `true` when the run was stopped by [`Docking::cancel`]; the poses are
    /// then the best ones found before the abort.
    pub cancelled: bool,
}

impl DockResult {
    /// The best pose, if any.
    pub fn best(&self) -> Option<&Pose> {
        self.poses.first()
    }

    /// Poses within `range` kcal/mol of the best one.
    pub fn within_energy_range(&self, range: f64) -> Vec<&Pose> {
        match self.best() {
            Some(b) => self
                .poses
                .iter()
                .filter(|p| p.energy <= b.energy + range)
                .collect(),
            None => Vec::new(),
        }
    }
}

/// A docking job.
#[derive(Debug)]
pub struct Docking {
    /// The system being docked.
    pub system: System,
    /// The options in force.
    pub options: DockOptions,
    /// The ligand's original PDBQT text.
    pub ligand_source: String,
    /// The receptor's original PDBQT text.
    pub receptor_source: String,
    /// The most recent result.
    pub result: Option<DockResult>,
    /// The pause / abort token [`Docking::run`] polls.
    pub cancel_token: CancelToken,
    /// `true` while a [`Docking::run`] call is executing.
    running: Arc<AtomicBool>,
}

/// Clears [`Docking::running`] even if the search panics.
struct RunningGuard(Arc<AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Docking {
    /// Prepare a docking job from PDBQT text.
    pub fn from_strings(
        receptor_pdbqt: &str,
        ligand_pdbqt: &str,
        options: DockOptions,
    ) -> Result<Docking> {
        let system = build_system(receptor_pdbqt, ligand_pdbqt, &options)?;
        Ok(Docking {
            system,
            options,
            ligand_source: ligand_pdbqt.to_string(),
            receptor_source: receptor_pdbqt.to_string(),
            result: None,
            cancel_token: CancelToken::new(),
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    // -----------------------------------------------------------------------
    // Run control (module D: 暂停 / 恢复 / 强行终止)
    // -----------------------------------------------------------------------

    /// Request an abort. Idempotent and safe to call from any thread; `run()`
    /// returns the best poses it had found within a few milliseconds.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Request a pause. A running search blocks at its next step boundary with
    /// a short sleep, so a paused run costs no CPU.
    pub fn pause(&self) {
        self.cancel_token.pause();
    }

    /// Resume a paused search. Never resurrects a cancelled one.
    pub fn resume(&self) {
        self.cancel_token.resume();
    }

    /// `true` while a search is executing and has not been cancelled.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire) && !self.cancel_token.is_cancelled()
    }

    /// The pause / abort state of the job.
    pub fn cancel_state(&self) -> CancelState {
        self.cancel_token.state()
    }

    /// Re-arm a cancelled job so that `run()` can be called again.
    pub fn reset_cancel(&self) -> bool {
        self.cancel_token.rearm()
    }
}

impl Docking {
    /// Re-centre the search box on the ligand's current position.
    pub fn center_box_on_ligand(&mut self, buffer: f64) {
        let range = self.system.movable.ligand.atoms;
        let atoms = &self.system.movable.atoms;
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        for i in range.0..range.1 {
            lo = lo.min(atoms[i].coords);
            hi = hi.max(atoms[i].coords);
        }
        let center = (lo + hi) * 0.5;
        let half = (hi - lo) * 0.5;
        self.options.box_ = GridBox::around(center, half, buffer, self.options.box_.spacing)
            .with_even_voxels(self.options.box_.force_even_voxels);
    }

    /// Score the ligand in its current (input) position.
    pub fn score(&mut self) -> ScoreComponents {
        let conf = self.system.movable.initial_conf();
        self.system.force_exact = self.options.refine;
        self.system.current_components(&conf)
    }

    /// Run the global search, refine and score the poses.
    pub fn run(&mut self) -> Result<DockResult> {
        // A fresh run starts from a clean slate: an abort that was requested
        // (and acted upon) by a previous run must not make this one a no-op.
        self.cancel_token.rearm();
        let token = self.cancel_token.clone();
        self.run_with_token(&token)
    }

    /// [`Docking::run`] against an explicit cancellation token.
    ///
    /// Unlike [`Docking::run`] this does **not** re-arm the token: a token that
    /// is already cancelled makes the call return the fallback pose
    /// immediately, which is what lets one token abort a whole batch of jobs
    /// (`dock_batch` shares a single token between every job, including the
    /// ones that have not started yet).
    ///
    /// The token may be flipped from another thread at any time; the search
    /// then unwinds within one step and the result reports the poses found so
    /// far, with [`DockResult::cancelled`] set.
    pub fn run_with_token(&mut self, token: &CancelToken) -> Result<DockResult> {
        self.running.store(true, Ordering::Release);
        let _guard = RunningGuard(Arc::clone(&self.running));

        // The box may have been changed after construction: rebuild the grid so
        // that the search and the final scores agree.
        self.rebuild_grid()?;

        let seed = self.options.search.effective_seed();
        let corner1 = self.options.box_.corner1();
        let corner2 = self.options.box_.corner2();

        let template = self.system.clone();
        let make = move |_i: usize| System::fork(&template);

        let mut poses = run_search_with_cancel(
            make,
            corner1,
            corner2,
            &self.options.search,
            seed,
            Some(token),
        );
        let cancelled = token.is_cancelled();
        if poses.is_empty() {
            let what = if cancelled {
                "the run was cancelled before any pose was found"
            } else {
                "the search produced no poses; check the grid box and the ligand"
            };
            return Err(DockError::Invalid(what.to_string()));
        }

        // --- refinement and final scoring use the exact scorer --------------
        self.system.force_exact = self.options.refine;
        let steps = self
            .options
            .search
            .local_steps_for_method(self.system.movable.atoms.len(), self.options.search.local_search);

        for p in poses.iter_mut() {
            if token.is_cancelled() {
                break;
            }
            if self.options.refine {
                refine_pose_with_cancel(&mut self.system, p, steps, Some(token));
            } else {
                let comps = self.system.score(&p.conf, Caps::authentic());
                p.energy = comps.total;
                p.components = comps;
                p.coords = self.system.ligand_coords();
                p.in_box = self.system.within_box(0.0);
            }
        }
        poses.retain(|p| p.energy < MAX_F / 2.0 && p.energy.is_finite());
        poses.sort_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        poses.truncate(self.options.search.num_poses);
        if poses.is_empty() {
            return Err(DockError::Invalid(
                "every candidate pose was rejected during refinement".to_string(),
            ));
        }

        // --- RMSD relative to the best pose ---------------------------------
        let best_atoms = pose_atoms(&self.system, &poses[0].conf);
        for p in poses.iter_mut() {
            let atoms = pose_atoms(&self.system, &p.conf);
            p.lower_bound = crate::molecule::rmsd_lower_bound(&atoms, &best_atoms);
            p.upper_bound = crate::molecule::rmsd_upper_bound(&atoms, &best_atoms);
        }

        let grid_mb = self
            .system
            .shared
            .grid
            .as_ref()
            .map(|g| g.memory_mb())
            .unwrap_or(0);
        let grid_points = self
            .system
            .shared
            .grid
            .as_ref()
            .map(|g| g.num_points())
            .unwrap_or(0);

        let result = DockResult {
            poses,
            seed,
            grid_mb,
            grid_points,
            num_tors: self.system.shared.num_tors,
            num_movable_atoms: self.system.movable.atoms.len(),
            num_dof: self.system.movable.layout.num_torsions_total(),
            exact: self.options.refine,
            cancelled: token.is_cancelled(),
        };
        self.result = Some(result.clone());
        Ok(result)
    }

    /// Rebuild the affinity grid for the current box.
    pub fn rebuild_grid(&mut self) -> Result<()> {
        let shared = Arc::make_mut(&mut self.system.shared);
        shared.box_ = self.options.box_;
        shared.dims = self.options.box_.dims();
        shared.slope = self.options.slope;
        shared.use_grid = self.options.use_grid && shared.sf.is_grid_capable();
        shared.noncache.dims = shared.dims;
        if !shared.use_grid {
            shared.grid = None;
            return Ok(());
        }
        let mut wanted: Vec<XsType> = Vec::new();
        for a in &self.system.movable.atoms {
            if let Some(t) = grid_type(a.xs) {
                if !wanted.contains(&t) {
                    wanted.push(t);
                }
            }
        }
        if wanted.is_empty() {
            wanted.extend_from_slice(&GRID_TYPES);
        }
        let mut g =
            AffinityGrid::from_dims(shared.dims, self.options.box_.spacing, self.options.slope);
        let receptor = shared.receptor.clone();
        let sf = Arc::clone(&shared.sf);
        g.populate(&receptor, sf.as_ref(), &wanted);
        shared.grid = Some(g);
        Ok(())
    }

    /// Render the poses as a multi-model PDBQT document.
    pub fn poses_pdbqt(&self, energy_range: f64) -> String {
        let Some(res) = &self.result else {
            return String::new();
        };
        let best = res.poses.first().map(|p| p.energy).unwrap_or(0.0);
        let mut out = String::new();
        let mut sys = self.system.clone();
        for (i, p) in res.poses.iter().enumerate() {
            if p.energy > best + energy_range {
                continue;
            }
            let remarks = vec![
                format!(
                    "VINA RESULT:    {:9.3}  {:9.3}  {:9.3}",
                    p.energy, p.lower_bound, p.upper_bound
                ),
                format!(
                    "INTER + INTRA:  {:12.3}",
                    p.components.inter + p.components.intra
                ),
                format!("INTER:          {:12.3}", p.components.inter),
                format!("INTRA:          {:12.3}", p.components.intra),
                format!("CONF_INDEPENDENT:{:12.3}", p.components.conf_independent),
                format!("UNBOUND:        {:12.3}", p.components.unbound),
                format!("OPEN DOCKING MODE {}", i + 1),
            ];
            sys.movable.apply(&p.conf);
            let range = sys.movable.ligand.atoms;
            let coords: Vec<DVec3> = (range.0..range.1).map(|k| sys.movable.coords[k]).collect();
            out.push_str(&crate::io::pdbqt::write_pose_pdbqt(
                &self.system.shared.ligand_records,
                &self.system.shared.top,
                &coords,
                &remarks,
                Some(i + 1),
            ));
        }
        out
    }
}

/// Re-minimise a pose with the exact (non-grid) scorer.
///
/// This is the post-search refinement of the reference implementation: the grid
/// is only an approximation of the true interaction energy, so every reported
/// pose is finally relaxed on the exact surface.
pub fn refine_pose(system: &mut System, pose: &mut Pose, steps: usize) -> f64 {
    refine_pose_with_cancel(system, pose, steps, None)
}

/// [`refine_pose`] with cooperative cancellation.
pub fn refine_pose_with_cancel(
    system: &mut System,
    pose: &mut Pose,
    steps: usize,
    cancel: Option<&CancelToken>,
) -> f64 {
    let layout = system.layout().clone();
    let authentic = Caps::authentic();
    let mut conf = pose.conf.clone();
    let _ = bfgs_with_cancel(
        |c, g| system.eval_deriv(c, authentic, g),
        &mut conf,
        &layout,
        steps,
        cancel,
    );
    let comps = system.score(&conf, authentic);
    pose.conf = conf;
    pose.energy = comps.total;
    pose.components = comps;
    pose.coords = system.ligand_coords();
    pose.in_box = system.within_box(0.0);
    pose.energy
}

/// Ligand atoms of a pose (used for RMSD reporting).
fn pose_atoms(system: &System, conf: &Conf) -> Vec<Atom> {
    let mut sys = system.clone();
    sys.movable.apply(conf);
    sys.movable.ligand_atoms()
}

/// Build a system from PDBQT files on disk.
pub fn docking_from_files(
    receptor_path: &std::path::Path,
    ligand_path: &std::path::Path,
    options: DockOptions,
) -> Result<Docking> {
    let receptor = std::fs::read_to_string(receptor_path)?;
    let ligand = std::fs::read_to_string(ligand_path)?;
    Docking::from_strings(&receptor, &ligand, options)
}

/// The ligand's input conformation (identity rotation, zero torsions).
pub fn initial_conf(system: &System) -> Conf {
    Conf {
        ligands: system
            .movable
            .layout
            .ligands
            .iter()
            .map(|&n| LigandConf::null(system.movable.ligand.root.origin, n))
            .collect(),
        flex: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Batch docking (high-throughput virtual screening)
// ---------------------------------------------------------------------------

/// One independent docking job of a batch.
///
/// Every job carries its own receptor and ligand text, so the jobs share no
/// state at all and can be spread over every core.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Caller-supplied identifier, echoed back in the outcome.
    pub label: String,
    /// Receptor PDBQT text.
    pub receptor: String,
    /// Ligand PDBQT text.
    pub ligand: String,
    /// Per-job options (box, force field, search parameters).
    pub options: DockOptions,
}

impl BatchJob {
    /// A job with the given label and inputs.
    pub fn new(label: impl Into<String>, receptor: impl Into<String>, ligand: impl Into<String>, options: DockOptions) -> BatchJob {
        BatchJob {
            label: label.into(),
            receptor: receptor.into(),
            ligand: ligand.into(),
            options,
        }
    }
}

/// The outcome of one [`BatchJob`].
#[derive(Debug, Clone)]
pub struct BatchOutcome {
    /// The job's label, echoed back.
    pub label: String,
    /// The docking result, when the job succeeded.
    pub result: Option<DockResult>,
    /// The failure message, when the job did not.
    pub error: Option<String>,
    /// `true` when the batch-wide token was cancelled while this job ran.
    pub cancelled: bool,
}

impl BatchOutcome {
    /// `true` when the job produced a result.
    pub fn is_ok(&self) -> bool {
        self.result.is_some()
    }

    /// The best affinity of the job, if it produced poses.
    pub fn best_affinity(&self) -> Option<f64> {
        self.result.as_ref().and_then(|r| r.best()).map(|p| p.energy)
    }
}

/// Run independent docking jobs in parallel with `rayon`.
///
/// * `threads` — `None` (or `0`) uses rayon's global pool; otherwise a private
///   pool with exactly that many workers is used, so a GUI can bound the batch
///   to the cores the user selected.
/// * `cancel` — one token shared by every job; cancelling it stops the whole
///   batch and each job reports the poses it had found.
///
/// The returned vector has exactly the same length and order as `jobs`.
pub fn dock_batch(
    jobs: Vec<BatchJob>,
    threads: Option<usize>,
    cancel: Option<&CancelToken>,
) -> Vec<BatchOutcome> {
    let run_one = |job: &BatchJob| -> BatchOutcome {
        let token = match cancel {
            Some(t) => t.clone(),
            None => CancelToken::new(),
        };
        let mut docking = match Docking::from_strings(&job.receptor, &job.ligand, job.options.clone())
        {
            Ok(d) => d,
            Err(e) => {
                return BatchOutcome {
                    label: job.label.clone(),
                    result: None,
                    error: Some(e.to_string()),
                    cancelled: token.is_cancelled(),
                }
            }
        };
        docking.cancel_token = token.clone();
        match docking.run_with_token(&token) {
            Ok(r) => BatchOutcome {
                label: job.label.clone(),
                cancelled: r.cancelled,
                result: Some(r),
                error: None,
            },
            Err(e) => BatchOutcome {
                label: job.label.clone(),
                result: None,
                error: Some(e.to_string()),
                cancelled: token.is_cancelled(),
            },
        }
    };

    let pool = threads
        .filter(|n| *n > 0)
        .and_then(|n| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok());
    match pool {
        Some(pool) => pool.install(|| jobs.par_iter().map(run_one).collect()),
        None => jobs.par_iter().map(run_one).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancel::CancelToken;
    use crate::math::DVec3;

    const LIGAND: &str = include_str!("../tests/data/erlotinib.pdbqt");
    const RECEPTOR: &str = include_str!("../tests/data/egfr.pdbqt");

    fn small_options() -> DockOptions {
        DockOptions {
            search: SearchParams {
                exhaustiveness: 2,
                num_poses: 3,
                global_steps: Some(10),
                local_steps: Some(5),
                seed: 1234,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn erlotinib_box() -> (DVec3, DVec3) {
        let lig = parse_ligand_pdbqt(LIGAND).expect("fixture");
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        for a in &lig.atoms {
            lo = lo.min(a.coords);
            hi = hi.max(a.coords);
        }
        ((lo + hi) * 0.5, (hi - lo) + DVec3::splat(12.0))
    }

    #[test]
    fn run_rearms_a_previous_abort() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        let mut d = Docking::from_strings(RECEPTOR, LIGAND, options).expect("build");
        d.cancel();
        assert!(d.cancel_token.is_cancelled());
        let r = d.run().expect("an aborted job must still be runnable");
        assert!(
            !r.cancelled,
            "run() must clear a stale abort before starting"
        );
        assert!(!r.poses.is_empty());
        assert!(!d.is_running());
    }

    /// The GUI's "abort" button: the worker thread is inside `run()`.
    #[test]
    fn a_cancelled_run_returns_what_it_has() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        options.search.exhaustiveness = 4;
        options.search.global_steps = Some(4000);
        let mut d = Docking::from_strings(RECEPTOR, LIGAND, options).expect("build");
        let token = d.cancel_token.clone();
        let worker = std::thread::spawn(move || d.run());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!worker.is_finished(), "the run finished far too quickly");
        token.cancel();
        let r = worker
            .join()
            .expect("the worker must not panic")
            .expect("a cancelled run still reports its poses");
        assert!(r.cancelled);
        assert!(!r.poses.is_empty());
        assert!(r.poses[0].energy.is_finite());
    }

    /// The GUI's "pause" button: the worker thread must freeze, not spin.
    #[test]
    fn a_paused_run_freezes_and_resumes() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        options.search.exhaustiveness = 1;
        options.search.global_steps = Some(4000);
        let mut d = Docking::from_strings(RECEPTOR, LIGAND, options).expect("build");
        let token = d.cancel_token.clone();
        assert!(token.pause());
        assert_eq!(d.cancel_state(), CancelState::Paused);
        let worker = std::thread::spawn(move || d.run());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            !worker.is_finished(),
            "a paused run must not complete its work"
        );
        assert!(token.resume());
        std::thread::sleep(std::time::Duration::from_millis(20));
        token.cancel();
        let r = worker
            .join()
            .expect("the worker must not panic")
            .expect("a resumed run reports its poses");
        assert!(!r.poses.is_empty());
        assert!(r.cancelled);
    }

    /// The AutoDock 4 flavoured pipeline end to end: island GA + Solis-Wets.
    #[test]
    fn the_lga_solis_pipeline_runs_end_to_end() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        options.search.use_island_ga = true;
        options.search.local_search = crate::search::LocalSearch::SolisWets;
        options.search.islands = 2;
        options.search.population = 6;
        options.search.generations = 2;
        options.search.local_steps = Some(40);
        let mut d = Docking::from_strings(RECEPTOR, LIGAND, options).expect("build");
        let r = d.run().expect("the Solis-Wets LGA must produce poses");
        assert!(!r.cancelled);
        assert!(!r.poses.is_empty());
        assert!(r.poses[0].energy.is_finite());
        assert!(r.poses[0].energy < 0.0, "affinity {}", r.poses[0].energy);
    }

    #[test]
    fn batch_runs_every_job_and_keeps_the_order() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        let jobs: Vec<BatchJob> = (0..4)
            .map(|i| {
                let mut o = options.clone();
                o.search.seed = 100 + i;
                BatchJob::new(format!("job-{i}"), RECEPTOR, LIGAND, o)
            })
            .collect();
        let out = dock_batch(jobs, Some(2), None);
        assert_eq!(out.len(), 4);
        for (i, o) in out.iter().enumerate() {
            assert_eq!(o.label, format!("job-{i}"));
            assert!(o.is_ok(), "job {i} failed: {:?}", o.error);
            assert!(o.best_affinity().is_some());
        }
    }

    #[test]
    fn a_broken_job_reports_an_error_without_killing_the_batch() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        let jobs = vec![
            BatchJob::new("good", RECEPTOR, LIGAND, options.clone()),
            BatchJob::new("bad", RECEPTOR, "not a pdbqt file at all", options.clone()),
        ];
        let out = dock_batch(jobs, None, None);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_ok());
        assert!(!out[1].is_ok());
        assert!(out[1].error.is_some());
    }

    #[test]
    fn a_cancelled_batch_still_returns_outcomes() {
        let (center, size) = erlotinib_box();
        let mut options = small_options();
        options.box_ = GridBox::new(center, size, 0.5);
        options.use_grid = false;
        let jobs: Vec<BatchJob> = (0..3)
            .map(|i| BatchJob::new(format!("j{i}"), RECEPTOR, LIGAND, options.clone()))
            .collect();
        let token = CancelToken::new();
        token.cancel();
        let out = dock_batch(jobs, None, Some(&token));
        assert_eq!(out.len(), 3);
        for o in &out {
            assert!(o.cancelled);
            assert!(o.is_ok(), "cancelled job {:?}", o.error);
            assert!(o.result.as_ref().map(|r| !r.poses.is_empty()).unwrap_or(false));
        }
    }
}

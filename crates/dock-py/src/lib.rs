// SPDX-License-Identifier: GPL-3.0-or-later
//! # dock-py — the Python binding layer of OpenDocking
//!
//! `dock-py` exposes the [`dock_core`] kernel to Python through PyO3, with a
//! NumPy-facing surface:
//!
//! * coordinates can be handed in and taken out as `numpy.ndarray` without a
//!   copy (`PyReadonlyArray2<f64>` borrows the buffer; `PyArray2::from_vec`
//!   hands ownership to NumPy),
//! * poses are exposed both as a plain list of Python dictionaries (ergonomic)
//!   and as packed `float64` arrays (fast),
//! * the whole docking object is `Send`, so Python threads can drive several
//!   systems concurrently.
//!
//! Nothing in this crate duplicates kernel logic: every entry point is a thin,
//! typed wrapper around `dock-core`.
//!
//! ## Run control (pause / resume / abort)
//!
//! [`PyDocking`] is a **frozen** `#[pyclass]`: every method takes `&self` and
//! the mutable state lives behind one `Mutex`. That is what makes the run
//! control usable the way a GUI needs it:
//!
//! ```python
//! worker = threading.Thread(target=d.run)
//! worker.start()
//! d.pause()          # blocks the search at its next step boundary
//! d.resume()
//! d.cancel()         # run() returns the best poses found so far
//! worker.join()
//! ```
//!
//! The pause / abort word itself is an `AtomicU8` shared with the search
//! workers, so `pause()`, `resume()` and `cancel()` never take the mutex and
//! can be called from any thread — including while `run()` (which *does* hold
//! the mutex and releases the GIL) is executing.

use std::sync::{Mutex, MutexGuard};

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyAny;
use pyo3::types::{PyDict, PyList};

use dock_core::cancel::CancelToken;
use dock_core::docking::{
    dock_batch as core_dock_batch, BatchJob, BatchOutcome, DockOptions, DockResult, Docking,
};
use dock_core::math::DVec3;
use dock_core::scoring::grid::GridBox;
use dock_core::scoring::{make_scoring_function, SfChoice, Weights};
use dock_core::search::{LocalSearch, SearchParams};

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn parse_sf(name: &str) -> PyResult<SfChoice> {
    SfChoice::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown scoring function {name:?}; expected one of vina, vinardo, ad4"
        ))
    })
}

/// The global search selected by the `search=` keyword.
///
/// * `monte_carlo` — Vina's iterated local search (BFGS local minimisation);
/// * `lga` — island-model Lamarckian GA with BFGS local minimisation;
/// * `lga_solis` — the same GA with AutoDock 4's Solis-Wets local search.
fn parse_search(name: &str) -> PyResult<(bool, LocalSearch)> {
    let key = name.trim().to_ascii_lowercase().replace('-', "_");
    match key.as_str() {
        "monte_carlo" | "mc" | "ils" | "iterated_local_search" => {
            Ok((false, LocalSearch::Bfgs))
        }
        "lga" | "ga" | "island" | "genetic" => Ok((true, LocalSearch::Bfgs)),
        "lga_solis" | "lga_solis_wets" | "lga_sw" | "solis" | "solis_wets" => {
            Ok((true, LocalSearch::SolisWets))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown search {other:?}; expected one of monte_carlo, lga, lga_solis"
        ))),
    }
}

/// The inverse of [`parse_search`], for the `Docking.search` getter.
fn search_name(use_island_ga: bool, local: LocalSearch) -> &'static str {
    match (use_island_ga, local) {
        (false, _) => "monte_carlo",
        (true, LocalSearch::Bfgs) => "lga",
        (true, LocalSearch::SolisWets) => "lga_solis",
    }
}

fn vec3(t: (f64, f64, f64)) -> DVec3 {
    DVec3::new(t.0, t.1, t.2)
}

/// A 3-D point or vector as Python sees it.
type Vec3 = (f64, f64, f64);

// ---------------------------------------------------------------------------
// Scoring-function introspection
// ---------------------------------------------------------------------------

/// The term weights of a force field.
#[pyclass(name = "Weights", module = "odock._odock")]
#[derive(Clone)]
pub struct PyWeights {
    inner: Weights,
}

#[pymethods]
impl PyWeights {
    /// The default weights of a force field.
    #[staticmethod]
    fn default_for(sf: &str) -> PyResult<PyWeights> {
        Ok(PyWeights {
            inner: Weights::default_for(parse_sf(sf)?),
        })
    }

    /// The six (Vina) / five (Vinardo, AD4) distance-dependent weights.
    #[getter]
    fn terms(&self) -> Vec<f64> {
        self.inner.terms.clone()
    }

    /// The torsional weight.
    #[getter]
    fn rot(&self) -> f64 {
        self.inner.rot
    }

    fn __repr__(&self) -> String {
        format!(
            "Weights(terms={:?}, rot={})",
            self.inner.terms, self.inner.rot
        )
    }
}

/// Properties of the force field that is in use.
#[pyclass(name = "ScoringFunction", module = "odock._odock")]
pub struct PyScoringFunction {
    choice: SfChoice,
    cutoff: f64,
    max_cutoff: f64,
    num_terms: usize,
    grid_capable: bool,
    terms: Vec<f64>,
    rot: f64,
}

#[pymethods]
impl PyScoringFunction {
    #[new]
    #[pyo3(signature = (choice = "vina"))]
    fn new(choice: &str) -> PyResult<Self> {
        let c = parse_sf(choice)?;
        let sf = make_scoring_function(c, None);
        Ok(PyScoringFunction {
            choice: c,
            cutoff: sf.cutoff(),
            max_cutoff: sf.max_cutoff(),
            num_terms: sf.num_terms(),
            grid_capable: sf.is_grid_capable(),
            terms: sf.weights().terms.clone(),
            rot: sf.weights().rot,
        })
    }

    /// Force-field name.
    #[getter]
    fn name(&self) -> String {
        self.choice.name().to_string()
    }

    /// Interaction cutoff in Angstrom.
    #[getter]
    fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Largest cutoff of any term (macrocycle glue) in Angstrom.
    #[getter]
    fn max_cutoff(&self) -> f64 {
        self.max_cutoff
    }

    /// Number of distance-dependent terms.
    #[getter]
    fn num_terms(&self) -> usize {
        self.num_terms
    }

    /// Whether an affinity grid can be built for this force field.
    #[getter]
    fn grid_capable(&self) -> bool {
        self.grid_capable
    }

    /// The term weights.
    #[getter]
    fn terms(&self) -> Vec<f64> {
        self.terms.clone()
    }

    /// The torsional weight.
    #[getter]
    fn rot(&self) -> f64 {
        self.rot
    }

    /// Evaluate the pairwise energy of two typed atoms at distance `r`.
    ///
    /// `t1` / `t2` are X-Score type indices (see `odock.XS_TYPES`).
    #[pyo3(signature = (t1, t2, r))]
    fn pair_energy(&self, t1: usize, t2: usize, r: f64) -> PyResult<f64> {
        use dock_core::atom::XsType;
        if t1 >= XsType::COUNT || t2 >= XsType::COUNT {
            return Err(PyValueError::new_err("X-Score type index out of range"));
        }
        let sf = make_scoring_function(self.choice, None);
        Ok(sf.pair_energy_xs(XsType::from_index(t1), XsType::from_index(t2), r))
    }

    fn __repr__(&self) -> String {
        format!(
            "ScoringFunction({:?}, cutoff={}, terms={})",
            self.choice.name(),
            self.cutoff,
            self.num_terms
        )
    }
}

// ---------------------------------------------------------------------------
// The docking object
// ---------------------------------------------------------------------------

/// A docking job.
///
/// ```python
/// from odock import _odock as core
/// d = core.Docking(receptor_pdbqt, ligand_pdbqt, center=(1.0, 2.0, 3.0),
///                  size=(22.5, 22.5, 22.5), exhaustiveness=8, seed=42,
///                  search="lga_solis")
/// result = d.run()
/// print(result["poses"][0]["affinity"])
/// ```
///
/// `search` accepts `"monte_carlo"` (the default), `"lga"` and `"lga_solis"`.
#[pyclass(name = "Docking", module = "odock._odock", frozen)]
pub struct PyDocking {
    inner: Mutex<Docking>,
    /// A clone of the job's pause / abort word.
    ///
    /// Kept outside the mutex on purpose: `cancel()`, `pause()`, `resume()`
    /// and `is_running()` must work while `run()` is executing and holding the
    /// mutex, which is exactly the situation the GUI's run monitor is in.
    token: CancelToken,
}

impl PyDocking {
    fn lock(&self) -> PyResult<MutexGuard<'_, Docking>> {
        self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("the docking job was left in an inconsistent state by a panic")
        })
    }
}

#[pymethods]
impl PyDocking {
    #[new]
    #[pyo3(signature = (
        receptor_pdbqt,
        ligand_pdbqt,
        *,
        center = (0.0, 0.0, 0.0),
        size = (22.5, 22.5, 22.5),
        spacing = 0.375,
        scoring = "vina",
        exhaustiveness = 8,
        num_poses = 9,
        seed = 0,
        use_grid = true,
        refine = true,
        min_rmsd = 1.0,
        energy_range = 3.0,
        use_island_ga = false,
        islands = 4,
        population = 32,
        generations = 20,
        global_steps = None,
        local_steps = None,
        search = "monte_carlo",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        receptor_pdbqt: &str,
        ligand_pdbqt: &str,
        center: (f64, f64, f64),
        size: (f64, f64, f64),
        spacing: f64,
        scoring: &str,
        exhaustiveness: usize,
        num_poses: usize,
        seed: u64,
        use_grid: bool,
        refine: bool,
        min_rmsd: f64,
        energy_range: f64,
        use_island_ga: bool,
        islands: usize,
        population: usize,
        generations: usize,
        global_steps: Option<usize>,
        local_steps: Option<usize>,
        search: &str,
    ) -> PyResult<Self> {
        if spacing <= 0.0 {
            return Err(PyValueError::new_err("spacing must be positive"));
        }
        if size.0 <= 0.0 || size.1 <= 0.0 || size.2 <= 0.0 {
            return Err(PyValueError::new_err("box size must be positive"));
        }
        let (mut island, local_search) = parse_search(search)?;
        // The legacy boolean still selects the LGA; an explicit `search=` wins.
        if use_island_ga {
            island = true;
        }
        let mut options = DockOptions {
            sf_choice: parse_sf(scoring)?,
            box_: GridBox::new(vec3(center), vec3(size), spacing),
            use_grid,
            refine,
            energy_range,
            ..Default::default()
        };
        options.search = SearchParams {
            exhaustiveness,
            num_poses,
            min_rmsd,
            seed,
            use_island_ga: island,
            local_search,
            islands,
            population,
            generations,
            global_steps,
            local_steps,
            ..Default::default()
        };
        let inner = Docking::from_strings(receptor_pdbqt, ligand_pdbqt, options).map_err(err)?;
        let token = inner.cancel_token.clone();
        Ok(PyDocking {
            inner: Mutex::new(inner),
            token,
        })
    }

    // -----------------------------------------------------------------------
    // Run control
    // -----------------------------------------------------------------------

    /// Request an abort.
    ///
    /// `run()` returns the best poses found so far (usually within a few
    /// milliseconds); the result dictionary's `cancelled` key is then `True`.
    /// Safe to call from any thread, including while `run()` is executing.
    fn cancel(&self) {
        self.token.cancel();
    }

    /// Request a pause.
    ///
    /// A running search blocks at its next step boundary (sleeping, not
    /// spinning) until `resume()` or `cancel()` is called.
    fn pause(&self) {
        self.token.pause();
    }

    /// Resume a paused search.
    fn resume(&self) {
        self.token.resume();
    }

    /// `True` while a search is executing and has not been cancelled.
    fn is_running(&self) -> bool {
        match self.inner.try_lock() {
            Ok(d) => d.is_running(),
            // The mutex is held by `run()`: a search is definitely in flight.
            Err(_) => !self.token.is_cancelled(),
        }
    }

    /// The pause / abort state: `"running"`, `"paused"` or `"cancelled"`.
    fn cancel_state(&self) -> &'static str {
        self.token.state().name()
    }

    /// Re-arm a cancelled job so that `run()` can be called again.
    ///
    /// `run()` already does this automatically; the explicit call is for code
    /// that wants to clear the abort without starting a run.
    fn reset_cancel(&self) {
        self.token.rearm();
    }

    // -----------------------------------------------------------------------
    // Setup
    // -----------------------------------------------------------------------

    /// Re-centre the box on the ligand's input position, padded by `buffer` Å.
    #[pyo3(signature = (buffer = 5.0))]
    fn center_box_on_ligand(&self, buffer: f64) -> PyResult<()> {
        self.lock()?.center_box_on_ligand(buffer);
        Ok(())
    }

    /// The search box actually in use, as `(center, size)`.
    fn search_box(&self) -> PyResult<(Vec3, Vec3)> {
        let d = self.lock()?;
        let b = d.options.box_;
        Ok((
            (b.center.x, b.center.y, b.center.z),
            (b.size.x, b.size.y, b.size.z),
        ))
    }

    /// The force field in use (`"vina"`, `"vinardo"` or `"ad4"`).
    #[getter]
    fn scoring(&self) -> PyResult<String> {
        Ok(self.lock()?.options.sf_choice.name().to_string())
    }

    /// The global search in use (`"monte_carlo"`, `"lga"` or `"lga_solis"`).
    #[getter]
    fn search(&self) -> PyResult<String> {
        let d = self.lock()?;
        Ok(search_name(
            d.options.search.use_island_ga,
            d.options.search.local_search,
        )
        .to_string())
    }

    // -----------------------------------------------------------------------
    // Energy evaluation
    // -----------------------------------------------------------------------

    /// Score the ligand at its input position.
    ///
    /// Returns a dictionary with `affinity`, `inter`, `intra`,
    /// `conf_independent` and `unbound`.
    fn score(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let c = self.lock()?.score();
        components_to_dict(py, &c)
    }

    /// Run the docking search.
    ///
    /// Returns a dictionary with `poses`, `seed`, `grid_mb`, `grid_points`,
    /// `num_tors`, `num_movable_atoms`, `num_dof`, `exact` and `cancelled`.
    ///
    /// The GIL is released for the duration of the search, so other Python
    /// threads can poll `is_running()` and call `pause()` / `resume()` /
    /// `cancel()` while it runs.
    fn run(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // The mutex guard must be created *inside* the detached closure: a
        // `MutexGuard` is not `Send` and therefore cannot cross the boundary.
        let result = py
            .detach(|| -> std::result::Result<DockResult, String> {
                let mut guard = self.inner.lock().map_err(|_| {
                    "the docking job was left in an inconsistent state by a panic".to_string()
                })?;
                guard.run().map_err(|e| e.to_string())
            })
            .map_err(PyValueError::new_err)?;
        result_to_dict(py, &result)
    }

    /// Score the ligand at its input position, without taking the run mutex.
    ///
    /// Deprecated alias kept for callers of the original API.
    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.lock()?.result {
            Some(r) => result_to_dict(py, r),
            None => Ok(py.None()),
        }
    }

    /// Poses as a PDBQT document (multi-model).
    #[pyo3(signature = (energy_range = 3.0))]
    fn poses_pdbqt(&self, energy_range: f64) -> PyResult<String> {
        Ok(self.lock()?.poses_pdbqt(energy_range))
    }

    /// Heavy-atom coordinates of every pose, packed as an
    /// `(n_poses, n_atoms, 3)` `float64` array.
    fn poses_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.lock()?;
        let Some(res) = &d.result else {
            return Err(PyRuntimeError::new_err(
                "run() must be called before poses_coords()",
            ));
        };
        let n_poses = res.poses.len();
        let n_atoms = res.poses.first().map(|p| p.coords.len()).unwrap_or(0);
        let mut flat = Vec::with_capacity(n_poses * n_atoms * 3);
        for p in &res.poses {
            for c in &p.coords {
                flat.push(c.x);
                flat.push(c.y);
                flat.push(c.z);
            }
        }
        PyArray1::from_vec(py, flat)
            .reshape([n_poses * n_atoms, 3])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// All ligand atom coordinates of a pose, in the kernel's internal order
    /// (the same order as `ligand_atom_names()`), hydrogens included.
    fn pose_coords<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.lock()?;
        let Some(res) = &d.result else {
            return Err(PyRuntimeError::new_err("run() must be called first"));
        };
        let p = res
            .poses
            .get(index)
            .ok_or_else(|| PyValueError::new_err(format!("pose {index} does not exist")))?;
        let mut sys = d.system.clone();
        sys.movable.apply(&p.conf);
        let range = sys.movable.ligand.atoms;
        let n = range.1 - range.0;
        let mut flat = Vec::with_capacity(n * 3);
        for i in range.0..range.1 {
            let c = sys.movable.coords[i];
            flat.push(c.x);
            flat.push(c.y);
            flat.push(c.z);
        }
        PyArray1::from_vec(py, flat)
            .reshape([n, 3])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Ligand atom names in the kernel's internal (flat) order.
    ///
    /// `pose_coords(i)[k]` is the position of `ligand_atom_names()[k]`.
    fn ligand_atom_names(&self) -> PyResult<Vec<String>> {
        let d = self.lock()?;
        let range = d.system.movable.ligand.atoms;
        Ok((range.0..range.1)
            .map(|i| d.system.movable.atoms[i].name.clone())
            .collect())
    }

    /// Ligand element symbols in the kernel's internal order.
    fn ligand_atom_elements(&self) -> PyResult<Vec<String>> {
        let d = self.lock()?;
        let range = d.system.movable.ligand.atoms;
        Ok((range.0..range.1)
            .map(|i| format!("{:?}", d.system.movable.atoms[i].el))
            .collect())
    }

    /// Number of atoms in the ligand (including hydrogens).
    fn num_ligand_atoms(&self) -> PyResult<usize> {
        let d = self.lock()?;
        let range = d.system.movable.ligand.atoms;
        Ok(range.1 - range.0)
    }

    /// Zero-copy ingest of ligand coordinates: replaces the ligand's starting
    /// position with the rows of `coords` (an `(n_atoms, 3)` `float64` array).
    ///
    /// The array is borrowed, not copied; `coords` must stay alive for the
    /// duration of the call, which Python guarantees.
    fn set_ligand_coords(&self, coords: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        let shape = coords.shape();
        if shape[1] != 3 {
            return Err(PyValueError::new_err(
                "coords must have shape (n_atoms, 3)",
            ));
        }
        let slice = coords
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let n = shape[0];
        let mut d = self.lock()?;
        if n != d.system.movable.coords.len() {
            return Err(PyValueError::new_err(format!(
                "expected {n} coordinates but the ligand has {} atoms",
                d.system.movable.coords.len()
            )));
        }
        for i in 0..n {
            d.system.movable.coords[i] =
                DVec3::new(slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]);
        }
        Ok(())
    }

    /// The ligand's current coordinates as an `(n_atoms, 3)` `float64` array.
    fn ligand_coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = self.lock()?;
        let coords = &d.system.movable.coords;
        let mut flat = Vec::with_capacity(coords.len() * 3);
        for c in coords {
            flat.push(c.x);
            flat.push(c.y);
            flat.push(c.z);
        }
        PyArray1::from_vec(py, flat)
            .reshape([coords.len(), 3])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        let d = self.lock()?;
        Ok(format!(
            "Docking(atoms={}, torsions={}, box={:?}, search={})",
            d.system.movable.atoms.len(),
            d.system.shared.num_torsions(),
            d.options.box_.size,
            search_name(d.options.search.use_island_ga, d.options.search.local_search),
        ))
    }
}

fn components_to_dict(py: Python<'_>, c: &dock_core::scoring::ScoreComponents) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("affinity", c.total)?;
    d.set_item("total", c.total)?;
    d.set_item("inter", c.inter)?;
    d.set_item("intra", c.intra)?;
    d.set_item("conf_independent", c.conf_independent)?;
    d.set_item("unbound", c.unbound)?;
    Ok(d.into())
}

/// Write a [`DockResult`] into `d`.
fn fill_result_dict(d: &Bound<'_, PyDict>, r: &DockResult) -> PyResult<()> {
    d.set_item("seed", r.seed)?;
    d.set_item("grid_mb", r.grid_mb)?;
    d.set_item("grid_points", r.grid_points)?;
    d.set_item("num_tors", r.num_tors)?;
    d.set_item("num_movable_atoms", r.num_movable_atoms)?;
    d.set_item("num_dof", r.num_dof)?;
    d.set_item("exact", r.exact)?;
    d.set_item("cancelled", r.cancelled)?;
    d.set_item("best_affinity", r.best().map(|p| p.energy))?;
    let py = d.py();
    let poses = PyList::empty(py);
    for (i, p) in r.poses.iter().enumerate() {
        let pd = PyDict::new(py);
        pd.set_item("index", i)?;
        pd.set_item("affinity", p.energy)?;
        pd.set_item("rmsd_lower_bound", p.lower_bound)?;
        pd.set_item("rmsd_upper_bound", p.upper_bound)?;
        pd.set_item("in_box", p.in_box)?;
        pd.set_item("inter", p.components.inter)?;
        pd.set_item("intra", p.components.intra)?;
        pd.set_item("conf_independent", p.components.conf_independent)?;
        pd.set_item("unbound", p.components.unbound)?;
        pd.set_item("num_atoms", p.coords.len())?;
        // The pose's conformation, so Python can rebuild it if needed.
        let lig = p.conf.ligands.first();
        pd.set_item(
            "position",
            lig.map(|l| (l.position.x, l.position.y, l.position.z)),
        )?;
        pd.set_item(
            "orientation",
            lig.map(|l| (l.orientation.x, l.orientation.y, l.orientation.z, l.orientation.w)),
        )?;
        pd.set_item("torsions", lig.map(|l| l.torsions.clone()))?;
        poses.append(pd)?;
    }
    d.set_item("poses", poses)?;
    Ok(())
}

fn result_to_dict(py: Python<'_>, r: &DockResult) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    fill_result_dict(&d, r)?;
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Batch docking (high-throughput virtual screening)
// ---------------------------------------------------------------------------

/// The keyword arguments of one docking job, with the same defaults as the
/// [`PyDocking`] constructor.
#[derive(Debug, Clone)]
struct JobSpec {
    receptor: Option<String>,
    ligand: Option<String>,
    label: String,
    center: (f64, f64, f64),
    size: (f64, f64, f64),
    spacing: f64,
    scoring: String,
    exhaustiveness: usize,
    num_poses: usize,
    seed: u64,
    use_grid: bool,
    refine: bool,
    min_rmsd: f64,
    energy_range: f64,
    islands: usize,
    population: usize,
    generations: usize,
    global_steps: Option<usize>,
    local_steps: Option<usize>,
    search: String,
}

impl Default for JobSpec {
    fn default() -> Self {
        JobSpec {
            receptor: None,
            ligand: None,
            label: String::new(),
            center: (0.0, 0.0, 0.0),
            size: (22.5, 22.5, 22.5),
            spacing: 0.375,
            scoring: "vina".to_string(),
            exhaustiveness: 8,
            num_poses: 9,
            seed: 0,
            use_grid: true,
            refine: true,
            min_rmsd: 1.0,
            energy_range: 3.0,
            islands: 4,
            population: 32,
            generations: 20,
            global_steps: None,
            local_steps: None,
            search: "monte_carlo".to_string(),
        }
    }
}

impl JobSpec {
    /// Overlay every key present in `d`.
    fn apply<'py>(&mut self, d: &Bound<'py, PyDict>) -> PyResult<()> {
        // A macro rather than a generic helper: `FromPyObject` is parameterised
        // by two lifetimes in PyO3 0.27, and calling `extract::<T>()` at a
        // concrete type keeps the error type concrete too.
        macro_rules! job_get {
            ($key:expr, $t:ty) => {
                match d.get_item($key)? {
                    Some(v) if !v.is_none() => Some(v.extract::<$t>().map_err(|e| {
                        PyValueError::new_err(format!("{}: {e}", $key))
                    })?),
                    _ => None,
                }
            };
        }

        if let Some(v) = job_get!("receptor", String).or(job_get!("receptor_pdbqt", String)) {
            self.receptor = Some(v);
        }
        if let Some(v) = job_get!("ligand", String).or(job_get!("ligand_pdbqt", String)) {
            self.ligand = Some(v);
        }
        if let Some(v) = job_get!("label", String).or(job_get!("name", String)) {
            self.label = v;
        }
        if let Some(v) = job_get!("center", (f64, f64, f64)) {
            self.center = v;
        }
        if let Some(v) = job_get!("size", (f64, f64, f64)) {
            self.size = v;
        }
        if let Some(v) = job_get!("spacing", f64) {
            self.spacing = v;
        }
        if let Some(v) = job_get!("scoring", String) {
            self.scoring = v;
        }
        if let Some(v) = job_get!("exhaustiveness", usize) {
            self.exhaustiveness = v;
        }
        if let Some(v) = job_get!("num_poses", usize) {
            self.num_poses = v;
        }
        if let Some(v) = job_get!("seed", u64) {
            self.seed = v;
        }
        if let Some(v) = job_get!("use_grid", bool) {
            self.use_grid = v;
        }
        if let Some(v) = job_get!("refine", bool) {
            self.refine = v;
        }
        if let Some(v) = job_get!("min_rmsd", f64) {
            self.min_rmsd = v;
        }
        if let Some(v) = job_get!("energy_range", f64) {
            self.energy_range = v;
        }
        if let Some(v) = job_get!("islands", usize) {
            self.islands = v;
        }
        if let Some(v) = job_get!("population", usize) {
            self.population = v;
        }
        if let Some(v) = job_get!("generations", usize) {
            self.generations = v;
        }
        // An explicit `None` resets an inherited budget.
        if d.contains("global_steps")? {
            self.global_steps = job_get!("global_steps", usize);
        }
        if d.contains("local_steps")? {
            self.local_steps = job_get!("local_steps", usize);
        }
        if let Some(v) = job_get!("search", String) {
            self.search = v;
        }
        Ok(())
    }

    /// Build the kernel options, validating the inputs.
    fn options(&self) -> PyResult<DockOptions> {
        let receptor = self
            .receptor
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("every job needs a `receptor` PDBQT string"))?;
        let ligand = self
            .ligand
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("every job needs a `ligand` PDBQT string"))?;
        let _ = (receptor, ligand);
        if self.spacing <= 0.0 {
            return Err(PyValueError::new_err("spacing must be positive"));
        }
        if self.size.0 <= 0.0 || self.size.1 <= 0.0 || self.size.2 <= 0.0 {
            return Err(PyValueError::new_err("box size must be positive"));
        }
        let (island, local_search) = parse_search(&self.search)?;
        let mut options = DockOptions {
            sf_choice: parse_sf(&self.scoring)?,
            box_: GridBox::new(vec3(self.center), vec3(self.size), self.spacing),
            use_grid: self.use_grid,
            refine: self.refine,
            energy_range: self.energy_range,
            ..Default::default()
        };
        options.search = SearchParams {
            exhaustiveness: self.exhaustiveness,
            num_poses: self.num_poses,
            min_rmsd: self.min_rmsd,
            seed: self.seed,
            use_island_ga: island,
            local_search,
            islands: self.islands,
            population: self.population,
            generations: self.generations,
            global_steps: self.global_steps,
            local_steps: self.local_steps,
            ..Default::default()
        };
        Ok(options)
    }

    /// The job as a kernel [`BatchJob`].
    fn to_batch_job(&self, index: usize) -> PyResult<BatchJob> {
        let options = self.options()?;
        let label = if self.label.is_empty() {
            format!("job-{index}")
        } else {
            self.label.clone()
        };
        Ok(BatchJob::new(
            label,
            self.receptor.clone().unwrap_or_default(),
            self.ligand.clone().unwrap_or_default(),
            options,
        ))
    }
}

/// One batch outcome as a dictionary (same shape as `Docking.run()`, plus
/// `label`, `ok` and `error`).
fn outcome_to_dict(py: Python<'_>, o: &BatchOutcome) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    match &o.result {
        Some(r) => fill_result_dict(&d, r)?,
        None => {
            d.set_item("seed", py.None())?;
            d.set_item("grid_mb", 0)?;
            d.set_item("grid_points", 0)?;
            d.set_item("num_tors", 0.0)?;
            d.set_item("num_movable_atoms", 0)?;
            d.set_item("num_dof", 0)?;
            d.set_item("exact", false)?;
            d.set_item("cancelled", o.cancelled)?;
            d.set_item("best_affinity", py.None())?;
            d.set_item("poses", PyList::empty(py))?;
        }
    }
    d.set_item("label", &o.label)?;
    d.set_item("ok", o.is_ok())?;
    d.set_item("error", o.error.clone())?;
    Ok(d.into())
}

/// Dock many independent ligands in parallel.
///
/// `jobs` is a sequence of dictionaries; every job needs `receptor` and
/// `ligand` PDBQT strings and may set any keyword of the `Docking` constructor
/// (`center`, `size`, `spacing`, `scoring`, `exhaustiveness`, `num_poses`,
/// `seed`, `use_grid`, `refine`, `min_rmsd`, `energy_range`, `islands`,
/// `population`, `generations`, `global_steps`, `local_steps`, `search`) plus
/// an optional `label` / `name`.
///
/// Any further keyword argument (`dock_batch(jobs, search="lga_solis")`) is
/// used as a default for every job.
///
/// Returns one dictionary per job, in the input order: the same shape as
/// `Docking.run()` with `label`, `ok` and `error` added. A job that fails does
/// not abort the batch — its `ok` is `False` and `error` explains why.
///
/// The GIL is released while the jobs run; `threads` bounds the `rayon` pool
/// (`None` uses every core).
#[pyfunction]
#[pyo3(signature = (jobs, *, threads = None, **defaults))]
fn dock_batch(
    py: Python<'_>,
    jobs: &Bound<'_, PyList>,
    threads: Option<usize>,
    defaults: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let mut base = JobSpec::default();
    if let Some(d) = defaults {
        base.apply(d)?;
    }
    if base.receptor.is_some() || base.ligand.is_some() {
        return Err(PyValueError::new_err(
            "a batch-wide `receptor`/`ligand` default is not supported; put them in each job",
        ));
    }
    let mut specs: Vec<JobSpec> = Vec::with_capacity(jobs.len());
    for (i, item) in jobs.iter().enumerate() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| PyValueError::new_err(format!("job {i} is not a dict")))?;
        let mut spec = base.clone();
        spec.apply(dict)?;
        specs.push(spec);
    }
    let mut rust_jobs = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        rust_jobs.push(spec.to_batch_job(i)?);
    }
    let outcomes = py.detach(move || core_dock_batch(rust_jobs, threads, None));
    let list = PyList::empty(py);
    for o in &outcomes {
        list.append(outcome_to_dict(py, o)?)?;
    }
    Ok(list.into())
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Kernel version.
#[pyfunction]
fn version() -> &'static str {
    dock_core::VERSION
}

/// Whether a GPU adapter is available for affinity-grid construction.
#[pyfunction]
fn gpu_available() -> bool {
    dock_core::gpu::available()
}

/// A human-readable description of the GPU backend (or why it is unavailable).
#[pyfunction]
fn gpu_description() -> String {
    dock_core::gpu::describe()
}

/// The X-Score atom-type names, indexed by type index.
#[pyfunction]
fn xs_type_names() -> Vec<String> {
    use dock_core::atom::XsType;
    (0..XsType::COUNT)
        .map(|i| format!("{:?}", XsType::from_index(i)))
        .collect()
}

/// Parse a PDBQT ligand and return `(n_atoms, n_torsions, torsdof)`.
#[pyfunction]
fn ligand_info(pdbqt: &str) -> PyResult<(usize, usize, u32)> {
    let l = dock_core::io::pdbqt::parse_ligand_pdbqt(pdbqt).map_err(err)?;
    Ok((l.atoms.len(), l.rotors.len(), l.torsdof))
}

/// Parse a PDBQT receptor and return the number of atoms.
#[pyfunction]
fn receptor_info(pdbqt: &str) -> PyResult<usize> {
    let r = dock_core::io::pdbqt::parse_receptor_pdbqt(pdbqt).map_err(err)?;
    Ok(r.atoms.len())
}

/// Perceive the X-Score atom types of a PDBQT document.
///
/// Returns one X-Score type name per atom, in file order, after bond
/// perception and typing exactly as the docking kernel sees them. This is the
/// entry point for auditing why a particular pose scores the way it does.
#[pyfunction]
#[pyo3(signature = (pdbqt, receptor = false))]
fn perceive_types(pdbqt: &str, receptor: bool) -> PyResult<Vec<String>> {
    let mut atoms = if receptor {
        dock_core::io::pdbqt::parse_receptor_pdbqt(pdbqt)
            .map_err(err)?
            .atoms
    } else {
        dock_core::io::pdbqt::parse_ligand_pdbqt(pdbqt)
            .map_err(err)?
            .atoms
    };
    dock_core::molecule::perceive_bonds(&mut atoms);
    dock_core::molecule::assign_types(&mut atoms);
    Ok(atoms.iter().map(|a| format!("{:?}", a.xs)).collect())
}

/// Perceive the AutoDock 4 atom types of a PDBQT document.
#[pyfunction]
#[pyo3(signature = (pdbqt, receptor = false))]
fn perceive_ad_types(pdbqt: &str, receptor: bool) -> PyResult<Vec<String>> {
    let atoms = if receptor {
        dock_core::io::pdbqt::parse_receptor_pdbqt(pdbqt)
            .map_err(err)?
            .atoms
    } else {
        dock_core::io::pdbqt::parse_ligand_pdbqt(pdbqt)
            .map_err(err)?
            .atoms
    };
    Ok(atoms
        .iter()
        .map(|a| a.ad.name().to_string())
        .collect())
}

/// Coordinates of a PDBQT document, in file order, as an `(n, 3)` array.
#[pyfunction]
#[pyo3(signature = (pdbqt, receptor = false))]
fn coordinates<'py>(
    py: Python<'py>,
    pdbqt: &str,
    receptor: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let atoms = if receptor {
        dock_core::io::pdbqt::parse_receptor_pdbqt(pdbqt)
            .map_err(err)?
            .atoms
    } else {
        dock_core::io::pdbqt::parse_ligand_pdbqt(pdbqt)
            .map_err(err)?
            .atoms
    };
    let n = atoms.len();
    let mut flat = Vec::with_capacity(n * 3);
    for a in &atoms {
        flat.push(a.coords.x);
        flat.push(a.coords.y);
        flat.push(a.coords.z);
    }
    PyArray1::from_vec(py, flat)
        .reshape([n, 3])
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Score a single pair of X-Score types, exposed for validation scripts.
#[pyfunction]
#[pyo3(signature = (t1, t2, r, scoring = "vina"))]
fn pair_energy(t1: &str, t2: &str, r: f64, scoring: &str) -> PyResult<f64> {
    use dock_core::atom::XsType;
    let find = |name: &str| -> PyResult<XsType> {
        (0..XsType::COUNT)
            .map(XsType::from_index)
            .find(|t| format!("{t:?}").eq_ignore_ascii_case(name))
            .ok_or_else(|| PyValueError::new_err(format!("unknown X-Score type {name:?}")))
    };
    let sf = make_scoring_function(parse_sf(scoring)?, None);
    Ok(sf.pair_energy_xs(find(t1)?, find(t2)?, r))
}

#[pymodule]
fn _odock(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", dock_core::VERSION)?;
    m.add_class::<PyWeights>()?;
    m.add_class::<PyScoringFunction>()?;
    m.add_class::<PyDocking>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_available, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_description, m)?)?;
    m.add_function(wrap_pyfunction!(xs_type_names, m)?)?;
    m.add_function(wrap_pyfunction!(ligand_info, m)?)?;
    m.add_function(wrap_pyfunction!(receptor_info, m)?)?;
    m.add_function(wrap_pyfunction!(perceive_types, m)?)?;
    m.add_function(wrap_pyfunction!(perceive_ad_types, m)?)?;
    m.add_function(wrap_pyfunction!(coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(pair_energy, m)?)?;
    m.add_function(wrap_pyfunction!(dock_batch, m)?)?;
    Ok(())
}

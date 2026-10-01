# Changelog

All notable changes to OpenDocking are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.0 — 2026-10-01

Screening at library scale, the numbers needed to judge a pose, a real
molecular surface, a workbench that behaves like an instrument, and the
engineering that makes it reproducible by someone else. 605 → 1063 tests.

### Added — reaching a conclusion

* `odock.consensus` — rescore poses with **Vina, Vinardo and AD4** at the run's
  own coordinates and combine them by **rank aggregation** (`rank`/`borda`/`z`),
  with per-field scores and ranks and **pairwise Spearman** agreement. The force
  fields genuinely disagree: mean rho 0.96 on 3PTB but **0.68 on 1M17**.
* `odock.metrics` — ligand efficiency (LE, LLE, BEI, SEI), a torsion-entropy
  estimate, MMFF94 ligand strain, and `StrainResult.reliable` decided from
  evidence (`require_reliable=True` refuses a number that cannot be trusted).
* `odock.analysis` — interaction **fingerprints** over (residue, type) pairs
  with a similarity matrix and a recurring-feature summary, and **water-mediated
  contacts**.
* `odock.report` — consensus/metrics/strain columns, fingerprint and
  pharmacophore sheets, a notebook-ready summary, streaming JSONL/CSV writers.

### Added — `odock screen`

* Library screening against one receptor or a **panel** of receptors.
* **Resumable by construction**: results flushed per molecule, and the campaign
  manifest written *before* the first docking, so a killed process resumes
  instead of re-docking; a file belonging to another campaign is refused with a
  named reason rather than silently appended to.
* Funnel with per-filter removal counts and first-failure attribution;
  `--dry-run` cost extrapolation; `--consensus` rescoring that can be added to
  a finished campaign without repeating any docking.
* One bad molecule is one `failed` row, not a dead run; Ctrl-C keeps completed
  work on disk.

### Added — seeing the chemistry

* `odock.sasa` — Shrake–Rupley SASA per atom and residue, burial against a
  stated reference, interface area, and the ligand's **buried contact area**
  (benzamidine hides 267 Å² = 94 % of itself).
* `odock.gui.surface` — a real **solvent-accessible surface** from the
  probe-centre field via marching tetrahedra, a solvent-excluded surface from
  the same field, coloured by **Kyte–Doolittle hydropathy** or by the
  **electrostatic potential** sampled from the project's own charges, with
  legend, colour-range control, pocket-lining mode and a clipping plane.
* `odock.interop` — export the live scene to **PyMOL** `.pml`, **ChimeraX**
  `.cxc`, a pose **PDB** carrying the per-atom property in the B-factor column,
  and a surface **OBJ**.

### Added — the workbench

* A **run monitor** (real per-pose energy trace, phase chips, elapsed clock),
  **pose comparison** (symmetry-aware RMSD, contact-fingerprint diff, copyable
  report), a **Ctrl+K command palette** generated from the menu bar, **session
  persistence** with recent files, a **light theme** with a compact density,
  drag-and-drop loading routed by file content, layout presets, an atom-inspect
  card, a measurement history, copy-view-to-clipboard, and pose-contact marks
  on the sequence ruler.

### Added — proof and engineering

* `odock.benchmark` — five bundled complexes, several seeds, reporting top-pose
  RMSD, `best ≤1 kcal`, the crystal mode's rank, and force-field agreement each
  with its sample size and seed spread. Every seed of every system finds the
  crystal mode; on 1M17 the top pose moves 3.2 Å between seeds while
  `best ≤1 kcal` stays at 1.28 Å.
* `benchmark/baseline.json` + `--check-baseline`, a regression gate whose
  tolerance comes from measured repeat-run noise — not the seed spread, which is
  a property of the system (reasoning recorded in the file).
* GitHub Actions: CI (Linux + Windows × Python 3.9/3.10/3.12, with
  anti-silent-skip guards for the demo data and the GUI), a scheduled benchmark
  gate, wheel/sdist builds with a clean-venv install smoke test; `Makefile`
  targets identical to CI; issue/PR templates; `tools/inspect_dist.py`, which
  fails when an artefact contains anything internal and self-tests its rules.
* New docs: `BENCHMARK.md`, `SCIENCE.md`, `SCREENING.md`, `VISUALIZATION.md`,
  plus refreshed architecture, scoring, validation and user guides.

### Fixed

* **AD4 produced NaN affinities**: 128 Gasteiger charges were not finite (8
  written literally as `inf`) and the writer sanitised only NaN. Non-finite
  charges are now zeroed **and reported**.
* **Prepared ligands lost every X–H bond** on an RDKit round trip because the
  added hydrogens carried a different residue name from the heavy atoms — the
  reason MMFF94 could not type the demo ligand.
* **A restart after a real process death re-docked the whole library** and
  appended duplicates, because the manifest was written only at the end; and a
  **changed `--seed` on resume silently mixed two campaigns**.
* **Ligand strain could be silently wrong**: the documented escape hatch did not
  help (68.6 → 60.8 kcal/mol) and relaxed coordinates were matched to atoms by
  file order. The element sequence is now checked and a mismatch raises.
* **The published wheel metadata pointed at the wrong repository**, the wheel
  shipped 32 `__pycache__` files, and the sdist included two internal documents.
* **`cargo fmt` failed outright** (an alias shadowed `cargo-fmt`), and 33 clippy
  findings made `-D warnings` unpassable — now acknowledged by name so the gate
  catches anything new.
* The window could not be made narrower than **1134 px**, and a pre-existing
  unwrapped label pushed it to **1654 px** with poses loaded; both floors are
  gone (932 × 593) and the guard now asserts the mechanism.
* The run monitor attributed a real duration to the wrong phase while running;
  `make_demo.py` and two scripts crashed on a GBK console printing Å;
  `fetch_ligand_sdf` 404'd after RCSB moved to `_ideal.sdf`; the light theme
  washed out the 3-D legend ink; the measurement hint was hidden behind the
  floating tool strip.

## 0.1.0 — 2026-09-30

The first public release: a complete AutoDock-family docking toolchain built on
a pure-Rust kernel, with a Python API, a command line and a 3-D workbench.

### Kernel (`dock-core`, `dock-py`)

* Rigid-cluster / torsion-tree kinematics with forward kinematics and a
  projected degree-of-freedom gradient.
* Three force fields — AutoDock **Vina** (default), **Vinardo** and **AutoDock
  4.2** — with an analytic first derivative for every term, each verified
  against central finite differences.
* Trilinearly interpolated affinity grids with an analytic gradient and an
  out-of-box penalty; every reported pose is refined on the exact pairwise
  surface.
* Three searches: Metropolis Monte-Carlo iterated local search, the island-model
  Lamarckian GA, and the LGA with a **Solis-Wets** adaptive local search.
* `rayon` data parallelism, deterministic seeding, and pause / resume / abort
  (`Docking.pause`, `resume`, `cancel`, `is_running`).
* Batch docking of independent ligands from released-GIL threads
  (`_odock.dock_batch`).
* Optional **wgpu** compute backend (`--features gpu`) for grid construction,
  batch forward kinematics and batch atom-pair evaluation, numerically equal to
  the CPU path.
* PDBQT reader/writer, including the nested `ROOT`/`BRANCH`/`TORSDOF` topology
  language.

### Python layer

* Receptor preparation: HETATM classification and cleaning, structural-water
  retention by distance, co-crystallised-ligand extraction, cofactor and ion
  policies, missing-atom detection, pH-aware protonation (HID/HIE/HIP),
  polar-hydrogen-only conversion, Gasteiger/Kollman charges, AD4 typing, and
  flexible-residue PDBQT export.
* Ligand preparation: `.smi`/`.sdf`/`.mol`/`.mol2`/`.pdb`/`.pdbqt` input in
  batches, ETKDGv3 embedding, MMFF94/MMFF94s/UFF minimisation,
  rotatable-bond perception with an interactive bond locker, and the torsion
  tree writer.
* Library pre-filters: Lipinski, Veber and PAINS.
* Blind pocket detection with volume, buriedness and residue labelling.
* Analysis: symmetry-aware RMSD, RMSD clustering, the six non-covalent
  interaction types with adjustable cut-offs, 2-D interaction diagrams (SVG),
  pose interpolation, and XLSX/CSV reports.
* Exporters: AutoGrid **GPF**, AutoDock 4 **DPF**, Vina **config**, and cleaned
  **PDB** with a correct `MASTER`/`END` tail; RCSB downloads by PDB ID or
  chemical-component ID.
* CLI: `info`, `prepare receptor|ligand`, `box`, `dock`, `score`, `split`,
  `pocket`, `filter`, `cluster`, `interactions`, `diagram`, `report`, `fetch`,
  `export gpf|dpf|config|pdb`, `gui`; bare `odock` opens the workbench.
* Workbench (PyQt6 + ModernGL): cartoon / ribbon / tube / space-filling /
  ball-and-stick / sticks / dots protein styles and four ligand styles,
  valence-aware bond perception, an interactive draggable grid box, the residue
  ruler with click-to-select, the atom panel, interaction highlighting with
  adjustable cut-offs, pose playback, and an English / Simplified Chinese
  interface.

### Documentation

* `docs/ARCHITECTURE.md`, `docs/DATA_STRUCTURES.md`, `docs/SCORING.md`,
  `docs/USER_GUIDE.md` and `docs/VALIDATION.md`; `demo/` ships two complete,
  regenerable example systems (3PTB and 1M17).

### Known limitations

Recorded in full in [`docs/VALIDATION.md`](docs/VALIDATION.md#6-what-is-not-validated).
In short: the search is rigid-receptor (a flexible-residue PDBQT can be written,
but the kernel's receptor reader folds it back to rigid and says so); the
Kollman charge model is the united-atom scheme, not the AMBER residue tables;
GPU acceleration covers grid construction and batch kernels only; SSAO is a
screen-space approximation; and macrocycle closure sampling is not specially
optimised.

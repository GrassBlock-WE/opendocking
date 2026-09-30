# Changelog

All notable changes to OpenDocking are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — unreleased

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

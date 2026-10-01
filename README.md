# OpenDocking (`odock`)

> A modern, fully open-source molecular docking toolchain: a pure-Rust
> high-performance kernel with a Python workbench and a 3-D GUI.

OpenDocking is a ground-up re-implementation of the complete AutoDock docking
stack — AutoDock 4, AutoDock Vina and AutoDock-GPU — released as a single
GPL-3.0 project. It keeps the physics and the file formats of the family it
replaces, drops the C++/CUDA toolchain lock-in, and adds a modern Python API.

```text
   ┌──────────────────────────────────────────────────────────────────┐
   │  odock (CLI)          odock-gui (PyQt6 + ModernGL 3-D workbench) │
   ├──────────────────────────────────────────────────────────────────┤
   │  python/odock    RDKit perception · PDBQT I/O · results · boxes  │
   ├──────────────────────────────────────────────────────────────────┤
   │  dock-py         PyO3 + NumPy zero-copy bindings                 │
   ├──────────────────────────────────────────────────────────────────┤
   │  dock-core       kinematics · scoring · search · grids · wgpu    │
   └──────────────────────────────────────────────────────────────────┘
```

## Features

**Kernel (`crates/dock-core`, pure Rust, no `unsafe`)**

* Rigid clusters plus a rotatable-bond **kinematic tree**; forward kinematics
  from torsion angles to all-atom Cartesian coordinates in one `O(n)` pass.
* **Three force fields**, each matching its reference implementation term for
  term: AutoDock **Vina** (default), **Vinardo** and **AutoDock 4.2**.
* **Analytic first derivatives** for every term, projected onto the tree's
  degrees of freedom by the rigid-body chain rule. Every derivative is verified
  against central finite differences in the test suite.
* **Trilinearly interpolated affinity grids** with an exact analytic gradient of
  the interpolated energy and a linear out-of-box penalty — so the search runs
  on the grid and the reported energies come from the exact pairwise scorer.
* **Parallel search**: BFGS local optimisation (Armijo backtracking, Vina's
  inverse-Hessian update), Monte-Carlo iterated local search with Metropolis
  acceptance, and an **island-model Lamarckian genetic algorithm**.
  `rayon` drives every independent random start across all cores.
* Deterministic: the same seed reproduces a run bit for bit.
* Optional **`wgpu` compute backend** (`--features gpu`) that builds the
  affinity grid on Vulkan / Metal / Direct3D 12.

**Python layer (`python/odock`)**

* RDKit-based receptor/ligand preparation and PDBQT writing, implemented from
  the format specification (no AutoDockTools code).
* A high-level API: `prepare_receptor`, `prepare_ligand`, `box_from_ligand`,
  `dock`, `score`.
* **Consensus scoring**: rescore one pose set with Vina, Vinardo and AD4 and
  combine them by rank aggregation, with the pairwise Spearman agreement, so a
  ranking that only one force field supports is visible as such.
* **Ligand efficiency and strain**: LE, LLE, BEI, SEI, a torsion-entropy
  estimate, and a ligand-strain correction (kernel `intra` plus an MMFF94
  relaxation).
* **Interaction fingerprints**: a fixed-length (residue, interaction-type) vector
  per pose, a similarity matrix across poses, the recurring contacts of the top
  poses, and water-mediated bridges.
* **Ligand cheminformatics**: circular and path fingerprints, Tanimoto/Dice/
  Tversky similarity, analogue search, Murcko scaffolds and series, R-group
  decomposition and matched molecular pairs with affinity deltas, MaxMin and
  sphere-exclusion diversity picking, and Butina clustering — see
  [`docs/CHEMINFORMATICS.md`](docs/CHEMINFORMATICS.md).
* **Ensemble docking**: dock against a *set* of receptor conformations, superposed
  on their binding site so one box means the same thing in every frame; merge and
  cluster the poses across conformations and measure how robust a binding mode is
  to the receptor moving (cross-receptor rescoring at fixed coordinates, with its
  sample sizes) — see [`docs/ENSEMBLE.md`](docs/ENSEMBLE.md).
* **Cryptic and transient pockets**: detect the cavities of every conformation,
  map them into the common frame, match them across structures by their lining
  residues, and report which sites open and close — separated from the pocket
  detector's own measured noise floor, so "we found a cryptic site" and "we found
  noise" are different answers — see [`docs/POCKETS.md`](docs/POCKETS.md).
* A `odock` command-line interface for scripts and pipelines.
* A PyQt6 + ModernGL workbench: drag the grid box, watch the score live, browse
  poses.

**Graphics**

* One WGSL compute kernel, three desktop backends. AutoDock-GPU needs CUDA;
  OpenDocking needs a Vulkan/Metal/D3D12 driver, or nothing at all (the CPU path
  is always available and always correct).

## Quick start

```bash
# 1. Build the extension into your virtual environment.
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install maturin rdkit numpy
maturin develop --release

# 2. Open the 3-D workbench. `odock` on its own is the interactive entry point.
odock
odock receptor.pdbqt poses.pdbqt        # or straight onto some structures

# 3. Or use the command line.
odock prepare receptor receptor.pdb receptor.pdbqt
odock prepare ligand ligand.sdf ligand.pdbqt
odock box --ligand ligand.pdbqt --out box.json --buffer 8
odock dock -r receptor.pdbqt -l ligand.pdbqt --box box.json -o poses.pdbqt -v

# 4. Or drive it from Python.
python -c "
import odock
r,_,_ = odock.prepare_receptor('receptor.pdb')
l,_,_ = odock.prepare_ligand('ligand.sdf')
box   = odock.box_from_ligand(l, buffer=8)
res   = odock.dock_text(r, l, box, exhaustiveness=16, seed=42)
print(res.table())
"
```

## The library API

```python
import odock

# --- preparation -----------------------------------------------------------
receptor_mol, receptor_pdbqt, report = odock.prepare_receptor(
    "4EY7.pdb", "receptor.pdbqt", keep_water=False, keep_hetero=True
)
ligand_mol, ligand_pdbqt, report = odock.prepare_ligand(
    "donepezil.sdf", "ligand.pdbqt"
)

# --- the search box --------------------------------------------------------
box = odock.box_from_ligand(ligand_mol, buffer=8.0)          # around a ligand
box = odock.box_from_selection(receptor_mol,                # around a residue
                               lambda a: a.GetPDBResidueInfo()
                                          and a.GetPDBResidueInfo().GetResidueNumber() == 337,
                               buffer=6.0)

# --- docking ---------------------------------------------------------------
result = odock.dock("receptor.pdbqt", "ligand.pdbqt", box,
                    scoring="vina", exhaustiveness=16, num_poses=9, seed=42)
print(result.table())          # Vina-style affinity table
print(result.best_affinity)    # kcal/mol
open("poses.pdbqt", "w").write(result.to_pdbqt())
```

### What a modeller can conclude from a run

A single affinity does not say whether the ranking is a property of the physics
or of the parameterisation, whether the hit is efficient, whether the pose is
strained, or which contacts would survive a second run.  A handful of calls
answer those questions, and
[`docs/SCIENCE.md`](docs/SCIENCE.md) explains every approximation they make:

```python
import odock
from odock import analysis, consensus, metrics, report

ligand_mol, ligand_pdbqt, prep = odock.prepare_ligand("ligand.sdf", "ligand.pdbqt")
box    = odock.box_from_ligand(ligand_mol, buffer=8.0)
result = odock.dock("receptor.pdbqt", ligand_pdbqt, box, seed=42)

# Do the three force fields agree?  (rank aggregation, not an energy average)
combined = consensus.consensus_score(result, "receptor.pdbqt", box)
print(combined.table(), combined.agreement)          # per-field score, rank, rho

# Is the hit efficient, and is the pose strained?
print(metrics.efficiency_metrics(result.best().affinity, mol=ligand_mol).as_dict())
print(metrics.pose_strain(result, result.poses[0]).strain)      # kcal/mol

# Which contacts reproduce across the best poses?  (plus water bridges)
ligands = [odock.pose_to_mol(p, ligand_mol, prep.atom_order) for p in result.poses]
fingerprints = analysis.pose_fingerprints("receptor.pdbqt", ligands)
print(fingerprints.pharmacophore(top=3).table())

# Paste-ready summary, a spreadsheet with a fingerprint sheet, JSONL for a screen
print(report.notebook_summary(result, consensus=combined, fingerprints=fingerprints,
                              ligand_mol=ligand_mol))
report.write_xlsx("run.xlsx", result, consensus=combined, fingerprints=fingerprints)
with report.ResultStream("screen.jsonl") as stream:
    stream.append({"ligand": "aspirin", "affinity": -7.1})
```

The same measurements on the bundled demo make the point: 3PTB (benzamidine, 9
heavy atoms) is ranked consistently by all three force fields (mean Spearman
rho **0.962**), while 1M17 (erlotinib, 29 heavy atoms, 11 rotors) is not (**0.683**,
with `rho(vina, ad4) = 0.500`) — and the contact that recurs across its top three
consensus poses is **MET769**, the EGFR hinge.

## Repository layout

```text
Cargo.toml                  Rust workspace
pyproject.toml              maturin build + Python packaging
crates/dock-core/           the kernel (pure Rust, #![deny(unsafe_code)])
    src/math.rs             quaternions, rigid-body primitives
    src/rng.rs              deterministic PCG-XSH-RR
    src/atom.rs             AD4 + X-Score typing and parameter tables
    src/molecule.rs         atoms, bonds, graph-based bond perception
    src/kinematics.rs       rigid clusters, torsion tree, forward kinematics
    src/scoring/            potentials, gradients, affinity grids, non-cache
    src/search/             BFGS, Monte-Carlo ILS, island GA, pose container
    src/io/pdbqt.rs         PDBQT reader and writer
    src/gpu/                optional wgpu backend (grid.wgsl)
    src/docking.rs          the orchestrator (Vina's `Vina` equivalent)
crates/dock-py/             PyO3 + NumPy bindings
python/odock/               Python API, CLI, preparation, GUI
tests/                      Python integration and validation tests
docs/                       architecture, data structures, scoring derivation,
                            science layer, user guide, screening, cheminformatics,
                            receptor ensembles and cryptic pockets
```

## Design notes

**Why a tree, not a graph.** A rigid body plus a spanning tree over the
rotatable bonds spans exactly the same conformation space as the full molecular
graph, but every conformation is generated in a single forward pass with no
constraint solving. Rings are simply never crossed by the spanning tree, so each
ring stays rigid — the same approximation AutoDock 4 and Vina make.

**Why both a grid and an exact scorer.** The search evaluates the scoring
function millions of times; a pre-computed grid turns that into eight array
reads per atom. But an interpolated energy is not the true energy, so every
reported pose is finally relaxed on the exact pairwise surface. Both paths share
the same force-field code, so they can never drift apart.

**Why the reported affinity is `E_inter / (1 + w_rot·N_tors)`.** The ligand's
internal energy is identical in the bound and unbound states with a rigid
receptor, so it cancels exactly; the torsional term is the entropy cost of
freezing `N_tors` rotatable bonds. This is the published Vina convention, and
OpenDocking reproduces it.

**Numerical conventions.** Quaternions are `(x, y, z, w)`; the rotation matrix
matches `glam::DMat3::from_quat` and Vina's `quaternion_to_r3` bit for bit
(pinned by a unit test). Angles are wrapped into `(-π, π]`. Energies are
kcal/mol, distances Å.

## Testing

```bash
cargo test --workspace                        # 71 kernel unit tests
cargo test -p dock-core --features gpu        # + 4 wgpu backend tests
python -m pytest tests -v                     # the integration suite
python -m pytest tests -m slow -v             # + the crystallographic validation
python tests/validate_3ptb.py                 # the same validation, as a script

python -m odock.benchmark                     # the five-system re-docking benchmark
python -m odock.benchmark --check-baseline    # fail on a scoring regression
```

The Rust suite includes finite-difference verification of every analytic
derivative — the force-field terms, the grid gradients, and the full
degree-of-freedom gradient through the kinematic tree — plus a
forward-kinematics isometry test, BFGS convergence tests and a
GPU-versus-CPU affinity-grid comparison.

`python -m odock.benchmark` re-docks five bundled crystal complexes with three
seeds and reports top-pose accuracy, the rank of the correct pose, the
score-versus-RMSD correlation and the three force fields' agreement **with their
spread**, because a single number hides the 3.2 Å of seed-to-seed movement on
the flexible system. [`docs/BENCHMARK.md`](docs/BENCHMARK.md) has the measured
table, the recorded baseline in `benchmark/baseline.json`, and an explicit list
of what five systems do not establish (they are not CASF).

## Validation

### The scoring function against the reference potentials

`crates/dock-core/tests/real_ligand.rs` and the validation script both compare
the kernel's intermolecular energy against an **independent, literal
transcription of AutoDock Vina's `potentials.h`** — the Gaussian, repulsion,
hydrophobic and H-bond terms, the 8 Å cutoff, the `slope_step` ramps and the
`curl` cap, summed in double precision over every receptor atom within range.
On a 2 000-atom and a 3 000-atom receptor the two agree to **1 × 10⁻¹⁵
kcal/mol**, which is floating-point noise: the implementation is not merely
"similar to Vina", it evaluates the same function.

### Re-docking a real complex

The acceptance test re-docks the co-crystallised ligand of **PDB 3PTB**
(benzamidine in bovine trypsin) from scratch: the receptor and ligand are
prepared from the raw PDB, a box is built around the experimental ligand, and
the search starts from random conformations.

| quantity | value |
|---|---|
| top-pose RMSD to the crystal structure (heavy atoms, no superposition) | **1.13 Å** |
| … after optimal superposition | 0.38 Å |
| top-pose affinity | −6.21 kcal/mol |
| affinity of the *crystal* pose | −5.81 kcal/mol |
| independent Monte-Carlo runs (`exhaustiveness`) | 16 |
| seed | 42 |
| wall time | ≈ 5 s |
| affinity grid | 150 920 points, 3 MB |

For reference, the experimental binding free energy of benzamidine to trypsin is
about −8.5 kcal/mol, and AutoDock Vina reports roughly −8 to −9 for this system.
The full workflow, including preparation, is:

```bash
odock prepare receptor 3PTB.pdb receptor.pdbqt --strip BEN
odock prepare ligand   ligand.sdf ligand.pdbqt
# the active site: a plain JSON file, easy to hand to a front-end
odock box   --ligand ligand.pdbqt --buffer 8 --out box.json
odock dock  -r receptor.pdbqt -l ligand.pdbqt --box box.json -e 16 --seed 42 -o poses.pdbqt
```

`box.json` is the entire active-site definition - a centre, an edge length
per axis and a grid spacing:

```json
{
  "center": [-1.856, 14.366, 16.748],
  "size": [17.883, 19.950, 20.514],
  "spacing": 0.375
}
```

A holo structure keeps its ligand as a HETATM residue; `--strip BEN` removes it
(otherwise the receptor occupies the very site being docked into).
`prepare_receptor` warns when it detects such a residue.

### A flexible, drug-like ligand

`demo/` (regenerated by `python examples/make_demo.py`) also docks
**erlotinib** into the EGFR kinase domain of PDB 1M17: 29 heavy atoms, 11
rotatable bonds. The search finds the experimental binding mode — it is the pose
at **1.4 Å** in the output — but the empirical force field ranks it below poses
that differ from one another by less than 0.1 kcal/mol. That is a property of
the Vina energy function on a highly flexible ligand (and of stripping the
crystallographic waters, which mediate part of this particular binding mode),
not of the engine: the rigid benzamidine case above is reproduced exactly. The
demo README states this plainly rather than reporting only the flattering
number.

## Validation summary

[`docs/VALIDATION.md`](docs/VALIDATION.md) is the validation report: what is
implemented where, the numbers measured on this tree (the automated suites, the
3PTB re-docking acceptance test, the five-system benchmark) and the limitations
that remain. `tests/simulate_workbench.py` drives the real window through **48
user steps** with Qt input events — the startup layout and the 3-D viewport, then
the File, Receptor, Ligand, Grid, Docking, Poses, Analysis and View menus — and
writes an illustrated report to `out/simulation/report.md`.

**Reproducible runs and reports.** A finished run can be saved as one verifiable
file (`odock project save`), reopened anywhere without its original paths
(`odock project open`), re-run and checked against its own record
(`odock project reproduce`), compared with another run field by field
(`odock project compare`), indexed across a whole campaign (`odock project
index`) and turned into one self-contained HTML document with the figures inside
it (`odock report-html`). [`docs/PROJECTS.md`](docs/PROJECTS.md) documents the
container, the verification semantics, the schema version as the compatibility
contract, and what a reproduction does and does not establish.

## Status and known limitations

Implemented and tested: the Vina, Vinardo and AD4 force fields with analytic
gradients; the rigid-body/torsion kinematic tree; trilinear affinity grids;
BFGS local search, Monte-Carlo iterated local search and the island-model
Lamarckian genetic algorithm; multi-core execution with `rayon`; deterministic
seeding; PDBQT reading and writing; RDKit-based preparation; the CLI; and the
PyQt6 + ModernGL workbench.

Deliberately out of scope for this release, with the extension points already in
place:

* **Flexible receptor side chains.** The kernel carries the machinery
  (`MovableModel::flex`, `TreeKind::Flex`, `build_flex_tree`, the flex terms in
  the scoring code and the flex pair list) and is unit-tested, but the PDBQT
  reader currently folds a flexible-residue file into the rigid receptor and
  emits a warning. Treating a side chain as rigid is a controlled,
  clearly-reported approximation; treating it as flexible without a validated
  parser would not be.
* **Explicit solvation and metal coordination.** The Vina force field is an
  implicit-solvent empirical potential; only the AD4 force field models
  desolvation and electrostatics.
* **The GPU path accelerates grid construction only.** The Monte-Carlo/BFGS
  search stays on the CPU, where its short, branchy per-atom work is faster than
  a kernel launch. The `gpu` feature is off by default so a plain build never
  needs a graphics driver.

## Licence

Copyright (C) 2024-2025 The OpenDocking Project.

OpenDocking is free software: you can redistribute it and/or modify it under the
terms of the **GNU General Public License, version 3 or later**. See
[LICENSE](LICENSE).

### Upstream attribution

OpenDocking is an independent re-implementation, but it deliberately follows the
published algorithms and the freely licensed reference implementations:

* **AutoDock Vina** — Copyright (c) 2006-2010, The Scripps Research Institute.
  Apache License 2.0. Reference for the Vina/Vinardo potentials, the analytic
  derivative formulation, the iterated-local-search protocol, the PDBQT topology
  conventions and the AD4 parameter table in `atom_constants.h`.
* **AutoDock 4** — Copyright (c) 1989-2007, The Scripps Research Institute. GPL.
  Reference for the AD4.2 force-field form and the AD4 atom typing.
* **Meeko** — Copyright (c) Forli Lab, Scripps Research. LGPL-2.1. Reference for
  the *behaviour* of ligand preparation (united-atom hydrogens, rotatable-bond
  selection, macrocycle handling). No code was copied.

No proprietary or closed-source component was used: in particular, **no
AutoDockTools (ADT / MGLTools) source was read, borrowed or copied.** All PDBQT
and receptor/ligand pre-processing in this project was written from the format
specification plus RDKit.

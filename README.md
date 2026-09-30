```text
 ██████╗ ██████╗ ███████╗███╗   ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║
╚██████╔╝██║     ███████╗██║ ╚████║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝
    ██████╗  ██████╗  ██████╗██╗  ██╗██╗███╗   ██╗ ██████╗
    ██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██║████╗  ██║██╔════╝
    ██║  ██║██║   ██║██║     █████╔╝ ██║██╔██╗ ██║██║  ███╗
    ██║  ██║██║   ██║██║     ██╔═██╗ ██║██║╚██╗██║██║   ██║
    ██████╔╝╚██████╔╝╚██████╗██║  ██╗██║██║ ╚████║╚██████╔╝
    ╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝
```

**English** · [简体中文](README_zh.md)

**A complete, open-source molecular docking and virtual screening toolchain —
from a raw PDB structure to an analysed binding pose.**

* **Pure-Rust kernel** — Vina, Vinardo and AutoDock 4 force fields with fully
  analytic gradients, trilinear affinity grids, Monte-Carlo iterated local
  search, an island-model Lamarckian genetic algorithm and Solis-Wets adaptive
  local search, parallelised with `rayon` and optionally accelerated on the GPU
  through `wgpu` + WGSL.
* **Python-native API** — PyO3 bindings with zero-copy NumPy transfer,
  cooperative pause / resume / abort, and parallel batch docking.
* **End-to-end toolchain** — receptor cleaning and pH-aware protonation, ligand
  preparation with force-field minimisation and click-to-lock rotatable bonds,
  blind pocket detection, an interactive search box, symmetric-aware RMSD
  clustering, six interaction types at adjustable distance cut-offs, 2-D
  interaction diagrams and Excel/CSV reports.
* **3-D workbench** — cartoon, ribbon, space-filling, ball-and-stick and
  skeletal styles, a clickable residue ruler, an atom inspector and interaction
  highlighting that points at the exact atom pairs. English and Chinese UI.
* **Verified** — 605 Python tests and 116 Rust tests pass; benzamidine re-docks
  into trypsin (PDB 3PTB) to **1.13 Å** RMSD with no superposition.

GPL-3.0-or-later.

```text
   ┌──────────────────────────────────────────────────────────────────────┐
   │  odock (CLI)            odock.gui (PyQt6 + ModernGL 3-D workbench)   │
   ├──────────────────────────────────────────────────────────────────────┤
   │  python/odock   preparation · pockets · docking API · analysis ·     │
   │                 exports · reports                                    │
   ├──────────────────────────────────────────────────────────────────────┤
   │  dock-py        PyO3 + NumPy zero-copy bindings                      │
   ├──────────────────────────────────────────────────────────────────────┤
   │  dock-core      kinematics · scoring · search · grids · wgpu         │
   └──────────────────────────────────────────────────────────────────────┘
```

## What it does

**Receptor preparation** (`python/odock/chem/receptor.py`, `charges.py`, `flex.py`)

* HETATM classification and cleaning: drop solvent by residue-name pattern
  (`HOH`, `WAT`, `DOD`, `TIP3`, …), keep structural waters within a distance of
  the binding site, strip co-crystallised ligands and export them as the
  reference pose for self-docking, and protect cofactors (heme, NAD/NADH, FAD,
  PLP, ATP) and catalytic metal ions while dropping free counter-ions.
* Missing-atom detection against standard amino-acid and nucleotide templates.
* pH-aware protonation (default pH 7.4, range 0–14): His is resolved into HID /
  HIE / HIP from its local environment, Asp/Glu are deprotonated and Lys/Arg
  protonated.
* Polar-hydrogen-only conversion: hydrogens on carbon are removed and their
  charge collapsed onto the parent atom (the AutoDock united-atom convention).
* Gasteiger and Kollman-style charge models, and the full AD4 atom-type mapping
  (`C A N NA OA SA S HD F Cl Br I`, plus the metal types).
* Flexible-residue PDBQT export (`BEGIN_RES`, nested `BRANCH`/`ENDBRANCH`,
  `END_RES`).

**Ligand preparation** (`chem/ligand.py`, `filters.py`)

* One reader for `.smi` (single and multi-line), `.sdf`, `.mol2`, `.mol`,
  `.pdb`, `.pdbqt`, bare SMILES text, RDKit molecules, and any mixture of those.
* ETKDGv3 2-D → 3-D embedding, then MMFF94 / MMFF94s / UFF minimisation
  (200–1000 gradient steps, with a geometry rollback if the energy rises).
* Rotatable-bond perception that excludes amides, alkynes, aromatic/conjugated
  ring bonds and meaningless terminal rotations — and an interactive bond
  locker: click two atoms in the 3-D view to freeze or release a torsion.
* Largest-rigid-cluster root selection and a depth-first torsion tree written as
  the AutoDock `ROOT`/`BRANCH`/`TORSDOF` language.
* Drug-likeness pre-filters: Lipinski's rule of five, Veber, and PAINS (RDKit's
  480-pattern catalog plus a built-in SMARTS fallback set).

**Search box and pocket detection** (`prepare.py`, `pocket.py`)

* Four ways to define the box: the native ligand's envelope, one to four
  selected residues' centroid, explicit centre/size numbers, or a ligand's own
  input position. The box is a plain JSON file (`odock box`), easy to hand to a
  front-end.
* Grid spacing from 0.1 Å to 1.0 Å (default 0.375 Å) and a precision of
  0.001 Å on the centre — both draggable in the workbench, with the numbers and
  the 3-D box bound in both directions.
* **Blind pocket detection** on a ligand-free receptor: a rolling-probe grid,
  26-direction buriedness, connected-component cavities, ranked by volume and
  enclosure, each with its volume in Å³ and the residues lining it.

**Docking engines** (`crates/dock-core`, `docking.py`)

* Three force fields, each matching its reference implementation term for term:
  AutoDock **Vina** (default), **Vinardo** and **AutoDock 4.2**.
* Analytic first derivatives for every term, projected onto the tree's degrees
  of freedom; every derivative is checked against central finite differences.
* Trilinearly interpolated affinity grids with an analytic gradient and a linear
  out-of-box penalty: the search runs on the grid, and every reported pose is
  finally relaxed on the exact pairwise surface.
* Three searches: Metropolis-corrected **Monte-Carlo iterated local search**,
  the island-model **Lamarckian genetic algorithm**, and the LGA with a
  **Solis-Wets** local search (what AutoDock 4 actually uses). `rayon`
  parallelises the independent starts; a run is reproducible bit for bit from
  its seed.
* **Pause, resume and abort** during a run (`Docking.pause/resume/cancel`), with
  the aborted run returning the best pose it found, and **batch docking**
  (`_odock.dock_batch(jobs, threads=…)`) that releases the GIL and keeps going
  when one job fails.
* An optional **`wgpu` compute backend** (`--features gpu`) that builds affinity
  grids on Vulkan / Metal / Direct3D 12; the CPU path is always available and
  numerically equivalent.

**Analysis and reporting** (`analysis.py`, `report.py`, `export.py`)

* Symmetry-aware RMSD (graph-based equivalent-atom matching, no SciPy) and RMSD
  clustering of a pose set (default 2.0 Å cutoff) with an energy-best
  representative per cluster.
* Six non-covalent interaction types with published geometric criteria, each
  cut-off adjustable: hydrogen bonds, salt bridges, π–π stacking
  (face-to-face vs T-shaped), cation–π, hydrophobic contacts and steric clashes.
* A 2-D interaction topology diagram as SVG, a pose-playback interpolation
  helper, and results tables as XLSX (openpyxl) or CSV.
* Input-file exporters: AutoGrid **GPF**, AutoDock 4 **DPF**, AutoDock Vina
  **config**, a cleaned **PDB** with a proper `MASTER`/`END` tail, and RCSB
  downloads by PDB or chemical-component ID.

**The 3-D workbench** (`python/odock/gui`, PyQt6 + ModernGL)

* Protein styles: cartoon (α-helix coils, β-sheet arrows, loop tubes), ribbon,
  tube, space-filling (true van der Waals radii), ball-and-stick, sticks, dots.
  Ligand styles: ball-and-stick, sticks, wireframe, spheres.
* PyMOL-style valence-aware bond perception (Pyykkö covalent radii, valence
  caps, shortest-bond-first) with an adjustable connectivity mode and cutoff.
* A residue ruler under the viewport: a tick every five residues with the real
  numbering, one-letter codes coloured by residue type, and click / Shift-range /
  Ctrl-append selection that highlights the atoms in 3-D.
* An atom panel listing the selected atoms' name, element, residue, chain,
  coordinates, AD4 type and charge, with an element summary and PDBQT export.
* Interaction annotation with adjustable distance cut-offs, clash highlighting,
  a translucent reference-pose overlay, distance measurement, camera controls,
  SSAO and coordinate axes.
* The full pipeline in one window: import structures, clean and protonate the
  receptor, build and minimise a ligand, find pockets, drag the grid box, run
  and abort a docking job, inspect poses, annotate interactions, cluster, and
  export every format.
* **English / Simplified Chinese interface**, switchable at run time (the
  catalogue is key-for-key identical in both languages).

## Prerequisites

OpenDocking is a Rust extension plus a Python package, so **the Rust toolchain is
not optional**:

| you need | why |
|---|---|
| a **Rust toolchain** (stable, 1.75+) | the numerical kernel, `crates/dock-core` and `crates/dock-py`, is Rust |
| **Python 3.9+** and `pip` | the API, the CLI and the workbench |
| **`maturin`** (`pip install maturin`) | builds and installs the extension as `odock._odock` |
| `rdkit` | receptor/ligand preparation, the chemistry layer and most of the test-suite |
| `numpy` | a hard runtime dependency (installed with the package) |
| `PyQt6` + `moderngl` | optional — only the 3-D workbench |
| `openpyxl` | optional — only XLSX reports |

> **Build before you import.** `odock/__init__.py` imports the compiled
> `odock._odock` module, and that binary (`.pyd`/`.so`) is a build artefact that
> is deliberately not in the repository. On a fresh clone, `import odock`,
> `odock …` and `pytest` all fail with
> `ModuleNotFoundError: No module named 'odock._odock'` until
> **`maturin develop --release`** has been run once in the same virtual
> environment. This applies to the test-suite as much as to the CLI.

## Quick start

```bash
# 1. A virtual environment, with the Python-side dependencies.
python -m venv .venv
.venv/Scripts/activate                 # Windows
source .venv/bin/activate              # Linux / macOS
pip install maturin rdkit numpy

# 2. Build the Rust extension into that environment (required, see above).
maturin develop --release

# 3. Optional extras: the workbench and the Excel reports.
pip install PyQt6 moderngl openpyxl

# 4. Open the 3-D workbench (`odock` on its own is the interactive entry point).
odock gui
```

Then, from the command line or from Python:

```bash
odock info                                  # versions, force fields, GPU status
odock prepare receptor 3PTB.pdb receptor.pdbqt --strip BEN
odock prepare ligand ligand.sdf ligand.pdbqt
odock box --ligand ligand.pdbqt --buffer 8 --out box.json
odock dock -r receptor.pdbqt -l ligand.pdbqt --box box.json -e 16 --seed 42 -o poses.pdbqt
odock interactions -r receptor.pdbqt -l ligand.pdbqt
odock report -p poses.pdbqt -r receptor.pdbqt -o report.xlsx
```

`odock` must resolve to this project's environment: `maturin develop` installs
the console script into `.venv/Scripts` (or `.venv/bin`). If an older `odock` is
on `PATH`, call the environment's copy explicitly or activate the environment
first.

## Command-line reference

Run `odock --help`, or `odock <command> --help`, for the full option list. Every
subcommand is a thin wrapper around the Python API, so anything the CLI does can
also be scripted.

| command | what it does | typical use |
|---|---|---|
| `odock info` | kernel version, Python/RDKit versions, the three force fields, GPU status | `odock info` |
| `odock prepare receptor` | clean, protonate and type a receptor into PDBQT | `odock prepare receptor rec.pdb rec.pdbqt --strip BEN --keep-water` |
| `odock prepare ligand` | read, embed, minimise and type a ligand into PDBQT | `odock prepare ligand lig.sdf lig.pdbqt --name donepezil` |
| `odock box` | build the search box (ligand, residue, chain or explicit numbers) | `odock box --ligand lig.pdbqt --buffer 8 --out box.json` |
| `odock dock` | run a docking search | `odock dock -r rec.pdbqt -l lig.pdbqt --box box.json -e 16 --search lga_solis -o poses.pdbqt` |
| `odock score` | score a ligand where the file puts it (no search) | `odock score -r rec.pdbqt -l poses.pdbqt --json` |
| `odock split` | split a multi-model pose file | `odock split poses.pdbqt --outdir poses` |
| `odock pocket` | blind cavity detection, ranked | `odock pocket -r rec.pdbqt --min-volume 100 --json-out pockets.json` |
| `odock filter` | Lipinski / Veber / PAINS report for one or many files | `odock filter -i library.sdf -i more.smi` |
| `odock cluster` | symmetry-aware RMSD clustering of the poses | `odock cluster -p poses.pdbqt --cutoff 2.0` |
| `odock interactions` | the non-covalent interaction profile | `odock interactions -r rec.pdbqt -l lig.pdbqt` |
| `odock diagram` | the 2-D interaction topology diagram (SVG) | `odock diagram -r rec.pdbqt -l lig.pdbqt -o interactions.svg` |
| `odock report` | the results table as XLSX or CSV | `odock report -p poses.pdbqt -r rec.pdbqt -o report.xlsx` |
| `odock fetch` | download a PDB entry or a chemical component from RCSB | `odock fetch 3PTB -o 3PTB.pdb` / `odock fetch BEN --ligand` |
| `odock export` | write AutoGrid/AutoDock/Vina input files and a cleaned PDB | `odock export gpf -r rec.pdbqt -o rec.gpf --box box.json` |
| `odock gui` | the 3-D workbench | `odock gui -r rec.pdbqt -l lig.pdbqt -p poses.pdbqt` |

`dock` accepts `--search {monte_carlo,lga,lga_solis}`, `--scoring {vina,vinardo,ad4}`,
`--exhaustiveness`, `--num-poses`, `--seed`, `--min-rmsd` and `--energy-range`;
`pocket`, `filter`, `cluster` and `interactions` write machine-readable results
with `--json-out`. The interaction cut-offs (`--hbond`, `--salt`, `--pi`,
`--cation-pi`, `--hydrophobic`, `--clash-ratio`) are shared by `interactions` and
`diagram`.

## Python API

```python
import odock
from odock import analysis, pocket, report, filters

# --- preparation -----------------------------------------------------------
receptor_mol, receptor_pdbqt, _ = odock.prepare_receptor("3PTB.pdb", keep_water=False)
ligand_mol, ligand_pdbqt, _ = odock.prepare_ligand("ligand.sdf", name="BEN")

# --- the search box (or use pocket.find_pockets for a blind search) --------
box = odock.box_from_ligand(ligand_mol, buffer=8.0)

# --- dock ------------------------------------------------------------------
result = odock.dock(
    receptor_pdbqt, ligand_pdbqt, box,
    scoring="vina", exhaustiveness=16, num_poses=9, seed=42,
    search="lga_solis",          # or "monte_carlo" / "lga"
)
print(result.table())
print(result.best_affinity)      # kcal/mol
open("poses.pdbqt", "w").write(result.to_pdbqt())

# --- analysis --------------------------------------------------------------
contacts = analysis.profile_interactions(receptor_mol, ligand_mol)
print(analysis.interaction_summary(contacts, receptor_mol, ligand_mol))
analysis.interaction_diagram_svg(receptor_mol, ligand_mol, contacts, path="dock.svg")
print(filters.drug_like(ligand_mol)["passed"])
print(pocket.find_pockets(receptor_mol)[0])
report.write_xlsx("results.xlsx", result)
```

For a run you need to steer — pause, resume, abort — build the engine yourself
and drive it; batch jobs go through the kernel's parallel entry point:

```python
from odock.docking import build_engine, result_from_engine

engine = build_engine(receptor_pdbqt, ligand_pdbqt, box, exhaustiveness=32)
engine.pause()          # Docking ▸ pause
engine.resume()
engine.cancel()         # returns the best pose found so far
raw = engine.run()

from odock import _odock
results = _odock.dock_batch([{"receptor": r, "ligand": l, "center": c, "size": s}
                             for r, l, c, s in jobs], threads=8)
```

## Verification

Everything below was measured on this tree; [`docs/VALIDATION.md`](docs/VALIDATION.md)
gives the commands, the raw output and the limitations.

| | |
|---|---|
| Rust tests | `cargo test --workspace` — **116 passed** (+5 integration tests, +1 doc test) |
| Python tests | `python -m pytest tests -q` — **605 passed, 1 skipped** |
| Workbench session | `python tests/simulate_workbench.py` — **48 steps, 48 passed, 0 failed** |
| Re-docking, PDB 3PTB | top-pose RMSD **1.133 Å** with no superposition, affinity **−6.213** kcal/mol (crystal pose −5.806), 16 MC runs, seed 42, 3.0 s |
| Scoring function | matches an independent transcription of AutoDock Vina's potentials to floating-point noise on 2 000- and 3 000-atom receptors |

The acceptance test re-docks benzamidine into bovine trypsin from the raw crystal
structure — split, prepare both partners, derive the box from the experimental
ligand, and search from random conformations:

```bash
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42
# → best-pose RMSD to the crystal structure: 1.133 Å
#   RESULT: PASS — the top pose reproduces the experimental binding mode
```

`demo/` holds two complete, regenerable examples — benzamidine/trypsin and
erlotinib/EGFR ([`demo/README.md`](demo/README.md)), regenerated by
`python examples/make_demo.py`. The flexible erlotinib case is reported honestly
there: the search finds the experimental binding mode (1.43 Å), but the force
field ranks four near-degenerate poses above it.

## Repository layout

```text
Cargo.toml                   Rust workspace (members, shared metadata, profiles)
pyproject.toml               maturin build backend, packaging, pytest settings
crates/dock-core/            the kernel (pure Rust, #![deny(unsafe_code)])
    src/math.rs              quaternions, rigid-body primitives
    src/rng.rs               deterministic PCG-XSH-RR generator
    src/atom.rs              AD4 and X-Score typing, parameter tables
    src/molecule.rs          atoms, bonds, geometric bond perception
    src/kinematics.rs        rigid clusters, torsion tree, forward kinematics
    src/scoring/             potentials and gradients, affinity grids, exact scorer
    src/search/              BFGS, Monte-Carlo ILS, island GA, Solis-Wets, poses
    src/io/pdbqt.rs          PDBQT reader and writer
    src/gpu/                 optional wgpu backend (grid.wgsl)
    src/docking.rs           the orchestrator (Vina's `Vina` equivalent)
crates/dock-py/              PyO3 + NumPy bindings (`odock._odock`)
python/odock/                preparation, pockets, docking API, analysis, GUI
    cli.py                   the `odock` console script
    chem/                    receptor, charges, ligand, flexible residues
    gui/                     the PyQt6 + ModernGL workbench
tests/                       Python test-suite and the validation scripts
examples/make_demo.py        regenerates `demo/`
demo/3ptb/, demo/egfr/       two end-to-end example systems
docs/                        architecture, data structures, scoring, user guide
```

## Documentation

* [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — installation, every subcommand,
  the Python API, tuning, and a troubleshooting section.
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the tree is organised,
  what every Rust module does, the data flow of a run, and the threading model.
* [`docs/DATA_STRUCTURES.md`](docs/DATA_STRUCTURES.md) — the core Rust data
  structures, field by field, with their upstream equivalents.
* [`docs/SCORING.md`](docs/SCORING.md) — the three force fields, term by term,
  with the derivation of every analytic gradient and the reported affinity.
* [`docs/VALIDATION.md`](docs/VALIDATION.md) — what was measured, how to
  reproduce it, and what is explicitly *not* validated.

## Contributing

Bug reports, force-field checks and new tests are all welcome — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the build, the test commands and the
code layout. Changes are listed in [`CHANGELOG.md`](CHANGELOG.md).

## Licence

Copyright (C) 2024–2025 The OpenDocking Project.

OpenDocking is free software: you can redistribute it and/or modify it under the
terms of the **GNU General Public License, version 3 or later**. See
[`LICENSE`](LICENSE) for the full text.

### Upstream attribution

OpenDocking is an independent re-implementation. It deliberately follows the
published algorithms and the freely licensed reference implementations:

* **AutoDock Vina** — Copyright (c) 2006–2010, The Scripps Research Institute.
  Apache License 2.0. Reference for the Vina and Vinardo potentials and their
  derivatives, the iterated-local-search protocol, the BFGS inverse-Hessian
  update, the PDBQT topology conventions, the trilinear grid evaluation and the
  AD4 parameter table.
* **AutoDock 4** — Copyright (c) 1989–2007, The Scripps Research Institute. GPL.
  Reference for the AD4.2 force-field form, the distance-dependent dielectric,
  the desolvation term, the Lamarckian GA, the Solis-Wets adaptive local search,
  and the GPF/DPF file formats.
* **Meeko** — Copyright (c) Forli Lab, Scripps Research. LGPL-2.1. Behavioural
  reference for ligand preparation (united-atom hydrogens, rotatable-bond
  selection, macrocycle handling). No Meeko code is linked or copied.
* **AutoDock-GPU** — LGPL-2.1. Architectural reference for heterogeneous
  acceleration.

No proprietary or closed-source component is used: in particular, **no
AutoDockTools (ADT / MGLTools) source was read, borrowed or copied.** All PDBQT
and receptor/ligand pre-processing here is written from the format specification
plus RDKit.

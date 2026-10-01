# OpenDocking user guide

OpenDocking (`odock`) is a molecular docking toolchain: a pure-Rust kernel that
does the physics, a Python package that prepares the structures and drives it,
and a `odock` command-line interface on top of both.

This guide takes you from an empty virtual environment to a docked pose file,
and then through the API, the tuning knobs and the things that commonly go
wrong. Every command and every code snippet below was checked against the
implementation in `python/odock/cli.py`, `python/odock/docking.py` and
`python/odock/prepare.py`; every option name and every default is the real one.

The algorithmic background lives in [SCORING.md](SCORING.md), the internals in
[ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_STRUCTURES.md](DATA_STRUCTURES.md).

---

## 1. What you need

| Requirement | Version | Needed for |
|---|---|---|
| Python | ≥ 3.9 (`requires-python`) | always |
| NumPy | ≥ 1.22 | always (declared dependency) |
| RDKit | ≥ 2022.9 | preparing receptors/ligands and reading structure files (`[chem]` extra) |
| Rust toolchain (`cargo`) | ≥ 1.75, edition 2021 | building the kernel extension |
| maturin | ≥ 1.5, < 2.0 | building/installing the extension |
| PyQt6 ≥ 6.5, ModernGL ≥ 5.8 | — | the optional 3-D workbench (`[gui]` extra) |

A GPU is **not** required. The CPU path is always available and always correct;
see [GPU notes](#9-gpu-notes).

---

## 2. Installation

### 2.1 Into a virtual environment

```bash
python -m venv .venv

# Windows
.venv/Scripts/activate

# Linux / macOS
source .venv/bin/activate

pip install --upgrade pip
pip install maturin rdkit numpy
```

### 2.2 Build and install the extension

```bash
maturin develop --release
```

This compiles `crates/dock-py` (the `odock._odock` extension module) and installs
the pure-Python package from `python/` into the active interpreter. Always build
with `--release`: the kernel is numerically dense and a debug build is an order
of magnitude slower.

### 2.3 Optional extras

```bash
pip install "opendocking[chem]"   # RDKit: receptor/ligand preparation
pip install "opendocking[gui]"    # PyQt6 + ModernGL: the 3-D workbench
pip install "opendocking[dev]"    # pytest + RDKit
```

The distribution is named `opendocking`; it installs the `odock` package and the
`odock` console script.

### 2.4 Check the installation

```bash
odock info
```

```text
OpenDocking 0.1.0
  python        : 3.10.6 (.../.venv/Scripts/python.exe)
  numpy         : 2.2.6
  rdkit         : 2026.03.1
  vina          : cutoff 8.00 Å, 6 terms, grid=yes
  vinardo       : cutoff 8.00 Å, 5 terms, grid=yes
  ad4           : cutoff 20.48 Å, 5 terms, grid=no
  gpu           : GPU backend unavailable: gpu error: dock-core was built without the `gpu` feature; ...
```

The `gpu` line is expected to say "unavailable" unless the kernel was built with
the `gpu` feature; see [GPU notes](#9-gpu-notes). `grid=no` for AD4 is by design:
the AD4 force field needs per-atom charges and is always evaluated exactly.

On a Windows console whose code page cannot represent `Å` (for example GBK /
cp936), `odock info` can fail with a `UnicodeEncodeError`. Fix it with
`set PYTHONIOENCODING=utf-8` (or `chcp 65001`) before running the command; see
[Troubleshooting](#10-troubleshooting).

### 2.5 If you only want the Python API

`maturin develop --release` is still required: `odock/__init__.py` imports the
compiled module first, so a missing extension fails immediately with a clear
error rather than a confusing `AttributeError` later.

---

## 3. Preparing structures

Docking needs a **receptor PDBQT** and a **ligand PDBQT**. The preparation is
RDKit-based and written from the PDBQT format specification plus the behaviour
documented by Meeko; no AutoDockTools (ADT / MGLTools) code is involved.

### 3.1 Receptor

```bash
# input .pdb or .pdbqt, output .pdbqt (stdout when omitted)
odock prepare receptor receptor.pdb receptor.pdbqt

# keep waters, drop every non-standard residue, do not add hydrogens, hard-fail
odock prepare receptor 4EY7.pdb rec.pdbqt --keep-water --no-hetero --no-hydrogens --strict

# holo structure: delete the co-crystallised ligand and the sulfate ion
odock prepare receptor holo.pdb apo.pdbqt --strip BEN SO4
```

| Option | Default | Meaning |
|---|---|---|
| `input` | — | input `.pdb` or `.pdbqt` file |
| `out` | stdout | output `.pdbqt` |
| `--keep-water` | off | keep crystallographic waters (`HOH`, `WAT`, `DOD`) |
| `--no-hetero` | off | drop all non-standard residues (metals, cofactors, ligands) |
| `--no-hydrogens` | off | do not add polar hydrogens |
| `--strip RESNAME [RESNAME ...]` | off | delete these residue names |
| `--strict` | off | fail instead of warning when polar hydrogens cannot be added |

Defaults: waters are **removed**, hetero residues are **kept** (metals and
cofactors usually matter for binding), and polar hydrogens are **added**.

`--strip` is the one to remember for a *holo* structure: leaving the
co-crystallised ligand in the receptor blocks the very site you want to dock
into. Residue names are compared case-insensitively, and preparation warns when
it detects a likely organic ligand left in the receptor.

Polar hydrogens are placed from chemistry rather than from implicit valences,
because a PDB file carries coordinates but no bond orders:

* **N** — one hydrogen per missing valence position (backbone amide N: one;
  lysine NZ: three; proline N, with three heavy neighbours: none);
* **O** — only for a single heavy neighbour held by a bond longer than 1.30 Å
  (hydroxyls, phenols) or for a lone oxygen (water); carbonyl oxygens get none,
  which is what keeps a hydrogen from being placed *inside* the binding site;
* **S** — one hydrogen when it has a single heavy neighbour (cysteine);
* **histidine** — only one of the two imidazole nitrogens is protonated.

From Python:

```python
import odock

receptor_mol, receptor_pdbqt, report = odock.prepare_receptor(
    "receptor.pdb",
    "receptor.pdbqt",      # pass None to skip writing
    keep_water=False,
    keep_hetero=True,
    strip=["BEN", "SO4"],  # None (default) deletes nothing
    add_polar_hydrogens=True,
    strict=False,
)
print(report.summary())
for w in report.warnings:
    print("warning:", w)
```

### 3.2 Ligand

```bash
# input .sdf/.mol/.mol2/.pdb/.smi, output .pdbqt (stdout when omitted)
odock prepare ligand ligand.sdf ligand.pdbqt --name donepezil

odock prepare ligand ligand.smi lig.pdbqt --keep-nonpolar --flexible-amides \
    --no-optimize --seed 1234
```

| Option | Default | Meaning |
|---|---|---|
| `input` | — | `.sdf`, `.mol`, `.mol2`, `.pdb`, `.smi`/`.smiles` |
| `out` | stdout | output `.pdbqt` |
| `--name` | `ligand` | name used in the `REMARK` block |
| `--keep-nonpolar` | off | keep non-polar hydrogens (do not merge them into carbon) |
| `--no-hydrogens` | off | do not add hydrogens |
| `--flexible-amides` | off | treat amide bonds as rotors |
| `--no-optimize` | off | skip the force-field pre-optimisation |
| `--smiles` | — | the ligand's SMILES, used as a bond-order template (see below) |
| `--seed` | `20240101` | embedding seed |

#### Bond orders: always pass `--smiles` for a ligand taken from a PDB

A PDB file stores atoms and coordinates but **no bond orders**: every bond reads
as a single bond. RDKit's valence model then behaves as if the ring nitrogens
were amines, and `--no-hydrogens` off would add hydrogens to them, turning
aromatic ring nitrogens into H-bond donors that do not exist. It also corrupts
the rotatable-bond count.

Pass the SMILES and the bond orders are copied from the template onto the
crystal coordinates, so aromaticity, donors and acceptors are exact:

```bash
odock prepare ligand ligand_from_pdb.pdb lig.pdbqt --smiles 'COCCOc1cc2c(cc1OCCOC)ncnc2Nc3cccc(c3)C#C'
```

Without a template the tool falls back to a conservative rule that needs only
the connectivity and the geometry (a ring nitrogen gets no hydrogen; an oxygen
gets one only when its bond is longer than a carbonyl's 1.30 Å) and says so in a
warning. Preparation always reports the **out-of-plane extent** of the result:

```text
ligand: 29 -> 30 atoms, 22 non-polar H merged, 23 H added, 11 rotatable bonds,
        3-D extent 0.90 A out of plane
```

A 2-D depiction gives ~0.00 Å. If a 2-D structure is passed in, preparation does
not silently dock a flat molecule: it re-embeds a 3-D conformer with ETKDG,
sets `reembedded`, and warns that the resulting pose is no longer the input
geometry.

What preparation does, in order:

1. read the structure (a 3-D conformer is generated with ETKDGv3 if the input has
   none, falling back to random coordinates);
2. add hydrogens if the structure has none;
3. pre-optimise with MMFF94 when all parameters are available, otherwise UFF
   (500 iterations), unless `--no-optimize`;
4. **strip non-polar hydrogens** (those bound to carbon) — the AutoDock
   united-atom convention. Polar hydrogens (bound to N, O or S) are kept,
   because the Vina force field derives H-bond donors from their presence in the
   bond graph;
5. assign AutoDock 4 atom types, donor/acceptor flags and Gasteiger-Marsili
   partial charges;
6. choose the rotatable bonds (non-ring, non-aromatic single bonds between two
   heavy atoms that each have at least one further heavy neighbour; amide bonds
   are frozen unless `--flexible-amides`);
7. build the rigid-fragment tree, choose the root as the heavy atom closest to
   the centroid, and write the PDBQT `ROOT` / `BRANCH a b` / `TORSDOF` topology.

From Python:

```python
import odock

ligand_mol, ligand_pdbqt, report = odock.prepare_ligand(
    "donepezil.sdf",
    "ligand.pdbqt",
    name="donepezil",
    add_hydrogens=True,
    strip_nonpolar=True,
    rigid_amides=True,
    embed=True,
    optimize=True,
    seed=20240101,
)
print(report.summary())
```

`report` is a `PreparationReport`:

| Attribute | Meaning |
|---|---|
| `kind` | `"ligand"` or `"receptor"` |
| `n_atoms_in`, `n_atoms_out` | atom counts before/after |
| `n_hydrogens_added` | hydrogens added |
| `n_nonpolar_hydrogens_removed` | non-polar hydrogens merged away |
| `n_rotatable_bonds` | rotatable bonds found (ligand) |
| `n_metal_atoms` | Mg/Mn/Zn/Ca/Fe atoms kept |
| `atom_order` | **RDKit atom indices in the order the kernel assigns movable atoms** |
| `warnings` | non-fatal observations |

`report.atom_order` is the key to mapping a pose back onto your RDKit molecule —
see [Poses back in RDKit](#76-poses-back-in-rdkit).

### 3.3 Reading other formats

```python
import odock

mol = odock.read_structure("ligand.mol2")     # SDF, MOL, MOL2, PDB, PDBQT, SMILES, XYZ
```

`read_structure(path, sanitize=True)` dispatches on the file extension. PDBQT
inputs are read as PDB with proximity bonding and without sanitisation.

---

## 4. Choosing the search box

The box is where the ligand is allowed to go. Docking only searches inside it;
anything outside is pushed back by a linear penalty (slope 1e6 kcal/mol/Å), which
is a wall, not a soft restraint.

Three equivalent ways to specify it:

```bash
# 1. a JSON file written by `odock box`
odock box --ligand ligand.pdbqt --out box.json --buffer 8
odock dock -r receptor.pdbqt -l ligand.pdbqt --box box.json -o poses.pdbqt

# 2. explicit centre and edge lengths
odock dock -r receptor.pdbqt -l ligand.pdbqt \
    --center 10.5 22.0 16.5 --size 22.5 22.5 22.5 -o poses.pdbqt

# 3. derived from the ligand's own input position (blind-ish docking)
odock dock -r receptor.pdbqt -l ligand.pdbqt --box-from-ligand --buffer 8 -o poses.pdbqt
```

### 4.1 `odock box`

| Option | Default | Meaning |
|---|---|---|
| `--ligand` | — | centre the box on this structure |
| `--receptor` | — | derive the box from this structure (all heavy atoms) |
| `--residue` | — | centre the box on this residue number (needs `--receptor`) |
| `--chain` | — | restrict `--residue` to this chain |
| `--buffer` | `5.0` | padding in Å |
| `--spacing` | `0.375` | grid spacing in Å |
| `--out` | — | also write the box as JSON |

If `--receptor` and `--residue` are given, the selection is that residue (and
that chain); if only `--receptor` is given, the box covers the whole receptor.
The JSON is printed to stdout and, with `--out`, also written to the file:

```json
{
  "center": [-1.8555, 14.366, 16.748],
  "size": [17.883, 19.95, 20.514],
  "spacing": 0.375
}
```

### 4.2 Box geometry and how it is rounded

The requested `size` is rounded **up** to whole voxels of `spacing` on every
axis, so the realised box is never smaller than you asked for. With the default
0.375 Å spacing, a 22.5 Å box becomes 60 voxels per axis (22.5 Å exactly).

Advice:

* Use `--buffer 8` around a reference ligand when you know the binding site (this
  is what the 3PTB validation uses). `buffer=5` (the default) is tighter and
  faster.
* The ligand's *input* position must be inside the box, otherwise the search
  starts inside a penalty wall. `odock dock` prints the box it uses; the
  per-pose `in_box` flag tells you whether the final pose stayed inside.
* Bigger boxes are not free: the grid memory and build time grow with the
  volume, and the conformational space grows with it. 20–25 Å per axis is the
  usual working range for a known site.
* `spacing` is a grid-resolution knob. Leave it at 0.375 Å unless you have a
  reason; the exact refinement after the search mitigates the interpolation
  error, but the search itself is only as accurate as the grid.

From Python:

```python
import odock

box = odock.box_from_ligand(ligand_mol, buffer=8.0)             # around a ligand
box = odock.box_from_selection(receptor_mol,                    # around a residue
                               lambda a: a.GetPDBResidueInfo() is not None
                                         and a.GetPDBResidueInfo().GetResidueNumber() == 337,
                               buffer=6.0)
box = odock.box_from_points(coords_n_by_3, buffer=5.0)          # around any points
box = odock.box_from_smiles_ligand("c1ccccc1C(=O)O", buffer=8.0)  # blind docking
print(box.center, box.size, box.spacing, box.volume)
```

`BoxSpec` is a frozen dataclass `(center, size, spacing=0.375)` that validates
itself: a non-positive `size` or `spacing` raises `ValueError` immediately. It
also offers `corner1`, `corner2`, `contains(point, margin=0.0)` and `as_dict()`.

---

## 5. The command line

```text
odock [-h] [--version] {info,prepare,box,dock,score,split,gui} ...
```

| Subcommand | Purpose |
|---|---|
| `odock info` | kernel version, Python/NumPy/RDKit versions, force fields, GPU status |
| `odock prepare receptor` | prepare a rigid receptor PDBQT |
| `odock prepare ligand` | prepare a ligand PDBQT with its rotatable-bond tree |
| `odock box` | build a search box and print/write it as JSON |
| `odock dock` | run a docking |
| `odock score` | score a ligand at its input position (no search) |
| `odock split` | split a multi-model PDBQT into one file per model |
| `odock` (no arguments) | **launch the 3-D workbench** — the default entry point |
| `odock gui` | launch the 3-D workbench (same thing, explicit) |

### 5.1 `odock dock`

| Option | Default | Meaning |
|---|---|---|
| `-r`, `--receptor` | required | receptor PDBQT |
| `-l`, `--ligand` | required | ligand PDBQT |
| `-o`, `--out` | — | write the poses as a multi-model PDBQT |
| `--json-out` | — | write the poses as JSON |
| `--box` | — | box JSON written by `odock box` |
| `--box-from-ligand` | off | derive the box from the ligand's input position |
| `--center X Y Z` | — | explicit box centre |
| `--size X Y Z` | — | explicit box edge lengths |
| `--buffer` | `5.0` | padding for `--box-from-ligand` |
| `--spacing` | `0.375` | grid spacing in Å |
| `-s`, `--scoring` | `vina` | `vina`, `vinardo` or `ad4` |
| `-e`, `--exhaustiveness` | `8` | number of independent Monte-Carlo runs |
| `-n`, `--num-poses` | `9` | maximum number of poses to report |
| `--seed` | `0` | RNG seed; `0` draws one from the OS |
| `--min-rmsd` | `1.0` | poses closer than this are considered identical |
| `--energy-range` | `3.0` | reporting window in kcal/mol (for `-o`) |
| `--no-grid` | off | use the exact scorer during the search too |
| `--no-refine` | off | skip the exact post-refinement |
| `--ga` | off | use the island-model genetic algorithm |
| `--islands` | `4` | number of GA islands |
| `--population` | `32` | GA population per island |
| `--generations` | `20` | GA generations per island |
| `-q`, `--quiet` | off | send the results table to stderr instead of stdout |

One of `--box`, `--box-from-ligand`, or `--center` + `--size` must be given:

```text
error: no search box: pass --box FILE, or --center X Y Z --size X Y Z,
       or --box-from-ligand with --buffer
```

A complete run:

```bash
odock dock -r receptor.pdbqt -l ligand.pdbqt --box box.json \
    -o poses.pdbqt --json-out poses.json -s vina -e 16 -n 9 --seed 42
```

Stderr carries the progress lines, stdout the table. An actual run (the 3PTB
system, `--box-from-ligand --buffer 8 -e 2 --seed 7`, no `-o`) prints:

```text
# stderr
box: BoxSpec(center=(-1.86, 14.37, 16.75), size=(17.9, 20.0, 20.5) Å, V=7319 Å³)
search: 13 movable atoms, 1 torsions, N_tors=1, grid 150920 points (3 MB), seed=7, 2.01s

# stdout
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -8.009      0.000      0.000
   2       -7.979      0.088      1.601
   3       -6.955      2.721      3.535
   4       12.513      9.134     10.096

best affinity: -8.009 kcal/mol
```

`grid 150920 points` is the number of grid *sample points* (`n_x * n_y * n_z`);
the reported `(3 MB)` is the memory of all per-type maps
(`n_x * n_y * n_z * n_types` `f64` values). A `wrote <file>` line appears on
stderr when `-o` is used.

> **Note on the README quick start.** The README shows `odock dock ... -v`.
> The current CLI has no `-v`; verbose output is the default and `-q/--quiet` is
> the way to suppress the table on stdout.

> **Note on `--json-out`.** It writes `{seed, elapsed, box, grid_points,
> poses}` with one dictionary per pose, produced by `Pose.to_dict()`. The
> coordinate array is excluded (it is large and recoverable with `pose_to_mol`
> or `to_pdbqt`); pass `include_coords=True` to `to_dict()` if you want it as a
> nested list. See [JSON output](#75-json-output).

### 5.2 `odock score`

| Option | Default | Meaning |
|---|---|---|
| `-r`, `--receptor` | required | receptor PDBQT |
| `-l`, `--ligand` | required | ligand PDBQT |
| `--box` | — | box JSON (only used for the out-of-box penalty) |
| `--box-from-ligand` | off | derive the box from the ligand |
| `--center X Y Z`, `--size X Y Z` | — | explicit box |
| `--buffer` | `5.0` | padding for `--box-from-ligand` |
| `--spacing` | `0.375` | grid spacing |
| `--scoring` | `vina` | `vina`, `vinardo` or `ad4` |
| `--json` | off | print the components as JSON |

With no box at all, a huge (200 Å) box is used so that the out-of-box penalty is
effectively zero for any realistic structure and the number is the pure
interaction energy.

```bash
odock score -r receptor.pdbqt -l ligand.pdbqt
```

For the prepared 3PTB ligand at its crystallographic position:
```text
affinity          :     -7.659 kcal/mol
  inter           :     -8.106
  intra           :     -0.044
  conf-independent:      0.448
  unbound         :     -0.044
```

`score` does **not** search: it evaluates the ligand at the coordinates in the
ligand file. A multi-model pose file is accepted too: the reader stops at the end
of the first model, so `odock score` reports the score of mode 1. Split the file
when you want to score the other modes:

```bash
odock split poses.pdbqt --outdir poses
odock score -r receptor.pdbqt -l poses/poses_1.pdbqt
```

or score one model in memory with the Python API — see
[§7.4](#74-scoring-without-searching).

### 5.3 `odock split`

```bash
odock split poses.pdbqt --outdir poses
```

Splits a multi-model PDBQT into `poses/poses_1.pdbqt`, `poses/poses_2.pdbqt`, …
(`--outdir` defaults to `poses`).

### 5.4 The 3-D workbench — `odock` on its own

`odock` **with no arguments opens the workbench**. That is the default entry
point for interactive use; the subcommands are what you reach for when
scripting.

```bash
odock                                            # empty workbench, load from the UI
odock receptor.pdbqt poses.pdbqt                 # open straight on those structures
odock poses.pdbqt                                # browse poses on their own
odock gui -r receptor.pdbqt -l ligand.pdbqt -p poses.pdbqt   # the explicit form
```

The bare-file form inspects each file and does the obvious thing: a file with
several `MODEL` records is loaded as poses, a file with a `ROOT`/`BRANCH`
topology as a ligand, anything else as a receptor. Up to three files are
accepted; anything that is not an existing file is left to the normal
command-line parsing, so a typo still produces the usual "invalid choice"
message rather than a surprise window.

`odock gui` is the same launcher with explicit options:

| Option | Meaning |
|---|---|
| `-r`, `--receptor` | receptor PDBQT to load |
| `-l`, `--ligand` | ligand PDBQT to load |
| `-p`, `--poses` | pose PDBQT written by `odock dock` |

The workbench is `odock.gui` (a package: `structure`, `viewport` and the
PyQt6 `app`), imported lazily so that a headless install still imports `odock`
cleanly. If PyQt6 / ModernGL are missing, the command reports:

```text
error: the GUI needs PyQt6 and ModernGL (<...>).
       install them with: pip install 'opendocking[gui]'
```

The GUI is a viewer *and* a driver: it loads a receptor, a ligand and optionally
a pose file, shows the search box in 3-D (edited from the Grid panel, not by
dragging it), and runs the same Rust kernel the CLI and the Python API use.
Everything in it is optional — the CLI and the Python API never import it.

**Mouse and keyboard**

| input | effect |
|---|---|
| left drag | orbit the camera |
| right drag | pan |
| wheel | zoom |
| left click on an atom | select its residue (shift/ctrl adds) |
| hover an atom | the read-out card, bottom right, names it |
| double click | centre the camera on that residue |
| pose slider / table row | show that mode |
| **ctrl** + click a second table row | compare the two poses (see 5.5) |
| `Ctrl+K` | command palette |
| `Ctrl+Shift+C` | copy the 3-D view to the clipboard |
| `Ctrl+Shift+R` | restore the saved session |
| `Home` | reset the view |

The search box is edited from the **Grid** tab (centre, size, spacing) and the
Grid menu; a drag in the 3-D view orbits or pans and never moves the box, so a
mis-aimed drag can never change what is being docked.

**Rendering note.** The viewport renders with a ModernGL context that it owns
and blits the image into the widget, rather than sharing Qt's GL context. That
is deliberate: a shared context let Qt's own GL state (`glColorMask(1,0,0,0)`
and its blend function) discard the geometry, which produced a window with an
empty 3-D area on some drivers. It also means the widget needs no display to
render, so `tests/test_gui_viewport.py` can check it head-lessly.

The `Renderer` draws atoms as *impostor spheres* — one camera-facing quad each,
with the sphere ray-cast in the fragment shader and `gl_FragDepth` written from
the analytic surface, so spheres interpenetrate correctly. A 3 000-atom receptor
plus its ligand renders in a few milliseconds.

If OpenGL 3.3 is unavailable the 3-D area shows an explanatory message and the
rest of the workbench (loading, scoring, docking, the pose table) keeps working.

### 5.5 The instrument panels

The panels below are what turn the window from a viewer into something you can
read a run from. Everything here is available from the menus; nothing needs a
config file or a hidden gesture.

#### The run monitor

`Docking ▸ Start` (F5) fills the **Run monitor**, docked to the right of the
pose table. It shows:

* the **phase readout** — `Grid › Search › Refine › Done` — with a tick on the
  stages that are finished and the length each one actually took. The running
  stage counts up as it goes; a finished stage keeps its measured length. A
  stage's length is always the *difference* between two clock marks, never a
  mark itself, so a number can never sit under the wrong label;
* a **live elapsed clock**, the engine and exhaustiveness in use, the number of
  reported poses and the grid size (points and MB);
* the **best affinity of the session**, with the run number it came from;
* the **energy trace**: affinity against pose, one point per reported pose, the
  newest run bright, earlier runs dimmed for comparison, and the best-so-far as
  a dashed rule.

**What the trace is, exactly.** The kernel returns its refined pose energies
*once, at the end of a run* (`Docking::run` holds its state for the whole
search), so there is no per-iteration energy to plot and the panel does not
invent one. The points you see are the real reported affinities, drawn the
moment each run returns; the phase chips, the clock and the status line are
genuinely live while the search runs. The panel says so on screen under the
plot, and `python/odock/gui/dashboard.py` records the honest route to a live
convergence curve (a progress callback out of the Rust search) as a TODO.

A cancelled run is still recorded, labelled as cancelled, and its partial pose
list is drawn.

#### Comparing two poses

**Ctrl-click a second row** in the results table. The **Pose comparison** panel
opens beside the Inspector with the answer to "which of these two should I
believe?":

* **Symmetric-aware RMSD** — `fitted` (after optimal superposition, the number
  that answers "same binding mode or not?") and `in place` (no superposition,
  the crystallographic figure of merit). Symmetry-equivalent atoms of the same
  element are allowed to stand in for one another, so a benzene flipped by one
  carbon is not reported as a 2.8 Å error.
* **Affinity difference** — the two affinities and their difference, with a
  same/different binding-mode verdict (≤ 2 Å fitted *and* mostly the same
  contacts).
* **Contact fingerprint** — every receptor residue within 4.5 Å of the ligand,
  split into *both*, *only the first* and *only the second*, each with the
  closest atom pair and its distance.

The same split appears on the **sequence ruler** under the 3-D view: green
where both poses touch a residue, red where only one does. Outside a comparison
the ruler marks the displayed pose's contacts in amber, so "what does this pose
touch?" is a glance rather than a table. `Analysis ▸ Compare poses` repeats the
comparison for the current selection, and **Copy report** puts the whole thing
on the clipboard as plain text.

#### The command palette (`Ctrl+K`)

Type a few letters of what you want — `cmp`, `box`, `csv` — and press Enter.
The palette is rebuilt **from the menu bar every time it opens**, so it can
never go stale: every entry it offers is a real action with its real shortcut,
and a new menu entry appears in it immediately. Menu paths are shown as
tooltips, the arrow keys move the highlight, and the footer counts the commands
found.

#### The session

The workbench remembers where you were. It autosaves the loaded files, the
search box, the engine settings, the selected residues, the pose you were
looking at, the theme, the dock arrangement and the splitter sizes, and writes
the file again whenever you close the window.

* On launch it **offers** the last session — a question when a display can
  answer it, otherwise a line in the log and `File ▸ Restore session`
  (`Ctrl+Shift+R`).
* `File ▸ Recent files` lists what you opened lately (newest first, files that
  have since been deleted are not offered) and reopens the right structure:
  a `MODEL` file as poses, a `ROOT` file as a ligand, anything else as a
  receptor.
* A session with nothing loaded never overwrites the saved one, so a bare
  launch that you close again cannot destroy the session it just offered.
* The file lives at `%APPDATA%\OpenDocking\session.json` on Windows and
  `~/.config/opendocking/session.json` elsewhere. Set `ODOCK_SESSION_FILE` to
  put it somewhere else.

#### Theme, density and layout

`View ▸ Theme` chooses **Dark** or **Light**, and `View ▸ Density` chooses
**Comfortable** or **Compact** (font size, control padding, table padding and
tab sizes). The 3-D view has its own background colour per theme — a light
window around a near-black viewport reads as a hole in the window — and so does
the sequence ruler, which paints itself.

`View ▸ Layout` arranges the windows for a job in one click:

| preset | what it opens |
|---|---|
| **Docking** | workspace, inspector on the Engine tab, pose table and run monitor |
| **Analysis** | inspector on Receptor, pose table, selection panel; workspace and dashboard closed |
| **Compare** | pose table, run monitor and the pose-comparison panel, everything else closed |

Every dock can also be toggled individually from `View ▸ Panels`.

#### Loading structures by drag and drop

Drop a `.pdbqt` or `.pdb` file anywhere on the window. The file is routed by
what it *says* it is, never by its name: a `MODEL` record or a `VINA RESULT`
remark means poses, a `ROOT` block means a flexible ligand, anything else is a
receptor. Several files can be dropped at once, and a file the workbench cannot
read is reported in the log and a message box instead of being silently
ignored.

#### Measurements, the atom read-out and screenshots

* The **measure tool** (View ▸ Measure) turns two atom clicks into a distance,
  and the **Measurements** tab of the run monitor lists them with their atoms
  as a table you can copy or clear. They are also drawn in the 3-D view and
  listed in the workspace tree.
* **View ▸ Inspect atom** toggles the hover read-out: point at any atom and a
  small card names its residue, atom, element, AD4 type, charge and position,
  and the status bar repeats it. It is on by default and costs nothing when the
  cursor is still.
* `View ▸ Copy view` (`Ctrl+Shift+C`) puts the current 3-D image — including its
  legend — on the clipboard, ready to paste into a slide. `File ▸ Export ▸
  Screenshot…` still writes a high-resolution PNG to disk.

---

## 6. Understanding the output

### 6.1 The results table

The first four modes of the recorded 3PTB validation run
(`--exhaustiveness 16 --seed 42`, six modes written in total):

```text
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -7.991      0.000      0.000
   2       -7.797      0.202      1.608
   3       -7.307      2.614      3.602
   4       -7.007      2.276      3.267
```

* **mode** — pose rank, best first. Modes are deduplicated: two poses closer
  than `--min-rmsd` (1.0 Å) count as one, and the better one is kept.
* **affinity** — the reported binding affinity in kcal/mol. More negative means
  better predicted binding. It is `ScoreComponents::total`, i.e. the Vina
  convention `(E_inter + E_intra - E_unbound) / (1 + w_rot * N_tors)`, or for
  AD4 `E_inter + w_rot * N_tors`. See
  [SCORING.md](SCORING.md#torsional-penalty-and-the-reported-affinity).
* **rmsd l.b.** — permutation-invariant heavy-atom RMSD to the best pose, a lower
  bound on the true symmetry-corrected RMSD ("lower bound").
* **rmsd u.b.** — heavy-atom RMSD to the best pose with a fixed atom-to-atom
  correspondence ("upper bound"). The best pose is 0.000/0.000 by definition.

The two RMSD columns compare poses *to each other*, not to any experimental
structure. To compare with a crystal pose, use `odock.aligned_rmsd` (§7.6).

The CLI additionally prints `best affinity: -7.991 kcal/mol` unless `-q` is
given.

### 6.2 The pose file

Each `MODEL` carries a `REMARK` block:

```text
MODEL 1
REMARK VINA RESULT:       -7.991      0.000      0.000
REMARK INTER + INTRA:        -8.501
REMARK INTER:                -8.458
REMARK INTRA:                -0.043
REMARK CONF_INDEPENDENT:      0.467
REMARK UNBOUND:              -0.043
REMARK OPEN DOCKING MODE 1
ROOT
ATOM ... (the root rigid fragment)
ENDROOT
BRANCH    2   3
ATOM ... (the branch; atom 3 stays rigid relative to the parent frame)
ENDBRANCH    2   3
TORSDOF 1
ENDMDL
```

| Field | Meaning |
|---|---|
| `VINA RESULT` | affinity, `rmsd l.b.`, `rmsd u.b.` — the same three numbers as the table |
| `INTER + INTRA` | the raw sum of the intermolecular and intramolecular terms |
| `INTER` | ligand–receptor energy |
| `INTRA` | ligand internal energy (intra-ligand pairs plus macrocycle glue pairs) |
| `CONF_INDEPENDENT` | `total - (inter + intra - unbound)`: the shift produced by the torsional rule |
| `UNBOUND` | the ligand's internal energy in isolation; equal to `INTRA` for Vina/Vinardo, `0` for AD4 |
| `OPEN DOCKING MODE n` | pose index |

`--energy-range` (default 3.0) controls which poses are written: modes more than
that many kcal/mol above the best one are omitted from the file, even if
`--num-poses` is larger.

The writer re-emits the original `ROOT`/`BRANCH`/`TORSDOF` topology, so the pose
file is a valid PDBQT that can be re-docked.

---

## 7. The Python API

Everything the CLI does is available from Python; the CLI is a thin wrapper.

### 7.1 A complete run

```python
import odock

# --- prepare ---------------------------------------------------------------
receptor_mol, receptor_pdbqt, rec_report = odock.prepare_receptor(
    "receptor.pdb", "receptor.pdbqt", keep_water=False
)
ligand_mol, ligand_pdbqt, lig_report = odock.prepare_ligand(
    "ligand.sdf", "ligand.pdbqt", name="ligand"
)
print(rec_report.summary())
print(lig_report.summary())

# --- box -------------------------------------------------------------------
box = odock.box_from_ligand(ligand_mol, buffer=8.0)
print(box)

# --- dock ------------------------------------------------------------------
result = odock.dock(
    "receptor.pdbqt",
    "ligand.pdbqt",
    box,
    scoring="vina",
    exhaustiveness=16,
    num_poses=9,
    seed=42,
)

print(result.summary())              # metadata + table
print(result.best_affinity)          # -7.991 (or None)
open("poses.pdbqt", "w").write(result.to_pdbqt())
open("poses.pdb", "w").write(result.to_pdb())      # PDB view, REMARKs kept
```

`dock(receptor, ligand, box, *, ...)` accepts either a path or the file's *text*
(it decides by sniffing the first record), so PDBQT strings work directly:

```python
result = odock.dock(receptor_text, ligand_text, box, exhaustiveness=8, seed=42)
```

Full signature:

```python
odock.dock(
    receptor, ligand, box, *,
    scoring="vina",            # "vina" | "vinardo" | "ad4"
    exhaustiveness=8,
    num_poses=9,
    seed=0,                    # 0 = draw a seed from the OS
    use_grid=True,
    refine=True,
    min_rmsd=1.0,
    energy_range=3.0,
    use_island_ga=False,
    islands=4,
    population=32,
    generations=20,
    global_steps=None,         # None = Vina's heuristic
    local_steps=None,          # None = (25 + n_atoms)/3
) -> odock.DockResult
```

Also available: `odock.dock_text(receptor_pdbqt, ligand_pdbqt, box, **kwargs)`,
which is `dock` with the intent made explicit.

### 7.2 `DockResult`

| Attribute / method | Meaning |
|---|---|
| `poses` | list of `Pose`, best first |
| `best()` | the best `Pose`, or `None` |
| `best_affinity` | the best affinity in kcal/mol, or `None` |
| `within(energy_range=3.0)` | poses within that window of the best |
| `table(energy_range=None)` | the results table as a string |
| `summary()` | metadata line + best affinity + table |
| `to_pdbqt()` | the poses as a multi-model PDBQT document |
| `to_pdb()` | the same poses as a multi-model PDB document |
| `seed` | the seed actually used (never 0) |
| `grid_mb`, `grid_points` | affinity-grid memory (MB) and sample-point count (0 when unused) |
| `num_tors` | the `N_tors` value that entered the torsional penalty |
| `num_movable_atoms` | movable atoms of the system |
| `num_dof` | number of torsions (excluding the rigid body) |
| `exact` | whether the reported energies were refined on the exact surface |
| `receptor_pdbqt`, `ligand_pdbqt` | the exact texts that were docked |
| `box` | the `BoxSpec` used |
| `elapsed` | wall time of the run, in seconds |
| `ligand_atom_order` | ligand atom names in the kernel's internal order |

| `Pose` attribute | Meaning |
|---|---|
| `index` | 0-based pose index |
| `affinity` | reported affinity (kcal/mol) |
| `rmsd_lower_bound`, `rmsd_upper_bound` | the two RMSD columns |
| `inter`, `intra`, `conf_independent`, `unbound` | the energy decomposition |
| `in_box` | whether the ligand is inside the search box |
| `num_atoms` | ligand atoms (including hydrogens) |
| `position`, `orientation`, `torsions` | the raw conformation: translation, quaternion `(x, y, z, w)`, torsion angles in radians |
| `coords` | `(n_atoms, 3)` NumPy array in the kernel's internal atom order |

### 7.3 Inspecting the poses

```python
for pose in result.poses:
    print(f"mode {pose.index + 1}: {pose.affinity:8.3f} kcal/mol  "
          f"rmsd_lb {pose.rmsd_lower_bound:5.3f}  in_box={pose.in_box}")

best = result.best()
print("translation", best.position)
print("quaternion ", best.orientation)
print("torsions   ", best.torsions)
print("coords     ", best.coords.shape)      # (n_atoms, 3) float64
```

`Pose.coords` is in the kernel's internal atom order, which is what
`result.ligand_atom_order` lists; `Docking.pose_coords(i)` (the low-level
extension API) returns the same data.

### 7.4 Scoring without searching

```python
import odock

components = odock.score("receptor.pdbqt", "ligand.pdbqt")
for name, value in components.items():
    print(f"{name:18s} {value:10.3f}")
```

```text
affinity              -7.659
total                 -7.659
inter                 -8.106
intra                 -0.044
conf_independent       0.448
unbound               -0.044
```

```python
odock.score(
    receptor, ligand,
    box=None,          # None = no out-of-box penalty (a 200 Å box is used)
    *,
    scoring="vina",
    refine=True,
) -> Dict[str, float]
```

`score` evaluates the ligand exactly where the file puts it. To score one model
out of a pose file, pass just that model's text:

```python
text = open("poses.pdbqt").read().splitlines()
start = next(i for i, l in enumerate(text) if l.startswith("MODEL"))
end = next(i for i, l in enumerate(text) if l.startswith("ENDMDL"))
model1 = "\n".join(text[start + 1:end]) + "\n"

print(odock.score(open("receptor.pdbqt").read(), model1))
# {'affinity': -7.991, 'total': -7.991, 'inter': -8.458, 'intra': -0.043,
#  'conf_independent': 0.467, 'unbound': -0.043}
```

For a single pairwise interaction — useful for teaching and for checking a force
field — use `score_pair`, which takes X-Score type names from `odock.XS_TYPES`:

```python
import odock

print(odock.XS_TYPES[:10])            # ['CH', 'CP', 'NP', 'ND', 'NA', 'NDA', 'OP', 'OD', 'OA', 'ODA']
print(odock.score_pair("CH", "CH", 3.8))              # -0.035579 kcal/mol (gauss1 well)
print(odock.score_pair("ND", "OA", 2.8))              # includes the H-bond plateau
print(odock.score_pair("CH", "CH", 3.3, "vinardo"))
```

### 7.5 JSON output

`Pose.to_dict(include_coords=False)` returns the pose's fields with the two RMSD
keys renamed (`rmsd_lb`, `rmsd_ub`) and the coordinate array omitted, so the
result is JSON-serialisable straight away:

```python
import json

payload = {
    "seed": result.seed,
    "elapsed": result.elapsed,
    "box": result.box.as_dict(),
    "grid_points": result.grid_points,
    "num_tors": result.num_tors,
    "poses": [p.to_dict() for p in result.poses],
}
open("poses.json", "w").write(json.dumps(payload, indent=2) + "\n")
```

`to_dict()` keys: `index`, `affinity`, `rmsd_lb`, `rmsd_ub`, `inter`, `intra`,
`conf_independent`, `unbound`, `in_box`, `num_atoms`, `position`,
`orientation`, `torsions`. Pass `include_coords=True` to add `coords` as a
nested list — useful, but it makes the document large.

`odock dock --json-out FILE` writes exactly this document (minus `num_tors`,
which the CLI payload does not include); the coordinates always come from the
PDBQT written with `-o`.

### 7.6 Poses back in RDKit

`report.atom_order` from `prepare_ligand` maps the kernel's internal atom order
onto RDKit indices, and `pose_to_mol` uses it:

```python
import odock
from rdkit.Chem import AllChem

ligand_mol, ligand_pdbqt, lig_report = odock.prepare_ligand("ligand.sdf")
box = odock.box_from_ligand(ligand_mol, buffer=8.0)
result = odock.dock("receptor.pdbqt", ligand_pdbqt, box, exhaustiveness=16, seed=42)

for pose in result.poses:
    mol = odock.pose_to_mol(pose, ligand_mol, lig_report.atom_order)
    mol.SetProp("_Name", f"pose_{pose.index + 1}_{pose.affinity:.3f}")
    AllChem.MolToMolFile(mol, f"pose_{pose.index + 1}.sdf")
```

`pose_to_mol(pose, template_mol, atom_order)` returns a copy of `template_mol`
with a fresh conformer holding the pose; it raises `ValueError` if the pose has
no coordinates or if `atom_order` does not match the pose's atom count (which
means the ligand was re-prepared).

To compare a pose with a crystal structure, use `aligned_rmsd`:

```python
rmsd_no_fit, rmsd_fitted = odock.aligned_rmsd(pose_mol, crystal_mol, heavy_only=True)
print(f"{rmsd_no_fit:.3f} Å without superposition, {rmsd_fitted:.3f} Å after")
```

The first number is the crystallographic figure of merit — no alignment is
allowed, because the docking must reproduce the experimental pose *in the same
frame*. The second is RDKit's symmetry-corrected RMSD after optimal
superposition. `tests/validate_3ptb.py` shows the full pattern.

### 7.7 Version, GPU and type introspection

```python
import odock

print(odock.kernel_version())      # '0.1.0' — the Rust kernel version
print(odock.__version__)           # the same string
print(odock.gpu_available())       # False unless the kernel was built with `gpu`
print(odock.gpu_description())     # a human-readable explanation
print(odock.XS_TYPES)              # the 32 X-Score type names, indexed by type id
```

The low-level extension is available as `odock._odock` and exposes
`ScoringFunction`, `Weights`, and the module functions `version`,
`gpu_available`, `gpu_description`, `xs_type_names`, `ligand_info`,
`receptor_info`, `pair_energy`.

```python
from odock import _odock as core

sf = core.ScoringFunction("vina")
print(sf.name, sf.cutoff, sf.max_cutoff, sf.num_terms, sf.grid_capable, sf.rot)
print(core.Weights.default_for("ad4").terms)   # [0.1662, 0.1209, 0.1406, 0.1322, 50.0]
print(core.ligand_info(open("ligand.pdbqt").read()))    # (n_atoms, n_rotors, torsdof)
print(core.receptor_info(open("receptor.pdbqt").read()))  # n_atoms
```

Custom term weights are **not** exposed through the Python API in this release:
`DockOptions::weights` is reachable only from Rust.

---

## 8. Tuning

### 8.1 `exhaustiveness`

The number of independent Monte-Carlo runs. Each run randomises the ligand inside
the box, minimises, mutates, minimises again and accepts or rejects with the
Metropolis criterion, independently of the others; the runs are distributed over
all cores with `rayon`.

* Default 8. Vina's own default is 8 as well.
* Wall time scales roughly linearly with `exhaustiveness` divided by the number
  of cores; the *result* improves with the chance of finding the global minimum.
* Practical guidance: 8 for a well-defined site and a rigid ligand, 16–32 for a
  flexible ligand or a blind-ish box, more only if you can afford it.
* Because the runs are independent, increasing `exhaustiveness` never makes a
  previously found pose worse — it can only add better ones (up to
  `num_poses`).

### 8.2 `seed`

* `seed=0` (the default) draws a seed from the OS entropy pool and reports it in
  `result.seed`; record it to reproduce the run later.
* Any other value makes the run **exactly** reproducible: same seed, same
  parameters, same platform → same pose list, bit for bit. The RNG is a
  self-contained PCG-XSS-RR, not a library version's stream.
* Reproducing the documented validation is a one-liner:

```bash
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42
```

### 8.3 Choosing a force field

| `--scoring` | Terms | Grid | When to use |
|---|---|---|---|
| `vina` (default) | gauss1, gauss2, repulsion, hydrophobic, H-bond, glue | yes | the baseline; matches AutoDock Vina's own scoring function |
| `vinardo` | gauss, repulsion, hydrophobic, H-bond, glue | yes | an alternative re-parameterisation; often sharper on small polar ligands |
| `ad4` | 12-6 vdW, 12-10 H-bond, screened electrostatics, desolvation | **no** | when you need AutoDock 4.2 physics (charges, desolvation); noticeably slower because every evaluation is exact |

AD4 specifics:

* The grid is never built for AD4 (`is_grid_capable()` is false), so `--no-grid`
  is implied and the search itself runs on the exact scorer.
* AD4 is the only force field that uses partial charges. The preparation layer
  writes Gasteiger-Marsili charges into PDBQT columns 71–76; a ligand file
  without them will show zero electrostatics.
* AD4's torsional term is **additive** (`E + 0.2983 * N_tors`), not a divisor,
  and its ligand internal energy is not subtracted from the reported affinity.

### 8.4 Poses and reporting

* `-n/--num-poses` (default 9) is the size of the pose container. Fewer poses
  means less bookkeeping and less RMSD work, not a worse search.
* `--min-rmsd` (default 1.0 Å) is the deduplication threshold. Lower it to see
  near-duplicate minima; raise it to compress the table.
* `--energy-range` (default 3.0 kcal/mol) only affects which poses are *written*
  by `-o`.
* `--no-refine` skips the exact post-refinement; the reported energies then come
  from the grid and `result.exact` is `False`. Useful for a fast scan, not for
  numbers you intend to publish.
* `--no-grid` uses the exact scorer inside the search as well. Much slower, and
  only worth it to confirm that a suspicious grid value is not an interpolation
  artefact.

### 8.5 Advanced search controls

| Control | Where | Default | Effect |
|---|---|---|---|
| `global_steps` | Python / Rust only | `None` → `70*3*(50 + n_atoms + 10*n_dof)/2` | number of Monte-Carlo steps per run |
| `local_steps` | Python / Rust only | `None` → `(25 + n_atoms)/3` | BFGS steps per local minimisation |
| `use_island_ga` | CLI `--ga`, Python | off | island-model Lamarckian genetic algorithm instead of parallel MC |
| `islands`, `population`, `generations` | CLI `--islands/--population/--generations`, Python | 4 / 32 / 20 | GA size |
| `elites` | Rust only | 2 | elites carried over per generation |
| `temperature` | Rust only | 1.2 kcal/mol | Metropolis temperature (= 600 K with R = 2 cal/mol/K) |
| `mutation_amplitude` | Rust only | 2.0 Å | size of the one-DOF mutation |
| `max_evals` | Rust only | 0 (unlimited) | hard cap on energy evaluations per run |

The `global_steps`/`local_steps` overrides are `None` by default, which selects
Vina's own heuristics; supply a number only to force a specific budget. Note that
`temperature`, `mutation_amplitude`, `max_evals` and `elites` are **not** plumbed
through to the Python API in this release — set them from Rust if you need them.

---

## 9. GPU notes

* The GPU backend is the optional `gpu` feature of the **`dock-core`** crate
  (`wgpu` 25 with the Vulkan / Metal / Direct3D 12 / WGSL backends, plus
  `bytemuck` and `pollster`). It is off by default so that a plain build never
  needs a graphics driver.
* It accelerates **affinity-grid construction only** — a pure gather over
  sample points. The conformational search stays on the CPU, where the workload
  is latency-bound and branchy; the CPU search already uses every core through
  `rayon`.
* There is no CUDA dependency. It needs a working Vulkan, Metal or Direct3D 12
  driver, or it is simply not used.
* **The Python wheel built from this configuration cannot enable it.**
  `pyproject.toml` builds `crates/dock-py`, and that crate declares no features
  of its own and does not forward `dock-core/gpu`, so `odock.gpu_available()` is
  `False` and `odock dock` always builds its grid on the CPU. This is a
  configuration limitation, not an error.
* To exercise the GPU path, build the kernel from Rust:

```bash
cargo build -p dock-core --features gpu
cargo test  -p dock-core --features gpu     # the tests use the CPU fallback
```

  The WGSL kernel is `crates/dock-core/src/gpu/grid.wgsl`; `gpu::populate_grid`
  tries it first and falls back to the CPU implementation whenever there is no
  adapter, the device is lost, or the force field is not grid-capable (AD4),
  printing one line to stderr.
* `odock info` and `odock.gpu_description()` tell you exactly where you stand.

```text
gpu : GPU backend unavailable: gpu error: dock-core was built without the
      `gpu` feature; rebuild with `cargo build --features gpu` to enable the
      wgpu backend
```

* The CPU grid builder and the GPU grid builder evaluate the same terms at the
  same sample points, so results do not depend on which one ran (modulo `f32`
  rounding).

---

## 10. Troubleshooting

### `ImportError: RDKit is required for PDBQT preparation.`

```text
RDKit is required for PDBQT preparation. Install it with
`pip install rdkit` (or `pip install opendocking[chem]`).
```

Preparation, box construction from a molecule and reading structure files all
need RDKit. `odock dock` on existing PDBQT files does **not** need it, but the
CLI's `--box-from-ligand` does (it reads the ligand with RDKit to find its
position), so install the `[chem]` extra if you want that path.

### `ModuleNotFoundError: No module named 'odock._odock'`

The compiled extension is not installed in the active interpreter. Activate the
right environment and rebuild:

```bash
maturin develop --release
```

If you switched virtual environments or upgraded Python, rebuild again — the
extension is version-specific.

### `UnicodeEncodeError: 'gbk' codec can't encode character '\xc5'`

Seen on a Windows console whose code page cannot represent `Å` (the force-field
lines of `odock info` contain it):

```text
File ".../odock/cli.py", line 93, in cmd_info
UnicodeEncodeError: 'gbk' codec can't encode character '\xc5' in position 30
```

Workaround:

```powershell
$env:PYTHONIOENCODING = "utf-8"    # PowerShell
# or: set PYTHONIOENCODING=utf-8   (cmd)
# or: chcp 65001                    (switch the console to UTF-8)
odock info
```

### `no compatible GPU adapter` / the `gpu` line says "unavailable"

Either the kernel was built without the `gpu` feature (the default, and the only
possibility for the Python wheel configured here) or no Vulkan/Metal/D3D12
adapter is present. Nothing is broken: the CPU path is always used and always
correct. See [GPU notes](#9-gpu-notes).

### The ligand ends up outside the box / the search produces no poses

Symptoms: poses with `in_box = False`, a suspiciously large positive affinity,
or

```text
Error: invalid input: the search produced no poses; check the grid box and the ligand
```

Causes and fixes:

* the ligand's input position is outside the box → use
  `--box-from-ligand --buffer 8`, or move `--center`, or enlarge `--size`;
* the box is far too small for the ligand → the whole molecule is inside the
  penalty wall;
* the box is in the wrong place because the receptor and the ligand come from
  different coordinate frames (a very common mistake: a ligand from a
  homology model, or a receptor that was re-oriented). Check `odock box`'s
  printed centre against the ligand's coordinates;
* the ligand was prepared with `--keep-nonpolar` on a big molecule, which
  inflates the atom count and slows everything down without changing the
  physics — not an error, but rarely what you want.

The out-of-box penalty is a *linear* wall of 1e6 kcal/mol/Å, so a pose that ends
up outside the box is reported as a huge positive energy, not as a silent
mistake.

### Flexible receptor

This release docks **rigid receptors only**, and it says so instead of pretending
otherwise:

* A receptor file containing flexible-residue records (`BEGIN_RES` / `END_RES` /
  `ROOT` / `BRANCH` / `TORSDOF`) is accepted, but every such record is recorded
  as a `ParseIssue` with the message
  `treating flexible-residue record "..." as rigid: OpenDocking 0.1.0 docks rigid
  receptors`. The side-chain atoms are kept (rigidly) so the box still excludes
  them, and the numbers are *not* flexible-receptor numbers.
* The `flex` half of the machinery exists in the kernel (`TreeKind::Flex`,
  `build_flex_tree`, `DofLayout::flex`, `Shared::flex_pairs`), but the docking
  orchestrator does not wire it up: `build_system` always builds
  `flex: Vec::new()` and an empty flex layout. Treat it as groundwork for a
  future release.

Practical consequence: a residue that has to move to admit the ligand will not
move. If that matters for your system, dock into a receptor where the relevant
side chain is already in an open conformation, or model the side chain yourself
and dock into that.

### Multi-model receptor files are rejected

```text
Error: parse error on line 12: multi-MODEL receptor files are not supported;
       split the models first
```

A receptor with more than one `MODEL` is ambiguous. Split it first (for example
with `odock split` for PDBQT, or by extracting the first model) and dock against
exactly one structure.

### `odock score` and multi-model files

`odock score` reads only the **first model** of a pose file. That is almost always
what you want — scoring mode 1 reproduces the docking report exactly (`-7.991`
for the 3PTB mode 1) — but if you meant a different mode, split the file first
with `odock split` and score the individual model, or pass that model's text
directly (see [§7.4](#74-scoring-without-searching)).

### Missing `TORSDOF` / inconsistent `BRANCH` records

The ligand parser is strict about the topology language:

* `missing TORSDOF keyword` — the file has no `TORSDOF` line;
* `inconsistent branch numbers: BRANCH a b vs ENDBRANCH c d` — a mismatched
  pair;
* `atom <n> has not been found in this branch` — `BRANCH a b` names an
  attachment atom that is not inside the block;
* `ROOT without ENDROOT`, `BRANCH without ENDBRANCH` — unbalanced blocks;
* `the ligand ROOT level must not declare an attachment atom` — a `BRANCH` at
  the top level.

Any of these means the file was edited by hand or written by another tool with
different conventions. Re-prepare the ligand with `odock prepare ligand`.

### `ligand contains no atoms` / `receptor contains no atoms`

The file parsed, but nothing usable came out of it. Usual causes: a PDB without
`ATOM`/`HETATM` records, an empty selection, or a file whose records use a
different name (`HETATM` is accepted; `ATOM  ` is accepted; a bare `XXXX` line
is not).

### `box size must be positive` / `spacing must be positive`

`BoxSpec` (and the PyO3 constructor) validate eagerly. This is what you get for
`--size 0 0 0` or a negative `--buffer` that makes the computed size non-positive.

### `every candidate pose was rejected during refinement`

Every pose either failed to refine to a finite energy or landed above half the
maximum representable energy. In practice this means the ligand is buried in the
receptor wall or the box is wrong; see the ligand-outside-the-box entries above.

### The run is slower than expected

* `--scoring ad4` is the usual explanation: no grid, everything exact.
* `--no-grid` disables the grid.
* `--exhaustiveness` 32 with a large ligand is simply a lot of work.
* A very large box: the grid holds `n_x * n_y * n_z * n_types` `f64` values;
  `odock dock` reports both figures in its stderr line (`grid 150920 points
  (3 MB)` for the 3PTB example above).
* Debug builds (`maturin develop` without `--release`) are roughly an order of
  magnitude slower.

---

## 11. FAQ

**Do I need AutoDock or AutoDock Vina installed?**
No. OpenDocking is self-contained; it re-implements the algorithms and reads and
writes the PDBQT format itself.

**Is my receptor/ligand chemistry compatible?**
If AutoDock Vina accepts the PDBQT, OpenDocking does. Ligands need the
`ROOT`/`BRANCH`/`TORSDOF` topology (which `odock prepare ligand` writes);
receptors are rigid `ATOM`/`HETATM` records with AD4 atom types in the last two
columns.

**What does a negative affinity mean?**
It is a *prediction* of binding free energy in kcal/mol in the Vina convention:
more negative = better predicted binding. It is not a measured ΔG, and empirical
scoring functions of this family have limited absolute accuracy — compare
affinities within one series, computed with the same force field and the same
box.

**How do I know whether the docking worked?**
Re-dock a ligand whose bound pose is known: the top-pose RMSD to the crystal
structure is the standard figure of merit. `tests/validate_3ptb.py` does exactly
that for PDB 3PTB and reports `1.124 Å` (no superposition) with affinity
`-7.991 kcal/mol` at `--exhaustiveness 16 --seed 42`. Anything under ~2 Å means
the correct binding mode was found.

**Why is the affinity I get from `odock score` different from the docking
result?**
Because they are different conformations. `odock score` evaluates the ligand
exactly where the file puts it; the docked pose has been optimised. On 3PTB, the
crystallographic position scores `-7.659 kcal/mol` and the top docked pose scores
`-7.991` — the search found a slightly better minimum 1.124 Å away from the
crystal position.

**Can I dock several ligands at once?**
There is no batch subcommand in this release: dock them one at a time. A loop is
fine, and so is running several `odock` processes in parallel — but note that a
single run already uses every core through `rayon`, so process-level parallelism
only helps if you have more ligands than cores.

**Can I use `-o` output as input for another run?**
Yes. The pose writer re-emits the full PDBQT topology, so
`odock dock -l poses.pdbqt ...` or `odock score -l poses.pdbqt ...` works on a
single-model file (`odock split` first if the file has several models).

**Are the runs reproducible across machines?**
Within one platform and one build, yes, bit for bit: the RNG is implemented
in-tree, bond perception is deterministic and the parallel result collection is
ordered. Across platforms, floating-point differences (and different `rayon`
thread counts only in the sense of scheduling, not results) can change the last
digits, so treat "same seed" as a within-platform guarantee.

**Why does the reported affinity differ from the objective the search
minimises?**
Because the torsional penalty `1/(1 + w_rot * N_tors)` is a topology-only
constant: the search minimises the undivided energy and applies the divisor when
reporting. That matches the reference implementation.

**Why do hydrogens not appear in the energy terms?**
In the Vina/Vinardo force fields every hydrogen is typed with the "no X-Score
type" sentinel, so it contributes no energy of its own — but it still decides
whether its heavy neighbour is an H-bond donor. Non-polar hydrogens are merged
into their carbon during preparation; polar ones are kept precisely so this
typing works. See
[SCORING.md](SCORING.md#hydrogens-and-x-score-typing).

**Can I change the scoring weights?**
Not from Python in this release. `Weights` is introspectable
(`odock._odock.Weights.default_for("vina").terms`) but `dock()` and the PyO3
`Docking` constructor do not accept an override; `DockOptions::weights` is only
reachable from Rust.

**Where is the GUI?**
`odock gui` loads the 3-D workbench (`odock.gui`, PyQt6 + ModernGL, the `[gui]`
extra) lazily. It is optional and never blocks the headless workflow: if the
extras are missing, `launch_gui` raises an `ImportError` naming the exact pip
command, and the CLI turns a failure to import the package into the same
message.

**What is the licence, and what is it derived from?**
OpenDocking is GPL-3.0-or-later. AutoDock Vina (Apache-2.0, The Scripps Research
Institute) and AutoDock 4 (GPL) are the algorithmic references; Meeko (LGPL-2.1)
is the behavioural reference for preparation. No AutoDockTools (ADT / MGLTools)
code was read, borrowed or copied.

---

## 12. Quick reference

```bash
# install
python -m venv .venv && .venv/Scripts/activate
pip install maturin rdkit numpy
maturin develop --release
odock info

# prepare
odock prepare receptor receptor.pdb receptor.pdbqt
odock prepare ligand ligand.sdf ligand.pdbqt --name myligand

# box
odock box --ligand ligand.pdbqt --out box.json --buffer 8

# dock
odock dock -r receptor.pdbqt -l ligand.pdbqt --box box.json \
    -o poses.pdbqt -s vina -e 16 -n 9 --seed 42

# score / split
odock score -r receptor.pdbqt -l ligand.pdbqt
odock split poses.pdbqt --outdir poses

# validate
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42
```

```python
import odock

receptor_mol, receptor_pdbqt, _ = odock.prepare_receptor("receptor.pdb")
ligand_mol, ligand_pdbqt, lig_report = odock.prepare_ligand("ligand.sdf")
box = odock.box_from_ligand(ligand_mol, buffer=8.0)
result = odock.dock(receptor_pdbqt, ligand_pdbqt, box, exhaustiveness=16, seed=42)
print(result.table())
open("poses.pdbqt", "w").write(result.to_pdbqt())

mol = odock.pose_to_mol(result.best(), ligand_mol, lig_report.atom_order)
print(odock.aligned_rmsd(mol, crystal_mol))
```

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
affinity          :     -5.806 kcal/mol
  inter           :     -6.146
  intra           :     -0.044
  conf-independent:      0.339
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
a pose file, lets you drag the search box in 3-D, and runs the same Rust kernel
the CLI and the Python API use. Everything in it is optional — the CLI and the
Python API never import it.

**Mouse and keyboard**

| input | effect |
|---|---|
| left drag | orbit the camera |
| right drag | pan |
| **shift** + left drag | move the search box in the plane of the screen |
| wheel | zoom |
| pose slider / table row | show that mode |
| `Frame all` / `Frame ligand` | re-centre the camera |

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

### 5.5 Pocket detection, filters and clustering

Four subcommands answer questions that come up before and after a docking run.
All of them accept `--json-out FILE` where a machine-readable result makes sense.

**`odock pocket` — blind cavity detection.** No ligand and no prior knowledge of
the site are needed: the receptor is rasterised, a rolling probe marks the space
a water-sized sphere cannot occupy, and the surviving buried regions are grouped
into cavities and ranked.

```bash
odock pocket -r receptor.pdbqt --spacing 1.0 --min-volume 100 --max 10
odock pocket -r receptor.pdbqt --json-out pockets.json
```

```text
#  center (x, y, z)              volume/Å³  score  residues
-  ----------------------------  ---------  -----  ---------------------------
1  (   17.31,    0.58,   27.83)  403.0      0.78   LEU123 A, VAL235 A, ILE238 A
2  (   -0.37,   18.27,   19.16)  192.0      0.73   GLN192 A, SER195 A, SER214 A
```

`--spacing` is the probe grid step (smaller is slower and finer), `--min-volume`
discards small cavities, `--probe` sets the probe radius (1.4 Å is water) and
`--max` caps the list. The score prefers cavities that are large *and* genuinely
enclosed; it is a shortlisting heuristic, not a binding-affinity prediction. The
residues column names the residues within 5 Å of the cavity centre, so a pocket
can be recognised at a glance. In the workbench the same search fills the pocket
dialog, and one click aims the search box at the chosen cavity.

**`odock filter` — library pre-filters.** One report for as many files as you
like, mixing formats freely:

```bash
odock filter -i library.sdf -i more.smi --json-out filters.json
```

```text
name          MW     LogP  HBD  HBA  RotB  tPSA  Lipinski  Veber  PAINS  verdict
------------  -----  ----  ---  ---  ----  ----  --------  -----  -----  -------
benzoic acid  122.1  1.38  1    1    1     37.3  pass      pass   pass   PASS
```

The table gives the descriptors each rule uses (MW, LogP, HBD, HBA, RotB, tPSA)
and the verdict of Lipinski, Veber and PAINS; the rule breaches are listed under
the table, and the JSON form keeps the per-filter detail. Only descriptors are
computed — no conformers are generated — so a large SDF is triaged quickly.

**`odock cluster` — symmetry-aware RMSD clustering.** Poses that landed in the
same place are one binding mode, however many of them the search reported:

```bash
odock cluster -p poses.pdbqt --cutoff 2.0
odock cluster -p poses.pdbqt -l ligand.pdbqt      # + RMSD to a reference
```

```text
cluster  size  representative  affinity  mean RMSD/Å
-------  ----  --------------  --------  -----------
1        2     mode 1          -6.210    0.065
2        1     mode 3          -5.309    0.000
3        2     mode 4          -5.045    1.673
4        1     mode 6          -4.414    0.000
```

(the bundled `demo/3ptb/poses.pdbqt`: six poses, four clusters). Atom
equivalences (the two oxygens of a carboxylate, the six carbons of a benzene
ring) are handled by graph-based matching, so a 180° ring flip counts as the same
pose. With `-l`, the reference ligand's RMSD is added for each cluster
representative — the redocking check, per cluster.

**`odock interactions` and `odock diagram`.** The interaction profile of one
pose, and the 2-D topology diagram of the same contacts:

```bash
odock interactions -r receptor.pdbqt -l ligand.pdbqt
odock diagram -r receptor.pdbqt -l ligand.pdbqt -o interactions.svg
```

On the bundled `demo/egfr/` system (erlotinib in the EGFR kinase domain), the
profile reproduces the known binding mode — the hinge hydrogen bond to Met769
plus the hydrophobic contacts around it:

```text
kind         receptor      ligand        d/Å   detail
-----------  ------------  ------------  ----  -----------------------
hbond        MET769 A N    AQ4999 A N2   2.70  MET769:N->AQ4999:N2
hydrophobic  LEU764 A CB   AQ4999 A C1   3.43  LEU764:CB...AQ4999:C1
hydrophobic  THR766 A CG2  AQ4999 A C1   3.53  THR766:CG2...AQ4999:C1
...

summary: hbond 1, hydrophobic 9
key residues: LEU694, LEU764, LEU768, LYS721, MET769, THR766
```

Six interaction types are detected — hydrogen bonds, salt bridges, π–π stacking
(face-to-face or T-shaped), cation–π, hydrophobic contacts and steric clashes —
and every geometric cut-off is a flag (`--hbond`, `--salt`, `--pi`,
`--cation-pi`, `--hydrophobic`, `--clash-ratio`), shared by both commands, so the
diagram always matches the profile it was drawn from.

### 5.6 Reports, standard input files and downloads

**`odock report`** turns a pose file into a results table. XLSX or CSV, decided
by `--csv` or the file extension:

```bash
odock report -p poses.pdbqt -r receptor.pdbqt -o report.xlsx
odock report -p poses.pdbqt -o report.csv --csv
```

With `-r`, the "key interacting residues" column is filled in per mode. XLSX
writing needs `openpyxl`; without it the writer degrades to CSV and says so.

**`odock export`** writes the input files that the original AutoDock tools read,
so an OpenDocking run can be handed to AutoGrid/AutoDock, or archived for
reproducibility:

```bash
odock export gpf    -r receptor.pdbqt -o receptor.gpf --box box.json
odock export dpf    -r receptor.pdbqt -l ligand.pdbqt -o receptor.dpf --box box.json
odock export config -r receptor.pdbqt -l ligand.pdbqt -o vina.txt --box box.json -e 32
odock export pdb    -i receptor.pdbqt -o receptor.pdb
```

* `gpf` — an AutoGrid grid parameter file: odd `npts`, spacing, the map list for
  every atom type (read from the receptor unless `--ligand-types` says
  otherwise), `elecmap`, `dsolvmap` and the dielectric.
* `dpf` — an AutoDock 4 docking parameter file: ligand types and torsions taken
  from the ligand PDBQT, the map references, `move`/`about`, and the search
  section chosen with `--parameters {lga,ga,ls,none}`.
* `config` — an AutoDock Vina configuration file (`receptor`, `ligand`,
  `center_*`, `size_*`, `exhaustiveness`, …).
* `pdb` — a cleaned structure as PDB, with the `MASTER`/`END` tail.

**`odock fetch`** downloads from the RCSB, with the identifier validated before
any request is made:

```bash
odock fetch 3PTB -o 3PTB.pdb            # a whole entry
odock fetch BEN --ligand -o BEN.sdf     # a chemical component
```

A 404, a refused connection or a timeout is reported as a one-line error naming
the identifier and the reason — never as a traceback.

---

## 6. Understanding the output

### 6.1 The results table

The first four modes of the recorded 3PTB validation run
(`--exhaustiveness 16 --seed 42`, seven modes inside the 3 kcal/mol window):

```text
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -6.213      0.000      0.000
   2       -6.191      0.060      1.602
   3       -5.035      2.410      3.547
   4       -4.935      2.777      3.567
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

The CLI additionally prints `best affinity: -6.213 kcal/mol` unless `-q` is
given.

### 6.2 The pose file

Each `MODEL` carries a `REMARK` block:

```text
MODEL 1
REMARK VINA RESULT:       -6.213      0.000      0.000
REMARK INTER + INTRA:        -6.619
REMARK INTER:                -6.576
REMARK INTRA:                -0.043
REMARK CONF_INDEPENDENT:      0.363
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
print(result.best_affinity)          # -6.213 (or None)
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
affinity              -5.806
total                 -5.806
inter                 -6.146
intra                 -0.044
conf_independent       0.339
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
# {'affinity': -6.213, 'total': -6.213, 'inter': -6.576, 'intra': -0.043,
#  'conf_independent': 0.363, 'unbound': -0.043}
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
what you want — scoring mode 1 reproduces the docking report exactly (`-6.213`
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
that for PDB 3PTB and reports `1.133 Å` (no superposition) with affinity
`-6.213 kcal/mol` at `--exhaustiveness 16 --seed 42`. Anything under ~2 Å means
the correct binding mode was found.

**Why is the affinity I get from `odock score` different from the docking
result?**
Because they are different conformations. `odock score` evaluates the ligand
exactly where the file puts it; the docked pose has been optimised. On 3PTB, the
crystallographic position scores `-5.806 kcal/mol` and the top docked pose scores
`-6.213` — the search found a slightly better minimum 1.133 Å away from the
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

# pockets, filters, clustering, interactions
odock pocket -r receptor.pdbqt --max 10 --json-out pockets.json
odock filter -i library.sdf -i more.smi --json-out filters.json
odock cluster -p poses.pdbqt -l ligand.pdbqt --cutoff 2.0
odock interactions -r receptor.pdbqt -l ligand.pdbqt --hbond 3.5
odock diagram -r receptor.pdbqt -l ligand.pdbqt -o interactions.svg

# reports and standard input files
odock report -p poses.pdbqt -r receptor.pdbqt -o report.xlsx
odock report -p poses.pdbqt -o report.csv --csv
odock fetch 3PTB -o 3PTB.pdb
odock export gpf    -r receptor.pdbqt -o receptor.gpf --box box.json
odock export dpf    -r receptor.pdbqt -l ligand.pdbqt -o receptor.dpf --box box.json
odock export config -r receptor.pdbqt -l ligand.pdbqt -o vina.txt --box box.json -e 32
odock export pdb    -i receptor.pdbqt -o receptor.pdb

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

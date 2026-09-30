# OpenDocking validation

Every number in this file was measured on this source tree. Each section names
the command that reproduces it, so nothing here has to be taken on trust.

```bash
# Rust kernel: 116 tests plus the integration suite.
cargo test --workspace

# The same tests with the optional wgpu backend compiled in.
cargo test -p dock-core --features gpu

# Python: the whole suite, including the GUI offscreen tests.
python -m pytest tests -q

# The crystallographic acceptance test, end to end.
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42

# Drive the real workbench through a full user session.
python tests/simulate_workbench.py
```

---

## 1. Automated suites

| suite | command | result |
|---|---|---|
| Rust kernel | `cargo test --workspace` | **116 passed** (+ 5 integration tests in `crates/dock-core/tests/real_ligand.rs`, + 1 doc test) |
| Rust, GPU backend | `cargo test -p dock-core --features gpu` | **116 passed** |
| Python | `.venv/Scripts/python.exe -m pytest tests -q` | **605 passed, 1 skipped** (85 s) |

The single skipped Python test writes the style contact sheets
(`tests/test_gui_viewport.py`); set `ODOCK_STYLE_SHOTS=1` to run it. The GUI
tests drive Qt on the `offscreen` platform, so they need no display.

What the suites actually check, beyond "the code runs":

* **Every analytic derivative against central finite differences** — each Vina,
  Vinardo and AD4 term, the interpolated grid gradient, the exact scorer's
  Cartesian gradient, and the full degree-of-freedom gradient through the
  kinematic tree. The AD4 electrostatics and desolvation terms are compared
  relatively because they vary by several kcal/mol over 1e-6 Å at short range.
* **Kinematics** — the identity conformation reproduces the input structure,
  torsion rotation moves only the moving side, and a rigid-body transform is an
  isometry. The 3PTB/1M17 fixtures exercise *nested* `BRANCH` records, which a
  single-torsion ligand never does.
* **Search** — BFGS convergence on an analytic surface and the Solis-Wets
  adaptive step schedule.
* **GPU versus CPU** — the grid builder, the batch forward-kinematics expansion
  and the batch atom-pair evaluation are compared against their CPU reference
  implementations and must agree to floating-point noise. The tests pass on a
  machine with no graphics adapter too (the CPU fallback runs), so a GPU build is
  never *required* for a correct result.
* **Cancellation** — pause, resume and abort are exercised under concurrency;
  an aborted run returns the best pose it has found and releases its memory.
* **Determinism** — the same seed reproduces a pose list bit for bit.
* **PDBQT round trips** — including the nested topology language and the exact
  column placement of the AD4 atom type.

---

## 2. Re-docking a real complex: PDB 3PTB

`tests/validate_3ptb.py` is the end-to-end acceptance test. It splits the
crystal structure of bovine trypsin with benzamidine, prepares both partners
through the OpenDocking pipeline, derives the search box from the
crystallographic ligand, docks the ligand back from random starting
conformations, and measures the heavy-atom RMSD of every pose against the
experimental position — **without superposition**, so the figure is the
crystallographic one rather than a fit to the answer.

```bash
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42
```

```text
target                 PDB 3PTB (bovine trypsin + benzamidine)
force field            vina
independent MC runs    16               (--exhaustiveness 16)
seed                   42
top-pose RMSD          1.133 Å          (heavy atoms, no superposition)
affinity (mode 1)      -6.213 kcal/mol
affinity (crystal pose) -5.806 kcal/mol
poses in the 3 kcal/mol window  7
grid                   150 920 points, 3 MB
wall time              3.0 s
verdict                PASS (threshold 2.0 Å)
```

The pose file of that run records the full decomposition:

```text
mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1       -6.213      0.000      0.000
   2       -6.191      0.060      1.602
   3       -5.035      2.410      3.547
   4       -4.935      2.777      3.567
   5       -4.887      1.379      2.449
   6       -4.403      2.487      3.329
   7       -4.228      3.402      4.435

mode 1 REMARK block:
  VINA RESULT:       -6.213      0.000      0.000
  INTER + INTRA:       -6.619
  INTER:               -6.576
  INTRA:               -0.043
  CONF_INDEPENDENT:     0.363
  UNBOUND:             -0.043
```

The numbers are self-consistent with the published Vina convention (see
[`SCORING.md`](SCORING.md#torsional-penalty-and-the-reported-affinity)):
`INTER + INTRA - UNBOUND = -6.576`, the torsional divisor for this ligand is
`N_tors = 1` and `w_rot = 0.05846`, so the reported affinity is
`-6.576 / 1.05846 = -6.213`, and `CONF_INDEPENDENT` is the
`0.363` difference to the undivided objective.

Mode 1 is 1.133 Å from the crystal pose and the crystal pose itself scores
−5.806 kcal/mol, i.e. the search found a slightly *better* minimum than the
experimental geometry, next to it. That is the expected outcome for a rigid
one-rotor ligand: the pose is recovered, and the residual 1.1 Å is the
(chemically unimportant) in-plane orientation of the amidine group.

For context: the experimental binding free energy of benzamidine to trypsin is
about −8.5 kcal/mol, and AutoDock Vina reports roughly −8 to −9 for this system.
The absolute affinity this force field produces (−6.2) is therefore weaker than
the experimental estimate, which is normal for an empirical scoring function of
this family; the *binding mode*, which is what docking is used for, is reproduced
to 1.1 Å. Compare affinities within one series and one box, never across
programs.

---

## 3. A flexible, drug-like ligand: PDB 1M17

The bundled demo (regenerated by `python examples/make_demo.py`, and quoted in
[`demo/README.md`](../demo/README.md)) docks erlotinib into the EGFR kinase
domain — 30 heavy atoms, 11 rotatable bonds, a partly water-mediated binding
mode, and the crystallographic waters stripped by default.

| system | ligand | heavy atoms | N_tors | top-pose RMSD | best RMSD within 1 kcal/mol |
|---|---|---|---|---|---|
| 3PTB (trypsin) | benzamidine | 13 | 1 | **1.13 Å** | 1.13 Å |
| 1M17 (EGFR) | erlotinib | 30 | 11 | 4.69 Å | **1.43 Å** |

RMSD is the heavy-atom RMSD against the experimental pose with no
superposition. The last column is the usual top-N measure: the best RMSD among
all poses within 1 kcal/mol of the top score.

Read the second row honestly: the search **finds** the experimental binding
mode — it is the pose at 1.43 Å — but the empirical force field ranks four
near-degenerate poses above it (they differ by 0.11 kcal/mol). That is a
property of the Vina energy function for a molecule with 11 rotatable bonds, not
of the search: the rigid case above is reproduced to 1.1 Å by the same engine.

---

## 4. Blind pocket detection

`odock pocket` needs no knowledge of the ligand. Run on the receptor prepared by
the validation above (`odock pocket -r 3ptb_receptor.pdbqt`, the default 1.0 Å
probe grid, minimum cavity volume 100 Å³) it reports seven cavities, and the
trypsin S1 specificity pocket is among them:

| | |
|---|---|
| rank by the heuristic score | 2 of 7 |
| volume | 192 Å³ |
| centre to the crystal benzamidine centroid | 4.6 Å |
| residues within 5 Å of the centre | GLN192, SER195, SER214, TRP215, GLY216 |

That is the correct site (the benzamidine amidine hydrogen-bonds into the
Asp189/Ser190 region of this pocket), found without any prior information about
where the ligand sits. Note that the ranking is a heuristic, not a promise: the
detector is meant to shortlist cavities for inspection, and it does.

---

## 5. Driving the real workbench

`tests/simulate_workbench.py` starts the actual `QMainWindow`, drives it with Qt
mouse and keyboard events through a complete user session — importing
structures, cleaning and protonating the receptor, building a ligand from
SMILES, filtering it, finding pockets, dragging the grid box, docking,
inspecting interactions, clustering poses, exporting every report format, and
switching the interface language — and writes an illustrated report with a
screenshot of each step to `out/simulation/report.md` (a local artefact; `out/`
is not part of the repository).

```bash
python tests/simulate_workbench.py          # headless (offscreen Qt platform)
python tests/simulate_workbench.py --show   # on a real display
python tests/simulate_workbench.py --fast   # skip the in-workbench docking step
```

Recorded result on this tree:

```text
steps    : 48
passed   : 48
failed   : 0
skipped  : 0
exit code: 0
```

The script exits non-zero if any step fails, so it doubles as the GUI smoke test
that the unit tests cannot be: `tests/test_gui*.py` cover the widget logic, and
this covers the workflow.

---

## 6. What is *not* validated

Known limitations, stated so that they cannot be mistaken for oversights:

1. **The search is rigid-receptor.** A flexible-residue PDBQT can be written
   correctly (`BEGIN_RES`, nested `BRANCH`/`ENDBRANCH`, `END_RES`, with
   continuous serial numbers), and the kernel carries the flexible machinery
   (`MovableModel::flex`, `TreeKind::Flex`, `build_flex_tree`, the flexible
   terms and their pair list) with unit tests — but the *receptor reader*
   currently folds a flexible file back to a rigid receptor and emits an
   explicit parse note. Treating a side chain as rigid is a controlled,
   clearly-reported approximation.
2. **Charges.** The Kollman model is the published united-atom scheme
   (Gasteiger with hydrogen merging), not the Kollman/AMBER residue charge
   tables, which are not available offline. For a hydrogen-free crystal protein
   Gasteiger charges do not converge; the module then returns a conserved
   all-zero set and says so. The AD4 electrostatic term therefore needs an
   externally supplied charge set to be quantitatively meaningful.
3. **The 150 Da ligand threshold.** Following the specification, a small ligand
   such as benzamidine (112 Da) is not classified as a co-crystallised ligand
   automatically. The workbench's strip action also accepts an explicit residue
   name, and falls back to carbon-containing `other` residues with four or more
   heavy atoms.
4. **GPU scope.** The `wgpu` backend accelerates affinity-grid construction,
   batch forward kinematics and batch atom-pair evaluation. The Monte-Carlo/LGA
   search loop stays on the CPU, where its short branchy work is faster than a
   kernel launch. The `gpu` feature is off by default, so a plain build needs no
   graphics driver.
5. **SSAO** is a screen-space approximation (linearised depth, a 12-sample
   kernel), not ray-traced ambient occlusion.
6. **Pocket buriedness** counts whether protein is reached along the 26 cubic
   directions within 5 Å by default. Setting `ray_length=spacing` recovers the
   literal nearest-neighbour definition, but then misses 3PTB's real S1 site.
7. **Macrocycle handling** is basic: Vina's linear-attraction ("glue") term is
   implemented, but closure sampling is not specially optimised.
8. **No AutoDockTools (ADT / MGLTools) code was read, borrowed or copied.**
   Everything PDBQT-related is written from the format specification plus RDKit.

---

## 7. Attribution

The algorithms and file formats follow their published descriptions and the
freely licensed reference implementations. See the attribution section of
[`../README.md`](../README.md) for the full list; in short: AutoDock Vina
(Apache-2.0) for the Vina/Vinardo potentials and the iterated-local-search
protocol, AutoDock 4 (GPL) for the AD4 force field, the Lamarckian GA, the
Solis-Wets local search and the GPF/DPF formats, and Meeko (LGPL-2.1) as the
behavioural reference for ligand preparation.

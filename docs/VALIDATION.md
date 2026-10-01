# OpenDocking validation

Every number in this file was measured on this source tree. Each section names
the command that reproduces it, so nothing here has to be taken on trust.

```bash
# Rust kernel: the unit tests plus the real-ligand integration suite.
cargo test --workspace

# The same tests with the optional wgpu backend compiled in.
cargo test -p dock-core --features gpu

# Python: the whole suite, including the GUI offscreen tests.
python -m pytest tests -q

# The crystallographic acceptance test, end to end.
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42

# The five-system re-docking benchmark, three seeds per system.
python -m odock.benchmark

# Drive the real workbench through a full user session.
python tests/simulate_workbench.py
```

---

## 1. Automated suites

| suite | command | result |
|---|---|---|
| Rust kernel | `cargo test --workspace` | **116 passed** (+ 5 integration tests in `crates/dock-core/tests/real_ligand.rs`, + 1 doc test) |
| Rust, GPU backend | `cargo test -p dock-core --features gpu` | **116 passed** |
| Python | `.venv/Scripts/python.exe -m pytest tests -q` | **1294 passed, 1 skipped** (557 s, measured 2026-10-01) |
| Workbench session | `python tests/simulate_workbench.py` | **48 steps, 48 passed, 0 failed** |
| Benchmark | `python -m odock.benchmark` | 5 systems × 3 seeds, see [§4](#4-the-re-docking-benchmark) |

The Python and benchmark rows were re-measured on this revision. The Rust rows
are unchanged from the 0.1.0 cut: no file under `crates/` has been modified since
(`docs/BENCHMARK.md` records the machine and the kernel version alongside the
benchmark numbers).

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
* **Determinism** — the same seed reproduces a pose list bit for bit; the
  benchmark repeats a 26-minute, 15-docking run field for field (§4).
* **PDBQT round trips** — including the nested topology language and the exact
  column placement of the AD4 atom type.
* **The science layer** — ligand-efficiency metrics against hand-checkable
  arithmetic, ligand strain against a force-field minimum, interaction
  fingerprints and water bridges against hand-placed geometry, and the consensus
  rank aggregation against worked three-pose examples.
* **The documentation itself** (`tests/test_docs.py`) — every local link
  resolves, the published docs name no private file, and the shipped sources
  cite no internal brief. Both failure modes have happened here: the README once
  pointed at a gitignored hand-over document, and a staging pass once cleaned the
  citations in a *copy* of the sources while the working tree kept them.

### The release content scan, and why the working tree shows hits

`tools/inspect_dist.py` and the release content scan check both directions: every
promised file is present in an artefact, and nothing internal is. The scan is a
**byte** scan for strings that must never ship — private document names, absolute
developer paths, the vendored comparison binary.

Point it at the **staged published set**, not at the working tree, and expect a
handful of hits **inside the exclusion machinery**: `.gitignore`, the maturin
`include`/`exclude` list in `pyproject.toml`, the inspect tool's own deny-list and
its self-test, and the CI workflow that runs the tool. Those files have to spell
the names they keep out — that is their job — so a working tree *cannot* be
byte-clean by construction, and a hit in one of them is **not** a defect to fix
by weakening the machinery. What matters is that a scan of the published set
finds those strings nowhere else: no hit in `docs/`, `python/`, `tests/`, the
README or `CONTRIBUTING.md`.

Real leaks were caught this way, and are worth remembering as examples:

* `benchmark/baseline.json` recorded the absolute checkout path of the machine
  that produced it. The baseline is documentation — "run this to reproduce" — and
  a path that exists on one laptop is worse than useless. It is now recorded as
  `python -m odock.benchmark --seeds 42,7,2024 ...`, normalised at the boundary
  (`benchmark.portable_command`) so a baseline re-cut from an older run cannot
  inherit a path either.
* `.gitignore` used to ignore the whole `demo/` directory (with a hand-written
  exception for the screening library), which silently dropped 18 files the
  release is supposed to contain. The rule now ignores exactly the vendored
  AutoDock Vina binary: the demo data ships, the third-party executable does not.

The check that *guards* these rules is `out/verify/release_check.py`: it applies
the tree's own `.gitignore` in memory and fails when a file of a release content
family would not be published.

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
target                  PDB 3PTB (bovine trypsin + benzamidine)
force field             vina
independent MC runs     16               (--exhaustiveness 16)
seed                    42
top-pose RMSD           1.133 A          (heavy atoms, no superposition)
affinity (mode 1)       -6.213 kcal/mol
affinity (crystal pose) -5.806 kcal/mol
poses reported          7
grid                    150 920 points, 3 MB
wall time               2.1 s
verdict                 PASS (threshold 2.0 A)
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
`demo/README.md`) docks erlotinib into the EGFR kinase domain — 29 heavy atoms,
11 rotatable bonds, a partly water-mediated binding mode, and the
crystallographic waters stripped by default.

| system | ligand | heavy atoms | N_tors | top-pose RMSD | best RMSD within 1 kcal/mol |
|---|---|---|---|---|---|
| 3PTB (trypsin) | benzamidine | 9 | 1 | **1.13 Å** | 1.13 Å |
| 1M17 (EGFR) | erlotinib | 29 | 11 | 4.69 Å | **1.43 Å** |

RMSD is the heavy-atom RMSD against the experimental pose with no
superposition. The last column is the usual top-N measure: the best RMSD among
all poses within 1 kcal/mol of the top score.

Read the second row honestly: the search **finds** the experimental binding
mode — it is the pose at 1.43 Å — but the empirical force field ranks four
near-degenerate poses above it. That is a property of the Vina energy function
for a molecule with 11 rotatable bonds, not of the search: the rigid case above
is reproduced to 1.1 Å by the same engine. §4 measures how much that ranking
moves when the seed changes.

---

## 4. The re-docking benchmark

`python -m odock.benchmark` re-docks five bundled crystal complexes with three
seeds each and reports accuracy, the rank of the correct pose, the
score-versus-RMSD correlation and the three force fields' agreement **with their
spread** — because a single number hides the 3.2 Å of seed-to-seed movement on
the flexible system. [`BENCHMARK.md`](BENCHMARK.md) has the full table and, more
importantly, the list of what five systems do *not* establish.

```bash
python -m odock.benchmark                     # 26 minutes on 16 cores
python -m odock.benchmark --check-baseline    # fail on a scoring regression
```

| system | ligand | heavy | rotors | top RMSD | seed spread | best ≤1 kcal | rank | rho (n) | agreement range | weakest pair |
|---|---|---|---|---|---|---|---|---|---|---|
| 3PTB | BEN | 9 | 1 | **1.13 Å** | 0.00 | 1.13 Å | 1 | +0.78 (15) | +0.82…+0.91 | vina~ad4 |
| 1STP | BTN | 16 | 5 | **1.03 Å** | 0.01 | 1.03 Å | 1 | — | — | — |
| 3ERT | OHT | 29 | 8 | **1.51 Å** | 0.43 | 1.35 Å | 1 | +0.64 (8) | +0.81…+0.88 | vinardo~ad4 |
| 1M17 | AQ4 | 29 | 11 | **1.51 Å** | **3.17** | 1.28 Å | 1 | +0.51 (9) | +0.52…+0.69 | vina~ad4 |
| 1HVR | XK2 | 46 | 8 | 10.07 Å | 0.00 | **0.79 Å** | 2 | −1.00 (2) | −0.33…+1.00 | vinardo~ad4 |

Every seed of every system finds a pose within 2 Å of the crystal. Three things
in that table matter more than the averages, and all three are arguments against
reading one number:

* **1M17's top-pose RMSD moves 3.17 Å between seeds** (4.69 / 1.51 / 1.70 Å)
  while `best ≤1 kcal` stays at 1.28–1.44 Å in every seed. The search finds the
  experimental mode every time; the *ranking* of near-degenerate poses is what
  moves.
* **1HVR's top pose is 10.1 Å from the crystal in every seed**, and the crystal
  mode is at **rank 2, 0.79–1.10 Å, inside 1 kcal/mol**. A benchmark reporting
  only `top RMSD` would score this as a failure; the answer is in the output.
* **AD4 is the outlier**: the weakest force-field pair involves AD4 on every
  measurable system (vina~ad4 twice, vinardo~ad4 twice). An ablation — zeroing
  every receptor charge, which removes AD4's electrostatics and its
  charge-dependent desolvation — changes its correlation with Vina by 0.02 while
  moving its own ranking on 4 of 9 poses, and AD4 spans 3.19 kcal/mol where Vina
  spans 0.23 over the same poses. That is a scale difference in the
  parameterisation, not a defect; see
  [`BENCHMARK.md`](BENCHMARK.md#why-ad4-disagrees-with-vina-measured-not-asserted).

**Regression gate.** `benchmark/baseline.json` records the measured values, the
exact command, the environment and the tolerances, and `--check-baseline` fails
when a system is worse by more than 0.15 Å on a top-pose RMSD, one rank, or 0.2
on a correlation. The tolerance is *repeat-run* noise — measured as exactly zero
on this machine, because a re-run of the same command is bit-for-bit identical
(verified: the 15-docking run repeated on the settled revision differs in **0
fields**) — and deliberately *not* the seed spread, which is an order of
magnitude larger and would produce a gate that never fires.

---

## 5. Blind pocket detection

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

## 6. Driving the real workbench

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
```

The script exits non-zero if any step fails, so it doubles as the GUI smoke test
that the unit tests cannot be: `tests/test_gui*.py` cover the widget logic, and
this covers the workflow.

---

## 7. What is *not* validated

Known limitations, stated so that they cannot be mistaken for oversights:

1. **The search is rigid-receptor.** A flexible-residue PDBQT can be written
   correctly (`BEGIN_RES`, nested `BRANCH`/`ENDBRANCH`, `END_RES`, with
   continuous serial numbers), and the kernel carries the flexible machinery
   (`MovableModel::flex`, `TreeKind::Flex`, `build_flex_tree`, the flexible
   terms and their pair list) with unit tests — but the *receptor reader*
   currently folds a flexible file back to a rigid receptor and emits an
   explicit parse note. Treating a side chain as rigid is a controlled,
   clearly-reported approximation.
2. **Charges.** The Gasteiger model is what the writer uses, and it does **not**
   converge for a fraction of a crystallographic protein's atoms: on the bundled
   3PTB receptor 128 charges came back non-finite and were written as `0.000`,
   which the writer now reports through a warning. Those atoms contribute
   nothing to the AD4 electrostatic term, so an AD4 score on such a receptor is
   quantitative only to the extent that the charge set is; supply an external
   charge set when it matters.
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
8. **The force-field strain of a PDBQT-perceived molecule is unreliable.** A
   PDBQT carries no bond orders, so RDKit sees a benzene ring as cyclohexane; for
   benzamidine that turns a true MMFF94 strain of ~2.5 kcal/mol into 68.6, and a
   ligand prepared from a bare PDB (without a `smiles=` template) is in the same
   position. `odock.metrics` marks such a result `reliable=False` and
   `require_reliable=True` makes it an error; the kernel strain, whose two
   endpoints come from one potential, is the number the report prints.
9. **Five benchmark systems are not CASF.** They are all rigid-receptor,
   single-ligand, drug-like complexes prepared with a CCD SMILES template;
   accuracy on them is a regression signal, not a capability claim.
10. **No AutoDockTools (ADT / MGLTools) code was read, borrowed or copied.**
    Everything PDBQT-related is written from the format specification plus RDKit.
11. **Ligand-based methods are not validated against activity data.** The
    cheminformatics layer ([`CHEMINFORMATICS.md`](CHEMINFORMATICS.md)), the
    pharmacophore layer ([`PHARMACOPHORE.md`](PHARMACOPHORE.md)), the ligand-based
    benchmark ([`LBVS.md`](LBVS.md)) and the hit-triage layer
    ([`TRIAGE.md`](TRIAGE.md)) are validated against hand-computed arithmetic and
    the bundled library, not against an assay: a Tanimoto similarity, a Murcko
    scaffold, a matched-pair delta, a pharmacophore fit and a structural alert are
    all *structural* statements. The one labelled set this repository contains is
    six actives in seventeen molecules, which cannot establish enrichment, and the
    alert catalogues flag a third of the bundled pool — both the APIs and the
    documents say so where the numbers are quoted.

---

## 8. Attribution

The algorithms and file formats follow their published descriptions and the
freely licensed reference implementations. See the attribution section of
[`../README.md`](../README.md) for the full list; in short: AutoDock Vina
(Apache-2.0) for the Vina/Vinardo potentials and the iterated-local-search
protocol, AutoDock 4 (GPL) for the AD4 force field, the Lamarckian GA, the
Solis-Wets local search and the GPF/DPF formats, and Meeko (LGPL-2.1) as the
behavioural reference for ligand preparation.

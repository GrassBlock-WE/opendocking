# Changelog

All notable changes to OpenDocking are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.2 — 2026-10-01

A patch release whose main purpose is to remove files that should never have
been published.

### Fixed

* **Interaction lines were drawn to the wrong endpoint.** A dash segment is
  `(pos_a, rgba, pos_b, rgba)` — fourteen floats — so the second position starts
  at index **7**; the tube builder read index **6**, i.e. `(alpha, y_b, z_b)`.
  Every contact was therefore swept from a real atom to a nonsense point, so a
  2.8 Å contact was drawn **14–21 Å long** and read as an interaction acting
  across the protein. Every swept tube is now at most `DASH_LENGTH` (0.45 Å) and
  lies on the line between the two named atoms to within 0.05 Å. The reported
  distances were always correct, which is why an earlier check of the
  interaction *rows* could not see this: **verifying the data is not verifying
  the pixels.**
* **Hiding the receptor left lines pointing at nothing.** `_draw_interactions`
  read `scene.receptor` without consulting `show_receptor`, and the
  interaction-focus pass drew its ball-and-stick mesh without the visibility
  check its sphere half already had. Both now respect it: a hidden receptor
  draws no contacts and no emphasis (24 046 px → 0 px).
* **The search box was the same colour as a hydrogen bond** — RGB distance
  **0.112**, indistinguishable by eye. The box is now teal (0.15, 0.55, 0.55),
  ≥ 0.415 from every interaction colour.
* **Four Chrome profile directories reached the published repository**
  (`odock-chrome-*`, each with `metadata`, `settings.dat`,
  `CrashpadMetrics-active.pma`). They were left behind by the Chrome-based PDF
  attempt, which correctly reported that no PDF was produced but wrote its
  profile into the working directory — because the environment temp directory is
  not writable, so it fell back to the checkout. They are deleted here and the
  family is now denied in every rule that decides what ships.
* Contact dashes are drawn longer (33 segments → 20 for the same five contacts)
  and every interaction now has an endpoint marker at each atom it names, so a
  line visibly lands on something instead of disappearing behind the ribbon.

### Added

* `odock docs` — a self-contained documentation site built from the published
  markdown, with a search index, a rendered-link guard, and a generator that
  reads the source with `ast` rather than importing it.
* `odock tutorial` — the whole toolchain on the bundled data in one runnable
  command, with the numbers it measured printed inline.
* `odock release` — prepare / stage / check / notes / publish, encoding the
  failures that were previously made by hand.

### Changed

* Every new documentation page is part of the site's set-equality check, so a
  module or document cannot be missing from it silently.
## 0.2.1 — 2026-10-01

The ligand side of the science, ensembles instead of one rigid structure, runs
that can be reopened and reproduced, and a CLI that cannot silently lose a
command. 1063 → 1398 tests.

### Added — ligand cheminformatics

* `odock.ligandsim` — six fingerprint kinds behind one RDKit-free type
  (Morgan/ECFP, RDKit path, atom pair, torsion, MACCS, 3-D pharmacophore) with
  Tanimoto/Dice/Tversky, a blocked similarity matrix, analogue search, and
  best-over-conformers 3-D search that reports its cost.
* `odock.scaffold` — Murcko and generic skeletons, scaffold groups and series,
  an MCS core, **R-group decomposition** into positional `R1..Rk` with
  unmatched rows reported, and **matched molecular pairs** with affinity deltas.
* Library diversity: MaxMin and sphere exclusion, Butina clustering (checked
  against RDKit's own implementation), and `odock screen --diverse N`.
* `odock.pharmacophore` — models built from **recurring features with per-feature
  member support**, scored against a library with a documented fit and an
  envelope constraint. Measured on the five ring-amidines: four features, every
  one supported 5/5, sweeping ranks 1–5 at 0.918–0.944 against a best decoy of
  0.497; warfarin is rejected with 18 of its 23 heavy atoms outside the envelope.
* `odock.decoys` + `odock.lbvs` — property-matched decoy generation with a
  Tanimoto ceiling, and an enrichment benchmark reporting **EF1 %, EF5 %, AUC and
  BEDROC with bootstrap intervals** alongside random and property-only controls.
  The quality report says when its own decoy set is not matched rather than
  hiding it.

### Added — ensembles and cryptic pockets

* `odock.ensemble` — several receptor conformations validated as the same
  protein (sequence identity, site superposition, a named refusal when they are
  not) and docked against a shared box, with **pose clustering across
  conformations** and a per-ligand **robustness score**. Measured: on ERα
  (3ERT/1ERE_A) the winning conformation is the receptor rather than the seed,
  and equal-affinity ligands separate on robustness (caffeine 0.814 vs
  benzamidine 0.004).
* `odock.pockets` across an ensemble — per-conformation cavities mapped into a
  common frame and tracked across structures, with **closure / narrowed /
  detector disagreement** verdicts and the detector's own noise floor reported
  beside every claim. Measured: three cavities close when the agonist binds
  (410→4, 352→22, 366→4 Å³, each 5.4–5.7× the local resolution) while the
  trypsin control yields **zero** candidates (largest apparent change 40 Å³
  against a 39 Å³ resolution).
* `odock ensemble {align,dock,screen,pockets}`; a ligand-free conformation is
  refused for box derivation with the reason named.

### Added — reproducible, presentable runs

* `odock.project` — a self-contained `.odockproj` (inputs with SHA-256, every
  setting, the seed, the poses, a schema version), with save / open / verify /
  info / extract. A flipped byte is caught and named.
* `odock project reproduce` — re-runs the docking from the recorded inputs and
  reports **PASS / FAIL / INCONCLUSIVE** with the numbers; the default tolerance
  is **exactly zero** because the engine is deterministic (measured
  `max |Δaffinity| = 0.000`, `top-pose RMSD = 0.000 Å`, two reproductions
  byte-identical).
* `odock project compare` and `odock project index` — several runs side by side
  with the differing settings named field by field, and a campaign index page.
* `odock report-html` — a single self-contained HTML report (inline SVG, base64
  figures, zero external references) with an explicit "what this run does not
  establish" section.

### Fixed

* **`odock <cmd> --help` crashed for every subcommand.** argparse `%`-formats
  help strings, and a new command's one-line help contained a literal `EF1%`.
  Every `format_help` raised `ValueError: unsupported format character ','`,
  which also broke `odock` with no arguments and `odock --help`.
* **A symmetric core was ring-shifted during pharmacophore alignment.** Taking
  RDKit's *first* core match left para analogues' amidines on the wrong side,
  costing them both donor features (fit 0.00, ranks 16–17 beside 0.94
  analogues); matches are now enumerated and the lowest RMSD kept.
* **Self-similarity leaked into the ligand-based benchmark.** With the actives
  as their own queries all three methods scored AUC and BEDROC 1.000 for free;
  leave-one-out is now the API default for fingerprint, pharmacophore and shape.
* **A verdict rule fired on its own counterexample** in the pocket tracker: a
  track absent where the free volume was *higher* is detector disagreement, not
  a cryptic site, and is now excluded from the candidate list.
* A test pinned a **suite count** rather than the shape of the result, so the
  documentation guard broke whenever any test landed anywhere.
* `docs/VALIDATION.md` and the README were brought back in step with the tree,
  and internal-document citations were removed from the shipped sources at the
  source rather than only in the staged copy.

### Added — engineering

* `tests/test_cli_surface.py` — the complete expected subcommand set asserted in
  **both** directions, that building the parser emits no registration warning,
  that every registrar ran, and that every command has a help page. It exists
  because a cleanup script once deleted a region of `cli.py` and took two other
  workstreams' registration calls with it, dropping six subcommands while the
  suite stayed green.
* `python/odock/cli_ext.py` — one extension entry point, so contributed
  subcommands register in their own modules and `cli.py` is edited once.

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

Recorded in full in [`docs/VALIDATION.md`](docs/VALIDATION.md#7-what-is-not-validated).
In short: the search is rigid-receptor (a flexible-residue PDBQT can be written,
but the kernel's receptor reader folds it back to rigid and says so); the
Kollman charge model is the united-atom scheme, not the AMBER residue tables;
GPU acceleration covers grid construction and batch kernels only; SSAO is a
screen-space approximation; and macrocycle closure sampling is not specially
optimised.

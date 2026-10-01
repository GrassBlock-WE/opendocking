# Docking against several receptor conformations

A receptor is not a statue. A single crystal structure is one snapshot of a
mobile protein, and a docking search that minimises into that snapshot rewards
the poses that happen to fit *it*. This document describes OpenDocking's answer:
treat a **set** of conformations as the receptor, dock into all of them, and
combine the results so that a pose found in several structures is worth more than
a pose found in one.

Everything below is implemented in [`odock.ensemble`](../python/odock/ensemble.py),
reachable from the command line as `odock ensemble align|dock|screen`, and
measured on real crystal structures by
[`examples/ensemble_validation.py`](../examples/ensemble_validation.py).

```text
                       ┌── align ──────────┐
  PDB files / NMR models                  │  sequence identity, site fit, per-residue
        │                                 │  displacement; refuse a mismatched set
        ▼                                 │
   odock ensemble ── dock ────────────────┤  one box, one seed per conformation,
        │                                 │  merged ranking, cross-conformation modes
        │                                 │
        └── screen ───────────────┐       │  the odock.screen campaign, unchanged,
                                  │       │  plus one ensemble row per molecule
                                  ▼       ▼
                     robustness: is the winning mode there in every structure?
```

## 1. What an ensemble is, and what makes a set one receptor

`odock ensemble` accepts several PDB/PDBQT files, or one multi-model PDB (an NMR
ensemble, a set of snapshots), and expands them into a list of conformations —
one per file or per `MODEL`. The first is the **reference** unless `--reference`
says otherwise.

Three checks run before anything is docked, and each one either passes or
refuses **with the numbers**:

* **Sequence identity and overlap.** The chains of every conformation are
  aligned globally (Needleman–Wunsch on the one-letter sequences of the standard
  residues) and the best chain pairing is reported. Fewer identical residues than
  `--min-identity` (default 0.90) is a refusal, not a warning, because docking
  one ligand into a set of different proteins produces numbers that cannot be
  compared. Residue *numbers* are deliberately not used for the correspondence:
  two crystal structures of one protein routinely number their residues
  differently, and the sequence alignment is the thing that is actually right
  about which residue is which.
* **A shared binding-site frame.** The site is
  - an explicit list (`--site ASP189,SER190,A:195`), else
  - the residues within `--site-radius` Å of the **box centre** (default 8 Å), else
  - the residues within `--site-radius` Å of a reference residue
    (`--site-ligand OHT`: the co-crystallised ligand).
  Every conformation is superposed onto the reference on those residues' CA
  atoms (`--site-atoms ca|backbone|all`) by a proper-rotation Kabsch fit. This is
  what makes **one box valid in every frame**, and it is why the box is checked
  again afterwards: a box that contains no atom of a conformation is refused
  (`--allow-box-mismatch` overrides, loudly).
* **The box.** `--box box.json`, `--center`/`--size`, or `--box-ligand RESNAME`
  (the box around the co-crystallised ligand of the reference, padded by
  `--buffer`).

A co-crystallised ligand that sits **inside the search box** is stripped
automatically and reported: a residue occupying the site being docked into is not
part of the receptor. Ions, phosphate and other small groups are kept — metals
are wanted by the force field, and they do not fill the site.

### Measured: three genuine multi-conformation cases

All three pairs are two real crystal structures of the same protein, shipping in
`tests/data`. `python examples/ensemble_validation.py` reproduces the table.

| case | reference → fitted | residues | identity | overlap | site residues | site RMSD (Å) | whole-protein CA RMSD (Å) | largest site displacement (Å) |
|---|---|---|---|---|---|---|---|---|
| ERα antagonist vs agonist | 3ERT (4-OHT) → 1ERE chain A (estradiol) | 247 → 235 | **0.944** | 0.947 | 14 | **0.444** | **4.551** | LEU525 (helix 12) **2.010**; ASP351 1.709 |
| HIV-1 protease isolates | 1HVR (XK263) → 1HXW (A-84538) | 198 → 198 | **0.970** | 0.970 | 22 | **0.407** | 0.630 | ILE50 B **1.589**; ILE50 A 1.440 (the flap tips) |
| trypsin, the control | 3PTB (benzamidine) → 2PTN (ligand-free) | 223 → 223 | **1.000** | 1.000 | 22 | **0.144** | 0.118 | GLN192 1.133 |

Read the ERα row the way a modeller would: the **binding site itself fits to
0.44 Å**, the whole protein moves 4.55 Å around it, and the residues that move
are the ones the literature names — **helix 12 (LEU525, 2.0 Å)** and
**ASP351 (1.7 Å)**, the residue whose position decides agonist against
antagonist. The HIV-1 row is the same story one scale down: the site fits to
0.4 Å and the **Ile50 flap tips move 1.4–1.6 Å**. The trypsin row is the control:
two structures of the same protein whose sites differ by 0.14 Å must look like
one, and at the site level they do — section 5 shows what even that still costs
in kcal/mol.

The 1HVR/1HXW pair is also the case the sequence check exists for: **0.970** is
below one but far above the 0.90 floor, so the set is accepted *and the number is
printed*. Refusing it would be wrong (it is the same protein);
accepting it silently would hide the three sequence differences.

## 2. Docking every conformation, and merging the poses

```bash
odock ensemble dock \
    -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT -l estradiol.sdf \
    -e 16 -n 9 --seed 42 -o ensemble_poses.pdbqt
```

* every conformation is docked with the **same box** and the **same seed**. The
  same seed is deliberate: the ensemble is a paired comparison, and a different
  seed per conformation would confound the receptor's motion with the search's
  own randomness. The consequence — the winner can still be seed luck — is
  measured in section 5, not assumed away.
* every pose from every conformation is pooled into **one ranking** (best
  affinity first), and each pose keeps the conformation that produced it, its
  index inside that conformation, and its rank inside that conformation.
* the pooled poses are **clustered across conformations**
  (`--cluster-rmsd`, default 2.0 Å) by `odock.analysis.cluster_poses` —
  single-linkage on the symmetry-aware, no-superposition RMSD, the project's own
  pose comparison. Because the conformations were superposed first, all poses are
  already in one frame, so "the same binding mode in two receptors" means what it
  says. Each cluster reports its size, how many *distinct* conformations support
  it, the affinity spread inside it, and the largest pairwise RMSD in it.
* `-o FILE` writes the merged ranking as one multi-model PDBQT, in rank order,
  each model carrying the toolkit's standard `REMARK  VINA RESULT:` line plus
  `REMARK  ODOCK ENSEMBLE: conformation=… cluster=… rank=…`. It is an ordinary
  pose file — `odock cluster`, `odock report` and `odock.cli._read_poses` are
  asserted to read it in `tests/test_prepare_cli.py`, and the workbench's pose
  loader takes the same `MODEL`/`REMARK VINA RESULT` records.

`--no-superpose` skips the fit and only validates and reports: for a set that is
already in one frame (a trajectory, or a run of `odock ensemble align --outdir`).

## 3. Cross-receptor consensus, and what "robust" means as a number

`odock.consensus` already answers "do the force fields agree?" for one receptor.
An ensemble adds the other axis, and `odock ensemble` uses `odock.consensus` for
both halves:

* **within one conformation**: `conformation_consensus` rescored each
  conformation's own poses with vina, vinardo and ad4
  (`odock.consensus.consensus_score`) and reports the mean pairwise Spearman rho
  **with its sample size** — the number of poses in that conformation. This is
  also what `odock ensemble screen --consensus` runs per receptor.
* **across conformations**: `cross_rescore` scores **every pose in every
  conformation**, at fixed coordinates, with `odock.consensus.rescore_poses` —
  the same code path and the same box. `refine=False` is what makes it a
  measurement: refining in the new receptor would let the pose relax into it and
  hide the clash it should pay for. The result is a `poses × conformations`
  matrix, quoted with its sample size `n_scored`.

The **robustness score** of the winning binding mode is then

```text
support     = conformations that produced a pose in the winning cluster / n
persistence = conformations whose own best pose is in that cluster   / n
movement    = median, over the cluster's poses, of (max_k E_k − min_k E)
              where E_k is that pose's affinity in conformation k

robustness  = support × persistence × exp(−movement / 2.0 kcal/mol)
```

Every component is reported next to the score, together with the mean movement,
the **worst single penalty**, the number of poses whose movement exceeds
10 kcal/mol (a pose that cannot be placed in another conformation at all, i.e. a
clash — counted separately so one broken pose cannot decide the summary), and the
sample sizes (`n_poses`, `n_scored`, `n_consensus`). The 2.0 kcal/mol scale is a
**convention** chosen so the number lands in `[0, 1]` with a useful spread, not a
fitted constant; the raw kcal/mol numbers are always printed beside it.

### Measured: estradiol into the two ERα structures

`odock ensemble dock -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT -l EST.sdf -e 4 -n 5`

| conformation | best affinity (kcal/mol) |  |
|---|---|---|
| 3ERT (antagonist, helix 12 open) | **−9.313** | estradiol is not its ligand |
| 1ERE_A (agonist, estradiol-bound) | **−11.381** | the winner |
| ensemble best | **−11.381** | from 1ERE_A |

The winning mode is found in **both** conformations (support 2/2, persistence
2/2), so support and persistence are 1.0, and the score is decided by the
movement: the median affinity spread of the cluster's poses on a receptor swap is
**4.27 kcal/mol** (mean 5.23, worst single penalty 8.92), giving
**robustness 0.118**. The pose *survives* the receptor moving; the *number* does
not. One pose in the same run scores **+196 kcal/mol** in the other conformation:
that mode exists only because 3ERT's helix 12 is open, and the ensemble says so
instead of reporting it as the best pose.

## 4. The ensemble-aware screen

```bash
odock ensemble screen \
    -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT \
    -i library.sdf -o results --top 50 --robustness-top 10
```

This **is** the `odock screen` campaign — `odock.screen.screen_ligands`, called
unchanged — with the receptor set replaced by the aligned ensemble. Everything
that makes a campaign survivable is therefore inherited, not reimplemented:

* the append-and-flush `results.jsonl` (or `results.csv`) with **one row per
  (conformation, ligand)**, so an interrupted campaign resumes exactly where it
  stopped and every row says which receptor it came from;
* the library cache (`library/`), the per-receptor shortlists
  (`top_<conformation>.pdbqt`), the per-molecule timeout, the filter funnel, the
  interaction profiling and `--consensus`;
* the run manifest. The aligned receptor PDBQTs are written into
  `<out>/ensemble/`, and **only when their bytes change** — rewriting an
  identical file would change its mtime, and screening hashes the receptor files
  to decide whether a directory is resumable.

On top of those rows it adds, in the same directory:

* `ensemble.jsonl` / `ensemble.csv` — **one row per molecule**: best affinity and
  which conformation produced it, mean, worst, spread, the winning mode's support
  and persistence, the cross-receptor movement, the robustness score, and the
  sample sizes;
* `ensemble_top.pdbqt` — the best pose of each top molecule, one `MODEL` each,
  with the support and robustness in its `REMARK` lines;
* `--robustness-top N` (default 10) decides how many of the best molecules get
  the per-molecule clustering and cross-rescoring, because that is the only part
  that costs extra: `poses × conformations` exact rescorings per molecule. Every
  molecule gets its per-conformation best/mean/spread for free from the rows.

An interrupted ensemble screen is resumed by re-running the same command: the
campaign resumes, and the ensemble layer is recomputed from the rows on disk
(it is post-processing, exactly like `--consensus`, so it can be added to a
finished campaign without invalidating a single row).

## 5. Does the ensemble change the answer? (measured)

Five real ligands docked into 3ERT alone, into 1ERE_A alone, and into the
ensemble (`-e 4 -n 5`, seed 42; `python examples/ensemble_validation.py`).

| ligand | 3ERT | 1ERE_A | ensemble best | gap (kcal/mol) | robustness |
|---|---|---|---|---|---|
| estradiol | −9.666 | −11.375 | −11.375 | **1.709** | 0.171 |
| aspirin | −6.068 | −6.420 | −6.420 | 0.352 | 0.009 |
| naphthol | −6.965 | −6.979 | −6.979 | 0.014 | 0.528 |
| caffeine | −5.584 | −5.724 | −5.724 | 0.140 | 0.814 |
| benzamidine | −5.548 | −6.074 | −6.074 | 0.526 | 0.004 |

Three honest observations, none of them flattering to a naive ensemble story:

1. **The top-ranked ligand did not change.** Estradiol is first under 3ERT, under
   1ERE_A and in the ensemble — which is reassuring rather than disappointing:
   the agonist structure is estradiol's own complex, and a method that shuffled
   the ranking here would be broken. The *numbers* moved by up to 1.71 kcal/mol,
   which is larger than the 0.5 kcal/mol that separates the remaining four
   ligands from each other: docking into one structure would have answered
   "estradiol by 3.6 kcal/mol" instead of "by 5.3".
2. **The winning conformation never changed**: across the five ligands in this
   table, the two trypsin ligands, and estradiol under three seeds, the same
   structure won every time (1ERE_A for ERα, 2PTN for trypsin) — 7 ligands and 3
   seeds, 0 flips. The measured gaps are stable too (estradiol 1.709 / 1.715 /
   1.718 kcal/mol; robustness 0.171 / 0.175 / 0.332), so here the winner is a
   property of the receptor, not of the seed. That is a *measurement on this
   pair*: a set of structures that really does divide a ligand's poses between
   them — an open and a closed channel, say — would flip, and the ensemble is
   what makes the flip visible instead of picking one structure's opinion.
3. **The robustness column separates ligands that the affinity ranking does not.**
   Benzamidine and naphthol have the *same* tiny conformational gap (0.526 vs
   0.014 kcal/mol), yet their robustness differs by two orders of magnitude
   (0.004 vs 0.528). Benzamidine's top pose in 3ERT is not its top pose in
   1ERE_A at all — a small, nearly symmetric cation in a tight pocket has several
   orientations within a few hundredths of a kcal/mol — while naphthol and
   caffeine put the same mode in the same place in both structures. That is the
   question an ensemble is for, and it is invisible in a single-structure table.

The trypsin control (`python examples/ensemble_validation.py --case trypsin
--fast`) is the other end, and it is *not* "nothing happens":

| ligand | 3PTB | 2PTN | gap (kcal/mol) | robustness |
|---|---|---|---|---|
| estradiol | −6.690 | −7.004 | 0.314 | **0.709** |
| benzamidine | −6.103 | −5.545 | 0.558 | **0.616** |

3PTB and 2PTN share their binding site to **0.14 Å** CA RMSD — and the affinity
of the same ligand still moves by **0.31–0.56 kcal/mol** between them, which is
larger than the gap that separates many ligands from one another in a screening
table. What the ensemble adds here is therefore not a better number but the
*evidence* that the number is structural noise rather than a property of the
ligand: both modes are reproduced in both structures (robustness 0.62 and 0.71,
support 2/2), while in the ERα pair even the same ligand's mode survived with a
4.3 kcal/mol spread. A single-structure affinity carries at least this much
conformational noise, and an ensemble is how you see it.

## 6. What this does not establish

* **Conformational selection versus induced fit cannot be separated.** Docking
  scores a *fixed* receptor. A pose that scores well in one conformation of the
  ensemble may be a pose the ligand selects, or one it would have induced in a
  structure the ensemble does not contain. No amount of scoring on rigid
  receptors distinguishes those; that needs free-energy methods, and the honest
  claim is only "this pose is compatible with these structures".
* **The ensemble is only as good as the structures in it.** Every conformation
  carries **equal weight** — no Boltzmann weighting, no experimental evidence, no
  measure of how populated a structure is in solution. A poorly refined structure
  or a crystal-packing artefact votes exactly as loudly as the best one. There is
  no ensemble validation here: no cross-validation against held-out structures,
  and no check that the set spans the true conformational space.
* **Every conformation is rigid.** Side chains do not move within a run, so
  induced fit *inside* a conformation is invisible. The kernel carries flexible
  side chains (`MovableModel::flex`) but the PDBQT reader folds them into the
  rigid receptor, so an ensemble of rigid structures is the supported
  approximation.
* **The frame is a rigid-body fit on the site's CA atoms.** No flexible
  alignment, no normal-mode interpolation, no molecular dynamics. A conformation
  whose site moves by more than a couple of Ångströms is superposed at the cost
  of everything else, one box may no longer cover every frame, and the
  displacement table is the only warning. Over 2 Å site RMSD a warning is
  printed.
* **The robustness score is a convention, not a probability.** The formula, its
  components and its 2.0 kcal/mol scale are stated above so the number can be
  recomputed by hand from the printed parts. It is not calibrated against
  experiment, and a high score does not mean "this ligand binds".
* **A near-symmetric small ligand can score low while agreeing perfectly.** Mode
  support is an RMSD-2 Å statement: benzamidine's two orientations are chemically
  the same pose and geometrically 2+ Å apart, so its support is 1/2 even though
  the affinities agree to 0.001 kcal/mol. Read the affinity columns next to the
  score.
* **One seed per conformation.** The same seed is used everywhere, so the
  comparison is paired, but the winner can still be seed luck. Section 5 measures
  that for one case (three seeds, same winner) — it is a measurement, not a
  guarantee, and a production run should repeat.
* **Cross-rescoring uses the docking force field.** A vina pose is rescored by
  vina in another conformation (and by all three fields in
  `--consensus`); a pose that exists only because of a force field's bias is not
  detected by this axis.
* **Nothing here is CASF.** One pair with a large real difference (ERα), one with
  a modest one (HIV-1 protease) and one control (trypsin) is a demonstration that
  the machinery measures what it claims, not a benchmark of ensemble docking.
* **The pocket analysis has its own, longer list** ([`docs/POCKETS.md`](POCKETS.md)
  §7): two snapshots cannot show that a cavity is druggable or populated in
  solution, the detector's absolute volume moves 18–57 % when nothing but its own
  settings change, and "a pocket was not found here" is not the same statement as
  "the pocket is closed here".

## 7. The workbench

**No Ensemble menu was added.** `gui/` is being changed concurrently by the
workbench work (the dashboard, pose comparison and theming), and
`tests/simulate_workbench.py` pins the window to 48 user steps; a menu added on
top of an in-flight rework is a collision risk with no scientific payoff, and the
mandate allows skipping it explicitly. What an ensemble does instead is produce
files the existing workbench already opens:

* `odock ensemble dock -o poses.pdbqt` is a standard multi-model pose file
  (ranked across conformations, one `REMARK  ODOCK ENSEMBLE:` line per model), so
  `odock gui -p poses.pdbqt` steps a pose through the merged ranking;
* `odock ensemble screen` writes `ensemble_top.pdbqt` in the same form;
* `odock ensemble pockets --pockets-pdb pockets.pdb` writes every cavity point as
  a `HETATM` record in the common frame, with the conformation in the residue-name
  column and the cavity in the chain column, so the aligned receptor plus that
  file shows which structures reveal which cavities;
* `odock ensemble align --outdir DIR` writes every aligned conformation as
  `DIR/<label>.pdb` (and `.pdbqt` with `--pdbqt`), which the workbench loads as
  separate receptors — the per-conformation alignment and the per-residue
  displacement are in the report and in `--json-out`.

## 8. Cryptic and transient pockets

The other half of the ensemble story is the site that exists in *some*
conformations and not others: `odock ensemble pockets` detects the cavities of
every conformation with the project's pocket detector, puts them in the same
common frame this module builds, matches them across structures by centroid and
lining residues, and separates a cavity that opens from a detector that changed
its mind — against the detector's own measured noise floor.

That is a document of its own, with the measured numbers on the ERα pair and the
trypsin control: [`docs/POCKETS.md`](POCKETS.md).

## 9. Reproducing every number

```bash
python examples/ensemble_validation.py                       # the whole report
python examples/ensemble_validation.py --case trypsin --fast # the control
odock ensemble align -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT
odock ensemble dock  -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT -l EST.sdf -e 4 -n 5 --seed 42
odock ensemble screen -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT -i library.smi -o results
python -m pytest tests/test_ensemble.py -q                   # the same numbers as tests
```

The structures are in `tests/data` (`3ERT.pdb`, `1ERE_A.pdb`, `1HVR.pdb`,
`1HXW.pdb`, `3PTB.pdb`, `2PTN.pdb`, `EST.sdf`, `BTN.sdf`); `1ERE_A.pdb` is chain
A of RCSB entry 1ERE, 1HXW.pdb is entry 1HXW, and 2PTN.pdb is entry 2PTN, all
fetched with `odock fetch` and committed so the numbers above are reproducible
offline.

## 9. API

```python
from odock import ensemble
from odock.prepare import box_from_points

# 1. read, validate, superpose onto a shared binding-site frame
conformations = ensemble.read_conformations(["3ERT.pdb", "1ERE_A.pdb"])
box = box_from_points(ensemble.ligand_coords(conformations[0], "OHT"), buffer=6.0)
aligned = ensemble.align_conformations(conformations, box=box, site_radius=8.0)
print(aligned.table(), aligned.displacement_table())

# or everything in one step, prepared for docking
built = ensemble.build_ensemble(["3ERT.pdb", "1ERE_A.pdb"], box=box, site_ligand="OHT")

# 2. dock every conformation, merge and cluster the poses
result = ensemble.dock_ensemble("EST.sdf", built, exhaustiveness=8, num_poses=9, seed=42)
print(result.text())                       # winner, per-conformation best, modes
open("poses.pdbqt", "w").write(result.to_pdbqt())

# 3. cross-receptor rescoring, consensus, robustness
cross = ensemble.cross_rescore(result, built)                  # poses x conformations
consensus = ensemble.conformation_consensus(result, built)     # per conformation
score = ensemble.robustness_score(result, cross, consensus=consensus)
print(score.text())                                            # with sample sizes

# 4. the campaign
from odock.screen import ScreenConfig
summary = ensemble.screen_ensemble(ScreenConfig(
    receptors=["3ERT.pdb", "1ERE_A.pdb"], inputs=["library.sdf"], box=box, outdir="results",
))
print(summary.text())
```

`odock.ensemble` imports nothing from `odock.cli`; the command line is registered
by `odock.ensemble.add_ensemble_parser(sub)`, which is the only thing
`odock.cli` needs from this module.

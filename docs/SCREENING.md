# Screening a library

`odock dock` docks one ligand. `odock screen` docks a **library**: it reads a
multi-format file, filters it, docks every survivor against one or more
receptors, and writes results that survive being interrupted.

```bash
odock screen -r receptor.pdbqt -i library.sdf --box box.json -o results --exhaustiveness 8 --jobs 4
```

This page is a tutorial with a **runnable example on the bundled data**, then a
reference for every option and every file the command writes.

Everything here is also available from Python:

```python
import odock
from odock.screen import ScreenConfig, screen_ligands

summary = screen_ligands(ScreenConfig(
    receptors=["receptor.pdbqt"],
    inputs=["library.sdf"],
    box=odock.BoxSpec(center=(10.0, 20.0, 30.0), size=(22.5, 22.5, 22.5)),
    outdir="results",
    exhaustiveness=8,
    top=50,
))
print(summary.text())
for record in summary.ranked(summary.receptor_names[0], limit=10):
    print(f"{record.name:30s} {record.affinity:8.3f} kcal/mol  LE {record.ligand_efficiency:.2f}")
```

---

## 1. A runnable example

The repository bundles a trypsin structure (PDB 3PTB, the same system as the
re-docking validation), its active-site box, and a small screening library of 17
molecules (`demo/library.smi`): six benzamidine analogues, nine drug-like
compounds, one molecule that fails Lipinski's LogP rule and one PAINS quinone.

### 1.1 Look before you leap

```bash
odock screen \
    -r demo/3ptb/receptor.pdbqt \
    -i demo/library.smi \
    --box demo/3ptb/box.json \
    -o out/3ptb-screen \
    --dry-run
```

`--dry-run` reads, filters and prepares the library, **docks the first molecule
once to measure this machine**, and reports what it would do:

```text
probing this machine with one real docking (benzamidine, exhaustiveness 4) ...

dry run: 15 molecule(s) would be docked against 1 receptor(s)
#   name                MW     LogP   tors  atoms
--  ------------------  -----  -----  ----  -----
1   benzamidine         120.2  0.97   1     12
2   benzamidine_methyl  134.2  1.23   2     12
3   benzylamidine       134.2  1.17   2     13
4   hydroxybenzamidine  136.2  0.68   1     14
5   fluorobenzamidine   138.1  1.11   1     13
6   chloro_benzamidine  154.6  1.62   1     13
7   caffeine            194.2  -1.03  0     14
8   aspirin             180.2  1.31   3     14
9   ibuprofen           206.3  3.07   4     16
10  salicylic_acid      138.1  1.09   1     12
11  acetanilide         135.2  1.64   1     11
12  paracetamol         151.2  1.35   1     13
13  nicotinamide        122.1  0.18   1     11
14  isonicotinic_acid   123.1  0.78   1     10
15  warfarin            308.3  3.61   4     24

removed by the filters (2 molecule(s)):
  triphenylene                     Lipinski: LogP 5.15 > 5
  benzoquinone                     PAINS: PAINS quinone_A(370)

the estimate is in the report below; nothing was docked
OpenDocking screen — 1 receptor(s) x 15 molecule(s)
seed: 1657272984
filters: 17 read -> 15 kept (2 removed by Lipinski/Veber/PAINS)
  Lipinski       1 failed this filter (1 first here)
  Veber          0 failed this filter (0 first here)
  PAINS          1 failed this filter (1 first here)

dry run — nothing was docked
estimated cost: 54 s (0:54) for 15 molecule(s) x 1 receptor(s), about 3.61 s per
docking, 16 core(s) assumed
dominant molecule: warfarin (~5 s of the total)
measured: benzamidine took 1.97 s at exhaustiveness 4 (0.411 s per unit)
the kernel rebuilds the affinity grid for every molecule; the estimate scales one
measured docking by the size of the library
```

Two things in there matter:

* **the funnel** — 17 read, 15 kept, and which filter removed what. The numbers
  overlap on purpose (`per_filter` counts every molecule that failed that rule)
  and the `(1 first here)` figure attributes each removed molecule to the *first*
  rule it broke, so those numbers add up to the 2 that were dropped;
* **the cost** — one real docking, extrapolated. A model of this kernel is good
  to about an order of magnitude; a measurement is not.

`--dry-run` writes `library.csv` and the prepared `library/` PDBQT files but no
results, so the campaign itself starts instantly.

### 1.2 Run it

```bash
odock screen \
    -r demo/3ptb/receptor.pdbqt \
    -i demo/library.smi \
    --box demo/3ptb/box.json \
    -o out/3ptb-screen \
    --exhaustiveness 8 --top 5 --seed 42
```

```text
=== demo/3ptb/receptor.pdbqt: 15 to dock of 15 (0 already done) ===
    box BoxSpec(center=(-1.86, 14.37, 16.75), size=(17.9, 19.9, 20.5) Å, V=7319 Å³)
  [######################]    15/15    100.0%  ok 15    failed 0     0.14/s  ETA    0:00  best  -7.338
OpenDocking screen — 1 receptor(s) x 15 molecule(s)
results: out/3ptb-screen/results.jsonl
seed: 42
filters: 17 read -> 15 kept (2 removed by Lipinski/Veber/PAINS)
  Lipinski       1 failed this filter (1 first here)
  Veber          0 failed this filter (0 first here)
  PAINS          1 failed this filter (1 first here)

=== demo/3ptb/receptor.pdbqt ===
15 of 15 docked
rank  name                affinity  MW     LE    tors  poses  key residues
----  ------------------  --------  -----  ----  ----  -----  ---------------------------------------------
1     warfarin            -7.338    308.3  0.32  4     4      GLN192, GLY193, LEU99, SER195, TRP215, VAL213
2     hydroxybenzamidine  -6.420    136.2  0.64  1     9      ASP189, GLY219, SER195, VAL213
3     salicylic_acid      -6.303    138.1  0.63  1     6      GLY219, VAL213
4     fluorobenzamidine   -6.234    138.1  0.62  1     3      ASP189, SER190, VAL213
5     chloro_benzamidine  -6.180    154.6  0.62  1     9      ASP189, GLN192, GLY219, VAL213
6     paracetamol         -5.984    151.2  0.54  1     9      GLN192, SER190, VAL213
7     benzamidine         -5.901    120.2  0.66  1     8      ASP189, GLY219, VAL213
8     nicotinamide        -5.823    122.1  0.65  1     9      ASP189, GLN192, GLY219
9     aspirin             -5.719    180.2  0.44  3     4      GLN192, GLY216, SER195, VAL213
10    ibuprofen           -5.693    206.3  0.38  4     9      GLN192, TRP215, VAL213
11    isonicotinic_acid   -5.587    123.1  0.62  1     4      ASP189, VAL213
12    benzylamidine       -5.517    134.2  0.55  2     4      SER190, VAL213
13    benzamidine_methyl  -5.516    134.2  0.55  2     6      ASP189, GLN192, VAL213
14    caffeine            -5.329    194.2  0.38  0     9      SER190, SER195
15    acetanilide         -4.972    135.2  0.50  1     2      GLN192
top 5 shortlist: out/3ptb-screen/top_receptor.pdbqt

summary: out/3ptb-screen/summary.json

15 of 15 docking(s) succeeded, 0 failed, 0 timed out (103.8 s this run, 15 molecule(s) docked)
best: warfarin at -7.338 kcal/mol (receptor)
```

Benzamidine — the crystal ligand — is found in the S1 specificity pocket
(`ASP189, GLY219, VAL213`, rank 7 at `-5.901 kcal/mol`), which is the re-docking
validation reproduced inside a library run. The table is ranked by affinity; `LE`
is ligand efficiency (kcal/mol per heavy atom) and `key residues` comes from
[`odock.analysis`](../python/odock/analysis.py).

The clock is a *throughput* figure for the whole machine: with `--jobs 16` the
15 molecules ran concurrently.

### 1.3 Which hits survive a change of force field — `--consensus`

A ranked list from one scoring function answers "what does *this* potential
think". A campaign has to answer a harder question: which of these hits are still
hits when the potential changes. `--consensus` rescues that question from a
footnote into a column: it takes the `--top N` shortlist (or the top
`--consensus-top`, 50 by default), rescores **every pose at its docked
coordinates** with vina, vinardo and ad4 — no search, no refinement, so the only
thing that changes between columns is the potential — and combines them into one
rank, with the agreement between the fields reported as the mean pairwise
Spearman rho.

```bash
odock screen \
    -r demo/3ptb/receptor.pdbqt \
    -i demo/library.smi \
    --box demo/3ptb/box.json \
    -o out/3ptb-screen \
    --exhaustiveness 8 --top 6 --consensus --seed 42
```

```text
consensus over the top 6 hit(s) (rank, vina, vinardo, ad4):
force-field agreement: mean pairwise Spearman rho 0.73 on 6 hit(s)
  (vina~vinardo 0.94, vina~ad4 0.66, vinardo~ad4 0.60) — the hit list is mostly a
  property of the molecules; over all 19 pose(s) rho 0.90
rank  name                vina (rank)   vinardo (rank)  ad4 (rank)    consensus  poses  docked pose?
----  ------------------  ------------  --------------  ------------  ---------  -----  ------------
1     hydroxybenzamidine  -6.420 (1.0)  -5.821 (1.0)    -5.791 (1.0)  0.000      4      yes
2     chloro_benzamidine  -6.181 (3.0)  -5.137 (4.0)    -5.053 (2.0)  0.400      4      yes
3     fluorobenzamidine   -6.234 (2.0)  -5.588 (2.0)    -4.326 (5.0)  0.400      3      yes
4     benzamidine         -6.038 (4.0)  -5.290 (3.0)    -4.400 (3.0)  0.467      3      yes
5     benzamidine_methyl  -5.493 (5.0)  -4.690 (5.0)    -4.396 (4.0)  0.733      3      no
6     benzylamidine       -5.155 (6.0)  -4.475 (6.0)    -3.715 (6.0)  1.000      2      yes
```

How to read it:

* **the rank columns** are the hit's rank *under that force field*, computed from
  the affinity of the pose the docking itself chose (mode 1). `benzamidine` is
  4th under vina, 3rd under vinardo and 3rd under ad4 → 4th by consensus;
* **consensus** is the weighted mean of the normalised ranks on a fixed `[0, 1]`
  scale: 0 means the pose is best under every field, 1 means worst under every
  field (`--consensus-method borda` or `z` switches the scale);
* **docked pose?** says whether the binding mode the search picked is *still* that
  molecule's best mode once all three fields have voted. `benzamidine_methyl` is
  the one hit here where the search and the consensus disagree about the mode —
  worth a look before it goes into a report;
* **the rho** is the headline: 0.73 here, with vina~vinardo at 0.94 and
  vina~ad4 at 0.66. When the fields agree this well, an ordering is mostly a
  property of the molecules; when they do not, it is mostly a property of the
  potential, and the honest response is to take more than the top few hits
  forward.

  Across systems the pattern is stable and worth quoting plainly: **vina and
  vinardo agree closely, and ad4 is the outlier.** On this 6-hit slice the pair
  correlations are vina~vinardo 0.94, vinardo~ad4 0.60 and vina~ad4 0.66; an
  independent re-verification of this pipeline on the EGFR system (erlotinib,
  11 torsions) measured a mean of **0.68**, again with vina~ad4 the weakest pair
  at 0.66. So a screening hit list ordered by vina survives a change to vinardo
  almost unchanged, and can move substantially under AutoDock 4 — which is the
  honest signal that a top-10 by one empirical potential is a shortlist to
  investigate, not a verdict.

`--consensus` is pure post-processing: it changes no docking, so it can be added
to a campaign that has already finished (running the same command again docks
nothing and only adds the tables). It writes `consensus_<receptor>.csv` and
`consensus_<receptor>.json`, and never fails the run — if `odock.consensus` is
unavailable the campaign still completes and the reason is recorded in
`summary.json`.

```bash
odock screen ... --consensus --consensus-top 100 --consensus-method borda
```

The cost is one exact scoring pass per pose of the shortlist under each field
(about 0.2 s per pose for three fields on the 3PTB system), so it scales with
`--consensus-top × --num-poses`, not with the library.

### 1.4 Look at the hits

```bash
odock gui -r demo/3ptb/receptor.pdbqt -p out/3ptb-screen/top_receptor.pdbqt
odock cluster -p out/3ptb-screen/top_receptor.pdbqt --cutoff 2.0
odock report  -p out/3ptb-screen/poses/receptor/000015_warfarin.pdbqt -o warfarin.xlsx
```

### 1.5 Interrupt it, then continue

A real campaign runs for hours. Kill the run at any point (`Ctrl-C`), then type
the same command again:

```text
resume: 7 molecule x receptor row(s) already in results.jsonl
=== demo/3ptb/receptor.pdbqt: 8 to dock of 15 (7 already done) ===
```

Resuming is the **default** when the output directory already holds results. A
molecule that was already docked is never docked twice, and a record that was
already written is never rewritten: the results file is the state of the run.

### 1.6 A factory reset

```bash
odock screen ... --no-resume      # discards this directory's results and starts over
```

Only the files this command owns are removed (`results.jsonl`/`results.csv`,
`summary.*`, `library.csv`, `run.json`, `poses/`, `library/`); nothing else in
the directory is touched.

---

## 2. What the pipeline does, in order

1. **Validate.** Files exist, the box is a `BoxSpec`, the numbers make sense.
2. **Check the box against every receptor.** A box that contains *no* receptor
   atom is refused: it means the receptor and the box come from different
   coordinate frames, and every molecule would come back with a huge positive
   energy. (The 3PTB box holds 335 of its 1994 atoms; the EGFR box holds 164 of
   its 2985; either box applied to the other receptor holds exactly zero.)
3. **Read the library** with
   [`odock.chem.ligand.read_ligands`](../python/odock/chem/ligand.py), one input
   file at a time, in file order. A `.pdbqt` library is used as-is — each `MODEL`
   becomes one library member.
4. **Filter** each molecule with [`odock.filters.drug_like`](../python/odock/filters.py)
   (Lipinski, Veber, PAINS) unless `--no-filter`.
5. **Prepare** each survivor into ligand PDBQT (embedding + optimisation +
   torsion tree) with `odock.prepare.prepare_ligand`.
6. **Cache** steps 3–5 in `library.csv` and `library/`, keyed by a hash of the
   inputs, the filter switch and the preparation settings. A restart with the
   same library reuses it — filtering and embedding a 100 000-molecule library
   costs far more than docking a handful of it.
7. **Read the results file back** and skip the `(receptor, ligand)` pairs
   already in it.
8. **Dock the rest** in a thread pool, writing one record per molecule as it
   completes (append + flush, `fsync` every `--checkpoint-every` molecules).
9. **Rank and report**: `summary.json`, `summary.csv`, the printed table, and
   the `--top N` shortlist of every receptor, plus an optional `--json-out`.
   With `--consensus`, the shortlist is then rescored by every force field (step
   6 of section 1.3) — still without docking anything.

The funnel's `N read` counts the records **RDKit could parse**. A line of a
`.smi` that is not a molecule is reported separately and is not counted:

```text
filters: 2 read -> 2 kept (0 removed by Lipinski/Veber/PAINS)
  note: 1 record(s) could not be parsed by RDKit and are NOT counted above
```

`library.csv` likewise holds one row per molecule that parsed; the unparsable
records are listed in `summary.json` under `library.unreadable`.

### 2.1 Why threads, and what `--jobs` means

The kernel releases the GIL for the duration of a search, so the pool gives real
parallelism without pickling anything, and each worker keeps the full
`DockResult` (coordinates, PDBQT text, atom order) that a process pool would have
to ship back.

`--jobs` is the number of *molecules* in flight; `0` (the default) means one per
core. Each docking also uses every core internally for its grid build, so more
jobs is not always faster: on a 16-core machine, ten molecules of the bundled
library at `--exhaustiveness 1` took 39 s at `--jobs 1`, 21 s at `--jobs 4` and
20 s at `--jobs 8` or `--jobs 16`. Measure with `--dry-run` and a `--limit` if
your library is large.

### 2.2 The kernel rebuilds the grid for every molecule

Every library member has its own torsion tree, so each docking job parses the
receptor and builds its own affinity grid. For a campaign that is the dominant
cost per molecule (~0.33 s for the 3PTB box) and it does not amortise over the
library. Two honest consequences:

* `--spacing 1.0` makes a much smaller grid and a much faster scan, at the price
  of a coarser search (the reported energies are still refined exactly, but the
  search explores a coarser surface). Treat it as a triage setting.
* a large box costs every molecule. `--dry-run` prints the grid point count in
  `library.csv`-adjacent terms; `estimate_library()` reports it in Python.

---

## 3. Reference

### 3.1 Options

| Option | Default | Meaning |
|---|---|---|
| `-r`, `--receptor FILE` | required | receptor PDBQT; **repeat** for a panel |
| `-i`, `--input FILE` | required | library file (`.sdf`/`.smi`/`.mol2`/`.mol`/`.pdb`/`.pdbqt`); repeat to combine |
| `-o`, `--out DIR` | required | output directory |
| `--box FILE` | — | box JSON written by `odock box` |
| `--center X Y Z`, `--size X Y Z` | — | explicit box |
| `--spacing Å` | box value, else `0.375` | grid spacing |
| `-s`, `--scoring` | `vina` | `vina`, `vinardo` or `ad4` |
| `-e`, `--exhaustiveness` | `8` | search effort per molecule |
| `-n`, `--num-poses` | `9` | poses kept per molecule |
| `--min-rmsd` | `1.0` | pose deduplication cutoff (Å) |
| `--energy-range` | `3.0` | reporting window (kcal/mol) |
| `--no-grid`, `--no-refine` | off | as in `odock dock` |
| `--search`, `--islands`, `--population`, `--generations` | as `odock dock` | search protocol |
| `--seed N` | `0` | campaign seed; `0` draws one and prints it. Part of the campaign identity: changing it on resume is refused |
| `--jobs N` | `0` (one per core) | molecules docked concurrently |
| `--timeout SECONDS` | — | per-molecule wall-clock limit |
| `--checkpoint-every N` | `20` | records between `fsync`s and manifest refreshes (every record is flushed) |
| `--no-filter` | off | dock even the non-drug-like molecules |
| `--no-optimize` | off | skip the ligand pre-optimisation |
| `--limit N` | — | dock only the first N molecules (a trial run) |
| `--top N` | `0` | write the N best molecules as a multi-model PDBQT |
| `--csv` / `--jsonl` | `jsonl` | results format |
| `--no-resume` | off | discard this directory's results and start over |
| `--no-interactions` | off | skip the per-pose interaction profiling |
| `--no-poses` | off | no pose file per molecule (and no shortlist) |
| `--dry-run` | off | report the filtered library and the cost, dock nothing |
| `--allow-box-mismatch` | off | dock even when the box misses the receptor |
| `--consensus` | off | rescore the shortlist with vina/vinardo/ad4 and rank the hits by agreement |
| `--consensus-top N` | `--top`, else `50` | molecules covered by `--consensus` |
| `--consensus-method` | `rank` | `rank`, `borda` or `z` — how the fields are combined |
| `--json-out FILE` | — | write the run summary as JSON |
| `-q`, `--quiet` | off | no progress line, no table |

### 3.2 `--seed` makes the campaign reproducible

`--seed 42` makes the *whole* run deterministic: every molecule gets a seed
derived from `crc32(receptor | ligand key)` mixed with the campaign seed, so
molecule *N* is docked identically whether it was docked in the original run or
after a restart. Ligand preparation uses a fixed ETKDG seed (`20240101`), so the
prepared geometry does not depend on the docking seed either.

Because a molecule's seed depends on the campaign seed, the seed is part of the
campaign identity: resuming with a different `--seed` is refused (section 3.6)
rather than mixing two seeds in one results file.

`--seed 0` (the default) draws a seed, prints it, records it in `run.json`, and
reuses it on resume.

### 3.3 Per-molecule timeouts

```bash
odock screen ... --timeout 120
```

A molecule that exceeds the limit is cancelled (the kernel checks its cancel
token at the next Monte-Carlo step, so relief is prompt but not instantaneous)
and recorded with `status: "timeout"`. The run continues; timeouts are counted
separately from failures. A timed-out molecule is *recorded*, so a resume does
not retry it forever — delete the record if you want another try.

Use it when a library contains a few pathological molecules (a long flexible
chain, a ligand that keeps hitting the refinement failure path): one such
molecule otherwise holds a slot for as long as it likes.

### 3.4 Exit codes

| Code | Meaning |
|---|---|
| `0` | at least one molecule docked; partial failures still exit `0` |
| `2` | nothing to do, or the inputs contradict each other (empty library, every molecule filtered out, a box that misses the receptor, a results directory from another campaign, a missing file) |
| `3` | the library was fine and **every** docking failed or timed out |
| `130` | interrupted; the completed molecules are on disk and the same command resumes |

### 3.5 Files in the output directory

| Path | Contents |
|---|---|
| `results.jsonl` (or `results.csv`) | one record per molecule × receptor, appended as they complete |
| `summary.json` | the run manifest: config, funnel, counts, ranked records |
| `summary.csv` | the ranked table of every receptor, failures last |
| `library.csv` | **every** molecule read: descriptors, filter verdict, violations, prepared PDBQT path |
| `library/<n>_<name>.pdbqt` | the prepared ligand of each survivor (the cache) |
| `poses/<receptor>/<n>_<name>.pdbqt` | the poses of one molecule, multi-model |
| `top_<receptor>.pdbqt` | the `--top N` shortlist: the best pose of each of the N best molecules, one `MODEL` each |
| `top_<receptor>.csv` | the same shortlist as a table |
| `consensus_<receptor>.csv` | the `--consensus` hit table: per-field affinity and rank, consensus rank, whether the docked mode survived |
| `consensus_<receptor>.json` | the same plus the field correlations and the full pose-level ranking |
| `run.json` | the identity of the campaign: the two hashes, the seed, the measured rate |

A `results.jsonl` record (the real record of the first molecule of the run above,
with its `poses` and `interactions` lists shortened):

```json
{
  "receptor": "receptor", "ligand": "library.smi#1", "name": "benzamidine",
  "source": "demo/library.smi", "index": 0, "status": "ok",
  "affinity": -5.900566019865151,
  "n_heavy": 9, "n_atoms": 12, "n_torsions": 1,
  "molecular_weight": 120.155,
  "ligand_efficiency": 0.6556184466516835, "le_source": "odock.metrics",
  "key_residues": "ASP189, GLY219, VAL213", "n_interactions": 4, "in_box": true,
  "seed": 872650673, "elapsed": 6.801, "grid_points": 150920,
  "pose_file": "poses/receptor/000001_benzamidine.pdbqt",
  "poses": [{"index": 0, "affinity": -5.900566019865151, "rmsd_lb": 0.0,
             "rmsd_ub": 0.0, "in_box": true},
            {"index": 1, "affinity": -5.874014133974516, "rmsd_lb": 0.0825,
             "rmsd_ub": 1.9375, "in_box": true}],
  "properties": {"MW": 120.155, "LogP": 0.9707, "HBD": 2.0, "HBA": 1.0,
                 "RotB": 1.0, "tPSA": 49.87},
  "interactions": [{"kind": "hbond", "subtype": "",
                    "receptor_atom": 1415, "ligand_atom": 8,
                    "distance": 2.912215994736656,
                    "detail": "LIG1:N9->GLY219:O"}],
  "violations": [], "error": ""
}
```

A failed molecule has the same shape with `status: "failed"` and a populated
`error`; a timeout has `status: "timeout"`.

### 3.6 Resuming, and why a changed campaign is refused

`run.json` records two hashes:

* `library_hash` — the input files (path, size, mtime), the filter switch, the
  preparation settings. It decides *which* molecules are docked.
* `docking_hash` — the receptors, the box, the force field, the search settings
  **and the campaign seed**. It decides *what the numbers mean*. The seed belongs
  here because every molecule's docking seed is derived from it: resuming a
  `--seed 7` campaign with `--seed 8` would dock the shared molecules under a
  second seed and put two different experiments in one file, so it is refused.

When a setting changed, the refusal names it:

```text
odock screen: error: the docking settings changed since this output directory was written
  (stored c5728c14885c0726, now 569b1a1a9e3e0cbc).
       changed: seed: 7 -> 8
       Mixing two campaigns in one results file would produce incomparable numbers -- and,
       for a different seed, would dock the shared molecules under a second seed.
```

The way out is a new `-o` directory (recommended, keeps both campaigns) or
`--no-resume` (discards the old results, explicitly).

A `--dry-run` writes `run.json` too, but a dry run docks nothing, so it does not
constrain a later run.

#### The manifest is written before the first docking

`run.json` is written as soon as the library has been read, and refreshed at
every `--checkpoint-every` molecules and after every receptor. That is what makes
a **hard kill** resumable — `SIGKILL`, a lost node, a pulled plug. The result
rows carry no state of their own, so a campaign killed after 3 of 17 molecules
leaves a manifest that says which campaign those 3 rows belong to, and the
restart reports:

```text
resume: 3 molecule x receptor row(s) already in results.jsonl
=== demo/3ptb/receptor.pdbqt: 12 to dock of 15 (3 already done) ===
```

If the results file survives but `run.json` does not (a run killed by an older
release, or a lost manifest write), those rows are **still** resumed — never
silently re-docked — with a warning and three cheap consistency checks:

* **the file must be this campaign's.** A row naming a receptor this run does not
  screen makes the whole file another campaign's output; the run stops and names
  the foreign receptor instead of appending a second campaign behind a summary
  that would never mention it:

  ```text
  odock screen: error: results.jsonl belongs to another campaign: it holds 3 row(s) for
  the receptor(s) 'receptor', which this run does not screen (it screens 'egfr_receptor').
  ```

* the **campaign seed is recovered from the rows themselves**. Every row stores
  the per-molecule seed it was docked with, and that seed is
  `campaign_seed + crc32(receptor|ligand)`; the restart inverts the formula, so
  `odock screen` with no `--seed` continues the *same* campaign and prints
  `campaign seed recovered from the results file: 7`;
* the rows must have been docked on the grid this box produces.

If the rows disagree with each other — they were docked under more than one seed,
which is what an older release left behind when it accepted `--seed 8` on a
`--seed 7` campaign — the run stops and says the file already mixes two
campaigns. A file belonging to another campaign, or mixing two seeds, is refused,
never appended to; `--no-resume` is the way to discard it.

### 3.7 Panels: several receptors

```bash
odock screen -r wt/receptor.pdbqt -r mutant/receptor.pdbqt \
             -i library.smi --box site.json -o panel --top 20
```

One result set per receptor: `receptor` distinguishes the rows, `top_wt_receptor.pdbqt`
and `top_mutant_receptor.pdbqt` are separate shortlists, and `summary.csv` holds
every receptor. The receptors share **one** box, so they are expected to share a
coordinate frame — that is what a mutant/isoform/homology panel is. If a
receptor does not contain a single atom of the box, the run refuses rather than
producing meaningless numbers.

The receptor name in the results is the file stem, qualified by its parent
directory when two receptors would otherwise collide
(`wt/receptor.pdbqt` → `wt_receptor`).

### 3.8 Reading the results from Python

```python
from odock.screen import read_records, screen_ligands

records = read_records("results/results.jsonl")     # or results.csv
hits = sorted((r for r in records if r.status == "ok"), key=lambda r: r.affinity)
for record in hits[:10]:
    print(record.name, record.affinity, record.ligand_efficiency, record.key_residues)
```

`read_records` tolerates a truncated final line, which is exactly what a killed
process leaves behind: the record it was writing is simply redone on resume.

### 3.9 Consensus from Python

```python
from odock.screen import ScreenConfig, run_consensus, screen_ligands

summary = screen_ligands(ScreenConfig(..., top=50, consensus=True))
report = summary.consensus[summary.receptor_names[0]]
print(report.agreement, report.table())          # mean pairwise rho + hit table
print(report.rows[0])                            # per-field scores and ranks

# Or run it on its own, against a summary you already have:
report = run_consensus(config, summary, "receptor", receptor_text)
```

`run_consensus` never raises: a missing `odock.consensus` or an unhappy one
lands in `report.error` and the campaign is untouched.

### 3.10 Cost estimation in Python

```python
from odock.screen import estimate_cost, estimate_library, grid_points

print(grid_points(box))                        # 150920 for the 3PTB demo box
print(estimate_cost(n_ligands=1000, n_torsions=3, n_atoms=25, exhaustiveness=8))
```

`estimate_cost` is the model the dry run starts from; `estimate_library` sums it
over a prepared library and names the molecule that dominates.

---

## 4. Practical advice

* **Run `--dry-run` first.** It is one docking plus the library preparation, and
  it tells you the funnel, the cost and the molecule that will dominate it.
* **Filter before you dock.** A 100 000-molecule library where 40 % fails
  Lipinski or PAINS is 40 % of the campaign you do not need to pay for; the
  funnel in `library.csv` is what you cite when someone asks what happened to
  their compound.
* **Set `--timeout`** for a real library (a few times the dry run's
  per-molecule figure) so one pathological molecule cannot hold a core.
* **Use `--limit 200`** to sanity-check a new library and a new box before
  committing the whole campaign; then run it without the limit in the same
  directory — the first 200 are already done.
* **Finish with `--consensus`.** A shortlist of 50 hits costs a couple of minutes
  to rescore with all three force fields, and it is the difference between "the
  top 10 by vina" and "the 10 that every force field agrees about".
* **Keep the seed** you used in the paper with the results: `run.json` records
  it.
* **A single-seed result is a shortlist, not a verdict — and this is measured,
  not a disclaimer.** The bundled benchmark
  ([BENCHMARK.md](BENCHMARK.md)) re-docks five crystal complexes with three seeds
  each. On the flexible system (erlotinib in EGFR, 29 heavy atoms, 11 rotors) the
  *top-scoring* pose is **4.69 Å** from the crystal at seed 42, **1.51 Å** at seed
  7 and **1.70 Å** at seed 2024 — a **3.2 Å** swing in the headline number of the
  same calculation on the same machine — while the pose within 1 kcal/mol of the
  top stays at 1.28–1.44 Å in every seed. On the rigid system (benzamidine in
  trypsin) the same three seeds give 1.13 Å, 1.13 Å, 1.13 Å, spread 0.00.
  So: a rigid, well-behaved ligand reproduces exactly, and a flexible one does
  not, which is a property of the search landscape rather than of this pipeline.
  Before ordering two hits that differ by less than ~1 kcal/mol, or quoting the
  top-1 rank of a flexible ligand, run the campaign with a second `--seed` (a new
  `-o` directory, the same inputs) and check that the ordering survives. The
  benchmark exists so that this advice is a number rather than a warning.
* **The box is the physics.** `odock screen` refuses a box that does not touch
  the receptor, but it cannot know that you meant a different pocket. Check the
  printed box centre against `odock box`'s output, and use
  `--allow-box-mismatch` only when you know what you are doing.

---

## 5. See also

* [USER_GUIDE.md](USER_GUIDE.md) — the rest of the command line and the Python API.
  (Its FAQ used to say "there is no batch subcommand"; `odock screen` is it.)
* [SCORING.md](SCORING.md) — what the numbers mean.
* [BENCHMARK.md](BENCHMARK.md) — how accurate a run is, and how much it moves
  with the seed; `python -m odock.benchmark --check-baseline` is the regression
  gate for the physics.
* [SCIENCE.md](SCIENCE.md) — the efficiency metrics, the strain calculation and
  the fingerprint/pharmacophore summaries this pipeline feeds.
* Scaling further: `odock screen` already uses every core. On a cluster, run one
  process per receptor (`-r` once each, `-o` per receptor) or shard the library
  with `--limit`/`-i` per node — the results of a shard are ordinary result files
  and can be concatenated.

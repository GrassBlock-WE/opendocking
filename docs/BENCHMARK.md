# The OpenDocking re-docking benchmark

`python -m odock.benchmark` re-docks a small set of bundled crystal complexes
through the documented `odock` path and reports what a docking engine is judged
on: how close the top pose is to the crystal, whether the correct binding mode
was found at all, how well the score orders the poses, whether the three force
fields agree, and how much all of that moves when the seed changes.

It exists because unit tests cannot see this class of regression. The Rust suite
verifies every analytic derivative against finite differences; none of those
tests would notice a scoring change that is numerically perfect and physically
worse. `benchmark/baseline.json` records what the engine measured when the
baseline was cut, and `--check-baseline` fails when a change makes a system worse
than that by more than the recorded tolerance.

```bash
# the full benchmark: five systems, three seeds (about 20 minutes on 8 cores)
python -m odock.benchmark

# a quick look: half the search effort, the two cheap systems
python -m odock.benchmark --systems 3PTB,1STP --seeds 42 --fast

# the regression check, and re-recording the baseline after an intended change
python -m odock.benchmark --check-baseline
python -m odock.benchmark --write-baseline
```

## The systems

Every system is defined by data that is in the repository: the crystal file in
`tests/data/` plus the ligand's SMILES, taken from the RCSB chemical component
dictionary (`<ID>_ideal.sdf`) and recorded in
[`benchmark.py`](../python/odock/benchmark.py). Nothing is downloaded at run
time, and the benchmark is reproducible offline.

| PDB | ligand | protein | heavy atoms | rotors | resolution | why it is here |
|---|---|---|---|---|---|---|
| 3PTB | BEN | bovine trypsin | 9 | 1 | 1.70 Å | the acceptance test: small, stiff, must be exact |
| 1STP | BTN | streptavidin | 16 | 5 | 2.60 Å | very buried, high affinity: does the search converge |
| 3ERT | OHT | ERα ligand-binding domain | 29 | 9 | 1.90 Å | a flexible drug in a large, mostly hydrophobic site |
| 1M17 | AQ4 | EGFR kinase domain | 29 | 11 | 2.60 Å | the documented force-field limit (11 rotors) |
| 1HVR | XK2 | HIV-1 protease | 46 | 8 | 1.80 Å | the hardest case here: 46 heavy atoms |

The SMILES matters more than it looks. A ligand prepared from a bare PDB has no
bond orders; RDKit then sees a benzene ring as cyclohexane, and the engine is
measured through a handicap that has nothing to do with the engine. Every system
here is prepared with `smiles=`, which is also the documented recipe
(see [`SCIENCE.md`](SCIENCE.md) §3).

## What is measured

| metric | definition | direction |
|---|---|---|
| `top_rmsd` | heavy-atom RMSD of the top-ranked pose to the crystal, **no superposition** | lower is better |
| `fitted_rmsd` | the same pose after optimal, symmetry-aware superposition | lower is better |
| `best_within_1kcal` | smallest no-superposition RMSD among poses within 1 kcal/mol of the best | lower is better |
| `rank_of_correct` | 1-based rank of the first pose within 2 Å (no superposition); 0 when none | lower is better |
| `score_rmsd_rho` | Spearman between the pose scores and their no-superposition RMSD | **negative** is better |
| `agreement` | mean pairwise Spearman between Vina, Vinardo and AD4 rescoring the same poses | higher is better |

`top_rmsd` with no superposition is the crystallographic figure of merit: the
docking had to find the experimental pose in the same coordinate frame. The
fitted number is reported beside it because the two together separate "found the
wrong site" from "found the right site, slightly rotated".

`best_within_1kcal` is the fair measure for a flexible ligand. When several poses
sit within a few hundredths of a kcal/mol, the *ranking* between them is not
meaningful even though the correct binding mode is in the output; the standard
"top-N success" figure asks the question that can actually be answered.

**Every correlation is reported with its sample size and its spread across
seeds.** A rho over three poses is not a measurement, and a rho is not a property
of a receptor: measured on this project's own screening campaign, the mean
force-field agreement was 0.76 on one run and 0.82 on another with identical
settings, while the *ordering* of the pairs was stable both times (Vina~AD4
weakest). The benchmark therefore prints `rho (n)` and a range, never a bare
number.

## The measured baseline

Recorded 2026-10-01 with `python -m odock.benchmark` (five systems, seeds 42/7/2024,
Vina, `num_poses=20 energy_range=5.0 min_rmsd=0.5`). The values are in
[`benchmark/baseline.json`](../benchmark/baseline.json) together with the exact
command, the environment and the tolerances.

The recorded run was then **repeated on the settled revision**, with all three
teammates' work merged: the two runs differ in **0 fields across 5 systems and 15
docking runs** — every per-seed top RMSD, `best ≤1 kcal`, rank, correlation,
agreement, pose count and affinity, and every aggregate. That is not a
coincidence to celebrate; it is the property the tolerance depends on, and it is
why the gate is a real gate rather than a coin toss.

| system | ligand | heavy | rotors | poses | top RMSD | seed spread | best ≤1 kcal | rank | rho (n) | rho range | agreement range | weakest pair |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3PTB | BEN | 9 | 1 | 15 | **1.13 Å** | 0.00 | 1.13 Å | 1 | +0.78 (15) | +0.72…+0.85 | +0.82…+0.91 | vina~ad4 |
| 1STP | BTN | 16 | 5 | 1 | **1.03 Å** | 0.01 | 1.03 Å | 1 | — | — | — | — |
| 3ERT | OHT | 29 | 8 | 8 | **1.51 Å** | 0.43 | 1.35 Å | 1 | +0.62 (8) | +0.43…+0.81 | +0.81…+0.88 | vinardo~ad4 |
| 1M17 | AQ4 | 29 | 11 | 9 | **1.51 Å** | **3.17** | 1.28 Å | 1 | +0.55 (9) | +0.32…+0.79 | +0.52…+0.69 | vina~ad4 |
| 1HVR | XK2 | 46 | 8 | 2 | **10.07 Å** | 0.00 | **0.79 Å** | 2 | −1.00 (2) | — | −0.33…+1.00 | vinardo~ad4 |

"top RMSD" is the best over the three seeds and "seed spread" the difference
between the best and worst seed, so the pair of columns separates *accuracy* from
*stability*. "rank" is the best rank over the seeds; `correct_in_all_runs` is
true for every system here — the crystal binding mode is found by every seed of
every system.

Three things in that table are worth more than the averages:

* **1M17's top-pose RMSD moves 3.17 Å between seeds** — 4.69 Å at seed 42,
  1.51 Å at seed 7, 1.70 Å at seed 2024 — while `best ≤1 kcal` stays at
  1.28–1.44 Å in every seed. The search finds the experimental binding mode every
  time; the *ranking* of near-degenerate poses is what moves. A user who runs one
  seed and reads the top pose is looking at 3 Å of seed noise on this system.
* **1HVR's top pose is 10.1 Å from the crystal in every seed**, yet a pose within
  1 kcal/mol of it reproduces the crystal mode to **0.79–1.10 Å** at rank 2. The
  correct answer is in the output, one rank below the top, consistently. This is
  the strongest case in the set for reporting `best ≤1 kcal` beside `top RMSD`.
* **The weakest force-field pair always involves AD4** (vina~ad4 twice,
  vinardo~ad4 twice), but *which* pair it is depends on the system, and the
  absolute agreement ranges from −0.33 to +1.00. So "the fields agree with rho
  0.8" is not a claim anyone should make; "AD4 is the outlier, and the mean
  agreement moves by ~0.2 between seeds and systems" is.

### Worked example: 1HVR, where `top RMSD` alone is a lie

XK263 in HIV-1 protease, 46 heavy atoms, eight rotors, the hardest system here.
Every seed tells the same story:

```
seed    42        7      2024
top RMSD     10.07 A  10.08 A  10.08 A     <- the pose the score prefers
rank of the crystal mode      2        2        2
RMSD of that pose           1.04 A   0.79 A   1.10 A   <- 1.10 A from the crystal
affinity gap between them   < 1 kcal/mol
```

Read the first line alone and the engine failed by 10 Å. Read the third and it
reproduced the experimental binding mode to under 1.1 Å. Both are true: the
search found the crystal mode, and the force field prefers a *different* minimum
less than a kcal/mol away, stably across seeds. That is not a search failure and
it is not noise — it is an empirical scoring function doing what empirical
scoring functions do on a ligand with eight rotors.

The consequence is practical: **a benchmark that reported only `top RMSD` would
score this system as a catastrophic failure and hide the fact that the answer is
in the output.** `best ≤1 kcal` and `rank_of_correct` exist precisely so that the
report distinguishes "did not find it" from "found it and ranked it second", and
this is the case that proves the distinction matters. The same effect appears in
milder form on 3ERT (rank 1, 1.5 Å) and 1M17 (rank 1–3, 1.3–1.4 Å).

**Is the ranking stable across seeds?** Two different answers, and the difference
is the point:

* The **top binding-mode cluster is stable**: `consensus.reproducibility` is
  **1.0 for all five systems** — pooling the poses of the three seeds and
  clustering them by symmetry-aware RMSD, every seed contributes to the cluster
  that holds the overall best-scoring pose. The seeds agree about which mode the
  score prefers.
* The **identity of that mode is not**: on 1M17 the top pose is 4.69 Å from the
  crystal at one seed and 1.51 Å at another; on 1HVR all three seeds agree on a
  top pose 10.1 Å away while the crystal mode sits at rank 2 inside 1 kcal/mol.
* The **ranking quality moves with the seed**: `score_rmsd_rho` spans
  +0.32…+0.79 on 1M17 and +0.43…+0.81 on 3ERT, and the force-field agreement
  range is ±0.1–0.2 wide. A single-seed run therefore cannot support a
  0.5 kcal/mol ordering, which is exactly the number this benchmark exists to
  put in front of a user.

**Hardware and wall time.** Windows 10.0.26200 (Windows 11), AMD Ryzen
(AMD64 Family 25 Model 68), **16 logical cores**, Python 3.10.6, kernel 0.1.0 —
all recorded in `benchmark/baseline.json`, because a timing or an RMSD without
the machine it was measured on is not a measurement. Wall time for the recorded
run: **26 minutes** for 5 systems × 3 seeds (3PTB 9 s, 1STP 42 s, 3ERT 328 s,
1M17 346 s, 1HVR 830 s; preparation is under a second per system). `--fast`
halves each system's exhaustiveness and takes about a quarter of the time;
`--systems 3PTB,1STP --seeds 42` is under a minute and is what a smoke check
should use.

## Why AD4 disagrees with Vina (measured, not asserted)

The kernel exposes the component split (inter/intra/unbound/torsion) but not
per-term values, so the two terms Vina does not have at all — AD4's screened
electrostatics and its charge-dependent desolvation — were removed instead, by
rewriting every receptor charge to zero (`out/ad4_ablation.py`):

| system | poses | rho(vina,vinardo) | rho(vina,ad4) as shipped | rho(vina,ad4) with all charges zeroed | AD4 rank changes | AD4 spread vs Vina spread |
|---|---|---|---|---|---|---|
| 3PTB | 6 | +0.94 | +1.00 | **+1.00** | 0/6 | 2.47 vs 1.80 kcal/mol |
| 1M17 | 9 | +0.92 | +0.50 | **+0.48** | 4/9 | 3.19 vs 0.23 kcal/mol |

* Removing **every** charge moves AD4's ranking on 4 of 9 erlotinib poses but
  moves its correlation with Vina by **0.02**. The disagreement is therefore not
  carried by electrostatics or desolvation.
* It is not the zeroed non-finite Gasteiger charges either: the EGFR receptor has
  **none** (the 3PTB receptor has 128 zeroed charges) and the disagreement is the
  same there, so the preparation defect does not explain it.
* The scale is the difference. On the same nine poses AD4 spans **3.19 kcal/mol**
  where Vina spans **0.23**: AD4's stiffer 12-6 van der Waals term, explicit
  12-10 H-bonds and desolvation amplify differences Vina barely resolves, and
  poses 0.23 kcal/mol apart under Vina get reordered. That is a genuine
  force-field difference — the honest reading is *not* "AD4 is broken", it is
  "AD4's numbers are on a different scale and its ordering of near-degenerate
  poses should not be averaged with Vina's".

Consequence for the consensus module: averaging AD4's *ranks* with Vina's is
appropriate; averaging its *energies* would not be (which is why
:mod:`odock.consensus` aggregates ranks, see [`SCIENCE.md`](SCIENCE.md) §1).

## What the regression check does, and what it deliberately does not

`--check-baseline` re-runs the benchmark with the seeds recorded in the baseline
and fails when a system is worse by more than the recorded tolerance (0.15 Å on
`top_rmsd` / `best_within_1kcal`, one rank, 0.2 on `score_rmsd_rho`).

**Why the tolerance is not the seed spread — do not "fix" this back.** The check
re-runs *the same command with the same seeds*, so the only noise it has to
absorb is **repeat-run noise**. That was measured, not assumed: two runs of 3PTB
and of 1STP with identical settings agree to **six decimals** (the engine is
bit-for-bit deterministic on a given machine, as the README claims), so the
tolerance is `max(0.15 Å, measured repeat noise)` — the floor exists only so a
different core count reordering the parallel reduction cannot flip the gate.

The **seed-to-seed spread** is a different quantity, an order of magnitude
larger (up to **3.17 Å** for 1M17), and it says how much the *system* moves when
the search restarts. It is recorded per system as `top_rmsd_spread` and reported
in the table — that is its job. Using it as the tolerance was tried first and
produced a **4.76 Å** allowance, under which no plausible regression would ever
fail: a check that exists on paper and not in practice. The derivation is stored
in `benchmark/baseline.json` itself, next to the number, so the reasoning travels
with the file.

The check also warns when the run's seeds differ from the baseline's, because a
best-over-seeds can only improve with more seeds and the comparison would not be
like for like.

## What this does not establish

Read this before quoting any number above.

* **Five systems is not CASF.** The standard docking benchmarks use dozens to
  hundreds of complexes across many protein families. Five well-behaved,
  drug-like, single-ligand complexes measure a regression, not a capability.
* **The receptor is rigid.** Side chains and the backbone are fixed, waters are
  stripped. A binding mode that needs a side-chain rearrangement will be missed
  by the engine *and* by this benchmark.
* **One preparation protocol.** Every ligand is prepared with a CCD SMILES
  template and default settings. Different protonation states, tautomers or
  embedding seeds are not exercised.
* **The crystal pose is the reference, and it is a model.** A 2.6 Å structure's
  coordinates carry roughly 0.3-0.5 Å of experimental uncertainty, so a
  "0.4 Å" and a "0.8 Å" top pose are not distinguishable here.
* **`rank_of_correct` and `score_rmsd_rho` measure the ranking, not the search.**
  A run can find the crystal pose and rank it fourth (1M17 does) — that is a
  force-field observation, and the benchmark reports it rather than hiding it in
  an average.
* **Nothing here is a comparison with another engine.** The numbers are
  OpenDocking's own, measured on this hardware with these settings. Reproducing
  them elsewhere is the point; beating someone else's is a different exercise,
  and would need the same systems, the same preparation and the same box.

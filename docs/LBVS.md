# Ligand-based virtual screening — a benchmark with decoys, controls and intervals

`docs/CHEMINFORMATICS.md` ends with a caveat: "six actives in seventeen molecules
is a smoke test, not an enrichment claim". This document is what removes that
caveat — and, where it cannot be removed, measures how far it can be pushed.

Three things make an enrichment number mean something, and this layer implements
all three:

1. **Decoys that are matched to the actives** (`odock.decoys`) — matched on MW,
   LogP, HBD, HBA, rotatable bonds and net charge within stated tolerances, and
   topologically dissimilar so they are not analogues in disguise. The property
   distributions are reported side by side with their standardised mean
   differences, because an unmatched decoy set inflates every metric.
2. **Metrics with intervals** (`odock.lbvs`) — EF1 %, EF5 %, AUC and BEDROC(α = 20)
   with **bootstrap confidence intervals over molecules**, plus a random ranking
   and a property-only ranking as controls. "No signal" and "trivial signal" are
   printed in the same table as the methods.
3. **Leave-one-out scoring** — the actives are their own queries otherwise. Score
   each active against the *other* actives (2-D), against a model that never saw
   it (pharmacophore), or against a reference active that is not itself (shape).
   Without this every method reports AUC 1.000 for free, and the measured leak is
   in §4.

```bash
odock decoys -a demo/library.smi -p demo/decoys.smi -n 5 -o out/decoys.smi --json-out out/decoys.json
odock lbvs   -a actives.smi -d out/decoys.smi --methods fingerprint,pharmacophore,overlay,crude \
             --shape overlay --conformers 2 --bootstrap 200 --prefilter 0.05 \
             --prefilter-method usr --json-out out/lbvs.json
```

The shape row has **two engines**: `--shape overlay` (the default — analytic Gaussian
shape and electrostatic fields at an optimised pose, per-term reporting, best over
conformers) and `--shape crude` (the original single-pose grid overlay).  Both are
kept, both are measured on the same actives and decoys in §2 and §3, and the delta is
reported whether or not it flatters the new code.  `docs/CONFORMERS.md` is the
reference for the ensembles behind both, for the overlay's parameters and for the
pre-filter's recall.

## 1. The decoy set, and the evidence that it is matched

`demo/decoys.smi` is a **hand-curated pool of 125 drug-like molecules** —
metabolites, drugs, fragments, amino acids and aminoheterocycles — written out in
this repository because a DUD-E download is not possible offline and a vendor
catalogue would not be redistributable. It is a *pool*, not a decoy set:
`match_decoys` selects from it.

Selecting five decoys for each of the five ring-amidines gives 25 molecules, and
the quality report is the part that matters:

```text
property            actives            decoys     SMD
MW          136.65 +- 12.28   134.17 +- 15.12    0.17
LogP           1.12 +- 0.35      1.07 +- 0.45    0.12
HBD            2.20 +- 0.45      1.68 +- 0.63    0.86
HBA            1.20 +- 0.45      1.76 +- 0.60   -0.97
RotB           1.20 +- 0.45      1.32 +- 0.69   -0.18
charge         0.00 +- 0.00      0.00 +- 0.00    0.00
max |SMD| 0.97: NOT matched (|SMD| > 0.5)
```

**The tool says its own decoy set is not matched, and it is right.** MW, LogP,
rotatable bonds and charge are matched to two tenths of a standard deviation; HBD and
HBA are not (SMD 0.86 and −0.97), because the amidines have two to three donors
and one to two acceptors while the pool's amines, amides and aminoheterocycles
tend to have fewer donors and more acceptors. The matching is per-active (every
decoy is inside *its* active's tolerances), and the set-level refinement that
follows (a swap pass that reduces the worst |SMD|) cannot manufacture molecules
the pool does not contain. Seven `min_similarity`-band runs and three pool
expansions later, that is the floor of what this pool can do, and the benchmark
reports it rather than quoting the enrichment factors alone.

Two more reported facts about the set:

* **Every selected decoy is inside the topology ceiling**: `max Tanimoto to any
  active` ranges 0.14-0.35, and the analogues are listed as rejected
  (`aniline`, `phenol`, `benzoic_acid`, `4_aminophenol`, `4_fluoroaniline` — the
  last two are genuine analogues of the hydroxy and fluoro amidines).
* **The similarity floor is a knob worth having.** With `--min-similarity 0.3`
  the decoys are *hard* (they look like the actives without being analogues),
  which is the band a method has to be able to lose on — see §3.

## 2. The benchmark, easy decoys: two engines, and the new one loses

5 actives (the ring-amidines) + 25 easy decoys (Tanimoto < 0.35), all methods
leave-one-out, ETKDGv3 seed 20240101, 2 conformers, 200 bootstrap resamples:

| method | EF1 % | EF5 % | AUC | BEDROC(20) | 95 % CI (AUC / BEDROC) |
|---|---|---|---|---|---|
| fingerprint (Morgan r2) | 6.00 | 6.00 | **1.000** | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| pharmacophore fit | 6.00 | 6.00 | **1.000** | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| shape + electrostatics, **overlay** (new default) | 6.00 | 6.00 | **0.920** | 0.941 | [0.79, 1.00] / [0.47, 1.00] |
| shape only, **overlay** | 0.00 | 0.00 | 0.864 | 0.549 | [0.71, 0.97] / [0.38, 1.00] |
| shape + electrostatics, **crude grid** | 6.00 | 6.00 | **1.000** | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| shape only, **crude grid** | 6.00 | 6.00 | 0.992 | 0.990 | [0.95, 1.00] / [0.84, 1.00] |
| *random* (control) | 0.00 | 3.00 | 0.656 | 0.633 | [0.37, 0.93] / [0.40, 0.98] |
| *property: MW* (control) | 0.00 | 0.00 | 0.480 | 0.437 | [0.17, 0.76] / [0.34, 0.72] |
| *property: LogP* (control) | 0.00 | 0.00 | 0.448 | 0.417 | [0.23, 0.77] / [0.29, 0.65] |

EF1 % = 6.00 means "the top 1 % of 30 molecules is one molecule, and it is an
active, while a random pick would be an active once in six" — the maximum the set
size allows. Read the table as:

* **The controls behave.** A random ranking lands at AUC 0.656 with a 95 %
  interval [0.37, 0.93] that contains 0.5, and the property-only rankings sit at
  0.448-0.480: no signal, and trivially wrong signal, both visibly present. Without
  those rows a reader cannot tell 1.000 from 0.98-with-a-lucky-split.
* **The rebuilt overlay does not improve the numbers, and it is reported that way.**
  The crude grid scorer is perfect here and the analytic overlay is not: it puts
  `anthranilamide` (0.889) and `anthranilic_acid` (0.840) above
  `hydroxybenzamidine` (0.759) and `fluorobenzamidine` (0.695).  The two are close
  analogues of the actives in shape — a benzene ring with a small substituent —
  and the free pose search lets each find the superposition that maximises its own
  overlap, which is precisely what the crude scorer's single pharmacophore-frame
  pose prevents.  §5 decomposes the loss; `docs/CONFORMERS.md` §6 has the variant
  table that shows the pose freedom is *not* the cause.
* **Neither interval separates the engines, and comparing intervals was the wrong
  test.** [0.79, 1.00] against [1.00, 1.00] on 5 actives and 25 decoys is a case
  study.  §8 does it properly — a **paired** bootstrap of the difference on the same
  molecules — and the answer is: overlay − crude = **−0.080 [−0.229, +0.000]**, with a
  minimum detectable difference of **0.173 AUC**.  The gap is half of what this set
  can detect, so the honest reading is "the new overlay did not beat the old one
  here", not "the old one is better".
* **All the methods that see an amidine are near-perfect anyway.** The actives are
  a *congeneric series* (five benzamidine analogues) and the decoys are
  deliberately dissimilar (Tanimoto < 0.35), which is the easy regime: any method
  that can see a phenylamidine wins.

## 3. The hard band: where the methods separate

Selecting decoys that are topologically *closer* to the actives
(`--min-similarity 0.35 --max-similarity 0.6`) yields only five decoys — the pool
runs out, and the shortfall is reported — but the table changes (the hard set's
max |SMD| is 1.37, driven by MW, HBD and RotB, so it is a *harder* set on the
metrics too):

| method | EF1 % | EF5 % | AUC | BEDROC(20) | 95 % CI (AUC / BEDROC) |
|---|---|---|---|---|---|
| fingerprint (Morgan r2) | 2.00 | 2.00 | **0.920** | 0.998 | [0.68, 1.00] / [0.87, 1.00] |
| pharmacophore fit | 2.00 | 2.00 | 1.000 | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| shape + electrostatics, **overlay** | 0.00 | 0.00 | **0.600** | 0.135 | [0.21, 1.00] / [0.00, 1.00] |
| shape only, **overlay** | 0.00 | 0.00 | 0.640 | 0.135 | [0.24, 1.00] / [0.00, 1.00] |
| shape + electrostatics, **crude grid** | 2.00 | 2.00 | **0.920** | 0.998 | [0.68, 1.00] / [0.87, 1.00] |
| shape only, **crude grid** | 2.00 | 2.00 | 0.880 | 0.984 | [0.62, 1.00] / [0.14, 1.00] |
| *random* (control) | 2.00 | 2.00 | 0.600 | 0.982 | [0.17, 1.00] / [0.02, 1.00] |
| *property: MW* (control) | 0.00 | 0.00 | 0.200 | 0.016 | [0.00, 0.57] / [0.00, 0.98] |

Here the new overlay is **at the random control** (AUC 0.600, BEDROC 0.135 against
0.600 / 0.982 for random): it puts `4_hydroxybenzamide` (0.990) and `benzoic_acid`
first, again molecules whose ring is the actives' ring and whose substituent is
small.  The crude grid scorer holds 0.920 and the pharmacophore model stays perfect,
because a decoy that resembles one active by shape need not present the
donor/acceptor/aromatic pattern the *series* shares. **That is the kind of statement
this benchmark can make**, and the wide interval is the price of saying it from five
decoys.

A third cut (`--min-similarity 0.5 --max-similarity 0.7`) finds **one** decoy:
with 1 decoy and 5 actives the metric degenerates (EF1 % k = 1, AUC defined over
five comparisons) and the tool reports the shortfall. That is the honest end of
the knob: a benchmark can be made hard, or it can be made well-powered, and this
pool cannot do both.

## 4. The leak that would have made all of this meaningless

The first version of this benchmark scored the library with the actives as the
queries — and the actives are *in* the library. Every active is 100 % similar to
itself, so the 2-D method, the pharmacophore model (built from all the actives)
and the shape overlay (referenced to one of them) all reported **AUC 1.000,
BEDROC 1.000, EF1 % 6.00** before any docking-like thinking happened. Three
"perfect" methods measuring their own training set is the most common way a
ligand-based benchmark lies.

The fix is in the API, not in the docs: `leave_one_out=True` is the default on all
three methods —

* the fingerprint method scores an active against the **other** actives;
* the pharmacophore method builds **one model per active, leaving it out**, and
  reports each molecule's mean fit over the folds;
* the shape overlay takes the best overlay over reference actives **excluding the
  molecule itself**.

The test that pins it (`test_fingerprint_scores_leave_the_molecule_out_of_its_own_query`)
asserts both directions: with `leave_one_out=False` the top score is exactly 1.0
and the note says "self-similarity NOT removed (leaky)"; with the default it is
below 1.0 and the note says "leave-one-out". The numbers in §2 and §3 are the
leave-one-out numbers.

## 5. Ligand-based pre-filtering for a docking campaign

The realistic use of these scores is a pre-filter: dock the top fraction of the
library instead of all of it, and accept the active recall it costs. Measured on
the 142-molecule library (the 17 demo molecules + the 125-molecule pool), with the
two filters the tool offers:

| filter | keep | molecules docked | actives kept | active recall | estimated docking |
|---|---|---|---|---|---|
| fingerprint (2-D) | 5 % | 8 of 142 | **5 of 5** | 100 % | 436 s → **25 s** (94 % saved) |
| fingerprint (2-D) | 10 % | 15 of 142 | 5 of 5 | 100 % | 436 s → 46 s (89 % saved) |
| USR (3-D) | 5 % | 8 of 142 | **2 of 5** | **40 %** | 436 s → 25 s (94 % saved) |
| USR (3-D) | 10 % | 15 of 142 | 2 of 5 | 40 % | 436 s → 46 s (89 % saved) |
| USR (3-D) | 20 % | 29 of 142 | 4 of 5 | 80 % | 436 s → 89 s (80 % saved) |
| fingerprint, 17 molecules only | 6 % | 2 of 17 | 2 of 5 | 40 % | — |

The workload figures are :func:`odock.screen.estimate_cost`'s own estimates (a
fixed per-molecule cost plus a search cost that grows with exhaustiveness and the
torsion count), not measurements of a run — the module is *imported*, never
edited, and the note in the JSON says so.

**The USR pre-filter's recall is the number that makes its speedup readable, and it
is bad**: a shape descriptor that ignores element identity cannot separate five
amidines from a pool of flat aromatics, so a 5 % cut throws away three of them.  The
2-D filter keeps all five at the same cut, for free, and the honest recommendation
is therefore *use the fingerprint filter unless the screen is genuinely 3-D*, and if
it is, take 20 % and re-measure the recall on the library in hand.  The USR
pre-filter also reports the overlay cost it avoids as a **measurement** (1.23 s on
the 8 kept molecules against 21.8 s extrapolated to the full library, 94 % saved)
and the descriptors themselves cost 0.165 s for all 142 — it is the conformer
ensembles, needed either way, that dominate.  `docs/CONFORMERS.md` §4 has the table
and the self-match row that shows why a recall must be leave-one-out.

## 6. What this benchmark does not establish

* **Decoys are not inactive molecules.** A decoy is a molecule nobody has reported
  binding *that* target; it is presumed inactive, not known to be. Every decoy set
  therefore contains an unknown number of undetected actives, which *depresses*
  the measured enrichment.
* **The decoy set is not property-matched, and the tool says so**: max |SMD| 0.97
  (easy set) to 1.37 (hard set) comes from HBD/HBA (easy) and MW/HBD/RotB (hard), and
  the direction of the bias is not predictable — a fingerprint may reward or punish
  the difference. Enrichment factors from this repository are therefore **case-study
  numbers with a visible caveat**, not DUD-E numbers.
* **One target is a case study.** All of this is benzamidine against trypsin's S1
  pocket, in a 30-molecule set of which five are actives. Nothing here transfers to
  another target, another chemotype or a real screening library.
* **The ensemble size was tested and is not the explanation.** §2 and §3 run with the
  published protocol's 2 conformers; re-running them at 16 and at the rotor-scaled
  default (which triples the conformers kept and takes coverage from 59 % to 69 % of
  the achievable ceiling) moves the overlay's AUC by 0.016 (easy) and 0.040 (hard) and
  leaves the crude grid scorer at 1.000 on the easy band — while making it *worse*
  (0.920 → 0.680) on the hard band, because a scorer that maximises over poses gains
  more chances from more geometries.  The five actives keep one conformer each at any
  attempt count, so they cannot move at all.  `docs/CONFORMERS.md` §7 has the table
  and the cost (4–5× the embedding time).
* **Congeneric actives make the easy regime trivial.** With five close analogues
  and dissimilar decoys every method is perfect; the hard band separates them, and
  both are reported, but a benchmark that can only be run on one series cannot
  rank methods in general.
* **The metrics are not interchangeable.** BEDROC answers "how early", EF1 %
  "what is in the very top", AUC "the whole list". They disagreed in §3 (BEDROC
  0.998 with AUC 0.920) because BEDROC's exponential weight is concentrated where
  the actives are. Quoting one without the others is how a metric becomes a
  headline.
* **This set cannot rank the two shape engines**, and §2/§3 say so with overlapping
  intervals.  The measured verdict is narrower and must not be over-read: the
  analytic overlay did not beat the single-pose grid scorer on the bundled decoy
  bands, and on the hard band it did not beat the random control.  A shape score
  is geometric similarity, never a binding prediction, and the crude grid scorer is
  still there (`--shape crude`) because on this evidence it is the better *ranker*.
* **A fingerprint benchmark measures the fingerprint.** Nothing in this document
  says anything about the chemistry the actives do; it says how well a scoring
  function ranks the actives above the decoys in one small labelled set.
* **A handful of targets is still not CASF.** The enlarged set is five targets, and
  only one of them (trypsin) can be given property-matched decoys offline, so the
  *stratified* estimate rests on a single stratum.  Five targets would be a start; the
  benchmark literature this would be compared against uses dozens.
* **Co-crystallised ligands are actives by construction.** Every active in
  `demo/actives.smi` is an active because a structure contains it, which carries the
  selection bias that comes with it: these are the molecules that crystallised and
  were deposited, not a random sample of binders, and a series built from one
  structure's analogues inherits that structure's chemotype.
* **Enrichment here does not transfer.** The MDD in §8.1 is a property of *this*
  labelled set; a larger set would have a smaller one, and no number in this document
  predicts enrichment on another target, another chemotype or a real screening
  library.  Where §8 reports "no difference resolvable", it means *at this power*.

## 7. Reproducing the numbers

```bash
# the decoys (easy set), with the matching evidence
odock decoys -a actives.smi -p demo/decoys.smi -n 5 \
    -o out/decoys-easy.smi --json-out out/decoys-easy.json

# the hard band: decoys that look like the actives without being analogues
odock decoys -a actives.smi -p demo/decoys.smi -n 5 \
    --min-similarity 0.35 --max-similarity 0.6 -o out/decoys-hard.smi

# the benchmark: both shape engines, two controls, intervals, and the pre-filter
odock lbvs -a actives.smi -d out/decoys-easy.smi \
    --methods fingerprint,pharmacophore,overlay,overlay_only,crude,crude_only \
    --conformers 2 --bootstrap 200 --prefilter 0.05 --prefilter-method usr \
    --json-out out/lbvs-easy.json

odock conformers -i demo/library.smi --n-conformers 16 --json-out out/conformers.json

# the multi-target set: what can be benchmarked, and what cannot (and why)
python -c "from odock import lbvs; from odock.chem.ligand import read_ligands; \
r = lbvs.benchmark_per_target(lbvs.read_targets('demo/actives.smi'), \
read_ligands('demo/decoys.smi', embed=False), per_active=5, \
methods=('fingerprint','pharmacophore','overlay','crude'), conformers=2, bootstrap=200); \
print(r.per_target_table()); print(r.paired_table(samples=3000)); print(r.skipped)"

# ~32 s for 30 molecules on this machine (six methods, 200 resamples); the
# analytic overlay dominates the cost at ~13 ms per molecule-versus-reference pair
```

The same thing from Python:

```python
from odock import decoys, lbvs
from odock.chem.ligand import read_ligands

actives = [m for m in read_ligands("demo/library.smi", embed=False)
           if m.GetProp("_Name") in ("benzamidine", "benzamidine_methyl",
                                     "hydroxybenzamidine", "fluorobenzamidine",
                                     "chloro_benzamidine")]
pool = read_ligands("demo/decoys.smi", embed=False)

selection = decoys.match_decoys(actives, pool, per_active=5)
print(selection.quality_table())

decoys_mols = [m for m in pool if m.GetProp("_Name") in set(selection.names())]
report = lbvs.benchmark(actives, decoys_mols, conformers=2, bootstrap=200,
                        methods=("fingerprint", "pharmacophore", "overlay", "crude"),
                        decoy_quality=selection.quality())
print(report.table())
print(report.paired_table(samples=3000))     # the paired test, with the MDD
print(report.power(samples=3000))            # what this set can detect at all
print(lbvs.prefilter(list(read_ligands("demo/library.smi", embed=False)) + pool,
                     actives, keep=0.05)["workload_saved_fraction"])
# the 3-D filter instead of the 2-D one, with its measured recall and cost
print(lbvs.prefilter(list(read_ligands("demo/library.smi", embed=False)) + pool,
                     actives, keep=0.05, method="usr"))
```

## 8. Can this benchmark rank the methods at all? A power analysis

Everything above compared **independent** confidence intervals, and that is the wrong
test: the engines score the *same* molecules, so most of each interval's width is
shared molecule-to-molecule noise that cancels in a paired difference.  `odock.lbvs`
now does the paired test and, with it, reports the **minimum detectable difference**
(MDD) — the smallest gap the set could have resolved — beside the gap it observed.

### 8.1 The deficit, quantified

Paired bootstrap of AUC(overlay) − AUC(crude) on the published easy band, 3 000
resamples, same molecules, same labels:

```text
delta -0.080   SE 0.0616   MDD 0.173   95% CI [-0.229, +0.000]   resolvable: NO
```

* The paired interval is **[−0.229, +0.000]** where the two marginal intervals were
  [0.79, 1.00] and [1.00, 1.00].  The paired test is sharper, and it still does not
  exclude zero — but it comes close enough that the direction is now stated rather
  than implied.
* The **MDD is 0.173 AUC**: this set can only ever distinguish differences of about
  0.17 or more.  The observed gap is 0.080, i.e. **half** the detectable difference.
  A benchmark that could only ever detect 0.3 AUC would be worth even less; that is
  the finding, and it is why §8.3 enlarges the set.
* **The resample count is not the limiter.**  The paired SE is 0.0616 at 200
  resamples and 0.0630 at 20 000 (MDD 0.173 → 0.177): four hundred times the compute
  moves nothing.  Bootstrap resamples reduce *Monte-Carlo* noise in the interval, not
  *statistical* noise in the estimate, and only the second one matters here.

### 8.2 The enlargement, and where it stops

`demo/actives.smi` collects every **documented** active the repository can build a
target from: the five ring-amidines of `demo/library.smi` (benzamidine is the
co-crystallised ligand of 3PTB), plus the co-crystallised ligands of 3ERT and 1ERE_A
(ERα), 1HVR (HIV-1 protease), 1M17 (EGFR) and 1STP (streptavidin).  The four ligands
with a bundled structure file are written out from it; erlotinib is checked in
`tests/test_lbvs.py` against its published formula (C22H23N3O4) and the 29 heavy atoms
1M17 contains.  Ritonavir (1HXW) is **left out**: a hand-written SMILES did not
reproduce C37H48N6O5S2 and deriving bond orders from the coordinates fails on every
ligand in `tests/data`, so it is excluded rather than guessed at.

`odock.lbvs.benchmark_per_target` then matches decoys to each target's actives
separately, removing any molecule that is an active of *another* target, and reports
what it could not score:

| target | actives | matched decoys | max abs SMD | status |
|---|---|---|---|---|
| trypsin | 5 | 25 | 0.97 | benchmarked (the published set, reproduced) |
| eralpha | 2 | **0** | — | no decoy inside the actives' tolerances: the pool reaches 296 Da, OHT is 387 |
| hiv_protease | **1** | 0 | — | leave-one-out leaves no reference with one active |
| egfr | **1** | 0 | — | leave-one-out leaves no reference with one active |
| streptavidin | **1** | 0 | — | leave-one-out leaves no reference with one active |

**This is the limit of what can be concluded offline**, and it is two limits at once:
the repository has one co-crystallised ligand per target for four of five targets, and
the bundled decoy pool tops out at 296 Da so it cannot match a 387 Da ligand, let
alone the 607 Da HIV protease inhibitor.  A target needs **two** actives for a
leave-one-out benchmark to score it at all, and no amount of decoy growth fixes
`hiv_protease`, `egfr` or `streptavidin`.  The tool says so in the report
(`StratifiedReport.skipped`) instead of dropping them silently.

### 8.3 The power ladder: what more actives and more decoys actually buy

Every row below is the same actives, the same pool, the same methods and the same
controls; only the size changes.  Row 1 is the published protocol and reproduces its
numbers exactly.

| set | actives | decoys | max abs SMD | fp | pharmacophore | overlay | crude | paired MDD |
|---|---|---|---|---|---|---|---|---|
| published | 5 | 25 | 0.97 | 1.000 | 1.000 | 0.920 | 1.000 | 0.174 |
| + benzylamidine | 6 | 27 | 1.33 | 1.000 | 0.852 | 0.790 | 0.827 | 0.354 |
| 10 decoys/active | 5 | 32 | 1.17 | 1.000 | 1.000 | 0.900 | 1.000 | 0.205 |
| 20 decoys/active | 5 | 35 | 1.24 | 1.000 | 1.000 | 0.909 | 1.000 | 0.186 |
| 6 actives, 10/active | 6 | 32 | 1.47 | 1.000 | 0.849 | 0.760 | 0.833 | 0.365 |
| 6 actives, 20/active | 6 | 34 | 1.53 | 1.000 | 0.848 | 0.770 | 0.833 | 0.370 |

And the comparison the whole section exists for, in each row:

| set | delta | 95 % CI | MDD | resolvable |
|---|---|---|---|---|
| published | −0.080 | [−0.229, +0.000] | 0.173 | no |
| + benzylamidine | −0.037 | [−0.167, +0.069] | 0.165 | no |
| 10/active | −0.100 | [−0.275, +0.000] | 0.201 | no |
| 20/active | −0.091 | [−0.255, +0.000] | 0.183 | no |
| 6 actives, 10/active | −0.073 | [−0.222, +0.054] | 0.192 | no |
| 6 actives, 20/active | −0.064 | [−0.210, +0.053] | 0.181 | no |

* **More decoys buy no power.**  Going from 25 to 35 matched decoys (the pool runs
  out at 20 per active) leaves the MDD at 0.186–0.205 against 0.174 — the AUC is
  dominated by the *active* count, so four times the decoys is worth almost nothing.
  This is the cheap power gain that is not one.
* **One more active costs power rather than adding it.**  Adding benzylamidine takes
  the MDD from 0.174 to 0.354 and takes every method down with it (pharmacophore 1.000
  → 0.852, overlay 0.920 → 0.790) because the sixth amidine is neither property-matched
  by the pool (max |SMD| 0.97 → 1.33) nor ranked consistently by the methods.  More
  actives is not more power when the added active is an outlier: `demo/actives.smi`
  therefore keeps the published five and says why.
* **Resolving the observed gap would take ~23 actives** (5 · (0.173/0.080)²), at the
  same decoy count: four to five times everything the repository has, and the
  repository has 11 actives in total of which only 5 belong to a benchmarkable target.

### 8.4 The answer

**The engines are still indistinguishable, and now that is a measured statement
rather than a shrug.**  With the largest honest set that can be built offline:

* every pair of methods has been compared with a **paired, stratified** test;
* the resolved differences are **none** — 0 of 6 pairs, on every row of §8.3;
* two pairs are *tied by construction* (fingerprint and pharmacophore both rank this
  congeneric series perfectly, so their difference is exactly 0.000 with a zero-width
  interval: they are not distinguishable because there is nothing to distinguish);
* the one pair with a direction, overlay versus crude, is −0.080 ± a detectable
  difference of 0.173, i.e. the honest verdict remains "the set cannot tell them
  apart", with the sign of the gap stated.

**What would change it**: ~20-25 actives from matched targets, not more decoys, not
more resamples, and not a better score.  §8.2 shows exactly which data would do it
(two or more co-crystallised ligands per target, and a decoy pool that reaches 300-750
Da) and where to get it (a DUD-E-style set, or a CCD/published series per target,
neither of which is available offline in this repository).


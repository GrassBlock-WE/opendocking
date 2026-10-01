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
odock lbvs   -a actives.smi -d out/decoys.smi --methods fingerprint,pharmacophore,shape \
             --conformers 2 --bootstrap 200 --prefilter 0.05 --json-out out/lbvs.json
```

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
MW          136.65 +- 12.28   134.62 +- 16.51    0.13
LogP           1.12 +- 0.35      1.10 +- 0.41    0.06
HBD            2.20 +- 0.45      1.75 +- 0.64    0.74
HBA            1.20 +- 0.45      1.65 +- 0.59   -0.80
RotB           1.20 +- 0.45      1.30 +- 0.73   -0.14
charge         0.00 +- 0.00      0.00 +- 0.00    0.00
max |SMD| 0.80: NOT matched (|SMD| > 0.5)
```

**The tool says its own decoy set is not matched, and it is right.** MW, LogP,
rotatable bonds and charge are matched to a tenth of a standard deviation; HBD and
HBA are not (SMD 0.74 and −0.80), because the amidines have two to three donors
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

## 2. The benchmark, easy decoys: everything is perfect, and that is a finding

5 actives (the ring-amidines) + 25 easy decoys (Tanimoto < 0.35), all methods
leave-one-out, ETKDGv3 seed 20240101, 2 conformers, 200 bootstrap resamples:

| method | EF1 % | EF5 % | AUC | BEDROC(20) | 95 % CI (AUC / BEDROC) |
|---|---|---|---|---|---|
| fingerprint (Morgan r2) | 6.00 | 6.00 | **1.000** | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| pharmacophore fit | 6.00 | 6.00 | **1.000** | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| shape + electrostatics | 6.00 | 6.00 | **1.000** | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
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
* **All three methods are perfect, so the benchmark cannot separate them.** The
  actives are a *congeneric series* (five benzamidine analogues) and the decoys
  are deliberately dissimilar (Tanimoto < 0.35), which is the easy regime: any
  method that can see a phenylamidine wins. The honest conclusion is not "the
  fingerprint is as good as the pharmacophore"; it is that **this active set
  cannot answer that question** — and the interval [1.00, 1.00] is exact, not
  reassuring.

## 3. The hard band: where the methods separate

Selecting decoys that are topologically *closer* to the actives
(`--min-similarity 0.35 --max-similarity 0.6`) yields only five decoys — the pool
runs out, and the shortfall is reported — but the table changes:

| method | EF1 % | EF5 % | AUC | BEDROC(20) | 95 % CI (AUC / BEDROC) |
|---|---|---|---|---|---|
| fingerprint (Morgan r2) | 2.00 | 2.00 | **0.920** | 0.998 | [0.68, 1.00] / [0.87, 1.00] |
| pharmacophore fit | 2.00 | 2.00 | 1.000 | 1.000 | [1.00, 1.00] / [1.00, 1.00] |
| shape + electrostatics | 2.00 | 2.00 | **0.920** | 0.998 | [0.68, 1.00] / [0.87, 1.00] |
| *random* (control) | 2.00 | 2.00 | 0.600 | 0.982 | [0.17, 1.00] / [0.02, 1.00] |
| *property: MW* (control) | 0.00 | 0.00 | 0.200 | 0.016 | [0.00, 0.57] / [0.00, 0.98] |

The 2-D fingerprint and the shape overlay lose 0.08 of AUC to decoys that look
like the actives; the pharmacophore model — built from the other four amidines —
still separates them perfectly, because a decoy that resembles one active by
fingerprint need not present the donor/acceptor/aromatic pattern the *series*
shares. **That is the kind of statement this benchmark can make**, and the wide
interval [0.68, 1.00] is the price of saying it from five decoys.

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
the 142-molecule library (the 17 demo molecules + the 125-molecule pool):

| keep | molecules docked | actives kept | active recall | estimated docking |
|---|---|---|---|---|
| 5 % | 8 of 142 | **5 of 5** | 100 % | 436 s → **25 s** (94 % saved) |
| 10 % | 15 of 142 | 5 of 5 | 100 % | 436 s → 46 s (89 % saved) |
| 6 % (17 molecules only) | 2 of 17 | 2 of 5 | 40 % | — |

The workload figures are :func:`odock.screen.estimate_cost`'s own estimates (a
fixed per-molecule cost plus a search cost that grows with exhaustiveness and the
torsion count), not measurements of a run — the module is *imported*, never
edited, and the note in the JSON says so. The point is the shape of the trade: on
a congeneric series a 5 % cut keeps every active, because every active is one of
the queries; on a heterogeneous library the same cut is where the recall starts
to fall, and the function names the actives it threw away.

## 6. What this benchmark does not establish

* **Decoys are not inactive molecules.** A decoy is a molecule nobody has reported
  binding *that* target; it is presumed inactive, not known to be. Every decoy set
  therefore contains an unknown number of undetected actives, which *depresses*
  the measured enrichment.
* **The decoy set is not property-matched, and the tool says so**: max |SMD| 0.80
  (easy set) to 1.37 (hard set) comes from HBD/HBA, and the direction of the bias
  is not predictable — a fingerprint may reward or punish the difference. Enrichment
  factors from this repository are therefore **case-study numbers with a visible
  caveat**, not DUD-E numbers.
* **One target is a case study.** All of this is benzamidine against trypsin's S1
  pocket, in a 30-molecule set of which five are actives. Nothing here transfers to
  another target, another chemotype or a real screening library.
* **Congeneric actives make the easy regime trivial.** With five close analogues
  and dissimilar decoys every method is perfect; the hard band separates them, and
  both are reported, but a benchmark that can only be run on one series cannot
  rank methods in general.
* **The metrics are not interchangeable.** BEDROC answers "how early", EF1 %
  "what is in the very top", AUC "the whole list". They disagreed in §3 (BEDROC
  0.998 with AUC 0.920) because BEDROC's exponential weight is concentrated where
  the actives are. Quoting one without the others is how a metric becomes a
  headline.
* **The shape/electrostatic overlay is crude**: one rigid alignment, blurred
  occupancy grids at 0.5 Å, van der Waals radii from :func:`odock.sasa.radius_of`,
  Gasteiger charges, and a documented 0.5/0.5 weighting. It is not ROCS, not
  EShape, and it has no colour-force-field typing.
* **A fingerprint benchmark measures the fingerprint.** Nothing in this document
  says anything about the chemistry the actives do; it says how well a scoring
  function ranks the actives above the decoys in one small labelled set.

## 7. Reproducing the numbers

```bash
# the decoys (easy set), with the matching evidence
odock decoys -a actives.smi -p demo/decoys.smi -n 5 \
    -o out/decoys-easy.smi --json-out out/decoys-easy.json

# the hard band: decoys that look like the actives without being analogues
odock decoys -a actives.smi -p demo/decoys.smi -n 5 \
    --min-similarity 0.35 --max-similarity 0.6 -o out/decoys-hard.smi

# the benchmark: three methods, two controls, intervals, and the pre-filter trade
odock lbvs -a actives.smi -d out/decoys-easy.smi \
    --methods fingerprint,pharmacophore,shape --conformers 2 --bootstrap 200 \
    --prefilter 0.05 --json-out out/lbvs-easy.json

# ~18 s for 30 molecules on this machine; the shape overlay dominates the cost
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
                        decoy_quality=selection.quality())
print(report.table())
print(lbvs.prefilter(list(read_ligands("demo/library.smi", embed=False)) + pool,
                     actives, keep=0.05)["workload_saved_fraction"])
```

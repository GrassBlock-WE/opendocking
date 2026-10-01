# Pharmacophore models — build from evidence, screen with a shape constraint

A pharmacophore is the ligand-based counterpart of docking. Docking asks what the
receptor does with a molecule; a pharmacophore asks whether the molecule can
present the chemical features that the **known** ligands share, in the same
relative geometry. It is the method to reach for when there is no receptor
structure, when the receptor is too flexible for a rigid-protein docking run, or
when a series exists and the question is "what do these actives have in common,
and what else could do the same".

[`odock.pharmacophore`](../python/odock/pharmacophore.py) implements the whole
small-molecule workflow:

| function | what it does |
|---|---|
| `features_from_mol` | perceive donor / acceptor / aromatic / hydrophobic / positive / negative features on one 3-D structure |
| `build_model` | derive the features that **recur across a set of members**, with per-feature support, plus a shape constraint |
| `fit_score` / `screen` | place a molecule on the model (rigid alignment on its core or on a matching feature triple) and score the fit, with a documented tolerance and miss penalty |
| `enrichment` | precision@k, recall@k, enrichment factor and AUC on a labelled set — **with its sample size** |

```bash
# build a model from the benzamidine series and screen the demo library with it
odock pharmacophore build -i demo/library.smi --core 'N=C(N)c1ccccc1' -o model.json
odock pharmacophore screen -m model.json -i demo/library.smi --actives actives.smi
odock pharmacophore show -m model.json
```

**What a pharmacophore is not.** It is a *hypothesis* derived from the members it
was given. It cannot contain a feature the actives do not share, it cannot tell an
active from a decoy that presents the same features in the same places, and a high
fit score is a geometric statement — never an activity. A model built from one
molecule is an anecdote, and `build_model` says so: below
`MIN_MEANINGFUL_MEMBERS = 3` contributing members, `model.is_meaningful` is false
and the notes explain that a "recurring" feature may be one molecule's
decoration. §6 lists what the method does not establish.

## 1. Building a model from evidence

The demo library's five ring-amidines (the 3PTB ligand benzamidine and four
analogues) are a series, so they are evidence about the trypsin S1 pocket:

```bash
odock pharmacophore build -i demo/library.smi --core 'N=C(N)c1ccccc1' -o model.json
```

```text
pharmacophore model: 4 feature(s) from 5 of 5 member(s), support >= 50%
core: N=C(N)c1ccccc1
label          family              x       y       z    tol  support  fraction
aromatic#1     aromatic         1.07    0.01   -0.04   1.50        5     100%
donor#1        donor           -2.33    1.08    0.80   1.50        5     100%
donor#2        donor           -2.55   -0.84   -0.34   1.50        5     100%
hydrophobic#1  hydrophobic      1.07    0.01   -0.04   1.50        5     100%
shape constraint: envelope, 49 heavy-atom position(s) from the members,
                  1.2 Å tolerance, 60% of a candidate's atoms must be inside
note: members were superposed on the core N=C(N)c1ccccc1 (9 atoms, MCS or explicit)
note: 12 member(s) were dropped: benzylamidine: does not contain the core ...
```

Four features, **every one supported by all five members**: the aromatic ring and
its lumped hydrophobe sit on the ring centroid, and the two donors are the amidine
nitrogens (2.2 Å apart). The `support` column is the whole point of building from
a set rather than from one molecule — a feature supported by 1 of 5 would be one
analogue's decoration, and `--threshold` is what excludes it.

Two implementation choices are worth stating because they change the answer:

* **The alignment picks the *best* core match, not the first one.** A symmetric
  core has several substructure matches, and RDKit's first one can be ring-shifted,
  which leaves a para analogue's amidine on the wrong side of the ring. That was a
  real defect in this layer: with the first match, fluorobenzamidine and
  chloro_benzamidine missed *both* donor features (fit 0.00, ranks 16 and 17) while
  their analogues scored 0.94. Enumerating the matches and keeping the lowest-RMSD
  one fixes it (`_best_core_match`), and the test that pins it is
  `test_model_alignment_of_a_para_analogue_places_the_amidine_correctly`: the
  donors of a fresh conformer land within 0.7 Å of the model's, well inside the
  1.5 Å tolerance.
* **The threshold is a support fraction, not a count.** With three members
  (benzamidine, its 4-hydroxy and 4-fluoro analogues) the para *acceptor* is
  supported by 2 of 3 members (0.67) because both the phenol oxygen and the aryl
  fluorine present one: at `--threshold 0.6` it enters the model, at `0.7` it does
  not. That is the difference between "two of my three actives have an acceptor
  there" and "one does".

`--frame shared` (instead of the default `--frame align`) takes the input
coordinates as they are, which is what **docked poses** or a crystallographic
series need: they already share a frame, and superposing them again would destroy
the information that makes them comparable.

## 2. Screening, measured

```bash
odock pharmacophore screen -m model.json -i demo/library.smi --conformers 4
```

| rank | molecule | fit | matched | shape | how it was placed |
|---|---|---|---|---|---|
| 1 | benzamidine_methyl | 0.944 | 4/4 | ok | core |
| 2 | benzamidine | 0.944 | 4/4 | ok | core |
| 3 | hydroxybenzamidine | 0.938 | 4/4 | ok | core |
| 4 | chloro_benzamidine | 0.918 | 4/4 | ok | core |
| 5 | fluorobenzamidine | 0.770 | 4/4 | ok | core |
| 6 | aspirin | 0.497 | 3/4 | ok | feature triple |
| … | salicylic_acid, ibuprofen, benzylamidine | 0.50-0.43 | 3/4 | ok | feature triple |
| 14-16 | caffeine, triphenylene, benzoquinone | 0.000 | — | n/a | **no rigid alignment** |
| 17 | warfarin | 0.000 | 3/4 | **clash** | core |

**The five ring-amidines take the top five places**, at 0.918-0.944, and the best
decoy reaches 0.497 — a gap of 0.42. The mechanism is visible in the table: the
actives are placed **on the model's own core** (a rigid fit that reproduces both
donors and the ring), while the decoys only share a feature triple and are placed
by that, which leaves one feature unmatched.

Two molecules' fates are worth reading carefully, because they are the honest part:

* **warfarin matches three of the four features and is still rejected.** It put 18
  of its 23 heavy atoms more than 1.2 Å outside the members' envelope: it can
  present the amidine-like donors and the ring, but not inside the shape the
  actives occupy.
* **caffeine, triphenylene and benzoquinone are not scored at all.** They share
  neither the core nor a triple of the model's feature families, so no rigid
  alignment exists. `rejected` with that reason is reported instead of a low fit
  that would look like a measurement.

The score is documented and hand-checked in the tests:

```text
fit = (sum of matched weights - miss_penalty * n_missed) / n_features
weight of a matched feature = 1 - distance / tolerance        (linear ramp, 0..1)
```

so `fit` is in `[0, 1]`, a missed feature costs a full feature's worth at the
default `--miss-penalty 1.0`, and a molecule that is not placed scores 0 rather
than something in between.

### Cost

| step | measured |
|---|---|
| build from 5 members (embed + align + perceive + cluster) | 0.12-0.20 s |
| screen 17 molecules, 4 conformers each (68 conformers) | 1.0-2.3 s |
| the same screen at 16 conformers per molecule | 9.5 s, **identical ranking** |

The top-six ranking was unchanged between 4, 8 and 16 conformers per molecule — a
rigid fit of one conformer per molecule is often enough for a series as rigid as
this one, and the cost buys nothing here. It would not be true for a flexible
series, where the conformer that reproduces the model's geometry may not be the
first one embedded.

## 3. Enrichment, and how little it proves

```bash
odock pharmacophore screen -m model.json -i demo/library.smi --actives actives.smi
```

```text
enrichment: 5 of the 6 best molecule(s) are actives (83% precision, 83% recall),
            base rate 35.3%, enrichment factor 2.36, AUC 0.9545
  note: 6 active(s) in 17 molecule(s) is far too small a labelled set for an
        enrichment claim: the factor is quantised (one molecule moves it by tens
        of percent) and the confidence interval spans any conclusion. Report it as
        a smoke test only.
```

The labelled set is the honest one available offline: **actives** = the six
amidines of the demo library (all known trypsin S1-pocket binders; benzamidine is
the 3PTB crystal ligand), **decoys** = the other eleven drug-like molecules, which
have no known trypsin affinity. The model finds five of the six in its top six. The
sixth, benzylamidine, sits at rank 9: its amidine is one methylene away from the
ring, so it does not contain the model's core.

**That is not an enrichment result.** Six actives in seventeen molecules means one
molecule moves the enrichment factor by tens of percent, and an AUC of 0.9545 on
17 molecules has a confidence interval that spans almost everything. The function
computes these numbers because a reader needs to see them beside the sample size;
it also emits the caveat, and the CLI prints it. A real enrichment claim needs
hundreds of actives and decoys and a split that was not chosen after the fact (see
[`BENCHMARK.md`](BENCHMARK.md) for what this project considers a defensible
benchmark).

## 4. The shape constraint

A bag of features is not a pharmacophore: the features constrain where certain
groups must be, and the shape says where atoms may be at all. Two constructions
are offered:

* **`--shape envelope`** (default): every member's heavy-atom positions are stored
  (49 points for the five amidines) and a candidate must keep at least
  `--min-coverage 0.6` of its heavy atoms within `--shape-tolerance 1.2 Å` of one
  of those points. Measured on the demo library: **four molecules are rejected —
  warfarin by the envelope (18 of 23 atoms outside) and three because they cannot
  be aligned at all.** Without the constraint warfarin scores 0.31 and ranks
  among the decoys, which is the shape's contribution made visible.
* **`--shape spheres`**: exclusion spheres on the recurring scaffold atoms that
  carry **no** feature. On a small rigid series this comes out **empty** — the
  features already cover every recurring atom — and the model says so in a note
  rather than pretending it has a shape constraint. It is the construction to use
  when the members have a large featureless scaffold.

Neither is the receptor's excluded volume, and the difference matters: a
ligand-derived envelope cannot reject a molecule that clashes with the *protein*,
only one that sticks out of the actives' own shape. A structure-based screen uses
the receptor for that.

## 5. Features, in one table

| family | perceived from | position |
|---|---|---|
| `donor` | RDKit's `BaseFeatures.fdef` (or a curated SMARTS fallback if the build has no fdef) | the heteroatom |
| `acceptor` | as above | the heteroatom |
| `aromatic` | ring perception | the ring centroid |
| `hydrophobic` | the lumped ring hydrophobe, with the per-atom ones it covers dropped | the ring centroid |
| `positive` / `negative` | the fdef's ionisable groups | the charged centre |

`ZnBinder` is dropped by default: a zinc-binding site is a property of the
receptor, not of the ligand, so it cannot be part of a ligand-derived model.

## 6. What this does not establish

* **A pharmacophore is a hypothesis, not a discovery.** It contains exactly the
  features the members present. If the actives bind through an interaction the
  feature set does not model (a halogen bond, a metal, a covalent warhead, a
  water-mediated bridge), no model built from them will contain it.
* **A high fit is not activity.** The score is geometric: the features are in the
  right places. Affinity depends on the rest of the molecule, the protein's
  flexibility, the protonation states, the solvation — none of which a
  ligand-derived model sees. A fit of 0.94 and a fit of 0.77 are both "this
  molecule can present the pattern"; the ranking between them is a hypothesis.
* **A congeneric model screens for that series.** The model above was built from
  five amidines superposed on their shared core, so it can only accept molecules
  that contain that core (or a feature triple of it). It found the six amidines
  and nothing else — a useful *series* filter, useless for scaffold hopping.
* **Rigid fitting is a real limitation.** Each library molecule is placed by one
  rigid transform (onto the core, or onto a matched feature triple) and scored as
  it is: no torsion sampling, no flexible superposition. A flexible analogue whose
  best conformation is not among the embedded ones scores low for that reason
  alone. The measured consequence here is small (4, 8 and 16 conformers give the
  same top six) but it is a property of *this* rigid series, not of the method.
* **The number of members bounds the claim.** Below three contributing members the
  model is flagged as an anecdote. Three to ten members buy "these features recur";
  they do not buy a validated model. A published pharmacophore model is built from
  tens of actives and tested on an external set.
* **The enrichment numbers are illustrative.** Six actives and eleven decoys, all
  chosen by hand into one small library, cannot establish enrichment. The AUC and
  the enrichment factor are reported with their sample size for that reason.
* **Nothing here validates a shape.** The envelope is a point cloud from one
  conformer per member, not a surface, and the `1.2 Å` slack is a convention, not a
  measured property of the actives.

## 7. Reproducing the numbers

```bash
LIB=demo/library.smi
CORE='N=C(N)c1ccccc1'

# the model (5 of 17 members contain that core; the other 12 are reported)
odock pharmacophore build -i $LIB --core "$CORE" -o out/pharm-model.json
odock pharmacophore show  -m out/pharm-model.json

# the screen: + the labelled set, the CSV and the JSON
odock pharmacophore screen -m out/pharm-model.json -i $LIB \
    --conformers 4 \
    --actives out/pharm-actives.smi \
    --scores-out out/pharm-scores.csv --json-out out/pharm-screen.json

# the same numbers from Python
python - <<'PY'
from odock import pharmacophore as P
from odock.chem.ligand import read_ligands

library = read_ligands("demo/library.smi", embed=False)
amidines = [m for m in library
            if m.GetProp("_Name") in ("benzamidine", "benzamidine_methyl",
                                      "hydroxybenzamidine", "fluorobenzamidine",
                                      "chloro_benzamidine")]
model = P.build_model(amidines, frame="align", core="N=C(N)c1ccccc1")
print(model.table())
hits = P.screen(model, library, conformers=4, seed=20240101)
print(hits.table())
print(P.enrichment(hits, {"benzamidine", "benzamidine_methyl", "benzylamidine",
                          "hydroxybenzamidine", "fluorobenzamidine", "chloro_benzamidine"}))
PY
```

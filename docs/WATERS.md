# Water networks, conserved waters, and the ones a ligand displaces

Every docking pipeline throws the crystallographic waters away first, and the
first question a medicinal chemist asks about a hit is which of them it displaces.
This document is the measured answer for the bundled systems: the water network of
each conformation, which water sites survive across an ensemble, which are
displaced by a ligand (and by which atom), and whether a per-pose displacement
count says anything about affinity at the sample size this repository has.

```bash
odock ensemble waters -r 3PTB.pdb 2PTN.pdb --box-ligand BEN --json-out waters.json
python examples/ensemble_validation.py --waters-only     # every number below
python -m pytest tests/test_waters.py -q                 # the same numbers as tests
```

## 1. The network

`water_network` builds the hydrogen-bond graph of one conformation: waters are
nodes, an edge is an O···O distance within `--hbond` (3.5 Å, the standard
crystallographic criterion), and a water–protein contact is an N or O of the
protein within the same distance. "Hydrogen bond" here means *the heavy atoms are
close enough for one*: crystallographic waters carry no hydrogens, and pretending
to know their orientation would be invention.

Measured on the bundled trypsins:

| | 3PTB (holo, benzamidine) | 2PTN (ligand-free) |
|---|---|---|
| waters | **62** | **82** |
| water–water H-bonds | **20** | **35** |
| connected components | **42** | **48** |
| largest component | **6** | **7** |
| branching waters (≥ 2 neighbours) | **7** | **12** |
| water–protein contacts | 141 | 174 |

Both structures are mostly *isolated* waters (42 and 48 components for 62 and 82
waters): the hydrogen-bonded network is sparse and made of small clusters, which
is what a 1.5–1.8 Å trypsin structure looks like. The ligand-free structure has
more waters, more bonds and a larger biggest cluster, because the site that
benzamidine occupies in 3PTB is water-filled in 2PTN — which is the next section.

## 2. Conserved, moved, displaced, transient

`compare_water_sites` pools the waters of every conformation and clusters them
into **sites**: a site is a leader cluster of water oxygens within
`--match-radius` (1.5 Å), seeded in conformation order so the result is
deterministic. Each site then gets a verdict from two numbers — how many
conformations hold a water there (occupancy) and how far the matched positions
spread (displacement):

| verdict | criterion | meaning |
|---|---|---|
| **conserved** | occupancy ≥ 80 %, displacement ≤ 1.0 Å | the water is a property of the site |
| **displaced** | a non-water residue sits within `--occlusion-radius` (2.5 Å) of an *empty* site | something else took its place |
| **moved** | occupancy ≥ 80 %, displacement > 1.0 Å | present throughout, but travelling with the structure |
| **transient** | present in some conformations and not others, nothing to blame | the structures disagree about it |
| **absent** | present in fewer than a fifth | a one-off |

Two details of the ordering matter and are deliberate:

* **displaced is checked before moved.** With two conformations a water present in
  one of them has an occupancy of exactly 50 %, and calling that "moved" would
  claim the water travelled when in fact it is not there at all in the other
  structure.
* **every non-protein residue is a candidate displacer**, ions and buffer
  components included, and the report *names* what each one is. An ion occupies
  space the same way a ligand does; a reader can tell a drug from a sulfate from
  the name.

## 3. Measured: benzamidine against the ligand-free structure

This is the one pair in `tests/data` that differs in **both** water content and
ligand content (3PTB carries benzamidine, 2PTN does not), so it is the offline
check the mandate asks for: *is the classifier able to say that a ligand displaced
a water that experiment shows it displacing?*

89 sites across the pair: **55 conserved, 32 transient, 2 displaced, 0 moved,
0 absent**.

The two displaced sites, with their protein contacts and the ligand atoms standing
in them:

| site | position (Å) | the water hydrogen-bonded to | occupied by |
|---|---|---|---|
| 63 | (−2.25, 14.49, 12.49) | **ASP189 A, GLY219 A, LYS224 A** | **BEN1 A:N1** (the amidine nitrogen) |
| 86 | (−2.15, 13.20, 15.31) | **ASP189 A, SER190 A** | BEN1 A:C1, BEN1 A:C, BEN1 A:N1, BEN1 A:N2 |

That is chemically the right answer, not a coincidence of geometry: **ASP189 is
the S1 specificity residue** — the aspartate that binds the amidine of
benzamidine (and the side chains of Lys/Arg substrates) — and LYS224 and GLY219
line the same pocket. The analysis says: *benzamidine's amidine nitrogen takes the
place of a water that was hydrogen bonded to Asp189*. Both sites are present in
the ligand-free 2PTN structure and empty in the holo 3PTB one, which is what
"displaced" means here, and both are asserted in `tests/test_waters.py`.

The pessimistic reading is also worth stating: **two waters out of 89 sites**, for
a ligand of nine heavy atoms occupying the site. The classifier finds a real
displacement; it does not find that the site is water-mediated.

## 4. Does displacing conserved waters track affinity? (measured, underpowered)

The question a screening campaign wants answered is whether the count of conserved
waters a pose displaces correlates with its affinity. Measured on the demo library
(17 molecules, benzamidine excluded because it *is* the ligand of 3PTB and would
be scoring the answer) docked into 3PTB with the ligand stripped, counting the 55
conserved sites within 3.5 Å of any ligand heavy atom:

| ligand | affinity | displaced (best pose) | mean over poses |
|---|---|---|---|
| warfarin | −7.332 | 3 | 3.00 |
| hydroxybenzamidine | −6.517 | 3 | 3.00 |
| triphenylene | −6.278 | 2 | 2.00 |
| salicylic acid | −6.247 | 1 | 1.00 |
| fluorobenzamidine | −6.226 | 3 | 3.00 |
| chloro_benzamidine | −6.175 | 1 | 1.00 |
| benzamidine | −6.022 | 2 | 2.00 |
| paracetamol | −6.002 | 3 | 3.00 |
| nicotinamide | −5.812 | 1 | 1.00 |
| ibuprofen | −5.764 | 3 | 3.00 |
| aspirin | −5.694 | 2 | 2.00 |
| isonicotinic acid | −5.605 | 1 | 1.00 |
| benzylamidine | −5.495 | 2 | 2.00 |
| benzamidine_methyl | −5.462 | 1 | 1.00 |
| caffeine | −5.338 | 3 | 2.50 |
| acetanilide | −4.970 | 3 | 3.00 |
| benzoquinone | −4.858 | 1 | 1.00 |

**Spearman rho = −0.191, 95 % bootstrap CI [−0.668, +0.348], n = 17** for the best
pose, and **rho = −0.250, CI [−0.701, +0.292], n = 17** for the mean over poses
(2000 resamples, seeded). The sign is the intuitive one — more displaced conserved
waters, slightly better affinity — and **the interval spans zero**: at this n the
relationship is not resolvable. The module refuses to present the point estimate
on its own: below five pairs it returns `not resolvable` instead of a number, and
at n = 17 it returns the interval next to the rho so the reader can see that the
data cannot separate the hypothesis from noise.

What would make it resolvable: more ligands than 17, more than one receptor, and
poses that actually vary in how many conserved waters they touch. In this set
almost every pose displaces 1–3 of the 55 conserved waters, so the predictor has
very little variance to correlate with — which is itself the finding: **on this
system, "displaces a conserved water" is nearly constant across ligands**, and a
near-constant predictor cannot rank anything.

## 5. What this does not establish

* **Crystallographic waters only.** Crystal waters are the ones the experiment
  resolved; a site that is water-filled in solution and disordered in the crystal
  is invisible here. Nothing in this module places or predicts a water.
* **"Hydrogen bond" is a distance criterion.** Crystallographic waters have no
  hydrogens, so an O···O contact at 3.4 Å is counted whether or not the
  orientation could support a bond.
* **Occupancy is a statement about the input set.** With two crystal structures,
  50 % occupancy means "one of two"; with NMR models or MD snapshots the same
  number means something quite different. The report always prints the
  denominator.
* **"Displaced" is geometric.** A residue near an empty site is treated as the
  cause; correlation is not causation, and a site can be empty because it is
  disordered rather than because something took its place.
* **The displacement count is not an energy.** It counts conserved sites a pose
  comes within 3.5 Å of. A conserved water that a pose merely brushes costs
  nothing in the Vina scoring used here, and the *cost* of displacing one is not
  modelled at all — the score is a hypothesis generator, not a free-energy term.
* **The docks were run without waters.** The per-pose count is a geometric check
  against water positions measured from the crystal structures, not a docking run
  in which the waters competed with the ligand.
* **The correlation is underpowered** at n = 17 with an interval spanning zero,
  and the predictor is nearly constant on this system. Treat section 4 as a
  demonstration of the method and of the reporting discipline, not as a result
  about water displacement and affinity.
* **Two conformations cannot test conservation.** The pair above is two
  structures of the same protein; "conserved" across two is a weak statement, and
  the 32 transient sites are exactly what two snapshots produce.

## 6. Reproducing every number

```bash
python examples/ensemble_validation.py --waters-only      # §1, §3, §4
odock ensemble waters -r 3PTB.pdb 2PTN.pdb --box-ligand BEN \
    --json-out waters.json --outdir out/waters
python -m pytest tests/test_waters.py -q                  # §1, §2, §3 and the power rule
```

## 7. API

```python
from odock import ensemble, waters

conformations = ensemble.read_conformations(["3PTB.pdb", "2PTN.pdb"], keep_water=True)
box = ...                                     # any box on the site
ensemble.align_conformations(conformations, box=box, superpose=True)

network = waters.water_network(conformations[0])
print(network.n_waters, network.n_edges, network.n_components, network.largest_component)

analysis = waters.compare_water_sites(conformations)
print(analysis.text())
print(analysis.summary()["counts"])           # conserved / moved / displaced / transient
for site in analysis.displaced:
    print(site.center, site.contacts, site.displaced_by)

counts = waters.pose_displacement(result.poses, analysis.conserved, radius=3.5)
print(waters.correlate([c["displaced"] for c in counts],
                       [c["affinity"] for c in counts]))   # rho, n, CI or "not resolvable"
```

`WaterAnalysis` exposes `labels`, `networks`, `sites`, `by_class()`, `conserved`,
`displaced`, `summary()`, `table()`, `text()` and `as_dict()`; `WaterSite` exposes
`occupancy`, `displacement`, `classification`, `contacts`, `displaced_by` and the
per-conformation `observations`; `WaterNetwork` exposes `n_waters`, `n_edges`,
`n_components`, `largest_component`, `bridges()` and `protein_contacts`.

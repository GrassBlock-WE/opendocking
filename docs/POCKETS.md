# Cryptic and transient pockets across an ensemble

A rigid structure answers "where can a ligand bind *here*". The reason people
look at ensembles is the different question: **which cavities exist in some
conformations and not others** — the site that is closed in the structure you
happen to have and open in one you do not.

`odock ensemble pockets` answers it with the machinery already in the project:
the pocket detector of [`odock.pocket`](../python/odock/pocket.py), the common
binding-site frame of [`odock.ensemble`](../python/odock/ensemble.py), and a
series of measurements defined in [`odock.pockets`](../python/odock/pockets.py)
that separate *a cavity opening* from *the detector changing its mind*. The last
part is the whole difficulty, and most of this document is about it.

```bash
odock ensemble pockets \
    -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT --region-radius 14 \
    --json-out pockets.json --pockets-pdb pockets.pdb
```

## 1. The pipeline

1. **Detect** (`pockets.detect_pockets`): `find_pockets` on each conformation
   with the co-crystallised ligand stripped — a holo structure's site is not a
   cavity while its inhibitor is sitting in it — and every pocket reported in the
   ensemble's common frame. Only heavy atoms are rasterised.
2. **Describe** every pocket with four things:
   * the detector's own **volume** and score;
   * its **lining residues**: residues with a heavy atom within `--lining-radius`
     (5 Å) of any cavity *point*, mapped onto the reference structure's residue
     numbering through the sequence alignment, so "the same residue" is a fact
     and not a coincidence of numbering;
   * a **buriedness** computed here from the receptor atoms: the fraction of the
     26 cubic directions from the cavity's own points in which a receptor atom
     lies within `vdw + probe` of the ray, within 5 Å;
   * the **local free volume** in a sphere of `--local-radius` (6 Å) around the
     cavity's *deepest point* — measured from the same probe rule the detector
     uses, and defined **even where no pocket was found**. This is what makes
     "closed here, open there" a continuous measurement instead of a
     found/not-found flag. The anchor is the point of maximum clearance, not the
     centroid: a crescent-shaped cavity can have its centroid inside an atom.
3. **Match** across conformations (`pockets.match_pockets`): two pockets are the
   same cavity when their centroids are within `--match-radius` (4 Å) **and**
   their lining residues overlap by at least `--min-overlap` (0.5 Jaccard). Both
   are needed — centroids alone confuse the two ends of a channel, residues alone
   confuse two cavities cut out of the same wall. The reference's pockets seed
   the tracks; the assignment is greedy, and `lining_jaccard` plus
   `centroid_spread` are printed so a wrong assignment is visible.
4. **Measure the detector's own noise** (`pockets.detector_noise`): the same
   structures re-detected under five jittered settings — probe 1.3/1.4/1.5 Å and
   grid spacing 0.9/1.0/1.1 Å. Anything below that floor is the method's
   resolution, not the protein's motion.
5. **Trace** every track's local free volume through *every* conformation,
   including the ones that reveal nothing at that point, and rank the tracks by
   `presence × absence × change` (see §4).

## 2. The measurement that decides everything: the detector is noisy

The same structure, unchanged, re-detected at five settings:

| structure | volume change of a tracked pocket | presence |
|---|---|---|
| 3ERT (ERα antagonist) | **±103 Å³ (35 % median)** | 7–11 pockets in the region |
| 1ERE_A (ERα agonist) | **±89 Å³ (57 % median)** | 3–6 pockets in the region |
| 3PTB (trypsin) | ±51 Å³ (18 % median) | 1–2 pockets in the region |
| 2PTN (trypsin) | ±136 Å³ (41 % median) | 1–2 pockets in the region |

The detector's *absolute volume* is therefore **not a quantitative read-out**:
sub-pocket partitioning cuts a cavity differently at a different grid spacing, so
the same cavity's volume moves by tens of percent when nothing about the protein
changes. The numbers move again when `--max-pockets` or the region changes
(a 12-pocket cap on the same pair gives ±159 Å³ on 3ERT), which is why every
number in this document is quoted with its configuration.

What *is* readable is:

* **presence**, checked for reproducibility (`stability`: the fraction of the
  jittered settings that reproduce the track where it was found, which must be
  ≥ 0.6);
* the **local free volume** at a fixed point, whose resolution is measured the
  same way by jittering the probe at that anchor (typically 19–71 Å³ here);
* the **lining residues**, which are chemical identities, not detector artefacts.

The report prints all of this, and labels the volume as context rather than
evidence.

## 3. Verdicts, and the two ways an "absence" can be fake

A cavity that the detector does not report in some conformation can mean three
different things, and this module distinguishes them by comparing the free
volume where it is *absent* with the free volume where it is *present*
(`closure_fraction`):

| `closure_fraction` | verdict | meaning |
|---|---|---|
| `< 0.2` (`closed_elsewhere`) | **closure** | the space is filled — a cryptic/transient site |
| `0.2 … 0.8` (`narrowed`) | **narrowed** | the site is still open, but occluded — a real change, not a cryptic site |
| `≥ 0.8` (`presence_artefact`) | **detector disagreement** | the free volume is still there; the detector simply stopped calling it a pocket |

Measured on the ERα pair, this is not a hypothetical distinction: one track's
free volume is **656 Å³ in the structure that does not reveal it** against
**565 Å³ in the one that does** (86 % remains — a detector disagreement), and
another keeps **51 %** of its free volume (a genuine narrowing). A rule that only
asked "was a pocket found" would have reported both as cryptic sites.

A track is **confident** when all three hold:

1. `stability ≥ 0.6` — its presence, where it is found, survives the jitter;
2. `changing` — its free volume moves by more than **2 ×** the measured local
   resolution (`CHANGE_MARGIN`), a stated convention;
3. the cavity is either comfortably above the detector's cut-off
   (`volume_min > 2 × --min-volume`) **or** demonstrably closed where it is
   absent (`closed_elsewhere`). The second clause is what rescues a genuine
   cryptic site whose volume is marginal: its volume is at the threshold, its
   *collapse* is not.

and **cryptic** when it is confident and either transient (found in some
conformations but not all) or absent from the reference structure. The ranking
heuristic is `cryptic_score = presence × absence × change` with
`presence = 0.5 + 0.5·support`, `absence = 2` when the reference does not reveal
it else `1`, and `change = local_free_change / local_noise` capped at 20 — every
factor is printed, so the number can be recomputed by hand.

## 4. Measured: the estrogen receptor, antagonist against agonist

3ERT (4-hydroxytamoxifen, helix 12 displaced) and 1ERE chain A (estradiol, helix
12 packed over the pocket): the same pair whose ensemble docking appears in
[`docs/ENSEMBLE.md`](ENSEMBLE.md). Region 14 Å around the 4-OHT box, `--min-volume
50`, `--max-pockets 12`.

| track | found in | volume (Å³) | free volume at the anchor (Å³) | resolution | ratio | verdict |
|---|---|---|---|---|---|---|
| 1 | 3ERT only | 474 | **410 → 4** | 71 | **5.7×** | closure → **cryptic** |
| 2 | 3ERT only | 144 | **352 → 22** | 61 | **5.4×** | closure → **cryptic** |
| 4 | 3ERT only | 55 | **366 → 4** | 64 | **5.7×** | closure, volume marginal → **cryptic** |
| 5 | 3ERT only | 160 | 630 → 323 | 46 | 6.7× | **narrowed** (51 % remains) → not cryptic |
| 6 | 1ERE_A only | 131 | 656 → 565 | 19 | 4.8× | **detector disagreement** (86 %) → not cryptic |
| 3 | both | 59–150 | 158 → 167 | 41 | 0.2× | unchanged → not cryptic |

With the stability check on, **three** cavities are reported: tracks 1, 2 and 4,
all present in 3ERT and gone in 1ERE_A, with stabilities 1.00, 0.80 and 0.60.
The three that close are lined by

* **THR347, ASP351, GLU380, TRP383, GLU419, ASN519, MET522, GLU523** (track 1,
  the largest: 474 Å³, 410 Å³ of free volume closed to 4 Å³);
* **MET343, GLY344, LEU345, LEU346, THR347, ASN348, ALA350, ASP351** (track 2 —
  the estradiol pocket's own core);
* **THR347, ALA350, ASP351, LEU354, TRP383, LEU387, LEU525, LEU536** (track 4 —
  including **LEU525**, the helix-12 residue that moves 2.01 Å in the
  superposition).

Every one of them is lined by the **same residues** in both structures
(`lining_jaccard = 1.00`): the structures show the same residues *moved*, not two
different pockets that happen to overlap. This is the textbook behaviour of the
ERα ligand-binding domain — with the antagonist bound, helix 12 is displaced and
the surface it normally covers is a set of open cavities; with the agonist bound,
helix 12 packs over them and they are filled.

Reversing the frame (`--reference 1`, box from the estradiol in 1ERE_A) flips the
semantics: "absent from the reference" then describes a cavity of 3ERT that
1ERE_A does not reveal — measured, the biggest such signal is a point with **9 Å³
of free volume in 1ERE_A against 360 Å³ in 3ERT** (a 351 Å³ opening, 175× the
local resolution of 2 Å³) whose *detector volume* is only 50 Å³, exactly at
`--min-volume`. It is the free-volume collapse, not the volume, that identifies
it. With the stability measure on, that track scores 0.40 and is **not** reported
— its presence is not reproduced under the jittered settings, and the module
declines to call it. Both facts are in the output.

## 5. Measured: the control, and why it matters

3PTB and 2PTN share their binding site to **0.14 Å** CA RMSD (section 1 of
[`docs/ENSEMBLE.md`](ENSEMBLE.md)). A cryptic site here would be a statement
about the detector, not the protein. Region 12 Å, `--max-pockets 10`:

| track | found in | volume (Å³) | free volume 3PTB / 2PTN (Å³) | resolution | ratio | verdict |
|---|---|---|---|---|---|---|
| 3 | 2PTN only | 236 | 519 / 554 | 43 | 0.8× | disagreement (107 %) → rejected |
| 1 | 3PTB only | 166 | 509 / 549 | 39 | **1.03×** | disagreement (108 %) → rejected |
| 2 | 3PTB only | 151 | 642 / 629 | 31 | 0.4× | disagreement (98 %) → rejected |

**Zero cryptic candidates**, and the reason is quantitative rather than
asserted: at the same point in the two structures the free volume differs by
7.6 %, and the largest apparent change (40 Å³) is 1.03× the detector's own
resolution of 39 Å³ — below the 2× margin. The ERα closures sit 5–11× above
theirs. That is the difference between "we found a cryptic site" and "we found
noise", measured on the same pipeline.

The raw detector output *does* differ between the two trypsins — one cavity is
flagged in each structure and not the other — and the report says so, with the
numbers that reject it. A control that passed because the detector happened to
agree would have proved nothing.

## 6. Measured: synthetic cavities, where the arithmetic is exact

| quantity | value |
|---|---|
| 300 carbon atoms on a sphere of radius 8 Å: free volume in a 6 Å sphere | **485 Å³** (analytic ≈ 493 Å³) |
| … with the interior filled by a 3-D lattice (27 atoms, 2.5 Å spacing) | **0 Å³** |
| buriedness at the centre, ray 10 Å | **1.000** (all 26 directions hit the shell) |
| buriedness at the centre, ray 5 Å | **0.000** (the shell is out of reach) |
| detector output on the empty shell | 5 sub-pockets, \|centre\| ≈ 3.0 Å, **79–86 Å³** each, buriedness 0.945–0.950 |
| sum of those volumes | **410 Å³**, within 15 % of the independent 485 Å³ |

The last row is a cross-check between two independent pieces of code — the
detector's grid count and this module's free-volume measurement — and the whole
row is exact arithmetic, which is why `tests/test_pockets.py` can pin it.

## 7. What this does not establish

* **Two crystal structures cannot show that a cavity is druggable.** They cannot
  show that a ligand binds there, that the cavity is populated in solution, or
  that it is reachable. They show that two experimental snapshots differ in
  where their free space is, and nothing more.
* **No thermodynamics.** Every structure is one vote; there is no Boltzmann
  weighting, no population, no free-energy estimate. An NMR ensemble or an MD
  trajectory would be a better input than two crystal forms, but even then this
  module would only measure geometry.
* **Absence is not closure.** That is why §3 exists. The `narrowed` and
  `presence_artefact` verdicts are computed, printed and excluded from the
  cryptic list, and the ERα pair produced both.
* **The detector's absolute volume is not reproducible** (±18–57 % on an
  unchanged structure, and it moves with `--max-pockets` and the region). Any
  argument that rests on "the pocket grew from 150 to 200 Å³" is resting on the
  method's noise unless it is compared against the measured floor.
* **A "pocket" is a sub-pocket.** The detector's `pocket_radius` (6 Å) cuts a
  connected cavity into pieces, so a large site appears as several tracks. The
  synthetic 9.8 Å cavity becomes five tracks; their sum is meaningful, an
  individual one is a fragment.
* **The lining comparison needs the sequence alignment.** It is used whenever the
  conformations come through `align_conformations`; given bare
  `Conformation` objects the module warns that it is comparing each structure's
  own numbering, and `lining_jaccard` must be read with that in mind.
* **What "absent from the reference" means depends on the reference.** Anchoring
  the frame on 3ERT reports the cavities that *close* when the agonist binds;
  anchoring on 1ERE_A reports the ones that *open*. Both are measured above, and
  they are different lists.
* **A ligand-free structure cannot define the box at all.** `--reference 1` with
  2PTN and `--box-ligand BEN` fails with *"2PTN has no residue named 'BEN'; its
  non-standard residues are CA"*, which is the honest answer: derive the box from
  the ligand-bearing sibling (as
  [`examples/ensemble_validation.py`](../examples/ensemble_validation.py) does,
  with a note) and accept that the box is then in a frame that differs from the
  reference's by the measured site RMSD.
* **Two conformations only.** "Found in 1 of 2" is 50 % support by construction;
  this pair cannot distinguish a rare state from a common one. The support
  fraction is a description of the input set, not a population.
* **The region matters.** Without `--region-radius` the report is about the
  receptor's whole void network, including internal cavities and crystal-contact
  grooves that have nothing to do with the site of interest.
* **The HIV-1 protease pair is not measured here for pockets.** `1HVR`/`1HXW`
  is aligned and docked in [`docs/ENSEMBLE.md`](ENSEMBLE.md), but the pocket
  analysis has not been run on it yet; the command in the header reproduces it in
  about a minute if you want the numbers.

## 8. Reproducing every number

```bash
python examples/ensemble_validation.py --pockets-only          # §4, §5, §6
odock ensemble pockets -r 3ERT.pdb 1ERE_A.pdb --box-ligand OHT \
    --region-radius 14 --json-out pockets.json --pockets-pdb pockets.pdb
odock ensemble pockets -r 3PTB.pdb 2PTN.pdb --box-ligand BEN \
    --region-radius 12 --max-pockets 10
python -m pytest tests/test_pockets.py -q                       # the same numbers
```

Operational note: redirect the report to a file rather than piping it into a
filter that stops reading (`odock ... | Select-Object -First 20` on Windows kills
the Python process when the pipe closes — that is what lost the first control
run). `odock.cli.main` does catch `BrokenPipeError`, but a filter that *kills*
its upstream is not a closed pipe.

## 9. API

```python
from odock import ensemble, pockets
from odock.prepare import box_from_points

conformations = ensemble.read_conformations(["3ERT.pdb", "1ERE_A.pdb"])
box = box_from_points(ensemble.ligand_coords(conformations[0], "OHT"), buffer=6.0)
aligned = ensemble.align_conformations(conformations, box=box, site_radius=8.0)

comparison = pockets.compute_comparison(
    aligned, box=box, region_radius=14.0, min_volume=50.0, max_pockets=12,
)
print(comparison.text())                     # the report, with its own noise floor
for track in comparison.cryptic():           # ranked transient/cryptic cavities
    print(track.as_dict())                   # every measured component
    print(track.openness_trace())            # per conformation: volume + free volume
open("pockets.pdb", "w").write(comparison.pocket_pdb())   # for the workbench

# one conformation, no comparison:
atoms = pockets.pocket_atoms(aligned.conformations[0], box=box)
observations = pockets.detect_pockets(aligned.conformations[0], box=box, atoms=atoms)
print(pockets.local_free_volume((30.0, -2.0, 24.0), atoms))   # Å³ of free space
print(pockets.buriedness(observations[0].points, atoms))      # 0..1, 26 directions
print(pockets.clearest_point(observations[0].points, atoms))  # the anchor it uses
```

`PocketComparison` exposes `tracks`, `detections`, `noise`, `parameters`,
`ranked()`, `cryptic()`, `confident()`, `table()`, `cryptic_table()`, `text()`,
`as_dict()` and `pocket_pdb()`; `PocketTrack` exposes `support`, `transient`,
`absent_from_reference`, `volume_min/max/change/ratio`, `local_free`,
`local_free_change`, `local_noise`, `closure_fraction`, `closed_elsewhere`,
`narrowed`, `presence_artefact`, `changing`, `stability`, `confident`, `cryptic`,
`cryptic_score`, `lining_reference()`, `lining_union()`, `lining_jaccard()` and
`openness_trace()`.

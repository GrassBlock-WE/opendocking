# Conformer ensembles and the shape/electrostatic overlay

Every 3-D ligand-based answer this project gives — the pharmacophore fit, the shape
overlay, the 3-D similarity search — is `max(score over the geometries that were
generated)`, and nothing else.  This document is about those geometries and about
the overlay that reads them: what is generated, what it costs, what the score
means, and where it stops being worth anything.

Two modules, both measured here rather than asserted:

* :mod:`odock.conformers` — ETKDGv3 ensembles with RMSD, energy and torsion-coverage
  reporting, and the 12 USR descriptors a pre-filter can afford;
* :mod:`odock.overlay` — analytic Gaussian shape and Gasteiger electrostatic
  fields, compared at the best pose over the best conformer over the best
  reference, with the two terms always reported separately.

```bash
odock conformers -i demo/libraries/library.smi                       # ensemble quality + accounting
odock conformers -i demo/libraries/library.smi --json-out out/conformers.json
odock conformers -i demo/libraries/library.smi --fixed-attempts      # the old fixed 16-attempt rule
odock conformers -i poses.sdf --keep-input                 # measure the poses as given
odock lbvs -a actives.smi -d decoys.smi --methods overlay,crude --conformers 2 \
           --prefilter 0.05 --prefilter-method usr --json-out out/lbvs.json
```

## 1. The ensemble: a sample, measured, and where it comes from

`build_ensemble` embeds with ETKDGv3 at a fixed seed, one thread, `pruneRmsThresh`
set to the RMSD cut; minimises with MMFF94 (MMFF94s and UFF as fallbacks); applies an
energy window; prunes greedily by heavy-atom RMSD; and reports what survived.

The number of **embedding attempts** is the one thing the caller usually leaves
alone, and the accounting below says it should not be: the default is
`min(128, max(16, 8 · (1 + rotatable bonds)))`, so a rigid ring gets 16 attempts and
a nine-rotor chain gets 80, while an explicit `n_conformers` is honoured exactly so a
published protocol still reproduces.  `rotor_scaled_attempts` is a pure function of
the molecule — no time, no memory, no seed — and a methyl rotor does not count,
because `odock.chem.ligand.rotatable_bonds` excludes methyls, amides, ring bonds and
symmetric ends.

### 1.1 Where the conformers go

`odock conformers -i demo/libraries/library.smi` (17 molecules, rotor-scaled default):

```text
conformers: 17 of 17 molecule(s) embedded (100%), 2.82 conformer(s) kept on average
  (median 1, min 1) from 20.7 attempt(s) per molecule, 0.183 s per molecule
  accounting: attempts -> embedded -> (-energy, -RMSD) -> kept: 352 -> 56 -> (-3, -5) -> 48
    (dominant loss: embedding)
  mean torsion coverage 30% of its 33% ceiling (97% of what the conformer count allows)
```

**The dominant loss is the embedding stage — and that is the generator working, not
failing.**  ETKDGv3 refuses to return two conformers closer than the pruning
threshold, so sixteen attempts on benzamidine return *one* geometry: the molecule has
one shape.  The RMSD pruning (5 conformers over the library) and the energy window
(3) remove almost nothing.  On a varied-flexibility set the same accounting is
starker:

| molecule | rotors | attempts | embedded | −energy | −RMSD | kept | coverage | ceiling |
|---|---|---|---|---|---|---|---|---|
| benzene | 0 | 16 | 1 | 0 | 0 | 1 | — | — |
| benzamidine | 1 | 16 | 1 | 0 | 0 | 1 | 17 % | 17 % |
| paracetamol | 1 | 16 | 2 | 0 | 1 | 1 | 17 % | 17 % |
| warfarin | 4 | 40 | 25 | 3 | 2 | **20** | **100 %** | 100 % |
| ibuprofen | 4 | 40 | 9 | 0 | 0 | 9 | 59 % | 100 % |
| decane | 7 | 64 | 61 | 0 | 1 | **60** | **100 %** | 100 % |
| octanediol | 7 | 64 | 57 | 0 | 0 | **57** | **100 %** | 100 % |
| peg3 | 9 | 80 | 76 | 0 | 0 | **76** | **90 %** | 100 % |

**Coverage is capped by the conformer count**, which is why the report prints the
ceiling beside it: `k` conformers can occupy at most `k` distinct bins on a bond, and
a bond has six, so the geometric mean over the rotors cannot exceed
`min(1, kept / 6)`.  17 % coverage from one conformer is *at* its ceiling; the same
17 % from six conformers would not be, and without the ceiling the two are
indistinguishable.

### 1.2 The fix, and what it bought

Before → after, on the varied-flexibility set (the "before" is the old fixed 16
attempts, i.e. `--fixed-attempts`):

| measure | before | after |
|---|---|---|
| attempts per molecule | 16.0 | 42.0 |
| conformers kept per molecule | 8.50 | **28.12** |
| mean torsion coverage | 59 % | 69 % |
| coverage as a fraction of its ceiling | 83 % | **93 %** |
| warfarin coverage | 69 % | **100 %** |
| decane / octanediol coverage | 85 % / 88 % | **100 % / 100 %** |
| peg3 coverage | 77 % | 90 % |
| seconds per molecule | 0.229 | **0.941** |

On the bundled library: 2.18 → **2.82** conformers kept, coverage 28 % → 30 %,
coverage/ceiling 95 % → **97 %**, cost 0.117 → 0.205 s/molecule.

**The stated target is 90 % of the ceiling for any molecule with a rotatable bond,
and a note when it is missed.**  After the change the varied set sits at 93 % and the
bundled library at 97 %; the exception is ibuprofen at 59 % of a 100 % ceiling, and
the module says why in a note rather than padding the ensemble: ETKDGv3 returns only
nine distinct shapes at 0.5 Å from forty attempts, because a shape-space ceiling is a
property of the molecule and not of the budget.  A rigid molecule is never padded to
look better either — benzene keeps exactly one conformer at any attempt count.

**The thresholds that were *not* changed, and why.**  The embedder's duplicate filter
stays equal to the pruning threshold: measured, the conformers ETKDGv3 returns at
0.5 Å are already 0.56–0.88 Å apart, so a 0.25 Å embedder threshold embeds ~3× more
and the greedy prune then discards every extra one — cost with no coverage.  The
10 kcal/mol energy window also stays: it removes 0–3 conformers on this set, and at
300 K (kT ≈ 0.6 kcal/mol) a 16 kT window discards unrelaxed outliers, not accessible
states.  Widening it to 25 kcal/mol changed no molecule's coverage by more than a
few points once the attempt count was scaled.

**The cost, stated.**  Coverage costs ~4× the time on the varied set (0.229 → 0.941
s/molecule) and ~5× on the 142-molecule library through the pre-filter path (8.1 →
42–48 s of embedding), because the scaling only spends attempts on the molecules that
have rotors.  `--fixed-attempts` (or `n_conformers=16`, or `scale_with_rotors=False`)
restores the old behaviour exactly, and `odock conformers` prints the accounting so
the trade is visible before it is paid.

Through the pre-filter path (4 attempts, the old default) the 142-molecule library —
the 17 demo molecules plus the 125-molecule pool — embeds at **1.50 conformers per
molecule in 8.1 s (0.057 s/molecule)**.  USR descriptors over those ensembles cost
**0.16 s for the whole library (1.1 ms per molecule)** — about 50× cheaper than the
embedding that has to happen either way, and one overlay costs about what twelve
libraries' worth of descriptors cost.  That is the whole case for a pre-filter: it
is cheap *next to the thing it avoids*, and nothing else about it is cheap.

A molecule that already carries 3-D coordinates — a docked pose, a crystal structure,
a multi-model SDF — is measured **as it is** when the caller asks for that
(`odock conformers --keep-input`, `build_ensemble(use_input_conformers=True)`):
nothing is embedded, minimised or pruned, the energies stay unavailable and a note
says so, because re-embedding a pose in order to report on it would replace the very
geometry the report is about.  A `.smi` file has no coordinates to keep, and the
command says which molecules it embedded instead.

## 2. The overlay: parameters, and the pose it finds

A heavy atom carries the normalised Gaussian density `exp(-|x − r|² / 2σ²)`, so the
overlap integral of two molecules is analytic:

```
O_AB = Σ_i Σ_j exp(-|r_i − s_j|² / 4σ²)            (the (πσ²)^{3/2} prefactor cancels)
shape  = O_AB / (O_AA + O_BB − O_AB)               Tanimoto   (the default)
       = O_AB / sqrt(O_AA · O_BB)                  Carbo      (shape_metric="carbo")
ESP    = Carbo index of the two Gasteiger fields   (raw, in [-1, 1])
score  = 0.5 · shape + 0.5 · max(0, ESP)           (documented convention, not a fit)
```

The algebra is hand-checkable and is checked in `tests/test_overlay.py`: one atom
against itself is exactly 1.0, two single-atom molecules 1 Å apart at σ = 0.5 give
`1/e = 0.3679`, `shape_tanimoto(1, 2, 2) = 1/3` and `carbo_index(1, 2, 2) = 0.5`.
A molecule overlaid on a rigid copy of itself scores exactly 1.000 on both terms.

**The alignment is six-dimensional and deterministic.**  The probe is centred on its
heavy-atom centroid and then the search maximises the *shape* term (not the
combined score — see below) over:

1. a fixed seed set — the identity, the 24 proper rotations of a cube, the 24
   principal-axis alignments and 256 quasi-uniform rotations from a fixed PCG64
   stream (all evaluated in one batched kernel call);
2. a multi-resolution refinement from the best two seeds: at each of 9 levels, 64
   poses are drawn by perturbing the incumbent with a random rotation of at most
   `0.4 · 2^-level` radians and a Gaussian translation of scale `0.5 · 2^-level` Å,
   and the best improvement is accepted.

Only proper rotations are allowed, so a molecule is never mirrored onto another.
Nothing in the answer depends on the wall clock, the thread count or the platform.

| pair (σ = 0.5 Å) | search finds | chemically correct superposition | gap |
|---|---|---|---|
| benzamidine / hydroxybenzamidine | 0.921 | 0.923 | −0.002 |
| benzamidine / fluorobenzamidine | 0.750 | 0.742 | +0.008 |
| benzamidine / chloro_benzamidine | 0.905 | 0.903 | +0.002 |
| benzamidine / benzamidine_methyl | 0.904 | 0.918 | −0.014 |
| benzamidine / benzylamidine | 0.621 | 0.636 | −0.015 |

"The chemically correct superposition" is the probe placed on the reference through
their maximum common substructure with a proper rotation (Kabsch); it is a *lower*
bound on the true shape optimum, so a search that lands below it has failed.  The
table is why the search is built the way it is: the first version searched rotations
only, from 49 structured seeds, and scored benzamidine/hydroxybenzamidine at
**0.833** — because centring both molecules on their own heavy-atom centroid is not
the translation the best overlay wants, and a para-hydroxyl moves the centroid by
~0.3 Å.  Adding the translation degree of freedom took it to 0.896; adding the
quasi-uniform cloud and the multi-resolution refinement took it to 0.921, against
0.923 for the chemical pose, while a brute-force sweep of 8 000 random rotations
with a coarse translation grid reached only 0.890 on the same pair.

**σ = 0.5 Å, measured.**  Benzene against:

| probe | σ = 0.5 Å | σ = 0.8 Å |
|---|---|---|
| pyridine | 0.997 | 0.998 |
| toluene | 0.887 | 0.944 |
| naphthalene | 0.599 | 0.753 |
| cyclohexane (chair) | 0.926 | 0.989 |

At σ = 0.8 Å a flat ring and a puckered chair are the same blob (0.989) and a methyl
costs 0.056; at 0.5 Å the third dimension is visible again.  The combined-score
ordering of the bundled library is identical at σ = 0.4, 0.5, 0.6 and 0.8 (amidine
ranks 1, 2, 3, 7, 8 either way), so σ was chosen on the shape term's own geometry,
where the difference is large and measurable.

**The alignment maximises shape, not the combined score.**  Letting the electrostatic
term steer the pose is measurably worse: benzamidine/hydroxybenzamidine comes out at
shape 0.898 instead of 0.913, benzamidine/fluorobenzamidine at 0.548 instead of
0.724, and benzylamidine at 0.620 instead of 0.690 — a marginally better field
agreement buys a visibly wrong pose.  The field is therefore *read* at the
shape-optimal pose (`align_on="shape"`), and `align_on="combined"` is available for
comparison.

**Cost.**  One molecule-versus-reference overlay is **~13 ms** on this machine for a
17-heavy-atom pair (1 433 candidate poses evaluated), and the batched kernel is what
makes that affordable: the same search evaluated one pose at a time costs ~50 µs of
interpreter overhead per pose and dominated the arithmetic.  A 30-molecule benchmark
with two conformers and five references is ~3 s of overlay; the 142-molecule library
is ~22 s, which is why the next section exists.

## 3. What each term contributes

The two terms are reported for every molecule (`Ranking.details`), because the
combined score hides which one did the ranking.  On the bundled library
(σ = 0.5 Å, 4 conformers, leave-one-out over the five ring-amidines):

| rank | electrostatics on | score | shape only | score |
|---|---|---|---|---|
| 1 | benzamidine_methyl | 0.947 | salicylic_acid | 0.972 |
| 2 | benzamidine | 0.939 | nicotinamide | 0.969 |
| 3 | chloro_benzamidine | 0.936 | chloro_benzamidine | 0.919 |
| 4 | nicotinamide | 0.898 | benzamidine | 0.919 |
| 5 | salicylic_acid | 0.868 | benzamidine_methyl | 0.904 |
| 6 | isonicotinic_acid | 0.802 | hydroxybenzamidine | 0.900 |
| 7 | hydroxybenzamidine | 0.759 | isonicotinic_acid | 0.899 |
| 8 | fluorobenzamidine | 0.695 | fluorobenzamidine | 0.833 |

* **The shape term cannot rank the series.**  Salicylic acid and nicotinamide are
  benzene rings carrying one small substituent, and the shape Tanimoto — normalised
  by the self-overlap — is nearly blind to that: 0.972 and 0.969 against a
  benzamidine reference.  With the electrostatic term off, **one** amidine is in the
  top three.
* **The electrostatic term carries the series.**  With it on, all three amidines
  that differ from the reference only at the amidine-bearing carbon take ranks 1–3
  (mean amidine rank 4.2 of 17, against 9.0 for a random ranking).  The two
  para-substituted amidines still land at 7 and 8: a para-hydroxyl changes the
  Gasteiger field enough that the Carbo index of the two fields drops to 0.62.
* **So the overlay recovers the series as an enrichment, not as the pharmacophore
  model's 1–5 sweep.**  `docs/PHARMACOPHORE.md` gets ranks 1–5 at 0.918–0.944 from
  the donor/acceptor/aromatic pattern; a shape overlay is simply not that
  measurement, and this document says so rather than quoting a favourable cut.
* **Weighting.**  0.5/0.5 matches the older grid scorer deliberately, so the
  benchmark measures the method rather than a re-tuned weighting.  Measured
  alternatives on the easy decoy band: shape weight 0.8 → AUC 0.888, 0.2 → 0.904,
  0.5 → 0.920.

## 4. The pre-filter: the speedup and its price

A 3-D screen of a library costs one overlay per (molecule × reference × conformer).
USR descriptors — 12 numbers per conformer, rotation-invariant — can cut the library
first.  Measured on the 142-molecule library with the five ring-amidines as actives,
leave-one-out (`odock.overlay.usr_prefilter`):

| keep | molecules kept | actives kept | active recall | with self-match | estimated docking |
|---|---|---|---|---|---|
| 5 % | 8 of 142 | **2 of 5** | **40 %** | 5 of 5 | 436 s → **25 s** (94 %) |
| 10 % | 15 of 142 | 2 of 5 | 40 % | 5 of 5 | 436 s → 46 s (89 %) |
| 20 % | 29 of 142 | 4 of 5 | 80 % | 5 of 5 | 436 s → 89 s (80 %) |

The overlay cost it saves is *measured*, not asserted: the overlay is run on the kept
molecules through ensembles that were already built (never re-embedded — that would
double the dominant cost), and the result is extrapolated linearly to the full
library.  At 5 % that is **1.23 s measured on 8 molecules against 21.8 s
extrapolated (94 % saved)**; the descriptors themselves cost 0.16 s for all 142.

**The recall is the number that matters, and here it is bad.**  USR is a shape
summary: it ignores element identity (benzene–pyridine 0.981, measured in
`tests/test_conformers.py`), so the pool's other flat aromatics outrank the
amidines.  On the same library the 2-D fingerprint pre-filter keeps **5 of 5**
actives at the same 5 % cut.  The honest recommendation is therefore:

* if a 2-D filter can be used, use it — it is free and it dominates USR here;
* if a 3-D filter is required, **20 % is the smallest cut this library supports**
  (80 % recall), and the recall must be re-measured on the library in hand, because
  40 % at 5 % is a property of *this* pool, not a constant;
* the `self_match_recall` column is reported next to the recall because with
  self-match allowed the filter keeps every active for free — an active is its own
  best match, and quoting that number as a recall is the leak this whole benchmark
  exists to remove.

## 5. What this does not establish

* **The ensemble is a sample, not the space.**  ETKDGv3 at one seed samples what its
  knowledge-based torsion terms believe is reasonable; a high coverage number means
  "this sample is spread out", not "this is the conformational ensemble".  A rigid
  molecule's ensemble is one geometry, `conformers.py` says so in a note, and every
  score from it is a statement about that one geometry.
* **A conformer-resolved score is not a binding prediction.**  The overlay has no
  receptor, no dielectric, no desolvation, no entropy and no atom typing beyond
  Gasteiger.  It measures geometric and electrostatic similarity at the best pose it
  can find; two molecules can be very similar in shape and do entirely different
  chemistry.
* **The alignment is one of many minima.**  The search is a multi-start local
  method with a fixed budget: it reaches the chemically correct superposition on the
  pairs tested and beats a coarse brute force, and it is not a global optimiser.  The
  reported score is `max over the poses that were tried`.
* **A pre-filter's speedup means nothing without its recall**, and this one's recall
  is 40 % at 5 %.  The table above is the whole statement; the speedup alone would be
  a lie of omission.
* **σ, the weights and the coefficient are conventions.**  Every one of them is
  documented, measured where it matters (σ on the shape term, the weights on the
  easy decoy band) and exposed as an argument; none of them is fitted to activity
  data, because there is none in this repository.
* **The benchmark that judges the overlay is a case study** — five actives, one
  target, one chemotype.  See the next section and `docs/LBVS.md`.

## 6. What it was worth: the measured delta

`odock.lbvs` scores the same actives and decoys with both engines, so the delta is
visible on the same data (details and the full tables in `docs/LBVS.md`):

| band | overlay (new) AUC / BEDROC | crude grid (old) AUC / BEDROC |
|---|---|---|
| easy (25 decoys) | 0.920 / 0.941 | **1.000 / 1.000** |
| hard (5 decoys) | 0.600 / 0.135 | **0.920 / 0.998** |

**The overlay did not improve the enrichment numbers on the bundled decoy bands, and
this repository reports that rather than the rewrite.**  The decomposition says why,
and it is not a bug in the search:

| variant | easy-band AUC |
|---|---|
| overlay, default | 0.920 |
| overlay, alignment maximises the combined score | 0.912 |
| overlay, shape weight 0.8 | 0.888 |
| overlay, shape weight 0.2 | 0.904 |
| overlay, one reference (benzamidine) | 0.704 |
| overlay, pose search almost disabled | 0.848 |

Removing the pose freedom makes it *worse* (0.848), so the free alignment is not what
costs the enrichment.  The difference is in the score itself: the grid scorer's
cosine over Gasteiger charge *grids* is more forgiving of a polar substituent
(hydroxybenzamidine scores 0.917 there against 0.759 for the field Carbo index), and
its fixed pharmacophore-frame pose is a strong regulariser on a set whose decoys are
dissimilar.  Both bootstrap intervals overlap on both bands (easy: overlay
[0.79, 1.00] against crude [1.00, 1.00]), so this set cannot rank the engines — it can
only say that the new one did not beat the old one here.

The overlay is the default engine anyway, for reasons the benchmark does not measure:
it is the general method (the grid scorer needs a pharmacophore model to build its
frame, so it cannot score an arbitrary pair), it reproduces the chemically correct
superposition where the grid scorer's pose is arbitrary, it is best-over-conformers,
and it reports both terms per molecule instead of one number.  `--shape crude`
restores the old numbers exactly, and `--methods overlay,crude` prints both.

## 7. The hypothesis: were the thin ensembles the reason?

The numbers in §6 invite an explanation that does not blame the score: the benchmark
ran with **two** conformers per molecule, `odock conformers` reported **2.18 kept of
16**, and the crude scorer's advantage is partly its fixed pose.  If every 3-D method
is handicapped by a thin ensemble before it starts, then fixing the ensemble should
move the ranking.  §1.2 fixed it.  This section is the result, and **the hypothesis is
falsified**: better ensembles did not change the ranking.

**The overlay versus the crude grid, same actives, same decoys, same controls:**

| band | conformers | overlay AUC / BEDROC | overlay shape-only | crude grid AUC / BEDROC |
|---|---|---|---|---|
| easy | 2 (published protocol) | 0.920 / 0.941 | 0.864 / 0.549 | **1.000 / 1.000** |
| easy | 16 (old default) | 0.936 / 0.944 | 0.824 / 0.534 | **1.000 / 1.000** |
| easy | rotor-scaled (new default) | 0.936 / 0.944 | 0.824 / 0.534 | **1.000 / 1.000** |
| hard | 2 (published protocol) | 0.600 / 0.135 | 0.640 / 0.135 | **0.920 / 0.998** |
| hard | 16 (old default) | 0.640 / 0.135 | 0.640 / 0.135 | 0.680 / 0.133 |
| hard | rotor-scaled (new default) | 0.640 / 0.135 | 0.640 / 0.135 | 0.680 / 0.133 |

* **The overlay moves by 0.016 of AUC on the easy band and 0.040 on the hard band,
  and never reaches the crude scorer.**  The easy band's 0.920 → 0.936 is the only
  gain, it is inside the bootstrap interval, and it does not change a single rank at
  the top (EF1 % and EF5 % are 6.00 and 6.00 either way).
* **On the hard band the extra conformers make the *crude* scorer worse** (0.920 →
  0.680): its pose comes from the pharmacophore frame, so giving it more geometries
  gives the decoys more chances to find a good one.  A better ensemble is not
  automatically a better ranking — for a scorer that maximises over poses, it can be
  a worse one.
* **The actives are rigid.**  The five ring-amidines keep **one** conformer each at
  16 attempts, at 40, and at any count in between, so no ensemble change can move
  them; the decoys that beat them are flat aromatics with the actives' ring shape.
  That is the structural reason the hypothesis could not have held on this set, and
  it was visible in §1.1 before the experiment was run.

**The USR pre-filter recall is unchanged too**, at every cut, while its cost rises
~5×:

| keep | conformers | actives kept | recall | conf/molecule | embedding |
|---|---|---|---|---|---|
| 5 % | 4 attempts | 2 of 5 | 40 % | 1.50 | 8.1 s |
| 5 % | rotor-scaled | 2 of 5 | 40 % | 3.65 | 48.4 s |
| 10 % | 4 attempts | 2 of 5 | 40 % | 1.50 | 8.5 s |
| 10 % | rotor-scaled | 2 of 5 | 40 % | 3.65 | 42.0 s |
| 20 % | 4 attempts | 4 of 5 | 80 % | 1.50 | 8.1 s |
| 20 % | rotor-scaled | 4 of 5 | 80 % | 3.65 | 42.4 s |

**What this establishes.**  The overlay's loss to the grid scorer on the bundled
decoys is *not* an artefact of a thin ensemble: tripling the conformer count, taking
the mean coverage from 59 % to 69 % and the coverage-ceiling ratio from 83 % to 93 %
does not move the ranking.  The explanation therefore shifts back to the score form
— the grid scorer's charge-grid cosine is more forgiving of a polar substituent, and
its fixed pose regularises — which is where §6 left it.  One hypothesis is dead and
the other survives, on the same data, which is the useful outcome.

**What it does not establish.**  A negative result on five actives against 25
dissimilar decoys is a case study, and both intervals overlap on both bands.  The
ensemble fix is justified by §1.2 — coverage against its ceiling, on molecules with
rotors — and not by this benchmark; a reader who only cares about the bundled
enrichment numbers should note that the fix costs 4–5× the embedding time and buys
nothing there, and that `--fixed-attempts` turns it off.

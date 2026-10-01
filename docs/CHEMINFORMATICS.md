# Ligand cheminformatics — similarity, scaffolds, series, R-groups, diversity

Docking answers "how does this molecule bind". A real project also asks the
*ligand-side* questions, and they are all 2-D questions:

* **What else looks like this hit?** — fingerprints and a similarity search over
  the library ([`odock.ligandsim`](../python/odock/ligandsim.py), `odock similar`).
* **Is my library diverse, and which subset should I dock first?** — MaxMin and
  sphere-exclusion picking, Butina clustering, scaffold coverage
  (`odock diverse`, `odock screen --diverse N`).
* **What do the actives share?** — Bemis-Murcko scaffolds, scaffold families and
  the R-group matrix ([`odock.scaffold`](../python/odock/scaffold.py),
  `odock scaffolds`, `odock rgroups`).
* **Which analogues should I make next?** — a series table with affinity deltas
  and a matched-molecular-pair analysis (`odock rgroups --mmp`,
  `odock scaffolds --report`).

Everything in this document is **measured on this tree** with the bundled
library; every number can be reproduced with the command printed beside it.

**The one thing to take away.** A 2-D similarity is a statement about
*structure*, never about activity. A Murcko scaffold is a *heuristic* that groups
molecules, not a series. An affinity delta is only a structure-activity result
when the affinity was measured — or docked consistently — and it is larger than
the docking noise. The last section lists what each method does not establish,
with the measured error bars.

## The library every number below comes from

[`demo/libraries/library.smi`](../demo/libraries/library.smi) — 17 molecules, hand-checked in so a
clone can run the whole walk-through offline. Six of them are a benzamidine
series (the 3PTB ligand and five analogues), nine are drug-like molecules and two
are deliberate filter failures (triphenylene on LogP, benzoquinone as a PAINS
quinone).

```bash
odock scaffolds -i demo/libraries/library.smi
```

```text
count  best dA  scaffold                      representative
-----  -------  ----------------------------  --------------
11     -6.420   c1ccccc1                      ibuprofen
2      -5.823   c1ccncc1                      nicotinamide
1               O=C1C=CC(=O)C=C1              benzoquinone
1      -5.329   O=c1[nH]c(=O)c2[nH]cnc2[nH]1  caffeine
1      -7.338   O=c1oc2ccccc2cc1Cc1ccccc1     warfarin
1               c1ccc2c(c1)c1ccccc1c1ccccc21  triphenylene

scaffolds: 6 distinct Murcko scaffold(s) for 17 molecule(s)
largest series: 11 molecule(s) on c1ccccc1
best scaffold by affinity: O=c1oc2ccccc2cc1Cc1ccccc1 (-7.338)
```

**Six scaffolds, and the largest one holds eleven molecules** — but that group is
*not* a series: benzamidine, aspirin, ibuprofen and paracetamol all reduce to
`c1ccccc1` because a Murcko scaffold keeps rings and linkers and strips terminal
substituents. The R-group table (§4) is what separates them again. That is the
single most misunderstood property of scaffold analysis, and it is visible in one
line of output here.

## 1. Fingerprints and similarity (`odock.ligandsim`)

Six fingerprint kinds sit behind one documented API; no RDKit object ever escapes
it, so a caller can serialise, cache and diff a fingerprint without importing
RDKit:

| kind | what it sees | length | default |
|---|---|---|---|
| `morgan` | circular substructures (Morgan/ECFP); radius 2 = ECFP4 | 2048 | ✔ |
| `rdkit` | every linear and branched path of 1–7 bonds | 2048 | |
| `atom_pair` | (atom type, atom type, topological distance) triples | 2048 | |
| `torsion` | the four-atom topological torsions | 2048 | |
| `maccs` | the 166 public MACCS substructure keys | 167 | |
| `pharmacophore` | Gobbi pharmacophore *pairs*, from one 3-D conformer | ~40 000 (sparse) | |

`use_features=True` on Morgan is the feature (FCFP) variant: it hashes
pharmacophore features instead of atom environments.

Three coefficients are implemented, each with the closed form and a hand-checked
test: **Tanimoto** `|A∩B| / |A∪B|`, **Dice** `2|A∩B| / (|A|+|B|)` and **Tversky**
`|A∩B| / (|A∩B| + α|A\B| + β|B\A|)` (α = β = 1 is Tanimoto, α = β = ½ is Dice,
α < β is a "query is the smaller structure" search).

### The measured answer to "what looks like benzamidine"

```bash
odock similar -q 'N=C(N)c1ccccc1' -i demo/libraries/library.smi --cutoff 0.5
```

| rank | molecule | Tanimoto (Morgan r2, 2048 bits) |
|---|---|---|
| 1 | benzamidine | 1.0000 |
| 2 | hydroxybenzamidine | **0.5714** |
| 3 | fluorobenzamidine | 0.5455 |
| — | benzamidine_methyl | 0.4583 |
| — | benzylamidine | 0.4583 |
| — | chloro_benzamidine | 0.4400 |

**The closest analogue of benzamidine in the library is its 4-hydroxy analogue at
0.571.** Only two of the other sixteen molecules clear the usual 0.5 "similar"
cut-off; the 2-chloro analogue is at 0.44 and is *not* reported at that cut-off.
The ordering is chemically sensible (the 4-substituted analogues above the
N-methyl and the one-carbon-extended benzylamidine, the 2-substituted below
them), which is what a 2-D fingerprint is good at — and nothing here says the
4-hydroxy analogue binds differently, only that it *looks* most like the query.

The number is hand-checkable, and the test that pins it says so: benzamidine sets
**16** Morgan bits, its 4-hydroxy analogue **17**, they share **12**, so the union
is `16 + 17 − 12 = 21` and the similarity is `12/21 = 4/7 = 0.5714285714…`
(`tests/test_ligandsim.py::test_tanimoto_of_benzamidine_and_its_hydroxy_analogue_is_four_sevenths`).

### Cost

| search | work | measured |
|---|---|---|
| 2-D, 17 molecules | 17 comparisons | 0.37 ms (**≈ 22 µs per molecule**) |
| 2-D, 100 000 molecules | 10⁵ comparisons | ≈ 2 s per query, single core |
| similarity matrix, 1 000 × 2 048 bits | one blocked BLAS multiply | well under a second |

The matrix is computed in row blocks, so a 100 000-member library can be walked
without ever materialising its 10¹⁰-entry matrix; two empty fingerprints score
**0.0** rather than 1.0, because a "perfect match" between two structures nobody
described would be the wrong answer.

## 2. Multi-conformer similarity — and what it costs

Only the `pharmacophore` fingerprint is conformer-dependent. Morgan, path,
atom-pair, torsion and MACCS are 2-D graph descriptions: every conformer of a
molecule yields the *same* bits, so a "multi-conformer" search over them would be
the 2-D search repeated and is not offered. `find_analogues_3d` therefore embeds
both sides with ETKDGv3 (8 conformers per molecule by default) and scores a
library member as the **best Tanimoto over every conformer pair** — an honest
definition ("some conformation of this molecule looks like some conformation of
that one") with an honest price:

| step | measured on this machine |
|---|---|
| embed 8 conformers + 8 pharmacophore fingerprints | **75 ms per molecule** |
| one conformer-pair comparison | **62 µs** |
| 17 molecules × 8 conformers (1 088 pairs) | 1.27 s to build + 0.07 s to search |
| the same 2-D search | 0.37 ms |

So the 3-D path is roughly **three thousand times** more expensive per molecule,
and it needs a 3-D structure it had to invent (the conformers depend on the
embedding seed, so the score does too).

It is also *not* automatically the better answer, and the demo shows why. The
best 3-D analogue of benzamidine in the library is **benzamidine_methyl at
1.000** — exactly tied with the query itself — while the 2-D search puts it fifth
at 0.458:

```text
2-D (Morgan r2):  1. benzamidine 1.000   2. hydroxybenzamidine 0.571  …  5. benzamidine_methyl 0.458
3-D (pharmacophore): 1. benzamidine 1.000  2. benzamidine_methyl 1.000  3. chloro_benzamidine 1.000
```

The reason is that an N-methyl adds no pharmacophore feature (no new donor,
acceptor, aromatic ring, positive or negative ionisable group) and does not
change the distances between the features that do exist, so a two-point
pharmacophore model cannot see it. That is a property of the *model*, not a bug —
and it is why the 3-D fingerprint should be read as a shape/pharmacophore
question ("could this scaffold present the same groups at the same distances"),
not as a better 2-D similarity.

## 3. Scaffolds and series (`odock.scaffold`)

* `murcko_scaffold` / `scaffold_of` — the Bemis-Murcko framework, optionally the
  **generic** skeleton (every atom and bond type erased).
* `scaffold_groups` — the library grouped by scaffold, with counts, a
  representative molecule (the heaviest member) and the affinity of each member.
* `scaffold_clusters` — those scaffolds clustered by similarity (Butina, see §6),
  which is what turns "six scaffolds" into "`k` chemotype families".
* `most_common_scaffold` — the default core for the R-group decomposition: the
  chemotype the library is actually built around.
* `maximum_common_substructure` — RDKit's MCS as the "common core of a set".

Measured on the demo library:

| view | result |
|---|---|
| Murcko scaffolds | **6** (benzene ×11, pyridine ×2, caffeine, warfarin, triphenylene, benzoquinone) |
| generic skeletons | **5** — benzene and pyridine merge into `C1CCCCC1` (13 molecules), because the element is erased |
| scaffold families at 0.65 | 6 (nothing merges) |
| scaffold families at 0.30 | **5** — benzene and pyridine scaffolds are at Tanimoto 1/3 |
| largest series | 11 molecules on `c1ccccc1` |

Two heuristics are worth stating plainly, because the tool shows them:

* **The most common scaffold is not the most common *series*.** With
  `--core mcs` the MCS of the six amidines is `Cc1ccccc1` (toluene), not
  `N=C(N)c1ccccc1`: benzylamidine (`N=C(N)Cc1ccccc1`) does not contain the
  amidine-on-ring motif, so that one outlier drags the common core down to
  toluene and only one molecule still matches it. The MCS of the whole 17-molecule
  library is **empty** — caffeine and warfarin share no ring system at all. A
  common core exists for a *series*, which is why the default core is the most
  common scaffold and the CLI names the core it used.
* **A generic skeleton answers a different question.** Erasing atom types merges
  benzene with pyridine (6 scaffolds → 5 skeletons) but also merges an aromatic
  ring with a saturated one, so it measures shape, not chemotype.

## 4. R-group decomposition (`odock rgroups`)

Given a core, every molecule is split into its substituent at every attachment
point, and the result is the molecule × R-group matrix a medicinal chemist asks
for on day one:

```bash
odock rgroups -i demo/libraries/library.smi --core 'N=C(N)c1ccccc1' --affinities out/3ptb-screen/results.jsonl
```

| name | affinity | matched | R1 | R2 | R3 |
|---|---|---|---|---|---|
| benzamidine | −5.901 | yes | H | H | H |
| benzamidine_methyl | −5.516 | yes | *C | H | H |
| hydroxybenzamidine | −6.420 | yes | H | *O | H |
| fluorobenzamidine | −6.234 | yes | H | *F | H |
| chloro_benzamidine | −6.180 | yes | H | H | *Cl |
| benzylamidine | — | **no** | | | |

An R-group is written with its attachment point as `*` (`*C` methyl, `*O`
hydroxyl, `*C(=O)O` carboxylic acid) and a hydrogen as `H`. The marker is kept on
purpose: a bare `O` would be water and a bare `C(=O)O` formic acid, and neither is
the substituent.

Four properties of the implementation matter more than the table itself:

1. **The labels are positional, not accidental.** RDKit numbers attachment
   points in the order it found them; this layer renumbers them `R1..Rk` by the
   canonical rank of the core atom they sit on, so the same chemical position
   always gets the same column and two runs can be diffed.
2. **The alignment is global.** The decomposition is RDKit's
   `rdRGroupDecomposition`, which solves the assignment over *the whole set* at
   once, so a symmetric core does not get a different (but equivalent) numbering
   for every molecule.
3. **Molecules that do not match are reported, not dropped.** Benzylamidine is
   `matched=no` with the reason; triphenylene is `matched=no` with a different
   reason — it contains a benzene ring, but the "substituent" would have to close
   a ring (a fused-ring analogue), which a core plus *acyclic* R-groups cannot
   express. Silently emitting a five-atom "R-group" that contains a ring closure
   would be a wrong answer with the shape of a right one.
4. **It writes a spreadsheet.** `-o table.xlsx` writes the matrix plus a `series`
   sheet and a `core` provenance sheet; `-o table.csv` writes the matrix alone
   when `openpyxl` is unavailable. Both are also available as JSON
   (`--json-out`) for a pipeline.

With the *scaffold-derived* core (`--core` omitted → `c1ccccc1`) 12 of 17
molecules match and 3 attachment points are found; the amidine of benzamidine
then shows up as a *substituent* (`*C(=N)N`) rather than as part of the core.
Both views are correct; they answer different questions.

## 5. Series and matched molecular pairs

`series_table` adds the affinity and a delta against a reference member (by
default the first matched molecule, i.e. the parent for a library written
parent-first). `matched_pairs` reports every pair of matched molecules that
differ in **exactly one** attachment point — the smallest controlled experiment a
series contains.

### The worked example

```bash
odock rgroups -i demo/libraries/library.smi --core 'N=C(N)c1ccccc1' \
    --affinities out/3ptb-screen/results.jsonl --mmp
```

```text
label from                  rg        to                    rg               dA  dheavy  significant
R1    benzamidine           H         hydroxybenzamidine    *O           -0.520      +1  no
R1    benzamidine           H         fluorobenzamidine     *F           -0.330      +1  no
R1    hydroxybenzamidine    *O        fluorobenzamidine     *F           +0.190      +0  no
```

(That output is the six-molecule `--core` case in the test suite; on the full
library with the amidine core the same pair appears as `R2`.)

The para **H → OH** change, with affinities docked at exhaustiveness 8 and
seed 42 into 3PTB, is

```text
hydroxybenzamidine − benzamidine = −6.420006909246549 − (−5.900566019865151)
                                 = −0.519440889381398 kcal/mol
```

so the hydroxyl looks **0.52 kcal/mol better**, and the fluoro analogue 0.33
kcal/mol better, and the chloro 0.28 better. Read the rank order and you would
make the 4-hydroxy analogue next. Read the noise and you would not.

### What a delta needs before it means anything

* **It needs the affinity to exist.** `matched_pairs` keeps only pairs where both
  molecules have an affinity (`require_affinity=False` reports the rest with
  `delta=None`); a series with no affinities prints structures and no deltas
  rather than a table of zeroes.
* **It needs to be bigger than the noise of the number.** On this project's own
  benchmark the affinity ranking of a *flexible* ligand is not reproducible: 1M17
  (erlotinib, 11 rotors) moves its top pose by **3.17 Å** between search seeds,
  and the documentation warns against ordering two hits that differ by less than
  about **1 kcal/mol** ([`BENCHMARK.md`](BENCHMARK.md) §"What the regression
  check does", [`SCREENING.md`](SCREENING.md) §"What this does not establish").
  `DOCKING_NOISE_KCAL = 1.0` encodes exactly that, and **all five matched pairs
  of the benzamidine series are marked `significant = no`** — every delta above is
  smaller than the noise floor. The honest reading is "the docking cannot tell
  these apart"; a chemist can still see the *direction*, and an assay is what
  settles it.
* **A difference of one attachment point is not a difference of one atom.**
  `H → *C(=O)O` adds three heavy atoms, a donor and an acceptor. The table
  therefore reports how many heavy atoms each substituent has and marks the pairs
  that change the size by more than two. On the demo library with the benzene core
  there are 17 single-point pairs and exactly **five** of them clear the 1 kcal/mol
  noise floor: four involve warfarin (`+13` to `+14` heavy atoms, deltas of −1.4 to
  −2.4 kcal/mol — two very different molecules that happen to share a core, not a
  substituent effect) and one is `acetanilide → paracetamol`, a genuine `H → OH`
  change at `−1.012 kcal/mol` sitting exactly on the noise floor. Without the size
  column the first four would look like the strongest structure-activity signal in
  the library; with it, they are visibly not comparisons at all.
* **It needs a consistent protocol.** All affinities in one series must come from
  one receptor, one box, one force field and one set of search settings; docked
  values from two campaigns are not comparable, which is why the screening
  pipeline refuses to mix them in one results file and why `--affinities` reads
  the campaign's own manifest-stamped rows.

## 6. Diversity selection and clustering

* `maxmin_pick` — greedy farthest-point: start from a chosen molecule, then
  repeatedly add the molecule whose *nearest* already-picked neighbour is the
  farthest. `O(N·n)` comparisons, no matrix.
* `sphere_exclusion_pick` — one pass in library order, keeping a molecule only
  when it is below `cutoff` similarity to every molecule already kept.
* `butina_cluster` — Butina's sphere-exclusion around the most-connected molecule,
  with counts, singletons and a **medoid** representative. It is verified against
  RDKit's own `Butina.ClusterData`: the two agree on the demo library at
  Tanimoto cut-offs 0.50, 0.45 and 0.40 (same partition, same clusters).
* `scaffold_coverage` — the fraction of the library's scaffold space a subset
  covers, which is the number that says whether a subset is actually diverse.

### Measured: how small a subset can be

```bash
odock diverse -i demo/libraries/library.smi -n 6
```

| method | subset | molecules | worst pair inside | scaffolds covered |
|---|---|---|---|---|
| MaxMin | 3 | 17.6 % | 0.125 | 2 of 6 = **33 %** |
| MaxMin | 6 | 35.3 % | 0.216 | 4 of 6 = **67 %** |
| MaxMin | 9 | 52.9 % | 0.263 | 6 of 6 = **100 %** |
| MaxMin | 12 | 70.6 % | 0.448 | 6 of 6 = 100 % |
| sphere exclusion, cutoff 0.3 | 9 | 52.9 % | 0.263 | 6 of 6 = 100 % |
| sphere exclusion, cutoff 0.5 | 14 | 82.4 % | 0.458 | 6 of 6 = 100 % |
| sphere exclusion, cutoff 0.7 | 17 | 100 % | 0.625 | 6 of 6 = 100 % |

Two things are visible in that table. **Six MaxMin picks cover two thirds of the
scaffold space for a third of the molecules** (the picks are benzamidine,
caffeine, paracetamol, benzoquinone, ibuprofen, warfarin; their tightest internal
similarity is 0.216), and the first six molecules of the file — the six amidines —
cover **one** scaffold for the same 35 %. **Sphere exclusion is the weaker of the
two methods here**: at a 0.5 cut-off it needs 82 % of the molecules to reach the
100 % coverage MaxMin reaches with 53 %, because it never reconsiders the first
molecule it kept and its result depends on library order. At the tighter 0.3
cut-off the two happen to agree on this small library — 9 molecules each, tightest
pair 0.263 — which is a reminder that with 17 molecules the two algorithms can
coincide.

The 17-molecule library is small enough that the whole picture can be checked by
hand; at library scale (10⁵ molecules, n = 1 000) MaxMin costs 10⁸ comparisons,
which is roughly half an hour at the 22 µs-per-comparison rate measured above —
and the reason `start="centroid"` — the one option that needs an extra N² pass —
is opt-in.

### Butina clustering of the molecules themselves

At Tanimoto 0.45 the demo library is 12 clusters (10 singletons) and the largest
cluster is exactly the five ring-amidines, with benzamidine as the medoid — the
same grouping a chemist would draw, produced without being told about the series.
Representatives are medoids (the member closest to the rest of the cluster), not
"the first member", so a representative is a molecule worth looking at.

### `--diverse N` inside a screening campaign

```bash
odock screen -r demo/systems/3ptb/receptor.pdbqt -i demo/libraries/library.smi --box demo/systems/3ptb/box.json \
    -o out/3ptb-screen --diverse 6 -e 8 --seed 42
```

The subset is selected **from the molecules that pass the drug-likeness filters**,
so `--diverse 100` really is 100 dockable molecules, and it is written to
`library_diverse.sdf` next to the results (properties preserved) together with
`diverse.json`. Measured on the bundled campaign:

| quantity | value |
|---|---|
| eligible molecules (after Lipinski/Veber/PAINS) | 15 of 17 |
| subset at `--diverse 6` | **6 (40 %)** |
| worst pair inside the subset | 0.222 |
| scaffold coverage of the eligible set | **4 of 4 = 100 %** |
| comparison cost | 70, in 0.8 ms |
| campaign wall time | 46.4 s for 6 molecules (exhaustiveness 4, 6 jobs) |

The subset file is only rewritten when the selection changes, because its size
and mtime ride in the campaign's library hash: regenerating an identical subset
would change the hash and break the resume of a campaign that is already half
done. That behaviour is pinned by a test
(`tests/test_prepare_cli.py::test_screen_diverse_replaces_the_library_with_the_subset`).

## 7. What these methods do not establish

Read this before quoting any number above.

* **2-D similarity is not activity.** Two molecules at Tanimoto 0.85 can have
  different activities and two at 0.4 can have the same one; similarity is a
  statement about shared substructures and nothing else. No threshold in this
  document was validated against an assay — the 0.7 default is the published
  convention, not a measurement of this library. Similarity also ignores
  stereochemistry unless `--chirality` is asked for, ignores protonation and
  tautomers, and is blind to what the receptor does with the molecule.
* **A Murcko scaffold is a heuristic.** It strips terminal substituents, so a
  substituent series collapses onto its parent ring (the eleven benzenes above);
  it says nothing about whether the ring is *made* the same way, and it is
  computed per molecule with no reference to the library. `--generic` erases atom
  types for a shape-only view.
* **An R-group decomposition is only as good as its core.** A core that is not
  shared leaves molecules unmatched (reported, with the reason); a symmetric core
  can be numbered in more than one equivalent way (the global alignment makes the
  *set* consistent, not the position chemically canonical); a fused-ring analogue
  cannot be expressed at all and is reported as unmatched.
* **A matched pair is not a measured substituent effect.** It needs the affinity
  to be measured — or docked under one consistent protocol — and it needs the
  delta to clear the noise. This project measured a **3.17 Å** seed-to-seed pose
  swing on 1M17 and warns against ordering hits within **~1 kcal/mol**; every
  delta in the worked series is smaller than that, and the table says so per row.
  Docked affinities are also not free energies, and an assay's IC50 has its own
  error. A pair that changes the substituent size by many atoms changes several
  properties at once and is reported with the heavy-atom count for that reason.
* **Diversity selection is a heuristic with a knobs problem.** MaxMin depends on
  its first pick (reported), sphere exclusion on library order, and both on the
  fingerprint and cut-off chosen; "6 of 17 covers 4 of 6 scaffolds" is a statement
  about Murcko scaffolds as grouping devices, not a guarantee that docking those
  six finds everything the library contains. A diverse subset can miss a
  congeneric series' best member by construction.
* **Multi-conformer similarity is conformer- and seed-dependent, and expensive.**
  The conformers are invented by ETKDGv3, the pharmacophore model counts
  features and distances (not atoms), and the demo shows a methyl analogue
  scoring 1.000 against its parent — a 3-D score is not automatically a better
  answer than a 2-D one.
* **Nothing here is a library-design method.** These functions describe a library
  that exists; they do not predict which molecule to synthesise, whether a
  scaffold is patentable, or whether an analogue is synthetically accessible.

## Reproducing the numbers

```bash
# fingerprints, similarity and analogues
odock similar -q 'N=C(N)c1ccccc1' -i demo/libraries/library.smi --cutoff 0.5
odock similar -q 'N=C(N)c1ccccc1' -i demo/libraries/library.smi --3d --conformers 8 --cutoff 0.0

# diversity, with the scaffold coverage of the subset
odock diverse -i demo/libraries/library.smi -n 6 -o out/diverse.sdf --json-out out/diverse.json
odock diverse -i demo/libraries/library.smi -n 6 --method sphere --cutoff 0.3

# scaffolds, series and the whole chemistry page
odock scaffolds -i demo/libraries/library.smi
odock scaffolds -i demo/libraries/library.smi --affinities out/3ptb-screen/results.jsonl --report

# the R-group matrix and the matched pairs
odock rgroups -i demo/libraries/library.smi --core 'N=C(N)c1ccccc1' \
    --affinities out/3ptb-screen/results.jsonl --mmp -o out/rgroups.xlsx

# the affinities the deltas above use (≈ 80 s for the 15 survivors)
odock screen -r demo/systems/3ptb/receptor.pdbqt -i demo/libraries/library.smi --box demo/systems/3ptb/box.json \
    -o out/3ptb-screen -e 8 --seed 42 --jobs 8 --no-poses
```

The Python API is the same thing:

```python
from odock import ligandsim, scaffold
from odock.chem.ligand import read_ligands

library = read_ligands("demo/libraries/library.smi", embed=False)
fingerprints = ligandsim.fingerprint_set(library, smiles=True)

print(ligandsim.find_analogues("N=C(N)c1ccccc1", fingerprints, cutoff=0.5).table())
print(ligandsim.diversity_subset(fingerprints, 6).as_dict())
print(scaffold.scaffold_groups(library))
print(scaffold.report_section(library, affinities={"benzamidine": -5.9}))
```

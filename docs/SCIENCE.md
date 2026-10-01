# OpenDocking: the science layer

Everything in this document is about what a **modeller can conclude from a
finished run**. The kernel answers "what is the energy of this pose"; the science
layer answers the questions that decide whether the pose is worth following up:

* do the three force fields even agree about the ranking,
* is this hit efficient, or is it just big and greasy,
* is the pose strained, i.e. did the ligand have to fold into an unlikely shape
  to make these contacts,
* which contacts survive across the best poses — the reproducible part of the
  binding mode,
* and would a second seed have found the same thing.

The code lives in `python/odock/consensus.py`, `python/odock/metrics.py`,
`python/odock/analysis.py` (interaction fingerprints) and `python/odock/report.py`.
Everything is plain functions and dataclasses: no Qt, no plotting library, no
RDKit at import time, so the CLI, the workbench and a notebook can all call it.

**See also.** [`SCORING.md`](SCORING.md) derives the physics this layer measures;
[`USER_GUIDE.md`](USER_GUIDE.md) documents the command line and the workbench
panels that display these numbers — the run dashboard, the pose comparison view,
the command palette, session persistence and the themes. This document is about
the science, not the interface.

```python
import odock
from odock import consensus, metrics, analysis, report

result = odock.dock("receptor.pdbqt", "ligand.pdbqt", box, seed=42)

# 1. Do the force fields agree?
combined = consensus.consensus_score(result, "receptor.pdbqt", box)
print(combined.table())
print("mean Spearman rho:", round(combined.agreement, 3))

# 2. Is the hit efficient?
bundle = metrics.efficiency_metrics(result.best().affinity, mol=prepared_ligand)
print(bundle.as_dict())

# 3. Which contacts reproduce?
ligands = [odock.pose_to_mol(p, prepared_ligand, report.atom_order) for p in result.poses]
fingerprints = analysis.pose_fingerprints("receptor.pdbqt", ligands)
print(fingerprints.pharmacophore(top=3).table())

# 4. Would another seed have found it?  (run the same system with several seeds)
print(consensus.reproducibility([run_a, run_b, run_c], cutoff=2.0).table())

# 5. Write it down
summary = report.notebook_summary(result, consensus=combined, fingerprints=fingerprints)
```

---

## 1. Consensus scoring: rank aggregation, not an average of energies

The kernel implements three force fields — Vina, Vinardo and AutoDock 4 — and
they are on different scales. For the same pose on the same system Vina reports
about −6.2 kcal/mol and AD4 about −6.5, and the *spread between poses* differs
too. Averaging raw energies would let whichever field has the widest spread
dominate, and a constant offset between fields would be meaningless anyway.

`consensus_score(poses, receptor, box, scorings=("vina", "vinardo", "ad4"))`
therefore rescores every pose with each force field **at the coordinates the run
produced** (no search, no refinement — `refine=False`), and combines the results
by rank:

| method | what it is | scale |
|---|---|---|
| `"rank"` (default) | the mean of each pose's rank within a field, normalised to `[0, 1]` by `(rank − 1)/(n − 1)` | 0 = best under every field, 1 = worst under every field |
| `"borda"` | the same without normalisation — the classical rank sum | `[1, n]` |
| `"z"` | the mean of `(x − mean)/sd` within each field, population sd | standard deviations |

All three are lower-is-better. Ties share their average rank, which is what
stops the aggregation inventing an ordering between two poses a force field
scores identically. A pose whose score is not finite in a field is given the
**worst** badness in that field and named in `ConsensusPose.missing`: it is never
dropped silently, and it is never rewarded for a missing number.

### Reading the agreement

`ConsensusResult.correlations` carries the Spearman rho between every pair of
fields, `matrix()` gives the `(k, k)` matrix, and `agreement` is the mean of the
off-diagonal. Spearman — not Pearson — because the question is whether the two
fields agree about the **ordering**, which is exactly what a rank correlation
answers, and because it is robust to the non-linear relationship between the
potentials. Ties use average ranks, which is the standard correction. An
undefined correlation (fewer than two usable poses, or a constant field) is
reported as `nan`, not as zero.

A worked example, from the bundled demo (measured, not illustrative):

| system | poses | rho(vina,vinardo) | rho(vina,ad4) | rho(vinardo,ad4) | mean |
|---|---|---|---|---|---|
| 3PTB, benzamidine, 9 heavy atoms | 6 | 0.943 | 1.000 | 0.943 | **0.962** |
| 1M17, erlotinib, 29 heavy atoms, 11 rotors | 9 | 0.917 | 0.500 | 0.633 | **0.683** |

The rigid ligand is reproduced by all three fields; the flexible one is not. The
mean rho of **0.683** — driven by `rho(vina, ad4) = 0.500` — is the quantitative
form of the caveat the README already states about erlotinib: the search finds
the experimental pose (it is the pose at 1.43 Å, ranked 4th), and the empirical
force fields, not the engine, are what fail to rank it first. A user reading only
`result.table()` sees nine numbers between −6.71 and −6.94 and no warning; a user
reading the consensus table sees that two of the three fields rank the set almost
independently and that the ranking is therefore soft.

On the same system, Vina and AD4 both put pose 1 first while Vinardo prefers
pose 2 — by 0.03 kcal/mol, a difference far below any force field's accuracy,
which is why the consensus reports ranks rather than pretending to resolve it.

### What it does *not* do

Rescoring is exact pairwise evaluation of the *same* coordinates. It cannot
rescue a pose the search never found, it does not re-relax the pose under the new
potential (an AD4-optimised geometry is not Vina's minimum), and it inherits
every force field's systematic error. It is a robustness check on a ranking, not
a source of new physics.

### Why AD4 is usually the outlier (measured)

On the bundled benchmark AD4 is a party to the weakest pair on **every** system,
and its scale is the reason: on the same nine erlotinib poses AD4 spans
**3.19 kcal/mol** where Vina spans **0.23**. Zeroing *every* receptor charge —
which removes AD4's screened electrostatics and its charge-dependent desolvation
entirely — moves AD4's ranking on 4 of those 9 poses but changes its correlation
with Vina by **0.02** (+0.50 → +0.48); and the EGFR receptor has none of the
zeroed non-finite charges that affect 3PTB, so the preparation defect cannot
explain it either. The disagreement is the parameterisation and the scale, not a
bug — see
[`BENCHMARK.md`](BENCHMARK.md#why-ad4-disagrees-with-vina-measured-not-asserted).
It is also the reason this module aggregates **ranks**: averaging AD4's energies
with Vina's would let the wider scale win.

---

## 2. Ligand-efficiency metrics

With `delta_g` in kcal/mol and `p = -delta_g / 1.37`:

```
LE   = -delta_g / HAC          ligand efficiency, kcal/mol per heavy atom
LLE  = p - logP                lipophilic ligand efficiency
BEI  = 1000 * p / MW           binding efficiency index
SEI  = 100 * p / TPSA          surface efficiency index
```

`1.37` is the conventional kcal/mol per log unit (`RT ln 10 = 1.364` at 298 K);
`metrics.KCAL_PER_LOG` exposes it. LogP is RDKit's Crippen `MolLogP`, TPSA is
Ertl's fragment sum, MW is the average mass — the standard models, named here
rather than hidden.

Note that LLE is **not** `1.37 * LE - logP`: dividing by the heavy-atom count and
subtracting logP gives a much smaller number that is not the published quantity.

Measured on the bundled erlotinib pose (affinity −6.939 kcal/mol, 29 heavy atoms,
LogP 2.11, TPSA 73.0 Å², 11 rotors):

```
heavy_atoms 29   LE 0.239   LLE 2.96   BEI 12.19   SEI 6.94
entropy penalty 7.16 kcal/mol
```

**Undefined inputs return `float("nan")`, never an exception and never a silent
zero.** Zero heavy atoms, a non-finite affinity, no LogP for LLE, zero TPSA for
SEI: all `nan`, so a screening loop can put a hundred thousand ligands through
and simply blank the column. Type errors (a string where a number belongs) still
raise. `LigandMetrics.ok` is `True` only when every metric that is present is
finite — which is deliberately `False` for benzene, whose SEI genuinely does not
exist.

### The entropy term is an estimate

`entropy_penalty(n)` = `n · R · T · ln(3)` = **0.651 kcal/mol per rotatable bond**
at 298.15 K. That is the crudest defensible model: every rotor independent, every
one threefold, no residual entropy in the bound state. It is exposed so it can be
*compared* with the kernel's own torsional term (Vina divides by
`1 + 0.05846·N_tors`; AD4 adds `0.2983·N_tors`), not substituted for it. Both are
reported, and neither is presented as a measurement.

Two measured properties of it are worth knowing before quoting the number:

* **It is a constant offset inside one run, so it cannot change a pose ranking.**
  Every pose of a ligand shares the same `N_tors`, so the penalty is identical for
  every row of a run (the report writes the same value in the whole column). It
  only becomes a *comparison* when two ligands with different torsion counts are
  put side by side — which is what a screening table does, and what a pose table
  does not. An identically-filled entropy column is therefore correct, not broken.
* **It is sensitive to the constant.** For 11 rotors: 4.52 kcal/mol with two
  states per rotor, 7.16 with three (the default), 11.68 with six — a factor of
  2.6 across the plausible range, against a spread between erlotinib poses of
  0.23 kcal/mol. The value is a model assumption, not a measurement, and
  `states_per_rotor=` is exposed so the assumption can be varied rather than
  hidden.

---

## 3. Ligand strain

`metrics.ligand_strain(pose_pdbqt, receptor=..., box=..., scoring="vina")` reports
two numbers, and they answer different questions.

**`strain` — the kernel-consistent one.** The kernel's own `intra` term for the
pose minus the same term for the *same topology* after a force-field relaxation:

```
strain = intra(pose) - intra(relaxed)      # both from `scoring`'s potential
```

Because both endpoints come from one potential, the difference is on the docking
energy scale and can be added to an affinity. That is the correction Vina's own
convention leaves out: for Vina and Vinardo the kernel cancels the ligand's
internal energy against the unbound reference **at the same conformation**, which
implicitly assumes the free ligand already sits in the bound geometry. A pose
that reaches its contacts by folding into a strained conformation is therefore
over-rewarded, and this is the first-order fix.

**`force_field_strain` — the medicinal-chemistry one.** The plain MMFF94 strain:
the MMFF94 energy of the pose geometry minus that of the locally minimised
conformer of the same molecule, via `odock.chem.ligand.minimize` (which owns the
documented MMFF94 → MMFF94s → UFF fallback). This is the number a chemist means
by "strain", but it is on MMFF94's scale, so it is **not** added to an affinity.

Honest statements about both:

* **The relaxation is local.** It starts from the pose geometry and minimises, so
  the result is a strain relative to the *nearest* conformer basin, not to the
  global minimum. A pose in a high-energy basin is not penalised for being in the
  wrong basin — that is what a conformer search would be for.
* **The kernel strain can be slightly negative** (measured: −0.07 kcal/mol on one
  erlotinib pose). The relaxation is driven by a different force field, so the
  geometry it prefers can be marginally worse in the docking potential. The
  number is reported as measured; it is not clamped to zero.
* **The kernel's intra term is small** for the united-atom potentials, because
  Vina excludes hydrogens from the intra sum: measured −0.04 kcal/mol for
  benzamidine, and −0.07…+2.49 kcal/mol across nine erlotinib poses. A small
  correction is a small correction.
* **A PDBQT carries no bond orders.** A molecule perceived from a PDBQT loses its
  aromaticity, so the MMFF94 strain of a PDBQT-perceived ligand can be dominated
  by mis-perceived chemistry. This was measured on the bundled benzamidine: the
  MMFF94 strain is **68.6 kcal/mol** when the molecule is perceived from the pose
  PDBQT, **60.8 kcal/mol** when the *prepared* ligand is supplied as `mol=`
  (because that ligand was itself prepared from a PDB without a SMILES template
  and has no bond orders either), and **2.5 kcal/mol** for the same molecule
  prepared from SMILES with correct aromaticity. The intended ligand's strain is
  the last one; the first two are artefacts.
* **So `mol=` is necessary but not sufficient, and `reliable` is decided from
  evidence rather than optimism.** `StrainResult.reliable` is `True` only when a
  molecule was supplied **and** either
  1. a `PreparationReport` attests that its bond orders came from the input
     (`report.chemistry_trusted`), or
  2. with no report, nothing looks wrong: the pose PDBQT does not call an atom
     aromatic while the molecule has no aromatic atom, and the molecule is not an
     all-single-bond ring system — the shape a PDB-perceived molecule takes, and
     the one case that cannot be told apart from a genuinely saturated ligand by
     looking at the molecule alone.

  Every failed check puts the reason in `StrainResult.note`, and
  `require_reliable=True` turns the flag into a `ValueError` for a caller who
  would rather fail than read a number that may be two orders of magnitude out.
  The documented recipe is therefore the only path that *guarantees* a reliable
  force-field strain:

  ```python
  mol, pdbqt, prep = odock.prepare_ligand("ligand.sdf")        # or with smiles=...
  strain = odock.metrics.ligand_strain(
      pose_pdbqt, receptor=receptor, box=box,
      mol=mol, atom_order=prep.atom_order, preparation=prep,   # one call
      require_reliable=True,
  )
  assert strain.reliable
  ```
* **The supplied molecule must be in the PDBQT's atom order.** The relaxed
  coordinates are patched back onto the pose document by file order, so a
  molecule prepared from SMILES (whose atoms are ordered by the SMILES) would
  otherwise have its geometry written onto the wrong atoms and still return a
  plausible number. `atom_order=report.atom_order` renumbers it for you, and a
  mismatch that survives that raises `ValueError`, naming the expected and the
  actual element sequences.
* `metrics.strain_corrected_ranking(affinities, strains)` re-ranks by
  `affinity + strain` and keeps both ranks, so the effect of the correction is
  visible rather than silently applied. It takes whatever strains the caller
  computed, so it inherits their reliability — pass the kernel `strain`, not an
  unreliably perceived MMFF94 one.

---

## 4. Interaction fingerprints

`profile_interactions` gives a list of contacts per pose. A **fingerprint** turns
that into a fixed-length vector over `(residue, interaction-type)` pairs, which is
what makes two poses comparable:

```python
fingerprints = analysis.pose_fingerprints(receptor, ligands)   # one per pose
fingerprints.labels()            # the schema, in column order
fingerprints.matrix              # (poses, features) counts
fingerprints.similarity()        # (poses, poses) Tanimoto over the bits
fingerprints.pharmacophore(top=3, min_frequency=0.5)
```

* The feature is `(residue, type)`, not residue: two poses that both touch ASP189,
  one by a salt bridge and the other by a hydrophobic contact, are **not** the same
  binding mode.
* Counts, not only bits, because three hydrogen bonds to one residue are not the
  same as one. `similarity(metric="tanimoto"|"dice"|"cosine")` — Tanimoto and Dice
  over the bits, cosine over the counts.
* Two poses that make no contact at all have an **undefined** Tanimoto (`nan`),
  not 1.0. The diagonal of the similarity matrix is still exactly 1.0.
* Clashes are never features: a clash is a defect, not a contact, and counting one
  would make two poses look similar because both are bad.
* **Schema.** By default the union of the features the poses present is used. Use
  `FingerprintSchema.from_receptor(receptor)` for a vector space fixed by the
  receptor instead, which is what lets fingerprints from different ligands — or
  from a second run — be compared.
* **Water-mediated contacts.** `water_mediated_contacts(receptor, ligand)` finds a
  water oxygen within `cutoff` (3.5 Å) of a polar ligand atom *and* of a polar
  receptor atom. No angle is tested and the water's hydrogens are not required,
  because a crystallographic water is usually an oxygen only — the two-leg
  distance criterion is the standard first-pass definition and it is stated here
  rather than implied. Waters in the receptor are never reported as a residue:
  a water is not a pharmacophore feature, so `pose_fingerprints` removes direct
  water contacts and adds the bridge instead.

Measured on the demo data:

| system | result |
|---|---|
| 3PTB, 6 poses | poses 1 and 2 (0.05 Å apart) have Tanimoto **1.000** — the same binding mode; the contact recurring across all six is `VAL213:hydrophobic` (5/6). With the 62 crystallographic waters in the receptor, `VAL227:O…HOH416:O…BEN1:N1` is found at 2.84 Å. |
| 1M17, 9 poses | poses 3/4 and 8/9 are identical modes; across the **top three consensus poses** the recurring features are `LEU694`, `LYS721`, `LEU764`, `THR766` hydrophobic contacts plus **`MET769:hbond`** — the known EGFR hinge contact. A method that finds the hinge from the docked poses alone is doing something right. |

The 3PTB receptor, 1M17 receptor and other bundled files accept PDBQT text or a
path directly; `_structure` parses it, so a CLI does not have to pre-parse.

---

## 5. Reproducibility across independent runs

One run cannot tell you whether the top pose is a property of the system or of
the seed. `consensus.reproducibility(results, cutoff=2.0)` pools the poses of
several runs (the same system, different seeds), clusters them with the existing
symmetry-aware RMSD clustering, and reports:

```
runs                3
poses               6
cluster cutoff      2.00 A
clusters            2 (sizes 4, 2)
top cluster         4 poses from runs 1, 2
reproducibility     0.67 (2/3 runs found the top mode)
```

`reproducibility` is `number of runs contributing a pose to the winning cluster /
number of runs`. `1.0` means every seed converged to the same binding mode;
`1/9` means the top pose is a fluke of one seed. With fewer than two usable runs
it is `nan` — one run cannot be reproduced — and a run whose poses carry no
coordinates is reported in `notes` rather than silently counted.

---

## 6. Reporting a run

`report.result_rows` keeps the historical five columns in place and adds the
efficiency columns (always) plus the consensus and strain columns (opt-in):

```
Mode | Binding Energy | RMSD l.b. | RMSD u.b. | Key residues |
Heavy atoms | LE | LogP | LLE | BEI | SEI | Entropy penalty | Strain |
Consensus rank | Consensus score | <field> | <field> rank ...
```

* `write_csv` / `write_jsonl` / `write_xlsx`. JSONL normalises non-finite numbers
  to `null`, because `NaN` is not valid JSON and a reader that fails on the 900th
  ligand of a screen is worse than useless.
* `write_xlsx(fingerprints=...)` adds a `Fingerprints` sheet (pose × feature
  counts) and a `Pharmacophore` sheet (the recurring contacts, most frequent
  first).
* `ResultStream` is an append-only CSV/JSONL writer that never holds the table in
  memory, for a screen too large to build as one list. A CSV stream fixes its
  columns from the first row; a later unknown key is dropped with a warning
  instead of silently corrupting the file.
* `notebook_summary` is the paste-ready block: the run's parameters, the best
  pose with its efficiency numbers, the consensus ranking with its Spearman rho,
  the recurring pharmacophore, and every approximation named next to it.
* `decomposition_table_for` prints the per-pose `inter / intra / unbound /
  torsion` decomposition, exposing the identity the kernel documents
  (`base = inter + intra − unbound`, `affinity = base + torsion`).

The workbench shows the same numbers through its own panels — the run dashboard,
the pose comparison view, the command palette, session persistence and the
themes. Those are user-interface concerns and are documented in
[`USER_GUIDE.md`](USER_GUIDE.md); nothing in this module imports Qt, so the GUI
is a consumer of this API rather than a part of it.

---

## 7. Defects found while building this, and what was done

These are recorded rather than quietly fixed, because a published 0.1.0 shipped
them.

**Non-finite Gasteiger charges reached the PDBQT file.** The demo receptor
`demo/systems/3ptb/receptor.pdbqt` as shipped carried **128 non-finite Gasteiger
charges**: 8 rendered as the literal string `inf`, and the other 120 rendered as
`0.000` because the guard only tested `isnan` and the value was `NaN`. RDKit's
charge model does not converge for a fraction of a protein's atoms. The
consequence was silent and pose-dependent: AD4 electrostatics returned a `NaN`
*affinity* for any pose with a pair touching one of those atoms (measured: demo
pose 3 scored `nan` under AD4 while the other five were finite), while Vina and
Vinardo were unaffected because they use an 8 Å cutoff and no electrostatics.
Fixed in `pdbqt.gasteiger_charges`: a non-finite charge is written as `0.000`
**and reported** through a `UserWarning` that names the count and the first
atoms, so the lost electrostatic term is visible. `demo/systems/3ptb/receptor.pdbqt` was
regenerated through the same `prepare_receptor` path: 8 lines change, no
non-finite charge remains, and AD4 scores every demo pose.

**A prepared ligand could be split across two residues.** `prepare_ligand` added
polar hydrogens with `Chem.AddHs`, which leaves them without PDB residue
information; they were written as `LIG A 1` while the heavy atoms kept the
deposited residue (`BEN`). RDKit's proximity bonding does not connect atoms
across a change of residue name, so reading such a ligand PDBQT back lost
**every X–H bond** and MMFF94 could not type the molecule at all — which is why
`ligand_strain` used to fall back to UFF on the demo ligand. Fixed: the ligand
path now inherits the parent residue onto the added hydrogens (with the writers'
unique `H<serial>` naming kept, so two hydrogens on one nitrogen are not both
called `N1`). A round trip of the prepared benzamidine now gives 13 atoms, 13
bonds and 4 X–H bonds. `odock.metrics` still unifies residues when it reads a
document, because PDBQT files written before this fix — including other files in
`demo/` — are still in circulation.

**`examples/make_demo.py` crashed on a non-UTF-8 Windows console.** It prints the
search box, which contains an Ångström sign, and a GBK or cp1252 console raised
`UnicodeEncodeError` before the run started. Fixed by reconfiguring stdout and
stderr to UTF-8 (what `odock`'s own CLI already does).

**The demo directory mixes two runs.** `demo/systems/3ptb/poses.pdbqt` (6 models,
−6.210 … −4.414) comes from the CLI walkthrough at `exhaustiveness = 32`, as
`config.txt` records, while `demo/systems/3ptb/result.json` (4 modes, −6.212 … −4.513)
and `demo/README.md` come from `python examples/make_demo.py --fast`
(`exhaustiveness = 4`). Both are reproducible — re-running `make_demo.py --fast`
reproduces `result.json`, `README.md`, `receptor.pdbqt` and `ligand.pdbqt` exactly
— but they describe different runs, so the `poses.pdbqt` in the tree is not the
file `result.json` was computed from.

---

## 8. Limitations, stated plainly

* **The consensus is a robustness check, not new physics.** It cannot fix a
  force field's systematic error; it can only show it.
* **Strain is basin-local.** Both strain numbers compare the pose with a *local*
  minimum of the same molecule. A pose in the wrong basin is not penalised for
  that.
* **The entropy term is a model, not a measurement** (0.651 kcal/mol per rotor),
  and it ignores the residual entropy of the bound state and any correlation
  between rotors. It is constant within a run, so it cannot change a pose ranking
  — it is a cross-ligand comparison, and its value moves by a factor of 2.6
  between two and six states per rotor.
* **The MMFF94 strain of a PDBQT-perceived molecule is unreliable** whenever the
  file carries no bond orders, and a molecule prepared from a PDB *without* a
  SMILES template is in the same position — measured at 68.6 and 60.8 kcal/mol
  against a true 2.5. The result carries `reliable=False` and the `note` says
  where the chemistry came from; `require_reliable=True` makes it an error, and
  the documented recipe (`mol=`, `atom_order=report.atom_order`,
  `preparation=report`) is the only path that sets `reliable=True` by
  attestation rather than by absence of suspicion. The kernel `strain` (both
  endpoints in one potential) is the robust number, and it is the one the report
  prints.
* **Water bridges use a distance-only criterion** — no angle, no bridging
  geometry, no occupancy. It finds candidates, not proof.
* **`reproducibility` measures convergence, not correctness.** Nine seeds can
  agree on a pose that is wrong; it says the answer is stable, not that it is
  right.
* **LogP/LLE/BEI/SEI are blank unless the caller supplies the ligand molecule.**
  They are not guessed from a PDBQT, because a PDBQT carries no bond orders and a
  guessed LogP would be worse than an empty cell.
* **Ligand efficiency uses the prepared ligand's heavy-atom count** (united-atom
  hydrogens merged), which is the usual convention but not the only one; the
  column header says "Heavy atoms" so the number is auditable.

# The OpenDocking multi-target panel

`python -m odock.panel` docks a set of ligands into a set of receptors under one
shared protocol and reports the **affinity matrix**, each ligand's **selectivity
profile** — best target, the gap to the second best, and how many receptors score
within a stated window of the best — and the **box that produced every column**,
because none of those numbers means anything without it.

It answers the question a screening campaign asks next. `odock screen` says which
of my molecules fit this protein; a panel asks which of these proteins my molecule
fits, and whether the answer is a preference or an artefact.

**A panel of crystal structures is not a selectivity assay.** The receptors are
rigid, each is in one conformation, their pockets are not the same size, and one
force field scores all of them. What a panel can do is show that a ranking is
plausible, flag a ligand that scores equally well everywhere, and quantify how
much of a "preference" survives a change of seed. That is what the measurement in
this document is for.

```bash
# the bundled panel: seven crystal structures, seven co-crystallised ligands
python -m odock.panel --out runs/panel --json runs/panel.json --csv runs/panel.csv

# a subset, a tighter window, and the hit rate only (the bundled ligands)
python -m odock.panel --out runs/quick --receptors 3PTB,1STP --window 0.5

# your own library: 'SMILES name' lines, or an SDF
python -m odock.panel --out runs/mine --ligands my_hits.smi --receptors 3PTB,1M17
```

The Python API is the same code path:

```python
from odock.panel import bundled_panel, bundled_ligands, profile, PanelProtocol

panel = bundled_panel(protocol=PanelProtocol(exhaustiveness=1, seeds=(42,)))
ligands = {name: spec.smiles for name, spec in bundled_ligands().items()}
result = profile(ligands, panel, "runs/panel",
                 homes={name: spec.home for name, spec in bundled_ligands().items()})

print(result.matrix())      # ligands x receptors, kcal/mol
print(result.table())       # the matrix plus each ligand's selectivity profile
print(result.box_table())   # the caveat, as numbers
print(result.hit_rate())    # how many ligands prefer their own protein
```

## 1. The bundled panel

Seven structures, all already in `tests/data`, so a panel run needs no network.
Each member's box comes from **its own co-crystallised ligand** plus one shared
buffer (8 Å) — the `reference` box policy.

| receptor | protein | family | reference ligand | query ligand |
|---|---|---|---|---|
| 3PTB | bovine trypsin | serine protease | BEN (benzamidine) | BEN |
| 1STP | streptavidin | biotin-binding | BTN (biotin) | BTN |
| 3ERT | human ERα ligand-binding domain | nuclear receptor | OHT (4-hydroxytamoxifen) | OHT |
| 1ERE_A | human ERα ligand-binding domain | nuclear receptor | EST (estradiol) | EST |
| 1M17 | EGFR kinase domain | protein kinase | AQ4 (erlotinib) | AQ4 |
| 1HVR | HIV-1 protease | aspartic protease | XK2 (XK263) | XK2 |
| 1HXW | HIV-1 protease | aspartic protease | RIT (ritonavir) | RIT |

Two families appear twice — the ERα pair (3ERT, 1ERE_A) and the HIV-1 protease
pair (1HVR, 1HXW). That is deliberate: a ligand that binds one member of a family
and not the other is a different finding from a ligand that binds neither, and a
single structure per family cannot tell them apart.

`tests/data/2PTN.pdb` is a trypsin structure with **no** co-crystallised ligand.
It is a legitimate receptor but not a member of this panel under the `reference`
policy, because there is no ligand to derive a comparable box from — and the
module refuses it by name rather than inventing a box (see §6).

The query ligands are the same molecules as the reference ligands, which is what
makes the hit rate checkable by hand: biotin should prefer streptavidin,
erlotinib EGFR, benzamidine trypsin, XK263 and ritonavir HIV-1 protease,
estradiol and tamoxifen the ERα pair.

### The protocol, and the mistake it exists to prevent

Every member's receptor is prepared with **its own co-crystallised ligand
stripped** (`prepare_receptor(..., strip=[residue])`). That sounds obvious and it
is not: the ligand defines the box, so it is tempting to leave it in place — and
leaving it in place blocks the very site the query ligands are docked into. The
first run of this module did exactly that, and the matrix looked like a
selectivity result: a 50-atom ligand scored **+17.7 kcal/mol** "into" the
estradiol site of 1ERE_A, large ligands clashed everywhere, and the hit rate was
1/7 for a reason that had nothing to do with binding. The table below is the
corrected run.
`tests/test_panel.py::test_the_prepared_receptor_does_not_contain_its_own_crystal_ligand`
asserts the strip for all seven members, so the mistake cannot come back quietly.

The rest of the protocol is one setting for every cell: Vina, exhaustiveness 1,
nine poses, energy range 3 kcal/mol, **no per-pose refinement** (`refine=False` —
a panel is a screening comparison, and the BFGS pass costs more time than the
ranking it would change), and one seed unless `--seeds` says otherwise. Waters
are dropped; ions, cofactors and modified residues are kept, because they are
part of the site.

<!-- BOX-TABLE -->

## 2. The matrix

<!-- MATRIX -->

## 3. The selectivity profile and the hit rate

<!-- HIT-RATE -->

## 4. Uncertainty: what one seed is worth

<!-- SPREAD -->

## 5. Reading a panel honestly

Three numbers decide whether a profile means anything, and the module reports all
three beside the affinity:

* **the box table.** A receptor whose box holds 500 receptor atoms and 12 000 Å³
  is not competing on the same terms as one with 350 atoms and 7 300 Å³. When a
  ligand's best target has a materially larger box than its own protein, that is
  the first explanation to check, and `hit_rate()` labels it `box`.
* **the window count.** `<=` in the table is how many receptors are within the
  window of the best. One is a preference; three is a ligand that does not
  distinguish them, whatever the ranking says.
* **the three-field agreement** (`agreement` in the JSON and the CSV). Each cell
  is rescored with Vina, Vinardo and AD4, and the number is their mean pairwise
  rank correlation on that cell's own poses. Below 0.5 the cell's *internal*
  ordering is already unstable, so a 0.2 kcal/mol difference between two
  receptors is below the noise. It is not reported below three poses, where a
  correlation is +1 or −1 by construction.

And one caution that is not a column: **a within-window count is a heuristic.**
The window is a constant (1 kcal/mol by default), it is applied to means over
whatever seeds were run, and it says nothing about whether the difference is real
— only that it is smaller than the tool's resolution. It is a way to stop a
ranking being over-read, not a statistical test.

## 6. Refusals: a mixed panel is not a panel

A cross-target comparison is only meaningful under one protocol and one box
derivation rule, so the panel refuses the alternatives and names the field:

```text
PanelError: this panel mixes box policies (field: box_policy): the protocol derives
every box from its member's reference ligand ('reference'), but 1M17 carries an
explicit box. Give every member a box or none — a comparison between a derived and
a hand-placed box is not a comparison.
```

```text
PanelError: 2PTN has no co-crystallised reference ligand, so the 'reference' box
policy cannot place its box (field: ligand). Either drop 2PTN from the panel, or
build it with box_policy='explicit' and give every member its own box.
```

```text
PanelError: the panel mixes box buffers (field: box_buffer): 1STP declares 4.0 Å
but the shared protocol says 8.0 Å. A cross-target comparison needs one buffer
everywhere.
```

The same refusals exist for a duplicate receptor (`pdb_id`), an empty panel
(`members`), a non-positive window (`window`), an unknown policy (`box_policy`),
and the explicit policy missing a box for a member (`box`). Every one is a
`PanelError` — a `ValueError` — so a caller can catch it, and every message names
the field to change.

## 7. API

| object | what it is |
|---|---|
| `PanelMember` | one receptor: its `pdb_id`, the reference ligand residue, protein and family, an optional explicit `box` |
| `PanelProtocol` | the settings that must not vary: box policy and buffer, force field, exhaustiveness, pose count, seeds, agreement/fingerprint switches |
| `Panel` | members + protocol + window; `validate()`, `subset()`, `replaced()` |
| `build_panel()` / `bundled_panel()` | a validated panel (all seven, or a subset) |
| `derive_boxes()` | the box for every member, plus volume, receptor atoms and grid points |
| `profile()` | run every receptor × ligand campaign; returns a `PanelResult` |
| `PanelResult` | `matrix()`, `table()`, `box_table()`, `hit_rate()`, `fingerprint_similarity()`, `as_dict()`, `write_json()`, `write_csv()` |
| `PanelProfile` | one ligand: `best`, `second`, `gap`, `within_window`, `home_rank`, `best_is_home`, `outcome()` |
| `selectivity_profile()` | the arithmetic above, callable on hand-made cells |
| `hit_rate()` | the hit rate and a classified reason for every miss |
| `add_panel_parser()` | registers `odock panel` (called from `odock.cli_ext`) |

Everything is orchestration:

* docking and the per-receptor campaigns are `odock.screen` — one
  `screen_ligands()` campaign per receptor, with the panel's box;
* the three-field agreement is `odock.consensus.consensus_score()`;
* the fingerprints are `odock.analysis.pose_fingerprints()`;
* the efficiency is `odock.metrics.efficiency_metrics()`, either via the
  `screen` record that already computed it or called again when a record carries
  an affinity but no efficiency.

Nothing in `panel.py` docks, scores, fingerprints or measures efficiency itself.

## 8. Cost

<!-- COST -->

## 9. What this does not establish

* **Seven structures are not a selectivity panel.** They are seven crystal
  structures chosen because they were already in the repository. A selectivity
  claim needs a curated, size-matched panel across a target family, run in both
  directions — this is a plausibility check on a ranking.
* **The receptors are rigid.** Every one is a single conformation with fixed side
  chains. A ligand that would reorganise the pocket is scored against a pocket
  that cannot reorganise, and the error is different for every receptor.
* **A shared protocol does not make pockets comparable.** Identical settings and
  one box rule remove *protocol* differences; they do not remove the fact that a
  larger pocket gives a buried ligand more favourable contacts. That is why the
  box table is printed with the matrix and why `hit_rate()` classifies a miss as
  `box` when the winner's box is 25 % larger.
* **A within-window count is a heuristic**, not a significance test: a fixed
  window, applied to means over whatever seeds were run.
* **One force field scores every receptor.** Vina's parameters are not tuned per
  target, and this project has measured how differently its three force fields
  order the same poses (see [`BENCHMARK.md`](BENCHMARK.md)) — AD4 is the outlier,
  which is why the agreement column exists.
* **Crystal waters and ions are included**, because they are part of the site;
  no attempt is made to model bridging waters or to compare hydration across
  receptors.
* **A weak search is not a fair ranking.** The default exhaustiveness for a panel
  is 1 — a panel is a screening tool — and a ligand whose search did not converge
  can look selective because its pose in the wrong pocket is bad. `--exhaustiveness`
  is the knob, and the profiles are only as good as the search that produced them.

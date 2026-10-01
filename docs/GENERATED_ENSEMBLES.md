# Ensembles generated from a single structure

Every measurement in [`docs/ENSEMBLE.md`](ENSEMBLE.md) and
[`docs/POCKETS.md`](POCKETS.md) needs several conformations of one receptor, and
most users have exactly one crystal structure. `odock ensemble generate` builds a
**modelled** ensemble from it by sampling the side chains of the binding-site
residues:

```bash
odock ensemble generate -r 3PTB.pdb --box-ligand BEN \
    --max-rotamers 5 --min-rmsd 0.3 --combinations 16 --seed 7 \
    --outdir generated/ --json-out generated.json

odock ensemble dock   -r generated/*.pdb --no-superpose --box-ligand BEN -l BTN.sdf
odock ensemble pockets -r generated/*.pdb --no-superpose --box box.json
```

**What it is, before anything else:** a rigid-rotamer model. The side chain keeps
its own internal geometry and only its :math:`\chi` angles change; the backbone
never moves; the rotamers come from a documented three-fold staggered grid
(:math:`-60°, 180°, 60°`) plus the native value read from the structure, not from
a statistical rotamer library. It is a way to *use* the ensemble machinery when
one structure is all there is, and it is not experimental evidence about the
protein.

## 1. The algorithm

1. **Site selection** — exactly the site definition the ensemble machinery already
   uses (`odock.ensemble.select_site`): the residues within `--site-radius`
   (default 6 Å) of the box centre, of a reference ligand (`--box-ligand`), or an
   explicit `--site` list. So a generated ensemble and an experimental pair
   describe the same binding site.
2. **:math:`\chi` assignment** — a table of the standard IUPAC/Dunbrack
   :math:`\chi` definitions per residue (`CHI_ATOMS`), with the alternative atom
   names found in the wild (`CD1`/`CD`, `OD1`/`OD`, …) and the modified residues
   mapped through the project's one-letter table. Glycine and alanine have no
   :math:`\chi`; proline's ring makes a rigid-rotamer model meaningless; all three
   are **reported**, never silently dropped.
3. **Enumeration** — every combination of three staggered values per
   :math:`\chi`, plus the native value, rotated about each bond with an exact
   Rodrigues rotation. The moving set is the atoms on the far side of the bond,
   perceived from a coordinate-distance bond graph, so the rotation cannot leak
   into the backbone.
4. **Clash filter** — a candidate is rejected when a moved heavy atom comes
   closer to a *fixed* atom than the sum of their van der Waals radii minus
   `--clash-tolerance` (default 0.4 Å). The fixed set is the whole protein except
   the residue being rotated: the other site residues stay in it at their native
   rotamers, which is what makes this a one-at-a-time model — a rotamer has to be
   clear of its neighbours too.
5. **Deduplication and capping** — clash-free candidates closer than
   `--rotamer-rmsd` (0.5 Å) to one already kept are duplicates; of the distinct
   survivors, the native rotamer plus the `--max-rotamers - 1` *most spread-out*
   ones are kept. Keeping the extremes is deliberate: the point of the ensemble is
   to span the residue's mobility, not to cluster around one value.
6. **Assembly** — one member per kept rotamer of one residue (the interpretable
   sweep), plus `--combinations N` seeded members that change several residues at
   once.
7. **Pruning** — a candidate whose site side chains are within `--min-rmsd` of an
   already-kept member is dropped, greedily, with the native member always first.

## 2. Measured on 3PTB (the trypsin S1 site)

`python examples/ensemble_validation.py --generate-only` reproduces this section.
Site: 14 residues around the benzamidine box; `--max-rotamers 5 --min-rmsd 0.3
--combinations 16 --seed 7`.

| residue | χ | enumerated | clash-free | kept | rejected by clash | duplicates | capped |
|---|---|---|---|---|---|---|---|
| ASP189 | 2 | 16 | 3 | 2 | 13 | 1 | 0 |
| SER190 | 1 | 4 | 2 | 1 | 2 | 1 | 0 |
| CYS191 | 1 | 4 | **0** | **0** | 4 | 0 | 0 |
| GLN192 | 3 | 64 | 21 | 5 | 43 | 12 | 4 |
| SER195 | 1 | 4 | 2 | 1 | 2 | 1 | 0 |
| VAL213 | 1 | 4 | 3 | 2 | 1 | 1 | 0 |
| SER214 | 1 | 4 | 2 | 1 | 2 | 1 | 0 |
| TRP215 | 2 | 16 | 4 | 2 | 12 | 2 | 0 |
| GLY216, GLY219, GLY226 | – | 0 | 0 | 0 | 0 | 0 | 0 | *(no side chain)* |
| SER217 | 1 | 4 | 3 | 2 | 1 | 1 | 0 |
| CYS220 | 1 | 4 | **0** | **0** | 4 | 0 | 0 |
| VAL227 | 1 | 4 | 3 | 2 | 1 | 1 | 0 |

The bookkeeping is exact and printed: `enumerated = clash-free + rejected` and
`clash-free = kept + duplicates + capped`, for every residue (asserted in
`tests/test_generate.py`). Two residues (**CYS191**, **CYS220**) have **no**
rotamer that clears the frozen protein at a 0.4 Å tolerance; they are reported
with that reason and left at their experimental rotamer, which is the honest
outcome for a site this tightly packed — the two sulphur atoms sit in a wall.

**The ensemble**: 11 members after pruning (native + 10). Site side-chain RMSD
**0.378–1.015 Å**, largest per-residue displacement **2.910 Å** (GLN192), and
**0.000 Å of backbone movement** by construction — asserted in the tests, because
it is the method's central limitation, not a footnote.

## 3. Is the generated ensemble the right size? (the comparison that matters)

The report prints the generated spread next to the experimental pairs this project
validates against ([`docs/ENSEMBLE.md`](ENSEMBLE.md) §1):

| pair | experimental site RMSD (CA) | experimental max displacement | generated max displacement | ratio |
|---|---|---|---|---|
| 3ERT vs 1ERE_A (ERα) | 0.444 Å | 2.010 Å | 2.910 Å | **1.45×** |
| 1HVR vs 1HXW (HIV-1 protease) | 0.407 Å | 1.589 Å | 2.910 Å | **1.83×** |
| 3PTB vs 2PTN (trypsin, control) | 0.144 Å | 1.133 Å | 2.910 Å | **2.57×** |

Read the two halves separately, because they are measured differently:

* the generated **site side-chain RMSD** (0.378–1.015 Å) *brackets* the
  experimental CA site RMSDs (0.144–0.444 Å) — the modelled members differ from
  the native about as much as the experimental structures differ from each other;
* the generated **per-residue displacement** (up to 2.910 Å) is **1.45–2.57×** the
  experimental one. The modelled ensemble is therefore *broader* in amplitude at a
  few residues. That is deliberate (the extras are chosen for maximum spread and a
  rigid rotamer has no strain energy to stop it) and it is a caveat as much as a
  feature: a real side chain would rarely sit at the far end of its grid, so the
  ensemble over-covers.

The comparison is computed from `generate.EXPERIMENTAL_REFERENCES`, so the ratio
cannot drift away from the measured constants (asserted in the tests).

## 4. What the existing analysis says about a generated ensemble

The point of generating one is to run the analysis on it. Same 11 members, the
pocket analysis in the frame they already share (`--no-superpose`), region 12 Å
around the benzamidine box:

| track | found in | volume (Å³) | free volume (Å³) | closure fraction | cryptic |
|---|---|---|---|---|---|
| 1 | 11/11 | 86–241 | 481–509 | – | no |
| 2 | 5/11 | 151–192 | 558–642 | **1.00** | no |
| 3 | 2/11 | 225 | 569–611 | **1.07** | no |

**Zero cryptic candidates** — and the reason is the one this project's vocabulary
was built for. Every transient track has a *closure fraction* of 1.00 or 1.07: the
free volume where the cavity was not found is as large as where it was, so the
differences are detector disagreements, not cavities opening or closing. Compare
the experimental ERα pair, where the same measurement gives closure fractions of
0.01, 0.06 and 0.11 on three cavities that really do close.

That is the honest limit of the method, as a measured result: **moving side chains
by up to 2.9 Å does not close a cavity in this site**, because closing the ERα
cavities needs the backbone (helix 12, 2.010 Å of it). A side-chain ensemble can
change what a search finds *inside* a site — which rotamer a ligand has to fit
around — but it cannot create or destroy the site itself.

### 4b. What the docking says: the affinity moves less than the crystallography

Benzamidine (`tests/data/BTN.sdf`) docked into all 11 members, `-e 2 -n 3 --seed
42`, the same box:

| member | best affinity (kcal/mol) | | member | best affinity (kcal/mol) |
|---|---|---|---|---|
| native | −5.945 | | gen_6 | **−5.708** (worst) |
| gen_1 | −5.884 | | gen_7 | −6.122 |
| gen_2 | −6.030 | | gen_8 | −6.012 |
| gen_3 | −5.969 | | gen_9 | −5.991 |
| gen_4 | **−6.127** (best) | | gen_10 | −6.119 |
| gen_5 | −6.058 | | | |

**The affinity spread over the generated ensemble is 0.419 kcal/mol** (−5.708 to
−6.127). Put that next to the number measured for a *rigid single structure*: the
two experimental trypsins 3PTB and 2PTN have binding sites that differ by only
0.144 Å, and they move the same ligand's affinity by **0.558 kcal/mol**
(−6.103 vs −5.545, [`docs/ENSEMBLE.md`](ENSEMBLE.md) §5). So sampling the site's
side chains around one structure produces **less** affinity variation than simply
choosing a different crystal structure of the same protein — a useful scale for a
user deciding whether a modelled ensemble is worth docking into: it perturbs the
number by about 0.4 kcal/mol, not by several.

The modes, for completeness (33 poses over 11 conformations, 363 exact
cross-receptor scores): the best pose is found in **7 of 11** conformations, but
only **3 of 11** conformations put their *own* best pose there (persistence 0.27,
robustness 0.083), and the median receptor-swap movement of that mode is
**1.467 kcal/mol**. This is a quick run (`-e 2 -n 3`), not a production one; it is
here because the question "how much does the affinity move when only the side
chains move?" is the first thing a user asks.

## 5. Cost

| quantity | value |
|---|---|
| build time for 14 residues, 11 members | **0.32 s** (0.029 s per conformation) |
| storage per member (3PTB, 1 628 atoms, PDB) | **131 KB** |
| enumeration cost for the largest residue (GLN192, 3 χ) | 64 candidates, all clash-tested |
| ditto for a 4-χ residue (LYS, ARG) | 256 candidates |

The whole cost is the clash filter, which is a NumPy distance matrix per
candidate; a 2 000-atom receptor with a 20-residue site builds in well under a
second. Preparing PDBQT files for the members (`--pdbqt`, RDKit) dominates the
wall time when it is asked for.

## 6. Backbone motion: an elastic network (and why it is not the whole answer)

Side-chain sampling cannot close a cavity (§4 measured closure fractions of
1.00/1.07 against the experimental 0.01/0.06/0.11), because closing the ERα
cavities needs **helix 12 to move 2.010 Å**. `odock ensemble modes` supplies
backbone motion from one structure: an anisotropic network model (ANM) over the
Cα atoms, whose low-frequency modes are the collective motions that
connectivity supports.

```bash
odock ensemble modes -r 3ERT.pdb --box-ligand OHT \
    --modes 3 --compare 1ERE_A.pdb --outdir mode_ensemble/ --json-out modes.json
odock ensemble pockets -r mode_ensemble/*.pdb --no-superpose --box-ligand OHT
```

**The algorithm.** Nodes are the Cα atoms of every standard residue (a helix
swings *relative* to the rest of the domain, so a site-only network cannot
produce that motion); two residues are connected when their Cα atoms are within
`--cutoff` (default **13 Å**, the standard ANM value — it is what reproduces
experimental B-factors and low-frequency motions without turning the chain into
one rigid block). The Hessian is the standard ANM form, one 3×3 block per pair;
the six rigid-body modes are removed; the remaining eigenvalues ascend.
Amplitudes follow equipartition (``sqrt(1/λ)``, so softer modes move more), and
the **whole field is then rescaled so the member's site RMSD equals its target**.
That last step is the calibration, and it is labelled as one: the defaults are the
experimental site RMSDs this project measured, so the ensemble *brackets*
experiment by construction rather than by prediction.

### Measured

| quantity | 3PTB | 3ERT (ERα) |
|---|---|---|
| Cα nodes | 223 | 246 |
| springs at 13 Å | 4 379 | 4 004 |
| non-zero modes (`3N−6`) | 663 | 732 |
| modes sampled | 3 | 3 |
| build time (Hessian + diagonalisation) | **2.0 s** | **14.5 s** |
| site RMSD achieved vs target | 0.144 → **0.144** | 0.444 → **0.444** (exact) |
| largest per-residue displacement at that site RMSD | 0.295 Å | **0.588 Å** |

The calibration is exact for every member (asserted in the tests), so the spread
brackets the experimental targets by construction. The last row is the first
honest caveat: at the *same site RMSD* as the ERα pair, the modes move the site
uniformly (0.588 Å) where experiment concentrates **2.010 Å in helix 12 alone** —
the modes distribute the motion, they do not localise it.

### The direction, which is a separate question from the amplitude

`--compare OTHER.pdb` reports the cosine overlap between each mode and the
observed Cα difference, restricted to the site as well as over the whole chain (a
flexible terminus can otherwise dominate the cosine: on this pair the four largest
whole-chain displacements are residues **544–548**, the free C-terminal tail, not
helix 12). Measured for 3ERT → 1ERE_A, 234 of 246 nodes matched:

| | overlap |
|---|---|
| the 3 sampled modes, over the 14 site nodes | **−0.002, −0.010, −0.010** |
| best of the first 30 modes, whole chain | +0.199 (mode 16) |
| best of the first 30 modes, site nodes | **+0.296** (mode 29) |

**The low-frequency modes are essentially orthogonal to the experimental motion.**
The amplitude matches (after calibration); the direction does not, at any mode
among the first 30. That is the honest verdict, and it is why the next measurement
comes out the way it does.

### The measurement that matters: does it produce the ERα closures?

The pocket analysis on a 9-member mode ensemble of 3ERT (targets 0.144/0.407/0.444,
three modes):

| | experimental 3ERT vs 1ERE_A | mode ensemble |
|---|---|---|
| cryptic candidates | **3** | **0** |
| closure fraction of the transient tracks | **0.01, 0.06, 0.11** | **0.98 – 1.33** |
| the best cavity's free volume | 410 → **4 Å³** | 397 → 417 Å³ (**±2.5 %**) |

So: **the modelled backbone does not close the cavities**, and the reason is
measured at two levels — the modes do not point along the observed motion (0.03
site overlap), and the motion they do produce leaves the free volume at every
cavity anchor unchanged (closure fractions near 1, versus 0.01–0.11). One track is
found in all 9 members with its detector volume swinging 208–542 Å³ while its free
volume moves 397–417 Å³: the detector's sub-pocket partition moving, not the
protein.

### What this establishes, and what it does not

* It establishes a **usable, deterministic backbone ensemble** at a calibrated
  amplitude, and it measures the amplitude the experimental pairs actually have
  (0.29–0.95× their maximum displacement at the same site RMSD).
* It also establishes, with numbers, that **an ANM of one structure is not a
  substitute for a second crystal structure** when the difference is a
  ligand-driven rigid-body displacement of a helix. The modes are global soft
  motions of the contact graph; helix 12's 2 Å move under a bulky antagonist is
  neither in their direction nor localised the way experiment localises it.
* An elastic network is a **harmonic approximation**: it cannot reproduce a helix
  unravelling, a loop reorganising or any anharmonic transition; it assumes the
  connectivity of the input structure; and a displacement along a mode is a
  **direction at a chosen amplitude, not a sampled state**. There is no
  temperature, no population and no free energy here.
* The Cα displacement is transferred to **all atoms of each residue rigidly**
  (standard coarse-graining): intra-residue geometry is exact, inter-residue
  geometry is not. The filter counts two things — *severe* overlaps (below 0.6 ×
  the vdW sum, relative to the input structure) reject a member, and *squeezed*
  contacts (0.75×) are reported for every member because the rigid transfer
  produces dozens of them at a 0.2 Å amplitude. Measured on 3PTB, an
  experimental-scale site RMSD of 0.41–0.44 Å costs 1–10 severe overlaps, which is
  why the default allows up to 10 and prints the count for each member.

## 7. What this does not establish

* **It is not experimental evidence.** A generated ensemble samples a rotamer
  grid; it says nothing about which conformations exist, which are populated, or
  how the protein moves.
* **The backbone never moves** (0.000 Å, asserted). Anything that requires a loop
  or a helix to rearrange — the ERα helix 12, 2.010 Å — is unreachable. Section 4
  measures that consequence instead of asserting it.
* **The grid is not a rotamer library.** Three staggered values per χ is the
  classic first approximation; a backbone-dependent library (Dunbrack) would
  sample the real preferences and is the obvious upgrade. The native value is
  always included, so the experimental rotamer is never lost.
* **No strain energy.** A rigid rotamer is either clash-free or not; there is no
  energy to say that one clear rotamer is much more costly than another, so the
  survivors are spread out by RMSD rather than weighted by likelihood. The
  ensemble therefore over-covers (section 3).
* **Residues that cannot be sampled are reported, not sampled anyway**: CYS191 and
  CYS220 here, plus every glycine. A site where most residues are in a wall will
  produce a thin ensemble, and the report says so rather than inventing rotamers.
* **Equal weights.** Every member counts once, as everywhere else in this project;
  there is no population, no Boltzmann factor, no MD.
* **One-at-a-time is not a coupled model.** The sweep changes one residue per
  member, so it does not sample correlated rotamer states; `--combinations N`
  adds seeded simultaneous changes but is still not a statistical model of the
  side-chain network.
* **The comparison in section 3 is a sanity check, not a validation.** Agreement
  in amplitude does not mean the generated members are the conformations the
  protein visits; it means they are the right *size* to exercise the machinery.

## 8. Reproducing every number

```bash
python examples/ensemble_validation.py --generate-only      # §2, §3, §4, §5
odock ensemble generate -r 3PTB.pdb --box-ligand BEN \
    --max-rotamers 5 --min-rmsd 0.3 --combinations 16 --seed 7 \
    --outdir out/generated_3ptb --json-out out/generated_3ptb.json
odock ensemble pockets -r out/generated_3ptb/*.pdb --no-superpose \
    --box demo/systems/3ptb/box.json --region-radius 12 --max-pockets 10

odock ensemble modes -r 3ERT.pdb --box-ligand OHT --modes 3 \
    --compare 1ERE_A.pdb --outdir out/mode_ensemble --json-out out/modes.json
odock ensemble pockets -r out/mode_ensemble/*.pdb --no-superpose \
    --box-ligand OHT --region-radius 14 --max-pockets 12

python -m pytest tests/test_generate.py tests/test_modes.py -q   # the same numbers
```

## 9. API

```python
from odock import ensemble, generate
from odock.prepare import box_from_points

conformation = ensemble.read_conformations(["3PTB.pdb"])[0]
box = box_from_points(ensemble.ligand_coords(conformation, "BEN"), buffer=6.0)

generated = generate.generate_ensemble(
    conformation, box=box, site_radius=6.0, max_rotamers=5, min_rmsd=0.3,
    combinations=16, seed=7,
)
print(generated.text())            # sampling, spread, comparison with experiment
print(generated.comparison())      # the ratios, computed from the constants
generated.write("generated/")      # native.pdb, gen_1.pdb, ...
# then, because everything is in one frame:
built = ensemble.build_ensemble(sorted(__import__("pathlib").Path("generated").glob("*.pdb")),
                                box=box, superpose=False)
result = ensemble.dock_ensemble("BTN.sdf", built, exhaustiveness=8, seed=42)
```

`GeneratedEnsemble` exposes `n_conformations`, `residues` (per-residue
`RotamerResidue` with `chi`, `enumerated`, `rotamers` (clash-free), `n_kept`,
`rejected`, `duplicates`, `capped`, `unresolved`, `native_angles`), `spread` (per
member: `site_rmsd`, `max_displacement`, `displacement`, `changed`),
`displacement`, `comparison()`, `table()`, `ensemble_table()`, `text()`,
`as_dict()` and `write()`.

```python
from odock import modes

# backbone motion, calibrated to the experimental site RMSDs
network = modes.build_network(conformation, cutoff=13.0, modes=3)
print(network.n_nodes, network.springs, network.n_modes, network.selected)
print(modes.mode_amplitudes(network))          # sqrt(1/lambda), soft modes larger

generated = modes.generate_modes(
    conformation, box=box, cutoff=13.0, modes=3, targets=(0.444, 0.407, 0.144),
)
print(generated.text())
generated.write("mode_ensemble/")

# is the observed motion in the modes, or only its magnitude?
field, unmatched = modes.experimental_field(network, native, other, alignment)
print(modes.direction_overlap(network, field, index=0))            # whole chain
print(modes.direction_overlap(network, field, index=0, nodes=site_nodes))
```

`GeneratedModes` exposes `native`, `network` (`n_nodes`, `springs`, `components`,
`n_modes`, `selected`, `eigenvalues`, `mode()`, `amplitudes()`), `conformations`,
`spread` (per member: `mode`, `target`, `site_rmsd`, `max_displacement`, `clashes`
(severe), `squeezed`, `kept`), `overlaps`, `site_keys`, `comparison()`, `table()`,
`text()`, `as_dict()` and `write()`.

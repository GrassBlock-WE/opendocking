# Covalent docking: warheads, reactive geometry, a score term and the adduct

Covalent inhibitors are designed on a geometry the non-covalent scorer cannot see:
the warhead has to sit at **bonding distance from the nucleophile, in the right
orientation**. This document describes the layer that measures that —
`odock.covalent` — and, more importantly, the **experimental knowledge it does not
have**.

## 1. What this is not, before anything else

Read this section before reading a number in the rest of the document.

* **It cannot know whether the warhead reacts with the protein.** Acrylamides are
  Michael acceptors and cysteines are nucleophiles; whether *this* acrylamide
  reacts with *this* cysteine, at *this* site, in *this* protein, is experimental
  knowledge (a cysteine reactivity dataset, a mass-spec adduct, a time course). The
  module has no electrophilicity index, no pKa, no nucleophilicity, no rate, no
  barrier and no reaction energy. It measures where the atoms are.
* **It cannot know whether the cysteine is nucleophilic.** It perceives the sulfur
  by residue and atom name — the way a chemist does — and reports every candidate.
  A disulfide-bonded cysteine, a cysteine in an occluded pocket and a catalytic
  cysteine all look identical to it. On the bundled 1M17 receptor it finds **six**
  cysteines (CYS751, 757, 773, 794, 915, 926) and none of them is the EGFR C797
  that the covalent EGFR inhibitors target: the construct does not contain C797.
  That is the point: perception is not identification.
* **It cannot know the mechanism.** It assigns one from the warhead's substructure
  (Michael for an acrylamide, SN2 for a chloroacetamide), which is a classification
  of the *pattern*, not a determination for the *molecule*. Reversible Michael
  acceptors, covalent-but-slow inhibitors and non-covalent binders of the same
  cysteine all get the same treatment.
* **The criteria are geometry, not chemistry.** `bonded` means "the sulfur is within
  2.0 Å at an angle inside the warhead's window". It does not mean the bond exists;
  it means the geometry is compatible with one forming.
* **A `near_attack` classification is a hypothesis for a chemist.** It says "this
  pose could form this bond, if the chemistry is what the literature says". It is
  not a prediction that the compound inhibits the protein, and a `non_productive`
  classification does not exonerate a molecule: a different pose, a different
  protonation state or a different conformer may reach.
* **The strain number inherits every limitation `odock.metrics` documents.** It is
  a force-field energy difference on a **truncated fragment** (the ligand plus the
  cysteine's S and CB, with the rest of the protein frozen), so it contains the
  cap's interaction with the ligand, and it is only as good as the force field's
  typing. §5 shows a case where that number is dominated by an artefact and says so.

## 2. Warhead perception

`perceive_warheads(mol)` matches the documented table below and names the
**electrophilic atom** — the atom the sulfur attacks — plus the **reference atom**
the approach angle is measured against. Every pattern names its atoms by SMARTS map
number, and the map number is resolved to a query-atom index explicitly: writing
`[OX1:3]1[#6:1][#6:2]1` and indexing the match with the map number silently returns
the wrong atom, which is why that lookup lives in one function with a test.

| warhead | mechanism | attacked atom | reference atom | angle window |
|---|---|---|---|---|
| acrylamide | Michael | the beta carbon (terminal CH2) | the alpha carbon | 70-150° |
| alpha-cyanoacrylamide | Michael | the beta carbon | the alpha carbon | 70-150° |
| vinyl sulfonamide | Michael | the beta carbon | the alpha carbon | 70-150° |
| vinyl sulfone | Michael | the beta carbon | the alpha carbon | 70-150° |
| chloroacetamide | SN2 | the CH2 carrying the halogen | the halogen (leaving group) | 140-180° |
| epoxide | SN2 (ring opening) | a ring carbon | the other ring carbon | 120-180° |
| nitrile | nitrile addition | the sp carbon | the nitrogen | 140-180° |
| catch-all alkene | Michael | the terminal CH2 | the CH | 70-150° |
| catch-all halide | SN2 | the carbon | the halogen | 140-180° |

Measured on this tree:

```
acrylamide             [(acrylamide, michael, atom 0, reference 1)]
alpha-cyanoacrylamide  [(alpha_cyanoacrylamide, michael, atom 0, ref 1), (nitrile, nitrile, atom 2, ref 3)]
vinyl sulfonamide      [(vinyl_sulfonamide, michael, atom 0, reference 1)]
chloroacetamide        [(chloroacetamide, sn2, atom 1, reference 0)]
epoxide                [(epoxide, sn2, atom 1, reference 0)]
nitrile                [(nitrile, nitrile, atom 1, reference 2)]
```

The two catch-all patterns are reported at lower confidence and are suppressed when
a named warhead already claimed the same electrophilic atom (an acrylamide is also
"any terminal alkene", and is reported once). A molecule with no warhead returns an
empty list, not a guess.

## 3. The reactive geometry

`classify_attack(ligand, warhead, nucleophile)` measures two numbers and classifies
them against the documented criteria (`CovalentCriteria`):

| class | criterion |
|---|---|
| **bonded** | S-electrophile ≤ **2.0 Å** and the approach angle inside the warhead's window |
| **near_attack** | S-electrophile ≤ **4.0 Å** and the angle inside the window, and not bonded |
| **non_productive** | anything else — with the failing criterion named in `reason` |

The distance thresholds are the ones the literature uses: a C–S single bond is about
1.82 Å (so 2.0 Å is the top of the "formed" window) and a near-attack complex is
2.5-4 Å. The angle is measured at the electrophile between the reference atom and
the sulfur: about **107°** for a Michael acceptor (the Bürgi–Dunitz trajectory onto
the π* orbital) and about **180°** for an SN2 backside attack. A degenerate
arrangement has no angle, cannot be classified, and is reported as
`non_productive` with that reason rather than being quietly assigned a number.

Measured on a constructed acrylamide placed against a cysteine sulfur:

| S-C distance | angle | class | quality | covalent term |
|---|---|---|---|---|
| 1.70 Å | 107.0° | bonded | 1.000 | −6.273 |
| 1.82 Å | 107.0° | bonded | 1.000 | −4.436 |
| 2.00 Å | 107.0° | bonded | 1.000 | −3.000 |
| 2.50 Å | 107.0° | near_attack | 1.000 | −3.000 |
| 2.80 Å | 107.0° | near_attack | 1.000 | −3.000 |
| 3.50 Å | 107.0° | near_attack | 1.000 | −3.000 |
| 4.00 Å | 107.0° | near_attack | 1.000 | −3.000 |
| 4.30 Å | 107.0° | non_productive | 0.500 | −1.500 |
| 4.60 Å | 107.0° | non_productive | 0.000 | 0.000 |
| 5.00 Å | 107.0° | non_productive | 0.000 | 0.000 |

And at a fixed 2.5 Å, walking the sulfur around the electrophile: 107° is
`near_attack`, 0° (in line with the alpha carbon) is `non_productive` with the
reason "angle 0.0 deg outside the 70-150 deg window for a michael warhead".

## 4. The covalent term, and why it has no cliff

The term is one number added to an existing non-covalent score — **not** a rewritten
force field:

```
term = -(near_attack_reward · q + bond_reward · b) + adduct_strain_k · excess² · b
```

* `q` is a smooth quality in [0, 1]: a raised-cosine window over distance and over
  angle, so a pose **outside** the criteria gets zero rather than a near-miss
  reward, and a pose **at** a threshold is not stepped over;
* `b` is a smooth bond fraction: 0 above the bonded distance, 1 once the bond is
  clearly formed, ramped over 0.5 Å;
* `excess` is how far the measured S–C–C angle is outside a tetrahedral tolerance
  around 109.5°, which is the adduct's geometric distortion.

Defaults: near-attack reward 3.0, bond reward 5.0, ramp widths 0.6 Å and 20°, adduct
strain coefficient 0.03 kcal/mol/deg², tetrahedral tolerance 10°.

**Continuity is measured, not asserted.** `continuity_scan` sweeps the distance and
the angle through the criteria and reports the largest step between samples. The
decisive check is that **halving the sample spacing halves the largest step** — a
cliff would keep the same jump:

```
distance: max step 1.5451 kcal/mol at 0.100 A  ->  0.7725 at 0.050 A   (ratio 2.00)
angle:    max step 0.8087 kcal/mol at 3.50 deg ->  0.4110 at 1.75 deg  (ratio 1.97)
```

A step the size of the reward (3.0) would be a cliff; the measured step is a bounded
gradient (~15 kcal/mol/Å at the steepest), which is what a smooth window gives.

**What the term changes.** Three poses with *identical* non-covalent affinity, scored
with the term (measured):

| pose | class | d (Å) | angle | term | total | rank |
|---|---|---|---|---|---|---|
| reactive | near_attack | 2.80 | 107.0° | −3.000 | −9.000 | 3 → **1** |
| away | non_productive | 6.00 | 107.0° | 0.000 | −6.000 | 1 → 2 |
| misaligned | non_productive | 2.80 | 0.0° | 0.000 | −6.000 | 2 → 3 |

The non-covalent score cannot separate them at all (all −6.000). The term puts the
pose that can form the bond first, keeps the unreachable pose at the score it
earned, and — importantly — does **not** reward the pose that is close but pointing
the wrong way: being near is not being reactive.

## 5. Building and relaxing the adduct

`build_adduct(pose, warhead, nucleophile)` does the chemistry of the *topology*:
for a Michael addition the alkene becomes a single bond, the alpha carbon gains the
proton the cysteine loses (left to `AddHs`, which places it at a geometry-consistent
position — placing that hydrogen by hand is how a "proton" ends up inside another
atom and a force field reports a million kcal/mol), and the sulfur is capped with the
cysteine's own CB when the receptor provides one. The bond itself is **formed** by
placing the sulfur at 1.82 Å along the approach vector the pose already has; every
other atom stays where the pose put it, so the relaxation that follows measures the
pose's distortion.

`relax_adduct` calls `odock.chem.ligand.minimize` — the project's one minimiser,
with its documented MMFF94 → MMFF94s → UFF fallback — and `adduct_strain` returns an
`odock.metrics.StrainResult` built from that minimiser's `odock_energy_before` and
`odock_energy_after`, with the `reliable` flag set (the adduct's chemistry is
explicit in the molecule, not perceived from a bond-order-free PDBQT).

Measured on the constructed acrylamide/CYS797 fixture:

| stage | C–S | S–C–C | inside the published ranges |
|---|---|---|---|
| formed | 1.820 Å | 107.0° | yes |
| relaxed | 1.825 Å | 112.2° | yes |

The ranges are C–S **1.70-1.95 Å** and S–C–C **95-125°** (a C–S single bond is
~1.82 Å; the angle at the former electrophilic carbon is tetrahedral).

**The strain number, and its honest edge.** On that clash-free fixture the
relaxation has nothing to do: **0.0015 kcal/mol**. Dropping the same independently
embedded ligand next to 1M17's CYS751 (whose CB sits where the crystal puts it)
gives **263 kcal/mol** (bonded start) and **389 kcal/mol** (near-attack start) — and
that number is the **clash between an independently embedded ligand and the truncated
cysteine**, not a chemical strain. It is reported because it is the honest content of
the measurement: the strain of a truncated fragment contains the cap, and the module
does not pretend otherwise. A real docking pose, generated in the pocket, is the
input this number is meant for.

## 6. Using it

```bash
odock covalent -i pose.sdf -r receptor.pdbqt --nucleophile CYS797 \
    --adduct adducts.sdf --json-out covalent.json
```

The command prints every candidate nucleophile, every warhead with its electrophilic
atom, and for each pair the distance, the angle, the class, the covalent term split
into its parts, and — for a productive pair — the relaxed adduct's C–S and S–C–C
values, its strain and its `reliable` flag. Measured on the bundled data with an
acrylamide placed 2.8 Å from 1M17's CYS773 at 107°:

```
probe_reactive: 1 warhead(s)
  acrylamide -> CYS773:SG: d 2.80 A, angle 107.0 deg, near_attack
    term -3.000 kcal/mol (near 3.000, bond 0.000, strain 0.000)
    adduct: C-S 1.82 A, S-C-C 110.0 deg, in range: True
    strain: 284.997 kcal/mol (reliable: True, MMFF94)

1 productive pair(s). A near-attack geometry is a hypothesis for a chemist: this tool
does not know whether the warhead reacts or whether the cysteine is nucleophilic.
```

That 285 kcal/mol is the §5 artefact, not a chemistry result: the acrylamide was
embedded independently and translated next to the cysteine, so its phenyl tail sits
where the truncated cysteine's CB sits. A pose generated in the pocket is the input
this number is meant for, and the command prints it rather than hiding it.

The library API:

```python
from odock import covalent

hits = covalent.perceive_warheads(ligand)             # which warhead, which atom
targets = covalent.perceive_nucleophiles(receptor)    # which cysteine, with geometry
geometry = covalent.classify_attack(ligand, hits[0], targets[0])
term = covalent.covalent_term(geometry)               # one number, documented parts
total = covalent.covalent_score(non_covalent, geometry)

if geometry.productive:
    adduct = covalent.relax_adduct(covalent.build_adduct(ligand, hits[0], targets[0]))
    print(covalent.adduct_quality(adduct))            # C-S and S-C-C against the ranges
    strain = covalent.adduct_strain(adduct)           # StrainResult, with .reliable
```

The subcommand is registered through `odock.cli_ext` by one line —
`(".covalent", "add_covalent_parser")` — so `odock.cli` itself is not edited.

## 7. Validation, and what it does not establish

Measured, on this tree, by `tests/test_covalent.py` (29 tests, ~1.5 s):

* **perception** — nine warhead types recognised with their mechanism, the
  electrophilic atom hand-checked (the terminal CH2 of an acrylamide, the CH2
  carrying the chlorine of a chloroacetamide), catch-all suppression, a molecule
  with no warhead returns nothing;
* **nucleophiles** — the bundled 1M17 receptor is read (its PDBQT type column
  rewritten) and its six cysteines found by residue/atom name with their CB caps; a
  receptor without residue information falls back to a geometric rule *and says so*;
* **geometry** — the constructed reactive pose classifies `bonded` at 1.82 Å and
  `near_attack` at 2.8 Å with the ideal 107° approach, the same molecule at 6 Å is
  out of reach, and the same distance pointing the wrong way is `non_productive`
  with the angle named;
* **the term** — rewards near-attack, pays more once the bond forms, pays **nothing**
  beyond reach, penalises a strained S–C–C angle by the documented quadratic, and is
  continuous (the halving-the-spacing test above);
* **the adduct** — built, formed at 1.82 Å, relaxed through the project's minimiser,
  measured against the published C–S and S–C–C ranges, with the strain machinery
  reporting `reliable=True`; the SN2 adduct loses its leaving group and the epoxide
  adduct opens its ring;
* **the ranking** — the constructed three-pose case in §4.

**What it does not establish.** None of the above is evidence that a compound is a
covalent inhibitor of anything. The validation is geometric self-consistency: the
classifier classifies by the criteria it documents, the term is the function it
claims to be, the adduct geometry lands inside published ranges, and the ranking
changes for the documented reason. It says nothing about reactivity, selectivity,
kinetics, residence time or cell activity — see §1.

## 8. Reproducing the numbers

```bash
python -m pytest tests/test_covalent.py -q
```

The measurements in this document come from the same constructions the tests use
(an embedded acrylamide, a cysteine sulfur with a CB cap, and the bundled
`demo/systems/1m17/receptor.pdbqt`); the strain figures in §5 are printed by the
adduct fixture, which builds the fragment, forms the bond, relaxes it and reports
`adduct_quality` and `adduct_strain`.

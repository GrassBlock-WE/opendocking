# Scoring a ligand against the pocket

Every 3-D method in this project compared a ligand to **another ligand**: the
pharmacophore fit, the shape/electrostatic overlay, the USR pre-filter.  This module
compares a ligand to the **receptor**: the pocket's shape and its electrostatic
potential, sampled from one structure, and a rigid-body placement of each conformer
inside them.

```bash
odock pocket-score -r demo/3ptb/receptor.pdbqt -i demo/library.smi \
    --box demo/3ptb/box.json --reference-ligand demo/3ptb/ligand.pdbqt \
    --keep 0.05 --actives benzamidine --actives hydroxybenzamidine --json-out out/pocket.json
```

## 1. The pocket field, and what it costs

`PocketField.build` gives :func:`odock.gui.surface.scalar_field` the box (plus a
small margin) and the receptor atoms that fall inside the box plus a **4 Å shell**:

* **shape** — the probe-centre field `F(p) = min_i(|p − c_i| − (r_i + probe))`,
  negative inside the receptor's van der Waals volume, zero at contact, positive in
  the open space a ligand may occupy.  The field is *local*: an atom further than the
  shell cannot change it inside the box, so the shell is the answer, not an
  approximation of it.  On 3PTB, 700 of 1 630 receptor atoms are inside it.
* **electrostatics** — :func:`odock.gui.surface.electrostatic_potential` on the same
  grid, `φ(p) = 332.0637 Σ q_i / (ε r)`, with the receptor's own PDBQT charges and the
  AutoDock distance-dependent dielectric (ε = 4·r).

Cost, measured on the bundled 3PTB box (17.9 × 20.0 × 20.5 Å):

| spacing | voxels | build | per voxel |
|---|---|---|---|
| 0.5 Å | 112 455 | 4.99 s | 44 µs |
| **0.8 Å (default)** | 28 768 | **1.20 s** | 42 µs |
| 1.0 Å | 14 950 | 0.63 s | 42 µs |

**The build is linear in voxels × receptor atoms and it is dominated by the
electrostatic potential**: on the same box `scalar_field` takes 0.05 s while
`electrostatic_potential` takes 7.6 s at 178 k voxels.  The field is built **once per
receptor**, so this is the one-off cost of a screen, not a per-ligand one.

## 2. The score, and what it costs per pose

A pose is scored by looking the field up at each ligand heavy atom (nearest voxel):

```
shape = mean_i exp(-(F_i - 0.8)² / (2 · 1.5²)) · (1 - clash_fraction) · size_factor
ESP   = Σ_i q_i · φ(r_i)                                    kcal/mol, signed
score = 0.5 · shape + 0.5 · min(1, max(0, -ESP / 10))        the overlay's convention
```

Measured on this machine:

| operation | cost |
|---|---|
| `pocket_score` on given coordinates (9-atom ligand) | **182 µs/pose** |
| `place_and_score`, the full rigid search | **10 ms** per conformer (689 poses, **15 µs/pose** in one batch) |
| ranking `demo/library.smi` (17 molecules, 2 conformers, 13 516 poses) | **0.8 s** |

The batch is where the speed comes from: the search scores 48-64 poses per array
operation, so the per-pose cost falls from 182 µs to 15 µs.  Scaling with pocket size
is the *voxel count* in the build (linear, above) and nothing per pose: the lookup is
two fancy-indexes per atom whatever the grid holds.

**Three things the score had to be taught, each because the obvious version failed on
the measurement:**

1. **Counted contacts saturate.**  Every heavy atom of every docked benzamidine pose
   is "in contact", so a contact count is 1.000 for all nine poses and discriminates
   nothing.  The term is a Gaussian kernel around a *snug* contact (0.8 Å) instead.
2. **A contact kernel also saturates across molecules.**  Ranked by the kernel alone,
   all 17 demo molecules scored 0.994-0.996: a 20 Å pocket always has room to put a
   small ligand in contact.  The score needs a **size** term.
3. **The size term needs a reference.**  `--reference-ligand` calibrates it on the
   ligand the pocket is *known* to bind: the 3PTB co-crystallised benzamidine is
   175 Å³, and a ligand half or one-and-a-half times that keeps `exp(-0.5) = 0.61`.
   Without a reference the term is off, the score saturates, and the command says so.

## 3. Validation: the correlation with the docked affinity

`demo/3ptb/result.json` holds the re-docking validation of the 3PTB ligand: four
benzamidine poses with Vina affinities from −6.21 to −4.51 kcal/mol.  Scoring each
pose gives:

| affinity | score | shape | ESP | clash |
|---|---|---|---|---|
| −6.212 | 0.471 | 0.941 | +21.8 | 0 |
| −6.191 | 0.474 | 0.949 | +20.7 | 0 |
| −5.035 | 0.462 | 0.923 | +19.7 | 0 |
| −4.513 | 0.477 | 0.954 | +15.7 | 0 |

| term | Spearman ρ | n | 95 % CI | minimum detectable \|ρ\| | resolvable |
|---|---|---|---|---|---|
| combined | +0.400 | 4 | [−1.00, +1.00] | 0.99 | **no** |
| shape | +0.400 | 4 | [−1.00, +1.00] | 0.99 | **no** |
| ESP | −1.000 | 4 | [−1.00, −1.00] | 0.99 | "yes", and it means nothing |

**n = 4 cannot validate anything, and this document does not claim otherwise.**  The
minimum detectable |ρ| for a rank correlation at this sample size is **0.99**: only a
perfect correlation could clear the bar, which is why the ESP column is flagged as
resolvable — a perfect rank order across four poses of *one ligand* is a coincidence
worth reporting, not a result.  What n would be needed:

| n | minimum detectable \|ρ\| |
|---|---|
| 6 | 0.92 |
| 10 | 0.79 |
| 20 | 0.59 |
| 40 | 0.43 |

The repository has no larger docked ground truth: `odock lbvs`'s benchmark could not
be enlarged either (`docs/LBVS.md` §8), and running Vina over a library is the one
thing that would produce it.  So the honest statement is: **the correlation is not
resolvable at the data available, and the point estimates above must not be quoted as
a validation.**

### 3.1 The electrostatic term says the opposite of the chemistry

The 3PTB charges are for **neutral** benzamidine, and Gasteiger puts −0.30 e on the
amidine nitrogens.  They sit in the Asp189 pocket's −42 kcal/(mol·e) potential, so
`q·φ` is **+12.9 and +14.2 kcal/mol** and the total interaction is **+21 kcal/mol —
unfavourable**, for the ligand whose amidinium–Asp189 salt bridge is the interaction
that defines this pocket.  The term is not merely weak here; its sign is wrong,
because the protonation state the file was prepared in is not the one that binds.

That is why the ESP contribution is 0 in every row above (the clamp
`max(0, −ESP/10)` catches a positive energy) and the combined score is effectively
shape-only.  It is reported rather than hidden: **an electrostatic complementarity is
only as good as the charges it is given**, and this repository prepares ligands in
their neutral form.  `--no-electrostatics` makes the reduction explicit.

**And the flag now makes it impossible to miss.**  `PoseScore` carries `esp_term`,
`esp_term_used` and an `esp_note`, so a caller cannot report a combined score under an
electrostatic label when the clamp swallowed the term; `rank_library` adds an aggregate
note naming how many molecules that happened to, and the CLI prints it.

### 3.2 What protonating the ligand actually costs

`docs/PROTONATION.md` §1 measures three charge assignments at the same crystal pose.
Correcting the *state* is not enough — Gasteiger keeps a formally cationic nitrogen
negative — and what flips the sign is restoring the group's **formal charge**:

| assignment | amidine N | Σ q·φ (kcal/mol) | ESP term | score |
|---|---|---|---|---|
| (a) as prepared (file charges, neutral) | −0.62 e | +22.09 | 0.000 | 0.471 |
| (b) Gasteiger on the amidinium | −0.11 e | +20.98 | 0.000 | 0.471 |
| (c) (b) with the formal +1 restored | +1.00 e | **−28.75** | 1.000 | **0.971** |
| (c′) the correction applied to (a)'s charges | +1.00 e | −13.95 | 1.000 | **0.971** |

**The magnitude depends on where the correction starts (15 kcal/mol between the last
two rows) and the conclusion does not**: the sign flips, the term is used, and the
score goes 0.471 → 0.971.  **And it changes the ranking**: the five amidines take ranks
1, 2, 3, 10, 11 with the clamp active and 1, 2, 3, 5, 6 with the correction
(`rank_library(..., formal_charge_correction=True)`).  The salt bridge it is about is
real and short: the two amidine nitrogens of the co-crystallised pose are **2.87 Å**
from the nearest Asp189 oxygen.

## 4. An honest use case: a pre-filter before docking

Ranking `demo/library.smi` (17 molecules, the five ring-amidines are the documented
trypsin binders) and cutting, against the 2-D fingerprint filter on the same library:

| keep | molecules kept | pocket: binders kept | pocket recall | fingerprint recall |
|---|---|---|---|---|
| 5 % | 1 of 17 | 1 of 5 | 20 % | 20 % |
| 10 % | 2 of 17 | 2 of 5 | 40 % | 40 % |
| 20 % | 4 of 17 | 3 of 5 | **60 %** | **80 %** |

**The structure-based filter is not better than the cheap 2-D one, and at a 20 % cut
it is worse (3 of 5 against 4 of 5).**  A fast filter that loses binders the
fingerprint keeps is not an improvement, so this is reported as a negative result and
the recommendation is unchanged: use the fingerprint pre-filter unless there is a
reason the 2-D similarity cannot see the chemistry (`docs/LBVS.md` §5).

The ranking itself (default grid, size reference 175 Å³) is chemically sensible at the
top — hydroxybenzamidine, benzamidine and fluorobenzamidine are the first three, and
the 25-heavy-atom warfarin is last at 0.004 — but the cut is **grid-sensitive**: at
0.75 Å spacing the same run puts benzamidine first and isonicotinic acid second.  A
pre-filter threshold from this score is a heuristic, not a decision.

## 5. What this does not establish

* **A shape complementarity is not a binding energy.**  There is no desolvation, no
  entropy, no induced fit, no cooperativity and no explicit hydrogen bonding; a
  molecule can be perfectly complementary to a pocket and not bind it, and the reverse.
* **The receptor does not move.**  The field comes from one rigid structure, and the
  search is rigid-body: no flexible sidechains, no receptor relaxation.  Sidechain
  motion is often the difference between a clash and a bind.
* **A pocket field inherits the ensemble's limitations and adds its own.**  Every
  conformer limitation `docs/CONFORMERS.md` documents applies, plus: the box is a
  hypothesis about where the ligand goes, and a ligand outside it is scored 0 rather
  than "not scored".
* **The electrostatic term is only as good as the charges.**  Measured here, they give
  the wrong sign for the very interaction the pocket is known for (§3.1).  Treat the
  ESP column as a diagnostic of the charges, not as an energy.
* **The validation is not resolvable.**  n = 4 poses of one ligand, minimum detectable
  |ρ| = 0.99 (§3).  This module has *not* been shown to rank binders; it has been shown
  to be fast, to be well defined, and to be worse than a fingerprint pre-filter at a
  20 % cut.
* **The cut is grid-sensitive** (0.75 Å against 0.8 Å reorders the top of the list), so
  a threshold from this score should never be the decision — it is a way to make a
  library smaller before the thing that actually ranks.

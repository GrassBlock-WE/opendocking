# Fragments: screen small, then grow or link

Fragment-based drug discovery runs the other way round from everything else in this
project. Instead of docking a lead-sized molecule, it screens **small** things and then
**grows** or **links** them, and it judges a hit by **ligand efficiency** — affinity per
heavy atom — because a 12-atom fragment at −6 kcal/mol is more interesting than a 30-atom
one at −7. `odock.fragments` supplies that workflow: a fragment set, an efficiency-ranked
screen, a growth enumerator and a linker enumerator.

**Read the two failures before the one success.**  On the bundled 3PTB system a fragment
does **not** land where the ligand's substructure sits, and the reason is not a tuning
problem: the pocket score has no way to price a buried ion, so its electrostatic term is
unusable in **both** directions.  Everything downstream — growing, linking, and the
efficiency numbers the screen prints — inherits that.  The one genuinely good result is
at the end of the next section, and it is good precisely because it does not depend on
the placement being right.

## 1. The dual failure: clamped is shape-only, corrected saturates

Benzamidine split into its two fragments, each placed in the 3PTB pocket independently,
compared against the position of the corresponding substructure in the **co-crystallised**
pose. No re-alignment: both coordinate sets are already in the receptor frame, so the
number answers "did the fragment land where the ligand's substructure is".

| fragment | HA | charges | score | ESP (kcal/mol) | ESP term | **RMSD to the crystal substructure** |
|---|---|---|---|---|---|---|
| amidine `HC(=N)N` | 3 | as prepared (neutral) | 0.500 | +… (unfavourable) | **0.000, clamped** | **6.87 Å** |
| amidine `HC(=N)N` | 3 | formal charge restored | **1.000** | −… (deep) | **1.000, saturated** | **20.61 Å** |
| phenyl `c1ccccc1` | 6 | as prepared | 0.495 | unfavourable | 0.000, clamped | **4.47 Å** |
| phenyl `c1ccccc1` | 6 | formal charge restored | 0.495 | still unfavourable | 0.000, clamped | **23.04 Å** |

**Both regimes fail, and they fail differently.**

* **Clamped** (the left column): the interaction energy comes out positive, the clamp
  `max(0, −ESP/10)` sends the term to zero, and the score is `0.5 × shape`. With a
  three-heavy-atom fragment almost no shape information exists, so the placement search
  is not discriminating — 6.87 Å and 4.47 Å away from the truth.
* **Corrected** (the second row): restoring the amidinium's formal +1 (via
  `protonation.correct_formal_charges`) *un-clamps* the term — and then it
  **saturates**. A buried ion sits in a field deep enough that `−ESP/10` exceeds 1, the
  term pins at its maximum, and the combined score becomes half shape plus half a
  **constant**. The pose then collapses onto the electrostatic minimum: **20.61 Å**,
  three times further from the truth than the clamped placement it replaced.

The missing physics is the point, and it is the sentence to take away: **there is no
desolvation or repulsion balance to price a buried ion, and at three heavy atoms that
interaction is the entire score.** The same defect appeared in `docs/PROTONATION.md` for
the *full* ligand, where the fix was correct and necessary; for a *fragment* the fix is
correct and insufficient, because a fragment has no shape to fall back on.

### The trap this creates, and why the screen now shouts

Correcting the charges made the electrostatic term "usable" and made the table **more
misleading**. Ranking the fragment library by efficiency:

| | top fragment | LE | score |
|---|---|---|---|
| as prepared (ESP clamped for everything) | 8 HA, `Nc1cc[nH]c(=O)n1` | **0.618** | 0.495 |
| after the formal-charge correction | the same fragment | **1.243** | **0.995** |

The method did not improve — the *number* doubled, because a saturated term contributes a
constant that every fragment with a charged group collects. That is the clearest example
in this project of a correction making a table look more like a ranking while the
underlying measurement got no better, and `screen_fragments` now states the regime at the
top of its notes rather than leaving the number to speak for itself:

> BY SHAPE ONLY: the electrostatic term is clamped to zero for 33 of 41 fragment(s) even
> after the formal-charge correction, so their poses were decided by shape alone. At 3-6
> heavy atoms that is a weak basis for a ranking, and the LE column for those fragments
> should be read as a hypothesis rather than a score.

`tests/test_fragments.py` **pins the failure as a failure**: the placement-RMSD test
asserts the RMSD is *large* and that the amidine's score saturates. If a future change
makes fragment placement work, that test starts failing, which is the signal to rewrite
this page.

### 1.1 A correction that changes numbers without improving them — three instances

This is the pattern this project keeps meeting, and the instances are worth grouping
because the *shape* of the mistake is what repeats: a plausible physical fix is applied,
the numbers move, and the measurement the fix was supposed to improve does not.

| the "correction" | what it moved | what it did **not** move |
|---|---|---|
| restoring the amidinium's formal charge (task-32, `docs/PROTONATION.md`) | the interaction energy, +22.09 → −28.75 kcal/mol; the score 0.471 → 0.971 | the pose — the placement **got worse**, 6.87 → **20.61 Å**, because the term saturates at 1.000 and dominates the 0.5/0.5 combination |
| the same correction reaching the fragment screen (task-45) | the top fragment's LE, **0.618 → 1.243** at score 0.995 | the method: a saturated term adds a *constant* that every charged fragment collects |
| desolvation + repulsion penalties, inside the search objective (task-48) | the amidine's score, 0.999 → 0.749 | the placement, **20.61 Å before and after**, because both penalties are pinned at their maximum for every atom of every pose — constants that cancel in a comparison |
| the same penalties with thresholds calibrated on `PocketField.sample()` **at ligand-atom positions** rather than on the grid (task-48, final) | the score again, 0.749 → **0.923**, and the penalties finally **vary** (repulsion 0.153 at the pose found vs 0.499 at the crystal position) | the placement, **still 20.61 Å**, now for a diagnosed reason: **both penalties charge for depth**, so the correct (more buried) crystal position pays *more* penalty than the wrong, shallower pose. For a fragment they are **anti-correlated with the right answer** |

The last row also corrects the row above it.  The constant penalties were **not** a units
error — `sample()` and `shape_field` are the same field (0.444/0.444 at every point
tested) — they were a **population** error: the grid is dominated by solvent and open
voxels (p50 = −1.15) while a ligand atom in a pocket samples the deep tail (p50 =
**−9.93**, p5 = **−42.19**).  Thresholds taken from the grid pinned both penalties at
maximum for every pose.  Fixing the population made the terms *vary* and did not make the
placement *correct*, which isolates the remaining failure to the **form** of the terms
rather than to their calibration: a burial-scaled penalty and a depth ramp both reward
being shallow, and a fragment that binds is deep.  Whether a desolvation term can be
written that is not anti-correlated with burial for a *fragment* is the open question.

Each row is a correct observation of a *number* and an incorrect expectation about a
*result*.  The defence, used four times here, is to state the quantity the change is
supposed to move **before** making it and then measure *that*, rather than the number that
moved by itself — which is why the table in §1 reports RMSD beside score, and why the
tripwire test asserts the RMSD rather than the score.

### 1.2 Why a desolvation term cannot be an objective — the fifth instance and its cause

Separating the two quantities as they should be (explicit vdW overlap for clash; complex
SASA × polarity for desolvation) produced a falsifiable test that fails in a way which
answers the open question outright.  At the crystal substructure versus the pose the search
prefers:

| fragment | term | crystal (correct) | found (20.61 Å away) | verdict |
|---|---|---|---|---|
| amidine (3 HA) | vdW overlap | **0.000** | **0.000** | flat — correctly, and it decides nothing |
| amidine (3 HA) | desolvation | **0.501** | **0.000** | **wrong sign** |
| phenyl (6 HA) | vdW overlap | 0.000 | 0.000 | flat |
| phenyl (6 HA) | desolvation | 0.114 | 0.003 | **wrong sign** |

**The desolvation term is not buggy — it is physically right, and that is precisely why it
cannot be the objective.**  The correct pose buries the amidine's polar nitrogen, so it
loses solvent-accessible surface and pays 0.501.  The wrong pose the search prefers is
*solvent-exposed*, so it pays nothing.  **A pure desolvation penalty can only ever reward
not binding.**

In a real scoring function the desolvation cost of burying a polar group is *paid for* by
the interaction that burial buys — a hydrogen bond, a salt bridge — with
``ΔG = ΔE_interaction + ΔG_desolv``: the desolvation is a **counterweight**, never a term to
maximise alone.  Here the counterweight is missing because the interaction term is clamped
to zero for these fragments (`docs/PROTONATION.md`), so the objective reduces to
``−desolvation`` — and optimising that means maximising solvent exposure.  The two findings
close on each other:

> **A desolvation term needs an un-clamped interaction term to balance it, and the
> formal-charge correction is what un-clamps it — so the correction has to happen *before*
> the search.**  In `place_fragment` it happens after, which is why every objective variant
> so far has pushed the fragment out of the pocket.

The vdW overlap term is the one part that behaved as intended: flat at both poses, exactly
what a clash detector should be, contributing nothing to a ranking it should not decide.
It stays.

## 2. The one positive result: growing recovers the crystallographic ligand

From the **three-heavy-atom amidine fragment alone**, enumerated at its own attachment
vectors from a documented reagent set, placed by a locally computed Kabsch fit onto the
pose it came from, clash-filtered against the pocket field and ranked by **incremental
ligand efficiency** `ΔLE = −(ΔG_child − ΔG_parent) / added heavy atoms`:

| rank | reagent | Δ added HA | total HA | ΔLE | LE | product |
|---|---|---|---|---|---|---|
| 1 | phenyl | 6 | 9 | **−0.898** | 0.512 | `[H]C(N)=Nc1ccccc1` |
| 2 | pyridin-3-yl | 6 | 9 | −0.907 | 0.506 | `[H]C(N)=Nc1cccnc1` |
| 3 | pyridin-3-yl | 6 | 9 | −0.930 | 0.491 | `[H]C(=N)Nc1cccnc1` |
| 4 | phenyl | 6 | 9 | −0.933 | 0.489 | `[H]C(=N)Nc1ccccc1` |
| 5 | trifluoromethyl | 4 | 7 | −1.318 | 0.675 | `[H]C(N)=NC(F)(F)F` |
| 6 | sulfonamide | 4 | 7 | −1.356 | 0.654 | `[H]C(N)=NS(N)(=O)=O` |

**Rank 1 is benzamidine: the actual 3PTB ligand, recovered from a three-atom fragment by
enumeration.** Drop the exploration cap hydrogen and the leader is C₇H₈N₂ with a phenyl
ring and an amidine N–C–N unit — the ligand that crystallises in this pocket. The
aromatic preference a chemist would expect for the S1 pocket is what the table shows, and
it is the one result here that does not depend on the placement being right, because the
ranking is over growths of *the same* placed fragment.

Of **40 candidates generated** from 3 attachment vectors × 14 reagents, **38 survived**
and 2 were rejected for overlapping the receptor (`phenyl@1` and `pyridin-3-yl@1`: "2
atom(s) overlap the receptor (pocket field below -0.6 A) — a clash is not a suggestion").
Rejections carry their reason, so "how many survived" is auditable rather than trusted.

### ΔLE cannot separate the top two

ΔLE is a heuristic computed from one conformer and one score, so its noise is *estimated*
rather than assumed away. `growth_spread` re-runs the growth at three seeds:

```
seeds 20240101, 7, 99 -> ΔLE [-0.8987, -0.8941, -0.8987]
spread 0.0046 kcal/mol   agree: False
```

Phenyl and pyridin-3-yl are **within 0.005 of each other**, so the honest statement is
**"ΔLE cannot separate the top two reagents"** — not "phenyl won". A rank is not a result.

## 3. Linking: the enumerator is right, its input is wrong

Two fragments placed in the pocket, a linker enumerated from a documented set, filtered by
the geometry the two attachment vectors demand and by the pocket's free volume. The two
*placements* sit **4.19 Å** apart, so a linker must span 3.59–6.69 Å:

| linker | span | verdict |
|---|---|---|
| `direct` (`[*][*]`, a direct bond) | 1.50 Å | rejected — cannot cover 4.19 Å |
| `methylene` / `ketone` | 2.51 / 2.56 Å | rejected — too short |
| `sulfonamide` | 3.45 Å | rejected — too short |
| `amide`, `reverse_amide`, `methylamine`, `ethynyl`, `ether`, `propylene`, `E_alkene` | 3.78–4.82 Å | **build** (11–15 HA), then rejected: fragment A pinned at RMSD 0.03–0.07 Å but fragment B at **5.68–7.52 Å** |
| `piperazine` | 5.69 Å | rejected — B at 7.52 Å |

**The falsifiable target was that linking two fragments of benzamidine would recover a
direct bond, and it is not met** — but not because the enumerator is wrong. Benzamidine's
amidine is bonded *straight* to its phenyl, so the correct linker is `direct` at ~1.5 Å;
the enumerator rejects it, correctly, because the **placements** demand 4.19 Å. The
enumerator would have found the right answer if section 1 held. That is a chain of
reasoning with a named cause, not an excuse: the failure is upstream, in the placement.

## 4. What this is not

* **Enumeration plus scoring, not chemistry.** A suggested growth or linker has **no
  synthetic route, no yield, no protecting groups, no reagent availability and no
  commercial catalogue** behind it. Every suggested analogue is a hypothesis.
* **The reagent and linker sets are short on purpose.** Fourteen reagents and thirteen
  linkers, documented in `fragments.GROWTH_REAGENTS` and `fragments.LINKERS`. A small set
  a reader can audit beats a large one nobody can; a chemist should read the tables as
  *examples of the method*, not as a proposal.
* **The heavy-atom floor of 8 is a convention, not chemistry.** It keeps a 3-atom binder
  from winning on a technicality (that is why the validation fragments, at 3 and 6 heavy
  atoms, are *below* it). The window is reported, not assumed: `n_dropped_heavy` counts
  what it excluded.
* **LE from a score is a ranking device.** With no measured affinity, `ΔG` is the pocket
  score mapped through 10 kcal/mol — the same reference the electrostatic clamp uses — and
  every row is marked `measured=False`. A high-LE fragment from a docking-shaped score is
  a hypothesis to test, not a hit.
* **Fragment placement on this pocket is not trustworthy at 3–6 heavy atoms**, in either
  electrostatic regime. Fixing that needs a scoring function with a desolvation or
  repulsion term — not a tuning of the clamp.
* **The growth and linker enumerators are not validated end to end.** Growth's *ranking*
  is validated by the benzamidine result; its *placement* inherits section 1. Linking is
  unvalidated, for the reason in section 3.

## 5. Using it

```python
from odock import fragments as F, pocket_score as PS

pocket = PS.PocketField.build(atoms, center=box["center"], size=box["size"])
screen = F.screen_fragments(library, pocket)          # ranked by LE, floor 8 HA
print(screen.table(10)); print("\n".join(screen.notes))

placed = F.place_fragment(fragment, pocket)
growth = F.grow_fragment(placed, pocket)              # ΔLE-ranked analogues
print(growth.table(10))

print(F.growth_spread(placed, pocket, seeds=(20240101, 7, 99)))   # the noise, not assumed

left  = F.place_fragment(frag_a, pocket)
right = F.place_fragment(frag_b, pocket)
print(F.link_fragments(left, right, pocket).table())
```

`tests/test_fragments.py` (12 tests) covers the window and the generator, the split's
real-hydrogen capping, the docked-pose mapping, both electrostatic failure modes, the
placement failure, the benzamidine recovery, the ΔLE noise, and the linker span filter.
The three 3PTB tests are marked `slow`.

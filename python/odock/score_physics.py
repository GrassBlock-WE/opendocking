# SPDX-License-Identifier: GPL-3.0-or-later
"""The two terms the pocket score was missing: desolvation and repulsion.

`docs/FRAGMENTS.md` §1 records a diagnosis rather than a complaint: fragment placement
fails in **both** electrostatic regimes because the score has no way to price a buried
ion.  Clamped, the term is zero and the placement is shape-only; corrected, the term
saturates at its maximum, contributes a constant, and the pose collapses onto the field
minimum (20.61 Å from the crystal position).  Both regimes are explained by the absence
of two terms, and this module supplies them:

* :func:`desolvation_penalty` — burying a polar or charged atom costs energy that the
  score never charged for.  The **standard crude form** is a linear SASA-proportional
  term (``ΔG_desolv ≈ Σ k_i · ΔSASA_i``); this is that form, with the per-atom
  coefficient ``k_i`` split into a **burial** fraction (from
  :func:`odock.sasa.sasa_per_atom`, so it uses the surface the project already computes)
  and a **polarity** factor (per element and per charge).  It is crude on purpose and it
  is not fitted to any system: what it has to achieve is that a buried amidinium stops
  being *infinitely rewarded*, which is a sign test, not a calibration.
* :func:`repulsion_penalty` — nothing punishes overlap, which is why the combined value
  can pin at 1.000.  This is a bounded soft-clash ramp over the pocket field: 0 outside
  the surface, rising linearly to a hard stop over a stated depth range, so a deep
  overlap can subtract at most :data:`REPULSION_WEIGHT` and the total **cannot saturate**.

Both are combined by :func:`physical_score`, which subtracts them from the weighted
shape/electrostatic sum and clamps the result to ``[0, 1]``.

**This module is NOT in any default path, and that is deliberate.**
``fragments.place_fragment(physics_enabled=...)`` defaults to **False** and nothing else
calls this module.  The terms are correct as functions and hand-checked, but they are
wired into the fragment path *after* the pose search rather than inside it, and:

    **A scoring term that is not in the search objective cannot affect the search.**
    Applied to the winning pose, a term can change which candidate you *prefer*; it
    cannot change which pose you *find*.

Measured with the flag on: the amidine's score moved 0.999 -> 0.749 (the repulsion
penalty — the real saturation fix) while its placement RMSD stayed **20.61 A**, and the
desolvation penalty came out **0.000-0.001**.  The cause of that zero is **not** the
proxy, and the first hypothesis here (burial computed in vacuum) is **wrong — measured
and falsified**: the per-atom SASA of the amidine is *identical* alone and against
receptor + ligand (58.25 / 40.69 / 60.62 Å²), because **the pose the search finds is not
buried at all** — it sits fully solvent-exposed, 20.61 Å from the crystal position, at a
field minimum that happens to be outside the pocket.  So the term is inert *as a
consequence of the placement failure it was meant to fix*: an exposed ion correctly pays
no desolvation penalty, and the buried pose that would pay one is never found.

That is a causal loop, and it has one exit: **the penalties must be inside the search
objective**, so the search can *find* a buried pose instead of an exposed saturated one.
Applied afterwards, no coefficient, threshold or element weight can help — the worst case
for this fragment if every atom were fully buried is only 0.296 after the weight, and
that pose never appears.  Measured facts, for whoever picks this up:

* per-atom SASA, amidine alone vs receptor + ligand: **58.25 / 40.69 / 60.62** both ways;
* `FULLY_EXPOSED_SASA = 35 Å²` therefore classifies all three atoms as exposed (burial 0);
* polarity factors are **0.487 / 1.000 / 0.290** (charges 0.243 / 0.612 / 0.145), so the
  term *would* bite on a buried cation — the maximum penalty for this fragment is 0.592
  before the weight, and it is 0.000 for the pose actually found.

Both fixes are known and neither is done here: the integration belongs in
`pocket_score`'s objective (the safer shape is an objective hook that `fragments` passes,
rather than editing that module's shared combination), and the burial proxy should be
re-tested against a complex-based SASA once a buried pose is actually reachable.

So **do not turn this on until the objective integration and the complex-based burial land
together**, and re-measure the four rows of `docs/FRAGMENTS.md` §1 when they do.  A term
that makes scores *different without making them better* is the "corrected but not
improved" hazard this project keeps meeting; the honest state for a measured but unwired
module is to say so rather than to ship a false default.

**What this is not.**  A linear SASA term with a fixed per-element coefficient is the
crude end of implicit-solvent models: it ignores curvature, screening, counter-ions and
the difference between a charged group's solvation in bulk and in a low-dielectric
pocket.  A real treatment is a Poisson-Boltzmann or generalised-Born calculation.  The
claim here is narrow and testable — a buried ion is now *charged for* — and §5 of
`docs/FRAGMENTS.md` says so beside the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sasa import sasa_per_atom

__all__ = [
    "DESOLVATION_WEIGHT",
    "REPULSION_WEIGHT",
    "OVERLAP_REFERENCE",
    "OVERLAP_THRESHOLD",
    "FULLY_EXPOSED_SASA",
    "SURFACE_FIELD",
    "DEEP_FIELD",
    "POLAR_ELEMENTS",
    "PhysicsBreakdown",
    "burial_per_atom",
    "polarity_per_atom",
    "desolvation_penalty",
    "repulsion_penalty",
    "physical_score",
]

#: The most a desolvation penalty can subtract from a combined score.  Chosen to
#: **bound** the term rather than to fit anything: half the scale, so no single term can
#: flip a good shape complementarity on its own, and so the fragment and ligand paths use
#: the same number.  Which way the 3PTB numbers move with it is a *test* of the form, not
#: an input to it.
DESOLVATION_WEIGHT = 0.5
#: The most a repulsion penalty can subtract, for the same reason: with this bounded,
#: ``combined`` cannot reach 1.000 by burying an ion.
REPULSION_WEIGHT = 0.5

#: Per-atom SASA (Å²) at or above which an atom counts as fully solvent-exposed.  A
#: carbon in a free amino acid is roughly 20-40 Å² depending on its position, so this is
#: an upper bound for a small molecule rather than a per-element constant.
FULLY_EXPOSED_SASA = 35.0

#: The pocket field value at which the surface begins (no overlap) and the value that
#: counts as a hard overlap.  Between them the repulsion ramps linearly; below
#: :data:`DEEP_FIELD` it is pinned, which is what keeps it bounded.
#:
#: **These are percentiles of ``PocketField.sample()`` at LIGAND-ATOM positions — the
#: quantity the scoring hook actually receives — and NOT percentiles of the grid arrays.**
#: ``sample()`` and ``shape_field`` are the *same* field (they agree to three decimals at
#: every point tested: 0.444/0.444, 0.295/0.295, 1.078/1.078), so this was never a units
#: error: it is a **population** error, and it is why two earlier calibrations both
#: produced constants.  The grid's distribution is dominated by solvent and open voxels
#: (3PTB grid p50 = −1.15, min = −3.08), while a ligand atom sitting in a pocket samples
#: the deep tail (3PTB at ligand positions p50 = **−9.93**, p5 = **−42.19**).  Thresholds
#: taken from the grid therefore pinned both penalties at maximum for every atom of every
#: pose.  Measured on both systems:
#:
#: ============  ==========  ==========  ==========
#: sampling      p50         p5          min
#: ============  ==========  ==========  ==========
#: 3PTB atoms    −9.93       −42.19      −47.29
#: 1M17 atoms    −3.16       −28.27      −30.30
#: 3PTB grid     −1.15       −2.35       −3.08
#: 1M17 grid     −1.05       −2.34       −3.07
#: ============  ==========  ==========  ==========
#:
#: The values below are **3PTB's** (its p50 and p5).  They do **not** transfer to 1M17 —
#: p50 −3.16 and p5 −28.27, a factor of three away — so a two-system calibration needs a
#: per-pocket percentile rather than a module constant, which is not implemented here.
SURFACE_FIELD = -9.93
DEEP_FIELD = -42.19

#: Elements that pay a desolvation penalty even when neutral: burying an amide or a
#: hydroxyl is not free, it is simply cheaper than burying an ion.
POLAR_ELEMENTS = ("N", "O", "S")

#: The charge magnitude that counts as "fully charged" for the polarity factor (e).
CHARGE_UNIT = 0.5
#: What a neutral polar atom contributes relative to a fully charged one (dimensionless).
NEUTRAL_POLAR_FACTOR = 0.35


@dataclass
class PhysicsBreakdown:
    """The two penalties and the score they produce, with the inputs that made them."""

    shape: float = 0.0
    esp_term: float = 0.0
    #: Mean per-atom polarity x burial, in ``[0, 1]``, and the penalty it scales to.
    desolvation: float = 0.0
    desolvation_penalty: float = 0.0
    #: Mean soft-clash ramp over the ligand's atoms, in ``[0, 1]``, and its penalty.
    repulsion: float = 0.0
    repulsion_penalty: float = 0.0
    #: The combined value **before** the penalties (the score the project had) and after.
    without_physics: float = 0.0
    with_physics: float = 0.0
    n_atoms: int = 0
    buried_polar: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": round(float(self.shape), 4),
            "esp_term": round(float(self.esp_term), 4),
            "desolvation": round(float(self.desolvation), 4),
            "desolvation_penalty": round(float(self.desolvation_penalty), 4),
            "repulsion": round(float(self.repulsion), 4),
            "repulsion_penalty": round(float(self.repulsion_penalty), 4),
            "without_physics": round(float(self.without_physics), 4),
            "with_physics": round(float(self.with_physics), 4),
            "n_atoms": int(self.n_atoms),
            "buried_polar": int(self.buried_polar),
        }


def burial_per_atom(coords, radii=None) -> np.ndarray:
    """``1 - min(1, SASA / FULLY_EXPOSED_SASA)``: 1 buried, 0 fully exposed.

    Uses :func:`odock.sasa.sasa_per_atom`, so the burial fraction comes from the surface
    the project already computes rather than from a second, private definition.
    """
    values = np.asarray(coords, dtype=float).reshape(-1, 3)
    if values.size == 0:
        return np.zeros(0, dtype=float)
    surfaces = np.asarray(sasa_per_atom(values, radii), dtype=float).reshape(-1)
    return 1.0 - np.clip(surfaces / float(FULLY_EXPOSED_SASA), 0.0, 1.0)


def polarity_per_atom(elements: Sequence[str], charges: Sequence[float]) -> np.ndarray:
    """How much an atom cares about being buried, in ``[0, 1]``.

    A charged atom is the expensive case (``|q| / CHARGE_UNIT`` capped at 1), a neutral
    N/O/S costs :data:`NEUTRAL_POLAR_FACTOR` of that, and a carbon or a halogen is free.
    This is the per-atom ``k_i`` of a linear SASA term, split so that the *burial* comes
    from geometry and the *polarity* from chemistry.
    """
    labels = [str(element).upper() for element in elements]
    values = np.asarray(charges, dtype=float).reshape(-1)
    out = np.zeros(len(labels), dtype=float)
    for index, label in enumerate(labels):
        charge = abs(float(values[index])) if index < values.size else 0.0
        if charge > 0.0:
            out[index] = min(1.0, charge / float(CHARGE_UNIT))
        elif label in POLAR_ELEMENTS:
            out[index] = float(NEUTRAL_POLAR_FACTOR)
    return out


def desolvation_penalty(
    coords, elements: Sequence[str], charges: Sequence[float], radii=None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """``(mean penalty, per-atom penalty, per-atom burial)``, all in ``[0, 1]``.

    ``Σ_i polarity_i · burial_i / n_atoms`` — the mean, not the sum, so the value does
    not grow with molecule size and can be compared between a fragment and a ligand.
    """
    burial = burial_per_atom(coords, radii)
    polarity = polarity_per_atom(elements, charges)
    count = min(burial.size, polarity.size)
    if count == 0:
        return 0.0, np.zeros(0, dtype=float), burial
    per_atom = polarity[:count] * burial[:count]
    return float(per_atom.mean()), per_atom, burial


def repulsion_penalty(field_values) -> Tuple[float, np.ndarray]:
    """``(mean soft-clash ramp, per-atom ramp)`` over the pocket field, in ``[0, 1]``.

    ``clamp((SURFACE_FIELD - F) / (SURFACE_FIELD - DEEP_FIELD), 0, 1)``: zero at the
    surface, one at :data:`DEEP_FIELD`, **pinned** below it.  The pinning is the whole
    point — an unbounded repulsion would simply replace one saturation with another.
    """
    values = np.asarray(field_values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0, np.zeros(0, dtype=float)
    span = float(SURFACE_FIELD - DEEP_FIELD)
    ramp = np.clip((float(SURFACE_FIELD) - values) / span, 0.0, 1.0)
    return float(ramp.mean()), ramp


def physical_score(
    *,
    shape: float,
    esp_term: float,
    desolvation: float,
    repulsion: float,
    shape_weight: float = 0.5,
    desolvation_weight: float = DESOLVATION_WEIGHT,
    repulsion_weight: float = REPULSION_WEIGHT,
) -> Tuple[float, float]:
    """``(score without the penalties, score with them)``, each in ``[0, 1]``.

    ``without = w·shape + (1-w)·esp_term`` is exactly the score this project had;
    ``with = clamp(without - w_d·desolvation - w_r·repulsion, 0, 1)``.  Reporting both is
    what makes a before/after comparison possible without a second scoring path.
    """
    without = float(shape_weight) * float(shape) + (1.0 - float(shape_weight)) * float(esp_term)
    with_physics = without - float(desolvation_weight) * float(desolvation) \
        - float(repulsion_weight) * float(repulsion)
    return float(without), float(min(1.0, max(0.0, with_physics)))


def search_objective(
    polarity: Optional[Sequence[float]] = None,
    *,
    shape_weight: float = 0.5,
    desolvation_weight: float = DESOLVATION_WEIGHT,
    repulsion_weight: float = REPULSION_WEIGHT,
):
    """A **vectorised** objective for :func:`odock.pocket_score.place_and_score`.

    This is the entry point for the *search*, and it exists because of the trap stated in
    the module header: applied to the winner, a penalty cannot change which pose is found.
    The hook receives the per-pose ``(k, n)`` pocket field values, so both penalties are
    computed **for every candidate pose** with array arithmetic and no second sampling
    pass:

    * **repulsion** — the same bounded ramp,
      `clamp((SURFACE_FIELD − F)/(SURFACE_FIELD − DEEP_FIELD), 0, 1)`, meaned over the
      atoms.  Pinned at :data:`DEEP_FIELD`, so it can subtract at most
      :data:`REPULSION_WEIGHT` and the value cannot saturate.
    * **desolvation** — a **field-based burial proxy**: an atom deep inside the protein
      has a low (negative) pocket field, so `burial ≈ clamp(−F / −DEEP_FIELD, 0, 1)`.
      This is a *search-time stand-in* for the SASA term and it is deliberately cheap: a
      per-pose solvent-accessible surface would cost a full SASA calculation for every
      candidate pose, which is why the SASA form stays the one used for the final
      re-score.  The two are monotone in the same direction — deeper field, more burial —
      and the substitution is stated here rather than hidden.

    ``polarity`` is the per-atom vector from :func:`polarity_per_atom`, aligned with the
    ligand's heavy-atom columns.  Without it every atom counts equally, and a mis-sized
    vector is ignored rather than silently mis-indexed.
    """
    weights = (float(shape_weight), float(desolvation_weight), float(repulsion_weight))
    span = float(SURFACE_FIELD - DEEP_FIELD)

    def objective(shape, esp_term, combined, field_values):
        values = np.asarray(field_values, dtype=float)
        ramp = np.clip((float(SURFACE_FIELD) - values) / span, 0.0, 1.0)
        repulsion = ramp.mean(axis=1)
        burial = np.clip(-values / float(-DEEP_FIELD), 0.0, 1.0)
        desolvation = burial.mean(axis=1)
        if polarity is not None and values.size:
            per_atom = np.asarray(polarity, dtype=float).reshape(1, -1)
            if per_atom.shape[1] == values.shape[1]:
                desolvation = (per_atom * burial).mean(axis=1)
        return np.clip(
            np.asarray(combined, dtype=float)
            - weights[1] * desolvation - weights[2] * repulsion,
            0.0, 1.0,
        )

    return objective


#: The overlap that counts as a hard clash, in Å² of summed vdW interpenetration per
#: ligand atom.  Normalises :func:`overlap_penalty` into ``[0, 1]``.
OVERLAP_REFERENCE = 1.0
#: Overlap begins below this fraction of the summed van der Waals radii.  The term is
#: **flat** for atoms that merely sit near each other — what the pocket field could not
#: express — and rises only past contact.
OVERLAP_THRESHOLD = 0.9


def overlap_penalty(
    ligand_coords,
    ligand_radii,
    receptor_coords,
    receptor_radii,
    *,
    reference: float = OVERLAP_REFERENCE,
    threshold: float = OVERLAP_THRESHOLD,
) -> Tuple[float, np.ndarray]:
    """``(mean penalty, per-atom penalty)`` from **explicit atom-atom overlap**, in ``[0, 1]``.

    The clash term has to be a *different measurement* from burial, and this is it: a soft
    van der Waals interpenetration over ligand-receptor pairs,

        ``overlap_ij = max(0, threshold·(r_i + r_j) − d_ij)``,
        ``penalty_i = min(1, Σ_j overlap_ij / reference)``

    which is **flat while atoms merely sit near each other** and rises steeply once they
    interpenetrate.  The pocket field could not do this: its values at ligand positions run
    to −47, so any ramp on it is monotone in *depth* rather than in overlap — which is
    exactly how the previous version came to reward a shallow pose.  The field keeps its own
    job (shape and enclosure) and never appears here.

    Pairs beyond the summed radii contribute exactly zero, so the term is a sum of positive
    parts and has no cutoff to tune.
    """
    ligand = np.asarray(ligand_coords, dtype=float).reshape(-1, 3)
    radii_ligand = np.asarray(ligand_radii, dtype=float).reshape(-1)
    receptor = np.asarray(receptor_coords, dtype=float).reshape(-1, 3)
    radii_receptor = np.asarray(receptor_radii, dtype=float).reshape(-1)
    if ligand.size == 0:
        return 0.0, np.zeros(0, dtype=float)
    if receptor.size == 0:
        return 0.0, np.zeros(ligand.shape[0], dtype=float)
    squared = ((ligand[:, None, :] - receptor[None, :, :]) ** 2).sum(axis=2)
    touch = float(threshold) * (radii_ligand[:, None] + radii_receptor[None, :])
    overlap = np.clip(touch - np.sqrt(np.maximum(squared, 0.0)), 0.0, None)
    per_atom = np.clip(overlap.sum(axis=1) / float(reference), 0.0, 1.0)
    return float(per_atom.mean()), per_atom


def complex_desolvation_penalty(
    ligand_coords,
    elements: Sequence[str],
    charges: Sequence[float],
    ligand_radii,
    receptor_coords,
    receptor_radii,
) -> Tuple[float, np.ndarray]:
    """``(mean penalty, per-atom penalty)`` from **polar burial in the complex**, ``[0, 1]``.

    ``Σ_i polarity_i · burial_i / n``, where ``burial`` comes from
    :func:`odock.sasa.sasa_per_atom` run on **receptor + ligand together** — the real
    quantity, not a proxy — and ``polarity`` is :func:`polarity_per_atom`, so a buried
    charged nitrogen pays and a buried carbon pays **nothing at all** (polarity 0).

    That is the difference from the previous version, which weighted *depth in the pocket
    field*: a deep carbon was punished, so the term rewarded being shallow and the correct
    pose ended up paying more than the wrong one.  Here **depth never enters** — burial is
    the only geometric input and polarity the only chemical one.
    """
    ligand = np.asarray(ligand_coords, dtype=float).reshape(-1, 3)
    radii_ligand = np.asarray(ligand_radii, dtype=float).reshape(-1)
    receptor = np.asarray(receptor_coords, dtype=float).reshape(-1, 3)
    radii_receptor = np.asarray(receptor_radii, dtype=float).reshape(-1)
    if ligand.size == 0:
        return 0.0, np.zeros(0, dtype=float)
    if receptor.size:
        all_coords = np.vstack([ligand, receptor])
        all_radii = np.concatenate([radii_ligand, radii_receptor])
    else:
        all_coords, all_radii = ligand, radii_ligand
    surfaces = np.asarray(sasa_per_atom(all_coords, all_radii), dtype=float).reshape(-1)
    ligand_surfaces = surfaces[: ligand.shape[0]]
    burial = 1.0 - np.clip(ligand_surfaces / float(FULLY_EXPOSED_SASA), 0.0, 1.0)
    polarity = polarity_per_atom(elements, charges)
    count = min(burial.size, polarity.size)
    if count == 0:
        return 0.0, np.zeros(0, dtype=float)
    per_atom = polarity[:count] * burial[:count]
    return float(per_atom.mean()), per_atom


def breakdown(
    coords,
    elements: Sequence[str],
    charges: Sequence[float],
    radii=None,
    field_values=None,
    *,
    shape: float = 0.0,
    esp_term: float = 0.0,
    shape_weight: float = 0.5,
) -> PhysicsBreakdown:
    """Everything :func:`physical_score` needs, plus the inputs, for a report."""
    desolvation, per_atom, burial = desolvation_penalty(coords, elements, charges, radii)
    repulsion = 0.0
    if field_values is not None:
        repulsion, _ramp = repulsion_penalty(field_values)
    without, with_physics = physical_score(
        shape=shape, esp_term=esp_term, desolvation=desolvation, repulsion=repulsion,
        shape_weight=shape_weight,
    )
    polarity = polarity_per_atom(elements, charges)
    count = min(polarity.size, burial.size)
    buried_polar = int(
        np.sum((polarity[:count] >= 0.5) & (burial[:count] >= 0.5))
    ) if count else 0
    result = PhysicsBreakdown(
        shape=float(shape), esp_term=float(esp_term),
        desolvation=desolvation,
        desolvation_penalty=float(DESOLVATION_WEIGHT) * desolvation,
        repulsion=repulsion, repulsion_penalty=float(REPULSION_WEIGHT) * repulsion,
        without_physics=without, with_physics=with_physics,
        n_atoms=int(count), buried_polar=buried_polar,
    )
    result.notes = [
        "desolvation is a linear SASA-proportional term with a per-element and per-charge "
        "coefficient: the crude standard form, not an implicit-solvent calculation",
        f"repulsion is a bounded soft-clash ramp pinned at {DEEP_FIELD:g} (pocket field), "
        f"so it subtracts at most {REPULSION_WEIGHT:g} and the score cannot saturate",
    ]
    if buried_polar:
        result.notes.append(
            f"{buried_polar} buried polar/charged atom(s): the term this score did not "
            "previously charge for"
        )
    return result

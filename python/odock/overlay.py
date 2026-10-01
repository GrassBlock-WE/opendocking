# SPDX-License-Identifier: GPL-3.0-or-later
"""A real 3-D shape and electrostatic overlay, and the pre-filter that scales it.

The ligand-based benchmark in :mod:`odock.lbvs` needs a shape row that is not a
rigid pose.  This module is that row, and it is built so that every number in it
can be reproduced by hand:

* **Shape** — every heavy atom carries a normalised Gaussian density
  ``exp(-|x - r|² / 2σ²)`` and two molecules are compared by the overlap integral
  ``O_AB = ∫ ρ_A ρ_B dx``.  Because a product of two Gaussians is a Gaussian, that
  integral is analytic:

  ``O_AB = Σ_i Σ_j exp(-|r_i - s_j|² / 4σ²)``  (times ``(πσ²)^{3/2}``, a constant
  that cancels in every reported ratio).

  The reported similarity is the **shape Tanimoto**
  ``O_AB / (O_AA + O_BB - O_AB)`` — the convention ROCS made standard — with the
  **Carbo index** ``O_AB / √(O_AA · O_BB)`` available as ``shape_metric="carbo"``.
  Both are 1.0 for a molecule overlaid on itself and 0.0 for two molecules that do
  not touch, and both are pinned by hand computations in `tests/test_overlay.py`.

* **Electrostatics** — Gasteiger partial charges on the same atoms, giving a field
  ``φ(x) = Σ q_i exp(-|x - r_i|² / 2σ²)``, compared by the Carbo index of the two
  fields (analytic for the same reason).  The raw index is reported in ``[-1, 1]``
  and the *combined* score uses ``max(0, ·)`` exactly as the older crude scorer
  did, so the two can be compared without a changed convention hiding the
  difference.

* **Alignment** — the probe is superposed on the reference by maximising the score
  itself over a **deterministic** set of rotations: the 24 proper rotations of a
  cube, the 24 combinations of the two molecules' principal axes, then a hill
  climb on SO(3) with a halving step.  No random seed, no conformer-specific
  jitter: two runs, two machines, the same number.  Molecules are centred on their
  heavy-atom centroid and only rotation is optimised — a deliberate limit, stated
  rather than hidden.

* **Conformers** — a molecule is scored by its *best* conformer, and the ensemble
  comes from :func:`odock.conformers.build_ensemble`, whose RMSD spread, torsion
  coverage and energy window are carried along with the score.  A molecule that
  already carries 3-D coordinates is used as it is: re-embedding a docked pose
  would silently replace the geometry being scored.

* **The pre-filter** (:func:`usr_prefilter`) — a library of any size can be cut
  down with the 12 USR descriptors (:func:`odock.conformers.usr_descriptors`),
  which cost microseconds per molecule, before any overlay is run.  The function
  reports the **active recall** it costs at the requested fraction, the overlay
  cost it saves (measured on the kept molecules and extrapolated linearly), and
  the docking workload from :func:`odock.screen.estimate_cost`.  A speedup without
  its recall is not a result, so both are always printed together.

**What this is not.**  A shape/electrostatic overlay score is a statement about
geometry, not about binding: it has no receptor, no dielectric, no desolvation,
no entropy and no atom typing beyond Gasteiger.  The ensemble it scores is a
*sample* of a modelled space at one seed, so a score is ``max(score over the
sampled geometries)`` and nothing more.  `docs/CONFORMERS.md` and `docs/LBVS.md`
carry the measured tables and the "what this does not establish" section.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from . import conformers as _conformers

__all__ = [
    "DEFAULT_SIGMA",
    "DEFAULT_CONFORMERS",
    "DEFAULT_SEED",
    "DEFAULT_SHAPE_WEIGHT",
    "DEFAULT_REFINE_LEVELS",
    "DEFAULT_REFINE_STEP",
    "DEFAULT_POPULATION",
    "DEFAULT_TRANSLATION_STEP",
    "DEFAULT_RESTARTS",
    "DEFAULT_ROTATION_SAMPLES",
    "SHAPE_METRICS",
    "ALIGN_OBJECTIVES",
    "OverlayScore",
    "PlacedMolecule",
    "OverlayResult",
    "require_rdkit",
    "gasteiger_charges",
    "heavy_coordinates",
    "gaussian_overlap",
    "shape_tanimoto",
    "carbo_index",
    "prepare_molecule",
    "prepare_library",
    "rigid_overlay",
    "best_overlay",
    "screen",
    "screen_placed",
    "usr_profile",
    "usr_similarity_to",
    "usr_prefilter",
]

#: Gaussian width of one heavy atom's density (Å).  Measured, not assumed: at
#: σ = 0.8 Å the densities are so fat that a flat benzene ring and a puckered
#: cyclohexane chair score 0.989 against each other, and the demo library's
#: nicotinamide outranks the benzamidines on shape alone.  σ = 0.5 Å separates the
#: pairs that must separate (benzene/toluene 0.80, benzene/naphthalene 0.51,
#: benzamidine/nicotinamide 0.885) while a molecule still scores exactly 1.000
#: against itself; `docs/CONFORMERS.md` carries the sweep.
DEFAULT_SIGMA = 0.5
#: Conformer **attempts** per molecule, or ``None`` for the rotor-scaled rule of
#: :func:`odock.conformers.rotor_scaled_attempts`.  ``None`` is the default because
#: the attempt count is the binding constraint on an ensemble (measured: a 4-rotor
#: molecule's torsion coverage rises from 69 % at 16 attempts to 100 % at 64, while
#: the pruning and the energy window remove almost nothing).  An explicit integer is
#: honoured exactly, so a published protocol calling for 2 conformers still gets 2.
DEFAULT_CONFORMERS: Optional[int] = None
#: The fixed ETKDGv3 seed (``odock.conformers.DEFAULT_SEED``).
DEFAULT_SEED = _conformers.DEFAULT_SEED
#: Weight of the shape term in ``combined = w · shape + (1 − w) · max(0, ESP)``.
#: 0.5 matches the older crude scorer, deliberately: the benchmark then measures
#: the *method*, not a re-tuned weighting.
DEFAULT_SHAPE_WEIGHT = 0.5
#: The alignment search's schedule.  It is a **multi-resolution random refinement**:
#: from the best seed poses, ``population`` perturbed poses are drawn at each of
#: ``refine_levels`` levels, the perturbation scale halving every level from
#: ``refine_step`` radians (rotation) and ``translation_step`` Å (translation), and
#: the best accepted.  Every number here is measured against the chemically correct
#: superposition of the benzamidine series in `docs/CONFORMERS.md`; the defaults
#: reach it (benzamidine/hydroxybenzamidine 0.9232 vs 0.923, benzamidine_methyl
#: 0.9238 vs 0.923) in ~10 ms per overlay, where an axis-aligned pattern search of
#: the same cost stalled at 0.861.
DEFAULT_REFINE_LEVELS = 9
DEFAULT_REFINE_STEP = 0.4
DEFAULT_TRANSLATION_STEP = 0.5
DEFAULT_RESTARTS = 2
DEFAULT_POPULATION = 64
#: How many quasi-uniform start rotations :func:`_seed_rotations` adds on top of the
#: 49 structured ones; the whole seed set is one batched kernel evaluation.
DEFAULT_ROTATION_SAMPLES = 256
DEFAULT_ROTATION_SEED = 20240101

#: The shape coefficients the module offers.
SHAPE_METRICS = ("tanimoto", "carbo")

#: What the alignment search maximises.  ``"shape"`` (the default) is deliberate:
#: the electrostatic Carbo index is a noisy objective, and maximising the combined
#: score lets a marginally better field agreement buy a visibly wrong pose —
#: measured, benzamidine/hydroxybenzamidine came out at shape 0.60 when the search
#: maximised the sum, against 0.94 when it maximises shape and the field is read at
#: that pose.  ``"combined"`` is available for comparison and is not the default.
ALIGN_OBJECTIVES = ("shape", "combined")


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for the shape/electrostatic overlay. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# The geometry: analytic Gaussian overlaps
# ---------------------------------------------------------------------------


def heavy_coordinates(mol, conf_id: int = 0) -> np.ndarray:
    """The heavy-atom coordinates of one conformer, as an ``(n, 3)`` array."""
    conf = mol.GetConformer(int(conf_id))
    return np.asarray(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        ],
        dtype=float,
    )


def gasteiger_charges(mol) -> np.ndarray:
    """Gasteiger partial charges, one per atom, in the molecule's own atom order.

    Hydrogens are added on a **copy** when the molecule has none, so the heavy
    atoms keep their indices (RDKit appends the new atoms); the values returned
    therefore line up with :func:`heavy_coordinates`.  A molecule the model cannot
    type comes back as zeros rather than as an exception: the shape term is still
    meaningful, and the electrostatic term then contributes nothing, which the
    caller can see from the reported raw index.
    """
    from rdkit.Chem import AllChem

    work = Chem.Mol(mol)
    try:
        if not any(atom.GetAtomicNum() == 1 for atom in work.GetAtoms()):
            work = Chem.AddHs(work)
        AllChem.ComputeGasteigerCharges(work)
    except Exception:  # pragma: no cover - defensive
        return np.zeros(mol.GetNumAtoms(), dtype=float)
    values: List[float] = []
    for atom in work.GetAtoms():
        try:
            values.append(float(atom.GetDoubleProp("_GasteigerCharge")))
        except Exception:
            values.append(0.0)
    charge = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if charge.shape[0] != mol.GetNumAtoms():  # pragma: no cover - defensive
        return np.zeros(mol.GetNumAtoms(), dtype=float)
    return charge


def heavy_charges(mol, charges: Optional[np.ndarray] = None) -> np.ndarray:
    """The charge of every heavy atom of ``mol``, in :func:`heavy_coordinates` order."""
    values = gasteiger_charges(mol) if charges is None else np.asarray(charges, dtype=float)
    return np.asarray(
        [values[i] for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1],
        dtype=float,
    )


def _gaussian_kernel(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    """``exp(-|a_i - b_j|² / 4σ²)`` for every pair: the whole overlap, as a matrix.

    Written through ``|a − b|² = |a|² + |b|² − 2 a·b`` so the pair distances are one
    matrix product rather than an ``(n, m, 3)`` broadcast — this is the inner loop of
    the rotation search, and it runs ~100 times per overlay.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)
    squared = (
        np.einsum("ij,ij->i", a, a)[:, None]
        + np.einsum("ij,ij->i", b, b)[None, :]
        - 2.0 * (a @ b.T)
    )
    return np.exp(-np.maximum(squared, 0.0) / (4.0 * float(sigma) ** 2))


def gaussian_overlap(a: np.ndarray, b: np.ndarray, sigma: float = DEFAULT_SIGMA) -> float:
    """``O_AB = Σ_i Σ_j exp(-|r_i − s_j|² / 4σ²)``, the Gaussian overlap integral.

    The ``(πσ²)^{3/2}`` prefactor of ``∫ ρ_A ρ_B`` is omitted because it cancels
    in every ratio this module reports.  One atom on itself gives exactly 1.0; one
    atom against another a distance ``d`` away gives ``exp(−d²/4σ²)``; and identical
    coordinate sets give each molecule's self overlap, which is why
    :func:`shape_tanimoto` is exactly 1.0 for a molecule overlaid on itself.
    """
    return float(_gaussian_kernel(np.asarray(a, dtype=float), np.asarray(b, dtype=float), sigma).sum())


def shape_tanimoto(overlap: float, self_a: float, self_b: float) -> float:
    """``O_AB / (O_AA + O_BB − O_AB)``, the overlap Tanimoto, in ``[0, 1]``."""
    denominator = float(self_a) + float(self_b) - float(overlap)
    if denominator <= 0.0:  # pragma: no cover - only when every density is empty
        return 0.0
    return max(0.0, min(1.0, float(overlap) / denominator))


def carbo_index(overlap: float, self_a: float, self_b: float) -> float:
    """``O_AB / √(O_AA · O_BB)``, the Carbo similarity, in ``[−1, 1]``.

    For the shape densities it is always non-negative (the overlap is a sum of
    non-negative Gaussians); for the *charge* fields it can be negative, which is
    a real statement — the two fields point the other way — and is reported
    rather than clipped, with only the combined score clamping it.
    """
    left = float(self_a)
    right = float(self_b)
    if left <= 0.0 or right <= 0.0:
        return 0.0
    return float(overlap) / math.sqrt(left * right)


def _axes(a: np.ndarray) -> np.ndarray:
    """The principal axes of a point set, columns in decreasing-variance order."""
    if a.shape[0] < 2:
        return np.eye(3)
    centred = a - a.mean(axis=0)
    covariance = centred.T @ centred
    values, vectors = np.linalg.eigh(covariance)
    return vectors[:, np.argsort(values)[::-1]]


def _cube_rotations() -> List[np.ndarray]:
    """The 24 proper rotations of a cube, as signed permutation matrices."""
    out: List[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            for row, column in enumerate(perm):
                matrix[row, column] = signs[row]
            if float(np.linalg.det(matrix)) > 0.5:
                out.append(matrix)
    return out


_CUBE_ROTATIONS: Tuple[np.ndarray, ...] = tuple(_cube_rotations())


def _uniform_rotations(count: int, seed: int) -> np.ndarray:
    """A deterministic, quasi-uniform cloud of ``count`` rotations, as ``(k, 3, 3)``.

    Shoemake's quaternion construction from three uniform variates, driven by a
    fixed PCG64 stream: the same seed gives the same cloud on every run, which is
    what keeps an "overlay score" reproducible rather than merely repeatable.
    """
    if count <= 0:
        return np.zeros((0, 3, 3), dtype=float)
    rng = np.random.default_rng(int(seed))
    u = rng.random((int(count), 3))
    root = np.sqrt(1.0 - u[:, 0])
    quaternions = np.stack(
        [
            root * np.sin(2 * math.pi * u[:, 1]),
            root * np.cos(2 * math.pi * u[:, 1]),
            np.sqrt(u[:, 0]) * np.sin(2 * math.pi * u[:, 2]),
            np.sqrt(u[:, 0]) * np.cos(2 * math.pi * u[:, 2]),
        ],
        axis=1,
    )
    x, y, z, w = quaternions.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def _seed_rotations(
    a: np.ndarray, b: np.ndarray, *, samples: int = DEFAULT_ROTATION_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """The starting rotations of the alignment search, as a ``(k, 3, 3)`` array.

    Three parts, in this order: the identity; the 24 proper rotations of a cube
    (every way to permute the coordinate axes); the 24 rotations that map ``b``'s
    principal axes onto ``a``'s; and a quasi-uniform cloud of ``samples`` further
    rotations.  The structured part handles the exact symmetries and the cloud
    handles everything else — measured, a rotation-only search from 49 structured
    seeds scored benzamidine/hydroxybenzamidine at shape 0.833 where the
    chemically correct superposition gives 0.923, and the cloud plus a 6-DOF local
    search is what closes that gap.
    """
    seeds: List[np.ndarray] = [np.eye(3)]
    seeds.extend(_CUBE_ROTATIONS)
    axes_a = _axes(a)
    axes_b = _axes(b)
    for rotation in _CUBE_ROTATIONS:
        candidate = axes_a @ rotation @ axes_b.T
        if float(np.linalg.det(candidate)) > 0.5:
            seeds.append(candidate)
    cloud = _uniform_rotations(int(samples), int(seed))
    if cloud.shape[0]:
        return np.concatenate([np.asarray(seeds), cloud], axis=0)
    return np.asarray(seeds)


def _axis_angle(axis: Tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix for ``angle`` radians about ``axis``."""
    vector = np.asarray(axis, dtype=float)
    vector = vector / float(np.linalg.norm(vector))
    x, y, z = (float(value) for value in vector)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _small_rotations(scale: float, count: int, rng) -> np.ndarray:
    """``count`` rotations of at most ``scale`` radians, as ``(count, 3, 3)``.

    A uniform random axis and a uniform angle in ``[0, scale]``, in one vectorised
    Rodrigues formula.  These are the perturbations the refinement draws at each
    level; drawing them rather than stepping along a fixed axis set is what makes
    the search reach the pose a pattern search walks past.
    """
    axis = rng.normal(size=(int(count), 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = float(scale) * rng.random(int(count))
    x, y, z = axis.T
    skew = np.zeros((int(count), 3, 3), dtype=float)
    skew[:, 0, 1], skew[:, 0, 2] = -z, y
    skew[:, 1, 0], skew[:, 1, 2] = z, -x
    skew[:, 2, 0], skew[:, 2, 1] = -y, x
    return (
        np.eye(3)[None, :, :]
        + np.sin(angle)[:, None, None] * skew
        + (1.0 - np.cos(angle))[:, None, None] * (skew @ skew)
    )


# ---------------------------------------------------------------------------
# The score of one overlay
# ---------------------------------------------------------------------------


@dataclass
class OverlayScore:
    """One shape/electrostatic overlay and the terms it was built from."""

    #: Shape similarity, in ``[0, 1]`` (:func:`shape_tanimoto` or :func:`carbo_index`).
    shape: float = 0.0
    #: Raw electrostatic Carbo index, in ``[-1, 1]`` (0.0 when not computed).
    esp: float = 0.0
    #: ``shape_weight · shape + (1 − shape_weight) · max(0, esp)``.
    combined: float = 0.0
    #: The conformer of the probe that won, and of the reference.
    conformer: int = 0
    reference_conformer: int = 0
    #: Which candidate rotation won, and how many were evaluated in total.
    rotation: int = 0
    n_rotations: int = 0
    #: The reference this score is against (empty for a direct pair).
    reference: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": round(float(self.shape), 4),
            "esp": round(float(self.esp), 4),
            "combined": round(float(self.combined), 4),
            "conformer": int(self.conformer),
            "reference_conformer": int(self.reference_conformer),
            "rotation": int(self.rotation),
            "n_rotations": int(self.n_rotations),
            "reference": self.reference,
        }


def _batch_terms(
    target: np.ndarray,
    target_charges: np.ndarray,
    target_self: float,
    target_self_esp: float,
    probe: np.ndarray,
    probe_charges: np.ndarray,
    probe_self: float,
    probe_self_esp: float,
    rotations: np.ndarray,
    translations: np.ndarray,
    *,
    sigma: float,
    shape_metric: str,
    shape_weight: float,
    electrostatic: bool,
    align_on: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(objective, combined, shape, esp)`` for a whole batch of candidate poses.

    One call evaluates every ``(rotation, translation)`` pair in the batch with
    vectorised linear algebra, which is what makes the alignment affordable: the
    alternative — a Python loop around a per-pose kernel — costs ~50 µs of
    interpreter overhead per pose and dominates the actual arithmetic.

    All three terms are always computed, because the kernel is already paid for and
    because a reader needs to see whether the electrostatic term is carrying the
    ranking or dragging it down.
    """
    moved = np.einsum("kij,nj->kni", rotations, probe) + translations[:, None, :]
    squared = (
        np.einsum("ai,ai->a", target, target)[None, :, None]
        + np.einsum("kni,kni->kn", moved, moved)[:, None, :]
        - 2.0 * np.einsum("ai,kni->kan", target, moved)
    )
    kernel = np.exp(-np.maximum(squared, 0.0) / (4.0 * float(sigma) ** 2))
    overlap = kernel.sum(axis=(1, 2))
    shape = np.clip(
        _shape_ratio(overlap, target_self, probe_self, shape_metric), 0.0, 1.0
    )
    esp = np.zeros_like(shape)
    if target_self_esp > 0.0 and probe_self_esp > 0.0:
        field = np.einsum("a,kab,b->k", target_charges, kernel, probe_charges)
        esp = np.clip(
            field / math.sqrt(target_self_esp * probe_self_esp), -1.0, 1.0
        )
    if electrostatic:
        combined = float(shape_weight) * shape + (1.0 - float(shape_weight)) * np.maximum(
            0.0, esp
        )
    else:
        combined = shape
    objective = shape if align_on == "shape" else combined
    return objective, combined, shape, esp


def _shape_ratio(
    overlap: np.ndarray, self_a: float, self_b: float, shape_metric: str
) -> np.ndarray:
    """Vectorised :func:`shape_tanimoto` / :func:`carbo_index` over an overlap array."""
    if shape_metric == "carbo":
        denominator = math.sqrt(max(self_a, 0.0) * max(self_b, 0.0))
        if denominator <= 0.0:
            return np.zeros_like(overlap)
        return overlap / denominator
    denominator = float(self_a) + float(self_b) - overlap
    return np.where(denominator > 0.0, overlap / np.maximum(denominator, 1e-12), 0.0)


def rigid_overlay(
    target: np.ndarray,
    target_charges: np.ndarray,
    probe: np.ndarray,
    probe_charges: np.ndarray,
    *,
    sigma: float = DEFAULT_SIGMA,
    shape_metric: str = "tanimoto",
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
    electrostatic: bool = True,
    align_on: str = "shape",
    refine_levels: int = DEFAULT_REFINE_LEVELS,
    refine_step: float = DEFAULT_REFINE_STEP,
    translation_step: float = DEFAULT_TRANSLATION_STEP,
    n_restarts: int = DEFAULT_RESTARTS,
    population: int = DEFAULT_POPULATION,
    rotation_samples: int = DEFAULT_ROTATION_SAMPLES,
    seed: int = DEFAULT_ROTATION_SEED,
) -> OverlayScore:
    """The best rigid overlay of ``probe``'s coordinates onto ``target``'s.

    The search is **six-dimensional**, not rotation-only.  Both point sets start
    centred on their own heavy-atom centroid, and then the translation is optimised
    along with the rotation, because the centroid of an asymmetric molecule is not
    the translation the best overlay wants: measured, a para-hydroxyl shifts the
    centroid by ~0.3 Å and a rotation-only search scored benzamidine against
    hydroxybenzamidine at shape 0.833 where the chemically correct superposition
    gives 0.923.

    The poses evaluated are, in order:

    1. the deterministic seed set of :func:`_seed_rotations` — the identity, the 24
       proper rotations of a cube, the 24 principal-axis alignments and
       ``rotation_samples`` quasi-uniform rotations — each at zero translation,
       scored in one batched kernel evaluation;
    2. a **multi-resolution random refinement** from the ``n_restarts`` best seeds:
       at each of ``refine_levels`` levels, ``population`` poses are drawn by
       perturbing the incumbent with a random rotation of at most
       ``refine_step · 2^-level`` radians and a Gaussian translation of scale
       ``translation_step · 2^-level`` Å, and the best improvement is accepted.
       The perturbation stream comes from a fixed PCG64 seed, so the search is
       reproducible: no wall-clock, no thread count, no platform in the answer.

    Only proper rotations are allowed: a molecule is never mirrored onto another,
    which would fit a shape that cannot exist.  This is the function to test by
    hand — it takes coordinates, not molecules, and makes no conformer or charge
    decisions.  ``align_on`` is ``"shape"`` by default (see
    :data:`ALIGN_OBJECTIVES`): the pose that maximises the shape overlap is found
    first and the electrostatic index is *read* there, which is what stops a noisy
    field-agreement term from buying a visibly wrong superposition.
    """
    if shape_metric not in SHAPE_METRICS:
        raise ValueError(f"unknown shape metric {shape_metric!r}; use one of {list(SHAPE_METRICS)}")
    if align_on not in ALIGN_OBJECTIVES:
        raise ValueError(
            f"unknown alignment objective {align_on!r}; use one of {list(ALIGN_OBJECTIVES)}"
        )
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    probe = np.asarray(probe, dtype=float).reshape(-1, 3)
    target_charges = np.asarray(target_charges, dtype=float).reshape(-1)
    probe_charges = np.asarray(probe_charges, dtype=float).reshape(-1)
    if target_charges.shape[0] != target.shape[0] or probe_charges.shape[0] != probe.shape[0]:
        raise ValueError("each coordinate set needs exactly one charge per atom")
    empty = OverlayScore(combined=0.0)
    if target.shape[0] == 0 or probe.shape[0] == 0:
        return empty

    target = target - target.mean(axis=0)
    probe = probe - probe.mean(axis=0)
    target_self = gaussian_overlap(target, target, sigma)
    probe_self = gaussian_overlap(probe, probe, sigma)
    target_self_esp = float(
        (target_charges[:, None] * _gaussian_kernel(target, target, sigma) * target_charges[None, :]).sum()
    )
    probe_self_esp = float(
        (probe_charges[:, None] * _gaussian_kernel(probe, probe, sigma) * probe_charges[None, :]).sum()
    )

    def evaluate(rotations: np.ndarray, translations: np.ndarray):
        return _batch_terms(
            target, target_charges, target_self, target_self_esp,
            probe, probe_charges, probe_self, probe_self_esp,
            rotations, translations,
            sigma=sigma, shape_metric=shape_metric, shape_weight=shape_weight,
            electrostatic=electrostatic, align_on=align_on,
        )

    seeds = _seed_rotations(
        target, probe, samples=int(rotation_samples), seed=int(seed)
    )
    zero = np.zeros((seeds.shape[0], 3))
    seed_objective, seed_combined, seed_shape, seed_esp = evaluate(seeds, zero)
    evaluated = int(seeds.shape[0])
    ranked = np.argsort(-seed_objective)
    best = OverlayScore(
        shape=float(seed_shape[ranked[0]]), esp=float(seed_esp[ranked[0]]),
        combined=float(seed_combined[ranked[0]]), rotation=int(ranked[0]),
        n_rotations=evaluated,
    )
    best_value = float(seed_objective[ranked[0]])
    best_rotation = seeds[ranked[0]]
    rng = np.random.default_rng(int(seed))
    for start in ranked[: max(1, int(n_restarts))]:
        rotation = seeds[start].copy()
        translation = np.zeros(3)
        value = float(seed_objective[start])
        for level in range(max(1, int(refine_levels))):
            scale = float(refine_step) * (0.5 ** level)
            perturbations = _small_rotations(scale, int(population), rng)
            candidates = np.einsum("kij,jl->kil", perturbations, rotation)
            offsets = translation + rng.normal(
                scale=max(0.02, float(translation_step) * (0.5 ** level)),
                size=(int(population), 3),
            )
            objective, combined, shape, esp = evaluate(candidates, offsets)
            evaluated += int(candidates.shape[0])
            pick = int(np.argmax(objective))
            if float(objective[pick]) > value + 1e-12:
                value = float(objective[pick])
                rotation, translation = candidates[pick], offsets[pick]
                if value > best_value + 1e-12:
                    best_value = value
                    best = OverlayScore(
                        shape=float(shape[pick]), esp=float(esp[pick]),
                        combined=float(combined[pick]), rotation=int(start),
                        n_rotations=evaluated,
                    )
    best.n_rotations = evaluated
    return best


# ---------------------------------------------------------------------------
# Molecules: conformers, charges, ensembles
# ---------------------------------------------------------------------------


@dataclass
class PlacedMolecule:
    """A molecule's heavy-atom conformers, charges and ensemble quality."""

    name: str = ""
    mol: Any = None
    #: Heavy-atom coordinates, one ``(n, 3)`` array per kept conformer.
    conformers: List[np.ndarray] = field(default_factory=list)
    #: Gasteiger charge of every heavy atom (conformer-independent).
    charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: The ensemble the conformers came from, when this module built one.
    ensemble: Any = None
    #: ``"built"`` (ETKDGv3 through :mod:`odock.conformers`) or ``"given"``.
    source: str = ""
    seconds: float = 0.0
    error: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def n_conformers(self) -> int:
        return len(self.conformers)

    @property
    def torsion_coverage(self) -> float:
        return float(getattr(self.ensemble, "torsion_coverage", 0.0)) if self.ensemble else 0.0

    def self_overlaps(self, sigma: float = DEFAULT_SIGMA) -> List[float]:
        return [gaussian_overlap(coords, coords, sigma) for coords in self.conformers]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_conformers": int(self.n_conformers),
            "source": self.source,
            "torsion_coverage": round(self.torsion_coverage, 4),
            "seconds": round(float(self.seconds), 4),
            "error": self.error,
            "notes": list(self.notes),
        }


def _mol_name(mol, fallback: str = "ligand") -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def prepare_molecule(
    mol,
    *,
    conformers: Optional[int] = DEFAULT_CONFORMERS,
    seed: int = DEFAULT_SEED,
    name: str = "",
    rebuild: bool = False,
    energy_window: float = _conformers.DEFAULT_ENERGY_WINDOW,
) -> PlacedMolecule:
    """Build (or take) the conformers of one molecule and its heavy-atom charges.

    A molecule that already carries 3-D coordinates is used **as it is** unless
    ``rebuild=True``: re-embedding a docked pose or a crystal structure would
    silently replace the geometry being scored, which is the one thing an overlay
    must not do.  A molecule without coordinates goes through
    :func:`odock.conformers.build_ensemble`, so the ensemble's RMSD spread, torsion
    coverage and energy window come back attached to the score — and with
    ``conformers=None`` (the default) the number of embedding **attempts** is scaled
    with the molecule's rotatable bonds.
    """
    require_rdkit()
    start = time.perf_counter()
    label = str(name) or _mol_name(mol)
    work = Chem.Mol(mol)
    if work.GetNumAtoms() == 0:
        return PlacedMolecule(
            name=label, mol=work, source="given", error="the molecule has no atoms",
            seconds=time.perf_counter() - start,
        )
    charges = heavy_charges(work)
    if work.GetNumConformers() > 0 and not rebuild:
        kept = tuple(range(work.GetNumConformers()))
        if conformers is not None and int(conformers) > 0:
            kept = kept[: int(conformers)]
        coords = [heavy_coordinates(work, index) for index in kept]
        notes = [
            f"{len(kept)} conformer(s) taken from the input coordinates (not re-embedded); "
            "pass rebuild=True to sample an ETKDGv3 ensemble instead"
        ]
        return PlacedMolecule(
            name=label, mol=work, conformers=coords, charges=charges,
            source="given", seconds=time.perf_counter() - start, notes=notes,
        )
    ensemble = _conformers.build_ensemble(
        work, n_conformers=conformers, seed=int(seed), energy_window=energy_window,
        name=label,
    )
    if ensemble.error:
        return PlacedMolecule(
            name=label, mol=ensemble.mol, source="built", error=ensemble.error,
            seconds=time.perf_counter() - start,
        )
    coords = (
        [heavy_coordinates(ensemble.mol, index) for index in ensemble.kept]
        if ensemble.kept else []
    )
    if not coords:
        return PlacedMolecule(
            name=label, mol=ensemble.mol, source="built",
            error="the ensemble kept no conformer",
            seconds=time.perf_counter() - start,
        )
    # The charges are computed on the molecule the ensemble's conformers live in,
    # so heavy-atom order and charge order are the same list.
    charges = heavy_charges(ensemble.mol)
    return PlacedMolecule(
        name=label, mol=ensemble.mol, conformers=coords, charges=charges,
        ensemble=ensemble, source="built", seconds=time.perf_counter() - start,
    )


def best_overlay(
    reference: PlacedMolecule,
    probe: PlacedMolecule,
    *,
    sigma: float = DEFAULT_SIGMA,
    shape_metric: str = "tanimoto",
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
    electrostatic: bool = True,
    align_on: str = "shape",
    refine_levels: int = DEFAULT_REFINE_LEVELS,
    refine_step: float = DEFAULT_REFINE_STEP,
    population: int = DEFAULT_POPULATION,
    rotation_samples: int = DEFAULT_ROTATION_SAMPLES,
    n_restarts: int = DEFAULT_RESTARTS,
    seed: int = DEFAULT_ROTATION_SEED,
    label: str = "",
) -> OverlayScore:
    """The best overlay over every (reference conformer, probe conformer) pair.

    This is the "best-over-conformers" contract in one function: a flexible probe
    is not penalised for the conformer the embedder happened to list first, and the
    winner's indices are reported so a reader can see *which* geometry scored.

    The best pair is chosen by ``combined`` — the score that is actually reported —
    whichever objective the internal alignment maximised, so a probe whose best
    *pose* is shape-driven still wins on the number the library is ranked by.
    """
    best = OverlayScore(reference=label)
    if not reference.conformers or not probe.conformers:
        return best
    total = 0
    for reference_index, reference_coords in enumerate(reference.conformers):
        for probe_index, probe_coords in enumerate(probe.conformers):
            score = rigid_overlay(
                reference_coords, reference.charges,
                probe_coords, probe.charges,
                sigma=sigma, shape_metric=shape_metric, shape_weight=shape_weight,
                electrostatic=electrostatic, align_on=align_on,
                refine_levels=refine_levels, refine_step=refine_step,
                population=population, rotation_samples=rotation_samples,
                n_restarts=n_restarts, seed=seed,
            )
            total += score.n_rotations
            score.n_rotations = total
            if score.combined > best.combined or best.n_rotations == 0:
                score.reference = label
                score.reference_conformer = reference_index
                score.conformer = probe_index
                best = score
    best.reference = label
    return best


# ---------------------------------------------------------------------------
# Ranking a library
# ---------------------------------------------------------------------------


@dataclass
class OverlayResult:
    """A library ranked by the overlay, with the terms and the cost of the run."""

    entries: List[Tuple[str, float]] = field(default_factory=list)
    details: Dict[str, OverlayScore] = field(default_factory=dict)
    #: Per-molecule ensemble quality: conformer count and torsion coverage.
    ensembles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)
    n_library: int = 0
    n_pairs: int = 0
    n_rotation_evaluations: int = 0
    sigma: float = DEFAULT_SIGMA
    shape_metric: str = "tanimoto"
    shape_weight: float = DEFAULT_SHAPE_WEIGHT
    electrostatic: bool = True
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> List[str]:
        return [name for name, _ in self.entries]

    @property
    def scores(self) -> List[float]:
        return [score for _, score in self.entries]

    def rank_of(self, name: str) -> Optional[int]:
        for position, (entry, _) in enumerate(self.entries, start=1):
            if entry == name:
                return position
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_library": int(self.n_library),
            "n_pairs": int(self.n_pairs),
            "n_rotation_evaluations": int(self.n_rotation_evaluations),
            "sigma": round(float(self.sigma), 4),
            "shape_metric": self.shape_metric,
            "shape_weight": round(float(self.shape_weight), 4),
            "electrostatic": bool(self.electrostatic),
            "references": list(self.references),
            "seconds": round(float(self.seconds), 4),
            "entries": [
                [name, round(float(score), 6), self.details[name].as_dict()]
                for name, score in self.entries
                if name in self.details
            ],
            "ensembles": dict(self.ensembles),
            "notes": list(self.notes),
        }

    def table(self, limit: int = 0) -> str:
        rows = self.entries if limit <= 0 else self.entries[: int(limit)]
        lines = [
            f"{'rank':<5}{'name':<30}{'combined':>9}{'shape':>8}{'ESP':>8}{'conf':>6}{'ref':>5}",
            "-" * 5 + "-" * 30 + "-" * 9 + "-" * 8 + "-" * 8 + "-" * 6 + "-" * 5,
        ]
        for rank, (name, score) in enumerate(rows, start=1):
            detail = self.details.get(name)
            shape = detail.shape if detail else float("nan")
            esp = detail.esp if detail else float("nan")
            conformer = detail.conformer if detail else -1
            reference = detail.reference_conformer if detail else -1
            lines.append(
                f"{rank:<5}{name[:29]:<30}{score:>9.3f}{shape:>8.3f}{esp:>8.3f}"
                f"{conformer:>6}{reference:>5}"
            )
        if limit > 0 and len(self.entries) > limit:
            lines.append(f"... and {len(self.entries) - limit} more molecule(s)")
        return "\n".join(lines)


def _library_items(molecules: Sequence[Any]) -> Tuple[List[Any], List[str]]:
    mols: List[Any] = []
    names: List[str] = []
    for index, item in enumerate(molecules):
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            raise ValueError(f"library item {index + 1} is not a parsable molecule")
        mols.append(mol)
        names.append(_mol_name(mol, f"ligand_{index + 1}"))
    return mols, names


def prepare_library(
    molecules: Sequence[Any],
    *,
    conformers: Optional[int] = DEFAULT_CONFORMERS,
    seed: int = DEFAULT_SEED,
    names: Optional[Sequence[str]] = None,
    rebuild: bool = False,
) -> Tuple[List[PlacedMolecule], List[str]]:
    """Prepare every molecule of a library, returning the placed molecules and labels.

    Split out of :func:`screen` so a pre-filter can prepare each molecule **once**,
    use the ensembles for the cheap USR descriptors, and hand the same ensembles to
    the overlay for the molecules that survive.  Re-embedding the survivors would
    double the run's dominant cost and make the measured overlay time meaningless.
    """
    mols, labels = _library_items(molecules)
    if names is not None:
        labels = [str(names[i]) if i < len(names) else labels[i] for i in range(len(labels))]
    placed = [
        prepare_molecule(mol, conformers=conformers, seed=seed, rebuild=rebuild)
        for mol in mols
    ]
    return placed, labels


def screen_placed(
    placed_library: Sequence[PlacedMolecule],
    placed_references: Sequence[PlacedMolecule],
    labels: Sequence[str],
    *,
    leave_one_out: bool = True,
    sigma: float = DEFAULT_SIGMA,
    shape_metric: str = "tanimoto",
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
    electrostatic: bool = True,
    align_on: str = "shape",
    refine_levels: int = DEFAULT_REFINE_LEVELS,
    population: int = DEFAULT_POPULATION,
    rotation_samples: int = DEFAULT_ROTATION_SAMPLES,
    n_restarts: int = DEFAULT_RESTARTS,
    seed: int = DEFAULT_ROTATION_SEED,
    seconds: float = 0.0,
) -> OverlayResult:
    """The overlay ranking itself, over already-prepared molecules."""
    if not placed_references:
        raise ValueError("no active could be prepared, so no overlay is possible")
    reference_names = {item.name for item in placed_references}
    order = {label: index for index, label in enumerate(labels)}
    entries: List[Tuple[str, float]] = []
    details: Dict[str, OverlayScore] = {}
    ensembles: Dict[str, Dict[str, Any]] = {}
    n_pairs = 0
    n_rotations = 0
    failed: List[str] = []
    for label, placed in zip(labels, placed_library):
        ensembles[label] = placed.as_dict()
        if placed.error or not placed.conformers:
            failed.append(f"{label}: {placed.error or 'no conformer'}")
            entries.append((label, 0.0))
            details[label] = OverlayScore(combined=0.0, shape=0.0, esp=0.0)
            continue
        best = OverlayScore()
        for item in placed_references:
            if leave_one_out and label == item.name and label in reference_names:
                continue
            score = best_overlay(
                item, placed, sigma=sigma, shape_metric=shape_metric,
                shape_weight=shape_weight, electrostatic=electrostatic,
                align_on=align_on, refine_levels=refine_levels,
                population=population, rotation_samples=rotation_samples,
                n_restarts=n_restarts, seed=seed, label=item.name,
            )
            n_pairs += 1
            n_rotations += score.n_rotations
            if score.combined > best.combined:
                best = score
        entries.append((label, float(best.combined)))
        details[label] = best
    entries.sort(key=lambda item: (-item[1], order.get(item[0], 0)))
    notes = [
        f"{len(placed_references)} reference active(s), best overlay"
        + (" excluding the molecule itself (leave-one-out)" if leave_one_out else
           " (leaky: a reference can be the molecule itself)"),
        f"shape {shape_metric} at sigma {float(sigma):.2f} A, weight {float(shape_weight):.2f}; "
        + ("electrostatics on (Carbo of the Gasteiger fields, clamped at 0 in the sum)"
           if electrostatic else "electrostatics reported but not combined"),
        f"the alignment maximises {align_on} similarity",
        f"{n_pairs} overlay(s), {n_rotations} rotation evaluations",
    ]
    if failed:
        notes.append(
            f"{len(failed)} molecule(s) could not be prepared and score 0: "
            + "; ".join(failed[:4])
            + (" ..." if len(failed) > 4 else "")
        )
    return OverlayResult(
        entries=entries, details=details, ensembles=ensembles,
        references=[item.name for item in placed_references],
        n_library=len(placed_library), n_pairs=n_pairs, n_rotation_evaluations=n_rotations,
        sigma=float(sigma), shape_metric=shape_metric,
        shape_weight=float(shape_weight), electrostatic=bool(electrostatic),
        seconds=float(seconds), notes=notes,
    )


def screen(
    library: Sequence[Any],
    actives: Sequence[Any],
    *,
    conformers: Optional[int] = DEFAULT_CONFORMERS,
    seed: int = DEFAULT_SEED,
    names: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
    leave_one_out: bool = True,
    sigma: float = DEFAULT_SIGMA,
    shape_metric: str = "tanimoto",
    shape_weight: float = DEFAULT_SHAPE_WEIGHT,
    electrostatic: bool = True,
    align_on: str = "shape",
    refine_levels: int = DEFAULT_REFINE_LEVELS,
    population: int = DEFAULT_POPULATION,
    rotation_samples: int = DEFAULT_ROTATION_SAMPLES,
    n_restarts: int = DEFAULT_RESTARTS,
    seed_rotations: int = DEFAULT_ROTATION_SEED,
    rebuild: bool = False,
) -> OverlayResult:
    """Rank a library by its best overlay on the actives' conformers.

    Every library molecule is scored against every placed active and keeps its
    best score; with ``leave_one_out`` (the default) an active is scored against
    the **other** actives, because an active is in the library and overlaying it on
    itself is a free 1.0.  The shape and electrostatic terms are reported
    separately for every molecule, whether or not they are combined, so a reader
    can see what each term contributed.

    ``reference`` names a single active to overlay onto (the cheap mode); without
    it every placed active is a reference.
    """
    require_rdkit()
    start = time.perf_counter()
    placed_library, labels = prepare_library(
        library, conformers=conformers, seed=seed, names=names, rebuild=rebuild
    )
    placed_references, _ = prepare_library(
        actives, conformers=conformers, seed=seed, rebuild=rebuild
    )
    placed_references = [item for item in placed_references if not item.error]
    if reference:
        chosen = [item for item in placed_references if item.name == reference]
        if not chosen:
            raise ValueError(
                f"the reference {reference!r} is not one of the actives that could be placed"
            )
        placed_references = chosen
    return screen_placed(
        placed_library, placed_references, labels,
        leave_one_out=leave_one_out, sigma=sigma, shape_metric=shape_metric,
        shape_weight=shape_weight, electrostatic=electrostatic, align_on=align_on,
        refine_levels=refine_levels, population=population,
        rotation_samples=rotation_samples, n_restarts=n_restarts,
        seed=seed_rotations, seconds=time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------
# The USR pre-filter
# ---------------------------------------------------------------------------


def usr_profile(placed: PlacedMolecule) -> List[np.ndarray]:
    """The USR descriptor of every kept conformer of a placed molecule.

    The descriptors come from :func:`odock.conformers.usr_descriptors`; a conformer
    whose descriptor is undefined (fewer than two heavy atoms) is dropped, and a
    molecule with no usable conformer returns an empty list and is ranked last.
    """
    out: List[np.ndarray] = []
    if placed.mol is None or placed.ensemble is None:
        # Coordinates taken as given: the descriptor is computed on a copy that
        # carries exactly one conformer at a time.
        for coords in placed.conformers:
            work = Chem.Mol(placed.mol)
            work.RemoveAllConformers()
            conf = Chem.Conformer(work.GetNumAtoms())
            heavy = [
                atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1
            ]
            for slot, index in enumerate(heavy):
                conf.SetAtomPosition(
                    index, (float(coords[slot, 0]), float(coords[slot, 1]), float(coords[slot, 2]))
                )
            work.AddConformer(conf, assignId=True)
            values = _conformers.usr_descriptors(work)
            if values is not None:
                out.append(values)
        return out
    for index in placed.ensemble.kept:
        values = _conformers.usr_descriptors(placed.ensemble.mol, conf_id=int(index))
        if values is not None:
            out.append(values)
    return out


def usr_similarity_to(
    probe: List[np.ndarray], reference: List[np.ndarray]
) -> float:
    """The best USR similarity over the two conformer sets (0.0 when either is empty)."""
    best = 0.0
    for left in probe:
        for right in reference:
            value = float(_conformers.usr_similarity(left, right))
            if value > best:
                best = value
    return best


def usr_prefilter(
    library: Sequence[Any],
    actives: Sequence[Any],
    *,
    keep: float = 0.05,
    conformers: Optional[int] = DEFAULT_CONFORMERS,
    seed: int = DEFAULT_SEED,
    names: Optional[Sequence[str]] = None,
    leave_one_out: bool = True,
    exhaustiveness: int = 8,
    cores: Optional[int] = None,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """Cut a library down with USR descriptors, and measure what the cut cost.

    The pre-filter's job is to avoid the overlay, so it must be *cheap* and the
    only number that makes it legitimate is the **active recall** it costs.  This
    function reports, for the requested fraction:

    * how many molecules survive and how many actives they contain, with the
      actives that were thrown away **named**;
    * the docking workload before and after, from :func:`odock.screen.estimate_cost`
      (imported, never edited — a planning figure, not a measurement);
    * the overlay cost, as the **measured** seconds on the kept molecules
      extrapolated linearly to the full library, because that is the cost the
      pre-filter exists to avoid;
    * how long the descriptors themselves took, per molecule.

    With ``leave_one_out`` an active is scored against the *other* actives.  Without
    it every active matches itself perfectly and the recall is 100 % for free —
    which the returned ``self_match_recall`` shows, so the leak is visible in the
    output and not just in the docs.
    """
    require_rdkit()
    from .screen import estimate_cost

    start = time.perf_counter()
    placed, labels = prepare_library(
        library, conformers=conformers, seed=seed, names=names, rebuild=rebuild
    )
    placed_actives, _ = prepare_library(
        actives, conformers=conformers, seed=seed, rebuild=rebuild
    )
    placed_actives = [item for item in placed_actives if not item.error]
    if not placed_actives:
        raise ValueError("no active could be prepared, so no USR descriptor is available")
    if not 0.0 < float(keep) <= 1.0:
        raise ValueError(f"keep must be in (0, 1], got {keep!r}")

    prepared_seconds = sum(item.seconds for item in placed) + sum(
        item.seconds for item in placed_actives
    )
    descriptor_start = time.perf_counter()
    profiles = [usr_profile(item) for item in placed]
    active_profiles = {item.name: usr_profile(item) for item in placed_actives}
    descriptor_seconds = time.perf_counter() - descriptor_start

    def best_usr(label: str, own: List[np.ndarray], exclude_self: bool) -> float:
        best = 0.0
        for name, others in active_profiles.items():
            if exclude_self and leave_one_out and label == name:
                continue
            value = usr_similarity_to(own, others)
            if value > best:
                best = value
        return best

    values = [
        (label, best_usr(label, own, exclude_self=True), best_usr(label, own, exclude_self=False))
        for label, own in zip(labels, profiles)
    ]
    ranked = sorted(range(len(values)), key=lambda index: (-values[index][1], index))
    # The same ranking with the self-match left in: on a congeneric series it keeps
    # every active for free, and reporting both is the difference between a recall
    # figure and a tautology.
    ranked_self = sorted(range(len(values)), key=lambda index: (-values[index][2], index))
    n_total = len(ranked)
    cut = max(1, int(math.ceil(float(keep) * n_total)))
    kept_indices = ranked[:cut]
    kept_names = {labels[index] for index in kept_indices}
    self_kept_names = {labels[index] for index in ranked_self[:cut]}

    active_labels = {item.name for item in placed_actives}
    active_in_library = [label for label in labels if label in active_labels]
    survived = [label for label in active_in_library if label in kept_names]
    self_survived = [label for label in active_in_library if label in self_kept_names]
    workload = estimate_cost(n_ligands=n_total, exhaustiveness=exhaustiveness, cores=cores)
    workload_kept = estimate_cost(n_ligands=cut, exhaustiveness=exhaustiveness, cores=cores)

    # The overlay cost the pre-filter avoids, measured rather than asserted: the
    # overlay is run on the kept molecules (through the ensembles already built)
    # and scaled by the ratio of library sizes.
    overlay_start = time.perf_counter()
    overlay_result = screen_placed(
        [placed[index] for index in kept_indices], placed_actives,
        [labels[index] for index in kept_indices], leave_one_out=leave_one_out,
    )
    overlay_seconds_kept = time.perf_counter() - overlay_start
    full_overlay_estimate = overlay_seconds_kept * (n_total / cut) if cut else 0.0

    notes = [
        "the docking cost model is odock.screen's own estimate (a fixed per-molecule "
        "cost plus a search cost that grows with exhaustiveness and the torsion "
        "count); it is a planning figure, not a measurement",
        "the pre-filter's own cost is one conformer ensemble per molecule; the overlay "
        "cost it saves is the *measured* overlay time on the kept molecules "
        "extrapolated linearly to the full library, which holds because every "
        "molecule is overlaid against every reference",
        "USR is a rotation-invariant shape summary: it ignores element identity, so a "
        "recall figure here is the price of the descriptor's cheapness, not a claim "
        "about its accuracy",
    ]
    return {
        "method": "usr",
        "keep": float(keep),
        "n_library": n_total,
        "n_kept": cut,
        "fraction_kept": cut / n_total if n_total else 0.0,
        "n_actives": len(active_in_library),
        "actives_kept": len(survived),
        "actives_lost": sorted(set(active_in_library) - set(survived)),
        "active_recall": (len(survived) / len(active_in_library)) if active_in_library else 0.0,
        "self_match_recall": (len(self_survived) / len(active_in_library))
        if active_in_library else 0.0,
        "self_match_actives_kept": len(self_survived),
        "estimated_seconds_full": round(workload, 1),
        "estimated_seconds_kept": round(workload_kept, 1),
        "workload_saved_fraction": (1.0 - workload_kept / workload) if workload > 0 else 0.0,
        "conformers_per_molecule": float(
            np.mean([item.n_conformers for item in placed]) if placed else 0.0
        ),
        "preparation_seconds": round(prepared_seconds, 4),
        "descriptor_seconds": round(descriptor_seconds, 4),
        "overlay_seconds_kept": round(overlay_seconds_kept, 4),
        "overlay_seconds_full_estimate": round(full_overlay_estimate, 2),
        "overlay_saved_fraction": (
            1.0 - overlay_seconds_kept / full_overlay_estimate
            if full_overlay_estimate > 0 else 0.0
        ),
        "overlay_pairs_kept": int(overlay_result.n_pairs),
        "kept_names": [labels[index] for index in kept_indices],
        "seconds": round(time.perf_counter() - start, 4),
        "notes": notes,
    }

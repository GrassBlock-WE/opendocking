# SPDX-License-Identifier: GPL-3.0-or-later
"""Cryptic and transient pockets across an ensemble of receptor conformations.

The reason people look at an ensemble at all is the site that is *not* there in
the structure they happen to have: a cavity that is closed in one conformation
and open in another.  This module finds those, using the existing
:mod:`odock.pocket` detector unchanged, and puts every pocket of every
conformation into the common frame that :mod:`odock.ensemble` builds.

The pipeline:

1. **Detect** with :func:`odock.pocket.find_pockets` on each conformation, with
   the co-crystallised ligand stripped (a holo structure's site is not a cavity
   while the ligand is in it) and every pocket reported in the common frame.
2. **Describe** every pocket with measurements that do not depend on the
   detector's internal bookkeeping: the volume it reports, the lining residues
   (residues with an atom within `lining_radius` of any cavity *point*, mapped
   onto the reference's residue numbering through the sequence alignment), a
   buriedness computed here from the receptor atoms, and a
   :func:`local_free_volume` -- the free volume in a sphere around the pocket's
   centre, which is defined *even where the detector finds no pocket at all*.
   That last one is what makes "closed here, open there" a continuous
   measurement instead of a found/not-found flag.
3. **Match** the pockets across conformations by centroid distance *and* the
   overlap of their lining residues (:func:`match_pockets`), producing one
   :class:`PocketTrack` per cavity.
4. **Measure the detector's own noise** (:func:`detector_noise`): the same
   conformation re-detected with a jittered probe radius and grid spacing.  A
   volume change smaller than that floor is the detector's resolution, not the
   protein's motion, and the report says so.
5. **Rank** the tracks by how many conformations reveal them, whether the
   reference structure reveals them, and how much they change.

What this cannot establish is written down in ``docs/POCKETS.md``: an ensemble
of two crystal structures cannot show that a cavity is druggable, that a ligand
binds there, or that the cavity exists in solution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .pocket import VDW_RADII, Pocket, find_pockets
from .prepare import BoxSpec

__all__ = [
    "DEFAULT_LINING_RADIUS",
    "DEFAULT_LOCAL_RADIUS",
    "DEFAULT_MATCH_RADIUS",
    "DEFAULT_MIN_OVERLAP",
    "LOCAL_SPACING",
    "NOISE_JITTERS",
    "PocketError",
    "PocketObservation",
    "PocketTrack",
    "PocketComparison",
    "NoiseFloor",
    "add_pockets_subparser",
    "buriedness",
    "compute_comparison",
    "detect_pockets",
    "detector_noise",
    "local_free_volume",
    "match_pockets",
    "pocket_atoms",
    "reference_residue_map",
]

PathLike = Union[str, "os.PathLike"]  # noqa: F821 - documentation alias

#: How close a residue's atoms must come to a cavity *point* (not the centroid)
#: to count as lining it.
DEFAULT_LINING_RADIUS = 5.0

#: Radius of the sphere whose free volume is measured around a pocket centre,
#: in every conformation -- the continuous "is it open here" measurement.
DEFAULT_LOCAL_RADIUS = 6.0

#: Centroid distance (Å) at which two pockets from two conformations may be the
#: same cavity.
DEFAULT_MATCH_RADIUS = 4.0

#: Jaccard overlap of the lining residues two pockets must share to be matched.
DEFAULT_MIN_OVERLAP = 0.5

#: Grid spacing (Å) of the local free-volume measurement.  One ångström is the
#: resolution of the trace; the openness floor is measured on the *same* grid, so
#: a relative change between two conformations is unaffected by the choice.
LOCAL_SPACING = 1.0

#: ``(probe, spacing)`` pairs the detector is re-run with to measure its own
#: run-to-run noise on one unchanged structure.
NOISE_JITTERS: Tuple[Tuple[float, float], ...] = (
    (1.3, 1.0), (1.4, 1.0), (1.5, 1.0), (1.4, 0.9), (1.4, 1.1),
)

#: How many cavity points are sampled when measuring buriedness.  A cavity holds
#: hundreds of points and they are strongly correlated, so a sample is enough
#: and keeps the 26-ray test cheap.
_BURIAL_SAMPLE = 64

#: How many times the detector's own measured resolution an openness change has
#: to exceed before it counts as a change at all.  A convention, like the
#: robustness scale in :mod:`odock.ensemble`, and it is the line that separates
#: the ERα result (openness changes of 5-11x the resolution) from the trypsin
#: control (1.03x, i.e. a detector threshold moving, not a cavity opening).
CHANGE_MARGIN = 2.0

#: When a cavity is not found in a conformation, how far its free volume there
#: has to collapse, as a fraction of its free volume where it *is* found, for the
#: absence to be called a closure rather than a detection threshold.  This is
#: what rescues a genuine cryptic site whose detector volume sits near
#: ``--min-volume``: its volume is marginal, its *opening* is not.
CLOSED_FRACTION = 0.2

#: Above this fraction, an absence is not a closure at all: the free volume is
#: still there and the detector simply did not call it a pocket.  Measured on the
#: ERα pair, where one track's free volume is 656 Å³ in the structure that does
#: not reveal it against 565 Å³ in the one that does -- a detector disagreement,
#: not a cryptic site, and a rule that only looked at "was it found" would have
#: reported it as one.
NARROWED_FRACTION = 0.8

#: The 26 cubic directions, the same set :mod:`odock.pocket` uses.
_DIRECTIONS: np.ndarray = np.array(
    [
        (dx, dy, dz)
        for dx in (-1.0, 0.0, 1.0)
        for dy in (-1.0, 0.0, 1.0)
        for dz in (-1.0, 0.0, 1.0)
        if (dx, dy, dz) != (0.0, 0.0, 0.0)
    ],
    dtype=float,
)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True)


class PocketError(RuntimeError):
    """The pocket comparison cannot be made as asked."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = int(code)


# ---------------------------------------------------------------------------
# Geometry: the two measurements this module defines
# ---------------------------------------------------------------------------


def _heavy_atoms(atoms: Sequence[Any]) -> List[Any]:
    return [atom for atom in atoms if not getattr(atom, "is_hydrogen", False)]


def _radii(atoms: Sequence[Any]) -> np.ndarray:
    return np.array(
        [VDW_RADII.get(getattr(atom, "element", "") or "", 1.70) for atom in atoms],
        dtype=float,
    )


def _nearby_atoms(
    points: np.ndarray, atoms: Sequence[Any], reach: float, probe: float
) -> Tuple[np.ndarray, np.ndarray]:
    """The atom centres and inflated radii that can possibly reach `points`.

    An exact pre-filter, not an approximation: an atom whose centre is further
    than ``reach`` from *every* point, plus its own inflated radius, cannot wall a
    ray from any of them or block a probe placed at any of them.  On a 2 000-atom
    receptor this leaves a few hundred, and it is what makes a 26-direction ray
    test per cavity point affordable.
    """
    centres = np.array(
        [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
    ).reshape(-1, 3)
    radii = _radii(atoms) + float(probe)
    if centres.size == 0:
        return centres, radii
    # Bounding sphere of the points, so the filter is one distance test per atom.
    centre = points.mean(axis=0)
    extent = (
        float(np.sqrt(((points - centre) ** 2).sum(axis=1)).max())
        if len(points)
        else 0.0
    )
    distance = np.sqrt(((centres - centre) ** 2).sum(axis=1))
    keep = distance <= extent + float(reach) + radii
    return centres[keep], radii[keep]


def buriedness(
    points: np.ndarray,
    atoms: Sequence[Any],
    *,
    probe: float = 1.4,
    ray_length: float = 5.0,
    sample: int = _BURIAL_SAMPLE,
) -> float:
    """Fraction of the 26 cubic directions from `points` that are walled by protein.

    For every sampled point and every one of the 26 directions, the ray is
    tested against every receptor atom as a sphere of radius
    ``vdw(atom) + probe``: a direction is *walled* when the ray passes within
    that radius of an atom centre within `ray_length` Å.  ``1.0`` means the
    point is completely enclosed; a point on the surface has roughly half its
    directions open.

    This is the continuous quantity the detector's own ``buriedness`` threshold
    is applied to; it is recomputed here because
    :class:`odock.pocket.Pocket` reports the volume, the score and the points,
    but not the per-point burial the score was built from, and a cryptic-pocket
    report has to show *how* open a cavity became, not only that it was found.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0 or not len(atoms):
        return 0.0
    if pts.shape[0] > sample:
        step = max(1, pts.shape[0] // int(sample))
        pts = pts[::step]
    centres, radii = _nearby_atoms(pts, atoms, float(ray_length), float(probe))
    if centres.size == 0:
        return 0.0

    walled = np.zeros(pts.shape[0], dtype=float)
    for direction in _DIRECTIONS:
        # Ray-sphere: project every atom onto the ray, keep the ones in front of
        # the point and within `ray_length`, and ask whether the perpendicular
        # distance is inside the atom's inflated radius.
        delta = centres[None, :, :] - pts[:, None, :]
        along = (delta * direction[None, None, :]).sum(axis=2)
        perpendicular = np.sqrt(
            np.maximum((delta ** 2).sum(axis=2) - along ** 2, 0.0)
        )
        hit = (along > 0.0) & (along <= float(ray_length)) & (perpendicular < radii[None, :])
        walled += hit.any(axis=1).astype(float)
    return float((walled / _DIRECTIONS.shape[0]).mean())


def local_free_volume(
    center: Sequence[float],
    atoms: Sequence[Any],
    *,
    radius: float = DEFAULT_LOCAL_RADIUS,
    probe: float = 1.4,
    spacing: float = LOCAL_SPACING,
) -> float:
    """Free volume (Å³) in a sphere of `radius` around `center`.

    A point counts as free when the probe sphere placed there overlaps no
    receptor atom, which is exactly the detector's own rule; the count is
    multiplied by ``spacing ** 3``.  Unlike "was a pocket found here", this is
    defined for every conformation, so it traces a cavity opening and closing.
    """
    if radius <= 0:
        raise PocketError(f"the local radius must be positive, got {radius}")
    if spacing <= 0:
        raise PocketError(f"the local spacing must be positive, got {spacing}")
    if not len(atoms):
        return 4.0 / 3.0 * np.pi * float(radius) ** 3
    half = int(np.ceil(float(radius) / float(spacing)))
    offsets = np.arange(-half, half + 1, dtype=float) * float(spacing)
    grid = np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    keep = (grid ** 2).sum(axis=1) <= float(radius) ** 2
    centre = np.asarray(center, dtype=float).reshape(3)
    grid = grid[keep] + centre

    centres, radii = _nearby_atoms(grid, atoms, float(radius), float(probe))
    if centres.size == 0:
        return float(grid.shape[0]) * float(spacing) ** 3
    blocked = np.zeros(grid.shape[0], dtype=bool)
    chunk = 2048
    for start in range(0, grid.shape[0], chunk):
        block = grid[start : start + chunk]
        distance = np.sqrt(
            ((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        )
        blocked[start : start + chunk] = (distance < radii[None, :]).any(axis=1)
    return float((~blocked).sum()) * float(spacing) ** 3


def clearest_point(
    points: Sequence[Sequence[float]], atoms: Sequence[Any]
) -> Optional[Tuple[float, float, float]]:
    """The cavity point farthest from any atom: its own "deepest" point.

    A cavity's **centroid** is a poor place to measure openness: a crescent- or
    ring-shaped cavity can have its centroid inside an atom.  The point of
    maximum clearance is inside the free space by construction, it is stable
    under a small conformational change, and it is where a ligand would sit.
    Used as the anchor for :func:`local_free_volume`, so the openness trace
    compares the same place in every conformation.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.size == 0:
        return None
    if not len(atoms):
        return tuple(float(v) for v in pts[0])
    centres = np.array(
        [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
    )
    radii = _radii(atoms)
    clearance = np.full(pts.shape[0], np.inf)
    chunk = 4096
    for start in range(0, pts.shape[0], chunk):
        block = pts[start : start + chunk]
        distance = np.sqrt(
            ((block[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        ) - radii[None, :]
        clearance[start : start + chunk] = distance.min(axis=1)
    best = int(np.argmax(clearance))
    return (float(pts[best, 0]), float(pts[best, 1]), float(pts[best, 2]))


# ---------------------------------------------------------------------------
# Which atoms are the receptor?
# ---------------------------------------------------------------------------


def pocket_atoms(
    conformation: Any,
    *,
    box: Optional[BoxSpec] = None,
    strip: Optional[Sequence[str]] = None,
    keep_hetero: bool = True,
) -> List[Any]:
    """The atoms pocket detection should see: heavy, protein-only.

    Two things are removed that are not part of the receptor a cavity lives in:

    * hydrogens -- the force field's polar hydrogens are irrelevant to a
      geometric cavity probe and only slow it down;
    * a co-crystallised **organic** residue that reaches into `box`: a holo
      structure's binding site is not a cavity while the inhibitor is sitting in
      it, and detecting pockets around the ligand would report the ligand's own
      volume as a pocket.  Ions and small groups are kept, as everywhere else in
      this project.

    Waters are already gone: :meth:`odock.ensemble.Conformation.chain_residues`
    and the reader drop them, and `keep_hetero=False` removes every non-standard
    residue.
    """
    names = {str(name).strip().upper() for name in (strip or ()) if str(name).strip()}
    half = None if box is None else tuple(float(size) / 2.0 for size in box.size)
    centre = None if box is None else tuple(float(v) for v in box.center)
    for residues in conformation.chain_residues().values():
        for residue in residues:
            if residue.is_standard or residue.is_water:
                continue
            heavy = [atom for atom in residue.atoms if not atom.is_hydrogen]
            if len(heavy) < 6:
                continue  # an ion or a small group: part of the site
            if half is None:
                continue
            if any(
                all(abs(coordinate - centre[i]) <= half[i] for i, coordinate in enumerate((a.x, a.y, a.z)))
                for a in heavy
            ):
                names.add(residue.res_name.strip().upper())

    out: List[Any] = []
    for atom in _heavy_atoms(conformation.atoms):
        upper = str(atom.res_name).strip().upper()
        if upper in names:
            continue
        if not keep_hetero and upper not in _STANDARD_NAMES:
            continue
        out.append(atom)
    return out


#: Residue names :mod:`odock.ensemble` treats as protein (imported lazily to keep
#: the two modules independent at import time).
def _standard_names() -> frozenset:
    from .ensemble import CODES

    return frozenset(CODES)


_STANDARD_NAMES = _standard_names()


# ---------------------------------------------------------------------------
# Observations: one pocket of one conformation
# ---------------------------------------------------------------------------


@dataclass
class PocketObservation:
    """One detected pocket of one conformation, in the ensemble's common frame."""

    conformation: str
    index: int
    center: Tuple[float, float, float]
    volume: float
    score: float
    n_points: int
    #: Cavity sample points, in the common frame.
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)
    #: Lining residues with a heavy atom within `lining_radius` of a cavity point.
    lining: List[str] = field(default_factory=list)
    #: The same residues as this conformation spells them (``"ASP351 A"`` may be
    #: residue 351 here and 353 in the reference).
    lining_local: List[str] = field(default_factory=list)
    #: Fraction of the 26 directions walled from the cavity's own points.
    buriedness: float = float("nan")
    #: Free volume in a sphere of `local_radius` around the centre (Å³).
    local_free_volume: float = float("nan")
    #: Lining residues that could not be mapped onto the reference's numbering.
    n_unmapped_lining: int = 0

    @property
    def openness(self) -> float:
        """``1 - buriedness``: how much of the 26-direction shell is solvent."""
        return float("nan") if not np.isfinite(self.buriedness) else 1.0 - self.buriedness

    def as_dict(self, *, points: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "conformation": self.conformation,
            "index": int(self.index),
            "center": [float(v) for v in self.center],
            "volume": self.volume,
            "score": self.score,
            "n_points": int(self.n_points),
            "lining": list(self.lining),
            "lining_local": list(self.lining_local),
            "n_unmapped_lining": int(self.n_unmapped_lining),
            "buriedness": self.buriedness,
            "openness": self.openness,
            "local_free_volume": self.local_free_volume,
        }
        if points:
            data["points"] = np.asarray(self.points, dtype=float).tolist()
        return data


def reference_residue_map(aligned: Any, index: int) -> Dict[Tuple[str, int, str], str]:
    """``{(chain, res_id, res_name) in conformation `index`} -> reference label``.

    Built from the sequence alignment the superposition already computed, so a
    lining residue of a pocket in conformation *i* is named the way the reference
    structure names it.  Without this, comparing "which residues line this
    cavity" between two structures that number their residues differently would
    compare strings, not residues.
    """
    from .ensemble import Alignment

    if index == aligned.reference:
        return {}
    alignment: Alignment = aligned.alignments[index]
    reference = aligned.conformations[aligned.reference]
    conformation = aligned.conformations[index]
    out: Dict[Tuple[str, int, str], str] = {}
    for chain_alignment in alignment.chain_alignments:
        reference_residues = reference.standard_residues(chain_alignment.chain)
        other_residues = conformation.standard_residues(chain_alignment.other_chain)
        for i, j in chain_alignment.pairs():
            if i < len(reference_residues) and j < len(other_residues):
                other = other_residues[j]
                out[other.key] = reference_residues[i].label
    return out


def _lining_residues(
    conformation: Any, points: np.ndarray, radius: float
) -> Tuple[List[str], List[str], int]:
    """Residues within `radius` of any cavity point, in file order."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    local: List[Tuple[str, int, str]] = []
    if pts.size:
        for residues in conformation.chain_residues().values():
            for residue in residues:
                if residue.is_water:
                    continue
                coords = residue.coords(heavy_only=True)
                if not coords.size:
                    continue
                distance = np.sqrt(
                    ((coords[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
                )
                if float(distance.min()) <= float(radius):
                    local.append(residue.key)
    labels = [f"{name}{res_id} {chain}".strip() for chain, res_id, name in local]
    return labels, [f"{chain}:{res_id}:{name}" for chain, res_id, name in local], len(local)


def detect_pockets(
    conformation: Any,
    *,
    spacing: float = 1.0,
    probe: float = 1.4,
    min_volume: float = 50.0,
    max_pockets: int = 40,
    buriedness_threshold: float = 0.55,
    ray_length: float = 5.0,
    box: Optional[BoxSpec] = None,
    strip: Optional[Sequence[str]] = None,
    keep_hetero: bool = True,
    lining_radius: float = DEFAULT_LINING_RADIUS,
    local_radius: float = DEFAULT_LOCAL_RADIUS,
    residue_map: Optional[Dict[Tuple[str, int, str], str]] = None,
    region_center: Optional[Sequence[float]] = None,
    region_radius: float = 0.0,
    atoms: Optional[Sequence[Any]] = None,
) -> List[PocketObservation]:
    """Run the detector on one conformation and describe every pocket.

    Parameters
    ----------
    conformation
        A :class:`odock.ensemble.Conformation`, already in the common frame.
    min_volume, max_pockets, spacing, probe, buriedness_threshold
        Passed to :func:`odock.pocket.find_pockets`.  The defaults here are
        deliberately more generous than the CLI's (50 Å³, 40 pockets) so that
        "no pocket here" means "no cavity above 50 Å³", not "no cavity in the
        top ten"; the report prints the cut-offs it used.
    region_center, region_radius
        When `region_radius` is positive, only pockets whose ``center`` lies
        within it of `region_center` are kept -- the region of interest, so a
        whole-receptor scan does not drown the answer in surface grooves.
    residue_map
        The reference mapping from :func:`reference_residue_map`; ``lining``
        carries reference labels and ``lining_local`` the local ones.
    atoms
        The receptor atoms to detect on; defaults to :func:`pocket_atoms`.
    """
    if atoms is None:
        atoms = pocket_atoms(conformation, box=box, strip=strip, keep_hetero=keep_hetero)
    if not atoms:
        return []
    found = find_pockets(
        atoms,
        spacing=float(spacing),
        probe=float(probe),
        min_volume=float(min_volume),
        max_pockets=int(max_pockets),
        buriedness=float(buriedness_threshold),
        ray_length=float(ray_length),
    )
    out: List[PocketObservation] = []
    for pocket in found:
        centre = tuple(float(v) for v in pocket.center)
        if region_radius and region_center is not None:
            if float(np.linalg.norm(np.asarray(centre) - np.asarray(region_center, dtype=float))) > float(region_radius):
                continue
        points = np.asarray(pocket.points, dtype=float).reshape(-1, 3)
        local, keys, _count = _lining_residues(conformation, points, lining_radius)
        mapped: List[str] = []
        unmapped = 0
        for chain, res_id, name in (item.split(":") for item in keys):
            label = (residue_map or {}).get((chain, int(res_id), name))
            if label is None:
                label = f"{name}{res_id} {chain}".strip()
                unmapped += 1
            mapped.append(label)
        out.append(
            PocketObservation(
                conformation=conformation.label,
                index=int(pocket.index),
                center=centre,
                volume=float(pocket.volume),
                score=float(pocket.score),
                n_points=int(pocket.n_points),
                points=points,
                lining=mapped,
                lining_local=local,
                buriedness=buriedness(
                    points, atoms, probe=float(probe), ray_length=float(ray_length)
                ),
                local_free_volume=local_free_volume(
                    centre, atoms, radius=float(local_radius), probe=float(probe)
                ),
                n_unmapped_lining=int(unmapped),
            )
        )
    # The cap applies *inside* the region of interest: the detector's own
    # `max_pockets` ranks over the whole receptor, so a region would otherwise
    # inherit dozens of small surface grooves that happen to fall in it.
    if region_radius and len(out) > int(max_pockets):
        out = out[: int(max_pockets)]
    return out


# ---------------------------------------------------------------------------
# Matching pockets across conformations
# ---------------------------------------------------------------------------


def _jaccard(first: Sequence[str], second: Sequence[str]) -> float:
    left, right = set(first), set(second)
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(first, dtype=float) - np.asarray(second, dtype=float)))


@dataclass
class PocketTrack:
    """One cavity followed across the conformations of an ensemble.

    ``observations[label]`` holds the pocket this conformation reveals, or
    ``None`` when the detector found nothing there.  The summary properties are
    the answer to "is this pocket cryptic, and how much does it change?":

    ``support``
        fraction of conformations that reveal it at all;
    ``transient``
        revealed by some conformations but not all;
    ``absent_from_reference``
        the reference structure does not reveal it -- the pattern cryptic sites
        are usually described by;
    ``volume_change`` / ``volume_ratio``
        the largest open-to-closed difference, in Å³ and as a ratio;
    ``local_free_change``
        the same comparison on :func:`local_free_volume`, which is defined even
        where no pocket was found, so a cavity that closes *completely* still has
        a number;
    ``lining_reference`` / ``lining_jaccard``
        the residues lining it and how consistent that set is across the
        conformations -- high overlap means "the same residues, moved", low
        overlap means the two pockets only look close by;
    ``stability``
        fraction of the jittered detector settings that reproduce it (see
        :func:`detector_noise`);
    ``changing``
        the change exceeds the detector's own noise floor;
    ``confident`` / ``cryptic``
        the two flags the report uses, defined in :meth:`as_dict`.
    """

    index: int
    observations: Dict[str, Optional[PocketObservation]] = field(default_factory=dict)
    reference_label: str = ""
    #: Detector-volume noise floor measured on the same structures (Å³).  Large:
    #: see :class:`NoiseFloor` -- the detector's absolute volume depends on its
    #: own sub-pocket partition, so it is *not* a quantity to read to 10 Å³.
    noise_volume: float = float("nan")
    #: The floor that matters for the openness trace: how much the local free
    #: volume at the anchor moves when only the probe radius changes.
    local_noise: float = float("nan")
    stability: float = float("nan")
    #: The point the openness trace is measured at (deepest cavity point).
    anchor: Optional[Tuple[float, float, float]] = None
    #: Per-conformation free volume in the sphere around the anchor.
    local_free: Dict[str, float] = field(default_factory=dict)
    #: The detector's cut-off this run used, for the near-threshold check.
    min_volume: float = 50.0

    # -- geometry ---------------------------------------------------------

    @property
    def labels_present(self) -> List[str]:
        return [label for label, obs in self.observations.items() if obs is not None]

    @property
    def labels_absent(self) -> List[str]:
        return [label for label, obs in self.observations.items() if obs is None]

    @property
    def n_found(self) -> int:
        return len(self.labels_present)

    @property
    def n_conformations(self) -> int:
        return len(self.observations)

    @property
    def support(self) -> float:
        return self.n_found / self.n_conformations if self.n_conformations else 0.0

    @property
    def transient(self) -> bool:
        return 0 < self.n_found < self.n_conformations

    @property
    def absent_from_reference(self) -> bool:
        return self.observations.get(self.reference_label) is None

    def volumes(self) -> Dict[str, float]:
        return {
            label: obs.volume
            for label, obs in self.observations.items()
            if obs is not None
        }

    @property
    def volume_max(self) -> float:
        values = list(self.volumes().values())
        return max(values) if values else 0.0

    @property
    def volume_min(self) -> float:
        values = list(self.volumes().values())
        return min(values) if values else 0.0

    @property
    def volume_change(self) -> float:
        return self.volume_max - self.volume_min

    @property
    def volume_ratio(self) -> float:
        smallest = self.volume_min
        return self.volume_max / smallest if smallest > 0 else float("inf")

    @property
    def local_free_change(self) -> float:
        values = [value for value in self.local_free.values() if np.isfinite(value)]
        if len(values) < 2:
            return 0.0
        return float(max(values) - min(values))

    @property
    def local_free_min(self) -> float:
        values = [value for value in self.local_free.values() if np.isfinite(value)]
        return float(min(values)) if values else float("nan")

    @property
    def local_free_max(self) -> float:
        values = [value for value in self.local_free.values() if np.isfinite(value)]
        return float(max(values)) if values else float("nan")

    def centroid_spread(self) -> float:
        centres = [
            obs.center for obs in self.observations.values() if obs is not None
        ]
        if len(centres) < 2:
            return 0.0
        return max(
            _distance(centres[a], centres[b])
            for a in range(len(centres))
            for b in range(a + 1, len(centres))
        )

    def lining_reference(self) -> List[str]:
        """The lining residues of the reference observation, or of the first one."""
        reference = self.observations.get(self.reference_label)
        if reference is not None:
            return list(reference.lining)
        for observation in self.observations.values():
            if observation is not None:
                return list(observation.lining)
        return []

    def lining_union(self) -> List[str]:
        out: List[str] = []
        for observation in self.observations.values():
            if observation is None:
                continue
            for label in observation.lining:
                if label not in out:
                    out.append(label)
        return out

    def lining_jaccard(self) -> float:
        """Mean pairwise Jaccard of the lining sets, over the conformations present."""
        sets = [obs.lining for obs in self.observations.values() if obs is not None]
        if len(sets) < 2:
            return 1.0
        values = [
            _jaccard(sets[a], sets[b])
            for a in range(len(sets))
            for b in range(a + 1, len(sets))
        ]
        return float(np.mean(values))

    @property
    def buriedness_mean(self) -> float:
        values = [
            obs.buriedness
            for obs in self.observations.values()
            if obs is not None and np.isfinite(obs.buriedness)
        ]
        return float(np.mean(values)) if values else float("nan")

    # -- the flags and the ranking ----------------------------------------

    @property
    def changing(self) -> bool:
        """Does the cavity's *openness* move by more than the method's resolution?

        Measured on the local free volume at the anchor -- a quantity this module
        defines and can reproduce -- and not on the detector's volume, whose
        run-to-run noise is 18-57 % because sub-pocket partitioning cuts a cavity
        differently at a different grid spacing (see :class:`NoiseFloor`).  The
        change must exceed :data:`CHANGE_MARGIN` times that resolution: the
        trypsin control's 40 Å³ change against a 39 Å³ resolution is the detector
        threshold moving, not a cavity opening, and a one-ångström grid has no
        business calling it a finding.
        """
        if np.isfinite(self.local_noise):
            return self.local_free_change > CHANGE_MARGIN * float(self.local_noise)
        if not np.isfinite(self.noise_volume):
            return self.volume_ratio > 1.0
        return self.volume_change > float(self.noise_volume)

    @property
    def closure_fraction(self) -> float:
        """Free volume where the cavity is absent over where it is present.

        ``nan`` when the track was found everywhere or nowhere.  ``0`` means the
        space is completely blocked where the detector found nothing; ``1`` means
        the space is just as open there and only the detector's flag changed.
        """
        present = [
            self.local_free.get(label, float("nan")) for label in self.labels_present
        ]
        absent = [self.local_free.get(label, float("nan")) for label in self.labels_absent]
        present = [value for value in present if np.isfinite(value)]
        absent = [value for value in absent if np.isfinite(value)]
        if not present or not absent:
            return float("nan")
        open_volume = max(present)
        if open_volume <= 0:
            return 0.0
        return max(absent) / open_volume

    @property
    def closed_elsewhere(self) -> bool:
        """Is the cavity genuinely *closed* where it was not found?

        True when the free volume in the conformations that do not reveal it
        collapses below :data:`CLOSED_FRACTION` of the free volume in the ones
        that do.  That is the physical statement "the pocket is gone", as opposed
        to "the detector's cut-off moved", and it is what lets a genuine cryptic
        site whose detector volume sits near ``--min-volume`` still count.
        """
        fraction = self.closure_fraction
        return bool(np.isfinite(fraction) and fraction < CLOSED_FRACTION)

    @property
    def presence_artefact(self) -> bool:
        """Is an absence *not* a closure: the free volume is still there?

        The mirror image of :attr:`closed_elsewhere`, and the reason a rule that
        only asked "was a pocket found" would over-call: a track can be missing
        from a conformation whose free volume at the same point is as large as
        the one that revealed it.  That is the detector disagreeing with the
        geometry, and this module reports it as such instead of as a cryptic site.
        """
        fraction = self.closure_fraction
        return bool(np.isfinite(fraction) and fraction >= NARROWED_FRACTION)

    @property
    def narrowed(self) -> bool:
        """Is the cavity partly occluded where it was not found?

        Between :data:`CLOSED_FRACTION` and :data:`NARROWED_FRACTION`: the site
        still holds free volume, but less of it -- a real conformational change
        that is not a cryptic site, reported as its own verdict.
        """
        fraction = self.closure_fraction
        return bool(
            np.isfinite(fraction)
            and CLOSED_FRACTION <= fraction < NARROWED_FRACTION
        )

    @property
    def near_threshold(self) -> bool:
        """Is the smallest volume of this track close to the detector's cut-off?

        A cavity found only because it is a few Å³ above ``--min-volume`` may be
        absent from another conformation simply because it fell below the line.
        The report says so rather than calling it a cryptic site.
        """
        threshold = getattr(self, "min_volume", None)
        if threshold is None or not np.isfinite(self.volume_min):
            return False
        return self.volume_min < 2.0 * float(threshold)

    @property
    def anchored(self) -> bool:
        """Is the openness trace anchored on a real cavity point?"""
        return self.anchor is not None and bool(self.local_free)

    @property
    def confident(self) -> bool:
        """Does the track pass the artefact checks?

        * its presence -- *where it is found* -- is reproduced under the
          jittered detector settings (``stability >= 0.6``);
        * its openness moves by more than :data:`CHANGE_MARGIN` times the
          method's resolution (:attr:`changing`);
        * the cavity is either comfortably above the detector's cut-off
          (``volume_min > 2 x --min-volume``) **or** demonstrably closed where it
          was not found (:attr:`closed_elsewhere`), so an absence is not just a
          cavity falling below the line.
        """
        if np.isfinite(self.stability) and self.stability < 0.6:
            return False
        if self.presence_artefact:
            return False
        if not self.changing:
            return False
        return bool(not self.near_threshold or self.closed_elsewhere)

    @property
    def cryptic(self) -> bool:
        """Transient, or absent from the reference, and passing the checks."""
        return bool(self.confident and (self.transient or self.absent_from_reference))

    @property
    def cryptic_score(self) -> float:
        """A ranking heuristic -- printed parts, no probability.

        ``presence x absence x change``, with

        * ``presence = 0.5 + 0.5 * support`` -- a cavity found once is half as
          credible as one found in every conformation, but a genuinely cryptic
          site is usually found once and is not thrown away for it;
        * ``absence = 2.0`` when the reference structure does not reveal it (the
          classic cryptic-site definition), else ``1.0``;
        * ``change = local_free_change / local_noise``, capped at 20 -- how many
          times the method's own resolution the cavity opens or closes by.

        Every component is in the table, so the number can be recomputed by hand.
        """
        presence = 0.5 + 0.5 * self.support
        absence = 2.0 if self.absent_from_reference else 1.0
        if np.isfinite(self.local_noise) and self.local_noise > 0:
            change = min(self.local_free_change / self.local_noise, 20.0)
        else:
            change = 1.0
        return float(presence * absence * max(change, 0.0))

    def openness_trace(self) -> str:
        """Per conformation: volume, the local free volume, and whether it was found."""
        parts = []
        for label, observation in self.observations.items():
            free = self.local_free.get(label, float("nan"))
            if observation is None:
                parts.append(f"{label}:-- ({free:.0f} A^3 free)" if np.isfinite(free) else f"{label}:--")
            else:
                parts.append(f"{label}:{observation.volume:.0f} ({free:.0f} free)")
        return ", ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "track": int(self.index),
            "reference": self.reference_label,
            "n_found": int(self.n_found),
            "n_conformations": int(self.n_conformations),
            "support": self.support,
            "transient": bool(self.transient),
            "absent_from_reference": bool(self.absent_from_reference),
            "cryptic": bool(self.cryptic),
            "confident": bool(self.confident),
            "changing": bool(self.changing),
            "volume_max": self.volume_max,
            "volume_min": self.volume_min,
            "volume_change": self.volume_change,
            "volume_ratio": None if not np.isfinite(self.volume_ratio) else self.volume_ratio,
            "volume_change_vs_noise": (
                None if not np.isfinite(self.noise_volume) or self.noise_volume <= 0
                else self.volume_change / self.noise_volume
            ),
            "local_free_min": self.local_free_min,
            "local_free_max": self.local_free_max,
            "local_free_change": self.local_free_change,
            "local_noise": self.local_noise,
            "changing": bool(self.changing),
            "closed_elsewhere": bool(self.closed_elsewhere),
            "presence_artefact": bool(self.presence_artefact),
            "narrowed": bool(self.narrowed),
            "closure_fraction": (
                None if not np.isfinite(self.closure_fraction) else self.closure_fraction
            ),
            "anchor": None if self.anchor is None else [float(v) for v in self.anchor],
            "near_threshold": bool(self.near_threshold),
            "centroid_spread": self.centroid_spread(),
            "lining_reference": self.lining_reference(),
            "lining_union": self.lining_union(),
            "lining_jaccard": self.lining_jaccard(),
            "buriedness_mean": self.buriedness_mean,
            "stability": self.stability,
            "noise_volume": self.noise_volume,
            "cryptic_score": self.cryptic_score,
            "observations": {
                label: (None if obs is None else obs.as_dict())
                for label, obs in self.observations.items()
            },
            "local_free_volume": dict(self.local_free),
        }

    def row(self) -> List[str]:
        return [
            str(self.index + 1),
            f"{self.n_found}/{self.n_conformations}",
            "yes" if self.absent_from_reference else "no",
            "yes" if self.transient else "no",
            f"{self.volume_min:.0f}-{self.volume_max:.0f}",
            f"{self.local_free_min:.0f}-{self.local_free_max:.0f}",
            f"{self.lining_jaccard():.2f}",
            f"{self.stability:.2f}" if np.isfinite(self.stability) else "--",
            "yes" if self.cryptic else ("maybe" if self.confident else "no"),
            f"{self.cryptic_score:.2f}",
        ]

    @property
    def score_parts(self) -> str:
        """The cryptic score with its three factors spelled out."""
        presence = 0.5 + 0.5 * self.support
        absence = 2.0 if self.absent_from_reference else 1.0
        if np.isfinite(self.local_noise) and self.local_noise > 0:
            change = min(self.local_free_change / self.local_noise, 20.0)
        else:
            change = 1.0
        return (
            f"presence {presence:.2f} x absence {absence:.1f} x change {change:.2f} "
            f"= {self.cryptic_score:.2f}"
        )


def match_pockets(
    detections: Dict[str, List[PocketObservation]],
    labels: Sequence[str],
    *,
    reference: str,
    match_radius: float = DEFAULT_MATCH_RADIUS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> List[PocketTrack]:
    """Follow each cavity through the conformations that reveal it.

    Two pockets are the same cavity when their centroids are within
    `match_radius` **and** their lining residues overlap by at least
    `min_overlap` Jaccard.  Both conditions are needed: centroids alone confuse
    two ends of a channel, and residue overlap alone confuses two different
    cavities cut out of the same wall of protein.

    The reference's pockets seed the tracks; every later conformation's pockets
    are assigned to the best available track (smallest centroid distance, ties
    broken by larger overlap) and the leftovers start new tracks.  That is a
    greedy assignment, documented rather than hidden: with one pocket per site
    per conformation -- the normal case -- it is exact, and the report prints
    ``centroid_spread`` and ``lining_jaccard`` so a wrong assignment is visible.
    """
    order = [reference] + [label for label in labels if label != reference]
    tracks: List[PocketTrack] = []
    for label in order:
        if label not in detections:
            continue
        # Tracks that already hold an observation of this conformation cannot
        # take another one.
        for pocket in detections[label]:
            best: Optional[Tuple[float, int]] = None
            for position, track in enumerate(tracks):
                if track.observations.get(label) is not None:
                    continue
                neighbours = [
                    obs for obs in track.observations.values() if obs is not None
                ]
                if not neighbours:
                    continue
                nearest = min(neighbours, key=lambda obs: _distance(obs.center, pocket.center))
                distance = _distance(nearest.center, pocket.center)
                if distance > float(match_radius):
                    continue
                overlap = _jaccard(nearest.lining, pocket.lining)
                if overlap < float(min_overlap):
                    continue
                cost = distance + (1.0 - overlap) * float(match_radius)
                if best is None or cost < best[0]:
                    best = (cost, position)
            if best is None:
                track = PocketTrack(index=len(tracks), reference_label=reference)
                for other in order:
                    track.observations.setdefault(other, None)
                track.observations[label] = pocket
                tracks.append(track)
            else:
                tracks[best[1]].observations[label] = pocket
    for position, track in enumerate(tracks):
        track.index = position
    return tracks


# ---------------------------------------------------------------------------
# The detector's own noise
# ---------------------------------------------------------------------------


@dataclass
class NoiseFloor:
    """How much a pocket's volume moves when nothing but the settings change.

    The detector is re-run on **one unchanged structure** with a jittered probe
    radius and grid spacing.  Whatever volume change that produces is the
    resolution of the method: a conformational change smaller than it is not
    evidence of anything.  The same tracks are also what ``stability`` scores.
    """

    conformation: str
    settings: List[Tuple[float, float]] = field(default_factory=list)
    n_pockets: List[int] = field(default_factory=list)
    #: ``|ΔV| / V`` for every track matched across the jittered runs.
    relative_changes: List[float] = field(default_factory=list)
    #: The largest absolute volume change (Å³) of a track across those runs.
    absolute_change: float = float("nan")
    tracks: int = 0

    @property
    def volume_fraction_median(self) -> float:
        return float(np.median(self.relative_changes)) if self.relative_changes else float("nan")

    @property
    def volume_fraction_max(self) -> float:
        return float(max(self.relative_changes)) if self.relative_changes else float("nan")

    @property
    def volume_change(self) -> float:
        """The floor a volume change must exceed, in Å³."""
        return self.absolute_change

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conformation": self.conformation,
            "settings": [[float(probe), float(spacing)] for probe, spacing in self.settings],
            "n_pockets": [int(value) for value in self.n_pockets],
            "tracks": int(self.tracks),
            "relative_change_median": self.volume_fraction_median,
            "relative_change_max": self.volume_fraction_max,
            "absolute_change": self.absolute_change,
        }

    def text(self) -> str:
        return (
            f"detector noise on {self.conformation} (unchanged structure, "
            f"{len(self.settings)} setting(s)): volume moves by at most "
            f"{self.absolute_change:.0f} A^3 "
            f"({100.0 * self.volume_fraction_max:.0f}% of a pocket, "
            f"{self.tracks} track(s), median {100.0 * self.volume_fraction_median:.0f}%)"
        )


def detector_noise(
    conformation: Any,
    *,
    jitters: Sequence[Tuple[float, float]] = NOISE_JITTERS,
    min_volume: float = 50.0,
    max_pockets: int = 40,
    buriedness_threshold: float = 0.55,
    ray_length: float = 5.0,
    box: Optional[BoxSpec] = None,
    strip: Optional[Sequence[str]] = None,
    match_radius: float = DEFAULT_MATCH_RADIUS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    lining_radius: float = DEFAULT_LINING_RADIUS,
    atoms: Optional[Sequence[Any]] = None,
    region_center: Optional[Sequence[float]] = None,
    region_radius: float = 0.0,
) -> NoiseFloor:
    """Measure the detector's run-to-run noise on one unchanged structure."""
    settings = [(float(probe), float(spacing)) for probe, spacing in jitters]
    if not settings:
        return NoiseFloor(conformation=conformation.label)
    keys = {label: index for index, label in enumerate([conformation.label])}
    detections: Dict[str, List[PocketObservation]] = {}
    counts: List[int] = []
    for index, (probe, spacing) in enumerate(settings):
        label = f"{conformation.label}#{index}"
        found = detect_pockets(
            conformation, spacing=spacing, probe=probe, min_volume=min_volume,
            max_pockets=max_pockets, buriedness_threshold=buriedness_threshold,
            ray_length=ray_length, box=box, strip=strip, lining_radius=lining_radius,
            region_center=region_center, region_radius=region_radius, atoms=atoms,
        )
        for observation in found:
            observation.conformation = label
        detections[label] = found
        counts.append(len(found))
    order = list(detections)
    tracks = match_pockets(
        detections, order, reference=order[0], match_radius=match_radius,
        min_overlap=min_overlap,
    )
    del keys
    relative: List[float] = []
    absolute = 0.0
    for track in tracks:
        volumes = list(track.volumes().values())
        if len(volumes) < 2:
            continue
        change = max(volumes) - min(volumes)
        mean = float(np.mean(volumes))
        absolute = max(absolute, change)
        if mean > 0:
            relative.append(change / mean)
    return NoiseFloor(
        conformation=conformation.label, settings=settings, n_pockets=counts,
        relative_changes=relative, absolute_change=absolute, tracks=len(tracks),
    )


def _track_stability(
    track: PocketTrack,
    detections: Dict[str, List[PocketObservation]],
    *,
    match_radius: float,
    min_overlap: float,
) -> Tuple[float, Dict[str, float]]:
    """Under how many of the jittered settings is this track reproduced?

    Returns ``(stability, local_free_volume per setting)``.  A track counts as
    reproduced in a setting when a pocket of that setting matches its *seed*
    observation by the same distance-and-overlap rule used for matching.
    """
    seed = None
    for observation in track.observations.values():
        if observation is not None:
            seed = observation
            break
    if seed is None:
        return float("nan"), {}
    hits = 0
    total = 0
    free: Dict[str, float] = {}
    for label, pockets in detections.items():
        total += 1
        matched = any(
            _distance(pocket.center, seed.center) <= match_radius
            and _jaccard(pocket.lining, seed.lining) >= min_overlap
            for pocket in pockets
        )
        if matched:
            hits += 1
        nearest = min(
            pockets,
            key=lambda pocket: _distance(pocket.center, seed.center),
            default=None,
        )
        if nearest is not None:
            free[label] = nearest.local_free_volume
    return (hits / total if total else float("nan")), free


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


@dataclass
class PocketComparison:
    """Every cavity of every conformation, matched into tracks and described.

    Attributes
    ----------
    tracks
        One :class:`PocketTrack` per cavity, in the order the reference's pockets
        seeded them.
    detections
        ``label -> [PocketObservation]``: the raw detector output per
        conformation, in the common frame.
    noise
        ``label -> NoiseFloor``: the detector's own resolution on that structure.
    parameters
        Every cut-off the run used, so a number can be reproduced exactly.
    """

    labels: List[str]
    reference: str
    tracks: List[PocketTrack] = field(default_factory=list)
    detections: Dict[str, List[PocketObservation]] = field(default_factory=dict)
    noise: Dict[str, NoiseFloor] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    region: Optional[BoxSpec] = None
    warnings: List[str] = field(default_factory=list)
    elapsed: float = 0.0

    @property
    def n_conformations(self) -> int:
        return len(self.labels)

    @property
    def noise_volume(self) -> float:
        """The largest detector noise floor measured, in Å³."""
        values = [
            floor.absolute_change
            for floor in self.noise.values()
            if np.isfinite(floor.absolute_change)
        ]
        return float(max(values)) if values else float("nan")

    def ranked(self, limit: Optional[int] = None) -> List[PocketTrack]:
        """Tracks by descending :attr:`PocketTrack.cryptic_score`."""
        ordered = sorted(
            self.tracks,
            key=lambda track: (-track.cryptic_score, -track.n_found, track.index),
        )
        return ordered if limit is None else ordered[: int(limit)]

    def cryptic(self, limit: Optional[int] = None) -> List[PocketTrack]:
        """Tracks that are transient or absent from the reference, and pass their checks."""
        out = [track for track in self.ranked() if track.cryptic]
        return out if limit is None else out[: int(limit)]

    def confident(self, limit: Optional[int] = None) -> List[PocketTrack]:
        out = [track for track in self.ranked() if track.confident]
        return out if limit is None else out[: int(limit)]

    # -- reporting --------------------------------------------------------

    def _render(self, tracks: Sequence[PocketTrack], limit: int) -> str:
        headers = [
            "track", "found", "not in ref", "transient", "V (A^3)",
            "free V (A^3)", "lining J", "stability", "cryptic", "score",
        ]
        body = [track.row() for track in tracks[: max(1, int(limit))]]
        widths = [len(header) for header in headers]
        for row in body:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * width for width in widths)]
        lines.extend(render(row) for row in body)
        return "\n".join(lines)

    def table(self, limit: int = 20) -> str:
        """Every track, ranked by the cryptic score."""
        return self._render(self.ranked(), limit)

    def cryptic_table(self, limit: int = 20) -> str:
        """Only the tracks the report calls cryptic (or absent from the reference)."""
        return self._render(self.cryptic(), limit)

    def as_dict(self, *, points: bool = False) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "reference": self.reference,
            "n_conformations": self.n_conformations,
            "parameters": dict(self.parameters),
            "region": None if self.region is None else self.region.as_dict(),
            "noise": {label: floor.as_dict() for label, floor in self.noise.items()},
            "noise_volume": self.noise_volume,
            "n_pockets": {
                label: len(pockets) for label, pockets in self.detections.items()
            },
            "tracks": [
                {
                    **track.as_dict(),
                    "observations": {
                        label: (
                            None if observation is None
                            else observation.as_dict(points=points)
                        )
                        for label, observation in track.observations.items()
                    },
                }
                for track in self.tracks
            ],
            "warnings": list(self.warnings),
        }

    def text(self, *, limit: int = 12) -> str:
        """The whole human-readable report."""
        lines = [
            f"OpenDocking pocket ensemble — {self.n_conformations} conformation(s): "
            f"{', '.join(self.labels)} (reference {self.reference})",
            "",
            "detector: "
            + ", ".join(
                f"{key}={value}"
                for key, value in self.parameters.items()
            ),
        ]
        if self.region is not None:
            lines.append(f"region of interest: {self.region}")
        lines.append("")
        for label in self.labels:
            floor = self.noise.get(label)
            if floor is not None:
                lines.append(f"  {label}: {len(self.detections.get(label, []))} pocket(s); {floor.text()}")
            else:
                lines.append(f"  {label}: {len(self.detections.get(label, []))} pocket(s)")
        lines += [
            "",
            f"cavity tracks ({len(self.tracks)}):",
            self.table(limit),
            "",
            "cryptic candidates (transient or absent from the reference, and stable "
            "across the detector's own settings):",
        ]
        candidates = self.cryptic(limit)
        if candidates:
            lines.append(self.cryptic_table(limit))
            lines.append("")
            for track in candidates:
                lines.append(
                    f"  track {track.index + 1}: found in {track.n_found}/"
                    f"{track.n_conformations} conformation(s) "
                    f"({', '.join(track.labels_present)}); "
                    f"volume {track.volume_min:.0f} -> {track.volume_max:.0f} A^3; "
                    f"local free volume {track.local_free_min:.0f} -> "
                    f"{track.local_free_max:.0f} A^3; "
                    f"lining {', '.join(track.lining_reference()[:6])}"
                    + ("" if track.lining_reference() else "(none)")
                )
                lines.append(f"    openness trace: {track.openness_trace()}")
                lines.append(
                    f"    openness change {track.local_free_change:.0f} A^3 vs the "
                    f"method's resolution {track.local_noise:.0f} A^3; lining "
                    f"Jaccard {track.lining_jaccard():.2f}; centroid spread "
                    f"{track.centroid_spread():.2f} A; score {track.score_parts}"
                )
                if track.near_threshold:
                    lines.append(
                        f"    note: the smallest volume ({track.volume_min:.0f} A^3) is "
                        f"within a factor of two of --min-volume "
                        f"({track.min_volume:.0f} A^3): absence in another "
                        "conformation may be this cut-off, not the protein"
                    )
                if track.lining_jaccard() < 0.6 and track.n_found > 1:
                    lines.append(
                        "    note: the lining residues differ between the "
                        "conformations: these may be two different pockets that "
                        "happen to overlap"
                    )
        else:
            lines.append("  (none)")
            lines.append("")
            lines.append(self._why_no_candidates())
        lines += [
            "",
            "reading the volumes: the detector's absolute volume is only "
            "reproducible to the noise below, because it cuts a cavity into "
            "sub-pockets and a different grid spacing cuts it differently. The "
            "presence of a cavity and the local free volume are the quantities to "
            "read; the volume is context.",
            "",
            self.noise_line(),
        ]
        if self.warnings:
            lines += [""] + [f"warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)

    def noise_line(self) -> str:
        """One line per conformation: the detector's measured resolution there."""
        parts = []
        for label in self.labels:
            floor = self.noise.get(label)
            if floor is None:
                continue
            parts.append(
                f"{label}: volume ±{floor.absolute_change:.0f} A^3 "
                f"({100.0 * floor.volume_fraction_median:.0f}% median, "
                f"{len(floor.settings)} setting(s)); "
                f"presence {min(floor.n_pockets)}-{max(floor.n_pockets)} pocket(s)"
            )
        return "detector resolution — " + "; ".join(parts)

    def _why_no_candidates(self) -> str:
        """Say why nothing qualified, so an empty list is informative."""
        if not self.tracks:
            return (
                "  no cavity was found at all in the region searched: check the "
                "region, or lower --min-volume"
            )
        transient = [track for track in self.tracks if track.transient]
        if not transient:
            return (
                "  every cavity was found in every conformation: no transient "
                "site in this ensemble"
            )
        reasons = []
        if all(track.stability < 0.6 for track in transient if np.isfinite(track.stability)):
            reasons.append(
                "their presence is not reproducible under the detector's own "
                "settings (stability < 0.6)"
            )
        if all(not track.changing for track in transient):
            reasons.append(
                "their openness does not change by more than the method's "
                "resolution"
            )
        if all(track.near_threshold for track in transient):
            reasons.append(
                "their volume sits within a factor of two of --min-volume, so "
                "absence elsewhere may be the cut-off"
            )
        if not reasons:
            reasons.append("none of them passed every check")
        return (
            f"  {len(transient)} cavity track(s) are transient, but "
            + "; and ".join(reasons)
            + ". Raise --min-volume for a sharper presence test, or treat these as "
            "marginal."
        )

    def pocket_pdb(self) -> str:
        """Every cavity point as a ``HETATM`` record, for the workbench.

        Each point is written as a dummy atom named after its track, so a viewer
        that loads the aligned receptor plus this file shows exactly which
        conformations reveal which cavities, in the common frame.
        """
        lines: List[str] = ["REMARK  ODOCK ENSEMBLE POCKETS"]
        serial = 0
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        for track in self.tracks:
            chain = alphabet[track.index % len(alphabet)]
            for label, observation in track.observations.items():
                if observation is None:
                    continue
                # The residue name is the conformation the points came from and
                # the chain is the track, so a viewer can colour by cavity and
                # still see which structures revealed it.
                resname = "".join(c for c in str(label).upper() if c.isalnum())[:3] or "PKT"
                for point in np.asarray(observation.points, dtype=float):
                    serial += 1
                    lines.append(
                        f"HETATM{serial:5d}  C   {resname:>3} {chain}"
                        f"{track.index % 9999 + 1:4d}    "
                        f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}  1.00  0.00           C"
                    )
        lines.append("END")
        return "\n".join(lines) + "\n"


def compute_comparison(
    aligned: Any,
    *,
    box: Optional[BoxSpec] = None,
    strip: Optional[Sequence[str]] = None,
    keep_hetero: bool = True,
    spacing: float = 1.0,
    probe: float = 1.4,
    min_volume: float = 50.0,
    max_pockets: int = 40,
    buriedness_threshold: float = 0.55,
    ray_length: float = 5.0,
    lining_radius: float = DEFAULT_LINING_RADIUS,
    local_radius: float = DEFAULT_LOCAL_RADIUS,
    match_radius: float = DEFAULT_MATCH_RADIUS,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    region_radius: float = 0.0,
    noise: bool = True,
    progress: Optional[Any] = None,
) -> PocketComparison:
    """Detect, describe, match and rank the pockets of an aligned ensemble.

    `aligned` is an :class:`odock.ensemble.AlignedEnsemble` -- the conformations
    are already in one frame, which is what makes a centroid comparison between
    them meaningful.  A plain sequence of
    :class:`odock.ensemble.Conformation` objects is accepted too (already in one
    frame), with the first as the reference.

    Returns
    -------
    :class:`PocketComparison`, whose ``text()`` is the report and whose
    ``cryptic()`` is the ranked list of cavities the reference does not show.
    """
    import time

    from .ensemble import AlignedEnsemble, Alignment

    started = time.perf_counter()
    warnings: List[str] = []
    if isinstance(aligned, AlignedEnsemble):
        conformations = list(aligned.conformations)
        labels = list(aligned.labels)
        reference = labels[int(aligned.reference)]
        alignment = aligned
    else:
        conformations = list(aligned)
        labels = [conformation.label for conformation in conformations]
        if not labels:
            raise PocketError("no conformation to analyse")
        reference = labels[0]
        alignment = AlignedEnsemble(
            conformations=conformations,
            alignments=[
                Alignment(
                    label=conformation.label, reference=reference, identity=1.0,
                    overlap=1.0, matched_residues=conformation.n_residues(),
                    common_residues=conformation.n_residues(),
                    reference_frame=(conformation.label == reference),
                )
                for conformation in conformations
            ],
            site=[], reference=0, box=box,
        )
    if not labels:
        raise PocketError("no conformation to analyse")

    mapped: List[Dict[Tuple[str, int, str], str]] = []
    for index in range(len(conformations)):
        if isinstance(aligned, AlignedEnsemble):
            mapped.append(reference_residue_map(alignment, index))
        else:
            mapped.append({})
    if not isinstance(aligned, AlignedEnsemble) and len(conformations) > 1:
        warnings.append(
            "the conformations were compared without a sequence alignment, so the "
            "lining residues of a pocket are reported in each structure's own "
            "numbering: read `lining_jaccard` with that in mind"
        )

    region_center = None
    if region_radius and box is not None:
        region_center = tuple(float(v) for v in box.center)

    atoms: Dict[str, List[Any]] = {}
    detections: Dict[str, List[PocketObservation]] = {}
    for label, conformation, residue_map in zip(labels, conformations, mapped):
        atoms[label] = pocket_atoms(
            conformation, box=box, strip=strip, keep_hetero=keep_hetero
        )
        detections[label] = detect_pockets(
            conformation, spacing=spacing, probe=probe, min_volume=min_volume,
            max_pockets=max_pockets, buriedness_threshold=buriedness_threshold,
            ray_length=ray_length, box=box, strip=strip, keep_hetero=keep_hetero,
            lining_radius=lining_radius, local_radius=local_radius,
            residue_map=residue_map, region_center=region_center,
            region_radius=region_radius, atoms=atoms[label],
        )
        if progress is not None:
            progress(label, len(detections[label]))

    tracks = match_pockets(
        detections, labels, reference=reference, match_radius=match_radius,
        min_overlap=min_overlap,
    )

    # The openness trace: measured at *one* point per track -- the deepest point
    # of its reference cavity -- in every conformation, including the ones where
    # the detector found nothing there.  That is what makes "closed here, open
    # there" a continuous measurement instead of found/not-found, and it is
    # anchored on a point *inside* the cavity rather than on a centroid that can
    # sit in an atom.
    for track in tracks:
        seed = track.observations.get(reference) or next(
            (observation for observation in track.observations.values() if observation),
            None,
        )
        track.min_volume = float(min_volume)
        if seed is None:
            continue
        track.anchor = clearest_point(seed.points, atoms[reference])
        if track.anchor is None:
            track.anchor = tuple(float(v) for v in seed.center)
        for label in labels:
            track.local_free[label] = local_free_volume(
                track.anchor, atoms[label], radius=local_radius, probe=probe
            )
        # The openness floor: the same point, the same structure, only the probe
        # radius jittered.  That is the resolution of the trace.
        values = [
            local_free_volume(track.anchor, atoms[reference], radius=local_radius, probe=jitter)
            for jitter, _spacing in NOISE_JITTERS
        ]
        track.local_noise = float(max(values) - min(values)) if len(values) > 1 else float("nan")

    floors: Dict[str, NoiseFloor] = {}
    jittered: Dict[str, Dict[str, List[PocketObservation]]] = {}
    if noise:
        for label, conformation, residue_map in zip(labels, conformations, mapped):
            per_setting: Dict[str, List[PocketObservation]] = {}
            for index, (jitter_probe, jitter_spacing) in enumerate(NOISE_JITTERS):
                setting = f"{label}#{index}"
                found = detect_pockets(
                    conformation, spacing=jitter_spacing, probe=jitter_probe,
                    min_volume=min_volume, max_pockets=max_pockets,
                    buriedness_threshold=buriedness_threshold, ray_length=ray_length,
                    box=box, strip=strip,
                    keep_hetero=keep_hetero, lining_radius=lining_radius,
                    local_radius=local_radius, residue_map=residue_map,
                    region_center=region_center, region_radius=region_radius,
                    atoms=atoms[label],
                )
                for observation in found:
                    observation.conformation = setting
                per_setting[setting] = found
            jittered[label] = per_setting
            floors[label] = detector_noise(
                conformation, min_volume=min_volume, max_pockets=max_pockets,
                buriedness_threshold=buriedness_threshold, ray_length=ray_length,
                box=box, strip=strip,
                match_radius=match_radius, min_overlap=min_overlap,
                lining_radius=lining_radius, atoms=atoms[label],
                region_center=region_center, region_radius=region_radius,
            )
        floor = max(
            (entry.absolute_change for entry in floors.values() if np.isfinite(entry.absolute_change)),
            default=float("nan"),
        )
    else:
        floor = float("nan")
        warnings.append(
            "the detector's own noise was not measured (--no-noise): a volume "
            "change is not compared against the resolution of the method"
        )

    for track in tracks:
        track.noise_volume = floor
        if noise:
            # Stability: *where the track was found*, is its presence reproduced
            # under the jittered settings?  Averaging over every conformation
            # instead would make a perfectly robust transient cavity score 0.5
            # merely for being absent elsewhere -- which is the observation, not
            # a defect -- so the average is taken over the conformations that
            # reveal it.
            seed = track.observations.get(reference) or next(
                (observation for observation in track.observations.values() if observation),
                None,
            )
            present = track.labels_present
            if seed is not None and present:
                fractions = []
                for label in present:
                    per_setting = jittered.get(label, {})
                    if not per_setting:
                        continue
                    hits = sum(
                        1
                        for pockets in per_setting.values()
                        if any(
                            _distance(pocket.center, seed.center) <= match_radius
                            and _jaccard(pocket.lining, seed.lining) >= min_overlap
                            for pocket in pockets
                        )
                    )
                    fractions.append(hits / len(per_setting))
                track.stability = (
                    float(np.mean(fractions)) if fractions else float("nan")
                )
    return PocketComparison(
        labels=labels,
        reference=reference,
        tracks=tracks,
        detections=detections,
        noise=floors,
        parameters={
            "spacing": float(spacing),
            "probe": float(probe),
            "min_volume": float(min_volume),
            "max_pockets": int(max_pockets),
            "buriedness": float(buriedness_threshold),
            "lining_radius": float(lining_radius),
            "local_radius": float(local_radius),
            "match_radius": float(match_radius),
            "min_overlap": float(min_overlap),
            "region_radius": float(region_radius),
            "noise_jitters": len(NOISE_JITTERS) if noise else 0,
        },
        region=box,
        warnings=warnings,
        elapsed=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# The command line: `odock ensemble pockets`
# ---------------------------------------------------------------------------


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def cmd_ensemble_pockets(args) -> int:
    """``odock ensemble pockets``: which cavities exist in which conformations."""
    from .ensemble import (
        EnsembleError,
        _cli_write_json,
        _ensemble_box,
        _ensemble_inputs,
        align_conformations,
        ligand_coords,
        read_conformations,
    )

    paths = _ensemble_inputs(args)
    box = _ensemble_box(args, paths)
    try:
        conformations = read_conformations(paths, keep_water=bool(args.keep_water))
        site_ligand = None
        if args.site_ligand:
            site_ligand = ligand_coords(
                conformations[int(args.reference)], args.site_ligand
            )
        aligned = align_conformations(
            conformations,
            reference=int(args.reference),
            box=box,
            site=args.site,
            site_ligand=site_ligand,
            site_radius=float(args.site_radius),
            max_site_residues=int(args.max_site_residues),
            atoms=str(args.site_atoms),
            superpose=bool(args.superpose),
            min_identity=float(args.min_identity),
            min_residues=int(args.min_residues),
            min_site_residues=int(args.min_site_residues),
            allow_box_mismatch=True,
        )
        comparison = compute_comparison(
            aligned,
            box=box,
            strip=list(args.strip or []) or None,
            keep_hetero=not args.no_hetero,
            spacing=float(args.grid_spacing),
            probe=float(args.probe),
            min_volume=float(args.min_volume),
            max_pockets=int(args.max_pockets),
            buriedness_threshold=float(args.buriedness),
            lining_radius=float(args.lining_radius),
            local_radius=float(args.local_radius),
            match_radius=float(args.match_radius),
            min_overlap=float(args.min_overlap),
            region_radius=float(args.region_radius),
            noise=not args.no_noise,
            progress=(
                None
                if args.quiet
                else (lambda label, count: _cli_eprint(
                    f"  {label}: {count} pocket(s)"
                ))
            ),
        )
    except (PocketError, EnsembleError) as exc:
        _cli_eprint(f"odock ensemble pockets: error: {exc}")
        return int(getattr(exc, "code", 2))

    if not args.quiet:
        print(comparison.text(limit=int(args.top)))
        print()
    if args.pockets_pdb:
        Path(args.pockets_pdb).write_text(comparison.pocket_pdb(), encoding="utf-8")
        _cli_eprint(f"wrote {args.pockets_pdb}")
    if args.json_out:
        _cli_write_json(args.json_out, comparison.as_dict())
    return 0


def add_pockets_subparser(ensub: Any) -> None:
    """Register ``odock ensemble pockets`` on the ensemble subparsers."""
    from .ensemble import _add_ensemble_arguments

    parser = ensub.add_parser(
        "pockets",
        help="follow cavities across the conformations: transient and cryptic sites",
        description=(
            "Detect pockets in every conformation with the pocket detector, put "
            "them all in the common frame, match them across conformations by "
            "centroid and lining residues, and report the transient and cryptic "
            "sites -- which cavities are open in some structures and closed in "
            "others.  The detector's own run-to-run noise is measured so a change "
            "smaller than the method's resolution is not called a finding, and the "
            "openness of every cavity is traced through every conformation, "
            "including the ones where no cavity was found."
        ),
    )
    _add_ensemble_arguments(parser, allow_box_mismatch=False)
    parser.add_argument("--box", help="box JSON written by `odock box`")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--box-ligand", metavar="RESNAME",
        help="take the box (and the region of interest) from this residue of the "
             "reference structure",
    )
    parser.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    parser.add_argument("--spacing", type=float, default=0.375, help="grid spacing of the search box (Å)")
    parser.add_argument(
        "--grid-spacing", type=float, default=1.0,
        help="spacing of the pocket detector's grid (Å); 0.9-1.1 is the range the "
             "noise floor is measured over",
    )
    parser.add_argument("--probe", type=float, default=1.4, help="probe radius (Å, water)")
    parser.add_argument(
        "--min-volume", type=float, default=50.0,
        help="cavities smaller than this are not reported (Å³); the report calls a "
             "cavity absent when nothing above this threshold is found there",
    )
    parser.add_argument("--max-pockets", type=int, default=40, help="detector output cap")
    parser.add_argument("--buriedness", type=float, default=0.55, help="detector burial threshold")
    parser.add_argument(
        "--lining-radius", type=float, default=DEFAULT_LINING_RADIUS,
        help="how close a residue must come to a cavity point to line it (Å)",
    )
    parser.add_argument(
        "--local-radius", type=float, default=DEFAULT_LOCAL_RADIUS,
        help="radius of the sphere whose free volume traces a cavity opening (Å)",
    )
    parser.add_argument(
        "--match-radius", type=float, default=DEFAULT_MATCH_RADIUS,
        help="centroid distance at which two pockets may be the same cavity (Å)",
    )
    parser.add_argument(
        "--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP,
        help="lining-residue Jaccard two pockets must share to be matched",
    )
    parser.add_argument(
        "--region-radius", type=float, default=0.0,
        help="only report pockets within this distance of the box centre (Å; "
             "0 = the whole receptor)",
    )
    parser.add_argument(
        "--no-noise", action="store_true",
        help="skip the jittered re-runs that measure the detector's own noise and "
             "each track's stability (faster, weaker claims)",
    )
    parser.add_argument(
        "--pockets-pdb", metavar="FILE",
        help="write every cavity point as HETATM records in the common frame, for "
             "the workbench (residue name = conformation, chain = cavity)",
    )
    parser.add_argument("--top", type=int, default=12, help="rows to print")
    parser.add_argument("--json-out", help="write the whole comparison as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no report")
    parser.set_defaults(func=cmd_ensemble_pockets)




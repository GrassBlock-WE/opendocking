# SPDX-License-Identifier: GPL-3.0-or-later
"""Cavities, channels and their apertures: what a watertight surface can answer.

Why this module exists
----------------------
A solvent-accessible surface says where the protein ends and the solvent
begins; it does not say whether a pocket is a **closed cavity**, a **channel**
with two ends, or a **shallow groove** — and those are different chemical
situations. This module classifies the empty space around a structure and
measures the **aperture** of everything that opens to the outside: how many
openings it has, the radius of the narrowest ball that can still pass through
(the bottleneck), and which residues line that bottleneck.

The model
---------
Everything is built on one field, the same probe-centre field the surface uses::

    F(p) = min_i ( |p - c_i| - (r_i + probe) )

so ``A = {F > 0}`` is the set of positions a probe of radius ``probe`` (1.4 Å,
water) may occupy. Two definitions then do all the work:

``A_core``
    the positions where a ball of radius :data:`CORE_RADIUS` (2.6 Å, a probe and
    a methyl group) still fits inside ``A``. A pocket is a *place a ligand-like
    group can sit*, which is why the classification is made on ``A_core`` and
    not on the raw accessible set: every surface dimple is "accessible", and
    none of them is a pocket.
``bottleneck radius``
    for a pocket, the largest radius ``r`` at which its core is still connected
    to the bulk solvent through ``A_r`` (the positions where a ball of radius
    ``r`` fits). Bisection on ``r``: each step is one connected-component
    labelling, and the answer is a geometric property of the structure, not of
    the grid — a test checks it against a cylinder whose bottleneck radius is
    ``R_wall - r_atom - probe`` in closed form.

Classification, with the criterion stated
-----------------------------------------
1. **bulk** — the components of ``A`` that touch a face of the analysis box.
   This is the "bulk solvent" hypothesis, and it is the one assumption that can
   fail: if the box hugs the atoms, a pocket whose mouth lies outside the box
   has nothing to connect to and looks enclosed. :func:`analyse_cavities`
   therefore *guarantees* the box contains a solvent layer
   (``probe + CORE_RADIUS + 2·spacing`` of padding, raised automatically), and
   with ``auto_margin=False`` it reports ``box_too_tight`` and refuses to call
   anything enclosed rather than guessing. Both branches are tested.
2. **enclosed** — a component of ``A`` that is not connected to any face and
   encloses at least :data:`MIN_VOLUME` Å³. A probe of radius 1.4 Å cannot reach
   it at all.
3. **open** — a component of ``A_core`` that is not part of the core's bulk. Its
   volume is the **core volume** (the part a 2.6 Å ball fits in), which is a
   stated lower bound on the space a ligand could occupy, and its aperture is
   measured by the bottleneck radius, the number of openings, and the residues
   around the bottleneck.
4. **shallow** — an open component below :data:`MIN_VOLUME` Å³, i.e. a dent in
   the surface rather than a site.

What this is not
----------------
:attr:`Pocket.geometric_score` is a **geometric heuristic**: a weighted mean of
enclosure, volume and the hydrophobicity of the lining, with the weights stated
in the code because they are a choice and not a fit. It is **not** a
druggability score, it was not trained on anything, it has not been correlated
with ligand efficiency or with experimental binding, and no claim is made that
it predicts whether a compound will bind. It exists so that a table of pockets
can be sorted, and the docs say so wherever it is printed.

Cost
----
A flood fill and a distance transform over a ~500 000-point grid. The module
uses :mod:`scipy.ndimage` for the connected-component labelling and the exact
Euclidean distance transform — this is a real dependency of *this* feature and
:func:`analyse_cavities` raises a documented ``ImportError`` with the pip command
when it is missing, exactly as the preparation code does for RDKit. Measured
costs are in ``docs/VISUALIZATION.md`` §8 and in ``CavityReport.seconds``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .sasa import PROBE_RADIUS, radius_of, residue_key

__all__ = [
    "BOX_MARGIN",
    "CAVITY_SPACING",
    "CORE_RADIUS",
    "MIN_VOLUME",
    "CavityReport",
    "CavitySettings",
    "Pocket",
    "analyse_cavities",
    "geometric_score",
    "probe_field",
    "sphere_fibonacci",
]

#: Default grid spacing in Å for the analysis. Coarser than the surface default
#: because a flood fill is over *volume* rather than area, and because a channel
#: has to be resolved by two or three cells to percolate. The cost table in the
#: docs is measured at this setting.
CAVITY_SPACING = 0.6

#: Padding around the atoms before the box faces are used as "the outside", in Å.
#: It only has to hold a solvent layer; the guarantee that makes the bulk
#: hypothesis hold is computed in :func:`analyse_cavities`.
BOX_MARGIN = 3.0

#: The radius of the largest ball that must fit for a place to count as a
#: pocket rather than a dimple, in Å (a water probe plus a methyl group).
CORE_RADIUS = 2.6

#: The smallest core volume worth reporting as a pocket, in Å³. About a methyl
#: group's worth of space, so a groove between two helices does not become a
#: "site".
MIN_VOLUME = 25.0

#: The largest bottleneck radius worth searching for, in Å. Above this the
#: "channel" is a gap between domains rather than an aperture.
MAX_BOTTLENECK = 6.0

#: Bisection steps for the bottleneck radius. Each step is one labelling, so the
#: precision is (MAX_BOTTLENECK - CORE_RADIUS)/2**steps Å.
BOTTLENECK_STEPS = 7


class _Missing:
    """A stand-in so the module imports without scipy and fails with advice."""

    def __getattr__(self, name: str):  # pragma: no cover - the error path
        raise ImportError(
            "the cavity analysis needs scipy for the connected-component "
            "labelling and the Euclidean distance transform; install it with "
            "`pip install scipy` (or `pip install opendocking[analysis]`)"
        )


try:  # pragma: no cover - exercised by the environment, not by a branch
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - a numpy-only installation
    _ndimage = _Missing()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# settings and results
# ---------------------------------------------------------------------------


@dataclass
class CavitySettings:
    """Everything that decides what counts as a pocket."""

    #: Grid spacing in Å; ``0`` picks one from the point budget.
    spacing: float = CAVITY_SPACING
    probe: float = PROBE_RADIUS
    #: The ball that has to fit for a place to be a pocket.
    core_radius: float = CORE_RADIUS
    #: Padding around the atoms, in Å, before the box faces are "outside".
    margin: float = BOX_MARGIN
    #: Raise the margin automatically to the value that makes the bulk
    #: hypothesis hold. With this off, a box that is too tight is *reported*
    #: (``CavityReport.box_too_tight``) and nothing is called enclosed.
    auto_margin: bool = True
    min_volume: float = MIN_VOLUME
    #: Cap on the number of pockets reported, largest first.
    max_pockets: int = 12
    max_points: int = 2_000_000
    #: How far the LIGSITE scan looks for protein on each side of a cell, in Å.
    #: Large enough to span a pocket (a mouth narrower than this is found), small
    #: enough that a shallow dent on a convex surface is not.
    enclosure_radius: float = 10.0
    #: How many of the six scan directions must see protein for a cell to count
    #: as inside a pocket. Five of six is the LIGSITE convention: a cell is in
    #: the pocket when every direction but the mouth is blocked. Two would fill
    #: the whole surface with "pockets"; six would demand a sealed hole and
    #: would miss every open site.
    min_directions: int = 5
    #: A mouth at least this wide makes a pocket a "shallow" groove rather than
    #: a site, in Å.
    wide_mouth: float = 4.0
    #: Search range of the bottleneck bisection, in Å.
    max_bottleneck: float = MAX_BOTTLENECK


@dataclass
class Pocket:
    """One cavity, channel or dent, with the geometry that classifies it."""

    label: str
    kind: str
    #: For an enclosed void the whole accessible volume; for an open pocket the
    #: *core* volume (see the module docstring). Å³.
    volume: float
    points: int
    centre: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    #: Number of separate openings to the bulk solvent (0 for enclosed).
    openings: int
    #: Radius of the narrowest ball that still passes through the aperture, Å.
    #: ``None`` for an enclosed void.
    bottleneck_radius: Optional[float]
    #: Residues lining the bottleneck (empty for an enclosed void).
    bottleneck_residues: List[str] = field(default_factory=list)
    #: Residues lining the pocket.
    lining_residues: List[str] = field(default_factory=list)
    #: Mean Kyte-Doolittle hydropathy of the lining atoms, on ``[0, 1]``.
    hydrophobicity: float = 0.0
    #: ``1 - bottleneck/core_radius`` for an open pocket (1.0 when enclosed):
    #: how closed the aperture is *relative to the core that defines the
    #: pocket*. A geometric ratio, not a physical quantity.
    enclosure: float = 0.0
    #: A weighted mean of enclosure, volume and lining hydrophobicity, with the
    #: weights stated in :func:`geometric_score`. **Not a druggability score.**
    geometric_score: float = 0.0
    #: The atoms nearest to the pocket's core, for a caller that wants to draw
    #: or select them.
    atom_indices: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "volume": round(float(self.volume), 1),
            "points": int(self.points),
            "centre": [round(float(v), 2) for v in self.centre],
            "extent": [round(float(v), 2) for v in self.extent],
            "openings": int(self.openings),
            "bottleneck_radius": (
                None if self.bottleneck_radius is None else round(float(self.bottleneck_radius), 2)
            ),
            "bottleneck_residues": list(self.bottleneck_residues),
            "lining_residues": list(self.lining_residues),
            "hydrophobicity": round(float(self.hydrophobicity), 3),
            "enclosure": round(float(self.enclosure), 3),
            "geometric_score": round(float(self.geometric_score), 3),
        }


@dataclass
class CavityReport:
    """The pockets of one structure, and the numbers behind the classification."""

    pockets: List[Pocket] = field(default_factory=list)
    spacing: float = CAVITY_SPACING
    probe: float = PROBE_RADIUS
    core_radius: float = CORE_RADIUS
    grid: Tuple[int, int, int] = (0, 0, 0)
    grid_points: int = 0
    box: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    margin: float = BOX_MARGIN
    #: The number of box faces the bulk solvent touches (6 when all of them).
    bulk_faces: int = 0
    #: True when the box could not be shown to contain a solvent layer, so no
    #: component is called "enclosed" (they are reported as "unknown").
    box_too_tight: bool = False
    #: Volume of the largest connected accessible region, in Å³.
    bulk_volume: float = 0.0
    accessible_volume: float = 0.0
    #: How many cells the enclosure scan accepted, before clustering.
    enclosing_cells: int = 0
    #: ``{directions covered: cell count}`` over the grid, so the criterion's
    #: behaviour on a structure is inspectable rather than assumed.
    direction_histogram: Dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0
    #: How many labellings and how long the aperture search took, in seconds.
    aperture_seconds: float = 0.0

    def of_kind(self, kind: str) -> List[Pocket]:
        return [pocket for pocket in self.pockets if pocket.kind == kind]

    @property
    def enclosed_volume(self) -> float:
        return sum(pocket.volume for pocket in self.of_kind("enclosed"))

    def table(self, limit: int = 12) -> str:
        rows = [
            "pocket      kind       vol Å³  open  neck Å  score  centre               lining"
        ]
        for pocket in self.pockets[:limit]:
            neck = "  —  " if pocket.bottleneck_radius is None else f"{pocket.bottleneck_radius:5.2f}"
            lining = ", ".join(pocket.lining_residues[:4])
            rows.append(
                f"{pocket.label:<11} {pocket.kind:<9} {pocket.volume:>7.1f} "
                f"{pocket.openings:>4}  {neck}  {pocket.geometric_score:>5.2f}  "
                f"({pocket.centre[0]:>5.1f},{pocket.centre[1]:>6.1f},"
                f"{pocket.centre[2]:>6.1f})  {lining}"
            )
        return "\n".join(rows)

    def as_dict(self) -> Dict[str, object]:
        return {
            "spacing": float(self.spacing),
            "probe": float(self.probe),
            "core_radius": float(self.core_radius),
            "grid": [int(size) for size in self.grid],
            "grid_points": int(self.grid_points),
            "box": [[float(v) for v in self.box[0]], [float(v) for v in self.box[1]]],
            "margin": float(self.margin),
            "bulk_faces": int(self.bulk_faces),
            "box_too_tight": bool(self.box_too_tight),
            "bulk_volume": round(float(self.bulk_volume), 1),
            "accessible_volume": round(float(self.accessible_volume), 1),
            "enclosing_cells": int(self.enclosing_cells),
            "direction_histogram": {
                int(key): int(value) for key, value in self.direction_histogram.items()
            },
            "seconds": round(float(self.seconds), 3),
            "aperture_seconds": round(float(self.aperture_seconds), 3),
            "pockets": [pocket.as_dict() for pocket in self.pockets],
        }


def geometric_score(enclosure: float, volume: float, hydrophobicity: float) -> float:
    """A sorted-table heuristic, and nothing more.

    ``0.45·enclosure + 0.35·min(1, volume/400) + 0.20·hydrophobicity``.

    The weights are a *choice*: they are stated here so the number can be
    argued with, and they were not fitted to any data set. 400 Å³ is the usual
    order of magnitude for a small-molecule binding site. This is **not** a
    druggability score, it does not predict binding, and the docs repeat that
    wherever the column is printed.
    """
    clamped = min(1.0, max(0.0, float(volume) / 400.0))
    return float(
        0.45 * min(1.0, max(0.0, enclosure))
        + 0.35 * clamped
        + 0.20 * min(1.0, max(0.0, hydrophobicity))
    )


# ---------------------------------------------------------------------------
# the field and the box
# ---------------------------------------------------------------------------


def sphere_fibonacci(count: int) -> np.ndarray:
    """``(count, 3)`` unit vectors on a golden-spiral lattice.

    The same sampling the solvent-accessible surface uses, re-exported here so a
    caller (and the test suite) can build a spherical shell of atoms without
    importing the viewer.
    """
    from .sasa import sphere_points

    return sphere_points(int(count))


def _atom_arrays(atoms) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(
        [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
    ).reshape(-1, 3)
    return coords, np.asarray([radius_of(getattr(a, "element", None)) for a in atoms])


def probe_field(
    axes: Sequence[np.ndarray],
    coords: np.ndarray,
    inflated: np.ndarray,
    clamp: float,
) -> np.ndarray:
    """``min_i(|p - c_i| - s_i)`` on a regular grid, one atom block at a time.

    The same field, and the same block-fill strategy, as the surface builder —
    expressed here rather than imported from ``odock.gui.surface`` so that a
    command-line cavity analysis never imports the viewer's package.
    ``tests/test_cavity.py`` asserts the two implementations agree exactly, which
    is what makes the duplication safe.
    """
    shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    spacing = float(axes[0][1] - axes[0][0]) if shape[0] > 1 else 1.0
    out = np.full(shape, clamp, dtype=float)
    if coords.shape[0] == 0:
        return out
    search = float(np.max(inflated)) + clamp
    for atom in range(coords.shape[0]):
        centre = coords[atom]
        window = []
        starts = []
        for axis in range(3):
            values = axes[axis]
            low = int(np.searchsorted(values, centre[axis] - search, side="left"))
            high = int(np.searchsorted(values, centre[axis] + search, side="right"))
            low = max(0, min(low, shape[axis]))
            high = max(low, min(high, shape[axis]))
            starts.append(low)
            window.append(values[low:high])
        if any(part.size == 0 for part in window):
            continue
        block_shape = tuple(len(part) for part in window)
        distance_sq = np.zeros(block_shape, dtype=float)
        for axis in range(3):
            offset = (window[axis] - centre[axis]).reshape(
                [-1 if index == axis else 1 for index in range(3)]
            )
            distance_sq += offset * offset
        np.sqrt(distance_sq, out=distance_sq)
        distance_sq -= float(inflated[atom])
        block = out[
            starts[0] : starts[0] + block_shape[0],
            starts[1] : starts[1] + block_shape[1],
            starts[2] : starts[2] + block_shape[2],
        ]
        np.minimum(block, distance_sq, out=block)
    return out


def _axes(coords: np.ndarray, margin: float, spacing: float) -> List[np.ndarray]:
    low = coords.min(axis=0) - margin
    high = coords.max(axis=0) + margin
    axes = []
    for axis in range(3):
        values = np.arange(float(low[axis]), float(high[axis]) + 0.5 * spacing, spacing)
        if values.size < 3:  # pragma: no cover - a degenerate extent
            values = np.arange(float(low[axis]), float(low[axis]) + 3.0 * spacing, spacing)
        axes.append(values)
    return axes


def _face_counts(labels: np.ndarray) -> Dict[int, int]:
    """``{label: number of faces it touches}`` over the six box faces."""
    found: Dict[int, set] = {}
    faces = (
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    )
    for index, face in enumerate(faces):
        for value in np.unique(face):
            if value == 0:
                continue
            found.setdefault(int(value), set()).add(index)
    return {label: len(faces) for label, faces in found.items()}


def _scan_sides(protein: np.ndarray, axis: int, reach: int) -> np.ndarray:
    """Cells with protein within ``reach`` on **both** sides along ``axis``.

    This is the LIGSITE protein-solvent-protein test (Levitt & Banaszak,
    *J. Mol. Graph.* **10** (1992) 229; Hendlich *et al.*, *J. Mol. Biol.*
    **267** (1997) 542): a place is inside a pocket when a scan along an axis
    finds the protein on both sides, and in open solvent when it does not.

    It replaces the "fraction of the neighbourhood that is protein" measure this
    module first used, and the reason is worth recording: a *volume* fraction
    measures the wrong thing. An 11 Å cube centred on a flat surface is about
    half protein (0.5), while the same cube on a pocket reaches out of the
    mouth into the solvent and is *more* accessible (0.32 measured on 3PTB) —
    the ordering comes out inverted. The directional test has no such problem:
    parallel to a flat surface there is no protein on either side, so a flat
    surface never qualifies, and a pocket is enclosed along all three axes.

    Computed with two cumulative scans per axis, so it is O(cells) and needs no
    library beyond NumPy.
    """
    length = protein.shape[axis]
    positions = np.arange(length)
    protein = protein.astype(bool)
    # Nearest protein index at or before each cell, and at or after it.
    before = np.where(protein, positions.reshape(_shape_for(axis)), -10**6)
    before = np.maximum.accumulate(before, axis=axis)
    after = np.where(protein, positions.reshape(_shape_for(axis)), 10**6)
    after = np.minimum.accumulate(after[tuple(slice(None, None, -1) for _ in range(3))], axis=axis)
    after = after[tuple(slice(None, None, -1) for _ in range(3))]
    cells = positions.reshape(_shape_for(axis))
    return ((cells - before) <= reach) & ((after - cells) <= reach)


def _shape_for(axis: int) -> Tuple[int, int, int]:
    """A shape that broadcasts an index vector along ``axis`` for a 3-D grid."""
    shape = [1, 1, 1]
    shape[axis] = -1
    return tuple(shape)


def _side_masks(protein: np.ndarray, axis: int, reach: int):
    """``(before, after)``: is there protein within ``reach`` on each side?

    Two cumulative scans per axis, so the whole test is O(cells) and needs
    nothing beyond NumPy. ``before[i]`` is true when a protein cell lies at
    ``i`` or up to ``reach`` cells *below* ``i`` along ``axis``; ``after`` is the
    same looking the other way.
    """
    length = protein.shape[axis]
    positions = np.arange(length)
    shape = _shape_for(axis)
    protein = protein.astype(bool)
    before = np.where(protein, positions.reshape(shape), -10**6)
    before = np.maximum.accumulate(before, axis=axis)
    after = np.where(protein, positions.reshape(shape), 10**6)
    flipped = tuple(slice(None, None, -1) for _ in range(3))
    after = np.minimum.accumulate(after[flipped], axis=axis)[flipped]
    cells = positions.reshape(shape)
    return ((cells - before) <= reach), ((after - cells) <= reach)


def _shape_for(axis: int) -> Tuple[int, int, int]:
    """A shape that broadcasts an index vector along ``axis`` for a 3-D grid."""
    shape = [1, 1, 1]
    shape[axis] = -1
    return tuple(shape)


def _directions_covered(protein: np.ndarray, reach: int) -> np.ndarray:
    """How many of the six axis directions see protein within ``reach`` (0-6).

    The classic LIGSITE count: a place *inside* a pocket has protein on both
    sides across the pocket (four directions) and on the floor (one), while the
    direction out of the mouth sees nothing — five of six. Requiring all six
    would demand a sealed hole and would miss every open site, which is what the
    first version of this function did: it found a 27 Å³ dent 16 Å from the
    ligand in trypsin and missed the S1 site entirely.

    The count is accumulated as integers: in NumPy, adding two boolean arrays is
    a logical **or**, so ``before + after`` would collapse to 0/1 and the count
    would never exceed 1. That mistake was made while writing this, which is why
    the report carries the histogram of the value and a test pins the count.
    """
    total = np.zeros(protein.shape, dtype=np.int8)
    for axis in range(3):
        before, after = _side_masks(protein, axis, reach)
        total += before.astype(np.int8)
        total += after.astype(np.int8)
    return total


def _label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Six-connected components of a boolean mask (0 is background)."""
    structure = _ndimage.generate_binary_structure(3, 1)
    labels, count = _ndimage.label(mask, structure=structure)
    return labels, int(count)


# ---------------------------------------------------------------------------
# residues and atoms
# ---------------------------------------------------------------------------


def _residue_label(atom) -> str:
    chain, res_id, res_name = residue_key(atom)
    name = f"{res_name}{res_id}" if res_name else str(res_id)
    return f"{chain}/{name}" if chain else name


def _nearest_atoms(points: np.ndarray, coords: np.ndarray, chunk: int = 4096):
    """``(index, distance)`` of the nearest atom for every point."""
    index = np.zeros(len(points), dtype=np.int64)
    distance = np.zeros(len(points), dtype=float)
    if coords.shape[0] == 0:  # pragma: no cover - defensive
        return index, distance
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        delta = block[:, None, :] - coords[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        best = np.argmin(squared, axis=1)
        index[start : start + chunk] = best
        distance[start : start + chunk] = np.sqrt(squared[np.arange(len(block)), best])
    return index, distance


def hydrophobicity_for(atoms) -> np.ndarray:
    """Per-atom hydrophobicity, from the surface module's documented scale.

    Imported lazily: the scale lives with the surface because that is what
    paints it, and importing it at module level would make a command-line cavity
    analysis pull in the viewer's package.
    """
    try:
        from .gui.surface import atom_hydrophobicity

        return np.asarray(atom_hydrophobicity(atoms, scale="residue"), dtype=float)
    except Exception:  # pragma: no cover - a caller that strips the viewer
        return np.full(len(list(atoms)), 0.5, dtype=float)


# ---------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------


def analyse_cavities(
    atoms,
    settings: Optional[CavitySettings] = None,
    *,
    residue_labels: Optional[Sequence[str]] = None,
) -> CavityReport:
    """Classify the empty space around ``atoms`` and measure its apertures.

    See the module docstring for the criterion. ``residue_labels`` may override
    the label of every atom (used by the tests to make a synthetic shell read as
    a named residue); otherwise the atoms' own chain/residue fields are used.
    """
    started = time.perf_counter()
    options = settings or CavitySettings()
    atoms = list(atoms)
    report = CavityReport(
        spacing=float(options.spacing or CAVITY_SPACING),
        probe=float(options.probe),
        core_radius=float(options.core_radius),
        margin=float(options.margin),
    )
    if not atoms:
        report.seconds = time.perf_counter() - started
        return report

    coords, radii = _atom_arrays(atoms)
    spacing = float(report.spacing)
    # The guarantee that makes "a face of the box is the outside" true: the box
    # must contain a solvent layer at least as thick as the largest ball the
    # analysis reasons about. With auto_margin off, the caller's margin is used
    # as given and a missing bulk layer is *reported* instead of assumed.
    needed = float(options.probe) + float(options.core_radius) + 2.0 * spacing
    margin = max(float(options.margin), needed) if options.auto_margin else float(options.margin)

    labels = None
    face_counts: Dict[int, int] = {}
    field = None
    axes: List[np.ndarray] = []
    for attempt in range(3):
        axes = _axes(coords, margin, spacing)
        field = probe_field(
            axes, coords, radii + float(options.probe), clamp=2.0 * spacing
        )
        labels, _count = _label(field > 0.0)
        face_counts = _face_counts(labels)
        if face_counts:
            break
        # No cell of the accessible set reached a face: the box is clipping the
        # solvent layer. Grow it and try again.
        margin *= 2.0
        report.margin = margin
        if not options.auto_margin:
            break

    assert labels is not None and field is not None
    shape = labels.shape
    report.grid = tuple(int(size) for size in shape)
    report.grid_points = int(shape[0] * shape[1] * shape[2])
    report.margin = margin
    report.box = (
        tuple(float(axis[0]) for axis in axes),
        tuple(float(axis[-1]) for axis in axes),
    )
    cell_volume = spacing**3

    bulk_labels = set(face_counts)
    bulk_faces = max(face_counts.values()) if face_counts else 0
    report.bulk_faces = int(bulk_faces)
    # The hypothesis fails when the accessible set never reaches a face, or when
    # some face has no accessible cell at all: a pocket whose mouth is outside
    # the box would then look enclosed, so nothing is called enclosed.
    report.box_too_tight = bool(not face_counts or bulk_faces < 6)

    # -- enclosed voids: accessible components the bulk does not reach -------
    sizes = np.bincount(labels.reshape(-1))
    pockets: List[Pocket] = []
    for label in range(1, len(sizes)):
        if sizes[label] == 0 or label in bulk_labels:
            continue
        volume = float(sizes[label]) * cell_volume
        if volume < float(options.min_volume):
            continue
        cells = np.argwhere(labels == label)
        kind = "unknown" if report.box_too_tight else "enclosed"
        pockets.append(
            _make_pocket(
                atoms,
                coords,
                axes,
                cells,
                kind=kind,
                volume=volume,
                openings=0,
                bottleneck=None,
                spacing=spacing,
                residue_labels=residue_labels,
                index=len(pockets) + 1,
                core_radius=float(options.core_radius),
            )
        )

    report.accessible_volume = float(np.count_nonzero(labels)) * cell_volume
    report.bulk_volume = float(
        sum(sizes[label] for label in bulk_labels) * cell_volume
    ) if bulk_labels else 0.0

    # -- open pockets: the LIGSITE scan, then the aperture -------------------
    # A cell qualifies when the protein is on both sides of it along all three
    # axes within ``enclosure_reach``: that is a place surrounded by protein,
    # whether or not it is sealed. Whether it is sealed is then decided by the
    # accessible component it belongs to, and *how* it opens by the bisection.
    aperture_started = time.perf_counter()
    report.enclosing_cells = 0
    if not report.box_too_tight:
        reach = max(1, int(round(float(options.enclosure_radius) / spacing)))
        protein = field < 0.0
        covered = _directions_covered(protein, reach)
        report.direction_histogram = {
            int(value): int(count)
            for value, count in enumerate(
                np.bincount(covered.reshape(-1), minlength=7).tolist()
            )
        }
        candidates = (labels > 0) & (covered >= int(options.min_directions))
        report.enclosing_cells = int(np.count_nonzero(candidates))
        distance = _ndimage.distance_transform_edt(labels > 0, sampling=spacing)
        cluster_labels, cluster_count = _label(candidates)
        cluster_sizes = np.bincount(cluster_labels.reshape(-1))
        for cluster in range(1, cluster_count + 1):
            volume = float(cluster_sizes[cluster]) * cell_volume
            if volume < float(options.min_volume):
                continue
            mask = cluster_labels == cluster
            # Which accessible component holds it? An enclosed one is already
            # reported above, so only the bulk-connected clusters are new.
            holder = set(np.unique(labels[mask]).tolist()) - {0}
            if not (holder & bulk_labels):
                continue
            cells = np.argwhere(mask)
            bottleneck, openings, neck_cells = _aperture(
                mask,
                labels > 0,
                distance,
                labels,
                bulk_labels,
                spacing,
                float(options.max_bottleneck),
            )
            pocket = _make_pocket(
                atoms,
                coords,
                axes,
                cells,
                kind="open",
                volume=volume,
                openings=openings,
                bottleneck=bottleneck,
                spacing=spacing,
                residue_labels=residue_labels,
                index=len(pockets) + 1,
                neck_cells=neck_cells,
                core_radius=float(options.core_radius),
            )
            if bottleneck is None or bottleneck >= float(options.wide_mouth):
                # A mouth this wide is a groove in the surface, not a site.
                pocket.kind = "open" if bottleneck is None else "shallow"
            pockets.append(pocket)
    report.aperture_seconds = time.perf_counter() - aperture_started

    # Shallow: the smallest open components, reported but demoted.
    for pocket in pockets:
        if pocket.kind == "open" and pocket.volume < 2.0 * float(options.min_volume):
            pocket.kind = "shallow"

    pockets.sort(key=lambda item: (-item.volume, item.label))
    for index, pocket in enumerate(pockets, start=1):
        pocket.label = f"P{index}"
    report.pockets = pockets[: int(options.max_pockets)]
    report.seconds = time.perf_counter() - started
    return report


def _aperture(
    pocket_cells: np.ndarray,
    accessible: np.ndarray,
    distance: np.ndarray,
    labels: np.ndarray,
    bulk_labels: Iterable[int],
    spacing: float,
    max_radius: float,
    margin: float = 6.0,
) -> Tuple[Optional[float], int, np.ndarray]:
    """``(bottleneck radius, opening count, neck cells)`` for one pocket.

    The bottleneck is the **narrowest cross-section of the path from the pocket
    to the outside**, found by bisection on the radius of the ball that can still
    make the trip:

    * the *seed* is the pocket's deepest cell (the largest ball that fits inside
      it) — not the whole cluster, which may spill out of the mouth;
    * ``reach(r)`` asks whether that seed is connected to the **bulk solvent**
      (the accessible component that touches a face of the analysis box) through
      cells where a ball of radius ``r`` fits (``distance >= r``);
    * the largest ``r`` for which that holds is the bottleneck. A funnel whose
      mouth is narrower than its floor converges to the mouth; a groove with a
      wide mouth converges to the pocket's own depth, i.e. "nothing is pinching".

    The search is **local**: every labelling runs on the pocket's bounding box
    grown by ``margin`` Å, which is what keeps the cost at seconds. A pocket whose
    only exit lies further away than that is measured as sealed — pockets are
    local by construction and the margin is six times the spacing of the finest
    default grid, so this is stated rather than discovered.

    The openings are the separate narrow bands on that path, taken just below the
    bottleneck. A pocket with two mouths narrower than its body gives two; a bore
    of *uniform* radius gives one, because there the narrowest cross-section is
    the whole bore — a property of the definition, not of the grid.
    """
    cells = np.argwhere(pocket_cells)
    if cells.size == 0:  # pragma: no cover - defensive
        return None, 0, np.zeros((0, 3), dtype=np.int64)
    grow = int(math.ceil((float(max_radius) + margin) / spacing)) + 1
    low = np.maximum(cells.min(axis=0) - grow, 0)
    high = np.minimum(cells.max(axis=0) + grow + 1, np.asarray(labels.shape))
    view = tuple(slice(int(low[axis]), int(high[axis])) for axis in range(3))

    pocket_crop = pocket_cells[view]
    accessible_crop = accessible[view]
    distance_crop = distance[view]
    labels_crop = labels[view]
    if not np.any(labels_crop):  # pragma: no cover - defensive
        return None, 0, np.zeros((0, 3), dtype=np.int64)
    bulk_crop = {
        int(value) for value in np.unique(labels_crop).tolist() if value in set(bulk_labels)
    }
    if not bulk_crop:
        # The bulk is not inside the crop: the pocket's exit is further away than
        # the local search reaches. Reported as sealed rather than guessed.
        return None, 0, np.zeros((0, 3), dtype=np.int64)

    deepest = float(distance_crop[pocket_crop].max())
    seed = pocket_crop & (distance_crop >= deepest - 1e-9)

    def reach(radius: float) -> bool:
        mask = accessible_crop & (distance_crop >= radius)
        if not mask.any():
            return False
        found, count = _label(mask)
        if count == 0:  # pragma: no cover - defensive
            return False
        mine = set(np.unique(found[seed & mask]).tolist()) - {0}
        if not mine:
            return False
        for label in mine:
            component = found == label
            touching = {
                int(value)
                for value in np.unique(labels_crop[component]).tolist()
                if value
            }
            if touching & bulk_crop:
                return True
        return False

    if not reach(deepest):
        # Sealed even at its own depth: this is an enclosed void.
        return None, 0, np.zeros((0, 3), dtype=np.int64)

    low_r = deepest
    high_r = max(deepest, float(max_radius)) + 1e-6
    for _ in range(BOTTLENECK_STEPS):
        middle = 0.5 * (low_r + high_r)
        if reach(middle):
            low_r = middle
        else:
            high_r = middle
    bottleneck = low_r

    # The **necks** at the critical radius: cells whose inradius is within a
    # third of a cell of the bottleneck. Counting everything thinner than the
    # bottleneck instead pulls in every shallow dimple of the surface and
    # reports hundreds of "openings" (measured: 82 to 602 on 3PTB and EGFR,
    # which is nonsense), so the band is centred on the critical radius where
    # the pinches actually are.
    #
    # The count is the number of separate necks, and a **single continuous bore
    # counts as one**: a straight channel has one narrowest cross-section even
    # though it has two exits — the field is *openings* (bottlenecks), not ends.
    # A pocket with two distinct mouths narrower than its body gives two.
    structure = _ndimage.generate_binary_structure(3, 1)
    mask = accessible_crop & (distance_crop >= max(0.0, bottleneck - 0.5 * spacing))
    found, count = _label(mask)
    # ``wide`` is the part of the pocket that is *not* pinched. A bore of uniform
    # radius has none, which is exactly what turns "no neck found" into "one
    # continuous bore" rather than into a wrong count.
    wide = accessible_crop & (distance_crop >= bottleneck + 0.5 * spacing)
    # One vectorised membership test for the whole crop: doing it per component
    # (a full-array comparison each) made this loop cost tens of seconds.
    bulk_mask_crop = np.isin(labels_crop, np.fromiter(bulk_crop, dtype=np.int64))
    neck_points: List[np.ndarray] = []
    if count:
        mine = set(np.unique(found[seed & mask]).tolist()) - {0}
        for label in mine:
            component = found == label
            touching = {
                int(value)
                for value in np.unique(labels_crop[component]).tolist()
                if value
            }
            if not (touching & bulk_crop):
                continue
            band = component & (np.abs(distance_crop - bottleneck) <= 0.35 * spacing)
            band_labels, band_count = _label(band)
            for index in range(1, band_count + 1):
                cluster = band_labels == index
                if not cluster.any():  # pragma: no cover - defensive
                    continue
                grown = _ndimage.binary_dilation(cluster, structure=structure)
                if not (grown & wide).any():
                    continue          # a dimple, not a neck on the path
                if not (grown & bulk_mask_crop).any():
                    continue
                neck_points.append(np.argwhere(cluster) + low[None, :])
    openings = max(1, len(neck_points))
    if neck_points:
        neck_cells = np.concatenate(neck_points, axis=0)
    else:  # pragma: no cover - a single continuous bore
        band_cells = np.argwhere(
            accessible_crop & (np.abs(distance_crop - bottleneck) <= 0.35 * spacing)
        )
        neck_cells = (
            band_cells + low[None, :] if band_cells.size else np.zeros((0, 3), dtype=np.int64)
        )
    return bottleneck, openings, neck_cells


def _bulk_mask(labels: np.ndarray, bulk_labels: Iterable[int]) -> np.ndarray:
    """A boolean mask of the cells whose component is one of ``bulk_labels``."""
    wanted = set(int(value) for value in bulk_labels)
    out = np.zeros(labels.shape, dtype=bool)
    for value in np.unique(labels).tolist():
        if int(value) in wanted:
            out |= labels == value
    return out


def _make_pocket(
    atoms,
    coords: np.ndarray,
    axes: Sequence[np.ndarray],
    cells: np.ndarray,
    *,
    kind: str,
    volume: float,
    openings: int,
    bottleneck: Optional[float],
    spacing: float,
    residue_labels: Optional[Sequence[str]],
    index: int,
    neck_cells: Optional[np.ndarray] = None,
    core_radius: float = CORE_RADIUS,
) -> Pocket:
    """Assemble one pocket record from the grid cells that make it up."""
    world = np.stack(
        [axes[axis][cells[:, axis]] for axis in range(3)], axis=1
    )
    centre = world.mean(axis=0)
    extent = world.max(axis=0) - world.min(axis=0)
    # The lining: the atoms nearest to the core cells.
    near_index, _near_distance = _nearest_atoms(world, coords)
    lining = []
    for atom_index in np.unique(near_index).tolist():
        atom = atoms[int(atom_index)]
        label = (
            residue_labels[int(atom_index)]
            if residue_labels is not None and int(atom_index) < len(residue_labels)
            else _residue_label(atom)
        )
        if label and label not in lining:
            lining.append(label)
    hydrophobicity = 0.0
    if len(np.unique(near_index)):
        per_atom = hydrophobicity_for(atoms)
        hydrophobicity = float(per_atom[np.unique(near_index)].mean())

    bottleneck_residues: List[str] = []
    if neck_cells is not None and len(neck_cells):
        neck_world = np.stack(
            [axes[axis][neck_cells[:, axis]] for axis in range(3)], axis=1
        )
        neck_index, _ = _nearest_atoms(neck_world, coords)
        for atom_index in np.unique(neck_index).tolist():
            atom = atoms[int(atom_index)]
            label = (
                residue_labels[int(atom_index)]
                if residue_labels is not None and int(atom_index) < len(residue_labels)
                else _residue_label(atom)
            )
            if label and label not in bottleneck_residues:
                bottleneck_residues.append(label)

    core_radius = float(core_radius)
    if bottleneck is None:
        enclosure = 1.0
    else:
        enclosure = max(0.0, min(1.0, 1.0 - float(bottleneck) / max(core_radius, 1e-6)))
    return Pocket(
        label=f"P{index}",
        kind=kind,
        volume=float(volume),
        points=int(len(cells)),
        centre=tuple(float(value) for value in centre),
        extent=tuple(float(value) for value in extent),
        openings=int(openings),
        bottleneck_radius=None if bottleneck is None else float(bottleneck),
        bottleneck_residues=bottleneck_residues,
        lining_residues=lining,
        hydrophobicity=hydrophobicity,
        enclosure=enclosure,
        geometric_score=geometric_score(enclosure, volume, hydrophobicity),
        atom_indices=sorted({int(value) for value in np.unique(near_index).tolist()}),
    )

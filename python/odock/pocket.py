# SPDX-License-Identifier: GPL-3.0-or-later
"""De-novo blind pocket detection on a ligand-free receptor.

This is the "grid cavity probe" of module C.1 of the requirements: the receptor
is rasterised onto a regular grid, every grid point that a rolling spherical
probe cannot occupy is marked *blocked*, and the surviving free points are
grouped into cavities.

The parameters that matter are

``spacing``
    edge length of one grid cell (Å); each free point contributes
    ``spacing ** 3`` Å³ of volume;
``probe``
    radius of the rolling probe (Å). A grid point is blocked when it lies
    within ``vdw(atom) + probe`` of any receptor atom, i.e. when the probe --
    modelled as a hard sphere -- would overlap that atom;
``buriedness``
    the fraction of the 26 cubic directions that must be walled by protein
    before a free point counts as buried.

Everything is NumPy; there is no SciPy and no RDKit dependency, and every
neighbour test is an array shift, so the cost is linear in the number of grid
points and a 3000-atom receptor is searched in a couple of seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .prepare import BoxSpec, box_from_points

__all__ = [
    "VDW_RADII",
    "DEFAULT_RAY_LENGTH",
    "Pocket",
    "box_from_pocket",
    "find_pockets",
    "residues_within",
]


# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

#: van der Waals radii in Å (Bondi 1964, plus the values RDKit's periodic table
#: reports for the elements Bondi did not cover). Vina and AutoDock 4 use the
#: same set, so the interaction criteria in :mod:`odock.analysis` share it.
VDW_RADII: Dict[str, float] = {
    "H": 1.20, "He": 1.40,
    "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Sc": 2.15, "Ti": 2.11, "V": 2.07, "Cr": 2.06,
    "Mn": 2.05, "Fe": 2.04, "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
    "Rb": 3.03, "Sr": 2.49, "Mo": 2.17, "Ru": 2.07, "Rh": 2.02, "Pd": 2.05,
    "Ag": 2.03, "Cd": 2.18, "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06,
    "I": 1.98, "Xe": 2.16,
    "Pt": 2.13, "Au": 2.14, "Hg": 2.23, "Tl": 1.96, "Pb": 2.02, "Bi": 2.07,
}

#: Radius used for an element the table does not know (carbon-like).
_DEFAULT_RADIUS = 1.70

#: Default ray length (Å) used to decide whether a free grid point is walled
#: in. Five ångström is about the radius of a drug-sized binding site, so a
#: point at the centre of such a site sees protein in every direction.
DEFAULT_RAY_LENGTH = 5.0

#: Default minimum separation (Å) between two pocket centres inside one
#: connected cavity system. 6 Å is about the diameter of a drug-sized site, so
#: a single spherical cavity stays one pocket while a channel network is cut
#: into several.
DEFAULT_POCKET_RADIUS = 6.0

#: How far below a cavity system's most buried cell a grid point may be and
#: still be allowed to seed a sub-pocket, expressed as a fraction of the 26
#: cubic directions (3/26 ~ 0.115). Larger values give more, smaller pockets.
DEFAULT_CORE_DEPTH = 3.0 / 26.0

#: Hard ceiling on the number of grid points: a silly ``spacing`` must produce
#: a clear error instead of an out-of-memory kill. 12 M points is a 229 Å cube
#: at 1 Å spacing, several times the volume of the largest known receptor.
_MAX_GRID_POINTS = 12_000_000

#: Atoms are rasterised this many at a time: large enough to amortise NumPy's
#: call overhead, small enough to keep the ``(chunk, offsets, 3)`` temporaries
#: in cache.
_CHUNK = 64

#: The 26 cubic directions.
_DIRECTIONS: Tuple[Tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class Pocket:
    """One candidate binding site.

    Attributes
    ----------
    index
        0-based rank; 0 is the most promising pocket.
    center
        Centroid of the cavity sample points (Å).
    volume
        ``n_points * spacing ** 3`` (Å³).
    score
        Heuristic druggability score, higher is better; the formula is
        documented in :func:`find_pockets`.
    residue_labels
        Receptor residues within 5 Å of `center`, as ``"ASP189 A"``.
    n_points
        Number of grid points the cavity consists of.
    points
        The cavity sample points themselves (Å), for rendering.
    """

    index: int
    center: Tuple[float, float, float]
    volume: float
    score: float
    residue_labels: List[str] = field(default_factory=list)
    n_points: int = 0
    points: List[Tuple[float, float, float]] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        residues = ", ".join(self.residue_labels[:4]) or "-"
        return (
            f"Pocket {self.index + 1}: center=({self.center[0]:.1f}, "
            f"{self.center[1]:.1f}, {self.center[2]:.1f}) A, "
            f"V={self.volume:.0f} A^3, score={self.score:.2f}, {residues}"
        )


# ---------------------------------------------------------------------------
# Accept an RDKit molecule, Atom-like objects or a coordinate array
# ---------------------------------------------------------------------------


@dataclass
class _Atom:
    """The few atom properties the finder needs, independent of the source."""

    element: str
    x: float
    y: float
    z: float
    name: str = ""
    res_name: str = ""
    res_id: int = 0
    chain: str = ""


def _rdkit_atoms(mol) -> List[_Atom]:
    if mol.GetNumConformers() == 0:
        raise ValueError("the structure has no conformer; nothing to search")
    conf = mol.GetConformer()
    atoms: List[_Atom] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        info = atom.GetPDBResidueInfo()
        atoms.append(
            _Atom(
                element=atom.GetSymbol(),
                x=float(pos.x),
                y=float(pos.y),
                z=float(pos.z),
                name=(info.GetName().strip() if info is not None else ""),
                res_name=(info.GetResidueName().strip() if info is not None else ""),
                res_id=(int(info.GetResidueNumber()) if info is not None else 0),
                chain=(info.GetChainId().strip() if info is not None else ""),
            )
        )
    return atoms


def _as_atoms(source) -> List[_Atom]:
    """Normalise `source` into a list of :class:`_Atom`.

    Accepted inputs:

    * an RDKit ``Mol`` (residue labels come from its PDB residue info);
    * a sequence of Atom-like objects exposing ``element``/``x``/``y``/``z``
      and optionally ``name``/``res_name``/``res_id``/``chain`` -- exactly what
      :class:`odock.gui.structure.Atom` provides;
    * an ``(N, 3)`` coordinate array (every atom is treated as carbon);
    * a sequence of ``(x, y, z)`` triples.
    """
    if isinstance(source, np.ndarray) and source.ndim == 2:
        pts = np.asarray(source, dtype=float)
        if pts.shape[1] != 3:
            raise ValueError("a coordinate array must have shape (N, 3)")
        return [_Atom("C", float(x), float(y), float(z)) for x, y, z in pts]
    if hasattr(source, "GetNumAtoms") and hasattr(source, "GetConformer"):
        return _rdkit_atoms(source)

    try:
        items = list(source)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError(
            "expected an RDKit molecule, a sequence of atoms or an (N, 3) array"
        ) from exc

    atoms: List[_Atom] = []
    for item in items:
        if all(hasattr(item, attr) for attr in ("element", "x", "y", "z")):
            atoms.append(
                _Atom(
                    element=str(getattr(item, "element") or "C"),
                    x=float(item.x),
                    y=float(item.y),
                    z=float(item.z),
                    name=str(getattr(item, "name", "") or ""),
                    res_name=str(getattr(item, "res_name", "") or ""),
                    res_id=int(getattr(item, "res_id", 0) or 0),
                    chain=str(getattr(item, "chain", "") or ""),
                )
            )
        else:
            try:
                x, y, z = (float(v) for v in item)
            except Exception as exc:
                raise TypeError(f"cannot interpret {item!r} as an atom") from exc
            atoms.append(_Atom("C", x, y, z))
    return atoms


def _residue_label(res_name: str, res_id: int, chain: str) -> str:
    """``("ASP", 189, "A")`` -> ``"ASP189 A"``."""
    label = f"{res_name}{res_id}"
    return f"{label} {chain}" if chain else label


# ---------------------------------------------------------------------------
# Grid primitives
# ---------------------------------------------------------------------------


def _shifted(a: np.ndarray, delta: Sequence[int], fill) -> np.ndarray:
    """``out[i] = a[i + delta]``, with `fill` outside the grid."""
    out = np.empty(a.shape, dtype=a.dtype)
    _shift_into(a, delta, fill, out)
    return out


def _shift_into(a: np.ndarray, delta: Sequence[int], fill, out: np.ndarray) -> None:
    """``out[i] = a[i + delta]``, with `fill` outside the grid.

    Writes into a caller-supplied buffer so the propagation loop does not
    allocate 26 arrays per iteration.
    """
    out[...] = fill
    dst, src = [], []
    for axis, d in enumerate(delta):
        n = a.shape[axis]
        if d >= 0:
            dst.append(slice(0, max(0, n - d)))
            src.append(slice(d, n))
        else:
            dst.append(slice(-d, n))
            src.append(slice(0, max(0, n + d)))
    out[tuple(dst)] = a[tuple(src)]


def _sphere_offsets(reach: int) -> np.ndarray:
    """Integer grid offsets covering a ball of `reach` cells."""
    axis = np.arange(-reach, reach + 1)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    offs = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.int64)
    keep = (offs**2).sum(axis=1) <= reach * reach
    return offs[keep]


def _mark_blocked(
    coords: np.ndarray,
    cutoff: np.ndarray,
    origin: np.ndarray,
    dims: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """The grid mask that is `True` wherever the probe cannot fit.

    Atoms are grouped by their grid-space reach so a group shares one offset
    table, and processed in chunks so the ``(chunk, offsets, 3)`` temporaries
    stay small. This is the cell-list idea in vectorised form: the work is
    proportional to ``sum((2 * reach_i + 1) ** 3)`` rather than to
    ``n_grid * n_atoms``.
    """
    shape = tuple(int(d) for d in dims)
    flat = np.zeros(int(np.prod(dims)), dtype=bool)
    centres = np.floor((coords - origin) / spacing).astype(np.int64)
    reach = np.ceil(cutoff / spacing).astype(np.int64)

    for r in np.unique(reach):
        offs = _sphere_offsets(int(r))
        members = np.nonzero(reach == r)[0]
        # A grid cell centre can be up to spacing * sqrt(3) / 2 away from the
        # atom, so the offset table keeps one cell of slack.
        limit = float(cutoff[members].max()) + spacing
        offs = offs[(offs**2).sum(axis=1) * spacing * spacing <= limit * limit]
        for start in range(0, len(members), _CHUNK):
            chunk = members[start : start + _CHUNK]
            pts = centres[chunk][:, None, :] + offs[None, :, :]
            inside = np.all((pts >= 0) & (pts < dims), axis=-1)
            clipped = np.clip(pts, 0, dims - 1)
            lin = (
                (clipped[..., 0] * dims[1] + clipped[..., 1]) * dims[2]
                + clipped[..., 2]
            )
            xyz = origin + clipped * spacing
            d2 = ((xyz - coords[chunk][:, None, :]) ** 2).sum(axis=-1)
            flat[lin[inside & (d2 <= (cutoff[chunk] ** 2)[:, None])]] = True
    return flat.reshape(shape)


def _shell_counts(mask: np.ndarray) -> np.ndarray:
    """Number of `True` cells among the 26 neighbours of every cell."""
    src = mask.astype(np.uint8)
    counts = np.zeros(mask.shape, dtype=np.uint8)
    for delta in _DIRECTIONS:
        counts += _shifted(src, delta, np.uint8(0))
    return counts


def _partition(
    points: np.ndarray,
    burial: np.ndarray,
    radius: float,
    core_depth: float,
) -> np.ndarray:
    """Cut one cavity system into sub-pockets around its most buried cells.

    A protein's buried void space is normally a single 26-connected network, so
    the connected-component step alone yields one sprawling "pocket" per
    receptor that swallows several chemical sites. This partition restores a
    usable ranking:

    1. the *core* of the system is the set of cells whose burial is within
       `core_depth` of the system maximum;
    2. the core is covered greedily, most buried first, so that no two centres
       lie within `radius` of each other;
    3. every cell of the system is assigned to its nearest centre, which turns
       the centres into compact, ligand-sized pockets.

    Restricting the centres to the core is what keeps the partition from
    shattering a cavity into one fragment per leftover cell. ``radius <= 0``
    (or a single cell) disables the partition and returns one group, which is
    the literal connected-system behaviour.

    Returns an integer group label per input point, contiguous from 0.
    """
    n = len(points)
    if n <= 1 or radius <= 0:
        return np.zeros(n, dtype=np.int64)

    cutoff = float(burial.max()) - float(core_depth)
    candidates = np.flatnonzero(burial >= cutoff)
    order = candidates[np.argsort(-burial[candidates], kind="stable")]

    claimed = np.zeros(n, dtype=bool)
    centres: List[int] = []
    radius2 = float(radius) ** 2
    for i in order:
        if claimed[i]:
            continue
        centres.append(int(i))
        claimed |= ((points - points[i]) ** 2).sum(axis=1) <= radius2
        if claimed.all():
            break

    if len(centres) == 1:
        return np.zeros(n, dtype=np.int64)

    centre_points = points[np.asarray(centres, dtype=np.int64)]
    best = np.full(n, np.inf)
    groups = np.zeros(n, dtype=np.int64)
    for k, centre in enumerate(centre_points):
        d2 = ((points - centre) ** 2).sum(axis=1)
        closer = d2 < best
        best[closer] = d2[closer]
        groups[closer] = k
    del best
    return groups


def _walled_directions(blocked: np.ndarray, steps: int) -> np.ndarray:
    """Count the cubic directions that meet protein within `steps` cells.

    ``steps == 1`` is the literal "how many of the 26 neighbouring grid cells
    are blocked" test. Larger values look further down the same 26 rays, which
    is what lets a grid point in the *middle* of a cavity count as buried
    rather than only the shell that touches the protein.
    """
    hits = np.zeros(blocked.shape, dtype=np.uint8)
    for delta in _DIRECTIONS:
        reached = _shifted(blocked, delta, False)
        for _ in range(1, steps):
            reached = reached | _shifted(reached, delta, False)
        hits += reached.astype(np.uint8)
    return hits


def _label_components(mask: np.ndarray) -> np.ndarray:
    """Label the 26-connected components of `mask`.

    Returns an array of the same shape that is ``-1`` outside `mask` and, for
    every cell of a component, the flat index (C order) of that component's
    first cell -- a stable component id.

    The labelling is min-label propagation: every set cell starts with its own
    flat index and the 26 shifted copies are minimised in repeatedly until
    nothing changes. The fixed point of "smallest label among my neighbours" is
    exactly the component minimum, so the result is exact; the number of
    iterations is bounded by the component diameter in cells.
    """
    shape = mask.shape
    flat = np.flatnonzero(mask.ravel())
    if flat.size == 0:
        return np.full(shape, -1, dtype=np.int64)

    coords = np.unravel_index(flat, shape)
    lo = [int(c.min()) for c in coords]
    hi = [int(c.max()) + 1 for c in coords]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    sub = mask[box]
    del coords

    big = np.iinfo(np.int32).max
    labels = np.full(sub.shape, big, dtype=np.int32)
    idx = np.arange(sub.size, dtype=np.int32).reshape(sub.shape)
    labels[sub] = idx[sub]
    del idx
    scratch = np.empty(sub.shape, dtype=np.int32)
    empty = ~sub

    # Only the bounding box of the seeds is propagated: a cavity is small
    # compared with the receptor, so this cuts the working array by one to two
    # orders of magnitude. The empty cells are reset to `big` after every shift,
    # otherwise the minimum would leak a label across the free space between two
    # separate cavities and merge them.
    for _ in range(2 * int(max(sub.shape)) + 8):
        before = labels.copy()
        for delta in _DIRECTIONS:
            _shift_into(labels, delta, big, scratch)
            np.minimum(labels, scratch, out=labels)
            labels[empty] = big
        if np.array_equal(labels, before):
            break
    del before

    out = np.full(shape, -1, dtype=np.int64)
    out[box] = np.where(sub, labels, -1)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def residues_within(mol, center, radius: float) -> List[str]:
    """Residues whose closest atom lies within `radius` Å of `center`.

    Labels read ``"ASP189 A"`` and are sorted by chain, residue number and
    residue name. Atoms that carry no residue information are skipped, so a
    bare coordinate array yields an empty list.
    """
    atoms = _as_atoms(mol)
    if not atoms:
        return []
    centre = np.asarray(center, dtype=float).reshape(3)
    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    dist = np.linalg.norm(coords - centre, axis=1)

    best: Dict[Tuple[str, int, str], float] = {}
    for atom, d in zip(atoms, dist):
        if d > radius or not atom.res_name:
            continue
        key = (atom.chain, atom.res_id, atom.res_name)
        if key not in best or d < best[key]:
            best[key] = float(d)
    return [_residue_label(name, res_id, chain) for chain, res_id, name in sorted(best)]


def find_pockets(
    mol,
    *,
    spacing: float = 1.0,
    probe: float = 1.4,
    min_volume: float = 100.0,
    max_pockets: int = 10,
    buriedness: float = 0.55,
    ray_length: float = DEFAULT_RAY_LENGTH,
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    core_depth: float = DEFAULT_CORE_DEPTH,
) -> List[Pocket]:
    """Find candidate binding pockets on a ligand-free receptor.

    Parameters
    ----------
    mol
        The receptor: an RDKit ``Mol``, a sequence of
        :class:`odock.gui.structure.Atom`-like objects, or an ``(N, 3)``
        coordinate array.
    spacing
        Grid spacing in Å (default 1.0).
    probe
        Probe radius in Å (default 1.4, water).
    min_volume
        Cavities smaller than this are discarded (Å³, default 100).
    max_pockets
        Return at most this many pockets, best score first.
    buriedness
        A free grid point becomes a *seed* once at least this fraction of the
        26 cubic directions is walled by protein (default 0.55).
    ray_length
        How far down those 26 directions to look for protein (Å). With
        ``ray_length=spacing`` this degenerates to the immediate-neighbour
        test; the 5 Å default is what makes the interior of a cavity count as
        buried as well as the shell that touches the protein.
    pocket_radius
        Minimum separation (Å) between two pocket centres. The 26-connected
        components of the buried free points are the receptor's *cavity
        systems*, and a real protein normally has exactly one of them -- its
        internal void network -- which would swallow several chemical sites.
        Each system is therefore cut into sub-pockets around its most buried
        cells (see :func:`_partition`) so the ranked list is usable. Pass ``0``
        to report one pocket per connected system, which is the literal
        connected-component behaviour.
    core_depth
        How far below a system's most buried cell a grid point may lie and
        still seed a sub-pocket, as a fraction of the 26 directions
        (default ``3/26``). Only relevant when `pocket_radius` is positive.

    Returns
    -------
    ``list[Pocket]`` sorted by descending score and renumbered ``0..n-1``.
    The score is the documented heuristic

    ``score = (volume / 100 A^3) ** (1/3) * mean_buriedness * enclosure``

    with ``enclosure = 1 - mean(fraction of the 26 neighbours that are free but
    not buried)``. The cube root stops a pocket from being ranked first merely
    for being large, the second factor prefers points that are genuinely walled
    in, and the third penalises shallow grooves that open onto bulk solvent.
    """
    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if probe < 0:
        raise ValueError(f"probe must not be negative, got {probe}")
    if min_volume < 0:
        raise ValueError(f"min_volume must not be negative, got {min_volume}")
    if max_pockets < 0:
        raise ValueError(f"max_pockets must not be negative, got {max_pockets}")
    if not 0.0 <= buriedness <= 1.0:
        raise ValueError(f"buriedness must be in [0, 1], got {buriedness}")
    if ray_length <= 0:
        raise ValueError(f"ray_length must be positive, got {ray_length}")
    if pocket_radius < 0:
        raise ValueError(f"pocket_radius must not be negative, got {pocket_radius}")
    if not 0.0 <= core_depth <= 1.0:
        raise ValueError(f"core_depth must be in [0, 1], got {core_depth}")

    atoms = _as_atoms(mol)
    if not atoms or max_pockets == 0:
        return []

    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    radii = np.array(
        [VDW_RADII.get(a.element, _DEFAULT_RADIUS) for a in atoms], dtype=float
    )
    cutoff = radii + float(probe)

    padding = float(cutoff.max()) + float(spacing)
    origin = coords.min(axis=0) - padding
    upper = coords.max(axis=0) + padding
    dims = np.maximum(np.floor((upper - origin) / spacing).astype(np.int64) + 1, 1)
    if int(np.prod(dims)) > _MAX_GRID_POINTS:
        raise ValueError(
            f"a {spacing:g} A grid over this receptor needs "
            f"{int(np.prod(dims)):,} points (limit {_MAX_GRID_POINTS:,}); "
            "increase spacing"
        )

    blocked = _mark_blocked(coords, cutoff, origin, dims, spacing)
    free = ~blocked

    steps = max(1, int(round(float(ray_length) / float(spacing))))
    burial = _walled_directions(blocked, steps).astype(np.float32) / np.float32(26.0)

    seeds = free & (burial >= np.float32(buriedness))
    if not seeds.any():
        return []

    labels = _label_components(seeds)
    del seeds

    # Free space that is not buried: the solvent-exposed side of a groove.
    open_neighbours = _shell_counts(free & (labels < 0)).astype(np.float32)

    li, lj, lk = np.nonzero(labels >= 0)
    comp = labels[li, lj, lk]
    del labels
    order = np.argsort(comp, kind="stable")
    comp = comp[order]
    li, lj, lk = li[order], lj[order], lk[order]
    starts = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    ends = np.r_[starts[1:], comp.size]

    spacing3 = float(spacing) ** 3
    found: List[Tuple[float, Pocket]] = []
    for s, e in zip(starts, ends):
        gi, gj, gk = li[s:e], lj[s:e], lk[s:e]
        xs = origin[0] + gi * spacing
        ys = origin[1] + gj * spacing
        zs = origin[2] + gk * spacing
        world = np.stack([xs, ys, zs], axis=1)
        groups = _partition(
            world,
            burial[gi, gj, gk],
            float(pocket_radius),
            float(core_depth),
        )

        for g in range(int(groups.max()) + 1):
            sel = groups == g
            count = int(sel.sum())
            volume = count * spacing3
            if volume < float(min_volume):
                continue
            gx, gy, gz = xs[sel], ys[sel], zs[sel]
            centre = (float(gx.mean()), float(gy.mean()), float(gz.mean()))
            mean_burial = float(burial[gi[sel], gj[sel], gk[sel]].mean())
            exposure = float(
                (open_neighbours[gi[sel], gj[sel], gk[sel]] / np.float32(26.0)).mean()
            )
            enclosure = max(0.0, 1.0 - exposure)
            score = (volume / 100.0) ** (1.0 / 3.0) * mean_burial * enclosure
            found.append(
                (
                    score,
                    Pocket(
                        index=0,
                        center=centre,
                        volume=volume,
                        score=score,
                        residue_labels=residues_within(atoms, centre, 5.0),
                        n_points=count,
                        points=[
                            (float(x), float(y), float(z))
                            for x, y, z in zip(gx, gy, gz)
                        ],
                    ),
                )
            )

    found.sort(key=lambda item: (-item[0], -item[1].n_points))
    out = [pocket for _, pocket in found[: int(max_pockets)]]
    for i, pocket in enumerate(out):
        pocket.index = i
    return out


def box_from_pocket(
    pocket: Pocket, *, buffer: float = 0.0, spacing: float = 0.375
) -> BoxSpec:
    """The smallest axis-aligned search box containing `pocket`.

    `buffer` is added on every side, so ``buffer=2.0`` gives a box that is 4 Å
    larger in each dimension. A pocket without sample points falls back to a
    cube around its centre.
    """
    if pocket.points:
        pts = np.asarray(pocket.points, dtype=float)
    elif pocket.center:
        pts = np.asarray([pocket.center], dtype=float)
    else:  # pragma: no cover - defensive
        raise ValueError("the pocket has neither points nor a centre")
    return box_from_points(pts, buffer=float(buffer), spacing=float(spacing))

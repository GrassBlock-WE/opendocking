# SPDX-License-Identifier: GPL-3.0-or-later
"""Solvent-accessible surface area, per atom, per residue and per pose.

Why this module exists
----------------------
"It fits" is not a measurement. A docking workbench has to answer three
questions with a number, and this module is what computes them:

* **How exposed is this residue?** — the SASA of every atom, summed per residue.
* **How much of it does the partner hide?** — the buried area and buried
  fraction of every residue against a stated reference, plus the list of the
  most buried residues.
* **How much ligand surface does the pocket actually hide?** — the ligand's
  buried contact area, i.e. its SASA alone minus its SASA inside the complex.

The model
---------
Shrake & Rupley, *J. Mol. Biol.* **79** (1973) 351: the solvent-accessible
surface of a molecule is the boundary of the volume swept by the centre of a
probe sphere of radius :data:`PROBE_RADIUS` (1.4 Å, a water molecule) rolled
over the van der Waals surface. Numerically the probe centre is a set of
points on a sphere of radius ``r_i + probe`` around every atom; a point is
*accessible* when no other atom's own probe sphere contains it, and an atom's
SASA is

    A_i = 4 π (r_i + probe)² · (accessible points / total points).

The van der Waals radii are Bondi's (*J. Phys. Chem.* **68** (1964) 441) — the
same table the workbench already renders with (`odock.gui.structure`), so the
picture and the number can never disagree. The sphere points are a
**golden-spiral** (Fibonacci) lattice: it is deterministic, needs no table and
its quadrature error falls as ``1/n_points`` for a cap boundary. The default
:data:`DEFAULT_POINTS` = 92 is the count Shrake & Rupley used; the exact
two-sphere test in the test suite measures the residual error at that count so
the number in the code and the number in the test cannot drift apart.

Two properties of the construction are relied on elsewhere and are tested
explicitly:

* **Monotonicity.** Adding an occluding atom can only remove accessible points,
  so an atom's SASA never increases when its neighbourhood gets more crowded.
  Burial is therefore always in ``[0, 1]`` and no clamping is needed.
* **Rigid-motion invariance.** Each point is generated from the atom's own
  centre, so translating or rotating the whole structure changes no area.

Cost
----
The neighbour search is a uniform cell list with a cell as wide as the largest
possible overlap (``2 · (r_max + probe)``), so the work is linear in the atom
count for a fixed density. A 2 500-atom receptor at 92 points costs a fraction
of a second; :func:`ligand_buried_contact_area` is far cheaper still because it
only ever evaluates the ligand's own points.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ATOMIC_RADII",
    "DEFAULT_POINTS",
    "PROBE_RADIUS",
    "BurialReport",
    "ResidueArea",
    "burial",
    "buried_contact_per_pose",
    "interface_area",
    "ligand_buried_contact_area",
    "pose_burial_svg",
    "radius_of",
    "residue_key",
    "residue_sasa",
    "residue_slots",
    "sasa_of_atoms",
    "sasa_per_atom",
    "sasa_summary",
    "sphere_points",
]


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------

#: Van der Waals radii in Å: Bondi, *J. Phys. Chem.* **68** (1964) 441, with
#: the values the project already renders (`odock.gui.structure._RADII`) so the
#: surface area and the picture come from one table. An element outside the
#: table gets :data:`DEFAULT_RADIUS`; that is deliberately the carbon radius,
#: because an unparameterised atom in a docking input is almost always an
#: organic heavy atom the file typed sloppily.
ATOMIC_RADII: Dict[str, float] = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "Si": 2.10,
    "B": 1.92,
    "Se": 1.90,
    "As": 1.85,
    "Te": 2.06,
    # The metals a prepared PDBQT can carry. Bondi has no entry for most of
    # them; these are the ionic radii of Shannon (1976) rounded to the nearest
    # 0.05 Å for the common oxidation state, which is what a docking model
    # actually contains.
    "Mg": 1.73,
    "Mn": 1.73,
    "Zn": 1.39,
    "Ca": 1.94,
    "Fe": 1.72,
    "Na": 2.27,
    "K": 2.75,
    "Cu": 1.40,
    "Ni": 1.63,
    "Co": 1.67,
}

#: The radius an element outside :data:`ATOMIC_RADII` is given (carbon's).
DEFAULT_RADIUS = 1.70

#: The solvent probe radius in Å: a water molecule, the value Shrake & Rupley
#: used and the value every SASA implementation in the field defaults to.
PROBE_RADIUS = 1.4

#: Sphere points per atom. 92 is the count of the original paper; the residual
#: quadrature error is measured in `tests/test_sasa.py`.
DEFAULT_POINTS = 92

#: Grid cell width for the neighbour search, as a multiple of the largest
#: possible overlap. Two inflated spheres touch when their centres are closer
#: than ``R_i + R_j ≤ 2 · R_max``, and a cell that wide guarantees every such
#: partner is in one of the 27 cells around the probe atom.
_CELL_FACTOR = 2.0


def radius_of(element: object) -> float:
    """The Bondi van der Waals radius of an element symbol, in Å."""
    text = str(element or "").strip()
    if not text:
        return DEFAULT_RADIUS
    if len(text) == 1:
        key = text.upper()
    else:
        key = text[0].upper() + text[1:].lower()
    return ATOMIC_RADII.get(key, DEFAULT_RADIUS)


# ---------------------------------------------------------------------------
# sphere sampling
# ---------------------------------------------------------------------------


def sphere_points(count: int = DEFAULT_POINTS) -> np.ndarray:
    """``(count, 3)`` unit vectors on a golden-spiral (Fibonacci) lattice.

    The spiral is used instead of a latitude/longitude grid because a
    lat/long grid clusters points at the poles: for a *cap* — which is exactly
    what one atom cuts out of another — that bias shows up as an area error
    that grows with the cap's polar angle. The golden spiral is near-uniform,
    deterministic and needs no table.
    """
    n = int(count)
    if n <= 0:
        raise ValueError(f"count must be positive, got {count!r}")
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)
    index = np.arange(n, dtype=float)
    # The (i + 1/2)/n offset keeps the points symmetric about the equator.
    z = 1.0 - (2.0 * index + 1.0) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = index * golden_angle
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])


def _coords_and_radii(atoms, radii=None) -> Tuple[np.ndarray, np.ndarray]:
    """``(coords, radii)`` from Atom-like objects, or from raw coordinate rows.

    ``radii`` may be given explicitly (a per-atom array, index-aligned with
    ``atoms``) which is how a caller substitutes its own radius set. Without
    it the elements of the atoms are used.
    """
    if radii is not None:
        coords = np.asarray(
            [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
        ).reshape(-1, 3)
        table = np.asarray(radii, dtype=float).reshape(-1)
        if table.size != coords.shape[0]:
            raise ValueError(
                f"radii has {table.size} entries for {coords.shape[0]} atoms"
            )
        return coords, table
    rows = []
    table = []
    for atom in atoms:
        try:
            rows.append((float(atom.x), float(atom.y), float(atom.z)))
        except AttributeError:  # a plain (x, y, z) sequence
            rows.append((float(atom[0]), float(atom[1]), float(atom[2])))
        table.append(radius_of(getattr(atom, "element", None)))
    return np.asarray(rows, dtype=float).reshape(-1, 3), np.asarray(table, dtype=float)


class _CellList:
    """A uniform spatial hash over atom centres.

    One cell is as wide as the largest centre distance at which two probe
    spheres can still overlap, so a partner is always among the 27 cells around
    the atom being tested.
    """

    def __init__(self, coords: np.ndarray, cell: float) -> None:
        self.cell = max(float(cell), 1e-6)
        self.cells: Dict[Tuple[int, int, int], List[int]] = {}
        if coords.size == 0:
            return
        keys = np.floor(coords / self.cell).astype(np.int64)
        for index, key in enumerate(map(tuple, keys.tolist())):
            self.cells.setdefault(key, []).append(index)
        self._keys = keys

    def neighbours(self, index: int) -> List[int]:
        """Indices in the 27 cells around atom ``index`` (including itself)."""
        cx, cy, cz = (int(v) for v in self._keys[index])
        out: List[int] = []
        cells = self.cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    found = cells.get((cx + dx, cy + dy, cz + dz))
                    if found:
                        out.extend(found)
        return out


def sasa_per_atom(
    coords,
    radii,
    *,
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    extra_coords=None,
    extra_radii=None,
) -> np.ndarray:
    """Per-atom SASA in Å² for a set of spheres, occluded by an extra set.

    Parameters
    ----------
    coords, radii:
        ``(n, 3)`` centres and ``(n,)`` van der Waals radii, in Å.
    probe:
        Probe radius in Å (:data:`PROBE_RADIUS` by default).
    points:
        Sphere points per atom. The quadrature error falls with this count.
    extra_coords, extra_radii:
        A second sphere set that occludes but is not itself scored. This is
        what makes the *complex* area of the ligand a single call: pass the
        ligand as ``coords`` and the receptor as ``extra_coords``. The extras
        may be empty.

    Returns
    -------
    numpy.ndarray
        ``(n,) float64`` areas in Å². An atom completely inside another
        sphere is ``0.0``; an isolated atom is exactly ``4 π (r + probe)²``
        (every point is accessible), which is the property the test suite
        pins.
    """
    centres = np.asarray(coords, dtype=float).reshape(-1, 3)
    table = np.asarray(radii, dtype=float).reshape(-1)
    if centres.shape[0] != table.size:
        raise ValueError(
            f"radii has {table.size} entries for {centres.shape[0]} atoms"
        )
    count = centres.shape[0]
    out = np.zeros(count, dtype=float)
    if count == 0:
        return out

    probe = float(probe)
    if probe < 0.0:
        raise ValueError(f"probe must be non-negative, got {probe!r}")
    unit = sphere_points(points)
    inflated = table + probe

    if extra_coords is not None and len(np.asarray(extra_coords).reshape(-1, 3)):
        other = np.asarray(extra_coords, dtype=float).reshape(-1, 3)
        other_radii = np.asarray(extra_radii, dtype=float).reshape(-1)
        if other.shape[0] != other_radii.size:
            raise ValueError(
                f"extra_radii has {other_radii.size} entries for {other.shape[0]} atoms"
            )
        all_coords = np.concatenate([centres, other], axis=0)
        all_inflated = np.concatenate([inflated, other_radii + probe], axis=0)
        scored = count
    else:
        all_coords = centres
        all_inflated = inflated
        scored = count

    if all_coords.shape[0] == 0:  # pragma: no cover - defensive
        return out

    cell = _CELL_FACTOR * float(np.max(all_inflated))
    grid = _CellList(all_coords, cell)

    # One pass over the candidates turns the cell list into exact partner
    # lists: two probe spheres only interact when their centres are closer than
    # the sum of their inflated radii, and a scalar test rejects a distant
    # candidate far more cheaply than the point cloud test below.
    partners: List[List[int]] = [[] for _ in range(scored)]
    for index in range(scored):
        own = float(inflated[index])
        centre = centres[index]
        for candidate in grid.neighbours(index):
            if candidate == index:
                continue
            delta = centre - all_coords[candidate]
            limit = own + float(all_inflated[candidate])
            if float(np.dot(delta, delta)) < limit * limit:
                partners[index].append(candidate)

    for index in range(scored):
        radius = float(inflated[index])
        # The point cloud of this atom, generated once for every atom because
        # the unit sphere is shared.
        cloud = centres[index] + radius * unit
        blocked = np.zeros(unit.shape[0], dtype=bool)
        for partner in partners[index]:
            other_radius = float(all_inflated[partner])
            delta = cloud - all_coords[partner]
            distance_sq = np.einsum("ij,ij->i", delta, delta)
            blocked |= distance_sq < other_radius * other_radius
            if blocked.all():
                break
        accessible = 1.0 - float(blocked.mean())
        out[index] = accessible * 4.0 * math.pi * radius * radius
    return out


def sasa_of_atoms(
    atoms,
    *,
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    radii=None,
    other=(),
    other_radii=None,
) -> np.ndarray:
    """:func:`sasa_per_atom` for Atom-like objects (``x``, ``y``, ``z``, ``element``).

    ``other`` is a second atom sequence that occludes but is not scored.
    """
    coords, table = _coords_and_radii(atoms, radii)
    extra_coords = None
    if other_radii is None and len(other):
        extra_coords, other_radii = _coords_and_radii(other)
    elif len(other):
        extra_coords = np.asarray(
            [[float(a.x), float(a.y), float(a.z)] for a in other], dtype=float
        ).reshape(-1, 3)
    return sasa_per_atom(
        coords,
        table,
        probe=probe,
        points=points,
        extra_coords=extra_coords,
        extra_radii=other_radii,
    )


def sasa_summary(atoms, **kwargs) -> Dict[str, float]:
    """Total SASA plus the count of atoms that carry any exposure at all."""
    areas = sasa_of_atoms(atoms, **kwargs)
    exposed = int((areas > 1e-6).sum()) if areas.size else 0
    return {
        "atoms": int(areas.size),
        "total": float(areas.sum()) if areas.size else 0.0,
        "exposed_atoms": exposed,
        "buried_atoms": int(areas.size) - exposed,
    }


# ---------------------------------------------------------------------------
# residues
# ---------------------------------------------------------------------------


def residue_key(atom) -> Tuple[str, int, str]:
    """``(chain, res_id, res_name)`` — the residue identity used throughout."""
    try:
        res_id = int(getattr(atom, "res_id", 0) or 0)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        res_id = 0
    return (
        str(getattr(atom, "chain", "") or ""),
        res_id,
        str(getattr(atom, "res_name", "") or ""),
    )


def residue_slots(atoms) -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    """``(slot_of_atom, keys)`` mapping every atom onto its residue index.

    ``(chain, res_id, res_name)`` identifies a residue, so a residue whose name
    differs only by an insertion code is folded into one slot — the same rule
    ``odock.gui.bonds`` uses, so a burial report names the residue the user
    sees in the sequence ruler.
    """
    keys: List[Tuple[str, int, str]] = []
    lookup: Dict[Tuple[str, int, str], int] = {}
    slots = np.zeros(len(atoms), dtype=np.int64)
    for index, atom in enumerate(atoms):
        key = residue_key(atom)
        slot = lookup.get(key)
        if slot is None:
            slot = len(keys)
            lookup[key] = slot
            keys.append(key)
        slots[index] = slot
    return slots, keys


@dataclass(frozen=True)
class ResidueArea:
    """The SASA of one residue, and what a partner hides of it."""

    chain: str
    res_id: int
    res_name: str
    atoms: int
    area: float
    reference_area: Optional[float] = None

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain, self.res_id, self.res_name)

    @property
    def buried_area(self) -> Optional[float]:
        """``reference - complex`` in Å², never negative."""
        if self.reference_area is None:
            return None
        return max(0.0, float(self.reference_area) - float(self.area))

    @property
    def buried_fraction(self) -> Optional[float]:
        """The buried fraction of the reference area, in ``[0, 1]``."""
        if not self.reference_area:
            return None
        return min(1.0, self.buried_area / float(self.reference_area))

    @property
    def label(self) -> str:
        name = f"{self.res_name}{self.res_id}" if self.res_name else str(self.res_id)
        return f"{self.chain}/{name}" if self.chain else name

    def as_dict(self) -> Dict[str, object]:
        return {
            "key": list(self.key),
            "label": self.label,
            "chain": self.chain,
            "res_id": self.res_id,
            "res_name": self.res_name,
            "atoms": self.atoms,
            "area": round(float(self.area), 2),
            "reference_area": (
                None if self.reference_area is None else round(float(self.reference_area), 2)
            ),
            "buried_area": (
                None if self.buried_area is None else round(float(self.buried_area), 2)
            ),
            "buried_fraction": (
                None if self.buried_fraction is None else round(float(self.buried_fraction), 4)
            ),
        }


def _group_areas(atoms, areas: np.ndarray, slots, keys) -> List[ResidueArea]:
    totals = np.zeros(len(keys), dtype=float)
    counts = np.zeros(len(keys), dtype=np.int64)
    np.add.at(totals, slots, areas)
    np.add.at(counts, slots, 1)
    out: List[ResidueArea] = []
    for slot, (chain, res_id, res_name) in enumerate(keys):
        out.append(
            ResidueArea(
                chain=chain,
                res_id=int(res_id),
                res_name=res_name,
                atoms=int(counts[slot]),
                area=float(totals[slot]),
            )
        )
    return out


def residue_sasa(
    atoms,
    *,
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    radii=None,
    other=(),
    other_radii=None,
) -> List[ResidueArea]:
    """Per-residue SASA of ``atoms`` (optionally occluded by ``other``)."""
    areas = sasa_of_atoms(
        atoms,
        probe=probe,
        points=points,
        radii=radii,
        other=other,
        other_radii=other_radii,
    )
    slots, keys = residue_slots(atoms)
    return _group_areas(atoms, areas, slots, keys)


# ---------------------------------------------------------------------------
# burial
# ---------------------------------------------------------------------------


@dataclass
class BurialReport:
    """Per-residue burial against a stated reference.

    ``reference`` is the sentence the numbers have to be read with:

    ``"free"``
        every residue measured **on its own** (its own atoms only) — the
        classical folding burial. The per-residue ``buried_fraction`` is the
        number to quote from this one, because its denominator is that residue's
        own isolated area.
    ``"unbound"``
        the same structure **without the partner** (``other``) — the binding
        burial: how much surface the partner itself hides. These are the totals
        that are internally consistent, and they agree exactly with
        :func:`interface_area`.

    **Two traps this class guards against**, both of which cost a sibling
    workstream a −357 kcal/mol nonpolar term before they were found:

    * :attr:`reference_total` is the **sum of the reference areas**, not "the
      free area of the structure". For ``reference="free"`` that sum is over *N*
      residues each measured in isolation, so on a 1 994-atom receptor it is
      58 377 Å² against an intact-receptor SASA of 9 275 Å² — a factor of six,
      and subtracting it from the complex area is meaningless. Use
      :attr:`free_structure_area` (``None`` for ``"free"``, exact for
      ``"unbound"``) or :func:`interface_area` for "how much area did the partner
      bury".
    * :attr:`buried_area` and :attr:`buried_fraction` are therefore **``None``**
      for ``reference="free"``: a per-residue sum has no single comparable total.
      A caller that wants the folding ΔASA can sum ``residue.buried_area`` over
      :attr:`residues` (or read :attr:`summed_buried_area`); a caller that wants
      binding burial should use ``reference="unbound"``.
    """

    residues: List[ResidueArea] = field(default_factory=list)
    reference: str = "free"
    total_area: float = 0.0
    reference_total: float = 0.0
    partner_atoms: int = 0
    probe: float = PROBE_RADIUS
    points: int = DEFAULT_POINTS
    seconds: float = 0.0

    @property
    def free_structure_area(self) -> Optional[float]:
        """The intact structure's own area, or ``None`` when it was not measured.

        For ``reference="unbound"`` this is exactly ``sasa_of_atoms(atoms)`` — a
        property of the structure alone, independent of what it is compared
        against. For ``reference="free"`` the report never measured the intact
        structure, so it answers ``None`` rather than something that looks like an
        area and is not one.
        """
        return float(self.reference_total) if self.reference == "unbound" else None

    @property
    def summed_buried_area(self) -> float:
        """``Σ (reference_i − area_i)`` over residues, whatever the reference.

        A legitimate quantity — the ΔASA of folding for ``reference="free"`` —
        but a sum of *per-residue* differences, not a surface. Named so it cannot
        be mistaken for :attr:`buried_area`.
        """
        return float(
            sum(
                max(0.0, float(item.reference_area or 0.0) - float(item.area))
                for item in self.residues
            )
        )

    @property
    def buried_area(self) -> Optional[float]:
        """What the partner hid, in Å² — ``None`` unless ``reference="unbound"``.

        Only the ``"unbound"`` reference measures one structure twice (with and
        without the partner), so only there is the difference a surface. For
        ``reference="free"`` this is ``None`` by design; see the class docstring.
        """
        if self.reference != "unbound":
            return None
        return max(0.0, float(self.reference_total) - float(self.total_area))

    @property
    def buried_fraction(self) -> Optional[float]:
        """The buried fraction of the reference area, or ``None`` (see above)."""
        if self.reference != "unbound":
            return None
        if not self.reference_total:
            return 0.0
        return min(1.0, (self.buried_area or 0.0) / float(self.reference_total))

    def most_buried(self, count: int = 10) -> List[ResidueArea]:
        """The residues with the largest buried area, largest first."""
        ranked = [item for item in self.residues if (item.buried_area or 0.0) > 0.0]
        ranked.sort(key=lambda item: (-float(item.buried_area or 0.0), item.label))
        return ranked[: max(0, int(count))]

    def as_dict(self) -> Dict[str, object]:
        return {
            "reference": self.reference,
            "probe": self.probe,
            "points": self.points,
            "partner_atoms": self.partner_atoms,
            "total_area": round(float(self.total_area), 2),
            "reference_total": round(float(self.reference_total), 2),
            "free_structure_area": (
                None
                if self.free_structure_area is None
                else round(float(self.free_structure_area), 2)
            ),
            "buried_area": (
                None if self.buried_area is None else round(float(self.buried_area), 2)
            ),
            "buried_fraction": (
                None if self.buried_fraction is None else round(float(self.buried_fraction), 4)
            ),
            "summed_buried_area": round(float(self.summed_buried_area), 2),
            "seconds": round(float(self.seconds), 4),
            "residues": [item.as_dict() for item in self.residues],
        }

    def table(self, count: int = 12) -> str:
        """A fixed-width table of the most buried residues (for the log)."""
        rows = ["residue      area Å²  buried Å²  buried %"]
        for item in self.most_buried(count):
            rows.append(
                f"{item.label:<12} {item.area:8.1f} "
                f"{(item.buried_area or 0.0):9.1f} "
                f"{100.0 * (item.buried_fraction or 0.0):8.1f}"
            )
        share = (
            100.0 * self.summed_buried_area / self.reference_total
            if self.reference_total
            else 0.0
        )
        rows.append(
            f"{'TOTAL':<12} {self.total_area:8.1f} "
            f"{self.summed_buried_area:9.1f} {share:8.1f}"
        )
        return "\n".join(rows)


def burial(
    atoms,
    other=(),
    *,
    reference: str = "free",
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    radii=None,
    other_radii=None,
) -> BurialReport:
    """Per-residue SASA and burial of ``atoms`` in the presence of ``other``.

    Parameters
    ----------
    atoms:
        The structure to report on (normally the receptor).
    other:
        The partner that occludes it (normally the ligand). Empty is allowed:
        the report then describes the structure as it is, with the reference
        still meaningful for ``reference="free"``.
    reference:
        ``"free"`` (each residue alone) or ``"unbound"`` (``atoms`` without
        ``other``). See :class:`BurialReport` — in particular, use
        ``reference="unbound"`` (or :func:`interface_area`) when the question is
        how much surface the partner buried, because only that reference measures
        one structure twice.
    """
    started = time.perf_counter()
    atoms = list(atoms)
    other = list(other)
    key = str(reference).strip().lower()
    if key in ("free", "free_residue", "isolated", "residue"):
        key = "free"
    elif key in ("unbound", "apo", "complex", "receptor", "partner"):
        key = "unbound"
    else:
        raise ValueError(
            f"reference must be 'free' or 'unbound', got {reference!r}"
        )

    areas = sasa_of_atoms(
        atoms,
        probe=probe,
        points=points,
        radii=radii,
        other=other,
        other_radii=other_radii,
    )
    slots, keys = residue_slots(atoms)
    grouped = _group_areas(atoms, areas, slots, keys)

    reference_areas = np.zeros(len(keys), dtype=float)
    if key == "free":
        # Each residue on its own: its atoms, nothing else.
        for slot in range(len(keys)):
            members = [atom for index, atom in enumerate(atoms) if slots[index] == slot]
            if not members:
                continue
            sub_radii = None
            if radii is not None:
                sub_radii = [radii[index] for index in range(len(atoms)) if slots[index] == slot]
            reference_areas[slot] = float(
                sasa_of_atoms(members, probe=probe, points=points, radii=sub_radii).sum()
            )
    else:
        # The same structure without the partner: one SASA pass, no partner.
        alone = sasa_of_atoms(atoms, probe=probe, points=points, radii=radii)
        np.add.at(reference_areas, slots, alone)

    residues = [
        ResidueArea(
            chain=item.chain,
            res_id=item.res_id,
            res_name=item.res_name,
            atoms=item.atoms,
            area=item.area,
            reference_area=float(reference_areas[slot]),
        )
        for slot, item in enumerate(grouped)
    ]
    return BurialReport(
        residues=residues,
        reference=key,
        total_area=float(areas.sum()) if areas.size else 0.0,
        reference_total=float(reference_areas.sum()),
        partner_atoms=len(other),
        probe=float(probe),
        points=int(points),
        seconds=time.perf_counter() - started,
    )


# ---------------------------------------------------------------------------
# the ligand's own buried contact area
# ---------------------------------------------------------------------------


def ligand_buried_contact_area(
    ligand,
    receptor,
    *,
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    radii=None,
    receptor_radii=None,
) -> Dict[str, float]:
    """How much ligand surface the pocket hides, in Å².

    ``free_area`` is the ligand's SASA on its own, ``complex_area`` its SASA
    with the receptor present, and ``buried_area`` the difference — the area
    of ligand surface that is in contact with the receptor. Only the ligand's
    own points are ever evaluated, so this is cheap enough to run for every
    pose of a docking run (see :func:`buried_contact_per_pose`).

    The returned ``exposed_fraction`` is the share of the ligand's surface that
    stays in contact with solvent: a fully enclosed ligand is ``0.0``.
    """
    started = time.perf_counter()
    ligand = list(ligand)
    receptor = list(receptor)
    free = sasa_of_atoms(ligand, probe=probe, points=points, radii=radii)
    complex_areas = sasa_of_atoms(
        ligand,
        probe=probe,
        points=points,
        radii=radii,
        other=receptor,
        other_radii=receptor_radii,
    )
    free_total = float(free.sum()) if free.size else 0.0
    complex_total = float(complex_areas.sum()) if complex_areas.size else 0.0
    buried = max(0.0, free_total - complex_total)
    return {
        "free_area": free_total,
        "complex_area": complex_total,
        "buried_area": buried,
        "buried_fraction": (buried / free_total) if free_total > 0.0 else 0.0,
        "exposed_fraction": (complex_total / free_total) if free_total > 0.0 else 0.0,
        "ligand_atoms": float(len(ligand)),
        "receptor_atoms": float(len(receptor)),
        "contact_atoms": float(int((complex_areas < free - 1e-9).sum()))
        if complex_areas.size
        else 0.0,
        "seconds": time.perf_counter() - started,
    }


def interface_area(
    receptor,
    ligand,
    *,
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    radius: Optional[float] = None,
) -> Dict[str, float]:
    """The buried area of the *receptor* side of one complex, in Å².

    **This is the function to use for "how much area did the partner bury"**, on
    either side of the contact: the receptor SASA is measured twice — alone and
    with the partner present — and the difference is a surface, not a sum. It
    agrees exactly with ``burial(receptor, ligand, reference="unbound")``, and a
    test asserts that agreement so the two can never drift apart.

    ``radius`` limits the receptor to the atoms within that distance of the
    ligand, which is what makes this affordable on a 3 000-atom protein: the
    buried surface of a contact is local by construction, so a shell is not an
    approximation of the answer, it *is* the answer (the test suite pins that the
    shell and the whole protein agree).

    For the ligand's own hidden area use :func:`ligand_buried_contact_area`. Do
    **not** subtract ``burial(...).reference_total`` for
    ``reference="free"`` from anything: that number is the sum of every residue's
    *isolated* area (see :class:`BurialReport`).
    """
    receptor = list(receptor)
    ligand = list(ligand)
    used = receptor
    if radius is not None and ligand:
        centre = np.asarray(
            [[float(a.x), float(a.y), float(a.z)] for a in ligand], dtype=float
        )
        coords = np.asarray(
            [[float(a.x), float(a.y), float(a.z)] for a in receptor], dtype=float
        )
        if coords.size:
            # A whole-ligand bounding-box reject first, then the exact test.
            low = centre.min(axis=0) - float(radius)
            high = centre.max(axis=0) + float(radius)
            inside = np.all((coords >= low) & (coords <= high), axis=1)
            candidates = np.nonzero(inside)[0]
            keep = []
            for index in candidates:
                delta = centre - coords[index]
                if float(np.min(np.einsum("ij,ij->i", delta, delta))) <= radius * radius:
                    keep.append(int(index))
            used = [receptor[index] for index in keep]

    free = sasa_of_atoms(used, probe=probe, points=points)
    bound = sasa_of_atoms(
        used, probe=probe, points=points, other=ligand, other_radii=None
    )
    free_total = float(free.sum()) if free.size else 0.0
    bound_total = float(bound.sum()) if bound.size else 0.0
    buried = max(0.0, free_total - bound_total)
    return {
        "free_area": free_total,
        "complex_area": bound_total,
        "buried_area": buried,
        "buried_fraction": (buried / free_total) if free_total > 0.0 else 0.0,
        "receptor_atoms_used": float(len(used)),
        "receptor_atoms_total": float(len(receptor)),
        "seconds": 0.0,
    }


def buried_contact_per_pose(
    poses: Sequence[Sequence],
    receptor,
    *,
    probe: float = PROBE_RADIUS,
    points: int = DEFAULT_POINTS,
    affinity: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """One :func:`ligand_buried_contact_area` record per pose.

    ``affinity`` (kcal/mol, one per pose) is copied into the record when it is
    given, so a caller can plot buried area against score without joining two
    lists itself.
    """
    out: List[Dict[str, float]] = []
    for index, pose in enumerate(poses):
        record = ligand_buried_contact_area(pose, receptor, probe=probe, points=points)
        record["pose"] = float(index)
        if affinity is not None and index < len(affinity):
            record["affinity"] = float(affinity[index])
        out.append(record)
    return out


#: Colours of the pose-burial figure, in the same visual language as the
#: surface palettes: the buried area is the quantity, the score is annotation.
_PLOT_INK = "#20242c"
_PLOT_GRID = "#c8ccd4"
_PLOT_BAR = "#3f7fc4"
_PLOT_BAR_TOP = "#f0a02a"
_PLOT_BG = "#ffffff"


def pose_burial_svg(
    records: Sequence[Dict[str, float]],
    *,
    width: int = 760,
    height: int = 380,
    title: str = "buried contact area per pose",
) -> str:
    """A bar chart of the buried contact area of every pose, as SVG.

    SVG rather than a bitmap because the figure is meant to be *used*: it drops
    into a paper, a slide or a notebook at any size, and it is text, so it can
    be diffed. The affinity of each pose is printed above its bar rather than
    given a second axis — the point of the figure is the shape of the burial
    across a pose set, and a second y-axis would invite reading a correlation
    that the numbers do not establish.

    A record is one entry of :func:`buried_contact_per_pose`; the function only
    needs ``buried_area`` and, optionally, ``affinity``. An empty list still
    produces a valid (empty) figure with its axes, so a caller never has to
    special-case "nothing to plot".
    """
    rows = list(records or [])
    areas = [max(0.0, float(row.get("buried_area", 0.0) or 0.0)) for row in rows]
    top = max(areas) if areas else 1.0
    top = top if top > 0.0 else 1.0
    # A round axis maximum, so the gridlines land on readable numbers.
    step = 10.0 ** math.floor(math.log10(top))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if top <= multiple * step:
            top = multiple * step
            break
    else:  # pragma: no cover - defensive
        top = 10.0 * step

    margin_left, margin_right, margin_top, margin_bottom = 62.0, 18.0, 44.0, 46.0
    plot_width = max(40.0, width - margin_left - margin_right)
    plot_height = max(40.0, height - margin_top - margin_bottom)
    count = max(1, len(rows))
    slot = plot_width / count
    bar_width = max(2.0, min(46.0, slot * 0.62))

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" '
        f'height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        f'<rect width="{int(width)}" height="{int(height)}" fill="{_PLOT_BG}"/>',
        f'<text x="{margin_left:.0f}" y="24" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="14" font-weight="600" fill="{_PLOT_INK}">{_escape(title)}</text>',
    ]

    # The y axis: five gridlines with their values in Å².
    ticks = 5
    for index in range(ticks + 1):
        value = top * index / ticks
        y = margin_top + plot_height - plot_height * index / ticks
        parts.append(
            f'<line x1="{margin_left:.1f}" y1="{y:.1f}" x2="{margin_left + plot_width:.1f}" '
            f'y2="{y:.1f}" stroke="{_PLOT_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="11" '
            f'fill="{_PLOT_INK}">{value:.0f}</text>'
        )
    parts.append(
        f'<text x="16" y="{margin_top - 14:.1f}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="10" fill="#6b7280">buried Å²</text>'
    )

    for index, row in enumerate(rows):
        area = max(0.0, float(row.get("buried_area", 0.0) or 0.0))
        bar_height = plot_height * (area / top) if top > 0.0 else 0.0
        x = margin_left + slot * index + (slot - bar_width) / 2.0
        y = margin_top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{_PLOT_BAR}"/>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="2.5" '
            f'fill="{_PLOT_BAR_TOP}"/>'
        )
        label = f"{area:.0f}"
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{max(margin_top + 10, y - 14):.1f}" '
            f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" fill="{_PLOT_INK}">{label}</text>'
        )
        score = row.get("affinity")
        if score is not None:
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{max(margin_top + 22, y - 2):.1f}" '
                f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
                f'font-size="10" fill="#6b7280">{float(score):.1f}</text>'
            )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 18:.1f}" '
            f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="11" fill="{_PLOT_INK}">{int(row.get("pose", index)) + 1}</text>'
        )

    parts.append(
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 10:.0f}" '
        f'text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="11" fill="#6b7280">pose (the score in kcal/mol is above each bar)</text>'
    )
    parts.append(
        f'<line x1="{margin_left:.1f}" y1="{margin_top + plot_height:.1f}" '
        f'x2="{margin_left + plot_width:.1f}" y2="{margin_top + plot_height:.1f}" '
        f'stroke="{_PLOT_INK}" stroke-width="1"/>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _escape(text: object) -> str:
    """Minimal XML escaping, so a title can never break the document."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

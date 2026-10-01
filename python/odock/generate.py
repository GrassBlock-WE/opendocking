# SPDX-License-Identifier: GPL-3.0-or-later
"""Generating an ensemble from a single structure, by sampling side-chain rotamers.

Every measurement in :mod:`odock.ensemble` and :mod:`odock.pockets` needs several
conformations of the same receptor, and most users have exactly one crystal
structure.  This module builds a *modelled* ensemble from it: the residues lining
the binding site are rotated about their :math:`\\chi` dihedrals on a staggered
grid, each candidate is clash-filtered against the rest of the protein, and the
survivors are assembled into a set of conformations that share the experimental
backbone.

What it is, stated up front and repeated in ``docs/GENERATED_ENSEMBLES.md``:

* it is a **rigid-rotamer model** — the side chain keeps its own internal
  geometry and only the :math:`\\chi` angles change;
* it **cannot move the backbone**, so it cannot produce the very thing the
  experimental pairs in this project demonstrate: the 3ERT/1ERE_A difference is
  helix 12 moving 2.01 Å, and no amount of side-chain sampling will reach it;
* it samples a **documented grid**, not a statistical rotamer library: three
  staggered values per :math:`\\chi` (:math:`-60°`, :math:`180°`, :math:`60°`)
  plus the native value read from the structure.  A backbone-dependent library
  would be better and is the obvious upgrade;
* every conformation is **deterministic** for a given seed and grid, and the
  report says exactly which residues moved, how far, and how many candidates the
  clash filter rejected.

The point of the generated set is to be *usable*: it is a drop-in input for
``odock ensemble dock``, ``odock ensemble screen`` and ``odock ensemble pockets``
(``superpose=False``, because everything is already in one frame).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ensemble import Conformation, EnsembleError, read_conformations, select_site
from .pocket import VDW_RADII
from .prepare import BoxSpec

__all__ = [
    "CHI_ATOMS",
    "DEFAULT_CLASH_TOLERANCE",
    "DEFAULT_GRID_ANGLES",
    "DEFAULT_MAX_ROTAMERS",
    "DEFAULT_ROTAMER_RMSD",
    "DEFAULT_SITE_RADIUS",
    "GeneratedEnsemble",
    "Rotamer",
    "RotamerResidue",
    "add_generate_subparser",
    "dihedral",
    "enumerate_rotamers",
    "generate_ensemble",
    "set_dihedral",
]

#: How close two heavy atoms have to be to count as bonded when the graph of a
#: residue is perceived from coordinates.  The longest bond in a protein side
#: chain is C–S at 1.81 Å, so 2.0 Å is a safe ceiling and never bridges two
#: residues (the shortest inter-residue heavy-atom contact is a hydrogen bond at
#: ~2.6 Å).
BOND_CUTOFF = 2.0

#: The :math:`\\chi` dihedral definitions per residue, as PDB atom-name
#: quadruples (the four atoms whose torsion is the dihedral).  Standard
#: IUPAC/Dunbrack definitions.  Glycine and alanine have none; proline's ring
#: makes a rigid-rotamer model meaningless, so both are skipped (and reported).
CHI_ATOMS: Dict[str, Tuple[Tuple[str, str, str, str], ...]] = {
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}

#: Alternative atom names seen in the wild for the fourth atom of a dihedral.
CHI_ALIASES: Dict[str, Tuple[str, ...]] = {
    "CD1": ("CD1", "CD"),
    "CG1": ("CG1", "CG"),
    "OD1": ("OD1", "OD"),
    "OE1": ("OE1", "OE"),
    "ND1": ("ND1", "ND"),
    "SD": ("SD",),
    "SG": ("SG",),
}

#: A non-standard residue name that maps onto a standard one (protonation and
#: modification variants) uses the standard residue's dihedrals.
def _chi_for(res_name: str) -> Tuple[Tuple[str, str, str, str], ...]:
    from .ensemble import CODES

    name = str(res_name).strip().upper()
    if name in CHI_ATOMS:
        return CHI_ATOMS[name]
    code = CODES.get(name)
    for candidate, table in CHI_ATOMS.items():
        if CODES.get(candidate) == code and code:
            return table
    return ()


#: The staggered grid: three values per :math:`\\chi`, the classic
#: ``g+``/``t``/``g-`` rotamers.
DEFAULT_GRID_ANGLES: Tuple[float, ...] = (-60.0, 180.0, 60.0)

#: How many rotamers per residue survive into the ensemble (the native one
#: included).
DEFAULT_MAX_ROTAMERS = 3

#: Two rotamers of one residue closer than this (heavy-atom RMSD, Å) count as
#: the same rotamer and the duplicate is dropped.
DEFAULT_ROTAMER_RMSD = 0.5

#: A candidate is rejected when a moved atom comes closer than the sum of the
#: van der Waals radii minus this tolerance (Å).  Positive values give the
#: sampling room to breathe, which is what a rigid-rotamer model needs because
#: the rest of the protein is frozen at its experimental position.
DEFAULT_CLASH_TOLERANCE = 0.4

#: Residues whose side chains are sampled: those with a heavy atom within this
#: distance of the site centre (or of any ligand atom).
DEFAULT_SITE_RADIUS = 6.0

#: Residue types that are never sampled, with the reason.
SKIPPED_RESIDUES: Dict[str, str] = {
    "GLY": "no side chain",
    "ALA": "no chi dihedral",
    "PRO": "the ring makes a rigid-rotamer model meaningless",
}


# ---------------------------------------------------------------------------
# Dihedral geometry: rotate a side chain about a bond
# ---------------------------------------------------------------------------


def _coords(residue: Any) -> np.ndarray:
    return np.array(
        [[float(atom.x), float(atom.y), float(atom.z)] for atom in residue.atoms], dtype=float
    ).reshape(-1, 3)


def _element(atom: Any) -> str:
    return str(getattr(atom, "element", "") or "C")


def _radii(residue_or_atoms: Sequence[Any]) -> np.ndarray:
    return np.array(
        [VDW_RADII.get(_element(atom), 1.70) for atom in residue_or_atoms], dtype=float
    )


def dihedral(points: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """The dihedral angle ``a-b-c-d`` in degrees, in ``(-180, 180]``."""
    p0, p1, p2, p3 = (np.asarray(points, dtype=float)[i] for i in (a, b, c, d))
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m, n2))
    return float(math.degrees(math.atan2(y, x)))


def set_dihedral(
    points: np.ndarray, a: int, b: int, c: int, d: int, target: float,
    moving: Sequence[int],
) -> np.ndarray:
    """Rotate the atoms in `moving` about the bond ``b-c`` to make ``a-b-c-d`` = `target`.

    The sign of the rotation that a dihedral read-out corresponds to is a
    convention (IUPAC measures it looking down the ``b->c`` axis), and getting it
    backwards is a silent 2φ error in every sampled rotamer.  So the rotation is
    applied, the dihedral is read back, and the opposite rotation is used if the
    first attempt did not land on the target -- correctness by construction rather
    than by remembering a sign.
    """
    out = np.array(points, dtype=float).copy()
    current = dihedral(out, a, b, c, d)
    delta = math.radians(float(target) - current)
    for sign in (1.0, -1.0):
        candidate = _rotate_about(out, b, c, sign * delta, moving)
        achieved = (dihedral(candidate, a, b, c, d) - float(target) + 180.0) % 360.0 - 180.0
        if abs(achieved) <= 1e-6:
            return candidate
    return candidate


def _rotate_about(
    points: np.ndarray, b: int, c: int, delta: float, moving: Sequence[int]
) -> np.ndarray:
    """Rotate `moving` by `delta` radians about the axis ``b-c`` (Rodrigues)."""
    out = np.array(points, dtype=float).copy()
    axis = out[c] - out[b]
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:  # pragma: no cover - degenerate geometry
        return out
    axis = axis / norm
    origin = out[b]
    cos_d, sin_d = math.cos(delta), math.sin(delta)
    indices = list(moving)
    rel = out[indices] - origin
    rotated = (
        rel * cos_d
        + np.cross(axis, rel) * sin_d
        + np.outer((rel @ axis), axis) * (1.0 - cos_d)
    )
    out[indices] = rotated + origin
    return out


def _side_chain_graph(residue: Any) -> List[List[int]]:
    """Intra-residue bonds, perceived from coordinates (heavy atoms only)."""
    coords = _coords(residue)
    n = coords.shape[0]
    adjacency: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(coords[i] - coords[j])) <= BOND_CUTOFF:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def _moving_atoms(adjacency: Sequence[Sequence[int]], pivot: int, far: int) -> List[int]:
    """Atoms on the `far` side of the `pivot`-`far` bond (the part that rotates)."""
    seen = {far}
    stack = [far]
    out: List[int] = []
    while stack:
        node = stack.pop()
        out.append(node)
        for neighbour in adjacency[node]:
            if neighbour == pivot or neighbour in seen:
                continue
            seen.add(neighbour)
            stack.append(neighbour)
    return sorted(out)


@dataclass
class Rotamer:
    """One candidate side-chain conformation of one residue."""

    angles: Tuple[float, ...]
    coords: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))
    clashes: int = 0
    native: bool = False
    #: Heavy-atom RMSD of the moved side chain against the native rotamer (Å).
    rmsd: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "angles": [float(value) for value in self.angles],
            "clashes": int(self.clashes),
            "native": bool(self.native),
            "rmsd": float(self.rmsd),
        }


@dataclass
class RotamerResidue:
    """The rotamer sampling of one site residue."""

    residue: Any
    label: str
    chi: int
    enumerated: int = 0
    #: Candidates that pass the clash filter.
    clash_free: List[Rotamer] = field(default_factory=list)
    #: Candidates the clash filter rejects.
    rejected: int = 0
    #: Clash-free candidates dropped as duplicates of one already kept.
    duplicates: int = 0
    #: Clash-free, distinct candidates dropped because ``--max-rotamers`` is full.
    capped: int = 0
    #: The rotamers that go into the ensemble (the native one first).
    kept: List[Rotamer] = field(default_factory=list)
    unresolved: str = ""
    #: Native chi values, in degrees.
    native_angles: Tuple[float, ...] = ()

    @property
    def rotamers(self) -> int:
        """Rotamers that survived the clash filter."""
        return len(self.clash_free)

    @property
    def n_kept(self) -> int:
        """Rotamers that the ensemble actually uses."""
        return len(self.kept)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "residue": self.label,
            "chi": int(self.chi),
            "native_angles": [float(value) for value in self.native_angles],
            "enumerated": int(self.enumerated),
            "clash_free": int(self.rotamers),
            "rejected_by_clash": int(self.rejected),
            "duplicates": int(self.duplicates),
            "capped": int(self.capped),
            "kept": int(self.n_kept),
            "unresolved": self.unresolved,
            "rotamers": [rotamer.as_dict() for rotamer in self.clash_free],
        }


def enumerate_rotamers(
    conformation: Conformation,
    residues: Sequence[Any],
    *,
    other_atoms: Optional[Sequence[Any]] = None,
    grid: Sequence[float] = DEFAULT_GRID_ANGLES,
    clash_tolerance: float = DEFAULT_CLASH_TOLERANCE,
    rotamer_rmsd: float = DEFAULT_ROTAMER_RMSD,
    max_rotamers: int = DEFAULT_MAX_ROTAMERS,
) -> List[RotamerResidue]:
    """Enumerate, clash-filter and prune the rotamers of `residues`.

    Parameters
    ----------
    conformation
        The structure being sampled (used for the residue's own geometry).
    residues
        :class:`odock.ensemble.Residue` objects to sample.
    other_atoms
        The set of atoms a moved side chain must not clash with.  Defaults to every
        heavy atom of the conformation; the residue being sampled is then removed
        from that set per residue, while the *other* sampled residues stay in it at
        their native rotamers -- which is exactly the one-at-a-time model: a
        rotamer must be clear of the frozen rest of the protein, including its
        neighbours in the site.
    grid
        The :math:`\\chi` grid, in degrees.  The native value is always included.
    clash_tolerance
        A candidate is rejected when a moved heavy atom is closer to a fixed atom
        than the sum of their van der Waals radii minus this tolerance.
    rotamer_rmsd
        Duplicates closer than this to an already-kept rotamer are dropped.
    max_rotamers
        How many rotamers per residue survive (the native one included); extras are
        chosen for maximum spread, so the ensemble spans the residue's mobility
        instead of clustering around one value.
    """
    if other_atoms is None:
        other_atoms = [
            atom
            for chain in conformation.chain_residues().values()
            for residue in chain
            for atom in residue.atoms
            if not atom.is_hydrogen
        ]
    universe = list(other_atoms)

    out: List[RotamerResidue] = []
    for residue in residues:
        key = residue.key
        fixed_atoms = [
            atom
            for atom in universe
            if not (
                atom.chain == key[0] and atom.res_id == key[1] and atom.res_name == key[2]
            )
        ]
        fixed = np.array(
            [[float(atom.x), float(atom.y), float(atom.z)] for atom in fixed_atoms],
            dtype=float,
        ).reshape(-1, 3)
        fixed_radii = _radii(fixed_atoms)
        entry = RotamerResidue(
            residue=residue,
            label=residue.label,
            chi=len(_chi_for(residue.res_name)),
        )
        table = _chi_for(residue.res_name)
        if not table:
            entry.unresolved = SKIPPED_RESIDUES.get(
                residue.res_name.strip().upper(), "no chi dihedral defined"
            )
            out.append(entry)
            continue

        names = [atom.name for atom in residue.atoms]
        indices: List[Tuple[int, int, int, int]] = []
        for quadruple in table:
            resolved = []
            for name in quadruple:
                for candidate in CHI_ALIASES.get(name, (name,)):
                    if candidate in names:
                        resolved.append(names.index(candidate))
                        break
                else:
                    break
            if len(resolved) != 4:
                break
            indices.append((resolved[0], resolved[1], resolved[2], resolved[3]))
        if len(indices) != len(table):
            entry.unresolved = (
                "the side chain is incomplete in this structure "
                f"(needs {', '.join(table[-1])})"
            )
            out.append(entry)
            continue

        adjacency = _side_chain_graph(residue)
        native_coords = _coords(residue)
        native_angles = tuple(
            dihedral(native_coords, *quadruple) for quadruple in indices
        )
        entry.native_angles = native_angles

        # Every combination of grid angles, with the native value added to each
        # chi's option list so the experimental rotamer is always a candidate.
        options = []
        for value in native_angles:
            values = [float(value)]
            for angle in grid:
                if abs(((float(angle) - float(value) + 180.0) % 360.0) - 180.0) > 1e-6:
                    values.append(float(angle))
            options.append(values)

        moving = [
            _moving_atoms(adjacency, quadruple[2], quadruple[3])
            for quadruple in indices
        ]
        heavy = [index for index, atom in enumerate(residue.atoms) if not atom.is_hydrogen]
        side_radii = _radii(residue.atoms)

        candidates: List[Rotamer] = []
        entry.enumerated = int(np.prod([len(values) for values in options]))
        for combination in _product(options):
            coords = native_coords
            for quadruple, angle, atoms in zip(indices, combination, moving):
                coords = set_dihedral(coords, *quadruple, float(angle), atoms)
            # Clash filter: only the atoms that actually moved are tested.
            moved = sorted({index for atoms in moving for index in atoms})
            clashes = _count_clashes(
                coords[moved], side_radii[moved], fixed, fixed_radii, clash_tolerance
            )
            candidate = Rotamer(
                angles=tuple(float(value) for value in combination),
                coords=coords,
                clashes=clashes,
                native=all(
                    abs(((a - b + 180.0) % 360.0) - 180.0) <= 1e-6
                    for a, b in zip(combination, native_angles)
                ),
                rmsd=float(
                    np.sqrt(((coords[moved] - native_coords[moved]) ** 2).sum(axis=1).mean())
                )
                if moved
                else 0.0,
            )
            if candidate.clashes == 0:
                candidates.append(candidate)
            else:
                entry.rejected += 1

        # Deduplicate, then keep the native plus the most spread-out survivors.
        survivors: List[Rotamer] = []
        for candidate in sorted(candidates, key=lambda item: (not item.native, item.rmsd)):
            if any(
                float(
                    np.sqrt(
                        ((candidate.coords[heavy] - other.coords[heavy]) ** 2)
                        .sum(axis=1)
                        .mean()
                    )
                )
                <= rotamer_rmsd
                for other in survivors
            ):
                entry.duplicates += 1
                continue
            survivors.append(candidate)
        entry.clash_free = candidates
        entry.duplicates = len(candidates) - len(survivors)
        if not candidates:
            # Not a crash and not a silent skip: a residue whose every rotamer,
            # including the experimental one, clashes with the frozen protein is
            # reported so the reader knows why it does not appear in the ensemble.
            entry.unresolved = (
                f"every rotamer clashes with the frozen protein "
                f"({entry.rejected} of {entry.enumerated} rejected at "
                f"{_clash_tolerance_text(clash_tolerance)})"
            )
        native_kept = [item for item in survivors if item.native]
        others = sorted(
            (item for item in survivors if not item.native),
            key=lambda item: -item.rmsd,
        )
        chosen_others = others[: max(0, int(max_rotamers) - 1)]
        entry.capped = len(others) - len(chosen_others)
        kept = native_kept[:1] + chosen_others
        kept.sort(key=lambda item: (not item.native, -item.rmsd))
        entry.kept = kept
        if kept and not any(item.native for item in kept):
            entry.unresolved = (
                "the experimental rotamer clashes with the frozen protein; the "
                "closest clear rotamer is used instead"
            )
        out.append(entry)
    return out


def _clash_tolerance_text(tolerance: float) -> str:
    return f"a tolerance of {float(tolerance):.1f} A"


def _product(options: Sequence[Sequence[float]]) -> Iterable[Tuple[float, ...]]:
    """Cartesian product, iterative so an empty option list yields nothing."""
    result: List[Tuple[float, ...]] = [()]
    for values in options:
        result = [prefix + (float(value),) for prefix in result for value in values]
    return result


def _count_clashes(
    moved: np.ndarray,
    moved_radii: np.ndarray,
    fixed: np.ndarray,
    fixed_radii: np.ndarray,
    tolerance: float,
) -> int:
    """How many moved atoms are inside a fixed atom's van der Waals shell?"""
    if moved.size == 0 or fixed.size == 0:
        return 0
    distance = np.sqrt(((moved[:, None, :] - fixed[None, :, :]) ** 2).sum(axis=2))
    limit = moved_radii[:, None] + fixed_radii[None, :] - float(tolerance)
    return int((distance < limit).any(axis=1).sum())


# ---------------------------------------------------------------------------
# Assembling the ensemble
# ---------------------------------------------------------------------------

#: The measured site differences of the experimental pairs this project validates
#: against (``docs/ENSEMBLE.md`` section 1).  The generated ensemble's side-chain
#: displacement is printed next to these, because that comparison is the only
#: thing that says whether a modelled ensemble is the right size to be useful.
#: Note carefully what is being compared: the experimental *site RMSD* is a CA
#: RMSD after a binding-site fit (it includes backbone motion), while a generated
#: conformation has no backbone motion at all; the comparable number is the
#: per-residue heavy-atom displacement.
EXPERIMENTAL_REFERENCES: Tuple[Dict[str, Any], ...] = (
    {
        "pair": "3ERT vs 1ERE_A (ERα antagonist vs agonist)",
        "site_rmsd_ca": 0.444,
        "max_displacement": 2.010,
        "moved_by": "helix 12 (LEU525) and ASP351",
        "side_chains_only": False,
    },
    {
        "pair": "1HVR vs 1HXW (HIV-1 protease isolates)",
        "site_rmsd_ca": 0.407,
        "max_displacement": 1.589,
        "moved_by": "the Ile50 flap tips",
        "side_chains_only": False,
    },
    {
        "pair": "3PTB vs 2PTN (trypsin, the control)",
        "site_rmsd_ca": 0.144,
        "max_displacement": 1.133,
        "moved_by": "GLN192, a surface loop",
        "side_chains_only": False,
    },
)


@dataclass
class GeneratedEnsemble:
    """A modelled ensemble built from one structure by side-chain sampling."""

    native: Conformation
    conformations: List[Conformation] = field(default_factory=list)
    residues: List[RotamerResidue] = field(default_factory=list)
    site_labels: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)
    #: Per member: side-chain RMSD of the site, the largest per-residue
    #: displacement, and which residues took a non-native rotamer.
    spread: List[Dict[str, Any]] = field(default_factory=list)
    #: ``residue label -> largest side-chain displacement over the members`` (Å).
    displacement: Dict[str, float] = field(default_factory=dict)
    source: Optional[Path] = None

    @property
    def n_conformations(self) -> int:
        return len(self.conformations)

    @property
    def sampled(self) -> List[RotamerResidue]:
        return [entry for entry in self.residues if entry.rotamers > 1]

    @property
    def unresolved(self) -> List[RotamerResidue]:
        return [entry for entry in self.residues if entry.unresolved]

    # -- reporting --------------------------------------------------------

    def table(self) -> str:
        """One row per sampled residue: chi, enumerated, accepted, kept, rejected."""
        headers = [
            "residue", "rotamers/chi", "enumerated", "clash-free", "kept",
            "rejected by clash", "duplicates", "capped", "notes",
        ]
        rows = []
        for entry in self.residues:
            rows.append([
                entry.label,
                str(entry.chi) if entry.chi else "-",
                str(entry.enumerated),
                str(entry.rotamers),
                str(entry.n_kept),
                str(entry.rejected),
                str(entry.duplicates),
                str(entry.capped),
                entry.unresolved or "",
            ])
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * width for width in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def ensemble_table(self, limit: int = 20) -> str:
        """One row per conformation: its side-chain RMSD and what moved."""
        headers = ["member", "site RMSD (A)", "max displacement (A)", "changed residues"]
        rows = []
        for index, entry in enumerate(self.spread[: max(1, limit)]):
            rows.append([
                "native" if index == 0 else f"gen_{index}",
                f"{entry['site_rmsd']:.3f}",
                f"{entry['max_displacement']:.3f}",
                ", ".join(entry["changed"][:4]) or "-",
            ])
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * width for width in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def comparison(self) -> List[Dict[str, Any]]:
        """The generated side-chain spread against the experimental pairs.

        This is the check that makes a modelled ensemble useful rather than
        decorative: if the generated set moves side chains far *less* than the
        experimental pairs this project validates against, docking into it will
        not test the same thing.
        """
        generated_site = [
            entry["site_rmsd"] for entry in self.spread[1:]
        ]
        generated_max = max(
            (entry["max_displacement"] for entry in self.spread), default=0.0
        )
        out = []
        for reference in EXPERIMENTAL_REFERENCES:
            out.append(
                {
                    "pair": reference["pair"],
                    "experimental_site_rmsd_ca": reference["site_rmsd_ca"],
                    "experimental_max_displacement": reference["max_displacement"],
                    "experimental_moved_by": reference["moved_by"],
                    "generated_site_rmsd_mean": (
                        float(np.mean(generated_site)) if generated_site else 0.0
                    ),
                    "generated_site_rmsd_max": (
                        float(max(generated_site)) if generated_site else 0.0
                    ),
                    "generated_max_displacement": generated_max,
                    "amplitude_ratio": (
                        generated_max / reference["max_displacement"]
                        if reference["max_displacement"]
                        else float("nan")
                    ),
                    "generated_backbone_rmsd": 0.0,
                }
            )
        return out

    def text(self, *, limit: int = 20) -> str:
        """The whole human-readable report."""
        lines = [
            f"OpenDocking generated ensemble — {self.n_conformations} conformation(s) "
            f"from {self.native.label}",
            "",
            "sampling: "
            + ", ".join(f"{key}={value}" for key, value in self.parameters.items()),
            "",
            f"site residues ({len(self.site_labels)}): " + ", ".join(self.site_labels),
            "",
            self.table(),
            "",
            f"site side-chain spread over {self.n_conformations} member(s) "
            f"(backbone unchanged by construction):",
            self.ensemble_table(limit),
            "",
            "comparison with the experimental pairs this project validates against "
            "(docs/ENSEMBLE.md section 1):",
        ]
        headers = ["pair", "site RMSD CA (A)", "max disp (A)", "generated max disp (A)", "ratio"]
        rows = []
        for entry in self.comparison():
            rows.append([
                entry["pair"],
                f"{entry['experimental_site_rmsd_ca']:.3f}",
                f"{entry['experimental_max_displacement']:.3f}",
                f"{entry['generated_max_displacement']:.3f}",
                f"{entry['amplitude_ratio']:.2f}",
            ])
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines.append(render(headers))
        lines.append("  ".join("-" * width for width in widths))
        lines.extend(render(row) for row in rows)
        lines += [
            "",
            "the experimental site RMSD is a CA RMSD after a binding-site fit and "
            "therefore includes backbone motion; a generated member has none "
            "(0.000 A) and only its side chains move, so the comparable figure is "
            "the per-residue displacement.",
            "a generated ensemble is NOT experimental evidence: it samples a "
            "rotamer grid, it cannot move the backbone, and it will miss anything "
            "that requires a loop or a helix to rearrange.",
        ]
        if self.unresolved:
            lines += [
                "",
                "residues that could not be sampled as asked:",
            ] + [
                f"  {entry.label}: {entry.unresolved}" for entry in self.unresolved
            ]
        if self.warnings:
            lines += [""] + [f"warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": None if self.source is None else str(self.source),
            "native": self.native.label,
            "n_conformations": self.n_conformations,
            "site": list(self.site_labels),
            "parameters": dict(self.parameters),
            "elapsed": round(float(self.elapsed), 4),
            "residues": [entry.as_dict() for entry in self.residues],
            "spread": list(self.spread),
            "displacement": {key: float(value) for key, value in self.displacement.items()},
            "comparison": self.comparison(),
            "warnings": list(self.warnings),
        }

    def write(
        self,
        outdir: Union[str, Path],
        *,
        prepare: bool = False,
        strip: Optional[Sequence[str]] = None,
        keep_water: bool = False,
    ) -> List[Path]:
        """Write every member as ``<outdir>/<label>.pdb`` (and ``.pdbqt``).

        The member files are ordinary PDB documents in one frame, so they are
        accepted by ``odock ensemble dock/screen/pockets --no-superpose`` and by
        ``odock prepare receptor``.
        """
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for index, conformation in enumerate(self.conformations):
            name = "native" if index == 0 else f"gen_{index}"
            target = directory / f"{name}.pdb"
            target.write_text(conformation.text(), encoding="utf-8")
            written.append(target)
        if prepare:
            from .prepare import prepare_receptor

            for index, path in enumerate(list(written)):
                _receptor, text, _report = prepare_receptor(
                    str(path), None, keep_water=keep_water, strip=list(strip or []) or None
                )
                (path.with_suffix(".pdbqt")).write_text(text, encoding="utf-8")
        return written


def generate_ensemble(
    conformation: Conformation,
    *,
    site: Optional[str] = None,
    box: Optional[BoxSpec] = None,
    site_radius: float = DEFAULT_SITE_RADIUS,
    max_site_residues: int = 40,
    grid: Sequence[float] = DEFAULT_GRID_ANGLES,
    max_rotamers: int = DEFAULT_MAX_ROTAMERS,
    rotamer_rmsd: float = DEFAULT_ROTAMER_RMSD,
    clash_tolerance: float = DEFAULT_CLASH_TOLERANCE,
    min_rmsd: float = 0.5,
    combinations: int = 0,
    seed: int = 20240101,
    source: Optional[Path] = None,
) -> GeneratedEnsemble:
    """Build a modelled ensemble by sampling the site residues' rotamers.

    Parameters
    ----------
    conformation
        The single experimental structure to sample.
    site, box, site_radius, max_site_residues
        Which residues to sample: exactly the site definition
        :func:`odock.ensemble.select_site` already uses, so a generated ensemble
        and an experimental pair describe the same binding site.
    grid, max_rotamers, rotamer_rmsd, clash_tolerance
        Passed to :func:`enumerate_rotamers`.
    min_rmsd
        Ensemble pruning: a candidate conformation whose site side chains are
        within this RMSD of an already-kept member is dropped, so the members are
        distinct rather than 30 near-copies of each other.
    combinations
        Extra members sampled from *simultaneous* rotamer changes (a seeded random
        combination per member), on top of the one-residue-at-a-time sweep.  ``0``
        keeps only the sweep, which is the interpretable default: one member per
        rotamer of one residue.
    seed
        Seed for the combination sampling; the sweep is deterministic without it.
    """
    import time

    started = time.perf_counter()
    warnings: List[str] = []
    site_residues = select_site(
        conformation, site=site, box=box, radius=site_radius, max_residues=max_site_residues
    )
    residues = [entry.residue for entry in site_residues]
    if not residues:
        raise EnsembleError(
            "no site residues to sample: pass --box, --site or --site-ligand"
        )
    entries = enumerate_rotamers(
        conformation, residues, grid=grid, clash_tolerance=clash_tolerance,
        rotamer_rmsd=rotamer_rmsd, max_rotamers=max_rotamers,
    )
    if not any(entry.rotamers > 1 for entry in entries):
        warnings.append(
            "no residue has a rotamer that passes the clash filter: this site is "
            "too packed to sample in a rigid-rotamer model, or the tolerance is "
            "too strict"
        )

    native_coords = {id(atom): np.array([atom.x, atom.y, atom.z]) for atom in conformation.atoms}
    # Members are rebuilt with *new* Atom objects (their coordinates changed), so
    # atoms are identified by their position in the conformation's atom list,
    # which every member preserves, and never by object identity.
    position_of = {id(atom): index for index, atom in enumerate(conformation.atoms)}
    site_positions = [
        position_of[id(atom)]
        for residue in residues
        for atom in residue.atoms
        if not atom.is_hydrogen
    ]
    site_position_set = set(site_positions)
    native_site = np.array(
        [
            [conformation.atoms[position].x, conformation.atoms[position].y,
             conformation.atoms[position].z]
            for position in site_positions
        ],
        dtype=float,
    ).reshape(-1, 3)
    residue_positions = {
        entry.label: [
            position_of[id(atom)] for atom in entry.residue.atoms if not atom.is_hydrogen
        ]
        for entry in entries
    }

    def member_site(member: Conformation) -> np.ndarray:
        return np.array(
            [
                [member.atoms[position].x, member.atoms[position].y, member.atoms[position].z]
                for position in site_positions
            ],
            dtype=float,
        ).reshape(-1, 3)

    def build(changes: Dict[Any, np.ndarray]) -> Conformation:
        """The native structure with the given residues' coordinates replaced."""
        atoms = []
        records = []
        from .ensemble import Atom as _Atom
        from .ensemble import _rewrite_coords

        for atom, line in zip(conformation.atoms, conformation.records):
            replacement = changes.get(id(atom))
            if replacement is None:
                atoms.append(atom)
                records.append(line)
                continue
            x, y, z = (float(value) for value in replacement)
            atoms.append(
                _Atom(
                    name=atom.name, element=atom.element, res_name=atom.res_name,
                    res_id=atom.res_id, chain=atom.chain, x=x, y=y, z=z,
                    altloc=atom.altloc, record=atom.record,
                )
            )
            records.append(_rewrite_coords(line, (x, y, z)) if line else line)
        return Conformation(
            label=conformation.label, atoms=atoms, records=records,
            path=conformation.path, model=conformation.model, source=conformation.source,
        )

    def coordinates(entry: RotamerResidue, rotamer: Rotamer) -> Dict[Any, np.ndarray]:
        return {
            id(atom): rotamer.coords[index]
            for index, atom in enumerate(entry.residue.atoms)
        }

    members: List[Tuple[str, Conformation, List[str]]] = [
        ("native", conformation, [])
    ]
    # One member per extra rotamer of one residue: the interpretable sweep.
    for entry in entries:
        for rotamer in entry.kept:
            if rotamer.native:
                continue
            members.append(
                (
                    f"{entry.label} chi={','.join(f'{value:.0f}' for value in rotamer.angles)}",
                    build(coordinates(entry, rotamer)),
                    [entry.label],
                )
            )
    # Optional simultaneous changes, seeded.
    if combinations and len(entries) > 0:
        rng = np.random.default_rng(int(seed))
        candidates = [entry for entry in entries if len(entry.kept) > 1]
        for index in range(int(combinations)):
            if not candidates:
                break
            changes: Dict[Any, np.ndarray] = {}
            changed: List[str] = []
            for entry in candidates:
                choice = int(rng.integers(0, len(entry.kept)))
                rotamer = entry.kept[choice]
                if rotamer.native:
                    continue
                changes.update(coordinates(entry, rotamer))
                changed.append(f"{entry.label}")
            members.append((f"combo_{index}", build(changes), changed))

    # Prune: greedy, keeping the native first and then the members farthest from
    # everything kept so far, so the ensemble spans the mobility.
    kept: List[Tuple[str, Conformation, List[str], np.ndarray]] = []
    for label, member, changed in members:
        coords = member_site(member)
        if kept:
            distances = [
                float(np.sqrt(((coords - other[3]) ** 2).sum(axis=1).mean()))
                for other in kept
            ]
            if min(distances) < float(min_rmsd):
                continue
        kept.append((label, member, changed, coords))

    spread: List[Dict[str, Any]] = []
    displacement: Dict[str, float] = {entry.label: 0.0 for entry in entries}
    for index, (label, member, changed, coords) in enumerate(kept):
        rmsd = float(np.sqrt(((coords - native_site) ** 2).sum(axis=1).mean()))
        per_residue: Dict[str, float] = {}
        for entry in entries:
            positions = residue_positions[entry.label]
            if not positions:
                continue
            native_residue = np.array(
                [
                    [
                        conformation.atoms[position].x, conformation.atoms[position].y,
                        conformation.atoms[position].z,
                    ]
                    for position in positions
                ],
                dtype=float,
            ).reshape(-1, 3)
            member_residue = np.array(
                [
                    [member.atoms[position].x, member.atoms[position].y,
                     member.atoms[position].z]
                    for position in positions
                ],
                dtype=float,
            ).reshape(-1, 3)
            if native_residue.shape != member_residue.shape or native_residue.size == 0:
                continue
            value = float(
                np.sqrt(((member_residue - native_residue) ** 2).sum(axis=1).mean())
            )
            per_residue[entry.label] = value
            displacement[entry.label] = max(displacement.get(entry.label, 0.0), value)
        spread.append(
            {
                "member": label,
                "index": index,
                "site_rmsd": rmsd,
                "max_displacement": max(per_residue.values(), default=0.0),
                "displacement": per_residue,
                "changed": changed,
            }
        )

    return GeneratedEnsemble(
        native=conformation,
        conformations=[member for _label, member, _changed, _coords in kept],
        residues=entries,
        site_labels=[entry.label for entry in site_residues],
        parameters={
            "grid": [float(value) for value in grid],
            "max_rotamers": int(max_rotamers),
            "rotamer_rmsd": float(rotamer_rmsd),
            "clash_tolerance": float(clash_tolerance),
            "site_radius": float(site_radius),
            "min_rmsd": float(min_rmsd),
            "combinations": int(combinations),
            "seed": int(seed),
        },
        elapsed=time.perf_counter() - started,
        warnings=warnings,
        spread=spread,
        displacement=displacement,
        source=source,
    )


# ---------------------------------------------------------------------------
# The command line: `odock ensemble generate`
# ---------------------------------------------------------------------------


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def cmd_ensemble_generate(args) -> int:
    """``odock ensemble generate``: sample a site's side chains into an ensemble."""
    from .ensemble import _cli_box, _cli_write_json

    paths = [str(path) for entry in (getattr(args, "receptor", None) or []) for path in (
        entry if isinstance(entry, (list, tuple)) else [entry]
    )]
    if not paths:
        raise SystemExit("error: pass one -r/--receptor FILE")
    if len(paths) > 1:
        raise SystemExit(
            "error: `ensemble generate` builds an ensemble from *one* structure; "
            "use `ensemble align` for a set you already have"
        )
    try:
        conformations = read_conformations(paths, keep_water=bool(args.keep_water))
        conformation = conformations[0]
        box = None
        if args.box_ligand:
            # The site is defined by where the co-crystallised ligand sits, so the
            # box is derived from it here rather than read from a file.
            from .ensemble import ligand_coords
            from .prepare import box_from_points

            points = ligand_coords(conformation, args.box_ligand)
            box = box_from_points(
                points, buffer=float(args.buffer), spacing=float(args.spacing)
            )
            _cli_eprint(
                f"box: {box} (from residue {args.box_ligand} of {conformation.label})"
            )
        elif getattr(args, "box", None) or (args.center and args.size):
            box = _cli_box(args)
        generated = generate_ensemble(
            conformation,
            site=args.site,
            box=box,
            site_radius=float(args.site_radius),
            max_site_residues=int(args.max_site_residues),
            grid=_parse_angles(args.grid_angles),
            max_rotamers=int(args.max_rotamers),
            rotamer_rmsd=float(args.rotamer_rmsd),
            clash_tolerance=float(args.clash_tolerance),
            min_rmsd=float(args.min_rmsd),
            combinations=int(args.combinations),
            seed=int(args.seed),
            source=Path(paths[0]),
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble generate: error: {exc}")
        return int(exc.code)

    if not args.quiet:
        print(generated.text(limit=int(args.top)))
        print()
        _cli_eprint(
            f"generated {generated.n_conformations} conformation(s) in "
            f"{generated.elapsed:.1f} s"
        )
    if args.outdir:
        written = generated.write(
            args.outdir, prepare=bool(args.pdbqt), strip=args.strip,
            keep_water=bool(args.keep_water),
        )
        _cli_eprint(f"wrote {len(written)} structure(s) to {args.outdir}")
    if args.json_out:
        _cli_write_json(args.json_out, generated.as_dict())
    return 0


def _parse_angles(text: str) -> Tuple[float, ...]:
    """``"-60,180,60"`` -> ``(-60.0, 180.0, 60.0)``."""
    values = []
    for part in str(text).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise EnsembleError(
                f"cannot read the rotamer grid {text!r}: {token!r} is not a number"
            ) from exc
    if not values:
        raise EnsembleError("the rotamer grid is empty; pass --grid-angles -60,180,60")
    return tuple(values)


def add_generate_subparser(ensub: Any) -> None:
    """Register ``odock ensemble generate`` on the ensemble subparsers."""
    parser = ensub.add_parser(
        "generate",
        help="build an ensemble from one structure by sampling site rotamers",
        description=(
            "Most users have one crystal structure and every ensemble measurement "
            "needs several.  This samples the side chains of the binding-site "
            "residues about their chi dihedrals on a staggered grid, clash-filters "
            "each candidate against the rest of the frozen protein, assembles the "
            "survivors into a set of conformations, and prints the site's "
            "side-chain spread next to the experimental pairs this project "
            "validates against.  It is a MODEL: the backbone never moves, so it "
            "cannot reproduce a difference like helix 12's 2.01 A."
        ),
    )
    parser.add_argument(
        "-r", "--receptor", action="append", nargs="+", required=True, metavar="FILE",
        help="the one structure to sample (PDB or PDBQT)",
    )
    parser.add_argument("--box", help="box JSON written by `odock box`")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--box-ligand", metavar="RESNAME",
        help="take the site from the position of this residue (a co-crystallised ligand)",
    )
    parser.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    parser.add_argument("--spacing", type=float, default=0.375, help="grid spacing of the box (Å)")
    parser.add_argument(
        "--site", metavar="RES[,RES...]",
        help="the residues to sample, e.g. --site ASP189,SER190; the default is the "
             "residues around the box centre",
    )
    parser.add_argument(
        "--site-radius", type=float, default=DEFAULT_SITE_RADIUS,
        help="residues within this distance of the site centre are sampled (Å)",
    )
    parser.add_argument(
        "--max-site-residues", type=int, default=40,
        help="cap on the number of sampled residues",
    )
    parser.add_argument(
        "--grid-angles", default=",".join(str(value) for value in DEFAULT_GRID_ANGLES),
        help="the chi grid in degrees; the native value is always included as well",
    )
    parser.add_argument(
        "--max-rotamers", type=int, default=DEFAULT_MAX_ROTAMERS,
        help="rotamers kept per residue (the native one included); the extras are the "
             "most spread-out survivors",
    )
    parser.add_argument(
        "--rotamer-rmsd", type=float, default=DEFAULT_ROTAMER_RMSD,
        help="rotamers closer than this to one already kept are duplicates (Å)",
    )
    parser.add_argument(
        "--clash-tolerance", type=float, default=DEFAULT_CLASH_TOLERANCE,
        help="how far inside the van der Waals shell a moved atom may come before "
             "the rotamer is rejected (Å); 0 means a hard filter",
    )
    parser.add_argument(
        "--min-rmsd", type=float, default=0.5,
        help="ensemble pruning: a member whose site side chains are within this "
             "RMSD of one already kept is dropped (Å)",
    )
    parser.add_argument(
        "--combinations", type=int, default=0, metavar="N",
        help="also sample N members that change several residues at once (seeded)",
    )
    parser.add_argument("--seed", type=int, default=20240101, help="seed for --combinations")
    parser.add_argument(
        "-o", "--outdir", help="write each member as <outdir>/native.pdb, gen_1.pdb, ...",
    )
    parser.add_argument(
        "--pdbqt", action="store_true",
        help="also prepare each written member as a receptor PDBQT (needs RDKit)",
    )
    parser.add_argument("--strip", nargs="+", metavar="RESNAME", help="residues to delete when preparing")
    parser.add_argument("--keep-water", action="store_true", help="keep crystallographic waters")
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    parser.add_argument("--json-out", help="write the whole report as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no report")
    parser.set_defaults(func=cmd_ensemble_generate)



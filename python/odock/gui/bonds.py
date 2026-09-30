# SPDX-License-Identifier: GPL-3.0-or-later
"""Bond perception for the 3-D workbench, following PyMOL's model.

The user's worry is concrete: a rendered stick must join the *right* two atoms,
so the rule here is *prefer a missing bond over an invented one*, and every rule
below is documented with the reason it exists.

PyMOL's model
-------------
PyMOL never guesses from one distance threshold.  It exposes two settings and
combines them with a per-element valence table:

``connect_mode``
    ``0`` bonds *any* pair closer than ``connect_cutoff`` ("distance only");
    ``3`` (the default) is the automatic mode.
``connect_cutoff``
    the distance used by mode ``0`` (3.6 A here, PyMOL's default).
``valence``
    ``(0, 0, 0, 0)`` by default, meaning "use the built-in per-element limits",
    so an atom is never given more bonds than its element allows.

:func:`perceive_bonds` reproduces that shape, with the automatic mode built on
a published covalent-radius table instead of on a single cutoff:

1. **candidates** — every pair of atoms no further apart than
   ``r_cov(i) + r_cov(j) + 0.45 A``.  The radii are Pyykkö & Atsumi,
   *Chem. Eur. J.* **15** (2009) 186, single-bond values (``H`` 0.32, ``C``
   0.75, ``N`` 0.71, ``O`` 0.63, ``S`` 1.03 A, ...).  The 0.45 A tolerance
   sits comfortably above every real single bond and far below a hydrogen bond:
   an ``N···H`` contact is ~1.9 A, while ``r_N + r_H + 0.45`` is 1.48 A, and an
   ``O···H`` contact of 2.6 A is more than an Angstrom beyond its 1.40 A limit.
2. **greedy valence** — candidates are sorted shortest first and accepted while
   *both* atoms are still below their valence limit, so the strongest contacts
   win the remaining valence and an over-coordinated site degrades gracefully.
3. **chemistry guards** — H–H is never bonded (the classic methyl/H-bond false
   positive), metals and noble gases are never bonded (an ion is drawn as a
   sphere, not wired to its coordination sphere), and no element exceeds the
   built-in valence table.  The table holds each element's *maximum* — the value
   :func:`expected_valence` returns, so the cap never truncates a legitimate
   structure: H 1, C 4, N 4 (3 in an amine, 4 in an ammonium), O 2, S 6 (2 in a
   thioether, 6 in a sulfone), P 5 (3/5), halogens 1.  :func:`normal_valences`
   gives the narrower list of unmistakeable valences for a caller that wants it,
   and ``valence=`` overrides single entries of the table.

``kind="receptor"`` additionally forbids bonds that span two residues, except a
peptide ``C–N`` and a disulfide ``S–S`` — the only two covalent cross-links a
protein has.  ``kind="ligand"`` allows them, because a prepared ligand is
routinely split over several HETATM records: the bundled 3PTB benzamidine is
``BEN A 1`` for its nine heavy atoms and ``LIG A 1`` for its four hydrogens.

A cross-residue ``C–N`` counts as a peptide bond when the two atoms are in one
chain, one is the backbone carbonyl carbon (atom name ``C``) and the other the
backbone amide nitrogen (name ``N``), and their residues are neighbours in the
chain — either consecutive residue numbers or consecutive in the file.  The file
order matters: a deposited structure routinely skips numbers where a loop is
missing, and those links are perfectly real.  The bundled 3PTB.pdb has six such
gaps, and each of the six ``C···N`` pairs is a genuine 1.29–1.36 A peptide bond
that a "consecutive residue number" rule silently loses.

Distance mode
-------------
``connect_mode=0`` is the naive PyMOL behaviour kept for comparison: any pair
within ``connect_cutoff``, with no valence table, no H–H rule and no
metal/noble-gas rule (the ``kind`` residue rule still applies, because it is a
property of the structure, not of the mode).  It is deliberately *more*
promiscuous than automatic mode — on a receptor it wires hydrogens to their
hydrogen-bond partners and joins every atom inside a coordination sphere — which
is exactly the behaviour this module exists to replace.  It remains useful as a
debugging view and as the reference for :func:`bond_report`'s ``over_valent``.

Bond orders are a rendering hint only: an organic bond (C, N, O, S) much shorter
than the sum of its single-bond radii is promoted to double or triple while
spare valence allows it.  Nothing about *which* atoms are joined depends on
them, and an aromatic C–C (1.39 A / 1.50 A = 0.927) deliberately stays single.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Bond",
    "CONNECT_MODE_DISTANCE",
    "CONNECT_MODE_AUTO",
    "DEFAULT_CONNECT_CUTOFF",
    "MAX_BOND_TOLERANCE",
    "COVALENT_RADII",
    "SKIP_ELEMENTS",
    "perceive_bonds",
    "bond_report",
    "expected_valence",
    "normal_valences",
    "is_bondable",
    "covalent_radius",
    "bond_pairs",
    "guess_bonds",
]

# ---------------------------------------------------------------------------
# PyMOL's two knobs
# ---------------------------------------------------------------------------

#: ``connect_mode`` 0 — bond any pair closer than ``connect_cutoff``.
CONNECT_MODE_DISTANCE = 0

#: ``connect_mode`` 3 — automatic (PyMOL's default), the mode described above.
CONNECT_MODE_AUTO = 3

#: The distance used by :data:`CONNECT_MODE_DISTANCE` (PyMOL's default).
DEFAULT_CONNECT_CUTOFF = 3.6

#: How far beyond ``r_cov(i) + r_cov(j)`` a pair may still be a bond, in A.
MAX_BOND_TOLERANCE = 0.45


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------

#: Single-bond covalent radii in A: Pyykkö & Atsumi, *Chem. Eur. J.* **15**
#: (2009) 186.  An unknown element falls back to 0.8 A, which makes it bond only
#: over a very short distance.
COVALENT_RADII: Dict[str, float] = {
    "H": 0.32, "He": 0.46,
    "Li": 1.33, "Be": 1.02, "B": 0.85, "C": 0.75, "N": 0.71, "O": 0.63,
    "F": 0.64, "Ne": 0.67,
    "Na": 1.55, "Mg": 1.39, "Al": 1.24, "Si": 1.17, "P": 1.11, "S": 1.03,
    "Cl": 0.99, "Ar": 0.96,
    "K": 1.96, "Ca": 1.71, "Sc": 1.48, "Ti": 1.36, "V": 1.34, "Cr": 1.22,
    "Mn": 1.19, "Fe": 1.16, "Co": 1.11, "Ni": 1.10, "Cu": 1.12, "Zn": 1.18,
    "Ga": 1.24, "Ge": 1.21, "As": 1.21, "Se": 1.16, "Br": 1.14, "Kr": 1.17,
    "Rb": 2.10, "Sr": 1.85, "Y": 1.63, "Zr": 1.54, "Nb": 1.47, "Mo": 1.38,
    "Tc": 1.28, "Ru": 1.25, "Rh": 1.25, "Pd": 1.20, "Ag": 1.28, "Cd": 1.36,
    "In": 1.42, "Sn": 1.40, "Sb": 1.40, "Te": 1.36, "I": 1.39, "Xe": 1.40,
    "Cs": 2.32, "Ba": 1.96, "La": 1.80, "Ce": 1.63, "Lu": 1.62, "Hf": 1.52,
    "Ta": 1.46, "W": 1.37, "Re": 1.31, "Os": 1.29, "Ir": 1.22, "Pt": 1.23,
    "Au": 1.24, "Hg": 1.33, "Tl": 1.44, "Pb": 1.44, "Bi": 1.51,
}

#: Elements that never take part in a covalent bond in this viewer: the metals
#: (an ion is drawn as a sphere, not wired to its coordination sphere) and the
#: noble gases.
SKIP_ELEMENTS = frozenset(
    {
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
        "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce",
        "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
        "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
        "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    }
)

#: The largest covalent radius of an element this module is willing to bond,
#: used to size the neighbourhood search.
_MAX_BONDABLE_RADIUS = max(
    radius
    for symbol, radius in COVALENT_RADII.items()
    if symbol not in SKIP_ELEMENTS
)

#: The bond count an element is *allowed* to reach: the hard cap of automatic
#: mode, and the value :func:`expected_valence` returns.  ``N`` reaches 4 for an
#: ammonium/protonated amine; ``S`` and ``P`` reach their hypervalent forms
#: (sulfone/sulfate, phosphate); the halogens take one bond.
_MAX_VALENCE: Dict[str, int] = {
    "H": 1, "B": 3, "C": 4, "N": 4, "O": 2, "F": 1,
    "Si": 4, "P": 5, "S": 6, "Cl": 1,
    "Ge": 4, "As": 3, "Se": 2, "Br": 1,
    "Te": 2, "I": 1,
}

#: The valences that count as *normal* for an element, exposed by
#: :func:`normal_valences` for callers that want a chemistry check tighter than
#: the hard cap.  A hydrogen-suppressed PDBQT reports its backbone nitrogens at
#: 2 and its carbonyl oxygens at 1, so these lists describe the chemistry of a
#: fully protonated molecule, not the contents of a prepared file.
_NORMAL_VALENCES: Dict[str, Tuple[int, ...]] = {
    "H": (1,), "B": (3,), "C": (4,), "N": (3, 4), "O": (2,), "F": (1,),
    "Si": (4,), "P": (3, 5), "S": (2, 6), "Cl": (1,),
    "Ge": (4,), "As": (3, 5), "Se": (2, 4, 6), "Br": (1,), "Te": (2, 4, 6),
    "I": (1,),
}

#: Valence for an element that is neither in :data:`SKIP_ELEMENTS` nor in
#: :data:`_MAX_VALENCE`.
DEFAULT_VALENCE = 4

#: The elements whose bond lengths are calibrated well enough to guess a bond
#: order.  Everything else stays single.
_ORDER_ELEMENTS = frozenset({"C", "N", "O", "S"})

#: A bond this much shorter than the sum of the two single-bond radii is drawn
#: as a double / triple bond, while spare valence allows it.  Calibrated on real
#: bonds: C=O 1.23/1.38 = 0.89, C=C 1.34/1.50 = 0.89, C=N 1.28/1.46 = 0.88,
#: C#C 1.20/1.50 = 0.80, C#N 1.16/1.46 = 0.79, while a C–C single bond is
#: 1.53/1.50 = 1.02 and an aromatic C–C is 1.39/1.50 = 0.93 (kept single: the
#: ring is delocalised and a wrong double bond looks worse than a plain stick).
_DOUBLE_RATIO = 0.90
_TRIPLE_RATIO = 0.80


@dataclass(frozen=True)
class Bond:
    """One perceived bond.  ``a < b`` always, ``order`` is 1, 2 or 3.

    It unpacks like the ``(a, b)`` tuple the workbench used before, so picking
    code and ``for i, j in bonds`` keep working, while the bond order stays
    available to the renderer.
    """

    a: int
    b: int
    order: int = 1

    def __iter__(self):
        return iter((self.a, self.b))

    def as_tuple(self) -> Tuple[int, int]:
        return (self.a, self.b)


# ---------------------------------------------------------------------------
# element helpers
# ---------------------------------------------------------------------------


def _normalise_symbol(symbol: object) -> str:
    """``"CA"`` / ``"ca"`` -> ``"Ca"``; an empty symbol becomes ``"C"``."""
    text = str(symbol or "").strip()
    if not text:
        return "C"
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:].lower()


def covalent_radius(symbol: str) -> float:
    """Pyykkö single-bond radius of an element in A (0.8 A if unknown)."""
    return COVALENT_RADII.get(_normalise_symbol(symbol), 0.8)


def is_bondable(symbol: str) -> bool:
    """False for metals and noble gases, which are never bonded."""
    return _normalise_symbol(symbol) not in SKIP_ELEMENTS


def expected_valence(element: str) -> int:
    """The largest number of bonds an atom of ``element`` may carry.

    This is the cap automatic mode enforces and the value
    :func:`bond_report` compares against.  ``N`` is 4 (ammonium), ``S`` 6 and
    ``P`` 5 (their expanded valences), the halogens 1; a metal or a noble gas is
    0, so it is never bonded; an element outside the table gets
    :data:`DEFAULT_VALENCE`.
    """
    symbol = _normalise_symbol(element)
    if symbol in SKIP_ELEMENTS:
        return 0
    return _MAX_VALENCE.get(symbol, DEFAULT_VALENCE)


def normal_valences(element: str) -> Tuple[int, ...]:
    """The valences that are chemically unremarkable for ``element``."""
    return _NORMAL_VALENCES.get(_normalise_symbol(element), ())


# ---------------------------------------------------------------------------
# input plumbing
# ---------------------------------------------------------------------------


def _points_from(atoms: Sequence, coords) -> List[Tuple[float, float, float]]:
    """The coordinates to work on, as plain ``(x, y, z)`` float triples."""
    if coords is not None:
        points = []
        for position in coords:
            try:
                x, y, z = position[0], position[1], position[2]
            except (TypeError, IndexError, KeyError) as exc:
                raise ValueError(f"coordinates must be (n, 3): {position!r}") from exc
            points.append((float(x), float(y), float(z)))
        return points
    return [(float(a.x), float(a.y), float(a.z)) for a in atoms]


def _symbols_from(atoms: Sequence, elements, count: int) -> List[str]:
    if elements is not None:
        symbols = [_normalise_symbol(symbol) for symbol in elements]
        if len(symbols) != count:
            raise ValueError(
                f"elements has {len(symbols)} entries for {count} coordinates"
            )
        return symbols
    if len(atoms) != count:
        raise ValueError(
            "pass elements= when atoms and coords disagree on the atom count"
        )
    out = []
    for atom in atoms:
        symbol = getattr(atom, "element", None) or getattr(atom, "name", "") or "C"
        out.append(_normalise_symbol(symbol))
    return out


def _residues_from(atoms: Sequence, count: int):
    """Residue identity, so the ``kind="receptor"`` rules can run.

    Returns ``(slot_of, chains, res_ids, order)``: the slot of each atom's
    residue, the chain and number of each slot, and the position of each slot
    inside its own chain (file order).  A residue is identified by
    ``(chain, res_id)``, which merges a PDB insertion code (``GLY 184A`` is
    written with the same number as ``TYR 184``) into the residue it belongs to.
    """
    keys: Dict[Tuple[str, int], int] = {}
    chains: List[str] = []
    res_ids: List[int] = []
    order: List[int] = []
    per_chain: Dict[str, int] = {}
    slot_of: List[int] = []
    for index in range(count):
        atom = atoms[index] if index < len(atoms) else None
        chain = str(getattr(atom, "chain", "") or "")
        try:
            res_id = int(getattr(atom, "res_id", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            res_id = 0
        key = (chain, res_id)
        slot = keys.get(key)
        if slot is None:
            slot = len(keys)
            keys[key] = slot
            chains.append(chain)
            res_ids.append(res_id)
            order.append(per_chain.get(chain, 0))
            per_chain[chain] = order[-1] + 1
        slot_of.append(slot)
    return slot_of, chains, res_ids, order


def _atom_names_from(atoms: Sequence, count: int) -> List[str]:
    names: List[str] = []
    for index in range(count):
        atom = atoms[index] if index < len(atoms) else None
        names.append(str(getattr(atom, "name", "") or "").strip().upper())
    return names


def _atom_label(atom, index: int) -> str:
    """``A/ASP102:OD1`` — enough context to find the atom in the viewer."""
    if atom is None:
        return f"atom {index}"
    name = str(getattr(atom, "name", "") or "?").strip()
    res_name = str(getattr(atom, "res_name", "") or "").strip()
    chain = str(getattr(atom, "chain", "") or "").strip()
    try:
        res_id = int(getattr(atom, "res_id", 0) or 0)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        res_id = 0
    prefix = f"{res_name}{res_id}" if res_name else ""
    if chain:
        prefix = f"{chain}/{prefix}" if prefix else chain
    return f"{prefix}:{name}" if prefix else name


def _element_of(atom) -> str:
    symbol = getattr(atom, "element", None) or getattr(atom, "name", "") or "C"
    return _normalise_symbol(symbol)


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------


def _grid_pairs(points, reach: Sequence[float]):
    """Yield ``(i, j, distance)`` for ``i < j`` pairs in neighbouring cells.

    Atoms are binned into cubes as wide as the largest reach, so a pair that can
    satisfy any reach test lies in the same or in a face-adjacent cube and the
    walk stays linear in the atom count instead of quadratic — which is what
    keeps a whole-receptor load interactive.  Distances are only *bounded* here;
    the caller applies the exact per-mode test.
    """
    count = len(points)
    cell = max(reach) if reach else 0.0
    if cell <= 0.0:
        return
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for index in range(count):
        x, y, z = points[index]
        key = (
            int(math.floor(x / cell)),
            int(math.floor(y / cell)),
            int(math.floor(z / cell)),
        )
        grid.setdefault(key, []).append(index)

    for i in range(count):
        xi, yi, zi = points[i]
        cx = int(math.floor(xi / cell))
        cy = int(math.floor(yi / cell))
        cz = int(math.floor(zi / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if j <= i:
                            continue
                        xj, yj, zj = points[j]
                        ex = xj - xi
                        ey = yj - yi
                        ez = zj - zi
                        distance = math.sqrt(ex * ex + ey * ey + ez * ez)
                        if distance > reach[i] and distance > reach[j]:
                            continue
                        yield (i, j, distance)


def _candidates(
    points,
    symbols: Sequence[str],
    connect_mode: int,
    connect_cutoff: float,
) -> List[Tuple[float, int, int]]:
    """Geometric candidates ``(distance, i, j)`` for one connect mode.

    No residue rule and no valence rule is applied here: automatic mode leaves
    both to :func:`perceive_bonds`, and distance mode applies neither.
    """
    radii = [covalent_radius(symbol) for symbol in symbols]
    if connect_mode == CONNECT_MODE_AUTO:
        reach = [r + MAX_BOND_TOLERANCE + _MAX_BONDABLE_RADIUS for r in radii]
        cutoff = 0.0
    else:
        cutoff = float(connect_cutoff)
        if cutoff <= 0.0:
            raise ValueError(f"connect_cutoff must be positive, got {connect_cutoff!r}")
        reach = [cutoff] * len(points)

    out: List[Tuple[float, int, int]] = []
    for i, j, distance in _grid_pairs(points, reach):
        if distance < 1e-6:
            # Coincident atoms: a zero-length stick is never a bond.
            continue
        if connect_mode == CONNECT_MODE_AUTO:
            if not is_bondable(symbols[i]) or not is_bondable(symbols[j]):
                continue
            if symbols[i] == "H" and symbols[j] == "H":
                # Two hydrogens are never bonded: a methyl's H...H contact is
                # 1.78 A and an H-bond donor/acceptor pair can be closer still.
                continue
            if distance > radii[i] + radii[j] + MAX_BOND_TOLERANCE:
                continue
        elif distance > cutoff:
            continue
        out.append((distance, i, j))
    return out


# ---------------------------------------------------------------------------
# chemistry guards
# ---------------------------------------------------------------------------


def _is_peptide_c_n(name_i: str, name_j: str) -> bool:
    """Are these the backbone carbonyl carbon and the backbone amide nitrogen?"""
    if not name_i or not name_j:
        return False
    return {name_i, name_j} == {"C", "N"}


def _cross_residue_allowed(
    i: int,
    j: int,
    symbols: Sequence[str],
    names: Sequence[str],
    slot_of: Sequence[int],
    chains: Sequence[str],
    res_ids: Sequence[int],
    order: Sequence[int],
    kind: str,
) -> bool:
    """May these two atoms be bonded if they sit in different residues?

    A ligand is routinely split over several HETATM records with different
    residue names and numbers (3PTB: ``BEN A 1`` + ``LIG A 1``), and the ligand
    group has no other residues to leak into, so anything but ``kind="receptor"``
    is permissive.  A receptor may only be cross-linked by a peptide C–N and by
    a disulfide S–S.

    The distance test has already run, so only a genuine bond reaches this
    point: a peptide C–N is ~1.33 A (1.91 A limit) and an S–S bond ~2.05 A
    (2.51 A limit).  The extra conditions are what keep a *side-chain* clash —
    an ``ARG NH1`` against a neighbouring carbonyl, say — from being read as a
    backbone link, and what keep two chains apart.
    """
    first, second = slot_of[i], slot_of[j]
    if first == second:
        return True
    if kind != "receptor":
        return True
    pair = {symbols[i], symbols[j]}
    if pair == {"S"}:
        # A disulfide bridge, the only other covalent cross-link a protein has.
        return True
    if pair != {"C", "N"} or chains[first] != chains[second]:
        return False
    if abs(res_ids[first] - res_ids[second]) != 1 and abs(order[first] - order[second]) != 1:
        return False
    return _is_peptide_c_n(names[i], names[j])


def _valence_limits(symbols: Sequence[str], valence: Optional[Mapping]) -> List[int]:
    """The per-atom cap of automatic mode, with the caller's overrides applied.

    ``valence`` maps an element symbol to its limit; a missing entry, ``None``
    or a non-positive value means "use the built-in table", which is the
    behaviour of PyMOL's all-zero ``valence`` setting.
    """
    limits = [expected_valence(symbol) for symbol in symbols]
    if not valence:
        return limits
    if not isinstance(valence, Mapping):
        raise ValueError("valence must map an element symbol to its bond limit")
    overrides: Dict[str, int] = {}
    for key, value in valence.items():
        if value is None:
            continue
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"valence[{key!r}] = {value!r} is not an integer") from exc
        if limit > 0:
            overrides[_normalise_symbol(key)] = limit
    if not overrides:
        return limits
    return [overrides.get(symbol, limit) for symbol, limit in zip(symbols, limits)]


def _assign_orders(
    accepted: Sequence[Tuple[float, int, int]],
    symbols: Sequence[str],
    limits: Sequence[int],
    degree: Sequence[int],
) -> List[Bond]:
    """Promote short organic bonds to double/triple while valence allows."""
    spare = [max(0, limits[index] - degree[index]) for index in range(len(symbols))]
    out: List[Bond] = []
    for distance, i, j in accepted:
        order = 1
        if symbols[i] in _ORDER_ELEMENTS and symbols[j] in _ORDER_ELEMENTS:
            ratio = distance / (covalent_radius(symbols[i]) + covalent_radius(symbols[j]))
            if ratio <= _TRIPLE_RATIO and spare[i] >= 2 and spare[j] >= 2:
                order = 3
            elif ratio <= _DOUBLE_RATIO and spare[i] >= 1 and spare[j] >= 1:
                order = 2
        extra = order - 1
        spare[i] -= extra
        spare[j] -= extra
        out.append(Bond(a=i, b=j, order=order))
    return out


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def perceive_bonds(
    atoms,
    *,
    kind: str = "ligand",
    connect_mode: int = CONNECT_MODE_AUTO,
    connect_cutoff: float = DEFAULT_CONNECT_CUTOFF,
    valence: Optional[Mapping[str, int]] = None,
    coords=None,
    elements=None,
) -> List[Bond]:
    """Perceive the bonds of a structure, PyMOL-style.

    Parameters
    ----------
    atoms:
        ``odock.gui.structure.Atom``-like objects (``element``, ``name``,
        ``res_name``, ``res_id``, ``chain``, ``x``/``y``/``z``), used for the
        coordinates, the elements and the residue identity.
    kind:
        ``"receptor"`` enforces the cross-residue rule (peptide C–N and
        disulfide S–S only); anything else is treated as a ligand and may span
        residues.
    connect_mode:
        :data:`CONNECT_MODE_AUTO` (default) or :data:`CONNECT_MODE_DISTANCE`;
        any other value is rejected.
    connect_cutoff:
        The distance used by distance mode, in A.
    valence:
        Optional ``{element: limit}`` override of the built-in valence table.
        Missing entries, ``None`` and non-positive values keep the built-in
        limit, i.e. the default is PyMOL's all-zero ``valence`` setting.
    coords, elements:
        Optional overrides for structures that are not ``Atom`` objects.  When
        they are given they must describe the same ``atoms`` in the same order.

    Returns
    -------
    list[Bond]
        Sorted by ``(a, b)``, with ``a < b`` and no duplicate pair.
    """
    atoms = list(atoms)
    points = _points_from(atoms, coords)
    count = len(points)
    # The overrides are validated before the "nothing to bond" shortcut, so a
    # mismatched elements= list is an error rather than a silent empty result.
    symbols = _symbols_from(atoms, elements, count)
    if count < 2:
        return []
    if connect_mode not in (CONNECT_MODE_AUTO, CONNECT_MODE_DISTANCE):
        raise ValueError(
            f"connect_mode must be {CONNECT_MODE_AUTO} (auto) or "
            f"{CONNECT_MODE_DISTANCE} (distance), got {connect_mode!r}"
        )
    limits = _valence_limits(symbols, valence)
    slot_of, chains, res_ids, order = _residues_from(atoms, count)
    names = _atom_names_from(atoms, count)

    # Shortest first: the strongest contacts win the remaining valence.  This is
    # what turns "every distance test passed" into a chemically consistent graph
    # for an atom with too many close neighbours.
    candidates = sorted(
        _candidates(points, symbols, connect_mode, connect_cutoff),
        key=lambda item: (item[0], item[1], item[2]),
    )

    degree = [0] * count
    accepted: List[Tuple[float, int, int]] = []
    for distance, i, j in candidates:
        if not _cross_residue_allowed(
            i, j, symbols, names, slot_of, chains, res_ids, order, kind
        ):
            continue
        if connect_mode == CONNECT_MODE_AUTO and (
            degree[i] >= limits[i] or degree[j] >= limits[j]
        ):
            continue
        degree[i] += 1
        degree[j] += 1
        accepted.append((distance, i, j))

    bonds = _assign_orders(accepted, symbols, limits, degree)
    bonds.sort(key=lambda bond: (bond.a, bond.b))
    return bonds


def bond_pairs(bonds: Iterable) -> List[Tuple[int, int]]:
    """``[(a, b), ...]`` — the shape the renderer and the legacy API use."""
    out: List[Tuple[int, int]] = []
    for bond in bonds or ():
        if hasattr(bond, "a") and hasattr(bond, "b"):
            out.append((int(bond.a), int(bond.b)))
            continue
        try:
            first, second = bond[0], bond[1]
        except (TypeError, IndexError, KeyError):  # pragma: no cover - defensive
            continue
        out.append((int(first), int(second)))
    return out


def guess_bonds(
    atoms,
    factor: float = 1.25,
    max_bonds: int = 4,
    *,
    kind: str = "ligand",
    connect_mode: int = CONNECT_MODE_AUTO,
    connect_cutoff: float = DEFAULT_CONNECT_CUTOFF,
    valence: Optional[Mapping[str, int]] = None,
) -> List[Tuple[int, int]]:
    """Legacy-compatible wrapper: :func:`perceive_bonds` as plain index pairs.

    ``factor`` and ``max_bonds`` belong to the old ``factor * (r_i + r_j)``
    distance test and to its degree cap; both are superseded by the covalent
    radius table and the valence table, so they are accepted and ignored.  Use
    :func:`perceive_bonds` directly when the bond order is wanted.
    """
    del factor, max_bonds  # superseded, kept for signature compatibility
    return bond_pairs(
        perceive_bonds(
            atoms,
            kind=kind,
            connect_mode=connect_mode,
            connect_cutoff=connect_cutoff,
            valence=valence,
        )
    )


def _bond_indices(bond) -> Tuple[int, int]:
    """``(i, j)`` with ``i < j`` from a :class:`Bond` or a plain pair."""
    if hasattr(bond, "a") and hasattr(bond, "b"):
        first, second = int(bond.a), int(bond.b)
    else:
        first, second = int(bond[0]), int(bond[1])
    return (first, second) if first <= second else (second, first)


def _bond_order(bond) -> int:
    try:
        order = int(getattr(bond, "order", None) or 1)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 1
    return order if order > 0 else 1


def _atom_distance(first, second) -> float:
    ex = float(first.x) - float(second.x)
    ey = float(first.y) - float(second.y)
    ez = float(first.z) - float(second.z)
    return math.sqrt(ex * ex + ey * ey + ez * ez)


def bond_report(atoms, bonds) -> dict:
    """Diagnostics for the bond-check dialog.

    Everything the user needs in order to convince themselves that the
    connectivity is right: how many bonds were perceived, the degree histogram,
    the longest bond with the two atoms it joins, and every atom whose degree is
    impossible for its element (``over_valent``) or zero (``isolated``).

    ``over_valent`` is the number that matters: with the valence table enforced
    it is empty for :func:`perceive_bonds` output, and it fills up as soon as
    distance mode or a hand-made bond list is reported — which is exactly what
    the dialog is for.
    """
    atoms = list(atoms)
    count = len(atoms)

    pairs: List[Tuple[int, int]] = []
    order_counts: Dict[int, int] = {}
    seen = set()
    for bond in bonds or ():
        try:
            i, j = _bond_indices(bond)
        except (TypeError, IndexError, KeyError, ValueError):  # pragma: no cover
            continue
        if i == j or not (0 <= i < count and 0 <= j < count):
            continue
        if (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
        order = _bond_order(bond)
        order_counts[order] = order_counts.get(order, 0) + 1

    degree = [0] * count
    lengths: List[Tuple[float, int, int]] = []
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
        lengths.append((_atom_distance(atoms[i], atoms[j]), i, j))

    histogram: Dict[int, int] = {}
    for value in degree:
        histogram[value] = histogram.get(value, 0) + 1

    over_valent: List[dict] = []
    isolated: List[str] = []
    for index in range(count):
        atom = atoms[index]
        symbol = _element_of(atom)
        limit = expected_valence(symbol)
        if not is_bondable(symbol):
            continue
        if degree[index] == 0:
            isolated.append(_atom_label(atom, index))
            continue
        if degree[index] > limit:
            over_valent.append(
                {
                    "index": index,
                    "atom": _atom_label(atom, index),
                    "element": symbol,
                    "degree": degree[index],
                    "expected": limit,
                }
            )

    longest = None
    if lengths:
        value, i, j = max(lengths)
        longest = {
            "a": i,
            "b": j,
            "length": value,
            "elements": (_element_of(atoms[i]), _element_of(atoms[j])),
            "labels": (_atom_label(atoms[i], i), _atom_label(atoms[j], j)),
        }

    values = sorted(length for length, _, _ in lengths)
    return {
        "n_atoms": count,
        "n_bonds": len(pairs),
        "n_single": order_counts.get(1, 0),
        "n_double": order_counts.get(2, 0),
        "n_triple": order_counts.get(3, 0),
        "mean_length": (sum(values) / len(values)) if values else None,
        "median_length": (values[len(values) // 2] if values else None),
        "min_length": (values[0] if values else None),
        "max_degree": max(degree) if degree else 0,
        "degree_histogram": dict(sorted(histogram.items())),
        "longest": longest,
        "over_valent": over_valent,
        "isolated": len(isolated),
        "isolated_atoms": isolated,
    }

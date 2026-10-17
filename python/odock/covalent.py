# SPDX-License-Identifier: GPL-3.0-or-later
"""Covalent docking: warheads, reactive geometry, a covalent score term and adducts.

Non-covalent docking scores shape and contacts; a covalent inhibitor also has to
put its **warhead at bonding distance from a nucleophile, in the right
orientation**.  This module is that layer, and it is deliberately small: it
perceives the chemistry, classifies the geometry, adds one score term, and builds
and relaxes the adduct with the machinery the project already has.

What it does
------------
* :func:`perceive_warheads` — common covalent warheads from substructure
  (acrylamide, vinyl sulfonamide, chloroacetamide, epoxide, nitrile,
  alpha-cyanoacrylamide) plus a documented catch-all set, naming the
  **electrophilic atom** and the reference atom the approach angle is measured
  against;
* :func:`perceive_nucleophiles` — the target nucleophile, usually the catalytic
  cysteine, by residue/atom name and by its own geometry;
* :func:`classify_attack` — distance **and** approach angle between the two, with
  the criteria stated in :class:`CovalentCriteria`, classified as ``bonded``,
  ``near_attack`` or ``non_productive`` (with the reason);
* :func:`covalent_term` — one documented term added to an existing non-covalent
  score: a smooth reward for near-attack geometry, a bond-formation term once the
  bond can form, and a penalty for a strained adduct.  Every component is a
  continuous function of distance and angle, so the score has no cliff at the
  classification boundary (:func:`continuity_scan` measures that);
* :func:`build_adduct` / :func:`relax_adduct` / :func:`adduct_strain` — form the
  bond, re-minimise the adduct locally through
  :func:`odock.chem.ligand.minimize` (the project's one minimiser), and report the
  strain through :func:`odock.metrics.ligand_strain`, including its ``reliable``
  flag;
* :func:`covalent_ranking` — how the term changes a ranking, beside the
  non-covalent score of the same poses.

What this is **not**
--------------------
This is a **geometry** layer, not a chemistry model.  It does not compute an
electrophilicity index, a pKa, a nucleophilicity, a rate or a reaction energy, and
it cannot know whether the warhead in hand actually reacts with the protein in
hand: that is experimental knowledge.  A ``near_attack`` classification is a
**hypothesis for a chemist** — "this pose could form this bond, if the chemistry
is what the literature says it is" — and never a prediction that the compound is
an inhibitor.  The strain number inherits every limitation
:mod:`odock.metrics` documents (force-field typing, and the ``reliable`` flag for
chemistry perceived from a bond-order-free PDBQT).  ``docs/COVALENT.md`` §1 is
this section.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import rdMolTransforms

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    rdMolTransforms = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

__all__ = [
    "WARHEAD_NAMES",
    "NUCLEOPHILE_RESIDUES",
    "ATTACK_CLASSES",
    "DEFAULT_CRITERIA",
    "ADDUCT_GEOMETRY",
    "WarheadPattern",
    "WarheadHit",
    "Nucleophile",
    "CovalentCriteria",
    "AttackGeometry",
    "CovalentTerm",
    "PoseCovalent",
    "Adduct",
    "require_rdkit",
    "perceive_warheads",
    "perceive_nucleophiles",
    "classify_attack",
    "covalent_term",
    "covalent_score",
    "covalent_ranking",
    "continuity_scan",
    "build_adduct",
    "relax_adduct",
    "adduct_strain",
    "adduct_quality",
    "interaction_change",
]

#: The warhead names :func:`perceive_warheads` can report, in report order.
WARHEAD_NAMES: Tuple[str, ...] = (
    "acrylamide",
    "alpha_cyanoacrylamide",
    "vinyl_sulfonamide",
    "vinyl_sulfone",
    "chloroacetamide",
    "epoxide",
    "nitrile",
    "catch_all_alkene",
    "catch_all_halide",
)

#: Residues whose sulfur/selenium is treated as the reactive nucleophile.
NUCLEOPHILE_RESIDUES: Tuple[str, ...] = ("CYS", "SEC", "CYX", "CSO")

#: The three geometry classes :func:`classify_attack` can report.
ATTACK_CLASSES: Tuple[str, ...] = ("bonded", "near_attack", "non_productive")

#: The adduct geometry this module measures and judges (Angstrom, degrees).
ADDUCT_GEOMETRY: Dict[str, Tuple[float, float]] = {
    # A C-S single bond is 1.81-1.82 A in the published crystal structures of
    # cysteine adducts; the accepted "formed" window is deliberately wider.
    "c_s_bond": (1.70, 1.95),
    # The S-C-C angle at the former electrophilic carbon is tetrahedral after a
    # Michael addition (109.5 +/- 10 in the adducts we can check).
    "s_c_c_angle": (95.0, 125.0),
}


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for covalent docking. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Warheads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarheadPattern:
    """One warhead definition: where it is, what is attacked, how it approaches.

    ``smarts`` uses atom map numbers: ``attack`` names the electrophilic atom (the
    atom the sulfur attacks), ``reference`` the atom the approach angle is measured
    against (the alpha carbon for a Michael acceptor, the leaving group for an
    SN2 halide), and ``leaving_group`` the atom that departs, when there is one.

    ``ideal_angle`` is the published approach geometry for the mechanism: about
    107 degrees for a Michael acceptor (the Burgi-Dunitz trajectory onto the
    pi* orbital) and about 180 degrees for an SN2 backside attack.
    """

    name: str
    smarts: str
    attack: int
    reference: int
    mechanism: str
    ideal_angle: float
    angle_window: Tuple[float, float]
    leaving_group: int = 0
    description: str = ""
    catch_all: bool = False


#: The documented warhead table.  Every pattern names its atoms by map number, so
#: a hit can report *which* atom is attacked without a second SMARTS.
WARHEAD_PATTERNS: Tuple[WarheadPattern, ...] = (
    WarheadPattern(
        name="acrylamide",
        smarts="[CH2:1]=[CH:2][C:3](=[OX1:4])[NX3:5]",
        attack=1,
        reference=2,
        mechanism="michael",
        ideal_angle=107.0,
        angle_window=(70.0, 150.0),
        description="Michael acceptor: the beta carbon is attacked by the cysteine sulfur",
    ),
    WarheadPattern(
        name="alpha_cyanoacrylamide",
        smarts="[CH2:1]=[C:2]([C:6]#[NX1:7])[C:3](=[OX1:4])[NX3:5]",
        attack=1,
        reference=2,
        mechanism="michael",
        ideal_angle=107.0,
        angle_window=(70.0, 150.0),
        description="Michael acceptor with alpha-cyano activation (faster, still Michael)",
    ),
    WarheadPattern(
        name="vinyl_sulfonamide",
        smarts="[CH2:1]=[CH:2][S:3](=[OX1:4])(=[OX1:5])[NX3:6]",
        attack=1,
        reference=2,
        mechanism="michael",
        ideal_angle=107.0,
        angle_window=(70.0, 150.0),
        description="Michael acceptor on a sulfonamide (a strong, reversible acceptor)",
    ),
    WarheadPattern(
        name="vinyl_sulfone",
        smarts="[CH2:1]=[CH:2][S:3](=[OX1:4])(=[OX1:5])[#6:6]",
        attack=1,
        reference=2,
        mechanism="michael",
        ideal_angle=107.0,
        angle_window=(70.0, 150.0),
        description="Michael acceptor on a sulfone",
    ),
    WarheadPattern(
        name="chloroacetamide",
        smarts="[Cl,Br,I:6][CH2:1][C:2](=[OX1:3])[NX3:4]",
        attack=1,
        reference=6,
        mechanism="sn2",
        ideal_angle=180.0,
        angle_window=(140.0, 180.0),
        leaving_group=6,
        description="SN2: the sulfur displaces the halide from the alpha carbon",
    ),
    WarheadPattern(
        name="epoxide",
        smarts="[OX2:3]1[#6:1][#6:2]1",
        attack=1,
        reference=2,
        mechanism="sn2",
        ideal_angle=160.0,
        angle_window=(120.0, 180.0),
        description="Epoxide opening: SN2 at a ring carbon, ring strain supplying the driving force",
    ),
    WarheadPattern(
        name="nitrile",
        smarts="[NX1:2]#[CX2:1]",
        attack=1,
        reference=2,
        mechanism="nitrile",
        ideal_angle=180.0,
        angle_window=(140.0, 180.0),
        description="Nitrile: attack on the sp carbon, in line with the C-N triple bond",
    ),
    WarheadPattern(
        name="catch_all_alkene",
        smarts="[CH2:1]=[CH:2]",
        attack=1,
        reference=2,
        mechanism="michael",
        ideal_angle=107.0,
        angle_window=(70.0, 150.0),
        catch_all=True,
        description="catch-all: any terminal alkene, reported at lower confidence",
    ),
    WarheadPattern(
        name="catch_all_halide",
        smarts="[CX4:1][Cl,Br,I:2]",
        attack=1,
        reference=2,
        mechanism="sn2",
        ideal_angle=180.0,
        angle_window=(140.0, 180.0),
        leaving_group=2,
        catch_all=True,
        description="catch-all: any alkyl halide, reported at lower confidence",
    ),
)

_PATTERN_QUERIES: Dict[str, Any] = {}
_PATTERN_MAPS: Dict[str, Dict[int, int]] = {}


def _pattern_query(pattern: WarheadPattern):
    key = pattern.name
    if key not in _PATTERN_QUERIES:
        query = Chem.MolFromSmarts(pattern.smarts)
        if query is None:  # pragma: no cover - a broken pattern in the table
            raise RuntimeError(f"the warhead pattern {pattern.name!r} does not compile")
        _PATTERN_QUERIES[key] = query
        # The match tuple is indexed by *query atom order*, not by map number, so
        # the map numbers in the SMARTS have to be resolved to query atom indices
        # explicitly.  Writing `[OX1:3]1[#6:1][#6:2]1` and indexing with the map
        # number is how this silently returns the wrong atom.
        _PATTERN_MAPS[key] = {
            int(atom.GetAtomMapNum()): atom.GetIdx() for atom in query.GetAtoms()
        }
    return _PATTERN_QUERIES[key]


def _map_index(pattern: WarheadPattern, map_number: int) -> int:
    _pattern_query(pattern)
    table = _PATTERN_MAPS[pattern.name]
    if map_number not in table:  # pragma: no cover - a table typo
        raise RuntimeError(
            f"the warhead pattern {pattern.name!r} has no atom mapped to {map_number}"
        )
    return table[map_number]


@dataclass
class WarheadHit:
    """One warhead found in a molecule."""

    name: str
    mechanism: str
    #: Index of the electrophilic atom (the one the nucleophile attacks).
    electrophile: int
    #: Index of the atom the approach angle is measured against.
    reference: int
    #: Leaving group index, or ``-1``.
    leaving_group: int = -1
    #: Every atom of the matched pattern.
    atoms: Tuple[int, ...] = ()
    #: Whether this came from a catch-all pattern rather than a named warhead.
    catch_all: bool = False
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mechanism": self.mechanism,
            "electrophile": int(self.electrophile),
            "reference": int(self.reference),
            "leaving_group": int(self.leaving_group),
            "atoms": [int(i) for i in self.atoms],
            "catch_all": bool(self.catch_all),
            "description": self.description,
        }

    def label(self) -> str:
        return f"{self.name}({self.mechanism}, atom {self.electrophile})"


def perceive_warheads(mol, *, include_catch_all: bool = True) -> List[WarheadHit]:
    """Every warhead in ``mol``, with the electrophilic atom named.

    Named warheads are reported first and catch-all matches second (and a catch-all
    that reproduces an atom already claimed by a named warhead is suppressed, so a
    chloroacetamide is not reported twice).  The result is ordered by pattern
    table order and then by matched atom, so it is deterministic.
    """
    require_rdkit()
    hits: List[WarheadHit] = []
    claimed: set = set()
    for pattern in WARHEAD_PATTERNS:
        if pattern.catch_all and not include_catch_all:
            continue
        query = _pattern_query(pattern)
        for match in mol.GetSubstructMatches(query):
            attack = int(match[_map_index(pattern, pattern.attack)])
            reference = int(match[_map_index(pattern, pattern.reference)])
            if pattern.catch_all and attack in claimed:
                continue
            if attack == reference:  # pragma: no cover - defensive
                continue
            leaving = (
                int(match[_map_index(pattern, pattern.leaving_group)])
                if pattern.leaving_group
                else -1
            )
            hits.append(
                WarheadHit(
                    name=pattern.name,
                    mechanism=pattern.mechanism,
                    electrophile=attack,
                    reference=reference,
                    leaving_group=leaving,
                    atoms=tuple(int(i) for i in match),
                    catch_all=pattern.catch_all,
                    description=pattern.description,
                )
            )
            if not pattern.catch_all:
                claimed.add(attack)
    hits.sort(key=lambda hit: (WARHEAD_NAMES.index(hit.name) if hit.name in WARHEAD_NAMES else 99,
                               hit.electrophile))
    return hits


def _pattern_for(name: str) -> WarheadPattern:
    for pattern in WARHEAD_PATTERNS:
        if pattern.name == name:
            return pattern
    raise ValueError(f"unknown warhead {name!r}; known: {list(WARHEAD_NAMES)}")


# ---------------------------------------------------------------------------
# Nucleophiles
# ---------------------------------------------------------------------------


@dataclass
class Nucleophile:
    """The acceptor atom: usually the sulfur of a cysteine."""

    index: int
    element: str
    position: Tuple[float, float, float]
    residue_name: str = ""
    residue_number: int = 0
    chain: str = ""
    atom_name: str = ""
    #: A second atom of the same residue (CB), used to build a truncated cap.
    cap_position: Optional[Tuple[float, float, float]] = None
    note: str = ""

    def label(self) -> str:
        if self.residue_name:
            return f"{self.residue_name}{self.residue_number}:{self.atom_name or self.element}"
        return f"{self.element}{self.index}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "element": self.element,
            "position": [round(float(v), 3) for v in self.position],
            "residue_name": self.residue_name,
            "residue_number": int(self.residue_number),
            "chain": self.chain,
            "atom_name": self.atom_name,
            "cap_position": None if self.cap_position is None else [round(float(v), 3) for v in self.cap_position],
            "note": self.note,
        }


def _conformer_positions(mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


#: AutoDock atom type to element symbol, for reading a receptor PDBQT.  RDKit's
#: PDB reader derives the element from columns 77-78, where a PDBQT carries the
#: AutoDock type instead, and refuses the atom ("Cannot determine element"); this
#: map is what makes the receptor readable without guessing from atom names.
_AD_TYPE_TO_ELEMENT: Dict[str, str] = {
    "A": "C", "C": "C", "N": "N", "NA": "N", "NS": "N", "OA": "O", "OS": "O",
    "S": "S", "SA": "S", "H": "H", "HD": "H", "HS": "H", "P": "P", "F": "F",
    "CL": "Cl", "BR": "Br", "I": "I", "B": "B", "SI": "Si", "SE": "Se",
    "ZN": "Zn", "MG": "Mg", "CA": "Ca", "FE": "Fe", "MN": "Mn", "K": "K", "CU": "Cu",
}


def _pdbqt_to_readable_pdb(text: str) -> str:
    """Rewrite a PDBQT's AutoDock type column as an element symbol."""
    out: List[str] = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            fields = line.split()
            ad_type = fields[-1].upper() if fields else ""
            element = _AD_TYPE_TO_ELEMENT.get(ad_type, "C")
            line = line[:76].ljust(76) + f"{element:>2}"
        out.append(line)
    return "\n".join(out) + "\n"


def _load_receptor(receptor):
    """Accept a path, a PDB/PDBQT text, or an RDKit molecule."""
    require_rdkit()
    if receptor is None:
        return None
    if isinstance(receptor, Chem.Mol):
        return receptor
    from pathlib import Path

    if isinstance(receptor, (str, Path)):
        path = Path(receptor)
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
        elif isinstance(receptor, Path) or "\n" not in str(receptor):
            return None
        else:
            text = str(receptor)
    elif isinstance(receptor, str):
        text = receptor
    else:  # pragma: no cover - defensive
        from .prepare import read_structure

        return read_structure(receptor)
    if not text.strip():
        return None
    # A receptor PDBQT is a PDB document whose AutoDock atom type sits in the
    # element columns, so it is rewritten into a block RDKit can read: that keeps
    # the residue and atom names the nucleophile search needs.  A *ligand* PDBQT
    # (ROOT/BRANCH tree) is converted by the project's own converter first.
    candidates: List[str] = []
    if "\nROOT" in "\n" + text or "\nBRANCH" in "\n" + text:
        from .prepare import pdbqt_to_pdb_block

        candidates.append(pdbqt_to_pdb_block(text))
    candidates.append(_pdbqt_to_readable_pdb(text))
    for candidate in candidates:
        if not candidate.strip():
            continue
        mol = Chem.MolFromPDBBlock(
            candidate, removeHs=False, sanitize=False, proximityBonding=True
        )
        if mol is not None and mol.GetNumAtoms() > 0:
            return mol
    return None


def perceive_nucleophiles(
    receptor,
    *,
    residues: Sequence[str] = NUCLEOPHILE_RESIDUES,
    atom_names: Sequence[str] = ("SG", "SE"),
    element: str = "S",
) -> List[Nucleophile]:
    """Every candidate nucleophile in ``receptor``, with its geometry.

    The primary path is by **residue and atom name**: a ``CYS`` residue's ``SG``
    (or a selenocysteine's ``SE``), which is how a covalent inhibitor's target is
    identified in practice.  A receptor whose atoms carry no PDB residue
    information falls back to a geometric rule — every non-ring sulfur with at
    least one heavy neighbour is a candidate — and the returned
    :attr:`Nucleophile.note` says so, because a solvent-exposed thiol and a
    catalytic one are not the same thing and this module cannot tell them apart.

    The ``cap_position`` of each candidate is the residue's ``CB``, used by
    :func:`build_adduct` to cap the truncated cysteine with a methyl-like group.
    """
    require_rdkit()
    mol = _load_receptor(receptor)
    if mol is None:
        return []
    if mol.GetNumConformers() == 0:
        return []
    coords = _conformer_positions(mol)
    wanted = {str(name).strip().upper() for name in residues}
    names = {str(name).strip().upper() for name in atom_names}
    found: List[Nucleophile] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        info = atom.GetPDBResidueInfo()
        residue_name = info.GetResidueName().strip().upper() if info else ""
        atom_name = info.GetName().strip().upper() if info else ""
        if residue_name in wanted and atom_name in names:
            position = coords[atom.GetIdx()]
            cap = None
            for neighbour in atom.GetNeighbors():
                neighbour_info = neighbour.GetPDBResidueInfo()
                if neighbour_info and neighbour_info.GetName().strip().upper() == "CB":
                    cap = tuple(float(v) for v in coords[neighbour.GetIdx()])
                    break
            found.append(
                Nucleophile(
                    index=int(atom.GetIdx()),
                    element=symbol,
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    residue_name=residue_name,
                    residue_number=int(info.GetResidueNumber()) if info else 0,
                    chain=info.GetChainId().strip() if info else "",
                    atom_name=atom_name,
                    cap_position=cap,
                    note="identified by residue and atom name",
                )
            )
        elif (
            not residue_name
            and symbol.upper() == str(element).upper()
            and not atom.IsInRing()
            and any(neighbour.GetAtomicNum() > 1 for neighbour in atom.GetNeighbors())
        ):
            position = coords[atom.GetIdx()]
            found.append(
                Nucleophile(
                    index=int(atom.GetIdx()),
                    element=symbol,
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    note=(
                        "no residue information: this is a geometric sulfur candidate, and "
                        "whether it is the catalytic nucleophile is not decided here"
                    ),
                )
            )
    found.sort(key=lambda item: item.index)
    # A deposited file can carry the same chain twice (a rigid and a rewritten
    # copy, which is how the bundled 1M17 receptor is written).  Two records of the
    # same residue at the same position are one candidate, so the second is dropped
    # — but a genuinely different copy of the protein is *not* silently merged: the
    # position is part of the key, so a real dimer keeps both chains.
    unique: List[Nucleophile] = []
    seen: set = set()
    for item in found:
        key = (item.chain, item.residue_number, item.atom_name) if item.residue_name else (
            item.element,
            tuple(round(value, 2) for value in item.position),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


# ---------------------------------------------------------------------------
# Reactive geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CovalentCriteria:
    """The geometry criteria, stated as numbers so they can be argued with.

    ``bonded_max`` / ``near_attack_max`` are sulfur-to-electrophile distances.
    ``angle_window`` is per warhead (a Michael acceptor is approached at about 107
    degrees, an SN2 halide from behind at about 180), so the criteria object holds
    the distance thresholds and the smooth-ramp widths, and the warhead holds the
    angle.
    """

    bonded_max: float = 2.0
    near_attack_max: float = 4.0
    #: Width of the smooth transition at each distance threshold (Angstrom).
    distance_ramp: float = 0.6
    #: Width of the smooth transition at each end of the angle window (degrees).
    angle_ramp: float = 20.0
    #: Reward at a perfect near-attack geometry (kcal/mol, negative = favourable).
    near_attack_reward: float = 3.0
    #: Additional reward once the bond is formed (kcal/mol).
    bond_reward: float = 5.0
    #: Penalty coefficient for a strained adduct, per degree squared.
    adduct_strain_k: float = 0.03
    #: The tetrahedral S-C-C angle a formed Michael adduct should have (degrees).
    adduct_ideal_angle: float = 109.5
    #: How far that angle may deviate before the strain penalty starts (degrees).
    adduct_angle_tolerance: float = 10.0
    #: Width of the transition into the bonded regime (Angstrom).  The bond reward
    #: (5 kcal/mol) is spread over this width, which is what keeps the term's
    #: gradient bounded instead of stepping at the bonding distance.
    bond_ramp: float = 0.5

    def distance_window(self) -> Tuple[float, float]:
        return (float(self.bonded_max), float(self.near_attack_max))


#: The criteria :func:`classify_attack` and :func:`covalent_term` use by default.
DEFAULT_CRITERIA = CovalentCriteria()


def _unit(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """The angle at ``b`` between ``ba`` and ``bc``, in degrees."""
    left = _unit(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    right = _unit(np.asarray(c, dtype=float) - np.asarray(b, dtype=float))
    if left is None or right is None:
        return None
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _smooth_window(value: float, low: float, high: float, soft: float) -> float:
    """``1`` inside ``[low, high]``, a raised-cosine ramp to ``0`` outside.

    This is the function that makes the score continuous: at ``low`` and ``high``
    it is exactly ``1`` and it decays with a continuous derivative, so nothing
    steps when a pose crosses a threshold.
    """
    if soft <= 0:
        return 1.0 if low <= value <= high else 0.0
    if value < low:
        distance = low - value
        if distance >= soft:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * distance / soft))
    if value > high:
        distance = value - high
        if distance >= soft:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * distance / soft))
    return 1.0


def _ramp(value: float, start: float, end: float) -> float:
    """``0`` at ``start``, ``1`` at ``end``, a raised cosine in between."""
    if end == start:
        return 1.0 if value >= end else 0.0
    fraction = (value - start) / (end - start)
    fraction = max(0.0, min(1.0, fraction))
    return 0.5 * (1.0 - math.cos(math.pi * fraction))


@dataclass
class AttackGeometry:
    """The measured geometry of one warhead/nucleophile pair, and its class."""

    warhead: str
    nucleophile: str
    #: Sulfur to electrophilic carbon, Angstrom.
    distance: float
    #: Approach angle at the electrophile (S-C-reference), degrees.
    angle: Optional[float]
    classification: str = "non_productive"
    #: Which criterion decided the class, for a report.
    reason: str = ""
    #: The smooth quality in ``[0, 1]`` the score term uses.
    quality: float = 0.0
    #: ``1`` once the bond is judged formed, ramping with distance.
    bond_fraction: float = 0.0

    @property
    def bonded(self) -> bool:
        return self.classification == "bonded"

    @property
    def near_attack(self) -> bool:
        return self.classification == "near_attack"

    @property
    def productive(self) -> bool:
        return self.classification in ("bonded", "near_attack")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "warhead": self.warhead,
            "nucleophile": self.nucleophile,
            "distance": round(float(self.distance), 4),
            "angle": None if self.angle is None else round(float(self.angle), 3),
            "classification": self.classification,
            "reason": self.reason,
            "quality": round(float(self.quality), 4),
            "bond_fraction": round(float(self.bond_fraction), 4),
        }

    def table(self) -> str:
        angle = "n/a" if self.angle is None else f"{self.angle:.1f} deg"
        return (
            f"{self.warhead} -> {self.nucleophile}: d {self.distance:.2f} A, "
            f"angle {angle}, {self.classification} ({self.reason})"
        )


def _geometry_from_points(
    warhead: WarheadHit,
    attack_point: np.ndarray,
    reference_point: np.ndarray,
    nucleophile_point: np.ndarray,
    *,
    criteria: CovalentCriteria,
    nucleophile_label: str = "nucleophile",
) -> AttackGeometry:
    """Measure and classify one triple of points (the core of the classifier).

    Factored out so :func:`classify_attack` and :func:`continuity_scan` evaluate
    *the same* function: a scan that measured a different formula would prove
    nothing about the score.
    """
    attack_point = np.asarray(attack_point, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    nucleophile_point = np.asarray(nucleophile_point, dtype=float)
    distance = float(np.linalg.norm(nucleophile_point - attack_point))
    angle = _angle_deg(reference_point, attack_point, nucleophile_point)
    pattern = _pattern_for(warhead.name)
    low_angle, high_angle = pattern.angle_window
    quality = _smooth_window(distance, 0.0, criteria.near_attack_max, criteria.distance_ramp)
    if angle is not None:
        quality *= _smooth_window(angle, low_angle, high_angle, criteria.angle_ramp)
    else:
        quality = 0.0
    bond_fraction = _ramp(distance, criteria.bonded_max, criteria.bonded_max - criteria.bond_ramp)
    if angle is None:
        classification = "non_productive"
        reason = "the approach angle is undefined (degenerate geometry)"
    elif distance <= criteria.bonded_max + 1e-9 and low_angle <= angle <= high_angle:
        classification = "bonded"
        reason = (
            f"d <= {criteria.bonded_max:.1f} A and angle within "
            f"{low_angle:.0f}-{high_angle:.0f} deg"
        )
    elif distance <= criteria.near_attack_max + 1e-9 and low_angle <= angle <= high_angle:
        classification = "near_attack"
        reason = (
            f"d <= {criteria.near_attack_max:.1f} A and angle within "
            f"{low_angle:.0f}-{high_angle:.0f} deg, but not at bonding distance"
        )
    elif distance > criteria.near_attack_max:
        classification = "non_productive"
        reason = f"d {distance:.2f} A > {criteria.near_attack_max:.1f} A: out of reach"
    else:
        classification = "non_productive"
        reason = (
            f"angle {angle:.1f} deg outside the {low_angle:.0f}-{high_angle:.0f} deg "
            f"window for a {warhead.mechanism} warhead"
        )
    return AttackGeometry(
        warhead=warhead.name,
        nucleophile=nucleophile_label,
        distance=distance,
        angle=angle,
        classification=classification,
        reason=reason,
        quality=float(quality),
        bond_fraction=float(bond_fraction),
    )


def classify_attack(
    ligand_mol,
    warhead: WarheadHit,
    nucleophile: Nucleophile,
    *,
    criteria: CovalentCriteria = DEFAULT_CRITERIA,
    conf_id: int = 0,
) -> AttackGeometry:
    """Measure and classify one warhead/nucleophile pair.

    The criteria are geometric and are stated in :class:`CovalentCriteria` and on
    the warhead:

    * **bonded** — distance <= ``bonded_max`` (2.0 A: a C-S single bond is about
      1.82 A, and 2.0 A is the top of the accepted window) *and* the approach angle
      inside the warhead's window;
    * **near_attack** — distance <= ``near_attack_max`` (4.0 A) and the angle
      inside the window, and not bonded;
    * **non_productive** — anything else, with the failing criterion named in
      :attr:`AttackGeometry.reason`.

    The angle is measured at the electrophilic atom between the warhead's reference
    atom (alpha carbon, leaving group, ring neighbour or nitrile nitrogen) and the
    nucleophile: the trajectory a Michael acceptor needs is about 107 degrees and an
    SN2 backside attack about 180.  A missing angle (a degenerate arrangement)
    cannot be classified and is reported as ``non_productive`` with that reason.
    """
    require_rdkit()
    if ligand_mol.GetNumConformers() == 0:
        raise ValueError(
            "the ligand needs 3-D coordinates to measure a reactive geometry "
            "(pass a docked pose or an embedded molecule)"
        )
    if conf_id < 0 or conf_id >= ligand_mol.GetNumConformers():
        raise IndexError(f"conformer {conf_id} does not exist")
    conf = ligand_mol.GetConformer(int(conf_id))

    def _point(index: int) -> np.ndarray:
        position = conf.GetAtomPosition(int(index))
        return np.array([position.x, position.y, position.z], dtype=float)

    return _geometry_from_points(
        warhead,
        _point(warhead.electrophile),
        _point(warhead.reference),
        np.asarray(nucleophile.position, dtype=float),
        criteria=criteria,
        nucleophile_label=nucleophile.label(),
    )


# ---------------------------------------------------------------------------
# The covalent score term
# ---------------------------------------------------------------------------


@dataclass
class CovalentTerm:
    """The covalent contribution to a score, split into its documented parts."""

    #: The total, kcal/mol: negative is favourable.
    value: float = 0.0
    #: The near-attack reward (>= 0, subtracted from the score).
    near_attack: float = 0.0
    #: The bond-formation reward (>= 0, subtracted once the bond can form).
    bond: float = 0.0
    #: The strained-adduct penalty (>= 0, added back).
    strain_penalty: float = 0.0
    #: The geometry the term was evaluated on.
    geometry: Optional[AttackGeometry] = None

    @property
    def favourable(self) -> bool:
        return self.value < 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "value": round(float(self.value), 4),
            "near_attack": round(float(self.near_attack), 4),
            "bond": round(float(self.bond), 4),
            "strain_penalty": round(float(self.strain_penalty), 4),
            "geometry": self.geometry.as_dict() if self.geometry is not None else None,
        }

    def table(self) -> str:
        lines = [
            f"covalent term: {self.value:+.3f} kcal/mol",
            f"  near-attack reward {self.near_attack:.3f}, bond formation {self.bond:.3f}, "
            f"adduct strain penalty {self.strain_penalty:.3f}",
        ]
        if self.geometry is not None:
            lines.append(f"  {self.geometry.table()}")
        return "\n".join(lines)


def covalent_term(
    geometry: AttackGeometry,
    *,
    criteria: CovalentCriteria = DEFAULT_CRITERIA,
) -> CovalentTerm:
    """The documented covalent term for one geometry (kcal/mol, negative = good).

    ``term = -(near_attack_reward * q + bond_reward * b) + adduct_strain_k * excess^2 * b``

    where ``q`` is the smooth geometry quality in ``[0, 1]`` (a raised-cosine
    window over distance and angle, so it never steps at a threshold), ``b`` is the
    smooth bond fraction in ``[0, 1]`` (0 above the bonded distance, 1 once the bond
    is clearly formed) and ``excess`` is how far the measured S-C-C angle is outside
    the tetrahedral tolerance.  A pose that cannot reach therefore gets **zero**
    rather than a near-miss reward: ``q`` is zero beyond the criteria, so being
    "close" only helps while the geometry is actually productive.

    The strain penalty is deliberately *not* the ligand strain (that is measured
    separately, by :func:`odock.metrics.ligand_strain`); it is the geometric
    distortion of the adduct's S-C-C angle, which the pose alone determines.
    """
    quality = max(0.0, min(1.0, float(geometry.quality)))
    bond = max(0.0, min(1.0, float(geometry.bond_fraction)))
    near_attack = float(criteria.near_attack_reward) * quality
    bond_energy = float(criteria.bond_reward) * bond * quality
    excess = 0.0
    if geometry.angle is not None:
        deviation = abs(float(geometry.angle) - float(criteria.adduct_ideal_angle))
        excess = max(0.0, deviation - float(criteria.adduct_angle_tolerance))
    strain_penalty = float(criteria.adduct_strain_k) * excess * excess * bond
    value = -(near_attack + bond_energy) + strain_penalty
    return CovalentTerm(
        value=float(value),
        near_attack=float(near_attack),
        bond=float(bond_energy),
        strain_penalty=float(strain_penalty),
        geometry=geometry,
    )


def covalent_score(
    non_covalent: float,
    geometry: AttackGeometry,
    *,
    criteria: CovalentCriteria = DEFAULT_CRITERIA,
) -> float:
    """The non-covalent affinity plus the covalent term (kcal/mol).

    This is how the term is meant to be used: the force field answers "how well does
    this pose fit", the covalent term answers "and can it form the bond from here".
    Neither replaces the other, and the sum is still not a binding free energy.
    """
    return float(non_covalent) + float(covalent_term(geometry, criteria=criteria).value)


@dataclass
class PoseCovalent:
    """One pose's non-covalent score, its covalent term and the resulting rank."""

    name: str
    non_covalent: float
    term: float
    total: float
    classification: str
    distance: float
    angle: Optional[float]
    rank_before: int = 0
    rank_after: int = 0
    reason: str = ""

    @property
    def rank_change(self) -> int:
        """Positive means the covalent term promoted the pose."""
        return int(self.rank_before) - int(self.rank_after)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "non_covalent": round(float(self.non_covalent), 4),
            "term": round(float(self.term), 4),
            "total": round(float(self.total), 4),
            "classification": self.classification,
            "distance": round(float(self.distance), 3),
            "angle": None if self.angle is None else round(float(self.angle), 2),
            "rank_before": int(self.rank_before),
            "rank_after": int(self.rank_after),
            "rank_change": int(self.rank_change),
            "reason": self.reason,
        }


def covalent_ranking(
    poses: Sequence[Tuple[str, float, AttackGeometry]],
    *,
    criteria: CovalentCriteria = DEFAULT_CRITERIA,
) -> List[PoseCovalent]:
    """Rank the same poses by the non-covalent score and by the corrected one.

    ``poses`` is ``[(name, non_covalent_affinity, geometry), ...]``.  The return is
    ordered by the corrected score (best first) and every row carries both ranks, so
    the *effect of the term* is the number a reader can check: a pose that only wins
    because it is close but badly oriented loses rank, and one that can form the
    bond gains it.
    """
    terms = [(name, float(affinity) + covalent_term(geometry, criteria=criteria).value,
              float(affinity), geometry) for name, affinity, geometry in poses]
    before = sorted(terms, key=lambda item: (item[2], item[0]))
    rank_before = {name: position + 1 for position, (name, _, _, _) in enumerate(before)}
    after = sorted(terms, key=lambda item: (item[1], item[0]))
    rows: List[PoseCovalent] = []
    for position, (name, total, affinity, geometry) in enumerate(after, start=1):
        rows.append(
            PoseCovalent(
                name=name,
                non_covalent=affinity,
                term=total - affinity,
                total=total,
                classification=geometry.classification,
                distance=geometry.distance,
                angle=geometry.angle,
                rank_before=rank_before[name],
                rank_after=position,
                reason=geometry.reason,
            )
        )
    return rows


def continuity_scan(
    warhead: WarheadHit,
    attack_point: np.ndarray,
    reference_point: np.ndarray,
    *,
    criteria: CovalentCriteria = DEFAULT_CRITERIA,
    samples: int = 41,
    angle_samples: int = 41,
    angle_distance: float = 2.5,
) -> Dict[str, Any]:
    """Measure whether the term is continuous across its own thresholds.

    Two scans, both evaluated through the *same* geometry function the classifier
    uses: the **distance** is swept from ``bonded_max - 1`` to
    ``near_attack_max + 1`` A at the warhead's ideal angle, and the **angle** is
    swept past both ends of the warhead's window at a near-attack distance.  The
    largest step between neighbouring samples is reported, because a step change at
    a threshold is how these scores become unfalsifiable: the raised-cosine windows
    bound it to about the reward divided by the ramp width, and the numbers make
    that checkable rather than asserted.
    """
    require_rdkit()
    pattern = _pattern_for(warhead.name)
    low_angle, high_angle = pattern.angle_window
    attack = np.asarray(attack_point, dtype=float)
    unit = _unit(np.asarray(reference_point, dtype=float) - attack)
    if unit is None:  # pragma: no cover - defensive
        raise ValueError("the reference atom coincides with the electrophilic atom")
    plane = _unit(np.cross(unit, np.array([0.0, 0.0, 1.0])))
    if plane is None:
        plane = _unit(np.cross(unit, np.array([0.0, 1.0, 0.0])))
    if plane is None:  # pragma: no cover - defensive
        raise ValueError("no plane perpendicular to the reference direction")

    def _direction(theta_degrees: float) -> np.ndarray:
        theta = math.radians(float(theta_degrees))
        return math.cos(theta) * unit + math.sin(theta) * plane

    distances = np.linspace(
        max(0.5, float(criteria.bonded_max) - 1.0),
        float(criteria.near_attack_max) + 1.0,
        max(3, int(samples)),
    )
    distance_values: List[float] = []
    distance_classes: List[str] = []
    ideal = float(pattern.ideal_angle)
    for distance in distances:
        nucleophile = attack + float(distance) * _direction(ideal)
        geometry = _geometry_from_points(
            warhead, attack, np.asarray(reference_point, dtype=float), nucleophile,
            criteria=criteria, nucleophile_label="scan",
        )
        distance_values.append(float(covalent_term(geometry, criteria=criteria).value))
        distance_classes.append(geometry.classification)

    angles = np.linspace(low_angle - 30.0, high_angle + 30.0, max(3, int(angle_samples)))
    angle_values: List[float] = []
    for theta in angles:
        nucleophile = attack + float(angle_distance) * _direction(float(theta))
        geometry = _geometry_from_points(
            warhead, attack, np.asarray(reference_point, dtype=float), nucleophile,
            criteria=criteria, nucleophile_label="scan",
        )
        angle_values.append(float(covalent_term(geometry, criteria=criteria).value))

    def _largest_step(values: Sequence[float], coordinates: Sequence[float]) -> Tuple[float, float]:
        if len(values) < 2:
            return 0.0, 0.0
        steps = [abs(values[index + 1] - values[index]) for index in range(len(values) - 1)]
        spacing = abs(float(coordinates[1]) - float(coordinates[0])) or 1.0
        largest = max(steps)
        return float(largest), float(largest / spacing)

    distance_step, distance_step_rate = _largest_step(distance_values, list(distances))
    angle_step, angle_step_rate = _largest_step(angle_values, list(angles))
    return {
        "n_samples": len(distance_values),
        "distance_range": [float(distances[0]), float(distances[-1])],
        "distance_spacing": float(distances[1] - distances[0]),
        "max_distance_step": distance_step,
        "max_distance_step_per_angstrom": distance_step_rate,
        "distance_classes": distance_classes,
        "distance_values": [round(float(value), 4) for value in distance_values],
        "angle_range": [float(angles[0]), float(angles[-1])],
        "angle_spacing": float(angles[1] - angles[0]),
        "max_angle_step": angle_step,
        "max_angle_step_per_degree": angle_step_rate,
        "angle_values": [round(float(value), 4) for value in angle_values],
        "note": (
            "the term is built from raised-cosine windows, so the largest step is "
            "bounded by the reward divided by the ramp width; a step of the size of "
            "the reward would mean a threshold cliff"
        ),
    }


# ---------------------------------------------------------------------------
# Building and relaxing the adduct
# ---------------------------------------------------------------------------


@dataclass
class Adduct:
    """A covalently bound fragment: the ligand plus the cysteine's S and CB."""

    mol: Any = None
    #: Atom indices in :attr:`mol`: the sulfur, the former electrophile, and the
    #: atom the angle is measured against (the alpha carbon, or the ring neighbour).
    sulfur: int = -1
    electrophile: int = -1
    reference: int = -1
    warhead: str = ""
    mechanism: str = ""
    nucleophile: str = ""
    #: ``(before, after)`` coordinates of the whole fragment, if it was relaxed.
    relaxed: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.mol is not None and self.sulfur >= 0 and self.electrophile >= 0

    def bond_length(self) -> Optional[float]:
        return self._measure("bond")

    def bond_angle(self) -> Optional[float]:
        return self._measure("angle")

    def _measure(self, kind: str) -> Optional[float]:
        if not self.ok or self.mol.GetNumConformers() == 0:
            return None
        conf = self.mol.GetConformer()
        if kind == "bond":
            return float(rdMolTransforms.GetBondLength(conf, int(self.sulfur), int(self.electrophile)))
        try:
            return float(
                rdMolTransforms.GetAngleDeg(
                    conf, int(self.reference), int(self.electrophile), int(self.sulfur)
                )
            )
        except Exception:  # pragma: no cover - a degenerate arrangement
            return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "warhead": self.warhead,
            "mechanism": self.mechanism,
            "nucleophile": self.nucleophile,
            "relaxed": bool(self.relaxed),
            "bond_length": None if self.bond_length() is None else round(self.bond_length(), 4),
            "bond_angle": None if self.bond_angle() is None else round(self.bond_angle(), 4),
            "notes": list(self.notes),
        }


def build_adduct(
    ligand_mol,
    warhead: WarheadHit,
    nucleophile: Nucleophile,
    *,
    proton_transfer: bool = True,
    cap_with_cb: bool = True,
    form_bond: bool = True,
    bond_length: float = 1.82,
) -> Adduct:
    """Form the covalent bond and return the adduct fragment.

    The chemistry of the *topology* is the warhead's: for a Michael acceptor the
    sulfur adds to the beta carbon, the alkene becomes a single bond and the alpha
    carbon gains the proton the cysteine loses, so the adduct is the ligand's atoms
    plus an S, a proton and (when ``proton_transfer``) nothing else.  The sulfur is
    capped with the cysteine's own ``CB`` when the nucleophile carries one, or with
    hydrogens otherwise — the rest of the protein is frozen and is not part of this
    fragment, which is the standard truncation and is recorded in the notes.

    ``form_bond`` places the sulfur at ``bond_length`` (1.82 A, a C-S single bond)
    **along the approach vector the pose already has**, leaving every other atom at
    the pose's coordinates.  That is the step "form the bond": the pose decides the
    direction, and the local relaxation afterwards (``relax_adduct``) measures how
    far that pose geometry had to move.  With ``form_bond=False`` the donor atom
    stays exactly where the pose put it, which is what a caller wants in order to
    inspect the *unformed* geometry — but note that a force field will not pull a
    bond that is a whole Angstrom too long, so the subsequent minimisation would
    report a converged geometry it has barely touched.
    """
    require_rdkit()
    if ligand_mol.GetNumConformers() == 0:
        raise ValueError("the ligand needs 3-D coordinates to build an adduct")
    if not (0 <= warhead.electrophile < ligand_mol.GetNumAtoms()):
        raise ValueError("the warhead's electrophilic atom is not in the ligand")
    pattern = _pattern_for(warhead.name)
    if pattern.mechanism == "sn2" and warhead.leaving_group < 0 and warhead.name != "epoxide":
        raise ValueError(
            f"the {warhead.name} warhead has no leaving group, so an SN2 adduct "
            "cannot be built without inventing one"
        )
    work = Chem.RWMol(ligand_mol)
    notes: List[str] = []
    conf = work.GetConformer()
    if pattern.mechanism == "sn2" and warhead.leaving_group >= 0:
        # The leaving group departs: remove it (its neighbours keep their valence).
        remove = int(warhead.leaving_group)
        work.RemoveAtom(remove)
        notes.append(
            f"the leaving group (atom {remove} of the input) was removed: this is the "
            "SN2 adduct, not a two-step intermediate"
        )
        # Atom indices shift after a removal; recompute the attack atom and the
        # reference atom in the new numbering.
        def _shift(index: int) -> int:
            return index - 1 if index > remove else index

        electrophile = _shift(int(warhead.electrophile))
        reference = _shift(int(warhead.reference))
    else:
        electrophile = int(warhead.electrophile)
        reference = int(warhead.reference)
    sulfur_position = np.asarray(nucleophile.position, dtype=float)
    sulfur = work.AddAtom(Chem.Atom(16))
    work.AddBond(int(electrophile), int(sulfur), Chem.BondType.SINGLE)
    work.GetConformer().SetAtomPosition(
        int(sulfur),
        (float(sulfur_position[0]), float(sulfur_position[1]), float(sulfur_position[2])),
    )
    # The *topology* of the addition, not just the bond: a Michael acceptor's alkene
    # becomes a single bond (leaving a valence error otherwise — the electrophilic
    # carbon would carry two double bonds), a nitrile's triple bond becomes a
    # double, and an epoxide opens its C-O bond so the sulfur's carbon and the
    # neighbouring carbon are no longer in a ring.
    if pattern.mechanism == "michael":
        work.RemoveBond(int(electrophile), int(reference))
        work.AddBond(int(electrophile), int(reference), Chem.BondType.SINGLE)
        notes.append("the alkene was saturated: the Michael addition makes the C=C a C-C bond")
    elif pattern.mechanism == "nitrile":
        work.RemoveBond(int(electrophile), int(reference))
        work.AddBond(int(electrophile), int(reference), Chem.BondType.DOUBLE)
        notes.append("the nitrile became an imine (C#N -> C=N)")
    elif warhead.name == "epoxide":
        oxygen_atom = None
        for atom_index in warhead.atoms:
            if work.GetAtomWithIdx(int(atom_index)).GetSymbol() == "O":
                oxygen_atom = int(atom_index)
                break
        if oxygen_atom is not None:
            try:
                work.RemoveBond(int(electrophile), oxygen_atom)
                notes.append(
                    "the epoxide ring opened at the attacked carbon; the O stays on the "
                    "other carbon"
                )
            except Exception:  # pragma: no cover - already absent
                pass
    # The proton the cysteine loses is *not* placed by hand: saturating the alkene
    # (or turning the nitrile into an imine, or opening the epoxide) leaves exactly
    # one open valence on the reference atom, and `AddHs` fills it with a hydrogen at
    # a geometry-consistent position.  Placing it manually is how a "proton" ends up
    # 1.0 A inside another atom and a force field reports a million kcal/mol.
    if proton_transfer and pattern.mechanism == "michael":
        notes.append("the alpha carbon gained the proton the cysteine loses (the Michael adduct)")
    elif proton_transfer and pattern.mechanism == "nitrile":
        notes.append("the imine nitrogen gained the proton the cysteine loses")
    elif proton_transfer and pattern.mechanism == "epoxide":
        notes.append("the ring oxygen gained the proton the cysteine loses")
    if cap_with_cb and nucleophile.cap_position is not None:
        cap = work.AddAtom(Chem.Atom(6))
        work.AddBond(int(sulfur), int(cap), Chem.BondType.SINGLE)
        cap_position = np.asarray(nucleophile.cap_position, dtype=float)
        work.GetConformer().SetAtomPosition(
            int(cap), (float(cap_position[0]), float(cap_position[1]), float(cap_position[2]))
        )
        notes.append(
            "the sulfur is capped with the cysteine's own CB; the rest of the protein is "
            "frozen and is not part of this fragment"
        )
    out = work.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"the adduct could not be fully sanitised ({exc})")
    out = Chem.AddHs(out, addCoords=True)
    if form_bond:
        # Place the sulfur at bonding distance along the approach vector the pose
        # already defines.  Every other atom keeps the pose's coordinates, so the
        # relaxation that follows measures the *pose's* distortion.
        conf = out.GetConformer()
        attack_point = np.asarray(
            [
                conf.GetAtomPosition(int(electrophile)).x,
                conf.GetAtomPosition(int(electrophile)).y,
                conf.GetAtomPosition(int(electrophile)).z,
            ],
            dtype=float,
        )
        sulfur_point = np.asarray(
            [
                conf.GetAtomPosition(int(sulfur)).x,
                conf.GetAtomPosition(int(sulfur)).y,
                conf.GetAtomPosition(int(sulfur)).z,
            ],
            dtype=float,
        )
        direction = _unit(sulfur_point - attack_point)
        if direction is None:  # pragma: no cover - coincident atoms
            direction = np.array([0.0, 0.0, 1.0])
        placed = attack_point + float(bond_length) * direction
        conf.SetAtomPosition(int(sulfur), (float(placed[0]), float(placed[1]), float(placed[2])))
        notes.append(
            f"the bond was formed by placing the sulfur at {float(bond_length):.2f} A along "
            "the pose's own approach vector; every other atom is at the pose geometry, so "
            "the local relaxation measures the pose's distortion"
        )
    return Adduct(
        mol=out,
        sulfur=int(sulfur),
        electrophile=int(electrophile),
        reference=int(reference),
        warhead=warhead.name,
        mechanism=pattern.mechanism,
        nucleophile=nucleophile.label(),
        notes=notes,
    )


def relax_adduct(
    adduct: Adduct,
    *,
    force_field: str = "MMFF94",
    steps: int = 500,
) -> Adduct:
    """Relax the adduct locally with the project's minimiser.

    This calls :func:`odock.chem.ligand.minimize` — the one minimiser this project
    has, with its documented MMFF94 -> MMFF94s -> UFF fallback and its clamped step
    count — rather than starting a second one.  "Locally" is literal: the fragment
    is the ligand plus the cysteine's S and CB, so the protein is frozen by
    construction and nothing outside the adduct moves.
    """
    require_rdkit()
    from .chem.ligand import minimize

    if not adduct.ok:
        return adduct
    minimised = minimize(adduct.mol, force_field=force_field, steps=int(steps))
    notes = list(adduct.notes)
    note = ""
    try:
        note = minimised.GetProp("odock_minimize_note")
    except Exception:  # pragma: no cover - defensive
        note = ""
    if note:
        notes.append(note)
    return replace(adduct, mol=minimised, relaxed=True, notes=notes)


def adduct_quality(adduct: Adduct) -> Dict[str, Any]:
    """The adduct's measured geometry against the published ranges.

    ``c_s_bond`` should be 1.70-1.95 A (a C-S single bond is about 1.82) and
    ``s_c_c_angle`` 95-125 degrees (tetrahedral at the former electrophilic
    carbon).  The result says whether each is inside its range and by how much it
    is not, so a distorted adduct is reported rather than merely scored.
    """
    require_rdkit()
    bond = adduct.bond_length()
    angle = adduct.bond_angle()
    out: Dict[str, Any] = {
        "bond_length": None if bond is None else round(float(bond), 4),
        "bond_angle": None if angle is None else round(float(angle), 4),
        "ranges": {key: list(value) for key, value in ADDUCT_GEOMETRY.items()},
    }
    if bond is not None:
        low, high = ADDUCT_GEOMETRY["c_s_bond"]
        out["bond_ok"] = bool(low <= bond <= high)
        out["bond_deviation"] = round(float(0.0 if low <= bond <= high else min(abs(bond - low), abs(bond - high))), 4)
    else:
        out["bond_ok"] = False
        out["bond_deviation"] = None
    if angle is not None:
        low, high = ADDUCT_GEOMETRY["s_c_c_angle"]
        out["angle_ok"] = bool(low <= angle <= high)
        out["angle_deviation"] = round(float(0.0 if low <= angle <= high else min(abs(angle - low), abs(angle - high))), 4)
    else:
        out["angle_ok"] = False
        out["angle_deviation"] = None
    out["ok"] = bool(out["bond_ok"] and out["angle_ok"])
    return out


def adduct_strain(adduct: Adduct, *, force_field: str = "MMFF94", steps: int = 500):
    """The strain the adduct carries, as an :class:`odock.metrics.StrainResult`.

    The number is the same one ``ligand_strain`` computes for a *ligand* — the
    force-field energy of the formed geometry minus the energy of its local
    minimum — and it is produced by the project's one minimiser
    (:func:`odock.chem.ligand.minimize`, whose ``odock_energy_before`` and
    ``odock_energy_after`` properties are the two endpoints), not by a second
    implementation of the relaxation.

    Why not the PDBQT round trip ``ligand_strain`` uses for a pose: a PDBQT carries
    no bond orders, so its *reliable* path needs ``prepare_ligand``'s atom-order
    map, and that map does not survive on an adduct fragment — the hydrogen merge
    reorders the molecule, and ``ligand_strain`` then refuses the molecule instead
    of patching coordinates onto the wrong atoms (which is exactly the behaviour its
    own docstring demands).  The chemistry here is explicit in the molecule, so
    ``reliable`` is ``True`` and the ``note`` says which quantity was measured over
    which geometry.  The kernel ``intra`` half is left ``nan``: a fragment has no
    receptor context.
    """
    require_rdkit()
    from .metrics import StrainResult

    if not adduct.ok:
        raise ValueError("the adduct is not built, so there is no strain to measure")
    relaxed = (
        adduct
        if adduct.relaxed
        else relax_adduct(adduct, force_field=force_field, steps=steps)
    )
    mol = relaxed.mol

    def _energy(prop: str) -> Optional[float]:
        try:
            value = float(mol.GetProp(prop).strip())
        except Exception:
            return None
        return value if math.isfinite(value) else None

    before = _energy("odock_energy_before")
    after = _energy("odock_energy_after")
    ran = ""
    try:
        ran = mol.GetProp("odock_force_field").strip()
    except Exception:
        ran = ""
    note_parts = [
        "the adduct fragment's force-field energy before and after a local "
        "relaxation through odock.chem.ligand.minimize",
        "the kernel intra half is not reported: a fragment has no receptor context",
    ]
    if ran:
        note_parts.append(f"force field {ran}")
    if before is None or after is None:
        note_parts.append("the energies are unavailable, so the strain is nan")
        return StrainResult(force_field=ran, reliable=False, note="; ".join(note_parts))
    strain = float(before) - float(after)
    note_parts.append(
        "reliable: the adduct's chemistry is explicit in the molecule, not perceived "
        "from a bond-order-free PDBQT"
    )
    return StrainResult(
        force_field=ran,
        force_field_strain=strain,
        energy_pose=float(before),
        energy_relaxed=float(after),
        reliable=bool(ran) and math.isfinite(strain),
        note="; ".join(note_parts),
    )


def interaction_change(
    receptor,
    ligand_before,
    ligand_after,
    *,
    limit: int = 8,
) -> Dict[str, Any]:
    """How the interaction profile changes when the ligand is relaxed into the adduct.

    Both arguments are RDKit molecules in the same frame (the pose geometry and the
    relaxed fragment's ligand atoms), so the difference is the geometry's, not the
    frame's.  The covalent S-C bond itself is **not** an interaction kind in
    :func:`odock.analysis.profile_interactions`; it is reported separately, which is
    the honest split: the profile describes the non-covalent contacts, the geometry
    describes the bond.
    """
    from .analysis import profile_interactions

    before = profile_interactions(receptor, ligand_before)
    after = profile_interactions(receptor, ligand_after)

    def _key(item) -> Tuple[str, str]:
        return (item.kind, item.detail)

    before_keys = {_key(item) for item in before}
    after_keys = {_key(item) for item in after}
    added = [item for item in after if _key(item) not in before_keys]
    lost = [item for item in before if _key(item) not in after_keys]

    def _counts(items) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts

    return {
        "before": _counts(before),
        "after": _counts(after),
        "added": [str(item) for item in added[:limit]],
        "lost": [str(item) for item in lost[:limit]],
        "n_added": len(added),
        "n_lost": len(lost),
        "note": (
            "the covalent S-C bond is not an interaction kind; it is described by the "
            "adduct geometry, and this profile covers the non-covalent contacts only"
        ),
    }


# ---------------------------------------------------------------------------
# The command line entry point
# ---------------------------------------------------------------------------


def add_covalent_parser(subparsers) -> None:
    """Register ``odock covalent`` on ``subparsers``.

    The one entry point :mod:`odock.cli_ext` needs: the registrar tuple is
    ``(".covalent", "add_covalent_parser")``.
    """
    parser = subparsers.add_parser(
        "covalent",
        help="covalent docking: warheads, reactive geometry, adducts",
        description=(
            "Measure whether a ligand's warhead can reach a receptor's nucleophile: "
            "perceive the warhead and its electrophilic atom, find the candidate "
            "cysteines by residue and atom name, classify the distance and approach "
            "angle as bonded / near-attack / non-productive, add the documented "
            "covalent term to the non-covalent score, and build and relax the adduct "
            "through the project's own minimiser.  These are geometry criteria, not "
            "chemistry: whether the warhead reacts and whether the cysteine is "
            "nucleophilic are experimental facts this tool does not have, and a "
            "near-attack pose is a hypothesis for a chemist, not a predicted inhibitor."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True, metavar="FILE",
        help="ligand file with 3-D coordinates (.sdf/.mol2/.smi/.pdb)",
    )
    parser.add_argument(
        "-r", "--receptor", required=True, metavar="FILE",
        help="receptor (.pdbqt or .pdb); its cysteines are the candidate nucleophiles",
    )
    parser.add_argument(
        "--nucleophile", metavar="SPEC",
        help="restrict to one residue, e.g. CYS797, CYS-797 or 'CYS 797 A'",
    )
    parser.add_argument(
        "--adduct", metavar="FILE",
        help="write the relaxed adduct of every productive pair as an SDF",
    )
    parser.add_argument("--json-out", metavar="FILE", help="write the full report as JSON")
    parser.set_defaults(func=cmd_covalent)


def _parse_nucleophile_spec(spec: str) -> Optional[Tuple[str, int]]:
    """``CYS797`` / ``CYS-797`` / ``CYS 797 A`` -> ``("CYS", 797)``."""
    text = str(spec).strip().upper().replace("-", " ").replace("_", " ")
    parts = [part for part in text.split() if part]
    if not parts:
        return None
    name = parts[0]
    number = None
    for part in parts[1:]:
        if part.isdigit():
            number = int(part)
            break
    if number is None:
        digits = "".join(ch for ch in name if ch.isdigit())
        name = "".join(ch for ch in name if not ch.isdigit())
        if digits:
            number = int(digits)
    return (name, int(number)) if number is not None else (name, 0)


def cmd_covalent(args) -> int:
    """Run the covalent analysis for one ligand against one receptor."""
    import json
    from pathlib import Path

    from .cli import _eprint, _lazy, _read_library

    covalent = _lazy("odock.covalent", "covalent docking")
    mols, names = _read_library([args.input])
    receptor = covalent._load_receptor(args.receptor)
    if receptor is None:
        _eprint(f"odock covalent: could not read the receptor {args.receptor!r}")
        return 1
    nucleophiles = covalent.perceive_nucleophiles(receptor)
    if not nucleophiles:
        _eprint(
            "odock covalent: no nucleophile candidates in the receptor (no CYS SG found, "
            "and no geometric sulfur either)"
        )
        return 1
    if args.nucleophile:
        wanted = _parse_nucleophile_spec(args.nucleophile)
        if wanted is not None:
            name, number = wanted
            filtered = [
                item
                for item in nucleophiles
                if (not number or item.residue_number == number)
                and (not name or item.residue_name == name)
            ]
            if not filtered:
                _eprint(
                    f"odock covalent: {args.nucleophile!r} matches none of the "
                    f"{len(nucleophiles)} candidate(s): "
                    + ", ".join(item.label() for item in nucleophiles)
                )
                return 1
            nucleophiles = filtered
    print(f"receptor: {len(nucleophiles)} nucleophile candidate(s)")
    for item in nucleophiles:
        print(f"  {item.label()} at ({item.position[0]:.2f}, {item.position[1]:.2f}, "
              f"{item.position[2]:.2f}) {('- ' + item.note) if item.note else ''}")

    writer = None

    def _writer():
        """Open the adduct SDF on the first productive pair.

        An empty SDF is a worse outcome than no file at all — it looks like an
        adduct that failed to write — so the file is only created when there is
        something to put in it.
        """
        nonlocal writer
        if writer is None and args.adduct:
            target = Path(args.adduct)
            target.parent.mkdir(parents=True, exist_ok=True)
            from rdkit import Chem

            writer = Chem.SDWriter(str(target))
        return writer

    report: Dict[str, Any] = {"receptor": str(args.receptor), "ligands": []}
    productive = 0
    for mol, name in zip(mols, names):
        warheads = covalent.perceive_warheads(mol)
        print()
        print(f"{name}: {len(warheads)} warhead(s)")
        if not warheads:
            print("  no documented warhead: nothing to classify (this is not a 'safe' verdict)")
        entry: Dict[str, Any] = {"name": name, "warheads": [hit.as_dict() for hit in warheads],
                                 "pairs": []}
        for warhead in warheads:
            for nucleophile in nucleophiles:
                geometry = covalent.classify_attack(mol, warhead, nucleophile)
                term = covalent.covalent_term(geometry)
                print(f"  {geometry.table()}")
                print(f"    term {term.value:+.3f} kcal/mol "
                      f"(near {term.near_attack:.3f}, bond {term.bond:.3f}, "
                      f"strain {term.strain_penalty:.3f})")
                pair: Dict[str, Any] = {"geometry": geometry.as_dict(),
                                        "term": term.as_dict()}
                if geometry.productive:
                    productive += 1
                    adduct = covalent.build_adduct(mol, warhead, nucleophile)
                    relaxed = covalent.relax_adduct(adduct)
                    quality = covalent.adduct_quality(relaxed)
                    strain = covalent.adduct_strain(relaxed)
                    print(f"    adduct: C-S {quality['bond_length']:.2f} A, "
                          f"S-C-C {quality['bond_angle']:.1f} deg, "
                          f"in range: {quality['ok']}")
                    print(f"    strain: {strain.force_field_strain:.3f} kcal/mol "
                          f"(reliable: {strain.reliable}, {strain.force_field})")
                    pair["adduct"] = quality
                    pair["strain"] = strain.as_dict()
                    handle = _writer()
                    if handle is not None:
                        for atom in relaxed.mol.GetAtoms():
                            atom.SetProp(
                                "_Name",
                                f"{name}_{warhead.name}_{nucleophile.label()}"
                                .replace(":", ""),
                            )
                        handle.write(relaxed.mol)
                entry["pairs"].append(pair)
        report["ligands"].append(entry)
    if writer is not None:
        writer.close()
        _eprint(f"wrote {args.adduct}")
    print()
    print(f"{productive} productive pair(s). A near-attack geometry is a hypothesis for a "
          "chemist: this tool does not know whether the warhead reacts or whether the "
          "cysteine is nucleophilic.")
    if args.json_out:
        from .cli import _write_json

        _write_json(args.json_out, report)
    return 0

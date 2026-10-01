# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-processing analysis of docked poses: module E of the requirements.

Three things live here:

* **clustering** -- symmetry-aware RMSD between poses and single-linkage
  clustering of the docked conformations (E.1);
* **interaction profiling** -- the six non-covalent interaction types with the
  exact geometric criteria of E.2, plus the LigPlot-style 2-D diagram and the
  pose interpolation used by the conformation player (E.3);
* nothing else: file exports live in :mod:`odock.report`.

Input conventions
-----------------
Every entry point accepts either

* a sequence of **Atom-like objects**, i.e. anything exposing ``element``,
  ``x``/``y``/``z`` and optionally ``name``/``res_name``/``res_id``/``chain``
  (and ``charge`` for AutoDock partial charges) -- this is exactly
  :class:`odock.gui.structure.Atom`, and it is what the workbench passes; or
* an **RDKit molecule**, whose bonds, ring perception and formal charges are
  used directly.

Coordinates are ``(N, 3) float64`` arrays. Hydrogens are recognised by the
element symbol ``H`` (or ``D``).
"""

from __future__ import annotations

import math
import os
import xml.sax.saxutils as _saxutils
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pocket import VDW_RADII

__all__ = [
    "Cluster",
    "FingerprintKey",
    "FingerprintSchema",
    "FingerprintSet",
    "Interaction",
    "InteractionFingerprint",
    "INTERACTION_COLORS",
    "INTERACTION_LABELS",
    "INTERACTION_KINDS",
    "COVALENT_RADII",
    "PharmacophoreSummary",
    "VDW_RADII",
    "WATER_KIND",
    "WATER_RESNAMES",
    "cluster_poses",
    "fingerprint_similarity",
    "interaction_diagram_svg",
    "interaction_fingerprint",
    "interaction_summary",
    "interpolate_coords",
    "pharmacophore_summary",
    "pose_fingerprints",
    "profile_interactions",
    "similarity_matrix",
    "symmetry_aware_rmsd",
    "symmetry_classes",
    "water_mediated_contacts",
]


# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

#: Single-bond covalent radii in Å (Cordero 2008), used to perceive bonds when
#: the input is a list of Atom-like objects rather than an RDKit molecule.
COVALENT_RADII: Dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Mo": 1.54, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
    "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38,
    "I": 1.39, "Xe": 1.40,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
}
_DEFAULT_COVALENT = 0.77

#: Elements that are never a hydrogen-bond donor or acceptor.
_HYDROGEN = frozenset(("H", "D"))
_POLAR = frozenset(("N", "O", "S"))

#: Metal cations that always count as a cationic centre.
_METALS = frozenset(
    ("Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba",
     "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Cd", "Hg")
)

#: A C-O bond shorter than this is a carbonyl rather than a hydroxyl/ether.
_CARBONYL_CUTOFF = 1.30

#: Bond-perception tolerance: two atoms are bonded when their distance is below
#: this factor times the sum of their covalent radii.
_BOND_FACTOR = 1.30

#: Above this many non-template atoms the geometric ring perception is skipped;
#: it is a small-molecule fallback, not a whole-protein algorithm.
_MAX_RING_ATOMS = 400

#: Maximum ring size considered by the geometric perception (aromatic rings are
#: 5- or 6-membered).
_MAX_RING_SIZE = 6

#: A ring counts as aromatic when every atom is within this distance (Å) of the
#: ring's best-fit plane.
_PLANAR_TOLERANCE = 0.20

#: Colours required by E.2 for the 3-D annotation and the 2-D diagram.
INTERACTION_COLORS: Dict[str, str] = {
    "hbond": "#00bcd4",        # cyan
    "salt_bridge": "#e91e63",  # magenta
    "pi_pi": "#00c853",        # emerald green
    "cation_pi": "#ff9800",    # orange
    "hydrophobic": "#9e9e9e",  # grey
    "clash": "#ff1744",        # bright red
    "water_bridge": "#3f51b5", # indigo, for the water-mediated case
}

#: Human-readable names for the legend.
INTERACTION_LABELS: Dict[str, str] = {
    "hbond": "Hydrogen bond",
    "salt_bridge": "Salt bridge",
    "pi_pi": "pi-pi stacking",
    "cation_pi": "Cation-pi",
    "hydrophobic": "Hydrophobic",
    "clash": "Steric clash",
    "water_bridge": "Water-mediated",
}

#: The interaction kinds, best first.  ``clash`` is not a pharmacophore feature
#: and is excluded from the fingerprints; the fingerprints add
#: :data:`WATER_KIND` when water bridges are detected.
INTERACTION_KINDS: Tuple[str, ...] = (
    "hbond",
    "salt_bridge",
    "pi_pi",
    "cation_pi",
    "hydrophobic",
)

#: Residue names that are crystallographic water, for the water-bridge scan.
WATER_RESNAMES = frozenset(("HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"))

#: The interaction kind used for a water-mediated contact.
WATER_KIND = "water_bridge"

#: Order in which interactions are reported.
_KIND_ORDER = {
    "hbond": 0,
    "salt_bridge": 1,
    "pi_pi": 2,
    "cation_pi": 3,
    "hydrophobic": 4,
    "clash": 5,
}

#: Aromatic rings of the standard residues, by PDB atom name. Using the
#: residue dictionary avoids running ring perception over a whole protein.
_TEMPLATE_RINGS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HID": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HIE": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HIP": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HSD": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HSE": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "HSP": (("CG", "ND1", "CD2", "CE1", "NE2"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    ),
}

#: Rendering palette for the 2-D diagram.
_ELEMENT_FILL = {
    "C": "#cfd8dc", "N": "#7986cb", "O": "#e57373", "S": "#ffd54f",
    "P": "#ffb74d", "F": "#a5d6a7", "Cl": "#a5d6a7", "Br": "#bcaaa4",
    "I": "#b39ddb", "H": "#eceff1",
}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """One cluster of docked poses.

    Attributes
    ----------
    index
        0-based rank after sorting by best energy, then by size.
    members
        Indices into the input pose list, ascending.
    representative
        The lowest-energy member (the cluster's best pose).
    best_energy
        Energy of the representative, or ``None`` when no energies were given.
    mean_rmsd
        Mean pairwise symmetry-aware RMSD between the members (Å); ``0.0`` for
        a one-member cluster.
    """

    index: int
    members: List[int]
    representative: int
    best_energy: Optional[float]
    mean_rmsd: float

    def __len__(self) -> int:
        return len(self.members)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        energy = "-" if self.best_energy is None else f"{self.best_energy:.2f}"
        return (
            f"Cluster {self.index + 1}: {len(self.members)} poses, "
            f"best={energy} kcal/mol, mean RMSD={self.mean_rmsd:.2f} A"
        )


@dataclass
class Interaction:
    """One non-covalent contact between the receptor and the ligand.

    Attributes
    ----------
    kind
        ``"hbond"``, ``"salt_bridge"``, ``"pi_pi"``, ``"cation_pi"``,
        ``"hydrophobic"`` or ``"clash"``.
    a
        Atom index into the **receptor**. For ring- and charge-based contacts
        this is a representative atom (the first atom of the ring, or one atom
        of the charged group); the reported `distance` is then measured between
        the group centres, not between ``a`` and ``b``.
    b
        Atom index into the **ligand**, with the same convention.
    distance
        The geometric quantity the criterion was applied to (Å).
    detail
        Short human-readable description, e.g. ``"SER195:OG->LIG1:O1"``,
        ``"face-to-face"`` or ``"ASP189:OD1/OD2...LIG1:N1"``.
    subtype
        ``"face"`` or ``"edge"`` for ``pi_pi``, otherwise empty.
    residue
        ``(res_name, res_id, chain)`` of the **receptor** residue the contact
        belongs to, filled in by :func:`profile_interactions` and
        :func:`water_mediated_contacts`.  It is what makes an interaction
        fingerprint possible without re-reading the receptor, and it is ``None``
        for an interaction a caller built by hand.
    """

    kind: str
    a: int
    b: int
    distance: float
    detail: str = ""
    subtype: str = ""
    residue: Optional[Tuple[str, int, str]] = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind}({self.a}, {self.b}, {self.distance:.2f} A) {self.detail}"


# ---------------------------------------------------------------------------
# Internal representation
# ---------------------------------------------------------------------------


@dataclass
class _Atom:
    index: int
    element: str
    x: float
    y: float
    z: float
    name: str = ""
    res_name: str = ""
    res_id: int = 0
    chain: str = ""
    charge: float = 0.0
    formal_charge: int = 0
    aromatic: bool = False


@dataclass
class _Structure:
    atoms: List[_Atom]
    coords: np.ndarray
    bonds: List[Tuple[int, int]]
    adjacency: List[List[int]]
    rings: List[List[int]]
    hydrogens: Dict[int, List[int]]
    donors: List[int]
    acceptors: List[int]
    hydrophobic_carbons: List[int]
    cations: List[Tuple[List[int], np.ndarray, str]]
    anions: List[Tuple[List[int], np.ndarray, str]]
    has_hydrogens: bool
    bond_orders: Optional[Dict[Tuple[int, int], float]] = None

    @property
    def elements(self) -> List[str]:
        return [a.element for a in self.atoms]

    def label(self, index: int) -> str:
        """``"SER195:OG"`` for an atom, degrading to the element symbol."""
        atom = self.atoms[index]
        name = atom.name or atom.element
        if atom.res_name:
            return f"{atom.res_name}{atom.res_id}:{name}"
        return name


# ---------------------------------------------------------------------------
# Reading either an RDKit molecule or Atom-like objects
# ---------------------------------------------------------------------------


def _read_rdkit(mol):
    atoms: List[_Atom] = []
    conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
    for atom in mol.GetAtoms():
        if conf is not None:
            pos = conf.GetAtomPosition(atom.GetIdx())
            x, y, z = float(pos.x), float(pos.y), float(pos.z)
        else:
            x = y = z = 0.0
        info = atom.GetPDBResidueInfo()
        atoms.append(
            _Atom(
                index=atom.GetIdx(),
                element=atom.GetSymbol(),
                x=x,
                y=y,
                z=z,
                name=(info.GetName().strip() if info is not None else ""),
                res_name=(info.GetResidueName().strip() if info is not None else ""),
                res_id=(int(info.GetResidueNumber()) if info is not None else 0),
                chain=(info.GetChainId().strip() if info is not None else ""),
                formal_charge=int(atom.GetFormalCharge()),
                aromatic=bool(atom.GetIsAromatic()),
            )
        )
    bonds: List[Tuple[int, int]] = []
    orders: Dict[Tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds.append((i, j))
        orders[(i, j)] = orders[(j, i)] = float(bond.GetBondTypeAsDouble())

    rings: List[List[int]] = []
    try:
        from rdkit import Chem

        raw = [list(r) for r in Chem.GetSymmSSSR(mol)]
    except Exception:  # pragma: no cover - ring perception is best effort
        raw = []
    aromatic = [r for r in raw if len(r) in (5, 6) and all(atoms[i].aromatic for i in r)]
    if aromatic:
        rings = aromatic
    else:
        # An unsanitised PDB-derived molecule carries no aromatic flags; fall
        # back to the geometric test on the rings RDKit did perceive.
        rings = [r for r in raw if len(r) in (5, 6)]
    return atoms, bonds, rings, orders


def _read_sequence(source):
    atoms: List[_Atom] = []
    for index, item in enumerate(source):
        if not all(hasattr(item, attr) for attr in ("element", "x", "y", "z")):
            raise TypeError(f"cannot interpret {item!r} as an atom")
        atoms.append(
            _Atom(
                index=index,
                element=str(getattr(item, "element") or "C"),
                x=float(item.x),
                y=float(item.y),
                z=float(item.z),
                name=str(getattr(item, "name", "") or ""),
                res_name=str(getattr(item, "res_name", "") or ""),
                res_id=int(getattr(item, "res_id", 0) or 0),
                chain=str(getattr(item, "chain", "") or ""),
                charge=float(getattr(item, "charge", 0.0) or 0.0),
            )
        )
    return atoms


def _perceive_bonds(coords: np.ndarray, elements: Sequence[str]):
    """Distance-based bond perception, good enough for the geometry rules.

    A uniform cell list keeps this linear: for every atom only the 27
    neighbouring cells are examined, so a 3000-atom receptor costs a
    millisecond-scale scan rather than 4.5 million pair tests.
    """
    n = len(coords)
    if n < 2:
        return []
    radii = np.array(
        [COVALENT_RADII.get(e, _DEFAULT_COVALENT) for e in elements], dtype=float
    )
    cell = 2.0 * _BOND_FACTOR * float(radii.max())
    keys = np.floor(coords / cell).astype(np.int64)

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)

    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        kx, ky, kz = (int(v) for v in keys[i])
        candidates: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    got = buckets.get((kx + dx, ky + dy, kz + dz))
                    if got:
                        candidates.extend(got)
        if not candidates:
            continue
        cand = np.asarray([j for j in candidates if j > i], dtype=np.int64)
        if cand.size == 0:
            continue
        d = np.linalg.norm(coords[cand] - coords[i], axis=1)
        limit = _BOND_FACTOR * (radii[i] + radii[cand])
        for j in cand[d <= limit]:
            bonds.append((i, int(j)))
    return bonds


def _adjacency(n: int, bonds: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)
    return adj


def _geometric_rings(atoms, coords, elements, adjacency, allowed: Sequence[int]):
    """5- and 6-membered rings of a small molecule, by geometry alone.

    A cycle is kept when it is planar (every atom within
    :data:`_PLANAR_TOLERANCE` of the best-fit plane) and every carbon in it is
    sp2 (at most three connections). That rejects cyclohexane and other
    saturated rings, which a pure planarity test does not.
    """
    allowed_set = set(int(i) for i in allowed)
    found: Dict[Tuple[int, ...], List[int]] = {}

    def walk(start: int, node: int, path: List[int], visited: set) -> None:
        for nxt in adjacency[node]:
            if nxt == start:
                if len(path) >= 3:
                    found.setdefault(tuple(sorted(path)), list(path))
            elif (
                nxt in allowed_set
                and nxt not in visited
                and nxt > start
                and len(path) < _MAX_RING_SIZE
            ):
                visited.add(nxt)
                path.append(nxt)
                walk(start, nxt, path, visited)
                path.pop()
                visited.discard(nxt)

    for start in sorted(allowed_set):
        walk(start, start, [start], {start})

    rings: List[List[int]] = []
    for key, _ring in found.items():
        if len(key) not in (5, 6):
            continue
        pts = coords[list(key)]
        centred = pts - pts.mean(axis=0)
        if np.linalg.norm(centred) < 1e-9:
            continue
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        deviation = float(np.abs(centred @ vt[2]).max())
        if deviation > _PLANAR_TOLERANCE:
            continue
        if not all(elements[i] != "C" or len(adjacency[i]) <= 3 for i in key):
            continue
        rings.append(sorted(key))
    return rings


def _template_rings(atoms) -> Tuple[List[List[int]], set]:
    """Aromatic rings of PHE/TYR/TRP/HIS taken from the residue dictionary."""
    residues: Dict[Tuple[str, int, str], Dict[str, int]] = {}
    for atom in atoms:
        if not atom.res_name or not atom.name:
            continue
        key = (atom.chain, atom.res_id, atom.res_name)
        residues.setdefault(key, {})[atom.name] = atom.index
    rings: List[List[int]] = []
    used: set = set()
    for (_chain, _res_id, res_name), names in residues.items():
        templates = _TEMPLATE_RINGS.get(res_name)
        if not templates:
            continue
        for template in templates:
            if all(name in names for name in template):
                ring = sorted(names[name] for name in template)
                rings.append(ring)
                used.update(ring)
    return rings, used


def _is_carbonyl(adjacency, hydrogens, bond_orders, coords, carbon, oxygen) -> bool:
    """Whether C-O is a carbonyl (a double bond, however it is represented)."""
    if bond_orders is not None:
        order = bond_orders.get((carbon, oxygen))
        if order is not None:
            return order >= 1.5
    if hydrogens.get(oxygen):
        return False
    return bool(
        len(adjacency[oxygen]) == 1
        and np.linalg.norm(coords[carbon] - coords[oxygen]) <= _CARBONYL_CUTOFF
    )


def _classify(atoms, coords, elements, adjacency, hydrogens, rings, bond_orders):
    """Donors, acceptors, non-polar carbons and charged centres.

    The rules are standard and deliberately conservative:

    * a **donor** is N/O/S with a non-positive formal charge that carries a
      hydrogen -- or, when the structure carries no hydrogens at all, any such
      heteroatom (the H-bond angle is then estimated geometrically, see
      :func:`profile_interactions`);
    * an **acceptor** is O or S with a non-positive formal charge, or N with a
      non-positive formal charge that is not an amide nitrogen and not already
      saturated;
    * a **hydrophobic carbon** is a carbon whose neighbours are only carbon and
      hydrogen;
    * a **carboxylate, phosphate or sulfate** group, a deprotonated oxygen, a
      metal ion, an ammonium nitrogen, a guanidinium carbon and a protonated
      aromatic nitrogen are the charged centres.
    """
    has_h = any(a.element in _HYDROGEN for a in atoms)

    donors: List[int] = []
    acceptors: List[int] = []
    hydrophobic: List[int] = []
    for i, atom in enumerate(atoms):
        z = atom.element
        heavy = [j for j in adjacency[i] if elements[j] not in _HYDROGEN]
        if z in _POLAR and atom.formal_charge <= 0:
            if hydrogens.get(i) or not has_h:
                donors.append(i)
            if z in ("O", "S"):
                acceptors.append(i)
            else:
                amide = False
                for j in heavy:
                    if elements[j] != "C":
                        continue
                    for k in adjacency[j]:
                        if elements[k] == "O" and _is_carbonyl(
                            adjacency, hydrogens, bond_orders, coords, j, k
                        ):
                            amide = True
                if not amide and len(adjacency[i]) < 4:
                    acceptors.append(i)
        elif z == "C" and all(elements[j] in ("C", "H", "D") for j in adjacency[i]):
            hydrophobic.append(i)

    cations: List[Tuple[List[int], np.ndarray, str]] = []
    anions: List[Tuple[List[int], np.ndarray, str]] = []
    covered: set = set()

    def centre(indices: Sequence[int]) -> np.ndarray:
        return coords[list(indices)].mean(axis=0)

    def label(indices: Sequence[int]) -> str:
        first = atoms[indices[0]]
        stem = f"{first.res_name}{first.res_id}:" if first.res_name else ""
        return stem + "/".join(atoms[i].name or atoms[i].element for i in indices)

    for i, atom in enumerate(atoms):
        z = atom.element
        if z == "C":
            oxygens = [j for j in adjacency[i] if elements[j] == "O"]
            if (
                len(oxygens) == 2
                and all(not hydrogens.get(j) for j in oxygens)
                and all(len(adjacency[j]) == 1 for j in oxygens)
            ):
                anions.append((list(oxygens), centre(oxygens), label(oxygens)))
                covered.update(oxygens)
        elif z in ("P", "S"):
            oxygens = [j for j in adjacency[i] if elements[j] == "O"]
            free = [j for j in oxygens if not hydrogens.get(j)]
            if len(oxygens) >= 3 and free:
                anions.append((free, centre(free), label(free)))
                covered.update(free)
        elif z == "O" and atom.formal_charge <= 0:
            heavy = [j for j in adjacency[i] if elements[j] not in _HYDROGEN]
            if len(heavy) == 1 and not hydrogens.get(i) and i not in covered:
                anions.append(([i], coords[i], label([i])))
                covered.add(i)
        elif z == "N" and atom.formal_charge <= 0:
            nitrogens = [j for j in adjacency[i] if elements[j] == "N"]
            if len(adjacency[i]) == 4:
                # Four connections (heavy or hydrogen) is a quaternary or
                # protonated ammonium nitrogen.
                cations.append(([i], coords[i], label([i])))
            elif len(nitrogens) == 3 and any(hydrogens.get(j) for j in nitrogens):
                group = [i] + nitrogens
                cations.append((group, centre(group), label(group)))
                covered.update(group)
        if z in _METALS:
            cations.append(([i], coords[i], label([i])))

    # Protonated aromatic ring nitrogens (imidazolium, pyridinium).
    ring_atoms: set = set()
    for ring in rings:
        ring_atoms.update(ring)
    for i, atom in enumerate(atoms):
        if atom.element != "N" or i not in ring_atoms:
            continue
        if len(adjacency[i]) == 3 and hydrogens.get(i):
            cations.append(([i], coords[i], label([i])))

    # Explicit charges always win over the structural guesses.
    cation_atoms = {a for group, _c, _l in cations for a in group}
    anion_atoms = {a for group, _c, _l in anions for a in group}
    for i, atom in enumerate(atoms):
        if atom.formal_charge >= 1 and i not in cation_atoms:
            cations.append(([i], coords[i], label([i])))
        elif atom.formal_charge <= -1 and i not in anion_atoms:
            anions.append(([i], coords[i], label([i])))
        elif atom.formal_charge == 0 and abs(atom.charge) >= 0.75 and i not in covered:
            if atom.charge > 0:
                cations.append(([i], coords[i], label([i])))
            else:
                anions.append(([i], coords[i], label([i])))

    return donors, acceptors, hydrophobic, cations, anions, has_h


def _read_pdbqt(source) -> List["PoseAtom"]:
    """Atoms of a PDBQT document given as text or as a path."""
    from .consensus import pdbqt_atoms

    raw = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
    if not raw.lstrip().startswith(("ATOM", "HETATM", "REMARK", "ROOT", "MODEL")):
        path = Path(raw)
        if path.exists() and path.is_file():
            raw = path.read_text(encoding="utf-8", errors="replace")
    atoms = pdbqt_atoms(raw)
    if not atoms:
        raise ValueError(
            "the given text is neither PDBQT nor a path to a PDBQT/PDB file with "
            "ATOM records"
        )
    return atoms


def _structure(source) -> _Structure:
    """Normalise `source` into a :class:`_Structure`.

    Idempotent: passing a :class:`_Structure` back in returns it unchanged, so
    callers that profile many poses against one receptor normalise the receptor
    (and its bond perception) exactly once.

    `source` may also be **PDBQT text** or a path to a PDBQT/PDB file; the
    document is parsed with :func:`odock.consensus.pdbqt_atoms` (imported lazily,
    because :mod:`odock.consensus` is the layer that owns the PDBQT plumbing and
    it never imports this module back).
    """
    if isinstance(source, _Structure):
        return source

    if isinstance(source, (str, os.PathLike)):
        source = _read_pdbqt(source)

    bonds: List[Tuple[int, int]]
    orders: Optional[Dict[Tuple[int, int], float]]
    rings: List[List[int]] = []

    if hasattr(source, "GetNumAtoms") and hasattr(source, "GetAtoms"):
        atoms, bonds, rings, orders = _read_rdkit(source)
    else:
        atoms = _read_sequence(source)
        orders = None
        if not atoms:
            return _Structure(
                atoms=[], coords=np.zeros((0, 3)), bonds=[], adjacency=[],
                rings=[], hydrogens={}, donors=[], acceptors=[],
                hydrophobic_carbons=[], cations=[], anions=[],
                has_hydrogens=False,
            )
        coords0 = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
        bonds = _perceive_bonds(coords0, [a.element for a in atoms])

    n = len(atoms)
    coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float).reshape(n, 3)
    elements = [a.element for a in atoms]
    adjacency = _adjacency(n, bonds)

    hydrogens: Dict[int, List[int]] = {}
    for i, j in bonds:
        if elements[i] in _HYDROGEN and elements[j] not in _HYDROGEN:
            hydrogens.setdefault(j, []).append(i)
        elif elements[j] in _HYDROGEN and elements[i] not in _HYDROGEN:
            hydrogens.setdefault(i, []).append(j)

    if not rings:
        template, used = _template_rings(atoms)
        rings = list(template)
        rest = [i for i in range(n) if i not in used]
        if rest and len(rest) <= _MAX_RING_ATOMS:
            rings.extend(_geometric_rings(atoms, coords, elements, adjacency, rest))

    donors, acceptors, hydrophobic, cations, anions, has_h = _classify(
        atoms, coords, elements, adjacency, hydrogens, rings, orders
    )
    return _Structure(
        atoms=atoms,
        coords=coords,
        bonds=list(bonds),
        adjacency=adjacency,
        rings=rings,
        hydrogens=hydrogens,
        donors=donors,
        acceptors=acceptors,
        hydrophobic_carbons=hydrophobic,
        cations=cations,
        anions=anions,
        has_hydrogens=has_h,
        bond_orders=orders,
    )


def _pairs_between(a: _Structure, sel_a, b: _Structure, sel_b, cutoff: float):
    """All ``(i, j, d)`` with ``i`` in `sel_a`, ``j`` in `sel_b`, ``d <= cutoff``."""
    sel_a = np.asarray(list(sel_a), dtype=np.int64)
    sel_b = np.asarray(list(sel_b), dtype=np.int64)
    empty = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=float),
    )
    if sel_a.size == 0 or sel_b.size == 0:
        return empty

    cell = max(float(cutoff), 0.5)
    pa = a.coords[sel_a]
    keys = np.floor(pa / cell).astype(np.int64)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for pos, key in enumerate(map(tuple, keys)):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(pos)

    out_a: List[np.ndarray] = []
    out_b: List[np.ndarray] = []
    out_d: List[np.ndarray] = []
    for j in sel_b:
        point = b.coords[j]
        key = np.floor(point / cell).astype(np.int64)
        candidate: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    got = buckets.get(
                        (int(key[0]) + dx, int(key[1]) + dy, int(key[2]) + dz)
                    )
                    if got:
                        candidate.extend(got)
        if not candidate:
            continue
        local = np.asarray(candidate, dtype=np.int64)
        d = np.linalg.norm(pa[local] - point, axis=1)
        keep = d <= cutoff
        if not keep.any():
            continue
        out_a.append(sel_a[local[keep]])
        out_b.append(np.full(int(keep.sum()), int(j), dtype=np.int64))
        out_d.append(d[keep])

    if not out_a:
        return empty
    return np.concatenate(out_a), np.concatenate(out_b), np.concatenate(out_d)


def _ring_geometry(structure: _Structure, ring: Sequence[int]):
    """Centroid and unit normal of a ring."""
    pts = structure.coords[list(ring)]
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    if len(pts) >= 3 and np.linalg.norm(centred) > 1e-9:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        normal = np.asarray(vt[2], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            normal = normal / norm
            return centroid, normal
    return centroid, np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# E.1 -- symmetry-aware RMSD and clustering
# ---------------------------------------------------------------------------

_UNPROBED = object()
_SCIPY_LSA: Any = _UNPROBED


def _scipy_lsa():
    """scipy's assignment solver when it happens to be installed, else None.

    scipy is *not* a dependency; the probe is lazy and cached, and the inline
    Hungarian solver is used whenever it is missing.
    """
    global _SCIPY_LSA
    if _SCIPY_LSA is _UNPROBED:
        try:
            from scipy.optimize import linear_sum_assignment

            _SCIPY_LSA = linear_sum_assignment
        except Exception:  # pragma: no cover - scipy absent by design
            _SCIPY_LSA = None
    return _SCIPY_LSA


def _hungarian(cost: np.ndarray) -> np.ndarray:
    """Minimum-cost perfect assignment of a square matrix (O(n³) JV).

    Returns, for every row, the column it is assigned to. This is the inline
    implementation used when SciPy is unavailable; it is exact.
    """
    a = np.asarray(cost, dtype=float)
    if a.ndim != 2:
        raise ValueError("the cost matrix must be two-dimensional")
    n, m = a.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n != m:
        raise ValueError(f"the cost matrix must be square, got {n}x{m}")

    inf = np.inf
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)  # p[column] = row (1-based)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            cur = a[i0 - 1] - u[i0] - v[1:]
            cur = np.where(used[1:], inf, cur)
            better = cur < minv[1:]
            if better.any():
                minv[1:][better] = cur[better]
                way[1:][better] = j0
            # The augmenting path may only step onto a column that is still
            # free, so the candidate set excludes the columns already used by
            # the alternating tree.
            candidate = np.where(used[1:], inf, minv[1:])
            j1 = int(np.argmin(candidate)) + 1
            delta = float(candidate[j1 - 1])
            np.add.at(u, p[used], delta)
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    assignment = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] > 0:
            assignment[int(p[j]) - 1] = j - 1
    return assignment


def _assign(cost: np.ndarray):
    """Solve a square assignment problem; SciPy when present, else inline."""
    a = np.asarray(cost, dtype=float)
    solver = _scipy_lsa()
    if solver is not None and a.shape[0] >= 1:
        rows, cols = solver(a)
        return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    return np.arange(a.shape[0], dtype=np.int64), _hungarian(a)


def symmetry_classes(mol) -> List[int]:
    """RDKit symmetry classes: the orbits of the molecular graph automorphisms.

    ``Chem.CanonicalRankAtoms(mol, breakTies=False)`` gives interchangeable
    atoms the same integer. Pass the result as the `elements` argument of
    :func:`symmetry_aware_rmsd` (or ``elements=`` of :func:`cluster_poses`) for
    graph-accurate symmetry handling instead of element-only matching.
    """
    from rdkit import Chem

    return [int(rank) for rank in Chem.CanonicalRankAtoms(mol, breakTies=False)]


def symmetry_aware_rmsd(a, b, elements) -> float:
    """Minimum RMSD of `a` onto `b` over the ligand's automorphisms.

    Both inputs are ``(N, 3)`` arrays of the *same* molecule in the same atom
    order, and `elements` is a length-`N` sequence of per-atom equivalence
    labels: atoms may only be mapped onto atoms carrying the same label.

    * With element symbols (``["C", "C", "O", ...]``) the matching is
      element-only -- the standard, safe fallback.
    * With :func:`symmetry_classes` output it is restricted to the orbits of
      the molecular graph, which is what makes the equivalent oxygens of a
      carboxylate and the six carbons of a benzene interchangeable.

    The result is **exact for the matching it is given**: the correspondence is
    solved as a linear assignment problem (SciPy's solver when SciPy happens to
    be installed, otherwise the inline O(n³) Hungarian algorithm), which is the
    exact minimum over all bijections that respect the labels. It is *not* the
    graph-isomorphism minimum unless graph-accurate labels are supplied --
    element-only labels can over-permute. No rigid-body superposition is
    performed: two poses are compared in the frame they were produced in,
    exactly as pose clustering requires.
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.shape != B.shape:
        raise ValueError(
            f"the two coordinate sets must have the same shape, got {A.shape} and {B.shape}"
        )
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N, 3), got {A.shape}")
    labels = list(elements)
    n = A.shape[0]
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels were given for {n} atoms")
    if n == 0:
        return 0.0

    groups: Dict[Any, List[int]] = {}
    for i, label in enumerate(labels):
        try:
            key: Any = ("value", label)
            hash(key)
        except TypeError:  # pragma: no cover - unhashable label
            key = ("repr", repr(label))
        groups.setdefault(key, []).append(i)

    total = 0.0
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        diff = A[idx][:, None, :] - B[idx][None, :, :]
        cost = (diff * diff).sum(axis=-1)
        rows, cols = _assign(cost)
        total += float(cost[rows, cols].sum())
    return float(math.sqrt(total / n))


def cluster_poses(
    coords,
    *,
    cutoff: float = 2.0,
    elements=None,
    energies=None,
) -> List[Cluster]:
    """Single-linkage clustering of docked poses by symmetry-aware RMSD.

    Parameters
    ----------
    coords
        An ``(M, N, 3)`` array or a list of ``M`` ``(N, 3)`` arrays, one per
        pose, all in the same atom order.
    cutoff
        RMSD at which two poses belong to the same cluster (Å, default 2.0 --
        the heavy-atom threshold of E.1). Single linkage: a pose joins a
        cluster as soon as it is within `cutoff` of *any* member, so transitive
        chains merge.
    elements
        Length-`N` per-atom equivalence labels (element symbols, or the output
        of :func:`symmetry_classes`). ``None`` treats every atom as
        interchangeable, which is only sensible for a homonuclear molecule.
    energies
        Length-`M` per-pose energies, used to pick each cluster's
        representative (the lowest energy) and to sort the clusters.

    Returns
    -------
    Clusters sorted by best energy (ascending; ``None`` energies last), then by
    descending size, renumbered ``0..k-1``. `mean_rmsd` is the mean pairwise
    symmetry-aware RMSD inside the cluster.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    C = np.asarray(coords, dtype=float)
    if C.ndim == 2:
        C = C[None, :, :]
    if C.ndim != 3 or C.shape[2] != 3:
        raise ValueError(f"coords must be (M, N, 3) or a list of (N, 3), got {C.shape}")
    m, n = C.shape[0], C.shape[1]
    if m == 0:
        return []

    labels: List[Any] = [0] * n if elements is None else list(elements)
    if len(labels) != n:
        raise ValueError(f"{len(labels)} labels were given for {n} atoms")

    en: Optional[np.ndarray] = None
    if energies is not None:
        en = np.asarray(list(energies), dtype=float).ravel()
        if en.size != m:
            raise ValueError(f"{en.size} energies were given for {m} poses")

    distance = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            d = symmetry_aware_rmsd(C[i], C[j], labels)
            distance[i, j] = distance[j, i] = d

    parent = list(range(m))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(m):
        for j in range(i + 1, m):
            if distance[i, j] <= cutoff:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    groups: Dict[int, List[int]] = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)

    clusters: List[Cluster] = []
    for members in groups.values():
        members.sort()
        if en is None:
            representative = members[0]
            best: Optional[float] = None
        else:
            representative = min(members, key=lambda k: (float(en[k]), k))
            best = float(en[representative])
        if len(members) > 1:
            values = [
                distance[members[a], members[b]]
                for a in range(len(members))
                for b in range(a + 1, len(members))
            ]
            mean = float(np.mean(values))
        else:
            mean = 0.0
        clusters.append(
            Cluster(
                index=0,
                members=members,
                representative=representative,
                best_energy=best,
                mean_rmsd=mean,
            )
        )

    clusters.sort(
        key=lambda c: (
            c.best_energy is None,
            0.0 if c.best_energy is None else c.best_energy,
            -len(c.members),
            c.representative,
        )
    )
    for i, cluster in enumerate(clusters):
        cluster.index = i
    return clusters


# ---------------------------------------------------------------------------
# E.2 -- interaction profiling
# ---------------------------------------------------------------------------


def _best_hbond_angle(donor: _Structure, d_idx: int, acceptor_point: np.ndarray):
    """Largest D-H...A angle in degrees, or ``None`` when it cannot be known."""
    d = donor.coords[d_idx]
    hydrogens = donor.hydrogens.get(d_idx, ())
    if hydrogens:
        best = -1.0
        for h_idx in hydrogens:
            h = donor.coords[h_idx]
            v1 = d - h
            v2 = acceptor_point - h
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos = float(np.dot(v1, v2) / (n1 * n2))
            best = max(best, math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
        return None if best < 0.0 else best
    if donor.has_hydrogens:
        # The structure has hydrogens but not on this donor: it cannot donate.
        return None
    # No hydrogens anywhere: estimate the missing H direction from the valence
    # geometry -- the hydrogen points away from the sum of the heavy bonds.
    heavy = [j for j in donor.adjacency[d_idx] if donor.atoms[j].element not in _HYDROGEN]
    if not heavy:
        return None
    direction = np.zeros(3)
    for j in heavy:
        v = d - donor.coords[j]
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            direction += v / norm
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    # The hydrogens sit opposite the heavy bonds, one bond length away.
    pseudo_h = d + direction / norm
    v1 = d - pseudo_h
    v2 = acceptor_point - pseudo_h
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def profile_interactions(
    receptor,
    ligand,
    *,
    hbond: float = 3.5,
    salt: float = 4.0,
    pi: float = 4.5,
    cation_pi: float = 5.0,
    hydrophobic: float = 4.0,
    clash_ratio: float = 0.75,
) -> List[Interaction]:
    """Every non-covalent interaction between `receptor` and `ligand`.

    The geometric criteria are exactly the ones required by E.2:

    ================  =====================================================
    ``hbond``         D...A <= 3.5 Å **and** the D-H...A angle >= 120°
    ``salt_bridge``   distance between the charged centres <= 4.0 Å
    ``pi_pi``         aromatic ring centroids <= 4.5 Å, split at 45° into
                      ``"face-to-face"`` (``subtype="face"``) and
                      ``"T-shaped"`` (``subtype="edge"``)
    ``cation_pi``     cationic centre to ring centroid <= 5.0 Å
    ``hydrophobic``   non-polar carbon to non-polar carbon <= 4.0 Å
    ``clash``         d <= 0.75 x (vdw_radius_1 + vdw_radius_2)
    ================  =====================================================

    The keyword arguments are those cutoffs in Å; `clash_ratio` is the fraction
    of the van der Waals contact distance below which two atoms clash.

    Both arguments accept an RDKit molecule or a sequence of Atom-like objects
    (see the module docstring); the workbench passes the latter. When the input
    carries no hydrogens at all -- a receptor straight out of a PDB, for
    instance -- the missing D-H direction is estimated from the donor's valence
    geometry, which is the only way to keep the 120° rule meaningful. A donor
    that carries hydrogens but none on the atom in question is skipped: an atom
    with no hydrogen cannot donate.

    Returns the interactions ordered by kind (H-bond, salt bridge, pi-pi,
    cation-pi, hydrophobic, clash) and then by distance. A donor hydrogen and
    its acceptor are closer than ``0.75 * (r1 + r2)`` by construction, so a pair
    already classified as a hydrogen bond is never also reported as a clash --
    the H-bond is the more informative label for the same geometry.
    """
    for name, value in (
        ("hbond", hbond),
        ("salt", salt),
        ("pi", pi),
        ("cation_pi", cation_pi),
        ("hydrophobic", hydrophobic),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if clash_ratio <= 0:
        raise ValueError(f"clash_ratio must be positive, got {clash_ratio}")

    rec = _structure(receptor)
    lig = _structure(ligand)
    out: List[Interaction] = []
    if not rec.atoms or not lig.atoms:
        return out

    receptor_donor, from_receptor = _hbonds(rec, lig, float(hbond), True)
    ligand_donor, from_ligand = _hbonds(lig, rec, float(hbond), False)
    out.extend(receptor_donor)
    out.extend(ligand_donor)

    # A donor hydrogen and its acceptor sit closer than 0.75 x (r1 + r2) by
    # construction, so the clash scan is told to ignore exactly those pairs: a
    # hydrogen bond is a favourable contact and must not be reported as a
    # steric collision.
    acceptered = from_receptor | from_ligand

    out.extend(_salt_bridges(rec, lig, float(salt)))
    out.extend(_pi_stacking(rec, lig, float(pi)))
    out.extend(_cation_pi_contacts(rec, lig, float(cation_pi)))
    out.extend(_hydrophobic_contacts(rec, lig, float(hydrophobic)))
    out.extend(_clashes(rec, lig, float(clash_ratio), acceptered))

    out.sort(key=lambda item: (_KIND_ORDER[item.kind], item.distance))
    # Cache the receptor residue on every interaction: the fingerprints and the
    # report both need it, and re-deriving it per consumer means re-normalising
    # the receptor's bond perception each time.
    for item in out:
        if item.residue is None and 0 <= item.a < len(rec.atoms):
            atom = rec.atoms[item.a]
            if atom.res_name:
                item.residue = (atom.res_name, atom.res_id, atom.chain)
    return out


def _repack(
    a_idx: int,
    b_idx: int,
    distance: float,
    kind: str,
    detail: str,
    subtype: str = "",
) -> Interaction:
    """Build an Interaction whose ``a``/``b`` always mean receptor/ligand.

    Callers hand the two indices over in receptor/ligand order, swapping them
    when the geometric side that produced the contact was the ligand.
    """
    return Interaction(
        kind=kind,
        a=int(a_idx),
        b=int(b_idx),
        distance=float(distance),
        detail=detail,
        subtype=subtype,
    )


def _hbonds(
    donor_side: _Structure,
    acceptor_side: _Structure,
    cutoff: float,
    donor_is_receptor: bool,
):
    """H-bonds with D...A <= `cutoff` and D-H...A >= 120 degrees.

    Returns ``(interactions, pairs)`` where `pairs` holds the
    ``(receptor index, ligand index)`` of every donor-hydrogen/acceptor couple,
    so the clash scan can skip them.
    """
    results: List[Interaction] = []
    pairs: set = set()
    ia, ib, dist = _pairs_between(
        donor_side, donor_side.donors, acceptor_side, acceptor_side.acceptors, cutoff
    )
    for d_idx, a_idx, d in zip(ia.tolist(), ib.tolist(), dist.tolist()):
        angle = _best_hbond_angle(donor_side, d_idx, acceptor_side.coords[a_idx])
        if angle is not None and angle < 120.0:
            continue
        detail = f"{donor_side.label(d_idx)}->{acceptor_side.label(a_idx)}"
        if donor_is_receptor:
            results.append(_repack(d_idx, a_idx, d, "hbond", detail))
            hydrogens = donor_side.hydrogens.get(d_idx, ())
            for h_idx in hydrogens or (d_idx,):
                pairs.add((int(h_idx), int(a_idx)))
        else:
            results.append(_repack(a_idx, d_idx, d, "hbond", detail))
            hydrogens = donor_side.hydrogens.get(d_idx, ())
            for h_idx in hydrogens or (d_idx,):
                pairs.add((int(a_idx), int(h_idx)))
    return results, pairs


def _salt_bridges(rec: _Structure, lig: _Structure, cutoff: float) -> List[Interaction]:
    results: List[Interaction] = []

    def scan(cation_side: _Structure, anion_side: _Structure, cation_is_receptor: bool):
        if not cation_side.cations or not anion_side.anions:
            return
        cat_centres = np.array([c for _, c, _ in cation_side.cations])
        an_centres = np.array([c for _, c, _ in anion_side.anions])
        diff = cat_centres[:, None, :] - an_centres[None, :, :]
        d = np.sqrt((diff * diff).sum(axis=-1))
        for i, j in zip(*np.nonzero(d <= cutoff)):
            cat_atoms, _, cat_label = cation_side.cations[int(i)]
            an_atoms, _, an_label = anion_side.anions[int(j)]
            centre_cat = cat_centres[int(i)]
            centre_an = an_centres[int(j)]
            a_idx = min(
                cat_atoms,
                key=lambda k: float(np.linalg.norm(cation_side.coords[k] - centre_an)),
            )
            b_idx = min(
                an_atoms,
                key=lambda k: float(np.linalg.norm(anion_side.coords[k] - centre_cat)),
            )
            detail = f"{cat_label}...{an_label}"
            if cation_is_receptor:
                results.append(_repack(a_idx, b_idx, d[i, j], "salt_bridge", detail))
            else:
                results.append(_repack(b_idx, a_idx, d[i, j], "salt_bridge", detail))

    scan(rec, lig, True)
    scan(lig, rec, False)
    return results


def _pi_stacking(rec: _Structure, lig: _Structure, cutoff: float) -> List[Interaction]:
    results: List[Interaction] = []
    if not rec.rings or not lig.rings:
        return results
    rec_geom = [_ring_geometry(rec, ring) for ring in rec.rings]
    lig_geom = [_ring_geometry(lig, ring) for ring in lig.rings]
    for ri, (rc, rn) in enumerate(rec_geom):
        for li, (lc, ln) in enumerate(lig_geom):
            d = float(np.linalg.norm(rc - lc))
            if d > cutoff:
                continue
            cos = abs(float(np.dot(rn, ln)))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if angle <= 45.0:
                detail, subtype = "face-to-face", "face"
            else:
                detail, subtype = "T-shaped", "edge"
            results.append(
                Interaction(
                    kind="pi_pi",
                    a=int(rec.rings[ri][0]),
                    b=int(lig.rings[li][0]),
                    distance=d,
                    detail=detail,
                    subtype=subtype,
                )
            )
    return results


def _cation_pi_contacts(
    rec: _Structure, lig: _Structure, cutoff: float
) -> List[Interaction]:
    results: List[Interaction] = []

    def scan(ring_side: _Structure, cation_side: _Structure, rings_are_receptor: bool):
        if not ring_side.rings or not cation_side.cations:
            return
        geom = [_ring_geometry(ring_side, ring) for ring in ring_side.rings]
        centres = np.array([c for _, c, _ in cation_side.cations])
        for ri, (centroid, _normal) in enumerate(geom):
            d = np.sqrt(((centres - centroid) ** 2).sum(axis=1))
            for ci in np.nonzero(d <= cutoff)[0].tolist():
                atoms, _, cat_label = cation_side.cations[ci]
                ring_idx = int(ring_side.rings[ri][0])
                a_idx = min(
                    atoms,
                    key=lambda k: float(np.linalg.norm(cation_side.coords[k] - centroid)),
                )
                detail = f"{cat_label}->pi({ring_side.label(ring_idx)})"
                if rings_are_receptor:
                    results.append(_repack(ring_idx, a_idx, d[ci], "cation_pi", detail))
                else:
                    results.append(_repack(a_idx, ring_idx, d[ci], "cation_pi", detail))

    scan(rec, lig, True)
    scan(lig, rec, False)
    return results


def _hydrophobic_contacts(
    rec: _Structure, lig: _Structure, cutoff: float
) -> List[Interaction]:
    results: List[Interaction] = []
    ia, ib, dist = _pairs_between(
        rec, rec.hydrophobic_carbons, lig, lig.hydrophobic_carbons, cutoff
    )
    for i, j, d in zip(ia.tolist(), ib.tolist(), dist.tolist()):
        results.append(
            Interaction(
                kind="hydrophobic",
                a=int(i),
                b=int(j),
                distance=float(d),
                detail=f"{rec.label(i)}...{lig.label(j)}",
            )
        )
    return results


def _clashes(
    rec: _Structure, lig: _Structure, clash_ratio: float, hydrogen_bonded: set
) -> List[Interaction]:
    results: List[Interaction] = []
    radii_rec = np.array([VDW_RADII.get(a.element, 1.70) for a in rec.atoms], dtype=float)
    radii_lig = np.array([VDW_RADII.get(a.element, 1.70) for a in lig.atoms], dtype=float)
    widest = float(clash_ratio * (radii_rec.max() + radii_lig.max()))
    ia, ib, dist = _pairs_between(
        rec, range(len(rec.atoms)), lig, range(len(lig.atoms)), widest
    )
    for i, j, d in zip(ia.tolist(), ib.tolist(), dist.tolist()):
        if (i, j) in hydrogen_bonded:
            continue
        if d <= clash_ratio * (radii_rec[i] + radii_lig[j]):
            results.append(
                Interaction(
                    kind="clash",
                    a=int(i),
                    b=int(j),
                    distance=float(d),
                    detail=f"{rec.label(i)}...{lig.label(j)}",
                )
            )
    return results


def interaction_summary(interactions, receptor, ligand, *, limit: int = 6) -> str:
    """The "key interacting residues" column: ``"ASP189, GLY216, TRP215"``.

    Residues are counted once per interaction and the `limit` most-contacted
    ones are kept; the survivors are then printed in alphabetical order of
    residue name and then residue number, which is the order the requirements
    show (``"ASP123, GLY45"``, ``"ASP189, GLY216, TRP215"``) and is stable
    across runs. Only the **receptor** side is reported: the ligand's own
    residue label (``LIG1``) is noise in a results table, and clashes are
    excluded because a clash is not an interaction.

    `ligand` is accepted for interface compatibility and used to validate that
    every interaction's ligand index exists.
    """
    if limit <= 0:
        return ""
    rec = _structure(receptor)
    lig = _structure(ligand)
    counts: Dict[Tuple[int, str, str], int] = {}
    for item in interactions:
        if item.kind == "clash":
            continue
        if not 0 <= item.a < len(rec.atoms) or not 0 <= item.b < len(lig.atoms):
            raise ValueError(
                f"interaction ({item.a}, {item.b}) is out of range for "
                f"{len(rec.atoms)} receptor and {len(lig.atoms)} ligand atoms"
            )
        atom = rec.atoms[item.a]
        if not atom.res_name:
            continue
        key = (atom.res_name, atom.res_id, atom.chain)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    selected = sorted(key for key, _ in ranked[: int(limit)])
    return ", ".join(f"{name}{res_id}" for name, res_id, _chain in selected)


# ---------------------------------------------------------------------------
# Interaction fingerprints (E.2 output -> a fixed-length vector per pose)
# ---------------------------------------------------------------------------


def _water_indices(structure: _Structure, names=WATER_RESNAMES) -> List[int]:
    """Indices of the atoms whose residue name is a water."""
    return [
        i
        for i, atom in enumerate(structure.atoms)
        if atom.res_name.strip().upper() in names
    ]


def water_mediated_contacts(
    receptor,
    ligand,
    *,
    cutoff: float = 3.5,
    water_names=WATER_RESNAMES,
) -> List[Interaction]:
    """Water bridges: a water that H-bonds the ligand *and* the receptor.

    The geometric criterion is deliberately the simple, checkable one: a water
    oxygen within `cutoff` Å of a polar ligand atom (a donor or an acceptor, as
    :func:`profile_interactions` classifies them) **and** within `cutoff` Å of a
    polar receptor atom.  No angle is tested and the water's own hydrogens are
    not required, because a crystallographic water is usually an oxygen only --
    the two-leg distance criterion is the standard first-pass definition and it
    is stated here rather than implied.

    Returns one :class:`Interaction` per (receptor atom, water, ligand atom)
    triple, with ``kind="water_bridge"``, ``a`` the closest polar receptor atom,
    ``b`` the closest polar ligand atom, ``distance`` the **shorter** of the two
    legs, and ``detail`` naming all three (``"ASP189:OD2...HOH301...LIG1:N1"``).
    The waters must be part of the receptor structure handed in; a receptor with
    no waters yields an empty list, which is the correct answer for a structure
    whose solvent was stripped.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")

    rec = _structure(receptor)
    lig = _structure(ligand)
    if not rec.atoms or not lig.atoms:
        return []

    water_atoms = set(_water_indices(rec, water_names))
    oxygens = [i for i in sorted(water_atoms) if rec.atoms[i].element.upper() == "O"]
    if not oxygens:
        return []

    receptor_polar = sorted(
        (set(rec.donors) | set(rec.acceptors)) - water_atoms
    )
    ligand_polar = sorted(set(lig.donors) | set(lig.acceptors))
    if not receptor_polar or not ligand_polar:
        return []

    receptor_points = rec.coords[receptor_polar]
    ligand_points = lig.coords[ligand_polar]

    found: List[Interaction] = []
    for water in oxygens:
        centre = rec.coords[water]
        to_receptor = np.linalg.norm(receptor_points - centre, axis=1)
        to_ligand = np.linalg.norm(ligand_points - centre, axis=1)
        if to_receptor.size == 0 or to_ligand.size == 0:
            continue
        receptor_best = int(np.argmin(to_receptor))
        ligand_best = int(np.argmin(to_ligand))
        if to_receptor[receptor_best] > cutoff or to_ligand[ligand_best] > cutoff:
            continue
        a = receptor_polar[receptor_best]
        b = ligand_polar[ligand_best]
        detail = f"{rec.label(a)}...{rec.label(water)}...{lig.label(b)}"
        residue_atom = rec.atoms[a]
        found.append(
            Interaction(
                kind=WATER_KIND,
                a=int(a),
                b=int(b),
                distance=float(min(to_receptor[receptor_best], to_ligand[ligand_best])),
                detail=detail,
                residue=(
                    (residue_atom.res_name, residue_atom.res_id, residue_atom.chain)
                    if residue_atom.res_name
                    else None
                ),
            )
        )
    found.sort(key=lambda item: (item.a, item.b, item.distance))
    return found


@dataclass(frozen=True)
class FingerprintKey:
    """One feature of an interaction fingerprint: a residue and a contact type.

    ``hydrophobic``/``hbond``/... contacts with ``SER195`` are different
    features even though they come from the same residue, which is what makes a
    fingerprint comparable between poses: two poses that both touch ASP189 but
    one by a salt bridge and the other by a hydrophobic contact are not the same
    binding mode.
    """

    res_name: str
    res_id: int
    chain: str
    kind: str

    @property
    def label(self) -> str:
        """``"ASP189:hbond"``."""
        residue = f"{self.res_name}{self.res_id}" if self.res_name else f"#{self.res_id}"
        return f"{residue}:{self.kind}"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


@dataclass
class FingerprintSchema:
    """The fixed feature space a set of fingerprints is expressed in.

    Two ways to build one:

    * :meth:`observed` -- the union of the features a set of poses actually
      presents.  Compact, and the right choice for comparing the poses of one
      run against each other.
    * :meth:`from_receptor` -- every residue the receptor has, crossed with
      every interaction kind.  Truly fixed: the same schema can then be reused
      for a different ligand, or for a second run on the same receptor, which is
      what makes a cross-run fingerprint comparison possible.
    """

    keys: Tuple[FingerprintKey, ...]
    #: ``key -> column``, built once so encoding many poses is O(features).
    index: Dict[FingerprintKey, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.keys = tuple(self.keys)
        if not self.index:
            self.index = {key: i for i, key in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def column(self, key: FingerprintKey) -> int:
        """The column of `key`, or ``-1`` when it is not part of this schema."""
        return self.index.get(key, -1)

    @classmethod
    def observed(
        cls,
        interactions,
        *,
        kinds: Sequence[str] = INTERACTION_KINDS,
        include_water: bool = True,
    ) -> "FingerprintSchema":
        """The union of the features present in `interactions`.

        `interactions` is either a single interaction list or an iterable of
        lists (one per pose).  The keys are sorted, so the schema -- and
        therefore every fingerprint built from it -- is deterministic.
        """
        allowed = set(kinds) | ({WATER_KIND} if include_water else set())
        keys = set()
        for group in _as_groups(interactions):
            for item in group:
                key = _fingerprint_key(item)
                if key is not None and key.kind in allowed:
                    keys.add(key)
        return cls(keys=tuple(sorted(keys, key=_key_sort)))

    @classmethod
    def from_receptor(
        cls,
        receptor,
        *,
        kinds: Sequence[str] = INTERACTION_KINDS,
        include_water: bool = False,
    ) -> "FingerprintSchema":
        """Every residue of `receptor` crossed with every interaction kind.

        The result is independent of the ligand, so fingerprints from different
        ligands or different runs land in the same vector space.  Waters are
        excluded unless `include_water` is set, and the ``water_bridge`` feature
        is a property of a *residue plus a bridging water*, so it is only added
        when asked for.
        """
        structure = _structure(receptor)
        kinds = tuple(kinds) + ((WATER_KIND,) if include_water else ())
        residues = {}
        waters = set(_water_indices(structure))
        for i, atom in enumerate(structure.atoms):
            if not atom.res_name or i in waters:
                continue
            residues[(atom.chain, atom.res_id, atom.res_name)] = None
        keys = [
            FingerprintKey(chain=chain, res_id=res_id, res_name=name, kind=kind)
            for (chain, res_id, name) in residues
            for kind in kinds
        ]
        return cls(keys=tuple(sorted(keys, key=_key_sort)))


def _key_sort(key: FingerprintKey):
    return (key.chain, key.res_id, key.res_name, _KIND_ORDER.get(key.kind, 99), key.kind)


def _as_groups(interactions) -> List[List[Interaction]]:
    """Normalise ``[interactions]`` / ``[[...], [...]]`` into a list of lists."""
    if interactions is None:
        return []
    items = list(interactions)
    if not items:
        return []
    if isinstance(items[0], Interaction):
        return [items]
    return [list(group) for group in items]


def _fingerprint_key(item: Interaction) -> Optional[FingerprintKey]:
    """The feature an interaction contributes, or ``None`` for a non-feature."""
    if item.kind == "clash":
        return None
    residue = getattr(item, "residue", None)
    if residue is None:
        return None
    return FingerprintKey(
        res_name=residue[0], res_id=int(residue[1]), chain=str(residue[2]), kind=item.kind
    )


@dataclass
class InteractionFingerprint:
    """A pose's interactions as a fixed-length count vector.

    Attributes
    ----------
    schema
        The feature space the vector is expressed in.
    counts
        ``(K,)`` integer counts: how many contacts of each (residue, type) the
        pose makes.  Counts rather than bits, because three hydrogen bonds to
        one residue are not the same as one; :attr:`bits` gives the binary view.
    labels
        ``(K,)`` object array of the human-readable feature labels, in column
        order, so a table can be written without consulting the schema.
    """

    schema: FingerprintSchema
    counts: np.ndarray

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=float).ravel()
        if self.counts.shape[0] != len(self.schema):
            raise ValueError(
                f"the fingerprint has {self.counts.shape[0]} values for a "
                f"{len(self.schema)}-feature schema"
            )

    def __len__(self) -> int:
        return int(self.counts.shape[0])

    @property
    def bits(self) -> np.ndarray:
        """The binary view: 1 where the pose makes at least one such contact."""
        return (self.counts > 0).astype(int)

    @property
    def labels(self) -> np.ndarray:
        return np.array([key.label for key in self.schema.keys], dtype=object)

    def present(self) -> List[FingerprintKey]:
        """The features this pose actually presents, in schema order."""
        return [key for key, value in zip(self.schema.keys, self.counts) if value > 0]

    def to_dict(self, *, include_empty: bool = False) -> Dict[str, Any]:
        """``{"labels": [...], "counts": [...], "bits": [...]}``.

        With `include_empty` every column is written; without it only the
        features the pose presents, which keeps a JSONL file of a large screen
        readable.
        """
        pairs = [
            (key.label, float(count))
            for key, count in zip(self.schema.keys, self.counts)
            if include_empty or count > 0
        ]
        return {
            "labels": [label for label, _ in pairs],
            "counts": [count for _, count in pairs],
            "bits": [1 if count > 0 else 0 for _, count in pairs],
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"InteractionFingerprint({len(self)} features, {int((self.counts > 0).sum())} present)"


def interaction_fingerprint(
    interactions,
    schema: FingerprintSchema,
    *,
    receptor=None,
    ligand=None,
) -> InteractionFingerprint:
    """Encode one pose's interactions into `schema`.

    `interactions` is the output of :func:`profile_interactions` for a single
    pose (an iterable of lists is accepted and flattened only when it holds one
    pose).  `receptor`/`ligand` are needed to resolve a contact back to its
    residue when the interaction objects did not come from
    :func:`profile_interactions`; the normal path does not need them because the
    profiler attaches the residue key to every interaction it returns.
    """
    groups = _as_groups(interactions)
    if len(groups) > 1:
        raise ValueError(
            "interaction_fingerprint encodes one pose; pass a single "
            "interaction list, or use pose_fingerprints for a set"
        )
    items = groups[0] if groups else []
    counts = np.zeros(len(schema), dtype=float)
    for item in items:
        key = _fingerprint_key(item)
        if key is None and receptor is not None:
            # An interaction built by hand carries no residue; resolve it from
            # the receptor rather than dropping the contact silently.
            key = _resolve_key(item, receptor, ligand)
        if key is None:
            continue
        column = schema.column(key)
        if column >= 0:
            counts[column] += 1.0
    return InteractionFingerprint(schema=schema, counts=counts)


def _resolve_key(item, receptor, ligand) -> Optional[FingerprintKey]:
    """The feature of an interaction that carries no residue key of its own."""
    try:
        rec = _structure(receptor)
    except Exception:  # pragma: no cover - a receptor that cannot be read
        return None
    if not 0 <= item.a < len(rec.atoms):
        return None
    atom = rec.atoms[item.a]
    if not atom.res_name:
        return None
    return FingerprintKey(atom.res_name, atom.res_id, atom.chain, item.kind)


def fingerprint_similarity(a, b, *, metric: str = "tanimoto") -> float:
    """Similarity of two fingerprints.

    ``"tanimoto"`` (the default)
        ``|A n B| / |A u B|`` over the **bit** vectors -- the standard
        fingerprint Tanimoto.  ``1.0`` when the two poses make the same
        contacts, ``0.0`` when they share none, ``nan`` when neither pose makes
        any contact at all (the empty-versus-empty case is undefined, not 1.0).
    ``"cosine"``
        The cosine of the **count** vectors, so a pose that makes three
        hydrogen bonds to a residue is closer to one that makes two than to one
        that makes none.
    ``"dice"``
        ``2|A n B| / (|A| + |B|)`` over the bits -- Dice/Sørensen.
    """
    first = np.asarray(getattr(a, "counts", a), dtype=float).ravel()
    second = np.asarray(getattr(b, "counts", b), dtype=float).ravel()
    if first.shape != second.shape:
        raise ValueError(
            f"the two fingerprints must have the same length, got "
            f"{first.shape[0]} and {second.shape[0]}"
        )
    if metric == "cosine":
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 0.0:
            return float("nan")
        return float(np.dot(first, second) / denominator)
    if metric not in ("tanimoto", "dice"):
        raise ValueError(
            f"unknown fingerprint metric {metric!r}; use 'tanimoto', 'cosine' or 'dice'"
        )
    bits_a = first > 0
    bits_b = second > 0
    shared = int(np.logical_and(bits_a, bits_b).sum())
    union = int(np.logical_or(bits_a, bits_b).sum())
    if metric == "dice":
        total = int(bits_a.sum()) + int(bits_b.sum())
        return float("nan") if total == 0 else 2.0 * shared / total
    if union == 0:
        return float("nan")
    return shared / union


def similarity_matrix(fingerprints, *, metric: str = "tanimoto") -> np.ndarray:
    """The ``(M, M)`` pairwise similarity matrix, diagonal exactly 1.0.

    The diagonal is set to 1.0 rather than computed, so a pose that makes no
    contact at all still compares to itself as identical while its off-diagonal
    Tanimoto stays ``nan``.
    """
    items = [np.asarray(getattr(fp, "counts", fp), dtype=float).ravel() for fp in fingerprints]
    count = len(items)
    out = np.full((count, count), np.nan, dtype=float)
    for i in range(count):
        out[i, i] = 1.0
        for j in range(i + 1, count):
            value = fingerprint_similarity(items[i], items[j], metric=metric)
            out[i, j] = out[j, i] = value
    return out


@dataclass
class PharmacophoreSummary:
    """The contacts that recur across the best-ranked poses.

    A feature that appears in every one of the top poses is the reproducible
    part of the binding mode -- the part worth designing against -- while a
    feature that appears in one pose only is as likely to be a scoring artefact
    as a real contact.
    """

    keys: List[FingerprintKey] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    frequency: List[float] = field(default_factory=list)
    n_poses: int = 0
    interaction_types: Dict[str, int] = field(default_factory=dict)

    def table(self, limit: Optional[int] = None) -> str:
        """A plain-text table: feature, how many poses present it, frequency."""
        rows = list(zip(self.keys, self.counts, self.frequency))
        if limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            return "no recurring interaction (the selected poses share no feature)"
        width = max(len(key.label) for key, _c, _f in rows)
        lines = [f"{'feature'.ljust(width)}  poses  fraction", "-" * (width + 15)]
        for key, count, frequency in rows:
            lines.append(
                f"{key.label.ljust(width)}  {count:>5d}  {frequency:>8.2f}"
            )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_poses": self.n_poses,
            "features": [
                {"label": key.label, "residue": f"{key.res_name}{key.res_id}",
                 "chain": key.chain, "kind": key.kind, "poses": count,
                 "frequency": frequency}
                for key, count, frequency in zip(self.keys, self.counts, self.frequency)
            ],
            "interaction_types": dict(self.interaction_types),
        }


def pharmacophore_summary(fingerprints, *, min_frequency: float = 0.5) -> PharmacophoreSummary:
    """Which (residue, interaction type) features recur across `fingerprints`.

    Parameters
    ----------
    fingerprints
        The fingerprints of the poses to summarise -- normally the top poses of
        a consensus ranking, so that "recurring" means "recurring among the
        poses the scoring function actually prefers".
    min_frequency
        Keep only features present in at least this fraction of the poses.
        ``0.5`` (the default) keeps what a majority of the poses agree on.

    Returns
    -------
    :class:`PharmacophoreSummary`, ordered by descending frequency and then by
    the feature order, so it is deterministic.
    """
    items = [np.asarray(getattr(fp, "counts", fp), dtype=float).ravel() for fp in fingerprints]
    total = len(items)
    summary = PharmacophoreSummary(n_poses=total)
    if total == 0:
        return summary

    ordered = list(fingerprints)
    schema = getattr(ordered[0], "schema", None)
    if schema is None:
        raise TypeError("pharmacophore_summary needs InteractionFingerprint objects")

    width = len(schema)
    if any(item.shape[0] != width for item in items):
        raise ValueError("the fingerprints do not share one schema")

    stack = np.vstack(items)
    present = (stack > 0).sum(axis=0)
    threshold = float(min_frequency) * total
    order = sorted(
        (column for column in range(width) if present[column] >= threshold),
        key=lambda column: (-int(present[column]), _key_sort(schema.keys[column])),
    )
    for column in order:
        summary.keys.append(schema.keys[column])
        summary.counts.append(int(present[column]))
        summary.frequency.append(float(present[column]) / total)
    for key in summary.keys:
        summary.interaction_types[key.kind] = summary.interaction_types.get(key.kind, 0) + 1
    return summary


@dataclass
class FingerprintSet:
    """The fingerprints of a whole pose set, plus the matrices built from them."""

    schema: FingerprintSchema
    fingerprints: List[InteractionFingerprint]
    #: The profiled interactions per pose, in the same order.
    interactions: List[List[Interaction]] = field(default_factory=list)
    #: Water bridges per pose, when the scan was asked for.
    water_bridges: List[List[Interaction]] = field(default_factory=list)
    receptor: Any = None
    ligands: List[Any] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fingerprints)

    @property
    def matrix(self) -> np.ndarray:
        """The ``(M, K)`` count matrix, one row per pose."""
        if not self.fingerprints:
            return np.zeros((0, len(self.schema)), dtype=float)
        return np.vstack([fp.counts for fp in self.fingerprints])

    @property
    def bits(self) -> np.ndarray:
        """The ``(M, K)`` binary matrix."""
        return (self.matrix > 0).astype(int)

    def similarity(self, *, metric: str = "tanimoto") -> np.ndarray:
        """The ``(M, M)`` similarity matrix across the poses."""
        return similarity_matrix(self.fingerprints, metric=metric)

    def pharmacophore(self, *, top: Optional[int] = None, min_frequency: float = 0.5):
        """The recurring features, optionally restricted to the first `top` poses."""
        chosen = self.fingerprints if top is None else self.fingerprints[: int(top)]
        return pharmacophore_summary(chosen, min_frequency=min_frequency)

    def labels(self) -> List[str]:
        """The schema labels, in column order (a header row for the matrix)."""
        return [key.label for key in self.schema.keys]


def pose_fingerprints(
    receptor,
    ligands,
    *,
    include_water: bool = True,
    kinds: Sequence[str] = INTERACTION_KINDS,
    schema: Optional[FingerprintSchema] = None,
    water_cutoff: float = 3.5,
    **cutoffs,
) -> FingerprintSet:
    """Profile every ligand against one receptor and fingerprint the results.

    Parameters
    ----------
    receptor
        The receptor, in any form :func:`profile_interactions` accepts.  Waters
        in it are **not** reported as interacting residues: a water is not a
        pharmacophore feature.  They are used for the water-bridge scan instead
        (pass ``include_water=False`` to skip it).
    ligands
        One entry per pose: an RDKit molecule or a sequence of Atom-like
        objects, in any order.
    kinds
        The feature kinds the schema spans.
    schema
        An explicit feature space, e.g. :meth:`FingerprintSchema.from_receptor`
        when fingerprints from several ligands must be comparable.  By default
        the union of the features these poses present is used.
    **cutoffs
        Passed through to :func:`profile_interactions` (``hbond``, ``salt``,
        ``pi``, ``cation_pi``, ``hydrophobic``, ``clash_ratio``).

    Returns
    -------
    :class:`FingerprintSet`.  Clashes are never part of a fingerprint: a clash
    is a defect, not a contact, and counting one would make two poses look
    similar because both are bad.
    """
    receptor_structure = _structure(receptor)
    water_atoms = set(_water_indices(receptor_structure))

    interactions_per_pose: List[List[Interaction]] = []
    bridges_per_pose: List[List[Interaction]] = []
    for ligand in ligands:
        ligand_structure = _structure(ligand)
        contacts = [
            item
            for item in profile_interactions(
                receptor_structure, ligand_structure, **cutoffs
            )
            if item.kind != "clash" and item.a not in water_atoms
        ]
        interactions_per_pose.append(contacts)
        if include_water:
            bridges_per_pose.append(
                water_mediated_contacts(
                    receptor_structure, ligand_structure, cutoff=water_cutoff
                )
            )
        else:
            bridges_per_pose.append([])

    if schema is None:
        schema = FingerprintSchema.observed(
            interactions_per_pose + bridges_per_pose,
            kinds=tuple(kinds),
            include_water=include_water,
        )

    fingerprints = [
        interaction_fingerprint(list(contacts) + list(bridges), schema)
        for contacts, bridges in zip(interactions_per_pose, bridges_per_pose)
    ]
    return FingerprintSet(
        schema=schema,
        fingerprints=fingerprints,
        interactions=interactions_per_pose,
        water_bridges=bridges_per_pose,
        receptor=receptor,
        ligands=list(ligands),
    )


# ---------------------------------------------------------------------------
# E.3 -- 2-D topology diagram and pose interpolation
# ---------------------------------------------------------------------------


def _escape(text: Any) -> str:
    """XML-escape a string for use in SVG text or an attribute."""
    return _saxutils.escape(str(text), {'"': "&quot;", "'": "&apos;"})


def _project(coords: np.ndarray, heavy: Sequence[int]):
    """Project 3-D coordinates onto the plane that shows the most of them.

    The two dominant principal components of the heavy atoms are used as the
    drawing plane, so a flat aromatic ligand keeps its shape instead of being
    projected onto an arbitrary axis.
    """
    idx = np.asarray(list(heavy), dtype=np.int64)
    if idx.size == 0:
        idx = np.arange(coords.shape[0], dtype=np.int64)
    pts = coords[idx]
    centre = pts.mean(axis=0)
    centred = pts - centre
    if len(pts) >= 3 and float(np.linalg.norm(centred)) > 1e-9:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        basis = np.asarray(vt[:2], dtype=float)
    else:
        basis = np.eye(3)[:2]
    return (coords - centre) @ basis.T, basis, centre


def interaction_diagram_svg(
    receptor,
    ligand,
    interactions,
    *,
    path=None,
    title: str = "",
    width: int = 900,
    height: int = 700,
) -> str:
    """A LigPlot/PoseView-style 2-D topology diagram of the contacts.

    The ligand is drawn in the middle: its heavy atoms are projected onto the
    plane that shows the most of the molecule (the two dominant principal
    components, so a flat ligand keeps its shape), bonds are drawn between
    perceived neighbours, and every interacting residue is placed on a ring
    around the ligand in the direction the residue actually lies in 3-D.
    Contacts are dashed lines coloured by interaction type
    (:data:`INTERACTION_COLORS`); hydrophobic contacts are drawn as arcs, as
    LigPlot does. A legend names every type.

    The result is a self-contained SVG document, returned as a string and, when
    `path` is given, also written there. It is plain string building: no
    RDKit, no plotting library, no optional dependency. The document is
    well-formed XML in the SVG namespace and opens in any browser.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")

    rec = _structure(receptor)
    lig = _structure(ligand)
    heavy = [i for i, a in enumerate(lig.atoms) if a.element not in _HYDROGEN]
    if not heavy:
        heavy = list(range(len(lig.atoms)))

    if lig.atoms:
        projected, basis, lig_centre = _project(lig.coords, heavy)
    else:  # pragma: no cover - an empty ligand has nothing to draw
        projected = np.zeros((0, 2))
        basis = np.eye(3)[:2]
        lig_centre = np.zeros(3)

    cx, cy = width / 2.0, height / 2.0
    if heavy:
        pts = projected[np.asarray(heavy, dtype=np.int64)]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
    else:
        lo = hi = np.zeros(2)
    extent = np.maximum(hi - lo, 1e-6)
    scale = min(width * 0.68 / extent[0], height * 0.60 / extent[1], 45.0)
    mid = (lo + hi) / 2.0

    def to_canvas(xy) -> Tuple[float, float]:
        local = (np.asarray(xy, dtype=float) - mid) * scale
        return cx + float(local[0]), cy - float(local[1])

    def project_point(point) -> Tuple[float, float]:
        return to_canvas(basis @ (np.asarray(point, dtype=float) - lig_centre))

    atom_xy = [to_canvas(p) for p in projected]

    # -- group the contacts by receptor residue ----------------------------
    node_keys: List[Tuple[int, str, str]] = []
    node_anchor: Dict[Tuple[int, str, str], int] = {}
    valid: List[Interaction] = []
    for item in interactions:
        if not 0 <= item.a < len(rec.atoms) or not 0 <= item.b < len(lig.atoms):
            continue
        atom = rec.atoms[item.a]
        key = (atom.res_id, atom.res_name or "LIG", atom.chain)
        if key not in node_anchor:
            node_anchor[key] = item.a
            node_keys.append(key)
        valid.append(item)

    n_nodes = len(node_keys)
    natural: List[float] = []
    for index, key in enumerate(node_keys):
        planar = basis @ (rec.coords[node_anchor[key]] - lig_centre)
        if float(np.linalg.norm(planar)) < 1e-6:
            natural.append(2.0 * math.pi * index / max(n_nodes, 1))
        else:
            natural.append(math.atan2(-float(planar[1]), float(planar[0])))

    # Keep the circular order but enforce a minimum angular separation so two
    # residues on the same side of the ligand do not overlap.
    separation = min(2.0 * math.pi / max(n_nodes, 1), 0.6)
    angles = list(natural)
    ordered = sorted(range(n_nodes), key=lambda k: angles[k])
    previous = None
    for k in ordered:
        value = angles[k]
        if previous is not None and value < previous + separation:
            value = previous + separation
        angles[k] = value
        previous = value
    if n_nodes:
        shift = float(np.mean([angles[k] - natural[k] for k in range(n_nodes)]))
        angles = [a - shift for a in angles]

    rx, ry = width * 0.40, height * 0.38
    node_xy: Dict[Tuple[int, str, str], Tuple[float, float]] = {}
    for k, key in enumerate(node_keys):
        node_xy[key] = (cx + rx * math.cos(angles[k]), cy + ry * math.sin(angles[k]))

    # -- assemble the document ---------------------------------------------
    body: List[str] = []
    body.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>'
    )
    heading = title or "Ligand interaction diagram"
    body.append(
        f'<text x="{cx:.1f}" y="34" text-anchor="middle" '
        'font-family="Helvetica,Arial,sans-serif" font-size="20" '
        f'font-weight="bold" fill="#263238">{_escape(heading)}</text>'
    )

    # Ligand skeleton.
    for i, j in lig.bonds:
        if i >= len(atom_xy) or j >= len(atom_xy):
            continue
        if lig.atoms[i].element in _HYDROGEN or lig.atoms[j].element in _HYDROGEN:
            continue
        x1, y1 = atom_xy[i]
        x2, y2 = atom_xy[j]
        body.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#546e7a" stroke-width="2.5" stroke-linecap="round"/>'
        )

    for i in heavy:
        x, y = atom_xy[i]
        element = lig.atoms[i].element
        fill = _ELEMENT_FILL.get(element, "#b39ddb")
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{fill}" '
            'stroke="#37474f" stroke-width="1.2"/>'
        )
        body.append(
            f'<text x="{x:.1f}" y="{y + 3.4:.1f}" text-anchor="middle" '
            'font-family="Helvetica,Arial,sans-serif" font-size="9" '
            f'fill="#102027">{_escape(element)}</text>'
        )

    def ligand_anchor(item: Interaction) -> Tuple[float, float]:
        if item.kind in ("pi_pi", "cation_pi"):
            for ring in lig.rings:
                if ring and int(ring[0]) == item.b:
                    return project_point(lig.coords[list(ring)].mean(axis=0))
        return atom_xy[item.b]

    # Residue nodes.
    for key in node_keys:
        x, y = node_xy[key]
        label = f"{key[1]}{key[0]}" if key[1] != "LIG" else key[1]
        box_w = 16.0 + 7.0 * len(label)
        body.append(
            f'<rect x="{x - box_w / 2:.1f}" y="{y - 13:.1f}" width="{box_w:.1f}" '
            'height="26" rx="13" ry="13" fill="#eceff1" stroke="#90a4ae" '
            'stroke-width="1.4"/>'
        )
        body.append(
            f'<text x="{x:.1f}" y="{y + 4.5:.1f}" text-anchor="middle" '
            'font-family="Helvetica,Arial,sans-serif" font-size="13" '
            f'fill="#263238">{_escape(label)}</text>'
        )

    # Contacts.
    for item in valid:
        node = (rec.atoms[item.a].res_id, rec.atoms[item.a].res_name or "LIG",
                rec.atoms[item.a].chain)
        nx, ny = node_xy[node]
        lx, ly = ligand_anchor(item)
        colour = INTERACTION_COLORS.get(item.kind, "#607d8b")
        if item.kind == "hydrophobic":
            # LigPlot draws hydrophobic contacts as an arc around the ligand
            # atom, opening towards the residue.
            direction = math.atan2(ny - ly, nx - lx)
            radius = 17.0
            a1, a2 = direction - 0.85, direction + 0.85
            x1 = lx + radius * math.cos(a1)
            y1 = ly + radius * math.sin(a1)
            x2 = lx + radius * math.cos(a2)
            y2 = ly + radius * math.sin(a2)
            body.append(
                f'<path d="M {x1:.1f} {y1:.1f} A {radius:.1f} {radius:.1f} 0 0 1 '
                f'{x2:.1f} {y2:.1f}" fill="none" stroke="{colour}" '
                'stroke-width="2.4" stroke-linecap="round"/>'
            )
            body.append(
                f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" '
                f'stroke="{colour}" stroke-width="1" stroke-dasharray="1 6" '
                'opacity="0.65"/>'
            )
            continue
        dx, dy = nx - lx, ny - ly
        length = math.hypot(dx, dy) or 1.0
        stop = 16.0
        ex = nx - dx / length * stop
        ey = ny - dy / length * stop
        body.append(
            f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            f'stroke="{colour}" stroke-width="2" stroke-dasharray="7 5" '
            'stroke-linecap="round"/>'
        )

    # Legend.
    kinds = list(INTERACTION_KINDS) + ["clash", WATER_KIND]
    legend_w, legend_h = 236.0, 34.0 + 22.0 * len(kinds)
    body.append(
        f'<rect x="16" y="16" width="{legend_w}" height="{legend_h}" rx="8" ry="8" '
        'fill="#fafafa" stroke="#b0bec5" stroke-width="1"/>'
    )
    body.append(
        '<text x="30" y="38" font-family="Helvetica,Arial,sans-serif" '
        'font-size="13" font-weight="bold" fill="#37474f">Legend</text>'
    )
    for k, kind in enumerate(kinds):
        y = 62 + 22 * k
        body.append(
            f'<line x1="30" y1="{y}" x2="60" y2="{y}" '
            f'stroke="{INTERACTION_COLORS[kind]}" stroke-width="2.4" '
            'stroke-dasharray="7 5"/>'
        )
        body.append(
            f'<text x="70" y="{y + 4.5}" font-family="Helvetica,Arial,sans-serif" '
            f'font-size="12" fill="#37474f">{_escape(INTERACTION_LABELS[kind])}</text>'
        )

    counts = [
        (kind, sum(1 for it in valid if it.kind == kind))
        for kind in kinds
        if any(it.kind == kind for it in valid)
    ]
    footer = ", ".join(f"{n} {INTERACTION_LABELS[kind].lower()}" for kind, n in counts)
    if not footer:
        footer = "no contacts within the criteria"
    body.append(
        f'<text x="{cx:.1f}" y="{height - 18}" text-anchor="middle" '
        'font-family="Helvetica,Arial,sans-serif" font-size="12" '
        f'fill="#546e7a">{_escape(footer)}</text>'
    )

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )
    if path is not None:
        from pathlib import Path

        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document, encoding="utf-8")
    return document


def interpolate_coords(a, b, t) -> np.ndarray:
    """Linear interpolation between two poses, for the conformation player.

    ``t`` is clamped to ``[0, 1]``: ``t=0`` returns `a`, ``t=1`` returns `b` and
    ``t=0.5`` the midpoint. The interpolation is linear in Cartesian space --
    documented, and enough for playback; the caller can ease ``t`` (for example
    with a smoothstep) if a softer start and finish is wanted. Both inputs must
    be ``(N, 3)`` arrays with the same shape.

    The result is a new array; neither input is modified.
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.shape != B.shape:
        raise ValueError(
            f"the two poses must have the same shape, got {A.shape} and {B.shape}"
        )
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N, 3), got {A.shape}")
    fraction = float(np.clip(float(t), 0.0, 1.0))
    return (1.0 - fraction) * A + fraction * B

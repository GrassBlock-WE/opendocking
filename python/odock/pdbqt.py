# SPDX-License-Identifier: GPL-3.0-or-later
"""PDBQT writing, built on RDKit.

This module implements the AutoDock PDBQT dialect from its specification and
from the *behaviour* documented by Meeko. No AutoDockTools (ADT / MGLTools)
source was consulted, and no code was copied from it; the chemistry comes from
RDKit and the format rules from the published PDBQT grammar.

Layout produced for a ligand (which is exactly the kinematic tree `dock-core`
consumes)::

    ROOT
    <atoms of the root rigid fragment>
    ENDROOT
    BRANCH   a   b
    <atom b, then the remaining atoms of the fragment it opens>
    ...
    ENDBRANCH   a   b
    TORSDOF n

``b`` is the atom that stays *rigid relative to the parent frame* — the rotation
axis runs from ``a`` to ``b``. It is written inside the branch but belongs to
the parent's rigid cluster, which is what makes the nesting unambiguous.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Ad4Types",
    "LigandTree",
    "assign_ad4_types",
    "build_ligand_tree",
    "format_atom_line",
    "gasteiger_charges",
    "nonpolar_hydrogens",
    "polar_hydrogens",
    "rotatable_bonds",
    "strip_nonpolar_hydrogens",
    "write_ligand_pdbqt",
    "write_receptor_pdbqt",
]

# ---------------------------------------------------------------------------
# RDKit import guard
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    _HAVE_RDKIT = False


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for PDBQT preparation. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Atom typing
# ---------------------------------------------------------------------------

#: The canonical AutoDock 4 atom types this writer can emit.
AD4_TYPES = (
    "C", "A", "N", "O", "P", "S", "H", "F", "I",
    "NA", "OA", "SA", "HD", "Mg", "Mn", "Zn", "Ca", "Fe",
    "Cl", "Br", "Si", "At", "G0", "G1", "G2", "G3",
    "CG0", "CG1", "CG2", "CG3", "W",
)

#: Metals with a dedicated AD4 type.
_METAL_TYPES = {
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
    "Si": "Si", "Cl": "Cl", "Br": "Br", "I": "I", "F": "F",
}

# Standard H-bond donor / acceptor SMARTS (the definitions used by RDKit's
# own BaseFeatures.fdef).
_DONOR_SMARTS = Chem.MolFromSmarts(
    "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"
) if _HAVE_RDKIT else None
_ACCEPTOR_SMARTS = Chem.MolFromSmarts(
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0])]"
) if _HAVE_RDKIT else None


@dataclass(frozen=True)
class Ad4Types:
    """Per-atom AutoDock typing."""

    types: Dict[int, str] = field(default_factory=dict)
    donors: Set[int] = field(default_factory=set)
    acceptors: Set[int] = field(default_factory=set)

    def __getitem__(self, idx: int) -> str:
        return self.types[idx]


def _element(atom) -> str:
    return atom.GetSymbol().capitalize()


def _has_aromatic_neighbour(atom) -> bool:
    return any(n.GetIsAromatic() for n in atom.GetNeighbors())


def assign_ad4_types(mol) -> Ad4Types:
    """Assign AutoDock 4 atom types, donor and acceptor flags.

    The result is what `dock-core` needs: it re-derives the X-Score
    (Vina/Vinardo) typing from the element, the AD4 type *and the bonding
    graph*, so the two flags that actually matter here are

    * whether an `N`/`O`/`S` is an H-bond acceptor, and
    * whether a hydrogen is polar (`HD`) rather than non-polar (`H`).
    """
    require_rdkit()
    donors: Set[int] = set()
    acceptors: Set[int] = set()
    if _DONOR_SMARTS is not None:
        for match in mol.GetSubstructMatches(_DONOR_SMARTS):
            donors.add(int(match[0]))
    if _ACCEPTOR_SMARTS is not None:
        for match in mol.GetSubstructMatches(_ACCEPTOR_SMARTS):
            acceptors.add(int(match[0]))

    types: Dict[int, str] = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        el = _element(atom)
        if el == "H":
            heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            polar = bool(heavy) and _element(heavy[0]) in ("N", "O", "S")
            types[idx] = "HD" if polar else "H"
        elif el == "C":
            types[idx] = "A" if atom.GetIsAromatic() else "C"
        elif el == "N":
            types[idx] = "NA" if idx in acceptors else "N"
        elif el == "O":
            # Every neutral oxygen in a drug-like molecule is an acceptor in
            # the AD4 typing; positively charged/protonated oxygens are rare
            # enough that treating them as OA only costs a small amount of
            # H-bond strength.
            types[idx] = "OA"
        elif el == "S":
            types[idx] = "SA" if idx in acceptors else "S"
        elif el == "P":
            types[idx] = "P"
        elif el in _METAL_TYPES:
            types[idx] = _METAL_TYPES[el]
        else:
            # Unknown element: fall back to the closest element type so the
            # molecule still docks (Vina does the same via `Se -> S`).
            types[idx] = "S" if el == "Se" else "C"
    return Ad4Types(types=types, donors=donors, acceptors=acceptors)


# ---------------------------------------------------------------------------
# Hydrogens
# ---------------------------------------------------------------------------


def nonpolar_hydrogens(mol) -> List[int]:
    """Indices of hydrogens bound to carbon (merged away by Meeko/Vina)."""
    require_rdkit()
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if heavy and heavy[0].GetAtomicNum() == 6:
            out.append(atom.GetIdx())
    return out


def polar_hydrogens(mol) -> List[int]:
    """Indices of hydrogens bound to N, O or S."""
    require_rdkit()
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if heavy and heavy[0].GetAtomicNum() in (7, 8, 16):
            out.append(atom.GetIdx())
    return out


def strip_nonpolar_hydrogens(mol):
    """Return a copy with non-polar hydrogens removed.

    Non-polar hydrogens are absorbed into their parent carbon (the united-atom
    convention of AutoDock). Polar hydrogens are kept because the Vina force
    field derives the donor flag from their presence in the bond graph.
    """
    require_rdkit()
    drop = set(nonpolar_hydrogens(mol))
    if not drop:
        return Chem.Mol(mol)
    em = Chem.RWMol(mol)
    for idx in sorted(drop, reverse=True):
        em.RemoveAtom(idx)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------


def gasteiger_charges(mol, *, warn: bool = True) -> List[float]:
    """Gasteiger-Marsili partial charges, one per atom.

    Returns zeros when RDKit cannot assign charges (e.g. an unsanitizable
    protein), because charges only affect the AD4 force field.

    A charge that is not finite -- RDKit's charge model does not converge for a
    few protein atoms and reports ``nan`` or ``inf`` for them -- is written as
    ``0.0`` and reported, because the alternatives are both worse: an ``inf``
    charge silently turns every AD4 electrostatic pair that touches the atom
    into a ``NaN`` total, and a silent substitution hides the lost term.  The
    warning is emitted once per molecule and names the number of atoms and the
    first few by residue and atom name.  Pass ``warn=False`` for a bulk call that
    reports it itself.
    """
    require_rdkit()
    charges = [0.0] * mol.GetNumAtoms()
    bad: List[str] = []
    try:
        work = Chem.Mol(mol)
        AllChem.ComputeGasteigerCharges(work)
        for atom in work.GetAtoms():
            if not atom.HasProp("_GasteigerCharge"):
                continue
            try:
                c = float(atom.GetProp("_GasteigerCharge"))
            except ValueError:
                c = 0.0
            if not math.isfinite(c):
                bad.append(_atom_label(atom))
                c = 0.0
            charges[atom.GetIdx()] = c
    except Exception:
        pass
    if bad and warn:
        warnings.warn(
            f"{len(bad)} Gasteiger charge(s) were not finite and were written as "
            f"0.000 ({', '.join(bad[:4])}"
            + (", ..." if len(bad) > 4 else "")
            + "); the AD4 electrostatic term involving them is lost, so treat an "
            "AD4 score on this receptor as approximate",
            UserWarning,
            stacklevel=3,
        )
    return charges


def _atom_label(atom) -> str:
    """``"LEU67:N"`` for an RDKit atom with PDB information, else its index."""
    info = atom.GetPDBResidueInfo()
    if info is None:
        return f"atom {atom.GetIdx() + 1}"
    return f"{info.GetResidueName().strip()}{info.GetResidueNumber()}:{info.GetName().strip()}"


# ---------------------------------------------------------------------------
# Rotatable bonds and the kinematic tree
# ---------------------------------------------------------------------------

# Amide C-N and ester/amide-like C-O bonds are treated as rigid by default:
# their rotational barrier is high enough that treating them as rotors mostly
# adds search dimensions without adding real conformational freedom.
_AMIDE_SMARTS = Chem.MolFromSmarts("[NX3][CX3](=[OX1])") if _HAVE_RDKIT else None


def _is_amide_bond(bond) -> bool:
    if _AMIDE_SMARTS is None:
        return False
    a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
    if a1.GetAtomicNum() == 7 and a2.GetAtomicNum() == 6:
        n, c = a1, a2
    elif a2.GetAtomicNum() == 7 and a1.GetAtomicNum() == 6:
        n, c = a2, a1
    else:
        return False
    mol = bond.GetOwningMol()
    # A carbonyl carbon: C double-bonded to O.
    for nb in c.GetNeighbors():
        if nb.GetAtomicNum() != 8:
            continue
        cbond = mol.GetBondBetweenAtoms(c.GetIdx(), nb.GetIdx())
        if cbond is not None and cbond.GetBondType() == Chem.BondType.DOUBLE:
            return True
    return False


def _heavy_degree(atom) -> int:
    return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)


def rotatable_bonds(mol, rigid_amides: bool = True) -> List[int]:
    """Indices of the rotatable bonds, in a deterministic order.

    A bond is rotatable when it is a non-ring, non-aromatic single bond between
    two heavy atoms that each have at least one further heavy neighbour (which
    excludes terminal groups such as methyls and hydroxyls).
    """
    require_rdkit()
    ring_info = mol.GetRingInfo()
    out: List[int] = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.GetIsAromatic():
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        if _heavy_degree(a1) < 2 or _heavy_degree(a2) < 2:
            continue
        if rigid_amides and _is_amide_bond(bond):
            continue
        out.append(bond.GetIdx())
    out.sort()
    return out


@dataclass
class LigandTree:
    """A rigid-fragment tree ready for PDBQT serialisation."""

    #: Atom indices of the root rigid fragment.
    root_atoms: List[int]
    #: ``(parent_atom, child_atom, subtree)`` for every torsion.
    children: List[Tuple[int, int, "LigandTree"]]
    #: Total number of torsions.
    num_torsions: int

    def iter_nodes(self) -> Iterable["LigandTree"]:
        yield self
        for _, _, sub in self.children:
            yield from sub.iter_nodes()


def build_ligand_tree(mol, root: Optional[int] = None) -> LigandTree:
    """Split `mol` into rigid fragments and build the torsion tree.

    The default root is the heavy atom closest to the molecular centroid, which
    keeps the rigid root compact and therefore keeps the rigid-body translation
    degree of freedom well conditioned.
    """
    require_rdkit()
    rot = set(rotatable_bonds(mol))
    n = mol.GetNumAtoms()

    # Fragment membership: connected components of the graph without rotors.
    fragment = [-1] * n
    fragments: List[List[int]] = []
    for start in range(n):
        if fragment[start] != -1:
            continue
        fid = len(fragments)
        stack = [start]
        fragment[start] = fid
        members = [start]
        while stack:
            cur = stack.pop()
            for bond in mol.GetAtomWithIdx(cur).GetBonds():
                if bond.GetIdx() in rot:
                    continue
                other = bond.GetOtherAtomIdx(cur)
                if fragment[other] == -1:
                    fragment[other] = fid
                    members.append(other)
                    stack.append(other)
        fragments.append(sorted(members))

    if root is None:
        conf = mol.GetConformer()
        centroid = [0.0, 0.0, 0.0]
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        for idx in heavy:
            p = conf.GetAtomPosition(idx)
            centroid[0] += p.x
            centroid[1] += p.y
            centroid[2] += p.z
        inv = 1.0 / max(len(heavy), 1)
        centroid = [c * inv for c in centroid]
        best, best_d = heavy[0], float("inf")
        for idx in heavy:
            p = conf.GetAtomPosition(idx)
            d = (p.x - centroid[0]) ** 2 + (p.y - centroid[1]) ** 2 + (p.z - centroid[2]) ** 2
            if d < best_d:
                best, best_d = idx, d
        root = best

    root_fragment = fragment[root]
    total_torsions = [0]

    def build(fid: int, incoming_attach: Optional[int]) -> LigandTree:
        members = [i for i in fragments[fid] if i != incoming_attach]
        children: List[Tuple[int, int, LigandTree]] = []
        # Every rotor that leaves this fragment and has not been claimed yet.
        claimed = {incoming_attach} if incoming_attach is not None else set()
        for idx in fragments[fid]:
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                if bond.GetIdx() not in rot:
                    continue
                other = bond.GetOtherAtomIdx(idx)
                if fragment[other] == fid or other in claimed:
                    continue
                if fragment[other] in visited:
                    continue
                claimed.add(other)
                visited.add(fragment[other])
                total_torsions[0] += 1
                children.append((idx, other, build(fragment[other], other)))
        return LigandTree(root_atoms=members, children=children, num_torsions=0)

    visited = {root_fragment}
    tree = build(root_fragment, None)
    tree.num_torsions = total_torsions[0]
    for node in tree.iter_nodes():
        node.num_torsions = total_torsions[0]
    return tree


def flatten_tree(tree: LigandTree) -> List[int]:
    """The atom order `dock-core` assigns to a ligand.

    The kernel's reader assigns movable atom indices by walking the topology
    *frame by frame*: a frame's own atoms first, then the attachment atoms of
    all of its child branches, then the child branches themselves. This helper
    reproduces that order exactly, so the *k*-th coordinate returned by
    ``Docking.pose_coords`` belongs to ``flatten_tree(tree)[k]``.
    """
    out: List[int] = []

    def walk(node: LigandTree, attach: Optional[int]) -> None:
        for idx in node.root_atoms:
            if attach is not None and idx == attach:
                continue
            out.append(idx)
        for _, child_atom, _ in node.children:
            out.append(child_atom)
        for _, child_atom, sub in node.children:
            walk(sub, child_atom)

    walk(tree, None)
    return out


# ---------------------------------------------------------------------------
# Line formatting
# ---------------------------------------------------------------------------


def format_atom_line(
    serial: int,
    name: str,
    res_name: str,
    chain: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    charge: float,
    ad_type: str,
    record: str = "ATOM  ",
    occupancy: float = 1.0,
    bfactor: float = 0.0,
) -> str:
    """Render one PDBQT atom record with exact column placement."""
    name4 = (name[:4]).ljust(4)
    res3 = (res_name[:3]).rjust(3)
    line = (
        f"{record[:6]:<6}"
        f"{serial:>5d}"
        f" "
        f"{name4}"
        f" "
        f"{res3}"
        f" "
        f"{(chain or ' ')[0]:1}"
        f"{res_seq:>4d}"
        f"    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}"
        f"    "
        f"{charge:6.3f}"
        f" "
        f"{ad_type:>2}"
    )
    return line


# ---------------------------------------------------------------------------
# Ligand writing
# ---------------------------------------------------------------------------


def _atom_name(mol, idx: int) -> str:
    atom = mol.GetAtomWithIdx(idx)
    if atom.HasProp("_TriposAtomName"):
        return atom.GetProp("_TriposAtomName")
    if atom.GetPDBResidueInfo() is not None:
        nm = atom.GetPDBResidueInfo().GetName().strip()
        if nm:
            return nm
    return f"{_element(atom)}{idx + 1}"


def _residue_info(mol, idx: int) -> Tuple[str, str, int]:
    atom = mol.GetAtomWithIdx(idx)
    info = atom.GetPDBResidueInfo()
    if info is not None:
        return (
            info.GetResidueName().strip() or "LIG",
            info.GetChainId().strip() or "A",
            int(info.GetResidueNumber()) or 1,
        )
    return ("LIG", "A", 1)


def write_ligand_pdbqt(
    mol,
    name: str = "ligand",
    rigid_amides: bool = True,
    root: Optional[int] = None,
    include_header: bool = True,
    tree: Optional[LigandTree] = None,
    return_order: bool = False,
):
    """Serialise `mol` as a ligand PDBQT document.

    The molecule must have a 3-D conformer. Non-polar hydrogens should already
    have been stripped (see :func:`strip_nonpolar_hydrogens`).

    When `return_order` is true the result is ``(text, order)`` where `order`
    lists the RDKit atom indices in the order the kernel will assign them (see
    :func:`flatten_tree`).
    """
    require_rdkit()
    if mol.GetNumConformers() == 0:
        raise ValueError("the ligand has no 3-D conformer; embed it first")
    conf = mol.GetConformer()
    types = assign_ad4_types(mol)
    charges = gasteiger_charges(mol)
    if tree is None:
        tree = build_ligand_tree(mol, root=root)

    header: List[str] = []
    if include_header:
        header.append(f"REMARK  OpenDocking ligand preparation: {name}")
        header.append("REMARK  AutoDock atom types assigned from RDKit perception")
        header.append(f"REMARK  torsions = {tree.num_torsions}")

    lines: List[str] = list(header)
    lines.append("ROOT")
    serial = [0]
    # RDKit index -> PDBQT serial. The emission order follows the kinematic
    # tree, which is *not* the RDKit index order, so the serials must be
    # recorded rather than assumed: a `BRANCH a b` record refers to serials.
    serial_of: Dict[int, int] = {}

    def serial_for(idx: int) -> int:
        if idx not in serial_of:
            serial[0] += 1
            serial_of[idx] = serial[0]
        return serial_of[idx]

    def emit(idx: int) -> None:
        n = serial_for(idx)
        p = conf.GetAtomPosition(idx)
        res_name, chain, res_seq = _residue_info(mol, idx)
        lines.append(
            format_atom_line(
                n,
                _atom_name(mol, idx),
                res_name,
                chain,
                res_seq,
                p.x,
                p.y,
                p.z,
                charges[idx],
                types[idx],
                record="HETATM" if res_name != "LIG" else "ATOM  ",
            )
        )

    def emit_children(node: LigandTree) -> None:
        """Emit one `BRANCH` block per torsion, in tree order."""
        for parent_atom, child_atom, sub in node.children:
            serial[0] += 1
            p = conf.GetAtomPosition(child_atom)
            res_name, chain, res_seq = _residue_info(mol, child_atom)
            # `BRANCH a b` refers to *serials*, so the axis atoms must be
            # numbered before the record that names them is written.
            parent_serial = serial_for(parent_atom)
            child_serial = serial_for(child_atom)
            lines.append(f"BRANCH {parent_serial:>4d} {child_serial:>4d}")
            lines.append(
                format_atom_line(
                    child_serial,
                    _atom_name(mol, child_atom),
                    res_name,
                    chain,
                    res_seq,
                    p.x,
                    p.y,
                    p.z,
                    charges[child_atom],
                    types[child_atom],
                    record="HETATM" if res_name != "LIG" else "ATOM  ",
                )
            )
            for idx in sub.root_atoms:
                if idx == child_atom:
                    continue
                emit(idx)
            emit_children(sub)
            lines.append(f"ENDBRANCH {parent_serial:>4d} {child_serial:>4d}")

    # ROOT holds the rigid cluster of the root frame; every BRANCH block then
    # introduces one torsion, whose attachment atom is written at the top of the
    # block and belongs to the parent frame.
    for idx in tree.root_atoms:
        emit(idx)
    lines.append("ENDROOT")
    emit_children(tree)
    lines.append(f"TORSDOF {tree.num_torsions}")
    text = "\n".join(lines) + "\n"
    if return_order:
        return text, flatten_tree(tree)
    return text


def write_receptor_pdbqt(mol, include_header: bool = True) -> str:
    """Serialise `mol` as a rigid receptor PDBQT document."""
    require_rdkit()
    if mol.GetNumConformers() == 0:
        raise ValueError("the receptor has no 3-D conformer")
    conf = mol.GetConformer()
    types = assign_ad4_types(mol)
    charges = gasteiger_charges(mol)
    lines: List[str] = []
    if include_header:
        lines.append("REMARK  OpenDocking receptor preparation")
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        p = conf.GetAtomPosition(idx)
        res_name, chain, res_seq = _residue_info(mol, idx)
        lines.append(
            format_atom_line(
                idx + 1,
                _atom_name(mol, idx),
                res_name,
                chain,
                res_seq,
                p.x,
                p.y,
                p.z,
                charges[idx],
                types[idx],
                record="ATOM  " if atom.GetAtomicNum() > 1 else "HETATM",
            )
        )
    lines.append("TER")
    return "\n".join(lines) + "\n"

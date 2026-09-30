# SPDX-License-Identifier: GPL-3.0-or-later
"""Flexible-receptor PDBQT: split side chains out of the rigid framework.

`the project brief` module A.4 asks for the AutoDock flexible-residue representation: the
backbone stays in the rigid receptor and each selected side chain is emitted
between ``BEGIN_RES``/``END_RES`` as a nested ``BRANCH`` torsion tree, so that
AutoDock and AutoDock Vina can treat those chi angles as search dimensions.

The side-chain torsion tree is produced by the same writer the ligands use
(:func:`odock.pdbqt.write_ligand_pdbqt`), which is the point: a side chain *is*
a ligand as far as the format is concerned, and reusing the tested writer means
the ``BRANCH`` numbering cannot drift from the ``ATOM`` serials.

Known limitation, stated here rather than hidden: OpenDocking's own search is
rigid-receptor in this release. Its PDBQT reader folds ``BEGIN_RES`` records back
into the rigid framework and says so in the parse issues. The file written here
is therefore complete and AutoDock-compatible, but docking *with* the flexible
side chains requires a kernel that consumes them.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["write_flexible_receptor_pdbqt", "backbone_atom_names", "split_flexible_residues"]

#: Heavy backbone atoms; everything else in the residue is side chain.
BACKBONE_HEAVY = ("N", "CA", "C", "O", "OXT", "OT1", "OT2")

_ATOM_SERIAL = slice(6, 11)
_BRANCH = re.compile(r"^(END)?BRANCH\s+(\d+)\s+(\d+)\s*$")


def backbone_atom_names() -> Tuple[str, ...]:
    """Atom names that stay in the rigid framework."""
    return BACKBONE_HEAVY


def _residue_key(atom) -> Tuple[str, str, int]:
    return (atom.res_name.strip().upper(), (atom.chain or " ").strip(), int(atom.res_id))


def _normalise_label(label: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Accept "ILE16", "ILE A16", "16" or "ILEA16"."""
    text = "".join(str(label).split()).upper()
    match = re.match(r"^([A-Z]{3})?([A-Z]?)(-?\d+)$", text)
    if not match:
        return None, None, None
    name, chain, number = match.groups()
    return name or None, chain or None, int(number)


def split_flexible_residues(mol, labels: Iterable[str]):
    """Split `mol` into ``(rigid_mol, [(label, side_chain_mol), ...])``.

    The rigid molecule keeps every atom except the selected side chains; each
    side chain is returned as its own molecule in the original atom order.
    """
    require_rdkit()
    wanted = []
    for label in labels:
        name, chain, number = _normalise_label(label)
        if number is None:
            raise ValueError(f"cannot parse the residue label {label!r}")
        wanted.append((name, chain, number))

    side_chain: Dict[int, Tuple[str, List[int]]] = {}
    for atom in mol.GetAtoms():
        if atom.GetPDBResidueInfo() is None:
            continue
        name, chain, number = _residue_key_from_atom(atom)
        for want_name, want_chain, want_number in wanted:
            if number != want_number:
                continue
            if want_name is not None and name != want_name:
                continue
            if want_chain is not None and chain != want_chain:
                continue
            if atom.GetSymbol() == "H" and not _is_backbone_hydrogen(atom):
                side_chain.setdefault(number, (f"{name} {chain}{number}", []))[1].append(
                    atom.GetIdx()
                )
            elif atom.GetAtomicNum() > 1 and atom.GetPDBResidueInfo().GetName().strip() not in BACKBONE_HEAVY:
                side_chain.setdefault(number, (f"{name} {chain}{number}", []))[1].append(
                    atom.GetIdx()
                )
            break

    rigid = Chem.RWMol(mol)
    payload: List[Tuple[str, object]] = []
    remove_all: List[int] = []
    for _, (label, indices) in sorted(side_chain.items()):
        if not indices:
            continue
        editable = Chem.RWMol(mol)
        keep = set(indices)
        for index in sorted(
            (i for i in range(editable.GetNumAtoms()) if i not in keep), reverse=True
        ):
            editable.RemoveAtom(index)
        payload.append((label, editable.GetMol()))
        remove_all.extend(indices)
    # Every index is in the *original* numbering, so all the side chains are
    # removed from the rigid copy in one descending pass. Removing them residue
    # by residue would invalidate the indices of the residues that follow.
    for index in sorted(remove_all, reverse=True):
        rigid.RemoveAtom(index)
    return rigid.GetMol(), payload


def _residue_key_from_atom(atom) -> Tuple[str, str, int]:
    info = atom.GetPDBResidueInfo()
    return (
        info.GetResidueName().strip().upper(),
        info.GetChainId().strip(),
        int(info.GetResidueNumber()),
    )


def _is_backbone_hydrogen(atom) -> bool:
    info = atom.GetPDBResidueInfo()
    if info is None:
        return False
    name = info.GetName().strip().upper()
    # Backbone amide / alpha hydrogens: H, H1..H3, HA, HA2, HA3.
    return name in ("H", "HA", "HA2", "HA3") or name in ("H1", "H2", "H3")


def _renumber(block: Sequence[str], offset: int) -> List[str]:
    """Shift the serials of a ligand block so they continue from the rigid part.

    ``write_ligand_pdbqt`` numbers from 1; a flexible receptor needs one
    continuous serial series, and the ``BRANCH`` records must reference the new
    numbers. Serial 0 means "no offset" and is left alone.
    """
    mapping: Dict[int, int] = {}

    def new_serial(old: int) -> int:
        if old == 0:
            return 0
        if old not in mapping:
            mapping[old] = len(mapping) + 1 + offset
        return mapping[old]

    out: List[str] = []
    for line in block:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            record = line[:6]
            rest = line[11:] if len(line) > 11 else ""
            try:
                serial = int(line[_ATOM_SERIAL])
            except ValueError:
                out.append(line)
                continue
            out.append(f"{record}{new_serial(serial):5d}{rest}")
            continue
        match = _BRANCH.match(line.strip())
        if match:
            end, first, second = match.groups()
            prefix = "ENDBRANCH" if end else "BRANCH"
            out.append(f"{prefix} {new_serial(int(first)):4d} {new_serial(int(second)):4d}")
            continue
        out.append(line)
    return out


def write_flexible_receptor_pdbqt(mol, residues: Sequence[str], *, offset_serial: int = 0) -> str:
    """Write `mol` as a flexible-receptor PDBQT.

    Parameters
    ----------
    residues
        Residue labels to make flexible, e.g. ``["TRP215", "LYS60"]`` or
        ``["TRP A215"]``. A label that matches nothing is reported in the
        returned text as a ``REMARK`` rather than silently ignored.
    """
    require_rdkit()
    import odock

    rigid, payload = split_flexible_residues(mol, residues)

    lines: List[str] = [
        "REMARK  OpenDocking flexible receptor",
        "REMARK  rigid framework written first; each flexible residue follows",
        "REMARK  between a labelled begin/end pair with a side-chain torsion tree",
    ]
    rigid_text = odock.write_receptor_pdbqt(rigid)
    rigid_atoms = [
        line
        for line in rigid_text.splitlines()
        if line.startswith(("ATOM", "HETATM", "TER"))
    ]
    if offset_serial:
        rigid_atoms = _renumber(rigid_atoms, offset_serial - 1)
    serial = 0
    for line in rigid_atoms:
        if line.startswith(("ATOM", "HETATM")):
            try:
                serial = max(serial, int(line[_ATOM_SERIAL]))
            except ValueError:
                pass
    lines.extend(rigid_atoms)

    found = set()
    for label, side in payload:
        name = label.split()[0]
        chain, number = "", 0
        for atom in side.GetAtoms():
            if atom.GetPDBResidueInfo() is not None:
                _, chain, number = _residue_key_from_atom(atom)
                break
        block = odock.write_ligand_pdbqt(side, name=name)
        body = [
            line
            for line in block.splitlines()
            if line.strip()
            and not line.startswith(("ROOT", "ENDROOT", "TORSDOF", "REMARK"))
        ]
        body = _renumber(body, serial)
        for line in body:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    serial = max(serial, int(line[_ATOM_SERIAL]))
                except ValueError:
                    pass
        lines.append(f"BEGIN_RES {name:>3s} {chain or ' '} {number:4d}")
        lines.extend(body)
        lines.append(f"END_RES {name:>3s} {chain or ' '} {number:4d}")
        found.add((name, chain, number))

    for label in residues:
        name, chain, number = _normalise_label(label)
        if number is None:
            continue
        if not any(
            number == got_number and (name is None or name == got_name)
            for got_name, _, got_number in found
        ):
            lines.append(f"REMARK  WARNING: no flexible residue matched {label}")
    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def require_rdkit():  # pragma: no cover - trivial guard
    """Import RDKit or explain what is missing."""
    global Chem
    try:
        from rdkit import Chem as _Chem  # noqa: PLC0415

        Chem = _Chem
        return _Chem
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("the flexible-receptor writer needs RDKit") from exc


Chem = None  # populated by require_rdkit()

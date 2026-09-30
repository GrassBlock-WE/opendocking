# SPDX-License-Identifier: GPL-3.0-or-later
"""Structure I/O for the 3-D workbench.

Deliberately free of RDKit: the GUI must render whatever ``odock dock`` wrote,
even on an installation that only has the optional GUI extras.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Atom", "Model", "element_color", "parse_pdbqt", "guess_bonds"]

#: AutoDock atom type -> element symbol.
AD_TYPE_ELEMENT: Dict[str, str] = {
    "C": "C", "A": "C", "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C",
    "N": "N", "NA": "N", "O": "O", "OA": "O", "S": "S", "SA": "S",
    "P": "P", "H": "H", "HD": "H", "F": "F", "I": "I", "Cl": "Cl",
    "Br": "Br", "Si": "Si", "At": "At",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
    "W": "H", "G0": "H", "G1": "H", "G2": "H", "G3": "H",
}

#: CPK-ish palette, tuned for a dark viewport.
_COLORS: Dict[str, Tuple[float, float, float]] = {
    "H": (0.92, 0.92, 0.92),
    "C": (0.55, 0.58, 0.62),
    "N": (0.32, 0.52, 0.95),
    "O": (0.92, 0.30, 0.28),
    "S": (0.90, 0.80, 0.25),
    "P": (0.95, 0.55, 0.20),
    "F": (0.45, 0.90, 0.45),
    "Cl": (0.35, 0.85, 0.35),
    "Br": (0.72, 0.40, 0.20),
    "I": (0.55, 0.30, 0.75),
    "Mg": (0.55, 0.90, 0.55),
    "Mn": (0.75, 0.60, 0.85),
    "Zn": (0.65, 0.65, 0.85),
    "Ca": (0.45, 0.85, 0.65),
    "Fe": (0.85, 0.55, 0.35),
}

#: van der Waals-ish radii for rendering (Å).
_RADII: Dict[str, float] = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
    "Mg": 1.73, "Mn": 1.73, "Zn": 1.39, "Ca": 1.94, "Fe": 1.72,
}


def element_color(element: str) -> Tuple[float, float, float]:
    """RGB colour for an element symbol."""
    return _COLORS.get(element, (0.75, 0.55, 0.85))


def element_radius(element: str) -> float:
    """Render radius for an element symbol."""
    return _RADII.get(element, 1.6)


@dataclass
class Atom:
    """One parsed PDBQT atom."""

    name: str
    element: str
    res_name: str
    res_id: int
    chain: str
    x: float
    y: float
    z: float
    charge: float = 0.0
    ad_type: str = "C"
    serial: int = 0

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Model:
    """One `MODEL` of a PDBQT document (or the whole rigid receptor)."""

    atoms: List[Atom] = field(default_factory=list)
    remarks: List[str] = field(default_factory=list)
    affinity: Optional[float] = None
    rmsd_lower: Optional[float] = None
    rmsd_upper: Optional[float] = None

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def center(self) -> Tuple[float, float, float]:
        if not self.atoms:
            return (0.0, 0.0, 0.0)
        n = float(len(self.atoms))
        return (
            sum(a.x for a in self.atoms) / n,
            sum(a.y for a in self.atoms) / n,
            sum(a.z for a in self.atoms) / n,
        )

    def bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        if not self.atoms:
            return ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        lo = (min(a.x for a in self.atoms), min(a.y for a in self.atoms), min(a.z for a in self.atoms))
        hi = (max(a.x for a in self.atoms), max(a.y for a in self.atoms), max(a.z for a in self.atoms))
        return lo, hi


#: Elements whose symbol is also a common protein atom name. In a polymer the
#: name wins ("CA" is the alpha carbon); in a HETATM record whose *residue* is
#: that element the element wins, which is how a calcium ion avoids being drawn
#: and typed as a carbon.
AMBIGUOUS_ELEMENTS = {
    "CA": "Ca", "CD": "Cd", "CO": "Co", "CU": "Cu", "FE": "Fe", "HG": "Hg",
    "MG": "Mg", "MN": "Mn", "MO": "Mo", "NA": "Na", "NI": "Ni", "PB": "Pb",
    "PT": "Pt", "SE": "Se", "ZN": "Zn", "CL": "Cl", "BR": "Br", "IOD": "I",
    "KS": "K", "K": "K",
}


def _element_of(name: str, ad_type: str, res_name: str = "", is_hetero: bool = False) -> str:
    letters = "".join(c for c in name if c.isalpha())
    residue = res_name.strip().upper()
    # A HETATM residue whose name is an element symbol and whose single atom
    # carries the same name *is* that element. This is checked before `ad_type`
    # because a plain PDB writes a calcium ion as `HETATM ... CA  CA ... C`, and
    # trusting the AD4 column there turns a metal into a carbon.
    if is_hetero and residue in AMBIGUOUS_ELEMENTS and letters.upper() == residue:
        return AMBIGUOUS_ELEMENTS[residue]
    element = AD_TYPE_ELEMENT.get(ad_type)
    if element:
        return element
    two = letters[:2].capitalize()
    if two in ("Cl", "Br", "Si", "At"):
        return two
    return letters[:1].upper() or "C"


def _parse_atom(line: str) -> Optional[Atom]:
    if len(line) < 54:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain = line[21:22].strip()
    try:
        res_id = int(line[22:26])
    except ValueError:
        res_id = 0
    try:
        serial = int(line[6:11])
    except ValueError:
        serial = 0
    charge = 0.0
    ad_type = ""
    if len(line) >= 79:
        try:
            charge = float(line[70:76])
        except ValueError:
            charge = 0.0
        ad_type = line[77:].strip()
    if not ad_type:
        toks = line.split()
        if toks:
            ad_type = toks[-1]
    return Atom(
        name=name,
        element=_element_of(name, ad_type, res_name, line.startswith("HETATM")),
        res_name=res_name,
        res_id=res_id,
        chain=chain,
        x=x,
        y=y,
        z=z,
        charge=charge,
        ad_type=ad_type,
        serial=serial,
    )


def parse_pdbqt(text: str) -> List[Model]:
    """Parse a PDBQT document into one :class:`Model` per `MODEL` record.

    A receptor (no `MODEL` records) yields exactly one model.
    """
    models: List[Model] = []
    current: Optional[Model] = None
    pending_remarks: List[str] = []

    def start() -> Model:
        m = Model()
        m.remarks.extend(pending_remarks)
        models.append(m)
        return m

    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = start()
            continue
        if line.startswith("ENDMDL"):
            current = None
            continue
        if line.startswith("REMARK"):
            if current is None and not models:
                pending_remarks.append(line.strip())
            if current is not None:
                current.remarks.append(line.strip())
            if "VINA RESULT" in line:
                parts = line.replace("REMARK", "").strip().split(":", 1)[-1].split()
                try:
                    vals = [float(p) for p in parts[:3]]
                except ValueError:
                    vals = []
                if len(vals) == 3:
                    target = current if current is not None else (models[0] if models else None)
                    if target is not None:
                        target.affinity, target.rmsd_lower, target.rmsd_upper = vals
            continue
        if line.startswith(("ATOM", "HETATM")):
            atom = _parse_atom(line)
            if atom is None:
                continue
            if current is None:
                if not models:
                    current = start()
                elif len(models) == 1:
                    current = models[0]
                else:
                    current = models[-1]
            assert current is not None
            current.atoms.append(atom)
    models = [m for m in models if m.atoms]
    if not models:
        raise ValueError("no atoms found in the PDBQT document")
    if models[0].affinity is None:
        # Remarks may have preceded the first MODEL record.
        for line in pending_remarks:
            if "VINA RESULT" in line:
                parts = line.replace("REMARK", "").strip().split(":", 1)[-1].split()
                try:
                    vals = [float(p) for p in parts[:3]]
                except ValueError:
                    continue
                if len(vals) == 3:
                    models[0].affinity, models[0].rmsd_lower, models[0].rmsd_upper = vals
    return models


def read_pdbqt(path) -> List[Model]:
    """Read and parse a PDBQT file."""
    return parse_pdbqt(Path(path).read_text(encoding="utf-8", errors="replace"))


#: Covalent radii for bond perception.
_COVALENT = {
    "H": 0.37, "C": 0.77, "N": 0.75, "O": 0.73, "S": 1.02, "P": 1.06,
    "F": 0.71, "Cl": 0.99, "Br": 1.14, "I": 1.33,
}


def guess_bonds(
    atoms: Sequence[Atom], factor: float = 1.25, max_bonds: int = 4
) -> List[Tuple[int, int]]:
    """Deprecated alias of :func:`odock.gui.bonds.perceive_bonds`.

    The old implementation was a plain ``factor × (r_i + r_j)`` distance test
    with a degree cap, which happily joined the two hydrogens of a rotating
    methyl or an N···H hydrogen-bond pair. It is kept only because callers
    exist; ``factor`` and ``max_bonds`` are accepted and ignored.

    The primary path in the workbench is
    :func:`odock.gui.bonds.perceive_bonds`, which returns :class:`Bond` objects
    that carry the bond order and unpack as ``(a, b)``.
    """
    from .bonds import bond_pairs, perceive_bonds  # local: avoid an import cycle

    return bond_pairs(perceive_bonds(atoms, kind="ligand"))

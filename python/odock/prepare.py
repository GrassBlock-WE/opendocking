# SPDX-License-Identifier: GPL-3.0-or-later
"""Receptor and ligand preparation for OpenDocking.

Everything chemical in this module goes through RDKit; the PDBQT serialisation
lives in :mod:`odock.pdbqt`. The rules implemented here are the publicly
documented ones:

* non-polar hydrogens are merged into their parent carbon (the AutoDock
  united-atom convention), polar hydrogens are kept because the Vina force
  field derives H-bond donors from their presence in the bond graph;
* Gasteiger-Marsili charges are attached for the AD4 force field (the Vina and
  Vinardo force fields ignore charges entirely);
* rotatable bonds are non-ring, non-aromatic single bonds whose two heavy
  endpoints each carry at least one further heavy neighbour, with amide bonds
  frozen by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import pdbqt as _pdbqt

__all__ = [
    "BoxSpec",
    "PreparationReport",
    "box_from_ligand",
    "box_from_points",
    "box_from_selection",
    "box_from_smiles_ligand",
    "prepare_ligand",
    "prepare_receptor",
    "pdbqt_to_pdb_block",
    "_organic_hetero_residues",
    "planarity",
    "read_structure",
]

PathLike = Union[str, os.PathLike]


def _require_rdkit():
    _pdbqt.require_rdkit()
    from rdkit import Chem
    from rdkit.Chem import AllChem

    return Chem, AllChem


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxSpec:
    """An axis-aligned search box."""

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    spacing: float = 0.375

    def __post_init__(self) -> None:
        # Coerce to plain float tuples so that a JSON round trip
        # (`BoxSpec(**json.loads(box.as_dict()))`) compares equal.
        object.__setattr__(self, "center", tuple(float(x) for x in self.center))
        object.__setattr__(self, "size", tuple(float(x) for x in self.size))
        object.__setattr__(self, "spacing", float(self.spacing))
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("center and size must have exactly three components")
        if any(s <= 0 for s in self.size):
            raise ValueError(f"box size must be positive, got {self.size}")
        if self.spacing <= 0:
            raise ValueError(f"spacing must be positive, got {self.spacing}")

    @property
    def volume(self) -> float:
        """Box volume in Å³."""
        return float(self.size[0] * self.size[1] * self.size[2])

    @property
    def corner1(self) -> Tuple[float, float, float]:
        """Lower corner."""
        return tuple(c - s / 2.0 for c, s in zip(self.center, self.size))  # type: ignore[return-value]

    @property
    def corner2(self) -> Tuple[float, float, float]:
        """Upper corner."""
        return tuple(c + s / 2.0 for c, s in zip(self.center, self.size))  # type: ignore[return-value]

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        """Whether a point lies inside the box."""
        lo, hi = self.corner1, self.corner2
        return all(lo[i] - margin <= point[i] <= hi[i] + margin for i in range(3))

    def as_dict(self, digits: int = 4) -> Dict[str, object]:
        """The JSON form of the box — the interface a front-end exchanges.

        Coordinates are rounded so the file reads as data rather than as
        floating-point noise (`-1.8555` instead of `-1.8555000000000001`); four
        decimals is 0.1 mA, far below anything the force field can resolve.
        """
        return {
            "center": [round(float(v), digits) for v in self.center],
            "size": [round(float(v), digits) for v in self.size],
            "spacing": round(float(self.spacing), digits),
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BoxSpec(center=({self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f}), "
            f"size=({self.size[0]:.1f}, {self.size[1]:.1f}, {self.size[2]:.1f}) Å, "
            f"V={self.volume:.0f} Å³)"
        )


def planarity(mol) -> float:
    """RMS distance of the atoms from their best-fit plane, in Å.

    This is the number that tells a user whether a structure is genuinely
    three-dimensional: a 2-D depiction gives ~0.00 Å, benzene (planar by
    chemistry) gives ~0.00 Å as well but is legitimate, and a drug-like
    molecule gives 0.2-1.0 Å. It is reported so that "is my ligand 3-D?" is an
    observation rather than an assumption.
    """
    _require_rdkit()
    if mol.GetNumConformers() == 0 or mol.GetNumAtoms() < 4:
        return 0.0
    coords = _coords_of(mol)
    centred = coords - coords.mean(axis=0)
    # Singular values of the centred coordinates: the smallest one measures the
    # out-of-plane spread.
    return float(np.linalg.svd(centred, compute_uv=False)[2] / np.sqrt(len(coords)))


def _has_3d_conformer(mol) -> bool:
    """Whether `mol` carries usable three-dimensional coordinates.

    RDKit's file readers set the conformer's ``Is3D`` flag, which is the only
    reliable test: a genuinely planar molecule such as benzene is legitimately
    flat, so a purely geometric planarity test would wrongly reject it and
    re-embed a perfectly good crystal structure.
    """
    if mol.GetNumConformers() == 0:
        return False
    try:
        return bool(mol.GetConformer().Is3D())
    except Exception:  # pragma: no cover - very old RDKit
        return True


def _coords_of(mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=float,
    )


def box_from_points(points: np.ndarray, buffer: float = 5.0, spacing: float = 0.375) -> BoxSpec:
    """Smallest box containing `points`, padded by `buffer` Å."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("points must be an (n, 3) array with n > 0")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    size = (hi - lo) + 2.0 * buffer
    size = np.maximum(size, 4.0 * spacing)  # never degenerate
    return BoxSpec(
        center=(float(center[0]), float(center[1]), float(center[2])),
        size=(float(size[0]), float(size[1]), float(size[2])),
        spacing=spacing,
    )


def box_from_ligand(mol, buffer: float = 5.0, spacing: float = 0.375, heavy_only: bool = True) -> BoxSpec:
    """Box around a (reference) ligand, padded by `buffer` Å."""
    _require_rdkit()
    idx = [a.GetIdx() for a in mol.GetAtoms() if (not heavy_only or a.GetAtomicNum() > 1)]
    if not idx:
        idx = list(range(mol.GetNumAtoms()))
    return box_from_points(_coords_of(mol)[idx], buffer=buffer, spacing=spacing)


def box_from_smiles_ligand(smiles: str, buffer: float = 5.0, spacing: float = 0.375, seed: int = 20240101) -> BoxSpec:
    """Embed `smiles` and build a box around it (useful for blind docking)."""
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("RDKit failed to embed the ligand")
    return box_from_ligand(mol, buffer=buffer, spacing=spacing)


def box_from_selection(mol, selector, buffer: float = 5.0, spacing: float = 0.375) -> BoxSpec:
    """Box around a user-selected subset of atoms.

    `selector(atom) -> bool` is applied to every atom of `mol`.
    """
    _require_rdkit()
    idx = [a.GetIdx() for a in mol.GetAtoms() if selector(a)]
    if not idx:
        raise ValueError("the atom selection is empty")
    return box_from_points(_coords_of(mol)[idx], buffer=buffer, spacing=spacing)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

_READERS = {
    ".sdf": "sdf",
    ".mol": "mol",
    ".mol2": "mol2",
    ".pdb": "pdb",
    ".pdbqt": "pdbqt",
    ".smi": "smi",
    ".smiles": "smi",
    ".xyz": "xyz",
}

#: AutoDock atom type to element symbol, for converting PDBQT back to PDB.
_AD_TYPE_ELEMENT = {
    "C": "C", "A": "C", "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C",
    "N": "N", "NA": "N", "O": "O", "OA": "O", "S": "S", "SA": "S",
    "P": "P", "H": "H", "HD": "H", "F": "F", "I": "I", "Cl": "Cl",
    "Br": "Br", "Si": "Si", "At": "At",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
}


def _carbonyl_like(mol, oxygen) -> bool:
    """`True` when an oxygen is a carbonyl rather than a hydroxyl.

    A PDB file carries coordinates but no bond orders, so nothing distinguishes
    `C=O` from `C-OH` except the bond length: carbonyls are ~1.23 Å, hydroxyls
    and ethers 1.31-1.43 Å.
    """
    conf = mol.GetConformer()
    neighbours = [n for n in oxygen.GetNeighbors() if n.GetAtomicNum() > 1]
    if len(neighbours) != 1:
        return False
    heavy = neighbours[0]
    p = conf.GetAtomPosition(oxygen.GetIdx())
    q = conf.GetAtomPosition(heavy.GetIdx())
    d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
    return d <= _CARBONYL_CUTOFF


def conservative_polar_hydrogens(mol, add_coords: bool = True):
    """Add only the polar hydrogens that a bond-order-less structure implies.

    A PDB ligand has a correct heavy-atom skeleton but no bond orders, so
    RDKit's valence model over-protonates: every ring nitrogen looks like an
    amine and picks up a hydrogen. Those phantom hydrogens turn aromatic ring
    nitrogens into H-bond *donors* and place atoms inside the pocket, which is
    exactly the kind of error that silently ruins a docking run.

    The rule used here needs only the connectivity and the geometry:

    * **N** outside a ring with at most two heavy neighbours keeps its missing
      valences (a primary amine gets three hydrogens, an aniline one);
    * **N** inside a ring gets none — a ring nitrogen with two heavy neighbours
      is aromatic or unsaturated and carries no hydrogen in a neutral molecule;
    * **O** with a single heavy neighbour gets one hydrogen only when the bond
      is longer than a carbonyl (see :func:`_carbonyl_like`);
    * **S** with a single heavy neighbour gets one hydrogen.

    Everything else is marked `NoImplicit` so RDKit cannot add hydrogens that
    this rule did not ask for.
    """
    Chem, _ = _require_rdkit()
    em = Chem.RWMol(mol)
    targets: List[int] = []
    for atom in em.GetAtoms():
        z = atom.GetAtomicNum()
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if z not in (7, 8, 16):
            atom.SetNoImplicit(True)
            continue
        in_ring = atom.IsInRing()
        if z == 7:
            want = 0 if in_ring or len(heavy) > 2 else 3 - len(heavy)
        elif z == 8:
            want = 1 if (len(heavy) == 1 and not _carbonyl_like(em, atom)) else 0
        else:
            want = 1 if len(heavy) == 1 else 0
        if want > 0:
            targets.append(atom.GetIdx())
        else:
            atom.SetNoImplicit(True)
    out = em.GetMol()
    if targets:
        from rdkit.Chem import AllChem as _AllChem

        out = _AllChem.AddHs(out, explicitOnly=False, addCoords=add_coords, onlyOnAtoms=targets)
    # `SetNoImplicit` survives `AddHs`, so no atom gains a hydrogen we did not ask
    # for; the molecule is nevertheless sanitised so that ring perception and
    # aromaticity stay consistent.
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    return out


def apply_template_bond_orders(mol, smiles: str):
    """Copy the bond orders of a SMILES template onto a 3-D heavy-atom skeleton.

    This is the exact fix for a PDB-derived ligand: the coordinates come from the
    experiment, the chemistry comes from the template.
    """
    Chem, _ = _require_rdkit()
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"cannot parse the template SMILES {smiles!r}")
    try:
        from rdkit.Chem import AllChem as _AllChem

        fixed = _AllChem.AssignBondOrdersFromTemplate(template, mol)
    except Exception as exc:
        raise ValueError(
            f"the template SMILES does not match the 3-D structure: {exc}"
        ) from exc
    try:
        Chem.SanitizeMol(fixed)
    except Exception:
        pass
    return fixed


def pdbqt_to_pdb_block(text: str) -> str:
    """Rewrite a PDBQT document as a PDB block RDKit can read.

    Columns 77-78 of a PDBQT record hold the *AutoDock* type (`OA`, `HD`, `A`,
    ...), which RDKit's PDB parser cannot map to an element. This helper
    replaces those two columns with the matching element symbol, drops the
    topology records, and leaves everything else untouched.
    """
    out = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if len(line) < 80:
                line = line.ljust(80)
            ad_type = line[77:].strip()
            element = _AD_TYPE_ELEMENT.get(ad_type)
            if element is None:
                # Fall back to the atom name.
                name = line[12:16].strip()
                letters = "".join(c for c in name if c.isalpha())
                element = letters[:2] if letters[:2] in ("Cl", "Br", "Si", "At") else letters[:1]
                element = element.capitalize() if element else "C"
            # RDKit reads the element from 1-based columns 77-78, i.e. 0-based
            # 76-77, which is exactly where the PDBQT atom type starts.
            line = line[:76] + f"{element:>2}"
            out.append(line)
        elif line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "MODEL", "ENDMDL", "TORSDOF")):
            continue
        elif line.startswith("TER"):
            out.append(line)
    out.append("END")
    return "\n".join(out) + "\n"


def read_structure(path: PathLike, sanitize: bool = True):
    """Read a structure file into an RDKit molecule.

    Supported: SDF, MOL, MOL2, PDB, PDBQT, SMILES and XYZ.
    """
    Chem, _ = _require_rdkit()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    ext = p.suffix.lower()
    kind = _READERS.get(ext)
    if kind is None:
        raise ValueError(
            f"unsupported file extension {ext!r}; supported: {sorted(_READERS)}"
        )

    if kind == "sdf":
        supplier = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=sanitize)
        mols = [m for m in supplier]
        if not mols:
            raise ValueError(f"{p} contains no readable molecule")
        mol = mols[0]
    elif kind == "mol":
        mol = Chem.MolFromMolFile(str(p), removeHs=False, sanitize=sanitize)
    elif kind == "mol2":
        mol = Chem.MolFromMol2File(str(p), removeHs=False, sanitize=sanitize)
    elif kind == "pdb":
        mol = Chem.MolFromPDBFile(str(p), removeHs=False, sanitize=sanitize)
    elif kind == "pdbqt":
        mol = Chem.MolFromPDBBlock(
            pdbqt_to_pdb_block(p.read_text(encoding="utf-8", errors="replace")),
            removeHs=False,
            sanitize=False,
            proximityBonding=True,
        )
    elif kind == "smi":
        text = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        if not text:
            raise ValueError(f"{p} is empty")
        mol = Chem.MolFromSmiles(text[0].split()[0])
    else:  # xyz
        mol = Chem.MolFromXYZFile(str(p))
    if mol is None:
        raise ValueError(f"RDKit could not parse {p}")
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
    return mol


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class PreparationReport:
    """What preparation did, for logging and for the CLI."""

    kind: str
    n_atoms_in: int = 0
    n_atoms_out: int = 0
    n_hydrogens_added: int = 0
    n_nonpolar_hydrogens_removed: int = 0
    n_rotatable_bonds: int = 0
    n_metal_atoms: int = 0
    #: True when a 3-D conformer had to be generated because the input was 2-D.
    reembedded: bool = False
    #: RMS distance of the atoms from their best-fit plane (Å). A 2-D structure
    #: scores ~0; a genuine three-dimensional molecule is well above 0.05 Å.
    thickness: float = 0.0
    #: RDKit atom indices in the order the kernel assigns movable atoms.
    atom_order: List[int] = field(default_factory=list)
    #: True when the ligand's bond orders came from the input itself (an SDF,
    #: MOL2, MOL, SMILES or ``smiles=`` template) rather than being inferred from
    #: geometry.  A structure read from a bare PDB or PDBQT carries no bond
    #: orders, so the ring/amide/aromatic chemistry is then a guess.
    #:
    #: This is the machine-readable form of the warning that already accompanies
    #: the untrusted case, and it is what
    #: :func:`odock.metrics.ligand_strain` uses to decide whether a force-field
    #: strain measured on this molecule can be trusted.
    chemistry_trusted: bool = False
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """A one-line summary."""
        return (
            f"{self.kind}: {self.n_atoms_in} -> {self.n_atoms_out} atoms, "
            f"{self.n_nonpolar_hydrogens_removed} non-polar H merged, "
            f"{self.n_hydrogens_added} H added, "
            f"{self.n_rotatable_bonds} rotatable bonds, "
            f"3-D extent {self.thickness:.2f} A out of plane"
        )


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------


def prepare_ligand(
    source: Union[PathLike, object],
    out: Optional[PathLike] = None,
    *,
    name: str = "ligand",
    smiles: Optional[str] = None,
    add_hydrogens: bool = True,
    strip_nonpolar: bool = True,
    rigid_amides: bool = True,
    embed: bool = True,
    optimize: bool = True,
    seed: int = 20240101,
    keep_nonpolar: Optional[bool] = None,
) -> Tuple[object, str, PreparationReport]:
    """Prepare a ligand for docking.

    Parameters
    ----------
    source
        A path (SDF/MOL/MOL2/PDB/SMILES) or an RDKit molecule.
    out
        Where to write the PDBQT. When ``None`` nothing is written.
    smiles
        The ligand's SMILES. Supplying it is strongly recommended whenever the
        structure comes from a PDB file, because a PDB carries no bond orders:
        the SMILES is used as a template so that aromaticity, tautomers and
        H-bond donors/acceptors are exact instead of inferred from geometry.
    keep_nonpolar
        Deprecated alias for ``strip_nonpolar=False``.

    Returns
    -------
    ``(mol, pdbqt_text, report)`` — the prepared molecule, its PDBQT text and a
    :class:`PreparationReport`.
    """
    Chem, AllChem = _require_rdkit()
    report = PreparationReport(kind="ligand")
    if keep_nonpolar is not None:
        strip_nonpolar = not keep_nonpolar

    #: File kinds that carry real bond orders. A PDB (or PDBQT) ligand does not:
    #: it has coordinates and connectivity but every bond reads as a single bond.
    trusted_kinds = {"sdf", "mol", "mol2", "smi", "smiles", "xyz"}
    source_kind = "object"

    if isinstance(source, (str, os.PathLike)):
        raw = str(source)
        p = Path(raw)
        if p.exists():
            suffix = p.suffix.lower().lstrip(".")
            source_kind = _READERS.get("." + suffix, suffix)
            if suffix in ("smi", "smiles"):
                lines = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                mol = Chem.MolFromSmiles(lines[0].split()[0])
                if mol is None:
                    raise ValueError(f"cannot parse the SMILES in {p}")
            else:
                mol = read_structure(p, sanitize=True)
        else:
            # Not a path: accept a bare SMILES string, which is the fastest way
            # to get a ligand into a docking run.
            source_kind = "smiles"
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                raise ValueError(
                    f"{raw!r} is neither an existing file nor a parsable SMILES string"
                )
    else:
        mol = Chem.Mol(source)

    report.n_atoms_in = mol.GetNumAtoms()

    # --- chemistry --------------------------------------------------------
    # `smiles` gives exact bond orders for a structure read from a PDB, which
    # otherwise has none: without them RDKit's valence model protonates every
    # ring nitrogen and the ligand acquires phantom H-bond donors.
    chemistry_trusted = source_kind in trusted_kinds
    if smiles:
        mol = apply_template_bond_orders(mol, smiles)
        chemistry_trusted = True
        report.warnings.append(
            "bond orders taken from the supplied SMILES template"
        )
    elif not chemistry_trusted and source_kind in ("pdb", "pdbqt"):
        report.warnings.append(
            f"a {source_kind.upper()} file carries no bond orders; polar hydrogens "
            "were placed with a bond-length and ring-membership rule. Pass "
            "smiles=... for exact chemistry."
        )
    # Machine-readable form of the same fact, for callers that have to decide
    # whether a number derived from this molecule can be trusted (the ligand
    # strain is the one that matters): a molecule whose bond orders were guessed
    # has no aromatic rings and no carbonyls, and a force field relaxation of it
    # relaxes a different molecule.
    report.chemistry_trusted = bool(chemistry_trusted)

    # A ligand must be three-dimensional before it can be docked. A 2-D
    # structure (the usual output of a drawing tool or a vendor catalogue) would
    # be scored as a flat molecule collapsed onto a plane, which is physically
    # meaningless — RDKit reports it through the conformer's `Is3D` flag.
    if not _has_3d_conformer(mol):
        if not embed:
            raise ValueError(
                "the ligand has no 3-D conformer and embedding is disabled; "
                "generate 3-D coordinates first or leave embed=True"
            )
        why = "2-D coordinates" if mol.GetNumConformers() else "no conformer"
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        if AllChem.EmbedMolecule(mol, params) != 0:
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise RuntimeError("RDKit failed to generate a 3-D conformer")
        report.reembedded = True
        report.warnings.append(
            f"the input carried {why}; a 3-D conformer was generated with ETKDG "
            f"(seed {int(seed)}), so the resulting pose is NOT the input geometry"
        )
    elif add_hydrogens and not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        before = mol.GetNumAtoms()
        if chemistry_trusted:
            mol = Chem.AddHs(mol, addCoords=True)
        else:
            mol = conservative_polar_hydrogens(mol, add_coords=True)
        report.n_hydrogens_added = mol.GetNumAtoms() - before

    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception as exc:  # pragma: no cover - best effort
            report.warnings.append(f"geometry optimisation skipped: {exc}")

    report.thickness = planarity(mol)

    if strip_nonpolar:
        before = mol.GetNumAtoms()
        mol = _pdbqt.strip_nonpolar_hydrogens(mol)
        report.n_nonpolar_hydrogens_removed = before - mol.GetNumAtoms()

    report.n_atoms_out = mol.GetNumAtoms()
    report.n_rotatable_bonds = len(_pdbqt.rotatable_bonds(mol, rigid_amides=rigid_amides))
    report.n_metal_atoms = sum(
        1 for a in mol.GetAtoms() if a.GetSymbol() in ("Mg", "Mn", "Zn", "Ca", "Fe")
    )

    # RDKit's `AddHs` leaves the new hydrogens with no PDB residue information, so
    # a ligand prepared from a PDB would be written with its heavy atoms in
    # `BEN` and its polar hydrogens in `LIG`.  That is not cosmetic: a reader
    # that groups atoms by residue -- RDKit's own proximity bonding, for one --
    # then refuses to bond across the boundary, every X-H bond is lost, and the
    # structure can no longer be typed by MMFF94.  The receptor path already
    # inherits the parent residue; the ligand path now does the same.
    # `inherit_name=False` keeps the writers' unique `H<serial>` naming for the
    # hydrogens: a ligand's two N-H hydrogens must not both be called `N1`.
    mol = _inherit_residue_info(mol, inherit_name=False)

    tree = _pdbqt.build_ligand_tree(mol)
    text, order = _pdbqt.write_ligand_pdbqt(
        mol, name=name, rigid_amides=rigid_amides, tree=tree, return_order=True
    )
    report.atom_order = order
    if out is not None:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
    return mol, text, report


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------

#: The 20 standard amino acids plus the common nucleotides.
_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASX", "GLX", "UNK", "SEC", "PYL",
    "A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU", "DI",
}

#: Residue names that are never treated as a co-crystallised ligand.
_KNOWN_HETERO = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "ACT", "ACE", "NH2",
    "MG", "MN", "ZN", "CA", "FE", "NA", "K", "CL", "BR", "IOD", "FMT",
    "MSE", "SEP", "TPO", "PTR", "CSO", "KCX", "MLY", "M3L", "HYP", "PCA",
    "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "NDG", "BGC",
}


def _organic_hetero_residues(mol) -> Dict[str, int]:
    """Non-standard residues that look like an organic ligand.

    A residue qualifies when it is not a standard residue, is not in the
    solvent/ion/common-modification list, and contains a carbon atom. Those are
    exactly the residues a user usually wants to strip before docking.
    """
    _require_rdkit()
    seen: Dict[str, int] = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        name = info.GetResidueName().strip().upper()
        if not name or name in _KNOWN_HETERO:
            continue
        # `AtomPDBResidueInfo` does not expose `GetIsStandardResidue` in every
        # RDKit build, so the standard-residue test is done by name.
        if name in _STANDARD_RESIDUES:
            continue
        if atom.GetAtomicNum() != 6:
            continue
        key = f"{name} {info.GetChainId().strip() or '_'}{info.GetResidueNumber()}"
        seen[key] = seen.get(key, 0) + 1
    return seen


def prepare_receptor(
    source: Union[PathLike, object],
    out: Optional[PathLike] = None,
    *,
    keep_water: bool = False,
    keep_hetero: bool = True,
    strip: Optional[Sequence[str]] = None,
    add_polar_hydrogens: bool = True,
    strict: bool = False,
) -> Tuple[object, str, PreparationReport]:
    """Prepare a rigid receptor for docking.

    Waters are removed by default and non-standard residues are kept (metals and
    cofactors usually matter for binding). Only hydrogens on N, O and S are
    added: those are the ones the force field uses to decide whether an atom is
    an H-bond donor, and adding them is far more reliable than protonating a
    whole protein.

    Parameters
    ----------
    strip
        Residue names to delete, e.g. ``strip=["BEN", "SO4"]``. Use it whenever
        the input PDB is a *holo* complex: leaving the co-crystallised ligand in
        the receptor blocks the very site being docked into. When a likely
        organic ligand is detected the report carries a warning naming it.
    """
    Chem, AllChem = _require_rdkit()
    report = PreparationReport(kind="receptor")

    if isinstance(source, (str, os.PathLike)):
        p = Path(source)
        if p.suffix.lower() == ".pdbqt":
            mol = read_structure(p, sanitize=False)
        else:
            mol = None
            for sanitize in (True, False):
                try:
                    mol = Chem.MolFromPDBFile(
                        str(p),
                        removeHs=False,
                        sanitize=sanitize,
                        proximityBonding=True,
                    )
                except Exception:
                    mol = None
                if mol is not None:
                    if not sanitize:
                        report.warnings.append(
                            "the receptor could not be sanitised; bond orders are "
                            "geometry-derived and H-bond typing may be approximate"
                        )
                    break
            if mol is None:
                raise ValueError(f"RDKit could not parse the receptor {p}")
    else:
        mol = Chem.Mol(source)

    report.n_atoms_in = mol.GetNumAtoms()

    if not _has_3d_conformer(mol):
        report.warnings.append(
            "the receptor file carries no 3-D coordinates; a docking run against "
            "it would be meaningless"
        )

    # --- drop waters -------------------------------------------------------
    if not keep_water:
        em = Chem.RWMol(mol)
        drop = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetPDBResidueInfo() is not None
            and a.GetPDBResidueInfo().GetResidueName().strip() in ("HOH", "WAT", "DOD")
        ]
        for idx in sorted(drop, reverse=True):
            em.RemoveAtom(idx)
        if drop:
            mol = em.GetMol()
            report.warnings.append(f"removed {len(drop)} water atoms")

    # --- drop explicitly requested residues ---------------------------------
    if strip:
        wanted = {s.strip().upper() for s in strip}
        em = Chem.RWMol(mol)
        drop = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetPDBResidueInfo() is not None
            and a.GetPDBResidueInfo().GetResidueName().strip().upper() in wanted
        ]
        for idx in sorted(drop, reverse=True):
            em.RemoveAtom(idx)
        if drop:
            mol = em.GetMol()
            report.warnings.append(
                f"removed {len(drop)} atoms of stripped residues {sorted(wanted)}"
            )

    # --- warn about a co-crystallised ligand -------------------------------
    # A holo structure keeps its ligand as a HETATM residue; leaving it in place
    # makes the receptor occupy exactly the site being docked into.
    leftovers = _organic_hetero_residues(mol)
    if leftovers:
        report.warnings.append(
            "the receptor still contains organic non-standard residues "
            f"({', '.join(sorted(leftovers))}); if one of them is the "
            "co-crystallised ligand, remove it with strip=[...] or keep_hetero=False"
        )

    # --- drop everything that is not a standard residue --------------------
    if not keep_hetero:
        em = Chem.RWMol(mol)
        drop = []
        for a in mol.GetAtoms():
            info = a.GetPDBResidueInfo()
            if info is not None and info.GetResidueName().strip():
                if not info.GetIsStandardResidue():
                    drop.append(a.GetIdx())
        for idx in sorted(drop, reverse=True):
            em.RemoveAtom(idx)
        if drop:
            mol = em.GetMol()
            report.warnings.append(f"removed {len(drop)} hetero atoms")

    # --- polar hydrogens ---------------------------------------------------
    if add_polar_hydrogens:
        targets = polar_hydrogen_sites(mol)
        if targets:
            try:
                before = mol.GetNumAtoms()
                mol = Chem.AddHs(
                    mol, explicitOnly=False, addCoords=True, onlyOnAtoms=targets
                )
                report.n_hydrogens_added = mol.GetNumAtoms() - before
                mol = _inherit_residue_info(mol)
            except Exception as exc:
                msg = f"could not add polar hydrogens: {exc}"
                if strict:
                    raise RuntimeError(msg) from exc
                report.warnings.append(msg)

    report.n_atoms_out = mol.GetNumAtoms()
    report.n_metal_atoms = sum(
        1 for a in mol.GetAtoms() if a.GetSymbol() in ("Mg", "Mn", "Zn", "Ca", "Fe")
    )

    text = _pdbqt.write_receptor_pdbqt(mol)
    if out is not None:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text, encoding="utf-8")
    return mol, text, report


#: Standard residue names whose ring nitrogens must not both be protonated.
_HIS_NAMES = {"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"}

#: A carbon-oxygen bond shorter than this is a carbonyl, not a hydroxyl.
_CARBONYL_CUTOFF = 1.30


def polar_hydrogen_sites(mol) -> List[int]:
    """Atom indices that should receive polar hydrogens.

    A PDB file carries coordinates but no bond orders, so RDKit's valence model
    cheerfully reports an implicit hydrogen on every carbonyl oxygen and every
    backbone carbonyl carbon. Adding those would put hydrogens *inside* the
    binding site and produce large spurious clashes — which is exactly what an
    unprepared receptor must not do.

    The sites are therefore chosen from chemistry, not from implicit valences:

    * **N** — one hydrogen per missing valence position (backbone amide N:
      one; lysine NZ: three; proline N: none, because it already has three
      heavy neighbours);
    * **O** — only for a single heavy neighbour held by a bond longer than
      :data:`_CARBONYL_CUTOFF` Å (hydroxyls, phenols) or for a lone oxygen
      (water);
    * **S** — one hydrogen when it has a single heavy neighbour (cysteine).
    """
    _require_rdkit()
    conf = mol.GetConformer()
    targets: List[int] = []
    seen_his: Dict[Tuple[str, int, str], int] = {}

    def heavy_neighbours(atom):
        return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]

    def existing_h(atom) -> int:
        return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)

    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in (7, 8, 16):
            continue
        heavy = heavy_neighbours(atom)
        heavy_deg = len(heavy)
        have = existing_h(atom)
        info = atom.GetPDBResidueInfo()
        res_name = info.GetResidueName().strip() if info is not None else ""
        res_key = (
            res_name,
            int(info.GetResidueNumber()) if info is not None else 0,
            info.GetChainId().strip() if info is not None else "",
        )

        if z == 7:
            if heavy_deg > 2:
                continue  # proline / already saturated
            if res_name in _HIS_NAMES and heavy_deg == 2:
                # Only one of the two imidazole nitrogens is protonated.
                if res_key in seen_his:
                    continue
                seen_his[res_key] = atom.GetIdx()
            want = max(0, 3 - heavy_deg)
        elif z == 8:
            if heavy_deg == 0:
                want = 2
            elif heavy_deg > 1:
                want = 0
            else:
                p = conf.GetAtomPosition(atom.GetIdx())
                q = conf.GetAtomPosition(heavy[0].GetIdx())
                d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
                want = 1 if d > _CARBONYL_CUTOFF else 0
        else:  # sulfur
            want = max(0, 2 - heavy_deg) if heavy_deg <= 1 else 0

        if want - have > 0:
            targets.append(atom.GetIdx())
    return targets


def _inherit_residue_info(mol, *, inherit_name: bool = True):
    """Copy the parent residue metadata onto freshly added hydrogens.

    RDKit's `AddHs` creates bare atoms with no PDB residue information, which
    would show up in the PDBQT as `LIG A 1`. Inheriting the parent's residue
    name, number and chain keeps the output chemically readable (and keeps the
    hydrogens next to their heavy atom in any downstream visualisation).

    This is not cosmetic.  A reader that groups atoms by residue -- RDKit's own
    proximity bonding, for instance -- does not bond across a change of residue
    name, so a ligand whose hydrogens sit in a different residue from its heavy
    atoms loses every X-H bond on a round trip and can no longer be typed by
    MMFF94.

    Parameters
    ----------
    inherit_name
        ``True`` (the default, and what the receptor path has always done) names
        each new hydrogen after its parent, so an amide hydrogen is called ``N``.
        ``False`` leaves the name empty, and the PDBQT writers then fall back to
        their own unique ``H<serial>`` convention -- which is what a ligand
        needs, because two hydrogens on one nitrogen would otherwise both be
        called ``N1``.
    """
    Chem, _ = _require_rdkit()
    from rdkit.Chem import rdmolops

    em = Chem.RWMol(mol)
    todo = []
    for atom in em.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetPDBResidueInfo() is not None:
            continue
        parents = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if not parents:
            continue
        info = parents[0].GetPDBResidueInfo()
        if info is None:
            continue
        todo.append((atom.GetIdx(), info.GetName() if inherit_name else "", info))
    for idx, name, info in todo:
        atom = em.GetAtomWithIdx(idx)
        info_cls = getattr(Chem, "AtomPDBResidueInfo", None) or getattr(rdmolops, "AtomPDBResidueInfo", None)
        if info_cls is None:
            break
        atom.SetMonomerInfo(info_cls(name, serialNumber=idx + 1))
        new_info = atom.GetPDBResidueInfo()
        new_info.SetResidueName(info.GetResidueName())
        new_info.SetResidueNumber(info.GetResidueNumber())
        new_info.SetChainId(info.GetChainId())
        new_info.SetIsHeteroAtom(info.GetIsHeteroAtom())
        new_info.SetOccupancy(1.0)
        new_info.SetTempFactor(0.0)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    return out

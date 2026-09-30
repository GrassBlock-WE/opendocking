# SPDX-License-Identifier: GPL-3.0-or-later
"""Receptor deep cleaning, missing-atom detection and pH-aware protonation.

This is module A of the brief: everything that happens to a receptor *before* it
is written as PDBQT.

What the module decides, and why
--------------------------------
**HETATM classification.**  A crystallographic structure mixes four very
different kinds of non-polymer residue: solvent, free ions, catalytic cofactors
and the co-crystallised ligand.  They must not be treated alike — water and
buffer ions are noise, a cofactor is chemistry, and a leftover ligand occupies
exactly the site that is about to be docked into.  The classifier therefore
works from (a) the published residue-name lists for solvent/ions/cofactors and
(b) *measured* properties for the ligand test (more than one heavy atom and more
than 150 Da of heavy-atom mass), so that an unknown ligand is still found without
any hand-written ligand list.  Standard amino acids and nucleotides are polymer
residues, not hetero groups, and never enter the inventory; modified residues
(MSE, SEP, TPO, ...) are polymer too, so a selenomethionine is never mistaken for
a ligand.

**Missing atoms.**  A PDB file with a disordered or unmodelled side chain is
extremely common and silently changes the pocket.  Every standard residue is
compared against its heavy-atom template and the deficit is reported.

**Protonation.**  pH decides the *chemical* state: Asp/Glu carboxylates, the
His tautomer, Lys/Arg amines, Cys/Tyr, and the two chain termini.  The state is
chosen with a Henderson-Hasselbalch criterion against the published side-chain
pKa values, the histidine tautomer additionally from its local H-bond
environment, and the hydrogens that the chosen state requires are placed on N,
O and S only.  Hydrogens bound to carbon are never added and never touched —
they are absorbed into their parent carbon by
:func:`strip_nonpolar_hydrogens`.

Hydrogen coordinates
--------------------
Hydrogens added by :func:`protonate` are placed geometrically: bonds are built
in the direction that best completes the existing coordination (the negative sum
of the neighbour unit vectors), at the standard X-H bond length, and for
multi-hydrogen groups (NH2, NH3+, CH-like nitrogen) the hydrogens are spread on
a cone at the tetrahedral angle with the azimuth chosen to keep them as far as
possible from the existing heavy neighbours.  This is a documented geometric
approximation, not a force-field-optimised placement: the H positions are good
enough for typing and for grid-based docking, but a caller that needs
experimental-quality N-H/O-H geometry should minimise the receptor afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "CleanPolicy",
    "HeteroInventory",
    "HeteroResidue",
    "KIND_COFACTOR",
    "KIND_ION",
    "KIND_LIGAND",
    "KIND_OTHER",
    "KIND_SOLVENT",
    "COFACTOR_NAMES",
    "COUNTERION_NAMES",
    "METAL_CATION_NAMES",
    "SOLVENT_NAMES",
    "STANDARD_AMINO_ACIDS",
    "STANDARD_NUCLEOTIDES",
    "STANDARD_RESIDUE_ATOMS",
    "classify_hetero",
    "clean_receptor",
    "extract_ligands",
    "missing_atoms",
    "protonate",
    "strip_nonpolar_hydrogens",
]

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Point3D = None  # type: ignore[assignment]
    _HAVE_RDKIT = False


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for receptor preparation. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Residue-name tables
# ---------------------------------------------------------------------------

KIND_SOLVENT = "solvent"
KIND_LIGAND = "ligand"
KIND_ION = "ion"
KIND_COFACTOR = "cofactor"
KIND_OTHER = "other"

#: The five kinds, in the order they are reported.
KINDS: Tuple[str, ...] = (
    KIND_SOLVENT,
    KIND_LIGAND,
    KIND_ION,
    KIND_COFACTOR,
    KIND_OTHER,
)

#: Water residue names. The exact list from the brief plus the common
#: variants; any name starting with one of :data:`SOLVENT_PREFIXES` is solvent
#: too, which catches ``TIP3P``, ``TIP4PEW``, ``SPC/E`` and friends without
#: having to enumerate every water model in existence.
SOLVENT_NAMES = frozenset(
    {"HOH", "WAT", "DOD", "TIP3", "H2O", "TIP4", "SPC", "SOL", "D2O"}
)
SOLVENT_PREFIXES: Tuple[str, ...] = ("HOH", "WAT", "DOD", "TIP", "SPC", "SOL", "T3P", "T4P", "D2O")

#: Metal cations (brief module A.1). Kept as residue names because that is how
#: they appear in a PDB file.
METAL_CATION_NAMES = frozenset(
    {"ZN", "MG", "CA", "MN", "FE", "CU", "CO", "NI", "CD", "HG",
     "NA", "K", "CS", "RB", "SR", "BA"}
)

#: Counter-ions and common crystallisation additives (brief module A.1).
COUNTERION_NAMES = frozenset(
    {"CL", "BR", "IOD", "F", "SO4", "PO4", "NO3", "ACT", "EDO", "GOL",
     "FMT", "MES", "TRS"}
)

#: Catalytic cofactors: never dropped by accident.
COFACTOR_NAMES = frozenset(
    {"HEM", "HEC", "HEME", "NAD", "NAP", "NADH", "NAH", "FAD", "FMN", "PLP",
     "PMP", "ATP", "ADP", "AMP", "GTP", "GDP", "SAM", "SAH", "COA", "TPP",
     "THF", "UQ", "MQ", "PQQ", "B12", "CLA", "BCL", "HTH"}
)

#: The 20 standard amino acids (plus the two ambiguous codes).
STANDARD_AMINO_ACIDS = frozenset(
    {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
     "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
     "ASX", "GLX"}
)

#: Post-translationally modified residues that are still *polymer*: they sit in
#: the chain, so they must never be classified as a co-crystallised ligand even
#: when their mass is above the ligand threshold (selenomethionine is 169 Da of
#: heavy atoms and would otherwise be stripped out of the protein).
MODIFIED_AMINO_ACIDS = frozenset(
    {"MSE", "SEC", "PYL", "SEP", "TPO", "PTR", "CSO", "KCX", "MLY", "M3L",
     "HYP", "PCA", "FME", "LLP", "CME", "CSD", "OCS", "SAC", "CAS", "ALY",
     "NEP", "HIC", "MHO", "SME", "YCM"}
)

#: Standard nucleotides.
STANDARD_NUCLEOTIDES = frozenset(
    {"A", "C", "G", "U", "I", "DA", "DC", "DG", "DT", "DU", "DI",
     "ADE", "CYT", "GUA", "THY", "URI", "PSU", "5MC", "1MA", "5MU", "7MG"}
)

#: Residues that belong to the polymer and therefore never enter the hetero
#: inventory.
POLYMER_RESIDUES = (
    STANDARD_AMINO_ACIDS | MODIFIED_AMINO_ACIDS | STANDARD_NUCLEOTIDES
)

#: A non-standard residue is a co-crystallised ligand when it has more than one
#: heavy atom and more than this much heavy-atom mass (brief module A.1).
LIGAND_MIN_MASS = 150.0

#: pH at which the average side chain is half-deprotonated. Values are the
#: published pKa of the free amino acid side chains (Creighton, *Proteins*, 2nd
#: ed.); the termini use the standard values for a peptide backbone.
PKA = {
    "ASP": 3.9,
    "GLU": 4.3,
    "HIS": 6.0,
    "CYS": 8.3,
    "TYR": 10.1,
    "LYS": 10.5,
    "ARG": 12.5,
    "N_TERM": 8.0,
    "C_TERM": 3.1,
}

#: Residue names that mean "histidine", whichever tautomer the file records.
HIS_RESIDUE_NAMES = frozenset({"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "HIS1"})

#: A C-O bond shorter than this is a carbonyl rather than a hydroxyl. A PDB file
#: carries no bond orders, so this is the only way to tell them apart.
CARBONYL_CUTOFF = 1.30

#: H-bond partner search radius for the histidine tautomer decision (Å).
HIS_PARTNER_CUTOFF = 3.5

#: A carboxylate within this distance of a histidine ring nitrogen makes the
#: imidazole doubly protonated (brief module A.2).
HIS_CARBOXYLATE_CUTOFF = 4.0

#: Standard X-H bond lengths (Å), used when placing hydrogens.
XH_BOND_LENGTH = {7: 1.01, 8: 0.96, 16: 1.34}

#: Tetrahedral angle, the default H-X-H opening.
TETRAHEDRAL = math.radians(109.47)


# ---------------------------------------------------------------------------
# Standard heavy-atom templates (missing-atom detection)
# ---------------------------------------------------------------------------

#: Oxygen names that mean "the backbone carbonyl oxygen" in some files.
_O = ("O", "O1", "OT1")
_BB = ("N", "CA", "C", _O)

#: Heavy atoms of every standard residue. An entry may be a tuple of
#: alternatives, meaning "any one of these names satisfies the requirement".
STANDARD_RESIDUE_ATOMS: Dict[str, Tuple[Union[str, Tuple[str, ...]], ...]] = {
    "ALA": _BB + ("CB",),
    "ARG": _BB + ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": _BB + ("CB", "CG", "OD1", "ND2"),
    "ASP": _BB + ("CB", "CG", "OD1", "OD2"),
    "CYS": _BB + ("CB", "SG"),
    "GLN": _BB + ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": _BB + ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": _BB,
    "HIS": _BB + ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": _BB + ("CB", "CG1", "CG2", "CD1"),
    "LEU": _BB + ("CB", "CG", "CD1", "CD2"),
    "LYS": _BB + ("CB", "CG", "CD", "CE", "NZ"),
    "MET": _BB + ("CB", "CG", "SD", "CE"),
    "PHE": _BB + ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": _BB + ("CB", "CG", "CD"),
    "SER": _BB + ("CB", "OG"),
    "THR": _BB + ("CB", "OG1", "CG2"),
    "TRP": _BB + ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": _BB + ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": _BB + ("CB", "CG1", "CG2"),
    # Modified residues with a different atom set.
    "MSE": _BB + ("CB", "CG", "SE", "CE"),
    "SEC": _BB + ("CB", "SE"),
}

_SUGAR = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")
_SUGAR_D = ("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'")

STANDARD_RESIDUE_ATOMS.update(
    {
        "A": _SUGAR + ("O2'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
        "G": _SUGAR + ("O2'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
        "C": _SUGAR + ("O2'", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
        "U": _SUGAR + ("O2'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
        "I": _SUGAR + ("O2'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N3", "C4"),
        "DA": _SUGAR_D + ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
        "DG": _SUGAR_D + ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
        "DC": _SUGAR_D + ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
        "DT": _SUGAR_D + ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", ("C5M", "C7")),
        "DI": _SUGAR_D + ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N3", "C4"),
        "PSU": _SUGAR + ("O2'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", ("C5M", "C7"), "C6"),
        "5MC": _SUGAR_D + ("N1", "C2", "O2", "N3", "C4", "N4", "C5", ("C5M", "C7"), "C6"),
        "1MA": _SUGAR + ("O2'", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
        "7MG": _SUGAR + ("O2'", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
        "5MU": _SUGAR + ("O2'", "N1", "C2", "O2", "N3", "C4", "O4", "C5", ("C5M", "C7"), "C6"),
        "ADE": ("N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
        "GUA": ("N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
        "CYT": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
        "THY": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", ("C5M", "C7")),
        "URI": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    }
)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def _atomic_weight(mol, idx: int) -> float:
    """Average atomic weight of one atom (0.0 for an element RDKit cannot name)."""
    try:
        table = Chem.GetPeriodicTable()
        weight = float(table.GetAtomicWeight(mol.GetAtomWithIdx(idx).GetAtomicNum()))
        if weight > 0:
            return weight
    except Exception:  # pragma: no cover - defensive
        pass
    return 0.0


def _residue_key(mol, idx: int):
    """``(chain, res_id, insertion, name)`` of an atom, or ``None`` without info."""
    info = mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
    if info is None:
        return None
    name = info.GetResidueName().strip().upper()
    if not name:
        return None
    return (
        info.GetChainId().strip(),
        int(info.GetResidueNumber()),
        info.GetInsertionCode().strip(),
        name,
    )


def _group_residues(mol) -> Dict[Tuple[str, int, str, str], List[int]]:
    """Group atom indices by residue, in order of first appearance."""
    groups: Dict[Tuple[str, int, str, str], List[int]] = {}
    for atom in mol.GetAtoms():
        key = _residue_key(mol, atom.GetIdx())
        if key is None:
            continue
        groups.setdefault(key, []).append(atom.GetIdx())
    return groups


def is_solvent_name(name: str) -> bool:
    """Whether a residue name denotes water (exact list or a known prefix)."""
    name = name.strip().upper()
    if name in SOLVENT_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in SOLVENT_PREFIXES)


@dataclass(frozen=True)
class HeteroResidue:
    """One non-polymer residue of a structure.

    ``mass`` is the sum of the IUPAC average atomic weights of the *heavy*
    atoms only, because that is the quantity the brief compares against the
    150 Da ligand threshold (hydrogens are absent from most crystal structures
    and would make the test inconsistent between files).
    """

    name: str
    chain: str
    res_id: int
    kind: str
    atom_indices: Tuple[int, ...]
    heavy_atoms: int
    mass: float

    @property
    def label(self) -> str:
        """``"BEN A1"`` — the name a user sees in the GUI and in the log.

        The insertion code is deliberately not part of the label so that it
        matches the frozen interface; :attr:`atom_indices` still identifies the
        residue unambiguously.
        """
        return f"{self.name} {self.chain}{self.res_id}"


@dataclass
class HeteroInventory:
    """The non-polymer residues of a structure, sorted by chain and residue id."""

    residues: List[HeteroResidue] = field(default_factory=list)

    def of_kind(self, kind: str) -> List[HeteroResidue]:
        """Every residue of one kind, in inventory order."""
        return [res for res in self.residues if res.kind == kind]

    def labels(self, kind: Optional[str] = None) -> List[str]:
        """The labels, optionally restricted to one kind."""
        return [
            res.label for res in self.residues if kind is None or res.kind == kind
        ]

    def summary(self) -> Dict[str, int]:
        """``{kind: count}`` for every kind, zeros included.

        All five keys are always present so that a caller (the GUI's residue
        tree, a report table) can index the result without a key check.
        """
        counts = {kind: 0 for kind in KINDS}
        for res in self.residues:
            counts[res.kind] = counts.get(res.kind, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.residues)

    def __iter__(self) -> Iterable[HeteroResidue]:
        return iter(self.residues)


def _classify_residue(mol, name: str, heavy: Sequence[int]) -> Optional[str]:
    """Kind of one residue, or ``None`` when it is part of the polymer.

    The order of the tests is the policy: a name that is both a possible ligand
    and a cofactor (HEM, NAD, ...) must be protected before the mass test can
    call it a ligand, and solvent must be recognised before an ion.
    """
    if is_solvent_name(name):
        return KIND_SOLVENT
    if name in COFACTOR_NAMES:
        return KIND_COFACTOR
    if name in METAL_CATION_NAMES or name in COUNTERION_NAMES:
        return KIND_ION
    if name in POLYMER_RESIDUES:
        return None
    mass = sum(_atomic_weight(mol, idx) for idx in heavy)
    if len(heavy) > 1 and mass > LIGAND_MIN_MASS:
        return KIND_LIGAND
    # Everything else is neither polymer nor a convincing ligand: a single heavy
    # atom, a tiny fragment, an unmodelled density blob. It is reported so the
    # user can see it, and it is kept by the cleaning policy.
    return KIND_OTHER


def classify_hetero(mol) -> HeteroInventory:
    """Classify every non-polymer residue of `mol` into the five kinds.

    Atoms without PDB residue information (an SDF, a bare SMILES) have no
    residues to classify; the result is then an empty inventory. Modified
    residues and standard nucleotides are polymer and stay out of the inventory
    entirely.
    """
    require_rdkit()
    residues: List[HeteroResidue] = []
    for key, indices in _group_residues(mol).items():
        chain, res_id, _insertion, name = key
        heavy = [
            idx for idx in indices if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
        ]
        kind = _classify_residue(mol, name, heavy)
        if kind is None:
            continue
        mass = sum(_atomic_weight(mol, idx) for idx in heavy)
        residues.append(
            HeteroResidue(
                name=name,
                chain=chain,
                res_id=res_id,
                kind=kind,
                atom_indices=tuple(indices),
                heavy_atoms=len(heavy),
                mass=mass,
            )
        )
    residues.sort(key=lambda res: (res.chain, res.res_id, res.name))
    return HeteroInventory(residues=residues)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


@dataclass
class CleanPolicy:
    """What :func:`clean_receptor` is allowed to delete.

    ``keep_residues`` is the escape hatch for the "keep the catalytic metal"
    workflow: a residue is protected either by its label (``"ZN A301"``) or by
    its name (``"ZN"``, every zinc of the structure). Protection wins over every
    drop flag.
    """

    drop_solvent: bool = True
    #: Keep only waters whose oxygen is within this many Å of a ligand atom.
    #: ``None`` drops every water. Only meaningful together with ``drop_solvent``.
    keep_water_within: Optional[float] = None
    drop_ligands: bool = True
    drop_free_ions: bool = True
    keep_cofactors: bool = True
    #: Labels ("ZN A301") or names ("ZN") that must never be removed.
    keep_residues: Sequence[str] = ()
    #: Labels ("BEN A1") or names ("BEN") that must be removed whatever their
    #: classification. This is what lets a user strip a co-crystallised ligand
    #: that is too small for the 150 Da rule — benzamidine in 3PTB is 112 Da and
    #: is therefore *not* a ``KIND_LIGAND``, but a user still wants it gone.
    #: Removal wins over ``keep_residues`` and over every other flag.
    drop_residues: Sequence[str] = ()


def _protected_set(keep_residues: Sequence[str]) -> set:
    """Normalise the protection list so "ZN A301", "zna301" and "ZN" all match."""
    out = set()
    for entry in keep_residues or ():
        text = "".join(str(entry).split()).upper()
        if text:
            out.add(text)
    return out


def _is_protected(res: HeteroResidue, protected: set) -> bool:
    if not protected:
        return False
    if res.name.upper() in protected:
        return True
    return "".join(res.label.split()).upper() in protected


def _coords(mol, indices: Sequence[int]) -> np.ndarray:
    """``(n, 3)`` coordinates of the given atom indices."""
    conf = mol.GetConformer()
    out = np.empty((len(indices), 3), dtype=float)
    for row, idx in enumerate(indices):
        point = conf.GetAtomPosition(int(idx))
        out[row] = (point.x, point.y, point.z)
    return out


def _remove_atoms(mol, indices: Iterable[int]):
    """Copy of `mol` without the given atoms, conformer and residue info kept."""
    drop = sorted({int(i) for i in indices}, reverse=True)
    if not drop:
        return Chem.Mol(mol)
    em = Chem.RWMol(mol)
    for idx in drop:
        em.RemoveAtom(idx)
    out = em.GetMol()
    # Sanitisation is best effort: an unsanitised protein is a legitimate input
    # and a valence the default model rejects must not abort the cleaning.
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.FastFindRings(out)
        except Exception:  # pragma: no cover - defensive
            pass
    return out


def _waters_kept(mol, waters: Sequence[HeteroResidue], ligand_atoms: Sequence[int], cutoff: float):
    """Split waters into (kept, dropped) by distance from the ligand.

    The distance is measured from each water *oxygen* to the closest ligand
    atom, which is the definition in the brief ("structural waters": a water
    that hydrogen-bonds the ligand). A water without an oxygen cannot satisfy
    the test and is dropped.
    """
    if not ligand_atoms or mol.GetNumConformers() == 0:
        return [], list(waters)
    ligand_xyz = _coords(mol, ligand_atoms)
    cutoff2 = float(cutoff) ** 2
    kept: List[HeteroResidue] = []
    dropped: List[HeteroResidue] = []
    for res in waters:
        oxygens = [
            idx for idx in res.atom_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() == 8
        ]
        close = False
        if oxygens:
            xyz = _coords(mol, oxygens)
            # (n_oxygens, n_ligand) squared distances, minimised over both axes.
            delta = xyz[:, None, :] - ligand_xyz[None, :, :]
            close = bool((np.einsum("ijk,ijk->ij", delta, delta) <= cutoff2).any())
        (kept if close else dropped).append(res)
    return kept, dropped


def clean_receptor(mol, policy: Optional[CleanPolicy] = None):
    """Apply a :class:`CleanPolicy` to `mol`.

    Returns
    -------
    ``(cleaned_mol, log_lines, inventory)`` — the cleaned molecule, human
    readable log lines (one per decision) and the inventory of the *input*, so
    that the caller can still show the user what was in the file.
    """
    require_rdkit()
    policy = policy if policy is not None else CleanPolicy()
    if policy.keep_water_within is not None and float(policy.keep_water_within) <= 0:
        raise ValueError(
            f"keep_water_within must be positive, got {policy.keep_water_within}"
        )

    inventory = classify_hetero(mol)
    if not inventory.residues:
        return (
            Chem.Mol(mol),
            ["no non-standard residues found; the receptor is unchanged"],
            inventory,
        )

    protected = _protected_set(policy.keep_residues)
    forced = _protected_set(policy.drop_residues)
    ligand_residues = inventory.of_kind(KIND_LIGAND)
    ligand_atoms = [idx for res in ligand_residues for idx in res.atom_indices]
    ligand_names = ", ".join(res.label for res in ligand_residues)

    drop: List[int] = []
    log: List[str] = []

    # Residues the user named explicitly are removed first, whatever they are.
    if forced:
        eaten = [
            res
            for res in inventory.residues
            if "".join(res.label.split()).upper() in forced
            or res.name.upper() in forced
        ]
        if eaten:
            drop.extend(idx for res in eaten for idx in res.atom_indices)
            log.append(
                "removed "
                + ", ".join(f"{res.label} ({res.kind})" for res in eaten)
                + " on request"
            )
            named = set(eaten)
            inventory = HeteroInventory(
                residues=[res for res in inventory.residues if res not in named]
            )
            ligand_residues = inventory.of_kind(KIND_LIGAND)
            ligand_atoms = [idx for res in ligand_residues for idx in res.atom_indices]
            ligand_names = ", ".join(res.label for res in ligand_residues)

    waters = inventory.of_kind(KIND_SOLVENT)
    if policy.drop_solvent and waters:
        if policy.keep_water_within is None:
            drop.extend(idx for res in waters for idx in res.atom_indices)
            log.append(
                f"removed {len(waters)} solvent residues "
                f"({sum(len(res.atom_indices) for res in waters)} atoms)"
            )
        elif not ligand_atoms:
            drop.extend(idx for res in waters for idx in res.atom_indices)
            log.append(
                f"removed {len(waters)} solvent residues: no ligand is present to "
                f"measure the {policy.keep_water_within:g} A distance against"
            )
        elif mol.GetNumConformers() == 0:
            drop.extend(idx for res in waters for idx in res.atom_indices)
            log.append(
                f"removed {len(waters)} solvent residues: the molecule has no 3-D "
                "conformer, so structural waters cannot be identified"
            )
        else:
            kept, gone = _waters_kept(
                mol, waters, ligand_atoms, float(policy.keep_water_within)
            )
            drop.extend(idx for res in gone for idx in res.atom_indices)
            log.append(
                f"kept {len(kept)} structural waters within "
                f"{policy.keep_water_within:g} A of {ligand_names}, "
                f"removed {len(gone)}"
            )

    for res in inventory.residues:
        if res.kind == KIND_SOLVENT:
            continue
        if _is_protected(res, protected):
            log.append(f"kept {res.label} (protected by the keep_residues policy)")
            continue
        if res.kind == KIND_LIGAND:
            if policy.drop_ligands:
                drop.extend(res.atom_indices)
                log.append(
                    f"removed co-crystal ligand {res.label} "
                    f"({res.heavy_atoms} heavy atoms, {res.mass:.1f} Da)"
                )
            else:
                log.append(f"kept co-crystal ligand {res.label} (drop_ligands=False)")
        elif res.kind == KIND_ION:
            origin = "metal cation" if res.name in METAL_CATION_NAMES else "counter-ion"
            if policy.drop_free_ions:
                drop.extend(res.atom_indices)
                log.append(f"removed free ion {res.label} ({origin})")
            else:
                log.append(f"kept ion {res.label} ({origin})")
        elif res.kind == KIND_COFACTOR:
            if policy.keep_cofactors:
                log.append(f"kept cofactor {res.label} ({res.heavy_atoms} heavy atoms)")
            else:
                drop.extend(res.atom_indices)
                log.append(f"removed cofactor {res.label} (keep_cofactors=False)")
        else:  # KIND_OTHER
            log.append(
                f"kept unclassified non-standard residue {res.label} "
                f"({res.heavy_atoms} heavy atoms, {res.mass:.1f} Da)"
            )

    cleaned = _remove_atoms(mol, drop)
    if drop:
        log.insert(
            0,
            f"cleaning removed {len(drop)} of {mol.GetNumAtoms()} atoms; "
            f"{cleaned.GetNumAtoms()} remain",
        )
    return cleaned, log, inventory


def extract_ligands(mol, inventory: Optional[HeteroInventory] = None):
    """Every detected co-crystal ligand as its own molecule.

    Returns ``[(label, mol_of_that_residue), ...]`` so that the GUI can export
    the native ligand and use its coordinates as the self-docking reference
    pose. Coordinates, atom order and per-atom residue information are kept; the
    bond orders of a PDB-derived ligand are whatever the reader perceived, so a
    caller that needs exact chemistry must still supply a SMILES template
    (``odock.chem.ligand`` / ``prepare_ligand(smiles=...)``).
    """
    require_rdkit()
    inv = inventory if inventory is not None else classify_hetero(mol)
    out: List[Tuple[str, object]] = []
    for res in inv.of_kind(KIND_LIGAND):
        keep = set(res.atom_indices)
        em = Chem.RWMol(mol)
        for idx in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
            em.RemoveAtom(idx)
        sub = em.GetMol()
        try:
            sub.SetProp("_Name", res.label)
        except Exception:  # pragma: no cover - defensive
            pass
        out.append((res.label, sub))
    return out


# ---------------------------------------------------------------------------
# Missing atoms
# ---------------------------------------------------------------------------


def _template_names(template: Sequence[Union[str, Tuple[str, ...]]]):
    """Normalise a template into ``[(report_name, {accepted_names}), ...]``."""
    out = []
    for entry in template:
        if isinstance(entry, str):
            out.append((entry, {entry}))
        else:
            out.append((entry[0], set(entry)))
    return out


def _atom_name(mol, idx: int) -> str:
    info = mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
    if info is None:
        return ""
    return info.GetName().strip().upper()


def missing_atoms(mol) -> List[str]:
    """Residues whose standard heavy-atom set is incomplete.

    Each line reads ``"ASP A189 missing OD1, OD2"``: the residue label uses the
    same ``NAME CHAIN+RESID`` form as :attr:`HeteroResidue.label`, followed by
    the heavy atoms the template expects but the file does not contain, in
    chemical order. Hydrogens are ignored, ``OXT`` is optional (it only exists on
    a chain terminus) and atom-name alternatives such as ``O``/``O1``/``OT1``
    are accepted. Residues that are not in the template table (non-standard
    residues, ligands, ions) are skipped.
    """
    require_rdkit()
    out: List[Tuple[Tuple[str, int, str], str]] = []
    for key, indices in _group_residues(mol).items():
        chain, res_id, insertion, name = key
        template = STANDARD_RESIDUE_ATOMS.get(name)
        if template is None:
            continue
        present = {_atom_name(mol, idx) for idx in indices}
        missing = [
            report
            for report, accepted in _template_names(template)
            if not (accepted & present)
        ]
        if missing:
            label = f"{name} {chain}{res_id}"
            out.append(((chain, res_id, insertion), f"{label} missing {', '.join(missing)}"))
    out.sort(key=lambda item: item[0])
    return [line for _, line in out]


# ---------------------------------------------------------------------------
# pH-aware protonation
# ---------------------------------------------------------------------------


@dataclass
class _Residue:
    """Internal view of one residue: its key plus ``atom name -> index``."""

    name: str
    chain: str
    res_id: int
    insertion: str
    atoms: Dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain, self.res_id, self.insertion)

    @property
    def label(self) -> str:
        return f"{self.name} {self.chain}{self.res_id}"


def _residue_views(mol) -> List[_Residue]:
    """Every residue with PDB information, in molecule order."""
    views: Dict[Tuple[str, int, str, str], _Residue] = {}
    for atom in mol.GetAtoms():
        key = _residue_key(mol, atom.GetIdx())
        if key is None:
            continue
        chain, res_id, insertion, name = key
        view = views.get(key)
        if view is None:
            view = _Residue(name=name, chain=chain, res_id=res_id, insertion=insertion)
            views[key] = view
        atom_name = _atom_name(mol, atom.GetIdx())
        if atom_name and atom_name not in view.atoms:
            view.atoms[atom_name] = atom.GetIdx()
    return list(views.values())


def _is_amino_acid(name: str) -> bool:
    return name in STANDARD_AMINO_ACIDS or name in MODIFIED_AMINO_ACIDS


def _heavy_neighbours(mol, idx: int) -> List[int]:
    return [
        n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() > 1
    ]


def _h_neighbours(mol, idx: int) -> List[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1]


def _distance(mol, i: int, j: int) -> float:
    conf = mol.GetConformer()
    a = conf.GetAtomPosition(int(i))
    b = conf.GetAtomPosition(int(j))
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _kekule_mol(mol):
    """A copy of `mol` with integer bond orders, for valence arithmetic.

    Aromatic perception mixes single and double bonds into a bond order of 1.5,
    which makes a chemical valence wrong: a pyrrole nitrogen with two aromatic
    bonds and a hydrogen sums to 4 and would look like an ammonium. Kekulising a
    *copy* (the returned molecule keeps its aromatic flags, which the AD4
    ``A`` typing depends on) gives the integer orders the valence rules need.
    """
    work = Chem.Mol(mol)
    try:
        Chem.Kekulize(work, clearAromaticFlags=True)
    except Exception:
        try:
            Chem.SanitizeMol(work)
            Chem.Kekulize(work, clearAromaticFlags=True)
        except Exception:  # pragma: no cover - an input RDKit cannot kekulise
            pass
    return work


def _bond_order(kek, i: int, j: int) -> float:
    bond = kek.GetBondBetweenAtoms(int(i), int(j))
    return 0.0 if bond is None else float(bond.GetBondTypeAsDouble())


def _heavy_valence(kek) -> Dict[int, float]:
    """Sum of the integer bond orders of every atom (heavy neighbours only)."""
    return {
        atom.GetIdx(): sum(
            b.GetBondTypeAsDouble()
            for b in atom.GetBonds()
            if b.GetOtherAtom(atom).GetAtomicNum() > 1
        )
        for atom in kek.GetAtoms()
    }


def _fused_ring_bonds(probe, seeds: Iterable[int]):
    """Bond and atom indices of the fused ring system(s) containing `seeds`.

    A histidine imidazole is one ring, but the indole of a tryptophan is two
    fused rings sharing an edge, and a kekule flip started in one of them has to
    be solved over the whole system: moving a double bond onto a fusion atom
    without moving the one on the other side would leave that atom pentavalent.
    """
    info = probe.GetRingInfo()
    atom_rings = [set(r) for r in info.AtomRings()]
    bond_rings = [list(b) for b in info.BondRings()]
    if len(atom_rings) != len(bond_rings):  # pragma: no cover - defensive
        return set(), set()
    seed_set = set(seeds)
    selected = [i for i, ring in enumerate(atom_rings) if ring & seed_set]
    changed = True
    while changed:
        changed = False
        for i, ring in enumerate(atom_rings):
            if i in selected:
                continue
            if any(len(ring & atom_rings[j]) >= 2 for j in selected):
                selected.append(i)
                changed = True
    bonds: set = set()
    atoms: set = set()
    for i in selected:
        bonds.update(bond_rings[i])
        atoms.update(atom_rings[i])
    return bonds, atoms


def _rekekulize_rings(work, seeds: Sequence[int], zero_h: Sequence[int]):
    """Re-derive the kekule structure of the rings containing `seeds`.

    Used when a hydrogen has been placed on a ring nitrogen that the perceived
    bond orders had given a double bond: the nitrogen would be four-valent and
    neutral, which does not exist. Marking the ring system aromatic and letting
    RDKit re-kekulise — with the atoms that carry no hydrogen explicitly marked
    `NoImplicit` so the valence model cannot invent one — yields the bond orders
    that match the new hydrogens (the *other* ring nitrogen takes the double
    bond, which is exactly the HID/HIE distinction).

    Returns a new molecule, or ``None`` when RDKit cannot re-kekulise the
    system. The caller keeps its original molecule in that case.
    """
    try:
        probe = Chem.RWMol(work)
        probe.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(probe)
        bonds, atoms = _fused_ring_bonds(probe, seeds)
        if not bonds:
            return None
        for idx in zero_h:
            if 0 <= int(idx) < probe.GetNumAtoms():
                probe.GetAtomWithIdx(int(idx)).SetNoImplicit(True)
        for bond_idx in bonds:
            bond = probe.GetBondWithIdx(int(bond_idx))
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
        for idx in atoms:
            probe.GetAtomWithIdx(int(idx)).SetIsAromatic(True)
        Chem.Kekulize(probe, clearAromaticFlags=True)
        return probe
    except Exception:
        return None


def _is_acceptor_atom(atom) -> bool:
    """A minimal, ring-info-free H-bond acceptor test.

    Used for the histidine environment. It is intentionally generous: any
    neutral N/O/S counts. The decision only needs the *ranking* of nearby polar
    atoms, and a generous acceptor list makes "the carboxylate is the closest
    acceptor" a stricter test rather than a looser one.
    """
    if atom.GetAtomicNum() not in (7, 8, 16):
        return False
    return atom.GetFormalCharge() <= 0


def _histidine_state(mol, res: _Residue, ph: float):
    """``(tautomer, reason)`` for one histidine.

    The rule implemented is the one in the brief:

    * pH below the imidazole pKa (6.0) protonates both nitrogens -> ``HIP``;
    * otherwise a nearby Asp/Glu carboxylate inside 4 Å that is the *closest*
    acceptor to a ring nitrogen means a strong, charge-assisted H-bond ->
      ``HIP``;
    * otherwise the tautomer is the one whose nitrogen has an H-bond partner:
      a partner at ND1 only -> ``HID`` (the proton sits on ND1), at NE2 only ->
      ``HIE``;
    * with no partner at all the physiological default ``HIE`` is used.
    """
    if ph < PKA["HIS"]:
        return "HIP", f"pH {ph:g} < pKa {PKA['HIS']:g}, both ring nitrogens protonated"

    nd1 = res.atoms.get("ND1")
    ne2 = res.atoms.get("NE2")
    if nd1 is None or ne2 is None or mol.GetNumConformers() == 0:
        return "HIE", "default tautomer (no 3-D environment to inspect)"

    carboxylate_names = {"OD1", "OD2", "OE1", "OE2"}
    carboxylate_residues = {"ASP", "GLU", "ASX", "GLX"}
    sites = {"ND1": nd1, "NE2": ne2}
    # The nearest acceptor to each ring nitrogen, searched over the wider
    # carboxylate radius; the two rules below then apply their own cutoff.
    nearest = {}
    for site, idx in sites.items():
        best = None
        for atom in mol.GetAtoms():
            if atom.GetIdx() in (nd1, ne2) or not _is_acceptor_atom(atom):
                continue
            dist = _distance(mol, idx, atom.GetIdx())
            if dist > HIS_CARBOXYLATE_CUTOFF:
                continue
            if best is None or dist < best[0]:
                best = (dist, atom)
        nearest[site] = best

    for site, best in nearest.items():
        if best is None:
            continue
        dist, atom = best
        info = atom.GetPDBResidueInfo()
        residue = info.GetResidueName().strip().upper() if info is not None else ""
        atom_name = info.GetName().strip().upper() if info is not None else ""
        if residue in carboxylate_residues and atom_name in carboxylate_names:
            chain = info.GetChainId().strip() if info is not None else ""
            resid = int(info.GetResidueNumber()) if info is not None else 0
            return (
                "HIP",
                f"{residue}{chain}{resid} {atom_name} at {dist:.2f} A is the closest "
                f"acceptor to {site} (limit {HIS_CARBOXYLATE_CUTOFF:g} A)",
            )

    partners = {
        site: (best[0] if best is not None and best[0] <= HIS_PARTNER_CUTOFF else None)
        for site, best in nearest.items()
    }
    if partners["ND1"] is not None and partners["NE2"] is None:
        return "HID", f"partner at ND1 ({partners['ND1']:.2f} A) only"
    if partners["NE2"] is not None and partners["ND1"] is None:
        return "HIE", f"partner at NE2 ({partners['NE2']:.2f} A) only"
    if partners["ND1"] is not None and partners["NE2"] is not None:
        return "HIE", "partners at both ring nitrogens; physiological default HIE"
    return (
        "HIE",
        f"no H-bond partner within {HIS_PARTNER_CUTOFF:g} A; physiological default HIE",
    )


def _carboxylate_hydrogens(
    mol, res: _Residue, names: Sequence[str], protonated: bool, kek
):
    """Which carboxylate oxygen carries the proton (and which carries the -1).

    When the state is protonated exactly one oxygen carries a hydrogen; when it
    is deprotonated exactly one oxygen carries the negative charge. Both must
    sit on a **single-bonded** oxygen: the double-bonded one is the carbonyl and
    is neutral in either state. Within that constraint an oxygen that already
    has a hydrogen wins, then the conventional naming (OD2 / OE2).
    """
    present = [name for name in names if name in res.atoms]
    if not present:
        return None, None
    with_h = [name for name in present if _h_neighbours(mol, res.atoms[name])]
    single = [
        name
        for name in present
        if max(
            (
                b.GetBondTypeAsDouble()
                for b in kek.GetAtomWithIdx(res.atoms[name]).GetBonds()
            ),
            default=1.0,
        )
        < 1.5
    ]
    if protonated:
        if with_h:
            return with_h[0], None
        if single:
            return single[-1], None
        return present[-1], None
    without_h = [name for name in present if name not in with_h]
    for name in single:
        if name in without_h:
            return None, name
    return None, (without_h[0] if without_h else present[0])


def _protein_targets(
    mol,
    res: _Residue,
    ph: float,
    n_terminal: bool,
    c_terminal: bool,
    his_state: Optional[str],
    kek,
    charges: Dict[int, int],
    fallbacks: List[Tuple[int, Tuple[int, ...]]],
    charge_protected_n: set,
    log: List[str],
) -> Dict[int, int]:
    """Desired hydrogen count per heteroatom of one amino-acid residue.

    `charges` receives the formal charges that follow from the state alone (the
    carboxylate -1, the lysine and N-terminal ammonium +1). `fallbacks` receives
    ``(atom, group)`` pairs for the two delocalised cations — the guanidinium
    and the imidazolium — whose +1 is only booked on the conventional atom when
    the valence of the group does not identify the charged nitrogen by itself.
    `charge_protected_n` lists the nitrogens that are *meant* to be four-valent
    (an ammonium or an iminium); their kekule structure must not be rewritten.
    """
    targets: Dict[int, int] = {}
    name = res.name
    # --- backbone ---------------------------------------------------------
    if "N" in res.atoms:
        idx = res.atoms["N"]
        degree = len(_heavy_neighbours(mol, idx))
        if n_terminal:
            protonated = ph < PKA["N_TERM"]
            # A neutral amine carries 3 - degree hydrogens; the ammonium carries
            # one more. N-terminal proline therefore gets 2 (not 3) hydrogens,
            # because its nitrogen already has two heavy neighbours.
            targets[idx] = max(0, 3 - degree) + (1 if protonated else 0)
            if protonated:
                charges[idx] = 1
                charge_protected_n.add(idx)
            log.append(
                f"{res.label}: N-terminus {'+1' if protonated else '0'} "
                f"({'NH3+' if protonated else 'NH2'}, pH {ph:g} "
                f"{'<' if protonated else '>='} pKa {PKA['N_TERM']:g})"
            )
        else:
            targets[idx] = max(0, 3 - degree)
    if "O" in res.atoms:
        targets[res.atoms["O"]] = 0  # backbone carbonyl

    # --- side chain -------------------------------------------------------
    if name in ("ASP", "GLU"):
        protonated = ph < PKA[name]
        names = ("OD1", "OD2") if name == "ASP" else ("OE1", "OE2")
        present = [atom_name for atom_name in names if atom_name in res.atoms]
        if present:
            donor, acceptor = _carboxylate_hydrogens(mol, res, names, protonated, kek)
            for atom_name in present:
                targets[res.atoms[atom_name]] = 0
            if protonated and donor is not None:
                targets[res.atoms[donor]] = 1
            elif not protonated and acceptor is not None:
                charges[res.atoms[acceptor]] = -1
            log.append(
                f"{res.label}: {0 if protonated else -1} "
                f"({'COOH' if protonated else 'COO-'}, pH {ph:g} "
                f"{'<' if protonated else '>='} pKa {PKA[name]:g})"
            )
    elif name == "LYS" and "NZ" in res.atoms:
        protonated = ph < PKA["LYS"]
        targets[res.atoms["NZ"]] = 3 if protonated else 2
        if protonated:
            charges[res.atoms["NZ"]] = 1
            charge_protected_n.add(res.atoms["NZ"])
        log.append(
            f"{res.label}: {'+1' if protonated else '0'} "
            f"({'NH3+' if protonated else 'NH2'}, pH {ph:g} "
            f"{'<' if protonated else '>='} pKa {PKA['LYS']:g})"
        )
    elif name == "ARG":
        protonated = ph < PKA["ARG"]
        if "NE" in res.atoms:
            targets[res.atoms["NE"]] = 1
        terminals = [n for n in ("NH1", "NH2") if n in res.atoms]
        # In the neutral guanidine one terminal nitrogen holds the double bond
        # and therefore only one hydrogen; the other is the NH2. Without
        # perceived bond orders the conventional NH1/NH2 assignment is used.
        cz = res.atoms.get("CZ")
        double_bonded = (
            [n for n in terminals if _bond_order(kek, res.atoms[n], cz) > 1.5]
            if cz is not None
            else []
        )
        if protonated:
            for atom_name in terminals:
                targets[res.atoms[atom_name]] = 2
        else:
            for atom_name in terminals:
                if double_bonded:
                    targets[res.atoms[atom_name]] = 1 if atom_name in double_bonded else 2
                else:
                    targets[res.atoms[atom_name]] = 1 if atom_name == "NH1" else 2
        # The guanidinium charge only has to be booked by hand when the valence
        # of the group does not identify the charged nitrogen by itself. Only
        # the three guanidinium nitrogens belong to the group: the backbone
        # nitrogen can be a charged N-terminus and must not mask the test.
        if protonated and "CZ" in res.atoms:
            group = tuple(
                res.atoms[n] for n in ("NE", "NH1", "NH2") if n in res.atoms
            )
            fallbacks.append((res.atoms["CZ"], group))
            charge_protected_n.update(group)
        log.append(
            f"{res.label}: {'+1' if protonated else '0'} "
            f"(guanidinium, pH {ph:g} "
            f"{'<' if protonated else '>='} pKa {PKA['ARG']:g})"
        )
    elif name in HIS_RESIDUE_NAMES:
        state = his_state or "HIE"
        if "ND1" in res.atoms:
            targets[res.atoms["ND1"]] = 1 if state in ("HID", "HIP") else 0
        if "NE2" in res.atoms:
            targets[res.atoms["NE2"]] = 1 if state in ("HIE", "HIP") else 0
        if state == "HIP" and "NE2" in res.atoms:
            # The imidazolium charge is delocalised. When the valence cannot
            # decide — an all-single-bond input — the conventional NE2 carries
            # the formal +1 (documented choice); with perceived bond orders the
            # four-valent ring nitrogen gets it instead. Only the two ring
            # nitrogens form the group: the backbone nitrogen may itself be a
            # charged N-terminus.
            group = tuple(
                res.atoms[n] for n in ("ND1", "NE2") if n in res.atoms
            )
            fallbacks.append((res.atoms["NE2"], group))
            charge_protected_n.update(group)
    elif name == "CYS" and "SG" in res.atoms:
        protonated = ph < PKA["CYS"]
        targets[res.atoms["SG"]] = 1 if protonated else 0
        if not protonated:
            charges[res.atoms["SG"]] = -1
        log.append(
            f"{res.label}: {'0' if protonated else '-1'} "
            f"({'SH' if protonated else 'S-'}, pH {ph:g} "
            f"{'<' if protonated else '>='} pKa {PKA['CYS']:g})"
        )
    elif name == "TYR" and "OH" in res.atoms:
        protonated = ph < PKA["TYR"]
        targets[res.atoms["OH"]] = 1 if protonated else 0
        if not protonated:
            charges[res.atoms["OH"]] = -1
        log.append(
            f"{res.label}: {'0' if protonated else '-1'} "
            f"({'OH' if protonated else 'O-'}, pH {ph:g} "
            f"{'<' if protonated else '>='} pKa {PKA['TYR']:g})"
        )
    elif name == "SER" and "OG" in res.atoms:
        targets[res.atoms["OG"]] = 1
    elif name == "THR" and "OG1" in res.atoms:
        targets[res.atoms["OG1"]] = 1
    elif name == "ASN" and "ND2" in res.atoms:
        targets[res.atoms["ND2"]] = 2
    elif name == "GLN" and "NE2" in res.atoms:
        targets[res.atoms["NE2"]] = 2
    elif name == "TRP" and "NE1" in res.atoms:
        targets[res.atoms["NE1"]] = 1
    elif name == "MET" and "SD" in res.atoms:
        targets[res.atoms["SD"]] = 0

    # --- C-terminus -------------------------------------------------------
    if c_terminal and "OXT" in res.atoms:
        protonated = ph < PKA["C_TERM"]
        targets[res.atoms["OXT"]] = 1 if protonated else 0
        if not protonated:
            charges[res.atoms["OXT"]] = -1
        log.append(
            f"{res.label}: C-terminus {'0' if protonated else '-1'} "
            f"({'COOH' if protonated else 'COO-'}, pH {ph:g} "
            f"{'<' if protonated else '>='} pKa {PKA['C_TERM']:g})"
        )
    return targets


def _generic_target(mol, idx: int) -> int:
    """Hydrogen count for a heteroatom without a residue-specific rule.

    This is the documented valence rule used for cofactors, modified residues
    and any ligand handed to :func:`protonate`:

    * **N** -- ``3 - heavy_neighbours`` (primary amine 2, secondary 1, tertiary
      0, and a primary amide nitrogen 2, which is correct because the peptide
      resonance does not add a chemical neighbour);
    * **aromatic ring N** -- the input is trusted (a pyridine N has none, a
      pyrrole N has one) because a PDB file without hydrogens cannot tell the
      two apart reliably;
    * **O** -- two for a lone oxygen (water), one for a single heavy neighbour
      whose bond is longer than a carbonyl, none otherwise; without 3-D
      coordinates the input's own hydrogen count is kept, so no phantom
      hydrogen is placed inside the pocket and no real hydroxyl is stripped.
    * **S** -- one for a single heavy neighbour (thiol), none otherwise.
    """
    atom = mol.GetAtomWithIdx(idx)
    z = atom.GetAtomicNum()
    degree = len(_heavy_neighbours(mol, idx))
    if atom.GetFormalCharge() < 0:
        return 0
    if z == 7:
        in_ring = False
        try:
            in_ring = bool(atom.IsInRing())
        except Exception:  # pragma: no cover - ring info missing
            in_ring = False
        if in_ring and degree >= 2:
            return len(_h_neighbours(mol, idx))
        return max(0, 3 - degree)
    if z == 8:
        if degree == 0:
            return 2
        if degree > 1:
            return 0
        if mol.GetNumConformers() == 0:
            # Without coordinates a carbonyl cannot be told from a hydroxyl, so
            # the input's own hydrogen count is trusted rather than guessed.
            return len(_h_neighbours(mol, idx))
        heavy = _heavy_neighbours(mol, idx)[0]
        try:
            length = _distance(mol, idx, heavy)
        except Exception:  # pragma: no cover - defensive
            return 0
        return 1 if length > CARBONYL_CUTOFF else 0
    # sulfur
    return 1 if degree <= 1 else 0


def _hydrogen_directions(mol, idx: int, count: int) -> List[np.ndarray]:
    """Unit directions for `count` hydrogens added to atom `idx`.

    The primary direction is the negative sum of the neighbour unit vectors
    (the direction that best completes the coordination). Multi-hydrogen groups
    are spread on a cone at the tetrahedral angle; the azimuth is chosen from
    twelve samples so that the added hydrogens stay as far as possible from the
    existing heavy neighbours.
    """
    conf = mol.GetConformer()
    origin = conf.GetAtomPosition(int(idx))
    origin = np.array([origin.x, origin.y, origin.z], dtype=float)
    neighbours = []
    for heavy in _heavy_neighbours(mol, idx):
        point = conf.GetAtomPosition(int(heavy))
        delta = np.array([point.x, point.y, point.z], dtype=float) - origin
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            neighbours.append(delta / norm)
    if neighbours:
        direction = -np.sum(neighbours, axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            # Linear coordination (e.g. a nitrile-like N): any perpendicular.
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(neighbours[0] @ reference)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            direction = np.cross(neighbours[0], reference)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:  # pragma: no cover - numerically unreachable
                direction = np.array([1.0, 0.0, 0.0])
                norm = 1.0
        direction = direction / norm
    else:
        direction = np.array([1.0, 0.0, 0.0])

    # An orthonormal basis with `direction` as the first axis.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(direction, reference)
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(direction, e1)

    if count == 1:
        return [direction]

    def spread(azimuth: float) -> List[np.ndarray]:
        """`count` directions on a cone around `direction`."""
        out = []
        for k in range(count):
            phi = azimuth + 2.0 * math.pi * k / count
            out.append(
                math.cos(TETRAHEDRAL) * direction
                + math.sin(TETRAHEDRAL) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            )
        return out

    # Pick the azimuth whose hydrogens are furthest from the heavy neighbours.
    best, best_score = None, -1.0
    for step in range(12):
        candidate = spread(2.0 * math.pi * step / 12.0)
        score = min(
            float(np.linalg.norm(np.array(h) - neighbour))
            for h in candidate
            for neighbour in (neighbours or [np.array([0.0, 0.0, 1.0])])
        )
        if score > best_score:
            best, best_score = candidate, score
    return best or [direction]


def _add_hydrogens(work, additions: Dict[int, int], log: List[str]) -> int:
    """Append hydrogens with 3-D coordinates; returns how many were added."""
    if not additions:
        return 0
    has_conf = work.GetNumConformers() > 0
    total = 0
    for idx, count in additions.items():
        parent = work.GetAtomWithIdx(int(idx))
        element = parent.GetAtomicNum()
        length = XH_BOND_LENGTH.get(element, 1.0)
        if has_conf:
            directions = _hydrogen_directions(work, int(idx), count)
            parent_pos = work.GetConformer().GetAtomPosition(int(idx))
            base = np.array([parent_pos.x, parent_pos.y, parent_pos.z], dtype=float)
        else:
            directions = [np.array([1.0, 0.0, 0.0])] * count
            base = np.zeros(3)
        parent_info = parent.GetPDBResidueInfo()
        parent_name = parent_info.GetName().strip() if parent_info is not None else "X"
        for direction in directions:
            hydrogen = Chem.Atom(1)
            try:
                hydrogen.SetNoImplicit(True)
            except Exception:  # pragma: no cover - defensive
                pass
            info = Chem.AtomPDBResidueInfo()
            info.SetName(("H" + parent_name)[:4])
            if parent_info is not None:
                info.SetResidueName(parent_info.GetResidueName())
                info.SetChainId(parent_info.GetChainId())
                info.SetResidueNumber(parent_info.GetResidueNumber())
                info.SetIsHeteroAtom(parent_info.GetIsHeteroAtom())
            info.SetOccupancy(1.0)
            info.SetTempFactor(0.0)
            hydrogen.SetMonomerInfo(info)
            new_idx = work.AddAtom(hydrogen)
            work.AddBond(int(idx), new_idx, Chem.BondType.SINGLE)
            if has_conf:
                point = base + np.asarray(direction, dtype=float) * length
                work.GetConformer().SetAtomPosition(
                    new_idx, Point3D(float(point[0]), float(point[1]), float(point[2]))
                )
            total += 1
    log.append(f"added {total} polar hydrogens")
    return total


def protonate(mol, ph: float = 7.4):
    """pH-adaptive protonation of a receptor.

    Returns ``(mol, log_lines)``.

    The chemical state is decided by a Henderson-Hasselbalch criterion against
    the published side-chain pKa values (:data:`PKA`): a group carries its acidic
    proton when ``pH < pKa``.

    ==============  ==========================================================
    group           state
    ==============  ==========================================================
    Asp, Glu        COOH (pH < 3.9 / 4.3) or COO- with a formal -1
    His             HIP (pH < 6.0, or a carboxylate is the closest acceptor
                    within 4 Å), otherwise HID/HIE from the local environment
    Cys, Tyr        SH / OH (protonated) or S- / O- with a formal -1
    Lys             NH3+ with a formal +1, or neutral NH2 (pH >= 10.5)
    Arg             guanidinium +1 (booked on CZ), or the neutral guanidine
    N-terminus      NH3+ (+1) on the first residue of each chain
    C-terminus      COO- (-1) on OXT, or COOH below pH 3.1
    ==============  ==========================================================

    Hydrogens are added on N, O and S only, exactly the number the chosen state
    requires; hydrogens that contradict the state (a proton on a deprotonated
    carboxylate, a third hydrogen on a charged histidine) are removed. Every
    hydrogen bound to carbon is left untouched.

    Two details make the formal charges come out right rather than merely
    plausible:

    * a **kekule repair**. A crystal structure has no hydrogens, so RDKit's bond
      perception may put an imidazole double bond on the very nitrogen the
      chosen tautomer has to protonate — which would leave a four-valent neutral
      nitrogen. The affected ring system is re-kekulised with the hydrogens
      present, so the double bond moves to the other ring nitrogen: that *is*
      the HID/HIE distinction;
    * the **valence decides the charge**. Where the bond orders are known, a
      four-valent nitrogen is an ammonium or an iminium and takes the +1,
      wherever it sits in a delocalised group (the iminium nitrogen of an
      imidazolium, the double-bonded terminal nitrogen of a guanidinium). Only
      when the input carries no bond orders at all is the conventional atom
      charged instead (CZ for the guanidinium, NE2 for the imidazolium).

    Raises
    ------
    ValueError
        When `ph` is outside ``[0, 14]`` or not finite.
    """
    require_rdkit()
    try:
        ph = float(ph)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ph must be a number, got {ph!r}") from exc
    if not math.isfinite(ph) or not (0.0 <= ph <= 14.0):
        raise ValueError(f"ph must be within [0, 14], got {ph}")

    if mol.GetNumAtoms() == 0:
        return Chem.Mol(mol), ["the molecule is empty; nothing to protonate"]

    log: List[str] = []
    views = _residue_views(mol)
    if not views:
        log.append(
            "no PDB residue information: only the generic polar-hydrogen rule "
            "is applied (no pKa state can be assigned)"
        )

    # --- chain termini ----------------------------------------------------
    per_chain: Dict[str, List[_Residue]] = {}
    for view in views:
        per_chain.setdefault(view.chain, []).append(view)
    n_terminal: set = set()
    c_terminal: set = set()
    for _chain, chain_views in per_chain.items():
        amino = sorted(
            (v for v in chain_views if _is_amino_acid(v.name)),
            key=lambda v: (v.res_id, v.insertion),
        )
        if amino:
            n_terminal.add(amino[0].key)
            c_terminal.add(amino[-1].key)
    for view in views:
        if not _is_amino_acid(view.name):
            continue
        if "N" in view.atoms and len(_heavy_neighbours(mol, view.atoms["N"])) <= 1:
            n_terminal.add(view.key)  # a free amine, e.g. a truncated chain start
        if "OXT" in view.atoms:
            c_terminal.add(view.key)

    # --- per-residue targets ---------------------------------------------
    # Valences are first read from a kekulised copy of the *input*: it is what
    # tells a guanidinium's double-bonded nitrogen, a carboxylate's carbonyl
    # oxygen and an all-single-bond file apart. After the hydrogens have been
    # placed the same valences are recomputed on the modified molecule (see
    # below), because hydrogens change the kekule structure.
    kek = _kekule_mol(mol)
    targets: Dict[int, int] = {}
    charges: Dict[int, int] = {}
    fallbacks: List[Tuple[int, Tuple[int, ...]]] = []
    charge_protected_n: set = set()
    for view in views:
        if _is_amino_acid(view.name):
            his_state = None
            if view.name in HIS_RESIDUE_NAMES:
                his_state, reason = _histidine_state(mol, view, ph)
                if his_state in ("HID", "HIE", "HIP"):
                    log.append(f"{view.label}: {his_state} ({reason})")
            targets.update(
                _protein_targets(
                    mol,
                    view,
                    ph,
                    view.key in n_terminal,
                    view.key in c_terminal,
                    his_state,
                    kek,
                    charges,
                    fallbacks,
                    charge_protected_n,
                    log,
                )
            )

    # Any heteroatom that no rule has claimed: cofactors, modified residues,
    # ligand-like material and (below) molecules with no residue information.
    for view in views:
        for idx in view.atoms.values():
            if idx in targets:
                continue
            if mol.GetAtomWithIdx(idx).GetAtomicNum() not in (7, 8, 16):
                continue
            targets[idx] = _generic_target(mol, idx)
    for atom in mol.GetAtoms():
        if atom.GetIdx() in targets:
            continue
        if atom.GetAtomicNum() not in (7, 8, 16):
            continue
        if _residue_key(mol, atom.GetIdx()) is not None:
            continue  # already handled above
        targets[atom.GetIdx()] = _generic_target(mol, atom.GetIdx())

    # --- reconcile --------------------------------------------------------
    # The edits are made on the *kekulised* copy: an input that arrives with
    # perceived aromatic bonds (a sanitised MOL/SDF, or `complex_parts`) cannot
    # be re-kekulised while a ring nitrogen is still four-valent, while the
    # integer-bond copy always can.
    work = Chem.RWMol(kek)
    # Heteroatoms that must not carry a hydrogen: the kekule repair below needs
    # to know them, because a pyridine-type ring nitrogen is only recognisable
    # as such when the valence model cannot give it one implicitly.
    zero_h = {idx for idx, want in targets.items() if want == 0}
    removals: List[int] = []
    additions: Dict[int, int] = {}
    for idx, want in targets.items():
        have = _h_neighbours(work, idx)
        if want > len(have):
            additions[idx] = want - len(have)
        elif want < len(have):
            removals.extend(have[want:])
    if removals:
        drop = set(removals)
        # Heavy atoms keep their relative order, so the new index of a heavy
        # atom is its old index minus the number of removed atoms before it.
        shift = sorted(drop)

        def remap(old: int) -> int:
            return old - sum(1 for r in shift if r < old)

        additions = {
            remap(idx): count for idx, count in additions.items() if idx not in drop
        }
        charges = {
            remap(idx): value for idx, value in charges.items() if idx not in drop
        }
        targets = {
            remap(idx): count for idx, count in targets.items() if idx not in drop
        }
        fallbacks = [
            (remap(atom), tuple(remap(i) for i in group))
            for atom, group in fallbacks
            if atom not in drop
        ]
        charge_protected_n = {remap(i) for i in charge_protected_n if i not in drop}
        zero_h = {remap(i) for i in zero_h if i not in drop}
        for idx in sorted(drop, reverse=True):
            work.RemoveAtom(idx)
        log.append(f"removed {len(drop)} hydrogens that contradict the chosen state")

    if additions:
        _add_hydrogens(work, additions, log)

    # A hydrogen placed on a ring nitrogen that the perceived bond orders gave a
    # double bond would make that nitrogen four-valent and neutral, which does
    # not exist. The chemically correct answer is a kekule flip — the double
    # bond moves to the other ring nitrogen — and that is what the histidine
    # HID/HIE choice actually means. Nitrogens that are *meant* to be
    # four-valent (an ammonium, an iminium) are excluded: for them the valence
    # is turned into a formal charge below instead.
    valence = _heavy_valence(work)
    needs_flip = [
        int(idx)
        for idx, want in targets.items()
        if want > 0
        and 0 <= idx < work.GetNumAtoms()
        and work.GetAtomWithIdx(int(idx)).GetAtomicNum() == 7
        and int(idx) not in charge_protected_n
        and valence.get(idx, 0.0) + want >= 3.999
    ]
    if needs_flip:
        flipped = _rekekulize_rings(work, needs_flip, sorted(zero_h))
        if flipped is not None:
            work = flipped
            valence = _heavy_valence(work)
            log.append(
                f"re-kekulised {len(needs_flip)} ring nitrogen(s) so the new "
                "hydrogens do not leave a four-valent neutral nitrogen"
            )

    for atom in work.GetAtoms():
        try:
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
        except Exception:  # pragma: no cover - defensive
            pass

    for idx, value in charges.items():
        if 0 <= idx < work.GetNumAtoms():
            work.GetAtomWithIdx(int(idx)).SetFormalCharge(int(value))
    # A neutral nitrogen can never be four-valent. That is exactly how an
    # ammonium is told from an amine and an iminium from an imine, so where the
    # bond orders are known the formal charge follows from the valence instead
    # of from a convention: the +1 lands on the atom that really holds the extra
    # bond (the iminium nitrogen of an imidazolium, the double-bonded terminal
    # nitrogen of a guanidinium, the ammonium nitrogen of a lysine).
    charged_groups = set()
    for idx, want in targets.items():
        if not (0 <= idx < work.GetNumAtoms()):
            continue
        atom = work.GetAtomWithIdx(int(idx))
        if atom.GetAtomicNum() != 7:
            continue
        if valence.get(idx, 0.0) + want >= 3.999:
            atom.SetFormalCharge(1)
    for group in (group for _atom, group in fallbacks):
        if any(
            0 <= i < work.GetNumAtoms()
            and work.GetAtomWithIdx(int(i)).GetAtomicNum() == 7
            and work.GetAtomWithIdx(int(i)).GetFormalCharge() == 1
            for i in group
        ):
            charged_groups.add(group)
    for atom, group in fallbacks:
        # Only when the valence did not identify the charged nitrogen (an
        # all-single-bond input) is the conventional atom charged.
        if group in charged_groups or not (0 <= atom < work.GetNumAtoms()):
            continue
        work.GetAtomWithIdx(int(atom)).SetFormalCharge(1)

    out = work.GetMol()
    sanitized = True
    try:
        Chem.SanitizeMol(out)
    except Exception:
        sanitized = False
        try:
            Chem.FastFindRings(out)
        except Exception:  # pragma: no cover - defensive
            pass
    log.append(f"net formal charge: {Chem.GetFormalCharge(out):+d} (pH {ph:g})")
    if not sanitized:
        log.append(
            "warning: the protonated molecule could not be sanitised; its valence "
            "model is approximate and downstream typing may be degraded"
        )
    return out, log


# ---------------------------------------------------------------------------
# United-atom hydrogens
# ---------------------------------------------------------------------------


def _nonpolar_hydrogens(mol) -> List[int]:
    out = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if heavy and heavy[0].GetAtomicNum() == 6:
            out.append(atom.GetIdx())
    return out


def _gasteiger_values(mol) -> List[float]:
    """Gasteiger charges for `mol`, honouring an already published charge set."""
    from . import charges as _charges

    return [float(v) for v in _charges.gasteiger_charges(mol)]


def strip_nonpolar_hydrogens(mol, *, collapse_charges: bool = True):
    """Delete every hydrogen bound to carbon; return ``(mol, n_removed)``.

    This is the AutoDock united-atom step. With ``collapse_charges=True`` the
    removed hydrogen's partial charge is added onto its parent carbon, so the
    total charge of the molecule is conserved *exactly* (the charges are
    Gasteiger-Marsili values, either recomputed here or read back from a
    ``_GasteigerCharge`` property that a previous call published). The merged
    values are written onto the surviving atoms as ``_GasteigerCharge``, which
    is what :func:`odock.chem.charges.gasteiger_charges` reads, so the charge a
    PDBQT writer puts on the carbon is the united-atom charge rather than a
    freshly recomputed value for a graph that no longer has the hydrogens.

    With ``collapse_charges=False`` the hydrogens are deleted and no charge is
    touched: useful when the caller wants the topology change only (e.g. Vina's
    scoring, which ignores charges completely).
    """
    require_rdkit()
    drop = set(_nonpolar_hydrogens(mol))
    if not drop:
        return Chem.Mol(mol), 0

    values: Optional[List[float]] = None
    if collapse_charges:
        values = _gasteiger_values(mol)

    em = Chem.RWMol(mol)
    if collapse_charges and values is not None:
        for idx in drop:
            atom = em.GetAtomWithIdx(idx)
            parents = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if not parents:
                continue
            parent = parents[0].GetIdx()
            values[parent] = values[parent] + values[idx]
            values[idx] = 0.0
        for idx in range(em.GetNumAtoms()):
            if idx in drop:
                continue
            em.GetAtomWithIdx(idx).SetDoubleProp("_GasteigerCharge", float(values[idx]))
    for idx in sorted(drop, reverse=True):
        em.RemoveAtom(idx)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.FastFindRings(out)
        except Exception:  # pragma: no cover - defensive
            pass
    return out, len(drop)
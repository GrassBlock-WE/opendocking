# SPDX-License-Identifier: GPL-3.0-or-later
"""Docking against a *set* of receptor conformations, not one rigid structure.

A receptor is not a statue.  A single crystal structure is one snapshot of a
mobile protein, and a search that miniaturises into that snapshot rewards the
poses that happen to fit *it*.  This module builds the machinery for the honest
alternative: take several conformations of the same receptor, put them in one
common frame, dock the same ligand (or library) against every one of them, and
then reason across the results.

The pipeline has four stages, each of which is also usable on its own:

1. **Read and validate** (:func:`read_conformations`, :func:`align_conformations`).
   Several PDB/PDBQT files, or one multi-model PDB (an NMR ensemble, a set of
   NMR/MD snapshots), become a list of :class:`Conformation`.  They are only
   accepted as one ensemble when the residues say they are the same receptor:
   a sequence alignment reports the identity and overlap, and the set is
   *refused*, with the numbers, when the identity is too low.
2. **Superpose** (``ensemble.align_conformations``).  Every conformation is
   fitted onto a reference (the first by default) on its **binding-site**
   residues, so the shared search box is valid in every frame.  The report
   carries the site backbone RMSD, the whole-protein CA RMSD and the
   per-residue displacement of the site.
3. **Dock and merge** (:func:`dock_ensemble`).  Each conformation is docked with
   the same box, then every pose from every conformation is pooled into one
   ranked list and clustered *across* conformations: the same binding mode found
   in three receptors is a much stronger signal than one pose in one structure.
4. **Rescore and score robustness** (:func:`cross_rescore`,
   :func:`robustness_score`).  :mod:`odock.consensus` rescores a pose set with
   every force field *within* one conformation; this module adds the other axis
   -- every pose is rescored in every conformation -- so "is this pose robust to
   the receptor moving?" is a measured number (the affinity spread on a receptor
   swap) instead of a story.  Every correlation is reported with its sample size.

The library-scale form is :func:`screen_ensemble`, which is a thin layer over
:func:`odock.screen.screen_ligands` -- the same resumable results file, the same
one-row-per-(receptor, ligand) records, the same shortlist -- plus an
ensemble summary that aggregates those rows per ligand.

What this does *not* establish is written down in ``docs/ENSEMBLE.md``: docking
cannot separate conformational selection from induced fit, and an ensemble is
only as good as the structures in it.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .prepare import BoxSpec

__all__ = [
    "CODES",
    "CLASH_SPREAD",
    "DEFAULT_CLUSTER_RMSD",
    "DEFAULT_SITE_RADIUS",
    "DEFAULT_STRIP",
    "ENSEMBLE_CSV",
    "ENSEMBLE_DIR",
    "ENSEMBLE_JSONL",
    "ENSEMBLE_TOP",
    "ROBUSTNESS_SCALE",
    "EnsembleError",
    "Atom",
    "Residue",
    "Conformation",
    "ChainAlignment",
    "Alignment",
    "AlignedEnsemble",
    "Ensemble",
    "EnsemblePose",
    "PoseCluster",
    "EnsembleDockResult",
    "CrossRescoring",
    "RobustnessScore",
    "EnsembleLigandRow",
    "EnsembleScreenSummary",
    "add_ensemble_parser",
    "build_ensemble",
    "cluster_ensemble_poses",
    "conformation_consensus",
    "cross_rescore",
    "dock_ensemble",
    "kabsch",
    "ligand_coords",
    "align_conformations",
    "read_conformations",
    "robustness_score",
    "screen_ensemble",
    "residue_label",
    "parse_site_spec",
    "compare_sequences",
    "select_site",
]

PathLike = Union[str, os.PathLike]

#: The three-letter -> one-letter code table, including the modified residues
#: that routinely appear in a crystallographic chain (selenomethionine, the
#: phospho-residues, the common protonation variants of histidine).  A residue
#: that is not in this table is not part of the protein sequence; it is a
#: hetero-group and is reported as such rather than silently called ``X``.
CODES: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # Selenium/metal and engineered variants.
    "MSE": "M", "SEC": "U", "PYL": "O", "CSO": "C", "CSD": "C", "CME": "C",
    "CSX": "C", "OCS": "C", "CAS": "C",
    # Histidine protonation states and other tautomer spellings.
    "HSD": "H", "HSE": "H", "HSP": "H", "HID": "H", "HIE": "H", "HIP": "H",
    # Phospho/other post-translational variants.
    "SEP": "S", "TPO": "T", "PTR": "Y", "MLY": "K", "M3L": "K", "KCX": "K",
    "LLP": "K", "ALY": "K", "PCA": "E", "FME": "M", "HYP": "P", "SNN": "N",
    "TYS": "Y", "ABA": "A", "ORN": "K", "NLE": "L", "AIB": "A", "SAR": "G",
    "DVA": "V", "DAL": "A", "DLE": "L", "DAS": "D", "DGN": "Q", "DHI": "H",
    # Ambiguity codes, kept so they align rather than break a chain.
    "ASX": "B", "GLX": "Z", "UNK": "X", "XAA": "X",
}

#: AutoDock type -> element, for reading a PDBQT.  A copy of the table in
#: :mod:`odock.consensus`; deliberately duplicated so the structural layer has no
#: import edge into the scoring layer.
_AD4_TYPES: Dict[str, str] = {
    "C": "C", "A": "C", "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C",
    "N": "N", "NA": "N", "O": "O", "OA": "O", "S": "S", "SA": "S",
    "P": "P", "H": "H", "HD": "H", "F": "F", "I": "I", "Cl": "Cl",
    "Br": "Br", "Si": "Si", "At": "At", "W": "H",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
    "G0": "H", "G1": "H", "G2": "H", "G3": "H",
}

#: Residue names that are solvent, and are never part of a receptor.
WATER_NAMES = frozenset(("HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"))

#: Default radius (Å) around a box centre or a ligand used to pick the
#: binding-site residues that define the frame.
DEFAULT_SITE_RADIUS = 8.0

#: Default RMSD (Å) at which two poses from two conformations are the same mode.
DEFAULT_CLUSTER_RMSD = 2.0

#: The backbone atom names used by ``--site-atoms backbone``.
BACKBONE_ATOMS = ("N", "CA", "C", "O")

#: Residue names stripped from a bundled holo structure when it is prepared.
DEFAULT_STRIP = ("BEN", "STI", "OHT", "XK2", "MTX", "ERD")


class EnsembleError(RuntimeError):
    """A receptor set cannot be used as an ensemble.

    Carries the exit code the CLI should use (2: the inputs contradict each
    other), the same convention :class:`odock.screen.ScreenError` follows.
    """

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = int(code)


# ---------------------------------------------------------------------------
# Reading structures: no RDKit, no kernel, just the fixed columns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """One ``ATOM``/``HETATM`` record of a PDB or PDBQT file."""

    name: str
    element: str
    res_name: str
    res_id: int
    chain: str
    x: float
    y: float
    z: float
    altloc: str = ""
    record: str = "ATOM"

    @property
    def residue_key(self) -> Tuple[str, int, str]:
        """``("A", 189, "ASP")``."""
        return (self.chain, int(self.res_id), self.res_name)

    @property
    def is_hydrogen(self) -> bool:
        """Whether this atom is a hydrogen (or deuterium).

        The element column is authoritative: a mercury named ``HG`` has element
        ``Hg`` and is *not* a hydrogen, while the ``HG`` of a serine has element
        ``H``.  Only when the column is empty does the name decide.
        """
        element = self.element.strip().upper()
        if element:
            return element in ("H", "D")
        name = self.name.strip().upper()
        return name == "H" or (name.startswith("H") and name[1:2].isdigit())

    @property
    def is_hetero(self) -> bool:
        """Whether the residue is not a standard amino acid."""
        return self.res_name.strip().upper() not in CODES

    @property
    def label(self) -> str:
        """``"ASP189 A"``, the project-wide residue label."""
        return residue_label(self.res_name, self.res_id, self.chain)


def residue_label(res_name: str, res_id: int, chain: str) -> str:
    """``("ASP", 189, "A")`` -> ``"ASP189 A"`` (the :mod:`odock.pocket` spelling)."""
    label = f"{str(res_name).strip()}{int(res_id)}"
    return f"{label} {chain}".strip() if chain else label


def _element_from_name(name: str, res_name: str = "") -> str:
    """The element of an atom record that carries no element column.

    In a *protein* residue the two-letter trap is real and common: the alpha
    carbon is named ``CA``, which naively reads as calcium, and ``SE`` in a
    selenomethionine is selenium while ``HG`` in a serine is a hydrogen.  So a
    standard residue takes its element from the first letter of the atom name,
    with selenium as the one two-letter exception; anything else (an ion, a
    metal, a ligand) gets the two-letter treatment.
    """
    letters = "".join(c for c in str(name) if c.isalpha())
    if not letters:
        return "C"
    if CODES.get(str(res_name).strip().upper()):
        return "Se" if letters[:2].upper() == "SE" else letters[:1].upper()
    two = letters[:2].capitalize()
    if two in ("Cl", "Br", "Si", "At", "Na", "Mg", "Mn", "Zn", "Ca", "Fe", "Se", "Cu", "Ni", "Co"):
        return two
    return letters[:1].upper()


def _looks_like_pdbqt(text: str, path: Optional[Path] = None) -> bool:
    """Whether `text` uses the PDBQT layout (atom type in the last column).

    The two formats put different things in columns 77-79 -- PDB its element,
    PDBQT the AutoDock atom type -- and guessing per line confuses a sodium
    (``NA``) with an aromatic nitrogen.  So the decision is per document.
    """
    if path is not None and str(path).lower().endswith(".pdbqt"):
        return True
    for line in text.splitlines():
        if line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")):
            return True
    return False


def _parse_atom(
    line: str, *, pdbqt: bool
) -> Optional[Atom]:
    """Parse one ``ATOM``/``HETATM`` line, or return ``None`` when unusable."""
    if len(line) < 54:
        return None
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    name = line[12:16].strip()
    altloc = line[16:17].strip()
    if altloc not in ("", "A", "1"):
        return None  # a duplicate alternate location: keep the first only
    res_name = line[17:20].strip()
    chain = line[21:22].strip()
    try:
        res_id = int(line[22:26])
    except ValueError:
        return None

    element = ""
    if pdbqt:
        ad_type = line[77:].strip() if len(line) >= 78 else ""
        if ad_type:
            element = _AD4_TYPES.get(ad_type, "") or _AD4_TYPES.get(ad_type.upper(), "")
            if not element and ad_type[:1].isalpha():
                element = _element_from_name(name, res_name)
    else:
        column = line[76:78].strip() if len(line) >= 78 else ""
        if column and column.isalpha():
            element = column.capitalize() if len(column) == 2 else column.upper()
    if not element:
        element = _element_from_name(name, res_name)
    return Atom(
        name=name or element,
        element=element,
        res_name=res_name,
        res_id=res_id,
        chain=chain,
        x=x,
        y=y,
        z=z,
        altloc=altloc,
        record="HETATM" if line.startswith("HETATM") else "ATOM",
    )


def _rewrite_coords(line: str, xyz: Sequence[float]) -> str:
    """Return `line` with its coordinate columns replaced."""
    padded = line if len(line) >= 54 else line.ljust(54)
    x, y, z = (float(v) for v in xyz)
    return padded[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + padded[54:]


@dataclass
class Residue:
    """One residue of one conformation: its atoms, in file order."""

    chain: str
    res_id: int
    res_name: str
    atoms: List[Atom] = field(default_factory=list)

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.chain, int(self.res_id), self.res_name)

    @property
    def label(self) -> str:
        return residue_label(self.res_name, self.res_id, self.chain)

    @property
    def code(self) -> str:
        """The one-letter code, or ``""`` for a non-standard residue."""
        return CODES.get(self.res_name.strip().upper(), "")

    @property
    def is_standard(self) -> bool:
        return bool(self.code)

    @property
    def is_water(self) -> bool:
        return self.res_name.strip().upper() in WATER_NAMES

    def coords(self, *, heavy_only: bool = True) -> np.ndarray:
        atoms = [a for a in self.atoms if not (heavy_only and a.is_hydrogen)]
        return np.array([[a.x, a.y, a.z] for a in atoms], dtype=float).reshape(-1, 3)

    def atom_names(self, *, heavy_only: bool = True) -> List[str]:
        return [a.name for a in self.atoms if not (heavy_only and a.is_hydrogen)]


def _split_models(text: str) -> List[Tuple[int, List[str]]]:
    """``[(model_number, lines), ...]``; a file without ``MODEL`` is model 1."""
    blocks: List[Tuple[int, List[str]]] = []
    current: Optional[List[str]] = None
    number = 0
    for line in text.splitlines():
        if line.startswith("MODEL"):
            number += 1
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current is not None:
                blocks.append((number, current))
            current = None
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        blocks.append((number, current))
    if not blocks:
        # No MODEL records at all: the whole document is one conformation.
        lines = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
        return [(0, lines)]
    return [(number, lines) for number, lines in blocks]


@dataclass
class Conformation:
    """One receptor conformation: a label, the atoms, and the records they came from.

    ``records`` holds the original ``ATOM``/``HETATM`` lines in the same order as
    :attr:`atoms`, so a conformation can be transformed and written back out
    without losing anything the original file carried (occupancies, B factors,
    chain identifiers, atom names).
    """

    label: str
    atoms: List[Atom]
    records: List[str] = field(default_factory=list)
    path: Optional[Path] = None
    model: int = 0
    source: str = ""
    #: Cached ``chain -> residues`` mapping (:meth:`chain_residues`).
    _chains: Optional["OrderedDict[str, List[Residue]]"] = field(
        default=None, repr=False, compare=False
    )

    # -- geometry ---------------------------------------------------------

    def coords(self, *, heavy_only: bool = False) -> np.ndarray:
        atoms = [a for a in self.atoms if not (heavy_only and a.is_hydrogen)]
        return np.array([[a.x, a.y, a.z] for a in atoms], dtype=float).reshape(-1, 3)

    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> "Conformation":
        """This conformation with ``x -> R x + t`` applied to every atom."""
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(translation, dtype=float).reshape(3)
        xyz = self.coords() @ rotation.T + translation
        atoms = [
            Atom(
                name=a.name, element=a.element, res_name=a.res_name, res_id=a.res_id,
                chain=a.chain, x=float(p[0]), y=float(p[1]), z=float(p[2]),
                altloc=a.altloc, record=a.record,
            )
            for a, p in zip(self.atoms, xyz)
        ]
        records = [
            _rewrite_coords(line, point) if point is not None else line
            for line, point in zip(self.records, xyz)
        ]
        return Conformation(
            label=self.label, atoms=atoms, records=records, path=self.path,
            model=self.model, source=self.source,
        )

    # -- sequence and residues --------------------------------------------

    def chain_residues(self) -> "OrderedDict[str, List[Residue]]":
        """``chain -> residues`` in file order (waters dropped).

        One pass over the atoms: a residue is a contiguous run of atoms sharing
        ``(chain, res_id, res_name, altloc-kept)``.  Waters are excluded -- they
        are not part of the receptor and their presence varies wildly between
        structures of the same protein.
        """
        if self._chains is not None:
            return self._chains
        chains: "OrderedDict[str, List[Residue]]" = OrderedDict()
        for atom in self.atoms:
            if atom.res_name.strip().upper() in WATER_NAMES:
                continue
            residues = chains.setdefault(atom.chain, [])
            if residues and residues[-1].key == atom.residue_key:
                residues[-1].atoms.append(atom)
            else:
                residues.append(
                    Residue(
                        chain=atom.chain, res_id=atom.res_id, res_name=atom.res_name,
                        atoms=[atom],
                    )
                )
        self._chains = chains
        return chains

    def standard_residues(self, chain: str) -> List[Residue]:
        """The protein residues of `chain`, i.e. the ones with a one-letter code."""
        return [r for r in self.chain_residues().get(chain, []) if r.is_standard]

    def sequence_by_chain(self) -> "OrderedDict[str, str]":
        """``chain -> one-letter sequence`` over the standard residues."""
        out: "OrderedDict[str, str]" = OrderedDict()
        for chain, residues in self.chain_residues().items():
            sequence = "".join(r.code or "X" for r in residues if r.is_standard)
            if sequence:
                out[chain] = sequence
        return out

    def sequence(self) -> str:
        """Every chain's sequence concatenated in file order."""
        return "".join(self.sequence_by_chain().values())

    def n_residues(self, *, standard_only: bool = True) -> int:
        return sum(
            1
            for residues in self.chain_residues().values()
            for residue in residues
            if residue.is_standard or not standard_only
        )

    def hetero_residues(self) -> List[str]:
        """Labels of the non-standard, non-water residues (ligands, metals)."""
        return [
            residue.label
            for residues in self.chain_residues().values()
            for residue in residues
            if not residue.is_standard and not residue.is_water
        ]

    # -- text -------------------------------------------------------------

    def text(self) -> str:
        """The conformation as a PDB document (its own records, transformed)."""
        head = f"REMARK   ODOCK ENSEMBLE: {self.label}"
        if self.model:
            head += f" model {self.model}"
        lines = [head]
        lines.extend(self.records)
        lines.append("END")
        return "\n".join(lines) + "\n"

    def replace_records(self, records: Sequence[str]) -> "Conformation":
        """A copy carrying `records` (kept in step with :meth:`atoms`)."""
        return Conformation(
            label=self.label, atoms=list(self.atoms), records=list(records),
            path=self.path, model=self.model, source=self.source,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "path": None if self.path is None else str(self.path),
            "model": int(self.model),
            "n_atoms": len(self.atoms),
            "n_residues": self.n_residues(),
            "chains": list(self.sequence_by_chain()),
            "hetero": self.hetero_residues(),
        }


def _labels_for(paths: Sequence[Path]) -> List[str]:
    """Unique, human-readable labels for a list of files (stems, de-duplicated)."""
    stems = [p.stem for p in paths]
    counts: Dict[str, int] = {}
    for stem in stems:
        counts[stem] = counts.get(stem, 0) + 1
    out: List[str] = []
    used: set = set()
    for path, stem in zip(paths, stems):
        label = stem
        if counts[stem] > 1:
            parent = path.parent.name
            if parent and parent not in (".", ".."):
                label = f"{parent}_{stem}"
        while label in used:
            label = f"{label}_"
        used.add(label)
        out.append(label)
    return out


def read_conformations(
    source: Union[PathLike, Sequence[PathLike]],
    *,
    label: Optional[str] = None,
    keep_water: bool = False,
) -> List[Conformation]:
    """Read one or more structure files into a list of conformations.

    A file with ``MODEL`` records (an NMR ensemble, a set of snapshots) becomes
    one :class:`Conformation` per model, labelled ``<stem>#<n>``; a file without
    them becomes a single conformation labelled ``<stem>``.  Several files are
    labelled by their stems, with the parent directory's name appended only when
    two stems collide.

    Parameters
    ----------
    source
        A path, or a sequence of paths.
    keep_water
        Keep crystallographic waters.  They are dropped by default: their
        presence and their positions vary between structures of the same
        receptor, so they carry no shared frame.

    Raises
    ------
    EnsembleError
        A file is missing, unreadable, or holds no atom record.
    """
    paths = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    paths = [Path(p) for p in paths]
    if not paths:
        raise EnsembleError("no receptor file given")
    given = _labels_for(paths)
    conformations: List[Conformation] = []
    for path, base in zip(paths, given):
        if not path.exists():
            raise EnsembleError(f"no such file: {path}")
        if not path.is_file():
            raise EnsembleError(f"not a file: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        pdbqt = _looks_like_pdbqt(text, path)
        blocks = _split_models(text)
        kept = 0
        for number, lines in blocks:
            atoms: List[Atom] = []
            records: List[str] = []
            for line in lines:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                atom = _parse_atom(line, pdbqt=pdbqt)
                if atom is None:
                    continue
                if not keep_water and atom.res_name.strip().upper() in WATER_NAMES:
                    continue
                atoms.append(atom)
                records.append(line)
            if not atoms:
                continue
            kept += 1
            name = label or base
            if len(blocks) > 1:
                name = f"{name}#{number}"
            conformations.append(
                Conformation(
                    label=name, atoms=atoms, records=records, path=path,
                    model=number, source=str(path),
                )
            )
        if not kept:
            raise EnsembleError(
                f"{path} holds no ATOM/HETATM record that could be read as a "
                "receptor"
            )
    return conformations


# ---------------------------------------------------------------------------
# Are these the same receptor?
# ---------------------------------------------------------------------------


def _global_align(
    a: str, b: str, *, match: float = 2.0, mismatch: float = -1.0, gap: float = -2.0
) -> Tuple[str, str]:
    """Needleman-Wunsch global alignment of two sequences.

    A plain affine-free linear gap model: the sequences are the same protein, so
    the alignment is easy, and a simple, inspectable scoring scheme is worth more
    here than a BLOSUM matrix.  Returns the two gapped strings.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return ("-" * m, b) if n == 0 else (a, "-" * n)
    score = np.empty((n + 1, m + 1), dtype=float)
    score[0, :] = np.arange(m + 1) * gap
    score[:, 0] = np.arange(n + 1) * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_prev = score[i - 1]
        row = score[i]
        for j in range(1, m + 1):
            diagonal = row_prev[j - 1] + (match if ai == b[j - 1] else mismatch)
            up = row_prev[j] + gap
            left = row[j - 1] + gap
            row[j] = diagonal if diagonal >= up and diagonal >= left else (up if up >= left else left)

    out_a: List[str] = []
    out_b: List[str] = []
    i, j = n, m
    while i > 0 and j > 0:
        diagonal = score[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
        if score[i, j] == diagonal:
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif score[i, j] == score[i - 1, j] + gap:
            out_a.append(a[i - 1])
            out_b.append("-")
            i -= 1
        else:
            out_a.append("-")
            out_b.append(b[j - 1])
            j -= 1
    while i > 0:
        out_a.append(a[i - 1])
        out_b.append("-")
        i -= 1
    while j > 0:
        out_a.append("-")
        out_b.append(b[j - 1])
        j -= 1
    return "".join(reversed(out_a)), "".join(reversed(out_b))


@dataclass
class ChainAlignment:
    """The alignment of one chain of one conformation onto one chain of another."""

    chain: str
    other_chain: str
    aligned: str
    other_aligned: str
    n_columns: int
    n_matched: int

    @property
    def identity(self) -> float:
        """Matched columns over aligned columns (both sequences in the pair)."""
        return self.n_matched / self.n_columns if self.n_columns else 0.0

    @property
    def overlap(self) -> float:
        """Matched residues over the shorter of the two chains."""
        shorter = min(
            sum(1 for c in self.aligned if c != "-"),
            sum(1 for c in self.other_aligned if c != "-"),
        )
        return self.n_matched / shorter if shorter else 0.0

    def pairs(self) -> List[Tuple[int, int]]:
        """``(index in this chain, index in the other chain)`` for every match.

        The indices are positions in each chain's *standard* residue list, which
        is what :meth:`Conformation.standard_residues` returns.
        """
        out: List[Tuple[int, int]] = []
        i = j = 0
        for ca, cb in zip(self.aligned, self.other_aligned):
            if ca != "-" and cb != "-":
                out.append((i, j))
            if ca != "-":
                i += 1
            if cb != "-":
                j += 1
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chain": self.chain,
            "other_chain": self.other_chain,
            "identity": self.identity,
            "overlap": self.overlap,
            "n_matched": self.n_matched,
            "n_columns": self.n_columns,
        }


def compare_sequences(
    reference: Conformation, other: Conformation
) -> List[ChainAlignment]:
    """Align `other`'s chains onto `reference`'s, best pair first.

    Chains are matched greedily by identity, so a homodimer lines up with a
    homodimer and a chain that has no counterpart is left unmatched instead of
    being forced onto an unrelated chain.
    """
    sequences = reference.sequence_by_chain()
    others = other.sequence_by_chain()
    candidates: List[Tuple[float, int, str, str, str, str]] = []
    for chain, sequence in sequences.items():
        for other_chain, other_sequence in others.items():
            aligned, other_aligned = _global_align(sequence, other_sequence)
            matched = sum(1 for x, y in zip(aligned, other_aligned) if x == y and x != "-")
            identity = matched / max(1, len(aligned))
            candidates.append(
                (-identity, -matched, chain, other_chain, aligned, other_aligned)
            )
    candidates.sort()
    used: set = set()
    used_other: set = set()
    out: List[ChainAlignment] = []
    for neg_identity, neg_matched, chain, other_chain, aligned, other_aligned in candidates:
        if chain in used or other_chain in used_other:
            continue
        used.add(chain)
        used_other.add(other_chain)
        out.append(
            ChainAlignment(
                chain=chain, other_chain=other_chain, aligned=aligned,
                other_aligned=other_aligned, n_columns=len(aligned),
                n_matched=-neg_matched,
            )
        )
    return out


def _summary_identity(alignments: Sequence[ChainAlignment]) -> Tuple[float, float, int, int]:
    """``(identity, overlap, matched, covered)`` over a set of chain alignments."""
    matched = sum(a.n_matched for a in alignments)
    columns = sum(a.n_columns for a in alignments)
    covered = sum(
        sum(1 for c in a.aligned if c != "-") for a in alignments
    )
    identity = matched / columns if columns else 0.0
    overlap = matched / covered if covered else 0.0
    return identity, overlap, matched, covered


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------


def kabsch(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """The optimal rigid transform ``x -> R x + t`` taking `mobile` onto `target`.

    Returns ``(rotation, translation, rmsd)``.  The rotation is a proper
    rotation (``det = +1``): a reflection is never the answer to "how did this
    protein move".
    """
    p = np.asarray(mobile, dtype=float).reshape(-1, 3)
    q = np.asarray(target, dtype=float).reshape(-1, 3)
    if p.shape != q.shape:
        raise ValueError(f"cannot superpose {p.shape} onto {q.shape}")
    if p.shape[0] < 3:
        raise EnsembleError(
            f"at least three atoms are needed to superpose two frames, got {p.shape[0]}"
        )
    centre_p = p.mean(axis=0)
    centre_q = q.mean(axis=0)
    covariance = (p - centre_p).T @ (q - centre_q)
    # numpy's SVD gives ``covariance = v @ diag(s) @ wt``, i.e. the classical
    # ``V S W^T``, so the optimal rotation ``W D V^T`` is ``wt.T @ D @ v.T``.
    v, _s, wt = np.linalg.svd(covariance)
    determinant = float(np.linalg.det(v @ wt))
    correction = np.diag([1.0, 1.0, 1.0 if determinant > 0 else -1.0])
    rotation = wt.T @ correction @ v.T
    translation = centre_q - rotation @ centre_p
    residual = p @ rotation.T + translation - q
    rmsd = float(np.sqrt((residual ** 2).sum(axis=1).mean()))
    return rotation, translation, rmsd


def _pair_residue_atoms(
    first: Residue, second: Residue, *, atoms: str, heavy_only: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The matched atoms of two residues, by atom name.

    Parameters
    ----------
    atoms
        ``"ca"`` (the CA carbon), ``"backbone"`` (N, CA, C, O) or ``"all"``.
    """
    wanted: Optional[set] = None
    if atoms == "ca":
        wanted = {"CA"}
    elif atoms == "backbone":
        wanted = set(BACKBONE_ATOMS)
    elif atoms != "all":
        raise EnsembleError(f"unknown atom selection {atoms!r}; use ca, backbone or all")

    def index(residue: Residue) -> Dict[str, Atom]:
        out: Dict[str, Atom] = {}
        for atom in residue.atoms:
            if heavy_only and atom.is_hydrogen:
                continue
            if wanted is not None and atom.name not in wanted:
                continue
            out.setdefault(atom.name, atom)
        return out

    left = index(first)
    right = index(second)
    names = sorted(
        name for name in left if name in right
    )
    # Backbone first, so a caller reading the first rows sees the stable atoms.
    names.sort(key=lambda name: (BACKBONE_ATOMS.index(name) if name in BACKBONE_ATOMS else 99, name))
    a = np.array([[left[n].x, left[n].y, left[n].z] for n in names], dtype=float).reshape(-1, 3)
    b = np.array([[right[n].x, right[n].y, right[n].z] for n in names], dtype=float).reshape(-1, 3)
    return a, b, names


# ---------------------------------------------------------------------------
# Site selection
# ---------------------------------------------------------------------------


def parse_site_spec(spec: str) -> List[Tuple[str, Any]]:
    """Parse a ``--site`` list into ``(chain, residue)`` requests.

    Accepted spellings, comma-separated: ``189``, ``ASP189``, ``A:189``,
    ``A:ASP189``, ``A/189``.  An empty chain matches any chain.
    """
    out: List[Tuple[str, Any]] = []
    for raw in str(spec).replace(";", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        chain = ""
        body = token
        for separator in (":", "/"):
            if separator in body:
                head, _, tail = body.partition(separator)
                if head.strip():
                    chain = head.strip()
                    body = tail.strip()
                break
        digits = "".join(c for c in body if c.isdigit())
        letters = "".join(c for c in body if c.isalpha())
        if not digits:
            raise EnsembleError(
                f"cannot read the residue {token!r} of --site; write it as "
                "'ASP189', '189' or 'A:189'"
            )
        out.append((chain.upper(), (int(digits), letters.upper() or None)))
    return out


@dataclass
class SiteResidue:
    """One residue of the reference that defines the binding-site frame."""

    chain: str
    index: int
    residue: Residue

    @property
    def label(self) -> str:
        return self.residue.label


def _residues_near(
    conformation: Conformation, points: np.ndarray, radius: float
) -> List[SiteResidue]:
    """Standard residues with any heavy atom within `radius` of any point."""
    out: List[SiteResidue] = []
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return out
    for chain, residues in conformation.chain_residues().items():
        index = 0
        for residue in residues:
            if not residue.is_standard:
                continue
            coords = residue.coords(heavy_only=True)
            if coords.size:
                distance = np.sqrt(((coords[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
                if float(distance.min()) <= radius:
                    out.append(SiteResidue(chain=chain, index=index, residue=residue))
            index += 1
    return out


def select_site(
    conformation: Conformation,
    *,
    site: Optional[str] = None,
    box: Optional[BoxSpec] = None,
    ligand: Optional[np.ndarray] = None,
    radius: float = DEFAULT_SITE_RADIUS,
    max_residues: int = 40,
) -> List[SiteResidue]:
    """The binding-site residues that define the common frame.

    In order of preference:

    * `site` -- an explicit list (:func:`parse_site_spec`);
    * `box` -- every residue with an atom within `radius` Å of the box centre.
      The frame exists so that the box means the same thing in every
      conformation, so the fit is anchored on the residues the box describes;
    * `ligand` -- every residue with an atom within `radius` Å of a ligand
      coordinate set (the co-crystallised ligand, for instance), used when there
      is no box.

    A site wider than `max_residues` is trimmed to the `max_residues` residues
    closest to the site centre: a fit on 200 residues is a global fit, and the
    point of this transform is to put the *site* in one frame.
    """
    if site:
        candidates: List[Tuple[str, int, Residue]] = [
            (chain, index, residue)
            for chain, residues in conformation.chain_residues().items()
            for index, residue in enumerate(r for r in residues if r.is_standard)
        ]
        wanted = parse_site_spec(site)
        out: List[SiteResidue] = []
        missing: List[str] = []
        for chain, (res_id, res_name) in wanted:
            for chain_id, index, residue in candidates:
                if residue.res_id != res_id:
                    continue
                if res_name and residue.res_name.upper() != res_name:
                    continue
                if chain and chain_id != chain:
                    continue
                out.append(SiteResidue(chain=chain_id, index=index, residue=residue))
                break
            else:
                missing.append(f"{chain}:{res_id}" if chain else str(res_id))
        if missing:
            raise EnsembleError(
                f"the site residue(s) {', '.join(missing)} are not in "
                f"{conformation.label} ({conformation.path})"
            )
        return out

    if box is not None:
        # The box comes first on purpose.  The frame exists so that *this box*
        # means the same thing in every conformation, so the fit is anchored on
        # the residues the box describes: a residue within `radius` of the box
        # centre.  A ligand is the better site definition when there is no box
        # (a co-crystallised inhibitor at the site), which is why --site-ligand
        # is used when no box is given.
        centre = np.asarray(box.center, dtype=float).reshape(1, 3)
        found = _residues_near(conformation, centre, radius)
        if not found:
            raise EnsembleError(
                f"no residue of {conformation.label} lies within {radius:g} A of the "
                f"box centre ({box.center[0]:.2f}, {box.center[1]:.2f}, "
                f"{box.center[2]:.2f}); the box and the receptor are in different "
                "frames, or --site-radius is too small.\n"
                "       The box must be given in the *reference* structure's frame: "
                "with --reference N the box has to come from structure N (for "
                "example --box-ligand <a residue of that structure>)."
            )
        return _trim_site(found, centre, max_residues)

    if ligand is not None:
        points = np.asarray(ligand, dtype=float).reshape(-1, 3)
        found = _residues_near(conformation, points, radius)
        if not found:
            raise EnsembleError(
                f"no residue of {conformation.label} lies within {radius:g} A of the "
                "reference ligand"
            )
        return _trim_site(found, points.mean(axis=0).reshape(1, 3), max_residues)

    raise EnsembleError(
        "no binding site: pass --site RES[,RES...], or a --box (with --site-radius), "
        "or a reference ligand (--site-ligand)"
    )


def ligand_coords(
    conformation: Conformation, res_name: str, *, chain: str = ""
) -> np.ndarray:
    """Heavy-atom coordinates of a named residue of `conformation`.

    This is how a box or a site is derived from a *holo* structure without any
    external ligand file: the co-crystallised inhibitor that is already in the
    file (``XK2`` in 1HVR, ``OHT`` in 3ERT) is exactly where the site is.

    Raises
    ------
    EnsembleError
        The residue is not in the conformation, or holds no heavy atom.
    """
    wanted = str(res_name).strip().upper()
    matches = [
        residue
        for residues in conformation.chain_residues().values()
        for residue in residues
        if residue.res_name.strip().upper() == wanted
        and (not chain or residue.chain.upper() == str(chain).upper())
    ]
    if not matches:
        present = sorted(
            {
                residue.res_name
                for residues in conformation.chain_residues().values()
                for residue in residues
                if not residue.is_standard
            }
        )
        raise EnsembleError(
            f"{conformation.label} has no residue named {res_name!r}; its "
            f"non-standard residues are {', '.join(present) or 'none'}"
        )
    points = [r.coords(heavy_only=True) for r in matches]
    points = [p for p in points if p.size]
    if not points:
        raise EnsembleError(f"residue {res_name!r} of {conformation.label} has no heavy atom")
    return np.vstack(points)


def _trim_site(
    residues: Sequence[SiteResidue], centre: np.ndarray, limit: int
) -> List[SiteResidue]:
    if limit <= 0 or len(residues) <= limit:
        return list(residues)

    def distance(site: SiteResidue) -> float:
        coords = site.residue.coords(heavy_only=True)
        if not coords.size:
            return math.inf
        return float(np.sqrt(((coords - centre) ** 2).sum(axis=1)).min())

    ordered = sorted(residues, key=lambda s: (distance(s), s.chain, s.index))
    kept = ordered[:limit]
    return [s for s in residues if s in kept]


# ---------------------------------------------------------------------------
# The aligned ensemble
# ---------------------------------------------------------------------------


@dataclass
class Alignment:
    """How one conformation had to move to sit on the reference's site frame."""

    label: str
    reference: str
    #: Sequence identity over the aligned columns of every matched chain pair.
    identity: float
    #: Matched residues over the residues the two structures have in common.
    overlap: float
    matched_residues: int
    common_residues: int
    chain_alignments: List[ChainAlignment] = field(default_factory=list)
    n_site_residues: int = 0
    n_site_atoms: int = 0
    #: RMSD of the binding-site atoms the fit used, in Å (0 for the reference).
    site_rmsd: float = 0.0
    #: CA RMSD of every matched residue (the whole protein) after that fit.
    global_rmsd: float = 0.0
    #: Per-residue heavy-atom RMSD of the site, ``label -> Å``.
    displacement: Dict[str, float] = field(default_factory=dict)
    rotation: Optional[List[List[float]]] = None
    translation: Optional[List[float]] = None
    reference_frame: bool = False

    @property
    def max_displacement(self) -> float:
        return max(self.displacement.values()) if self.displacement else 0.0

    @property
    def mean_displacement(self) -> float:
        if not self.displacement:
            return 0.0
        return float(np.mean(list(self.displacement.values())))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "reference": self.reference,
            "reference_frame": bool(self.reference_frame),
            "identity": self.identity,
            "overlap": self.overlap,
            "matched_residues": int(self.matched_residues),
            "common_residues": int(self.common_residues),
            "chains": [c.as_dict() for c in self.chain_alignments],
            "n_site_residues": int(self.n_site_residues),
            "n_site_atoms": int(self.n_site_atoms),
            "site_rmsd": self.site_rmsd,
            "global_rmsd": self.global_rmsd,
            "mean_displacement": self.mean_displacement,
            "max_displacement": self.max_displacement,
            "displacement": dict(self.displacement),
            "rotation": self.rotation,
            "translation": self.translation,
        }


@dataclass
class AlignedEnsemble:
    """A set of conformations in one common (binding-site) frame."""

    conformations: List[Conformation]
    alignments: List[Alignment]
    site: List[str] = field(default_factory=list)
    reference: int = 0
    box: Optional[BoxSpec] = None
    atom_selection: str = "ca"
    warnings: List[str] = field(default_factory=list)

    @property
    def labels(self) -> List[str]:
        return [c.label for c in self.conformations]

    def reference_conformation(self) -> Conformation:
        return self.conformations[self.reference]

    def table(self) -> str:
        """One row per conformation: identity, overlap, RMSDs, site size."""
        headers = [
            "conformation", "residues", "identity", "overlap", "site",
            "site RMSD", "CA RMSD", "max disp", "mean disp",
        ]
        rows = []
        for conformation, alignment in zip(self.conformations, self.alignments):
            rows.append([
                conformation.label,
                str(conformation.n_residues()),
                f"{alignment.identity:.3f}",
                f"{alignment.overlap:.3f}",
                str(alignment.n_site_residues),
                f"{alignment.site_rmsd:.3f}",
                f"{alignment.global_rmsd:.3f}",
                f"{alignment.max_displacement:.3f}",
                f"{alignment.mean_displacement:.3f}",
            ])
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def displacement_table(self, limit: int = 40) -> str:
        """Per-site-residue displacement, worst first (after the fit)."""
        if not self.site:
            return "(no binding-site residues)"
        rows = []
        for label in self.site:
            values = [
                a.displacement.get(label) for a in self.alignments if not a.reference_frame
            ]
            values = [v for v in values if v is not None]
            worst = max(values) if values else 0.0
            rows.append((label, values, worst))
        rows.sort(key=lambda item: -item[2])
        headers = ["residue", "worst (A)"] + [
            a.label for a in self.alignments if not a.reference_frame
        ]
        body = []
        for label, values, worst in rows[: max(1, limit)]:
            body.append(
                [label, f"{worst:.3f}"] + ["--" if v is None else f"{v:.3f}" for v in values]
            )
        widths = [len(h) for h in headers]
        for row in body:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in body)
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.conformations[self.reference].label,
            "reference_index": int(self.reference),
            "atom_selection": self.atom_selection,
            "site": list(self.site),
            "box": None if self.box is None else self.box.as_dict(),
            "conformations": [c.as_dict() for c in self.conformations],
            "alignments": [a.as_dict() for a in self.alignments],
            "warnings": list(self.warnings),
        }

    def text(self) -> str:
        """The whole alignment report, as the CLI prints it."""
        lines = [
            f"OpenDocking ensemble — {len(self.conformations)} conformation(s), "
            f"reference {self.conformations[self.reference].label}",
            "",
            self.table(),
            "",
            f"binding site ({len(self.site)} residue(s), fitted on "
            f"{self.atom_selection} atoms):",
        ]
        if self.site:
            lines.append("  " + ", ".join(self.site))
        lines.append("")
        lines.append("per-residue binding-site displacement after the fit (Å):")
        lines.append(self.displacement_table())
        if self.warnings:
            lines.append("")
            lines.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(lines)

    # -- writing the aligned receptors ------------------------------------

    def aligned_pdb_text(self, index: int) -> str:
        """Conformation `index` (already fitted) as PDB text."""
        return self.conformations[index].text()

    def write_pdb(self, outdir: PathLike) -> List[Path]:
        """Write every aligned conformation as ``<outdir>/<label>.pdb``."""
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for conformation in self.conformations:
            target = directory / f"{_safe_stem(conformation.label)}.pdb"
            _refuse_to_clobber(target, conformation)
            _write_if_changed(target, conformation.text())
            written.append(target)
        return written

    def receptor_pdbqt(
        self,
        index: int,
        *,
        outdir: PathLike,
        strip: Optional[Sequence[str]] = None,
        keep_water: bool = False,
        keep_hetero: bool = True,
    ) -> Path:
        """Prepare conformation `index` into a receptor PDBQT.

        The aligned coordinates go through :func:`odock.prepare.prepare_receptor`,
        which is what adds the polar hydrogens the force field needs; writing the
        PDBQT by hand would produce a receptor the kernel's typing cannot use.
        """
        from .prepare import prepare_receptor

        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(self.conformations[index].label)
        pdb_path = directory / f"{stem}.pdb"
        pdbqt_path = directory / f"{stem}.pdbqt"
        _refuse_to_clobber(pdb_path, self.conformations[index])
        _refuse_to_clobber(pdbqt_path, self.conformations[index])
        _write_if_changed(pdb_path, self.conformations[index].text())
        _, text, _report = prepare_receptor(
            str(pdb_path), None, keep_water=keep_water, keep_hetero=keep_hetero,
            strip=strip,
        )
        _write_if_changed(pdbqt_path, text)
        return pdbqt_path


def _safe_stem(label: str) -> str:
    """A file-name-safe version of a conformation label."""
    out = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(label))
    return out or "conformation"


def _refuse_to_clobber(target: Path, conformation: Conformation) -> None:
    """Never write an aligned conformation on top of the file it came from.

    ``--outdir`` is easy to point at the directory the structures were read from,
    and the aligned copy has the same stem as its source.  Silently overwriting
    the input would corrupt the very thing the run is a measurement of (and the
    next run would read the already-fitted file).
    """
    if conformation.path is None:
        return
    try:
        same = Path(target).resolve() == Path(conformation.path).resolve()
    except OSError:  # pragma: no cover - unresolvable path
        same = False
    if same:
        raise EnsembleError(
            f"refusing to write the aligned {conformation.label} onto its own input "
            f"file {conformation.path}: choose an --outdir that is not the directory "
            "the structures were read from"
        )


def _write_if_changed(path: Path, text: str) -> bool:
    """Write `text` to `path` only when it differs; return whether it was written.

    Screening resumes by comparing the *file signature* (size and mtime) of every
    receptor, so rewriting an identical aligned receptor on every invocation
    would refuse to resume a campaign that is perfectly resumable.  Comparing the
    content first keeps the mtime stable, and therefore the campaign hash stable.
    """
    path = Path(path)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except OSError:  # pragma: no cover - unreadable: rewrite it
            pass
    path.write_text(text, encoding="utf-8")
    return True


def align_conformations(
    conformations: Sequence[Conformation],
    *,
    reference: int = 0,
    box: Optional[BoxSpec] = None,
    site: Optional[str] = None,
    site_ligand: Optional[np.ndarray] = None,
    site_radius: float = DEFAULT_SITE_RADIUS,
    max_site_residues: int = 40,
    atoms: str = "ca",
    superpose: bool = True,
    min_identity: float = 0.90,
    min_residues: int = 10,
    min_site_residues: int = 3,
    allow_box_mismatch: bool = False,
) -> AlignedEnsemble:
    """Validate that a set of conformations is one receptor, and put it in one frame.

    Every conformation is aligned onto `reference` on the residues of the
    **binding site**, so the shared search box means the same thing in every
    frame.  The site comes from `site` (explicit), the box centre, or a reference
    ligand (:func:`select_site`).

    The set is refused -- with the numbers that justify the refusal -- when the
    sequence identity is below `min_identity`, when fewer than `min_residues`
    residues are shared, or when the binding site holds fewer than
    `min_site_residues` residues.  ``superpose=False`` keeps the input frames and
    only *reports* the alignment statistics (for a set that has already been
    superposed, or when the caller wants the numbers without touching the
    coordinates).

    Returns
    -------
    :class:`AlignedEnsemble`, whose ``conformations`` are the fitted copies.
    """
    if not conformations:
        raise EnsembleError("no conformation to align")
    if not 0 <= int(reference) < len(conformations):
        raise EnsembleError(
            f"the reference conformation {reference} does not exist "
            f"(the ensemble has {len(conformations)})"
        )
    if atoms not in ("ca", "backbone", "all"):
        raise EnsembleError(f"unknown atom selection {atoms!r}; use ca, backbone or all")
    reference_index = int(reference)
    ref = conformations[reference_index]

    try:
        site_residues = select_site(
            ref, site=site, box=box, ligand=site_ligand, radius=site_radius,
            max_residues=max_site_residues,
        )
    except EnsembleError:
        raise
    if len(site_residues) < int(min_site_residues):
        raise EnsembleError(
            f"the binding site of {ref.label} holds {len(site_residues)} residue(s), "
            f"fewer than the {min_site_residues} needed to define a frame; widen "
            "--site-radius, pass --site, or lower --min-site-residues"
        )

    alignments: List[Alignment] = []
    fitted: List[Conformation] = []
    warnings: List[str] = []
    for index, conformation in enumerate(conformations):
        if index == reference_index:
            alignments.append(
                Alignment(
                    label=conformation.label, reference=ref.label, identity=1.0,
                    overlap=1.0, matched_residues=conformation.n_residues(),
                    common_residues=conformation.n_residues(),
                    n_site_residues=len(site_residues),
                    n_site_atoms=_count_site_atoms(site_residues, atoms),
                    site_rmsd=0.0,
                    global_rmsd=0.0,
                    displacement={s.label: 0.0 for s in site_residues},
                    rotation=np.eye(3).tolist(),
                    translation=[0.0, 0.0, 0.0],
                    reference_frame=True,
                )
            )
            fitted.append(conformation)
            continue

        chains = compare_sequences(ref, conformation)
        identity, overlap, matched, covered = _summary_identity(chains)
        common = sum(
            sum(1 for c in a.other_aligned if c != "-") for a in chains
        )
        if not chains or matched == 0:
            raise EnsembleError(
                f"{conformation.label} ({conformation.path}) shares no residue with "
                f"{ref.label} ({ref.path}): the two sequences are "
                f"{len(ref.sequence())} and {len(conformation.sequence())} residues "
                "long and align without a single match. They are not the same "
                "receptor."
            )
        if identity < float(min_identity):
            raise EnsembleError(
                f"{conformation.label} is not the same receptor as {ref.label}: "
                f"sequence identity {identity:.3f} over {sum(a.n_columns for a in chains)} "
                f"aligned columns ({matched} identical residues), below --min-identity "
                f"{min_identity:.2f}.\n"
                "       Refusing to dock one ligand into a set of different proteins: "
                "the affinity would not be comparable across the ensemble. Relax "
                "--min-identity only if the difference is a real, intended mutation "
                "series."
            )
        if matched < int(min_residues):
            raise EnsembleError(
                f"{conformation.label} shares only {matched} residue(s) with "
                f"{ref.label}, fewer than --min-residues {int(min_residues)}; the "
                "structures are too incomplete to compare."
            )

        pairs, site_labels = _map_site(site_residues, chains, conformation)
        if not pairs:
            raise EnsembleError(
                f"none of the {len(site_residues)} binding-site residue(s) of "
                f"{ref.label} could be located in {conformation.label}; the site "
                f"({' ,'.join(s.label for s in site_residues[:5])}...) is not shared"
            )
        mobile: List[np.ndarray] = []
        target: List[np.ndarray] = []
        for ref_residue, other_residue in pairs:
            reference_atoms, other_atoms, _names = _pair_residue_atoms(
                ref_residue, other_residue, atoms=atoms, heavy_only=True
            )
            if reference_atoms.shape[0]:
                # `other` moves onto the reference, so it is the mobile set.
                mobile.append(other_atoms)
                target.append(reference_atoms)
        if not mobile:
            raise EnsembleError(
                f"the binding site of {ref.label} and {conformation.label} share no "
                f"fitted atom (selection {atoms!r}); try --site-atoms backbone or all"
            )
        mobile_atoms = np.vstack(mobile)
        target_atoms = np.vstack(target)
        rotation, translation, site_rmsd = kabsch(mobile_atoms, target_atoms)

        # Global CA RMSD over every matched residue, *after* that site fit: it
        # says how much of the protein moved while the site stayed put.
        global_rmsd, n_ca = _global_ca_rmsd(ref, conformation, chains, rotation, translation)
        displacement = _site_displacement(pairs, rotation, translation)

        if site_rmsd > 2.0:
            warnings.append(
                f"{conformation.label}: the binding site itself moves {site_rmsd:.2f} A "
                f"RMSD onto {ref.label}. That is real conformational change, and it "
                "is also approaching the limit at which one box can serve both "
                "frames."
            )
        if n_ca and global_rmsd > 5.0:
            warnings.append(
                f"{conformation.label}: the whole protein moves {global_rmsd:.2f} A "
                f"(CA) while the site moves {site_rmsd:.2f} A; the ensemble spans a "
                "large conformational change, so a single global box may not cover "
                "every pose."
            )
        alignments.append(
            Alignment(
                label=conformation.label, reference=ref.label, identity=identity,
                overlap=overlap, matched_residues=matched, common_residues=common,
                chain_alignments=chains, n_site_residues=len(pairs),
                n_site_atoms=int(mobile_atoms.shape[0]), site_rmsd=float(site_rmsd),
                global_rmsd=float(global_rmsd),
                displacement={label: displacement.get(label, float("nan")) for label in site_labels},
                rotation=rotation.tolist(), translation=translation.tolist(),
            )
        )
        fitted.append(
            conformation.transformed(rotation, translation) if superpose else conformation
        )

    site_labels = [s.label for s in site_residues]
    ensemble = AlignedEnsemble(
        conformations=fitted, alignments=alignments, site=site_labels,
        reference=reference_index, box=box, atom_selection=atoms, warnings=warnings,
    )
    if box is not None:
        complaints = []
        for conformation in ensemble.conformations:
            complaint = _box_complaint(box, conformation)
            if complaint:
                complaints.append(complaint)
        if complaints and not allow_box_mismatch:
            raise EnsembleError(
                "the search box does not touch every conformation:\n       "
                + "\n       ".join(complaints)
                + "\n       Pass --allow-box-mismatch to dock anyway."
            )
        warnings.extend(complaints)
    return ensemble


def _count_site_atoms(site: Sequence[SiteResidue], atoms: str) -> int:
    total = 0
    for entry in site:
        for atom in entry.residue.atoms:
            if atom.is_hydrogen:
                continue
            if atoms == "ca" and atom.name != "CA":
                continue
            if atoms == "backbone" and atom.name not in BACKBONE_ATOMS:
                continue
            total += 1
    return total


def _map_site(
    site: Sequence[SiteResidue],
    chains: Sequence[ChainAlignment],
    conformation: Conformation,
) -> Tuple[List[Tuple[Residue, Residue]], List[str]]:
    """Locate every site residue of the reference inside `conformation`.

    The correspondence comes from the sequence alignment, not from the residue
    numbers: two crystal structures of one protein frequently number their
    residues differently, and a sequence alignment is the thing that is actually
    right about which residue is which.
    """
    out: List[Tuple[Residue, Residue]] = []
    labels: List[str] = []
    lookup: Dict[Tuple[str, int], Residue] = {}
    for alignment in chains:
        other_residues = conformation.standard_residues(alignment.other_chain)
        for i, j in alignment.pairs():
            if j < len(other_residues):
                lookup[(alignment.chain, i)] = other_residues[j]
    for entry in site:
        other = lookup.get((entry.chain, entry.index))
        if other is None:
            continue
        out.append((entry.residue, other))
        labels.append(entry.label)
    return out, labels


def _global_ca_rmsd(
    ref: Conformation,
    other: Conformation,
    chains: Sequence[ChainAlignment],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Tuple[float, int]:
    """CA RMSD of all matched residues after applying ``R, t`` to `other`."""
    left: List[Tuple[float, float, float]] = []
    right: List[Tuple[float, float, float]] = []
    for alignment in chains:
        ref_residues = ref.standard_residues(alignment.chain)
        other_residues = other.standard_residues(alignment.other_chain)
        for i, j in alignment.pairs():
            first = _ca_of(other_residues[j])
            second = _ca_of(ref_residues[i])
            if first is None or second is None:
                continue
            left.append(first)
            right.append(second)
    if len(left) < 3:
        return 0.0, 0
    p = np.asarray(left, dtype=float)
    q = np.asarray(right, dtype=float)
    moved = p @ rotation.T + translation
    return float(np.sqrt(((moved - q) ** 2).sum(axis=1).mean())), len(left)


def _ca_of(residue: Residue) -> Optional[Tuple[float, float, float]]:
    for atom in residue.atoms:
        if atom.name == "CA" and atom.element.strip().upper() == "C":
            return (atom.x, atom.y, atom.z)
    return None


def _site_displacement(
    pairs: Sequence[Tuple[Residue, Residue]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Dict[str, float]:
    """Per-residue heavy-atom RMSD of the site after the rigid fit."""
    out: Dict[str, float] = {}
    for ref_residue, other_residue in pairs:
        reference_atoms, other_atoms, _names = _pair_residue_atoms(
            ref_residue, other_residue, atoms="all", heavy_only=True
        )
        if reference_atoms.shape[0] == 0:
            out[ref_residue.label] = float("nan")
            continue
        moved = other_atoms @ rotation.T + translation
        out[ref_residue.label] = float(
            np.sqrt(((moved - reference_atoms) ** 2).sum(axis=1).mean())
        )
    return out


def _box_complaint(box: BoxSpec, conformation: Conformation) -> Optional[str]:
    """The sharp box-versus-frame test, on parsed atoms instead of raw text."""
    half = tuple(float(size) / 2.0 for size in box.size)
    inside = 0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for atom in conformation.atoms:
        point = (atom.x, atom.y, atom.z)
        for axis in range(3):
            lo[axis] = min(lo[axis], point[axis])
            hi[axis] = max(hi[axis], point[axis])
        if all(abs(point[i] - float(box.center[i])) <= half[i] for i in range(3)):
            inside += 1
    if inside:
        return None
    return (
        f"no atom of {conformation.label} lies inside the search box (0 of "
        f"{len(conformation.atoms)} atoms; box centre "
        f"({box.center[0]:.2f}, {box.center[1]:.2f}, {box.center[2]:.2f}), size "
        f"({box.size[0]:.1f}, {box.size[1]:.1f}, {box.size[2]:.1f}) A; the receptor "
        f"spans x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}, "
        f"z {lo[2]:.1f}..{hi[2]:.1f} A). The superposition and the box disagree: "
        "pass a box in the reference frame."
    )


# ---------------------------------------------------------------------------
# The dockable ensemble
# ---------------------------------------------------------------------------


def _read_text(source: Union[PathLike, str]) -> str:
    """The text of `source`: the file's contents when it is a path, else itself."""
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - a path too long to be a path
            pass
    return str(source)


_SCRATCH: List[str] = []


def _scratch_dir() -> Path:
    """A temporary directory that lives as long as the process."""
    path = tempfile.mkdtemp(prefix="odock-ensemble-")
    _SCRATCH.append(path)
    return Path(path)


@atexit.register
def _cleanup_scratch() -> None:  # pragma: no cover - process teardown
    for path in _SCRATCH:
        shutil.rmtree(path, ignore_errors=True)


@dataclass
class Ensemble:
    """A receptor set that is ready to dock into: one frame, one box, many PDBQTs."""

    labels: List[str]
    receptors: List[str]
    box: BoxSpec
    paths: List[Optional[Path]] = field(default_factory=list)
    alignment: Optional[AlignedEnsemble] = None
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.receptors):
            raise EnsembleError(
                f"{len(self.labels)} label(s) for {len(self.receptors)} receptor(s)"
            )
        if not self.paths:
            self.paths = [None] * len(self.receptors)

    @property
    def n_conformations(self) -> int:
        return len(self.receptors)

    def table(self) -> str:
        rows = []
        for label, text, path in zip(self.labels, self.receptors, self.paths):
            atoms = sum(1 for line in text.splitlines() if line.startswith(("ATOM", "HETATM")))
            rows.append([
                label,
                str(atoms),
                "--" if path is None else str(path),
                "" if self.alignment is None else f"{self.alignment.alignments[self.labels.index(label)].site_rmsd:.3f}",
            ])
        headers = ["conformation", "atoms", "pdbqt", "site RMSD"]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "paths": [None if p is None else str(p) for p in self.paths],
            "box": self.box.as_dict(),
            "n_conformations": self.n_conformations,
            "alignment": None if self.alignment is None else self.alignment.as_dict(),
            "warnings": list(self.warnings),
        }


def build_ensemble(
    sources: Union[PathLike, Sequence[PathLike]],
    *,
    box: BoxSpec,
    outdir: Optional[PathLike] = None,
    superpose: bool = True,
    reference: int = 0,
    site: Optional[str] = None,
    site_ligand: Optional[str] = None,
    site_ligand_coords: Optional[np.ndarray] = None,
    site_radius: float = DEFAULT_SITE_RADIUS,
    max_site_residues: int = 40,
    atoms: str = "ca",
    strip: Optional[Sequence[str]] = None,
    keep_water: bool = False,
    keep_hetero: bool = True,
    min_identity: float = 0.90,
    min_residues: int = 10,
    min_site_residues: int = 3,
    allow_box_mismatch: bool = False,
) -> Ensemble:
    """Turn receptor files into a dockable :class:`Ensemble`.

    The inputs may be several PDB/PDBQT files, or one multi-model PDB; they are
    validated as one receptor, superposed onto a common binding-site frame
    (:func:`align_conformations`) and prepared into receptor PDBQTs.

    Two paths deliberately avoid touching a byte of the user's files:

    * a single input, unchanged -- an ensemble of one is just a normal run;
    * a set of inputs that are already ``.pdbqt`` when ``superpose=False`` -- the
      common "these frames are already aligned" case.  Screening hashes the
      receptor files it is given, so passing the originals through keeps the
      campaign resumable.

    Everything else is written into `outdir` (or a scratch directory), and the
    write is skipped when the file already holds the same bytes, so a resumed
    campaign keeps its receptor signatures and therefore resumes.
    """
    if not isinstance(box, BoxSpec):
        raise EnsembleError(f"box must be a BoxSpec, got {type(box).__name__}")
    paths = [sources] if isinstance(sources, (str, os.PathLike)) else list(sources)
    if not paths:
        raise EnsembleError("no receptor file given: pass at least one PDB/PDBQT")
    conformations = read_conformations(paths, keep_water=keep_water)
    warnings: List[str] = []
    if len(conformations) < 2:
        warnings.append(
            "only one conformation: this is an ordinary single-structure docking "
            "run, and no cross-conformation evidence is generated"
        )

    sites: Optional[np.ndarray] = site_ligand_coords
    if sites is None and site_ligand:
        sites = ligand_coords(conformations[int(reference)], site_ligand)

    already_pdbqt = all(
        str(conformation.source).lower().endswith(".pdbqt") for conformation in conformations
    )
    pass_through = already_pdbqt and not superpose

    if pass_through:
        labels = [c.label for c in conformations]
        receptors = [_read_text(c.source) for c in conformations]
        alignment = None
    elif len(conformations) == 1:
        # One conformation needs no fit -- but it does need preparing, exactly
        # like the conformations of a larger set.
        only = conformations[0]
        alignment = AlignedEnsemble(
            conformations=[only],
            alignments=[
                Alignment(
                    label=only.label, reference=only.label, identity=1.0, overlap=1.0,
                    matched_residues=only.n_residues(), common_residues=only.n_residues(),
                    reference_frame=True,
                )
            ],
            site=[], reference=0, box=box, atom_selection=atoms,
        )
        directory = Path(outdir) if outdir is not None else _scratch_dir()
        strip_names = _strip_names(only, box, strip)
        if strip_names:
            warnings.append(
                f"{only.label}: stripped {', '.join(sorted(strip_names))} from the "
                "receptor — a residue inside the search box cannot be part of the "
                "receptor being docked into"
            )
        written = alignment.receptor_pdbqt(
            0, outdir=directory, strip=sorted(strip_names), keep_water=keep_water,
            keep_hetero=keep_hetero,
        )
        return _finish_ensemble(
            [only.label], [written.read_text(encoding="utf-8")], [written], box,
            alignment, warnings, allow_box_mismatch,
        )
    else:
        alignment = align_conformations(
            conformations, reference=int(reference), box=box, site=site,
            site_ligand=sites, site_radius=site_radius,
            max_site_residues=max_site_residues, atoms=atoms, superpose=superpose,
            min_identity=min_identity, min_residues=min_residues,
            min_site_residues=min_site_residues, allow_box_mismatch=allow_box_mismatch,
        )
        warnings.extend(alignment.warnings)
        labels = alignment.labels
        directory = Path(outdir) if outdir is not None else _scratch_dir()
        receptors = []
        receptor_paths: List[Optional[Path]] = []
        for index, conformation in enumerate(alignment.conformations):
            strip_names = _strip_names(conformation, box, strip)
            if strip_names:
                warnings.append(
                    f"{conformation.label}: stripped {', '.join(sorted(strip_names))} "
                    "from the receptor — a residue inside the search box cannot be "
                    "part of the receptor being docked into"
                )
            written = alignment.receptor_pdbqt(
                index, outdir=directory, strip=sorted(strip_names),
                keep_water=keep_water, keep_hetero=keep_hetero,
            )
            receptors.append(written.read_text(encoding="utf-8"))
            receptor_paths.append(written)
        return _finish_ensemble(
            labels, receptors, receptor_paths, box, alignment, warnings, allow_box_mismatch
        )

    receptor_paths = [Path(str(c.source)) for c in conformations]
    return _finish_ensemble(
        labels, receptors, receptor_paths, box, alignment, warnings, allow_box_mismatch
    )


def _strip_names(
    conformation: Conformation, box: BoxSpec, explicit: Optional[Sequence[str]]
) -> set:
    """Residue names to delete from `conformation` before docking into `box`.

    The explicit ``strip`` list wins.  On top of it, any *organic* non-standard
    residue that reaches into the search box is stripped: a co-crystallised
    inhibitor sitting in the site being docked into is not part of the receptor,
    and leaving it there makes the run meaningless (every pose clashes with it).
    Ions, phosphate and other small groups are kept -- they are usually part of
    the site, and metals are explicitly wanted by the force field.
    """
    out = {str(name).strip().upper() for name in (explicit or ()) if str(name).strip()}
    half = tuple(float(size) / 2.0 for size in box.size)
    for residues in conformation.chain_residues().values():
        for residue in residues:
            if residue.is_standard or residue.is_water:
                continue
            heavy = [a for a in residue.atoms if not a.is_hydrogen]
            if len(heavy) < 6:
                continue
            inside = any(
                all(abs(coordinate - float(box.center[i])) <= half[i] for i, coordinate in enumerate((a.x, a.y, a.z)))
                for a in heavy
            )
            if inside:
                out.add(residue.res_name.strip().upper())
    return out


def _finish_ensemble(
    labels: Sequence[str],
    receptors: Sequence[str],
    paths: Sequence[Optional[Path]],
    box: BoxSpec,
    alignment: Optional[AlignedEnsemble],
    warnings: List[str],
    allow_box_mismatch: bool,
) -> Ensemble:
    """Validate the shared box against every receptor, then build the ensemble."""
    from .screen import receptor_atoms_in_box

    complaints: List[str] = []
    counts: List[int] = []
    for label, text in zip(labels, receptors):
        inside = receptor_atoms_in_box(text, box)
        counts.append(inside)
        if not inside:
            complaints.append(
                f"no receptor atom of {label} lies inside the search box (box centre "
                f"({box.center[0]:.2f}, {box.center[1]:.2f}, {box.center[2]:.2f}), size "
                f"({box.size[0]:.1f}, {box.size[1]:.1f}, {box.size[2]:.1f}) A)"
            )
    if complaints:
        detail = "\n       ".join(complaints)
        if not allow_box_mismatch:
            raise EnsembleError(
                f"the search box does not touch {len(complaints)} of the "
                f"{len(labels)} conformation(s):\n       {detail}\n"
                "       One box must describe the same site in every frame: "
                "superpose the set (--superpose), or pass a box in the reference "
                "frame. Pass --allow-box-mismatch to dock anyway."
            )
        warnings.extend(complaints)
    if len(set(counts)) > 1 and min(counts) > 0:
        warnings.append(
            "the box holds a different number of atoms in different conformations "
            f"({min(counts)}..{max(counts)}): that is the binding site moving, and it "
            "is the point of the ensemble — but a pose that fits one frame may clash "
            "in another"
        )
    return Ensemble(
        labels=list(labels), receptors=list(receptors), box=box, paths=list(paths),
        alignment=alignment, warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Docking every conformation, and merging the poses
# ---------------------------------------------------------------------------


@dataclass
class EnsemblePose:
    """One pose of one conformation, in the merged ensemble result."""

    ligand: str
    conformation: str
    conformation_index: int
    pose_index: int
    affinity: float
    coords: np.ndarray = field(repr=False)
    elements: List[str] = field(default_factory=list)
    cluster: int = -1
    rank: int = 0
    #: The pose's own conformation's rank of it (1 = that conformation's best).
    local_rank: int = 0
    #: Affinity of this pose in every conformation (kcal/mol); ``nan`` where the
    #: pose was not rescored there (see :func:`cross_rescore`).
    cross: Dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.conformation}#{self.pose_index + 1}"

    def as_dict(self, *, include_coords: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "rank": int(self.rank),
            "conformation": self.conformation,
            "conformation_index": int(self.conformation_index),
            "pose": int(self.pose_index + 1),
            "local_rank": int(self.local_rank),
            "affinity": self.affinity,
            "cluster": int(self.cluster),
        }
        if self.cross:
            data["cross_affinity"] = dict(self.cross)
            data["cross_spread"] = self.cross_spread
        if include_coords:
            data["coords"] = np.asarray(self.coords, dtype=float).tolist()
        return data

    @property
    def cross_spread(self) -> float:
        """``max - min`` of the affinity over the conformations it was scored in."""
        values = [v for v in self.cross.values() if v is not None and math.isfinite(v)]
        return (max(values) - min(values)) if len(values) > 1 else 0.0


@dataclass
class PoseCluster:
    """A binding mode: poses from one or more conformations that superpose."""

    index: int
    members: List[int]
    representative: int
    best_affinity: float
    mean_affinity: float
    spread: float
    conformations: List[str] = field(default_factory=list)
    max_rmsd: float = 0.0

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def n_conformations(self) -> int:
        return len(self.conformations)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cluster": int(self.index),
            "size": self.size,
            "n_conformations": self.n_conformations,
            "conformations": list(self.conformations),
            "representative": int(self.representative),
            "best_affinity": self.best_affinity,
            "mean_affinity": self.mean_affinity,
            "spread": self.spread,
            "max_rmsd": self.max_rmsd,
        }


@dataclass
class EnsembleDockResult:
    """Every pose from every conformation of one ligand, merged and clustered."""

    ligand: str
    labels: List[str]
    results: List[Any]
    poses: List[EnsemblePose]
    clusters: List[PoseCluster]
    box: BoxSpec
    scoring: str = "vina"
    elapsed: float = 0.0
    seed: int = 0
    ligand_pdbqt: str = ""
    alignment: Optional[AlignedEnsemble] = None
    robustness: Optional["RobustnessScore"] = None
    rescoring: Optional["CrossRescoring"] = None
    consensus: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # -- accessors --------------------------------------------------------

    def best(self) -> Optional[EnsemblePose]:
        """The best-scoring pose of the whole ensemble."""
        return self.poses[0] if self.poses else None

    @property
    def best_affinity(self) -> Optional[float]:
        pose = self.best()
        return None if pose is None else pose.affinity

    def best_per_conformation(self) -> Dict[str, Optional[float]]:
        """The best affinity found in each conformation."""
        out: Dict[str, Optional[float]] = {label: None for label in self.labels}
        for pose in self.poses:
            current = out.get(pose.conformation)
            if current is None or pose.affinity < current:
                out[pose.conformation] = pose.affinity
        return out

    def winning_conformation(self) -> Optional[str]:
        pose = self.best()
        return None if pose is None else pose.conformation

    def top_cluster(self) -> Optional[PoseCluster]:
        return self.clusters[0] if self.clusters else None

    def cluster_of(self, pose: EnsemblePose) -> Optional[PoseCluster]:
        for cluster in self.clusters:
            if cluster.index == pose.cluster:
                return cluster
        return None

    # -- reporting --------------------------------------------------------

    def conformation_table(self) -> str:
        """Per conformation: best affinity, number of poses, seed, wall time."""
        rows = []
        for label, result in zip(self.labels, self.results):
            best = result.best_affinity
            rows.append([
                label,
                "--" if best is None else f"{best:.3f}",
                str(len(result.poses)),
                str(result.seed),
                f"{result.elapsed:.2f}",
            ])
        headers = ["conformation", "best affinity", "poses", "seed", "seconds"]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def cluster_table(self, limit: int = 12) -> str:
        """The binding modes found across the ensemble, best first."""
        rows = []
        for cluster in self.clusters[: max(1, limit)]:
            rows.append([
                str(cluster.index + 1),
                str(cluster.size),
                str(cluster.n_conformations),
                f"{cluster.best_affinity:.3f}",
                f"{cluster.mean_affinity:.3f}",
                f"{cluster.spread:.3f}",
                f"{cluster.max_rmsd:.2f}",
                ",".join(cluster.conformations),
            ])
        headers = [
            "cluster", "poses", "confs", "best", "mean", "spread (A)",
            "max RMSD (A)", "conformations",
        ]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def table(self, limit: int = 20) -> str:
        """The merged ranking: every pose of every conformation, best first."""
        rows = []
        for pose in self.poses[: max(1, limit)]:
            rows.append([
                str(pose.rank),
                pose.conformation,
                str(pose.pose_index + 1),
                f"{pose.affinity:.3f}",
                str(pose.cluster + 1),
                f"{len(self.clusters[pose.cluster].members) if 0 <= pose.cluster < len(self.clusters) else 0}",
            ])
        headers = ["rank", "conformation", "pose", "affinity", "cluster", "cluster size"]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def text(self, *, limit: int = 20) -> str:
        """The whole human-readable report."""
        lines = [
            f"OpenDocking ensemble docking — {self.ligand}, "
            f"{len(self.labels)} conformation(s), scoring {self.scoring}",
            f"box: {self.box}",
            f"seed: {self.seed} (the same seed for every conformation)",
            "",
            self.conformation_table(),
        ]
        best = self.best()
        if best is None:
            lines.append("")
            lines.append("no pose: every conformation failed")
            return "\n".join(lines)
        lines += [
            "",
            f"best overall: {best.affinity:.3f} kcal/mol from {best.conformation} "
            f"(pose {best.pose_index + 1})",
            "best per conformation: "
            + ", ".join(
                f"{label} {value:.3f}" if value is not None else f"{label} --"
                for label, value in self.best_per_conformation().items()
            ),
            "",
            f"merged ranking ({len(self.poses)} pose(s) from "
            f"{len(self.labels)} conformation(s)):",
            self.table(limit),
            "",
            "binding modes across the ensemble:",
            self.cluster_table(),
        ]
        if self.robustness is not None:
            lines += ["", self.robustness.text()]
        if self.rescoring is not None:
            lines += ["", self.rescoring.text()]
        if self.warnings:
            lines += [""] + [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def as_dict(self, *, full: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "ligand": self.ligand,
            "scoring": self.scoring,
            "seed": int(self.seed),
            "box": self.box.as_dict(),
            "labels": list(self.labels),
            "elapsed": round(float(self.elapsed), 4),
            "best_affinity": self.best_affinity,
            "winning_conformation": self.winning_conformation(),
            "best_per_conformation": self.best_per_conformation(),
            "clusters": [c.as_dict() for c in self.clusters],
            "robustness": None if self.robustness is None else self.robustness.as_dict(),
            "rescoring": None if self.rescoring is None else self.rescoring.as_dict(),
            "consensus": self.consensus or None,
            "warnings": list(self.warnings),
        }
        if full:
            data["poses"] = [p.as_dict() for p in self.poses]
        return data

    def to_pdbqt(self, *, limit: Optional[int] = None) -> str:
        """Every merged pose as one multi-model PDBQT, in merged rank order.

        Each model keeps the pose's own ``REMARK VINA RESULT`` line, so the file
        is readable by the rest of the toolkit (:func:`odock.cli._read_poses`,
        ``odock cluster``, :mod:`odock.report`), and carries an extra
        ``REMARK  ODOCK ENSEMBLE:`` line naming the conformation, the cluster and
        the per-conformation rank that produced it.
        """
        from .consensus import pose_pdbqt_from_coords, pdbqt_models

        blocks = []
        chosen = self.poses if limit is None else self.poses[: int(limit)]
        for number, pose in enumerate(chosen, start=1):
            template = self.ligand_pdbqt
            text = pose_pdbqt_from_coords(template, pose.coords) if template else ""
            if not text:
                models = pdbqt_models(self.ligand_pdbqt)
                text = models[0] if models else ""
            if not text:
                continue
            lines = text.splitlines()
            remark = (
                f"REMARK  ODOCK ENSEMBLE: conformation={pose.conformation} "
                f"pose={pose.pose_index + 1} cluster={pose.cluster + 1} "
                f"rank={pose.rank}"
            )
            if pose.cross:
                remark += " cross_spread=%.3f" % pose.cross_spread
            head = lines[:1]
            body = lines[1:]
            result_line = f"REMARK  VINA RESULT:    {pose.affinity:.3f}      0.000      0.000"
            block = head + [result_line, remark] + body
            blocks.append(f"MODEL     {number}\n" + "\n".join(block) + "\nENDMDL\n")
        return "".join(blocks)


def _open_engine(
    receptor_text: str,
    ligand_text: str,
    box: BoxSpec,
    *,
    scoring: str,
    exhaustiveness: int,
    num_poses: int,
    seed: int,
    use_grid: bool,
    refine: bool,
    min_rmsd: float,
    energy_range: float,
    search: Optional[str],
    islands: int,
    population: int,
    generations: int,
):
    """Build the kernel engine for one conformation.

    :func:`odock.docking.dock` is deliberately not used here: it re-interprets a
    string argument as a path when it does not look like PDBQT, and this module
    always holds text.  The engine is built directly and converted back with
    :func:`odock.docking.result_from_engine`, so a pose is the same object the
    single-structure path produces.
    """
    from . import _odock

    return _odock.Docking(
        receptor_text,
        ligand_text,
        center=tuple(float(x) for x in box.center),
        size=tuple(float(x) for x in box.size),
        spacing=float(box.spacing),
        scoring=str(scoring),
        exhaustiveness=int(exhaustiveness),
        num_poses=int(num_poses),
        seed=int(seed),
        use_grid=bool(use_grid),
        refine=bool(refine),
        min_rmsd=float(min_rmsd),
        energy_range=float(energy_range),
        use_island_ga=str(search or "") == "lga",
        islands=int(islands),
        population=int(population),
        generations=int(generations),
        **({"search": str(search)} if search is not None else {}),
    )


def _ligand_elements(ligand_text: str) -> List[str]:
    """Per-atom element labels of a ligand PDBQT, in file (= kernel) order."""
    from .consensus import pdbqt_atoms, pdbqt_models

    models = pdbqt_models(ligand_text) or [ligand_text]
    return [atom.element for atom in pdbqt_atoms(models[0])]


def _ligand_pdbqt(
    ligand: Union[PathLike, str], name: Optional[str] = None
) -> Tuple[str, str]:
    """``(pdbqt text, name)`` for a ligand, preparing it when it is not a PDBQT.

    Accepting an SDF here is not sugar: the co-crystallised ligand of a structure
    is usually at hand as an SDF (an ideal-geometry component) or as a residue of
    the receptor, and forcing the caller to run ``odock prepare ligand`` first
    would just move the same call one command earlier.  The preparation itself is
    :func:`odock.prepare.prepare_ligand`, so the ligand the ensemble docks is the
    ligand every other command docks.
    """
    text = _read_text(ligand)
    if not text.strip():
        raise EnsembleError(f"{ligand} is empty")
    path = Path(str(ligand)) if isinstance(ligand, (str, os.PathLike)) else None
    resolved_name = name or (path.stem if path is not None and path.is_file() else "ligand")
    pdbqt = _looks_like_pdbqt(text, path)
    if pdbqt:
        return text, resolved_name
    from .prepare import prepare_ligand

    source: Any = str(path) if path is not None and path.is_file() else text
    _mol, prepared, _report = prepare_ligand(source, None, name=resolved_name)
    if not prepared.strip():
        raise EnsembleError(f"{ligand} could not be prepared into a ligand PDBQT")
    return prepared, resolved_name


def dock_ensemble(
    ligand: Union[PathLike, str],
    ensemble: Ensemble,
    *,
    scoring: str = "vina",
    exhaustiveness: int = 8,
    num_poses: int = 9,
    seed: int = 0,
    min_rmsd: float = 1.0,
    energy_range: float = 3.0,
    use_grid: bool = True,
    refine: bool = True,
    search: Optional[str] = None,
    islands: int = 4,
    population: int = 32,
    generations: int = 20,
    cluster_rmsd: float = DEFAULT_CLUSTER_RMSD,
    ligand_name: Optional[str] = None,
    progress: Optional[Any] = None,
) -> EnsembleDockResult:
    """Dock one ligand into every conformation and merge the poses.

    The **same** `seed` is used for every conformation, on purpose: the ensemble
    is a paired comparison, and a different seed per conformation would confound
    the receptor's motion with the search's own randomness.  Two consequences are
    stated in ``docs/ENSEMBLE.md``: the winner can still be seed luck, and a
    repeated run is the way to measure that (``odock ensemble dock`` runs the
    whole set under one seed, and repeats are the caller's business).

    Returns
    -------
    :class:`EnsembleDockResult` with the merged ranking, the cross-conformation
    clusters, and -- because it is free once the poses exist -- nothing else; call
    :func:`cross_rescore` / :func:`robustness_score` for the robustness numbers.
    """
    import time as _time

    from .docking import result_from_engine

    ligand_text, name = _ligand_pdbqt(ligand, ligand_name)
    elements = _ligand_elements(ligand_text)
    started = _time.perf_counter()
    results = []
    poses: List[EnsemblePose] = []
    for index, (label, receptor_text) in enumerate(zip(ensemble.labels, ensemble.receptors)):
        box = ensemble.box
        engine = _open_engine(
            receptor_text, ligand_text, box, scoring=scoring,
            exhaustiveness=exhaustiveness, num_poses=num_poses, seed=seed,
            use_grid=use_grid, refine=refine, min_rmsd=min_rmsd,
            energy_range=energy_range, search=search, islands=islands,
            population=population, generations=generations,
        )
        t0 = _time.perf_counter()
        raw = engine.run()
        result = result_from_engine(
            engine, raw, elapsed=_time.perf_counter() - t0, box=box,
            receptor_pdbqt=receptor_text, ligand_pdbqt=ligand_text,
            energy_range=energy_range, scoring=scoring,
        )
        results.append(result)
        for pose_index, pose in enumerate(result.poses):
            coords = pose.coords
            if coords is None:
                continue
            poses.append(
                EnsemblePose(
                    ligand=name, conformation=label, conformation_index=index,
                    pose_index=pose_index, affinity=float(pose.affinity),
                    coords=np.asarray(coords, dtype=float), elements=list(elements),
                    local_rank=pose_index + 1,
                )
            )
        if progress is not None:
            try:
                progress(label, index + 1, len(ensemble.labels), result)
            except Exception:  # pragma: no cover - a reporter must not kill a run
                pass

    clusters = cluster_ensemble_poses(poses, cutoff=cluster_rmsd)
    # Merged ranking: best affinity first, ties broken by conformation then pose.
    order = sorted(
        range(len(poses)),
        key=lambda k: (poses[k].affinity, poses[k].conformation_index, poses[k].pose_index),
    )
    for rank, position in enumerate(order, start=1):
        poses[position].rank = rank
    merged = [poses[k] for k in order]
    # The clusters were built over `poses` in docking order; the result exposes
    # the merged order, so the member indices must move with the poses or every
    # later lookup (robustness, cross-rescoring) would read the wrong row.
    remap = {old: new for new, old in enumerate(order)}
    for cluster in clusters:
        cluster.members = [remap[member] for member in cluster.members]
        cluster.representative = remap[cluster.representative]
    return EnsembleDockResult(
        ligand=name, labels=list(ensemble.labels), results=results, poses=merged,
        clusters=clusters, box=ensemble.box, scoring=scoring,
        elapsed=_time.perf_counter() - started, seed=int(seed),
        ligand_pdbqt=ligand_text, alignment=ensemble.alignment,
        warnings=list(ensemble.warnings),
    )


def cluster_ensemble_poses(
    poses: Sequence[EnsemblePose], *, cutoff: float = DEFAULT_CLUSTER_RMSD
) -> List[PoseCluster]:
    """Cluster poses pooled from every conformation, by symmetry-aware RMSD.

    The grouping is :func:`odock.analysis.cluster_poses` -- single-linkage on the
    symmetry-aware no-superposition RMSD, the project's own pose comparison -- and
    all poses are already in one frame (the conformations were superposed), so
    "same mode in two receptors" means what it says.  What this function adds is
    the cross-conformation bookkeeping: how many *different* conformations
    support a cluster, and how far apart the cluster's members are.
    """
    if not poses:
        return []
    from .analysis import cluster_poses, symmetry_aware_rmsd

    coords = np.asarray([p.coords for p in poses], dtype=float)
    elements = list(poses[0].elements) if poses[0].elements else None
    energies = [p.affinity for p in poses]
    raw = cluster_poses(coords, cutoff=cutoff, elements=elements, energies=energies)

    out: List[PoseCluster] = []
    for rank, cluster in enumerate(raw):
        members = list(cluster.members)
        for member in members:
            poses[member].cluster = rank
        conformations: List[str] = []
        for member in members:
            label = poses[member].conformation
            if label not in conformations:
                conformations.append(label)
        values = [poses[m].affinity for m in members]
        spread = (max(values) - min(values)) if len(values) > 1 else 0.0
        max_rmsd = 0.0
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                distance = symmetry_aware_rmsd(
                    coords[members[a]], coords[members[b]], elements or ["C"] * coords.shape[1]
                )
                max_rmsd = max(max_rmsd, float(distance))
        out.append(
            PoseCluster(
                index=rank, members=members, representative=int(cluster.representative),
                best_affinity=float(min(values)) if values else float("nan"),
                mean_affinity=float(np.mean(values)) if values else float("nan"),
                spread=float(spread), conformations=conformations,
                max_rmsd=float(max_rmsd),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Cross-receptor rescoring and the robustness score
# ---------------------------------------------------------------------------

#: The kcal/mol scale of the robustness penalty.  A pose whose affinity moves by
#: this much when the receptor is swapped keeps ``exp(-1) = 0.37`` of its
#: support-weighted score.  It is a *convention* chosen so the number lands in
#: ``[0, 1]`` with a useful spread, not a fitted constant; the raw spread is
#: always reported next to it so nobody has to trust the scale.
ROBUSTNESS_SCALE = 2.0

#: A receptor-swap spread above this many kcal/mol is not "the number moved", it
#: is "this pose cannot be placed in that conformation at all": the atoms are
#: inside the protein.  Counting those separately keeps one broken pose from
#: silently dominating the summary -- which is why the score uses the *median*
#: spread and the count is reported next to it.
CLASH_SPREAD = 10.0


@dataclass
class CrossRescoring:
    """Every pose evaluated in every conformation, at fixed coordinates.

    This is the measurement behind "is this pose robust to the receptor moving?".
    No search and no refinement happen: the pose is scored exactly where the
    docking put it, in each receptor's frame, with the kernel's exact pairwise
    scorer (:func:`odock.consensus.rescore_poses`).  A pose that only exists
    because one particular side chain happens to be in one particular place pays
    for it here, in kcal/mol.
    """

    labels: List[str]
    scoring: str
    affinities: np.ndarray
    #: The conformation each row (pose) was docked into.
    owns: List[str] = field(default_factory=list)
    refine: bool = False

    @property
    def shape(self) -> Tuple[int, int]:
        """``(poses, conformations)``: the sample size of the measurement."""
        return (int(self.affinities.shape[0]), int(self.affinities.shape[1]))

    def own(self, row: int) -> float:
        """The affinity this pose has in its *own* conformation."""
        if not self.owns:
            return float("nan")
        column = self.labels.index(self.owns[row])
        return float(self.affinities[row, column])

    def spread(self, row: int) -> float:
        """``max - min`` over the conformations, in kcal/mol."""
        values = self.affinities[row]
        values = values[np.isfinite(values)]
        return float(values.max() - values.min()) if values.size > 1 else 0.0

    def worst_penalty(self, row: int) -> float:
        """How much the pose *loses* in the worst conformation, relative to its own.

        ``max_k E_k - E_own``: zero when the pose's own receptor is its worst; a
        positive number is the price of putting the same coordinates in another
        frame.
        """
        values = self.affinities[row]
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        return float(values.max() - self.own(row))

    def best_conformation(self, row: int) -> Optional[str]:
        values = self.affinities[row]
        finite = np.isfinite(values)
        if not finite.any():
            return None
        return self.labels[int(np.argmin(np.where(finite, values, np.inf)))]

    def mean_spread(self) -> float:
        if not self.shape[0]:
            return float("nan")
        return float(np.mean([self.spread(row) for row in range(self.shape[0])]))

    def rows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        count = self.shape[0] if limit is None else min(int(limit), self.shape[0])
        return [
            {
                "pose": row + 1,
                "conformation": self.owns[row] if row < len(self.owns) else "",
                "own": self.own(row),
                "spread": self.spread(row),
                "worst_penalty": self.worst_penalty(row),
                **{label: float(v) for label, v in zip(self.labels, self.affinities[row])},
            }
            for row in range(count)
        ]

    def table(self, limit: int = 12) -> str:
        """One row per pose, one column per conformation (kcal/mol)."""
        rows = []
        for row in range(min(limit, self.shape[0])):
            cells = [str(row + 1)]
            for column in range(self.shape[1]):
                value = float(self.affinities[row, column])
                cells.append("--" if not math.isfinite(value) else f"{value:.3f}")
            cells.append(f"{self.spread(row):.3f}")
            rows.append(cells)
        headers = ["pose"] + list(self.labels) + ["spread"]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def text(self) -> str:
        return (
            f"cross-receptor rescoring ({self.scoring}, fixed coordinates, "
            f"{self.shape[0]} pose(s) x {self.shape[1]} conformation(s) = "
            f"{self.shape[0] * self.shape[1]} exact scores"
            + (", after refinement" if self.refine else ", no refinement")
            + f"); mean affinity spread on a receptor swap "
            f"{self.mean_spread():.3f} kcal/mol\n"
            + self.table()
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scoring": self.scoring,
            "conformations": list(self.labels),
            "pose_count": self.shape[0],
            "conformation_count": self.shape[1],
            "n_scores": self.shape[0] * self.shape[1],
            "mean_spread": self.mean_spread(),
            "affinities": np.asarray(self.affinities, dtype=float).tolist(),
        }


def cross_rescore(
    result: EnsembleDockResult,
    ensemble: Ensemble,
    *,
    scoring: str = "vina",
    refine: bool = False,
) -> CrossRescoring:
    """Rescore every pose of `result` in every conformation of `ensemble`.

    The rescoring itself is :func:`odock.consensus.rescore_poses`, so a cross
    score is computed by exactly the same code path as a within-conformation
    consensus score: one potential, one box, fixed coordinates.  ``refine=False``
    (the default) is what makes the comparison meaningful -- refining in the new
    receptor would let the pose relax into it and hide the clash it should pay
    for.  Pass ``refine=True`` to measure the relaxed variant instead; the choice
    is recorded on the result.

    Row `i` of the returned matrix is ``result.poses[i]`` -- the merged ranking
    order -- so a cluster's member indices can be used directly.
    """
    from .consensus import pose_pdbqt_from_coords, rescore_poses

    chosen = list(result.poses)
    template = result.ligand_pdbqt
    if not template:
        raise EnsembleError("the result carries no ligand PDBQT template to rebuild poses from")
    documents = []
    for pose in chosen:
        text = pose_pdbqt_from_coords(template, pose.coords)
        if not text:
            raise EnsembleError(
                "a pose could not be rebuilt from the ligand template; the ligand "
                "was prepared differently from the one that was docked"
            )
        documents.append(text)
    matrix = np.full((len(documents), ensemble.n_conformations), float("nan"), dtype=float)
    for column, receptor in enumerate(ensemble.receptors):
        raw = rescore_poses(
            documents, receptor, ensemble.box, [scoring], refine=refine
        )
        matrix[:, column] = np.asarray(raw[scoring]["affinity"], dtype=float)
    rescoring = CrossRescoring(
        labels=list(ensemble.labels), scoring=scoring, affinities=matrix,
        owns=[pose.conformation for pose in chosen], refine=bool(refine),
    )
    for row, pose in enumerate(chosen):
        pose.cross = {
            label: float(matrix[row, column])
            for column, label in enumerate(ensemble.labels)
        }
    return rescoring


def conformation_consensus(
    result: EnsembleDockResult,
    ensemble: Ensemble,
    *,
    scorings: Sequence[str] = ("vina", "vinardo", "ad4"),
    method: str = "rank",
) -> Dict[str, Any]:
    """Rescore each conformation's own poses with every force field.

    This is :func:`odock.consensus.consensus_score` applied *within* each
    conformation -- the pose set of that conformation, that receptor, that box --
    which is what the rest of the project means by consensus.  The ensemble layer
    then combines those per-conformation answers (see :func:`robustness_score`)
    instead of pretending that a rank from one receptor means the same thing as a
    rank from another.  Each :class:`odock.consensus.ConsensusResult` carries its
    own pairwise Spearman correlations, so the sample size travels with every rho.
    """
    from .consensus import consensus_score

    out: Dict[str, Any] = {}
    for label, docked, receptor in zip(ensemble.labels, result.results, ensemble.receptors):
        if not getattr(docked, "poses", None):
            continue
        out[label] = consensus_score(
            docked, receptor, ensemble.box, scorings=tuple(scorings), method=method
        )
    return out


@dataclass
class RobustnessScore:
    """How well the winning binding mode survives the receptor moving.

    The three components are reported separately and combined into one number in
    ``[0, 1]``:

    ``support``
        the fraction of conformations that produced a pose in the winning cluster;
    ``persistence``
        the fraction of conformations whose *own* best pose is in that cluster --
        support says "some pose landed there", persistence says "it was the best
        pose there";
    ``movement``
        the mean affinity spread, in kcal/mol, of the cluster's poses when each is
        scored in every conformation (``cross_spread``), or -- when no
        cross-rescoring was done -- the spread of the cluster's own affinities.

    ``score = support * persistence * exp(-movement / ROBUSTNESS_SCALE)``.

    Everything is accompanied by its sample size: ``n_poses`` poses,
    ``n_scored`` pose x conformation scores, ``n_consensus`` within-conformation
    consensus rescorings.
    """

    ligand: str
    cluster: int
    n_conformations: int
    n_supporting: int
    support: float
    persistence: float
    conformations_supporting: List[str] = field(default_factory=list)
    conformations_persisting: List[str] = field(default_factory=list)
    best_affinity: float = float("nan")
    mean_affinity: float = float("nan")
    affinity_spread: float = float("nan")
    cross_spread: float = float("nan")
    cross_spread_median: float = float("nan")
    worst_penalty: float = float("nan")
    #: How many members of the winning mode cannot be placed in another
    #: conformation at all (a receptor-swap spread above :data:`CLASH_SPREAD`).
    clashing: int = 0
    movement: float = float("nan")
    score: float = float("nan")
    n_poses: int = 0
    n_scored: int = 0
    n_consensus: int = 0
    consensus_persistence: Optional[float] = None
    consensus_agreement: Dict[str, float] = field(default_factory=dict)
    method: str = "support x persistence x exp(-movement / %.1f kcal/mol)" % ROBUSTNESS_SCALE

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ligand": self.ligand,
            "cluster": int(self.cluster),
            "n_conformations": int(self.n_conformations),
            "n_supporting": int(self.n_supporting),
            "support": self.support,
            "persistence": self.persistence,
            "conformations_supporting": list(self.conformations_supporting),
            "conformations_persisting": list(self.conformations_persisting),
            "best_affinity": self.best_affinity,
            "mean_affinity": self.mean_affinity,
            "affinity_spread": self.affinity_spread,
            "cross_spread": self.cross_spread,
            "cross_spread_median": self.cross_spread_median,
            "worst_penalty": self.worst_penalty,
            "clashing": int(self.clashing),
            "movement": self.movement,
            "score": self.score,
            "n_poses": int(self.n_poses),
            "n_scored": int(self.n_scored),
            "n_consensus": int(self.n_consensus),
            "consensus_persistence": self.consensus_persistence,
            "consensus_agreement": dict(self.consensus_agreement),
            "method": self.method,
        }

    def text(self) -> str:
        """The robustness answer, with every number's sample size attached."""
        lines = [
            f"robustness: {self.score:.3f}  "
            f"(support {self.support:.2f} x persistence {self.persistence:.2f} x "
            f"exp(-{_fmt_num(self.movement)}/{ROBUSTNESS_SCALE:.1f}))",
            f"  winning mode: cluster {self.cluster + 1}, {self.n_supporting} of "
            f"{self.n_conformations} conformation(s) support it "
            f"({', '.join(self.conformations_supporting) or '--'}), best "
            f"{_fmt_num(self.best_affinity)} kcal/mol",
            f"  persistence: {self.persistence:.2f} — the best pose of "
            f"{len(self.conformations_persisting)} of {self.n_conformations} "
            f"conformation(s) is in this mode "
            f"({', '.join(self.conformations_persisting) or '--'})",
            f"  affinity movement: {_fmt_num(self.movement)} kcal/mol (median of the "
            f"receptor-swap spreads; mean {_fmt_num(self.cross_spread)}, within the "
            f"ensemble {_fmt_num(self.affinity_spread)}, worst single penalty "
            f"{_fmt_num(self.worst_penalty)})",
            f"  sample size: {self.n_poses} pose(s), {self.n_scored} pose x conformation "
            f"exact score(s), {self.n_consensus} within-conformation consensus "
            "rescorings",
        ]
        if self.clashing:
            lines.append(
                f"  {self.clashing} of {self.n_poses} pose(s) move by more than "
                f"{CLASH_SPREAD:.0f} kcal/mol on a receptor swap: those binding modes "
                "exist in one conformation only"
            )
        if self.consensus_persistence is not None:
            lines.append(
                f"  consensus persistence: {self.consensus_persistence:.2f} — the "
                "three force fields' own top pose inside a conformation is in this "
                "mode that fraction of the time"
            )
        if self.consensus_agreement:
            lines.append(
                "  per-conformation force-field agreement (mean pairwise Spearman "
                "rho, sample size = poses in that conformation): "
                + ", ".join(
                    f"{label} {_fmt_num(value)}"
                    for label, value in self.consensus_agreement.items()
                )
            )
        return "\n".join(lines)


def _fmt_num(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if not math.isfinite(number) else f"{number:.3f}"


def robustness_score(
    result: EnsembleDockResult,
    rescoring: Optional[CrossRescoring] = None,
    *,
    cluster: Optional[int] = None,
    consensus: Optional[Dict[str, Any]] = None,
) -> RobustnessScore:
    """Measure how robust the winning binding mode is.

    Parameters
    ----------
    result
        An :class:`EnsembleDockResult` (its poses must be clustered; they are
        after :func:`dock_ensemble`).
    rescoring
        A :func:`cross_rescore` result.  When given, the affinity movement is the
        receptor-swap spread; when omitted, it falls back to the spread of the
        cluster's own affinities and the cross columns are ``nan`` -- a weaker
        statement, and the text says so.
    cluster
        Which cluster to judge; ``None`` judges the best-scoring one.
    consensus
        A :func:`conformation_consensus` mapping, which adds the per-conformation
        force-field agreement and the consensus persistence.
    """
    if not result.poses:
        raise EnsembleError("no pose to score for robustness")
    target = result.clusters[0] if cluster is None else None
    if cluster is not None:
        for candidate in result.clusters:
            if candidate.index == int(cluster):
                target = candidate
                break
        else:
            raise EnsembleError(
                f"cluster {cluster} does not exist (the ensemble has "
                f"{len(result.clusters)})"
            )
    assert target is not None
    members = [result.poses[index] for index in target.members]
    values = [pose.affinity for pose in members]
    labels = list(result.labels)
    supporting = list(target.conformations)
    persisting = []
    for label in labels:
        own = [pose for pose in result.poses if pose.conformation == label]
        if not own:
            continue
        best = min(own, key=lambda pose: pose.affinity)
        if best.cluster == target.index:
            persisting.append(label)

    movements: List[float] = []
    penalties: List[float] = []
    if rescoring is not None and rescoring.shape[0] == len(result.poses):
        for index in target.members:
            movements.append(rescoring.spread(index))
            penalty = rescoring.worst_penalty(index)
            if math.isfinite(penalty):
                penalties.append(penalty)
    cross_spread = float(np.mean(movements)) if movements else float("nan")
    cross_median = float(np.median(movements)) if movements else float("nan")
    # The median, not the mean: one pose that cannot be placed in another
    # conformation at all (a +200 kcal/mol clash) would otherwise decide the
    # whole score.  The mean, the worst penalty and the clash count are all
    # reported, so nothing is hidden by the choice.
    movement = cross_median if movements else (max(values) - min(values) if len(values) > 1 else 0.0)
    clashing = sum(1 for value in movements if value > CLASH_SPREAD)
    score = float(
        len(supporting) / max(1, len(labels))
        * (len(persisting) / max(1, len(labels)))
        * math.exp(-max(0.0, movement) / ROBUSTNESS_SCALE)
    )

    consensus_persistence: Optional[float] = None
    agreement: Dict[str, float] = {}
    n_consensus = 0
    if consensus:
        hits = 0
        total = 0
        for label, combined in consensus.items():
            n_consensus += 1
            rho = getattr(combined, "agreement", float("nan"))
            if rho is not None and math.isfinite(float(rho)):
                agreement[label] = float(rho)
            top = getattr(combined, "poses", None)
            if not top:
                continue
            total += 1
            index = int(top[0].index)
            pose = next(
                (p for p in result.poses if p.conformation == label and p.pose_index == index),
                None,
            )
            if pose is not None and pose.cluster == target.index:
                hits += 1
        if total:
            consensus_persistence = hits / total

    return RobustnessScore(
        ligand=result.ligand, cluster=int(target.index), n_conformations=len(labels),
        n_supporting=len(supporting),
        support=len(supporting) / max(1, len(labels)),
        persistence=len(persisting) / max(1, len(labels)),
        conformations_supporting=supporting, conformations_persisting=persisting,
        best_affinity=float(min(values)) if values else float("nan"),
        mean_affinity=float(np.mean(values)) if values else float("nan"),
        affinity_spread=float(max(values) - min(values)) if len(values) > 1 else 0.0,
        cross_spread=cross_spread,
        cross_spread_median=cross_median,
        worst_penalty=float(max(penalties)) if penalties else float("nan"),
        clashing=clashing,
        movement=float(movement),
        score=score, n_poses=len(result.poses),
        n_scored=0 if rescoring is None else rescoring.shape[0] * rescoring.shape[1],
        n_consensus=n_consensus, consensus_persistence=consensus_persistence,
        consensus_agreement=agreement,
    )


# ---------------------------------------------------------------------------
# The ensemble-aware screen
# ---------------------------------------------------------------------------

#: The per-ligand ensemble summary, written beside the screening results.
ENSEMBLE_JSONL = "ensemble.jsonl"
ENSEMBLE_CSV = "ensemble.csv"
#: The merged shortlist: the best pose of each top ligand, one ``MODEL`` each.
ENSEMBLE_TOP = "ensemble_top.pdbqt"
#: Where the aligned receptor PDBQTs are materialised inside the output directory.
ENSEMBLE_DIR = "ensemble"


@dataclass
class EnsembleLigandRow:
    """One molecule's ensemble verdict: best, per-conformation spread, robustness."""

    ligand: str
    name: str
    n_receptors: int = 0
    n_ok: int = 0
    best: Optional[float] = None
    best_receptor: str = ""
    mean: Optional[float] = None
    worst: Optional[float] = None
    spread: Optional[float] = None
    #: How many conformations produced a pose in the winning mode.
    n_supporting: int = 0
    persistence: float = float("nan")
    robustness: float = float("nan")
    cross_spread: float = float("nan")
    cross_spread_median: float = float("nan")
    worst_penalty: float = float("nan")
    cluster_poses: int = 0
    n_clusters: int = 0
    top_cluster_size: int = 0
    conformations_supporting: str = ""
    n_scored: int = 0
    pose_file: str = ""
    error: str = ""

    #: The CSV column order, kept explicit so a CSV is diffable.
    CSV_COLUMNS: Tuple[str, ...] = (
        "ligand", "name", "n_receptors", "n_ok", "best", "best_receptor", "mean",
        "worst", "spread", "n_supporting", "persistence", "robustness",
        "cross_spread", "cross_spread_median", "worst_penalty", "n_clusters",
        "top_cluster_size", "conformations_supporting", "n_scored", "pose_file", "error",
    )
    def as_dict(self) -> Dict[str, Any]:
        return {
            "ligand": self.ligand,
            "name": self.name,
            "n_receptors": int(self.n_receptors),
            "n_ok": int(self.n_ok),
            "best": self.best,
            "best_receptor": self.best_receptor,
            "mean": self.mean,
            "worst": self.worst,
            "spread": self.spread,
            "n_supporting": int(self.n_supporting),
            "persistence": self.persistence,
            "robustness": self.robustness,
            "cross_spread": self.cross_spread,
            "cross_spread_median": self.cross_spread_median,
            "worst_penalty": self.worst_penalty,
            "n_clusters": int(self.n_clusters),
            "top_cluster_size": int(self.top_cluster_size),
            "conformations_supporting": self.conformations_supporting,
            "n_scored": int(self.n_scored),
            "pose_file": self.pose_file,
            "error": self.error,
        }

    def row(self) -> List[Any]:
        """The CSV row, in :attr:`CSV_COLUMNS` order."""
        data = self.as_dict()
        return [_csv_cell(data[column]) for column in self.CSV_COLUMNS]


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


@dataclass
class EnsembleScreenSummary:
    """The ensemble layer of a campaign: one row per molecule, across receptors."""

    config: Any
    summary: Any
    ensemble: Ensemble
    ligands: List[EnsembleLigandRow] = field(default_factory=list)
    elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)
    json_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    top_path: Optional[Path] = None
    robustness_covered: int = 0

    @property
    def labels(self) -> List[str]:
        return list(self.ensemble.labels)

    def ranked(self, limit: Optional[int] = None) -> List[EnsembleLigandRow]:
        rows = [row for row in self.ligands if row.best is not None]
        rows.sort(key=lambda row: float(row.best))
        return rows if limit is None else rows[: int(limit)]

    def table(self, limit: int = 20) -> str:
        """One row per molecule: best affinity, spread, and whether it is robust."""
        rows = []
        for row in self.ranked(limit):
            rows.append([
                row.name[:28],
                f"{row.best:.3f}",
                row.best_receptor,
                str(row.n_ok),
                "--" if row.spread is None else f"{row.spread:.3f}",
                "--" if not math.isfinite(row.robustness) else f"{row.robustness:.3f}",
                "--" if not math.isfinite(row.cross_spread) else f"{row.cross_spread:.3f}",
                f"{row.top_cluster_size}/{row.cluster_poses}" if row.cluster_poses else "--",
            ])
        headers = [
            "name", "best", "best receptor", "confs", "spread", "robustness",
            "cross spread", "mode support",
        ]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def render(cells):
            return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * w for w in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def text(self, *, limit: int = 20) -> str:
        lines = [
            f"OpenDocking ensemble screen — {len(self.ligands)} molecule(s) x "
            f"{len(self.ensemble.labels)} conformation(s)",
        ]
        if self.summary is not None and getattr(self.summary, "results_path", None):
            lines.append(f"results: {self.summary.results_path}")
        lines.append(f"conformations: {', '.join(self.ensemble.labels)}")
        if self.json_path is not None:
            lines.append(f"ensemble results: {self.json_path}")
        lines += ["", self.table(limit)]
        if self.robustness_covered:
            lines.append("")
            lines.append(
                f"robustness is reported for the {self.robustness_covered} best "
                "molecule(s): it needs their poses, and each one costs "
                "poses x conformations exact rescoring(s)"
            )
        if self.warnings:
            lines.append("")
            lines.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(lines)

    def as_dict(self, *, full: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "n_ligands": len(self.ligands),
            "n_conformations": len(self.ensemble.labels),
            "conformations": list(self.ensemble.labels),
            "receptors": [None if p is None else str(p) for p in self.ensemble.paths],
            "elapsed": round(float(self.elapsed), 4),
            "robustness_covered": int(self.robustness_covered),
            "warnings": list(self.warnings),
            "alignment": None if self.ensemble.alignment is None else self.ensemble.alignment.as_dict(),
        }
        if full:
            data["ligands"] = [row.as_dict() for row in self.ligands]
        return data

    def write(self) -> Dict[str, Path]:
        """Write ``ensemble.jsonl`` (or CSV), the CSV twin, and the shortlist."""
        outdir = Path(self.config.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        import csv as _csv

        json_path = outdir / ENSEMBLE_JSONL
        json_path.write_text(
            "".join(json.dumps(row.as_dict(), default=str) + "\n" for row in self.ligands),
            encoding="utf-8",
        )
        csv_path = outdir / ENSEMBLE_CSV
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = _csv.writer(handle)
            writer.writerow(EnsembleLigandRow.CSV_COLUMNS)
            writer.writerows(row.row() for row in self.ligands)
        self.json_path = json_path
        self.csv_path = csv_path
        written = {"json": json_path, "csv": csv_path}
        top = _write_ensemble_shortlist(outdir, self)
        if top is not None:
            self.top_path = top
            written["top"] = top
        return written


def _write_ensemble_shortlist(
    outdir: Path, summary: EnsembleScreenSummary, limit: int = 0
) -> Optional[Path]:
    """The best pose of the best molecules, one model each, as ``ensemble_top.pdbqt``.

    The per-receptor shortlists :func:`odock.screen._write_shortlist` writes answer
    "what does this structure's search find".  This one answers the ensemble
    question: the pose that survived the *most* conformations goes first, so the
    file opens on the consensus mode rather than on the single best number.
    """
    from .consensus import pdbqt_models

    rows = summary.ranked(limit or None)
    if not rows:
        return None
    chunks: List[str] = []
    for rank, row in enumerate(rows, start=1):
        path = row.pose_file
        if not path:
            continue
        try:
            text = (outdir / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        models = pdbqt_models(text)
        if not models:
            continue
        block = models[0]
        lines = block.splitlines()
        if lines and lines[0].startswith("MODEL"):
            lines[0] = f"MODEL     {rank}"
            block = "\n".join(lines) + "\n"
        chunks.append(
            f"REMARK OPEN DOCKING ENSEMBLE rank {rank} ligand {row.name} "
            f"best {row.best:.3f} kcal/mol ({row.best_receptor}) "
            f"support {row.n_supporting}/{row.n_receptors} "
            f"robustness {_fmt_num(row.robustness)}\n" + block
        )
    if not chunks:
        return None
    target = outdir / ENSEMBLE_TOP
    target.write_text("".join(chunks), encoding="utf-8")
    return target


def _poses_from_document(
    text: str, *, conformation: str, conformation_index: int, name: str
) -> List[EnsemblePose]:
    """Read a pose file (multi-model PDBQT) into :class:`EnsemblePose` objects."""
    from .consensus import pdbqt_atoms, pdbqt_models

    poses: List[EnsemblePose] = []
    for index, block in enumerate(pdbqt_models(text)):
        atoms = pdbqt_atoms(block)
        if not atoms:
            continue
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
        affinity = float("nan")
        for line in block.splitlines():
            if line.startswith("REMARK") and "VINA RESULT" in line.upper():
                try:
                    affinity = float(line.split(":")[1].split()[0])
                except (IndexError, ValueError):
                    pass
                break
        poses.append(
            EnsemblePose(
                ligand=name, conformation=conformation,
                conformation_index=conformation_index, pose_index=index,
                affinity=affinity, coords=coords,
                elements=[atom.element for atom in atoms], local_rank=index + 1,
            )
        )
    return poses


def screen_ensemble(
    config: Any,
    ensemble: Optional[Ensemble] = None,
    *,
    cluster_rmsd: float = DEFAULT_CLUSTER_RMSD,
    robustness_top: int = 10,
    cross: bool = True,
    write: bool = True,
    progress: bool = True,
    **build_kwargs: Any,
) -> EnsembleScreenSummary:
    """Screen a library against every conformation, then aggregate per molecule.

    The campaign itself is :func:`odock.screen.screen_ligands`, unchanged: the
    same resumable ``results.jsonl``, the same one-row-per-(receptor, ligand)
    records, the same ``--top`` shortlist per receptor, the same library cache.
    This function only (a) points that campaign at the aligned receptor PDBQTs,
    and (b) adds the ensemble layer on top of the rows it produced -- best and
    worst over the conformations, the spread, and, for the best
    `robustness_top` molecules, the pose clustering and the cross-receptor
    rescoring that turn "is it robust?" into a number.

    Parameters
    ----------
    config
        An :class:`odock.screen.ScreenConfig` whose ``receptors`` are the
    `ensemble` or the files to build it from.
    ensemble
        A ready :class:`Ensemble`; built from ``config.receptors`` when omitted
        (``**build_kwargs`` goes to :func:`build_ensemble`).
    robustness_top
        How many of the best molecules get the expensive per-molecule analysis
        (pose clustering, cross-receptor rescoring).  ``0`` disables it.
    cross
        Run :func:`cross_rescore` for those molecules; it is what makes the
        robustness score a receptor-swap measurement instead of a within-set one.
    """
    from . import screen as screen_module

    started = time.perf_counter()
    warnings: List[str] = []
    if ensemble is None:
        receptors_outdir = Path(config.outdir) / ENSEMBLE_DIR
        build_kwargs.setdefault("superpose", False)
        ensemble = build_ensemble(
            config.receptors, box=config.box, outdir=receptors_outdir, **build_kwargs
        )
    warnings.extend(ensemble.warnings)

    config = _retarget(config, ensemble)
    summary = screen_module.screen_ligands(config)
    result = EnsembleScreenSummary(
        config=config, summary=summary, ensemble=ensemble, elapsed=0.0,
        warnings=list(warnings),
    )

    grouped: "OrderedDict[str, List[Any]]" = OrderedDict()
    for record in summary.records:
        grouped.setdefault(record.ligand, []).append(record)
    members = {member.key: member for member in getattr(summary, "members", [])}

    rows: List[EnsembleLigandRow] = []
    for key, records in grouped.items():
        ok = [
            record for record in records
            if record.status == "ok" and record.affinity is not None
        ]
        row = EnsembleLigandRow(
            ligand=key, name=records[0].name, n_receptors=len(records), n_ok=len(ok),
            error="" if ok else (records[0].error or records[0].status),
        )
        if ok:
            values = [float(record.affinity) for record in ok]
            best_record = min(ok, key=lambda record: float(record.affinity))
            row.best = min(values)
            row.best_receptor = best_record.receptor
            row.mean = float(np.mean(values))
            row.worst = max(values)
            row.spread = row.worst - row.best
            row.pose_file = best_record.pose_file
        rows.append(row)
    result.ligands = rows

    if robustness_top and summary.n_ok():
        candidates = [row for row in result.ranked(robustness_top) if row.pose_file]
        if not candidates:
            warnings.append(
                "no pose file was written (--no-poses), so the per-molecule "
                "clustering and robustness could not be measured"
            )
        else:
            _measure_rows(
                result, candidates, grouped, members, ensemble=ensemble, config=config,
                cluster_rmsd=cluster_rmsd, cross=cross, progress=progress,
            )
            result.robustness_covered = len(candidates)
    result.warnings = warnings
    result.elapsed = time.perf_counter() - started
    if write:
        written = result.write()
        if progress:
            print(f"wrote {written['json']}")
    return result


def _retarget(config: Any, ensemble: Ensemble) -> Any:
    """A copy of `config` pointing at the ensemble's receptor PDBQTs.

    When every conformation came straight from a file the caller gave us
    (:attr:`Ensemble.paths`), those files are kept as they are: ``screen`` hashes
    its receptor files to decide whether a directory is resumable, and copying
    them would change the signature and refuse a campaign that is perfectly
    resumable.
    """
    from dataclasses import replace

    paths: List[Any] = []
    for path, label in zip(ensemble.paths, ensemble.labels):
        paths.append(Path(path) if path is not None else None)
    if any(path is None for path in paths) or len(paths) != len(config.receptors):
        return config
    return replace(config, receptors=paths)


def _measure_rows(
    result: EnsembleScreenSummary,
    rows: Sequence[EnsembleLigandRow],
    grouped: Dict[str, List[Any]],
    members: Dict[str, Any],
    *,
    ensemble: Ensemble,
    config: Any,
    cluster_rmsd: float,
    cross: bool,
    progress: bool,
) -> None:
    """Cluster and rescore the poses of the shortlisted molecules, in place."""
    outdir = Path(config.outdir)
    for row in rows:
        records = [
            record for record in grouped.get(row.ligand, [])
            if record.status == "ok" and record.pose_file
        ]
        poses: List[EnsemblePose] = []
        for record in records:
            try:
                text = (outdir / record.pose_file).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                row.error = row.error or f"unreadable pose file: {exc}"
                continue
            index = ensemble.labels.index(record.receptor) if record.receptor in ensemble.labels else 0
            found = _poses_from_document(
                text, conformation=record.receptor, conformation_index=index,
                name=row.name,
            )
            # The record's own affinities are authoritative: they are what the
            # results file holds, and they survive a resumed campaign.
            for pose, entry in zip(found, record.poses or []):
                value = entry.get("affinity") if isinstance(entry, dict) else None
                if value is not None:
                    pose.affinity = float(value)
            poses.extend(pose for pose in found if math.isfinite(pose.affinity))
        if not poses:
            row.error = row.error or "no pose to analyse"
            continue
        clusters = cluster_ensemble_poses(poses, cutoff=cluster_rmsd)
        member = members.get(row.ligand)
        template = ""
        if member is not None and member.pdbqt_file:
            try:
                template = (outdir / member.pdbqt_file).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                template = ""
        analysis = EnsembleDockResult(
            ligand=row.name, labels=list(ensemble.labels), results=[], poses=poses,
            clusters=clusters, box=config.box, scoring=config.scoring,
            ligand_pdbqt=template,
        )
        rescoring = None
        if cross and template:
            try:
                rescoring = cross_rescore(analysis, ensemble, scoring=config.scoring)
            except EnsembleError as exc:
                result.warnings.append(f"{row.name}: cross-rescoring failed ({exc})")
        score = robustness_score(analysis, rescoring)
        row.n_clusters = len(clusters)
        row.cluster_poses = len(poses)
        row.top_cluster_size = clusters[0].size if clusters else 0
        row.n_supporting = score.n_supporting
        row.persistence = score.persistence
        row.robustness = score.score
        row.cross_spread = score.cross_spread
        row.cross_spread_median = score.cross_spread_median
        row.worst_penalty = score.worst_penalty
        row.conformations_supporting = ",".join(score.conformations_supporting)
        row.n_scored = score.n_scored
        if progress:
            print(
                f"  {row.name[:28]:<28} robustness {row.robustness:.3f} "
                f"(support {score.n_supporting}/{score.n_conformations}, "
                f"cross spread {_fmt_num(score.cross_spread)} kcal/mol, "
                f"{score.n_scored} rescore(s))"
            )


# ---------------------------------------------------------------------------
# The command line: `odock ensemble {align,dock,screen}`
# ---------------------------------------------------------------------------
#
# The whole subcommand lives here, not in :mod:`odock.cli`, which contains only
# one line calling :func:`add_ensemble_parser`.  That keeps the CLI file -- which
# several features are edited into at once -- out of this feature's blast radius,
# and it keeps this module testable without going through argparse.  The few
# helpers below are deliberately local copies of their ``cli`` counterparts
# (``_eprint``, ``_load_box``, ``_write_json``, ``_screen_config``) rather than
# imports, for the same reason: one module owning its own wiring.


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def _cli_write_json(path: Any, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    _cli_eprint(f"wrote {path}")


def _cli_lazy(module: str, feature: str):
    """Import an optional part of the package, or say why it is unavailable."""
    import importlib

    try:
        return importlib.import_module(module, __package__)
    except ImportError as exc:  # pragma: no cover - packaging failure
        _cli_eprint(f"odock: {feature} is unavailable: cannot import {module} ({exc}).")
        raise SystemExit(2)


def _ensemble_inputs(args) -> List[str]:
    """Flatten ``-r a.pdb b.pdb -r c.pdb`` into ``[a.pdb, b.pdb, c.pdb]``."""
    raw = getattr(args, "receptor", None) or []
    out: List[str] = []
    for entry in raw:
        if isinstance(entry, (list, tuple)):
            out.extend(str(item) for item in entry)
        else:
            out.append(str(entry))
    if not out:
        raise SystemExit("error: pass at least one -r/--receptor FILE")
    return out


def _cli_box(args) -> BoxSpec:
    """Resolve ``--box`` or ``--center/--size`` (the ``cli._load_box`` semantics)."""
    spacing = float(getattr(args, "spacing", None) or 0.375)
    if getattr(args, "box", None):
        data = json.loads(Path(args.box).read_text(encoding="utf-8"))
        if "center" not in data or "size" not in data:
            raise SystemExit(f"error: {args.box} is not a box file (no center/size)")
        if getattr(args, "center", None) or getattr(args, "size", None):
            _cli_eprint(
                f"note: --box {args.box} wins over --center/--size; the explicit "
                "coordinates are ignored"
            )
        return BoxSpec(
            center=tuple(float(x) for x in data["center"]),
            size=tuple(float(x) for x in data["size"]),
            spacing=float(data.get("spacing", spacing)),
        )
    if getattr(args, "center", None) and getattr(args, "size", None):
        return BoxSpec(
            center=tuple(float(x) for x in args.center),
            size=tuple(float(x) for x in args.size),
            spacing=spacing,
        )
    raise SystemExit(
        "error: no search box: pass --box FILE, or --center X Y Z --size X Y Z, or "
        "--box-ligand RESNAME (with --buffer)"
    )


def _ensemble_box(args, paths: Sequence[str]) -> BoxSpec:
    """The shared box of an ensemble, from --box, --center/--size or --box-ligand."""
    from .prepare import box_from_points

    reference = int(getattr(args, "reference", 0) or 0)
    if reference < 0 or reference >= len(paths):
        raise SystemExit(
            f"error: --reference {reference} does not exist (there are {len(paths)} "
            "input file(s))"
        )
    if getattr(args, "box_ligand", None) and not getattr(args, "box", None):
        conformations = read_conformations([paths[reference]], keep_water=True)
        coords = ligand_coords(conformations[0], args.box_ligand)
        box = box_from_points(
            coords, buffer=float(getattr(args, "buffer", 6.0)),
            spacing=float(getattr(args, "spacing", None) or 0.375),
        )
        _cli_eprint(
            f"box: {box} (from residue {args.box_ligand} of {paths[reference]}, "
            f"buffer {float(getattr(args, 'buffer', 6.0)):g} A)"
        )
        return box
    return _cli_box(args)


def _build_ensemble_from_args(args, box: BoxSpec) -> Ensemble:
    """Turn parsed ensemble arguments into a dockable :class:`Ensemble`."""
    paths = _ensemble_inputs(args)
    keep_water = bool(getattr(args, "keep_water", False))
    conformations = read_conformations(paths, keep_water=keep_water)
    site_ligand = None
    if getattr(args, "site_ligand", None):
        reference = int(getattr(args, "reference", 0) or 0)
        site_ligand = ligand_coords(conformations[reference], args.site_ligand)
    return build_ensemble(
        paths,
        box=box,
        outdir=getattr(args, "outdir", None),
        superpose=bool(getattr(args, "superpose", True)),
        reference=int(getattr(args, "reference", 0) or 0),
        site=getattr(args, "site", None),
        site_ligand_coords=site_ligand,
        site_radius=float(getattr(args, "site_radius", 8.0)),
        max_site_residues=int(getattr(args, "max_site_residues", 40)),
        atoms=str(getattr(args, "site_atoms", "ca")),
        strip=list(getattr(args, "strip", None) or []) or None,
        keep_water=keep_water,
        keep_hetero=not bool(getattr(args, "no_hetero", False)),
        min_identity=float(getattr(args, "min_identity", 0.90)),
        min_residues=int(getattr(args, "min_residues", 10)),
        min_site_residues=int(getattr(args, "min_site_residues", 3)),
        allow_box_mismatch=bool(getattr(args, "allow_box_mismatch", False)),
    )


def _screen_config_from_args(args, screen, box: BoxSpec):
    """Build a :class:`odock.screen.ScreenConfig` from the parsed arguments.

    A deliberate copy of ``cli._screen_config``: it reads only argument
    attributes, so the two screens -- the campaign and the ensemble campaign --
    share one flag set and one meaning for every flag, without this module
    importing the CLI.
    """
    return screen.ScreenConfig(
        receptors=list(_ensemble_inputs(args)),
        inputs=list(args.input),
        box=box,
        outdir=args.out,
        scoring=args.scoring,
        exhaustiveness=args.exhaustiveness,
        num_poses=args.num_poses,
        min_rmsd=args.min_rmsd,
        energy_range=args.energy_range,
        search=args.search,
        islands=args.islands,
        population=args.population,
        generations=args.generations,
        use_grid=not args.no_grid,
        refine=not args.no_refine,
        seed=args.seed,
        jobs=args.jobs,
        timeout=args.timeout,
        checkpoint_every=args.checkpoint_every,
        filters=not args.no_filter,
        optimize=not args.no_optimize,
        limit=args.limit,
        top=args.top,
        fmt="csv" if args.csv else "jsonl",
        resume=not args.no_resume,
        interactions=not args.no_interactions,
        write_poses=not args.no_poses,
        progress=not args.quiet,
        dry_run=args.dry_run,
        allow_box_mismatch=bool(getattr(args, "allow_box_mismatch", False)),
        consensus=args.consensus,
        consensus_top=args.consensus_top,
        consensus_method=args.consensus_method,
    )


def cmd_ensemble_align(args) -> int:
    """``odock ensemble align``: validate a receptor set and put it in one frame."""
    paths = _ensemble_inputs(args)
    box = None
    if getattr(args, "box", None) or (args.center and args.size) or getattr(args, "box_ligand", None):
        box = _ensemble_box(args, paths)
    try:
        conformations = read_conformations(paths, keep_water=bool(args.keep_water))
        site_ligand = None
        if args.site_ligand:
            site_ligand = ligand_coords(conformations[int(args.reference)], args.site_ligand)
        aligned = align_conformations(
            conformations,
            reference=int(args.reference),
            box=box,
            site=args.site,
            site_ligand=site_ligand,
            site_radius=float(args.site_radius),
            max_site_residues=int(args.max_site_residues),
            atoms=str(args.site_atoms),
            superpose=bool(args.superpose),
            min_identity=float(args.min_identity),
            min_residues=int(args.min_residues),
            min_site_residues=int(args.min_site_residues),
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble align: error: {exc}")
        return int(exc.code)

    if not args.quiet:
        print(aligned.text())
    if getattr(args, "outdir", None):
        target = Path(args.outdir)
        for path in aligned.write_pdb(target):
            _cli_eprint(f"wrote {path}")
        if getattr(args, "pdbqt", False):
            for index in range(len(aligned.conformations)):
                # Same rule as the docking path: a co-crystallised ligand inside
                # the box is not part of the receptor, so the PDBQT written here
                # is what an ensemble dock would actually use.  The aligned .pdb
                # above keeps every atom, ligand included, because seeing where
                # the ligand sits is half the point of inspecting an alignment.
                explicit = list(args.strip or [])
                names = (
                    _strip_names(aligned.conformations[index], box, explicit)
                    if box is not None else set(explicit)
                )
                prepared = aligned.receptor_pdbqt(
                    index, outdir=target, strip=sorted(names),
                    keep_water=bool(args.keep_water), keep_hetero=not args.no_hetero,
                )
                _cli_eprint(f"wrote {prepared}")
    if args.json_out:
        _cli_write_json(args.json_out, aligned.as_dict())
    return 0


def cmd_ensemble_dock(args) -> int:
    """``odock ensemble dock``: dock one ligand into every conformation and merge."""
    paths = _ensemble_inputs(args)
    box = _ensemble_box(args, paths)
    try:
        built = _build_ensemble_from_args(args, box)
        if not args.quiet:
            print(built.table())
            print()
        result = dock_ensemble(
            args.ligand, built, scoring=args.scoring,
            exhaustiveness=int(args.exhaustiveness), num_poses=int(args.num_poses),
            seed=int(args.seed), min_rmsd=float(args.min_rmsd),
            energy_range=float(args.energy_range), use_grid=not args.no_grid,
            refine=not args.no_refine, search=args.search, islands=int(args.islands),
            population=int(args.population), generations=int(args.generations),
            cluster_rmsd=float(args.cluster_rmsd),
            progress=None if args.quiet else _dock_progress,
        )
        rescoring = (
            None if args.no_cross
            else cross_rescore(result, built, scoring=args.scoring)
        )
        consensus = conformation_consensus(result, built) if args.consensus else None
        result.rescoring = rescoring
        result.consensus = {
            label: combined.as_dict() for label, combined in (consensus or {}).items()
        }
        result.robustness = robustness_score(result, rescoring, consensus=consensus)
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble dock: error: {exc}")
        return int(exc.code)

    if not args.quiet:
        print(result.text())
    if args.out:
        Path(args.out).write_text(result.to_pdbqt(), encoding="utf-8")
        _cli_eprint(f"wrote {args.out} ({len(result.poses)} pose(s))")
    if args.json_out:
        _cli_write_json(args.json_out, result.as_dict())
    if result.best() is None:
        _cli_eprint("odock ensemble dock: error: no conformation produced a pose")
        return 3
    return 0


def _dock_progress(label: str, index: int, total: int, docked: Any) -> None:
    best = docked.best_affinity
    _cli_eprint(
        f"  docked {label} ({index}/{total}): "
        + ("--" if best is None else f"{best:.3f}") + " kcal/mol"
    )


def cmd_ensemble_screen(args) -> int:
    """``odock ensemble screen``: the screening campaign, with more than one receptor."""
    screen = _cli_lazy("odock.screen", "library screening")
    paths = _ensemble_inputs(args)
    if getattr(args, "box", None) or (args.center and args.size):
        box = _cli_box(args)
    else:
        box = _ensemble_box(args, paths)
    _cli_eprint(f"box: {box}")
    try:
        built = _build_ensemble_from_args(args, box)
        config = _screen_config_from_args(args, screen, box)
        summary = screen_ensemble(
            config, built, cluster_rmsd=float(args.cluster_rmsd),
            robustness_top=int(args.robustness_top), cross=not args.no_cross,
            progress=not args.quiet,
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble screen: error: {exc}")
        return int(exc.code)
    except screen.ScreenError as exc:
        _cli_eprint(f"odock ensemble screen: error: {exc}")
        return int(exc.code)

    if args.json_out:
        _cli_write_json(args.json_out, summary.as_dict(full=False))
    if summary.summary is not None and getattr(summary.summary, "dry_run", False):
        if not args.quiet:
            print(summary.summary.text())
        return 0
    n_ok = 0 if summary.summary is None else summary.summary.n_ok()
    line = (
        f"{n_ok} docking(s) succeeded over {len(built.labels)} conformation(s), "
        f"{len(summary.ligands)} molecule(s) ranked ({summary.elapsed:.1f} s this run)"
    )
    if not args.quiet:
        print(summary.text())
        print()
        print(line)
    else:
        _cli_eprint(f"odock ensemble screen: {line}")
    if summary.summary is not None and summary.summary.interrupted:
        return 130
    if n_ok == 0:
        _cli_eprint("odock ensemble screen: error: no molecule docked successfully")
        return 3
    return 0


def _add_ensemble_arguments(
    parser: argparse.ArgumentParser, *, allow_box_mismatch: bool = True
) -> None:
    """The flags that describe how a set of structures becomes one ensemble."""
    parser.add_argument(
        "-r", "--receptor", action="append", nargs="+", required=True, metavar="FILE",
        help="receptor PDB, PDBQT or multi-model PDB; every file, and every MODEL of a "
             "multi-model file, is one conformation of the ensemble (pass several "
             "files, or repeat the flag)",
    )
    parser.add_argument(
        "--superpose", action=argparse.BooleanOptionalAction, default=True,
        help="fit every conformation onto the reference on its binding-site residues, "
             "so one box describes the same site in every frame (the default); "
             "--no-superpose only validates and reports the set, for files that are "
             "already in one frame",
    )
    parser.add_argument(
        "--reference", type=int, default=0, metavar="N",
        help="index of the conformation everything is superposed onto (0 = the first)",
    )
    parser.add_argument(
        "--site", metavar="RES[,RES...]",
        help="the binding-site residues that define the frame, e.g. "
             "--site ASP189,SER190,A:195; the default is the residues around the "
             "reference ligand or the box",
    )
    parser.add_argument(
        "--site-ligand", metavar="RESNAME",
        help="use the residues around this residue of the reference structure as the "
             "site, e.g. --site-ligand OHT for the co-crystallised ligand",
    )
    parser.add_argument(
        "--site-radius", type=float, default=8.0,
        help="how far around the reference ligand or the box centre a residue still "
             "counts as part of the binding site (Å)",
    )
    parser.add_argument(
        "--site-atoms", choices=["ca", "backbone", "all"], default="ca",
        help="which atoms of the site residues the superposition uses",
    )
    parser.add_argument(
        "--max-site-residues", type=int, default=40,
        help="keep at most this many site residues (the closest to the site centre): a "
             "fit on 200 residues is a global fit, not a site fit",
    )
    parser.add_argument(
        "--strip", nargs="+", metavar="RESNAME",
        help="residue names to delete from every conformation before docking; a "
             "co-crystallised ligand inside the search box is stripped automatically",
    )
    parser.add_argument("--keep-water", action="store_true", help="keep crystallographic waters")
    parser.add_argument("--no-hetero", action="store_true", help="drop every non-standard residue")
    parser.add_argument(
        "--min-identity", type=float, default=0.90,
        help="refuse a set whose sequence identity to the reference is below this",
    )
    parser.add_argument(
        "--min-residues", type=int, default=10,
        help="refuse a conformation that shares fewer residues than this with the reference",
    )
    parser.add_argument(
        "--min-site-residues", type=int, default=3,
        help="refuse a set whose binding site holds fewer residues than this",
    )
    if allow_box_mismatch:
        parser.add_argument(
            "--allow-box-mismatch", action="store_true",
            help="dock even when the box touches no atom of a conformation",
        )


def _add_ensemble_screen_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags ``odock ensemble screen`` shares with ``odock screen``.

    One flag set, one meaning: :func:`_screen_config_from_args` builds the very
    same :class:`odock.screen.ScreenConfig` these flags describe for the campaign.
    """
    parser.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    parser.add_argument("-o", "--out", required=True, help="output directory")
    parser.add_argument("--box", help="box JSON written by `odock box`")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--box-ligand", metavar="RESNAME",
        help="derive the box from this residue of the reference structure (its "
             "co-crystallised ligand), padded by --buffer",
    )
    parser.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    parser.add_argument(
        "--spacing", type=float, default=None,
        help="grid spacing in Å (default: the spacing stored in --box, else 0.375)",
    )
    parser.add_argument(
        "-s", "--scoring", default="vina", choices=["vina", "vinardo", "ad4"],
        help="force field",
    )
    parser.add_argument("-e", "--exhaustiveness", type=int, default=8, help="search effort")
    parser.add_argument("-n", "--num-poses", type=int, default=9, help="poses kept per molecule")
    parser.add_argument("--min-rmsd", type=float, default=1.0, help="pose deduplication cutoff (Å)")
    parser.add_argument("--energy-range", type=float, default=3.0, help="reporting window (kcal/mol)")
    parser.add_argument("--no-grid", action="store_true", help="use the exact scorer in the search")
    parser.add_argument("--no-refine", action="store_true", help="skip the exact refinement")
    parser.add_argument(
        "--search", choices=["monte_carlo", "lga", "lga_solis"], default=None,
        help="search protocol (default: monte_carlo)",
    )
    parser.add_argument("--islands", type=int, default=4, help="GA islands")
    parser.add_argument("--population", type=int, default=32, help="GA population per island")
    parser.add_argument("--generations", type=int, default=20, help="GA generations")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="seed for the whole campaign (0 draws one and prints it); the same seed is "
             "used for every conformation, so the comparison is paired",
    )
    parser.add_argument("--jobs", type=int, default=0, help="molecules docked concurrently")
    parser.add_argument("--timeout", type=float, default=None, help="per-molecule wall-clock limit (s)")
    parser.add_argument("--checkpoint-every", type=int, default=20, help="durable checkpoint interval")
    parser.add_argument("--no-filter", action="store_true", help="dock even the non-drug-like molecules")
    parser.add_argument("--no-optimize", action="store_true", help="skip the ligand pre-optimisation")
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="dock only the first N molecules")
    parser.add_argument(
        "--top", type=int, default=0, metavar="N",
        help="write the N best molecules as a multi-model PDBQT shortlist per "
             "conformation, plus the merged ensemble_top.pdbqt",
    )
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--csv", action="store_true", help="write the results as CSV")
    fmt.add_argument("--jsonl", action="store_true", help="write the results as JSONL (default)")
    parser.add_argument("--no-resume", action="store_true", help="discard the results in --out and start over")
    parser.add_argument("--no-interactions", action="store_true", help="skip the per-pose interaction profiling")
    parser.add_argument("--no-poses", action="store_true", help="do not write per-molecule pose files")
    parser.add_argument("--dry-run", action="store_true", help="report the filtered library and the cost")
    parser.add_argument(
        "--consensus", action="store_true",
        help="rescore each conformation's shortlist with vina, vinardo and ad4 (the "
             "within-conformation half of the consensus)",
    )
    parser.add_argument("--consensus-top", type=int, default=0, metavar="N")
    parser.add_argument("--consensus-method", choices=["rank", "borda", "z"], default="rank")
    parser.add_argument(
        "--cluster-rmsd", type=float, default=DEFAULT_CLUSTER_RMSD,
        help="RMSD at which two poses from two conformations are the same binding mode (Å)",
    )
    parser.add_argument(
        "--robustness-top", type=int, default=10, metavar="N",
        help="how many of the best molecules get the per-molecule robustness analysis "
             "(0 disables it: it needs their poses and costs poses x conformations "
             "exact rescorings each)",
    )
    parser.add_argument(
        "--no-cross", action="store_true",
        help="skip the cross-receptor rescoring: the robustness score then uses the "
             "affinity spread inside the ensemble instead of the receptor-swap spread",
    )
    parser.add_argument(
        "--allow-box-mismatch", action="store_true",
        help="dock even when the box touches no atom of a conformation",
    )
    parser.add_argument("--json-out", help="write the ensemble summary as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no progress line and no table")


def add_ensemble_parser(sub) -> None:
    """Register ``odock ensemble`` (and its three subcommands) on `sub`.

    Called from :func:`odock.cli.build_parser` with the CLI's subparsers object,
    which is the only thing this module needs from the command line.
    """
    en = sub.add_parser(
        "ensemble",
        help="dock against several receptor conformations, not one rigid structure",
        description=(
            "A receptor is not a statue. These subcommands validate that a set of "
            "structures is one receptor, superpose them on their binding site so a "
            "single box means the same thing in every frame, dock (or screen) against "
            "all of them, and combine the results across conformations."
        ),
    )
    ensub = en.add_subparsers(dest="kind", required=True)

    ea = ensub.add_parser(
        "align",
        help="validate a receptor set, superpose it, and report the alignment",
        description=(
            "Check that the structures are the same receptor (sequence identity and "
            "overlap), fit each one onto the reference on its binding-site residues, "
            "and report the site RMSD, the whole-protein CA RMSD and the per-residue "
            "binding-site displacement."
        ),
    )
    _add_ensemble_arguments(ea, allow_box_mismatch=False)
    ea.add_argument("--box", help="box JSON: also defines the site as the residues around it")
    ea.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ea.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ea.add_argument("--box-ligand", metavar="RESNAME", help="derive the site from this residue's position")
    ea.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    ea.add_argument("--spacing", type=float, default=0.375, help="grid spacing for --box-ligand (Å)")
    ea.add_argument("--outdir", help="write the aligned conformations there (.pdb, and .pdbqt with --pdbqt)")
    ea.add_argument("--pdbqt", action="store_true", help="also prepare each aligned conformation as a receptor PDBQT")
    ea.add_argument("--json-out", help="write the alignment report as JSON")
    ea.add_argument("-q", "--quiet", action="store_true", help="do not print the report")
    ea.set_defaults(func=cmd_ensemble_align)

    ed = ensub.add_parser(
        "dock",
        help="dock one ligand against every conformation and merge the poses",
        description=(
            "Dock the same ligand into every conformation with the same box and the "
            "same seed, pool every pose into one ranking, cluster the poses across "
            "conformations, and -- unless --no-cross is given -- rescore every pose in "
            "every conformation so that the robustness of a binding mode is a number "
            "in kcal/mol."
        ),
    )
    _add_ensemble_arguments(ed)
    ed.add_argument("-l", "--ligand", required=True, help="ligand PDBQT, or any file `odock prepare ligand` reads")
    ed.add_argument("-o", "--out", help="write the merged poses as one multi-model PDBQT")
    ed.add_argument("--json-out", help="write the whole result as JSON")
    ed.add_argument("--box", help="box JSON written by `odock box`")
    ed.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ed.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    ed.add_argument("--box-ligand", metavar="RESNAME", help="derive the box from this residue of the reference structure")
    ed.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    ed.add_argument("--spacing", type=float, default=0.375, help="grid spacing in Å")
    ed.add_argument("-s", "--scoring", default="vina", choices=["vina", "vinardo", "ad4"], help="force field")
    ed.add_argument("-e", "--exhaustiveness", type=int, default=8)
    ed.add_argument("-n", "--num-poses", type=int, default=9)
    ed.add_argument("--seed", type=int, default=0, help="0 draws a random seed; the same seed is used for every conformation")
    ed.add_argument("--min-rmsd", type=float, default=1.0)
    ed.add_argument("--energy-range", type=float, default=3.0)
    ed.add_argument("--no-grid", action="store_true", help="use the exact scorer in the search too")
    ed.add_argument("--no-refine", action="store_true", help="skip exact post-refinement")
    ed.add_argument("--search", choices=["monte_carlo", "lga", "lga_solis"], default=None)
    ed.add_argument("--islands", type=int, default=4)
    ed.add_argument("--population", type=int, default=32)
    ed.add_argument("--generations", type=int, default=20)
    ed.add_argument(
        "--cluster-rmsd", type=float, default=DEFAULT_CLUSTER_RMSD,
        help="RMSD at which two poses from two conformations are the same mode (Å)",
    )
    ed.add_argument(
        "--no-cross", action="store_true",
        help="skip scoring every pose in every conformation (faster, no receptor-swap spread)",
    )
    ed.add_argument(
        "--consensus", action="store_true",
        help="also rescore each conformation's poses with vina, vinardo and ad4",
    )
    ed.add_argument("-q", "--quiet", action="store_true", help="do not print the report")
    ed.set_defaults(func=cmd_ensemble_dock)

    es = ensub.add_parser(
        "screen",
        help="screen a library against every conformation (resumable)",
        description=(
            "The `odock screen` campaign with more than one receptor: the same "
            "resumable results file with one row per (conformation, ligand), the same "
            "library cache and per-conformation shortlist, plus ensemble.jsonl (best, "
            "worst, spread and robustness per molecule) and the merged "
            "ensemble_top.pdbqt shortlist."
        ),
    )
    _add_ensemble_arguments(es, allow_box_mismatch=False)
    _add_ensemble_screen_arguments(es)
    es.set_defaults(func=cmd_ensemble_screen)

    # `pockets` lives in odock/pockets.py; it needs the same receptor-set flags,
    # which is why it is registered from here rather than from cli_ext.
    from .pockets import add_pockets_subparser

    add_pockets_subparser(ensub)

    # Same for `generate`: it belongs to this group and needs none of the
    # receptor-set flags, but it is the answer to "I only have one structure".
    from .generate import add_generate_subparser

    add_generate_subparser(ensub)

    # ... and `modes`, which supplies the motion side-chain sampling provably
    # cannot: the backbone.
    from .modes import add_modes_subparser

    add_modes_subparser(ensub)

    # ... and `waters`, which is about what every other path throws away.
    from .waters import add_waters_subparser

    add_waters_subparser(ensub)

    # ... and `coupling`, the question the mode set *can* answer even though it
    # cannot predict a conformational change.
    from .coupling import add_coupling_subparser

    add_coupling_subparser(ensub)






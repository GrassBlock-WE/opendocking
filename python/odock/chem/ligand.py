# SPDX-License-Identifier: GPL-3.0-or-later
"""Ligand chemistry: multi-format input, 3-D embedding, force-field minimisation,
rotatable-bond perception and the kinematic torsion tree.

This module implements the ligand-chemistry requirements on top of RDKit only:

* :func:`read_ligands` — batch-aware input for ``.smi``, ``.sdf``, ``.mol2``,
  ``.mol``, ``.pdb``, ``.pdbqt`` and bare SMILES strings.  Every molecule comes
  back three-dimensional: input that only carries a 2-D depiction is embedded
  with ETKDGv3, exactly as :mod:`odock.prepare` does.
* :func:`embed_3d` — the ETKDGv3 embedder on its own.
* :func:`minimize` — MMFF94 / MMFF94s / UFF gradient minimisation with a
  documented fallback chain, reporting the energy before and after on the
  molecule itself.
* :func:`rotatable_bonds` / :func:`rotation_reason` — torsion perception with
  every exclusion the project requires, each with a machine-readable reason so
  the GUI can explain *why* a bond is rigid.
* :func:`torsion_tree` — the kinematic ``ROOT -> BRANCH`` tree the AutoDock
  PDBQT dialect is built from.

Nothing in here writes PDBQT; that stays in :mod:`odock.pdbqt`.
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - RDKit is a hard dependency for chemistry
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from ..prepare import pdbqt_to_pdb_block

__all__ = [
    "FORCE_FIELDS",
    "MIN_STEPS",
    "MAX_STEPS",
    "DEFAULT_SEED",
    "ROTATABLE",
    "RIGID_HYDROGEN",
    "RIGID_BOND_ORDER",
    "RIGID_AROMATIC",
    "RIGID_RING",
    "RIGID_AMIDE",
    "RIGID_TERMINAL",
    "RIGID_SYMMETRIC",
    "RIGID_LOCKED",
    "RIGID_REASONS",
    "REASON_NOTES",
    "TorsionTree",
    "embed_3d",
    "minimize",
    "read_ligands",
    "rotation_reason",
    "rotatable_bonds",
    "torsion_tree",
]

#: The force fields :func:`minimize` can drive, in fallback order.
FORCE_FIELDS: Tuple[str, ...] = ("MMFF94", "MMFF94s", "UFF")

#: The project requirements ask for 200-1000 gradient steps; anything outside is
#: clamped.
MIN_STEPS = 200
MAX_STEPS = 1000

#: The default ETKDGv3 seed.  Fixed so that a repeated run is reproducible.
DEFAULT_SEED = 20240101


# ---------------------------------------------------------------------------
# RDKit import guard
# ---------------------------------------------------------------------------


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for ligand chemistry. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


@contextlib.contextmanager
def _quiet_rdkit() -> Iterator[None]:
    """Silence RDKit's C++ log while a force field is probed.

    A molecule that a force field cannot type makes RDKit print things such as
    ``UFFTYPER: Unrecognized atom type`` straight to stderr.  That is useful
    during development and noise in a virtual-screening run of 10^5 molecules,
    so the reason is recorded on the molecule instead (``odock_*`` properties)
    and the raw message is suppressed here.
    """
    block = getattr(rdBase, "BlockLogs", None)
    if block is None:
        yield
        return
    with block():  # type: ignore[union-attr]
        yield


# ---------------------------------------------------------------------------
# Small geometry / graph helpers
# ---------------------------------------------------------------------------


def _has_3d_conformer(mol) -> bool:
    """Whether ``mol`` carries usable three-dimensional coordinates.

    RDKit's file readers set the conformer's ``Is3D`` flag, which is the only
    reliable test: benzene is legitimately flat, so a geometric planarity test
    would wrongly re-embed a perfectly good crystal structure.  Same rule as
    :mod:`odock.prepare`.
    """
    if mol.GetNumConformers() == 0:
        return False
    try:
        return bool(mol.GetConformer().Is3D())
    except Exception:  # pragma: no cover - very old RDKit
        return True


def _has_explicit_hydrogens(mol) -> bool:
    return any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())


def _is_flat_depiction(mol) -> bool:
    """Whether the coordinates are a 2-D sketch rather than a 3-D structure.

    RDKit's ``Is3D`` flag is set by the file readers, but the MOL2 reader has no
    dimensionality field to read and always claims 3-D.  A drawing (or a vendor
    MOL2 written with ``z = 0.0000``) is recognisable exactly: one coordinate
    axis has *zero* extent while the other two do not.  A linear molecule also
    has zero extent on two axes and is deliberately not matched — it is 1-D, not
    a flat depiction of a 3-D molecule, and re-embedding it would achieve
    nothing.  A genuinely planar 3-D molecule is caught by this test when it
    happens to be written axis-aligned, which is harmless: re-embedding a planar
    molecule produces a planar molecule.
    """
    if mol.GetNumConformers() == 0 or mol.GetNumAtoms() < 3:
        return False
    coords = _positions(mol)
    spread = coords.max(axis=0) - coords.min(axis=0)
    return int(np.count_nonzero(spread == 0.0)) == 1


def _needs_embedding(mol) -> bool:
    """Whether :func:`embed_3d` should generate a fresh 3-D conformer."""
    return not _has_3d_conformer(mol) or _is_flat_depiction(mol)


def _heavy_degree(atom) -> int:
    """Number of heavy (non-hydrogen) neighbours."""
    return sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)


def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def _positions(mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


def _set_positions(mol, coords: np.ndarray) -> None:
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(
            i, (float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
        )


# ---------------------------------------------------------------------------
# 3-D embedding
# ---------------------------------------------------------------------------


def embed_3d(mol, *, seed: int = DEFAULT_SEED, force: bool = False):
    """Return a copy of ``mol`` carrying an ETKDGv3 three-dimensional conformer.

    Parameters
    ----------
    mol
        Any RDKit molecule.  Hydrogens are added first when the molecule carries
        none *and* its atoms still have implicit valences to give: a MOL2 record
        marks every atom ``NoImplicit``, so its hydrogens are never invented
        here (adding polar hydrogens is :mod:`odock.prepare`'s job).
    seed
        ETKDGv3 random seed; fixed by default so a run is reproducible.
    force
        Embed even when the molecule already has a 3-D conformer.  Without it an
        existing 3-D geometry (a crystal pose, say) is kept untouched — the same
        rule :func:`odock.prepare.prepare_ligand` uses, which is what stops a
        reference ligand from being silently reshaped.  :func:`read_ligands` and
        :func:`minimize` pass ``force=True`` for a molecule that is only a flat
        sketch (see :func:`_is_flat_depiction`), which the ``Is3D`` flag alone
        cannot detect in a MOL2 file.

    The first embedding attempt uses the plain ETKDGv3 distance-geometry
    engine; when that fails (fused rings, macrocycles, exotic valences) the
    attempt is repeated with ``useRandomCoords``, which is slower but far more
    forgiving.  A molecule that defeats both raises :class:`RuntimeError`.
    """
    Chem_, AllChem_ = _require_rdkit()
    work = Chem_.Mol(mol)
    if not force and _has_3d_conformer(work):
        return work
    if work.GetNumAtoms() == 0:
        raise RuntimeError("cannot embed a molecule with no atoms")
    if not _has_explicit_hydrogens(work):
        work = Chem_.AddHs(work)

    params = AllChem_.ETKDGv3()
    params.randomSeed = int(seed)
    status = None
    error: Optional[BaseException] = None
    for use_random in (False, True):
        params.useRandomCoords = use_random
        try:
            status = AllChem_.EmbedMolecule(work, params)
        except Exception as exc:  # pragma: no cover - RDKit raises rarely
            error = exc
            status = 1
        if status == 0:
            work.SetProp("odock_embed_method", "ETKDGv3" if not use_random else "ETKDGv3+random")
            work.SetProp("odock_embed_seed", str(int(seed)))
            return work
    raise RuntimeError(
        "RDKit failed to generate a 3-D conformer with ETKDGv3 "
        f"(seed {int(seed)}, status {status}{', ' + str(error) if error else ''})"
    )


def _require_rdkit():
    require_rdkit()
    return Chem, AllChem


# ---------------------------------------------------------------------------
# Multi-format, batch-aware reading
# ---------------------------------------------------------------------------

#: Extension (without the dot) -> reader kind.
_FORMATS = {
    "smi": "smi",
    "smiles": "smi",
    "ism": "smi",
    "sdf": "sdf",
    "sd": "sdf",
    "mol": "mol",
    "mol2": "mol2",
    "pdb": "pdb",
    "ent": "pdb",
    "pdbqt": "pdbqt",
}

_TEXT_FORMATS = ("smi",)


def _normalise_format(fmt: Optional[str]) -> Optional[str]:
    if fmt is None:
        return None
    key = str(fmt).strip().lower().lstrip(".")
    if key not in _FORMATS:
        raise ValueError(
            f"unsupported ligand format {fmt!r}; supported: {sorted(set(_FORMATS))}"
        )
    return _FORMATS[key]


def _split_records(text: str, marker: str) -> List[str]:
    """Split a structure file on a line-oriented record marker."""
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip() == marker:
            if current:
                blocks.append("\n".join(current) + "\n")
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current) + "\n")
    return [b for b in blocks if b.strip()]


def _split_mol2_records(text: str) -> List[str]:
    """Split a MOL2 document into records, keeping each ``@<TRIPOS>MOLECULE``.

    Unlike :func:`_split_records` this preserves the text verbatim (including
    the final newline): RDKit's MOL2 reader is strict about the record shape and
    returns ``None`` for a block that lost its terminating line break.
    """
    lines = text.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "@<TRIPOS>MOLECULE"
    ]
    if not starts:
        return []
    bounds = starts + [len(lines)]
    return [
        "".join(lines[bounds[i] : bounds[i + 1]]) for i in range(len(starts))
    ]


def _split_models(text: str) -> List[str]:
    """Split a PDB/PDBQT document into its ``MODEL`` records.

    A single-model document (no ``MODEL`` record at all) is returned as one
    block, so the caller can treat both shapes identically.
    """
    if "\nMODEL" not in "\n" + text or "ENDMDL" not in text:
        return [text]
    blocks: List[str] = []
    current: Optional[List[str]] = None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                blocks.append("\n".join(current))
            current = None
            continue
        if current is not None:
            current.append(line)
    return [b for b in blocks if b.strip()] or [text]


def _sanitize_best_effort(mol):
    """Sanitise in place, ignoring failure; return the molecule."""
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def _mol_from_text_block(kind: str, block: str):
    """Parse one record, degrading from sanitised to unsanitised chemistry."""
    if kind == "sdf" or kind == "mol":
        mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
            return _sanitize_best_effort(mol)
        return mol
    if kind == "mol2":
        mol = Chem.MolFromMol2Block(block, removeHs=False, sanitize=True)
        if mol is None:
            mol = Chem.MolFromMol2Block(block, removeHs=False, sanitize=False)
            return _sanitize_best_effort(mol)
        return mol
    if kind == "pdb":
        mol = Chem.MolFromPDBBlock(
            block, removeHs=False, sanitize=True, proximityBonding=True
        )
        if mol is None:
            mol = Chem.MolFromPDBBlock(
                block, removeHs=False, sanitize=False, proximityBonding=True
            )
            return _sanitize_best_effort(mol)
        return mol
    if kind == "pdbqt":
        # A PDBQT carries AutoDock atom types in the element columns; the
        # shared converter rewrites them into an RDKit-readable PDB block.
        mol = Chem.MolFromPDBBlock(
            pdbqt_to_pdb_block(block),
            removeHs=False,
            sanitize=False,
            proximityBonding=True,
        )
        return _sanitize_best_effort(mol)
    raise ValueError(f"no block parser for {kind!r}")


def _read_file(path: Path, kind: str) -> Tuple[List[object], List[str]]:
    """Read every record of ``path``; returns ``(molecules, notes)``."""
    mols: List[object] = []
    notes: List[str] = []

    if kind == "smi":
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            mol = Chem.MolFromSmiles(fields[0])
            if mol is None:
                notes.append(f"line {lineno}: {fields[0]!r} is not a parsable SMILES")
                continue
            if len(fields) > 1:
                mol.SetProp("_Name", " ".join(fields[1:]))
            mols.append(mol)
        return mols, notes

    if kind == "sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        try:
            count = len(supplier)
        except Exception as exc:  # pragma: no cover - defensive
            count = 0
            notes.append(f"could not count the SDF records: {exc}")
        blocks: Optional[List[str]] = None
        for index in range(count):
            try:
                mol = supplier[index]
            except Exception as exc:
                mol = None
                notes.append(f"record {index + 1}: {exc}")
            if mol is None:
                # Fall back to the raw record so a molecule that only fails
                # sanitisation is still delivered.
                if blocks is None:
                    blocks = _split_records(
                        path.read_text(encoding="utf-8", errors="replace"), "$$$$"
                    )
                if index < len(blocks):
                    mol = _mol_from_text_block("sdf", blocks[index])
                    if mol is not None:
                        notes.append(
                            f"record {index + 1} was read with best-effort sanitisation"
                        )
            if mol is None:
                notes.append(f"record {index + 1}: RDKit could not parse it")
                continue
            mols.append(mol)
        return mols, notes

    if kind == "mol":
        mol = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
        if mol is None:
            text = path.read_text(encoding="utf-8", errors="replace")
            mol = _mol_from_text_block("mol", text)
            if mol is not None:
                notes.append("the molecule was read with best-effort sanitisation")
        if mol is not None:
            mols.append(mol)
        return mols, notes

    if kind == "mol2":
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = _split_mol2_records(text)
        if not blocks:
            mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
            if mol is None:
                mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
                mol = _sanitize_best_effort(mol)
                if mol is not None:
                    notes.append("the molecule was read with best-effort sanitisation")
            if mol is not None:
                mols.append(mol)
            return mols, notes
        for index, block in enumerate(blocks):
            mol = _mol_from_text_block("mol2", block)
            if mol is None:
                notes.append(f"record {index + 1}: RDKit could not parse it")
                continue
            mols.append(mol)
        return mols, notes

    if kind in ("pdb", "pdbqt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = _split_models(text)
        if kind == "pdbqt":
            notes.append(
                "a PDBQT carries no bond orders; aromaticity and bond orders are "
                "best-effort unless a SMILES template is applied"
            )
        for index, block in enumerate(blocks):
            mol = _mol_from_text_block(kind, block)
            if mol is None:
                notes.append(f"record {index + 1}: RDKit could not parse it")
                continue
            mols.append(mol)
        return mols, notes

    raise ValueError(f"no reader for {kind!r}")  # pragma: no cover - guarded


def _read_smiles_text(text: str) -> Tuple[List[object], List[str]]:
    mols: List[object] = []
    notes: List[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        mol = Chem.MolFromSmiles(fields[0])
        if mol is None:
            notes.append(f"line {lineno}: {fields[0]!r} is not a parsable SMILES")
            continue
        if len(fields) > 1:
            mol.SetProp("_Name", " ".join(fields[1:]))
        mols.append(mol)
    return mols, notes


def _is_problem_note(note: str) -> bool:
    """Whether a reader note means a record was dropped or degraded."""
    markers = (
        "not a parsable",
        "could not parse",
        "could not be read",
        "could not count",
        "best-effort sanitisation",
        "is not a parsable SMILES",
    )
    return any(marker in note for marker in markers)


def read_ligands(
    source,
    *,
    fmt: Optional[str] = None,
    name: str = "ligand",
    embed: bool = True,
    seed: int = DEFAULT_SEED,
) -> List[object]:
    """Read one or many ligands from ``source`` and return RDKit molecules.

    Parameters
    ----------
    source
        One of

        * a path to ``.smi`` / ``.sdf`` / ``.mol2`` / ``.mol`` / ``.pdb`` /
          ``.pdbqt``;
        * a bare SMILES string, or multi-line ``"SMILES name"`` text (what a
          ``.smi`` file contains);
        * an RDKit molecule, or any iterable of the above (mixed freely).

    fmt
        Override the format instead of trusting the extension.  The ``.smi``
        format is treated as text, so ``read_ligands("CCO", fmt="smi")`` works.
    name
        Base name for molecules whose file carries no title.
    embed
        Embed input that only has 2-D coordinates with ETKDGv3 (default).  This
        covers both a file that declares itself 2-D and a flat sketch that
        claims to be 3-D (a vendor MOL2 drawn with ``z = 0``).  Pass ``False`` to
        keep the input geometry untouched.
    seed
        ETKDGv3 seed.

    Returns
    -------
    Every record of a batch file, in file order.  A record that RDKit cannot
    parse is skipped with a :class:`UserWarning` naming it; if *no* record of a
    non-empty source could be read, :class:`ValueError` is raised instead.

    Notes
    -----
    Contrary to the strict interface listing this function also accepts the
    ``embed`` and ``seed`` keywords; both are optional and default to the
    documented behaviour, so existing calls are unaffected.
    """
    require_rdkit()
    kind_override = _normalise_format(fmt)
    mols, notes = _collect(source, kind_override)

    if not mols:
        if notes:
            raise ValueError("no ligand could be read: " + "; ".join(notes[:8]))
        raise ValueError(f"no ligand found in {source!r}")

    if len(notes) > 0 and any(_is_problem_note(n) for n in notes):
        problems = [n for n in notes if _is_problem_note(n)]
        warnings.warn(
            f"{len(problems)} problem(s) while reading {source!r}: "
            + "; ".join(problems[:8]),
            UserWarning,
            stacklevel=2,
        )

    out: List[object] = []
    for index, mol in enumerate(mols):
        if not mol.HasProp("_Name") or not mol.GetProp("_Name").strip():
            mol.SetProp("_Name", name if len(mols) == 1 else f"{name}_{index + 1}")
        if embed and _needs_embedding(mol):
            mol = embed_3d(mol, seed=seed, force=True)
        out.append(mol)
    return out


def _collect(source, kind_override: Optional[str]) -> Tuple[List[object], List[str]]:
    """Resolve ``source`` into ``(molecules, notes)`` without embedding."""
    Chem_ = Chem
    mols: List[object] = []
    notes: List[str] = []

    if Chem_ is not None and isinstance(source, Chem_.Mol):
        return [Chem_.Mol(source)], notes

    if isinstance(source, (str, os.PathLike)):
        raw = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
        path = Path(raw)
        if path.exists() and path.is_file():
            kind = kind_override or _FORMATS.get(path.suffix.lower().lstrip("."))
            if kind is None:
                raise ValueError(
                    f"cannot infer the ligand format from {path.suffix!r}; "
                    f"pass fmt=... ({sorted(set(_FORMATS))})"
                )
            return _read_file(path, kind)
        if kind_override is not None:
            if kind_override in _TEXT_FORMATS:
                return _read_smiles_text(raw)
            if "\n" in raw:
                return _read_text_records(kind_override, raw)
            raise FileNotFoundError(f"no such file: {raw}")
        # Not a path: one or more "SMILES name" records, which is exactly what a
        # `.smi` file holds.  A SMILES never contains whitespace, so a string
        # with more than one token is always a named record.
        if "\n" in raw or len(raw.split()) > 1:
            return _read_smiles_text(raw)
        mol = Chem_.MolFromSmiles(raw)
        if mol is None:
            raise ValueError(
                f"{raw!r} is neither an existing file nor a parsable SMILES string"
            )
        return [mol], notes

    # An iterable of sources: paths, SMILES, molecules.
    try:
        items = list(source)
    except TypeError:
        raise TypeError(
            f"cannot read ligands from {type(source).__name__!r}; expected a path, "
            "a SMILES string, a molecule or an iterable of those"
        ) from None
    for item in items:
        sub_mols, sub_notes = _collect(item, kind_override)
        mols.extend(sub_mols)
        notes.extend(sub_notes)
    return mols, notes


def _read_text_records(kind: str, text: str) -> Tuple[List[object], List[str]]:
    """Parse in-memory text for the block formats (SDF/MOL2/PDB/PDBQT)."""
    notes: List[str] = []
    mols: List[object] = []
    if kind == "sdf":
        blocks = _split_records(text, "$$$$")
    elif kind == "mol2":
        blocks = _split_mol2_records(text)
        if not blocks:
            blocks = [text]
    elif kind in ("pdb", "pdbqt"):
        blocks = _split_models(text)
    else:  # mol
        blocks = [text]
    for index, block in enumerate(blocks):
        mol = _mol_from_text_block(kind, block)
        if mol is None:
            notes.append(f"record {index + 1}: RDKit could not parse it")
            continue
        mols.append(mol)
    return mols, notes


# ---------------------------------------------------------------------------
# Force-field minimisation
# ---------------------------------------------------------------------------


def _normalise_force_field(force_field: str) -> str:
    wanted = str(force_field).strip()
    for known in FORCE_FIELDS:
        if wanted.lower() == known.lower():
            return known
    raise ValueError(
        f"unknown force field {force_field!r}; supported: {list(FORCE_FIELDS)}"
    )


def _make_force_field(mol, name: str):
    """Build the RDKit force-field object, or raise with the reason it cannot."""
    if name == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(mol):
            raise ValueError("UFF has no parameters for every atom of this molecule")
        ff = AllChem.UFFGetMoleculeForceField(mol)
    else:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            raise ValueError(f"{name} has no parameters for every atom of this molecule")
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=name)
        if props is None:
            raise ValueError(f"{name} could not type this molecule")
        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
    if ff is None:
        raise ValueError(f"{name} could not build a force-field object")
    return ff


def _minimize_result(mol, *, notes, force_field, steps, before, after, status=None):
    mol.SetProp("odock_energy_before", _format_energy(before))
    mol.SetProp("odock_energy_after", _format_energy(after))
    mol.SetProp("odock_minimize_steps", str(int(steps)))
    if force_field is not None:
        mol.SetProp("odock_force_field", force_field)
    if status is not None:
        mol.SetProp("odock_minimize_converged", "1" if status == 0 else "0")
    mol.SetProp("odock_minimize_note", "; ".join(notes))
    return mol


def _format_energy(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def minimize(mol, *, force_field: str = "MMFF94", steps: int = 500):
    """Gradient-minimise ``mol`` and report the energy before and after.

    Parameters
    ----------
    mol
        The ligand.  A molecule without a 3-D conformer is embedded first; the
        input molecule is never modified — a copy is returned.
    force_field
        One of :data:`FORCE_FIELDS`.  The name is matched case-insensitively.
    steps
        Gradient steps, clamped into ``[200, 1000]`` as the project requires.

    Returns
    -------
    The minimised copy.  It always carries

    * ``odock_energy_before`` / ``odock_energy_after`` — kcal/mol, six decimals
      (``"nan"`` when no force field could type the molecule);
    * ``odock_force_field`` — the force field that actually ran;
    * ``odock_minimize_steps`` — the clamped step count;
    * ``odock_minimize_converged`` — ``"1"`` when the minimiser converged;
    * ``odock_minimize_note`` — why a fallback happened, if one did.

    The force field is never allowed to raise: a molecule MMFF94 cannot type
    (boron, most metals, ...) falls through MMFF94s to UFF, and a molecule no
    force field can type comes back with the input geometry and ``nan``
    energies plus the reason in ``odock_minimize_note``.  If the minimiser
    returns a geometry that is *not* lower in energy — which a force field
    re-typed at the end of a step can do — the input coordinates are restored
    so the energy reported can never increase.
    """
    Chem_, _ = _require_rdkit()
    name = _normalise_force_field(force_field)
    if isinstance(steps, bool) or not isinstance(steps, (int, float)):
        raise ValueError(f"steps must be a number, got {steps!r}")
    n_steps = max(MIN_STEPS, min(MAX_STEPS, int(steps)))

    work = Chem_.Mol(mol)
    notes: List[str] = []
    if _needs_embedding(work):
        try:
            work = embed_3d(work, force=True)
            notes.append(
                "the input had no 3-D conformer; it was embedded with ETKDGv3 "
                "before minimisation"
            )
        except Exception as exc:
            notes.append(
                f"no 3-D conformer could be generated ({exc}); the molecule was "
                "returned unminimised"
            )
            return _minimize_result(
                work, notes=notes, force_field=None, steps=n_steps, before=None, after=None
            )

    before_coords = _positions(work)
    fallback_order = [name] + [f for f in FORCE_FIELDS if f != name]
    for candidate in fallback_order:
        with _quiet_rdkit():
            try:
                ff = _make_force_field(work, candidate)
            except Exception as exc:
                notes.append(f"{candidate} could not type the molecule: {exc}")
                continue
            try:
                energy_before = float(ff.CalcEnergy())
                if not math.isfinite(energy_before):
                    notes.append(f"{candidate} returned a non-finite energy")
                    continue
                status = int(ff.Minimize(maxIts=n_steps))
                energy_after = float(ff.CalcEnergy())
            except Exception as exc:  # pragma: no cover - RDKit rarely raises here
                _set_positions(work, before_coords)
                notes.append(f"{candidate} failed during minimisation: {exc}")
                continue
        if not math.isfinite(energy_after):
            _set_positions(work, before_coords)
            notes.append(f"{candidate} returned a non-finite energy after minimisation")
            continue
        if energy_after > energy_before:
            _set_positions(work, before_coords)
            notes.append(
                f"{candidate} did not lower the energy "
                f"({energy_before:.4f} -> {energy_after:.4f} kcal/mol); the input "
                "geometry was kept"
            )
            energy_after = energy_before
        if candidate != name:
            notes.append(f"fell back to {candidate}")
        return _minimize_result(
            work,
            notes=notes,
            force_field=candidate,
            steps=n_steps,
            before=energy_before,
            after=energy_after,
            status=status,
        )

    notes.append("no force field in " + ", ".join(FORCE_FIELDS) + " could type the molecule")
    return _minimize_result(
        work, notes=notes, force_field=None, steps=n_steps, before=None, after=None
    )


# ---------------------------------------------------------------------------
# Rotatable-bond perception
# ---------------------------------------------------------------------------

#: The bond is a genuine torsion.
ROTATABLE = "rotatable"
#: The bond involves a hydrogen (only the H itself spins).
RIGID_HYDROGEN = "hydrogen"
#: Double, triple or otherwise non-single bond — rotating it would break the
#: pi system.  This is the rule that locks an alkyne (C#C).
RIGID_BOND_ORDER = "bond-order"
#: Aromatic bond: the ring's conjugated system makes it rigid.
RIGID_AROMATIC = "aromatic"
#: Any other ring bond: rotating it would have to break the ring.
RIGID_RING = "ring"
#: Amide C-N: resonance with the adjacent carbonyl gives it partial double-bond
#: character, so it is locked.
RIGID_AMIDE = "amide"
#: One end has a single heavy neighbour (methyl, -OH, -NH2, -CF3, halide), so
#: no heavy atom moves when the bond turns.
RIGID_TERMINAL = "terminal"
#: One end carries three or more *equivalent* heavy substituents (tert-butyl,
#: trimethylammonium, ...): the rotation only permutes identical groups.
RIGID_SYMMETRIC = "symmetric"
#: The caller pinned the bond with ``locked=[...]``.
RIGID_LOCKED = "locked"

#: Machine-readable reason -> one-line explanation, for the GUI and the logs.
REASON_NOTES: Dict[str, str] = {
    ROTATABLE: "free torsion",
    RIGID_HYDROGEN: "the bond involves a hydrogen; only the hydrogen itself turns",
    RIGID_BOND_ORDER: "not a single bond (double/triple): the pi system would break",
    RIGID_AROMATIC: "aromatic bond inside a conjugated ring system",
    RIGID_RING: "ring bond: rotation would have to break the ring",
    RIGID_AMIDE: "amide C-N: resonance gives it partial double-bond character",
    RIGID_TERMINAL: "terminal group (methyl, -OH, -NH2, -CF3, halide): "
    "no heavy atom moves",
    RIGID_SYMMETRIC: "three or more equivalent substituents (tert-butyl, "
    "symmetric quaternary/tertiary centre): rotation only permutes identical groups",
    RIGID_LOCKED: "locked by the user",
}

#: Every reason :func:`rotation_reason` can return.
RIGID_REASONS: Tuple[str, ...] = (
    RIGID_HYDROGEN,
    RIGID_BOND_ORDER,
    RIGID_AROMATIC,
    RIGID_RING,
    RIGID_AMIDE,
    RIGID_TERMINAL,
    RIGID_SYMMETRIC,
    RIGID_LOCKED,
)

def _amide_bond_indices(mol) -> set:
    """Bond indices of every amide C-N bond in ``mol``.

    An amide is a nitrogen single-bonded to a carbon that is double-bonded to an
    oxygen.  The resonance ``N-C=O <-> N+=C-O-`` gives that bond roughly 40 %
    double-bond character and a rotation barrier near 20 kcal/mol, so treating
    it as a torsion would add a search dimension along a mode the molecule does
    not have.

    The test walks the *bond graph* instead of using a SMARTS because a SMARTS
    would depend on the hydrogen model: a MOL2 file marks every atom
    ``NoImplicit``, so ``[NX3]`` would fail to match an ordinary secondary amide.
    This mirrors :func:`odock.pdbqt._is_amide_bond`.
    """
    out = set()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if {begin.GetAtomicNum(), end.GetAtomicNum()} != {6, 7}:
            continue
        carbon = begin if begin.GetAtomicNum() == 6 else end
        for neighbour in carbon.GetNeighbors():
            if neighbour.GetAtomicNum() != 8:
                continue
            carbonyl = mol.GetBondBetweenAtoms(carbon.GetIdx(), neighbour.GetIdx())
            if carbonyl is not None and carbonyl.GetBondType() == Chem.BondType.DOUBLE:
                out.add(bond.GetIdx())
                break
    return out


def _symmetry_ranks(mol) -> Sequence[int]:
    """Canonical atom ranks with ties *not* broken, i.e. symmetry classes."""
    try:
        return Chem.CanonicalRankAtoms(mol, breakTies=False)
    except Exception:  # pragma: no cover - defensive
        return [0] * mol.GetNumAtoms()


def _symmetric_end(atom, other_idx: int, ranks: Sequence[int]) -> bool:
    """Whether rotating about ``atom``-``other`` only permutes equal groups.

    ``atom`` sits on the rotation axis, so the rotation is a symmetry operation
    of the molecule exactly when its remaining substituents can be permuted by a
    rotation about that axis.  With three or more substituents at a tetrahedral
    (or trigonal-bipyramidal, or octahedral) centre, three *identical*
    substituents force a threefold axis along the ``atom``-``other`` bond — the
    tert-butyl, the symmetric quaternary carbon and the trimethylammonium case.

    Two identical substituents are deliberately *not* enough: their local
    symmetry is a mirror plane, and rotating by 120 degrees would have to move
    the odd substituent (the hydrogen of an isopropyl group, say) as well, which
    changes the conformation.  That is why ``CC(C)CC`` keeps its torsion while
    ``CC(C)(C)CC`` does not.
    """
    neighbours = [
        n
        for n in atom.GetNeighbors()
        if n.GetAtomicNum() > 1 and n.GetIdx() != other_idx
    ]
    if len(neighbours) < 3:
        return False
    return len({ranks[n.GetIdx()] for n in neighbours}) == 1


def _locked_keys(mol, locked: Sequence[Tuple[int, int]]) -> set:
    """Locked atom pairs that are real bonds of ``mol``.

    Pairs that are not bonds are ignored rather than rejected: the GUI's bond
    locker holds indices alongside a molecule that the user may have edited, and
    an index it can no longer resolve simply means "nothing to lock".
    """
    keys = set()
    for pair in locked or ():
        try:
            i, j = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if i == j or i < 0 or j < 0:
            continue
        if i >= mol.GetNumAtoms() or j >= mol.GetNumAtoms():
            continue
        if mol.GetBondBetweenAtoms(i, j) is None:
            continue
        keys.add(_pair_key(i, j))
    return keys


def _bond_index(mol, bond) -> int:
    if isinstance(bond, int):
        if 0 <= bond < mol.GetNumBonds():
            return bond
        raise IndexError(f"bond index {bond} is out of range")
    try:
        i, j = int(bond[0]), int(bond[1])
    except (TypeError, ValueError, IndexError):
        raise TypeError(
            "bond must be a bond index or an (i, j) atom pair, got "
            f"{bond!r}"
        ) from None
    found = mol.GetBondBetweenAtoms(i, j)
    if found is None:
        raise ValueError(f"atoms {i} and {j} are not bonded")
    return found.GetIdx()


def _reason_for_bond(
    mol, bond, *, locked_keys: set, amide_bonds: set, ranks: Sequence[int]
) -> str:
    """Classify one bond; the order of the tests is the order of precedence."""
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    if _pair_key(begin.GetIdx(), end.GetIdx()) in locked_keys:
        return RIGID_LOCKED
    if begin.GetAtomicNum() == 1 or end.GetAtomicNum() == 1:
        return RIGID_HYDROGEN
    if bond.GetIsAromatic():
        return RIGID_AROMATIC
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return RIGID_BOND_ORDER
    if bond.IsInRing():
        return RIGID_RING
    if bond.GetIdx() in amide_bonds:
        return RIGID_AMIDE
    if _heavy_degree(begin) < 2 or _heavy_degree(end) < 2:
        return RIGID_TERMINAL
    if _symmetric_end(begin, end.GetIdx(), ranks) or _symmetric_end(
        end, begin.GetIdx(), ranks
    ):
        return RIGID_SYMMETRIC
    return ROTATABLE


def rotation_reason(mol, bond, *, locked: Sequence[Tuple[int, int]] = ()) -> str:
    """Why ``bond`` is or is not a torsion, as one of the reason constants.

    ``bond`` is either a bond index or an ``(i, j)`` atom pair.  The return
    value is :data:`ROTATABLE` or one of :data:`RIGID_REASONS`; pair it with
    :data:`REASON_NOTES` for a sentence to show the user.

    This helper classifies a single bond, so it re-derives the amide matches and
    the symmetry classes every call.  Use :func:`rotatable_bonds` to classify a
    whole molecule.

    Examples
    --------
    >>> rotation_reason(Chem.MolFromSmiles("CC(=O)NC"), (1, 2))  # amide C-N
    'amide'
    >>> rotation_reason(Chem.MolFromSmiles("CC#CC"), (1, 2))     # alkyne
    'bond-order'
    """
    _require_rdkit()
    index = _bond_index(mol, bond)
    bond_obj = mol.GetBondWithIdx(index)
    return _reason_for_bond(
        mol,
        bond_obj,
        locked_keys=_locked_keys(mol, locked),
        amide_bonds=_amide_bond_indices(mol),
        ranks=_symmetry_ranks(mol),
    )


def rotatable_bonds(
    mol, *, locked: Sequence[Tuple[int, int]] = ()
) -> List[Tuple[int, int]]:
    """The rotatable bonds of ``mol`` as sorted ``(i, j)`` atom pairs.

    A bond is a torsion only when *all* of the following hold; the first test
    that fails is the reason :func:`rotation_reason` reports.

    ==========================  ============================================
    rule                        why
    ==========================  ============================================
    no hydrogen                 only the H itself would turn
    single bond                 double/triple bonds (an alkyne's C#C) are rigid
    not aromatic                conjugated ring systems are rigid
    not in a ring               a ring torsion would break the ring
    not an amide                C-N resonance gives it double-bond character
    both ends non-terminal      a methyl/-OH/-NH2/-CF3 has nothing to move
    no symmetric end            a tert-butyl or symmetric quaternary centre
                                only permutes equivalent groups
    not in ``locked``           the user pinned it
    ==========================  ============================================

    ``locked`` holds ``(i, j)`` atom pairs to force rigid — what the GUI's
    interactive bond locker passes.  Pairs that are not bonds of ``mol`` are
    ignored.  The result is in bond-index order with each pair normalised to
    ``(min, max)``, so it is deterministic and independent of the caller's
    atom ordering conventions.
    """
    _require_rdkit()
    locked_keys = _locked_keys(mol, locked)
    amide_bonds = _amide_bond_indices(mol)
    ranks = _symmetry_ranks(mol)
    out: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if _reason_for_bond(
            mol, bond, locked_keys=locked_keys, amide_bonds=amide_bonds, ranks=ranks
        ) == ROTATABLE:
            out.append(
                _pair_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            )
    return out


# ---------------------------------------------------------------------------
# Kinematic torsion tree
# ---------------------------------------------------------------------------


@dataclass
class TorsionTree:
    """A rigid-fragment tree: the ``ROOT -> BRANCH`` structure of AutoDock PDBQT.

    ``root_atoms`` are the atoms of this rigid fragment (excluding the atom that
    attaches it to its parent — that one belongs to the parent's frame, exactly
    as the PDBQT dialect writes it).  ``children`` holds
    ``(parent_atom, attach_atom, subtree)``: ``parent_atom`` is the axis atom in
    this frame, ``attach_atom`` the axis atom in the child frame.
    """

    #: Atom that seeds the fragment (informational; any member identifies it).
    root: int
    #: Atom indices of this rigid fragment.
    root_atoms: Tuple[int, ...]
    #: ``(parent_atom, attach_atom, subtree)`` for every torsion leaving it.
    children: List[Tuple[int, int, "TorsionTree"]] = field(default_factory=list)
    #: Number of torsions in this subtree; on the root, the molecule's TORSDOF.
    num_torsions: int = 0
    #: Rotors that had to be folded into a rigid fragment because the torsion
    #: tree cannot contain a cycle.  Ring bonds are never torsions, so with
    #: correct ring perception this stays empty; it is a safety net for a
    #: molecule whose ring perception is incomplete (an unsanitised PDBQT, for
    #: instance).  Root only.
    dropped_torsions: Tuple[Tuple[int, int], ...] = ()

    @property
    def torsdof(self) -> int:
        """The molecule's total torsional degrees of freedom."""
        return self.num_torsions

    @property
    def torsions(self) -> List[Tuple[int, int]]:
        """``(parent_atom, attach_atom)`` of every torsion, depth-first."""
        out: List[Tuple[int, int]] = []

        def walk(node: "TorsionTree") -> None:
            for parent, attach, sub in node.children:
                out.append((parent, attach))
                walk(sub)

        walk(self)
        return out

    def iter_nodes(self) -> Iterator["TorsionTree"]:
        """Yield this node and every descendant, depth-first."""
        yield self
        for _, _, sub in self.children:
            yield from sub.iter_nodes()

    def atoms(self) -> Tuple[int, ...]:
        """Every atom index in this subtree, sorted."""
        out = set(self.root_atoms)
        for _, attach, sub in self.children:
            out.add(attach)
            out |= set(sub.atoms())
        return tuple(sorted(out))

    def as_dict(self) -> Dict[str, object]:
        """A JSON-serialisable view, for the GUI's torsion-tree panel."""
        return {
            "root": self.root,
            "root_atoms": list(self.root_atoms),
            "num_torsions": self.num_torsions,
            "dropped_torsions": [list(p) for p in self.dropped_torsions],
            "children": [
                {"parent_atom": p, "attach_atom": a, "node": sub.as_dict()}
                for p, a, sub in self.children
            ],
        }


def _planar_centroid(coords: Optional[np.ndarray], indices: Sequence[int]) -> Optional[np.ndarray]:
    if coords is None or len(indices) == 0:
        return None
    return coords[list(indices)].mean(axis=0)


def torsion_tree(mol, *, locked: Sequence[Tuple[int, int]] = ()) -> TorsionTree:
    """Build the kinematic torsion tree of ``mol``.

    The molecule is split into rigid fragments by removing every rotatable bond
    (:func:`rotatable_bonds`, so ``locked`` applies).  The **largest** fragment
    becomes the root — the fused ring system or the central carbon that the
    project requirements ask the root finder to pick — with the fragment closest
    to the molecular centroid breaking a size tie.  Depth-first search then opens
    one ``BRANCH`` per rotor leaving a frame.

    A torsion tree cannot contain a cycle — AutoDock has no syntax for a closed
    loop — so a second rotor closing a ring of fragments is folded into its
    frame; those pairs are reported in
    :attr:`TorsionTree.dropped_torsions`.  Because a ring bond is never a
    torsion, that can only happen when ring perception is incomplete.
    """
    require_rdkit()
    rotors = rotatable_bonds(mol, locked=locked)
    rotor_keys = set(rotors)
    n_atoms = mol.GetNumAtoms()

    # Rigid fragments: connected components of the graph with the rotors cut.
    fragment = [-1] * n_atoms
    fragments: List[List[int]] = []
    for start in range(n_atoms):
        if fragment[start] != -1:
            continue
        fid = len(fragments)
        stack = [start]
        fragment[start] = fid
        members = [start]
        while stack:
            current = stack.pop()
            for bond in mol.GetAtomWithIdx(current).GetBonds():
                if _pair_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) in rotor_keys:
                    continue
                other = bond.GetOtherAtomIdx(current)
                if fragment[other] == -1:
                    fragment[other] = fid
                    members.append(other)
                    stack.append(other)
        fragments.append(sorted(members))

    coords: Optional[np.ndarray] = _positions(mol) if mol.GetNumConformers() else None
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1] or list(range(n_atoms))
    molecular_centroid = _planar_centroid(coords, heavy)

    def root_key(fid: int):
        members = fragments[fid]
        distance = 0.0
        if molecular_centroid is not None:
            own = _planar_centroid(coords, members)
            if own is not None:
                distance = float(np.linalg.norm(own - molecular_centroid))
        # Largest fragment first, then closest to the molecular centre, then
        # lowest atom index so the choice is fully deterministic.
        return (-len(members), distance, members[0])

    root_fid = min(range(len(fragments)), key=root_key)
    visited = {root_fid}
    used_rotors: set = set()
    dropped: List[Tuple[int, int]] = []

    def build(fid: int, incoming_attach: Optional[int]) -> TorsionTree:
        members = tuple(i for i in fragments[fid] if i != incoming_attach)
        children: List[Tuple[int, int, TorsionTree]] = []
        claimed = {incoming_attach} if incoming_attach is not None else set()
        for idx in fragments[fid]:
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                pair = _pair_key(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                if pair not in rotor_keys or pair in used_rotors:
                    # Not a rotor, or the very rotor that opened this frame: it
                    # is already represented by the branch we came in through.
                    continue
                other = bond.GetOtherAtomIdx(idx)
                if fragment[other] == fid or other in claimed:
                    continue
                if fragment[other] in visited:
                    # Closing a cycle of fragments: the torsion tree has to stay
                    # a tree, so this rotor becomes part of the rigid frame.
                    dropped.append(pair)
                    continue
                claimed.add(other)
                visited.add(fragment[other])
                used_rotors.add(pair)
                children.append((idx, other, build(fragment[other], other)))
        # A bridging atom (an ether oxygen between two rotors, say) is the
        # incoming attachment and owns no atom of its own frame.
        seed = incoming_attach if incoming_attach is not None and not members else (
            members[0] if members else fid
        )
        return TorsionTree(
            root=seed, root_atoms=members, children=children
        )

    tree = build(root_fid, None)
    total = len(tree.torsions)
    tree.num_torsions = total
    tree.dropped_torsions = tuple(sorted(set(dropped)))
    return tree

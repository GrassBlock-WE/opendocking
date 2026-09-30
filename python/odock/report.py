# SPDX-License-Identifier: GPL-3.0-or-later
"""Tabular export of a docking result: the CSV/Excel entry of the Analysis menu.

The table is the one the workbench shows at the bottom of the window:

    Rank | Binding Energy (kcal/mol) | RMSD l.b. | RMSD u.b. | key residues

:func:`result_rows` builds it, :func:`write_csv` and :func:`write_xlsx` write
it. The "key interacting residues" column is filled by profiling the receptor
against each pose with :mod:`odock.analysis`; pass the receptor either as an
object (RDKit molecule or a sequence of Atom-like objects) or let the module
parse ``result.receptor_pdbqt`` for you.
"""

from __future__ import annotations

import csv
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from . import analysis as _analysis

__all__ = ["COLUMNS", "CSV_HEADERS", "result_rows", "write_csv", "write_xlsx"]

PathLike = Union[str, os.PathLike]

#: Column keys, in order.
COLUMNS: Sequence[str] = ("mode", "affinity", "rmsd_lb", "rmsd_ub", "residues")

#: Header text for each column.
CSV_HEADERS: Dict[str, str] = {
    "mode": "Mode",
    "affinity": "Binding Energy (kcal/mol)",
    "rmsd_lb": "RMSD l.b. (A)",
    "rmsd_ub": "RMSD u.b. (A)",
    "residues": "Key interacting residues",
}

#: AutoDock atom type -> element, for reading a PDBQT without RDKit.
_AD_TYPE_ELEMENT: Dict[str, str] = {
    "C": "C", "A": "C", "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C",
    "N": "N", "NA": "N", "O": "O", "OA": "O", "S": "S", "SA": "S",
    "P": "P", "H": "H", "HD": "H", "F": "F", "I": "I", "Cl": "Cl",
    "Br": "Br", "Si": "Si", "At": "At", "W": "H",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
    "G0": "H", "G1": "H", "G2": "H", "G3": "H",
}


@dataclass
class _Atom:
    """The subset of :class:`odock.gui.structure.Atom` this module needs."""

    name: str
    element: str
    res_name: str
    res_id: int
    chain: str
    x: float
    y: float
    z: float
    charge: float = 0.0


def _element_from_pdbqt(name: str, ad_type: str) -> str:
    element = _AD_TYPE_ELEMENT.get(ad_type)
    if element:
        return element
    letters = "".join(c for c in name if c.isalpha())
    two = letters[:2].capitalize()
    if two in ("Cl", "Br", "Si", "At"):
        return two
    return letters[:1].upper() or "C"


def parse_pdbqt_atoms(text: str) -> List[_Atom]:
    """Read the ``ATOM``/``HETATM`` records of a PDBQT document, in order.

    Only the fields the interaction profiler needs are kept, and the element is
    taken from the AutoDock type columns (77-78), which is exact -- unlike the
    atom *name*, which is only a convention.
    """
    atoms: List[_Atom] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        name = line[12:16].strip()
        ad_type = line[77:].strip() if len(line) >= 79 else ""
        if not ad_type:
            tokens = line.split()
            ad_type = tokens[-1] if tokens else ""
        try:
            res_id = int(line[22:26])
        except ValueError:
            res_id = 0
        charge = 0.0
        if len(line) >= 76:
            try:
                charge = float(line[70:76])
            except ValueError:
                charge = 0.0
        atoms.append(
            _Atom(
                name=name,
                element=_element_from_pdbqt(name, ad_type),
                res_name=line[17:20].strip(),
                res_id=res_id,
                chain=line[21:22].strip(),
                x=x,
                y=y,
                z=z,
                charge=charge,
            )
        )
    return atoms


def _first_model(text: str) -> str:
    """The first ``MODEL`` block of a multi-model document (or the whole text)."""
    if "MODEL" not in text:
        return text
    lines: List[str] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("MODEL"):
            if inside:
                break
            inside = True
            continue
        if line.startswith("ENDMDL"):
            break
        if inside:
            lines.append(line)
    return "\n".join(lines)


def _receptor_source(receptor, result):
    """The receptor in a form :mod:`odock.analysis` understands, or ``None``."""
    if receptor is not None:
        if isinstance(receptor, (str, os.PathLike)):
            path = Path(str(receptor))
            text = (
                path.read_text(encoding="utf-8", errors="replace")
                if path.exists()
                else str(receptor)
            )
            return parse_pdbqt_atoms(text) or None
        return receptor
    text = getattr(result, "receptor_pdbqt", "") or ""
    if not text.strip():
        return None
    return parse_pdbqt_atoms(text) or None


def _ligand_atoms(result, pose) -> List[_Atom]:
    """The ligand of one pose, in the kernel's atom order.

    ``ligand_pdbqt`` is written in exactly the order the kernel flattens the
    kinematic tree, which is the order ``pose.coords`` uses (see
    ``docs/DATA_STRUCTURES.md``), so the file's atom records can be paired with
    the pose coordinates one to one.
    """
    text = getattr(result, "ligand_pdbqt", "") or ""
    if not text.strip():
        text = _first_model(getattr(result, "to_pdbqt", lambda: "")() or "")
    atoms = parse_pdbqt_atoms(_first_model(text))
    coords = getattr(pose, "coords", None)
    if coords is not None:
        array = getattr(coords, "shape", None)
        if array is not None and len(array) == 2 and int(array[0]) == len(atoms):
            for atom, xyz in zip(atoms, coords):
                atom.x, atom.y, atom.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return atoms


def result_rows(result, *, receptor=None) -> List[Dict[str, Any]]:
    """The report table as a list of rows, one per pose.

    Keys are :data:`COLUMNS`: ``mode`` (1-based rank), ``affinity``,
    ``rmsd_lb``, ``rmsd_ub`` and ``residues`` (``"ASP189, GLY216"``).

    `receptor` may be an RDKit molecule, a sequence of Atom-like objects, a
    path to (or the text of) a receptor PDBQT, or ``None``. With ``None`` the
    module falls back to ``result.receptor_pdbqt``; if that is empty too, the
    residue column is blank. A pose that cannot be profiled (no receptor, no
    ligand records, a malformed structure) also leaves the column blank rather
    than failing the export -- exporting a table must not be able to lose a
    completed docking run.
    """
    rows: List[Dict[str, Any]] = []
    poses = list(getattr(result, "poses", None) or [])
    rec_structure = None
    try:
        source = _receptor_source(receptor, result)
        if source is not None:
            # Normalise once: bond perception on a few thousand receptor atoms
            # is by far the most expensive step, and every pose reuses it.
            candidate = _analysis._structure(source)
            if candidate.atoms:
                rec_structure = candidate
    except Exception as exc:
        warnings.warn(f"could not read the receptor for the report: {exc}", stacklevel=2)
        rec_structure = None

    warned = False
    for position, pose in enumerate(poses):
        residues = ""
        if rec_structure is not None and rec_structure.atoms:
            try:
                lig_atoms = _ligand_atoms(result, pose)
                if lig_atoms:
                    lig_structure = _analysis._structure(lig_atoms)
                    contacts = _analysis.profile_interactions(
                        rec_structure, lig_structure
                    )
                    residues = _analysis.interaction_summary(
                        contacts, rec_structure, lig_structure
                    )
            except Exception as exc:
                if not warned:
                    warnings.warn(
                        f"interaction profiling failed for the report: {exc}",
                        stacklevel=2,
                    )
                    warned = True
        rows.append(
            {
                "mode": position + 1,
                "affinity": float(getattr(pose, "affinity", 0.0) or 0.0),
                "rmsd_lb": float(getattr(pose, "rmsd_lower_bound", 0.0) or 0.0),
                "rmsd_ub": float(getattr(pose, "rmsd_upper_bound", 0.0) or 0.0),
                "residues": residues,
            }
        )
    return rows


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and str(parent) not in (".", "") and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: PathLike, result, *, receptor=None) -> str:
    """Write the pose table to `path` as CSV (UTF-8, one header row).

    Returns the path that was written, as a string.
    """
    rows = result_rows(result, receptor=receptor)
    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writerow({key: CSV_HEADERS[key] for key in COLUMNS})
        for row in rows:
            writer.writerow(row)
    return str(target)


def write_xlsx(path: PathLike, result, *, receptor=None) -> str:
    """Write the pose table to `path` as an Excel workbook.

    Sheet ``Poses`` holds the table (styled, frozen header, numeric formats) and
    sheet ``Run`` holds the run metadata. `openpyxl` is required; when it is not
    installed the call degrades to :func:`write_csv` on the same path with a
    ``.csv`` suffix, emits a warning, and returns the CSV path that was
    actually written.
    """
    rows = result_rows(result, receptor=receptor)
    target = Path(path)

    try:
        from openpyxl import Workbook
    except Exception:  # pragma: no cover - openpyxl is bundled, but be safe
        warnings.warn(
            "openpyxl is not available; the report was written as CSV instead",
            stacklevel=2,
        )
        return write_csv(target.with_suffix(".csv"), result, receptor=receptor)

    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Poses"

    header_font = Font(bold=True, color="FFFFFFFF")
    header_fill = PatternFill("solid", fgColor="FF37474F")
    for column, key in enumerate(COLUMNS, start=1):
        cell = sheet.cell(row=1, column=column, value=CSV_HEADERS[key])
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for index, row in enumerate(rows, start=2):
        sheet.cell(row=index, column=1, value=int(row["mode"]))
        sheet.cell(row=index, column=2, value=float(row["affinity"])).number_format = "0.000"
        sheet.cell(row=index, column=3, value=float(row["rmsd_lb"])).number_format = "0.000"
        sheet.cell(row=index, column=4, value=float(row["rmsd_ub"])).number_format = "0.000"
        sheet.cell(row=index, column=5, value=str(row["residues"]))

    for column, width in enumerate((8, 26, 14, 14, 46), start=1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A2"
    if rows:
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{len(rows) + 1}"

    run = workbook.create_sheet("Run")
    run.append(["OpenDocking report"])
    run["A1"].font = Font(bold=True, size=14)
    run.append([])
    metadata = [
        ("poses", len(rows)),
        ("best affinity (kcal/mol)", _best_affinity(result)),
        ("seed", getattr(result, "seed", None)),
        ("exact scoring", getattr(result, "exact", None)),
        ("movable atoms", getattr(result, "num_movable_atoms", None)),
        ("degrees of freedom", getattr(result, "num_dof", None)),
        ("N_tors", getattr(result, "num_tors", None)),
        ("grid points", getattr(result, "grid_points", None)),
        ("grid megabytes", getattr(result, "grid_mb", None)),
        ("elapsed (s)", getattr(result, "elapsed", None)),
        ("box", _box_text(result)),
    ]
    for label, value in metadata:
        run.append([label, value])
        run.cell(row=run.max_row, column=1).font = Font(bold=True)
    run.column_dimensions["A"].width = 26
    run.column_dimensions["B"].width = 48

    _ensure_parent(target)
    workbook.save(str(target))
    return str(target)


def _best_affinity(result) -> Optional[float]:
    best = getattr(result, "best_affinity", None)
    if best is not None:
        return float(best)
    poses = list(getattr(result, "poses", None) or [])
    if not poses:
        return None
    return float(min(float(getattr(p, "affinity", 0.0) or 0.0) for p in poses))


def _box_text(result) -> str:
    box = getattr(result, "box", None)
    if box is None:
        return ""
    center = getattr(box, "center", (0.0, 0.0, 0.0))
    size = getattr(box, "size", (0.0, 0.0, 0.0))
    return (
        "center ("
        + ", ".join(f"{float(v):.3f}" for v in center)
        + ") size ("
        + ", ".join(f"{float(v):.3f}" for v in size)
        + ")"
    )

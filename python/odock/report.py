# SPDX-License-Identifier: GPL-3.0-or-later
"""Tabular export of a docking result: the CSV/Excel entry of the Analysis menu.

The base table is the one the workbench shows at the bottom of the window:

    Rank | Binding Energy | RMSD l.b. | RMSD u.b. | key residues

and every column after it is optional and additive, so an existing consumer of
the first five columns keeps working:

    Heavy atoms | LE | LogP | LLE | BEI | SEI | Entropy | Strain |
    Consensus rank | Consensus score

* **Ligand efficiency** comes from :mod:`odock.metrics`.  Ligand efficiency
  needs nothing but the affinity and the heavy-atom count, so it is filled for
  every run; LogP/TPSA/BEI/SEI need RDKit descriptors, which the caller supplies
  as `ligand_mol=` or `descriptors=` (they are *not* guessed from a PDBQT, which
  carries no bond orders and would give a wrong LogP).
* **Strain** is opt-in (`strain=True`) because it costs a force-field
  minimisation per pose, and it needs a receptor.
* **Consensus** is opt-in (`consensus=True`, with `box=` and `scorings=`),
  because it re-scores every pose with up to three force fields.  Pass an
  already-computed :class:`odock.consensus.ConsensusResult` to reuse one.

:func:`write_xlsx` can add a ``Fingerprints`` sheet (the pose x feature count
matrix) and a ``Pharmacophore`` sheet (the contacts that recur across the top
poses) when a :class:`odock.analysis.FingerprintSet` is supplied.

Two things exist for very large result sets: :func:`write_jsonl` and
:class:`ResultStream`, an append-only writer that never holds the whole table in
memory.  JSON is written with ``NaN`` normalised to ``null``, because ``NaN`` is
not valid JSON and a reader that fails on the 900th ligand of a screen is worse
than useless.
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import analysis as _analysis
from . import consensus as _consensus
from . import metrics as _metrics

__all__ = [
    "COLUMNS",
    "CSV_HEADERS",
    "CONSENSUS_COLUMNS",
    "METRIC_COLUMNS",
    "ResultStream",
    "columns_for",
    "decomposition_table_for",
    "notebook_summary",
    "parse_pdbqt_atoms",
    "result_rows",
    "write_csv",
    "write_jsonl",
    "write_rows",
    "write_xlsx",
]

PathLike = Union[str, os.PathLike]

#: Column keys, in order.  The first five are the historical table and never
#: move; everything after them is additive.
COLUMNS: Sequence[str] = (
    "mode",
    "affinity",
    "rmsd_lb",
    "rmsd_ub",
    "residues",
    "heavy_atoms",
    "ligand_efficiency",
    "logp",
    "lle",
    "bei",
    "sei",
    "entropy_penalty",
    "strain",
    "consensus_rank",
    "consensus_score",
)

#: The columns contributed by :mod:`odock.metrics`.
METRIC_COLUMNS: Sequence[str] = (
    "heavy_atoms",
    "ligand_efficiency",
    "logp",
    "lle",
    "bei",
    "sei",
    "entropy_penalty",
    "strain",
)

#: The columns contributed by :mod:`odock.consensus`.
CONSENSUS_COLUMNS: Sequence[str] = ("consensus_rank", "consensus_score")

#: Header text for each column.
CSV_HEADERS: Dict[str, str] = {
    "mode": "Mode",
    "affinity": "Binding Energy (kcal/mol)",
    "rmsd_lb": "RMSD l.b. (A)",
    "rmsd_ub": "RMSD u.b. (A)",
    "residues": "Key interacting residues",
    "heavy_atoms": "Heavy atoms",
    "ligand_efficiency": "LE (kcal/mol/HA)",
    "logp": "LogP",
    "lle": "LLE",
    "bei": "BEI",
    "sei": "SEI",
    "entropy_penalty": "Entropy penalty (kcal/mol)",
    "strain": "Ligand strain (kcal/mol)",
    "consensus_rank": "Consensus rank",
    "consensus_score": "Consensus score",
}

#: Suffix -> writer chosen by :class:`ResultStream` and :func:`write_rows`.
_JSON_SUFFIXES = (".jsonl", ".ndjson", ".json")
_CSV_SUFFIXES = (".csv", ".tsv")

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


# ---------------------------------------------------------------------------
# Column bookkeeping
# ---------------------------------------------------------------------------


def columns_for(rows, *, base: Sequence[str] = COLUMNS) -> List[str]:
    """The header for a list of row dictionaries: `base` plus any extra keys.

    Extra keys (a per-force-field consensus column, for instance) are appended
    in first-seen order, so the table stays deterministic and the historical
    columns keep their positions.  `base` columns that no row carries are
    dropped, which is what keeps a CSV written from a partial set readable.
    """
    ordered: List[str] = []
    seen = set()
    for key in base:
        ordered.append(key)
        seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return [key for key in ordered if any(key in row for row in rows) or key in base]


def _header(key: str) -> str:
    """The human header for a column key (``consensus_vina_rank`` -> readable)."""
    if key in CSV_HEADERS:
        return CSV_HEADERS[key]
    if key == "consensus_method":
        return "Consensus method"
    if key.startswith("consensus_") and key.endswith("_rank"):
        return f"{key[len('consensus_'):-len('_rank')].capitalize()} rank"
    if key.startswith("consensus_"):
        return f"{key[len('consensus_'):].capitalize()} (kcal/mol)"
    return key.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _ligand_descriptors(ligand_mol, descriptors) -> Dict[str, float]:
    """Merge a descriptor mapping (or an RDKit molecule) into one dict."""
    found: Dict[str, float] = {}
    if descriptors:
        for key, value in dict(descriptors).items():
            try:
                found[key] = float(value)
            except (TypeError, ValueError):
                continue
    if ligand_mol is not None:
        for key, value in _metrics.descriptors(ligand_mol).items():
            found.setdefault(key, value)
    return found


def _consensus_for(
    result,
    receptor,
    consensus,
    box,
    scorings,
) -> Tuple[Optional[_consensus.ConsensusResult], Optional[Dict[str, Any]]]:
    """Resolve the `consensus` argument into a result plus per-pose lookups."""
    if consensus is None or consensus is False:
        return None, None
    if isinstance(consensus, _consensus.ConsensusResult):
        computed = consensus
    else:
        computed = _consensus.consensus_score(
            result,
            receptor if receptor is not None else getattr(result, "receptor_pdbqt", ""),
            box if box is not None else getattr(result, "box", None),
            scorings=scorings,
        )
    by_index = {pose.index: pose for pose in computed.poses}
    return computed, by_index


def result_rows(
    result,
    *,
    receptor=None,
    ligand_mol=None,
    descriptors=None,
    consensus=None,
    box=None,
    scorings: Sequence[str] = _consensus.DEFAULT_SCORINGS,
    strain: bool = False,
    strain_scoring: Optional[str] = None,
    strain_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """The report table as a list of rows, one per pose.

    The first five keys are :data:`COLUMNS`'s first five and are always present
    (``mode``, ``affinity``, ``rmsd_lb``, ``rmsd_ub``, ``residues``).  The
    metric keys are present too, but are ``None`` (blank in a CSV, empty in a
    spreadsheet) when their input is missing:

    ==================  =====================================================
    ``heavy_atoms``     counted from ``result.ligand_pdbqt``; exact and free
    ``ligand_efficiency``  ``-affinity / heavy_atoms``
    ``logp``, ``lle``, ``bei``, ``sei``
                        need `ligand_mol` or `descriptors`; blank otherwise
    ``entropy_penalty``  from the torsion count (chemical if known, else the
                        kernel's ``N_tors``)
    ``strain``          only with ``strain=True`` and a receptor
    ``consensus_rank``, ``consensus_score``
                        only with ``consensus=``
    ==================  =====================================================

    `receptor` may be an RDKit molecule, a sequence of Atom-like objects, a path
    to (or the text of) a receptor PDBQT, or ``None``.  With ``None`` the module
    falls back to ``result.receptor_pdbqt``; if that is empty too, the residue
    column is blank.  A pose that cannot be profiled (no receptor, no ligand
    records, a malformed structure) also leaves the column blank rather than
    failing the export -- exporting a table must not be able to lose a completed
    docking run.
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

    descriptors_found = _ligand_descriptors(ligand_mol, descriptors)
    heavy = descriptors_found.get("heavy_atoms")
    if heavy is None:
        template = getattr(result, "ligand_pdbqt", "") or ""
        if template.strip():
            heavy = float(_metrics.heavy_atom_count(template))
        else:
            first = _first_model(getattr(result, "to_pdbqt", lambda: "")() or "")
            heavy = float(_metrics.heavy_atom_count(first)) if first.strip() else None

    num_torsions = descriptors_found.get("num_torsions")
    if num_torsions is None:
        kernel_tors = getattr(result, "num_tors", None)
        if kernel_tors is not None:
            num_torsions = float(kernel_tors)

    computed_consensus, consensus_by_index = _consensus_for(
        result, receptor, consensus, box, scorings
    )

    strains: List[Optional[float]] = [None] * len(poses)
    if strain:
        if rec_structure is None:
            warnings.warn(
                "strain was requested but no receptor was available; the column "
                "is left blank",
                stacklevel=2,
            )
        else:
            scoring = strain_scoring or getattr(result, "scoring", None) or "vina"
            for position, pose in enumerate(poses):
                try:
                    measured = _metrics.pose_strain(
                        result,
                        pose,
                        receptor=receptor,
                        box=box,
                        scoring=scoring,
                        **(strain_options or {}),
                    )
                    strains[position] = (
                        measured.strain if math.isfinite(measured.strain) else None
                    )
                except Exception as exc:
                    warnings.warn(
                        f"ligand strain failed for pose {position + 1}: {exc}",
                        stacklevel=2,
                    )

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

        affinity = float(getattr(pose, "affinity", 0.0) or 0.0)
        metrics_row = _metrics.efficiency_metrics(
            affinity,
            heavy_atoms=int(heavy) if heavy else None,
            molecular_weight=descriptors_found.get("molecular_weight"),
            logp=descriptors_found.get("logp"),
            tpsa=descriptors_found.get("tpsa"),
            num_torsions=num_torsions,
            strain=strains[position],
        )
        row: Dict[str, Any] = {
            "mode": position + 1,
            "affinity": affinity,
            "rmsd_lb": float(getattr(pose, "rmsd_lower_bound", 0.0) or 0.0),
            "rmsd_ub": float(getattr(pose, "rmsd_upper_bound", 0.0) or 0.0),
            "residues": residues,
            "heavy_atoms": metrics_row.heavy_atoms,
            "ligand_efficiency": _blank(metrics_row.ligand_efficiency),
            "logp": _blank(metrics_row.logp) if metrics_row.logp is not None else None,
            "lle": _blank(metrics_row.lle),
            "bei": _blank(metrics_row.bei),
            "sei": _blank(metrics_row.sei),
            "entropy_penalty": _blank(metrics_row.entropy_penalty),
            "strain": metrics_row.strain,
            # Present but empty unless a consensus was asked for: a stable table
            # shape is worth more to a downstream reader than a shorter header.
            "consensus_rank": None,
            "consensus_score": None,
        }
        index = int(getattr(pose, "index", position))
        if consensus_by_index is not None:
            combined = consensus_by_index.get(index)
            if combined is not None:
                row["consensus_rank"] = combined.consensus_rank
                row["consensus_score"] = combined.consensus_score
                for name, value in combined.scores.items():
                    row[f"consensus_{name}"] = _blank(value)
                    row[f"consensus_{name}_rank"] = combined.ranks.get(name)
            if computed_consensus is not None:
                row["consensus_method"] = computed_consensus.method
        rows.append(row)
    return rows


def _blank(value: Optional[float]) -> Optional[float]:
    """A non-finite metric becomes ``None``, so a table shows a blank not "nan"."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - callers pass floats
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Flat-file writers
# ---------------------------------------------------------------------------


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and str(parent) not in (".", "") and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    """``NaN``/``inf`` -> ``None``; numpy scalars -> Python scalars."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str)):
        return value
    if hasattr(value, "item"):  # numpy scalar
        try:
            return _json_safe(value.item())
        except Exception:  # pragma: no cover - exotic numpy dtype
            return str(value)
    return value


def write_csv(
    path: PathLike,
    result=None,
    *,
    receptor=None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    **kwargs,
) -> str:
    """Write the pose table to `path` as CSV (UTF-8, one header row).

    Pass a `result` (the historical call) or pre-built `rows`.  Returns the path
    that was written, as a string.
    """
    table = list(rows) if rows is not None else result_rows(result, receptor=receptor, **kwargs)
    columns = columns_for(table)
    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writerow({key: _header(key) for key in columns})
        for row in table:
            writer.writerow({key: _cell(row.get(key)) for key in columns})
    return str(target)


def _cell(value: Any) -> Any:
    """A CSV cell: ``None`` and non-finite numbers become the empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_jsonl(
    path: PathLike,
    rows: Iterable[Dict[str, Any]],
    *,
    ensure_ascii: bool = False,
) -> str:
    """Write one JSON object per line, with non-finite numbers as ``null``.

    JSON has no ``NaN``: a naive ``json.dumps`` of a table with one missing
    metric writes ``NaN``, which strict parsers (and most downstream tooling)
    reject.  Every non-finite number is written as ``null`` instead, and the
    keys keep their insertion order.
    """
    target = Path(path)
    _ensure_parent(target)
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            clean = {str(key): _json_safe(value) for key, value in dict(row).items()}
            handle.write(json.dumps(clean, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    return str(target)


def write_rows(
    path: PathLike,
    rows: Iterable[Dict[str, Any]],
    *,
    fmt: Optional[str] = None,
) -> str:
    """Write `rows` as JSONL or CSV, chosen from the suffix (JSONL by default)."""
    target = Path(path)
    suffix = target.suffix.lower()
    kind = fmt
    if kind is None:
        if suffix in _CSV_SUFFIXES:
            kind = "tsv" if suffix == ".tsv" else "csv"
        else:
            kind = "jsonl"
    if kind in ("jsonl", "json", "ndjson"):
        return write_jsonl(target, rows)
    if kind in ("csv", "tsv"):
        delimiter = "\t" if kind == "tsv" else ","
        table = list(rows)
        columns = columns_for(table)
        _ensure_parent(target)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction="ignore", delimiter=delimiter
            )
            writer.writerow({key: _header(key) for key in columns})
            for row in table:
                writer.writerow({key: _cell(row.get(key)) for key in columns})
        return str(target)
    raise ValueError(f"unknown row format {kind!r}; use 'jsonl', 'csv' or 'tsv'")


class ResultStream:
    """An append-only CSV/JSONL writer for very large result sets.

    A screen of a hundred thousand ligands cannot be built as one list of
    dictionaries and then written: the table is bigger than the useful part of
    RAM, and a crash halfway through should still leave a readable file.  This
    class opens the file once, writes the header on the first row, and flushes
    every `flush_every` rows (``1`` by default, so an interrupted run leaves a
    complete-prefix file).

    Use it as a context manager, or call :meth:`close` yourself::

        with ResultStream("screen.jsonl") as stream:
            for ligand in library:
                stream.append(row_for(ligand))
        print(stream.n_written, "rows")

    A CSV stream fixes its columns from the first row; a later row that carries
    an unknown key has that key dropped (a warning is issued once).  JSONL has
    no such constraint -- every row is self-describing.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        fmt: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        flush_every: int = 1,
        ensure_ascii: bool = False,
    ) -> None:
        target = Path(path)
        suffix = target.suffix.lower()
        if fmt is None:
            fmt = "csv" if suffix in _CSV_SUFFIXES else "jsonl"
        self.path = target
        self.fmt = str(fmt).lower()
        if self.fmt not in ("csv", "tsv", "jsonl", "json", "ndjson"):
            raise ValueError(f"unknown stream format {fmt!r}")
        self.columns = list(columns) if columns else []
        self.flush_every = int(flush_every)
        self.ensure_ascii = bool(ensure_ascii)

        self._handle: Any = None
        self._writer: Any = None
        self._columns: List[str] = []
        self._since_flush = 0
        self._written = 0
        self._warned = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "ResultStream":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> "ResultStream":
        """Open (or truncate) the destination file."""
        if self._handle is not None:
            return self
        _ensure_parent(self.path)
        self._handle = self.path.open("w", encoding="utf-8", newline="")
        if self.fmt in ("csv", "tsv"):
            self._writer = csv.DictWriter(
                self._handle,
                fieldnames=[],
                extrasaction="ignore",
                delimiter="\t" if self.fmt == "tsv" else ",",
            )
        return self

    def close(self) -> None:
        """Flush and close; safe to call twice."""
        if self._handle is None:
            return
        try:
            self._handle.flush()
        finally:
            self._handle.close()
            self._handle = None
            self._writer = None

    # -- writing -----------------------------------------------------------

    @property
    def n_written(self) -> int:
        """How many rows have been written so far."""
        return self._written

    def append(self, row: Dict[str, Any]) -> None:
        """Write one row."""
        self.open()
        if self.fmt in ("csv", "tsv"):
            clean = {str(key): _cell(value) for key, value in dict(row).items()}
            if not self._columns:
                self._columns = columns_for([clean], base=self.columns or ())
                self._writer.fieldnames = self._columns
                self._writer.writerow({key: _header(key) for key in self._columns})
            unknown = [key for key in clean if key not in self._columns]
            if unknown and not self._warned:
                warnings.warn(
                    "the stream's columns were fixed by its first row; "
                    f"{len(unknown)} key(s) ({', '.join(unknown[:3])}) will be "
                    "dropped from later rows",
                    stacklevel=2,
                )
                self._warned = True
            self._writer.writerow(clean)
        else:
            clean = {str(key): _json_safe(value) for key, value in dict(row).items()}
            self._handle.write(json.dumps(clean, ensure_ascii=self.ensure_ascii) + "\n")
        self._written += 1
        self._since_flush += 1
        if self.flush_every > 0 and self._since_flush >= self.flush_every:
            self._handle.flush()
            self._since_flush = 0

    def extend(self, rows: Iterable[Dict[str, Any]]) -> int:
        """Write many rows; returns how many were written."""
        count = 0
        for row in rows:
            self.append(row)
            count += 1
        return count


# ---------------------------------------------------------------------------
# Spreadsheet
# ---------------------------------------------------------------------------

_FINGERPRINT_SHEET = "Fingerprints"
_PHARMACOPHORE_SHEET = "Pharmacophore"


def _style_header(sheet, values: Sequence[Any], row: int = 1) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    font = Font(bold=True, color="FFFFFFFF")
    fill = PatternFill("solid", fgColor="FF37474F")
    for column, value in enumerate(values, start=1):
        cell = sheet.cell(row=row, column=column, value=value)
        cell.font = font
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center")


def _write_fingerprint_sheet(workbook, fingerprints, modes: Sequence[str]) -> None:
    """Pose x feature count matrix, plus the bits, on one sheet."""
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet(_FINGERPRINT_SHEET)
    labels = fingerprints.labels()
    _style_header(sheet, ["Mode", "Present"] + labels)
    matrix = fingerprints.matrix
    for row_index, pose in enumerate(fingerprints.fingerprints):
        sheet.cell(row=row_index + 2, column=1, value=modes[row_index])
        sheet.cell(row=row_index + 2, column=2, value=int((matrix[row_index] > 0).sum()))
        for column, value in enumerate(matrix[row_index], start=3):
            cell = sheet.cell(row=row_index + 2, column=column, value=float(value))
            cell.number_format = "0"
    sheet.column_dimensions["A"].width = 8
    sheet.column_dimensions["B"].width = 9
    for column in range(3, len(labels) + 3):
        sheet.column_dimensions[get_column_letter(column)].width = 18
    sheet.freeze_panes = "C2"


def _write_pharmacophore_sheet(workbook, fingerprints, limit: int = 200) -> None:
    """The recurring contacts of the pose set, most frequent first."""
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet(_PHARMACOPHORE_SHEET)
    summary = fingerprints.pharmacophore(min_frequency=0.0)
    _style_header(sheet, ["Feature", "Poses", "Fraction", "Kind", "Residue"])
    for index, (key, count, frequency) in enumerate(
        zip(summary.keys, summary.counts, summary.frequency)
    ):
        if index >= limit:
            break
        sheet.cell(row=index + 2, column=1, value=key.label)
        sheet.cell(row=index + 2, column=2, value=int(count))
        sheet.cell(row=index + 2, column=3, value=float(frequency)).number_format = "0.00"
        sheet.cell(row=index + 2, column=4, value=key.kind)
        sheet.cell(row=index + 2, column=5, value=f"{key.res_name}{key.res_id}")
    for column, width in ((1, 34), (2, 8), (3, 10), (4, 16), (5, 14)):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A2"


def write_xlsx(
    path: PathLike,
    result=None,
    *,
    receptor=None,
    rows: Optional[Iterable[Dict[str, Any]]] = None,
    fingerprints=None,
    fingerprints_top: Optional[int] = None,
    **kwargs,
) -> str:
    """Write the pose table to `path` as an Excel workbook.

    Sheet ``Poses`` holds the table (styled, frozen header, numeric formats) and
    sheet ``Run`` holds the run metadata.  When `fingerprints` is an
    :class:`odock.analysis.FingerprintSet`, two more sheets are added:
    ``Fingerprints`` (the pose x feature count matrix) and ``Pharmacophore``
    (the contacts that recur across the poses -- restricted to the first
    `fingerprints_top` when given, which is how the "reproducible part of the
    top poses" is written down).

    `openpyxl` is required; when it is not installed the call degrades to
    :func:`write_csv` on the same path with a ``.csv`` suffix, emits a warning,
    and returns the CSV path that was actually written.
    """
    table = list(rows) if rows is not None else result_rows(result, receptor=receptor, **kwargs)
    target = Path(path)

    try:
        from openpyxl import Workbook
    except Exception:  # pragma: no cover - openpyxl is bundled, but be safe
        warnings.warn(
            "openpyxl is not available; the report was written as CSV instead",
            stacklevel=2,
        )
        return write_csv(target.with_suffix(".csv"), rows=table)

    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Poses"

    columns = columns_for(table)
    _style_header(sheet, [_header(key) for key in columns])
    numeric = {
        "affinity", "rmsd_lb", "rmsd_ub", "ligand_efficiency", "logp", "lle", "bei",
        "sei", "entropy_penalty", "strain", "consensus_score",
    }
    for row_index, row in enumerate(table, start=2):
        for column_index, key in enumerate(columns, start=1):
            value = row.get(key)
            cell = sheet.cell(row=row_index, column=column_index, value=_cell(value))
            if key in numeric and isinstance(value, (int, float)):
                cell.number_format = "0.000"
            elif isinstance(value, int):
                cell.number_format = "0"
    for column, key in enumerate(columns, start=1):
        width = 10 if key not in ("residues",) else 46
        if key in ("affinity",):
            width = 22
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A2"
    if table:
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(table) + 1}"

    run = workbook.create_sheet("Run")
    run.append(["OpenDocking report"])
    run["A1"].font = Font(bold=True, size=14)
    run.append([])
    metadata = [
        ("poses", len(table)),
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
    best_pose = _best_consensus_pose(table)
    if best_pose is not None:
        metadata.extend(
            [
                ("consensus method", best_pose.get("consensus_method")),
                ("best consensus rank", best_pose.get("consensus_rank")),
                ("best consensus score", best_pose.get("consensus_score")),
            ]
        )
    for label, value in metadata:
        run.append([label, value])
        run.cell(row=run.max_row, column=1).font = Font(bold=True)
    run.column_dimensions["A"].width = 26
    run.column_dimensions["B"].width = 48

    if fingerprints is not None:
        modes = [row["mode"] for row in table] or [
            index + 1 for index in range(len(fingerprints))
        ]
        _write_fingerprint_sheet(workbook, fingerprints, modes)
        if fingerprints_top is not None:
            fingerprints = _top_fingerprints(fingerprints, int(fingerprints_top))
        _write_pharmacophore_sheet(workbook, fingerprints)

    _ensure_parent(target)
    workbook.save(str(target))
    return str(target)


def _top_fingerprints(fingerprints, top: int):
    """A shallow copy of `fingerprints` restricted to its first `top` poses."""
    from dataclasses import replace

    top = max(0, top)
    return replace(
        fingerprints,
        fingerprints=list(fingerprints.fingerprints)[:top],
        interactions=list(fingerprints.interactions)[:top],
        water_bridges=list(fingerprints.water_bridges)[:top],
        ligands=list(fingerprints.ligands)[:top],
    )


def _best_consensus_pose(rows) -> Optional[Dict[str, Any]]:
    ranked = [row for row in rows if row.get("consensus_rank") == 1]
    return ranked[0] if ranked else None


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


# ---------------------------------------------------------------------------
# The notebook summary and the energy decomposition
# ---------------------------------------------------------------------------


def decomposition_table_for(result, *, consensus=None, scorings=None, box=None, receptor=None) -> str:
    """The per-pose energy decomposition as text, for a notebook or a log.

    Uses the components a :class:`odock.consensus.ConsensusResult` already
    carries when one is given; otherwise it rescores with `scorings` (the run's
    own force field by default), which costs one exact evaluation per pose.
    """
    if isinstance(consensus, _consensus.ConsensusResult):
        rows = _consensus.decomposition_rows(consensus)
    else:
        names = tuple(scorings or (getattr(result, "scoring", None) or "vina",))
        raw = _consensus.rescore_poses(
            result,
            receptor if receptor is not None else getattr(result, "receptor_pdbqt", ""),
            box if box is not None else getattr(result, "box", None),
            names,
        )
        rows = _consensus.decomposition_rows(raw)
    return _consensus.decomposition_table(rows)


def notebook_summary(
    result,
    *,
    receptor=None,
    consensus=None,
    box=None,
    scorings: Sequence[str] = _consensus.DEFAULT_SCORINGS,
    fingerprints=None,
    ligand_mol=None,
    descriptors=None,
    rows: Optional[List[Dict[str, Any]]] = None,
    top: int = 3,
    title: str = "OpenDocking run summary",
) -> str:
    """A plain-text summary a user can paste into a lab notebook.

    Every number is one that was actually computed, and every approximation is
    named next to it: the strain's force field, the fact that ligand efficiency
    uses the heavy-atom count of the *prepared* ligand, and the consensus
    agreement (a low Spearman rho is reported as a soft ranking, not smoothed
    over).  Nothing here needs RDKit, Qt or a spreadsheet.

    `ligand_mol` / `descriptors` fill the LogP/LLE/BEI/SEI columns, exactly as
    for :func:`result_rows`; without them the summary says they are blank rather
    than printing a guessed number.  Pass `rows` when the caller has already
    built the table, so the interaction profiling is not repeated.
    """
    poses = list(getattr(result, "poses", None) or [])
    lines: List[str] = [title, "=" * len(title), ""]

    seed = getattr(result, "seed", None)
    scoring = getattr(result, "scoring", None)
    parts = [f"{len(poses)} pose(s)"]
    if scoring:
        parts.append(str(scoring))
    if seed is not None:
        parts.append(f"seed {seed}")
    tors = getattr(result, "num_tors", None)
    if tors is not None:
        parts.append(f"N_tors {float(tors):g}")
    movable = getattr(result, "num_movable_atoms", None)
    if movable:
        parts.append(f"{int(movable)} movable atoms")
    elapsed = getattr(result, "elapsed", None)
    if elapsed:
        parts.append(f"{float(elapsed):.2f} s")
    lines.append("run          : " + ", ".join(parts))
    box = getattr(result, "box", None)
    if box is not None:
        lines.append(
            "box          : center ("
            + ", ".join(f"{float(v):.2f}" for v in box.center)
            + ") size ("
            + ", ".join(f"{float(v):.2f}" for v in box.size)
            + f") spacing {float(box.spacing):g} A"
        )

    rows = (
        rows
        if rows is not None
        else result_rows(
            result,
            receptor=receptor,
            consensus=consensus,
            box=box,
            scorings=scorings,
            ligand_mol=ligand_mol,
            descriptors=descriptors,
        )
    )
    if rows:
        best = rows[0]
        detail = [f"mode {best['mode']}", f"{best['affinity']:.3f} kcal/mol"]
        if best.get("heavy_atoms"):
            detail.append(f"{best['heavy_atoms']:.0f} heavy atoms")
        if best.get("ligand_efficiency") is not None:
            detail.append(f"LE {best['ligand_efficiency']:.3f} kcal/mol/HA")
        if best.get("lle") is not None:
            detail.append(f"LLE {best['lle']:.2f}")
        lines.append("best pose    : " + ", ".join(detail))
        if best.get("residues"):
            lines.append("contacts     : " + str(best["residues"]))
        entropy = best.get("entropy_penalty")
        if entropy is not None and rows and all(
            row.get("entropy_penalty") == entropy for row in rows
        ):
            # The number is the same for every pose of one ligand, so it cannot
            # separate them; saying so prevents "the column looks broken".
            lines.append(
                f"entropy est. : {entropy:.3f} kcal/mol per pose -- the same for "
                "every pose of this ligand"
            )
            lines.append(
                "               (a cross-ligand comparison, not a pose "
                "discriminator; the model is n·R·T·ln 3)"
            )
        lines.append("")

    computed, _ = _consensus_for(result, receptor, consensus, box, scorings)
    if computed is not None:
        ordered = computed.poses
        lines.append(
            f"consensus    : {', '.join(computed.scorings)} "
            f"({computed.method} aggregation)"
        )
        for pose in ordered[: max(1, int(top))]:
            cells = "  ".join(
                f"{name} {pose.scores.get(name, float('nan')):.3f} "
                f"(rank {pose.ranks.get(name, float('nan')):.0f})"
                for name in computed.scorings
            )
            lines.append(
                f"  rank {pose.consensus_rank} = mode {pose.index + 1} "
                f"(score {pose.consensus_score:.3f}): {cells}"
            )
        agreement = computed.agreement
        if math.isfinite(agreement):
            verdict = (
                "the force fields agree about the ordering"
                if agreement >= 0.8
                else "the force fields only partly agree; treat the ranking as soft"
            )
            lines.append(f"  mean Spearman rho = {agreement:+.3f} -- {verdict}")
        missing = [pose.index + 1 for pose in ordered if pose.missing]
        if missing:
            lines.append(
                "  no usable score for mode(s) "
                + ", ".join(str(index) for index in missing)
                + " under some force fields"
            )
        lines.append("")

    if any("strain" in row and row["strain"] is not None for row in rows):
        strains = [row["strain"] for row in rows if row.get("strain") is not None]
        lines.append(
            "strain       : kernel intra(pose) - intra(relaxed), "
            f"{min(strains):+.3f} to {max(strains):+.3f} kcal/mol"
        )
        lines.append(
            "               (the relaxation is a local force-field minimisation "
            "from the pose geometry,"
        )
        lines.append(
            "                so it is a strain relative to the nearest basin, "
            "not a global search;"
        )
        lines.append(
            "                the molecule used is the one perceived from the PDBQT, "
            "so pass a"
        )
        lines.append(
            "                chemistry-complete ligand to odock.metrics for the "
            "MMFF94 figure)"
        )
        lines.append("")

    if fingerprints is not None:
        summary = fingerprints.pharmacophore(
            top=max(1, int(top)) if len(fingerprints) > int(top) else None,
            min_frequency=0.5,
        )
        lines.append(
            f"pharmacophore: recurring contacts among the top {summary.n_poses} "
            f"pose(s)"
        )
        if summary.keys:
            for key, count, frequency in zip(summary.keys, summary.counts, summary.frequency):
                lines.append(
                    f"  {key.label:<28} {count}/{summary.n_poses} poses "
                    f"({frequency:.0%})"
                )
        else:
            lines.append("  none -- the top poses agree on no single contact")
        water = sum(len(bridges) for bridges in fingerprints.water_bridges)
        if water:
            lines.append(f"  {water} water-mediated contact(s) detected")
        lines.append("")

    if poses and any(getattr(pose, "in_box", True) is False for pose in poses):
        lines.append("warning      : at least one pose left the search box")
    lines.append(
        "note         : ligand efficiency uses the prepared ligand's heavy-atom "
        "count."
    )
    if any(row.get("logp") is None for row in rows):
        lines.append(
            "               LogP/LLE/BEI/SEI are blank because no ligand molecule "
            "was supplied;"
        )
        lines.append(
            "               they are not guessed from the PDBQT, which carries no "
            "bond orders."
        )
    if rows and all(row.get("strain") is None for row in rows):
        lines.append(
            "               Strain is blank: it is opt-in (strain=True) because it "
            "costs a"
        )
        lines.append("               force-field minimisation per pose.")
    return "\n".join(lines).rstrip() + "\n"

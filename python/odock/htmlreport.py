# SPDX-License-Identifier: GPL-3.0-or-later
"""A docking run as one self-contained HTML document.

The spreadsheet ``odock.report`` writes is for a spreadsheet.  What a user hands
to a colleague, attaches to a lab note or quotes in a paper is a document: one
file, the figures inside it, and the caveats next to the numbers.

This module writes exactly that, and it makes three promises that are checked by
``tests/test_htmlreport.py`` rather than asserted in prose:

* **self-contained** -- figures are inlined (vector SVG in the document, raster
  previews as ``data:image/png;base64``), there is no stylesheet, script or image
  fetched from anywhere, and :func:`write_html_report` refuses to return a
  document that contains an ``http://``/``https://`` reference;
* **machine-readable too** -- the same numbers are written as a JSON sibling and
  a plain-text sibling (:func:`write_html_report`), so a script does not have to
  parse HTML;
* **honest** -- the report carries an explicit "what this run does not establish"
  section, taken from the run's own manifest when the report is generated from a
  project, and it says which figures it could not draw and why (for instance a
  2-D interaction diagram needs the receptor and the posed ligand).

A PDF is produced **only** when a converter is genuinely available on the
machine (WeasyPrint, or a headless Chrome/Edge, both discovered at run time and
neither a dependency of this package).  When there is none, the command says so
and the HTML is the deliverable -- see :func:`pdf_converter`.
"""

from __future__ import annotations

import base64
import hashlib
import html as _html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import project as _project

__all__ = [
    "HtmlReport",
    "ReportFiles",
    "REPORT_FORMAT",
    "REPORT_VERSION",
    "add_report_html_parser",
    "build_report",
    "build_comparison_report",
    "build_campaign_index",
    "build_study_diff_report",
    "build_study_report",
    "cmd_report_html",
    "find_external_references",
    "pdf_converter",
    "rasterise_svg",
    "study_diff_text",
    "write_campaign_index",
    "write_comparison_report",
    "write_html_report",
    "write_pdf",
    "write_study_diff",
    "write_study_report",
]

PathLike = Union[str, os.PathLike]

REPORT_FORMAT = "odock-html-report"
REPORT_VERSION = 1

#: The order and the headings of the report's sections.  Kept as data so the
#: tests can assert that a section exists without matching prose.
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("summary", "The run at a glance"),
    ("reproducibility", "Reproducibility manifest"),
    ("inputs", "Inputs and their hashes"),
    ("preparation", "Preparation"),
    ("box", "Search box"),
    ("engine", "Engine settings"),
    ("ranking", "Ranking table"),
    ("interactions", "Interaction profile"),
    ("quality", "Pose-quality metrics"),
    ("figures", "Figures"),
    ("limits", "What this run does not establish"),
)

#: Settings whose "not recorded" state means something: the report prints a dash
#: rather than an empty cell, and the JSON keeps ``null``.
_ENGINE_LABELS: Tuple[Tuple[str, str], ...] = (
    ("scoring", "force field"),
    ("seed", "random seed"),
    ("exhaustiveness", "independent Monte-Carlo runs"),
    ("num_poses", "poses requested"),
    ("search", "search protocol"),
    ("islands", "GA islands"),
    ("population", "GA population"),
    ("generations", "GA generations"),
    ("min_rmsd", "minimum RMSD between modes (Å)"),
    ("energy_range", "reporting window (kcal/mol)"),
    ("use_grid", "affinity grid used"),
    ("refine", "exact post-refinement"),
)

_EXTERNAL_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (
        "a URL in a fetched attribute",
        re.compile(
            r"\b(?:src|href|data|action|poster|formaction|srcset|background)\s*=\s*"
            r"[\"']?\s*(?:https?:)?//",
            re.IGNORECASE,
        ),
    ),
    ("a protocol-relative URL in an attribute", re.compile(r"[\"'(]//[A-Za-z0-9]")),
    ("an external stylesheet", re.compile(r"<link\b", re.IGNORECASE)),
    ("an external script", re.compile(r"<script[^>]*\bsrc\s*=", re.IGNORECASE)),
    ("an external frame or object", re.compile(r"<(iframe|object|embed)\b", re.IGNORECASE)),
    ("a CSS url() reference", re.compile(r"url\(\s*(?![\"']?(?:data:|#))", re.IGNORECASE)),
    ("a CSS @import", re.compile(r"@import\b", re.IGNORECASE)),
)

_RASTERISER_WARNING = (
    "no raster preview was produced: install the [gui] extra (PyQt6 ships "
    "QtSvg) or CairoSVG to rasterise the vector figures; the vector figure is "
    "in the document either way"
)

_FONT_WARNING = (
    "the raster preview was skipped because the headless Qt platform this "
    "process is using has no font database, and a figure whose labels render as "
    "boxes is worse than no raster at all; the vector figure in this document "
    "carries the labels (set QT_QPA_FONTDIR, or run outside "
    "QT_QPA_PLATFORM=offscreen, to get the raster too)"
)


# ---------------------------------------------------------------------------
# Escaping and small HTML builders
# ---------------------------------------------------------------------------


def _escape(value: Any) -> str:
    return _html.escape("" if value is None else str(value), quote=True)


def _dash(value: Any) -> str:
    """``None`` and non-finite numbers are a dash, never the word "None"."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        if value in (float("inf"), float("-inf")):
            return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _escape(value)


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    return f"{number:.{digits}f}"


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    table_id: Optional[str] = None,
    numeric: Sequence[int] = (),
) -> str:
    """A plain HTML table; no CSS framework, no classes that need a stylesheet."""
    attributes = f' id="{_escape(table_id)}"' if table_id else ""
    parts = [f"<table{attributes}>", "<thead><tr>"]
    for index, header in enumerate(headers):
        align = ' class="num"' if index in numeric else ""
        parts.append(f"<th{align}>{_escape(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for index, cell in enumerate(row):
            align = ' class="num"' if index in numeric else ""
            parts.append(f"<td{align}>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _definition_list(pairs: Sequence[Tuple[str, Any]]) -> str:
    parts = ["<dl>"]
    for label, value in pairs:
        parts.append(f"<dt>{_escape(label)}</dt><dd>{value}</dd>")
    parts.append("</dl>")
    return "\n".join(parts)


def _bullet_list(items: Iterable[str]) -> str:
    entries = [f"<li>{item}</li>" for item in items]
    return "<ul>\n" + "\n".join(entries) + "\n</ul>" if entries else "<p>none</p>"


def _json_block(payload: Any, element_id: str) -> str:
    """A JSON island inside the document, for a machine reader."""
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    text = text.replace("</", "<\\/")  # cannot close the script element early
    return f'<script type="application/json" id="{_escape(element_id)}">\n{text}\n</script>'


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------


def find_external_references(document: str) -> List[str]:
    """Every reference in `document` that would need the network.

    Returns human-readable descriptions (``"a URL in a fetched attribute: …"``),
    so a failure names what it found instead of only that something was found.

    A URL that appears as *text* (in a title a user typed, say) is not a
    reference: nothing fetches it.  This function therefore looks at attributes
    and CSS, and the tests that want the stronger "the document contains no
    ``http://`` at all" statement make it on documents generated from run data.
    """
    found: List[str] = []
    for label, pattern in _EXTERNAL_PATTERNS:
        match = pattern.search(document)
        if match:
            start = max(0, match.start() - 20)
            found.append(f"{label}: …{document[start:match.end() + 30]}…")
    return found


def _strip_namespaces(svg: str) -> str:
    """An SVG string that can be inlined in HTML without any URL in it.

    An inline ``<svg>`` in an HTML5 document is in the SVG namespace already, so
    the ``xmlns`` attribute is redundant -- and it is the one thing in a typical
    SVG that contains an ``http://`` URL.  Removing it is what makes the
    "no external reference" check a simple, total statement about the document.
    """
    text = svg.strip()
    text = re.sub(r"<\?xml[^>]*\?>", "", text)
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r'\s+xmlns(?::[A-Za-z0-9]+)?="[^"]*"', "", text)
    text = re.sub(r"\s+xmlns(?::[A-Za-z0-9]+)?='[^']*'", "", text)
    return text.strip()


def _with_namespace(svg: str) -> str:
    """The SVG as a standalone document, for a rasteriser that needs one."""
    text = svg.strip()
    if "xmlns=" in text:
        return text
    return text.replace(
        "<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def rasterise_svg(svg: str, *, width: int = 900, height: int = 700) -> Optional[bytes]:
    """Rasterise an SVG to PNG bytes, or ``None`` when it cannot be done here.

    QtSvg (part of the optional ``[gui]`` extra) is used when it is importable.
    It renders into a ``QImage``: no window is created and none is needed.  A
    ``QApplication`` is created on first use if the process has none -- the same
    thing the workbench does -- because Qt's font database is unavailable
    without one, and a figure whose labels are empty boxes is worse than no
    raster.

    ``None`` is returned, rather than a degraded image, when no rasteriser is
    installed, or when the Qt platform in use has no fonts and the figure
    contains text.  Nothing is installed on demand: the caller reports what could
    not be drawn.
    """
    try:
        from PyQt6.QtCore import QByteArray, QBuffer, QIODevice
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtSvg import QSvgRenderer
    except Exception:
        return None
    if _qt_application() is None:
        return None
    if "<text" in svg and not _fonts_available():
        return None
    try:
        renderer = QSvgRenderer(QByteArray(_with_namespace(svg).encode("utf-8")))
        if not renderer.isValid():
            return None
        image = QImage(int(width), int(height), QImage.Format.Format_ARGB32)
        image.fill(0xFFFFFFFF)
        painter = QPainter(image)
        try:
            renderer.render(painter)
        finally:
            painter.end()
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not image.save(buffer, "PNG"):
            return None
        data = bytes(buffer.data())
        return data if data[:8] == b"\x89PNG\r\n\x1a\n" else None
    except Exception:  # pragma: no cover - a broken Qt install must not fail a report
        return None


_QT_STATE: Dict[str, Any] = {"app": None, "tried": False, "fonts": None}


def _browser_profile_directory() -> Path:
    """A profile directory for the headless browser, or a refusal.

    The failure this prevents is specific and measured: `tempfile.mkdtemp` falls back
    to the **working directory** when the environment's temp directory is not
    writable, so a browser profile appears next to the source — and it stays there
    when the browser is still holding a file open at cleanup time.  Four such
    directories (`odock-chrome-*`) reached the published 0.2.1 tree.

    So the parent is chosen deliberately and checked: the environment's temp
    directory, and the directory is refused if it did not land there.  A caller
    without a writable temp directory gets no PDF and a complete HTML report, which
    is the existing contract for "no converter available".
    """
    parent = Path(tempfile.gettempdir())
    try:
        profile = Path(tempfile.mkdtemp(prefix="odock-chrome-", dir=parent))
    except OSError as exc:
        raise PdfUnavailable(
            "no PDF was produced: there is no writable temporary directory for the "
            f"headless browser's profile ({_project.portable_path(parent)}: "
            f"{type(exc).__name__}). The HTML report is complete and self-contained; "
            "print it from a browser instead."
        )
    try:
        inside_working_directory = profile.resolve().is_relative_to(Path.cwd().resolve())
    except (OSError, ValueError):  # pragma: no cover - an unresolvable path
        inside_working_directory = False
    if inside_working_directory:
        # `tempfile` fell back to the working directory: refuse here rather than
        # write a browser profile next to the source.
        shutil.rmtree(profile, ignore_errors=True)
        raise PdfUnavailable(
            "no PDF was produced: the temporary directory resolves inside the working "
            f"directory ({_project.portable_path(profile)}), so the browser's profile "
            "would be created next to the source. Set TMPDIR/TEMP to a writable "
            "directory. The HTML report is complete and self-contained."
        )
    return profile


def _remove_browser_profile(profile: Path) -> Optional[Path]:
    """Remove the profile directory, retrying once; return it if it survived.

    A browser releases its files asynchronously, so the first `rmtree` can fail on
    Windows.  The retry is short and bounded, and a surviving directory is
    **returned rather than ignored**: the leaked 0.2.1 directories are what silence
    looked like.
    """
    for attempt in range(3):
        if not profile.exists():
            return None
        shutil.rmtree(profile, ignore_errors=True)
        if not profile.exists():
            return None
        time.sleep(0.25 * (attempt + 1))
    return profile if profile.exists() else None


def _preferred_platforms() -> List[Optional[str]]:
    """The Qt platform plugins to try, most capable first.

    The default platform is preferred: on a desktop it is the one that can see
    the system fonts, and it creates no window by itself.  On a machine without
    a display server (the usual headless CI) the default plugin would abort the
    process rather than raise, so the offscreen plugin is used directly there.
    """
    if os.name == "nt" or sys.platform == "darwin":
        return [None, "offscreen"]
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return [None, "offscreen"]
    return ["offscreen"]


def _qt_application():
    """A Qt application for offscreen rendering, created once per process."""
    if _QT_STATE["tried"]:
        return _QT_STATE["app"]
    _QT_STATE["tried"] = True
    try:
        from PyQt6.QtWidgets import QApplication as _Factory
    except Exception:
        try:
            from PyQt6.QtGui import QGuiApplication as _Factory  # type: ignore[assignment]
        except Exception:
            return None
    try:
        existing = _Factory.instance()
        if existing is not None:
            _QT_STATE["app"] = existing
            return existing
        for platform_name in _preferred_platforms():
            saved = os.environ.get("QT_QPA_PLATFORM")
            if platform_name is not None:
                os.environ["QT_QPA_PLATFORM"] = platform_name
            try:
                app = _Factory([])
            except Exception:  # pragma: no cover - depends on the Qt install
                app = None
            finally:
                if saved is None:
                    os.environ.pop("QT_QPA_PLATFORM", None)
                else:
                    os.environ["QT_QPA_PLATFORM"] = saved
            if app is not None:
                _QT_STATE["app"] = app
                return app
    except Exception:  # pragma: no cover - defensive
        return None
    return None


def _fonts_available() -> bool:
    """Whether the Qt in this process has a font database (checked once)."""
    if _QT_STATE["fonts"] is not None:
        return bool(_QT_STATE["fonts"])
    try:
        from PyQt6.QtGui import QFontDatabase

        available = bool(QFontDatabase.families())
    except Exception:  # pragma: no cover - defensive
        available = False
    _QT_STATE["fonts"] = available
    return available


def _bar_chart_svg(
    affinities: Sequence[float],
    *,
    width: int = 760,
    height: int = 260,
    highlight: int = 0,
) -> str:
    """A hand-written SVG bar chart of the pose affinities (no plotting library)."""
    if not affinities:
        return ""
    lowest = min(affinities)
    highest = max(affinities)
    span = max(highest - lowest, 0.5)
    left, right, top, bottom = 60.0, width - 20.0, 24.0, height - 40.0
    slot = (right - left) / len(affinities)
    bar_width = max(3.0, min(38.0, slot * 0.62))

    def y_of(value: float) -> float:
        # Affinity is negative and lower is better: the best pose is the tallest bar.
        return bottom - (0.85 + (value - lowest) / span * 0.85) * (bottom - top) / 1.7

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="img" aria-label="affinity of each pose">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#263238" stroke-width="1"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#263238" stroke-width="1"/>',
    ]
    for tick in range(4):
        value = lowest + span * tick / 3.0
        y = y_of(value)
        parts.append(
            f'<line x1="{left - 4:.1f}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" '
            'stroke="#90a4ae" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="11" fill="#37474f">'
            f"{value:.1f}</text>"
        )
    for index, value in enumerate(affinities):
        x = left + index * slot + (slot - bar_width) / 2.0
        y = y_of(value)
        colour = "#1565c0" if index == highlight else "#90a4ae"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bottom - y:.1f}" fill="{colour}"><title>mode {index + 1}: '
            f"{value:.3f} kcal/mol</title></rect>"
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
            f'font-family="Helvetica,Arial,sans-serif" font-size="10" fill="#37474f">'
            f"{index + 1}</text>"
        )
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="{height - 10:.1f}" text-anchor="middle" '
        'font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#37474f">'
        "pose (mode) — bar height is the reported affinity, lower is better</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _ligand_depiction_svg(ligand_mol, *, width: int = 420, height: int = 320) -> Optional[str]:
    """A 2-D depiction of the ligand, when RDKit and a real molecule are available."""
    if ligand_mol is None:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception:  # pragma: no cover - RDKit is the optional chemistry extra
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, ligand_mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:  # pragma: no cover - a drawing failure must not fail a report
        return None


@dataclass
class _Figure:
    figure_id: str
    caption: str
    svg: str = ""
    png: Optional[bytes] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.figure_id,
            "caption": self.caption,
            "kind": "svg" + ("+png" if self.png else ""),
            "svg_chars": len(self.svg),
            "png_bytes": len(self.png) if self.png else 0,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The report itself
# ---------------------------------------------------------------------------


@dataclass
class HtmlReport:
    """A built report, before anything is written."""

    title: str
    html: str
    payload: Dict[str, Any]
    text: str
    figures: List[_Figure] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_images(self) -> int:
        """How many base64 raster images the document embeds."""
        return len(re.findall(r'src="data:image/png;base64,', self.html))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"HtmlReport({self.title!r}, {len(self.html)} chars, {len(self.figures)} figures)"


@dataclass
class ReportFiles:
    """What :func:`write_html_report` wrote."""

    path: Path
    json_path: Optional[Path] = None
    text_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    n_figures: int = 0
    n_images: int = 0
    size_bytes: int = 0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "html": str(self.path),
            "json": str(self.json_path) if self.json_path else None,
            "text": str(self.text_path) if self.text_path else None,
            "pdf": str(self.pdf_path) if self.pdf_path else None,
            "n_figures": int(self.n_figures),
            "n_images": int(self.n_images),
            "size_bytes": int(self.size_bytes),
            "notes": list(self.notes),
        }


def _input_entry(role: str, text: str, source: Any = "") -> Dict[str, Any]:
    """One input row for the manifest table: name, size and SHA-256."""
    data = text.encode("utf-8")
    name = Path(str(source)).name if source else f"{role.split(':')[-1]}.pdbqt"
    return {
        "role": role,
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "media_type": _project.media_type_for(name),
        "source": (
            _project.portable_path(source)
            if isinstance(source, (str, os.PathLike)) and str(source)
            else ""
        ),
    }


def _resolve_inputs(
    result: Any,
    *,
    project: Any,
    receptor: Any,
    ligand: Any,
    ligand_mol: Any,
    box: Any,
    rows: Optional[Sequence[Mapping[str, Any]]],
    interaction_profile: Optional[Mapping[str, Any]],
    preparation: Optional[Mapping[str, Any]],
    engine: Optional[Mapping[str, Any]],
    inputs: Optional[Sequence[Mapping[str, Any]]],
    seed: Optional[int],
    scoring: Optional[str],
    does_not_establish: Optional[Sequence[str]],
    notes: List[str],
) -> Dict[str, Any]:
    """Everything the sections below need, from a project or from loose objects."""
    resolved: Dict[str, Any] = {
        "project": None,
        "result": None,
        "receptor_text": "",
        "ligand_text": "",
        "ligand_mol": ligand_mol,
        "box": None,
        "rows": list(rows) if rows is not None else [],
        "interaction_profile": dict(interaction_profile or {}),
        "preparation": dict(preparation or {}),
        "engine": dict(engine or {}),
        "inputs": [dict(item) for item in (inputs or [])],
        "does_not_establish": list(does_not_establish or ()),
        "reproducibility": {},
        "campaign": {},
        "manifest": {},
        "stored_figures": [],
        "created": "",
        "warnings": [],
        "analysis": {},
        "seed": None,
        "scoring": None,
    }

    if project is not None:
        loaded = project if isinstance(project, _project.Project) else _project.open_project(project)
        resolved["project"] = loaded
        resolved["manifest"] = dict(loaded.manifest)
        resolved["result"] = loaded.result()
        resolved["receptor_text"] = loaded.receptor_text()
        resolved["ligand_text"] = loaded.ligand_text()
        resolved["box"] = loaded.box()
        resolved["engine"] = {**loaded.engine, **resolved["engine"]}
        resolved["preparation"] = {**loaded.preparation, **resolved["preparation"]}
        resolved["inputs"] = loaded.inputs or resolved["inputs"]
        resolved["reproducibility"] = loaded.reproducibility
        resolved["campaign"] = loaded.campaign
        resolved["created"] = loaded.created
        resolved["warnings"] = loaded.warnings
        resolved["does_not_establish"] = loaded.does_not_establish
        analysis = loaded.analysis()
        resolved["analysis"] = analysis
        if rows is None:
            resolved["rows"] = list(analysis.get("rows") or [])
        if interaction_profile is None:
            resolved["interaction_profile"] = dict(analysis.get("interaction_profile") or {})
        resolved["pose_quality"] = dict(analysis.get("pose_quality") or {})
        resolved["pharmacophore"] = dict(analysis.get("pharmacophore") or {})
        resolved["analysis_flags"] = list(analysis.get("flags") or [])
        resolved["seed"] = seed if seed is not None else loaded.engine.get("seed")
        resolved["scoring"] = scoring or loaded.engine.get("scoring")
        for entry in loaded.entries:
            if entry.role == _project.ROLE_FIGURE:
                resolved["stored_figures"].append((entry.label, loaded.read(entry.arcname)))
        return resolved

    if result is None:
        raise ValueError(
            "no run to report: pass a DockResult, a pose PDBQT, a dock JSON "
            "mapping, or project=<file>"
        )
    from .project import result_from_dock_json, result_from_pdbqt

    def _from_dock_json(payload: Mapping[str, Any]) -> Any:
        notes.append(
            "the run came from a dock JSON document, which carries the pose "
            "scores but not the coordinates: the ranking table is complete and "
            "the figures that need coordinates are omitted"
        )
        return result_from_dock_json(payload, box=box, receptor=receptor, ligand=ligand)

    if isinstance(result, Mapping):
        loaded_result = _from_dock_json(result)
    elif isinstance(result, (str, os.PathLike)):
        text = os.fspath(result)
        looks_like_text = "\n" in text or text.lstrip()[:6].upper() in (
            "MODEL ", "REMARK", "ATOM  ", "HETATM", "ROOT",
        )
        if looks_like_text:
            loaded_result = result_from_pdbqt(
                text, box=box, receptor=receptor, ligand=ligand
            )
        else:
            source = Path(text)
            if source.suffix.lower() == ".json":
                loaded_result = _from_dock_json(
                    json.loads(source.read_text(encoding="utf-8"))
                )
            else:
                loaded_result = result_from_pdbqt(
                    source, box=box, receptor=receptor, ligand=ligand
                )
    else:
        loaded_result = result

    resolved["result"] = loaded_result
    if box is not None:
        resolved["box"] = _project._coerce_box(box)
    else:
        candidate = getattr(loaded_result, "box", None)
        resolved["box"] = _project._coerce_box(candidate) if candidate is not None else None
    if ligand is not None:
        ligand_bytes, ligand_name = _project._as_text_or_bytes(ligand, what="ligand")
        resolved["ligand_text"] = ligand_bytes.decode("utf-8", "replace")
        if not any(item.get("role") == _project.ROLE_LIGAND for item in resolved["inputs"]):
            resolved["inputs"].append(
                _input_entry(
                    _project.ROLE_LIGAND,
                    resolved["ligand_text"],
                    ligand if isinstance(ligand, (str, os.PathLike)) else ligand_name,
                )
            )
    else:
        resolved["ligand_text"] = getattr(loaded_result, "ligand_pdbqt", "") or ""
    if receptor is not None:
        receptor_bytes, receptor_name = _project._as_text_or_bytes(receptor, what="receptor")
        resolved["receptor_text"] = receptor_bytes.decode("utf-8", "replace")
        if not any(item.get("role") == _project.ROLE_RECEPTOR for item in resolved["inputs"]):
            resolved["inputs"].insert(
                0,
                _input_entry(
                    _project.ROLE_RECEPTOR,
                    resolved["receptor_text"],
                    receptor if isinstance(receptor, (str, os.PathLike)) else receptor_name,
                ),
            )
    else:
        resolved["receptor_text"] = getattr(loaded_result, "receptor_pdbqt", "") or ""

    resolved["seed"] = seed if seed is not None else getattr(loaded_result, "seed", None)
    if seed is None and not getattr(loaded_result, "seed_recorded", True):
        resolved["seed"] = None
    resolved["scoring"] = scoring or getattr(loaded_result, "scoring", None)

    # The analysis is the *project's* analysis, so a report of a run and a report
    # of the project that wrapped it carry the same numbers, computed once.
    if rows is None:
        computed = _project.analyse_result(
            loaded_result,
            receptor=resolved["receptor_text"] or None,
            ligand=resolved["ligand_text"] or None,
        )
        resolved["rows"] = list(computed.get("rows") or [])
        if interaction_profile is None:
            resolved["interaction_profile"] = dict(computed.get("interaction_profile") or {})
        resolved["pose_quality"] = dict(computed.get("pose_quality") or {})
        resolved["pharmacophore"] = dict(computed.get("pharmacophore") or {})
        resolved["analysis_flags"] = list(computed.get("flags") or [])
    else:
        resolved["rows"] = list(rows)
        resolved["pose_quality"] = _project._pose_quality(loaded_result, resolved["rows"])
        if interaction_profile is None and resolved["receptor_text"].strip():
            resolved["interaction_profile"] = _profile_from_texts(
                resolved["receptor_text"], loaded_result
            )
    if interaction_profile is not None:
        resolved["interaction_profile"] = dict(interaction_profile)
    resolved.setdefault("pharmacophore", {})
    resolved.setdefault("analysis_flags", [])
    return resolved


def _profile_from_texts(receptor_text: str, result: Any) -> Dict[str, Any]:
    """The best pose's interaction profile, computed from stored/loose structures."""
    from . import analysis as _analysis

    receptor_atoms = _report_atoms(receptor_text)
    pose_atoms = _analysis_ligand_atoms(result)
    if not receptor_atoms or not pose_atoms:
        return {}
    try:
        interactions = _analysis.profile_interactions(receptor_atoms, pose_atoms)
    except Exception:  # pragma: no cover - defensive
        return {}
    return {
        "mode": 1,
        "counts": _count_kinds(interactions),
        "key_residues": _analysis.interaction_summary(interactions, receptor_atoms, pose_atoms),
        "rows": [
            {
                "kind": getattr(item, "kind", "?"),
                "receptor_label": _label(receptor_atoms, getattr(item, "a", -1)),
                "ligand_label": _label(pose_atoms, getattr(item, "b", -1)),
                "distance": float(getattr(item, "distance", float("nan"))),
                "detail": getattr(item, "detail", ""),
            }
            for item in interactions
        ],
    }


def _report_atoms(text: str):
    from .report import parse_pdbqt_atoms

    return parse_pdbqt_atoms(text)


def _analysis_ligand_atoms(result: Any):
    from .report import _ligand_atoms

    poses = list(getattr(result, "poses", None) or [])
    if not poses:
        return []
    try:
        return _ligand_atoms(result, poses[0])
    except Exception:  # pragma: no cover - defensive
        return []


def _count_kinds(interactions: Iterable[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in interactions:
        kind = str(getattr(item, "kind", "?"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _label(atoms: Sequence[Any], index: int) -> str:
    if not 0 <= int(index) < len(atoms):
        return f"atom {index}"
    atom = atoms[int(index)]
    return f"{atom.res_name}{atom.res_id}:{atom.name}"


def build_report(
    result: Any = None,
    *,
    project: Any = None,
    receptor: Any = None,
    ligand: Any = None,
    ligand_mol: Any = None,
    box: Any = None,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    interaction_profile: Optional[Mapping[str, Any]] = None,
    preparation: Optional[Mapping[str, Any]] = None,
    engine: Optional[Mapping[str, Any]] = None,
    inputs: Optional[Sequence[Mapping[str, Any]]] = None,
    does_not_establish: Optional[Sequence[str]] = None,
    command: Optional[Union[str, Sequence[str]]] = None,
    title: Optional[str] = None,
    created: Optional[str] = None,
    seed: Optional[int] = None,
    scoring: Optional[str] = None,
    extra_figures: Optional[Mapping[str, Any]] = None,
    rasterise: bool = True,
    include_diagram: bool = True,
    notes: Optional[Sequence[str]] = None,
) -> HtmlReport:
    """Build the report in memory.

    See :func:`write_html_report` for the parameters; this function is the one to
    use when the document is not going to a file (a notebook, an HTTP response,
    a test).
    """
    collected: List[str] = [str(note) for note in (notes or ())]
    data = _resolve_inputs(
        result,
        project=project,
        receptor=receptor,
        ligand=ligand,
        ligand_mol=ligand_mol,
        box=box,
        rows=rows,
        interaction_profile=interaction_profile,
        preparation=preparation,
        engine=engine,
        inputs=inputs,
        seed=seed,
        scoring=scoring,
        does_not_establish=does_not_establish,
        notes=collected,
    )

    run_result = data["result"]
    poses = list(getattr(run_result, "poses", None) or [])
    stored = data["project"]
    report_title = str(
        title
        or (stored.title if stored is not None else "")
        or (getattr(run_result, "ligand_name", "") or "")
        or "OpenDocking run report"
    )
    if report_title == "OpenDocking run report":
        report_title = "OpenDocking run report"

    engine_block = dict(data["engine"])
    engine_block.setdefault("scoring", data["scoring"] or getattr(run_result, "scoring", None))
    engine_block.setdefault("seed", data["seed"])
    engine_block.setdefault("num_poses", len(poses))
    tool_version = _tool_version()
    kernel_version = _kernel_version()
    created_at = str(created or data["created"] or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    if command is not None:
        command_text = _project.portable_command(command)
    else:
        command_text = str(data["reproducibility"].get("command") or "")
    input_hashes = dict(data["reproducibility"].get("input_sha256") or {})
    if not input_hashes:
        for item in data["inputs"]:
            if item.get("sha256"):
                input_hashes[str(item.get("role"))] = item["sha256"]

    reproducibility = {
        "tool": "opendocking",
        "tool_version": tool_version,
        "kernel_version": kernel_version,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "created_utc": created_at,
        "command": command_text,
        "seed": engine_block.get("seed"),
        "scoring": engine_block.get("scoring"),
        "input_sha256": input_hashes,
        "schema_version": stored.schema_version if stored is not None else _project.SCHEMA_VERSION,
        "project_sha256": (
            _sha256_file(stored.path) if stored is not None and stored.path.exists() else None
        ),
        "note": (
            "The same inputs (by SHA-256), the same tool version and the same seed "
            "reproduce the same poses; a different kernel version may legitimately "
            "produce different ones."
        ),
    }

    # -- figures -----------------------------------------------------------
    figures: List[_Figure] = []
    affinities = [float(getattr(pose, "affinity", 0.0) or 0.0) for pose in poses]
    if affinities:
        chart = _bar_chart_svg(affinities)
        figures.append(
            _Figure(
                "ranking",
                "Reported affinity of each pose; the best-scoring pose is highlighted.",
                svg=chart,
            )
        )
    diagram_svg = ""
    if include_diagram and data["receptor_text"].strip() and poses:
        diagram_svg = _interaction_diagram(
            data["receptor_text"], run_result, data["interaction_profile"], report_title
        )
        if diagram_svg:
            figures.append(
                _Figure(
                    "interaction-diagram",
                    "2-D interaction diagram of the best pose, in the plane that "
                    "shows the most of the ligand.",
                    svg=diagram_svg,
                )
            )
        else:
            collected.append(
                "the 2-D interaction diagram could not be drawn (it needs the "
                "receptor and the posed ligand); the contact table is below"
            )
    depiction = _ligand_depiction_svg(ligand_mol)
    if depiction:
        figures.append(
            _Figure("ligand", "The ligand molecule (2-D depiction).", svg=depiction)
        )
    for label, payload in data["stored_figures"]:
        if payload[:8] == b"\x89PNG\r\n\x1a\n":
            figures.append(_Figure(label, f"Figure stored with the project ({label}).", png=payload))
        else:
            figures.append(
                _Figure(
                    label,
                    f"Figure stored with the project ({label}).",
                    svg=payload.decode("utf-8", "replace"),
                )
            )
    for label, payload in (extra_figures or {}).items():
        data_payload = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        if data_payload[:8] == b"\x89PNG\r\n\x1a\n":
            figures.append(_Figure(str(label), f"Figure {label}.", png=data_payload))
        else:
            figures.append(
                _Figure(str(label), f"Figure {label}.", svg=data_payload.decode("utf-8", "replace"))
            )

    fonts_missing = False
    if rasterise:
        for figure in figures:
            if not figure.svg:
                continue
            png = rasterise_svg(figure.svg)
            if png:
                figure.png = png
            elif "<text" in figure.svg and _fonts_available() is False:
                fonts_missing = True
        if not any(figure.png for figure in figures) and any(f.svg for f in figures):
            collected.append(_FONT_WARNING if fonts_missing else _RASTERISER_WARNING)
    else:
        collected.append("raster previews were disabled (rasterise=False)")

    # -- sections ----------------------------------------------------------
    quality = dict(data.get("pose_quality") or {})
    profile = dict(data.get("interaction_profile") or {})
    limits = list(data["does_not_establish"]) or list(_project.RUN_LIMITS)
    if data.get("analysis_flags"):
        limits = limits + [f"this run: {flag}" for flag in data["analysis_flags"]]
    for warning in data["warnings"]:
        limits = limits + [f"this project: {warning}"]

    body: List[str] = []
    body.append(
        f'<header><h1>{_escape(report_title)}</h1>'
        f'<p class="subtitle">OpenDocking report — one self-contained HTML file; '
        f"no network access is needed to read it. Generated {_escape(created_at)}.</p></header>"
    )

    # 1. summary
    best_row = data["rows"][0] if data["rows"] else {}
    summary_pairs: List[Tuple[str, Any]] = [
        ("poses reported", _number(len(poses), 0)),
        ("best affinity (kcal/mol)", _number(getattr(run_result, "best_affinity", None))),
        ("force field", _escape(engine_block.get("scoring"))),
        ("seed", _escape(engine_block.get("seed"))),
        ("heavy atoms", _number(best_row.get("heavy_atoms"), 0)),
        ("ligand efficiency (kcal/mol/HA)", _number(best_row.get("ligand_efficiency"))),
        ("key residues of the best pose", _escape(best_row.get("residues") or "—")),
    ]
    body.append(_section("summary", dict(SECTIONS)["summary"], _definition_list(summary_pairs)))

    # 2. reproducibility
    repro_pairs = [
        ("tool", f"opendocking {_escape(tool_version)}"),
        ("kernel", _escape(kernel_version)),
        ("python", _escape(reproducibility["python"])),
        ("platform", _escape(reproducibility["platform"])),
        ("created (UTC)", _escape(created_at)),
        ("scoring", _escape(reproducibility["scoring"])),
        ("seed", _escape(reproducibility["seed"])),
        ("command", f"<code>{_escape(command_text or 'not recorded')}</code>"),
        ("schema version", _escape(reproducibility["schema_version"])),
        (
            "project SHA-256",
            f"<code>{_escape(reproducibility['project_sha256'] or '—')}</code>",
        ),
        ("note", _escape(reproducibility["note"])),
    ]
    body.append(
        _section(
            "reproducibility",
            dict(SECTIONS)["reproducibility"],
            _definition_list(repro_pairs)
            + _json_block(reproducibility, "odock-reproducibility"),
        )
    )

    # 3. inputs
    if data["inputs"]:
        rows_html = [
            [
                _escape(item.get("role")),
                _escape(item.get("name")),
                _number(item.get("size"), 0),
                f"<code>{_escape(str(item.get('sha256') or '')[:32])}…</code>",
            ]
            for item in data["inputs"]
        ]
        body.append(
            _section(
                "inputs",
                dict(SECTIONS)["inputs"],
                _table(["role", "file", "bytes", "SHA-256"], rows_html, table_id="inputs",
                       numeric=(2,)),
            )
        )
    else:
        body.append(
            _section(
                "inputs",
                dict(SECTIONS)["inputs"],
                "<p>No input file was recorded with this report; the numbers below "
                "came from the run object alone.</p>",
            )
        )

    # 4. preparation
    preparation = data["preparation"]
    if preparation:
        rows_html = []
        for role, block in preparation.items():
            if not isinstance(block, Mapping):
                continue
            rows_html.append(
                [
                    _escape(role),
                    _escape(block.get("kind")),
                    _number(block.get("n_atoms_in"), 0),
                    _number(block.get("n_atoms_out"), 0),
                    _number(block.get("n_hydrogens_added"), 0),
                    _number(block.get("n_rotatable_bonds"), 0),
                    _number(block.get("thickness"), 2),
                    _dash(block.get("chemistry_trusted")),
                ]
            )
        body.append(
            _section(
                "preparation",
                dict(SECTIONS)["preparation"],
                _table(
                    ["role", "kind", "atoms in", "atoms out", "H added", "rotors",
                     "3-D extent (Å)", "bond orders trusted"],
                    rows_html,
                    table_id="preparation",
                    numeric=(2, 3, 4, 5, 6),
                )
                if rows_html
                else "<p>No preparation report was recorded.</p>",
            )
        )
    else:
        body.append(
            _section(
                "preparation",
                dict(SECTIONS)["preparation"],
                "<p>No preparation report was recorded, so this report does not "
                "describe how the inputs were made.</p>",
            )
        )

    # 5. box
    box = data["box"]
    if box is not None:
        box_pairs = [
            ("centre (Å)", _escape(", ".join(f"{float(v):.3f}" for v in box.center))),
            ("size (Å)", _escape(", ".join(f"{float(v):.3f}" for v in box.size))),
            ("spacing (Å)", _escape(f"{float(box.spacing):g}")),
        ]
        body.append(_section("box", dict(SECTIONS)["box"], _definition_list(box_pairs)))
    else:
        body.append(
            _section(
                "box",
                dict(SECTIONS)["box"],
                "<p>No search box was recorded: the report cannot say where the "
                "search was allowed to look.</p>",
            )
        )

    # 6. engine settings
    engine_pairs = []
    for key, label in _ENGINE_LABELS:
        value = engine_block.get(key)
        rendered = "not recorded" if value is None else _dash(value)
        engine_pairs.append((label, rendered))
    body.append(
        _section(
            "engine",
            dict(SECTIONS)["engine"],
            _definition_list(engine_pairs)
            + _json_block(engine_block, "odock-engine"),
        )
    )

    # 7. ranking table
    if data["rows"]:
        ranking_rows = []
        for row in data["rows"]:
            ranking_rows.append(
                [
                    _number(row.get("mode"), 0),
                    _number(row.get("affinity")),
                    _number(row.get("rmsd_lb")),
                    _number(row.get("rmsd_ub")),
                    _number(row.get("heavy_atoms"), 0),
                    _number(row.get("ligand_efficiency")),
                    _escape(row.get("residues") or ""),
                ]
            )
        body.append(
            _section(
                "ranking",
                dict(SECTIONS)["ranking"],
                _table(
                    ["mode", "affinity (kcal/mol)", "RMSD l.b. (Å)", "RMSD u.b. (Å)",
                     "heavy atoms", "LE (kcal/mol/HA)", "key interacting residues"],
                    ranking_rows,
                    table_id="ranking",
                    numeric=(0, 1, 2, 3, 4, 5),
                ),
            )
        )
    else:
        body.append(
            _section(
                "ranking",
                dict(SECTIONS)["ranking"],
                "<p>No pose table could be computed for this run.</p>",
            )
        )

    # 8. interaction profile
    interaction_parts: List[str] = []
    if profile.get("key_residues"):
        interaction_parts.append(
            f"<p><strong>Key residues of the best pose:</strong> "
            f"{_escape(profile.get('key_residues'))}</p>"
        )
    counts = profile.get("counts") or {}
    if counts:
        interaction_parts.append(
            _table(
                ["interaction", "count"],
                [[_escape(kind), _number(count, 0)] for kind, count in sorted(counts.items())],
                table_id="interaction-counts",
                numeric=(1,),
            )
        )
    if profile.get("rows"):
        interaction_parts.append(
            _table(
                ["kind", "receptor atom", "ligand atom", "d (Å)", "detail"],
                [
                    [
                        _escape(item.get("kind")),
                        _escape(item.get("receptor_label")),
                        _escape(item.get("ligand_label")),
                        _number(item.get("distance"), 2),
                        _escape(item.get("detail")),
                    ]
                    for item in profile["rows"]
                ],
                table_id="interactions",
                numeric=(3,),
            )
        )
    if not interaction_parts:
        interaction_parts.append(
            "<p>No interaction profile is available: it needs both the receptor "
            "and the posed ligand, and one of them was not available.</p>"
        )
    if data.get("pharmacophore") and data["pharmacophore"].get("keys"):
        pharmacophore_rows = [
            [
                _escape(key.get("label")),
                _escape(key.get("kind")),
                _number(count, 0),
                f"{_number(frequency * 100, 0)}%",
            ]
            for key, count, frequency in zip(
                data["pharmacophore"]["keys"],
                data["pharmacophore"].get("counts") or [],
                data["pharmacophore"].get("frequency") or [],
            )
        ]
        interaction_parts.append(
            "<h3>Contacts recurring across the top poses</h3>"
            + _table(
                ["feature", "kind", "poses", "frequency"],
                pharmacophore_rows,
                table_id="pharmacophore",
                numeric=(2,),
            )
        )
    body.append(
        _section(
            "interactions", dict(SECTIONS)["interactions"], "\n".join(interaction_parts)
        )
    )

    # 9. pose-quality metrics
    quality_pairs = [
        ("poses reported", _number(quality.get("n_poses"), 0)),
        ("best affinity (kcal/mol)", _number(quality.get("best_affinity"))),
        ("worst reported affinity (kcal/mol)", _number(quality.get("worst_affinity"))),
        ("spread of the reported poses (kcal/mol)", _number(quality.get("spread"))),
        ("poses within 1 kcal/mol of the best", _number(quality.get("n_within_1_kcal"), 0)),
        ("heavy atoms", _number(quality.get("heavy_atoms"), 0)),
        ("ligand efficiency of the best pose", _number(quality.get("ligand_efficiency"))),
    ]
    quality_html = _definition_list(quality_pairs)
    flags = list(quality.get("flags") or [])
    quality_html += "<h3>Flags</h3>" + (
        _bullet_list([_escape(flag) for flag in flags])
        if flags
        else "<p>No pose-quality flag was raised for this run.</p>"
    )
    if data["rows"] and any(row.get("strain") is not None for row in data["rows"]):
        quality_html += (
            "<p>Ligand strain is reported per pose in the analysis (it is the "
            "kernel's intra(pose) minus intra(relaxed), one local minimisation per "
            "pose, not a global search).</p>"
        )
    body.append(
        _section("quality", dict(SECTIONS)["quality"], quality_html)
    )

    # 10. figures
    figure_html: List[str] = []
    for figure in figures:
        parts = [f'<figure id="figure-{_escape(figure.figure_id)}">']
        if figure.png:
            encoded = base64.b64encode(figure.png).decode("ascii")
            parts.append(
                f'<img alt="{_escape(figure.caption)}" '
                f'src="data:image/png;base64,{encoded}"/>'
            )
        if figure.svg:
            parts.append(f'<div class="vector">{_strip_namespaces(figure.svg)}</div>')
        parts.append(f"<figcaption>{_escape(figure.caption)}</figcaption></figure>")
        figure_html.append("\n".join(parts))
    if not figure_html:
        figure_html.append("<p>No figure could be drawn for this run.</p>")
    body.append(_section("figures", dict(SECTIONS)["figures"], "\n".join(figure_html)))

    # 11. limits
    body.append(
        _section(
            "limits",
            dict(SECTIONS)["limits"],
            _bullet_list([_escape(item) for item in limits]),
        )
    )

    notes_block = ""
    if collected:
        notes_block = "<section id=\"notes\"><h2>Notes on this report</h2>" + _bullet_list(
            [_escape(note) for note in collected]
        ) + "</section>"

    document = _document(report_title, "\n".join(body) + notes_block)
    external = find_external_references(document)
    if external:
        # A self-containment failure is a bug in this module, not a user error:
        # fail loudly rather than ship a document that needs the network.
        raise RuntimeError(
            "the report would not be self-contained: " + "; ".join(external)
        )

    payload = {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "created_utc": created_at,
        "title": report_title,
        "tool": {
            "name": "opendocking",
            "version": tool_version,
            "kernel_version": kernel_version,
        },
        "reproducibility": reproducibility,
        "inputs": data["inputs"],
        "preparation": data["preparation"],
        "box": box.as_dict() if box is not None else None,
        "engine": engine_block,
        "ranking": data["rows"],
        "interaction_profile": profile,
        "pharmacophore": data.get("pharmacophore") or {},
        "pose_quality": quality,
        "figures": [figure.as_dict() for figure in figures],
        "does_not_establish": limits,
        "notes": collected,
        "project": (
            {
                "name": stored.path.name,
                "schema_version": stored.schema_version,
                "n_entries": len(stored.entries),
            }
            if stored is not None
            else None
        ),
    }

    text = _plain_text(
        report_title,
        created_at,
        run_result,
        data,
        engine_block,
        quality,
        limits,
        collected,
        reproducibility,
    )
    return HtmlReport(
        title=report_title,
        html=document,
        payload=payload,
        text=text,
        figures=figures,
        notes=collected,
    )


def _section(anchor: str, heading: str, content: str) -> str:
    return f'<section id="{_escape(anchor)}">\n<h2>{_escape(heading)}</h2>\n{content}\n</section>'


#: The report stylesheet, as one string.
#:
#: Exposed rather than buried in :func:`_document` so the documentation site can
#: use the *same* conventions instead of inventing a second look (it imports this
#: and adds only the layout rules a multi-page site needs: sidebar, search box).
#: `tests/test_htmlreport.py` pins the report output byte-for-byte, so this is the
#: one definition of both.
DOCUMENT_CSS = """
:root { color-scheme: light; }
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
        margin: 0 auto; max-width: 62rem; padding: 1.5rem; color: #212121;
        line-height: 1.45; background: #ffffff; }
h1 { font-size: 1.6rem; margin-bottom: 0.2rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #cfd8dc;
      padding-bottom: 0.25rem; }
h3 { font-size: 1rem; margin-top: 1.2rem; }
p.subtitle { color: #546e7a; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 0.6rem 0; font-size: 0.92rem; }
th, td { border: 1px solid #cfd8dc; padding: 0.32rem 0.5rem; text-align: left;
          vertical-align: top; }
th { background: #eceff1; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
dl { display: grid; grid-template-columns: minmax(12rem, max-content) 1fr;
      gap: 0.15rem 1rem; margin: 0.6rem 0; }
dt { font-weight: 600; }
dd { margin: 0; }
code { background: #f5f5f5; padding: 0.05rem 0.25rem; word-break: break-all; }
figure { margin: 1rem 0; }
figure img, figure svg { max-width: 100%; height: auto;
                          border: 1px solid #eceff1; background: #fff; }
figcaption { color: #546e7a; font-size: 0.88rem; margin-top: 0.3rem; }
ul { margin: 0.4rem 0 0.4rem 1.2rem; padding: 0; }
section#limits { background: #fff8e1; padding: 0.5rem 1rem 1rem 1rem;
                  border: 1px solid #ffe082; border-radius: 4px; }
footer { margin-top: 2rem; color: #546e7a; font-size: 0.85rem;
          border-top: 1px solid #cfd8dc; padding-top: 0.6rem; }
"""


def _document(title: str, body: str) -> str:
    """The whole document: one inline stylesheet, no external resource."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="generator" content="opendocking"/>
<title>{_escape(title)}</title>
<style>{DOCUMENT_CSS}</style>
</head>
<body>
{body}
<footer>
<p>Generated by OpenDocking {_escape(_tool_version())} (kernel
{_escape(_kernel_version())}) on {_escape(platform.python_version())}. This file is
self-contained: the tables and every figure are inside it, and reading it needs no
network access.</p>
</footer>
</body>
</html>
"""


def _plain_text(
    title: str,
    created_at: str,
    run_result: Any,
    data: Mapping[str, Any],
    engine_block: Mapping[str, Any],
    quality: Mapping[str, Any],
    limits: Sequence[str],
    notes: Sequence[str],
    reproducibility: Mapping[str, Any],
) -> str:
    """The plain-text sibling: paste-ready, and the same numbers as the HTML."""
    lines: List[str] = [title, "=" * len(title), ""]
    lines.append(f"generated     : {created_at}")
    lines.append(
        f"tool          : opendocking {reproducibility.get('tool_version')} "
        f"(kernel {reproducibility.get('kernel_version')}, "
        f"{reproducibility.get('platform')})"
    )
    lines.append(f"scoring       : {engine_block.get('scoring')}")
    lines.append(f"seed          : {engine_block.get('seed')}")
    lines.append(f"poses         : {len(getattr(run_result, 'poses', []) or [])}")
    lines.append(f"best affinity : {_number(getattr(run_result, 'best_affinity', None))} kcal/mol")
    command = reproducibility.get("command")
    if command:
        lines.append(f"command       : {command}")
    lines.append("")

    rows = list(data.get("rows") or [])
    if rows:
        lines.append("mode |   affinity | rmsd l.b. | rmsd u.b. | heavy |     LE | key residues")
        lines.append("-----+------------+-----------+-----------+-------+--------+--------------")
        for row in rows:
            lines.append(
                f"{int(row.get('mode') or 0):>4d} | "
                f"{_number(row.get('affinity')):>10} | "
                f"{_number(row.get('rmsd_lb')):>9} | "
                f"{_number(row.get('rmsd_ub')):>9} | "
                f"{_number(row.get('heavy_atoms'), 0):>5} | "
                f"{_number(row.get('ligand_efficiency')):>6} | "
                f"{row.get('residues') or ''}"
            )
        lines.append("")

    profile = data.get("interaction_profile") or {}
    if profile.get("key_residues"):
        lines.append(f"key residues  : {profile['key_residues']}")
        counts = ", ".join(f"{k} {v}" for k, v in sorted((profile.get("counts") or {}).items()))
        if counts:
            lines.append(f"interactions  : {counts}")
        lines.append("")

    for item in data.get("inputs") or []:
        lines.append(
            f"input         : {item.get('role')} {item.get('name')} "
            f"sha256 {str(item.get('sha256') or '')[:16]}…"
        )
    if data.get("inputs"):
        lines.append("")

    lines.append("pose quality")
    lines.append("------------")
    for label, value in (
        ("spread of the reported poses (kcal/mol)", _number(quality.get("spread"))),
        ("poses within 1 kcal/mol of the best", _number(quality.get("n_within_1_kcal"), 0)),
        ("heavy atoms", _number(quality.get("heavy_atoms"), 0)),
    ):
        lines.append(f"  {label:<42} {value}")
    for flag in quality.get("flags") or []:
        lines.append(f"  flag: {flag}")
    lines.append("")

    lines.append("What this run does not establish")
    lines.append("--------------------------------")
    for item in limits:
        lines.append(f"  - {item}")
    if notes:
        lines.append("")
        lines.append("Notes")
        lines.append("-----")
        for note in notes:
            lines.append(f"  - {note}")
    return "\n".join(lines).rstrip() + "\n"


def _interaction_diagram(
    receptor_text: str, result: Any, profile: Mapping[str, Any], title: str
) -> str:
    """The 2-D diagram of the best pose, or an empty string when it cannot be drawn."""
    from . import analysis as _analysis

    receptor_atoms = _report_atoms(receptor_text)
    ligand_atoms = _analysis_ligand_atoms(result)
    if not receptor_atoms or not ligand_atoms:
        return ""
    try:
        interactions = _analysis.profile_interactions(receptor_atoms, ligand_atoms)
        return _analysis.interaction_diagram_svg(
            # A short title on purpose: the diagram draws its legend in the top
            # left, so a long title collides with it.  The report's own heading
            # and the figure caption carry the run's title.
            receptor_atoms,
            ligand_atoms,
            interactions,
            title="Best pose",
        )
    except Exception:  # pragma: no cover - a drawing failure must not fail a report
        return ""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version() -> str:
    try:
        from . import project as _p

        return _p.tool_version()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _kernel_version() -> str:
    try:
        from .docking import kernel_version

        return str(kernel_version())
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def write_html_report(
    path: PathLike,
    result: Any = None,
    *,
    json_out: Optional[PathLike] = None,
    text_out: Optional[PathLike] = None,
    pdf: bool = False,
    pdf_out: Optional[PathLike] = None,
    quiet: bool = False,
    **kwargs,
) -> ReportFiles:
    """Write the self-contained HTML report, plus its JSON and text siblings.

    Parameters
    ----------
    path
        Where to write the ``.html``.
    result
        The run: a :class:`odock.DockResult`, a pose PDBQT (path or text), a dock
        JSON mapping or path, or ``project=<.odockproj>``.
    json_out, text_out
        The machine-readable and plain-text siblings.  Defaults:
        ``<path>.json`` and ``<path>.txt``.  Pass ``json_out=False`` to skip one.
    pdf
        Also produce a PDF, if a converter is available (:func:`pdf_converter`).
        When none is, the report still succeeds and ``notes`` says so.
    **kwargs
        Everything :func:`build_report` accepts (``project``, ``receptor``,
        ``ligand``, ``rows``, ``engine``, ``extra_figures``, ...).

    Returns
    -------
    :class:`ReportFiles` -- the paths written and the notes about what could not
    be drawn, so a caller (and the CLI) can report it rather than guess.
    """
    built = build_report(result, **kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(built.html, encoding="utf-8")
    files = ReportFiles(
        path=target,
        n_figures=len(built.figures),
        n_images=built.n_images,
        size_bytes=target.stat().st_size,
        notes=list(built.notes),
    )

    if json_out is not False:
        destination = Path(json_out) if json_out else target.with_suffix(".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(built.payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        files.json_path = destination
    if text_out is not False:
        destination = Path(text_out) if text_out else target.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(built.text, encoding="utf-8")
        files.text_path = destination

    if pdf:
        destination = Path(pdf_out) if pdf_out else target.with_suffix(".pdf")
        try:
            files.pdf_path = write_pdf(built.html, destination)
        except PdfUnavailable as exc:
            files.notes.append(str(exc))
    return files


class PdfUnavailable(RuntimeError):
    """No PDF converter is available (or it failed); the HTML is the deliverable."""


#: Places a headless Chromium may live, per platform.  Discovered, never required.
_BROWSER_CANDIDATES: Tuple[str, ...] = (
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "msedge",
)

_WINDOWS_BROWSERS: Tuple[str, ...] = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)


def pdf_converter() -> Optional[Tuple[str, str]]:
    """``(kind, executable)`` for an available HTML->PDF converter, else ``None``.

    Two kinds are recognised, both discovered rather than depended on:

    * ``("weasyprint", "")`` -- the Python library, when it is importable;
    * ``("browser", "<path>")`` -- a headless Chromium/Edge, which is present on
      most desktops and needs ``--print-to-pdf`` only.

    Set ``ODOCK_PDF_CONVERTER`` to an executable path to override the search.
    """
    override = os.environ.get("ODOCK_PDF_CONVERTER")
    if override:
        return ("browser", override)
    try:
        import weasyprint  # noqa: F401

        return ("weasyprint", "")
    except Exception:
        pass
    for name in _BROWSER_CANDIDATES:
        found = shutil.which(name)
        if found:
            return ("browser", found)
    if os.name == "nt":
        for candidate in _WINDOWS_BROWSERS:
            if Path(candidate).exists():
                return ("browser", candidate)
    return None


def write_pdf(document: Union[str, PathLike], path: PathLike, *, timeout: float = 180.0) -> Path:
    """Render an HTML document (text or a path) to `path` as a PDF.

    Raises :class:`PdfUnavailable` with an actionable message when no converter is
    installed, or when the converter that is installed failed.  The HTML report
    never depends on this: it is a bonus format, and the module says so rather
    than adding a heavy dependency for it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    converter = pdf_converter()
    if converter is None:
        raise PdfUnavailable(
            "no HTML-to-PDF converter is available on this machine, so the report "
            "was written as HTML only (that file is self-contained and prints "
            "from a browser). Install WeasyPrint or a Chromium/Edge browser, or "
            "point ODOCK_PDF_CONVERTER at one, to get a PDF."
        )
    kind, executable = converter

    temporary_html: Optional[Path] = None
    if isinstance(document, (str, os.PathLike)) and not str(document).lstrip().startswith(
        ("<", "<!DOCTYPE")
    ):
        source = Path(os.fspath(document))
        if not source.exists():
            raise PdfUnavailable(f"no such HTML document: {source}")
        html_path = source
    else:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        )
        try:
            handle.write(str(document))
        finally:
            handle.close()
        html_path = Path(handle.name)
        temporary_html = html_path

    if kind == "weasyprint":
        try:
            import weasyprint

            weasyprint.HTML(filename=str(html_path)).write_pdf(str(target))
        except Exception as exc:  # pragma: no cover - depends on the install
            raise PdfUnavailable(
                "WeasyPrint could not render the report: "
                + _project.redact_paths(str(exc))
            )
        finally:
            if temporary_html is not None:
                temporary_html.unlink(missing_ok=True)
        return target

    url = html_path.resolve().as_uri()
    # A private profile directory: a shared one is often locked by a running
    # browser, and writing into the user's own profile is not something a report
    # generator should do.
    #
    # `tempfile.mkdtemp` alone is not enough, and this is the incident: when the
    # environment's temp directory is not writable, `tempfile` **silently falls back
    # to the working directory**, and the profile it creates there is left behind if
    # the browser keeps a handle open past the cleanup (`shutil.rmtree` fails on
    # Windows and the old code swallowed it with `ignore_errors=True`).  Four
    # `odock-chrome-*` directories reached the published 0.2.1 tree that way.  So the
    # directory is created only where it belongs, the fallback is refused, and the
    # cleanup reports itself if it cannot finish.
    profile = _browser_profile_directory()
    command = [
        executable,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--disable-crash-reporter",
        "--disable-breakpad",
        f"--user-data-dir={profile}",
        "--virtual-time-budget=8000",
        f"--print-to-pdf={target}",
        url,
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=float(timeout), check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PdfUnavailable(
            "no PDF was produced: the headless browser could not print the report ("
            + _project.redact_paths(str(exc))
            + "). The HTML report is complete and self-contained; print it from a "
            "browser instead."
        )
    finally:
        leftover = _remove_browser_profile(profile)
        if temporary_html is not None:
            temporary_html.unlink(missing_ok=True)
    if leftover:
        raise PdfUnavailable(
            "no PDF was produced, and the browser's profile directory could not be "
            f"removed ({_project.portable_path(leftover)}): delete it before "
            "publishing, or run with a writable temp directory. The HTML report is "
            "complete and self-contained."
        )
    if not target.exists() or target.stat().st_size < 400:
        detail = _project.redact_paths(
            (completed.stderr or b"").decode("utf-8", "replace").strip()
        )[:400]
        raise PdfUnavailable(
            "no PDF was produced: the headless browser produced no output"
            + (f" ({detail})" if detail else f" (exit {completed.returncode})")
            + ". The HTML report is complete and self-contained; print it from a "
            "browser if a PDF is needed."
        )
    with open(target, "rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise PdfUnavailable("the converter wrote something that is not a PDF")
    return target


# ---------------------------------------------------------------------------
# Comparing several runs, and a campaign index
#
# Both pages reuse the renderer above (one document builder, one stylesheet, the
# same self-containment check) rather than growing a second one: a comparison is
# a report about runs, and an index is a report about reports.
# ---------------------------------------------------------------------------

#: What a *comparison* does not establish -- printed on every comparison page.
COMPARISON_LIMITS: Tuple[str, ...] = (
    "A delta is arithmetic, not a verdict: a lower score is a better score *for "
    "this force field in this box*, which is not the same as a better binder.",
    "Comparing runs whose inputs differ (a different receptor, ligand or box) "
    "compares two experiments, not two settings; the input hashes are on the page "
    "so this is visible rather than assumed.",
    "The per-mode table compares pose 1 with pose 1, pose 2 with pose 2.  The "
    "modes are ranked independently by each run, so a row is not the same "
    "physical pose on both sides.",
    "The engine is deterministic for a given release and seed, so a difference "
    "between two runs is a difference in what was asked for -- unless the two "
    "runs came from different kernel builds, in which case the differences may "
    "include the build.",
    "Nothing here is a statistical statement: two runs are two runs, and no "
    "confidence interval can be computed from them.",
)


def _comparison_payload(document: Any) -> Dict[str, Any]:
    """Accept a comparison payload as a mapping or as its JSON text."""
    if isinstance(document, Mapping):
        return dict(document)
    if isinstance(document, (str, os.PathLike)) and Path(os.fspath(document)).suffix.lower() == ".json":
        return json.loads(Path(os.fspath(document)).read_text(encoding="utf-8"))
    if isinstance(document, str):
        return json.loads(document)
    raise ValueError(
        "a comparison must be the mapping (or JSON document) that "
        "odock.project.compare_projects returns"
    )


def build_comparison_report(
    comparison: Any,
    *,
    title: Optional[str] = None,
    created: Optional[str] = None,
    extra_sections: Optional[Sequence[Tuple[str, str, str]]] = None,
    extra_limits: Optional[Sequence[str]] = None,
    payload_extras: Optional[Mapping[str, Any]] = None,
) -> HtmlReport:
    """Build the side-by-side comparison page for several runs.

    `extra_sections` are ``(anchor, heading, html)`` triples appended after the
    comparison's own sections and before the limits.  That is how a study diff
    adds its protocol, membership and hit-list tables **without a second
    renderer**: the document builder, the stylesheet, the self-containment check
    and the limits section are not forked.  `extra_limits` are added to the
    caveat list and `payload_extras` to the JSON sibling.
    """
    payload = _comparison_payload(comparison)
    runs = list(payload.get("runs") or [])
    if not runs:
        raise ValueError("the comparison holds no run")
    pairs = list(payload.get("pairwise") or [])
    report_title = str(title or f"Comparison of {len(runs)} runs")
    created_at = str(created or payload.get("created_utc") or _now())
    notes: List[str] = []
    if payload.get("problems"):
        notes.extend(str(item) for item in payload["problems"])

    body: List[str] = [
        f'<header><h1>{_escape(report_title)}</h1>'
        f'<p class="subtitle">OpenDocking comparison — baseline '
        f'<strong>{_escape(payload.get("baseline"))}</strong>, generated '
        f"{_escape(created_at)}. Affinity deltas are the later run minus the "
        "baseline, per mode.</p></header>"
    ]

    glance_rows = [
        [
            _escape(run.get("name")),
            _escape(run.get("title")),
            _number(run.get("n_poses"), 0),
            _number(run.get("best_affinity")),
            _number(run.get("spread")),
            _escape(run.get("seed")),
            _escape(run.get("scoring")),
            _number(run.get("ligand_efficiency")),
            _number((run.get("clusters") or {}).get("n_clusters"), 0),
            _dash(run.get("verify_ok")),
        ]
        for run in runs
    ]
    body.append(
        _section(
            "runs",
            "The runs at a glance",
            _table(
                ["run", "title", "poses", "best affinity (kcal/mol)", "spread",
                 "seed", "force field", "LE (kcal/mol/HA)", "clusters", "verifies"],
                glance_rows,
                table_id="runs",
                numeric=(2, 3, 4, 7, 8),
            ),
        )
    )

    # -- settings that differ ---------------------------------------------
    all_fields: List[str] = []
    for pair in pairs:
        for difference in pair.get("settings_differences") or []:
            if difference["field"] not in all_fields:
                all_fields.append(str(difference["field"]))
    if all_fields:
        rows = []
        for field in all_fields:
            values = []
            for run in runs:
                value = (run.get("engine") or {}).get(field)
                values.append("—" if value is None else _escape(value))
            rows.append([_escape(field)] + values)
        body.append(
            _section(
                "settings",
                "Engine settings that differ",
                _table(["field"] + [_escape(run.get("name")) for run in runs], rows,
                       table_id="settings")
                + "<p>The comparison was asked for with different settings for the "
                "fields above; every other setting is identical across these runs."
                "</p>",
            )
        )
    else:
        body.append(
            _section(
                "settings",
                "Engine settings that differ",
                "<p>These runs recorded identical engine settings: any difference "
                "below comes from the inputs or from the code, not from the "
                "settings.</p>",
            )
        )

    # -- inputs ------------------------------------------------------------
    input_roles: List[str] = []
    for run in runs:
        for item in run.get("inputs") or []:
            role = str(item.get("role"))
            if role not in input_roles:
                input_roles.append(role)
    if input_roles:
        rows = []
        for role in input_roles:
            digests = []
            for run in runs:
                digest = ""
                for item in run.get("inputs") or []:
                    if str(item.get("role")) == role:
                        digest = str(item.get("sha256") or "")
                digests.append(f"<code>{_escape(digest[:16])}…</code>" if digest else "—")
            rows.append([_escape(role)] + digests)
        body.append(
            _section(
                "inputs",
                "Inputs and their hashes",
                _table(["role"] + [_escape(run.get("name")) for run in runs], rows,
                       table_id="inputs"),
            )
        )

    # -- per-mode affinities and deltas -----------------------------------
    baseline = runs[0]
    baseline_affinities = list(baseline.get("affinities") or [])
    if baseline_affinities and pairs:
        rows = []
        for index in range(len(baseline_affinities)):
            row = [_number(index + 1, 0), _number(baseline_affinities[index])]
            for pair in pairs:
                delta = (pair.get("per_mode_affinity_delta") or [])
                row.append(_number(delta[index]) if index < len(delta) else "—")
            rows.append(row)
        headers = ["mode", "baseline (kcal/mol)"] + [
            f"{pair.get('b')} − baseline" for pair in pairs
        ]
        body.append(
            _section(
                "deltas",
                "Per-mode affinity deltas",
                _table(headers, rows, table_id="deltas",
                       numeric=tuple(range(0, len(headers)))),
            )
        )

    # -- pairwise numbers --------------------------------------------------
    if pairs:
        rows = [
            [
                _escape(pair.get("a")),
                _escape(pair.get("b")),
                f"{int(pair.get('pose_count_delta') or 0):+d}",
                _number(pair.get("best_affinity_delta")),
                _number(pair.get("max_affinity_delta")),
                _number(pair.get("top_pose_rmsd")),
                _number(pair.get("spearman_rho"), 2),
                f"{int(pair.get('cluster_count_delta') or 0):+d}",
            ]
            for pair in pairs
        ]
        body.append(
            _section(
                "pairwise",
                "Top-pose, ranking and clustering differences",
                _table(
                    ["run A", "run B", "Δposes", "Δbest (kcal/mol)",
                     "max |Δ| per mode", "top-pose RMSD (Å)", "Spearman rho",
                     "Δclusters"],
                    rows,
                    table_id="pairwise",
                    numeric=(2, 3, 4, 5, 6, 7),
                )
                + "<p>The top-pose RMSD is symmetry-aware and measured without "
                "superposition, so it is the movement of the pose in the same "
                "frame, not a best-fit distance.</p>",
            )
        )

    # -- clustering --------------------------------------------------------
    cluster_rows = [
        [
            _escape(run.get("name")),
            _number((run.get("clusters") or {}).get("n_clusters"), 0),
            _number((run.get("clusters") or {}).get("largest"), 0),
            _escape(", ".join(
                str(mode) for mode in (run.get("clusters") or {}).get("representatives") or []
            ) or "—"),
        ]
        for run in runs
    ]
    body.append(
        _section(
            "clusters",
            f"Clustering at {_number(payload.get('cutoff_angstrom'), 2)} Å",
            _table(["run", "clusters", "largest cluster", "representative modes"],
                   cluster_rows, table_id="clusters", numeric=(1, 2)),
        )
    )

    for anchor, heading, content in extra_sections or ():
        body.append(_section(str(anchor), str(heading), content))

    limits = list(COMPARISON_LIMITS)
    for item in extra_limits or ():
        if str(item) not in limits:
            limits.append(str(item))
    for run in runs:
        for item in run.get("does_not_establish") or []:
            if item not in limits:
                limits.append(str(item))
    body.append(
        _section(
            "limits",
            "What this comparison does not establish",
            _bullet_list([_escape(item) for item in limits]),
        )
    )

    notes_block = ""
    if notes:
        notes_block = '<section id="notes"><h2>Notes on this comparison</h2>' + _bullet_list(
            [_escape(note) for note in notes]
        ) + "</section>"

    document = _document(report_title, "\n".join(body) + notes_block)
    external = find_external_references(document)
    if external:
        raise RuntimeError(
            "the comparison would not be self-contained: " + "; ".join(external)
        )
    payload = dict(payload)
    payload.setdefault("title", report_title)
    if payload_extras:
        payload.update(dict(payload_extras))
    payload["does_not_establish"] = limits
    payload["tool"] = {
        "name": "opendocking",
        "version": _tool_version(),
        "kernel_version": _kernel_version(),
    }
    return HtmlReport(
        title=report_title,
        html=document,
        payload=payload,
        text=_comparison_text(report_title, created_at, runs, pairs, payload),
        figures=[],
        notes=notes,
    )


def _comparison_text(
    title: str,
    created_at: str,
    runs: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> str:
    lines = [title, "=" * len(title), "", f"generated : {created_at}",
             f"baseline  : {payload.get('baseline')}", ""]
    for run in runs:
        lines.append(
            f"{str(run.get('name')):<28} poses {run.get('n_poses')!s:<3} "
            f"best {_number(run.get('best_affinity')):>8} kcal/mol  "
            f"spread {_number(run.get('spread'))}  seed {run.get('seed')}  "
            f"{run.get('scoring')}  clusters {(run.get('clusters') or {}).get('n_clusters')}"
        )
    for pair in pairs:
        lines += [
            "",
            f"{pair.get('a')} -> {pair.get('b')}",
            f"  poses {int(pair.get('pose_count_delta') or 0):+d}, best "
            f"{_number(pair.get('best_affinity_delta'))} kcal/mol, top-pose RMSD "
            f"{_number(pair.get('top_pose_rmsd'))} A, max |delta| "
            f"{_number(pair.get('max_affinity_delta'))} kcal/mol, rho "
            f"{_number(pair.get('spearman_rho'), 2)}",
        ]
        for difference in pair.get("settings_differences") or []:
            left, right = (difference.get("values") or [None, None])[:2]
            lines.append(f"  setting {difference.get('field')}: {left} -> {right}")
        for difference in pair.get("input_differences") or []:
            lines.append(f"  input {difference.get('role')}: changed")
    lines += ["", "What this comparison does not establish", "-" * 39]
    for item in COMPARISON_LIMITS:
        lines.append(f"  - {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_comparison_report(
    path: PathLike,
    comparison: Any,
    *,
    title: Optional[str] = None,
    json_out: Optional[PathLike] = None,
    text_out: Optional[PathLike] = None,
) -> ReportFiles:
    """Write the comparison page plus its JSON and text siblings."""
    built = build_comparison_report(comparison, title=title)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(built.html, encoding="utf-8")
    files = ReportFiles(
        path=target,
        n_figures=0,
        n_images=built.n_images,
        size_bytes=target.stat().st_size,
        notes=list(built.notes),
    )
    if json_out is not False:
        destination = Path(json_out) if json_out else target.with_suffix(".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(built.payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        files.json_path = destination
    if text_out is not False:
        destination = Path(text_out) if text_out else target.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(built.text, encoding="utf-8")
        files.text_path = destination
    return files


#: What a campaign index does not establish.
INDEX_LIMITS: Tuple[str, ...] = (
    "An index row is a summary of a project, not the run: open the project (and "
    "verify it) before quoting a number.",
    "Rows from different campaigns are not comparable unless their docking hash "
    "matches; the hash is on the page precisely so that this is checkable.",
    "A verified row means the stored bytes have not changed; it does not mean the "
    "run is correct.",
    "The ranking is by best affinity across the whole directory, which mixes "
    "different receptors, boxes and force fields when the directory holds more "
    "than one campaign.",
)


def build_campaign_index(
    document: Any,
    *,
    title: Optional[str] = None,
    created: Optional[str] = None,
) -> HtmlReport:
    """Build the campaign index page over a directory of projects."""
    payload = dict(document) if isinstance(document, Mapping) else json.loads(
        Path(os.fspath(document)).read_text(encoding="utf-8")
    )
    entries = list(payload.get("entries") or [])
    if not entries:
        raise ValueError("the index holds no project")
    report_title = str(title or f"Campaign index — {payload.get('n_projects')} project(s)")
    created_at = str(created or payload.get("created_utc") or _now())
    notes = [str(item) for item in payload.get("problems") or []]

    rows = []
    for entry in entries:
        href = _escape(entry.get("report_href") or entry.get("relative") or entry.get("name"))
        rows.append(
            [
                f'<a href="{href}">{_escape(entry.get("label"))}</a>',
                _escape(entry.get("title")),
                _number(entry.get("n_poses"), 0),
                _number(entry.get("best_affinity")),
                _escape(entry.get("seed")),
                _escape(entry.get("scoring")),
                _number(entry.get("heavy_atoms"), 0),
                _number(entry.get("ligand_efficiency")),
                _escape(entry.get("key_residues")),
                _dash(entry.get("verify_ok")),
                _number((entry.get("size_bytes") or 0) / 1024.0, 1),
            ]
        )
    body = [
        f'<header><h1>{_escape(report_title)}</h1>'
        f'<p class="subtitle">OpenDocking campaign index — {payload.get("n_projects")} '
        f'project(s) under <code>{_escape(payload.get("directory"))}</code>, '
        f'{payload.get("n_verified")} verified. Generated {_escape(created_at)}. '
        "Every link is a file next to this page.</p></header>"
    ]
    body.append(
        _section(
            "index",
            "Projects",
            _table(
                ["project", "title", "poses", "best affinity (kcal/mol)", "seed",
                 "force field", "heavy atoms", "LE (kcal/mol/HA)", "key residues",
                 "verifies", "KiB"],
                rows,
                table_id="projects",
                numeric=(2, 3, 6, 7, 10),
            )
            + "<p>The link points at the member's HTML report when one sits next to "
            "it, and at the project file itself otherwise.</p>",
        )
    )
    if payload.get("campaign_hashes") or payload.get("library_hashes"):
        body.append(
            _section(
                "campaign",
                "Campaign identity",
                _definition_list(
                    [
                        ("projects", _escape(payload.get("n_projects"))),
                        ("verified", _escape(payload.get("n_verified"))),
                        ("best affinity (kcal/mol)", _number(payload.get("best_affinity"))),
                        ("worst affinity (kcal/mol)", _number(payload.get("worst_affinity"))),
                        ("docking hash(es)",
                         "<code>" + _escape(", ".join(payload.get("campaign_hashes") or []))
                         + "</code>"),
                        ("library hash(es)",
                         "<code>" + _escape(", ".join(payload.get("library_hashes") or []))
                         + "</code>"),
                        ("pattern", _escape(payload.get("pattern"))),
                    ]
                )
                + "<p>Two projects belong to the same campaign when their docking "
                "hash matches: the same receptors, box, force field, search "
                "settings and seed.  The library hash identifies the library that "
                "was screened.</p>",
            )
        )
    body.append(
        _section(
            "limits",
            "What this index does not establish",
            _bullet_list([_escape(item) for item in INDEX_LIMITS]),
        )
    )
    notes_block = ""
    if notes:
        notes_block = '<section id="notes"><h2>Notes on this index</h2>' + _bullet_list(
            [_escape(note) for note in notes]
        ) + "</section>"
    document = _document(report_title, "\n".join(body) + notes_block)
    external = find_external_references(document)
    if external:
        raise RuntimeError("the index would not be self-contained: " + "; ".join(external))

    text = [report_title, "=" * len(report_title), ""]
    text.append(f"directory : {payload.get('directory')}")
    text.append(f"projects  : {payload.get('n_projects')} ({payload.get('n_verified')} verified)")
    text.append("")
    for entry in entries:
        text.append(
            f"  {str(entry.get('label'))[:40]:<42} {_number(entry.get('best_affinity')):>8} "
            f"kcal/mol  poses {entry.get('n_poses')}  seed {entry.get('seed')}  "
            f"{'OK' if entry.get('verify_ok') else 'FAILED'}"
        )
    text += ["", "What this index does not establish", "-" * 35]
    for item in INDEX_LIMITS:
        text.append(f"  - {item}")
    payload = dict(payload)
    payload.setdefault("title", report_title)
    payload["tool"] = {"name": "opendocking", "version": _tool_version(),
                       "kernel_version": _kernel_version()}
    return HtmlReport(
        title=report_title,
        html=document,
        payload=payload,
        text="\n".join(text).rstrip() + "\n",
        figures=[],
        notes=notes,
    )


def write_campaign_index(
    path: PathLike,
    index: Any,
    *,
    title: Optional[str] = None,
    json_out: Optional[PathLike] = None,
    text_out: Optional[PathLike] = None,
) -> ReportFiles:
    """Write the campaign index page plus its JSON and text siblings."""
    built = build_campaign_index(index, title=title)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(built.html, encoding="utf-8")
    files = ReportFiles(
        path=target,
        n_figures=0,
        n_images=built.n_images,
        size_bytes=target.stat().st_size,
        notes=list(built.notes),
    )
    if json_out is not False:
        destination = Path(json_out) if json_out else target.with_suffix(".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(built.payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        files.json_path = destination
    if text_out is not False:
        destination = Path(text_out) if text_out else target.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(built.text, encoding="utf-8")
        files.text_path = destination
    return files


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Studies: a diff of two collections, and a report over one
#
# Both are built on the comparison renderer above, through its
# `extra_sections` hook: the study-specific tables are *sections*, not a second
# document builder, so the self-containment guarantee and the limits section are
# the same code.
# ---------------------------------------------------------------------------


def _protocol_table(differences: Sequence[Mapping[str, Any]]) -> str:
    """The metadata/protocol fields that differ, with both values."""
    if not differences:
        return (
            "<p>The two studies recorded identical metadata and protocol fields: "
            "any difference in the members is a difference in the molecules or the "
            "runs, not in what was asked of them.</p>"
        )
    rows = [
        [
            _escape(item.get("field")),
            _escape(item.get("kind")),
            # A field one study never recorded is shown as a dash, not as an empty
            # cell that reads like "false" or "zero".
            _dash(item.get("values", [None, None])[0]),
            _dash(item.get("values", [None, None])[1]),
            _escape(item.get("note")),
        ]
        for item in differences
    ]
    return _table(
        ["field", "kind", "study A", "study B", "note"],
        rows,
        table_id="protocol",
    )


def _membership_table(payload: Mapping[str, Any]) -> str:
    """Members added, removed and shared."""
    members = payload.get("members") or {}
    parts: List[str] = []
    added = list(members.get("added") or [])
    removed = list(members.get("removed") or [])
    shared = list(members.get("shared") or [])
    parts.append(
        f"<p>{len(shared)} shared member(s), {len(added)} added, {len(removed)} "
        "removed.</p>"
    )
    rows = []
    for item in shared:
        rows.append(["shared", _escape(item.get("label")), "—", ""])
    for item in added:
        rows.append(
            [
                "added",
                _escape(item.get("label")),
                _number(item.get("best_affinity")),
                f"rank {_number(item.get('rank_b'), 0)} in study B",
            ]
        )
    for item in removed:
        rows.append(
            [
                "removed",
                _escape(item.get("label")),
                _number(item.get("best_affinity")),
                f"rank {_number(item.get('rank_a'), 0)} in study A",
            ]
        )
    if rows:
        parts.append(
            _table(["change", "member", "best affinity (kcal/mol)", "note"], rows,
                   table_id="membership", numeric=(2,))
        )
    return "\n".join(parts)


def _hit_list_table(payload: Mapping[str, Any]) -> str:
    """The movement of the hit list: ranks and top-N membership, with numbers."""
    hit = payload.get("hit_list") or {}
    rows = [
        [
            _escape(item.get("member")),
            _number(item.get("best_a")),
            _number(item.get("best_b")),
            _number(item.get("best_affinity_delta")),
            _number(item.get("rank_a"), 0),
            _number(item.get("rank_b"), 0),
            f"{int(item['rank_delta']):+d}" if item.get("rank_delta") is not None else "—",
            _escape(item.get("movement")),
        ]
        for item in hit.get("members") or []
    ]
    table = (
        _table(
            ["member", "best A", "best B", "Δbest (kcal/mol)", "rank A", "rank B",
             "Δrank", "movement"],
            rows,
            table_id="hitlist",
            numeric=(1, 2, 3, 4, 5, 6),
        )
        if rows
        else "<p>No member appears in both studies, so no rank moved.</p>"
    )
    top_n = int(hit.get("top_n") or 0)
    entered = ", ".join(str(name) for name in hit.get("entered_top") or []) or "none"
    left = ", ".join(str(name) for name in hit.get("left_top") or []) or "none"
    return (
        table
        + f"<p>Top {top_n}: <strong>entered</strong> {_escape(entered)}; "
        f"<strong>left</strong> {_escape(left)}. Ranks are within each study, and a "
        "rank can move because the pool changed rather than because the molecule "
        "did.</p>"
    )


def build_study_diff_report(
    payload: Any,
    *,
    title: Optional[str] = None,
    created: Optional[str] = None,
) -> HtmlReport:
    """The study diff page, built on the comparison renderer."""
    document = (
        dict(payload)
        if isinstance(payload, Mapping)
        else json.loads(Path(os.fspath(payload)).read_text(encoding="utf-8"))
    )
    studies = document.get("studies") or {}
    left = studies.get("a") or {}
    right = studies.get("b") or {}
    report_title = str(
        title or f"{left.get('name')} → {right.get('name')} (study diff)"
    )
    comparison = document.get("comparison") or {}
    extras = [
        (
            "protocol",
            "Protocol and metadata that changed",
            _protocol_table(list(document.get("metadata_differences") or [])),
        ),
        ("membership", "Members added, removed, shared", _membership_table(document)),
        (
            "hitlist",
            "How the hit list moved",
            _hit_list_table(document),
        ),
    ]
    built = build_comparison_report(
        comparison,
        title=report_title,
        created=created or document.get("created_utc"),
        extra_sections=extras,
        extra_limits=list(document.get("does_not_establish") or []),
        payload_extras={"studies": studies, "study_diff": {
            "top_n": document.get("top_n"),
            "metadata_differences": document.get("metadata_differences"),
            "members": document.get("members"),
            "hit_list": document.get("hit_list"),
        }},
    )
    # Replace the run-oriented lead paragraph with one about the two studies.
    built.html = built.html.replace(
        f'<p class="subtitle">OpenDocking comparison — baseline '
        f'<strong>{_escape(comparison.get("baseline"))}</strong>, generated ',
        f'<p class="subtitle">OpenDocking study diff — '
        f'<strong>{_escape(left.get("name"))}</strong> ({left.get("n_members")} member(s), '
        f'{_escape(left.get("created_utc"))}) → '
        f'<strong>{_escape(right.get("name"))}</strong> ({right.get("n_members")} member(s), '
        f'{_escape(right.get("created_utc"))}), generated ',
        1,
    )
    built.title = report_title
    built.text = study_diff_text(document, title=report_title)
    return built


def study_diff_text(payload: Mapping[str, Any], *, title: Optional[str] = None) -> str:
    """The plain-text sibling of a study diff."""
    studies = payload.get("studies") or {}
    left, right = studies.get("a") or {}, studies.get("b") or {}
    name = str(title or f"{left.get('name')} -> {right.get('name')}")
    lines = [
        name,
        "=" * len(name),
        "",
        f"study A : {left.get('name')} ({left.get('n_members')} member(s), "
        f"{left.get('created_utc')}, operator {left.get('operator')})",
        f"study B : {right.get('name')} ({right.get('n_members')} member(s), "
        f"{right.get('created_utc')}, operator {right.get('operator')})",
        "",
        "Protocol and metadata that changed",
        "----------------------------------",
    ]
    differences = list(payload.get("metadata_differences") or [])
    if not differences:
        lines.append("  none: the two studies recorded identical metadata")
    for item in differences:
        values = item.get("values") or [None, None]
        lines.append(
            f"  {item.get('field')}: {values[0]} -> {values[1]}"
            + (f" ({item['note']})" if item.get("note") else "")
        )
    members = payload.get("members") or {}
    lines += ["", "Members", "-------"]
    lines.append(
        f"  shared {len(members.get('shared') or [])}, "
        f"added {len(members.get('added') or [])}, "
        f"removed {len(members.get('removed') or [])}"
    )
    for item in members.get("added") or []:
        lines.append(f"  + {item.get('label')} ({_number(item.get('best_affinity'))} kcal/mol)")
    for item in members.get("removed") or []:
        lines.append(f"  - {item.get('label')} ({_number(item.get('best_affinity'))} kcal/mol)")
    hit = payload.get("hit_list") or {}
    lines += [
        "",
        f"How the hit list moved (top {hit.get('top_n')})",
        "-" * 34,
        f"  {'member':<28} {'best A':>8} {'best B':>8} {'dBest':>8} "
        f"{'rank A':>7} {'rank B':>7} {'dRank':>6}  movement",
    ]
    for item in hit.get("members") or []:
        rank_delta = item.get("rank_delta")
        lines.append(
            f"  {str(item.get('member'))[:28]:<28} {_number(item.get('best_a')):>8} "
            f"{_number(item.get('best_b')):>8} {_number(item.get('best_affinity_delta')):>8} "
            f"{_number(item.get('rank_a'), 0):>7} {_number(item.get('rank_b'), 0):>7} "
            f"{(f'{int(rank_delta):+d}' if rank_delta is not None else '-'):>6}  "
            f"{item.get('movement')}"
        )
    entered = ", ".join(str(item) for item in hit.get("entered_top") or []) or "none"
    left_top = ", ".join(str(item) for item in hit.get("left_top") or []) or "none"
    lines.append(f"  entered the top {hit.get('top_n')}: {entered}")
    lines.append(f"  left the top {hit.get('top_n')}: {left_top}")
    lines += ["", "What this comparison does not establish", "-" * 38]
    for item in payload.get("does_not_establish") or []:
        lines.append(f"  - {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_study_diff(
    path: PathLike,
    payload: Any,
    *,
    title: Optional[str] = None,
    json_out: Optional[PathLike] = None,
    text_out: Optional[PathLike] = None,
) -> ReportFiles:
    """Write the study diff page plus its JSON and text siblings."""
    built = build_study_diff_report(payload, title=title)
    return _write_siblings(path, built, json_out=json_out, text_out=text_out)


def build_study_report(
    payload: Any,
    *,
    title: Optional[str] = None,
    created: Optional[str] = None,
    study_dir: Optional[PathLike] = None,
    page_dir: Optional[PathLike] = None,
) -> HtmlReport:
    """One self-contained page over a study.

    `report_href` in the payload is relative to the *study directory*.  When the
    page is written somewhere else (and `study_dir` is given), the links are
    rewritten relative to `page_dir` so they still resolve: a page whose links
    only work from one directory is a broken page.
    """
    document = (
        dict(payload)
        if isinstance(payload, Mapping)
        else json.loads(Path(os.fspath(payload)).read_text(encoding="utf-8"))
    )
    entries = list(document.get("entries") or [])
    if not entries:
        raise ValueError("the study holds no member")
    if study_dir is not None and page_dir is not None:
        base = Path(study_dir).resolve()
        page = Path(page_dir).resolve()
        rewritten = []
        for entry in entries:
            entry = dict(entry)
            href = str(entry.get("report_href") or "")
            if href and entry.get("has_report"):
                try:
                    entry["report_href"] = _project._as_posix(
                        str((base / href).resolve().relative_to(page))
                    )
                except ValueError:
                    entry["report_href"] = _project._as_posix(
                        os.path.relpath(str(base / href), str(page))
                    )
            rewritten.append(entry)
        entries = rewritten
        document["entries"] = entries
        document["top"] = [
            entry for entry in entries if entry.get("rank", 0) <= int(document.get("top_n") or 10)
        ]
    report_title = str(title or f"{document.get('name')} — study report")
    created_at = str(created or document.get("created_utc") or _now())
    notes: List[str] = []
    verify = document.get("verify") or {}
    if verify.get("problems"):
        notes.extend(str(item) for item in verify["problems"])
    if not verify.get("ok", True):
        notes.append(
            "at least one member did not verify; the page marks the members "
            "individually rather than hiding the failure in a total"
        )

    body: List[str] = [
        f'<header><h1>{_escape(report_title)}</h1>'
        f'<p class="subtitle">OpenDocking study report — {len(entries)} member(s), '
        f'{document.get("n_verified")} verified, generated {_escape(created_at)}. '
        "Every link is a file next to this page.</p></header>"
    ]

    audit = document.get("audit") or {}
    body.append(
        _section(
            "study",
            "The study",
            _definition_list(
                [
                    ("name", _escape(document.get("name"))),
                    ("title", _escape(document.get("title"))),
                    ("created (UTC)", _escape(document.get("created_utc_study"))),
                    ("operator", _escape(audit.get("operator") or "not recorded")),
                    ("tool version", _escape(audit.get("tool_version"))),
                    ("members", _escape(document.get("n_members"))),
                    ("best affinity (kcal/mol)", _number(document.get("best_affinity"))),
                    ("worst affinity (kcal/mol)", _number(document.get("worst_affinity"))),
                ]
            ),
        )
    )

    metadata = document.get("metadata") or {}
    protocol = document.get("protocol") or {}
    body.append(
        _section(
            "protocol",
            "Protocol and metadata",
            _definition_list(
                [
                    ("target", _escape(metadata.get("target") or "not recorded")),
                    ("library", _escape(metadata.get("library") or "not recorded")),
                    (
                        "receptors",
                        _escape(", ".join(str(item) for item in metadata.get("receptors") or [])
                                or "not recorded"),
                    ),
                ]
                + [(f"protocol.{key}", _escape(value)) for key, value in sorted(protocol.items())]
                + [("notes", _escape("; ".join(document.get("notes") or []) or "none"))]
            ),
        )
    )

    body.append(
        _section(
            "members",
            "Members",
            _table(
                ["member", "title", "best affinity (kcal/mol)", "poses", "seed",
                 "force field", "heavy atoms", "LE (kcal/mol/HA)", "key residues",
                 "verifies", "KiB"],
                [
                    [
                        (
                            f'<a href="{_escape(entry.get("report_href"))}">'
                            f'{_escape(entry.get("label"))}</a>'
                            if entry.get("has_report")
                            else _escape(entry.get("label"))
                        ),
                        _escape(entry.get("title")),
                        _number(entry.get("best_affinity")),
                        _number(entry.get("n_poses"), 0),
                        _escape(entry.get("seed")),
                        _escape(entry.get("scoring")),
                        _number(entry.get("heavy_atoms"), 0),
                        _number(entry.get("ligand_efficiency")),
                        _escape(entry.get("key_residues")),
                        _dash(entry.get("verify_ok")),
                        _number((entry.get("size_bytes") or 0) / 1024.0, 1),
                    ]
                    for entry in entries
                ],
                table_id="members",
                numeric=(2, 3, 6, 7, 10),
            ),
        )
    )

    top = list(document.get("top") or [])[: int(document.get("top_n") or 10)]
    body.append(
        _section(
            "top",
            f"Top {document.get('top_n')} across the study",
            _table(
                ["rank", "member", "best affinity (kcal/mol)", "LE (kcal/mol/HA)",
                 "key residues", "verifies"],
                [
                    [
                        _number(entry.get("rank"), 0),
                        _escape(entry.get("label")),
                        _number(entry.get("best_affinity")),
                        _number(entry.get("ligand_efficiency")),
                        _escape(entry.get("key_residues")),
                        _dash(entry.get("verify_ok")),
                    ]
                    for entry in top
                ],
                table_id="top",
                numeric=(0, 2, 3),
            )
            + "<p>The ranking is by best affinity across this study's members only, "
            "and only members whose numbers come from the same protocol are "
            "comparable.</p>",
        )
    )

    limits = list(document.get("does_not_establish") or [])
    body.append(
        _section(
            "limits",
            "What this study does not establish",
            _bullet_list([_escape(item) for item in limits]),
        )
    )
    notes_block = ""
    if notes:
        notes_block = '<section id="notes"><h2>Notes on this study</h2>' + _bullet_list(
            [_escape(note) for note in notes]
        ) + "</section>"
    built = HtmlReport(
        title=report_title,
        html=_document(report_title, "\n".join(body) + notes_block),
        payload=document,
        text=_study_report_text(report_title, document),
        figures=[],
        notes=notes,
    )
    external = find_external_references(built.html)
    if external:
        raise RuntimeError("the study report would not be self-contained: " + "; ".join(external))
    return built


def _study_report_text(title: str, document: Mapping[str, Any]) -> str:
    lines = [
        title,
        "=" * len(title),
        "",
        f"name      : {document.get('name')}",
        f"directory : {document.get('directory')}",
        f"created   : {document.get('created_utc_study')}",
        f"operator  : {(document.get('audit') or {}).get('operator') or 'not recorded'}",
        f"members   : {document.get('n_members')} ({document.get('n_verified')} verified)",
        f"best      : {_number(document.get('best_affinity'))} kcal/mol",
        "",
    ]
    protocol = document.get("protocol") or {}
    if protocol:
        lines.append("protocol")
        lines.append("--------")
        for key, value in sorted(protocol.items()):
            lines.append(f"  {key:<24} {value}")
        lines.append("")
    lines.append("members")
    lines.append("-------")
    for entry in document.get("entries") or []:
        lines.append(
            f"  {entry.get('rank'):>3}. {str(entry.get('label'))[:32]:<34} "
            f"{_number(entry.get('best_affinity')):>8} kcal/mol  "
            f"{'OK' if entry.get('verify_ok') else 'FAILED'}"
        )
    lines += ["", "What this study does not establish", "-" * 35]
    for item in document.get("does_not_establish") or []:
        lines.append(f"  - {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_study_report(
    path: PathLike,
    payload: Any,
    *,
    title: Optional[str] = None,
    study_dir: Optional[PathLike] = None,
    json_out: Optional[PathLike] = None,
    text_out: Optional[PathLike] = None,
) -> ReportFiles:
    """Write the study report page plus its JSON and text siblings.

    Pass `study_dir` when the page is written outside the study directory: the
    member links are then rewritten relative to the page, so they resolve from
    wherever the file ends up.
    """
    target = Path(path)
    built = build_study_report(
        payload, title=title, study_dir=study_dir, page_dir=target.parent
    )
    return _write_siblings(target, built, json_out=json_out, text_out=text_out)


def _write_siblings(
    path: PathLike,
    built: HtmlReport,
    *,
    json_out: Optional[PathLike],
    text_out: Optional[PathLike],
) -> ReportFiles:
    """Write an HTML document and its JSON/text siblings (shared by the pages)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(built.html, encoding="utf-8")
    files = ReportFiles(
        path=target,
        n_figures=len(built.figures),
        n_images=built.n_images,
        size_bytes=target.stat().st_size,
        notes=list(built.notes),
    )
    if json_out is not False:
        destination = Path(json_out) if json_out else target.with_suffix(".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(built.payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        files.json_path = destination
    if text_out is not False:
        destination = Path(text_out) if text_out else target.with_suffix(".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(built.text, encoding="utf-8")
        files.text_path = destination
    return files


# ---------------------------------------------------------------------------
# The command line
#
# Kept in this module (not in cli.py) so one agent owns it: `cli.build_parser`
# only calls :func:`add_report_html_parser`.
# ---------------------------------------------------------------------------


def _cli_err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _is_project_file(path: str) -> bool:
    """Whether `path` is a project: a ZIP file by suffix or by content."""
    candidate = Path(path)
    if not candidate.exists():
        return candidate.suffix.lower() == _project.PROJECT_SUFFIX
    if candidate.suffix.lower() == _project.PROJECT_SUFFIX:
        return True
    try:
        import zipfile

        return zipfile.is_zipfile(candidate)
    except OSError:  # pragma: no cover - unreadable file
        return False


def _report_command(args) -> str:
    """The `odock report-html` line that would reproduce this report.

    Rebuilt from the parsed arguments: ``sys.argv`` belongs to whatever started
    the process when the CLI is driven programmatically.
    """
    parts: List[str] = ["odock", "report-html"]
    if args.project:
        parts += ["--project", str(args.project)]
    elif args.input:
        parts.append(str(args.input))
    for flag, value in (
        ("-p", args.poses), ("-r", args.receptor), ("-l", args.ligand),
        ("-b", args.box), ("--title", args.title), ("--seed", args.seed),
        ("--scoring", args.scoring),
    ):
        if value is not None:
            parts += [flag, str(value)]
    parts += ["-o", str(args.out)]
    for flag, value in (
        ("--no-raster", args.no_raster), ("--no-diagram", args.no_diagram),
        ("--pdf", args.pdf),
    ):
        if value:
            parts.append(flag)
    return _project.portable_command(parts)


def cmd_report_html(args) -> int:
    """Write the self-contained HTML report of a run (plus its siblings)."""
    source: Any = args.input
    kwargs: Dict[str, Any] = {}
    if args.project:
        kwargs["project"] = args.project
        if source is not None:
            _cli_err(f"note: --project {args.project} is used; {source} is ignored")
        if args.poses:
            _cli_err("note: --poses is ignored with --project (the project stores its own)")
        source = None
    elif source is not None and _is_project_file(str(source)):
        kwargs["project"] = source
        source = None
    elif source is not None and Path(str(source)).suffix.lower() == ".json":
        payload = json.loads(Path(str(source)).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and "poses" in payload:
            if args.poses:
                # The pose file carries the coordinates; the JSON the settings.
                from .project import result_from_pdbqt

                kwargs["result"] = result_from_pdbqt(
                    args.poses,
                    box=args.box,
                    receptor=args.receptor,
                    ligand=args.ligand,
                    seed=payload.get("seed"),
                    scoring=args.scoring or payload.get("scoring"),
                )
            else:
                kwargs["result"] = payload
        elif isinstance(payload, Mapping) and ("records" in payload or "config" in payload):
            _cli_err(
                f"odock report-html: error: {source} is a screening *summary*, not one "
                "run. Wrap its members into projects with `odock project screen -s "
                f"{Path(str(source)).parent} -o projects`, then report one of them "
                "(or report a single molecule's pose file)."
            )
            return 2
        else:
            _cli_err(
                f"odock report-html: error: {source} is not a dock JSON document "
                "(no 'poses'); `odock dock --json-out FILE` writes one"
            )
            return 2
    elif source is None:
        _cli_err("odock report-html: error: pass an input file or --project")
        return 2

    if "result" not in kwargs and source is not None:
        kwargs["result"] = source
    if args.poses and "result" not in kwargs and "project" not in kwargs:
        kwargs["result"] = args.poses
    for name in ("receptor", "ligand", "box"):
        value = getattr(args, name)
        if value is not None:
            kwargs[name] = value
    if args.title:
        kwargs["title"] = args.title
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.scoring:
        kwargs["scoring"] = args.scoring
    if "result" not in kwargs and "project" not in kwargs:
        _cli_err("odock report-html: error: no run to report")
        return 2

    command = args.command or _report_command(args)
    json_out: Any = args.json_out if args.json_out else None
    text_out: Any = args.text_out if args.text_out else None
    if args.no_siblings:
        json_out = text_out = False
    try:
        files = write_html_report(
            args.out,
            command=command,
            json_out=json_out,
            text_out=text_out,
            pdf=bool(args.pdf),
            rasterise=not args.no_raster,
            include_diagram=not args.no_diagram,
            **kwargs,
        )
    except (ValueError, _project.ProjectError) as exc:
        _cli_err(f"odock report-html: error: {exc}")
        return 2

    if args.json:
        print(json.dumps(files.as_dict(), indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(
            f"{files.path.name}: {files.size_bytes} bytes, {files.n_figures} figure(s), "
            f"{files.n_images} embedded image(s)"
        )
        for note in files.notes:
            print(f"  note: {note}")
    written = [files.path]
    for path in (files.json_path, files.text_path, files.pdf_path):
        if path is not None:
            written.append(path)
    _cli_err("wrote " + ", ".join(str(path) for path in written))
    return 0


def add_report_html_parser(sub) -> None:
    """Register ``odock report-html`` on `sub` (called from :mod:`odock.cli`).

    Idempotent: several agents add their own parsers to the same subparser list,
    and a re-applied call must not be an error.
    """
    choices = getattr(sub, "choices", None)
    if isinstance(choices, Mapping) and "report-html" in choices:
        return
    parser = sub.add_parser(
        "report-html",
        help="write a self-contained HTML report of a run",
        description=(
            "One HTML file holding the run: the inputs with their SHA-256, the "
            "preparation summary, the box, the engine settings and the seed, the "
            "ranking table, the interaction profile with the 2-D diagram, the "
            "pose-quality metrics, and an explicit 'what this run does not "
            "establish' section.  Figures are embedded, nothing is fetched, and "
            "the same numbers are written as .json and .txt siblings."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="a pose PDBQT, a dock JSON document, or a .odockproj",
    )
    parser.add_argument(
        "--project",
        help="report a project file instead of a loose pose file",
    )
    parser.add_argument(
        "-p", "--poses",
        help="the pose PDBQT, when the input is a dock JSON document without coordinates",
    )
    parser.add_argument("-r", "--receptor", help="receptor PDBQT (for the interaction profile)")
    parser.add_argument("-l", "--ligand", help="ligand PDBQT (for the heavy-atom count)")
    parser.add_argument("-b", "--box", help="box JSON written by `odock box`")
    parser.add_argument("-o", "--out", required=True, help="output .html")
    parser.add_argument("--json-out", help="machine-readable sibling (default: <out>.json)")
    parser.add_argument("--text-out", help="plain-text sibling (default: <out>.txt)")
    parser.add_argument("--no-siblings", action="store_true",
                        help="write only the HTML")
    parser.add_argument(
        "--title", help="report title (default: the project's title, or a generic one)"
    )
    parser.add_argument("--seed", type=int, help="the seed the run used, when it is not stored")
    parser.add_argument(
        "--scoring", choices=["vina", "vinardo", "ad4"], help="the force field the run used"
    )
    parser.add_argument(
        "--command", help="the exact command to record as the provenance (default: this one)"
    )
    parser.add_argument(
        "--no-raster", action="store_true",
        help="do not embed a PNG preview of the vector figures",
    )
    parser.add_argument(
        "--no-diagram", action="store_true", help="skip the 2-D interaction diagram"
    )
    parser.add_argument(
        "--pdf", action="store_true",
        help="also write a PDF, if a converter is available (never required)",
    )
    parser.add_argument("--json", action="store_true", help="print what was written as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no summary")
    parser.set_defaults(func=cmd_report_html)

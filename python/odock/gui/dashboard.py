# SPDX-License-Identifier: GPL-3.0-or-later
"""The instrument panels of the workbench.

Everything here is about *reading a run*, not about starting one:

* :class:`RunDashboard` — the live run monitor that sits beside the pose table:
  the phase readout (grid → search → refine → done), the best affinity of the
  session, a ticking elapsed clock and :class:`EnergyTrace`, an
  energy-versus-iteration plot drawn with ``QPainter`` (matplotlib is not a
  dependency of this project).
* :class:`PoseComparison` / :func:`compare_poses` — the question a chemist asks
  when choosing between two binding modes: how far apart are they (with the
  ligand's own symmetry taken into account), how much affinity separates them,
  and which receptor residues does one touch that the other does not.
* :class:`CommandPalette` — Ctrl+K. Its entries are collected *from the menu bar
  at the moment it opens*, so a palette can never go stale when a menu entry is
  added, renamed or translated.
* :class:`SessionStore` / :class:`RecentFiles` / layout presets — the session
  that survives a restart.
* :data:`DARK` / :data:`LIGHT` and :func:`stylesheet` — two themes and two
  densities, each with the viewport clear colour it was designed around.

Honesty note about the trace
----------------------------

The docking kernel reports its refined pose energies **once, at the end of a
run**: ``Docking::run`` holds its state mutex for the whole search, so a
per-iteration energy is not observable from Python, and this module invents no
numbers. The trace therefore plots the *real* data the kernel hands over — one
point per pose, in rank order, for this run and the previous ones — while the
phase strip, the elapsed clock and the status line are genuinely live while the
search is in flight. That is what "readable during a run" honestly means here,
and the panel says so on screen rather than drawing a fictional descent curve.

TODO (a kernel change, deliberately not faked here): a real convergence curve
needs a progress callback out of the Rust search. The hook already exists in
spirit — ``CancelToken`` is polled at every step boundary — so a counter plus an
"best energy so far" atomic published on that poll, surfaced as an optional
``Docking.run(progress=...)`` argument, would let :class:`RunDashboard` append
genuine ``(iteration, energy)`` samples as they happen. Until then the trace
stays as described: real points, no interpolated fiction.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from .i18n import EN as _EN
from .i18n import tr

__all__ = [
    "DARK",
    "DENSITIES",
    "LIGHT",
    "LAYOUT_PRESETS",
    "MEASUREMENT_KINDS",
    "MEASUREMENT_ORDER",
    "PHASES",
    "THEMES",
    "Annotation",
    "Command",
    "CommandPalette",
    "CommandStack",
    "ConsoleInput",
    "ConsoleWidget",
    "Contact",
    "EnergyTrace",
    "FingerprintDiff",
    "InteractionTable",
    "Measurement",
    "MeasurementHistory",
    "PaletteEntry",
    "PhaseRecord",
    "PoseComparison",
    "PoseComparisonWidget",
    "RecentFiles",
    "RmsdResult",
    "RunDashboard",
    "RunTrace",
    "SessionHistory",
    "SessionStore",
    "Theme",
    "angle_degrees",
    "atom_readout",
    "centroid",
    "classify_structure",
    "collect_entries",
    "compare_poses",
    "consolidate_by_residue",
    "contact_map",
    "dihedral_degrees",
    "fingerprint_diff",
    "format_measurement",
    "fuzzy_score",
    "line_plane_angle",
    "measurement_expected_atoms",
    "measurement_kind_label",
    "measurement_unit",
    "measurement_value",
    "measurements_csv",
    "plane_normal",
    "plane_plane_angle",
    "point_plane_distance",
    "rank_entries",
    "stylesheet",
    "symmetry_aware_rmsd",
    "viewport_colors",
]

# ---------------------------------------------------------------------------
# The run model
# ---------------------------------------------------------------------------

#: The pipeline stages the phase readout shows, in order.
PHASES: Tuple[str, ...] = ("grid", "search", "refine", "done")

#: ``phase.*`` translation keys, one per stage.
PHASE_KEYS: Dict[str, str] = {name: f"phase.{name}" for name in PHASES}


def format_duration(seconds: Optional[float]) -> str:
    """``12.4 s`` / ``1:05`` / ``1:02:03``, or an em dash when unknown."""
    if seconds is None:
        return "—"
    try:
        value = float(seconds)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "—"
    if not math.isfinite(value) or value < 0.0:
        return "—"
    if value < 60.0:
        return f"{value:.1f} s"
    whole = int(round(value))
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}:{secs:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def format_energy(value: Optional[float], digits: int = 2) -> str:
    """``-7.12``, or an em dash when the value is unknown."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.{int(digits)}f}"


@dataclass
class PhaseRecord:
    """One stage of a run: when it started and when it ended."""

    name: str
    started: float
    finished: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.finished is None:
            return None
        return max(0.0, self.finished - self.started)

    def to_dict(self) -> dict:
        return {"name": self.name, "started": self.started, "finished": self.finished}

    @classmethod
    def from_dict(cls, payload: dict) -> "PhaseRecord":
        return cls(
            name=str(payload.get("name", "")),
            started=float(payload.get("started", 0.0)),
            finished=(
                None if payload.get("finished") is None else float(payload["finished"])
            ),
        )


@dataclass
class RunTrace:
    """One docking run as the dashboard saw it.

    ``energies`` holds the affinity of every reported pose, in rank order — the
    real numbers the kernel returned. ``iterations`` is ``1..len(energies)`` and
    exists so the plot's x-axis has a name that cannot be misread as a Monte
    Carlo step the kernel never exposed.
    """

    energies: List[float] = field(default_factory=list)
    phases: List[PhaseRecord] = field(default_factory=list)
    elapsed: float = 0.0
    scoring: str = ""
    grid_points: int = 0
    grid_mb: int = 0
    num_tors: float = 0.0
    exhaustiveness: int = 1
    seed: int = 0
    cancelled: bool = False
    label: str = ""

    @property
    def iterations(self) -> List[int]:
        return list(range(1, len(self.energies) + 1))

    @property
    def best(self) -> Optional[float]:
        return min(self.energies) if self.energies else None

    @property
    def worst(self) -> Optional[float]:
        return max(self.energies) if self.energies else None

    @property
    def points(self) -> List[Tuple[int, float]]:
        return list(zip(self.iterations, self.energies))

    def phase(self, name: str) -> Optional[PhaseRecord]:
        for record in self.phases:
            if record.name == name:
                return record
        return None

    def to_dict(self) -> dict:
        return {
            "energies": [float(value) for value in self.energies],
            "phases": [record.to_dict() for record in self.phases],
            "elapsed": float(self.elapsed),
            "scoring": self.scoring,
            "grid_points": int(self.grid_points),
            "grid_mb": int(self.grid_mb),
            "num_tors": float(self.num_tors),
            "exhaustiveness": int(self.exhaustiveness),
            "seed": int(self.seed),
            "cancelled": bool(self.cancelled),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RunTrace":
        return cls(
            energies=[float(v) for v in payload.get("energies", [])],
            phases=[PhaseRecord.from_dict(p) for p in payload.get("phases", [])],
            elapsed=float(payload.get("elapsed", 0.0)),
            scoring=str(payload.get("scoring", "")),
            grid_points=int(payload.get("grid_points", 0)),
            grid_mb=int(payload.get("grid_mb", 0)),
            num_tors=float(payload.get("num_tors", 0.0)),
            exhaustiveness=int(payload.get("exhaustiveness", 1)),
            seed=int(payload.get("seed", 0)),
            cancelled=bool(payload.get("cancelled", False)),
            label=str(payload.get("label", "")),
        )


class SessionHistory:
    """Every run of this session, newest last, capped."""

    def __init__(self, runs: Optional[Iterable[RunTrace]] = None, cap: int = 12) -> None:
        self.cap = max(1, int(cap))
        self.runs: List[RunTrace] = list(runs or ())

    def add(self, run: RunTrace) -> RunTrace:
        self.runs.append(run)
        if len(self.runs) > self.cap:
            del self.runs[: len(self.runs) - self.cap]
        return run

    def clear(self) -> None:
        self.runs = []

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    @property
    def best(self) -> Optional[float]:
        values = [run.best for run in self.runs if run.best is not None]
        return min(values) if values else None

    @property
    def best_run(self) -> Optional[RunTrace]:
        for run in reversed(self.runs):
            if run.best is not None and run.best == self.best:
                return run
        return None

    def energies(self) -> List[float]:
        return [value for run in self.runs for value in run.energies]

    def to_list(self) -> List[dict]:
        return [run.to_dict() for run in self.runs]

    @classmethod
    def from_list(cls, payload: Sequence[dict], cap: int = 12) -> "SessionHistory":
        return cls((RunTrace.from_dict(item) for item in payload or ()), cap=cap)


# ---------------------------------------------------------------------------
# The command palette: fuzzy matching over the menu bar
# ---------------------------------------------------------------------------


def clean_label(text: str) -> str:
    """A menu label without its accelerator and trailing ellipsis."""
    return str(text or "").replace("&", "").rstrip("…").strip()


def fuzzy_score(query: str, text: str) -> Optional[int]:
    """How well ``query`` matches ``text``; ``None`` when it does not.

    A contiguous (case- and space-insensitive) hit always beats a subsequence
    hit, an earlier hit beats a later one, and a hit at a word start beats one
    in the middle — which is what makes ``cmp`` find "Compare poses" and ``go``
    find "Show 3-D grid box" ahead of "Toggle axes".
    """
    q = "".join(str(query or "").lower().split())
    if not q:
        return 0
    target = str(text or "").lower()
    flat = "".join(target.split())
    if not flat:
        return None
    position = flat.find(q)
    if position >= 0:
        score = 1000 - min(position, 400)
        if position == 0:
            score += 120
        elif position == len(target) - len(q):
            score += 40
        if q in target:
            score += 30  # the query spanned no space
        return score
    # Fall back to an ordered subsequence, penalising the gaps it skipped.
    previous = -1
    first = -1
    gaps = 0
    for char in q:
        found = target.find(char, previous + 1)
        if found < 0:
            return None
        if first < 0:
            first = found
        elif previous >= 0:
            gaps += found - previous - 1
        previous = found
    return max(1, 500 - min(gaps, 300) - min(first, 200))


@dataclass(frozen=True)
class PaletteEntry:
    """One runnable leaf action, with the menu path that leads to it."""

    path: str
    text: str
    action: object

    @property
    def shortcut(self) -> str:
        try:
            return self.action.shortcut().toString()
        except Exception:  # pragma: no cover - defensive
            return ""


def collect_entries(menu_bar, *, max_depth: int = 3) -> List[PaletteEntry]:
    """Every actionable :class:`QAction` of ``menu_bar``, in menu order.

    Collected at call time — never cached — so adding, renaming or translating a
    menu entry can never leave the palette stale. Separators, empty labels and
    permanently disabled entries are skipped; checkable entries are kept,
    because "Toggle axes" is exactly the kind of thing a palette is for.
    """
    entries: List[PaletteEntry] = []

    def walk(menu, prefix: List[str], depth: int) -> None:
        for action in menu.actions():
            if action.isSeparator():
                continue
            text = clean_label(action.text())
            if not text:
                continue
            submenu = action.menu()
            if submenu is not None:
                if depth < max_depth:
                    walk(submenu, prefix + [text], depth + 1)
                continue
            if not action.isEnabled() and not action.isCheckable():
                continue
            entries.append(PaletteEntry(" ▸ ".join(prefix + [text]), text, action))

    for action in menu_bar.actions():
        menu = action.menu()
        if menu is None:
            continue
        walk(menu, [clean_label(action.text())], 1)
    return entries


def rank_entries(
    entries: Sequence[PaletteEntry], query: str, limit: Optional[int] = None
) -> List[PaletteEntry]:
    """Entries matching ``query``, best first, order-stable on ties."""
    query = str(query or "").strip()
    if not query:
        ranked = list(entries)
    else:
        scored: List[Tuple[int, int, PaletteEntry]] = []
        for order, entry in enumerate(entries):
            score = fuzzy_score(query, entry.text)
            path_score = fuzzy_score(query, entry.path)
            if score is None:
                score = path_score
            elif path_score is not None:
                score = max(score, path_score - 15)
            if score is None:
                continue
            scored.append((-score, order, entry))
        scored.sort(key=lambda item: (item[0], item[1]))
        ranked = [item[2] for item in scored]
    return ranked if limit is None else ranked[: int(limit)]


class CommandPalette(QtWidgets.QDialog):
    """Ctrl+K: filter and run any menu-bar action.

    The list is rebuilt from the menu bar every time the dialog is shown, so it
    always reflects the current language and the current menu contents.
    """

    def __init__(self, window, parent=None) -> None:
        super().__init__(parent if parent is not None else window)
        self.setObjectName("commandPalette")
        self.setWindowTitle(tr("palette.title"))
        self.setModal(True)
        self._window = window
        self._entries: List[PaletteEntry] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.edit = QtWidgets.QLineEdit()
        self.edit.setObjectName("paletteEdit")
        self.edit.setPlaceholderText(tr("palette.placeholder"))
        self.edit.textChanged.connect(self._refill)
        self.edit.installEventFilter(self)
        layout.addWidget(self.edit)

        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("paletteList")
        self.list.setUniformItemSizes(True)
        self.list.itemActivated.connect(lambda _item: self.run_selected())
        self.list.itemClicked.connect(lambda _item: self.run_selected())
        layout.addWidget(self.list, 1)

        self.hint = QtWidgets.QLabel("")
        self.hint.setObjectName("paletteHint")
        layout.addWidget(self.hint)

        self.resize(520, 380)
        self.refresh()

    # -- contents -----------------------------------------------------------

    def refresh(self) -> None:
        """Re-collect every action from the menu bar, then re-filter."""
        self._entries = collect_entries(self._window.menuBar())
        self._refill(self.edit.text())

    def entries(self) -> List[PaletteEntry]:
        return list(self._entries)

    def matches(self, query: Optional[str] = None) -> List[PaletteEntry]:
        return rank_entries(
            self._entries, self.edit.text() if query is None else query
        )

    def _refill(self, _text: str = "") -> None:
        matches = self.matches()
        self.list.clear()
        for entry in matches:
            label = entry.text
            if entry.shortcut:
                label = f"{entry.text}    ({entry.shortcut})"
            item = QtWidgets.QListWidgetItem(label)
            item.setToolTip(entry.path)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry.path)
            self.list.addItem(item)
        if matches:
            self.list.setCurrentRow(0)
        self.hint.setText(
            tr("palette.hint", n=len(self._entries), shown=len(matches))
            if matches
            else tr("palette.empty")
        )

    # -- running ------------------------------------------------------------

    def selected_entry(self) -> Optional[PaletteEntry]:
        row = self.list.currentRow()
        matches = self.matches()
        if 0 <= row < len(matches):
            return matches[row]
        return None

    def run_selected(self) -> bool:
        """Close the palette and trigger the highlighted action."""
        entry = self.selected_entry()
        if entry is None:
            return False
        self.accept()
        # Deferred by one event-loop turn: the palette is destroyed by `accept`
        # and an action that opens a modal dialog must not do so underneath a
        # widget that is being torn down.
        QtCore.QTimer.singleShot(0, entry.action.trigger)
        return True

    def eventFilter(self, source, event) -> bool:  # noqa: N802 - Qt naming
        if source is self.edit and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Up):
                step = 1 if key == QtCore.Qt.Key.Key_Down else -1
                count = self.list.count()
                if count:
                    row = (self.list.currentRow() + step) % count
                    self.list.setCurrentRow(row)
                return True
            if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                return self.run_selected()
        return super().eventFilter(source, event)


# ---------------------------------------------------------------------------
# Pose comparison: symmetry-aware RMSD and the contact fingerprint
# ---------------------------------------------------------------------------


def _heavy(atoms: Sequence) -> List:
    return [atom for atom in atoms if str(getattr(atom, "element", "")).upper() != "H"]


@dataclass(frozen=True)
class RmsdResult:
    """A symmetric-aware RMSD in two conventions (Å)."""

    in_place: float
    fitted: float
    pairs: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"RmsdResult(in_place={self.in_place:.3f}, fitted={self.fitted:.3f})"


def _hungarian(cost: Sequence[Sequence[float]]) -> List[int]:
    """Minimum-cost assignment (JV / e-maxx), ``cost`` rectangular with n ≤ m.

    Returns, for every row, the column it is assigned to. Written out here so
    that the symmetry-aware RMSD below needs no SciPy.
    """
    rows = len(cost)
    if rows == 0:
        return []
    cols = len(cost[0])
    if rows > cols:
        raise ValueError("the cost matrix must have at least as many columns")
    inf = float("inf")
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * rows
    for j in range(1, cols + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def _kabsch(source, target) -> Tuple["object", "object"]:
    """Rotation and translation that map ``source`` onto ``target``.

    Returns ``(R, t)`` with ``R @ source.T + t ≈ target.T``. A degenerate set
    (fewer than three points, or a singular covariance) falls back to the
    identity rotation, which is the honest answer for a straight line of atoms.
    """
    import numpy as np

    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    centre_src = src.mean(axis=0)
    centre_dst = dst.mean(axis=0)
    a = src - centre_src
    b = dst - centre_dst
    if len(src) < 3:
        return np.eye(3), centre_dst - centre_src
    covariance = b.T @ a
    try:
        u, _s, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:  # pragma: no cover - singular input
        return np.eye(3), centre_dst - centre_src
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    rotation = u @ correction @ vt
    translation = centre_dst - rotation @ centre_src
    return rotation, translation


def _element_groups(elements: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for index, element in enumerate(elements):
        groups.setdefault(str(element).upper(), []).append(index)
    return groups


def _assignment_rmsd(a, b, elements, *, fit: bool, iterations: int = 4) -> Tuple[float, List[int]]:
    """RMSD between ``a`` and ``b`` allowing same-element atoms to swap."""
    import numpy as np

    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    groups = _element_groups(elements)
    for key, rows in groups.items():
        columns = [i for i, e in enumerate(elements) if str(e).upper() == key]
        if len(rows) != len(columns):  # pragma: no cover - checked by the caller
            raise ValueError(f"element {key} appears a different number of times")

    def best_assignment(target: "np.ndarray") -> List[int]:
        chosen = [-1] * len(A)
        for key, rows in groups.items():
            columns = [i for i, e in enumerate(elements) if str(e).upper() == key]
            cost = [
                [float(np.sum((A[i] - target[j]) ** 2)) for j in columns]
                for i in rows
            ]
            local = _hungarian(cost)
            for row, column in zip(rows, local):
                chosen[row] = columns[column]
        return chosen

    assignment = list(range(len(A)))
    if fit:
        for _ in range(max(1, int(iterations))):
            rotation, translation = _kabsch(B[assignment], A)
            moved = (rotation @ B.T).T + translation
            updated = best_assignment(moved)
            if updated == assignment:
                break
            assignment = updated
        rotation, translation = _kabsch(B[assignment], A)
        moved = (rotation @ B.T).T + translation
    else:
        # No superposition: the symmetry correction is the whole point here, so
        # the assignment still has to be optimised on the coordinates as they
        # are.
        moved = B
        assignment = best_assignment(moved)
    total = 0.0
    for index, column in enumerate(assignment):
        delta = A[index] - moved[column]
        total += float(np.dot(delta, delta))
    return math.sqrt(total / max(1, len(A))), assignment


def symmetry_aware_rmsd(
    atoms_a: Sequence,
    atoms_b: Sequence,
    *,
    heavy_only: bool = True,
    fit: bool = True,
) -> RmsdResult:
    """RMSD between two ligands of the same chemistry, symmetry-corrected.

    Two poses of the same molecule are only comparable once the atoms of one are
    allowed to stand in for the symmetry-equivalent atoms of the other: without
    that, a benzene ring rotated by 60° reports a huge RMSD for a pose that is
    physically identical. This computes both conventions:

    * ``in_place`` — no superposition, which is the crystallographic figure of
      merit (the docking must land the ligand in the experimental frame);
    * ``fitted`` — after the optimal rigid superposition, which is the number
      that answers "is this the same binding mode?".

    Only the coordinates are used, so no RDKit is required.
    """
    left = _heavy(atoms_a) if heavy_only else list(atoms_a)
    right = _heavy(atoms_b) if heavy_only else list(atoms_b)
    if not left or not right:
        raise ValueError("cannot compare an empty ligand")
    if len(left) != len(right):
        raise ValueError(
            f"ligands differ in size: {len(left)} atoms against {len(right)}"
        )
    elements_left = [str(getattr(a, "element", "C")).upper() for a in left]
    elements_right = [str(getattr(b, "element", "C")).upper() for b in right]
    if sorted(elements_left) != sorted(elements_right):
        raise ValueError("the two ligands do not have the same elements")
    coords_left = [[a.x, a.y, a.z] for a in left]
    coords_right = [[b.x, b.y, b.z] for b in right]
    in_place, _ = _assignment_rmsd(
        coords_left, coords_right, elements_left, fit=False
    )
    if fit:
        fitted, _ = _assignment_rmsd(
            coords_left, coords_right, elements_left, fit=True
        )
    else:  # pragma: no cover - kept for callers that want one number only
        fitted = in_place
    return RmsdResult(in_place=in_place, fitted=fitted, pairs=len(left))


#: Key that identifies a receptor residue: ``(chain, res_id, res_name)``.
ResidueKey = Tuple[str, int, str]


def residue_key(atom) -> ResidueKey:
    """The residue identity of an atom."""
    return (
        str(getattr(atom, "chain", "") or ""),
        int(getattr(atom, "res_id", 0) or 0),
        str(getattr(atom, "res_name", "") or "").strip(),
    )


def residue_label(key: ResidueKey, *, chain: bool = False) -> str:
    """``TYR337`` (or ``A:TYR337`` when the chain is asked for)."""
    chain_id, res_id, res_name = key
    name = f"{res_name}{res_id}"
    return f"{chain_id}:{name}" if chain and chain_id else name


@dataclass(frozen=True)
class Contact:
    """The closest approach of one receptor residue to the ligand."""

    key: ResidueKey
    distance: float
    ligand_atom: str = ""
    receptor_atom: str = ""

    @property
    def label(self) -> str:
        return residue_label(self.key)

    @property
    def detail(self) -> str:
        if self.ligand_atom and self.receptor_atom:
            return f"{self.receptor_atom}…{self.ligand_atom} {self.distance:.2f} Å"
        return f"{self.distance:.2f} Å"


def contact_map(
    ligand_atoms: Sequence,
    receptor_atoms: Sequence,
    *,
    cutoff: float = 4.5,
    heavy_only: bool = True,
) -> Dict[ResidueKey, Contact]:
    """Every receptor residue within ``cutoff`` Å of the ligand.

    A residue is kept at the distance of its closest ligand atom contact, and
    that atom pair is remembered so the panel can show *which* atoms touch. The
    receptor is binned into a spatial hash of one ``cutoff``-sized cell, so this
    stays linear in the number of atoms for a 3 000-atom protein.
    """
    ligand = _heavy(ligand_atoms) if heavy_only else list(ligand_atoms)
    if not ligand or not receptor_atoms:
        return {}
    limit = float(cutoff)
    cell = max(0.5, limit)
    buckets: Dict[Tuple[int, int, int], List] = {}
    for atom in receptor_atoms:
        index = (
            math.floor(atom.x / cell),
            math.floor(atom.y / cell),
            math.floor(atom.z / cell),
        )
        buckets.setdefault(index, []).append(atom)

    best: Dict[ResidueKey, Contact] = {}
    best_sq: Dict[ResidueKey, float] = {}
    limit_sq = limit * limit
    for ligand_atom in ligand:
        base = (
            math.floor(ligand_atom.x / cell),
            math.floor(ligand_atom.y / cell),
            math.floor(ligand_atom.z / cell),
        )
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    bucket = buckets.get((base[0] + dx, base[1] + dy, base[2] + dz))
                    if not bucket:
                        continue
                    for atom in bucket:
                        ox = atom.x - ligand_atom.x
                        oy = atom.y - ligand_atom.y
                        oz = atom.z - ligand_atom.z
                        distance_sq = ox * ox + oy * oy + oz * oz
                        if distance_sq > limit_sq:
                            continue
                        key = residue_key(atom)
                        if best_sq.get(key, float("inf")) <= distance_sq:
                            continue
                        best_sq[key] = distance_sq
                        best[key] = Contact(
                            key=key,
                            distance=math.sqrt(distance_sq),
                            ligand_atom=str(getattr(ligand_atom, "name", "")),
                            receptor_atom=str(getattr(atom, "name", "")),
                        )
    return best


@dataclass
class FingerprintDiff:
    """Which residues each pose touches, and which differ."""

    shared: List[Tuple[ResidueKey, float, float]] = field(default_factory=list)
    only_a: List[Contact] = field(default_factory=list)
    only_b: List[Contact] = field(default_factory=list)

    @property
    def n_shared(self) -> int:
        return len(self.shared)

    @property
    def n_only_a(self) -> int:
        return len(self.only_a)

    @property
    def n_only_b(self) -> int:
        return len(self.only_b)

    @property
    def identical(self) -> bool:
        return not self.only_a and not self.only_b

    @property
    def union_size(self) -> int:
        return self.n_shared + self.n_only_a + self.n_only_b

    def jaccard(self) -> float:
        """Contact-set overlap, 1.0 when the two fingerprints are identical."""
        union = self.union_size
        return 1.0 if union == 0 else self.n_shared / union


def fingerprint_diff(
    first: Dict[ResidueKey, Contact], second: Dict[ResidueKey, Contact]
) -> FingerprintDiff:
    """Set difference of two :func:`contact_map` results."""
    shared = []
    for key in sorted(set(first) & set(second)):
        shared.append((key, first[key].distance, second[key].distance))
    shared.sort(key=lambda item: (item[1], item[2]))
    only_a = sorted(
        (first[key] for key in set(first) - set(second)), key=lambda c: c.distance
    )
    only_b = sorted(
        (second[key] for key in set(second) - set(first)), key=lambda c: c.distance
    )
    return FingerprintDiff(shared=shared, only_a=only_a, only_b=only_b)


@dataclass
class PoseComparison:
    """The whole answer to "which of these two poses should I believe?"."""

    index_a: int
    index_b: int
    affinity_a: Optional[float]
    affinity_b: Optional[float]
    rmsd: RmsdResult
    diff: FingerprintDiff
    cutoff: float = 4.5

    @property
    def affinity_delta(self) -> Optional[float]:
        """``affinity_b - affinity_a``; negative means the second is stronger."""
        if self.affinity_a is None or self.affinity_b is None:
            return None
        return float(self.affinity_b) - float(self.affinity_a)

    @property
    def same_mode(self) -> bool:
        """Heuristic: within 2 Å after fitting, and mostly the same contacts."""
        return self.rmsd.fitted <= 2.0 and self.diff.jaccard() >= 0.5

    def as_text(self) -> str:
        """The whole answer as plain text, for the clipboard or a report.

        Deliberately the same numbers, in the same order, as the panel: a chemist
        pasting this into a notebook must not get a different story from the one
        on screen.
        """
        first, second = self.index_a + 1, self.index_b + 1
        lines = [
            tr("compare.heading", a=first, b=second),
            f"  {tr('compare.rmsd')}: "
            + tr(
                "compare.rmsd.value",
                fitted=self.rmsd.fitted,
                in_place=self.rmsd.in_place,
                n=self.rmsd.pairs,
            ),
        ]
        delta = self.affinity_delta
        if delta is None:
            lines.append(f"  {tr('compare.delta')}: {tr('compare.value.placeholder')}")
        else:
            lines.append(
                f"  {tr('compare.delta')}: "
                + tr(
                    "compare.delta.value",
                    delta=delta,
                    a=format_energy(self.affinity_a),
                    b=format_energy(self.affinity_b),
                )
                + " "
                + (
                    tr("compare.verdict.same")
                    if self.same_mode
                    else tr("compare.verdict.different")
                )
            )
        lines.append(
            f"  {tr('compare.contacts', cutoff=self.cutoff)}: "
            + tr(
                "compare.contacts.counts",
                shared=self.diff.n_shared,
                only_a=self.diff.n_only_a,
                only_b=self.diff.n_only_b,
            )
        )
        for key, left, right in self.diff.shared:
            lines.append(
                f"    {residue_label(key):<8} {tr('compare.both'):<12} "
                f"{left:.2f} / {right:.2f}"
            )
        for contact in self.diff.only_a:
            lines.append(
                f"    {contact.label:<8} {tr('compare.only_a', index=first):<12} "
                f"{contact.distance:.2f}"
            )
        for contact in self.diff.only_b:
            lines.append(
                f"    {contact.label:<8} {tr('compare.only_b', index=second):<12} "
                f"{contact.distance:.2f}"
            )
        return "\n".join(lines)


def compare_poses(
    first_atoms: Sequence,
    second_atoms: Sequence,
    receptor_atoms: Sequence,
    *,
    index_a: int = 0,
    index_b: int = 1,
    affinity_a: Optional[float] = None,
    affinity_b: Optional[float] = None,
    cutoff: float = 4.5,
) -> PoseComparison:
    """Compare two poses: symmetric-aware RMSD, affinity Δ, contact diff."""
    rmsd = symmetry_aware_rmsd(first_atoms, second_atoms)
    fingerprints = (
        contact_map(first_atoms, receptor_atoms, cutoff=cutoff),
        contact_map(second_atoms, receptor_atoms, cutoff=cutoff),
    )
    return PoseComparison(
        index_a=int(index_a),
        index_b=int(index_b),
        affinity_a=affinity_a,
        affinity_b=affinity_b,
        rmsd=rmsd,
        diff=fingerprint_diff(fingerprints[0], fingerprints[1]),
        cutoff=float(cutoff),
    )


# ---------------------------------------------------------------------------
# Structure classification (drag and drop)
# ---------------------------------------------------------------------------

def looks_like_structure(text: str) -> bool:
    """Whether a block of text starts with a record the viewer understands."""
    head = str(text or "").lstrip()[:24].upper()
    return head.startswith(
        ("ATOM", "HETATM", "ROOT", "MODEL", "REMARK", "TER", "COMPND", "HEADER")
    )


def classify_structure(text: str) -> Optional[str]:
    """``"poses"``, ``"ligand"`` or ``"receptor"`` for a PDBQT/PDB document.

    The viewer draws all three, so a dropped file has to be routed to the right
    loader. The rules are the ones the file itself states, never the file name:
    ``MODEL`` or a ``VINA RESULT`` remark means a pose collection, a ``ROOT``
    block means a flexible ligand, and anything else is a receptor.
    """
    if not looks_like_structure(text):
        return None
    upper = str(text).upper()
    # `MODEL` first: a multi-model document is a pose collection, full stop. A
    # `ROOT` block then proves a flexible ligand, and only after that does a
    # bare `VINA RESULT` remark (a single docked pose) mean poses.
    if "MODEL" in upper:
        return "poses"
    if "\nROOT" in f"\n{upper}" or upper.startswith("ROOT"):
        return "ligand"
    if "VINA RESULT" in upper:
        return "poses"
    return "receptor"


# ---------------------------------------------------------------------------
# Themes and density
# ---------------------------------------------------------------------------

#: The QSS template. ``$name`` placeholders (never ``{}``) because the sheet is
#: full of braces.
_QSS = Template(
    """
    QMainWindow, QWidget { background: $window_bg; color: $window_fg; font-size: $font_size; }
    QGroupBox { border: 1px solid $group_border; border-radius: 5px; margin-top: 9px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; color: $group_title; }
    QPushButton {
        background: $button_bg; border: 1px solid $button_border; border-radius: 4px;
        padding: $pad_v $pad_w;
    }
    QPushButton:hover { background: $button_hover; }
    QPushButton#primary { background: $primary_bg; border-color: $primary_border; font-weight: bold; }
    QPushButton#primary:hover { background: $primary_hover; }
    QPushButton:disabled { color: $muted; }
    QTableWidget, QTreeWidget, QListWidget, QPlainTextEdit, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox {
        background: $input_bg; border: 1px solid $input_border; border-radius: 3px;
        padding: $item_pad; selection-background-color: $sel_bg; selection-color: $sel_fg;
    }
    QTableWidget::item { padding: $item_pad; }
    QHeaderView::section {
        background: $button_bg; color: $muted; border: 0px; border-right: 1px solid $group_border;
        padding: $item_pad;
    }
    QTabWidget::pane { border: 1px solid $group_border; }
    QTabBar::tab {
        background: $tab_bg; padding: $tab_pad_v $tab_pad_w; border: 1px solid $input_border;
    }
    QTabBar::tab:selected { background: $tab_sel; }
    QMenuBar { background: $window_bg; }
    QMenuBar::item:selected, QMenu::item:selected { background: $sel_bg; color: $sel_fg; }
    QMenu { background: $input_bg; border: 1px solid $input_border; padding: 2px; }
    QFrame#toolStrip { background: $strip_bg; border: 1px solid $button_border;
                       border-radius: 6px; }
    QToolButton { color: $toolbutton_fg; padding: 2px $item_pad; }
    QToolButton:hover { background: $button_hover; border-radius: 4px; }
    QProgressBar { border: 1px solid $input_border; border-radius: 3px; text-align: center; }
    QProgressBar::chunk { background: $primary_bg; }
    QSplitter::handle { background: $splitter; }
    QSplitter::handle:hover { background: $splitter_hover; }
    QStatusBar { background: $statusbar; }
    QLabel#dashboardValue { color: $accent; font-size: $value_size; font-weight: bold; }
    QLabel#dashboardCaption, QLabel#dashboardNote, QLabel#paletteHint {
        color: $muted; font-size: $small_size;
    }
    QLabel#compareHeading { color: $group_title; font-weight: bold; }
    QLabel#compareHint { color: $muted; font-size: $small_size; }
    QListWidget#paletteList { font-size: $value_size; padding: 2px; }
    QLineEdit#paletteEdit { font-size: $value_size; padding: $pad_v $pad_w; }
    """
)


@dataclass(frozen=True)
class Theme:
    """A colour scheme, including the one the 3-D view paints itself."""

    name: str
    viewport_clear: Tuple[float, float, float, float]
    viewport_canvas: Tuple[int, int, int]
    palette: Dict[str, str]

    def color(self, key: str) -> str:
        return self.palette[key]

    def brush(self, key: str, alpha: Optional[int] = None) -> QtGui.QColor:
        """A :class:`QColor` from a ``#rrggbb`` palette entry."""
        color = QtGui.QColor(self.palette[key])
        if alpha is not None:
            color.setAlpha(max(0, min(255, int(alpha))))
        return color


DARK = Theme(
    name="dark",
    # The renderer's own clear colour. Deliberately a shade lighter than the Qt
    # chrome so the 3-D area reads as a viewport, not as a hole in the window.
    viewport_clear=(0.086, 0.094, 0.125, 1.0),
    viewport_canvas=(22, 24, 32),
    palette={
        "window_bg": "#10141c",
        "window_fg": "#d8e2ee",
        "group_border": "#232b3a",
        "group_title": "#7fa7d0",
        "button_bg": "#1b2230",
        "button_border": "#2c3648",
        "button_hover": "#243046",
        "primary_bg": "#1d5f8a",
        "primary_border": "#2f86bd",
        "primary_hover": "#24719f",
        "input_bg": "#151b26",
        "input_border": "#26303f",
        "sel_bg": "#1d5f8a",
        "sel_fg": "#f2f7fc",
        "tab_bg": "#151b26",
        "tab_sel": "#1d5f8a",
        "strip_bg": "rgba(18, 24, 34, 190)",
        "toolbutton_fg": "#cfe2f5",
        "splitter": "#232b3a",
        "splitter_hover": "#2f86bd",
        "statusbar": "#0c1016",
        "muted": "#8fa3ba",
        "accent": "#5ec1f5",
        "plot_bg": "#0b0f16",
        "plot_border": "#26303f",
        "plot_grid": "#1d2532",
        "plot_text": "#9fb3c8",
        "plot_line": "#4fc3f7",
        "plot_marker": "#eaf6ff",
        "plot_prev": "#5f7387",
        "plot_best": "#f0b429",
        "plot_live": "#f0b429",
        "phase_pending": "#1b2230",
        "phase_pending_fg": "#7c8ca1",
        "phase_active": "#1d5f8a",
        "phase_active_fg": "#f2f7fc",
        "phase_done": "#1f4636",
        "phase_done_fg": "#8fe0b4",
    },
)

LIGHT = Theme(
    name="light",
    # A light viewport needs a *light* clear colour: a dark render background
    # under a light window chrome looks like a hole punched in the window. It is
    # deliberately a mid neutral grey rather than near-white, so the pale end of
    # the atom palette (white hydrogens, grey carbons) still separates from it.
    viewport_clear=(0.784, 0.804, 0.831, 1.0),
    viewport_canvas=(200, 205, 212),
    palette={
        "window_bg": "#f2f4f8",
        "window_fg": "#1d2733",
        "group_border": "#ccd4e0",
        "group_title": "#1d5f8a",
        "button_bg": "#e7ecf4",
        "button_border": "#bcc7d7",
        "button_hover": "#d7e0ec",
        "primary_bg": "#1d6fa5",
        "primary_border": "#155a86",
        "primary_hover": "#185f8e",
        "input_bg": "#ffffff",
        "input_border": "#c3cddc",
        "sel_bg": "#cfe2f5",
        "sel_fg": "#12212e",
        "tab_bg": "#e7ecf4",
        "tab_sel": "#bcd8ef",
        "strip_bg": "rgba(255, 255, 255, 215)",
        "toolbutton_fg": "#23303f",
        "splitter": "#d4dbe6",
        "splitter_hover": "#7aa9cc",
        "statusbar": "#e4e9f1",
        "muted": "#5b6b7d",
        "accent": "#12557e",
        "plot_bg": "#ffffff",
        "plot_border": "#c3cddc",
        "plot_grid": "#e4e9f1",
        "plot_text": "#4a5a6b",
        "plot_line": "#1d6fa5",
        "plot_marker": "#0d3a55",
        "plot_prev": "#9aa8b8",
        "plot_best": "#b06a00",
        "plot_live": "#b06a00",
        "phase_pending": "#e7ecf4",
        "phase_pending_fg": "#77879b",
        "phase_active": "#1d6fa5",
        "phase_active_fg": "#ffffff",
        "phase_done": "#d6efdd",
        "phase_done_fg": "#1c6b3f",
    },
)

THEMES: Dict[str, Theme] = {"dark": DARK, "light": LIGHT}

#: ``name -> (font size, control padding, tab padding, item padding)``.
DENSITIES: Dict[str, Dict[str, str]] = {
    "comfortable": {
        "font_size": "12px",
        "small_size": "10px",
        "value_size": "15px",
        "pad_v": "4px",
        "pad_w": "9px",
        "tab_pad_v": "5px",
        "tab_pad_w": "12px",
        "item_pad": "3px",
    },
    "compact": {
        "font_size": "11px",
        "small_size": "9px",
        "value_size": "13px",
        "pad_v": "2px",
        "pad_w": "6px",
        "tab_pad_v": "3px",
        "tab_pad_w": "8px",
        "item_pad": "1px",
    },
}


def theme_named(name: str) -> Theme:
    """A theme by name, defaulting to the dark one."""
    return THEMES.get(str(name or "").lower(), DARK)


def density_named(name: str) -> Dict[str, str]:
    """A density by name, defaulting to the comfortable one."""
    return DENSITIES.get(str(name or "").lower(), DENSITIES["comfortable"])


def stylesheet(theme: Theme, density: str = "comfortable") -> str:
    """The window style sheet for one theme and density."""
    values = dict(theme.palette)
    values.update(density_named(density))
    return _QSS.substitute(values)


def viewport_colors(theme: Theme) -> Tuple[Tuple[float, float, float, float], Tuple[int, int, int]]:
    """``(clear_rgba, canvas_rgb)`` for the 3-D view of ``theme``."""
    return theme.viewport_clear, theme.viewport_canvas


# ---------------------------------------------------------------------------
# Session persistence: recent files, the autosaved session, layout presets
# ---------------------------------------------------------------------------

#: Bumped when the on-disk session layout changes incompatibly.
SESSION_VERSION = 1


def default_session_path() -> Path:
    """Where the session is autosaved.

    ``ODOCK_SESSION_FILE`` wins, which is how the tests (and CI) keep the
    developer's own session out of harm's way.
    """
    override = os.environ.get("ODOCK_SESSION_FILE")
    if override:
        return Path(override)
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "OpenDocking" / "session.json"
    return Path.home() / ".opendocking" / "session.json"


class RecentFiles:
    """A most-recent-first list of paths, case-insensitively de-duplicated."""

    def __init__(self, paths: Optional[Iterable[str]] = None, cap: int = 8) -> None:
        self.cap = max(1, int(cap))
        self._paths: List[str] = []
        for path in paths or ():
            self.add(path)

    def add(self, path) -> Optional[str]:
        text = str(path or "").strip()
        if not text:
            return None
        key = os.path.normcase(os.path.abspath(text))
        self._paths = [
            item
            for item in self._paths
            if os.path.normcase(os.path.abspath(item)) != key
        ]
        self._paths.insert(0, text)
        del self._paths[self.cap :]
        return text

    def drop(self, path) -> None:
        key = os.path.normcase(os.path.abspath(str(path)))
        self._paths = [
            item
            for item in self._paths
            if os.path.normcase(os.path.abspath(item)) != key
        ]

    def replace(self, paths: Sequence[str]) -> None:
        """Become ``paths`` **in place**, keeping this object's identity.

        The window holds this instance, so a reload must update it rather than
        hand back a new list the window would never see.
        """
        self._paths = RecentFiles.from_list(paths, cap=self.cap).paths()

    def clear(self) -> None:
        self._paths = []

    def paths(self) -> List[str]:
        return list(self._paths)

    def existing(self) -> List[str]:
        """Only the entries that are still on disk."""
        return [item for item in self._paths if Path(item).is_file()]

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self):
        return iter(self._paths)

    def to_list(self) -> List[str]:
        return self.paths()

    @classmethod
    def from_list(cls, payload: Sequence[str], cap: int = 8) -> "RecentFiles":
        """Rebuild the list *in the order it was written* (newest first).

        :meth:`add` puts its argument at the head, so feeding a stored list
        through it one entry at a time would silently reverse the history.
        """
        out = cls(cap=cap)
        seen = set()
        paths: List[str] = []
        for item in payload or ():
            text = str(item or "").strip()
            if not text:
                continue
            key = os.path.normcase(os.path.abspath(text))
            if key in seen:
                continue
            seen.add(key)
            paths.append(text)
        out._paths = paths[: out.cap]
        return out


class SessionStore:
    """The session file: loaded files, box, engine, selection and layout.

    Every method is defensive: a corrupt or unreadable session must never stop
    the workbench from starting, so failures are swallowed and reported as
    ``None``/``False``.
    """

    def __init__(self, path: Optional[Path] = None, *, cap: int = 8) -> None:
        self.path = Path(path) if path is not None else default_session_path()
        self.recent = RecentFiles(cap=cap)
        self.payload: Optional[dict] = None

    @classmethod
    def default(cls) -> "SessionStore":
        return cls(default_session_path())

    def load(self) -> Optional[dict]:
        try:
            raw = self.path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except Exception:
            self.payload = None
            return None
        if not isinstance(payload, dict):
            self.payload = None
            return None
        self.recent.replace(payload.get("recent", []))
        self.payload = payload
        return payload

    def save(self, payload: dict) -> bool:
        data = dict(payload)
        data["version"] = SESSION_VERSION
        data["saved_at"] = time.time()
        data["recent"] = self.recent.to_list()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            return False
        self.payload = data
        return True

    def clear(self) -> bool:
        self.payload = None
        self.recent.clear()
        try:
            self.path.unlink()
        except FileNotFoundError:
            return True
        except Exception:
            return False
        return True


#: ``name -> layout``. A preset is a whole workspace arrangement, because
#: "docking" and "reading the answer" want different panels open.
LAYOUT_PRESETS: Dict[str, dict] = {
    "docking": {
        "docks": {
            "workspace": True,
            "inspector": True,
            "bottom": True,
            "selection": False,
            "dashboard": True,
            "comparison": False,
        },
        "inspector_tab": 3,
        "pose_stacked": False,
        "raise_dashboard": True,
    },
    "analysis": {
        "docks": {
            "workspace": False,
            "inspector": True,
            "bottom": True,
            "selection": True,
            "dashboard": False,
            "comparison": False,
        },
        "inspector_tab": 0,
        "pose_stacked": False,
        "raise_dashboard": False,
    },
    "compare": {
        "docks": {
            "workspace": False,
            "inspector": False,
            "bottom": True,
            "selection": False,
            "dashboard": True,
            "comparison": True,
        },
        "inspector_tab": 0,
        "pose_stacked": False,
        "raise_dashboard": False,
    },
}


def layout_preset(name: str) -> dict:
    """A copy of a named layout preset; unknown names give the docking one."""
    preset = LAYOUT_PRESETS.get(str(name or "").lower(), LAYOUT_PRESETS["docking"])
    return {
        "docks": dict(preset["docks"]),
        "inspector_tab": int(preset["inspector_tab"]),
        "pose_stacked": bool(preset["pose_stacked"]),
        "raise_dashboard": bool(preset["raise_dashboard"]),
    }


# ---------------------------------------------------------------------------
# The 3-D hover readout
# ---------------------------------------------------------------------------


def atom_readout(atom, *, kind: str = "", index: int = -1) -> List[Tuple[str, str]]:
    """``(label, value)`` rows describing one atom under the cursor.

    The labels are translated; the values are data and stay as they are.
    """
    if atom is None:
        return []
    rows: List[Tuple[str, str]] = []
    if kind:
        rows.append((tr("inspect.source"), tr(f"inspect.source.{kind}")))
    rows.extend(
        [
            (
                tr("inspect.residue"),
                residue_label(residue_key(atom), chain=True),
            ),
            (tr("inspect.atom"), f"{getattr(atom, 'name', '')} #{int(index) + 1}"),
            (tr("inspect.element"), str(getattr(atom, "element", ""))),
            (tr("inspect.ad_type"), str(getattr(atom, "ad_type", ""))),
            (tr("inspect.charge"), f"{float(getattr(atom, 'charge', 0.0)):+.3f}"),
            (
                tr("inspect.position"),
                f"{float(atom.x):.2f}  {float(atom.y):.2f}  {float(atom.z):.2f}",
            ),
        ]
    )
    return rows


def atom_readout_text(atom, *, kind: str = "", index: int = -1) -> str:
    """The same rows as one line, for the status bar."""
    rows = atom_readout(atom, kind=kind, index=index)
    return "  ·  ".join(f"{label} {value}" for label, value in rows)


# ---------------------------------------------------------------------------
# Geometry: the arithmetic behind every measurement
# ---------------------------------------------------------------------------
#
# Plain functions on plain coordinate triples, so the numbers can be checked
# against hand-computed values without a window anywhere near them. Every angle
# is in degrees, every distance in ångström, and the conventions are stated per
# function because a dihedral without its convention is three different numbers.


def _vec(point) -> Tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a):
    length = _norm(a)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def angle_degrees(first, vertex, third) -> float:
    """The angle ``first–vertex–third`` in degrees, 0…180.

    The angle that is *measured at* ``vertex``, which is the middle atom the user
    clicked — the same convention the previous single-distance tool used for its
    atom order.
    """
    a = _sub(_vec(first), _vec(vertex))
    b = _sub(_vec(third), _vec(vertex))
    denominator = _norm(a) * _norm(b)
    if denominator < 1e-12:
        return 0.0
    cosine = max(-1.0, min(1.0, _dot(a, b) / denominator))
    return math.degrees(math.acos(cosine))


def dihedral_degrees(p0, p1, p2, p3) -> float:
    """The ``p0–p1–p2–p3`` torsion in degrees, in ``(-180, 180]``.

    The IUPAC sign convention: looking down the ``p1→p2`` axis, a clockwise
    rotation of the far bond is positive. A planar *syn* arrangement is 0° and a
    planar *anti* arrangement is ±180°.
    """
    b0 = _sub(_vec(p0), _vec(p1))
    b1 = _sub(_vec(p2), _vec(p1))
    b2 = _sub(_vec(p3), _vec(p2))
    b1u = _unit(b1)
    if _norm(b1) < 1e-12:
        return 0.0
    v = _sub(b0, tuple(b1u[axis] * _dot(b0, b1u) for axis in range(3)))
    w = _sub(b2, tuple(b1u[axis] * _dot(b2, b1u) for axis in range(3)))
    x = _dot(v, w)
    y = _dot(_cross(b1u, v), w)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(y, x))


def centroid(points) -> Optional[Tuple[float, float, float]]:
    """The arithmetic mean of ``points`` (``None`` for an empty selection)."""
    if not points:
        return None
    count = float(len(points))
    return (
        sum(_vec(p)[0] for p in points) / count,
        sum(_vec(p)[1] for p in points) / count,
        sum(_vec(p)[2] for p in points) / count,
    )


def plane_normal(points) -> Optional[Tuple[float, float, float]]:
    """A unit normal of the best-fit plane through ``points``.

    Three points define the plane exactly (the cross product of two edges); more
    than three use the smallest singular vector of the centred coordinate
    matrix, which is the plane that minimises the squared out-of-plane distance —
    the honest thing to draw through a ring or a side chain.
    """
    points = [_vec(p) for p in points]
    if len(points) < 3:
        return None
    middle = centroid(points)
    if len(points) == 3:
        normal = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
    else:
        import numpy as np

        centred = np.asarray(
            [[p[0] - middle[0], p[1] - middle[1], p[2] - middle[2]] for p in points],
            dtype=float,
        )
        try:
            _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
            return None
        normal = tuple(float(value) for value in vt[-1])
    length = _norm(normal)
    if length < 1e-12:
        return None
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def plane_plane_angle(first, second) -> Optional[float]:
    """The acute angle between two planes, in degrees, 0…90.

    Computed from the normals as ``arccos(|n1·n2|)``: parallel planes are 0° and
    perpendicular planes are 90°, which is what a chemist means by "the angle
    between these two rings".
    """
    n1 = _vec(first)
    n2 = _vec(second)
    denominator = _norm(n1) * _norm(n2)
    if denominator < 1e-12:
        return None
    cosine = abs(_dot(n1, n2) / denominator)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def line_plane_angle(direction, normal) -> Optional[float]:
    """The acute angle between a line and a plane, in degrees, 0…90.

    ``0`` means the bond lies *in* the plane, ``90`` means it is perpendicular to
    it — the convention `line-plane angle` always has, and the opposite of the
    angle between the line and the *normal*.
    """
    d = _vec(direction)
    n = _vec(normal)
    denominator = _norm(d) * _norm(n)
    if denominator < 1e-12:
        return None
    sine = abs(_dot(_unit(d), _unit(n)))
    return math.degrees(math.asin(max(0.0, min(1.0, sine))))


def point_plane_distance(point, origin, normal) -> Optional[float]:
    """The signed distance from ``point`` to the plane through ``origin``."""
    n = _unit(_vec(normal))
    if _norm(n) < 1e-12:
        return None
    return _dot(_sub(_vec(point), _vec(origin)), n)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

#: Every measurable quantity: ``kind -> atoms it consumes``. ``split`` marks where
#: the second group starts for the compound kinds (plane↔plane, plane↔bond).
MEASUREMENT_KINDS: Dict[str, Dict[str, object]] = {
    "distance": {"atoms": 2, "unit": "Å", "split": None},
    "angle": {"atoms": 3, "unit": "°", "split": None},
    "dihedral": {"atoms": 4, "unit": "°", "split": None},
    "centroid": {"atoms": 1, "unit": "xyz", "split": None},
    "plane": {"atoms": 3, "unit": "normal", "split": None},
    "plane_angle": {"atoms": 6, "unit": "°", "split": 3},
    "plane_bond": {"atoms": 5, "unit": "°", "split": 3},
}

#: The order the measure menu offers them in.
MEASUREMENT_ORDER = (
    "distance",
    "angle",
    "dihedral",
    "centroid",
    "plane",
    "plane_angle",
    "plane_bond",
)


def measurement_unit(kind: str) -> str:
    """The unit of ``kind``: Å, degrees, or a word for the composite answers."""
    return str(MEASUREMENT_KINDS.get(kind, {}).get("unit", ""))


def measurement_expected_atoms(kind: str) -> int:
    """How many atoms ``kind`` consumes before it can be evaluated."""
    return int(MEASUREMENT_KINDS.get(kind, {}).get("atoms", 2))  # type: ignore[arg-type]


def measurement_split(kind: str) -> Optional[int]:
    """Where the second group of a compound measurement begins."""
    split = MEASUREMENT_KINDS.get(kind, {}).get("split")
    return None if split is None else int(split)  # type: ignore[arg-type]


def measurement_value(kind: str, points: Sequence, split: Optional[int] = None):
    """The value of ``kind`` for the coordinates ``points``.

    Returns a float for the numeric kinds and a tuple for the composite ones
    (a centroid is three numbers, a plane is its origin and normal), or ``None``
    when there are not enough points or they are degenerate.
    """
    points = [_vec(point) for point in points]
    if kind == "distance":
        if len(points) < 2:
            return None
        return math.dist(points[0], points[1])
    if kind == "angle":
        if len(points) < 3:
            return None
        return angle_degrees(points[0], points[1], points[2])
    if kind == "dihedral":
        if len(points) < 4:
            return None
        return dihedral_degrees(points[0], points[1], points[2], points[3])
    if kind == "centroid":
        return centroid(points)
    if kind == "plane":
        return plane_normal(points)
    if kind in ("plane_angle", "plane_bond"):
        if split is None:
            split = measurement_split(kind)
        if split is None or len(points) < split + 2:
            return None
        first = plane_normal(points[:split])
        if first is None:
            return None
        if kind == "plane_angle":
            second = plane_normal(points[split:])
            return None if second is None else plane_plane_angle(first, second)
        direction = _sub(points[split + 1], points[split])
        return line_plane_angle(direction, first)
    return None  # pragma: no cover - unknown kind


def format_measurement(kind: str, value) -> str:
    """A value as the user reads it, with its unit and no false precision."""
    unit = measurement_unit(kind)
    if value is None:
        return "—"
    if kind == "centroid" and isinstance(value, tuple):
        return f"{value[0]:.2f}  {value[1]:.2f}  {value[2]:.2f}"
    if kind == "plane" and isinstance(value, tuple):
        return f"normal {value[0]:+.3f} {value[1]:+.3f} {value[2]:+.3f}"
    if not isinstance(value, (int, float)):  # pragma: no cover - defensive
        return str(value)
    if unit == "°":
        return f"{float(value):.2f}°"
    if unit == "Å":
        return f"{float(value):.3f} Å"
    return f"{float(value):.3f}"


@dataclass
class Measurement:
    """One geometric question asked of the scene, with the atoms it asked about.

    ``refs`` are ``("receptor" | "ligand", atom_index)`` pairs — the same
    references the ruler, the selection table and the 3-D picker all produce, so
    the entry paths cannot disagree about which atom was meant.
    """

    kind: str
    refs: List[Tuple[str, int]] = field(default_factory=list)
    label: str = ""
    visible: bool = True
    colour: Optional[Tuple[float, float, float]] = None

    @property
    def split(self) -> Optional[int]:
        return measurement_split(self.kind)

    @property
    def expected_atoms(self) -> int:
        return measurement_expected_atoms(self.kind)

    @property
    def unit(self) -> str:
        return measurement_unit(self.kind)

    def is_complete(self) -> bool:
        return len(self.refs) >= self.expected_atoms

    def value(self, points: Sequence) -> object:
        return measurement_value(self.kind, points, self.split)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "refs": [list(ref) for ref in self.refs],
            "label": self.label,
            "visible": bool(self.visible),
            "colour": None if self.colour is None else list(self.colour),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Measurement":
        colour = payload.get("colour")
        return cls(
            kind=str(payload.get("kind", "distance")),
            refs=[(str(ref[0]), int(ref[1])) for ref in payload.get("refs", [])],
            label=str(payload.get("label", "")),
            visible=bool(payload.get("visible", True)),
            colour=None if not colour else tuple(float(c) for c in colour),
        )


@dataclass
class Annotation:
    """A text label pinned to an atom, a residue or a measurement."""

    text: str
    #: ``("receptor"|"ligand", index)`` | ``("residue", chain, res_id, res_name)``
    #: | ``("measurement", index)`` — one anchor, three spellings.
    anchor: Tuple = ()
    colour: Tuple[float, float, float] = (1.0, 0.85, 0.35)
    visible: bool = True

    ANCHOR_KINDS = ("atom", "residue", "measurement")

    @property
    def kind(self) -> str:
        return str(self.anchor[0]) if self.anchor else "atom"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "anchor": list(self.anchor),
            "colour": list(self.colour),
            "visible": bool(self.visible),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Annotation":
        anchor = payload.get("anchor") or ()
        return cls(
            text=str(payload.get("text", "")),
            anchor=tuple(anchor),
            colour=tuple(float(c) for c in payload.get("colour", (1.0, 0.85, 0.35))),
            visible=bool(payload.get("visible", True)),
        )


def measurements_csv(rows: Sequence[dict]) -> str:
    """The measurement list as CSV, for the clipboard or a spreadsheet.

    The header is written in English with ASCII units (``A``, ``deg``) so the
    file is portable; the values are the same ones the table shows.
    """
    lines = ["kind,atoms,value,unit,label"]
    for row in rows:
        atoms = " ".join(str(item) for item in row.get("atoms", []))
        label = str(row.get("label", "")).replace('"', "'")
        lines.append(
            f'{row.get("kind", "")},"{atoms}",{row.get("value", "")},'
            f'{str(row.get("unit", "")).replace("Å", "A").replace("°", "deg")},"{label}"'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Undo / redo
# ---------------------------------------------------------------------------


@dataclass
class Command:
    """One reversible scene edit.

    ``undo`` and ``redo`` are callables that put the scene (and the panels that
    describe it) back and forward again. ``merge_key`` is what makes a slider
    drag one step: two commands with the same non-``None`` key, pushed close
    together in time, are merged by :meth:`Command.merge_into`.
    """

    name: str
    undo: object
    redo: object
    merge_key: Optional[str] = None
    stamp: float = field(default_factory=time.monotonic)

    def merge_into(self, other: "Command") -> "Command":
        """Fold ``other`` (the newer command) into this one.

        The *oldest* undo is kept — undoing a drag must go back to where the drag
        started — and the newest redo is taken, so redoing lands on the final
        state. The merged step keeps this command's timestamp so a continuing
        drag keeps coalescing.
        """
        return Command(
            name=other.name or self.name,
            undo=self.undo,
            redo=other.redo,
            merge_key=self.merge_key,
            stamp=self.stamp,
        )


class CommandStack:
    """A bounded undo/redo stack with time-windowed coalescing."""

    def __init__(self, cap: int = 100, merge_window: float = 1.2) -> None:
        self.cap = max(1, int(cap))
        self.merge_window = max(0.0, float(merge_window))
        self._undo: List[Command] = []
        self._redo: List[Command] = []

    # -- state --------------------------------------------------------------

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    @property
    def redo_depth(self) -> int:
        return len(self._redo)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_name(self) -> str:
        return self._undo[-1].name if self._undo else ""

    def redo_name(self) -> str:
        return self._redo[-1].name if self._redo else ""

    def commands(self) -> List[Command]:
        return list(self._undo)

    def clear(self) -> None:
        self._undo = []
        self._redo = []

    # -- editing ------------------------------------------------------------

    def push(self, command: Command, *, execute: bool = True) -> Command:
        """Run ``command`` (unless already done) and record it for undo."""
        if execute:
            command.redo()
        if (
            command.merge_key
            and self.merge_window > 0
            and self._undo
            and self._undo[-1].merge_key == command.merge_key
            and abs(command.stamp - self._undo[-1].stamp) <= self.merge_window
        ):
            self._undo[-1] = self._undo[-1].merge_into(command)
        else:
            self._undo.append(command)
            if len(self._undo) > self.cap:
                del self._undo[: len(self._undo) - self.cap]
        # A new edit invalidates the redo branch, exactly like every editor.
        self._redo = []
        return self._undo[-1]

    def undo(self) -> Optional[Command]:
        if not self._undo:
            return None
        command = self._undo.pop()
        command.undo()
        self._redo.append(command)
        return command

    def redo(self) -> Optional[Command]:
        if not self._redo:
            return None
        command = self._redo.pop()
        command.redo()
        self._undo.append(command)
        return command


#: Kinds whose *drawn* lines are reduced to one per receptor residue. The detector
#: enumerates every hydrophobic-carbon pair inside the cut-off — the honest answer
#: to "what touches what" — but seventeen near-parallel grey lines are unreadable,
#: so the drawing shows the closest contact per residue while the table keeps every
#: pair and the legend states both numbers.
CONSOLIDATE_KINDS = ("hydrophobic",)


def consolidate_by_residue(
    interactions: Sequence, receptor: Sequence
) -> Tuple[List, Dict[str, Tuple[int, int]]]:
    """One line per receptor residue for the kinds in :data:`CONSOLIDATE_KINDS`.

    Returns ``(drawn, counts)`` with ``counts[kind] = (detected, drawn)``. Every
    other kind passes through untouched, and the *closest* contact of a residue is
    the one drawn, so the line still shows the atom pair that matters.
    """
    drawn: List = []
    best: Dict[Tuple[str, tuple], object] = {}
    detected: Dict[str, int] = {}
    for item in interactions:
        kind = str(getattr(item, "kind", ""))
        detected[kind] = detected.get(kind, 0) + 1
        if kind not in CONSOLIDATE_KINDS:
            drawn.append(item)
            continue
        key = (kind, _residue_of(item, receptor))
        current = best.get(key)
        if current is None or float(getattr(item, "distance", 1e9)) < float(
            getattr(current, "distance", 1e9)
        ):
            best[key] = item
    drawn.extend(best.values())
    counts: Dict[str, Tuple[int, int]] = {}
    for kind, total in detected.items():
        if kind in CONSOLIDATE_KINDS:
            counts[kind] = (total, sum(1 for key in best if key[0] == kind))
        else:
            counts[kind] = (total, total)
    return drawn, counts


def _residue_of(item, receptor: Sequence) -> tuple:
    """The receptor residue an interaction belongs to, for the grouping above."""
    residue = getattr(item, "residue", None)
    if residue:
        return tuple(residue)
    try:
        atom = receptor[int(getattr(item, "a", -1))]
    except (IndexError, TypeError, ValueError):
        return ("?", -1, "?")
    return (str(atom.chain), int(atom.res_id), str(atom.res_name))


# ---------------------------------------------------------------------------
# The console
# ---------------------------------------------------------------------------


class ConsoleWidget(QtWidgets.QPlainTextEdit):
    """A Python console bound to the live session.

    Why a Python console and not a spawned CLI: ``odock dock`` starts a *new*
    process and cannot see the receptor you have open, so typing command lines
    would mean re-specifying everything on disk and would not replace the mouse.
    A console against the live objects does replace it — it drives the same
    ``window`` the menus drive, so the log, the undo stack and every panel stay in
    sync, and an action that is undoable from a menu is undoable from here.

    The widget owns the *interface* only (prompt, history, completion, captured
    output); the executor is injected by the window, which is what keeps every
    command on the workbench's own code paths.
    """

    #: Prompt for a fresh statement, and for a continued block.
    PROMPT = ">>> "
    CONTINUED = "... "

    def __init__(self, executor, namespace=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("consoleEdit")
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self._executor = executor
        self._namespace = namespace or (lambda: {})
        self._history: List[str] = []
        self._history_index = 0
        self._buffer: List[str] = []
        self._prompt_at = 0
        self._print_banner()
        self._new_prompt()

    # -- the transcript -----------------------------------------------------

    def _write(self, text: str) -> None:
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        self.insertPlainText(text)

    def _print_banner(self) -> None:
        self.appendPlainText(tr("console.banner"))
        self.appendPlainText(tr("console.no_sandbox"))

    def _new_prompt(self) -> None:
        """Start a fresh input line (or continue a block)."""
        prompt = self.CONTINUED if self._buffer else self.PROMPT
        self.appendPlainText(prompt)
        self._prompt_at = self.document().characterCount() - 1
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

    def current_input(self) -> str:
        """The text typed after the current prompt."""
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.setPosition(self._prompt_at, QtGui.QTextCursor.MoveMode.KeepAnchor)
        return cursor.selectedText().replace("\u2029", "\n")

    def prompt_position(self) -> int:
        return self._prompt_at

    # -- running ------------------------------------------------------------

    def submit(self) -> str:
        """Run what is on the prompt line; returns the text that was submitted."""
        line = self.current_input()
        if not line.strip() and not self._buffer:
            self._new_prompt()
            return ""
        self._buffer.append(line)
        source = "\n".join(self._buffer)
        if _needs_continuation(source):
            # A trailing colon (or an unclosed bracket) means the block is not
            # finished: the classic ``for x in y:`` plus an indented body.
            self._new_prompt()
            return line
        self._history.append(source)
        self._history_index = len(self._history)
        self._buffer = []
        output, error = self._executor(source)
        if output:
            self.appendPlainText(output.rstrip("\n"))
        if error:
            self.appendPlainText(error.rstrip("\n"))
        self._new_prompt()
        return source

    def execute(self, source: str) -> str:
        """Run ``source`` as if it had been typed (used by tests and the menus)."""
        self._write(source)
        return self.submit()

    # -- history and completion --------------------------------------------

    def history(self) -> List[str]:
        return list(self._history)

    def recall(self, step: int) -> None:
        """Replace the input line with a history entry (``step`` ±1)."""
        if not self._history:
            return
        index = max(0, min(self._history_index + step, len(self._history)))
        self._history_index = index
        text = "" if index == len(self._history) else self._history[index]
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.setPosition(self._prompt_at, QtGui.QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(text)
        self.setTextCursor(cursor)

    def candidates(self, prefix: str) -> List[str]:
        """Names matching ``prefix``, from the namespace or from an attribute."""
        if not prefix:
            return []
        if "." in prefix:
            head, _, tail = prefix.rpartition(".")
            try:
                value = eval(head, {"__builtins__": {}}, dict(self._namespace()))
            except Exception:
                return []
            names = [name for name in dir(value) if not name.startswith("_")]
            return [f"{head}.{name}" for name in names if name.startswith(tail)]
        return sorted(
            name
            for name in self._namespace()
            if name.startswith(prefix) and not name.startswith("_")
        )

    def complete(self) -> List[str]:
        """Tab: complete the current word, or list the candidates."""
        import re as _re

        text = self.current_input()
        prefix = _re.split(r"[^A-Za-z0-9_.]", text)[-1]
        if not prefix:
            return []
        names = self.candidates(prefix)
        if not names:
            return []
        common = names[0]
        for name in names[1:]:
            while common and not name.startswith(common):
                common = common[:-1]
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if len(common) > len(prefix):
            cursor.insertText(common[len(prefix):])
        else:
            # Ambiguous: list them, like a shell, and leave the line alone.
            self.appendPlainText("  ".join(names))
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)
        return names

    # -- events -------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)  # a literal newline inside the block
                return
            self.submit()
            return
        if key == QtCore.Qt.Key.Key_Up:
            self.recall(-1)
            return
        if key == QtCore.Qt.Key.Key_Down:
            self.recall(1)
            return
        if key == QtCore.Qt.Key.Key_Tab:
            self.complete()
            return
        if key in (QtCore.Qt.Key.Key_Backspace, QtCore.Qt.Key.Key_Left):
            # Never let editing walk back over the prompt: the text after it is
            # what the executor is handed.
            if self.textCursor().position() <= self._prompt_at:
                return
        super().keyPressEvent(event)


def _needs_continuation(source: str) -> bool:
    """Whether a block is unfinished (``:``, an open bracket, a trailing ``\\``)."""
    text = source.rstrip()
    if not text:
        return False
    if text.endswith("\\"):
        return True
    try:
        compile(source + "\n", "<console>", "exec")
    except SyntaxError as exc:
        message = str(exc)
        return (
            "unexpected EOF" in message
            or "was never closed" in message
            or "expected an indented block" in message
        )
    except ValueError:  # pragma: no cover - a null byte and friends
        return False
    return False


class ConsoleInput(QtWidgets.QLineEdit):
    """The console's single input line, living inside the log panel.

    The transcript is the log view itself — output on top, one line to type at the
    bottom — so reading and typing happen in one place instead of two tabs. The
    widget owns only the *interface*: prompt, history, completion and continuation
    state. The executor is injected by the window, which is what keeps every
    command on the workbench's own code paths (log, undo stack, panels).

    Why a Python console and not a spawned CLI: ``odock dock`` starts a *new*
    process and cannot see the receptor you have open, so typing command lines
    would mean re-specifying everything on disk and would not replace the mouse. A
    console against the live objects does replace it.
    """

    #: `>>>` for a fresh statement, `... ` for a continued block.
    PROMPT = ">>> "
    CONTINUED = "... "

    def __init__(self, executor, namespace=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("consoleInput")
        # The prompt is a label beside the line (see the window), so no
        # placeholder text is needed here: it would repeat the prompt.
        self._executor = executor
        self._namespace = namespace or (lambda: {})
        self._history: List[str] = []
        self._history_index = 0
        self._buffer: List[str] = []
        self.returnPressed.connect(self.submit)

    # -- state --------------------------------------------------------------

    def prompt(self) -> str:
        """The prompt to show for the line being typed."""
        return self.CONTINUED if self._buffer else self.PROMPT

    def history(self) -> List[str]:
        return list(self._history)

    def current_input(self) -> str:
        return self.text()

    def pending(self) -> List[str]:
        return list(self._buffer)

    # -- running ------------------------------------------------------------

    def submit(self) -> str:
        """Run what is typed; returns the source that was submitted."""
        line = self.text()
        self.clear()
        if not line.strip() and not self._buffer:
            return ""
        self._buffer.append(line)
        source = "\n".join(self._buffer)
        if _needs_continuation(source):
            # A trailing colon (or an unclosed bracket) means the block is not
            # finished: the classic ``for x in y:`` plus an indented body.
            self.promptChanged.emit(self.prompt())
            return line
        self._history.append(source)
        self._history_index = len(self._history)
        self._buffer = []
        self.promptChanged.emit(self.prompt())
        self._executor(source)
        return source

    def execute(self, source: str) -> str:
        """Run ``source`` as if it had been typed (used by tests and the menus)."""
        self.setText(source)
        return self.submit()

    # -- history and completion --------------------------------------------

    def recall(self, step: int) -> None:
        """Replace the input line with a history entry (``step`` ±1)."""
        if not self._history:
            return
        index = max(0, min(self._history_index + step, len(self._history)))
        self._history_index = index
        self.setText("" if index == len(self._history) else self._history[index])

    def candidates(self, prefix: str) -> List[str]:
        """Names matching ``prefix``, from the namespace or from an attribute."""
        if not prefix:
            return []
        if "." in prefix:
            head, _, tail = prefix.rpartition(".")
            try:
                value = eval(head, {"__builtins__": {}}, dict(self._namespace()))
            except Exception:
                return []
            names = [name for name in dir(value) if not name.startswith("_")]
            return [f"{head}.{name}" for name in names if name.startswith(tail)]
        return sorted(
            name
            for name in self._namespace()
            if name.startswith(prefix) and not name.startswith("_")
        )

    def complete(self) -> List[str]:
        """Tab: complete the current word, or list the candidates in the log."""
        import re as _re

        text = self.current_input()
        prefix = _re.split(r"[^A-Za-z0-9_.]", text)[-1]
        if not prefix:
            return []
        names = self.candidates(prefix)
        if not names:
            return []
        common = names[0]
        for name in names[1:]:
            while common and not name.startswith(common):
                common = common[:-1]
        if len(common) > len(prefix):
            self.setText(text + common[len(prefix):])
        else:
            # Ambiguous: list them (the window prints them into the transcript).
            self.completionHint.emit("  ".join(names))
        return names

    #: Emitted with the candidate list when Tab cannot complete a single name.
    completionHint = QtCore.pyqtSignal(str)

    #: Emitted with the prompt to show: `>>>` or the continuation `... `.
    promptChanged = QtCore.pyqtSignal(str)

    # -- events -------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        key = event.key()
        modifiers = event.modifiers()
        if key == QtCore.Qt.Key.Key_Up:
            self.recall(-1)
            return
        if key == QtCore.Qt.Key.Key_Down:
            self.recall(1)
            return
        if key == QtCore.Qt.Key.Key_Tab:
            self.complete()
            return
        if key == QtCore.Qt.Key.Key_Escape:
            # Escape hands the keyboard back to the 3-D view: the shortcuts that
            # live on the window must keep working while nobody is typing.
            self.editingFinished.emit()
            return
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Ctrl+C/V/X/A belong to the text line; every other Ctrl combination
            # is a workbench shortcut (Ctrl+Z undo, Ctrl+K palette, …) and must
            # reach the window instead of being swallowed here.
            if key not in (
                QtCore.Qt.Key.Key_C,
                QtCore.Qt.Key.Key_V,
                QtCore.Qt.Key.Key_X,
                QtCore.Qt.Key.Key_A,
            ):
                event.ignore()
                return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


def _font(size: int, *, bold: bool = False) -> QtGui.QFont:
    font = QtGui.QFont("Segoe UI", size)
    if bold:
        font.setWeight(QtGui.QFont.Weight.DemiBold)
    return font


class PhaseStrip(QtWidgets.QWidget):
    """``Grid › Search › Refine › Done`` with the active stage lit up."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.theme: Theme = DARK
        self.done: List[str] = []
        self.active: Optional[str] = None
        self.duration: Dict[str, Optional[float]] = {}
        self.setMinimumHeight(30)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    def set_state(
        self,
        done: Sequence[str],
        active: Optional[str],
        duration: Optional[Dict[str, Optional[float]]] = None,
    ) -> None:
        self.done = [name for name in done if name in PHASES]
        self.active = active if active in PHASES else None
        self.duration = dict(duration or {})
        self.update()

    def reset(self) -> None:
        self.done = []
        self.active = None
        self.duration = {}
        self.update()

    def _chips(self) -> List[Tuple[str, str]]:
        chips = []
        for name in PHASES:
            label = tr(PHASE_KEYS[name])
            seconds = self.duration.get(name)
            # A stage shorter than a tenth of a second is not worth a number:
            # "Grid 0.0 s" reads as a broken read-out rather than a fast grid.
            if seconds is not None and seconds >= 0.1:
                label = f"{label} {format_duration(seconds)}"
            chips.append((name, label))
        return chips

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setFont(_font(8))
        metrics = painter.fontMetrics()
        theme = self.theme
        chips = self._chips()
        arrow = 14
        widths = [metrics.horizontalAdvance(text) + 20 for _name, text in chips]
        total = sum(widths) + arrow * (len(chips) - 1)
        if total > self.width():  # pragma: no cover - very narrow dock
            widths = [
                max(28, int(width * self.width() / total)) for width in widths
            ]
        x = 2.0
        height = min(22.0, max(16.0, self.height() - 8.0))
        y = (self.height() - height) / 2.0
        for position, ((name, text), width) in enumerate(zip(chips, widths)):
            if name in self.done:
                fill, ink = theme.color("phase_done"), theme.color("phase_done_fg")
                prefix = "✔ "
            elif name == self.active:
                fill, ink = theme.color("phase_active"), theme.color("phase_active_fg")
                prefix = ""
            else:
                fill = theme.color("phase_pending")
                ink = theme.color("phase_pending_fg")
                prefix = ""
            rect = QtCore.QRectF(x, y, float(width), height)
            painter.setPen(QtGui.QPen(theme.brush("group_border"), 1))
            painter.setBrush(QtGui.QColor(fill))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QtGui.QColor(ink))
            painter.drawText(
                rect,
                int(QtCore.Qt.AlignmentFlag.AlignCenter),
                metrics.elidedText(prefix + text, QtCore.Qt.TextElideMode.ElideRight, int(width) - 8),
            )
            x += width
            if position < len(chips) - 1:
                painter.setPen(QtGui.QColor(theme.palette["muted"]))
                painter.drawText(
                    QtCore.QRectF(x, y, float(arrow), height),
                    int(QtCore.Qt.AlignmentFlag.AlignCenter),
                    "›",
                )
                x += arrow
        painter.end()


class EnergyTrace(QtWidgets.QWidget):
    """Energy against iteration, drawn with :class:`QPainter`.

    The most recent run is drawn bright, the earlier ones dim, and the
    best-so-far of the session is a dashed rule across the plot. While a run is
    in flight the live status line and the previous trace stay on screen, so the
    panel is readable *during* a run — see the module docstring for why a
    per-iteration curve cannot honestly be drawn before the kernel returns.
    """

    MARGIN_LEFT = 52
    MARGIN_RIGHT = 12
    MARGIN_TOP = 14
    MARGIN_BOTTOM = 26

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.theme: Theme = DARK
        self.history: List[RunTrace] = []
        self.status = ""
        self.running = False
        self.setMinimumHeight(110)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.update()

    def set_history(self, history: Sequence[RunTrace]) -> None:
        self.history = [run for run in history if run.energies]
        self.update()

    def set_status(self, text: str, *, running: bool = False) -> None:
        self.status = str(text or "")
        self.running = bool(running)
        self.update()

    def plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self.MARGIN_LEFT,
            self.MARGIN_TOP,
            max(10.0, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT),
            max(10.0, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM),
        )

    def ranges(self) -> Tuple[float, float, float, float]:
        """``(min_iteration, max_iteration, low_energy, high_energy)``.

        ``low`` is the *best* (most negative) affinity and it is drawn at the
        bottom: the axis reads like every other energy plot, so an improving run
        visibly descends.
        """
        values = [run.energies for run in self.history]
        flattened = [value for series in values for value in series]
        iterations = max((len(series) for series in values), default=1)
        if not flattened:
            return 1.0, 2.0, -10.0, 0.0
        low = min(flattened)
        high = max(flattened)
        span = high - low
        if span < 1.0:
            # A single pose, or a set that is almost degenerate: give the axis a
            # readable width instead of amplifying floating-point noise.
            centre = (low + high) / 2.0
            low, high = centre - 0.5, centre + 0.5
        else:
            low -= span * 0.12
            high += span * 0.12
        return 1.0, float(max(2, iterations + (1 if iterations > 2 else 0))), low, high

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        theme = self.theme
        painter.fillRect(self.rect(), theme.brush("plot_bg"))
        rect = self.plot_rect()

        x_min, x_max, low, high = self.ranges()
        span = max(1e-9, high - low)

        def to_x(iteration: float) -> float:
            fraction = (float(iteration) - x_min) / max(1e-9, x_max - x_min)
            return rect.left() + fraction * rect.width()

        def to_y(energy: float) -> float:
            fraction = (float(energy) - low) / span
            return rect.bottom() - fraction * rect.height()

        painter.setPen(QtGui.QPen(theme.brush("plot_border"), 1))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        painter.setFont(_font(7))
        metrics = painter.fontMetrics()
        # -- horizontal grid and energy labels ------------------------------
        ticks = 4
        for step in range(ticks + 1):
            value = low + span * step / ticks
            y = to_y(value)
            painter.setPen(QtGui.QPen(theme.brush("plot_grid"), 1))
            painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            painter.setPen(QtGui.QColor(theme.palette["plot_text"]))
            painter.drawText(
                QtCore.QRectF(0, y - 8, self.MARGIN_LEFT - 6, 16),
                int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter),
                f"{value:.1f}",
            )
        # -- vertical grid and iteration labels ------------------------------
        count = int(round(x_max - x_min))
        stride = max(1, int(math.ceil(count / 8.0)))
        for iteration in range(int(x_min), count + 1, stride):
            x = to_x(iteration)
            painter.setPen(QtGui.QPen(theme.brush("plot_grid"), 1))
            painter.drawLine(
                QtCore.QPointF(x, rect.top()), QtCore.QPointF(x, rect.bottom())
            )
            painter.setPen(QtGui.QColor(theme.palette["plot_text"]))
            painter.drawText(
                QtCore.QRectF(x - 16, rect.bottom() + 4, 32, 14),
                int(QtCore.Qt.AlignmentFlag.AlignHCenter),
                str(iteration),
            )
        painter.setPen(QtGui.QColor(theme.palette["plot_text"]))
        painter.drawText(
            QtCore.QRectF(rect.left(), rect.bottom() + 4, rect.width(), 14),
            int(QtCore.Qt.AlignmentFlag.AlignRight),
            tr("plot.axis.iteration"),
        )
        painter.save()
        painter.translate(11, rect.center().y())
        painter.rotate(-90)
        painter.drawText(
            QtCore.QRectF(-rect.height() / 2, -8, rect.height(), 14),
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            tr("plot.axis.energy"),
        )
        painter.restore()

        series = self.history
        if not series:
            painter.setPen(QtGui.QColor(theme.palette["muted"]))
            painter.setFont(_font(8))
            painter.drawText(
                rect.adjusted(8, 8, -8, -8),
                int(QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap),
                tr("plot.empty"),
            )
        else:
            # The earlier runs recede; the newest one carries the eye.
            for order, run in enumerate(series):
                newest = order == len(series) - 1
                color = theme.brush("plot_line") if newest else theme.brush("plot_prev")
                pen = QtGui.QPen(color, 2.0 if newest else 1.2)
                if not newest:
                    pen.setStyle(QtCore.Qt.PenStyle.DashLine)
                painter.setPen(pen)
                points = [
                    QtCore.QPointF(to_x(iteration), to_y(energy))
                    for iteration, energy in run.points
                ]
                if len(points) == 1:
                    marker = points[0]
                    painter.drawLine(
                        QtCore.QPointF(marker.x(), rect.top()), marker
                    )
                else:
                    painter.drawPolyline(points)
                if newest:
                    painter.setBrush(theme.brush("plot_marker"))
                    painter.setPen(QtGui.QPen(color, 1.0))
                    for point in points:
                        painter.drawEllipse(point, 2.6, 2.6)

            best = min(energy for run in series for energy in run.energies)
            y = to_y(best)
            pen = QtGui.QPen(theme.brush("plot_best"), 1.0)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(rect.left(), y), QtCore.QPointF(rect.right(), y))
            painter.setFont(_font(7))
            painter.drawText(
                QtCore.QRectF(rect.left() + 6, y - 14, 150, 13),
                int(QtCore.Qt.AlignmentFlag.AlignLeft),
                tr("plot.best", value=format_energy(best)),
            )

        if self.status:
            painter.setFont(_font(8))
            painter.setPen(
                QtGui.QColor(
                    theme.palette["plot_live"] if self.running else theme.palette["plot_text"]
                )
            )
            painter.drawText(
                QtCore.QRectF(rect.left() + 6, rect.top() + 4, rect.width() - 12, 16),
                int(QtCore.Qt.AlignmentFlag.AlignRight),
                metrics.elidedText(self.status, QtCore.Qt.TextElideMode.ElideRight, int(rect.width()) - 12),
            )
        painter.end()

    # A stable, testable summary of what is on screen.
    def drawn_points(self) -> int:
        return sum(len(run.energies) for run in self.history)


class MeasurementHistory(QtWidgets.QWidget):
    """Every measurement of the session: type, atoms it asked about, value.

    One row per geometric answer — distance, angle, dihedral, centroid, plane or
    a plane-versus-plane / plane-versus-bond angle — with the atoms named, so the
    table and the overlays in the 3-D view describe the same thing. Copy gives
    the table as text; Copy CSV gives a portable file body.
    """

    clearRequested = QtCore.pyqtSignal()
    copyRequested = QtCore.pyqtSignal()
    csvRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.summary = QtWidgets.QLabel(tr("measure.empty"))
        self.summary.setObjectName("dashboardCaption")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setObjectName("measureTable")
        self.table.setHorizontalHeaderLabels(
            [tr("measure.col.kind"), tr("measure.col.atoms"), tr("measure.col.value")]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_copy = QtWidgets.QPushButton(tr("measure.copy"))
        self.btn_copy.clicked.connect(self.copyRequested.emit)
        self.btn_csv = QtWidgets.QPushButton(tr("measure.csv"))
        self.btn_csv.clicked.connect(self.csvRequested.emit)
        self.btn_clear = QtWidgets.QPushButton(tr("measure.clear"))
        self.btn_clear.clicked.connect(self.clearRequested.emit)
        row.addWidget(self.btn_copy)
        row.addWidget(self.btn_csv)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        layout.addLayout(row)

        self._rows: List[dict] = []

    def rows(self) -> List[dict]:
        return [dict(item) for item in self._rows]

    def set_measurements(self, measurements: Sequence[dict]) -> None:
        """Rows are ``{"kind", "atoms", "value", "unit", "label"}``."""
        self._rows = [dict(item) for item in measurements]
        self.table.setRowCount(len(self._rows))
        for index, item in enumerate(self._rows):
            atoms = item.get("atoms") or []
            if isinstance(atoms, str):
                atoms = [atoms]
            value = item.get("value")
            unit = str(item.get("unit", ""))
            shown = "—" if value in (None, "") else f"{value}{(' ' + unit) if unit else ''}"
            cells = (
                measurement_kind_label(str(item.get("kind", ""))),
                ", ".join(str(atom) for atom in atoms),
                str(shown).strip(),
            )
            for column, text in enumerate(cells):
                entry = QtWidgets.QTableWidgetItem(text)
                entry.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                if item.get("label"):
                    entry.setToolTip(str(item["label"]))
                self.table.setItem(index, column, entry)
        self.table.resizeColumnsToContents()
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        self.summary.setText(
            tr("measure.count", n=len(self._rows))
            if self._rows
            else tr("measure.empty")
        )

    def as_text(self) -> str:
        lines = [
            "\t".join(
                [
                    tr("measure.col.kind"),
                    tr("measure.col.atoms"),
                    tr("measure.col.value"),
                ]
            )
        ]
        for item in self._rows:
            atoms = item.get("atoms") or []
            if isinstance(atoms, str):
                atoms = [atoms]
            unit = str(item.get("unit", ""))
            value = item.get("value")
            shown = "—" if value in (None, "") else f"{value}{(' ' + unit) if unit else ''}"
            lines.append(
                "\t".join(
                    [
                        measurement_kind_label(str(item.get("kind", ""))),
                        ", ".join(str(atom) for atom in atoms),
                        str(shown).strip(),
                    ]
                )
            )
        return "\n".join(lines)

    def as_csv(self) -> str:
        return measurements_csv(self._rows)

    def retranslate(self) -> None:
        self.table.setHorizontalHeaderLabels(
            [tr("measure.col.kind"), tr("measure.col.atoms"), tr("measure.col.value")]
        )
        self.btn_copy.setText(tr("measure.copy"))
        self.btn_csv.setText(tr("measure.csv"))
        self.btn_clear.setText(tr("measure.clear"))
        self._refresh_summary()


def measurement_kind_label(kind: str) -> str:
    """The translated name of a measurement kind (falls back to the code)."""
    key = f"measure.kind.{kind}"
    return tr(key) if key in _EN else str(kind)


class AnnotationTable(QtWidgets.QWidget):
    """The labels pinned to the scene, with where they are pinned.

    Editing happens through the dialog (text and colour); this panel is the list
    that says what exists, where it points, and whether it is currently drawn.
    """

    copyRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("annotationTable")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.heading = QtWidgets.QLabel(tr("annotation.empty"))
        self.heading.setObjectName("dashboardCaption")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setObjectName("annotationRows")
        self.table.setHorizontalHeaderLabels(
            [
                tr("annotation.col.text"),
                tr("annotation.col.anchor"),
                tr("annotation.col.shown"),
            ]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_copy = QtWidgets.QPushButton(tr("measure.copy"))
        self.btn_copy.clicked.connect(self.copyRequested.emit)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        layout.addLayout(row)

        self._rows: List[dict] = []

    def rows(self) -> List[dict]:
        return [dict(item) for item in self._rows]

    def set_annotations(self, annotations: Sequence[dict]) -> None:
        self._rows = [dict(item) for item in annotations]
        self.table.setRowCount(len(self._rows))
        for index, item in enumerate(self._rows):
            colour = item.get("colour") or (1.0, 0.85, 0.35)
            cells = (
                str(item.get("text", "")),
                str(item.get("anchor", "")),
                tr("annotation.yes") if item.get("visible", True) else tr("annotation.no"),
            )
            for column, text in enumerate(cells):
                entry = QtWidgets.QTableWidgetItem(text)
                entry.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                if column == 0:
                    entry.setForeground(
                        QtGui.QColor(
                            int(float(colour[0]) * 255),
                            int(float(colour[1]) * 255),
                            int(float(colour[2]) * 255),
                        )
                    )
                self.table.setItem(index, column, entry)
        self.table.resizeColumnsToContents()
        self.heading.setText(
            tr("annotation.count", n=len(self._rows))
            if self._rows
            else tr("annotation.empty")
        )

    def as_text(self) -> str:
        lines = [
            "\t".join(
                [
                    tr("annotation.col.text"),
                    tr("annotation.col.anchor"),
                    tr("annotation.col.shown"),
                ]
            )
        ]
        for item in self._rows:
            lines.append(
                "\t".join(
                    [
                        str(item.get("text", "")),
                        str(item.get("anchor", "")),
                        tr("annotation.yes")
                        if item.get("visible", True)
                        else tr("annotation.no"),
                    ]
                )
            )
        return "\n".join(lines)

    def retranslate(self) -> None:
        self.table.setHorizontalHeaderLabels(
            [
                tr("annotation.col.text"),
                tr("annotation.col.anchor"),
                tr("annotation.col.shown"),
            ]
        )
        self.btn_copy.setText(tr("measure.copy"))
        self.heading.setText(
            tr("annotation.count", n=len(self._rows))
            if self._rows
            else tr("annotation.empty")
        )


class InteractionTable(QtWidgets.QWidget):
    """What every drawn interaction line connects, as a table.

    This is the answer to "I can see the lines but not what they join": the
    renderer draws a dash from a ligand atom to a receptor atom, and this panel
    names both ends of every dash — kind, receptor atom, ligand atom, distance —
    so a line in the 3-D view can be read off the table (and the other way
    round). It also reports how many are hidden by the View ▸ Interaction lines
    filter, so a filtered view can never be mistaken for "nothing found".
    """

    copyRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("interactionTable")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.heading = QtWidgets.QLabel(tr("interactions.empty"))
        self.heading.setObjectName("dashboardCaption")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setObjectName("interactionRows")
        self.table.setHorizontalHeaderLabels(
            [
                tr("interactions.col.kind"),
                tr("interactions.col.receptor"),
                tr("interactions.col.ligand"),
                tr("interactions.col.distance"),
            ]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_copy = QtWidgets.QPushButton(tr("measure.copy"))
        self.btn_copy.clicked.connect(self.copyRequested.emit)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        layout.addLayout(row)

        self._rows: List[dict] = []
        self._hidden = 0

    # -- contents -----------------------------------------------------------

    def rows(self) -> List[dict]:
        return [dict(row) for row in self._rows]

    def set_interactions(
        self,
        interactions: Sequence,
        receptor: Sequence,
        ligand: Sequence,
        *,
        hidden: int = 0,
        note: str = "",
        shown: bool = True,
    ) -> None:
        """Fill the table from the interaction objects the renderer draws.

        ``hidden`` is how many detected pairs the kind filter keeps out of the
        3-D view; ``note`` spells out any other reduction in the heading's
        tooltip; ``shown`` is whether the view is drawing contacts at all. The
        rows always list every detection, so the table stays the full record even
        while the drawing is switched off.
        """
        self._hidden = int(hidden)
        self._note = str(note)
        self._shown = bool(shown)
        rows: List[dict] = []
        for index, item in enumerate(interactions or ()):
            rows.append(
                {
                    "kind": str(getattr(item, "kind", "")),
                    "receptor": _atom_label(receptor, getattr(item, "a", -1)),
                    "ligand": _atom_label(ligand, getattr(item, "b", -1)),
                    "distance": getattr(item, "distance", None),
                    "detail": str(getattr(item, "detail", "")),
                    "index": index,
                }
            )
        self.table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            distance = row["distance"]
            cells = (
                interaction_label(row["kind"]),
                row["receptor"],
                row["ligand"],
                f"{float(distance):.2f}" if distance is not None else "—",
            )
            for column, text in enumerate(cells):
                entry = QtWidgets.QTableWidgetItem(text)
                entry.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                entry.setToolTip(row["detail"] or f"{row['receptor']} - {row['ligand']}")
                self.table.setItem(index, column, entry)
        self.table.resizeColumnsToContents()
        self._rows = rows
        self._refresh_heading()

    def _refresh_heading(self) -> None:
        if not getattr(self, "_shown", True):
            # The drawing is off: say so and say how to turn it on, instead of
            # showing an empty table that looks like a failure.
            self.heading.setText(tr("interactions.empty"))
            self.heading.setToolTip("")
            return
        if not self._rows:
            self.heading.setText(tr("interactions.empty"))
            self.heading.setToolTip("")
            return
        text = tr("interactions.count", n=len(self._rows))
        if self._hidden:
            text += " · " + tr("interactions.hidden", n=self._hidden)
        self.heading.setText(text)
        # The tooltip is where the *reason* goes: a kind filter and the
        # one-line-per-residue reduction both reduce the drawing, and "17 → 3"
        # has to be explainable without guessing which one did it.
        self.heading.setToolTip(getattr(self, "_note", ""))

    def as_text(self) -> str:
        lines = [
            "\t".join(
                [
                    tr("interactions.col.kind"),
                    tr("interactions.col.receptor"),
                    tr("interactions.col.ligand"),
                    "Å",
                ]
            )
        ]
        for row in self._rows:
            distance = row["distance"]
            lines.append(
                "\t".join(
                    [
                        interaction_label(row["kind"]),
                        row["receptor"],
                        row["ligand"],
                        f"{float(distance):.2f}" if distance is not None else "—",
                    ]
                )
            )
        return "\n".join(lines)

    def retranslate(self) -> None:
        self.table.setHorizontalHeaderLabels(
            [
                tr("interactions.col.kind"),
                tr("interactions.col.receptor"),
                tr("interactions.col.ligand"),
                tr("interactions.col.distance"),
            ]
        )
        self.btn_copy.setText(tr("measure.copy"))
        self._refresh_heading()


def _atom_label(atoms: Sequence, index) -> str:
    """``SER195:OG`` for an atom index, or ``—`` when it does not exist."""
    try:
        atom = atoms[int(index)]
    except (IndexError, TypeError, ValueError):
        return "—"
    name = str(getattr(atom, "name", ""))
    residue = str(getattr(atom, "res_name", ""))
    res_id = getattr(atom, "res_id", "")
    return f"{residue}{res_id}:{name}" if residue else name


def interaction_label(kind: str) -> str:
    """The translated name of an interaction kind (falls back to the code)."""
    key = f"interaction.{kind}"
    return tr(key) if key in _EN else str(kind)


class RunDashboard(QtWidgets.QWidget):
    """The run monitor: phases, live read-outs and the energy trace."""

    #: How often the elapsed clock is refreshed while a run is in flight.
    TICK_MS = 200

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("runDashboard")
        self.theme: Theme = DARK
        self.history = SessionHistory()
        self._started: Optional[float] = None
        self._active_phase: Optional[str] = None
        self._done: List[str] = []
        #: Absolute clock marks the worker reported at each stage boundary, and
        #: the lengths derived from them. Kept apart on purpose: a mark is not a
        #: duration.
        self._marks: Dict[str, Optional[float]] = {}
        self._durations: Dict[str, Optional[float]] = {}
        self._started_wall: Dict[str, float] = {}
        #: Settings of the run in flight, so the engine read-out is populated
        #: before the first pose exists.
        self._pending: Dict[str, object] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.phases = PhaseStrip()
        layout.addWidget(self.phases)

        readouts = QtWidgets.QHBoxLayout()
        readouts.setSpacing(14)
        self.captions: Dict[str, QtWidgets.QLabel] = {}
        self.values: Dict[str, QtWidgets.QLabel] = {}
        for key in ("best", "elapsed", "poses", "grid", "scoring"):
            cell = QtWidgets.QVBoxLayout()
            cell.setSpacing(0)
            caption = QtWidgets.QLabel("")
            caption.setObjectName("dashboardCaption")
            value = QtWidgets.QLabel("—")
            value.setObjectName("dashboardValue")
            cell.addWidget(caption)
            cell.addWidget(value)
            readouts.addLayout(cell)
            self.captions[key] = caption
            self.values[key] = value
        readouts.addStretch(1)
        layout.addLayout(readouts)

        self.trace = EnergyTrace()
        layout.addWidget(self.trace, 1)

        self.note = QtWidgets.QLabel("")
        self.note.setObjectName("dashboardNote")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

        self.retranslate()
        self.reset()

    # -- theme / language ---------------------------------------------------

    def set_theme(self, theme: Theme) -> None:
        self.theme = theme
        self.phases.set_theme(theme)
        self.trace.set_theme(theme)

    def retranslate(self) -> None:
        for key, caption in self.captions.items():
            caption.setText(tr(f"dashboard.{key}"))
        self.note.setText(tr("dashboard.note"))
        self.phases.update()
        self.trace.update()
        self.refresh_readouts()

    # -- run lifecycle ------------------------------------------------------

    def reset(self, *, clear_history: bool = True) -> None:
        """Return to the idle read-out.

        ``clear_history`` is ``False`` when the *widgets* are being rebuilt (a
        language or theme switch): the run history belongs to the window and must
        survive, only the panel around it is new.
        """
        if clear_history:
            self.history.clear()
            self._pending = {}
        self._started = None
        self._active_phase = None
        self._done = []
        self._marks = {}
        self._durations = {}
        self._started_wall = {}
        self.phases.reset()
        self.trace.set_history(list(self.history))
        self.trace.set_status("", running=False)
        self._timer.stop()
        self.refresh_readouts()

    def begin_run(self, *, scoring: str = "", exhaustiveness: int = 0, seed: int = 0) -> None:
        """A run has just been started: light the grid phase and start the clock."""
        self._started = time.perf_counter()
        self._active_phase = "grid"
        self._done = []
        self._marks = {"grid": 0.0}
        self._started_wall = {"grid": self._started}
        self._durations = {}
        self.phases.set_state([], "grid", self._live_durations())
        self.trace.set_status(tr("dashboard.status.starting"), running=True)
        self._timer.start()
        self._pending = {"scoring": scoring, "exhaustiveness": exhaustiveness, "seed": seed}
        self.refresh_readouts()

    def set_phase(self, name: str, elapsed: Optional[float] = None) -> None:
        """A pipeline stage was entered.

        ``elapsed`` is the worker's clock **since the run started**, not a
        duration. Treating that mark as the length of the stage it ends is how
        the search chip once read "40.0 s" for a 35 s search — it was showing the
        *refine* start — and why the grid chip lost its number on the transition
        after it. The length of a finished stage is therefore the difference
        between two marks, and a completed number is kept rather than reset.
        """
        if name not in PHASES:
            return
        if self._active_phase == name:
            return
        previous = self._active_phase
        if previous is not None:
            self._close_phase(previous, elapsed)
            if previous not in self._done:
                self._done.append(previous)
        self._active_phase = None if name == "done" else name
        if name != "done":
            # Every stage starts where the previous one was marked, so a missing
            # mark on the first event of a run cannot invent a duration.
            self._marks[name] = float(elapsed) if elapsed is not None else None
            self._started_wall[name] = time.perf_counter()
        self.phases.set_state(self._done, self._active_phase, self._live_durations())
        self.trace.set_status(tr(PHASE_KEYS[name]), running=name != "done")
        self.refresh_readouts()

    def _close_phase(self, name: str, mark: Optional[float]) -> None:
        """Record how long ``name`` took, from the marks around it."""
        start = self._marks.get(name)
        if start is None or mark is None:
            return
        self._durations[name] = max(0.0, float(mark) - float(start))

    def _live_durations(self) -> Dict[str, Optional[float]]:
        """The finished stages' real lengths, plus the running stage's tick.

        Only the running stage ever shows a number that is still growing, and it
        is measured by this process's own monotonic clock — the same
        ``perf_counter`` the elapsed read-out uses.
        """
        durations: Dict[str, Optional[float]] = {}
        for name in self._done:
            durations[name] = self._durations.get(name)
        active = self._active_phase
        if active is not None:
            started = self._started_wall.get(active)
            durations[active] = (
                None if started is None else max(0.0, time.perf_counter() - started)
            )
        return durations

    def finish_run(self, run: RunTrace) -> RunTrace:
        """Record a completed run and show its trace."""
        for name in PHASES:
            if name in self._done or name == self._active_phase:
                continue
            self._done.append(name)
        self._active_phase = None
        # The trace's phase records are authoritative once they exist: they are
        # the worker's own marks, and `refine` is deliberately a zero-length span
        # there because the kernel does not expose a duration for it.
        for record in run.phases:
            if record.duration is not None:
                self._durations[record.name] = record.duration
        self.phases.set_state(self._done, None, self._live_durations())
        self._timer.stop()
        self._started = None
        # Idempotent: the window may already have added the run to the shared
        # history before handing it over.
        if not any(existing is run for existing in self.history.runs):
            self.history.add(run)
        self.trace.set_history(list(self.history))
        self.trace.set_status(
            tr("dashboard.status.cancelled") if run.cancelled else tr("phase.done"),
            running=False,
        )
        self.refresh_readouts()
        return run

    def fail_run(self, message: str = "") -> None:
        self._timer.stop()
        self._active_phase = None
        self.trace.set_status(
            tr("dashboard.status.failed") if not message else message, running=False
        )
        self.refresh_readouts()

    # -- read-outs ----------------------------------------------------------

    def elapsed(self) -> Optional[float]:
        if self._started is None:
            return None
        return time.perf_counter() - self._started

    def _tick(self) -> None:
        """Refresh the clock and the running stage's chip while a run is going."""
        active = self._active_phase
        if active is not None:
            self.phases.set_state(self._done, active, self._live_durations())
        self.refresh_readouts()

    def refresh_readouts(self) -> None:
        """Recompute the five numbers from the history and the live clock."""
        best = self.history.best
        best_run = self.history.best_run
        if best is not None:
            index = self.history.runs.index(best_run) + 1 if best_run else 0
            self.values["best"].setText(
                f"{format_energy(best)}" + (f"  #{index}" if index else "")
            )
        else:
            self.values["best"].setText("—")
        live = self.elapsed()
        last = self.history.runs[-1] if self.history.runs else None
        seconds = live if live is not None else (last.elapsed if last else None)
        self.values["elapsed"].setText(format_duration(seconds))
        poses = len(last.energies) if last else 0
        self.values["poses"].setText(str(poses))
        if last and last.grid_points:
            self.values["grid"].setText(f"{last.grid_points} pt · {last.grid_mb} MB")
        else:
            self.values["grid"].setText("—")
        pending = getattr(self, "_pending", {}) or {}
        scoring = (last.scoring if last else "") or str(pending.get("scoring", ""))
        if scoring:
            exhaustiveness = last.exhaustiveness if last else int(
                pending.get("exhaustiveness", 0) or 0
            )
            self.values["scoring"].setText(
                f"{scoring} ×{exhaustiveness}" if exhaustiveness else str(scoring)
            )
        else:
            self.values["scoring"].setText("—")


class PoseComparisonWidget(QtWidgets.QWidget):
    """The two-pose panel: RMSD, affinity Δ and the contact fingerprint diff."""

    #: Emitted by the Copy button: the window owns the clipboard.
    copyRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("poseComparison")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.heading = QtWidgets.QLabel(tr("compare.hint"))
        self.heading.setObjectName("compareHint")
        self.heading.setWordWrap(True)
        layout.addWidget(self.heading)

        self.rmsd_label = QtWidgets.QLabel("")
        self.rmsd_label.setObjectName("compareHeading")
        layout.addWidget(self.rmsd_label)
        self.rmsd_value = QtWidgets.QLabel(tr("compare.value.placeholder"))
        layout.addWidget(self.rmsd_value)

        self.delta_label = QtWidgets.QLabel("")
        self.delta_label.setObjectName("compareHeading")
        layout.addWidget(self.delta_label)
        self.delta_value = QtWidgets.QLabel(tr("compare.value.placeholder"))
        layout.addWidget(self.delta_value)

        self.contacts_label = QtWidgets.QLabel("")
        self.contacts_label.setObjectName("compareHeading")
        layout.addWidget(self.contacts_label)
        self.contacts_summary = QtWidgets.QLabel("")
        self.contacts_summary.setObjectName("compareHint")
        self.contacts_summary.setWordWrap(True)
        layout.addWidget(self.contacts_summary)

        self.contacts = QtWidgets.QTableWidget(0, 3)
        self.contacts.setObjectName("compareContacts")
        self.contacts.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.contacts.verticalHeader().setVisible(False)
        layout.addWidget(self.contacts, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_copy = QtWidgets.QPushButton(tr("compare.copy"))
        self.btn_copy.clicked.connect(self.copyRequested.emit)
        row.addWidget(self.btn_copy)
        row.addStretch(1)
        layout.addLayout(row)

        self._comparison: Optional[PoseComparison] = None
        self._detail_widgets = (
            self.rmsd_label,
            self.rmsd_value,
            self.delta_label,
            self.delta_value,
            self.contacts_label,
            self.contacts_summary,
            self.contacts,
            self.btn_copy,
        )
        self.retranslate()
        self.clear()  # nothing to compare yet: no empty read-out rows

    def comparison(self) -> Optional[PoseComparison]:
        return self._comparison

    def _show_details(self, visible: bool) -> None:
        for widget in self._detail_widgets:
            widget.setVisible(bool(visible))

    def clear(self) -> None:
        self._comparison = None
        self.heading.setText(tr("compare.hint"))
        self.rmsd_label.setText("")
        self.delta_label.setText("")
        self.contacts_label.setText("")
        self.rmsd_value.setText("")
        self.delta_value.setText("")
        self.contacts_summary.setText("")
        self.contacts.setRowCount(0)
        self._show_details(False)

    def set_comparison(self, comparison: PoseComparison) -> None:
        self._comparison = comparison
        self._show_details(True)
        first = comparison.index_a + 1
        second = comparison.index_b + 1
        self.heading.setText(tr("compare.heading", a=first, b=second))
        self.rmsd_label.setText(
            tr("compare.rmsd", a=first, b=second)
        )
        self.rmsd_value.setText(
            tr(
                "compare.rmsd.value",
                fitted=comparison.rmsd.fitted,
                in_place=comparison.rmsd.in_place,
                n=comparison.rmsd.pairs,
            )
        )
        self.delta_label.setText(tr("compare.delta", a=first, b=second))
        delta = comparison.affinity_delta
        if delta is None:
            self.delta_value.setText(tr("compare.value.placeholder"))
        else:
            self.delta_value.setText(
                tr(
                    "compare.delta.value",
                    delta=delta,
                    a=format_energy(comparison.affinity_a),
                    b=format_energy(comparison.affinity_b),
                )
                + " "
                + (
                    tr("compare.verdict.same")
                    if comparison.same_mode
                    else tr("compare.verdict.different")
                )
            )
        diff = comparison.diff
        self.contacts_label.setText(tr("compare.contacts", cutoff=comparison.cutoff))
        self.contacts_summary.setText(
            tr(
                "compare.contacts.counts",
                shared=diff.n_shared,
                only_a=diff.n_only_a,
                only_b=diff.n_only_b,
            )
        )
        self._fill_contacts(comparison)

    def _fill_contacts(self, comparison: PoseComparison) -> None:
        rows: List[Tuple[str, str, str]] = []
        for key, first, second in comparison.diff.shared:
            rows.append(
                (
                    residue_label(key),
                    tr("compare.both"),
                    f"{first:.2f} / {second:.2f}",
                )
            )
        for contact in comparison.diff.only_a:
            rows.append(
                (
                    residue_label(contact.key),
                    tr("compare.only_a", index=comparison.index_a + 1),
                    f"{contact.distance:.2f}",
                )
            )
        for contact in comparison.diff.only_b:
            rows.append(
                (
                    residue_label(contact.key),
                    tr("compare.only_b", index=comparison.index_b + 1),
                    f"{contact.distance:.2f}",
                )
            )
        self.contacts.setRowCount(len(rows))
        for index, values in enumerate(rows):
            for column, text in enumerate(values):
                entry = QtWidgets.QTableWidgetItem(text)
                entry.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                self.contacts.setItem(index, column, entry)
        self.contacts.resizeColumnsToContents()

    def retranslate(self) -> None:
        self.contacts.setHorizontalHeaderLabels(
            [
                tr("compare.col.residue"),
                tr("compare.col.which"),
                tr("compare.col.distance"),
            ]
        )
        self.btn_copy.setText(tr("compare.copy"))
        if self._comparison is None:
            self.heading.setText(tr("compare.hint"))
        else:
            self.set_comparison(self._comparison)


# ---------------------------------------------------------------------------
# Small clipboard helpers
# ---------------------------------------------------------------------------


def copy_text_to_clipboard(text: str) -> bool:
    """Put ``text`` on the system clipboard; ``False`` without a QApplication."""
    clipboard = QtWidgets.QApplication.clipboard()
    if clipboard is None:  # pragma: no cover - no GUI application
        return False
    clipboard.setText(str(text))
    return True


def copy_pixmap_to_clipboard(pixmap: QtGui.QPixmap) -> bool:
    """Put an image on the system clipboard."""
    clipboard = QtWidgets.QApplication.clipboard()
    if clipboard is None:  # pragma: no cover - no GUI application
        return False
    clipboard.setPixmap(pixmap)
    return True

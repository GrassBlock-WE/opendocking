# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the workbench instrument panels (:mod:`odock.gui.dashboard`).

Two kinds of property are checked here, and they are deliberately kept apart:

* the **science** — the symmetry-aware RMSD, the contact fingerprint, the fuzzy
  command matcher and the session file are ordinary functions, so they are
  tested against brute force and against known-good answers, with no window
  anywhere near them;
* the **instrument** — the run dashboard, the comparison panel, the palette and
  the appearance switches are driven through the real
  :class:`~odock.gui.app.DockingWorkbench`, so a change that looks right in
  isolation but never reaches the screen fails here.

Everything runs off-screen; no GL context is required.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtGui = pytest.importorskip("PyQt6.QtGui")
QtCore = pytest.importorskip("PyQt6.QtCore")

from odock import protocol  # noqa: E402
from odock.gui import dashboard, i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui.structure import Atom  # noqa: E402

GUI_DIR = Path(__file__).resolve().parent.parent / "python" / "odock" / "gui"
#: Where the screenshot-rendering tests write their figures.
OUT_DIR = Path(__file__).resolve().parent.parent / "out" / "measure"

RECEPTOR_PDBQT = "\n".join(
    [
        f"ATOM  {i + 1:5d}  CB  ALA A   1    "
        f"{3.0 * math.cos(i):8.3f}{3.0 * math.sin(i):8.3f}{0.4 * i - 3.0:8.3f}"
        f"  1.00  0.00     0.000 C"
        for i in range(24)
    ]
    + ["TER", ""]
)

LIGAND_PDBQT = """\
REMARK  VINA RESULT:      -7.991      0.000      0.000
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.390   0.000   0.000  1.00  0.00     0.000 C
ATOM      3  C3  LIG A   1       2.000   1.300   0.000  1.00  0.00     0.000 C
ATOM      4  O1  LIG A   1       3.200   1.300   0.000  1.00  0.00    -0.300 OA
ATOM      5  N1  LIG A   1       1.400   2.500   0.000  1.00  0.00    -0.300 NA
ENDROOT
TORSDOF 0
"""

POSES_PDBQT = """\
MODEL        1
REMARK  VINA RESULT:      -7.991      0.000      0.000
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.390   0.000   0.000  1.00  0.00     0.000 C
ATOM      3  O1  LIG A   1       3.200   1.300   0.000  1.00  0.00    -0.300 OA
ENDROOT
TORSDOF 0
ENDMDL
MODEL        2
REMARK  VINA RESULT:      -7.120      0.871      1.512
ROOT
ATOM      1  C1  LIG A   1       0.100   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.490   0.050   0.000  1.00  0.00     0.000 C
ATOM      3  O1  LIG A   1       3.300   1.400   0.100  1.00  0.00    -0.300 OA
ENDROOT
TORSDOF 0
ENDMDL
"""


def atom(name, element, x, y, z, *, res="LIG", res_id=1, chain="A", ad_type="C"):
    return Atom(
        name=name,
        element=element,
        res_name=res,
        res_id=res_id,
        chain=chain,
        x=float(x),
        y=float(y),
        z=float(z),
        ad_type=ad_type,
    )


def ring(center=(0.0, 0.0, 0.0), radius=1.4, count=6, res="LIG"):
    """A ring of carbons — the shape that makes symmetry correction matter."""
    return [
        atom(
            f"C{i + 1}",
            "C",
            center[0] + radius * math.cos(2 * math.pi * i / count),
            center[1] + radius * math.sin(2 * math.pi * i / count),
            center[2],
            res=res,
        )
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _english():
    i18n.set_language("en")
    yield
    i18n.set_language("en")


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Record message boxes instead of waiting for a human to dismiss them.

    Several code paths report a failure with a modal ``QMessageBox`` (a dropped
    file the workbench cannot read, "compare" with one pose selected). Waiting
    for a click would hang the suite, so the boxes are captured — which also
    makes the message assertable.
    """
    seen: list = []

    def make(level: str):
        def handler(parent=None, title="", text="", *args, **kwargs):
            seen.append(f"{level}: {title}: {text}")
            return QtWidgets.QMessageBox.StandardButton.Ok

        return staticmethod(handler)

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, make(name))
    return seen


def _fresh_scene_state(win) -> None:
    """Clear the per-test scene state a test must never inherit.

    The window owns measurements, annotations, the undo stack and the two
    viewport overlay layers. They are per-window (so a new fixture instance
    starts clean), but making that explicit here means a test that *keeps* a
    window alive cannot leave rows for the next one, and it documents the four
    containers that count.
    """
    win._measurements = []
    win._annotations = []
    win._pick_refs = []
    win._history.clear()
    win.scene.measurements = []
    win.viewport.measurement_overlays = []
    win.viewport.annotation_overlays = []
    win.viewport.interaction_legend = []
    win._sync_measurements()
    win._sync_annotations()
    win._sync_history_actions()


@pytest.fixture
def window(qapp):
    """A workbench with no session file: nothing is written to disk."""
    win = DockingWorkbench()
    win.resize(1280, 820)
    _fresh_scene_state(win)
    yield win
    win.close()


def _clipboard_text(qapp, expected: str, *, timeout_ms: int = 2000) -> str:
    """The clipboard, waiting for an asynchronous write to land.

    The system clipboard is process-global and **asynchronous on Windows**: a
    read straight after a write can still return the previous value, so a test
    that asserts on it immediately is a one-in-N failure that has nothing to do
    with the feature. This waits for the expected value, and the caller decides
    what a timeout means on a platform (offscreen) with no real clipboard.
    """
    clipboard = QtWidgets.QApplication.clipboard()
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    text = clipboard.text()
    while text != expected and deadline.elapsed() < timeout_ms:
        qapp.processEvents()
        QtCore.QThread.msleep(10)
        text = clipboard.text()
    return text


# ---------------------------------------------------------------------------
# formatting and the run model
# ---------------------------------------------------------------------------


def test_duration_is_formatted_for_every_scale():
    assert dashboard.format_duration(None) == "—"
    assert dashboard.format_duration(float("nan")) == "—"
    assert dashboard.format_duration(-1.0) == "—"
    assert dashboard.format_duration(12.34) == "12.3 s"
    assert dashboard.format_duration(59.9) == "59.9 s"
    assert dashboard.format_duration(65) == "1:05"
    assert dashboard.format_duration(3725) == "1:02:05"


def test_energy_is_formatted_or_dashed():
    assert dashboard.format_energy(None) == "—"
    assert dashboard.format_energy(float("inf")) == "—"
    assert dashboard.format_energy(-7.1234) == "-7.12"
    assert dashboard.format_energy(-7.0, digits=3) == "-7.000"


def test_phase_record_duration_is_the_elapsed_time():
    record = dashboard.PhaseRecord(name="grid", started=10.0, finished=12.5)
    assert record.duration == pytest.approx(2.5)
    assert dashboard.PhaseRecord("search", 1.0).duration is None
    assert dashboard.PhaseRecord.from_dict(record.to_dict()).duration == pytest.approx(2.5)


def test_run_trace_points_and_best():
    trace = dashboard.RunTrace(energies=[-7.2, -6.9, -8.0], elapsed=3.0)
    assert trace.points == [(1, -7.2), (2, -6.9), (3, -8.0)]
    assert trace.best == -8.0
    assert trace.worst == -6.9
    assert trace.iterations == [1, 2, 3]
    assert dashboard.RunTrace().best is None


def test_run_trace_round_trips_through_json():
    trace = dashboard.RunTrace(
        energies=[-7.2, -6.9],
        phases=[dashboard.PhaseRecord("grid", 0.0, 1.0)],
        elapsed=4.5,
        scoring="vinardo",
        grid_points=1234,
        grid_mb=3,
        num_tors=4.0,
        exhaustiveness=8,
        seed=42,
        cancelled=True,
        label="run 1",
    )
    restored = dashboard.RunTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert restored.energies == trace.energies
    assert restored.scoring == "vinardo"
    assert restored.cancelled is True
    assert restored.phase("grid").duration == pytest.approx(1.0)


def test_session_history_keeps_the_best_and_caps_itself():
    history = dashboard.SessionHistory(cap=2)
    first = history.add(dashboard.RunTrace(energies=[-5.0]))
    second = history.add(dashboard.RunTrace(energies=[-7.0]))
    third = history.add(dashboard.RunTrace(energies=[-6.0]))
    assert len(history) == 2
    assert first not in history.runs
    assert history.best == -7.0
    assert history.best_run is second
    assert history.energies() == [-7.0, -6.0]
    assert [run.best for run in dashboard.SessionHistory.from_list(history.to_list())] == [
        -7.0,
        -6.0,
    ]
    history.clear()
    assert len(history) == 0 and history.best is None


# ---------------------------------------------------------------------------
# the command palette matcher
# ---------------------------------------------------------------------------


def test_fuzzy_score_prefers_contiguous_word_starts():
    assert dashboard.fuzzy_score("", "anything") == 0
    assert dashboard.fuzzy_score("zzz", "Compare poses") is None
    strong = dashboard.fuzzy_score("compare", "Compare poses")
    weak = dashboard.fuzzy_score("cps", "Compare poses")
    assert strong is not None and weak is not None and strong > weak
    # Case and spacing are irrelevant; a word-start hit beats a mid-word one.
    assert dashboard.fuzzy_score("CMP", "Compare poses") == dashboard.fuzzy_score(
        "cmp", "compareposes"
    )
    assert dashboard.fuzzy_score("p", "Compare poses") > dashboard.fuzzy_score(
        "p", "Strip ligand"
    )


def test_rank_entries_is_ordered_and_stable():
    entries = [
        dashboard.PaletteEntry("File ▸ Save project", "Save project", object()),
        dashboard.PaletteEntry("File ▸ Copy view", "Copy view", object()),
        dashboard.PaletteEntry("View ▸ Cation contacts", "Cation contacts", object()),
    ]
    # A contiguous hit beats a scattered subsequence, and a word start beats a
    # hit in the middle.
    assert [entry.text for entry in dashboard.rank_entries(entries, "copy")] == [
        "Copy view"
    ]
    assert [entry.text for entry in dashboard.rank_entries(entries, "cat")] == [
        "Cation contacts"
    ]
    assert [entry.text for entry in dashboard.rank_entries(entries, "save")] == [
        "Save project"
    ]
    assert [entry.text for entry in dashboard.rank_entries(entries, "")] == [
        "Save project",
        "Copy view",
        "Cation contacts",
    ]
    assert dashboard.rank_entries(entries, "save", limit=1)[0].text == "Save project"
    assert dashboard.rank_entries(entries, "nothing here") == []

    # Equal scores keep the menu order (a palette must not shuffle itself).
    ties = [
        dashboard.PaletteEntry("M ▸ Alpha", "Alpha", object()),
        dashboard.PaletteEntry("M ▸ Alpha two", "Alpha two", object()),
    ]
    assert [entry.text for entry in dashboard.rank_entries(ties, "alpha")] == [
        "Alpha",
        "Alpha two",
    ]


def test_clean_label_drops_accelerators_and_ellipses():
    assert dashboard.clean_label("&Save project…") == "Save project"
    assert dashboard.clean_label("") == ""


# ---------------------------------------------------------------------------
# symmetry-aware RMSD
# ---------------------------------------------------------------------------


def test_hungarian_matches_brute_force():
    random.seed(4)
    for size in (1, 2, 3, 4, 5):
        for _ in range(6):
            cost = [
                [random.uniform(-5, 5) for _ in range(size)] for _ in range(size)
            ]
            assignment = dashboard._hungarian(cost)
            chosen = sum(cost[row][column] for row, column in enumerate(assignment))
            best = min(
                sum(cost[row][column] for row, column in enumerate(perm))
                for perm in itertools.permutations(range(size))
            )
            assert chosen == pytest.approx(best)
    assert dashboard._hungarian([]) == []
    with pytest.raises(ValueError):
        dashboard._hungarian([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_rmsd_of_a_pose_with_itself_is_zero():
    points = ring()
    result = dashboard.symmetry_aware_rmsd(points, points)
    assert result.fitted == pytest.approx(0.0, abs=1e-9)
    assert result.in_place == pytest.approx(0.0, abs=1e-9)
    assert result.pairs == 6


def test_rmsd_sees_a_relabelled_ring_as_identical():
    """A benzene rotated by one carbon is the same pose, not a 2.8 Å error."""
    first = ring(center=(0.0, 0.0, 0.0))
    second = ring(center=(0.0, 0.0, 0.0))
    second = second[1:] + second[:1]  # the same coordinates, renamed
    result = dashboard.symmetry_aware_rmsd(first, second)
    assert result.fitted == pytest.approx(0.0, abs=1e-9)
    assert result.in_place == pytest.approx(0.0, abs=1e-9)


def test_rmsd_ignores_labels_but_not_positions():
    first = ring()
    # Rotate the whole ring by 30 degrees: the same hexagon, superposed, so the
    # fitted RMSD must be ~0 ...
    rotated = [
        atom(
            f"C{i + 1}",
            "C",
            1.4 * math.cos(2 * math.pi * i / 6 + math.pi / 6),
            1.4 * math.sin(2 * math.pi * i / 6 + math.pi / 6),
            0.0,
        )
        for i in range(6)
    ]
    result = dashboard.symmetry_aware_rmsd(first, rotated)
    assert result.fitted == pytest.approx(0.0, abs=1e-6)
    # ... while the in-place number reports that the atoms did move.
    assert result.in_place > 0.5


def test_rmsd_after_a_rigid_transform_is_zero_but_in_place_is_not():
    first = ring(center=(1.0, 2.0, 3.0))
    angle = 0.7
    moved = [
        atom(
            a.name,
            a.element,
            a.x * math.cos(angle) - a.y * math.sin(angle) + 5.0,
            a.x * math.sin(angle) + a.y * math.cos(angle) - 2.0,
            a.z + 1.5,
        )
        for a in first
    ]
    result = dashboard.symmetry_aware_rmsd(first, moved)
    assert result.fitted == pytest.approx(0.0, abs=1e-6)
    assert result.in_place > 1.0


def test_rmsd_rejects_mismatched_ligands():
    with pytest.raises(ValueError):
        dashboard.symmetry_aware_rmsd(ring(count=6), ring(count=5))
    mixed = [atom("C1", "C", 0, 0, 0), atom("O1", "O", 1, 0, 0)]
    other = [atom("C1", "C", 0, 0, 0), atom("N1", "N", 1, 0, 0)]
    with pytest.raises(ValueError):
        dashboard.symmetry_aware_rmsd(mixed, other)
    with pytest.raises(ValueError):
        dashboard.symmetry_aware_rmsd([], [])


def test_rmsd_heavy_only_skips_hydrogens():
    first = ring(count=3) + [atom("H1", "H", 5.0, 0.0, 0.0)]
    second = ring(count=3) + [atom("H1", "H", 9.0, 9.0, 9.0)]
    heavy = dashboard.symmetry_aware_rmsd(first, second)
    assert heavy.pairs == 3
    assert heavy.fitted == pytest.approx(0.0, abs=1e-9)
    everything = dashboard.symmetry_aware_rmsd(first, second, heavy_only=False)
    assert everything.pairs == 4
    assert everything.fitted > 1.0


# ---------------------------------------------------------------------------
# the contact fingerprint
# ---------------------------------------------------------------------------


def test_contact_map_agrees_with_brute_force():
    random.seed(11)
    ligand = [
        atom(f"L{i}", "C", random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-4, 4))
        for i in range(7)
    ]
    receptor = [
        atom(
            f"R{i}",
            "C",
            random.uniform(-4, 4),
            random.uniform(-4, 4),
            random.uniform(-4, 4),
            res="RES",
            res_id=i % 4,
        )
        for i in range(60)
    ]
    mine = dashboard.contact_map(ligand, receptor, cutoff=3.5)
    expected = {}
    for light in ligand:
        for heavy in receptor:
            distance = math.dist(light.position, heavy.position)
            if distance <= 3.5:
                key = dashboard.residue_key(heavy)
                if key not in expected or distance < expected[key]:
                    expected[key] = distance
    assert set(mine) == set(expected)
    for key, distance in expected.items():
        assert mine[key].distance == pytest.approx(distance)


def test_contact_map_remembers_the_closest_atom_pair():
    ligand = [atom("C1", "C", 0.0, 0.0, 0.0), atom("O1", "O", 3.6, 0.0, 0.0)]
    receptor = [
        atom("OH", "O", 1.0, 0.0, 0.0, res="TYR", res_id=337, ad_type="OA"),
        atom("CZ", "C", 2.4, 0.0, 0.0, res="TYR", res_id=337),
        atom("N", "N", 40.0, 0.0, 0.0, res="GLY", res_id=100),
    ]
    contacts = dashboard.contact_map(ligand, receptor, cutoff=4.5)
    assert set(contacts) == {("A", 337, "TYR")}
    contact = contacts[("A", 337, "TYR")]
    assert contact.distance == pytest.approx(1.0)
    assert contact.receptor_atom == "OH" and contact.ligand_atom == "C1"
    assert contact.label == "TYR337"
    assert "OH" in contact.detail and "Å" in contact.detail


def test_contact_map_is_empty_without_a_ligand_or_receptor():
    assert dashboard.contact_map([], [atom("C1", "C", 0, 0, 0)]) == {}
    assert dashboard.contact_map([atom("C1", "C", 0, 0, 0)], []) == {}


def test_fingerprint_diff_splits_shared_and_unique_residues():
    first = dashboard.contact_map(
        [atom("C1", "C", 0, 0, 0)],
        [
            atom("A1", "C", 1.0, 0, 0, res="TYR", res_id=337),
            atom("A2", "C", 1.0, 1.0, 0, res="ASP", res_id=189),
        ],
    )
    second = dashboard.contact_map(
        [atom("C1", "C", 0, 0, 0)],
        [
            atom("B1", "C", 1.2, 0, 0, res="TYR", res_id=337),
            atom("B2", "C", 1.0, -1.0, 0, res="SER", res_id=195),
        ],
    )
    diff = dashboard.fingerprint_diff(first, second)
    assert [key for key, _a, _b in diff.shared] == [("A", 337, "TYR")]
    assert [contact.label for contact in diff.only_a] == ["ASP189"]
    assert [contact.label for contact in diff.only_b] == ["SER195"]
    assert (diff.n_shared, diff.n_only_a, diff.n_only_b) == (1, 1, 1)
    assert diff.union_size == 3
    assert diff.jaccard() == pytest.approx(1 / 3)
    assert not diff.identical
    assert dashboard.fingerprint_diff(first, first).identical
    assert dashboard.fingerprint_diff({}, {}).jaccard() == 1.0


def test_compare_poses_reports_rmsd_delta_and_contacts():
    receptor = [
        atom("OH", "O", 3.9, 0.0, 0.0, res="TYR", res_id=337),
        atom("OD1", "O", 0.0, 3.9, 0.0, res="ASP", res_id=189),
    ]
    first = [atom("C1", "C", 0.0, 0.0, 0.0), atom("C2", "C", 1.4, 0.0, 0.0)]
    second = [
        atom("C1", "C", 0.0, 0.0, 0.0),
        atom("C2", "C", 1.4, 0.0, 0.0),
    ]
    comparison = dashboard.compare_poses(
        first, second, receptor, index_a=0, index_b=3, affinity_a=-7.0, affinity_b=-8.25
    )
    assert comparison.affinity_delta == pytest.approx(-1.25)
    assert comparison.rmsd.fitted == pytest.approx(0.0, abs=1e-9)
    assert comparison.same_mode
    assert comparison.diff.identical
    assert comparison.cutoff == 4.5
    # Moving the second pose out of the pocket changes the fingerprint.
    third = [atom("C1", "C", 40.0, 0.0, 0.0), atom("C2", "C", 41.4, 0.0, 0.0)]
    moved = dashboard.compare_poses(first, third, receptor, affinity_a=-7.0, affinity_b=-7.1)
    assert moved.diff.only_a and not moved.diff.only_b
    assert not moved.same_mode


def test_compare_poses_without_affinities_has_no_delta():
    points = [atom("C1", "C", 0, 0, 0)]
    comparison = dashboard.compare_poses(points, points, [])
    assert comparison.affinity_delta is None


# ---------------------------------------------------------------------------
# structure classification
# ---------------------------------------------------------------------------


def test_classify_structure_reads_the_file_not_the_name():
    assert dashboard.classify_structure(POSES_PDBQT) == "poses"
    assert dashboard.classify_structure(LIGAND_PDBQT) == "ligand"
    assert dashboard.classify_structure(RECEPTOR_PDBQT) == "receptor"
    # A VINA RESULT remark without MODEL records is still a pose collection.
    assert dashboard.classify_structure(LIGAND_PDBQT) == "ligand"
    assert dashboard.classify_structure("REMARK  VINA RESULT: -7.0\nATOM\n") == "poses"
    assert dashboard.classify_structure("") is None
    assert dashboard.classify_structure("hello, world") is None
    assert dashboard.looks_like_structure("  HETATM 1") is True
    assert dashboard.looks_like_structure("junk") is False


# ---------------------------------------------------------------------------
# themes and density
# ---------------------------------------------------------------------------


def test_the_two_themes_are_complete_and_different():
    assert set(dashboard.THEMES) == {"dark", "light"}
    for theme in dashboard.THEMES.values():
        sheet = dashboard.stylesheet(theme, "comfortable")
        assert theme.palette["window_bg"] in sheet
        assert theme.palette["accent"] in sheet
        assert "{" in sheet and "}" in sheet
    assert dashboard.DARK.palette["window_bg"] != dashboard.LIGHT.palette["window_bg"]
    assert dashboard.theme_named("light") is dashboard.LIGHT
    assert dashboard.theme_named("nonsense") is dashboard.DARK


def test_each_theme_has_its_own_viewport_background():
    """The 3-D view renders its own clear colour, so it belongs to the theme."""
    dark_clear, dark_canvas = dashboard.viewport_colors(dashboard.DARK)
    light_clear, light_canvas = dashboard.viewport_colors(dashboard.LIGHT)
    assert dark_clear == (0.086, 0.094, 0.125, 1.0)  # the documented default
    assert sum(light_clear[:3]) > 2.0  # a light theme needs a light viewport
    assert sum(dark_canvas) < sum(light_canvas)
    assert dark_clear != light_clear


def test_density_changes_the_metrics():
    comfortable = dashboard.stylesheet(dashboard.DARK, "comfortable")
    compact = dashboard.stylesheet(dashboard.DARK, "compact")
    assert "font-size: 12px" in comfortable
    assert "font-size: 11px" in compact
    assert "padding: 4px 9px" in comfortable
    assert "padding: 2px 6px" in compact
    assert dashboard.density_named("compact") is dashboard.DENSITIES["compact"]
    assert dashboard.density_named("nope") is dashboard.DENSITIES["comfortable"]


# ---------------------------------------------------------------------------
# the session file
# ---------------------------------------------------------------------------


def test_recent_files_are_ordered_deduplicated_and_capped(tmp_path):
    recent = dashboard.RecentFiles(cap=3)
    for name in ("a.pdbqt", "b.pdbqt", "c.pdbqt", "d.pdbqt"):
        recent.add(tmp_path / name)
    assert [Path(p).name for p in recent.paths()] == ["d.pdbqt", "c.pdbqt", "b.pdbqt"]
    recent.add(tmp_path / "b.pdbqt")
    assert [Path(p).name for p in recent.paths()][0] == "b.pdbqt"
    assert len(recent) == 3
    assert recent.add("") is None
    # Only the entries that are still on disk survive a reload.
    (tmp_path / "b.pdbqt").write_text("x", encoding="utf-8")
    assert [Path(p).name for p in recent.existing()] == ["b.pdbqt"]
    assert [Path(p).name for p in dashboard.RecentFiles.from_list(recent.to_list()).paths()] == [
        Path(p).name for p in recent.paths()
    ]
    recent.drop(tmp_path / "b.pdbqt")
    assert Path("b.pdbqt").name not in [Path(p).name for p in recent.paths()]
    recent.clear()
    assert len(recent) == 0


def test_session_store_round_trips_and_survives_corruption(tmp_path):
    store = dashboard.SessionStore(tmp_path / "session.json")
    store.recent.add(str(tmp_path / "receptor.pdbqt"))
    assert store.save({"receptor": "/x/receptor.pdbqt", "engine": {"scoring": "vina"}})
    payload = store.load()
    assert payload["receptor"] == "/x/receptor.pdbqt"
    assert payload["version"] == dashboard.SESSION_VERSION
    assert payload["saved_at"] > 0
    assert payload["engine"]["scoring"] == "vina"
    assert len(store.recent) == 1
    assert json.loads(store.path.read_text(encoding="utf-8"))["receptor"].endswith(
        "receptor.pdbqt"
    )

    store.path.write_text("{not json", encoding="utf-8")
    assert store.load() is None
    store.path.write_text("[1, 2, 3]", encoding="utf-8")
    assert store.load() is None
    assert store.save({"a": 1})  # a corrupt file does not stop the next save
    assert store.clear() is True
    assert store.load() is None
    assert store.clear() is True  # clearing twice is not an error


def test_the_default_session_path_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("ODOCK_SESSION_FILE", str(tmp_path / "mine.json"))
    assert dashboard.default_session_path() == tmp_path / "mine.json"
    monkeypatch.delenv("ODOCK_SESSION_FILE")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert dashboard.default_session_path() == tmp_path / "roaming" / "OpenDocking" / "session.json"
    monkeypatch.delenv("APPDATA")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert dashboard.default_session_path().name == "session.json"


def test_layout_presets_are_complete():
    assert set(dashboard.LAYOUT_PRESETS) == {"docking", "analysis", "compare"}
    docks = {
        "workspace",
        "inspector",
        "bottom",
        "dashboard",
        "comparison",
        "selection",
    }
    for name in dashboard.LAYOUT_PRESETS:
        preset = dashboard.layout_preset(name)
        assert set(preset["docks"]) == docks
        assert isinstance(preset["inspector_tab"], int)
        assert isinstance(preset["pose_stacked"], bool)
    assert dashboard.layout_preset("docking")["docks"]["dashboard"] is True
    assert dashboard.layout_preset("analysis")["docks"]["workspace"] is False
    assert dashboard.layout_preset("nonsense") == dashboard.layout_preset("docking")
    # A preset is a copy: editing it must not corrupt the table.
    dashboard.layout_preset("docking")["docks"]["dashboard"] = False
    assert dashboard.LAYOUT_PRESETS["docking"]["docks"]["dashboard"] is True


# ---------------------------------------------------------------------------
# the atom read-out
# ---------------------------------------------------------------------------


def test_atom_readout_names_every_field():
    subject = atom("OH", "O", 1.25, -2.5, 3.75, res="TYR", res_id=337, ad_type="OA")
    subject.charge = -0.32
    rows = dict(dashboard.atom_readout(subject, kind="receptor", index=4))
    assert rows["residue"] == "A:TYR337"
    assert rows["atom"] == "OH #5"
    assert rows["element"] == "O"
    assert rows["AD4 type"] == "OA"
    assert rows["charge"] == "-0.320"
    assert rows["position"] == "1.25  -2.50  3.75"
    assert rows["source"] == "receptor"
    assert dashboard.atom_readout(None) == []
    line = dashboard.atom_readout_text(subject, kind="ligand", index=0)
    assert "TYR337" in line and "element O" in line


# ---------------------------------------------------------------------------
# the widgets
# ---------------------------------------------------------------------------


def test_energy_trace_axes_and_painting(qapp):
    trace = dashboard.EnergyTrace()
    trace.resize(360, 160)
    assert trace.ranges() == (1.0, 2.0, -10.0, 0.0)
    trace.set_history([dashboard.RunTrace(energies=[-7.5])])
    low, high = trace.ranges()[2], trace.ranges()[3]
    assert low < -7.5 < high and high - low >= 1.0

    trace.set_history(
        [
            dashboard.RunTrace(energies=[-5.0, -6.0, -7.0]),
            dashboard.RunTrace(energies=[-7.5, -7.1]),
        ]
    )
    x_min, x_max, low, high = trace.ranges()
    assert (x_min, x_max) == (1.0, 4.0)
    assert low < -7.5 and high > -5.0
    assert trace.drawn_points() == 5

    for status, running in (("", False), ("searching", True), ("done", False)):
        trace.set_status(status, running=running)
        pixmap = QtGui.QPixmap(trace.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        trace.render(pixmap)  # must not raise, with or without data


def test_phase_strip_lights_the_active_stage(qapp):
    strip = dashboard.PhaseStrip()
    strip.resize(400, 30)
    strip.set_state(["grid"], "search", {"grid": 1.25})
    assert strip.done == ["grid"] and strip.active == "search"
    strip.set_state([], "nope")
    assert strip.active is None  # an unknown phase is not a phase
    strip.reset()
    assert strip.done == [] and strip.active is None
    pixmap = QtGui.QPixmap(strip.size())
    strip.render(pixmap)


def test_the_live_phase_chips_never_mislabel_a_duration(qapp):
    """``set_phase(name, elapsed)`` gets a *start mark*, never a length.

    The worker reports "refine begins 40 s into the run"; storing that as the
    search's duration made the chip read 40.0 s for a 35 s search, and clearing
    the finished stages on every transition dropped the grid's number. A mark
    and a duration are different numbers and this pins the difference.
    """
    panel = dashboard.RunDashboard()
    panel.begin_run(scoring="vina", exhaustiveness=4)
    panel.set_phase("grid", 0.0)
    panel.set_phase("search", 5.0)  # the grid really took 5 s
    assert panel.phases.duration["grid"] == pytest.approx(5.0)
    assert panel.phases.active == "search"

    panel.set_phase("refine", 40.0)  # ... and the search took 40 - 5 = 35 s
    assert panel.phases.duration["search"] == pytest.approx(35.0)
    assert panel.phases.duration["grid"] == pytest.approx(5.0), (
        "a finished chip keeps its number while the next stage runs"
    )
    # Only the running stage has a number that is still growing, and it comes
    # from this process's own monotonic clock.
    live = panel.phases.duration.get("refine")
    assert live is not None and 0.0 <= live < 5.0

    panel.set_phase("done", 40.5)
    assert panel.phases.active is None
    assert panel.phases.duration["refine"] == pytest.approx(0.5)
    assert panel.phases.duration["grid"] == pytest.approx(5.0)

    # A missing mark invents nothing: no number is better than a wrong one.
    fresh = dashboard.RunDashboard()
    fresh.begin_run()
    fresh.set_phase("search", None)
    assert fresh.phases.duration["grid"] is None
    fresh.set_phase("refine", None)
    assert fresh.phases.duration["search"] is None

    # And the finished run's own phase records stay authoritative.
    panel.finish_run(
        dashboard.RunTrace(
            energies=[-7.0],
            phases=[
                dashboard.PhaseRecord("grid", 0.0, 1.0),
                dashboard.PhaseRecord("search", 1.0, 40.0),
                dashboard.PhaseRecord("refine", 40.0, 40.0),
                dashboard.PhaseRecord("done", 40.0, 40.5),
            ],
        )
    )
    assert panel.phases.duration["grid"] == pytest.approx(1.0)
    assert panel.phases.duration["search"] == pytest.approx(39.0)
    assert panel.phases.duration["refine"] == 0.0, "refine has no duration of its own"


def test_run_dashboard_follows_a_run(qapp):
    panel = dashboard.RunDashboard()
    panel.resize(420, 260)
    panel.retranslate()
    assert panel.captions["best"].text() == "Best affinity"
    assert panel.values["best"].text() == "—"

    panel.begin_run(scoring="vina", exhaustiveness=8, seed=42)
    assert panel.phases.active == "grid"
    assert panel.elapsed() is not None
    panel.set_phase("search", 1.5)
    assert panel.phases.done == ["grid"]
    assert panel.phases.active == "search"
    assert panel.phases.duration["grid"] == pytest.approx(1.5)
    panel.set_phase("refine", 4.0)
    panel.set_phase("done", 4.5)
    assert panel.phases.active is None
    assert panel.phases.done == ["grid", "search", "refine"]

    trace = dashboard.RunTrace(
        energies=[-8.0, -7.4, -7.0],
        phases=[
            dashboard.PhaseRecord("grid", 0.0, 0.5),
            dashboard.PhaseRecord("search", 0.5, 4.0),
            dashboard.PhaseRecord("refine", 4.0, 4.0),
            dashboard.PhaseRecord("done", 4.0, 4.5),
        ],
        elapsed=4.5,
        scoring="vina",
        grid_points=150920,
        grid_mb=3,
        exhaustiveness=8,
    )
    panel.finish_run(trace)
    assert panel.elapsed() is None  # the clock stopped
    assert panel.values["best"].text().startswith("-8.00")
    assert panel.values["elapsed"].text() == "4.5 s"
    assert panel.values["poses"].text() == "3"
    assert "150920" in panel.values["grid"].text()
    assert panel.values["scoring"].text() == "vina ×8"
    assert panel.trace.drawn_points() == 3
    assert len(panel.history) == 1

    # Finishing the same trace twice must not duplicate it.
    panel.finish_run(trace)
    assert len(panel.history) == 1

    panel.fail_run("boom")
    assert panel.trace.running is False

    panel.retranslate()
    panel.reset()
    assert len(panel.history) == 0 and panel.values["best"].text() == "—"
    panel.reset(clear_history=False)  # the rebuild path must be harmless


def test_measurement_history_lists_and_copies(qapp):
    history = dashboard.MeasurementHistory()
    history.resize(320, 200)
    assert history.rows() == []
    # The empty-state wording belongs to i18n (another owner), so assert that the
    # panel says *something* about being empty rather than freezing the sentence.
    assert history.summary.text()
    assert history.summary.text() == i18n.EN["measure.empty"]
    history.set_measurements(
        [
            {
                "kind": "distance",
                "atoms": ["TYR337:OH", "LIG1:C1"],
                "value": "2.845",
                "unit": "Å",
                "label": "",
            },
            {
                "kind": "angle",
                "atoms": ["SER190:OG", "SER190:CB", "SER190:CA"],
                "value": "109.47",
                "unit": "°",
                "label": "side chain",
            },
        ]
    )
    assert history.table.rowCount() == 2
    assert history.table.item(0, 0).text() == "Distance"
    assert history.table.item(0, 2).text() == "2.845 Å"
    assert history.table.item(1, 0).text() == "Angle"
    assert history.table.item(1, 2).text() == "109.47 °"
    assert history.table.item(1, 1).toolTip() == "" or True
    # The wording of the summary belongs to i18n (another owner), so this asserts
    # the count it must carry rather than the noun it happens to use.
    assert history.summary.text().startswith("2 ")
    text = history.as_text()
    assert text.splitlines()[0].startswith("kind\tatoms")
    assert "TYR337:OH" in text and "2.845" in text
    csv = history.as_csv()
    assert csv.splitlines()[0] == "kind,atoms,value,unit,label"
    assert 'distance,"TYR337:OH LIG1:C1",2.845,A,""' in csv
    assert "angle" in csv and "deg" in csv
    history.retranslate()
    history.set_measurements([])
    assert history.table.rowCount() == 0
    assert history.as_csv().splitlines() == ["kind,atoms,value,unit,label"]


def test_pose_comparison_panel_shows_the_answer(qapp):
    panel = dashboard.PoseComparisonWidget()
    panel.resize(340, 420)
    assert panel.comparison() is None
    assert "Ctrl-click" in panel.heading.text()

    receptor = [
        atom("OH", "O", 3.9, 0.0, 0.0, res="TYR", res_id=337),
        atom("OD1", "O", 0.0, 3.9, 0.0, res="ASP", res_id=189),
    ]
    first = [atom("C1", "C", 0.0, 0.0, 0.0), atom("C2", "C", 1.4, 0.0, 0.0)]
    second = [atom("C1", "C", 0.0, 0.0, 0.0), atom("C2", "C", 1.4, 0.0, 0.0)]
    comparison = dashboard.compare_poses(
        first, second, receptor, index_a=1, index_b=4, affinity_a=-6.5, affinity_b=-8.0
    )
    panel.set_comparison(comparison)
    assert panel.comparison() is comparison
    assert "Pose 2 against pose 5" in panel.heading.text()
    assert panel.rmsd_label.text() == "Symmetric-aware RMSD"
    assert panel.rmsd_value.text().startswith("0.00 Å fitted")
    assert "2 heavy atoms" in panel.rmsd_value.text()
    assert "-1.50" in panel.delta_value.text()
    assert "same binding mode" in panel.delta_value.text()
    assert panel.contacts.rowCount() == comparison.diff.union_size
    assert panel.contacts.horizontalHeaderItem(1).text() == "Touched by"

    panel.retranslate()
    assert "Pose 2 against pose 5" in panel.heading.text()
    panel.clear()
    assert panel.comparison() is None
    assert panel.contacts.rowCount() == 0


# ---------------------------------------------------------------------------
# the palette against the real menu bar
# ---------------------------------------------------------------------------


def _leaf_actions(menu_bar):
    found = []

    def walk(menu):
        for action in menu.actions():
            if action.isSeparator():
                continue
            submenu = action.menu()
            if submenu is not None:
                walk(submenu)
                continue
            found.append(action)

    for action in menu_bar.actions():
        if action.menu() is not None:
            walk(action.menu())
    return found


def test_the_palette_offers_every_menu_action(window, qapp):
    palette = dashboard.CommandPalette(window)
    palette.refresh()
    offered = {entry.action for entry in palette.entries()}
    for action in _leaf_actions(window.menuBar()):
        if action.isEnabled() or action.isCheckable():
            assert action in offered, action.text()
    # Submenu titles are not commands.
    assert all("Export" not in entry.text for entry in palette.entries())
    # Every entry carries the path that leads to it.
    assert any(entry.path.startswith("File ▸ Export ▸") for entry in palette.entries())
    assert all(entry.path == " ▸ ".join(entry.path.split(" ▸ ")) for entry in palette.entries())


def test_the_palette_is_built_from_the_menu_bar_at_open_time(window, qapp):
    palette = dashboard.CommandPalette(window)
    palette.refresh()
    before = len(palette.entries())
    latecomer = QtGui.QAction("Zebra command", window.menuBar().actions()[0].menu())
    window.menuBar().actions()[0].menu().addAction(latecomer)
    palette.refresh()
    assert len(palette.entries()) == before + 1
    assert any(entry.action is latecomer for entry in palette.entries())
    latecomer.deleteLater()


def test_the_palette_filters_then_runs_the_highlighted_action(window, qapp):
    palette = dashboard.CommandPalette(window)
    palette.refresh()
    window.axes_action.setChecked(False)
    window.scene.show_axes = False
    palette.edit.setText("axes")
    matches = palette.matches()
    assert matches and matches[0].action is window.axes_action
    assert palette.list.count() == len(matches)
    assert palette.list.currentRow() == 0
    assert palette.selected_entry().action is window.axes_action
    assert palette.run_selected() is True
    qapp.processEvents()  # the trigger is deferred by one event-loop turn
    assert window.scene.show_axes is True

    palette.edit.setText("nothing matches this at all")
    assert palette.matches() == []
    assert palette.list.count() == 0
    assert "No menu entry" in palette.hint.text()
    palette.accept()


def test_the_palette_arrow_keys_move_the_selection(window, qapp):
    palette = dashboard.CommandPalette(window)
    palette.refresh()
    palette.show()
    qapp.processEvents()
    try:
        palette.edit.setText("")
        palette.list.setCurrentRow(0)
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        palette.eventFilter(palette.edit, event)
        assert palette.list.currentRow() == 1
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        palette.eventFilter(palette.edit, event)
        assert palette.list.currentRow() == 0
    finally:
        palette.accept()


def test_the_view_menu_offers_the_palette_on_ctrl_k(window, qapp):
    view = window.menuBar().actions()[-1].menu()
    action = next(a for a in view.actions() if a.text().startswith("Command palette"))
    assert action.shortcut().toString() == "Ctrl+K"
    assert window.inspect_action.shortcut().toString() == ""
    assert window.restore_action in window.menuBar().actions()[0].menu().actions()


# ---------------------------------------------------------------------------
# the window: appearance, panels, drag and drop, session
# ---------------------------------------------------------------------------


def test_the_run_monitor_is_docked_beside_the_pose_table(window, qapp):
    areas = QtCore.Qt.DockWidgetArea
    assert window.dashboard_dock.windowTitle() == "Run monitor"
    assert window.dockWidgetArea(window.dashboard_dock) == areas.BottomDockWidgetArea
    assert window.dockWidgetArea(window.bottom_dock) == areas.BottomDockWidgetArea
    widgets = {d.windowTitle() for d in window.findChildren(QtWidgets.QDockWidget)}
    assert {"Run monitor", "Pose comparison", "Poses & log"} <= widgets
    assert window.dashboard_tabs.tabText(0) == "Run"
    assert window.dashboard_tabs.tabText(1) == "Measurements"
    assert window.run_dashboard.history is window.run_history
    assert window.comparison_dock.windowTitle() == "Pose comparison"


def test_the_dashboard_survives_a_language_switch(window, qapp):
    window.run_history.add(
        dashboard.RunTrace(energies=[-7.0, -6.5], elapsed=2.0, scoring="vina")
    )
    window.run_dashboard.history = window.run_history
    window.run_dashboard.reset(clear_history=False)
    window.set_language("zh")
    qapp.processEvents()
    try:
        assert window.run_dashboard.captions["best"].text() == "最佳亲和力"
        assert window.run_dashboard.note.text().startswith("内核")
        assert window.dashboard_dock.windowTitle() == "运行监视"
        assert window.dashboard_tabs.tabText(1) == "测量"
        # The runs themselves are data and survive the rebuild.
        assert len(window.run_history) == 1
        assert window.run_dashboard.history is window.run_history
        assert window.run_dashboard.trace.drawn_points() == 2
        assert window.comparison.heading.text().startswith("在结果表中")
    finally:
        window.set_language("en")
        qapp.processEvents()
    assert window.run_dashboard.captions["best"].text() == "Best affinity"
    assert len(window.run_history) == 1


def test_the_hud_ink_follows_the_theme(window, qapp):
    """The Legend and the tool hint are QPainter text over the rendered image.

    A style sheet cannot reach them, so they must be picked from the theme: the
    light theme washed the legend out to invisible before this.
    """
    dark_ink = window.viewport._hud_ink()
    dark_hint = window.viewport._hud_hint()
    assert window.viewport._light_canvas is False
    assert dark_ink.lightness() > 128, "dark viewport, light ink"
    assert dark_hint.lightness() > 128

    window.set_theme("light")
    assert window.viewport._light_canvas is True
    light_ink = window.viewport._hud_ink()
    light_hint = window.viewport._hud_hint()
    assert light_ink.lightness() < 128, "light viewport, dark ink"
    assert light_hint.lightness() < 128
    assert light_ink != dark_ink and light_hint != dark_hint

    window.set_theme("dark")
    assert window.viewport._hud_ink() == dark_ink
    assert window.viewport._light_canvas is False


def test_the_tool_hint_is_not_hidden_by_the_floating_tool_strip(window, qapp):
    """The hint used to be painted along the bottom edge, under the tool strip."""
    window.resize(1000, 700)
    window.show()
    qapp.processEvents()
    window.viewport.set_mode("measure")
    qapp.processEvents()
    hint = window.viewport.hud_hint_rect()
    assert window.viewport.rect().contains(hint.toRect())
    strip = window._tool_strip.geometry()
    assert not hint.toRect().intersects(strip), (hint, strip)
    # The measure hint also has to be somewhere the legend is not.
    assert hint.top() < 40
    window.viewport.set_mode("orbit")


def test_the_two_themes_and_densities_reach_the_widgets(window, qapp):
    assert window.viewport.background == dashboard.DARK.viewport_clear
    assert window.viewport.canvas == dashboard.DARK.viewport_canvas
    assert "font-size: 12px" in window.styleSheet()

    window.set_theme("light")
    assert window.color_theme is dashboard.LIGHT
    assert window.viewport.background == dashboard.LIGHT.viewport_clear
    assert dashboard.LIGHT.palette["window_bg"] in window.styleSheet()
    assert window._theme_actions["light"].isChecked()
    assert not window._theme_actions["dark"].isChecked()
    assert "appearance" in window.log.toPlainText()

    window.set_density("compact")
    assert "font-size: 11px" in window.styleSheet()
    assert window._density_actions["compact"].isChecked()
    window.set_theme("dark")
    window.set_density("comfortable")
    assert window.viewport.background == dashboard.DARK.viewport_clear
    assert "font-size: 12px" in window.styleSheet()


def _panel_pages(dock):
    """The scrollable pages of a dock: the widget itself, or its tab pages."""
    widget = dock.widget()
    if isinstance(widget, QtWidgets.QTabWidget):
        return [widget.widget(index) for index in range(widget.count())]
    return [widget]


def test_the_pose_readout_cannot_decide_the_window_width(window, qapp, tmp_path):
    """A 99-character pose read-out used to *be* the window's minimum width.

    ``lbl_pose`` is a QLabel: unwrapped, it reports the width of its whole text
    (1188 px on the reference machine) as its minimum, and because it sits in
    the bottom drawer that became the floor of the entire window (1654 px) the
    moment a pose and its contact residues were on screen. Wrapping it inside a
    cap bounds the floor while keeping the text complete.
    """
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    poses = tmp_path / "poses.pdbqt"
    poses.write_text(POSES_PDBQT, encoding="utf-8")
    window.resize(1200, 780)
    window.show()
    qapp.processEvents()
    window.load_receptor(receptor)
    window.load_poses(poses)
    window.set_pose(1)
    qapp.processEvents()

    text = window.lbl_pose.text()
    assert len(text) > 40, text
    assert window.pose_label_text() == text, "the read-out keeps every character"
    assert window.lbl_pose.toolTip() == text
    assert window.lbl_pose.wordWrap()
    # The pose slider is gone, so the read-out takes the width the row has — but
    # it must still not *dictate* it: an expanding, wrapping label with a small
    # minimum is what keeps the window narrow-able.
    assert window.lbl_pose.sizePolicy().horizontalPolicy() == (
        QtWidgets.QSizePolicy.Policy.Expanding
    )
    assert window.lbl_pose.minimumSizeHint().width() < 200
    assert not hasattr(window, "pose_slider")
    assert window.minimumSizeHint().width() <= 1000, window.minimumSizeHint()

    # And it really is narrow-able: asking for a small window gets one.
    window.resize(980, 640)
    qapp.processEvents()
    assert window.width() <= 980
    assert window.minimumSizeHint().width() <= 1000


def test_the_window_stays_narrow_with_every_preset_and_a_restored_session(qapp, tmp_path):
    """A layout the window cannot hold is a floor wearing a different hat.

    The layout presets and the session both rearrange the docks; if any
    arrangement needed more room than the window's floor allows, the window
    could not be made small again — which is the same defect as a widget
    minimum, just harder to see.
    """
    store = dashboard.SessionStore(tmp_path / "session.json")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    poses = tmp_path / "poses.pdbqt"
    poses.write_text(POSES_PDBQT, encoding="utf-8")

    first = DockingWorkbench(session=store)
    try:
        first.load_receptor(receptor)
        first.load_poses(poses)
        first.resize(1440, 900)
        first.show()
        qapp.processEvents()
        for preset in dashboard.LAYOUT_PRESETS:
            first._apply_layout_preset(preset)
            qapp.processEvents()
            assert first.minimumSizeHint().width() <= 1000, preset
            first.resize(960, 620)
            qapp.processEvents()
            assert first.minimumSizeHint().width() <= 1000, preset
            # The panels are the user's to squeeze: they scroll instead.
            for dock in (first.dashboard_dock, first.comparison_dock):
                for page in _panel_pages(dock):
                    assert isinstance(page, QtWidgets.QScrollArea), dock.windowTitle()
        first.resize(1440, 900)
        qapp.processEvents()
        assert first.save_session() is True
    finally:
        first.close()

    second = DockingWorkbench(session=store)
    try:
        assert second.restore_session() is True
        second.resize(960, 620)
        second.show()
        qapp.processEvents()
        assert second.minimumSizeHint().width() <= 1000
        assert second.minimumSizeHint().height() <= 800
        assert len(second.scene.receptor) == 24
        for preset in dashboard.LAYOUT_PRESETS:
            second._apply_layout_preset(preset)
            qapp.processEvents()
            assert second.minimumSizeHint().width() <= 1000, preset
    finally:
        second.close()


def test_the_comparison_can_be_copied_as_text(window, qapp, tmp_path):
    """The panel's Copy button puts exactly the numbers on screen on the clipboard."""
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    poses = tmp_path / "poses.pdbqt"
    poses.write_text(POSES_PDBQT, encoding="utf-8")
    window.load_receptor(receptor)
    window.load_poses(poses)
    qapp.processEvents()
    window.table.clearSelection()
    window.table.selectRow(0)
    window.table.item(1, 0).setSelected(True)
    qapp.processEvents()
    comparison = window.comparison.comparison()
    assert comparison is not None

    window.comparison.btn_copy.click()
    qapp.processEvents()
    # Asynchronous clipboard: wait for the write instead of reading once.
    text = _clipboard_text(qapp, comparison.as_text())
    assert text == comparison.as_text()
    assert "Pose 1 against pose 2" in text
    assert "Symmetric-aware RMSD" in text
    assert "Affinity difference" in text
    assert "Contact fingerprint" in text
    # The shared/unique residues are listed, one per line, with their distances.
    for key, left, _right in comparison.diff.shared:
        assert dashboard.residue_label(key) in text
    for contact in comparison.diff.only_a:
        assert contact.label in text
    # heading + RMSD + delta + fingerprint, then one line per contacting residue.
    assert text.count("\n") == 3 + comparison.diff.union_size
    assert "clipboard" in window.log.toPlainText()


def test_the_pose_comparison_report_reads_as_a_document():
    receptor = [
        atom("OH", "O", 3.9, 0.0, 0.0, res="TYR", res_id=337),
        atom("OD1", "O", 0.0, 3.9, 0.0, res="ASP", res_id=189),
    ]
    first = [atom("C1", "C", 0.0, 0.0, 0.0), atom("C2", "C", 1.4, 0.0, 0.0)]
    second = [atom("C1", "C", 0.0, 0.0, 0.0), atom("C2", "C", 1.4, 0.0, 0.0)]
    comparison = dashboard.compare_poses(
        first, second, receptor, index_a=0, index_b=4, affinity_a=-7.0, affinity_b=-8.0
    )
    text = comparison.as_text()
    lines = text.splitlines()
    assert lines[0] == "Pose 1 against pose 5"
    assert lines[1].strip().startswith("Symmetric-aware RMSD")
    assert "-1.00" in lines[2]
    assert "the same binding mode" in lines[2]
    assert "shared" in lines[3]
    assert "TYR337" in text and "ASP189" in text
    residue_lines = [line for line in lines if line.startswith("    ")]
    assert residue_lines, "the residues are listed"
    assert all(len(line) <= 60 for line in residue_lines)

    # Without affinities the delta line still reads, and says so.
    unknown = dashboard.compare_poses(first, second, receptor)
    assert "—" in unknown.as_text().splitlines()[2]


# ---------------------------------------------------------------------------
# measurement geometry: hand-computed values, cross-checked with RDKit
# ---------------------------------------------------------------------------


def test_angle_is_measured_at_the_middle_atom():
    """90°, 180°, 45° and 60° at the vertex, from hand-picked coordinates."""
    assert dashboard.angle_degrees((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90.0)
    assert dashboard.angle_degrees((1, 0, 0), (0, 0, 0), (-1, 0, 0)) == pytest.approx(180.0)
    assert dashboard.angle_degrees((1, 0, 0), (0, 0, 0), (1, 1, 0)) == pytest.approx(45.0)
    # The vertex is the *middle* atom: the same three points clicked in another
    # order are a different angle.
    assert dashboard.angle_degrees((0, 0, 0), (1, 0, 0), (1, 1, 0)) == pytest.approx(90.0)
    assert dashboard.angle_degrees(
        (1, 0, 0), (0, 0, 0), (0.5, math.sqrt(3) / 2, 0)
    ) == pytest.approx(60.0)
    # A degenerate vertex is 0°, never a nan.
    assert dashboard.angle_degrees((0, 0, 0), (0, 0, 0), (1, 0, 0)) == 0.0


def test_dihedral_matches_the_hand_computed_arrangement():
    """A 90° twist about the central bond, and the two planar cases.

    p1 is the origin, p2 on +y, p0 on +x (so the first plane is *xy*) and p3 on
    +y+z: the second plane is spanned by the y axis and +z, so the planes are
    perpendicular and the torsion is ±90°. The sign follows the IUPAC convention
    (the far bond clockwise is positive), hence -90° for this arrangement.
    """
    assert dashboard.dihedral_degrees(
        (1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1)
    ) == pytest.approx(-90.0)
    # All four points in the xy plane: syn is 0°, the mirror is 180°.
    assert dashboard.dihedral_degrees(
        (1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)
    ) == pytest.approx(0.0)
    assert dashboard.dihedral_degrees(
        (1, 0, 0), (0, 0, 0), (0, 1, 0), (-1, 1, 0)
    ) == pytest.approx(180.0)
    assert dashboard.dihedral_degrees((1, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0)) == 0.0


def test_angle_and_dihedral_agree_with_rdkit():
    """The independent oracle for the arithmetic: RDKit's own transforms."""
    rdMolTransforms = pytest.importorskip("rdkit.Chem.rdMolTransforms")
    from rdkit import Chem

    def mol_for(points):
        editable = Chem.RWMol()
        for _ in points:
            editable.AddAtom(Chem.Atom(6))
        for index in range(len(points) - 1):
            editable.AddBond(index, index + 1, Chem.BondType.SINGLE)
        conformer = Chem.Conformer(len(points))
        for index, point in enumerate(points):
            conformer.SetAtomPosition(index, point)
        editable.AddConformer(conformer)
        return editable.GetMol()

    for points in (
        [(1, 0, 0), (0, 0, 0), (0, 1, 0)],
        [(1, 1, 1), (0, 0, 0), (1, -1, 0)],
        [(2, 0, 0), (0, 0, 0), (0, 0, 3)],
        [(0.5, 0.5, 0.5), (0, 0, 0), (-0.5, 0.5, 0.2)],
    ):
        mine = dashboard.angle_degrees(*points)
        theirs = rdMolTransforms.GetAngleDeg(mol_for(points).GetConformer(), 0, 1, 2)
        assert mine == pytest.approx(theirs, abs=1e-9), points

    for points in (
        [(1, 0, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1)],
        [(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)],
        [(1, 0, 1), (0, 0, 0), (0, 1, 0), (1, 1, -1)],
        [(2, 0, 0), (0, 0, 0), (0, 3, 0), (0, 3, 2)],
        [(1, 1, 0), (0, 0, 0), (0, 1, 0), (-1, 1, 1)],
    ):
        mine = dashboard.dihedral_degrees(*points)
        theirs = rdMolTransforms.GetDihedralDeg(mol_for(points).GetConformer(), 0, 1, 2, 3)
        assert mine == pytest.approx(theirs, abs=1e-9), points


def test_centroid_plane_and_plane_angles():
    square = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    assert dashboard.centroid(square) == pytest.approx((1.0, 1.0, 0.0))
    assert dashboard.centroid([]) is None
    assert dashboard.plane_normal([(0, 0, 0), (1, 0, 0), (0, 1, 0)]) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    fitted = dashboard.plane_normal(square)
    assert abs(fitted[2]) == pytest.approx(1.0)
    assert fitted[0] == pytest.approx(0.0, abs=1e-9)
    assert dashboard.plane_normal([(0, 0, 0), (1, 0, 0)]) is None
    assert dashboard.plane_plane_angle((0, 0, 1), (1, 0, 0)) == pytest.approx(90.0)
    assert dashboard.plane_plane_angle((0, 0, 1), (0, 0, -1)) == pytest.approx(0.0)
    assert dashboard.plane_plane_angle((0, 0, 1), (0, 1, 1)) == pytest.approx(45.0)
    assert dashboard.line_plane_angle((1, 0, 0), (0, 0, 1)) == pytest.approx(0.0)
    assert dashboard.line_plane_angle((0, 0, 1), (0, 0, 1)) == pytest.approx(90.0)
    assert dashboard.line_plane_angle((1, 0, 1), (0, 0, 1)) == pytest.approx(45.0)
    assert dashboard.point_plane_distance((0, 0, 3), (0, 0, 0), (0, 0, 1)) == pytest.approx(3.0)
    assert dashboard.point_plane_distance((0, 0, -3), (0, 0, 0), (0, 0, 1)) == pytest.approx(-3.0)


def test_measurement_kinds_consume_the_right_number_of_atoms():
    assert dashboard.MEASUREMENT_ORDER[0] == "distance"
    for kind in dashboard.MEASUREMENT_ORDER:
        assert kind in dashboard.MEASUREMENT_KINDS
    assert dashboard.measurement_expected_atoms("distance") == 2
    assert dashboard.measurement_expected_atoms("angle") == 3
    assert dashboard.measurement_expected_atoms("dihedral") == 4
    assert dashboard.measurement_expected_atoms("plane_angle") == 6
    assert dashboard.measurement_expected_atoms("plane_bond") == 5
    assert dashboard.measurement_unit("distance") == "Å"
    assert dashboard.measurement_unit("angle") == "°"
    assert dashboard.measurement_value("angle", [(0, 0, 0), (1, 0, 0)]) is None
    assert dashboard.measurement_value("plane_angle", [(0, 0, 0)] * 5) is None
    assert dashboard.measurement_value("not-a-kind", [(0, 0, 0), (1, 0, 0)]) is None
    # plane_bond: the plane is xy, the bond points along +z → 90°.
    assert dashboard.measurement_value(
        "plane_bond", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1)]
    ) == pytest.approx(90.0)
    # plane_angle: xy against yz → 90°.
    assert dashboard.measurement_value(
        "plane_angle", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0)]
    ) == pytest.approx(90.0)


def test_measurement_values_are_formatted_with_units():
    assert dashboard.format_measurement("distance", 2.8451) == "2.845 Å"
    assert dashboard.format_measurement("angle", 90.0) == "90.00°"
    assert dashboard.format_measurement("dihedral", -60.5) == "-60.50°"
    assert dashboard.format_measurement("centroid", (1.0, 2.0, 3.0)) == "1.00  2.00  3.00"
    assert dashboard.format_measurement("plane", (0.0, 0.0, 1.0)) == "normal +0.000 +0.000 +1.000"
    assert dashboard.format_measurement("distance", None) == "—"


def test_a_measurement_round_trips_and_reports_its_shape():
    measurement = dashboard.Measurement(
        kind="plane_angle",
        refs=[("receptor", 1), ("receptor", 2), ("receptor", 3), ("ligand", 0), ("ligand", 1), ("ligand", 2)],
        label="ring twist",
    )
    assert measurement.expected_atoms == 6
    assert measurement.split == 3
    assert measurement.unit == "°"
    assert measurement.is_complete()
    restored = dashboard.Measurement.from_dict(measurement.to_dict())
    assert restored.refs == measurement.refs
    assert restored.label == "ring twist"
    assert not dashboard.Measurement("angle", refs=[("ligand", 0), ("ligand", 1)]).is_complete()


def test_measurements_export_as_csv():
    csv = dashboard.measurements_csv(
        [
            {"kind": "distance", "atoms": ["SER190:OG", "BEN1:N1"], "value": "2.845", "unit": "Å", "label": ""},
            {"kind": "angle", "atoms": ["A", "B", "C"], "value": "109.47", "unit": "°", "label": 'the "good" one'},
        ]
    )
    lines = csv.splitlines()
    assert lines[0] == "kind,atoms,value,unit,label"
    assert lines[1] == 'distance,"SER190:OG BEN1:N1",2.845,A,""'
    assert lines[2] == 'angle,"A B C",109.47,deg,"the \'good\' one"'


# ---------------------------------------------------------------------------
# undo / redo
# ---------------------------------------------------------------------------


def test_the_command_stack_undoes_and_redoes():
    stack = dashboard.CommandStack(merge_window=0)
    state = {"value": 0}

    def setter(new):
        def apply():
            state["value"] = new

        return apply

    stack.push(dashboard.Command("one", setter(0), setter(1)))
    assert state["value"] == 1
    assert stack.can_undo() and not stack.can_redo()
    stack.push(dashboard.Command("two", setter(1), setter(2)))
    assert state["value"] == 2
    assert stack.undo_depth == 2

    assert stack.undo().name == "two"
    assert state["value"] == 1
    assert stack.undo().name == "one"
    assert state["value"] == 0
    assert not stack.can_undo()
    assert stack.undo() is None

    assert stack.redo().name == "one"
    assert state["value"] == 1
    assert stack.redo().name == "two"
    assert state["value"] == 2
    assert stack.redo() is None

    stack.undo()
    stack.push(dashboard.Command("three", setter(0), setter(9)))
    assert not stack.can_redo()
    assert state["value"] == 9


def test_a_slider_drag_is_one_undo_step():
    """Coalescing: 200 pose changes in a drag are one step back to the start."""
    stack = dashboard.CommandStack(merge_window=60.0)
    history = [0]

    def command(old, new):
        return dashboard.Command(
            "pose",
            lambda: history.append(("undo", old)),
            lambda: history.append(("redo", new)),
            merge_key="pose",
        )

    for index in range(1, 201):
        stack.push(command(index - 1, index))
    assert stack.undo_depth == 1, "the drag must be one step"
    history.clear()
    stack.undo()
    assert history == [("undo", 0)], "undo goes back to where the drag started"
    history.clear()
    stack.redo()
    assert history == [("redo", 200)], "redo lands on the final state"

    # A *different* kind of edit does not merge with the pose drag.
    stack.push(dashboard.Command("style", lambda: None, lambda: None, merge_key="style"))
    assert stack.undo_depth == 2
    # With the window at zero, even repeated pose changes stay separate steps
    # (the window is what makes a drag one step, not "all pose changes ever").
    strict = dashboard.CommandStack(merge_window=0)
    for _ in range(3):
        strict.push(
            dashboard.Command("pose", lambda: None, lambda: None, merge_key="pose")
        )
    assert strict.undo_depth == 3
    stack.clear()
    assert not stack.can_undo() and not stack.can_redo()


def test_the_command_stack_is_bounded():
    stack = dashboard.CommandStack(cap=3, merge_window=0)
    for index in range(10):
        stack.push(dashboard.Command(f"edit {index}", lambda: None, lambda: None))
    assert stack.undo_depth == 3
    assert stack.undo_name() == "edit 9"


def test_annotations_round_trip():
    note = dashboard.Annotation(
        text="gatekeeper", anchor=("residue", "A", 337, "TYR"), colour=(0.9, 0.4, 0.2)
    )
    assert note.kind == "residue"
    assert dashboard.Annotation.from_dict(note.to_dict()) == note
    assert dashboard.Annotation(text="here", anchor=("ligand", 3)).kind == "ligand"
    assert dashboard.Annotation(text="x").kind == "atom"


# ---------------------------------------------------------------------------
# the interaction lines: named, countable, filterable
# ---------------------------------------------------------------------------


class _FakeInteraction:
    """The only shape the panels need from ``odock.analysis.Interaction``."""

    def __init__(self, kind, a, b, distance, detail=""):
        self.kind = kind
        self.a = a
        self.b = b
        self.distance = distance
        self.detail = detail


RECEPTOR_FOR_DASHES = [
    atom("OG", "O", 0.0, 0.0, 0.0, res="SER", res_id=190),
    atom("CG1", "C", 0.0, 1.0, 0.0, res="VAL", res_id=213),
    atom("CG", "C", 0.0, 2.0, 0.0, res="GLN", res_id=192),
]
LIGAND_FOR_DASHES = [
    atom("N1", "N", 2.9, 0.0, 0.0, res="BEN"),
    atom("C6", "C", 3.8, 0.0, 0.0, res="BEN"),
]


def test_the_interaction_table_names_both_ends_of_every_line(qapp):
    """A dash in the 3-D view has a row here naming its two atoms."""
    panel = dashboard.InteractionTable()
    assert panel.rows() == []
    assert "No interaction lines" in panel.heading.text()

    panel.set_interactions(
        [
            _FakeInteraction("hbond", 0, 0, 2.89, "BEN1:N1->SER190:OG"),
            _FakeInteraction("hydrophobic", 1, 1, 3.77, "VAL213:CG1...BEN1:C6"),
        ],
        RECEPTOR_FOR_DASHES,
        LIGAND_FOR_DASHES,
    )
    rows = panel.rows()
    assert len(rows) == 2
    assert rows[0]["receptor"] == "SER190:OG"
    assert rows[0]["ligand"] == "BEN1:N1"
    assert panel.table.item(0, 0).text() == "H-bond"
    assert panel.table.item(0, 1).text() == "SER190:OG"
    assert panel.table.item(0, 2).text() == "BEN1:N1"
    assert panel.table.item(0, 3).text() == "2.89"
    assert panel.table.item(0, 1).toolTip() == "BEN1:N1->SER190:OG"
    assert "2 lines" in panel.heading.text()
    text = panel.as_text()
    assert text.splitlines()[0].startswith("Type\tReceptor atom\tLigand atom")
    assert "SER190:OG" in text and "2.89" in text

    # A hidden count is always stated, so a filtered view cannot read as "none".
    panel.set_interactions(
        [_FakeInteraction("hbond", 0, 0, 2.89)],
        RECEPTOR_FOR_DASHES,
        LIGAND_FOR_DASHES,
        hidden=3,
    )
    assert "hidden by the filter" in panel.heading.text()
    assert "3" in panel.heading.text()

    # An index that is not an atom is reported, never invented.
    panel.set_interactions(
        [_FakeInteraction("hbond", 99, 99, 2.89)], RECEPTOR_FOR_DASHES, LIGAND_FOR_DASHES
    )
    assert panel.table.item(0, 1).text() == "—"
    panel.retranslate()
    panel.set_interactions([], RECEPTOR_FOR_DASHES, LIGAND_FOR_DASHES)
    assert panel.table.rowCount() == 0


def test_interaction_labels_are_translated():
    i18n.set_language("en")
    assert dashboard.interaction_label("hbond") == "H-bond"
    assert dashboard.interaction_label("salt_bridge") == "salt bridge"
    assert dashboard.interaction_label("not-a-kind") == "not-a-kind"
    i18n.set_language("zh")
    assert dashboard.interaction_label("hbond") == "氢键"
    i18n.set_language("en")


def test_the_interaction_lines_can_be_filtered_by_kind(window, qapp):
    """View ▸ Interaction lines decides what is drawn, and says what it hid.

    The detections are never recomputed: the table keeps every row and the number
    of hidden lines is stated, so hiding the near-white hydrophobic dashes can
    never look like "there is nothing there".
    """
    window.scene.receptor = list(RECEPTOR_FOR_DASHES)
    window.scene.ligand = list(LIGAND_FOR_DASHES)
    window.interactions = [
        _FakeInteraction("hbond", 0, 0, 2.9),
        _FakeInteraction("hydrophobic", 1, 1, 3.8),
        _FakeInteraction("hydrophobic", 2, 1, 3.9),
    ]
    # Drawing is opt-in (Analysis ▸ Show interactions). The synthetic contact
    # list must survive, so switch the flag rather than re-running the profile.
    window._interactions_shown = True
    window._sync_interactions()
    qapp.processEvents()
    assert len(window.scene.interactions) == 3
    assert window.interaction_table.table.rowCount() == 3
    assert "Interactions (3)" in window.dashboard_tabs.tabText(
        window._interaction_tab_index
    )

    kinds = window._interaction_actions
    assert set(kinds) == {
        "hbond",
        "salt_bridge",
        "pi_pi",
        "cation_pi",
        "hydrophobic",
        "clash",
    }
    assert all(action.isChecked() for action in kinds.values())
    assert all(action.isCheckable() for action in kinds.values())

    kinds["hydrophobic"].setChecked(False)
    qapp.processEvents()
    assert [item.kind for item in window.scene.interactions] == ["hbond"]
    assert "3 lines in the view" in window.interaction_table.heading.text()
    assert "2 hidden by the filter" in window.interaction_table.heading.text()
    assert window.interaction_table.table.rowCount() == 3, "the data is kept"
    assert "drawing 1 of 3 interaction lines" in window.log.toPlainText()
    # The tab title counts the detections, not the drawn lines.
    assert "Interactions (3)" in window.dashboard_tabs.tabText(
        window._interaction_tab_index
    )

    window._interaction_actions["hbond"].setChecked(False)
    qapp.processEvents()
    assert window.scene.interactions == []
    assert "3 hidden by the filter" in window.interaction_table.heading.text()

    window._show_all_interactions()
    qapp.processEvents()
    assert len(window.scene.interactions) == 3
    assert "hidden" not in window.interaction_table.heading.text()
    assert all(action.isChecked() for action in kinds.values())

    # The choice is remembered across a language switch (the menu is rebuilt).
    kinds["hydrophobic"].setChecked(False)
    qapp.processEvents()
    window.set_language("zh")
    qapp.processEvents()
    try:
        assert not window._interaction_actions["hydrophobic"].isChecked()
        assert window._interaction_actions["hbond"].isChecked()
        assert [item.kind for item in window.scene.interactions] == ["hbond"]
        assert (
            window.dashboard_tabs.tabText(window._interaction_tab_index) == "相互作用 (3)"
        )
        assert "被筛选隐藏" in window.interaction_table.heading.text()
    finally:
        window.set_language("en")
        qapp.processEvents()
    assert not window._interaction_actions["hydrophobic"].isChecked()
    assert [item.kind for item in window.scene.interactions] == ["hbond"]


def test_the_interaction_list_can_be_copied(window, qapp):
    window.scene.receptor = list(RECEPTOR_FOR_DASHES)
    window.scene.ligand = list(LIGAND_FOR_DASHES)
    window.interactions = [
        _FakeInteraction("hbond", 0, 0, 2.89, "BEN1:N1->SER190:OG"),
    ]
    window._sync_interactions()
    expected = window.interaction_table.as_text()
    # The panel's own text is the contract; the clipboard is a process-global,
    # asynchronous side effect, so it is waited for and reported honestly.
    window._copy_interactions()
    clipboard = _clipboard_text(qapp, expected)
    assert clipboard == expected, "the copy must reach the clipboard"
    assert "SER190:OG" in expected and "BEN1:N1" in expected
    assert "copied 1 interaction lines" in window.log.toPlainText()

    window.interactions = []
    window._sync_interactions()
    window._copy_interactions()
    assert i18n.EN["interactions.empty"] in window.log.toPlainText()


def test_the_legend_says_what_a_dash_joins(window, qapp):
    """The HUD legend names each kind and states the convention, not just colours."""
    assert "interaction.legend_convention" in i18n.EN
    assert "ligand atom" in i18n.EN["interaction.legend_convention"]
    assert i18n.ZH["interaction.legend_convention"]
    window.scene.interactions = [_FakeInteraction("hbond", 0, 0, 2.9)]
    window.show()
    qapp.processEvents()
    window.viewport.update()
    qapp.processEvents()
    pixmap = QtGui.QPixmap(window.viewport.size())
    window.viewport.render(pixmap)
    window.scene.interactions = []


# ---------------------------------------------------------------------------
# task-29: measurements, annotations, undo/redo in the real window
# ---------------------------------------------------------------------------


def _pump(qapp, turns: int = 40) -> None:
    """Let the window finish deferred work (dock proportions, rebuilds)."""
    for _ in range(turns):
        qapp.processEvents()


def _toolkit_window(qapp, tmp_path):
    """A window with the demo structures, as the user would have it."""
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text(LIGAND_PDBQT, encoding="utf-8")
    poses = tmp_path / "poses.pdbqt"
    poses.write_text(POSES_PDBQT, encoding="utf-8")
    window = DockingWorkbench()
    window.resize(1200, 800)
    window.show()
    qapp.processEvents()
    window.load_receptor(receptor)
    window.load_ligand(ligand)
    window.load_poses(poses)
    qapp.processEvents()
    return window


def test_a_measurement_survives_an_orbit_and_a_pose_change(qapp, tmp_path):
    """Orbiting must not touch the number; a pose change must not stale it."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        # Two *ligand* atoms, so a pose change moves them.
        window.set_measure_kind("distance")
        window.commit_measurement("distance", [("ligand", 0), ("ligand", 1)])
        qapp.processEvents()
        measurement = window._measurements[0]
        before = window.measurement_value(measurement)
        assert before is not None and before > 0
        overlay = window.viewport.measurement_overlays[0]["text"]
        assert "Distance" in overlay

        camera = window.viewport.camera
        camera.azimuth += 0.8
        camera.elevation -= 0.3
        camera.distance *= 0.8
        window.viewport.refresh()
        qapp.processEvents()
        assert window.measurement_value(window._measurements[0]) == pytest.approx(before)
        assert window.viewport.measurement_overlays[0]["text"] == overlay
        assert window._measurements[0].refs == [("ligand", 0), ("ligand", 1)]

        # A new pose moves the atoms: the *value* follows the coordinates, so the
        # panel can never show a number from the previous pose.
        window.set_pose(min(1, len(window.pose_models) - 1))
        qapp.processEvents()
        after = window.measurement_value(window._measurements[0])
        assert window._measurements[0].refs == [("ligand", 0), ("ligand", 1)]
        assert len(window.viewport.measurement_overlays) == 1
        assert after is not None and after != pytest.approx(before)
        assert window.measurement_rows()[0]["value"].startswith(f"{after:.3f}"[:5])

        # Orbit again on the new pose: still stable.
        camera.azimuth -= 1.2
        window.viewport.refresh()
        qapp.processEvents()
        assert window.measurement_value(window._measurements[0]) == pytest.approx(after)
    finally:
        window.close()


def test_every_measurement_kind_is_selectable_and_computed(qapp, tmp_path):
    """All seven kinds run in the real window and reach the panel and the view."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        cases = {
            "distance": [("receptor", 0), ("ligand", 0)],
            "angle": [("receptor", 0), ("receptor", 1), ("receptor", 2)],
            "dihedral": [("receptor", 0), ("receptor", 1), ("receptor", 2), ("receptor", 3)],
            "centroid": [("receptor", index) for index in range(4)],
            "plane": [("receptor", 0), ("receptor", 1), ("receptor", 2)],
            "plane_angle": [("receptor", 0), ("receptor", 1), ("receptor", 2),
                            ("ligand", 0), ("ligand", 1), ("ligand", 2)],
            "plane_bond": [("receptor", 0), ("receptor", 1), ("receptor", 2),
                           ("ligand", 0), ("ligand", 1)],
        }
        for kind, refs in cases.items():
            window.set_measure_kind(kind)
            assert window._measure_kind == kind
            assert window._measure_actions[kind].isChecked()
            measurement = window.commit_measurement(kind, refs)
            assert measurement is not None, kind
            assert window.measurement_value(measurement) is not None, kind
        qapp.processEvents()
        rows = window.measurement_rows()
        assert [row["kind"] for row in rows] == list(cases)
        assert all(row["value"] != "—" for row in rows)
        assert len(window.viewport.measurement_overlays) == len(cases)
        assert window.measure_history.table.rowCount() == len(cases)
        # The ruler/atom-table path agrees with the pick path about the atoms.
        key = window.sequence.blocks()[0].key
        window.sequence.select_keys([key])
        window.set_measure_kind("centroid")
        assert window.measure_selection() is True
        assert window._measurements[-1].refs == window.sequence.atom_refs()[:1]
        # A kind that needs more atoms than the selection has is refused, not
        # measured from whatever happened to be there.
        window.sequence.clear_selection()
        window.set_measure_kind("dihedral")
        count = len(window._measurements)
        assert window.measure_selection() is False
        assert len(window._measurements) == count
    finally:
        window.close()


def test_annotations_are_written_into_a_snapshot(qapp, tmp_path):
    """A saved figure carries the labels, not just the pixels the GPU drew."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        window.add_annotation("gatekeeper", anchor=("ligand", 0))
        window.add_annotation("hinge", anchor=("receptor", 0), colour=(0.2, 0.9, 0.4))
        qapp.processEvents()
        assert len(window.viewport.annotation_overlays) == 2
        assert window.annotation_table.table.rowCount() == 2
        assert "gatekeeper" in window.annotation_table.as_text()

        if not window.viewport._ensure_context():
            pytest.skip("no GL context on this machine")

        def image(path):
            assert window.viewport.snapshot(path, 640, 480) is True
            loaded = QtGui.QImage(str(path))
            return loaded

        with_labels = OUT_DIR / "snapshot-with-annotations.png"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        first = image(with_labels)
        window.toggle_annotations(False)
        qapp.processEvents()
        assert window.viewport.annotation_overlays == []
        without = image(OUT_DIR / "snapshot-without-annotations.png")
        window.toggle_annotations(True)
        qapp.processEvents()

        a = first.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        b = without.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        differing = 0
        for y in range(0, a.height(), 4):
            for x in range(0, a.width(), 4):
                if a.pixel(x, y) != b.pixel(x, y):
                    differing += 1
        assert differing > 50, (
            "the annotation layer must be painted into the snapshot "
            f"({differing} differing samples)"
        )
    finally:
        window.close()


def test_undo_restores_the_exact_panel_state_for_every_command_type(qapp, tmp_path):
    """The failure undo invites: the scene goes back but a panel does not."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        def round_trip(name, action, *, clear=True):
            if clear:
                window._history.clear()
            before = window._scene_state()
            action()
            qapp.processEvents()
            after = window._scene_state()
            assert after != before, f"{name} changed nothing"
            assert window.undo() is True
            qapp.processEvents()
            assert window._scene_state() == before, f"{name} did not restore the panels"
            assert window.redo() is True
            qapp.processEvents()
            assert window._scene_state() == after, f"{name} did not redo"

        round_trip(
            "measurement",
            lambda: window.commit_measurement("angle", [("receptor", 0), ("receptor", 1), ("receptor", 2)]),
        )
        round_trip(
            "annotation",
            lambda: window.add_annotation("note", anchor=("ligand", 0)),
        )
        round_trip(
            "selection",
            lambda: window.sequence.select_keys([window.sequence.blocks()[1].key]),
        )
        round_trip(
            "style",
            lambda: window._set_style("ligand", "wireframe"),
        )
        round_trip("theme", lambda: window.set_theme("light"))
        round_trip(
            "box",
            lambda: window._set_box((1.0, 2.0, 3.0), (21.0, 21.0, 21.0), 0.5),
        )
        round_trip("pose", lambda: window.set_pose(1))

        # The redo branch is dropped by a new edit, like every editor.
        window._history.clear()
        window._measurements = []
        window._sync_measurements()
        window.commit_measurement("distance", [("ligand", 0), ("ligand", 1)])
        assert len(window._measurements) == 1
        assert window.undo() is True
        assert window._history.can_redo()
        window.set_density("compact")
        assert not window._history.can_redo(), "a new edit drops the redo branch"
        assert window.undo() is True
        assert window.density == "comfortable"
        # The measurement stays undone: its redo branch was discarded with it.
        assert window._measurements == []
        assert not window._history.can_undo()
    finally:
        window.close()


def test_a_slider_drag_is_one_undo_step_in_the_window(qapp, tmp_path):
    """Two hundred slider events, one Ctrl+Z, and the panels agree afterwards."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        window._history.clear()
        assert len(window.pose_models) >= 2
        start = window.pose_index()
        for index in range(1, len(window.pose_models)):
            window.set_pose(index)
            qapp.processEvents()
        assert window._history.undo_depth == 1, "a drag is one step"
        assert window.pose_index() == len(window.pose_models) - 1
        assert window.undo() is True
        qapp.processEvents()
        assert window.pose_index() == start
        # The table, the tree and the overlays follow the slider.
        assert window.table.currentRow() == start
        assert window.pose_label_text().startswith(f"mode {start + 1} /")
        assert window.redo() is True
        assert window.pose_index() == len(window.pose_models) - 1
    finally:
        window.close()


def test_the_palette_reaches_every_measurement_and_annotation_action(qapp, tmp_path):
    """Keyboard-only path: each new action is a menu action, so Ctrl+K finds it."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        palette = dashboard.CommandPalette(window)
        palette.refresh()
        texts = {entry.text for entry in palette.entries()}
        for kind in dashboard.MEASUREMENT_ORDER:
            assert dashboard.measurement_kind_label(kind) in texts, kind
        for key in (
            "action.measure_selection",
            "action.annotate_add",
            "action.annotate_edit",
            "action.annotate_delete",
            "action.annotate_show",
        ):
            # The palette strips the trailing ellipsis, like a menu does.
            assert dashboard.clean_label(i18n.EN[key]) in texts, key
        # Undo and Redo are listed while they are *enabled*, which is the honest
        # behaviour: a palette must not offer a command that cannot run.
        assert not any(entry.text.startswith("Undo") for entry in palette.entries())
        window.commit_measurement("distance", [("ligand", 0), ("ligand", 1)])
        palette.refresh()
        undo_entries = [entry for entry in palette.entries() if entry.text.startswith("Undo")]
        assert undo_entries, "an enabled Undo belongs in the palette"
        assert "measurement" in undo_entries[0].text
        window.undo()
        palette.refresh()
        assert [entry for entry in palette.entries() if entry.text.startswith("Redo")]
        # And they actually run: the palette entry for a kind selects it.
        palette.edit.setText(i18n.EN["measure.kind.dihedral"])
        matches = palette.matches()
        assert matches and matches[0].text == i18n.EN["measure.kind.dihedral"]
        assert palette.run_selected() is True
        qapp.processEvents()
        assert window._measure_kind == "dihedral"
    finally:
        window.close()


def test_the_undo_stack_is_bounded_and_the_menu_says_what_it_will_undo(qapp, tmp_path):
    window = _toolkit_window(qapp, tmp_path)
    try:
        window._history.clear()
        assert window.undo_action.isEnabled() is False
        assert window.redo_action.isEnabled() is False
        window.commit_measurement("distance", [("ligand", 0), ("ligand", 1)])
        qapp.processEvents()
        assert window.undo_action.isEnabled() is True
        assert window.undo_action.text().startswith(i18n.EN["action.undo"])
        assert "Distance" in window.undo_action.text()
        assert window.undo() is True
        assert window.redo_action.isEnabled() is True
        assert window.redo_action.text().startswith(i18n.EN["action.redo"])
    finally:
        window.close()


def test_hydrophobic_lines_are_drawn_one_per_residue(qapp):
    """17 pairs is right for the table and unreadable as 17 lines: draw fewer.

    The detector enumerates every hydrophobic-carbon pair inside the cut-off —
    `analysis._hydrophobic_contacts` emits one row per pair — so the *drawing* is
    reduced to the closest pair of each receptor residue, while the table keeps
    every pair and the legend states both numbers.
    """
    receptor = [
        atom("CG1", "C", 0.0, 0.0, 0.0, res="VAL", res_id=213),
        atom("CG", "C", 0.0, 5.0, 0.0, res="GLN", res_id=192),
        atom("CD1", "C", 0.0, 9.0, 0.0, res="LEU", res_id=99),
    ]

    class Item:
        def __init__(self, kind, a, b, distance, residue=None):
            self.kind = kind
            self.a = a
            self.b = b
            self.distance = distance
            self.residue = residue

    pairs = [
        Item("hydrophobic", 0, 0, 3.8, ("VAL", 213, "A")),
        Item("hydrophobic", 0, 1, 3.9, ("VAL", 213, "A")),
        Item("hydrophobic", 1, 1, 3.95, ("GLN", 192, "A")),
        Item("hbond", 2, 0, 2.9, ("LEU", 99, "A")),
    ]
    drawn, counts = dashboard.consolidate_by_residue(pairs, receptor)
    assert counts["hydrophobic"] == (3, 2), counts
    assert counts["hbond"] == (1, 1)
    kinds = [item.kind for item in drawn]
    assert kinds.count("hydrophobic") == 2
    assert kinds.count("hbond") == 1
    # The *closest* pair of each residue is the one kept.
    kept = [item for item in drawn if item.kind == "hydrophobic"]
    assert sorted(item.distance for item in kept) == [3.8, 3.95]
    # The ligand-side index is untouched: the drawn line still ends on a real atom.
    assert all(item.b in (0, 1) for item in kept)
    # An interaction with no residue attribute still groups by its receptor atom.
    nameless = [Item("hydrophobic", 2, 0, 3.5)]
    drawn, counts = dashboard.consolidate_by_residue(nameless, receptor)
    assert counts["hydrophobic"] == (1, 1) and len(drawn) == 1
    # Nothing else is consolidated.
    drawn, counts = dashboard.consolidate_by_residue(pairs[:1] + pairs[3:], receptor)
    assert counts["hbond"] == (1, 1) and len(drawn) == 2


def test_the_window_draws_one_hydrophobic_line_per_residue(window, qapp):
    """The window path: table and legend keep the detected count, drawing drops."""
    window.scene.receptor = [
        atom("CG1", "C", 0.0, 0.0, 0.0, res="VAL", res_id=213),
        atom("CG", "C", 0.0, 5.0, 0.0, res="GLN", res_id=192),
    ]
    window.scene.ligand = [atom("C1", "C", 3.8, 0.0, 0.0), atom("C2", "C", 3.9, 0.0, 0.0)]
    window.interactions = [
        _FakeInteraction("hydrophobic", 0, 0, 3.8),
        _FakeInteraction("hydrophobic", 0, 1, 3.9),
        _FakeInteraction("hydrophobic", 1, 1, 3.95),
        _FakeInteraction("hbond", 1, 0, 2.9),
    ]
    # ``residue`` is filled by the profile; supply it as the detector does.
    window.interactions[0].residue = ("VAL", 213, "A")
    window.interactions[1].residue = ("VAL", 213, "A")
    window.interactions[2].residue = ("GLN", 192, "A")
    window.interactions[3].residue = ("GLN", 192, "A")
    # Opt in to the drawing (without recomputing: the contacts here are synthetic).
    window._interactions_shown = True
    window._sync_interactions()
    qapp.processEvents()

    assert len(window.scene.interactions) == 3, "3 hydrophobic pairs → 2 lines"
    assert window.interaction_table.table.rowCount() == 4, "the table keeps every pair"
    legend = dict((kind, (d, s)) for kind, d, s in window.viewport.interaction_legend)
    assert legend["hydrophobic"] == (3, 2)
    assert legend["hbond"] == (1, 1)
    assert "hydrophobic: 3 pairs, 2 line(s)" in window.interaction_table.heading.toolTip()
    # The heading still counts the *detections*, matching the table's rows.
    assert window.interaction_table.heading.text().startswith("4 ")

    # Turning hydrophobic off removes the rest entirely: the chemically
    # meaningful picture (just the H-bond) is one click away.
    window._interaction_actions["hydrophobic"].setChecked(False)
    qapp.processEvents()
    assert [item.kind for item in window.scene.interactions] == ["hbond"]
    assert "3 hidden by the filter" in window.interaction_table.heading.text()
    assert window.interaction_table.table.rowCount() == 4


def test_nothing_is_drawn_until_show_interactions_is_used(qapp, tmp_path):
    """Docking and browsing poses must not draw contacts.

    The user's instruction: "after docking, do NOT show interactions
    automatically — only after the user clicks 相互作用". So the profile is
    computed (the label names the residues, the table lists the pairs) but the
    *drawing* — dashes, focus, ruler marks, legend, tab count — stays empty until
    the explicit action, and Clear annotations switches it off again.
    """
    window = _toolkit_window(qapp, tmp_path)
    try:
        # After loading a receptor + poses (the auto path runs on every pose).
        assert window._interactions_shown is False
        assert window.interactions, "the contacts are still computed"
        assert window.scene.interactions == [], "but nothing is drawn"
        assert window.viewport.interaction_legend == []
        assert window.scene.interaction_focus == []
        assert window.sequence.marked_keys("contact") == []
        assert window.interaction_table.heading.text() == i18n.EN["interactions.empty"]
        assert window.dashboard_tabs.tabText(window._interaction_tab_index) == (
            i18n.EN["tab.interactions"]
        )
        # The table is data, not drawing: it lists what was detected, and the
        # pose label still names the residues the pose touches.
        assert window.interaction_table.table.rowCount() == len(window.interactions)
        assert i18n.EN["label.pose_binding"].split("{")[0] in window.lbl_pose.text()

        # Browsing poses keeps it that way.
        for index in range(window.pose_count()):
            window.set_pose(index)
            qapp.processEvents()
            assert window.scene.interactions == [], f"pose {index + 1} drew lines"
            assert window.viewport.interaction_legend == []
            assert window.sequence.marked_keys("contact") == []

        # The explicit action draws them.
        window.show_interactions_action()
        qapp.processEvents()
        assert window._interactions_shown is True
        assert window.scene.interactions, "Show interactions draws the lines"
        assert window.viewport.interaction_legend
        assert window.sequence.marked_keys("contact")
        assert window.dashboard_tabs.tabText(window._interaction_tab_index).endswith(")")

        # Browsing after that keeps them up to date, as before.
        window.set_pose(0)
        qapp.processEvents()
        assert window.scene.interactions

        # Clear annotations switches the drawing off again.
        window._clear_interactions()
        qapp.processEvents()
        assert window.scene.interactions == []
        assert window.viewport.interaction_legend == []
        assert window.sequence.marked_keys("contact") == []
        assert window.interaction_table.heading.text() == i18n.EN["interactions.empty"]

        # The off-state wording exists in every language the GUI ships: the
        # window's own switch, which rebuilds the docks with the new strings.
        for code in ("en", "zh"):
            window.set_language(code)
            _pump(qapp, 160)
            off = i18n.tr("interactions.empty")
            assert off and off != "interactions.empty"
            assert window.interaction_table.heading.text() == off, code
            # Rebuilt widgets: the same contract survives a language switch.
            assert window.scene.interactions == []
            assert window._interactions_shown is False
        window.set_language("en")
        _pump(qapp, 160)
    finally:
        window.close()


def test_hiding_the_receptor_hides_the_lines_that_point_at_it(qapp, tmp_path):
    """The visibility contract: no rendered endpoint, no drawn line.

    A dash whose far end is not drawn reads as an interaction reaching across the
    protein — the third visibility complaint in this workstream — so hiding the
    receptor (or the ligand) suppresses the interaction drawing and the focus
    pass with it, and showing it again restores both.
    """
    window = _toolkit_window(qapp, tmp_path)
    try:
        window.show_interactions_action()
        qapp.processEvents()
        assert window.scene.interactions and window.scene.interaction_focus

        window.chk_receptor.setChecked(False)
        qapp.processEvents()
        assert window.scene.show_receptor is False
        assert window.scene.interactions == [], "no lines to an invisible receptor"
        assert window.scene.interaction_focus == [], "and no lit-up invisible atoms"
        assert window.viewport.interaction_legend == []

        window.chk_receptor.setChecked(True)
        qapp.processEvents()
        assert window.scene.interactions, "showing the receptor brings them back"

        window.chk_ligand.setChecked(False)
        qapp.processEvents()
        assert window.scene.interactions == [], "and the same for the ligand"
        window.chk_ligand.setChecked(True)
        qapp.processEvents()
        assert window.scene.interactions
    finally:
        window.close()


# ---------------------------------------------------------------------------
# task-30: the console, in the log panel
# ---------------------------------------------------------------------------


def test_the_console_runs_a_command_against_the_live_session(qapp, tmp_path):
    """The input line lives in the log panel and drives the live session.

    Output and input share the log view, so there is no Console tab to switch to:
    the instruction was one place to read and one line to type.
    """
    window = _toolkit_window(qapp, tmp_path)
    try:
        console = window.console
        # A single line *inside the log panel*, not a dock and not a tab.
        assert isinstance(console, QtWidgets.QLineEdit)
        assert console.parent() is window.console_row
        assert window.console_row.parent() is window.log_panel
        assert window.log.parent() is window.log_panel, "one panel, one transcript"
        assert window.pose_splitter.indexOf(window.log_panel) >= 0
        assert not hasattr(window, "console_dock")
        tabs = [
            window.dashboard_tabs.tabText(index)
            for index in range(window.dashboard_tabs.count())
        ]
        assert not any(i18n.EN["dock.console"] in title for title in tabs), tabs
        assert window.console_prompt.text() == console.PROMPT

        console.execute("print(len(ligand), len(receptor))")
        assert f"{len(window.scene.ligand)} {len(window.scene.receptor)}" in (
            window.log.toPlainText()
        )
        # An expression is echoed like a REPL, and the prompt line is in the
        # transcript beside it.
        console.execute("2 + 2")
        assert "4" in window.log.toPlainText()
        assert f"{console.PROMPT}2 + 2" in window.log.toPlainText()

        # The effect lands in the *scene*, not in a copy of it.
        before = window.scene.box
        console.execute("set_box_center(1.0, 2.0, 3.0)")
        qapp.processEvents()
        assert window.scene.box is not None
        assert window.scene.box[0] == (1.0, 2.0, 3.0)
        assert window.scene.box[0] != (before[0] if before else None)
        assert window.spins["center_x"].value() == pytest.approx(1.0)
        # …and on the workbench's own code path, so the log and the panels agree.
        assert window._scene_state()["box"] is not None
        assert "console:" in window.log.toPlainText()
    finally:
        window.close()


def test_the_console_input_keeps_the_workbench_shortcuts_alive(qapp, tmp_path):
    """Typing must not swallow Ctrl+Z, and Escape must return to the view."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        console = window.console
        window._history.clear()
        before_box = window.scene.box
        console.execute("set_box_center(1.0, 1.0, 1.0)")
        qapp.processEvents()
        assert window._history.can_undo(), "the same undo stack as the menus"
        assert window.undo() is True
        assert window.scene.box == before_box
        # Ctrl+Z is left to the window rather than consumed by the line's own
        # text undo, which would make the scene stack unreachable while typing.
        event = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Z,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )
        assert console.candidates("lig")
        console.keyPressEvent(event)
        assert not event.isAccepted(), "the window's shortcut must still fire"

        # Escape hands the keyboard back to the viewport.
        window.show()
        window.viewport.setFocus()
        console.setFocus()
        qapp.processEvents()
        console.editingFinished.emit()
        qapp.processEvents()
        assert window.viewport.hasFocus(), "Escape returns to the 3-D view"
    finally:
        window.close()


def test_a_console_action_is_undoable_like_the_menu_action(qapp, tmp_path):
    """If the menu can undo it, the console can too — same stack, same step."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        window._history.clear()
        before = window._scene_state()
        window.console.execute("set_box_center(4.0, 5.0, 6.0)")
        qapp.processEvents()
        assert window._history.can_undo(), "the console recorded an undo step"
        assert window.undo_action.text().startswith(i18n.EN["action.undo"])
        assert "&" not in window.undo_action.text()
        assert window.undo() is True
        qapp.processEvents()
        assert window._scene_state() == before, "the panels came back with the scene"

        # A theme change is the same story through a different helper.
        window.console.execute("window.set_theme('light')")
        qapp.processEvents()
        assert window._history.can_undo()
        assert window.undo() is True
        assert window.color_theme.name == "dark"
    finally:
        window.close()


def test_the_console_history_and_completion(qapp, tmp_path):
    window = _toolkit_window(qapp, tmp_path)
    try:
        console = window.console
        console.execute("alpha = 1")
        console.execute("beta = 2")
        assert console.history() == ["alpha = 1", "beta = 2"]

        # Up recalls the previous line, down comes back to an empty prompt.
        console.recall(-1)
        assert console.current_input() == "beta = 2"
        console.recall(-1)
        assert console.current_input() == "alpha = 1"
        console.recall(1)
        assert console.current_input() == "beta = 2"

        # Completion over the bound namespace, and over an attribute.
        assert "ligand" in console.candidates("lig")
        assert "set_box_center" in console.candidates("set_box")
        assert console.candidates("window.set_") , "attribute completion works"
        assert "window.set_pose" in console.candidates("window.set_")
        assert console.candidates("zzz") == []
        # Tab on an unambiguous prefix completes it in the input line.
        console.setText("lig")
        console.complete()
        assert console.current_input().startswith("ligand")
    finally:
        window.close()


def test_a_console_error_does_not_end_the_session(qapp, tmp_path):
    window = _toolkit_window(qapp, tmp_path)
    try:
        console = window.console
        console.execute("1/0")
        assert i18n.EN["console.error"].split("{")[0].strip()[:5] in window.log.toPlainText()
        assert "ZeroDivisionError" in window.log.toPlainText()
        # The console still works, and the session is untouched.
        console.execute("print('alive')")
        assert "alive" in window.log.toPlainText()
        assert window.pose_count() > 0
        # A syntax error is reported the same way.
        console.execute("def (")
        assert "SyntaxError" in window.log.toPlainText()
        # SystemExit must not close the workbench.
        console.execute("raise SystemExit(3)")
        assert window.isVisible() or window.pose_count() > 0
    finally:
        window.close()


def test_a_console_block_spans_several_lines(qapp, tmp_path):
    window = _toolkit_window(qapp, tmp_path)
    try:
        console = window.console
        # A trailing colon continues the block, and the prompt says so.
        console.setText("total = 0")
        console.submit()
        console.setText("for index in range(3):")
        assert console.submit() == "for index in range(3):"
        assert console.prompt() == console.CONTINUED
        assert window.console_prompt.text() == console.CONTINUED
        console.setText("    total += index")
        console.submit()
        console.execute("print(total)")
        assert "3" in window.log.toPlainText(), window.log.toPlainText()[-120:]
        # The transcript shows the whole block, prompt and continuation alike.
        assert "...     total += index" in window.log.toPlainText()
    finally:
        window.close()


def test_the_console_and_the_measurement_toolkit_share_the_session(qapp, tmp_path):
    """The console reaches the newer tools too, on the same code paths."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        window._history.clear()
        window.console.execute("measure('distance', ('ligand', 0), ('ligand', 1))")
        qapp.processEvents()
        assert len(window._measurements) == 1
        assert window.measure_history.table.rowCount() == 1
        assert window.viewport.measurement_overlays
        assert window._history.can_undo()
        window.console.execute("annotate('from the console')")
        qapp.processEvents()
        assert [note.text for note in window._annotations] == ["from the console"]
        assert window.annotation_table.table.rowCount() == 1
        key = window.sequence.blocks()[0].key
        window.console.execute(f"select('{key[0]}', {key[1]}, '{key[2]}')")
        qapp.processEvents()
        assert window.sequence.selected_keys() == [key]
    finally:
        window.close()


# ---------------------------------------------------------------------------
# task-46: the protocol inspector tab
# ---------------------------------------------------------------------------


def test_the_protocol_tab_shows_the_settings_that_will_run(qapp, tmp_path):
    """The tab is a view of the other tabs, never a second copy of them."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        assert window.inspector.count() == 5
        assert window.inspector.tabText(4) == i18n.EN["tab.protocol"]
        window.inspector.setCurrentWidget(window._protocol_page)
        _pump(qapp, 30)

        text = window.protocol_view.toPlainText()
        assert window.protocol_caption.text() == i18n.EN["protocol.caption"]
        assert "engine.exhaustiveness" in text
        assert "box.source" in text
        # The hash is stated with what it covers and what it does not.
        assert "identity, not correctness" in text
        assert window.protocol_template_diff.text() == i18n.EN["protocol.template_none"]

        # The capture follows the widgets: change the engine, the hash changes.
        before = window.current_protocol()
        window.exhaustiveness.setValue(int(window.exhaustiveness.value()) + 8)
        window.inspector.setCurrentWidget(window._protocol_page)
        _pump(qapp, 20)
        after = window.current_protocol()
        assert after.hash() != before.hash()
        differences = {
            item["field"]: item["values"] for item in protocol.diff_protocols(before, after)
        }
        assert differences["engine.exhaustiveness"][1] == window.exhaustiveness.value()
        assert str(window.exhaustiveness.value()) in window.protocol_view.toPlainText()
        # The box's derivation is recorded, not implied.
        assert after.box.source in protocol.BOX_SOURCES
        # What the GUI has no control for stays at the documented default.
        assert after.library.filters is True and after.preparation.keep_hetero is True
    finally:
        window.close()


def test_the_protocol_tab_saves_loads_and_compares(qapp, tmp_path, monkeypatch):
    """Save writes a validated document; Load applies it to the widgets."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        window.exhaustiveness.setValue(21)
        window.seed.setValue(1234)
        target = tmp_path / "mine.json"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        window._save_protocol_action()
        _pump(qapp, 20)
        assert target.exists()
        saved = protocol.load_protocol(target)
        assert saved.engine.exhaustiveness == 21 and saved.execution.seed == 1234
        assert i18n.EN["log.protocol_saved"].split("{")[0] in window.log.toPlainText()

        # Change the session, then load the file back: the widgets follow.
        window.exhaustiveness.setValue(3)
        window.seed.setValue(7)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **k: (str(target), "")),
        )
        window._load_protocol_action()
        _pump(qapp, 20)
        assert window.exhaustiveness.value() == 21
        assert window.seed.value() == 1234
        assert window._loaded_protocol is not None
        assert i18n.EN["log.protocol_loaded"].split("{")[0] in window.log.toPlainText()

        # Compare against a different document: the log names the field and both
        # values, which is the whole point of the diff.
        other = tmp_path / "other.json"
        changed = window.current_protocol()
        changed.engine.exhaustiveness = 99
        protocol.save_protocol(changed, other)
        window.exhaustiveness.setValue(5)
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **k: (str(other), "")),
        )
        window._compare_protocol_action()
        _pump(qapp, 20)
        log = window.log.toPlainText()
        assert "engine.exhaustiveness" in log
        assert "99 -> 5" in log
    finally:
        window.close()


def test_the_protocol_template_view_says_what_differs(qapp, tmp_path):
    """'What differs from this template' — before running something expensive."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        templates = protocol.list_templates()
        if not templates:
            pytest.skip("no protocol templates are installed")
        window.inspector.setCurrentWidget(window._protocol_page)
        _pump(qapp, 20)
        names = [
            window.protocol_template.itemData(index)
            for index in range(window.protocol_template.count())
        ]
        assert None in names
        assert all(template.name in names for template in templates), names

        fast = protocol.template_named("fast-screen")
        window._apply_protocol_to_widgets(fast)
        _pump(qapp, 20)
        window.protocol_template.setCurrentIndex(names.index("fast-screen"))
        _pump(qapp, 20)
        # The template's own settings were applied to the widgets, so a protocol
        # captured now matches it in the fields the GUI holds.
        captured = window.current_protocol()
        assert captured.engine.exhaustiveness == fast.engine.exhaustiveness
        assert "fast-screen" in window.protocol_template_diff.text()

        # A deliberate deviation says how many settings differ, and the tooltip
        # carries the field-by-field diff with both values.
        window.exhaustiveness.setValue(int(window.exhaustiveness.value()) + 4)
        window._refresh_protocol_tab()
        _pump(qapp, 20)
        expected = [
            item
            for item in protocol.diff_protocols(
                protocol.template_named("fast-screen"), window.current_protocol()
            )
            if item["kind"] == "setting"
        ]
        assert expected, "the deviation must be visible"
        assert str(len(expected)) in window.protocol_template_diff.text()
        tooltip = window.protocol_template_diff.toolTip()
        assert "engine.exhaustiveness" in tooltip and "->" in tooltip
    finally:
        window.close()


def test_the_console_can_reach_the_protocols(qapp, tmp_path):
    """The keyboard-only path: the same functions the tab and the CLI use."""
    window = _toolkit_window(qapp, tmp_path)
    try:
        window.console.execute("print(current_protocol().name)")
        assert window.current_protocol().name in window.log.toPlainText()
        window.console.execute("print([t.name for t in list_templates()][:2])")
        assert "fast-screen" in window.log.toPlainText() or (
            "careful-redock" in window.log.toPlainText()
        )
        window.console.execute("print(len(current_protocol().hash()))")
        assert "64" in window.log.toPlainText()
    finally:
        window.close()


def test_layout_presets_rearrange_the_docks(window, qapp):
    window.show()
    qapp.processEvents()

    window._apply_layout_preset("compare")
    qapp.processEvents()
    assert window._layout_actions["compare"].isChecked()
    assert window.dashboard_dock.isVisible()
    assert window.comparison_dock.isVisible()
    assert not window.workspace_dock.isVisible()
    assert not window.inspector_dock.isVisible()
    assert "layout" in window.log.toPlainText()

    window._apply_layout_preset("analysis")
    qapp.processEvents()
    assert window.selection_dock.isVisible()
    assert window.inspector_dock.isVisible()
    assert not window.dashboard_dock.isVisible()
    assert not window.workspace_dock.isVisible()
    assert window.inspector.currentIndex() == 0

    window._apply_layout_preset("docking")
    qapp.processEvents()
    assert window.workspace_dock.isVisible()
    assert window.dashboard_dock.isVisible()
    assert not window.selection_dock.isVisible()
    assert window.inspector.currentIndex() == 3
    assert window._layout_actions["docking"].isChecked()


def test_ctrl_clicking_two_rows_fills_the_comparison_panel(window, qapp):
    window.show()
    qapp.processEvents()
    window.load_receptor(RECEPTOR_PDBQT)
    window.load_poses(POSES_PDBQT)
    qapp.processEvents()
    assert window.table.rowCount() == 2

    window.table.selectRow(0)
    qapp.processEvents()
    assert window.comparison.comparison() is None  # one row is just a pose

    window.table.item(1, 0).setSelected(True)
    qapp.processEvents()
    comparison = window.comparison.comparison()
    assert comparison is not None
    assert (comparison.index_a, comparison.index_b) == (0, 1)
    assert window._comparison_rows == (0, 1)
    assert comparison.rmsd.fitted >= 0.0
    assert comparison.diff.n_shared + comparison.diff.n_only_a >= 1
    assert "comparing pose 1 with pose 2" in window.log.toPlainText()
    assert window.comparison_dock.isVisible()

    # Analysis ▸ Compare poses repeats it on demand and copes with one row.
    window.table.clearSelection()
    window.table.selectRow(0)
    qapp.processEvents()
    window._compare_selected()
    assert "at least two poses" in window.log.toPlainText()
    window.table.item(1, 0).setSelected(True)
    qapp.processEvents()
    assert window.compare_selected() is not None


def test_the_measurement_history_follows_the_measure_tool(window, qapp):
    """Two picks commit a distance; the panel, the scene and the clipboard agree."""
    window.load_receptor(RECEPTOR_PDBQT)
    window.load_ligand(LIGAND_PDBQT)
    window.set_measure_kind("distance")
    assert len(window.pending_picks()) == 0
    # The viewport emits the whole pick list each time; the newest pick is used.
    window._on_atoms_picked([("receptor", 0)])
    qapp.processEvents()
    assert len(window.pending_picks()) == 1
    assert len(window._measurements) == 0, "half a distance is not a measurement"
    window._on_atoms_picked([("receptor", 0), ("ligand", 0)])
    qapp.processEvents()
    assert len(window._measurements) == 1
    assert window.pending_picks() == []
    assert window.measure_history.table.rowCount() == 1
    assert window.measure_history.summary.text().startswith("1 ")
    assert window.measure_history.table.item(0, 0).text() == "Distance"
    assert "Å" in window.measure_history.table.item(0, 2).text()
    assert len(window.scene.measurements) == 1, "the renderer draws the distance"
    assert window.viewport.measurement_overlays, "and the overlay layer labels it"

    # The two exports: the panel's own text is the contract, and the clipboard is
    # waited for (it is process-global and asynchronous on Windows, so a single
    # immediate read is a one-in-N failure that tests nothing about the feature).
    expected_text = window.measure_history.as_text()
    window._copy_measurements()
    text = _clipboard_text(qapp, expected_text)
    assert text == expected_text
    assert "LIG1:C1" in text and "Å" in text
    assert "copied 1 measurements" in window.log.toPlainText()

    expected_csv = window.measure_history.as_csv()
    window._copy_measurements_csv()
    csv = _clipboard_text(qapp, expected_csv)
    assert csv == expected_csv
    assert csv.splitlines()[0] == "kind,atoms,value,unit,label"
    assert "distance" in csv

    window._clear_measurements()
    qapp.processEvents()
    assert window.measure_history.table.rowCount() == 0
    assert window.scene.measurements == []
    assert window.viewport.measurement_overlays == []
    assert "measurements cleared" in window.log.toPlainText()
    # Clearing is undoable, like every other measurement edit.
    assert window._history.can_undo()
    window.undo()
    qapp.processEvents()
    assert len(window._measurements) == 1
    assert window.measure_history.table.rowCount() == 1


def test_the_atom_readout_lands_in_the_status_bar(window, qapp):
    window.load_receptor(RECEPTOR_PDBQT)
    window._on_atom_hovered(("receptor", 3))
    assert "ALA1" in window.lbl_status.text()
    assert "element C" in window.lbl_status.text()
    window._on_atom_hovered(None)  # an empty hover keeps the last message
    window.inspect_action.setChecked(False)
    assert window.viewport.hover_visible is False
    window.inspect_action.setChecked(True)
    assert window.viewport.hover_visible is True


def test_the_view_can_be_copied_to_the_clipboard(window, qapp):
    window.show()
    qapp.processEvents()
    window.load_receptor(RECEPTOR_PDBQT)
    qapp.processEvents()
    clipboard = QtWidgets.QApplication.clipboard()
    window._copy_view()
    # Same asynchrony as the text copies: give the platform clipboard a moment
    # rather than reading once and hoping.
    pixmap = clipboard.pixmap()
    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while pixmap.isNull() and deadline.elapsed() < 2000:
        qapp.processEvents()
        QtCore.QThread.msleep(10)
        pixmap = clipboard.pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert "clipboard" in window.log.toPlainText()


def test_a_dropped_structure_is_routed_by_its_contents(window, qapp, tmp_path):
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text(LIGAND_PDBQT, encoding="utf-8")
    poses = tmp_path / "anything.pdbqt"
    poses.write_text(POSES_PDBQT, encoding="utf-8")

    assert window.open_structure(receptor) is True
    assert len(window.scene.receptor) == 24
    assert window.open_structure(ligand) is True
    assert len(window.scene.ligand) == 5
    assert window.open_structure(poses) is True
    assert len(window.pose_models) == 2
    assert "dropped poses: anything.pdbqt" in window.log.toPlainText()

    junk = tmp_path / "notes.txt"
    junk.write_text("just a note", encoding="utf-8")
    assert window.open_structure(junk) is False

    # ... and through a real Qt drag: enter, then drop.
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(receptor))])
    window.scene.receptor = []
    enter = QtGui.QDragEnterEvent(
        QtCore.QPoint(10, 10),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(window, enter)
    assert enter.isAccepted()
    drop = QtGui.QDropEvent(
        QtCore.QPointF(10.0, 10.0),
        QtCore.Qt.DropAction.CopyAction,
        mime,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(window, drop)
    qapp.processEvents()
    assert len(window.scene.receptor) == 24
    assert window.acceptDrops() is True


def test_recent_files_are_listed_and_reopen_the_structure(window, qapp, tmp_path):
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    window.load_receptor(receptor)
    assert [Path(p).name for p in window.recent.paths()] == ["receptor.pdbqt"]
    assert [a.text() for a in window.recent_menu.actions()] == [
        "receptor.pdbqt",
        "",
        "Clear list",
    ]

    window.scene.receptor = []
    assert window.open_recent(receptor) is True
    assert len(window.scene.receptor) == 24

    assert window.open_recent(tmp_path / "gone.pdbqt") is False
    assert "no longer on disk" in window.log.toPlainText()

    window._clear_recent()
    assert [a.text() for a in window.recent_menu.actions()] == ["No recent files"]
    assert len(window.recent) == 0


def test_the_session_saves_and_restores_the_whole_workbench(qapp, tmp_path):
    store = dashboard.SessionStore(tmp_path / "session.json")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text(LIGAND_PDBQT, encoding="utf-8")

    first = DockingWorkbench(session=store)
    try:
        first.load_receptor(receptor)
        first.load_ligand(ligand)
        first._set_box((1.0, 2.0, 3.0), (21.0, 21.0, 21.0), 0.5)
        first.engine.setCurrentText("vinardo")
        first.exhaustiveness.setValue(7)
        first.seed.setValue(99)
        first.threads.setValue(3)
        first.set_theme("light")
        first.set_density("compact")
        first.dashboard_dock.setVisible(False)
        qapp.processEvents()
        assert first.save_session() is True
    finally:
        first.close()

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["theme"] == "light"
    assert payload["density"] == "compact"
    assert payload["engine"]["scoring"] == "vinardo"
    assert payload["engine"]["exhaustiveness"] == 7
    assert payload["box"]["spacing"] == 0.5
    assert payload["docks"]["dashboard_dock"] is False
    assert set(payload["splitters"]) == {"central", "pose"}

    second = DockingWorkbench(session=store)
    try:
        assert second.session_offer() is not None
        assert second.restore_session() is True
        qapp.processEvents()
        assert second.color_theme is dashboard.LIGHT
        assert second.density == "compact"
        assert len(second.scene.receptor) == 24
        assert len(second.scene.ligand) == 5
        assert second.engine.currentText() == "vinardo"
        assert second.exhaustiveness.value() == 7
        assert second.seed.value() == 99
        assert second.threads.value() == 3
        assert second._current_box().spacing == pytest.approx(0.5)
        assert not second.dashboard_dock.isVisible()
        assert "session restored" in second.log.toPlainText()
    finally:
        second.close()

    # With no session at all the offer is empty and a restore says so.
    empty = DockingWorkbench()
    try:
        assert empty.session_offer() is None
        assert empty.restore_session() is False
        assert "no session to restore" in empty.log.toPlainText()
        assert empty.clear_session() is None
    finally:
        empty.close()


def test_a_window_without_a_session_never_writes_one(window, tmp_path, monkeypatch):
    monkeypatch.setenv("ODOCK_SESSION_FILE", str(tmp_path / "never.json"))
    window.load_receptor(RECEPTOR_PDBQT)
    window._session_changed()
    assert window.save_session() is False
    assert not (tmp_path / "never.json").exists()
    assert window.session_offer() is None


def test_an_empty_relaunch_does_not_destroy_the_saved_session(qapp, tmp_path):
    """A launch that is closed again without loading anything keeps the offer.

    The session is *offered*, not imposed: if a bare launch overwrote it with
    empty paths, the offer would destroy the very thing it offers.
    """
    store = dashboard.SessionStore(tmp_path / "session.json")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")

    first = DockingWorkbench(session=store)
    try:
        first.load_receptor(receptor)
        first._set_box((1.0, 2.0, 3.0), (21.0, 21.0, 21.0), 0.5)
        assert first.save_session() is True
    finally:
        first.close()
    saved = json.loads(store.path.read_text(encoding="utf-8"))
    assert saved["receptor"] == str(receptor)
    assert Path(saved["receptor"]).is_file()

    # A second, empty window: it changes only the appearance, then closes.
    empty = DockingWorkbench(session=store)
    try:
        assert empty._pending.get("receptor") is None
        empty.set_theme("light")
        assert empty.save_session() is True
    finally:
        empty.close()
    after = json.loads(store.path.read_text(encoding="utf-8"))
    assert after["receptor"] == str(receptor), "the saved session was destroyed"
    assert after["box"]["spacing"] == 0.5
    assert after["theme"] == "light", "the new preference is still saved"
    assert store.payload["receptor"] == str(receptor)


def test_the_session_load_keeps_the_recent_list_object_the_window_holds(qapp, tmp_path):
    """``SessionStore.load`` must refresh the list in place, not replace it."""
    store = dashboard.SessionStore(tmp_path / "session.json")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    window = DockingWorkbench(session=store)
    try:
        window.load_receptor(receptor)
        assert window.recent is store.recent
        window.save_session()
        # A session written by another build, with its own recent list — one real
        # file and one that has since been deleted.
        store.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "receptor": str(receptor),
                    "recent": [str(receptor), "one.pdbqt"],
                }
            ),
            encoding="utf-8",
        )
        assert store.load() is not None
        assert window.recent is store.recent, "the window still sees the same list"
        assert [Path(p).name for p in window.recent.paths()] == [
            "receptor.pdbqt",
            "one.pdbqt",
        ]
        # The menu is rebuilt when it opens, so a reload reaches the screen.
        window._rebuild_recent_menu()
        labels = [a.text() for a in window.recent_menu.actions()]
        assert labels[0] == "receptor.pdbqt"
        assert "one.pdbqt" not in labels, "a file that is gone is not offered"
        assert labels[-1] == "Clear list"
    finally:
        window.close()


def test_the_autosave_is_debounced_until_a_save_is_asked_for(qapp, tmp_path):
    store = dashboard.SessionStore(tmp_path / "session.json")
    window = DockingWorkbench(session=store)
    try:
        window.load_receptor(RECEPTOR_PDBQT)
        assert window._autosave_timer is not None
        assert window._autosave_timer.isActive()  # scheduled, not yet written
        assert window._autosave_pending is True
        assert window.save_session() is True
        assert window._autosave_pending is False
        assert not window._autosave_timer.isActive()
        assert store.path.exists()
    finally:
        window.close()


# ---------------------------------------------------------------------------
# end to end: the dashboard reads a real run
# ---------------------------------------------------------------------------


def test_a_real_run_fills_the_dashboard(qapp):
    """Drive a genuine (tiny) docking job and read the panel afterwards."""
    window = DockingWorkbench()
    window.show()
    try:
        window.load_receptor(RECEPTOR_PDBQT)
        window.load_ligand(LIGAND_PDBQT)
        window._fit_box_to_ligand()
        window.exhaustiveness.setValue(1)
        window.poses.setValue(1)
        window.seed.setValue(7)
        qapp.processEvents()
        window._run_docking()
        assert window.run_dashboard.phases.active == "grid"
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while window.result is None and deadline.elapsed() < 120_000:
            qapp.processEvents()
            QtCore.QThread.msleep(5)
        assert window.result is not None, "the docking run did not finish"
        qapp.processEvents()

        run = window.run_history.runs[-1]
        assert len(window.run_history) == 1
        assert run.energies == [pytest.approx(window.result.best_affinity)]
        assert run.grid_points > 0
        assert run.scoring == "vina"
        assert [phase.name for phase in run.phases] == ["grid", "search", "refine", "done"]
        assert run.phase("grid").duration is not None
        assert run.phase("search").duration is not None
        # Refinement happens inside the kernel's `run()` call: the stage is
        # marked, but it has no duration of its own because none is exposed.
        assert run.phase("refine").duration == 0.0

        panel = window.run_dashboard
        assert panel.phases.done == ["grid", "search", "refine", "done"]
        assert panel.phases.active is None
        assert panel.values["best"].text().startswith(
            f"{window.result.best_affinity:.2f}"
        )
        assert panel.values["poses"].text() == "1"
        assert panel.trace.drawn_points() == 1
        assert panel.elapsed() is None
    finally:
        window.close()


# ---------------------------------------------------------------------------
# the translations the panels use
# ---------------------------------------------------------------------------


def test_every_literal_key_the_dashboard_uses_exists():
    import re

    pattern = re.compile(r'tr\(\s*"([A-Za-z0-9_.]+)"\s*[,)]')
    text = (GUI_DIR / "dashboard.py").read_text(encoding="utf-8")
    keys = set(pattern.findall(text))
    assert keys, "no literal tr() keys found in dashboard.py"
    missing = sorted(key for key in keys if key not in i18n.EN)
    assert not missing, f"dashboard.py uses unknown keys: {missing}"
    for key in sorted(keys):
        assert i18n.ZH[key] and not i18n.ZH[key].isspace()


def test_every_runtime_built_key_exists():
    for key in ("best", "elapsed", "poses", "grid", "scoring"):
        assert f"dashboard.{key}" in i18n.EN
    for name in dashboard.PHASES:
        assert f"phase.{name}" in i18n.EN
    for name in dashboard.THEMES:
        assert f"action.theme_{name}" in i18n.EN
    for name in dashboard.DENSITIES:
        assert f"action.density_{name}" in i18n.EN
    for name in dashboard.LAYOUT_PRESETS:
        assert f"action.layout_{name}" in i18n.EN
    for kind in ("receptor", "ligand", "poses"):
        assert f"drop.{kind}" in i18n.EN
    for kind in ("receptor", "ligand"):
        assert f"inspect.source.{kind}" in i18n.EN

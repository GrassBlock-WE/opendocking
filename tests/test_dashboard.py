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

from odock.gui import dashboard, i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui.structure import Atom  # noqa: E402

GUI_DIR = Path(__file__).resolve().parent.parent / "python" / "odock" / "gui"

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


@pytest.fixture
def window(qapp):
    """A workbench with no session file: nothing is written to disk."""
    win = DockingWorkbench()
    win.resize(1280, 820)
    yield win
    win.close()


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
    assert "No measurements" in history.summary.text()
    history.set_measurements(
        [
            {"kind": "distance", "a": "TYR337:OH", "b": "LIG1:C1", "value": 2.845},
            {"kind": "distance", "a": "ASP189:OD1", "b": "LIG1:N1", "value": 3.1},
        ]
    )
    assert history.table.rowCount() == 2
    assert history.table.item(0, 2).text() == "2.845"
    assert "2 distances" in history.summary.text()
    text = history.as_text()
    assert text.splitlines()[0].startswith("Atom A\tAtom B")
    assert "TYR337:OH" in text and "2.845" in text
    history.retranslate()
    history.set_measurements([])
    assert history.table.rowCount() == 0


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
    window.pose_slider.setValue(1)
    qapp.processEvents()

    text = window.lbl_pose.text()
    assert len(text) > 40, text
    assert window.pose_label_text() == text, "the read-out keeps every character"
    assert window.lbl_pose.toolTip() == text
    assert window.lbl_pose.wordWrap()
    assert window.lbl_pose.maximumWidth() == window.POSE_LABEL_WIDTH
    assert window.lbl_pose.minimumSizeHint().width() <= window.POSE_LABEL_WIDTH + 8
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

    QtWidgets.QApplication.clipboard().setText("")
    window.comparison.btn_copy.click()
    qapp.processEvents()
    text = QtWidgets.QApplication.clipboard().text()
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
    window.load_receptor(RECEPTOR_PDBQT)
    window.load_ligand(LIGAND_PDBQT)
    window._on_atoms_picked([("receptor", 0), ("ligand", 0)])
    qapp.processEvents()
    assert len(window._measurements) == 1
    assert window.measure_history.table.rowCount() == 1
    assert "1 distances" in window.measure_history.summary.text()

    window._copy_measurements()
    text = QtWidgets.QApplication.clipboard().text()
    assert "LIG1:C1" in text and "Å" in text
    assert "copied 1 measurements" in window.log.toPlainText()

    window._clear_measurements()
    qapp.processEvents()
    assert window.measure_history.table.rowCount() == 0
    assert window.scene.measurements == []
    assert "measurements cleared" in window.log.toPlainText()


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
    window._copy_view()
    pixmap = QtWidgets.QApplication.clipboard().pixmap()
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

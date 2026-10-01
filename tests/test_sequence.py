# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the residue ruler under the 3-D viewport.

The ruler answers "where am I in the sequence?" and doubles as a picker, so the
tests pin three things: the numbering rule (a tick and a number every five
residues, counted from the first residue of the row), the one-letter codes and
the non-polymer names, and the click / shift-click / ctrl-click semantics that
drive the 3-D selection and the atom list.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtCore = pytest.importorskip("PyQt6.QtCore")
QtGui = pytest.importorskip("PyQt6.QtGui")
QTest = pytest.importorskip("PyQt6.QtTest").QTest

from odock.gui.sequence import (  # noqa: E402
    AMINO_ACIDS,
    STANDARD_AMINO_ACIDS,
    RulerPalette,
    SequenceTrack,
    class_color,
    one_letter,
    residue_class,
)
from odock.gui.structure import Atom, parse_pdbqt  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    """One QApplication for the module (autouse: every test makes widgets)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_atoms(residues, chain: str = "A", start: int = 1, group_shift: float = 0.0):
    """Atoms for ``residues`` = ``[(name, n_atoms), ...]``, one residue each."""
    atoms = []
    for offset, (name, count) in enumerate(residues):
        for index in range(count):
            atoms.append(
                Atom(
                    name="CA",
                    element="C",
                    res_name=name,
                    res_id=start + offset,
                    chain=chain,
                    x=float(index) + group_shift,
                    y=float(offset),
                    z=0.0,
                    charge=0.0,
                    ad_type="C",
                )
            )
    return atoms


def track(residues=None, *, start: int = 1, chain: str = "A", n: int = 12):
    """A ruler over ``n`` alanine residues (unless ``residues`` is given)."""
    widget = SequenceTrack()
    widget.set_structure(
        make_atoms(residues or [("ALA", 1)] * n, chain=chain, start=start)
    )
    return widget


# ---------------------------------------------------------------------------
# nomenclature
# ---------------------------------------------------------------------------


def test_all_twenty_standard_residues_have_their_one_letter_code():
    expected = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    assert len(STANDARD_AMINO_ACIDS) == 20
    for name, code in expected.items():
        assert one_letter(name) == code, name
    assert {one_letter(name) for name in expected} == set("ACDEFGHIKLMNPQRSTVWY")


def test_non_polymer_entities_keep_their_real_name():
    for name in ("BEN", "CA", "ZN", "HOH", "NAD", "SO4", "MG"):
        assert one_letter(name) is None, name
    ruler = track([("ALA", 1), ("BEN", 1), ("ZN", 1), ("HOH", 1), ("NAD", 1)])
    labels = [block.label for block in ruler.blocks()]
    assert labels == ["A", "BEN", "ZN", "HOH", "NAD"]


def test_every_colour_class_has_a_colour_and_covers_the_codes():
    seen = {residue_class("X", code) for code in AMINO_ACIDS.values()}
    assert seen <= set(class_color(name) and name for name in seen)
    for name in ("hydrophobic", "polar", "acidic", "basic", "gly", "pro", "cys"):
        red, green, blue = class_color(name)
        assert 0.0 <= red <= 1.0 and 0.0 <= green <= 1.0 and 0.0 <= blue <= 1.0
    assert residue_class("ASP") == "acidic"
    assert residue_class("LYS") == "basic"
    assert residue_class("SER") == "polar"
    assert residue_class("LEU") == "hydrophobic"
    assert residue_class("GLY") == "gly"
    assert residue_class("PRO") == "pro"
    assert residue_class("CYS") == "cys"
    assert residue_class("HOH") == "water"
    assert residue_class("BEN") == "other"


# ---------------------------------------------------------------------------
# the ruler
# ---------------------------------------------------------------------------


def test_one_cell_per_residue_in_numeric_order(qapp):
    ruler = SequenceTrack()
    # Deliberately out of order in the file: the ruler sorts by residue number.
    atoms = make_atoms([("GLY", 1)], start=3) + make_atoms([("ALA", 1)], start=1)
    ruler.set_structure(atoms)
    assert [(block.res_name, block.res_id) for block in ruler.blocks()] == [
        ("ALA", 1),
        ("GLY", 3),
    ]


def test_the_ticks_land_every_five_residues(qapp):
    ruler = track(n=40)
    labels = [res_id for _label, res_id in ruler.tick_labels()]
    assert labels == [1, 5, 10, 15, 20, 25, 30, 35, 40]
    # Five residues of one cell width between two interval boundaries.
    xs = [x for _label, x, _res in ruler.ticks()]
    for first, second in zip(xs[1:], xs[2:]):
        assert second - first == 5 * SequenceTrack.CELL_WIDTH


def test_the_ticks_follow_a_numbering_that_does_not_start_at_one(qapp):
    # 3PTB starts at residue 16, so the first interval is 16..20 and the first
    # boundary tick is labelled 20.
    ruler = track(n=15, start=16)
    labels = [res_id for _label, res_id in ruler.tick_labels()]
    assert labels == [16, 20, 25, 30]


def test_the_last_residue_number_is_always_labelled(qapp):
    ruler = track(n=7)  # one full interval plus two residues
    assert [res_id for _label, res_id in ruler.tick_labels()] == [1, 5, 7]


def test_a_cell_is_one_cell_wide_and_an_entity_is_wider(qapp):
    ruler = track([("ALA", 1), ("ALA", 1), ("BEN", 1), ("ALA", 1)])
    polymer = [block for block in ruler.blocks() if block.kind == "amino"]
    entities = ruler.tail()
    assert len(polymer) == 3 and len(entities) == 1
    assert all(
        ruler.cell_rect(block.index).width() == SequenceTrack.CELL_WIDTH - 1
        for block in polymer
    )
    assert all(
        ruler.cell_rect(block.index).width() == SequenceTrack.ENTITY_WIDTH - 1
        for block in entities
    )


def test_multiple_chains_get_one_row_each(qapp):
    ruler = SequenceTrack()
    ruler.set_structure(make_atoms([("ALA", 1)] * 3, chain="A", start=1))
    ruler.set_structure(make_atoms([("GLY", 1)] * 4, chain="B", start=10))
    # A new receptor replaces the old rows.
    assert [label for label, _cells in ruler.rows()] == ["B"]
    ruler.set_structure(
        make_atoms([("ALA", 1)] * 3, chain="A", start=1)
        + make_atoms([("GLY", 1)] * 4, chain="B", start=10)
    )
    rows = ruler.rows()
    assert [label for label, _cells in rows] == ["A", "B"]
    assert [len(cells) for _label, cells in rows] == [3, 4]


def test_the_entities_ride_in_one_tail_at_the_end(qapp):
    """A ligand is a cell of the tail, not a row of its own."""
    ruler = SequenceTrack()
    ruler.set_structure(make_atoms([("ALA", 1)] * 3, chain="A"))
    ruler.set_structure(make_atoms([("BEN", 3)], chain="A", start=1), "ligand")
    # One row per polymer chain, and the ligand rides at the end of it.
    assert [label for label, _cells in ruler.rows()] == ["A"]
    assert [block.label for block in ruler.tail()] == ["BEN"]
    assert ruler.tail_keys() == [("A", 1, "BEN")]
    polymer = [block.index for block in ruler.blocks() if block.kind == "amino"]
    assert ruler.tail()[0].index > max(polymer), "the tail comes after every residue"
    assert len(ruler.blocks()) == 4
    # Reloading the receptor drops the ligand that belonged to the old one.
    ruler.set_structure(make_atoms([("ALA", 1)] * 2, chain="A"))
    assert [label for label, _cells in ruler.rows()] == ["A"]
    assert ruler.tail() == []
    assert len(ruler.blocks()) == 2


def test_the_tail_carries_no_numbers_and_keeps_real_names(qapp):
    ruler = track([("ALA", 1)] * 12 + [("BEN", 2), ("ZN", 1), ("HOH", 1)])
    assert [block.label for block in ruler.tail()] == ["BEN", "ZN", "HOH"]
    assert [block.res_name for block in ruler.tail()] == ["BEN", "ZN", "HOH"]
    # The numbering still belongs to the polymer, and stops at its last residue.
    assert [res_id for _label, res_id in ruler.tick_labels()] == [1, 5, 10, 12]
    tail_left = min(ruler.cell_rect(block.index).left() for block in ruler.tail())
    assert all(x < tail_left for _label, x, _res in ruler.ticks()), (
        "no tick may sit over the entity tail"
    )
    # A gap sets the tail apart from the residue run.
    last_residue_right = max(
        ruler.cell_rect(block.index).right()
        for block in ruler.blocks()
        if block.kind == "amino"
    )
    assert tail_left > last_residue_right


def test_a_track_with_a_ligand_still_has_one_row_per_chain(qapp):
    ruler = SequenceTrack()
    ruler.set_structure(
        make_atoms([("ALA", 1)] * 3, chain="A", start=1)
        + make_atoms([("GLY", 1)] * 4, chain="B", start=10)
    )
    ruler.set_structure(make_atoms([("BEN", 2), ("NAD", 3)], chain="A"), "ligand")
    rows = ruler.rows()
    assert [label for label, _cells in rows] == ["A", "B"], (
        "the ligand must not add a row"
    )
    assert [len(ruler.tail())] == [2]
    # Only the last row carries the tail.
    assert [block.kind != "amino" for block in rows[0][1]] == [False] * 3
    assert any(block.kind != "amino" for block in rows[1][1])
    # The order is deterministic: chain then residue number.
    assert [block.res_name for block in ruler.tail()] == ["BEN", "NAD"]
    assert ruler.tail_keys() == sorted(ruler.tail_keys(), key=lambda key: (key[0], key[1]))


def test_clear_empties_the_ruler(qapp):
    ruler = track(n=5)
    assert ruler.blocks()
    ruler.clear()
    assert ruler.blocks() == []
    assert ruler.selected_keys() == []
    assert ruler.tick_labels() == []


def test_the_ruler_scales_to_a_large_receptor(qapp):
    ruler = track(n=2000)
    assert len(ruler.blocks()) == 2000
    # 13 px per residue, so it scrolls rather than squeezing the codes together.
    assert ruler.minimumWidth() >= 2000 * SequenceTrack.CELL_WIDTH
    assert ruler.track_height() <= 4 * (SequenceTrack.TICK_HEIGHT + SequenceTrack.CODE_HEIGHT)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _click(ruler, index, modifier=QtCore.Qt.KeyboardModifier.NoModifier):
    QTest.mouseClick(
        ruler, QtCore.Qt.MouseButton.LeftButton, modifier, ruler.cell_center(index)
    )


def test_a_click_selects_and_a_second_click_clears(qapp):
    ruler = track(n=10)
    seen = []
    ruler.selectionChanged.connect(lambda keys: seen.append(list(keys)))
    _click(ruler, 3)
    keys = ruler.selected_keys()
    assert keys == [ruler.blocks()[3].key]
    assert seen[-1] == keys
    _click(ruler, 3)
    assert ruler.selected_keys() == []
    assert seen[-1] == []


def test_a_click_replaces_the_selection(qapp):
    ruler = track(n=10)
    _click(ruler, 3)
    _click(ruler, 6)
    assert ruler.selected_indices() == [6]


def test_shift_click_selects_the_range_from_the_anchor(qapp):
    ruler = track(n=10)
    _click(ruler, 2)
    _click(ruler, 6, QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert ruler.selected_indices() == [2, 3, 4, 5, 6]
    # And backwards, from the same anchor.
    _click(ruler, 2)
    _click(ruler, 0, QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert ruler.selected_indices() == [0, 1, 2]


def test_ctrl_click_adds_and_removes_one_residue(qapp):
    ruler = track(n=10)
    _click(ruler, 2)
    _click(ruler, 5, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert ruler.selected_indices() == [2, 5]
    _click(ruler, 7, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert ruler.selected_indices() == [2, 5, 7]
    _click(ruler, 5, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert ruler.selected_indices() == [2, 7]


def test_a_click_on_empty_space_clears_the_selection(qapp):
    ruler = track(n=4)
    ruler.resize(600, ruler.track_height())
    _click(ruler, 1)
    assert ruler.selected_keys()
    QTest.mouseClick(
        ruler,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QPoint(2, ruler.height() - 2),
    )
    assert ruler.selected_keys() == []


def test_selected_keys_round_trip(qapp):
    ruler = track(n=12)
    keys = [ruler.blocks()[1].key, ruler.blocks()[5].key]
    ruler.select_keys(keys, emit=False)
    assert ruler.selected_keys() == keys
    again = SequenceTrack()
    again.set_structure(make_atoms([("ALA", 1)] * 12))
    again.select_keys(ruler.selected_keys(), emit=False)
    assert again.selected_keys() == keys


def test_atom_refs_follow_the_selection(qapp):
    ruler = SequenceTrack()
    ruler.set_structure(make_atoms([("ALA", 3), ("GLY", 2)]))
    ruler.select_keys([ruler.blocks()[0].key], emit=False)
    assert ruler.atom_refs() == [("receptor", 0), ("receptor", 1), ("receptor", 2)]
    ruler.select_keys(ruler.block_keys(), emit=False)
    assert ruler.atom_refs() == [("receptor", index) for index in range(5)]


def test_arrow_keys_move_the_selection(qapp):
    ruler = track(n=6)
    ruler.select_keys([ruler.blocks()[1].key], emit=False)
    QTest.keyClick(ruler, QtCore.Qt.Key.Key_Right)
    assert ruler.selected_indices() == [2]
    QTest.keyClick(ruler, QtCore.Qt.Key.Key_Right)
    assert ruler.selected_indices() == [3]
    QTest.keyClick(ruler, QtCore.Qt.Key.Key_Left)
    assert ruler.selected_indices() == [2]


def test_a_double_click_activates_the_residue(qapp):
    ruler = SequenceTrack()
    ruler.set_structure(make_atoms([("ALA", 2), ("GLY", 1)]))
    seen = []
    ruler.residueActivated.connect(lambda refs: seen.append(list(refs)))
    QTest.mouseDClick(
        ruler,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        ruler.cell_center(0),
    )
    assert seen == [[("receptor", 0), ("receptor", 1)]]
    assert ruler.selected_indices() == [0]


def test_the_ruler_paints_and_keeps_its_pixels(qapp):
    """A smoke test for the painter: it must lay out and paint without raising."""
    ruler = track([("ALA", 1)] * 6 + [("BEN", 2), ("ZN", 1)])
    ruler.resize(ruler.minimumWidth(), ruler.track_height())
    ruler.select_keys([ruler.blocks()[2].key], emit=False)
    ruler.show()
    qapp.processEvents()
    pixmap = ruler.grab()
    assert pixmap.width() == ruler.width()
    assert pixmap.height() == ruler.height()
    image = pixmap.toImage()
    # The selected cell is drawn in a brighter colour than an unselected one.
    selected = image.pixelColor(ruler.cell_center(2))
    plain = image.pixelColor(ruler.cell_center(4))
    assert selected != plain
    ruler.close()


# ---------------------------------------------------------------------------
# the window: selection -> 3-D highlight -> atom list -> Copy as PDBQT
# ---------------------------------------------------------------------------

DEMO = Path(__file__).resolve().parent.parent / "demo" / "3ptb"


@pytest.fixture()
def window(qapp):
    """The real workbench with the bundled 3PTB receptor and ligand."""
    from odock.gui import i18n
    from odock.gui.app import DockingWorkbench, _configure_surface_format

    receptor, ligand = DEMO / "receptor.pdbqt", DEMO / "ligand.pdbqt"
    if not receptor.exists() or not ligand.exists():  # pragma: no cover
        pytest.skip(f"missing demo structure {receptor}")
    i18n.set_language("en")
    _configure_surface_format()
    workbench = DockingWorkbench(
        receptor=receptor.read_text(encoding="utf-8"),
        ligand=ligand.read_text(encoding="utf-8"),
    )
    workbench.resize(1200, 800)
    workbench.show()
    qapp.processEvents()
    yield workbench
    i18n.set_language("en")
    workbench.close()


def test_the_ruler_shows_the_receptor_and_its_ligand(window):
    ruler = window.sequence
    rows = [label for label, _cells in ruler.rows()]
    assert rows == ["A"], "one row per chain: the entities ride in the tail"
    polymer = [block for block in ruler.blocks() if block.kind == "amino"]
    assert len(polymer) > 200
    # 3PTB is numbered from 16, so the first interval is 16..20.
    labels = [res_id for _label, res_id in ruler.tick_labels()]
    assert labels[:4] == [16, 20, 25, 30]
    assert polymer[0].res_id == 16
    entities = {block.label for block in ruler.tail()}
    assert "BEN" in entities, entities
    assert max(block.index for block in polymer) < min(
        block.index for block in ruler.tail()
    )


def test_a_selection_reaches_the_scene_and_the_atom_list(window):
    ruler = window.sequence
    target = next(block for block in ruler.blocks() if block.res_name == "GLY")
    ruler.select_keys([target.key])

    refs = ruler.atom_refs()
    assert refs and all(group == "receptor" for group, _index in refs)
    assert window.scene.selection == refs

    table = window.selection_table
    assert table.rowCount() == len(refs)
    assert table.columnCount() == 9
    rows = [
        [table.item(row, column).text() for column in range(table.columnCount())]
        for row in range(table.rowCount())
    ]
    assert all(row[2].startswith("GLY") for row in rows)
    assert all(row[3] == target.chain for row in rows)
    names = {row[0] for row in rows}
    assert "N" in names and "CA" in names
    nitrogen = next(row for row in rows if row[0] == "N")
    assert nitrogen[1] == "N"
    assert nitrogen[7] == "N"  # the AD4 type of the backbone nitrogen
    assert nitrogen[8].startswith("-")  # its charge, signed

    counts: dict = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    summary = window.lbl_selection_summary.text()
    assert f"· {len(refs)} atom(s)" in summary
    for element, count in counts.items():
        assert f"{element} {count}" in summary
    assert "selection:" in window.log.toPlainText()


def test_a_structure_change_keeps_the_marks_that_still_exist(window):
    """Reloading a structure keeps the picked residues; stale ones go.

    Switching pose reloads the ligand (fresh atom objects), and choosing a pose
    must not throw the selection away — so the marks survive by residue key and
    the atom references follow the new lists.
    """
    ruler = window.sequence
    receptor_block = next(block for block in ruler.blocks() if block.kind == "amino")
    ruler.select_keys([receptor_block.key])
    assert window.selection_table.rowCount() > 0

    ligand = (DEMO / "ligand.pdbqt").read_text(encoding="utf-8")
    window.load_ligand(ligand)
    assert ruler.selected_keys() == [receptor_block.key], "the mark survives"
    assert window.scene.selection == ruler.atom_refs()

    # A different structure whose residues do not exist any more drops them.
    window.load_receptor("ATOM      1  CA  ALA A1000     0.000   0.000   0.000  "
                         "1.00  0.00     0.000  C\nTER\n")
    assert ruler.selected_keys() == []
    assert window.scene.selection == []
    assert window.selection_table.rowCount() == 0


def test_copy_as_pdbqt_writes_a_valid_block(window, tmp_path, monkeypatch):
    from odock.gui.structure import parse_pdbqt

    target = tmp_path / "selection.pdbqt"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "")),
    )
    ruler = window.sequence
    block = next(item for item in ruler.blocks() if item.kind == "amino")
    ruler.select_keys([block.key])
    refs = ruler.atom_refs()
    assert refs

    window._copy_selection_pdbqt()

    assert target.exists()
    text = target.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "ROOT"
    assert lines[-2:] == ["ENDROOT", "TORSDOF 0"]
    assert sum(1 for line in lines if line.startswith("ATOM")) == len(refs)
    # It has to be readable back: the point of the export.
    models = parse_pdbqt(text)
    assert len(models[0].atoms) == len(refs)
    assert {atom.serial for atom in models[0].atoms} == set(range(1, len(refs) + 1))
    assert "wrote" in window.log.toPlainText()


def test_copy_without_a_selection_says_so(window):
    window.sequence.clear_selection()
    before = window.log.toPlainText().count("select a residue")
    window._copy_selection_pdbqt()
    assert window.log.toPlainText().count("select a residue") == before + 1


def test_a_double_click_centres_the_camera(window):
    ruler = window.sequence
    camera = window.viewport.camera
    before = (tuple(camera.target), camera.distance)
    block = ruler.blocks()[40]
    QTest.mouseDClick(
        ruler,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        ruler.cell_center(40),
    )
    assert (tuple(camera.target), camera.distance) != before
    assert ruler.selected_keys() == [block.key]


def test_a_language_switch_keeps_the_ruler_and_the_marks(window):
    from odock.gui import i18n

    ruler = window.sequence
    ruler.select_keys([ruler.blocks()[5].key])
    keys = ruler.selected_keys()
    cells = len(ruler.blocks())

    window.set_language("zh")
    assert window.selection_dock.windowTitle() == "选择"
    assert window.selection_table.horizontalHeaderItem(0).text() == "原子"
    assert window.selection_table.horizontalHeaderItem(8).text() == "电荷"
    assert window.btn_copy_pdbqt.text() == "复制为 PDBQT"
    assert window.sequence.tick_labels()[:2] == [("A", 16), ("A", 20)]
    assert len(window.sequence.blocks()) == cells
    assert window.sequence.selected_keys() == keys
    assert window.scene.selection

    window.set_language("en")
    assert window.selection_dock.windowTitle() == "Selection"
    assert window.btn_copy_pdbqt.text() == "Copy as PDBQT"
    assert len(window.sequence.blocks()) == cells
    assert window.sequence.selected_keys() == keys


def test_centring_on_a_residue_drops_a_binding_site_clip(window):
    """A pocket clip hides every atom outside its radius — including the one
    the user just double-clicked, so focusing a residue has to clear it."""
    window.viewport.frame_binding_site(clip=True)
    assert window.scene.receptor_cutoff is not None
    assert window.scene.front_clip is True
    assert len(window.scene.visible_receptor()) < len(window.scene.receptor)

    ruler = window.sequence
    camera = window.viewport.camera
    before = tuple(camera.target)
    QTest.mouseDClick(
        ruler,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        ruler.cell_center(len(ruler.blocks()) // 2),
    )
    assert window.scene.receptor_cutoff is None
    assert window.scene.receptor_radius == 0.0
    assert window.scene.front_clip is False
    # Every atom is drawable again, and the camera did move to the residue.
    assert len(window.scene.visible_receptor()) == len(window.scene.receptor)
    assert tuple(camera.target) != before


# ---------------------------------------------------------------------------
# task-9: an empty startup, resizable panels, click selection, a steady camera
# ---------------------------------------------------------------------------


def _project(viewport, point_or_atom) -> QtCore.QPoint:
    """The widget position of a world point, with the camera the picker uses."""
    try:
        world = (point_or_atom.x, point_or_atom.y, point_or_atom.z)
    except AttributeError:
        world = tuple(point_or_atom)[:3]
    width, height = viewport.width(), viewport.height()
    clip = (
        np.asarray(viewport.camera.projection(width, height))
        @ np.asarray(viewport.camera.view())
        @ np.asarray([*world, 1.0])
    )
    ndc = clip[:3] / clip[3]
    return QtCore.QPoint(int((ndc[0] + 1) / 2 * width), int((1 - ndc[1]) / 2 * height))


def _click_view(viewport, point, modifier=QtCore.Qt.KeyboardModifier.NoModifier):
    """A real left click inside the 3-D view, with an optional modifier."""
    QTest.mouseClick(viewport, QtCore.Qt.MouseButton.LeftButton, modifier, point)


def _drag_view(
    viewport,
    start,
    end,
    *,
    button=QtCore.Qt.MouseButton.LeftButton,
    modifier=QtCore.Qt.KeyboardModifier.NoModifier,
    steps: int = 6,
):
    """A real press → several moves → release, like a hand on the mouse."""
    QTest.mousePress(viewport, button, modifier, start)
    for step in range(1, steps + 1):
        fraction = step / steps
        QTest.mouseMove(
            viewport,
            QtCore.QPoint(
                int(start.x() + (end.x() - start.x()) * fraction),
                int(start.y() + (end.y() - start.y()) * fraction),
            ),
        )
    QTest.mouseRelease(viewport, button, modifier, end)


def _rendered_frame(window, width: int = 320, height: int = 240):
    """One frame straight from the renderer, for buffer-level comparisons."""
    data = window.viewport.renderer.render_image(
        window.viewport.camera, width, height
    )
    return np.frombuffer(data, dtype="u1").reshape(height, width, 3)


def _former_handle_points(box):
    """The six face centres the renderer used to put drag grips on."""
    centre, size = box
    points = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            position = list(centre)
            position[axis] += sign * size[axis] / 2.0
            points.append(tuple(position))
    return points


def test_a_fresh_window_draws_nothing(qapp):
    """No structures means an empty view: no stray cube and no handle atoms.

    The window used to push the placeholder grid spin values (0 / 2 Å) into the
    scene while the menus were built, which drew a small cube with six orange
    face-centre spheres on an otherwise blank view.
    """
    from odock.gui import i18n
    from odock.gui.app import DockingWorkbench, _configure_surface_format

    i18n.set_language("en")
    _configure_surface_format()
    window = DockingWorkbench()
    window.resize(900, 700)
    window.show()
    qapp.processEvents()
    try:
        renderer = window.viewport.renderer
        assert renderer is not None, f"no GL context: {window.viewport._gl_error}"
        assert window.scene.box is None
        assert window.scene.selection == []
        assert (getattr(window.scene, "ghost_ligand", None) or []) == []
        assert window.scene.show_axes is False
        assert window.scene.receptor == [] and window.scene.ligand == []
        assert renderer.receptor_count == 0 and renderer.ligand_count == 0
        assert renderer.mesh_receptor_vertices == 0
        assert renderer.mesh_ligand_vertices == 0
        assert not window.chk_box.isChecked() and not window.box_action.isChecked()
        # The placeholders are still in the spin boxes, they just stay there.
        assert window.spins["size_x"].value() == 0.0
        assert window.spins["center_x"].value() == 0.0

        window._new_project()
        assert window.scene.box is None
        window.set_language("zh")
        assert window.scene.box is None, "a rebuild must not re-seed the box"
        window.set_language("en")
        assert window.scene.box is None

        # A box appears exactly when it is asked for.
        window._toggle_box(True)
        assert window.scene.box is not None and window._box_visible
        window._toggle_box(False)
        assert window.scene.box is None
    finally:
        i18n.set_language("en")
        window.close()


def test_loading_a_ligand_defines_the_search_box(window):
    """The box arrives with the ligand, and the widgets agree that it is shown."""
    assert window.scene.box is not None
    assert window._box_visible
    assert window.chk_box.isChecked() and window.box_action.isChecked()


def test_the_pose_dock_sits_side_by_side_and_can_stack(window):
    splitter = window.pose_splitter
    assert splitter.orientation() == QtCore.Qt.Orientation.Horizontal
    assert splitter.count() == 2
    assert splitter.handleWidth() >= 6, "the divider has to be grabbable"
    assert not splitter.childrenCollapsible()
    left, right = splitter.widget(0), splitter.widget(1)
    assert left is not None and right is not None
    assert left.findChild(QtWidgets.QTableWidget) is window.table
    # The pose slider was removed on purpose: the table is how a pose is chosen,
    # and the summary label that used to share its row now spans it.
    assert left.findChild(QtWidgets.QSlider) is None
    assert window.lbl_pose.parent() is left
    assert right is window.log

    window.stack_action.setChecked(True)
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    assert splitter.orientation() == QtCore.Qt.Orientation.Vertical
    assert window._pose_stacked
    window._set_pose_split(False)
    assert splitter.orientation() == QtCore.Qt.Orientation.Horizontal
    assert not window.stack_action.isChecked()


def test_no_panel_is_locked_to_a_size(window):
    """Every dock and divider has to be the user's to resize."""
    for dock in (
        window.workspace_dock,
        window.inspector_dock,
        window.bottom_dock,
        window.dashboard_dock,
        window.comparison_dock,
        window.selection_dock,
    ):
        # Nothing hands out a minimum the user cannot live with, and nothing is
        # pinned to a fixed size policy.
        assert dock.minimumWidth() <= 500, dock.windowTitle()
        assert dock.minimumHeight() <= 200, dock.windowTitle()
        policy = dock.sizePolicy()
        assert policy.horizontalPolicy() != QtWidgets.QSizePolicy.Policy.Fixed
        assert policy.verticalPolicy() != QtWidgets.QSizePolicy.Policy.Fixed
    # The inspector used to demand 810 px because of the Grid tab's spin grid;
    # inside a scroll area it is narrow, and that keeps the whole window small.
    assert window.inspector_dock.minimumWidth() <= 300
    inspector_area = window.inspector_dock.widget()
    assert isinstance(inspector_area, QtWidgets.QScrollArea)
    assert inspector_area.widget() is window.inspector

    # The run monitor and the pose comparison are *content*, and content width
    # follows the font metrics: a fallback font made the measurement table
    # 638 px wide and pushed the whole window's minimum to 1134 px. Both panels
    # live in scroll areas instead, so they compress and scroll rather than
    # dictating the floor — which is exactly what the next assertions pin.
    dashboard_pages = [
        window.dashboard_tabs.widget(tab)
        for tab in range(window.dashboard_tabs.count())
    ]
    assert dashboard_pages and all(
        isinstance(page, QtWidgets.QScrollArea) for page in dashboard_pages
    )
    assert isinstance(window.comparison_dock.widget(), QtWidgets.QScrollArea)
    assert window.dashboard_dock.widget() is window.dashboard_tabs
    for dock in (window.dashboard_dock, window.comparison_dock):
        assert dock.minimumWidth() <= 200, dock.windowTitle()
        assert dock.minimumHeight() <= 200, dock.windowTitle()
    for panel in (window.run_dashboard, window.measure_history):
        dock_min = window.dashboard_dock.minimumWidth()
        content = panel.minimumSizeHint().width()
        assert dock_min < content, (
            f"{panel.objectName()} dictates the dock minimum: {dock_min} "
            f"against a content minimum of {content}"
        )
    assert window.minimumSizeHint().width() <= 1000
    # Both dividers are wide enough to grab and neither child can be collapsed.
    for splitter in (window.central_splitter, window.pose_splitter):
        assert splitter.handleWidth() >= 6
        assert not splitter.childrenCollapsible()
    # The ruler grows to fit its rows, and has no fixed height any more.
    assert window.sequence_area.minimumHeight() <= 220
    assert window.sequence_area.maximumHeight() > 500
    assert window.sequence_area.minimumHeight() >= window.sequence.track_height()


def test_a_click_in_the_view_selects_the_atom_under_the_cursor(window):
    viewport = window.viewport
    viewport.frame_all()
    QtWidgets.QApplication.instance().processEvents()
    point = _project(viewport, window.scene.receptor[40])
    hit = viewport.renderer.pick_atom(
        viewport.camera, viewport.width(), viewport.height(), point.x(), point.y()
    )
    assert hit is not None and hit[0] == "receptor"

    _click_view(viewport, point)
    QtWidgets.QApplication.instance().processEvents()

    atom = window.scene.receptor[hit[1]]
    assert window.sequence.selected_keys() == [(atom.chain, atom.res_id, atom.res_name)]
    assert window.scene.selection == window.sequence.atom_refs()
    assert window.scene.selection
    assert window.selection_table.rowCount() == len(window.scene.selection)
    assert window.lbl_selection_summary.text()


def test_shift_and_ctrl_click_add_and_empty_space_clears(window):
    viewport = window.viewport
    viewport.frame_all()
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    _click_view(viewport, _project(viewport, window.scene.receptor[40]))
    qapp.processEvents()
    first = window.sequence.selected_keys()
    assert len(first) == 1

    _click_view(viewport, _project(viewport, window.scene.ligand[0]), QtCore.Qt.KeyboardModifier.ShiftModifier)
    qapp.processEvents()
    added = window.sequence.selected_keys()
    assert len(added) == 2 and set(first) < set(added), added

    _click_view(viewport, _project(viewport, window.scene.ligand[0]), QtCore.Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert window.sequence.selected_keys() == first, "ctrl toggles that residue back off"

    _click_view(viewport, QtCore.QPoint(4, 4))
    qapp.processEvents()
    assert window.sequence.selected_keys() == []
    assert window.scene.selection == []
    assert window.selection_table.rowCount() == 0


def test_a_drag_selects_nothing(window):
    """Orbiting the camera must not pick an atom on the way."""
    viewport = window.viewport
    window.sequence.select_keys([window.sequence.blocks()[0].key])
    keys = window.sequence.selected_keys()
    azimuth = viewport.camera.azimuth
    start = QtCore.QPoint(viewport.width() // 2, viewport.height() // 2)
    QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, start)
    for step in range(1, 7):
        QTest.mouseMove(viewport, QtCore.QPoint(start.x() + step * 10, start.y() + step * 4))
    QTest.mouseRelease(
        viewport,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.QPoint(start.x() + 60, start.y() + 24),
    )
    QtWidgets.QApplication.instance().processEvents()
    assert window.sequence.selected_keys() == keys
    assert viewport.camera.azimuth != azimuth, "the drag still orbited"


def test_the_measure_and_bond_tools_keep_their_own_clicks(window):
    viewport = window.viewport
    viewport.frame_all()
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    window.sequence.select_keys([window.sequence.blocks()[1].key])
    keys = window.sequence.selected_keys()

    viewport.set_mode("measure")
    qapp.processEvents()
    point = _project(viewport, window.scene.receptor[10])
    expected = viewport.renderer.pick_atom(
        viewport.camera,
        viewport.width(),
        viewport.height(),
        point.x(),
        point.y(),
        ligand_first=False,
    )
    _click_view(viewport, point)
    qapp.processEvents()
    assert expected is not None, "the fixture puts an atom under that pixel"
    assert viewport._selection == [expected], "the measure tool collects the pick"
    assert window.sequence.selected_keys() == keys, "and leaves the selection alone"

    viewport.set_mode("bond")
    qapp.processEvents()
    _click_view(viewport, _project(viewport, window.scene.ligand[0]))
    qapp.processEvents()
    assert window.sequence.selected_keys() == keys, "the bond tool does not select either"

    viewport.set_mode("orbit")
    qapp.processEvents()


def test_picks_use_scene_indices_under_a_binding_site_clip(window):
    """With a cut-off active the visible list is filtered; indices must map back."""
    viewport = window.viewport
    viewport.frame_binding_site(clip=True)
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    visible = window.scene.visible_receptor()
    assert visible and len(visible) < len(window.scene.receptor)
    lookup = {id(atom): index for index, atom in enumerate(window.scene.receptor)}
    # Find a visible atom that the ray to its own centre actually resolves, so
    # the assertion is about the index mapping and not about occlusion.
    target = scene_index = point = None
    for candidate in visible:
        candidate_point = _project(viewport, candidate)
        hit = viewport.renderer.pick_atom(
            viewport.camera,
            viewport.width(),
            viewport.height(),
            candidate_point.x(),
            candidate_point.y(),
        )
        if hit == ("receptor", lookup[id(candidate)]):
            target, scene_index, point = candidate, lookup[id(candidate)], candidate_point
            break
    assert target is not None, "no unoccluded atom in the clipped receptor"

    _click_view(viewport, point)
    qapp.processEvents()
    assert any(index == scene_index for _group, index in window.scene.selection)
    assert window.sequence.selected_keys() == [(target.chain, target.res_id, target.res_name)]


def test_the_first_pose_load_frames_the_pocket_once(qapp):
    """Poses frame the site when they arrive, and on request — never per pose."""
    from odock.gui import i18n
    from odock.gui.app import DockingWorkbench, _configure_surface_format

    i18n.set_language("en")
    _configure_surface_format()
    window = DockingWorkbench(
        receptor=(DEMO / "receptor.pdbqt").read_text(encoding="utf-8"),
        ligand=(DEMO / "ligand.pdbqt").read_text(encoding="utf-8"),
    )
    window.resize(1100, 800)
    window.show()
    qapp.processEvents()
    try:
        viewport = window.viewport
        viewport.frame_all()
        qapp.processEvents()
        whole = viewport.camera.distance

        window.load_poses((DEMO / "poses.pdbqt").read_text(encoding="utf-8"))
        qapp.processEvents()
        assert viewport.camera.distance < whole, "the first pose load frames the pocket"
        # Framing is automatic; the cut-away is not. Only the explicit action
        # below clips, so nothing a pose does can change the protein's cut plane.
        assert window.scene.receptor_cutoff is None
        assert window.scene.front_clip is False

        # View ▸ Frame binding site does the same thing on demand ...
        viewport.frame_all()
        qapp.processEvents()
        assert window.scene.receptor_cutoff is None
        view = window.menuBar().actions()[-1].menu()
        action = next(
            item for item in view.actions() if item.text().startswith("Frame binding")
        )
        action.trigger()
        qapp.processEvents()
        assert window.scene.receptor_cutoff is not None
        assert window.scene.front_clip is True
        # The menu must not empty the view: QAction.triggered emits a bool, and
        # connected straight to ``frame_binding_site`` it used to arrive as
        # ``radius=False`` — a 0 Å clip ball that hid every receptor atom.
        visible = window.scene.visible_receptor()
        assert 0 < len(visible) < len(window.scene.receptor), (
            f"the menu action clipped the receptor away: {len(visible)} atoms"
        )
        assert window.scene.receptor_radius >= 8.0
        assert window.viewport.camera.distance < whole

        # The launch path (odock-gui -r -l -p) has to frame too, even though the
        # GL context — and its own camera framing — only appear on the first
        # paint, which happens after the constructor has finished.
        built = DockingWorkbench(
            receptor=(DEMO / "receptor.pdbqt").read_text(encoding="utf-8"),
            ligand=(DEMO / "ligand.pdbqt").read_text(encoding="utf-8"),
            poses=(DEMO / "poses.pdbqt").read_text(encoding="utf-8"),
        )
        built.show()
        qapp.processEvents()
        try:
            assert built.scene.receptor_cutoff is None
            assert built.viewport.camera.distance < whole
            camera = built.viewport.camera
            numbers = (camera.azimuth, camera.elevation, camera.distance, tuple(camera.target))
            built.set_pose(1)
            qapp.processEvents()
            assert (
                camera.azimuth,
                camera.elevation,
                camera.distance,
                tuple(camera.target),
            ) == numbers
        finally:
            built.close()
    finally:
        i18n.set_language("en")
        window.close()


def test_changing_the_pose_keeps_every_camera_number_and_the_clip(window):
    """Poses are compared from one viewpoint: the camera must not move at all."""
    from odock.gui import i18n

    viewport = window.viewport
    window.load_poses((DEMO / "poses.pdbqt").read_text(encoding="utf-8"))
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    camera = viewport.camera
    window.set_pose(0)
    qapp.processEvents()
    window.sequence.select_keys([window.sequence.blocks()[3].key])
    qapp.processEvents()

    before = (camera.azimuth, camera.elevation, camera.distance, tuple(camera.target))
    clip_before = (
        window.scene.receptor_cutoff,
        window.scene.receptor_radius,
        window.scene.front_clip,
    )
    keys_before = window.sequence.selected_keys()
    ligand_before = window.scene.ligand
    rows_before = window.selection_table.rowCount()
    frame_before = _rendered_frame(window)

    window.set_pose(2)
    qapp.processEvents()

    assert (
        camera.azimuth,
        camera.elevation,
        camera.distance,
        tuple(camera.target),
    ) == before, "a pose change moved the camera"
    assert window.scene.ligand is not ligand_before, "the ligand really changed"
    # ... and the *drawn* ligand changed too. Checking ``scene.ligand`` alone
    # cannot see a stale GL instance buffer, which is what a missing refresh
    # leaves behind: the viewport would keep showing the previous pose.
    assert not np.array_equal(frame_before, _rendered_frame(window)), (
        "the viewport still draws the previous pose"
    )
    assert (
        window.scene.receptor_cutoff,
        window.scene.receptor_radius,
        window.scene.front_clip,
    ) == clip_before, "a pose change touched the clip"
    assert window.sequence.selected_keys() == keys_before
    assert window.selection_table.rowCount() == rows_before
    assert "mode 3" in window.lbl_pose.text()
    i18n.set_language("en")


def test_the_viewport_drags_never_move_the_box(window):
    """Orbit, pan and picking keep working; no drag touches the search box.

    The box is read-only in the 3-D view — the Grid tab's centre / size / spacing
    fields (and the menu actions: Fit to ligand, Centre on ligand, Align to
    residue, Detect pockets, Whole protein) are the only things that set it — so
    a drag from anywhere, including a former face handle, leaves it alone.
    """
    viewport = window.viewport
    qapp = QtWidgets.QApplication.instance()
    viewport.frame_all()
    qapp.processEvents()
    camera = viewport.camera

    # A press point that is not near any face centre, so step 1 really is a plain
    # orbit drag. The face centres come from the box itself, so this does not
    # depend on the renderer still exposing drag handles.
    grips = [
        _project(viewport, point) for point in _former_handle_points(window.scene.box)
    ]
    free = None
    for fraction_x, fraction_y in (
        (0.2, 0.25),
        (0.8, 0.25),
        (0.2, 0.8),
        (0.8, 0.8),
        (0.5, 0.12),
    ):
        point = QtCore.QPoint(
            int(viewport.width() * fraction_x), int(viewport.height() * fraction_y)
        )
        if all((point - grip).manhattanLength() > 40 for grip in grips):
            free = point
            break
    assert free is not None, "no point free of the box's face centres"

    # 1. a left drag from a handle-free point orbits — and selects nothing.
    azimuth = camera.azimuth
    _drag_view(viewport, free, QtCore.QPoint(free.x() + 120, free.y() + 40))
    qapp.processEvents()
    assert camera.azimuth != azimuth, "a left drag must orbit the camera"
    assert window.sequence.selected_keys() == [], "a drag must not select an atom"

    # 2. a right drag pans.
    target = tuple(camera.target)
    _drag_view(
        viewport,
        free,
        QtCore.QPoint(free.x() + 60, free.y() + 30),
        button=QtCore.Qt.MouseButton.RightButton,
    )
    qapp.processEvents()
    assert tuple(camera.target) != target, "a right drag must pan the camera"

    # 3. shift+left drag orbits like a plain drag and leaves the box alone.
    box_before = (tuple(window.scene.box[0]), tuple(window.scene.box[1]))
    spins_before = (
        window.spins["center_x"].value(),
        window.spins["size_x"].value(),
    )
    azimuth = camera.azimuth
    _drag_view(
        viewport,
        free,
        QtCore.QPoint(free.x() + 70, free.y() + 20),
        modifier=QtCore.Qt.KeyboardModifier.ShiftModifier,
    )
    qapp.processEvents()
    assert camera.azimuth != azimuth, "a shift drag must orbit now"
    assert (tuple(window.scene.box[0]), tuple(window.scene.box[1])) == box_before

    # 4. a drag starting on a former face handle changes nothing either: the box
    #    is read-only in the view (the Grid fields and the menu actions set it).
    centre, size = window.scene.box
    former_handle = (centre[0] + size[0] / 2.0, centre[1], centre[2])
    handle_point = _project(viewport, former_handle)
    _drag_view(
        viewport, handle_point, QtCore.QPoint(handle_point.x() + 60, handle_point.y())
    )
    qapp.processEvents()
    assert (tuple(window.scene.box[0]), tuple(window.scene.box[1])) == box_before
    assert (
        window.spins["center_x"].value(),
        window.spins["size_x"].value(),
    ) == spins_before, "a drag must not write the box back into the Grid fields"

    # 5. the Grid tab's fields are the documented way to change it. The spin
    #    boxes carry three decimals, so allow for their rounding.
    window.spins["center_x"].setValue(spins_before[0] + 2.5)
    qapp.processEvents()
    assert window.scene.box[0][0] == pytest.approx(
        box_before[0][0] + 2.5, abs=2e-3
    )
    window.spins["size_x"].setValue(box_before[1][0] + 3.0)
    qapp.processEvents()
    assert window.scene.box[1][0] == pytest.approx(box_before[1][0] + 3.0, abs=2e-3)


def test_the_interaction_focus_is_sticky_across_poses(window):
    """Show interactions is a one-time click; browsing keeps the emphasis.

    The user ruled: "换pose时应该自动更新". So the request is remembered and the
    emphasis is re-derived from the *current* pose's contacts — never stale
    indices from the pose that was replaced — while the camera and the clip stay
    untouched, which is what makes two poses comparable.
    """
    from odock.gui import i18n

    pytest.importorskip("odock.analysis")

    qapp = QtWidgets.QApplication.instance()
    window.load_poses((DEMO / "poses.pdbqt").read_text(encoding="utf-8"))
    qapp.processEvents()
    assert window._focus_requested is False
    assert window.scene.interaction_focus == [], "nothing is emphasised yet"

    window._annotate_interactions()
    qapp.processEvents()
    assert window.interactions, "the pose must have contacts to annotate"
    assert window._focus_requested is True
    assert list(window.scene.interaction_focus) == list(window.interactions)

    # Browsing every pose keeps the emphasis, re-derived each time.
    camera = window.viewport.camera
    before = (camera.azimuth, camera.elevation, camera.distance, tuple(camera.target))
    clip_before = (
        window.scene.receptor_cutoff,
        window.scene.receptor_radius,
        window.scene.front_clip,
    )
    for index in range(len(window.pose_models)):
        window.set_pose(index)
        qapp.processEvents()
        assert window.scene.interactions, "the dashes follow the pose"
        assert list(window.scene.interaction_focus) == list(window.interactions), (
            f"pose {index + 1} lost the emphasis"
        )
    assert (
        camera.azimuth,
        camera.elevation,
        camera.distance,
        tuple(camera.target),
    ) == before, "the emphasis must not move the camera"
    assert (
        window.scene.receptor_cutoff,
        window.scene.receptor_radius,
        window.scene.front_clip,
    ) == clip_before, "the emphasis must not touch the clip"

    # A language switch rebuilds the window: the request has to survive it, and
    # the emphasis has to be back on the reloaded pose.
    i18n.set_language("zh")
    qapp.processEvents()
    try:
        assert window._focus_requested is True
        assert list(window.scene.interaction_focus) == list(window.interactions)
    finally:
        i18n.set_language("en")
        qapp.processEvents()

    # Clear annotations forgets the request, so browsing cannot bring it back.
    window._clear_interactions()
    qapp.processEvents()
    assert window._focus_requested is False
    assert window.scene.interaction_focus == []
    assert window.scene.interactions == []
    if window.pose_models:
        window.set_pose(1)
        qapp.processEvents()
        assert window.scene.interaction_focus == [], "cleared means cleared"

    # And a new project starts clean.
    window._new_project()
    qapp.processEvents()
    assert window._focus_requested is False
    assert window.scene.interaction_focus == []


def test_loading_a_ligand_keeps_the_receptor_in_view(qapp):
    """Importing a ligand must not leave the camera inside the fitted box.

    ``load_ligand`` used to call ``frame_ligand``, a ~15 Å close-up: with a
    receptor already loaded that put the camera inside the box fitted around the
    ligand, where the translucent fill buried the whole structure.
    """
    from odock.gui import i18n
    from odock.gui.app import DockingWorkbench, _configure_surface_format

    i18n.set_language("en")
    _configure_surface_format()
    window = DockingWorkbench()
    window.resize(1000, 760)
    window.show()
    qapp.processEvents()
    try:
        window.load_receptor((DEMO / "receptor.pdbqt").read_text(encoding="utf-8"))
        qapp.processEvents()
        window.load_ligand((DEMO / "ligand.pdbqt").read_text(encoding="utf-8"))
        qapp.processEvents()

        assert len(window.scene.receptor) > 100, "the receptor is still loaded"
        _centre, size = window.scene.box
        assert size[0] > 10.0, "the box was still fitted to the ligand"
        # Outside the box: the camera distance has to clear its half-diagonal, or
        # the fill sits in front of the structure.
        half_diagonal = (sum(value * value for value in size) ** 0.5) / 2.0
        assert window.viewport.camera.distance > half_diagonal, (
            f"camera {window.viewport.camera.distance:.2f} A is inside the "
            f"{size[0]:.1f}x{size[1]:.1f}x{size[2]:.1f} A box"
        )
        assert window.viewport.renderer.receptor_count > 100
        frame = _rendered_frame(window)
        painted = int((np.abs(frame - frame[0, 0]).sum(axis=2) > 12).sum())
        assert painted > 0, "the viewport draws nothing at all"
    finally:
        window.close()


def test_the_box_opacity_action_wires_the_dialog_value(window, monkeypatch):
    """Grid ▸ Box opacity… writes ``Scene.box_alpha`` (the renderer's fill)."""
    from odock.gui import dialogs

    if getattr(dialogs, "BoxOpacityDialog", None) is None:
        pytest.skip("the box-opacity dialog lands with the renderer side of task-10")

    class FakeDialog:
        def __init__(self, alpha, parent=None):
            self.opened_with = alpha
            self.parent = parent

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def value(self):
            return 0.75

    monkeypatch.setattr(dialogs, "BoxOpacityDialog", FakeDialog)
    qapp = QtWidgets.QApplication.instance()
    window.scene.box_alpha = 0.35
    window._choose_box_opacity()
    qapp.processEvents()
    assert window.scene.box_alpha == pytest.approx(0.75)
    assert "box opacity" in window.log.toPlainText()

    # Cancel keeps the value it had.
    class Cancelled(FakeDialog):
        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(dialogs, "BoxOpacityDialog", Cancelled)
    window._choose_box_opacity()
    assert window.scene.box_alpha == pytest.approx(0.75)


def test_the_play_poses_menu_item_actually_plays(window):
    """A non-checkable action must not hand ``checked=False`` to the player.

    ``QAction.triggered`` emits a bool and PyQt passes it to any slot that can
    take one, so ``Analysis ▸ Play poses`` connected straight to
    ``_toggle_playback(checked=True)`` received ``False`` and stopped the player
    instead of starting it — a dead menu item.
    """
    from odock.gui import i18n

    qapp = QtWidgets.QApplication.instance()
    window.load_poses((DEMO / "poses.pdbqt").read_text(encoding="utf-8"))
    qapp.processEvents()
    analysis_name = i18n.tr("menu.analysis").replace("&", "")
    analysis = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.menu() is not None
        and action.text().replace("&", "") == analysis_name
    )
    play_name = i18n.tr("action.play_poses").replace("&", "")
    play = next(
        action
        for action in analysis.actions()
        if action.text().replace("&", "") == play_name
    )

    window.viewport.stop_animation()
    assert window.viewport._anim_timer is None
    play.trigger()
    qapp.processEvents()
    assert window.viewport._anim_timer is not None, "the menu item must start playing"
    play.trigger()
    qapp.processEvents()
    assert window.viewport._anim_timer is not None, "triggering it again keeps playing"
    window.viewport.stop_animation()
    assert window.viewport._anim_timer is None, "and the player can still be stopped"


# ---------------------------------------------------------------------------
# the ruler palette and the pose-contact marks
# ---------------------------------------------------------------------------


def test_the_ruler_takes_the_palette_of_the_active_theme(window):
    """The ruler paints its own background, so the theme has to reach it.

    It is a plain ``QWidget`` with a ``QPainter``: a style sheet cannot recolour
    the strip it fills, and a near-black ruler under a light window reads as a
    rendering bug.
    """
    from odock.gui import dashboard
    from odock.gui.sequence import DARK_RULER, LIGHT_RULER

    assert window.color_theme is dashboard.DARK
    assert window.sequence.palette is DARK_RULER

    window.set_theme("light")
    assert window.sequence.palette is LIGHT_RULER
    assert LIGHT_RULER.background != DARK_RULER.background
    # Both palettes are complete: nothing paints with a missing colour.
    for palette in (DARK_RULER, LIGHT_RULER):
        for field in RulerPalette.__dataclass_fields__:
            value = getattr(palette, field)
            assert isinstance(value, str) and value.startswith("#"), (field, value)

    window.set_theme("dark")
    assert window.sequence.palette is DARK_RULER

    # The painter runs on both palettes without raising.
    for palette in (DARK_RULER, LIGHT_RULER):
        window.sequence.set_palette(palette)
        pixmap = QtGui.QPixmap(window.sequence.size())
        window.sequence.render(pixmap)
    window.sequence.set_palette(DARK_RULER)
    assert window.sequence.set_palette(None) is None
    assert window.sequence.palette is DARK_RULER


def test_contact_marks_name_the_residues_a_pose_touches(window):
    """The ruler marks belong to the *opt-in* interaction drawing.

    Nothing is marked while the drawing is off (the pose contacts are reported in
    the label and the table, not drawn), and Analysis ▸ Show interactions turns
    the marks on; Clear annotations takes them away again.
    """
    ruler = window.sequence
    assert not ruler.marked_keys("contact"), "off by default: nothing is drawn"
    window._annotate_interactions(quiet=True)
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    marked = ruler.marked_keys("contact")
    assert marked, "the imported ligand touches residues"
    # Every mark is a residue the ruler actually draws, and BEN (the ligand)
    # is not one of them: the marks belong to the receptor sequence.
    keys = set(ruler.block_keys())
    assert set(marked) <= keys
    assert all(key[2] not in ("BEN", "HOH") for key in marked)
    assert 1 <= len(marked) < 60, "a ligand contacts a pocket, not the whole protein"

    # The mark is a function of the coordinates alone, so recomputing it gives
    # exactly the same residues.
    ruler.clear_contact_marks()
    assert ruler.contact_marks() == {}
    window._mark_pose_contacts()
    assert ruler.marked_keys("contact") == marked

    # Browsing poses recomputes them for the pose on screen.
    window.load_poses((DEMO / "poses.pdbqt").read_text(encoding="utf-8"))
    qapp.processEvents()
    for index in range(len(window.pose_models)):
        window.set_pose(index)
        qapp.processEvents()
        current = ruler.marked_keys("contact")
        assert current, f"pose {index + 1} touches nothing?"
        assert set(current) <= keys

    # Clearing the annotations clears the marks as well.
    window._clear_interactions()
    qapp.processEvents()
    assert ruler.contact_marks() == {}


def test_the_ruler_marks_survive_a_reload_of_the_same_structure():
    """Marks are residue keys, so a reload puts them back on the same residues.

    Switching pose reloads the ligand through ``set_structure``; if the marks
    were cell indices they would slide onto whatever residue now sits at that
    index.
    """
    ruler = track(n=12)
    wanted = [("A", 3, "ALA"), ("A", 9, "ALA")]
    ruler.set_contact_marks({"contact": wanted})
    assert ruler.marked_keys("contact") == wanted

    ruler.set_structure(make_atoms([("ALA", 1)] * 12))
    assert ruler.marked_keys("contact") == wanted

    # A residue that no longer exists simply disappears from the marks.
    ruler.set_structure(make_atoms([("ALA", 1)] * 4))
    assert ruler.marked_keys("contact") == [("A", 3, "ALA")]

    ruler.set_structure(make_atoms([("ALA", 1)] * 4, start=10))
    assert ruler.marked_keys("contact") == []


def test_contact_marks_accept_only_known_layers_and_survive_painting():
    from odock.gui.sequence import CONTACT_MARKS, MARK_ORDER

    ruler = track(n=12)
    ruler.set_contact_marks(
        {
            "contact": [("A", 2, "ALA")],
            "shared": [("A", 2, "ALA"), ("A", 5, "ALA")],
            "unique": [("A", 5, "ALA")],
            "nonsense": [("A", 1, "ALA")],
        }
    )
    assert set(ruler.contact_marks()) == {"contact", "shared", "unique"}
    assert ruler.marked_keys("nonsense") == []
    assert ruler.marked_keys("shared") == [("A", 2, "ALA"), ("A", 5, "ALA")]

    # Two layers on one cell stack in a fixed order, so both stay visible.
    stacked_block = next(b for b in ruler._blocks if b.key == ("A", 2, "ALA"))
    stacked = [name for name in MARK_ORDER if name in ruler._marks_of(stacked_block)]
    assert stacked == ["contact", "shared"]
    assert set(CONTACT_MARKS) == set(MARK_ORDER)

    pixmap = QtGui.QPixmap(ruler.size())
    ruler.render(pixmap)
    # The bands really are painted: the marked cell differs from an unmarked one.
    image = pixmap.toImage()
    marked = ruler.cell_rect(stacked_block.index)
    plain = ruler.cell_rect(next(b for b in ruler._blocks if b.key == ("A", 1, "ALA")).index)
    marked_px = image.pixelColor(marked.center().x(), marked.bottom() - 2)
    plain_px = image.pixelColor(plain.center().x(), plain.bottom() - 2)
    assert marked_px != plain_px

    ruler.clear_contact_marks()
    assert ruler.contact_marks() == {}
    ruler.clear_contact_marks()  # idempotent
    for name in MARK_ORDER:
        assert ruler.marked_keys(name) == []


def test_a_comparison_marks_the_shared_and_the_unique_residues(window):
    """Ctrl-clicking two poses splits the ruler into agreed / disagreed."""
    from odock.gui import dashboard

    window.load_poses((DEMO / "poses.pdbqt").read_text(encoding="utf-8"))
    qapp = QtWidgets.QApplication.instance()
    qapp.processEvents()
    window.table.clearSelection()
    window.table.selectRow(0)
    window.table.item(1, 0).setSelected(True)
    qapp.processEvents()

    comparison = window.comparison.comparison()
    assert comparison is not None, "two selected rows must produce a comparison"
    ruler = window.sequence
    shared = ruler.marked_keys("shared")
    unique = ruler.marked_keys("unique")
    assert "contact" not in ruler.contact_marks(), "the split replaces the plain mark"
    assert set(shared) == {key for key, _a, _b in comparison.diff.shared}
    assert set(unique) == {
        contact.key
        for contact in list(comparison.diff.only_a) + list(comparison.diff.only_b)
    }
    assert not (set(shared) & set(unique)), "a residue cannot be both"

    # Comparing the same pose with itself agrees about everything it touches.
    same = dashboard.compare_poses(
        window.pose_models[0].atoms,
        window.pose_models[0].atoms,
        window.scene.receptor,
    )
    window._mark_comparison_contacts(same)
    assert ruler.marked_keys("unique") == []
    assert ruler.marked_keys("shared") == sorted(
        ruler.marked_keys("shared"), key=lambda key: ruler.block_keys().index(key)
    )

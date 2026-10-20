# SPDX-License-Identifier: GPL-3.0-or-later
"""Search boxes are a collection: identity, per-box visibility, and deletion.

The single flag this replaces shared visibility with existence, so hiding the box
could leave no way back to it and nothing distinguished "hidden" from "gone". Every
test here is one of the user's stated fears, or the operation that has to stay
distinct so the fear cannot come back:

* three boxes, delete the middle one — the other two keep their own names in the
  menu;
* hide a box — it keeps its menu entry and can be turned back on (**the exact
  scenario the user described**);
* delete a box — the entry is gone, and a delete cannot hit the wrong box;
* two overlapping boxes are both drawn (pixel counts, not a claim);
* resizing one box leaves the others exactly as they were;
* a snapshot round-trips three boxes *and* their visibility;
* undo/redo covers add, delete, move, resize, show and hide;
* the active box may be hidden — the case where "hidden" would most easily be
  read as "deleted".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odock.gui import dashboard, i18n
from odock.gui.app import DockingWorkbench

DEMO = Path(__file__).resolve().parent.parent / "demo" / "systems" / "3ptb"


@pytest.fixture
def qapp():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _pump(qapp, turns: int = 40) -> None:
    for _ in range(turns):
        qapp.processEvents()


@pytest.fixture
def window(qapp):
    """A workbench with a receptor and a ligand, so a first box exists."""
    if not (DEMO / "receptor.pdbqt").exists():
        pytest.skip("the bundled 3PTB demo files are not present")
    win = DockingWorkbench(
        receptor=(DEMO / "receptor.pdbqt").read_text(encoding="utf-8"),
        ligand=(DEMO / "ligand.pdbqt").read_text(encoding="utf-8"),
    )
    win.resize(1200, 800)
    win.show()
    _pump(qapp, 60)
    yield win
    win.close()


def _menu_entries(window):
    """``[(name, checked)]`` for the checkable Views-menu entries, in order."""
    return [
        (action.text(), action.isChecked())
        for action in window.boxes_menu.actions()
        if action.isCheckable() and action.text()
    ]


def _three_boxes(window):
    """Ensure exactly three boxes exist, and return them."""
    while len(window.boxes) > 3:
        window.boxes.remove(window.boxes[-1].id)
    while len(window.boxes) < 3:
        window._add_box()
    window._sync_boxes_view()
    return list(window.boxes)


# ---------------------------------------------------------------------------
# The user's exact scenario, and its three neighbours
# ---------------------------------------------------------------------------


def test_three_boxes_deleted_from_the_middle_keep_their_own_names(window, qapp):
    boxes = _three_boxes(window)
    names = [box.name for box in boxes]
    first, middle, last = boxes

    window._activate_box(middle.id)
    window._delete_active_box()
    _pump(qapp, 10)

    remaining = [box.name for box in window.boxes]
    assert remaining == [names[0], names[2]]
    # The menu lists the survivors **by their own names** — not "box 1"/"box 2"
    # renumbered, which would make a user think a different box appeared.
    assert [name for name, _checked in _menu_entries(window)] == remaining
    assert first.name in remaining and last.name in remaining
    assert middle.name not in remaining


def test_hiding_a_box_keeps_its_menu_entry_and_checking_it_brings_it_back(window):
    """The user's words: hiding must never mean 'you can never open it again'."""
    boxes = _three_boxes(window)
    target = boxes[1]
    before = len(window.boxes)

    window._set_box_visibility(target.id, False)
    _qapp = window.viewport.window() and None
    # Still in the collection, still in the menu, unchecked — not gone.
    assert len(window.boxes) == before
    entries = dict(_menu_entries(window))
    assert target.name in entries, entries
    assert entries[target.name] is False

    # Checking its entry brings it back, without touching any other box.
    action = next(
        item for item in window.boxes_menu.actions() if item.text() == target.name
    )
    action.setChecked(True)
    assert window.boxes.get(target.id).visible is True
    assert len(window.boxes) == before
    assert dict(_menu_entries(window))[target.name] is True
    # And the others were never disturbed.
    for box in boxes:
        if box.id != target.id:
            assert window.boxes.get(box.id).visible is True


def test_delete_removes_the_entry_and_cannot_hit_the_wrong_box(window):
    boxes = _three_boxes(window)
    first, middle, last = boxes
    window._activate_box(middle.id)
    window._delete_active_box()

    # The deleted box's entry is gone, and only it.
    menu_names = [name for name, _checked in _menu_entries(window)]
    assert middle.name not in menu_names
    assert set(menu_names) == {first.name, last.name}

    # Deleting a *different* box removes that one, not the previously active one
    # and not the first one by accident.
    window._activate_box(last.id)
    window._delete_active_box()
    assert [box.name for box in window.boxes] == [first.name]
    assert first.id in [box.id for box in window.boxes]


def test_the_active_box_may_be_hidden_and_the_dock_keeps_editing_it(window):
    """Hidden ≠ gone: the active box can be hidden, and it stays the edited one."""
    boxes = _three_boxes(window)
    active = boxes[2]
    window._activate_box(active.id)

    window._set_box_visibility(active.id, False)
    # No solid volume for a hidden active box...
    assert window.scene.box is None
    # ...while the *other* visible boxes are still drawn as wireframes.
    assert [
        entry["name"]
        for entry in window.viewport.extra_boxes
        if entry.get("wireframe", True)
    ] == [boxes[0].name, boxes[1].name]
    # The dock still says which box it edits, and says it is hidden.
    assert active.name in window.box_active_label.text()
    assert i18n.tr("boxes.hidden") in window.box_active_label.text()
    # And editing it works: the active box is the one the fields write to.
    window.spins["size_x"].setValue(31.0)
    assert window.boxes.get(active.id).width == pytest.approx(31.0)
    # The others are untouched by that edit.
    assert window.boxes.get(boxes[0].id).width != pytest.approx(31.0)


def test_a_hidden_row_says_how_to_get_the_box_back(window):
    """The row is where someone concludes a hidden box is gone, so it says how."""
    boxes = _three_boxes(window)
    target = boxes[0]
    window._set_box_visibility(target.id, False)
    tips = [
        (window.box_list.item(i).text(), window.box_list.item(i).toolTip())
        for i in range(window.box_list.count())
    ]
    tooltip = next(tip for text, tip in tips if text.startswith(target.name))
    assert tooltip, "a hidden row must carry the route back"
    assert i18n.tr("boxes.hidden") in tooltip
    assert i18n.tr("menu.boxes") in tooltip
    assert target.name in tooltip


# ---------------------------------------------------------------------------
# Drawing: two boxes, both readable
# ---------------------------------------------------------------------------


def test_two_overlapping_boxes_are_both_drawn(window, qapp):
    """Pixel evidence, not a claim: hiding the second box changes the picture."""
    QtGui = pytest.importorskip("PyQt6.QtGui")
    boxes = _three_boxes(window)
    # Two boxes with the same centre and a big overlap: the case where a solid
    # volume would swallow the other's outline.
    for box in boxes:
        box.center = (0.0, 0.0, 0.0)
        box.size = (20.0, 20.0, 20.0)
        box.visible = True
    window.boxes.set_active(boxes[0].id)
    window._sync_boxes_view()
    _pump(qapp, 30)

    assert window.scene.box is not None, "the active box is the solid volume"
    wireframes = [
        entry for entry in window.viewport.extra_boxes if entry.get("wireframe", True)
    ]
    assert len(wireframes) == 2, "the other two are wireframes"
    # The active box contributes a label-only entry: in a view with three boxes
    # the one the dock is editing must be named, not merely filled.
    labels = [entry for entry in window.viewport.extra_boxes if not entry.get("wireframe", True)]
    assert len(labels) == 1 and window.boxes.active_name() in labels[0]["name"]

    if not window.viewport._ensure_context():
        pytest.skip("no GL context on this machine")

    import tempfile

    def frame(name):
        path = Path(tempfile.mkdtemp()) / name
        assert window.viewport.snapshot(path, 700, 520) is True
        image = QtGui.QImage(str(path)).convertToFormat(
            QtGui.QImage.Format.Format_RGB888
        )
        return image

    with_wireframes = frame("both.png")
    # Hide the other two: the picture must change, which is what proves they were
    # painted rather than merely listed.
    saved = [(box.id, box.visible) for box in window.boxes]
    for box in window.boxes:
        if box.id != boxes[0].id:
            window.boxes.show(box.id, False)
    window._sync_boxes_view()
    _pump(qapp, 30)
    without = frame("active-only.png")
    for box_id, visible in saved:
        window.boxes.show(box_id, visible)
    window._sync_boxes_view()
    _pump(qapp, 30)

    differing = 0
    for y in range(0, with_wireframes.height(), 3):
        for x in range(0, with_wireframes.width(), 3):
            if with_wireframes.pixel(x, y) != without.pixel(x, y):
                differing += 1
    assert differing > 40, (
        "two overlapping boxes must both be drawn: "
        f"{differing} differing samples between the frame with and without them"
    )


# ---------------------------------------------------------------------------
# Independent dimensions and independent resizing
# ---------------------------------------------------------------------------


def test_each_box_has_its_own_width_height_and_depth(window):
    boxes = _three_boxes(window)
    first, second, third = boxes
    untouched = tuple(third.size)
    # Independently settable dimensions: a slab, a column and a cube.
    window._activate_box(first.id)
    window.spins["size_x"].setValue(30.0)
    window.spins["size_y"].setValue(10.0)
    window.spins["size_z"].setValue(14.0)
    window._activate_box(second.id)
    window.spins["size_x"].setValue(12.0)
    window.spins["size_y"].setValue(26.0)
    window.spins["size_z"].setValue(9.0)

    assert window.boxes.get(first.id).size == pytest.approx((30.0, 10.0, 14.0))
    assert window.boxes.get(second.id).size == pytest.approx((12.0, 26.0, 9.0))
    # The box nobody edited is exactly as it was: `_add_box` copies the active
    # box's dimensions, so the baseline is that box's own size, captured
    # before the edits — not the theoretical default.
    assert window.boxes.get(third.id).size == pytest.approx(untouched)
    # And the model exposes the three named dimensions.
    box = window.boxes.get(first.id)
    assert (box.width, box.height, box.depth) == pytest.approx((30.0, 10.0, 14.0))


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


def test_snapshot_and_restore_round_trip_three_boxes_and_visibility(window, qapp):
    boxes = _three_boxes(window)
    window._activate_box(boxes[1].id)
    window._set_box_visibility(boxes[0].id, False)
    window.boxes.resize(boxes[2].id, (11.0, 12.0, 13.0))
    window._sync_boxes_view()
    expected = window.boxes.fingerprint()

    state = window._capture_ui_state()
    _pump(qapp, 20)

    # A fresh window restored from the snapshot must have the same collection.
    restored = DockingWorkbench()
    restored.resize(1200, 800)
    restored.show()
    _pump(qapp, 40)
    restored._restore_ui_state(state)
    _pump(qapp, 60)
    try:
        assert restored.boxes.fingerprint() == expected
        assert len(restored.boxes) == 3
        assert restored.boxes.active_name() == boxes[1].name
        assert [box.visible for box in restored.boxes] == [
            box.visible for box in window.boxes
        ]
        # The renderer channel agrees: one solid volume (the visible active box).
        assert restored.scene.box is not None
        assert sorted(
            entry["name"]
            for entry in restored.viewport.extra_boxes
            if entry.get("wireframe", True)
        ) == sorted(
            box.name for box in restored.boxes if box.visible and box.id != boxes[1].id
        )
    finally:
        restored.close()


def test_the_session_payload_carries_the_whole_collection(window):
    boxes = _three_boxes(window)
    window._set_box_visibility(boxes[2].id, False)
    window._sync_boxes_view()
    payload = window._session_payload()
    stored = payload.get("boxes") or {}
    assert len(stored.get("boxes", [])) == 3
    assert stored["active_box"] == window.boxes.active_id
    assert [item["visible"] for item in stored["boxes"]] == [
        box.visible for box in window.boxes
    ]
    # It is JSON-serialisable, which is what the session file needs.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Undo covers every operation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    ["add", "delete", "move", "resize", "show", "hide"],
)
def test_undo_and_redo_cover_every_box_operation(window, qapp, action):
    boxes = _three_boxes(window)
    window._activate_box(boxes[0].id)
    window._history.clear()
    target = boxes[0]
    if action == "show":
        window._set_box_visibility(target.id, False)
        window._history.clear()
    before = window.boxes.fingerprint()

    if action == "add":
        window._add_box()
    elif action == "delete":
        window._delete_active_box()
    elif action == "move":
        window.spins["center_x"].setValue(float(target.center[0]) + 7.0)
    elif action == "resize":
        window.spins["size_y"].setValue(float(target.height) + 6.0)
    elif action == "show":
        window._set_box_visibility(target.id, True)
    elif action == "hide":
        window._set_box_visibility(target.id, False)
    _pump(qapp, 10)

    after = window.boxes.fingerprint()
    assert after != before, f"{action} changed nothing"

    assert window.undo() is True
    _pump(qapp, 10)
    assert window.boxes.fingerprint() == before, f"undo did not restore {action}"

    assert window.redo() is True
    _pump(qapp, 10)
    assert window.boxes.fingerprint() == after, f"redo did not reapply {action}"

# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the 3-D workbench widget.

These run head-lessly: the viewport owns its GL context and presents the result
with ``QPainter``, so the Qt platform plugin does not matter. That property is
itself the fix for the bug these tests guard against — a `QOpenGLWidget` whose
GL area stayed blank because Qt's compositor left ``glColorMask(1, 0, 0, 0)``
behind and never composited the framebuffer it had bound.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

# The whole point: no display is required.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
QtGui = pytest.importorskip("PyQt6.QtGui")
QtCore = pytest.importorskip("PyQt6.QtCore")
pytest.importorskip("moderngl")
QTest = pytest.importorskip("PyQt6.QtTest").QTest

from odock.gui import i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui.structure import parse_pdbqt  # noqa: E402
from odock.gui.viewport import (  # noqa: E402
    LIGAND_STYLES,
    PROTEIN_STYLES,
    Camera,
    Renderer,
    Scene,
)

BACKGROUND = np.array([22.0, 24.0, 32.0])

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

RECEPTOR_PDBQT = "\n".join(
    [
        f"ATOM  {i + 1:5d}  CB  ALA A   1    "
        f"{3.0 * np.cos(i):8.3f}{3.0 * np.sin(i):8.3f}{1.2 * i - 3.0:8.3f}"
        f"  1.00  0.00     0.000 C"
        for i in range(24)
    ]
    + ["TER", ""]
)


@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the module."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _english():
    """The host locale is Chinese, so pin the language the other tests assume."""
    i18n.set_language("en")
    yield
    i18n.set_language("en")


def _painted(pixmap) -> tuple[int, int]:
    image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    raw = np.frombuffer(bytes(ptr), dtype="u1").reshape(h, image.bytesPerLine())
    arr = raw[:, : w * 3].reshape(h, w, 3).astype(int)
    differing = int((np.abs(arr - BACKGROUND).sum(axis=2) > 12).sum())
    return differing, w * h


def test_offscreen_renderer_draws_into_its_own_framebuffer():
    """The GL layer alone (no Qt) must produce pixels."""
    import moderngl

    atoms = parse_pdbqt(LIGAND_PDBQT)[0].atoms
    scene = Scene(ligand=atoms)
    scene.box = ((1.5, 1.2, 0.0), (8.0, 8.0, 8.0))
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no OpenGL 3.3 context: {exc}")

    width = height = 320
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((width, height), 3)],
        depth_attachment=ctx.depth_texture((width, height)),
    )
    fbo.use()
    renderer = Renderer(ctx, scene)
    camera = Camera()
    camera.frame(*scene.bounds())
    renderer.draw(camera, width, height)
    data = fbo.read(components=3)
    arr = np.frombuffer(data, dtype="u1").reshape(height, width, 3).astype(int)
    differing = int((np.abs(arr - BACKGROUND).sum(axis=2) > 12).sum())
    assert renderer.ligand_count == 5
    assert differing > 0.02 * width * height, f"only {differing} pixels drawn"


def test_viewport_widget_paints_the_scene(qapp):
    """The widget must present what the renderer produced."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(700, 520)
    window.show()
    qapp.processEvents()

    viewport = window.viewport
    assert viewport.renderer is not None, f"no GL context: {viewport._gl_error}"
    assert viewport.renderer.receptor_count > 0
    assert viewport.renderer.ligand_count == 5

    differing, total = _painted(viewport.grab())
    window.close()
    assert differing > 0.01 * total, f"the 3-D area is blank ({differing}/{total} px)"


def test_viewport_reports_a_missing_context_instead_of_crashing(qapp, monkeypatch):
    """Without OpenGL the widget must degrade gracefully, not raise."""
    import moderngl

    def boom(*args, **kwargs):
        raise RuntimeError("no GL here")

    monkeypatch.setattr(moderngl, "create_standalone_context", boom)
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(400, 300)
    window.show()
    qapp.processEvents()
    viewport = window.viewport
    assert viewport.renderer is None
    assert "no GL here" in (viewport._gl_error or "")
    # Painting must still work (it draws a message instead of the scene).
    viewport.grab()
    window.close()


def test_loading_a_missing_file_does_not_block(qapp):
    """A scripted load of a bad path must not open a modal dialog."""
    window = DockingWorkbench()
    window.resize(400, 300)
    window.show()
    qapp.processEvents()
    assert window.load_receptor("this/does/not/exist.pdbqt") is False
    assert window.load_ligand("this/does/not/exist.pdbqt") is False
    assert window.load_poses("this/does/not/exist.pdbqt") is False
    assert "cannot read" in window.log.toPlainText()
    window.close()


def test_choosing_a_pose_switches_the_displayed_model(qapp):
    """Browsing poses must update the ligand shown in the viewport.

    The pose slider was removed from the dock (the table is how a pose is chosen
    now), so this drives the window's pose API, which is what the table, the
    arrow keys, the player and the console all call.
    """
    text = "MODEL 1\n" + LIGAND_PDBQT + "ENDMDL\n"
    text += "MODEL 2\n" + LIGAND_PDBQT.replace("0.000   0.000   0.000", "4.000   0.000   0.000") + "ENDMDL\n"
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, poses=text)
    window.resize(500, 400)
    window.show()
    qapp.processEvents()
    assert len(window.pose_models) == 2
    assert window.table.rowCount() == 2

    first = window.scene.ligand[0].x
    window.set_pose(1)
    qapp.processEvents()
    assert window.scene.ligand[0].x != first
    window.close()


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------

POSES_PDBQT = "MODEL 1\n" + LIGAND_PDBQT + "ENDMDL\nMODEL 2\n" + LIGAND_PDBQT + "ENDMDL\n"

MENU_TITLES = ["File", "Receptor", "Ligand", "Grid", "Docking", "Analysis", "View"]


def _menu_titles(window) -> list:
    return [a.text().replace("&", "") for a in window.menuBar().actions()]


def _dock_titles(window) -> set:
    return {d.windowTitle() for d in window.findChildren(QtWidgets.QDockWidget)}


def test_language_switch_rebuilds_the_ui_and_keeps_the_session(qapp):
    """``set_language`` must relabel everything without losing the session."""
    i18n.set_language("zh")
    window = DockingWorkbench(
        receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT, poses=POSES_PDBQT
    )
    window.resize(700, 520)
    window.show()
    qapp.processEvents()
    window.search.setCurrentIndex(1)
    window.exhaustiveness.setValue(12)
    window.flex_list.addItem("ALA1")
    window._refresh_flex_label()
    window.inspector.setCurrentIndex(2)
    window.workspace_dock.setVisible(False)
    qapp.processEvents()

    # Built in Chinese.
    assert window.windowTitle() == "OpenDocking 工作台"
    assert _menu_titles(window) == ["文件(F)", "受体(R)", "配体(L)", "网格(G)", "对接(D)", "分析(A)", "视图(V)"]
    assert [window.inspector.tabText(i) for i in range(4)] == ["受体", "配体", "网格", "引擎"]
    assert window.btn_run.text().endswith("开始对接")
    assert "构象与日志" in _dock_titles(window)
    assert "构象" in window.lbl_status.text()

    window.set_language("en")
    qapp.processEvents()

    # Back to English, and the session is intact.
    assert window.windowTitle() == "OpenDocking workbench"
    assert _menu_titles(window) == MENU_TITLES
    assert [window.inspector.tabText(i) for i in range(4)] == [
        "Receptor",
        "Ligand",
        "Grid",
        "Engine",
    ]
    assert window.btn_run.text().endswith("Start docking")
    assert "Poses & log" in _dock_titles(window)
    assert window.tree.headerItem().text(0) == "Project"
    assert window.table.horizontalHeaderItem(4).text() == "Key residues"

    assert window.search.currentIndex() == 1
    assert window.exhaustiveness.value() == 12
    assert window.inspector.currentIndex() == 2
    assert window.workspace_dock.isHidden()
    assert [window.flex_list.item(i).text() for i in range(window.flex_list.count())] == [
        "ALA1"
    ]
    window.workspace_dock.setVisible(True)

    assert window.receptor_text is not None and "ATOM" in window.receptor_text
    assert window.ligand_text is not None and "ATOM" in window.ligand_text
    assert len(window.scene.receptor) == 24
    assert len(window.scene.ligand) == 5
    assert len(window.pose_models) == 2
    assert window.table.rowCount() == 2
    # The log is history, so it keeps the language it was written in; it must
    # still be there in full.
    log = window.log.toPlainText()
    assert "受体：" in log
    assert "配体：" in log
    assert "language: English" in log
    window.close()


def test_every_visible_label_is_translated(qapp):
    """The seven menus, the docks, the tabs and the tool strip, both ways."""
    window = DockingWorkbench()
    window.resize(500, 400)
    window.show()
    qapp.processEvents()
    try:
        for code, expected in (("zh", "文件(F)"), ("en", "File")):
            window.set_language(code)
            qapp.processEvents()
            assert _menu_titles(window)[0] == expected
            assert window.inspector.tabText(0) == ("受体" if code == "zh" else "Receptor")
            assert window.workspace_dock.windowTitle() == (
                "工作区" if code == "zh" else "Workspace"
            )
            tips = [
                b.toolTip()
                for b in window._tool_strip.findChildren(QtWidgets.QToolButton)
            ]
            assert (tips[0] == "重置视图") if code == "zh" else (tips[0] == "reset the view")
    finally:
        window.set_language("en")
        window.close()


def test_the_language_submenu_is_radio_style_and_switches(qapp):
    window = DockingWorkbench()
    window.show()
    qapp.processEvents()

    def language_menu():
        # The View menu is the last one on the bar, whatever the language.
        view = window.menuBar().actions()[-1].menu()
        return next(
            a.menu()
            for a in view.actions()
            if a.menu() is not None and a.text() in ("Language", "语言")
        )

    entries = {a.text(): a for a in language_menu().actions()}
    assert set(entries) == {"English", "简体中文"}
    assert entries["English"].isChecked()
    assert not entries["简体中文"].isChecked()

    entries["简体中文"].trigger()
    QTest.qWait(80)
    assert i18n.current_language() == "zh"
    assert _menu_titles(window)[0] == "文件(F)"
    entries = {a.text(): a for a in language_menu().actions()}
    assert entries["简体中文"].isChecked()
    assert not entries["English"].isChecked()

    entries["English"].trigger()
    QTest.qWait(80)
    assert i18n.current_language() == "en"
    window.close()


def test_open_project_restores_the_saved_language(qapp, tmp_path, monkeypatch):
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    ligand = tmp_path / "ligand.pdbqt"
    ligand.write_text(LIGAND_PDBQT, encoding="utf-8")
    project = tmp_path / "session.json"
    project.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "zh",
                "receptor": str(receptor),
                "ligand": str(ligand),
                "engine": {"scoring": "vinardo"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(project), "")),
    )

    i18n.set_language("en")
    window = DockingWorkbench()
    window.show()
    qapp.processEvents()
    try:
        window._open_project()
        qapp.processEvents()
        assert i18n.current_language() == "zh"
        assert window.inspector.tabText(0) == "受体"
        assert len(window.scene.receptor) == 24
        assert window.receptor_text is not None
        assert len(window.scene.ligand) == 5
        assert window.engine.currentText() == "vinardo"
        assert str(project.name) in window.log.toPlainText()
    finally:
        window.set_language("en")
        window.close()


def test_saving_a_project_records_the_language(qapp, tmp_path, monkeypatch):
    target = tmp_path / "saved.json"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(target), "")),
    )
    i18n.set_language("zh")
    window = DockingWorkbench()
    window.show()
    qapp.processEvents()
    try:
        window._save_project()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["language"] == "zh"
    finally:
        i18n.set_language("en")
        window.close()


# ---------------------------------------------------------------------------
# Styles: bond geometry, ribbon and cartoon
# ---------------------------------------------------------------------------


def _backbone_pdbqt() -> str:
    """A tiny "protein": a ten-residue alpha-helix then a four-residue strand.

    Only CA atoms drive the cartoon, so the secondary-structure assignment has to
    come from the documented primary signal — the C-alpha distance pattern
    (helix: d(i,i+3) ≈ 5.0 Å and d(i,i+4) ≈ 6.2 Å; strand: d(i,i+2) ≈ 6.6 Å and
    d(i,i+3) ≈ 9.9 Å) — with no backbone dihedrals available at all. A bonded
    CB/OG side chain is added per residue so the sticks and ball-and-stick
    styles have real geometry to draw.
    """
    lines = []
    serial = 0

    def atom(name: str, res_id: int, x: float, y: float, z: float) -> None:
        nonlocal serial
        serial += 1
        lines.append(
            f"ATOM  {serial:5d} {name:<4s} ALA A{res_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00     0.000  C"
        )

    for index in range(10):  # ideal alpha-helix: 2.3 Å radius, 100°, 1.5 Å rise
        angle = math.radians(100.0 * index)
        cx = 2.3 * math.cos(angle)
        cy = 2.3 * math.sin(angle)
        cz = 1.5 * index
        atom("CA", index + 1, cx, cy, cz)
        atom("CB", index + 1, cx, cy, cz + 1.53)  # a real, bonded side chain
        atom("OG", index + 1, cx, cy + 1.43, cz + 1.53)
    for index in range(4):  # extended strand: 3.3 Å per residue
        cx, cy = 3.3 * index, 14.0
        atom("CA", 11 + index, cx, cy, 13.5)
        atom("CB", 11 + index, cx, cy, 15.03)
        atom("OG", 11 + index, cx, cy + 1.43, 15.03)
    return "\n".join(lines) + "\nTER\n"


HELIX_AND_STRAND = _backbone_pdbqt()


def test_the_backbone_fixture_is_what_the_cartoon_needs():
    """Guard the fixture itself: 14 residues, two secondary-structure elements."""
    from odock.gui.viewport import ca_trace, secondary_structure

    points, _residues, _kept = ca_trace(parse_pdbqt(HELIX_AND_STRAND)[0].atoms)
    assert len(points) == 14
    codes = secondary_structure(points)
    assert codes.count("H") >= 4, codes
    assert codes.count("E") >= 2, codes


def test_unknown_dihedrals_do_not_erase_a_helix():
    """A missing phi/psi is unknown information, not a contradiction."""
    from odock.gui.viewport import ca_trace, secondary_structure

    points, _residues, _kept = ca_trace(parse_pdbqt(HELIX_AND_STRAND)[0].atoms)
    codes = secondary_structure(points, [(None, None)] * len(points))
    assert codes.count("H") >= 4, codes
    assert codes.count("E") >= 2, codes


def test_every_protein_style_renders_geometry_and_switching_is_reversible(qapp):
    window = DockingWorkbench(receptor=HELIX_AND_STRAND, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"

    meshed = {"cartoon", "ribbon", "tube", "sticks", "ball_stick"}
    counts = {}
    for style in PROTEIN_STYLES:
        window._set_style("receptor", style)
        qapp.processEvents()
        assert window.scene.style_protein == style
        assert window.scene.style_receptor == style, "the legacy alias must follow"
        if style in meshed:
            assert renderer.mesh_receptor_vertices > 0, f"{style} built no triangles"
        else:
            assert renderer.receptor_count > 0, f"{style} drew no spheres"
        counts[style] = renderer.mesh_receptor_vertices

    # And a round trip rebuilds exactly the same geometry.
    window._set_style("receptor", "spheres")
    qapp.processEvents()
    assert renderer.mesh_receptor_vertices == 0
    window._set_style("receptor", "cartoon")
    qapp.processEvents()
    assert renderer.mesh_receptor_vertices == counts["cartoon"] > 0
    window.close()


def test_every_ligand_style_renders_geometry_and_switching_is_reversible(qapp):
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"
    assert window.scene.ligand_bonds, "the ligand must have perceived bonds"

    meshed = {"ball_stick", "sticks", "wireframe"}
    for style in LIGAND_STYLES:
        window._set_style("ligand", style)
        qapp.processEvents()
        assert window.scene.style_ligand == style
        if style in meshed:
            assert renderer.mesh_ligand_vertices > 0, f"{style} built no triangles"
        else:
            assert renderer.ligand_count == len(window.scene.ligand)

    window._set_style("ligand", "spheres")
    qapp.processEvents()
    assert renderer.mesh_ligand_vertices == 0
    window._set_style("ligand", "ball_stick")
    qapp.processEvents()
    assert renderer.mesh_ligand_vertices > 0
    window.close()


def test_the_view_menu_offers_every_style_and_the_bond_check(qapp):
    window = DockingWorkbench(receptor=HELIX_AND_STRAND, ligand=LIGAND_PDBQT)
    window.show()
    qapp.processEvents()
    try:
        view = window.menuBar().actions()[-1].menu()
        submenus = {a.text(): a.menu() for a in view.actions() if a.menu() is not None}
        protein = submenus["Protein style"]
        ligand = submenus["Ligand style"]
        assert [a.text() for a in protein.actions()]
        assert len(protein.actions()) == len(PROTEIN_STYLES)
        assert len(ligand.actions()) == len(LIGAND_STYLES)
        # Radio style: exactly one tick per submenu.
        assert sum(1 for a in protein.actions() if a.isChecked()) == 1
        assert sum(1 for a in ligand.actions() if a.isChecked()) == 1

        cartoon = next(a for a in protein.actions() if a.text() == "Cartoon")
        cartoon.trigger()
        assert window.scene.style_protein == "cartoon"
        assert cartoon.isChecked()
        assert sum(1 for a in protein.actions() if a.isChecked()) == 1

        wireframe = next(a for a in ligand.actions() if a.text() == "Wireframe")
        wireframe.trigger()
        assert window.scene.style_ligand == "wireframe"
        assert wireframe.isChecked()

        # The submenu titles are translated like everything else.
        assert any(a.text() == "Bond check…" for a in view.actions())
        window.set_language("zh")
        qapp.processEvents()
        view = window.menuBar().actions()[-1].menu()
        assert any(a.text() == "键检查…" for a in view.actions())
        titles = [a.text() for a in view.actions() if a.menu() is not None]
        assert "蛋白样式" in titles and "配体样式" in titles
    finally:
        i18n.set_language("en")
        window.close()


def test_the_bond_check_dialog_shows_the_perceived_connectivity(qapp):
    from odock.gui import dialogs
    from odock.gui.bonds import bond_report, perceive_bonds

    atoms = parse_pdbqt(LIGAND_PDBQT)[0].atoms
    report = bond_report(atoms, perceive_bonds(atoms, kind="ligand"))
    assert report["n_bonds"] >= 3
    dialog = dialogs.BondCheckDialog([("Ligand", report)], None)
    try:
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 0).text() == "Ligand"
        assert dialog.table.item(0, 2).text() == str(report["n_bonds"])
        assert dialog.table.item(0, 3).text()  # the degree histogram
        assert "Ligand" in dialog.detail.text()
        assert dialog.detail.text().strip()
        # It must lay out and paint, not merely construct.
        dialog.show()
        qapp.processEvents()
        painted = dialog.grab()
        assert painted.width() > 200 and painted.height() > 100
    finally:
        dialog.reject()


def test_perceived_bonds_are_used_for_the_rendered_geometry(qapp):
    """The sticks come from the perception module, not from a distance test."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.show()
    qapp.processEvents()
    try:
        from odock.gui.bonds import Bond

        bonds = window.scene.ligand_bonds
        assert bonds and all(isinstance(bond, Bond) for bond in bonds)
        atoms = window.scene.ligand
        longest = max(
            math.dist(atoms[b.a].position, atoms[b.b].position) for b in bonds
        )
        assert longest < 2.0, f"a perceived bond is {longest:.2f} Å long"
    finally:
        window.close()


@pytest.mark.skipif(
    os.environ.get("ODOCK_STYLE_SHOTS") != "1",
    reason="set ODOCK_STYLE_SHOTS=1 to write the out/styles contact sheets",
)
def test_style_shots_opt_in(qapp):
    """An opt-in visual check: one rendered image per style.

    ``$env:ODOCK_STYLE_SHOTS='1'; .\\.venv\\Scripts\\python.exe -m pytest
    tests/test_gui_viewport.py -q -k style_shots`` writes
    ``out/styles/protein-<style>.png`` and ``ligand-<style>.png`` with the other
    group hidden, so the geometry of the style under test is what fills the
    frame. The images are meant to be looked at, not asserted.
    """
    demo = Path(__file__).resolve().parent.parent / "demo" / "systems" / "3ptb"
    receptor = (demo / "receptor.pdbqt").read_text(encoding="utf-8")
    ligand = (demo / "ligand.pdbqt").read_text(encoding="utf-8")
    out = Path(__file__).resolve().parent.parent / "out" / "styles"
    out.mkdir(parents=True, exist_ok=True)

    window = DockingWorkbench(receptor=receptor, ligand=ligand)
    window.resize(1000, 720)
    window.show()
    qapp.processEvents()
    try:
        window._toggle_ligand(False)
        window.viewport.frame_all()
        qapp.processEvents()
        for style in PROTEIN_STYLES:
            window._set_style("receptor", style)
            qapp.processEvents()
            assert window.viewport.snapshot(out / f"protein-{style}.png", 1200, 900)

        window._toggle_ligand(True)
        window._toggle_receptor(False)
        window.viewport.frame_ligand()
        qapp.processEvents()
        for style in LIGAND_STYLES:
            window._set_style("ligand", style)
            qapp.processEvents()
            assert window.viewport.snapshot(out / f"ligand-{style}.png", 1200, 900)
    finally:
        window.close()
    written = sorted(path.name for path in out.glob("*.png"))
    assert len(written) >= len(PROTEIN_STYLES) + len(LIGAND_STYLES)


# ---- sequence track and selection panel ----


def test_the_residue_ruler_sits_under_the_viewport(qapp):
    """The ruler is in the central splitter, below the 3-D view, drag-able."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(1000, 700)
    window.show()
    qapp.processEvents()
    try:
        splitter = window.central_splitter
        assert splitter.indexOf(window.viewport) < splitter.indexOf(window.sequence_area)
        # Same column, and the ruler is the lower of the two.
        assert window.sequence_area.y() > window.viewport.y()
        assert window.sequence_area.height() > 0
        assert splitter.handleWidth() >= 6, "the divider has to be grabbable"
        assert not splitter.childrenCollapsible()
        loader = window.sequence
        assert loader.rows()  # the fixture's 24 CB atoms form one residue
    finally:
        window.close()


def test_a_ruler_selection_highlights_atoms_and_fills_the_panel(qapp):
    """One click in the ruler must reach the scene and the atom table."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.show()
    qapp.processEvents()
    try:
        ruler = window.sequence
        assert len(ruler.blocks()) >= 2
        ruler.select_keys([ruler.blocks()[0].key])
        qapp.processEvents()
        assert window.scene.selection == ruler.atom_refs()
        assert window.scene.selection
        assert window.selection_table.rowCount() == len(window.scene.selection)
        assert window.selection_table.columnCount() == 9
        assert window.lbl_selection_summary.text()
        # Clearing the ruler clears the 3-D selection again.
        ruler.clear_selection()
        qapp.processEvents()
        assert window.scene.selection == []
        assert window.selection_table.rowCount() == 0
    finally:
        window.close()


def test_the_selection_panel_is_in_the_panels_menu(qapp):
    window = DockingWorkbench()
    window.show()
    qapp.processEvents()
    try:
        view = window.menuBar().actions()[-1].menu()
        panels = next(
            action.menu()
            for action in view.actions()
            if action.menu() is not None and action.text() == "Panels"
        )
        titles = [action.text() for action in panels.actions()]
        assert "Selection" in titles
        assert "Workspace" in titles
    finally:
        window.close()


# ---------------------------------------------------------------------------
# Space filling, the selection and the reference-pose overlays
# ---------------------------------------------------------------------------


def _rendered_frame(window, width: int = 360, height: int = 270) -> np.ndarray:
    """One frame straight from the GL context, as an (h, w, 3) int array.

    Ambient occlusion is switched off for the duration: these tests are about
    which atoms are drawn, not about the shading, and the post-pass doubles the
    render cost.
    """
    scene = window.scene
    previous = scene.ssao
    scene.ssao = False
    try:
        data = window.viewport.renderer.render_image(
            window.viewport.camera, width, height
        )
    finally:
        scene.ssao = previous
    return np.frombuffer(data, dtype="u1").reshape(height, width, 3).astype(int)


def _sphere_radii(renderer, window, which: str) -> np.ndarray:
    """The radii of the instanced spheres a style would draw (white-box).

    ``_instances_for`` is exactly what gets uploaded, so this is the honest way
    to assert a style's radii without measuring pixels.
    """
    atoms = window.scene.ligand if which == "ligand" else window.scene.receptor
    return renderer._instances_for(atoms, which)[:, 3]


def test_spacefill_draws_every_atom_at_its_real_van_der_waals_radius(qapp):
    """``spacefill`` must be 1:1 — no shrink factor at all."""
    from odock.gui.structure import element_radius

    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"

    elements = [atom.element for atom in window.scene.ligand]
    carbon = elements.index("C")
    oxygen = elements.index("O")

    window._set_style("ligand", "spheres")
    qapp.processEvents()
    shrunk = _sphere_radii(renderer, window, "ligand")

    window._set_style("ligand", "spacefill")
    qapp.processEvents()
    assert window.scene.style_ligand == "spacefill"
    assert renderer.ligand_count == len(window.scene.ligand)
    assert renderer.mesh_ligand_vertices == 0, "a space-filling ligand has no sticks"

    full = _sphere_radii(renderer, window, "ligand")
    assert full[carbon] == pytest.approx(element_radius("C"), abs=1e-6)
    assert full[oxygen] == pytest.approx(element_radius("O"), abs=1e-6)
    assert full[carbon] > shrunk[carbon] * 1.5, "space filling must be visibly larger"

    # The protein follows the same rule, so a ligand in the pocket can be read
    # against the receptor's real surface.
    window._set_style("receptor", "spacefill")
    qapp.processEvents()
    assert renderer.receptor_count == len(window.scene.receptor)
    receptor_radii = _sphere_radii(renderer, window, "receptor")
    carbon_radius = element_radius("C")
    assert np.allclose(receptor_radii, carbon_radius, atol=1e-6)
    window.close()


def test_the_selection_is_drawn_and_an_empty_selection_is_a_no_op(qapp):
    """``scene.selection`` must paint when set and draw nothing when empty."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"
    window.viewport.frame_ligand()
    qapp.processEvents()

    window.scene.selection = []
    window.viewport.refresh()
    qapp.processEvents()
    empty = _rendered_frame(window)
    assert renderer.selection_count == 0

    window.scene.selection = [("ligand", 0), ("ligand", 1), ("receptor", 0)]
    window.viewport.refresh()
    qapp.processEvents()
    assert renderer.selection_count == 3
    marked = _rendered_frame(window)
    changed = int((np.abs(marked - empty).sum(axis=2) > 12).sum())
    assert changed > 40, f"only {changed} pixels changed for three selected atoms"

    # Malformed or out-of-range entries are skipped, not raised on.
    window.scene.selection = [("ligand", 10_000), ("nonsense", 0), ("ligand", "x")]
    window.viewport.refresh()
    qapp.processEvents()
    assert renderer.selection_count == 0

    window.scene.selection = []
    window.viewport.refresh()
    qapp.processEvents()
    restored = _rendered_frame(window)
    assert int(np.abs(restored - empty).max()) == 0, "clearing must restore the frame"
    window.close()


def test_the_reference_ghost_overlay_is_drawn_only_when_set(qapp):
    """The translucent reference pose is its own pass, and it is optional."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"
    window.viewport.frame_ligand()
    qapp.processEvents()

    assert window.scene.ghost_ligand == []
    assert renderer.ghost_count == 0
    without = _rendered_frame(window)

    window.scene.ghost_ligand = list(window.scene.ligand)
    window.scene.ghost_scale = 2.2  # make the overlay obvious for the pixel test
    window.viewport.refresh()
    qapp.processEvents()
    assert renderer.ghost_count == len(window.scene.ligand)
    assert renderer.ligand_count == len(window.scene.ligand), "the pose stays drawn"
    with_ghost = _rendered_frame(window)
    assert int((np.abs(with_ghost - without).sum(axis=2) > 12).sum()) > 40

    window.scene.ghost_ligand = []
    window.viewport.refresh()
    qapp.processEvents()
    assert renderer.ghost_count == 0
    window.close()


def test_the_interaction_distances_dialog_round_trips(qapp):
    """The dialog is pre-filled, returns the cut-offs, and can be reset."""
    from odock.gui import dialogs

    dialog = dialogs.InteractionThresholdsDialog()
    assert dialog.values() == pytest.approx(dialogs.DEFAULT_INTERACTION_THRESHOLDS)
    dialog.set_values({"hbond": 4.2, "clash_ratio": 0.6})
    values = dialog.values()
    assert values["hbond"] == pytest.approx(4.2)
    assert values["clash_ratio"] == pytest.approx(0.6)
    assert set(values) == set(dialogs.DEFAULT_INTERACTION_THRESHOLDS)
    # The ranges the specification asks for.
    assert 0.5 <= values["hydrophobic"] <= 10.0
    assert 0.3 <= values["clash_ratio"] <= 1.2
    dialog.reset()
    assert dialog.values() == pytest.approx(dialogs.DEFAULT_INTERACTION_THRESHOLDS)
    dialog.close()


def test_the_workbench_keeps_and_uses_the_interaction_thresholds(qapp, monkeypatch):
    """View ▸ Interaction distances… stores the answer and re-annotates with it."""
    from odock.gui import dialogs

    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    assert window.interaction_thresholds == dialogs.DEFAULT_INTERACTION_THRESHOLDS

    real = dialogs.InteractionThresholdsDialog

    class AutoAccept(real):  # a dialog a human would have filled in and accepted
        def exec(self):
            self.set_values({"hbond": 4.0, "clash_ratio": 0.9})
            return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(dialogs, "InteractionThresholdsDialog", AutoAccept)
    window._choose_interaction_distances()
    qapp.processEvents()
    assert window.interaction_thresholds["hbond"] == pytest.approx(4.0)
    assert window.interaction_thresholds["clash_ratio"] == pytest.approx(0.9)
    assert "interaction distances" in window.log.toPlainText()

    # A rejected dialog must leave the values alone.
    class Reject(real):
        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(dialogs, "InteractionThresholdsDialog", Reject)
    window._choose_interaction_distances()
    assert window.interaction_thresholds["hbond"] == pytest.approx(4.0)

    # The cut-offs live on the window, so rebuilding the UI for another language
    # keeps them (and the menu entry that edits them).
    window.set_language("zh")
    qapp.processEvents()
    assert window.interaction_thresholds["hbond"] == pytest.approx(4.0)
    assert window.interaction_thresholds["clash_ratio"] == pytest.approx(0.9)
    window.set_language("en")
    qapp.processEvents()
    assert window.interaction_thresholds["hbond"] == pytest.approx(4.0)
    window.close()


def test_selecting_a_pose_reannotates_but_never_moves_the_camera(qapp, monkeypatch):
    """Choosing a pose shows where it binds — without stealing the camera.

    Each pose change re-runs the contact search with the user's
    ``interaction_thresholds``, refreshes the table row and the pose label, and
    leaves every camera number exactly as the user left it. Framing the binding
    site (and with it the pocket clip and the front cut-away) is an explicit
    action, `View ▸ Frame binding site`.
    """
    from odock.gui import i18n

    analysis = pytest.importorskip("odock.analysis")
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, poses=POSES_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    assert len(window.pose_models) == 2

    # Watch which cut-offs the automatic annotation is given.
    seen: list = []
    real_profile = analysis.profile_interactions

    def spy(receptor, ligand, **kwargs):
        seen.append(dict(kwargs))
        return real_profile(receptor, ligand, **kwargs)

    monkeypatch.setattr(analysis, "profile_interactions", spy)
    window.interaction_thresholds["hbond"] = 4.4

    window.viewport.frame_all()
    qapp.processEvents()
    before = (
        tuple(window.viewport.camera.target),
        float(window.viewport.camera.distance),
        float(window.viewport.camera.azimuth),
        float(window.viewport.camera.elevation),
    )

    window.set_pose(1)
    qapp.processEvents()
    assert window._pose_index == 1

    # The contacts of this pose were recomputed, with the stored thresholds...
    assert seen, "choosing a pose must re-run the contact search"
    assert seen[-1]["hbond"] == pytest.approx(4.4)
    assert isinstance(window.scene.interactions, list)
    # What is *drawn* is the detection set reduced for legibility: the kind filter
    # plus one hydrophobic line per receptor residue. Every drawn line must
    # therefore be a detected one, and a residue must appear at most once.
    detected = {(item.kind, item.a, item.b) for item in window.interactions}
    drawn = {(item.kind, item.a, item.b) for item in window.scene.interactions}
    assert drawn <= detected
    assert len(drawn) == len(window.scene.interactions), "no duplicate lines"
    hydrophobic = [
        item for item in window.scene.interactions if item.kind == "hydrophobic"
    ]
    assert len(hydrophobic) == len({tuple(item.residue) for item in hydrophobic})

    # ...the pose label names what it binds (when it binds anything)...
    label = window.lbl_pose.text()
    assert "mode 2" in label
    summary = analysis.interaction_summary(
        window.interactions, window.scene.receptor, window.scene.ligand
    )
    if summary:
        assert summary in label, label

    # ...and not one camera number moved.
    after = (
        tuple(window.viewport.camera.target),
        float(window.viewport.camera.distance),
        float(window.viewport.camera.azimuth),
        float(window.viewport.camera.elevation),
    )
    assert after == before, f"a pose change moved the camera: {before} -> {after}"

    # The reference overlay is the first pose, and it is opt-in.
    window._toggle_ghost(True)
    qapp.processEvents()
    assert window._ghost_visible
    assert len(window.scene.ghost_ligand) == len(window.pose_models[0].atoms)
    window.set_pose(0)
    qapp.processEvents()
    assert window.scene.ghost_ligand == [], "pose 1 is its own reference"
    window._toggle_ghost(False)
    assert window.scene.ghost_ligand == []

    # View ▸ Frame binding site is the explicit way to get the pocket view: it
    # frames the site and clips the receptor (receptor_cutoff + front cut-away).
    frame_action = next(
        action
        for bar in window.menuBar().actions()
        for action in (bar.menu().actions() if bar.menu() else [])
        if action.text().replace("&", "") == i18n.tr("action.frame_site")
    )
    frame_action.trigger()
    qapp.processEvents()
    assert window.scene.front_clip, "framing the site must enable the front cut-away"
    assert window.scene.receptor_cutoff is not None
    assert 0 < len(window.scene.visible_receptor()) < len(window.scene.receptor)
    assert window.viewport.camera.distance < before[1]

    # Home / Reset view leaves it again.
    window.viewport.frame_all()
    qapp.processEvents()
    assert not window.scene.front_clip
    assert window.scene.receptor_cutoff is None
    assert len(window.scene.visible_receptor()) == len(window.scene.receptor)
    window.close()


# ---------------------------------------------------------------------------
# The search box, the interaction emphasis, and a pose-invariant protein
# ---------------------------------------------------------------------------


class _Contact:
    """A minimal ``analysis.Interaction``: the focus reads only ``a``/``b``."""

    def __init__(self, a: int, b: int, kind: str = "hbond") -> None:
        self.a = a
        self.b = b
        self.kind = kind
        self.distance = 2.9
        self.detail = ""


def test_the_box_is_a_translucent_fill_without_grips(qapp):
    """No orange spheres; a fill whose opacity is respected; edges always on."""
    from odock.gui import dialogs

    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"

    window.scene.box = ((1.5, 1.2, 0.0), (8.0, 8.0, 8.0))
    assert window.scene.show_box_handles is False, "the grips must be off by default"
    assert window.scene.box_alpha == pytest.approx(dialogs.DEFAULT_BOX_ALPHA)

    window.viewport.grab()
    assert renderer.handle_count == 0, "the six face grips must not be drawn"
    assert renderer.box_fill_triangles == 6, "three camera-facing faces, two triangles each"
    assert renderer.box_edge_vertices == 24

    # 0.0 hides the fill and keeps the edges, as documented.
    window.scene.box_alpha = 0.0
    window.viewport.grab()
    assert renderer.box_fill_triangles == 0
    assert renderer.box_edge_vertices == 24

    # The grips still exist behind the flag, and so does the picking API.
    window.scene.show_box_handles = True
    window.viewport.grab()
    assert renderer.handle_count == 6
    assert len(renderer.box_handle_positions()) == 6
    window.scene.show_box_handles = False
    window.close()


def test_the_box_opacity_dialog_round_trips_and_clamps(qapp):
    from odock.gui import dialogs

    assert dialogs.DEFAULT_BOX_ALPHA == pytest.approx(0.22)
    dialog = dialogs.BoxOpacityDialog(dialogs.DEFAULT_BOX_ALPHA)
    assert dialog.value() == pytest.approx(dialogs.DEFAULT_BOX_ALPHA)
    dialog.set_value(0.7)
    assert dialog.value() == pytest.approx(0.7)
    dialog.set_value(-3.0)
    assert dialog.value() == pytest.approx(0.0)
    dialog.set_value(9.0)
    assert dialog.value() == pytest.approx(1.0)
    dialog.close()


def test_no_box_fill_when_the_camera_is_inside_the_box(qapp):
    """From inside, every face would cover the viewport: fill off, edges on."""
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"

    # A box big enough to contain the camera whatever `frame_all` chooses.
    window.scene.box = ((0.0, 0.0, 0.0), (400.0, 400.0, 400.0))
    window.viewport.frame_all()
    qapp.processEvents()
    eye = window.viewport.camera.eye()
    centre, size = window.scene.box
    half = [value / 2 for value in size]
    assert all(abs(eye[i] - centre[i]) <= half[i] for i in range(3)), (
        "the fixture is supposed to put the camera inside the box"
    )
    window.viewport.grab()
    assert renderer.box_fill_triangles == 0, "the fill must be skipped from inside"
    assert renderer.box_edge_vertices == 24, "the edges must still outline the box"

    # With the camera outside, the fill comes back.
    window.scene.box = ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    window.viewport.frame_all()
    qapp.processEvents()
    window.viewport.grab()
    assert renderer.box_fill_triangles == 6
    window.close()


def test_the_box_fill_never_paints_over_an_atom_in_front_of_it(qapp):
    """The user's bug: after loading a ligand the receptor looked washed out.

    Reproduces the reported sequence on the bundled 3PTB data: load the
    receptor, load the ligand (which fits the box to it and frames the ligand
    close, so the camera sits just outside the box with the box filling the
    view), then compare a frame with the box against one without it. Because the
    fill is depth-tested, the receptor atoms nearer to the camera than the box's
    near face keep *their exact pixels*; with the old depth-test-off fill every
    one of them was tinted, which is what hid the receptor.
    """
    root = Path(__file__).resolve().parent.parent
    receptor = root / "demo" / "systems" / "3ptb" / "receptor.pdbqt"
    ligand = root / "demo" / "systems" / "3ptb" / "ligand.pdbqt"
    if not (receptor.exists() and ligand.exists()):  # pragma: no cover - data guard
        pytest.skip("the bundled 3PTB files are not present")

    window = DockingWorkbench()
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    assert window.load_receptor(str(receptor)) is True
    window.load_ligand(str(ligand))  # fits the box to the ligand
    qapp.processEvents()
    assert window.scene.box is not None
    box = window.scene.box
    # Place the camera explicitly just outside the box (this is the state the
    # user reported, and the worst case for a translucent fill), rather than
    # inheriting whatever framing `load_ligand` happens to use.
    centre = np.asarray(box[0], dtype="f8")
    half = np.asarray(box[1], dtype="f8") / 2.0
    window.viewport.camera.target = tuple(centre)
    window.viewport.camera.distance = float(np.linalg.norm(half)) * 1.15
    window.viewport.refresh()
    qapp.processEvents()
    eye = np.asarray(window.viewport.camera.eye(), dtype="f8")
    assert not bool(np.all(np.abs(eye - centre) <= half)), "the camera must be outside"

    width, height = 360, 270
    window.scene.box = None
    qapp.processEvents()
    reference = _rendered_frame(window, width, height)

    window.scene.box = box
    window.scene.box_alpha = 0.35
    qapp.processEvents()
    with_box = _rendered_frame(window, width, height)

    window.scene.box_alpha = 0.0  # edges only: the fill contributes nothing
    qapp.processEvents()
    edges_only = _rendered_frame(window, width, height)

    drawn = np.abs(reference - BACKGROUND).sum(axis=2) > 12
    assert drawn.sum() > 2000, f"nothing was drawn ({int(drawn.sum())} px)"

    # Pixels that show something and are *identical* with and without the box:
    # those are the atoms the fill is behind, i.e. the ones it must not touch.
    untouched = (np.abs(with_box - reference).sum(axis=2) <= 12) & drawn
    assert untouched.sum() >= 200, (
        f"the fill touched every drawn pixel ({int(untouched.sum())} of "
        f"{int(drawn.sum())} survived): it is painting over atoms in front of it"
    )

    # ...and the fill is genuinely drawn (the difference from the edge-only frame
    # is the fill itself).
    fill_pixels = int((np.abs(with_box - edges_only).sum(axis=2) > 12).sum())
    assert fill_pixels > 200, f"the fill was not drawn ({fill_pixels} px changed)"

    # The receptor must not have collapsed into the wash: a healthy share of the
    # frame keeps its own colour rather than turning into the box tint.
    assert untouched.sum() > 0.02 * int(drawn.sum()), (
        f"only {int(untouched.sum())} of {int(drawn.sum())} drawn pixels are "
        "un-tinted; the receptor is being washed out"
    )
    window.close()


def test_the_interaction_dashes_are_visible_not_hairlines(qapp):
    """The user's complaint: the contacts could not be seen.

    A GL line is one pixel wide whatever the hardware, which measured ~44 changed
    pixels for six contacts on a 700x500 frame. Each dash is now a tube whose
    radius is chosen for a fixed on-screen width, so the pass must contribute an
    order of magnitude more than that.
    """
    root = Path(__file__).resolve().parent.parent
    receptor = root / "demo" / "systems" / "3ptb" / "receptor.pdbqt"
    ligand = root / "demo" / "systems" / "3ptb" / "ligand.pdbqt"
    poses = root / "demo" / "systems" / "3ptb" / "poses.pdbqt"
    if not (receptor.exists() and ligand.exists() and poses.exists()):
        pytest.skip("the bundled 3PTB files are not present")  # pragma: no cover

    window = DockingWorkbench(
        receptor=str(receptor), ligand=str(ligand), poses=str(poses)
    )
    window.resize(800, 560)
    window.show()
    qapp.processEvents()
    assert window.viewport.renderer is not None
    window.set_pose(2)
    qapp.processEvents()
    window._annotate_interactions(quiet=True)
    qapp.processEvents()
    assert window.interactions, "the pose must have contacts for this test"

    viewport = window.viewport
    viewport.frame_binding_site(clip=False)
    qapp.processEvents()
    # Isolate the dash pass from the emphasis.
    window.scene.interaction_focus = []
    viewport.refresh(upload_receptor=True)
    qapp.processEvents()

    width, height = 700, 500
    with_dashes = _rendered_frame(window, width, height)
    saved = list(window.scene.interactions)
    window.scene.interactions = []
    qapp.processEvents()
    without = _rendered_frame(window, width, height)
    window.scene.interactions = saved
    qapp.processEvents()

    changed = int((np.abs(with_dashes - without).sum(axis=2) > 12).sum())
    # The budget is per contact, not absolute: the number of drawn lines depends
    # on the pose and on the one-line-per-residue reduction, so a fixed total
    # would measure the pose rather than the dash width.
    per_contact = changed / max(1, len(saved))
    assert per_contact > 60, (
        f"the contact dashes are hairline-thin again: {changed} pixels changed "
        f"for {len(saved)} contacts ({per_contact:.0f} each; the old gl.LINES "
        f"pass measured ~44 in total)"
    )
    window.close()


def test_the_emphasis_suppresses_the_base_spheres_it_replaces(qapp):
    """A focused residue drawn twice is what buried the ball-and-stick."""
    root = Path(__file__).resolve().parent.parent
    receptor = root / "demo" / "systems" / "3ptb" / "receptor.pdbqt"
    ligand = root / "demo" / "systems" / "3ptb" / "ligand.pdbqt"
    poses = root / "demo" / "systems" / "3ptb" / "poses.pdbqt"
    if not (receptor.exists() and ligand.exists() and poses.exists()):
        pytest.skip("the bundled 3PTB files are not present")  # pragma: no cover

    window = DockingWorkbench(
        receptor=str(receptor), ligand=str(ligand), poses=str(poses)
    )
    window.resize(800, 560)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None
    window._annotate_interactions(quiet=True)
    qapp.processEvents()
    assert window.scene.interaction_focus, "Show interactions must set the focus"

    focused_receptor = renderer.focus_receptor_atoms
    focused_ligand = renderer.focus_ligand_atoms
    assert focused_receptor > 0 and focused_ligand > 0
    assert renderer.focus_mesh_receptor_vertices > 0, "the ball-and-stick must exist"

    # The base pass drops exactly the atoms the emphasis draws itself, so the
    # sticks are not buried inside their own spheres.
    with_focus = renderer.receptor_count
    ligand_with_focus = renderer.ligand_count
    saved = list(window.scene.interaction_focus)
    window.scene.interaction_focus = []
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()
    without = renderer.receptor_count
    ligand_without = renderer.ligand_count
    window.scene.interaction_focus = saved
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()

    assert without - with_focus == focused_receptor, (
        f"expected {focused_receptor} focused receptor spheres to be suppressed, "
        f"got {without - with_focus}"
    )
    assert ligand_without - ligand_with_focus == focused_ligand
    assert renderer.receptor_count == with_focus, "and the counts must come back"
    assert renderer.focus_sphere_count == focused_receptor + focused_ligand
    window.close()


def test_loading_a_ligand_does_not_wash_out_the_receptor(qapp):
    """After a ligand import the receptor must still be plainly visible."""
    window = DockingWorkbench()
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    assert window.load_receptor(RECEPTOR_PDBQT) is True
    window.load_ligand(LIGAND_PDBQT)  # frames the ligand, close to the box
    qapp.processEvents()
    assert window.scene.box is not None, "the ligand load defines the search box"

    frame = _rendered_frame(window, 360, 270)
    painted = int((np.abs(frame - BACKGROUND).sum(axis=2) > 12).sum())
    assert painted > 2000, f"almost nothing is drawn after the ligand load ({painted} px)"

    # The box colour over the background is the "pure wash"; it must not be the
    # bulk of what is on screen, otherwise the receptor is hidden behind it.
    wash = np.array([0.30, 0.85, 0.95]) * window.scene.box_alpha * 255 + BACKGROUND * (
        1.0 - window.scene.box_alpha
    )
    is_wash = np.abs(frame - wash).sum(axis=2) <= 30
    assert is_wash.sum() < 0.6 * painted, (
        f"the box fill dominates the frame: {int(is_wash.sum())} wash pixels of "
        f"{painted} painted"
    )
    window.close()


def test_every_pose_draws_the_same_protein(qapp):
    """The user's bug: the protein's "transparency" changed with the pose.

    Nothing automatic may enable the front cut-away, and only the ligand may
    differ from pose to pose. The receptor is compared pixel by pixel with the
    ligand hidden, which is the strongest form of "the protein did not change".
    """
    # Two genuinely different poses: the second shifts the first ligand atom by
    # 4 Å, so "the ligand moved" is testable as well. (``POSES_PDBQT`` holds two
    # identical models, so it cannot show that.)
    poses = (
        "MODEL 1\n" + LIGAND_PDBQT + "ENDMDL\nMODEL 2\n"
        + LIGAND_PDBQT.replace("0.000   0.000   0.000", "4.000   0.000   0.000")
        + "ENDMDL\n"
    )
    window = DockingWorkbench(receptor=RECEPTOR_PDBQT, poses=poses)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    assert window.viewport.renderer is not None
    assert len(window.pose_models) == 2

    states = []
    for index in range(len(window.pose_models)):
        window.set_pose(index)
        qapp.processEvents()
        states.append(
            (
                window.scene.front_clip,
                window.scene.receptor_cutoff,
                window.scene.receptor_radius,
            )
        )
    assert states == [(False, None, 0.0)] * len(window.pose_models), states

    # Pixel-identical protein across the poses: the ligand, the per-pose
    # contact dashes *and* the per-pose emphasis are cleared, so what remains to
    # compare is the protein itself. (The emphasis is meant to follow the pose —
    # it is what `Analysis ▸ Show interactions` asks for — so it is excluded
    # here rather than asserted away.)
    window.viewport.frame_all()
    qapp.processEvents()
    window.scene.show_ligand = False
    window.scene.interactions = []
    window.scene.interaction_focus = []
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()

    def clear_annotation() -> None:
        window.scene.interactions = []
        window.scene.interaction_focus = []
        window.viewport.refresh(upload_receptor=True)
        qapp.processEvents()

    try:
        frames = []
        for index in range(len(window.pose_models)):
            window.set_pose(index)
            qapp.processEvents()
            clear_annotation()
            frames.append(_rendered_frame(window))
        assert np.array_equal(frames[0], frames[1]), (
            "the protein changed between poses: "
            f"{int((np.abs(frames[0] - frames[1]).sum(axis=2) > 0).sum())} pixels differ"
        )
    finally:
        window.scene.show_ligand = True

    # With the ligand shown the two poses do differ — otherwise the test above
    # would pass for the wrong reason.
    window.viewport.frame_ligand()
    qapp.processEvents()
    shots = []
    for index in range(len(window.pose_models)):
        window.set_pose(index)
        qapp.processEvents()
        shots.append(_rendered_frame(window))
    assert not np.array_equal(shots[0], shots[1])
    window.close()


def test_the_interaction_focus_emphasises_the_site_and_clears(qapp):
    """Focusing draws the residues as ball-and-stick over a dimmed protein."""
    window = DockingWorkbench(receptor=HELIX_AND_STRAND, ligand=LIGAND_PDBQT)
    window.resize(640, 480)
    window.show()
    qapp.processEvents()
    renderer = window.viewport.renderer
    assert renderer is not None, f"no GL context: {window.viewport._gl_error}"
    assert window.scene.interaction_focus == []

    base_mesh = renderer.mesh_receptor_vertices
    base_spheres = renderer.receptor_count

    window.scene.interaction_focus = [_Contact(0, 0), _Contact(3, 1)]
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()
    # A residue is emphasised whole, so its side chain comes with its C-alpha.
    assert renderer.focus_residues >= 2
    assert renderer.focus_receptor_atoms > 2
    assert renderer.focus_ligand_atoms == 2
    assert renderer.focus_mesh_receptor_vertices > 0, "the residues must be ball-and-stick"
    assert renderer.focus_mesh_ligand_vertices > 0
    window.viewport.grab()
    assert renderer.focus_sphere_count == (
        renderer.focus_receptor_atoms + renderer.focus_ligand_atoms
    )

    # Clearing restores exactly the previous geometry.
    window.scene.interaction_focus = []
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()
    window.viewport.grab()
    assert renderer.focus_mesh_receptor_vertices == 0
    assert renderer.focus_sphere_count == 0
    assert renderer.focus_receptor_atoms == 0
    assert renderer.mesh_receptor_vertices == base_mesh
    assert renderer.receptor_count == base_spheres

    # A malformed focus list is ignored, never raised on.
    window.scene.interaction_focus = ["nonsense", (1,), None, object()]
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()
    window.viewport.grab()
    assert renderer.focus_receptor_atoms == 0
    assert renderer.focus_sphere_count == 0

    # The explicit API agrees with the scene attribute (and takes residue keys).
    counts = renderer.set_interaction_focus([0], [0])
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()
    assert counts["residues"] >= 1
    assert counts["receptor_atoms"] > 1
    assert counts["ligand_atoms"] == 1
    renderer.clear_interaction_focus()
    window.viewport.refresh(upload_receptor=True)
    qapp.processEvents()
    assert renderer.focus_receptor_atoms == 0
    window.close()

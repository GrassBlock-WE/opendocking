# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the 3-D workbench's non-GUI layers and, when a GPU is present, its
OpenGL render path.

The Qt widget itself cannot be exercised head-lessly (the ``offscreen`` Qt
platform has no OpenGL), so the renderer is instead driven through a standalone
ModernGL context and the resulting framebuffer is inspected. That exercises the
shaders, the instanced upload and the camera maths — everything that can
actually be wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from odock.gui import structure as gui_structure
from odock.gui.viewport import Camera, Scene

PDBQT = """\
REMARK  VINA RESULT:      -7.991      0.000      0.000
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.390   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
BRANCH    2   3
ATOM      3  O1  LIG A   1       1.900   1.200   0.000  1.00  0.00    -0.300 OA
ENDBRANCH    2   3
TORSDOF 1
"""


def test_parse_pdbqt_extracts_atoms_and_the_affinity():
    models = gui_structure.parse_pdbqt(PDBQT)
    assert len(models) == 1
    model = models[0]
    assert len(model) == 3
    assert model.affinity == pytest.approx(-7.991)
    assert [a.element for a in model.atoms] == ["C", "C", "O"]
    assert model.atoms[2].charge == pytest.approx(-0.3)
    lo, hi = model.bounding_box()
    assert lo[0] == pytest.approx(0.0) and hi[0] == pytest.approx(1.9)


def test_parse_pdbqt_handles_multiple_models():
    text = PDBQT + PDBQT.replace("ROOT", "MODEL 2\nROOT") + "ENDMDL\n"
    models = gui_structure.parse_pdbqt(text)
    assert len(models) == 2


def test_bond_perception_of_a_small_molecule():
    atoms = gui_structure.parse_pdbqt(PDBQT)[0].atoms
    bonds = gui_structure.guess_bonds(atoms)
    # C1-C2 and C2-O1 are the only bonds within covalent distance.
    assert set(bonds) == {(0, 1), (1, 2)}


def test_element_colours_are_distinct_for_the_common_heteroatoms():
    colors = {
        el: gui_structure.element_color(el) for el in ("C", "N", "O", "S")
    }
    assert len(set(colors.values())) == 4


def test_camera_frames_a_bounding_box():
    cam = Camera()
    cam.frame((-10, -10, -10), (10, 10, 10))
    assert cam.target == pytest.approx((0.0, 0.0, 0.0))
    assert cam.distance >= 20.0
    view = cam.view()
    assert view.shape == (4, 4)
    proj = cam.projection(800, 600)
    assert proj[1, 1] > 0 and proj[3, 2] == -1.0


def test_scene_distance_cutoff_keeps_only_nearby_atoms():
    atoms = gui_structure.parse_pdbqt(PDBQT)[0].atoms
    scene = Scene(receptor=atoms, ligand=atoms)
    assert len(scene.visible_receptor()) == 3
    # Only the atom at (1.39, 0, 0) is within 0.6 A of (1.6, 0, 0).
    scene.receptor_cutoff = (1.6, 0.0, 0.0)
    scene.receptor_radius = 0.6
    near = scene.visible_receptor()
    assert len(near) == 1 and near[0].name == "C2"


# ---------------------------------------------------------------------------
# The OpenGL path
# ---------------------------------------------------------------------------


def _render_offscreen(scene, width=320, height=320):
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no OpenGL 3.3 context available: {exc}")
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((width, height), 4)],
        depth_attachment=ctx.depth_texture((width, height)),
    )
    fbo.use()
    from odock.gui.viewport import Renderer

    renderer = Renderer(ctx, scene)
    cam = Camera()
    cam.frame(*scene.bounds())
    renderer.draw(cam, width, height)
    data = np.frombuffer(fbo.read(components=3), dtype="u1").reshape(height, width, 3)
    return renderer, data


def test_renderer_draws_the_ligand_and_the_box():
    atoms = gui_structure.parse_pdbqt(PDBQT)[0].atoms
    scene = Scene(ligand=atoms)
    scene.box = ((0.5, 0.4, 0.0), (6.0, 6.0, 6.0))
    renderer, data = _render_offscreen(scene)
    assert renderer.ligand_count == 3
    background = np.array([0.086, 0.094, 0.125]) * 255.0
    differing = int((np.abs(data.astype(float) - background).sum(axis=2) > 12).sum())
    assert differing > 200, f"only {differing} pixels were drawn"


def test_renderer_survives_an_empty_scene():
    scene = Scene()
    renderer, data = _render_offscreen(scene)
    assert renderer.ligand_count == 0
    assert data.shape == (320, 320, 3)

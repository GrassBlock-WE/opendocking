# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.interop` — the files another program has to open.

The exports are checked by *reading them back*: the PDB is parsed with the
project's own reader, the column positions are asserted against the format
specification, and the scripts are checked for the commands they promise. The
one thing a test cannot do here is run PyMOL or ChimeraX, so the commands are
pinned to the documented vocabulary of each program instead.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from odock import interop
from odock.gui.structure import Atom, parse_pdbqt


def atom(element: str, x: float, y: float, z: float, **kwargs) -> Atom:
    fields = {
        "res_name": "LIG",
        "res_id": 1,
        "chain": "A",
        "name": f"{element}1",
        "ad_type": element,
    }
    fields.update(kwargs)
    return Atom(element=element, x=x, y=y, z=z, **fields)


class Interaction:
    """The shape ``odock.analysis.Interaction`` has, without importing it."""

    def __init__(self, kind: str, a: int, b: int) -> None:
        self.kind = kind
        self.a = a
        self.b = b


class FakeSurface:
    """The duck-typed surface the exporter accepts."""

    def __init__(self, count: int = 4, mode: str = "sas", property_name: str = "hydrophobicity"):
        self.vertices = [[float(i), 0.0, 0.0] for i in range(count)]
        self.normals = [[0.0, 0.0, 1.0] for _ in range(count)]
        self.colors = [[0.2, 0.4, 0.6] for _ in range(count)]
        self.values = [i / max(1, count - 1) for i in range(count)]
        self.triangles = [[0, 1, 2], [1, 2, 3]] if count >= 4 else []
        self.mode = mode
        self.property_name = property_name


@pytest.fixture()
def state() -> interop.SceneState:
    receptor = [
        atom("C", 0.0, 0.0, 0.0, res_name="ALA", res_id=1, name="CA", serial=17),
        atom("O", 1.2, 0.3, 0.0, res_name="ALA", res_id=1, name="O", serial=18),
        atom("N", 2.4, -0.2, 0.4, res_name="ASP", res_id=2, name="N", serial=19),
    ]
    ligand = [
        atom("C", 3.4, 0.0, 0.0, res_name="LIG", res_id=1, name="C1"),
        atom("O", 4.2, 0.6, 0.0, res_name="LIG", res_id=1, name="O1"),
    ]
    return interop.SceneState(
        receptor=receptor,
        ligand=ligand,
        receptor_bonds=[(0, 1), (1, 2)],
        ligand_bonds=[(0, 1)],
        interactions=[Interaction("hbond", 1, 1), Interaction("hydrophobic", 2, 0)],
        measurements=[((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), "3.0 A")],
        box=((1.0, 1.0, 1.0), (8.0, 6.0, 4.0)),
        style_protein="cartoon",
        style_ligand="ball_stick",
        receptor_values=[0.1, 0.2, 0.9],
        ligand_values=[-1.0, 0.5],
        property_name="hydrophobicity",
        property_unit="0 polar -> 1 apolar",
        property_range=(0.0, 1.0),
        show_surface=True,
        surface_mode="ses",
        surface=FakeSurface(mode="ses"),
        camera={"target": (0.0, 0.0, 0.0), "distance": 40.0, "azimuth": 0.6, "elevation": 0.35},
        title="unit test",
        receptor_source="receptor.pdbqt",
        ligand_source="ligand.pdbqt",
    )


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------


def test_pdb_lines_are_column_correct(state):
    text = interop.pose_pdb_text(state)
    records = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert len(records) == 5
    for line in records:
        assert len(line) >= 78
        # The columns the format fixes: coordinates, occupancy, B-factor and
        # element. A shift in any of them is a structure in the wrong place.
        float(line[30:38])
        float(line[38:46])
        float(line[46:54])
        assert 0.0 <= float(line[54:60]) <= 1.0
        float(line[60:66])
        assert line[76:78].strip() in ("C", "O", "N")
        assert line[6:11].strip().isdigit()
        assert line[22:26].strip().lstrip("-").isdigit()


def test_a_one_letter_element_is_right_shifted_and_a_two_letter_one_is_not():
    """``CA`` is a carbon alpha in a protein and calcium as an ion."""
    carbon_alpha = atom("C", 0.0, 0.0, 0.0, name="CA", res_name="ALA", res_id=3)
    calcium = atom("Ca", 0.0, 0.0, 0.0, name="CA", res_name="CA", res_id=1)
    assert interop.pdb_atom_line(carbon_alpha, 1)[12:16] == " CA "
    assert interop.pdb_atom_line(calcium, 1)[12:16] == "CA  "
    assert interop.pdb_atom_line(calcium, 1)[76:78] == "Ca"


def test_serials_are_renumbered_from_one_and_the_two_structures_are_split(state):
    text = interop.pose_pdb_text(state)
    lines = text.splitlines()
    serials = [
        int(line[6:11])
        for line in lines
        if line.startswith(("ATOM", "HETATM"))
    ]
    assert serials == [1, 2, 3, 4, 5]
    kinds = [line[:6].strip() for line in lines if line.startswith(("ATOM", "HETATM"))]
    assert kinds == ["ATOM", "ATOM", "ATOM", "HETATM", "HETATM"]
    assert "TER" in lines
    assert lines[-1] == "END"


def test_the_property_lands_in_the_b_factor_column(state):
    text = interop.pose_pdb_text(state)
    records = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert [float(line[60:66]) for line in records] == pytest.approx(
        [0.1, 0.2, 0.9, -1.0, 0.5], abs=1e-9
    )
    assert "REMARK  PROPERTY  hydrophobicity" in text
    assert "REMARK  RANGE     0.0000 1.0000" in text
    assert "REMARK  SURFACE   SES" in text


def test_the_pose_pdb_round_trips_through_the_project_reader(state):
    """The strongest check available: our own parser reads it back."""
    text = interop.pose_pdb_text(state)
    models = parse_pdbqt(text)
    assert len(models) == 1
    read = models[0].atoms
    assert len(read) == 5
    expected = list(state.receptor) + list(state.ligand)
    for got, wanted in zip(read, expected):
        assert got.x == pytest.approx(wanted.x, abs=1e-3)
        assert got.y == pytest.approx(wanted.y, abs=1e-3)
        assert got.z == pytest.approx(wanted.z, abs=1e-3)
        assert got.element == wanted.element
        assert got.res_name == wanted.res_name
        assert got.res_id == wanted.res_id
        assert got.name == wanted.name


def test_a_ligand_only_export_has_no_receptor(state):
    text = interop.pose_pdb_text(state, include_receptor=False)
    assert "ATOM  " not in text
    assert text.count("HETATM") == 2


def test_an_empty_scene_still_writes_a_valid_file():
    text = interop.pose_pdb_text(interop.SceneState())
    assert text.splitlines()[-1] == "END"
    assert any("NO ATOMS" in line for line in text.splitlines())


def test_write_pose_pdb_returns_the_path(tmp_path, state):
    target = interop.write_pose_pdb(tmp_path / "pose.pdb", state)
    assert target == tmp_path / "pose.pdb"
    assert target.read_text(encoding="utf-8").startswith("REMARK")


# ---------------------------------------------------------------------------
# PyMOL
# ---------------------------------------------------------------------------


def test_the_pymol_script_names_the_files_and_hides_everything(state):
    script = interop.pymol_script(state)
    assert "load receptor.pdb, receptor" in script
    assert "load ligand.pdb, ligand" in script
    assert "hide everything" in script


@pytest.mark.parametrize("style", sorted(interop.STYLE_PROTEIN))
def test_every_protein_style_emits_its_documented_command(style):
    state = interop.SceneState(receptor=[atom("C", 0.0, 0.0, 0.0)], style_protein=style)
    script = interop.pymol_script(state)
    for command in interop.STYLE_PROTEIN[style]:
        expected = command.format(obj="receptor", scale=f"{state.receptor_scale:g}")
        assert expected in script, (style, expected)


@pytest.mark.parametrize("style", sorted(interop.STYLE_LIGAND))
def test_every_ligand_style_emits_its_documented_command(style):
    ligand = [atom("C", 0.0, 0.0, 0.0)]
    state = interop.SceneState(ligand=ligand, style_ligand=style)
    script = interop.pymol_script(state)
    for command in interop.STYLE_LIGAND[style]:
        expected = command.format(obj="ligand", scale=f"{state.ball_scale:g}")
        assert expected in script, (style, expected)


def test_the_style_tables_cover_the_viewport_styles():
    """A style added to the renderer must not silently vanish from the export."""
    viewport = pytest.importorskip("odock.gui.viewport")
    assert set(viewport.PROTEIN_STYLES) <= set(interop.STYLE_PROTEIN)
    assert set(viewport.LIGAND_STYLES) <= set(interop.STYLE_LIGAND)


def test_the_interaction_colours_match_the_renderer():
    viewport = pytest.importorskip("odock.gui.viewport")
    for kind, colour in viewport.INTERACTION_COLORS.items():
        assert interop.INTERACTION_COLORS[kind] == pytest.approx(colour)


def test_every_interaction_becomes_a_distance_object(state):
    script = interop.pymol_script(state)
    assert "distance hbond_1, receptor and index 2, ligand and index 2" in script
    assert "distance hydrophobic_2, receptor and index 3, ligand and index 1" in script
    # Coloured per type, and the dash geometry is set on the object itself.
    assert "color hbond_color, hbond_1" in script
    assert "color hydrophobic_color, hydrophobic_2" in script
    assert "set dash_radius, 0.08, hbond_1" in script


def test_an_out_of_range_interaction_is_reported_not_dropped_quietly():
    state = interop.SceneState(
        receptor=[atom("C", 0.0, 0.0, 0.0)],
        ligand=[atom("C", 3.0, 0.0, 0.0)],
        interactions=[Interaction("hbond", 99, 0)],
    )
    script = interop.pymol_script(state)
    assert "skipped: atom index out of range" in script
    assert "distance" not in script


def test_measurements_and_the_box_become_pseudoatoms(state):
    script = interop.pymol_script(state)
    assert "pseudoatom measure_1a, pos=[0.000, 0.000, 0.000]" in script
    assert "pseudoatom measure_1b, pos=[3.000, 0.000, 0.000]" in script
    assert "distance m1, measure_1a, measure_1b" in script
    assert 'label m1, "3.0 A"' in script
    for corner in range(8):
        assert f"pseudoatom box_{corner}, pos=[" in script
    assert "show dots, box_*" in script


def test_the_surface_commands_carry_the_property_range(state):
    script = interop.pymol_script(state)
    assert "show surface, receptor" in script
    assert "spectrum b, yellow_white_blue, receptor, minimum=0.0000, maximum=1.0000" in script
    assert "set transparency, 0.00, receptor" in script


def test_an_electrostatic_surface_uses_the_diverging_palette(state):
    state.show_surface = True
    state.surface_property = "electrostatic"
    script = interop.pymol_script(state)
    assert "spectrum b, bluered, receptor, minimum=0.0000, maximum=1.0000" in script


def test_the_camera_becomes_an_orthonormal_set_view(state):
    script = interop.pymol_script(state)
    line = next(line for line in script.splitlines() if line.startswith("set_view ("))
    numbers = [float(value) for value in line[len("set_view (") : -1].split(",")]
    assert len(numbers) == 18
    assert all(math.isfinite(value) for value in numbers)
    right, up, forward = numbers[0:3], numbers[3:6], numbers[6:9]
    for vector in (right, up, forward):
        assert math.sqrt(sum(component * component for component in vector)) == pytest.approx(1.0)
    for first, second in ((right, up), (right, forward), (up, forward)):
        # The matrix is written with six decimals, so orthogonality holds to
        # the rounding of the text, not to machine precision.
        assert sum(a * b for a, b in zip(first, second)) == pytest.approx(0.0, abs=1e-5)
    # The eye is placed at the camera's own orbital position (to the six
    # decimals the matrix is written with).
    eye = numbers[9:12]
    expected = interop.camera_vectors(state.camera)[3]
    assert eye == pytest.approx(list(expected), abs=1e-5)


def test_a_scene_without_a_camera_has_no_set_view():
    script = interop.pymol_script(interop.SceneState(receptor=[atom("C", 0, 0, 0)]))
    assert "set_view" not in script


def test_the_exported_camera_matches_the_viewport_convention():
    """The camera maths is duplicated on purpose; it must not drift."""
    viewport = pytest.importorskip("odock.gui.viewport")
    camera = viewport.Camera(target=(1.0, 2.0, 3.0), distance=25.0, azimuth=0.9, elevation=0.3)
    state = interop.SceneState(
        camera={
            "target": camera.target,
            "distance": camera.distance,
            "azimuth": camera.azimuth,
            "elevation": camera.elevation,
        }
    )
    right, up, forward, eye, _target, _distance = interop.camera_vectors(state.camera)
    matrix = camera.view()
    # Rows of the view matrix are the camera axes; our right/up/forward are the
    # same axes in world space, so they must match row by row.
    assert right == pytest.approx(list(matrix[0][:3]), abs=1e-5)
    assert up == pytest.approx(list(matrix[1][:3]), abs=1e-5)
    assert forward == pytest.approx([-value for value in matrix[2][:3]], abs=1e-5)
    assert eye == pytest.approx(camera.eye(), abs=1e-9)


# ---------------------------------------------------------------------------
# ChimeraX
# ---------------------------------------------------------------------------


def test_the_chimerax_script_opens_styles_and_says_the_camera_is_not_reproduced(state):
    script = interop.chimerax_script(state)
    assert "open receptor.pdb" in script
    assert "open ligand.pdb" in script
    assert "cartoon #1" in script
    assert "style stick #2" in script
    assert "no equivalent of PyMOL's set_view" in script
    assert "set_view (" not in script
    assert "zoom #2" in script


def test_every_chimerax_style_is_emitted():
    for style in interop.CHIMERAX_PROTEIN:
        state = interop.SceneState(receptor=[atom("C", 0, 0, 0)], style_protein=style)
        script = interop.chimerax_script(state)
        for command in interop.CHIMERAX_PROTEIN[style]:
            assert command.format(obj="#1", scale="0.2") in script, style
    for style in interop.CHIMERAX_LIGAND:
        state = interop.SceneState(ligand=[atom("C", 0, 0, 0)], style_ligand=style)
        script = interop.chimerax_script(state)
        for command in interop.CHIMERAX_LIGAND[style]:
            assert command.format(obj="#2", scale="0.55") in script, style


def test_chimerax_selects_by_residue_and_name(state):
    script = interop.chimerax_script(state)
    assert "distance hbond_1 #1/A:1@O #2/A:1@O1" in script
    assert "color hbond_1 #33e6f2" in script


def test_chimerax_surface_uses_bfactor_colouring(state):
    script = interop.chimerax_script(state)
    assert "surface #1" in script
    assert "transparency 0" in script
    assert "color #1 byattribute bfactor palette yellow:white:blue range 0.0000,1.0000" in script


# ---------------------------------------------------------------------------
# OBJ and the bundle
# ---------------------------------------------------------------------------


def test_surface_obj_counts_match_the_lines():
    text = interop.surface_obj_text(FakeSurface(count=4))
    lines = text.splitlines()
    assert sum(1 for line in lines if line.startswith("v ")) == 4
    assert sum(1 for line in lines if line.startswith("f ")) == 2
    assert lines[0] == "# OpenDocking surface"
    assert "f 1 2 3" in lines
    # Per-vertex colour rides along on the vertex line.
    assert len(lines[3].split()) == 7


def test_surface_obj_of_nothing_is_a_header():
    assert interop.surface_obj_text(None).startswith("# OpenDocking surface")
    assert interop.surface_obj_text(FakeSurface(count=0)).count("v ") == 0


def test_export_bundle_writes_every_promised_file(tmp_path, state):
    manifest = interop.export_bundle(tmp_path / "bundle", state)
    for key in ("receptor", "ligand", "pose", "pymol", "chimerax", "surface"):
        path = manifest[key]
        assert isinstance(path, Path)
        assert path.is_file(), key
        assert path.stat().st_size > 40, key
    assert manifest["pymol"].name == "odock.pml"
    assert manifest["chimerax"].name == "odock.cxc"
    # The scripts must point at the files that were actually written.
    pymol = manifest["pymol"].read_text(encoding="utf-8")
    assert "load odock_receptor.pdb, receptor" in pymol
    assert "load odock_ligand.pdb, ligand" in pymol
    # The surface travels as geometry with its colours.
    assert "v " in manifest["surface"].read_text(encoding="utf-8")
    # The bundle's own pose file is the receptor plus the ligand.
    assert "ATOM" in manifest["pose"].read_text(encoding="utf-8")
    assert "HETATM" in manifest["pose"].read_text(encoding="utf-8")


def test_export_bundle_honours_a_custom_stem(tmp_path, state):
    manifest = interop.export_bundle(tmp_path, state, stem="run42")
    assert manifest["pose"].name == "run42_pose.pdb"
    assert "run42_receptor.pdb" in manifest["pymol"].read_text(encoding="utf-8")


def test_export_bundle_without_a_surface_reports_none(tmp_path):
    state = interop.SceneState(receptor=[atom("C", 0, 0, 0)], ligand=[atom("C", 3, 0, 0)])
    manifest = interop.export_bundle(tmp_path, state)
    assert manifest["surface"] is None
    assert "surface" not in manifest["files"]


def test_an_empty_scene_exports_without_raising(tmp_path):
    manifest = interop.export_bundle(tmp_path, interop.SceneState())
    for key in ("receptor", "ligand", "pose", "pymol", "chimerax"):
        assert manifest[key].is_file()
    assert "hide everything" in manifest["pymol"].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the module stays importable without the GUI stack
# ---------------------------------------------------------------------------


def test_importing_interop_does_not_import_qt_or_moderngl():
    """A command-line export must not need the GUI extras.

    Checked in a fresh interpreter, because this test session has already
    imported half of them. (RDKit is not asserted: ``import odock`` pulls it in
    through the public preparation API, and ``interop`` itself never touches
    it.)
    """
    code = (
        "import sys; import odock.interop; "
        "print('Qt' in ''.join(sorted(m for m in sys.modules if m.startswith('PyQt'))), "
        "'moderngl' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
        check=True,
    )
    assert result.stdout.strip() == "False False", result.stdout


def test_the_pdb_writer_never_emits_a_non_finite_coordinate():
    bad = atom("C", float("nan"), float("inf"), 0.0)
    line = interop.pdb_atom_line(bad, 1)
    # A viewer that reads "nan" in a coordinate column fails to open the file,
    # so the writer has to be able to produce something parseable.
    assert "nan" not in line.lower()
    assert len(line) >= 78


# ---------------------------------------------------------------------------
# the round trip: the live scene -> the files -> the scene's own values
# ---------------------------------------------------------------------------
#
# The scripts are generated *from* the scene, so the test that matters is that
# they still describe that scene once they are on disk: the same interaction
# colours, the same representation, the same view matrix, and the same per-atom
# property in the B-factor column. These drive a real window offscreen and read
# the exported files back.

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

RECEPTOR_PDBQT = "\n".join(
    [
        # Two residue types on purpose: the exported property must be able to
        # vary, or the B-factor check would pass on a constant column.
        f"ATOM  {index + 1:5d}  CA  {'ALA' if index % 2 == 0 else 'LEU'} A"
        f"{1 if index % 2 == 0 else 2:4d}    "
        f"{3.0 * math.cos(index / 3.0):8.3f}{3.0 * math.sin(index / 3.0):8.3f}"
        f"{0.7 * index - 3.0:8.3f}  1.00  0.00     0.000 C"
        for index in range(24)
    ]
    + ["TER", ""]
)

LIGAND_PDBQT = "\n".join(
    [
        "REMARK  VINA RESULT:      -7.991      0.000      0.000",
        "ROOT",
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C",
        "ATOM      2  C2  LIG A   1       1.390   0.000   0.000  1.00  0.00     0.000 C",
        "ATOM      3  O1  LIG A   1       2.000   1.300   0.000  1.00  0.00    -0.300 OA",
        "ENDROOT",
        "TORSDOF 0",
        "",
    ]
)


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    pytest.importorskip("moderngl")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def live_scene(tmp_path, qapp):
    """A real workbench with a receptor, a ligand, a surface and two contacts."""
    from odock.gui import i18n
    from odock.gui import surface as surface_module
    from odock.gui.app import DockingWorkbench

    i18n.set_language("en")
    receptor_file = tmp_path / "receptor.pdbqt"
    ligand_file = tmp_path / "ligand.pdbqt"
    receptor_file.write_text(RECEPTOR_PDBQT, encoding="utf-8")
    ligand_file.write_text(LIGAND_PDBQT, encoding="utf-8")

    window = DockingWorkbench()
    window.load_receptor(str(receptor_file))
    window.load_ligand(str(ligand_file))
    surface = surface_module.build_surface(
        window.scene.receptor,
        surface_module.SurfaceSettings(
            mode="sas", spacing=0.6, property="hydrophobicity"
        ),
    )
    window.viewport.set_surface(surface)
    window.scene.interactions = [
        Interaction("hbond", 0, 0),
        Interaction("hydrophobic", 4, 1),
    ]
    window.scene.style_protein = "cartoon"
    window.scene.style_ligand = "sticks"
    window.viewport.camera.azimuth = 1.05
    window.viewport.camera.elevation = 0.22
    window.viewport.camera.distance = 31.5
    window.viewport.camera.target = (0.5, -0.5, 0.25)
    yield window
    window.close()
    window.deleteLater()
    i18n.set_language("en")


def test_the_exported_pymol_colours_are_the_scene_colours(live_scene, tmp_path):
    """Round trip: the RGB in the script is the RGB the renderer used."""
    from odock.gui.viewport import INTERACTION_COLORS

    state = live_scene._interop_state()
    manifest = interop.export_bundle(tmp_path / "out", state)
    script = manifest["pymol"].read_text(encoding="utf-8")
    assert state.interactions, "the scene has no contacts to export"
    for item in state.interactions:
        kind = item.kind
        red, green, blue, _alpha = INTERACTION_COLORS[kind]
        assert f"set_color {kind}_color, [{red:.3f}, {green:.3f}, {blue:.3f}]" in script
        assert f"color {kind}_color, {kind}_" in script
    # ChimeraX gets the same colours, as #rrggbb.
    chimerax = manifest["chimerax"].read_text(encoding="utf-8")
    for index, item in enumerate(state.interactions, start=1):
        red, green, blue, _alpha = INTERACTION_COLORS[item.kind]
        hexcode = "#%02x%02x%02x" % (
            int(round(red * 255)),
            int(round(green * 255)),
            int(round(blue * 255)),
        )
        assert f"color {item.kind}_{index} {hexcode}" in chimerax


def test_the_exported_representation_is_the_scene_representation(live_scene, tmp_path):
    state = live_scene._interop_state()
    manifest = interop.export_bundle(tmp_path / "out", state)
    script = manifest["pymol"].read_text(encoding="utf-8")
    assert state.style_protein == live_scene.scene.style_protein
    assert state.style_ligand == live_scene.scene.style_ligand
    for command in interop.STYLE_PROTEIN[state.style_protein]:
        assert command.format(obj="receptor", scale=f"{state.receptor_scale:g}") in script
    for command in interop.STYLE_LIGAND[state.style_ligand]:
        assert command.format(obj="ligand", scale=f"{state.ball_scale:g}") in script


def test_the_exported_view_matrix_is_the_live_camera(live_scene, tmp_path):
    """The 18 numbers must describe the camera the frame was drawn with."""
    state = live_scene._interop_state()
    manifest = interop.export_bundle(tmp_path / "out", state)
    script = manifest["pymol"].read_text(encoding="utf-8")
    line = next(line for line in script.splitlines() if line.startswith("set_view ("))
    numbers = [float(value) for value in line[len("set_view (") : -1].split(",")]
    right, up, forward, eye, target, _distance = interop.camera_vectors(state.camera)
    assert numbers[0:3] == pytest.approx(list(right), abs=1e-5)
    assert numbers[3:6] == pytest.approx(list(up), abs=1e-5)
    assert numbers[6:9] == pytest.approx(list(forward), abs=1e-5)
    assert numbers[9:12] == pytest.approx(list(eye), abs=1e-5)
    assert numbers[12:15] == pytest.approx(list(target), abs=1e-5)
    # ... and that camera is the widget's own, not a copy taken at load time.
    assert eye == pytest.approx(live_scene.viewport.camera.eye(), abs=1e-6)
    assert target == pytest.approx(live_scene.viewport.camera.target, abs=1e-9)


def test_the_exported_pdb_carries_the_surface_property_per_atom(live_scene, tmp_path):
    """The B-factor column is the bridge: it is what makes ``spectrum b`` work."""
    surface = live_scene.scene.surface
    assert surface is not None and surface.property_name == "hydrophobicity"
    state = live_scene._interop_state()
    assert state.receptor_values is not None
    assert len(state.receptor_values) == len(live_scene.scene.receptor)
    assert state.property_range == pytest.approx(surface.value_range)
    text = interop.pose_pdb_text(state)
    records = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    values = [float(line[60:66]) for line in records[: len(state.receptor_values)]]
    assert values == pytest.approx([float(v) for v in state.receptor_values], abs=0.005)
    assert len({round(value, 3) for value in values}) > 1     # not a constant
    assert "REMARK  PROPERTY  hydrophobicity" in text
    # And the script asks for the ramp that column drives.
    script = interop.pymol_script(state)
    assert "spectrum b, yellow_white_blue, receptor" in script

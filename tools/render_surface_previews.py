# SPDX-License-Identifier: GPL-3.0-or-later
"""Render the molecular-surface figure set into ``out/surface/``.

Run it with the project's interpreter::

    .venv\\Scripts\\python.exe tests\\..\\tools\\render_surface_previews.py

It drives the real workbench (offscreen), builds the surfaces from the bundled
3PTB demo, writes a PNG per variant with its legend and scale bar, and prints
the measured statistics (triangle count, area, build time) for each one. The
screenshots in ``docs/VISUALIZATION.md`` are exactly these files.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Headless Qt, but with the real font directory: without it every label in the
# legend and the scale bar renders as tofu boxes.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402

from PyQt6 import QtWidgets  # noqa: E402

from odock.gui import i18n  # noqa: E402
from odock.gui import surface as surface_module  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402

RECEPTOR = ROOT / "demo" / "3ptb" / "receptor.pdbqt"
LIGAND = ROOT / "demo" / "3ptb" / "ligand.pdbqt"
POSES = ROOT / "demo" / "3ptb" / "poses.pdbqt"
OUT = ROOT / "out" / "surface"


def pocket_centre(window) -> tuple:
    ligand = window.scene.ligand
    return tuple(
        sum(float(getattr(atom, axis)) for atom in ligand) / len(ligand)
        for axis in ("x", "y", "z")
    )


def build(window, label: str, **kwargs):
    """Build a surface from the loaded receptor and report what it cost."""
    settings = surface_module.SurfaceSettings(**kwargs)
    started = time.perf_counter()
    surface = surface_module.build_surface(window.scene.receptor, settings)
    wall = time.perf_counter() - started
    stats = surface.stats
    print(
        f"{label:<34} {surface.mode.upper():>3}  "
        f"{stats['atoms']:>5} atoms  {stats['grid_points']:>8} grid  "
        f"{surface.triangles_count:>7} tri  {surface.area:>8.1f} A2  "
        f"{wall:>6.2f} s"
    )
    return surface


def shoot(window, name: str, width: int = 1280, height: int = 900) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    ok = window.viewport.snapshot(path, width, height)
    print(f"{'wrote' if ok else 'FAILED':>7} {path.relative_to(ROOT)}")


def main() -> int:
    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.resize(1280, 900)
    window.show()
    app.processEvents()

    window.load_receptor(str(RECEPTOR))
    window.load_ligand(str(LIGAND))
    window.load_poses(str(POSES))
    app.processEvents()

    centre = pocket_centre(window)
    window.scene.show_receptor = True
    window.scene.show_ligand = True
    window.scene.style_protein = "cartoon"
    window.scene.style_ligand = "ball_stick"
    window.scene.surface_alpha = 1.0
    # The demo ligand is nine heavy atoms: at the default ball scale it is a
    # speck inside the pocket, and the whole point of these figures is the
    # pocket *around the ligand*.
    window.scene.ball_scale = 0.95
    window.scene.box = None

    # The bundled demo receptor has no hydrogens and its PDBQT carries the
    # Gasteiger charges the preparation wrote, so the electrostatic picture
    # uses exactly the charges the force field scored with.
    charges = [float(getattr(atom, "charge", 0.0) or 0.0) for atom in window.scene.receptor]
    print(f"receptor charges: {sum(1 for c in charges if c)} of {len(charges)} non-zero")

    # -- 1. the pocket, SES, hydrophobicity (the headline picture) ----------
    ses = build(
        window,
        "pocket SES hydrophobicity",
        mode="ses",
        spacing=0.45,
        centre=centre,
        radius=11.0,
        property="hydrophobicity",
        bonds=window.scene.receptor_bonds,
    )
    window.viewport.set_surface(ses)
    window.scene.show_surface = True
    window.viewport.frame_binding_site(radius=7.5)
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "01_pocket_ses_hydrophobicity.png")

    # -- 2. the same pocket as a solvent-accessible surface -----------------
    sas = build(
        window,
        "pocket SAS hydrophobicity",
        mode="sas",
        spacing=0.45,
        centre=centre,
        radius=11.0,
        property="hydrophobicity",
        bonds=window.scene.receptor_bonds,
    )
    window.viewport.set_surface(sas)
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "02_pocket_sas_hydrophobicity.png")

    # -- 3. the pocket coloured by the electrostatic potential --------------
    esp = build(
        window,
        "pocket SAS potential",
        mode="sas",
        spacing=0.45,
        centre=centre,
        radius=11.0,
        property="electrostatic",
    )
    window.viewport.set_surface(esp)
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "03_pocket_electrostatic.png")

    # -- 4. the pocket lining only, with the lining residues highlighted ----
    keys = set()
    for atom in window.scene.receptor:
        delta = np.array(
            [
                float(atom.x) - centre[0],
                float(atom.y) - centre[1],
                float(atom.z) - centre[2],
            ]
        )
        if float(np.dot(delta, delta)) <= 25.0:
            keys.add(
                (
                    str(getattr(atom, "chain", "") or ""),
                    int(getattr(atom, "res_id", 0) or 0),
                    str(getattr(atom, "res_name", "") or ""),
                )
            )
    lining = build(
        window,
        "pocket lining SES + highlight",
        mode="ses",
        spacing=0.4,
        centre=centre,
        radius=8.0,
        property="hydrophobicity",
        bonds=window.scene.receptor_bonds,
        highlighted_residues=keys,
    )
    window.viewport.set_surface(lining)
    window.scene.show_receptor = False
    window.scene.show_ligand = True
    window.scene.interactions = []
    window.viewport.frame_binding_site(radius=6.5)
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "04_pocket_lining_highlight.png")
    window.scene.show_receptor = True
    window.load_poses(str(POSES))
    app.processEvents()

    # -- 5. the cut plane ---------------------------------------------------
    window.viewport.set_surface(ses)
    window.scene.interactions = []
    eye = window.viewport.camera.eye()
    normal = np.array([centre[i] - eye[i] for i in range(3)])
    normal = normal / float(np.linalg.norm(normal))
    projected = [
        float(np.dot(normal, np.array([float(a.x), float(a.y), float(a.z)])))
        for a in window.scene.ligand
    ]
    middle = float(np.dot(normal, np.array(centre)))
    offset = max(0.0, middle - min(projected)) + 0.5
    window.viewport.set_surface(ses)
    window.scene.surface_clip = (
        tuple(float(v) for v in normal),
        float(-middle + offset),
    )
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "05_pocket_cut_open.png")
    window.scene.surface_clip = None

    # -- 6. the whole receptor, SAS: the surface on its own -----------------
    whole = build(
        window,
        "whole receptor SAS",
        mode="sas",
        spacing=0.0,
        property="hydrophobicity",
    )
    window.viewport.set_surface(whole)
    window.scene.surface_alpha = 1.0
    window.scene.show_receptor = False
    window.scene.show_ligand = True
    window.viewport.frame_all()
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "06_whole_receptor_surface.png")

    # -- 7. the same surface translucent over the cartoon ------------------
    window.scene.show_receptor = True
    window.scene.style_protein = "cartoon"
    window.scene.surface_alpha = 0.55
    window.viewport.refresh()
    app.processEvents()
    shoot(window, "07_whole_receptor_translucent.png")
    window.scene.surface_alpha = 1.0

    # -- 8. the ligand's own buried contact area ---------------------------
    from odock import sasa as sasa_module

    record = sasa_module.ligand_buried_contact_area(
        window.scene.ligand, window.scene.receptor
    )
    print(
        "ligand burial: {free_area:.1f} A2 free, {complex_area:.1f} A2 bound, "
        "{buried_area:.1f} A2 buried ({share:.1f} %)".format(
            share=100.0 * record["buried_fraction"], **record
        )
    )
    report = sasa_module.burial(
        window.scene.receptor, window.scene.ligand, reference="unbound"
    )
    print(report.table(8))
    print(report.as_dict()["buried_fraction"], "of the receptor surface by the ligand")

    # -- 8. the buried contact area of every pose, as a chart --------------
    affinities = [getattr(model, "affinity", None) for model in window.pose_models]
    per_pose = []
    for index, model in enumerate(window.pose_models):
        record = sasa_module.ligand_buried_contact_area(
            model.atoms, window.scene.receptor
        )
        record["pose"] = float(index)
        if affinities[index] is not None:
            record["affinity"] = float(affinities[index])
        per_pose.append(record)
    svg = sasa_module.pose_burial_svg(
        per_pose, title="buried contact area per pose (3PTB, benzamidine)"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "08_burial_per_pose.svg").write_text(svg, encoding="utf-8")
    from PyQt6 import QtGui, QtSvg

    renderer = QtSvg.QSvgRenderer()
    renderer.load(svg.encode("utf-8"))
    image = QtGui.QImage(760, 380, QtGui.QImage.Format.Format_RGB888)
    image.fill(0xFFFFFFFF)
    painter = QtGui.QPainter(image)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    image.save(str(OUT / "08_burial_per_pose.png"))
    print(f"wrote {OUT.relative_to(ROOT)}\\08_burial_per_pose.png")
    for record in per_pose:
        print(
            "  pose {pose:.0f}  {score}  {area:7.1f} A2 buried  ({share:.1f} %)".format(
                pose=record["pose"],
                score="   —" if "affinity" not in record else f"{record['affinity']:6.2f}",
                area=record["buried_area"],
                share=100.0 * record["buried_fraction"],
            )
        )

    window.close()
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

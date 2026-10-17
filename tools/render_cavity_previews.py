# SPDX-License-Identifier: GPL-3.0-or-later
"""Screenshots of the cavity painting, in both themes, with the counts.

Renders the workbench's own state: receptor with a surface, painted by cavity
class, with the bottleneck spheres as the analysis measured them. Writes one
PNG per theme and prints the per-class vertex counts and the bottleneck radii, so
the images and the numbers in the report come from the same run.

    .venv\\Scripts\\python.exe -X utf8 tools\\render_cavity_previews.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from _tool import use_utf8_streams  # noqa: E402

use_utf8_streams()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402
from PyQt6 import QtGui, QtWidgets  # noqa: E402

from odock import cavity  # noqa: E402
from odock.gui import dashboard, i18n  # noqa: E402
from odock.gui import surface as surface_module  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui.structure import read_pdbqt  # noqa: E402

OUT = ROOT / "out" / "cavities"
WIDTH, HEIGHT = 1000, 720
SYSTEMS = (
    ("3ptb", "trypsin + benzamidine"),
    ("1m17", "EGFR + erlotinib"),
)


def frame(window) -> np.ndarray:
    data = window.viewport.renderer.render_image(
        window.viewport.camera, WIDTH, HEIGHT, background=window.viewport.background
    )
    return np.frombuffer(data, dtype="u1").reshape(HEIGHT, WIDTH, 3).astype(int)


def main() -> int:
    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    OUT.mkdir(parents=True, exist_ok=True)
    window = DockingWorkbench()
    window.resize(WIDTH + 220, HEIGHT + 60)
    window.show()
    app.processEvents()
    window.viewport._ensure_context()
    for directory, label in SYSTEMS:
        receptor_path = ROOT / "demo" / "systems" / directory / "receptor.pdbqt"
        if not receptor_path.is_file():
            print(f"{label}: not bundled")
            continue
        window.load_receptor(str(receptor_path))
        ligand_path = ROOT / "demo" / "systems" / directory / "ligand.pdbqt"
        if ligand_path.is_file():
            window.load_ligand(str(ligand_path))
        receptor = list(window.scene.receptor)
        # Build the surface directly at a coarse spacing: the picture is about the
        # *classification*, and a 1.0 Å SAS keeps the run to seconds. The counts
        # and the bottleneck radii do not depend on it (the analysis grid is its
        # own setting), which is why they are printed next to the image.
        surface = surface_module.build_surface(
            receptor, surface_module.SurfaceSettings(mode="sas", spacing=1.0)
        )
        window.viewport.set_surface(surface)
        window.scene.show_surface = True

        started = time.perf_counter()
        report = cavity.analyse_cavities(
            receptor, cavity.CavitySettings(spacing=0.7)
        )
        elapsed = time.perf_counter() - started
        counts = surface_module.paint_by_cavity_class(surface, report)
        window.scene.bottlenecks = surface_module.cavity_bottleneck_annotations(report)
        window.viewport.set_surface(surface)
        window.viewport.frame_all()
        window.scene.box = None
        window.viewport.refresh()
        app.processEvents()

        print(f"=== {label} ({len(receptor)} atoms) ===")
        print(
            f"  {len(report.pockets)} pockets: "
            + ", ".join(
                f"{name} {len(report.of_kind(name))}"
                for name in ("enclosed", "open", "shallow")
            )
            + f", channel {sum(1 for p in report.pockets if p.channel)}"
        )
        for pocket in report.pockets[:4]:
            neck = (
                "  —  "
                if pocket.bottleneck_radius is None
                else f"{pocket.bottleneck_radius:.2f} Å"
            )
            print(
                f"    {pocket.label} {pocket.kind:<8} {pocket.volume:7.1f} Å³  "
                f"neck {neck:<8} score {pocket.geometric_score:.2f}  "
                f"lining {', '.join(pocket.lining_residues[:4])}"
            )
        print(f"  painted vertices: {counts}")
        print(f"  analysis {elapsed:.2f} s, {report.grid_points} grid points")
        print(
            "  markers: "
            + ", ".join(
                f"{item['label']} r={item['radius']:.2f}"
                for item in window.scene.bottlenecks[:4]
            )
        )
        for theme_name in ("dark", "light"):
            window.color_theme = dashboard.theme_named(theme_name)
            window._apply_style()
            window.viewport.set_theme(window.color_theme)
            app.processEvents()
            image = frame(window)
            picture = QtGui.QImage(
                image.astype("u1").tobytes(), WIDTH, HEIGHT, WIDTH * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            painter = QtGui.QPainter(picture)
            try:
                window.viewport._paint_overlay(painter)
            finally:
                painter.end()
            path = OUT / f"cavities_{directory}_{theme_name}.png"
            picture.save(str(path))
            print(f"  wrote {path.relative_to(ROOT)}")
        print()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

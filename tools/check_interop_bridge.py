# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual end-to-end check of the interop bridge (not part of the pytest suite).

Builds a pocket surface in the real workbench, exports the bundle through the
same code path the File ▸ Export menu uses, and prints the manifest. Run it
with the project interpreter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PyQt6 import QtWidgets  # noqa: E402

from odock import interop  # noqa: E402
from odock.gui import i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui import surface as surface_module  # noqa: E402

OUT = ROOT / "out" / "surface" / "interop"


def main() -> int:
    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.load_receptor(str(ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt"))
    window.load_ligand(str(ROOT / "demo" / "systems" / "3ptb" / "ligand.pdbqt"))
    window.load_poses(str(ROOT / "demo" / "systems" / "3ptb" / "poses.pdbqt"))
    app.processEvents()

    centre = window._surface_centre()
    result = surface_module.build_surface(
        window.scene.receptor,
        surface_module.SurfaceSettings(
            mode="ses", spacing=0.5, centre=centre, radius=10.0,
            property="hydrophobicity", bonds=window.scene.receptor_bonds,
        ),
    )
    window.viewport.set_surface(result)
    app.processEvents()

    state = window._interop_state()
    print("scene state: receptor", len(state.receptor), "ligand", len(state.ligand))
    print("interactions", len(state.interactions), "measurements", len(state.measurements))
    print("surface", state.show_surface, state.surface_mode, state.surface_property)
    print("range", state.property_range)
    print("receptor values", None if state.receptor_values is None else len(state.receptor_values))
    if state.receptor_values:
        spread = max(state.receptor_values) - min(state.receptor_values)
        print("value spread", round(spread, 4), "distinct",
              len({round(v, 6) for v in state.receptor_values}))
    print("camera", state.camera)

    manifest = interop.export_bundle(OUT, state)
    for key, path in manifest["files"].items():
        print(f"  {key:<9} {path.name:<26} {path.stat().st_size:>8} bytes")
    pml = manifest["pymol"].read_text(encoding="utf-8")
    cxc = manifest["chimerax"].read_text(encoding="utf-8")
    print("pml: distances", pml.count("distance "), "set_view", "set_view (" in pml,
          "spectrum", "spectrum b" in pml)
    print("cxc: distances", cxc.count("distance "), "byattribute",
          "byattribute bfactor" in cxc)
    for line in pml.splitlines():
        if line.startswith(("set_view", "spectrum", "show surface", "load")):
            print("   ", line[:110])
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

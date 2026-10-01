# SPDX-License-Identifier: GPL-3.0-or-later
"""Is a pocket-lining surface open or closed? Measure, then decide.

Also reports the volume of the closed form and the cap area a rim needs, which
is what `docs/VISUALIZATION.md` quotes for the pocket-lining mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from odock.gui import surface as surface_module  # noqa: E402
from odock.gui.structure import read_pdbqt  # noqa: E402


def main() -> int:
    receptor = read_pdbqt(ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt")[0].atoms
    ligand = read_pdbqt(ROOT / "demo" / "systems" / "3ptb" / "ligand.pdbqt")[0].atoms
    centre = tuple(
        float(value)
        for value in np.asarray([[a.x, a.y, a.z] for a in ligand]).mean(axis=0)
    )
    print(f"binding site centre {tuple(round(v, 2) for v in centre)}")
    print()
    header = (
        f"{'build':<26}{'atoms':>6}{'tri':>9}{'area A2':>10}{'loops':>7}"
        f"{'cap tri':>9}{'cap A2':>9}{'closed':>8}{'volume A3':>11}{'s':>7}"
    )
    print(header)
    print("-" * len(header))

    cases = [
        ("full receptor SAS", dict(mode="sas")),
        ("pocket lining SAS r=11", dict(mode="sas", centre=centre, radius=11.0)),
        ("pocket lining SAS r=8", dict(mode="sas", centre=centre, radius=8.0)),
        ("pocket lining SES r=8", dict(mode="ses", centre=centre, radius=8.0)),
    ]
    for label, options in cases:
        result = surface_module.build_surface(
            receptor, surface_module.SurfaceSettings(spacing=0.65, **options)
        )
        loops = surface_module.mesh_boundary_loops(result.vertices, result.triangles)
        capped_vertices, capped_triangles, cap_triangles = (
            surface_module.cap_open_mesh(result.vertices, result.triangles)
        )
        cap_area = surface_module.surface_area(
            capped_vertices[result.vertices_count :], capped_triangles[-cap_triangles:]
        ) if cap_triangles else 0.0
        closed = not surface_module.mesh_boundary_loops(
            capped_vertices, capped_triangles
        )
        volume = surface_module.surface_volume(capped_vertices, capped_triangles)
        print(
            f"{label:<26}{result.stats['atoms']:>6}{result.triangles_count:>9}"
            f"{result.area:>10.1f}{len(loops):>7}{cap_triangles:>9}{cap_area:>9.1f}"
            f"{str(closed):>8}{volume:>11.1f}{result.stats['seconds']:>7.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

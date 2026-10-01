# SPDX-License-Identifier: GPL-3.0-or-later
"""The SAS/SES convergence table behind the documented default spacing.

The surface mesh is a triangulated isosurface, so its area sits *under* the
analytic Shrake–Rupley area and closes as the grid is refined. That gap is the
honest cost of the representation, and the default spacing should be chosen from
this table rather than from convenience: the table prints, per setting, the grid
it allocates, the triangles it produces, the mesh area, the error against the
analytic integral, and the wall time.

Run it with the project interpreter:

    .venv\\Scripts\\python.exe -X utf8 tools\\surface_convergence.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from odock import sasa as sasa_module  # noqa: E402
from odock.gui import surface as surface_module  # noqa: E402
from odock.gui.structure import read_pdbqt  # noqa: E402

RECEPTOR = ROOT / "demo" / "3ptb" / "receptor.pdbqt"
SPACINGS = (1.20, 1.00, 0.80, 0.65, 0.50, 0.40)
MODES = ("sas", "ses")


def main() -> int:
    atoms = read_pdbqt(RECEPTOR)[0].atoms
    analytic = float(sasa_module.sasa_of_atoms(atoms).sum())
    print(f"receptor: {len(atoms)} atoms from {RECEPTOR.name}")
    print(f"analytic SASA (Shrake-Rupley, 92 points, 1.4 A probe): {analytic:.1f} A2")
    print()

    header = (
        f"{'mode':>4} {'spacing':>8} {'grid points':>12} {'triangles':>10} "
        f"{'area A2':>9} {'vs SR %':>8} {'seconds':>8}"
    )
    print(header)
    print("-" * len(header))
    for mode in MODES:
        for spacing in SPACINGS:
            started = time.perf_counter()
            result = surface_module.build_surface(
                atoms, surface_module.SurfaceSettings(mode=mode, spacing=spacing)
            )
            elapsed = time.perf_counter() - started
            stats = result.stats
            area = float(stats["area"])
            # For the SAS the reference is the analytic Shrake-Rupley integral,
            # so the column is the *error* of the mesh. The SES has no analytic
            # reference in this project (MSMS is not available offline), so the
            # column is the SES/SAS *ratio*, which is the physically meaningful
            # number and which the table shows converging.
            ratio = 100.0 * area / analytic
            print(
                f"{mode.upper():>4} {spacing:>8.2f} {int(stats['grid_points']):>12} "
                f"{int(stats['triangles']):>10} {area:>9.1f} {ratio:>8.2f} "
                f"{elapsed:>8.2f}"
            )
            del result
        print()

    # The automatic spacing: what the builder picks from the point budget.
    started = time.perf_counter()
    automatic = surface_module.build_surface(
        atoms, surface_module.SurfaceSettings(mode="sas", spacing=0.0)
    )
    elapsed = time.perf_counter() - started
    print(
        f"automatic (spacing 0, max_points {surface_module.DEFAULT_MAX_POINTS}): "
        f"{automatic.spacing:.2f} A, {automatic.stats['grid_points']} points, "
        f"{automatic.triangles_count} triangles, {automatic.area:.1f} A2 "
        f"({100.0 * (automatic.area / analytic - 1.0):+.2f} %), {elapsed:.2f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

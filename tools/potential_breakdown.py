# SPDX-License-Identifier: GPL-3.0-or-later
"""Which residues make the 3PTB S1 pocket electronegative?

Prints the potential at the pocket surface under both dielectric models and the
per-residue decomposition of it, which is the number the colour map cannot show.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from odock.gui import surface as surface_module  # noqa: E402
from odock.gui.structure import read_pdbqt  # noqa: E402


def main() -> int:
    receptor = read_pdbqt(ROOT / "demo" / "3ptb" / "receptor.pdbqt")[0].atoms
    ligand = read_pdbqt(ROOT / "demo" / "3ptb" / "ligand.pdbqt")[0].atoms
    centre = tuple(
        float(value)
        for value in np.asarray([[a.x, a.y, a.z] for a in ligand]).mean(axis=0)
    )
    print(f"{len(receptor)} receptor atoms, site centre {tuple(round(v, 2) for v in centre)}")
    print()

    for dielectric in ("distance", "uniform"):
        result = surface_module.build_surface(
            receptor,
            surface_module.SurfaceSettings(
                mode="sas",
                spacing=0.65,
                centre=centre,
                radius=9.0,
                property="electrostatic",
                dielectric=dielectric,
            ),
        )
        low, high = result.value_range
        print(
            f"=== dielectric={dielectric} (eps="
            f"{'4r' if dielectric == 'distance' else '4'}) ==="
        )
        print(
            f"  surface {result.triangles_count} triangles, "
            f"legend range {low:+.2f} / {high:+.2f} kcal/(mol.e), "
            f"values {result.values.min():+.1f} .. {result.values.max():+.1f}"
        )
        started = time.perf_counter()
        decomposition = surface_module.electrostatic_decomposition(
            result.vertices,
            receptor,
            group="residue",
            dielectric=dielectric,
            top=10,
            focus=centre,
        )
        elapsed = time.perf_counter() - started
        print(
            f"  {decomposition['points']} of {decomposition['points_available']} "
            f"surface points sampled, mean potential "
            f"{decomposition['total_mean']:+.2f} kcal/(mol.e) ({elapsed:.2f} s)"
        )
        print(
            f"  charges: {decomposition['charges']} of {decomposition['atoms']} atoms "
            f"non-zero, sum {decomposition['total_charge']:+.1f} e"
        )
        print(
            f"  at the site centre: {decomposition['focus_total']:+.2f} kcal/(mol.e)"
        )
        print(
            f"  {'residue':<12}{'atoms':>6}{'charge':>8}{'at site':>10}"
            f"{'share':>8}{'mean':>9}{'extreme':>10}"
        )
        for group in decomposition["groups"]:
            share = (
                group.at_focus / decomposition["focus_total"] * 100.0
                if decomposition["focus_total"]
                else 0.0
            )
            print(
                f"  {group.label:<12}{group.atoms:>6}{group.charge:>+8.2f}"
                f"{group.at_focus:>+10.2f}{share:>+7.1f}%"
                f"{group.mean:>+9.2f}{group.extreme:>+10.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: GPL-3.0-or-later
"""Cavities and channels in the bundled systems, measured.

Prints, per bundled receptor: the classification (enclosed / open / shallow),
the volume, the aperture (openings, bottleneck radius, bottleneck residues), the
lining, the geometric heuristic, and the wall time. Also reports how the answer
moves with the two settings that matter (grid spacing and the scan radius), so
the defaults are a measurement rather than a taste.

    .venv\\Scripts\\python.exe -X utf8 tools\\cavity_report.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from odock import cavity  # noqa: E402
from odock.gui.structure import read_pdbqt  # noqa: E402

SYSTEMS = (
    ("3ptb", "3ptb", "trypsin S1 (benzamidine)"),
    ("egfr", "egfr", "EGFR kinase (erlotinib)"),
)


def load(name: str):
    receptor = read_pdbqt(ROOT / "demo" / name / "receptor.pdbqt")[0].atoms
    ligand_path = ROOT / "demo" / name / "ligand.pdbqt"
    ligand = read_pdbqt(ligand_path)[0].atoms if ligand_path.is_file() else []
    return receptor, ligand


def report(name: str, label: str, receptor, ligand) -> None:
    started = time.perf_counter()
    result = cavity.analyse_cavities(receptor)
    elapsed = time.perf_counter() - started
    print(f"=== {label} ({len(receptor)} atoms) ===")
    print(
        f"  grid {result.grid} = {result.grid_points} points at {result.spacing} A, "
        f"margin {result.margin:.1f} A, bulk faces {result.bulk_faces}, "
        f"box too tight: {result.box_too_tight}"
    )
    print(
        f"  accessible {result.accessible_volume:.0f} A3, bulk {result.bulk_volume:.0f} A3, "
        f"enclosure cells {result.enclosing_cells}, "
        f"pockets {len(result.pockets)} "
        f"(enclosed {len(result.of_kind('enclosed'))}, open {len(result.of_kind('open'))}, "
        f"shallow {len(result.of_kind('shallow'))})"
    )
    print(f"  total {elapsed:.2f} s (aperture search {result.aperture_seconds:.2f} s)")
    print(result.table(10))
    if ligand:
        centre = np.asarray([[a.x, a.y, a.z] for a in ligand]).mean(axis=0)
        ranked = sorted(
            result.pockets,
            key=lambda p: float(np.linalg.norm(np.asarray(p.centre) - centre)),
        )
        best = ranked[0] if ranked else None
        if best is not None:
            distance = float(np.linalg.norm(np.asarray(best.centre) - centre))
            print(
                f"  nearest pocket to the ligand: {best.label} {best.kind} at "
                f"{distance:.1f} A from the ligand centroid, "
                f"{best.volume:.1f} A3, neck "
                f"{'—' if best.bottleneck_radius is None else format(best.bottleneck_radius, '.2f')} A, "
                f"openings {best.openings}, score {best.geometric_score:.2f}"
            )
            print(f"    lining: {', '.join(best.lining_residues[:8])}")
            print(f"    bottleneck: {', '.join(best.bottleneck_residues[:8]) or '—'}")
    print()


def sensitivity(receptor) -> None:
    print("=== sensitivity (3PTB) ===")
    print(f"  {'spacing':>8} {'scan A':>7} {'pockets':>8} {'enclosed':>9} {'open':>5} "
          f"{'shallow':>8} {'cells':>9} {'seconds':>8}")
    for spacing in (0.8, 0.6, 0.5):
        for radius in (8.0, 10.0, 12.0):
            started = time.perf_counter()
            result = cavity.analyse_cavities(
                receptor,
                cavity.CavitySettings(spacing=spacing, enclosure_radius=radius),
            )
            elapsed = time.perf_counter() - started
            print(
                f"  {spacing:>8.2f} {radius:>7.1f} {len(result.pockets):>8} "
                f"{len(result.of_kind('enclosed')):>9} {len(result.of_kind('open')):>5} "
                f"{len(result.of_kind('shallow')):>8} {result.enclosing_cells:>9} "
                f"{elapsed:>8.2f}"
            )
    print()


def main() -> int:
    first = None
    for directory, name, label in SYSTEMS:
        path = ROOT / "demo" / directory / "receptor.pdbqt"
        if not path.is_file():
            print(f"=== {label}: not bundled ===")
            continue
        receptor, ligand = load(directory)
        report(name, label, receptor, ligand)
        if first is None:
            first = receptor
    if first is not None:
        sensitivity(first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

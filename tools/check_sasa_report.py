# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the workbench SASA report and print the exact text it shows.

Also prints both references side by side on the bundled 3PTB complex, so the
distinction the docs now make is visible as numbers:

    .venv\\Scripts\\python.exe -X utf8 tools\\check_sasa_report.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from _tool import use_utf8_streams  # noqa: E402

use_utf8_streams()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PyQt6 import QtWidgets  # noqa: E402

from odock import sasa  # noqa: E402
from odock.gui import i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui.structure import read_pdbqt  # noqa: E402


def main() -> int:
    receptor = read_pdbqt(ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt")[0].atoms
    ligand = read_pdbqt(ROOT / "demo" / "systems" / "3ptb" / "ligand.pdbqt")[0].atoms
    free = float(sasa.sasa_of_atoms(receptor).sum())
    as_free = sasa.burial(receptor, ligand, reference="free")
    as_unbound = sasa.burial(receptor, ligand, reference="unbound")
    interface = sasa.interface_area(receptor, ligand)
    print("3PTB, one receptor and one ligand, three routes to a number")
    print(f"  sasa_of_atoms(receptor)                       {free:11.1f} Å²  (a property of the receptor)")
    print(f"  burial(free).reference_total                  {as_free.reference_total:11.1f} Å²  = Σ isolated residues ({len(as_free.residues)})")
    print(f"  burial(free).free_structure_area             {str(as_free.free_structure_area):>12}     (never measured, so None)")
    print(f"  burial(free).buried_area                     {str(as_free.buried_area):>12}     (no comparable total)")
    print(f"  burial(free).summed_buried_area               {as_free.summed_buried_area:11.1f} Å²  Σ per-residue ΔASA")
    print(f"  burial(unbound).reference_total               {as_unbound.reference_total:11.1f} Å²  == the line above")
    print(f"  burial(unbound).buried_area                   {as_unbound.buried_area:11.3f} Å²")
    print(f"  interface_area()['buried_area']               {interface['buried_area']:11.3f} Å²  == the line above")
    print(f"  the ratio the estimator tripped over          {as_free.reference_total / free:11.2f}x")
    print()

    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.resize(900, 600)
    window.load_receptor(str(ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt"))
    window.load_ligand(str(ROOT / "demo" / "systems" / "3ptb" / "ligand.pdbqt"))

    shown = {}

    def capture(title, text, svg=None):
        shown["title"] = title
        shown["text"] = text

    def synchronous(work, done, label=None):
        done(work())

    window._show_report = capture
    window._run_background = synchronous
    window._sasa_report()
    app.processEvents()
    print("the workbench dialog, verbatim (reference = free, the default)")
    print("-" * 72)
    print(shown.get("title", "<no dialog>"))
    print(shown.get("text", "<no text>"))

    i18n.set_language("zh")
    window._run_background = synchronous
    window._sasa_report()
    app.processEvents()
    print()
    print("the same dialog in Chinese")
    print("-" * 72)
    print(shown.get("text", "<no text>"))
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
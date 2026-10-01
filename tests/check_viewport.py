"""Does the 3-D workbench actually paint the molecule?

Renders the real window off-screen — the Qt platform plugin is irrelevant here,
because the viewport owns its GL context and blits the result with ``QPainter``
— then grabs the 3-D area and counts the pixels that are not the clear colour.
Exits non-zero when the area is empty.

Usage::

    python tests/check_viewport.py                        # the bundled 3PTB demo
    python tests/check_viewport.py --headless             # no display needed
    python tests/check_viewport.py rec.pdbqt lig.pdbqt out.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

if "--headless" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.argv.remove("--headless")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from odock.gui.app import DockingWorkbench, _configure_surface_format  # noqa: E402

positional = [a for a in sys.argv[1:] if not a.startswith("-")]
RECEPTOR = positional[0] if positional else str(ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt")
LIGAND = positional[1] if len(positional) > 1 else str(ROOT / "demo" / "systems" / "3ptb" / "poses.pdbqt")
OUT = Path(positional[2]) if len(positional) > 2 else ROOT / "out" / "viewport_check.png"

BACKGROUND = np.array([22.0, 24.0, 32.0])


def to_array(image):
    """Return ``(uint8 pixels, signed copy for comparisons)``.

    The unsigned copy matters: ``QImage`` must be handed 8-bit pixels, and
    feeding it the widened ``int`` array produces a silently black PNG.
    """
    if isinstance(image, QtGui.QPixmap):
        image = image.toImage()
    image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    raw = np.frombuffer(bytes(ptr), dtype="u1").reshape(h, image.bytesPerLine())
    pixels = np.ascontiguousarray(raw[:, : w * 3].reshape(h, w, 3))
    return pixels, pixels.astype(int)


def main() -> int:
    _configure_surface_format()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = DockingWorkbench(receptor=RECEPTOR, ligand=LIGAND)
    window.resize(1000, 660)
    window.show()

    outcome = {"ok": False}

    def inspect() -> None:
        viewport = window.viewport
        renderer = viewport.renderer
        print(f"Qt platform          : {app.platformName()}")
        print(f"viewport widget      : {viewport.width()} x {viewport.height()}")
        if renderer is None:
            print(f"GL context           : UNAVAILABLE ({viewport._gl_error})")
        else:
            print(
                f"GL context           : {viewport.ctx.info.get('GL_VERSION')} | "
                f"{viewport.ctx.info.get('GL_RENDERER')}"
            )
            print(
                f"instances            : receptor={renderer.receptor_count} "
                f"ligand={renderer.ligand_count}"
            )

        pixels, area = to_array(viewport.grab())
        painted = int((np.abs(area - BACKGROUND).sum(axis=2) > 12).sum())
        total = area.shape[0] * area.shape[1]
        print(
            f"3-D area             : {area.shape[1]}x{area.shape[0]}  "
            f"painted={painted} ({100.0 * painted / total:.2f} %)"
        )
        print(
            f"mean RGB             : {pixels.reshape(-1, 3).mean(axis=0).round(1).tolist()}"
        )

        OUT.parent.mkdir(parents=True, exist_ok=True)
        QtGui.QImage(
            pixels.tobytes(),
            pixels.shape[1],
            pixels.shape[0],
            pixels.shape[1] * 3,
            QtGui.QImage.Format.Format_RGB888,
        ).save(str(OUT))
        print(f"saved                : {OUT}")

        outcome["ok"] = painted > 0.02 * total
        print()
        print("PASS: the 3-D view paints the molecule" if outcome["ok"]
              else "FAIL: the 3-D view is blank")
        app.quit()

    QtCore.QTimer.singleShot(1200, inspect)
    app.exec()
    return 0 if outcome["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

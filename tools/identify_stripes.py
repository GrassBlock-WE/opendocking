# SPDX-License-Identifier: GPL-3.0-or-later
"""Identify the pass behind a reported artefact, by ablation.

A user reported "a long blue/cyan strip that reads as a hydrogen bond at a long
distance" and "a dozen or more parallel stripes" across a 24.5 x 24.0 x 24.0 A
search box, with three interactions annotated. This script reproduces that
state as closely as the bundled demo allows and then replaces **one draw pass at
a time** with a no-op, counting the pixels that change. The pass that owns the
stripes is the one with the large count; everything else is a control.

It also prints the measured RGB of the candidate overlays next to the six
interaction colours, because "the box looks like an interaction" is a question
about two numbers.

    .venv\\Scripts\\python.exe -X utf8 tools\\identify_stripes.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

import numpy as np  # noqa: E402

from PyQt6 import QtGui, QtWidgets  # noqa: E402

from odock.gui import dashboard, i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402
from odock.gui.viewport import (  # noqa: E402
    BOX_COLOR,
    INTERACTION_COLORS,
    INTERACTION_LABELS,
)

OUT = ROOT / "out" / "interactions"
WIDTH, HEIGHT = 1100, 800
REFERENCE = np.array(dashboard.DARK.viewport_clear[:3]) * 255.0


def frame(window) -> np.ndarray:
    data = window.viewport.renderer.render_image(
        window.viewport.camera, WIDTH, HEIGHT, background=window.viewport.background
    )
    return np.frombuffer(data, dtype="u1").reshape(HEIGHT, WIDTH, 3).astype(int)


def changed(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.abs(a - b).sum(axis=2) > 12).sum())


def rgb(colour) -> tuple:
    return tuple(int(round(float(value) * 255)) for value in colour[:3])


def distance(first, second) -> float:
    return float(
        np.linalg.norm(np.asarray(first[:3], dtype=float) - np.asarray(second[:3], dtype=float))
    )


def noop(name):
    def _noop(*args, **kwargs):
        return None

    _noop.__name__ = f"noop_{name}"
    return _noop


def ablate(window, label: str, base: np.ndarray, patch, results: dict) -> None:
    """Render with ``patch`` applied to the renderer, and count the change."""
    saved = {name: getattr(window.viewport.renderer, name) for name in patch}
    try:
        for name, value in patch.items():
            setattr(window.viewport.renderer, name, value)
        without = frame(window)
    finally:
        for name, value in saved.items():
            setattr(window.viewport.renderer, name, value)
    results[label] = changed(base, without)


def main() -> int:
    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.resize(WIDTH, HEIGHT)
    window.load_receptor(str(ROOT / "demo" / "3ptb" / "receptor.pdbqt"))
    window.load_ligand(str(ROOT / "demo" / "3ptb" / "ligand.pdbqt"))
    # The contacts of the native ligand, computed through the same analysis call
    # the pose-annotation path uses — this keeps the tool independent of the
    # workbench's own pose widgets (and of any edit in flight in them).
    from odock import analysis

    window.scene.interactions = list(
        analysis.profile_interactions(
            window.scene.receptor,
            window.scene.ligand,
            **window.interaction_thresholds,
        )
    )
    # The reported state: a cartoon protein, a large search box, interactions
    # annotated, the binding site framed.
    window.scene.style_protein = "cartoon"
    window.scene.style_ligand = "ball_stick"
    window.scene.box = ((-1.9, 14.4, 16.7), (24.5, 24.0, 24.0))
    window.viewport._ensure_context()
    # Frame the *box*, the way the user's screenshot shows it: the whole cube in
    # view, protein inside it. Framing only the binding site puts the camera
    # inside the box, where the fill covers every pixel and every ablation
    # measures the box.
    window.viewport.frame_box()
    window.viewport.refresh()
    app.processEvents()

    kinds = [getattr(item, "kind", "") for item in window.scene.interactions]
    print("reproduced state")
    print(f"  protein style {window.scene.style_protein}, box "
          f"{window.scene.box[1]} centred {tuple(round(v, 2) for v in window.scene.box[0])}")
    print(f"  interactions: {len(kinds)} {kinds}")
    for kind in dict.fromkeys(kinds):
        count = kinds.count(kind)
        print(f"    {kind:<12} {INTERACTION_LABELS.get(kind, kind):<18} {count}")
    print(f"  interaction thresholds in force: "
          f"{window.interaction_thresholds}")
    if window.scene.measurements:
        print(f"  measurements: {len(window.scene.measurements)}")
    print()

    renderer = window.viewport.renderer
    base = frame(window)
    results: dict = {}
    ablate(window, "search box fill", base, {"_draw_box_fill": noop("box_fill")}, results)
    ablate(window, "search box edges", base, {"_draw_box": noop("box")}, results)
    ablate(window, "interaction dashes", base, {"_draw_interaction_tubes": noop("tubes")}, results)
    ablate(window, "interaction markers", base, {"_draw_interaction_markers": noop("markers")}, results)
    ablate(window, "protein/ligand mesh", base, {"_draw_mesh": noop("mesh")}, results)
    ablate(window, "axes", base, {"_draw_axes": noop("axes")}, results)
    ablate(window, "ghost ligand", base, {"_draw_ghost": noop("ghost")}, results)
    ablate(window, "contact lines (all)", base, {"_draw_interactions": noop("interactions")}, results)

    print(f"ablation over {WIDTH}x{HEIGHT}, threshold |dRGB| > 12")
    for label, count in sorted(results.items(), key=lambda item: -item[1]):
        print(f"  {label:<22} {count:>7} px")
    print()

    print("colours, measured (RGB 0-255)")
    box = tuple(float(value) for value in BOX_COLOR)
    print(f"  search box edges/fill (BOX_COLOR)  {tuple(int(round(v * 255)) for v in box)}")
    for kind, colour in INTERACTION_COLORS.items():
        gap = distance(box, colour)
        flag = "  <-- nearly the same colour" if gap < 0.2 else ""
        print(f"  {kind:<12} {rgb(colour)}   distance from the box colour {gap:.3f}{flag}")
    print(f"  box colour vs dark canvas   {distance(box, dashboard.DARK.viewport_clear):.3f}")
    print(f"  box colour vs light canvas  {distance(box, dashboard.LIGHT.viewport_clear):.3f}")

    # How many separate dashes one contact is broken into: the count the user
    # described as "a dozen or more parallel stripes".
    print()
    print("stripe count (segments drawn for the same contacts)")
    for label, (length, gap) in (
        ("previous dash 0.28/0.20", (0.28, 0.20)),
        (f"current    {renderer.DASH_LENGTH:.2f}/{renderer.DASH_GAP:.2f}",
         (renderer.DASH_LENGTH, renderer.DASH_GAP)),
    ):
        saved = (renderer.DASH_LENGTH, renderer.DASH_GAP)
        renderer.DASH_LENGTH, renderer.DASH_GAP = length, gap
        counted = {}
        original = renderer._draw_interaction_tubes

        def counter(segments, *args, **kwargs):
            counted["n"] = len(segments)
            return None

        renderer._draw_interaction_tubes = counter
        try:
            frame(window)
        finally:
            renderer._draw_interaction_tubes = original
            renderer.DASH_LENGTH, renderer.DASH_GAP = saved
        print(f"  {label:<26} {counted.get('n', 0):>4} segments for "
              f"{len(window.scene.interactions)} contacts "
              f"({counted.get('n', 0) / max(1, len(window.scene.interactions)):.1f} each)")

    # Save the reproduced frame for inspection, with the HUD the user sees, in
    # both themes: a colour that reads on the dark canvas can vanish on the light.
    OUT.mkdir(parents=True, exist_ok=True)
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
        path = OUT / f"stripes_reproduced_{theme_name}.png"
        picture.save(str(path))
        print(f"wrote {path.relative_to(ROOT)}")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

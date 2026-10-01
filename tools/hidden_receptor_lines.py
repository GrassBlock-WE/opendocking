# SPDX-License-Identifier: GPL-3.0-or-later
"""Abate the "lines pointing at nothing" report: the receptor hidden.

A user reported long parallel strips radiating from a small atom cluster into
empty space, with the receptor hidden, the box hidden and 27 interactions in the
tree. This script reproduces that state as closely as the demo allows and
answers the four questions with numbers:

1. are the contact lines drawn when the receptor is hidden (and are their far
   endpoints drawn at all)?
2. what is the **drawn endpoint distance** of every line, against its own
   threshold?
3. is the visible cluster the **interaction focus** pass rather than the ligand?
4. does a **clash** line behave like a contact?

The "before" state is reconstructed by making the visibility test always pass
(``Renderer._atom_is_drawn`` returns ``True``), which is exactly what the code
did before the fix, so the before/after pixel counts are measured rather than
argued.

    .venv\\Scripts\\python.exe -X utf8 tools\\hidden_receptor_lines.py
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

from odock import analysis  # noqa: E402
from odock.gui import dashboard, i18n  # noqa: E402
from odock.gui.app import DockingWorkbench  # noqa: E402

OUT = ROOT / "out" / "interactions"
WIDTH, HEIGHT = 1000, 700


def frame(window) -> np.ndarray:
    data = window.viewport.renderer.render_image(
        window.viewport.camera, WIDTH, HEIGHT, background=window.viewport.background
    )
    return np.frombuffer(data, dtype="u1").reshape(HEIGHT, WIDTH, 3).astype(int)


def changed(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.abs(a - b).sum(axis=2) > 12).sum())


def noop(*args, **kwargs):
    return None


def ablate(window, label, base, patch, results):
    renderer = window.viewport.renderer
    saved = {name: getattr(renderer, name) for name in patch}
    try:
        for name, value in patch.items():
            setattr(renderer, name, value)
        without = frame(window)
    finally:
        for name, value in saved.items():
            setattr(renderer, name, value)
    results[label] = changed(base, without)


def main() -> int:
    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.resize(WIDTH, HEIGHT)
    window.load_receptor(str(ROOT / "demo" / "3ptb" / "receptor.pdbqt"))
    window.load_ligand(str(ROOT / "demo" / "3ptb" / "ligand.pdbqt"))
    window.scene.interactions = list(
        analysis.profile_interactions(
            window.scene.receptor, window.scene.ligand, **window.interaction_thresholds
        )
    )
    # The reported state: receptor hidden, search box hidden, the contact set
    # emphasised (which is what leaves a cluster of atoms visible).
    window.scene.show_receptor = False
    window.scene.show_ligand = True
    window.scene.box = None
    window.scene.interaction_focus = list(window.scene.interactions)
    window.viewport._ensure_context()
    window.viewport.frame_binding_site(radius=9.0)
    window.viewport.refresh()
    app.processEvents()

    renderer = window.viewport.renderer
    thresholds = dict(window.interaction_thresholds)
    print("reproduced state")
    print(f"  receptor shown: {window.scene.show_receptor}, "
          f"box: {window.scene.box}, ligand shown: {window.scene.show_ligand}")
    print(f"  interactions in the scene: {len(window.scene.interactions)}")
    print(f"  thresholds: {thresholds}")
    print(f"  focus active: {bool(window.scene.interaction_focus)}, "
          f"focus receptor atoms: {renderer.focus_receptor_atoms}, "
          f"focus ligand atoms: {renderer.focus_ligand_atoms}")
    print()

    # -- 2. the drawn endpoint distance of every line ------------------------
    print("drawn endpoints vs the threshold that admitted the row")
    worst = None
    for item in window.scene.interactions:
        kind = getattr(item, "kind", "")
        a, b = int(getattr(item, "a", 0)), int(getattr(item, "b", 0))
        pa = window.scene.receptor[a]
        pb = window.scene.ligand[b]
        distance = float(
            np.linalg.norm(
                np.array([pa.x, pa.y, pa.z]) - np.array([pb.x, pb.y, pb.z])
            )
        )
        limit = thresholds.get(kind, thresholds.get("hbond"))
        drawn = renderer._atom_is_drawn("receptor", a) and renderer._atom_is_drawn(
            "ligand", b
        )
        print(
            f"  {kind:<12} receptor {a:>5} ligand {b:>2}  {distance:5.2f} A  "
            f"limit {limit!s:>5}  endpoints drawn: {drawn}"
        )
        if worst is None or distance > worst[0]:
            worst = (distance, kind, limit)
    if worst:
        print(
            f"  longest line: {worst[0]:.2f} A against its {worst[1]} limit "
            f"{worst[2]} -> {'WITHIN' if worst[0] <= float(worst[2]) else 'OVER'} the threshold"
        )
    print()

    # -- 1/3. the passes that are still drawing -----------------------------
    results = {}
    base = frame(window)
    ablate(window, "contact lines", base, {"_draw_interactions": noop}, results)
    ablate(window, "emphasis (focus pass)", base, {"_draw_focus": noop}, results)
    ablate(window, "ligand mesh (bonds)", base, {"_draw_mesh": noop}, results)
    ablate(window, "ligand spheres", base, {"ligand_count": 0}, results)
    print(f"ablation with the receptor hidden, {WIDTH}x{HEIGHT}, |dRGB| > 12")
    for label, count in sorted(results.items(), key=lambda item: -item[1]):
        print(f"  {label:<24} {count:>7} px")
    print(f"  contacts skipped by the visibility rule: {renderer.interactions_skipped}")
    print(f"  endpoint markers drawn: {renderer.marker_count}")
    print()

    # -- before/after: the same frame with the old behaviour ----------------
    fixed = frame(window)
    saved = renderer._atom_is_drawn
    renderer._atom_is_drawn = lambda *args, **kwargs: True
    try:
        before = frame(window)
    finally:
        renderer._atom_is_drawn = saved
    print("before/after, with the receptor hidden")
    print(f"  contact pixels drawn to invisible atoms: {changed(fixed, before)}")
    print(f"  contact lines drawn before / after: {len(window.scene.interactions)} / "
          f"{len(window.scene.interactions) - renderer.interactions_skipped}")

    OUT.mkdir(parents=True, exist_ok=True)
    for theme_name in ("dark", "light"):
        window.color_theme = dashboard.theme_named(theme_name)
        window._apply_style()
        window.viewport.set_theme(window.color_theme)
        app.processEvents()
        for label, image in (("after", frame(window)), ("before", None)):
            if image is None:
                saved = renderer._atom_is_drawn
                renderer._atom_is_drawn = lambda *args, **kwargs: True
                try:
                    image = frame(window)
                finally:
                    renderer._atom_is_drawn = saved
            picture = QtGui.QImage(
                image.astype("u1").tobytes(), WIDTH, HEIGHT, WIDTH * 3,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            painter = QtGui.QPainter(picture)
            try:
                window.viewport._paint_overlay(painter)
            finally:
                painter.end()
            path = OUT / f"hidden_receptor_{theme_name}_{label}.png"
            picture.save(str(path))
            print(f"wrote {path.relative_to(ROOT)}")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

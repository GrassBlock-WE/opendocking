# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure what the interaction dashes contribute to a frame, on both themes.

A user reported "many white lines after docking" that they could not identify.
The ablation below answers that with pixels rather than opinion. It renders the
live scene with a loaded pose (the view a user has after docking: cartoon
protein, ligand in the pocket, contacts annotated) and:

* replaces one draw pass at a time with a no-op and counts the pixels that
  change, which says *which* pass owns the lines — the whole interaction pass,
  the **hydrophobic** contacts, and the perceived-bond mesh as the control;
* renders the same frame with the old and the new hydrophobic swatch and counts,
  among the pixels the swatch governs, how many read as white (luminance
  >= 0.90) before and after;
* evaluates the shader's own colour law (``colour·(0.34 + 0.72·d) + d²⁴·0.28…``)
  over the visible width of a dash, which is *why* an achromatic pale swatch
  turns into a white line under the headlight.

Run with the project interpreter:

    .venv\\Scripts\\python.exe -X utf8 tools\\measure_interactions.py
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
from odock.gui.viewport import INTERACTION_COLORS  # noqa: E402

OUT = ROOT / "out" / "interactions"
WIDTH, HEIGHT = 900, 700
OLD = (0.62, 0.64, 0.68, 0.75)
NEW = INTERACTION_COLORS["hydrophobic"]
BACKGROUND = {
    "dark": np.array(dashboard.DARK.viewport_clear[:3]) * 255.0,
    "light": np.array(dashboard.LIGHT.viewport_clear[:3]) * 255.0,
}


def frame(window) -> np.ndarray:
    data = window.viewport.renderer.render_image(
        window.viewport.camera, WIDTH, HEIGHT, background=window.viewport.background
    )
    return np.frombuffer(data, dtype="u1").reshape(HEIGHT, WIDTH, 3).astype(int)


def luminance(pixels: np.ndarray) -> np.ndarray:
    """Perceptual luminance in ``[0, 1]`` for an ``(..., 3)`` uint8 image."""
    if pixels.size == 0:
        return np.zeros(pixels.shape[:-1], dtype=float)
    return (
        0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]
    ) / 255.0


def changed(a: np.ndarray, b: np.ndarray) -> int:
    return int((np.abs(a - b).sum(axis=2) > 12).sum())


def chroma(colour) -> float:
    values = [float(value) for value in colour[:3]]
    return max(values) - min(values)


def shaded_samples(swatch, steps: int = 400) -> np.ndarray:
    """The shader's own colour law across the visible width of a dash.

    ``MESH_FS`` shades a fragment as ``colour·(0.34 + 0.72·d) + spec + rim``
    with ``d = |dot(normal, light)|``, ``spec = d²⁴·0.28`` and a small rim term.
    Across the visible half of a dash tube ``d`` sweeps from 0 to 1, so sampling
    ``d`` uniformly gives the colours the dash actually takes on screen.
    """
    d = np.linspace(1.0, 0.0, int(steps))[:, None]
    spec = np.power(d, 24.0) * 0.28
    rim = np.power(1.0 - d, 3.0) * 0.12
    base = np.asarray(swatch[:3], dtype=float)[None, :]
    return np.clip(base * (0.34 + 0.72 * d) + spec + rim, 0.0, 1.0)


def white_fraction(swatch, threshold: float = 0.90) -> float:
    """Fraction of a dash's on-screen shading that reads as white."""
    return float((luminance(shaded_samples(swatch)) >= threshold).mean())


def measure(window, theme: str) -> dict:
    """Ablate one pass at a time, and compare the two hydrophobic swatches."""
    original = INTERACTION_COLORS["hydrophobic"]

    def render_with(swatch) -> np.ndarray:
        INTERACTION_COLORS["hydrophobic"] = swatch
        return frame(window)

    try:
        with_new = render_with(NEW)
        with_old = render_with(OLD)
        # -- the whole interaction pass ------------------------------------
        interactions = window.scene.interactions
        window.scene.interactions = []
        without = render_with(NEW)
        # -- the hydrophobic contacts only ---------------------------------
        kept = [item for item in interactions if getattr(item, "kind", "") != "hydrophobic"]
        window.scene.interactions = kept
        without_hydrophobic = render_with(NEW)
        window.scene.interactions = interactions
        # -- the perceived-bond mesh, as the control -----------------------
        renderer = window.viewport.renderer
        saved = renderer._draw_mesh

        def no_mesh(*args, **kwargs):
            return None

        renderer._draw_mesh = no_mesh
        try:
            without_mesh = render_with(NEW)
        finally:
            renderer._draw_mesh = saved
    finally:
        INTERACTION_COLORS["hydrophobic"] = original

    # The pixels the hydrophobic swatch governs: exactly those that differ
    # between the two renders, so the outline (unchanged by a swatch) cannot
    # confound the count.
    swatch_mask = np.abs(with_new - with_old).sum(axis=2) > 12

    def stats(image: np.ndarray) -> dict:
        pixels = image[swatch_mask] if swatch_mask.any() else np.zeros((0, 3), dtype=int)
        luma = luminance(pixels)
        return {
            "mean_rgb": tuple(int(round(v)) for v in pixels.mean(axis=0))
            if pixels.size
            else (0, 0, 0),
            "median_luma": float(np.median(luma)) if luma.size else 0.0,
            "p99_luma": float(np.percentile(luma, 99)) if luma.size else 0.0,
            "light_px": int((luma >= 0.55).sum()) if luma.size else 0,
        }

    new_stats = stats(with_new)
    old_stats = stats(with_old)
    return {
        "interaction_pass_px": changed(with_new, without),
        "hydrophobic_px": changed(with_new, without_hydrophobic),
        "mesh_pass_px": changed(with_new, without_mesh),
        "swatch_px": int(swatch_mask.sum()),
        "new": new_stats,
        "old": old_stats,
        "shaded_white_old": white_fraction(OLD),
        "shaded_white_new": white_fraction(NEW),
        "background_luma": float(luminance(BACKGROUND[theme].reshape(1, 1, 3))[0, 0]),
    }


def report(theme: str, values: dict) -> None:
    background = values["background_luma"]
    print(f"\n=== {theme} viewport (background luma {background:.3f}) ===")
    print(f"  interaction pass owns          {values['interaction_pass_px']:>6} px")
    print(f"  hydrophobic contacts own       {values['hydrophobic_px']:>6} px")
    print(f"  receptor+ligand mesh owns      {values['mesh_pass_px']:>6} px   (control)")
    print(f"  pixels the swatch governs      {values['swatch_px']:>6} px")
    print(
        f"  swatch luminance / chroma      {luminance(NEW):.3f} / {chroma(NEW):.3f} after, "
        f"{luminance(OLD):.3f} / {chroma(OLD):.3f} before"
    )
    print(
        f"  contrast to the canvas         {abs(luminance(NEW) - background):.3f} after, "
        f"{abs(luminance(OLD) - background):.3f} before"
    )
    print(
        f"  drawn pixels: light (>=0.55)   {values['new']['light_px']:>6} px after, "
        f"{values['old']['light_px']:>6} px before"
    )
    print(
        f"  drawn pixels: median / p99     {values['new']['median_luma']:.3f} / "
        f"{values['new']['p99_luma']:.3f} after, {values['old']['median_luma']:.3f} / "
        f"{values['old']['p99_luma']:.3f} before"
    )
    print(
        f"  shader law reaching white      {values['shaded_white_new']:.3f} after, "
        f"{values['shaded_white_old']:.3f} before   (luma >= 0.90)"
    )


def save(window, theme: str, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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
    path = OUT / name
    picture.save(str(path))
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> int:
    i18n.set_language("en")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.resize(WIDTH, HEIGHT)
    window.load_receptor(str(ROOT / "demo" / "3ptb" / "receptor.pdbqt"))
    window.load_ligand(str(ROOT / "demo" / "3ptb" / "ligand.pdbqt"))
    window.load_poses(str(ROOT / "demo" / "3ptb" / "poses.pdbqt"))
    # The view a user has after docking: a cartoon protein, the ligand in the
    # pocket, and the contacts annotated in front of it.
    window.scene.style_protein = "cartoon"
    window.scene.style_ligand = "ball_stick"
    window.viewport._ensure_context()
    window.viewport.frame_binding_site(radius=10.0)
    window.viewport.refresh()
    app.processEvents()
    kinds = [getattr(item, "kind", "") for item in window.scene.interactions]
    print(f"interactions: {kinds}")
    print(f"no-op ablation, {WIDTH}x{HEIGHT}, threshold |dRGB| > 12")
    print(f"hydrophobic swatch: {NEW} (was {OLD})")

    for theme_name in ("dark", "light"):
        window.color_theme = dashboard.theme_named(theme_name)
        window._apply_style()
        window.viewport.set_theme(window.color_theme)
        app.processEvents()
        report(theme_name, measure(window, theme_name))
        save(window, theme_name, f"interactions_{theme_name}.png")
        INTERACTION_COLORS["hydrophobic"] = OLD
        save(window, theme_name, f"interactions_{theme_name}_before.png")
        INTERACTION_COLORS["hydrophobic"] = NEW

    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

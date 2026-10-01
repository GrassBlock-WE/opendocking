# SPDX-License-Identifier: GPL-3.0-or-later
"""The OpenDocking 3-D workbench.

The window follows the required layout: a seven-menu bar, a workspace
tree on the left, the 3-D viewport in the middle with a floating tool strip, an
inspector with Receptor/Ligand/Grid/Engine tabs on the right, and a collapsible
bottom drawer holding the pose table, the run monitor and the log.

Design notes that are not obvious from the outside:

* **The viewport is a plain ``QWidget``.** Sharing a GL context with Qt's
  compositor made the 3-D area silently blank (Qt leaves ``glColorMask(1,0,0,0)``
  behind and never composites the framebuffer it binds). See
  :class:`ViewportWidget`.
* **Chemistry happens on RDKit molecules, drawing happens on atoms.** The
  receptor menu edits a `Mol` and rewrites the PDBQT; the viewport only ever
  sees the lightweight :class:`~odock.gui.structure.Atom` list.
* **Every optional module is imported lazily.** The workbench must start, load
  structures and dock even when the chemistry or analysis extras are missing.
* **Every visible string comes from** :mod:`odock.gui.i18n`. The language is a
  process-wide setting and labels are read while the widgets are built, so
  :meth:`DockingWorkbench.set_language` rebuilds the interface in place instead
  of trying to mutate it.
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from . import dashboard, dialogs, i18n
from .. import protocol
from .i18n import tr
from .sequence import SequenceTrack
from . import sequence as sequence_module
from .structure import Model, guess_bonds, parse_pdbqt
from .viewport import (
    INTERACTION_COLORS,
    INTERACTION_LABELS,
    LIGAND_STYLES,
    PROTEIN_STYLES,
    Camera,
    Renderer,
    Scene,
)

__all__ = ["DockingWorkbench", "ViewportWidget", "launch_gui", "main"]


# ---------------------------------------------------------------------------
# Optional pieces (other modules / extras)
# ---------------------------------------------------------------------------


def _try_import(name: str):
    """Import ``odock.<name>`` without making the workbench depend on it."""
    try:
        import importlib

        return importlib.import_module(f"odock.{name}")
    except Exception:  # pragma: no cover - depends on what has been installed
        return None


def np_array(values):
    """NumPy helper, kept out of module import time."""
    import numpy as np

    return np.asarray(values, dtype=float)


_SURFACE_FORMAT: Optional[QtGui.QSurfaceFormat] = None


def _configure_surface_format() -> QtGui.QSurfaceFormat:
    """Ask for a core-profile 3.3 surface with a depth buffer."""
    global _SURFACE_FORMAT
    if _SURFACE_FORMAT is not None:
        return _SURFACE_FORMAT
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    _SURFACE_FORMAT = fmt
    return fmt


def _release_viewport(viewport) -> None:
    """Release the GL resources a viewport owns, best effort.

    Shared by :meth:`DockingWorkbench.closeEvent` and by the language switch,
    which throws the whole viewport away and builds a new one.
    """
    try:
        viewport.stop_animation()
        if viewport.framebuffer is not None:
            viewport.framebuffer.release()
            viewport.framebuffer = None
        if viewport.renderer is not None:
            viewport.renderer.release()
            viewport.renderer = None
        if viewport.ctx is not None:
            viewport.ctx.release()
            viewport.ctx = None
    except Exception:  # pragma: no cover - best effort teardown
        pass


def _interaction_label(kind: str) -> str:
    """The HUD legend label for an interaction kind, in the current language."""
    key = f"interaction.{kind}"
    return tr(key) if key in i18n.EN else INTERACTION_LABELS.get(kind, kind)


def _clear_menu_shortcuts(menu) -> None:
    """Empty every shortcut under ``menu``.

    The window is rebuilt while the old menus are still alive (Qt destroys
    widgets at the next event-loop turn), and two live ``Ctrl+S`` actions make
    Qt report an ambiguous shortcut.
    """
    for action in menu.actions():
        submenu = action.menu()
        if submenu is not None:
            _clear_menu_shortcuts(submenu)
        else:
            action.setShortcut(QtGui.QKeySequence())


def _perceived_bonds(atoms: Sequence, kind: str) -> list:
    """Bonds for ``atoms`` from :mod:`odock.gui.bonds`, never a distance test.

    The perception module is the primary path — it is the one that knows a
    rotating methyl's hydrogens are not bonded to each other. It is imported
    lazily and every failure falls back to the historical
    :func:`odock.gui.structure.guess_bonds`, so a work in progress over there can
    never stop a structure from loading.
    """
    if not atoms:
        return []
    try:
        from .bonds import perceive_bonds

        return list(perceive_bonds(atoms, kind=kind))
    except Exception as exc:  # pragma: no cover - depends on the bonds module
        sys.stderr.write(f"bond perception unavailable ({exc}); using the fallback\n")
    try:
        return [tuple(pair) for pair in guess_bonds(atoms)]
    except Exception:  # pragma: no cover - defensive
        return []


def _bond_pairs(bonds: Sequence) -> List[Tuple[int, int]]:
    """``(i, j)`` index pairs from :class:`~odock.gui.bonds.Bond` or tuples."""
    out: List[Tuple[int, int]] = []
    for bond in bonds or ():
        if hasattr(bond, "a") and hasattr(bond, "b"):
            out.append((int(bond.a), int(bond.b)))
            continue
        try:
            out.append((int(bond[0]), int(bond[1])))
        except Exception:  # pragma: no cover - defensive
            continue
    return out


def _history_silent(function):
    """Run a method with undo recording switched off.

    Loading structures, opening a project and restoring a session all *set* the
    scene: none of them is a user edit, and none of them belongs on the undo
    stack (Ctrl+Z after opening a file must undo the last edit, not the file).
    """

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        previous = getattr(self, "_suppress_history", False)
        self._suppress_history = True
        try:
            return function(self, *args, **kwargs)
        finally:
            self._suppress_history = previous

    return wrapper


def _scrollable_panel(widget, minimum: Tuple[int, int] = (150, 110)):
    """Put ``widget`` in a scroll area so it compresses instead of blocking.

    A panel that paints its own content reports a minimum size derived from the
    font metrics and from whatever it happens to hold. Two such panels side by
    side in one dock area then *add up* to a window that can no longer be made
    narrow — which is exactly what the run monitor did when its measurement
    table was 638 px wide under a fallback font (a window minimum of 1134 px).

    Inside a scroll area the panel keeps its own size and gains scrollbars, so
    the window keeps a small floor while the panel stays readable at whatever
    size the user gives it.
    """
    area = QtWidgets.QScrollArea()
    area.setObjectName("panelArea")
    area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    area.setWidgetResizable(True)
    area.setWidget(widget)
    area.setMinimumSize(int(minimum[0]), int(minimum[1]))
    area.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Preferred,
        QtWidgets.QSizePolicy.Policy.Preferred,
    )
    return area


# ---------------------------------------------------------------------------
# The 3-D viewport widget
# ---------------------------------------------------------------------------


class ViewportWidget(QtWidgets.QWidget):
    """Draws the scene and turns mouse input into camera, box and pick actions.

    # Why this is not a `QOpenGLWidget`

    Sharing a GL context with Qt means sharing it with Qt's compositor, which
    does not restore the state this renderer depends on: on the reference
    machine Qt leaves ``glColorMask(1, 0, 0, 0)`` and its own blend function
    behind, and the framebuffer it binds during ``paintGL`` is not the one it
    composites. The result is a window whose 3-D area is silently empty.

    Instead the renderer owns a **standalone** ModernGL context and renders into
    its own framebuffer, and the resulting image is blitted into the widget with
    ``QPainter``. Nothing about the drawing depends on Qt's GL state, so the
    viewport is correct on every platform and is even testable off-screen.
    """

    atomsPicked = QtCore.pyqtSignal(list)
    bondPicked = QtCore.pyqtSignal(int, int)
    cameraChanged = QtCore.pyqtSignal()
    #: A left click (not a drag) with the orbit tool: ``(hit_or_None, additive)``
    #: where ``hit`` is ``("receptor"|"ligand", atom_index)``. An empty click
    #: carries ``None`` so the window can clear the selection.
    atomClicked = QtCore.pyqtSignal(object, bool)
    #: The atom under the cursor while hovering: ``(hit_or_None)``. Used by the
    #: status bar, which reports the atom the user is pointing at.
    atomHovered = QtCore.pyqtSignal(object)

    MODE_ORBIT = "orbit"
    MODE_MEASURE = "measure"
    MODE_BOND = "bond"

    #: Minimum seconds between two hover picks. A pick walks every atom, so
    #: without this a fast mouse over a 3 000-atom protein would spend more time
    #: picking than drawing.
    HOVER_INTERVAL = 0.04

    def __init__(self, scene: Scene, parent=None) -> None:
        super().__init__(parent)
        self.scene = scene
        self.renderer: Optional[Renderer] = None
        self.camera = Camera()
        self.ctx = None
        self.framebuffer = None
        self._size = (0, 0)
        self._last_pos = None
        self._drag_mode: Optional[str] = None
        #: Where the current press started and how far it has travelled, so a
        #: release can tell a click (select) from a drag (orbit/pan/box).
        self._press_pos = None
        self._drag_length = 0.0
        self._gl_error: Optional[str] = None
        self._render_error: Optional[str] = None
        self.mode = self.MODE_ORBIT
        self._selection: List[Tuple[str, int]] = []
        #: A framing request made before the GL context existed (see
        #: :meth:`frame_binding_site_when_ready`).
        self._pending_focus: Optional[str] = None
        self._anim_timer: Optional[QtCore.QTimer] = None
        self._anim_frames: List[List] = []
        self._anim_index = 0
        #: The colour the 3-D renderer clears to, and the flat colour QPainter
        #: fills the widget with before the image is blitted. Both belong to the
        #: theme: a light window chrome around a near-black viewport reads as a
        #: hole in the window, and vice versa.
        self.background: Tuple[float, float, float, float] = (
            dashboard.DARK.viewport_clear
        )
        self.canvas: Tuple[int, int, int] = dashboard.DARK.viewport_canvas
        #: Whether the viewport canvas is light, which decides the HUD ink.
        self._light_canvas = False
        #: The atom-under-the-cursor readout. On by default; View ▸ Inspect
        #: atom switches it off.
        self.hover_visible = True
        self._hover = None
        self._hover_at = 0.0
        #: The measurement and annotation layers, drawn with QPainter over the
        #: rendered image — so they are theme-aware, they can carry text, and
        #: they land in a snapshot as well as on screen. Each entry is a plain
        #: dict built by the window: {"kind", "points", "split", "text", ...}.
        self.measurement_overlays: List[dict] = []
        self.annotation_overlays: List[dict] = []
        #: ``(kind, detected, drawn)`` for the interaction legend.
        self.interaction_legend: List[tuple] = []
        #: Set while a high-resolution snapshot is being painted, so the hover
        #: card (a cursor affordance, not part of the figure) stays out of it.
        self._snapshotting = False
        self.setMinimumSize(420, 320)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_theme(self, theme) -> None:
        """Take the viewport colours of ``theme`` (its own clear colour)."""
        clear, canvas = dashboard.viewport_colors(theme)
        self.background = tuple(clear)
        self.canvas = tuple(canvas)
        # The HUD is drawn with QPainter over the rendered image, so its ink has
        # to follow the theme as well: light-grey legend text is invisible on a
        # light viewport.
        self._light_canvas = sum(self.canvas) > 380
        # The element palette is tuned for a dark canvas; on the light one the
        # renderer lifts those colours towards white, so a mid-grey carbon reads
        # as a stick instead of as a dark stroke. The upload is forced, or the
        # change would only appear after the next style switch.
        self.scene.light_background = self._light_canvas
        if self.renderer is not None:
            self.renderer.dirty_receptor = True
            self.renderer.dirty_ligand = True
        self.update()

    def hover_readout(self) -> Optional[Tuple[str, int]]:
        """The atom currently under the cursor, or ``None``."""
        return self._hover

    # -- measurement and annotation layers ---------------------------------

    #: World position -> widget coordinates, or ``None`` behind the camera.
    def project_point(self, point, width: Optional[int] = None, height: Optional[int] = None):
        """Project a world point to widget pixels using this widget's camera."""
        import numpy as np

        width = self.width() if width is None else width
        height = self.height() if height is None else height
        if width <= 0 or height <= 0:
            return None
        clip = (
            np.asarray(self.camera.projection(width, height), dtype=float)
            @ np.asarray(self.camera.view(), dtype=float)
            @ np.asarray([float(point[0]), float(point[1]), float(point[2]), 1.0])
        )
        if clip[3] <= 1e-6:
            return None  # behind the eye: never draw a mirrored overlay
        ndc = clip[:3] / clip[3]
        return QtCore.QPointF(
            (ndc[0] + 1.0) * 0.5 * width,
            (1.0 - ndc[1]) * 0.5 * height,
        )

    def set_measurement_overlays(self, overlays: Sequence[dict]) -> None:
        self.measurement_overlays = [dict(item) for item in overlays]
        self.update()

    def set_annotation_overlays(self, overlays: Sequence[dict]) -> None:
        self.annotation_overlays = [dict(item) for item in overlays]
        self.update()

    def set_interaction_legend(self, entries: Sequence) -> None:
        """``(kind, detected, drawn)`` per interaction kind, for the HUD legend.

        The legend counts what was *detected* (so it matches the Interactions
        table) and appends the number of lines actually drawn when the two differ
        — the honest way to show "17 contacts, 3 lines" after the drawing is
        reduced to one line per residue.
        """
        self.interaction_legend = [tuple(entry) for entry in entries]
        self.update()

    def _overlay_stroke(self, painter: QtGui.QPainter, colour, *, width=2.0, dash=False):
        """A bright stroke over a dark halo, so it reads on either theme.

        The halo is what keeps an amber arc legible on a light viewport *and*
        keeps it from being confused with the interaction dashes it crosses.
        """
        pen = QtGui.QPen(QtGui.QColor(12, 14, 20, 190), width + 2.0)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        if dash:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        return QtGui.QPen(
            QtGui.QColor(
                int(float(colour[0]) * 255),
                int(float(colour[1]) * 255),
                int(float(colour[2]) * 255),
            ),
            width,
        )

    def _paint_measurement_overlay(self, painter: QtGui.QPainter) -> None:
        """The measurement layer: lines, arcs, planes, normals and labels."""
        if not self.measurement_overlays:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QtGui.QFont("Segoe UI", 8))
        self._placed_labels = []
        for overlay in self.measurement_overlays:
            points = [
                projected
                for projected in (
                    self.project_point(point) for point in overlay.get("points", [])
                )
                if projected is not None
            ]
            if len(points) < 2:
                continue
            colour = overlay.get("colour", (1.0, 0.84, 0.28))
            kind = str(overlay.get("kind", "distance"))

            # 1. the lines that join the picked atoms
            stroke = self._overlay_stroke(painter, colour, width=2.0)
            painter.setPen(QtGui.QPen(QtGui.QColor(12, 14, 20, 190), 4.0))
            for first, second in zip(points, points[1:]):
                painter.drawLine(first, second)
            painter.setPen(stroke)
            for first, second in zip(points, points[1:]):
                painter.drawLine(first, second)

            # 2. the geometry that makes the number readable
            if kind == "angle" and len(points) >= 3:
                self._draw_arc(painter, points[1], points[0], points[2], colour)
            elif kind == "dihedral" and len(points) >= 4:
                self._draw_arc(painter, points[1], points[0], points[2], colour)
                self._draw_arc(painter, points[2], points[1], points[3], colour)
            elif kind == "plane" and len(points) >= 3:
                self._draw_plane(painter, overlay, points, colour)
            elif kind in ("plane_angle", "plane_bond") and len(points) >= 5:
                self._draw_plane(painter, overlay, points, colour)

            # 3. the endpoints, then the value
            painter.setBrush(QtGui.QColor(12, 14, 20, 200))
            painter.setPen(stroke)
            for point in points:
                painter.drawEllipse(point, 3.4, 3.4)
            self._draw_label(painter, points[len(points) // 2], str(overlay.get("text", "")), colour)
        painter.restore()

    def _draw_arc(self, painter, vertex, first, second, colour) -> None:
        """The arc of an angle: a short radius bow at the vertex."""
        import math as _math

        start = _math.degrees(
            _math.atan2(-(first.y() - vertex.y()), first.x() - vertex.x())
        )
        end = _math.degrees(
            _math.atan2(-(second.y() - vertex.y()), second.x() - vertex.x())
        )
        span = (end - start) % 360.0
        if span > 180.0:
            span -= 360.0
        radius = 26.0
        rect = QtCore.QRectF(
            vertex.x() - radius, vertex.y() - radius, radius * 2, radius * 2
        )
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(12, 14, 20, 190), 4.0))
        painter.drawArc(rect, int(start * 16), int(span * 16))
        painter.setPen(self._overlay_stroke(painter, colour, width=2.0))
        painter.drawArc(rect, int(start * 16), int(span * 16))

    def _draw_plane(self, painter, overlay, points, colour) -> None:
        """A translucent quadrilateral plus its normal, for a plane fit."""
        plane_points = points
        split = overlay.get("split")
        if split:
            plane_points = points[: int(split)]
        if len(plane_points) < 3:
            return
        # The quad is spanned by the two widest directions of the point cloud,
        # projected to the screen: it is a *drawing* of the plane, not a claim
        # about its extent, so it is deliberately generous and translucent.
        middle = QtCore.QPointF(
            sum(point.x() for point in plane_points) / len(plane_points),
            sum(point.y() for point in plane_points) / len(plane_points),
        )
        span = max(
            28.0,
            max(abs(point.x() - middle.x()) for point in plane_points) * 1.8,
            max(abs(point.y() - middle.y()) for point in plane_points) * 1.8,
        )
        quad = [
            QtCore.QPointF(middle.x() - span, middle.y() - span * 0.5),
            QtCore.QPointF(middle.x() + span, middle.y() - span * 0.5),
            QtCore.QPointF(middle.x() + span, middle.y() + span * 0.5),
            QtCore.QPointF(middle.x() - span, middle.y() + span * 0.5),
        ]
        fill = QtGui.QColor(
            int(float(colour[0]) * 255),
            int(float(colour[1]) * 255),
            int(float(colour[2]) * 255),
            46,
        )
        painter.setBrush(fill)
        painter.setPen(self._overlay_stroke(painter, colour, width=1.4, dash=True))
        painter.drawPolygon(quad)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

    def _draw_label(self, painter, anchor: QtCore.QPointF, text: str, colour) -> None:
        """A value chip next to its geometry, legible on either theme.

        Chips are stacked: measurements are often taken on the same handful of
        atoms, and two labels drawn at the same midpoint would hide each other's
        number — less readable than no overlay at all.
        """
        if not text:
            return
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 12
        height = metrics.height() + 4
        x = max(4.0, min(anchor.x() + 10, self.width() - width - 4.0))
        y = max(4.0, min(anchor.y() - height - 6, self.height() - height - 4.0))
        placed = getattr(self, "_placed_labels", None)
        if placed is None:
            placed = self._placed_labels = []
        for _attempt in range(10):
            candidate = QtCore.QRectF(x, y, width, height)
            if not any(candidate.intersects(other) for other in placed):
                break
            y += height + 2
            if y > self.height() - height - 4.0:
                y = max(4.0, anchor.y() + 10)
        rect = QtCore.QRectF(x, y, width, height)
        placed.append(rect)
        background = QtGui.QColor(252, 253, 255, 225) if self._light_canvas else QtGui.QColor(12, 16, 22, 210)
        painter.setPen(QtGui.QPen(QtGui.QColor(90, 110, 130, 200), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 4.0, 4.0)
        ink = QtGui.QColor(20, 28, 38) if self._light_canvas else QtGui.QColor(
            int(float(colour[0]) * 255),
            int(float(colour[1]) * 255),
            int(float(colour[2]) * 255),
        )
        painter.setPen(ink)
        painter.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), text)

    def _paint_annotation_overlay(self, painter: QtGui.QPainter) -> None:
        """The annotation layer: a leader line and a coloured label chip."""
        if not self.annotation_overlays:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setFont(QtGui.QFont("Segoe UI", 8))
        # One stacking pool for both layers: a label and an annotation chip must
        # not be drawn on top of each other either.
        self._placed_labels = list(getattr(self, "_placed_labels", []))
        for overlay in self.annotation_overlays:
            anchor = self.project_point(overlay.get("point", (0.0, 0.0, 0.0)))
            if anchor is None:
                continue
            colour = overlay.get("colour", (1.0, 0.85, 0.35))
            text = str(overlay.get("text", ""))
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(text) + 14
            height = metrics.height() + 6
            x = max(4.0, min(anchor.x() + 22.0, self.width() - width - 4.0))
            y = max(4.0, min(anchor.y() - height - 16.0, self.height() - height - 4.0))
            rect = QtCore.QRectF(x, y, width, height)

            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(self._overlay_stroke(painter, colour, width=1.4))
            painter.drawLine(QtCore.QPointF(anchor.x() + 3, anchor.y() - 3), QtCore.QPointF(x, y + height))

            background = QtGui.QColor(252, 253, 255, 232) if self._light_canvas else QtGui.QColor(12, 16, 22, 215)
            painter.setBrush(background)
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(
                        int(float(colour[0]) * 255),
                        int(float(colour[1]) * 255),
                        int(float(colour[2]) * 255),
                    ),
                    1.4,
                )
            )
            painter.drawRoundedRect(rect, 5.0, 5.0)
            painter.setPen(
                QtGui.QColor(20, 28, 38)
                if self._light_canvas
                else QtGui.QColor(
                    int(float(colour[0]) * 255),
                    int(float(colour[1]) * 255),
                    int(float(colour[2]) * 255),
                )
            )
            painter.drawText(rect, int(QtCore.Qt.AlignmentFlag.AlignCenter), text)
        painter.restore()


    # -- context and framebuffer -------------------------------------------

    def frame_binding_site_when_ready(self) -> None:
        """Frame the pocket now, or once the GL context has made a camera.

        The context is created on the first paint and :meth:`_ensure_context`
        frames the whole scene when it does, so a framing asked for during
        construction (``odock-gui -r -l -p``) would be silently overwritten. The
        request is remembered and re-applied right after that framing.
        """
        if self.renderer is None:
            self._pending_focus = "binding_site"
        else:
            self.frame_binding_site()

    def _ensure_context(self) -> bool:
        """Create the standalone GL context and the renderer on first use."""
        if self.renderer is not None:
            return True
        if self._gl_error is not None:
            return False
        try:
            import moderngl

            self.ctx = moderngl.create_standalone_context(require=330)
            self.renderer = Renderer(self.ctx, self.scene)
            lo, hi = self.scene.bounds()
            self.camera.frame(lo, hi)
            if self._pending_focus == "binding_site":
                self._pending_focus = None
                self.frame_binding_site()
        except Exception as exc:  # pragma: no cover - depends on the machine
            self._gl_error = str(exc)
            return False
        return True

    def _ensure_framebuffer(self, width: int, height: int) -> None:
        if self.renderer is None:
            return
        if self.framebuffer is not None and self._size == (width, height):
            return
        ctx = self.ctx
        if self.framebuffer is not None:
            self.framebuffer.release()
        self.framebuffer = ctx.framebuffer(
            color_attachments=[ctx.texture((width, height), 3)],
            depth_attachment=ctx.depth_texture((width, height)),
        )
        self._size = (width, height)

    # -- Qt painting --------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(*self.canvas))
        if not self._ensure_context():
            painter.setPen(QtGui.QColor(220, 160, 120))
            painter.drawText(
                self.rect().adjusted(12, 12, -12, -12),
                int(QtCore.Qt.TextFlag.TextWordWrap),
                tr("viewport.no_gl", detail=self._gl_error or ""),
            )
            painter.end()
            return

        ratio = self.devicePixelRatioF()
        width = max(1, int(self.width() * ratio))
        height = max(1, int(self.height() * ratio))
        self._ensure_framebuffer(width, height)
        assert self.framebuffer is not None

        self.framebuffer.use()
        try:
            self.renderer.draw(
                self.camera, width, height, background=self.background,
                target=self.framebuffer,
            )
        except Exception as exc:
            # A rendering failure must never take the whole application down:
            # PyQt turns an escaping exception in a virtual method into an
            # abort, which would lose the user's session. Report it once, keep
            # the rest of the workbench usable.
            if self._render_error is None:
                self._render_error = str(exc)
                sys.stderr.write(f"viewport render error: {exc}\n")
            painter.setPen(QtGui.QColor(220, 160, 120))
            painter.drawText(
                self.rect().adjusted(12, 40, -12, -12),
                int(QtCore.Qt.TextFlag.TextWordWrap),
                tr("viewport.render_error", detail=self._render_error or ""),
            )
            painter.end()
            return
        data = self.framebuffer.read(components=3)
        image = QtGui.QImage(
            data, width, height, width * 3, QtGui.QImage.Format.Format_RGB888
        ).copy()  # `data` is a temporary view; QImage must own its pixels
        painter.drawImage(self.rect(), image)

        self._paint_overlay(painter)
        painter.end()

    #: The QPainter HUD ink for a dark and for a light viewport. The HUD is drawn
    #: *over* the rendered image, so a style sheet cannot reach it: these are
    #: picked from ``_light_canvas``, which :meth:`set_theme` sets.
    HUD_INK = ((210, 220, 230), (24, 33, 44))
    HUD_HINT = ((255, 200, 120), (150, 82, 0))

    def _hud_ink(self) -> QtGui.QColor:
        return QtGui.QColor(*self.HUD_INK[1 if self._light_canvas else 0])

    def _hud_hint(self) -> QtGui.QColor:
        return QtGui.QColor(*self.HUD_HINT[1 if self._light_canvas else 0])

    def hud_hint_rect(self) -> QtCore.QRectF:
        """Where the tool hint is drawn: top right, right-aligned.

        It used to be drawn along the bottom edge, which is exactly where the
        floating tool strip sits — so the hint that tells you what the measure
        tool does was the one thing the tool strip hid.
        """
        return QtCore.QRectF(0.0, 8.0, max(10.0, self.width() - 12.0), 18.0)

    def _paint_overlay(self, painter: QtGui.QPainter) -> None:
        """A small HUD: the interaction legend, the surface colour bar, the tool."""
        interactions = self.scene.interactions or []
        legend = self.interaction_legend or [
            (
                kind,
                sum(1 for item in interactions if getattr(item, "kind", "") == kind),
                sum(1 for item in interactions if getattr(item, "kind", "") == kind),
            )
            for kind in dict.fromkeys(
                getattr(item, "kind", "") for item in interactions
            )
        ]
        if legend:
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            y = 14
            for kind, detected, drawn in legend:
                r, g, b, _ = INTERACTION_COLORS.get(kind, (0.8, 0.8, 0.8, 1.0))
                painter.setBrush(QtGui.QColor(int(r * 255), int(g * 255), int(b * 255)))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawEllipse(10, y - 6, 8, 8)
                painter.setPen(self._hud_ink())
                # The count is what was *detected* (it matches the table); when the
                # drawing is reduced, the number of lines is shown beside it.
                text = (
                    f"{_interaction_label(kind)} ({detected})"
                    if detected == drawn
                    else f"{_interaction_label(kind)} ({detected} → {drawn} lines)"
                )
                painter.drawText(24, y + 2, text)
                y += 16
            # What a dash *is*: the legend names each kind, and this line says
            # what the dashes join, so a line is never an anonymous stroke.
            painter.setPen(self._hud_ink().darker(115))
            painter.drawText(24, y + 2, tr("interaction.legend_convention"))
            y += 16

        # The surface colour bar and the scale bar are part of the picture, not
        # decoration: without the first a colour map cannot be read, and without
        # the second no distance in the image can be believed.
        self._paint_legend(painter, self.width(), self.height())
        self._paint_scale_bar(painter, self.width(), self.height())

        # The measurement and annotation layers sit above everything the renderer
        # draws, including the interaction dashes: that ordering, plus the dark
        # halo every stroke carries, is what keeps an arc or a plane readable
        # instead of becoming one more anonymous line.
        self._paint_measurement_overlay(painter)
        self._paint_annotation_overlay(painter)

        if self.mode != self.MODE_ORBIT:
            painter.setPen(self._hud_hint())
            painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.DemiBold))
            hint = {
                self.MODE_MEASURE: tr("viewport.hint_measure"),
                self.MODE_BOND: tr("viewport.hint_bond"),
            }.get(self.mode, self.mode)
            painter.drawText(
                self.hud_hint_rect(),
                int(QtCore.Qt.AlignmentFlag.AlignRight)
                | int(QtCore.Qt.AlignmentFlag.AlignVCenter),
                hint,
            )

        self._paint_hover(painter)

    #: The colour bar: a 14 px vertical ramp with five tick labels, drawn over
    #: the rendered image so a screenshot carries it.
    LEGEND_BAR_WIDTH = 14
    LEGEND_BAR_HEIGHT = 104

    def _paint_legend(self, painter: QtGui.QPainter, width: int, height: int) -> None:
        """The surface colour bar, its title and its tick labels.

        The stops come from the surface itself (``Scene.legend_stops``), which
        is what keeps the bar honest: it shows the range the *builder* used, not
        the range the caller thought it asked for. The labels are numbers only —
        the unit is in the title — and the whole block sits on a translucent
        panel so it stays readable over bright geometry.
        """
        stops = self.scene.legend_stops(count=5)
        if not stops or width < 120 or height < 160:
            return
        bar_width = self.LEGEND_BAR_WIDTH
        bar_height = min(self.LEGEND_BAR_HEIGHT, max(60, height // 4))
        left = 16.0
        # A charge caveat needs its own line under the title: an ESP map that may
        # be wrong-signed must not look like an answer.
        caveat = bool(getattr(self.scene.surface, "charge_warning", None))
        painter.save()
        try:
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            metrics = painter.fontMetrics()
            extra = (metrics.height() + 2) if caveat else 0
            top = height - 22.0 - bar_height - extra
            title = self._legend_title()
            title_width = min(
                max(80.0, float(metrics.horizontalAdvance(title)) + 4.0), width - 24.0
            )
            panel = QtCore.QRectF(
                left - 5.0,
                top - metrics.height() - 8.0,
                bar_width + 12.0 + max(
                    title_width,
                    max(
                        float(
                            metrics.horizontalAdvance(f"{label}   ")
                        )
                        for _fraction, _colour, label in stops
                    ),
                ),
                bar_height + metrics.height() + 16.0,
            )
            background = QtGui.QColor(10, 14, 20, 150)
            if self._light_canvas:
                background = QtGui.QColor(250, 252, 255, 190)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(panel, 5.0, 5.0)

            gradient = QtGui.QLinearGradient(
                QtCore.QPointF(left, top + bar_height), QtCore.QPointF(left, top)
            )
            for fraction, colour, _label in stops:
                gradient.setColorAt(
                    min(1.0, max(0.0, float(fraction))),
                    QtGui.QColor(
                        int(colour[0] * 255), int(colour[1] * 255), int(colour[2] * 255)
                    ),
                )
            painter.setPen(QtGui.QPen(QtGui.QColor(150, 160, 175, 200), 1))
            painter.setBrush(QtGui.QBrush(gradient))
            painter.drawRect(QtCore.QRectF(left, top, bar_width, bar_height))

            painter.setPen(self._hud_ink())
            painter.drawText(
                QtCore.QRectF(
                    left - 2.0, top - metrics.height() - 4.0, panel.width(), metrics.height() + 2
                ),
                int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                title,
            )
            if caveat:
                painter.setPen(self._hud_hint())
                painter.drawText(
                    QtCore.QRectF(
                        left - 2.0,
                        top - 4.0,
                        panel.width(),
                        metrics.height() + 2,
                    ),
                    int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                    tr("legend.charge_warning"),
                )
                painter.setPen(self._hud_ink())
            for fraction, _colour, label in stops:
                y = top + (1.0 - float(fraction)) * bar_height
                painter.drawLine(
                    QtCore.QPointF(left + bar_width, y),
                    QtCore.QPointF(left + bar_width + 4, y),
                )
                painter.drawText(
                    QtCore.QRectF(
                        left + bar_width + 7.0, y - metrics.height() / 2, 120, metrics.height() + 2
                    ),
                    int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                    label,
                )
        finally:
            painter.restore()

    def _legend_title(self) -> str:
        """``hydrophobicity (0 polar → 1 apolar)`` as one line."""
        surface = self.scene.surface
        name = str(getattr(surface, "property_name", "") or "")
        try:
            import odock.gui.surface as _surface

            label, unit = _surface.PROPERTY_LABELS.get(name, (name, ""))
        except Exception:  # pragma: no cover - defensive
            label, unit = name, ""
        return f"{label} ({unit})" if unit else label

    def _paint_scale_bar(self, painter: QtGui.QPainter, width: int, height: int) -> None:
        """A length bar in Å, derived from the very projection used to draw.

        ``u_proj[1][1] = 1/tan(fov/2)`` says how many pixels one Å at the
        camera's target distance covers, so the bar is a measurement of the
        image rather than a decoration: the same length in the model is the
        same number of pixels on screen.
        """
        if width < 160 or height < 120:
            return
        focus = float(self.camera.distance)
        tangent = math.tan(math.radians(max(1.0, float(self.camera.fov)) / 2.0))
        if tangent <= 1e-6 or focus <= 0.0:  # pragma: no cover - degenerate camera
            return
        world_per_pixel = 2.0 * focus * tangent / max(1, height)
        chosen = None
        for length in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
            pixels = length / world_per_pixel
            if pixels > 0.30 * width:
                break
            chosen = (length, pixels)
        if chosen is None:
            return
        length, pixels = chosen
        painter.save()
        try:
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            metrics = painter.fontMetrics()
            label = tr("legend.scale", length=length)
            text_width = float(metrics.horizontalAdvance(label)) + 10.0
            x0 = (width - pixels) / 2.0
            y = height - 26.0
            background = QtGui.QColor(10, 14, 20, 150)
            if self._light_canvas:
                background = QtGui.QColor(250, 252, 255, 190)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(
                QtCore.QRectF(
                    x0 - 8.0,
                    y - metrics.height() - 8.0,
                    pixels + 16.0,
                    metrics.height() + 20.0,
                ),
                4.0,
                4.0,
            )
            painter.setPen(QtGui.QPen(self._hud_ink(), 2))
            painter.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x0 + pixels, y))
            for tick in (x0, x0 + pixels):
                painter.drawLine(QtCore.QPointF(tick, y - 4), QtCore.QPointF(tick, y + 4))
            painter.setPen(self._hud_ink())
            painter.drawText(
                QtCore.QRectF(x0 - 20.0, y - metrics.height() - 5.0, pixels + 40.0, metrics.height() + 2),
                int(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            del text_width
        finally:
            painter.restore()


    def _paint_hover(self, painter: QtGui.QPainter) -> None:
        """The atom the cursor is over: one small card, bottom right.

        Hovering is how a user finds out *which* atom they are looking at
        without clicking anything and without disturbing the selection, so the
        card is read-only and carries the fields the Selection dock would show.
        """
        hit = self._hover
        if not hit or self.mode != self.MODE_ORBIT or self._snapshotting:
            return
        kind, index = hit
        atoms = self.scene.ligand if kind == "ligand" else self.scene.receptor
        try:
            atom = atoms[int(index)]
        except (IndexError, TypeError):
            return
        rows = dashboard.atom_readout(atom, kind=kind, index=int(index))
        if not rows:
            return
        painter.setFont(QtGui.QFont("Segoe UI", 8))
        metrics = painter.fontMetrics()
        width = max(
            metrics.horizontalAdvance(f"{label}: {value}") for label, value in rows
        ) + 18
        height = 10 + len(rows) * (metrics.height() + 1)
        x = max(6, self.width() - width - 10)
        y = max(6, self.height() - height - 10)
        rect = QtCore.QRectF(x, y, width, height)
        background = QtGui.QColor(12, 16, 22, 205)
        if self._light_canvas:  # a light theme needs a light card
            background = QtGui.QColor(252, 253, 255, 225)
        painter.setPen(QtGui.QPen(QtGui.QColor(90, 110, 130, 200), 1))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 5.0, 5.0)
        painter.setPen(self._hud_ink())
        line = rect.top() + 4
        for label, value in rows:
            painter.drawText(
                QtCore.QRectF(rect.left() + 8, line, width - 16, metrics.height() + 1),
                int(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter),
                f"{label}: {value}",
            )
            line += metrics.height() + 1

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        ratio = self.devicePixelRatioF()
        self._ensure_framebuffer(
            max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))
        )

    # -- interaction --------------------------------------------------------

    def _device_pos(self, event):
        ratio = self.devicePixelRatioF()
        return event.position().x() * ratio, event.position().y() * ratio

    def _device_size(self):
        ratio = self.devicePixelRatioF()
        return max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._last_pos = event.position()
        self._press_pos = event.position()
        self._drag_length = 0.0
        self.setFocus()
        if self.renderer is None:
            return
        width, height = self._device_size()
        x, y = self._device_pos(event)

        if self.mode == self.MODE_BOND and event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pick_bond(x, y)
            return
        if self.mode == self.MODE_MEASURE and event.button() == QtCore.Qt.MouseButton.LeftButton:
            hit = self.renderer.pick_atom(self.camera, width, height, x, y, ligand_first=False)
            if hit is not None:
                self._selection.append(hit)
                self.scene.highlight = [
                    index for kind, index in self._selection if kind == "ligand"
                ]
                self.refresh()
                self.atomsPicked.emit(list(self._selection))
            return

        # The search box is read-only in the 3-D view: the Grid tab's centre /
        # size / spacing fields, and the menu actions that set the box (Fit to
        # ligand, Centre on ligand, Align to residue, Detect pockets, Whole
        # protein), are the only ways to change it. A drag therefore never moves
        # or resizes it — shift is just a click modifier (add to the selection)
        # and a shift+drag orbits like any other left drag.
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_mode = "orbit"
        elif event.button() == QtCore.Qt.MouseButton.RightButton:
            self._drag_mode = "pan"

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        pressed = self._drag_mode
        travelled = self._drag_length
        button = event.button()
        modifiers = event.modifiers()
        self._drag_mode = None
        self._last_pos = None
        self._press_pos = None
        self._drag_length = 0.0
        # A click that did not turn into a drag, with the orbit tool active,
        # picks the atom under the cursor. The measure and bond tools own their
        # presses (they consumed them in mousePressEvent), and a box *resize*
        # owns "resize", so none of them fight this. Shift is bound to "drag the
        # search box", but only movement moves the box, so a shift *click* (which
        # never moved anything) is free to mean "add to the selection".
        if (
            button != QtCore.Qt.MouseButton.LeftButton
            or self.mode != self.MODE_ORBIT
            or pressed not in ("orbit", "box")
            or travelled > 4.0
            or self.renderer is None
        ):
            return
        x, y = self._device_pos(event)
        hit = self.renderer.pick_atom(self.camera, *self._device_size(), x, y)
        additive = bool(
            modifiers
            & (
                QtCore.Qt.KeyboardModifier.ShiftModifier
                | QtCore.Qt.KeyboardModifier.ControlModifier
            )
        )
        self.atomClicked.emit(hit, additive)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.renderer is None:
            return
        width, height = self._device_size()
        x, y = self._device_pos(event)
        hit = self.renderer.pick_atom(self.camera, width, height, x, y)
        if hit is not None:
            self.scene.highlight = [hit[1]] if hit[0] == "ligand" else []
            self.refresh()
            # Only the measure tool wants a pick feed: a double click in orbit
            # mode selects (the release already did) and must not silently add
            # measurement points.
            if self.mode == self.MODE_MEASURE:
                self.atomsPicked.emit([hit])

    def _pick_bond(self, x: float, y: float) -> None:
        """Toggle a ligand bond by clicking its two atoms in turn."""
        hit = self.renderer.pick_atom(
            self.camera, *self._device_size(), x, y, ligand_first=True
        )
        if hit is None or hit[0] != "ligand":
            self._selection.clear()
            return
        self._selection.append(hit)
        if len(self._selection) < 2:
            self.scene.highlight = [hit[1]]
            self.refresh()
            return
        (_, first), (_, second) = self._selection[-2], self._selection[-1]
        self._selection.clear()
        if first != second:
            self.bondPicked.emit(int(first), int(second))
            self.refresh()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_mode is None:
            self._update_hover(event)
        if self._last_pos is None or self._drag_mode is None:
            self._last_pos = event.position()
            return
        delta = event.position() - self._last_pos
        self._last_pos = event.position()
        self._drag_length += abs(delta.x()) + abs(delta.y())

        if self._drag_mode == "orbit":
            self.camera.azimuth -= delta.x() * 0.01
            self.camera.elevation = max(
                -1.5, min(1.5, self.camera.elevation + delta.y() * 0.01)
            )
        elif self._drag_mode == "pan":
            self._pan(delta)
        else:
            return
        self.cameraChanged.emit()
        self.refresh()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover is not None:
            self._hover = None
            self.atomHovered.emit(None)
            self.update()
        super().leaveEvent(event)

    def _update_hover(self, event) -> None:
        """Pick the atom under the cursor, throttled, and report changes."""
        if not self.hover_visible or self.renderer is None or self.mode != self.MODE_ORBIT:
            if self._hover is not None:
                self._hover = None
                self.atomHovered.emit(None)
                self.update()
            return
        now = time.monotonic()
        if now - self._hover_at < self.HOVER_INTERVAL:
            return
        self._hover_at = now
        try:
            hit = self.renderer.pick_atom(
                self.camera, *self._device_size(), *self._device_pos(event)
            )
        except Exception:  # pragma: no cover - a pick must never break a paint
            return
        if hit == self._hover:
            return
        self._hover = hit
        self.atomHovered.emit(hit)
        self.update()

    def set_hover_visible(self, visible: bool) -> None:
        self.hover_visible = bool(visible)
        if not self.hover_visible and self._hover is not None:
            self._hover = None
            self.atomHovered.emit(None)
        self.update()

    def _screen_axes(self):
        """Screen right/up expressed in world axes, for panning and dragging."""
        azimuth = self.camera.azimuth
        elevation = self.camera.elevation
        right = (-math.sin(azimuth), math.cos(azimuth), 0.0)
        up = (
            -math.cos(azimuth) * math.sin(elevation),
            -math.sin(azimuth) * math.sin(elevation),
            math.cos(elevation),
        )
        return right, up

    def _pan(self, delta) -> None:
        scale = self.camera.distance * 0.0016
        right, up = self._screen_axes()
        target = list(self.camera.target)
        for i in range(3):
            target[i] += (-delta.x() * right[i] + delta.y() * up[i]) * scale
        self.camera.target = tuple(target)

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self.camera.distance = max(2.0, self.camera.distance * (0.88 ** steps))
        self.cameraChanged.emit()
        self.refresh()

    # -- helpers used by the window ----------------------------------------

    def refresh(self, upload_receptor: bool = False) -> None:
        if self.renderer is None:
            self.update()
            return
        if upload_receptor:
            self.renderer.dirty_receptor = True
        self.renderer.dirty_ligand = True
        self.update()

    def frame_all(self) -> None:
        """Show the whole structure, undoing a pocket clip as well."""
        self.scene.receptor_cutoff = None
        self.scene.receptor_radius = 0.0
        self.scene.front_clip = False
        lo, hi = self.scene.bounds()
        self.camera.frame(lo, hi)
        self.refresh(upload_receptor=True)

    def frame_ligand(self) -> None:
        atoms = self.scene.ligand
        if not atoms:
            self.frame_all()
            return
        xs = [a.x for a in atoms]
        ys = [a.y for a in atoms]
        zs = [a.z for a in atoms]
        self.camera.frame(
            (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)), margin=3.0
        )
        self.refresh()

    def frame_binding_site(
        self,
        radius: float = 8.0,
        margin: float = 1.25,
        *,
        clip: bool = False,
        clip_radius: Optional[float] = None,
    ) -> None:
        """Frame the ligand together with the receptor atoms around it.

        Selecting a pose asks "where does it bind?", and framing the whole
        protein hides exactly that: the camera would sit inside the protein mass
        and show a wall of atoms. So the receptor can be clipped to the pocket
        (``scene.receptor_cutoff``/``receptor_radius``, which is what
        :meth:`Scene.visible_receptor` already implements) and the camera is
        aimed at the ligand plus every visible receptor atom within ``radius`` Å
        of it. :meth:`frame_all` undoes the clip.

        ``clip`` defaults to **False**, and that default is load-bearing: the
        cut-away plane is placed relative to the *current* ligand, so while it is
        on, two poses of the same protein are cut away differently and the user
        sees the protein's "transparency" change with the pose. Only the explicit
        ``View ▸ Frame binding site`` action asks for it.
        """
        ligand = self.scene.ligand
        if not ligand:
            self.frame_all()
            return
        # ``QAction.triggered`` emits a bool and PyQt hands it to any slot that
        # can take one, so a menu entry connected straight to this method passes
        # ``checked`` as ``radius`` — a 0 Å pocket that hides the whole
        # structure behind an empty cut-away. A bool therefore means "no
        # argument was given".
        if isinstance(radius, bool):
            radius = 8.0
        if isinstance(margin, bool):
            margin = 1.25
        points = [(float(a.x), float(a.y), float(a.z)) for a in ligand]
        centre = tuple(sum(p[axis] for p in points) / len(points) for axis in range(3))
        if clip:
            self.scene.receptor_cutoff = centre
            self.scene.receptor_radius = float(
                radius + 2.0 if clip_radius is None else clip_radius
            )
            # A ligand inside a pocket is *behind* protein material; without the
            # front clip a space-filling receptor would hide it completely.
            self.scene.front_clip = True
        low = [min(p[axis] for p in points) - radius for axis in range(3)]
        high = [max(p[axis] for p in points) + radius for axis in range(3)]
        # Only atoms inside the grown box can be within `radius` of the ligand,
        # so the distance test runs over the pocket, never the whole protein.
        for atom in self.scene.visible_receptor():
            if not (
                low[0] <= atom.x <= high[0]
                and low[1] <= atom.y <= high[1]
                and low[2] <= atom.z <= high[2]
            ):
                continue
            limit = radius * radius
            for lx, ly, lz in points[: len(ligand)]:
                dx = atom.x - lx
                dy = atom.y - ly
                dz = atom.z - lz
                if dx * dx + dy * dy + dz * dz <= limit:
                    points.append((float(atom.x), float(atom.y), float(atom.z)))
                    break
        lo = tuple(min(p[axis] for p in points) for axis in range(3))
        hi = tuple(max(p[axis] for p in points) for axis in range(3))
        self.camera.frame(lo, hi, margin=margin)
        self.refresh(upload_receptor=True)

    def frame_box(self) -> None:
        if self.scene.box is None:
            self.frame_all()
            return
        center, size = self.scene.box
        lo = tuple(center[i] - size[i] / 2 for i in range(3))
        hi = tuple(center[i] + size[i] / 2 for i in range(3))
        self.camera.frame(lo, hi, margin=1.4)
        self.refresh()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._selection.clear()
        self.refresh()

    def set_surface(self, surface) -> dict:
        """Hand the built surface to the scene (and to the renderer, if any).

        The surface lives on the scene, so the frame loop picks it up on the
        first paint even when no GL context exists yet.
        """
        self.scene.surface = surface
        if self.renderer is not None:
            return self.renderer.set_surface(surface)
        self.update()
        return dict(getattr(surface, "stats", {}) or {}) if surface is not None else {}

    def snapshot(self, path, width: int = 2400, height: int = 1600, annotate: bool = True) -> bool:
        """Render a high-resolution image straight from the GL context.

        ``annotate`` draws the colour bar and the scale bar into the file. A
        figure without its legend cannot be read, and the whole point of the
        export is a picture someone else can interpret.
        """
        if not self._ensure_context() or self.renderer is None:
            return False
        data = self.renderer.render_image(
            self.camera, int(width), int(height), background=self.background
        )
        image = QtGui.QImage(
            data, int(width), int(height), int(width) * 3,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        if annotate:
            painter = QtGui.QPainter(image)
            self._snapshotting = True
            try:
                # The measurement and annotation layers are in *widget*
                # coordinates, so they are drawn under a scale that maps those
                # onto the figure: a saved picture carries the same arcs, planes
                # and labels the screen shows.
                sx = image.width() / max(1, self.width())
                sy = image.height() / max(1, self.height())
                painter.save()
                painter.scale(sx, sy)
                self._paint_measurement_overlay(painter)
                self._paint_annotation_overlay(painter)
                painter.restore()
                # A 2400 px figure needs a bigger font than the 600 px widget.
                scale = max(1.0, image.height() / 800.0)
                painter.scale(scale, scale)
                self._paint_legend(painter, int(image.width() / scale), int(image.height() / scale))
                self._paint_scale_bar(
                    painter, int(image.width() / scale), int(image.height() / scale)
                )
            finally:
                self._snapshotting = False
                painter.end()
        return bool(image.save(str(path)))

    # -- pose animation -----------------------------------------------------

    def play_animation(self, frames: Sequence[Sequence], interval_ms: int = 90) -> None:
        """Interpolate between poses (the conformer player)."""
        from .structure import Atom

        self.stop_animation()
        template = self.scene.ligand
        if len(frames) < 2 or not template:
            return
        self._anim_frames = [
            [
                Atom(
                    name=a.name, element=a.element, res_name=a.res_name,
                    res_id=a.res_id, chain=a.chain, x=float(c[0]), y=float(c[1]),
                    z=float(c[2]), charge=a.charge, ad_type=a.ad_type, serial=a.serial,
                )
                for a, c in zip(template, coords)
            ]
            for coords in frames
        ]
        self._anim_index = 0
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self._advance_animation)
        timer.start(max(20, int(interval_ms)))
        self._anim_timer = timer

    def _advance_animation(self) -> None:
        if not self._anim_frames:
            self.stop_animation()
            return
        self.scene.ligand = self._anim_frames[self._anim_index]
        self._anim_index = (self._anim_index + 1) % len(self._anim_frames)
        self.refresh()

    def stop_animation(self) -> None:
        if self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = None
        self._anim_frames = []


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


class _DockWorker(QtCore.QThread):
    """Runs one docking job off the GUI thread, with pause and abort.

    It drives ``odock._odock.Docking`` directly rather than the convenience
    wrapper so that the cooperative pause/cancel handle is reachable while the
    search is running.
    """

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    ticked = QtCore.pyqtSignal(str)
    #: ``(phase, seconds since the run started)``. The phases are the ones this
    #: thread can actually observe: the grid is built by ``build_engine``, the
    #: search (and, inside it, the kernel's refinement) is ``run()``, and "done"
    #: is the run returning. No per-iteration energy exists to report — the
    #: kernel holds its state for the whole search — so none is invented.
    phase = QtCore.pyqtSignal(str, float)

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.settings = settings
        self.engine = None
        self.trace: Optional[dashboard.RunTrace] = None
        self.phases: List[dashboard.PhaseRecord] = []
        self.elapsed = 0.0

    def _enter(self, name: str, elapsed: float) -> None:
        """Record and announce a phase boundary."""
        self.phases.append(dashboard.PhaseRecord(name=name, started=float(elapsed)))
        self.phase.emit(name, float(elapsed))

    def run(self) -> None:  # noqa: D401
        try:
            import time as _time

            from odock.docking import build_engine, result_from_engine

            s = self.settings
            t0 = _time.perf_counter()
            self._enter("grid", 0.0)
            self.ticked.emit(tr("worker.building_grid"))
            # `build_engine` + `run()` rather than the `dock()` convenience
            # wrapper: holding the engine is what makes pause and abort
            # reachable, and the result conversion is shared with `dock()` so
            # that the two can never drift apart.
            engine = build_engine(
                s["receptor_text"],
                s["ligand_text"],
                s["box"],
                scoring=s["scoring"],
                exhaustiveness=s["exhaustiveness"],
                num_poses=s["num_poses"],
                seed=s["seed"],
                use_grid=s["use_grid"],
                refine=True,
                min_rmsd=s["min_rmsd"],
                energy_range=s["energy_range"],
                use_island_ga=s["use_island_ga"],
                islands=s["islands"],
                population=s["population"],
                generations=s["generations"],
                search=s.get("search"),
            )
            self.engine = engine
            self._close_last(t0)
            self._enter("search", _time.perf_counter() - t0)
            self.ticked.emit(tr("worker.searching"))
            raw = engine.run()
            search_seconds = _time.perf_counter() - t0
            self._close_last(t0)
            # The refinement lives inside `run()` and returns with it: the stage
            # is marked complete when its results are in hand, and carries no
            # separate duration because the kernel does not expose one.
            self._enter("refine", search_seconds)
            # The refinement returns inside `run()`, so there is no separate
            # duration to report: the stage is marked with a zero-length span
            # rather than a made-up number.
            self.phases[-1].finished = search_seconds
            self.elapsed = _time.perf_counter() - t0
            self.ticked.emit(tr("worker.search_done", seconds=search_seconds))
            if raw.get("cancelled"):
                self.ticked.emit(tr("worker.cancelled"))
            result = result_from_engine(
                engine,
                raw,
                elapsed=search_seconds,
                box=s["box"],
                receptor_pdbqt=s["receptor_text"],
                ligand_pdbqt=s["ligand_text"],
                energy_range=s["energy_range"],
            )
            self._enter("done", self.elapsed)
            self._close_last(t0)
            self.trace = dashboard.RunTrace(
                energies=[float(pose.affinity) for pose in result.poses],
                phases=list(self.phases),
                elapsed=float(self.elapsed),
                scoring=str(s["scoring"]),
                grid_points=int(getattr(result, "grid_points", 0) or 0),
                grid_mb=int(getattr(result, "grid_mb", 0) or 0),
                num_tors=float(getattr(result, "num_tors", 0.0) or 0.0),
                exhaustiveness=int(s["exhaustiveness"]),
                seed=int(s["seed"]),
                cancelled=bool(raw.get("cancelled")),
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # pragma: no cover - reported to the user
            import traceback

            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _close_last(self, t0: float) -> None:
        """Finish the phase that is currently open."""
        if not self.phases:
            return
        record = self.phases[-1]
        record.finished = time.perf_counter() - t0


    # -- control -----------------------------------------------------------

    def pause(self) -> None:
        if self.engine is not None:
            try:
                self.engine.pause()
                return
            except Exception:
                pass
        self.ticked.emit(tr("worker.no_pause"))

    def resume(self) -> None:
        if self.engine is not None:
            try:
                self.engine.resume()
                return
            except Exception:
                pass
        self.ticked.emit(tr("worker.no_resume"))

    def cancel(self) -> None:
        if self.engine is not None:
            try:
                self.engine.cancel()
                self.ticked.emit(tr("worker.abort_requested"))
                return
            except Exception:
                pass
        self.ticked.emit(tr("worker.no_abort"))


class _CallWorker(QtCore.QThread):
    """Runs an arbitrary callable off the GUI thread and returns its result."""

    finished_ok = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    ticked = QtCore.pyqtSignal(str)

    def __init__(self, function, *args, label: str = "", **kwargs) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.label = label

    def run(self) -> None:  # noqa: D401
        try:
            if self.label:
                self.ticked.emit(self.label)
            self.finished_ok.emit(self.function(*self.args, **self.kwargs))
        except Exception as exc:  # pragma: no cover - reported to the user
            import traceback

            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# The workbench
# ---------------------------------------------------------------------------


class DockingWorkbench(QtWidgets.QMainWindow):
    """The main window.

    Every label comes from :mod:`odock.gui.i18n` and is read while the widgets
    are built, so :meth:`set_language` rebuilds the interface in place rather
    than trying to walk the existing widget tree and relabel it.
    """

    def __init__(
        self,
        receptor: Optional[str] = None,
        ligand: Optional[str] = None,
        poses: Optional[str] = None,
        parent=None,
        *,
        session: Optional["dashboard.SessionStore"] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("app.title"))
        self.resize(1360, 860)

        self.scene = Scene()
        self.receptor_text: Optional[str] = None
        self.ligand_text: Optional[str] = None
        self.ligand_models: List[Model] = []
        self.pose_models: List[Model] = []
        self.result = None
        self.pockets: List = []
        self.clusters: List = []
        self.interactions: List = []
        #: Whether the user has asked for the interaction emphasis (Analysis ▸
        #: Show interactions). The request is sticky, so browsing poses keeps the
        #: site highlighted — but it starts off, and Clear annotations (or a new
        #: project) switches it off.
        self._focus_requested = False
        #: Whether the interaction *drawing* is on. Off by default: docking and
        #: pose browsing report contacts (the pose label names the residues, the
        #: table lists the pairs) but draw nothing until Analysis ▸ Show
        #: interactions — a dash is what people read as "these two things
        #: interact", so it is opt-in and Clear annotations switches it off.
        self._interactions_shown = False
        #: How the current box was arrived at (``ligand``/``pocket``/``manual``/
        #: ``receptor``).  Recorded because the same numbers reached by different
        #: routes are different decisions: a box fitted to the ligand follows it
        #: to a new pose, a hand-placed one does not, and a protocol says which.
        self._box_source = "ligand"
        #: The template currently being compared against, and the last protocol
        #: loaded from a file (for the tab's diff label).
        self._loaded_protocol = None
        self._protocol_view = None
        self.receptor_mol = None
        self.ligand_mol = None
        self.inventory = None
        self._reference_pose = None
        self._worker: Optional[QtCore.QThread] = None
        self._background: Optional[QtCore.QThread] = None
        self._pending: Dict[str, object] = {}
        self._measurements: List[dashboard.Measurement] = []
        #: Text labels pinned to the scene (View ▸ Annotate).
        self._annotations: List[dashboard.Annotation] = []
        self._annotation_visible = True
        #: What the next pick will measure, and the picks collected so far.
        self._measure_kind = "distance"
        self._pick_refs: List[Tuple[str, int]] = []
        #: The undo stack, and a guard so a rebuild or a load never records a step.
        self._history = dashboard.CommandStack()
        self._suppress_history = False
        self._locked_bonds: List[tuple] = []
        #: Interaction distance cut-offs (View ▸ Interaction distances…). They
        #: live on the window, not in a widget, so re-annotating and a language
        #: switch both keep the user's choice.
        self.interaction_thresholds: Dict[str, float] = dict(
            dialogs.DEFAULT_INTERACTION_THRESHOLDS
        )
        #: The reference-pose overlay (View ▸ Show reference pose) and the pose
        #: currently on screen; both survive a language switch.
        self._ghost_visible = False
        self._pose_index = 0
        self._charge_model = "gasteiger"
        #: The molecular surface. The *settings* live on the window (so a
        #: language switch, which rebuilds every widget, cannot lose them), the
        #: built mesh lives on the scene (where the renderer finds it), and the
        #: cut plane lives in both because the shader reads it from the scene.
        from . import surface as _surface_module

        self.surface_settings = _surface_module.SurfaceSettings()
        self._surface_alpha = 1.0
        self._surface_clip = None
        self._pocket_only = False
        self._highlight_pocket = False
        #: How wide a shell "pocket lining only" builds. Wide enough to hold a
        #: probe around the lining, which is what makes the shell the answer
        #: rather than an approximation of it.
        self._pocket_radius = 9.0
        #: Radio actions of the two View ▸ style submenus, keyed by style name.
        self._style_actions: Dict[str, Dict[str, QtGui.QAction]] = {}
        #: The pose text is kept so that a language switch can reload it.
        self.poses_text: Optional[str] = None
        #: Pause state, kept as a flag because the button label is translated.
        self._paused = False
        #: Whether the search box is meant to be visible. A fresh workbench has
        #: no box at all: pushing the placeholder spin values into the scene at
        #: startup used to draw a 2 Å cube and its six handle spheres on an
        #: otherwise empty view, which is exactly what the user complained about.
        self._box_visible = False
        #: Whether the pose table and the log are stacked instead of side by side.
        self._pose_stacked = False
        #: Panel state that is expensive to recompute and worth restoring.
        self._maps_computed = False
        self._filter_verdict: Optional[Tuple[bool, Tuple[str, ...]]] = None
        #: The instrument panels. A language switch rebuilds every widget, so
        #: the *data* of the dashboard, the comparison and the measurements
        #: lives on the window and is re-applied to the fresh widgets.
        self.run_history = dashboard.SessionHistory(cap=12)
        self._comparison: Optional[dashboard.PoseComparison] = None
        self._comparison_rows: Tuple[int, int] = (0, 1)
        #: Appearance and arrangement, both switchable from the View menu.
        self.color_theme: dashboard.Theme = dashboard.DARK
        self.density = "comfortable"
        self._preset = "docking"
        #: The session file. ``None`` disables persistence entirely, which is
        #: what a test (or a plain ``DockingWorkbench()``) gets: nothing is ever
        #: written to the user's session behind their back.
        self.session: Optional[dashboard.SessionStore] = session
        self.recent = (
            session.recent if session is not None else dashboard.RecentFiles()
        )
        self._autosave_timer: Optional[QtCore.QTimer] = None
        self._autosave_pending = False
        #: Which interaction kinds are *drawn* (View ▸ Interaction lines).
        #: ``None`` means "all of them", which is what a fresh window wants.
        self._interaction_filter: Optional[set] = None

        self._build_viewport()
        self._build_inspector()
        self._build_workspace()
        self._build_bottom()
        self._build_dashboard()
        self._build_console()
        self._build_comparison()
        self._build_selection_panel()
        self._build_menus()
        self._build_statusbar()
        self._apply_style()
        self._apply_dock_proportions()
        self._install_autosave()
        self.setAcceptDrops(True)

        if receptor:
            self.load_receptor(receptor)
        if ligand:
            self.load_ligand(ligand)
        if poses:
            self.load_poses(poses)
        if not self.scene.has_content():
            self._log(tr("log.ready"))

    # -- language -----------------------------------------------------------

    def set_language(self, code: str) -> None:
        """Switch the interface language, keeping the whole session.

        Widgets read their labels while they are built, so the honest way to
        relabel them is to build them again. Everything that lives on the
        window — the loaded structures, the molecules, the poses, the results,
        the locked bonds, the flexible residues and the log — survives;
        everything that lives in a widget is captured first and re-applied
        afterwards.
        """
        i18n.set_language(code)  # raises ValueError for an unknown code
        state = self._capture_ui_state()
        self._teardown_ui()
        self._build_viewport()
        self._build_inspector()
        self._build_workspace()
        self._build_bottom()
        self._build_dashboard()
        self._build_console()
        self._build_comparison()
        self._build_selection_panel()
        self._build_menus()
        self._build_statusbar()
        self._apply_style()
        self.setWindowTitle(tr("app.title"))
        self._restore_ui_state(state)
        self._session_changed()

    def _capture_ui_state(self) -> dict:
        """Everything the rebuild has to put back."""
        return {
            "receptor_text": self.receptor_text,
            "ligand_text": self.ligand_text,
            "poses_text": self.poses_text,
            "pending": dict(self._pending),
            "box": self.scene.box,
            "spacing": float(self.spacing.value()),
            "box_visible": self.chk_box.isChecked(),
            "focus_requested": self._focus_requested,
            "pose_index": self.pose_index() if self.pose_models else None,
            "pose_row": self.table.currentRow(),
            "flex": [
                self.flex_list.item(i).text() for i in range(self.flex_list.count())
            ],
            "filters": self._filter_verdict,
            "maps": self._maps_computed,
            "camera": (
                tuple(self.viewport.camera.target),
                self.viewport.camera.distance,
                self.viewport.camera.azimuth,
                self.viewport.camera.elevation,
            ),
            "had_renderer": self.viewport.renderer is not None,
            "tool": self.viewport.mode,
            "docks": {
                name: not getattr(self, name).isHidden()
                for name in (
                    "workspace_dock",
                    "inspector_dock",
                    "bottom_dock",
                    "dashboard_dock",
                    "comparison_dock",
                    "selection_dock",
                )
            },
            "tab": self.inspector.currentIndex(),
            "sequence_keys": self.sequence.selected_keys(),
            "pose_stacked": self._pose_stacked,
            "pose_split": list(self.pose_splitter.sizes()),
            "central_split": list(self.central_splitter.sizes()),
            "running": bool(self._worker is not None and self._worker.isRunning()),
            "paused": self._paused,
            "theme": self.color_theme.name,
            "density": self.density,
            "preset": self._preset,
            "interaction_filter": (
                None
                if self._interaction_filter is None
                else sorted(self._interaction_filter)
            ),
            "surface": {
                "settings": self.surface_settings,
                "alpha": float(self._surface_alpha),
                "clip": self._surface_clip,
                "pocket_only": bool(self._pocket_only),
                "highlight": bool(self._highlight_pocket),
                "show": bool(self.scene.show_surface),
                "legend": bool(self.scene.surface_legend),
            },
            "history": self.run_history.to_list(),
            "measurements": [item.to_dict() for item in self._measurements],
            "annotations": [item.to_dict() for item in self._annotations],
            "annotation_visible": bool(self._annotation_visible),
            "measure_kind": str(self._measure_kind),
            "comparison_rows": tuple(self._comparison_rows),
            "comparison": self._comparison is not None,
            "engine": {
                "scoring": self.engine.currentText(),
                "search": self.search.currentIndex(),
                "exhaustiveness": self.exhaustiveness.value(),
                "poses": self.poses.value(),
                "seed": self.seed.value(),
                "energy_range": self.energy_range.value(),
                "islands": self.islands.value(),
                "population": self.population.value(),
                "generations": self.generations.value(),
                "use_grid": self.use_grid.isChecked(),
                "threads": self.threads.value(),
            },
            "charge_model": self._charge_model,
            "log": self.log.toPlainText(),
            "status": self.lbl_status.text(),
        }

    def _teardown_ui(self) -> None:
        """Drop the menus, the docks, the status widgets and the viewport."""
        bar = self.menuBar()
        for action in list(bar.actions()):
            menu = action.menu()
            bar.removeAction(action)
            if menu is not None:
                _clear_menu_shortcuts(menu)
                menu.setParent(None)
                menu.deleteLater()
            else:  # pragma: no cover - the bar only ever holds menus
                action.setParent(None)
                action.deleteLater()

        for name in (
            "workspace_dock",
            "inspector_dock",
            "bottom_dock",
            "dashboard_dock",
            "comparison_dock",
            "selection_dock",
        ):
            dock = getattr(self, name, None)
            if dock is not None:
                self.removeDockWidget(dock)
                dock.setParent(None)
                dock.deleteLater()
                setattr(self, name, None)
        self.run_dashboard = None
        self.measure_history = None
        self.comparison = None

        for name in ("lbl_status", "lbl_energy"):
            label = getattr(self, name, None)
            if label is not None:
                self.statusBar().removeWidget(label)
                label.setParent(None)
                label.deleteLater()
                setattr(self, name, None)

        strip = getattr(self, "_tool_strip", None)
        if strip is not None:
            strip.setParent(None)
            strip.deleteLater()
            self._tool_strip = None

        viewport = getattr(self, "viewport", None)
        central = self.centralWidget()
        if central is not None:
            central.setParent(None)
            central.deleteLater()
        if viewport is not None:
            _release_viewport(viewport)
            self.viewport = None

    @_history_silent
    def _restore_ui_state(self, state: dict) -> None:
        """Put the captured session back into the freshly built widgets."""
        self._charge_model = state["charge_model"]

        # The appearance goes back first: it is what the user sees while the
        # rest of the interface is being repopulated.
        self.color_theme = dashboard.theme_named(state.get("theme", "dark"))
        self.density = str(state.get("density", "comfortable"))
        self._preset = str(state.get("preset", self._preset))
        self._apply_style()
        self._sync_appearance_actions()

        engine = state["engine"]
        self.engine.setCurrentText(engine["scoring"])
        self.search.setCurrentIndex(engine["search"])
        self.exhaustiveness.setValue(engine["exhaustiveness"])
        self.poses.setValue(engine["poses"])
        self.seed.setValue(engine["seed"])
        self.energy_range.setValue(engine["energy_range"])
        self.islands.setValue(engine["islands"])
        self.population.setValue(engine["population"])
        self.generations.setValue(engine["generations"])
        self.use_grid.setChecked(engine["use_grid"])
        self.threads.setValue(engine["threads"])

        self.flex_list.clear()
        for name in state["flex"]:
            self.flex_list.addItem(name)
        self._refresh_flex_label()

        # The sticky interaction emphasis has to be back in place *before* the
        # structures load: the first pose annotates itself from it, so restoring
        # it later would silently drop the emphasis on a language switch.
        self._focus_requested = bool(state.get("focus_requested", False))

        if state["receptor_text"]:
            self.load_receptor(state["receptor_text"])
        if state["ligand_text"]:
            self.load_ligand(state["ligand_text"])
        if state["poses_text"]:
            self.load_poses(state["poses_text"])
        # The ruler was rebuilt empty; put the marked residues back.
        if state.get("sequence_keys"):
            self.sequence.select_keys(state["sequence_keys"])
        # Keep the two dividers where the user left them: the layout must
        # survive the rebuild, not snap back to the defaults.
        self._pose_stacked = bool(state.get("pose_stacked", False))
        self.pose_splitter.setOrientation(
            QtCore.Qt.Orientation.Vertical
            if self._pose_stacked
            else QtCore.Qt.Orientation.Horizontal
        )
        if state.get("pose_split"):
            self.pose_splitter.setSizes(list(state["pose_split"]))
        if state.get("central_split"):
            self.central_splitter.setSizes(list(state["central_split"]))
        self._fit_sequence_area()

        # ``load_*`` rewrite these from whatever they were handed, so the
        # remembered values go back last.
        self._pending = dict(state["pending"])
        self._box_visible = bool(state["box_visible"])
        self.chk_box.setChecked(self._box_visible)
        if self.box_action is not None:
            self.box_action.setChecked(self._box_visible)
        if state["box"] is not None:
            self._set_box(state["box"][0], state["box"][1], state["spacing"])
        else:
            self.scene.box = None
            self.spacing.setValue(state["spacing"])
            self._refresh_box_label()

        self._filter_verdict = state["filters"]
        self._refresh_filter_label()
        if state["maps"] and self.scene.box is not None:
            self._compute_maps()
        else:
            self._maps_computed = False
            self.lbl_maps.setText(tr("label.no_grid"))

        if state["pose_index"] is not None and self.pose_models:
            self.set_pose(state["pose_index"], record=False)
            if state["pose_row"] >= 0:
                self.table.selectRow(state["pose_row"])

        self._set_running(state["running"])
        self._paused = bool(state["paused"]) and bool(state["running"])
        if self._paused:
            self.btn_pause.setText(tr("btn.resume"))

        # The surface survives the rebuild: the mesh on the scene, the settings
        # here, and the widgets the new menu built have to agree with them.
        surface_state = state.get("surface") or {}
        if surface_state.get("settings") is not None:
            self.surface_settings = surface_state["settings"]
        self._surface_alpha = float(surface_state.get("alpha", self._surface_alpha))
        self._surface_clip = surface_state.get("clip", self._surface_clip)
        self._pocket_only = bool(surface_state.get("pocket_only", self._pocket_only))
        self._highlight_pocket = bool(
            surface_state.get("highlight", self._highlight_pocket)
        )
        self.scene.surface_alpha = self._surface_alpha
        self.scene.surface_clip = self._surface_clip
        self.scene.show_surface = bool(surface_state.get("show", self.scene.show_surface))
        self.scene.surface_legend = bool(
            surface_state.get("legend", self.scene.surface_legend)
        )
        for name, value in (
            ("surface_action", self.scene.show_surface and self.scene.surface is not None),
            ("legend_action", self.scene.surface_legend),
            ("pocket_action", self._pocket_only),
            ("highlight_action", self._highlight_pocket),
        ):
            self._set_action_checked(getattr(self, name, None), bool(value))
        self._sync_surface_actions()
        if self.scene.surface is not None and self.viewport.renderer is not None:
            self.viewport.set_surface(self.scene.surface)

        target, distance, azimuth, elevation = state["camera"]
        if state["had_renderer"]:
            # Building the context frames the camera, so restore it afterwards.
            self.viewport._ensure_context()
            self.viewport.camera.target = target
            self.viewport.camera.distance = distance
            self.viewport.camera.azimuth = azimuth
            self.viewport.camera.elevation = elevation
        self.viewport.set_mode(state["tool"])
        self.viewport.refresh()

        for name, visible in state["docks"].items():
            dock = getattr(self, name, None)
            if dock is not None:
                dock.setVisible(visible)
        # The docks were rebuilt around the same content, so the opening
        # proportions have to be re-applied with them — on the next event-loop
        # turn, because a hint during the rebuild is ignored.
        self._apply_dock_proportions()
        self._schedule_dock_proportions()
        self.inspector.setCurrentIndex(state["tab"])

        # The instrument panels: their data survives the rebuild, the widgets
        # around it are new.
        self.run_history = dashboard.SessionHistory.from_list(state.get("history", []))
        if self.run_dashboard is not None:
            self.run_dashboard.history = self.run_history
            self.run_dashboard.reset(clear_history=False)
        self._measurements = [
            dashboard.Measurement.from_dict(item)
            for item in state.get("measurements", [])
        ]
        self._annotations = [
            dashboard.Annotation.from_dict(item)
            for item in state.get("annotations", [])
        ]
        self._annotation_visible = bool(state.get("annotation_visible", True))
        self._measure_kind = str(state.get("measure_kind", self._measure_kind))
        self._sync_measure_actions()
        self._sync_measurements()
        self._sync_annotations()
        self._comparison_rows = tuple(state.get("comparison_rows", (0, 1)))
        self._comparison = None
        if state.get("comparison") and len(self.pose_models) >= 2:
            self._update_comparison(*self._comparison_rows[:2])
        elif self.comparison is not None:
            self.comparison.clear()

        # View ▸ Interaction lines survives the rebuild with everything else: the
        # menu actions are new widgets, so they are re-ticked from the saved set
        # and the drawn lines are recomputed from it.
        saved_filter = state.get("interaction_filter")
        self._interaction_filter = (
            None if saved_filter is None else {str(kind) for kind in saved_filter}
        )
        for kind, action in getattr(self, "_interaction_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(
                self._interaction_filter is None or kind in self._interaction_filter
            )
            action.blockSignals(False)
        self._sync_interactions()

        self.log.setPlainText(state["log"])
        self.lbl_status.setText(state["status"])
        self._log(
            tr("log.language", language=i18n.language_name(i18n.current_language()))
        )

    # -- dock proportions ---------------------------------------------------

    #: Opening width of the left (workspace) dock, in pixels. Fixed and narrow:
    #: the tree elides its labels, so the project tree needs very little.
    DOCK_WIDTH_LEFT = 180
    #: The right dock area (inspector, plus the selection and pose-comparison
    #: panels that share it) opens at this fraction of the window width, clamped
    #: to the bounds below. Proportional rather than fixed because a fixed 560 px
    #: right dock is 43.8% of a 1280 px window — wider than the viewport, which is
    #: the complaint this answers.
    DOCK_RIGHT_FRACTION = 0.35
    DOCK_WIDTH_RIGHT_MIN = 320
    DOCK_WIDTH_RIGHT_MAX = 560

    def _dock_widths(self) -> Tuple[int, int]:
        """``(left, right)`` opening widths for the current window width."""
        right = int(round(max(1, self.width()) * self.DOCK_RIGHT_FRACTION))
        return (
            self.DOCK_WIDTH_LEFT,
            max(self.DOCK_WIDTH_RIGHT_MIN, min(self.DOCK_WIDTH_RIGHT_MAX, right)),
        )

    def _apply_dock_proportions(self) -> None:
        """Open narrow left, wide right, and leave everything else to the centre.

        These are *hints* (`resizeDocks`), not floors: either divider can still be
        dragged, and a window resize gives every extra pixel to the central
        viewport — measured, a 1600 → 1800 px resize moved the centre by +200 px
        and the two side docks by 0.
        """
        if getattr(self, "workspace_dock", None) is None:
            return
        left, right = self._dock_widths()
        docks = [self.workspace_dock]
        sizes = [left]
        if getattr(self, "inspector_dock", None) is not None:
            docks.append(self.inspector_dock)
            sizes.append(right)
        self.resizeDocks(docks, sizes, QtCore.Qt.Orientation.Horizontal)

    def _schedule_dock_proportions(self) -> None:
        """Re-apply the opening proportions once the layout has settled.

        ``resizeDocks`` is *silently ignored* while the window is being rebuilt:
        the language switch tears the docks and the central widget down first, so
        a hint applied during the rebuild does nothing and Qt distributes the
        width itself — measured, that put 584 px on each side dock out of a
        1600 px window and squeezed the viewport to its 420 px minimum. Deferring
        the hint by one event-loop turn makes it land on a live layout.
        """
        QtCore.QTimer.singleShot(0, self._apply_dock_proportions)

    # -- the protocol tab ---------------------------------------------------

    def _build_protocol_tab(self):
        """The current settings as a shareable protocol: Save / Load / Compare.

        The tab is a *view* of what the other tabs say. Nothing here holds
        settings of its own: :meth:`current_protocol` reads the widgets, so a
        protocol can never describe something the workbench is not set to do, and
        saving cannot record a value the user has not seen.
        """
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.protocol_caption = QtWidgets.QLabel(tr("protocol.caption"))
        self.protocol_caption.setWordWrap(True)
        self.protocol_caption.setObjectName("dashboardCaption")
        layout.addWidget(self.protocol_caption)

        self.protocol_view = QtWidgets.QPlainTextEdit()
        self.protocol_view.setReadOnly(True)
        self.protocol_view.setObjectName("protocolView")
        self.protocol_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.protocol_view.setMinimumHeight(120)
        layout.addWidget(self.protocol_view, 1)

        row = QtWidgets.QHBoxLayout()
        self._act_row_button(row, tr("btn.protocol_save"), self._save_protocol_action)
        self._act_row_button(row, tr("btn.protocol_load"), self._load_protocol_action)
        self._act_row_button(row, tr("btn.protocol_diff"), self._compare_protocol_action)
        layout.addLayout(row)

        template_row = QtWidgets.QHBoxLayout()
        template_row.addWidget(QtWidgets.QLabel(tr("protocol.template")))
        self.protocol_template = QtWidgets.QComboBox()
        self.protocol_template.setObjectName("protocolTemplate")
        self.protocol_template.addItem(tr("protocol.template_none"), None)
        for template in self._protocol_templates():
            self.protocol_template.addItem(template.name, template.name)
        self.protocol_template.currentIndexChanged.connect(
            lambda _index: self._refresh_protocol_tab()
        )
        template_row.addWidget(self.protocol_template, 1)
        layout.addLayout(template_row)

        self.protocol_template_diff = QtWidgets.QLabel("")
        self.protocol_template_diff.setWordWrap(True)
        self.protocol_template_diff.setObjectName("dashboardCaption")
        layout.addWidget(self.protocol_template_diff)
        self._protocol_page = panel
        return panel

    def _act_row_button(self, row, text, slot):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda _checked=False: slot())
        row.addWidget(button)
        return button

    def _protocol_templates(self) -> List["protocol.Protocol"]:
        """The shipped templates, or an empty list when none are installed."""
        try:
            return protocol.list_templates()
        except Exception:  # pragma: no cover - a broken template must not crash
            return []

    def _on_inspector_tab_changed(self, index: int) -> None:
        widget = self.inspector.widget(index)
        if widget is self._protocol_tab_widget():
            self._refresh_protocol_tab()

    #: The Engine tab lists these labels in this order; the protocol records the
    #: *value* the kernel takes (``cli.py``'s `--search` choices), never the
    #: translated label the user read.
    SEARCH_VALUES = ("monte_carlo", "lga", "lga_solis")

    def current_protocol(self) -> "protocol.Protocol":
        """The settings the inspector is showing, as a protocol.

        Built from the widgets themselves — the box spins, the engine controls,
        the seed, the interaction thresholds — through
        :func:`odock.protocol.capture_session`, so the document and the panels
        cannot disagree about what the workbench is set to.

        What the GUI has no control for (whether water and hetero groups are kept,
        the library filters, the preparation flags) stays at the documented
        default in the captured document: a protocol must not claim the workbench
        is set to something it has no widget for. The CLI's
        ``protocol save --set section.field=value`` is how those are chosen.
        """
        box = self._current_box()
        payload = {
            "box": (
                {
                    "source": self._box_source,
                    "center": list(box.center),
                    "size": list(box.size),
                    "spacing": float(box.spacing),
                }
                if box is not None
                else {"source": self._box_source}
            ),
            "engine": {
                "scoring": self.engine.currentText(),
                "search": self.SEARCH_VALUES[
                    max(0, min(self.search.currentIndex(), len(self.SEARCH_VALUES) - 1))
                ],
                "exhaustiveness": int(self.exhaustiveness.value()),
                "num_poses": int(self.poses.value()),
                "energy_range": float(self.energy_range.value()),
                "islands": int(self.islands.value()),
                "population": int(self.population.value()),
                "generations": int(self.generations.value()),
                "use_grid": bool(self.use_grid.isChecked()),
            },
            "execution": {"seed": int(self.seed.value()), "jobs": int(self.threads.value())},
            "thresholds": {"interactions": dict(self.interaction_thresholds)},
        }
        return protocol.capture_session(payload, name=self._protocol_name())

    def _protocol_name(self) -> str:
        """A name a user will recognise in a directory listing."""
        receptor = self._pending.get("receptor")
        stem = Path(str(receptor)).stem if receptor else "session"
        return f"{stem}-{'box' if self.scene.box is not None else 'nobox'}"

    def _refresh_protocol_tab(self) -> None:
        try:
            captured = self.current_protocol()
        except Exception as exc:  # pragma: no cover - reported, never fatal
            self.protocol_view.setPlainText(str(exc))
            return
        self._protocol_view = captured
        self.protocol_view.setPlainText("\n".join(captured.summary_lines()))
        template = self.protocol_template.currentData()
        if not template:
            self.protocol_template_diff.setText(tr("protocol.template_none"))
            return
        chosen = next(
            (item for item in self._protocol_templates() if item.name == template), None
        )
        if chosen is None:  # pragma: no cover - the combo is filled from the same list
            self.protocol_template_diff.setText(tr("protocol.template_none"))
            return
        differences = protocol.diff_protocols(chosen, captured)
        settings = [item for item in differences if item["kind"] == "setting"]
        if not settings:
            self.protocol_template_diff.setText(
                tr("protocol.template_same", name=chosen.name)
            )
        else:
            self.protocol_template_diff.setText(
                tr("protocol.template_differs", n=len(settings), name=chosen.name)
            )
        self.protocol_template_diff.setToolTip(
            protocol.protocol_diff_text(chosen, captured, differences)
        )

    def _save_protocol_action(self) -> None:
        """Write the live settings as a protocol, validated before it is written."""
        try:
            captured = self.current_protocol()
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("btn.protocol_save"), f"{captured.name}.json", ""
        )
        if not path:
            return
        try:
            protocol.save_protocol(captured, path)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._log(tr("log.protocol_saved", path=Path(path).name))
        self._refresh_protocol_tab()

    def _load_protocol_action(self) -> None:
        """Apply a protocol to the *widgets*, so the panels show what will run."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("btn.protocol_load"), "", ""
        )
        if not path:
            return
        try:
            loaded = protocol.load_protocol(path)
            loaded.validate(for_run=True)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._apply_protocol_to_widgets(loaded)
        self._loaded_protocol = loaded
        self._log(tr("log.protocol_loaded", path=Path(path).name))
        self._refresh_protocol_tab()

    def _apply_protocol_to_widgets(self, loaded: "protocol.Protocol") -> None:
        """Put a protocol into the controls it describes.

        Only what the GUI owns: the box, the engine and search settings, the seed,
        the filter switch. Preparation flags and thresholds are carried in the
        document but the *run* is what uses them, so loading cannot pretend the
        GUI has applied something it does not hold.
        """
        if loaded.box.center and loaded.box.size:
            self._set_box(
                loaded.box.center, loaded.box.size, spacing=float(loaded.box.spacing)
            )
        self.engine.setCurrentText(str(loaded.engine.scoring))
        if loaded.engine.search in self.SEARCH_VALUES:
            self.search.setCurrentIndex(self.SEARCH_VALUES.index(loaded.engine.search))
        self.exhaustiveness.setValue(int(loaded.engine.exhaustiveness))
        self.poses.setValue(int(loaded.engine.num_poses))
        self.energy_range.setValue(float(loaded.engine.energy_range))
        self.islands.setValue(int(loaded.engine.islands))
        self.population.setValue(int(loaded.engine.population))
        self.generations.setValue(int(loaded.engine.generations))
        self.use_grid.setChecked(bool(loaded.engine.use_grid))
        self.seed.setValue(int(loaded.execution.seed))
        self.threads.setValue(int(loaded.execution.jobs))
        self.interaction_thresholds = dict(loaded.thresholds.interactions)
        self._set_box_source(loaded.box.source)
        self.viewport.refresh()

    def _compare_protocol_action(self) -> None:
        """The diff against a saved protocol, in the log and in the tab."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("btn.protocol_diff"), "", ""
        )
        if not path:
            return
        try:
            other = protocol.load_protocol(path)
            captured = self.current_protocol()
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        differences = protocol.diff_protocols(other, captured)
        settings = [item for item in differences if item["kind"] == "setting"]
        self._log(
            tr("protocol.template_differs", n=len(settings), name=Path(path).name)
            if settings
            else tr("protocol.template_same", name=Path(path).name)
        )
        for line in protocol.protocol_diff_text(other, captured, differences).splitlines():
            self._log(line)
        self.protocol_view.setPlainText(
            protocol.protocol_diff_text(other, captured, differences)
        )

    def _build_console(self) -> None:
        """The console input line, at the bottom of the log panel.

        Output and input live in the same place: the transcript is the log view
        the workbench already writes to, and this adds one line under it. There is
        no separate Console tab — reading and typing must not need a tab switch.
        """
        console_input = dashboard.ConsoleInput(
            self._console_execute, self.console_namespace
        )
        console_input.setObjectName("consoleInput")
        console_input.setToolTip(tr("console.banner"))
        console_input.completionHint.connect(self.log.appendPlainText)
        console_input.editingFinished.connect(self._focus_viewport)
        self.console = console_input
        prompt = QtWidgets.QLabel(console_input.PROMPT)
        prompt.setObjectName("consolePrompt")
        prompt.setMinimumWidth(
            prompt.fontMetrics().horizontalAdvance(console_input.PROMPT) + 4
        )
        self.console_prompt = prompt
        # The prompt follows the *block state* (a continued block shows `... `),
        # which the widget announces — editing the line alone cannot know it.
        console_input.promptChanged.connect(self.console_prompt.setText)

        self.console_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.console_row)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(4)
        row.addWidget(prompt)
        row.addWidget(console_input, 1)

        # The log panel becomes "the log, plus the line you type into": the
        # splitter keeps one widget per side, so the container replaces the bare
        # log in it (and the log keeps its identity for everything that reads it).
        self.log_panel = QtWidgets.QWidget()
        self.log_panel.setObjectName("logPanel")
        index = self.pose_splitter.indexOf(self.log)
        if index >= 0:
            self.pose_splitter.replaceWidget(index, self.log_panel)
        panel = QtWidgets.QVBoxLayout(self.log_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(0)
        panel.addWidget(self.log, 1)
        panel.addWidget(self.console_row)

    def _clear_console(self) -> None:
        """Clear the transcript — the log view *is* the console's output."""
        self.log.clear()
        self.console.clear()
        self._log(tr("action.console_clear"))

    def _focus_viewport(self) -> None:
        """Give the keyboard back to the 3-D view (Escape in the console)."""
        view = getattr(self, "viewport", None)
        if view is not None:
            view.setFocus()

    def console_namespace(self) -> dict:
        """The names a console command can use.

        The live objects come first (so ``scene.`` and ``window.`` are the session
        you are looking at, not a copy), then the helpers that mirror the CLI —
        named so that someone who knows the command line can guess them.
        """
        namespace = {
            "window": self,
            "scene": self.scene,
            "viewport": self.viewport,
            "receptor": self.scene.receptor,
            "ligand": self.scene.ligand,
            "poses": self.pose_models,
            "box": self.scene.box,
            "interactions": self.interactions,
            "measurements": self._measurements,
            "annotations": self._annotations,
            # Convenience functions, named after their CLI counterparts.
            "dock": self.console_dock_run,
            "set_box_center": self.console_set_box_center,
            "set_box_size": self.console_set_box_size,
            "set_pose": self.set_pose,
            "measure": self.console_measure,
            "annotate": self.add_annotation,
            "frame_binding_site": lambda: self.viewport.frame_binding_site(clip=False),
            "frame_all": self.viewport.frame_all,
            "save_project": self.console_save_project,
            "load_receptor": self.load_receptor,
            "load_ligand": self.load_ligand,
            "load_poses": self.load_poses,
            "export_pymol": self.console_export_pymol,
            "select": self.console_select,
            "undo": self.undo,
            "redo": self.redo,
            "log": self._log,
            # Protocols, on the same code paths as the inspector tab: the console
            # is how a user reaches a run without a library browser.
            "current_protocol": self.current_protocol,
            "load_protocol": protocol.load_protocol,
            "save_protocol": protocol.save_protocol,
            "diff_protocols": protocol.diff_protocols,
            "run_protocol": protocol.run_protocol,
            "list_templates": protocol.list_templates,
        }
        namespace["help"] = self.console_help
        return namespace

    def console_help(self) -> str:
        """The bound names and helpers, as ``help()`` prints them."""
        names = sorted(self.console_namespace())
        return (
            tr("console.help_title")
            + "\n  "
            + "\n  ".join(names)
            + "\n"
            + tr("console.no_sandbox")
        )

    def _console_execute(self, source: str):
        """Run one console block: output and errors go into the transcript.

        The transcript *is* the log view, so the prompt line is echoed there and
        the captured stdout/stderr is appended to it — one place to read, one line
        to type. Everything it touches is the window's own API, so the log, the
        undo stack and the panels stay in sync, and an exception is reported
        instead of ending the session.
        """
        import contextlib
        import io
        import traceback

        namespace = getattr(self, "_console_globals", None)
        if namespace is None:
            namespace = self._console_globals = {"__name__": "__odock_console__"}
        namespace.update(self.console_namespace())
        out, err = io.StringIO(), io.StringIO()
        lines = source.splitlines()
        # Echo the block the way a REPL does: `>>>` on the first line and the
        # continuation prompt on the rest, so the transcript shows what was typed.
        if lines and lines[0].strip():
            self.log.appendPlainText(f"{dashboard.ConsoleInput.PROMPT}{lines[0]}")
            for extra in lines[1:]:
                self.log.appendPlainText(f"{dashboard.ConsoleInput.CONTINUED}{extra}")
        first = lines[0].strip() if lines else ""
        error = ""
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(source, "<odock-console>", "single"), namespace)
        except SystemExit:
            error = tr("console.error", message="SystemExit ignored")
        except Exception as exc:
            error = tr("console.error", message=f"{type(exc).__name__}: {exc}")
            if os.environ.get("ODOCK_CONSOLE_TRACEBACK"):
                error += "\n" + traceback.format_exc()
        captured = (out.getvalue() + err.getvalue()).rstrip("\n")
        if captured:
            self.log.appendPlainText(captured)
        if error:
            self.log.appendPlainText(error.rstrip("\n"))
        if first:
            self._log(tr("log.console", line=first))
        return captured, error

    # -- the convenience functions the console exposes ----------------------

    def console_dock_run(self, **kwargs) -> str:
        """``dock(exhaustiveness=8)``: start the run the Docking menu starts."""
        if kwargs.get("pose") is not None:
            self.set_pose(int(kwargs["pose"]), record=False)
        for widget, key in (
            (self.exhaustiveness, "exhaustiveness"),
            (self.poses, "poses"),
            (self.seed, "seed"),
        ):
            if kwargs.get(key) is not None:
                widget.setValue(int(kwargs[key]))
        if kwargs.get("scoring"):
            self.engine.setCurrentText(str(kwargs["scoring"]))
        self._run_docking()
        return "docking started"

    def console_set_box_center(self, x, y, z) -> str:
        """``set_box_center(1, 2, 3)``: move the search box, undoably."""
        box = self._current_box()
        size = box.size if box is not None else (20.0, 20.0, 20.0)
        spacing = box.spacing if box is not None else float(self.spacing.value())
        self._set_box((float(x), float(y), float(z)), size, spacing)
        return f"box centre {float(x)}, {float(y)}, {float(z)}"

    def console_set_box_size(self, x, y, z) -> str:
        """``set_box_size(20, 20, 20)``: resize the search box, undoably."""
        box = self._current_box()
        centre = box.center if box is not None else (0.0, 0.0, 0.0)
        spacing = box.spacing if box is not None else float(self.spacing.value())
        self._set_box(centre, (float(x), float(y), float(z)), spacing)
        return f"box size {float(x)}, {float(y)}, {float(z)}"

    def console_measure(self, kind: str = "distance", *refs):
        """``measure("angle", ("receptor", 0), ...)``: the same undoable measurement."""
        measurement = self.commit_measurement(str(kind), list(refs))
        if measurement is None:
            return "—"
        return dashboard.format_measurement(
            kind, self.measurement_value(measurement)
        )

    def console_select(self, chain, res_id, res_name) -> str:
        """``select("A", 195, "SER")``: select a residue on the ruler."""
        self.sequence.select_keys([(str(chain), int(res_id), str(res_name))])
        return f"selected {res_name}{res_id}"

    def console_save_project(self, path) -> str:
        """``save_project("x.json")``: the same file the File menu writes."""
        from pathlib import Path as _Path

        target = _Path(str(path))
        target.write_text(
            json.dumps(self._project_payload(), indent=2) + "\n", encoding="utf-8"
        )
        self._log(tr("log.saved_project", name=target.name))
        return f"saved {target}"

    def console_export_pymol(self, path) -> str:
        """``export_pymol("session.pml")``: the interop writer, when installed."""
        interop = _try_import("interop")
        if interop is None or not hasattr(interop, "pymol_script"):
            return "the interop module (PyMOL/ChimeraX export) is not available"
        script = interop.pymol_script(
            self.scene.receptor, self.scene.ligand, **self.export_viewer_state()
        )
        from pathlib import Path as _Path

        target = _Path(str(path))
        target.write_text(script, encoding="utf-8")
        return f"wrote {target}"

    # -- the instrument panels ----------------------------------------------

    def _build_dashboard(self) -> None:
        """The run monitor, docked to the right of the pose table.

        A tab widget holds the live run dashboard and the measurement history:
        both are "what the instrument has recorded", and both are wanted beside
        the poses rather than in a dialog.
        """
        dock = QtWidgets.QDockWidget(tr("dock.dashboard"), self)
        dock.setObjectName("dashboardDock")
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.dashboard_tabs = QtWidgets.QTabWidget()
        self.run_dashboard = dashboard.RunDashboard()
        self.run_dashboard.history = self.run_history
        self.run_dashboard.trace.set_history(list(self.run_history))
        self.measure_history = dashboard.MeasurementHistory()
        self.measure_history.clearRequested.connect(self._clear_measurements)
        self.measure_history.copyRequested.connect(self._copy_measurements)
        # The interaction lines the 3-D view is drawing, one row per line, so the
        # dashes stop being anonymous.
        self.interaction_table = dashboard.InteractionTable()
        self.interaction_table.copyRequested.connect(self._copy_interactions)
        # The labels pinned to the scene, so "what did I write on this pose?" has
        # an answer in the panel and not only in the view.
        self.annotation_table = dashboard.AnnotationTable()
        self.annotation_table.copyRequested.connect(self._copy_annotations)
        # Scroll areas, so the two panels compress (and scroll) rather than
        # dictating how narrow the window may become — they sit beside the pose
        # table, and their minimums would otherwise add up with its.
        self.dashboard_tabs.addTab(
            _scrollable_panel(self.run_dashboard), tr("tab.run")
        )
        self.dashboard_tabs.addTab(
            _scrollable_panel(self.measure_history), tr("tab.measure")
        )
        # The tab page is the *scroll area* around the panel, so its index is
        # remembered here: `indexOf(self.interaction_table)` would be -1 and the
        # tab title would never pick up the line count.
        self._interaction_tab_index = self.dashboard_tabs.addTab(
            _scrollable_panel(self.interaction_table), tr("tab.interactions")
        )
        self._annotation_tab_index = self.dashboard_tabs.addTab(
            _scrollable_panel(self.annotation_table), tr("menu.annotate")
        )
        self.dashboard_tabs.setCurrentIndex(0)
        dock.setWidget(self.dashboard_tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        # Beside the pose table, not on top of it: `splitDockWidget` puts the
        # dashboard to the right inside the bottom drawer, so the poses and the
        # run they came from are visible in the same glance.
        self.splitDockWidget(self.bottom_dock, dock, QtCore.Qt.Orientation.Horizontal)
        # An opening *proportion*, not a floor: `resizeDocks` is a hint the user
        # can drag away, `setMinimumWidth` would not be.
        self.resizeDocks(
            [self.bottom_dock, dock], [2, 1], QtCore.Qt.Orientation.Horizontal
        )
        # The drawer opens tall enough for the phase strip, the read-outs and most
        # of the trace: the panel is meant to be readable while a run is going on,
        # not only after the user drags the divider.
        self.resizeDocks([self.bottom_dock], [280], QtCore.Qt.Orientation.Vertical)
        self.dashboard_dock = dock

    def _build_comparison(self) -> None:
        """The two-pose panel, tabbed with the inspector."""
        dock = QtWidgets.QDockWidget(tr("dock.comparison"), self)
        dock.setObjectName("comparisonDock")
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.comparison = dashboard.PoseComparisonWidget()
        self.comparison.copyRequested.connect(self._copy_comparison)
        dock.setWidget(_scrollable_panel(self.comparison))
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.tabifyDockWidget(self.inspector_dock, dock)
        self.inspector_dock.raise_()
        self.comparison_dock = dock

    # -- construction -------------------------------------------------------

    def _build_viewport(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.viewport = ViewportWidget(self.scene)
        self.viewport.atomsPicked.connect(self._on_atoms_picked)
        self.viewport.bondPicked.connect(self._on_bond_picked)
        self.viewport.atomClicked.connect(self._on_atom_clicked)
        self.viewport.atomHovered.connect(self._on_atom_hovered)
        self.viewport.set_theme(self.color_theme)

        # The residue ruler sits directly under the 3-D view, always visible, so
        # the sequence and the picture share one glance.
        self.sequence = SequenceTrack()
        self.sequence.selectionChanged.connect(self._on_sequence_selection)
        self.sequence.residueActivated.connect(self._focus_residue)
        self.sequence_area = QtWidgets.QScrollArea()
        self.sequence_area.setObjectName("sequenceArea")
        self.sequence_area.setWidget(self.sequence)
        self.sequence_area.setWidgetResizable(True)
        self.sequence_area.setMinimumHeight(56)  # drag the handle for more
        self.sequence_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.sequence_area.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # The picture and the ruler share the central area through a splitter, so
        # the user can give either of them as much room as they want.
        self.central_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.central_splitter.setObjectName("centralSplitter")
        self.central_splitter.setChildrenCollapsible(False)
        self.central_splitter.setHandleWidth(6)
        self.central_splitter.addWidget(self.viewport)
        self.central_splitter.addWidget(self.sequence_area)
        self.central_splitter.setStretchFactor(0, 1)
        self.central_splitter.setStretchFactor(1, 0)
        layout.addWidget(self.central_splitter, 1)
        # The ruler needs little: its own row plus the scrollbar, and the rest to
        # the 3-D view (the user can drag the divider).
        self.central_splitter.setSizes([1, self.sequence.track_height() + 18])
        self.setCentralWidget(central)

        strip = QtWidgets.QFrame(self.viewport)
        strip.setObjectName("toolStrip")
        strip_layout = QtWidgets.QHBoxLayout(strip)
        strip_layout.setContentsMargins(6, 4, 6, 4)
        strip_layout.setSpacing(4)
        for text, tip, slot in (
            ("⟲", tr("tip.reset_view"), self.viewport.frame_all),
            ("◎", tr("tip.frame_ligand"), self.viewport.frame_ligand),
            ("▣", tr("tip.frame_box"), self.viewport.frame_box),
            ("↔", tr("tip.measure"), lambda: self._set_tool("measure")),
            ("🔗", tr("tip.bond"), lambda: self._set_tool("bond")),
            ("⤢", tr("tip.screenshot"), self._save_screenshot),
        ):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.setAutoRaise(True)
            button.clicked.connect(slot)
            strip_layout.addWidget(button)
        strip.adjustSize()
        self._tool_strip = strip

    def _build_workspace(self) -> None:
        dock = QtWidgets.QDockWidget(tr("dock.workspace"), self)
        dock.setObjectName("workspaceDock")
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels([tr("tree.project"), tr("tree.atoms")])
        # The tree is the left dock's content, so *it* decides how narrow that dock
        # can be. Fixed narrow columns plus elision mean a long project label
        # shortens instead of pushing the viewport aside; the full text is still on
        # the item (and in the row's tooltip).
        self.tree.setColumnWidth(0, 150)
        self.tree.setColumnWidth(1, 70)
        self.tree.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.tree.itemClicked.connect(self._on_tree_clicked)
        dock.setWidget(self.tree)
        dock.setMinimumWidth(self.DOCK_WIDTH_LEFT)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.workspace_dock = dock
        self._rebuild_tree()

    def _build_inspector(self) -> None:
        dock = QtWidgets.QDockWidget(tr("dock.inspector"), self)
        dock.setObjectName("inspectorDock")
        self.inspector = QtWidgets.QTabWidget()
        self.inspector.addTab(self._build_receptor_tab(), tr("tab.receptor"))
        self.inspector.addTab(self._build_ligand_tab(), tr("tab.ligand"))
        self.inspector.addTab(self._build_grid_tab(), tr("tab.grid"))
        self.inspector.addTab(self._build_engine_tab(), tr("tab.engine"))
        self.inspector.addTab(self._build_protocol_tab(), tr("tab.protocol"))
        # The view is rebuilt when the tab is opened rather than on every widget
        # change: the *document* is always captured live at Save/Compare time, so
        # only the on-screen list could ever be a moment stale.
        self.inspector.currentChanged.connect(self._on_inspector_tab_changed)
        # The Grid tab's three-column spin grid wants ~810 px on its own, which
        # would stop the user narrowing the inspector (and the window). Inside a
        # scroll area the panel is as narrow as the user likes and the wide tab
        # simply scrolls.
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("inspectorArea")
        scroll.setWidget(self.inspector)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        dock.setWidget(scroll)
        dock.setMinimumWidth(300)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.inspector_dock = dock

    def _build_receptor_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        box = QtWidgets.QGroupBox(tr("group.structure"))
        form = QtWidgets.QVBoxLayout(box)
        self.lbl_receptor = QtWidgets.QLabel(tr("label.no_receptor"))
        self.lbl_receptor.setWordWrap(True)
        form.addWidget(self.lbl_receptor)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (
            (tr("btn.load"), self._choose_receptor),
            (tr("btn.clean"), self._open_hetero_dialog),
            (tr("btn.protonate"), self._open_protonation_dialog),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        form.addLayout(row)
        self.chk_receptor = QtWidgets.QCheckBox(tr("chk.show_receptor"))
        self.chk_receptor.setChecked(self.scene.show_receptor)
        self.chk_receptor.toggled.connect(self._toggle_receptor)
        form.addWidget(self.chk_receptor)
        layout.addWidget(box)

        box = QtWidgets.QGroupBox(tr("group.flexible"))
        form = QtWidgets.QVBoxLayout(box)
        self.flex_list = QtWidgets.QListWidget()
        self.flex_list.setMaximumHeight(90)
        form.addWidget(self.flex_list)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton(tr("btn.add_residue"))
        add.clicked.connect(self._choose_flexible_residues)
        clear = QtWidgets.QPushButton(tr("btn.clear"))
        clear.clicked.connect(self.flex_list.clear)
        row.addWidget(add)
        row.addWidget(clear)
        form.addLayout(row)
        self.lbl_flex = QtWidgets.QLabel(tr("label.no_flex"))
        self.lbl_flex.setWordWrap(True)
        form.addWidget(self.lbl_flex)
        write = QtWidgets.QPushButton(tr("btn.write_flex"))
        write.clicked.connect(self._write_flexible_pdbqt)
        form.addWidget(write)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_ligand_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        box = QtWidgets.QGroupBox(tr("group.ligand"))
        form = QtWidgets.QVBoxLayout(box)
        self.lbl_ligand = QtWidgets.QLabel(tr("label.no_ligand"))
        self.lbl_ligand.setWordWrap(True)
        form.addWidget(self.lbl_ligand)
        row = QtWidgets.QHBoxLayout()
        for text, slot in (
            (tr("btn.load"), self._choose_ligand),
            (tr("btn.from_smiles"), self._ligand_from_smiles),
            (tr("btn.minimise"), self._minimise_ligand),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        form.addLayout(row)
        self.lbl_filters = QtWidgets.QLabel(tr("label.filters_none"))
        self.lbl_filters.setWordWrap(True)
        form.addWidget(self.lbl_filters)
        check = QtWidgets.QPushButton(tr("btn.drug_likeness"))
        check.clicked.connect(self._run_filters)
        form.addWidget(check)
        self.chk_ligand = QtWidgets.QCheckBox(tr("chk.show_ligand"))
        self.chk_ligand.setChecked(self.scene.show_ligand)
        self.chk_ligand.toggled.connect(self._toggle_ligand)
        form.addWidget(self.chk_ligand)
        layout.addWidget(box)

        box = QtWidgets.QGroupBox(tr("group.torsion"))
        form = QtWidgets.QVBoxLayout(box)
        self.lbl_bonds = QtWidgets.QLabel(tr("label.no_ligand_short"))
        self.lbl_bonds.setWordWrap(True)
        form.addWidget(self.lbl_bonds)
        row = QtWidgets.QHBoxLayout()
        detect = QtWidgets.QPushButton(tr("btn.detect_torsions"))
        detect.clicked.connect(self._detect_bonds)
        lock = QtWidgets.QPushButton(tr("btn.lock_bond"))
        lock.clicked.connect(lambda: self._set_tool("bond"))
        row.addWidget(detect)
        row.addWidget(lock)
        form.addLayout(row)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_grid_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        box = QtWidgets.QGroupBox(tr("group.search_box"))
        form = QtWidgets.QGridLayout(box)
        self.spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for row, key in enumerate(
            ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
        ):
            label = QtWidgets.QLabel(tr(f"grid.{key}"))
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-999.0, 999.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.5)
            spin.valueChanged.connect(self._on_box_spin)
            form.addWidget(label, row // 3, (row % 3) * 2)
            form.addWidget(spin, row // 3, (row % 3) * 2 + 1)
            self.spins[key] = spin
        self.spacing = QtWidgets.QDoubleSpinBox()
        self.spacing.setRange(0.1, 1.0)
        self.spacing.setDecimals(3)
        self.spacing.setSingleStep(0.025)
        self.spacing.setValue(0.375)
        self.spacing.setSuffix(tr("unit.angstrom"))
        form.addWidget(QtWidgets.QLabel(tr("grid.spacing")), 2, 0)
        form.addWidget(self.spacing, 2, 1)
        self.lbl_box_info = QtWidgets.QLabel("—")
        self.lbl_box_info.setWordWrap(True)
        form.addWidget(self.lbl_box_info, 3, 0, 1, 6)
        layout.addWidget(box)

        box = QtWidgets.QGroupBox(tr("group.pocket"))
        form = QtWidgets.QVBoxLayout(box)
        for text, slot in (
            (tr("btn.fit_ligand"), self._fit_box_to_ligand),
            (tr("btn.centre_ligand"), self._center_box_on_ligand),
            (tr("btn.align_residue"), self._box_from_residues),
            (tr("btn.whole_protein"), self._box_whole_protein),
            (tr("btn.detect_pockets"), self._detect_pockets),
        ):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(slot)
            form.addWidget(button)
        self.chk_box = QtWidgets.QCheckBox(tr("chk.show_box"))
        self.chk_box.setChecked(self._box_visible)
        self.chk_box.toggled.connect(self._toggle_box)
        form.addWidget(self.chk_box)
        self.btn_maps = QtWidgets.QPushButton(tr("btn.grid_info"))
        self.btn_maps.clicked.connect(self._compute_maps)
        form.addWidget(self.btn_maps)
        self.lbl_maps = QtWidgets.QLabel(tr("label.no_grid"))
        self.lbl_maps.setWordWrap(True)
        form.addWidget(self.lbl_maps)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_engine_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        box = QtWidgets.QGroupBox(tr("group.engine"))
        form = QtWidgets.QFormLayout(box)
        self.engine = QtWidgets.QComboBox()
        self.engine.addItems(["vina", "vinardo", "ad4"])
        form.addRow(tr("engine.force_field"), self.engine)
        self.search = QtWidgets.QComboBox()
        self.search.addItems(
            [
                tr("engine.search_mc"),
                tr("engine.search_lga"),
                tr("engine.search_lga_solis"),
            ]
        )
        form.addRow(tr("engine.search"), self.search)
        self.exhaustiveness = QtWidgets.QSpinBox()
        self.exhaustiveness.setRange(1, 512)
        self.exhaustiveness.setValue(8)
        form.addRow(tr("engine.exhaustiveness"), self.exhaustiveness)
        self.poses = QtWidgets.QSpinBox()
        self.poses.setRange(1, 100)
        self.poses.setValue(9)
        form.addRow(tr("engine.poses"), self.poses)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2**31 - 1)
        self.seed.setValue(42)
        form.addRow(tr("engine.seed"), self.seed)
        self.energy_range = QtWidgets.QDoubleSpinBox()
        self.energy_range.setRange(0.1, 20.0)
        self.energy_range.setValue(3.0)
        self.energy_range.setSuffix(tr("unit.kcal"))
        form.addRow(tr("engine.energy_range"), self.energy_range)
        self.islands = QtWidgets.QSpinBox()
        self.islands.setRange(1, 64)
        self.islands.setValue(4)
        form.addRow(tr("engine.islands"), self.islands)
        self.population = QtWidgets.QSpinBox()
        self.population.setRange(10, 2000)
        self.population.setValue(150)
        form.addRow(tr("engine.population"), self.population)
        self.generations = QtWidgets.QSpinBox()
        self.generations.setRange(1, 10000)
        self.generations.setValue(27)
        form.addRow(tr("engine.generations"), self.generations)
        self.use_grid = QtWidgets.QCheckBox(tr("engine.use_grid"))
        self.use_grid.setChecked(True)
        form.addRow("", self.use_grid)
        self.threads = QtWidgets.QSpinBox()
        self.threads.setRange(0, 256)
        self.threads.setValue(0)
        self.threads.setSpecialValueText(tr("engine.all_cores"))
        form.addRow(tr("engine.threads"), self.threads)
        layout.addWidget(box)

        box = QtWidgets.QGroupBox(tr("group.run"))
        form = QtWidgets.QVBoxLayout(box)
        self.btn_run = QtWidgets.QPushButton(tr("btn.start"))
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run_docking)
        form.addWidget(self.btn_run)
        row = QtWidgets.QHBoxLayout()
        self.btn_pause = QtWidgets.QPushButton(tr("btn.pause"))
        self.btn_pause.clicked.connect(self._pause_docking)
        self.btn_abort = QtWidgets.QPushButton(tr("btn.abort"))
        self.btn_abort.clicked.connect(self._abort_docking)
        self.btn_pause.setEnabled(False)
        self.btn_abort.setEnabled(False)
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_abort)
        form.addLayout(row)
        self.btn_score = QtWidgets.QPushButton(tr("btn.score"))
        self.btn_score.clicked.connect(self._score_current)
        form.addWidget(self.btn_score)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _build_bottom(self) -> None:
        dock = QtWidgets.QDockWidget(tr("dock.bottom"), self)
        dock.setObjectName("bottomDock")
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | QtCore.Qt.DockWidgetArea.TopDockWidgetArea
        )
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        poses = QtWidgets.QWidget()
        poses_layout = QtWidgets.QVBoxLayout(poses)
        poses_layout.setContentsMargins(0, 0, 0, 0)
        poses_layout.setSpacing(4)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            [
                tr("col.rank"),
                tr("col.energy"),
                tr("col.rmsd_lb"),
                tr("col.rmsd_ub"),
                tr("col.key_residues"),
            ]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        # Extended, not single: Ctrl-clicking a second row is how two poses are
        # put side by side in the comparison panel.
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.setMinimumHeight(90)
        poses_layout.addWidget(self.table, 1)

        # The pose read-out and the player. The slider that used to sit here is
        # gone: it cost a whole row of dock height, and the pose *table* above is
        # the way to change poses (with the arrow keys and Play). The summary now
        # spans the row instead of competing with a slider for its width, which is
        # what the wrap was already written for.
        row = QtWidgets.QHBoxLayout()
        self.lbl_pose = QtWidgets.QLabel(tr("label.no_poses"))
        # A plain, unwrapped QLabel reports its *whole* text as its minimum width,
        # so "mode 1 / 6 · affinity · rmsd · binding: <residues>" (1188 px on the
        # reference machine) silently became the minimum width of the entire
        # window. Wrapping it bounds that, keeps the text complete for callers that
        # read it back, and the tooltip repeats it in one line.
        self.lbl_pose.setWordWrap(True)
        self.lbl_pose.setMinimumWidth(0)
        self.lbl_pose.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.lbl_pose.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(self.lbl_pose, 1)
        self.btn_play = QtWidgets.QPushButton(tr("btn.play"))
        self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self._toggle_playback)
        row.addWidget(self.btn_play)
        poses_layout.addLayout(row)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumHeight(60)

        # Side by side by default: stacking the table over the log wasted the
        # dock's height. View ▸ Stack pose dock flips the orientation, and the
        # divider is drag-able either way.
        self.pose_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.pose_splitter.setObjectName("poseSplitter")
        self.pose_splitter.setChildrenCollapsible(False)
        self.pose_splitter.setHandleWidth(6)
        self.pose_splitter.addWidget(poses)
        self.pose_splitter.addWidget(self.log)
        self.pose_splitter.setStretchFactor(0, 3)
        self.pose_splitter.setStretchFactor(1, 2)
        self.pose_splitter.setSizes([720, 520])
        layout.addWidget(self.pose_splitter, 1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        dock.setWidget(widget)
        dock.setMinimumHeight(140)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.bottom_dock = dock

    def _build_selection_panel(self) -> None:
        """The dock that lists the atoms the ruler selection covers."""
        dock = QtWidgets.QDockWidget(tr("dock.selection"), self)
        dock.setObjectName("selectionDock")
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 4, 6, 4)

        self.lbl_selection_summary = QtWidgets.QLabel(tr("label.no_selection"))
        self.lbl_selection_summary.setWordWrap(True)
        layout.addWidget(self.lbl_selection_summary)

        self.selection_table = QtWidgets.QTableWidget(0, 9)
        self.selection_table.setHorizontalHeaderLabels(
            [
                tr("col.atom_name"),
                tr("col.element"),
                tr("col.residue_label"),
                tr("col.chain"),
                tr("col.x"),
                tr("col.y"),
                tr("col.z"),
                tr("col.ad_type"),
                tr("col.charge"),
            ]
        )
        self.selection_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.selection_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.selection_table.verticalHeader().setVisible(False)
        self.selection_table.setMinimumHeight(120)
        layout.addWidget(self.selection_table, 1)

        self.btn_copy_pdbqt = QtWidgets.QPushButton(tr("btn.copy_pdbqt"))
        self.btn_copy_pdbqt.clicked.connect(self._copy_selection_pdbqt)
        layout.addWidget(self.btn_copy_pdbqt)

        dock.setWidget(widget)
        dock.setMinimumWidth(300)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.selection_dock = dock

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        self.lbl_status = QtWidgets.QLabel(tr("status.ready"))
        bar.addWidget(self.lbl_status, 1)
        self.lbl_energy = QtWidgets.QLabel("")
        bar.addPermanentWidget(self.lbl_energy)

    # -- menus --------------------------------------------------------------

    def _build_measure_menu(self, view: QtWidgets.QMenu) -> None:
        """View ▸ Measure: what the next atom picks will answer.

        Radio entries, so the choice is visible, and every one of them is a menu
        action — which is why the command palette can reach all of them for free
        and every measurement is available without the mouse.
        """
        menu = view.addMenu(tr("menu.measure"))
        group = QtGui.QActionGroup(menu)
        group.setExclusive(True)
        self._measure_actions: Dict[str, QtGui.QAction] = {}
        for kind in dashboard.MEASUREMENT_ORDER:
            label = dashboard.measurement_kind_label(kind)
            count = dashboard.measurement_expected_atoms(kind)
            action = QtGui.QAction(label, menu)
            action.setCheckable(True)
            action.setChecked(kind == self._measure_kind)
            action.setActionGroup(group)
            action.setStatusTip(tr("status.measure_pick", kind=label, have=0, need=count))
            action.triggered.connect(
                lambda _=False, value=kind: self.set_measure_kind(value)
            )
            menu.addAction(action)
            self._measure_actions[kind] = action
        menu.addSeparator()
        self._act(
            menu, tr("action.measure_selection"), self.measure_selection
        )

    def _build_annotate_menu(self, view: QtWidgets.QMenu) -> None:
        """View ▸ Annotate: text labels pinned to the scene.

        The colour is chosen in the label dialog, so there is no separate
        colour action and no hidden state: one dialog edits text *and* colour.
        """
        menu = view.addMenu(tr("menu.annotate"))
        self._act(menu, tr("action.annotate_add"), self._add_annotation_action)
        self._act(menu, tr("action.annotate_edit"), self._edit_annotation_action)
        self._act(menu, tr("action.annotate_delete"), self._delete_annotation_action)
        menu.addSeparator()
        self.annotate_action = self._act(
            menu,
            tr("action.annotate_show"),
            self.toggle_annotations,
            checkable=True,
        )
        # Setting the initial tick must not fire the slot: the menus are built
        # before the status bar, and the slot logs.
        self.annotate_action.blockSignals(True)
        self.annotate_action.setChecked(self._annotation_visible)
        self.annotate_action.blockSignals(False)

    def _act(self, menu, text, slot, *, shortcut=None, tip=None, checkable=False):
        # Parented to the menu, so tearing the menu down on a language switch
        # takes its actions (and their shortcuts) with it.
        action = QtGui.QAction(text, menu)
        if shortcut:
            action.setShortcut(shortcut)
        if tip:
            action.setStatusTip(tip)
        action.setCheckable(checkable)
        if checkable:
            action.toggled.connect(slot)
        else:
            action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu(tr("menu.file"))
        self._act(
            file_menu, tr("action.new_project"), self._new_project, shortcut="Ctrl+N"
        )
        self._act(
            file_menu, tr("action.open_project"), self._open_project, shortcut="Ctrl+O"
        )
        self._act(
            file_menu, tr("action.save_project"), self._save_project, shortcut="Ctrl+S"
        )
        file_menu.addSeparator()
        self._act(file_menu, tr("action.import_receptor"), self._choose_receptor)
        self._act(file_menu, tr("action.import_ligand"), self._choose_ligand)
        self._act(file_menu, tr("action.import_poses"), self._choose_poses)
        self._act(file_menu, tr("action.fetch_pdb"), self._fetch_pdb)
        file_menu.addSeparator()
        export = file_menu.addMenu(tr("menu.export"))
        self._act(
            export, tr("action.export_receptor"), lambda: self._export("receptor_pdbqt")
        )
        self._act(export, tr("action.export_ligand"), lambda: self._export("ligand_pdbqt"))
        self._act(export, tr("action.export_poses"), lambda: self._export("poses_pdbqt"))
        self._act(export, tr("action.export_cleaned"), lambda: self._export("cleaned_pdb"))
        self._act(
            export, tr("action.export_native"), lambda: self._export("native_ligand")
        )
        export.addSeparator()
        self._act(export, tr("action.export_gpf"), lambda: self._export("gpf"))
        self._act(export, tr("action.export_dpf"), lambda: self._export("dpf"))
        self._act(export, tr("action.export_config"), lambda: self._export("config"))
        export.addSeparator()
        self._act(export, tr("action.export_xlsx"), lambda: self._export("xlsx"))
        self._act(export, tr("action.export_csv"), lambda: self._export("csv"))
        self._act(export, tr("action.export_svg"), lambda: self._export("svg"))
        self._act(export, tr("action.export_screenshot"), self._save_screenshot)
        export.addSeparator()
        # The interop exports leave the program entirely, so they live at the
        # bottom of the same submenu the other "give me a file" actions do.
        self._act(export, tr("action.export_pymol"), lambda: self._export_scene("pymol"))
        self._act(
            export, tr("action.export_chimerax"), lambda: self._export_scene("chimerax")
        )
        self._act(
            export, tr("action.export_pose_pdb"), lambda: self._export_scene("pose")
        )
        self._act(
            export, tr("action.export_surface_obj"), lambda: self._export_scene("surface")
        )
        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu(tr("menu.recent"))
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
        self._rebuild_recent_menu()
        self.restore_action = self._act(
            file_menu,
            tr("action.restore_session"),
            self.restore_session,
            shortcut="Ctrl+Shift+R",
        )
        file_menu.addSeparator()
        self._act(file_menu, tr("action.quit"), self.close, shortcut="Ctrl+Q")

        receptor = bar.addMenu(tr("menu.receptor"))
        self._act(receptor, tr("action.remove_waters"), self._remove_waters)
        self._act(receptor, tr("action.strip_ligand"), self._strip_ligand)
        self._act(receptor, tr("action.hetero"), self._open_hetero_dialog)
        receptor.addSeparator()
        self._act(receptor, tr("action.protonation"), self._open_protonation_dialog)
        self._act(receptor, tr("action.assign_charges"), self._assign_charges)
        receptor.addSeparator()
        self._act(
            receptor, tr("action.flexible_residues"), self._choose_flexible_residues
        )
        self._act(receptor, tr("action.write_flex"), self._write_flexible_pdbqt)

        ligand = bar.addMenu(tr("menu.ligand"))
        self._act(ligand, tr("action.from_smiles"), self._ligand_from_smiles)
        self._act(ligand, tr("action.minimise"), self._minimise_ligand)
        self._act(ligand, tr("action.ligand_charges"), self._ligand_charges)
        self._act(ligand, tr("action.detect_torsions"), lambda: self._detect_bonds())
        self._act(ligand, tr("action.lock_bond"), lambda: self._set_tool("bond"))
        ligand.addSeparator()
        self._act(ligand, tr("action.drug_likeness"), self._run_filters)

        grid = bar.addMenu(tr("menu.grid"))
        self._act(grid, tr("action.fit_box"), self._fit_box_to_ligand)
        self._act(grid, tr("action.centre_box"), self._center_box_on_ligand)
        self._act(grid, tr("action.align_residue"), self._box_from_residues)
        self._act(grid, tr("action.detect_pockets"), self._detect_pockets)
        grid.addSeparator()
        self.box_action = self._act(
            grid, tr("action.show_box"), self._toggle_box, checkable=True
        )
        self.box_action.setChecked(self._box_visible)
        # The box is read-only in the 3-D view, so this dialog plus the centre /
        # size / spacing fields are how the user controls it.
        self._act(grid, tr("action.box_opacity"), self._choose_box_opacity)
        self._act(grid, tr("action.grid_info"), self._compute_maps)

        docking = bar.addMenu(tr("menu.docking"))
        for key, value in (
            ("action.use_vina", "vina"),
            ("action.use_vinardo", "vinardo"),
            ("action.use_ad4", "ad4"),
        ):
            action = QtGui.QAction(tr(key), docking)
            action.triggered.connect(
                lambda _=False, v=value: self.engine.setCurrentText(v)
            )
            docking.addAction(action)
        docking.addSeparator()
        self._act(docking, tr("action.start"), self._run_docking, shortcut="F5")
        self._act(docking, tr("action.pause_resume"), self._toggle_pause, shortcut="F6")
        self._act(docking, tr("action.abort"), self._abort_docking, shortcut="F7")
        self._act(docking, tr("action.score"), self._score_current)
        self._act(
            docking,
            tr("action.engine_settings"),
            lambda: self.inspector.setCurrentIndex(3),
        )

        analysis = bar.addMenu(tr("menu.analysis"))
        # Lambdas swallow the ``checked`` bool QAction.triggered emits: these two
        # slots take a ``quiet`` flag first, so a bare connection would silently
        # rely on the argument that happens to arrive matching the default.
        self._act(
            analysis, tr("action.show_interactions"), lambda: self._annotate_interactions()
        )
        self._act(analysis, tr("action.clear_annotations"), self._clear_interactions)
        self._act(analysis, tr("action.cluster_poses"), self._cluster_poses)
        self._act(analysis, tr("action.compare_poses"), self._compare_selected)
        analysis.addSeparator()
        self._act(analysis, tr("action.diagram_svg"), lambda: self._export("svg"))
        # The lambda swallows the ``checked`` bool QAction.triggered emits: this
        # action is not checkable, so connected straight to the method it would
        # arrive as ``checked=False`` and stop the player instead of starting it.
        self._act(analysis, tr("action.play_poses"), lambda: self._toggle_playback(True))
        analysis.addSeparator()
        self._act(analysis, tr("action.export_xlsx"), lambda: self._export("xlsx"))
        self._act(analysis, tr("action.export_csv"), lambda: self._export("csv"))
        analysis.addSeparator()
        # The three measurements that turn "it fits" into a number.
        self._act(analysis, tr("action.sasa_report"), self._sasa_report)
        self._act(analysis, tr("action.ligand_burial"), self._ligand_burial)
        self._act(analysis, tr("action.burial_per_pose"), self._burial_per_pose)

        view = bar.addMenu(tr("menu.view"))
        self._style_actions = {
            "receptor": self._build_style_menu(
                view, tr("menu.protein_style"), PROTEIN_STYLES, "receptor"
            ),
            "ligand": self._build_style_menu(
                view, tr("menu.ligand_style"), LIGAND_STYLES, "ligand"
            ),
        }
        self._build_surface_menu(view)
        view.addSeparator()
        # The palette is the fastest way to reach any of the actions below, so
        # it sits at the top of the View menu where the eye lands first.
        self._act(
            view, tr("action.palette"), self._open_palette, shortcut="Ctrl+K"
        )
        self._act(view, tr("action.bond_check"), self._open_bond_check)
        self._act(
            view,
            tr("action.interaction_distances"),
            self._choose_interaction_distances,
        )
        self._build_interaction_lines_menu(view)
        view.addSeparator()
        self.axes_action = self._act(
            view, tr("action.axes"), self._toggle_axes, checkable=True
        )
        self.axes_action.setChecked(self.scene.show_axes)
        self.ssao_action = self._act(
            view, tr("action.ssao"), self._toggle_ssao, checkable=True
        )
        self.ssao_action.setChecked(self.scene.ssao)
        self._act(view, tr("action.measure"), self._start_measure_distance)
        self._build_measure_menu(view)
        self._build_annotate_menu(view)
        self.undo_action = self._act(
            view, tr("action.undo"), self.undo, shortcut="Ctrl+Z"
        )
        self.redo_action = self._act(
            view, tr("action.redo"), self.redo, shortcut="Ctrl+Shift+Z"
        )
        self._sync_history_actions()
        self._act(view, tr("action.clear_measurements"), self._clear_measurements)
        self.ghost_action = self._act(
            view, tr("action.show_reference"), self._toggle_ghost, checkable=True
        )
        self.ghost_action.setChecked(self._ghost_visible)
        view.addSeparator()
        panels = view.addMenu(tr("menu.panels"))
        for dock in (
            self.workspace_dock,
            self.inspector_dock,
            self.bottom_dock,
            self.dashboard_dock,
            self.comparison_dock,
            self.selection_dock,
        ):
            panels.addAction(dock.toggleViewAction())
        panels.addSeparator()
        self._act(panels, tr("action.console_clear"), self._clear_console)
        self.stack_action = self._act(
            view, tr("action.stack_pose_dock"), self._set_pose_split, checkable=True
        )
        self.stack_action.setChecked(self._pose_stacked)
        view.addSeparator()
        self._build_appearance_menu(view)
        self.inspect_action = self._act(
            view,
            tr("action.inspect_atom"),
            self._toggle_inspect,
            checkable=True,
        )
        self.inspect_action.setChecked(True)
        self._act(view, tr("action.copy_view"), self._copy_view, shortcut="Ctrl+Shift+C")
        view.addSeparator()
        # The lambda swallows the ``checked`` bool that QAction.triggered emits:
        # connected straight to the method, it would arrive as ``radius=False``
        # and clip the receptor to a 0 Å ball (an empty view from the menu). The
        # cut-away is only ever enabled on this explicit request.
        self._act(
            view,
            tr("action.frame_site"),
            lambda: self.viewport.frame_binding_site(clip=True),
        )
        self._act(
            view, tr("action.reset_view"), self.viewport.frame_all, shortcut="Home"
        )
        view.addSeparator()
        self._build_language_menu(view)

    def _build_language_menu(self, view: QtWidgets.QMenu) -> None:
        """The radio-style language picker at the end of the View menu.

        The switch is deferred by one event-loop turn because rebuilding the
        window destroys the menu whose action is still being delivered.
        """
        language = view.addMenu(tr("menu.language"))
        group = QtGui.QActionGroup(language)
        group.setExclusive(True)
        current = i18n.current_language()
        for code in i18n.LANGUAGES:
            action = QtGui.QAction(i18n.language_name(code), language)
            action.setCheckable(True)
            action.setChecked(code == current)
            action.setActionGroup(group)
            action.triggered.connect(
                lambda _=False, c=code: QtCore.QTimer.singleShot(
                    0, lambda: self.set_language(c)
                )
            )
            language.addAction(action)

    def _build_interaction_lines_menu(self, view: QtWidgets.QMenu) -> None:
        """View ▸ Interaction lines: which contact lines are drawn.

        A docked pose can carry a lot of near-parallel hydrophobic dashes, and a
        wall of them hides the two or three contacts the user actually cares
        about. Each kind gets a tick, the choice is remembered by the session and
        across a language switch, and the Interactions panel always states how
        many lines the filter is holding back.
        """
        menu = view.addMenu(tr("menu.interaction_lines"))
        self._interaction_actions: Dict[str, QtGui.QAction] = {}
        for kind in INTERACTION_COLORS:
            action = QtGui.QAction(_interaction_label(kind), menu)
            action.setCheckable(True)
            action.setChecked(True)
            action.setStatusTip(tr(f"interaction.{kind}"))
            action.toggled.connect(self._apply_interaction_filter)
            menu.addAction(action)
            self._interaction_actions[kind] = action
        menu.addSeparator()
        self._act(menu, tr("action.interaction_all"), self._show_all_interactions)

    def _show_all_interactions(self, _checked: bool = False) -> None:
        for action in getattr(self, "_interaction_actions", {}).values():
            action.setChecked(True)
        self._apply_interaction_filter()

    def _build_appearance_menu(self, view: QtWidgets.QMenu) -> None:
        """View ▸ Theme / Density / Layout: the appearance and arrangement."""
        theme_menu = view.addMenu(tr("menu.theme"))
        group = QtGui.QActionGroup(theme_menu)
        group.setExclusive(True)
        self._theme_actions: Dict[str, QtGui.QAction] = {}
        for name, key in (("dark", "action.theme_dark"), ("light", "action.theme_light")):
            action = QtGui.QAction(tr(key), theme_menu)
            action.setCheckable(True)
            action.setChecked(name == self.color_theme.name)
            action.setActionGroup(group)
            action.triggered.connect(lambda _=False, n=name: self.set_theme(n))
            theme_menu.addAction(action)
            self._theme_actions[name] = action

        density_menu = view.addMenu(tr("menu.density"))
        group = QtGui.QActionGroup(density_menu)
        group.setExclusive(True)
        self._density_actions: Dict[str, QtGui.QAction] = {}
        for name, key in (
            ("comfortable", "action.density_comfortable"),
            ("compact", "action.density_compact"),
        ):
            action = QtGui.QAction(tr(key), density_menu)
            action.setCheckable(True)
            action.setChecked(name == self.density)
            action.setActionGroup(group)
            action.triggered.connect(lambda _=False, n=name: self.set_density(n))
            density_menu.addAction(action)
            self._density_actions[name] = action

        layout_menu = view.addMenu(tr("menu.layout"))
        group = QtGui.QActionGroup(layout_menu)
        group.setExclusive(True)
        self._layout_actions: Dict[str, QtGui.QAction] = {}
        for name, key in (
            ("docking", "action.layout_docking"),
            ("analysis", "action.layout_analysis"),
            ("compare", "action.layout_compare"),
        ):
            action = QtGui.QAction(tr(key), layout_menu)
            action.setCheckable(True)
            action.setChecked(name == self._preset)
            action.setActionGroup(group)
            action.triggered.connect(lambda _=False, n=name: self._apply_layout_preset(n))
            layout_menu.addAction(action)
            self._layout_actions[name] = action

    # -- appearance ---------------------------------------------------------

    def set_theme(self, name: str, *, record: bool = True) -> None:
        """Switch between the dark and the light theme."""
        theme = dashboard.theme_named(name)
        before = self.color_theme.name
        self.color_theme = theme
        self._apply_style()
        self._sync_appearance_actions()
        if record and before != theme.name:
            self._record(
                tr("menu.theme"),
                {"theme": before},
                {"theme": theme.name},
            )
        self._log(
            tr(
                "log.theme",
                theme=tr(f"action.theme_{theme.name}"),
                density=tr(f"action.density_{self.density}"),
            )
        )

    def set_density(self, name: str, *, record: bool = True) -> None:
        """Comfortable or compact: padding, tab sizes and font size."""
        before = self.density
        self.density = name if name in dashboard.DENSITIES else "comfortable"
        self._apply_style()
        self._sync_appearance_actions()
        if record and before != self.density:
            self._record(
                tr("menu.density"),
                {"density": before},
                {"density": self.density},
            )
        self._log(
            tr(
                "log.theme",
                theme=tr(f"action.theme_{self.color_theme.name}"),
                density=tr(f"action.density_{self.density}"),
            )
        )

    def _sync_appearance_actions(self) -> None:
        """Tick the radio entries that match the live appearance."""
        for name, action in getattr(self, "_theme_actions", {}).items():
            action.setChecked(name == self.color_theme.name)
        for name, action in getattr(self, "_density_actions", {}).items():
            action.setChecked(name == self.density)
        for name, action in getattr(self, "_layout_actions", {}).items():
            action.setChecked(name == self._preset)

    def _apply_layout_preset(self, name: str) -> None:
        """Arrange the docks for a job: docking, analysis or comparison."""
        preset = dashboard.layout_preset(name)
        self._preset = str(name).lower()
        for key, dock_name in (
            ("workspace", "workspace_dock"),
            ("inspector", "inspector_dock"),
            ("bottom", "bottom_dock"),
            ("dashboard", "dashboard_dock"),
            ("comparison", "comparison_dock"),
            ("selection", "selection_dock"),
        ):
            dock = getattr(self, dock_name, None)
            if dock is not None:
                dock.setVisible(bool(preset["docks"].get(key, True)))
        if self.inspector_dock is not None and self.inspector_dock.isVisible():
            self.inspector.setCurrentIndex(int(preset["inspector_tab"]))
        self._set_pose_split(bool(preset["pose_stacked"]))
        if preset["raise_dashboard"] and self.dashboard_dock is not None:
            self.dashboard_dock.raise_()
        if name == "compare" and self.comparison_dock is not None:
            self.comparison_dock.raise_()
        # Hiding or showing a dock makes Qt re-lay the whole dock area out from
        # its size hints, which collapses the wide right dock (measured: 560 →
        # 320 px). The proportions are re-stated, on the next turn.
        self._schedule_dock_proportions()
        self._sync_appearance_actions()
        self._log(tr("log.preset", name=tr(f"action.layout_{self._preset}")))
        self._session_changed()

    # -- styling ------------------------------------------------------------

    def _apply_style(self) -> None:
        """The theme and density of the whole window, viewport included."""
        self.setStyleSheet(dashboard.stylesheet(self.color_theme, self.density))
        viewport = getattr(self, "viewport", None)
        if viewport is not None:
            viewport.set_theme(self.color_theme)
        panel = getattr(self, "run_dashboard", None)
        if panel is not None:
            panel.set_theme(self.color_theme)
        ruler = getattr(self, "sequence", None)
        if ruler is not None:
            # The ruler paints its own background, so it cannot inherit this
            # sheet: it is handed the palette that matches the theme.
            ruler.set_palette(
                sequence_module.LIGHT_RULER
                if self.color_theme.name == "light"
                else sequence_module.DARK_RULER
            )

    # -- logging ------------------------------------------------------------

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)
        self.lbl_status.setText(message if len(message) < 120 else message[:117] + "…")
        # Anything worth telling the user is worth remembering: this one hook
        # makes loading, the box, the engine settings, the selection and every
        # menu action restore after a restart, without a save call at each site.
        self._session_changed()

    #: Kept for callers that still ask for it: the read-out used to be capped at
    #: this width while a pose slider shared its row. The slider is gone, so the
    #: label now expands to the whole row and this is only a *documented*
    #: comfortable reading width, not a limit the widget applies.
    POSE_LABEL_WIDTH = 420

    def _set_pose_label(self, text: str) -> None:
        """Show ``text`` in the pose read-out, wrapped, with a one-line tooltip.

        The label sits in the bottom drawer, so an unbounded one silently decides
        how narrow the window may become. The cap plus word wrap bounds that, the
        text itself stays complete (``lbl_pose.text()`` and
        :meth:`pose_label_text` return every character), and the tooltip repeats
        it unwrapped for a quick hover read.
        """
        text = str(text or "")
        self._pose_label_text = text
        self.lbl_pose.setText(text)
        self.lbl_pose.setToolTip(text)

    def pose_label_text(self) -> str:
        """The pose read-out in full, whatever the label is showing."""
        return getattr(self, "_pose_label_text", self.lbl_pose.text())

    # -- project ------------------------------------------------------------

    @_history_silent
    def _new_project(self) -> None:
        self.scene.receptor = []
        self.scene.ligand = []
        self.scene.interactions = []
        self.scene.highlight = []
        self.scene.locked_bonds = []
        self.scene.measurements = []
        self.scene.box = None
        self.receptor_text = self.ligand_text = None
        self.ligand_models = []
        self.pose_models = []
        self.result = None
        self.pockets = []
        self.clusters = []
        self.interactions = []
        self.scene.interaction_focus = []
        self._focus_requested = False
        self.receptor_mol = self.ligand_mol = None
        self.inventory = None
        self._reference_pose = None
        self._measurements = []
        self._locked_bonds = []
        self.table.setRowCount(0)
        self.poses_text = None
        self._maps_computed = False
        self._filter_verdict = None
        self.lbl_receptor.setText(tr("label.no_receptor"))
        self.lbl_ligand.setText(tr("label.no_ligand"))
        self._set_pose_label(tr("label.no_poses"))
        self.lbl_box_info.setText("—")
        self.lbl_maps.setText(tr("label.no_grid"))
        self.lbl_filters.setText(tr("label.filters_none"))
        self.lbl_bonds.setText(tr("label.no_ligand_short"))
        self.lbl_flex.setText(tr("label.no_flex"))
        self.flex_list.clear()
        self.sequence.clear()
        # A new project has no box: untick the widgets (signals blocked) so the
        # next rebuild cannot push the placeholder spin values into the scene and
        # draw a stray cube with six handles on an empty view.
        self._box_visible = False
        for widget in (
            getattr(self, "chk_box", None),
            getattr(self, "box_action", None),
        ):
            if widget is not None and widget.isChecked():
                widget.blockSignals(True)
                widget.setChecked(False)
                widget.blockSignals(False)
        self.viewport.refresh(upload_receptor=True)
        self._rebuild_tree()
        # A new project is a new instrument state: the dashboard forgets the
        # runs, the comparison and the measurements of the old one.
        self.run_history.clear()
        self._comparison = None
        if self.run_dashboard is not None:
            self.run_dashboard.history = self.run_history
            self.run_dashboard.reset()
        if self.comparison is not None:
            self.comparison.clear()
        self._measurements = []
        self._annotations = []
        self._pick_refs = []
        self._history.clear()
        self._sync_measurements()
        self._sync_annotations()
        self._sync_history_actions()
        self._log(tr("log.new_project"))

    def _project_payload(self) -> dict:
        """The project file's contents — one writer, two callers (menu, console)."""
        box = self._current_box()
        return {
            "version": 1,
            "language": i18n.current_language(),
            "receptor": self._pending.get("receptor"),
            "ligand": self._pending.get("ligand"),
            "poses": self._pending.get("poses"),
            "box": (
                {
                    "center": list(box.center),
                    "size": list(box.size),
                    "spacing": float(box.spacing),
                }
                if box is not None
                else None
            ),
            "engine": {
                "scoring": self.engine.currentText(),
                "exhaustiveness": self.exhaustiveness.value(),
                "num_poses": self.poses.value(),
                "seed": self.seed.value(),
                "search": self.search.currentIndex(),
            },
            "flexible_residues": [
                self.flex_list.item(i).text() for i in range(self.flex_list.count())
            ],
            "locked_bonds": [list(b) for b in self._locked_bonds],
        }

    def _save_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            tr("dialog.save_project"),
            "odock-project.json",
            tr("filter.project"),
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps(self._project_payload(), indent=2) + "\n", encoding="utf-8"
        )
        self._log(tr("log.saved_project", name=Path(path).name))

    @_history_silent
    def _open_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("dialog.open_project"), "", tr("filter.project")
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), True)
            return
        # The project remembers the language it was written in.
        language = payload.get("language")
        if isinstance(language, str) and language in i18n.LANGUAGES:
            self.set_language(language)
        self._new_project()
        if payload.get("receptor"):
            self.load_receptor(payload["receptor"])
        if payload.get("ligand"):
            self.load_ligand(payload["ligand"])
        if payload.get("poses"):
            self.load_poses(payload["poses"])
        box = payload.get("box")
        if isinstance(box, dict) and "center" in box:
            self._set_box(box["center"], box["size"], box.get("spacing", 0.375))
        for name in payload.get("flexible_residues", []):
            self.flex_list.addItem(str(name))
        self._locked_bonds = [tuple(b) for b in payload.get("locked_bonds", [])]
        self.scene.locked_bonds = list(self._locked_bonds)
        engine = payload.get("engine") or {}
        if engine.get("scoring"):
            self.engine.setCurrentText(str(engine["scoring"]))
        self._log(tr("log.opened_project", name=Path(path).name))

    # -- loading ------------------------------------------------------------

    def _choose_receptor(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            tr("dialog.open_receptor"),
            "",
            tr("filter.structures"),
        )
        if path:
            self.load_receptor(path, interactive=True)

    def _choose_ligand(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            tr("dialog.open_ligand"),
            "",
            tr("filter.ligands"),
        )
        if path:
            self.load_ligand(path, interactive=True)

    def _choose_poses(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, tr("dialog.open_poses"), "", tr("filter.pdbqt")
        )
        if path:
            self.load_poses(path, interactive=True)

    def _report_error(self, message: str, interactive: bool) -> None:
        """Surface a failure without ever blocking a scripted run."""
        self._log(tr("log.error", message=message))
        self.statusBar().showMessage(message[:200])
        if interactive:
            QtWidgets.QMessageBox.critical(self, tr("app.name"), message)

    @staticmethod
    def _read_pdbqt(source) -> Tuple[str, str]:
        """Accept either a path or the structure text itself."""
        if hasattr(source, "read_text"):
            return (
                source.read_text(encoding="utf-8", errors="replace"),
                Path(source).name,
            )
        text = str(source)
        head = text.lstrip()[:16].upper()
        if head.startswith(("ATOM", "HETATM", "ROOT", "MODEL", "REMARK", "TER")):
            return text, f"<{len(text.splitlines())} lines>"
        path = Path(text)
        return path.read_text(encoding="utf-8", errors="replace"), path.name

    @staticmethod
    def _remember_path(path) -> Optional[str]:
        try:
            candidate = Path(str(path))
            return str(candidate) if candidate.exists() else None
        except (OSError, ValueError):
            return None

    @_history_silent
    def load_receptor(self, path, interactive: bool = False) -> bool:
        try:
            text, label = self._read_pdbqt(path)
            models = parse_pdbqt(text)
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), interactive)
            return False
        self.receptor_text = text
        self._pending["receptor"] = self._remember_path(path)
        self._note_recent(path)
        self.scene.receptor = models[0].atoms
        self.scene.receptor_bonds = _perceived_bonds(self.scene.receptor, "receptor")
        self.sequence.set_structure(self.scene.receptor, "receptor")
        self.receptor_mol = self._rdkit_mol(text)
        self.inventory = self._classify(text)
        self.lbl_receptor.setText(
            tr("label.structure_n", name=label, n=len(self.scene.receptor))
        )
        self._log(tr("log.receptor", n=len(self.scene.receptor), name=label))
        if self.inventory is not None:
            summary = {
                key: value
                for key, value in self.inventory.summary().items()
                if key != "solvent"
            }
            if summary:
                self._log(
                    tr(
                        "log.hetero_groups",
                        groups=", ".join(
                            f"{k}={v}" for k, v in sorted(summary.items())
                        ),
                    )
                )
        self.viewport.refresh(upload_receptor=True)
        if not self.scene.ligand:
            self.viewport.frame_all()
        self._rebuild_tree()
        self._fit_sequence_area()
        return True

    @_history_silent
    def load_ligand(self, path, interactive: bool = False) -> bool:
        try:
            text, label = self._read_pdbqt(path)
            models = parse_pdbqt(text)
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), interactive)
            return False
        self.ligand_text = text
        self._pending["ligand"] = self._remember_path(path)
        self._note_recent(path)
        self.ligand_models = models
        self.scene.ligand = models[0].atoms
        self.scene.ligand_bonds = _perceived_bonds(self.scene.ligand, "ligand")
        self.sequence.set_structure(self.scene.ligand, "ligand")
        self.ligand_mol = self._rdkit_mol(text)
        self.lbl_ligand.setText(
            tr("label.structure_n", name=label, n=len(self.scene.ligand))
        )
        self._log(tr("log.ligand", n=len(self.scene.ligand), name=label))
        self.lbl_filters.setText(tr("label.filters_none"))
        self._detect_bonds(quiet=True)
        # The imported ligand is the pose on screen until a pose file arrives, so
        # the ruler marks the residues it touches — but only while the interaction
        # drawing is on: loading a structure must not draw contacts by itself.
        if self._interactions_shown:
            self._mark_pose_contacts()
        self.viewport.refresh()
        if self.scene.receptor:
            # Together with the receptor that surrounds it. A 15 Å close-up of the
            # ligand on its own puts the camera *inside* the box that is about to
            # be fitted around it, where the translucent fill buries the whole
            # structure; framing the entire protein instead would leave the ligand
            # a speck. This is the pocket view: the ligand plus the residues
            # within a few Å of it, without the cut-away.
            self.viewport.frame_binding_site(clip=False)
        else:
            self.viewport.frame_ligand()
        if self.receptor_text and self._box_is_default():
            self._fit_box_to_ligand()
        self._rebuild_tree()
        self._fit_sequence_area()
        return True

    @_history_silent
    def load_poses(self, path, interactive: bool = False) -> bool:
        try:
            text, label = self._read_pdbqt(path)
            models = parse_pdbqt(text)
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), interactive)
            return False
        self._pending["poses"] = self._remember_path(path)
        self._note_recent(path)
        self.poses_text = text
        first_poses = not self.pose_models
        self.pose_models = models
        self._populate_table()
        self._apply_pose(0)
        if first_poses:
            # Frame the pocket once, when the poses arrive. A pose *change* must
            # not move the camera, or two poses cannot be compared.
            self.viewport.frame_binding_site_when_ready()
        self._log(tr("log.poses", n=len(models), name=label))
        self._rebuild_tree()
        return True

    def _rdkit_mol(self, text: str):
        """Best-effort RDKit molecule for the chemistry menu.

        ``odock.read_structure`` takes a *path*, so PDBQT text is converted to a
        PDB block first — which is exactly what :func:`odock.pdbqt_to_pdb_block`
        provides — and handed to RDKit directly. Sanitisation is off because a
        PDBQT carries neither bond orders nor hydrogens on carbon.
        """
        try:
            from rdkit import Chem

            import odock

            stripped = text.lstrip()
            is_text = (
                stripped[:6].upper() in {"ATOM  ", "HETATM", "REMARK", "ROOT  ", "MODEL "}
                or "ATOM" in stripped[:2000]
                or "HETATM" in stripped[:2000]
            )
            if is_text:
                block = odock.pdbqt_to_pdb_block(text)
                mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
                if mol is None:
                    self._log(tr("log.rdkit_failed"))
                return mol
            return odock.read_structure(text, sanitize=False)
        except Exception as exc:
            self._log(tr("log.rdkit_unavailable", error=exc))
            return None

    def _classify(self, text: str):
        chem = _try_import("chem.receptor")
        if chem is None:
            return None
        try:
            mol = self._rdkit_mol(text)
            if mol is None:
                return None
            return chem.classify_hetero(mol)
        except Exception as exc:
            self._log(tr("log.hetero_unavailable", error=exc))
            return None

    # -- tree ---------------------------------------------------------------

    def _rebuild_tree(self) -> None:
        self.tree.clear()
        receptor = QtWidgets.QTreeWidgetItem(
            self.tree, [tr("tree.receptor"), str(len(self.scene.receptor))]
        )
        receptor.setExpanded(True)
        chains: Dict[str, int] = {}
        for atom in self.scene.receptor:
            key = atom.chain or "?"
            chains[key] = chains.get(key, 0) + 1
        for chain, count in sorted(chains.items()):
            QtWidgets.QTreeWidgetItem(
                receptor, [tr("tree.chain", name=chain), str(count)]
            )
        if self.inventory is not None:
            for kind in ("cofactor", "ion", "ligand", "other"):
                residues = self.inventory.of_kind(kind)
                if not residues:
                    continue
                node = QtWidgets.QTreeWidgetItem(
                    receptor, [tr(f"kind.{kind}"), str(len(residues))]
                )
                for residue in residues[:40]:
                    QtWidgets.QTreeWidgetItem(
                        node, [residue.label, str(residue.heavy_atoms)]
                    )
        if self.scene.box is not None:
            _, size = self.scene.box
            QtWidgets.QTreeWidgetItem(
                self.tree,
                [
                    tr("tree.search_box", x=size[0], y=size[1], z=size[2]),
                    "",
                ],
            )
        ligand = QtWidgets.QTreeWidgetItem(
            self.tree, [tr("tree.ligand"), str(len(self.scene.ligand))]
        )
        if self.scene.ligand:
            QtWidgets.QTreeWidgetItem(
                ligand, [tr("tree.bonds", n=len(self.scene.ligand_bonds)), ""]
            )
        if self.pockets:
            node = QtWidgets.QTreeWidgetItem(
                self.tree, [tr("tree.pockets"), str(len(self.pockets))]
            )
            for pocket in self.pockets[:12]:
                QtWidgets.QTreeWidgetItem(
                    node,
                    [
                        tr(
                            "tree.pocket",
                            index=pocket.index + 1,
                            volume=pocket.volume,
                        ),
                        f"{pocket.score:.2f}",
                    ],
                )
        if self.pose_models:
            node = QtWidgets.QTreeWidgetItem(
                self.tree, [tr("tree.results"), str(len(self.pose_models))]
            )
            node.setExpanded(True)
            for index, model in enumerate(self.pose_models[:50]):
                affinity = f"{model.affinity:.3f}" if model.affinity is not None else "-"
                entry = QtWidgets.QTreeWidgetItem(
                    node, [tr("tree.pose", index=index + 1), affinity]
                )
                # The pose index travels in the item data, not in its label, so
                # that clicking a pose keeps working in every language.
                entry.setData(0, QtCore.Qt.ItemDataRole.UserRole, index)
        if self.interactions:
            node = QtWidgets.QTreeWidgetItem(
                self.tree, [tr("tree.interactions"), str(len(self.interactions))]
            )
            # Clicking it turns the drawing on (it is the entry point the user
            # reached for), so the node carries its own marker.
            node.setData(
                0, QtCore.Qt.ItemDataRole.UserRole + 1, "interactions"
            )
            if not self._interactions_shown:
                node.setToolTip(0, tr("interactions.empty"))
        if self._measurements:
            QtWidgets.QTreeWidgetItem(
                self.tree, [tr("tree.measurements"), str(len(self._measurements))]
            )

    def _on_tree_clicked(self, item, column) -> None:
        """Pose rows switch the pose; the interactions row is a shortcut.

        The 相互作用 node in the left sidebar is the entry point the user
        actually clicked, so it does the same thing as Analysis ▸ Show
        interactions: it computes the contacts of the displayed pose *and* draws
        them. Nothing is drawn before one of those two is used.
        """
        index = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        if isinstance(index, int):
            self.set_pose(index)
            return
        if item is not None and item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) == "interactions":
            self._annotate_interactions()
            self.dashboard_tabs.setCurrentIndex(
                getattr(self, "_interaction_tab_index", 0)
            )

    def show_interactions_action(self) -> None:
        """The same entry point the menu item and the tree node both call."""
        self._annotate_interactions()

    # -- poses --------------------------------------------------------------

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.pose_models))
        current = self.pose_index()
        for row, model in enumerate(self.pose_models):
            affinity = model.affinity
            residues = self._interaction_residues_text() if row == current else ""
            values = [
                str(row + 1),
                f"{affinity:.3f}" if affinity is not None else "-",
                f"{model.rmsd_lower:.3f}" if model.rmsd_lower is not None else "-",
                f"{model.rmsd_upper:.3f}" if model.rmsd_upper is not None else "-",
                residues,
            ]
            for column, value in enumerate(values):
                entry = QtWidgets.QTableWidgetItem(value)
                entry.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                self.table.setItem(row, column, entry)
        self.table.resizeColumnsToContents()

    def _interaction_residues_text(self) -> str:
        analysis = _try_import("analysis")
        if analysis is None or not self.interactions:
            return ""
        try:
            return analysis.interaction_summary(
                self.interactions, self.scene.receptor, self.scene.ligand
            )
        except Exception:
            return ""

    def _apply_pose(self, index: int) -> None:
        if not self.pose_models:
            return
        index = max(0, min(index, len(self.pose_models) - 1))
        model = self.pose_models[index]
        self.scene.ligand = model.atoms
        self.scene.ligand_bonds = _perceived_bonds(model.atoms, "ligand")
        self.sequence.set_structure(model.atoms, "ligand")
        self.scene.highlight = []
        affinity = model.affinity
        text = tr("label.mode", index=index + 1, total=len(self.pose_models))
        if affinity is not None:
            text += tr("label.affinity", value=affinity)
            self.lbl_energy.setText(tr("label.affinity_short", value=affinity))
        if model.rmsd_lower is not None:
            text += tr("label.rmsd_lb", value=model.rmsd_lower)
        self._set_pose_label(text)
        # Selecting a pose is about where it binds: compute this pose's contacts
        # with the current cut-offs and name the residues it touches. The camera
        # is deliberately left alone so two poses can be compared.
        self._pose_index = index
        self._annotate_pose_contacts()
        self._sync_reference_pose()
        # The scene now holds a *different* ligand, so the uploaded geometry has
        # to be rebuilt: without this the viewport keeps drawing the previous
        # pose until the user happens to orbit or zoom (the camera itself is
        # deliberately left alone so two poses can be compared).
        self.viewport.refresh()
        self._rebuild_tree()

    def _annotate_pose_contacts(self) -> None:
        """The contacts of the displayed pose, with ``interaction_thresholds``.

        The profile is always computed — the pose label names the residues it
        touches and the Interactions table lists the pairs, and that is
        *information* — but nothing is **drawn** until the user asks for it:
        no dashes, no interaction focus and no contact marks on the ruler. The
        drawing is what people read as "these two things are interacting", so it
        is opt-in (Analysis ▸ Show interactions, or the 相互作用 tree item), and
        :meth:`_clear_interactions` turns it off again.
        """
        analysis = _try_import("analysis")
        # The contact marks are computed here, from the coordinates alone, so
        # a missing analysis extra never breaks a pose load.
        if self._interactions_shown:
            self._mark_pose_contacts()
        else:
            ruler = getattr(self, "sequence", None)
            if ruler is not None:
                ruler.clear_contact_marks()
        if analysis is None or not self.scene.receptor or not self.scene.ligand:
            return
        try:
            found = list(
                analysis.profile_interactions(
                    self.scene.receptor,
                    self.scene.ligand,
                    **self.interaction_thresholds,
                )
            )
        except Exception as exc:  # pragma: no cover - reported, never fatal
            self._report_error(str(exc), False)
            return
        self.interactions = found
        self._sync_interactions()
        # Pose browsing re-derives the focus from *this* pose's contacts whenever
        # drawing is on and the user asked for the emphasis, so the indices are
        # never stale, and applies none otherwise. The camera and the clip are
        # never touched here, so two poses stay comparable.
        if self._interactions_shown and self._focus_requested:
            self.scene.interaction_focus = list(found)
            self.viewport.refresh(upload_receptor=True)
        else:
            self.scene.interaction_focus = []
        summary = ""
        try:
            summary = analysis.interaction_summary(
                found, self.scene.receptor, self.scene.ligand
            )
        except Exception:  # pragma: no cover - defensive
            summary = ""
        if summary:
            self._set_pose_label(
                self.pose_label_text() + tr("label.pose_binding", residues=summary)
            )
        self._populate_table()

    #: How close a receptor residue has to come to the ligand to count as a
    #: contact. 4.5 Å between heavy atoms is the usual "touches" definition and
    #: is the same cut-off the pose-comparison panel uses, so the ruler, the
    #: comparison and the table all speak about the same set of residues.
    CONTACT_CUTOFF = 4.5

    def _mark_pose_contacts(self) -> None:
        """Underline the residues the displayed pose touches on the ruler.

        The ruler is the sequence view of the binding site, so the contacts
        belong on it: read the letters that are lit up and the answer to "what
        does this pose touch?" is a glance rather than a table.
        """
        ruler = getattr(self, "sequence", None)
        if ruler is None:
            return
        if not self.scene.receptor or not self.scene.ligand:
            ruler.clear_contact_marks()
            return
        try:
            contacts = dashboard.contact_map(
                self.scene.ligand,
                self.scene.receptor,
                cutoff=self.CONTACT_CUTOFF,
            )
        except Exception:  # pragma: no cover - a mark must never break a load
            ruler.clear_contact_marks()
            return
        ruler.set_contact_marks({"contact": list(contacts)})

    def _reference_atoms(self) -> list:
        """The ligand the displayed pose is compared against, if any.

        The native (co-crystallised) ligand when one was extracted, otherwise
        the first pose — which is the reference a docking run is judged against.
        """
        native = self._reference_pose
        if native is not None:
            atoms = self._mol_atoms(native)
            if atoms:
                return atoms
        if len(self.pose_models) > 1:
            return list(self.pose_models[0].atoms)
        return []

    @staticmethod
    def _mol_atoms(mol) -> list:
        """Lightweight :class:`~odock.gui.structure.Atom` list from an RDKit Mol."""
        from .structure import Atom

        try:
            conformer = mol.GetConformer()
        except Exception:
            return []
        try:
            label = (mol.GetProp("_Name") or "LIG").strip()[:3].upper() or "LIG"
        except Exception:
            label = "LIG"
        atoms = []
        for index, atom in enumerate(mol.GetAtoms()):
            try:
                position = conformer.GetAtomPosition(index)
            except Exception:
                return []
            symbol = atom.GetSymbol()
            atoms.append(
                Atom(
                    name=f"{symbol}{index + 1}",
                    element=symbol,
                    res_name=label,
                    res_id=1,
                    chain="R",
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                )
            )
        return atoms

    def _sync_reference_pose(self) -> None:
        """Show the reference ligand as a translucent overlay, or clear it."""
        if not self._ghost_visible or self._pose_index == 0:
            self.scene.ghost_ligand = []
            return
        atoms = self._reference_atoms()
        self.scene.ghost_ligand = list(atoms) if atoms else []

    def _toggle_ghost(self, checked: bool) -> None:
        """View ▸ Show reference pose: the first pose / native ligand, ghosted."""
        self._ghost_visible = bool(checked)
        self._sync_reference_pose()
        self.viewport.refresh()
        count = len(self.scene.ghost_ligand or [])
        self._log(
            tr("log.reference_on", n=count)
            if count
            else tr("log.reference_off")
        )

    def _choose_interaction_distances(self) -> None:
        """View ▸ Interaction distances…: edit the cut-offs of the search."""
        dialog = dialogs.InteractionThresholdsDialog(self.interaction_thresholds, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.interaction_thresholds = dialog.values()
        self._log(
            tr(
                "log.interaction_distances",
                summary=", ".join(
                    f"{key}={value:g}"
                    for key, value in sorted(self.interaction_thresholds.items())
                ),
            )
        )
        # Re-annotate with the new numbers so the view matches the dialog — but
        # never switch the drawing back on behind the user's back: with poses the
        # recompute respects the current state, and without poses there is nothing
        # to redraw unless it was already on.
        if self.pose_models:
            self._annotate_pose_contacts()
        elif self._interactions_shown:
            self._annotate_interactions(quiet=True)

    def pose_count(self) -> int:
        """How many poses are loaded."""
        return len(self.pose_models)

    def pose_index(self) -> int:
        """The pose on screen (0 when there is none)."""
        return int(self._pose_index) if self.pose_models else 0

    def set_pose(self, index: int, *, record: bool = True) -> None:
        """Show pose ``index`` — the single funnel the slider used to be.

        The pose *table* is now the only widget that changes this, so one method
        serves the table selection, the arrow keys, the playback timer, the
        console and an undo step. ``record`` keeps the coalescing key, which is
        what makes a fast sequence of changes one Ctrl+Z rather than two hundred.
        """
        if not self.pose_models:
            return
        index = max(0, min(int(index), len(self.pose_models) - 1))
        previous = self.pose_index()
        self._apply_pose(index)
        if record and previous != index:
            self._record(
                tr("label.pose"),
                {"pose": previous},
                {"pose": index},
                merge_key="pose",
            )
        if self.table.currentRow() != index:
            self.table.blockSignals(True)
            self.table.selectRow(index)
            self.table.blockSignals(False)

    def _on_pose_changed(self, value: int) -> None:
        """The old slider signal, kept for callers that still emit it."""
        self.set_pose(value)

    def _on_table_selection(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            return
        # Two ctrl-clicked rows *are* the pose-comparison request: that is the
        # gesture a chemist already has in their fingers, and the panel answers
        # immediately rather than behind a menu.
        if len(rows) >= 2:
            if self._update_comparison(rows[0], rows[1]) is not None:
                self._log(tr("log.compare", a=rows[0] + 1, b=rows[1] + 1))
                if self.comparison_dock is not None:
                    self.comparison_dock.raise_()
            return
        self.set_pose(min(rows))

    def _toggle_playback(self, checked: bool = True) -> None:
        if not self.pose_models:
            return
        if not checked:
            self.viewport.stop_animation()
            return
        analysis = _try_import("analysis")
        frames = [
            [[a.x, a.y, a.z] for a in model.atoms] for model in self.pose_models[:12]
        ]
        if analysis is not None and len(frames) > 1:
            smooth = []
            for first, second in zip(frames, frames[1:]):
                for step in range(6):
                    smooth.append(
                        np_array(
                            analysis.interpolate_coords(
                                np_array(first), np_array(second), step / 6.0
                            )
                        ).tolist()
                    )
            frames = smooth or frames
        self.viewport.play_animation(frames)
        self._log(tr("log.playing", n=len(self.pose_models)))

    # -- boxes --------------------------------------------------------------

    def _box_is_default(self) -> bool:
        return self.scene.box is None

    def _set_box_source(self, source: str) -> None:
        """Record how the box was derived, so a protocol can say so."""
        self._box_source = str(source)

    def _protocol_tab_widget(self):
        """The protocol tab's page, if it was built (it always is)."""
        return getattr(self, "_protocol_page", None)

    def _current_box(self):
        import odock

        if self.scene.box is None:
            return None
        center, size = self.scene.box
        return odock.BoxSpec(
            center=tuple(float(v) for v in center),
            size=tuple(float(v) for v in size),
            spacing=float(self.spacing.value()),
        )

    def _set_box(self, center, size, spacing: float = 0.375) -> None:
        previous = self.scene.box
        previous_spacing = float(self.spacing.value())
        self.scene.box = (
            tuple(float(v) for v in center),
            tuple(float(v) for v in size),
        )
        if self.spacing.value() != float(spacing):
            self.spacing.setValue(float(spacing))
        keys = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
        for key, value in zip(keys, list(center) + list(size)):
            spin = self.spins[key]
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        # A real box means "show it": the checkbox and the menu action follow,
        # with their signals blocked so this cannot push the box back out of the
        # placeholder spin values.
        self._box_visible = True
        for widget in (
            getattr(self, "chk_box", None),
            getattr(self, "box_action", None),
        ):
            if widget is not None and not widget.isChecked():
                widget.blockSignals(True)
                widget.setChecked(True)
                widget.blockSignals(False)
        self._refresh_box_label()
        self.viewport.refresh()
        self._rebuild_tree()
        # A box is a scene edit like any other, so it is undoable — including
        # going back to "there was no box".
        if not self._suppress_history and previous != self.scene.box:
            self._record(
                tr("menu.grid"),
                {
                    "box": None
                    if previous is None
                    else {
                        "center": list(previous[0]),
                        "size": list(previous[1]),
                        "spacing": previous_spacing,
                    }
                },
                {
                    "box": {
                        "center": list(self.scene.box[0]),
                        "size": list(self.scene.box[1]),
                        "spacing": float(self.spacing.value()),
                    }
                },
            )

    def _refresh_box_label(self) -> None:
        if self.scene.box is None:
            self.lbl_box_info.setText(tr("label.no_box"))
            return
        _, size = self.scene.box
        spacing = float(self.spacing.value())
        volume = float(size[0]) * float(size[1]) * float(size[2])
        npts = 1
        for axis in size:
            npts *= max(1, int(float(axis) / max(spacing, 1e-3)) + 1)
        self.lbl_box_info.setText(
            tr(
                "label.box_info",
                volume=volume,
                npts=npts,
                mb=npts * 4 / 1048576,
            )
        )

    def _push_box(self) -> None:
        center = tuple(
            self.spins[k].value() for k in ("center_x", "center_y", "center_z")
        )
        size = tuple(
            max(2.0, self.spins[k].value()) for k in ("size_x", "size_y", "size_z")
        )
        self.scene.box = (center, size)
        self._refresh_box_label()
        if self.chk_box.isChecked():
            self.viewport.refresh()

    def _on_box_spin(self) -> None:
        self._push_box()

    def _toggle_box(self, checked: bool) -> None:
        self._box_visible = bool(checked)
        if not checked:
            self.scene.box = None
        else:
            self._push_box()
        self.viewport.refresh()

    def _choose_box_opacity(self) -> None:
        """Set the translucent fill's opacity (the renderer always draws edges).

        ``Scene.box_alpha`` belongs to the renderer; it is read and written
        through ``getattr``/plain assignment so this stays usable on a build
        where the fill has not landed yet, and reports that instead of raising.
        """
        dialog_class = getattr(dialogs, "BoxOpacityDialog", None)
        if dialog_class is None:  # pragma: no cover - depends on the render side
            self._log(tr("log.no_box_opacity"))
            return
        dialog = dialog_class(float(getattr(self.scene, "box_alpha", 0.35)), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.scene.box_alpha = float(dialog.value())
        self.viewport.refresh()
        self._log(tr("log.box_opacity", value=self.scene.box_alpha))

    def _center_box_on_ligand(self) -> None:
        import odock

        if not self.scene.ligand:
            self._log(tr("log.load_ligand_first"))
            return
        box = odock.box_from_points(
            [[a.x, a.y, a.z] for a in self.scene.ligand], buffer=0.0
        )
        self._set_box_source("ligand")
        self._set_box(box.center, self._current_size(), self.spacing.value())
        self._log(tr("log.box_centred"))

    def _current_size(self):
        if self.scene.box is not None:
            return tuple(max(2.0, v) for v in self.scene.box[1])
        return (20.0, 20.0, 20.0)

    def _fit_box_to_ligand(self) -> None:
        import odock

        if not self.scene.ligand:
            self._log(tr("log.load_ligand_first"))
            return
        box = odock.box_from_points(
            [[a.x, a.y, a.z] for a in self.scene.ligand],
            buffer=8.0,
            spacing=self.spacing.value(),
        )
        self._set_box_source("ligand")
        self._set_box(box.center, box.size, box.spacing)
        self._log(tr("log.box_fitted", x=box.size[0], y=box.size[1], z=box.size[2]))

    def _box_whole_protein(self) -> None:
        if not self.scene.receptor:
            self._log(tr("log.load_receptor_first"))
            return
        atoms = self.scene.receptor
        lo = [min(a.x for a in atoms), min(a.y for a in atoms), min(a.z for a in atoms)]
        hi = [max(a.x for a in atoms), max(a.y for a in atoms), max(a.z for a in atoms)]
        center = [(lo[i] + hi[i]) / 2 for i in range(3)]
        size = [max(4.0, hi[i] - lo[i] + 2.0) for i in range(3)]
        self._set_box_source("receptor")
        self._set_box(center, size, self.spacing.value())
        self._log(tr("log.box_whole"))

    def _box_from_residues(self) -> None:
        label, ok = QtWidgets.QInputDialog.getText(
            self,
            tr("dialog.align_residue"),
            tr("dialog.align_residue.prompt"),
        )
        if not ok or not label.strip():
            return
        wanted = [
            part.strip().upper()
            for part in label.replace(";", ",").split(",")
            if part.strip()
        ]
        atoms = [
            a
            for a in self.scene.receptor
            if f"{a.res_name}{a.res_id}".upper() in wanted
        ]
        if not atoms:
            self._report_error(
                tr("log.no_atoms_for", names=", ".join(wanted)), True
            )
            return
        center = (
            sum(a.x for a in atoms) / len(atoms),
            sum(a.y for a in atoms) / len(atoms),
            sum(a.z for a in atoms) / len(atoms),
        )
        spread = 0.0
        for atom in atoms:
            spread = max(
                spread,
                abs(atom.x - center[0]),
                abs(atom.y - center[1]),
                abs(atom.z - center[2]),
            )
        size = tuple(max(16.0, 2 * spread + 6.0) for _ in range(3))
        self._set_box(center, size, self.spacing.value())
        self._log(tr("log.box_on_atoms", n=len(atoms), names=", ".join(wanted)))

    # -- receptor chemistry -------------------------------------------------

    def _remove_waters(self) -> None:
        chem = _try_import("chem.receptor")
        if chem is None or self.receptor_mol is None:
            self._log(tr("log.no_rec_chem"))
            return
        try:
            policy = chem.CleanPolicy(
                drop_solvent=True,
                drop_ligands=False,
                drop_free_ions=False,
                keep_cofactors=True,
                keep_water_within=None,
            )
            mol, log, _ = chem.clean_receptor(self.receptor_mol, policy)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._adopt_receptor_mol(mol, log or [tr("log.waters_removed")])

    def _strip_ligand(self) -> None:
        chem = _try_import("chem.receptor")
        if chem is None or self.receptor_mol is None or self.inventory is None:
            self._log(tr("log.no_receptor"))
            return
        ligands = self.inventory.of_kind("ligand")
        # A small co-crystal ligand — benzamidine in 3PTB is 112 Da — is below
        # the 150 Da rule and lands in ``other``. Anything in ``other`` that
        # carries carbon is still what the user means by "the ligand", so it is
        # offered as well rather than refusing to act.
        if not ligands:
            ligands = [
                residue
                for residue in self.inventory.of_kind("other")
                if residue.heavy_atoms >= 4
            ]
        if not ligands:
            self._log(tr("log.no_cocrystal"))
            return
        names = sorted({residue.name for residue in ligands})
        labels = tuple(residue.label for residue in ligands)
        try:
            policy = chem.CleanPolicy(
                drop_solvent=False,
                drop_ligands=True,
                drop_free_ions=False,
                keep_cofactors=True,
                drop_residues=labels + tuple(names),
            )
            mol, log, inventory = chem.clean_receptor(self.receptor_mol, policy)
            extracted = chem.extract_ligands(self.receptor_mol, self.inventory)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._adopt_receptor_mol(
            mol, list(log) + [tr("log.stripped", names=", ".join(names))]
        )
        if extracted:
            self._reference_pose = extracted[0][1]
            self._log(tr("log.kept_reference", name=extracted[0][0]))
        elif labels:
            # Below the mass threshold, so it is not returned by
            # `extract_ligands`; take it straight out of the input inventory.
            self._reference_pose = self._extract_residue_mol(ligands[0])
            if self._reference_pose is not None:
                self._log(tr("log.kept_reference", name=labels[0]))

    def _extract_residue_mol(self, residue):
        """Pull one hetero residue out of the receptor as its own molecule."""
        try:
            from rdkit import Chem

            indices = sorted(residue.atom_indices)
            editable = Chem.RWMol(self.receptor_mol)
            for index in sorted(
                (i for i in range(editable.GetNumAtoms()) if i not in set(indices)),
                reverse=True,
            ):
                editable.RemoveAtom(index)
            return editable.GetMol()
        except Exception as exc:  # pragma: no cover - defensive
            self._log(tr("log.extract_failed", label=residue.label, error=exc))
            return None

    def _adopt_receptor_mol(self, mol, log: Sequence[str]) -> None:
        """Replace the loaded receptor with an edited molecule."""
        import odock

        try:
            text = odock.write_receptor_pdbqt(mol)
        except Exception as exc:
            # An internal failure: log it rather than block on a dialog.
            self._log(tr("log.cannot_write_receptor", error=exc))
            return
        self.receptor_mol = mol
        self.load_receptor(text)
        for line in log:
            self._log(f"  {line}")

    def _open_hetero_dialog(self) -> None:
        if self.inventory is None:
            self._report_error(tr("log.load_receptor_first"), True)
            return
        dialog = dialogs.HeteroDialog(self.inventory, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        chem = _try_import("chem.receptor")
        if chem is None:
            return
        kept = dialog.kept_labels()
        try:
            policy = chem.CleanPolicy(
                drop_solvent=True,
                keep_water_within=dialog.keep_water_within(),
                drop_ligands=True,
                drop_free_ions=True,
                keep_cofactors=True,
                keep_residues=tuple(kept),
            )
            mol, log, _ = chem.clean_receptor(self.receptor_mol, policy)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._adopt_receptor_mol(
            mol, list(log) + [tr("log.kept_residues", n=len(kept))]
        )
        self._log(tr("log.hetero_applied", n=len(kept)))

    def _open_protonation_dialog(self) -> None:
        dialog = dialogs.ProtonationDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        chem = _try_import("chem.receptor")
        if chem is None:
            self._log(tr("log.no_rec_chem"))
            return
        if self.receptor_mol is None:
            self._report_error(tr("log.load_receptor_first"), True)
            return
        try:
            mol, log = chem.protonate(self.receptor_mol, ph=float(values["ph"]))
            if values["polar_only"]:
                mol, removed = chem.strip_nonpolar_hydrogens(mol, collapse_charges=True)
                log = list(log) + [tr("log.merged_h", n=removed)]
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._charge_model = values["charge_model"]
        self._adopt_receptor_mol(mol, log)
        self._log(tr("log.protonated", ph=values["ph"]))

    def _assign_charges(self) -> None:
        charges = _try_import("chem.charges")
        if charges is None or self.receptor_mol is None:
            self._log(tr("log.no_charge_module"))
            return
        try:
            values = charges.assign_charges(self.receptor_mol, model=self._charge_model)
            summary = charges.ad4_type_summary(self.receptor_mol)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        total = float(sum(values))
        self._log(
            tr(
                "log.charges_assigned",
                model=self._charge_model,
                total=total,
                types=", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
            )
        )

    def _choose_flexible_residues(self) -> None:
        if not self.scene.receptor:
            self._report_error(tr("log.load_receptor_first"), True)
            return
        labels = sorted({f"{a.res_name}{a.res_id}" for a in self.scene.receptor})
        chosen, ok = QtWidgets.QInputDialog.getItem(
            self,
            tr("dialog.flex_residue"),
            tr("dialog.flex_residue.prompt"),
            labels,
            0,
            False,
        )
        if not ok or not chosen:
            return
        existing = {self.flex_list.item(i).text() for i in range(self.flex_list.count())}
        if chosen not in existing:
            self.flex_list.addItem(chosen)
        self._refresh_flex_label()
        self._log(tr("log.flex_selected", name=chosen))

    def _refresh_flex_label(self) -> None:
        """Show the flexible residues, abbreviated to the first eight."""
        names = [self.flex_list.item(i).text() for i in range(self.flex_list.count())]
        if not names:
            self.lbl_flex.setText(tr("label.no_flex"))
            return
        self.lbl_flex.setText(
            tr("label.flex_n", n=len(names), names=", ".join(names[:8]))
            + ("…" if len(names) > 8 else "")
        )

    def _write_flexible_pdbqt(self) -> None:
        names = [self.flex_list.item(i).text() for i in range(self.flex_list.count())]
        if not names:
            self._report_error(tr("log.select_flex_first"), True)
            return
        flex = _try_import("chem.flex")
        if flex is None or self.receptor_mol is None:
            self._log(tr("log.flex_needs_chem"))
            return
        writer = flex.write_flexible_receptor_pdbqt
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            tr("dialog.write_flex"),
            "flexreceptor.pdbqt",
            tr("filter.pdbqt"),
        )
        if not path:
            return
        try:
            text = writer(self.receptor_mol, names)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        Path(path).write_text(text, encoding="utf-8")
        self._log(tr("log.wrote_flex", name=Path(path).name))

    # -- ligand chemistry ---------------------------------------------------

    def _ligand_from_smiles(self) -> None:
        text, ok = QtWidgets.QInputDialog.getText(
            self, tr("dialog.smiles"), tr("dialog.smiles.prompt")
        )
        if not ok or not text.strip():
            return
        try:
            import odock

            mol, pdbqt, report = odock.prepare_ligand(text.strip(), name="ligand")
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self.ligand_mol = mol
        self.load_ligand(pdbqt)
        self._log(tr("log.built_smiles", summary=report.summary()))

    def _minimise_ligand(self) -> None:
        ligand = _try_import("chem.ligand")
        if ligand is None or self.ligand_mol is None:
            self._log(tr("log.no_ligand_chem"))
            return
        choices = list(getattr(ligand, "FORCE_FIELDS", ("MMFF94", "UFF")))
        field, ok = QtWidgets.QInputDialog.getItem(
            self,
            tr("dialog.minimise"),
            tr("dialog.minimise.field"),
            choices,
            0,
            False,
        )
        if not ok:
            return
        steps, ok = QtWidgets.QInputDialog.getInt(
            self,
            tr("dialog.minimise"),
            tr("dialog.minimise.steps"),
            500,
            200,
            1000,
            50,
        )
        if not ok:
            return
        try:
            import odock

            mol = ligand.minimize(self.ligand_mol, force_field=field, steps=steps)
            text = odock.write_ligand_pdbqt(mol, name="ligand")
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        before = mol.GetProp("odock_energy_before") if mol.HasProp("odock_energy_before") else "?"
        after = mol.GetProp("odock_energy_after") if mol.HasProp("odock_energy_after") else "?"
        self.ligand_mol = mol
        self.load_ligand(text)
        self._log(
            tr(
                "log.minimised",
                field=field,
                steps=steps,
                before=before,
                after=after,
            )
        )

    def _ligand_charges(self) -> None:
        charges = _try_import("chem.charges")
        if charges is None or self.ligand_mol is None:
            self._log(tr("log.no_charge_module"))
            return
        try:
            values = charges.assign_charges(self.ligand_mol, model="gasteiger")
            summary = charges.ad4_type_summary(self.ligand_mol)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._log(
            tr(
                "log.ligand_charges",
                total=sum(values),
                types=", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
            )
        )

    def _detect_bonds(self, quiet: bool = False) -> None:
        ligand = _try_import("chem.ligand")
        bonds = None
        if ligand is not None and self.ligand_mol is not None:
            try:
                bonds = ligand.rotatable_bonds(
                    self.ligand_mol, locked=list(self._locked_bonds)
                )
            except Exception:
                bonds = None
        if bonds is None:
            bonds = _bond_pairs(self.scene.ligand_bonds)
        self.rotatable_bonds = bonds
        if not quiet:
            self._log(tr("log.rotatable_n", n=len(bonds)))
        self.lbl_bonds.setText(
            tr("label.rotatable_n", n=len(bonds), bonds=bonds)
            + (
                tr("label.locked_n", bonds=self._locked_bonds)
                if self._locked_bonds
                else ""
            )
        )

    def _on_bond_picked(self, first: int, second: int) -> None:
        bond = tuple(sorted((int(first), int(second))))
        locked = {tuple(sorted(b)) for b in self._locked_bonds}
        if bond in locked:
            self._locked_bonds = [
                b for b in self._locked_bonds if tuple(sorted(b)) != bond
            ]
            self._log(tr("log.bond_unlocked", bond=bond))
        else:
            self._locked_bonds.append(bond)
            self._log(tr("log.bond_locked", bond=bond))
        self.scene.locked_bonds = list(self._locked_bonds)
        self._detect_bonds()
        self.viewport.refresh()

    def _run_filters(self) -> None:
        filters = _try_import("filters")
        if filters is None or self.ligand_mol is None:
            self._log(tr("log.no_filter_module"))
            return
        try:
            report = filters.drug_like(self.ligand_mol)
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        report.setdefault("name", "ligand")
        dialogs.FilterDialog([report], self).exec()
        self._filter_verdict = (
            bool(report["passed"]),
            tuple(report.get("violations", [])),
        )
        self._refresh_filter_label()
        verdict = tr("verdict.passes") if report["passed"] else tr("verdict.fails")
        self._log(tr("log.filters", verdict=verdict, properties=report["properties"]))

    def _refresh_filter_label(self) -> None:
        """Re-render the drug-likeness line from the stored verdict."""
        if self._filter_verdict is None:
            self.lbl_filters.setText(tr("label.filters_none"))
            return
        passed, violations = self._filter_verdict
        verdict = tr("verdict.passes") if passed else tr("verdict.fails")
        detail = (
            "" if passed else tr("label.filters_detail", detail="; ".join(violations))
        )
        self.lbl_filters.setText(tr("label.filters_pass", verdict=verdict) + detail)

    # -- pockets and maps ---------------------------------------------------

    def _detect_pockets(self) -> None:
        pocket = _try_import("pocket")
        if pocket is None or not self.scene.receptor:
            self._report_error(tr("log.load_receptor_pocket"), True)
            return
        atoms = list(self.scene.receptor)
        self.progress.setRange(0, 0)
        self._run_background(
            lambda: pocket.find_pockets(atoms),
            self._on_pockets_found,
            label=tr("log.detecting_pockets"),
        )

    def _on_pockets_found(self, pockets) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.pockets = list(pockets or [])
        self._rebuild_tree()
        if not self.pockets:
            self._report_error(tr("log.no_cavity"), True)
            return
        best = self.pockets[0]
        self._log(
            tr(
                "log.pockets_found",
                n=len(self.pockets),
                volume=best.volume,
                center=", ".join(f"{v:.1f}" for v in best.center),
                residues=", ".join(best.residue_labels[:5]),
            )
        )
        dialog = dialogs.PocketDialog(self.pockets, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            chosen = dialog.selected()
            if chosen is not None:
                import odock

                box = odock.box_from_points(
                    [list(p) for p in chosen.points], buffer=2.0
                )
                self._set_box(box.center, box.size, self.spacing.value())
                self._log(tr("log.pocket_aimed", index=chosen.index + 1))

    def _compute_maps(self) -> None:
        box = self._current_box()
        if box is None:
            self._report_error(tr("log.no_box"), True)
            return
        npts = 1
        for axis in box.size:
            npts *= max(1, int(axis / max(box.spacing, 1e-3)) + 1)
        mb = npts * 4 / 1048576
        self.lbl_maps.setText(
            tr(
                "label.maps_info",
                npts=npts,
                mb=mb,
                n=len(self.scene.receptor),
            )
        )
        self._log(tr("log.maps", npts=npts, mb=mb))
        self._maps_computed = True
        self.progress.setValue(100)

    # -- analysis -----------------------------------------------------------

    def _annotate_interactions(self, quiet: bool = False) -> None:
        """Analysis ▸ Show interactions: compute *and draw* the contacts.

        This is the explicit switch. Nothing is drawn before it is pressed —
        docking and pose browsing only *report* contacts — and
        :meth:`_clear_interactions` turns the drawing off again.
        """
        self._interactions_shown = True
        self._mark_pose_contacts()
        analysis = _try_import("analysis")
        if analysis is None:
            if not quiet:
                self._log(tr("log.no_analysis"))
            return
        if not self.scene.receptor or not self.scene.ligand:
            if not quiet:
                self._log(tr("log.load_both"))
            return
        try:
            self.interactions = analysis.profile_interactions(
                self.scene.receptor, self.scene.ligand
            )
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self.scene.interactions = self.interactions
        self._sync_interactions()
        # Hand the same contacts to the renderer's focus mode, so the interacting
        # residues and the ligand atoms are emphasised straight after docking. The
        # request is sticky (see :meth:`_annotate_pose_contacts`); Clear
        # annotations is what switches it off again.
        self._focus_requested = True
        self.scene.interaction_focus = list(self.interactions)
        self.viewport.refresh(upload_receptor=True)
        counts: Dict[str, int] = {}
        for item in self.interactions:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        if not quiet:
            self._log(
                tr(
                    "log.interactions",
                    counts=", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                    or tr("label.none"),
                )
            )
        self._populate_table()
        self._rebuild_tree()

    def _clear_interactions(self) -> None:
        """Analysis ▸ Clear annotations: stop drawing contacts, keep the data.

        "Off" is a *drawing* state: the contacts are still computable and still
        listed, but no dashes, no focus and no ruler marks are drawn, so hiding
        the annotations cannot leave lines whose endpoints are invisible.
        """
        self._interactions_shown = False
        self.interactions = []
        self.scene.interactions = []
        # Clearing the annotations clears the emphasis with them, and forgets the
        # request so browsing poses does not bring it back.
        self._focus_requested = False
        self.scene.interaction_focus = []
        if getattr(self, "sequence", None) is not None:
            self.sequence.clear_contact_marks()
        self.viewport.refresh(upload_receptor=True)
        self._sync_interactions()
        self._populate_table()
        self._rebuild_tree()
        self._log(tr("log.annotations_cleared"))

    # -- interaction lines --------------------------------------------------

    def _apply_interaction_filter(self, *_args) -> None:
        """Draw only the kinds the View ▸ Interaction lines menu keeps ticked.

        The detections themselves are never recomputed or thrown away: the
        filter decides what is *drawn*, and the Interactions panel always reports
        how many lines are hidden, so a filtered view cannot read as "no contacts".
        """
        allowed = {
            kind
            for kind, action in getattr(self, "_interaction_actions", {}).items()
            if action.isChecked()
        }
        self._interaction_filter = allowed
        self._sync_interactions()
        if self.interactions:
            self._log(
                tr(
                    "log.interaction_filter",
                    shown=len(self.scene.interactions),
                    total=len(self.interactions),
                )
            )

    def _sync_interactions(self) -> None:
        """Re-draw and re-list whatever the visibility switches allow.

        Three reductions happen here, and all three are stated rather than hidden:

        * **nothing is drawn unless the user asked** (Analysis ▸ Show
          interactions, or the 相互作用 tree item): docking and pose browsing
          compute the contacts for the label and the table but draw no dashes, no
          focus and no ruler marks;
        * the kind filter (View ▸ Interaction lines) removes kinds;
        * hydrophobic contacts are drawn **one line per receptor residue**. The
          detector enumerates every hydrophobic-carbon pair inside 4.0 Å, which is
          the right answer for the table but reads as a lattice on screen; the
          table keeps every pair and the HUD legend shows ``detected → drawn``.

        Hiding the receptor also suppresses the lines: a dash whose far endpoint
        is not rendered reads as an interaction reaching across the protein, which
        is exactly the complaint this contract exists to prevent.
        """
        shown = bool(getattr(self, "_interactions_shown", False))
        receptor_visible = bool(getattr(self.scene, "show_receptor", True))
        ligand_visible = bool(getattr(self.scene, "show_ligand", True))
        draw = shown and receptor_visible and ligand_visible
        allowed = getattr(self, "_interaction_filter", None)
        if allowed is None:
            visible = list(self.interactions)
        else:
            visible = [item for item in self.interactions if str(item.kind) in allowed]
        if not draw:
            visible = []
        drawn, _counts = dashboard.consolidate_by_residue(visible, self.scene.receptor)
        legend = []
        if draw:
            for kind in sorted({str(item.kind) for item in self.interactions}):
                detected = sum(1 for item in self.interactions if str(item.kind) == kind)
                shown_lines = len([item for item in drawn if str(item.kind) == kind])
                legend.append((kind, detected, shown_lines))
        self.scene.interactions = drawn
        if not draw:
            # The focus pass draws atoms independently of the base receptor pass,
            # so leaving it on after "hide receptor" would light up residues that
            # are no longer there.
            self.scene.interaction_focus = []
        if getattr(self, "viewport", None) is not None:
            self.viewport.set_interaction_legend(legend)
        panel = getattr(self, "interaction_table", None)
        if panel is not None:
            note = "; ".join(
                f"{dashboard.interaction_label(kind)}: {detected} pairs, {shown_lines} line(s)"
                for kind, detected, shown_lines in legend
                if detected != shown_lines
            )
            panel.set_interactions(
                self.interactions,
                self.scene.receptor,
                self.scene.ligand,
                # "hidden" means excluded by the *kind filter* only: the
                # one-line-per-residue reduction is not a hidden contact, it is a
                # different way of drawing the same contact, and the heading
                # should not claim otherwise (the note spells it out).
                hidden=len(self.interactions) - len(visible),
                note=note,
                shown=shown,
            )
        self._refresh_interaction_tab()
        if getattr(self, "viewport", None) is not None:
            self.viewport.refresh()

    def _refresh_interaction_tab(self) -> None:
        """The tab carries the *detected* count, and none while drawing is off.

        Detected, not drawn: the number has to agree with the table's rows (the
        table is the record), while the HUD legend is the one that reports what is
        actually on screen. With the drawing off there is no count at all, so the
        tab cannot claim lines that are not there.
        """
        tabs = getattr(self, "dashboard_tabs", None)
        index = getattr(self, "_interaction_tab_index", -1)
        if tabs is None or index < 0:
            return
        count = (
            len(self.interactions) if getattr(self, "_interactions_shown", False) else 0
        )
        tabs.setTabText(
            index,
            tr("tab.interactions") if not count else f"{tr('tab.interactions')} ({count})",
        )

    def _copy_interactions(self) -> None:
        """Put the list of drawn lines on the clipboard as text."""
        panel = getattr(self, "interaction_table", None)
        if panel is None or not panel.rows():
            self._log(tr("interactions.empty"))
            return
        dashboard.copy_text_to_clipboard(panel.as_text())
        self._log(tr("log.interactions_copied", n=len(panel.rows())))

    def _cluster_poses(self) -> None:
        analysis = _try_import("analysis")
        if analysis is None or not self.pose_models:
            self._report_error(tr("log.load_poses_analysis"), True)
            return
        cutoff, ok = QtWidgets.QInputDialog.getDouble(
            self,
            tr("dialog.cluster"),
            tr("dialog.cluster.cutoff"),
            2.0,
            0.1,
            10.0,
            2,
        )
        if not ok:
            return
        coords = [[[a.x, a.y, a.z] for a in model.atoms] for model in self.pose_models]
        elements = [a.element for a in self.pose_models[0].atoms]
        energies = [model.affinity for model in self.pose_models]
        try:
            clusters = analysis.cluster_poses(
                coords, cutoff=float(cutoff), elements=elements, energies=energies
            )
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        for cluster in clusters:
            try:
                model = self.pose_models[cluster.representative]
                found = analysis.profile_interactions(self.scene.receptor, model.atoms)
                cluster.interactions = analysis.interaction_summary(
                    found, self.scene.receptor, model.atoms
                )
            except Exception:
                cluster.interactions = ""
        self.clusters = clusters
        dialogs.ClusterDialog(clusters, float(cutoff), self).exec()
        self._log(tr("log.clusters", n=len(clusters), cutoff=float(cutoff)))

    # -- measurements -------------------------------------------------------

    #: Colour of a measurement overlay, and of the translucent plane fill.
    MEASURE_COLOUR = (1.0, 0.84, 0.28)
    PLANE_COLOUR = (0.72, 0.60, 1.0)
    #: Default colour of a new annotation label.
    ANNOTATION_COLOUR = (1.0, 0.85, 0.35)

    def measurement_points(self, measurement) -> List[Tuple[float, float, float]]:
        """The coordinates a measurement's references point at, in order."""
        pool = {"receptor": self.scene.receptor, "ligand": self.scene.ligand}
        points: List[Tuple[float, float, float]] = []
        for group, index in measurement.refs:
            atoms = pool.get(str(group)) or []
            try:
                atom = atoms[int(index)]
            except (IndexError, TypeError, ValueError):
                continue
            points.append((float(atom.x), float(atom.y), float(atom.z)))
        return points

    def measurement_value(self, measurement):
        """The current value of ``measurement`` (recomputed, never cached)."""
        return measurement.value(self.measurement_points(measurement))

    def measurement_atom_labels(self, measurement) -> List[str]:
        pool = {"receptor": self.scene.receptor, "ligand": self.scene.ligand}
        labels = []
        for group, index in measurement.refs:
            atoms = pool.get(str(group)) or []
            try:
                atom = atoms[int(index)]
            except (IndexError, TypeError, ValueError):
                labels.append("—")
                continue
            labels.append(f"{atom.res_name}{atom.res_id}:{atom.name}")
        return labels

    def measurement_rows(self) -> List[dict]:
        """The table rows: one per measurement, with its value and unit."""
        rows = []
        for measurement in self._measurements:
            value = self.measurement_value(measurement)
            rows.append(
                {
                    "kind": measurement.kind,
                    "atoms": self.measurement_atom_labels(measurement),
                    "value": dashboard.format_measurement(measurement.kind, value),
                    "unit": measurement.unit,
                    "label": measurement.label,
                }
            )
        return rows

    def _measurement_prompt(self) -> str:
        """What the status bar says while a measurement is being picked."""
        needed = dashboard.measurement_expected_atoms(self._measure_kind)
        return tr(
            "status.measure_pick",
            kind=dashboard.measurement_kind_label(self._measure_kind),
            have=len(self._pick_refs),
            need=needed,
        )

    def set_measure_kind(self, kind: str) -> None:
        """Choose what the next picks will measure (View ▸ Measure)."""
        if kind not in dashboard.MEASUREMENT_KINDS:
            return
        self._measure_kind = kind
        self._pick_refs = []
        self._sync_measure_actions()
        if self.viewport.mode != ViewportWidget.MODE_MEASURE:
            self.viewport.set_mode(ViewportWidget.MODE_MEASURE)
        self.lbl_status.setText(self._measurement_prompt())
        self._log(
            tr("log.measure_kind", kind=dashboard.measurement_kind_label(kind))
        )

    def set_measure_kind_checked(self, *_args) -> None:
        """Menu slot for the radio entries (``triggered`` passes a bool)."""
        for kind, action in getattr(self, "_measure_actions", {}).items():
            if action.isChecked():
                self.set_measure_kind(kind)
                return

    def _sync_measure_actions(self) -> None:
        for kind, action in getattr(self, "_measure_actions", {}).items():
            if action.isChecked() != (kind == self._measure_kind):
                action.blockSignals(True)
                action.setChecked(kind == self._measure_kind)
                action.blockSignals(False)

    def add_measurement_pick(self, ref) -> None:
        """One atom picked in the 3-D view (or from the selection table)."""
        ref = (str(ref[0]), int(ref[1]))
        pool = {"receptor": self.scene.receptor, "ligand": self.scene.ligand}
        atoms = pool.get(ref[0]) or []
        if not (0 <= ref[1] < len(atoms)):
            return
        self._pick_refs.append(ref)
        needed = dashboard.measurement_expected_atoms(self._measure_kind)
        if len(self._pick_refs) >= needed:
            self.commit_measurement(self._measure_kind, list(self._pick_refs))
            self._pick_refs = []
            self.lbl_status.setText(self._measurement_prompt())
        else:
            self.lbl_status.setText(self._measurement_prompt())
        self.viewport.refresh()

    def pending_picks(self) -> List[Tuple[str, int]]:
        """The atoms picked so far for a measurement that is not complete yet."""
        return list(self._pick_refs)

    def cancel_picks(self) -> None:
        self._pick_refs = []
        self.viewport.refresh()

    def measure_selection(self) -> bool:
        """Measure the current ruler/atom-table selection with the active kind.

        The other entry path: the same references the 3-D picker produces, so the
        two cannot disagree about which atoms were meant.
        """
        refs = list(self.sequence.atom_refs())
        if not refs:
            self._log(tr("log.no_selection"))
            return False
        needed = dashboard.measurement_expected_atoms(self._measure_kind)
        if len(refs) < needed:
            self._log(
                tr(
                    "log.measure_needs",
                    kind=dashboard.measurement_kind_label(self._measure_kind),
                    need=needed,
                    have=len(refs),
                )
            )
            return False
        self.commit_measurement(self._measure_kind, refs[:needed])
        return True

    def commit_measurement(self, kind: str, refs) -> Optional[dashboard.Measurement]:
        """Add a measurement and record it as one undoable step."""
        measurement = dashboard.Measurement(kind=kind, refs=list(refs))
        if not measurement.is_complete():
            return None
        points = self.measurement_points(measurement)
        value = measurement.value(points)
        if value is None:
            self._report_error(tr("log.measure_degenerate"), True)
            return None
        before = [item.to_dict() for item in self._measurements]
        after = before + [measurement.to_dict()]
        self._record(
            tr("undo.add_measurement", kind=dashboard.measurement_kind_label(kind)),
            {"measurements": before},
            {"measurements": after},
        )
        self._log(
            tr(
                "log.measurement",
                kind=dashboard.measurement_kind_label(kind),
                atoms=" - ".join(self.measurement_atom_labels(measurement)),
                value=dashboard.format_measurement(kind, value),
            )
        )
        return measurement

    def remove_measurement(self, index: int) -> bool:
        """Delete one measurement, undoably."""
        if not (0 <= index < len(self._measurements)):
            return False
        before = [item.to_dict() for item in self._measurements]
        after = [item for position, item in enumerate(before) if position != index]
        self._record(
            tr("undo.remove_measurement"),
            {"measurements": before},
            {"measurements": after},
        )
        return True

    def _clear_measurements(self) -> None:
        if self._measurements:
            before = [item.to_dict() for item in self._measurements]
            self._record(
                tr("undo.clear_measurements"),
                {"measurements": before},
                {"measurements": []},
            )
        else:
            self._sync_measurements()
        self._log(tr("log.measure_cleared"))

    def _on_atoms_picked(self, selection) -> None:
        """A pick in the 3-D view feeds the active measurement."""
        if not selection:
            return
        self.add_measurement_pick(selection[-1])

    def _sync_measurements(self) -> None:
        """Push the measurements into the scene, the panel and the tree."""
        history = getattr(self, "measure_history", None)
        if history is not None:
            history.set_measurements(self.measurement_rows())
        # The renderer's own measurement pass draws the distance lines; the
        # richer overlays (arcs, planes, normals, labels) are painted by
        # `ViewportWidget._paint_measurement_overlay`, which also puts them into
        # a snapshot.
        self.scene.measurements = [
            (
                tuple(points[0]),
                tuple(points[1]),
                tuple(self.MEASURE_COLOUR) + (1.0,),
            )
            for measurement in self._measurements
            if measurement.kind == "distance" and measurement.visible
            for points in [self.measurement_points(measurement)]
            if len(points) >= 2
        ]
        if getattr(self, "viewport", None) is not None:
            self.viewport.set_measurement_overlays(self._measure_overlays())
            self.viewport.refresh()
        self._rebuild_tree()
        self._session_changed()

    def _measure_overlays(self) -> List[dict]:
        """Everything the viewport needs to draw the measurement layer."""
        overlays = []
        for index, measurement in enumerate(self._measurements):
            if not measurement.visible:
                continue
            points = self.measurement_points(measurement)
            if len(points) < dashboard.measurement_expected_atoms(measurement.kind):
                continue
            overlays.append(
                {
                    "index": index,
                    "kind": measurement.kind,
                    "points": points,
                    "split": measurement.split,
                    "text": f"{dashboard.measurement_kind_label(measurement.kind)} "
                    f"{dashboard.format_measurement(measurement.kind, measurement.value(points))}",
                    "colour": tuple(measurement.colour or self.MEASURE_COLOUR),
                }
            )
        return overlays

    def _annotation_anchor_point(self, note) -> Optional[Tuple[float, float, float]]:
        """Where an annotation is pinned, as world coordinates.

        Three spellings of one idea: an atom reference is that atom, a residue is
        the centroid of its atoms, and a measurement is the midpoint of the atoms
        it asked about — the same points the overlays draw, so a label can never
        float away from the thing it names.
        """
        anchor = tuple(note.anchor or ())
        if not anchor:
            return None
        pool = {"receptor": self.scene.receptor, "ligand": self.scene.ligand}
        if anchor[0] in ("receptor", "ligand") and len(anchor) >= 2:
            atoms = pool.get(anchor[0]) or []
            try:
                atom = atoms[int(anchor[1])]
            except (IndexError, TypeError, ValueError):
                return None
            return (float(atom.x), float(atom.y), float(atom.z))
        if anchor[0] == "residue" and len(anchor) >= 4:
            _kind, chain, res_id, res_name = anchor[:4]
            points = [
                (float(atom.x), float(atom.y), float(atom.z))
                for atom in list(self.scene.receptor) + list(self.scene.ligand)
                if str(atom.chain) == str(chain)
                and int(atom.res_id) == int(res_id)
                and str(atom.res_name) == str(res_name)
            ]
            return dashboard.centroid(points)
        if anchor[0] == "measurement" and len(anchor) >= 2:
            try:
                measurement = self._measurements[int(anchor[1])]
            except (IndexError, TypeError, ValueError):
                return None
            points = self.measurement_points(measurement)
            return None if not points else points[len(points) // 2]
        return None

    def _annotation_overlays(self) -> List[dict]:
        """Anchor points and text for the annotation layer."""
        overlays = []
        for index, note in enumerate(self._annotations):
            if not note.visible or not self._annotation_visible:
                continue
            point = self._annotation_anchor_point(note)
            if point is None:
                continue
            overlays.append(
                {
                    "index": index,
                    "point": point,
                    "text": note.text,
                    "colour": tuple(note.colour),
                }
            )
        return overlays
    def _copy_measurements(self) -> None:
        if not self._measurements:
            self._log(tr("measure.empty"))
            return
        dashboard.copy_text_to_clipboard(self.measure_history.as_text())
        self._log(tr("log.measure_copied", n=len(self._measurements)))

    def _copy_measurements_csv(self) -> None:
        if not self._measurements:
            self._log(tr("measure.empty"))
            return
        dashboard.copy_text_to_clipboard(self.measure_history.as_csv())
        self._log(tr("log.measure_csv", n=len(self._measurements)))

    # -- annotations --------------------------------------------------------

    def add_annotation(
        self, text: str, anchor=None, colour=None
    ) -> Optional[dashboard.Annotation]:
        """Pin a text label to an atom, a residue or a measurement."""
        anchor = tuple(anchor) if anchor else self.annotation_anchor()
        if not anchor:
            self._log(tr("log.annotation_no_anchor"))
            return None
        note = dashboard.Annotation(
            text=str(text).strip() or tr("annotation.default"),
            anchor=anchor,
            colour=tuple(colour or self.ANNOTATION_COLOUR),
        )
        before = [item.to_dict() for item in self._annotations]
        self._record(
            tr("undo.add_annotation"),
            {"annotations": before},
            {"annotations": before + [note.to_dict()]},
        )
        self._log(tr("log.annotation_added", text=note.text))
        return note

    def annotation_anchor(self):
        """The anchor a new label gets: the selection, else the displayed pose."""
        refs = list(self.sequence.atom_refs())
        if len(refs) == 1:
            return refs[0]
        if refs:
            keys = self.sequence.selected_keys()
            if keys:
                chain, res_id, res_name = keys[0]
                return ("residue", str(chain), int(res_id), str(res_name))
        if self._measurements:
            return ("measurement", len(self._measurements) - 1)
        atoms = self.scene.ligand or self.scene.receptor
        if atoms:
            group = "ligand" if self.scene.ligand else "receptor"
            return (group, 0)
        return None

    def edit_annotation(self, index: int, *, text=None, colour=None) -> bool:
        """Change a label's text or colour — one undo step."""
        if not (0 <= index < len(self._annotations)):
            return False
        before = [item.to_dict() for item in self._annotations]
        updated = dashboard.Annotation.from_dict(before[index])
        if text is not None:
            updated.text = str(text)
        if colour is not None:
            updated.colour = tuple(colour)
        after = list(before)
        after[index] = updated.to_dict()
        self._record(
            tr("undo.edit_annotation"),
            {"annotations": before},
            {"annotations": after},
        )
        return True

    def remove_annotation(self, index: int) -> bool:
        if not (0 <= index < len(self._annotations)):
            return False
        before = [item.to_dict() for item in self._annotations]
        after = [item for position, item in enumerate(before) if position != index]
        self._record(
            tr("undo.remove_annotation"),
            {"annotations": before},
            {"annotations": after},
        )
        return True

    def toggle_annotations(self, checked: bool) -> None:
        """View ▸ Annotate ▸ Show labels: hide them all without deleting them."""
        self._annotation_visible = bool(checked)
        self._sync_annotations()
        self._log(
            tr("log.annotations_shown")
            if self._annotation_visible
            else tr("log.annotations_hidden")
        )

    def _open_annotation_dialog(self, index: Optional[int] = None) -> None:
        """Add or edit a label through the dialog (text + colour)."""
        existing = None
        if index is not None and 0 <= index < len(self._annotations):
            existing = self._annotations[index]
        anchor_text = ""
        anchor = tuple(existing.anchor) if existing else tuple(self.annotation_anchor() or ())
        if anchor:
            anchor_text = self._annotation_anchor_text(anchor)
        dialog = dialogs.AnnotationDialog(
            existing.text if existing else "",
            tuple(existing.colour) if existing else self.ANNOTATION_COLOUR,
            anchor_text,
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        text, colour = dialog.values()
        if existing is None:
            self.add_annotation(text, colour=colour)
        else:
            self.edit_annotation(index, text=text, colour=colour)

    def _annotation_anchor_text(self, anchor) -> str:
        """A human description of what a label is pinned to."""
        if anchor and anchor[0] in ("receptor", "ligand") and len(anchor) >= 2:
            labels = self.measurement_atom_labels(
                dashboard.Measurement("distance", refs=[(anchor[0], anchor[1])])
            )
            return labels[0] if labels else "—"
        if anchor and anchor[0] == "residue" and len(anchor) >= 4:
            return f"{anchor[3]}{anchor[2]}"
        if anchor and anchor[0] == "measurement" and len(anchor) >= 2:
            return tr("annotation.on_measurement", index=int(anchor[1]) + 1)
        return "—"

    def export_viewer_state(self) -> dict:
        """Annotations and measurements for the interop export.

        Exactly the shape ``interop.SceneState`` takes: plain dicts, colours as
        ``(r, g, b)`` in 0–1, and the value the panel is showing, so the number in
        the exported file is the number on screen.
        """
        annotations = []
        for note in self._annotations:
            anchor = tuple(note.anchor or ())
            if anchor and anchor[0] == "residue" and len(anchor) >= 4:
                anchor_payload = {"residue": (anchor[1], anchor[2], anchor[3])}
            elif anchor and anchor[0] == "measurement" and len(anchor) >= 2:
                anchor_payload = {"measurement": int(anchor[1])}
            elif anchor:
                anchor_payload = (str(anchor[0]), int(anchor[1])) if len(anchor) >= 2 else None
            else:
                anchor_payload = None
            annotations.append(
                {
                    "text": note.text,
                    "anchor": anchor_payload,
                    "color": tuple(float(value) for value in note.colour),
                    "visible": bool(note.visible),
                }
            )
        measurements = []
        for measurement in self._measurements:
            value = self.measurement_value(measurement)
            if value is None:
                continue
            measurements.append(
                {
                    "kind": measurement.kind,
                    "refs": [(str(group), int(index)) for group, index in measurement.refs],
                    "split": measurement.split,
                    "value": value if isinstance(value, (int, float)) else None,
                    "unit": measurement.unit,
                }
            )
        return {"annotations": annotations, "viewer_measurements": measurements}

    def _add_annotation_action(self) -> None:
        self._open_annotation_dialog(None)

    def _edit_annotation_action(self) -> None:
        index = self._annotation_index_from_selection()
        if index is None:
            self._log(tr("log.annotation_none"))
            return
        self._open_annotation_dialog(index)

    def _delete_annotation_action(self) -> None:
        index = self._annotation_index_from_selection()
        if index is None:
            self._log(tr("log.annotation_none"))
            return
        self.remove_annotation(index)

    def _annotation_index_from_selection(self) -> Optional[int]:
        """The label the annotation table has selected, if any."""
        table = getattr(self, "annotation_table", None)
        if table is None:
            return None
        rows = sorted({index.row() for index in table.selectedIndexes()})
        if rows and 0 <= rows[0] < len(self._annotations):
            return rows[0]
        return len(self._annotations) - 1 if self._annotations else None

    def _copy_annotations(self) -> None:
        if not self._annotations:
            self._log(tr("annotation.empty"))
            return
        table = getattr(self, "annotation_table", None)
        text = table.as_text() if table is not None else ""
        dashboard.copy_text_to_clipboard(text)
        self._log(tr("log.annotations_copied", n=len(self._annotations)))

    def _annotation_rows(self) -> List[dict]:
        rows = []
        for note in self._annotations:
            anchor = tuple(note.anchor or ())
            if anchor and anchor[0] in ("receptor", "ligand") and len(anchor) >= 2:
                labels = self.measurement_atom_labels(
                    dashboard.Measurement("distance", refs=[(anchor[0], anchor[1])])
                )
                where = labels[0] if labels else "—"
            elif anchor and anchor[0] == "residue" and len(anchor) >= 4:
                where = f"{anchor[3]}{anchor[2]}"
            elif anchor and anchor[0] == "measurement" and len(anchor) >= 2:
                where = tr("annotation.on_measurement", index=int(anchor[1]) + 1)
            else:
                where = "—"
            rows.append(
                {
                    "text": note.text,
                    "anchor": where,
                    "colour": note.colour,
                    "visible": note.visible,
                }
            )
        return rows

    def _sync_annotations(self) -> None:
        table = getattr(self, "annotation_table", None)
        if table is not None:
            table.set_annotations(self._annotation_rows())
        if getattr(self, "viewport", None) is not None:
            self.viewport.set_annotation_overlays(self._annotation_overlays())
            self.viewport.refresh()
        self._session_changed()

    def _record(
        self,
        name: str,
        before: dict,
        after: dict,
        *,
        merge_key: Optional[str] = None,
    ) -> Optional[dashboard.Command]:
        """Apply ``after`` and remember how to get back to ``before``.

        The unit of undo is a *named field snapshot*: a measurement list, an
        annotation list, a pose index, a box, a style. One restore function
        applies any of them, so every command type goes through the same code
        path — which is what keeps the scene and the panels from drifting apart
        after an undo.
        """
        if self._suppress_history:
            self._apply_scene_fields(after)
            return None
        command = dashboard.Command(
            name=name,
            undo=lambda: self._apply_scene_fields(before),
            redo=lambda: self._apply_scene_fields(after),
            merge_key=merge_key,
        )
        self._history.push(command)
        self._sync_history_actions()
        return command

    def _apply_scene_fields(self, fields: dict) -> None:
        """Put a field snapshot back into the scene and every panel."""
        self._suppress_history = True
        try:
            if "measurements" in fields:
                self._measurements = [
                    dashboard.Measurement.from_dict(item) for item in fields["measurements"]
                ]
                self._sync_measurements()
            if "annotations" in fields:
                self._annotations = [
                    dashboard.Annotation.from_dict(item) for item in fields["annotations"]
                ]
                self._sync_annotations()
            if "pose" in fields and self.pose_models:
                # No recording: this *is* the restore path of an undo step.
                self.set_pose(int(fields["pose"]), record=False)
            if "selection" in fields:
                self.sequence.select_keys([tuple(key) for key in fields["selection"]])
            if "style_receptor" in fields:
                self._set_style("receptor", str(fields["style_receptor"]))
            if "style_ligand" in fields:
                self._set_style("ligand", str(fields["style_ligand"]))
            if "theme" in fields:
                self.set_theme(str(fields["theme"]), record=False)
            if "density" in fields:
                self.set_density(str(fields["density"]), record=False)
            if "box" in fields:
                box = fields["box"]
                if box is None:
                    self.scene.box = None
                    self._refresh_box_label()
                    self.viewport.refresh()
                else:
                    self._set_box(box["center"], box["size"], float(box.get("spacing", 0.375)))
            if "interaction_filter" in fields:
                saved = fields["interaction_filter"]
                self._interaction_filter = (
                    None if saved is None else {str(kind) for kind in saved}
                )
                for kind, action in getattr(self, "_interaction_actions", {}).items():
                    action.blockSignals(True)
                    action.setChecked(
                        self._interaction_filter is None
                        or kind in self._interaction_filter
                    )
                    action.blockSignals(False)
                self._sync_interactions()
        finally:
            self._suppress_history = False
        self._sync_history_actions()

    def undo(self) -> bool:
        """Ctrl+Z: step back one scene edit."""
        command = self._history.undo()
        if command is None:
            self._log(tr("log.nothing_to_undo"))
            return False
        self._log(tr("log.undone", name=command.name))
        self._sync_history_actions()
        return True

    def redo(self) -> bool:
        """Ctrl+Shift+Z: step forward again."""
        command = self._history.redo()
        if command is None:
            self._log(tr("log.nothing_to_redo"))
            return False
        self._log(tr("log.redone", name=command.name))
        self._sync_history_actions()
        return True

    def _sync_history_actions(self) -> None:
        """Name the next undo/redo step, so Ctrl+Z is never a blind step.

        The names are menu labels, and a menu label carries an accelerator ``&``;
        the mnemonic belongs to the menu, not to a sentence, so it is stripped
        here ("Undo Grid", not "Undo &Grid").
        """
        undo_action = getattr(self, "undo_action", None)
        redo_action = getattr(self, "redo_action", None)
        if undo_action is not None:
            undo_action.setEnabled(self._history.can_undo())
            name = self._history.undo_name().replace("&", "")
            undo_action.setText(
                tr("action.undo_name", name=name) if name else tr("action.undo")
            )
        if redo_action is not None:
            redo_action.setEnabled(self._history.can_redo())
            name = self._history.redo_name().replace("&", "")
            redo_action.setText(
                tr("action.redo_name", name=name) if name else tr("action.redo")
            )

    def _scene_state(self) -> dict:
        """Everything the panels show, as one comparable snapshot.

        The undo tests compare this before an action and after undoing it: if any
        row, label, overlay or read-out differs, the undo desynchronised the
        scene from the interface, which is the failure this feature invites.
        """
        return {
            "pose": self.pose_index() if self.pose_models else -1,
            "selection": [list(key) for key in self.sequence.selected_keys()],
            "style_receptor": self.scene.style_protein,
            "style_ligand": self.scene.style_ligand,
            "theme": self.color_theme.name,
            "density": self.density,
            "box": (
                None
                if self.scene.box is None
                else [
                    list(self.scene.box[0]),
                    list(self.scene.box[1]),
                    float(self.spacing.value()),
                ]
            ),
            "measurements": [item.to_dict() for item in self._measurements],
            "annotations": [item.to_dict() for item in self._annotations],
            "measure_table": (
                self.measure_history.as_text() if self.measure_history is not None else ""
            ),
            "tree": [
                (
                    self.tree.topLevelItem(row).text(0),
                    self.tree.topLevelItem(row).text(1),
                )
                for row in range(self.tree.topLevelItemCount())
            ],
            "measure_overlays": [item["text"] for item in self._measure_overlays()],
            "annotation_overlays": [
                item["text"] for item in self._annotation_overlays()
            ],
            "interaction_filter": (
                None
                if self._interaction_filter is None
                else sorted(self._interaction_filter)
            ),
            "pose_label": self.pose_label_text(),
        }

    def _set_tool(self, mode: str) -> None:
        target = (
            ViewportWidget.MODE_ORBIT if self.viewport.mode == mode else mode
        )
        self.viewport.set_mode(target)
        self._log(
            tr("log.tool_measure")
            if target == ViewportWidget.MODE_MEASURE
            else tr("log.tool_bond")
            if target == ViewportWidget.MODE_BOND
            else tr("log.tool_orbit")
        )

    def _start_measure_distance(self, _checked: bool = False) -> None:
        """View ▸ Measure distance: the short way into the measure tool."""
        self.set_measure_kind("distance")

    def _build_style_menu(self, parent, title, styles, which):
        """A radio-style style submenu; returns ``{style: action}``.

        The current style is the one ticked, so the menu states what is on
        screen. The actions are exclusive within their own group only, which is
        why the two submenus do not interfere.
        """
        menu = parent.addMenu(title)
        group = QtGui.QActionGroup(menu)
        group.setExclusive(True)
        current = (
            self.scene.style_protein if which == "receptor" else self.scene.style_ligand
        )
        actions: Dict[str, QtGui.QAction] = {}
        for style in styles:
            action = QtGui.QAction(tr(f"style.{style}"), menu)
            action.setCheckable(True)
            action.setChecked(style == current)
            action.setActionGroup(group)
            action.triggered.connect(
                lambda _=False, value=style: self._set_style(which, value)
            )
            menu.addAction(action)
            actions[style] = action
        return actions

    def _open_bond_check(self) -> None:
        """View ▸ Bond check: show the perceived connectivity, group by group.

        This is the answer to "are the sticks joining the right atoms?": the
        report lists the bond count, the degree histogram, the longest bond with
        the atoms it joins, and anything whose valence looks impossible.
        """
        try:
            from .bonds import bond_report
        except Exception as exc:  # pragma: no cover - depends on the bonds module
            self._report_error(tr("log.no_bonds_module", error=exc), True)
            return
        groups: List[Tuple[str, dict]] = []
        for label, atoms, bonds in (
            (tr("tree.receptor"), self.scene.receptor, self.scene.receptor_bonds),
            (tr("tree.ligand"), self.scene.ligand, self.scene.ligand_bonds),
        ):
            if not atoms:
                continue
            try:
                groups.append((label, bond_report(atoms, bonds)))
            except Exception as exc:  # pragma: no cover - reported to the user
                self._report_error(str(exc), True)
                return
        if not groups:
            self._log(tr("log.bond_check_empty"))
            return
        dialogs.BondCheckDialog(groups, self).exec()
        summary = ", ".join(
            f"{name}: {report.get('n_bonds', 0)}" for name, report in groups
        )
        self._log(tr("log.bond_check", summary=summary))

    def _set_style(self, which: str, value: str) -> None:
        before = (
            self.scene.style_protein if which == "receptor" else self.scene.style_ligand
        )
        if which == "receptor":
            self.scene.style_protein = value
            self.viewport.refresh(upload_receptor=True)
        else:
            self.scene.style_ligand = value
            self.viewport.refresh()
        after = (
            self.scene.style_protein if which == "receptor" else self.scene.style_ligand
        )
        if before != after:
            field = "style_receptor" if which == "receptor" else "style_ligand"
            self._record(
                tr(f"menu.{'protein' if which == 'receptor' else 'ligand'}_style"),
                {field: before},
                {field: after},
            )
        current = (
            self.scene.style_protein if which == "receptor" else self.scene.style_ligand
        )
        action = self._style_actions.get(which, {}).get(current)
        if action is not None and not action.isChecked():
            action.setChecked(True)
        self._log(
            tr("log.style", role=tr(f"style.role.{which}"), value=tr(f"style.{current}"))
        )

    def _toggle_axes(self, checked: bool) -> None:
        self.scene.show_axes = bool(checked)
        self.viewport.refresh()

    def _toggle_ssao(self, checked: bool) -> None:
        self.scene.ssao = bool(checked)
        self.viewport.refresh()

    def _toggle_receptor(self, checked: bool) -> None:
        """Show or hide the receptor — and with it, anything that points at it.

        The visibility contract, stated once and applied here: a drawn line or an
        emphasised atom whose other end is no longer rendered reads as an
        interaction reaching across the protein, so hiding the receptor hides the
        interaction drawing (and the focus pass) with it. Turning the receptor
        back on restores both if Show interactions is still active.
        """
        self.scene.show_receptor = bool(checked)
        self._sync_interactions()
        self.viewport.refresh(upload_receptor=True)

    def _toggle_ligand(self, checked: bool) -> None:
        """The same contract the other way round: no ligand, no lines to it."""
        self.scene.show_ligand = bool(checked)
        self._sync_interactions()
        self.viewport.refresh()

    # -- the molecular surface ----------------------------------------------
    #
    # The workbench used to answer "what shape does this protein present?" with
    # a sparse `dots` sampling, which is a picture of atoms, not a surface. The
    # surface built here is the real thing: a triangulated solvent-accessible
    # (or solvent-excluded) surface with a property on every vertex, painted
    # with a documented scale and drawn with a legend. It is built on a worker
    # thread — a 2 000-atom receptor takes seconds, and the window must not
    # freeze for them — and its cost is reported rather than hidden.

    def _set_action_checked(self, action, value: bool) -> None:
        """Set a checkable action's state without firing its slot.

        The initial state of a menu entry is read from the scene, and the scene
        is not ready to be acted on while the menu is being built (the status
        bar does not exist yet). Blocking the signal is the standard Qt way to
        state a fact rather than to issue a command.
        """
        if action is None:
            return
        previous = action.blockSignals(True)
        try:
            action.setChecked(bool(value))
        finally:
            action.blockSignals(previous)

    def _build_surface_menu(self, view: QtWidgets.QMenu) -> None:
        """View ▸ Surface: everything that decides what the surface looks like."""
        menu = view.addMenu(tr("menu.surface"))
        self.surface_action = self._act(
            menu, tr("action.surface_show"), self._toggle_surface, checkable=True
        )
        self._set_action_checked(
            self.surface_action, bool(self.scene.show_surface and self.scene.surface is not None)
        )
        self.legend_action = self._act(
            menu, tr("action.surface_legend"), self._toggle_legend, checkable=True
        )
        self._set_action_checked(self.legend_action, bool(self.scene.surface_legend))
        menu.addSeparator()
        self._act(menu, tr("action.surface_rebuild"), lambda: self._rebuild_surface())

        modes = menu.addMenu(tr("menu.surface_mode"))
        self._surface_mode_actions = self._radio_group(
            modes,
            (
                ("sas", tr("action.surface_mode_sas")),
                ("ses", tr("action.surface_mode_ses")),
            ),
            self.surface_settings.mode,
            self._set_surface_mode,
        )
        colours = menu.addMenu(tr("menu.surface_colour"))
        self._surface_property_actions = self._radio_group(
            colours,
            (
                ("hydrophobicity", tr("action.surface_by_hydrophobicity")),
                ("electrostatic", tr("action.surface_by_potential")),
                ("element", tr("action.surface_by_element")),
            ),
            self.surface_settings.property,
            self._set_surface_property,
        )
        self._act(menu, tr("action.surface_range"), self._choose_surface_range)
        self._act(menu, tr("action.surface_opacity"), self._choose_surface_opacity)
        dielectric = menu.addMenu(tr("menu.surface_dielectric"))
        self._surface_dielectric_actions = self._radio_group(
            dielectric,
            (
                ("distance", tr("action.surface_dielectric_distance")),
                ("uniform", tr("action.surface_dielectric_uniform")),
            ),
            self.surface_settings.dielectric,
            self._set_surface_dielectric,
        )
        self._act(menu, tr("action.surface_esp_breakdown"), self._potential_breakdown)
        menu.addSeparator()
        self.pocket_action = self._act(
            menu, tr("action.surface_pocket"), self._toggle_pocket_lining, checkable=True
        )
        self._set_action_checked(self.pocket_action, bool(self._pocket_only))
        self.highlight_action = self._act(
            menu, tr("action.surface_highlight"), self._toggle_pocket_highlight, checkable=True
        )
        self._set_action_checked(self.highlight_action, bool(self._highlight_pocket))
        menu.addSeparator()
        self._act(menu, tr("action.surface_cut_front"), lambda: self._cut_surface("front"))
        self._act(menu, tr("action.surface_cut_centre"), lambda: self._cut_surface("centre"))
        self._act(menu, tr("action.surface_cut_clear"), self._clear_surface_cut)
        menu.addSeparator()
        self._act(menu, tr("action.surface_stats"), self._surface_statistics)

    def _radio_group(self, parent, entries, current, slot) -> Dict[str, QtGui.QAction]:
        """An exclusive set of menu entries; returns ``{value: action}``."""
        group = QtGui.QActionGroup(parent)
        group.setExclusive(True)
        actions: Dict[str, QtGui.QAction] = {}
        for value, label in entries:
            action = QtGui.QAction(label, parent)
            action.setCheckable(True)
            action.setChecked(value == current)
            action.setActionGroup(group)
            action.triggered.connect(lambda _=False, v=value: slot(v))
            parent.addAction(action)
            actions[value] = action
        return actions

    def _surface_atoms(self) -> List:
        """The atoms a surface is built from: the visible receptor by default."""
        atoms = (
            list(self.scene.visible_receptor())
            if self.scene.show_receptor
            else list(self.scene.receptor)
        )
        return atoms or list(self.scene.receptor)

    def _surface_centre(self):
        """The point a pocket-limited surface is centred on, or ``None``."""
        ligand = self.scene.ligand
        if ligand:
            return tuple(
                sum(float(getattr(atom, axis)) for atom in ligand) / len(ligand)
                for axis in ("x", "y", "z")
            )
        if self.scene.box is not None:
            return tuple(float(value) for value in self.scene.box[0])
        centre = self.scene.bounds()
        return tuple((centre[0][axis] + centre[1][axis]) / 2.0 for axis in range(3))

    def _pocket_keys(self, radius: float = 5.0) -> set:
        """Residue keys of the receptor atoms lining the site."""
        centre = self._surface_centre()
        keys = set()
        if centre is None:
            return keys
        limit = radius * radius
        for atom in self.scene.receptor:
            dx = float(atom.x) - centre[0]
            dy = float(atom.y) - centre[1]
            dz = float(atom.z) - centre[2]
            if dx * dx + dy * dy + dz * dz <= limit:
                keys.add(
                    (
                        str(getattr(atom, "chain", "") or ""),
                        int(getattr(atom, "res_id", 0) or 0),
                        str(getattr(atom, "res_name", "") or ""),
                    )
                )
        return keys

    def _surface_settings(self):
        """The settings for the next build, from the window's own state."""
        from . import surface as surface_module

        base = self.surface_settings
        centre = radius = None
        if self._pocket_only:
            centre = self._surface_centre()
            # A shell wide enough to hold the lining plus a probe: the surface
            # of a contact is local by construction, so a shell is the answer
            # and not an approximation of it (odock.sasa.interface_area makes
            # the same argument for the same reason).
            radius = float(self._pocket_radius)
        highlighted = self._pocket_keys() if self._highlight_pocket else None
        return surface_module.SurfaceSettings(
            mode=base.mode,
            spacing=base.spacing,
            probe=base.probe,
            property=base.property,
            hydrophobicity_scale=base.hydrophobicity_scale,
            value_range=base.value_range,
            dielectric=base.dielectric,
            epsilon=base.epsilon,
            screening=base.screening,
            centre=centre,
            radius=radius,
            bonds=self.scene.receptor_bonds,
            highlighted_residues=highlighted,
            max_points=base.max_points,
            directions=base.directions,
            charge_caveat=self._surface_charge_caveat(),
        )

    def _surface_charge_caveat(self):
        """What the charge column behind an ESP map cannot say.

        An electrostatic surface is only as good as the charges under it, and
        the two ways that column lies are both detectable: it may not carry a
        group's **formal charge** (a formally cationic amidine whose nitrogens
        come out negative, because Gasteiger spreads the +1 over the ion), or it
        may not **conserve** charge at all. The chemistry detector is
        ``odock.protonation`` — imported, not re-implemented — and the ligand's
        coordinates are handed over in the order the *molecule* expects, which is
        the order the API checks and raises on.
        """
        from . import surface as surface_module

        if self.surface_settings.property != "electrostatic":
            return None
        try:
            ligand_mol = self.ligand_mol
        except Exception:  # pragma: no cover - defensive
            ligand_mol = None
        coordinates = None
        if ligand_mol is not None and getattr(ligand_mol, "GetNumConformers", None):
            try:
                if ligand_mol.GetNumConformers():
                    coordinates = ligand_mol.GetConformer().GetPositions()
            except Exception:  # pragma: no cover - defensive
                coordinates = None
        atoms = list(self.scene.ligand) or list(self.scene.receptor)
        if not atoms:
            return None
        try:
            return surface_module.charge_caveat(
                atoms,
                mol=ligand_mol,
                receptor_atoms=self.scene.receptor,
                ligand_coords=coordinates,
            )
        except Exception as exc:  # pragma: no cover - reported, not raised
            return f"the charge column could not be checked ({exc})"

    def _rebuild_surface(self) -> None:
        """Build the surface off the main thread and put it on screen.

        Everything that decides the mesh is read here, on the main thread, and
        handed to the worker as plain data: the worker then touches neither the
        scene nor any widget, which is what makes the build safe to run while
        the user keeps orbiting.
        """
        from . import surface as surface_module

        atoms = self._surface_atoms()
        if not atoms:
            self._log(tr("log.surface_no_receptor"))
            return
        try:
            settings = self._surface_settings()
        except Exception as exc:  # pragma: no cover - reported to the user
            self._report_error(str(exc), True)
            return
        self._log(
            tr(
                "log.surface_building",
                mode=settings.mode.upper(),
                atoms=len(atoms),
            )
        )

        def work():
            return surface_module.build_surface(atoms, settings)

        def done(result) -> None:
            self._on_surface_built(result)

        self._run_background(work, done, label=tr("action.surface_rebuild"))

    def _on_surface_built(self, result) -> None:
        """Put a finished surface on the scene and report its measured cost."""
        stats = self.viewport.set_surface(result)
        self.scene.show_surface = True
        if getattr(self, "surface_action", None) is not None:
            self.surface_action.setChecked(True)
        self.scene.surface_clip = self._surface_clip
        self.viewport.refresh()
        if not isinstance(stats, dict) or not stats.get("triangles"):
            self._log(tr("log.surface_empty"))
            return
        warning = stats.get("charge_warning")
        if warning:
            # A wrong-signed potential map looks exactly like a right one, so
            # the caveat is said out loud rather than left in the surface.
            self._log(tr("log.surface_charge_warning", detail=warning))
        self._log(
            tr(
                "log.surface_built",
                mode=str(result.mode).upper(),
                triangles=int(stats.get("triangles", 0)),
                area=f"{float(stats.get('area', 0.0)):.0f}",
                spacing=f"{float(stats.get('spacing', 0.0)):.2f}",
                seconds=f"{float(stats.get('seconds', 0.0)):.2f}",
                atoms=int(stats.get("atoms", 0)),
            )
        )
        self._session_changed()

    def _toggle_surface(self, checked: bool) -> None:
        wanted = bool(checked)
        if wanted and self.scene.surface is None and self.scene.receptor:
            # Nothing built yet: "show surface" means "build one".
            self._rebuild_surface()
            return
        self.scene.show_surface = wanted
        self.viewport.refresh()
        self._log(tr("log.surface_shown" if wanted else "log.surface_hidden"))

    def _toggle_legend(self, checked: bool) -> None:
        self.scene.surface_legend = bool(checked)
        self.viewport.update()

    def _set_surface_mode(self, mode: str) -> None:
        self.surface_settings.mode = mode
        self._log(tr("log.surface_mode", mode=str(mode).upper()))
        if self.scene.receptor:
            self._rebuild_surface()

    def _set_surface_property(self, name: str) -> None:
        self.surface_settings.property = name
        self._log(
            tr("log.surface_property", property=tr(f"property.{name}"))
        )
        if self.scene.receptor:
            self._rebuild_surface()

    def _choose_surface_range(self) -> None:
        """Set the colour range of the surface, or return it to automatic."""
        surface = self.scene.surface
        low, high = (
            getattr(surface, "value_range", (0.0, 1.0)) if surface else (0.0, 1.0)
        )
        value, ok = QtWidgets.QInputDialog.getText(
            self,
            tr("dialog.surface_range"),
            tr("label.surface_auto") + "\n" + tr("label.surface_min") + " / " + tr("label.surface_max"),
            text=f"{low:.4g}, {high:.4g}",
        )
        if not ok:
            return
        text = str(value).strip()
        if not text or text in ("0", "0,0", "auto"):
            self.surface_settings.value_range = None
            self._log(tr("log.surface_range_auto"))
        else:
            try:
                parts = [float(piece) for piece in text.replace(";", ",").split(",")]
                if len(parts) != 2 or parts[0] >= parts[1]:
                    raise ValueError
            except ValueError:
                self._report_error(
                    tr("log.surface_range_bad", text=text), True
                )
                return
            self.surface_settings.value_range = (parts[0], parts[1])
            self._log(
                tr(
                    "log.surface_range",
                    low=f"{parts[0]:.3g}",
                    high=f"{parts[1]:.3g}",
                )
            )
        if self.scene.receptor:
            self._rebuild_surface()

    def _choose_surface_opacity(self) -> None:
        value, ok = QtWidgets.QInputDialog.getDouble(
            self,
            tr("dialog.surface_opacity"),
            tr("label.surface_opacity"),
            float(self.scene.surface_alpha),
            0.05,
            1.0,
            2,
        )
        if not ok:
            return
        self._surface_alpha = float(value)
        self.scene.surface_alpha = float(value)
        self.viewport.refresh()
        self._log(tr("log.surface_opacity", value=f"{float(value):.2f}"))

    def _toggle_pocket_lining(self, checked: bool) -> None:
        wanted = bool(checked)
        if wanted and not self.scene.ligand and self.scene.box is None:
            self._report_error(tr("log.no_ligand"), True)
            return
        self._pocket_only = wanted
        if wanted:
            self._log(
                tr(
                    "log.surface_pocket",
                    atoms=len(self._surface_atoms()),
                )
            )
        else:
            self._log(tr("log.surface_whole"))
        if self.scene.receptor:
            self._rebuild_surface()

    def _toggle_pocket_highlight(self, checked: bool) -> None:
        """Mark the lining residues on the surface *and* as ball-and-stick.

        The tint alone is hard to read on a busy surface, so the same residues
        are also emphasised through the interaction-focus machinery the
        workbench already has: they are drawn as ball-and-stick while the rest
        of the structure recedes. That is the difference between a picture that
        says "the pocket is around here" and one that names the residues.
        """
        self._highlight_pocket = bool(checked)
        keys = self._pocket_keys() if self._highlight_pocket else set()
        if self._highlight_pocket:
            self._log(tr("log.surface_highlight", n=len(keys)))
        else:
            self._log(tr("log.surface_highlight_off"))
        renderer = self.viewport.renderer
        if renderer is not None:
            if keys:
                indices = [
                    index
                    for index, atom in enumerate(self.scene.receptor)
                    if (
                        str(getattr(atom, "chain", "") or ""),
                        int(getattr(atom, "res_id", 0) or 0),
                        str(getattr(atom, "res_name", "") or ""),
                    )
                    in keys
                ]
                renderer.set_interaction_focus(indices, [])
            else:
                renderer.clear_interaction_focus()
            self.viewport.dirty_ligand = True
        if self.scene.receptor and self.scene.surface is not None:
            self._rebuild_surface()

    def _cut_surface(self, where: str) -> None:
        """Cut the surface with a world-fixed plane through the binding site.

        The plane's normal is the *current* view direction, frozen at the
        moment the action runs: unlike the camera-space front clip, two poses of
        the same protein are then cut identically, which is what makes the two
        pictures comparable.
        """
        centre = self._surface_centre()
        if centre is None:
            self._report_error(tr("log.surface_no_receptor"), True)
            return
        camera = self.viewport.camera
        eye = camera.eye()
        normal = [centre[axis] - eye[axis] for axis in range(3)]
        length = math.sqrt(sum(component * component for component in normal))
        if length < 1e-9:  # pragma: no cover - degenerate camera
            return
        normal = [component / length for component in normal]
        offset = 0.0
        if where == "front" and self.scene.ligand:
            # Keep the whole ligand: put the plane 0.5 Å in front of its
            # frontmost atom, so what is removed is the protein wall between the
            # camera and the site rather than the site itself.
            projected = [
                sum(normal[axis] * float(getattr(atom, axis)) for axis in range(3))
                for atom in self.scene.ligand
            ]
            middle = sum(normal[axis] * centre[axis] for axis in range(3))
            offset = max(0.0, middle - min(projected)) + 0.5
        distance = -sum(normal[axis] * centre[axis] for axis in range(3)) + offset
        self._surface_clip = (tuple(normal), float(distance))
        self.scene.surface_clip = self._surface_clip
        self.viewport.refresh()
        self._log(
            tr(
                "log.surface_cut",
                x=f"{centre[0]:.1f}",
                y=f"{centre[1]:.1f}",
                z=f"{centre[2]:.1f}",
            )
        )

    def _clear_surface_cut(self) -> None:
        self._surface_clip = None
        self.scene.surface_clip = None
        self.viewport.refresh()
        self._log(tr("log.surface_cut_cleared"))

    def _sync_surface_actions(self) -> None:
        """Tick the surface radio entries that match the live settings."""
        for value, action in getattr(self, "_surface_mode_actions", {}).items():
            action.setChecked(value == self.surface_settings.mode)
        for value, action in getattr(self, "_surface_property_actions", {}).items():
            action.setChecked(value == self.surface_settings.property)
        for value, action in getattr(self, "_surface_dielectric_actions", {}).items():
            action.setChecked(value == self.surface_settings.dielectric)

    def _set_surface_dielectric(self, name: str) -> None:
        """Constant or distance-dependent dielectric for the potential.

        Both are documented models of the *same* Coulomb sum: ``ε = 4r`` is the
        AutoDock 4 convention the scoring kernel uses (a screened, local
        picture), ``ε = 4`` is the unscreened one. Switching rebuilds the
        surface, because the potential is sampled onto it.
        """
        self.surface_settings.dielectric = "distance" if name == "distance" else "uniform"
        self._log(
            tr(
                "log.surface_dielectric",
                model=tr(f"action.surface_dielectric_{self.surface_settings.dielectric}"),
            )
        )
        if self.scene.receptor and self.surface_settings.property == "electrostatic":
            self._rebuild_surface()

    def _potential_breakdown(self) -> None:
        """Which residues make this surface electropositive or negative.

        The colour map says *that* a pocket is negative; this says *why*, by
        summing each residue's own contribution to the Coulomb potential at the
        site. It is the same model the surface is painted with — the dialog
        repeats the charge-column diagnostic so an unconserving charge set
        cannot be mistaken for the real electrostatics.
        """
        from . import surface as surface_module

        surface = self.scene.surface
        receptor = list(self.scene.receptor)
        if surface is None or not receptor:
            self._report_error(tr("log.surface_failed"), True)
            return
        focus = self._surface_centre()
        dielectric = self.surface_settings.dielectric

        def work():
            # The painted property is irrelevant here: the question is what the
            # charges do to this surface, whether or not it is coloured by them.
            return surface_module.electrostatic_decomposition(
                self._sample_surface_points(surface),
                receptor,
                group="residue",
                dielectric=dielectric,
                focus=focus,
                top=20,
            )

        def done(report) -> None:
            self._show_potential_report(report)

        self._run_background(work, done, label=tr("action.surface_esp_breakdown"))

    @staticmethod
    def _sample_surface_points(surface, count: int = 2000):
        """A deterministic subsample of the surface vertices, for a breakdown.

        The decomposed potential does not have to be the *painted* property: a
        user may be looking at a hydrophobicity map and still want to know who
        makes the site negative. Sampling the vertices keeps the two independent.
        """
        vertices = getattr(surface, "vertices", None)
        if vertices is None or len(vertices) == 0:
            return []
        stride = max(1, int(len(vertices) // max(1, count)))
        return [tuple(float(v) for v in row) for row in vertices[::stride]]

    def _show_potential_report(self, report) -> None:
        groups = report.get("groups") or []
        lines = [tr("esp.header"), ""]
        lines.append(
            tr(
                "esp.model",
                dielectric=tr(
                    f"action.surface_dielectric_{report.get('dielectric', 'distance')}"
                ),
                epsilon=f"{float(report.get('epsilon', 4.0)):g}",
            )
        )
        lines.append(
            tr(
                "esp.charges",
                nonzero=int(report.get("charges", 0)),
                atoms=int(report.get("atoms", 0)),
                total=f"{float(report.get('total_charge', 0.0)):+.1f}",
            )
        )
        lines.append(
            tr(
                "esp.points",
                points=int(report.get("points", 0)),
                available=int(report.get("points_available", 0)),
            )
        )
        focus_total = report.get("focus_total")
        if focus_total is not None:
            lines.append(tr("esp.focus_total", value=f"{float(focus_total):+.2f}"))
            lines.append(
                tr(
                    "esp.focus_at",
                    x=f"{report['focus'][0]:.1f}",
                    y=f"{report['focus'][1]:.1f}",
                    z=f"{report['focus'][2]:.1f}",
                )
            )
        lines.append("")
        lines.append(tr("esp.table"))
        for group in groups:
            share = (
                100.0 * group.at_focus / float(focus_total)
                if focus_total not in (None, 0.0) and group.at_focus is not None
                else 0.0
            )
            lines.append(
                f"{group.label:<14} {group.atoms:>3}  {group.charge:>+6.2f}  "
                f"{(group.at_focus or 0.0):>+9.2f}  {share:>+7.1f}%  "
                f"{group.mean:>+8.2f}"
            )
        self._show_report(tr("dialog.esp_breakdown"), "\n".join(lines))
        if groups:
            self._log(
                tr(
                    "log.esp_breakdown",
                    residues=len(groups),
                    top=groups[0].label,
                    value=f"{(groups[0].at_focus or 0.0):+.2f}",
                )
            )

    def _surface_statistics(self) -> None:
        surface = self.scene.surface
        stats = self.scene.surface_stats()
        if surface is None or not stats:
            self._report_error(tr("log.surface_failed"), True)
            return
        grid = stats.get("grid") or (0, 0, 0)
        self._log(
            tr(
                "log.surface_stats",
                mode=str(stats.get("mode", surface.mode)).upper(),
                vertices=int(stats.get("vertices", surface.vertices_count)),
                triangles=int(stats.get("triangles", surface.triangles_count)),
                area=f"{float(stats.get('area', surface.area)):.0f}",
                spacing=f"{float(stats.get('spacing', surface.spacing)):.2f}",
                grid="×".join(str(int(size)) for size in grid),
                probe=f"{float(stats.get('probe', surface.probe)):.1f}",
                seconds=f"{float(stats.get('seconds', 0.0)):.2f}",
            )
        )
        # Closure and volume: a closed mesh has an inside, and "how big is this
        # pocket?" is a question a docking tool should be able to answer.
        volume = stats.get("volume")
        if volume is None:
            self._log(tr("log.surface_open"))
        else:
            self._log(
                tr(
                    "log.surface_volume",
                    volume=f"{float(volume):.0f}",
                    cap=int(stats.get("cap_triangles", 0) or 0),
                    cap_area=f"{float(stats.get('cap_area', 0.0)):.0f}",
                )
            )

    # -- SASA, burial and the interop exports --------------------------------

    def _show_report(self, title: str, text: str, svg: Optional[str] = None) -> None:
        """A read-only monospace report in a dialog, plus the log line.

        ``svg`` adds a *Save figure…* button: a table of numbers is what a
        modeller checks, and a figure is what they paste into a report, and the
        same computation produces both.
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        view = QtWidgets.QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setFont(QtGui.QFont("Consolas", 9))
        layout.addWidget(view)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        if svg:
            save = buttons.addButton(
                tr("btn.save_figure"), QtWidgets.QDialogButtonBox.ButtonRole.ActionRole
            )

            def write_figure() -> None:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(
                    dialog, tr("btn.save_figure"), "burial.svg", tr("filter.svg")
                )
                if not path:
                    return
                Path(path).write_text(svg, encoding="utf-8")
                self._log(tr("log.figure_saved", name=Path(path).name))

            save.clicked.connect(write_figure)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.resize(680, 460)
        dialog.exec()

    def _sasa_report(self) -> None:
        """Per-residue SASA and burial of the receptor, plus the ligand's own."""
        from odock import sasa as sasa_module

        receptor = list(self.scene.receptor)
        if not receptor:
            self._log(tr("log.surface_no_receptor"))
            return
        ligand = list(self.scene.ligand)

        def work():
            report = sasa_module.burial(receptor, ligand, reference="unbound")
            ligand_record = (
                sasa_module.ligand_buried_contact_area(ligand, receptor)
                if ligand
                else None
            )
            contact = sasa_module.interface_area(receptor, ligand) if ligand else None
            return report, ligand_record, contact

        def done(payload) -> None:
            report, ligand_record, contact = payload
            lines = [tr("sasa.header"), ""]
            lines.append(
                tr(
                    "sasa.model",
                    probe=f"{report.probe:.2f}",
                    points=report.points,
                )
            )
            lines.append(
                tr(
                    "sasa.reference.unbound"
                    if report.reference == "unbound"
                    else "sasa.reference.free"
                )
            )
            lines.append("")
            # The two references answer different questions, so the summary line
            # says which one produced it. For "free" there is no single buried
            # total to quote (see BurialReport: the reference is a sum of every
            # residue measured in isolation), so the row reports the per-residue
            # sum under its own name instead of a number that looks like a
            # surface.
            if report.buried_area is None:
                lines.append(
                    tr(
                        "sasa.total.free",
                        total=f"{report.total_area:.1f}",
                        isolated=f"{report.reference_total:.1f}",
                        summed=f"{report.summed_buried_area:.1f}",
                    )
                )
            else:
                lines.append(
                    tr(
                        "sasa.total",
                        total=f"{report.total_area:.1f}",
                        reference=f"{report.reference_total:.1f}",
                        buried=f"{report.buried_area:.1f}",
                        share=f"{100.0 * report.buried_fraction:.1f}%",
                    )
                )
            if ligand_record is not None:
                lines.append(
                    tr(
                        "sasa.ligand",
                        free=f"{ligand_record['free_area']:.1f}",
                        exposed=f"{ligand_record['complex_area']:.1f}",
                        buried=f"{ligand_record['buried_area']:.1f}",
                        share=f"{100.0 * ligand_record['buried_fraction']:.1f}%",
                    )
                )
            if contact is not None:
                lines.append(
                    tr(
                        "sasa.interface",
                        buried=f"{contact['buried_area']:.1f}",
                        atoms=int(contact["receptor_atoms_total"]),
                    )
                )
            lines.append("")
            lines.append(report.table(15))
            text = "\n".join(lines)
            self._show_report(tr("dialog.sasa_report"), text)
            self._log(
                tr(
                    "log.sasa_report",
                    residues=len(report.residues),
                    buried=(
                        f"{report.summed_buried_area:.1f}"
                        if report.buried_area is None
                        else f"{report.buried_area:.1f}"
                    ),
                    share=(
                        f"{100.0 * report.summed_buried_area / report.reference_total:.1f}%"
                        if report.buried_area is None and report.reference_total
                        else (
                            "—"
                            if report.buried_fraction is None
                            else f"{100.0 * report.buried_fraction:.1f}%"
                        )
                    ),
                )
            )

        self._run_background(work, done, label=tr("action.sasa_report"))

    def _ligand_burial(self) -> None:
        """How much ligand surface the pocket hides, for the pose on screen."""
        from odock import sasa as sasa_module

        ligand = list(self.scene.ligand)
        receptor = list(self.scene.receptor)
        if not ligand:
            self._log(tr("log.ligand_burial_none"))
            return
        record = sasa_module.ligand_buried_contact_area(ligand, receptor)
        self._log(
            tr(
                "log.ligand_burial",
                free=f"{record['free_area']:.1f}",
                exposed=f"{record['complex_area']:.1f}",
                buried=f"{record['buried_area']:.1f}",
                share=f"{100.0 * record['buried_fraction']:.1f}%",
            )
        )

    def _burial_per_pose(self) -> None:
        """The buried contact area of every pose: a number per pose, not a feel."""
        from odock import sasa as sasa_module

        receptor = list(self.scene.receptor)
        poses = [
            model.atoms for model in (self.pose_models or []) if getattr(model, "atoms", None)
        ]
        if not receptor or not poses:
            self._log(tr("log.ligand_burial_none"))
            return
        affinity = [
            getattr(model, "affinity", None) for model in (self.pose_models or [])
        ]
        records = sasa_module.buried_contact_per_pose(poses, receptor, affinity=affinity)
        lines = [tr("sasa.poses")]
        for index, record in enumerate(records):
            score = record.get("affinity")
            score_text = "     —" if score is None else f"{float(score):6.2f}"
            lines.append(
                f"{index + 1:>4}  {score_text}  {record['buried_area']:8.1f}  "
                f"{100.0 * record['buried_fraction']:7.1f}"
            )
        self._show_report(
            tr("dialog.burial_per_pose"),
            "\n".join(lines),
            svg=sasa_module.pose_burial_svg(
                records, title=tr("dialog.burial_per_pose")
            ),
        )
        best = max(records, key=lambda item: item["buried_area"])
        self._log(
            tr(
                "log.burial_per_pose",
                n=len(records),
                best=f"pose {int(best['pose']) + 1} {best['buried_area']:.1f} Å²",
            )
        )

    def _interop_state(self, *, property_name: str = "") -> "object":
        """The live scene as an :class:`odock.interop.SceneState`."""
        from odock import interop as interop_module

        surface = self.scene.surface
        values = None
        if surface is not None and getattr(surface, "values", None) is not None:
            # Per *atom* values, not per vertex: the B-factor column of a PDB is
            # per atom, and mapping a vertex property back onto atoms is what
            # makes `spectrum b` in PyMOL reproduce the surface colours.
            values = self._surface_values_per_atom(surface)
        range_ = getattr(surface, "value_range", None)
        return interop_module.SceneState(
            receptor=list(self.scene.receptor),
            ligand=list(self.scene.ligand),
            receptor_bonds=self.scene.receptor_bonds,
            ligand_bonds=self.scene.ligand_bonds,
            interactions=self.scene.interactions,
            measurements=self.scene.measurements,
            box=self.scene.box,
            style_protein=self.scene.style_protein,
            style_ligand=self.scene.style_ligand,
            show_receptor=self.scene.show_receptor,
            show_ligand=self.scene.show_ligand,
            receptor_scale=self.scene.receptor_scale,
            ball_scale=self.scene.ball_scale,
            receptor_values=values,
            property_name=property_name or str(getattr(surface, "property_name", "")),
            property_range=tuple(range_) if range_ else None,
            show_surface=bool(self.scene.show_surface and surface is not None),
            surface_mode=str(getattr(surface, "mode", self.surface_settings.mode)),
            surface_property=str(getattr(surface, "property_name", self.surface_settings.property)),
            surface_alpha=float(self.scene.surface_alpha),
            surface=surface,
            camera={
                "target": tuple(self.viewport.camera.target),
                "distance": float(self.viewport.camera.distance),
                "azimuth": float(self.viewport.camera.azimuth),
                "elevation": float(self.viewport.camera.elevation),
                "fov": float(self.viewport.camera.fov),
            },
            title=tr("app.title"),
        )

    def _surface_values_per_atom(self, surface):
        """A per-receptor-atom property array from a per-vertex surface.

        A vertex takes the value of the atom it belongs to, and an atom takes
        the mean over its own vertices — so the atom-level number the exported
        PDB carries is the same property the surface is painted with, aggregated
        the only way that is defined. Atoms the surface does not cover (a
        pocket-lining build leaves most of the protein out) stay at the neutral
        middle of the range rather than at zero, which would read as an extreme
        value in a diverging palette.
        """
        try:
            import numpy as np

            values = np.asarray(surface.values, dtype=float).reshape(-1)
            index = np.asarray(surface.atom_index, dtype=np.int64).reshape(-1)
            receptor = list(self.scene.receptor)
            if values.size == 0 or index.size != values.size:
                return None
            low, high = getattr(surface, "value_range", (0.0, 1.0))
            neutral = 0.5 * (float(low) + float(high))
            out = np.full(len(receptor), neutral, dtype=float)
            # ``atom_index`` indexes the *selected* atoms, so a pocket-lining
            # build has to be mapped back through ``selected`` before it means
            # anything in the scene. Getting this wrong paints the wrong atoms
            # with the right numbers, which is the one error a picture cannot
            # show you.
            source = index.copy()
            selected = getattr(surface, "selected", None)
            if selected is not None and len(selected):
                table = np.asarray(selected, dtype=np.int64)
                safe = np.clip(source, 0, max(0, table.size - 1))
                source = table[safe]
            valid = (index >= 0) & (source >= 0) & (source < len(receptor))
            if not valid.any():
                return out.tolist()
            sums = np.bincount(
                source[valid], weights=values[valid], minlength=len(receptor)
            )
            counts = np.bincount(source[valid], minlength=len(receptor))
            covered = counts > 0
            out[covered] = sums[covered] / counts[covered]
            return out.tolist()
        except Exception:  # pragma: no cover - a duck-typed surface
            return None

    def _export_scene(self, what: str) -> None:
        """File ▸ Export ▸ PyMOL / ChimeraX / PDB / OBJ: the whole scene at once.

        A `.pml` on its own is useless if the coordinates it names are not
        beside it, so the three interop exports write a *bundle*: the two
        structures, the pose, the script and (when there is one) the surface
        mesh. The file dialog picks a directory because that is what the user
        then hands to a collaborator.
        """
        from odock import interop as interop_module

        if not self.scene.has_content():
            self._report_error(tr("log.no_receptor"), True)
            return
        if what == "surface" and self.scene.surface is None:
            self._report_error(tr("log.surface_failed"), True)
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, tr(f"action.export_{'pose_pdb' if what == 'pose' else what}"), ""
        )
        if not directory:
            return
        try:
            state = self._interop_state()
            if what == "surface":
                path = Path(directory) / "odock_surface.obj"
                path.write_text(
                    interop_module.surface_obj_text(self.scene.surface), encoding="utf-8"
                )
                self._write_or_raise(str(path), "surface_obj")
                return
            manifest = interop_module.export_bundle(directory, state)
            if what == "pymol":
                self._write_or_raise(str(manifest["pymol"]), "pymol")
            elif what == "chimerax":
                self._write_or_raise(str(manifest["chimerax"]), "chimerax")
            else:
                self._write_or_raise(str(manifest["pose"]), "pose_pdb")
            self._log(
                tr(
                    "log.export_bundle",
                    n=len(manifest["files"]),
                    what=tr(f"action.export_{'pose_pdb' if what == 'pose' else what}"),
                    name=Path(directory).name,
                )
            )
        except Exception as exc:  # pragma: no cover - reported to the user
            self._report_error(str(exc), True)

    # -- exports ------------------------------------------------------------

    def _save_screenshot(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("dialog.screenshot"), "odock-view.png", tr("filter.png")
        )
        if not path:
            return
        width = max(1600, self.viewport.width() * 2)
        height = max(1000, self.viewport.height() * 2)
        if self.viewport.snapshot(path, width, height):
            self._log(
                tr("log.screenshot", width=width, height=height, name=Path(path).name)
            )
        else:
            self._report_error(tr("log.screenshot_failed"), True)

    def _write_or_raise(self, path: str, what: str) -> None:
        """Every export lands here so the log line is uniform."""
        self._log(tr("log.exported", what=tr(f"export.{what}"), name=Path(path).name))

    def _export(self, what: str) -> None:
        import odock

        box = self._current_box()
        defaults = {
            "receptor_pdbqt": ("receptor.pdbqt", "filter.pdbqt"),
            "ligand_pdbqt": ("ligand.pdbqt", "filter.pdbqt"),
            "poses_pdbqt": ("poses.pdbqt", "filter.pdbqt"),
            "cleaned_pdb": ("receptor.pdb", "filter.pdb"),
            "native_ligand": ("native.pdbqt", "filter.pdbqt"),
            "gpf": ("grid.gpf", "filter.gpf"),
            "dpf": ("dock.dpf", "filter.dpf"),
            "config": ("vina.conf", "filter.config"),
            "xlsx": ("results.xlsx", "filter.xlsx"),
            "csv": ("results.csv", "filter.csv"),
            "svg": ("interactions.svg", "filter.svg"),
        }
        default, kind = defaults[what]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("dialog.export", what=tr(f"export.{what}")), default, tr(kind)
        )
        if not path:
            return
        try:
            if what == "receptor_pdbqt":
                if not self.receptor_text:
                    raise RuntimeError(tr("log.no_receptor"))
                Path(path).write_text(self.receptor_text, encoding="utf-8")
            elif what == "ligand_pdbqt":
                if not self.ligand_text:
                    raise RuntimeError(tr("log.no_ligand"))
                Path(path).write_text(self.ligand_text, encoding="utf-8")
            elif what == "poses_pdbqt":
                if not self.pose_models:
                    raise RuntimeError(tr("log.no_poses"))
                Path(path).write_text(self._poses_text(), encoding="utf-8")
            elif what == "cleaned_pdb":
                export = _try_import("export")
                if export is None or self.receptor_mol is None:
                    raise RuntimeError(tr("log.no_export_module"))
                export.export_cleaned_pdb(self.receptor_mol, path)
            elif what == "native_ligand":
                if self._reference_pose is None:
                    raise RuntimeError(tr("log.strip_first"))
                Path(path).write_text(
                    odock.write_ligand_pdbqt(self._reference_pose, name="native"),
                    encoding="utf-8",
                )
            elif what == "gpf":
                export = _try_import("export")
                if export is None or box is None:
                    raise RuntimeError(tr("log.no_export_box"))
                export.export_gpf(
                    box, self._pending.get("receptor") or "receptor.pdbqt", path
                )
            elif what == "dpf":
                export = _try_import("export")
                if export is None or box is None:
                    raise RuntimeError(tr("log.no_export_box"))
                export.export_dpf(
                    box, self._pending.get("ligand") or "ligand.pdbqt", path
                )
            elif what == "config":
                export = _try_import("export")
                if export is None or box is None:
                    raise RuntimeError(tr("log.no_export_box"))
                export.export_vina_config(
                    box,
                    self._pending.get("receptor") or "receptor.pdbqt",
                    self._pending.get("ligand") or "ligand.pdbqt",
                    path,
                    exhaustiveness=self.exhaustiveness.value(),
                    num_modes=self.poses.value(),
                    seed=self.seed.value(),
                )
            elif what in ("xlsx", "csv"):
                report = _try_import("report")
                if report is None or self.result is None:
                    raise RuntimeError(tr("log.run_first"))
                if what == "xlsx":
                    report.write_xlsx(path, self.result, receptor=self.scene.receptor)
                else:
                    report.write_csv(path, self.result, receptor=self.scene.receptor)
            elif what == "svg":
                analysis = _try_import("analysis")
                if analysis is None or not self.interactions:
                    raise RuntimeError(tr("log.annotate_first"))
                analysis.interaction_diagram_svg(
                    self.scene.receptor,
                    self.scene.ligand,
                    self.interactions,
                    path=path,
                    title=tr("svg.title"),
                )
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unknown export {what}")
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._write_or_raise(path, what)

    def _poses_text(self) -> str:
        blocks = []
        for index, model in enumerate(self.pose_models):
            lines = [f"MODEL {index + 1}", "REMARK"]
            affinity = model.affinity
            if affinity is not None:
                lines.append(
                    f"REMARK  VINA RESULT:    {affinity:8.3f}  "
                    f"{model.rmsd_lower or 0.0:8.3f}  {model.rmsd_upper or 0.0:8.3f}"
                )
            lines.extend(self._atom_line(atom) for atom in model.atoms)
            lines.append("ENDMDL")
            blocks.append("\n".join(lines))
        return "\n".join(blocks) + "\n"

    @staticmethod
    def _atom_line(atom) -> str:
        element = (atom.element or "C")[:2]
        return (
            f"ATOM  {atom.serial:5d} {atom.name:<4s} {atom.res_name:<3s} "
            f"{atom.chain:1s}{atom.res_id:4d}    "
            f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
            f"  1.00  0.00    {atom.charge:6.3f} {atom.ad_type or element:>2s}"
        )

    def _fetch_pdb(self) -> None:
        dialog = dialogs.FetchDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        if not values["pdb_id"]:
            return
        fetch = _try_import("fetch")
        if fetch is None:
            self._report_error(tr("log.no_fetch_module"), True)
            return
        try:
            path = fetch.fetch_pdb(values["pdb_id"])
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self._log(tr("log.downloaded", pdb_id=values["pdb_id"], path=path))
        if values["as_ligand"]:
            self.load_ligand(path)
        else:
            self.load_receptor(path)

    # -- docking ------------------------------------------------------------

    def _engine_settings(self) -> dict:
        if not self.receptor_text or not self.ligand_text:
            raise RuntimeError(tr("log.load_both"))
        box = self._current_box()
        if box is None:
            self._fit_box_to_ligand()
            box = self._current_box()
        if box is None:
            raise RuntimeError(tr("log.no_box_defined"))
        search = {0: None, 1: "lga", 2: "lga_solis"}[self.search.currentIndex()]
        return {
            "receptor_text": self.receptor_text,
            "ligand_text": self.ligand_text,
            "box": box,
            "scoring": self.engine.currentText(),
            "exhaustiveness": self.exhaustiveness.value(),
            "num_poses": self.poses.value(),
            "seed": self.seed.value(),
            "use_grid": self.use_grid.isChecked(),
            "min_rmsd": 1.0,
            "energy_range": self.energy_range.value(),
            "use_island_ga": self.search.currentIndex() >= 1,
            "islands": self.islands.value(),
            "population": self.population.value(),
            "generations": self.generations.value(),
            "search": search,
        }

    def _run_docking(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._log(tr("log.run_in_progress"))
            return
        try:
            settings = self._engine_settings()
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        box = settings["box"]
        self._log(
            tr(
                "log.docking_start",
                scoring=settings["scoring"],
                exhaustiveness=settings["exhaustiveness"],
                seed=settings["seed"],
                x=box.size[0],
                y=box.size[1],
                z=box.size[2],
            )
        )
        worker = _DockWorker(settings)
        worker.ticked.connect(self._log)
        worker.phase.connect(self._on_dock_phase)
        worker.finished_ok.connect(self._on_docking_done)
        worker.failed.connect(self._on_docking_failed)
        worker.finished.connect(lambda: self._set_running(False))
        self._worker = worker
        self._set_running(True)
        self.progress.setRange(0, 0)
        if self.run_dashboard is not None:
            self.run_dashboard.begin_run(
                scoring=settings["scoring"],
                exhaustiveness=settings["exhaustiveness"],
                seed=settings["seed"],
            )
        worker.start()

    def _on_dock_phase(self, name: str, elapsed: float) -> None:
        if self.run_dashboard is not None:
            self.run_dashboard.set_phase(name, elapsed)

    def _set_running(self, running: bool) -> None:
        self.btn_run.setEnabled(not running)
        self.btn_pause.setEnabled(running)
        self.btn_abort.setEnabled(running)
        self.btn_score.setEnabled(not running)
        if not running:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self._paused = False
            self.btn_pause.setText(tr("btn.pause"))

    def _pause_docking(self) -> None:
        if self._worker is None:
            return
        # The button label is translated, so a flag carries the state.
        if not self._paused:
            self._worker.pause()
            self._paused = True
            self.btn_pause.setText(tr("btn.resume"))
            self.lbl_status.setText(tr("status.paused"))
        else:
            self._worker.resume()
            self._paused = False
            self.btn_pause.setText(tr("btn.pause"))
            self.lbl_status.setText(tr("status.running"))

    def _toggle_pause(self) -> None:
        self._pause_docking()

    def _abort_docking(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        self._log(tr("log.abort_requested"))

    def _on_docking_done(self, result) -> None:
        self.result = result
        self._log(result.table())
        self._log(
            tr(
                "log.docking_done",
                npts=result.grid_points,
                mb=result.grid_mb,
                tors=result.num_tors,
                n=len(result.poses),
            )
        )
        self._record_run(result)
        self.load_poses(result.to_pdbqt())
        self._log(tr("log.contacts_hint"))

    def _record_run(self, result) -> dashboard.RunTrace:
        """Put a finished run on the dashboard, with the real pose energies."""
        worker = self._worker
        trace = getattr(worker, "trace", None)
        if trace is None:  # pragma: no cover - only if the worker died oddly
            trace = dashboard.RunTrace(
                energies=[float(pose.affinity) for pose in result.poses],
                elapsed=float(getattr(result, "elapsed", 0.0) or 0.0),
                grid_points=int(result.grid_points or 0),
                grid_mb=int(result.grid_mb or 0),
                num_tors=float(result.num_tors or 0.0),
            )
        self.run_history.add(trace)
        if self.run_dashboard is not None:
            self.run_dashboard.history = self.run_history
            self.run_dashboard.finish_run(trace)
        return trace

    def _on_docking_failed(self, message: str) -> None:
        self._log(tr("log.docking_failed", message=message.splitlines()[0]))
        if self.run_dashboard is not None:
            self.run_dashboard.fail_run(tr("dashboard.status.failed"))
        self._report_error(message, True)

    def _score_current(self) -> None:
        import odock

        if not self.receptor_text or not self.scene.ligand:
            self._report_error(tr("log.load_both"), True)
            return
        try:
            box = self._current_box() or odock.box_from_points(
                [[a.x, a.y, a.z] for a in self.scene.ligand], buffer=8.0
            )
            value = odock.score(
                self.receptor_text,
                self._single_ligand_text(),
                box,
                scoring=self.engine.currentText(),
            )
        except Exception as exc:
            self._report_error(str(exc), True)
            return
        self.lbl_energy.setText(tr("label.score", value=value["affinity"]))
        self._log(
            tr(
                "log.score",
                affinity=value["affinity"],
                inter=value["inter"],
                intra=value["intra"],
            )
        )

    def _single_ligand_text(self) -> str:
        return "\n".join(
            ["ROOT"]
            + [self._atom_line(atom) for atom in self.scene.ligand]
            + ["ENDROOT", f"TORSDOF {len(self._locked_bonds)}"]
        )

    # -- background helpers -------------------------------------------------

    def _run_background(self, function, on_success, *, label: str = "") -> None:
        worker = _CallWorker(function, label=label)
        worker.ticked.connect(self._log)
        worker.finished_ok.connect(on_success)
        worker.failed.connect(self._on_background_failed)
        worker.finished.connect(lambda: self.progress.setRange(0, 100))
        self._background = worker
        worker.start()

    def _on_background_failed(self, message: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self._report_error(message, True)

    # -- sequence ruler and selection ---------------------------------------

    def _atom_pool(self) -> Dict[str, Sequence]:
        """The two atom lists a selection can point into."""
        return {"receptor": self.scene.receptor, "ligand": self.scene.ligand}

    def _atoms_for(self, refs) -> List:
        """The atoms of ``[("receptor"|"ligand", index), ...]``, skipping gaps."""
        pool = self._atom_pool()
        out = []
        for group, index in refs or ():
            atoms = pool.get(group) or []
            try:
                position = int(index)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            if 0 <= position < len(atoms):
                out.append(atoms[position])
        return out

    def _on_sequence_selection(self, keys) -> None:
        """The ruler selection becomes the 3-D selection and the atom list."""
        keys = list(keys or ())
        before = [list(item) for item in getattr(self, "_selection_keys", [])]
        after = [list(item) for item in keys]
        self._selection_keys = [tuple(item) for item in keys]
        if before != after:
            self._record(
                tr("dock.selection"),
                {"selection": before},
                {"selection": after},
                merge_key="selection",
            )
        refs = self.sequence.atom_refs(keys)
        self.scene.selection = refs
        self._populate_selection_table(refs)
        self.viewport.refresh()
        if refs:
            self._log(tr("log.selection", n=len(refs), residues=len(keys)))

    def _focus_residue(self, refs) -> None:
        """Centre the camera on a double-clicked residue."""
        atoms = self._atoms_for(refs)
        if not atoms:
            return
        lo = (min(a.x for a in atoms), min(a.y for a in atoms), min(a.z for a in atoms))
        hi = (max(a.x for a in atoms), max(a.y for a in atoms), max(a.z for a in atoms))
        # A binding-site clip (``frame_binding_site``) hides every receptor atom
        # farther than its radius from the ligand, which would hide the residue
        # the user just double-clicked — centring on one residue means looking at
        # the whole structure again, so the clip goes. ``frame_all`` is the other
        # thing that clears it.
        if getattr(self.scene, "front_clip", False) or (
            self.scene.receptor_cutoff is not None and self.scene.receptor_radius > 0
        ):
            self.scene.receptor_cutoff = None
            self.scene.receptor_radius = 0.0
            self.scene.front_clip = False
            self.viewport.refresh(upload_receptor=True)
        self.viewport.camera.frame(lo, hi, margin=2.2)
        self.viewport.refresh()

    def _populate_selection_table(self, refs) -> None:
        """Fill the Selection dock: one row per atom, a count per element."""
        table = getattr(self, "selection_table", None)
        if table is None:  # pragma: no cover - the panel is built up front
            return
        refs = list(refs or ())
        table.setRowCount(len(refs))
        counts: Dict[str, int] = {}
        for row, (group, index) in enumerate(refs):
            atoms = self._atom_pool().get(group) or []
            if not (0 <= int(index) < len(atoms)):
                continue
            atom = atoms[int(index)]
            element = atom.element or "?"
            counts[element] = counts.get(element, 0) + 1
            values = (
                atom.name,
                element,
                f"{atom.res_name}{atom.res_id}",
                atom.chain,
                f"{atom.x:.3f}",
                f"{atom.y:.3f}",
                f"{atom.z:.3f}",
                atom.ad_type,
                f"{atom.charge:+.3f}",
            )
            for column, value in enumerate(values):
                entry = QtWidgets.QTableWidgetItem(str(value))
                entry.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                table.setItem(row, column, entry)
        table.resizeColumnsToContents()
        if counts:
            summary = ", ".join(
                f"{element} {count}"
                for element, count in sorted(
                    counts.items(), key=lambda pair: (-pair[1], pair[0])
                )
            )
            self.lbl_selection_summary.setText(
                tr("label.selection_summary", counts=summary, total=len(refs))
            )
        else:
            self.lbl_selection_summary.setText(tr("label.no_selection"))

    def _copy_selection_pdbqt(self) -> None:
        """Write the selected atoms out as a rigid PDBQT block."""
        refs = self.sequence.atom_refs()
        if not refs:
            self._log(tr("log.no_selection"))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, tr("dialog.save_selection"), "selection.pdbqt", tr("filter.pdbqt")
        )
        if not path:
            return
        try:
            Path(path).write_text(self._selection_pdbqt(refs), encoding="utf-8")
        except Exception as exc:  # pragma: no cover - reported to the user
            self._report_error(str(exc), True)
            return
        self._log(tr("log.copied_pdbqt", n=len(refs), name=Path(path).name))

    def _selection_pdbqt(self, refs) -> str:
        """A rigid PDBQT block for ``refs``, with fresh serial numbers."""
        lines = ["ROOT"]
        serial = 0
        for atom in self._atoms_for(refs):
            serial += 1
            lines.append(self._atom_line(replace(atom, serial=serial)))
        lines.append("ENDROOT")
        lines.append("TORSDOF 0")
        return "\n".join(lines) + "\n"

    def _fit_sequence_area(self) -> None:
        """Give the ruler room for the rows it has, up to a sane cap.

        The splitter would otherwise hand it the 46-72 px minimum and hide the
        ligand / ion rows behind a scrollbar nobody knows about.
        """
        needed = self.sequence.track_height() + 18  # + the horizontal scrollbar
        self.sequence_area.setMinimumHeight(max(56, min(needed, 200)))

    def _set_pose_split(self, stacked: bool) -> None:
        """Stack the pose table over the log, or put them side by side."""
        self._pose_stacked = bool(stacked)
        orientation = (
            QtCore.Qt.Orientation.Vertical
            if self._pose_stacked
            else QtCore.Qt.Orientation.Horizontal
        )
        if self.pose_splitter.orientation() != orientation:
            self.pose_splitter.setOrientation(orientation)
        if self.stack_action.isChecked() != self._pose_stacked:
            self.stack_action.blockSignals(True)
            self.stack_action.setChecked(self._pose_stacked)
            self.stack_action.blockSignals(False)

    def _on_atom_clicked(self, hit, additive: bool) -> None:
        """A click in the 3-D view selects the residue of that atom.

        It goes through the ruler on purpose: the same call highlights the track,
        fills the Selection dock, sets ``scene.selection`` and lights the atom up
        in the view, so a click and a ruler click cannot drift apart.
        """
        ruler = self.sequence
        if hit is None:
            if not additive:
                ruler.clear_selection()
            return
        kind, index = hit
        atoms = self.scene.ligand if kind == "ligand" else self.scene.receptor
        try:
            position = int(index)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return
        if not (0 <= position < len(atoms)):
            return
        atom = atoms[position]
        key = (atom.chain, atom.res_id, atom.res_name)
        keys = ruler.selected_keys()
        if additive:
            # Shift/ctrl toggles one residue, exactly like ctrl-click on the ruler.
            remaining = [item for item in keys if item != key]
            keys = remaining if len(remaining) != len(keys) else keys + [key]
        else:
            keys = [key]
        ruler.select_keys(keys)
        block = next((item for item in ruler.blocks() if item.key == key), None)
        if block is not None:
            ruler.ensureVisible(block.index)

    # -- the command palette ------------------------------------------------

    def open_palette(self) -> "dashboard.CommandPalette":
        """Show Ctrl+K. Always rebuilt from the menu bar, so never stale."""
        palette = dashboard.CommandPalette(self, self)
        palette.refresh()
        palette.exec()
        return palette

    #: Kept as a method alias so a menu action (which passes ``checked``) can be
    #: connected to it directly without the argument mattering.
    def _open_palette(self, _checked: bool = False) -> None:
        self.open_palette()

    # -- the 3-D read-out and the clipboard ---------------------------------

    def _toggle_inspect(self, checked: bool) -> None:
        """View ▸ Inspect atom: the cursor's atom card in the 3-D view."""
        self.viewport.set_hover_visible(bool(checked))
        if not checked:
            self.lbl_status.setText(tr("status.ready"))

    def _on_atom_hovered(self, hit) -> None:
        """The status bar follows the cursor's atom."""
        if not hit:
            return
        kind, index = hit
        atoms = self.scene.ligand if kind == "ligand" else self.scene.receptor
        try:
            atom = atoms[int(index)]
        except (IndexError, TypeError):
            return
        self.lbl_status.setText(
            dashboard.atom_readout_text(atom, kind=kind, index=int(index))
        )

    def _copy_view(self) -> None:
        """Copy the 3-D view — including its HUD — to the clipboard."""
        try:
            pixmap = self.viewport.grab()
        except Exception as exc:  # pragma: no cover - reported to the user
            self._report_error(str(exc), True)
            return
        if pixmap.isNull() or not dashboard.copy_pixmap_to_clipboard(pixmap):
            self._report_error(tr("log.copy_view_failed"), True)
            return
        self._log(tr("log.copy_view"))

    # -- pose comparison ----------------------------------------------------

    def compare_selected(self) -> Optional[dashboard.PoseComparison]:
        """Compare the two poses the user ctrl-clicked in the results table."""
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if len(rows) < 2:
            return None
        return self._update_comparison(rows[0], rows[1])

    def _compare_selected(self, _checked: bool = False) -> None:
        comparison = self.compare_selected()
        if comparison is None:
            if len(self.pose_models) < 2:
                self._report_error(tr("log.compare_unavailable"), True)
            else:
                self._log(tr("log.compare_unavailable"))
            return
        if self.comparison_dock is not None:
            self.comparison_dock.raise_()

    def _copy_comparison(self) -> None:
        """Put the two-pose report on the clipboard as plain text."""
        comparison = getattr(self, "_comparison", None)
        if comparison is None:
            self._log(tr("log.compare_unavailable"))
            return
        dashboard.copy_text_to_clipboard(comparison.as_text())
        self._log(tr("log.compare_copied"))

    def _update_comparison(self, row_a: int, row_b: int) -> Optional[dashboard.PoseComparison]:
        """Compute and show the two-pose panel for rows ``row_a``/``row_b``."""
        panel = getattr(self, "comparison", None)
        if (
            panel is None
            or len(self.pose_models) < 2
            or not self.scene.receptor
            or row_a == row_b
        ):
            return None
        first, second = self.pose_models[row_a], self.pose_models[row_b]
        try:
            comparison = dashboard.compare_poses(
                first.atoms,
                second.atoms,
                self.scene.receptor,
                index_a=row_a,
                index_b=row_b,
                affinity_a=first.affinity,
                affinity_b=second.affinity,
            )
        except Exception as exc:
            panel.clear()
            self._log(tr("log.compare_failed", message=str(exc).splitlines()[0]))
            return None
        self._comparison = comparison
        self._comparison_rows = (int(row_a), int(row_b))
        panel.set_comparison(comparison)
        self._mark_comparison_contacts(comparison)
        return comparison

    def _mark_comparison_contacts(self, comparison: dashboard.PoseComparison) -> None:
        """The ruler shows *which* residues the two poses disagree about.

        Green where both poses touch the residue, red where only one of them
        does — the same split the fingerprint table lists, but readable against
        the sequence it belongs to. The diff already holds both contact maps, so
        nothing is recomputed here.
        """
        ruler = getattr(self, "sequence", None)
        if ruler is None:
            return
        ruler.set_contact_marks(
            {
                "shared": [key for key, _a, _b in comparison.diff.shared],
                "unique": [contact.key for contact in comparison.diff.only_a]
                + [contact.key for contact in comparison.diff.only_b],
            }
        )

    # -- recent files -------------------------------------------------------

    def _note_recent(self, path) -> None:
        remembered = self._remember_path(path)
        if remembered:
            self.recent.add(remembered)
        self._rebuild_recent_menu()
        self._session_changed()

    def _rebuild_recent_menu(self) -> None:
        """File ▸ Recent files, rebuilt from the list every time it opens."""
        menu = getattr(self, "recent_menu", None)
        if menu is None:
            return
        menu.clear()
        paths = self.recent.existing()
        if not paths:
            empty = QtGui.QAction(tr("action.recent_empty"), menu)
            empty.setEnabled(False)
            menu.addAction(empty)
            return
        for path in paths:
            action = QtGui.QAction(Path(path).name, menu)
            action.setStatusTip(path)
            action.triggered.connect(lambda _=False, p=path: self.open_recent(p))
            menu.addAction(action)
        menu.addSeparator()
        self._act(menu, tr("action.recent_clear"), self._clear_recent)

    def _clear_recent(self) -> None:
        self.recent.clear()
        self._rebuild_recent_menu()
        self._session_changed()

    def open_recent(self, path) -> bool:
        """Load a recent file, routing it by what the file *is*, not its name."""
        target = Path(path)
        if not target.is_file():
            self.recent.drop(str(target))
            self._rebuild_recent_menu()
            self._log(tr("log.recent_missing", name=target.name))
            return False
        return self.open_structure(target)

    # -- drag and drop ------------------------------------------------------

    def _dropped_paths(self, event) -> List[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        paths = []
        for url in mime.urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        return [path for path in paths if path]

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._dropped_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._dropped_paths(event):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = self._dropped_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        for path in paths:
            self.open_structure(path)

    def open_structure(self, path) -> bool:
        """Load one structure file, choosing the loader from its contents.

        The viewer draws receptors, ligands and poses, and the three loaders are
        different, so a dropped file is routed by what it *says* it is
        (:func:`odock.gui.dashboard.classify_structure`) — a ``MODEL`` record or
        a ``VINA RESULT`` remark means poses, a ``ROOT`` block means a flexible
        ligand, anything else is a receptor. Nothing is guessed from the suffix.
        """
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self._report_error(tr("log.drop_unsupported", name=target.name), True)
            self._log(tr("log.error", message=str(exc)))
            return False
        kind = dashboard.classify_structure(text)
        if kind is None:
            self._report_error(tr("log.drop_unsupported", name=target.name), True)
            return False
        loader = {
            "poses": self.load_poses,
            "ligand": self.load_ligand,
            "receptor": self.load_receptor,
        }[kind]
        ok = bool(loader(target))
        self._log(tr("log.dropped", what=tr(f"drop.{kind}"), name=target.name))
        return ok

    # -- the session --------------------------------------------------------

    def _install_autosave(self) -> None:
        """A debounced writer: any logged action schedules one save."""
        if self.session is None:
            return
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(1500)
        timer.timeout.connect(self.save_session)
        self._autosave_timer = timer

    def _session_changed(self) -> None:
        """Something worth remembering happened; save soon (never immediately).

        The debounce matters: a docking run logs dozens of lines in a burst, and
        rewriting the session file for each of them would be pointless.
        """
        timer = self._autosave_timer
        if timer is None:
            return
        self._autosave_pending = True
        timer.start()

    def _session_payload(self) -> dict:
        """Everything the next launch needs to put this session back.

        A window with nothing loaded keeps the *previous* session's structures:
        restoring is an offer, and a launch that is closed again without
        touching anything must not destroy the very session it offered (while
        the appearance and engine preferences it did change are still saved).
        """
        box = self._current_box()
        docks = {
            name: bool(getattr(self, name) is not None and not getattr(self, name).isHidden())
            for name in (
                "workspace_dock",
                "inspector_dock",
                "bottom_dock",
                "dashboard_dock",
                "comparison_dock",
                "selection_dock",
            )
        }
        payload = {
            "language": i18n.current_language(),
            "theme": self.color_theme.name,
            "density": self.density,
            "layout": self._preset,
            "receptor": self._pending.get("receptor"),
            "ligand": self._pending.get("ligand"),
            "poses": self._pending.get("poses"),
            "box": (
                {
                    "center": list(box.center),
                    "size": list(box.size),
                    "spacing": float(box.spacing),
                }
                if box is not None
                else None
            ),
            "engine": {
                "scoring": self.engine.currentText(),
                "search": self.search.currentIndex(),
                "exhaustiveness": self.exhaustiveness.value(),
                "num_poses": self.poses.value(),
                "seed": self.seed.value(),
                "energy_range": self.energy_range.value(),
                "islands": self.islands.value(),
                "population": self.population.value(),
                "generations": self.generations.value(),
                "use_grid": self.use_grid.isChecked(),
                "threads": self.threads.value(),
            },
            "selection": [list(item) for item in self.sequence.selected_keys()],
            "pose_index": self.pose_index() if self.pose_models else 0,
            "history": self.run_history.to_list(),
            "measurements": [item.to_dict() for item in self._measurements],
            "annotations": [item.to_dict() for item in self._annotations],
            "annotation_visible": bool(self._annotation_visible),
            "measure_kind": str(self._measure_kind),
            "docks": docks,
            "splitters": {
                "central": list(self.central_splitter.sizes()),
                "pose": list(self.pose_splitter.sizes()),
            },
        }
        if self.session is not None and not self.scene.has_content():
            previous = self.session.payload
            if previous is None:
                # A store that was never opened (a window built straight from a
                # path) still has a file worth preserving.
                previous = self.session.load()
            previous = previous or {}
            for key in ("receptor", "ligand", "poses"):
                if not payload.get(key):
                    payload[key] = previous.get(key)
            if payload["box"] is None:
                payload["box"] = previous.get("box")
        return payload

    def save_session(self) -> bool:
        """Write the session file now (and only if persistence is on)."""
        if self.session is None:
            return False
        timer = self._autosave_timer
        if timer is not None:
            timer.stop()
        self._autosave_pending = False
        return bool(self.session.save(self._session_payload()))

    @_history_silent
    def restore_session(self, payload: Optional[dict] = None) -> bool:
        """Put a saved session back, without touching anything it did not save.

        Called by File ▸ Restore session and by the launch offer. Every field is
        optional: a session written by an older build still restores whatever it
        does carry.
        """
        data = payload if payload is not None else (
            self.session.load() if self.session is not None else None
        )
        if not isinstance(data, dict):
            self._log(tr("log.session_none"))
            return False
        language = data.get("language")
        if isinstance(language, str) and language in i18n.LANGUAGES:
            if language != i18n.current_language():
                self.set_language(language)
        self.color_theme = dashboard.theme_named(data.get("theme", self.color_theme.name))
        self.density = str(data.get("density", self.density))
        self._apply_style()
        self._sync_appearance_actions()

        for key, loader in (
            ("receptor", self.load_receptor),
            ("ligand", self.load_ligand),
            ("poses", self.load_poses),
        ):
            path = data.get(key)
            if path and Path(str(path)).is_file():
                loader(path)

        box = data.get("box")
        if isinstance(box, dict) and box.get("center"):
            self._set_box(
                box["center"], box.get("size", (20.0, 20.0, 20.0)), box.get("spacing", 0.375)
            )
        engine = data.get("engine") or {}
        if engine.get("scoring"):
            self.engine.setCurrentText(str(engine["scoring"]))
        for widget, field in (
            (self.exhaustiveness, "exhaustiveness"),
            (self.poses, "num_poses"),
            (self.seed, "seed"),
            (self.energy_range, "energy_range"),
            (self.islands, "islands"),
            (self.population, "population"),
            (self.generations, "generations"),
            (self.threads, "threads"),
        ):
            if engine.get(field) is not None:
                widget.setValue(int(engine[field]))
        if engine.get("search") is not None:
            self.search.setCurrentIndex(int(engine["search"]))
        if engine.get("use_grid") is not None:
            self.use_grid.setChecked(bool(engine["use_grid"]))
        keys = data.get("selection") or []
        if keys:
            self.sequence.select_keys([tuple(item) for item in keys])
        history = data.get("history") or []
        if history:
            self.run_history = dashboard.SessionHistory.from_list(history)
            if self.run_dashboard is not None:
                self.run_dashboard.history = self.run_history
                self.run_dashboard.reset(clear_history=False)
        measurements = data.get("measurements") or []
        if measurements:
            self._measurements = [
                dashboard.Measurement.from_dict(item) for item in measurements
            ]
            self._sync_measurements()
        annotations = data.get("annotations") or []
        if annotations:
            self._annotations = [
                dashboard.Annotation.from_dict(item) for item in annotations
            ]
        if data.get("annotation_visible") is not None:
            self._annotation_visible = bool(data["annotation_visible"])
        if data.get("measure_kind") in dashboard.MEASUREMENT_KINDS:
            self._measure_kind = str(data["measure_kind"])
            self._sync_measure_actions()
        self._sync_annotations()
        docks = data.get("docks") or {}
        for name, visible in docks.items():
            dock = getattr(self, name, None)
            if dock is not None:
                dock.setVisible(bool(visible))
        splitters = data.get("splitters") or {}
        if splitters.get("central"):
            self.central_splitter.setSizes([int(v) for v in splitters["central"]])
        if splitters.get("pose"):
            self.pose_splitter.setSizes([int(v) for v in splitters["pose"]])
        preset = data.get("layout")
        if isinstance(preset, str) and preset in dashboard.LAYOUT_PRESETS:
            self._preset = preset
            self._sync_appearance_actions()
        name = None
        if self.session is not None:
            name = self.session.path.name
        self._log(tr("log.session_restored", name=name or "session.json"))
        return True

    def session_offer(self) -> Optional[str]:
        """A one-line description of the restorable session, or ``None``."""
        if self.session is None:
            return None
        payload = self.session.payload
        if not isinstance(payload, dict):
            return None

        def shown(key: str) -> str:
            value = payload.get(key)
            if not value:
                return tr("dialog.restore_session.none")
            return Path(str(value)).name

        return tr(
            "dialog.restore_session.detail",
            receptor=shown("receptor"),
            ligand=shown("ligand"),
            poses=shown("poses"),
        )

    def clear_session(self) -> None:
        """Forget the saved session (File ▸ Restore session's sibling)."""
        if self.session is not None:
            self.session.clear()
        self._log(tr("log.session_cleared"))

    # -- misc ---------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        strip = getattr(self, "_tool_strip", None)
        if strip is not None:
            strip.adjustSize()
            strip.move(12, max(12, self.viewport.height() - strip.height() - 12))
        # A resize re-states the proportions (they are a fraction of the window),
        # so the central viewport keeps the lion's share at every window size and
        # takes everything the resize adds. A divider the user dragged by hand
        # therefore lasts until the next resize — the proportions are the app's
        # policy, the drag is a temporary look.
        if getattr(self, "workspace_dock", None) is not None:
            self._apply_dock_proportions()

    def closeEvent(self, event) -> None:  # noqa: N802
        # The session is written once, here, so the dock layout and the pose the
        # user was looking at are what the next launch offers.
        timer = self._autosave_timer
        if timer is not None:
            timer.stop()
        if self.session is not None:
            self.save_session()
        for worker in (self._worker, self._background):
            if worker is not None and worker.isRunning():
                cancel = getattr(worker, "cancel", None)
                if cancel is not None:
                    cancel()
                worker.wait(2000)
        viewport = self.viewport
        _release_viewport(viewport)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_BENIGN_QT_MESSAGES = (
    "QFont::setPointSize",
    "QFontDatabase: Cannot find font directory",
    "This plugin does not support propagateSizeHints",
    "Note that Qt no longer ships fonts",
    "QOpenGLWidget is not supported",
)


def _quiet_qt_messages() -> None:
    """Forward Qt's messages to stderr, minus the known-benign ones."""
    if os.environ.get("ODOCK_VERBOSE_QT"):
        return

    def handler(_mode, _context, message: str) -> None:
        if any(noise in message for noise in _BENIGN_QT_MESSAGES):
            return
        sys.stderr.write(f"{message}\n")

    QtCore.qInstallMessageHandler(handler)


def launch_gui(
    receptor: Optional[str] = None,
    ligand: Optional[str] = None,
    poses: Optional[str] = None,
) -> int:
    """Create the Qt application and show the workbench.

    Returns the exit code of the Qt event loop, which is 0 when the window is
    closed normally: bare `odock` therefore exits successfully on a plain close.

    This is where the **session** lives: the store is opened, the last session is
    offered (File ▸ Restore session, and a question when the platform can ask
    one), and the session is written when the window closes. Constructing a
    ``DockingWorkbench`` directly — which is what the tests and the simulation
    harness do — leaves persistence off entirely, so nothing is ever written to a
    user's session behind their back.

    Set ``ODOCK_GUI_AUTOQUIT_MS`` to close the window automatically after that
    many milliseconds. It exists so that the launch path can be exercised
    without a human, and is the only behaviour it changes.
    """
    _quiet_qt_messages()
    _configure_surface_format()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    store = dashboard.SessionStore.default()
    payload = store.load()
    # English is the startup default, whatever the operating system's locale says:
    # a Chinese label is fewer characters but every one of them is full-width, so
    # the layout that fits is the English one (see the table in docs/USER_GUIDE.md).
    # The switch and its persistence are untouched — a session or project that
    # recorded 中文 reopens in 中文, because those set the language explicitly.
    if not (payload or {}).get("language"):
        i18n.set_language("en")
    window = DockingWorkbench(
        receptor=receptor, ligand=ligand, poses=poses, session=store
    )
    window.show()
    if payload and not any((receptor, ligand, poses)):
        detail = window.session_offer()
        if detail:
            window._log(
                tr("dialog.restore_session.text", detail=detail).replace("\n\n", " — ")
            )
            # A modal question needs somebody who can answer it: an offscreen
            # (test/CI) Qt platform cannot, so the offer stays in the status bar
            # and in File ▸ Restore session instead of blocking the launch.
            if (
                app.platformName() != "offscreen"
                and not os.environ.get("ODOCK_NO_RESTORE_PROMPT")
            ):
                answer = QtWidgets.QMessageBox.question(
                    window,
                    tr("dialog.restore_session"),
                    tr("dialog.restore_session.text", detail=detail),
                )
                if answer == QtWidgets.QMessageBox.StandardButton.Yes:
                    window.restore_session(payload)
    autoquit = os.environ.get("ODOCK_GUI_AUTOQUIT_MS")
    if autoquit:
        try:
            QtCore.QTimer.singleShot(max(1, int(autoquit)), app.quit)
        except ValueError:  # pragma: no cover - a typo must not break the GUI
            pass
    return int(app.exec())


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point (`python -m odock.gui`)."""
    import argparse

    parser = argparse.ArgumentParser(prog="odock-gui", description=__doc__)
    parser.add_argument("-r", "--receptor", help=tr("cli.receptor"))
    parser.add_argument("-l", "--ligand", help=tr("cli.ligand"))
    parser.add_argument("-p", "--poses", help=tr("cli.poses"))
    args = parser.parse_args(argv)
    return launch_gui(receptor=args.receptor, ligand=args.ligand, poses=args.poses)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

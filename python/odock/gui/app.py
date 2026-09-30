# SPDX-License-Identifier: GPL-3.0-or-later
"""The OpenDocking 3-D workbench.

The window follows the layout in the project brief §4: a seven-menu bar, a workspace
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

import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from . import dialogs, i18n
from .i18n import tr
from .sequence import SequenceTrack
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

    MODE_ORBIT = "orbit"
    MODE_MEASURE = "measure"
    MODE_BOND = "bond"

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
        self.setMinimumSize(420, 320)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

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
        painter.fillRect(self.rect(), QtGui.QColor(22, 24, 32))
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
            self.renderer.draw(self.camera, width, height, target=self.framebuffer)
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

    def _paint_overlay(self, painter: QtGui.QPainter) -> None:
        """A small HUD: the interaction legend and the active tool."""
        interactions = self.scene.interactions or []
        if interactions:
            kinds: List[str] = []
            for item in interactions:
                kind = getattr(item, "kind", "")
                if kind and kind not in kinds:
                    kinds.append(kind)
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            y = 14
            for kind in kinds:
                r, g, b, _ = INTERACTION_COLORS.get(kind, (0.8, 0.8, 0.8, 1.0))
                painter.setBrush(QtGui.QColor(int(r * 255), int(g * 255), int(b * 255)))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawEllipse(10, y - 6, 8, 8)
                painter.setPen(QtGui.QColor(210, 220, 230))
                count = sum(1 for i in interactions if getattr(i, "kind", "") == kind)
                painter.drawText(24, y + 2, f"{_interaction_label(kind)} ({count})")
                y += 16

        if self.mode != self.MODE_ORBIT:
            painter.setPen(QtGui.QColor(255, 200, 120))
            painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.DemiBold))
            hint = {
                self.MODE_MEASURE: tr("viewport.hint_measure"),
                self.MODE_BOND: tr("viewport.hint_bond"),
            }.get(self.mode, self.mode)
            painter.drawText(10, self.height() - 10, hint)

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

    def snapshot(self, path, width: int = 2400, height: int = 1600) -> bool:
        """Render a high-resolution image straight from the GL context."""
        if not self._ensure_context() or self.renderer is None:
            return False
        data = self.renderer.render_image(self.camera, int(width), int(height))
        image = QtGui.QImage(
            data, int(width), int(height), int(width) * 3,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        return bool(image.save(str(path)))

    # -- pose animation -----------------------------------------------------

    def play_animation(self, frames: Sequence[Sequence], interval_ms: int = 90) -> None:
        """Interpolate between poses (the project brief §E.3, the conformer player)."""
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

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.settings = settings
        self.engine = None

    def run(self) -> None:  # noqa: D401
        try:
            import time as _time

            from odock.docking import build_engine, result_from_engine

            s = self.settings
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
            self.ticked.emit(tr("worker.searching"))
            t0 = _time.perf_counter()
            raw = engine.run()
            elapsed = _time.perf_counter() - t0
            self.ticked.emit(tr("worker.search_done", seconds=elapsed))
            if raw.get("cancelled"):
                self.ticked.emit(tr("worker.cancelled"))
            result = result_from_engine(
                engine,
                raw,
                elapsed=elapsed,
                box=s["box"],
                receptor_pdbqt=s["receptor_text"],
                ligand_pdbqt=s["ligand_text"],
                energy_range=s["energy_range"],
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # pragma: no cover - reported to the user
            import traceback

            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

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
        self.receptor_mol = None
        self.ligand_mol = None
        self.inventory = None
        self._reference_pose = None
        self._worker: Optional[QtCore.QThread] = None
        self._background: Optional[QtCore.QThread] = None
        self._pending: Dict[str, object] = {}
        self._measurements: List[dict] = []
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

        self._build_viewport()
        self._build_inspector()
        self._build_workspace()
        self._build_bottom()
        self._build_selection_panel()
        self._build_menus()
        self._build_statusbar()
        self._apply_style()

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
        self._build_selection_panel()
        self._build_menus()
        self._build_statusbar()
        self._apply_style()
        self.setWindowTitle(tr("app.title"))
        self._restore_ui_state(state)

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
            "pose_index": self.pose_slider.value() if self.pose_models else None,
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

        for name in ("workspace_dock", "inspector_dock", "bottom_dock", "selection_dock"):
            dock = getattr(self, name, None)
            if dock is not None:
                self.removeDockWidget(dock)
                dock.setParent(None)
                dock.deleteLater()
                setattr(self, name, None)

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

    def _restore_ui_state(self, state: dict) -> None:
        """Put the captured session back into the freshly built widgets."""
        self._charge_model = state["charge_model"]

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
            self.pose_slider.setValue(state["pose_index"])
            if state["pose_row"] >= 0:
                self.table.selectRow(state["pose_row"])

        self._set_running(state["running"])
        self._paused = bool(state["paused"]) and bool(state["running"])
        if self._paused:
            self.btn_pause.setText(tr("btn.resume"))

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
        self.inspector.setCurrentIndex(state["tab"])

        self.log.setPlainText(state["log"])
        self.lbl_status.setText(state["status"])
        self._log(
            tr("log.language", language=i18n.language_name(i18n.current_language()))
        )

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
        self.tree.setColumnWidth(0, 220)
        self.tree.itemClicked.connect(self._on_tree_clicked)
        dock.setWidget(self.tree)
        dock.setMinimumWidth(200)
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
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.setMinimumHeight(90)
        poses_layout.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("label.pose")))
        self.pose_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.pose_slider.setMinimumWidth(60)  # keep the dock narrow-able
        self.pose_slider.setRange(0, 0)
        self.pose_slider.valueChanged.connect(self._on_pose_changed)
        row.addWidget(self.pose_slider, 1)
        self.lbl_pose = QtWidgets.QLabel(tr("label.no_poses"))
        row.addWidget(self.lbl_pose)
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
        analysis.addSeparator()
        self._act(analysis, tr("action.diagram_svg"), lambda: self._export("svg"))
        # The lambda swallows the ``checked`` bool QAction.triggered emits: this
        # action is not checkable, so connected straight to the method it would
        # arrive as ``checked=False`` and stop the player instead of starting it.
        self._act(analysis, tr("action.play_poses"), lambda: self._toggle_playback(True))
        analysis.addSeparator()
        self._act(analysis, tr("action.export_xlsx"), lambda: self._export("xlsx"))
        self._act(analysis, tr("action.export_csv"), lambda: self._export("csv"))

        view = bar.addMenu(tr("menu.view"))
        self._style_actions = {
            "receptor": self._build_style_menu(
                view, tr("menu.protein_style"), PROTEIN_STYLES, "receptor"
            ),
            "ligand": self._build_style_menu(
                view, tr("menu.ligand_style"), LIGAND_STYLES, "ligand"
            ),
        }
        view.addSeparator()
        self._act(view, tr("action.bond_check"), self._open_bond_check)
        self._act(
            view,
            tr("action.interaction_distances"),
            self._choose_interaction_distances,
        )
        view.addSeparator()
        self.axes_action = self._act(
            view, tr("action.axes"), self._toggle_axes, checkable=True
        )
        self.axes_action.setChecked(self.scene.show_axes)
        self.ssao_action = self._act(
            view, tr("action.ssao"), self._toggle_ssao, checkable=True
        )
        self.ssao_action.setChecked(self.scene.ssao)
        self._act(view, tr("action.measure"), lambda: self._set_tool("measure"))
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
            self.selection_dock,
        ):
            panels.addAction(dock.toggleViewAction())
        self.stack_action = self._act(
            view, tr("action.stack_pose_dock"), self._set_pose_split, checkable=True
        )
        self.stack_action.setChecked(self._pose_stacked)
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

    # -- styling ------------------------------------------------------------

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #10141c; color: #d8e2ee; font-size: 12px; }
            QGroupBox { border: 1px solid #232b3a; border-radius: 5px; margin-top: 9px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #7fa7d0; }
            QPushButton {
                background: #1b2230; border: 1px solid #2c3648; border-radius: 4px;
                padding: 4px 9px;
            }
            QPushButton:hover { background: #243046; }
            QPushButton#primary { background: #1d5f8a; border-color: #2f86bd; font-weight: bold; }
            QPushButton#primary:hover { background: #24719f; }
            QTableWidget, QTreeWidget, QListWidget, QPlainTextEdit, QLineEdit, QComboBox,
            QSpinBox, QDoubleSpinBox {
                background: #151b26; border: 1px solid #26303f; border-radius: 3px;
                selection-background-color: #1d5f8a;
            }
            QTabBar::tab { background: #151b26; padding: 5px 12px; border: 1px solid #26303f; }
            QTabBar::tab:selected { background: #1d5f8a; }
            QMenuBar::item:selected, QMenu::item:selected { background: #1d5f8a; }
            QFrame#toolStrip { background: rgba(18, 24, 34, 190); border: 1px solid #2c3648;
                               border-radius: 6px; }
            QToolButton { color: #cfe2f5; padding: 2px 6px; }
            QToolButton:hover { background: #243046; border-radius: 4px; }
            QProgressBar { border: 1px solid #26303f; border-radius: 3px; text-align: center; }
            QProgressBar::chunk { background: #1d5f8a; }
            QSplitter::handle { background: #232b3a; }
            QSplitter::handle:hover { background: #2f86bd; }
            QStatusBar { background: #0c1016; }
            """
        )

    # -- logging ------------------------------------------------------------

    def _log(self, message: str) -> None:
        self.log.appendPlainText(message)
        self.lbl_status.setText(message if len(message) < 120 else message[:117] + "…")

    # -- project ------------------------------------------------------------

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
        self.pose_slider.setRange(0, 0)
        self.poses_text = None
        self._maps_computed = False
        self._filter_verdict = None
        self.lbl_receptor.setText(tr("label.no_receptor"))
        self.lbl_ligand.setText(tr("label.no_ligand"))
        self.lbl_pose.setText(tr("label.no_poses"))
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
        self._log(tr("log.new_project"))

    def _save_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            tr("dialog.save_project"),
            "odock-project.json",
            tr("filter.project"),
        )
        if not path:
            return
        box = self._current_box()
        payload = {
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
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._log(tr("log.saved_project", name=Path(path).name))

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

    def load_receptor(self, path, interactive: bool = False) -> bool:
        try:
            text, label = self._read_pdbqt(path)
            models = parse_pdbqt(text)
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), interactive)
            return False
        self.receptor_text = text
        self._pending["receptor"] = self._remember_path(path)
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

    def load_ligand(self, path, interactive: bool = False) -> bool:
        try:
            text, label = self._read_pdbqt(path)
            models = parse_pdbqt(text)
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), interactive)
            return False
        self.ligand_text = text
        self._pending["ligand"] = self._remember_path(path)
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

    def load_poses(self, path, interactive: bool = False) -> bool:
        try:
            text, label = self._read_pdbqt(path)
            models = parse_pdbqt(text)
        except Exception as exc:
            self._report_error(tr("log.cannot_read", path=path, error=exc), interactive)
            return False
        self._pending["poses"] = self._remember_path(path)
        self.poses_text = text
        first_poses = not self.pose_models
        self.pose_models = models
        self._populate_table()
        self.pose_slider.setRange(0, max(0, len(models) - 1))
        self.pose_slider.setValue(0)
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
            QtWidgets.QTreeWidgetItem(
                self.tree, [tr("tree.interactions"), str(len(self.interactions))]
            )
        if self._measurements:
            QtWidgets.QTreeWidgetItem(
                self.tree, [tr("tree.measurements"), str(len(self._measurements))]
            )

    def _on_tree_clicked(self, item, column) -> None:
        index = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        if isinstance(index, int):
            self.pose_slider.setValue(index)

    # -- poses --------------------------------------------------------------

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.pose_models))
        current = self.pose_slider.value() if self.pose_slider.maximum() > 0 else 0
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
        self.lbl_pose.setText(text)
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

        This is the automatic half of "compute the possible interactions between
        receptor and ligand": choosing a pose immediately annotates its
        H-bonds, salt bridges, π–π and cation–π contacts, hydrophobic contacts
        and clashes, adds the contacting residues to the pose label, and fills
        the results table row.
        """
        analysis = _try_import("analysis")
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
        self.scene.interactions = list(found)
        # Pose browsing re-derives the focus from *this* pose's contacts whenever
        # the user has asked for the emphasis, so the indices are never stale, and
        # applies none otherwise. The camera and the clip are never touched here,
        # so two poses stay comparable.
        if self._focus_requested:
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
            self.lbl_pose.setText(
                self.lbl_pose.text() + tr("label.pose_binding", residues=summary)
            )
        self._populate_table()

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
        # Re-annotate with the new numbers so the view matches the dialog.
        if self.pose_models:
            self._annotate_pose_contacts()
        else:
            self._annotate_interactions(quiet=True)

    def _on_pose_changed(self, value: int) -> None:
        self._apply_pose(value)
        if self.table.currentRow() != value:
            self.table.blockSignals(True)
            self.table.selectRow(value)
            self.table.blockSignals(False)

    def _on_table_selection(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if rows:
            self.pose_slider.setValue(min(rows))

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
        self.interactions = []
        self.scene.interactions = []
        # Clearing the annotations clears the emphasis with them, and forgets the
        # request so browsing poses does not bring it back.
        self._focus_requested = False
        self.scene.interaction_focus = []
        self.viewport.refresh(upload_receptor=True)
        self._populate_table()
        self._rebuild_tree()
        self._log(tr("log.annotations_cleared"))

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

    def _clear_measurements(self) -> None:
        self._measurements = []
        self.scene.measurements = []
        self.viewport.refresh()
        self._rebuild_tree()

    def _on_atoms_picked(self, selection) -> None:
        if len(selection) < 2:
            return
        (kind_a, index_a), (kind_b, index_b) = selection[-2], selection[-1]
        pool = {"receptor": self.scene.receptor, "ligand": self.scene.ligand}
        try:
            first = pool[kind_a][index_a]
            second = pool[kind_b][index_b]
        except (KeyError, IndexError):
            return
        distance = math.dist((first.x, first.y, first.z), (second.x, second.y, second.z))
        self._measurements.append(
            {
                "kind": "distance",
                "a": f"{first.res_name}{first.res_id}:{first.name}",
                "b": f"{second.res_name}{second.res_id}:{second.name}",
                "value": distance,
            }
        )
        self.scene.measurements = list(self.scene.measurements) + [
            ((first.x, first.y, first.z), (second.x, second.y, second.z), (1.0, 0.85, 0.3, 1.0))
        ]
        self._log(
            tr(
                "log.distance",
                a=f"{first.res_name}{first.res_id}:{first.name}",
                b=f"{second.res_name}{second.res_id}:{second.name}",
                value=distance,
            )
        )
        self.viewport.refresh()
        self._rebuild_tree()

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
        if which == "receptor":
            self.scene.style_protein = value
            self.viewport.refresh(upload_receptor=True)
        else:
            self.scene.style_ligand = value
            self.viewport.refresh()
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
        self.scene.show_receptor = bool(checked)
        self.viewport.refresh(upload_receptor=True)

    def _toggle_ligand(self, checked: bool) -> None:
        self.scene.show_ligand = bool(checked)
        self.viewport.refresh()

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
        worker.finished_ok.connect(self._on_docking_done)
        worker.failed.connect(self._on_docking_failed)
        worker.finished.connect(lambda: self._set_running(False))
        self._worker = worker
        self._set_running(True)
        self.progress.setRange(0, 0)
        worker.start()

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
        self.load_poses(result.to_pdbqt())
        self._log(tr("log.contacts_hint"))

    def _on_docking_failed(self, message: str) -> None:
        self._log(tr("log.docking_failed", message=message.splitlines()[0]))
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

    # -- misc ---------------------------------------------------------------

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        strip = getattr(self, "_tool_strip", None)
        if strip is not None:
            strip.adjustSize()
            strip.move(12, max(12, self.viewport.height() - strip.height() - 12))

    def closeEvent(self, event) -> None:  # noqa: N802
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

    Set ``ODOCK_GUI_AUTOQUIT_MS`` to close the window automatically after that
    many milliseconds. It exists so that the launch path can be exercised
    without a human, and is the only behaviour it changes.
    """
    _quiet_qt_messages()
    _configure_surface_format()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = DockingWorkbench(receptor=receptor, ligand=ligand, poses=poses)
    window.show()
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

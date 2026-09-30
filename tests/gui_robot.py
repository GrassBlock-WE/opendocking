# SPDX-License-Identifier: GPL-3.0-or-later
"""A tiny GUI robot used to drive the workbench like a user.

The point of this module is to make "did the window actually work?" an
executable question. It deliberately uses **real Qt input events**
(:class:`PyQt6.QtTest.QTest`) for everything inside the 3-D viewport and for
buttons, so the code paths a human would exercise — mouse press, drag, release,
wheel, click — are the ones under test. Menu entries are triggered through
their :class:`QAction`, which is exactly what Qt does when the user picks the
item; the action is first looked up, so a deleted or renamed entry still fails
the test.

Every step records a screenshot, so the report that comes out of a run shows
what the window looked like at that moment, not just a boolean.
"""

from __future__ import annotations

import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402

__all__ = ["Robot", "Step"]


class Step:
    """One scripted interaction and its verdict."""

    def __init__(self, number: int, title: str, area: str) -> None:
        self.number = number
        self.title = title
        self.area = area
        self.observations: List[str] = []
        self.failures: List[str] = []
        self.skipped: Optional[str] = None
        self.screenshot: Optional[Path] = None

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "FAIL" if self.failures else "PASS"

    def expect(self, condition, description: str, detail: str = "") -> bool:
        if condition:
            self.observations.append(f"{description}{(' — ' + detail) if detail else ''}")
            return True
        self.failures.append(f"{description}{(' — ' + detail) if detail else ''}")
        return False

    def note(self, message: str) -> None:
        self.observations.append(message)

    def skip(self, reason: str) -> None:
        self.skipped = reason


class Robot:
    """Drives a :class:`~odock.gui.app.DockingWorkbench` and records the run."""

    def __init__(
        self,
        receptor: Optional[str] = None,
        ligand: Optional[str] = None,
        poses: Optional[str] = None,
        *,
        output_dir: Path,
        size: Tuple[int, int] = (1440, 900),
        autostart: bool = True,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        if str(root / "python") not in sys.path:
            sys.path.insert(0, str(root / "python"))

        from odock.gui.app import DockingWorkbench, _configure_surface_format

        _configure_surface_format()
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.modals: List[str] = []
        self._silence_modal_dialogs()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.steps: List[Step] = []
        self._index = 0
        self.window = DockingWorkbench()
        self.window.resize(*size)
        if autostart:
            self.window.show()
            self.pump(120)

        if receptor:
            self.window.load_receptor(receptor)
        if ligand:
            self.window.load_ligand(ligand)
        if poses:
            self.window.load_poses(poses)
        self.pump(120)

    # -- event pump ---------------------------------------------------------

    def _silence_modal_dialogs(self) -> None:
        """Replace the modal message boxes with recorders.

        A modal ``QMessageBox`` waits for a human, so an unattended run would
        hang forever on the first error report. Recording the text instead is
        what a GUI test harness should do — and it makes the message assertable,
        which is more useful than a screenshot of it.
        """
        record = self.modals

        def make(level: str):
            def handler(parent=None, title="", text="", *args, **kwargs):
                record.append(f"{level}: {title}: {text}")
                return QtWidgets.QMessageBox.StandardButton.Ok

            return staticmethod(handler)

        for name in ("critical", "warning", "information", "question"):
            setattr(QtWidgets.QMessageBox, name, make(name))

        # `QDialog.exec()` waits for a human too. Accepting immediately keeps the
        # caller's "if accepted: use the selection" path alive — which is the
        # path worth exercising — and the title is recorded so the report shows
        # which dialog opened.
        def auto_exec(dialog):
            record.append(f"dialog: {dialog.windowTitle()}")
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        QtWidgets.QDialog.exec = auto_exec

    def modal_text(self) -> str:
        return "\n".join(self.modals)

    def pump(self, ms: int = 60) -> None:
        """Let Qt process events for a while (timers, paints, workers)."""
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while deadline.elapsed() < ms:
            self.app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 10)
            QTest.qWait(5)

    def wait_for(self, predicate: Callable[[], bool], timeout_ms: int = 60000) -> bool:
        """Wait until ``predicate`` holds, pumping events meanwhile."""
        timer = QtCore.QElapsedTimer()
        timer.start()
        while timer.elapsed() < timeout_ms:
            if predicate():
                return True
            self.pump(50)
        return bool(predicate())

    # -- steps --------------------------------------------------------------

    @contextmanager
    def step(self, title: str, area: str = ""):
        self._index += 1
        current = Step(self._index, title, area)
        self.steps.append(current)
        try:
            yield current
        except Exception:
            current.failures.append("exception: " + traceback.format_exc(limit=3))
        finally:
            try:
                self.pump(80)
                current.screenshot = self.shot(f"{self._index:02d}-{_slug(title)}")
            except Exception:  # pragma: no cover - screenshotting must not abort
                pass
            print(f"[{current.status}] {current.number:>2}. {title}")
            for failure in current.failures:
                print(f"        ! {failure}")

    # -- screenshots --------------------------------------------------------

    def shot(self, name: str) -> Path:
        pixmap = self.window.grab()
        path = self.output_dir / f"{name}.png"
        pixmap.save(str(path))
        return path

    def viewport_shot(self, name: str) -> Path:
        path = self.output_dir / f"{name}.png"
        self.window.viewport.grab().save(str(path))
        return path

    # -- real input ---------------------------------------------------------

    def click(self, widget: QtWidgets.QWidget, *, double: bool = False) -> None:
        if double:
            QTest.mouseDClick(widget, QtCore.Qt.MouseButton.LeftButton)
        else:
            QTest.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)
        self.pump(40)

    def type_text(self, widget: QtWidgets.QLineEdit, text: str) -> None:
        widget.clear()
        QTest.keyClicks(widget, text)
        self.pump(20)

    def set_value(self, spin, value) -> None:
        spin.setValue(value)
        self.pump(20)

    def viewport_pos(self, x: float, y: float) -> QtCore.QPoint:
        """A position inside the viewport, in widget coordinates."""
        return QtCore.QPoint(int(x), int(y))

    def viewport_click(self, x: float, y: float, *, double: bool = False,
                       modifier=None) -> None:
        point = self.viewport_pos(x, y)
        if double:
            QTest.mouseDClick(self.window.viewport, QtCore.Qt.MouseButton.LeftButton,
                              modifier or QtCore.Qt.KeyboardModifier.NoModifier, point)
        else:
            QTest.mouseClick(self.window.viewport, QtCore.Qt.MouseButton.LeftButton,
                             modifier or QtCore.Qt.KeyboardModifier.NoModifier, point)
        self.pump(60)

    def viewport_drag(self, start, end, *, button=QtCore.Qt.MouseButton.LeftButton,
                      modifier=None, steps: int = 6) -> None:
        """Press, move in several increments (a real drag), release."""
        modifier = modifier or QtCore.Qt.KeyboardModifier.NoModifier
        widget = self.window.viewport
        start_point = self.viewport_pos(*start)
        end_point = self.viewport_pos(*end)
        QTest.mousePress(widget, button, modifier, start_point)
        for i in range(1, steps + 1):
            t = i / steps
            point = QtCore.QPoint(
                int(start_point.x() + (end_point.x() - start_point.x()) * t),
                int(start_point.y() + (end_point.y() - start_point.y()) * t),
            )
            QTest.mouseMove(widget, point)
            self.pump(12)
        QTest.mouseRelease(widget, button, modifier, end_point)
        self.pump(60)

    def wheel(self, delta: int = 120, x: Optional[float] = None, y: Optional[float] = None) -> None:
        widget = self.window.viewport
        point = self.viewport_pos(
            x if x is not None else widget.width() / 2,
            y if y is not None else widget.height() / 2,
        )
        event = QtGui.QWheelEvent(
            QtCore.QPointF(point),
            QtCore.QPointF(widget.mapToGlobal(point)),
            QtCore.QPoint(0, 0),
            QtCore.QPoint(0, delta),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
            QtCore.Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        self.app.sendEvent(widget, event)
        self.pump(40)

    # -- menus and widgets --------------------------------------------------

    def menu(self, title: str) -> QtWidgets.QMenu:
        for action in self.window.menuBar().actions():
            if action.text().replace("&", "") == title.replace("&", ""):
                return action.menu()
        raise AssertionError(f"no menu named {title!r}")

    def action(self, menu_title: str, item: str) -> QtGui.QAction:
        """Find a menu entry by (case-insensitive) prefix."""
        wanted = item.replace("&", "").lower()
        for action in self.menu(menu_title).actions():
            text = action.text().replace("&", "")
            if text.lower().startswith(wanted) or wanted in text.lower():
                return action
        raise AssertionError(f"no entry {item!r} in the {menu_title!r} menu")

    def trigger(self, menu_title: str, item: str) -> QtGui.QAction:
        """Trigger a menu entry exactly as clicking it would."""
        action = self.action(menu_title, item)
        action.trigger()
        self.pump(60)
        return action

    def button(self, text: str) -> QtWidgets.QPushButton:
        for button in self.window.findChildren(QtWidgets.QPushButton):
            if button.text().replace("&", "").startswith(text) or text in button.text():
                return button
        raise AssertionError(f"no button matching {text!r}")

    def tool_button(self, tooltip: str) -> QtWidgets.QToolButton:
        for button in self.window.findChildren(QtWidgets.QToolButton):
            if tooltip in (button.toolTip() or ""):
                return button
        raise AssertionError(f"no tool button with tooltip {tooltip!r}")

    # -- observations -------------------------------------------------------

    def log_text(self) -> str:
        return self.window.log.toPlainText()

    def log_tail(self, lines: int = 6) -> str:
        return "\n".join(self.log_text().splitlines()[-lines:])

    def scene_summary(self) -> dict:
        scene = self.window.scene
        return {
            "receptor": len(scene.receptor),
            "ligand": len(scene.ligand),
            "interactions": len(scene.interactions or []),
            "box": scene.box,
            "style_receptor": scene.style_receptor,
            "style_ligand": scene.style_ligand,
            "axes": scene.show_axes,
            "ssao": scene.ssao,
            "highlight": len(scene.highlight or []),
            "measurements": len(scene.measurements or []),
            "locked": len(scene.locked_bonds or []),
        }

    # -- report -------------------------------------------------------------

    def write_report(self, path: Path, title: str = "OpenDocking workbench — simulated user session") -> Path:
        passed = sum(1 for s in self.steps if s.status == "PASS")
        failed = [s for s in self.steps if s.status == "FAIL"]
        skipped = [s for s in self.steps if s.status == "SKIP"]
        lines = [
            f"# {title}",
            "",
            f"* steps: **{len(self.steps)}**",
            f"* passed: **{passed}**",
            f"* failed: **{len(failed)}**",
            f"* skipped: **{len(skipped)}**",
            f"* Qt platform: `{self.app.platformName()}`",
            "",
            "Every screenshot below was captured from the live window at that step.",
            "",
            "| # | area | step | result |",
            "|---|---|---|---|",
        ]
        for step in self.steps:
            lines.append(
                f"| {step.number} | {step.area} | {step.title} | **{step.status}** |"
            )
        lines.append("")
        for step in self.steps:
            lines.append(f"## {step.number}. {step.title}")
            lines.append("")
            if step.area:
                lines.append(f"*area: {step.area}*")
                lines.append("")
            if step.skipped:
                lines.append(f"**skipped:** {step.skipped}")
                lines.append("")
            for observation in step.observations:
                lines.append(f"* ✔ {observation}")
            for failure in step.failures:
                lines.append(f"* ✘ {failure}")
            lines.append("")
            if step.screenshot and step.screenshot.exists():
                lines.append(f"![{step.title}]({step.screenshot.name})")
                lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _slug(text: str) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:48] or "step"

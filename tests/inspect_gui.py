"""Build the workbench and report its structure — in both languages.

Every label is printed twice: once in English and once in 简体中文, with the
window rebuilt in place by ``DockingWorkbench.set_language`` exactly the way the
View ▸ Language menu does it. No display is required.

Run: python tests/inspect_gui.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

try:  # keep the Chinese labels readable when stdout is a cp936 console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - depends on the interpreter
    pass

from PyQt6 import QtWidgets  # noqa: E402

from odock.gui import i18n  # noqa: E402
from odock.gui.app import DockingWorkbench, _configure_surface_format  # noqa: E402


def describe(window: DockingWorkbench, code: str) -> None:
    """Print the whole widget structure for the language now in force."""
    label = i18n.language_name(code)
    print()
    print(f"=== {label} ({code}) ===")
    print("title       :", window.windowTitle())
    print("menus       :", [action.text() for action in window.menuBar().actions()])
    for action in window.menuBar().actions():
        menu = action.menu()
        if menu is not None:
            items = []
            for sub in menu.actions():
                if sub.menu() is not None:
                    items.append(f"{sub.text()}[{len(sub.menu().actions())}]")
                elif sub.isSeparator():
                    items.append("|")
                else:
                    items.append(sub.text())
            print(f"  {action.text():<10}: {' '.join(items)}")
    print("docks       :", [d.windowTitle() for d in window.findChildren(QtWidgets.QDockWidget)])
    print("inspector   :", [window.inspector.tabText(i) for i in range(window.inspector.count())])
    print("tree cols   :", [window.tree.headerItem().text(i) for i in range(window.tree.columnCount())])
    print("pose table  :", [window.table.horizontalHeaderItem(i).text()
                            for i in range(window.table.columnCount())])
    print("tool strip  :", [b.toolTip() for b in window._tool_strip.findChildren(QtWidgets.QToolButton)])
    print("buttons     :", [b.text() for b in (
        window.btn_run, window.btn_pause, window.btn_abort, window.btn_score,
        window.btn_play, window.btn_maps)])
    print("status      :", window.lbl_status.text())


def main() -> int:
    _configure_surface_format()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = DockingWorkbench()
    window.resize(1200, 800)
    window.show()
    app.processEvents()

    for code in ("en", "zh"):
        window.set_language(code)
        app.processEvents()
        describe(window, code)

    # Switching back has to leave a working, English window behind.
    window.set_language("en")
    app.processEvents()
    window.set_language("zh")
    app.processEvents()
    assert window.inspector.tabText(0) == "受体", "the language switch did not take"
    window.set_language("en")
    app.processEvents()
    assert window.inspector.tabText(0) == "Receptor", "the language did not revert"
    window.close()
    print()
    print("INSPECT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

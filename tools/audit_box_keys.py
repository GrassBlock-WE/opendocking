# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit the search-box collection strings (task-60).

Checks the things a translation can silently get wrong, and the one thing this
feature is *about*: the Chinese must not blur 隐藏 (rendering) with 删除
(existence). Run with no arguments:

    .venv\\Scripts\\python.exe -X utf8 tools\\audit_box_keys.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from _tool import use_utf8_streams  # noqa: E402

use_utf8_streams()

from odock.gui import i18n  # noqa: E402

#: The 18 keys, grouped the way the feature distinguishes them.
KEYS = (
    "menu.boxes",
    "tab.grid.boxes",
    "boxes.name",
    "boxes.active",
    "boxes.none",
    "boxes.visible",
    "boxes.hidden",
    "action.add_box",
    "action.delete_box",
    "action.box_show",
    "action.box_hide",
    "undo.add_box",
    "undo.delete_box",
    "undo.move_box",
    "undo.resize_box",
    "undo.show_box",
    "undo.hide_box",
    "log.box_active",
)

#: Keys that carry a placeholder, and which one.
PLACEHOLDERS = {
    "boxes.name": {"n"},
    "boxes.active": {"name"},
    "log.box_active": {"name"},
}

#: Characters that mean *rendering a box* in Chinese, and characters that mean
#: *the box no longer exists*. A string may use either; it must not use the
#: wrong one, which is the defect this whole feature exists to prevent.
RENDERING = ("显示", "隐藏")
EXISTENCE = ("删除", "添加", "移动", "调整")


def tokens(text: str) -> set:
    return set(re.findall(r"\{(\w+)\}", text))


def main() -> int:
    problems = []
    print(f"EN keys {len(i18n.EN)}, ZH keys {len(i18n.ZH)}")
    if set(i18n.EN) != set(i18n.ZH):
        problems.append(
            "key sets differ: %s" % sorted(set(i18n.EN) ^ set(i18n.ZH))[:8]
        )
    if len(i18n.EN) != len(i18n.ZH):
        problems.append(f"counts differ: {len(i18n.EN)} vs {len(i18n.ZH)}")

    print()
    print("present in both languages")
    for key in KEYS:
        en = i18n.EN.get(key)
        zh = i18n.ZH.get(key)
        if en is None or zh is None:
            problems.append(f"{key} missing in {'EN' if en is None else 'ZH'}")
            continue
        if not any(ord(character) > 127 for character in zh):
            problems.append(f"{key} ZH is ASCII: {zh!r}")
        if "&" in en or "&" in zh:
            problems.append(f"{key} contains an ampersand")
        expected = PLACEHOLDERS.get(key, set())
        if tokens(en) != expected:
            problems.append(f"{key} EN placeholders {tokens(en)} != {expected}")
        if tokens(zh) != expected:
            problems.append(f"{key} ZH placeholders {tokens(zh)} != {expected}")
        print(f"  {key:<20} EN {en!r:<28} ZH {zh!r}")

    print()
    print("length limits")
    for key in KEYS:
        if key.startswith("action."):
            length = len(i18n.EN[key])
            verdict = "ok" if length <= 22 else "TOO LONG"
            print(f"  {key:<20} {length:>2} chars  {verdict}")
            if length > 22:
                problems.append(f"{key} is {length} chars (limit 22)")
        if key.startswith("btn."):
            length = len(i18n.EN[key].rstrip("…"))
            if length > 18:
                problems.append(f"{key} is {length} chars (limit 18)")

    print()
    print("the distinction the feature is about: 隐藏 (rendering) vs 删除 (existence)")
    expectations = {
        "boxes.visible": ("显示", None),
        "boxes.hidden": ("隐藏", None),
        "action.box_show": ("显示", None),
        "action.box_hide": ("隐藏", None),
        "undo.show_box": ("显示", None),
        "undo.hide_box": ("隐藏", None),
        "action.delete_box": ("删除", None),
        "undo.delete_box": ("删除", None),
        "action.add_box": ("添加", None),
        "undo.add_box": ("添加", None),
        "undo.move_box": ("移动", None),
        "undo.resize_box": ("调整", None),
    }
    for key, (word, _unused) in expectations.items():
        text = i18n.ZH[key]
        if word not in text:
            problems.append(f"{key} should say {word} but says {text!r}")
        # The wrong family must not appear: a hide string may not say 删除, and a
        # delete string may not say 隐藏.
        wrong = EXISTENCE if word in RENDERING else RENDERING
        clash = [other for other in wrong if other in text]
        if clash:
            problems.append(f"{key} mixes families: {text!r} contains {clash}")
        print(f"  {key:<20} {text!r:<16} {'uses ' + word:<8} {'clean' if not clash else 'MIXED ' + str(clash)}")

    print()
    print("the workbench constructs in both languages")
    try:
        from PyQt6 import QtWidgets

        from odock.gui.app import DockingWorkbench

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        for language in ("en", "zh"):
            i18n.set_language(language)
            window = DockingWorkbench()
            window.resize(800, 560)
            app.processEvents()
            window.close()
            print(f"  {language}: constructed")
    except Exception as exc:  # pragma: no cover - reported, never fatal
        problems.append(f"a workbench could not be built: {type(exc).__name__}: {exc}")
    finally:
        i18n.set_language("en")

    print()
    if problems:
        print(f"FAILED: {len(problems)} problem(s)")
        for problem in problems:
            print("  -", problem)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

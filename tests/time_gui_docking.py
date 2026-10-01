"""Time the workbench's own docking path, with the worker's log visible."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from PyQt6 import QtWidgets  # noqa: E402

from gui_robot import Robot  # noqa: E402

robot = Robot(output_dir=ROOT / "out" / "timing")
window = robot.window
window.load_receptor(str(ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt"))
window.load_ligand(str(ROOT / "demo" / "systems" / "3ptb" / "ligand.pdbqt"))
robot.pump(300)
window._fit_box_to_ligand()
window.exhaustiveness.setValue(2)
window.poses.setValue(3)
window.seed.setValue(42)
robot.pump(60)
box = window._current_box()
print("box:", [round(v, 2) for v in box.center], [round(v, 2) for v in box.size], box.spacing)
print("exhaustiveness:", window.exhaustiveness.value(), "poses:", window.poses.value())

start = time.perf_counter()
window._run_docking()
finished = robot.wait_for(lambda: window.result is not None, timeout_ms=180000)
print(f"finished={finished} after {time.perf_counter() - start:.1f}s")
print("--- log ---")
print(robot.log_text()[-2500:])
print("--- modals ---")
print(robot.modal_text()[:1500])
window.close()

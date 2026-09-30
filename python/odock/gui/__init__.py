# SPDX-License-Identifier: GPL-3.0-or-later
"""The OpenDocking 3-D workbench: PyQt6 + ModernGL.

Importing this package requires the ``gui`` extra::

    pip install 'opendocking[gui]'

The workbench is a viewer *and* a driver: it loads a receptor PDBQT, a ligand
PDBQT and (optionally) a pose file, lets you drag the search box in 3-D, and
runs the same Rust kernel the CLI and the Python API use.
"""

from __future__ import annotations

__all__ = ["DockingWorkbench", "Renderer", "Scene", "Camera", "launch_gui", "main"]

from .structure import Atom, Model, parse_pdbqt
from .viewport import Camera, Renderer, Scene

try:  # pragma: no cover - depends on the optional PyQt6 dependency
    from .app import DockingWorkbench, launch_gui, main
except Exception as exc:  # pragma: no cover
    # PyQt6/ModernGL are optional; importing the viewer internals must still
    # work so that head-less environments can use `odock.gui.structure`.
    _IMPORT_ERROR = exc

    def launch_gui(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "the OpenDocking workbench needs PyQt6 and ModernGL; install them "
            "with `pip install 'opendocking[gui]'`"
        ) from _IMPORT_ERROR

    def main(*args, **kwargs):  # type: ignore[misc]
        return launch_gui(*args, **kwargs)

    DockingWorkbench = None  # type: ignore[assignment]

# SPDX-License-Identifier: GPL-3.0-or-later
"""One capability guard for the workbench tests.

A runner with no OpenGL driver is a **capability fact**, not a defect. Asserting that a
renderer exists turns "this machine cannot make a GL context" into a failing test --
measured on the windows-latest CI runner, where `moderngl` imports fine and creating the
context fails with

    wglCreateContextAttribsARB not found

`pytest.importorskip("moderngl")` cannot catch that, because the module is present. The
skip message names ``moderngl``/``PyQt6`` so that ``--require-gui`` still escalates it to a
failure: on a machine that declared the GUI extra, a missing context IS the bug, and that
flag exists to surface exactly that.

Three test modules need this, so it lives here rather than being copied into each of them.
"""

from __future__ import annotations

import pytest

__all__ = ["require_gl"]


def require_gl(viewport) -> None:
    """Skip unless ``viewport`` managed to create an OpenGL renderer.

    Args:
        viewport: anything with a ``renderer`` attribute (``None`` when GL failed) and a
            ``_gl_error`` string carrying the driver's own message.
    """
    if getattr(viewport, "renderer", None) is not None:
        return
    reason = getattr(viewport, "_gl_error", None) or "no details from the driver"
    pytest.skip(
        f"moderngl/PyQt6 cannot create an OpenGL context on this machine: {reason}"
    )

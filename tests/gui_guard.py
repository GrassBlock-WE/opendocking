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
    """Skip when the machine cannot make a GL context; fail when one was never asked for.

    The distinction matters and cost five passing tests the first time it was missed: with
    no renderer **and** a recorded GL error the machine cannot make a context (skip); with
    no renderer **and no error** the renderer was simply never created, which is a defect
    in the test or the workbench (fail). Treating the second as a skip turns a green run
    into a green run that did not run.

    Args:
        viewport: anything with a ``renderer`` attribute (``None`` until a context exists)
            and a ``_gl_error`` string recording why creation failed.
    """
    if getattr(viewport, "renderer", None) is not None:
        return
    reason = (getattr(viewport, "_gl_error", None) or "").strip()
    if not reason:
        pytest.fail(
            "the viewport has no renderer and recorded no GL error: the renderer was "
            "never created. That is a defect in the test or the workbench, not a machine "
            "without OpenGL -- a missing context always leaves an error behind."
        )
    pytest.skip(
        f"moderngl/PyQt6 cannot create an OpenGL context on this machine: {reason}"
    )

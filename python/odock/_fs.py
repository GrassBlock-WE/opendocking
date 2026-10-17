# SPDX-License-Identifier: GPL-3.0-or-later
"""Windows-safe removal of a scratch tree.

``shutil.rmtree`` raises ``PermissionError: [WinError 5]`` on a directory that
contains a **read-only** file, and an interrupted run leaves exactly that behind.
It bit this project twice in one working session -- ``odock tutorial`` (and the
two documentation tests that run it) and ``odock docs check``, both on a stale
``out/`` subtree -- and each time the traceback named neither the directory nor a
remedy, so the tool looked broken for a reason no user could act on.

The behaviour is identical at every call site, so it lives here once rather than
as four copies of a platform quirk.  ``shutil.rmtree(..., ignore_errors=True)``
is deliberately *not* used instead: swallowing the failure leaves half a tree and
a later, stranger error.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Union

__all__ = ["remove_tree"]

PathLike = Union[str, "os.PathLike[str]"]


def remove_tree(path: PathLike) -> None:
    """Remove a directory tree, clearing read-only flags first, then verify.

    Clearing the attribute **before** the removal is what makes this reliable
    rather than hopeful: on Windows a read-only directory cannot be unlinked, so
    ``rmtree`` stops part-way and reports ``WinError 5`` while its own error
    handler is already inside a failing call.

    Args:
        path: the directory to remove.  A path that does not exist is not an
            error -- callers use this to clear a previous run.

    Raises:
        OSError: if the tree is still there afterwards.  The message names the
            path and a remedy, because "Access is denied" on its own tells a user
            nothing they can act on.
    """
    target = Path(path)
    if not target.exists():
        return
    # The root itself first.  `os.walk` yields the root's *children*, so an empty
    # directory carrying the read-only attribute — the case that broke
    # `odock docs check` on a stale `out/docs-site` — never had its own flag
    # cleared, and `os.rmdir` then failed with WinError 5 on a directory that
    # looked empty to the code.
    try:
        os.chmod(target, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass
    for root, directories, files in os.walk(target):
        for name in files:
            try:
                os.chmod(os.path.join(root, name), stat.S_IWRITE)
            except OSError:
                pass  # it may have gone already; the removal reports what is left
        for name in directories:
            try:
                os.chmod(
                    os.path.join(root, name),
                    stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC,
                )
            except OSError:
                pass
    try:
        shutil.rmtree(target)
    except OSError as exc:
        # Re-raise with the path and a remedy: the raw exception says only
        # "Access is denied", which is what made the original bug unactionable.
        raise OSError(
            f"{target} could not be removed ({exc}).  Something is holding it open "
            f"-- close any viewer or editor using it, delete it by hand, or choose "
            f"another output directory."
        ) from exc
    if target.exists():
        raise OSError(
            f"{target} still exists after removal: something recreated it or is "
            f"holding it open.  Close any viewer or editor using it, or choose "
            f"another output directory."
        )

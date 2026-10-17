# SPDX-License-Identifier: GPL-3.0-or-later
"""One entry point for the analysis tools' output streams.

Why this file exists
--------------------
These tools print Å, Å², Å³, ·, — and → constantly, and on Windows a redirected
stdout is a **pipe**, which Python encodes with the platform codepage (cp936/GBK
on a Chinese Windows) rather than UTF-8. ``Å`` is not in GBK, so the first line of
a report raises::

    UnicodeEncodeError: 'gbk' codec can't encode character '\\xc5'

which means `python tools/cavity_report.py > report.txt` — the normal way anyone
saves a report — dies on the first table, while the same command in a terminal
looks perfectly fine. Measured, before this file existed: `cavity_report.py` and
`check_sasa_report.py` both exited 1 with that error when captured by a UTF-8
reader.

The fix is at the **boundary**, once, rather than at each print: the same lesson
as ``odock.cli._use_utf8_streams`` for the ``-m`` entry points, applied to the
tools. A tool with a report to print calls :func:`use_utf8_streams` before it
prints anything; ``errors="replace"`` is deliberate, because a mangled character
in a report is better than a traceback and a lost report.
"""

from __future__ import annotations

import sys


def use_utf8_streams() -> bool:
    """Make stdout/stderr UTF-8 for the rest of this process. Returns success.

    A no-op (returning ``False``) where ``reconfigure`` is unavailable — an
    older interpreter, or a stream someone replaced with a plain object — so a
    tool never fails because of its own output plumbing.
    """
    changed = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            changed = True
        except (ValueError, OSError):  # pragma: no cover - a closed or exotic stream
            continue
    return changed

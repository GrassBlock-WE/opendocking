# SPDX-License-Identifier: GPL-3.0-or-later
"""The output boundary: an `odock` command must survive a non-UTF-8 console.

Why this file exists
--------------------
On a Chinese Windows the console codepage is **GBK**, and a *redirected* stdout is a
**pipe**, which Python encodes with that codepage.  `odock` prints `Å`, `→`, `—`;
`Å` (U+00C5) is not in GBK, so:

* a command that printed it raised `UnicodeEncodeError` — the report died on its
  first line, while the same command in a terminal looked perfectly fine;
* a parent capturing with `encoding="utf-8"` got **`returncode == 0` and
  `stdout is None` at the same time**, because failing to decode leaves the attribute
  unset.  That is exactly how `tests/test_docs_site.py` failed for weeks while
  `python -m odock.tutorial --json` "worked" when run by hand.

Neither place a developer normally looks can see this: CI runs on a UTF-8 Linux
runner, and a shell usually has `PYTHONIOENCODING` set — the Lead's did, which is why
a green local run hid it.  So the test has to force the codepage rather than inherit
one.

The fix being pinned is a boundary fix, not a per-print one: `_use_utf8_streams()`
in `odock/cli.py` for the `-m` entry points, and `tools/_tool.py` for the analysis
tools.  A regression here is silent on a UTF-8 machine, which is the whole point of
asserting it explicitly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_gbk(argv: list[str]) -> subprocess.CompletedProcess:
    """Run a command with the child's console encoding forced to a non-UTF-8 codepage."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"
    env.pop("ODOCK_STYLE_SHOTS", None)
    return subprocess.run(
        argv, capture_output=True, env=env, cwd=str(ROOT), timeout=300
    )


def test_the_tutorial_json_survives_a_gbk_console():
    """The measured failure: two non-ASCII characters (`—` U+2014, `→` U+2192).

    The payload must come back as **valid UTF-8** even though the child was told its
    console speaks GBK, because the command reconfigures its own streams.  Reading it
    back as GBK would be the bug, not the fix.
    """
    completed = _run_gbk([sys.executable, "-m", "odock.tutorial", "--json"])
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-800:]
    text = completed.stdout.decode("utf-8")  # must not raise
    payload = json.loads(text)
    assert payload["format"] == "odock-tutorial"
    assert payload["ok"] is True
    assert len(payload["steps"]) == 8


def test_a_cli_subcommand_survives_a_gbk_console():
    """The same boundary, reached through `odock.cli` rather than a module entry.

    Both routes need the fix and they are separate code paths: `cli.main()` calls
    `_use_utf8_streams()` itself, while `python -m odock.tutorial` had to be routed
    to the same call because it never goes through `cli.main()` at all.
    """
    completed = _run_gbk([sys.executable, "-m", "odock.cli", "tutorial", "--json"])
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-800:]
    payload = json.loads(completed.stdout.decode("utf-8"))
    assert payload["format"] == "odock-tutorial"
    assert payload["ok"] is True


def test_the_release_gate_scripts_survive_a_gbk_console():
    """`inspect_dist.py` and `check_test_order.py` are release gates.

    A gate that raises `UnicodeEncodeError` because its own output was redirected is
    a gate that fails for a plumbing reason — the worst kind, because the report is
    what tells you whether the tree is sound.
    """
    completed = _run_gbk([sys.executable, "tools/inspect_dist.py", "--self-test"])
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")[-800:]
    text = completed.stdout.decode("utf-8")
    assert "self-test OK" in text

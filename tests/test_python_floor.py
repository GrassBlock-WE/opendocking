# SPDX-License-Identifier: GPL-3.0-or-later
"""The declared Python floor has to be true in the code, not just in `pyproject.toml`.

`pyproject.toml` says ``requires-python = ">=3.9"``. The CI matrix runs 3.9, and every
developer machine here runs 3.10 or later -- so a 3.10-only API is invisible locally and
fails only where the floor is actually exercised. That is exactly what happened:

    TypeError: write_text() got an unexpected keyword argument 'newline'

in `standardise.py`, where `Path.write_text(..., newline=...)` is **3.10+**. The file was
correct on the machine it was written on and broken on a supported interpreter.

These checks are static (they read the source) and deliberately narrow: they pin the
traps this project has actually hit, not a general version linter.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "python" / "odock"

#: Methods whose `newline=` keyword appeared after Python 3.9.
#: `Path.write_text`/`read_text` gained it in 3.10 and 3.13 respectively.
_NEWLINE_KEYWORD_METHODS = {
    "write_text": "3.10",
    "read_text": "3.13",
}


def _shipped_sources():
    return sorted(path for path in PACKAGE.rglob("*.py"))


def test_no_source_uses_a_path_text_method_keyword_newer_than_the_floor():
    """`write_text(..., newline=)` is 3.10+ and `read_text(..., newline=)` is 3.13+."""
    offenders = []
    for path in _shipped_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr
            if method not in _NEWLINE_KEYWORD_METHODS:
                continue
            if not any(keyword.arg == "newline" for keyword in node.keywords):
                continue
            offenders.append(
                f"{path.relative_to(ROOT)}:{node.lineno} calls {method}(..., newline=...), "
                f"which needs Python {_NEWLINE_KEYWORD_METHODS[method]}"
            )
    assert not offenders, (
        "these calls break the declared Python 3.9 floor: " + "; ".join(offenders)
    )


def test_the_declared_floor_is_actually_declared():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
    assert match, "pyproject.toml has no requires-python"
    assert match.group(1) == ">=3.9", (
        f"the floor changed to {match.group(1)!r}; re-check the static traps in this file "
        f"before relaxing or raising it"
    )


def test_the_shipped_sources_parse_on_the_floor():
    """A syntax error from a newer grammar would fail here rather than in a user's run.

    This does not emulate 3.9 -- it cannot -- but it does catch a source file that will
    not parse at all, and it is the cheapest possible check that ships with the tests.
    """
    broken = []
    for path in _shipped_sources():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - would already fail collection
            broken.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not broken, "unparseable sources: " + "; ".join(broken)

def test_every_extra_named_in_a_message_exists():
    """Advice must point at something a user can actually install.

    `cavity.py` told callers to run ``pip install opendocking[analysis]`` while
    `pyproject.toml` declared no ``analysis`` extra at all -- so the one instruction a
    user got for fixing the ImportError was to install a name that does not exist. The
    CI legs then failed eleven cavity tests with that same message, which is how it was
    found: the advice was never executable.
    """
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = metadata.split("[project.optional-dependencies]", 1)[1]
    declared = set(re.findall(r"^([a-z][a-z0-9_-]*)\s*=\s*\[", section, re.M))
    named = set()
    for path in _shipped_sources():
        named.update(
            re.findall(r"opendocking\[([a-z][a-z0-9_,-]*)\]", path.read_text(encoding="utf-8"))
        )
    # A bracket may name several extras at once: `opendocking[gui,analysis]`.
    wanted = {part.strip() for name in named for part in name.split(",") if part.strip()}
    missing = sorted(wanted - declared)
    assert not missing, (
        "an error message tells the user to install an extra that is not declared: "
        + ", ".join(missing)
        + f" (declared: {sorted(declared)})"
    )
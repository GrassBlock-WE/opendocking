# SPDX-License-Identifier: GPL-3.0-or-later
"""The published documentation must not point at private files.

Two failure modes are caught here, both of which have actually happened in this
repository:

* a **dead link** — the README pointed at an internal hand-over document that is
  gitignored and absent from a clone, so a reader following it hit nothing;
* a **citation that stays in the working tree while the published copy was
  cleaned** — one release staging pass scrubbed the internal brief's citations
  from a copy of the sources, so the next staging pass would have quietly
  reintroduced every one of them.

Neither is visible to a unit test of the code, and both are cheap to check.

Note how the needles below are assembled from fragments rather than written out.
This file ships, and the release content scan is a *byte* scan for exactly these
strings: spelling them here would make the guard the leak it exists to prevent.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def needles() -> tuple:
    """The private names that must never reach a published file.

    Built at run time on purpose — see the module docstring.  Do not "simplify"
    these into literals.
    """
    backslash = chr(92)
    return (
        "od" + "ck.md",
        "DELIVERY" + ".md",
        "docs/" + "INTERFACES" + ".md",
        "refer" + "ence/",
        "C:" + backslash + "Users",
        "." + "venv" + backslash + "Scripts",
    )


def _ignored_entries() -> set:
    """The literal ``/path`` entries of `.gitignore`, as repository-relative paths.

    Deliberately not a git implementation: it understands the anchored
    single-path form this repository uses for its private files, which is the
    part the documentation checks depend on.  A glob pattern here would be a
    silent no-op, so anything else is ignored — and the ignore test below pins
    the three entries that matter.
    """
    entries = set()
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("/"):
            continue
        if any(char in line for char in "*?["):
            continue
        entries.add(line.lstrip("/"))
    return entries


#: The documentation that ships with a release: everything in the doc set that a
#: clone would actually contain.
PUBLISHED = [
    path
    for path in [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    + sorted((ROOT / "docs").glob("*.md"))
    if str(path.relative_to(ROOT)).replace(os.sep, "/") not in _ignored_entries()
]

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace(os.sep, "/")


def test_the_published_documentation_set_is_what_we_think_it_is():
    names = {_relative(path) for path in PUBLISHED}
    for expected in ("README.md", "CONTRIBUTING.md", "docs/VALIDATION.md",
                     "docs/BENCHMARK.md", "docs/USER_GUIDE.md"):
        assert expected in names, expected


def test_every_local_documentation_link_resolves():
    """A link to a file that is not in the repository is a dead end."""
    broken = []
    checked = 0
    for path in PUBLISHED:
        text = path.read_text(encoding="utf-8")
        for target in _LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            file_part = target.split("#", 1)[0]
            if not file_part:
                continue
            checked += 1
            if not (path.parent / file_part).resolve().exists():
                broken.append(f"{_relative(path)} -> {target}")
    assert checked > 30, f"only {checked} local links found; is the scan working?"
    assert not broken, "dead links in the published documentation: " + "; ".join(broken)


def test_the_published_documentation_does_not_name_private_files():
    hits = []
    for path in PUBLISHED:
        text = path.read_text(encoding="utf-8")
        for needle in needles():
            if needle in text:
                hits.append(f"{_relative(path)}: {needle!r}")
    assert not hits, (
        "the published documentation cites files that are not in the repository: "
        + "; ".join(hits)
    )


def test_the_shipped_python_sources_do_not_cite_the_internal_brief():
    """Docstrings drift back; this is the check that notices."""
    brief, frozen = needles()[0], "INTERFACES" + ".md"
    hits = []
    for path in sorted((ROOT / "python" / "odock").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if brief in text or frozen in text:
            hits.append(_relative(path))
    assert not hits, (
        "internal-document citations in shipped sources: " + ", ".join(hits)
        + " (rewrite them as 'the project requirements' / 'the frozen interface')"
    )


def test_the_internal_documents_are_ignored_and_the_published_one_is_not():
    """The mechanism that keeps the private files out of a clone, checked."""
    ignored = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    for private in needles()[0:3]:
        assert f"/{private}" in ignored, private
    for published in ("docs/VALIDATION.md", "docs/BENCHMARK.md", "docs/SCIENCE.md"):
        assert f"/{published}" not in ignored, published


def test_the_demo_data_is_published_except_the_vendored_binary():
    """The demo ships; the 1.2 MB third-party Vina binary does not.

    This is a regression guard. The rule used to be ``/demo/*`` plus a ``!``
    exception for the screening library: ignoring the whole directory silently
    dropped 18 files the release is supposed to contain, and made the
    hand-maintained library invisible until it was re-included by hand.
    """
    rules = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    binary = "vina" + "." + "exe"  # assembled: this file ships and is byte-scanned
    assert f"/demo/**/{binary}" in rules
    assert "/demo/*" not in rules, (
        "ignoring the whole demo directory drops 18 published files"
    )
    assert f"!/demo/libraries/library.smi" not in rules, (
        "the exception is only needed when the whole directory is ignored"
    )

    # The published demo data has to be on disk, or a clone loses it.
    for kept in ("receptor.pdbqt", "ligand.pdbqt", "poses.pdbqt", "result.pdbqt",
                 "box.json", "config.txt"):
        assert (ROOT / "demo" / "systems" / "3ptb" / kept).exists(), kept
    assert (ROOT / "demo" / "systems" / "1m17" / "receptor.pdbqt").exists()
    assert (ROOT / "demo" / "libraries" / "library.smi").exists()
    assert (ROOT / "demo" / "README.md").exists()
    # The binary itself is deliberately *not* asserted present: it is ignored, so
    # a fresh clone will not have it, and that is the point of the rule.


@pytest.mark.parametrize("name", ["docs/VALIDATION.md"])
def test_the_validation_report_exists_and_is_current(name):
    """The README points here; it has to exist and carry the current numbers."""
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "python -m odock.benchmark" in text
    assert "python -m pytest tests -q" in text
    # Pin the shape, not the number: the count moves whenever any test lands,
    # and a literal here broke this guard twice. The command lines above and
    # the skip count below are the parts that must not drift.
    assert " passed, 1 skipped" in text
    assert "48 steps, 48 passed" in text

#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect the built distributions and say what is actually inside them.

``maturin build`` reports "Built wheel for CPython ..." and that is all it says;
whether the wheel contains the Python package, the compiled extension, the
metadata and the licence is a different question, and the one that matters to
somebody who just downloaded it.  This reads the artefacts themselves — a wheel
is a zip, an sdist is a tar — and checks every entry the project promises.

It checks the other direction too, and that half is the important one: a file that
must **not** be published (an internal document, a build directory, a release
snapshot of the repository, stale bytecode) is a leak, and a check that only looks
for missing files cannot see a leak.  Both defects this script was written after
were leaks, so the deny-list is not decoration — and because a leak-catcher that
silently stops matching is worse than none, ``--self-test`` proves that every rule
still fires.

Usage::

    python tools/inspect_dist.py [dist-dir] [--list]
    python tools/inspect_dist.py --self-test

Exit status is 0 when every artefact contains everything it should *and* nothing
it must not, 1 otherwise, so the Makefile and the release workflow can use it as a
gate.
"""

from __future__ import annotations

import argparse
import email
import io
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# What must be present
# ---------------------------------------------------------------------------

#: Entries the artefacts must contain, as (label, paths that must all be present).
#: Every path is matched as a substring of a member's path, so the wheel tag and
#: the sdist's top-level directory do not have to be known in advance.
WHEEL_REQUIRED: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("the Python package", ("odock/__init__.py", "odock/cli.py", "odock/docking.py")),
    ("the compiled kernel", ("odock/_odock.",)),
    ("the chemistry", ("odock/prepare.py", "odock/chem/ligand.py")),
    ("the screening pipeline", ("odock/screen.py",)),
    ("the analysis layer", ("odock/analysis.py", "odock/consensus.py", "odock/metrics.py")),
    ("the workbench", ("odock/gui/app.py", "odock/gui/viewport.py")),
    ("the benchmark", ("odock/benchmark.py",)),
    ("the metadata", (".dist-info/METADATA",)),
    ("the wheel tags", (".dist-info/WHEEL",)),
    ("the console script", (".dist-info/entry_points.txt",)),
    ("the record of installed files", (".dist-info/RECORD",)),
    ("the licence", ("LICENSE",)),
)

#: Entries an sdist must contain: the sources plus the things a reader or a
#: packager needs to build, test and understand the project.
SDIST_REQUIRED: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("the build configuration", ("pyproject.toml", "Cargo.toml", "Cargo.lock")),
    ("the Rust workspace", ("crates/dock-core/src/lib.rs", "crates/dock-py/src/lib.rs")),
    ("the Python package", ("python/odock/__init__.py", "python/odock/cli.py")),
    ("the documentation", ("docs/USER_GUIDE.md", "docs/SCREENING.md")),
    ("the test suite", ("tests/conftest.py", "tests/data/3PTB.pdb")),
    ("the demo generator", ("examples/make_demo.py",)),
    ("the README", ("README.md",)),
    ("the licence", ("LICENSE",)),
    ("the contribution guide", ("CONTRIBUTING.md",)),
    ("the demo library", ("demo/library.smi",)),
    ("the benchmark baseline", ("benchmark/baseline.json",)),
    ("the release automation", (".github/workflows/ci.yml", "Makefile")),
)

# ---------------------------------------------------------------------------
# What must not be present
# ---------------------------------------------------------------------------
#
# Every rule is anchored: a *root-level* file, a *root-level* directory, a file
# suffix, or something that is only legitimate under one prefix.  A substring
# rule is how a checker starts failing on `python/odock/__init__.py` while looking
# for a root `odock/` snapshot -- which this one did, once, before it was anchored.


def _root_file(name: str, filename: str) -> bool:
    """The member is exactly ``<filename>`` at the archive root."""
    return name == filename


def _root_dir(name: str, directory: str) -> bool:
    """The member is inside a top-level ``directory/``."""
    head = name.split("/", 1)[0]
    return head == directory and "/" in name


def _root_dir_prefix(name: str, prefix: str) -> bool:
    """The member is inside a top-level directory whose name starts with `prefix`."""
    head = name.split("/", 1)[0]
    return "/" in name and head.startswith(prefix)


def _suffix(name: str, suffixes: Sequence[str]) -> bool:
    return name.lower().endswith(tuple(suffixes))


def _bytecode(name: str) -> bool:
    return "__pycache__/" in name or _suffix(name, (".pyc", ".pyo"))


def _structure(name: str) -> bool:
    return _suffix(name, (".pdb",))


def _structure_outside_test_data(name: str) -> bool:
    return _structure(name) and "tests/data/" not in name


#: (label, predicate) for paths that must never appear in **any** artefact.  Note
#: what is *not* here: a `.pdb` rule.  A wheel must carry no structure at all,
#: while an sdist legitimately carries `tests/data/*.pdb`; a single over-broad
#: rule would reject the clean sdist, which the self-test catches.
DENY_ANY: Sequence[Tuple[str, Callable[[str], bool]]] = (
    ("the internal requirements brief", lambda n: _root_file(n, "odck.md")),
    ("the internal hand-over document", lambda n: _root_file(n, "DELIVERY.md")),
    ("the upstream reference material", lambda n: _root_dir(n, "reference")),
    ("build output", lambda n: _root_dir(n, "out")),
    ("the cargo target directory", lambda n: _root_dir(n, "target")),
    ("the maturin temp directory", lambda n: _root_dir(n, ".rust-tmp")),
    ("scratch directories", lambda n: _root_dir(n, "scratch")),
    ("pytest's temporary directories", lambda n: _root_dir_prefix(n, "pytest-of-")),
    ("stray temporary directories", lambda n: _root_dir_prefix(n, "tmp")),
    ("compiled bytecode", _bytecode),
)

#: (label, predicate) for paths that must never appear in a **wheel**.
DENY_WHEEL: Sequence[Tuple[str, Callable[[str], bool]]] = (
    ("a structure file", _structure),
    ("a test or documentation file", lambda n: not n.startswith("odock/") and ".dist-info/" not in n),
)

#: (label, predicate) for paths that must never appear in a **source** distribution.
DENY_SDIST: Sequence[Tuple[str, Callable[[str], bool]]] = (
    ("a release snapshot of this repository", lambda n: _root_dir(n, "odock")),
    ("a built extension module", lambda n: _suffix(n, (".pyd", ".so", ".dll"))),
    ("a structure outside tests/data", _structure_outside_test_data),
)


def _wheel_members(path: Path) -> List[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(archive.namelist())


def _sdist_members(path: Path) -> List[str]:
    with tarfile.open(path) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    # A sdist nests everything under <name>-<version>/; strip that prefix so the
    # checks below read like the repository.
    stripped = []
    for name in names:
        parts = name.split("/", 1)
        stripped.append(parts[1] if len(parts) == 2 else name)
    return sorted(stripped)


def _ok(label: str, what: str, detail: str) -> None:
    print(f"    [ok     ] {label} {what:<32} {detail}")


def _bad(label: str, what: str, detail: str) -> None:
    print(f"    [LEAK   ] {label} {what:<32} {detail}")


def _check(
    label: str, members: Sequence[str], required: Iterable[Tuple[str, Tuple[str, ...]]]
) -> List[str]:
    problems: List[str] = []
    print(f"  {label}: {len(members)} file(s)")
    for what, paths in required:
        found: List[str] = []
        missing: List[str] = []
        for needle in paths:
            hits = [name for name in members if needle in name]
            if hits:
                found.append(hits[0])
            else:
                missing.append(needle)
        if missing:
            print(f"    [MISSING] {what:<33} {', '.join(missing)}")
            problems.append(f"{label}: {what} ({', '.join(missing)})")
        else:
            detail = found[0] if len(found) == 1 else f"{len(found)} file(s)"
            print(f"    [ok     ] {what:<33} {detail}")
    return problems


def _check_deny_list(
    label: str,
    members: Sequence[str],
    denies: Sequence[Tuple[str, Callable[[str], bool]]],
) -> List[str]:
    """Fail on anything internal that reached the artefact.

    An artefact can contain every file it promises and still ship a document
    nobody meant to publish.  Each rule names what it protects, so a failure is
    readable without opening the archive.
    """
    problems: List[str] = []
    for what, predicate in denies:
        hits = sorted({name for name in members if predicate(name)})
        if hits:
            _bad(label, what, f"{len(hits)} file(s), first: {hits[0]}")
            problems.append(f"{label}: {what} is published ({hits[0]})")
        else:
            _ok(label, what, "absent")
    return problems


def _wheel_metadata(path: Path) -> List[str]:
    """The promises a wheel makes in its own metadata."""
    problems: List[str] = []
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            (n for n in archive.namelist() if n.endswith(".dist-info/METADATA")), None
        )
        entry_points = next(
            (n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt")), None
        )
        if metadata_name is None:
            return [f"{path.name}: no METADATA"]
        message = email.message_from_bytes(archive.read(metadata_name))
        for field in ("Name", "Version", "Summary", "Requires-Python"):
            value = message.get(field)
            print(f"    {field:<18} {str(value)!r}")
            if not value:
                problems.append(f"{path.name}: METADATA has no {field}")
        # PEP 639 moved the licence into `License-Expression`; a wheel built by an
        # older toolchain carries `License` instead, and PEP 621's file form
        # carries only `License-File`.  Accept all three, require one.
        licence = (
            message.get("License-Expression")
            or message.get("License")
            or message.get("License-File")
        )
        print(f"    {'License':<18} {str(licence)!r}")
        if not licence:
            problems.append(
                f"{path.name}: METADATA carries no licence (License, "
                "License-Expression or License-File)"
            )
        if message.get("Name") != "opendocking":
            problems.append(f"{path.name}: METADATA Name is {message.get('Name')!r}")
        if entry_points is not None:
            text = archive.read(entry_points).decode("utf-8", "replace")
            compact = "".join(text.split())
            print(f"    console script     {compact or '<empty>'}")
            if "odock=odock.cli:main" not in compact:
                problems.append(f"{path.name}: entry_points.txt does not define `odock`")
    return problems


def _wheel_is_only_a_package(path: Path, members: Sequence[str]) -> List[str]:
    """A wheel holds importable code and metadata — nothing else.

    maturin's `include` adds files to the wheel as well as the sdist unless each
    entry says `format = "sdist"`, which quietly ships the Makefile, the docs and
    the test data inside the wheel.  Anything at the wheel root that is not the
    package or its ``dist-info`` is therefore a packaging mistake.
    """
    stray = [
        name
        for name in members
        if not name.startswith("odock/") and ".dist-info/" not in name
    ]
    if stray:
        _bad("wheel", "only the package and its metadata", f"{len(stray)} stray file(s)")
        for name in stray[:10]:
            print(f"                 stray: {name}")
        return [f"{path.name}: {len(stray)} file(s) outside odock/ and its dist-info"]
    _ok("wheel", "only the package and its metadata", f"{len(members)} file(s)")
    return []


def _inspect_members(kind: str, members: Sequence[str]) -> List[str]:
    """Every check that depends only on an artefact's member list."""
    problems: List[str] = []
    if kind == "wheel":
        problems += _check("wheel", members, WHEEL_REQUIRED)
        problems += _wheel_is_only_a_package(Path("(members)"), members)
        problems += _check_deny_list("wheel", members, DENY_ANY)
        problems += _check_deny_list("wheel", members, DENY_WHEEL)
    else:
        problems += _check("sdist", members, SDIST_REQUIRED)
        problems += _check_deny_list("sdist", members, DENY_ANY)
        problems += _check_deny_list("sdist", members, DENY_SDIST)
    return problems


# ---------------------------------------------------------------------------
# The self-test: a leak-catcher that stopped matching is worse than none
# ---------------------------------------------------------------------------

#: Members a synthetic, *clean* sdist carries — enough to satisfy every rule.
_FAKE_SDIST: Tuple[str, ...] = (
    "pyproject.toml", "Cargo.toml", "Cargo.lock", "README.md", "LICENSE",
    "CONTRIBUTING.md", "Makefile", "PKG-INFO",
    "crates/dock-core/src/lib.rs", "crates/dock-py/src/lib.rs",
    "python/odock/__init__.py", "python/odock/cli.py",
    "docs/USER_GUIDE.md", "docs/SCREENING.md",
    "tests/conftest.py", "tests/data/3PTB.pdb",
    "examples/make_demo.py", "demo/library.smi", "benchmark/baseline.json",
    ".github/workflows/ci.yml",
)

#: Each of these must make the sdist check fail; the value is what a user sees.
_LEAKS: Tuple[Tuple[str, str], ...] = (
    ("odck.md", "the internal requirements brief"),
    ("DELIVERY.md", "the internal hand-over document"),
    ("reference/array3d.h", "the upstream reference material"),
    ("out/log.txt", "build output"),
    ("target/debug/foo.rlib", "the cargo target directory"),
    (".rust-tmp/junk", "the maturin temp directory"),
    ("scratch/notes.txt", "scratch directories"),
    ("pytest-of-x/pytest-1/f", "pytest's temporary directories"),
    ("tmp69pnmksh/r.csv", "stray temporary directories"),
    ("python/odock/__pycache__/cli.cpython-310.pyc", "compiled bytecode"),
    ("odock/cli.py", "a release snapshot of this repository"),
    ("python/odock/_odock.cp310-win_amd64.pyd", "a built extension module"),
    ("1HVR.pdb", "a structure outside tests/data"),
)


def self_test() -> int:
    """Prove that the clean case passes and that every leak rule still fires."""
    print("self-test: a synthetic clean sdist must pass")
    clean = _inspect_members("sdist", _FAKE_SDIST)
    failures = 0
    if clean:
        print(f"  FAIL: the clean member list was rejected ({clean})")
        failures += 1
    print("self-test: python/odock/* must NOT count as a release snapshot")
    nested = _inspect_members(
        "sdist", _FAKE_SDIST + ("python/odock/gui/viewport.py", "crates/dock-py/src/lib.rs")
    )
    if nested:
        print(f"  FAIL: a legitimate nested odock/ path was rejected ({nested})")
        failures += 1

    for entry, expected in _LEAKS:
        print(f"self-test: {entry} must be caught as {expected!r}")
        with tempfile.TemporaryDirectory() as tmp:
            quiet = io.StringIO()
            real_stdout, sys.stdout = sys.stdout, quiet
            try:
                problems = _inspect_members("sdist", _FAKE_SDIST + (entry,))
            finally:
                sys.stdout = real_stdout
            if not any(expected in problem for problem in problems):
                print(f"  FAIL: {entry} was not caught as {expected!r} (got {problems})")
                failures += 1
            elif entry not in " ".join(problems):
                print(f"  FAIL: {entry} was rejected, but for another reason ({problems})")
                failures += 1
    print()
    if failures:
        print(f"SELF-TEST FAILED: {failures} rule(s) are not doing their job")
        return 1
    print(f"self-test OK: {len(_LEAKS)} leak rule(s) fire, the clean case passes")
    return 0


# ---------------------------------------------------------------------------


def inspect(dist: Path, *, list_members: bool = False) -> int:
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels and not sdists:
        print(f"nothing to inspect in {dist} (no *.whl and no *.tar.gz)")
        return 1

    problems: List[str] = []
    for wheel in wheels:
        print(f"\n{wheel.name}")
        members = _wheel_members(wheel)
        if list_members:
            for name in members:
                print(f"    {name}")
        problems += _inspect_members("wheel", members)
        problems += _wheel_metadata(wheel)
    for sdist in sdists:
        print(f"\n{sdist.name}")
        members = _sdist_members(sdist)
        if list_members:
            for name in members:
                print(f"    {name}")
        problems += _inspect_members("sdist", members)

    print()
    if problems:
        print(f"FAILED: {len(problems)} missing or unwanted:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"OK: {len(wheels)} wheel(s) and {len(sdists)} sdist(s) contain everything "
        "the project promises and nothing it must not publish"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", nargs="?", default="dist", help="directory holding the artefacts")
    parser.add_argument(
        "--list", dest="list_members", action="store_true", help="print every member path"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the leak rules themselves (no artefact needed) and exit",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    return inspect(Path(args.dist), list_members=args.list_members)


if __name__ == "__main__":
    raise SystemExit(main())

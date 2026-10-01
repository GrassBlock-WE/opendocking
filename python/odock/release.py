# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock release``: cut a release mechanically, because doing it by hand failed.

Three process mistakes during the 0.2.1 cut, each of which this module now makes
impossible:

===============================  ================================================
what happened                     what the command does now
===============================  ================================================
a hand-written robocopy exclude   ``release stage`` enumerates the published set
list let **42 ``tmp*`` files**     with the repository's own ``.gitignore`` engine
into the snapshot, because         (``out/verify/release_check.py``'s, ported into
``/tmp*/`` matches directories     this package so it always exists), and **refuses
only                               to stage while any stray ``tmp*`` entry is in the
                                   root**, naming them
``Set-Content -Encoding UTF8``     ``release notes`` writes a **BOM-free** UTF-8
wrote a BOM and GitHub silently    JSON body and asserts the first byte is ``{``
rejected the release
a stale commit-message file        ``release publish`` derives the commit message
titled the 0.2.1 commit            from the version being released and **refuses a
"OpenDocking 0.2.0"                message that names a different version**
===============================  ================================================

Two rules run through all of it:

* **refuse, do not guess.** A version that is already different, a changelog
  without the section, a dirty tree, a missing gate report, a message naming
  another version — each stops the command with what was found and what to do.
* **verify, do not assume.** ``release publish`` checks afterwards that the remote
  commit message, the tag target, the release tag and the file count are what was
  asked for, and reports the comparison rather than a success.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import project as _project

__all__ = [
    "RELEASE_FORMAT",
    "RELEASE_VERSION",
    "CheckGate",
    "CheckReport",
    "ChangelogSection",
    "PublishResult",
    "PrepareResult",
    "ReleaseError",
    "StageResult",
    "TransportResult",
    "add_release_parser",
    "check_release",
    "changelog_section",
    "cmd_release_check",
    "cmd_release_notes",
    "cmd_release_prepare",
    "cmd_release_publish",
    "cmd_release_stage",
    "is_ignored",
    "load_gitignore_rules",
    "plan_version_bump",
    "prepare_release",
    "publish_release",
    "published_files",
    "release_notes_body",
    "stage_release",
    "version_locations",
]

PathLike = Union[str, os.PathLike]

RELEASE_FORMAT = "odock-release"
RELEASE_VERSION = 1

#: The default tag prefix.  `v0.2.1`, which is what the repository uses.
TAG_PREFIX = "v"


class ReleaseError(RuntimeError):
    """A release step refused to proceed, with the reason and the fix."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return super().__str__() + (f"\n  fix: {self.fix}" if self.fix else "")


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> Tuple[int, int, int]:
    """``"0.2.1"`` -> ``(0, 2, 1)``; anything else raises."""
    match = _VERSION_RE.match(str(text).strip().lstrip("v"))
    if not match:
        raise ReleaseError(
            f"{text!r} is not a three-part version",
            fix="pass a version such as 0.2.2 (the scheme this project records)",
        )
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def format_version(parts: Sequence[int]) -> str:
    return ".".join(str(int(part)) for part in parts)


@dataclass
class VersionBump:
    """What a requested version means relative to the current one."""

    current: str
    target: str
    kind: str  # "patch", "minor" or "major"
    next_patch: str
    next_minor: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current": self.current,
            "target": self.target,
            "kind": self.kind,
            "next_patch": self.next_patch,
            "next_minor": self.next_minor,
        }


def plan_version_bump(current: str, target: str, *, allow_minor: bool = False) -> VersionBump:
    """Decide whether `target` is a legitimate next version, or refuse.

    The recorded rule: **patch by default**.  A minor bump is a deliberate act and
    needs ``--minor``, so ``0.3.0`` cannot be reached by a typo in a shell history;
    a major bump is not covered by this tool at all (a major release is not a
    mechanical step), and neither is a move sideways or backwards.
    """
    now = parse_version(current)
    wanted = parse_version(target)
    if wanted == now:
        raise ReleaseError(
            f"{target} is the version already in the tree",
            fix="pass the next version, for example "
            f"{format_version((now[0], now[1], now[2] + 1))}",
        )
    if wanted < now:
        raise ReleaseError(
            f"{target} is older than the version in the tree ({current})",
            fix="a release never moves backwards; check the version you meant",
        )
    if wanted[0] != now[0]:
        raise ReleaseError(
            f"{target} changes the major version (from {current})",
            fix=(
                "a major release is not a mechanical step: do it deliberately, and "
                "record the migration in CHANGELOG.md before releasing"
            ),
        )
    next_patch = format_version((now[0], now[1], now[2] + 1))
    next_minor = format_version((now[0], now[1] + 1, 0))
    if wanted[1] != now[1]:
        if not allow_minor:
            raise ReleaseError(
                f"{target} is a minor bump (from {current}), which needs --minor",
                fix=(
                    f"re-run with --minor if that is what you mean; the patch bump "
                    f"is {next_patch}"
                ),
            )
        if wanted[2] != 0:
            raise ReleaseError(
                f"{target} looks like a half-made minor version (a minor bump ends "
                f"in .0, so {next_minor})",
                fix=f"use {next_minor} for the minor release, or {next_patch} for a patch",
            )
        return VersionBump(current, target, "minor", next_patch, next_minor)
    if wanted[2] <= now[2]:
        raise ReleaseError(
            f"{target} does not move forward within {now[0]}.{now[1]}.x "
            f"(the tree is at {current})",
            fix=f"the next patch version is {next_patch}",
        )
    return VersionBump(current, target, "patch", next_patch, next_minor)


# ---------------------------------------------------------------------------
# Where the version lives
# ---------------------------------------------------------------------------


@dataclass
class VersionLocation:
    """One place a version is written, and how to bump it."""

    path: Path
    label: str
    kind: str  # "project" | "workspace" | "inherited" | "dependency"
    version: Optional[str]
    line: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": _project._as_posix(str(self.path)),
            "label": self.label,
            "kind": self.kind,
            "version": self.version,
            "line": self.line,
        }


_SECTION_RE = re.compile(r"^\[([^\]]+)\]\s*$")
_VERSION_LINE_RE = re.compile(r'^(\s*version\s*=\s*)"([^"]+)"\s*$')
_INHERIT_RE = re.compile(r"^\s*version(?:\.workspace)?\s*=\s*(?:\{\s*workspace\s*=\s*true\s*\}|\{\s*workspace\s*=\s*true\s*\}|true)\s*$")
_PATH_DEP_RE = re.compile(
    r'^(\s*)([A-Za-z0-9_-]+)\s*=\s*\{\s*(?:version\s*=\s*"([^"]+)"\s*,\s*)?path\s*=\s*"([^"]+)"(.*)$'
)


def version_locations(root: PathLike) -> List[VersionLocation]:
    """Every place the version lives: the two manifests, the crates, the pins.

    * ``pyproject.toml`` ``[project] version``;
    * ``Cargo.toml`` ``[workspace.package] version``;
    * every workspace member's ``Cargo.toml``: it must **inherit** the version
      (``version.workspace = true``), and a member that hardcodes its own version
      is reported so it can be fixed rather than silently missed;
    * a path dependency on a workspace member that also pins a version
      (``dock-core = { version = "0.1.0", path = "../dock-core" }``): the pin is
      part of "every place it lives", and a stale one is how a release looks
      inconsistent.
    """
    base = Path(root)
    found: List[VersionLocation] = []

    pyproject = base / "pyproject.toml"
    if pyproject.is_file():
        section = ""
        for number, line in enumerate(pyproject.read_text(encoding="utf-8").splitlines(), 1):
            match = _SECTION_RE.match(line)
            if match:
                section = match.group(1)
                continue
            version_line = _VERSION_LINE_RE.match(line)
            if version_line and section == "project":
                found.append(
                    VersionLocation(pyproject, "[project] version", "project",
                                    version_line.group(2), number)
                )

    cargo = base / "Cargo.toml"
    workspace_members: List[str] = []
    if cargo.is_file():
        section = ""
        text = cargo.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            match = _SECTION_RE.match(line)
            if match:
                section = match.group(1)
                continue
            version_line = _VERSION_LINE_RE.match(line)
            if version_line and section == "workspace.package":
                found.append(
                    VersionLocation(cargo, "[workspace.package] version", "workspace",
                                    version_line.group(2), number)
                )
            if section == "workspace":
                members = re.match(r"^\s*members\s*=\s*\[(.*)$", line)
                if members:
                    body = members.group(1)
                    workspace_members.extend(re.findall(r'"([^"]+)"', body))

    member_dirs: List[Path] = []
    for pattern in (base / "crates", base / "packages"):
        if pattern.is_dir():
            member_dirs.extend(sorted(path for path in pattern.iterdir() if path.is_dir()))
    member_manifests = [directory / "Cargo.toml" for directory in member_dirs]
    member_manifests = [path for path in member_manifests if path.is_file()]
    member_names: Dict[str, Path] = {}
    for manifest in member_manifests:
        text = manifest.read_text(encoding="utf-8")
        name = re.search(r'(?m)^name\s*=\s*"([^"]+)"', text)
        if name:
            member_names[name.group(1)] = manifest.parent

    for manifest in member_manifests:
        text = manifest.read_text(encoding="utf-8")
        lines = text.splitlines()
        inherited = any(_INHERIT_RE.match(line) for line in lines)
        own = [
            (number, match.group(2))
            for number, line in enumerate(lines, 1)
            if (match := _VERSION_LINE_RE.match(line))
        ]
        label = f"{_project._as_posix(str(manifest.relative_to(base)))}"
        if inherited:
            found.append(VersionLocation(manifest, f"{label} (inherits)", "inherited", None))
        elif own:
            found.append(
                VersionLocation(manifest, f"{label} (own version)", "project", own[0][1], own[0][0])
            )
        else:
            found.append(VersionLocation(manifest, f"{label} (no version)", "inherited", None))

        for number, line in enumerate(lines, 1):
            match = _PATH_DEP_RE.match(line)
            if not match:
                continue
            dependency, version, relative = match.group(2), match.group(3), match.group(4)
            target = (manifest.parent / relative).resolve()
            if dependency in member_names or any(
                (directory / "Cargo.toml").resolve() == target for directory in member_dirs
            ):
                found.append(
                    VersionLocation(
                        manifest, f"{label}: {dependency} path pin", "dependency",
                        version, number,
                    )
                )
    return found


@dataclass
class PrepareResult:
    """What ``release prepare`` did (or would do)."""

    version: str
    previous: str
    kind: str
    dry_run: bool
    changed: List[VersionLocation] = field(default_factory=list)
    unchanged: List[VersionLocation] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "previous": self.previous,
            "kind": self.kind,
            "dry_run": bool(self.dry_run),
            "changed": [item.as_dict() for item in self.changed],
            "unchanged": [item.as_dict() for item in self.unchanged],
            "notes": list(self.notes),
        }

    def text(self) -> str:
        lines = [
            f"prepare {self.previous} -> {self.version} ({self.kind} bump)"
            + (" [dry run]" if self.dry_run else "")
        ]
        for item in self.changed:
            lines.append(f"  {'would write' if self.dry_run else 'wrote'} {item.label}: {item.version}")
        for item in self.unchanged:
            lines.append(f"  {item.label}: {'inherits the workspace version' if item.kind == 'inherited' else 'already ' + str(item.version)}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def prepare_release(
    root: PathLike,
    target: str,
    *,
    allow_minor: bool = False,
    dry_run: bool = False,
) -> PrepareResult:
    """Bump the version everywhere it lives, refusing an inconsistent tree.

    **Authoritative** locations are the manifests that *state* the version:
    ``pyproject.toml`` ``[project]``, ``Cargo.toml`` ``[workspace.package]``, and a
    crate that carries its own ``version`` instead of inheriting.  They must agree
    before the bump, and a disagreement is refused — a release must not be the
    thing that discovers the version was never consistent.

    A **path pin** (``dock-core = { version = "0.1.0", path = "../dock-core" }``) is
    derived metadata: cargo ignores it for a local build and only publishing
    happens to read it.  It is therefore *normalised* to the released version with
    a note rather than used to refuse the release — and the note is how a stale pin
    (0.1.0 against a 0.2.1 workspace, on this very tree) becomes visible.

    Other refusals: the requested version does not follow the recorded rule (see
    :func:`plan_version_bump`), there is nothing to bump, or no location was found.
    """
    base = Path(root)
    locations = version_locations(base)
    if not locations:
        raise ReleaseError(
            f"no version location was found under {_project.portable_path(base)}",
            fix="run this from the repository root (it looks for pyproject.toml and Cargo.toml)",
        )
    authorities = [
        item for item in locations if item.kind in ("project", "workspace") and item.version
    ]
    current_versions = {item.version for item in authorities}
    if len(current_versions) > 1:
        # Name the file as well as the field: "which manifest?" is the first
        # question a reader has, and the answer must not need a second command.
        detail = ", ".join(
            f"{_relative_to(item.path, base)}: {item.label}={item.version}"
            for item in authorities
        )
        raise ReleaseError(
            "the version locations already disagree: " + detail,
            fix=(
                "make them agree first (a release must not be the thing that "
                "discovers the mismatch); `odock release prepare` will then bump "
                "them together"
            ),
        )
    if not current_versions:
        raise ReleaseError(
            "no manifest carries a version of its own to bump",
            fix="expected a [project] version in pyproject.toml or [workspace.package] in Cargo.toml",
        )
    current = next(iter(current_versions))
    bump = plan_version_bump(current, target, allow_minor=allow_minor)

    result = PrepareResult(
        version=target, previous=current, kind=bump.kind, dry_run=bool(dry_run)
    )
    for item in locations:
        if item.kind in ("inherited",):
            result.unchanged.append(item)
            continue
        if item.kind == "dependency":
            result.notes.append(
                f"{item.label}: the path pin was {item.version}, normalised to the "
                "workspace version so the manifests stay consistent"
            )
        if item.version == target:
            result.unchanged.append(item)
            continue
        _rewrite_version(item, target, dry_run=dry_run)
        result.changed.append(item)
    if not result.changed:
        raise ReleaseError(
            f"nothing to change: every location already says {target}",
            fix="check the version you passed",
        )
    return result


def _relative_to(path: Path, base: Path) -> str:
    """A path as the reader sees it: relative to the tree being released."""
    try:
        return _project._as_posix(str(path.relative_to(base)))
    except ValueError:  # pragma: no cover - a location outside the tree
        return _project._as_posix(str(path))


def _rewrite_version(location: VersionLocation, target: str, *, dry_run: bool) -> None:
    """Replace the version on its recorded line, leaving the rest byte-identical."""
    if location.line is None:  # pragma: no cover - a location always has a line
        raise ReleaseError(f"{location.label} has no recorded line to rewrite")
    text = location.path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    index = location.line - 1
    line = lines[index]
    match = _VERSION_LINE_RE.match(line.rstrip("\r\n"))
    if match:
        newline = line[len(line.rstrip("\r\n")):]
        lines[index] = f'{match.group(1)}"{target}"{newline}'
    elif location.kind == "dependency":
        newline = line[len(line.rstrip("\r\n")):]
        replaced = re.sub(r'version\s*=\s*"[^"]*"', f'version = "{target}"', line.rstrip("\r\n"))
        if replaced == line.rstrip("\r\n"):
            raise ReleaseError(
                f"{location.label}: no version to replace on line {location.line}"
            )
        lines[index] = replaced + newline
    else:  # pragma: no cover - defensive
        raise ReleaseError(f"{location.label}: line {location.line} is not a version line")
    if not dry_run:
        location.path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# The published set: the .gitignore engine from out/verify/release_check.py
# ---------------------------------------------------------------------------

#: This block is a port of ``out/verify/release_check.py``'s engine (same Rule
#: class semantics, same matching rules, same "any ignored ancestor ignores the
#: path" behaviour).  It lives here because ``out/`` is not part of the repository
#: and a release step must not depend on a file that a clone does not have;
#: ``tests/test_release.py`` asserts that this port and the original script agree
#: on the current tree whenever the script is present.


class Rule:
    __slots__ = ("pattern", "anchored", "dir_only", "negated")

    def __init__(self, pattern: str, anchored: bool, dir_only: bool, negated: bool):
        self.pattern = pattern
        self.anchored = anchored
        self.dir_only = dir_only
        self.negated = negated

    def matches(self, rel_dir: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        if self.anchored:
            return _match_segments(self.pattern.split("/"), rel_dir.split("/"))
        return fnmatch.fnmatchcase(rel_dir.split("/")[-1], self.pattern)


def _match_segments(pat: Sequence[str], path: Sequence[str]) -> bool:
    if not pat:
        return not path
    head, rest = pat[0], pat[1:]
    if head == "**":
        return any(_match_segments(rest, path[index:]) for index in range(len(path) + 1))
    if not path:
        return False
    if not fnmatch.fnmatchcase(path[0], head):
        return False
    return _match_segments(rest, path[1:])


def load_gitignore_rules(gitignore: PathLike) -> List[Rule]:
    """The rules of a ``.gitignore``, exactly as ``release_check.py`` reads them."""
    rules: List[Rule] = []
    for raw in Path(gitignore).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        anchored = "/" in line
        if line.startswith("/"):
            line = line[1:]
            anchored = True
        rules.append(Rule(line, anchored, dir_only, negated))
    return rules


def is_ignored(rel: str, rules: Sequence[Rule]) -> bool:
    """Whether a repository-relative path is ignored (any ignored ancestor wins)."""
    parts = rel.split("/")
    ignored = False
    for index in range(len(parts)):
        prefix = "/".join(parts[: index + 1])
        is_dir = index < len(parts) - 1
        for rule in rules:
            if rule.matches(prefix, is_dir):
                ignored = not rule.negated
    return ignored


def _walk(root: PathLike, rules: Optional[Sequence[Rule]] = None) -> List[str]:
    """Every file in the tree, relative and POSIX-style, excluding ``.git``.

    Uses ``os.walk`` with an error callback rather than ``rglob``, for two reasons
    that the end-to-end run made concrete:

    * a directory can **vanish while the walk is reading it** (another process's
      temporary directory, in the run that found this), and ``rglob`` raises
      ``FileNotFoundError`` for the whole walk — the scan then reports "could not
      run" instead of a result;
    * with `rules` given, an **ignored directory is pruned**: git does not descend
      into one, its contents cannot be un-ignored, and skipping it is what makes
      the walk cheap on a tree with a `target/` directory.
    """
    base = Path(root)
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(base, onerror=lambda error: None):
        relative_dir = _project._as_posix(str(Path(dirpath).relative_to(base)))
        prefix = "" if relative_dir in (".", "") else relative_dir + "/"
        if rules is not None:
            # git does not descend into an ignored directory (nothing inside it can
            # be re-included), so pruning is both correct and what makes this cheap.
            dirnames[:] = [
                name for name in dirnames
                if name != ".git" and not is_ignored(prefix + name, rules)
            ]
        else:
            dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in filenames:
            relative = prefix + name
            # ... and a *file* rule still has to be applied: pruning directories
            # alone published 151 `.pyc`/`.log` artefacts in the run that found
            # this, because no directory matched `*.py[cod]`.
            if rules is not None and is_ignored(relative, rules):
                continue
            found.append(relative)
    return sorted(found)


def all_files(root: PathLike) -> List[str]:
    """Every file in the tree, ignoring no rules (for the "would it publish?" check)."""
    return _walk(root)


def published_files(root: PathLike) -> Tuple[List[str], List[str]]:
    """``(published, ignored)`` — what a clone of this tree would contain."""
    base = Path(root)
    rules = load_gitignore_rules(base / ".gitignore")
    published = _walk(base, rules)
    # `ignored` is reported for the stage summary; it is a count first, so the
    # expensive full walk is only paid when a caller actually needs the names.
    ignored = [name for name in _walk(base) if is_ignored(name, rules)]
    return published, ignored


#: Suffixes and directories that are build residue, not release content.
ARTIFACT_SUFFIXES = (".pyd", ".so", ".dll", ".pyc", ".pyo", ".log", ".npy")
ARTIFACT_DIRS = ("__pycache__", "target", "out", ".rust-tmp", ".pytest_cache", ".venv")

#: Files a release cannot be built without, when they exist in the tree: a
#: published set missing one of these is a staging defect, not a small omission.
REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "Cargo.toml",
    "Cargo.lock",
    ".gitignore",
    "crates/dock-core/Cargo.toml",
    "crates/dock-py/Cargo.toml",
    "tests/conftest.py",
    "python/odock/__init__.py",
    "docs/VALIDATION.md",
)


def stray_temp_entries(root: PathLike) -> List[str]:
    """``tmp*`` entries in the tree root — the residue that reached a snapshot once.

    The hand-written exclude list used ``/tmp*/``, which matches directories only,
    so **42 ``tmp*`` files** were staged into the release snapshot.  A release step
    therefore refuses while any ``tmp*`` entry is present and says how to clear
    them.
    """
    base = Path(root)
    return sorted(
        item.name for item in base.iterdir() if item.name.startswith("tmp")
    ) if base.is_dir() else []


#: Root-entry families that are debris from a build or a run, not release content.
#: ``tmp*`` is the original; the rest are the same failure mode with a different
#: name, and the browser-profile entry is the leak that reached the published 0.2.1
#: tree: four ``odock-chrome-*`` directories, each holding a Chrome profile, left by
#: the PDF path when the environment's temp directory was not writable.  The family
#: is listed rather than the names, because the prefix has already changed once.
GENERATED_ROOT_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("tmp", "a temporary directory or file (tempfile fell back to the working directory)"),
    ("odock-chrome-", "a headless-browser profile directory (the PDF path's Chrome profile)"),
    ("odock-report-", "a headless-browser profile directory (the PDF path's Chrome profile)"),
    ("odock-ensemble-", "a run's scratch directory (tempfile fell back to the working directory)"),
    ("chrome_", "a headless browser's own profile directory"),
    ("scoped_dir", "a headless browser's own profile directory"),
    ("Crashpad", "a headless browser's crash-report directory"),
)


def stray_generated_entries(root: PathLike) -> List[Tuple[str, str]]:
    """``(name, family)`` for every root entry that is debris rather than content."""
    base = Path(root)
    if not base.is_dir():
        return []
    found: List[Tuple[str, str]] = []
    for item in sorted(base.iterdir()):
        for prefix, family in GENERATED_ROOT_PREFIXES:
            if item.name.startswith(prefix):
                found.append((item.name, family))
                break
    return found


def _entry_summary(root: Path, names: Sequence[str]) -> List[Tuple[str, int, int]]:
    """``(top-level entry, files, bytes)`` for the staged set, biggest first.

    This is what a human reads before publishing: the guard's rules catch what
    somebody thought of, and a grouped listing of what is *about* to ship catches
    what nobody did — ``odock-chrome-*`` in that list is visible at a glance.
    """
    grouped: Dict[str, List[int]] = {}
    for name in names:
        head = name.split("/", 1)[0]
        try:
            size = (root / name).stat().st_size
        except OSError:  # pragma: no cover - a vanished file
            size = 0
        bucket = grouped.setdefault(head, [0, 0])
        bucket[0] += 1
        bucket[1] += size
    rows = [(head, counts[0], counts[1]) for head, counts in grouped.items()]
    return sorted(rows, key=lambda row: (-row[1], row[0]))



@dataclass
class StageResult:
    """What ``release stage`` wrote."""

    root: Path
    out: Path
    files: int
    bytes: int
    manifest_sha256: str
    ignored: int
    dry_run: bool
    notes: List[str] = field(default_factory=list)
    #: ``(top-level entry, files, bytes)`` — what a human reads before publishing.
    listing: List[Tuple[str, int, int]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "root": _location(self.root),
            "out": _location(self.out),
            "files": int(self.files),
            "bytes": int(self.bytes),
            "manifest_sha256": self.manifest_sha256,
            "ignored": int(self.ignored),
            "dry_run": bool(self.dry_run),
            "notes": list(self.notes),
            "entries": [
                {"entry": name, "files": files, "bytes": size}
                for name, files, size in self.listing
            ],
        }

    def text(self, *, listing: bool = True) -> str:
        lines = [
            f"stage {_project.portable_path(self.root)} -> {_project.portable_path(self.out)}"
            + (" [dry run]" if self.dry_run else ""),
            f"  published files : {self.files} ({self.bytes:,} bytes)",
            f"  ignored by .gitignore: {self.ignored}",
            f"  manifest sha256 : {self.manifest_sha256[:16]}…",
        ]
        for note in self.notes:
            lines.append(f"  note: {note}")
        if listing and self.listing:
            lines.append("")
            lines.append("  WHAT WILL BE PUBLISHED, by top-level entry "
                         "(read this before `release publish`):")
            for name, files, size in self.listing:
                lines.append(f"    {name:<28} {files:>5} file(s)  {size:>12,} bytes")
            lines.append(
                "    — anything here you do not recognise is a leak: fix the tree, "
                "re-stage, and read this list again."
            )
        return "\n".join(lines)


def stage_release(
    root: PathLike,
    out: PathLike,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> StageResult:
    """Build the published set from the working tree, honouring ``.gitignore``.

    Refusals: stray ``tmp*`` entries in the root (42 of them reached a snapshot
    once, because ``/tmp*/`` matches only directories); a published set that is
    empty or missing a required file; generated artefacts inside the published
    set; a destination that already holds files (a stale snapshot contributed
    files to a release once) unless ``--force``.
    """
    base = Path(root).resolve()
    destination = Path(out).resolve()
    if not base.is_dir():
        raise ReleaseError(
            f"{_project.portable_path(base)} is not a directory",
            fix="pass --root pointing at the repository root",
        )

    # Debris in the root, by family.  `tmp*` was the first (42 files reached a
    # snapshot); the browser-profile prefix is the one that reached the **published**
    # 0.2.1 tree, from the PDF path's Chrome profile.  Naming the family is what makes
    # the next one a refusal instead of a leak.
    stray = stray_generated_entries(base)
    if stray:
        families = sorted({family for _, family in stray})
        raise ReleaseError(
            f"{len(stray)} generated entr{'y' if len(stray) == 1 else 'ies'} in the "
            "release root: " + ", ".join(name for name, _ in stray[:8])
            + (" …" if len(stray) > 8 else "")
            + " — " + "; ".join(families),
            fix=(
                "delete them before staging (they come from a temp directory that is "
                "not writable, so `tempfile` fell back to the working directory, or "
                "from a browser profile the cleanup could not remove): remove them, "
                "set TMPDIR/TEMP to a writable directory, and run tests with "
                "`--basetemp .pytest-tmp`"
            ),
        )

    published, ignored = published_files(base)
    if not published:
        raise ReleaseError(
            f"the published set is empty under {_project.portable_path(base)}",
            fix="check .gitignore; `odock doctor` reports a temp directory that is not writable",
        )
    missing = [name for name in REQUIRED_FILES if not (base / name).exists()]
    absent_from_set = [
        name for name in REQUIRED_FILES if (base / name).exists() and name not in published
    ]
    if absent_from_set:
        raise ReleaseError(
            "a required file would not be published: " + ", ".join(absent_from_set),
            fix="fix the .gitignore rule that excludes it (see out/verify/release_check.py)",
        )
    bad = [
        name
        for name in published
        if name.endswith(ARTIFACT_SUFFIXES)
        or any(part in ARTIFACT_DIRS for part in name.split("/"))
    ]
    if bad:
        raise ReleaseError(
            f"{len(bad)} generated artefact(s) are in the published set: "
            + ", ".join(bad[:5]),
            fix="add them to .gitignore; a release ships source, not build residue",
        )
    if destination.exists() and any(destination.iterdir()) and not force:
        raise ReleaseError(
            f"{_project.portable_path(destination)} is not empty",
            fix=(
                "stage into a fresh directory, or pass --force to clear it (a stale "
                "snapshot mixed files into a release once)"
            ),
        )
    if missing:
        notes = [f"not present in this tree (and not required): {', '.join(missing)}"]
    else:
        notes = []

    digest = _manifest_digest(base, published)
    total = sum((base / name).stat().st_size for name in published)
    # What is about to ship, grouped by top-level entry: the guard's rules catch what
    # somebody thought of; a human reading this list catches what nobody did (an
    # `odock-chrome-*` entry is visible here at a glance).
    listing = _entry_summary(base, published)
    if dry_run:
        return StageResult(base, destination, len(published), total, digest,
                           len(ignored), True,
                           notes + ["nothing was written (dry run)"], listing)

    if destination.exists() and force:
        shutil.rmtree(destination)
    for name in published:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / name, target)
    written = sorted(
        _project._as_posix(str(path.relative_to(destination)))
        for path in destination.rglob("*")
        if path.is_file()
    )
    if written != published:
        raise ReleaseError(
            "the staged set does not match the published set: "
            f"{len(written)} on disk vs {len(published)} expected",
            fix="this is a bug in `release stage`; the mismatch is reported rather than hidden",
        )
    return StageResult(base, destination, len(published), total, digest,
                       len(ignored), False, notes, listing)


def _manifest_digest(root: Path, files: Sequence[str]) -> str:
    rows = []
    for name in sorted(files):
        path = root / name
        rows.append(f"{name}\t{path.stat().st_size}\t{_sha256(path)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The changelog and the release body
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^##\s+(?:\[)?v?(\d+\.\d+\.\d+)(?:\])?\s*(?:[—–-]\s*(.*))?$"
)


@dataclass
class ChangelogSection:
    """The section of ``CHANGELOG.md`` for one version."""

    version: str
    date: str
    body: str
    start_line: int
    end_line: int
    headings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "date": self.date,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "lines": len(self.body.splitlines()),
            "headings": list(self.headings),
        }


def changelog_section(changelog: PathLike, version: str) -> ChangelogSection:
    """Extract the changelog section for `version`, refusing an unusable file.

    Refusals: no version sections at all; no section for this version; a section
    that is empty; and a *newer* version heading than the one being released
    (releasing 0.2.1 while the changelog already documents 0.3.0 is a sign the
    version is wrong, not a changelog quirk).
    """
    path = Path(changelog)
    if not path.is_file():
        raise ReleaseError(
            f"{_project.portable_path(path)} does not exist",
            fix="a release needs a changelog section; create the file first",
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    headings: List[Tuple[int, str, str]] = []
    for number, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            headings.append((number, match.group(1), (match.group(2) or "").strip()))
    if not headings:
        raise ReleaseError(
            f"{_project.portable_path(path)} has no '## <version>' heading",
            fix="the changelog is the release body; add a section for the version",
        )
    versions = [item[1] for item in headings]
    if versions.count(version) > 1:
        raise ReleaseError(
            f"CHANGELOG.md has {versions.count(version)} sections for {version}",
            fix="keep one section per version",
        )
    for _, found, _ in headings:
        if parse_version(found) > parse_version(version):
            raise ReleaseError(
                f"CHANGELOG.md already documents {found}, which is newer than the "
                f"version being released ({version})",
                fix="check which version you are releasing, or fix the changelog",
            )
    for index, (number, found, date) in enumerate(headings):
        if found != version:
            continue
        end = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
        body_lines = lines[number + 1:end + 1]
        body = "\n".join(body_lines).strip("\n")
        if not body.strip():
            raise ReleaseError(
                f"the CHANGELOG.md section for {version} is empty",
                fix="write what changed; an empty release body tells a user nothing",
            )
        subheadings = [line for line in body_lines if line.startswith("#")]
        return ChangelogSection(version, date, body, number + 1, end, subheadings)
    raise ReleaseError(
        f"CHANGELOG.md has no section for {version}",
        fix=(
            f"add '## {version} — <date>' with what changed; the sections present are "
            + ", ".join(versions[:5])
        ),
    )


def release_notes_body(
    changelog: PathLike,
    version: str,
    *,
    tag_prefix: str = TAG_PREFIX,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """The GitHub release body as a JSON object (no BOM when written)."""
    section = changelog_section(changelog, version)
    tag = f"{tag_prefix}{version}"
    return {
        "tag_name": tag,
        "name": str(title or f"OpenDocking {version}"),
        "body": section.body,
        "draft": False,
        "prerelease": False,
        "changelog": section.as_dict(),
    }


def write_release_notes(
    changelog: PathLike,
    version: str,
    out: PathLike,
    *,
    tag_prefix: str = TAG_PREFIX,
    title: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Write the JSON body **without a BOM**, and assert the first byte is ``{``.

    ``Set-Content -Encoding UTF8`` wrote a BOM once and GitHub silently rejected
    the release.  The check is in the writer, so the same defect cannot ship
    again — and the test asserts the raw first byte.
    """
    payload = release_notes_body(changelog, version, tag_prefix=tag_prefix, title=title)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    data = text.encode("utf-8")
    if data[:1] != b"{":  # pragma: no cover - a BOM would fail here
        raise ReleaseError("the release body did not start with '{'")
    target.write_bytes(data)
    written = target.read_bytes()
    if written[:1] != b"{" or written[:3] == b"\xef\xbb\xbf":
        raise ReleaseError(
            f"{_project.portable_path(target)} has a byte-order mark",
            fix="write the body with encoding='utf-8'; a BOM makes GitHub reject it",
        )
    return target, payload


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


@dataclass
class TransportResult:
    """One command run by a gate or by publish."""

    argv: List[str]
    code: int
    stdout: str = ""
    stderr: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.code == 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "command": _project.portable_command(self.argv),
            "code": int(self.code),
            "seconds": round(float(self.seconds), 2),
            "output_tail": _tail(self.stdout or self.stderr),
        }


def _tail(text: str, limit: int = 400) -> str:
    return _project.redact_paths(" ".join(str(text).split()))[-limit:]


def run_command(
    argv: Sequence[str], *, timeout: float = 3600.0, cwd: Optional[PathLike] = None,
    stdin: str = "",
) -> TransportResult:
    """Run a command and capture it.  Injectable everywhere else in this module.

    `stdin` exists for the git-data API: a blob body is a file's base64 content, and
    on a command line that hits the argument-length limit (on Windows, at about
    32 kB).  Every API payload therefore travels on stdin, never in argv.
    """
    started = time.perf_counter()
    environment = dict(os.environ)
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout),
            cwd=str(cwd) if cwd is not None else None,
            env=environment,
            input=stdin if stdin else None,
        )
    except subprocess.TimeoutExpired:
        return TransportResult(list(argv), 124, "", f"timed out after {timeout:g} s",
                               time.perf_counter() - started)
    except OSError as exc:  # pragma: no cover - the command is missing
        return TransportResult(list(argv), 125, "", f"{type(exc).__name__}: {exc}",
                               time.perf_counter() - started)
    return TransportResult(
        list(argv), int(completed.returncode), completed.stdout or "",
        completed.stderr or "", time.perf_counter() - started,
    )


@dataclass
class CheckGate:
    """One gate of ``release check``, with its measured result."""

    name: str
    ok: bool
    detail: str
    numbers: Dict[str, Any] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)
    seconds: float = 0.0
    skipped: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ok": bool(self.ok),
            "skipped": bool(self.skipped),
            "detail": self.detail,
            "numbers": self.numbers,
            "command": _project.portable_command(self.command) if self.command else "",
            "seconds": round(float(self.seconds), 2),
        }


@dataclass
class CheckReport:
    """The aggregate answer to "would this release have shipped a defect?"."""

    root: Path
    gates: List[CheckGate] = field(default_factory=list)
    seconds: float = 0.0
    created_utc: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.gates) and all(gate.ok or gate.skipped for gate in self.gates)

    def failures(self) -> List[CheckGate]:
        return [gate for gate in self.gates if not gate.ok and not gate.skipped]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": "odock-release-check",
            "version": RELEASE_VERSION,
            "root": _location(self.root),
            "created_utc": self.created_utc,
            "ok": bool(self.ok),
            "seconds": round(float(self.seconds), 2),
            "gates": [gate.as_dict() for gate in self.gates],
        }

    def text(self) -> str:
        lines = [f"release check: {'PASS' if self.ok else 'FAIL'} ({self.seconds:.1f} s)"]
        for gate in self.gates:
            mark = "skip" if gate.skipped else ("ok" if gate.ok else "FAIL")
            lines.append(f"  [{mark:<4}] {gate.name}: {gate.detail}")
            if gate.numbers:
                lines.append(
                    "         "
                    + ", ".join(f"{key}={value}" for key, value in gate.numbers.items())
                )
        return "\n".join(lines)


#: The gates, in the order they are run: cheapest first, so a broken tree fails
#: fast instead of after the benchmark.
GATE_NAMES: Tuple[str, ...] = (
    "release content",
    "release_check.py",
    "inspect_dist self-test",
    "docs site",
    "doctor --strict",
    "benchmark --check-baseline",
    "pytest suite",
    "test order",
)


def check_release(
    root: PathLike,
    *,
    only: Optional[Sequence[str]] = None,
    quick: bool = False,
    timeout: float = 3600.0,
    runner=None,
) -> CheckReport:
    """Run every gate in one place, each reported with its measured result.

    The gates are the ones this project needs before a cut, and each exists
    because of a real defect: the content rules (a global `.gitignore` rule
    silently dropped 18 demo files once), the extension's own self-test, the leak
    rules, the environment (`odock doctor --strict`), the scoring baseline, and the
    suite.

    `quick=True` replaces the 26-minute benchmark re-docking with the baseline
    *validation* (systems, seeds, tolerances) and the full suite with
    ``-m "not slow"``; that is recorded in the gate detail rather than hidden.
    """
    base = Path(root).resolve()
    started = time.perf_counter()
    # Resolved here, not as a default argument: a default binds the function
    # object at definition time, which is what made an injected transport
    # impossible to swap in from the CLI.
    runner = runner or run_command
    report = CheckReport(root=base, created_utc=_now())
    wanted = {name.lower() for name in only} if only else None

    def selected(name: str) -> bool:
        return wanted is None or name.lower() in wanted

    # 1. the vendored content rules (always available) plus the original script.
    if selected("release content"):
        gate = CheckGate("release content", False, "", command=[])
        gate_started = time.perf_counter()
        try:
            published, ignored = published_files(base)
            problems = _content_problems(base, published)
            gate.ok = not problems
            gate.detail = (
                f"{len(published)} published file(s), {len(ignored)} ignored; "
                + ("clean" if not problems else "; ".join(problems[:3]))
            )
            gate.numbers = {"published": len(published), "ignored": len(ignored),
                            "problems": len(problems)}
        except Exception as exc:  # pragma: no cover - a broken tree
            gate.detail = f"the content rules could not run: {exc}"
        gate.seconds = time.perf_counter() - gate_started
        report.gates.append(gate)

    # 2. `out/verify/release_check.py`, when this tree has it (out/ is not published).
    if selected("release_check.py"):
        script = base / "out" / "verify" / "release_check.py"
        gate = CheckGate("release_check.py", False, "", command=[sys.executable, str(script), str(base)])
        if not script.exists():
            gate.skipped = True
            gate.ok = True
            gate.detail = (
                "skipped: out/verify/release_check.py is not in this tree (out/ is "
                "not published); the vendored content rules above ran instead"
            )
        else:
            result = runner(gate.command, timeout=max(timeout, 600.0))
            gate.ok = result.ok
            gate.seconds = result.seconds
            numbers = _parse_release_check(result.stdout)
            gate.numbers = numbers
            gate.detail = (
                "PASS" if result.ok else f"FAIL (exit {result.code}): {_tail(result.stdout or result.stderr, 200)}"
            )
        report.gates.append(gate)

    # 3. the leak rules' self-test.
    if selected("inspect_dist self-test"):
        tool = base / "tools" / "inspect_dist.py"
        gate = CheckGate("inspect_dist self-test", False, "", command=[sys.executable, str(tool), "--self-test"])
        if not tool.exists():
            gate.skipped = True
            gate.ok = True
            gate.detail = "skipped: tools/inspect_dist.py is not in this tree"
        else:
            result = runner(gate.command, timeout=max(timeout, 300.0))
            gate.ok = result.ok
            gate.seconds = result.seconds
            gate.numbers = {
                "ok_rules": result.stdout.count("[ok"),
                "leaks": result.stdout.count("[LEAK"),
            }
            gate.detail = "the leak rules pass their self-test" if result.ok else (
                f"FAIL (exit {result.code}): {_tail(result.stdout or result.stderr, 200)}"
            )
        report.gates.append(gate)

    # 4. the environment, strictly: a warning here fails the release on purpose.
    if selected("doctor --strict"):
        command = [sys.executable, "-m", "odock.cli", "doctor", "--strict", "--json"]
        gate = CheckGate("doctor --strict", False, "", command=command)
        result = runner(command, timeout=max(timeout, 300.0))
        gate.seconds = result.seconds
        payload = _last_json(result.stdout)
        gate.ok = result.ok
        if payload:
            gate.numbers = dict(payload.get("counts") or {})
            findings = [
                item.get("title", "")
                for item in payload.get("findings", [])
                if item.get("severity") in ("warning", "error")
            ]
            gate.detail = (
                "clean" if result.ok
                else f"{payload.get('counts', {}).get('warning', 0)} warning(s), "
                     f"{payload.get('counts', {}).get('error', 0)} error(s): "
                     + "; ".join(findings[:3])
            )
        else:  # pragma: no cover - an unparseable report
            gate.detail = f"exit {result.code}: {_tail(result.stdout or result.stderr, 200)}"
        report.gates.append(gate)

    # 5. the documentation site: it must build and every link must resolve, so a
    # broken doc link fails a release instead of shipping.
    if selected("docs site") or selected("documentation site"):
        command = [sys.executable, "-m", "odock.cli", "docs", "check", "--json",
                   "-o", str(base / "out" / "docs-site")]
        gate = CheckGate("docs site", False, "", command=command)
        result = runner(command, timeout=max(timeout, 600.0))
        payload = _last_json(result.stdout)
        gate.ok = result.ok
        gate.seconds = result.seconds
        if payload:
            counts = dict(payload.get("counts") or {})
            gate.numbers = counts
            errors = [
                item.get("title", "")
                for item in payload.get("findings", [])
                if item.get("severity") == "error"
            ]
            gate.detail = (
                "the site builds and every link resolves" if result.ok
                else f"{counts.get('error', 0)} error(s): " + "; ".join(errors[:3])
            )
        else:  # pragma: no cover - an unparseable report
            gate.detail = f"exit {result.code}: {_tail(result.stdout or result.stderr, 200)}"
        report.gates.append(gate)

    # 6. the scoring baseline: the real re-docking gate, or its validation.
    if selected("benchmark --check-baseline") or selected("benchmark baseline"):
        if quick:
            from .doctor import gate_benchmark_baseline

            payload = gate_benchmark_baseline(root=base)
            gate = CheckGate(
                "benchmark baseline", bool(payload.get("ok")),
                str(payload.get("detail")), dict(payload.get("numbers") or {}),
                command=[], seconds=float((payload.get("numbers") or {}).get("seconds") or 0.0),
            )
        else:
            command = [sys.executable, "-m", "odock.benchmark", "--check-baseline"]
            gate = CheckGate("benchmark --check-baseline", False, "", command=command)
            result = runner(command, timeout=max(timeout, 5400.0))
            gate.ok = result.ok
            gate.seconds = result.seconds
            gate.numbers = {"exit_code": result.code}
            gate.detail = (
                "every system re-docked within the recorded tolerances"
                if result.ok
                else f"FAIL (exit {result.code}): {_tail(result.stdout or result.stderr, 300)}"
            )
        report.gates.append(gate)

    # 6. the suite, and 7. the same suite in the reverse order.
    if selected("pytest suite"):
        command = [sys.executable, "-m", "pytest", "tests", "-q", "--no-header",
                   "-p", "no:cacheprovider"]
        if quick:
            command += ["-m", "not slow"]
        gate = CheckGate("pytest suite", False, "", command=command)
        result = runner(command, timeout=max(timeout, 3600.0))
        gate.ok = result.ok
        gate.seconds = result.seconds
        gate.numbers = _parse_pytest_summary(result.stdout)
        failing = re.findall(r"^FAILED (\S+)", result.stdout, re.M)
        if failing:
            # Name them all (the first end-to-end run truncated this and left a
            # reader with "14 failed" and no idea which): the failing set is the
            # whole point of the gate.
            gate.numbers["failing"] = failing[:25]
            gate.numbers["failing_total"] = len(failing)
            shown = ", ".join(failing[:6]) + (" …" if len(failing) > 6 else "")
            gate.detail = f"FAIL (exit {result.code}): {len(failing)} test(s) failed — {shown}"
        else:
            gate.detail = (
                "the suite is green" if result.ok
                else f"FAIL (exit {result.code}): {_tail(result.stdout, 300)}"
            )
        report.gates.append(gate)

    # 7. the order/flakiness instrument: the same suite in the reverse order, which
    # is the only gate that can tell *reproducibly red* from *order-dependent* from
    # *flaky*.  `tools/check_test_order.py` (shape-overlay) does the two runs; this
    # gate reads its three verdicts and, at the end, compares it with the suite gate
    # to name the third state explicitly.
    if selected("test order") or selected("test-order"):
        script = base / "tools" / "check_test_order.py"
        command = [sys.executable, str(script), "--markers", "not slow"]
        if quick:
            command += ["--sample", "12"]
        gate = CheckGate("test order", False, "", command=command)
        if not script.exists():
            gate.ok = True
            gate.skipped = True
            gate.detail = "skipped: tools/check_test_order.py is not in this tree"
        else:
            result = runner(command, timeout=max(timeout, 7200.0))
            gate.seconds = result.seconds
            gate.ok = result.ok
            gate.numbers = _parse_order_report(result.stdout)
            gate.numbers["sample"] = 12 if quick else 0
            if result.ok and not gate.numbers.get("failures"):
                gate.detail = (
                    "both orders agree and neither run failed"
                    + (" (a 12-file sample)" if quick else "")
                )
            elif result.ok:
                gate.detail = (
                    f"reproducibly red: {gate.numbers['failures']} test(s) fail in "
                    "both orders, so it is a property of the tests rather than of "
                    "the order — fix them"
                )
            else:
                changed = gate.numbers.get("changed") or []
                gate.detail = (
                    f"ORDER DEPENDENT (candidate): {len(changed)} test(s) changed their "
                    "outcome between the two orders"
                    + (": " + ", ".join(changed[:5]) if changed else "")
                    + " — a second run of the *same* order is needed to rule out load: "
                    "two orders run at different times, so a test whose outcome depends "
                    "on machine state changes with them.  The signal is real when the "
                    "same order repeats identically and the pair disagrees."
                )
        report.gates.append(gate)

    # The third state needs both gates to be read together: an order gate that says
    # "both orders agree" while the suite gate is red means neither "broken" nor
    # "order-dependent" — it means the failure does not follow from the code path,
    # and the thing to hunt is shared state, not an ordering.
    order_gate = next((gate for gate in report.gates if gate.name == "test order"), None)
    suite_gate = next((gate for gate in report.gates if gate.name == "pytest suite"), None)
    if (
        order_gate is not None and suite_gate is not None
        and order_gate.ok and not order_gate.skipped and not suite_gate.ok
        and not order_gate.numbers.get("failures")
    ):
        order_gate.numbers["flaky_signal"] = True
        order_gate.detail += (
            " — FLAKY SIGNAL: the suite gate is red while both orders agree, so the "
            "failure is neither reproducible nor order-dependent; hunt shared state "
            "(a module-level cache, a mutated default, a Qt singleton, an environment "
            "write, an unseeded generator) rather than reordering"
        )

    report.seconds = time.perf_counter() - started
    return report


def _parse_order_report(output: str) -> Dict[str, Any]:
    """Read `tools/check_test_order.py`'s verdict: both orders, failures, changes."""
    def count(label: str) -> Optional[int]:
        match = re.search(rf"{label} order:\s*(\d+)\s*failure", output)
        return int(match.group(1)) if match else None

    forward, reverse = count("forward"), count("reverse")
    changed: List[str] = []
    for match in re.finditer(r"fails only in the (?:forward|reverse) order:\s*(\S+)", output):
        changed.append(match.group(1))
    numbers: Dict[str, Any] = {
        "forward_failures": forward,
        "reverse_failures": reverse,
        "failures": max(item for item in (forward, reverse, 0) if item is not None),
        "changed": changed,
        "order_dependent": "ORDER DEPENDENCE" in output,
    }
    return numbers


def _content_problems(root: Path, published: Sequence[str]) -> List[str]:
    """The content rules, as problems: required files, artefacts, families."""
    problems: List[str] = []
    for name in REQUIRED_FILES:
        if (root / name).exists() and name not in published:
            problems.append(f"{name} exists but would not be published")
    artefacts = [
        name
        for name in published
        if name.endswith(ARTIFACT_SUFFIXES)
        or any(part in ARTIFACT_DIRS for part in name.split("/"))
    ]
    if artefacts:
        problems.append(f"{len(artefacts)} generated artefact(s) would be published")
    for label, pattern in (
        ("the Python package", r"^python/odock/(?:.*/)?[^/]+\.py$"),
        ("the test suite", r"^tests/(?:.*/)?[^/]+\.py$"),
        ("the bundled demo", r"^demo/(?:.*/)?[^/]+$"),
        ("the docs", r"^docs/[^/]+\.md$"),
    ):
        regex = re.compile(pattern)
        on_disk = [name for name in all_files(root) if regex.match(name)]
        excluded = [
            name for name in on_disk
            if name not in published and not name.endswith("vina.exe")
        ]
        if excluded:
            problems.append(f"{label}: {len(excluded)} file(s) would not be published")
    return problems


def _parse_pytest_summary(output: str) -> Dict[str, Any]:
    match = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    skipped = re.search(r"(\d+) skipped", output)
    return {
        "passed": int(match.group(1)) if match else None,
        "failed": int(failed.group(1)) if failed else 0,
        "skipped": int(skipped.group(1)) if skipped else 0,
    }


def _parse_release_check(output: str) -> Dict[str, Any]:
    numbers: Dict[str, Any] = {}
    for label, pattern in (
        ("files_on_disk", r"files on disk\s*:\s*(\d+)"),
        ("published", r"published\s*:\s*(\d+)"),
        ("ignored", r"ignored\s*:\s*(\d+)"),
        ("published_bytes", r"published size:\s*([\d,]+)"),
    ):
        match = re.search(pattern, output)
        if match:
            value = match.group(1).replace(",", "")
            numbers[label] = int(value)
    return numbers


def _last_json(text: str) -> Optional[Dict[str, Any]]:
    """The last JSON object in some output (a CLI may print a table first)."""
    decoder = json.JSONDecoder()
    for start in range(len(text)):
        if text[start] != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


# ---------------------------------------------------------------------------
# Publishing, and verifying that it happened
# ---------------------------------------------------------------------------


@dataclass
class PublishResult:
    """What ``release publish`` did, and what it verified afterwards."""

    version: str
    tag: str
    dry_run: bool
    commit: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    verification: List[Dict[str, Any]] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and all(item.get("ok") for item in self.verification) \
            and all(item.get("ok") for item in self.steps)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": "odock-release-publish",
            "version": self.version,
            "tag": self.tag,
            "dry_run": bool(self.dry_run),
            "commit": self.commit,
            "ok": bool(self.ok),
            "steps": self.steps,
            "verification": self.verification,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }

    def text(self) -> str:
        lines = [
            f"publish {self.tag}" + (" [dry run]" if self.dry_run else "")
            + (f" from commit {self.commit[:12]}" if self.commit else "")
        ]
        for step in self.steps:
            mark = "ok  " if step.get("ok") else "FAIL"
            lines.append(f"  [{mark}] {step.get('name')}: {step.get('detail')}")
        if self.verification:
            lines.append("  verification")
            for item in self.verification:
                mark = "ok  " if item.get("ok") else "FAIL"
                lines.append(
                    f"    [{mark}] {item.get('what')}: expected {item.get('expected')!r}, "
                    f"found {item.get('found')!r}"
                )
        for problem in self.problems:
            lines.append(f"  problem: {problem}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _version_in_text(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b\d+\.\d+\.\d+\b", text)))


#: A full commit id.  Short ids are accepted by git but not by these API calls.
_HEX40 = re.compile(r"^[0-9a-fA-F]{40}$")


def _gh_api(
    runner,
    slug: str,
    endpoint: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    method: Optional[str] = None,
    fields: Sequence[str] = (),
    jq: Optional[str] = None,
    timeout: float,
    cwd: Optional[Path] = None,
) -> TransportResult:
    """One ``gh api`` call; a JSON body goes in **through stdin**, not argv.

    A blob body is a file's base64 content: on a command line that runs into the
    argument-length limit for anything but a small file (and on Windows, much
    sooner), so every payload travels on stdin.
    """
    argv = ["gh", "api", endpoint]
    if method:
        argv += ["-X", method]
    for field in fields:
        argv += ["-f", field]
    if jq:
        argv += ["-q", jq]
    if payload is not None:
        argv += ["--input", "-"]
    try:
        return runner(
            argv, timeout=timeout, cwd=cwd,
            stdin=json.dumps(payload) if payload is not None else "",
        )
    except TypeError:  # pragma: no cover - a runner that takes no stdin
        return runner(argv, timeout=timeout, cwd=cwd)


def _api_object(result: TransportResult, what: str) -> Dict[str, Any]:
    payload = _last_json(result.stdout)
    if not result.ok or payload is None:
        raise ReleaseError(
            f"{what} failed (exit {result.code}): {_tail(result.stderr or result.stdout, 200)}",
            fix="check `gh auth status` and the repository name (--repo owner/name)",
        )
    return payload


def _remote_branch_head(runner, slug, branch, *, timeout, cwd) -> str:
    result = _gh_api(runner, slug, f"repos/{slug}/commits/{branch}", jq=".sha",
                     timeout=timeout, cwd=cwd)
    head = result.stdout.strip()
    if not result.ok or not _HEX40.match(head):
        raise ReleaseError(
            f"could not read the head of {branch} (exit {result.code}): "
            + _tail(result.stderr or result.stdout, 160),
            fix=(
                "an empty repository has no commit to build on; push an initial "
                "commit first, or use --commit with an existing one"
            ),
        )
    return head


def _remote_blob_paths(runner, slug, tree_sha, *, timeout, cwd) -> List[str]:
    result = _gh_api(runner, slug, f"repos/{slug}/git/trees/{tree_sha}?recursive=1",
                     timeout=timeout, cwd=cwd)
    payload = _api_object(result, "reading the base tree")
    return [
        str(item.get("path"))
        for item in payload.get("tree", [])
        if item.get("type") == "blob"
    ]


def _create_blob(runner, slug, data: bytes, *, timeout, cwd) -> str:
    import base64

    result = _gh_api(
        runner, slug, f"repos/{slug}/git/blobs",
        payload={"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"},
        timeout=timeout, cwd=cwd,
    )
    payload = _api_object(result, "creating a blob")
    sha = str(payload.get("sha", ""))
    if not _HEX40.match(sha):
        raise ReleaseError("the blob API returned no sha")
    return sha


def build_commit_via_api(
    runner,
    *,
    slug: str,
    tree_dir: PathLike,
    branch: str,
    message: str,
    timeout: float = 600.0,
    cwd: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[Optional[str], List[Dict[str, Any]], List[str]]:
    """Build a commit from a staged directory entirely through the GitHub API.

    Returns ``(commit_sha, steps, notes)``.  The order matters and is the whole
    safety argument for this mode:

    1. read the branch head (the parent) and its tree;
    2. upload one blob per file in `tree_dir`, and compute the **deletions** — the
       blobs the base tree has and `tree_dir` does not (the tree API with a
       ``base_tree`` only adds and updates, so a deletion has to be asked for);
    3. create the tree on that base;
    4. create the commit;
    5. **read the commit back** and compare its tree and parent with what was
       intended — and only a match lets the caller move the branch ref.

    Step 5 is why the destructive step does not depend on an HTTP 200: a
    successful-looking response that built the wrong object is caught before any
    ref moves.
    """
    stage = Path(tree_dir).resolve()
    if not stage.is_dir():
        raise ReleaseError(
            f"{_project.portable_path(stage)} is not a directory",
            fix="run `odock release stage <dir>` first and pass --tree-dir <dir>",
        )
    local = _walk(stage)
    if not local:
        raise ReleaseError(
            f"{_project.portable_path(stage)} holds no files",
            fix="stage the release first; the commit is built from the staged set",
        )
    steps: List[Dict[str, Any]] = []
    notes: List[str] = []

    head = _remote_branch_head(runner, slug, branch, timeout=timeout, cwd=cwd)
    base_tree_result = _gh_api(runner, slug, f"repos/{slug}/git/commits/{head}",
                               jq=".tree.sha", timeout=timeout, cwd=cwd)
    base_tree = base_tree_result.stdout.strip()
    if not base_tree_result.ok or not _HEX40.match(base_tree):
        raise ReleaseError(
            f"could not read the tree of {head[:12]}",
            fix="check the branch name and `gh auth status`",
        )
    steps.append({"name": "read the remote base", "ok": True,
                  "detail": f"{branch} at {head[:12]}, tree {base_tree[:12]}"})

    if dry_run:
        steps.append({"name": "build the commit (API)", "ok": True,
                      "detail": f"would upload {len(local)} blob(s), create a tree and "
                                f"a commit on {head[:12]}"})
        return None, steps, notes

    remote_paths = _remote_blob_paths(runner, slug, base_tree, timeout=timeout, cwd=cwd)
    deletions = sorted(set(remote_paths) - set(local))
    entries: List[Dict[str, Any]] = []
    for name in local:
        sha = _create_blob(runner, slug, (stage / name).read_bytes(),
                           timeout=timeout, cwd=cwd)
        entries.append({"path": name, "mode": "100644", "type": "blob", "sha": sha})
    for name in deletions:
        entries.append({"path": name, "mode": "100644", "type": "blob", "sha": None})
    steps.append({
        "name": "upload blobs", "ok": True,
        "detail": f"{len(local)} blob(s) created" + (
            f", {len(deletions)} deletion(s) requested" if deletions else ""
        ),
    })
    if deletions:
        notes.append(
            f"{len(deletions)} file(s) in the remote tree are not in the staged set "
            "and are deleted by this commit: " + ", ".join(deletions[:5])
        )

    tree_result = _gh_api(
        runner, slug, f"repos/{slug}/git/trees",
        payload={"base_tree": base_tree, "tree": entries},
        timeout=timeout, cwd=cwd,
    )
    tree_sha = str(_api_object(tree_result, "creating the tree").get("sha", ""))
    if not _HEX40.match(tree_sha):
        raise ReleaseError("the tree API returned no sha")
    steps.append({"name": "create the tree", "ok": True, "detail": tree_sha[:12]})

    commit_result = _gh_api(
        runner, slug, f"repos/{slug}/git/commits",
        payload={"message": message, "tree": tree_sha, "parents": [head]},
        timeout=timeout, cwd=cwd,
    )
    commit_sha = str(_api_object(commit_result, "creating the commit").get("sha", ""))
    if not _HEX40.match(commit_sha):
        raise ReleaseError("the commit API returned no sha")
    steps.append({"name": "create the commit", "ok": True, "detail": commit_sha[:12]})

    # The condition the whole mode rests on: read it back before anything moves.
    read_back = _gh_api(runner, slug, f"repos/{slug}/git/commits/{commit_sha}",
                        timeout=timeout, cwd=cwd)
    payload = _api_object(read_back, "reading the new commit back")
    found_tree = str((payload.get("tree") or {}).get("sha", ""))
    parents = [str(item.get("sha", "")) for item in payload.get("parents", []) or []]
    if found_tree != tree_sha or parents != [head]:
        raise ReleaseError(
            "the commit that was created does not match what was intended: "
            f"tree {found_tree[:12]} (expected {tree_sha[:12]}), "
            f"parents {[item[:12] for item in parents]} (expected {[head[:12]]})",
            fix=(
                "nothing has been moved: the branch ref is only updated after this "
                "comparison; investigate before retrying"
            ),
        )
    steps.append({"name": "verify the new commit", "ok": True,
                  "detail": f"tree {tree_sha[:12]} on parent {head[:12]} — safe to move the ref"})
    return commit_sha, steps, notes


def _tag_exists_remotely(runner, slug, tag, *, timeout, cwd) -> bool:
    result = _gh_api(runner, slug, f"repos/{slug}/git/ref/tags/{tag}",
                     timeout=timeout, cwd=cwd)
    return result.ok and _last_json(result.stdout) is not None


def publish_release(
    root: PathLike,
    version: str,
    *,
    check_report: Optional[Union[PathLike, Mapping[str, Any]]] = None,
    require_check: bool = True,
    message: Optional[str] = None,
    message_file: Optional[PathLike] = None,
    assets: Sequence[PathLike] = (),
    tag_prefix: str = TAG_PREFIX,
    branch: str = "main",
    remote: str = "origin",
    repo: Optional[str] = None,
    commit: Optional[str] = None,
    tree_dir: Optional[PathLike] = None,
    api_commit: bool = False,
    runner=None,
    dry_run: bool = False,
    timeout: float = 600.0,
) -> PublishResult:
    """Publish a release and **verify** it, rather than assuming it worked.

    Three ways to supply the commit, because a release is cut from more than one
    kind of place (this project's own agent workspace is a plain directory with no
    ``.git``, which is why the 0.2.1 commit had to be built through the API):

    * **default** — a git worktree: commit the version bump, tag, then push the
      refs through the API;
    * ``commit=SHA`` — the commit already exists, built by another tool or pushed
      by hand: no worktree is needed, and this command creates the tag ref, the
      release and the verification;
    * ``api_commit=True, tree_dir=DIR`` — build the commit **through the GitHub
      git-data API** from a directory produced by ``release stage``: blobs → tree
      (including deletions) → commit → **read the commit back and compare it with
      what was intended, and only then update the branch ref**, so the destructive
      step depends on a verified object graph rather than on an HTTP 200.

    Preconditions, all refused rather than worked around:

    * ``release check`` passed in this run (pass its ``--json`` report);
    * the tree is clean, or carries **only** the version bump this tool made;
    * a message (from ``--message`` or ``--message-file``) that names a different
      version is refused — a stale message file titled the 0.2.1 commit "0.2.0";
    * the tag must not already exist locally or remotely;
    * a changelog section for the version (the body has to come from somewhere).

    The verification re-reads the remote afterwards: the branch head, the commit
    message, the tag target, the release tag, the asset names and the file count.
    """
    base = Path(root).resolve()
    tag = f"{tag_prefix}{version}"
    result = PublishResult(version=version, tag=tag, dry_run=bool(dry_run))
    # Resolved here rather than as a default argument: a default binds the
    # function object at definition time, so a caller (or a test) could not
    # replace `run_command` to keep the whole path offline.
    runner = runner or run_command

    parse_version(version)
    if commit is not None and not _HEX40.match(str(commit)):
        raise ReleaseError(
            f"--commit {commit!r} is not a 40-character commit id",
            fix="pass the full commit sha (git rev-parse HEAD prints it)",
        )
    if api_commit and tree_dir is None:
        raise ReleaseError(
            "--api-commit needs the directory to build the commit from",
            fix=(
                "run `odock release stage <dir>` first and pass `--tree-dir <dir>`; "
                "the commit is built from exactly what was staged"
            ),
        )

    if require_check:
        payload: Optional[Mapping[str, Any]] = None
        if isinstance(check_report, Mapping):
            payload = check_report
        elif check_report is not None:
            path = Path(check_report)
            if not path.is_file():
                raise ReleaseError(
                    f"no release check report at {_project.portable_path(path)}",
                    fix="run `odock release check --json -o <file>` first",
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
        if payload is None:
            raise ReleaseError(
                "no `release check` report was given",
                fix="run `odock release check --json -o out/release-check.json` first",
            )
        if not payload.get("ok"):
            failures = [
                gate.get("name") for gate in payload.get("gates", []) if not gate.get("ok")
            ]
            raise ReleaseError(
                "`release check` did not pass in this run: " + ", ".join(failures or ["?"]),
                fix="fix what failed and re-run `odock release check`",
            )
        result.notes.append("the release check report says every gate passed")

    supplied_commit = commit is not None or api_commit
    if supplied_commit:
        # No worktree is needed (and none is required): the commit comes from
        # --commit or from the git-data API.  The branch ref is only moved after
        # the object graph has been read back and compared.
        slug = repo or _repo_slug(runner, base, timeout)
        if api_commit:
            commit_message = _commit_message(version, message, message_file)
            result.notes.append(f"commit message: {commit_message!r}")
            new_commit, steps, notes = build_commit_via_api(
                runner, slug=slug, tree_dir=tree_dir, branch=branch,
                message=commit_message, timeout=timeout, cwd=base, dry_run=dry_run,
            )
            result.steps.extend(steps)
            result.notes.extend(notes)
            if dry_run:
                result.steps.append({"name": "dry run", "ok": True,
                                     "detail": "nothing was created, tagged or published"})
                return result
            result.commit = new_commit
        else:
            result.commit = str(commit)
            result.steps.append({
                "name": "commit supplied", "ok": True,
                "detail": f"using {result.commit[:12]} (built outside this worktree)",
            })
            # It must actually exist on the remote, or every later step would
            # fail in a more confusing way.
            probe = _gh_api(runner, slug, f"repos/{slug}/git/commits/{result.commit}",
                            timeout=timeout, cwd=base)
            if not probe.ok:
                raise ReleaseError(
                    f"the commit {result.commit[:12]} does not exist on {slug}",
                    fix="push the commit (or build it through `--api-commit`) before publishing",
                )
            if _tag_exists_remotely(runner, slug, tag, timeout=timeout, cwd=base):
                raise ReleaseError(
                    f"the tag {tag} already exists on {slug}",
                    fix="delete the remote tag or release a different version",
                )
        changed = []

    if supplied_commit:
        commit_message = _commit_message(version, message, message_file)
        result.notes.append(f"commit message: {commit_message!r}")
        changed = []
    else:
        status = runner(["git", "status", "--porcelain"], timeout=timeout, cwd=base)
        if not status.ok:
            # Named `status_text`, not `message`: `message` is the caller's commit
            # message, and shadowing it put the git error into the release commit
            # message (caught by reading a real dry-run plan).
            status_text = _tail(status.stderr or status.stdout)
            if not dry_run:
                raise ReleaseError(
                    f"`git status` failed (exit {status.code}): {status_text}",
                    fix=(
                        "run this inside the repository's git worktree, or supply the "
                        "commit yourself with --commit (or --api-commit --tree-dir)"
                    ),
                )
            # A plan is still useful where there is no worktree (this project's own
            # agent workspace has no `.git`): say what could not be checked instead of
            # refusing to describe the steps.
            result.notes.append(
                f"`git status` could not run (exit {status.code}: {status_text}); the "
                "clean-tree check and the commit are only planned, not verified"
            )
            result.steps.append({
                "name": "clean-tree check",
                "ok": True,
                "detail": "not verifiable here: this is not a git worktree",
            })
            changed = []
        else:
            changed = [line for line in status.stdout.splitlines() if line.strip()]
        if changed:
            # Only the files that *carry the version* may be dirty: anything else is
            # someone's work in progress, and a release must not commit it.
            allowed = {
                _project._as_posix(str(item.path.relative_to(base)))
                for item in version_locations(base)
            }
            unexpected = [
                line for line in changed
                if line[3:].strip() not in allowed
            ]
            if unexpected:
                raise ReleaseError(
                    "the working tree has changes that are not this release's version bump: "
                    + "; ".join(unexpected[:5]),
                    fix="commit or stash them; `release publish` only ever commits the bump it made",
                )
            result.notes.append(
                f"the tree carried only the version bump ({len(changed)} file(s)); "
                "it will be committed with a message derived from the version"
            )
        commit_message = _commit_message(version, message, message_file)
        result.notes.append(f"commit message: {commit_message!r}")

    if dry_run:
        # A plan is worth more than a refusal: validate the changelog section here
        # too (so the third incident is caught by a dry run), and spell out every
        # step that would be taken — without writing, committing or calling out.
        section = changelog_section(base / "CHANGELOG.md", version)
        target = result.commit[:12] if result.commit else "<the new commit>"
        result.steps.append({
            "name": "release body",
            "ok": True,
            "detail": (
                f"CHANGELOG.md section {version} ({len(section.body.splitlines())} line(s)) "
                f"-> {_project.portable_path(_notes_file(base, version))}, no BOM"
            ),
        })
        if changed:
            result.steps.append({
                "name": "git commit", "ok": True,
                "detail": f"would commit {len(changed)} version file(s): {commit_message!r}",
            })
        if not supplied_commit:
            result.steps.append({"name": "git tag", "ok": True,
                                 "detail": f"would create the annotated tag {tag} at {target}"})
        else:
            result.steps.append({
                "name": "git tag", "ok": True,
                "detail": f"would create the tag ref {tag} at {target} through the API",
            })
        result.steps.append({
            "name": f"push {branch} (API)", "ok": True,
            "detail": f"would set refs/heads/{branch} to {target}",
        })
        result.steps.append({
            "name": f"push tag {tag} (API)", "ok": True,
            "detail": f"would create refs/tags/{tag} at {target}",
        })
        result.steps.append({
            "name": "gh release create", "ok": True,
            "detail": f"would create {tag} (OpenDocking {version}) from the notes file",
        })
        result.steps.append({"name": "dry run", "ok": True,
                             "detail": "nothing was committed, tagged or published"})
        return result

    if not supplied_commit:
        if changed:
            add = runner(["git", "add", "--"] + sorted(
                {line[3:].strip() for line in changed}
            ), timeout=timeout, cwd=base)
            result.steps.append({"name": "git add", "ok": add.ok,
                                 "detail": "staged the version files" if add.ok else _tail(add.stderr)})
            commit_step = runner(["git", "commit", "-m", commit_message], timeout=timeout, cwd=base)
            result.steps.append({"name": "git commit", "ok": commit_step.ok,
                                 "detail": commit_message if commit_step.ok
                                 else _tail(commit_step.stderr or commit_step.stdout)})
            if not commit_step.ok:
                result.problems.append("the version bump could not be committed")
                return result

        head = runner(["git", "rev-parse", "HEAD"], timeout=timeout, cwd=base)
        if not head.ok:
            result.problems.append("could not read HEAD")
            return result
        result.commit = head.stdout.strip()

        existing = runner(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                          timeout=timeout, cwd=base)
        if existing.ok:
            raise ReleaseError(
                f"the tag {tag} already exists locally",
                fix="delete it (`git tag -d " + tag + "`) or release a different version",
            )
        tag_step = runner(["git", "tag", "-a", tag, "-m", commit_message], timeout=timeout, cwd=base)
        result.steps.append({"name": "git tag", "ok": tag_step.ok,
                             "detail": tag if tag_step.ok else _tail(tag_step.stderr)})
        if not tag_step.ok:
            result.problems.append(f"could not create the tag {tag}")
            return result

    # The release body comes from the changelog: a version without a section has
    # nothing to publish, so this refuses here as well as in `release notes`.
    notes_target, notes_payload = write_release_notes(
        base / "CHANGELOG.md", version, _notes_file(base, version), tag_prefix=tag_prefix
    )
    result.steps.append({
        "name": "release notes",
        "ok": True,
        "detail": (
            f"{notes_payload['changelog']['lines']} changelog line(s) -> "
            f"{_project.portable_path(notes_target)} "
            f"(first byte {notes_target.read_bytes()[:1].decode('latin-1')!r}, no BOM)"
        ),
    })

    slug = slug if supplied_commit else (repo or _repo_slug(runner, base, timeout))
    branch_step = runner(
        ["gh", "api", f"repos/{slug}/git/refs/heads/{branch}", "-X", "PATCH",
         "-f", f"sha={result.commit}", "-F", "force=false"],
        timeout=timeout, cwd=base,
    )
    result.steps.append({
        "name": f"push {branch} (API)",
        "ok": branch_step.ok,
        "detail": f"refs/heads/{branch} -> {result.commit[:12]}" if branch_step.ok
        else _tail(branch_step.stderr or branch_step.stdout),
    })
    tag_push = runner(
        # The ref name must be the full `refs/tags/...`: `-f ref=v0.2.2` creates
        # nothing, and the verification below would then (correctly) fail.
        ["gh", "api", f"repos/{slug}/git/refs", "-f", f"ref=refs/tags/{tag}", "-f",
         f"sha={result.commit}"],
        timeout=timeout, cwd=base,
    )
    result.steps.append({
        "name": f"push tag {tag} (API)",
        "ok": tag_push.ok,
        "detail": f"refs/tags/{tag} -> {result.commit[:12]}" if tag_push.ok
        else _tail(tag_push.stderr or tag_push.stdout),
    })

    asset_args: List[str] = []
    for asset in assets:
        asset_args += ["--attach", str(asset)]
    release = runner(
        ["gh", "release", "create", tag, "--title", f"OpenDocking {version}",
         "--notes-file", str(notes_target), "--verify-tag", *asset_args],
        timeout=timeout, cwd=base,
    )
    result.steps.append({
        "name": "gh release create", "ok": release.ok,
        "detail": tag if release.ok else _tail(release.stderr or release.stdout),
    })

    result.verification = _verify_publish(
        runner, base, slug, tag, branch, result.commit, assets, timeout
    )
    if not all(item["ok"] for item in result.verification):
        result.problems.append(
            "the remote does not match what was requested; see the verification above"
        )
    return result


def _notes_file(root: Path, version: str) -> Path:
    """Where `release notes` writes the body, for `gh release create --notes-file`."""
    return root / "out" / "release" / f"{version}.json"


def _repo_slug(runner, root: Path, timeout: float) -> str:
    """``owner/name``, from ``gh`` if it can infer one, else from the manifests.

    ``gh repo view`` infers the repository from the *git remote*, so it returns
    nothing in a directory with no worktree — this project's own agent workspace.
    The manifests name the repository (``[project.urls]``, ``package.repository``),
    which is where the fallback reads it.
    """
    result = runner(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                    timeout=timeout, cwd=root)
    slug = result.stdout.strip()
    if slug:
        return slug
    for name in ("pyproject.toml", "Cargo.toml"):
        path = root / name
        if not path.is_file():
            continue
        match = re.search(
            r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
            path.read_text(encoding="utf-8"),
        )
        if match:
            return match.group(1).rstrip(".")
    raise ReleaseError(
        "could not work out the GitHub repository (owner/name)",
        fix=(
            "pass --repo owner/name; `gh repo view` needs a git remote, and this "
            "directory has none"
        ),
    )


def _commit_message(
    version: str, message: Optional[str], message_file: Optional[PathLike]
) -> str:
    """The commit message, with a stale one refused.

    A message file that names a *different* version is refused: reusing the 0.2.0
    file titled the 0.2.1 commit "OpenDocking 0.2.0", and it took a rewrite to
    notice.
    """
    text = message
    if text is None and message_file is not None:
        path = Path(message_file)
        if not path.is_file():
            raise ReleaseError(
                f"no message file at {_project.portable_path(path)}",
                fix="pass --message, or a file that exists",
            )
        text = path.read_text(encoding="utf-8").strip()
    if not text:
        return f"OpenDocking {version}"
    found = [item for item in _version_in_text(text) if item != version]
    if found:
        raise ReleaseError(
            f"the commit message names {', '.join(found)} but you are releasing {version}",
            fix=(
                "write the message for this release (a stale message file is how the "
                "0.2.1 commit was titled 0.2.0)"
            ),
        )
    if version not in text:
        text = f"OpenDocking {version}\n\n{text}"
    return text


def _verify_publish(
    runner,
    root: Path,
    slug: str,
    tag: str,
    branch: str,
    commit: str,
    assets: Sequence[PathLike],
    timeout: float,
) -> List[Dict[str, Any]]:
    """Re-read the remote and compare it with what was requested."""
    checks: List[Dict[str, Any]] = []

    def record(what: str, expected: Any, found: Any) -> None:
        checks.append({"what": what, "expected": expected, "found": found,
                       "ok": found == expected})

    head = runner(["gh", "api", f"repos/{slug}/commits/{branch}", "-q", ".sha"],
                  timeout=timeout, cwd=root)
    record(f"the remote {branch} head", commit, head.stdout.strip() if head.ok else f"<error {head.code}>")

    message = runner(["gh", "api", f"repos/{slug}/commits/{branch}", "-q", ".commit.message"],
                     timeout=timeout, cwd=root)
    expected_message = runner(["git", "log", "-1", "--pretty=%B"], timeout=timeout, cwd=root)
    record("the remote commit message",
           " ".join(expected_message.stdout.split()) if expected_message.ok else "",
           " ".join(message.stdout.split()) if message.ok else f"<error {message.code}>")

    ref = runner(["gh", "api", f"repos/{slug}/git/ref/tags/{tag}", "-q", ".object.sha"],
                 timeout=timeout, cwd=root)
    record(f"the tag {tag} target", commit, ref.stdout.strip() if ref.ok else f"<error {ref.code}>")

    release = runner(["gh", "api", f"repos/{slug}/releases/tags/{tag}"], timeout=timeout, cwd=root)
    payload = _last_json(release.stdout) if release.ok else None
    record(f"the release tag", tag,
           (payload or {}).get("tag_name", f"<error {release.code}>"))
    expected_assets = sorted(Path(item).name for item in assets)
    found_assets = sorted(
        item.get("name", "") for item in (payload or {}).get("assets", []) or []
    )
    record("the release asset names", expected_assets, found_assets)
    record("the release file count", len(expected_assets), len(found_assets))
    return checks


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _location(value: Any) -> str:
    """An absolute location, for a local administrative command (see doctor)."""
    if value is None:
        return "unknown"
    try:
        return str(Path(os.fspath(value)))
    except TypeError:  # pragma: no cover - a non-path value
        return str(value)


def _default_root() -> Path:
    from .doctor import _default_root as doctor_root

    found = doctor_root()
    return found if found is not None else Path.cwd()


def _cli_err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def cmd_release_prepare(args) -> int:
    try:
        result = prepare_release(
            args.root or _default_root(), args.version,
            allow_minor=bool(args.minor), dry_run=bool(args.dry_run),
        )
    except ReleaseError as exc:
        _cli_err(f"odock release prepare: {exc}")
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(result.text())
    return 0


def cmd_release_stage(args) -> int:
    try:
        result = stage_release(
            args.root or _default_root(), args.directory,
            force=bool(args.force), dry_run=bool(args.dry_run),
        )
    except ReleaseError as exc:
        _cli_err(f"odock release stage: {exc}")
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(result.text())
    return 0


def cmd_release_check(args) -> int:
    report = check_release(
        args.root or _default_root(),
        only=args.only or None,
        quick=bool(args.quick),
        timeout=float(args.timeout),
    )
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(report.text())
    return 0 if report.ok else 1


def cmd_release_notes(args) -> int:
    root = Path(args.root) if args.root else _default_root()
    changelog = Path(args.changelog) if args.changelog else root / "CHANGELOG.md"
    out = Path(args.out) if args.out else _notes_file(root, args.version)
    try:
        target, payload = write_release_notes(
            changelog, args.version, out, tag_prefix=args.tag_prefix, title=args.title
        )
    except ReleaseError as exc:
        _cli_err(f"odock release notes: {exc}")
        return 2
    first = target.read_bytes()[:1]
    if args.json:
        print(json.dumps({**payload, "written": _location(target),
                          "first_byte": first.decode("latin-1")}, indent=2, ensure_ascii=False))
    else:
        print(f"wrote {_project.portable_path(target)} ({target.stat().st_size} bytes, "
              f"first byte {first.decode('latin-1')!r}, no BOM)")
        print()
        print(payload["body"])
    return 0


def cmd_release_publish(args) -> int:
    try:
        result = publish_release(
            args.root or _default_root(),
            args.version,
            check_report=args.check_report,
            message=args.message,
            message_file=args.message_file,
            assets=args.asset or (),
            tag_prefix=args.tag_prefix,
            branch=args.branch,
            repo=args.repo,
            commit=args.commit,
            tree_dir=args.tree_dir,
            api_commit=bool(args.api_commit),
            dry_run=bool(args.dry_run),
        )
    except ReleaseError as exc:
        _cli_err(f"odock release publish: {exc}")
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(result.text())
    return 0 if result.ok else 1


def add_release_parser(sub) -> None:
    """Register ``odock release prepare|stage|check|notes|publish`` on `sub`."""
    choices = getattr(sub, "choices", None)
    if isinstance(choices, Mapping) and "release" in choices:
        return
    parser = sub.add_parser(
        "release",
        help="cut a release mechanically (prepare, stage, check, notes, publish)",
        description=(
            "The release steps this project performs by hand, each of which failed "
            "once: bump the version everywhere it lives (patch by default, --minor "
            "for a minor release), build the published set with the repository's own "
            ".gitignore engine and refuse while stray tmp* entries exist, run every "
            "gate in one place, extract the changelog section as a BOM-free JSON "
            "body, and publish while verifying the remote afterwards."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    def add_root(target) -> None:
        target.add_argument("--root", help="the repository root (default: this checkout)")

    prepare = actions.add_parser(
        "prepare",
        help="bump the version in every place it lives",
        description=(
            "Rewrite the version in pyproject.toml [project], Cargo.toml "
            "[workspace.package] and every path pin on a workspace member, after "
            "refusing a tree whose locations already disagree.  A patch bump is the "
            "default; a minor bump needs --minor; a major bump is refused."
        ),
    )
    prepare.add_argument("version", help="the version to release, e.g. 0.2.2")
    prepare.add_argument("--minor", action="store_true", help="this is a minor release")
    prepare.add_argument("--dry-run", action="store_true", help="report what would change")
    add_root(prepare)
    prepare.add_argument("--json", action="store_true", help="machine-readable result")
    prepare.set_defaults(func=cmd_release_prepare)

    stage = actions.add_parser(
        "stage",
        help="build the published set from the working tree",
        description=(
            "Copy exactly what a clone of this tree would contain, decided by the "
            "repository's own .gitignore.  Refuses while any stray tmp* entry sits "
            "in the root, and refuses a destination that already holds files."
        ),
    )
    stage.add_argument("directory", help="where to build the published set")
    stage.add_argument("--force", action="store_true", help="clear the destination first")
    stage.add_argument("--dry-run", action="store_true", help="report without writing")
    add_root(stage)
    stage.add_argument("--json", action="store_true", help="machine-readable result")
    stage.set_defaults(func=cmd_release_stage)

    check = actions.add_parser(
        "check",
        help="run every release gate in one place",
        description=(
            "Run the content rules, out/verify/release_check.py (when present), the "
            "leak-rule self-test, `odock doctor --strict`, the benchmark baseline "
            "check and the test suite, and report each with its measured result.  "
            "Non-zero if any gate fails.  --quick replaces the 26-minute benchmark "
            "re-docking with the baseline validation and skips the slow tests."
        ),
    )
    check.add_argument("--root", help="the repository root (default: this checkout)")
    check.add_argument("--only", action="append", metavar="GATE", help="run one gate (repeatable)")
    check.add_argument("--quick", action="store_true", help="validate instead of re-docking")
    check.add_argument("--timeout", type=float, default=3600.0, help="per-gate timeout (s)")
    check.add_argument("-o", "--out", help="write the report as JSON (for `publish`)")
    check.add_argument("--json", action="store_true", help="print the report as JSON")
    check.set_defaults(func=cmd_release_check)

    notes = actions.add_parser(
        "notes",
        help="extract the changelog section as a release body",
        description=(
            "Write the GitHub release body as JSON from the CHANGELOG.md section for "
            "this version.  Refuses a missing or empty section and a heading newer "
            "than the version, and the file never starts with a byte-order mark."
        ),
    )
    notes.add_argument("version", help="the version to extract")
    notes.add_argument("-o", "--out", help="output .json (default: out/release/<version>.json)")
    notes.add_argument("--changelog", help="the changelog file (default: CHANGELOG.md)")
    notes.add_argument("--title", help="the release title (default: OpenDocking <version>)")
    notes.add_argument("--tag-prefix", default=TAG_PREFIX, help="tag prefix (default: v)")
    add_root(notes)
    notes.add_argument("--json", action="store_true", help="print the body as JSON")
    notes.set_defaults(func=cmd_release_notes)

    publish = actions.add_parser(
        "publish",
        help="push the branch, the tag and the release, then verify the remote",
        description=(
            "Requires a passing `release check` report from this run.  Commits only "
            "the version bump, refuses a tag that exists and a commit message that "
            "names another version, pushes through the GitHub API (this environment "
            "cannot `git push`), and then re-reads the remote to confirm the branch "
            "head, the commit message, the tag target, the release tag and the file "
            "count.  Three ways to supply the commit: from a git worktree (default), "
            "with --commit SHA for one built by another tool, or with --api-commit "
            "--tree-dir DIR to build it through the GitHub git-data API (blobs, tree, "
            "commit, then the commit is read back and compared before any ref moves)."
        ),
    )
    publish.add_argument("version", help="the version being released")
    publish.add_argument(
        "--check-report", help="the JSON written by `release check -o FILE`"
    )
    publish.add_argument("--message", help="the commit message for the version bump")
    publish.add_argument("--message-file", help="a file holding the commit message")
    publish.add_argument(
        "--commit", help="a 40-character commit id that already exists (no worktree needed)"
    )
    publish.add_argument(
        "--api-commit", action="store_true",
        help="build the commit through the GitHub git-data API (needs --tree-dir)",
    )
    publish.add_argument(
        "--tree-dir", help="the staged directory the commit is built from (see `release stage`)"
    )
    publish.add_argument("--asset", action="append", help="a file to attach (repeatable)")
    publish.add_argument("--branch", default="main", help="the branch to update")
    publish.add_argument("--repo", help="owner/name (default: from `gh repo view`)")
    publish.add_argument("--tag-prefix", default=TAG_PREFIX, help="tag prefix (default: v)")
    publish.add_argument("--dry-run", action="store_true", help="plan without changing anything")
    add_root(publish)
    publish.add_argument("--json", action="store_true", help="machine-readable result")
    publish.set_defaults(func=cmd_release_publish)

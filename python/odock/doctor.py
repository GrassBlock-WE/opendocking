# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock doctor``: diagnose this installation, with a fix for every finding.

Every check in here exists because the corresponding failure cost real time in
this project, and none of them announced itself clearly:

===============================  ==================================================
incident                          what the doctor now says
===============================  ==================================================
a wheel whose compiled ``_odock``  the extension version, the distribution version
reported 0.1.0 while the tree      and the checkout's ``pyproject.toml`` are compared
said 0.2.0                         and a mismatch is a warning with the rebuild command
a *different* package shadowing    every ``odock`` package and ``opendocking``
``odock`` on ``sys.path``          distribution ahead of this one is named
``%TEMP%`` was not writable, so    the environment temp directory is probed for
``tempfile`` fell back to the CWD  writability, and the fallback is reported
and littered 165 ``tmp*`` dirs     with the count of stray directories
14 GUI tests failed with           an offscreen GL context is created in a *child*
``no GL context: CreateWindow``    process, so a driver failure is a finding
a headless Qt platform with no     the font database is queried, because a figure
fonts rasterised labels as boxes   whose labels are boxes is worse than none
===============================  ==================================================

Three properties make the output usable rather than a wall of facts:

* **every finding carries a severity, a reason and a fix.**  ``warning`` means
  "something you can act on"; ``error`` means "this cannot work".  The fix is a
  command or a setting, not advice.
* **``--json`` is the machine-readable form**, and ``--strict`` turns any warning
  into a non-zero exit so CI can gate on it.
* **``--self-test`` runs the real gates** — the benchmark baseline, the release
  content rules, and a smoke docking run on the bundled demo — each with its
  measured numbers.

What it cannot do is at the end of ``docs/DOCTOR.md``: it cannot tell you whether
your science is right.  A green doctor means the plumbing works.

The Qt and GL probes run in a **child process**: creating a ``QApplication`` in
the caller's interpreter would change its Qt state (and a missing GL driver can
abort a process rather than raise), so a diagnostic must not risk the thing it is
diagnosing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import project as _project

__all__ = [
    "DOCTOR_FORMAT",
    "DOCTOR_VERSION",
    "Finding",
    "DoctorReport",
    "add_doctor_parser",
    "check_documented_files",
    "check_extras",
    "check_extension_version",
    "check_packages",
    "check_python",
    "check_qt",
    "check_shadowing",
    "check_temp",
    "cmd_doctor",
    "gate_benchmark_baseline",
    "gate_release_content",
    "gate_smoke_dock",
    "run_doctor",
]

PathLike = Union[str, os.PathLike]

DOCTOR_FORMAT = "odock-doctor"
DOCTOR_VERSION = 1

SEVERITIES: Tuple[str, ...] = ("ok", "info", "warning", "error")

#: A package the tool cannot run without, and the extra that provides it.
CORE_MODULES: Tuple[str, ...] = ("numpy",)

#: Optional modules, the extra that installs them, and what they are for.  A
#: missing optional module is reported as information, not as a problem: a
#: headless install without the GUI extra is a legitimate installation.
OPTIONAL_MODULES: Tuple[Tuple[str, str, str], ...] = (
    ("rdkit", "chem", "structure preparation and the chemistry layer"),
    ("PyQt6", "gui", "the workbench and rasterising report figures"),
    ("moderngl", "gui", "the 3-D viewport"),
    ("openpyxl", "", "the Excel report"),
    ("scipy", "", "assignment problems in symmetry-aware RMSD"),
    ("pytest", "dev", "the test suite"),
    ("ruff", "dev", "the lint gate"),
)

#: What each extra buys, for the extras summary.
EXTRAS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("gui", ("PyQt6", "moderngl")),
    ("chem", ("rdkit",)),
    ("dev", ("pytest", "ruff")),
)

#: Files this repository promises.  Missing ones are a warning in a checkout and
#: information in an installed wheel (a wheel carries code, not documentation).
DOCUMENTED_FILES: Tuple[str, ...] = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "Cargo.toml",
    "CHANGELOG.md",
    "docs/VALIDATION.md",
    "docs/PROJECTS.md",
    "docs/STUDIES.md",
    "docs/DOCTOR.md",
    "docs/RELEASE.md",
    "docs/DOCS.md",
    "benchmark/baseline.json",
    "tools/inspect_dist.py",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    "demo/systems/3ptb/receptor.pdbqt",
    "demo/systems/3ptb/ligand.pdbqt",
    "demo/systems/3ptb/poses.pdbqt",
    "demo/systems/3ptb/box.json",
    "tests/data/3PTB.pdb",
)

#: The child program the temp probe runs.  It uses ``open()`` on a deterministic
#: name rather than ``tempfile``, because ``tempfile.NamedTemporaryFile`` *hangs*
#: where a write is blocked rather than refused — measured on this project.
_TEMP_PROBE = r"""
import json, os, tempfile
report = {"candidates": {}, "gettempdir": None, "error": None}
for name in ("TMPDIR", "TEMP", "TMP"):
    path = os.environ.get(name)
    if not path:
        continue
    entry = {"path": path, "is_dir": os.path.isdir(path), "writable": None}
    if entry["is_dir"]:
        probe = os.path.join(path, "odock-doctor-%d.tmp" % os.getpid())
        try:
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.unlink(probe)
            entry["writable"] = True
        except Exception as exc:
            entry["writable"] = False
            entry["error"] = type(exc).__name__
    report["candidates"][name] = entry
try:
    report["gettempdir"] = tempfile.gettempdir()
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
"""


#: Directories that are the same on every machine of a platform: a tool is
#: allowed to look in them (a browser for the optional PDF, a font directory for
#: the rasteriser).  A path *outside* these that exists on the machine running the
#: check is a developer path, and that is what the release-content gate fails on.
_SYSTEM_ROOTS: Tuple[str, ...] = (
    "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "ProgramData", "WINDIR",
    "SystemRoot", "CommonProgramFiles",
)


def _path_exists(token: str) -> bool:
    """Whether the path itself exists, tolerating an unreachable share.

    The **path itself**, deliberately not its parent.  Checking the parent made
    ``C:\\Users\\someone`` — a synthetic path in a test — count as a leak because
    ``C:\\Users`` exists, and a gate that fires on invention is a gate people turn
    off.  The cost is that a leaked path to a file that has since been deleted is
    not flagged; ``docs/DOCTOR.md`` therefore words this check as "an absolute path
    that exists on this machine", which is exactly what the code tests.

    A UNC path in a docstring can name a server that is not there, and ``exists()``
    raises rather than returning False in that case; an unreachable path is not a
    leak on this machine either way.
    """
    try:
        return bool(Path(token).exists())
    except (OSError, ValueError):
        return False


def _is_system_location(token: str) -> bool:
    """Whether a path is a well-known system location rather than somebody's disk.

    Two spellings count: the full path (``C:\\Program Files\\...``) and its
    truncated first component (``C:\\Program``), because the path scanner stops at
    whitespace — and a path that *starts* a system root is that root, however the
    text was cut.
    """
    try:
        candidate = Path(token)
    except (OSError, ValueError):  # pragma: no cover - not a path at all
        return False
    # A POSIX path is not "absolute" to pathlib on Windows, and a system location
    # there (`/usr/share/fonts`, `/opt/...`) still has to be recognised.
    if not (candidate.is_absolute() or str(token).startswith(("/", "\\"))):
        return False
    text = _project._as_posix(str(candidate)).lower().rstrip("/")
    for name in _SYSTEM_ROOTS:
        root = os.environ.get(name)
        if not root:
            continue
        root_text = _project._as_posix(root).lower().rstrip("/")
        if text.startswith(root_text) or root_text.startswith(text):
            return True
    return text.startswith(
        ("/usr", "/opt", "/etc", "/library", "/applications", "/system")
    )


#: The child program the Qt/GL probe runs.  Kept as a string so it can be run with
#: the *current* interpreter, and so a crash (a missing GL driver can abort the
#: process) costs a finding instead of the doctor.
_QT_PROBE = r"""
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
report = {"qt": None, "platform": os.environ.get("QT_QPA_PLATFORM"), "fonts": None,
          "gl_ok": False, "renderer": None, "gl_version": None, "error": None}
try:
    from PyQt6.QtCore import QT_VERSION_STR
    from PyQt6.QtGui import (QGuiApplication, QFontDatabase, QOffscreenSurface,
                             QOpenGLContext, QSurfaceFormat)
    report["qt"] = QT_VERSION_STR
    app = QGuiApplication.instance() or QGuiApplication([])
    try:
        report["fonts"] = len(QFontDatabase.families())
    except Exception as exc:
        report["fonts"] = None
    surface = QOffscreenSurface()
    surface.setFormat(QSurfaceFormat.defaultFormat())
    surface.create()
    context = QOpenGLContext()
    if context.create() and context.makeCurrent(surface):
        functions = context.functions()
        def string(name):
            value = functions.glGetString(name)
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8", "replace")
            return str(value) if value is not None else None
        report["gl_ok"] = True
        report["renderer"] = string(0x1F01)
        report["gl_version"] = string(0x1F02)
        context.doneCurrent()
    else:
        report["error"] = "QOpenGLContext.create()/makeCurrent() failed"
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report))
"""


# ---------------------------------------------------------------------------
# Findings and the report
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """One result of one check: what it is, why it matters, and what to do."""

    key: str
    title: str
    severity: str = "info"
    detail: str = ""
    fix: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")

    @property
    def actionable(self) -> bool:
        return self.severity in ("warning", "error")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "fix": self.fix,
            "data": self.data,
        }


@dataclass
class DoctorReport:
    """What ``odock doctor`` found, and what it measured."""

    findings: List[Finding] = field(default_factory=list)
    gates: List[Dict[str, Any]] = field(default_factory=list)
    generated_utc: str = ""
    seconds: float = 0.0

    def add(self, finding: Finding) -> Finding:
        self.findings.append(finding)
        return finding

    def counts(self) -> Dict[str, int]:
        found = {severity: 0 for severity in SEVERITIES}
        for finding in self.findings:
            found[finding.severity] = found.get(finding.severity, 0) + 1
        return found

    @property
    def errors(self) -> List[Finding]:
        return [item for item in self.findings if item.severity == "error"]

    @property
    def warnings(self) -> List[Finding]:
        return [item for item in self.findings if item.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.warnings

    def exit_code(self, *, strict: bool = False) -> int:
        """0 when healthy; 1 for an error, or for a warning under ``--strict``."""
        if self.errors:
            return 1
        if strict and self.warnings:
            return 1
        return 0

    def as_dict(self, *, strict: bool = False) -> Dict[str, Any]:
        return {
            "format": DOCTOR_FORMAT,
            "version": DOCTOR_VERSION,
            "generated_utc": self.generated_utc,
            "seconds": round(float(self.seconds), 3),
            "ok": bool(self.ok),
            "strict": bool(strict),
            "exit_code": self.exit_code(strict=strict),
            "counts": self.counts(),
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": _project.portable_path(sys.executable),
            },
            "platform": _project._platform_label(),
            "findings": [finding.as_dict() for finding in self.findings],
            "gates": list(self.gates),
        }

    def text(self, *, strict: bool = False, only: Optional[str] = None) -> str:
        """The human report: severities first, then the gates."""
        order = {name: index for index, name in enumerate(SEVERITIES)}
        lines: List[str] = []
        for finding in sorted(
            self.findings, key=lambda item: (-order[item.severity], item.key)
        ):
            if only and finding.severity != only:
                continue
            lines.append(f"[{finding.severity:<7}] {finding.title}")
            if finding.detail:
                lines.append(f"          why : {finding.detail}")
            if finding.fix:
                lines.append(f"          fix : {finding.fix}")
        if self.gates:
            lines.append("")
            lines.append("self-test")
            for gate in self.gates:
                mark = "ok" if gate.get("ok") else "FAILED"
                lines.append(f"  [{mark:<6}] {gate.get('name')}: {gate.get('detail')}")
                numbers = gate.get("numbers") or {}
                if numbers:
                    rendered = ", ".join(f"{key}={value}" for key, value in numbers.items())
                    lines.append(f"           {rendered}")
        counts = self.counts()
        lines.append("")
        lines.append(
            "doctor: "
            + ("OK" if self.ok else "PROBLEMS")
            + f" — {counts.get('error', 0)} error(s), {counts.get('warning', 0)} "
            f"warning(s), {counts.get('info', 0)} note(s)"
            + (f", {self.seconds:.2f} s" if self.seconds else "")
        )
        if self.ok:
            lines.append(
                "        the plumbing works; this says nothing about whether your "
                "science is right (see docs/DOCTOR.md)"
            )
        elif strict:
            lines.append("        --strict: a warning is a non-zero exit here")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Small process and file helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _location(value: Any) -> str:
    """An absolute location, for a *local* diagnostic.

    Everywhere else in this project a recorded path is normalised so it cannot
    leak into a published file.  A doctor is the opposite case: "another
    distribution shadows this one" is useless without *which one*, and the fix is
    a ``pip uninstall`` of a specific installation.  The output is therefore meant
    for the console and for a bug report — never pasted into a shipped file (see
    ``docs/DOCTOR.md``).
    """
    if value is None:
        return "unknown"
    try:
        return str(Path(os.fspath(value)))
    except TypeError:  # pragma: no cover - a non-path value
        return str(value)


def _run_child(
    code: str,
    *,
    timeout: float = 120.0,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> Tuple[int, str, str, float]:
    """Run `code` in a fresh interpreter; returns ``(code, stdout, stderr, secs)``.

    A child process is used wherever a check could abort the interpreter (a
    missing GL driver) or block in a syscall — and *every* child is bounded: a
    probe that never returns is killed and reported as exit 124, because a
    diagnostic that hangs is worse than no diagnostic.  An injected `env` is
    merged over the current environment, not substituted for it: a bare mapping
    would stop the interpreter itself from starting on Windows.
    """
    environment = dict(os.environ)
    if env:
        environment.update({str(key): str(value) for key, value in env.items()})
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=environment,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:  # pragma: no cover - cannot start a child at all
        return 125, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started
    try:
        stdout, stderr = process.communicate(timeout=float(timeout))
        return (
            int(process.returncode or 0),
            stdout or "",
            stderr or "",
            time.perf_counter() - started,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            stdout, stderr = process.communicate(timeout=10.0)
        except Exception:  # pragma: no cover - the pipes never closed
            stdout, stderr = "", ""
        return (
            124,
            stdout or "",
            (stderr or "") + f" [killed after {timeout:g} s]",
            time.perf_counter() - started,
        )


def _tail(text: str, limit: int = 300) -> str:
    """The last `limit` characters of a child's output, with paths redacted."""
    cleaned = _project.redact_paths(" ".join(str(text).split()))
    return cleaned[-limit:]


def _directory_of(package: str) -> Optional[Path]:
    try:
        module = importlib.import_module(package)
    except Exception:
        return None
    path = getattr(module, "__file__", None)
    if not path:
        return None
    return Path(path).resolve().parent


def _source_version(root: Optional[PathLike] = None) -> Optional[str]:
    """The version in the checkout's ``pyproject.toml``, when there is one."""
    base = Path(root) if root is not None else _default_root()
    manifest = base / "pyproject.toml"
    if not manifest.exists():
        return None
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - unreadable checkout
        return None
    match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    return match.group(1) if match else None


def _kernel_manifest_version(root: Optional[PathLike] = None) -> Optional[str]:
    """The version in ``crates/dock-py/Cargo.toml``, for the extension comparison."""
    base = Path(root) if root is not None else _default_root()
    for candidate in ("crates/dock-py/Cargo.toml", "Cargo.toml"):
        path = base / candidate
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable checkout
            continue
        match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
        if match:
            return match.group(1)
    return None


def _default_root() -> Optional[Path]:
    """The checkout this package is being run from, when it is a checkout.

    ``python/odock/doctor.py`` -> the repository root.  When OpenDocking is
    installed as a wheel there is no checkout, and the file checks say so rather
    than inventing one.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "python" / "odock").exists():
            return parent
    return None


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------


def check_python(*, executable: Optional[str] = None) -> Finding:
    """The interpreter itself, so a report says which one produced it."""
    return Finding(
        key="python",
        title=f"Python {platform.python_version()} ({platform.python_implementation()})",
        severity="info",
        detail=(
            "the interpreter running OpenDocking; a bug report without it is hard "
            "to reproduce"
        ),
        data={
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": _location(executable or sys.executable),
            "platform": _project._platform_label(),
            "prefix": _location(sys.prefix),
            "in_virtualenv": bool(sys.prefix != getattr(sys, "base_prefix", sys.prefix)),
        },
    )


def check_extension_version(
    *,
    extension_version: Optional[str] = None,
    package_version: Optional[str] = None,
    source_version: Optional[str] = None,
    kernel_version: Optional[str] = None,
    import_error: str = "",
) -> Finding:
    """The compiled kernel against the package and the checkout.

    This is the check that would have caught a wheel whose local ``_odock`` said
    0.1.0 while the tree said 0.2.0 — the version the user reads comes from the
    extension, not from the package metadata.
    """
    if extension_version is None and not import_error:
        try:
            from . import _odock

            extension_version = str(_odock.__version__)
        except Exception as exc:  # pragma: no cover - depends on the install
            import_error = f"{type(exc).__name__}: {exc}"
    if package_version is None:
        try:
            package_version = str(importlib.metadata.version("opendocking"))
        except Exception:
            package_version = None
    if source_version is None:
        source_version = _source_version()
    if kernel_version is None:
        kernel_version = _kernel_manifest_version()
    data = {
        "extension": extension_version,
        "distribution": package_version,
        "checkout": source_version,
        "kernel_manifest": kernel_version,
    }
    if import_error:
        return Finding(
            key="extension",
            title="the compiled kernel (odock._odock) could not be imported",
            severity="error",
            detail=f"without it nothing can dock or score: {_tail(import_error, 200)}",
            fix="build the extension into this environment: `maturin develop --release`",
            data=data,
        )
    mismatches = []
    if source_version and extension_version and source_version != extension_version:
        mismatches.append(
            f"the extension reports {extension_version} but this checkout says "
            f"{source_version}"
        )
    if package_version and extension_version and package_version != extension_version:
        mismatches.append(
            f"the installed distribution says {package_version} but the extension "
            f"reports {extension_version}"
        )
    if package_version and source_version and package_version != source_version:
        mismatches.append(
            f"the installed distribution says {package_version} but this checkout "
            f"says {source_version}"
        )
    if kernel_version and extension_version and kernel_version != extension_version:
        mismatches.append(
            f"the extension reports {extension_version} but the kernel manifest says "
            f"{kernel_version}"
        )
    if mismatches:
        return Finding(
            key="extension",
            title="the versions disagree: " + "; ".join(mismatches),
            severity="warning",
            detail=(
                "the number a user reads comes from the compiled extension, so a "
                "stale build answers for source it does not contain — the bug you "
                "are looking at may already be fixed in the source"
            ),
            fix=(
                "rebuild the extension: `maturin develop --release` (or "
                "`pip install --force-reinstall` the wheel that matches this tree)"
            ),
            data=data,
        )
    return Finding(
        key="extension",
        title=f"the compiled kernel reports {extension_version}",
        severity="ok",
        detail="the extension, the distribution and the checkout agree",
        data=data,
    )


def check_packages(*, versions: Optional[Mapping[str, Optional[str]]] = None) -> List[Finding]:
    """The core and optional dependencies, each with the extra that provides it.

    With `versions` given, the mapping is used instead of importing anything —
    which is what lets a test pin a missing RDKit without uninstalling it.
    """
    findings: List[Finding] = []
    for name in CORE_MODULES:
        value = versions.get(name) if versions is not None else _module_version(name)
        if value is None:
            findings.append(
                Finding(
                    key=f"package.{name}",
                    title=f"{name} is missing",
                    severity="error",
                    detail="OpenDocking cannot import without it",
                    fix="install the dependency: `pip install numpy`",
                )
            )
        else:
            findings.append(
                Finding(
                    key=f"package.{name}",
                    title=f"{name} {value}",
                    severity="ok",
                    detail="a required dependency",
                    data={"version": value},
                )
            )
    for name, extra, purpose in OPTIONAL_MODULES:
        value = versions.get(name) if versions is not None else _module_version(name)
        where = f"pip install 'opendocking[{extra}]'" if extra else f"pip install {name}"
        if value is None:
            findings.append(
                Finding(
                    key=f"package.{name}",
                    title=f"{name} is not installed",
                    severity="info",
                    detail=f"{purpose} is unavailable without it",
                    fix=where,
                    data={"extra": extra or None},
                )
            )
        else:
            findings.append(
                Finding(
                    key=f"package.{name}",
                    title=f"{name} {value}",
                    severity="ok",
                    detail=purpose,
                    data={"version": value, "extra": extra or None},
                )
            )
    return findings


def check_extras(*, versions: Optional[Mapping[str, Optional[str]]] = None) -> List[Finding]:
    """Whether each optional extra is importable, with the install command."""
    findings: List[Finding] = []
    for extra, modules in EXTRAS:
        missing = [
            name
            for name in modules
            if (versions.get(name) if versions is not None else _module_version(name)) is None
        ]
        if missing:
            findings.append(
                Finding(
                    key=f"extra.{extra}",
                    title=f"the [{extra}] extra is incomplete (missing {', '.join(missing)})",
                    severity="info",
                    detail="the features that need it will refuse with an ImportError",
                    fix=f"pip install 'opendocking[{extra}]'",
                    data={"missing": missing},
                )
            )
        else:
            findings.append(
                Finding(
                    key=f"extra.{extra}",
                    title=f"the [{extra}] extra is complete",
                    severity="ok",
                    data={"modules": list(modules)},
                )
            )
    return findings


def _module_version(name: str) -> Optional[str]:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return str(importlib.metadata.version(name))
    except Exception:
        return "installed (no version attribute)"


def check_shadowing(
    *,
    paths: Optional[Sequence[PathLike]] = None,
    package_dir: Optional[PathLike] = None,
    distribution_dir: Optional[PathLike] = None,
    executable_path: Optional[str] = None,
) -> List[Finding]:
    """Another ``odock`` package or distribution ahead of this one.

    A different package with the same import name shadowing this one is silent:
    the import succeeds and every result comes from code the user did not install.
    """
    findings: List[Finding] = []
    current_package = Path(package_dir).resolve() if package_dir else _directory_of("odock")
    current_distribution = (
        Path(distribution_dir).resolve()
        if distribution_dir
        else _distribution_dir("opendocking")
    )
    searched = [Path(os.fspath(entry or ".")).resolve() for entry in (paths if paths is not None else sys.path)]

    foreign_packages: List[Path] = []
    foreign_distributions: List[Path] = []
    ours_distributions: List[Path] = []
    for entry in searched:
        if not entry.is_dir():
            continue
        candidate = entry / "odock"
        marker = candidate / "__init__.py"
        if marker.exists() or (candidate / "_odock.pyd").exists() or (
            candidate / "_odock.so"
        ).exists():
            if current_package is None or candidate.resolve() != current_package:
                foreign_packages.append(candidate)
        for dist_info in sorted(entry.glob("*.dist-info")) + sorted(entry.glob("*.egg-info")):
            name = _distribution_name(dist_info)
            if name == "opendocking":
                ours_distributions.append(dist_info)
                if current_distribution is None or dist_info.resolve() != current_distribution:
                    foreign_distributions.append(dist_info)
            elif name in ("odock", "opendocking-core", "opendocking-gui"):
                foreign_distributions.append(dist_info)

    def describe(dist_info: Path) -> Dict[str, Any]:
        """A distribution as ``{name, version, path}``, so a finding is actionable."""
        metadata = _distribution_metadata(dist_info)
        return {
            "name": metadata.get("name") or _distribution_name(dist_info),
            "version": metadata.get("version") or _version_from_name(dist_info),
            "path": _location(dist_info),
            "directory": _location(dist_info.parent),
        }

    def describe_package(package: Path) -> Dict[str, Any]:
        """A foreign package as ``{path, version, distribution}``."""
        version = _package_version(package)
        located = _distribution_of_path(package.parent)
        return {
            "path": _location(package),
            "version": version,
            "distribution": describe(located) if located else None,
        }

    if foreign_packages:
        described = [describe_package(item) for item in foreign_packages]
        first = described[0]
        where = first["distribution"]["path"] if first["distribution"] else first["path"]
        version = first["version"] or (
            first["distribution"]["version"] if first["distribution"] else None
        )
        findings.append(
            Finding(
                key="shadowing.package",
                title=(
                    "another 'odock' package is on sys.path: "
                    f"{first['path']}"
                    + (f" (version {version})" if version else "")
                ),
                severity="warning",
                detail=(
                    "importing odock may load that package instead of this one, and "
                    "the module you are running would not be the module you installed: "
                    f"it is provided by {where}"
                ),
                fix=(
                    "uninstall the distribution that provides it "
                    f"(`pip uninstall {first['distribution']['name'] if first['distribution'] else '<name>'}`), "
                    "or remove its directory from PYTHONPATH/sys.path; then check with "
                    "`python -c \"import odock; print(odock.__file__)\"`"
                ),
                data={
                    "foreign": described,
                    "this_package": (
                        _location(current_package) if current_package else None
                    ),
                },
            )
        )
    else:
        findings.append(
            Finding(
                key="shadowing.package",
                title="no other 'odock' package shadows this one",
                severity="ok",
                detail=(
                    "this package: "
                    + (_location(current_package) if current_package else "unknown")
                ),
            )
        )

    if foreign_distributions:
        described = [describe(item) for item in foreign_distributions]
        first = described[0]
        findings.append(
            Finding(
                key="shadowing.distribution",
                title=(
                    f"a second odock distribution is installed: {first['name']} "
                    f"{first['version']} at {first['path']}"
                ),
                severity="warning",
                detail=(
                    "a distribution installed somewhere else can put its own 'odock' "
                    "package on sys.path and its own console script on PATH, so the "
                    "code that runs is not the code you installed here"
                ),
                fix=(
                    f"uninstall the one you did not mean to install "
                    f"(`pip uninstall {first['name']}`), then re-check with "
                    "`odock doctor`"
                ),
                data={
                    "foreign": described,
                    "this_distribution": (
                        _location(current_distribution)
                        if current_distribution
                        else None
                    ),
                },
            )
        )
    else:
        findings.append(
            Finding(
                key="shadowing.distribution",
                title="only one odock distribution is installed",
                severity="ok",
                detail=(
                    _location(current_distribution)
                    if current_distribution
                    else "this package is being run from a source tree"
                ),
            )
        )

    scripts = _console_scripts(executable_path)
    if scripts:
        first = scripts[0][1]
        expected = Path(sys.executable).parent
        if scripts[0][0] != expected.resolve():
            findings.append(
                Finding(
                    key="shadowing.script",
                    title=f"an 'odock' script on PATH belongs to another installation: {_location(first)}",
                    severity="warning",
                    detail=(
                        "typing `odock` runs that script, not this installation, so "
                        "the version you see is not the version you installed here"
                    ),
                    fix=(
                        "put this environment's Scripts/bin directory first on PATH, "
                        "or run the module directly with "
                        f"`{_location(sys.executable)} -m odock.cli`"
                    ),
                    data={"script": _location(first)},
                )
            )
        else:
            findings.append(
                Finding(
                    key="shadowing.script",
                    title="the 'odock' script on PATH is this installation's",
                    severity="ok",
                    data={"script": _location(first)},
                )
            )
    return findings


def check_temp(
    *,
    environ: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
    probe: bool = True,
    timeout: float = 20.0,
    probe_report: Optional[Mapping[str, Any]] = None,
    probe_error: str = "",
) -> Finding:
    """Whether the temp directory is writable — and where ``tempfile`` really goes.

    The failure mode is quiet: when ``TMPDIR``/``TEMP``/``TMP`` is not writable,
    ``tempfile`` falls back to the working directory, and every temporary file
    (including every test's ``tmp_path``) lands in the repository.  On this
    project that put **165 directories and 48 files** named ``tmp*`` in the
    repository root, and a release snapshot swallowed 42 of them.

    The writability probe runs in a **bounded child process**, because creating a
    temporary file is not guaranteed to fail: under a policy that blocks writes
    rather than denying them, ``tempfile.NamedTemporaryFile`` never returns.  That
    hang is itself reported, as a finding with a fix — a diagnostic that hangs
    would be worse than the problem.

    `environ`, `cwd`, `probe_report` and `probe_error` exist so a test can pin a
    situation without reproducing it on the machine running the tests.
    """
    environment = dict(os.environ if environ is None else environ)
    working = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    names = [
        name for name in ("TMPDIR", "TEMP", "TMP") if environment.get(name)
    ]
    if probe_report is None and not probe_error and probe and names:
        probe_report, probe_error = _probe_temp(environment, timeout=timeout)
    probe_report = dict(probe_report or {})
    entries = dict(probe_report.get("candidates") or {})

    unresolvable: List[str] = []
    unwritable: List[str] = []
    chosen: Optional[Path] = None
    for name in names:
        raw = str(environment.get(name) or "")
        entry = dict(entries.get(name) or {})
        resolved = Path(raw)
        usable = entry.get("is_dir") if entry else resolved.is_dir()
        if not usable:
            unresolvable.append(f"{name} ({_location(resolved)})")
            continue
        writable = entry.get("writable") if entry else None
        if probe and writable is False:
            unwritable.append(f"{name} ({_location(resolved)})")
            continue
        chosen = resolved
        break
    fallback = probe_report.get("gettempdir")
    fallback_path = Path(fallback) if fallback else None
    if chosen is None and probe_error:
        fallback_path = None
    stray_dirs = stray_files = 0
    if working.is_dir():
        for item in working.iterdir():
            if not item.name.startswith("tmp"):
                continue
            if item.is_dir():
                stray_dirs += 1
            else:
                stray_files += 1
    data = {
        "environment": {name: environment.get(name) for name in ("TMPDIR", "TEMP", "TMP")},
        "candidates": entries,
        "unresolvable": unresolvable,
        "unwritable": unwritable,
        "resolved": _location(chosen) if chosen else None,
        "tempfile_gettempdir": (
            _location(fallback_path) if fallback_path else fallback
        ),
        "probe_error": probe_error,
        "cwd": _location(working),
        "stray_tmp_directories_in_cwd": stray_dirs,
        "stray_tmp_files_in_cwd": stray_files,
    }
    stray_note = (
        f"{stray_dirs} stray tmp* director(ies) and {stray_files} tmp* file(s) are "
        "in the working directory now"
    )
    if probe_error:
        return Finding(
            key="temp",
            title="the temp-directory probe did not finish: writing to the environment temp directory blocks",
            severity="warning",
            detail=(
                f"{probe_error}.  A blocked write is worse than a refused one: "
                "`tempfile` (and anything waiting on it) hangs rather than raising. "
                f"{stray_note}"
            ),
            fix=(
                "point TMPDIR (POSIX) or TEMP (Windows) at a directory this process "
                "may actually write to, and check the security policy that blocks "
                "file creation there"
            ),
            data=data,
        )
    if chosen is None and names:
        return Finding(
            key="temp",
            title=(
                "no environment temp directory is usable ("
                + ", ".join(unwritable + unresolvable)
                + ")"
            ),
            severity="warning",
            detail=(
                "Python's tempfile falls back to the working directory, so "
                f"temporary files are written next to your source: {stray_note}"
            ),
            fix=(
                "point the environment at a writable directory — on POSIX "
                "`export TMPDIR=/tmp`, on Windows `set TEMP=%LOCALAPPDATA%\\Temp` — "
                "and for pytest use `--basetemp .pytest-tmp`"
            ),
            data=data,
        )
    if chosen is None:
        return Finding(
            key="temp",
            title=(
                "TMPDIR, TEMP and TMP are all unset, so tempfile resolves to "
                f"{_location(fallback_path) if fallback_path else 'the working directory'}"
            ),
            severity="warning",
            detail=(
                "Python then writes its temporary files wherever it falls back to; "
                f"{stray_note}"
            ),
            fix="set TMPDIR (POSIX) or TEMP (Windows) to a writable directory",
            data=data,
        )
    if unwritable or unresolvable:
        return Finding(
            key="temp",
            title=(
                f"the temp directory resolves to {_location(chosen)}, "
                f"not to {', '.join(unwritable + unresolvable)}"
            ),
            severity="warning",
            detail=(
                "the earlier candidate(s) were not usable, so Python silently fell "
                f"back; {stray_note}"
            ),
            fix=(
                "make the environment variable point at a writable directory, or "
                "unset it so the fallback is explicit"
            ),
            data=data,
        )
    return Finding(
        key="temp",
        title=f"the temp directory is writable: {_location(chosen)}",
        severity="ok",
        detail="temporary files go where they should",
        data=data,
    )


def _probe_temp(
    environ: Mapping[str, str], *, timeout: float = 20.0
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Probe the temp candidates in a bounded child; ``(report, error)``.

    The child uses ``open()``, not ``tempfile``: ``NamedTemporaryFile`` never
    returned in the environment that motivated this check (a policy blocked the
    creation instead of denying it), and a probe must be able to say "this blocks"
    rather than block itself.
    """
    returncode, stdout, stderr, _ = _run_child(
        _TEMP_PROBE, timeout=timeout, env=environ
    )
    if returncode == 124:
        return None, f"the writability probe was killed after {timeout:g} s"
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), ""
            except ValueError:
                continue
    return None, (
        f"the probe exited {returncode} without a result"
        + (f": {_tail(stderr, 160)}" if stderr.strip() else "")
    )


def check_qt(
    *,
    platform_name: str = "offscreen",
    run: bool = True,
    timeout: float = 120.0,
    qt_version: Optional[str] = None,
) -> Finding:
    """Can an offscreen Qt platform create a GL context, and does it have fonts?

    Run in a child process: a missing GL driver can abort the interpreter, and a
    ``QApplication`` created here would change the caller's Qt state.
    """
    if not run:
        return Finding(
            key="qt.gl",
            title="the Qt/GL probe was skipped (--no-qt)",
            severity="info",
            detail="the viewport and figure checks were not run",
        )
    if _module_version("PyQt6") is None:
        return Finding(
            key="qt.gl",
            title="PyQt6 is not installed, so there is no Qt platform to probe",
            severity="info",
            detail="the workbench and rasterised report figures need it",
            fix="pip install 'opendocking[gui]'",
        )
    code = _QT_PROBE
    if platform_name != "offscreen":
        code = code.replace('"offscreen"', json.dumps(platform_name))
    returncode, stdout, stderr, seconds = _run_child(code, timeout=timeout)
    payload: Dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                break
            except ValueError:
                continue
    data = {
        "qt": payload.get("qt") or qt_version,
        "platform": payload.get("platform") or platform_name,
        "font_families": payload.get("fonts"),
        "gl_ok": bool(payload.get("gl_ok")),
        "renderer": payload.get("renderer"),
        "gl_version": payload.get("gl_version"),
        "child_exit": returncode,
        "seconds": round(seconds, 2),
    }
    if not payload:
        return Finding(
            key="qt.gl",
            title=f"the Qt/GL probe could not run (child exit {returncode})",
            severity="warning",
            detail=(
                "the child process died before reporting, which is what a missing "
                "or broken graphics driver does: " + (_tail(stderr) or "no output")
            ),
            fix=(
                "install a working GL driver, or force software rendering "
                "(`QT_OPENGL=software`, Mesa llvmpipe) and retry"
            ),
            data=data,
        )
    if not payload.get("gl_ok"):
        return Finding(
            key="qt.gl",
            title=(
                f"no offscreen GL context could be created with "
                f"QT_QPA_PLATFORM={data['platform']}"
            ),
            severity="warning",
            detail=(
                "the workbench cannot render and the viewport tests fail with "
                "'no GL context' — the cause is the driver, not the tree"
                f"{': ' + str(payload.get('error')) if payload.get('error') else ''}"
            ),
            fix=(
                "run on a machine with a GPU driver, or force software rendering "
                "(`QT_OPENGL=software`) before running the GUI tests"
            ),
            data=data,
        )
    fonts = payload.get("fonts")
    if fonts == 0:
        return Finding(
            key="qt.gl",
            title=(
                "an offscreen GL context works, but the Qt platform has no fonts"
            ),
            severity="warning",
            detail=(
                "rasterised figures lose their labels (the rasteriser skips them "
                "rather than embedding boxes)"
            ),
            fix=(
                "set QT_QPA_FONTDIR to the system font directory (on Windows that "
                "is %WINDIR%\\Fonts, on Linux usually /usr/share/fonts)"
            ),
            data=data,
        )
    return Finding(
        key="qt.gl",
        title=(
            f"an offscreen GL context works (Qt {data.get('qt')}, "
            f"{fonts} font families)"
        ),
        severity="ok",
        detail=f"renderer: {data.get('renderer')} GL {data.get('gl_version')}",
        data=data,
    )


def check_documented_files(*, root: Optional[PathLike] = None) -> Finding:
    """The files the documentation promises are actually there.

    A missing demo file is the difference between "the quick start works" and
    "the quick start fails on the first command", and it is invisible until
    someone follows the README.
    """
    base = Path(root).resolve() if root is not None else _default_root()
    if base is None:
        return Finding(
            key="files",
            title="no source checkout next to this package (installed wheel)",
            severity="info",
            detail=(
                "the documentation, the demo data and the release tooling live in "
                "the source distribution, not in an installed wheel"
            ),
            fix="install the source distribution if you need them: `pip download --no-binary :all: opendocking`",
            data={"checked": 0, "missing": []},
        )
    missing = [name for name in DOCUMENTED_FILES if not (base / name).exists()]
    demo_missing = [name for name in missing if name.startswith("demo/")]
    data = {
        "root": _location(base),
        "checked": len(DOCUMENTED_FILES),
        "missing": missing,
    }
    if not missing:
        return Finding(
            key="files",
            title=f"all {len(DOCUMENTED_FILES)} promised files are present",
            severity="ok",
            detail=f"checked in {_location(base)}",
            data=data,
        )
    fix = "run `make demo` to regenerate demo/ (it is generated, not committed)"
    if not demo_missing:
        fix = "you are in a partial checkout: fetch it again, or ask for the sdist"
    return Finding(
        key="files",
        title=f"{len(missing)} promised file(s) are missing",
        severity="warning",
        detail="the documentation and the quick start refer to them: "
        + ", ".join(missing[:6])
        + (" …" if len(missing) > 6 else ""),
        fix=fix,
        data=data,
    )


def _distribution_name(dist_info: Path) -> str:
    """The distribution name of a ``*.dist-info`` directory (METADATA or the path)."""
    metadata = _distribution_metadata(dist_info)
    if metadata.get("name"):
        return str(metadata["name"]).lower().replace("-", "_")
    stem = dist_info.name
    for suffix in (".dist-info", ".egg-info"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.split("-")[0].lower().replace("-", "_")


def _version_from_name(dist_info: Path) -> Optional[str]:
    """The version embedded in a ``name-version.dist-info`` directory name."""
    stem = dist_info.name
    for suffix in (".dist-info", ".egg-info"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    parts = stem.split("-")
    return parts[1] if len(parts) > 1 else None


def _distribution_metadata(dist_info: Path) -> Dict[str, Optional[str]]:
    """``Name`` and ``Version`` from a ``METADATA``/``PKG-INFO`` file."""
    found: Dict[str, Optional[str]] = {}
    for name in ("METADATA", "PKG-INFO"):
        path = dist_info / name
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                lowered = line.lower()
                if lowered.startswith("name:") and "name" not in found:
                    found["name"] = line.split(":", 1)[1].strip()
                elif lowered.startswith("version:") and "version" not in found:
                    found["version"] = line.split(":", 1)[1].strip()
        except OSError:  # pragma: no cover - unreadable metadata
            continue
    return found


def _package_version(package: Path) -> Optional[str]:
    """The ``__version__`` a foreign ``odock`` package declares, if it is readable."""
    for name in ("__init__.py", "_version.py"):
        path = package / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - unreadable file
            continue
        match = re.search(r'(?m)^\s*__version__\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
    return None


def _distribution_of_path(directory: Path) -> Optional[Path]:
    """A ``*.dist-info`` in `directory`, when there is exactly one candidate."""
    if not directory.is_dir():
        return None
    found = sorted(directory.glob("*.dist-info")) + sorted(directory.glob("*.egg-info"))
    return found[0] if found else None


def _distribution_dir(name: str) -> Optional[Path]:
    try:
        distribution = importlib.metadata.distribution(name)
    except Exception:
        return None
    path = getattr(distribution, "_path", None)
    return Path(path).resolve() if path is not None else None


def _console_scripts(executable_path: Optional[str] = None) -> List[Tuple[Path, Path]]:
    """``(directory, script)`` for every ``odock`` executable on PATH."""
    found: List[Tuple[Path, Path]] = []
    search = executable_path if executable_path is not None else os.environ.get("PATH", "")
    for directory in str(search).split(os.pathsep):
        if not directory:
            continue
        for name in ("odock.exe", "odock", "odock-script.py", "odock.cmd", "odock.bat"):
            candidate = Path(directory) / name
            if candidate.exists():
                found.append((Path(directory).resolve(), candidate.resolve()))
                break
    return found


# ---------------------------------------------------------------------------
# The self-test gates
# ---------------------------------------------------------------------------


def gate_benchmark_baseline(
    *,
    root: Optional[PathLike] = None,
    full: bool = False,
    timeout: float = 3600.0,
) -> Dict[str, Any]:
    """The benchmark regression gate.

    The cheap form validates the recorded baseline — its systems, seeds, the
    tolerances and the recorded command — and reports those numbers; the full
    form re-docks the five systems with ``--check-baseline``, which takes tens of
    minutes and is therefore opt-in.
    """
    started = time.perf_counter()
    base = Path(root).resolve() if root is not None else _default_root()
    gate: Dict[str, Any] = {"name": "benchmark baseline", "ok": False, "detail": "", "numbers": {}}
    if base is None:
        gate["detail"] = "no checkout next to this package, so there is no baseline"
        gate["numbers"] = {"seconds": round(time.perf_counter() - started, 2)}
        return gate
    baseline_path = base / "benchmark" / "baseline.json"
    if not baseline_path.exists():
        gate["detail"] = f"the baseline file is missing ({_location(baseline_path)})"
        gate["numbers"] = {"seconds": round(time.perf_counter() - started, 2)}
        return gate
    try:
        document = json.loads(baseline_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        gate["detail"] = f"the baseline is not valid JSON: {exc}"
        gate["numbers"] = {"seconds": round(time.perf_counter() - started, 2)}
        return gate
    systems = document.get("systems") or {}
    tolerance = document.get("tolerance") or {}
    seeds = 0
    if systems:
        first = next(iter(systems.values()))
        seeds = len(first.get("seeds") or [])
    command = str(document.get("command") or "")
    leaked = _project.find_absolute_paths(command)
    gate["numbers"] = {
        "systems": len(systems),
        "seeds_per_system": seeds,
        "tolerance_top_rmsd": tolerance.get("top_rmsd"),
        "tolerance_best_within_1kcal": tolerance.get("best_within_1kcal"),
        "recorded_command": command or "not recorded",
        "absolute_paths_in_the_command": len(leaked),
        "seconds": round(time.perf_counter() - started, 2),
    }
    if full:
        returncode, stdout, stderr, seconds = _run_gate_command(
            [
                sys.executable,
                "-c",
                "import sys; from odock.benchmark import main; "
                "sys.exit(main(['--check-baseline', '--baseline', sys.argv[1]]))",
                str(baseline_path),
            ],
            timeout=timeout,
            cwd=base,
        )
        gate["ok"] = returncode == 0
        gate["detail"] = (
            "`--check-baseline` re-docked every system and agreed with the baseline"
            if returncode == 0
            else f"the full baseline check failed (exit {returncode}): {_tail(stderr or stdout)}"
        )
        gate["numbers"]["seconds"] = round(seconds, 2)
        gate["numbers"]["exit_code"] = returncode
        return gate
    problems: List[str] = []
    if not systems:
        problems.append("no system is recorded")
    if not tolerance:
        problems.append("no tolerance is recorded, so the gate cannot fire")
    if leaked:
        problems.append("the recorded command contains an absolute path")
    gate["ok"] = not problems
    gate["detail"] = (
        f"the recorded baseline is intact ({len(systems)} system(s) × {seeds} seed(s)); "
        "run `python -m odock.benchmark --check-baseline` for the full gate"
        if not problems
        else "; ".join(problems)
    )
    return gate


def _run_gate_command(
    command: Sequence[str], *, timeout: float, cwd: Optional[PathLike] = None
) -> Tuple[int, str, str, float]:
    started = time.perf_counter()
    environment = dict(os.environ)
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            timeout=float(timeout),
            env=environment,
            cwd=str(cwd) if cwd is not None else None,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:g} s", time.perf_counter() - started
    except OSError as exc:  # pragma: no cover - the interpreter is missing
        return 125, "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started
    return (
        int(completed.returncode),
        completed.stdout.decode("utf-8", "replace"),
        completed.stderr.decode("utf-8", "replace"),
        time.perf_counter() - started,
    )


def gate_release_content(
    *, root: Optional[PathLike] = None, timeout: float = 300.0
) -> Dict[str, Any]:
    """The release content rules: the leak self-test, and a scan of the shipped text.

    Two halves, because they catch different things: ``tools/inspect_dist.py
    --self-test`` proves the rules still *work*, and the scan looks for an actual
    developer path in the published text.

    The scan separates a path that **exists on this machine** from one that is an
    illustrative example in a docstring (``benchmark.py`` shows
    ``C:\\work\\repo\\.venv\\Scripts\\python.exe`` to explain what
    :func:`odock.benchmark.portable_command` rewrites).  Only the first kind is a
    leak — it names a real installation on somebody's disk — and only it fails the
    gate, so the check stays meaningful instead of drowning in documentation.
    """
    started = time.perf_counter()
    base = Path(root).resolve() if root is not None else _default_root()
    gate: Dict[str, Any] = {"name": "release content", "ok": False, "detail": "", "numbers": {}}
    if base is None:
        gate["detail"] = "no checkout next to this package, so the release rules cannot be run"
        gate["numbers"] = {"seconds": round(time.perf_counter() - started, 2)}
        return gate
    tool = base / "tools" / "inspect_dist.py"
    rules_ok = False
    rules_note = "the leak rules tool is missing"
    if tool.exists():
        returncode, stdout, stderr, _ = _run_gate_command(
            [sys.executable, str(tool), "--self-test"], timeout=timeout, cwd=base
        )
        rules_ok = returncode == 0
        rules_note = (
            "the leak rules pass their self-test"
            if rules_ok
            else f"the leak-rule self-test failed (exit {returncode}): {_tail(stderr or stdout)}"
        )
    scanned = 0
    existing: List[str] = []
    illustrative: List[str] = []
    seen: set = set()
    for pattern in ("docs/*.md", "README.md", "CONTRIBUTING.md", "python/odock/**/*.py",
                    "tests/*.py", "tools/*.py"):
        for path in sorted(base.glob(pattern)):
            if not path.is_file() or path in seen:  # `**` overlaps the deeper globs
                continue
            seen.add(path)
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            scanned += 1
            relative = _project._as_posix(str(path.relative_to(base)))
            for token in _project.find_absolute_paths(text):
                if _is_system_location(token):
                    continue
                if _path_exists(token):
                    existing.append(f"{relative}: {token}")
                else:
                    illustrative.append(relative)
    gate["ok"] = rules_ok and not existing
    gate["detail"] = (
        f"{rules_note}; {scanned} published file(s) scanned; "
        f"{len(existing)} absolute path(s) that exist on this machine"
        + (
            f" — first: {existing[0]}"
            if existing
            else f"; {len(illustrative)} illustrative path example(s) in docstrings "
            "(none of them exists here)"
        )
    )
    gate["numbers"] = {
        "rules_self_test": rules_ok,
        "files_scanned": scanned,
        "absolute_paths_found": len(existing),
        "illustrative_examples": len(illustrative),
        "seconds": round(time.perf_counter() - started, 2),
    }
    return gate


def gate_smoke_dock(
    *,
    root: Optional[PathLike] = None,
    exhaustiveness: int = 2,
    seed: int = 42,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """A real docking run on the bundled demo: does this installation actually work?

    Deliberately small (two Monte-Carlo runs) so the doctor stays a diagnostic
    rather than a benchmark.
    """
    started = time.perf_counter()
    base = Path(root).resolve() if root is not None else _default_root()
    gate: Dict[str, Any] = {"name": "smoke docking run", "ok": False, "detail": "", "numbers": {}}
    if base is None:
        gate["detail"] = "no checkout next to this package, so there is no bundled demo"
        return gate
    demo = base / "demo" / "systems" / "3ptb"
    needed = ("receptor.pdbqt", "ligand.pdbqt", "box.json")
    missing = [name for name in needed if not (demo / name).exists()]
    if missing:
        gate["detail"] = (
            "the bundled demo is incomplete (missing " + ", ".join(missing) + ")"
        )
        gate["numbers"] = {"seconds": round(time.perf_counter() - started, 2)}
        return gate
    try:
        import odock

        box = _project._coerce_box(demo / "box.json")
        result = odock.dock(
            str(demo / "receptor.pdbqt"),
            str(demo / "ligand.pdbqt"),
            box,
            exhaustiveness=int(exhaustiveness),
            seed=int(seed),
            num_poses=3,
        )
    except Exception as exc:
        gate["detail"] = f"the smoke run failed: {type(exc).__name__}: {_tail(str(exc), 200)}"
        gate["numbers"] = {"seconds": round(time.perf_counter() - started, 2)}
        return gate
    best = result.best_affinity
    poses = len(result.poses)
    ok = poses > 0 and best is not None and best == best
    gate["ok"] = bool(ok)
    gate["detail"] = (
        f"docked the bundled 3PTB demo: {poses} pose(s), best {best:.3f} kcal/mol"
        if ok
        else "the smoke run produced no scoreable pose"
    )
    gate["numbers"] = {
        "poses": poses,
        "best_affinity": None if best is None else round(float(best), 3),
        "exhaustiveness": int(exhaustiveness),
        "seed": int(seed),
        "grid_points": int(result.grid_points),
        "movable_atoms": int(result.num_movable_atoms),
        "seconds": round(time.perf_counter() - started, 2),
    }
    return gate


# ---------------------------------------------------------------------------
# The doctor itself
# ---------------------------------------------------------------------------


def run_doctor(
    *,
    root: Optional[PathLike] = None,
    include_qt: bool = True,
    self_test: bool = False,
    full: bool = False,
    timeout: float = 120.0,
    environ: Optional[Mapping[str, str]] = None,
) -> DoctorReport:
    """Run every check and (optionally) the three self-test gates."""
    started = time.perf_counter()
    report = DoctorReport(generated_utc=_now())
    report.add(check_python())
    report.add(check_extension_version())
    for finding in check_packages():
        report.add(finding)
    for finding in check_extras():
        report.add(finding)
    for finding in check_shadowing():
        report.add(finding)
    report.add(check_temp(environ=environ))
    report.add(check_qt(run=include_qt, timeout=timeout))
    report.add(check_documented_files(root=root))
    if self_test:
        report.gates.append(gate_benchmark_baseline(root=root, full=full, timeout=max(timeout, 600.0)))
        report.gates.append(gate_release_content(root=root, timeout=max(timeout, 300.0)))
        report.gates.append(gate_smoke_dock(root=root, timeout=max(timeout, 300.0)))
    report.seconds = time.perf_counter() - started
    return report


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _cli_err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def cmd_doctor(args) -> int:
    """Diagnose this installation and print a fix for every finding."""
    report = run_doctor(
        root=getattr(args, "root", None),
        include_qt=not getattr(args, "no_qt", False),
        self_test=bool(getattr(args, "self_test", False)),
        full=bool(getattr(args, "full", False)),
        timeout=float(getattr(args, "timeout", 120.0) or 120.0),
    )
    strict = bool(getattr(args, "strict", False))
    if getattr(args, "json", False):
        print(json.dumps(report.as_dict(strict=strict), indent=2, ensure_ascii=False, default=str))
    elif not getattr(args, "quiet", False):
        print(report.text(strict=strict))
    if report.errors or (strict and report.warnings):
        _cli_err(
            f"odock doctor: {len(report.errors)} error(s) and "
            f"{len(report.warnings)} warning(s)"
        )
    return report.exit_code(strict=strict)


def add_doctor_parser(sub) -> None:
    """Register ``odock doctor`` on `sub` (called from :mod:`odock.cli_ext`).

    Idempotent: a re-applied registration line is not an error.
    """
    choices = getattr(sub, "choices", None)
    if isinstance(choices, Mapping) and "doctor" in choices:
        return
    parser = sub.add_parser(
        "doctor",
        help="diagnose this installation and print a fix for every finding",
        description=(
            "Check the interpreter, the compiled kernel against the package and "
            "checkout versions, the dependencies and extras, whether another "
            "distribution shadows this one, whether the temp directory is "
            "writable, whether an offscreen GL context can be created, and whether "
            "the promised files are present.  Each finding says why it matters and "
            "what to do; --strict turns a warning into a non-zero exit for CI, and "
            "--self-test runs the benchmark baseline, the release content rules and "
            "a smoke docking run with their measured numbers."
        ),
    )
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero when there is any warning as well as any error",
    )
    parser.add_argument(
        "--self-test", action="store_true", dest="self_test",
        help="also run the benchmark baseline check, the release content rules and "
             "a smoke docking run on the bundled demo",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="with --self-test: run the full benchmark re-docking gate (minutes)",
    )
    parser.add_argument(
        "--no-qt", action="store_true", help="skip the Qt/GL probe"
    )
    parser.add_argument(
        "--root", help="the checkout to inspect (default: the one this package is in)"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="per-check timeout in seconds"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="print nothing on success")
    parser.set_defaults(func=cmd_doctor)

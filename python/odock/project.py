# SPDX-License-Identifier: GPL-3.0-or-later
"""A docking *run* as one reopenable, verifiable file (``.odockproj``).

A docking run normally leaves loose files behind: a receptor PDBQT, a ligand
PDBQT, a box JSON, a pose PDBQT, a results table, maybe an SVG diagram.  None of
them says which other files it belongs with, which settings produced it, or
whether any of it has changed since.  A project file is that missing container:
a single ``.odockproj`` that can be reopened elsewhere, re-verified, and turned
back into the same numbers.

Design contract
---------------

* **Self-contained.**  Everything needed to reopen the run is *inside* the file.
  The manifest may record where a file came from, but only as a normalised,
  display-only name (see :func:`portable_path`): opening a project never reads
  the original paths, and a project copied to another machine behaves
  identically.
* **Verifiable.**  Every stored entry carries a SHA-256 in the manifest *and* in
  a plain-text ``SHA256SUMS`` member.  :meth:`Project.verify` recomputes every
  hash, reports the entries that changed, the ones that are missing, the ones
  that are present but unlisted, and any disagreement between the manifest and
  ``SHA256SUMS``.  A hash is evidence that a file has *not changed*; it is never
  evidence that the file is *correct* (see ``docs/PROJECTS.md``).
* **Forward-compatible by refusing.**  ``schema_version`` is the compatibility
  contract.  A file written by a newer OpenDocking is refused with a message
  naming both versions; an older one is either migrated by a registered
  migration or refused with the reason.  There is no "best effort" parse.

Container layout
----------------

The file is a ZIP archive (deflate), written with fixed member timestamps so the
same content produces the same bytes::

    project.json          the manifest: schema, tool, run, box, engine,
                          preparation, inputs, analyses, reproducibility,
                          and the index of every stored file with its hash
    SHA256SUMS            ``<sha256>  <member>`` for every member above
    files/receptor.pdbqt  the prepared inputs and the run's artefacts
    files/ligand.pdbqt
    files/poses.pdbqt
    files/poses.json      the poses with full-precision coordinates
    files/box.json
    files/analysis.json
    files/figures/*.svg

Typical use::

    import odock
    from odock import project

    box = odock.box_from_ligand(ligand_mol, buffer=8.0)
    result = odock.dock("receptor.pdbqt", "ligand.pdbqt", box, seed=42)
    saved = project.save_project("run.odockproj", result, receptor="receptor.pdbqt",
                                 ligand="ligand.pdbqt", box=box)

    reopened = project.open_project("run.odockproj")
    print(reopened.verify().summary())
    assert reopened.result().best_affinity == result.best_affinity
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "DATA_PREFIX",
    "ENGINE_DEFAULTS",
    "FORMAT_NAME",
    "MANIFEST_NAME",
    "PROJECT_SUFFIX",
    "REPRODUCTION_KEYS",
    "ROLE_ANALYSIS",
    "ROLE_FIGURE",
    "ROLE_LIGAND",
    "ROLE_POSES",
    "ROLE_POSE_DATA",
    "ROLE_RECEPTOR",
    "RUN_LIMITS",
    "SCHEMA_VERSION",
    "SUMS_NAME",
    "Project",
    "ProjectError",
    "ReproduceResult",
    "SchemaVersionError",
    "StoredFile",
    "VerifyResult",
    "add_project_parser",
    "add_report_parsers",
    "analyse_result",
    "campaign_index",
    "compare_projects",
    "comparison_summary",
    "extract_project",
    "find_absolute_paths",
    "media_type_for",
    "normalise_paths",
    "open_project",
    "portable_command",
    "portable_path",
    "project_info",
    "redact_paths",
    "reproduce_project",
    "resolve_engine_settings",
    "result_from_dock_json",
    "result_from_pdbqt",
    "save_project",
    "save_screen_projects",
    "settings_difference",
    "tool_version",
    "verify_project",
]

PathLike = Union[str, os.PathLike]

#: The value of ``project.json["format"]``: a cheap "is this our file?" check.
FORMAT_NAME = "odock-project"

#: The layout this build writes.  Bump it when a reader of the previous version
#: could misread the new one.
SCHEMA_VERSION = 2

#: The oldest layout this build can still migrate onto :data:`SCHEMA_VERSION`.
#: A file below this is refused with the reason, never guessed at.
MIN_MIGRATABLE_VERSION = 1

PROJECT_SUFFIX = ".odockproj"
MANIFEST_NAME = "project.json"
SUMS_NAME = "SHA256SUMS"
DATA_PREFIX = "files/"

#: Roles a stored entry can have.  A role is metadata, not a file name, so a
#: project remains readable when a file has to be renamed.
ROLE_RECEPTOR = "receptor"
ROLE_LIGAND = "ligand"
ROLE_POSES = "poses"
ROLE_POSE_DATA = "pose-data"
ROLE_ANALYSIS = "analysis"
ROLE_BOX = "box"
ROLE_PREPARATION = "preparation"
ROLE_FIGURE = "figure"
ROLE_INPUT = "input"
ROLE_LOG = "log"

#: Suffix -> media type, for the manifest's ``media_type`` field.
_MEDIA_TYPES: Dict[str, str] = {
    ".pdbqt": "chemical/x-pdbqt",
    ".pdb": "chemical/x-pdb",
    ".sdf": "chemical/x-mdl-sdfile",
    ".mol": "chemical/x-mdl-molfile",
    ".mol2": "chemical/x-mol2",
    ".smi": "chemical/x-daylight-smiles",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".md": "text/markdown",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

#: What a completed docking run does **not** establish.  Shipped inside the
#: project file and printed in the HTML report, because the numbers are easy to
#: quote and the caveats are not.
RUN_LIMITS: Tuple[str, ...] = (
    "A project captures a *run*, not an interactive session: the poses, the "
    "settings and the analysis are frozen, but nothing about the session that "
    "produced them (a manually moved box, an intermediate edit) is recorded.",
    "The SHA-256 hashes prove that a stored file has not changed since the "
    "project was written.  They do not prove that a file is correct, that the "
    "input structure was right, or that the preparation did what its settings "
    "say.",
    "The affinity is the score of one empirical force field, not a measured "
    "binding free energy; compare values within one series and one box, never "
    "across programs.",
    "The search is rigid-receptor (unless the project says otherwise): receptor "
    "side-chain flexibility, explicit water and protonation states are the ones "
    "in the stored inputs.",
    "Re-running the *docking* from the stored inputs with the recorded seed "
    "reproduces the run only on the same tool version and kernel; a different "
    "version may legitimately produce different poses.",
    "A pose is a hypothesis about a binding mode.  Interaction fingerprints, "
    "ligand efficiency and strain describe that hypothesis; they do not "
    "validate it against experiment.",
    "The schema version, not the file extension, is the compatibility contract: "
    "a newer schema is refused rather than read approximately, and an older one "
    "is migrated only when a migration is registered for it.",
)

#: Absolute-path shapes that must never reach a published file.  The first is a
#: Windows drive path, the second and third are POSIX home directories, and the
#: last catches a UNC share.
_ABSOLUTE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"[A-Za-z]:[\\/][^\s\"'|<>;,)]*"),
    re.compile(r"/(?:home|Users|root)/[^\s\"'|<>;,)]*"),
    re.compile(r"\\\\[^\s\"'|<>;,)]*"),
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProjectError(Exception):
    """A project could not be read, written or verified."""


class SchemaVersionError(ProjectError):
    """The project's schema version is not one this build can honour.

    Raised rather than parsed approximately: misreading a newer layout would
    silently produce wrong poses or wrong hashes, which is exactly the failure a
    version number exists to prevent.
    """


# ---------------------------------------------------------------------------
# Path normalisation: a recorded path must never leak the machine it was made on
# ---------------------------------------------------------------------------


def _as_posix(text: str) -> str:
    return str(text).replace("\\", "/")


def portable_path(value: PathLike, *, base: Optional[PathLike] = None,
                  home: Optional[PathLike] = None) -> str:
    """`value` as a portable, machine-independent string.

    The rules, in order:

    1. a relative path is returned as-is (with ``/`` separators);
    2. a path inside `base` (the project's directory, or the working directory)
       becomes relative to it;
    3. a path inside the user's home becomes ``~/...``;
    4. anything else absolute is reduced to its **file name**.

    Rule 4 is deliberate.  A shipped file once recorded the absolute checkout
    path of the machine that produced it; a path that exists on one laptop is
    not information, and keeping its tail would still publish a stranger's
    directory layout.  The file name is what a reader can act on.
    """
    raw = os.fspath(value)
    if not raw:
        return ""
    path = Path(raw)
    # A POSIX path is not "absolute" to pathlib on Windows and a Windows path is
    # not absolute to pathlib elsewhere; both are absolute for our purpose.
    looks_absolute = bool(
        path.is_absolute()
        or raw.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", raw)
    )
    if not looks_absolute:
        return _as_posix(raw)

    def _resolved(candidate: Any) -> Optional[Path]:
        if candidate is None:
            return None
        try:
            return Path(candidate).resolve()
        except OSError:  # pragma: no cover - an unresolvable root is unusable
            return None

    # `base` and the working directory first: a path inside the checkout is best
    # reported relative to it, and only then does the home directory become a
    # useful (still non-absolute) shorthand.
    roots: List[Tuple[Optional[Path], str]] = [
        (_resolved(base), ""),
        (_resolved(Path.cwd()), ""),
        (_resolved(home if home is not None else Path.home()), "~/"),
    ]
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - non-existent drive on Windows
        resolved = path

    for root, prefix in roots:
        if root is None:
            continue
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        text = _as_posix(str(relative))
        if text in (".", ""):
            return prefix.rstrip("/") or "."
        return prefix + text
    return path.name or _as_posix(raw)


def find_absolute_paths(text: str) -> List[str]:
    """Every absolute-path-looking token in `text` (for tests and guards)."""
    if not isinstance(text, str):
        return []
    found: List[str] = []
    for pattern in _ABSOLUTE_PATTERNS:
        for match in pattern.finditer(text):
            token = match.group(0).rstrip(".,;:")
            if token and token not in found:
                found.append(token)
    return found


def redact_paths(text: str, *, base: Optional[PathLike] = None) -> str:
    """Replace every absolute path in `text` with its portable form."""
    if not isinstance(text, str):
        return text
    out = text
    for pattern in _ABSOLUTE_PATTERNS:
        # Right-to-left so the offsets of the matches still to be replaced stay
        # valid as the string changes length.
        matches = list(pattern.finditer(out))
        for match in reversed(matches):
            token = match.group(0)
            # A sentence-ending period is punctuation, not part of the file name.
            stripped = token.rstrip(".,;:")
            trailing = token[len(stripped):]
            replacement = portable_path(stripped, base=base) + trailing
            out = out[: match.start()] + replacement + out[match.end():]
    return out


def normalise_paths(payload: Any, *, base: Optional[PathLike] = None) -> Any:
    """Recursively redact absolute paths in a JSON-like structure.

    Applied to the whole manifest before it is written, so a leak cannot be
    introduced by one call site forgetting to normalise: every string that goes
    into ``project.json`` passes through here.
    """
    if isinstance(payload, str):
        return redact_paths(payload, base=base)
    if isinstance(payload, Mapping):
        return {str(key): normalise_paths(value, base=base) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [normalise_paths(item, base=base) for item in payload]
    return payload


def portable_command(argv: Optional[Union[str, Sequence[str]]] = None,
                     *, base: Optional[PathLike] = None) -> str:
    """The reproducible form of a command, with no machine-specific path.

    ``C:\\work\\opendocking\\.venv\\Scripts\\python.exe -m odock.cli dock ...``
    becomes ``python -m odock.cli dock ...``, and any argument inside the
    checkout becomes relative to it.  The result is what a reader can actually
    paste into a shell.  A single string is split on whitespace first, so a
    recorded command can be passed either way.
    """
    if argv is None:
        arguments = list(sys.argv)
    elif isinstance(argv, str):
        arguments = argv.split()
    else:
        arguments = list(argv)
    if not arguments:  # pragma: no cover - depends on how python was started
        return ""
    cleaned = list(arguments)
    # A leading interpreter path becomes ``python``; a script path under the
    # checkout becomes the module form so the line still runs.
    for index, token in enumerate(cleaned):
        if not token or token.startswith("-"):
            continue
        lowered = token.lower()
        if lowered.endswith(("python.exe", "python3", "python")):
            cleaned[index] = "python"
            continue
        if index > 0 and cleaned[index - 1] == "python" and (
            os.sep in token or "/" in token
        ):
            candidate = Path(token)
            if candidate.exists() and candidate.suffix == ".py":
                module = candidate.stem
                parent = candidate.parent.name
                package = f"{parent}.{module}" if parent not in ("", ".", "odock") else f"odock.{module}"
                cleaned[index] = f"-m {package}"
        if os.sep in token or "/" in token or token.startswith("~"):
            cleaned[index] = portable_path(token, base=base)
    text = " ".join(str(token) for token in cleaned if token != "")
    return redact_paths(" ".join(text.split()), base=base)


# ---------------------------------------------------------------------------
# Hashing and small helpers
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    """The hex SHA-256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def _sha256_stream(handle) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def media_type_for(name: str) -> str:
    """The media type recorded for a stored member."""
    suffix = Path(str(name)).suffix.lower()
    return _MEDIA_TYPES.get(suffix, "application/octet-stream")


def tool_version() -> str:
    """The installed distribution version, falling back to the API version."""
    return _distribution_version() or _fallback_version()


def _distribution_version() -> Optional[str]:
    """The version recorded by the installed distribution, when there is one."""
    try:
        from importlib.metadata import version as _distribution_version_of

        return str(_distribution_version_of("opendocking"))
    except Exception:
        return None


def _fallback_version() -> str:
    """The version the importable package reports when no metadata is present."""
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _kernel_version() -> str:
    try:
        from .docking import kernel_version

        return str(kernel_version())
    except Exception:  # pragma: no cover - a pure-Python install
        return "unknown"


def _platform_label() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_text_or_bytes(source: Any, *, what: str) -> Tuple[bytes, str]:
    """Resolve `source` into ``(bytes, display name)``.

    Accepts bytes, a path, or the document text itself.  Text is recognised by a
    newline or a record keyword, so a PDBQT string with a Windows-looking name is
    still treated as text.
    """
    if source is None:
        raise ProjectError(f"no {what} given")
    if isinstance(source, bytes):
        return source, what
    if isinstance(source, (str, os.PathLike)):
        text = os.fspath(source)
        looks_like_text = "\n" in text or text.lstrip()[:6].upper() in (
            "ATOM  ", "HETATM", "REMARK", "MODEL ", "ROOT", "COMPND", "HEADER",
        )
        if not looks_like_text:
            path = Path(text)
            if not path.exists():
                raise ProjectError(f"no such {what}: {text}")
            if path.is_dir():
                raise ProjectError(f"{what} is a directory: {text}")
            return path.read_bytes(), path.name
        return text.encode("utf-8"), what
    return bytes(source), what  # pragma: no cover - exotic source


def _coerce_box(box: Any):
    """A :class:`odock.BoxSpec` from a BoxSpec, mapping, or JSON file path."""
    from .prepare import BoxSpec

    if box is None:
        return None
    if isinstance(box, BoxSpec):
        return box
    if isinstance(box, (str, os.PathLike)):
        data = json.loads(Path(box).read_text(encoding="utf-8"))
        return _coerce_box(data)
    if isinstance(box, Mapping):
        payload = dict(box)
        return BoxSpec(
            center=tuple(float(v) for v in payload["center"]),
            size=tuple(float(v) for v in payload["size"]),
            spacing=float(payload.get("spacing", 0.375)),
        )
    raise ProjectError(
        "the box must be an odock.BoxSpec, a {'center','size','spacing'} mapping, "
        f"or a box JSON file, not {type(box).__name__}"
    )


# ---------------------------------------------------------------------------
# Poses: reading a multi-model PDBQT back into a DockResult
# ---------------------------------------------------------------------------

_VINA_RESULT = re.compile(
    r"^REMARK\s+(?:VINA\s+RESULT|OPEN\s*DOCKING\s*RESULT)\s*:?\s*"
    r"(-?[0-9.]+)\s+(-?[0-9.]+)\s+(-?[0-9.]+)",
    re.IGNORECASE,
)
_DECOMPOSITION = {
    "INTER": "inter",
    "INTRA": "intra",
    "CONF_INDEPENDENT": "conf_independent",
    "UNBOUND": "unbound",
}


def _split_models(text: str) -> List[str]:
    """The ``MODEL`` blocks of a PDBQT document (the whole text when there are none)."""
    lines = text.splitlines()
    if not any(line.startswith("MODEL") for line in lines):
        return [text] if text.strip() else []
    blocks: List[str] = []
    current: Optional[List[str]] = None
    for line in lines:
        if line.startswith("MODEL"):
            if current:
                blocks.append("\n".join(current) + "\n")
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current is not None:
                blocks.append("\n".join(current) + "\n")
            current = None
            continue
        if current is not None:
            current.append(line)
    if current:
        blocks.append("\n".join(current) + "\n")
    return blocks


def _remark_fields(block: str) -> Dict[str, float]:
    """The energy decomposition of one pose block, as far as it is recorded."""
    found: Dict[str, float] = {}
    for line in block.splitlines():
        if not line.startswith("REMARK"):
            continue
        body = line[len("REMARK"):].strip()
        if ":" not in body:
            continue
        name, _, value = body.partition(":")
        key = name.strip().upper().replace(" ", "_").replace("+", "_")
        if key == "INTER__INTRA":
            continue
        target = _DECOMPOSITION.get(name.strip().upper())
        if target is None:
            continue
        try:
            found[target] = float(value.split()[0])
        except (IndexError, ValueError):
            continue
    return found


def _inside_box(coords, box) -> bool:
    if box is None or coords is None:
        return True
    import numpy as np

    center = np.asarray([float(v) for v in box.center], dtype=float)
    size = np.asarray([float(v) for v in box.size], dtype=float)
    lower = center - size / 2.0 - 1e-9
    upper = center + size / 2.0 + 1e-9
    array = np.asarray(coords, dtype=float)
    if array.size == 0:
        return True
    return bool(((array >= lower) & (array <= upper)).all())


def result_from_pdbqt(
    poses: Any,
    *,
    box: Any = None,
    receptor: Any = None,
    ligand: Any = None,
    seed: Optional[int] = None,
    scoring: str = "vina",
    elapsed: float = 0.0,
    grid_points: int = 0,
    grid_mb: int = 0,
    num_tors: Optional[float] = None,
    num_movable_atoms: int = 0,
    num_dof: int = 0,
    exact: bool = True,
    cancelled: bool = False,
) -> Any:
    """Rebuild a :class:`odock.DockResult` from a multi-model pose PDBQT.

    This is what lets a run that exists only as files (a screening shortlist, a
    demo pose file, an old AutoDock Vina output) be wrapped in a project: the
    result object the rest of the tool consumes is reconstructed from the pose
    file, with the affinity and the energy decomposition read from the
    ``REMARK`` records and the coordinates from the ``ATOM``/``HETATM`` records.

    The values a PDBQT does not carry (``exhaustiveness``, the search protocol,
    the wall time) are the caller's to supply; what it cannot supply stays at its
    documented default rather than being guessed.
    """
    import numpy as np

    from .docking import DockResult, Pose
    from .report import parse_pdbqt_atoms

    text, _ = _as_text_or_bytes(poses, what="pose file")
    poses_text = text.decode("utf-8", errors="replace")
    blocks = _split_models(poses_text)
    if not blocks:
        raise ProjectError("the pose file holds no model")

    box_spec = _coerce_box(box)
    first_atoms = parse_pdbqt_atoms(blocks[0])
    atom_names = tuple(atom.name for atom in first_atoms)

    built: List[Any] = []
    for index, block in enumerate(blocks):
        atoms = parse_pdbqt_atoms(block)
        if not atoms:
            continue
        coords = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
        match = None
        for line in block.splitlines():
            match = _VINA_RESULT.match(line)
            if match:
                break
        affinity = float(match.group(1)) if match else 0.0
        rmsd_lb = float(match.group(2)) if match else 0.0
        rmsd_ub = float(match.group(3)) if match else 0.0
        fields = _remark_fields(block)
        built.append(
            Pose(
                index=index,
                affinity=affinity,
                rmsd_lower_bound=rmsd_lb,
                rmsd_upper_bound=rmsd_ub,
                inter=fields.get("inter", 0.0),
                intra=fields.get("intra", 0.0),
                conf_independent=fields.get("conf_independent", 0.0),
                unbound=fields.get("unbound", 0.0),
                in_box=_inside_box(coords, box_spec),
                num_atoms=len(atoms),
                position=None,
                orientation=None,
                torsions=None,
                coords=coords,
            )
        )
    if not built:
        raise ProjectError("the pose file holds no ATOM/HETATM record")

    receptor_text = ""
    if receptor is not None:
        receptor_bytes, _ = _as_text_or_bytes(receptor, what="receptor")
        receptor_text = receptor_bytes.decode("utf-8", errors="replace")
    ligand_text = ""
    if ligand is not None:
        ligand_bytes, _ = _as_text_or_bytes(ligand, what="ligand")
        ligand_text = ligand_bytes.decode("utf-8", errors="replace")

    if num_tors is None:
        match = re.search(r"^TORSDOF\s+([0-9.]+)", poses_text, re.MULTILINE)
        num_tors = float(match.group(1)) if match else 0.0

    result = DockResult(
        poses=built,
        seed=int(seed or 0),
        scoring=str(scoring),
        grid_mb=int(grid_mb),
        grid_points=int(grid_points),
        num_tors=float(num_tors),
        num_movable_atoms=int(num_movable_atoms) or len(first_atoms),
        num_dof=int(num_dof),
        exact=bool(exact),
        receptor_pdbqt=receptor_text,
        ligand_pdbqt=ligand_text,
        box=box_spec,
        elapsed=float(elapsed),
        ligand_atom_order=atom_names,
        _pdbqt=poses_text,
    )
    result.cancelled = bool(cancelled)
    # A PDBQT carries no seed.  Recording whether the caller supplied one is what
    # lets a report say "not recorded" instead of inventing a seed of 0.
    result.seed_recorded = seed is not None
    return result


# ---------------------------------------------------------------------------
# The manifest and its versions
# ---------------------------------------------------------------------------


def result_from_dock_json(
    payload: Mapping[str, Any],
    *,
    box: Any = None,
    receptor: Any = None,
    ligand: Any = None,
    scoring: Optional[str] = None,
) -> Any:
    """Rebuild a :class:`odock.DockResult` from the JSON ``odock dock --json-out`` writes.

    That document is ``{"seed", "elapsed", "box", "grid_points", "poses": [...]}``
    with the pose fields of :meth:`odock.Pose.to_dict`, i.e. **without**
    coordinates (they are large).  The result is therefore complete for a ranking
    table and empty for anything that needs geometry; the caller is expected to
    say so rather than pretend.
    """
    from .docking import DockResult, Pose

    if not isinstance(payload, Mapping):
        raise ProjectError(f"a dock JSON document must be an object, not {type(payload).__name__}")
    poses_payload = payload.get("poses")
    if not isinstance(poses_payload, list) or not poses_payload:
        raise ProjectError("the dock JSON document holds no pose")

    box_spec = _coerce_box(box if box is not None else payload.get("box"))
    poses: List[Any] = []
    for position, item in enumerate(poses_payload):
        if not isinstance(item, Mapping):
            continue
        poses.append(
            Pose(
                index=int(item.get("index", position)),
                affinity=float(item.get("affinity", 0.0) or 0.0),
                rmsd_lower_bound=float(
                    item.get("rmsd_lower_bound", item.get("rmsd_lb", 0.0)) or 0.0
                ),
                rmsd_upper_bound=float(
                    item.get("rmsd_upper_bound", item.get("rmsd_ub", 0.0)) or 0.0
                ),
                inter=float(item.get("inter", 0.0) or 0.0),
                intra=float(item.get("intra", 0.0) or 0.0),
                conf_independent=float(item.get("conf_independent", 0.0) or 0.0),
                unbound=float(item.get("unbound", 0.0) or 0.0),
                in_box=bool(item.get("in_box", True)),
                num_atoms=int(item.get("num_atoms", 0) or 0),
                position=item.get("position"),
                orientation=item.get("orientation"),
                torsions=item.get("torsions"),
                coords=None,
            )
        )
    if not poses:
        raise ProjectError("the dock JSON document holds no usable pose")

    receptor_text = ""
    if receptor is not None:
        receptor_text = _as_text_or_bytes(receptor, what="receptor")[0].decode("utf-8", "replace")
    ligand_text = ""
    if ligand is not None:
        ligand_text = _as_text_or_bytes(ligand, what="ligand")[0].decode("utf-8", "replace")

    result = DockResult(
        poses=poses,
        seed=int(payload.get("seed") or 0),
        scoring=str(scoring or payload.get("scoring") or "vina"),
        grid_mb=int(payload.get("grid_mb") or 0),
        grid_points=int(payload.get("grid_points") or 0),
        num_tors=float(payload.get("num_tors") or 0.0),
        num_movable_atoms=int(payload.get("num_movable_atoms") or 0),
        num_dof=int(payload.get("num_dof") or 0),
        exact=bool(payload.get("exact", True)),
        receptor_pdbqt=receptor_text,
        ligand_pdbqt=ligand_text,
        box=box_spec,
        elapsed=float(payload.get("elapsed") or 0.0),
        ligand_atom_order=tuple(str(name) for name in payload.get("ligand_atom_order") or ()),
        _pdbqt=str(payload.get("pdbqt") or ""),
    )
    result.cancelled = bool(payload.get("cancelled", False))
    result.seed_recorded = True
    return result


def _migrate_1_to_2(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Draft schema 1 -> schema 2.

    Schema 1 (the pre-release draft) listed its members under ``contents`` with
    a ``name`` key, carried no ``media_type``, and had no ``reproducibility``
    block.  The migration is total: every field schema 2 requires is derived from
    what schema 1 recorded, and nothing is invented.
    """
    migrated = dict(payload)
    contents = migrated.pop("contents", None) or []
    files: List[Dict[str, Any]] = []
    for item in contents:
        if not isinstance(item, Mapping):
            continue
        entry = dict(item)
        arcname = str(entry.pop("name", entry.get("arcname", "")))
        entry["arcname"] = arcname
        entry.setdefault("role", "input")
        entry.setdefault("media_type", media_type_for(arcname))
        entry.setdefault("label", Path(arcname).name)
        entry.setdefault("source", Path(arcname).name)
        files.append(entry)
    migrated["files"] = files
    migrated["schema_version"] = 2
    inputs = migrated.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, dict):
                item.setdefault("media_type", media_type_for(str(item.get("name", ""))))
    run = migrated.get("run")
    if isinstance(run, dict):
        run.setdefault("kind", "poses")
    migrated.setdefault(
        "reproducibility",
        {
            "command": "",
            "note": (
                "migrated from schema 1, which did not record a command; the "
                "inputs, the seed and the settings below are the schema-1 record"
            ),
        },
    )
    return migrated


#: Registered migrations, keyed by the version they read.
_MIGRATIONS = {1: _migrate_1_to_2}


def load_manifest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and (when needed) migrate a ``project.json`` payload.

    Raises :class:`SchemaVersionError` for a version this build cannot honour,
    naming both versions so the reader knows whether to upgrade OpenDocking or
    ask for an older export.
    """
    if not isinstance(payload, Mapping):
        raise ProjectError("the project manifest is not a JSON object")
    version = payload.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ProjectError(
            "this is not an OpenDocking project: project.json has no integer "
            "schema_version"
        )
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"this project was written with schema version {version}, and this "
            f"build of OpenDocking reads up to {SCHEMA_VERSION}. Upgrade "
            "OpenDocking, or ask the author for an export in schema "
            f"{SCHEMA_VERSION} (the schema version is the compatibility contract)."
        )
    if version < MIN_MIGRATABLE_VERSION:
        raise SchemaVersionError(
            f"this project uses schema version {version}, which predates the "
            f"oldest version this build can migrate ({MIN_MIGRATABLE_VERSION}); "
            "it is refused rather than misread."
        )
    migrated_from: Optional[int] = None
    current = dict(payload)
    while int(current["schema_version"]) < SCHEMA_VERSION:
        step = int(current["schema_version"])
        migration = _MIGRATIONS.get(step)
        if migration is None:  # pragma: no cover - registry gap, not user input
            raise SchemaVersionError(
                f"no migration is registered from schema {step} to {step + 1}"
            )
        current = migration(current)
        migrated_from = step
    if migrated_from is not None:
        current["migrated_from"] = migrated_from
    return current


def manifest_version_note(manifest: Mapping[str, Any]) -> str:
    """One line describing how this manifest's version was handled."""
    if manifest.get("migrated_from") is not None:
        return (
            f"schema {manifest['schema_version']} (migrated from "
            f"{manifest['migrated_from']})"
        )
    return f"schema {manifest['schema_version']}"


# ---------------------------------------------------------------------------
# Stored entries and the verification report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredFile:
    """One member of the archive, as the manifest describes it."""

    role: str
    arcname: str
    sha256: str
    size: int
    media_type: str = "application/octet-stream"
    label: str = ""
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "arcname": self.arcname,
            "sha256": self.sha256,
            "size": int(self.size),
            "media_type": self.media_type,
            "label": self.label or Path(self.arcname).name,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StoredFile":
        arcname = str(data.get("arcname") or data.get("name") or "")
        return cls(
            role=str(data.get("role") or "input"),
            arcname=arcname,
            sha256=str(data.get("sha256") or ""),
            size=int(data.get("size") or 0),
            media_type=str(data.get("media_type") or media_type_for(arcname)),
            label=str(data.get("label") or Path(arcname).name),
            source=str(data.get("source") or ""),
        )


@dataclass
class VerifyResult:
    """The outcome of :meth:`Project.verify`.

    ``problems`` holds every discrepancy, each one naming the member it is about,
    so a failure can be read without opening the archive.  ``ok`` is true only
    when every list is empty.
    """

    path: Path
    schema_version: int
    entries: int
    checked: int
    size_bytes: int
    problems: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unlisted: List[str] = field(default_factory=list)
    sums_problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.problems or self.changed or self.missing or self.unlisted
                    or self.sums_problems)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path.name,
            "ok": bool(self.ok),
            "schema_version": int(self.schema_version),
            "entries": int(self.entries),
            "checked": int(self.checked),
            "size_bytes": int(self.size_bytes),
            "changed": list(self.changed),
            "missing": list(self.missing),
            "unlisted": list(self.unlisted),
            "sums_problems": list(self.sums_problems),
            "problems": list(self.problems),
        }

    def summary(self) -> str:
        """A short human-readable verdict."""
        head = (
            f"{self.path.name}: {'OK' if self.ok else 'FAILED'} "
            f"({self.checked} of {self.entries} stored file(s) re-hashed, "
            f"schema {self.schema_version}, {self.size_bytes / 1024.0:.1f} KiB)"
        )
        lines = [head]
        for label, items in (
            ("changed", self.changed),
            ("missing", self.missing),
            ("present but unlisted", self.unlisted),
            ("SHA256SUMS", self.sums_problems),
            ("problem", self.problems),
        ):
            for item in items:
                lines.append(f"  {label}: {item}")
        if self.ok:
            lines.append(
                "  note: the hashes show that nothing has changed; they do not "
                "show that the run is correct."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Opening a project
# ---------------------------------------------------------------------------


class Project:
    """An open ``.odockproj``: its manifest, its entries, and its run.

    The archive is not held open.  :meth:`read` opens it per call, so a project
    object stays valid across a long analysis and there is no file handle to
    leak.
    """

    def __init__(
        self,
        path: PathLike,
        manifest: Dict[str, Any],
        entries: Sequence[StoredFile],
        *,
        sums: Optional[Dict[str, str]] = None,
        sums_arcnames: Optional[Sequence[str]] = None,
    ) -> None:
        self.path = Path(path)
        self.manifest = dict(manifest)
        self.entries: List[StoredFile] = list(entries)
        self._sums = dict(sums or {})
        self._sums_names: List[str] = list(sums_arcnames or [])
        self._cache: Dict[str, bytes] = {}

    # -- manifest views ----------------------------------------------------

    @property
    def schema_version(self) -> int:
        return int(self.manifest.get("schema_version", 0))

    @property
    def version_note(self) -> str:
        return manifest_version_note(self.manifest)

    @property
    def tool(self) -> Dict[str, Any]:
        tool = self.manifest.get("tool")
        return dict(tool) if isinstance(tool, Mapping) else {}

    @property
    def created(self) -> str:
        return str(self.manifest.get("created_utc") or "")

    @property
    def title(self) -> str:
        return str(self.manifest.get("title") or self.path.stem)

    @property
    def run(self) -> Dict[str, Any]:
        run = self.manifest.get("run")
        return dict(run) if isinstance(run, Mapping) else {}

    @property
    def engine(self) -> Dict[str, Any]:
        engine = self.manifest.get("engine")
        return dict(engine) if isinstance(engine, Mapping) else {}

    @property
    def preparation(self) -> Dict[str, Any]:
        prep = self.manifest.get("preparation")
        return dict(prep) if isinstance(prep, Mapping) else {}

    @property
    def reproducibility(self) -> Dict[str, Any]:
        block = self.manifest.get("reproducibility")
        return dict(block) if isinstance(block, Mapping) else {}

    @property
    def analyses(self) -> Dict[str, Any]:
        block = self.manifest.get("analyses")
        return dict(block) if isinstance(block, Mapping) else {}

    @property
    def inputs(self) -> List[Dict[str, Any]]:
        items = self.manifest.get("inputs")
        return [dict(item) for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []

    @property
    def campaign(self) -> Dict[str, Any]:
        block = self.manifest.get("campaign")
        return dict(block) if isinstance(block, Mapping) else {}

    @property
    def warnings(self) -> List[str]:
        items = self.manifest.get("warnings")
        return [str(item) for item in items] if isinstance(items, list) else []

    @property
    def does_not_establish(self) -> List[str]:
        items = self.manifest.get("does_not_establish")
        if isinstance(items, list) and items:
            return [str(item) for item in items]
        return list(RUN_LIMITS)

    # -- entries -----------------------------------------------------------

    def entry(self, role_or_name: str, *, index: int = 0) -> Optional[StoredFile]:
        """The entry for `role_or_name` (a role, a member name or a bare stem)."""
        matches = [
            item
            for item in self.entries
            if item.role == role_or_name
            or item.arcname == role_or_name
            or Path(item.arcname).name == role_or_name
        ]
        if not matches:
            return None
        if index >= len(matches):
            raise ProjectError(
                f"the project has {len(matches)} '{role_or_name}' entry(ies); "
                f"index {index} was asked for"
            )
        return matches[index]

    def roles(self) -> List[str]:
        seen: List[str] = []
        for item in self.entries:
            if item.role not in seen:
                seen.append(item.role)
        return seen

    def read(self, role_or_name: str, *, index: int = 0) -> bytes:
        """The bytes of one stored member.

        `role_or_name` is a role, a member name, or one of the two reserved
        members (``project.json``, ``SHA256SUMS``), which are reachable by name
        so that a report or a check can read the manifest itself.
        """
        if role_or_name in (MANIFEST_NAME, SUMS_NAME):
            with zipfile.ZipFile(self.path) as archive:
                try:
                    return archive.read(role_or_name)
                except KeyError:
                    raise ProjectError(f"{role_or_name} is missing from the archive")
        found = self.entry(role_or_name, index=index)
        if found is None:
            raise ProjectError(
                f"the project has no '{role_or_name}' entry; it holds: "
                + ", ".join(item.arcname for item in self.entries)
            )
        if found.arcname not in self._cache:
            with zipfile.ZipFile(self.path) as archive:
                try:
                    self._cache[found.arcname] = archive.read(found.arcname)
                except KeyError:
                    raise ProjectError(
                        f"{found.arcname} is listed in the manifest but missing "
                        "from the archive; run `odock project verify`"
                    )
        return self._cache[found.arcname]

    def text(self, role_or_name: str, *, index: int = 0) -> str:
        """One stored member, decoded as UTF-8 text."""
        return self.read(role_or_name, index=index).decode("utf-8", errors="replace")

    def has(self, role_or_name: str) -> bool:
        return self.entry(role_or_name) is not None

    # -- the run -----------------------------------------------------------

    def box(self):
        """The search box, or ``None`` when the project has none."""
        block = self.manifest.get("box")
        if not isinstance(block, Mapping):
            return None
        return _coerce_box(block)

    def receptor_text(self) -> str:
        return self.text(ROLE_RECEPTOR) if self.has(ROLE_RECEPTOR) else ""

    def ligand_text(self) -> str:
        return self.text(ROLE_LIGAND) if self.has(ROLE_LIGAND) else ""

    def poses_text(self) -> str:
        return self.text(ROLE_POSES) if self.has(ROLE_POSES) else ""

    def analysis(self) -> Dict[str, Any]:
        """The stored analysis, plus whatever the manifest carries."""
        block = dict(self.analyses)
        if self.has(ROLE_ANALYSIS):
            try:
                stored = json.loads(self.text(ROLE_ANALYSIS))
            except ValueError as exc:
                raise ProjectError(f"the stored analysis is not valid JSON: {exc}")
            if isinstance(stored, Mapping):
                merged = dict(stored)
                merged.update(block)
                return merged
        return block

    def result(self):
        """The run as a :class:`odock.DockResult`.

        The pose coordinates come from ``files/poses.json`` (full precision) when
        it is present, and from the stored PDBQT otherwise; the scalar settings
        come from the manifest, so a reopened result compares equal to the one
        that was saved.
        """
        import numpy as np

        from .docking import DockResult, Pose

        run = self.run
        engine = self.engine
        box = self.box()
        pose_records: List[Dict[str, Any]] = []
        if self.has(ROLE_POSE_DATA):
            payload = json.loads(self.text(ROLE_POSE_DATA))
            if isinstance(payload, Mapping):
                raw = payload.get("poses")
                if isinstance(raw, list):
                    pose_records = [dict(item) for item in raw if isinstance(item, Mapping)]

        ligand_text = self.ligand_text()
        poses: List[Any] = []
        if pose_records:
            for index, record in enumerate(pose_records):
                coords = record.get("coords")
                array = (
                    np.asarray(coords, dtype=float)
                    if isinstance(coords, list) and coords
                    else None
                )
                poses.append(
                    Pose(
                        index=int(record.get("index", index)),
                        affinity=float(record.get("affinity", 0.0) or 0.0),
                        rmsd_lower_bound=float(
                            record.get("rmsd_lower_bound", record.get("rmsd_lb", 0.0)) or 0.0
                        ),
                        rmsd_upper_bound=float(
                            record.get("rmsd_upper_bound", record.get("rmsd_ub", 0.0)) or 0.0
                        ),
                        inter=float(record.get("inter", 0.0) or 0.0),
                        intra=float(record.get("intra", 0.0) or 0.0),
                        conf_independent=float(record.get("conf_independent", 0.0) or 0.0),
                        unbound=float(record.get("unbound", 0.0) or 0.0),
                        in_box=bool(record.get("in_box", True)),
                        num_atoms=int(record.get("num_atoms", 0) or 0),
                        position=record.get("position"),
                        orientation=record.get("orientation"),
                        torsions=record.get("torsions"),
                        coords=array,
                    )
                )

        atom_order = run.get("ligand_atom_order")
        atom_order = tuple(str(name) for name in atom_order) if isinstance(atom_order, list) else ()
        result = DockResult(
            poses=poses,
            seed=int(run.get("seed") or 0),
            scoring=str(engine.get("scoring") or run.get("scoring") or "vina"),
            grid_mb=int(run.get("grid_mb") or 0),
            grid_points=int(run.get("grid_points") or 0),
            num_tors=float(run.get("num_tors") or 0.0),
            num_movable_atoms=int(run.get("num_movable_atoms") or 0),
            num_dof=int(run.get("num_dof") or 0),
            exact=bool(run.get("exact", True)),
            receptor_pdbqt=self.receptor_text(),
            ligand_pdbqt=ligand_text,
            box=box,
            elapsed=float(run.get("elapsed_s") or 0.0),
            ligand_atom_order=atom_order,
            _pdbqt=self.poses_text(),
        )
        result.cancelled = bool(run.get("cancelled", False))
        result.seed_recorded = run.get("seed") is not None
        if not poses and self.has(ROLE_POSES):
            # No pose-data member (a schema-1 project, say): rebuild from the
            # PDBQT, which is exact for everything the file carries.
            rebuilt = result_from_pdbqt(
                self.poses_text(),
                box=box,
                receptor=self.receptor_text() or None,
                ligand=ligand_text or None,
                seed=int(run.get("seed") or 0),
                scoring=str(engine.get("scoring") or run.get("scoring") or "vina"),
                elapsed=float(run.get("elapsed_s") or 0.0),
                grid_points=int(run.get("grid_points") or 0),
                grid_mb=int(run.get("grid_mb") or 0),
                num_tors=run.get("num_tors"),
            )
            rebuilt.ligand_atom_order = atom_order or rebuilt.ligand_atom_order
            return rebuilt
        return result

    def reanalyse(self, *, strain: bool = False) -> Dict[str, Any]:
        """Recompute the analysis from the *stored* inputs.

        This is the check that matters for a round trip: the numbers in
        ``files/analysis.json`` were computed from the same embedded receptor and
        ligand, so recomputing them in a fresh interpreter has to give the same
        values.  A comparison of the two dictionaries is what
        ``tests/test_project.py`` asserts.
        """
        result = self.result()
        receptor = self.receptor_text() or None
        ligand = self.ligand_text() or None
        return analyse_result(result, receptor=receptor, ligand=ligand, strain=strain)

    # -- verification ------------------------------------------------------

    def verify(self) -> VerifyResult:
        """Recompute every stored hash and report every discrepancy."""
        report = VerifyResult(
            path=self.path,
            schema_version=self.schema_version,
            entries=len(self.entries),
            checked=0,
            size_bytes=self.path.stat().st_size if self.path.exists() else 0,
        )
        if not self.path.exists():
            report.problems.append(f"{self.path.name} does not exist")
            return report
        if not zipfile.is_zipfile(self.path):
            report.problems.append(
                f"{self.path.name} is not a ZIP archive (a .odockproj file is a "
                "ZIP container)"
            )
            return report

        listed = {item.arcname for item in self.entries}
        with zipfile.ZipFile(self.path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                duplicates = sorted({name for name in names if names.count(name) > 1})
                report.problems.append(
                    "the archive holds duplicate member(s): " + ", ".join(duplicates)
                )
            present = set(names)
            for item in self.entries:
                if item.arcname not in present:
                    report.missing.append(item.arcname)
                    continue
                with archive.open(item.arcname) as handle:
                    digest, size = _sha256_stream(handle)
                report.checked += 1
                if digest != item.sha256:
                    report.changed.append(
                        f"{item.arcname}: sha256 {item.sha256[:12]}… -> {digest[:12]}…"
                    )
                if size != item.size:
                    report.changed.append(
                        f"{item.arcname}: size {item.size} -> {size} bytes"
                    )
                recorded = self._sums.get(item.arcname)
                if recorded is not None and recorded != digest:
                    report.sums_problems.append(
                        f"{item.arcname}: SHA256SUMS says {recorded[:12]}…, the file "
                        f"hashes to {digest[:12]}…"
                    )
            for name in sorted(present - listed - {MANIFEST_NAME, SUMS_NAME}):
                report.unlisted.append(name)
            if MANIFEST_NAME not in present:
                report.problems.append(f"{MANIFEST_NAME} is missing from the archive")
            if SUMS_NAME not in present:
                report.problems.append(
                    f"{SUMS_NAME} is missing: the archive cannot be checked against "
                    "its own checksum list"
                )
            elif MANIFEST_NAME in present:
                with archive.open(MANIFEST_NAME) as handle:
                    digest, _ = _sha256_stream(handle)
                recorded = self._sums.get(MANIFEST_NAME)
                if recorded is None:
                    report.sums_problems.append(
                        f"{MANIFEST_NAME} is not listed in {SUMS_NAME}"
                    )
                elif recorded != digest:
                    report.sums_problems.append(
                        f"{MANIFEST_NAME}: SHA256SUMS says {recorded[:12]}…, the file "
                        f"hashes to {digest[:12]}…"
                    )
            stored_sums = set(self._sums)
            for name in sorted(stored_sums - present):
                report.sums_problems.append(
                    f"{SUMS_NAME} lists {name}, which is not in the archive"
                )
        return report

    # -- output ------------------------------------------------------------

    def extract(self, directory: PathLike) -> List[Path]:
        """Write every stored file into `directory`, preserving names."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for item in self.entries:
            destination = target / item.arcname[len(DATA_PREFIX):] if item.arcname.startswith(
                DATA_PREFIX
            ) else target / Path(item.arcname).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(self.read(item.arcname))
            written.append(destination)
        return written

    def info(self) -> Dict[str, Any]:
        """Everything ``odock project info`` shows, as a JSON-able dictionary."""
        run = self.run
        engine = self.engine
        box = self.box()
        entries = [
            {
                "role": item.role,
                "arcname": item.arcname,
                "name": Path(item.arcname).name,
                "size": item.size,
                "sha256": item.sha256,
                "media_type": item.media_type,
                "source": item.source,
                "label": item.label,
            }
            for item in self.entries
        ]
        return {
            "path": self.path.name,
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "format": self.manifest.get("format"),
            "schema_version": self.schema_version,
            "schema_note": self.version_note,
            "min_reader_version": self.manifest.get("min_reader_version"),
            "tool": self.tool,
            "created_utc": self.created,
            "title": self.title,
            "kind": run.get("kind"),
            "n_entries": len(self.entries),
            "entries": entries,
            "inputs": self.inputs,
            "run": run,
            "engine": engine,
            "box": box.as_dict() if box is not None else None,
            "preparation": self.preparation,
            "analyses": {
                key: value
                for key, value in self.analysis().items()
                if key not in ("rows", "interaction_rows")
            },
            "n_pose_rows": len(self.analysis().get("rows") or []),
            "reproducibility": self.reproducibility,
            "campaign": self.campaign,
            "warnings": self.warnings,
            "does_not_establish": self.does_not_establish,
        }

    def describe(self) -> str:
        """A human-readable summary, as printed by ``odock project info``."""
        info = self.info()
        run = info["run"]
        engine = info["engine"]
        lines = [
            f"{info['path']}  ({info['size_bytes'] / 1024.0:.1f} KiB, "
            f"{info['n_entries']} stored file(s))",
            f"  format      : {info['format']} ({info['schema_note']})",
            f"  title       : {info['title']}",
            f"  created     : {info['created_utc']}",
            f"  tool        : opendocking {self.tool.get('version', '?')} "
            f"(kernel {self.tool.get('kernel_version', '?')}, "
            f"{self.tool.get('platform', '?')})",
            f"  run         : {run.get('kind', '?')}, {info['n_pose_rows']} pose(s), "
            f"best {run.get('best_affinity', '?')} kcal/mol",
            f"  scoring     : {engine.get('scoring', '?')}, "
            f"seed {_shown(engine.get('seed'))}",
        ]
        if info["box"]:
            box = info["box"]
            lines.append(
                "  box         : center ("
                + ", ".join(f"{v:.3f}" for v in box["center"])
                + ") size ("
                + ", ".join(f"{v:.3f}" for v in box["size"])
                + f") spacing {box['spacing']:g} A"
            )
        if info["inputs"]:
            lines.append("  inputs      :")
            for item in info["inputs"]:
                lines.append(
                    f"    {item.get('role', '?'):<12} {item.get('name', '?'):<28} "
                    f"{item.get('sha256', '')[:12]}…"
                )
        lines.append("  files       :")
        for item in info["entries"]:
            lines.append(
                f"    {item['role']:<12} {item['name']:<28} "
                f"{item['size']:>9} B  {item['sha256'][:12]}…"
            )
        command = self.reproducibility.get("command")
        if command:
            lines.append(f"  command     : {command}")
        for warning in self.warnings:
            lines.append(f"  warning     : {warning}")
        return "\n".join(lines)


def _read_sums(text: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse a ``SHA256SUMS`` document into ``{member: digest}`` plus its order."""
    found: Dict[str, str] = {}
    order: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        digest, name = parts[0].strip(), parts[1].strip()
        if name.startswith("*"):
            name = name[1:]
        found[name] = digest
        order.append(name)
    return found, order


def open_project(path: PathLike, *, verify: bool = False) -> Project:
    """Open a project file.

    Only the manifest is parsed here; the stored files are read on demand.  With
    `verify=True` every hash is recomputed first and a :class:`ProjectError` is
    raised if anything changed -- which is what ``odock project open`` does by
    default, because a project whose inputs have been edited is not the run it
    claims to be.
    """
    target = Path(path)
    if not target.exists():
        raise ProjectError(f"no such project: {target}")
    if not zipfile.is_zipfile(target):
        raise ProjectError(
            f"{target.name} is not an OpenDocking project: a .odockproj file is a "
            "ZIP archive holding project.json, SHA256SUMS and the run's files"
        )
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        if MANIFEST_NAME not in names:
            raise ProjectError(
                f"{target.name} has no {MANIFEST_NAME}: not an OpenDocking project"
            )
        try:
            payload = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProjectError(f"{MANIFEST_NAME} is not valid JSON: {exc}")
        manifest = load_manifest(payload)
        min_reader = manifest.get("min_reader_version")
        if isinstance(min_reader, int) and min_reader > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"this project needs a reader of at least schema {min_reader}, and "
                f"this build reads up to {SCHEMA_VERSION}: upgrade OpenDocking "
                "rather than opening it with an older one."
            )
        if str(manifest.get("format") or FORMAT_NAME) != FORMAT_NAME:
            raise ProjectError(
                f"{target.name} declares format {manifest.get('format')!r}, not "
                f"{FORMAT_NAME!r}"
            )
        raw_files = manifest.get("files")
        entries = [
            StoredFile.from_dict(item)
            for item in (raw_files if isinstance(raw_files, list) else [])
            if isinstance(item, Mapping)
        ]
        sums: Dict[str, str] = {}
        order: List[str] = []
        if SUMS_NAME in names:
            sums, order = _read_sums(archive.read(SUMS_NAME).decode("utf-8", "replace"))
    project = Project(target, manifest, entries, sums=sums, sums_arcnames=order)
    if verify:
        report = project.verify()
        if not report.ok:
            raise ProjectError(
                "the project did not verify; it is not the run it claims to be:\n"
                + report.summary()
            )
    return project


def verify_project(path: PathLike) -> VerifyResult:
    """Verify a project file, reporting a refusal as a failed report.

    A newer schema is a *refusal*, not a crash: the caller gets a
    :class:`VerifyResult` whose ``problems`` name both versions, which is what
    ``odock project verify`` prints.
    """
    target = Path(path)
    try:
        project = open_project(target)
    except ProjectError as exc:
        return VerifyResult(
            path=target,
            schema_version=0,
            entries=0,
            checked=0,
            size_bytes=target.stat().st_size if target.exists() else 0,
            problems=[str(exc)],
        )
    return project.verify()


def project_info(path: PathLike) -> Dict[str, Any]:
    """The machine-readable ``info`` view of a project."""
    return open_project(path).info()


def extract_project(path: PathLike, directory: PathLike) -> List[Path]:
    """Write every stored file of a project into `directory`."""
    return open_project(path).extract(directory)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _prepare_report_dict(report: Any) -> Dict[str, Any]:
    """A :class:`odock.prepare.PreparationReport` (or mapping) as a dictionary."""
    if report is None:
        return {}
    if isinstance(report, Mapping):
        return normalise_paths(dict(report))
    fields = (
        "kind", "n_atoms_in", "n_atoms_out", "n_hydrogens_added",
        "n_nonpolar_hydrogens_removed", "n_rotatable_bonds", "n_metal_atoms",
        "reembedded", "thickness", "chemistry_trusted", "warnings",
    )
    found: Dict[str, Any] = {}
    for name in fields:
        if hasattr(report, name):
            value = getattr(report, name)
            found[name] = list(value) if isinstance(value, (list, tuple)) and name == "warnings" else value
    if hasattr(report, "atom_order"):
        found["atom_order"] = [int(v) for v in getattr(report, "atom_order") or []]
    return normalise_paths(found)


def analyse_result(
    result: Any,
    *,
    receptor: Any = None,
    ligand: Any = None,
    strain: bool = False,
    top: int = 3,
) -> Dict[str, Any]:
    """The analysis a project stores: the ranking table, the contacts, the limits.

    Everything here is computed from the run and its stored inputs, so it can be
    *recomputed* on reopen and compared value for value.  The interaction profile
    is one pose deep (the best one) because that is what a report shows and what
    a project can afford to keep exact; the per-pose residue summary is in the
    ranking table.
    """
    from . import analysis as _analysis
    from . import report as _report

    rows: List[Dict[str, Any]] = []
    flags: List[str] = []
    receptor_atoms = None
    receptor_text = receptor if isinstance(receptor, str) else None
    if receptor_text:
        atoms = _report.parse_pdbqt_atoms(receptor_text)
        receptor_atoms = atoms or None
    try:
        rows = _report.result_rows(
            result,
            receptor=receptor_atoms or receptor,
            box=getattr(result, "box", None),
            strain=strain,
        )
    except Exception as exc:  # pragma: no cover - a report must not lose a run
        flags.append(f"the ranking table could not be computed: {exc}")

    best_profile: Dict[str, Any] = {}
    pharmacophore: Dict[str, Any] = {}
    fingerprints_error = ""
    ligand_text = ligand if isinstance(ligand, str) else getattr(result, "ligand_pdbqt", "")
    # The best pose's own coordinates are what the profile is built from; the
    # ligand PDBQT is only the atom-order template, so a run whose ligand file
    # was not supplied can still be profiled from the pose document.
    has_poses = bool(getattr(result, "poses", None))
    try:
        if receptor_atoms and (ligand_text.strip() or has_poses):
            rec = _analysis._structure(receptor_atoms)
            if rows:
                profiles = _profile_pose(result, int(rows[0].get("mode") or 1), receptor_atoms)
                if profiles is not None:
                    best_profile = profiles
            ligands = _ligands_for_poses(result)
            if rec.atoms and ligands:
                fingerprints = _analysis.pose_fingerprints(rec, ligands)
                summary = fingerprints.pharmacophore(
                    top=max(1, int(top)) if len(fingerprints) > int(top) else None,
                    min_frequency=0.5,
                )
                pharmacophore = {
                    "n_poses": int(summary.n_poses),
                    "keys": [
                        {
                            "label": key.label,
                            "kind": key.kind,
                            "res_name": key.res_name,
                            "res_id": key.res_id,
                        }
                        for key in summary.keys
                    ],
                    "counts": [int(v) for v in summary.counts],
                    "frequency": [float(v) for v in summary.frequency],
                    "n_features": len(fingerprints.labels()),
                }
    except Exception as exc:
        fingerprints_error = f"{type(exc).__name__}: {exc}"
        flags.append(f"the interaction profile could not be computed: {exc}")

    quality = _pose_quality(result, rows)
    for note in quality.get("flags", []):
        if note not in flags:
            flags.append(note)
    if strain:
        flags.append(
            "strain was requested; a per-pose force-field relaxation was run and "
            "is recorded per row"
        )
    if not receptor:
        flags.append(
            "no receptor was supplied, so the residue column and the interaction "
            "profile are empty; store the receptor PDBQT to make them available"
        )
    return {
        "rows": rows,
        "n_poses": len(rows),
        "best": {
            "mode": 1 if rows else None,
            "affinity": rows[0].get("affinity") if rows else None,
            "residues": rows[0].get("residues") if rows else "",
        },
        "interaction_profile": best_profile,
        "pharmacophore": pharmacophore,
        "pose_quality": quality,
        "flags": flags,
        "fingerprints_error": fingerprints_error,
    }


def _ligands_for_poses(result: Any) -> List[Any]:
    """The receptor-order ligand of every pose, as atom sequences."""
    from .report import _first_model, _ligand_atoms

    ligands = []
    for pose in getattr(result, "poses", None) or []:
        try:
            atoms = _ligand_atoms(result, pose)
        except Exception:
            atoms = []
        if atoms:
            ligands.append(atoms)
    return ligands


def _profile_pose(result: Any, mode: int, receptor_atoms: Optional[Sequence[Any]] = None) -> Optional[Dict[str, Any]]:
    """The non-covalent interactions of one pose, with readable atom labels."""
    from . import analysis as _analysis
    from .report import parse_pdbqt_atoms

    poses = list(getattr(result, "poses", None) or [])
    if not 1 <= mode <= len(poses):
        return None
    if receptor_atoms is None:
        receptor_atoms = parse_pdbqt_atoms(getattr(result, "receptor_pdbqt", "") or "")
    pose = poses[mode - 1]
    ligand_atoms = _ligand_pose_atoms(result, pose)
    if not receptor_atoms or not ligand_atoms:
        return None
    interactions = _analysis.profile_interactions(receptor_atoms, ligand_atoms)
    counts: Dict[str, int] = {}
    detail = []
    for item in interactions:
        kind = str(getattr(item, "kind", "?"))
        counts[kind] = counts.get(kind, 0) + 1
        detail.append(
            {
                "kind": kind,
                "receptor_atom": int(getattr(item, "a", -1)),
                "ligand_atom": int(getattr(item, "b", -1)),
                "receptor_label": _atom_label(receptor_atoms, getattr(item, "a", -1)),
                "ligand_label": _atom_label(ligand_atoms, getattr(item, "b", -1)),
                "distance": float(getattr(item, "distance", float("nan"))),
                "detail": str(getattr(item, "detail", "") or ""),
                "subtype": str(getattr(item, "subtype", "") or ""),
            }
        )
    return {
        "mode": int(mode),
        "counts": counts,
        "key_residues": _analysis.interaction_summary(
            interactions, receptor_atoms, ligand_atoms
        ),
        "rows": detail,
    }


def _ligand_pose_atoms(result: Any, pose: Any) -> List[Any]:
    """The pose's ligand atoms, coordinates taken from the pose itself."""
    from .report import _ligand_atoms

    try:
        return _ligand_atoms(result, pose)
    except Exception:  # pragma: no cover - defensive
        return []


def _atom_label(atoms: Sequence[Any], index: int) -> str:
    """``ASP189:OD1`` for one atom of a parsed structure."""
    if not 0 <= int(index) < len(atoms):
        return f"atom {index}"
    atom = atoms[int(index)]
    return f"{atom.res_name}{atom.res_id}:{atom.name}"


def _pose_quality(result: Any, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pose-quality numbers and the flags a reader should know about."""
    import math

    poses = list(getattr(result, "poses", None) or [])
    affinities = [float(getattr(pose, "affinity", 0.0) or 0.0) for pose in poses]
    spread = (max(affinities) - min(affinities)) if affinities else 0.0
    best = min(affinities) if affinities else None
    within_1 = sum(1 for value in affinities if best is not None and value <= best + 1.0)
    heavy = rows[0].get("heavy_atoms") if rows else None
    flags: List[str] = []
    if poses and any(getattr(pose, "in_box", True) is False for pose in poses):
        flags.append("at least one pose lies outside the search box")
    if len(poses) == 1:
        flags.append(
            "only one pose was reported, so nothing here can say whether a second "
            "binding mode exists"
        )
    if len(affinities) >= 3 and spread <= 0.1:
        flags.append(
            f"the reported poses span only {spread:.3f} kcal/mol, which is inside "
            "the force field's own noise: treat their order as arbitrary"
        )
    if within_1 and len(poses) and within_1 == len(poses) and len(poses) > 1:
        flags.append(
            f"every pose is within 1 kcal/mol of the best ({within_1} of "
            f"{len(poses)}), so the ranking carries little information"
        )
    return {
        "n_poses": len(poses),
        "best_affinity": best,
        "worst_affinity": max(affinities) if affinities else None,
        "spread": spread,
        "n_within_1_kcal": within_1,
        "heavy_atoms": heavy,
        "ligand_efficiency": rows[0].get("ligand_efficiency") if rows else None,
        "mean_affinity": (
            float(sum(affinities) / len(affinities)) if affinities else None
        ),
        "affinities_finite": all(math.isfinite(value) for value in affinities),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def _pose_record(pose: Any) -> Dict[str, Any]:
    """One pose, JSON-safe and full precision (so a round trip is exact)."""
    coords = getattr(pose, "coords", None)
    record: Dict[str, Any] = {
        "index": int(getattr(pose, "index", 0)),
        "affinity": float(getattr(pose, "affinity", 0.0) or 0.0),
        "rmsd_lower_bound": float(getattr(pose, "rmsd_lower_bound", 0.0) or 0.0),
        "rmsd_upper_bound": float(getattr(pose, "rmsd_upper_bound", 0.0) or 0.0),
        "inter": float(getattr(pose, "inter", 0.0) or 0.0),
        "intra": float(getattr(pose, "intra", 0.0) or 0.0),
        "conf_independent": float(getattr(pose, "conf_independent", 0.0) or 0.0),
        "unbound": float(getattr(pose, "unbound", 0.0) or 0.0),
        "in_box": bool(getattr(pose, "in_box", True)),
        "num_atoms": int(getattr(pose, "num_atoms", 0) or 0),
    }
    position = getattr(pose, "position", None)
    if position is not None:
        record["position"] = [float(v) for v in position]
    orientation = getattr(pose, "orientation", None)
    if orientation is not None:
        record["orientation"] = [float(v) for v in orientation]
    torsions = getattr(pose, "torsions", None)
    if torsions is not None:
        record["torsions"] = [float(v) for v in torsions]
    if coords is not None:
        record["coords"] = [[float(v) for v in row] for row in coords]
    return record


def _data_arcname(role: str, name: str, *, subdir: str = "") -> str:
    """The member name for a stored file: always inside ``files/``."""
    stem = Path(str(name)).name or role
    if subdir:
        return f"{DATA_PREFIX}{subdir.strip('/')}/{stem}"
    return f"{DATA_PREFIX}{stem}"


@dataclass
class _Member:
    """A file on its way into the archive."""

    role: str
    arcname: str
    data: bytes
    media_type: str
    label: str
    source: str

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.data)

    def entry(self) -> StoredFile:
        return StoredFile(
            role=self.role,
            arcname=self.arcname,
            sha256=self.sha256,
            size=len(self.data),
            media_type=self.media_type,
            label=self.label,
            source=self.source,
        )


def _member(role: str, arcname: str, data: bytes, *, label: str = "",
            source: str = "") -> _Member:
    return _Member(
        role=role,
        arcname=arcname,
        data=data,
        media_type=media_type_for(arcname),
        label=label or Path(arcname).name,
        source=source or Path(arcname).name,
    )


def save_project(
    path: PathLike,
    result: Any = None,
    *,
    receptor: Any = None,
    ligand: Any = None,
    box: Any = None,
    poses: Any = None,
    original_inputs: Optional[Mapping[str, Any]] = None,
    extra_files: Optional[Mapping[str, Any]] = None,
    engine: Optional[Mapping[str, Any]] = None,
    preparation: Optional[Mapping[str, Any]] = None,
    analyses: Optional[Mapping[str, Any]] = None,
    figures: Optional[Mapping[str, Any]] = None,
    campaign: Optional[Mapping[str, Any]] = None,
    command: Optional[Union[str, Sequence[str]]] = None,
    seed: Optional[int] = None,
    scoring: Optional[str] = None,
    title: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    created: Optional[str] = None,
    strain: bool = False,
    verify: bool = True,
) -> Project:
    """Write a docking run as a self-contained, verifiable ``.odockproj``.

    Parameters
    ----------
    path
        The file to write.  A missing ``.odockproj`` suffix is added.
    result
        The run: a :class:`odock.DockResult`, the text of a multi-model pose
        PDBQT, or a path to one.  A pose file is reconstructed with
        :func:`result_from_pdbqt`, which is how a run that exists only as files
        (a screening shortlist, a bundled demo) becomes a project.
    receptor, ligand
        The prepared PDBQT of each partner (a path, the text, or bytes).  Stored
        verbatim; the manifest records their SHA-256.
    box
        The search box: a :class:`odock.BoxSpec`, a mapping, or a box JSON path.
        Required for a pose-only save, because the box is part of what a run is.
    poses
        The pose PDBQT, when `result` is not the pose source.
    original_inputs
        ``{"receptor": "3PTB.pdb", "ligand": "BEN.sdf"}``: the **raw** input
        structures, stored alongside the PDBQT they were prepared into.
    extra_files
        ``{"name.ext": path_or_text}``: anything else that belongs to the run
        (a log, a CSV), stored under ``files/``.
    engine
        Engine settings to record, merged over the values derived from `result`
        (``exhaustiveness``, ``search``, ``min_rmsd``, ...).  A setting that is
        not recorded is written as ``null``; it is never guessed.
    preparation
        ``{"receptor": report_or_dict, "ligand": report_or_dict}`` from
        :func:`odock.prepare_receptor` / :func:`odock.prepare_ligand`.
    analyses
        A pre-computed analysis (see :func:`analyse_result`) to store instead of
        recomputing; individual keys override the computed ones.
    figures
        ``{"ranking": "<svg …>", "diagram": b"<png …>"}``: figures stored with the
        run so the HTML report can be regenerated without recomputing them.
    campaign, command, seed, scoring, title, notes, created
        Provenance: which screening campaign a member came from, the exact
        command, an override for the seed or the force field, a title, extra
        notes, and a fixed creation time (which makes a save reproducible).
    strain
        Also compute the opt-in per-pose ligand strain.
    verify
        Re-hash the archive after writing it and raise if anything mismatches.
        On by default: a project that does not verify is worse than no project.
    """
    from .docking import DockResult

    target = Path(path)
    if target.suffix != PROJECT_SUFFIX:
        target = target.with_suffix(PROJECT_SUFFIX)
    base = target.parent if str(target.parent) else Path(".")

    members: List[_Member] = []
    arcnames: Dict[str, str] = {}

    def add(member: _Member) -> None:
        if member.arcname in arcnames:
            stem = Path(member.arcname).stem
            suffix = Path(member.arcname).suffix
            index = 2
            while f"{DATA_PREFIX}{stem}_{index}{suffix}" in arcnames:
                index += 1
            member.arcname = f"{DATA_PREFIX}{stem}_{index}{suffix}"
        arcnames[member.arcname] = member.role
        members.append(member)

    # -- inputs ------------------------------------------------------------
    inputs_block: List[Dict[str, Any]] = []
    receptor_text = ""
    ligand_text = ""
    if receptor is not None:
        data, name = _as_text_or_bytes(receptor, what="receptor")
        arcname = _data_arcname(ROLE_RECEPTOR, name)
        display = portable_path(
            receptor if isinstance(receptor, (str, os.PathLike)) else name, base=base
        )
        add(_member(ROLE_RECEPTOR, arcname, data, label=f"prepared receptor ({name})",
                    source=display))
        receptor_text = data.decode("utf-8", errors="replace")
        inputs_block.append(
            {
                "role": ROLE_RECEPTOR,
                "name": Path(arcname).name,
                "arcname": arcname,
                "sha256": sha256_bytes(data),
                "size": len(data),
                "media_type": media_type_for(arcname),
                "source": display,
            }
        )
    if ligand is not None:
        data, name = _as_text_or_bytes(ligand, what="ligand")
        arcname = _data_arcname(ROLE_LIGAND, name)
        display = portable_path(
            ligand if isinstance(ligand, (str, os.PathLike)) else name, base=base
        )
        add(_member(ROLE_LIGAND, arcname, data, label=f"prepared ligand ({name})",
                    source=display))
        ligand_text = data.decode("utf-8", errors="replace")
        inputs_block.append(
            {
                "role": ROLE_LIGAND,
                "name": Path(arcname).name,
                "arcname": arcname,
                "sha256": sha256_bytes(data),
                "size": len(data),
                "media_type": media_type_for(arcname),
                "source": display,
            }
        )
    for role, source in (original_inputs or {}).items():
        data, name = _as_text_or_bytes(source, what=f"{role} input")
        arcname = _data_arcname(f"{ROLE_INPUT}:{role}", name, subdir="inputs")
        display = portable_path(
            source if isinstance(source, (str, os.PathLike)) else name, base=base
        )
        add(
            _member(
                f"{ROLE_INPUT}:{role}",
                arcname,
                data,
                label=f"input {role} ({name})",
                source=display,
            )
        )
        inputs_block.append(
            {
                "role": f"{ROLE_INPUT}:{role}",
                "name": Path(arcname).name,
                "arcname": arcname,
                "sha256": sha256_bytes(data),
                "size": len(data),
                "media_type": media_type_for(arcname),
                "source": display,
            }
        )

    # -- the run -----------------------------------------------------------
    box_spec = _coerce_box(box)
    poses_text = ""
    poses_display = "poses.pdbqt"
    if poses is not None:
        data, name = _as_text_or_bytes(poses, what="pose file")
        poses_text = data.decode("utf-8", errors="replace")
        poses_display = name
        add(_member(ROLE_POSES, _data_arcname(ROLE_POSES, name), data,
                    label=f"pose file ({name})",
                    source=portable_path(
                        poses if isinstance(poses, (str, os.PathLike)) else name, base=base
                    )))

    if result is None and poses is None:
        raise ProjectError(
            "nothing to save: pass a DockResult, a pose PDBQT (poses=...), or "
            "a path to one"
        )
    from_engine_result = isinstance(result, DockResult)
    if not from_engine_result:
        source = result if result is not None else poses
        result = result_from_pdbqt(
            source,
            box=box_spec,
            receptor=receptor_text or None,
            ligand=ligand_text or None,
            seed=None if seed is None else int(seed),
            scoring=str(scoring or "vina"),
        )
        if isinstance(source, (str, os.PathLike)) and "\n" not in str(source):
            poses_display = Path(str(source)).name
        if not poses_text:
            poses_text = result.to_pdbqt() or ""
    if not isinstance(result, DockResult):  # pragma: no cover - defensive
        raise ProjectError(f"cannot save a {type(result).__name__} as a project")

    if box_spec is None:
        box_spec = getattr(result, "box", None)
    if box_spec is None:
        raise ProjectError(
            "a project must record the search box: pass box=<BoxSpec|mapping|box.json>"
        )
    box_data = (json.dumps(box_spec.as_dict(), indent=2) + "\n").encode("utf-8")
    add(_member(ROLE_BOX, _data_arcname(ROLE_BOX, "box.json"), box_data,
                label="search box"))

    # The pose document is the run's primary artefact.  A DockResult carries it
    # in `_pdbqt` (that is exactly the text the kernel wrote); a result built by
    # hand may carry none, in which case the archive simply has no poses member
    # and the coordinates still come from `files/poses.json`.
    if not poses_text:
        poses_text = getattr(result, "_pdbqt", None) or ""
    if poses_text and not any(item.role == ROLE_POSES for item in members):
        add(_member(ROLE_POSES, _data_arcname(ROLE_POSES, poses_display),
                    poses_text.encode("utf-8"), label=f"pose file ({poses_display})"))

    if not receptor_text:
        receptor_text = getattr(result, "receptor_pdbqt", "") or ""
    if not ligand_text:
        ligand_text = getattr(result, "ligand_pdbqt", "") or ""

    # A result from `odock.dock` carries the exact receptor and ligand text it
    # ran on.  Storing them is what makes the project self-contained: without it
    # the interaction profile could never be recomputed on reopen.  They are
    # added *before* the analysis below so the analysis is computed from what a
    # reader of the project will actually see.
    carried: List[str] = []
    if receptor_text and not any(item.role == ROLE_RECEPTOR for item in members):
        data = receptor_text.encode("utf-8")
        arcname = _data_arcname(ROLE_RECEPTOR, "receptor.pdbqt")
        add(_member(ROLE_RECEPTOR, arcname, data,
                    label="prepared receptor (carried by the result object)"))
        inputs_block.append(
            {
                "role": ROLE_RECEPTOR,
                "name": Path(arcname).name,
                "arcname": arcname,
                "sha256": sha256_bytes(data),
                "size": len(data),
                "media_type": media_type_for(arcname),
                "source": "carried by the result object",
            }
        )
        carried.append("receptor")
    if ligand_text and not any(item.role == ROLE_LIGAND for item in members):
        data = ligand_text.encode("utf-8")
        arcname = _data_arcname(ROLE_LIGAND, "ligand.pdbqt")
        add(_member(ROLE_LIGAND, arcname, data,
                    label="prepared ligand (carried by the result object)"))
        inputs_block.append(
            {
                "role": ROLE_LIGAND,
                "name": Path(arcname).name,
                "arcname": arcname,
                "sha256": sha256_bytes(data),
                "size": len(data),
                "media_type": media_type_for(arcname),
                "source": "carried by the result object",
            }
        )
        carried.append("ligand")

    # Attach the stored inputs to the result so the analysis below sees exactly
    # what a reader of the project will see.
    if receptor_text:
        result.receptor_pdbqt = receptor_text
    if ligand_text:
        result.ligand_pdbqt = ligand_text

    pose_records = [_pose_record(pose) for pose in getattr(result, "poses", None) or []]
    pose_payload = {
        "n_poses": len(pose_records),
        "ligand_atom_order": [str(name) for name in getattr(result, "ligand_atom_order", ()) or ()],
        "poses": pose_records,
    }
    add(
        _member(
            ROLE_POSE_DATA,
            _data_arcname(ROLE_POSE_DATA, "poses.json"),
            (json.dumps(pose_payload, ensure_ascii=False) + "\n").encode("utf-8"),
            label="poses with full-precision coordinates",
        )
    )

    # -- analysis ----------------------------------------------------------
    computed = analyse_result(
        result,
        receptor=receptor_text or None,
        ligand=ligand_text or None,
        strain=strain,
    )
    if analyses:
        merged = dict(computed)
        merged.update(dict(analyses))
        computed = merged
    add(
        _member(
            ROLE_ANALYSIS,
            _data_arcname(ROLE_ANALYSIS, "analysis.json"),
            (json.dumps(computed, ensure_ascii=False, default=str) + "\n").encode("utf-8"),
            label="analysis results",
        )
    )

    preparation_block = {
        str(role): _prepare_report_dict(report)
        for role, report in (preparation or {}).items()
    }
    if preparation_block:
        add(
            _member(
                ROLE_PREPARATION,
                _data_arcname(ROLE_PREPARATION, "preparation.json"),
                (json.dumps(preparation_block, indent=2, default=str) + "\n").encode("utf-8"),
                label="preparation settings and outcomes",
            )
        )

    for name, payload in (figures or {}).items():
        data = payload if isinstance(payload, bytes) else str(payload).encode("utf-8")
        suffix = Path(str(name)).suffix or (".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".svg")
        member_name = str(name) if Path(str(name)).suffix else f"{name}{suffix}"
        add(
            _member(
                ROLE_FIGURE,
                _data_arcname(ROLE_FIGURE, member_name, subdir="figures"),
                data,
                label=f"figure {Path(member_name).stem}",
            )
        )
    for name, source in (extra_files or {}).items():
        data, resolved = _as_text_or_bytes(source, what=f"extra file {name}")
        add(
            _member(
                ROLE_LOG if Path(str(name)).suffix.lower() in (".log", ".txt") else "extra",
                _data_arcname("extra", str(name), subdir="extra"),
                data,
                label=f"extra file ({Path(resolved).name})",
                source=portable_path(
                    source if isinstance(source, (str, os.PathLike)) else resolved, base=base
                ),
            )
        )

    # -- the manifest ------------------------------------------------------
    best = None
    poses_list = list(getattr(result, "poses", None) or [])
    if poses_list:
        best = min(float(getattr(pose, "affinity", 0.0) or 0.0) for pose in poses_list)
    engine_block: Dict[str, Any] = {
        "scoring": str(scoring or getattr(result, "scoring", "vina") or "vina"),
        # A seed that nobody recorded stays unrecorded: the kernel's 0 means "draw
        # one at random", so writing 0 for an unknown seed would be a claim.
        "seed": (
            int(seed)
            if seed is not None
            else (
                int(getattr(result, "seed", 0) or 0)
                if getattr(result, "seed_recorded", True)
                else None
            )
        ),
        "exhaustiveness": None,
        "num_poses": len(poses_list),
        "search": None,
        "islands": None,
        "population": None,
        "generations": None,
        "min_rmsd": None,
        "energy_range": None,
        "use_grid": None,
        "refine": None,
        "use_island_ga": None,
    }
    for key, value in (engine or {}).items():
        if value is None:
            continue
        engine_block[str(key)] = value
    engine_block["scoring"] = str(engine_block.get("scoring") or "vina")
    if engine_block.get("seed") is not None:
        engine_block["seed"] = int(engine_block["seed"])
    engine_block["num_poses"] = int(engine_block.get("num_poses") or len(poses_list))

    run_block = {
        "kind": str(
            (campaign or {}).get("kind")
            or ("dock" if from_engine_result else "poses")
        ),
        "seed": engine_block["seed"],
        "scoring": engine_block["scoring"],
        "n_poses": len(poses_list),
        "best_affinity": best,
        "elapsed_s": float(getattr(result, "elapsed", 0.0) or 0.0),
        "grid_points": int(getattr(result, "grid_points", 0) or 0),
        "grid_mb": int(getattr(result, "grid_mb", 0) or 0),
        "num_tors": float(getattr(result, "num_tors", 0.0) or 0.0),
        "num_movable_atoms": int(getattr(result, "num_movable_atoms", 0) or 0),
        "num_dof": int(getattr(result, "num_dof", 0) or 0),
        "exact": bool(getattr(result, "exact", True)),
        "cancelled": bool(getattr(result, "cancelled", False)),
        "ligand_atom_order": [
            str(name) for name in getattr(result, "ligand_atom_order", ()) or ()
        ],
    }

    command_text = ""
    if command is not None:
        command_text = portable_command(command, base=base)

    warnings: List[str] = []
    extra_notes: List[str] = []
    if carried:
        extra_notes.append(
            "the "
            + " and ".join(carried)
            + " PDBQT was taken from the run object rather than from a file "
            "argument; it was stored so the project stays self-contained"
        )
    if not any(item.role == ROLE_RECEPTOR for item in members):
        warnings.append(
            "no receptor PDBQT is stored: the residue column and the interaction "
            "profile cannot be recomputed from this project"
        )
    for item in members:
        if not item.arcname.endswith((".pdbqt", ".pdb", ".sdf", ".mol", ".mol2")):
            continue
        leaked = find_absolute_paths(item.data.decode("utf-8", errors="replace"))
        if leaked:
            warnings.append(
                f"{item.arcname} contains {len(leaked)} absolute path(s) inside the "
                "structure data; the file was stored unchanged, so the paths were "
                "not normalised (they are your input's data, not this tool's "
                "metadata)"
            )

    manifest: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "min_reader_version": SCHEMA_VERSION,
        "created_utc": str(created or _now()),
        "title": str(title or target.stem),
        "tool": {
            "name": "opendocking",
            "version": tool_version(),
            "distribution_version": _distribution_version(),
            "version_source": (
                "distribution metadata" if _distribution_version() else
                "the importable package (no distribution metadata is installed)"
            ),
            "kernel_version": _kernel_version(),
            "python": platform.python_version(),
            "platform": _platform_label(),
        },
        "run": run_block,
        "engine": engine_block,
        "box": box_spec.as_dict(),
        "inputs": inputs_block,
        "preparation": preparation_block,
        "analyses": {
            "n_poses": computed.get("n_poses", 0),
            "best": computed.get("best", {}),
            "pose_quality": computed.get("pose_quality", {}),
            "pharmacophore": computed.get("pharmacophore", {}),
            "flags": computed.get("flags", []),
            "stored_in": _data_arcname(ROLE_ANALYSIS, "analysis.json"),
        },
        "files": [item.entry().as_dict() for item in members],
        "reproducibility": {
            "command": command_text,
            "seed": run_block["seed"],
            "scoring": run_block["scoring"],
            "input_sha256": {
                item["role"]: item["sha256"] for item in inputs_block
            },
            "platform": _platform_label(),
            "python": platform.python_version(),
            "tool_version": tool_version(),
            "kernel_version": _kernel_version(),
            "note": (
                "The same inputs (by SHA-256), the same tool version and the same "
                "seed reproduce the same poses; a different kernel version may "
                "legitimately produce different ones."
            ),
        },
        "does_not_establish": list(RUN_LIMITS),
        "notes": [str(note) for note in (notes or ())] + extra_notes,
        "warnings": list(warnings),
    }
    if campaign:
        manifest["campaign"] = normalise_paths(dict(campaign), base=base)

    # Every string in the manifest passes through the normaliser here, so a
    # forgotten call site cannot leak a path.
    manifest = normalise_paths(manifest, base=base)

    # -- write the archive -------------------------------------------------
    project_json = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    sums_lines = [f"{sha256_bytes(project_json)}  {MANIFEST_NAME}"]
    for item in members:
        sums_lines.append(f"{item.sha256}  {item.arcname}")
    sums = ("\n".join(sums_lines) + "\n").encode("utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(members, key=lambda item: item.arcname)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        _write_zip_member(archive, MANIFEST_NAME, project_json)
        _write_zip_member(archive, SUMS_NAME, sums)
        for item in ordered:
            _write_zip_member(archive, item.arcname, item.data)

    project = open_project(target)
    if verify:
        report = project.verify()
        if not report.ok:
            raise ProjectError(
                "the project was written but does not verify; this is a bug:\n"
                + report.summary()
            )
    return project


def _write_zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    """Add a member with a fixed timestamp, so the same content is the same bytes."""
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


# ---------------------------------------------------------------------------
# Wrapping a screening campaign
# ---------------------------------------------------------------------------


def _screen_module():
    """The :mod:`odock.screen` module, imported for its layout, never edited here."""
    from . import screen

    return screen


def save_screen_projects(
    screen_dir: PathLike,
    out: PathLike,
    *,
    top: int = 0,
    receptor: Optional[str] = None,
    only_ok: bool = True,
    figures: bool = False,
) -> List[Project]:
    """Turn the docked molecules of a screening campaign into projects.

    A screening run writes one row per molecule × receptor, one pose file per
    molecule and one prepared PDBQT per molecule; none of them is a *run* a
    colleague can reopen.  This reads the campaign's ``run.json`` and
    ``results.jsonl`` (as :mod:`odock.screen` wrote them -- the module is
    imported, not modified), and writes one ``.odockproj`` per molecule into
    `out`, each holding that molecule's poses, the prepared ligand, the receptor
    and the campaign's engine settings.

    `top` keeps only the N best rows by affinity (``0`` keeps every row);
    `receptor` selects one receptor of a panel, and `only_ok` skips rows whose
    status is not ``ok``.
    """
    from .docking import DockResult  # noqa: F401  (documented return payload)

    screen = _screen_module()
    directory = Path(screen_dir)
    manifest_path = directory / screen.MANIFEST_NAME
    if not manifest_path.exists():
        raise ProjectError(
            f"{directory} is not a screening output directory: it has no "
            f"{screen.MANIFEST_NAME}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    results_path = directory / str(manifest.get("results") or "results.jsonl")
    if not results_path.exists():
        raise ProjectError(f"the campaign's results file is missing: {results_path}")

    records: List[Dict[str, Any]] = []
    text = results_path.read_text(encoding="utf-8", errors="replace")
    if results_path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, Mapping):
                records.append(dict(payload))
    else:
        import csv as _csv

        for row in _csv.DictReader(text.splitlines()):
            records.append(dict(row))
    if not records:
        raise ProjectError(f"no row was found in {results_path}")
    if only_ok:
        records = [row for row in records if str(row.get("status") or "ok") == "ok"]
    if receptor:
        records = [row for row in records if str(row.get("receptor") or "") == receptor]
    if top and top > 0:
        records = sorted(
            records,
            key=lambda row: float(row.get("affinity") or float("inf")),
        )[: int(top)]

    receptors_by_name = {
        str(item.get("name") or Path(str(item.get("path") or "")).stem): Path(
            str(item.get("path"))
        )
        for item in (manifest.get("receptors") or [])
        if isinstance(item, Mapping)
    }
    box = manifest.get("box")
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    written: List[Project] = []
    for position, row in enumerate(records):
        pose_rel = str(row.get("pose_file") or "")
        pose_path = directory / pose_rel if pose_rel else None
        if pose_path is None or not pose_path.exists():
            continue
        receptor_name = str(row.get("receptor") or "")
        receptor_path = receptors_by_name.get(receptor_name)
        index = int(float(row.get("index") or 0))
        prepared = sorted((directory / screen.LIBRARY_DIR).glob(f"{index + 1:06d}_*.pdbqt"))
        ligand_path = prepared[0] if prepared else None
        label = str(row.get("name") or row.get("ligand") or pose_path.stem)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or "member"
        target = output / f"{safe}{PROJECT_SUFFIX}"
        engine = {
            "scoring": config.get("scoring"),
            "exhaustiveness": config.get("exhaustiveness"),
            "num_poses": config.get("num_poses"),
            "search": config.get("search"),
            "islands": config.get("islands"),
            "population": config.get("population"),
            "generations": config.get("generations"),
            "use_grid": config.get("use_grid"),
            "refine": config.get("refine"),
            "min_rmsd": config.get("min_rmsd"),
            "energy_range": config.get("energy_range"),
            "seed": row.get("seed"),
        }
        campaign = {
            "kind": "screen-member",
            "directory": portable_path(directory, base=output),
            "library_hash": manifest.get("library_hash"),
            "docking_hash": manifest.get("docking_hash"),
            "member": label,
            "member_index": index,
            "receptor": receptor_name,
            "n_library": manifest.get("n_library"),
            "n_dockable": manifest.get("n_dockable"),
            "created": manifest.get("created"),
            "command": (
                "odock screen " + " ".join(
                    f"--receptor {portable_path(path, base=output)}"
                    for path in (config.get("receptors") or [])
                )
            ).strip(),
        }
        project = save_project(
            target,
            poses=pose_path,
            receptor=receptor_path if receptor_path and receptor_path.exists() else None,
            ligand=ligand_path,
            box=box,
            engine=engine,
            campaign=campaign,
            command=campaign["command"],
            title=f"{label} ({receptor_name})",
            notes=[
                f"screening campaign member {position + 1} of {len(records)}; "
                "the campaign settings are recorded under `campaign`",
            ],
        )
        written.append(project)
    if not written:
        raise ProjectError(
            "no screened molecule could be wrapped: no row had a pose file on disk"
        )
    return written


# ---------------------------------------------------------------------------
# Reproducing a run
# ---------------------------------------------------------------------------

#: The engine settings that decide the *result*, with the kernel default used when
#: a project did not record one.  A reproduction that had to fall back to a
#: default cannot be conclusive when it disagrees: the difference may be the
#: missing setting rather than a real non-reproducibility.
REPRODUCTION_KEYS: Tuple[str, ...] = (
    "scoring", "seed", "exhaustiveness", "num_poses", "min_rmsd", "energy_range",
    "search", "islands", "population", "generations", "use_grid", "refine",
)

ENGINE_DEFAULTS: Dict[str, Any] = {
    "scoring": "vina",
    "seed": 0,
    "exhaustiveness": 8,
    "num_poses": 9,
    "min_rmsd": 1.0,
    "energy_range": 3.0,
    "search": None,
    "islands": 4,
    "population": 32,
    "generations": 20,
    "use_grid": True,
    "refine": True,
}


@dataclass
class ReproduceResult:
    """The outcome of re-running a project's docking and comparing it.

    ``verdict`` is one of:

    * ``PASS`` -- the re-run matched the stored run within the tolerances;
    * ``FAIL`` -- it did not, and every setting that decides the result was
      recorded, so the difference is a finding;
    * ``INCONCLUSIVE`` -- it did not match, but the project never recorded a
      setting that could explain it (or an input is missing), so the comparison
      cannot tell reproducibility from an incomplete record.
    """

    path: Path
    verdict: str
    tolerance: float
    rmsd_tolerance: float
    stored: Dict[str, Any] = field(default_factory=dict)
    reproduced: Dict[str, Any] = field(default_factory=dict)
    deltas: Dict[str, Any] = field(default_factory=dict)
    settings_used: Dict[str, Any] = field(default_factory=dict)
    settings_defaulted: List[str] = field(default_factory=list)
    stored_verified: bool = True
    verify_problems: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    seconds: float = 0.0
    recorded_to: Optional[Path] = None
    saved_as: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.verdict == "PASS"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": "odock-reproduction",
            "version": 1,
            "project": self.path.name,
            "verdict": self.verdict,
            "ok": bool(self.ok),
            "tolerance_kcal_per_mol": self.tolerance,
            "tolerance_rmsd_angstrom": self.rmsd_tolerance,
            "seconds": round(float(self.seconds), 3),
            "stored": self.stored,
            "reproduced": self.reproduced,
            "deltas": self.deltas,
            "settings_used": self.settings_used,
            "settings_not_recorded": list(self.settings_defaulted),
            "stored_project_verified": bool(self.stored_verified),
            "stored_project_problems": list(self.verify_problems),
            "problems": list(self.problems),
            "reproduced_project": (
                portable_path(self.saved_as) if self.saved_as is not None else None
            ),
            "note": (
                "A reproduction re-runs the same code on the same stored inputs, "
                "box, settings and seed.  A PASS is reproducibility, not "
                "correctness: it says nothing about whether the force field, the "
                "preparation or the pose is right."
            ),
        }

    def summary(self) -> str:
        lines = [
            f"{self.path.name}: {self.verdict} "
            f"(tolerance {self.tolerance:g} kcal/mol, {self.rmsd_tolerance:g} Å, "
            f"{self.seconds:.2f} s)"
        ]
        stored, reproduced, deltas = self.stored, self.reproduced, self.deltas
        lines.append(
            f"  poses       : stored {stored.get('n_poses')} -> "
            f"reproduced {reproduced.get('n_poses')}"
        )
        lines.append(
            f"  best        : stored {_fmt(stored.get('best_affinity'))} -> "
            f"reproduced {_fmt(reproduced.get('best_affinity'))} kcal/mol "
            f"(delta {_fmt(deltas.get('best_affinity_delta'))})"
        )
        lines.append(
            f"  affinity    : max |delta| {_fmt(deltas.get('max_affinity_delta'))} "
            f"kcal/mol over {deltas.get('n_compared') or 0} mode(s)"
        )
        lines.append(
            f"  top pose    : RMSD {_fmt(deltas.get('top_pose_rmsd'))} Å "
            f"(symmetry-aware, no superposition)"
        )
        if self.settings_defaulted:
            lines.append(
                "  not recorded: " + ", ".join(self.settings_defaulted)
                + " (the kernel defaults were used)"
            )
        if not self.stored_verified:
            lines.append(
                "  warning     : the stored project does not verify, so the "
                "'stored' side is not the run it claims to be"
            )
            for problem in self.verify_problems[:4]:
                lines.append(f"    {problem}")
        for problem in self.problems:
            lines.append(f"  problem     : {problem}")
        if self.verdict == "PASS":
            lines.append(
                "  note        : this is reproducibility, not correctness -- the same "
                "code on the same inputs; the physics is unchanged."
            )
        return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    """A number for a human line, or a dash."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _elements_of(poses_text: str) -> List[str]:
    """The ligand's element per atom, from the first model of a pose document."""
    from .report import parse_pdbqt_atoms

    block = _split_models(poses_text)
    if not block:
        return []
    return [atom.element for atom in parse_pdbqt_atoms(block[0])]


def resolve_engine_settings(engine: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """``(settings, not_recorded)`` for a reproduction.

    Every key in :data:`REPRODUCTION_KEYS` is returned; a value the project did
    not record falls back to the kernel default **and is named** in the second
    element, because a fallback is exactly what makes a mismatch ambiguous.
    """
    settings: Dict[str, Any] = {}
    defaulted: List[str] = []
    for key in REPRODUCTION_KEYS:
        value = engine.get(key)
        if value is None:
            defaulted.append(key)
            settings[key] = ENGINE_DEFAULTS[key]
        else:
            settings[key] = value
    return settings, defaulted


def reproduce_project(
    path: PathLike,
    *,
    tolerance: float = 0.0,
    rmsd_tolerance: float = 0.0,
    cutoff: float = 2.0,
    record: Optional[PathLike] = None,
    save_as: Optional[PathLike] = None,
) -> ReproduceResult:
    """Re-run a project's docking from its stored inputs and compare.

    The re-run uses the stored receptor, ligand and box, and the recorded engine
    settings and seed.  The comparison reports, with numbers:

    * the pose count on both sides;
    * the per-mode affinity difference and its maximum;
    * the symmetry-aware RMSD between the two top poses (no superposition);
    * the ranking table's agreement (the residues column included).

    ``tolerance`` is the affinity tolerance in kcal/mol and ``rmsd_tolerance``
    the top-pose tolerance in Å; both default to **zero**, because this engine is
    deterministic for a given release, seed and input: a re-run on the same
    machine reproduces a pose list bit for bit, and this repository's benchmark
    gate is built on that.  A non-zero tolerance is for a different machine or a
    different kernel build (a parallel reduction may reassociate floating-point
    additions); it is *your* statement about the noise floor, so it belongs in the
    record, which is why it is written next to the verdict.

    `record` writes the outcome as JSON (default ``<project>.reproduce.json``;
    pass ``False`` for none).  `save_as` writes the reproduced run as a new
    project whose provenance names the project it reproduces and the verdict.
    """
    from .docking import dock

    target = Path(path)
    started = time.perf_counter()
    loaded = open_project(target)
    verification = loaded.verify()
    outcome = ReproduceResult(
        path=target,
        verdict="INCONCLUSIVE",
        tolerance=float(tolerance),
        rmsd_tolerance=float(rmsd_tolerance),
        stored_verified=verification.ok,
        verify_problems=list(verification.problems)
        + list(verification.changed)
        + list(verification.missing)
        + list(verification.unlisted),
    )
    if not verification.ok:
        outcome.problems.append(
            "the stored project does not verify; the comparison below is against "
            "whatever the archive now holds, not against the run it recorded"
        )

    # A stored run that cannot even be read is a *result* of the reproduction
    # check, not a crash: the point of the command is to tell the user what is
    # wrong with their archive.
    stored_unreadable = False
    try:
        stored_result = loaded.result()
    except Exception as exc:
        stored_unreadable = True
        stored_result = None
        outcome.problems.append(
            f"the stored run could not be read: {type(exc).__name__}: {exc}"
        )
    stored_poses = list(getattr(stored_result, "poses", None) or []) if stored_result else []
    outcome.stored = {
        "n_poses": len(stored_poses),
        "best_affinity": getattr(stored_result, "best_affinity", None) if stored_result else None,
        "affinities": [float(getattr(pose, "affinity", 0.0) or 0.0) for pose in stored_poses],
        "rmsd_lower_bounds": [
            float(getattr(pose, "rmsd_lower_bound", 0.0) or 0.0) for pose in stored_poses
        ],
        "scoring": getattr(stored_result, "scoring", None) if stored_result else None,
        "seed": getattr(stored_result, "seed", None) if stored_result else None,
    }

    receptor_text = loaded.receptor_text()
    ligand_text = loaded.ligand_text()
    box = loaded.box()
    settings, defaulted = resolve_engine_settings(loaded.engine)
    outcome.settings_used = dict(settings)
    outcome.settings_defaulted = list(defaulted)

    missing: List[str] = []
    if not receptor_text.strip():
        missing.append("the receptor PDBQT")
    if not ligand_text.strip():
        missing.append("the ligand PDBQT")
    if box is None:
        missing.append("the search box")
    if missing:
        outcome.problems.append(
            "this project cannot be re-run: it does not store " + " or ".join(missing)
        )
        outcome.seconds = time.perf_counter() - started
        return _finish_reproduction(outcome, record, save_as)

    kwargs: Dict[str, Any] = {
        "scoring": str(settings["scoring"]),
        "seed": int(settings["seed"]),
        "exhaustiveness": int(settings["exhaustiveness"]),
        "num_poses": int(settings["num_poses"]),
        "min_rmsd": float(settings["min_rmsd"]),
        "energy_range": float(settings["energy_range"]),
        "islands": int(settings["islands"]),
        "population": int(settings["population"]),
        "generations": int(settings["generations"]),
        "use_grid": bool(settings["use_grid"]),
        "refine": bool(settings["refine"]),
    }
    if settings.get("search") is not None:
        kwargs["search"] = str(settings["search"])
    elif settings.get("use_island_ga"):
        kwargs["use_island_ga"] = True

    try:
        rerun = dock(receptor_text, ligand_text, box, **kwargs)
    except Exception as exc:  # a failed re-run is a result, not a crash
        outcome.problems.append(f"the re-run failed: {type(exc).__name__}: {exc}")
        outcome.seconds = time.perf_counter() - started
        return _finish_reproduction(outcome, record, save_as)

    fresh_poses = list(getattr(rerun, "poses", None) or [])
    fresh_affinities = [float(getattr(pose, "affinity", 0.0) or 0.0) for pose in fresh_poses]
    outcome.reproduced = {
        "n_poses": len(fresh_poses),
        "best_affinity": getattr(rerun, "best_affinity", None),
        "affinities": fresh_affinities,
        "scoring": getattr(rerun, "scoring", None),
        "seed": getattr(rerun, "seed", None),
        "grid_points": getattr(rerun, "grid_points", 0),
        "elapsed_s": round(float(getattr(rerun, "elapsed", 0.0) or 0.0), 3),
    }

    stored_affinities = outcome.stored["affinities"]
    paired = list(zip(stored_affinities, fresh_affinities))
    deltas = [abs(a - b) for a, b in paired]
    top_rmsd: Optional[float] = None
    elements = _elements_of(loaded.poses_text())
    if stored_poses and fresh_poses and elements:
        stored_coords = getattr(stored_poses[0], "coords", None)
        fresh_coords = getattr(fresh_poses[0], "coords", None)
        if (
            stored_coords is not None
            and fresh_coords is not None
            and len(stored_coords) == len(fresh_coords)
        ):
            try:
                from .analysis import symmetry_aware_rmsd

                top_rmsd = float(
                    symmetry_aware_rmsd(stored_coords, fresh_coords, elements)
                )
            except Exception as exc:  # pragma: no cover - defensive
                outcome.problems.append(f"the top-pose RMSD could not be computed: {exc}")

    ranking_equal: Optional[bool] = None
    residues_equal: Optional[bool] = None
    try:
        stored_rows = list(loaded.analysis().get("rows") or [])
        fresh_rows = list(
            analyse_result(rerun, receptor=receptor_text or None,
                           ligand=ligand_text or None).get("rows") or []
        )
        if stored_rows and fresh_rows and len(stored_rows) == len(fresh_rows):
            ranking_equal = all(
                abs(float(a.get("affinity") or 0.0) - float(b.get("affinity") or 0.0))
                <= float(tolerance)
                for a, b in zip(stored_rows, fresh_rows)
            )
            residues_equal = all(
                str(a.get("residues") or "") == str(b.get("residues") or "")
                for a, b in zip(stored_rows, fresh_rows)
            )
    except Exception as exc:  # pragma: no cover - defensive
        outcome.problems.append(f"the ranking tables could not be compared: {exc}")

    outcome.deltas = {
        "pose_count_delta": len(fresh_poses) - len(stored_poses),
        "n_compared": len(paired),
        "max_affinity_delta": max(deltas) if deltas else None,
        "mean_affinity_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "per_mode_affinity_delta": [float(value) for value in deltas],
        "best_affinity_delta": (
            None
            if outcome.stored["best_affinity"] is None
            or outcome.reproduced["best_affinity"] is None
            else float(
                outcome.reproduced["best_affinity"] - outcome.stored["best_affinity"]
            )
        ),
        "top_pose_rmsd": top_rmsd,
        "ranking_table_equal": ranking_equal,
        "residues_column_equal": residues_equal,
        "clusters": _cluster_counts(stored_poses, fresh_poses, elements, cutoff),
    }

    max_delta = outcome.deltas["max_affinity_delta"]
    same_count = len(fresh_poses) == len(stored_poses) and len(stored_poses) > 0
    affinity_ok = max_delta is not None and max_delta <= float(tolerance)
    rmsd_ok = top_rmsd is not None and top_rmsd <= float(rmsd_tolerance)
    ranking_ok = ranking_equal is not False and residues_equal is not False
    if stored_unreadable:
        outcome.verdict = "FAIL"
        outcome.problems.append(
            "the stored run is unreadable, so there is nothing to compare against"
        )
    elif same_count and affinity_ok and rmsd_ok and ranking_ok:
        outcome.verdict = "PASS"
    elif defaulted:
        outcome.verdict = "INCONCLUSIVE"
        outcome.problems.append(
            "the project did not record "
            + ", ".join(defaulted)
            + ", so this build used the kernel defaults; a difference cannot be "
            "attributed to the engine rather than to the missing setting"
        )
    else:
        outcome.verdict = "FAIL"
        if not same_count:
            outcome.problems.append(
                f"the re-run reported {len(fresh_poses)} pose(s) where the project "
                f"stores {len(stored_poses)}"
            )
        if not affinity_ok and max_delta is not None:
            outcome.problems.append(
                f"the largest per-mode affinity difference is {max_delta:.6f} "
                f"kcal/mol, above the {float(tolerance):g} kcal/mol tolerance"
            )
        if not rmsd_ok and top_rmsd is not None:
            outcome.problems.append(
                f"the top poses differ by {top_rmsd:.6f} Å, above the "
                f"{float(rmsd_tolerance):g} Å tolerance"
            )
        if ranking_equal is False:
            outcome.problems.append("the two ranking tables disagree")
        if residues_equal is False:
            outcome.problems.append("the two ranking tables disagree on the contacts")
    if top_rmsd is None and same_count:
        outcome.problems.append(
            "the top-pose RMSD could not be computed (no coordinates or a "
            "different atom count); the affinity comparison still applies"
        )

    outcome.seconds = time.perf_counter() - started
    return _finish_reproduction(outcome, record, save_as, rerun=rerun, loaded=loaded)


def _cluster_counts(
    stored_poses: Sequence[Any],
    fresh_poses: Sequence[Any],
    elements: Sequence[str],
    cutoff: float,
) -> Dict[str, Any]:
    """How each pose set clusters, so a comparison can report a regrouping."""
    from .analysis import cluster_poses

    def summarise(poses: Sequence[Any]) -> Dict[str, Any]:
        coords = [pose.coords for pose in poses if getattr(pose, "coords", None) is not None]
        if len(coords) < 2 or not elements:
            return {"n_clusters": len(coords), "largest": len(coords), "representatives": []}
        try:
            clusters = cluster_poses(
                coords,
                cutoff=float(cutoff),
                elements=list(elements)[: len(coords[0])],
                energies=[float(getattr(pose, "affinity", 0.0) or 0.0) for pose in poses],
            )
        except Exception:  # pragma: no cover - defensive
            return {"n_clusters": len(coords), "largest": len(coords), "representatives": []}
        return {
            "n_clusters": len(clusters),
            "largest": max((len(cluster.members) for cluster in clusters), default=0),
            "representatives": [int(cluster.representative) + 1 for cluster in clusters],
        }

    return {
        "cutoff": float(cutoff),
        "stored": summarise(stored_poses),
        "reproduced": summarise(fresh_poses),
    }


def _finish_reproduction(
    outcome: ReproduceResult,
    record: Optional[PathLike],
    save_as: Optional[PathLike],
    *,
    rerun: Any = None,
    loaded: Optional[Project] = None,
) -> ReproduceResult:
    """Write the evidence, and optionally the reproduced run as a new project."""
    if save_as is not None and rerun is not None and loaded is not None:
        notes = [
            f"reproduction of {outcome.path.name} "
            f"(sha256 {sha256_bytes(outcome.path.read_bytes())[:16]}…): "
            f"{outcome.verdict} at tolerance {outcome.tolerance:g} kcal/mol and "
            f"{outcome.rmsd_tolerance:g} Å",
            "the reproduction re-ran the same code on the same stored inputs, box, "
            "settings and seed; a PASS is reproducibility, not correctness",
        ]
        saved = save_project(
            save_as,
            rerun,
            receptor=loaded.receptor_text() or None,
            ligand=loaded.ligand_text() or None,
            box=loaded.box(),
            engine=outcome.settings_used,
            title=f"reproduction of {loaded.title}",
            notes=notes,
            campaign={
                "kind": "reproduction",
                "reproduced_project": outcome.path.name,
                "verdict": outcome.verdict,
                "tolerance_kcal_per_mol": outcome.tolerance,
                "tolerance_rmsd_angstrom": outcome.rmsd_tolerance,
            },
            created=None,
        )
        outcome.saved_as = saved.path
    # The evidence is written last, so it records the reproduced project's path
    # too when one was written.
    if record is not False:
        destination = (
            Path(record)
            if record is not None and record is not True and str(record) != ""
            else outcome.path.with_name(outcome.path.name + ".reproduce.json")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                normalise_paths(outcome.as_dict(), base=destination.parent),
                indent=2,
                ensure_ascii=False,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )
        outcome.recorded_to = destination
    return outcome


# ---------------------------------------------------------------------------
# Comparing runs
# ---------------------------------------------------------------------------


def settings_difference(
    first: Mapping[str, Any], second: Mapping[str, Any], keys: Sequence[str] = REPRODUCTION_KEYS
) -> List[Dict[str, Any]]:
    """The settings that differ between two engine blocks, field by field.

    A comparison that only says "the settings changed" is useless: this names the
    field, both values, and whether one side simply did not record it (the same
    shape the screening resume refusal uses for the same reason).
    """
    differences: List[Dict[str, Any]] = []
    for key in keys:
        left = first.get(key)
        right = second.get(key)
        if left == right:
            continue
        differences.append(
            {
                "field": key,
                "values": [left, right],
                "note": (
                    "not recorded on one side" if left is None or right is None else ""
                ),
            }
        )
    for key in sorted(set(first) | set(second)):
        if key in keys:
            continue
        left = first.get(key)
        right = second.get(key)
        if left != right:
            differences.append({"field": key, "values": [left, right], "note": "other"})
    return differences


def compare_projects(
    paths: Sequence[PathLike],
    *,
    cutoff: float = 2.0,
) -> Dict[str, Any]:
    """A side-by-side comparison of several runs, as a JSON-able dictionary.

    The numbers a comparison has to carry to answer "did my change help?":

    * each run's inputs and their SHA-256 (so a change of *input* is visible);
    * the engine settings, field by field, with the differences named;
    * the pose count, the best affinity, the spread and the ligand efficiency;
    * the per-mode affinity delta and the top-pose RMSD against the first run;
    * how each pose set clusters at `cutoff`, and the rank correlation between
      their affinities.

    `cutoff` is the RMSD clustering cutoff in Å used for the cluster comparison.
    """
    from . import consensus as _consensus

    if len(paths) < 2:
        raise ProjectError("a comparison needs at least two projects")
    runs: List[Dict[str, Any]] = []
    loaded: List[Project] = []
    problems: List[str] = []
    for path in paths:
        entry = open_project(path)
        loaded.append(entry)
        verification = entry.verify()
        result = entry.result()
        poses = list(getattr(result, "poses", None) or [])
        affinities = [float(getattr(pose, "affinity", 0.0) or 0.0) for pose in poses]
        analysis = entry.analysis()
        rows = list(analysis.get("rows") or [])
        elements = _elements_of(entry.poses_text())
        clusters = _cluster_counts(poses, [], elements, cutoff)["stored"]
        runs.append(
            {
                "name": entry.path.name,
                "path": portable_path(entry.path),
                "title": entry.title,
                "schema_version": entry.schema_version,
                "size_bytes": entry.path.stat().st_size if entry.path.exists() else 0,
                "verify_ok": verification.ok,
                "kind": entry.run.get("kind"),
                "seed": entry.engine.get("seed"),
                "scoring": entry.engine.get("scoring"),
                "n_poses": len(poses),
                "best_affinity": getattr(result, "best_affinity", None),
                "spread": (max(affinities) - min(affinities)) if affinities else None,
                "mean_affinity": (sum(affinities) / len(affinities)) if affinities else None,
                "affinities": affinities,
                "rmsd_lower_bounds": [
                    float(getattr(pose, "rmsd_lower_bound", 0.0) or 0.0) for pose in poses
                ],
                "heavy_atoms": rows[0].get("heavy_atoms") if rows else None,
                "ligand_efficiency": rows[0].get("ligand_efficiency") if rows else None,
                "key_residues": rows[0].get("residues") if rows else "",
                "engine": dict(entry.engine),
                "box": entry.box().as_dict() if entry.box() is not None else None,
                "inputs": entry.inputs,
                "campaign": entry.campaign,
                "clusters": clusters,
                "n_tors": entry.run.get("num_tors"),
                "grid_points": entry.run.get("grid_points"),
                "elapsed_s": entry.run.get("elapsed_s"),
                "created_utc": entry.created,
                "does_not_establish": entry.does_not_establish,
            }
        )
        if not verification.ok:
            problems.append(f"{entry.path.name} does not verify")

    baseline = loaded[0]
    baseline_poses = list(getattr(baseline.result(), "poses", None) or [])
    baseline_affinities = runs[0]["affinities"]
    baseline_elements = _elements_of(baseline.poses_text())
    pairs: List[Dict[str, Any]] = []
    for index in range(1, len(loaded)):
        other = loaded[index]
        other_runs = runs[index]
        poses = list(getattr(other.result(), "poses", None) or [])
        paired = list(zip(baseline_affinities, other_runs["affinities"]))
        deltas = [float(b - a) for a, b in paired]
        top_rmsd: Optional[float] = None
        if baseline_poses and poses and baseline_elements:
            left = getattr(baseline_poses[0], "coords", None)
            right = getattr(poses[0], "coords", None)
            if left is not None and right is not None and len(left) == len(right):
                try:
                    from .analysis import symmetry_aware_rmsd

                    top_rmsd = float(symmetry_aware_rmsd(left, right, baseline_elements))
                except Exception:  # pragma: no cover - defensive
                    top_rmsd = None
        rho: Optional[float] = None
        if len(baseline_affinities) == len(other_runs["affinities"]) and len(paired) >= 3:
            try:
                value = float(_consensus.spearman(baseline_affinities, other_runs["affinities"]))
                rho = None if value != value else value
            except Exception:  # pragma: no cover - defensive
                rho = None
        pairs.append(
            {
                "a": runs[0]["name"],
                "b": other_runs["name"],
                "pose_count_delta": len(poses) - len(baseline_poses),
                "best_affinity_delta": (
                    None
                    if other_runs["best_affinity"] is None
                    or runs[0]["best_affinity"] is None
                    else float(other_runs["best_affinity"] - runs[0]["best_affinity"])
                ),
                "per_mode_affinity_delta": deltas,
                "max_affinity_delta": max((abs(value) for value in deltas), default=None),
                "top_pose_rmsd": top_rmsd,
                "spearman_rho": rho,
                "cluster_count_delta": (
                    other_runs["clusters"]["n_clusters"] - runs[0]["clusters"]["n_clusters"]
                ),
                "largest_cluster_delta": (
                    other_runs["clusters"]["largest"] - runs[0]["clusters"]["largest"]
                ),
                "settings_differences": settings_difference(
                    dict(baseline.engine), dict(other.engine)
                ),
                "input_differences": _input_differences(baseline, other),
            }
        )

    return {
        "format": "odock-comparison",
        "version": 1,
        "created_utc": _now(),
        "cutoff_angstrom": float(cutoff),
        "baseline": runs[0]["name"],
        "runs": runs,
        "pairwise": pairs,
        "problems": problems,
        "note": (
            "Affinity deltas are the second run minus the first, per mode.  A "
            "negative delta means the second run scores better; whether that is an "
            "improvement is a scientific judgement, not a number this comparison "
            "can make."
        ),
    }


def _input_differences(first: Project, second: Project) -> List[Dict[str, Any]]:
    """Which stored inputs changed between two runs (by role and hash)."""
    left = {str(item.get("role")): str(item.get("sha256") or "") for item in first.inputs}
    right = {str(item.get("role")): str(item.get("sha256") or "") for item in second.inputs}
    differences: List[Dict[str, Any]] = []
    for role in sorted(set(left) | set(right)):
        if left.get(role) != right.get(role):
            differences.append(
                {
                    "role": role,
                    "in_first": bool(left.get(role)),
                    "in_second": bool(right.get(role)),
                    "same_hash": bool(left.get(role)) and left.get(role) == right.get(role),
                }
            )
    return differences


def comparison_summary(report: Mapping[str, Any]) -> str:
    """The comparison as a short text table, for a terminal."""
    runs = list(report.get("runs") or [])
    if not runs:
        return "no run to compare"
    header = ["run", "poses", "best", "spread", "seed", "scoring", "clusters"]
    rows = []
    for run in runs:
        rows.append(
            [
                str(run.get("name"))[:28],
                str(run.get("n_poses")),
                _fmt(run.get("best_affinity")),
                _fmt(run.get("spread")),
                str(run.get("seed")),
                str(run.get("scoring")),
                str((run.get("clusters") or {}).get("n_clusters")),
            ]
        )
    widths = [len(cell) for cell in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(header)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows
    ]
    for pair in report.get("pairwise") or []:
        lines.append("")
        lines.append(f"{pair['a']} -> {pair['b']}")
        lines.append(
            f"  poses {pair['pose_count_delta']:+d}, best affinity "
            f"{_fmt(pair['best_affinity_delta'])} kcal/mol, top-pose RMSD "
            f"{_fmt(pair['top_pose_rmsd'])} A, max per-mode delta "
            f"{_fmt(pair['max_affinity_delta'])} kcal/mol, rho {_fmt(pair['spearman_rho'], 2)}"
        )
        for difference in pair["settings_differences"]:
            left, right = difference["values"]
            lines.append(
                f"  setting {difference['field']}: {left} -> {right}"
                + (f" ({difference['note']})" if difference["note"] else "")
            )
        for difference in pair["input_differences"]:
            lines.append(
                f"  input {difference['role']}: "
                + ("only in the first" if not difference["in_second"] else
                   "only in the second" if not difference["in_first"] else
                   "different hash")
            )
    for problem in report.get("problems") or []:
        lines.append(f"  problem: {problem}")
    return "\n".join(lines)


def campaign_index(directory: PathLike, *, pattern: str = "*.odockproj") -> Dict[str, Any]:
    """Summarise every project in a directory, for the campaign index page."""
    root = Path(directory)
    if not root.exists():
        raise ProjectError(f"no such directory: {root}")
    found = sorted(root.rglob(pattern))
    if not found:
        raise ProjectError(f"no project matching {pattern} under {root}")
    entries: List[Dict[str, Any]] = []
    problems: List[str] = []
    campaign_hashes: set = set()
    library_hashes: set = set()
    for path in found:
        try:
            loaded = open_project(path)
        except ProjectError as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        verification = loaded.verify()
        result = loaded.result()
        rows = list(loaded.analysis().get("rows") or [])
        if loaded.campaign.get("docking_hash"):
            campaign_hashes.add(str(loaded.campaign["docking_hash"]))
        if loaded.campaign.get("library_hash"):
            library_hashes.add(str(loaded.campaign["library_hash"]))
        relative = path.relative_to(root)
        stem = path.name[: -len(path.suffix)] if path.suffix else path.name
        report_sibling = path.with_suffix(".html")
        entries.append(
            {
                "name": path.name,
                "label": stem,
                "relative": _as_posix(str(relative)),
                "title": loaded.title,
                "kind": loaded.run.get("kind"),
                "n_poses": int(loaded.run.get("n_poses") or 0),
                "best_affinity": loaded.run.get("best_affinity"),
                "seed": loaded.engine.get("seed"),
                "scoring": loaded.engine.get("scoring"),
                "heavy_atoms": rows[0].get("heavy_atoms") if rows else None,
                "ligand_efficiency": rows[0].get("ligand_efficiency") if rows else None,
                "key_residues": rows[0].get("residues") if rows else "",
                "size_bytes": path.stat().st_size,
                "verify_ok": verification.ok,
                "schema_version": loaded.schema_version,
                "created_utc": loaded.created,
                "campaign": loaded.campaign,
                "report_href": (
                    _as_posix(str(report_sibling.relative_to(root)))
                    if report_sibling.exists()
                    else _as_posix(str(relative))
                ),
                "has_report": report_sibling.exists(),
            }
        )
    if not entries:
        raise ProjectError(f"no readable project under {root}: " + "; ".join(problems))
    ranked = sorted(
        entries,
        key=lambda item: (
            item["best_affinity"] is None,
            item["best_affinity"] if item["best_affinity"] is not None else 0.0,
        ),
    )
    affinities = [item["best_affinity"] for item in entries if item["best_affinity"] is not None]
    return {
        "format": "odock-campaign-index",
        "version": 1,
        "created_utc": _now(),
        "directory": portable_path(root),
        "pattern": pattern,
        "n_projects": len(entries),
        "n_verified": sum(1 for item in entries if item["verify_ok"]),
        "best_affinity": min(affinities) if affinities else None,
        "worst_affinity": max(affinities) if affinities else None,
        "campaign_hashes": sorted(campaign_hashes),
        "library_hashes": sorted(library_hashes),
        "entries": ranked,
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# The command line
#
# The parser and the command functions live in this module (not in cli.py) so
# that one agent owns them: cli.build_parser only calls
# :func:`add_report_parsers`.  Everything here is a thin wrapper around the API
# above, so anything the commands do can also be scripted.
# ---------------------------------------------------------------------------


def _shown(value: Any) -> str:
    """A value for a human line: ``None`` is "not recorded", never "None"."""
    if value is None or value == "":
        return "not recorded"
    return str(value)


def _registered(sub, name: str) -> bool:
    """Whether `sub` already has a subcommand called `name`."""
    choices = getattr(sub, "choices", None)
    return isinstance(choices, Mapping) and name in choices


def _cli_err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _assignments(values: Optional[Sequence[str]], *, what: str) -> Dict[str, str]:
    """``["role=FILE", ...]`` -> ``{"role": "FILE"}``, with a clear error."""
    found: Dict[str, str] = {}
    for item in values or ():
        key, separator, value = str(item).partition("=")
        if not separator or not key.strip() or not value.strip():
            raise SystemExit(f"error: --{what} expects KEY=FILE, got {item!r}")
        found[key.strip()] = value.strip()
    return found


def cmd_project_reproduce(args) -> int:
    """Re-run a project's docking and report whether it reproduces."""
    try:
        outcome = reproduce_project(
            args.project,
            tolerance=float(args.tolerance),
            rmsd_tolerance=float(args.rmsd_tolerance),
            cutoff=float(args.cutoff),
            record=False if args.no_record else (args.record or None),
            save_as=args.save_as,
        )
    except ProjectError as exc:
        _cli_err(f"odock project reproduce: error: {exc}")
        return 2
    if args.json:
        print(json.dumps(outcome.as_dict(), indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(outcome.summary())
    if outcome.recorded_to is not None:
        _cli_err(f"recorded the evidence in {outcome.recorded_to}")
    if outcome.saved_as is not None:
        _cli_err(f"wrote the reproduced run to {outcome.saved_as}")
    _cli_err(
        f"reproduction of {outcome.path.name}: {outcome.verdict} "
        f"(max |delta affinity| {_fmt(outcome.deltas.get('max_affinity_delta'))} kcal/mol, "
        f"top-pose RMSD {_fmt(outcome.deltas.get('top_pose_rmsd'))} A, "
        f"{outcome.seconds:.2f} s)"
    )
    # A failed reproduction is a finding, not a usage error: exit 1 so a script
    # can gate on it, with the numbers on stdout either way.
    return 0 if outcome.verdict == "PASS" else 1


def cmd_project_compare(args) -> int:
    """Compare two or more runs side by side."""
    from . import htmlreport as _htmlreport

    paths = list(args.projects)
    if args.out is None and len(paths) < 2:
        _cli_err("odock project compare: error: comparing needs at least two projects")
        return 2
    try:
        report = compare_projects(paths, cutoff=float(args.cutoff))
    except ProjectError as exc:
        _cli_err(f"odock project compare: error: {exc}")
        return 2
    written: List[str] = []
    files = None
    if args.out:
        try:
            files = _htmlreport.write_comparison_report(
                args.out,
                report,
                title=args.title,
                json_out=args.json_out,
                text_out=args.text_out,
            )
        except (_project_error_types()) as exc:
            _cli_err(f"odock project compare: error: {exc}")
            return 2
        written = [str(files.path)]
        for path in (files.json_path, files.text_path):
            if path is not None:
                written.append(str(path))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(comparison_summary(report))
    if written:
        _cli_err("wrote " + ", ".join(written))
    return 0 if not report.get("problems") else 1


def _project_error_types() -> Tuple[type, ...]:
    """The exceptions the report writers raise for a bad input."""
    return (ProjectError, ValueError)


def cmd_project_index(args) -> int:
    """Build one self-contained page over a directory of projects."""
    from . import htmlreport as _htmlreport

    try:
        payload = campaign_index(args.directory, pattern=args.pattern)
    except ProjectError as exc:
        _cli_err(f"odock project index: error: {exc}")
        return 2
    written: List[str] = []
    if args.out:
        try:
            files = _htmlreport.write_campaign_index(
                args.out,
                payload,
                title=args.title,
                json_out=args.json_out,
                text_out=args.text_out,
            )
        except (_project_error_types()) as exc:
            _cli_err(f"odock project index: error: {exc}")
            return 2
        written = [str(files.path)]
        for path in (files.json_path, files.text_path):
            if path is not None:
                written.append(str(path))
    if args.json or not args.out:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(
            f"{payload['n_projects']} project(s), {payload['n_verified']} verified, "
            f"best {_fmt(payload['best_affinity'])} kcal/mol"
        )
        for entry in payload["entries"]:
            print(
                f"  {entry['label'][:40]:<42} {_fmt(entry['best_affinity']):>8} kcal/mol  "
                f"poses {entry['n_poses']:<3} seed {_shown(entry['seed']):<16} "
                f"{'OK' if entry['verify_ok'] else 'FAILED'}"
            )
    if written:
        _cli_err("wrote " + ", ".join(written))
    return 0 if payload["n_verified"] == payload["n_projects"] else 1


def _save_command(args) -> str:
    """The `odock project save` line that would reproduce this save.

    Rebuilt from the parsed arguments rather than read from ``sys.argv``: the
    command is recorded as provenance, and when the CLI is driven
    programmatically (a script, a test) ``sys.argv`` belongs to whatever started
    the process.
    """
    parts: List[str] = ["odock", "project", "save"]

    def add(flag: str, value: Any) -> None:
        if value is None or value is False or value == []:
            return
        if value is True:
            parts.append(flag)
            return
        parts.extend([flag, str(value)])

    add("-o", args.out)
    add("-p", args.poses)
    add("--dock-json", args.dock_json)
    add("-r", args.receptor)
    add("-l", args.ligand)
    add("-b", args.box)
    for item in args.original or ():
        add("--original", item)
    for item in args.extra or ():
        add("--extra", item)
    add("--engine-json", args.engine_json)
    add("--preparation-json", args.preparation_json)
    add("-e", args.exhaustiveness)
    add("-n", args.num_poses)
    add("--search", args.search)
    add("-s", args.scoring)
    add("--seed", args.seed)
    add("--min-rmsd", args.min_rmsd)
    add("--energy-range", args.energy_range)
    add("--no-grid", args.no_grid)
    add("--no-refine", args.no_refine)
    add("--strain", args.strain)
    add("--title", args.title)
    for note in args.note or ():
        add("--note", note)
    return portable_command(parts)


def cmd_project_save(args) -> int:
    """Write a completed run as a self-contained ``.odockproj``."""
    engine: Dict[str, Any] = {
        "exhaustiveness": args.exhaustiveness,
        "num_poses": args.num_poses,
        "search": args.search,
        "scoring": args.scoring,
        "seed": args.seed,
        "min_rmsd": args.min_rmsd,
        "energy_range": args.energy_range,
        "use_grid": False if args.no_grid else None,
        "refine": False if args.no_refine else None,
    }
    engine = {key: value for key, value in engine.items() if value is not None}
    if args.engine_json:
        try:
            extra = json.loads(Path(args.engine_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"error: cannot read --engine-json {args.engine_json}: {exc}")
        if not isinstance(extra, Mapping):
            raise SystemExit("error: --engine-json must hold a JSON object")
        engine.update(dict(extra))

    result: Any = None
    poses = args.poses
    if args.dock_json:
        try:
            payload = json.loads(Path(args.dock_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"error: cannot read --dock-json {args.dock_json}: {exc}")
        if not isinstance(payload, Mapping) or not payload.get("poses"):
            raise SystemExit(
                f"error: {args.dock_json} is not a dock JSON document (no 'poses'); "
                "`odock dock --json-out FILE` writes one"
            )
        if poses:
            # The pose file carries the coordinates; the JSON carries the settings.
            result = result_from_pdbqt(
                poses,
                box=args.box,
                receptor=args.receptor,
                ligand=args.ligand,
                seed=payload.get("seed"),
                scoring=args.scoring or payload.get("scoring"),
            )
        else:
            result = result_from_dock_json(
                payload,
                box=args.box,
                receptor=args.receptor,
                ligand=args.ligand,
                scoring=args.scoring,
            )

    preparation = None
    if args.preparation_json:
        try:
            preparation = json.loads(Path(args.preparation_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(
                f"error: cannot read --preparation-json {args.preparation_json}: {exc}"
            )

    command = args.command or _save_command(args)
    try:
        saved = save_project(
            args.out,
            result,
            poses=poses if result is None else None,
            receptor=args.receptor,
            ligand=args.ligand,
            box=args.box,
            original_inputs=_assignments(args.original, what="original"),
            extra_files=_assignments(args.extra, what="extra"),
            engine=engine,
            preparation=preparation,
            command=command,
            title=args.title,
            notes=args.note,
            strain=bool(args.strain),
        )
    except ProjectError as exc:
        _cli_err(f"odock project save: error: {exc}")
        return 2

    report = saved.verify()
    if args.json:
        payload = saved.info()
        payload["verify"] = report.as_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(saved.describe())
        print()
        print(report.summary())
    _cli_err(
        f"wrote {saved.path} ({saved.path.stat().st_size} bytes, "
        f"{len(saved.entries)} entries, verify={'OK' if report.ok else 'FAILED'})"
    )
    return 0 if report.ok else 1


def cmd_project_verify(args) -> int:
    """Recompute every stored hash of a project and report any change."""
    report = verify_project(args.project)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(report.summary())
    if not report.ok and not args.json:
        _cli_err("odock project verify: the project did not verify")
    return 0 if report.ok else 1


def cmd_project_info(args) -> int:
    """Report what a project contains, without recomputing its hashes."""
    try:
        loaded = open_project(args.project)
    except ProjectError as exc:
        _cli_err(f"odock project info: error: {exc}")
        return 2
    if args.json:
        print(json.dumps(loaded.info(), indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(loaded.describe())
    return 0


def cmd_project_open(args) -> int:
    """Verify a project, optionally extract it, and show what it holds."""
    try:
        loaded = open_project(args.project)
    except ProjectError as exc:
        _cli_err(f"odock project open: error: {exc}")
        return 2
    report = loaded.verify()
    payload: Dict[str, Any] = {"info": loaded.info(), "verify": report.as_dict()}
    extraction: List[str] = []
    if args.out:
        extraction = [str(path) for path in loaded.extract(args.out)]
        payload["extracted"] = extraction
    if args.reanalyse:
        analysis = loaded.reanalyse()
        payload["analysis"] = {
            "n_poses": analysis.get("n_poses"),
            "best": analysis.get("best"),
            "interaction_profile": analysis.get("interaction_profile"),
            "pose_quality": analysis.get("pose_quality"),
            "flags": analysis.get("flags"),
        }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif args.quiet:
        pass
    else:
        print(report.summary())
        if extraction:
            print(f"extracted {len(extraction)} file(s) to {args.out}")
        if args.reanalyse:
            best = payload["analysis"].get("best") or {}
            print(
                f"re-analysis : {payload['analysis'].get('n_poses')} pose(s), "
                f"best {best.get('affinity', '?')} kcal/mol"
            )
            for flag in payload["analysis"].get("flags") or []:
                print(f"  note: {flag}")
        print()
        print(loaded.describe())
    if not report.ok:
        _cli_err("odock project open: the project did not verify")
        return 1
    return 0


def cmd_project_screen(args) -> int:
    """Wrap each docked molecule of a screening campaign as its own project."""
    try:
        written = save_screen_projects(
            args.screen,
            args.out,
            top=int(args.top or 0),
            receptor=args.receptor,
            only_ok=not args.all,
        )
    except ProjectError as exc:
        _cli_err(f"odock project screen: error: {exc}")
        return 2
    payload = []
    for member in written:
        report = member.verify()
        payload.append({"project": str(member.path), "verify": report.as_dict()})
        if not report.ok:
            _cli_err(f"odock project screen: {member.path.name} did not verify")
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(f"{len(written)} project(s) written to {args.out}")
        for item in payload:
            name = Path(item["project"]).name
            print(f"  {name:<40} {'OK' if item['verify']['ok'] else 'FAILED'}")
    _cli_err(f"wrote {len(written)} project(s) to {args.out}")
    return 0 if all(item["verify"]["ok"] for item in payload) else 1


def add_project_parser(sub) -> None:
    """Register ``odock project save|screen|open|verify|info`` on `sub`.

    Called once from :func:`odock.cli.build_parser`; the commands and their
    arguments live here so a single module owns them.  Idempotent: several
    agents add their own parsers to the same subparser list, and a re-applied
    line must not be an error.
    """
    if _registered(sub, "project"):
        return
    parser = sub.add_parser(
        "project",
        help="save, verify and reopen a docking run as a .odockproj",
        description=(
            "A project is the run itself: the input structures and their "
            "SHA-256, the preparation settings, the box, the engine settings "
            "including the seed, the poses and the analysis, all in one "
            "verifiable file that reopens anywhere."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    save = actions.add_parser(
        "save",
        help="write a completed run as a project file",
        description=(
            "Save a run as a .odockproj.  The run comes from a pose PDBQT "
            "(--poses) or from a dock JSON document (--dock-json, optionally "
            "with --poses for the coordinates); the box comes from --box.  "
            "Anything else that belongs to the run (the raw inputs, a log, a "
            "figure) can be stored with it."
        ),
    )
    save.add_argument("-o", "--out", required=True, help="output .odockproj")
    save.add_argument("-p", "--poses", help="multi-model pose PDBQT of the run")
    save.add_argument("--dock-json", help="JSON written by `odock dock --json-out`")
    save.add_argument("-r", "--receptor", help="prepared receptor PDBQT")
    save.add_argument("-l", "--ligand", help="prepared ligand PDBQT")
    save.add_argument("-b", "--box", help="box JSON written by `odock box`")
    save.add_argument(
        "--original",
        action="append",
        metavar="ROLE=FILE",
        help="a raw input structure stored with the run, e.g. "
             "--original receptor=3PTB.pdb (repeatable)",
    )
    save.add_argument(
        "--extra",
        action="append",
        metavar="NAME=FILE",
        help="any other file that belongs to the run (repeatable)",
    )
    save.add_argument(
        "--engine-json",
        help="JSON object of engine settings, merged over the recorded ones",
    )
    save.add_argument(
        "--preparation-json",
        help="JSON object with the preparation reports, "
             'e.g. {"ligand": {"n_rotatable_bonds": 1}}',
    )
    save.add_argument("-e", "--exhaustiveness", type=int, help="independent MC runs")
    save.add_argument("-n", "--num-poses", type=int, help="poses requested")
    save.add_argument("--search", help="search protocol (monte_carlo, lga, lga_solis)")
    save.add_argument("-s", "--scoring", choices=["vina", "vinardo", "ad4"],
                      help="force field the run used")
    save.add_argument("--seed", type=int, help="the seed the run used")
    save.add_argument("--min-rmsd", type=float, help="pose deduplication cutoff (Å)")
    save.add_argument("--energy-range", type=float, help="reporting window (kcal/mol)")
    save.add_argument("--no-grid", action="store_true", help="the run used the exact scorer")
    save.add_argument("--no-refine", action="store_true", help="the run skipped refinement")
    save.add_argument("--strain", action="store_true", help="also compute the ligand strain")
    save.add_argument("--command", help="the exact command to record (default: this one)")
    save.add_argument("--title", help="a human title for the run")
    save.add_argument("--note", action="append", help="a free-text note (repeatable)")
    save.add_argument("--json", action="store_true", help="print the project info as JSON")
    save.add_argument("-q", "--quiet", action="store_true", help="no summary")
    save.set_defaults(func=cmd_project_save)

    verify = actions.add_parser(
        "verify",
        help="recompute every stored hash and report any change",
        description=(
            "Re-hash every stored file and compare it with the manifest, the "
            "plain-text checksum list and the archive itself.  A hash shows that "
            "nothing has changed; it does not show that the run is correct."
        ),
    )
    verify.add_argument("project", help="the .odockproj to check")
    verify.add_argument("--json", action="store_true", help="print the report as JSON")
    verify.add_argument("-q", "--quiet", action="store_true", help="print nothing on success")
    verify.set_defaults(func=cmd_project_verify)

    info = actions.add_parser(
        "info",
        help="what a project holds (no hashing)",
        description="Report the schema, tool, run, box, engine settings, inputs "
                    "and stored files of a project.",
    )
    info.add_argument("project", help="the .odockproj to describe")
    info.add_argument("--json", action="store_true", help="print everything as JSON")
    info.add_argument("-q", "--quiet", action="store_true", help="print nothing")
    info.set_defaults(func=cmd_project_info)

    open_parser = actions.add_parser(
        "open",
        help="verify a project, optionally extract it, and show what it holds",
        description=(
            "A project reopens anywhere: it needs no original path.  This "
            "verifies it, prints what it contains, and with --out writes every "
            "stored file into a directory so the rest of the tool can work on it "
            "again."
        ),
    )
    open_parser.add_argument("project", help="the .odockproj to open")
    open_parser.add_argument("--out", help="directory to extract the stored files into")
    open_parser.add_argument(
        "--reanalyse",
        action="store_true",
        help="recompute the analysis from the stored inputs and report it",
    )
    open_parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    open_parser.add_argument("-q", "--quiet", action="store_true",
                             help="only report what failed")
    open_parser.set_defaults(func=cmd_project_open)

    screen = actions.add_parser(
        "screen",
        help="wrap the docked molecules of a screening campaign as projects",
        description=(
            "Read a screening output directory (run.json + results.jsonl, as "
            "`odock screen` writes it) and write one project per molecule, each "
            "with its poses, its prepared ligand, the receptor and the campaign's "
            "engine settings."
        ),
    )
    screen.add_argument("-s", "--screen", required=True, metavar="DIR",
                        help="the screening output directory")
    screen.add_argument("-o", "--out", required=True, metavar="DIR",
                        help="directory for the .odockproj files")
    screen.add_argument("--top", type=int, default=0, help="wrap only the N best (0 = all)")
    screen.add_argument("--receptor", help="restrict to one receptor of a panel")
    screen.add_argument("--all", action="store_true",
                        help="include the rows whose status is not 'ok'")
    screen.add_argument("--json", action="store_true", help="print the result as JSON")
    screen.add_argument("-q", "--quiet", action="store_true", help="no per-file listing")
    screen.set_defaults(func=cmd_project_screen)

    reproduce = actions.add_parser(
        "reproduce",
        help="re-run a project's docking and report whether it reproduces",
        description=(
            "Reopen a project, re-run the docking from its stored inputs, box, "
            "engine settings and seed, and compare the result with the stored "
            "one: pose count, per-mode affinity, the top-pose RMSD and the "
            "ranking table.  PASS means the same code on the same inputs gives the "
            "same poses; it is reproducibility, not correctness."
        ),
    )
    reproduce.add_argument("project", help="the .odockproj to re-run")
    reproduce.add_argument(
        "--tolerance", type=float, default=0.0,
        help="affinity tolerance in kcal/mol (default 0: this engine is "
             "deterministic for a given release, seed and input)",
    )
    reproduce.add_argument(
        "--rmsd-tolerance", type=float, default=0.0, dest="rmsd_tolerance",
        help="top-pose RMSD tolerance in Å (default 0)",
    )
    reproduce.add_argument(
        "--cutoff", type=float, default=2.0, help="clustering cutoff for the report (Å)"
    )
    reproduce.add_argument(
        "--record", metavar="FILE",
        help="where to write the evidence (default: <project>.reproduce.json)",
    )
    reproduce.add_argument(
        "--no-record", action="store_true", help="do not write the evidence file"
    )
    reproduce.add_argument(
        "--save-as", metavar="FILE",
        help="also save the reproduced run as a new project carrying the verdict",
    )
    reproduce.add_argument("--json", action="store_true", help="print the outcome as JSON")
    reproduce.add_argument("-q", "--quiet", action="store_true", help="no summary")
    reproduce.set_defaults(func=cmd_project_reproduce)

    compare = actions.add_parser(
        "compare",
        help="compare several runs side by side",
        description=(
            "Compare two or more projects: the inputs and their hashes, the engine "
            "settings that differ (named field by field), the per-mode affinity "
            "deltas, the RMSD between the top poses, and how each pose set "
            "clusters.  With --out it is written as one self-contained HTML page "
            "plus JSON and text siblings."
        ),
    )
    compare.add_argument("projects", nargs="+", help="the .odockproj files to compare")
    compare.add_argument("-o", "--out", help="write the comparison as HTML")
    compare.add_argument("--json-out", help="machine-readable sibling (default: <out>.json)")
    compare.add_argument("--text-out", help="plain-text sibling (default: <out>.txt)")
    compare.add_argument("--cutoff", type=float, default=2.0,
                         help="clustering cutoff for the comparison (Å)")
    compare.add_argument("--title", help="title for the comparison page")
    compare.add_argument("--json", action="store_true", help="print the comparison as JSON")
    compare.add_argument("-q", "--quiet", action="store_true", help="no text table")
    compare.set_defaults(func=cmd_project_compare)

    index = actions.add_parser(
        "index",
        help="summarise a directory of projects as one page",
        description=(
            "Build one self-contained page over a directory of projects (a "
            "`odock project screen` output, say): the key numbers per member, a "
            "link to its report when one exists, the campaign hashes, and the "
            "verification state of each file."
        ),
    )
    index.add_argument("directory", help="directory to scan for projects")
    index.add_argument("-o", "--out", help="write the index as HTML")
    index.add_argument("--pattern", default="*.odockproj",
                       help="which files to include (default: *.odockproj, recursive)")
    index.add_argument("--json-out", help="machine-readable sibling (default: <out>.json)")
    index.add_argument("--text-out", help="plain-text sibling (default: <out>.txt)")
    index.add_argument("--title", help="title for the index page")
    index.add_argument("--json", action="store_true", help="print the index as JSON")
    index.add_argument("-q", "--quiet", action="store_true", help="no listing")
    index.set_defaults(func=cmd_project_index)


def add_report_parsers(sub) -> None:
    """Register this module's commands -- and the HTML report -- on `sub`.

    ``odock report-html`` belongs to :mod:`odock.htmlreport`, but the two
    commands are one feature (a project is what a report is usually generated
    from), so they are registered together from here and ``cli.py`` still has a
    single call to make.
    """
    add_project_parser(sub)
    from .htmlreport import add_report_html_parser

    add_report_html_parser(sub)

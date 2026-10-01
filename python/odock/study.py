# SPDX-License-Identifier: GPL-3.0-or-later
"""A study: a named, verifiable collection of docking runs.

A single project answers "what did this run do?".  A *study* answers the question
that comes next: "which runs belong together, what were they run for, and what
changed between two versions of the series?"

Layout::

    my-study/
        study.json          the manifest: metadata, protocol, members + hashes
        SHA256SUMS          <sha256>  <member> for study.json and every member
        members/<name>.odockproj
        <name>.html         (optional) the report of a member, next to it

The manifest records, for every member, the project file's SHA-256 as it was when
the member was added.  ``odock study verify`` re-hashes the files on disk and
reports a changed or missing member **by name**; it also verifies each member
project's own internal hashes, so a study is checked at both levels.

``odock study diff`` compares two studies and names, rather than describes:

* the metadata and **protocol fields that differ**, with both values;
* the members added and removed;
* how the **hit list moved**: per-member rank deltas within the study, and the
  molecules that entered or left a stated top-N.

Every path that goes into ``study.json`` passes through
:func:`odock.project.portable_path`, and the study files live *inside* the study
directory, so a study is a folder you can copy to another machine and verify
there — the same promise the project file makes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import project as _project

__all__ = [
    "MEMBERS_DIR",
    "SCHEMA_VERSION",
    "STUDY_FORMAT",
    "STUDY_MANIFEST",
    "SUMS_NAME",
    "Study",
    "StudyError",
    "StudyVerifyResult",
    "add_projects",
    "add_study_parser",
    "create_study",
    "load_study",
    "study_diff",
    "study_report",
    "verify_study",
]

PathLike = Union[str, os.PathLike]

STUDY_FORMAT = "odock-study"
SCHEMA_VERSION = 1
STUDY_MANIFEST = "study.json"
SUMS_NAME = "SHA256SUMS"
MEMBERS_DIR = "members"

#: Metadata keys a study understands; anything else the caller passes is kept
#: under its own name, so a study can carry a field this build does not know.
KNOWN_METADATA: Tuple[str, ...] = ("target", "receptors", "library", "protocol", "notes")

#: What a study diff does not establish.
DIFF_LIMITS: Tuple[str, ...] = (
    "A study is a collection, not a control: it says which runs were grouped and "
    "with which protocol, not that the grouping is a fair experiment.",
    "A rank delta is a movement within its own study.  Two studies with different "
    "members have different pools, so a rank can move because the pool changed "
    "rather than because the molecule did.",
    "A member that entered a top-N scored better *relative to this pool*, which is "
    "not the same as being a better binder.",
    "The affinities are one force field's scores; comparing studies run with "
    "different force fields compares two scales.",
    "Nothing here is a statistical test: the deltas are arithmetic on the recorded "
    "numbers, with no error model behind them.",
)


class StudyError(_project.ProjectError):
    """A study could not be created, read, verified or compared."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _study_directory(path: PathLike) -> Path:
    """The directory of a study: `X`, `X/study.json` or a members file."""
    target = Path(path)
    if target.is_file() and target.name == STUDY_MANIFEST:
        return target.parent
    return target


def _safe_name(text: str) -> str:
    """A member name that is safe as a file name and recognisable to a human."""
    cleaned = "".join(
        character if (character.isalnum() or character in "._-") else "_"
        for character in str(text)
    ).strip("._")
    return cleaned[:64] or "member"


def _summary_of(project: Any) -> Dict[str, Any]:
    """The member numbers a study index shows, read from the project itself."""
    run = project.run
    engine = project.engine
    rows = list(project.analysis().get("rows") or [])
    best = rows[0] if rows else {}
    return {
        "title": project.title,
        "kind": run.get("kind"),
        "n_poses": int(run.get("n_poses") or 0),
        "best_affinity": run.get("best_affinity"),
        "seed": engine.get("seed"),
        "scoring": engine.get("scoring"),
        "exhaustiveness": engine.get("exhaustiveness"),
        "num_tors": run.get("num_tors"),
        "heavy_atoms": best.get("heavy_atoms"),
        "ligand_efficiency": best.get("ligand_efficiency"),
        "key_residues": best.get("residues"),
        "created_utc": project.created,
        "schema_version": project.schema_version,
        "campaign": project.campaign,
        "audit": project.audit,
    }


# ---------------------------------------------------------------------------
# The study object
# ---------------------------------------------------------------------------


@dataclass
class StudyVerifyResult:
    """The outcome of :func:`verify_study`."""

    path: Path
    name: str
    n_members: int
    checked: int = 0
    ok_members: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unlisted: List[str] = field(default_factory=list)
    project_problems: List[str] = field(default_factory=list)
    sums_problems: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.changed or self.missing or self.unlisted or self.project_problems
            or self.sums_problems or self.problems
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "study": self.name,
            "path": self.path.name,
            "ok": bool(self.ok),
            "members": int(self.n_members),
            "checked": int(self.checked),
            "verified": list(self.ok_members),
            "changed": list(self.changed),
            "missing": list(self.missing),
            "unlisted": list(self.unlisted),
            "project_problems": list(self.project_problems),
            "sums_problems": list(self.sums_problems),
            "problems": list(self.problems),
        }

    def summary(self) -> str:
        head = (
            f"{self.name}: {'OK' if self.ok else 'FAILED'} "
            f"({self.checked} of {self.n_members} member(s) re-hashed)"
        )
        lines = [head]
        for label, items in (
            ("changed", self.changed),
            ("missing", self.missing),
            ("present but unlisted", self.unlisted),
            ("member project", self.project_problems),
            ("SHA256SUMS", self.sums_problems),
            ("problem", self.problems),
        ):
            for item in items:
                lines.append(f"  {label}: {item}")
        if self.ok:
            lines.append(
                "  note: this shows that no member file has changed since it was "
                "added; it does not show that the study is a good experiment."
            )
        return "\n".join(lines)


@dataclass
class Study:
    """An open study: its manifest and the projects it contains."""

    path: Path
    manifest: Dict[str, Any]

    # -- manifest views ----------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.path / STUDY_MANIFEST

    @property
    def name(self) -> str:
        return str(self.manifest.get("name") or self.path.name)

    @property
    def title(self) -> str:
        return str(self.manifest.get("title") or self.name)

    @property
    def metadata(self) -> Dict[str, Any]:
        block = self.manifest.get("metadata")
        return dict(block) if isinstance(block, Mapping) else {}

    @property
    def protocol(self) -> Dict[str, Any]:
        block = self.metadata.get("protocol")
        return dict(block) if isinstance(block, Mapping) else {}

    @property
    def audit(self) -> Dict[str, Any]:
        block = self.manifest.get("audit")
        found = dict(block) if isinstance(block, Mapping) else {}
        found.setdefault("operator", None)
        found.setdefault("tool_version", _project.tool_version())
        found.setdefault("created_utc", self.created_utc)
        return found

    @property
    def created_utc(self) -> str:
        return str(self.manifest.get("created_utc") or "")

    @property
    def notes(self) -> List[str]:
        items = self.manifest.get("notes")
        return [str(item) for item in items] if isinstance(items, list) else []

    @property
    def members(self) -> List[Dict[str, Any]]:
        items = self.manifest.get("members")
        return [dict(item) for item in items] if isinstance(items, list) else []

    def member(self, name: str) -> Optional[Dict[str, Any]]:
        """One member by name or label."""
        for item in self.members:
            if str(item.get("name")) == name or str(item.get("label")) == name:
                return item
        return None

    def member_path(self, member: Mapping[str, Any]) -> Path:
        """Where a member's project file is (or should be) inside the study."""
        relative = str(member.get("project") or "")
        return self.path / relative if relative else self.path / MEMBERS_DIR / "missing"

    def project(self, name: str) -> Optional[_project.Project]:
        """Open one member project, or ``None`` when its file is not there."""
        member = self.member(name)
        if member is None:
            return None
        path = self.member_path(member)
        if not path.exists():
            return None
        return _project.open_project(path)

    # -- verification ------------------------------------------------------

    def verify(self) -> StudyVerifyResult:
        """Re-hash every member project and verify each project's own contents."""
        result = StudyVerifyResult(
            path=self.path, name=self.name, n_members=len(self.members)
        )
        listed: List[str] = []
        for member in self.members:
            name = str(member.get("name"))
            path = self.member_path(member)
            relative = str(member.get("project") or path.name)
            listed.append(relative)
            if not path.exists():
                result.missing.append(f"{name} ({relative})")
                continue
            digest = _project.sha256_bytes(path.read_bytes())
            result.checked += 1
            if digest != str(member.get("sha256") or ""):
                result.changed.append(
                    f"{name} ({relative}): sha256 "
                    f"{str(member.get('sha256') or '')[:12]}… -> {digest[:12]}…"
                )
                continue
            try:
                inner = _project.open_project(path).verify()
            except _project.ProjectError as exc:
                result.project_problems.append(f"{name}: {exc}")
                continue
            if not inner.ok:
                for problem in inner.changed + inner.missing + inner.unlisted + inner.problems:
                    result.project_problems.append(f"{name}: {problem}")
            else:
                result.ok_members.append(name)
        members_dir = self.path / MEMBERS_DIR
        if members_dir.is_dir():
            for candidate in sorted(members_dir.glob("*.odockproj")):
                relative = _project._as_posix(str(candidate.relative_to(self.path)))
                if relative not in listed:
                    result.unlisted.append(relative)
        sums_path = self.path / SUMS_NAME
        if sums_path.exists():
            recorded, _ = _project._read_sums(sums_path.read_text(encoding="utf-8"))
            manifest_digest = _project.sha256_bytes(self.manifest_path.read_bytes())
            if str(recorded.get(STUDY_MANIFEST) or "") != manifest_digest:
                result.sums_problems.append(
                    f"{STUDY_MANIFEST}: SHA256SUMS says "
                    f"{str(recorded.get(STUDY_MANIFEST) or '')[:12]}…, the file hashes "
                    f"to {manifest_digest[:12]}…"
                )
            for name, digest in recorded.items():
                if name == STUDY_MANIFEST:
                    continue
                target = self.path / name
                if not target.exists():
                    result.sums_problems.append(
                        f"{SUMS_NAME} lists {name}, which is not in the study"
                    )
                    continue
                found = _project.sha256_bytes(target.read_bytes())
                if found != digest:
                    result.sums_problems.append(
                        f"{name}: SHA256SUMS says {digest[:12]}…, the file hashes to "
                        f"{found[:12]}…"
                    )
        return result

    # -- reporting ---------------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": STUDY_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "title": self.title,
            "created_utc": self.created_utc,
            "audit": self.audit,
            "metadata": self.metadata,
            "protocol": self.protocol,
            "notes": self.notes,
            "n_members": len(self.members),
            "members": self.members,
        }

    def describe(self) -> str:
        lines = [
            f"{self.name}  ({len(self.members)} member(s)) in {self.path}",
            f"  title       : {self.title}",
            f"  created     : {self.created_utc}",
            f"  operator    : {self.audit.get('operator') or 'not recorded'}",
        ]
        metadata = self.metadata
        if metadata.get("target"):
            lines.append(f"  target      : {metadata['target']}")
        if metadata.get("library"):
            lines.append(f"  library     : {metadata['library']}")
        receptors = metadata.get("receptors")
        if receptors:
            lines.append("  receptors   : " + ", ".join(str(item) for item in receptors))
        if self.protocol:
            lines.append(
                "  protocol    : "
                + ", ".join(f"{key}={value}" for key, value in sorted(self.protocol.items()))
            )
        best = [
            member for member in self.members if member.get("best_affinity") is not None
        ]
        if best:
            top = min(best, key=lambda item: float(item["best_affinity"]))
            lines.append(
                f"  best member : {top.get('label')} at {top['best_affinity']:.3f} kcal/mol"
            )
        for member in self.members:
            mark = "" if self.member_path(member).exists() else "  (missing)"
            lines.append(
                f"    {str(member.get('label'))[:32]:<34} "
                f"{_project._fmt(member.get('best_affinity')):>8} kcal/mol  "
                f"poses {member.get('n_poses')}{mark}"
            )
        if self.notes:
            for note in self.notes:
                lines.append(f"  note        : {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Creating, loading and extending a study
# ---------------------------------------------------------------------------


def _write_manifest(study: Study) -> None:
    """Write ``study.json`` and ``SHA256SUMS`` atomically."""
    study.manifest["updated_utc"] = _now()
    payload = _project.normalise_paths(study.manifest, base=study.path)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
    temporary = study.manifest_path.with_name(STUDY_MANIFEST + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, study.manifest_path)

    lines = [
        f"{_project.sha256_bytes(study.manifest_path.read_bytes())}  {STUDY_MANIFEST}"
    ]
    for member in study.members:
        relative = str(member.get("project") or "")
        target = study.path / relative
        if target.exists():
            lines.append(f"{_project.sha256_bytes(target.read_bytes())}  {relative}")
    (study.path / SUMS_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_study(path: PathLike, *, verify: bool = False) -> Study:
    """Open a study directory.

    Only the manifest is parsed here; member projects are opened on demand.  With
    `verify=True` every member is re-hashed first and a :class:`StudyError` is
    raised when something changed.
    """
    directory = _study_directory(path)
    manifest_path = directory / STUDY_MANIFEST
    if not manifest_path.exists():
        raise StudyError(
            f"{directory} is not a study: it has no {STUDY_MANIFEST} "
            "(create one with `odock study create`)"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StudyError(f"{manifest_path} is not valid JSON: {exc}")
    if not isinstance(payload, Mapping):
        raise StudyError(f"{manifest_path} is not a JSON object")
    # The study schema is versioned exactly like a project's: refusing beats
    # misreading a layout that means something else.
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise StudyError(
            f"this study uses schema version {version!r}, and this build reads "
            f"{SCHEMA_VERSION}: upgrade OpenDocking, or ask for an export in schema "
            f"{SCHEMA_VERSION} (the schema version is the compatibility contract)."
        )
    if str(payload.get("format") or STUDY_FORMAT) != STUDY_FORMAT:
        raise StudyError(
            f"{manifest_path} declares format {payload.get('format')!r}, not "
            f"{STUDY_FORMAT!r}"
        )
    study = Study(directory, dict(payload))
    if verify:
        result = study.verify()
        if not result.ok:
            raise StudyError(
                "the study did not verify; it is not the collection it claims to be:\n"
                + result.summary()
            )
    return study


def create_study(
    directory: PathLike,
    *,
    name: Optional[str] = None,
    title: Optional[str] = None,
    target: Optional[str] = None,
    receptors: Optional[Sequence[str]] = None,
    library: Optional[str] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    notes: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    operator: Optional[str] = None,
    created: Optional[str] = None,
    members: Optional[Sequence[PathLike]] = None,
    copy_members: bool = True,
) -> Study:
    """Create (or open, and extend) a study.

    The shared metadata is what makes a study more than a directory listing:
    `target`, `receptors`, `library`, a free-form `protocol` mapping (force field,
    exhaustiveness, box, thresholds...), and `notes`.  Anything else passed in
    `metadata` is stored as it is.

    With `members` the projects are added immediately, which is what the command
    line does for `study create ... PROJECT...`.
    """
    root = _study_directory(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / STUDY_MANIFEST
    if manifest_path.exists():
        study = load_study(root)
        study.manifest["metadata"] = {**study.metadata, **{
            key: value for key, value in {
                "target": target, "receptors": list(receptors) if receptors else None,
                "library": library,
                "protocol": dict(protocol) if protocol else None,
                "notes": list(notes) if notes else None,
            }.items() if value is not None
        }}
        if metadata:
            study.manifest["metadata"].update(dict(metadata))
        if title:
            study.manifest["title"] = str(title)
        if operator:
            study.manifest.setdefault("audit", {})["operator"] = str(operator)
    else:
        resolved_metadata: Dict[str, Any] = {}
        for key, value in (
            ("target", target),
            ("receptors", list(receptors) if receptors else None),
            ("library", library),
            ("protocol", dict(protocol) if protocol else None),
            ("notes", list(notes) if notes else None),
        ):
            if value is not None:
                resolved_metadata[key] = value
        if metadata:
            resolved_metadata.update(dict(metadata))
        created_at = str(created or _now())
        study = Study(
            root,
            {
                "format": STUDY_FORMAT,
                "schema_version": SCHEMA_VERSION,
                "min_reader_version": SCHEMA_VERSION,
                "name": str(name or root.name),
                "title": str(title or name or root.name),
                "created_utc": created_at,
                "audit": {
                    "operator": (str(operator) if operator else None),
                    "tool_version": _project.tool_version(),
                    "kernel_version": _project._kernel_version(),
                    "python": _project.platform.python_version(),
                    "platform": _project._platform_label(),
                    "created_utc": created_at,
                },
                "metadata": resolved_metadata,
                "notes": [str(note) for note in (notes or ())],
                "members": [],
            },
        )
    if members:
        if not manifest_path.exists():
            _write_manifest(study)
        add_projects(root, members, copy=copy_members, operator=operator)
        return load_study(root)
    _write_manifest(study)
    return load_study(root)


def add_projects(
    directory: PathLike,
    projects: Sequence[PathLike],
    *,
    copy: bool = True,
    label: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
    operator: Optional[str] = None,
) -> Study:
    """Add project files to a study.

    With `copy=True` (the default) each project is copied into
    ``members/<name>.odockproj``, so the study is self-contained; with
    `copy=False` the manifest records a portable **relative** path to the project
    where it is, which keeps a large existing directory tree in place but means
    the study is only verifiable while that path resolves.
    """
    root = _study_directory(directory)
    study = load_study(root)
    members = study.members
    taken = {str(item.get("name")) for item in members}
    for index, source in enumerate(projects):
        path = Path(source)
        if not path.exists():
            raise StudyError(f"no such project: {path}")
        loaded = _project.open_project(path)
        verification = loaded.verify()
        wanted = None
        if labels is not None and index < len(labels):
            wanted = labels[index]
        wanted = wanted or label or path.stem
        name = _safe_name(wanted)
        base = name
        counter = 2
        while name in taken:
            name = f"{base}_{counter}"
            counter += 1
        taken.add(name)
        target = root / MEMBERS_DIR / f"{name}.odockproj"
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(path, target)
            stored = target
            relative = _project._as_posix(f"{MEMBERS_DIR}/{name}.odockproj")
            origin = _project.portable_path(path, base=root)
        else:
            # A linked member must resolve as ``study_dir / relative``, so the
            # recorded path is relative to the *study*, not to the working
            # directory: a path relative to the CWD would break the moment the
            # study is opened from somewhere else.
            try:
                linked = os.path.relpath(path.resolve(), root.resolve())
            except ValueError:
                raise StudyError(
                    f"cannot link {path}: it is on a different drive from the study; "
                    "use the copy mode (the default) instead"
                )
            relative = _project._as_posix(linked)
            stored = root / relative
            if not stored.exists():  # pragma: no cover - relpath is checked above
                raise StudyError(f"cannot resolve the linked member {path}")
            origin = _project.portable_path(path, base=root)
        summary = _summary_of(loaded)
        members.append(
            {
                "name": name,
                "label": str(wanted),
                "project": relative,
                "sha256": _project.sha256_bytes(stored.read_bytes()),
                "size_bytes": stored.stat().st_size,
                "source": origin,
                "copied": bool(copy),
                "added_utc": _now(),
                "added_by": (str(operator) if operator else study.audit.get("operator")),
                "verified_at_add": bool(verification.ok),
                **summary,
            }
        )
    study.manifest["members"] = members
    _write_manifest(study)
    return load_study(root)


def verify_study(path: PathLike) -> StudyVerifyResult:
    """Verify a study, reporting a refusal as a failed result rather than raising."""
    try:
        study = load_study(path)
    except StudyError as exc:
        return StudyVerifyResult(
            path=_study_directory(path), name=Path(path).name, n_members=0,
            problems=[str(exc)],
        )
    return study.verify()


# ---------------------------------------------------------------------------
# Diffing two studies
# ---------------------------------------------------------------------------


def _rank_members(members: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Rank members by best affinity (1 = best); unscored members are absent."""
    scored = [item for item in members if item.get("best_affinity") is not None]
    ordered = sorted(scored, key=lambda item: float(item["best_affinity"]))
    return {str(item.get("name")): index + 1 for index, item in enumerate(ordered)}


def _top(members: Sequence[Mapping[str, Any]], top: int) -> List[str]:
    ranked = _rank_members(members)
    return [
        name for name, rank in sorted(ranked.items(), key=lambda item: item[1])
        if rank <= int(top)
    ]


def _metadata_differences(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    """The shared metadata and protocol fields that differ, named with both values."""
    differences: List[Dict[str, Any]] = []

    def compare(field: str, left: Any, right: Any, kind: str) -> None:
        if left == right:
            return
        differences.append(
            {
                "field": field,
                "kind": kind,
                "values": [left, right],
                "note": (
                    "not recorded in one study" if left is None or right is None else ""
                ),
            }
        )

    for key in ("target", "library"):
        compare(key, first.get(key), second.get(key), "metadata")
    compare(
        "receptors",
        sorted(str(item) for item in (first.get("receptors") or [])),
        sorted(str(item) for item in (second.get("receptors") or [])),
        "metadata",
    )
    left_protocol = first.get("protocol") or {}
    right_protocol = second.get("protocol") or {}
    for key in sorted(set(left_protocol) | set(right_protocol)):
        compare(
            f"protocol.{key}",
            left_protocol.get(key),
            right_protocol.get(key),
            "protocol",
        )
    for key in sorted((set(first) | set(second)) - {"target", "library", "receptors", "protocol", "notes"}):
        compare(key, first.get(key), second.get(key), "metadata")
    return differences


def study_diff(
    first: PathLike,
    second: PathLike,
    *,
    top: int = 10,
) -> Dict[str, Any]:
    """Compare two studies: the protocol, the members, and the hit list.

    The hit-list section is quantitative by construction: for every member the
    two studies share it reports the best affinity on each side, its **rank**
    within its own study, the **rank delta**, and whether it was inside the
    stated top-N before and after.  Members that only one side has are listed as
    entered/left rather than silently dropped.
    """
    left = load_study(first)
    right = load_study(second)
    top = max(1, int(top))

    def index(study: Study) -> Dict[str, Dict[str, Any]]:
        return {str(member.get("name")): member for member in study.members}

    left_members = index(left)
    right_members = index(right)
    shared = sorted(set(left_members) & set(right_members))
    added = sorted(set(right_members) - set(left_members))
    removed = sorted(set(left_members) - set(right_members))

    left_rank = _rank_members(left.members)
    right_rank = _rank_members(right.members)
    left_top = set(_top(left.members, top))
    right_top = set(_top(right.members, top))

    hit_rows: List[Dict[str, Any]] = []
    pairs: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for side, study, index_map in (("a", left, left_members), ("b", right, right_members)):
        for name, member in index_map.items():
            runs.append(
                {
                    "name": f"{study.name}:{member.get('label')}",
                    "title": member.get("title"),
                    "study": side,
                    "member": member.get("label"),
                    "n_poses": member.get("n_poses"),
                    "best_affinity": member.get("best_affinity"),
                    "spread": None,
                    "seed": member.get("seed"),
                    "scoring": member.get("scoring"),
                    "ligand_efficiency": member.get("ligand_efficiency"),
                    "heavy_atoms": member.get("heavy_atoms"),
                    "key_residues": member.get("key_residues"),
                    "verify_ok": (study.path / str(member.get("project"))).exists(),
                    "engine": {"seed": member.get("seed"), "scoring": member.get("scoring")},
                    "inputs": [],
                    "clusters": {},
                    "does_not_establish": [],
                    "affinities": [],
                }
            )

    for name in shared:
        old = left_members[name]
        new = right_members[name]
        before = old.get("best_affinity")
        after = new.get("best_affinity")
        delta = None if before is None or after is None else float(after) - float(before)
        rank_before = left_rank.get(name)
        rank_after = right_rank.get(name)
        rank_delta = (
            None if rank_before is None or rank_after is None else rank_after - rank_before
        )
        if name in left_top and name in right_top:
            movement = "stayed in the top N"
        elif name in right_top:
            movement = f"entered the top {top}"
        elif name in left_top:
            movement = f"left the top {top}"
        elif delta is None:
            movement = "unscored on one side"
        elif delta < 0:
            movement = "improved"
        elif delta > 0:
            movement = "worsened"
        else:
            movement = "unchanged"
        hit_rows.append(
            {
                "member": new.get("label"),
                "name": name,
                "best_a": before,
                "best_b": after,
                "best_affinity_delta": delta,
                "rank_a": rank_before,
                "rank_b": rank_after,
                "rank_delta": rank_delta,
                "in_top_a": name in left_top,
                "in_top_b": name in right_top,
                "movement": movement,
            }
        )
        # The numbers behind a member's movement, computed by the comparison code
        # rather than re-derived here.
        left_path = left.member_path(old)
        right_path = right.member_path(new)
        if left_path.exists() and right_path.exists():
            try:
                compared = _project.compare_projects([left_path, right_path])
                pair = dict(compared["pairwise"][0])
                pair["a"] = f"{left.name}:{old.get('label')}"
                pair["b"] = f"{right.name}:{new.get('label')}"
                pair["member"] = new.get("label")
                pairs.append(pair)
            except _project.ProjectError as exc:
                hit_rows[-1]["problem"] = str(exc)

    top_rows = [
        {
            "member": right_members[name].get("label"),
            "name": name,
            "best_affinity": right_members[name].get("best_affinity"),
            "rank_b": right_rank.get(name),
            "was_in_top_a": name in left_top,
        }
        for name in _top(right.members, top)
    ]

    return {
        "format": "odock-study-diff",
        "version": 1,
        "created_utc": _now(),
        "top_n": top,
        "studies": {
            "a": {"name": left.name, "title": left.title, "path": left.path.name,
                  "created_utc": left.created_utc, "operator": left.audit.get("operator"),
                  "n_members": len(left.members), "metadata": left.metadata},
            "b": {"name": right.name, "title": right.title, "path": right.path.name,
                  "created_utc": right.created_utc, "operator": right.audit.get("operator"),
                  "n_members": len(right.members), "metadata": right.metadata},
        },
        "metadata_differences": _metadata_differences(left.metadata, right.metadata),
        "members": {
            "shared": [
                {
                    "name": name,
                    "label": right_members[name].get("label"),
                    "label_a": left_members[name].get("label"),
                    "label_b": right_members[name].get("label"),
                }
                for name in shared
            ],
            "added": [
                {"name": name, "label": right_members[name].get("label"),
                 "best_affinity": right_members[name].get("best_affinity"),
                 "rank_b": right_rank.get(name)}
                for name in added
            ],
            "removed": [
                {"name": name, "label": left_members[name].get("label"),
                 "best_affinity": left_members[name].get("best_affinity"),
                 "rank_a": left_rank.get(name)}
                for name in removed
            ],
        },
        "hit_list": {
            "top_n": top,
            "members": hit_rows,
            "top_a": _top(left.members, top),
            "top_b": _top(right.members, top),
            "entered_top": sorted(right_top - left_top),
            "left_top": sorted(left_top - right_top),
            "top_table": top_rows,
        },
        "comparison": {
            "format": "odock-comparison",
            "version": 1,
            "created_utc": _now(),
            "cutoff_angstrom": 2.0,
            "baseline": runs[0]["name"] if runs else "",
            "runs": runs,
            "pairwise": pairs,
            "problems": [],
            "note": (
                "Each pairwise row pairs the same member in the two studies, so a "
                "delta is that molecule's movement, not a comparison of two "
                "different molecules."
            ),
        },
        "does_not_establish": list(DIFF_LIMITS),
    }


def study_report(directory: PathLike, *, top: int = 10, reports: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The numbers a study report shows, as a JSON-able dictionary.

    `reports` are the file names of member reports that exist, resolved to
    relative links; when it is omitted the study directory is scanned for
    ``<label>.html`` and ``members/<label>.html``.
    """
    study = load_study(directory)
    verification = study.verify()
    ranked = sorted(
        study.members,
        key=lambda item: (
            item.get("best_affinity") is None,
            item.get("best_affinity") if item.get("best_affinity") is not None else 0.0,
        ),
    )
    entries: List[Dict[str, Any]] = []
    for position, member in enumerate(ranked, start=1):
        relative = str(member.get("project") or "")
        label = str(member.get("label") or member.get("name"))
        candidates = (
            [study.path / name for name in reports]
            if reports
            else [
                study.path / f"{label}.html",
                study.path / f"{_safe_name(label)}.html",
                study.path / MEMBERS_DIR / f"{label}.html",
                study.path / MEMBERS_DIR / f"{_safe_name(label)}.html",
            ]
        )
        report_href = ""
        for candidate in candidates:
            if candidate.exists():
                report_href = _project._as_posix(str(candidate.relative_to(study.path)))
                break
        entries.append(
            {
                "rank": position,
                "name": str(member.get("name")),
                "label": label,
                "title": member.get("title"),
                "project": relative,
                "best_affinity": member.get("best_affinity"),
                "n_poses": member.get("n_poses"),
                "seed": member.get("seed"),
                "scoring": member.get("scoring"),
                "heavy_atoms": member.get("heavy_atoms"),
                "ligand_efficiency": member.get("ligand_efficiency"),
                "key_residues": member.get("key_residues"),
                "sha256": member.get("sha256"),
                "size_bytes": member.get("size_bytes"),
                "added_utc": member.get("added_utc"),
                "added_by": member.get("added_by"),
                "verify_ok": str(member.get("name")) in verification.ok_members,
                "report_href": report_href,
                "has_report": bool(report_href),
            }
        )
    affinities = [
        float(entry["best_affinity"])
        for entry in entries
        if entry["best_affinity"] is not None
    ]
    protocol = study.protocol
    return {
        "format": "odock-study-report",
        "version": 1,
        "created_utc": _now(),
        "name": study.name,
        "title": study.title,
        "directory": _project.portable_path(study.path),
        "created_utc_study": study.created_utc,
        "audit": study.audit,
        "metadata": study.metadata,
        "protocol": protocol,
        "notes": study.notes,
        "top_n": int(top),
        "n_members": len(study.members),
        "n_verified": len(verification.ok_members),
        "verify": verification.as_dict(),
        "best_affinity": min(affinities) if affinities else None,
        "worst_affinity": max(affinities) if affinities else None,
        "entries": entries,
        "top": entries[: max(1, int(top))],
        "does_not_establish": list(DIFF_LIMITS),
    }


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _cli_err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _json_scalar(text: str) -> Any:
    """A command-line value as a JSON scalar when it looks like one.

    ``16`` becomes the integer 16, ``0.375`` the float, ``true`` the boolean, and
    anything else stays a string — so ``--protocol scoring=vina
    --protocol exhaustiveness=16`` records a number where a number was meant,
    which is what a diff needs to compare.
    """
    stripped = str(text).strip()
    lowered = stripped.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _pairs(values: Optional[Sequence[str]], *, what: str) -> Dict[str, Any]:
    """``["key=value", ...]`` -> ``{key: value}`` with JSON-scalar coercion."""
    found: Dict[str, Any] = {}
    for item in values or ():
        key, separator, value = str(item).partition("=")
        if not separator or not key.strip():
            raise SystemExit(f"error: --{what} expects KEY=VALUE, got {item!r}")
        found[key.strip()] = _json_scalar(value)
    return found


def _load_json(path: Optional[str], what: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: cannot read {what} {path}: {exc}")
    if not isinstance(payload, Mapping):
        raise SystemExit(f"error: {what} must hold a JSON object")
    return dict(payload)


def cmd_study_create(args) -> int:
    """Create a study, optionally with its first members."""
    try:
        protocol = _load_json(args.protocol_json, "--protocol-json")
        protocol.update(_pairs(args.protocol, what="protocol"))
        metadata = _load_json(args.metadata_json, "--metadata-json")
        metadata.update(_pairs(args.metadata, what="metadata"))
        study = create_study(
            args.directory,
            name=args.name,
            title=args.title,
            target=args.target,
            receptors=args.receptor,
            library=args.library,
            protocol=protocol,
            notes=args.note,
            metadata=metadata,
            operator=args.operator,
            members=args.project,
            copy_members=not args.link,
        )
    except StudyError as exc:
        _cli_err(f"odock study create: error: {exc}")
        return 2
    if args.json:
        print(json.dumps(study.as_dict(), indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(study.describe())
    _cli_err(f"wrote {study.manifest_path} ({len(study.members)} member(s))")
    return 0


def cmd_study_add(args) -> int:
    """Add project files to a study."""
    try:
        study = add_projects(
            args.directory,
            args.projects,
            copy=not args.link,
            label=args.label,
            operator=args.operator,
        )
    except StudyError as exc:
        _cli_err(f"odock study add: error: {exc}")
        return 2
    if args.json:
        print(json.dumps(study.as_dict(), indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(study.describe())
    _cli_err(f"{study.manifest_path}: {len(study.members)} member(s)")
    return 0


def cmd_study_verify(args) -> int:
    """Re-hash every member of a study and verify each project."""
    report = verify_study(args.directory)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    elif not args.quiet:
        print(report.summary())
    if not report.ok:
        _cli_err("odock study verify: the study did not verify")
    return 0 if report.ok else 1


def cmd_study_diff(args) -> int:
    """Compare two studies: protocol, members, and how the hit list moved."""
    from . import htmlreport as _htmlreport

    try:
        payload = study_diff(args.first, args.second, top=int(args.top))
    except StudyError as exc:
        _cli_err(f"odock study diff: error: {exc}")
        return 2
    written: List[str] = []
    if args.out:
        try:
            files = _htmlreport.write_study_diff(
                args.out, payload, title=args.title,
                json_out=args.json_out, text_out=args.text_out,
            )
        except (StudyError, ValueError) as exc:
            _cli_err(f"odock study diff: error: {exc}")
            return 2
        written = [str(files.path)]
        for path in (files.json_path, files.text_path):
            if path is not None:
                written.append(str(path))
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(_htmlreport.study_diff_text(payload))
    if written:
        _cli_err("wrote " + ", ".join(written))
    return 0


def cmd_study_report(args) -> int:
    """Write the report of a study."""
    from . import htmlreport as _htmlreport

    try:
        payload = study_report(args.directory, top=int(args.top))
    except (StudyError, _project.ProjectError) as exc:
        _cli_err(f"odock study report: error: {exc}")
        return 2
    written: List[str] = []
    if args.out:
        try:
            files = _htmlreport.write_study_report(
                args.out, payload, title=args.title, study_dir=args.directory,
                json_out=args.json_out, text_out=args.text_out,
            )
        except (StudyError, ValueError) as exc:
            _cli_err(f"odock study report: error: {exc}")
            return 2
        written = [str(files.path)]
        for path in (files.json_path, files.text_path):
            if path is not None:
                written.append(str(path))
    if args.json or not args.out:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(f"{payload['name']}: {payload['n_members']} member(s), "
              f"{payload['n_verified']} verified, best {_project._fmt(payload['best_affinity'])}")
        for entry in payload["entries"]:
            print(
                f"  {str(entry['label'])[:34]:<36} "
                f"{_project._fmt(entry['best_affinity']):>8} kcal/mol  "
                f"rank {entry['rank']:<3} seed {_project._shown(entry['seed']):<16} "
                f"{'OK' if entry['verify_ok'] else 'FAILED'}"
            )
    if written:
        _cli_err("wrote " + ", ".join(written))
    return 0 if payload["n_verified"] == payload["n_members"] else 1


def add_study_parser(sub) -> None:
    """Register ``odock study create|add|verify|diff|report`` on `sub`.

    Called once from :func:`odock.cli_ext.register_extensions`; idempotent, so a
    re-applied registration line is not an error.
    """
    choices = getattr(sub, "choices", None)
    if isinstance(choices, Mapping) and "study" in choices:
        return
    parser = sub.add_parser(
        "study",
        help="group several runs into a named, verifiable study",
        description=(
            "A study is a named collection of docking runs with shared metadata "
            "(target, receptors, library, protocol) and a manifest that records "
            "every member's SHA-256, so a directory of results can be verified and "
            "two versions of a series can be compared."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    def add_metadata_arguments(target) -> None:
        target.add_argument("--name", help="short study name (default: the directory name)")
        target.add_argument("--title", help="a human title")
        target.add_argument("--target", help="the biological target, e.g. 'EGFR kinase domain'")
        target.add_argument(
            "--receptor", action="append", metavar="NAME",
            help="a receptor of the set (repeatable)",
        )
        target.add_argument("--library", help="the library that was screened")
        target.add_argument(
            "--protocol", action="append", metavar="KEY=VALUE",
            help="a protocol field, e.g. --protocol scoring=vina "
                 "--protocol exhaustiveness=16 (repeatable; numbers and booleans "
                 "are stored as such)",
        )
        target.add_argument(
            "--protocol-json", metavar="FILE",
            help="a JSON object of protocol fields, read from a file",
        )
        target.add_argument(
            "--metadata", action="append", metavar="KEY=VALUE",
            help="any other metadata field (repeatable)",
        )
        target.add_argument(
            "--metadata-json", metavar="FILE",
            help="a JSON object of other metadata, read from a file",
        )
        target.add_argument("--note", action="append", help="a free-text note (repeatable)")
        target.add_argument("--operator", help="who created it, for the audit block")

    create = actions.add_parser(
        "create",
        help="create a study (and optionally add its first projects)",
        description=(
            "Create study.json and SHA256SUMS in a directory.  The shared metadata "
            "is what makes the collection a study rather than a folder; projects "
            "given with --project (repeatable) are added and copied in straight "
            "away."
        ),
    )
    create.add_argument("directory", help="the study directory (created if needed)")
    create.add_argument(
        "--project", action="append", metavar="FILE",
        help="a project file to add as a member (repeatable)",
    )
    create.add_argument(
        "--link", action="store_true",
        help="reference the projects in place instead of copying them in",
    )
    add_metadata_arguments(create)
    create.add_argument("--json", action="store_true", help="print the study as JSON")
    create.add_argument("-q", "--quiet", action="store_true", help="no summary")
    create.set_defaults(func=cmd_study_create)

    add = actions.add_parser("add", help="add projects to a study")
    add.add_argument("directory", help="the study directory")
    add.add_argument("projects", nargs="+", help="project files to add")
    add.add_argument("--label", help="a label for the member (single-project adds)")
    add.add_argument(
        "--link", action="store_true",
        help="reference the project in place instead of copying it in",
    )
    add.add_argument("--operator", help="who added it, for the audit block")
    add.add_argument("--json", action="store_true", help="print the study as JSON")
    add.add_argument("-q", "--quiet", action="store_true", help="no summary")
    add.set_defaults(func=cmd_study_add)

    verify = actions.add_parser(
        "verify",
        help="re-hash every member and verify each project",
        description=(
            "Re-hash every member project against the SHA-256 recorded when it was "
            "added, check SHA256SUMS, and verify each project's own contents.  A "
            "changed member is reported by name."
        ),
    )
    verify.add_argument("directory", help="the study directory")
    verify.add_argument("--json", action="store_true", help="print the report as JSON")
    verify.add_argument("-q", "--quiet", action="store_true", help="nothing on success")
    verify.set_defaults(func=cmd_study_verify)

    diff = actions.add_parser(
        "diff",
        help="compare two studies: protocol, members and the hit list",
        description=(
            "Report what changed between two studies: the metadata and protocol "
            "fields with both values, the members added and removed, and how the "
            "hit list moved (per-member best-affinity and rank deltas, and the "
            "molecules that entered or left the top N)."
        ),
    )
    diff.add_argument("first", help="the baseline study")
    diff.add_argument("second", help="the study to compare against it")
    diff.add_argument("--top", type=int, default=10, help="the N of the top-N movement")
    diff.add_argument("-o", "--out", help="write the diff as a self-contained HTML page")
    diff.add_argument("--json-out", help="machine-readable sibling (default: <out>.json)")
    diff.add_argument("--text-out", help="plain-text sibling (default: <out>.txt)")
    diff.add_argument("--title", help="title for the diff page")
    diff.add_argument("--json", action="store_true", help="print the diff as JSON")
    diff.add_argument("-q", "--quiet", action="store_true", help="no text table")
    diff.set_defaults(func=cmd_study_diff)

    report = actions.add_parser(
        "report",
        help="write the study as one self-contained page",
        description=(
            "One HTML page over the study: the protocol, every member with its "
            "verification state and key numbers, the top-N table, and links to the "
            "member reports that sit beside it."
        ),
    )
    report.add_argument("directory", help="the study directory")
    report.add_argument("-o", "--out", help="write the report as HTML")
    report.add_argument("--top", type=int, default=10, help="how many members the top table shows")
    report.add_argument("--json-out", help="machine-readable sibling (default: <out>.json)")
    report.add_argument("--text-out", help="plain-text sibling (default: <out>.txt)")
    report.add_argument("--title", help="title for the report page")
    report.add_argument("--json", action="store_true", help="print the report as JSON")
    report.add_argument("-q", "--quiet", action="store_true", help="no listing")
    report.set_defaults(func=cmd_study_report)

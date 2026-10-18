# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock service``: let other software drive OpenDocking.

The only way to use this toolchain from another program used to be `subprocess`,
parse stdout, and hope the JSON schema did not move.  That keeps a good tool out of
other people's pipelines — a web front end, a notebook farm, an ELN, a LIMS.  This
module is the layer that makes it a **contract** instead of a convention, and it is
a thin one: every operation delegates to the functions the CLI already calls.

Five properties are load-bearing, and each exists because of a way this goes wrong:

* **Versioned.**  Every response carries ``schema_version``, and a request that
  names a version this build does not speak is refused **with both numbers named**
  — the same rule as the project container.  A caller must never have to parse a log
  line to learn it was speaking to the wrong build.
* **State, not latency.**  A submission returns an id immediately; the work happens
  on a worker thread.  Polling is the design, not a workaround: a docking campaign
  takes hours and no sane caller holds a socket for that.
* **No second bookkeeping file.**  A job *is* its campaign directory: the rows
  `screen` writes and the summary it leaves are the state, so a restart reattaches
  by reading them (`screen.read_records`, ``ScreenSummary.summary_path``) instead of
  trusting a file this module made up.
* **Cancellation is the existing one.**  ``screen_ligands(config, stop=event)``'s
  ``stop`` **is** the event ``_Runner.stop()`` sets, so the kernel notices at its
  next Monte-Carlo step, the rows already written stay, and the summary comes back
  with ``interrupted``.  This module adds no second path.
* **Security is stated.**  Loopback by default, an explicit opt-in for anything
  else, a shared token checked on every request, and cross-origin requests refused:
  a local HTTP endpoint that can run arbitrary docking is otherwise drivable by any
  page in a browser.  It is **not** a hardened multi-tenant server — see
  ``docs/SERVICE.md``.

No framework: four endpoints do not justify a dependency the user pays for forever,
so the adapter is ``http.server`` and the contract is plain dicts.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import project as _project
from . import screen as _screen

__all__ = [
    "SERVICE_FORMAT",
    "SCHEMA_VERSION",
    "ServiceError",
    "Service",
    "Job",
    "add_service_parser",
    "cmd_service",
    "serve",
]

SERVICE_FORMAT = "odock-service"
SCHEMA_VERSION = 1

#: The protocol keys a caller may set, mapped to ``screen.ScreenConfig`` fields.
#: A whitelist on purpose: an unknown key is refused before any work starts, rather
#: than being silently dropped by ``ScreenConfig(**kwargs)``.
PROTOCOL_FIELDS: Dict[str, str] = {
    "scoring": "scoring",
    "exhaustiveness": "exhaustiveness",
    "num_poses": "num_poses",
    "seed": "seed",
    "jobs": "jobs",
    "timeout": "timeout",
    "limit": "limit",
    "min_rmsd": "min_rmsd",
    "energy_range": "energy_range",
    "consensus": "consensus",
    "interactions": "interactions",
}

#: Job states.  ``queued`` and ``running`` are live; the rest are final.
STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_COMPLETED = "completed"
STATE_CANCELLED = "cancelled"
STATE_FAILED = "failed"
LIVE_STATES = (STATE_QUEUED, STATE_RUNNING)


class ServiceError(RuntimeError):
    """A request that cannot be honoured, with a stable code and a fix.

    `code` is what a caller branches on: ``schema_version`` and ``bad_request`` are
    the caller's mistake, ``not_found`` and ``conflict`` are about state,
    ``unauthorised`` and ``forbidden`` are the security refusals.
    """

    def __init__(self, message: str, *, code: str = "error", fix: str = "",
                 status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.fix = fix
        self.status = status

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "format": SERVICE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": {"code": self.code, "message": str(self), "fix": self.fix},
        }
        return payload


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def _require(payload: Mapping[str, Any], name: str, kind: Any) -> Any:
    """A required field, refused **before** any work starts."""
    if name not in payload or payload[name] is None:
        raise ServiceError(
            f"the request has no {name!r}",
            code="bad_request",
            fix=f"send {name!r}; the schema endpoint lists every field",
        )
    value = payload[name]
    if not isinstance(value, kind):
        expected = getattr(kind, "__name__", str(kind))
        raise ServiceError(
            f"{name!r} must be {expected}, not {type(value).__name__}",
            code="bad_request",
            fix=f"send {name!r} as {expected}",
        )
    return value


def _schema(request: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {
        "format": SERVICE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "capabilities": {
            "submit": True,
            "cancel": True,
            "poll": True,
            "streaming": False,
            "resume_from_disk": True,
        },
        "endpoints": ["submit", "status", "results", "cancel", "list", "schema"],
        "protocol_fields": sorted(PROTOCOL_FIELDS),
        "request": {
            "submit": {
                "schema_version": SCHEMA_VERSION,
                "receptor": "a PDBQT (or PDB) path",
                "library": {"path": "a .smi/.sdf file"} ,
                "box": {"center": [0.0, 0.0, 0.0], "size": [20.0, 20.0, 20.0],
                        "spacing": 0.375},
                "protocol": {"scoring": "vina", "exhaustiveness": 8, "seed": 0},
            }
        },
        "not": ["a scheduler", "multi-tenant", "authenticated beyond a shared token"],
    }


def _check_version(request: Mapping[str, Any]) -> None:
    """Refuse an unknown schema version, naming both sides."""
    if "schema_version" not in request:
        raise ServiceError(
            f"the request does not name a schema version; this service speaks "
            f"{SCHEMA_VERSION}",
            code="schema_version",
            fix=f"add \"schema_version\": {SCHEMA_VERSION} to the request",
        )
    sent = request["schema_version"]
    if not isinstance(sent, int) or sent != SCHEMA_VERSION:
        raise ServiceError(
            f"this service speaks {SERVICE_FORMAT} schema version {SCHEMA_VERSION}, "
            f"and the request names {sent!r}",
            code="schema_version",
            fix=f"send \"schema_version\": {SCHEMA_VERSION}, or use a client built "
                f"for schema {sent!r}",
        )


# ---------------------------------------------------------------------------
# A job
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """One campaign: what was asked, where it lives, and what it has done."""

    id: str
    request: Dict[str, Any]
    outdir: Path
    receptor: Path
    library: Path
    state: str = STATE_QUEUED
    created_utc: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    error: str = ""
    summary_path: Optional[Path] = None
    results_path: Optional[Path] = None
    _stop: Optional[threading.Event] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    # -- state from the campaign, never from a second file -----------------------
    def records(self) -> List[_screen.LigandRecord]:
        """The rows on disk.  Polling reads these: they are the durable truth."""
        if not self.results_path or not Path(self.results_path).is_file():
            return []
        try:
            return _screen.read_records(self.results_path)
        except Exception:  # pragma: no cover - a half-written row
            return []

    def summary(self) -> Optional[Dict[str, Any]]:
        """The summary the campaign wrote, if it got that far."""
        path = self.summary_path
        if path is None or not Path(path).is_file():
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - a partial write
            return None

    def is_live(self) -> bool:
        return self.state in LIVE_STATES

    def as_dict(self, *, include_results: bool = False) -> Dict[str, Any]:
        rows = self.records()
        summary = self.summary()
        done = sum(1 for row in rows if getattr(row, "status", "") == "ok")
        failed = sum(1 for row in rows if getattr(row, "status", "") not in ("ok", "", None))
        payload: Dict[str, Any] = {
            "format": SERVICE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "job": {
                "id": self.id,
                "state": self.state,
                "created_utc": self.created_utc,
                "started_utc": self.started_utc,
                "finished_utc": self.finished_utc,
                "outdir": _project.portable_path(self.outdir),
                "error": self.error,
                "progress": {
                    "completed": done,
                    "failed": failed,
                    "rows": len(rows),
                    "interrupted": bool((summary or {}).get("interrupted", False)),
                    # The campaign's own counts, because a job can finish **completed with
                    # zero rows** and that has to explain itself. Measured on the CI Windows
                    # leg: `test_submit_poll_fetch_in_process` reported `assert 0 == 2` for a
                    # job whose state was "completed", and nothing in the payload said
                    # whether the two molecules were never read, were filtered at the library
                    # stage, or were docked and the rows were written somewhere else. The
                    # summary already knows; surfacing it turns a mysterious zero into a
                    # diagnosis.
                    "campaign": _campaign_counts(summary),
                },
            },
        }
        if include_results:
            payload["results"] = [_record(row) for row in rows]
            payload["summary"] = summary
        return payload


def _record(row: Any) -> Dict[str, Any]:
    """One result row, flat enough for any client to consume."""
    return {
        "name": getattr(row, "name", ""),
        "ligand": getattr(row, "ligand", ""),
        "status": getattr(row, "status", ""),
        "affinity": _maybe_float(getattr(row, "affinity", None)),
        "ligand_efficiency": _maybe_float(getattr(row, "ligand_efficiency", None)),
        "n_heavy": getattr(row, "n_heavy", None),
        "n_torsions": getattr(row, "n_torsions", None),
        "key_residues": getattr(row, "key_residues", ""),
        "n_interactions": getattr(row, "n_interactions", None),
        "in_box": getattr(row, "in_box", None),
        "pose_file": getattr(row, "pose_file", "") or "",
        "error": getattr(row, "error", "") or "",
    }


def _maybe_float(value: Any) -> Optional[float]:
    try:
        return None if value is None else round(float(value), 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class Service:
    """The in-process service: pure Python, no transport required.

    A transport (the HTTP adapter below) is a thin layer over these methods, and a
    test drives them directly — which is why the contract can be verified without a
    socket.
    """

    def __init__(self, root=None, *, outdir=None, token: str = "",
                 allow_remote: bool = False, echo: bool = False) -> None:
        self.root = Path(root).resolve() if root is not None else _default_root()
        self.outdir = Path(outdir).resolve() if outdir is not None else (
            self.root / "out" / "service"
        )
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.allow_remote = bool(allow_remote)
        self.echo = echo
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.resume_from_disk()

    # -- operations -------------------------------------------------------------
    def handle(self, operation: str, payload: Optional[Mapping[str, Any]] = None, *,
               token: str = "", origin: str = "", host: str = "") -> Dict[str, Any]:
        """The single entry point: dispatch by operation, then by job id.

        Everything a transport needs is here, including the security checks, so the
        HTTP adapter cannot accidentally be the only place that enforces them.
        """
        payload = dict(payload or {})
        try:
            self._authorise(token=token, origin=origin, host=host)
            if operation == "schema":
                return _schema(payload)
            _check_version(payload)
            if operation == "submit":
                return self.submit(payload)
            if operation == "list":
                return self.list()
            if operation in ("status", "results", "cancel"):
                job_id = _require(payload, "id", str)
                if operation == "status":
                    return self.status(job_id)
                if operation == "results":
                    return self.results(job_id)
                return self.cancel(job_id, reason=str(payload.get("reason", "")))
            raise ServiceError(
                f"unknown operation {operation!r}", code="bad_request",
                fix="one of submit, status, results, cancel, list, schema",
            )
        except ServiceError as error:
            return error.as_dict()

    # -- security ---------------------------------------------------------------
    def _authorise(self, *, token: str, origin: str, host: str) -> None:
        """Shared token, loopback, and no cross-origin driver.

        A local HTTP endpoint that can run arbitrary docking is otherwise drivable
        by any page a browser loads: the token is what makes a page's request fail,
        and the origin check is what makes a page's *browser* refuse to try.
        """
        if not self.allow_remote and host and not _is_loopback(host):
            raise ServiceError(
                f"this service only answers on the loopback interface, and {host!r} "
                "is not one of its names",
                code="forbidden", status=403,
                fix="bind 127.0.0.1 (the default), or start with --allow-remote and a "
                    "token if you really mean to expose it",
            )
        if origin and not _origin_allowed(origin):
            raise ServiceError(
                f"a request from origin {origin!r} is refused: this service accepts "
                "no cross-origin caller",
                code="forbidden", status=403,
                fix="call it from a server-side process; a browser page must not drive "
                    "a docking service on the same machine",
            )
        if self.token and token != self.token:
            raise ServiceError(
                "the request did not present the service token",
                code="unauthorised", status=401,
                fix="send it as the X-ODock-Token header (or ?token= for a client that "
                    "cannot set headers)",
            )

    # -- submit ----------------------------------------------------------------
    def submit(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        # Pure validation first: a malformed request is refused before this touches
        # the disk or queues anything.
        protocol = self._resolve_protocol(request.get("protocol") or {})
        box = self._resolve_box(_require(request, "box", (dict, str, list)))
        receptor = Path(str(_require(request, "receptor", str))).expanduser()
        if not receptor.is_file():
            raise ServiceError(
                f"the receptor does not exist: {_project.portable_path(receptor)}",
                code="bad_request", fix="send a path that exists on the service's machine",
            )
        library_spec = _require(request, "library", (dict, str))
        library = self._resolve_library(library_spec)

        job_id = uuid.uuid4().hex[:16]
        outdir = self.outdir / job_id
        outdir.mkdir(parents=True, exist_ok=True)
        job = Job(
            id=job_id,
            request=dict(request),
            outdir=outdir,
            receptor=receptor.resolve(),
            library=library,
            created_utc=_now(),
            results_path=outdir / "results.jsonl",
            summary_path=outdir / "summary.json",
        )
        job._stop = threading.Event()
        with self._lock:
            self._jobs[job_id] = job
        job._thread = threading.Thread(
            target=self._work, args=(job, box, protocol), name=f"odock-service-{job_id}",
            daemon=True,
        )
        job._thread.start()
        return {"format": SERVICE_FORMAT, "schema_version": SCHEMA_VERSION, "ok": True,
                "job": {"id": job_id, "state": job.state, "poll": job_id}}

    def _resolve_library(self, spec: Any) -> Path:
        """A library file, or an inline list of SMILES written into the campaign."""
        if isinstance(spec, str):
            path = Path(spec).expanduser()
            if not path.is_file():
                raise ServiceError(
                    f"the library does not exist: {_project.portable_path(path)}",
                    code="bad_request", fix="send a path that exists, or an inline list",
                )
            return path.resolve()
        if isinstance(spec, dict) and isinstance(spec.get("smiles"), list):
            members = [str(item) for item in spec["smiles"] if str(item).strip()]
            if not members:
                raise ServiceError(
                    "the inline library has no molecule", code="bad_request",
                    fix="send at least one SMILES string",
                )
            names = list(spec.get("names") or [])
            # Written into the job's own directory: the campaign directory is the
            # state, so an inline library has to live inside it.
            path = self.outdir / f"inline-{uuid.uuid4().hex[:8]}.smi"
            lines = [
                f"{smiles} {names[index] if index < len(names) else f'ligand{index + 1}'}"
                for index, smiles in enumerate(members)
            ]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return path
        raise ServiceError(
            "the library must be a path, or {\"smiles\": [...]}",
            code="bad_request", fix="see the schema endpoint",
        )

    def _resolve_box(self, spec: Any) -> _screen.Any:
        from . import BoxSpec  # noqa: F401 - the kernel's box type, re-exported by odock

        if isinstance(spec, str):
            path = Path(spec).expanduser()
            if not path.is_file():
                raise ServiceError(
                    f"the box file does not exist: {_project.portable_path(path)}",
                    code="bad_request", fix="send a path that exists",
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _box_from(payload)
        if isinstance(spec, list):
            return _box_from({"center": spec})
        if isinstance(spec, dict):
            return _box_from(spec)
        raise ServiceError("the box must be an object, a list or a path",
                           code="bad_request", fix="see the schema endpoint")

    def _resolve_protocol(self, protocol: Any) -> Dict[str, Any]:
        if not isinstance(protocol, dict):
            raise ServiceError("the protocol must be an object", code="bad_request")
        unknown = sorted(set(protocol) - set(PROTOCOL_FIELDS))
        if unknown:
            raise ServiceError(
                "unknown protocol field(s): " + ", ".join(unknown),
                code="bad_request",
                fix="one of " + ", ".join(sorted(PROTOCOL_FIELDS)),
            )
        return {PROTOCOL_FIELDS[key]: value for key, value in protocol.items()}

    # -- the work --------------------------------------------------------------
    def _work(self, job: Job, box: Any, protocol: Dict[str, Any]) -> None:
        job.state = STATE_RUNNING
        job.started_utc = _now()
        config_kwargs: Dict[str, Any] = {
            "receptors": [job.receptor],
            "inputs": [job.library],
            "box": box,
            "outdir": job.outdir,
            "resume": True,      # the campaign directory is the state
            "progress": False,
            "fmt": "jsonl",
        }
        config_kwargs.update(protocol)
        try:
            summary = _screen.screen_ligands(
                _screen.ScreenConfig(**config_kwargs), stop=job._stop
            )
            job.results_path = summary.results_path or job.results_path
            job.summary_path = summary.summary_path or job.summary_path
            interrupted = bool(getattr(summary, "interrupted", False))
            job.state = STATE_CANCELLED if interrupted else STATE_COMPLETED
        except ServiceError:  # pragma: no cover - defensive
            raise
        except Exception as exc:  # a failed campaign is a failed job, not a crash
            job.state = STATE_FAILED
            job.error = f"{type(exc).__name__}: {_project.redact_paths(str(exc))[:300]}"
        finally:
            job.finished_utc = _now()
            if self.echo:
                print(f"[{job.id}] {job.state}", file=sys.stderr)

    # -- queries ---------------------------------------------------------------
    def status(self, job_id: str) -> Dict[str, Any]:
        return self._job(job_id).as_dict()

    def results(self, job_id: str) -> Dict[str, Any]:
        job = self._job(job_id)
        if job.is_live():
            raise ServiceError(
                f"job {job_id} is still {job.state}; results are fetched when it "
                "finishes or is cancelled",
                code="conflict", status=409,
                fix="poll status until state is completed/cancelled/failed",
            )
        return job.as_dict(include_results=True)

    def list(self) -> Dict[str, Any]:
        with self._lock:
            jobs = list(self._jobs.values())
        return {
            "format": SERVICE_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "ok": True,
            "jobs": [
                {"id": job.id, "state": job.state, "created_utc": job.created_utc,
                 "outdir": _project.portable_path(job.outdir)}
                for job in sorted(jobs, key=lambda item: item.created_utc)
            ],
        }

    def cancel(self, job_id: str, *, reason: str = "") -> Dict[str, Any]:
        job = self._job(job_id)
        if not job.is_live():
            raise ServiceError(
                f"job {job_id} is already {job.state} and cannot be cancelled",
                code="conflict", status=409,
                fix="conflicts are final; fetch its results instead",
            )
        if job._stop is not None:
            # The *existing* cancellation path: `_Runner.stop()` sets this event, the
            # kernel notices at its next Monte-Carlo step, and the rows already
            # written stay on disk.
            job._stop.set()
        job.state = STATE_CANCELLED
        job.finished_utc = job.finished_utc or _now()
        return {"format": SERVICE_FORMAT, "schema_version": SCHEMA_VERSION, "ok": True,
                "job": {"id": job.id, "state": job.state,
                        "reason": reason or "cancelled by the caller"}}

    def wait(self, job_id: str, *, timeout: float = 30.0) -> Dict[str, Any]:
        """Block until a job finishes (in-process callers only; a test helper)."""
        job = self._job(job_id)
        deadline = time.monotonic() + float(timeout)
        while job.is_live() and time.monotonic() < deadline:
            time.sleep(0.02)
        return job.as_dict(include_results=not job.is_live())

    def _job(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ServiceError(
                f"no job {job_id!r}", code="not_found", status=404,
                fix="`list` shows the jobs this service knows about",
            )
        return job

    # -- restart ---------------------------------------------------------------
    def resume_from_disk(self) -> None:
        """Reattach to campaigns already on disk, without a second state file.

        A job *is* its directory: the id is the directory name, the rows are
        `screen`'s result file and the completion is its summary.  A campaign that
        was interrupted (the process died) has rows but no ``completed`` summary, so
        it is reported as ``cancelled`` — the rows it wrote are kept, and calling
        `submit` again for the same directory resumes it through ``resume=True``.
        """
        if not self.outdir.is_dir():
            return
        for entry in sorted(self.outdir.iterdir()):
            if not entry.is_dir() or not _is_job_id(entry.name):
                continue
            rows_path = entry / "results.jsonl"
            summary_path = entry / "summary.json"
            if not rows_path.is_file() and not summary_path.is_file():
                continue
            summary = None
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):  # pragma: no cover - a partial write
                    summary = None
            finished = bool(summary) and not summary.get("interrupted", False)
            job = Job(
                id=entry.name,
                request={"resumed_from_disk": True},
                outdir=entry,
                receptor=Path(),
                library=Path(),
                state=STATE_COMPLETED if finished else STATE_CANCELLED,
                created_utc=_mtime(entry),
                finished_utc=_mtime(entry),
                results_path=rows_path,
                summary_path=summary_path,
            )
            self._jobs[entry.name] = job


def _is_job_id(name: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{16}", name))


def _box_from(payload: Mapping[str, Any]):
    from . import BoxSpec

    if not isinstance(payload, Mapping):
        raise ServiceError("a box must be an object", code="bad_request")
    center = payload.get("center") or payload.get("centre")
    size = payload.get("size")
    if center is None or size is None:
        raise ServiceError(
            "a box needs a center and a size",
            code="bad_request",
            fix='{"center": [x, y, z], "size": [x, y, z], "spacing": 0.375}',
        )
    try:
        return BoxSpec(
            center=tuple(float(value) for value in center),
            size=tuple(float(value) for value in size),
            spacing=float(payload.get("spacing", 0.375)),
        )
    except (TypeError, ValueError) as exc:
        raise ServiceError(f"the box is not usable: {exc}", code="bad_request") from exc


def _is_loopback(host: str) -> bool:
    name = str(host or "").strip().lower()
    if name.startswith("[") and name.endswith("]"):
        name = name[1:-1]
    if name in ("localhost", "::1", "127.0.0.1", ""):
        return True
    return name.startswith("127.") or name.startswith("::ffff:127.")


def _origin_allowed(origin: str) -> bool:
    """Only a *missing* or same-origin-as-file request: no web page may drive this."""
    text = str(origin or "").strip().lower()
    if not text or text == "null":
        return False if text == "null" else True
    return text in ("file://",)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _mtime(path: Path) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime))
    except OSError:  # pragma: no cover
        return ""


def _default_root() -> Path:
    from .doctor import _default_root as doctor_root

    found = doctor_root()
    return found if found is not None else Path.cwd()


# ---------------------------------------------------------------------------
# The HTTP adapter: http.server, four endpoints, no framework
# ---------------------------------------------------------------------------

ENDPOINTS: Dict[str, str] = {
    "/v1/schema": "schema",
    "/v1/jobs": "list",
}


def make_handler(service: Service):
    """A ``http.server`` handler bound to one service (kept small on purpose)."""
    import http.server

    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = f"odock-service/{SCHEMA_VERSION}"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter by default
            if service.echo:
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -- plumbing -----------------------------------------------------------
        def _origin(self) -> str:
            return self.headers.get("Origin") or self.headers.get("Referer") or ""

        def _token(self) -> str:
            header = self.headers.get("X-ODock-Token", "")
            if header:
                return header
            query = self.path.partition("?")[2]
            for part in query.split("&"):
                name, _, value = part.partition("=")
                if name == "token":
                    return value
            return ""

        def _host(self) -> str:
            return self.headers.get("Host", "").split(":")[0]

        def _send(self, payload: Mapping[str, Any], *, status: int = 200) -> None:
            body = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-ODock-Schema-Version", str(SCHEMA_VERSION))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ServiceError(f"the body is not JSON: {exc}", code="bad_request") from exc
            if not isinstance(payload, dict):
                raise ServiceError("the body must be a JSON object", code="bad_request")
            return payload

        def _dispatch(self, operation: str, payload: Optional[Mapping[str, Any]] = None) -> None:
            answer = service.handle(
                operation, payload, token=self._token(),
                origin=self._origin(), host=self._host(),
            )
            status = 200 if answer.get("ok") else _status_for(answer)
            self._send(answer, status=status)

        # -- routes -------------------------------------------------------------
        def do_GET(self) -> None:
            path = self.path.partition("?")[0].rstrip("/") or "/"
            parts = [part for part in path.split("/") if part]
            if path == "/v1/schema":
                return self._dispatch("schema")
            if path == "/v1/jobs":
                return self._dispatch("list")
            if len(parts) == 3 and parts[:2] == ["v1", "jobs"]:
                return self._dispatch("status", {"schema_version": SCHEMA_VERSION,
                                                 "id": parts[2]})
            if len(parts) == 4 and parts[:2] == ["v1", "jobs"] and parts[3] == "results":
                return self._dispatch("results", {"schema_version": SCHEMA_VERSION,
                                                  "id": parts[2]})
            self._send({"format": SERVICE_FORMAT, "schema_version": SCHEMA_VERSION,
                        "ok": False, "error": {"code": "not_found",
                                               "message": f"no route {path}"}}, status=404)

        def do_POST(self) -> None:
            path = self.path.partition("?")[0].rstrip("/")
            if path != "/v1/jobs":
                self._send({"format": SERVICE_FORMAT, "schema_version": SCHEMA_VERSION,
                            "ok": False,
                            "error": {"code": "not_found",
                                      "message": f"no route {path}"}}, status=404)
                return
            try:
                payload = self._body()
            except ServiceError as error:
                return self._send(error.as_dict(), status=error.status)
            self._dispatch("submit", payload)

        def do_DELETE(self) -> None:
            parts = [part for part in self.path.partition("?")[0].split("/") if part]
            if len(parts) != 3 or parts[:2] != ["v1", "jobs"]:
                self._send({"format": SERVICE_FORMAT, "schema_version": SCHEMA_VERSION,
                            "ok": False,
                            "error": {"code": "not_found", "message": "no route"}},
                           status=404)
                return
            self._dispatch("cancel", {"schema_version": SCHEMA_VERSION, "id": parts[2]})

    return Handler


def _status_for(answer: Mapping[str, Any]) -> int:
    error = answer.get("error") or {}
    code = error.get("code")
    return {
        "schema_version": 400,
        "bad_request": 400,
        "unauthorised": 401,
        "forbidden": 403,
        "not_found": 404,
        "conflict": 409,
    }.get(code, 500)


def serve(directory_or_root=None, *, host: str = "127.0.0.1", port: int = 8765,
          token: str = "", allow_remote: bool = False, outdir=None,
          ready: Optional[Callable[[Any], None]] = None, quiet: bool = False) -> int:
    """Serve the contract on ``http.server`` until interrupted.

    Loopback only unless `allow_remote`; a non-loopback bind **requires** a token,
    because an unauthenticated docking service on a network interface is an open
    door with a queue in front of it.
    """
    import http.server
    import socketserver

    if not _is_loopback(host) and not allow_remote:
        raise ServiceError(
            f"refusing to bind {host!r} without --allow-remote",
            code="forbidden", status=403,
            fix="bind 127.0.0.1 (the default), or pass --allow-remote deliberately",
        )
    if not _is_loopback(host) and not token:
        raise ServiceError(
            "refusing to expose this service on a non-loopback interface without a token",
            code="forbidden", status=403,
            fix="pass --token (or set ODOCK_SERVICE_TOKEN) as well as --allow-remote",
        )
    service = Service(directory_or_root, outdir=outdir, token=token,
                      allow_remote=allow_remote, echo=not quiet)
    handler = make_handler(service)
    with socketserver.ThreadingTCPServer((host, port), handler) as server:
        server.allow_reuse_address = True
        actual = server.server_address[1]
        if not quiet:
            print(f"odock service on http://{host}:{actual}/v1/schema")
            print("loopback only: a shared token is checked on every request, and "
                  "cross-origin requests are refused.")
        if ready is not None:
            ready(server)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            if not quiet:
                print("\nstopped")
    return 0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def cmd_service(args) -> int:
    action = getattr(args, "action", "serve")
    token = getattr(args, "token", "") or os.environ.get("ODOCK_SERVICE_TOKEN", "")
    if action == "serve":
        try:
            return serve(
                getattr(args, "root", None), host=args.host, port=int(args.port),
                token=token, allow_remote=bool(args.allow_remote),
                outdir=getattr(args, "out", None),
            )
        except ServiceError as error:
            print(f"odock service: {error}", file=sys.stderr)
            return 2
    service = Service(getattr(args, "root", None), outdir=getattr(args, "out", None),
                      token=token, allow_remote=bool(args.allow_remote))
    payload: Dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if action == "submit":
        payload.update({
            "receptor": args.receptor, "library": args.library, "box": args.box,
            "protocol": json.loads(args.protocol) if args.protocol else {},
        })
    elif action in ("status", "results", "cancel", "list") and action != "list":
        payload["id"] = args.id
    answer = service.handle(action, payload, token=token)
    print(json.dumps(answer, indent=2, ensure_ascii=False))
    return 0 if answer.get("ok") else 1


def add_service_parser(sub) -> None:
    """Register ``odock service`` on `sub` (idempotent)."""
    choices = getattr(sub, "choices", None)
    if isinstance(choices, Mapping) and "service" in choices:
        return
    parser = sub.add_parser(
        "service",
        help="drive OpenDocking from another program (typed, versioned, loopback)",
        description=(
            "A typed request/response contract over the existing functions: submit a "
            "campaign, poll it, fetch its results, cancel it, list what exists. Every "
            "response carries the schema version and an unknown one is refused naming "
            "both numbers. Loopback by default, a shared token checked per request, "
            "cross-origin requests refused. Not a scheduler and not multi-tenant."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    serve_parser = actions.add_parser("serve", help="serve the contract on HTTP")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--token", help="the shared token (default: ODOCK_SERVICE_TOKEN)")
    serve_parser.add_argument("--allow-remote", action="store_true",
                              help="permit a non-loopback bind (a token is required)")
    serve_parser.add_argument("--root", help="the repository root")
    serve_parser.add_argument("--out", help="where campaigns live (default: out/service)")
    serve_parser.set_defaults(func=cmd_service)

    submit_parser = actions.add_parser("submit", help="submit a campaign, print its id")
    submit_parser.add_argument("--receptor", required=True)
    submit_parser.add_argument("--library", required=True)
    submit_parser.add_argument("--box", required=True,
                               help="a box.json path, or JSON {center,size,spacing}")
    submit_parser.add_argument("--protocol", help="JSON object of protocol fields")
    submit_parser.add_argument("--root")
    submit_parser.add_argument("--out")
    submit_parser.set_defaults(func=cmd_service)

    for name, text in (("status", "poll one job"), ("results", "fetch a job's results"),
                       ("cancel", "cancel a job")):
        sub_parser = actions.add_parser(name, help=text)
        sub_parser.add_argument("id")
        sub_parser.add_argument("--root")
        sub_parser.add_argument("--out")
        sub_parser.set_defaults(func=cmd_service)

    list_parser = actions.add_parser("list", help="list the jobs this service knows")
    list_parser.add_argument("--root")
    list_parser.add_argument("--out")
    list_parser.set_defaults(func=cmd_service)

#: Summary keys that explain a result, surfaced in a job's ``progress``. A completed job
#: with zero rows is otherwise indistinguishable from a broken write: the campaign's own
#: counts say whether the molecules were read, filtered, or written elsewhere.
_CAMPAIGN_COUNT_KEYS: Tuple[str, ...] = (
    "library",
    "library_size",
    "n_library",
    "molecules",
    "dockable",
    "prepared",
    "skipped",
    "filtered",
    "rejected",
    "failed",
    "written",
    "rows",
    "receptors",
    "interrupted",
)


def _campaign_counts(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The numeric part of a campaign summary, for a job's progress payload."""
    if not summary:
        return {}
    return {
        key: summary[key]
        for key in _CAMPAIGN_COUNT_KEYS
        if key in summary and isinstance(summary[key], (int, bool))
    }

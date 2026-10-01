# SPDX-License-Identifier: GPL-3.0-or-later
"""A protocol: every decision a run depends on, as a versioned document.

The gap this closes
-------------------
A working pipeline used to live in a shell history. ``prepare`` flags, the box
and how it was derived, engine and search settings, the seed, the filters and
their thresholds were typed once, worked, and were then unrecorded — so a lab that
found settings that work could not hand them to a collaborator, and could not tell
whether a colleague's run used the same protocol.

A *protocol* is that set of decisions as one reviewable JSON document, with three
properties that make it shareable:

* **schema-versioned and refused, not guessed.** A document written by a newer
  build is refused naming both versions, exactly like the project container
  (``odock.project``). Misreading a newer layout would silently change what a run
  means, which is what a version number exists to prevent.
* **validated on load.** A field this build does not know is refused *by name*
  instead of being ignored: a protocol that quietly loses a setting is worse than
  one that fails, because the run still completes and the numbers are attributed
  to settings that were not used.
* **diffable.** Two protocols are compared field by field with **both** values,
  in the shape ``study diff`` already established
  (:func:`odock.study.study_diff`), so "what changed between your run and mine?"
  has an answer that is a sentence rather than a diff of two JSON files.

What a protocol is *not* — the limit, stated where the code is
-------------------------------------------------------------
A protocol captures **settings, not intent or provenance**. It cannot tell you
that the receptor was the right one, that the box covers the site you care about,
or that the pose is chemically sensible. Its **hash proves identity, not
correctness**: two runs whose recorded hash matches were asked for the same
things, which is a precondition for comparing their numbers and nothing more. The
docs repeat this because it is the failure mode of every "reproducible
pipeline" file ever shipped.

Reproducibility, precisely
--------------------------
:meth:`Protocol.hash` covers every recorded **setting**: the preparation flags,
the box (its numbers and how it was derived), the engine and search settings, the
seed, the library/filter switches and the thresholds, under the schema version
that defines what those fields mean.

It deliberately does **not** cover: the protocol's ``name`` and ``note`` (labels,
not decisions — two templates with identical settings hash alike, which is
correct), the ``provenance`` block (tool versions, python version, creation time —
those describe the machine, not the decision), and any **path** (a path is a
location, not a decision; the inputs' identity is recorded separately by the run's
own input digests). Two labs on two machines with the same settings therefore
compute the same hash, which is the whole point.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PROTOCOL_KIND",
    "PROTOCOL_SCHEMA_VERSION",
    "MIN_SUPPORTED_VERSION",
    "SUPPORTED_VERSIONS",
    "ProtocolError",
    "ProtocolVersionError",
    "ProtocolFieldError",
    "Protocol",
    "ProtocolRun",
    "as_protocol",
    "capture_session",
    "diff_protocols",
    "load_protocol",
    "list_templates",
    "protocol_diff_text",
    "protocol_text",
    "run_protocol",
    "save_protocol",
    "template_dir",
    "template_named",
    "verify_run",
    "add_protocol_parser",
]

#: What a protocol document says it is, so a stray JSON file is refused early.
PROTOCOL_KIND = "opendocking-protocol"

#: The layout this build writes.  The version is the compatibility contract, not
#: the file extension: a reader must refuse what it cannot interpret.
PROTOCOL_SCHEMA_VERSION = 1

#: The oldest version this build still understands.  There is one so far; the
#: constant exists so the refusal path is testable from the first release.
MIN_SUPPORTED_VERSION = 1

SUPPORTED_VERSIONS: Tuple[int, ...] = tuple(
    range(MIN_SUPPORTED_VERSION, PROTOCOL_SCHEMA_VERSION + 1)
)

#: Where the shipped templates live, relative to the checkout root.  ``examples/``
#: is where this project already keeps reviewable material a user is meant to
#: open, and it is in the sdist include list, so the templates travel with a
#: source release (see docs/PROTOCOLS.md for the packaging note).
TEMPLATE_PREFIX = "protocol-"


class ProtocolError(RuntimeError):
    """A protocol could not be read, validated or run."""


class ProtocolVersionError(ProtocolError):
    """The protocol's schema version is not one this build can honour.

    Raised with **both** numbers in the message, like
    :class:`odock.project.SchemaVersionError`, so the reader knows whether to
    upgrade OpenDocking or ask for a re-export.
    """


class ProtocolFieldError(ProtocolError):
    """A field is unknown, missing or invalid — named in the message.

    The field name is the point: "protocol.engine.exhaustivness" tells a user
    what to fix, where "invalid protocol" sends them to the JSON to hunt.
    """


# ---------------------------------------------------------------------------
# Serialisation helpers: explicit key lists, so an unknown field is detectable
# ---------------------------------------------------------------------------


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolFieldError(
            f"{where} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _check_keys(
    payload: Mapping[str, Any], allowed: Iterable[str], where: str
) -> None:
    """Refuse an unknown field by name.

    This is the "validated on load" half of the contract: a protocol naming a
    setting that no longer exists (or that a newer build added) is refused rather
    than silently ignored, because a dropped setting still runs and still produces
    numbers — attributed to settings that were not used.
    """
    known = set(allowed)
    unknown = sorted(str(key) for key in payload if key not in known)
    if unknown:
        raise ProtocolFieldError(
            f"{where} has no field {unknown[0]!r}"
            + (f" (also unknown: {', '.join(repr(k) for k in unknown[1:])})" if len(unknown) > 1 else "")
            + f"; known fields: {', '.join(sorted(known))}"
        )


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolFieldError(f"{where} must be a number, got {value!r}")
    return float(value)


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolFieldError(f"{where} must be a whole number, got {value!r}")
    return int(value)


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolFieldError(f"{where} must be true or false, got {value!r}")
    return bool(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ProtocolFieldError(f"{where} must be a string, got {value!r}")
    return value


def _string_list(value: Any, where: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ProtocolFieldError(f"{where} must be a list of strings, got {value!r}")
    return [_text(item, where) for item in value]


def _triple(value: Any, where: str) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ProtocolFieldError(f"{where} must be three numbers, got {value!r}")
    return tuple(_number(item, where) for item in value)  # type: ignore[return-value]


def _plain_dict(value: Any, where: str) -> Dict[str, Any]:
    if value is None:
        return {}
    mapping = _require_mapping(value, where)
    return {str(key): item for key, item in mapping.items()}


#: The interaction thresholds a run profiles a pose with.  Kept here as the
#: documented default set; ``odock.analysis`` owns the meaning of each number.
DEFAULT_INTERACTION_THRESHOLDS: Dict[str, float] = {
    "hbond": 3.5,
    "salt": 4.0,
    "hydrophobic": 4.0,
    "pi": 4.5,
    "cation_pi": 5.0,
    "clash_ratio": 0.75,
}

#: The modified amino acids ``prepare_receptor(keep_hetero=False)`` keeps.  Kept as
#: a *recorded* set rather than an imported private name: a protocol has to say
#: which semantics it was written against, and importing a private constant would
#: make the document silently change meaning when that constant does.
DEFAULT_MODIFIED_RESIDUES: Tuple[str, ...] = (
    "MSE", "SEP", "TPO", "PTR", "CSO", "KCX", "MLY", "M3L", "HYP", "PCA",
)


# ---------------------------------------------------------------------------
# The document, section by section
# ---------------------------------------------------------------------------


@dataclass
class LigandPreparation:
    """How library molecules are prepared (``prepare_ligand``'s flags)."""

    add_hydrogens: bool = True
    strip_nonpolar: bool = True
    rigid_amides: bool = True
    embed: bool = True
    optimize: bool = True

    KEYS = ("add_hydrogens", "strip_nonpolar", "rigid_amides", "embed", "optimize")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "add_hydrogens": bool(self.add_hydrogens),
            "strip_nonpolar": bool(self.strip_nonpolar),
            "rigid_amides": bool(self.rigid_amides),
            "embed": bool(self.embed),
            "optimize": bool(self.optimize),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "LigandPreparation":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        return cls(
            add_hydrogens=_boolean(payload.get("add_hydrogens", True), f"{where}.add_hydrogens"),
            strip_nonpolar=_boolean(payload.get("strip_nonpolar", True), f"{where}.strip_nonpolar"),
            rigid_amides=_boolean(payload.get("rigid_amides", True), f"{where}.rigid_amides"),
            embed=_boolean(payload.get("embed", True), f"{where}.embed"),
            optimize=_boolean(payload.get("optimize", True), f"{where}.optimize"),
        )


@dataclass
class Preparation:
    """Receptor and ligand preparation, including the ``--no-hetero`` semantics.

    ``keep_hetero`` (the CLI's ``--no-hetero`` inverts it) drops ligand-like
    non-standard residues while **keeping** :attr:`modified_residues`, because
    those are part of the chain: removing one leaves a chemically wrong receptor
    with a hole where a residue was.  The set is recorded in the protocol so a
    reader can see which semantics the document was written against.
    """

    keep_water: bool = False
    keep_hetero: bool = True
    strip: List[str] = field(default_factory=list)
    add_polar_hydrogens: bool = True
    strict: bool = False
    modified_residues: List[str] = field(default_factory=lambda: list(DEFAULT_MODIFIED_RESIDUES))
    ligand: LigandPreparation = field(default_factory=LigandPreparation)
    prepare_seed: int = 20240101

    KEYS = (
        "keep_water",
        "keep_hetero",
        "strip",
        "add_polar_hydrogens",
        "strict",
        "modified_residues",
        "ligand",
        "prepare_seed",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep_water": bool(self.keep_water),
            "keep_hetero": bool(self.keep_hetero),
            "strip": [str(name) for name in self.strip],
            "add_polar_hydrogens": bool(self.add_polar_hydrogens),
            "strict": bool(self.strict),
            "modified_residues": [str(name) for name in self.modified_residues],
            "ligand": self.ligand.to_dict(),
            "prepare_seed": int(self.prepare_seed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "Preparation":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        return cls(
            keep_water=_boolean(payload.get("keep_water", False), f"{where}.keep_water"),
            keep_hetero=_boolean(payload.get("keep_hetero", True), f"{where}.keep_hetero"),
            strip=_string_list(payload.get("strip"), f"{where}.strip"),
            add_polar_hydrogens=_boolean(
                payload.get("add_polar_hydrogens", True), f"{where}.add_polar_hydrogens"
            ),
            strict=_boolean(payload.get("strict", False), f"{where}.strict"),
            modified_residues=_string_list(
                payload.get("modified_residues", list(DEFAULT_MODIFIED_RESIDUES)),
                f"{where}.modified_residues",
            ),
            ligand=LigandPreparation.from_dict(
                payload.get("ligand", {}), f"{where}.ligand"
            ),
            prepare_seed=_integer(payload.get("prepare_seed", 20240101), f"{where}.prepare_seed"),
        )


#: How a box was arrived at.  Recorded because "the same numbers" reached by
#: different routes are different decisions: a box fitted to the ligand follows it
#: to a new pose, a hand-placed one does not.
BOX_SOURCES = ("ligand", "pocket", "manual", "receptor")


@dataclass
class BoxChoice:
    """The search box and, importantly, **how it was derived**."""

    source: str = "ligand"
    center: Optional[Tuple[float, float, float]] = None
    size: Optional[Tuple[float, float, float]] = None
    spacing: float = 0.375
    ligand_padding: float = 8.0
    pocket_index: Optional[int] = None

    KEYS = ("source", "center", "size", "spacing", "ligand_padding", "pocket_index")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "center": None if self.center is None else [float(v) for v in self.center],
            "size": None if self.size is None else [float(v) for v in self.size],
            "spacing": float(self.spacing),
            "ligand_padding": float(self.ligand_padding),
            "pocket_index": None if self.pocket_index is None else int(self.pocket_index),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "BoxChoice":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        source = _text(payload.get("source", "ligand"), f"{where}.source")
        if source not in BOX_SOURCES:
            raise ProtocolFieldError(
                f"{where}.source is {source!r}, which is not one of "
                + ", ".join(repr(item) for item in BOX_SOURCES)
            )
        pocket = payload.get("pocket_index")
        return cls(
            source=source,
            center=_triple(payload.get("center"), f"{where}.center"),
            size=_triple(payload.get("size"), f"{where}.size"),
            spacing=_number(payload.get("spacing", 0.375), f"{where}.spacing"),
            ligand_padding=_number(
                payload.get("ligand_padding", 8.0), f"{where}.ligand_padding"
            ),
            pocket_index=None if pocket is None else _integer(pocket, f"{where}.pocket_index"),
        )


@dataclass
class Engine:
    """Scoring, search and the refinement switch."""

    scoring: str = "vina"
    search: Optional[str] = None
    exhaustiveness: int = 8
    num_poses: int = 9
    min_rmsd: float = 1.0
    energy_range: float = 3.0
    islands: int = 4
    population: int = 32
    generations: int = 20
    use_grid: bool = True
    refine: bool = True

    KEYS = (
        "scoring",
        "search",
        "exhaustiveness",
        "num_poses",
        "min_rmsd",
        "energy_range",
        "islands",
        "population",
        "generations",
        "use_grid",
        "refine",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scoring": str(self.scoring),
            "search": None if self.search is None else str(self.search),
            "exhaustiveness": int(self.exhaustiveness),
            "num_poses": int(self.num_poses),
            "min_rmsd": float(self.min_rmsd),
            "energy_range": float(self.energy_range),
            "islands": int(self.islands),
            "population": int(self.population),
            "generations": int(self.generations),
            "use_grid": bool(self.use_grid),
            "refine": bool(self.refine),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "Engine":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        search = payload.get("search")
        return cls(
            scoring=_text(payload.get("scoring", "vina"), f"{where}.scoring"),
            search=None if search is None else _text(search, f"{where}.search"),
            exhaustiveness=_integer(payload.get("exhaustiveness", 8), f"{where}.exhaustiveness"),
            num_poses=_integer(payload.get("num_poses", 9), f"{where}.num_poses"),
            min_rmsd=_number(payload.get("min_rmsd", 1.0), f"{where}.min_rmsd"),
            energy_range=_number(payload.get("energy_range", 3.0), f"{where}.energy_range"),
            islands=_integer(payload.get("islands", 4), f"{where}.islands"),
            population=_integer(payload.get("population", 32), f"{where}.population"),
            generations=_integer(payload.get("generations", 20), f"{where}.generations"),
            use_grid=_boolean(payload.get("use_grid", True), f"{where}.use_grid"),
            refine=_boolean(payload.get("refine", True), f"{where}.refine"),
        )


@dataclass
class Execution:
    """Seed, parallelism and the checkpoint cadence."""

    seed: int = 0
    jobs: int = 0
    timeout: Optional[float] = None
    checkpoint_every: int = 20

    KEYS = ("seed", "jobs", "timeout", "checkpoint_every")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "jobs": int(self.jobs),
            "timeout": None if self.timeout is None else float(self.timeout),
            "checkpoint_every": int(self.checkpoint_every),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "Execution":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        timeout = payload.get("timeout")
        return cls(
            seed=_integer(payload.get("seed", 0), f"{where}.seed"),
            jobs=_integer(payload.get("jobs", 0), f"{where}.jobs"),
            timeout=None if timeout is None else _number(timeout, f"{where}.timeout"),
            checkpoint_every=_integer(
                payload.get("checkpoint_every", 20), f"{where}.checkpoint_every"
            ),
        )


@dataclass
class Library:
    """Library-level switches: filters, output shape, post-processing."""

    filters: bool = True
    limit: Optional[int] = None
    top: int = 0
    format: str = "jsonl"
    interactions: bool = True
    write_poses: bool = True
    consensus: bool = False
    consensus_top: int = 0
    consensus_method: str = "rank"

    KEYS = (
        "filters",
        "limit",
        "top",
        "format",
        "interactions",
        "write_poses",
        "consensus",
        "consensus_top",
        "consensus_method",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filters": bool(self.filters),
            "limit": None if self.limit is None else int(self.limit),
            "top": int(self.top),
            "format": str(self.format),
            "interactions": bool(self.interactions),
            "write_poses": bool(self.write_poses),
            "consensus": bool(self.consensus),
            "consensus_top": int(self.consensus_top),
            "consensus_method": str(self.consensus_method),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "Library":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        limit = payload.get("limit")
        return cls(
            filters=_boolean(payload.get("filters", True), f"{where}.filters"),
            limit=None if limit is None else _integer(limit, f"{where}.limit"),
            top=_integer(payload.get("top", 0), f"{where}.top"),
            format=_text(payload.get("format", "jsonl"), f"{where}.format"),
            interactions=_boolean(payload.get("interactions", True), f"{where}.interactions"),
            write_poses=_boolean(payload.get("write_poses", True), f"{where}.write_poses"),
            consensus=_boolean(payload.get("consensus", False), f"{where}.consensus"),
            consensus_top=_integer(payload.get("consensus_top", 0), f"{where}.consensus_top"),
            consensus_method=_text(
                payload.get("consensus_method", "rank"), f"{where}.consensus_method"
            ),
        )


@dataclass
class Thresholds:
    """The numbers that decide what counts as an interaction or a pocket.

    Interaction thresholds default to :data:`DEFAULT_INTERACTION_THRESHOLDS`; the
    pocket and strain blocks are free-form records owned by their modules
    (``odock.pockets``, ``odock.metrics``).  They are captured verbatim: a
    protocol must be able to say "these were the numbers", including for a
    threshold this build does not itself interpret.
    """

    interactions: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_INTERACTION_THRESHOLDS)
    )
    pocket: Dict[str, Any] = field(default_factory=dict)
    strain: Dict[str, Any] = field(default_factory=dict)

    KEYS = ("interactions", "pocket", "strain")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interactions": {
                str(key): float(value) for key, value in sorted(self.interactions.items())
            },
            "pocket": dict(sorted(self.pocket.items())),
            "strain": dict(sorted(self.strain.items())),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "Thresholds":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)
        interactions = _plain_dict(payload.get("interactions"), f"{where}.interactions")
        return cls(
            interactions={
                str(key): _number(value, f"{where}.interactions.{key}")
                for key, value in interactions.items()
            },
            pocket=_plain_dict(payload.get("pocket"), f"{where}.pocket"),
            strain=_plain_dict(payload.get("strain"), f"{where}.strain"),
        )


@dataclass
class Provenance:
    """Which build produced the document.  Excluded from the hash on purpose."""

    tool: Optional[str] = None
    kernel: Optional[str] = None
    python: Optional[str] = None
    created: Optional[str] = None

    KEYS = ("tool", "kernel", "python", "created")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": None if self.tool is None else str(self.tool),
            "kernel": None if self.kernel is None else str(self.kernel),
            "python": None if self.python is None else str(self.python),
            "created": None if self.created is None else str(self.created),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], where: str) -> "Provenance":
        payload = _require_mapping(payload, where)
        _check_keys(payload, cls.KEYS, where)

        def optional(key: str) -> Optional[str]:
            value = payload.get(key)
            return None if value is None else _text(value, f"{where}.{key}")

        return cls(
            tool=optional("tool"),
            kernel=optional("kernel"),
            python=optional("python"),
            created=optional("created"),
        )


# ---------------------------------------------------------------------------
# The protocol itself
# ---------------------------------------------------------------------------


@dataclass
class Protocol:
    """One shareable set of decisions, with the version that defines them."""

    name: str = "protocol"
    note: str = ""
    schema_version: int = PROTOCOL_SCHEMA_VERSION
    kind: str = PROTOCOL_KIND
    preparation: Preparation = field(default_factory=Preparation)
    box: BoxChoice = field(default_factory=BoxChoice)
    engine: Engine = field(default_factory=Engine)
    execution: Execution = field(default_factory=Execution)
    library: Library = field(default_factory=Library)
    thresholds: Thresholds = field(default_factory=Thresholds)
    provenance: Provenance = field(default_factory=Provenance)

    KEYS = (
        "kind",
        "schema_version",
        "name",
        "note",
        "preparation",
        "box",
        "engine",
        "execution",
        "library",
        "thresholds",
        "provenance",
    )

    #: The sections the hash covers.  Everything a *run* depends on; not the
    #: labels, not the machine, not the paths.
    HASHED_SECTIONS = (
        "preparation",
        "box",
        "engine",
        "execution",
        "library",
        "thresholds",
    )

    #: The sections the hash deliberately excludes, with the reason, so the
    #: report can print it rather than leaving a reader to guess.
    UNHASHED_SECTIONS = {
        "name": "a label, not a decision",
        "note": "a comment for humans",
        "kind": "the document type marker",
        "provenance": "describes the machine and the moment, not the decision",
    }

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The document, with sections in a stable order."""
        return {
            "kind": str(self.kind),
            "schema_version": int(self.schema_version),
            "name": str(self.name),
            "note": str(self.note),
            "preparation": self.preparation.to_dict(),
            "box": self.box.to_dict(),
            "engine": self.engine.to_dict(),
            "execution": self.execution.to_dict(),
            "library": self.library.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Protocol":
        """Read a protocol, refusing anything this build cannot honour."""
        payload = _require_mapping(payload, "protocol")
        kind = payload.get("kind", PROTOCOL_KIND)
        if kind != PROTOCOL_KIND:
            raise ProtocolFieldError(
                f"this is not an OpenDocking protocol: kind is {kind!r}, expected "
                f"{PROTOCOL_KIND!r}"
            )
        version = payload.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ProtocolFieldError(
                "protocol.schema_version must be a whole number "
                f"(got {version!r}); the schema version is the compatibility contract"
            )
        if version > PROTOCOL_SCHEMA_VERSION:
            raise ProtocolVersionError(
                f"this protocol was written with schema version {version}, and this "
                f"build of OpenDocking reads up to {PROTOCOL_SCHEMA_VERSION}. Upgrade "
                "OpenDocking, or ask the author for an export in schema "
                f"{PROTOCOL_SCHEMA_VERSION} (the schema version is the compatibility "
                "contract)."
            )
        if version < MIN_SUPPORTED_VERSION:
            raise ProtocolVersionError(
                f"this protocol uses schema version {version}, which predates the "
                f"oldest version this build understands ({MIN_SUPPORTED_VERSION}); "
                "it is refused rather than misread."
            )
        _check_keys(payload, cls.KEYS, "protocol")
        return cls(
            kind=PROTOCOL_KIND,
            schema_version=int(version),
            name=_text(payload.get("name", "protocol"), "protocol.name"),
            note=_text(payload.get("note", ""), "protocol.note"),
            preparation=Preparation.from_dict(
                payload.get("preparation", {}), "protocol.preparation"
            ),
            box=BoxChoice.from_dict(payload.get("box", {}), "protocol.box"),
            engine=Engine.from_dict(payload.get("engine", {}), "protocol.engine"),
            execution=Execution.from_dict(
                payload.get("execution", {}), "protocol.execution"
            ),
            library=Library.from_dict(payload.get("library", {}), "protocol.library"),
            thresholds=Thresholds.from_dict(
                payload.get("thresholds", {}), "protocol.thresholds"
            ),
            provenance=Provenance.from_dict(
                payload.get("provenance", {}), "protocol.provenance"
            ),
        )

    # -- identity -----------------------------------------------------------

    def hashed_settings(self) -> Dict[str, Any]:
        """Exactly what the hash covers, as a plain dictionary."""
        document = self.to_dict()
        return {
            "schema_version": int(self.schema_version),
            **{section: document[section] for section in self.HASHED_SECTIONS},
        }

    def hash(self) -> str:
        """The identity of these **settings** (see the module docstring).

        Canonical form: sorted keys, no whitespace, ASCII only — so two labs on
        two machines that recorded the same decisions compute the same digest.
        """
        canonical = json.dumps(
            self.hashed_settings(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def short_hash(self) -> str:
        return self.hash()[:12]

    def hash_note(self) -> str:
        """One sentence naming what the hash covers and what it does not."""
        covered = ", ".join(self.HASHED_SECTIONS)
        excluded = "; ".join(
            f"{name} ({reason})" for name, reason in sorted(self.UNHASHED_SECTIONS.items())
        )
        return (
            f"the hash covers {covered} under schema {self.schema_version}; it "
            f"deliberately excludes {excluded}; no path, hostname or timestamp is "
            "in it, and it proves identity, not correctness"
        )

    # -- validation ---------------------------------------------------------

    def validate(self, *, for_run: bool = False) -> "Protocol":
        """Refuse a protocol this build cannot run, naming the field.

        ``for_run=True`` additionally requires the settings a run cannot invent:
        a scoring function, and box numbers when the box is not derived from a
        ligand or a pocket.  Both kinds of refusal happen **before** a run starts,
        not midway through a library.
        """
        if not isinstance(self.engine.scoring, str) or not self.engine.scoring.strip():
            raise ProtocolFieldError("protocol.engine.scoring must not be empty")
        if int(self.engine.exhaustiveness) < 1:
            raise ProtocolFieldError(
                f"protocol.engine.exhaustiveness must be >= 1, got {self.engine.exhaustiveness}"
            )
        if int(self.engine.num_poses) < 1:
            raise ProtocolFieldError(
                f"protocol.engine.num_poses must be >= 1, got {self.engine.num_poses}"
            )
        if float(self.engine.min_rmsd) <= 0:
            raise ProtocolFieldError(
                f"protocol.engine.min_rmsd must be positive, got {self.engine.min_rmsd}"
            )
        if float(self.box.spacing) <= 0:
            raise ProtocolFieldError(
                f"protocol.box.spacing must be positive, got {self.box.spacing}"
            )
        if int(self.execution.checkpoint_every) < 1:
            raise ProtocolFieldError(
                "protocol.execution.checkpoint_every must be >= 1, got "
                f"{self.execution.checkpoint_every}"
            )
        if self.library.format not in ("jsonl", "csv"):
            raise ProtocolFieldError(
                f"protocol.library.format is {self.library.format!r}; use jsonl or csv"
            )
        if self.library.consensus_method not in ("rank", "borda", "z"):
            raise ProtocolFieldError(
                f"protocol.library.consensus_method is {self.library.consensus_method!r}; "
                "use rank, borda or z"
            )
        if float(self.box.ligand_padding) < 0:
            raise ProtocolFieldError(
                f"protocol.box.ligand_padding must not be negative, got {self.box.ligand_padding}"
            )
        for key, value in self.thresholds.interactions.items():
            if value <= 0:
                raise ProtocolFieldError(
                    f"protocol.thresholds.interactions.{key} must be positive, got {value}"
                )
        if for_run:
            if self.box.source in ("manual", "receptor"):
                if self.box.center is None:
                    raise ProtocolFieldError(
                        "protocol.box.center is required when box.source is "
                        f"{self.box.source!r}: there is nothing to derive it from"
                    )
                if self.box.size is None:
                    raise ProtocolFieldError(
                        "protocol.box.size is required when box.source is "
                        f"{self.box.source!r}: there is nothing to derive it from"
                    )
                for axis, value in zip("xyz", self.box.size):
                    if value <= 0:
                        raise ProtocolFieldError(
                            f"protocol.box.size.{axis} must be positive, got {value}"
                        )
            if self.box.source == "pocket" and self.box.pocket_index is None:
                raise ProtocolFieldError(
                    "protocol.box.pocket_index is required when box.source is 'pocket'"
                )
        self.validate_portable()
        return self

    def absolute_paths(self) -> List[Tuple[str, str]]:
        """``(field, value)`` for every path-like value that is absolute.

        The portability rule this project has shipped a violation of before: a
        document must not record the machine it was made on.  Reuses
        :func:`odock.project.portable_path` rather than re-deriving the rule, so
        there is one definition of "portable" in the tree.
        """
        from .project import portable_path

        found: List[Tuple[str, str]] = []
        for path, value in _walk_strings(self.to_dict(), "protocol"):
            as_posix = value.replace("\\", "/")
            if portable_path(value) != as_posix:
                found.append((path, value))
        return found

    def validate_portable(self) -> "Protocol":
        """Refuse a protocol that records an absolute path, naming the field."""
        leaks = self.absolute_paths()
        if leaks:
            field_name, value = leaks[0]
            raise ProtocolFieldError(
                f"{field_name} records an absolute path ({value!r}); a protocol must "
                "be portable between machines, so use a path relative to the project "
                "or a bare file name"
            )
        return self

    # -- convenience --------------------------------------------------------

    def summary_lines(self) -> List[str]:
        """The protocol as a short, greppable table — what ``odock protocol show``
        prints and what the inspector tab lists."""
        hashed = self.hashed_settings()
        lines = [
            f"protocol      : {self.name}",
            f"schema        : {self.schema_version} ({'supported' if self.schema_version in SUPPORTED_VERSIONS else 'UNSUPPORTED'})",
            f"hash          : {self.short_hash()}  ({self.hash_note()})",
        ]
        if self.note:
            lines.append(f"note          : {self.note}")
        for section in self.HASHED_SECTIONS:
            for key, value in sorted(hashed[section].items()):
                lines.append(f"{section + '.' + key:22s}: {_fmt_value(value)}")
        if self.provenance.to_dict() != Provenance().to_dict():
            for key, value in sorted(self.provenance.to_dict().items()):
                if value is not None:
                    lines.append(f"provenance.{key:11s}: {value}")
        return lines


def _walk_strings(node: Any, path: str) -> Iterable[Tuple[str, str]]:
    """Every string leaf, with its dotted path."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _walk_strings(value, f"{path}.{key}")
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk_strings(value, f"{path}[{index}]")


def _fmt_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{k}={_fmt_value(v)}" for k, v in sorted(value.items())) + "}"
    return str(value)


# ---------------------------------------------------------------------------
# Reading and writing
# ---------------------------------------------------------------------------


def load_protocol(source: Any) -> Protocol:
    """Read a protocol from a path, a JSON string or a mapping.

    ``source`` may be a :class:`Protocol` (returned unchanged), a path, the text
    of a document, or an already-parsed mapping — the four shapes the CLI, the GUI
    and the tests each naturally have.
    """
    if isinstance(source, Protocol):
        return source
    if isinstance(source, Mapping):
        return Protocol.from_dict(source)
    if isinstance(source, Path):
        payload = json.loads(source.read_text(encoding="utf-8"))
        return Protocol.from_dict(payload)
    if isinstance(source, str):
        text = source.strip()
        if text.startswith("{"):
            return Protocol.from_dict(json.loads(text))
        payload = json.loads(Path(text).read_text(encoding="utf-8"))
        return Protocol.from_dict(payload)
    raise ProtocolFieldError(
        f"cannot read a protocol from {type(source).__name__}; pass a path, JSON text or a mapping"
    )


def save_protocol(protocol: Protocol, path: Any, *, validate: bool = True) -> Path:
    """Write a protocol as reviewable JSON: sorted keys, two-space indent, newline.

    The exact convention the rest of the tree uses for its documents, so two
    protocols can be compared with ``git diff`` and read by a human.
    """
    if validate:
        protocol.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(protocol.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return target


# ---------------------------------------------------------------------------
# Diffing, in the shape study diff already established
# ---------------------------------------------------------------------------


def _flatten(node: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Dotted paths to scalar values, so a diff can name the field that changed."""
    flat: Dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def diff_protocols(first: Protocol, second: Protocol) -> List[Dict[str, Any]]:
    """Every field that differs, **with both values**, in ``study diff``'s shape.

    Each entry is ``{"field", "kind", "values": [left, right], "note"}`` — the
    same keys :func:`odock.study.study_diff` produces, so the two renderers and
    the two text writers agree about what a difference looks like.
    """
    left = _flatten(first.to_dict())
    right = _flatten(second.to_dict())
    differences: List[Dict[str, Any]] = []
    for field_name in sorted(set(left) | set(right)):
        before = left.get(field_name)
        after = right.get(field_name)
        if before == after:
            continue
        kind = "label" if field_name in ("name", "note", "kind") else (
            "provenance" if field_name.startswith("provenance") else "setting"
        )
        note = ""
        if before is None or after is None:
            note = "not recorded on one side"
        differences.append(
            {
                "field": field_name,
                "kind": kind,
                "values": [before, after],
                "note": note,
            }
        )
    return differences


def protocol_diff_text(
    first: Protocol, second: Protocol, differences: Optional[Sequence[Mapping[str, Any]]] = None
) -> str:
    """The plain-text sibling of the structured diff (``study diff``'s style)."""
    found = list(differences if differences is not None else diff_protocols(first, second))
    lines = [
        f"protocol diff: {first.name} -> {second.name}",
        f"  hash {first.short_hash()} -> {second.short_hash()}"
        + ("  (identical settings)" if first.hash() == second.hash() else ""),
    ]
    if not found:
        lines.append("  no field differs: the two protocols are the same protocol")
        return "\n".join(lines)
    for item in found:
        left, right = (list(item.get("values") or [None, None]) + [None, None])[:2]
        suffix = f"   [{item.get('note')}]" if item.get("note") else ""
        lines.append(
            f"  {item.get('kind', 'setting')} {item.get('field')}: "
            f"{_fmt_value(left)} -> {_fmt_value(right)}{suffix}"
        )
    return "\n".join(lines)


def protocol_text(protocol: Protocol) -> str:
    """``odock protocol show``: the whole document as a readable list."""
    return "\n".join(protocol.summary_lines())


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def template_dir() -> Optional[Path]:
    """Where the shipped templates live, or ``None`` if this is not a checkout.

    ``ODOCK_PROTOCOL_DIR`` overrides it, which is how a lab keeps its own
    starting points without editing the installation.
    """
    import os

    override = os.environ.get("ODOCK_PROTOCOL_DIR")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_dir() else None
    candidate = Path(__file__).resolve().parents[2] / "examples"
    return candidate if candidate.is_dir() else None


def list_templates() -> List[Protocol]:
    """Every shipped template, sorted by name (and never raising)."""
    directory = template_dir()
    if directory is None:
        return []
    found: List[Protocol] = []
    for path in sorted(directory.glob(f"{TEMPLATE_PREFIX}*.json")):
        try:
            found.append(load_protocol(path))
        except ProtocolError:
            # A broken template must not make `protocol list` unusable; the
            # validation test is what catches it.
            continue
    return found


def template_named(name: str) -> Protocol:
    """One template by name or by file name (``fast-screen`` or ``protocol-fast-screen.json``)."""
    wanted = str(name).strip()
    for protocol in list_templates():
        candidates = (
            protocol.name,
            f"{TEMPLATE_PREFIX}{protocol.name}",
            f"{TEMPLATE_PREFIX}{protocol.name}.json",
        )
        if wanted in candidates:
            return protocol
    available = ", ".join(protocol.name for protocol in list_templates()) or "none"
    raise ProtocolFieldError(f"no protocol template named {name!r}; available: {available}")


# ---------------------------------------------------------------------------
# Capturing a live session, and turning a protocol into a run
# ---------------------------------------------------------------------------


def as_protocol(source: Any, *, name: str = "protocol", note: str = "") -> Protocol:
    """A protocol from a ``ScreenConfig``-like object or a plain mapping.

    This is the bridge the GUI uses: whatever the inspector is currently showing
    becomes a protocol with one call, so the document cannot drift from the
    widgets.  The document is assembled as a *dict* and then read back through
    :meth:`Protocol.from_dict`, which is what normalises types (a box centre
    arrives as a list from ``as_dict``) and validates every field on the way in.
    """
    if isinstance(source, Protocol):
        return source
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        getter = getattr(source, "as_dict", None)
        if getter is None:
            raise ProtocolFieldError(
                f"cannot capture a protocol from {type(source).__name__}; it has no as_dict()"
            )
        payload = dict(getter())
    document = Protocol(name=name, note=note).to_dict()
    for section in Protocol.HASHED_SECTIONS:
        found = payload.get(section)
        if isinstance(found, Mapping):
            document[section].update(dict(found))
    box = payload.get("box")
    if not isinstance(box, Mapping) and box is not None:
        as_dict = getattr(box, "as_dict", None)
        if as_dict is not None:
            box = as_dict()
    if isinstance(box, Mapping):
        document["box"].update(dict(box))
    # A ScreenConfig keeps the docking/execution/library settings flat, so a flat
    # key lands in the first section that declares it.  Order matters: a key that
    # two sections share (`interactions`: the library's switch and the threshold
    # *table*) must go to the section that can hold the value's type.
    for key, value in payload.items():
        for section in ("preparation", "engine", "execution", "library", "box"):
            if isinstance(document[section], dict) and key in document[section]:
                document[section][key] = value
                break
    protocol = Protocol.from_dict(document)
    return protocol.validate()


def capture_session(source: Any, *, name: str = "this session", note: str = "") -> Protocol:
    """The live settings as a protocol, with this build's versions attached."""
    protocol = as_protocol(source, name=name, note=note)
    protocol.provenance = this_build()
    return protocol


def this_build() -> Provenance:
    """The provenance block for a document written *now* by *this* build."""
    import platform

    provenance = Provenance(python=platform.python_version())
    try:
        from .project import tool_version

        provenance.tool = tool_version()
    except Exception:  # pragma: no cover - a checkout without metadata
        provenance.tool = None
    try:
        from .docking import kernel_version

        provenance.kernel = str(kernel_version())
    except Exception:  # pragma: no cover - the kernel is optional at import time
        provenance.kernel = None
    return provenance


@dataclass
class ProtocolRun:
    """What a protocol-driven run produced, and how to check it later."""

    protocol: Protocol
    outdir: Path
    summary: Any = None
    record_path: Optional[Path] = None

    @property
    def hash(self) -> str:
        return self.protocol.hash()

    def matches(self, other: Protocol) -> bool:
        """Whether ``other`` would ask for the same run (identity, not quality)."""
        return self.protocol.hash() == other.hash()


def run_protocol(
    protocol: Protocol,
    *,
    receptor: Any,
    library: Any,
    outdir: Any,
    overrides: Optional[Mapping[str, Any]] = None,
    runner: Optional[Any] = None,
) -> ProtocolRun:
    """Run a protocol against one receptor and one library, reproducibly.

    The protocol supplies the *settings*; the receptor and the library are run
    **inputs** and stay out of the protocol (and out of its hash), because a path
    is a location and the inputs' identity belongs to the run's own digests.

    ``runner`` is the seam for tests and for callers with their own executor: it
    is called with the assembled config.  The default is
    :func:`odock.screen.screen_ligands`.
    """
    protocol = Protocol.from_dict(protocol.to_dict())
    if overrides:
        _apply_overrides(protocol, overrides)
    protocol.validate(for_run=True)

    config = _screen_config(protocol, receptor=receptor, library=library, outdir=outdir)
    if runner is None:
        from .screen import screen_ligands as runner  # type: ignore[assignment]
    summary = runner(config)

    run = ProtocolRun(protocol=protocol, outdir=Path(outdir), summary=summary)
    run.record_path = record_protocol(protocol, Path(outdir))
    return run


def _screen_config(protocol: Protocol, *, receptor: Any, library: Any, outdir: Any) -> Any:
    """A ``ScreenConfig`` assembled from the protocol and the run's inputs."""
    from .screen import ScreenConfig
    from .docking import BoxSpec  # the box type the kernel and screen share

    box = BoxSpec(
        center=tuple(float(v) for v in (protocol.box.center or (0.0, 0.0, 0.0))),
        size=tuple(float(v) for v in (protocol.box.size or (20.0, 20.0, 20.0))),
        spacing=float(protocol.box.spacing),
    )
    return ScreenConfig(
        receptors=[receptor],
        inputs=[library],
        box=box,
        outdir=outdir,
        scoring=protocol.engine.scoring,
        exhaustiveness=int(protocol.engine.exhaustiveness),
        num_poses=int(protocol.engine.num_poses),
        min_rmsd=float(protocol.engine.min_rmsd),
        energy_range=float(protocol.engine.energy_range),
        search=protocol.engine.search,
        islands=int(protocol.engine.islands),
        population=int(protocol.engine.population),
        generations=int(protocol.engine.generations),
        use_grid=bool(protocol.engine.use_grid),
        refine=bool(protocol.engine.refine),
        seed=int(protocol.execution.seed),
        jobs=int(protocol.execution.jobs),
        timeout=protocol.execution.timeout,
        checkpoint_every=int(protocol.execution.checkpoint_every),
        filters=bool(protocol.library.filters),
        optimize=bool(protocol.preparation.ligand.optimize),
        prepare_seed=int(protocol.preparation.prepare_seed),
        limit=protocol.library.limit,
        top=int(protocol.library.top),
        fmt=str(protocol.library.format),
        interactions=bool(protocol.library.interactions),
        write_poses=bool(protocol.library.write_poses),
        consensus=bool(protocol.library.consensus),
        consensus_top=int(protocol.library.consensus_top),
        consensus_method=str(protocol.library.consensus_method),
    )


def _apply_overrides(protocol: Protocol, overrides: Mapping[str, Any]) -> None:
    """Set dotted fields (``engine.exhaustiveness=16``) before validating.

    Used by ``odock protocol run --set`` and by the GUI's "run anyway" path: an
    override is a deliberate deviation, and it is visible in the diff rather than
    baked silently into a saved document.
    """
    for dotted, value in overrides.items():
        parts = str(dotted).split(".")
        if len(parts) != 2:
            raise ProtocolFieldError(
                f"override {dotted!r} must be section.field, e.g. engine.exhaustiveness"
            )
        section_name, field_name = parts
        section = getattr(protocol, section_name, None)
        if section is None or not hasattr(section, field_name):
            raise ProtocolFieldError(
                f"override names protocol.{section_name}.{field_name}, which does not exist"
            )
        setattr(section, field_name, value)


#: The file a run writes its protocol record to, next to the results.
PROTOCOL_RECORD_NAME = "protocol.json"
PROTOCOL_HASH_NAME = "protocol-hash.txt"


def record_protocol(protocol: Protocol, outdir: Path) -> Path:
    """Write the protocol and its hash beside a run's results.

    Two files, deliberately: ``protocol.json`` is the document a human reads (and
    can re-run), ``protocol-hash.txt`` is the one-line answer a script or a
    colleague can compare without parsing anything.  The run's own
    ``manifest.json`` is updated with the same hash when it exists, so
    "was this produced by that protocol?" is answerable from either file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    record = save_protocol(protocol, outdir / PROTOCOL_RECORD_NAME, validate=False)
    digest = protocol.hash()
    (outdir / PROTOCOL_HASH_NAME).write_text(
        f"{digest}  {protocol.name}  schema {protocol.schema_version}\n", encoding="utf-8"
    )
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = None
        if isinstance(manifest, dict):
            manifest["protocol"] = protocol_record(protocol)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8"
            )
    return record


def protocol_record(protocol: Protocol) -> Dict[str, Any]:
    """The block a run records about the protocol that produced it."""
    return {
        "name": protocol.name,
        "schema_version": protocol.schema_version,
        "hash": protocol.hash(),
        "hash_note": protocol.hash_note(),
        "document": PROTOCOL_RECORD_NAME,
    }


def verify_run(directory: Any, protocol: Optional[Protocol] = None) -> Tuple[bool, str]:
    """Was this result produced by this protocol?

    Returns ``(matches, detail)``.  With no ``protocol`` given it reports what the
    run recorded, which is the other half of sharing: a colleague can ask the
    question before they have your document.
    """
    outdir = Path(directory)
    recorded = None
    manifest_path = outdir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and isinstance(manifest.get("protocol"), dict):
                recorded = str(manifest["protocol"].get("hash") or "")
        except Exception:
            recorded = None
    if recorded is None:
        hash_path = outdir / PROTOCOL_HASH_NAME
        if hash_path.exists():
            recorded = hash_path.read_text(encoding="utf-8").split()[0]
    if recorded is None:
        return False, (
            f"{outdir} records no protocol hash: it was produced before protocols "
            "existed, or by a tool that does not record one"
        )
    if protocol is None:
        return True, f"the run records protocol {recorded} (no document given to compare)"
    mine = protocol.hash()
    if mine == recorded:
        return True, (
            f"the run records protocol {recorded}, which is exactly this protocol "
            f"({protocol.name}, schema {protocol.schema_version})"
        )
    return False, (
        f"the run records protocol {recorded}, but this protocol hashes to {mine} "
        f"({protocol.name}, schema {protocol.schema_version}): the settings differ"
    )


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _parse_value(text: str) -> Any:
    """A ``--set`` value: JSON if it parses, otherwise the raw string."""
    try:
        return json.loads(text)
    except Exception:
        return text


def add_protocol_parser(sub: Any) -> None:
    """Register ``odock protocol`` on the CLI's subparsers.

    Called from :func:`odock.cli_ext.register_extensions`; idempotent, so a
    double registration is harmless.
    """
    choices = getattr(sub, "choices", None) or {}
    if "protocol" in choices:
        return
    parser = sub.add_parser(
        "protocol",
        help="reusable, versioned run settings (list, show, validate, diff, run)",
        description=(
            "A protocol is every decision a run depends on — preparation flags, box "
            "and how it was derived, engine and search settings, the seed, filters and "
            "thresholds — as one versioned, diffable document. It captures settings, "
            "not intent: a matching hash means two runs were asked for the same "
            "things, not that either is right."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    listing = actions.add_parser("list", help="the shipped templates, with their note")
    listing.set_defaults(handler=_cmd_list)

    showing = actions.add_parser("show", help="print a protocol as a table")
    showing.add_argument("path", nargs="?", help="the protocol file")
    showing.add_argument("--template", help="a shipped template name instead of a path")
    showing.set_defaults(handler=_cmd_show)

    validating = actions.add_parser(
        "validate", help="check the version, the fields and run-readiness"
    )
    validating.add_argument("path", nargs="?", help="the protocol file")
    validating.add_argument("--template", help="a shipped template name instead of a path")
    validating.set_defaults(handler=_cmd_validate)

    diffing = actions.add_parser("diff", help="field-by-field differences, with both values")
    diffing.add_argument("first")
    diffing.add_argument("second")
    diffing.set_defaults(handler=_cmd_diff)

    hashing = actions.add_parser("hash", help="the settings hash, and what it covers")
    hashing.add_argument("path", nargs="?", help="the protocol file")
    hashing.add_argument("--template", help="a shipped template name instead of a path")
    hashing.set_defaults(handler=_cmd_hash)

    saving = actions.add_parser("save", help="write a protocol (from a template)")
    saving.add_argument("out", help="where to write it")
    saving.add_argument("--template", help="start from this shipped template")
    saving.add_argument("--from", dest="source", help="start from this protocol file")
    saving.add_argument("--name", help="override the protocol's name")
    saving.add_argument("--note", help="override the note (when to use it)")
    saving.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="change one setting, e.g. --set engine.exhaustiveness=16",
    )
    saving.set_defaults(handler=_cmd_save)

    running = actions.add_parser("run", help="run a protocol on one receptor and one library")
    running.add_argument("path", nargs="?", help="the protocol file")
    running.add_argument("--template", help="a shipped template name instead of a path")
    running.add_argument("-r", "--receptor", required=True, help="the prepared receptor")
    running.add_argument("-i", "--input", required=True, dest="library", help="the library")
    running.add_argument("-o", "--out", required=True, help="the output directory")
    running.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="change one setting for this run only",
    )
    running.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble and validate everything, then stop without docking",
    )
    running.set_defaults(handler=_cmd_run)


def _protocol_from_args(args: Any) -> Protocol:
    if getattr(args, "template", None):
        return template_named(args.template)
    path = getattr(args, "path", None)
    if not path:
        raise ProtocolError("give a protocol file, or --template NAME (see `odock protocol list`)")
    return load_protocol(path)


def _overrides_from_args(args: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in getattr(args, "overrides", None) or []:
        if "=" not in str(item):
            raise ProtocolFieldError(f"--set {item!r} must be SECTION.FIELD=VALUE")
        key, _, value = str(item).partition("=")
        out[key.strip()] = _parse_value(value.strip())
    return out


def _cmd_list(args: Any) -> int:
    templates = list_templates()
    if not templates:
        print("no protocol templates are installed (expected examples/protocol-*.json)")
        return 0
    for protocol in templates:
        print(f"{protocol.name:20s} {protocol.short_hash()}  {protocol.note}")
    return 0


def _cmd_show(args: Any) -> int:
    print(protocol_text(_protocol_from_args(args)))
    return 0


def _cmd_validate(args: Any) -> int:
    try:
        protocol = _protocol_from_args(args)
        protocol.validate(for_run=True)
    except ProtocolError as exc:
        # Reported by name (`protocol.engine.no_such_setting`) with exit 1: the
        # command's whole purpose is to say what is wrong with a document.
        print(f"invalid: {exc}")
        return 1
    print(f"ok: {protocol.name}, schema {protocol.schema_version}, hash {protocol.short_hash()}")
    print(f"  {protocol.hash_note()}")
    return 0


def _cmd_diff(args: Any) -> int:
    first = load_protocol(args.first)
    second = load_protocol(args.second)
    print(protocol_diff_text(first, second))
    return 0 if first.hash() == second.hash() else 1


def _cmd_hash(args: Any) -> int:
    protocol = _protocol_from_args(args)
    print(protocol.hash())
    print(f"  {protocol.hash_note()}")
    return 0


def _cmd_save(args: Any) -> int:
    if args.template:
        protocol = template_named(args.template)
    elif args.source:
        protocol = load_protocol(args.source)
    else:
        raise ProtocolError("give --template NAME or --from FILE to start from")
    overrides = _overrides_from_args(args)
    if overrides:
        _apply_overrides(protocol, overrides)
    if args.name:
        protocol.name = str(args.name)
    if args.note:
        protocol.note = str(args.note)
    protocol.validate()
    path = save_protocol(protocol, args.out)
    print(f"wrote {path}  ({protocol.name}, hash {protocol.short_hash()})")
    return 0


def _cmd_run(args: Any) -> int:
    protocol = _protocol_from_args(args)
    overrides = _overrides_from_args(args)
    if args.dry_run:
        if overrides:
            _apply_overrides(protocol, overrides)
        protocol.validate(for_run=True)
        config = _screen_config(
            protocol, receptor=args.receptor, library=args.library, outdir=args.out
        )
        config.validated()
        print(f"dry run: {protocol.name} (hash {protocol.short_hash()}) is ready")
        print(f"  receptor : {args.receptor}")
        print(f"  library  : {args.library}")
        print(f"  outdir   : {args.out}")
        print(f"  {protocol.hash_note()}")
        return 0
    run = run_protocol(
        protocol,
        receptor=args.receptor,
        library=args.library,
        outdir=args.out,
        overrides=overrides or None,
    )
    print(f"ran {run.protocol.name} (hash {run.hash}); record: {run.record_path}")
    return 0

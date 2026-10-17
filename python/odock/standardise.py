# SPDX-License-Identifier: GPL-3.0-or-later
"""Library standardisation: turn a vendor library into something a ranking means.

A screening project that reads ``demo/libraries/*.smi`` cleanly and then meets a real
vendor library discovers the difference immediately: the file arrives with salts and
counter-ions, mixtures, formal charges, duplicates under two tautomers, stereocentres
nobody specified, and a header row that is not a molecule at all.  A tool that accepts
whatever it is handed reports a number for the salt, or for one arbitrary component of
a mixture — and neither of those is the ligand anyone meant to rank.

This module is the ordered, switchable recipe that turns such a file into a standard
one, **and the audit that says what happened to every input record**::

    from odock.standardise import StandardiseConfig, standardise

    report = standardise("vendor_library.smi", config=StandardiseConfig())
    print(report.summary())                 # in / kept / rejected / merged, by reason
    open("clean.smi", "w").write(report.to_smi())
    open("audit.csv", "w").write(report.to_csv())
    assert report.balanced                  # nothing vanished unexplained

The recipe
----------
The order is part of the answer, so it is named, checked, recorded and hashed:

1. **parse** — read the record and sanitise it.  A structure that cannot be sanitised is
   **refused**, never repaired into something plausible: an unparsable SMILES, a valence
   the aromaticity model rejects, and a header row are all refusals with their position
   in the file.
2. **repair** — RDKit's documented
   :func:`~rdkit.Chem.MolStandardize.rdMolStandardize.Cleanup` normalisations (drawing
   artefacts, nitro groups, ...).  It reports whether the canonical SMILES changed, and it
   refuses metal-containing structures instead of guessing what the metal bonds to.
3. **desalt** — the largest **organic** component is kept and every other fragment is
   reported by SMILES, heavy-atom count and charge.  A tie for the largest component is
   **ambiguous** and refused unless ``largest_on_tie="first"`` accepts the documented
   tie-break.  The choice is cross-checked against
   :func:`~rdkit.Chem.MolStandardize.rdMolStandardize.FragmentParent`, so the rule this
   module documents and the rule RDKit implements are known to agree.
4. **charge** — ``keep``, ``neutralise`` (RDKit's ``Uncharger``, with every changed atom
   reported) or ``refuse_charged``.  Every record is asked
   :func:`odock.protonation.can_represent_formal_charge` — the same predicate the fragment
   work uses — so the audit says whether the molecule carries the charges its pH implies,
   including the honest observation that a *neutralised* carboxylic acid is not the
   species pH 7.4 describes.
5. **stereo** — ``keep``, ``drop`` (with the centres that lost their assignment named) or
   ``refuse`` (an unspecified centre changes the shape and therefore the score, so a
   library that has them is not ready to rank).
6. **dedupe** — by canonical isomeric SMILES and, separately, by a **tautomer-insensitive
   key** (the canonical tautomer's SMILES).  Which molecule each duplicate collapsed into,
   and whether the pair differed only by tautomer, is reported per record.
7. **filter** — configurable windows on heavy atoms, molecular weight, net charge and
   rotatable bonds.  Every rejection names the property, the measured value and the
   window; there is no silent count.

The step order is checked: a configuration that puts a step where it cannot be honest
(the filter before the neutralisation whose charge it would report, a dedupe key computed
before the stereochemistry it depends on) is refused at construction, naming the
constraint.  :data:`STEP_CONSTRAINTS` is both the check and the documentation.

What this is not
----------------
Standardisation is **not registration**.  There is no InChIKey round-trip against a
registry, no salt-form database and no tautomer enumeration beyond the one rule stated
above; an InChIKey is computed locally, for grouping in the audit, and is not a claim of
identity with any external record.  No step decides which tautomer or protonation state is
bioactive — it records the assumption it made.  And a standardised library **ranks
differently from the raw one**; that is a reason to report both, not to trust either
blindly.  ``docs/STANDARDISE.md`` is the honest version of this paragraph, with the
measurements and the list of cases this recipe *refuses* instead of guessing.

Determinism
-----------
The same input and the same configuration produce byte-identical output.  There is no
timestamp, no path, no hostname and no dictionary-order dependence anywhere in the text.
The header of a written ``.smi`` carries the recipe hash — which covers every setting, the
step order and the schema version, and deliberately excludes paths, the same reasoning as
:meth:`odock.protocol.Protocol.hash` — the tool and RDKit versions, and the column layout,
so a downstream file can be traced to how it was made.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem, rdBase
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize as _std

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    _std = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

__all__ = [
    "RECIPE_VERSION",
    "STEPS",
    "STEP_CONSTRAINTS",
    "DEFAULT_MAX_FRAGMENTS",
    "POLICIES",
    "CATEGORIES",
    "KEPT",
    "REJECTED",
    "MERGED",
    "CHANGE",
    "OBSERVATION",
    "require_rdkit",
    "Fragment",
    "Finding",
    "PropertyLimits",
    "StandardiseConfig",
    "MoleculeAudit",
    "StandardisedMolecule",
    "StandardiseReport",
    "standardise",
    "canonical_smiles",
    "tautomer_key",
    "add_standardise_parser",
    "cmd_standardise",
    "main",
]

#: The schema version of the recipe.  Bumping it changes every recipe hash, which is the
#: point: a hash that survived a change of *meaning* would be a lie.
RECIPE_VERSION = 1

#: The recipe, in the order it runs when no step is disabled.
STEPS: Tuple[str, ...] = ("parse", "repair", "desalt", "charge", "stereo", "dedupe", "filter")

#: Why each step has to sit where it does.  Checked by
#: :meth:`StandardiseConfig.validate_order`, so a reordering that would change the answer
#: is refused instead of producing a quietly different library.
STEP_CONSTRAINTS: Tuple[Tuple[str, str, str], ...] = (
    ("parse", "", "the record has to exist before anything can be done to it"),
    (
        "repair",
        "parse",
        "sanitisation happens in parse, so repair has nothing to clean before it",
    ),
    (
        "desalt",
        "repair",
        "a counter-ion is removed from the repaired structure, not from a drawing artefact",
    ),
    (
        "charge",
        "desalt",
        "the charge of the salt is not the charge of the molecule that is kept",
    ),
    (
        "stereo",
        "charge",
        "the stereo policy is a decision about the final species",
    ),
    (
        "dedupe",
        "stereo",
        "the dedupe key is computed on the structure the later steps produced; a key "
        "taken before the stereo policy makes the enantiomers that collapse under "
        "stereo='drop' look like two distinct molecules",
    ),
    (
        "filter",
        "dedupe",
        "the properties a filter rejects on are properties of the kept, deduplicated "
        "molecule",
    ),
)

#: A record with more components than this is a mixture, not a salt, and the
#: largest-component rule would be a guess.  (A metal complex plus a counter-ion plus a
#: solvate is the usual shape; the number is deliberately generous.)
DEFAULT_MAX_FRAGMENTS = 8

#: The accepted values of each policy switch, with the default first.
POLICIES: Dict[str, Tuple[str, ...]] = {
    "charge": ("keep", "neutralise", "refuse_charged"),
    "stereo": ("keep", "drop", "refuse"),
    "largest_on_tie": ("refuse", "first"),
}

#: The statuses an audit record can have.
KEPT = "kept"
REJECTED = "rejected"
MERGED = "merged"

#: The categories a rejection is recorded under.  ``unsupported`` means "this recipe does
#: not decide this case", which is different from ``invalid`` ("this record is not a
#: molecule") and from ``filter`` ("this molecule is outside the requested window").
CATEGORIES: Tuple[str, ...] = ("invalid", "unsupported", "stereo", "charge", "filter")

#: A finding that altered the molecule or its representation.
CHANGE = "change"
#: A finding that only records something worth knowing (a kept unspecified centre, a
#: disagreement with another implementation).  Observations do not make a molecule
#: "changed" in the audit.
OBSERVATION = "observe"

#: The property keys the filter and the audit use, in the order they are reported.
PROPERTY_KEYS: Tuple[str, ...] = (
    "heavy_atoms",
    "molecular_weight",
    "net_charge",
    "rotatable_bonds",
)


# ---------------------------------------------------------------------------
# RDKit guard
# ---------------------------------------------------------------------------


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for library standardisation. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


@contextlib.contextmanager
def _quiet_rdkit() -> Any:
    """Silence RDKit's C++ log while a structure is parsed or refused.

    A vendor library with thousands of bad rows makes RDKit print one SMILES parse
    error per row straight to stderr, which drowns the audit this command exists to
    produce.  The reason is recorded on the record instead, so the message is kept
    where it belongs — the same reasoning as :func:`odock.chem.ligand._quiet_rdkit`.
    """
    if not _HAVE_RDKIT:  # pragma: no cover - the caller has already required it
        yield
        return
    try:
        with rdBase.BlockLogs():
            yield
    except AttributeError:  # pragma: no cover - very old RDKit
        yield


# ---------------------------------------------------------------------------
# Audit pieces
# ---------------------------------------------------------------------------


@dataclass
class Fragment:
    """One component of a multi-component record, as the desalting step saw it."""

    #: Canonical SMILES of the component.
    smiles: str
    #: Heavy-atom count, which is the size the largest-component rule uses.
    heavy_atoms: int
    #: Whether the component contains carbon: only an organic component can be kept.
    organic: bool
    #: Formal net charge of the component.
    charge: int
    #: Whether this component is the one that survives.
    kept: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "smiles": self.smiles,
            "heavy_atoms": int(self.heavy_atoms),
            "organic": bool(self.organic),
            "charge": int(self.charge),
            "kept": bool(self.kept),
        }

    def describe(self) -> str:
        return f"{self.smiles} ({self.heavy_atoms} heavy atom(s), charge {self.charge:+d})"


@dataclass
class Finding:
    """One thing a step did (or noticed) at one record."""

    step: str
    detail: str
    #: :data:`CHANGE` or :data:`OBSERVATION`; observations do not count as changes.
    kind: str = CHANGE

    def as_dict(self) -> Dict[str, Any]:
        return {"step": self.step, "detail": self.detail, "kind": self.kind}

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.step}] {self.detail}"


def _digest(payload: Any) -> str:
    """A canonical digest: sorted keys, no whitespace, ASCII only."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_window(text: str, *, cast: Any = float) -> Tuple[Optional[Any], Optional[Any]]:
    """Parse ``"MIN:MAX"`` (either side may be empty) into a validated pair."""
    if not isinstance(text, str) or ":" not in text:
        raise ValueError(
            f"a window is written MIN:MAX (e.g. 8:60, or :500 for an upper bound only), "
            f"got {text!r}"
        )
    low, high = (part.strip() for part in text.split(":", 1))
    values: List[Optional[Any]] = []
    for side, raw in (("minimum", low), ("maximum", high)):
        if not raw:
            values.append(None)
            continue
        try:
            values.append(cast(raw))
        except (TypeError, ValueError):
            raise ValueError(f"the {side} of {text!r} is not a number: {raw!r}") from None
    if values[0] is not None and values[1] is not None and values[0] > values[1]:
        raise ValueError(f"the window {text!r} is empty: minimum > maximum")
    return (values[0], values[1])


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}" if not value.is_integer() else f"{value:.0f}"
    return str(value)


def _window_text(window: Tuple[Optional[Any], Optional[Any]]) -> str:
    low, high = window
    return f"{'-inf' if low is None else _number(low)}..{'+inf' if high is None else _number(high)}"


def _rounded(value: Any, places: int = 4) -> Any:
    """Round a reported float, so an audit table reads like numbers and not like floats."""
    return round(float(value), places) if isinstance(value, float) else value


@dataclass(frozen=True)
class PropertyLimits:
    """The windows the filter step applies, each optional and inclusive.

    Every bound is optional (``None``) and every violation of every record is reported,
    naming the property, the measured value and the window, because a count of silent
    drops is not a result.
    """

    heavy_atoms: Optional[Tuple[Optional[int], Optional[int]]] = None
    molecular_weight: Optional[Tuple[Optional[float], Optional[float]]] = None
    charge: Optional[Tuple[Optional[int], Optional[int]]] = None
    rotatable_bonds: Optional[Tuple[Optional[int], Optional[int]]] = None

    def __post_init__(self) -> None:
        for name, window in self.as_windows().items():
            if window is None:
                continue
            low, high = window
            if low is not None and high is not None and low > high:
                raise ValueError(f"the {name} window is empty: {low} > {high}")

    def as_windows(self) -> Dict[str, Optional[Tuple[Optional[Any], Optional[Any]]]]:
        return {
            "heavy_atoms": self.heavy_atoms,
            "molecular_weight": self.molecular_weight,
            "net_charge": self.charge,
            "rotatable_bonds": self.rotatable_bonds,
        }

    def active(self) -> bool:
        return any(window is not None for window in self.as_windows().values())

    def violations(self, properties: Dict[str, Any]) -> List[str]:
        """Every window this record misses, as a sentence naming the numbers.

        The property is named by its :data:`PROPERTY_KEYS` key (``net_charge``, not
        "charge"), so a report can count rejections by property mechanically.
        """
        out: List[str] = []
        for name, window in self.as_windows().items():
            if window is None:
                continue
            value = properties.get(name)
            if value is None or (isinstance(value, int) and value < 0 and name == "rotatable_bonds"):
                # A rotatable-bond count of -1 means the perception module was
                # unavailable: report no value rather than a wrong one.
                continue
            low, high = window
            if (low is not None and value < low) or (high is not None and value > high):
                out.append(f"{name} {_number(value)} outside {_window_text(window)}")
        return out

    def as_dict(self) -> Dict[str, Any]:
        return {
            name: (list(window) if window else None)
            for name, window in (
                ("heavy_atoms", self.heavy_atoms),
                ("molecular_weight", self.molecular_weight),
                ("charge", self.charge),
                ("rotatable_bonds", self.rotatable_bonds),
            )
        }


# ---------------------------------------------------------------------------
# The configuration: the recipe and its identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardiseConfig:
    """The recipe: which steps run, in what order, and the policy each applies.

    ``hash()`` covers **every** setting here (and the schema version) and nothing else:
    no path, no filename, no timestamp.  Two people who standardise two different vendor
    files with the same recipe get the same hash, which is the useful property — the same
    reasoning as :meth:`odock.protocol.Protocol.hash`.
    """

    recipe_version: int = RECIPE_VERSION
    order: Tuple[str, ...] = STEPS
    #: Per-step switches.  A step that is off is reported as skipped, never treated as if
    #: it had run and found nothing.
    repair: bool = True
    desalt: bool = True
    charge: bool = True
    stereo: bool = True
    dedupe: bool = True
    filters: bool = True
    #: ``keep`` | ``neutralise`` | ``refuse_charged``.
    charge_policy: str = "keep"
    #: ``keep`` | ``drop`` | ``refuse``.
    stereo_policy: str = "keep"
    #: ``refuse`` (the honest default) | ``first`` (documented tie-break).
    largest_on_tie: str = "refuse"
    #: Collapse duplicates that differ only by tautomer, via the canonical tautomer.
    tautomer_dedupe: bool = True
    #: A record with more components than this is refused as a mixture.
    max_fragments: int = DEFAULT_MAX_FRAGMENTS
    #: The property windows the filter applies.
    limits: PropertyLimits = field(default_factory=PropertyLimits)

    def __post_init__(self) -> None:
        for name, allowed, value in (
            ("charge_policy", POLICIES["charge"], self.charge_policy),
            ("stereo_policy", POLICIES["stereo"], self.stereo_policy),
            ("largest_on_tie", POLICIES["largest_on_tie"], self.largest_on_tie),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if int(self.recipe_version) < 1:
            raise ValueError(f"recipe_version must be >= 1, got {self.recipe_version!r}")
        if int(self.max_fragments) < 1:
            raise ValueError(f"max_fragments must be >= 1, got {self.max_fragments!r}")
        object.__setattr__(self, "order", tuple(str(step) for step in self.order))
        self.validate_order()

    # -- the order ---------------------------------------------------------

    def validate_order(self) -> None:
        """Refuse an order in which a step cannot mean what it claims.

        Each step's constraint is written down in :data:`STEP_CONSTRAINTS` and checked
        here rather than trusted from the caller, because the order changes the answer.
        """
        order = list(self.order)
        if len(set(order)) != len(order):
            raise ValueError(f"the step order repeats a step: {order}")
        unknown = [step for step in order if step not in STEPS]
        if unknown:
            raise ValueError(f"unknown step(s) {unknown}; the recipe is {list(STEPS)}")
        missing = [step for step in STEPS if step not in order]
        if missing:
            raise ValueError(
                f"the step order omits {missing}; every step must appear (disable a step "
                "with its switch instead of dropping it)"
            )
        for step, must_follow, why in STEP_CONSTRAINTS:
            if not must_follow:
                if order.index(step) != 0:
                    raise ValueError(f"{step} has to be first: {why}")
                continue
            if order.index(must_follow) > order.index(step):
                raise ValueError(f"{step} has to run after {must_follow}: {why}")

    def enabled(self) -> Tuple[str, ...]:
        """The steps that will actually run, in order."""
        switches = {
            "parse": True,
            "repair": bool(self.repair),
            "desalt": bool(self.desalt),
            "charge": bool(self.charge),
            "stereo": bool(self.stereo),
            "dedupe": bool(self.dedupe),
            "filter": bool(self.filters),
        }
        return tuple(step for step in self.order if switches.get(step, True))

    def skipped(self) -> Tuple[str, ...]:
        """The steps the configuration turns off, in recipe order."""
        running = set(self.enabled())
        return tuple(step for step in self.order if step not in running)

    # -- identity ----------------------------------------------------------

    def hashed_settings(self) -> Dict[str, Any]:
        """Exactly what the recipe hash covers, as a plain dictionary."""
        return {
            "recipe_version": int(self.recipe_version),
            "order": list(self.order),
            "enabled": list(self.enabled()),
            "charge_policy": self.charge_policy,
            "stereo_policy": self.stereo_policy,
            "largest_on_tie": self.largest_on_tie,
            "tautomer_dedupe": bool(self.tautomer_dedupe),
            "max_fragments": int(self.max_fragments),
            "limits": self.limits.as_dict(),
        }

    def hash(self) -> str:
        """The identity of these **settings** (see the module docstring)."""
        return _digest(self.hashed_settings())

    def short_hash(self) -> str:
        return self.hash()[:12]

    def hash_note(self) -> str:
        """One sentence naming what the hash covers and what it does not."""
        return (
            f"the recipe hash covers the {len(self.enabled())} enabled step(s), their "
            f"order and every policy under schema {self.recipe_version}; it excludes the "
            "input path, the machine and any timestamp, so the same recipe applied to "
            "another file has the same hash — it proves identity, not correctness"
        )

    # -- (de)serialisation -------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable view, including the hash and what runs."""
        return {
            **self.hashed_settings(),
            "skipped": list(self.skipped()),
            "hash": self.hash(),
            "hash_note": self.hash_note(),
            "step_constraints": [
                {"step": step, "after": must_follow, "why": why}
                for step, must_follow, why in STEP_CONSTRAINTS
            ],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StandardiseConfig":
        """Rebuild a configuration from :meth:`hashed_settings`; unknown keys are refused."""
        if not isinstance(payload, dict):
            raise ValueError(f"a configuration must be a mapping, got {type(payload).__name__}")
        known = {
            "recipe_version", "order", "enabled", "charge_policy", "stereo_policy",
            "largest_on_tie", "tautomer_dedupe", "max_fragments", "limits", "hash",
            "hash_note", "step_constraints", "skipped",
        }
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(
                f"unknown configuration key(s): {unknown}; the settings are "
                f"{sorted(known - {'hash', 'hash_note', 'step_constraints', 'skipped', 'enabled'})}"
            )
        limits = payload.get("limits") or {}
        if not isinstance(limits, dict):
            raise ValueError("limits must be a mapping of windows")
        windows: Dict[str, Any] = {}
        for name in ("heavy_atoms", "molecular_weight", "charge", "rotatable_bonds"):
            window = limits.get(name)
            windows[name] = tuple(window) if window else None
        return cls(
            recipe_version=int(payload.get("recipe_version", RECIPE_VERSION)),
            order=tuple(payload.get("order", STEPS)),
            charge_policy=str(payload.get("charge_policy", "keep")),
            stereo_policy=str(payload.get("stereo_policy", "keep")),
            largest_on_tie=str(payload.get("largest_on_tie", "refuse")),
            tautomer_dedupe=bool(payload.get("tautomer_dedupe", True)),
            max_fragments=int(payload.get("max_fragments", DEFAULT_MAX_FRAGMENTS)),
            limits=PropertyLimits(**windows),
        )


# ---------------------------------------------------------------------------
# The audit records
# ---------------------------------------------------------------------------


@dataclass
class MoleculeAudit:
    """What happened to one input record — the answer to "why is this not in the file?".

    Every record of the input produces exactly one of these, whatever happened to it, and
    :attr:`StandardiseReport.balanced` asserts that the three statuses account for them
    all.
    """

    #: 1-based line of a ``.smi``, or 1-based record of an ``.sdf``.
    position: int
    #: What :attr:`position` counts: ``"smi-line"`` or ``"sdf-record"``.
    position_kind: str
    name: str
    input_smiles: str
    status: str = KEPT
    #: For a rejection: why, naming the measured value or the offending substructure.
    reason: str = ""
    #: For a rejection: one of :data:`CATEGORIES`.
    category: str = ""
    #: The final SMILES, when the record survived (or produced the structure it merged).
    output_smiles: str = ""
    #: What each step did or noticed, in order.
    findings: List[Finding] = field(default_factory=list)
    #: Components the desalting step removed.
    removed_fragments: List[Fragment] = field(default_factory=list)
    #: For a merged record: the index into :attr:`StandardiseReport.kept` it collapsed into.
    merged_into: Optional[int] = None
    merged_into_name: str = ""
    merge_kind: str = ""
    canonical_key: str = ""
    tautomer_key: str = ""
    #: The InChIKey, computed locally; a grouping aid, not a registry identity.
    inchikey: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    #: The protonation predicate's one-line statement, when the charge step ran.
    charge_statement: str = ""
    #: Whether the molecule carries the formal charges its pH implies (predicate result).
    charge_consistent: Optional[bool] = None

    def note(self, step: str, detail: str, kind: str = CHANGE) -> None:
        self.findings.append(Finding(step, detail, kind))

    def observe(self, step: str, detail: str) -> None:
        self.note(step, detail, OBSERVATION)

    @property
    def changes(self) -> List[Finding]:
        return [finding for finding in self.findings if finding.kind == CHANGE]

    @property
    def changed(self) -> bool:
        """Whether any step altered the structure, its representation, or its components."""
        return bool(self.changes) or bool(self.removed_fragments)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position": int(self.position),
            "position_kind": self.position_kind,
            "name": self.name,
            "status": self.status,
            "category": self.category,
            "reason": self.reason,
            "input_smiles": self.input_smiles,
            "output_smiles": self.output_smiles,
            "changed": bool(self.changed),
            "findings": [finding.as_dict() for finding in self.findings],
            "removed_fragments": [fragment.as_dict() for fragment in self.removed_fragments],
            "merged_into": self.merged_into,
            "merged_into_name": self.merged_into_name,
            "merge_kind": self.merge_kind,
            "canonical_key": self.canonical_key,
            "tautomer_key": self.tautomer_key,
            "inchikey": self.inchikey,
            "properties": dict(self.properties),
            "charge_statement": self.charge_statement,
            "charge_consistent": self.charge_consistent,
        }

    def describe(self) -> str:
        """One line for a report or a log."""
        where = f"{self.position_kind.replace('-', ' ')} {self.position}"
        if self.status == KEPT:
            changes = "; ".join(finding.detail for finding in self.changes)
            return f"{where} {self.name}: kept{' — ' + changes if changes else ''}"
        if self.status == MERGED:
            return (
                f"{where} {self.name}: merged into {self.merged_into_name} "
                f"({self.merge_kind})"
            )
        return f"{where} {self.name}: {self.category} — {self.reason}"


@dataclass
class StandardisedMolecule:
    """One kept molecule, with the audit that produced it."""

    index: int
    name: str
    smiles: str
    mol: Any = field(repr=False, default=None)
    audit: Optional[MoleculeAudit] = None

    @property
    def inchikey(self) -> str:
        return self.audit.inchikey if self.audit is not None else ""

    @property
    def properties(self) -> Dict[str, Any]:
        return dict(self.audit.properties) if self.audit is not None else {}

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "index": int(self.index),
            "name": self.name,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "changed": bool(self.audit.changed) if self.audit is not None else False,
        }
        for key in PROPERTY_KEYS:
            value = self.properties.get(key)
            payload[key] = round(value, 4) if isinstance(value, float) else value
        if self.audit is not None:
            payload["record"] = self.audit.describe()
        return payload


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class StandardiseReport:
    """The cleaned library and the audit of every record that produced it."""

    config: StandardiseConfig
    audits: List[MoleculeAudit] = field(default_factory=list)
    kept: List[StandardisedMolecule] = field(default_factory=list)
    #: Where the records came from.  Informational: it never enters the recipe hash.
    source: str = ""
    position_kind: str = "smi-line"
    tool_version: str = ""
    rdkit_version: str = ""

    # -- counting ---------------------------------------------------------

    @property
    def n_in(self) -> int:
        return len(self.audits)

    @property
    def rejected(self) -> List[MoleculeAudit]:
        return [audit for audit in self.audits if audit.status == REJECTED]

    @property
    def merged(self) -> List[MoleculeAudit]:
        return [audit for audit in self.audits if audit.status == MERGED]

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    @property
    def n_merged(self) -> int:
        return len(self.merged)

    @property
    def balanced(self) -> bool:
        """``in == kept + rejected + merged`` — no record vanished unexplained."""
        return self.n_in == self.n_kept + self.n_rejected + self.n_merged

    def assert_balanced(self) -> None:
        """Raise when the invariant fails, naming the numbers."""
        if not self.balanced:
            lost = self.n_in - (self.n_kept + self.n_rejected + self.n_merged)
            raise AssertionError(
                f"standardisation lost track of {lost} record(s): in {self.n_in} != "
                f"kept {self.n_kept} + rejected {self.n_rejected} + merged {self.n_merged}"
            )

    # -- aggregates -------------------------------------------------------

    def rejected_by_category(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for audit in self.rejected:
            out[audit.category] = out.get(audit.category, 0) + 1
        return dict(sorted(out.items()))

    def rejected_by_property(self) -> Dict[str, int]:
        """Which filter window did the rejecting, counted per property."""
        out: Dict[str, int] = {}
        for audit in self.rejected:
            if audit.category != "filter":
                continue
            for key in PROPERTY_KEYS:
                if key in audit.reason:
                    out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def merged_by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for audit in self.merged:
            out[audit.merge_kind] = out.get(audit.merge_kind, 0) + 1
        return dict(sorted(out.items()))

    def changed_by_step(self) -> Dict[str, int]:
        """How many records each step actually altered (observations excluded)."""
        out: Dict[str, int] = {}
        for audit in self.audits:
            for step in {finding.step for finding in audit.changes}:
                out[step] = out.get(step, 0) + 1
        return dict(sorted(out.items()))

    def charge_observations(self) -> Dict[str, int]:
        """Kept molecules whose charges do, or do not, match what their pH implies."""
        out = {"consistent": 0, "inconsistent": 0, "unknown": 0}
        for molecule in self.kept:
            flag = molecule.audit.charge_consistent if molecule.audit is not None else None
            if flag is None:
                out["unknown"] += 1
            elif flag:
                out["consistent"] += 1
            else:
                out["inconsistent"] += 1
        return out

    # -- text -------------------------------------------------------------

    def summary(self) -> str:
        """The audit in words: what came in, what is left, and why."""
        lines = [
            f"standardised {self.n_in} record(s) from {self.source or '<memory>'}",
            f"  kept {self.n_kept}, rejected {self.n_rejected}, merged {self.n_merged}",
            f"  recipe {self.config.short_hash()} ({', '.join(self.config.enabled())})",
        ]
        if self.config.skipped():
            lines.append(f"  skipped: {', '.join(self.config.skipped())}")
        by_category = self.rejected_by_category()
        if by_category:
            lines.append(
                "  rejected by category: "
                + ", ".join(f"{name} {count}" for name, count in by_category.items())
            )
        by_property = self.rejected_by_property()
        if by_property:
            lines.append(
                "  filters that rejected: "
                + ", ".join(f"{name} {count}" for name, count in by_property.items())
            )
        by_kind = self.merged_by_kind()
        if by_kind:
            lines.append("  merged: " + ", ".join(f"{name} {count}" for name, count in by_kind.items()))
        by_step = self.changed_by_step()
        if by_step:
            lines.append(
                "  changed: " + ", ".join(f"{name} {count}" for name, count in by_step.items())
            )
        if self.config.charge:
            charge = self.charge_observations()
            lines.append(
                "  charge: "
                f"{charge['consistent']} record(s) carry the charges pH 7.4 implies, "
                f"{charge['inconsistent']} do not (recorded, not changed — that is the "
                f"assumption, see docs/STANDARDISE.md), {charge['unknown']} unknown"
            )
        lines.append(f"  recipe hash {self.config.hash()} — {self.config.hash_note()}")
        return "\n".join(lines)

    def headline(self) -> str:
        """One sentence, for a progress line or a log."""
        return (
            f"kept {self.n_kept} of {self.n_in} record(s): {self.n_rejected} rejected, "
            f"{self.n_merged} merged (recipe {self.config.short_hash()})"
        )

    # -- serialisation ----------------------------------------------------

    #: The columns of the ``.smi`` output.  Two, deliberately: a ``.smi`` that carries an
    #: extra column stops round-tripping its names through the rest of the toolchain
    #: (``odock.chem.ligand.read_ligands`` takes everything after the first field as the
    #: name).  The InChIKey lives in the audit instead.
    SMI_COLUMNS: Tuple[str, ...] = ("smiles", "name")

    def header(self) -> str:
        """The comment block a written ``.smi`` carries.

        Deterministic by construction: no timestamp, no path, no dictionary order.  It
        records the recipe hash, the versions that can change a canonical SMILES, and the
        column layout — everything a reader needs to know how the file was made.
        """
        config = self.config
        lines = [
            f"# odock standardise {self.tool_version} — recipe {config.hash()}",
            f"# recipe: schema {config.recipe_version}; steps {', '.join(config.enabled())}",
        ]
        if config.skipped():
            lines.append(f"# skipped: {', '.join(config.skipped())}")
        lines.append(
            f"# policies: charge={config.charge_policy} stereo={config.stereo_policy} "
            f"largest_on_tie={config.largest_on_tie} "
            f"tautomer_dedupe={str(config.tautomer_dedupe).lower()}"
        )
        if config.limits.active():
            lines.append(
                "# limits: "
                + ", ".join(
                    f"{name} {_window_text(window)}"
                    for name, window in config.limits.as_windows().items()
                    if window
                )
            )
        lines.append(f"# rdkit {self.rdkit_version}")
        lines.append(
            f"# counts: in {self.n_in}, kept {self.n_kept}, rejected {self.n_rejected}, "
            f"merged {self.n_merged}"
        )
        lines.append("# columns: SMILES<TAB>name")
        lines.append(
            "# standardisation is not registration: no registry was consulted and the "
            "InChIKeys in the audit are computed locally (docs/STANDARDISE.md)"
        )
        return "\n".join(lines)

    def to_smi(self) -> str:
        """The kept library as ``.smi`` text, with a traceable header.

        Deterministic: the same input and configuration produce byte-identical text.
        """
        lines = [self.header()]
        for molecule in self.kept:
            lines.append(f"{molecule.smiles}\t{molecule.name}")
        return "\n".join(lines) + "\n"

    #: The columns of :meth:`to_csv`, in order.
    AUDIT_COLUMNS: Tuple[str, ...] = (
        "position", "position_kind", "name", "status", "category", "reason",
        "input_smiles", "output_smiles", "merged_into", "merged_into_name", "merge_kind",
        "canonical_key", "tautomer_key", "inchikey", "heavy_atoms", "molecular_weight",
        "net_charge", "rotatable_bonds", "charge_consistent", "changed",
        "removed_fragments", "changes", "observations",
    )

    def to_csv(self) -> str:
        """The audit, one row per input record — including the rejects and the merges."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.AUDIT_COLUMNS)
        for audit in self.audits:
            properties = audit.properties
            writer.writerow(
                (
                    audit.position,
                    audit.position_kind,
                    audit.name,
                    audit.status,
                    audit.category,
                    audit.reason,
                    audit.input_smiles,
                    audit.output_smiles,
                    "" if audit.merged_into is None else audit.merged_into,
                    audit.merged_into_name,
                    audit.merge_kind,
                    audit.canonical_key,
                    audit.tautomer_key,
                    audit.inchikey,
                    properties.get("heavy_atoms", ""),
                    _rounded(properties.get("molecular_weight")),
                    properties.get("net_charge", ""),
                    properties.get("rotatable_bonds", ""),
                    "" if audit.charge_consistent is None else int(audit.charge_consistent),
                    "yes" if audit.changed else "no",
                    "; ".join(fragment.describe() for fragment in audit.removed_fragments),
                    "; ".join(finding.detail for finding in audit.changes),
                    "; ".join(
                        finding.detail
                        for finding in audit.findings
                        if finding.kind == OBSERVATION
                    ),
                )
            )
        return buffer.getvalue()

    def as_dict(self, *, full: bool = True) -> Dict[str, Any]:
        """A JSON-serialisable view: the recipe, the counts, and (optionally) every audit."""
        payload: Dict[str, Any] = {
            "recipe": self.config.as_dict(),
            "source": self.source,
            "position_kind": self.position_kind,
            "tool_version": self.tool_version,
            "rdkit_version": self.rdkit_version,
            "counts": {
                "in": self.n_in,
                "kept": self.n_kept,
                "rejected": self.n_rejected,
                "merged": self.n_merged,
                "balanced": self.balanced,
            },
            "rejected_by_category": self.rejected_by_category(),
            "rejected_by_property": self.rejected_by_property(),
            "merged_by_kind": self.merged_by_kind(),
            "changed_by_step": self.changed_by_step(),
            "charge_observations": self.charge_observations(),
            "kept": [molecule.as_dict() for molecule in self.kept],
        }
        if full:
            payload["audit"] = [audit.as_dict() for audit in self.audits]
        return payload

    def write(
        self,
        smi: Any = None,
        *,
        csv_path: Any = None,
        json_path: Any = None,
    ) -> List[Path]:
        """Write the cleaned library and/or the audit; returns the files written."""
        written: List[Path] = []
        if smi is not None:
            path = Path(smi)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_smi(), encoding="utf-8")
            written.append(path)
        if csv_path is not None:
            path = Path(csv_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # `Path.write_text(..., newline=...)` is **Python 3.10+** and this package
            # declares `requires-python = ">=3.9"`, so the keyword version raised
            # `TypeError: write_text() got an unexpected keyword argument 'newline'`
            # there. The local interpreter is 3.10, which is why no developer machine
            # could see it: the CI matrix runs 3.9, and that is the only place the
            # declared floor is actually tested. `csv` needs `newline=""` so Windows
            # does not turn the line endings into CRLF, so it is opened explicitly.
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(self.to_csv())
            written.append(path)
        if json_path is not None:
            path = Path(json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
            written.append(path)
        return written


# ---------------------------------------------------------------------------
# Reading records, with positions
# ---------------------------------------------------------------------------


@dataclass
class _Record:
    """One input record: where it was, what it is called, and how it was read."""

    position: int
    name: str
    smiles: str
    #: For an SDF record: the unsanitised molecule.  ``None`` for a ``.smi`` line.
    mol: Any = None


def _as_text(source: Any) -> Tuple[str, str]:
    """``(text, label)`` for a path or for the text itself.

    The label is used in messages and in the report's ``source`` field only; it never
    enters the recipe hash.
    """
    if isinstance(source, (str, Path)):
        raw = str(source)
        path = Path(raw)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace"), str(path)
        return raw, "<text>"
    raise TypeError(f"standardise() takes a path or SMILES/SDF text, got {type(source).__name__}")


def _read_smiles_records(text: str) -> List[_Record]:
    """Every ``SMILES<TAB>name`` record of a ``.smi`` document, with its line number.

    Comment lines (``#``) and blank lines are skipped: they are the file's header and its
    separators, not molecules.  A *table* header such as ``SMILES ID`` is **not** skipped —
    it is a record that fails to parse, which the parse step reports as invalid at its line
    number.  That distinction is deliberate: silently dropping something that looks like a
    molecule is how a library loses an entry.
    """
    records: List[_Record] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        records.append(
            _Record(
                position=number,
                name=" ".join(fields[1:]).strip() or f"line_{number}",
                smiles=fields[0],
            )
        )
    return records


def _read_sdf_records(text: str) -> Tuple[List[_Record], List[MoleculeAudit]]:
    """Every record of an SDF document, as an *unsanitised* molecule, with its number.

    Sanitisation is deliberately not done here: whether a structure can be sanitised is
    the parse step's decision, and it has to be reported as one.  A record RDKit cannot
    read at all becomes an invalid audit entry rather than a silent gap.
    """
    records: List[_Record] = []
    broken: List[MoleculeAudit] = []
    supplier = Chem.SDMolSupplier()
    supplier.SetData(text, sanitize=False, removeHs=False)
    for index in range(len(supplier)):
        position = index + 1
        try:
            with _quiet_rdkit():
                mol = supplier[index]
        except Exception as exc:  # pragma: no cover - defensive
            broken.append(
                _rejected(
                    MoleculeAudit(
                        position=position,
                        position_kind="sdf-record",
                        name=f"record_{position}",
                        input_smiles="",
                    ),
                    "invalid",
                    f"RDKit could not read this record: {exc}",
                )
            )
            continue
        if mol is None:
            broken.append(
                _rejected(
                    MoleculeAudit(
                        position=position,
                        position_kind="sdf-record",
                        name=f"record_{position}",
                        input_smiles="",
                    ),
                    "invalid",
                    "RDKit could not read this record (no molecule was produced)",
                )
            )
            continue
        name = ""
        try:
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name").strip()
        except Exception:  # pragma: no cover - defensive
            name = ""
        records.append(
            _Record(
                position=position,
                name=name or f"record_{position}",
                smiles=canonical_smiles(mol) if mol.GetNumAtoms() else "",
                mol=mol,
            )
        )
    return records, broken


def _records(source: Any) -> Tuple[List[_Record], List[MoleculeAudit], str, str]:
    """``(records, pre_rejected, label, position_kind)`` for any supported source."""
    text, label = _as_text(source)
    suffix = ""
    if isinstance(source, (str, Path)):
        candidate = Path(str(source))
        if candidate.is_file():
            suffix = candidate.suffix.lower()
    is_sdf = suffix in (".sdf", ".sd") or (
        suffix == "" and "$$$$" in text and "V2000" in text
    )
    if is_sdf:
        records, broken = _read_sdf_records(text)
        return records, broken, label, "sdf-record"
    if suffix in (".mol2", ".pdb", ".pdbqt", ".mol"):
        raise ValueError(
            f"{suffix} is not a supported standardisation input: it carries no reliable "
            "bond orders, which is exactly what the parse step has to judge. Convert it to "
            "SMILES or SDF first (this command reads .smi and .sdf); docs/STANDARDISE.md "
            "lists the format as unsupported rather than guessing."
        )
    return _read_smiles_records(text), [], label, "smi-line"


# ---------------------------------------------------------------------------
# Chemistry helpers
# ---------------------------------------------------------------------------


def canonical_smiles(mol: Any, *, isomeric: bool = True) -> str:
    """The canonical SMILES of `mol`, or ``""`` when RDKit cannot write one."""
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=bool(isomeric))
    except Exception:  # pragma: no cover - defensive
        return ""


def tautomer_key(mol: Any) -> str:
    """A **tautomer-insensitive** key: the canonical tautomer's canonical SMILES.

    This is the stated rule and the only tautomer handling in this module: RDKit's
    ``TautomerEnumerator`` proposes the tautomers and its own canonicalisation picks one.
    There is no enumeration beyond that, so a molecule whose tautomers the enumerator does
    not generate will not collapse with its partner — which is why the key is recorded per
    molecule and the limitation is documented rather than left implicit.
    """
    require_rdkit()
    try:
        with _quiet_rdkit():
            canonical = _std.TautomerEnumerator().Canonicalize(mol)
        return canonical_smiles(canonical) or canonical_smiles(mol)
    except Exception:  # pragma: no cover - a molecule the enumerator refuses
        return canonical_smiles(mol)


def _net_charge(mol: Any) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _is_organic(fragment: Any) -> bool:
    return any(atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms())


def _has_metal(mol: Any) -> bool:
    """Whether the molecule contains a metal, which this recipe refuses to model."""
    table = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        try:
            if table.GetElementSymbol(atom.GetAtomicNum()) in _METALS:
                return True
        except Exception:  # pragma: no cover - defensive
            continue
    return False


#: The metals that turn a record into an unsupported case.  Coordination chemistry is not
#: modelled by this recipe; see ``docs/STANDARDISE.md``, Unsupported.
_METALS = frozenset(
    (
        "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
        "Co", "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Mo", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce", "Pt", "Au", "Hg", "Tl", "Pb",
        "Bi",
    )
)


def _rotatable_bonds(mol: Any) -> int:
    """The engine's notion of a torsion, so the filter agrees with the docking.

    ``odock.chem.ligand.rotatable_bonds`` is the perception the kinematic tree is built
    from (amides frozen, a tert-butyl not a rotor), so a ``rotatable_bonds`` window counts
    exactly what the search will sample.  When the chemistry layer is unavailable the count
    is ``-1`` (unknown), never a different number.
    """
    try:
        from .chem.ligand import rotatable_bonds

        return int(len(rotatable_bonds(mol)))
    except Exception:  # pragma: no cover - chemistry layer unavailable
        return -1


def _properties(mol: Any) -> Dict[str, Any]:
    """The numbers the filter and the audit report."""
    try:
        weight = float(Descriptors.MolWt(mol))
    except Exception:  # pragma: no cover - defensive
        weight = float("nan")
    return {
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "molecular_weight": weight,
        "net_charge": _net_charge(mol),
        "rotatable_bonds": _rotatable_bonds(mol),
    }


def _inchikey(mol: Any) -> str:
    """The InChIKey, computed locally; a grouping aid, never a registry identity."""
    try:
        return str(Chem.MolToInchiKey(mol))
    except Exception:  # pragma: no cover - InChI support missing or the molecule refused
        return ""


# ---------------------------------------------------------------------------
# Rejection helpers
# ---------------------------------------------------------------------------


def _rejected(audit: MoleculeAudit, category: str, reason: str) -> MoleculeAudit:
    audit.status = REJECTED
    audit.category = category
    audit.reason = reason
    return audit


def _unsupported(audit: MoleculeAudit, reason: str) -> MoleculeAudit:
    return _rejected(audit, "unsupported", reason)


# ---------------------------------------------------------------------------
# The steps.  Each takes the working molecule and the audit and returns the new
# molecule, or ``None`` for "this record is out" (the audit says why).
# ---------------------------------------------------------------------------


def _step_parse(mol: Any, audit: MoleculeAudit, config: StandardiseConfig) -> Optional[Any]:
    """Sanitise the record, or refuse it.  Never repairs an ambiguous structure."""
    if mol is None:
        if not audit.input_smiles:
            _rejected(audit, "invalid", "the record holds no structure")
            return None
        with _quiet_rdkit():
            mol = Chem.MolFromSmiles(audit.input_smiles)
        if mol is None:
            _rejected(audit, "invalid", f"{audit.input_smiles!r} is not a parsable SMILES")
            return None
    if mol.GetNumAtoms() == 0:
        _rejected(audit, "invalid", "the record holds no atoms")
        return None
    # An explicit sanitisation is the point of this step: `MolFromSmiles` already sanitises,
    # an SDF record does not, and a *failure* here is the refusal this recipe documents
    # rather than something to repair silently.
    try:
        with _quiet_rdkit():
            Chem.SanitizeMol(mol)
    except Exception as exc:
        _rejected(
            audit,
            "invalid",
            f"the structure cannot be sanitised ({str(exc).splitlines()[0]}); it is "
            "refused rather than repaired into something plausible",
        )
        return None
    return mol


def _step_repair(mol: Any, audit: MoleculeAudit, config: StandardiseConfig) -> Optional[Any]:
    """RDKit's documented cleanup normalisations.

    The step has no policy of its own, so `config` is unused; every step takes the same
    ``(mol, audit, config)`` signature so the dispatch table stays uniform.

    Metal-containing structures are *not* refused here: a sodium counter-ion is exactly
    what desalting removes, so the metal check belongs on the molecule that survives (see
    :func:`_run_record`).
    """
    before = canonical_smiles(mol)
    try:
        repaired = _std.Cleanup(mol)
    except Exception as exc:  # pragma: no cover - cleanup rarely throws
        _unsupported(audit, f"RDKit's cleanup normalisations failed: {exc}")
        return None
    if repaired is None:  # pragma: no cover - defensive
        return mol
    after = canonical_smiles(repaired)
    if after and after != before:
        audit.note("repair", f"normalised {before} -> {after}")
    return repaired


def _step_desalt(mol: Any, audit: MoleculeAudit, config: StandardiseConfig) -> Optional[Any]:
    """Keep the largest organic component; report every fragment removed."""
    try:
        fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    except Exception as exc:
        _unsupported(
            audit,
            f"the record's components cannot be separated cleanly ({exc}); a mixture this "
            "recipe cannot split is not one it will pick a component from",
        )
        return None
    if len(fragments) <= 1:
        return mol
    if len(fragments) > int(config.max_fragments):
        _unsupported(
            audit,
            f"the record has {len(fragments)} components, more than the "
            f"{config.max_fragments} a salt is allowed to have; it looks like a mixture, "
            "and choosing a component would be a guess (raise max_fragments to change this)",
        )
        return None

    described = [
        Fragment(
            smiles=canonical_smiles(fragment),
            heavy_atoms=int(fragment.GetNumHeavyAtoms()),
            organic=_is_organic(fragment),
            charge=_net_charge(fragment),
        )
        for fragment in fragments
    ]
    organic = [index for index, fragment in enumerate(fragments) if _is_organic(fragment)]
    if not organic:
        _unsupported(
            audit,
            "no component is organic ("
            + ", ".join(fragment.smiles for fragment in described)
            + "), so there is no ligand to keep",
        )
        return None
    # Largest organic component; ties are broken alphabetically by canonical SMILES so the
    # order is deterministic — but a tie is *ambiguous*, and the default policy refuses it.
    ranked = sorted(organic, key=lambda index: (-described[index].heavy_atoms, described[index].smiles))
    best_index = ranked[0]
    best = described[best_index]
    ties = [
        described[index]
        for index in organic
        if described[index].heavy_atoms == best.heavy_atoms and index != best_index
    ]
    if ties and config.largest_on_tie == "refuse":
        names = ", ".join([best.smiles] + [fragment.smiles for fragment in ties])
        _unsupported(
            audit,
            f"ambiguous largest component: {len(ties) + 1} organic components of "
            f"{best.heavy_atoms} heavy atom(s) each ({names}); pass "
            "largest_on_tie='first' (or --on-tie first) to accept the alphabetical "
            "tie-break",
        )
        return None

    for index, fragment in enumerate(described):
        fragment.kept = index == best_index
    audit.removed_fragments = [fragment for fragment in described if not fragment.kept]
    kept = fragments[best_index]
    kept_smiles = canonical_smiles(kept)
    audit.note(
        "desalt",
        "kept the largest organic component "
        f"{best.describe()}; removed "
        + ", ".join(fragment.describe() for fragment in audit.removed_fragments),
    )
    # Cross-check the documented rule against RDKit's own reference rule: if they ever
    # disagree, one of them changed and the audit has to say so.
    try:
        reference = canonical_smiles(_std.FragmentParent(mol))
        if reference and reference != kept_smiles:
            audit.observe(
                "desalt",
                f"RDKit's FragmentParent would keep {reference}, this recipe kept "
                f"{kept_smiles} (the documented rule is 'largest organic component')",
            )
    except Exception:  # pragma: no cover - defensive
        pass
    return kept


def _step_charge(mol: Any, audit: MoleculeAudit, config: StandardiseConfig) -> Optional[Any]:
    """Apply the charge policy; record what the pH implies through the shared predicate."""
    from . import protonation

    before = _net_charge(mol)
    policy = config.charge_policy
    if policy == "refuse_charged" and before != 0:
        capability = protonation.can_represent_formal_charge(mol)
        _unsupported(
            audit,
            f"the molecule carries a net formal charge of {before:+d} and the policy is "
            f"refuse_charged ({capability.statement()})",
        )
        return None
    if policy == "neutralise" and before != 0:
        try:
            uncharged = _std.Uncharger().uncharge(mol)
        except Exception as exc:  # pragma: no cover - the uncharger rarely throws
            _unsupported(audit, f"the molecule could not be neutralised: {exc}")
            return None
        after = _net_charge(uncharged)
        changed: List[str] = []
        for index in range(min(uncharged.GetNumAtoms(), mol.GetNumAtoms())):
            was = mol.GetAtomWithIdx(index).GetFormalCharge()
            now = uncharged.GetAtomWithIdx(index).GetFormalCharge()
            if was != now:
                changed.append(f"{uncharged.GetAtomWithIdx(index).GetSymbol()}{index + 1}: {was:+d}->{now:+d}")
        if after != before:
            audit.note(
                "charge",
                f"neutralised {before:+d} -> {after:+d}"
                + (
                    f" ({len(changed)} atom(s): {', '.join(changed[:6])}"
                    + (", ..." if len(changed) > 6 else "") + ")"
                    if changed
                    else ""
                ),
            )
        mol = uncharged
    # The shared predicate: does this molecule carry the formal charges its pH implies?
    try:
        capability = protonation.can_represent_formal_charge(mol)
        audit.charge_statement = capability.statement()
        audit.charge_consistent = bool(capability.possible)
        if not capability.possible:
            audit.observe(
                "charge",
                f"pH {protonation.DEFAULT_PH:g} note: {capability.statement()}",
            )
    except Exception as exc:  # pragma: no cover - defensive
        audit.charge_statement = f"the protonation predicate failed: {exc}"
        audit.charge_consistent = None
    return mol


def _step_stereo(mol: Any, audit: MoleculeAudit, config: StandardiseConfig) -> Optional[Any]:
    """Apply the stereochemistry policy, naming the centres it decides about."""
    try:
        centres = list(
            Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        )
    except Exception:  # pragma: no cover - very old RDKit
        centres = []
    if not centres:
        return mol
    unassigned = [index for index, label in centres if str(label) == "?"]
    assigned = [index for index, label in centres if str(label) != "?"]
    policy = config.stereo_policy
    if policy == "refuse" and unassigned:
        _rejected(
            audit,
            "stereo",
            f"{len(unassigned)} stereocentre(s) with no specified configuration "
            f"(atom {unassigned[0] + 1}"
            f"{' and others' if len(unassigned) > 1 else ''}); an unspecified centre "
            "changes the shape and therefore the score, so the record is refused rather "
            "than ranked under an assumed configuration",
        )
        return None
    if policy == "drop" and assigned:
        stripped = Chem.Mol(mol)
        Chem.RemoveStereochemistry(stripped)
        audit.note(
            "stereo",
            f"dropped {len(assigned)} assigned stereocentre(s) (atoms "
            + ", ".join(str(index + 1) for index in assigned[:8])
            + (", ..." if len(assigned) > 8 else "")
            + "); enantiomers and diastereomers now share a dedupe key",
        )
        mol = stripped
    if unassigned:
        audit.observe(
            "stereo",
            f"{len(unassigned)} centre(s) left unspecified by the input (atoms "
            + ", ".join(str(index + 1) for index in unassigned[:8])
            + "); kept as given",
        )
    return mol


def _step_filter(mol: Any, audit: MoleculeAudit, config: StandardiseConfig) -> Optional[Any]:
    """Apply every property window, reporting each violation that fired."""
    properties = _properties(mol)
    audit.properties = properties
    violations = config.limits.violations(properties)
    if violations:
        _rejected(audit, "filter", "; ".join(violations))
        return None
    return mol


#: The step implementations, by name.  ``dedupe`` is not here: it needs the whole library,
#: not one record, and runs after the per-record loop.
_STEP_FUNCTIONS = {
    "parse": _step_parse,
    "repair": _step_repair,
    "desalt": _step_desalt,
    "charge": _step_charge,
    "stereo": _step_stereo,
    "filter": _step_filter,
}


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _run_record(
    record: _Record, config: StandardiseConfig, position_kind: str
) -> Tuple[MoleculeAudit, Any]:
    """Run the recipe over one record; returns ``(audit, mol or None)``."""
    audit = MoleculeAudit(
        position=record.position,
        position_kind=position_kind,
        name=record.name,
        input_smiles=record.smiles,
    )
    mol: Any = record.mol
    for step in config.enabled():
        if step == "dedupe":
            # Handled after the loop: it is a property of the library, not of a record.
            continue
        mol = _STEP_FUNCTIONS[step](mol, audit, config)
        if mol is None:
            return audit, None
    if mol is not None and _has_metal(mol):
        # The check is on the molecule that *survives*, not on the input: a sodium
        # counter-ion is what desalting removes, while a metal that is still there after
        # every step is a coordination compound this recipe will not guess at.
        _unsupported(
            audit,
            "the standardised molecule still contains a metal; coordination chemistry is "
            "not modelled by this recipe (docs/STANDARDISE.md, Unsupported)"
            + (
                ""
                if config.desalt
                else " — note that --no-desalt was given, so a counter-ion could not be "
                "removed either"
            ),
        )
        return audit, None
    if mol is not None:
        audit.output_smiles = canonical_smiles(mol)
        audit.canonical_key = audit.output_smiles
        audit.tautomer_key = (
            tautomer_key(mol) if config.tautomer_dedupe else audit.canonical_key
        )
        audit.inchikey = _inchikey(mol)
        if not audit.properties:
            audit.properties = _properties(mol)
    return audit, mol


def standardise(
    source: Any,
    *,
    config: Optional[StandardiseConfig] = None,
) -> StandardiseReport:
    """Standardise one library; returns the cleaned molecules **and** the full audit.

    Parameters
    ----------
    source
        A path to a ``.smi`` or ``.sdf`` file, or the text of one.
    config
        The recipe.  :class:`StandardiseConfig`'s defaults keep every structure-changing
        step (``charge="keep"``, ``stereo="keep"``) and only deduplicate, so the result
        differs from the input only where the input was not a library yet.

    Returns
    -------
    :class:`StandardiseReport`
        ``report.kept`` (the cleaned library), ``report.audits`` (one record per input, in
        input order, including the rejects and the merges) and the aggregates
        :meth:`~StandardiseReport.rejected_by_category`,
        :meth:`~StandardiseReport.rejected_by_property` and
        :meth:`~StandardiseReport.merged_by_kind`.  The invariant
        ``in == kept + rejected + merged`` is asserted before returning.
    """
    require_rdkit()
    config = config or StandardiseConfig()
    records, pre_rejected, label, position_kind = _records(source)
    report = StandardiseReport(
        config=config,
        source=label,
        position_kind=position_kind,
        tool_version=_version(),
        rdkit_version=str(Chem.rdBase.rdkitVersion),
    )

    # The dedupe keys are taken from the *finished* molecule (the constraints guarantee
    # dedupe runs after every structure-changing step), and the first occurrence wins: input
    # order is the only tie-break the user can see.
    seen_exact: Dict[str, int] = {}
    seen_tautomer: Dict[str, int] = {}
    for record in records:
        audit, mol = _run_record(record, config, position_kind)
        if mol is not None and config.dedupe:
            exact = audit.canonical_key
            tautomer = audit.tautomer_key
            keeper = seen_exact.get(exact)
            if keeper is None and config.tautomer_dedupe and tautomer:
                keeper = seen_tautomer.get(tautomer)
            if keeper is not None:
                kept_molecule = report.kept[keeper]
                audit.status = MERGED
                audit.merged_into = keeper
                audit.merged_into_name = kept_molecule.name
                audit.merge_kind = (
                    "exact duplicate" if exact == kept_molecule.smiles else "tautomer"
                )
                audit.note(
                    "dedupe",
                    f"collapsed into {kept_molecule.name} (record "
                    f"{kept_molecule.audit.position if kept_molecule.audit else keeper + 1}) "
                    f"as a {audit.merge_kind}: {exact} vs {kept_molecule.smiles}",
                )
                report.audits.append(audit)
                continue
            seen_exact.setdefault(exact, len(report.kept))
            if config.tautomer_dedupe and tautomer:
                seen_tautomer.setdefault(tautomer, len(report.kept))
        if mol is None:
            report.audits.append(audit)
            continue
        report.kept.append(
            StandardisedMolecule(
                index=len(report.kept),
                name=audit.name,
                smiles=audit.output_smiles,
                mol=mol,
                audit=audit,
            )
        )
        report.audits.append(audit)

    report.audits.extend(pre_rejected)
    report.audits.sort(key=lambda audit: (audit.position, audit.position_kind))
    report.assert_balanced()
    return report


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def config_from_args(args: Any) -> StandardiseConfig:
    """Build the recipe from parsed CLI arguments (shared by ``odock`` and ``python -m``)."""
    window = getattr(args, "heavy_atoms", None)
    return StandardiseConfig(
        order=tuple(getattr(args, "order", None) or STEPS),
        repair=not getattr(args, "no_repair", False),
        desalt=not getattr(args, "no_desalt", False),
        charge=not getattr(args, "no_charge", False),
        stereo=not getattr(args, "no_stereo", False),
        dedupe=not getattr(args, "no_dedupe", False),
        filters=not getattr(args, "no_filter", False),
        charge_policy=getattr(args, "charge_policy", "keep"),
        stereo_policy=getattr(args, "stereo", "keep"),
        largest_on_tie=getattr(args, "on_tie", "refuse"),
        tautomer_dedupe=not getattr(args, "exact_dedupe", False),
        max_fragments=int(getattr(args, "max_fragments", DEFAULT_MAX_FRAGMENTS)),
        limits=PropertyLimits(
            heavy_atoms=parse_window(window, cast=int) if window else None,
            molecular_weight=parse_window(args.mw) if getattr(args, "mw", None) else None,
            charge=parse_window(args.charge, cast=int) if getattr(args, "charge", None) else None,
            rotatable_bonds=(
                parse_window(args.rotatable, cast=int)
                if getattr(args, "rotatable", None)
                else None
            ),
        ),
    )


def cmd_standardise(args: Any) -> int:
    """``odock standardise``: standardise a library and write the audit with it."""
    import sys

    try:
        config = config_from_args(args)
    except ValueError as exc:
        print(f"odock standardise: error: {exc}", file=sys.stderr)
        return 2
    try:
        report = standardise(args.input, config=config)
    except (OSError, ValueError, ImportError, TypeError) as exc:
        print(f"odock standardise: error: {exc}", file=sys.stderr)
        return 2

    written = report.write(args.out, csv_path=args.audit, json_path=args.json)
    if getattr(args, "verbose", False):
        for audit in report.audits:
            if audit.status != KEPT:
                print(f"  {audit.describe()}", file=sys.stderr)
    if args.json:
        print(json.dumps(report.as_dict(full=False), indent=2))
    elif getattr(args, "quiet", False):
        print(report.headline(), file=sys.stderr)
    else:
        print(report.summary())
    for path in written:
        print(f"wrote {path}", file=sys.stderr)
    if report.n_kept == 0:
        # A run that kept nothing is a failed run, not a successful empty library.
        print(
            "odock standardise: error: no record survived; the audit says why",
            file=sys.stderr,
        )
        return 3
    return 0


def _add_arguments(parser: Any) -> None:
    """The flags shared by ``odock standardise`` and ``python -m odock.standardise``."""
    parser.add_argument("input", help="library .smi or .sdf (or its text)")
    parser.add_argument("-o", "--out", help="write the cleaned library as .smi")
    parser.add_argument(
        "--audit", help="write the per-record audit as CSV (changed/removed/merged/rejected)"
    )
    parser.add_argument("--json", help="write the whole report (recipe, counts, audit) as JSON")
    parser.add_argument(
        "--stereo", choices=POLICIES["stereo"], default="keep",
        help="keep stereochemistry, drop it with a warning, or refuse a record with an "
             "unspecified centre",
    )
    parser.add_argument(
        "--charge-policy", choices=POLICIES["charge"], default="keep",
        help="keep the formal charges as given, neutralise them, or refuse charged records",
    )
    parser.add_argument(
        "--on-tie", choices=POLICIES["largest_on_tie"], default="refuse",
        help="two organic components of the same size: refuse the record (default) or keep "
             "the first by canonical SMILES",
    )
    parser.add_argument("--heavy-atoms", metavar="MIN:MAX", help="heavy-atom window, e.g. 8:60")
    parser.add_argument("--mw", metavar="MIN:MAX", help="molecular-weight window, e.g. 120:700")
    parser.add_argument("--charge", metavar="MIN:MAX", help="net-charge window, e.g. -1:1")
    parser.add_argument("--rotatable", metavar="MIN:MAX", help="rotatable-bond window, e.g. 0:12")
    parser.add_argument(
        "--max-fragments", type=int, default=DEFAULT_MAX_FRAGMENTS, metavar="N",
        help="a record with more components than this is refused as a mixture",
    )
    parser.add_argument("--no-repair", action="store_true", help="skip RDKit's cleanup normalisations")
    parser.add_argument("--no-desalt", action="store_true", help="keep every component")
    parser.add_argument("--no-charge", action="store_true", help="skip the charge step")
    parser.add_argument("--no-stereo", action="store_true", help="skip the stereochemistry step")
    parser.add_argument("--no-dedupe", action="store_true", help="keep every duplicate")
    parser.add_argument("--no-filter", action="store_true", help="apply no property window")
    parser.add_argument(
        "--exact-dedupe", action="store_true",
        help="deduplicate by canonical SMILES only, not by the tautomer-insensitive key",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print every rejection to stderr")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the headline")


def add_standardise_parser(sub) -> None:
    """Register ``odock standardise`` on `sub` (idempotent)."""
    choices = getattr(sub, "choices", None)
    if isinstance(choices, dict) and "standardise" in choices:
        return
    parser = sub.add_parser(
        "standardise",
        help="standardise a vendor library (salts, charges, duplicates, stereo, filters) with an audit",
        description=(
            "Run the ordered standardisation recipe over a library: sanitise or refuse, strip "
            "counter-ions and keep the largest organic component, apply the charge and "
            "stereochemistry policies, collapse duplicates (by canonical SMILES and by a "
            "tautomer-insensitive key) and apply property windows — reporting, per record, "
            "what was changed, removed, merged or rejected.  The written .smi carries the "
            "recipe hash in its header and the audit is a CSV/JSON file.  Standardisation "
            "is not registration: see docs/STANDARDISE.md."
        ),
    )
    _add_arguments(parser)
    parser.set_defaults(func=cmd_standardise)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m odock.standardise``: the same command without the CLI."""
    import argparse

    parser = argparse.ArgumentParser(prog="odock standardise", description=__doc__)
    _add_arguments(parser)
    return cmd_standardise(parser.parse_args(list(argv) if argv is not None else None))
if __name__ == "__main__":  # pragma: no cover - module entry point
    # A `-m` entry does not pass through `cli.main()`, so it does not get the UTF-8
    # stream setup, and on a non-UTF-8 console code page a pipe receives bytes the
    # parent cannot decode (0xa1, GBK, from an em dash).  Same call, same reason.
    from .cli import _use_utf8_streams

    _use_utf8_streams()
    raise SystemExit(main())

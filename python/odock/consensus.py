# SPDX-License-Identifier: GPL-3.0-or-later
"""Consensus scoring: rescore one set of poses with several force fields.

The kernel implements three force fields -- Vina, Vinardo and AutoDock 4 -- and
they do not agree.  On the bundled 3PTB demo Vina and AD4 both put pose 1 first
while Vinardo puts pose 2 first, and the two best poses differ by 0.03 kcal/mol
under Vina but by 0.03 the other way under Vinardo.  Reporting one number hides
that; reporting three numbers and a correlation shows a modeller whether the
ranking is a property of the physics or of the parameterisation.

Why ranks and not an average of energies
----------------------------------------
The three fields are on different scales: for the same pose on the same system
Vina reports about -6.2 kcal/mol and AD4 about -6.5, and the *spread* between
poses also differs (Vina spans 1.8 kcal/mol over the six demo poses, Vinardo
1.8, AD4 2.5).  Averaging raw energies would therefore let whichever field has
the widest spread dominate, and a constant offset between fields would be
meaningless anyway.  This module aggregates **ranks** (or z-scores, which are the
scale-free version of the same idea) instead:

``method="rank"`` (default)
    Each pose's rank within a field, normalised to ``[0, 1]`` by
    ``(rank - 1) / (n - 1)`` so that 0 is the best pose and 1 the worst.  Ties
    share their average rank.  The consensus score is the (weighted) mean of
    those normalised ranks -- a normalised-rank sum, lower is better.
``method="borda"``
    The same, without the normalisation: the mean of the raw ranks.  Identical
    ordering when the field weights are equal and every list is complete; kept
    because it is the classical name for the method.
``method="z"``
    ``(x - mean) / sd`` within each field (population sd), averaged.  Lower is
    still better.  Use this when you want the *size* of a field's preference to
    count rather than only its ordering.

A pose whose score is not finite in a field (a NaN from a clashing configuration,
say) is given the worst badness in that field and named in
``ConsensusPose.missing``; it is never dropped silently and it is never rewarded
for a missing number.

The Spearman correlation between every pair of fields is reported alongside, in
:attr:`ConsensusResult.correlations` and :attr:`ConsensusResult.matrix`.
Spearman is used rather than Pearson because the question "do these two fields
agree about the *ordering*" is exactly what rank correlation answers, and it is
robust to the non-linear relationship between the potentials.  Ties are handled
by average ranks, which is the standard correction.

Everything here is plain functions and dataclasses: no Qt, no RDKit, no plotting
library, so a CLI and the workbench can both call it.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _odock
from .prepare import BoxSpec

__all__ = [
    "AD4_TYPES",
    "DEFAULT_SCORINGS",
    "PoseAtom",
    "ChargeContext",
    "ConsensusPose",
    "ConsensusResult",
    "EnergyRow",
    "ReproducibilityReport",
    "StrainRow",
    "average_ranks",
    "box_from_any",
    "charge_context",
    "consensus_score",
    "decomposition_rows",
    "decomposition_table",
    "normalised_ranks",
    "pdbqt_atoms",
    "pdbqt_models",
    "pose_pdbqt_from_coords",
    "reproducibility",
    "rescore_poses",
    "spearman",
    "strain_corrected_ranking",
    "with_coordinates",
    "z_scores",
]

#: The force fields the kernel can evaluate, in the default consensus order.
DEFAULT_SCORINGS: Tuple[str, ...] = ("vina", "vinardo", "ad4")

#: AutoDock atom type -> element, for reading a PDBQT without RDKit.  Identical
#: to the table :mod:`odock.report` uses; kept here so the pose plumbing has no
#: import edge back into the reporting layer.
AD4_TYPES: Dict[str, str] = {
    "C": "C", "A": "C", "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C",
    "N": "N", "NA": "N", "O": "O", "OA": "O", "S": "S", "SA": "S",
    "P": "P", "H": "H", "HD": "H", "F": "F", "I": "I", "Cl": "Cl",
    "Br": "Br", "Si": "Si", "At": "At", "W": "H",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
    "G0": "H", "G1": "H", "G2": "H", "G3": "H",
}

#: Atom types that are not heavy atoms (hydrogens, water sentinel, macrocycle
#: closure dummies).
_NOT_HEAVY = frozenset(("H", "HD", "W", "G0", "G1", "G2", "G3", "", "D"))


# ---------------------------------------------------------------------------
# Pose plumbing (shared by odock.metrics, odock.report and the CLI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseAtom:
    """One ``ATOM``/``HETATM`` record of a PDBQT document."""

    serial: int
    name: str
    element: str
    ad_type: str
    res_name: str
    res_id: int
    chain: str
    x: float
    y: float
    z: float
    charge: float = 0.0

    @property
    def is_heavy(self) -> bool:
        """Whether this atom counts as a heavy atom for ligand efficiency."""
        return self.ad_type.upper() not in _NOT_HEAVY and self.element.upper() != "H"

    @property
    def label(self) -> str:
        """``"SER195:OG"``, or the bare atom name without residue information."""
        if self.res_name:
            return f"{self.res_name}{self.res_id}:{self.name}"
        return self.name


def _element_from_pdbqt(name: str, ad_type: str) -> str:
    element = AD4_TYPES.get(ad_type)
    if element:
        return element
    letters = "".join(c for c in name if c.isalpha())
    two = letters[:2].capitalize()
    if two in ("Cl", "Br", "Si", "At"):
        return two
    return letters[:1].upper() or "C"


def pdbqt_atoms(text: str) -> List[PoseAtom]:
    """The ``ATOM``/``HETATM`` records of a PDBQT document, in file order.

    The element is taken from the AutoDock type column when it is present (which
    is exact) and from the atom name otherwise.  Records shorter than the fixed
    columns are skipped: a truncated line has no coordinates to trust.
    """
    atoms: List[PoseAtom] = []
    for line in str(text).splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if len(line) < 54:
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        name = line[12:16].strip()
        chain = line[21:22].strip()
        ad_type = line[77:].strip() if len(line) >= 79 else ""
        if not ad_type:
            tokens = line.split()
            ad_type = tokens[-1] if tokens else ""
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = len(atoms) + 1
        try:
            res_id = int(line[22:26])
        except ValueError:
            res_id = 0
        charge = 0.0
        if len(line) >= 76:
            try:
                charge = float(line[70:76])
            except ValueError:
                charge = 0.0
        atoms.append(
            PoseAtom(
                serial=serial,
                name=name,
                element=_element_from_pdbqt(name, ad_type),
                ad_type=ad_type,
                res_name=line[17:20].strip(),
                res_id=res_id,
                chain=chain,
                x=x,
                y=y,
                z=z,
                charge=charge,
            )
        )
    return atoms


def pdbqt_models(text: str) -> List[str]:
    """Split a multi-model PDBQT document into one document per pose.

    A document without ``MODEL`` records comes back as a single-element list, so
    a caller can always iterate.  The ``MODEL``/``ENDMDL`` records are kept: the
    kernel's ligand reader accepts them, and dropping them would lose the mode
    numbers a user sees in the file.
    """
    lines = str(text).splitlines()
    if not any(line.startswith("MODEL") for line in lines):
        return [str(text)] if str(text).strip() else []

    models: List[str] = []
    current: List[str] = []
    inside = False
    for line in lines:
        if line.startswith("MODEL"):
            if inside:
                models.append("\n".join(current) + "\n")
            current, inside = [line], True
        elif line.startswith("ENDMDL"):
            if inside:
                current.append(line)
                models.append("\n".join(current) + "\n")
            current, inside = [], False
        elif inside:
            current.append(line)
    if inside:  # a truncated final model without ENDMDL
        models.append("\n".join(current) + "\n")
    return [model for model in models if model.strip()]


def with_coordinates(text: str, coords) -> str:
    """Return `text` with the coordinates of its atom records replaced.

    The *k*-th ``ATOM``/``HETATM`` record receives ``coords[k]``; everything
    else -- topology, atom order, charges, types, remarks -- is untouched.  That
    is what makes a relaxed geometry comparable with the pose it came from: the
    force field sees exactly the same molecule, only moved.

    Raises
    ------
    ValueError
        When the number of coordinate rows does not match the number of atom
        records, or the coordinates are not ``(N, 3)``.  A silent partial patch
        would produce a physically meaningless hybrid.
    """
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"coordinates must have shape (N, 3), got {array.shape}")

    out: List[str] = []
    index = 0
    for line in str(text).splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if index >= array.shape[0]:
                raise ValueError(
                    f"the document has more atom records than the {array.shape[0]} "
                    "coordinate rows given"
                )
            x, y, z = (float(v) for v in array[index])
            padded = line if len(line) >= 54 else line.ljust(54)
            line = padded[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + padded[54:]
            index += 1
        out.append(line)
    if index != array.shape[0]:
        raise ValueError(
            f"the document has {index} atom records but {array.shape[0]} "
            "coordinate rows were given"
        )
    return "\n".join(out) + "\n"


def _pose_coords(pose) -> Optional[np.ndarray]:
    coords = getattr(pose, "coords", None)
    if coords is None:
        return None
    array = np.asarray(coords, dtype=float)
    return array if array.ndim == 2 and array.shape[1] == 3 else None


def pose_pdbqt_from_coords(template: str, pose) -> str:
    """Rebuild one pose's PDBQT from a topology `template` and a pose's coords.

    `template` is ``result.ligand_pdbqt`` (or any single-model ligand PDBQT) and
    `pose` is a :class:`odock.Pose` or an ``(N, 3)`` array in the kernel's atom
    order.  Returns ``""`` when either side is missing, so a caller can skip the
    pose instead of crashing a report.
    """
    coords = _pose_coords(pose) if not isinstance(pose, np.ndarray) else np.asarray(pose, float)
    if coords is None or not template:
        return ""
    if len(pdbqt_atoms(template)) != coords.shape[0]:
        return ""
    return with_coordinates(template, coords)


def _read_text(source: Union[str, os.PathLike]) -> str:
    """The text of `source`: the file's contents when it is a path, else itself."""
    raw = os.fspath(source) if isinstance(source, os.PathLike) else str(source)
    if raw.lstrip().startswith(("REMARK", "ATOM", "HETATM", "ROOT", "MODEL", "{")):
        return raw
    path = Path(raw)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return raw


def resolve_poses(poses, *, ligand_template: Optional[str] = None) -> List[str]:
    """Normalise `poses` into a list of single-pose PDBQT documents.

    Accepted inputs, in order of preference:

    * a :class:`odock.DockResult` -- each pose is rebuilt from
      ``ligand_pdbqt``/``pose.coords`` when possible, which is the only way to
      include poses the run's own multi-model file dropped beyond its energy
      window; falls back to splitting ``result.to_pdbqt()``;
    * a path to (or the text of) a multi-model PDBQT document;
    * a list of single-pose PDBQT documents (or of :class:`odock.Pose` objects,
      which then need `ligand_template`).
    """
    if poses is None:
        return []
    if hasattr(poses, "poses") and hasattr(poses, "to_pdbqt"):
        result = poses
        template = ligand_template or getattr(result, "ligand_pdbqt", "") or ""
        models: List[str] = []
        if template:
            for pose in list(getattr(result, "poses", None) or []):
                text = pose_pdbqt_from_coords(template, pose)
                if not text:
                    models = []
                    break
                models.append(text)
        if not models:
            models = pdbqt_models(result.to_pdbqt() or "")
        return models

    if isinstance(poses, (str, os.PathLike)):
        return pdbqt_models(_read_text(poses))

    if hasattr(poses, "coords"):  # a single Pose
        text = pose_pdbqt_from_coords(ligand_template or "", poses)
        return [text] if text else []

    items = list(poses)
    if items and all(isinstance(item, (str, os.PathLike)) for item in items):
        documents: List[str] = []
        for item in items:
            text = _read_text(item)
            # A list element that is itself a multi-model document is split, so
            # `[path]` and `path` behave the same way.
            documents.extend(pdbqt_models(text))
        return documents

    documents = []
    for item in items:
        if hasattr(item, "coords"):
            text = pose_pdbqt_from_coords(ligand_template or "", item)
            if not text:
                raise ValueError(
                    "a Pose was given without a ligand template; pass the "
                    "result's ligand_pdbqt as ligand_template="
                )
            documents.append(text)
        else:
            raise TypeError(f"cannot interpret {item!r} as a pose")
    return documents


def box_from_any(box):
    """Coerce `box` into an :class:`odock.BoxSpec`.

    Accepts a ``BoxSpec`` (returned unchanged), a path to a ``box.json``, a
    mapping with ``center``/``size``/``spacing`` (or the CLI's ``center_x``,
    ``size_x``, ``spacing`` spelling), or a ``((cx, cy, cz), (sx, sy, sz),
    spacing)`` tuple.  ``None`` becomes a 200 Å cube: big enough that no pose is
    ever outside it, which is the right default for pure rescoring where the box
    only contributes an out-of-box penalty.
    """
    if box is None:
        return BoxSpec(center=(0.0, 0.0, 0.0), size=(200.0, 200.0, 200.0), spacing=1.0)
    if isinstance(box, BoxSpec):
        return box
    if isinstance(box, (str, os.PathLike)):
        data = json.loads(_read_text(box))
        return box_from_any(data)
    if isinstance(box, dict):
        center = box.get("center")
        size = box.get("size")
        if center is None:
            center = tuple(float(box[k]) for k in ("center_x", "center_y", "center_z"))
        if size is None:
            size = tuple(float(box[k]) for k in ("size_x", "size_y", "size_z"))
        return BoxSpec(
            center=tuple(float(v) for v in center),
            size=tuple(float(v) for v in size),
            spacing=float(box.get("spacing", 0.375)),
        )
    if isinstance(box, (tuple, list)) and len(box) == 2:
        center, size = box
        return BoxSpec(
            center=tuple(float(v) for v in center), size=tuple(float(v) for v in size)
        )
    if isinstance(box, (tuple, list)) and len(box) == 3:
        center, size, spacing = box
        return BoxSpec(
            center=tuple(float(v) for v in center),
            size=tuple(float(v) for v in size),
            spacing=float(spacing),
        )
    raise TypeError(f"cannot interpret {box!r} as a search box")


# ---------------------------------------------------------------------------
# Ranking primitives
# ---------------------------------------------------------------------------


def average_ranks(values) -> np.ndarray:
    """Competition ranks (1 = smallest) with ties sharing their average rank.

    ``[10, 20, 20, 30]`` gives ``[1.0, 2.5, 2.5, 4.0]``.  This is the tie
    correction Spearman's rho is defined with, and the reason a rank-based
    consensus does not invent an ordering between two poses a force field scores
    identically.  ``nan`` inputs keep ``nan`` ranks and are excluded from the
    ordering; use :func:`normalised_ranks` when a total order is needed.
    """
    array = np.asarray(values, dtype=float).ravel()
    ranks = np.full(array.shape, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    if finite.size == 0:
        return ranks
    order = finite[np.argsort(array[finite], kind="stable")]
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and array[order[j + 1]] == array[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def normalised_ranks(values) -> np.ndarray:
    """Average ranks mapped onto ``[0, 1]``: 0 is the best pose, 1 the worst.

    ``(rank - 1) / (n - 1)`` over the poses that have a finite value; a single
    finite pose is given 0.0.  Ranks are computed over the *finite* values only,
    so a missing score neither shifts the other ranks nor earns a good one.
    """
    array = np.asarray(values, dtype=float).ravel()
    ranks = average_ranks(array)
    finite = np.isfinite(ranks)
    count = int(finite.sum())
    out = np.full(array.shape, np.nan, dtype=float)
    if count == 0:
        return out
    if count == 1:
        out[finite] = 0.0
        return out
    out[finite] = (ranks[finite] - 1.0) / (count - 1.0)
    return out


def z_scores(values) -> np.ndarray:
    """``(x - mean) / sd`` over the finite values, population sd (``ddof=0``).

    A constant field has zero spread, so every pose gets exactly 0.0 rather than
    an infinity: a force field that cannot separate the poses must not dominate
    the consensus.  ``nan`` inputs stay ``nan``.
    """
    array = np.asarray(values, dtype=float).ravel()
    out = np.full(array.shape, np.nan, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return out
    values_ = array[finite]
    mean = float(values_.mean())
    sd = float(values_.std())
    out[finite] = 0.0 if sd <= 0.0 else (values_ - mean) / sd
    return out


def spearman(a, b) -> float:
    """Spearman rank correlation of two score lists.

    Pearson's correlation of the average ranks -- the standard tie-corrected
    definition.  ``nan`` when fewer than two pairs are usable or when either
    list is constant (an undefined correlation, not a zero one).
    """
    first = np.asarray(a, dtype=float).ravel()
    second = np.asarray(b, dtype=float).ravel()
    if first.shape != second.shape:
        raise ValueError(
            f"the two score lists must have the same length, got "
            f"{first.shape[0]} and {second.shape[0]}"
        )
    usable = np.isfinite(first) & np.isfinite(second)
    if int(usable.sum()) < 2:
        return float("nan")
    ra = average_ranks(first[usable])
    rb = average_ranks(second[usable])
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denominator <= 0.0:
        return float("nan")
    return float((ra * rb).sum() / denominator)


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------


def _component_dict(components) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in dict(components).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def rescore_poses(
    poses,
    receptor,
    box=None,
    scorings: Sequence[str] = DEFAULT_SCORINGS,
    *,
    refine: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Rescore every pose with every force field, at fixed coordinates.

    No search and no refinement happen: the pose is evaluated exactly where the
    docking put it, with the kernel's exact pairwise scorer, so the only thing
    that changes between rows is the potential.

    Parameters
    ----------
    poses
        Anything :func:`resolve_poses` understands.
    receptor
        Receptor PDBQT text or a path to one.
    box
        Anything :func:`box_from_any` understands.  The box only contributes the
        out-of-box penalty; it must still be the box the run used, or a pose
        outside it would be charged a million kcal/mol per Ångström.
    scorings
        Force-field names, in the order the columns should appear.
    refine
        Passed to the kernel.  ``False`` (the default) is what "rescore" means:
        the coordinates are not touched.

    Returns
    -------
    ``{field: {"affinity": [...], "inter": [...], "intra": [...],
    "unbound": [...], "conf_independent": [...], "total": [...],
    "models": [...]}}``, one list entry per pose in input order.  A field the
    kernel returns a non-finite value for keeps the ``nan``; the caller decides
    what to do with it.
    """
    documents = resolve_poses(poses)
    receptor_text = _read_text(receptor)
    search_box = box_from_any(box)
    names = [str(name) for name in scorings]

    out: Dict[str, Dict[str, Any]] = {}
    for name in names:
        out[name] = {
            "affinity": [],
            "inter": [],
            "intra": [],
            "unbound": [],
            "conf_independent": [],
            "total": [],
            "models": documents,
        }
    for document in documents:
        for name in names:
            engine = _odock.Docking(
                receptor_text,
                document,
                center=tuple(float(v) for v in search_box.center),
                size=tuple(float(v) for v in search_box.size),
                spacing=float(search_box.spacing),
                scoring=name,
                use_grid=False,
                refine=bool(refine),
            )
            components = _component_dict(engine.score())
            column = out[name]
            for key in ("affinity", "inter", "intra", "unbound", "conf_independent", "total"):
                column[key].append(float(components.get(key, float("nan"))))
    return out


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


@dataclass
class ConsensusPose:
    """One pose's view across the force fields."""

    index: int
    #: Raw affinity per force field, in kcal/mol (``nan`` when unusable).
    scores: Dict[str, float] = field(default_factory=dict)
    #: The kernel's full component breakdown per force field.
    components: Dict[str, Dict[str, float]] = field(default_factory=dict)
    #: Average rank within each force field (1 = best).
    ranks: Dict[str, float] = field(default_factory=dict)
    #: Scale-free badness per force field, lower is better.
    badness: Dict[str, float] = field(default_factory=dict)
    #: Weighted mean badness -- the consensus score, lower is better.
    consensus_score: float = float("nan")
    #: Rank of ``consensus_score`` across the poses (1 = best).
    consensus_rank: int = 0
    #: Force fields that had no usable score for this pose.
    missing: Tuple[str, ...] = ()

    def row(self, prefix: str = "") -> Dict[str, Any]:
        """A flat dictionary suitable for a CSV/XLSX row."""
        data: Dict[str, Any] = {
            f"{prefix}mode": self.index + 1,
            f"{prefix}consensus_rank": self.consensus_rank,
            f"{prefix}consensus_score": self.consensus_score,
        }
        for name, value in self.scores.items():
            data[f"{prefix}{name}"] = value
            data[f"{prefix}{name}_rank"] = self.ranks.get(name, float("nan"))
        return data


@dataclass
class ChargeContext:
    """What the AD4 component's electrostatics can and cannot be trusted to mean.

    The AD4 kernel's electrostatic term is computed from the **PDBQT's** charges, so it
    inherits whatever protonation state and charge model the preparation produced and
    re-checks nothing.  The number is the kernel's and is not adjusted here; what this
    context fixes is the *report*: a consumer can no longer present an
    electrostatics-dependent number without saying whether the charges behind it can
    represent the formal charges the chemistry implies.

    ``ligand_known=False`` means the ligand's chemistry was not supplied, so the check
    could not be made at all — which is itself a caveat, and it is stated rather than
    left implicit.  See `docs/PROTONATION.md`.
    """

    #: Whether the ligand's chemistry was supplied at all.
    ligand_known: bool = False
    #: Whether a formal-charge correction was possible for this ligand
    #: (:func:`odock.protonation.can_represent_formal_charge` on the charges it was
    #: given), and whether this run applied one.
    correction_possible: bool = False
    correction_applied: bool = False
    #: A one-line statement of the charge state, for a table cell.
    state: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def needs_attention(self) -> bool:
        """Whether a report has to qualify its electrostatics.

        True when the ligand is unknown, or when its charges cannot represent the
        formal charges its groups imply and no correction was applied.
        """
        if not self.ligand_known:
            return True
        return bool(self.correction_possible and not self.correction_applied) or (
            bool(self.notes) and not self.correction_applied and self.state.startswith("cannot")
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ligand_known": bool(self.ligand_known),
            "correction_possible": bool(self.correction_possible),
            "correction_applied": bool(self.correction_applied),
            "needs_attention": bool(self.needs_attention),
            "state": self.state,
            "notes": list(self.notes),
        }


def charge_context(
    ligand=None,
    *,
    charges: Optional[Sequence[float]] = None,
    correction_applied: bool = False,
    ph: Optional[float] = None,
    name: str = "",
) -> ChargeContext:
    """What the AD4 component's charges can support, for one ligand.

    ``ligand`` is an RDKit molecule, a SMILES string or a path to a structure file;
    ``charges`` is the partial-charge set that went into the PDBQT (the file's AD4
    charges, usually).  Nothing is re-scored and no chemistry is changed: the AD4
    number is the kernel's, and this only says whether the charges behind it can
    represent the formal charges the chemistry implies.

    **The RDKit import is deferred to this function.**  This module is deliberately
    importable without RDKit (a CLI and the workbench both call it), and the whole
    protonation check lives in :mod:`odock.protonation`, which is where the charge
    reasoning belongs.
    """
    from . import protonation as _protonation

    context = ChargeContext(correction_applied=bool(correction_applied))
    if ligand is None:
        context.state = "the ligand's chemistry was not supplied"
        context.notes.append(
            "the AD4 component's electrostatics are computed from the PDBQT's charges, "
            "and this report was not given the ligand, so whether those charges can "
            "represent its formal charges could not be checked.  Pass ligand=<mol>, "
            "<SMILES> or a file path to have it checked (docs/PROTONATION.md)"
        )
        return context
    mol = ligand
    if isinstance(ligand, str):
        try:
            from rdkit import Chem as _Chem
        except Exception:  # pragma: no cover - RDKit is a hard dependency for chemistry
            context.state = "RDKit is unavailable, so the charges were not checked"
            context.notes.append(context.state)
            return context
        text = ligand.strip()
        if text and Path(text).exists():
            from .chem.ligand import read_ligands

            mols = read_ligands(text, embed=False)
            mol = mols[0] if mols else None
        else:
            mol = _Chem.MolFromSmiles(text)
    if mol is None:
        context.state = "the ligand could not be parsed"
        context.notes.append(
            "the ligand's chemistry could not be parsed, so the AD4 component's charges "
            "were not checked against its formal charges (docs/PROTONATION.md)"
        )
        return context
    label = str(name) or _protonation._mol_name(mol, "ligand")
    report = _protonation.protonation_report(
        mol, name=label, ph=ph if ph else _protonation.DEFAULT_PH
    )
    values = None
    if charges is not None:
        candidate = np.asarray(charges, dtype=float).reshape(-1)
        # The set has to line up with the molecule for any of this to mean anything; a
        # mismatched array is ignored rather than silently mis-indexed.
        values = candidate if candidate.shape[0] == mol.GetNumAtoms() else None
    capability = _protonation.can_represent_formal_charge(mol, values)
    context.ligand_known = True
    context.correction_possible = not capability.possible
    context.state = capability.statement()
    if capability.possible:
        context.notes.append(
            f"{label}: the charges behind the AD4 term can represent this molecule's "
            f"formal charges ({context.state})"
        )
        return context
    context.notes.append(
        f"{label}: {context.state}.  The AD4 number is the kernel's and is unchanged; "
        "what this says is that its electrostatic part was computed for a different "
        "species than the chemistry implies, so do not read the AD4 column as chemistry "
        "(docs/PROTONATION.md)"
    )
    for warning in report.warnings:
        context.notes.append(warning)
    return context


@dataclass
class ConsensusResult:
    """The ranked consensus of one pose set across several force fields.

    Attributes
    ----------
    poses
        One :class:`ConsensusPose` per input pose, **in consensus order** (best
        first), each keeping its original ``index``.
    scorings
        The force fields, in column order.
    method
        ``"rank"``, ``"borda"`` or ``"z"``.
    weights
        The per-field weight actually used (normalised to sum to 1).
    correlations
        ``[(field_a, field_b, rho), ...]`` for every unordered pair.
    charge
        A :class:`ChargeContext`: what the AD4 component's charges can support.  The
        AD4 number is not adjusted — this only stops the report implying more than
        the charges allow.
    """

    poses: List[ConsensusPose] = field(default_factory=list)
    scorings: Tuple[str, ...] = ()
    method: str = "rank"
    weights: Dict[str, float] = field(default_factory=dict)
    correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    charge: ChargeContext = field(default_factory=ChargeContext)

    # -- agreement ---------------------------------------------------------

    def correlation(self, a: str, b: str) -> float:
        """The Spearman rho between two force fields (``nan`` if unknown)."""
        for first, second, rho in self.correlations:
            if {first, second} == {a, b}:
                return rho
        return float("nan")

    @property
    def agreement(self) -> float:
        """The mean pairwise Spearman rho -- how much the fields agree at all.

        ``1.0`` means the three fields rank the poses identically, ``0.0`` means
        they are unrelated, a negative value means they actively disagree.  With
        a single force field there is no pair, so the property is ``nan``.
        """
        values = [rho for _a, _b, rho in self.correlations if math.isfinite(rho)]
        if not values:
            return float("nan")
        return float(np.mean(values))

    def matrix(self) -> np.ndarray:
        """The ``(k, k)`` Spearman matrix over :attr:`scorings`, diagonal 1.0."""
        size = len(self.scorings)
        out = np.full((size, size), np.nan, dtype=float)
        for i in range(size):
            out[i, i] = 1.0
            for j in range(i + 1, size):
                rho = self.correlation(self.scorings[i], self.scorings[j])
                out[i, j] = out[j, i] = rho
        return out

    # -- reporting ---------------------------------------------------------

    def rows(self, prefix: str = "") -> List[Dict[str, Any]]:
        """One flat dictionary per pose, in consensus order."""
        return [pose.row(prefix) for pose in self.poses]

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable view of the whole result."""
        return {
            "scorings": list(self.scorings),
            "method": self.method,
            "weights": dict(self.weights),
            "agreement": self.agreement,
            "charges": self.charge.as_dict(),
            "correlations": [
                {"a": a, "b": b, "rho": rho} for a, b, rho in self.correlations
            ],
            "poses": [
                {
                    "index": pose.index,
                    "mode": pose.index + 1,
                    "consensus_rank": pose.consensus_rank,
                    "consensus_score": pose.consensus_score,
                    "scores": dict(pose.scores),
                    "ranks": dict(pose.ranks),
                    "missing": list(pose.missing),
                }
                for pose in self.poses
            ],
        }

    def table(self, *, width: int = 10) -> str:
        """A plain-text table: per pose, each field's score and rank, consensus.

        One column per force field, each cell carrying the raw score with that
        field's rank in brackets, then the consensus rank.  This is the block a
        user pastes into a notebook; it is deliberately a fixed-width text table
        with no dependencies beyond the standard library.
        """
        header = ["mode", "consensus"] + [f"{name} (rank)" for name in self.scorings]
        lines = ["  ".join(f"{name:>{width}}" for name in header)]
        lines.append("  ".join("-" * width for _ in header))
        for pose in self.poses:
            cells = [
                f"{pose.index + 1:>{width}d}",
                f"{pose.consensus_score:>{width}.3f}",
            ]
            for name in self.scorings:
                value = pose.scores.get(name, float("nan"))
                rank = pose.ranks.get(name, float("nan"))
                if math.isfinite(value):
                    cells.append(f"{value:.3f} ({rank:.1f})".rjust(width))
                else:
                    cells.append(f"-- ({rank:.1f})".rjust(width))
            lines.append("  ".join(cells))
        return "\n".join(lines)


def _normalise_weights(scorings: Sequence[str], weights) -> Dict[str, float]:
    if weights is None:
        share = 1.0 / max(len(scorings), 1)
        return {name: share for name in scorings}
    if isinstance(weights, dict):
        raw = {name: float(weights.get(name, 0.0)) for name in scorings}
    else:
        values = [float(v) for v in weights]
        if len(values) != len(scorings):
            raise ValueError(
                f"{len(values)} weights were given for {len(scorings)} force fields"
            )
        raw = dict(zip(scorings, values))
    for name, value in raw.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"the weight of {name!r} must be a finite, non-negative number")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("the weights must not all be zero")
    return {name: value / total for name, value in raw.items()}


def consensus_score(
    poses,
    receptor,
    box=None,
    scorings: Sequence[str] = DEFAULT_SCORINGS,
    *,
    method: str = "rank",
    weights=None,
    components: Optional[Dict[str, Dict[str, Any]]] = None,
    ligand=None,
    ligand_charges: Optional[Sequence[float]] = None,
    formal_charge_correction_applied: bool = False,
) -> ConsensusResult:
    """Rescore `poses` with each force field and combine them into one ranking.

    Parameters
    ----------
    poses, receptor, box
        As for :func:`rescore_poses`.  `box` must be the box the run used.
    scorings
        Force fields to combine; any subset of ``("vina", "vinardo", "ad4")``,
        and repeats are allowed but pointless.
    method
        ``"rank"`` (normalised-rank sum, the default), ``"borda"`` (plain rank
        sum) or ``"z"`` (mean z-score).  All three are lower-is-better.
    weights
        Optional per-field weights, either a mapping or a sequence in the order
        of `scorings`.  Normalised to sum to 1.
    components
        A pre-computed :func:`rescore_poses` result, so a caller that has already
        paid for the rescoring does not pay twice.
    ligand, ligand_charges, formal_charge_correction_applied
        The ligand's chemistry (an RDKit molecule, a SMILES or a path) and the partial
        charges that went into its PDBQT.  Supplying them makes the report say what the
        AD4 component's electrostatics can support — the kernel's number is **not**
        adjusted — and leaving them out makes the report say that the check was not
        made.  See :func:`charge_context` and `docs/PROTONATION.md`.

    Returns
    -------
    :class:`ConsensusResult`, with the poses in consensus order.  With
    ``method="rank"`` (the default) the consensus score of a pose is the weighted
    mean of its normalised ranks, so it lives on a fixed ``[0, 1]`` scale no
    matter which force fields were asked for: 0.0 means the pose is best under
    every field, 1.0 means worst under every field.  ``"borda"`` reports the same
    mean on the raw rank scale ``[1, n]``, and ``"z"`` on a standard-deviation
    scale centred on 0.
    """
    names = tuple(str(name) for name in scorings)
    if not names:
        raise ValueError("at least one force field is required")
    if method not in ("rank", "borda", "z"):
        raise ValueError(
            f"unknown consensus method {method!r}; use 'rank', 'borda' or 'z'"
        )
    share = _normalise_weights(names, weights)

    raw = components if components is not None else rescore_poses(poses, receptor, box, names)
    affinities = {name: np.asarray(raw[name]["affinity"], dtype=float) for name in names}
    lengths = {int(array.size) for array in affinities.values()}
    if len(lengths) != 1:
        raise ValueError(f"the force fields returned different pose counts: {lengths}")
    count = lengths.pop()

    badness: Dict[str, np.ndarray] = {}
    ranks: Dict[str, np.ndarray] = {}
    for name in names:
        ranks[name] = average_ranks(affinities[name])
        if method == "z":
            badness[name] = z_scores(affinities[name])
        elif method == "borda":
            badness[name] = ranks[name]
        else:
            badness[name] = normalised_ranks(affinities[name])

    combos: List[ConsensusPose] = []
    for index in range(count):
        pose = ConsensusPose(index=index)
        terms: List[float] = []
        for name in names:
            value = float(affinities[name][index])
            pose.scores[name] = value
            pose.components[name] = {
                key: float(np.asarray(raw[name][key])[index])
                for key in (
                    "affinity",
                    "inter",
                    "intra",
                    "unbound",
                    "conf_independent",
                    "total",
                )
                if key in raw[name]
            }
            pose.ranks[name] = float(ranks[name][index])
            bad = float(badness[name][index])
            if not math.isfinite(bad):
                bad = 1.0
                pose.missing = pose.missing + (name,)
            pose.badness[name] = bad
            terms.append(share[name] * bad)
        pose.consensus_score = float(sum(terms))
        combos.append(pose)

    order = sorted(range(count), key=lambda k: (combos[k].consensus_score, combos[k].index))
    for rank, position in enumerate(order, start=1):
        combos[position].consensus_rank = rank
    ordered = [combos[k] for k in order]

    correlations: List[Tuple[str, str, float]] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            correlations.append((first, second, spearman(affinities[first], affinities[second])))

    return ConsensusResult(
        poses=ordered,
        scorings=names,
        method=method,
        weights=share,
        correlations=correlations,
        charge=charge_context(
            ligand, charges=ligand_charges,
            correction_applied=bool(formal_charge_correction_applied),
        ),
    )


# ---------------------------------------------------------------------------
# Alternative rankings built on the same poses
# ---------------------------------------------------------------------------


@dataclass
class EnergyRow:
    """The kernel's energy decomposition for one pose under one force field.

    The identities the kernel guarantees (see ``docs/SCORING.md``) are exposed
    as properties rather than recomputed at every call site:

    * ``base = inter + intra - unbound`` -- what the search minimised;
    * ``affinity = total`` -- the reported number, with the torsional term
      applied (a divisor for Vina/Vinardo, additive for AD4).
    """

    index: int
    scoring: str
    affinity: float = float("nan")
    inter: float = float("nan")
    intra: float = float("nan")
    unbound: float = float("nan")
    conf_independent: float = float("nan")

    @property
    def base(self) -> float:
        """``inter + intra - unbound``: the objective before the torsional term."""
        return self.inter + self.intra - self.unbound

    @property
    def torsion_term(self) -> float:
        """``affinity - base``: the conformational-entropy correction.

        Negative for Vina/Vinardo (the divisor makes the reported number smaller
        in magnitude), and for AD4 it is ``w_rot N_tors - intra``, exactly as the
        kernel defines ``ScoreComponents::conf_independent``.
        """
        return self.affinity - self.base

    def row(self, prefix: str = "") -> Dict[str, Any]:
        return {
            f"{prefix}mode": self.index + 1,
            f"{prefix}energy_{self.scoring}": self.affinity,
            f"{prefix}base_{self.scoring}": self.base,
            f"{prefix}inter_{self.scoring}": self.inter,
            f"{prefix}intra_{self.scoring}": self.intra,
            f"{prefix}unbound_{self.scoring}": self.unbound,
            f"{prefix}torsion_{self.scoring}": self.torsion_term,
        }


def decomposition_rows(result_or_components, *, pose_index=None) -> List[EnergyRow]:
    """Per-pose energy decomposition from a consensus result or a raw rescoring.

    `result_or_components` is either a :class:`ConsensusResult` (the normal
    case: the components come from whatever :func:`consensus_score` rescored) or
    the ``{field: {...}}`` mapping :func:`rescore_poses` returns.  `pose_index`
    renumbers the rows when the rescorings were of a subset -- with a
    :class:`ConsensusResult` the original ``index`` is kept either way, because
    a report has to be able to join the row back to its pose.
    """
    if isinstance(result_or_components, ConsensusResult):
        rows: List[EnergyRow] = []
        for pose in result_or_components.poses:
            for field_name, components in pose.components.items():
                rows.append(
                    EnergyRow(
                        index=pose.index,
                        scoring=field_name,
                        affinity=float(components.get("affinity", float("nan"))),
                        inter=float(components.get("inter", float("nan"))),
                        intra=float(components.get("intra", float("nan"))),
                        unbound=float(components.get("unbound", float("nan"))),
                        conf_independent=float(
                            components.get("conf_independent", float("nan"))
                        ),
                    )
                )
        return rows

    raw = result_or_components
    lengths = {len(raw[name]["affinity"]) for name in raw}
    count = lengths.pop() if len(lengths) == 1 else 0
    rows = []
    for position in range(count):
        for field_name in raw:
            column = raw[field_name]
            rows.append(
                EnergyRow(
                    index=position if pose_index is None else int(pose_index[position]),
                    scoring=field_name,
                    affinity=float(np.asarray(column["affinity"])[position]),
                    inter=float(np.asarray(column["inter"])[position]),
                    intra=float(np.asarray(column["intra"])[position]),
                    unbound=float(np.asarray(column["unbound"])[position]),
                    conf_independent=float(
                        np.asarray(column["conf_independent"])[position]
                    ),
                )
            )
    return rows


def decomposition_table(rows, *, width: int = 11) -> str:
    """A fixed-width text table of :class:`EnergyRow` values."""
    rows = list(rows)
    if not rows:
        return "no energy decomposition available"
    header = ["mode", "field", "affinity", "inter", "intra", "unbound", "torsion"]
    lines = ["  ".join(f"{name:>{width}}" for name in header)]
    lines.append("  ".join("-" * width for _ in header))
    for row in rows:
        lines.append(
            "  ".join(
                [
                    f"{row.index + 1:>{width}d}",
                    f"{row.scoring:>{width}}",
                    f"{row.affinity:>{width}.3f}",
                    f"{row.inter:>{width}.3f}",
                    f"{row.intra:>{width}.3f}",
                    f"{row.unbound:>{width}.3f}",
                    f"{row.torsion_term:>{width}.3f}",
                ]
            )
        )
    return "\n".join(lines)


@dataclass
class StrainRow:
    """One pose's affinity before and after the strain correction."""

    index: int
    affinity: float
    strain: float
    corrected: float
    rank: int = 0
    corrected_rank: int = 0


def strain_corrected_ranking(
    affinities,
    strains,
    *,
    indices=None,
) -> List[StrainRow]:
    """Rank poses by ``affinity + strain`` instead of by affinity alone.

    The correction is the one Vina's own convention leaves out: for Vina and
    Vinardo the kernel cancels the ligand's internal energy against the
    unbound reference *at the same conformation*, which assumes the free ligand
    already sits in the bound geometry.  A pose that reaches its contacts by
    folding into a strained conformation is therefore over-rewarded, and adding
    the strain back is the standard first-order fix.  For AD4 the intra term is
    not cancelled at all, so the correction should be applied only to a
    Vina/Vinardo score (or to a strain measured in AD4's own potential).

    Parameters
    ----------
    affinities, strains
        Equal-length sequences, in kcal/mol, one entry per pose.  A pose whose
        strain is not finite keeps its uncorrected affinity and is marked by
        ``strain = nan`` rather than being dropped.
    indices
        Optional original pose indices, for a table that has to join back.

    Returns
    -------
    :class:`StrainRow` objects, sorted by corrected affinity (best first), each
    carrying both ranks so the effect of the correction is visible.
    """
    values = np.asarray(affinities, dtype=float).ravel()
    penalty = np.asarray(strains, dtype=float).ravel()
    if values.shape != penalty.shape:
        raise ValueError(
            f"{values.shape[0]} affinities and {penalty.shape[0]} strains were given"
        )
    order = (
        np.arange(values.shape[0], dtype=int)
        if indices is None
        else np.asarray(indices, dtype=int).ravel()
    )
    if order.shape[0] != values.shape[0]:
        raise ValueError("indices must have one entry per pose")

    corrected = np.where(np.isfinite(penalty), values + penalty, values)
    rows = [
        StrainRow(
            index=int(order[i]),
            affinity=float(values[i]),
            strain=float(penalty[i]),
            corrected=float(corrected[i]),
        )
        for i in range(values.shape[0])
    ]
    for rank, position in enumerate(
        sorted(range(len(rows)), key=lambda k: (values[k], rows[k].index)), start=1
    ):
        rows[position].rank = rank
    for rank, position in enumerate(
        sorted(range(len(rows)), key=lambda k: (corrected[k], rows[k].index)), start=1
    ):
        rows[position].corrected_rank = rank
    return sorted(rows, key=lambda row: (row.corrected, row.index))


@dataclass
class ReproducibilityReport:
    """How stable the top binding mode is across independent runs.

    Attributes
    ----------
    n_runs, n_poses
        How many runs were compared and how many poses they contributed in
        total.
    cutoff
        The clustering RMSD (Å) that defines "the same binding mode".
    top_cluster
        Indices into the pooled pose list of every pose in the cluster that
        contains the overall best-scoring pose.
    top_cluster_runs
        The runs (0-based, in input order) that contributed at least one pose to
        that cluster, ascending.
    reproducibility
        ``len(top_cluster_runs) / n_runs`` -- the fraction of independent runs
        that found the winning mode.  This is the headline number: 1.0 means
        every seed converged to the same cluster, 1/9 means the top pose is a
        fluke of one seed.
    best_of_run_cluster
        For each run, the cluster (0-based, in the cluster ordering of
        :func:`odock.analysis.cluster_poses`) its best-scoring pose landed in.
    n_clusters
        How many clusters the pooled poses formed, and their sizes.
    """

    n_runs: int = 0
    n_poses: int = 0
    cutoff: float = 2.0
    top_cluster: List[int] = field(default_factory=list)
    top_cluster_runs: List[int] = field(default_factory=list)
    reproducibility: float = float("nan")
    best_of_run_cluster: List[int] = field(default_factory=list)
    cluster_sizes: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_sizes)

    @property
    def top_cluster_poses(self) -> int:
        return len(self.top_cluster)

    def table(self) -> str:
        runs = ", ".join(str(run + 1) for run in self.top_cluster_runs) or "none"
        return (
            f"runs                {self.n_runs}\n"
            f"poses               {self.n_poses}\n"
            f"cluster cutoff      {self.cutoff:.2f} A\n"
            f"clusters            {self.n_clusters} "
            f"(sizes {', '.join(str(s) for s in self.cluster_sizes[:8])}"
            + (", ..." if self.n_clusters > 8 else "")
            + ")\n"
            f"top cluster         {self.top_cluster_poses} poses from runs {runs}\n"
            f"reproducibility     {self.reproducibility:.2f} "
            f"({len(self.top_cluster_runs)}/{self.n_runs} runs found the top mode)"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_poses": self.n_poses,
            "cutoff": self.cutoff,
            "top_cluster_size": self.top_cluster_poses,
            "top_cluster_runs": self.top_cluster_runs,
            "reproducibility": self.reproducibility,
            "n_clusters": self.n_clusters,
            "cluster_sizes": self.cluster_sizes,
            "best_of_run_cluster": self.best_of_run_cluster,
            "notes": self.notes,
        }


def reproducibility(
    results,
    *,
    cutoff: float = 2.0,
    elements=None,
    energy_column: str = "affinity",
) -> ReproducibilityReport:
    """Cluster the poses of several independent runs and score the agreement.

    This answers the question a single docking run cannot: *is the top pose a
    property of the system or of the seed?*  Every pose of every run is pooled,
    clustered by symmetry-aware RMSD (:func:`odock.analysis.cluster_poses`), and
    the report says how many of the runs contributed a pose to the cluster that
    contains the overall best score.

    Parameters
    ----------
    results
        A sequence of :class:`odock.DockResult`, each from a different seed.  A
        run whose poses carry no coordinates is skipped with a note: clustering
        is impossible without them, and pretending otherwise would inflate the
        reproducibility.
    cutoff
        RMSD at which two poses count as the same binding mode (Å).
    elements
        Per-atom equivalence labels for the symmetry-aware RMSD, e.g.
        :func:`odock.analysis.symmetry_classes` of the prepared ligand.  Without
        them the matching is element-only.
    energy_column
        Which attribute of a pose to rank by; ``"affinity"`` by default.

    Returns
    -------
    :class:`ReproducibilityReport`.  With fewer than two usable runs the
    reproducibility is ``nan`` -- one run cannot be reproduced.
    """
    from . import analysis as _analysis

    pooled: List[np.ndarray] = []
    energies: List[float] = []
    owners: List[int] = []
    report = ReproducibilityReport(cutoff=float(cutoff))
    runs = list(results)
    for run_index, result in enumerate(runs):
        poses = list(getattr(result, "poses", None) or [])
        used = 0
        for pose in poses:
            coords = getattr(pose, "coords", None)
            if coords is None:
                continue
            array = np.asarray(coords, dtype=float)
            if array.ndim != 2 or array.shape[1] != 3:
                continue
            pooled.append(array)
            energies.append(float(getattr(pose, energy_column, float("nan")) or float("nan")))
            owners.append(run_index)
            used += 1
        if used == 0:
            report.notes.append(
                f"run {run_index + 1} contributed no pose with coordinates and was "
                "left out of the comparison"
            )

    report.n_poses = len(pooled)
    usable_runs = sorted(set(owners))
    report.n_runs = len(usable_runs)
    if report.n_poses == 0:
        report.notes.append("no pose carried coordinates")
        return report
    if report.n_runs < 2:
        report.notes.append(
            "fewer than two independent runs contributed poses, so no "
            "reproducibility can be measured"
        )
        return report

    sizes = {array.shape for array in pooled}
    if len(sizes) != 1:
        report.notes.append(
            f"the runs disagree about the ligand's atom count ({sorted(sizes)}); "
            "only the first size was clustered"
        )
    width = pooled[0].shape[0]
    keep = [i for i, array in enumerate(pooled) if array.shape[0] == width]
    coords = np.stack([pooled[i] for i in keep])
    ranked = [energies[i] for i in keep]
    run_of = [owners[i] for i in keep]

    clusters = _analysis.cluster_poses(
        coords, cutoff=cutoff, elements=elements, energies=ranked
    )
    report.cluster_sizes = [len(cluster.members) for cluster in clusters]
    if not clusters:
        return report

    # The cluster holding the overall best-scoring pose.
    best = min(range(len(ranked)), key=lambda i: (ranked[i], i))
    top = next(
        cluster for cluster in clusters if best in cluster.members
    )
    report.top_cluster = list(top.members)
    report.top_cluster_runs = sorted({run_of[i] for i in top.members})
    position_of_run = {run: position for position, run in enumerate(usable_runs)}
    report.reproducibility = len(report.top_cluster_runs) / report.n_runs

    cluster_of = {}
    for position, cluster in enumerate(clusters):
        for member in cluster.members:
            cluster_of[member] = position
    report.best_of_run_cluster = []
    for run in usable_runs:
        members = [i for i, owner in enumerate(run_of) if owner == run]
        if not members:
            report.best_of_run_cluster.append(-1)
            continue
        best_of_run = min(members, key=lambda i: (ranked[i], i))
        report.best_of_run_cluster.append(cluster_of.get(best_of_run, -1))
    del position_of_run
    return report

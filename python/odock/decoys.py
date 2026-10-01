# SPDX-License-Identifier: GPL-3.0-or-later
"""Property-matched decoy selection, and the evidence that it matched.

An enrichment factor is a ratio of small counts, and a decoy set that is easier
than the actives (smaller, less polar, less flexible) inflates it: the screen is
then being rewarded for finding the actives *bigger* or *more polar*, not for
finding them.  The matching is therefore the method, not a detail of it:

* :func:`property_profile` — the six properties every decoy is matched on:
  molecular weight, Crippen LogP, hydrogen-bond donors, acceptors, rotatable
  bonds and net formal charge;
* :func:`match_decoys` — for every active, pick up to ``per_active`` pool members
  within the stated tolerances **and** topologically dissimilar to every active
  (Morgan/Tanimoto below ``max_similarity``, so a decoy cannot be an analogue in
  disguise);
* :class:`DecoySet` — the assignment, the pool that was rejected and why, and
  :meth:`DecoySet.quality` — the two property distributions side by side with
  their standardised mean differences, which is the evidence that the matching
  worked.

What this does **not** do: it does not claim a decoy is inactive.  A molecule that
matches an active's properties and its topology is *presumed* inactive because
nobody has reported it binding that target; a curated pool from this repository
carries no assay data at all.  `docs/LBVS.md` states that plainly next to the
numbers it produces.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Crippen = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    Lipinski = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from .chem.ligand import rotatable_bonds as _rotatable_bonds

__all__ = [
    "PROPERTIES",
    "DEFAULT_TOLERANCES",
    "DEFAULT_MAX_SIMILARITY",
    "DEFAULT_PER_ACTIVE",
    "PropertyProfile",
    "Decoy",
    "DecoySet",
    "require_rdkit",
    "property_profile",
    "match_decoys",
]

#: The properties a decoy is matched on, in report order.  ``charge`` is the net
#: formal charge (usually 0, and the one property that must match *exactly* for a
#: charged active).
PROPERTIES: Tuple[str, ...] = ("MW", "LogP", "HBD", "HBA", "RotB", "charge")

#: Default tolerances: half a log unit of LogP is roughly a factor of three in
#: solubility, and the rest are the usual one-unit (or 10 %) windows.
DEFAULT_TOLERANCES: Dict[str, float] = {
    "MW": 25.0,
    "LogP": 0.5,
    "HBD": 1.0,
    "HBA": 1.0,
    "RotB": 1.0,
    "charge": 0.0,
}

#: A pool member whose Morgan/Tanimoto similarity to *any* active is at or above
#: this is not a decoy, it is an analogue.
DEFAULT_MAX_SIMILARITY = 0.35

#: Decoys per active.
DEFAULT_PER_ACTIVE = 5


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for decoy selection. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


def _crippen_logp(mol) -> float:
    """Crippen LogP, computed like :mod:`odock.filters` does (on an H-bearing copy).

    The same number the Lipinski filter reports, so a decoy set and a screening
    campaign cannot disagree about a molecule's LogP.
    """
    try:
        work = mol
        if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
            work = Chem.AddHs(mol)
        return float(Crippen.MolLogP(work))
    except Exception:  # pragma: no cover - defensive
        try:
            return float(Crippen.MolLogP(mol))
        except Exception:
            return float("nan")


@dataclass(frozen=True)
class PropertyProfile:
    """The six matching properties of one molecule."""

    MW: float
    LogP: float
    HBD: float
    HBA: float
    RotB: float
    charge: float

    def as_dict(self) -> Dict[str, float]:
        return {name: round(float(getattr(self, name)), 4) for name in PROPERTIES}

    def distance(self, other: "PropertyProfile", tolerances: Dict[str, float]) -> float:
        """The largest tolerance-scaled deviation, or ``inf`` when one is outside.

        ``0.0`` means an exact match on every property, ``1.0`` means every
        property is exactly at its tolerance.  Using the *maximum* (not a sum)
        keeps one badly matched property from being averaged away by five good
        ones.
        """
        worst = 0.0
        for name in PROPERTIES:
            tolerance = float(tolerances.get(name, 0.0))
            deviation = abs(float(getattr(self, name)) - float(getattr(other, name)))
            if tolerance <= 0.0:
                if deviation > 1e-9:
                    return math.inf
                continue
            worst = max(worst, deviation / tolerance)
        return worst


@dataclass
class Decoy:
    """One selected decoy and the active it was matched to."""

    index: int
    name: str
    active: str
    profile: PropertyProfile
    #: Tolerance-scaled property distance (0 = identical properties, 1 = at the
    #: tolerance of every property).
    distance: float
    #: Morgan/Tanimoto similarity to the *most similar* active.
    max_similarity: float
    smiles: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "name": self.name,
            "active": self.active,
            "properties": self.profile.as_dict(),
            "distance": round(float(self.distance), 4),
            "max_similarity": round(float(self.max_similarity), 4),
            "smiles": self.smiles,
        }


@dataclass
class DecoySet:
    """A matched decoy set: the decoys, the shortfalls, and the quality evidence."""

    decoys: List[Decoy] = field(default_factory=list)
    actives: List[str] = field(default_factory=list)
    n_pool: int = 0
    per_active: int = DEFAULT_PER_ACTIVE
    tolerances: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TOLERANCES))
    max_similarity: float = DEFAULT_MAX_SIMILARITY
    #: Pool members rejected because they were too similar to an active.
    rejected_similar: List[str] = field(default_factory=list)
    #: The similarity floor that defined the *hard* band (0.0 = no floor).
    min_similarity: float = 0.0
    #: Pool members rejected as too easy (below ``min_similarity``).
    rejected_easy: List[str] = field(default_factory=list)
    #: Pool members no active had room for (already outside the tolerances).
    rejected_property: List[str] = field(default_factory=list)
    #: Actives that could not be given ``per_active`` decoys, with the count found.
    shortfalls: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_decoys(self) -> int:
        return len(self.decoys)

    def names(self) -> List[str]:
        return [decoy.name for decoy in self.decoys]

    def active_names(self) -> List[str]:
        return list(self.actives)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_actives": len(self.actives),
            "n_decoys": int(self.n_decoys),
            "n_pool": int(self.n_pool),
            "per_active": int(self.per_active),
            "tolerances": {k: float(v) for k, v in self.tolerances.items()},
            "max_similarity": round(float(self.max_similarity), 4),
            "min_similarity": round(float(self.min_similarity), 4),
            "actives": list(self.actives),
            "decoys": [decoy.as_dict() for decoy in self.decoys],
            "rejected_similar": list(self.rejected_similar),
            "n_rejected_easy": len(self.rejected_easy),
            "n_rejected_property": len(self.rejected_property),
            "shortfalls": {k: int(v) for k, v in self.shortfalls.items()},
            "quality": self.quality(),
            "seconds": round(float(self.seconds), 4),
            "notes": list(self.notes),
        }

    # -- the evidence ------------------------------------------------------

    def quality(
        self, actives_profiles: Optional[Dict[str, PropertyProfile]] = None
    ) -> Dict[str, Any]:
        """The two property distributions side by side, and their separation.

        For every property: the active mean and spread, the decoy mean and
        spread, and the **standardised mean difference** (the difference of the
        means over the pooled standard deviation).  A set matched on the property
        has a small SMD; ``|SMD| > 0.5`` is the usual threshold for "these two
        groups are not matched", and it is reported honestly rather than hidden.
        """
        profiles = actives_profiles if actives_profiles is not None else self._active_profiles
        rows: Dict[str, Dict[str, float]] = {}
        for name in PROPERTIES:
            active_values = [float(getattr(profile, name)) for profile in profiles.values()]
            decoy_values = [float(getattr(decoy.profile, name)) for decoy in self.decoys]
            rows[name] = {
                "actives_mean": _mean(active_values),
                "actives_sd": _sd(active_values),
                "decoys_mean": _mean(decoy_values),
                "decoys_sd": _sd(decoy_values),
                "smd": _smd(active_values, decoy_values),
            }
        worst = max((abs(row["smd"]) for row in rows.values()), default=0.0)
        return {
            "properties": rows,
            "max_abs_smd": round(worst, 4),
            "matched": bool(worst <= 0.5),
        }

    _active_profiles: Dict[str, PropertyProfile] = field(default_factory=dict, repr=False)

    def table(self, limit: int = 0) -> str:
        """One line per decoy: name, the active it matches, distance, similarity."""
        rows = self.decoys if limit <= 0 else self.decoys[: int(limit)]
        if not rows:
            return "no decoy was selected"
        lines = [
            f"{'decoy':<26}{'active':<22}{'dist':>6}{'maxTan':>8}  properties",
            "-" * 26 + "-" * 22 + "-" * 6 + "-" * 8 + "  " + "-" * 40,
        ]
        for decoy in rows:
            props = decoy.profile.as_dict()
            text = " ".join(f"{name}={props[name]:g}" for name in PROPERTIES)
            lines.append(
                f"{decoy.name[:25]:<26}{decoy.active[:21]:<22}"
                f"{decoy.distance:>6.2f}{decoy.max_similarity:>8.2f}  {text}"
            )
        if limit > 0 and len(self.decoys) > limit:
            lines.append(f"... and {len(self.decoys) - limit} more decoy(s)")
        return "\n".join(lines)

    def quality_table(self) -> str:
        """The property-matching evidence, as a table."""
        report = self.quality()
        lines = [
            f"{'property':<9}{'actives':>18}{'decoys':>18}{'SMD':>8}",
            "-" * 9 + "-" * 18 + "-" * 18 + "-" * 8,
        ]
        for name in PROPERTIES:
            row = report["properties"][name]
            actives = f"{row['actives_mean']:.2f} +- {row['actives_sd']:.2f}"
            decoys = f"{row['decoys_mean']:.2f} +- {row['decoys_sd']:.2f}"
            lines.append(f"{name:<9}{actives:>18}{decoys:>18}{row['smd']:>8.2f}")
        lines.append(
            f"max |SMD| {report['max_abs_smd']:.2f}: "
            + ("matched" if report["matched"] else "NOT matched (|SMD| > 0.5)")
        )
        return "\n".join(lines)


def _mean(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    return sum(clean) / len(clean) if clean else float("nan")


def _sd(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return 0.0
    mean = sum(clean) / len(clean)
    return math.sqrt(sum((v - mean) ** 2 for v in clean) / (len(clean) - 1))


def _smd(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference with the pooled standard deviation."""
    left = [v for v in a if not math.isnan(v)]
    right = [v for v in b if not math.isnan(v)]
    if not left or not right:
        return float("nan")
    mean_a, mean_b = _mean(left), _mean(right)
    var_a = _sd(left) ** 2
    var_b = _sd(right) ** 2
    pooled = math.sqrt(((len(left) - 1) * var_a + (len(right) - 1) * var_b) / max(1, len(left) + len(right) - 2))
    if pooled <= 0:
        return 0.0 if abs(mean_a - mean_b) <= 1e-12 else math.inf
    return (mean_a - mean_b) / pooled


def property_profile(mol) -> PropertyProfile:
    """The six matching properties of ``mol``.

    MW, HBD and HBA come from RDKit's descriptors; LogP is the same Crippen model
    :mod:`odock.filters` uses; RotB is :func:`odock.chem.ligand.rotatable_bonds`,
    so the count is the torsions the docking engine would actually sample; charge
    is the net formal charge.
    """
    require_rdkit()
    return PropertyProfile(
        MW=float(Descriptors.MolWt(mol)),
        LogP=_crippen_logp(mol),
        HBD=float(Lipinski.NumHDonors(mol)),
        HBA=float(Lipinski.NumHAcceptors(mol)),
        RotB=float(len(_rotatable_bonds(mol))),
        charge=float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    )


def _mol_name(mol, fallback: str) -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def match_decoys(
    actives: Sequence[Any],
    pool: Sequence[Any],
    *,
    per_active: int = DEFAULT_PER_ACTIVE,
    tolerances: Optional[Dict[str, float]] = None,
    max_similarity: float = DEFAULT_MAX_SIMILARITY,
    min_similarity: float = 0.0,
    max_distance: float = 1.0,
    names: Optional[Sequence[str]] = None,
    pool_names: Optional[Sequence[str]] = None,
    seed: int = 20240101,
) -> DecoySet:
    """Select property-matched, topologically dissimilar decoys for ``actives``.

    Parameters
    ----------
    actives
        The molecules the decoys must be matched to (molecules or SMILES).
    pool
        The pool to select from (molecules or SMILES).  Nothing in it is claimed
        to be inactive; it is a pool of *candidates*.
    per_active
        How many decoys each active should get.  When the pool cannot supply that
        many within the tolerances, the shortfall is reported per active instead
        of the tolerance being quietly relaxed.
    tolerances
        Per-property windows (see :data:`DEFAULT_TOLERANCES`); ``charge`` defaults
        to an exact match.
    max_similarity
        Morgan/Tanimoto ceiling against every active.  A pool member at or above
        it is rejected as an analogue, and the rejected names are listed.
    min_similarity
        The *floor*: a pool member below it is too easy and is discarded.  The
        default 0.0 accepts everything that passes the ceiling, which on a
        congeneric active set produces a decoy set that any 2-D method separates
        perfectly (measured: see `docs/LBVS.md`).  Set it to build the **hard**
        decoy band — similar topology to the actives without being analogues —
        which is the set a benchmark has to be able to lose on.
    max_distance
        The tolerance-scaled property distance a candidate may have and still be a
        decoy.  ``1.0`` (the default) means *inside* every tolerance; relaxing it
        is possible but then the quality report will show the mismatch, which is
        the point of reporting it.
    seed
        Ties between equally good candidates are broken by a seeded shuffle, so a
        set is reproducible and not an artefact of the pool's order.

    Returns
    -------
    :class:`DecoySet` with the decoys, the rejections, the shortfalls and the
    property-matching evidence (:meth:`DecoySet.quality`).
    """
    require_rdkit()
    from .ligandsim import fingerprint, tanimoto

    start = time.perf_counter()
    selected_tolerances = dict(DEFAULT_TOLERANCES)
    if tolerances:
        selected_tolerances.update({k: float(v) for k, v in tolerances.items()})
    if int(per_active) < 1:
        raise ValueError(f"per_active must be >= 1, got {per_active!r}")

    active_items = list(actives)
    pool_items = list(pool)
    if not active_items:
        raise ValueError("no active to match decoys to")
    if not pool_items:
        raise ValueError("the pool is empty")

    active_mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in active_items]
    pool_mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in pool_items]
    for label, mols in (("active", active_mols), ("pool", pool_mols)):
        for index, mol in enumerate(mols):
            if mol is None:
                raise ValueError(f"{label} {index + 1} is not a parsable molecule")

    active_labels = [
        str(names[i]) if names is not None and i < len(names) else _mol_name(mol, f"active_{i + 1}")
        for i, mol in enumerate(active_mols)
    ]
    pool_labels = [
        str(pool_names[i]) if pool_names is not None and i < len(pool_names)
        else _mol_name(mol, f"pool_{i + 1}")
        for i, mol in enumerate(pool_mols)
    ]

    active_print = [
        fingerprint(mol, kind="morgan", radius=2, name=active_labels[i])
        for i, mol in enumerate(active_mols)
    ]
    active_profiles = {label: property_profile(mol) for label, mol in zip(active_labels, active_mols)}

    pool_print = []
    pool_profiles: List[PropertyProfile] = []
    for mol in pool_mols:
        pool_print.append(fingerprint(mol, kind="morgan", radius=2))
        pool_profiles.append(property_profile(mol))

    rejected_similar: List[str] = []
    rejected_easy: List[str] = []
    candidates: List[Tuple[int, Dict[str, float]]] = []
    for index, fp in enumerate(pool_print):
        worst = max(tanimoto(fp, other) for other in active_print)
        if worst >= float(max_similarity):
            rejected_similar.append(pool_labels[index])
            continue
        if worst < float(min_similarity):
            rejected_easy.append(pool_labels[index])
            continue
        distances = {
            label: pool_profiles[index].distance(active_profiles[label], selected_tolerances)
            for label in active_labels
        }
        candidates.append((index, distances))

    order = list(range(len(candidates)))
    random.Random(int(seed)).shuffle(order)
    # Global greedy matching: take the (active, candidate) pair with the smallest
    # tolerance-scaled property distance, over *all* actives at once.  A
    # per-active greedy pass would let the first active consume the only good
    # match for the last one, which is what makes a decoy set drift.
    pairs: List[Tuple[float, int, int]] = []
    for position, (index, distances) in enumerate(candidates):
        for active_index, label in enumerate(active_labels):
            distance = distances[label]
            if not math.isfinite(distance) or distance > float(max_distance) + 1e-9:
                continue
            pairs.append((float(distance), active_index, position))
    pairs.sort(key=lambda item: (item[0], order[item[2]], item[1]))
    chosen: Dict[str, List[Tuple[int, float, float]]] = {label: [] for label in active_labels}
    taken: set = set()
    for distance, active_index, position in pairs:
        label = active_labels[active_index]
        if len(chosen[label]) >= int(per_active):
            continue
        index, _ = candidates[position]
        if index in taken:
            continue
        taken.add(index)
        similarity = max(float(tanimoto(pool_print[index], other)) for other in active_print)
        chosen[label].append((index, distance, similarity))

    # Set-level refinement: per-active matching only guarantees that each decoy is
    # inside *its* active's tolerances; the aggregate distributions can still
    # drift, which is exactly what inflates an enrichment factor.  Repeatedly swap
    # in the unused candidate that reduces the worst standardised mean difference,
    # keeping every swap inside the per-active tolerances.
    def worst_smd(indices: Sequence[int]) -> float:
        values = {
            name: ([float(getattr(profile, name)) for profile in active_profiles.values()],
                   [float(getattr(pool_profiles[i], name)) for i in indices])
            for name in PROPERTIES
        }
        worst = 0.0
        for left, right in values.values():
            if not right:
                return math.inf
            score = _smd(left, right)
            if math.isfinite(score):
                worst = max(worst, abs(score))
        return worst

    def current_indices() -> List[int]:
        return [index for label in active_labels for index, _, _ in chosen[label]]

    best_score = worst_smd(current_indices())
    for _ in range(200):
        improved = False
        for label in active_labels:
            distances = {index: value for index, value, _ in chosen[label]}
            for slot, (old_index, old_distance, old_similarity) in enumerate(chosen[label]):
                for position, (index, candidate_distances) in enumerate(candidates):
                    if index in taken:
                        continue
                    distance = candidate_distances[label]
                    if not math.isfinite(distance) or distance > float(max_distance) + 1e-9:
                        continue
                    trial = current_indices()
                    trial.remove(old_index)
                    trial.append(index)
                    score = worst_smd(trial)
                    if score < best_score - 1e-9:
                        taken.discard(old_index)
                        taken.add(index)
                        similarity = max(
                            float(tanimoto(pool_print[index], other)) for other in active_print
                        )
                        chosen[label][slot] = (index, float(distance), similarity)
                        best_score = score
                        improved = True
                        break
                else:
                    continue
                break
            if improved:
                break
        if not improved:
            break

    decoys: List[Decoy] = []
    shortfalls: Dict[str, int] = {}
    for label in active_labels:
        if len(chosen[label]) < int(per_active):
            shortfalls[label] = len(chosen[label])
        for index, distance, similarity in chosen[label]:
            decoys.append(
                Decoy(
                    index=index,
                    name=pool_labels[index],
                    active=label,
                    profile=pool_profiles[index],
                    distance=float(distance),
                    max_similarity=float(similarity),
                    smiles=_safe_smiles(pool_mols[index]),
                )
            )
    decoys.sort(key=lambda decoy: (active_labels.index(decoy.active), decoy.distance, decoy.name))
    rejected_property = [
        pool_labels[index]
        for index, distances in candidates
        if index not in taken and all(not math.isfinite(v) for v in distances.values())
    ]
    notes: List[str] = []
    if shortfalls:
        notes.append(
            "the pool could not supply every active with "
            f"{int(per_active)} decoy(s) inside the tolerances: "
            + "; ".join(f"{label} got {count}" for label, count in shortfalls.items())
            + ". Report the shortfall rather than widening the tolerances."
        )
    if rejected_similar:
        notes.append(
            f"{len(rejected_similar)} pool member(s) were rejected as analogues "
            f"(Tanimoto >= {float(max_similarity):.2f} to an active): "
            + ", ".join(rejected_similar[:6])
            + (" ..." if len(rejected_similar) > 6 else "")
        )
    if float(min_similarity) > 0:
        notes.append(
            f"the similarity floor was {float(min_similarity):.2f}, so {len(rejected_easy)} "
            "member(s) were discarded as too easy: this is the *hard* decoy band, the one a "
            "method has to be able to lose on"
        )
    if not decoys:
        notes.append(
            "no decoy at all: the pool is too small, too dissimilar in properties, "
            "or every member is an analogue of an active"
        )
    result = DecoySet(
        decoys=decoys,
        actives=active_labels,
        n_pool=len(pool_mols),
        per_active=int(per_active),
        tolerances=selected_tolerances,
        max_similarity=float(max_similarity),
        min_similarity=float(min_similarity),
        rejected_similar=rejected_similar,
        rejected_easy=rejected_easy,
        rejected_property=rejected_property,
        shortfalls=shortfalls,
        seconds=time.perf_counter() - start,
        notes=notes,
    )
    result._active_profiles = active_profiles
    return result


def _safe_smiles(mol) -> str:
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - defensive
        return ""


def load_pool(path: Union[str, Path], *, fmt: Optional[str] = None) -> List[Any]:
    """Read a decoy pool (any format :mod:`odock.chem.ligand` reads)."""
    from .chem.ligand import read_ligands

    return read_ligands(path, fmt=fmt, embed=False)

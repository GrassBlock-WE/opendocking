# SPDX-License-Identifier: GPL-3.0-or-later
"""Pharmacophore models: build one from evidence, score a library, exclude volume.

A pharmacophore is the ligand-based counterpart of docking: instead of asking
what the receptor does with a molecule, it asks whether the molecule can present
the chemical features that the *known* ligands share, in the same relative
geometry.  This module implements the whole small-molecule workflow:

* :func:`features_from_mol` — perceive donor / acceptor / aromatic / hydrophobic /
  positive / negative features on one 3-D structure, with their positions;
* :func:`build_model` — derive the features that **recur across a set of members**
  (a series, a scaffold group, the top hits of a screen), reporting how many
  members support each one, and optionally a shape constraint derived from the
  recurring scaffold;
* :func:`fit_score` / :func:`screen` — fit a library molecule to the model (rigid
  alignment on the model's features or its core, feature matching with a
  documented tolerance and a miss penalty), rank the library and enforce the
  excluded volume;
* :func:`enrichment` — the labelled-set numbers (precision@k, recall@k, enrichment
  factor, AUC) *with* the sample size they were computed on, because on a
  seventeen-molecule library an enrichment factor is a smoke test, not a result.

What a pharmacophore model is **not**: it is a *hypothesis* derived from the
members it was given.  It cannot contain a feature the actives do not share, it
cannot distinguish an active from a decoy that presents the same features, and a
high fit score is a geometric statement, never an activity.  A model built from
one molecule is an anecdote — :func:`build_model` says so in its notes and
:data:`MIN_MEANINGFUL_MEMBERS` names the smallest set this project treats as
evidence.  `docs/PHARMACOPHORE.md` has the measured behaviour and the limits.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, ChemicalFeatures

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - RDKit is a hard dependency for chemistry
    Chem = None  # type: ignore[assignment]
    RDConfig = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    ChemicalFeatures = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

__all__ = [
    "FEATURE_FAMILIES",
    "MIN_MEANINGFUL_MEMBERS",
    "DEFAULT_RADIUS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOLERANCE",
    "DEFAULT_MISS_PENALTY",
    "DEFAULT_EXCLUDED_RADIUS",
    "HAVE_RDKIT",
    "PointFeature",
    "ExclusionSphere",
    "PharmacophoreModel",
    "FitResult",
    "PharmacophoreHits",
    "require_rdkit",
    "features_from_mol",
    "build_model",
    "fit_score",
    "screen",
    "enrichment",
]

#: The feature families a model can contain, in report order.
FEATURE_FAMILIES: Tuple[str, ...] = (
    "donor",
    "acceptor",
    "aromatic",
    "hydrophobic",
    "positive",
    "negative",
)

#: How the RDKit feature families map onto the six above.  ``None`` means the
#: family is dropped, with a note: a zinc-binding site is a receptor property and
#: has no meaning in a ligand-only model.
FAMILY_MAP: Dict[str, Optional[str]] = {
    "Donor": "donor",
    "Acceptor": "acceptor",
    "Aromatic": "aromatic",
    "Hydrophobe": "hydrophobic",
    "LumpedHydrophobe": "hydrophobic",
    "PosIonizable": "positive",
    "NegIonizable": "negative",
    "ZnBinder": None,
}

#: The smallest member set this project treats as evidence for a model.
#:
#: One or two molecules can only say "these features exist"; they cannot separate
#: a feature that *recurs* from one that belongs to a single member.  A model
#: built from fewer than this many members is still produced (it is useful for
#: inspecting a single hit) but it carries a note and
#: :attr:`PharmacophoreModel.is_meaningful` is false.
MIN_MEANINGFUL_MEMBERS = 3

#: Default cluster radius for a model feature, in Å.  Two features of the same
#: family from different members within this distance are treated as the same
#: feature; 1.5 Å is the usual pharmacophore tolerance.
DEFAULT_RADIUS = 1.5

#: Default minimum fraction of members that must support a feature.
DEFAULT_THRESHOLD = 0.5

#: Default matching tolerance when fitting a molecule to a model (Å).
DEFAULT_TOLERANCE = 1.5

#: Default miss penalty: a missed feature costs this many features' worth of
#: score.  1.0 makes ``fit = (sum of matched weights - n_missed) / n_features``.
DEFAULT_MISS_PENALTY = 1.0

#: Radius of one exclusion sphere derived from the recurring scaffold (Å).
DEFAULT_EXCLUDED_RADIUS = 1.0

#: Default shape mode: the actives' heavy-atom envelope.
DEFAULT_SHAPE = "envelope"

#: How far outside the envelope a candidate atom may sit before it counts as
#: outside (Å).  A little slack is deliberate: the members were embedded at one
#: seed each, so their envelope is a point set and not a surface.
DEFAULT_SHAPE_TOLERANCE = 1.2

#: Fraction of a candidate's heavy atoms that must lie inside the envelope.
DEFAULT_MIN_COVERAGE = 0.6

#: Radius used to cluster recurring scaffold atoms into exclusion spheres.
_EXCLUDED_CLUSTER_RADIUS = 0.75


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for pharmacophore models. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Feature perception
# ---------------------------------------------------------------------------

_FACTORY_CACHE: Dict[str, Any] = {}
_FALLBACK_SMARTS: Dict[str, Tuple[str, ...]] = {
    # A reduced, curated definition set, used only when this RDKit build has no
    # `BaseFeatures.fdef`.  It is deliberately small and named, and the model
    # records that it was used.
    "donor": ("[$([O,S,N;!H0]);!$([N;H0]=*)]",),
    "acceptor": ("[$([O,S;!H0]);!$([O,S;H1]=*)].", "[$([N;H2,H1,H0]);!$([N;H0]=*)]"),
    "aromatic": ("a1aaaaa1", "a1aaaa1"),
    "hydrophobic": ("[c,C;R]",),
    "positive": ("[$([N;H0,H1,H2;!$(N=*)]),$([n;H0])]",),
    "negative": ("[$([O;H0,-1]),$([S;H0,-1])]",),
}


def _feature_factory():
    """RDKit's default feature factory, or ``None`` when the build has none."""
    if "factory" in _FACTORY_CACHE:
        return _FACTORY_CACHE["factory"]
    factory = None
    try:
        import os

        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        if os.path.exists(fdef):
            factory = ChemicalFeatures.BuildFeatureFactory(fdef)
    except Exception:  # pragma: no cover - trimmed RDKit build
        factory = None
    _FACTORY_CACHE["factory"] = factory
    return factory


def _elements(mol) -> List[str]:
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def features_from_mol(
    mol,
    *,
    conf_id: int = 0,
    keep_zn_binder: bool = False,
) -> List[Tuple[str, Tuple[float, float, float], str]]:
    """Perceive the pharmacophore features of one 3-D conformer.

    Returns ``[(family, (x, y, z), type_name), ...]`` with the six families of
    :data:`FEATURE_FAMILIES`.  The positions are the ones RDKit's feature factory
    assigns: an atom position for a single-atom feature (a donor, an acceptor, an
    ionisable group) and the centroid for a multi-atom one (an aromatic ring, a
    lumped hydrophobe).

    Notes
    -----
    * The default factory reports one ``Hydrophobe`` per ring atom *and* one
      ``LumpedHydrophobe`` for the ring.  The per-atom ones are dropped when a
      lumped feature already covers those atoms, so an aromatic ring contributes
      **one** hydrophobic feature rather than six — the granularity a
      pharmacophore is read at.
    * ``ZnBinder`` is dropped by default: a zinc-binding site is a property of the
      receptor, not of the ligand, so it cannot be part of a ligand-derived model.
      Pass ``keep_zn_binder=True`` to keep it as a `"negative"`-family feature.
    """
    require_rdkit()
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "pharmacophore features need 3-D coordinates; embed the molecule first "
            "(odock.chem.ligand.embed_3d)"
        )
    if conf_id < 0 or conf_id >= mol.GetNumConformers():
        raise IndexError(
            f"conformer {conf_id} does not exist (the molecule has {mol.GetNumConformers()})"
        )
    conf = mol.GetConformer(int(conf_id))
    out: List[Tuple[str, Tuple[float, float, float], str]] = []
    lumped_atoms: List[set] = []
    raw: List[Tuple[str, str, Tuple[float, float, float], set]] = []
    factory = _feature_factory()
    if factory is not None:
        for feature in factory.GetFeaturesForMol(mol, confId=int(conf_id)):
            family = FAMILY_MAP.get(feature.GetFamily())
            if feature.GetFamily() == "ZnBinder" and keep_zn_binder:
                family = "negative"
            if family is None:
                continue
            position = feature.GetPos()
            atoms = set(int(i) for i in feature.GetAtomIds())
            raw.append(
                (
                    family,
                    str(feature.GetType()),
                    (float(position.x), float(position.y), float(position.z)),
                    atoms,
                )
            )
    else:  # pragma: no cover - exercised only on a build without BaseFeatures.fdef
        for family, patterns in _FALLBACK_SMARTS.items():
            for pattern in patterns:
                query = Chem.MolFromSmarts(pattern)
                if query is None:
                    continue
                for match in mol.GetSubstructMatches(query):
                    atoms = set(int(i) for i in match)
                    position = np.mean(
                        [
                            [
                                conf.GetAtomPosition(i).x,
                                conf.GetAtomPosition(i).y,
                                conf.GetAtomPosition(i).z,
                            ]
                            for i in sorted(atoms)
                        ],
                        axis=0,
                    )
                    raw.append(
                        (
                            family,
                            f"SMARTS:{pattern}",
                            (float(position[0]), float(position[1]), float(position[2])),
                            atoms,
                        )
                    )
    if factory is not None:
        # The per-atom hydrophobes a lumped feature already covers are noise.
        lumped = [atoms for family, _, _, atoms in raw if family == "hydrophobic" and len(atoms) > 1]
        kept: List[Tuple[str, str, Tuple[float, float, float], set]] = []
        for family, name, position, atoms in raw:
            if (
                family == "hydrophobic"
                and len(atoms) == 1
                and any(atoms <= bigger for bigger in lumped)
            ):
                continue
            kept.append((family, name, position, atoms))
        raw = kept
    for family, name, position, atoms in raw:
        out.append((family, position, name))
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class PointFeature:
    """One feature of a model: a family, a position and a tolerance."""

    #: One of :data:`FEATURE_FAMILIES`.
    family: str
    x: float
    y: float
    z: float
    #: Tolerance: a library feature this close (or closer) matches this one.
    radius: float = DEFAULT_RADIUS
    #: How many *distinct members* contributed a feature to this model feature.
    support: int = 0
    #: ``support / n_members``.
    fraction: float = 0.0
    #: Library indices of the supporting members.
    members: Tuple[int, ...] = ()
    #: Human-readable label, e.g. ``"donor#1"``.
    label: str = ""

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def distance_to(self, other: "PointFeature") -> float:
        return math.dist(self.position, other.position)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "label": self.label,
            "position": [round(self.x, 4), round(self.y, 4), round(self.z, 4)],
            "radius": round(float(self.radius), 4),
            "support": int(self.support),
            "fraction": round(float(self.fraction), 4),
            "members": [int(i) for i in self.members],
        }


@dataclass
class ExclusionSphere:
    """One excluded-volume sphere: space the model's members occupy."""

    x: float
    y: float
    z: float
    radius: float = DEFAULT_EXCLUDED_RADIUS
    support: int = 0
    fraction: float = 0.0
    members: Tuple[int, ...] = ()
    label: str = ""

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "position": [round(self.x, 4), round(self.y, 4), round(self.z, 4)],
            "radius": round(float(self.radius), 4),
            "support": int(self.support),
            "fraction": round(float(self.fraction), 4),
            "members": [int(i) for i in self.members],
        }


@dataclass
class PharmacophoreModel:
    """A pharmacophore hypothesis: recurring features plus a shape constraint.

    The shape constraint is either an **envelope** — the heavy-atom positions of
    every member, which a candidate has to sit inside — or a set of **exclusion
    spheres** on the recurring scaffold atoms that carry no feature.  The envelope
    is the default because it is the constraint that actually bites: a small rigid
    series' features cover its own atoms, so the sphere construction can come out
    empty (which the model then says in a note rather than pretending).
    """

    features: List[PointFeature] = field(default_factory=list)
    excluded: List[ExclusionSphere] = field(default_factory=list)
    #: ``[[x, y, z], ...]``: the heavy atoms of every member, in the model frame.
    envelope: List[List[float]] = field(default_factory=list)
    #: ``"envelope"``, ``"spheres"`` or ``"none"``.
    shape_mode: str = "envelope"
    #: How far outside the envelope a candidate atom may sit (Å).
    shape_tolerance: float = DEFAULT_SHAPE_TOLERANCE
    #: Fraction of a candidate's heavy atoms that must lie inside the envelope.
    min_coverage: float = DEFAULT_MIN_COVERAGE
    #: How many members the model was built from.
    n_members: int = 0
    #: How many members actually contributed (the rest were dropped).
    n_used: int = 0
    #: The minimum support fraction a feature had to reach.
    threshold: float = DEFAULT_THRESHOLD
    #: The cluster radius used to merge features across members.
    radius: float = DEFAULT_RADIUS
    #: The core the members were superposed on, when one was used.
    core: str = ""
    #: Coordinates of the core atoms in the model frame, in the core SMILES'
    #: atom order — the reference a library molecule is superposed onto.
    core_coords: List[List[float]] = field(default_factory=list)
    #: Element symbols of those core atoms.
    core_elements: List[str] = field(default_factory=list)
    source: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def is_meaningful(self) -> bool:
        """Whether enough members back the model to call a feature recurring."""
        return self.n_used >= MIN_MEANINGFUL_MEMBERS

    @property
    def has_shape(self) -> bool:
        return self.shape_mode == "spheres" and bool(self.excluded) or (
            self.shape_mode == "envelope" and bool(self.envelope)
        )

    def families(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for feature in self.features:
            counts[feature.family] = counts.get(feature.family, 0) + 1
        return counts

    def support_profile(self) -> List[int]:
        return [feature.support for feature in self.features]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_members": int(self.n_members),
            "n_used": int(self.n_used),
            "n_features": int(self.n_features),
            "threshold": round(float(self.threshold), 4),
            "radius": round(float(self.radius), 4),
            "core": self.core,
            "core_coords": [[round(v, 4) for v in row] for row in self.core_coords],
            "core_elements": list(self.core_elements),
            "shape_mode": self.shape_mode,
            "shape_tolerance": round(float(self.shape_tolerance), 4),
            "min_coverage": round(float(self.min_coverage), 4),
            "n_envelope_points": len(self.envelope),
            "envelope": [[round(float(v), 3) for v in row] for row in self.envelope],
            "source": self.source,
            "meaningful": bool(self.is_meaningful),
            "families": self.families(),
            "features": [feature.as_dict() for feature in self.features],
            "excluded": [sphere.as_dict() for sphere in self.excluded],
            "notes": list(self.notes),
        }

    def table(self) -> str:
        """A text table of the model: label, family, position, support."""
        lines = [
            f"pharmacophore model: {self.n_features} feature(s) from {self.n_used} of "
            f"{self.n_members} member(s), support >= {self.threshold:.0%}",
        ]
        if self.core:
            lines.append(f"core: {self.core}")
        lines.append(
            f"{'label':<15}{'family':<13}{'x':>8}{'y':>8}{'z':>8}{'tol':>7}{'support':>9}  fraction"
        )
        lines.append("-" * 15 + "-" * 13 + "-" * 8 * 3 + "-" * 7 + "-" * 9 + "  " + "-" * 8)
        for feature in self.features:
            lines.append(
                f"{feature.label:<15}{feature.family:<13}"
                f"{feature.x:>8.2f}{feature.y:>8.2f}{feature.z:>8.2f}"
                f"{feature.radius:>7.2f}{feature.support:>9}  {feature.fraction:>7.0%}"
            )
        if self.shape_mode == "envelope":
            lines.append(
                f"shape constraint: envelope, {len(self.envelope)} heavy-atom position(s) "
                f"from the members, {self.shape_tolerance:.1f} Å tolerance, "
                f"{self.min_coverage:.0%} of a candidate's atoms must be inside"
            )
        elif self.shape_mode == "spheres":
            lines.append(
                f"shape constraint: {len(self.excluded)} exclusion sphere(s) of radius "
                f"{self.excluded[0].radius:.2f} Å on the recurring scaffold"
                if self.excluded
                else "shape constraint: spheres requested but none recur "
                     "(the features already cover every recurring atom)"
            )
        else:
            lines.append("shape constraint: none (built with shape='none')")
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)

    # -- persistence -------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Write the model as JSON (no pickles: a model is data)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PharmacophoreModel":
        """Read a model written by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "features" not in data:
            raise ValueError(f"{path} is not a pharmacophore model file")
        features = []
        for row in data["features"]:
            x, y, z = (float(v) for v in row["position"])
            features.append(
                PointFeature(
                    family=str(row["family"]),
                    x=x,
                    y=y,
                    z=z,
                    radius=float(row.get("radius", DEFAULT_RADIUS)),
                    support=int(row.get("support", 0)),
                    fraction=float(row.get("fraction", 0.0)),
                    members=tuple(int(i) for i in row.get("members", ())),
                    label=str(row.get("label", "")),
                )
            )
        excluded = []
        for row in data.get("excluded", []):
            x, y, z = (float(v) for v in row["position"])
            excluded.append(
                ExclusionSphere(
                    x=x,
                    y=y,
                    z=z,
                    radius=float(row.get("radius", DEFAULT_EXCLUDED_RADIUS)),
                    support=int(row.get("support", 0)),
                    fraction=float(row.get("fraction", 0.0)),
                    members=tuple(int(i) for i in row.get("members", ())),
                    label=str(row.get("label", "")),
                )
            )
        return cls(
            features=features,
            excluded=excluded,
            envelope=[[float(v) for v in row] for row in data.get("envelope", [])],
            shape_mode=str(data.get("shape_mode", "envelope")),
            shape_tolerance=float(data.get("shape_tolerance", DEFAULT_SHAPE_TOLERANCE)),
            min_coverage=float(data.get("min_coverage", DEFAULT_MIN_COVERAGE)),
            n_members=int(data.get("n_members", 0)),
            n_used=int(data.get("n_used", 0)),
            threshold=float(data.get("threshold", DEFAULT_THRESHOLD)),
            radius=float(data.get("radius", DEFAULT_RADIUS)),
            core=str(data.get("core", "")),
            core_coords=[[float(v) for v in row] for row in data.get("core_coords", [])],
            core_elements=[str(v) for v in data.get("core_elements", [])],
            source=str(data.get("source", "")),
            notes=[str(n) for n in data.get("notes", [])],
        )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """The rigid transform that best superposes ``mobile`` onto ``target``.

    Returns ``(rotation, translation, rmsd)`` with ``x -> x @ rotation + translation``.
    Only proper rotations are allowed (the determinant is forced to ``+1``), so a
    molecule is never mirrored onto a model — which would fit a pharmacophore that
    cannot exist.
    """
    if mobile.shape != target.shape or mobile.shape[0] < 3:
        raise ValueError("Kabsch needs two matching point sets of at least three points")
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    left = mobile - mobile_centre
    right = target - target_centre
    covariance = left.T @ right
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt = vt.copy()
        vt[-1, :] *= -1
        rotation = u @ vt
    translation = target_centre - mobile_centre @ rotation
    moved = mobile @ rotation + translation
    rmsd = float(np.sqrt(((moved - target) ** 2).sum(axis=1).mean()))
    return rotation, translation, rmsd


def _apply(coords: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return coords @ rotation + translation


def _positions(mol, conf_id: int = 0) -> np.ndarray:
    conf = mol.GetConformer(int(conf_id))
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )


def _set_positions(mol, coords: np.ndarray, conf_id: int = 0) -> None:
    conf = mol.GetConformer(int(conf_id))
    for index in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(
            index, (float(coords[index, 0]), float(coords[index, 1]), float(coords[index, 2]))
        )


def _heavy_indices(mol) -> List[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def _mol_from_smiles(text: str):
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"{text!r} is not a parsable SMILES")
    return mol


def _ensure_conformer(mol, *, seed: int = 20240101, conformers: int = 1):
    """A copy of ``mol`` with a 3-D conformer (embedded when it has none)."""
    work = Chem.Mol(mol)
    if work.GetNumConformers() > 0:
        return work, False
    work = Chem.AddHs(work)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    count = max(1, int(conformers))
    if AllChem.EmbedMultipleConfs(work, numConfs=count, params=params) == 0:
        if AllChem.EmbedMolecule(work, randomSeed=int(seed), useRandomCoords=True) != 0:
            raise RuntimeError(
                "RDKit could not embed this molecule, so it cannot be part of a "
                "3D pharmacophore model"
            )
    return work, True


def _cluster_points(
    contributions: Sequence[Tuple[float, float, float], int],
    radius: float,
) -> List[Tuple[Tuple[float, float, float], List[int]]]:
    """Greedy clustering of points that carry a member index.

    Returns ``[(centroid, [member indices]), ...]``.  The algorithm is the same
    sphere-exclusion used by :func:`odock.ligandsim.butina_cluster` and for the
    same reason: it is deterministic, needs no geometry library, and reproducibility
    matters more here than optimality.  Ties are broken by the input order.
    """
    points = [np.asarray(p, dtype=float) for p, _ in contributions]
    owners = [int(owner) for _, owner in contributions]
    remaining = set(range(len(points)))
    clusters: List[Tuple[Tuple[float, float, float], List[int]]] = []
    while remaining:
        best_seed = -1
        best_count = -1
        for index in sorted(remaining):
            neighbours = [
                other
                for other in remaining
                if float(np.linalg.norm(points[index] - points[other])) <= float(radius)
            ]
            if len(neighbours) > best_count:
                best_count = len(neighbours)
                best_seed = index
        neighbours = [
            other
            for other in sorted(remaining)
            if float(np.linalg.norm(points[best_seed] - points[other])) <= float(radius)
        ]
        centroid = np.mean([points[other] for other in neighbours], axis=0)
        members = sorted({owners[other] for other in neighbours})
        clusters.append(((float(centroid[0]), float(centroid[1]), float(centroid[2])), members))
        remaining -= set(neighbours)
    return clusters


# ---------------------------------------------------------------------------
# Building a model
# ---------------------------------------------------------------------------


def _resolve_core(members: Sequence[Any], core: Optional[Union[str, Any]]) -> Tuple[Any, str]:
    """Return ``(core mol, core SMILES)`` for an alignment."""
    from .scaffold import maximum_common_substructure

    if core is None or (isinstance(core, str) and core.strip().lower() in ("mcs", "common")):
        text = maximum_common_substructure(list(members))
        if not text:
            raise ValueError(
                "the alignment frame needs a core, and no maximum common "
                "substructure could be found for these members; pass core='SMILES'"
            )
        return _mol_from_smiles(text), text
    if isinstance(core, str):
        mol = _mol_from_smiles(core.strip())
        return mol, Chem.MolToSmiles(mol)
    return core, Chem.MolToSmiles(core)


def build_model(
    members: Sequence[Any],
    *,
    frame: str = "shared",
    core: Optional[Union[str, Any]] = None,
    names: Optional[Sequence[str]] = None,
    radius: float = DEFAULT_RADIUS,
    threshold: float = DEFAULT_THRESHOLD,
    min_members: int = MIN_MEANINGFUL_MEMBERS,
    shape: str = DEFAULT_SHAPE,
    shape_tolerance: float = DEFAULT_SHAPE_TOLERANCE,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    excluded_radius: float = DEFAULT_EXCLUDED_RADIUS,
    seed: int = 20240101,
    source: str = "",
) -> PharmacophoreModel:
    """Derive the pharmacophore features that recur across ``members``.

    Parameters
    ----------
    members
        Molecules (or SMILES strings).  Each contributes one 3-D structure.
    frame
        ``"shared"`` (the default) means the members' coordinates are already
        comparable — docked poses, a crystallographic series, a set of structures
        exported in one frame.  ``"align"`` superposes every member onto the first
        one through ``core`` (an explicit SMILES, or the maximum common
        substructure of the set), which is what a series of SMILES from a file
        needs.  Members that do not contain the core are dropped with a note.
    core
        The superposition core for ``frame="align"``; ignored otherwise.  It is
        also recorded in the model as the reference a screening molecule is
        superposed onto.
    names
        Per-member names, for the notes and the labels.
    radius
        Two features of the same family from different members within this
        distance are treated as one model feature (Å).
    threshold
        Minimum fraction of members that must support a feature for it to enter
        the model.
    min_members
        Below this many *used* members the model is annotated as an anecdote (see
        :data:`MIN_MEANINGFUL_MEMBERS`); it is still built, because inspecting one
        molecule's features is a legitimate thing to want.
    shape
        ``"envelope"`` (default) records every member's heavy-atom positions and
        requires a candidate to sit inside them (``shape_tolerance``,
        ``min_coverage``).  ``"spheres"`` builds exclusion spheres from the
        recurring heavy atoms that carry no feature — which is empty when the
        features already cover the recurring scaffold, and then the model says so.
        ``"none"`` builds a pure feature model.
    excluded_radius
        Radius of the exclusion spheres in ``"spheres"`` mode (Å).

    Returns
    -------
    :class:`PharmacophoreModel`.  Every feature reports how many distinct members
    support it, and the notes say what was dropped and why.
    """
    require_rdkit()
    if str(frame).strip().lower() not in ("shared", "align"):
        raise ValueError(f"unknown frame {frame!r}; use 'shared' or 'align'")
    if str(shape).strip().lower() not in ("envelope", "spheres", "none"):
        raise ValueError(f"unknown shape mode {shape!r}; use envelope, spheres or none")
    if not 0.0 < float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    if float(radius) <= 0:
        raise ValueError(f"radius must be positive, got {radius!r}")
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError(f"min_coverage must be in [0, 1], got {min_coverage!r}")
    shape_mode = str(shape).strip().lower()
    raw = list(members)
    if not raw:
        raise ValueError("no member to build a model from")

    notes: List[str] = []
    labels = [
        str(names[i]) if names is not None and i < len(names) else f"member_{i + 1}"
        for i in range(len(raw))
    ]
    prepared: List[Any] = []
    skipped: List[str] = []
    for index, item in enumerate(raw):
        mol = _mol_from_smiles(item) if isinstance(item, str) else item
        try:
            work, _ = _ensure_conformer(mol, seed=seed)
        except Exception as exc:
            skipped.append(f"{labels[index]}: {exc}")
            continue
        prepared.append(work)
    if not prepared:
        raise ValueError("no member could be prepared in 3-D: " + "; ".join(skipped[:4]))

    core_mol = None
    core_text = ""
    core_coords: List[List[float]] = []
    used: List[Any] = []
    used_labels: List[str] = []
    used_original: List[int] = []
    align_frame = str(frame).strip().lower() == "align"
    if align_frame:
        core_mol, core_text = _resolve_core(prepared, core)
        reference_coords = _positions(prepared[0])
        match0 = prepared[0].GetSubstructMatch(core_mol)
        if not match0:
            raise ValueError(
                f"the first member does not contain the core {core_text!r}; choose a "
                "core that all members share"
            )
        for index, mol in enumerate(prepared):
            found = _best_core_match(mol, core_mol, reference_coords[list(match0)])
            if found is None:
                skipped.append(f"{labels[index]}: does not contain the core {core_text}")
                continue
            match, _ = found
            mobile = _positions(mol)[list(match)]
            target = reference_coords[list(match0)]
            rotation, translation, _ = _kabsch(mobile, target)
            conf = mol.GetConformer()
            moved = _apply(_positions(mol), rotation, translation)
            for atom_index in range(mol.GetNumAtoms()):
                position = moved[atom_index]
                conf.SetAtomPosition(
                    atom_index, (float(position[0]), float(position[1]), float(position[2]))
                )
            used.append(mol)
            used_labels.append(labels[index])
            used_original.append(index)
        if not used:
            raise ValueError("no member contains the core, so nothing could be aligned")
        # The reference core geometry, in the core's own atom order.
        core_positions = _positions(used[0])[list(used[0].GetSubstructMatch(core_mol))]
        core_coords = [[float(v) for v in row] for row in core_positions]
        notes.append(
            f"members were superposed on the core {core_text} "
            f"({core_mol.GetNumAtoms()} atoms, MCS or explicit)"
        )
    else:
        used = prepared
        used_labels = labels
        used_original = list(range(len(prepared)))
        if core is not None:
            core_mol, core_text = _resolve_core(used, core)
            match0 = used[0].GetSubstructMatch(core_mol)
            if match0:
                core_coords = [
                    [float(v) for v in row] for row in _positions(used[0])[list(match0)]
                ]
            else:
                notes.append(
                    f"the recorded core {core_text!r} does not match the first member; "
                    "it is stored without reference coordinates"
                )
    if skipped:
        notes.append(
            f"{len(skipped)} member(s) were dropped: " + "; ".join(skipped[:4])
            + (" ..." if len(skipped) > 4 else "")
        )
    n_members = len(raw)
    n_used = len(used)

    # -- recurring features ------------------------------------------------
    contributions: Dict[str, List[Tuple[Tuple[float, float, float], int]]] = {}
    for slot, mol in enumerate(used):
        for family, position, _ in features_from_mol(mol, conf_id=0):
            contributions.setdefault(family, []).append((position, slot))
    features: List[PointFeature] = []
    counters: Dict[str, int] = {}
    for family in FEATURE_FAMILIES:
        points = contributions.get(family)
        if not points:
            continue
        for centroid, members in _cluster_points(points, radius):
            fraction = len(members) / n_used if n_used else 0.0
            if fraction + 1e-9 < float(threshold):
                continue
            counters[family] = counters.get(family, 0) + 1
            features.append(
                PointFeature(
                    family=family,
                    x=centroid[0],
                    y=centroid[1],
                    z=centroid[2],
                    radius=float(radius),
                    support=len(members),
                    fraction=fraction,
                    members=tuple(used_original[slot] for slot in members),
                    label=f"{family}#{counters[family]}",
                )
            )
    features.sort(key=lambda f: (-f.fraction, -f.support, f.family, f.label))

    # -- the shape constraint ---------------------------------------------
    envelope: List[List[float]] = []
    excluded: List[ExclusionSphere] = []
    if shape_mode == "envelope":
        for mol in used:
            coords = _positions(mol)
            for atom_index in _heavy_indices(mol):
                position = coords[atom_index]
                envelope.append([float(position[0]), float(position[1]), float(position[2])])
    elif shape_mode == "spheres":
        feature_points = [np.asarray(feature.position) for feature in features]
        shape_points: List[Tuple[Tuple[float, float, float], int]] = []
        for slot, mol in enumerate(used):
            coords = _positions(mol)
            for atom_index in _heavy_indices(mol):
                position = coords[atom_index]
                if feature_points and min(
                    float(np.linalg.norm(position - point)) for point in feature_points
                ) <= float(radius):
                    continue
                shape_points.append(
                    ((float(position[0]), float(position[1]), float(position[2])), slot)
                )
        for order, (centroid, members) in enumerate(
            _cluster_points(shape_points, _EXCLUDED_CLUSTER_RADIUS), start=1
        ):
            fraction = len(members) / n_used if n_used else 0.0
            if fraction + 1e-9 < float(threshold):
                continue
            excluded.append(
                ExclusionSphere(
                    x=centroid[0],
                    y=centroid[1],
                    z=centroid[2],
                    radius=float(excluded_radius),
                    support=len(members),
                    fraction=fraction,
                    members=tuple(used_original[slot] for slot in members),
                    label=f"X{order}",
                )
            )
        if not excluded:
            notes.append(
                "the sphere shape constraint came out empty: every recurring heavy "
                "atom of these members is already part of a model feature, so there "
                "is no scaffold atom left to exclude. Use shape='envelope' for a "
                "constraint that bites on a small rigid series."
            )

    if n_used < int(min_members):
        notes.append(
            f"only {n_used} member(s) contributed, which is fewer than the "
            f"{int(min_members)} this project treats as evidence: a feature here can "
            "be one member's decoration rather than a recurring requirement, so read "
            "the support column and not the fit score"
        )
    if not features:
        notes.append(
            f"no feature reaches {float(threshold):.0%} support at a {float(radius):.1f} Å "
            "tolerance; lower the threshold or widen the radius"
        )
    return PharmacophoreModel(
        features=features,
        excluded=excluded,
        envelope=envelope,
        shape_mode=shape_mode,
        shape_tolerance=float(shape_tolerance),
        min_coverage=float(min_coverage),
        n_members=n_members,
        n_used=n_used,
        threshold=float(threshold),
        radius=float(radius),
        core=core_text,
        core_coords=core_coords,
        core_elements=[atom.GetSymbol() for atom in core_mol.GetAtoms()] if core_mol is not None else [],
        source=str(source),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Fitting and screening
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """How well one library molecule fits the model."""

    index: int
    name: str
    #: ``(sum of matched weights - miss_penalty * missed) / n_features``, in [0, 1].
    fit: float = 0.0
    #: Fraction of the model's features that were matched at all.
    coverage: float = 0.0
    n_matched: int = 0
    n_features: int = 0
    #: ``(label, distance, weight)`` for every matched model feature.
    matched: List[Tuple[str, float, float]] = field(default_factory=list)
    #: Labels of the model features that were missed.
    missed: List[str] = field(default_factory=list)
    #: Which conformer achieved the score, and how many were tried.
    conformer: int = -1
    n_conformers: int = 0
    #: ``"core"``, ``"features"``, ``"none"`` or ``""`` (not aligned).
    alignment: str = ""
    #: Library heavy atoms inside an excluded-volume sphere.
    clashes: int = 0
    #: Whether the shape constraint was satisfied.
    shape_ok: bool = True
    #: True when the shape constraint or the alignment rejected the molecule.
    rejected: bool = False
    reason: str = ""
    smiles: str = ""

    @property
    def n_missed(self) -> int:
        return len(self.missed)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "name": self.name,
            "fit": round(float(self.fit), 4),
            "coverage": round(float(self.coverage), 4),
            "n_matched": int(self.n_matched),
            "n_missed": int(self.n_missed),
            "n_features": int(self.n_features),
            "matched": [
                {"label": label, "distance": round(float(d), 4), "weight": round(float(w), 4)}
                for label, d, w in self.matched
            ],
            "missed": list(self.missed),
            "conformer": int(self.conformer),
            "n_conformers": int(self.n_conformers),
            "alignment": self.alignment,
            "clashes": int(self.clashes),
            "shape_ok": bool(self.shape_ok),
            "rejected": bool(self.rejected),
            "reason": self.reason,
            "smiles": self.smiles,
        }


@dataclass
class PharmacophoreHits:
    """A screened library, ranked by fit, with what the screen cost."""

    model: Optional[PharmacophoreModel] = None
    hits: List[FitResult] = field(default_factory=list)
    n_library: int = 0
    tolerance: float = DEFAULT_TOLERANCE
    miss_penalty: float = DEFAULT_MISS_PENALTY
    enforce_shape: bool = True
    seconds: float = 0.0
    n_alignments: int = 0
    n_conformers: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    @property
    def n_rejected(self) -> int:
        return sum(1 for hit in self.hits if hit.rejected)

    @property
    def best(self) -> Optional[FitResult]:
        return self.hits[0] if self.hits else None

    def scores(self) -> List[Tuple[str, float]]:
        return [(hit.name, float(hit.fit)) for hit in self.hits]

    def table(self, limit: int = 0) -> str:
        rows = self.hits if limit <= 0 else self.hits[: int(limit)]
        if not rows:
            return "no molecule was scored against the model"
        lines = [
            f"{'rank':<5}{'name':<30}{'fit':>7}{'matched':>9}{'missed':>7}{'shape':>7}  notes",
            "-" * 5 + "-" * 30 + "-" * 7 + "-" * 9 + "-" * 7 + "-" * 7 + "  " + "-" * 30,
        ]
        for rank, hit in enumerate(rows, start=1):
            note = hit.reason or hit.alignment
            if not hit.alignment and hit.rejected:
                shape = "n/a"
            else:
                shape = "ok" if hit.shape_ok else "clash"
            lines.append(
                f"{rank:<5}{hit.name[:29]:<30}{hit.fit:>7.3f}"
                f"{hit.n_matched:>4}/{hit.n_features:<4}{hit.n_missed:>7}"
                f"{shape:>7}  {note[:30]}"
            )
        if limit > 0 and len(self.hits) > limit:
            lines.append(f"... and {len(self.hits) - limit} more molecule(s)")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.as_dict() if self.model is not None else None,
            "n_library": int(self.n_library),
            "n_scored": int(self.n_hits),
            "n_rejected": int(self.n_rejected),
            "tolerance": round(float(self.tolerance), 4),
            "miss_penalty": round(float(self.miss_penalty), 4),
            "enforce_shape": bool(self.enforce_shape),
            "seconds": round(float(self.seconds), 4),
            "n_alignments": int(self.n_alignments),
            "n_conformers": int(self.n_conformers),
            "hits": [hit.as_dict() for hit in self.hits],
            "notes": list(self.notes),
        }


def _feature_points(mol, conf_id: int = 0) -> List[Tuple[str, np.ndarray]]:
    return [
        (family, np.asarray(position, dtype=float))
        for family, position, _ in features_from_mol(mol, conf_id=conf_id)
    ]


def _best_core_match(mol, core_mol, target: np.ndarray, conf_id: int = 0):
    """The core match with the lowest RMSD to ``target``, over *all* matches.

    A symmetric core has several substructure matches, and taking the first one
    can superpose a molecule onto a mirror-image or ring-shifted assignment of the
    core, which leaves the rest of the molecule visibly wrong even though every
    matched atom is "a" valid match.  Enumerating them and keeping the best fit is
    the difference between a model that scores a para analogue correctly and one
    that scores it zero because its substituent ended up on the wrong side.
    Returns ``(match, rmsd)`` or ``None``.
    """
    coords = _positions(mol, conf_id)
    best = None
    for match in mol.GetSubstructMatches(core_mol, uniquify=False):
        if len(match) != len(target):
            continue
        mobile = coords[list(match)]
        try:
            _, _, rmsd = _kabsch(mobile, target)
        except ValueError:  # pragma: no cover - fewer than three core atoms
            return None
        if best is None or rmsd < best[1]:
            best = (match, rmsd)
    return best


def _align_on_core(mol, model: PharmacophoreModel, conf_id: int = 0):
    """Superpose ``mol`` onto the model's core; returns the moved coordinates."""
    if not model.core or not model.core_coords:
        return None
    core_mol = Chem.MolFromSmiles(model.core)
    if core_mol is None:  # pragma: no cover - defensive
        return None
    target = np.asarray(model.core_coords, dtype=float)
    found = _best_core_match(mol, core_mol, target, conf_id=conf_id)
    if found is None:
        return None
    match, _ = found
    coords = _positions(mol, conf_id)
    mobile = coords[list(match)]
    try:
        rotation, translation, _ = _kabsch(mobile, target)
    except ValueError:  # pragma: no cover - fewer than three core atoms
        return None
    return _apply(coords, rotation, translation)


def _align_on_features(
    mol,
    model: PharmacophoreModel,
    conf_id: int = 0,
) -> Optional[Tuple[np.ndarray, str]]:
    """Superpose ``mol`` onto the model by matching feature triples.

    Enumerates every ordered triple of model features with three *different*
    families and every matching ordered triple of the molecule's features, and
    keeps the superposition with the lowest RMSD.  That is a rigid fit of the
    conformer as it is — no flexible fitting — and a molecule with no matching
    triple cannot be aligned at all, which the caller reports as a rejection
    rather than as a low score.
    """
    coords = _positions(mol, conf_id)
    mine = _feature_points(mol, conf_id)
    if len(mine) < 3 or model.n_features < 3:
        return None
    best: Optional[Tuple[float, np.ndarray, str]] = None
    for i in range(model.n_features):
        for j in range(i + 1, model.n_features):
            for k in range(j + 1, model.n_features):
                triple = (model.features[i], model.features[j], model.features[k])
                if len({feature.family for feature in triple}) < 3:
                    continue
                target = np.asarray([feature.position for feature in triple], dtype=float)
                for a in range(len(mine)):
                    if mine[a][0] != triple[0].family:
                        continue
                    for b in range(len(mine)):
                        if b == a or mine[b][0] != triple[1].family:
                            continue
                        for c in range(len(mine)):
                            if c in (a, b) or mine[c][0] != triple[2].family:
                                continue
                            mobile = np.asarray([mine[a][1], mine[b][1], mine[c][1]])
                            try:
                                rotation, translation, rmsd = _kabsch(mobile, target)
                            except ValueError:  # pragma: no cover - defensive
                                continue
                            if best is None or rmsd < best[0]:
                                best = (rmsd, rotation, translation)  # type: ignore[assignment]
    if best is None:
        return None
    _, rotation, translation = best
    return _apply(coords, rotation, translation), "features"


def _score_coordinates(
    coords: np.ndarray,
    mol,
    model: PharmacophoreModel,
    *,
    tolerance: float,
    miss_penalty: float,
    enforce_shape: bool,
) -> Tuple[float, float, List[Tuple[str, float, float]], List[str], int, bool]:
    """Score already-placed coordinates against the model.

    ``mol``'s conformer must already hold ``coords``: a feature position is an
    atom position or the centroid of a fixed set of atoms, so re-perceiving the
    features from the placed conformer gives exactly the placed geometry.
    """
    moved_features = _feature_points(mol)
    used = [False] * len(moved_features)
    matched: List[Tuple[str, float, float]] = []
    missed: List[str] = []
    total = 0.0
    for feature in model.features:
        best_index = -1
        best_distance = math.inf
        for index, (family, point) in enumerate(moved_features):
            if used[index] or family != feature.family:
                continue
            distance = float(np.linalg.norm(point - np.asarray(feature.position)))
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index < 0 or best_distance > float(tolerance):
            missed.append(feature.label)
            continue
        used[best_index] = True
        weight = 1.0 - (best_distance / float(tolerance))
        weight = max(0.0, weight)
        matched.append((feature.label, best_distance, weight))
        total += weight
    n_features = model.n_features
    if n_features == 0:
        return 0.0, 0.0, matched, missed, 0, True
    fit = (total - float(miss_penalty) * len(missed)) / n_features
    fit = min(1.0, max(0.0, fit))
    coverage = len(matched) / n_features
    clashes = 0
    shape_ok = True
    if enforce_shape and model.shape_mode == "spheres" and model.excluded:
        heavy = _heavy_indices(mol)
        for sphere in model.excluded:
            centre = np.asarray(sphere.position)
            for atom_index in heavy:
                if float(np.linalg.norm(coords[atom_index] - centre)) < float(sphere.radius):
                    clashes += 1
                    break
        shape_ok = clashes == 0
    elif enforce_shape and model.shape_mode == "envelope" and model.envelope:
        envelope = np.asarray(model.envelope, dtype=float)
        heavy = _heavy_indices(mol)
        for atom_index in heavy:
            distances = np.linalg.norm(envelope - coords[atom_index], axis=1)
            if float(distances.min()) > float(model.shape_tolerance):
                clashes += 1
        inside = len(heavy) - clashes
        shape_ok = bool(heavy) and (inside / len(heavy)) >= float(model.min_coverage)
    return fit, coverage, matched, missed, clashes, shape_ok


def fit_score(
    mol,
    model: PharmacophoreModel,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    miss_penalty: float = DEFAULT_MISS_PENALTY,
    conformers: int = 1,
    seed: int = 20240101,
    enforce_shape: bool = True,
    align: Optional[str] = None,
    index: int = 0,
    name: str = "",
) -> FitResult:
    """Fit one molecule to ``model``; the best conformer wins.

    ``align`` is ``None`` (try the model's core, then the feature triples, both
    rigid), ``"core"``, ``"features"`` or ``"none"`` (the coordinates are already
    in the model's frame).  The score is

    ``fit = (sum of matched weights - miss_penalty * n_missed) / n_features``

    where a matched feature's weight ramps linearly from 0 at ``tolerance`` to 1
    at zero distance, so the score is in ``[0, 1]`` and a missed feature costs a
    full feature's worth at the default ``miss_penalty``.  A molecule that cannot
    be aligned, or that puts a heavy atom inside an excluded-volume sphere while
    ``enforce_shape`` is set, is reported with ``rejected`` and a reason instead
    of a misleadingly high score.
    """
    require_rdkit()
    if not model.features:
        raise ValueError("the model has no features, so nothing can be fitted to it")
    if float(tolerance) <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance!r}")
    label = name or _mol_name(mol)
    work, _ = _ensure_conformer(mol, seed=seed, conformers=max(1, int(conformers)))
    best: Optional[FitResult] = None
    tried = work.GetNumConformers()
    for conf_id in range(tried):
        placed = None
        method = "none"
        wanted = str(align).strip().lower() if align is not None else "auto"
        if wanted in ("auto", "core"):
            placed = _align_on_core(work, model, conf_id=conf_id)
            if placed is not None:
                method = "core"
        if placed is None and wanted in ("auto", "features"):
            found = _align_on_features(work, model, conf_id=conf_id)
            if found is not None:
                placed, method = found
        if placed is None and wanted != "none":
            result = FitResult(
                index=index,
                name=label,
                n_features=model.n_features,
                conformer=conf_id,
                n_conformers=tried,
                rejected=True,
                reason=(
                    "no rigid alignment onto the model: the molecule shares neither "
                    "the core nor a triple of the model's feature families"
                ),
                smiles=_safe_smiles(mol),
            )
            if best is None or (result.fit, -result.clashes) > (best.fit, -best.clashes):
                best = result
            continue
        coords = placed if placed is not None else _positions(work, conf_id)
        if placed is not None:
            # The placed geometry has to be the scoring geometry, so the moved
            # coordinates go back into the conformer before the features are
            # perceived from it.
            _set_positions(work, coords, conf_id)
        fit, coverage, matched, missed, clashes, shape_ok = _score_coordinates(
            coords,
            work,
            model,
            tolerance=float(tolerance),
            miss_penalty=float(miss_penalty),
            enforce_shape=bool(enforce_shape),
        )
        reason = ""
        rejected = False
        if not shape_ok:
            rejected = True
            if model.shape_mode == "envelope":
                reason = (
                    f"{clashes} of {len(_heavy_indices(work))} heavy atom(s) fall more "
                    f"than {model.shape_tolerance:.1f} Å outside the members' envelope "
                    f"(< {model.min_coverage:.0%} inside)"
                )
            else:
                reason = (
                    f"{clashes} atom(s) inside the model's excluded volume (the shape of "
                    "the recurring scaffold)"
                )
        result = FitResult(
            index=index,
            name=label,
            fit=0.0 if rejected else fit,
            coverage=coverage,
            n_matched=len(matched),
            n_features=model.n_features,
            matched=matched,
            missed=missed,
            conformer=conf_id,
            n_conformers=tried,
            alignment=method,
            clashes=clashes,
            shape_ok=shape_ok,
            rejected=rejected,
            reason=reason,
            smiles=_safe_smiles(mol),
        )
        if best is None or result.fit > best.fit:
            best = result
    assert best is not None  # `tried >= 1` and the loop always assigns
    return best


def _mol_name(mol, fallback: str = "") -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def _safe_smiles(mol) -> str:
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - defensive
        return ""


def screen(
    model: PharmacophoreModel,
    molecules: Sequence[Any],
    *,
    names: Optional[Sequence[str]] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    miss_penalty: float = DEFAULT_MISS_PENALTY,
    conformers: int = 8,
    seed: int = 20240101,
    top: int = 0,
    enforce_shape: bool = True,
    align: Optional[str] = None,
    source: str = "",
) -> PharmacophoreHits:
    """Score a library against ``model`` and rank it by fit.

    Each molecule is embedded (``conformers`` conformers, ``seed``) unless it
    already carries 3-D coordinates, aligned to the model and scored by the best
    conformer.  The report carries what the screen cost (``seconds``,
    ``n_alignments``, ``n_conformers``) and how many molecules the shape
    constraint rejected, because "the model rejects 11 of 17" is a result about
    the model and not a detail.

    The scores are geometric fits, not activities: a molecule that ranks first
    presents the features in the right places, and whether that means it binds is
    a question only an assay answers.
    """
    require_rdkit()
    start = time.perf_counter()
    out: List[FitResult] = []
    alignments = 0
    total_conformers = 0
    prepared: List[Any] = []
    labels: List[str] = []
    notes: List[str] = []
    for index, item in enumerate(molecules):
        mol = _mol_from_smiles(item) if isinstance(item, str) else item
        label = (
            str(names[index])
            if names is not None and index < len(names)
            else _mol_name(mol, f"ligand_{index + 1}")
        )
        labels.append(label)
        prepared.append(mol)
    for index, mol in enumerate(prepared):
        result = fit_score(
            mol,
            model,
            tolerance=tolerance,
            miss_penalty=miss_penalty,
            conformers=conformers,
            seed=seed,
            enforce_shape=enforce_shape,
            align=align,
            index=index,
            name=labels[index],
        )
        alignments += result.n_conformers if result.alignment else 0
        total_conformers += result.n_conformers
        out.append(result)
    out.sort(key=lambda hit: (-hit.fit, hit.n_missed, hit.index))
    if top and int(top) > 0:
        out = out[: int(top)]
    if model.n_used < MIN_MEANINGFUL_MEMBERS:
        notes.append(
            f"the model was built from {model.n_used} member(s); a fit score against it "
            "is a geometric statement about those members' features, not evidence "
            "about activity"
        )
    return PharmacophoreHits(
        model=model,
        hits=out,
        n_library=len(prepared),
        tolerance=float(tolerance),
        miss_penalty=float(miss_penalty),
        enforce_shape=bool(enforce_shape),
        seconds=time.perf_counter() - start,
        n_alignments=alignments,
        n_conformers=total_conformers,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def enrichment(
    scored: Union[PharmacophoreHits, Sequence[Tuple[str, float]]],
    actives: Iterable[str],
    *,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """The labelled-set numbers for a ranked screen, with their sample size.

    ``k`` defaults to the number of actives (the usual "how many molecules would
    I have to test to find them all" question).  Returns ``n_actives``,
    ``n_total``, ``k``, ``precision_at_k``, ``recall_at_k``, ``enrichment_factor``
    (the observed active rate in the top ``k`` over the library's base rate) and
    ``auc`` (the Mann-Whitney form, computed on ranks).

    **What these numbers can and cannot say.**  An enrichment factor is a ratio of
    small counts: with six actives in a seventeen-molecule library one molecule
    moves it by tens of percent, and a "3x enrichment" from a set that small is a
    smoke test that the model is not random — not evidence that it will enrich a
    real library.  The result therefore carries the sample size explicitly and a
    note saying so; quote the numbers with ``n_actives``/``n_total`` beside them
    or not at all.
    """
    if isinstance(scored, PharmacophoreHits):
        ranked = scored.scores()
    else:
        ranked = [(str(name), float(value)) for name, value in scored]
    active_set = {str(name) for name in actives}
    n_total = len(ranked)
    n_actives = sum(1 for name, _ in ranked if name in active_set)
    top_k = int(k) if k else max(1, n_actives)
    top_k = max(1, min(top_k, n_total)) if n_total else 0
    hits_at_k = [name for name, _ in ranked[:top_k] if name in active_set]
    precision = len(hits_at_k) / top_k if top_k else 0.0
    recall = len(hits_at_k) / n_actives if n_actives else 0.0
    base_rate = n_actives / n_total if n_total else 0.0
    factor = (precision / base_rate) if base_rate else 0.0
    # Mann-Whitney AUC over the ranked list: 1.0 when every active outranks every
    # decoy, 0.5 for a random ranking.
    positives = [value for name, value in ranked if name in active_set]
    negatives = [value for name, value in ranked if name not in active_set]
    if positives and negatives:
        wins = sum(
            1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
        )
        auc = wins / (len(positives) * len(negatives))
    else:
        auc = float("nan")
    notes = []
    if n_actives < 5 or n_total < 30:
        notes.append(
            f"{n_actives} active(s) in {n_total} molecule(s) is far too small a "
            "labelled set for an enrichment claim: the factor is quantised (one "
            "molecule moves it by tens of percent) and the confidence interval "
            "spans any conclusion. Report it as a smoke test only."
        )
    return {
        "n_total": int(n_total),
        "n_actives": int(n_actives),
        "k": int(top_k),
        "n_actives_in_top_k": len(hits_at_k),
        "actives_in_top_k": list(hits_at_k),
        "precision_at_k": round(precision, 4),
        "recall_at_k": round(recall, 4),
        "base_rate": round(base_rate, 4),
        "enrichment_factor": round(factor, 4),
        "auc": None if math.isnan(auc) else round(auc, 4),
        "notes": notes,
    }

# SPDX-License-Identifier: GPL-3.0-or-later
"""A ligand-based virtual screening benchmark, with the statistics that mean something.

An enrichment factor on six actives is a smoke test.  This module is what turns
that into a claim a reader can check:

* **Metrics** — EF1 %, EF5 %, AUC and BEDROC(α = 20) (:func:`metrics`, each with
  its formula written down), each with a **bootstrap confidence interval** over
  molecules (:func:`bootstrap_ci`), because at these set sizes the interval is the
  honest part of the number;
* **Methods** — a 2-D fingerprint screen (:func:`fingerprint_scores`), a 3-D
  pharmacophore fit (:func:`pharmacophore_scores`) and a rigid
  shape/electrostatic volume overlay (:func:`shape_scores`);
* **Controls** — a random ranking and a property-only ranking
  (:func:`random_scores`, :func:`property_scores`), so "no signal" and "trivial
  signal" can be read beside the methods;
* **A pre-filter analysis** (:func:`prefilter`) — how much of the active set
  survives a fingerprint pre-filter at a stated fraction of the library, and what
  that does to the docking workload, using :mod:`odock.screen`'s own cost model
  (imported, never edited);
* :func:`benchmark` — the whole table: every method and both controls against the
  same actives and decoys.

What the numbers do and do not mean: they measure **how well a scoring function
ranks the actives above the decoys in this library**.  They are not a statement
about activity, they do not transfer to another target, and a decoy is not an
inactive molecule — it is one nobody has reported binding that target.
`docs/LBVS.md` states all of that next to the measured tables.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from .sasa import radius_of

__all__ = [
    "EF_FRACTIONS",
    "BEDROC_ALPHA",
    "BOOTSTRAP_SAMPLES",
    "Ranking",
    "Metrics",
    "MethodResult",
    "BenchmarkReport",
    "require_rdkit",
    "ef_at",
    "auc",
    "bedroc",
    "metrics",
    "bootstrap_ci",
    "fingerprint_scores",
    "pharmacophore_scores",
    "shape_scores",
    "random_scores",
    "property_scores",
    "prefilter",
    "benchmark",
]

#: Early-recognition fractions for EF1 % and EF5 %.
EF_FRACTIONS: Tuple[float, ...] = (0.01, 0.05)

#: BEDROC's weighting: α = 20 puts 80 % of the score in the top 8 % of the list
#: (the usual early-recognition setting).
BEDROC_ALPHA = 20.0

#: Bootstrap resamples.  1000 gives a stable percentile interval at the cost of
#: 1000 metric evaluations, which is nothing next to scoring a library.
BOOTSTRAP_SAMPLES = 1000


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for the ligand-based benchmark. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Rankings and metrics
# ---------------------------------------------------------------------------


@dataclass
class Ranking:
    """A scored library: one score per molecule, ranked best-first."""

    method: str
    #: ``(name, score)`` sorted by decreasing score; ties keep library order.
    entries: List[Tuple[str, float]] = field(default_factory=list)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> List[str]:
        return [name for name, _ in self.entries]

    @property
    def scores(self) -> List[float]:
        return [score for _, score in self.entries]

    def rank_of(self, name: str) -> Optional[int]:
        """The 1-based rank of ``name``, or ``None`` when it is not in the ranking."""
        for position, (entry, _) in enumerate(self.entries, start=1):
            if entry == name:
                return position
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n": len(self.entries),
            "seconds": round(float(self.seconds), 4),
            "entries": [[name, round(float(score), 6)] for name, score in self.entries],
            "notes": list(self.notes),
        }


def _check_inputs(scores: Sequence[float], labels: Sequence[bool]) -> None:
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have the same length, got {len(scores)} and {len(labels)}"
        )
    if not scores:
        raise ValueError("no molecule to evaluate")


def _ranked(scores: Sequence[float]) -> List[int]:
    """Indices sorted by decreasing score (ties by index, so it is deterministic)."""
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def ef_at(scores: Sequence[float], labels: Sequence[bool], fraction: float) -> Dict[str, float]:
    """Enrichment factor at the top ``fraction`` of the library.

    ``k = max(1, ceil(fraction * N))`` molecules are taken, and

    ``EF = (actives in the top k / k) / (total actives / N)``

    so EF = 1 is a random ranking.  With a small library ``k`` rounds up to 1 or 2,
    which is why the enrichment factor alone is not a result: 1 % of 30 molecules is
    one molecule, and that molecule is either an active or it is not.
    """
    _check_inputs(scores, labels)
    n = len(scores)
    n_actives = sum(1 for flag in labels if flag)
    k = max(1, int(math.ceil(float(fraction) * n)))
    order = _ranked(scores)[:k]
    hits = sum(1 for index in order if labels[index])
    base_rate = n_actives / n if n else 0.0
    observed = hits / k
    return {
        "fraction": float(fraction),
        "k": k,
        "actives_in_top_k": hits,
        "precision": observed,
        "recall": (hits / n_actives) if n_actives else 0.0,
        "base_rate": base_rate,
        "ef": (observed / base_rate) if base_rate else 0.0,
    }


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """ROC AUC by the Mann-Whitney statistic, ties counting half.

    ``0.5`` is a random ranking, ``1.0`` a perfect one.  It averages over the whole
    list, so it says nothing about early recognition — which is what BEDROC is for.
    """
    _check_inputs(scores, labels)
    positives = [float(score) for score, flag in zip(scores, labels) if flag]
    negatives = [float(score) for score, flag in zip(scores, labels) if not flag]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def bedroc(
    scores: Sequence[float], labels: Sequence[bool], *, alpha: float = BEDROC_ALPHA
) -> float:
    """Early-recognition score with BEDROC's exponential weight (α = 20).

    Truchon and Bayly's BEDROC weights each rank ``r`` of ``N`` by
    ``exp(-α r / N)`` so that α = 20 puts ~80 % of the weight on the top 8 % of the
    list.  This implementation normalises that weight explicitly and states the
    properties it guarantees, which the published constant does not make easy to
    verify:

    * the actives' share of the exponential weight is
      ``f = Σ_actives w(r_i)`` with ``Σ_r w(r) = 1`` (mid-rank weights);
    * a random ranking gives ``f = n_actives / N`` in expectation, a perfect one
      gives ``f_max = Σ_{r=1..n_actives} w(r)``;
    * the score is ``0.5 + 0.5 · (f − f_random) / (f_max − f_random)``, clamped to
      ``[0, 1]``.

    So **a perfect ranking scores 1.0 and a random ranking 0.5 in expectation**,
    which is exactly the calibration BEDROC advertises, and the tests pin both.  It
    is a *BEDROC-style* statistic, not a bit-for-bit copy of a specific
    implementation: read it beside EF1 % (what is in the very top) and AUC (the
    whole list), never on its own.
    """
    _check_inputs(scores, labels)
    n = len(scores)
    n_actives = sum(1 for flag in labels if flag)
    if n_actives == 0 or n_actives == n:
        return float("nan")
    weights = np.exp(-float(alpha) * (np.arange(1, n + 1) - 0.5) / n)
    weights = weights / weights.sum()
    order = _ranked(scores)
    observed = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            observed += float(weights[rank - 1])
    random_share = n_actives / n
    perfect_share = float(weights[:n_actives].sum())
    if perfect_share - random_share <= 0:  # pragma: no cover - alpha > 0
        return float("nan")
    value = 0.5 + 0.5 * (observed - random_share) / (perfect_share - random_share)
    return min(1.0, max(0.0, value))


def metrics(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    fractions: Sequence[float] = EF_FRACTIONS,
    alpha: float = BEDROC_ALPHA,
) -> Dict[str, Any]:
    """EF at each fraction, AUC and BEDROC, with the counts they came from."""
    _check_inputs(scores, labels)
    out: Dict[str, Any] = {
        "n": len(scores),
        "n_actives": int(sum(1 for flag in labels if flag)),
        "auc": round(auc(scores, labels), 4),
        "bedroc": round(bedroc(scores, labels, alpha=alpha), 4),
        "alpha": float(alpha),
    }
    for fraction in fractions:
        key = f"ef{int(round(float(fraction) * 100))}"
        out[key] = round(ef_at(scores, labels, fraction)["ef"], 4)
        out[f"{key}_k"] = ef_at(scores, labels, fraction)["k"]
    return out


def bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[bool],
    statistic: Callable[[Sequence[float], Sequence[bool]], float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    level: float = 0.95,
    seed: int = 20240101,
) -> Tuple[float, float]:
    """A percentile bootstrap interval for ``statistic`` over molecules.

    Molecules are resampled with replacement (scores and labels together), the
    statistic recomputed, and the ``level`` interval read off the sorted results.
    This is the interval that matters at small set sizes: with six actives in thirty
    molecules the EF1 % interval spans everything from "no actives in the top one"
    to "the top one is an active", and a benchmark that printed the point estimate
    alone would be hiding that.
    """
    _check_inputs(scores, labels)
    rng = random.Random(int(seed))
    n = len(scores)
    values: List[float] = []
    for _ in range(max(1, int(samples))):
        picks = [rng.randrange(n) for _ in range(n)]
        resampled_scores = [float(scores[i]) for i in picks]
        resampled_labels = [bool(labels[i]) for i in picks]
        if not any(resampled_labels) or all(resampled_labels):
            continue
        value = float(statistic(resampled_scores, resampled_labels))
        if math.isfinite(value):
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    tail = (1.0 - float(level)) / 2.0
    low = values[max(0, int(math.floor(tail * len(values))))]
    high = values[min(len(values) - 1, int(math.ceil((1.0 - tail) * len(values))) - 1)]
    return (float(low), float(high))


# ---------------------------------------------------------------------------
# Scoring methods
# ---------------------------------------------------------------------------


def _library_items(molecules: Sequence[Any]) -> Tuple[List[Any], List[str], List[str]]:
    mols: List[Any] = []
    names: List[str] = []
    smiles: List[str] = []
    for index, item in enumerate(molecules):
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            raise ValueError(f"library item {index + 1} is not a parsable molecule")
        mols.append(mol)
        names.append(_mol_name(mol, f"ligand_{index + 1}"))
        smiles.append(_safe_smiles(mol))
    return mols, names, smiles


def _mol_name(mol, fallback: str) -> str:
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


def fingerprint_scores(
    library: Sequence[Any],
    actives: Sequence[Any],
    *,
    kind: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    metric: str = "tanimoto",
    names: Optional[Sequence[str]] = None,
    leave_one_out: bool = True,
) -> Ranking:
    """Rank a library by 2-D similarity to the **most similar** active.

    The standard ligand-based screen: the score of a library molecule is its
    best similarity to any known active, so a molecule that resembles one active
    strongly is promoted even if it looks nothing like the others.  The
    fingerprint, radius and coefficient are the arguments of
    :func:`odock.ligandsim.similarity`.

    ``leave_one_out`` (default) matters more than it looks: the actives are
    normally *in* the library being screened, so scoring each active against the
    full active set gives it a similarity of 1.0 to itself and any method that
    does that reports a perfect AUC.  With leave-one-out an active is scored
    against the *other* actives only, which is the only version of this number
    that says anything about finding a new active.
    """
    require_rdkit()
    from .ligandsim import fingerprint, similarity

    start = time.perf_counter()
    mols, labels, _ = _library_items(library)
    if names is not None:
        labels = [str(names[i]) if i < len(names) else labels[i] for i in range(len(labels))]
    active_mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in actives]
    if not active_mols:
        raise ValueError("no active to score against")
    active_labels = [_mol_name(mol, f"active_{i + 1}") for i, mol in enumerate(active_mols)]
    active_fps = [
        fingerprint(mol, kind=kind, radius=radius, n_bits=n_bits) for mol in active_mols
    ]
    entries: List[Tuple[str, float]] = []
    for label, mol in zip(labels, mols):
        query = fingerprint(mol, kind=kind, radius=radius, n_bits=n_bits)
        scores = [
            similarity(query, other, metric=metric)
            for index, other in enumerate(active_fps)
            if not (leave_one_out and label == active_labels[index])
        ]
        if not scores:  # the only active in the set is the molecule itself
            entries.append((label, 0.0))
            continue
        entries.append((label, float(max(scores))))
    entries.sort(key=lambda item: -item[1])
    notes = ["leave-one-out" if leave_one_out else "self-similarity NOT removed (leaky)"]
    return Ranking(
        method=f"fingerprint_{kind}",
        entries=entries,
        seconds=time.perf_counter() - start,
        notes=notes,
    )


def pharmacophore_scores(
    library: Sequence[Any],
    actives: Sequence[Any],
    *,
    conformers: int = 4,
    seed: int = 20240101,
    core: Optional[str] = None,
    frame: str = "align",
    names: Optional[Sequence[str]] = None,
    leave_one_out: bool = True,
) -> Ranking:
    """Rank a library by fit to a pharmacophore model built from the actives.

    The model is built by :func:`odock.pharmacophore.build_model` (so its features
    carry support counts) and the library is scored by
    :func:`odock.pharmacophore.screen`; a molecule that cannot be aligned or that
    violates the shape constraint scores 0 and keeps the reason.

    ``leave_one_out`` (default) builds **one model per active, leaving that active
    out**, and reports each molecule's *mean* fit over the folds.  A model built
    from all the actives scores them perfectly by construction — they are its
    features — and a benchmark that does that measures nothing but the leak.  The
    mean over folds is the honest version: an active is scored by the models that
    never saw it, and a decoy is scored by every model.
    """
    require_rdkit()
    from .pharmacophore import build_model, screen

    start = time.perf_counter()
    mols, labels, _ = _library_items(library)
    if names is not None:
        labels = [str(names[i]) if i < len(names) else labels[i] for i in range(len(labels))]
    active_mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in actives]
    if not active_mols:
        raise ValueError("no active to build a model from")
    active_labels = [_mol_name(mol, f"active_{i + 1}") for i, mol in enumerate(active_mols)]

    folds: List[Tuple[List[Any], List[str]]] = []
    if leave_one_out and len(active_mols) >= 3:
        for index in range(len(active_mols)):
            members = [mol for j, mol in enumerate(active_mols) if j != index]
            folds.append((members, [active_labels[j] for j in range(len(active_mols)) if j != index]))
    else:
        folds.append((active_mols, active_labels))

    totals: Dict[str, float] = {label: 0.0 for label in labels}
    counts: Dict[str, int] = {label: 0 for label in labels}
    rejected: Dict[str, int] = {label: 0 for label in labels}
    feature_counts: List[int] = []
    for members, member_labels in folds:
        model = build_model(
            members, frame=frame, core=core, seed=seed, names=member_labels
        )
        feature_counts.append(model.n_features)
        hits = screen(model, mols, names=labels, conformers=conformers, seed=seed)
        for hit in hits.hits:
            totals[hit.name] = totals.get(hit.name, 0.0) + float(hit.fit)
            counts[hit.name] = counts.get(hit.name, 0) + 1
            if hit.rejected:
                rejected[hit.name] = rejected.get(hit.name, 0) + 1
    entries = [
        (label, totals[label] / counts[label] if counts[label] else 0.0) for label in labels
    ]
    entries.sort(key=lambda item: -item[1])
    notes = [
        f"{len(folds)} model(s), "
        + ("leave-one-out" if len(folds) > 1 else "built from every active (leaky)")
        + f"; features per model {sorted(set(feature_counts))}",
        f"mean fit over folds; {sum(1 for value in rejected.values() if value)} molecule(s) "
        "were rejected in at least one fold",
    ]
    return Ranking(
        method="pharmacophore",
        entries=entries,
        seconds=time.perf_counter() - start,
        notes=notes,
    )


# -- shape / electrostatic overlay ------------------------------------------

#: Voxel size of the shape/ESP overlay grids (Å).
_SHAPE_VOXEL = 0.5
#: Margin around the reference molecule the grid has to cover (Å).
_SHAPE_MARGIN = 5.0
#: Gaussian width used to soften each atom's occupancy (Å).
_SHAPE_BLUR = 1.6


def _grid_axes(reference: np.ndarray, margin: float = _SHAPE_MARGIN, voxel: float = _SHAPE_VOXEL):
    low = reference.min(axis=0) - margin
    high = reference.max(axis=0) + margin
    axes = [np.arange(low[i], high[i] + voxel, voxel) for i in range(3)]
    return axes


def _occupancy(coords: np.ndarray, molecules: Sequence[Any], axes, voxel: float) -> np.ndarray:
    """A blurred occupancy grid from the heavy atoms of one placed molecule."""
    shape = tuple(len(axis) for axis in axes)
    grid = np.zeros(shape, dtype=np.float32)
    if coords.shape[0] == 0:
        return grid
    voxels = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    for index in range(coords.shape[0]):
        centre = coords[index]
        distance2 = ((voxels - centre) ** 2).sum(axis=-1)
        grid += np.exp(-distance2 / (2.0 * _SHAPE_BLUR ** 2)).astype(np.float32)
    return grid


def _shape_overlap(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Tanimoto-like overlap of two occupancy grids (0 = no shape in common)."""
    intersection = float(np.minimum(grid_a, grid_b).sum())
    union = float(np.maximum(grid_a, grid_b).sum())
    return intersection / union if union > 0 else 0.0


def _charge_grid(
    coords: np.ndarray, charges: Sequence[float], axes, voxel: float
) -> np.ndarray:
    """A blurred partial-charge field, one value per voxel."""
    shape = tuple(len(axis) for axis in axes)
    grid = np.zeros(shape, dtype=np.float32)
    if coords.shape[0] == 0:
        return grid
    voxels = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    for index in range(coords.shape[0]):
        centre = coords[index]
        weight = float(charges[index])
        if weight == 0.0:
            continue
        distance2 = ((voxels - centre) ** 2).sum(axis=-1)
        grid += (weight * np.exp(-distance2 / (2.0 * _SHAPE_BLUR ** 2))).astype(np.float32)
    return grid


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float((a * b).sum())
    denominator = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return numerator / denominator if denominator > 0 else 0.0


def _gasteiger_charges(mol) -> np.ndarray:
    from rdkit.Chem import AllChem

    work = Chem.Mol(mol)
    try:
        if not any(atom.GetAtomicNum() == 1 for atom in work.GetAtoms()):
            work = Chem.AddHs(work)
        AllChem.ComputeGasteigerCharges(work)
    except Exception:  # pragma: no cover - defensive
        return np.zeros(work.GetNumAtoms(), dtype=float)
    values = []
    for atom in work.GetAtoms():
        try:
            values.append(float(atom.GetDoubleProp("_GasteigerCharge")))
        except Exception:
            values.append(0.0)
    values = np.asarray(values, dtype=float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def shape_scores(
    library: Sequence[Any],
    actives: Sequence[Any],
    *,
    conformers: int = 4,
    seed: int = 20240101,
    core: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
    electrostatic: bool = True,
    leave_one_out: bool = True,
) -> Ranking:
    """Rank a library by a rigid shape (and electrostatic) overlay on one active.

    The library molecule is placed on the actives' pharmacophore model frame (the
    same rigid alignment :func:`odock.pharmacophore.fit_score` uses) and compared
    with a **reference active** — the largest active by heavy-atom count, unless
    ``reference`` names one:

    * **shape** — the overlap coefficient ``Σ min(A, B) / Σ max(A, B)`` of two
      blurred occupancy grids (0.5 Å voxels, 1.6 Å Gaussian blur, van der Waals
      radii from :func:`odock.sasa.radius_of`);
    * **electrostatic** — the cosine between the two Gasteiger partial-charge
      fields on the same grid.  With ``electrostatic=True`` the reported score is
      ``0.5 · shape + 0.5 · max(0, cos)``, a documented convention rather than a
      fitted weighting.

    This is a crude overlay, not a validated shape/ESP method: no dielectric, no
    shielding, no atom typing beyond the Gasteiger model, and the alignment is
    rigid so a flexible analogue is penalised for its conformers.  It is here
    because a benchmark with only fingerprint and pharmacophore rows is a
    benchmark of 2-D and feature similarity, not of shape.
    """
    require_rdkit()
    from .pharmacophore import PharmacophoreModel, _align_on_core, _align_on_features, build_model

    start = time.perf_counter()
    mols, labels, _ = _library_items(library)
    if names is not None:
        labels = [str(names[i]) if i < len(names) else labels[i] for i in range(len(labels))]
    active_mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in actives]
    if not active_mols:
        raise ValueError("no active to overlay")
    model = build_model(active_mols, frame="align", core=core, seed=seed)
    # Place the actives in the model frame, then pick the reference.  A molecule
    # without 3-D coordinates has to be embedded first: the alignment reads the
    # conformer, so `conf_id=0` on a coordinate-free molecule is an error, not a
    # default.
    from .pharmacophore import _ensure_conformer

    placed_actives: List[Tuple[str, np.ndarray, List[float]]] = []
    for mol in active_mols:
        name = _mol_name(mol, "active")
        work = mol
        if work.GetNumConformers() == 0:
            work, _ = _ensure_conformer(work, seed=seed, conformers=1)
        placed = None
        for conf in range(work.GetNumConformers()):
            placed = _align_on_core(work, model, conf_id=conf)
            if placed is None:
                found = _align_on_features(work, model, conf_id=conf)
                placed = found[0] if found is not None else None
            if placed is not None:
                break
        if placed is None:
            continue
        heavy = [atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1]
        charges = _gasteiger_charges(work)
        placed_actives.append((name, placed[heavy], [float(charges[i]) for i in heavy]))
    if not placed_actives:
        raise ValueError("no active could be placed on the model frame, so no overlay is possible")
    if reference:
        chosen = [entry for entry in placed_actives if entry[0] == reference]
        if not chosen:
            raise ValueError(f"the reference {reference!r} is not one of the actives")
        references = chosen
    else:
        # Every placed active is a reference; a molecule's score is its best
        # overlay on a reference that is not itself (leave-one-out), and on any
        # reference when the molecule is a decoy.
        references = placed_actives
    axes = _grid_axes(np.vstack([entry[1] for entry in placed_actives]))
    reference_grids = {
        name: (_occupancy(coords, active_mols, axes, _SHAPE_VOXEL),
               _charge_grid(coords, charges, axes, _SHAPE_VOXEL))
        for name, coords, charges in references
    }
    ref_name = max(references, key=lambda entry: len(entry[1]))[0]
    entries: List[Tuple[str, float]] = []
    notes = [
        f"{len(references)} reference active(s); best overlay"
        + (" excluding the molecule itself" if leave_one_out else " (leaky: a reference can be the molecule)")
        + f"; largest is {ref_name}"
    ]
    active_label_set = {name for name, _, _ in placed_actives}
    for label, mol in zip(labels, mols):
        work = Chem.Mol(mol)
        if work.GetNumConformers() == 0:
            work, _ = _ensure_conformer(work, seed=seed, conformers=conformers)
        best_score = 0.0
        best_shape = 0.0
        for conf in range(work.GetNumConformers()):
            placed = _align_on_core(work, model, conf_id=conf)
            if placed is None:
                found = _align_on_features(work, model, conf_id=conf)
                placed = found[0] if found is not None else None
            if placed is None:
                continue
            heavy = [atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1]
            coords = placed[heavy]
            grid = _occupancy(coords, work, axes, _SHAPE_VOXEL)
            charges = None
            for name, coords_ref, charges_ref in references:
                if leave_one_out and label == name and label in active_label_set:
                    continue
                reference_grid, reference_charge = reference_grids[name]
                overlap = _shape_overlap(reference_grid, grid)
                if electrostatic:
                    if charges is None:
                        charges = _gasteiger_charges(work)
                    charge_grid = _charge_grid(
                        coords, [float(charges[i]) for i in heavy], axes, _SHAPE_VOXEL
                    )
                    score = 0.5 * overlap + 0.5 * max(0.0, _cosine(reference_charge, charge_grid))
                else:
                    score = overlap
                if score > best_score:
                    best_score = score
                    best_shape = overlap
        entries.append((label, float(best_score)))
    entries.sort(key=lambda item: -item[1])
    return Ranking(
        method="shape_esp" if electrostatic else "shape",
        entries=entries,
        seconds=time.perf_counter() - start,
        notes=notes,
    )


def random_scores(
    library: Sequence[Any], *, seed: int = 20240101, names: Optional[Sequence[str]] = None
) -> Ranking:
    """The negative control: a random ranking of the same library."""
    mols, labels, _ = _library_items(library)
    if names is not None:
        labels = [str(names[i]) if i < len(names) else labels[i] for i in range(len(labels))]
    rng = random.Random(int(seed))
    entries = [(label, rng.random()) for label in labels]
    entries.sort(key=lambda item: -item[1])
    return Ranking(method="random", entries=entries, notes=[f"seed {int(seed)}"])


def property_scores(
    library: Sequence[Any], *, by: str = "MW", names: Optional[Sequence[str]] = None
) -> Ranking:
    """The trivial-signal control: rank by a single property, ascending.

    A property-only ranking is the thing an enrichment number has to beat: if the
    actives are simply the biggest (or most lipophilic) molecules in the library,
    then any method that happens to correlate with size will look good.  The
    property names are :data:`odock.decoys.PROPERTIES`.
    """
    require_rdkit()
    from .decoys import PROPERTIES, property_profile

    if by not in PROPERTIES:
        raise ValueError(f"unknown property {by!r}; use one of {list(PROPERTIES)}")
    mols, labels, _ = _library_items(library)
    if names is not None:
        labels = [str(names[i]) if i < len(names) else labels[i] for i in range(len(labels))]
    # Ascending: smaller molecules first, so the ranking does not accidentally
    # favour the actives by size.
    entries = [(label, -float(getattr(property_profile(mol), by))) for label, mol in zip(labels, mols)]
    entries.sort(key=lambda item: -item[1])
    return Ranking(method=f"property_{by}", entries=entries, notes=[f"ranked by {by}, ascending"])


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    """One method's row: its ranked list and its metrics with intervals."""

    ranking: Ranking
    stats: Dict[str, Any] = field(default_factory=dict)
    intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def method(self) -> str:
        return self.ranking.method

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "stats": dict(self.stats),
            "intervals": {key: [round(lo, 4), round(hi, 4)] for key, (lo, hi) in self.intervals.items()},
            "ranking": self.ranking.as_dict(),
            "notes": list(self.notes),
        }


@dataclass
class BenchmarkReport:
    """Every method and control against the same actives and decoys."""

    results: List[MethodResult] = field(default_factory=list)
    n_actives: int = 0
    n_decoys: int = 0
    decoy_quality: Optional[Dict[str, Any]] = None
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return self.n_actives + self.n_decoys

    def by_method(self) -> Dict[str, MethodResult]:
        return {result.method: result for result in self.results}

    def table(self) -> str:
        """The headline table: EF1 %, EF5 %, AUC, BEDROC(20) with intervals."""
        lines = [
            f"{'method':<20}{'EF1%':>8}{'EF5%':>8}{'AUC':>8}{'BEDROC':>8}  95% CI (AUC / BEDROC)",
            "-" * 20 + "-" * 8 * 4 + "  " + "-" * 32,
        ]
        for result in self.results:
            stats = result.stats
            auc_ci = result.intervals.get("auc", (float("nan"), float("nan")))
            bedroc_ci = result.intervals.get("bedroc", (float("nan"), float("nan")))
            lines.append(
                f"{result.method[:19]:<20}{stats.get('ef1', float('nan')):>8.2f}"
                f"{stats.get('ef5', float('nan')):>8.2f}{stats.get('auc', float('nan')):>8.3f}"
                f"{stats.get('bedroc', float('nan')):>8.3f}  "
                f"[{auc_ci[0]:.2f}, {auc_ci[1]:.2f}] / [{bedroc_ci[0]:.2f}, {bedroc_ci[1]:.2f}]"
            )
        lines.append("")
        lines.append(
            f"{self.n_actives} active(s), {self.n_decoys} decoy(s), {self.n_total} molecule(s)"
        )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_actives": int(self.n_actives),
            "n_decoys": int(self.n_decoys),
            "n_total": int(self.n_total),
            "decoy_quality": self.decoy_quality,
            "seconds": round(float(self.seconds), 4),
            "results": [result.as_dict() for result in self.results],
            "notes": list(self.notes),
        }


def benchmark(
    actives: Sequence[Any],
    decoys: Sequence[Any],
    *,
    names: Optional[Sequence[str]] = None,
    decoy_names: Optional[Sequence[str]] = None,
    methods: Sequence[str] = ("fingerprint", "pharmacophore", "shape"),
    conformers: int = 4,
    seed: int = 20240101,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    decoy_quality: Optional[Dict[str, Any]] = None,
    controls: Sequence[str] = ("random", "property"),
) -> BenchmarkReport:
    """Score a labelled set with every method and both controls.

    ``actives`` and ``decoys`` are molecules or SMILES; every method ranks the
    combined set and the labels are "is this molecule an active".  Returns the
    metrics, the bootstrap intervals and the rankings, plus the decoy set's
    property-matching evidence when it is supplied (which it should be: without it
    the enrichment numbers cannot be read).
    """
    require_rdkit()
    start = time.perf_counter()
    active_mols, active_labels, _ = _library_items(actives)
    decoy_mols, decoy_labels, _ = _library_items(decoys)
    if decoy_names is not None:
        decoy_labels = [
            str(decoy_names[i]) if i < len(decoy_names) else decoy_labels[i]
            for i in range(len(decoy_labels))
        ]
    if names is not None:
        active_labels = [
            str(names[i]) if i < len(names) else active_labels[i] for i in range(len(active_labels))
        ]
    library = list(active_mols) + list(decoy_mols)
    labels = [True] * len(active_mols) + [False] * len(decoy_mols)
    display = list(active_labels) + list(decoy_labels)
    label_lookup = dict(zip(display, labels))
    active_set = set(active_labels)

    wanted = {str(name).strip().lower() for name in methods}
    results: List[MethodResult] = []
    calls: List[Tuple[str, Callable[[], Ranking]]] = []
    if "fingerprint" in wanted:
        calls.append(("fingerprint", lambda: fingerprint_scores(library, active_mols, names=display)))
    if "pharmacophore" in wanted:
        calls.append(
            (
                "pharmacophore",
                lambda: pharmacophore_scores(
                    library, active_mols, conformers=conformers, seed=seed, names=display
                ),
            )
        )
    if "shape" in wanted:
        calls.append(
            (
                "shape",
                lambda: shape_scores(
                    library, active_mols, conformers=conformers, seed=seed, names=display
                ),
            )
        )
    if "shape_only" in wanted:
        calls.append(
            (
                "shape_only",
                lambda: shape_scores(
                    library, active_mols, conformers=conformers, seed=seed, names=display,
                    electrostatic=False,
                ),
            )
        )
    for control in controls:
        key = str(control).strip().lower()
        if key == "random":
            calls.append(("random", lambda: random_scores(library, seed=seed, names=display)))
        elif key == "property":
            calls.append(("property_MW", lambda: property_scores(library, by="MW", names=display)))
            calls.append(
                ("property_LogP", lambda: property_scores(library, by="LogP", names=display))
            )
    for _, call in calls:
        ranking = call()
        scores = [score for _, score in ranking.entries]
        ordered_labels = [label_lookup.get(name, False) for name, _ in ranking.entries]
        stats = metrics(scores, ordered_labels)
        intervals = {
            "auc": bootstrap_ci(scores, ordered_labels, auc, samples=bootstrap, seed=seed),
            "bedroc": bootstrap_ci(
                scores, ordered_labels, lambda s, y: bedroc(s, y), samples=bootstrap, seed=seed
            ),
        }
        for fraction in EF_FRACTIONS:
            key = f"ef{int(round(fraction * 100))}"
            intervals[key] = bootstrap_ci(
                scores,
                ordered_labels,
                lambda s, y, f=fraction: ef_at(s, y, f)["ef"],
                samples=bootstrap,
                seed=seed,
            )
        results.append(MethodResult(ranking=ranking, stats=stats, intervals=intervals))
    order = [
        "fingerprint_morgan",
        "pharmacophore",
        "shape_esp",
        "shape_only",
        "random",
        "property_MW",
        "property_LogP",
    ]
    results.sort(key=lambda result: (order.index(result.method) if result.method in order else 99))
    report = BenchmarkReport(
        results=results,
        n_actives=len(active_mols),
        n_decoys=len(decoy_mols),
        decoy_quality=decoy_quality,
        seconds=time.perf_counter() - start,
    )
    if len(active_mols) < 10 or len(decoy_mols) < 30:
        report.notes.append(
            f"{len(active_mols)} active(s) and {len(decoy_mols)} decoy(s) is a *case study*, "
            "not a benchmark: the bootstrap intervals below are wide enough to contain most "
            "plausible answers, and no enrichment claim should be read from it."
        )
    if decoy_quality is not None and not decoy_quality.get("matched", False):
        report.notes.append(
            "the decoy set is NOT property-matched (max |SMD| "
            f"{decoy_quality.get('max_abs_smd')}); the metrics are therefore reported with "
            "that caveat rather than as a matched-decoy benchmark"
        )
    return report


# ---------------------------------------------------------------------------
# Pre-filtering for a docking campaign
# ---------------------------------------------------------------------------


def prefilter(
    library: Sequence[Any],
    actives: Sequence[Any],
    *,
    keep: float = 0.1,
    kind: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    metric: str = "tanimoto",
    exhaustiveness: int = 8,
    cores: Optional[int] = None,
) -> Dict[str, Any]:
    """What a fingerprint pre-filter keeps, and what it saves the docking run.

    The realistic use of a ligand-based screen is as a *pre-filter*: score the
    library by similarity to the actives, dock only the top ``keep`` fraction, and
    accept that some actives are thrown away.  This function reports exactly that
    trade: how many molecules survive, **how many actives survive**, and the
    docking workload before and after, using :func:`odock.screen.estimate_library`
    (imported, never modified) as the cost model.
    """
    require_rdkit()
    from .screen import estimate_cost

    ranking = fingerprint_scores(
        library, actives, kind=kind, radius=radius, n_bits=n_bits, metric=metric
    )
    n_total = len(ranking)
    cut = max(1, int(math.ceil(float(keep) * n_total)))
    kept = ranking.entries[:cut]
    active_labels = {_mol_name(Chem.MolFromSmiles(item) if isinstance(item, str) else item, f"active_{i + 1}")
                     for i, item in enumerate(actives)}
    # Active *names* are supplied by the caller's library; match on the label used
    # by the ranking, so the caller's titles must be the same on both sides.
    active_in_ranking = [name for name in ranking.names if name in active_labels]
    survived = [name for name, _ in kept if name in active_labels]
    workload = estimate_cost(
        n_ligands=n_total, exhaustiveness=exhaustiveness, cores=cores
    )
    workload_kept = estimate_cost(
        n_ligands=cut, exhaustiveness=exhaustiveness, cores=cores
    )
    return {
        "keep": float(keep),
        "n_library": n_total,
        "n_kept": cut,
        "fraction_kept": cut / n_total if n_total else 0.0,
        "n_actives": len(active_in_ranking),
        "actives_kept": len(survived),
        "actives_lost": sorted(set(active_in_ranking) - set(survived)),
        "active_recall": (len(survived) / len(active_in_ranking)) if active_in_ranking else 0.0,
        "estimated_seconds_full": round(workload, 1),
        "estimated_seconds_kept": round(workload_kept, 1),
        "workload_saved_fraction": (
            1.0 - workload_kept / workload if workload > 0 else 0.0
        ),
        "notes": [
            "the cost model is odock.screen's own estimate (a fixed per-molecule cost plus a "
            "search cost that grows with exhaustiveness and the torsion count); it is a "
            "planning figure, not a measurement"
        ],
    }

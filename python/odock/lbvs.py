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
from pathlib import Path
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
    "paired_difference",
    "minimum_detectable_difference",
    "fingerprint_scores",
    "pharmacophore_scores",
    "shape_scores",
    "random_scores",
    "property_scores",
    "prefilter",
    "benchmark",
    "read_targets",
    "stratified_difference",
    "benchmark_per_target",
    "StratifiedReport",
]

#: Early-recognition fractions for EF1 % and EF5 %.
EF_FRACTIONS: Tuple[float, ...] = (0.01, 0.05)

#: BEDROC's weighting: α = 20 puts 80 % of the score in the top 8 % of the list
#: (the usual early-recognition setting).
BEDROC_ALPHA = 20.0

#: Bootstrap resamples.  1000 gives a stable percentile interval at the cost of
#: 1000 metric evaluations, which is nothing next to scoring a library.
BOOTSTRAP_SAMPLES = 1000

#: The multiplier that turns a paired standard error into a minimum detectable
#: difference: ``z_{1-α/2} + z_{power} = 1.96 + 0.84 = 2.80`` for a two-sided 5 %
#: test at 80 % power.  Stated here because the whole point of the number is that a
#: reader can check the constant, not just the conclusion.
MDD_Z = 2.80



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
    #: Per-molecule detail (the shape and electrostatic terms of an overlay, the
    #: ensemble behind it), keyed by molecule name.  Empty for the 2-D methods.
    details: Dict[str, Any] = field(default_factory=dict)

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

    def terms(self) -> Dict[str, Any]:
        """Mean shape and electrostatic term over the molecules that have them."""
        shape = [
            float(item["shape"]) for item in self.details.values() if "shape" in item
        ]
        esp = [float(item["esp"]) for item in self.details.values() if "esp" in item]
        return {
            "mean_shape": (sum(shape) / len(shape)) if shape else 0.0,
            "mean_esp": (sum(esp) / len(esp)) if esp else 0.0,
            "n": len(shape),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n": len(self.entries),
            "seconds": round(float(self.seconds), 4),
            "entries": [[name, round(float(score), 6)] for name, score in self.entries],
            "notes": list(self.notes),
            "details": {name: dict(value) for name, value in self.details.items()},
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


def minimum_detectable_difference(
    standard_error: float, *, power: float = 0.8, level: float = 0.95
) -> float:
    """The smallest difference a test with this standard error can detect.

    ``(z_{1−α/2} + z_{power}) · SE`` — 2.80 · SE for a two-sided 5 % test at 80 %
    power, the convention :data:`MDD_Z` names.  The point of computing it is to put
    it **next to the observed gap**: a benchmark that can only ever detect a 0.30 AUC
    difference cannot say anything about two methods 0.05 apart, and saying so is
    more useful than reporting the intervals as if they settled it.
    """
    from statistics import NormalDist

    if not math.isfinite(float(standard_error)) or float(standard_error) < 0.0:
        return float("nan")
    z_alpha = NormalDist().inv_cdf(1.0 - (1.0 - float(level)) / 2.0)
    z_power = NormalDist().inv_cdf(float(power))
    return float(z_alpha + z_power) * float(standard_error)


def paired_difference(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    labels: Sequence[bool],
    *,
    metric: Callable[[Sequence[float], Sequence[bool]], float] = auc,
    samples: int = BOOTSTRAP_SAMPLES,
    level: float = 0.95,
    seed: int = 20240101,
) -> Dict[str, Any]:
    """Bootstrap the **paired** difference ``metric(a) − metric(b)`` on one set.

    Comparing two independent confidence intervals is the wrong test: the engines
    score the same molecules, so most of the width of each interval is the same
    molecule-to-molecule noise, and it cancels in the difference.  This resamples
    molecules once and recomputes *both* metrics on the same resample, which is the
    paired test, and it returns:

    * ``difference`` — the observed ``metric(a) − metric(b)``;
    * ``ci`` — the percentile interval of the **difference**;
    * ``se`` — its bootstrap standard error;
    * ``mdd`` — the minimum detectable difference at that standard error
      (:func:`minimum_detectable_difference`);
    * ``resolvable`` — whether the interval excludes zero, i.e. whether this set can
      tell the two apart at all.

    ``samples`` should be large enough that ``se`` is stable; the default is the
    module's :data:`BOOTSTRAP_SAMPLES`.
    """
    _check_inputs(scores_a, labels)
    _check_inputs(scores_b, labels)
    observed = float(metric(scores_a, labels)) - float(metric(scores_b, labels))
    rng = random.Random(int(seed))
    n = len(labels)
    values: List[float] = []
    for _ in range(max(1, int(samples))):
        picks = [rng.randrange(n) for _ in range(n)]
        resampled_labels = [bool(labels[i]) for i in picks]
        if not any(resampled_labels) or all(resampled_labels):
            continue
        left = float(metric([float(scores_a[i]) for i in picks], resampled_labels))
        right = float(metric([float(scores_b[i]) for i in picks], resampled_labels))
        if math.isfinite(left) and math.isfinite(right):
            values.append(left - right)
    if not values:
        return {
            "difference": observed, "ci": (float("nan"), float("nan")),
            "se": float("nan"), "mdd": float("nan"), "resolvable": False,
            "samples": 0, "n": n, "n_actives": int(sum(1 for flag in labels if flag)),
        }
    values.sort()
    tail = (1.0 - float(level)) / 2.0
    low = values[max(0, int(math.floor(tail * len(values))))]
    high = values[min(len(values) - 1, int(math.ceil((1.0 - tail) * len(values))) - 1)]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    se = math.sqrt(variance)
    return {
        "difference": round(observed, 4),
        "ci": (round(float(low), 4), round(float(high), 4)),
        "se": round(se, 4),
        "mdd": round(minimum_detectable_difference(se, level=level), 4),
        "resolvable": bool(low > 0.0 or high < 0.0),
        "samples": len(values),
        "n": n,
        "n_actives": int(sum(1 for flag in labels if flag)),
        "n_decoys": int(sum(1 for flag in labels if not flag)),
    }


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
    engine: str = "overlay",
    conformers: Optional[int] = None,
    seed: int = 20240101,
    core: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    reference: Optional[str] = None,
    electrostatic: bool = True,
    leave_one_out: bool = True,
    sigma: float = 0.5,
    shape_metric: str = "tanimoto",
    shape_weight: float = 0.5,
    align_on: str = "shape",
) -> Ranking:
    """Rank a library by a 3-D shape (and electrostatic) overlay on the actives.

    ``engine`` selects between the two implementations, both of which are kept so
    they can be compared on the same data:

    ``"overlay"`` (**the default**) — :mod:`odock.overlay`.  Analytic Gaussian
    shape densities at σ = 0.5 Å, a shape Tanimoto or Carbo index, a Gasteiger
    field Carbo index, best over each molecule's conformers and best over the
    reference actives, with a six-dimensional (rotation **and** translation)
    alignment from a deterministic seed set plus a multi-resolution refinement.
    It is the default because it is the one that finds the pose a chemically
    correct superposition finds — measured, benzamidine against
    hydroxybenzamidine scores 0.913 where the rigid grid scorer below cannot
    reach it — and because it reports both terms per molecule.  Cost: ~13 ms per
    molecule-versus-reference overlay on this machine.

    ``"crude"`` — the original rigid overlay: a blurred occupancy grid (0.5 Å
    voxels, 1.6 Å blur, van der Waals radii from :func:`odock.sasa.radius_of`)
    compared by ``Σ min(A, B) / Σ max(A, B)`` in a *single* pose, the pose the
    actives' pharmacophore model frame puts the molecule in, plus the cosine of
    the two Gasteiger charge grids.  It is kept behind this flag so the benchmark
    can measure the difference between the two on the same actives and decoys.

    With ``electrostatic=True`` both report ``shape_weight · shape +
    (1 − shape_weight) · max(0, ESP)``; with it off, the score is the shape term
    alone.  Neither is a validated shape/ESP method: there is no receptor, no
    dielectric, no desolvation and no atom typing beyond Gasteiger, and the score
    is a statement about geometry, never about binding.

    ``conformers`` is the number of embedding **attempts** per molecule: ``None``
    (the default) scales it with the rotatable-bond count
    (:func:`odock.conformers.rotor_scaled_attempts`), and an explicit integer is
    honoured exactly so a published protocol stays reproducible.  The grid engine has
    no rotor rule, so it falls back to 16 attempts when ``conformers`` is ``None``.
    """
    require_rdkit()
    key = str(engine).strip().lower()
    if key in ("overlay", "gaussian"):
        from .overlay import screen as overlay_screen

        result = overlay_screen(
            library, actives, conformers=conformers, seed=seed, names=names,
            reference=reference, leave_one_out=leave_one_out, sigma=sigma,
            shape_metric=shape_metric, shape_weight=shape_weight,
            electrostatic=electrostatic, align_on=align_on,
        )
        coverage = [
            float(item.get("torsion_coverage", 0.0))
            for item in result.ensembles.values()
            if item.get("n_conformers")
        ]
        counts = [int(item.get("n_conformers", 0)) for item in result.ensembles.values()]
        attempts = [int(item.get("requested", 0)) for item in result.ensembles.values()]
        notes = list(result.notes)
        if counts:
            notes.append(
                f"{sum(counts) / len(counts):.2f} conformer(s) per molecule on average "
                f"from {sum(attempts) / len(attempts):.1f} attempt(s)"
                + (
                    f", mean torsion coverage {sum(coverage) / len(coverage):.0%}"
                    if coverage else ""
                )
                + f"; {result.n_library} molecule(s) in {result.seconds:.2f} s"
            )
        return Ranking(
            method="overlay_esp" if electrostatic else "overlay_shape",
            entries=list(result.entries),
            seconds=result.seconds,
            notes=notes,
            details={name: detail.as_dict() for name, detail in result.details.items()},
        )

    if key not in ("crude", "grid"):
        raise ValueError(
            f"unknown shape engine {engine!r}; use 'overlay' (the default) or 'crude'"
        )
    return _crude_shape_scores(
        library, actives,
        conformers=16 if conformers is None else conformers,
        seed=seed, core=core, names=names,
        reference=reference, electrostatic=electrostatic,
        leave_one_out=leave_one_out,
    )


def _crude_shape_scores(
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
    """The original rigid grid overlay, kept behind ``shape_scores(engine="crude")``.

    The library molecule is placed on the actives' pharmacophore model frame (the
    same rigid alignment :func:`odock.pharmacophore.fit_score` uses) and compared
    with a reference active in that one pose: a blurred occupancy grid overlap
    (0.5 Å voxels, 1.6 Å Gaussian blur) and, optionally, the cosine between the two
    Gasteiger charge grids.  Its limits are the reason :mod:`odock.overlay` exists:
    the pose is fixed by the pharmacophore frame rather than optimised, the grid is
    an approximation of the analytic overlap, and a flexible analogue is judged on
    the conformer the embedder listed first.
    """
    require_rdkit()
    from .pharmacophore import _align_on_core, _align_on_features, build_model

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
        method="crude_esp" if electrostatic else "crude_shape",
        entries=entries,
        seconds=time.perf_counter() - start,
        notes=notes
        + [
            "this is the rigid grid overlay (engine='crude'); the pose comes from "
            "the pharmacophore frame and is not optimised"
        ],
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
    #: ``(name, is_active)`` for every molecule, in library order.  It is what makes
    #: the **paired** comparison possible: two methods are compared on the same
    #: molecules because a molecule's label does not depend on a ranking.
    labels: List[Tuple[str, bool]] = field(default_factory=list)
    decoy_quality: Optional[Dict[str, Any]] = None
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return self.n_actives + self.n_decoys

    def by_method(self) -> Dict[str, MethodResult]:
        return {result.method: result for result in self.results}

    def _aligned(self, method: str) -> Tuple[List[str], List[float], List[bool]]:
        """One method's scores **in library order**, with the labels in that order.

        The order matters and is the reason the first paired test was wrong: a ranking
        is sorted by that method's own scores, so two methods list the same molecules
        in different positions.  Re-ordering by name puts every method on the same
        molecule order, which is what makes the bootstrap *paired* — the same
        resampled molecule contributes its score under both engines.
        """
        result = self.by_method().get(method)
        if result is None:
            raise ValueError(
                f"no method {method!r} in this report; it has "
                f"{sorted(self.by_method())}"
            )
        score_of = {name: float(score) for name, score in result.ranking.entries}
        names = [name for name, _ in self.labels]
        labels = [bool(flag) for _, flag in self.labels]
        missing = [name for name in names if name not in score_of]
        if missing:
            raise ValueError(
                f"method {method!r} did not score {len(missing)} molecule(s) of this "
                f"report, e.g. {missing[:3]}; the comparison must be on the same set"
            )
        return names, [score_of[name] for name in names], labels

    def compare(
        self, method_a: str, method_b: str, *, samples: int = BOOTSTRAP_SAMPLES,
        metric: Callable[[Sequence[float], Sequence[bool]], float] = auc,
    ) -> Dict[str, Any]:
        """The **paired** difference between two methods on the same molecules.

        Comparing the two marginal confidence intervals is the wrong test and is why
        this benchmark could not rank its engines: both methods score the same
        molecules, so most of each interval's width is shared noise that cancels in
        the difference.  See :func:`paired_difference`.
        """
        _, scores_a, labels = self._aligned(method_a)
        _, scores_b, _ = self._aligned(method_b)
        report = paired_difference(
            scores_a, scores_b, labels, metric=metric, samples=samples,
            seed=20240101,
        )
        report["method_a"] = method_a
        report["method_b"] = method_b
        return report

    def paired_table(
        self, *, samples: int = BOOTSTRAP_SAMPLES,
        methods: Optional[Sequence[str]] = None,
        baseline: Optional[str] = None,
    ) -> str:
        """Every pairing of methods, as a difference with its interval and MDD.

        With ``baseline`` set, every other method is compared against it; otherwise
        every unordered pair is printed once.  The last column is the answer to "can
        this benchmark tell them apart at all", and it is the number that was missing
        from the first version of this document.
        """
        chosen = list(methods) if methods is not None else [r.method for r in self.results]
        pairs = (
            [(baseline, other) for other in chosen if other != baseline]
            if baseline
            else [
                (chosen[i], chosen[j])
                for i in range(len(chosen))
                for j in range(i + 1, len(chosen))
            ]
        )
        lines = [
            f"{'method A':<18}{'method B':<18}{'AUC A':>8}{'AUC B':>8}{'delta':>8}"
            f"  {'95% CI of the delta':<22}{'MDD':>7}  resolvable",
            "-" * 18 + "-" * 18 + "-" * 8 * 3 + "  " + "-" * 22 + "-" * 7 + "  " + "-" * 10,
        ]
        for left, right in pairs:
            comparison = self.compare(left, right, samples=samples)
            base = self.by_method()[left].stats.get("auc", float("nan"))
            other = self.by_method()[right].stats.get("auc", float("nan"))
            low, high = comparison["ci"]
            lines.append(
                f"{left[:17]:<18}{right[:17]:<18}{base:>8.3f}{other:>8.3f}"
                f"{comparison['difference']:>+8.3f}  "
                f"[{low:>+6.3f}, {high:>+6.3f}]      "
                f"{comparison['mdd']:>7.3f}  "
                + ("yes" if comparison["resolvable"] else "NO")
            )
        lines.append("")
        lines.append(
            "delta = AUC(A) - AUC(B) on the same molecules; MDD = the smallest "
            f"difference this set can detect (2.80 x SE, {samples} resamples); "
            "'resolvable' = the delta's interval excludes zero"
        )
        return "\n".join(lines)

    def power(self, *, samples: int = BOOTSTRAP_SAMPLES) -> Dict[str, Any]:
        """What this set can and cannot resolve.

        The smallest paired difference this benchmark can detect between any two of
        its methods, computed per pair and summarised — the number that says whether
        a conclusion is available at all.
        """
        methods = [result.method for result in self.results]
        rows = [
            self.compare(methods[i], methods[j], samples=samples)
            for i in range(len(methods))
            for j in range(i + 1, len(methods))
        ]
        mdds = [row["mdd"] for row in rows if math.isfinite(row["mdd"])]
        return {
            "n_actives": int(self.n_actives),
            "n_decoys": int(self.n_decoys),
            "n_total": int(self.n_total),
            "samples": int(samples),
            "min_mdd": round(min(mdds), 4) if mdds else float("nan"),
            "median_mdd": round(sorted(mdds)[len(mdds) // 2], 4) if mdds else float("nan"),
            "max_mdd": round(max(mdds), 4) if mdds else float("nan"),
            "resolvable_pairs": sum(1 for row in rows if row["resolvable"]),
            #: Pairs whose difference is exactly zero in every resample: two methods
            #: that ranked identically on this set.  They are not "detected as
            #: equal", there is simply nothing to detect, and an MDD of 0.0 in the
            #: summary would otherwise look like infinite power.
            "tied_pairs": sum(1 for row in rows if row["mdd"] == 0.0),
            "pairs": len(rows),
            "rows": rows,
        }

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
            "labels": [[name, bool(flag)] for name, flag in self.labels],
            "decoy_quality": self.decoy_quality,
            "power": self.power(),
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
    shape_engine: str = "overlay",
    conformers: Optional[int] = 4,
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

    The method names are ``fingerprint``, ``pharmacophore`` and the shape family:
    ``shape`` (the engine ``shape_engine`` names — ``overlay`` by default),
    ``overlay`` / ``overlay_only`` (the :mod:`odock.overlay` engine, with and
    without electrostatics), ``crude`` / ``crude_only`` (the rigid grid scorer) and
    ``shape_only`` (the ``shape`` engine without electrostatics).  Running
    ``fingerprint,pharmacophore,overlay,overlay_only,crude`` on the same set is the
    comparison `docs/LBVS.md` reports.
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

    def _shape(engine: str, electrostatic: bool):
        return lambda: shape_scores(
            library, active_mols, engine=engine, conformers=conformers, seed=seed,
            names=display, electrostatic=electrostatic,
        )

    if "fingerprint" in wanted:
        calls.append(("fingerprint", lambda: fingerprint_scores(library, active_mols, names=display)))
    if "pharmacophore" in wanted:
        calls.append(
            (
                "pharmacophore",
                lambda: pharmacophore_scores(
                    library, active_mols,
                    # odock.pharmacophore has no rotor rule of its own, so a
                    # `conformers=None` benchmark (the shape rows scale with the
                    # rotors) gives it the library default rather than failing on
                    # `int(None)`.
                    conformers=16 if conformers is None else conformers,
                    seed=seed, names=display,
                ),
            )
        )
    if "shape" in wanted:
        calls.append(("shape", _shape(shape_engine, True)))
    if "overlay" in wanted:
        calls.append(("overlay", _shape("overlay", True)))
    if "overlay_only" in wanted:
        calls.append(("overlay_only", _shape("overlay", False)))
    if "crude" in wanted:
        calls.append(("crude", _shape("crude", True)))
    if "crude_only" in wanted:
        calls.append(("crude_only", _shape("crude", False)))
    if "shape_only" in wanted:
        calls.append(("shape_only", _shape(shape_engine, False)))
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
        "overlay_esp",
        "overlay_shape",
        "shape",
        "crude_esp",
        "crude_shape",
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
        labels=list(zip(display, labels)),
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
    method: str = "fingerprint",
    kind: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    metric: str = "tanimoto",
    conformers: Optional[int] = None,
    seed: int = 20240101,
    exhaustiveness: int = 8,
    cores: Optional[int] = None,
) -> Dict[str, Any]:
    """What a pre-filter keeps, and what it saves the docking run.

    The realistic use of a ligand-based screen is as a *pre-filter*: score the
    library by similarity to the actives, dock only the top ``keep`` fraction, and
    accept that some actives are thrown away.  This function reports exactly that
    trade: how many molecules survive, **how many actives survive**, and the
    docking workload before and after, using :func:`odock.screen.estimate_cost`
    (imported, never modified) as the cost model.

    ``method`` selects the filter:

    ``"fingerprint"`` (the default) — Morgan/Tanimoto similarity to the actives.
    Free (no conformers), and the shape of the report every other filter matches.

    ``"usr"`` — the 12 USR shape descriptors of :mod:`odock.conformers`, delegated
    to :func:`odock.overlay.usr_prefilter`.  This is the filter that makes a *3-D*
    screen affordable: one conformer ensemble per molecule and microseconds per
    comparison, instead of an overlay against every reference.  Its return value
    carries extra keys — ``overlay_seconds_kept`` (measured), 
    ``overlay_seconds_full_estimate`` (that measurement extrapolated), and
    ``self_match_recall``, the recall the same filter would report if an active
    were allowed to match itself (100 % for free, which is the point of showing it).
    """
    require_rdkit()
    if str(method).strip().lower() in ("usr", "3d", "shape"):
        from .overlay import usr_prefilter

        return usr_prefilter(
            library, actives, keep=keep, conformers=conformers, seed=seed,
            exhaustiveness=exhaustiveness, cores=cores,
        )
    if str(method).strip().lower() not in ("fingerprint", "morgan", "2d"):
        raise ValueError(
            f"unknown pre-filter method {method!r}; use 'fingerprint' (the default) or 'usr'"
        )
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
        "method": "fingerprint",
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


# ---------------------------------------------------------------------------
# More than one target: stratification, and the pooled estimate
# ---------------------------------------------------------------------------


def read_targets(path: Union[str, Path]) -> Dict[str, List[Any]]:
    """Read a ``SMILES name target`` file into ``{target: [molecule, ...]}``.

    The format is the one `demo/actives.smi` ships: three whitespace-separated
    fields per line, comments allowed.  Targets keep the order they first appear in,
    so a report over them is reproducible.
    """
    require_rdkit()

    targets: Dict[str, List[Any]] = {}
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(
                f"{path}:{number}: expected 'SMILES name target', got {line!r}"
            )
        mol = Chem.MolFromSmiles(fields[0])
        if mol is None:
            raise ValueError(f"{path}:{number}: {fields[0]!r} is not a parsable SMILES")
        mol.SetProp("_Name", fields[1])
        targets.setdefault(fields[2], []).append(mol)
    if not targets:
        raise ValueError(f"{path} holds no actives")
    return targets


def _mean_over_strata(values: Sequence[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return (sum(clean) / len(clean)) if clean else float("nan")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    index = int(round(fraction * (len(sorted_values) - 1)))
    return float(sorted_values[max(0, min(len(sorted_values) - 1, index))])


def stratified_difference(
    strata: Sequence[Dict[str, Any]],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    level: float = 0.95,
    seed: int = 20240101,
) -> Dict[str, Any]:
    """The pooled AUC of each of two methods, and the **stratified** difference.

    ``strata`` is one dict per target with ``scores_a``, ``scores_b`` and ``labels``
    in a shared molecule order.  Each bootstrap resample draws molecules *within*
    every stratum and recomputes each stratum's AUC, so the pooled estimate is the
    mean over targets rather than the mean over molecules: a target with 100 decoys
    cannot outvote one with 5, which is the difference between a stratified estimate
    and a pooled one.

    Returns the point estimates and, for the paired difference, its interval, its
    standard error, the minimum detectable difference and whether it is resolvable —
    the same four numbers :func:`paired_difference` reports for a single target.
    """
    usable = [item for item in strata if item["scores_a"] and item["scores_b"]]
    if not usable:
        raise ValueError("no stratum has scores for both methods")
    point_a = _mean_over_strata(
        [float(auc(item["scores_a"], item["labels"])) for item in usable]
    )
    point_b = _mean_over_strata(
        [float(auc(item["scores_b"], item["labels"])) for item in usable]
    )
    rng = random.Random(int(seed))
    deltas: List[float] = []
    pooled_a: List[float] = []
    pooled_b: List[float] = []
    for _ in range(max(1, int(samples))):
        left: List[float] = []
        right: List[float] = []
        for item in usable:
            labels = list(item["labels"])
            n = len(labels)
            picks = [rng.randrange(n) for _ in range(n)]
            resampled = [bool(labels[i]) for i in picks]
            if not any(resampled) or all(resampled):
                continue
            scores_a = [float(item["scores_a"][i]) for i in picks]
            scores_b = [float(item["scores_b"][i]) for i in picks]
            value_a = float(auc(scores_a, resampled))
            value_b = float(auc(scores_b, resampled))
            if math.isfinite(value_a) and math.isfinite(value_b):
                left.append(value_a)
                right.append(value_b)
        if left:
            pooled_a.append(_mean_over_strata(left))
            pooled_b.append(_mean_over_strata(right))
            deltas.append(pooled_a[-1] - pooled_b[-1])
    if not deltas:
        return {
            "auc_a": point_a, "auc_b": point_b, "difference": point_a - point_b,
            "ci": (float("nan"), float("nan")), "se": float("nan"),
            "mdd": float("nan"), "resolvable": False, "samples": 0,
            "n_strata": len(usable),
        }
    deltas.sort()
    tail = (1.0 - float(level)) / 2.0
    low = _percentile(deltas, tail)
    high = _percentile(deltas, 1.0 - tail)
    mean = sum(deltas) / len(deltas)
    variance = sum((value - mean) ** 2 for value in deltas) / max(1, len(deltas) - 1)
    se = math.sqrt(variance)
    return {
        "auc_a": round(point_a, 4),
        "auc_b": round(point_b, 4),
        "difference": round(point_a - point_b, 4),
        "ci": (round(low, 4), round(high, 4)),
        "se": round(se, 4),
        "mdd": round(minimum_detectable_difference(se, level=level), 4),
        "resolvable": bool(low > 0.0 or high < 0.0),
        "samples": len(deltas),
        "n_strata": len(usable),
        "auc_a_ci": (
            round(_percentile(sorted(pooled_a), tail), 4),
            round(_percentile(sorted(pooled_a), 1.0 - tail), 4),
        ),
        "auc_b_ci": (
            round(_percentile(sorted(pooled_b), tail), 4),
            round(_percentile(sorted(pooled_b), 1.0 - tail), 4),
        ),
    }


@dataclass
class StratifiedReport:
    """One benchmark per target, plus the pooled estimate over the targets.

    A single target's quirk drives every conclusion when its molecules are pooled
    with another target's, so the per-target AUCs are kept and the pooled number is
    the mean over targets with an interval bootstrapped **within** them.  Targets
    that could not be given property-matched decoys are listed with the reason
    rather than quietly dropped.
    """

    reports: Dict[str, BenchmarkReport] = field(default_factory=dict)
    methods: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)
    decoys_per_target: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def targets(self) -> List[str]:
        return list(self.reports)

    def _scores(self, report: BenchmarkReport, method: str) -> Tuple[List[float], List[bool]]:
        _, scores, labels = report._aligned(method)
        return scores, labels

    def per_target_table(self) -> str:
        """AUC per method per target, with the pooled (mean-over-targets) column."""
        lines = [
            f"{'method':<20}" + "".join(f"{name[:12]:>14}" for name in self.targets)
            + f"{'pooled':>10}  targets",
            "-" * 20 + "-" * 14 * len(self.targets) + "-" * 10 + "  " + "-" * 7,
        ]
        for method in self.methods:
            values = []
            cells = ""
            for name in self.targets:
                stats = self.reports[name].by_method().get(method)
                value = stats.stats.get("auc", float("nan")) if stats else float("nan")
                values.append(value)
                cells += f"{value:>14.3f}"
            lines.append(
                f"{method[:19]:<20}{cells}{_mean_over_strata(values):>10.3f}"
                f"  {len(values):>7}"
            )
        lines.append("")
        lines.append(
            "the pooled column is the mean over targets, not over molecules: a target "
            "with 100 decoys does not outvote one with 5"
        )
        if self.skipped:
            lines.append("")
            for name, reason in self.skipped.items():
                lines.append(f"  not benchmarked — {name}: {reason}")
        return "\n".join(lines)

    def compare(
        self, method_a: str, method_b: str, *, samples: int = BOOTSTRAP_SAMPLES
    ) -> Dict[str, Any]:
        """The pooled, **stratified and paired** difference between two methods."""
        strata = []
        for name in self.targets:
            report = self.reports[name]
            scores_a, labels = self._scores(report, method_a)
            scores_b, _ = self._scores(report, method_b)
            strata.append({"scores_a": scores_a, "scores_b": scores_b, "labels": labels})
        result = stratified_difference(strata, samples=samples)
        result["method_a"] = method_a
        result["method_b"] = method_b
        return result

    def paired_table(self, *, samples: int = BOOTSTRAP_SAMPLES) -> str:
        """Every pair of methods, pooled across targets, with its MDD."""
        lines = [
            f"{'method A':<18}{'method B':<18}{'AUC A':>8}{'AUC B':>8}{'delta':>8}"
            f"  {'95% CI (stratified)':<22}{'MDD':>7}  resolvable",
            "-" * 18 + "-" * 18 + "-" * 8 * 3 + "  " + "-" * 22 + "-" * 7 + "  " + "-" * 10,
        ]
        for index, left in enumerate(self.methods):
            for right in self.methods[index + 1:]:
                comparison = self.compare(left, right, samples=samples)
                low, high = comparison["ci"]
                lines.append(
                    f"{left[:17]:<18}{right[:17]:<18}{comparison['auc_a']:>8.3f}"
                    f"{comparison['auc_b']:>8.3f}{comparison['difference']:>+8.3f}  "
                    f"[{low:>+6.3f}, {high:>+6.3f}]      {comparison['mdd']:>7.3f}  "
                    + ("yes" if comparison["resolvable"] else "NO")
                )
        lines.append("")
        lines.append(
            f"bootstrapped within {len(self.targets)} target(s), {samples} resamples; "
            "MDD = 2.80 x the paired SE; 'resolvable' = the delta's interval excludes zero"
        )
        return "\n".join(lines)

    def power(self, *, samples: int = BOOTSTRAP_SAMPLES) -> Dict[str, Any]:
        """The pooled minimum detectable difference over every method pair."""
        rows = [
            self.compare(self.methods[i], self.methods[j], samples=samples)
            for i in range(len(self.methods))
            for j in range(i + 1, len(self.methods))
        ]
        mdds = [row["mdd"] for row in rows if math.isfinite(row["mdd"])]
        return {
            "n_targets": len(self.targets),
            "n_actives": sum(report.n_actives for report in self.reports.values()),
            "n_decoys": sum(report.n_decoys for report in self.reports.values()),
            "samples": int(samples),
            "min_mdd": round(min(mdds), 4) if mdds else float("nan"),
            "median_mdd": round(sorted(mdds)[len(mdds) // 2], 4) if mdds else float("nan"),
            "max_mdd": round(max(mdds), 4) if mdds else float("nan"),
            "resolvable_pairs": sum(1 for row in rows if row["resolvable"]),
            "tied_pairs": sum(1 for row in rows if row["mdd"] == 0.0),
            "pairs": len(rows),
            "rows": rows,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "targets": {
                name: {
                    "n_actives": report.n_actives,
                    "n_decoys": report.n_decoys,
                    "auc": {
                        method: report.by_method()[method].stats.get("auc")
                        for method in self.methods
                        if method in report.by_method()
                    },
                    "decoy_quality": report.decoy_quality,
                }
                for name, report in self.reports.items()
            },
            "skipped": dict(self.skipped),
            "power": self.power(),
            "seconds": round(float(self.seconds), 4),
            "notes": list(self.notes),
        }


def benchmark_per_target(
    targets: Dict[str, Sequence[Any]],
    pool: Sequence[Any],
    *,
    per_active: int = 5,
    max_similarity: float = 0.35,
    min_similarity: float = 0.0,
    methods: Sequence[str] = ("fingerprint", "pharmacophore", "shape"),
    shape_engine: str = "overlay",
    conformers: Optional[int] = 4,
    seed: int = 20240101,
    bootstrap: int = BOOTSTRAP_SAMPLES,
    controls: Sequence[str] = ("random", "property"),
    min_actives: int = 2,
    min_decoys: int = 5,
) -> StratifiedReport:
    """Benchmark each target separately, then pool the targets.

    For every target the decoys are selected **from the pool, matched to that
    target's actives** (`odock.decoys.match_decoys`), and any pool member that is an
    active of *any* target in the mapping is removed first: a molecule that is an
    active elsewhere is not a decoy here, and silently scoring it as one is how a
    multi-target benchmark inflates itself.

    A target is skipped, with the reason recorded, when it has fewer than
    ``min_actives`` actives (leave-one-out cannot score a single active) or when the
    pool cannot supply ``min_decoys`` matched decoys for it.  That is the honest
    answer to "can this set be enlarged offline?" — the shortfalls are reported, not
    worked around.
    """
    require_rdkit()
    from .decoys import match_decoys

    start = time.perf_counter()
    all_active_names = {
        _mol_name(mol, f"active_{index + 1}")
        for members in targets.values()
        for index, mol in enumerate(members)
    }
    available = [
        mol for mol in pool if _mol_name(mol, "") not in all_active_names
    ]
    reports: Dict[str, BenchmarkReport] = {}
    skipped: Dict[str, str] = {}
    decoys_per_target: Dict[str, int] = {}
    notes: List[str] = []
    for name, members in targets.items():
        if len(members) < int(min_actives):
            skipped[name] = (
                f"{len(members)} active(s): leave-one-out leaves no reference, so the "
                f"target cannot be scored (needs {int(min_actives)})"
            )
            continue
        selection = match_decoys(
            members, available, per_active=int(per_active),
            max_similarity=float(max_similarity), min_similarity=float(min_similarity),
        )
        chosen = set(selection.names())
        decoys = [mol for mol in available if _mol_name(mol, "") in chosen]
        decoys_per_target[name] = len(decoys)
        if len(decoys) < int(min_decoys):
            skipped[name] = (
                f"{len(decoys)} matched decoy(s) of {int(per_active)} per active "
                f"requested (needs {int(min_decoys)}): the pool has nothing inside the "
                "actives' property tolerances, and an unmatched decoy set is worse "
                "than no set"
            )
            continue
        quality = selection.quality()
        reports[name] = benchmark(
            members, decoys, methods=methods, shape_engine=shape_engine,
            conformers=conformers, seed=seed, bootstrap=bootstrap,
            decoy_quality=quality, controls=controls,
        )
        notes.append(
            f"{name}: {len(members)} active(s), {len(decoys)} matched decoy(s), "
            f"pool match max |SMD| {quality['max_abs_smd']}"
        )
    if not reports:
        raise ValueError(
            "no target could be benchmarked: " + "; ".join(
                f"{name}: {reason}" for name, reason in skipped.items()
            )
        )
    return StratifiedReport(
        reports=reports,
        methods=[result.method for result in next(iter(reports.values())).results],
        skipped=skipped,
        decoys_per_target=decoys_per_target,
        seconds=time.perf_counter() - start,
        notes=notes,
    )

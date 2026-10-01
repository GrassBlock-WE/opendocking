# SPDX-License-Identifier: GPL-3.0-or-later
"""A reproducible re-docking benchmark, and the baseline that guards it.

The project had one anecdote (3PTB at 1.13 Å) and a lot of unit tests.  Unit
tests catch a broken derivative; they cannot catch a scoring change that quietly
makes the *physics* worse.  This module is the thing that can: it re-docks a
small set of bundled crystal complexes through the documented `odock` path and
reports the numbers a docking engine is judged on, in a form that can be
recorded and compared.

What is measured, per system and per seed
-----------------------------------------
``top_rmsd``
    Heavy-atom RMSD of the **top-ranked** pose to the crystallographic pose,
    with **no superposition** (the crystallographic figure of merit: the search
    had to find the experimental pose in the same coordinate frame).
``fitted_rmsd``
    The same pose after optimal superposition, symmetry-aware.
``best_within_1kcal``
    The smallest no-superposition RMSD among the poses within 1 kcal/mol of the
    best score -- the standard "did the search find the right mode at all"
    measure, which is the fair one for a flexible ligand whose near-degenerate
    poses cannot be ranked reliably.
``rank_of_correct``
    1-based rank of the first pose within ``hit_cutoff`` Å (default 2.0, no
    superposition); ``0`` when no pose is that close.
``score_rmsd_rho``
    Spearman correlation between the pose scores and their no-superposition
    RMSD.  **Negative is good**: it means the better-scoring poses are the ones
    closer to the crystal.  Always reported with the number of poses it was
    computed over, because a rho over three poses is not a measurement.
``agreement``
    Mean pairwise Spearman between Vina, Vinardo and AD4 rescoring the *same*
    poses (:mod:`odock.consensus`), with the per-pair values and the number of
    poses.  The per-pair ordering is the finding; the absolute value moves with
    the sample of poses and with the seed (measured: 0.76 on one campaign and
    0.82 on another with identical settings), so this module reports the spread
    across seeds rather than a single number.

Across seeds
------------
Every system is docked with several seeds, and the module reports the spread of
the top affinity, of the top-pose RMSD and of the ranking correlation, whether
the correct pose was found in *every* run, and how stable the top binding-mode
cluster is (:func:`odock.consensus.reproducibility`).  That is the number a user
needs before trusting a screen: an affinity that moves by 1.5 kcal/mol between
seeds does not support a 0.5 kcal/mol ordering.

The baseline
------------
``benchmark/baseline.json`` records the measured values, the exact command and
the settings they were measured with.  ``--check-baseline`` re-runs the
benchmark and fails when a system is worse than the baseline by more than the
recorded tolerance.  The tolerances are **derived from the measured seed
spread** (and stored in the file), not guessed: a threshold tighter than the
instrument's own noise produces a check that cries wolf and gets ignored.

What this does not establish
----------------------------
Read `docs/BENCHMARK.md`: five systems is not CASF, the receptor is rigid, only
one preparation protocol is exercised (a SMILES template from the CCD), and
accuracy on a handful of well-behaved complexes says nothing about a novel
chemotype.  The benchmark is a regression detector and a starting point for
comparison, not a claim of state-of-the-art accuracy.

Download-free
-------------
Every system is defined by data that is in the repository: the crystal file in
``tests/data`` and the ligand's SMILES (taken from the RCSB chemical component
dictionary and recorded here, so a run needs no network).  The SMILES matters:
a ligand prepared from a bare PDB has no bond orders, and the benchmark must
measure the engine, not the perception.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import consensus
from . import docking as _docking

__all__ = [
    "BASELINE_PATH",
    "DEFAULT_SEEDS",
    "SYSTEMS",
    "RunMetrics",
    "System",
    "SystemResult",
    "benchmark",
    "check_baseline",
    "main",
    "records",
    "table",
]

#: Where the committed baseline lives, relative to the repository root.
BASELINE_PATH = Path("benchmark") / "baseline.json"

#: Seeds used when the caller does not choose.  Three is the smallest number
#: that can show a spread rather than a coincidence.
DEFAULT_SEEDS: Tuple[int, ...] = (42, 7, 2024)

#: Poses within this many kcal/mol of the best are used for ``best_within_1kcal``.
ENERGY_WINDOW = 1.0

#: A pose this close to the crystal (heavy atoms, no superposition) counts as
#: having found the experimental binding mode.
HIT_CUTOFF = 2.0

#: Reporting settings for a benchmark run.  The benchmark asks for **more poses
#: than a user would**: a ranking correlation over three poses is not a
#: measurement, and a buried ligand can produce a single pose inside the
#: default 3 kcal/mol window (measured: biotin in streptavidin), which makes
#: both the correlation and the pose clustering undefined.  These settings cost
#: nothing in search time -- they only widen what is reported.
NUM_POSES = 20
ENERGY_RANGE = 5.0
MIN_RMSD = 0.5


@dataclass(frozen=True)
class System:
    """One benchmark complex.

    Attributes
    ----------
    pdb_id
        The RCSB entry, whose ``<ID>.pdb`` lives in ``tests/data``.
    ligand
        The three-letter residue name of the co-crystallised ligand.
    smiles
        The ligand's SMILES, taken from the RCSB chemical component dictionary
        (``<ID>_ideal.sdf``).  Passed to :func:`odock.prepare.prepare_ligand` so
        the prepared ligand has real bond orders: without it the ring systems
        are perceived as saturated and the engine is measured through a
        handicap that has nothing to do with the engine.
    buffer
        Box padding around the crystallographic ligand, in Å.
    exhaustiveness
        Monte-Carlo runs for this system.  Small, stiff ligands get more (they
        are cheap); a 46-heavy-atom ligand gets fewer, because the benchmark has
        to finish.
    note
        One line for the report: what makes this system interesting.
    """

    pdb_id: str
    ligand: str
    smiles: str
    buffer: float = 8.0
    exhaustiveness: int = 8
    note: str = ""


#: The bundled systems, easiest first.  Wall time on the reference machine
#: (16-core Windows, `--fast` in brackets) is in ``docs/BENCHMARK.md``.
SYSTEMS: Tuple[System, ...] = (
    System(
        "3PTB", "BEN", "NC(=N)c1ccccc1",
        buffer=8.0, exhaustiveness=16,
        note="benzamidine in trypsin: tiny and stiff, the acceptance test",
    ),
    System(
        "1STP", "BTN", "O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21",
        buffer=8.0, exhaustiveness=8,
        note="biotin in streptavidin: 16 heavy atoms, five rotors, very buried",
    ),
    System(
        "3ERT", "OHT",
        "CC/C(=C(\\c1ccc(O)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        buffer=6.0, exhaustiveness=6,
        note="4-hydroxytamoxifen in ERalpha: a flexible drug in a large site",
    ),
    System(
        "1M17", "AQ4", "COCCOc1cc2c(cc1OCCOC)ncnc2Nc3cccc(c3)C#C",
        buffer=4.0, exhaustiveness=4,
        note="erlotinib in EGFR: 11 rotors, the documented force-field limit",
    ),
    System(
        "1HVR", "XK2",
        "O=C1N(Cc2ccc3ccccc3c2)[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H]"
        "(Cc2ccccc2)N1Cc1ccc2ccccc2c1",
        buffer=6.0, exhaustiveness=2,
        note="XK263 in HIV-1 protease: 46 heavy atoms, the hardest case here",
    ),
)

_WATER = frozenset(("HOH", "WAT", "DOD"))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """One docking run of one system."""

    seed: int
    elapsed: float = 0.0
    poses: int = 0
    best_affinity: float = float("nan")
    top_rmsd: float = float("nan")
    fitted_rmsd: float = float("nan")
    best_within_1kcal: float = float("nan")
    rank_of_correct: int = 0
    score_rmsd_rho: float = float("nan")
    agreement: float = float("nan")
    correlations: Dict[str, float] = field(default_factory=dict)
    rmsds: List[float] = field(default_factory=list)
    affinities: List[float] = field(default_factory=list)
    clusters: int = 0
    top_cluster_size: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "elapsed": round(self.elapsed, 2),
            "poses": self.poses,
            "best_affinity": _round(self.best_affinity),
            "top_rmsd": _round(self.top_rmsd),
            "fitted_rmsd": _round(self.fitted_rmsd),
            "best_within_1kcal": _round(self.best_within_1kcal),
            "rank_of_correct": self.rank_of_correct,
            "score_rmsd_rho": _round(self.score_rmsd_rho),
            "agreement": _round(self.agreement),
            "correlations": {k: _round(v) for k, v in self.correlations.items()},
            "clusters": self.clusters,
            "top_cluster_size": self.top_cluster_size,
            "rmsds": [_round(v) for v in self.rmsds],
            "affinities": [_round(v) for v in self.affinities],
        }


@dataclass
class SystemResult:
    """Every run of one system, plus the spread across them."""

    system: System
    n_heavy: int = 0
    n_rotatable: int = 0
    receptor_atoms: int = 0
    prep_seconds: float = 0.0
    runs: List[RunMetrics] = field(default_factory=list)
    #: The kernel results themselves, kept for the cross-seed cluster analysis.
    docked: List[Any] = field(default_factory=list, repr=False)
    error: str = ""
    reproducibility: float = float("nan")
    top_cluster_runs: List[int] = field(default_factory=list)
    n_clusters: int = 0

    # -- aggregation -------------------------------------------------------

    def _values(self, attribute: str) -> List[float]:
        return [
            float(getattr(run, attribute))
            for run in self.runs
            if math.isfinite(float(getattr(run, attribute)))
        ]

    @property
    def top_rmsd(self) -> float:
        """The best top-pose RMSD over the seeds (the head-line accuracy)."""
        values = self._values("top_rmsd")
        return min(values) if values else float("nan")

    @property
    def top_rmsd_worst(self) -> float:
        values = self._values("top_rmsd")
        return max(values) if values else float("nan")

    @property
    def top_rmsd_spread(self) -> float:
        return self.top_rmsd_worst - self.top_rmsd

    @property
    def best_within_1kcal(self) -> float:
        values = self._values("best_within_1kcal")
        return min(values) if values else float("nan")

    @property
    def rank_of_correct(self) -> int:
        ranks = [run.rank_of_correct for run in self.runs if run.rank_of_correct]
        return min(ranks) if ranks else 0

    @property
    def affinity_spread(self) -> float:
        values = self._values("best_affinity")
        return (max(values) - min(values)) if values else float("nan")

    @property
    def rho_range(self) -> Tuple[float, float]:
        values = self._values("score_rmsd_rho")
        return (min(values), max(values)) if values else (float("nan"), float("nan"))

    @property
    def rho_mean(self) -> float:
        values = self._values("score_rmsd_rho")
        return statistics.fmean(values) if values else float("nan")

    @property
    def agreement_mean(self) -> float:
        values = self._values("agreement")
        return statistics.fmean(values) if values else float("nan")

    @property
    def agreement_range(self) -> Tuple[float, float]:
        values = self._values("agreement")
        return (min(values), max(values)) if values else (float("nan"), float("nan"))

    @property
    def poses(self) -> int:
        return max((run.poses for run in self.runs), default=0)

    @property
    def correct_in_all_runs(self) -> bool:
        """Whether *every* seed put a pose within the hit cutoff of the crystal."""
        return bool(self.runs) and all(run.rank_of_correct for run in self.runs)

    def pair_order(self) -> List[str]:
        """The force-field pairs ordered weakest-first, pooled over seeds."""
        totals: Dict[str, List[float]] = {}
        for run in self.runs:
            for key, value in run.correlations.items():
                if math.isfinite(value):
                    totals.setdefault(key, []).append(value)
        means = [(statistics.fmean(values), key) for key, values in totals.items()]
        return [key for _mean, key in sorted(means)]

    def pair_means(self) -> Dict[str, float]:
        totals: Dict[str, List[float]] = {}
        for run in self.runs:
            for key, value in run.correlations.items():
                if math.isfinite(value):
                    totals.setdefault(key, []).append(value)
        return {key: statistics.fmean(values) for key, values in totals.items()}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system.pdb_id,
            "ligand": self.system.ligand,
            "n_heavy": self.n_heavy,
            "n_rotatable": self.n_rotatable,
            "receptor_atoms": self.receptor_atoms,
            "prep_seconds": round(self.prep_seconds, 2),
            "exhaustiveness": self.system.exhaustiveness,
            "seeds": [run.seed for run in self.runs],
            "poses": self.poses,
            "top_rmsd": _round(self.top_rmsd),
            "top_rmsd_worst": _round(self.top_rmsd_worst),
            "top_rmsd_spread": _round(self.top_rmsd_spread),
            "fitted_rmsd": _round(min(self._values("fitted_rmsd"), default=float("nan"))),
            "best_within_1kcal": _round(self.best_within_1kcal),
            "rank_of_correct": self.rank_of_correct,
            "affinity_spread": _round(self.affinity_spread),
            "score_rmsd_rho": [_round(v) for v in self.rho_range],
            "agreement": [_round(v) for v in self.agreement_range],
            "pair_means": {k: _round(v) for k, v in self.pair_means().items()},
            "pair_order_weakest_first": self.pair_order(),
            "correct_in_all_runs": self.correct_in_all_runs,
            "reproducibility": _round(self.reproducibility),
            "top_cluster_runs": self.top_cluster_runs,
            "n_clusters": self.n_clusters,
            "error": self.error,
            "runs": [run.as_dict() for run in self.runs],
        }


def _round(value: float, digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Running one system
# ---------------------------------------------------------------------------


def _split_crystal(pdb: Path, ligand_resname: str):
    """``(receptor_mol, ligand_mol)`` out of one crystal file.

    Waters are dropped; the named residue is the ligand and everything else is
    the receptor (including ions and additives, which is what the site contains).
    """
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(
        str(pdb), removeHs=False, sanitize=False, proximityBonding=True
    )
    if mol is None:
        raise RuntimeError(f"RDKit could not read {pdb}")
    keep_receptor: List[int] = []
    keep_ligand: List[int] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip().upper() if info is not None else ""
        if name in _WATER:
            continue
        (keep_ligand if name == ligand_resname.upper() else keep_receptor).append(
            atom.GetIdx()
        )
    if not keep_ligand:
        raise RuntimeError(f"no {ligand_resname} residue in {pdb.name}")

    def subset(indices: Iterable[int]):
        editable = Chem.RWMol(mol)
        for index in sorted(set(range(mol.GetNumAtoms())) - set(indices), reverse=True):
            editable.RemoveAtom(index)
        out = editable.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            pass
        return out

    return subset(keep_receptor), subset(keep_ligand)


def run_system(
    system: System,
    *,
    data_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    exhaustiveness: Optional[int] = None,
    num_poses: int = NUM_POSES,
    energy_range: float = ENERGY_RANGE,
    min_rmsd: float = MIN_RMSD,
    scoring: str = "vina",
    box=None,
    workdir: Optional[Path] = None,
    verbose: bool = True,
) -> SystemResult:
    """Prepare and dock one system with several seeds, and measure the runs.

    Nothing is written unless `workdir` is given (then the prepared PDBQT files
    and the poses of each run are kept there, so a surprising number can be
    investigated).
    """
    import odock
    from rdkit import Chem

    result = SystemResult(system=system)
    pdb = Path(data_dir) / f"{system.pdb_id}.pdb"
    if not pdb.exists():
        result.error = f"missing {pdb}"
        return result

    started = time.perf_counter()
    receptor_mol, ligand_mol_in = _split_crystal(pdb, system.ligand)
    _, receptor_pdbqt, receptor_report = odock.prepare_receptor(
        receptor_mol, None, keep_water=False
    )
    ligand_mol, ligand_pdbqt, ligand_report = odock.prepare_ligand(
        ligand_mol_in, None, name=system.ligand, smiles=system.smiles, optimize=False
    )
    search_box = box if box is not None else odock.box_from_ligand(
        ligand_mol_in, buffer=system.buffer
    )
    result.prep_seconds = time.perf_counter() - started
    result.n_heavy = sum(1 for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1)
    result.n_rotatable = int(ligand_report.n_rotatable_bonds)
    result.receptor_atoms = int(receptor_report.n_atoms_out)

    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / f"{system.pdb_id}_receptor.pdbqt").write_text(
            receptor_pdbqt, encoding="utf-8"
        )
        (workdir / f"{system.pdb_id}_ligand.pdbqt").write_text(
            ligand_pdbqt, encoding="utf-8"
        )

    effort = int(exhaustiveness or system.exhaustiveness)
    reference = Chem.Mol(ligand_mol)
    per_seed: List[RunMetrics] = []
    docked_runs: List[Any] = []
    for seed in seeds:
        metrics = RunMetrics(seed=int(seed))
        try:
            clock = time.perf_counter()
            docked = odock.dock(
                receptor_pdbqt, ligand_pdbqt, search_box, scoring=scoring,
                exhaustiveness=effort, num_poses=int(num_poses), seed=int(seed),
                energy_range=float(energy_range), min_rmsd=float(min_rmsd),
            )
            metrics.elapsed = time.perf_counter() - clock
            metrics.poses = len(docked.poses)
            docked_runs.append(docked)
            if docked.poses:
                metrics.best_affinity = float(docked.best_affinity)
            for pose in docked.poses:
                mol = odock.pose_to_mol(pose, ligand_mol, ligand_report.atom_order)
                raw, fitted = odock.aligned_rmsd(mol, reference, heavy_only=True)
                metrics.rmsds.append(float(raw))
                metrics.affinities.append(float(pose.affinity))
                if pose.index == 0:
                    metrics.top_rmsd = float(raw)
                    metrics.fitted_rmsd = float(fitted)
            if metrics.rmsds:
                best = min(metrics.affinities)
                within = [
                    rmsd for rmsd, affinity in zip(metrics.rmsds, metrics.affinities)
                    if affinity <= best + ENERGY_WINDOW
                ]
                metrics.best_within_1kcal = min(within)
                close = [i for i, rmsd in enumerate(metrics.rmsds) if rmsd <= HIT_CUTOFF]
                metrics.rank_of_correct = (close[0] + 1) if close else 0
                metrics.score_rmsd_rho = consensus.spearman(
                    metrics.affinities, metrics.rmsds
                )
                clusters = consensus_clusters(
                    docked, ligand_report.atom_order, ligand_mol
                )
                if clusters:
                    metrics.clusters = len(clusters)
                    metrics.top_cluster_size = len(clusters[0].members)
            # The three force fields, on exactly these poses.
            models = consensus.pdbqt_models(docked.to_pdbqt() or "")
            if models:
                combined = consensus.consensus_score(
                    models, receptor_pdbqt, search_box
                )
                metrics.agreement = combined.agreement
                metrics.correlations = {
                    f"{a}~{b}": rho for a, b, rho in combined.correlations
                }
        except Exception as exc:  # pragma: no cover - reported, not raised
            result.error = f"seed {seed}: {type(exc).__name__}: {exc}"
        per_seed.append(metrics)
        if verbose:
            print(
                f"  {system.pdb_id} seed {seed}: top {metrics.top_rmsd:.2f} A, "
                f"best<=1 {metrics.best_within_1kcal:.2f} A, rank {metrics.rank_of_correct}, "
                f"rho {metrics.score_rmsd_rho:+.2f}, agree {metrics.agreement:+.2f}, "
                f"{metrics.poses} poses, {metrics.elapsed:.1f}s",
                flush=True,
            )

    result.runs = per_seed
    result.docked = docked_runs
    # Cross-seed binding-mode stability, measured properly: pool the poses of
    # every seed, cluster them by symmetry-aware RMSD, and ask how many seeds
    # contributed to the cluster holding the overall best pose.
    if len(docked_runs) >= 2:
        try:
            labels = [atom.GetSymbol() for atom in ligand_mol.GetAtoms()]
            report = consensus.reproducibility(
                docked_runs, cutoff=HIT_CUTOFF, elements=labels
            )
            result.reproducibility = report.reproducibility
            result.top_cluster_runs = list(report.top_cluster_runs)
            result.n_clusters = report.n_clusters
        except Exception as exc:  # pragma: no cover - reported, not raised
            result.error = (result.error + "; " if result.error else "") + (
                f"reproducibility: {type(exc).__name__}: {exc}"
            )
    else:
        # One seed cannot show cluster stability; fall back to the fraction of
        # runs that found the crystal pose at all, so the field is not silently
        # meaningless.
        result.reproducibility = _top_cluster_stability(per_seed)
    return result


def consensus_clusters(docked, atom_order, ligand_mol):
    """Cluster the poses of one run by symmetry-aware RMSD (no superposition)."""
    from . import analysis

    if not docked.poses:
        return []
    import numpy as np

    coords = []
    for pose in docked.poses:
        mol = _docking.pose_to_mol(pose, ligand_mol, atom_order)
        conformer = mol.GetConformer()
        coords.append(
            np.array(
                [
                    [conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y,
                     conformer.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())
                ],
                dtype=float,
            )
        )
    labels = [atom.GetSymbol() for atom in ligand_mol.GetAtoms()]
    return analysis.cluster_poses(
        coords, cutoff=HIT_CUTOFF, elements=labels,
        energies=[pose.affinity for pose in docked.poses],
    )


def _top_cluster_stability(runs: Sequence[RunMetrics]) -> float:
    """Fallback: the fraction of seeds that found *any* pose at the crystal.

    Used only when the raw results were not kept (a caller that re-analyses a
    saved JSON), or when there is a single seed.  :func:`run_system` uses
    :func:`odock.consensus.reproducibility` instead, which clusters the pooled
    poses of every seed and is the real measure.
    """
    usable = list(runs)
    if len(usable) < 2:
        return float("nan")
    agreed = [run for run in usable if run.rank_of_correct]
    return len(agreed) / len(usable)


# ---------------------------------------------------------------------------
# The whole benchmark
# ---------------------------------------------------------------------------


def benchmark(
    systems: Optional[Sequence[System]] = None,
    *,
    data_dir: Optional[Path] = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    exhaustiveness: Optional[int] = None,
    fast: bool = False,
    num_poses: int = NUM_POSES,
    energy_range: float = ENERGY_RANGE,
    min_rmsd: float = MIN_RMSD,
    workdir: Optional[Path] = None,
    verbose: bool = True,
) -> List[SystemResult]:
    """Run every system and return the measured results in order."""
    root = Path(__file__).resolve().parent.parent.parent
    data = Path(data_dir) if data_dir is not None else root / "tests" / "data"
    chosen = list(systems if systems is not None else SYSTEMS)
    results: List[SystemResult] = []
    for system in chosen:
        effort = exhaustiveness
        if fast and effort is None:
            effort = max(1, system.exhaustiveness // 2)
        if verbose:
            print(
                f"=== {system.pdb_id} ({system.ligand}): {system.note} "
                f"[exhaustiveness {effort or system.exhaustiveness}]",
                flush=True,
            )
        outcome = run_system(
            system, data_dir=data, seeds=seeds, exhaustiveness=effort,
            num_poses=num_poses, energy_range=energy_range, min_rmsd=min_rmsd,
            workdir=workdir, verbose=verbose,
        )
        if outcome.error and verbose:
            print(f"  !! {outcome.error}", flush=True)
        results.append(outcome)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def table(results: Sequence[SystemResult]) -> str:
    """The human-readable table: one row per system, spread across seeds."""
    header = (
        f"{'system':7s} {'ligand':6s} {'heavy':>5s} {'tors':>4s} {'poses':>5s} "
        f"{'topRMSD':>8s} {'spread':>6s} {'best<=1':>7s} {'rank':>4s} "
        f"{'rho (n)':>14s} {'rho range':>12s} {'agree range':>12s} "
        f"{'weakest pair':>18s} {'all':>4s} {'sec':>6s}"
    )
    lines = [header, "-" * len(header)]
    for outcome in results:
        if outcome.error and not outcome.runs:
            lines.append(f"{outcome.system.pdb_id:7s} FAILED: {outcome.error}")
            continue
        low, high = outcome.rho_range
        agree_low, agree_high = outcome.agreement_range
        order = outcome.pair_order()
        weakest = order[0] if order else "-"
        seconds = outcome.prep_seconds + sum(run.elapsed for run in outcome.runs)
        rho_text = f"{low:+.2f}..{high:+.2f}" if math.isfinite(low) else "--"
        mean_rho = outcome.rho_mean
        rho_cell = (
            f"{mean_rho:+.2f} (n={outcome.poses})" if math.isfinite(mean_rho) else "--"
        )
        agree_text = (
            f"{agree_low:+.2f}..{agree_high:+.2f}"
            if math.isfinite(agree_low) else "--"
        )
        lines.append(
            f"{outcome.system.pdb_id:7s} {outcome.system.ligand:6s} "
            f"{outcome.n_heavy:5d} {outcome.n_rotatable:4d} {outcome.poses:5d} "
            f"{outcome.top_rmsd:8.2f} {outcome.top_rmsd_spread:6.2f} "
            f"{outcome.best_within_1kcal:7.2f} {outcome.rank_of_correct:4d} "
            f"{rho_cell:>14s} {rho_text:>12s} {agree_text:>12s} {weakest:>18s} "
            f"{'yes' if outcome.correct_in_all_runs else 'no':>4s} {seconds:6.1f}"
        )
    lines.append("")
    lines.append(
        "topRMSD/best<=1: heavy-atom RMSD to the crystal, no superposition (A)."
    )
    lines.append(
        "rho: Spearman(score, RMSD) over the poses of one run; negative is good. "
        "The range is across seeds."
    )
    lines.append(
        "agree: mean pairwise Spearman between Vina/Vinardo/AD4 on the same poses; "
        "the range is across seeds, n is the pose count."
    )
    return "\n".join(lines)


def records(results: Sequence[SystemResult]) -> List[Dict[str, Any]]:
    """Flat rows for JSON/CSV, one per system (with the per-seed runs nested)."""
    return [outcome.as_dict() for outcome in results]


def write_json(path: Path, payload: Any) -> str:
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                      encoding="utf-8")
    return str(target)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    """One row per system, the nested fields flattened into JSON strings."""
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    flat: List[Dict[str, Any]] = []
    for row in rows:
        item = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                item[key] = json.dumps(value, sort_keys=True)
            else:
                item[key] = value
        flat.append(item)
    columns: List[str] = []
    for row in flat:
        for key in row:
            if key not in columns:
                columns.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in flat:
            writer.writerow(row)
    return str(target)


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


def environment() -> Dict[str, Any]:
    """What the numbers were measured on, so a difference can be interpreted."""
    import odock

    return {
        "odock": odock.__version__,
        "kernel": odock.kernel_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
    }


def portable_command(text: str) -> str:
    """A command string that is safe to publish.

    The recorded command is documentation, and a baseline that ships has to be
    reproducible by someone who did not run it here: an absolute checkout path or
    interpreter path in a published file leaks the developer's machine layout and
    points a reader at a directory that does not exist for them.

    Three things are normalised: this checkout's root becomes ``.`` (so
    ``--json <root>/out/b.json`` is recorded as ``--json out/b.json``), a leading
    interpreter path becomes ``python`` whatever it was
    (``C:\\work\\repo\\.venv\\Scripts\\python.exe -m odock.benchmark`` becomes
    ``python -m odock.benchmark``), and the whitespace is collapsed.  A path that
    is *not* this checkout's root and is not the interpreter is left alone: it is
    the caller's own argument, and rewriting it would misreport the command.
    """
    cleaned = " ".join(str(text).split())
    root = str(Path(__file__).resolve().parent.parent.parent)
    for prefix in (root + os.sep, root + "/"):
        cleaned = cleaned.replace(prefix, "")
    cleaned = cleaned.replace(root, ".")
    for interpreter in (sys.executable, sys.executable.replace(os.sep, "/")):
        if interpreter:
            cleaned = cleaned.replace(interpreter, "python")
    tokens = cleaned.split(" ")
    if tokens:
        first = tokens[0].lower()
        if first.endswith("benchmark.py"):
            # A run recorded as `<checkout>/python/odock/benchmark.py ...` (which
            # is what `sys.argv[0]` is under `python -m`) becomes the module form,
            # so the recorded command can actually be copied and run.
            tokens[0] = "python -m odock.benchmark"
        elif "python" in first and (os.sep in tokens[0] or "/" in tokens[0]):
            tokens[0] = "python"
    return " ".join(tokens)


def cli_command(argv: Optional[Sequence[str]] = None) -> str:
    """The reproducible form of a benchmark invocation, without local paths."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    return portable_command("python -m odock.benchmark " + " ".join(arguments)).strip()


#: Tolerance floor for the baseline check, in Å.  A repeat run of the same
#: command with the same seed is **bit-for-bit identical** on a given machine
#: (measured to six decimals on 3PTB and 1STP), so the tolerance only has to
#: absorb cross-machine differences -- a different core count reorders the
#: parallel reduction, which can flip a tie.  0.15 Å is five times below the
#: experimental uncertainty of the crystals themselves and far below any
#: regression worth catching.
REPEATABILITY_FLOOR = 0.15

#: Tolerance on the ranking correlation, for the same reason.  Measured
#: repeat-run variation: 0.000.
RHO_FLOOR = 0.2


def baseline_from(
    results: Sequence[SystemResult],
    *,
    tolerance: Optional[Dict[str, float]] = None,
    repeatability: Optional[Dict[str, float]] = None,
    command: str = "",
) -> Dict[str, Any]:
    """The baseline document for a measured run (see :func:`baseline_document`)."""
    return baseline_document(
        records(results), tolerance=tolerance, repeatability=repeatability,
        command=command, results=results,
    )


def baseline_document(
    rows: Sequence[Dict[str, Any]],
    *,
    tolerance: Optional[Dict[str, float]] = None,
    repeatability: Optional[Dict[str, float]] = None,
    command: str = "",
    results: Optional[Sequence[SystemResult]] = None,
) -> Dict[str, Any]:
    """The baseline document: measured values, tolerances and provenance.

    Works from the recorded rows, so a baseline can be re-cut from a saved
    benchmark JSON without re-running the docking.

    **Two different spreads are involved, and confusing them is how a
    regression check becomes useless.**  The check re-runs the *same command*
    with the *same seeds*, so what it has to absorb is **repeat-run noise** --
    measured as exactly zero on this machine (two runs of 3PTB and of 1STP agree
    to six decimals), floored at :data:`REPEATABILITY_FLOOR` for cross-machine
    differences.  The **seed-to-seed spread** is a different quantity and a much
    larger one (up to 3.17 Å for 1M17): it says how much the *system* moves when
    the search restarts, so it is recorded per system as ``top_rmsd_spread`` and
    reported, but it must never be the tolerance -- a 4.8 Å allowance would let a
    real regression through.
    """
    spreads = [
        float(row.get("top_rmsd_spread") or 0.0)
        for row in rows
        if row.get("top_rmsd_spread") is not None
    ]
    largest_seed_spread = max(spreads) if spreads else 0.0
    measured = dict(repeatability or {})
    default = {
        "top_rmsd": round(max(REPEATABILITY_FLOOR, float(measured.get("top_rmsd", 0.0))), 3),
        "best_within_1kcal": round(
            max(REPEATABILITY_FLOOR, float(measured.get("best_within_1kcal", 0.0))), 3
        ),
        "rank_of_correct": 1,
        "score_rmsd_rho": round(max(RHO_FLOOR, float(measured.get("score_rmsd_rho", 0.0))), 3),
    }
    if tolerance:
        default.update({key: float(value) for key, value in tolerance.items()})
    default["derivation"] = (
        "tolerance = repeat-run noise of the same command (measured 0.000 on "
        "this machine, floored at "
        f"{REPEATABILITY_FLOOR} A); the seed-to-seed spread is a different, "
        "larger quantity (largest measured here: "
        f"{largest_seed_spread:.2f} A) and is recorded per system as "
        "top_rmsd_spread, not used as a tolerance"
    )
    if results:
        seed_list = [run.seed for run in results[0].runs]
    else:
        seed_list = list(rows[0].get("seeds") or []) if rows else []
    document = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": command,
        "environment": environment(),
        "settings": {
            "seeds": list(seed_list),
            "num_poses": NUM_POSES,
            "energy_range": ENERGY_RANGE,
            "min_rmsd": MIN_RMSD,
            "energy_window": ENERGY_WINDOW,
            "hit_cutoff": HIT_CUTOFF,
        },
        "tolerance": default,
        "systems": {row["system"]: row for row in rows},
    }
    # Sanitised at the boundary as well: a baseline re-cut from an older run
    # inherits whatever command that run recorded, which may be an absolute path.
    document["command"] = portable_command(document.get("command", ""))
    return document


def check_baseline(
    results: Sequence[SystemResult],
    baseline: Dict[str, Any],
) -> List[str]:
    """Compare a run against a baseline; returns the failures (empty is good)."""
    tolerance = dict(baseline.get("tolerance") or {})
    stored = dict(baseline.get("systems") or {})
    failures: List[str] = []
    for outcome in results:
        reference = stored.get(outcome.system.pdb_id)
        if reference is None:
            continue
        if outcome.error and not outcome.runs:
            failures.append(f"{outcome.system.pdb_id}: the run failed ({outcome.error})")
            continue
        checks = (
            ("top_rmsd", outcome.top_rmsd, float(tolerance.get("top_rmsd", 0.35)), "A"),
            ("best_within_1kcal", outcome.best_within_1kcal,
             float(tolerance.get("best_within_1kcal", 0.35)), "A"),
        )
        for name, got, allowed, unit in checks:
            want = reference.get(name)
            if want is None or not math.isfinite(got):
                continue
            if got > float(want) + allowed:
                failures.append(
                    f"{outcome.system.pdb_id}: {name} {got:.2f} {unit} is worse than "
                    f"the baseline {float(want):.2f} (+{allowed:.2f} allowed)"
                )
        want_rho = reference.get("score_rmsd_rho")
        if isinstance(want_rho, list) and len(want_rho) == 2:
            allowed = float(tolerance.get("score_rmsd_rho", 0.5))
            # Both sides are [worst, best] over the seeds; "negative is good", so
            # the worst correlation is the *larger* number and that is the side
            # that has to stay within the allowance.
            if math.isfinite(outcome.rho_range[1]) and math.isfinite(float(want_rho[1])):
                if outcome.rho_range[1] > float(want_rho[1]) + allowed:
                    failures.append(
                        f"{outcome.system.pdb_id}: score_rmsd_rho worst {outcome.rho_range[1]:+.2f} "
                        f"is worse than the baseline {float(want_rho[1]):+.2f} "
                        f"(+{allowed:.2f} allowed)"
                    )
        if reference.get("correct_in_all_runs") and not outcome.correct_in_all_runs:
            failures.append(
                f"{outcome.system.pdb_id}: the correct pose was found in every "
                "baseline run but not in this one"
            )
    return failures


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m odock.benchmark",
        description=(
            "Re-dock the bundled crystal complexes and report accuracy, ranking "
            "correlation and force-field agreement, with the spread across seeds."
        ),
    )
    parser.add_argument("--systems", default="",
                        help="comma-separated PDB IDs (default: all)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="comma-separated seeds (default: 42,7,2024)")
    parser.add_argument("-e", "--exhaustiveness", type=int, default=None,
                        help="override every system's search effort")
    parser.add_argument("--fast", action="store_true",
                        help="halve each system's exhaustiveness")
    parser.add_argument("-n", "--num-poses", type=int, default=NUM_POSES,
                        help=f"poses kept per run (default {NUM_POSES})")
    parser.add_argument("--energy-range", type=float, default=ENERGY_RANGE,
                        help=f"reporting window in kcal/mol (default {ENERGY_RANGE})")
    parser.add_argument("--min-rmsd", type=float, default=MIN_RMSD,
                        help=f"pose deduplication cutoff (default {MIN_RMSD})")
    parser.add_argument("--data-dir", default=None,
                        help="where the <PDB_ID>.pdb files live (default tests/data)")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the full results here")
    parser.add_argument("--csv", dest="csv_out", default=None,
                        help="write one flat row per system here")
    parser.add_argument("--workdir", default=None,
                        help="keep the prepared PDBQT files and poses here")
    parser.add_argument("--baseline", default=str(BASELINE_PATH),
                        help=f"baseline file (default {BASELINE_PATH})")
    parser.add_argument("--write-baseline", action="store_true",
                        help="record this run as the baseline")
    parser.add_argument("--baseline-from", default=None,
                        help="write the baseline from a saved --json run instead of "
                             "re-docking")
    parser.add_argument("--check-baseline", action="store_true",
                        help="fail when a system is worse than the baseline")
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent.parent
    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path

    if args.baseline_from:
        # Re-cut the baseline from a recorded run: the values are the ones that
        # were measured, so nobody has to re-dock for twenty minutes to change a
        # tolerance.
        payload = json.loads(Path(args.baseline_from).read_text(encoding="utf-8"))
        rows = payload.get("systems") if isinstance(payload, dict) else payload
        if not isinstance(rows, list) or not rows:
            print(f"{args.baseline_from} holds no systems", file=sys.stderr)
            return 2
        recorded = str(payload.get("command") or "") if isinstance(payload, dict) else ""
        document = baseline_document(
            rows,
            command=(recorded + " | baseline re-cut with --baseline-from").strip(" |"),
        )
        print(f"wrote {write_json(baseline_path, document)}")
        return 0

    chosen = None
    if args.systems.strip():
        wanted = {name.strip().upper() for name in args.systems.split(",") if name.strip()}
        chosen = [system for system in SYSTEMS if system.pdb_id.upper() in wanted]
        missing = wanted - {system.pdb_id.upper() for system in chosen}
        if missing:
            print(f"unknown system(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    seeds = tuple(int(x) for x in str(args.seeds).split(",") if x.strip())
    results = benchmark(
        chosen,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        seeds=seeds,
        exhaustiveness=args.exhaustiveness,
        fast=args.fast,
        num_poses=int(args.num_poses),
        energy_range=float(args.energy_range),
        min_rmsd=float(args.min_rmsd),
        workdir=Path(args.workdir) if args.workdir else None,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print()
    print(table(results))

    if args.json_out:
        payload = {
            "command": cli_command(),
            "environment": environment(),
            "settings": {
                "seeds": list(seeds),
                "fast": bool(args.fast),
                "exhaustiveness": args.exhaustiveness,
            },
            "systems": records(results),
        }
        print(f"\nwrote {write_json(Path(args.json_out), payload)}")
    if args.csv_out:
        print(f"wrote {write_csv(Path(args.csv_out), records(results))}")

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = root / baseline_path
    if args.write_baseline:
        document = baseline_from(results, command=cli_command())
        print(f"\nwrote {write_json(baseline_path, document)}")
    if args.check_baseline:
        if not baseline_path.exists():
            print(f"\nno baseline at {baseline_path}", file=sys.stderr)
            return 2
        document = json.loads(baseline_path.read_text(encoding="utf-8"))
        stored_seeds = list((document.get("settings") or {}).get("seeds") or [])
        if stored_seeds and stored_seeds != list(seeds):
            print(
                f"warning: the baseline was measured with seeds {stored_seeds} and "
                f"this run uses {list(seeds)}; the two are not like for like "
                "(a best-over-seeds only improves with more seeds), so read the "
                "deltas with that in mind",
                file=sys.stderr,
            )
        failures = check_baseline(results, document)
        print()
        if failures:
            print("BASELINE CHECK FAILED")
            for line in failures:
                print(f"  - {line}")
            return 1
        compared = ", ".join(sorted(document.get("systems") or {}))
        print(f"baseline check passed ({compared})")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

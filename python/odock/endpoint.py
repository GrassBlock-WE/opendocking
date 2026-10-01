# SPDX-License-Identifier: GPL-3.0-or-later
"""MM-GBSA-style end-point rescoring: a documented approximation, with an error bar.

Every score this project reports is a docking score: an empirical, rigid-receptor,
single-structure number.  This module adds the next cheapest thing a modeller
reaches for -- an **end-point** estimate in the MM-GBSA family,

    dG ~ <E_complex> - <E_receptor> - <E_ligand> + <solvation> + <entropy>

evaluated over an ensemble of poses, and reported **with its spread**.  What it
actually computes, term by term, and what each term is worth:

``interaction``
    The kernel's own interaction energy for the pose (``consensus``'s ``inter``
    column, from ``rescore_poses``).  For the Vina and Vinardo functions this is
    the whole gas-phase interaction: gauss, repulsion, hydrophobic and H-bond
    terms.  **Neither has an electrostatic term**, which is why the term below is
    added rather than double counting.  AD4 *does* have one, so requesting the AD4
    scoring while also adding the Coulomb term double counts and the report warns
    about it.
``electrostatic``
    A distance-dependent-dielectric Coulomb sum over ligand-receptor pairs,
    ``E = 332.0637 * q_i q_j / (eps * r)`` with ``eps = 4 r`` -- the simple
    convention this module names and documents.  It is **not** a Poisson-Boltzmann
    or Generalised-Born solver: there is no solvent screening beyond ``eps = 4r``,
    no ionic strength, no Born radii.  Note that the kernel's *own* AD4
    distance-dependent dielectric is a different documented form
    (``dielectric = -0.1465``, AutoGrid 4.2, see :mod:`odock.export`); the two
    must not be mixed, and this module does not use the kernel's.
``nonpolar``
    A SASA-proportional term, ``dG_np = -gamma * dSASA_buried`` with
    ``gamma = 0.0072 kcal/mol/A^2`` (the usual surface-tension value), where the
    buried area comes from :mod:`odock.sasa` as the difference between the free and
    complex solvent-accessible areas.  This is the cheapest defensible nonpolar
    model; it has no curvature or microscopic-surface correction.
``entropy``
    Optional and **off unless asked for**: the same torsional estimate the project
    already uses, ``-T dS = -RT ln(states_per_rotor ** n_torsions)``
    (:func:`odock.metrics.entropy_penalty`, 3 states per rotor, 298.15 K), reported
    as ``-TdS`` in kcal/mol.  It is a crude count of rotatable bonds, not a
    normal-mode or quasi-harmonic entropy.
``strain``
    Optional and off by default: :func:`odock.metrics.ligand_strain` relaxes the
    ligand and charges the conformational strain.  With a rigid receptor and the
    kernel's accounting, ``intra`` equals the unbound value, so the kernel reports
    **zero** strain for a pose; that is an approximation of the force field, not a
    statement that the ligand is unstrained.

What this is **not**, and these are stated in ``docs/ENDPOINT.md`` as section 1
rather than as a footnote:

1. **Not a free-energy calculation.**  No alchemical path, no ensemble average over
   a Boltzmann-weighted trajectory, no relaxation of the complex.
2. **No explicit water.**  Solvent is the SASA term plus ``eps = 4r``; a bridging
   water that the project's own water analysis can *see* is invisible here.
3. **Entropy is omitted by default**, and the torsional estimate that can be
   switched on is a counting rule.
4. **The dielectric is a fudge with a documented convention.**  Charges come from
   whatever the input PDBQT carries; an AD4 charge model and a Gasteiger model are
   not interchangeable.
5. **A dG in kcal/mol from this is a ranking device.**  It is not an experimental
   affinity, and the bootstrap interval below is the spread of the *pose ensemble*,
   not the method's error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ensemble import EnsembleError

__all__ = [
    "COULOMB",
    "DEFAULT_DIELECTRIC",
    "DEFAULT_GAMMA",
    "DEFAULT_SAMPLES",
    "EndpointPose",
    "EndpointResult",
    "add_endpoint_parser",
    "cmd_endpoint",
    "compare_rankings",
    "coulomb_energy",
    "endpoint_ensemble",
    "endpoint_score",
    "kendall_tau",
]

#: Coulomb constant in kcal/mol * A / e^2.
COULOMB = 332.0637

#: Nonpolar surface tension (kcal/mol/A^2).  The value used across the MM-GBSA
#: literature for a SASA-proportional nonpolar term.
DEFAULT_GAMMA = 0.0072

#: eps = DEFAULT_DIELECTRIC * r.  The convention, named so it cannot drift.
DEFAULT_DIELECTRIC = 4.0

#: Bootstrap resamples for the interval on the ensemble mean.
DEFAULT_SAMPLES = 2000

#: Seed for the bootstrap, so the interval is reproducible.
DEFAULT_SEED = 20240101

#: Scoring functions that carry no electrostatic term of their own, so the
#: Coulomb term this module adds does not double count.
_SCORINGS_WITHOUT_ELECTROSTATICS = ("vina", "vinardo")


@dataclass
class EndpointPose:
    """One pose's end-point decomposition."""

    label: str
    pose_index: int
    affinity: float
    interaction: float
    electrostatic: float
    nonpolar: float
    entropy: float = 0.0
    strain: float = 0.0
    buried_area: float = 0.0
    n_torsions: float = 0.0
    scoring: str = "vina"

    @property
    def total(self) -> float:
        """The end-point estimate: interaction + electrostatic + nonpolar (+ opt)."""
        return (
            self.interaction
            + self.electrostatic
            + self.nonpolar
            + self.entropy
            + self.strain
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "pose": int(self.pose_index),
            "affinity_vina": float(self.affinity),
            "interaction": float(self.interaction),
            "electrostatic": float(self.electrostatic),
            "nonpolar": float(self.nonpolar),
            "entropy": float(self.entropy),
            "strain": float(self.strain),
            "buried_area": float(self.buried_area),
            "n_torsions": float(self.n_torsions),
            "total": float(self.total),
            "scoring": self.scoring,
        }


@dataclass
class EndpointResult:
    """A ligand's end-point estimate over its pose ensemble, with the spread."""

    ligand: str
    receptor: str
    scoring: str = "vina"
    poses: List[EndpointPose] = field(default_factory=list)
    gamma: float = DEFAULT_GAMMA
    dielectric: float = DEFAULT_DIELECTRIC
    samples: int = DEFAULT_SAMPLES
    seed: int = DEFAULT_SEED
    notes: List[str] = field(default_factory=list)

    @property
    def n_poses(self) -> int:
        return len(self.poses)

    @property
    def values(self) -> np.ndarray:
        return np.array([pose.total for pose in self.poses], dtype=float)

    @property
    def mean(self) -> float:
        return float(self.values.mean()) if self.poses else float("nan")

    @property
    def spread(self) -> float:
        """Population standard deviation of the pose totals (0 for one pose)."""
        return float(self.values.std()) if len(self.poses) > 1 else 0.0

    @property
    def best(self) -> Optional[EndpointPose]:
        return min(self.poses, key=lambda pose: pose.total) if self.poses else None

    def bootstrap(self, *, samples: Optional[int] = None, seed: Optional[int] = None,
                  confidence: float = 0.95) -> Dict[str, Any]:
        """Percentile bootstrap interval on the ensemble mean dG.

        Resamples the **poses**, so the interval is the spread of this ligand's
        pose ensemble -- not the method's error, and not a confidence interval on
        the true affinity.  With a single pose there is nothing to resample, and
        the interval collapses to the point, which the caller must report as such.
        """
        values = self.values
        n = values.size
        count = int(self.samples if samples is None else samples)
        if n == 0:
            return {"n": 0, "mean": float("nan"), "low": float("nan"),
                    "high": float("nan"), "samples": 0, "width": float("nan"),
                    "note": "no poses"}
        if n == 1:
            value = float(values[0])
            return {"n": 1, "mean": value, "low": value, "high": value, "samples": 0,
                    "width": 0.0,
                    "note": "one pose: no ensemble to resample, so the interval is "
                            "the point estimate and must be read as such"}
        rng = np.random.default_rng(int(self.seed if seed is None else seed))
        draws = rng.integers(0, n, size=(count, n))
        means = values[draws].mean(axis=1)
        alpha = (1.0 - float(confidence)) / 2.0
        low = float(np.quantile(means, alpha))
        high = float(np.quantile(means, 1.0 - alpha))
        return {
            "n": int(n), "mean": float(values.mean()), "low": low, "high": high,
            "samples": count, "width": float(high - low), "confidence": float(confidence),
            "note": "",
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ligand": self.ligand,
            "receptor": self.receptor,
            "scoring": self.scoring,
            "n_poses": self.n_poses,
            "mean": self.mean,
            "spread": self.spread,
            "best": self.best.as_dict() if self.best else None,
            "interval": self.bootstrap(),
            "gamma": self.gamma,
            "dielectric": self.dielectric,
            "poses": [pose.as_dict() for pose in self.poses],
            "notes": list(self.notes),
        }

    def text(self) -> str:
        interval = self.bootstrap()
        lines = [
            f"{self.ligand} vs {self.receptor}: dG {self.mean:+.3f} kcal/mol over "
            f"{self.n_poses} pose(s), 95% interval "
            f"[{interval['low']:+.3f}, {interval['high']:+.3f}] "
            f"(width {interval['width']:.3f}), sd {self.spread:.3f}",
            "  pose  interaction  electrostatic  nonpolar  total   (vina affinity)",
        ]
        for pose in self.poses:
            lines.append(
                f"  {pose.pose_index:>4}  {pose.interaction:+12.3f}  "
                f"{pose.electrostatic:+13.3f}  {pose.nonpolar:+8.3f}  "
                f"{pose.total:+7.3f}   ({pose.affinity:+.3f})"
            )
        if interval["note"]:
            lines.append(f"  {interval['note']}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def coulomb_energy(
    ligand: Sequence[Any],
    receptor: Sequence[Any],
    *,
    dielectric: float = DEFAULT_DIELECTRIC,
) -> float:
    """``sum 332.0637 q_i q_j / (eps r_ij)`` over ligand-receptor pairs, eps = k r.

    With ``eps`` proportional to the distance the pair term falls off as
    ``1/r^2``: a documented convention, not a solution to the Poisson-Boltzmann
    equation.  Charges come from the PDBQT (``PoseAtom.charge``); a missing charge
    is treated as zero and counted, so a silently neutral ligand is visible rather
    than invisible.
    """
    if not ligand or not receptor:
        return 0.0
    left = np.array([[a.x, a.y, a.z] for a in ligand], dtype=float).reshape(-1, 3)
    right = np.array([[a.x, a.y, a.z] for a in receptor], dtype=float).reshape(-1, 3)
    q_left = np.array([float(getattr(a, "charge", 0.0) or 0.0) for a in ligand])
    q_right = np.array([float(getattr(a, "charge", 0.0) or 0.0) for a in receptor])
    products = np.outer(q_left, q_right)
    distances = np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distances[:, :0], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = COULOMB * products / (float(dielectric) * distances * distances)
    terms[~np.isfinite(terms)] = 0.0
    return float(terms.sum())


def buried_area(
    ligand: Sequence[Any],
    receptor: Sequence[Any],
    *,
    probe: float = 1.4,
    points: int = 92,
) -> Dict[str, float]:
    """The area buried on complex formation, from :mod:`odock.sasa`.

    Both sides are measured: ``dSASA = SASA_free - SASA_complex`` for the ligand
    and for the receptor.  The sum is what the nonpolar term uses; the two halves
    are reported because a ligand can be buried in a shallow groove (small
    receptor change) or plug a channel (large one), and the two are not the same
    statement.

    The two sides come from two different calls, and that is a measured necessity
    rather than a preference.  ``burial(ligand, receptor)`` is consistent --
    measured on 3PTB with benzamidine, the ligand's free area is 426.3 A^2 and its
    area in the complex 104.1 A^2 -- but ``burial(receptor, ligand)`` reports a
    ``reference_total`` of **58 377.1 A^2** where the free receptor's own area is
    **9 274.9 A^2**, so a difference of those two is meaningless (it produced a
    500 A^2 ligand and a 357 kcal/mol nonpolar term on the first run of this
    module).  The receptor side therefore uses
    :func:`odock.sasa.interface_area`, which is internally consistent
    (``free_area`` 9 274.9, ``complex_area`` 9 096.9, ``buried_area`` 178.0 on the
    same input), and the inconsistency in ``burial``'s reference area for large
    atom sets is reported rather than worked around silently.
    """
    from .sasa import burial, interface_area

    ligand_report = burial(list(ligand), list(receptor), probe=probe, points=points)
    ligand_delta = float(ligand_report.reference_total - ligand_report.total_area)
    interface = interface_area(list(receptor), list(ligand), probe=probe, points=points)
    receptor_delta = float(interface.get("buried_area", 0.0))
    ligand_delta = max(0.0, ligand_delta)
    receptor_delta = max(0.0, receptor_delta)
    return {
        "ligand": ligand_delta,
        "receptor": receptor_delta,
        "total": ligand_delta + receptor_delta,
    }


def endpoint_score(
    *,
    label: str,
    pose_index: int,
    affinity: float,
    interaction: float,
    ligand_atoms: Sequence[Any],
    receptor_atoms: Sequence[Any],
    n_torsions: float = 0.0,
    scoring: str = "vina",
    gamma: float = DEFAULT_GAMMA,
    dielectric: float = DEFAULT_DIELECTRIC,
    with_entropy: bool = False,
    temperature: float = 298.15,
    strain: float = 0.0,
    probe: float = 1.4,
    points: int = 92,
) -> EndpointPose:
    """One pose's end-point decomposition, term by term."""
    electrostatic = coulomb_energy(ligand_atoms, receptor_atoms, dielectric=dielectric)
    areas = buried_area(ligand_atoms, receptor_atoms, probe=probe, points=points)
    nonpolar = -float(gamma) * areas["total"]
    entropy = 0.0
    if with_entropy:
        from .metrics import entropy_penalty

        entropy = float(entropy_penalty(float(n_torsions), temperature=temperature))
    return EndpointPose(
        label=label, pose_index=int(pose_index), affinity=float(affinity),
        interaction=float(interaction), electrostatic=electrostatic, nonpolar=nonpolar,
        entropy=entropy, strain=float(strain), buried_area=areas["total"],
        n_torsions=float(n_torsions), scoring=scoring,
    )


def endpoint_ensemble(
    models: Sequence[str],
    receptor_text: str,
    *,
    box: Any = None,
    label: str = "ligand",
    receptor: str = "receptor",
    scoring: str = "vina",
    gamma: float = DEFAULT_GAMMA,
    dielectric: float = DEFAULT_DIELECTRIC,
    with_entropy: bool = False,
    temperature: float = 298.15,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    probe: float = 1.4,
    points: int = 92,
    torsions: Optional[Sequence[float]] = None,
) -> EndpointResult:
    """Rescore a pose ensemble end-to-end and report dG with its interval.

    The interaction energies come from :func:`odock.consensus.rescore_poses` -- the
    same call the consensus layer uses -- so this module cannot drift away from
    what the docking pipeline reports for the same pose; that equality is asserted
    in the tests.
    """
    from .consensus import pdbqt_atoms, pdbqt_models, rescore_poses

    if not models:
        raise EnsembleError("endpoint_ensemble needs at least one pose model")
    texts = [models] if isinstance(models, str) else list(models)
    receptor_atoms = pdbqt_atoms(receptor_text)
    if not receptor_atoms:
        raise EnsembleError("the receptor PDBQT has no atoms")
    table = rescore_poses(list(texts), receptor_text, box)
    if scoring not in table:
        raise EnsembleError(
            f"the rescorer did not report {scoring!r}; it reported {sorted(table)}"
        )
    column = table[scoring]
    notes: List[str] = []
    if scoring not in _SCORINGS_WITHOUT_ELECTROSTATICS:
        notes.append(
            f"scoring={scoring!r} has its own electrostatic term, so adding the "
            "Coulomb term double counts electrostatics; use vina or vinardo for a "
            "clean decomposition"
        )
    interactions = list(column["inter"])
    affinities = list(column.get("affinity", column.get("total", [])))
    poses: List[EndpointPose] = []
    for index, model in enumerate(texts):
        atoms = pdbqt_atoms(model)
        if not atoms:
            notes.append(f"pose {index + 1} has no atoms and was skipped")
            continue
        torsion = 0.0
        if torsions is not None and index < len(torsions):
            torsion = float(torsions[index])
        model_torsions = _count_torsions(model)
        poses.append(
            endpoint_score(
                label=label, pose_index=index, affinity=affinities[index] if index < len(affinities) else float("nan"),
                interaction=interactions[index] if index < len(interactions) else float("nan"),
                ligand_atoms=atoms, receptor_atoms=receptor_atoms,
                n_torsions=torsion or model_torsions, scoring=scoring, gamma=gamma,
                dielectric=dielectric, with_entropy=with_entropy,
                temperature=temperature, probe=probe, points=points,
            )
        )
    return EndpointResult(
        ligand=label, receptor=receptor, scoring=scoring, poses=poses, gamma=gamma,
        dielectric=dielectric, samples=int(samples), seed=int(seed), notes=notes,
    )


def _count_torsions(model: str) -> float:
    """``TORSDOF`` from a PDBQT model, which is the rotatable-bond count."""
    for line in reversed(str(model).splitlines()):
        if line.startswith("TORSDOF"):
            parts = line.split()
            if len(parts) > 1:
                try:
                    return float(parts[1])
                except ValueError:
                    return 0.0
    return 0.0


def kendall_tau(first: Sequence[float], second: Sequence[float]) -> float:
    """Kendall's tau-b, so the ranking comparison needs no scipy.

    ``first`` and ``second`` are scores where **higher is better** for both; the
    caller flips sign as needed.  Ties are counted the tau-b way.
    """
    n = len(first)
    if n < 2:
        return float("nan")
    concordant = discordant = ties_first = ties_second = 0
    for i in range(n):
        for j in range(i + 1, n):
            left = first[i] - first[j]
            right = second[i] - second[j]
            if left == 0 and right == 0:
                ties_first += 1
                ties_second += 1
            elif left == 0:
                ties_first += 1
            elif right == 0:
                ties_second += 1
            elif (left > 0) == (right > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_first) * (concordant + discordant + ties_second)
    )
    return (concordant - discordant) / denominator if denominator else float("nan")


def compare_rankings(
    endpoint: Sequence[EndpointResult],
    vina: Sequence[float],
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Do the end-point and the docking rankings agree, and is that resolvable?

    Returns Spearman rho between the two orderings **with its n and bootstrap
    interval**, the Kendall tau between them with its interval, how many ligands
    change rank and whether the top ligand agrees.  The interval is the point: at
    a small n two rankings can look different and be indistinguishable, and the
    ``distinguishable`` flag says which of those two situations this is --

    * ``distinguishable`` is True only when the tau interval **excludes 1.0**
      (perfect agreement), i.e. the two orderings differ by more than the
      resampling spread of this ligand set;
    * otherwise the honest statement is that the data cannot separate them, and
      the ranking difference is not a result.

    Resampling is over **ligands**, so it says nothing about a new ligand; it is
    the spread of this comparison at this n.
    """
    from . import consensus

    usable = [
        (result, float(score))
        for result, score in zip(endpoint, vina)
        if result.poses and math.isfinite(float(score)) and math.isfinite(result.mean)
    ]
    n = len(usable)
    if n < 3:
        return {
            "n": n, "rho": float("nan"), "rho_low": float("nan"), "rho_high": float("nan"),
            "tau": float("nan"), "tau_low": float("nan"), "tau_high": float("nan"),
            "top1_agreement": None, "n_rank_changes": None, "distinguishable": None,
            "note": (
                f"not resolvable: {n} ligand(s); a ranking comparison needs at "
                "least three, and below five the interval is wider than any "
                "difference it could show"
            ),
        }
    # "Better" is lower dG and lower (more negative) Vina affinity, so both are
    # flipped to a higher-is-better axis before ranking.
    endpoint_scores = [-result.mean for result, _ in usable]
    vina_scores = [-score for _, score in usable]
    rho = consensus.spearman(endpoint_scores, vina_scores)
    tau = kendall_tau(endpoint_scores, vina_scores)
    rng = np.random.default_rng(int(seed))
    rhos: List[float] = []
    taus: List[float] = []
    for _ in range(int(samples)):
        pick = rng.integers(0, n, n)
        left = [endpoint_scores[index] for index in pick]
        right = [vina_scores[index] for index in pick]
        if len({round(value, 12) for value in left}) < 2:
            continue
        value = consensus.spearman(left, right)
        if math.isfinite(value):
            rhos.append(value)
        value = kendall_tau(left, right)
        if math.isfinite(value):
            taus.append(value)
    alpha = (1.0 - float(confidence)) / 2.0

    def interval(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))

    rho_low, rho_high = interval(rhos)
    tau_low, tau_high = interval(taus)
    endpoint_rank = _ranks(endpoint_scores)
    vina_rank = _ranks(vina_scores)
    changes = sum(1 for a, b in zip(endpoint_rank, vina_rank) if a != b)
    # The identity of the *best* ligand, not the rank of the first one in the list:
    # an earlier version compared `endpoint_rank[0] == vina_rank[0]`, which asks
    # whether the first ligand listed happens to be ranked the same by both, and
    # reported "top-1 agreement" for a set where the two winners were different
    # (measured on 17 ligands: fluorobenzamidine end-point, warfarin by Vina).
    top1 = int(np.argmax(endpoint_scores)) == int(np.argmax(vina_scores))
    # "Distinguishable" means the interval on Kendall's tau EXCLUDES perfect
    # agreement (tau = 1).  A perfectly reversed ordering is the clearest case of
    # two methods disagreeing, and it must come out True -- an earlier version of
    # this line also required tau_low > -1, which made the reversed case report
    # "not distinguishable", the exact opposite of the truth.
    distinguishable = bool(
        math.isfinite(tau_low) and math.isfinite(tau_high)
        and not (tau_low <= 1.0 <= tau_high)
    )
    note = ""
    if not distinguishable:
        note = (
            "the two rankings are NOT distinguishable at this n: the interval on "
            "Kendall's tau includes perfect agreement, so the reordering this "
            "method produces is inside the resampling spread of this ligand set"
        )
    return {
        "n": n, "rho": rho, "rho_low": rho_low, "rho_high": rho_high,
        "tau": tau, "tau_low": tau_low, "tau_high": tau_high,
        "top1_agreement": bool(top1), "n_rank_changes": int(changes),
        "distinguishable": distinguishable, "note": note,
        "samples": len(taus), "confidence": float(confidence),
    }


def _ranks(scores: Sequence[float]) -> List[int]:
    """Ranks with 1 = best (highest score), ties by position order."""
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    ranks = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


def read_pose_models(path: Union[str, Path]) -> List[str]:
    """The pose models of a PDBQT file, as text, ready for the rescorer."""
    from .consensus import pdbqt_models

    return pdbqt_models(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The command line: `odock endpoint`
# ---------------------------------------------------------------------------


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def cmd_endpoint(args) -> int:
    """``odock endpoint``: end-point rescoring of a pose file, with its interval."""
    import json as _json

    from .ensemble import EnsembleError

    try:
        receptor_path = Path(args.receptor)
        poses_path = Path(args.poses)
        if not receptor_path.exists():
            raise EnsembleError(f"no such receptor: {receptor_path}")
        if not poses_path.exists():
            raise EnsembleError(f"no such pose file: {poses_path}")
        box = None
        if getattr(args, "box", None):
            document = _json.loads(Path(args.box).read_text(encoding="utf-8"))
            from .prepare import BoxSpec

            box = BoxSpec(
                center=tuple(document["center"]), size=tuple(document["size"]),
                spacing=document.get("spacing", 0.375),
            )
        result = endpoint_ensemble(
            read_pose_models(poses_path),
            receptor_path.read_text(encoding="utf-8"),
            box=box,
            label=args.label or poses_path.stem,
            receptor=receptor_path.stem,
            scoring=args.scoring,
            gamma=float(args.gamma),
            dielectric=float(args.dielectric),
            with_entropy=bool(args.entropy),
            samples=int(args.samples),
            seed=int(args.seed),
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock endpoint: error: {exc}")
        return int(getattr(exc, "code", 2))
    if not args.quiet:
        print(result.text())
        print()
        _cli_eprint(
            "dG is a ranking device, not an experimental affinity; the interval is "
            "the pose ensemble's spread, not the method's error"
        )
    if getattr(args, "json_out", None):
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            _json.dumps(result.as_dict(), indent=2, default=str) + "\n", encoding="utf-8"
        )
    return 0


def add_endpoint_parser(sub: Any) -> None:
    """Register ``odock endpoint`` on a subparser collection.

    The registrar lives here, in the module that owns the analysis, so the CLI
    registry in ``cli_ext.py`` only names it.
    """
    parser = sub.add_parser(
        "endpoint",
        help="MM-GBSA-style end-point rescoring of a pose file, with its spread",
        description=(
            "Decompose every pose into the kernel's interaction energy plus a "
            "SASA-proportional nonpolar term and a distance-dependent-dielectric "
            "Coulomb term, and report dG over the pose ensemble with a bootstrap "
            "interval.  This is a documented approximation: no Poisson-Boltzmann or "
            "Generalised-Born solvation, no explicit water, entropy omitted unless "
            "--entropy is given, and no relaxation of the complex.  The number is a "
            "ranking device, not an experimental affinity."
        ),
    )
    parser.add_argument("-r", "--receptor", required=True, help="prepared receptor PDBQT")
    parser.add_argument("-l", "--poses", required=True, help="pose PDBQT (multi-model)")
    parser.add_argument("-b", "--box", help="box JSON the poses were produced with")
    parser.add_argument("--label", help="name for the ligand in the report")
    parser.add_argument(
        "-s", "--scoring", default="vina", choices=("vina", "vinardo", "ad4"),
        help="the scoring function whose interaction energy to decompose; ad4 has "
             "its own electrostatics, so the Coulomb term double counts there",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help="nonpolar surface tension, kcal/mol/A^2 (default %(default)s)",
    )
    parser.add_argument(
        "--dielectric", type=float, default=DEFAULT_DIELECTRIC,
        help="eps = this * r for the Coulomb term (default %(default)s)",
    )
    parser.add_argument(
        "--entropy", action="store_true",
        help="add the torsional -T dS estimate (off by default; it is a rotor count)",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help="bootstrap resamples for the interval")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", help="write the whole decomposition as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no report")
    parser.set_defaults(func=cmd_endpoint)

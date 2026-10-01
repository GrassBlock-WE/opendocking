# SPDX-License-Identifier: GPL-3.0-or-later
"""Conformer ensembles as a first-class, *measured* object.

Every 3-D ligand-based method in this project — the pharmacophore fingerprints, the
shape/electrostatic overlay, the 3-D similarity search — reads one number from
this module: how good the conformer ensemble behind the answer is.  Rather than
asserting it, :func:`build_ensemble` generates the ensemble with one documented
generator and returns the measurements that say whether it is any use:

* **RMSD coverage** — the pairwise heavy-atom RMSD distribution of the kept
  conformers (minimum, mean, maximum), so "did the ensemble actually explore the
  shape space" is a number and not a hope;
* **energy distribution** — MMFF94 (or UFF, reported which) energies after
  minimisation, the span of the kept set, and how many conformers the energy window
  removed;
* **torsion-space coverage** — for every rotatable bond, how many distinct 60-degree
  rotamer bins the ensemble visits, and the geometric mean of the per-bond
  coverage, which is the number that exposes a "10 conformers, all the same
  torsion" ensemble;
* **failures** — the molecules that did not embed at all, named with the reason;
* **cost** — seconds per molecule, so a caller can budget a library.

Determinism is part of the contract: one seed, one process, one thread, and the
same ensemble every time (the test suite pins it).

**Honesty.** A conformer ensemble is a *sample of a modelled space*: ETKDGv3
samples what its knowledge-based torsion terms believe is reasonable, at a fixed
seed, and the ensemble says nothing about how the molecule behaves in water, in a
protein, or at another seed.  A high coverage number means "this sample is spread
out", not "this is the conformational ensemble".  The measurements are there so a
reader can see what the sample is; they are not a validation of the physics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from .chem.ligand import rotatable_bonds as _rotatable_bonds

__all__ = [
    "DEFAULT_CONFORMERS",
    "DEFAULT_RMSD_PRUNE",
    "DEFAULT_ENERGY_WINDOW",
    "DEFAULT_SEED",
    "TORSION_BIN_DEGREES",
    "DEFAULT_ATTEMPTS_PER_ROTOR",
    "MIN_ATTEMPTS",
    "MAX_ATTEMPTS",
    "USR_DESCRIPTORS",
    "ConformerEnsemble",
    "require_rdkit",
    "rotor_scaled_attempts",
    "build_ensemble",
    "prune_by_rmsd",
    "rmsd_matrix",
    "torsion_bins",
    "ensemble_quality",
    "usr_descriptors",
    "usr_similarity",
]

#: The number of embedding attempts to use when the caller does not name one.
#: ``None`` means "let the rotor count decide", through :func:`rotor_scaled_attempts`.
DEFAULT_CONFORMERS: Optional[int] = None
#: Heavy-atom RMSD above which two conformers are kept as different (Angstrom).
DEFAULT_RMSD_PRUNE = 0.5
#: Energy window applied when a force field typed the molecule (kcal/mol).
DEFAULT_ENERGY_WINDOW = 10.0
#: The fixed ETKDGv3 seed; a library regenerates identically.
DEFAULT_SEED = 20240101
#: Rotamer bin width for the torsion-coverage measurement (degrees).
TORSION_BIN_DEGREES = 60

#: Embedding attempts per rotatable bond, and the floor and cap of the rule.
#:
#: Measured, not assumed (`docs/CONFORMERS.md` §1): the binding constraint on an
#: ensemble is **how many attempts the embedder was given**, not the pruning or the
#: energy window — a 4-rotor molecule reaches full torsion coverage at ~40-64
#: attempts and a 9-rotor one stops improving at ~64, while the RMSD pruning removes
#: 0-2 conformers and the energy window 0-3 on the same set.  8 per bond with a floor
#: of 16 puts warfarin (4 rotors) at 40 and a 9-rotor chain at 80, which is where the
#: coverage curve flattens; the cap keeps a 30-rotor molecule from running for
#: minutes.
DEFAULT_ATTEMPTS_PER_ROTOR = 8
MIN_ATTEMPTS = 16
MAX_ATTEMPTS = 128

#: Names of the 12 USR (ultrafast shape recognition) descriptors, in order: for
#: each of the four reference points the mean, the standard deviation and the
#: maximum distance to every atom.
USR_DESCRIPTORS: Tuple[str, ...] = tuple(
    f"{stat}_{reference}"
    for reference in ("centroid", "closest", "farthest", "farthest2")
    for stat in ("mean", "std", "max")
)


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for conformer generation. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# The ensemble
# ---------------------------------------------------------------------------


@dataclass
class ConformerEnsemble:
    """A molecule with its kept conformers and the measurements on them."""

    mol: Any = None
    name: str = ""
    #: How many embedding **attempts** were made and how many conformers came back.
    #: The gap is the embedding stage's own loss (ETKDGv3 refuses to return two
    #: conformers closer than ``rmsd_prune``, so a rigid molecule returns one from
    #: sixteen attempts, which is correct and not a failure).
    requested: int = 0
    embedded: int = 0
    #: Conformer indices kept, in the order they sit in ``mol``.
    kept: Tuple[int, ...] = ()
    #: MMFF94 or UFF energy per embedded conformer (``()`` when neither typed it).
    energies: Tuple[float, ...] = ()
    force_field: str = ""
    #: Conformers removed by the RMSD pruning and by the energy window.
    pruned_rmsd: int = 0
    pruned_energy: int = 0
    #: Rotatable bonds, and whether the attempt count was derived from them.
    rotors: int = 0
    scaled: bool = False
    #: Pairwise heavy-atom RMSD of the kept set (Angstrom).
    rmsd_min: float = 0.0
    rmsd_mean: float = 0.0
    rmsd_max: float = 0.0
    #: Per-rotatable-bond distinct 60-degree bins in the kept set, in bond order.
    torsion_bins: Tuple[int, ...] = ()
    torsion_coverage: float = 0.0
    seconds: float = 0.0
    error: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def n_conformers(self) -> int:
        return len(self.kept)

    @property
    def embedded_fraction(self) -> float:
        return (self.embedded / self.requested) if self.requested else 0.0

    @property
    def coverage_ceiling(self) -> float:
        """The most torsion coverage ``n_conformers`` kept conformers *can* reach.

        ``k`` conformers can occupy at most ``k`` distinct bins on one bond, and a
        bond has six, so the geometric mean over the rotors cannot exceed
        ``min(1, kept / 6)``.  A coverage of 59 % from 10 conformers is therefore
        very close to its ceiling of 100 % while a coverage of 59 % from 60
        conformers is not, and the two are otherwise indistinguishable.
        """
        if not self.torsion_bins:
            return 0.0
        return min(1.0, self.n_conformers / 6.0)

    def accounting(self) -> Dict[str, Any]:
        """The per-molecule loss accounting: where the requested conformers went.

        ``attempts -> embedded -> kept`` with each stage's loss named, plus the
        torsion coverage against its ceiling.  This is the table that says whether a
        thin ensemble is the generator's fault (many attempts, few embedded) or the
        molecule's (few rotatable bonds).
        """
        return {
            "name": self.name,
            "rotors": int(self.rotors),
            "scaled": bool(self.scaled),
            "requested": int(self.requested),
            "embedded": int(self.embedded),
            "lost_embedding": int(self.requested - self.embedded),
            "pruned_energy": int(self.pruned_energy),
            "pruned_rmsd": int(self.pruned_rmsd),
            "kept": int(self.n_conformers),
            "torsion_bins": [int(value) for value in self.torsion_bins],
            "torsion_coverage": round(float(self.torsion_coverage), 4),
            "coverage_ceiling": round(self.coverage_ceiling, 4),
            "seconds": round(float(self.seconds), 4),
            "error": self.error,
        }

    def accounting_line(self) -> str:
        """One line: ``attempts -> embedded -> (energy cut, RMSD cut) -> kept``."""
        return (
            f"{self.name:<28}{self.rotors:>3}{self.requested:>6}{self.embedded:>6}"
            f"{self.pruned_energy:>7}{self.pruned_rmsd:>7}{self.n_conformers:>6}"
            f"{self.torsion_coverage:>8.0%}{self.coverage_ceiling:>8.0%}"
            f"{self.seconds:>8.2f}"
        )

    @property
    def energy_span(self) -> Optional[float]:
        if len(self.energies) < 2:
            return None
        return max(self.energies) - min(self.energies)

    @property
    def best_energy(self) -> Optional[float]:
        return min(self.energies) if self.energies else None

    def conformer_indices(self) -> Tuple[int, ...]:
        """The kept conformer ids."""
        return self.kept

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "requested": int(self.requested),
            "embedded": int(self.embedded),
            "kept": int(self.n_conformers),
            "embedded_fraction": round(self.embedded_fraction, 4),
            "force_field": self.force_field,
            "rotors": int(self.rotors),
            "scaled": bool(self.scaled),
            "pruned_rmsd": int(self.pruned_rmsd),
            "pruned_energy": int(self.pruned_energy),
            "rmsd_min": round(float(self.rmsd_min), 4),
            "rmsd_mean": round(float(self.rmsd_mean), 4),
            "rmsd_max": round(float(self.rmsd_max), 4),
            "torsion_bins": [int(v) for v in self.torsion_bins],
            "torsion_coverage": round(float(self.torsion_coverage), 4),
            "coverage_ceiling": round(self.coverage_ceiling, 4),
            "energy_best": None if self.best_energy is None else round(self.best_energy, 4),
            "energy_span": None if self.energy_span is None else round(self.energy_span, 4),
            "seconds": round(float(self.seconds), 4),
            "error": self.error,
            "notes": list(self.notes),
        }

    def table(self) -> str:
        lines = [
            f"{self.name}: {self.n_conformers} conformer(s) kept of {self.embedded} "
            f"embedded ({self.requested} attempt(s)"
            + (", rotor-scaled" if self.scaled else "")
            + f") in {self.seconds:.3f} s",
            "  accounting: attempts -> embedded -> (-energy, -RMSD) -> kept: "
            f"{self.requested} -> {self.embedded} -> (-{self.pruned_energy}, "
            f"-{self.pruned_rmsd}) -> {self.n_conformers}",
        ]
        if self.error:
            lines.append(f"  failed: {self.error}")
            return "\n".join(lines)
        lines.append(
            f"  RMSD coverage: min {self.rmsd_min:.2f}, mean {self.rmsd_mean:.2f}, "
            f"max {self.rmsd_max:.2f} A"
        )
        if self.energies:
            span = "n/a" if self.energy_span is None else f"{self.energy_span:.2f}"
            best = "n/a" if self.best_energy is None else f"{self.best_energy:.2f}"
            lines.append(
                f"  {self.force_field}: best {best}, span {span} kcal/mol "
                f"({self.pruned_energy} cut by the window)"
            )
        else:
            lines.append("  energies: no force field could type this molecule")
        if self.torsion_bins:
            lines.append(
                f"  torsion coverage: {self.torsion_coverage:.0%} of its "
                f"{self.coverage_ceiling:.0%} ceiling "
                f"({self.rotors} rotor(s), bins {list(self.torsion_bins)}, "
                f"{TORSION_BIN_DEGREES} degrees each)"
            )
        else:
            lines.append("  torsion coverage: no rotatable bond (one shape to sample)")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _heavies(mol) -> List[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def rotor_scaled_attempts(
    mol,
    *,
    per_rotor: int = DEFAULT_ATTEMPTS_PER_ROTOR,
    minimum: int = MIN_ATTEMPTS,
    maximum: int = MAX_ATTEMPTS,
) -> int:
    """How many embedding attempts a molecule's rotatable-bond count justifies.

    ``min(maximum, max(minimum, per_rotor · (1 + rotors)))``, with the rotor count
    from :func:`odock.chem.ligand.rotatable_bonds` — so a methyl does not count, a
    symmetric end does not count, and "one shape" really means one shape.  This is
    the deterministic rule behind the ``n_conformers=None`` default; it depends on
    the molecule and nothing else (no time, no memory, no seed).
    """
    require_rdkit()
    rotors = len(_rotatable_bonds(mol))
    return int(
        min(
            int(maximum),
            max(int(minimum), int(per_rotor) * (1 + rotors)),
        )
    )


def build_ensemble(
    mol,
    *,
    n_conformers: Optional[int] = DEFAULT_CONFORMERS,
    seed: int = DEFAULT_SEED,
    rmsd_prune: float = DEFAULT_RMSD_PRUNE,
    energy_window: float = DEFAULT_ENERGY_WINDOW,
    prune: bool = True,
    minimize: bool = True,
    name: str = "",
    scale_with_rotors: bool = True,
    use_input_conformers: bool = False,
) -> ConformerEnsemble:
    """Generate, minimise, prune and *measure* a conformer ensemble.

    The generator is ETKDGv3 with explicit parameters: the fixed ``seed``, one
    thread (so the result cannot depend on scheduling) and ``pruneRmsThresh`` set
    to ``rmsd_prune`` so the embedder itself does not return duplicates.
    Minimisation is MMFF94 with an MMFF94s/UFF fallback, and the energy window keeps
    the conformers within ``energy_window`` kcal/mol of the best.

    ``n_conformers`` is the number of **embedding attempts**.  The default ``None``
    derives it from the rotatable-bond count with :func:`rotor_scaled_attempts`,
    because that is the binding constraint on an ensemble: measured, the RMSD
    pruning removes 0-2 conformers and the energy window 0-3, while a 4-rotor
    molecule's torsion coverage rises from 69 % at 16 attempts to 100 % at 64.  An
    explicit integer is used exactly — a caller who asks for 2 attempts gets 2, so a
    published protocol stays reproducible — and ``scale_with_rotors=False`` restores
    the fixed 16-attempt default for the ``None`` case.

    ``use_input_conformers`` measures the coordinates a molecule already carries
    instead of generating any: nothing is embedded, minimised or pruned, the energies
    stay unavailable, and a note says so.  A docked pose or a crystal structure must
    not be silently replaced by a fresh ETKDGv3 sample just to be reported on, and
    that is the only way either can be measured.

    Returns a :class:`ConformerEnsemble`: the molecule (whose conformers are the
    embedded ones, in order) together with the kept indices, the energies, the
    coverage measurements and the attempt accounting
    (:meth:`ConformerEnsemble.accounting`).  A molecule that cannot be embedded
    comes back with ``error`` set and no conformers rather than raising — one
    unembeddable molecule must not end a library run.
    """
    require_rdkit()
    start = time.perf_counter()
    label = str(name) or _mol_name(mol)
    if n_conformers is not None and int(n_conformers) < 1:
        raise ValueError(f"n_conformers must be >= 1, got {n_conformers!r}")
    if float(rmsd_prune) < 0:
        raise ValueError(f"rmsd_prune must be >= 0, got {rmsd_prune!r}")

    work = Chem.Mol(mol)
    if work.GetNumAtoms() == 0:
        return ConformerEnsemble(
            mol=work,
            name=label,
            requested=0 if n_conformers is None else int(n_conformers),
            error="the molecule has no atoms",
            seconds=time.perf_counter() - start,
        )
    if not any(atom.GetAtomicNum() == 1 for atom in work.GetAtoms()):
        work = Chem.AddHs(work)

    scaled = n_conformers is None
    if n_conformers is None:
        # `scale_with_rotors=False` restores the old fixed default (16 attempts),
        # which is also the floor of the scaled rule.
        attempts = rotor_scaled_attempts(work) if scale_with_rotors else int(MIN_ATTEMPTS)
    else:
        attempts = int(n_conformers)
    attempts = max(1, int(attempts))
    # One rotor list, used for the attempt accounting, the torsion coverage and the
    # ensemble record: `odock.chem.ligand.rotatable_bonds` excludes methyls, amides,
    # ring bonds and symmetric ends, so "no rotor" here really means one shape.
    rotor_bonds = _rotatable_bonds(work)
    n_rotors = len(rotor_bonds)

    # A molecule that already carries coordinates is *measured* rather than
    # re-embedded when the caller asks for that: re-embedding a docked pose or a
    # crystal structure would silently replace the geometry being reported.
    from_input = bool(use_input_conformers) and work.GetNumConformers() > 0
    if from_input:
        ids = list(range(work.GetNumConformers()))
        if n_conformers is not None and 0 < int(n_conformers) < len(ids):
            ids = ids[: int(n_conformers)]
    else:
        work.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.numThreads = 1
        # The embedder's duplicate filter is deliberately the same threshold the
        # post-prune uses: measured, the conformers ETKDGv3 returns at 0.5 A are
        # already 0.56-0.88 A apart, so a finer embedder threshold (0.25 A) embeds
        # ~3x more and the greedy prune then throws every extra one away — cost with
        # no coverage.
        params.pruneRmsThresh = float(rmsd_prune)
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.clearConfs = True
        try:
            ids = list(
                AllChem.EmbedMultipleConfs(work, numConfs=int(attempts), params=params)
            )
        except Exception as exc:  # pragma: no cover - RDKit raises rarely
            return ConformerEnsemble(
                mol=work,
                name=label,
                requested=int(attempts),
                rotors=int(n_rotors),
                scaled=scaled,
                error=f"{type(exc).__name__}: {exc}",
                seconds=time.perf_counter() - start,
            )
    if not ids:
        try:
            status = AllChem.EmbedMolecule(work, randomSeed=int(seed), useRandomCoords=True)
        except Exception:  # pragma: no cover - defensive
            status = 1
        if status != 0:
            return ConformerEnsemble(
                mol=work,
                name=label,
                requested=int(attempts),
                rotors=int(rotors),
                scaled=scaled,
                error="ETKDGv3 could not embed the molecule, even with random coordinates",
                seconds=time.perf_counter() - start,
            )
        ids = [0]

    embedded = len(ids)
    force_field = ""
    energies: List[float] = []
    if minimize and not from_input:
        for candidate in ("MMFF94", "MMFF94s", "UFF"):
            try:
                if candidate == "UFF":
                    if not AllChem.UFFHasAllMoleculeParams(work):
                        continue
                    result = AllChem.UFFOptimizeMoleculeConfs(work, numThreads=1, maxIters=200)
                else:
                    if not AllChem.MMFFHasAllMoleculeParams(work):
                        continue
                    result = AllChem.MMFFOptimizeMoleculeConfs(
                        work, numThreads=1, maxIters=200, mmffVariant=candidate
                    )
            except Exception:  # pragma: no cover - defensive
                continue
            values = [float(energy) for _, energy in result]
            if values and all(math.isfinite(value) for value in values):
                force_field = candidate
                energies = values
                break
    if not energies:
        energies = [float("nan")] * embedded

    keep = list(range(embedded))
    pruned_energy = 0
    if energies and all(math.isfinite(value) for value in energies):
        best = min(energies)
        keep = [index for index in keep if energies[index] <= best + float(energy_window) + 1e-9]
        pruned_energy = embedded - len(keep)

    pruned_rmsd = 0
    if prune and not from_input and len(keep) > 1:
        keep, pruned_rmsd = _greedy_rmsd_prune(work, keep, float(rmsd_prune))

    kept = tuple(keep)
    matrix = rmsd_matrix(work, conformers=kept)
    if matrix.shape[0] > 1:
        upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
        rmsd_min = float(upper.min())
        rmsd_mean = float(upper.mean())
        rmsd_max = float(upper.max())
    else:
        rmsd_min = rmsd_mean = rmsd_max = 0.0

    bins: List[int] = []
    coverage = 0.0
    if rotor_bonds:
        for bond in rotor_bonds:
            signature = torsion_bins(work, bond, conformers=kept)
            if signature:
                bins.append(len(signature))
        if bins:
            # The geometric mean of the per-bond bin counts, over the six bins a
            # bond can occupy: 1.0 means every rotor visited every 60-degree bin.
            coverage = float(np.prod([max(1, value) for value in bins]) ** (1.0 / len(bins)) / 6.0)
            coverage = min(1.0, coverage)
    notes: List[str] = []
    if pruned_rmsd:
        notes.append(
            f"{pruned_rmsd} conformer(s) were removed as redundant at "
            f"{float(rmsd_prune):.2f} A heavy-atom RMSD"
        )
    if from_input:
        notes.append(
            f"the {embedded} conformer(s) were taken from the input: nothing was "
            "embedded, minimised or pruned, so the RMSD and torsion measurements "
            "describe the coordinates as they were given"
        )
    elif not force_field:
        notes.append(
            "no force field could type this molecule, so the energies are unavailable "
            "and no energy window was applied"
        )
    if len(kept) < 3 and not from_input:
        notes.append(
            "fewer than three conformers survived; a shape or pharmacophore score "
            "computed from this ensemble is a statement about one or two geometries"
        )
    if scaled and n_rotors >= 3 and embedded * 2 < attempts:
        # Three rotors is where "the rotor count promised more shapes than the
        # molecule has" becomes a real caveat rather than noise: a one-rotor amidine
        # that returns a single conformer from sixteen attempts is simply rigid, and
        # the note would fire on twelve of the seventeen demo molecules.
        notes.append(
            f"the rotor-scaled rule asked for {attempts} attempt(s) and {embedded} "
            "conformer(s) came back: ETKDGv3 returns only geometries at least "
            f"{float(rmsd_prune):.2f} A apart, so a compact molecule has fewer "
            "distinct shapes than its rotor count suggests"
        )
    return ConformerEnsemble(
        mol=work,
        name=label,
        requested=int(attempts),
        embedded=embedded,
        rotors=int(n_rotors),
        scaled=scaled,
        kept=kept,
        energies=tuple(energies),
        force_field=force_field,
        pruned_rmsd=pruned_rmsd,
        pruned_energy=pruned_energy,
        rmsd_min=rmsd_min,
        rmsd_mean=rmsd_mean,
        rmsd_max=rmsd_max,
        torsion_bins=tuple(bins),
        torsion_coverage=coverage,
        seconds=time.perf_counter() - start,
        notes=notes,
    )


def _mol_name(mol, fallback: str = "ligand") -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def rmsd_matrix(
    mol, *, conformers: Optional[Sequence[int]] = None, heavy_only: bool = True
) -> np.ndarray:
    """The pairwise symmetry-aware RMSD matrix of a multi-conformer molecule.

    Uses RDKit's ``GetConformerRMSMatrix`` with ``prealigned=False``, which
    superposes each pair before measuring, on the heavy atoms by default, so a
    hydrogen flip does not count as a different shape.
    """
    require_rdkit()
    indices = list(conformers) if conformers is not None else list(range(mol.GetNumConformers()))
    if len(indices) < 2:
        return np.zeros((len(indices), len(indices)), dtype=float)
    atoms = _heavies(mol) if heavy_only else list(range(mol.GetNumAtoms()))
    # `GetConformerRMSMatrix` walks the molecule's own conformer order, so build a
    # copy holding exactly the conformers asked for.
    subset = Chem.Mol(mol)
    subset.RemoveAllConformers()
    for index in indices:
        subset.AddConformer(Chem.Conformer(mol.GetConformer(int(index))), assignId=True)
    values = list(AllChem.GetConformerRMSMatrix(subset, prealigned=False, atomIds=atoms))
    count = len(indices)
    matrix = np.zeros((count, count), dtype=float)
    position = 0
    for row in range(1, count):
        for column in range(row):
            matrix[row, column] = matrix[column, row] = float(values[position])
            position += 1
    return matrix


def prune_by_rmsd(
    mol, *, threshold: float = DEFAULT_RMSD_PRUNE, conformers: Optional[Sequence[int]] = None
) -> Tuple[List[int], int]:
    """Greedily drop conformers closer than ``threshold`` to a kept one.

    Returns ``(kept indices, number dropped)``.  The first conformer is always kept
    and the rest are visited in order, so the result is deterministic.
    """
    require_rdkit()
    indices = list(conformers) if conformers is not None else list(range(mol.GetNumConformers()))
    if len(indices) < 2 or float(threshold) <= 0:
        return indices, 0
    return _greedy_rmsd_prune(mol, indices, float(threshold))


def _greedy_rmsd_prune(mol, indices: Sequence[int], threshold: float) -> Tuple[List[int], int]:
    """Keep a conformer when it is at least ``threshold`` RMSD from every kept one."""
    matrix = rmsd_matrix(mol, conformers=indices)
    kept: List[int] = []
    kept_rows: List[int] = []
    for row, index in enumerate(indices):
        if all(matrix[row, other] >= threshold for other in kept_rows):
            kept.append(int(index))
            kept_rows.append(row)
    return kept, len(indices) - len(kept)


# ---------------------------------------------------------------------------
# Torsion-space coverage
# ---------------------------------------------------------------------------


def torsion_bins(
    mol,
    bond: Tuple[int, int],
    *,
    conformers: Optional[Sequence[int]] = None,
    bins: int = int(360 / TORSION_BIN_DEGREES),
) -> List[int]:
    """The distinct torsion bins the ensemble visits for one rotatable bond.

    The torsion is measured on the four atoms around the bond — the two bond atoms
    and one heavy neighbour on each side — for every conformer, and folded into
    ``bins`` equal bins over 360 degrees.  The *count* of distinct bins is the
    coverage measurement: one bin means the ensemble never moved that bond.
    """
    require_rdkit()
    begin, end = int(bond[0]), int(bond[1])
    left = [
        neighbour.GetIdx()
        for neighbour in mol.GetAtomWithIdx(begin).GetNeighbors()
        if neighbour.GetIdx() != end and neighbour.GetAtomicNum() > 1
    ]
    right = [
        neighbour.GetIdx()
        for neighbour in mol.GetAtomWithIdx(end).GetNeighbors()
        if neighbour.GetIdx() != begin and neighbour.GetAtomicNum() > 1
    ]
    if not left or not right:
        return []
    from rdkit.Chem import rdMolTransforms

    indices = list(conformers) if conformers is not None else list(range(mol.GetNumConformers()))
    seen = set()
    for index in indices:
        try:
            angle = rdMolTransforms.GetDihedralDeg(
                mol.GetConformer(int(index)), left[0], begin, end, right[0]
            )
        except Exception:  # pragma: no cover - a degenerate arrangement
            continue
        if angle is None or not math.isfinite(float(angle)):
            continue
        wrapped = (float(angle) + 360.0) % 360.0
        seen.add(int(wrapped // (360.0 / bins)) % bins)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Ensemble quality over a library
# ---------------------------------------------------------------------------


def ensemble_quality(ensembles: Sequence[ConformerEnsemble]) -> Dict[str, Any]:
    """Fold a library's ensembles into the numbers a report quotes.

    Returns the embed success rate, the mean and median conformer count, the torsion
    coverage **against the ceiling that conformer count allows**, the mean RMSD
    spread, the mean wall time per molecule, the names of the failures, and the
    **attempt accounting**: where the requested conformers went, stage by stage,
    summed over the library.  This is the function a benchmark calls before trusting
    a 3-D score: if half the library failed to embed, every 3-D ranking below it is a
    ranking of the other half — and if the attempts never reached the embedder, every
    ensemble is thin for a reason the caller can fix.
    """
    total = len(ensembles)
    ok = [item for item in ensembles if not item.error]
    failed = [(item.name, item.error) for item in ensembles if item.error]
    counts = [item.n_conformers for item in ok]
    coverages = [item.torsion_coverage for item in ok if item.torsion_bins]
    ceilings = [item.coverage_ceiling for item in ok if item.torsion_bins]
    spreads = [item.rmsd_max for item in ok if item.n_conformers > 1]
    times = [item.seconds for item in ensembles]
    attempts = sum(item.requested for item in ensembles)
    embedded = sum(item.embedded for item in ensembles)
    losses = {
        "embedding": int(attempts - embedded),
        "energy_window": int(sum(item.pruned_energy for item in ensembles)),
        "rmsd_prune": int(sum(item.pruned_rmsd for item in ensembles)),
    }
    dominant = max(losses, key=lambda key: losses[key]) if any(losses.values()) else "none"
    ratios = [
        (item.torsion_coverage / item.coverage_ceiling)
        for item in ok
        if item.torsion_bins and item.coverage_ceiling > 0
    ]
    return {
        "n_molecules": total,
        "n_embedded": len(ok),
        "embed_rate": (len(ok) / total) if total else 0.0,
        "conformers_mean": float(np.mean(counts)) if counts else 0.0,
        "conformers_median": float(np.median(counts)) if counts else 0.0,
        "conformers_min": int(min(counts)) if counts else 0,
        "torsion_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "coverage_ceiling_mean": float(np.mean(ceilings)) if ceilings else 0.0,
        "coverage_fraction_of_ceiling": float(np.mean(ratios)) if ratios else 0.0,
        "rmsd_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "rotors_mean": float(np.mean([item.rotors for item in ensembles])) if ensembles else 0.0,
        "scaled_attempts": sum(1 for item in ensembles if item.scaled),
        "attempts_total": int(attempts),
        "attempts_per_molecule": float(np.mean([item.requested for item in ensembles]))
        if ensembles else 0.0,
        "embedded_total": int(embedded),
        "losses": losses,
        "dominant_loss": dominant,
        "seconds_per_molecule": float(np.mean(times)) if times else 0.0,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# USR: the cheap shape descriptor a pre-filter can afford
# ---------------------------------------------------------------------------


def usr_descriptors(mol, conf_id: int = 0) -> Optional[np.ndarray]:
    """The 12 ultrafast-shape-recognition descriptors of one conformer.

    USR (Ballester and Richards, 2007) describes a shape by the distance
    distribution from four reference points — the centroid, the atom closest to it,
    the atom farthest from it, and the atom farthest from *that* — with the mean,
    the standard deviation and the maximum for each: 12 numbers that can be compared
    in microseconds.  It is deliberately *not* a shape overlap: it is a
    rotation-invariant summary, which is what a pre-filter needs and why its recall,
    not its correlation, is the number to report.

    The 12 numbers are computed here rather than taken from a library so the
    definition is in the source; the ordering is :data:`USR_DESCRIPTORS`.
    """
    require_rdkit()
    if mol.GetNumConformers() == 0:
        return None
    atoms = _heavies(mol)
    if len(atoms) < 2:
        return None
    conf = mol.GetConformer(int(conf_id))
    coords = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in atoms
        ],
        dtype=float,
    )
    centroid = coords.mean(axis=0)
    distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    closest = coords[int(np.argmin(distances_to_centroid))]
    farthest = coords[int(np.argmax(distances_to_centroid))]
    distances_to_farthest = np.linalg.norm(coords - farthest, axis=1)
    farthest2 = coords[int(np.argmax(distances_to_farthest))]
    out: List[float] = []
    for reference in (centroid, closest, farthest, farthest2):
        distances = np.linalg.norm(coords - reference, axis=1)
        out.extend([float(distances.mean()), float(distances.std()), float(distances.max())])
    return np.asarray(out, dtype=float)


def usr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """USR similarity ``1 / (1 + |a - b| / 12)``, in ``(0, 1]``.

    Identical shapes give 1.0.  The normalisation is a convention (the published
    form divides the Manhattan distance by the descriptor count) and it is stated
    here so a threshold means something: 0.8 is "the distance distributions differ
    by about 0.2 on average".
    """
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape or left.size == 0:
        return 0.0
    distance = float(np.abs(left - right).sum() / left.size)
    return 1.0 / (1.0 + distance)

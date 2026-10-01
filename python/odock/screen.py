# SPDX-License-Identifier: GPL-3.0-or-later
"""High-throughput screening: dock a whole library into one or more receptors.

``odock dock`` docks **one** ligand.  A virtual-screening campaign docks
thousands, runs for hours, and gets interrupted.  This module is the pipeline
for that job:

* read a multi-format library (:func:`odock.chem.ligand.read_ligands`), filter it
  with the drug-likeness rules (:mod:`odock.filters`) and report *how many
  molecules each filter removed*;
* prepare and dock every survivor in parallel;
* write one result record per molecule **as it completes** (append + flush), so
  an interrupted run keeps everything it had already done;
* on restart, skip the molecules that are already in the results file
  (resuming is the default when the output exists);
* rank the library, add ligand-efficiency metrics (:mod:`odock.metrics`, when
  available) and the key interacting residues (:mod:`odock.analysis`), write a
  ``--top N`` shortlist as a multi-model PDBQT, and — with ``--consensus`` —
  rescore that shortlist with every force field to show which hits survive a
  change of potential;
* never let one bad molecule kill the run: a failure is one row with a reason,
  counted and reported at the end.

The command line is :func:`odock.cli.cmd_screen`; this module is the API::

    from odock.screen import ScreenConfig, screen_ligands

    summary = screen_ligands(ScreenConfig(
        receptors=["receptor.pdbqt"],
        inputs=["library.sdf"],
        box=BoxSpec(center=(10.0, 20.0, 30.0), size=(22.5, 22.5, 22.5)),
        outdir="results",
        exhaustiveness=8,
    ))
    print(summary.text())

Design notes
------------
**Why threads, not processes.**  The kernel releases the GIL for the duration of
a search, so a :class:`~concurrent.futures.ThreadPoolExecutor` gives real
parallelism without pickling a byte, and every worker keeps the full
:class:`~odock.docking.DockResult` — pose coordinates, PDBQT text, atom order —
that a process pool would have to ship back.  Measured on the bundled 3PTB demo
this is as fast as the kernel's own ``dock_batch`` while giving per-molecule
completion granularity, which is exactly what makes the incremental write and
the per-molecule timeout possible.

**Why the results file is the state.**  There is no database and no side-car
bookkeeping: the results file *is* the record of what has been done, and a
restart reads it back and continues.  The expensive non-docking half of a big
library — filtering and embedding — is cached in ``library/`` beside it, so a
restart costs seconds instead of an hour.

**Why nothing is deleted implicitly.**  Changing the box, the force field or the
receptor while an output directory exists is almost always a mistake: the old
rows are not comparable with the new ones.  The run refuses with a message
instead of mixing two campaigns in one file.  Deleting the old results requires
an explicit ``resume=False``.

**Why the consensus is post-processing.**  ``--consensus`` rescoring the
shortlist changes no docking, so it is deliberately absent from the campaign
hashes: adding it to a finished run is allowed, re-runs nothing, and cannot
invalidate a single row.  That is also why a failure there is a warning — the
campaign is already complete and on disk — and never a failed run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
import warnings
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .prepare import BoxSpec

__all__ = [
    "EXIT_ALL_FAILED",
    "EXIT_INPUT",
    "EXIT_OK",
    "CONSENSUS_SCORINGS",
    "ConsensusReport",
    "FilterStats",
    "LibraryMember",
    "LigandRecord",
    "ScreenConfig",
    "ScreenError",
    "ScreenSummary",
    "check_box_against_receptor",
    "consensus_limit",
    "estimate_cost",
    "estimate_library",
    "ligand_seed",
    "mismatched_seed_rows",
    "read_records",
    "receptor_atoms_in_box",
    "receptor_bounds",
    "recover_campaign_seed",
    "run_consensus",
    "screen_ligands",
]

#: Everything went as well as it could.
EXIT_OK = 0
#: Nothing to do, or the inputs contradict each other (empty library, a box that
#: does not touch the receptor, a results directory belonging to another run).
EXIT_INPUT = 2
#: The library was fine and every docking failed.
EXIT_ALL_FAILED = 3

#: The per-run manifest: the settings the results in this directory belong to.
MANIFEST_NAME = "run.json"
#: The append-only result files.
JSONL_NAME = "results.jsonl"
CSV_NAME = "results.csv"
#: The ranked view written at the end of a run.
SUMMARY_CSV = "summary.csv"
SUMMARY_JSON = "summary.json"
#: The filtered library, and the prepared PDBQT of every survivor.
LIBRARY_CSV = "library.csv"
LIBRARY_DIR = "library"
POSES_DIR = "poses"

#: Names a ``resume=False`` run is allowed to remove, and nothing else: the
#: output directory may well be a directory the user cares about.
_OWNED_FILES = (JSONL_NAME, CSV_NAME, SUMMARY_CSV, SUMMARY_JSON, LIBRARY_CSV, MANIFEST_NAME)
_OWNED_DIRS = (LIBRARY_DIR, POSES_DIR)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

#: The fixed cost of one docking job: parsing the receptor, building the
#: affinity grid, building the system and writing the poses.  The kernel rebuilds
#: the grid for every library member, because every member has its own torsion
#: tree, so this is paid once per molecule and not once per campaign.  Measured
#: at 0.33 s for the 3PTB box (150 920 points), which is what the two constants
#: below reproduce.
_FIXED_SECONDS = 0.10
_FIXED_SECONDS_PER_GRID_POINT = 1.5e-6
#: The search cost per unit of ``exhaustiveness``.  It grows steeply with the
#: number of torsions — benzene is free, an 11-torsion drug costs seconds per
#: run — hence the power law rather than a linear term.  Fitted on the bundled
#: 3PTB receptor with ligands of 0 to 9 torsions (docs/SCREENING.md).
_SEARCH_SECONDS_BASE = 0.003
_SEARCH_SECONDS_PER_ATOM = 0.0025
_SEARCH_TORSION_EXPONENT = 1.6
#: The model is fitted on this many cores; a different count scales it.
_REFERENCE_CORES = 16
#: ``--dry-run`` measures the machine by docking the first molecule, at this
#: many exhaustiveness units at most: four is enough to separate the fixed cost
#: from the search cost, and bounds what a dry run can cost.
_PROBE_EXHAUSTIVENESS = 4


def search_seconds_per_unit(n_atoms: int, n_torsions: int) -> float:
    """The model's cost of one unit of ``exhaustiveness`` for one ligand."""
    torsions = max(0, int(n_torsions))
    return _SEARCH_SECONDS_BASE + _SEARCH_SECONDS_PER_ATOM * max(0, int(n_atoms)) * (
        1.0 + torsions ** _SEARCH_TORSION_EXPONENT
    )


def fixed_seconds(grid_points_value: int) -> float:
    """The model's per-molecule fixed cost (receptor, grid, system, writing)."""
    return _FIXED_SECONDS + _FIXED_SECONDS_PER_GRID_POINT * max(0, int(grid_points_value))


def estimate_cost(
    *,
    n_ligands: int = 1,
    n_atoms: int = 20,
    n_torsions: int = 3,
    grid_points: int = 150_000,
    exhaustiveness: int = 8,
    cores: Optional[int] = None,
) -> float:
    """Estimated wall-clock seconds for docking ``n_ligands`` molecules.

    A fixed receptor/grid cost per molecule plus a search cost that scales with
    ``exhaustiveness`` and steeply with the torsion count.  The *shape* is what
    this model is for: measured per-molecule times spread over a factor of ten
    for the same (atoms, torsions) pair, so the absolute figure is only good to
    about an order of magnitude.  ``--dry-run`` therefore replaces it with a
    measurement (see :func:`probe_cost`) and a running campaign reports an ETA
    from its own measured rate.
    """
    per_unit = search_seconds_per_unit(n_atoms, n_torsions)
    total = max(0, int(n_ligands)) * (
        fixed_seconds(grid_points) + max(0, int(exhaustiveness)) * per_unit
    )
    if cores:
        total *= _REFERENCE_CORES / max(1, int(cores))
    return float(total)


def estimate_library(
    members: Sequence["LibraryMember"],
    *,
    box: Optional[BoxSpec] = None,
    spacing: Optional[float] = None,
    exhaustiveness: int = 8,
    cores: Optional[int] = None,
) -> Dict[str, Any]:
    """Sum :func:`estimate_cost` over a library.

    Returns ``{"seconds", "per_molecule", "grid_points", "n_ligands", "cores",
    "slowest"}`` where ``slowest`` names the member that dominates the estimate —
    the answer to "why does this take so long?" is almost always one flexible
    molecule.
    """
    points = grid_points(box, spacing) if box is not None else 150_000
    total = 0.0
    slowest: Optional[LibraryMember] = None
    slowest_seconds = 0.0
    for member in members:
        seconds = estimate_cost(
            n_ligands=1,
            n_atoms=member.n_atoms or 20,
            n_torsions=member.n_torsions,
            grid_points=points,
            exhaustiveness=exhaustiveness,
            cores=cores,
        )
        total += seconds
        if seconds > slowest_seconds:
            slowest_seconds = seconds
            slowest = member
    n = len(members)
    return {
        "seconds": total,
        "per_molecule": (total / n) if n else 0.0,
        "grid_points": points,
        "n_ligands": n,
        "cores": cores or os.cpu_count() or 1,
        "slowest": slowest.name if slowest is not None else "",
        "slowest_seconds": slowest_seconds,
    }


def grid_points(box: BoxSpec, spacing: Optional[float] = None) -> int:
    """The number of grid samples a box produces, the way the kernel counts it.

    ``n_x * n_y * n_z`` where each axis is the box rounded **up** to whole voxels
    plus one sample — the convention that reproduces the ``grid N points`` figure
    ``odock dock`` prints for the bundled demos exactly (150 920 for 3PTB,
    95 550 for EGFR).
    """
    step = float(spacing or box.spacing)
    count = 1
    for size in box.size:
        count *= max(1, int(math.ceil(float(size) / step)) + 1)
    return int(count)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ScreenError(RuntimeError):
    """A screening run that cannot usefully continue.

    ``code`` is the process exit status: :data:`EXIT_INPUT` for anything the
    user has to fix, :data:`EXIT_ALL_FAILED` when the inputs were fine but every
    docking failed.
    """

    def __init__(self, message: str, code: int = EXIT_INPUT) -> None:
        super().__init__(message)
        self.code = int(code)


def _err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _out(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScreenConfig:
    """Everything one screening run needs.

    The two hashes — :meth:`library_hash` and :meth:`docking_hash` — are what
    makes resuming safe: a results directory belongs to exactly one library and
    one set of docking settings, and a run that finds a different pair refuses to
    mix them instead of quietly appending incomparable numbers.  ``--seed``
    deliberately appears in neither: it changes the search, not the identity of
    the campaign.
    """

    receptors: Sequence[Any]
    inputs: Sequence[Any]
    box: BoxSpec
    outdir: Any

    # -- docking ----------------------------------------------------------
    scoring: str = "vina"
    exhaustiveness: int = 8
    num_poses: int = 9
    min_rmsd: float = 1.0
    energy_range: float = 3.0
    search: Optional[str] = None
    islands: int = 4
    population: int = 32
    generations: int = 20
    use_grid: bool = True
    refine: bool = True

    # -- execution --------------------------------------------------------
    seed: int = 0
    jobs: int = 0
    timeout: Optional[float] = None
    checkpoint_every: int = 20

    # -- library ----------------------------------------------------------
    filters: bool = True
    optimize: bool = True
    prepare_seed: int = 20240101
    limit: Optional[int] = None

    # -- output -----------------------------------------------------------
    top: int = 0
    fmt: str = "jsonl"
    resume: bool = True
    interactions: bool = True
    write_poses: bool = True
    progress: bool = True
    dry_run: bool = False
    allow_box_mismatch: bool = False
    #: Rescore the shortlist with every force field and rank the hits by their
    #: agreement (see :class:`ConsensusReport`).  Post-processing only: it does
    #: not change a single docking, so it can be added to a finished campaign.
    consensus: bool = False
    #: How many molecules the consensus covers; ``0`` means ``top``, else 50.
    consensus_top: int = 0
    #: ``rank``, ``borda`` or ``z`` — how the force fields are combined.
    consensus_method: str = "rank"

    def __post_init__(self) -> None:
        self.receptors = [Path(p) for p in self.receptors]
        self.inputs = [Path(p) for p in self.inputs]
        self.outdir = Path(self.outdir)

    # -- validation -------------------------------------------------------

    def validated(self) -> "ScreenConfig":
        """Return this configuration, or raise :class:`ScreenError`."""
        if not self.receptors:
            raise ScreenError("no receptor given: pass --receptor FILE")
        if not self.inputs:
            raise ScreenError("no library given: pass -i/--input FILE")
        if isinstance(self.exhaustiveness, bool) or int(self.exhaustiveness) < 1:
            raise ScreenError(f"exhaustiveness must be >= 1, got {self.exhaustiveness!r}")
        if int(self.num_poses) < 1:
            raise ScreenError(f"num-poses must be >= 1, got {self.num_poses!r}")
        if float(self.min_rmsd) <= 0:
            raise ScreenError(f"min-rmsd must be positive, got {self.min_rmsd!r}")
        if self.timeout is not None and float(self.timeout) <= 0:
            raise ScreenError(f"timeout must be positive, got {self.timeout!r}")
        if int(self.checkpoint_every) < 1:
            raise ScreenError(
                f"checkpoint-every must be >= 1, got {self.checkpoint_every!r}"
            )
        if int(self.top) < 0:
            raise ScreenError(f"top must be >= 0, got {self.top!r}")
        if self.limit is not None and int(self.limit) < 1:
            raise ScreenError(f"limit must be >= 1, got {self.limit!r}")
        if self.fmt not in ("jsonl", "csv"):
            raise ScreenError(f"unknown result format {self.fmt!r}; use jsonl or csv")
        if self.consensus_method not in ("rank", "borda", "z"):
            raise ScreenError(
                f"unknown consensus method {self.consensus_method!r}; "
                "use rank, borda or z"
            )
        if int(self.consensus_top) < 0:
            raise ScreenError(f"consensus-top must be >= 0, got {self.consensus_top!r}")
        for path in list(self.receptors) + list(self.inputs):
            if not path.exists():
                raise ScreenError(f"no such file: {path}")
            if not path.is_file():
                raise ScreenError(f"not a file: {path}")
        if not isinstance(self.box, BoxSpec):
            raise ScreenError(f"box must be a BoxSpec, got {type(self.box).__name__}")
        return self

    # -- identity ---------------------------------------------------------

    def library_hash(self) -> str:
        """Fingerprint of everything that decides *which* molecules are docked.

        The input signatures (path, size, mtime), the filter switch and the
        preparation settings.  Deliberately independent of the docking seed, so
        re-running with another ``--seed`` reuses the prepared library.
        """
        return _digest(
            {
                "inputs": [_file_signature(p) for p in self.inputs],
                "filters": bool(self.filters),
                "optimize": bool(self.optimize),
                "prepare_seed": int(self.prepare_seed),
            }
        )

    def docking_hash(self) -> str:
        """Fingerprint of everything that decides *what the numbers mean*.

        The campaign seed belongs here: every molecule's docking seed is derived
        from it (see :func:`ligand_seed`), so two runs with different seeds are
        two different campaigns even though the box and the force field are the
        same.  Resuming one as the other would put incomparable numbers in one
        file — and would silently dock the molecules the two runs share under two
        different seeds.
        """
        return _digest(
            {
                "receptors": [_file_signature(p) for p in self.receptors],
                "box": [list(self.box.center), list(self.box.size), float(self.box.spacing)],
                "scoring": self.scoring,
                "exhaustiveness": int(self.exhaustiveness),
                "num_poses": int(self.num_poses),
                "min_rmsd": float(self.min_rmsd),
                "energy_range": float(self.energy_range),
                "search": self.search,
                "islands": int(self.islands),
                "population": int(self.population),
                "generations": int(self.generations),
                "use_grid": bool(self.use_grid),
                "refine": bool(self.refine),
                "seed": int(self.seed),
            }
        )

    def docking_settings(self) -> Dict[str, Any]:
        """The :meth:`docking_hash` inputs as a plain dictionary, for a diff.

        Kept next to :meth:`docking_hash` so the refusal message can name the
        setting that changed instead of only saying that "something" did.
        """
        return {
            "receptors": [str(p) for p in self.receptors],
            "box": self.box.as_dict(),
            "scoring": self.scoring,
            "exhaustiveness": int(self.exhaustiveness),
            "num_poses": int(self.num_poses),
            "min_rmsd": float(self.min_rmsd),
            "energy_range": float(self.energy_range),
            "search": self.search,
            "islands": int(self.islands),
            "population": int(self.population),
            "generations": int(self.generations),
            "use_grid": bool(self.use_grid),
            "refine": bool(self.refine),
            "seed": int(self.seed),
        }

    # -- paths ------------------------------------------------------------

    @property
    def results_name(self) -> str:
        return JSONL_NAME if self.fmt == "jsonl" else CSV_NAME

    @property
    def results_path(self) -> Path:
        return self.outdir / self.results_name

    def receptor_names(self) -> List[str]:
        """Short, unique-per-run names for the receptors; used in keys and paths.

        A panel usually holds ``trypsin/receptor.pdbqt``, ``mutant/receptor.pdbqt``:
        the file stems collide, so a colliding stem is qualified with its parent
        directory (``trypsin_receptor``), and only a genuine ambiguity falls back
        to a path digest.  These names are the ``receptor`` column of the results
        file, so they are stable and ride in the resume key.
        """
        stems = [_stem(path) for path in self.receptors]
        counts: Dict[str, int] = {}
        for stem in stems:
            counts[stem] = counts.get(stem, 0) + 1
        out: List[str] = []
        used: set = set()
        for path, stem in zip(self.receptors, stems):
            name = stem
            if counts[stem] > 1:
                parent = path.parent.name
                if parent and parent not in (".", ".."):
                    name = f"{_safe_name(parent)}_{stem}"
            if name in used:
                name = f"{name}_{_digest(str(path))[:6]}"
            used.add(name)
            out.append(name)
        return out

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable view, for the manifest and the summary."""
        return {
            "receptors": [str(p) for p in self.receptors],
            "inputs": [str(p) for p in self.inputs],
            "outdir": str(self.outdir),
            "box": self.box.as_dict(),
            "scoring": self.scoring,
            "exhaustiveness": int(self.exhaustiveness),
            "num_poses": int(self.num_poses),
            "min_rmsd": float(self.min_rmsd),
            "energy_range": float(self.energy_range),
            "search": self.search,
            "islands": int(self.islands),
            "population": int(self.population),
            "generations": int(self.generations),
            "use_grid": bool(self.use_grid),
            "refine": bool(self.refine),
            "seed": int(self.seed),
            "jobs": int(self.jobs),
            "timeout": None if self.timeout is None else float(self.timeout),
            "filters": bool(self.filters),
            "optimize": bool(self.optimize),
            "prepare_seed": int(self.prepare_seed),
            "top": int(self.top),
            "format": self.fmt,
            "interactions": bool(self.interactions),
            "write_poses": bool(self.write_poses),
            "consensus": bool(self.consensus),
            "consensus_top": int(self.consensus_top),
            "consensus_method": self.consensus_method,
            "library_hash": self.library_hash(),
            "docking_hash": self.docking_hash(),
        }


def _file_signature(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"path": str(path), "size": int(st.st_size), "mtime": int(st.st_mtime)}


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _stem(path: Path) -> str:
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(path).stem).strip("._-")
    return name or "receptor"


def _safe_name(text: str, limit: int = 48) -> str:
    out = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("._-")
    return (out or "ligand")[:limit]


def _draw_seed() -> int:
    """A seed to record, like ``odock dock``'s ``--seed 0``."""
    return random.SystemRandom().randrange(1, 2**31 - 1)


def ligand_seed(base_seed: int, receptor: str, key: str) -> int:
    """A stable per-(receptor, ligand) docking seed derived from the run seed.

    ``--seed`` makes a whole campaign reproducible; deriving one seed per
    molecule from ``crc32(receptor|key)`` also makes a molecule dock the same way
    whether it was docked in the original run or after a restart.  Python's
    ``hash()`` is salted per process and would break exactly that; ``crc32`` is
    not.

    The mapping is invertible (:func:`recover_campaign_seed`), which is what lets
    a campaign whose manifest was lost still be recognised from its own rows.
    """
    return int((int(base_seed) + _seed_mix(receptor, key)) % _SEED_MODULUS) + 1


#: The modulus of the per-molecule seed derivation, kept in one place because
#: :func:`recover_campaign_seed` has to invert it.
_SEED_MODULUS = 2**31 - 1


def _seed_mix(receptor: str, key: str) -> int:
    """The per-(receptor, ligand) part of :func:`ligand_seed`."""
    return zlib.crc32(f"{receptor}|{key}".encode("utf-8")) & 0xFFFFFFFF


def recover_campaign_seed(records: Sequence["LigandRecord"]) -> Optional[int]:
    """The campaign seed implied by a results file, or ``None``.

    Every row stores the per-molecule seed it was docked with, and that seed is
    ``base + crc32(receptor|ligand)`` modulo ``2**31 - 1``; subtracting the mix
    recovers the campaign seed.  That is what makes a results file whose
    ``run.json`` was lost resumable *without* guessing: the rows say which
    campaign they belong to, so a restart continues it instead of silently
    docking the same molecules under a second seed.

    Returns ``None`` when no row carries a seed, or when the rows disagree — two
    different seeds in one file mean the file is already a mixture.
    """
    candidates = set()
    for record in records:
        if not record.seed:
            continue
        candidates.add((int(record.seed) - 1 - _seed_mix(record.receptor, record.ligand)) % _SEED_MODULUS)
        if len(candidates) > 1:
            return None
    return candidates.pop() if candidates else None


def mismatched_seed_rows(
    records: Sequence["LigandRecord"], base_seed: int
) -> List[str]:
    """The rows that were docked under a different campaign seed.

    A file whose rows disagree about the seed is a mixture of two campaigns (the
    state an older release could leave behind by accepting ``--seed 8`` on a
    ``--seed 7`` campaign).  Naming the offenders turns a silently corrupt file
    into a message.
    """
    out: List[str] = []
    for record in records:
        if not record.seed:
            continue
        expected = ligand_seed(base_seed, record.receptor, record.ligand)
        if int(record.seed) != expected:
            out.append(record.ligand)
    return out


# ---------------------------------------------------------------------------
# Library members
# ---------------------------------------------------------------------------


@dataclass
class LibraryMember:
    """One molecule of the library, before or after filtering and preparation.

    A member with ``error`` set was read but could not be prepared; a member with
    ``passed`` false was removed by the drug-likeness filters.  Only a member
    that is *both* passed and error-free is docked.
    """

    #: Stable identity inside the library: ``"<file name>#<record number>"``.
    key: str
    index: int
    name: str
    source: str
    #: Prepared ligand PDBQT, relative to the output directory.
    pdbqt_file: str = ""
    n_atoms: int = 0
    n_heavy: int = 0
    n_torsions: int = 0
    #: RDKit's ``Descriptors.MolWt``, the same number the Lipinski check uses.
    molecular_weight: Optional[float] = None
    properties: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    failed_filters: List[str] = field(default_factory=list)
    passed: bool = True
    error: str = ""

    @property
    def dockable(self) -> bool:
        return bool(self.passed and not self.error and self.pdbqt_file)

    @property
    def first_failed(self) -> str:
        """The first filter this molecule failed, in Lipinski/Veber/PAINS order."""
        for name in ("Lipinski", "Veber", "PAINS"):
            if name in self.failed_filters:
                return name
        return self.failed_filters[0] if self.failed_filters else ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "index": self.index,
            "name": self.name,
            "source": self.source,
            "pdbqt_file": self.pdbqt_file,
            "n_atoms": self.n_atoms,
            "n_heavy": self.n_heavy,
            "n_torsions": self.n_torsions,
            "molecular_weight": self.molecular_weight,
            "properties": {k: float(v) for k, v in self.properties.items()},
            "violations": list(self.violations),
            "failed_filters": list(self.failed_filters),
            "passed": bool(self.passed),
            "error": self.error,
        }


@dataclass
class FilterStats:
    """How many molecules each pre-filter removed.

    ``per_filter`` counts every molecule that failed that filter, so the numbers
    overlap (a molecule can fail Lipinski *and* PAINS).  ``exclusive`` attributes
    each removed molecule to the **first** filter it failed, in the order
    Lipinski, Veber, PAINS; those numbers add up to ``n_removed``, which is what
    a funnel wants.
    """

    n_read: int = 0
    n_removed: int = 0
    n_prepare_failed: int = 0
    per_filter: Dict[str, int] = field(default_factory=dict)
    exclusive: Dict[str, int] = field(default_factory=dict)
    #: Records the file held but RDKit could not parse, so they never became
    #: molecules.  The funnel counts what parsed, and says so.
    unreadable: List[str] = field(default_factory=list)
    #: The full record of what was removed, for the ``--dry-run`` report.
    removed: List[Dict[str, str]] = field(default_factory=list)
    order: List[str] = field(default_factory=lambda: ["Lipinski", "Veber", "PAINS"])

    @property
    def n_kept(self) -> int:
        return self.n_read - self.n_removed

    @property
    def n_dockable(self) -> int:
        return self.n_kept - self.n_prepare_failed

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_read": self.n_read,
            "n_kept": self.n_kept,
            "n_removed": self.n_removed,
            "n_prepare_failed": self.n_prepare_failed,
            "n_dockable": self.n_dockable,
            "n_unreadable": len(self.unreadable),
            "unreadable": list(self.unreadable),
            "per_filter": dict(self.per_filter),
            "exclusive": dict(self.exclusive),
        }

    def lines(self) -> List[str]:
        out = [
            f"filters: {self.n_read} read -> {self.n_kept} kept "
            f"({self.n_removed} removed by Lipinski/Veber/PAINS)"
        ]
        for name in self.order:
            failed = self.per_filter.get(name, 0)
            own = self.exclusive.get(name, 0)
            out.append(f"  {name:<9} {failed:>6} failed this filter ({own} first here)")
        if self.n_prepare_failed:
            out.append(f"  {'prepare':<9} {self.n_prepare_failed:>6} could not be prepared")
        if self.unreadable:
            # "read" counts what RDKit could parse; anything else is not a
            # molecule and cannot be filtered, docked or counted in the funnel.
            out.append(
                f"  note: {len(self.unreadable)} record(s) could not be parsed by RDKit "
                "and are NOT counted above"
            )
            for message in self.unreadable[:3]:
                out.append(f"    {message[:160]}")
        return out


def filter_stats(members: Sequence[LibraryMember], *, filters: bool = True) -> FilterStats:
    """Fold a read library into the filter funnel."""
    stats = FilterStats(n_read=len(members))
    for member in members:
        if member.error:
            stats.n_prepare_failed += 1
        if not member.passed:
            stats.n_removed += 1
            for name in member.failed_filters:
                stats.per_filter[name] = stats.per_filter.get(name, 0) + 1
            first = member.first_failed
            if first:
                stats.exclusive[first] = stats.exclusive.get(first, 0) + 1
            stats.removed.append(
                {
                    "name": member.name,
                    "source": member.source,
                    "first_failed": first,
                    "violations": "; ".join(member.violations),
                }
            )
    if not filters:
        stats.per_filter = {}
        stats.exclusive = {}
        stats.removed = []
        stats.n_removed = 0
    return stats


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass
class LigandRecord:
    """One molecule × one receptor: the row that is appended to the results file."""

    receptor: str
    ligand: str
    name: str
    source: str = ""
    index: int = 0
    #: ``ok``, ``failed`` or ``timeout``.
    status: str = "ok"
    affinity: Optional[float] = None
    n_heavy: int = 0
    n_atoms: int = 0
    n_torsions: int = 0
    #: RDKit's ``Descriptors.MolWt``, so a consumer does not have to dig it out of
    #: :attr:`properties`.
    molecular_weight: Optional[float] = None
    ligand_efficiency: Optional[float] = None
    le_source: str = ""
    key_residues: str = ""
    n_interactions: int = 0
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    in_box: bool = True
    seed: int = 0
    elapsed: float = 0.0
    grid_points: int = 0
    poses: List[Dict[str, Any]] = field(default_factory=list)
    pose_file: str = ""
    error: str = ""
    violations: List[str] = field(default_factory=list)
    properties: Dict[str, float] = field(default_factory=dict)

    #: Column order of the CSV form, kept explicit so a CSV is diffable and a
    #: resumed run reads back exactly what it wrote.
    CSV_COLUMNS: Tuple[str, ...] = (
        "receptor", "ligand", "name", "source", "index", "status", "affinity",
        "ligand_efficiency", "le_source", "molecular_weight", "n_heavy", "n_atoms",
        "n_torsions", "MW", "LogP", "tpsa", "RotB", "key_residues",
        "n_interactions", "in_box", "seed", "elapsed", "grid_points", "pose_file",
        "violations", "error",
    )

    def as_dict(self, *, full: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "receptor": self.receptor,
            "ligand": self.ligand,
            "name": self.name,
            "source": self.source,
            "index": int(self.index),
            "status": self.status,
            "affinity": self.affinity,
            "n_heavy": int(self.n_heavy),
            "n_atoms": int(self.n_atoms),
            "n_torsions": int(self.n_torsions),
            "molecular_weight": self.molecular_weight,
            "ligand_efficiency": self.ligand_efficiency,
            "le_source": self.le_source,
            "key_residues": self.key_residues,
            "n_interactions": int(self.n_interactions),
            "in_box": bool(self.in_box),
            "seed": int(self.seed),
            "elapsed": round(float(self.elapsed), 4),
            "grid_points": int(self.grid_points),
            "pose_file": self.pose_file,
            "error": self.error,
            "violations": list(self.violations),
            "properties": {k: float(v) for k, v in self.properties.items()},
        }
        if full:
            data["interactions"] = list(self.interactions)
            data["poses"] = list(self.poses)
        return data

    def as_row(self) -> List[Any]:
        props = self.properties
        return [
            self.receptor,
            self.ligand,
            self.name,
            self.source,
            self.index,
            self.status,
            _num(self.affinity),
            _num(self.ligand_efficiency),
            self.le_source,
            _num(self.molecular_weight),
            self.n_heavy,
            self.n_atoms,
            self.n_torsions,
            _num(props.get("MW")),
            _num(props.get("LogP")),
            _num(props.get("tPSA")),
            _num(props.get("RotB")),
            self.key_residues,
            self.n_interactions,
            "1" if self.in_box else "0",
            self.seed,
            f"{self.elapsed:.3f}",
            self.grid_points,
            self.pose_file,
            "; ".join(str(v) for v in self.violations),
            self.error,
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LigandRecord":
        """Rebuild a record from a JSON object or a CSV row."""
        props: Dict[str, float] = {}
        raw_props = data.get("properties")
        if isinstance(raw_props, dict):
            props = {str(k): float(v) for k, v in raw_props.items() if _is_number(v)}
        for column, key in (("MW", "MW"), ("LogP", "LogP"), ("tpsa", "tPSA"), ("RotB", "RotB")):
            value = data.get(column)
            if _is_number(value):
                props.setdefault(key, float(value))
        violations = data.get("violations") or []
        if isinstance(violations, str):
            violations = [v for v in (part.strip() for part in violations.split(";")) if v]
        interactions = data.get("interactions") or []
        poses = data.get("poses") or []
        return cls(
            receptor=str(data.get("receptor", "")),
            ligand=str(data.get("ligand", "")),
            name=str(data.get("name", "")),
            source=str(data.get("source", "")),
            index=int(float(data.get("index") or 0)),
            status=str(data.get("status") or "ok"),
            affinity=_opt_float(data.get("affinity")),
            n_heavy=int(float(data.get("n_heavy") or 0)),
            n_atoms=int(float(data.get("n_atoms") or 0)),
            n_torsions=int(float(data.get("n_torsions") or 0)),
            molecular_weight=_opt_float(data.get("molecular_weight")),
            ligand_efficiency=_opt_float(data.get("ligand_efficiency")),
            le_source=str(data.get("le_source") or ""),
            key_residues=str(data.get("key_residues") or ""),
            n_interactions=int(float(data.get("n_interactions") or 0)),
            interactions=list(interactions) if isinstance(interactions, list) else [],
            in_box=_opt_bool(data.get("in_box"), True),
            seed=int(float(data.get("seed") or 0)),
            elapsed=float(data.get("elapsed") or 0.0),
            grid_points=int(float(data.get("grid_points") or 0)),
            poses=list(poses) if isinstance(poses, list) else [],
            pose_file=str(data.get("pose_file") or ""),
            error=str(data.get("error") or ""),
            violations=[str(v) for v in violations],
            properties=props,
        )


def _num(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _opt_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


# ---------------------------------------------------------------------------
# Reading and writing the results file
# ---------------------------------------------------------------------------


def read_records(path: Any) -> List[LigandRecord]:
    """Read a results file written by :func:`screen_ligands`.

    The format follows the suffix.  A truncated final line — exactly what a
    killed process leaves behind — is skipped rather than raised, because "the
    last write was cut off" is the normal state of a crashed screening run and
    the record it was writing is simply redone.
    """
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".csv":
        out: List[LigandRecord] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                out.append(LigandRecord.from_dict(dict(row)))
        return out
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict):
            out.append(LigandRecord.from_dict(data))
    return out


class _ResultWriter:
    """Append one record at a time, flushed (and periodically fsynced).

    The point of the class is subtle and central: a screening run may be killed
    at any moment, and the contract is that every molecule reported as done is on
    disk.  ``flush`` costs a syscall and is therefore done per record;
    ``os.fsync`` pushes the data past the operating-system cache as well, which
    is much more expensive, so it happens every ``checkpoint_every`` records --
    and that is also when ``on_checkpoint`` runs, so the run manifest is refreshed
    on the same cadence as the data it describes.
    """

    def __init__(
        self,
        path: Path,
        fmt: str,
        checkpoint_every: int = 20,
        on_checkpoint: Optional[Any] = None,
    ) -> None:
        self.path = Path(path)
        self.fmt = fmt
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.on_checkpoint = on_checkpoint
        self._since_sync = 0
        self._handle = None
        self.written = 0

    def __enter__(self) -> "_ResultWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        if self.fmt == "jsonl":
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        else:
            self._handle = self.path.open("a", encoding="utf-8", newline="")
            if new:
                csv.writer(self._handle).writerow(LigandRecord.CSV_COLUMNS)
                self._handle.flush()
        return self

    def write(self, record: LigandRecord) -> None:
        if self._handle is None:
            raise RuntimeError("the writer is not open")
        if self.fmt == "jsonl":
            self._handle.write(json.dumps(record.as_dict(), default=str) + "\n")
        else:
            csv.writer(self._handle).writerow(record.as_row())
        self._handle.flush()
        self.written += 1
        self._since_sync += 1
        if self._since_sync >= self.checkpoint_every:
            self.checkpoint()

    def checkpoint(self) -> None:
        """Push the buffered records all the way to the storage device."""
        if self._handle is None:
            return
        self._handle.flush()
        try:
            os.fsync(self._handle.fileno())
        except (OSError, ValueError):  # pragma: no cover - platform dependent
            pass
        self._since_sync = 0
        if self.on_checkpoint is not None:
            try:
                self.on_checkpoint()
            except Exception as exc:  # pragma: no cover - the run must survive it
                _err(f"odock screen: warning: could not refresh the run manifest ({exc})")

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self.checkpoint()
            self._handle.close()
            self._handle = None


# ---------------------------------------------------------------------------
# The run manifest
# ---------------------------------------------------------------------------


class _Manifest:
    """``run.json``: which campaign a results directory belongs to."""

    def __init__(self, path: Path, data: Dict[str, Any]) -> None:
        self.path = Path(path)
        self.data = data

    @classmethod
    def load(cls, path: Path) -> Optional["_Manifest"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return cls(path, data) if isinstance(data, dict) else None

    def save(self) -> None:
        """Write ``run.json`` atomically.

        A manifest that is half-written when the process dies is worse than no
        manifest at all, so the new document goes to a temporary file in the same
        directory and is moved into place with :func:`os.replace`, which is atomic
        on Windows and POSIX alike.
        """
        self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        text = json.dumps(self.data, indent=2, default=str) + "\n"
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, self.path)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def _clear_owned_outputs(outdir: Path) -> List[str]:
    """Remove the files this program owns inside ``outdir``; never anything else."""
    removed: List[str] = []
    for name in _OWNED_FILES:
        target = outdir / name
        if target.is_file():
            target.unlink()
            removed.append(name)
    for name in _OWNED_DIRS:
        target = outdir / name
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(name + "/")
    return removed


def _settings_diff(stored: Any, current: Dict[str, Any]) -> List[str]:
    """``"seed: 7 -> 8"`` for every docking setting that changed.

    A refusal that names the field is worth ten that say "the settings changed":
    a user who has just added ``--seed 8`` to a resumed command can see at once
    that this is the reason.
    """
    if not isinstance(stored, dict):
        return []
    out: List[str] = []
    for key, value in current.items():
        if key not in stored:
            continue
        before = stored[key]
        if isinstance(before, (dict, list)) or isinstance(value, (dict, list)):
            same = json.dumps(before, sort_keys=True, default=str) == json.dumps(
                value, sort_keys=True, default=str
            )
        else:
            same = before == value
        if not same:
            out.append(f"{key}: {_short(before)} -> {_short(value)}")
        if len(out) >= 6:
            break
    return out


def _short(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= 60 else text[:57] + "..."


def _validate_resume(manifest: _Manifest, config: ScreenConfig) -> None:
    if manifest.get("dry_run"):
        # A dry run docks nothing, so its settings constrain nothing; only the
        # seed and the prepared library are worth keeping from it.
        return
    stored_library = manifest.get("library_hash")
    stored_docking = manifest.get("docking_hash")
    if stored_library and stored_library != config.library_hash():
        raise ScreenError(
            "the library inputs changed since this output directory was written "
            f"(stored {stored_library}, now {config.library_hash()}).\n"
            "       The molecules already in the results file would no longer be the "
            "ones being docked.\n"
            "       Write to a new --out directory, or pass --no-resume to discard "
            f"the results in {config.outdir} (they are not deleted otherwise)."
        )
    if stored_docking and stored_docking != config.docking_hash():
        differences = _settings_diff(
            manifest.get("docking_settings"), config.docking_settings()
        )
        detail = ("\n       changed: " + "; ".join(differences)) if differences else ""
        raise ScreenError(
            "the docking settings changed since this output directory was written "
            f"(stored {stored_docking}, now {config.docking_hash()}).{detail}\n"
            "       Mixing two campaigns in one results file would produce "
            "incomparable numbers — and, for a different seed, would dock the shared "
            "molecules under a second seed.\n"
            "       Write to a new --out directory, or pass --no-resume to discard "
            f"the results in {config.outdir} (they are not deleted otherwise)."
        )


# ---------------------------------------------------------------------------
# The library: reading, filtering, preparing, caching
# ---------------------------------------------------------------------------


def _load_library(
    config: ScreenConfig, manifest: Optional[_Manifest]
) -> Tuple[List[LibraryMember], List[LibraryMember], FilterStats, str]:
    """Return ``(all_members, to_dock, stats, source)``.

    ``library.csv`` always describes the **whole** library — the filter funnel and
    the descriptors are properties of the library, not of the run — while
    ``--limit`` decides how many of the survivors are docked.  A cached library is
    reused only when it belongs to the same inputs, filters and preparation
    settings *and* every prepared PDBQT it references is still on disk: filtering
    and embedding a large library takes far longer than docking a handful of
    molecules, so not redoing it is the difference between a restart that costs
    seconds and one that costs an hour.
    """
    members: Optional[List[LibraryMember]] = None
    unreadable: List[str] = []
    source = "read"
    if (
        manifest is not None
        and config.resume
        and manifest.get("library_hash") == config.library_hash()
    ):
        cached = _read_library_csv(config.outdir / LIBRARY_CSV)
        if cached is not None:
            missing = [
                m.key
                for m in cached
                if m.dockable and not (config.outdir / m.pdbqt_file).is_file()
            ]
            if not missing:
                members = cached
                source = "cache"
                _out(
                    f"library: {len(cached)} molecule(s) of which "
                    f"{sum(1 for m in cached if m.dockable)} dockable, restored from "
                    f"{LIBRARY_CSV}"
                )
            else:
                _err(
                    "odock screen: the prepared library is incomplete "
                    f"({len(missing)} PDBQT file(s) missing, first: {missing[0]}); "
                    "reading the library again"
                )
    if members is None:
        members, unreadable = _read_library(config)
        _write_library_csv(config, members)
    stats = filter_stats(members, filters=config.filters)
    if unreadable:
        stats.unreadable = list(unreadable)
    else:
        cached_notes = manifest.get("library", {}).get("unreadable") if manifest else None
        if isinstance(cached_notes, list):
            stats.unreadable = [str(note) for note in cached_notes]
    dockable = [m for m in members if m.dockable]
    if config.limit is not None and len(dockable) > int(config.limit):
        _out(
            f"--limit {int(config.limit)}: docking {int(config.limit)} of the "
            f"{len(dockable)} dockable molecule(s)"
        )
        dockable = dockable[: int(config.limit)]
    return members, dockable, stats, source


def _read_library(config: ScreenConfig) -> Tuple[List[LibraryMember], List[str]]:
    """Read, filter and prepare every input file, one file at a time.

    Returns ``(members, unreadable)``: the second is the reader's complaints about
    records that never became molecules (a broken SMILES line, an unparsable SDF
    record), which the funnel reports rather than hiding.
    """
    members: List[LibraryMember] = []
    unreadable: List[str] = []
    for source in config.inputs:
        try:
            found, notes = _read_input_members(source, config, base=len(members))
        except ScreenError:
            raise
        except Exception as exc:
            raise ScreenError(f"cannot read {source}: {exc}")
        members.extend(found)
        unreadable.extend(notes)
    if not members:
        raise ScreenError(
            "no molecule could be read from the given library "
            f"({', '.join(str(p) for p in config.inputs)})"
        )
    dockable = [m for m in members if m.dockable]
    if not dockable:
        removed = sum(1 for m in members if not m.passed)
        if removed == len(members):
            raise ScreenError(
                f"the drug-likeness filters removed all {len(members)} molecule(s); "
                "run with --no-filter to dock them anyway, or check the library"
            )
        raise ScreenError(
            f"none of the {len(members)} molecule(s) could be prepared; nothing to dock"
        )
    failures = [(m.name, m.error) for m in members if m.error]
    for name, error in failures[:5]:
        _err(f"odock screen: {name}: {error}")
    if len(failures) > 5:
        _err(f"odock screen: ... and {len(failures) - 5} more preparation failure(s)")
    return members, unreadable


def _read_input_members(
    source: Path, config: ScreenConfig, base: int = 0
) -> Tuple[List[LibraryMember], List[str]]:
    """Every molecule of one input file, filtered and (when kept) prepared.

    ``base`` is how many molecules the earlier input files contributed: the
    library position, the resume key and the prepared-file name are all derived
    from it, so two input files can never collide — not even when they share a
    file name or contain a molecule with the same title.

    The second element of the result lists the reader's complaints: a record that
    RDKit cannot parse is not a molecule and cannot be docked, but it *was* in
    the file, so it is reported rather than silently dropped.
    """
    raw: List[Tuple[str, Any, Optional[str]]] = []
    unreadable: List[str] = []
    if source.suffix.lower() in (".pdbqt", ".pdbqt.gz"):
        for index, block in enumerate(_pdbqt_blocks(source.read_text(encoding="utf-8", errors="replace"))):
            raw.append((_pdbqt_name(block, source, index), _pdbqt_mol(block), block))
    else:
        from .chem.ligand import read_ligands

        try:
            # `read_ligands` warns about, and skips, a record it cannot parse;
            # capturing the warnings keeps that fact in the funnel.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                mols = read_ligands(source, embed=False)
        except ValueError as exc:
            raise ScreenError(f"cannot read {source}: {exc}")
        unreadable = [
            f"{source.name}: {warning.message}"
            for warning in caught
            if issubclass(warning.category, UserWarning)
        ]
        raw = [(_mol_name(mol, source, i), mol, None) for i, mol in enumerate(mols)]

    unfiltered = 0
    members: List[LibraryMember] = []
    for offset, (name, mol, block) in enumerate(raw):
        index = base + offset
        member = LibraryMember(
            key=f"{source.name}#{index + 1}",
            index=index,
            name=name,
            source=str(source),
        )
        try:
            if mol is None:
                if config.filters:
                    # A PDBQT library can be docked without RDKit, but then the
                    # filters have nothing to look at.  Say so rather than letting
                    # a molecule pass a check that never ran.
                    unfiltered += 1
            else:
                # The descriptors are computed even with --no-filter: the ranking
                # table, the CSV and the JSON record all report them, and a reader
                # who asked to skip the *verdict* did not ask to lose the numbers.
                # Only the verdict itself is conditional.
                verdict = _drug_like(mol)
                member.properties = {
                    k: float(v)
                    for k, v in (verdict.get("properties") or {}).items()
                    if _is_number(v)
                }
                member.molecular_weight = member.properties.get("MW")
                if config.filters:
                    member.passed = bool(verdict.get("passed", True))
                    member.violations = [str(v) for v in verdict.get("violations", [])]
                    member.failed_filters = [
                        result.name
                        for result in verdict.get("results", [])
                        if not getattr(result, "passed", True)
                    ]
            if member.passed:
                text = block if block is not None else _prepare(member.name, mol, config)
                member.n_atoms, member.n_torsions = _pdbqt_counts(text)
                member.n_heavy = _pdbqt_heavy(text)
                if not member.n_atoms and mol is not None:
                    member.n_atoms = int(mol.GetNumAtoms())
                target = (
                    config.outdir
                    / LIBRARY_DIR
                    / f"{index + 1:06d}_{_safe_name(member.name)}.pdbqt"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                member.pdbqt_file = _relpath(config.outdir, target)
        except Exception as exc:
            member.error = f"{type(exc).__name__}: {exc}"
        members.append(member)
    if unfiltered:
        _err(
            f"odock screen: warning: {unfiltered} molecule(s) of {source.name} could not "
            "be read with RDKit, so the drug-likeness filters did not run on them "
            "(the PDBQT itself is docked as it is)"
        )
    return members, unreadable


def _prepare(name: str, mol: Any, config: ScreenConfig) -> str:
    """Prepare one RDKit molecule into ligand PDBQT text."""
    from .prepare import prepare_ligand

    if mol is None:
        raise ValueError("the molecule could not be read")
    _, text, _report = prepare_ligand(
        mol, None, name=name, optimize=bool(config.optimize), seed=int(config.prepare_seed)
    )
    return text


def _drug_like(mol: Any) -> Dict[str, Any]:
    from .filters import drug_like

    return drug_like(mol)


def _mol_name(mol: Any, source: Path, index: int) -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return f"{source.stem}_{index + 1}"


def _pdbqt_blocks(text: str) -> List[str]:
    """The ``MODEL`` blocks of a PDBQT document (the whole text when there are none)."""
    lines = text.splitlines()
    if not any(line.startswith("MODEL") for line in lines):
        return [text] if text.strip() else []
    blocks: List[str] = []
    current: Optional[List[str]] = None
    for line in lines:
        if line.startswith("MODEL"):
            current = [line]
            continue
        if line.startswith("ENDMDL"):
            if current is not None:
                current.append(line)
                blocks.append("\n".join(current) + "\n")
            current = None
            continue
        if current is not None:
            current.append(line)
    if current:
        blocks.append("\n".join(current) + "\n")
    return blocks


_PDBQT_NAME = re.compile(r"^REMARK\s+OpenDocking ligand preparation:\s*(.+?)\s*$")


def _pdbqt_name(block: str, source: Path, index: int) -> str:
    for line in block.splitlines():
        match = _PDBQT_NAME.match(line)
        if match:
            return match.group(1)
        if line.startswith(("ATOM", "HETATM")):
            break
    return f"{source.stem}_{index + 1}"


def _pdbqt_mol(block: str) -> Any:
    """The RDKit view of a ligand PDBQT block, or ``None`` when it cannot be read."""
    try:
        from .chem.ligand import read_ligands

        mols = read_ligands(block, fmt="pdbqt", embed=False)
        return mols[0] if mols else None
    except Exception:
        return None


def _pdbqt_counts(text: str) -> Tuple[int, int]:
    """``(n_atoms, n_torsions)`` of a ligand PDBQT document."""
    atoms = 0
    torsions = 0
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms += 1
        elif line.startswith("TORSDOF"):
            try:
                torsions = int(line.split()[1])
            except (IndexError, ValueError):
                torsions = 0
    return atoms, torsions


def _pdbqt_heavy(text: str) -> int:
    """Number of non-hydrogen atoms of a ligand PDBQT."""
    count = 0
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        symbol = line[76:79].strip() if len(line) >= 79 else ""
        if not symbol:
            symbol = line[12:16].strip()[:1]
        if symbol.upper() not in ("H", "HD"):
            count += 1
    return count


def _relpath(outdir: Path, target: Path) -> str:
    """A stored path is always relative to the output directory, with ``/``.

    Forward slashes keep a results file readable on every platform and are
    accepted by :class:`pathlib.Path` on Windows as well, so a ``--out``
    directory can be copied between machines and still resolve.
    """
    try:
        return target.relative_to(outdir).as_posix()
    except ValueError:  # pragma: no cover - defensive
        return target.as_posix()


def _round4(value: Any) -> Any:
    """Four decimals, which is more than any descriptor or metric needs."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return ""
    return round(number, 4)


_LIBRARY_COLUMNS = (
    "index", "key", "name", "source", "passed", "failed_filters", "first_failed",
    "violations", "MW", "LogP", "HBD", "HBA", "RotB", "tPSA", "n_atoms",
    "n_heavy", "n_torsions", "molecular_weight", "pdbqt_file", "error",
    "properties_json",
)


def _write_library_csv(config: ScreenConfig, members: Sequence[LibraryMember]) -> None:
    """Write the whole filtered library, including the molecules the filters removed.

    A screening funnel is only defensible if you can show *what* was dropped and
    *why*, so every molecule that was read appears here with its descriptors, its
    violations and — for the survivors — the path of the PDBQT that was docked.
    """
    path = config.outdir / LIBRARY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_LIBRARY_COLUMNS)
        for member in members:
            props = member.properties
            writer.writerow(
                [
                    member.index,
                    member.key,
                    member.name,
                    member.source,
                    "1" if member.passed else "0",
                    ";".join(member.failed_filters),
                    member.first_failed,
                    "; ".join(str(v) for v in member.violations),
                    _round4(props.get("MW")),
                    _round4(props.get("LogP")),
                    _round4(props.get("HBD")),
                    _round4(props.get("HBA")),
                    _round4(props.get("RotB")),
                    _round4(props.get("tPSA")),
                    _round4(member.molecular_weight),
                    member.n_atoms,
                    member.n_heavy,
                    member.n_torsions,
                    member.pdbqt_file,
                    member.error,
                    # The six named columns above are for a human reading the CSV;
                    # the JSON keeps every descriptor the filters produced, so a
                    # record rebuilt from the cache is identical to a fresh one.
                    json.dumps(
                        {k: round(float(v), 6) for k, v in props.items()}, sort_keys=True
                    ),
                ]
            )


def _read_library_csv(path: Path) -> Optional[List[LibraryMember]]:
    """Read ``library.csv`` back; ``None`` when it is missing or unusable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError):
        return None
    if not rows:
        return None
    members: List[LibraryMember] = []
    for row in rows:
        props: Dict[str, float] = {}
        raw_json = row.get("properties_json")
        if raw_json:
            try:
                decoded = json.loads(raw_json)
            except ValueError:
                decoded = None
            if isinstance(decoded, dict):
                props = {str(k): float(v) for k, v in decoded.items() if _is_number(v)}
        for column, key in (
            ("MW", "MW"), ("LogP", "LogP"), ("HBD", "HBD"), ("HBA", "HBA"),
            ("RotB", "RotB"), ("tPSA", "tPSA"),
        ):
            if _is_number(row.get(column)):
                props[key] = float(row[column])
        failed = [name for name in str(row.get("failed_filters") or "").split(";") if name]
        violations = [
            part.strip() for part in str(row.get("violations") or "").split(";") if part.strip()
        ]
        members.append(
            LibraryMember(
                key=str(row.get("key") or ""),
                index=int(float(row.get("index") or 0)),
                name=str(row.get("name") or ""),
                source=str(row.get("source") or ""),
                pdbqt_file=str(row.get("pdbqt_file") or ""),
                n_atoms=int(float(row.get("n_atoms") or 0)),
                n_heavy=int(float(row.get("n_heavy") or 0)),
                n_torsions=int(float(row.get("n_torsions") or 0)),
                molecular_weight=props.get("MW"),
                properties=props,
                violations=violations,
                failed_filters=failed,
                passed=_opt_bool(row.get("passed"), False),
                error=str(row.get("error") or ""),
            )
        )
    return members


# ---------------------------------------------------------------------------
# Receptor / box sanity
# ---------------------------------------------------------------------------


def receptor_bounds(
    text: str,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """The axis-aligned bounding box of a receptor PDBQT, or ``None``.

    Parsed straight from the coordinate columns: no RDKit, no kernel, and it
    works on a file the kernel would reject for some unrelated reason.
    """
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    seen = False
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue
        seen = True
        for axis in range(3):
            lo[axis] = min(lo[axis], xyz[axis])
            hi[axis] = max(hi[axis], xyz[axis])
    if not seen:
        return None
    return (lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])


def receptor_atoms_in_box(text: str, box: BoxSpec) -> int:
    """How many receptor atoms lie inside the search box (exactly, not by bbox).

    This is the sharp test for a wrong coordinate frame: the 3PTB site box holds
    335 of its 1994 atoms and the EGFR box 164 of its 2985, while either box
    applied to the *other* receptor holds exactly zero.
    """
    count = 0
    half = tuple(size / 2.0 for size in box.size)
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue
        if all(abs(xyz[i] - box.center[i]) <= half[i] for i in range(3)):
            count += 1
    return count


def check_box_against_receptor(
    box: BoxSpec, text: str, label: str = "receptor"
) -> Optional[str]:
    """Return a human-readable complaint when `box` cannot belong to `text`.

    The test is deliberately the *sharp* one — not "does the box overlap the
    receptor's bounding box" but "does the box contain a receptor atom at all".
    A search box that holds no receptor atom samples empty space, and every
    molecule would come back with a huge positive energy; in practice it means
    the box and the receptor come from different coordinate frames (a receptor
    that was re-oriented, or a box copied from another structure).

    A box that contains the whole receptor is not an error: blind docking does
    that on purpose.  ``None`` means the pair is plausible.
    """
    bounds = receptor_bounds(text)
    if bounds is None:
        return None
    inside = receptor_atoms_in_box(text, box)
    if inside:
        return None
    lo, hi = bounds
    total = sum(1 for line in text.splitlines() if line.startswith(("ATOM", "HETATM")))
    return (
        f"no receptor atom of {label} lies inside the search box "
        f"(0 of {total} atoms; box centre "
        f"({box.center[0]:.2f}, {box.center[1]:.2f}, {box.center[2]:.2f}), size "
        f"({box.size[0]:.1f}, {box.size[1]:.1f}, {box.size[2]:.1f}) A; the receptor "
        f"spans x {lo[0]:.1f}..{hi[0]:.1f}, y {lo[1]:.1f}..{hi[1]:.1f}, "
        f"z {lo[2]:.1f}..{hi[2]:.1f} A)"
    )


# ---------------------------------------------------------------------------
# Ligand efficiency and interactions (optional modules, called defensively)
# ---------------------------------------------------------------------------


def _ligand_efficiency(affinity: Optional[float], n_heavy: int) -> Tuple[Optional[float], str]:
    """Ligand efficiency in kcal/mol per heavy atom, and where it came from.

    :mod:`odock.metrics` is the authority — it is written in parallel with this
    module, so it is imported lazily and every failure falls back to the
    two-line formula ``LE = -affinity / n_heavy``, labelled ``builtin``.
    """
    if affinity is None or n_heavy <= 0:
        return None, ""
    try:
        from . import metrics  # type: ignore

        function = getattr(metrics, "ligand_efficiency", None)
        if function is not None:
            value = float(function(float(affinity), int(n_heavy)))
            return (None if not math.isfinite(value) else value), "odock.metrics"
    except Exception:
        pass
    value = -float(affinity) / float(n_heavy)
    return (value if math.isfinite(value) else None), "builtin"


def _receptor_structure(receptor_mol: Any) -> Any:
    """A receptor structure for interaction profiling, normalised once.

    ``analysis._structure`` is documented as idempotent precisely so a caller
    profiling many poses against one receptor pays for the receptor's bond
    perception once; that turns ~60 ms of work per ligand into ~8 ms.  It is a
    private name, so its absence (an older or reworked analysis module) is not an
    error — the public entry points are then used as they are.
    """
    try:
        from . import analysis

        normalise = getattr(analysis, "_structure", None)
        if normalise is not None:
            return normalise(receptor_mol)
    except Exception:
        pass
    return receptor_mol


def profile_pose(
    receptor: Any, ligand_mol: Any
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """Key residues, the interaction count and the list, for one pose.

    Returns ``("", 0, [])`` when :mod:`odock.analysis` is unavailable or cannot
    handle the structures: interactions are a *reporting* nicety and must never
    turn a completed docking into a failed one.
    """
    try:
        from . import analysis

        interactions = analysis.profile_interactions(receptor, ligand_mol)
        summary = analysis.interaction_summary(interactions, receptor, ligand_mol)
        rows = [
            {
                "kind": str(getattr(item, "kind", "")),
                "subtype": str(getattr(item, "subtype", "")),
                "receptor_atom": int(getattr(item, "a", -1)),
                "ligand_atom": int(getattr(item, "b", -1)),
                "distance": float(getattr(item, "distance", float("nan"))),
                "detail": str(getattr(item, "detail", "")),
            }
            for item in interactions
        ]
        return str(summary or ""), len(interactions), rows
    except Exception:
        return "", 0, []


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def format_eta(seconds: Optional[float]) -> str:
    """``m:ss`` (or ``h:mm:ss``) for an ETA; ``--:--`` when it is unknown."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(seconds + 0.5)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _bar(fraction: float, width: int = 22) -> str:
    fraction = min(1.0, max(0.0, float(fraction)))
    return "#" * int(round(fraction * width)) + "-" * (width - int(round(fraction * width)))


class _Progress:
    """A live progress line: done/total, rate, ETA, failures, best affinity.

    On a terminal it is one carriage-return line that is rewritten; when stderr
    is a pipe or a log file — which is what a cluster job looks like — it becomes
    a line per update instead, so the log stays readable and greppable.
    """

    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = max(1, int(total))
        self.enabled = bool(enabled)
        self.tty = False
        try:
            self.tty = bool(sys.stderr.isatty())
        except Exception:  # pragma: no cover - replaced stream
            self.tty = False
        self.interval = 0.25 if self.tty else 1.0
        self._last = 0.0
        self._last_text = ""

    def line(
        self,
        done: int,
        *,
        ok: int,
        failed: int,
        rate: float,
        eta: Optional[float],
        best: Optional[float],
    ) -> str:
        best_text = "   --  " if best is None else f"{best:7.3f}"
        return (
            f"  [{_bar(done / self.total)}] {done:>5d}/{self.total:<5d} "
            f"{100.0 * done / self.total:5.1f}%  ok {ok:<5d} failed {failed:<4d} "
            f"{rate:5.2f}/s  ETA {format_eta(eta):>7}  best {best_text}"
        )

    def update(
        self,
        done: int,
        *,
        ok: int,
        failed: int,
        rate: float,
        eta: Optional[float],
        best: Optional[float],
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        finished = done >= self.total
        if not force and not finished and done > 1 and (now - self._last) < self.interval:
            return
        text = self.line(done, ok=ok, failed=failed, rate=rate, eta=eta, best=best)
        if self.tty:
            sys.stderr.write("\r" + text)
        else:
            sys.stderr.write(text + "\n")
        sys.stderr.flush()
        self._last = now
        self._last_text = text

    def finish(self) -> None:
        if self.enabled and self.tty and self._last_text:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._last_text = ""


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class ScreenSummary:
    """What a screening run did, ready to print and to serialise."""

    config: ScreenConfig
    records: List[LigandRecord] = field(default_factory=list)
    members: List[LibraryMember] = field(default_factory=list)
    stats: FilterStats = field(default_factory=FilterStats)
    seed: int = 0
    elapsed: float = 0.0
    interrupted: bool = False
    library_source: str = "read"
    estimate: Optional[Dict[str, Any]] = None
    results_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    top_paths: Dict[str, Path] = field(default_factory=dict)
    #: ``(name, error)`` of the molecules that could not even be prepared.
    library_failures: List[Tuple[str, str]] = field(default_factory=list)
    #: Per-receptor force-field consensus of the shortlist (``--consensus``).
    consensus: Dict[str, ConsensusReport] = field(default_factory=dict)
    dry_run: bool = False
    #: Measured seconds per docking, once at least one molecule is done.
    measured_per_molecule: Optional[float] = None
    #: How many dockings *this* run started and finished (a resumed run's earlier
    #: rows are in :attr:`records` but not in :attr:`elapsed`).
    completed_this_run: int = 0

    # -- accessors --------------------------------------------------------

    @property
    def receptor_names(self) -> List[str]:
        return self.config.receptor_names()

    def for_receptor(self, receptor: str) -> List[LigandRecord]:
        return [r for r in self.records if r.receptor == receptor]

    def n_ok(self, receptor: Optional[str] = None) -> int:
        return sum(
            1
            for r in self.records
            if r.status == "ok" and (receptor is None or r.receptor == receptor)
        )

    def n_timeout(self, receptor: Optional[str] = None) -> int:
        return sum(
            1
            for r in self.records
            if r.status == "timeout" and (receptor is None or r.receptor == receptor)
        )

    def n_failed(self, receptor: Optional[str] = None) -> int:
        return sum(
            1
            for r in self.records
            if r.status not in ("ok", "timeout")
            and (receptor is None or r.receptor == receptor)
        )

    def ranked(self, receptor: str, limit: Optional[int] = None) -> List[LigandRecord]:
        """Successful records of one receptor, best affinity first."""
        rows = [r for r in self.for_receptor(receptor) if r.status == "ok"]
        rows.sort(key=lambda r: (math.inf if r.affinity is None else float(r.affinity)))
        return rows if limit is None else rows[: int(limit)]

    @property
    def failures(self) -> List[LigandRecord]:
        return [r for r in self.records if r.status != "ok"]

    @property
    def exit_code(self) -> int:
        if self.dry_run:
            return EXIT_OK
        return EXIT_OK if self.n_ok() else EXIT_ALL_FAILED

    # -- reporting --------------------------------------------------------

    def table(self, receptor: str, limit: Optional[int] = None) -> str:
        """The ranked summary table of one receptor."""
        rows = self.ranked(receptor, limit)
        if not rows:
            return "(no successful docking)"
        has_le = any(r.ligand_efficiency is not None for r in rows)
        has_residues = any(r.key_residues for r in rows)
        headers = ["rank", "name", "affinity", "MW"]
        if has_le:
            headers.append("LE")
        headers += ["tors", "poses"]
        if has_residues:
            headers.append("key residues")
        body: List[List[str]] = []
        for i, record in enumerate(rows):
            cells = [
                str(i + 1),
                record.name[:28],
                "--" if record.affinity is None else f"{record.affinity:.3f}",
                _fmt(record.properties.get("MW"), "{:.1f}"),
            ]
            if has_le:
                cells.append(
                    "--" if record.ligand_efficiency is None else f"{record.ligand_efficiency:.2f}"
                )
            cells += [str(record.n_torsions), str(len(record.poses))]
            if has_residues:
                cells.append(record.key_residues or "-")
            body.append(cells)
        return _table(headers, body)

    def text(self, *, limit: Optional[int] = None) -> str:
        """The whole human-readable report."""
        lines: List[str] = []
        receptors = self.config.receptors
        scope = f"{len(self.members)} molecule(s)"
        if self.stats.n_dockable != len(self.members):
            scope += f" (of {self.stats.n_dockable} dockable, {self.stats.n_read} read)"
        lines.append(f"OpenDocking screen — {len(receptors)} receptor(s) x {scope}")
        if self.results_path is not None:
            lines.append(f"results: {self.results_path}")
        lines.append(f"seed: {self.seed}")
        lines.extend(self.stats.lines())
        if self.library_failures:
            lines.append(
                f"library failures: {len(self.library_failures)} molecule(s) could not "
                "be prepared and were not docked (see library.csv)"
            )
        if self.dry_run:
            estimate = self.estimate or {}
            lines.append("")
            lines.append("dry run — nothing was docked")
            lines.append(
                "estimated cost: {:.0f} s ({}) for {} molecule(s) x {} receptor(s), "
                "about {:.2f} s per docking, {} core(s) assumed".format(
                    float(estimate.get("seconds", 0.0)),
                    format_eta(estimate.get("seconds")),
                    len(self.members),
                    len(receptors),
                    float(estimate.get("per_molecule", 0.0)),
                    int(estimate.get("cores", 1)),
                )
            )
            if estimate.get("slowest"):
                lines.append(
                    "dominant molecule: {} (~{:.0f} s of the total)".format(
                        estimate["slowest"], float(estimate.get("slowest_seconds", 0.0))
                    )
                )
            if estimate.get("source") == "probe":
                lines.append(
                    "measured: {} took {:.2f} s at exhaustiveness {} ({:.3f} s per "
                    "unit)".format(
                        estimate.get("probe_name", "the first molecule"),
                        float(estimate.get("probe_seconds", 0.0)),
                        int(estimate.get("probe_exhaustiveness", 1)),
                        float(estimate.get("per_unit_seconds", 0.0)),
                    )
                )
            lines.append(
                "the kernel rebuilds the affinity grid for every molecule; the "
                "estimate scales one measured docking by the size of the library"
            )
            if self.measured_per_molecule:
                lines.append(
                    f"measured on this machine: {self.measured_per_molecule:.2f} s per "
                    "docking"
                )
            return "\n".join(lines)

        for path, name in zip(receptors, self.config.receptor_names()):
            rows = self.for_receptor(name)
            ok = self.n_ok(name)
            timeouts = self.n_timeout(name)
            errors = self.n_failed(name)
            lines.append("")
            lines.append(f"=== {path} ===")
            lines.append(
                f"{ok} of {len(rows)} docked"
                + (f", {errors} failed" if errors else "")
                + (f", {timeouts} timed out" if timeouts else "")
            )
            lines.append(self.table(name, limit))
            shown = 0
            for record in rows:
                if record.status == "ok":
                    continue
                if shown >= 3:
                    remaining = errors + timeouts - shown
                    if remaining > 0:
                        lines.append(f"  ... and {remaining} more (see the results file)")
                    break
                lines.append(f"  {record.name[:40]}: {record.status}: {record.error}")
                shown += 1
            if self.top_paths.get(name):
                lines.append(f"top {self.config.top} shortlist: {self.top_paths[name]}")
            report = self.consensus.get(name)
            if report is not None:
                lines.append("")
                if report.error:
                    lines.append(f"consensus: unavailable ({report.error})")
                    continue
                lines.append(
                    f"consensus over the top {len(report.rows)} hit(s) "
                    f"({report.method}, {', '.join(report.scorings)}):"
                )
                lines.append(report.agreement_line())
                lines.append(report.table())
                if report.csv_path is not None:
                    lines.append(f"consensus table: {report.csv_path}")
        if self.summary_path is not None:
            lines.append("")
            lines.append(f"summary: {self.summary_path}")
        return "\n".join(lines)

    def as_dict(self, *, full: bool = False) -> Dict[str, Any]:
        return {
            "version": 1,
            "receptors": [str(p) for p in self.config.receptors],
            "inputs": [str(p) for p in self.config.inputs],
            "config": self.config.as_dict(),
            "seed": self.seed,
            "elapsed": self.elapsed,
            "interrupted": bool(self.interrupted),
            "library_source": self.library_source,
            "library": self.stats.as_dict(),
            "library_failures": [
                {"name": name, "error": error} for name, error in self.library_failures
            ],
            "counts": {
                "library": len(self.members),
                "docked": self.n_ok(),
                "failed": self.n_failed(),
                "timeout": self.n_timeout(),
            },
            "completed_this_run": self.completed_this_run,
            "estimate": self.estimate,
            "measured_per_molecule": self.measured_per_molecule,
            "results": str(self.results_path) if self.results_path else None,
            "top": {k: str(v) for k, v in self.top_paths.items()},
            "consensus": {k: v.as_dict() for k, v in self.consensus.items()},
            "records": [r.as_dict(full=full) for r in self.records],
        }

    def write_summary(self, path: Optional[Path] = None) -> Path:
        """Write ``summary.json`` and ``summary.csv``; returns the JSON path."""
        target = Path(path) if path is not None else self.config.outdir / SUMMARY_JSON
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.as_dict(full=False), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        csv_path = self.config.outdir / SUMMARY_CSV
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(LigandRecord.CSV_COLUMNS)
            for receptor in self.receptor_names:
                for row in self.ranked(receptor):
                    writer.writerow(tuple(row.as_row()))
                for row in self.for_receptor(receptor):
                    if row.status != "ok":
                        writer.writerow(tuple(row.as_row()))
        self.summary_path = target
        return target


def _fmt(value: Any, spec: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    try:
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """A left-aligned, dash-underlined text table (the same shape as the CLI's)."""
    if not rows:
        return "(nothing)"
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def render(cells: Sequence[Any]) -> str:
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [render(headers), "  ".join("-" * w for w in widths)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Docking one molecule
# ---------------------------------------------------------------------------


class _Runner:
    """The thread pool, plus the ability to stop everything already running.

    A docking run cannot be interrupted from the outside except through the
    kernel's cancel token, which is asynchronous: the search notices it at the
    next Monte-Carlo step and returns the poses it has so far.  Holding the live
    engines here is what lets Ctrl-C come back in milliseconds instead of after
    the slowest molecule in flight.
    """

    def __init__(self, jobs: int) -> None:
        self.jobs = max(1, int(jobs))
        self.pool = ThreadPoolExecutor(max_workers=self.jobs, thread_name_prefix="odock-screen")
        self._lock = threading.Lock()
        self._engines: Dict[int, Any] = {}
        self._counter = 0
        self.stopping = threading.Event()

    def register(self, engine: Any) -> int:
        with self._lock:
            self._counter += 1
            token = self._counter
            self._engines[token] = engine
        return token

    def unregister(self, token: int) -> None:
        with self._lock:
            self._engines.pop(token, None)

    def stop(self) -> None:
        """Cancel every docking in flight and refuse to start new ones."""
        self.stopping.set()
        with self._lock:
            engines = list(self._engines.values())
        for engine in engines:
            try:
                engine.cancel()
            except Exception:  # pragma: no cover - defensive
                pass

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def _dock_member(
    member: LibraryMember,
    config: ScreenConfig,
    receptor_text: str,
    seed: int,
    runner: Optional[_Runner],
) -> Dict[str, Any]:
    """Dock one molecule.  Runs in a worker thread and never raises.

    The return value is a status plus, on success, a
    :class:`~odock.docking.DockResult`.  Every exception — an unparsable ligand, a
    box the kernel rejects, a search that found nothing — becomes a status,
    because one bad molecule in a library of 10 000 must cost one row, not the
    run.
    """
    started = time.perf_counter()
    engine = None
    token = -1
    timer: Optional[threading.Timer] = None
    try:
        if runner is not None and runner.stopping.is_set():
            return {"status": "interrupted", "result": None, "elapsed": 0.0, "error": ""}
        ligand_text = _read_ligand_text(config, member)
        from .docking import build_engine, result_from_engine

        engine = build_engine(
            receptor_text,
            ligand_text,
            config.box,
            scoring=config.scoring,
            exhaustiveness=int(config.exhaustiveness),
            num_poses=int(config.num_poses),
            seed=int(seed),
            use_grid=bool(config.use_grid),
            refine=bool(config.refine),
            min_rmsd=float(config.min_rmsd),
            energy_range=float(config.energy_range),
            search=config.search,
            islands=int(config.islands),
            population=int(config.population),
            generations=int(config.generations),
        )
        if runner is not None:
            token = runner.register(engine)
        if config.timeout:
            timer = threading.Timer(float(config.timeout), engine.cancel)
            timer.daemon = True
            timer.start()
        raw = engine.run()
        elapsed = time.perf_counter() - started
        if raw.get("cancelled"):
            if runner is not None and runner.stopping.is_set():
                return {"status": "interrupted", "result": None, "elapsed": elapsed, "error": ""}
            return {
                "status": "timeout",
                "result": None,
                "elapsed": elapsed,
                "error": f"cancelled after {elapsed:.1f}s by the per-molecule timeout "
                         f"({float(config.timeout):g}s)",
            }
        result = result_from_engine(
            engine,
            raw,
            elapsed=elapsed,
            box=config.box,
            receptor_pdbqt=receptor_text,
            ligand_pdbqt=ligand_text,
            energy_range=float(config.energy_range),
        )
        if not result.poses:
            return {
                "status": "failed",
                "result": None,
                "elapsed": elapsed,
                "error": "the search produced no pose (box in the wrong place or too small?)",
            }
        return {"status": "ok", "result": result, "elapsed": elapsed, "error": ""}
    except Exception as exc:
        return {
            "status": "failed",
            "result": None,
            "elapsed": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if timer is not None:
            timer.cancel()
        if runner is not None and token > 0:
            runner.unregister(token)


def _read_ligand_text(config: ScreenConfig, member: LibraryMember) -> str:
    path = config.outdir / member.pdbqt_file
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ScreenError(f"the prepared ligand {path} disappeared: {exc}")


def probe_cost(
    config: ScreenConfig,
    member: LibraryMember,
    receptor_text: str,
    receptor_name: str,
    base_seed: int,
) -> Dict[str, Any]:
    """Measure this machine by docking one real molecule, and extrapolate.

    A model of this kernel's cost cannot be better than an order of magnitude:
    two ligands with the same atom and torsion counts can differ threefold, and
    the machine's own load moves the figure again.  Measuring is cheap by
    comparison — one docking out of thousands — so ``--dry-run`` docks the first
    molecule (at most :data:`_PROBE_EXHAUSTIVENESS` units of exhaustiveness),
    separates the fixed cost from the search cost with the model, and scales the
    search part to the exhaustiveness actually asked for.

    Returns an empty dict when the probe itself cannot run; the caller then keeps
    the model's estimate.
    """
    e_probe = max(1, min(int(config.exhaustiveness), _PROBE_EXHAUSTIVENESS))
    points = grid_points(config.box)
    fixed = fixed_seconds(points)
    try:
        from .docking import build_engine

        ligand_text = _read_ligand_text(config, member)
        engine = build_engine(
            receptor_text,
            ligand_text,
            config.box,
            scoring=config.scoring,
            exhaustiveness=e_probe,
            num_poses=int(config.num_poses),
            seed=ligand_seed(base_seed, receptor_name, member.key),
            use_grid=bool(config.use_grid),
            refine=bool(config.refine),
            min_rmsd=float(config.min_rmsd),
            energy_range=float(config.energy_range),
            search=config.search,
            islands=int(config.islands),
            population=int(config.population),
            generations=int(config.generations),
        )
        started = time.perf_counter()
        engine.run()
        elapsed = time.perf_counter() - started
    except Exception as exc:
        _err(f"odock screen: the dry-run probe could not run ({exc}); using the cost model")
        return {}
    per_unit = max(0.0, elapsed - fixed) / e_probe
    one_at_full = fixed + per_unit * int(config.exhaustiveness)
    return {
        "source": "probe",
        "probe_name": member.name,
        "probe_exhaustiveness": e_probe,
        "probe_seconds": elapsed,
        "per_unit_seconds": per_unit,
        "per_molecule": one_at_full,
    }


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------


def screen_ligands(config: ScreenConfig) -> ScreenSummary:
    """Run one screening campaign and return its :class:`ScreenSummary`.

    The order of operations is the order a user can interrupt safely:

    1. validate the configuration and the receptors, and check the box against
       the receptor (a wrong coordinate frame is refused, not docked);
    2. read, filter and prepare the library — cached in ``library/`` so a restart
       does not redo it;
    3. write ``run.json`` **before the first docking**, so a process that is killed
       mid-campaign still leaves a manifest that names the campaign its rows
       belong to (and refresh it at every checkpoint);
    4. read the results file back and skip what is already done;
    5. dock the rest, writing and flushing one record as each molecule finishes;
    6. rank, write the summary and the shortlist, the consensus, and report.
    """
    config = config.validated()
    config.outdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    manifest_path = config.outdir / MANIFEST_NAME
    manifest = _Manifest.load(manifest_path)
    if not config.resume:
        # --no-resume means "discard and start over", whether or not a manifest
        # survived: it removes only the files this program owns.
        removed = _clear_owned_outputs(config.outdir)
        if removed:
            _err(f"odock screen: --no-resume removed {', '.join(removed)} from {config.outdir}")
        manifest = None
    if manifest is not None:
        _validate_resume(manifest, config)

    seed = int(config.seed) or (
        int(manifest.get("seed")) if manifest is not None and manifest.get("seed") else 0
    )
    if not seed and manifest is None:
        # No manifest and no --seed: the rows themselves say which campaign this
        # directory holds (see recover_campaign_seed), so the restart continues
        # that campaign instead of docking everything again under a new seed.
        rows_on_disk = read_records(config.results_path)
        recovered = recover_campaign_seed(rows_on_disk)
        if recovered:
            seed = recovered
            _out(f"campaign seed recovered from the results file: {seed}")
        elif rows_on_disk and any(row.seed for row in rows_on_disk):
            raise ScreenError(
                f"{config.results_path.name} already mixes more than one campaign "
                "seed; it cannot be continued safely.\n"
                "       Write to a new --out directory, or pass --no-resume to discard "
                f"the results in {config.outdir}."
            )
    if not seed:
        seed = _draw_seed()

    receptors = _load_receptors(config)
    _check_boxes(config, receptors)
    all_members, dockable, stats, library_source = _load_library(config, manifest)

    estimate = estimate_library(
        dockable,
        box=config.box,
        exhaustiveness=int(config.exhaustiveness),
        cores=_job_count(config.jobs),
    )
    # A previous campaign in this directory measured the real rate, which beats
    # any model; the estimate is replaced once this run has one of its own.
    measured = manifest.get("seconds_per_molecule") if manifest is not None else None
    summary = ScreenSummary(
        config=config,
        members=dockable,
        stats=stats,
        seed=seed,
        library_source=library_source,
        estimate=estimate,
        measured_per_molecule=float(measured) if measured else None,
        library_failures=[(m.name, m.error) for m in all_members if m.error],
        dry_run=bool(config.dry_run),
        results_path=None if config.dry_run else config.results_path,
    )
    n_molecules = len(dockable)
    n_receptors = len(receptors)
    # `estimate_library` prices one receptor; every receptor is screened with the
    # whole library, so the total is per-receptor × receptors and the per-docking
    # figure is the per-receptor figure over the number of molecules.
    per_receptor = float(estimate.get("seconds") or 0.0)
    if summary.measured_per_molecule:
        per_receptor = summary.measured_per_molecule * n_molecules
        estimate["measured_per_molecule"] = summary.measured_per_molecule
    estimate["per_receptor"] = per_receptor
    estimate["seconds"] = per_receptor * n_receptors
    estimate["per_molecule"] = per_receptor / max(1, n_molecules)

    if config.dry_run:
        _out("")
        _out(
            f"probing this machine with one real docking "
            f"({dockable[0].name}, exhaustiveness "
            f"{min(int(config.exhaustiveness), _PROBE_EXHAUSTIVENESS)}) ..."
        )
        probe = probe_cost(config, dockable[0], receptors[0][2], receptors[0][1], seed)
        if probe:
            estimate.update(probe)
            estimate["per_receptor"] = float(probe["per_molecule"]) * n_molecules
            estimate["seconds"] = estimate["per_receptor"] * n_receptors
        summary.elapsed = time.monotonic() - started
        _report_dry_run(summary, receptors)
        _save_manifest(manifest_path, config, summary, dockable, stats)
        return summary

    names = config.receptor_names()
    summary.records = _resume_records(config, summary, manifest, names, base_seed=seed)
    done = {(r.receptor, r.ligand) for r in summary.records}
    if summary.records:
        _out(
            f"resume: {len(summary.records)} molecule x receptor row(s) already in "
            f"{config.results_path.name}"
        )

    # The manifest goes to disk *before* the first docking and is refreshed at
    # every checkpoint.  A process killed mid-campaign therefore leaves a
    # manifest that names the campaign, and the restart can tell that the rows on
    # disk are its own; without this the rows were invisible and every molecule
    # was docked a second time.
    refresh = lambda: _save_manifest(  # noqa: E731 - a one-line callback
        manifest_path, config, summary, dockable, stats, phase="running"
    )
    refresh()

    runner = _Runner(_job_count(config.jobs))
    try:
        with _ResultWriter(
            config.results_path, config.fmt, config.checkpoint_every, on_checkpoint=refresh
        ) as writer:
            for path, name, text in receptors:
                pending = [m for m in dockable if (name, m.key) not in done]
                if not pending:
                    _out(f"=== {path}: all {len(dockable)} molecule(s) already done ===")
                    continue
                finished = _screen_receptor(
                    config,
                    summary,
                    runner,
                    writer,
                    receptor_name=name,
                    receptor_text=text,
                    receptor_label=str(path),
                    members=pending,
                    done_before=len(dockable) - len(pending),
                    base_seed=seed,
                )
                if finished is None:
                    summary.interrupted = True
                    break
                # One refresh per receptor keeps the manifest's counts close to
                # the truth even between checkpoints (a panel of receptors, or a
                # library smaller than --checkpoint-every).
                refresh()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        runner.stop()
        summary.interrupted = True
    finally:
        runner.shutdown()

    summary.elapsed = time.monotonic() - started
    if summary.completed_this_run:
        # A throughput figure: wall time per molecule at the concurrency this run
        # used.  It is what the next --dry-run in this directory should predict,
        # so it is deliberately not divided by the number of workers.
        summary.measured_per_molecule = summary.elapsed / summary.completed_this_run
    # The shortlist and the consensus are post-processing: they run after the
    # clock would otherwise stop, so the elapsed time is taken again afterwards
    # (and a resumed run that only post-processes reports that work too).
    _finalise(summary, receptors)
    summary.elapsed = time.monotonic() - started
    summary.write_summary()
    _save_manifest(manifest_path, config, summary, dockable, stats, interrupted=summary.interrupted)
    if summary.interrupted:
        _err("")
        _err(
            f"interrupted: the {len(summary.records)} docking(s) already finished are saved in "
            f"{config.results_path}; run the same command again to continue"
        )
    return summary


def _job_count(jobs: int) -> int:
    return max(1, int(jobs)) if jobs else max(1, os.cpu_count() or 1)


def _load_receptors(config: ScreenConfig) -> List[Tuple[Path, str, str]]:
    """``(path, unique name, PDBQT text)`` for every receptor."""
    out: List[Tuple[Path, str, str]] = []
    names = config.receptor_names()
    for path, name in zip(config.receptors, names):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ScreenError(f"cannot read the receptor {path}: {exc}")
        if not text.strip():
            raise ScreenError(f"the receptor {path} is empty")
        out.append((path, name, text))
    return out


def _check_boxes(config: ScreenConfig, receptors: Sequence[Tuple[Path, str, str]]) -> None:
    problems = []
    for path, _name, text in receptors:
        complaint = check_box_against_receptor(config.box, text, label=str(path))
        if complaint:
            problems.append(complaint)
            continue
        inside = receptor_atoms_in_box(text, config.box)
        if inside < 5:
            _err(
                f"odock screen: warning: only {inside} atom(s) of {path} lie inside the "
                "search box; a box this empty usually means the wrong coordinate frame"
            )
    if not problems:
        return
    message = "\n       ".join(problems)
    if config.allow_box_mismatch:
        _err(f"odock screen: warning: {message}")
        _err("       continuing because --allow-box-mismatch was given")
        return
    raise ScreenError(
        "the search box does not match the receptor:\n       "
        + message
        + "\n       check that the receptor and the box come from the same structure "
        "(same coordinate frame); pass --allow-box-mismatch to dock anyway."
    )


def _screen_receptor(
    config: ScreenConfig,
    summary: ScreenSummary,
    runner: _Runner,
    writer: _ResultWriter,
    *,
    receptor_name: str,
    receptor_text: str,
    receptor_label: str,
    members: Sequence[LibraryMember],
    done_before: int,
    base_seed: int,
) -> Optional[int]:
    """Dock every pending molecule of one receptor; ``None`` when interrupted."""
    total = len(members) + done_before
    _out("")
    _out(
        f"=== {receptor_label}: {len(members)} to dock of {total} "
        f"({done_before} already done) ==="
    )
    _out(f"    box {config.box}")
    progress = _Progress(total, config.progress)
    cached = _load_receptor_for_interactions(config, receptor_label)
    window = max(1, min(len(members), runner.jobs * 2 + 2))
    started = time.monotonic()
    ok = summary.n_ok(receptor_name)
    failed = summary.n_failed(receptor_name) + summary.n_timeout(receptor_name)
    best_values = [r.affinity for r in summary.ranked(receptor_name) if r.affinity is not None]
    best = min(best_values) if best_values else None
    done_here = 0
    next_index = 0
    # The seed is decided once, here, and carried to both the job and the record.
    inflight: Dict[Future, Tuple[LibraryMember, int]] = {}

    def submit(member: LibraryMember) -> None:
        seed = ligand_seed(base_seed, receptor_name, member.key)
        future = runner.pool.submit(_dock_member, member, config, receptor_text, seed, runner)
        inflight[future] = (member, seed)

    while next_index < len(members) or inflight:
        while len(inflight) < window and next_index < len(members):
            submit(members[next_index])
            next_index += 1
        if not inflight:
            break
        finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
        for future in finished:
            member, seed = inflight.pop(future)
            if future.cancelled():
                continue
            try:
                outcome = future.result()
            except Exception as exc:  # pragma: no cover - _dock_member never raises
                outcome = {"status": "failed", "result": None, "elapsed": 0.0, "error": str(exc)}
            if outcome["status"] == "interrupted":
                runner.stop()
                for pending in inflight:
                    pending.cancel()
                progress.finish()
                return None
            record = _make_record(
                config,
                member,
                receptor_name,
                outcome,
                seed=seed,
                cached_receptor=cached,
            )
            writer.write(record)
            summary.records.append(record)
            summary.completed_this_run += 1
            done_here += 1
            if record.status == "ok":
                ok += 1
                if record.affinity is not None:
                    best = record.affinity if best is None else min(best, record.affinity)
            else:
                failed += 1
            elapsed = time.monotonic() - started
            rate = done_here / elapsed if elapsed > 0 else 0.0
            eta = (len(members) - done_here) / rate if rate > 0 else None
            progress.update(
                done_before + done_here, ok=ok, failed=failed, rate=rate, eta=eta, best=best
            )
    progress.finish()
    return done_here


def _load_receptor_for_interactions(config: ScreenConfig, receptor_label: str) -> Any:
    """The receptor as a normalised structure for interaction profiling, once."""
    if not config.interactions:
        return None
    try:
        from .prepare import read_structure

        return _receptor_structure(read_structure(receptor_label))
    except Exception:
        return None


def _make_record(
    config: ScreenConfig,
    member: LibraryMember,
    receptor_name: str,
    outcome: Dict[str, Any],
    *,
    seed: int,
    cached_receptor: Any,
) -> LigandRecord:
    """Turn one docking outcome into the row that goes into the results file."""
    result = outcome.get("result")
    record = LigandRecord(
        receptor=receptor_name,
        ligand=member.key,
        name=member.name,
        source=member.source,
        index=member.index,
        status=str(outcome.get("status", "failed")),
        n_heavy=member.n_heavy,
        n_atoms=member.n_atoms,
        n_torsions=member.n_torsions,
        molecular_weight=member.molecular_weight,
        seed=seed,
        elapsed=float(outcome.get("elapsed") or 0.0),
        error=str(outcome.get("error") or ""),
        violations=list(member.violations),
        properties=dict(member.properties),
    )
    if result is None:
        return record

    best = result.best()
    record.affinity = None if best is None else float(best.affinity)
    record.in_box = bool(best.in_box) if best is not None else False
    record.grid_points = int(getattr(result, "grid_points", 0) or 0)
    record.ligand_efficiency, record.le_source = _ligand_efficiency(
        record.affinity, member.n_heavy
    )
    record.poses = [
        {
            "index": int(pose.index),
            "affinity": float(pose.affinity),
            "rmsd_lb": float(pose.rmsd_lower_bound),
            "rmsd_ub": float(pose.rmsd_upper_bound),
            "in_box": bool(pose.in_box),
        }
        for pose in result.poses
    ]
    if config.write_poses:
        record.pose_file = _write_pose_file(config, receptor_name, member, result)
    if config.interactions and cached_receptor is not None:
        ligand_mol = _pose_mol(result)
        if ligand_mol is not None:
            residues, count, rows = profile_pose(cached_receptor, ligand_mol)
            record.key_residues = residues
            record.n_interactions = count
            record.interactions = rows
    return record


def _pose_mol(result: Any) -> Any:
    """The best pose as an RDKit molecule, for interaction profiling."""
    try:
        from .chem.ligand import read_ligands

        block = _first_model(result.to_pdbqt())
        if not block:
            return None
        mols = read_ligands(block, fmt="pdbqt", embed=False)
        return mols[0] if mols else None
    except Exception:
        return None


def _first_model(text: str) -> str:
    """The first ``MODEL`` block of a pose document — the best pose."""
    current: Optional[List[str]] = None
    for line in text.splitlines():
        if line.startswith("MODEL"):
            current = [line]
            continue
        if line.startswith("ENDMDL"):
            if current is not None:
                current.append(line)
                return "\n".join(current) + "\n"
            continue
        if current is not None:
            current.append(line)
    return text if text.strip() else ""


def _renumber_model(block: str, number: int) -> str:
    """Renumber a single-model block, so a shortlist counts 1..N and not 1,1,1."""
    lines = block.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("MODEL"):
            lines[i] = f"MODEL     {number}"
            break
    return "\n".join(lines) + ("\n" if block.endswith("\n") else "")


def _write_pose_file(
    config: ScreenConfig, receptor_name: str, member: LibraryMember, result: Any
) -> str:
    """Write the poses of one molecule next to the results file."""
    text = result.to_pdbqt()
    if not text:
        return ""
    target = (
        config.outdir
        / POSES_DIR
        / receptor_name
        / f"{member.index + 1:06d}_{_safe_name(member.name)}.pdbqt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return _relpath(config.outdir, target)


# ---------------------------------------------------------------------------
# Ranking, shortlist and report
# ---------------------------------------------------------------------------


def _dedupe_rows(records: Sequence[LigandRecord], path: Path) -> List[LigandRecord]:
    """Keep the first row per ``(receptor, ligand)``, warning about the rest.

    A file written by a release that re-docked after a hard kill holds the same
    molecule twice.  Its numbers would then be counted twice in the report — the
    knock-on that made ``summary.json`` disagree with ``results.jsonl`` — so the
    duplicate is dropped here, with a warning, and the campaign continues from the
    first (oldest) row.
    """
    seen: Dict[Tuple[str, str], bool] = {}
    unique: List[LigandRecord] = []
    duplicates = 0
    for record in records:
        key = (record.receptor, record.ligand)
        if key in seen:
            duplicates += 1
            continue
        seen[key] = True
        unique.append(record)
    if duplicates:
        _err(
            f"odock screen: warning: {path.name} holds {duplicates} duplicate row(s) "
            "(the same molecule and receptor docked twice); keeping the first of each, "
            "so the report counts each molecule once"
        )
    return unique


def _resume_records(
    config: ScreenConfig,
    summary: ScreenSummary,
    manifest: Optional[_Manifest],
    names: Sequence[str],
    base_seed: int,
) -> List[LigandRecord]:
    """The rows of this campaign that are already on disk.

    The distinction the whole function turns on is which rows are **this** run's
    and which are not, and every identity check is made against the latter: a
    directory holding rows for a receptor this run does not screen is another
    campaign's output, and appending to it would hide its rows behind a summary
    that never mentions them.

    * a manifest and a results file — the normal restart.  The rows are this
      campaign's, whatever their number, and they are all kept;
    * a results file and **no manifest** — a campaign killed before the manifest
      existed (any release that only wrote it at the end) or a manifest write
      that was lost.  Silently re-docking would duplicate every row and redo
      hours of work, so the file *is* trusted, with a loud warning; the identity
      checks (the receptors the rows name, the grid they were docked on, the seed
      they were docked with) still refuse a file that belongs to another campaign;
    * neither — a fresh run.

    Whatever the case, every row is checked against the campaign seed this run is
    about to use: a row that was docked under a different seed means the file
    already holds two campaigns, and appending a third is refused.
    """
    results_path = config.results_path
    exists = results_path.exists() and results_path.stat().st_size > 0
    if not exists:
        return []
    records = read_records(results_path)
    if not records:
        return []

    names_set = set(names)
    foreign = [r for r in records if r.receptor not in names_set]
    if foreign:
        # This is the case a filter on `names` used to hide: when *no* row matches
        # the run, the checks below had nothing to look at and the campaign was
        # appended to as if the directory were empty.
        others = sorted({r.receptor for r in foreign})
        raise ScreenError(
            f"{results_path.name} belongs to another campaign: it holds "
            f"{len(foreign)} row(s) for the receptor(s) {', '.join(repr(o) for o in others)}, "
            f"which this run does not screen (it screens "
            f"{', '.join(repr(n) for n in names)}).\n"
            "       Write to a new --out directory, or pass --no-resume to discard "
            f"the results in {config.outdir}."
        )

    strangers = mismatched_seed_rows(records, base_seed)
    if strangers:
        raise ScreenError(
            f"{results_path.name} holds {len(strangers)} row(s) docked under a "
            f"different campaign seed (e.g. {strangers[0]}); the file already "
            "mixes two campaigns.\n"
            "       Write to a new --out directory, or pass --no-resume to discard "
            f"the results in {config.outdir}."
        )
    mine = _dedupe_rows(records, results_path)
    if manifest is not None:
        return mine

    _err(
        f"odock screen: warning: {results_path.name} has {len(records)} row(s) but "
        "there is no run.json: the campaign that wrote them was killed before it "
        "could record its settings."
    )
    expected = grid_points(config.box)
    grids = {int(r.grid_points) for r in records if r.grid_points}
    if grids and grids != {expected}:
        raise ScreenError(
            f"{results_path} was docked on a different grid ({sorted(grids)} points, "
            f"this box has {expected}).\n"
            "       Write to a new --out directory, or pass --no-resume to discard "
            f"the results in {config.outdir}."
        )
    _err(
        f"odock screen: resuming from those rows anyway (the grid matches); a fresh "
        "manifest is being written now."
    )
    return mine


def _finalise(summary: ScreenSummary, receptors: Sequence[Tuple[Path, str, str]]) -> None:
    """Write the ``--top N`` shortlist and the ``--consensus`` tables.

    The summary files themselves are written by the caller *after* this, so that
    the elapsed time they record includes the post-processing.
    """
    config = summary.config
    if config.top:
        for _path, name, _text in receptors:
            target = _write_shortlist(config, summary, name, int(config.top))
            if target is not None:
                summary.top_paths[name] = target
    _run_consensus(config, summary, receptors)


def _write_shortlist(
    config: ScreenConfig, summary: ScreenSummary, receptor_name: str, top: int
) -> Optional[Path]:
    """Concatenate the best pose of the ``top`` best molecules into one PDBQT.

    Only the best model of each molecule is written, one per ``MODEL`` record, so
    the file opens directly in the workbench (``odock gui -p top_x.pdbqt``) and
    is a valid multi-model PDBQT for ``odock split`` and ``odock report``.  A
    ``REMARK`` line names the library molecule each model came from, because a
    pose file that has lost the link to its ligand is useless as a shortlist.
    """
    rows = [r for r in summary.ranked(receptor_name) if r.pose_file]
    if not rows:
        return None
    target = config.outdir / f"top_{receptor_name}.pdbqt"
    chunks: List[str] = []
    csv_rows: List[List[Any]] = []
    for rank, record in enumerate(rows[: max(1, int(top))], start=1):
        try:
            text = (config.outdir / record.pose_file).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        block = _first_model(text)
        if not block:
            continue
        block = _renumber_model(block, len(chunks) + 1)
        affinity = "--" if record.affinity is None else f"{record.affinity:.3f}"
        efficiency = (
            "--" if record.ligand_efficiency is None else f"{record.ligand_efficiency:.3f}"
        )
        chunks.append(
            f"REMARK OPEN DOCKING SCREEN rank {rank} ligand {record.name} "
            f"affinity {affinity} kcal/mol\n"
            f"REMARK OPEN DOCKING SCREEN ligand_efficiency {efficiency} "
            f"key_residues {record.key_residues or '-'}\n" + block
        )
        csv_rows.append(
            [
                rank, record.name, record.affinity, record.ligand_efficiency,
                record.n_heavy, record.n_torsions, record.key_residues, record.pose_file,
            ]
        )
    if not chunks:
        return None
    target.write_text("".join(chunks), encoding="utf-8")
    with target.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("rank", "name", "affinity", "ligand_efficiency", "n_heavy",
             "n_torsions", "key_residues", "pose_file")
        )
        writer.writerows(csv_rows)
    return target


#: The force fields a consensus run combines by default.
CONSENSUS_SCORINGS: Tuple[str, ...] = ("vina", "vinardo", "ad4")
#: How many shortlisted molecules a consensus covers when neither
#: ``--consensus-top`` nor ``--top`` says otherwise.
DEFAULT_CONSENSUS_TOP = 50


@dataclass
class ConsensusReport:
    """How the shortlist of one receptor behaves under every force field.

    A screening table built on one scoring function answers "what does *this*
    force field think".  A campaign has to answer a harder question — *which hits
    survive a change of force field* — and that is what this is: every pose of the
    top N molecules is rescored at its docked coordinates with each force field
    (:func:`odock.consensus.consensus_score`, no search and no refinement), the
    fields are combined into one rank, and their agreement is measured as the mean
    pairwise Spearman rho.  A high rho means the ordering is a property of the
    molecules; a low one means the ordering is a property of the potential.
    """

    receptor: str
    scorings: Tuple[str, ...] = ()
    method: str = "rank"
    #: Mean pairwise Spearman rho between the fields (``nan`` when unknown).
    agreement: float = float("nan")
    #: The same figure over every pose of the shortlist, not just the best mode.
    pose_agreement: float = float("nan")
    #: ``(field_a, field_b, rho)`` for every pair.
    correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    #: One row per shortlisted molecule, in consensus order.
    rows: List[Dict[str, Any]] = field(default_factory=list)
    n_poses: int = 0
    elapsed: float = 0.0
    csv_path: Optional[Path] = None
    json_path: Optional[Path] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.rows)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "receptor": self.receptor,
            "scorings": list(self.scorings),
            "method": self.method,
            "agreement": self.agreement,
            "pose_agreement": self.pose_agreement,
            "correlations": [
                {"a": a, "b": b, "rho": rho} for a, b, rho in self.correlations
            ],
            "n_poses": self.n_poses,
            "n_ligands": len(self.rows),
            "elapsed": self.elapsed,
            "csv": str(self.csv_path) if self.csv_path else None,
            "json": str(self.json_path) if self.json_path else None,
            "error": self.error,
            "ligands": list(self.rows),
        }

    def table(self) -> str:
        """The per-molecule force-field table: score (rank) under each field."""
        if not self.rows:
            return "(no consensus)"
        headers = ["rank", "name"] + [f"{name} (rank)" for name in self.scorings]
        headers += ["consensus", "poses", "docked pose?"]
        body: List[List[str]] = []
        for row in self.rows:
            cells = [str(row["consensus_rank"]), str(row["name"])[:28]]
            for name in self.scorings:
                value = row.get(name)
                rank = row.get(f"{name}_rank")
                if value is None or (isinstance(value, float) and not math.isfinite(value)):
                    cells.append("--")
                else:
                    cells.append(f"{float(value):.3f} ({float(rank):.1f})")
            cells.append(f"{float(row['consensus_score']):.3f}")
            cells.append(f"{int(row.get('n_poses') or 0)}")
            agrees = row.get("docked_pose_agrees")
            cells.append("-" if agrees is None else ("yes" if agrees else "no"))
            body.append(cells)
        return _table(headers, body)

    def agreement_line(self) -> str:
        if not math.isfinite(self.agreement):
            return "force-field agreement: unknown"
        pairs = ", ".join(
            f"{a}~{b} {rho:.2f}" for a, b, rho in self.correlations if math.isfinite(rho)
        )
        verdict = (
            "the hit list is mostly a property of the molecules"
            if self.agreement >= 0.7
            else "the hit list is partly a property of the potential"
            if self.agreement >= 0.4
            else "the fields disagree: treat the ordering as soft"
        )
        line = (
            f"force-field agreement: mean pairwise Spearman rho {self.agreement:.2f} "
            f"on {len(self.rows)} hit(s) ({pairs}) — {verdict}"
        )
        if math.isfinite(self.pose_agreement):
            line += f"; over all {self.n_poses} pose(s) rho {self.pose_agreement:.2f}"
        return line


def consensus_limit(config: ScreenConfig, available: int) -> int:
    """How many molecules a consensus run covers: ``--consensus-top``, else ``--top``."""
    wanted = int(config.consensus_top) or int(config.top) or DEFAULT_CONSENSUS_TOP
    return max(1, min(int(available), wanted))


def _load_consensus():
    """Import :mod:`odock.consensus`, or raise :class:`ImportError` with a reason."""
    from . import consensus  # type: ignore

    if not hasattr(consensus, "consensus_score"):
        raise ImportError("odock.consensus has no consensus_score()")
    return consensus


def run_consensus(
    config: ScreenConfig,
    summary: ScreenSummary,
    receptor_name: str,
    receptor_text: str,
) -> ConsensusReport:
    """Rescore one receptor's shortlist with every force field and rank the hits.

    Two questions, one rescoring pass:

    * **which hits survive a change of force field** — the *ligand* ranking, built
      from the pose each ligand's own docking preferred (mode 1).  Its scores are
      the per-field affinities of that one pose, its ranks run 1..N, and the mean
      pairwise Spearman rho says how much the fields agree about the hit list;
    * **does the docking's chosen binding mode survive** — the *pose* ranking,
      built from every pose of the shortlist, which says whether mode 1 is still
      the ligand's best mode under the consensus.

    Never raises: the campaign is already on disk and a missing or unhappy
    :mod:`odock.consensus` must not turn a finished screening run into a failed
    one.  The reason is in :attr:`ConsensusReport.error` when it could not run.
    """
    started = time.perf_counter()
    report = ConsensusReport(receptor=receptor_name, method=config.consensus_method)
    try:
        consensus = _load_consensus()
    except ImportError as exc:
        report.error = f"consensus scoring is unavailable: {exc}"
        return report

    rows = [r for r in summary.ranked(receptor_name) if r.pose_file]
    if not rows:
        report.error = "no docked molecule has a pose file (was --no-poses used?)"
        return report
    shortlist = rows[: consensus_limit(config, len(rows))]

    documents: List[str] = []
    #: Ligand index of every document; the first document of a ligand is its mode 1.
    owners: List[int] = []
    for index, record in enumerate(shortlist):
        try:
            text = (config.outdir / record.pose_file).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            continue
        for model in consensus.pdbqt_models(text):
            documents.append(model)
            owners.append(index)
    if not documents:
        report.error = "the shortlist holds no readable pose"
        return report

    report.scorings = tuple(
        str(name)
        for name in (getattr(consensus, "DEFAULT_SCORINGS", None) or CONSENSUS_SCORINGS)
    )
    try:
        components = consensus.rescore_poses(
            documents, receptor_text, config.box, report.scorings
        )
        # One representative per ligand — the pose the docking itself ranked first.
        representatives: List[int] = []
        seen: set = set()
        for position, owner in enumerate(owners):
            if owner not in seen:
                seen.add(owner)
                representatives.append(position)
        ligand_result = consensus.consensus_score(
            [documents[position] for position in representatives],
            receptor_text,
            config.box,
            scorings=report.scorings,
            method=config.consensus_method,
            components={
                name: {"affinity": [components[name]["affinity"][position] for position in representatives]}
                for name in report.scorings
            },
        )
        pose_result = consensus.consensus_score(
            documents,
            receptor_text,
            config.box,
            scorings=report.scorings,
            method=config.consensus_method,
            components=components,
        )
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    report.n_poses = len(documents)
    report.agreement = float(getattr(ligand_result, "agreement", float("nan")))
    report.pose_agreement = float(getattr(pose_result, "agreement", float("nan")))
    report.correlations = [
        (str(a), str(b), float(rho))
        for a, b, rho in getattr(ligand_result, "correlations", [])
    ]

    # The ligand's own best mode under the consensus, for the "does the docked
    # pose survive?" column.
    best_pose_by_ligand: Dict[int, Any] = {}
    for pose in pose_result.poses:
        owner = owners[int(pose.index)]
        current = best_pose_by_ligand.get(owner)
        if current is None or _better_consensus(pose, current):
            best_pose_by_ligand[owner] = pose

    ligand_rows: List[Dict[str, Any]] = []
    for entry in ligand_result.poses:
        position = representatives[int(entry.index)]
        owner = owners[position]
        record = shortlist[owner]
        row: Dict[str, Any] = {
            "name": record.name,
            "ligand": record.ligand,
            "consensus_rank": int(entry.consensus_rank),
            "consensus_score": float(entry.consensus_score),
            "docked_affinity": record.affinity,
            "docked_rank": rows.index(record) + 1,
            "n_poses": sum(1 for item in owners if item == owner),
            "docked_pose_agrees": (
                int(best_pose_by_ligand[owner].index) == position
                if owner in best_pose_by_ligand
                else None
            ),
            "pose_file": record.pose_file,
        }
        for name in report.scorings:
            row[name] = float(entry.scores.get(name, float("nan")))
            row[f"{name}_rank"] = float(entry.ranks.get(name, float("nan")))
        ligand_rows.append(row)
    ligand_rows.sort(key=lambda item: int(item["consensus_rank"]))
    report.rows = ligand_rows
    report.elapsed = time.perf_counter() - started
    return report


def _better_consensus(first: Any, second: Any) -> bool:
    """Whether `first` beats `second` on the consensus score (``nan`` loses)."""
    a = float(getattr(first, "consensus_score", float("nan")))
    b = float(getattr(second, "consensus_score", float("nan")))
    if not math.isfinite(a):
        return False
    if not math.isfinite(b):
        return True
    return a < b


def _write_consensus(config: ScreenConfig, report: ConsensusReport) -> None:
    """Write ``consensus_<receptor>.csv`` and ``.json`` for one receptor."""
    if not report.rows:
        return
    columns = ["consensus_rank", "name", "ligand", "docked_affinity", "docked_rank",
               "consensus_score"]
    for name in report.scorings:
        columns += [name, f"{name}_rank"]
    columns += ["n_poses", "docked_pose_agrees", "pose_file"]
    csv_path = config.outdir / f"consensus_{report.receptor}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in report.rows:
            cells = []
            for column in columns:
                value = row.get(column, "")
                if column == "docked_pose_agrees" and value is not None:
                    # 1/0, like `in_box` in the results file, so both tables parse
                    # the same way.
                    value = 1 if value else 0
                cells.append("" if value is None else value)
            writer.writerow(cells)
    report.csv_path = csv_path
    json_path = config.outdir / f"consensus_{report.receptor}.json"
    json_path.write_text(
        json.dumps(report.as_dict(), indent=2, default=str) + "\n", encoding="utf-8"
    )
    report.json_path = json_path


def _run_consensus(
    config: ScreenConfig, summary: ScreenSummary, receptors: Sequence[Tuple[Path, str, str]]
) -> None:
    """Consensus-score every receptor's shortlist, in place.

    The result is printed by :meth:`ScreenSummary.text` — once, with the rest of
    the report — so this only announces the work and records it.
    """
    if not config.consensus:
        return
    for _path, name, text in receptors:
        available = len([r for r in summary.ranked(name) if r.pose_file])
        if not available:
            report = ConsensusReport(receptor=name, method=config.consensus_method)
            report.error = (
                "no docked molecule has a pose file; the consensus needs the poses "
                "(--no-poses was used?)"
            )
            summary.consensus[name] = report
            _err(f"odock screen: consensus scoring skipped for {name}: {report.error}")
            continue
        _out(
            f"\nconsensus ({name}): rescoring the top "
            f"{consensus_limit(config, available)} of {available} hit(s) with "
            f"{', '.join(CONSENSUS_SCORINGS)} ..."
        )
        report = run_consensus(config, summary, name, text)
        if report.error:
            _err(f"odock screen: consensus scoring failed for {name}: {report.error}")
        else:
            _write_consensus(config, report)
        summary.consensus[name] = report


def _report_dry_run(
    summary: ScreenSummary, receptors: Sequence[Tuple[Path, str, str]]
) -> None:
    _out("")
    _out(
        f"dry run: {len(summary.members)} molecule(s) would be docked against "
        f"{len(receptors)} receptor(s)"
    )
    rows = []
    for i, member in enumerate(summary.members[:60], start=1):
        rows.append(
            [
                str(i),
                member.name[:28],
                _fmt(member.properties.get("MW"), "{:.1f}"),
                _fmt(member.properties.get("LogP"), "{:.2f}"),
                str(member.n_torsions),
                str(member.n_atoms),
            ]
        )
    _out(_table(["#", "name", "MW", "LogP", "tors", "atoms"], rows))
    if len(summary.members) > 60:
        _out(f"... and {len(summary.members) - 60} more (the full list is in library.csv)")
    if summary.stats.removed:
        _out("")
        _out(f"removed by the filters ({len(summary.stats.removed)} molecule(s)):")
        for row in summary.stats.removed[:15]:
            _out(f"  {row['name'][:30]:<32} {row['violations']}")
        if len(summary.stats.removed) > 15:
            _out(f"  ... and {len(summary.stats.removed) - 15} more (see library.csv)")
    _out("")
    _out("the estimate is in the report below; nothing was docked")


def _save_manifest(
    path: Path,
    config: ScreenConfig,
    summary: ScreenSummary,
    members: Sequence[LibraryMember],
    stats: FilterStats,
    *,
    interrupted: bool = False,
    phase: str = "done",
) -> None:
    """Write (or refresh) ``run.json``.

    Called before the first docking and at every checkpoint, so the file always
    names the campaign the rows on disk belong to.  ``seconds_per_molecule`` is
    only overwritten when this run actually measured one: an early manifest must
    not erase the measurement a previous campaign in the same directory left
    behind.
    """
    manifest = _Manifest.load(path) or _Manifest(path, {})
    measured = summary.measured_per_molecule or manifest.get("seconds_per_molecule")
    manifest.data.update(
        {
            "version": 1,
            "created": manifest.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "odock": _version(),
            "library_hash": config.library_hash(),
            "docking_hash": config.docking_hash(),
            "docking_settings": config.docking_settings(),
            "seed": summary.seed,
            "config": config.as_dict(),
            "box": config.box.as_dict(),
            "inputs": [_file_signature(p) for p in config.inputs],
            "receptors": [
                {"path": str(p), "name": name}
                for p, name in zip(config.receptors, config.receptor_names())
            ],
            "library": stats.as_dict(),
            "n_library": len(members),
            "n_dockable": len(summary.members),
            "counts": {
                "docked": summary.n_ok(),
                "failed": summary.n_failed(),
                "timeout": summary.n_timeout(),
                "completed_this_run": summary.completed_this_run,
            },
            "estimate_seconds": (summary.estimate or {}).get("seconds"),
            "seconds_per_molecule": measured,
            "elapsed": summary.elapsed,
            "dry_run": bool(config.dry_run),
            "interrupted": bool(interrupted),
            "phase": str(phase),
            "results": config.results_name,
            "format": config.fmt,
        }
    )
    manifest.save()


def _version() -> str:
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"

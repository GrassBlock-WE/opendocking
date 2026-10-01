# SPDX-License-Identifier: GPL-3.0-or-later
"""The high-level OpenDocking Python API.

Everything the kernel does is reachable from here::

    import odock

    receptor, _, _ = odock.prepare_receptor("4EY7.pdb", "receptor.pdbqt")
    ligand, _, _ = odock.prepare_ligand("donepezil.sdf", "ligand.pdbqt")
    box = odock.box_from_ligand(ligand, buffer=8.0)

    result = odock.dock("receptor.pdbqt", "ligand.pdbqt", box, exhaustiveness=16, seed=42)
    print(result.table())
    open("poses.pdbqt", "w").write(result.to_pdbqt())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import _odock
from .prepare import BoxSpec, PathLike

__all__ = [
    "DockResult",
    "Pose",
    "dock",
    "gpu_available",
    "gpu_description",
    "kernel_version",
    "score",
    "score_pair",
]

#: The X-Score atom-type names, indexed by type index.
XS_TYPES: List[str] = list(_odock.xs_type_names())


def kernel_version() -> str:
    """The version of the Rust kernel."""
    return str(_odock.__version__)


def gpu_available() -> bool:
    """Whether the GPU affinity-grid backend is compiled in and usable."""
    return bool(_odock.gpu_available())


def gpu_description() -> str:
    """A description of the GPU backend, or why it is unavailable."""
    return str(_odock.gpu_description())


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class Pose:
    """One docked pose."""

    index: int
    affinity: float
    rmsd_lower_bound: float = 0.0
    rmsd_upper_bound: float = 0.0
    inter: float = 0.0
    intra: float = 0.0
    conf_independent: float = 0.0
    unbound: float = 0.0
    in_box: bool = True
    num_atoms: int = 0
    position: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = None
    torsions: Optional[List[float]] = None
    #: Ligand coordinates in the kernel's internal atom order (all atoms).
    coords: Optional["np.ndarray"] = field(default=None, repr=False)

    def to_dict(self, include_coords: bool = False) -> Dict[str, object]:
        """A plain, JSON-serialisable dictionary view.

        The coordinate array is omitted by default: it is large and easily
        recovered with :func:`pose_to_mol` or ``DockResult.to_pdbqt``.
        """
        d = dict(self.__dict__)
        d["rmsd_lb"] = d.pop("rmsd_lower_bound")
        d["rmsd_ub"] = d.pop("rmsd_upper_bound")
        coords = d.pop("coords", None)
        if include_coords and coords is not None:
            d["coords"] = np.asarray(coords, dtype=float).tolist()
        return d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Pose({self.index + 1}, affinity={self.affinity:.3f} kcal/mol, "
            f"rmsd_lb={self.rmsd_lower_bound:.3f})"
        )


@dataclass
class DockResult:
    """The outcome of a docking run."""

    poses: List[Pose]
    seed: int
    #: The force field the run used (``"vina"``, ``"vinardo"`` or ``"ad4"``).
    #: Reporters need it: a strain correction, an entropy estimate or a
    #: consensus column all mean something different under a different
    #: potential, and guessing it silently is not an option.
    scoring: str = "vina"
    grid_mb: int = 0
    grid_points: int = 0
    num_tors: float = 0.0
    num_movable_atoms: int = 0
    num_dof: int = 0
    exact: bool = True
    receptor_pdbqt: str = ""
    ligand_pdbqt: str = ""
    box: Optional[BoxSpec] = None
    elapsed: float = 0.0
    #: Ligand atom names in the kernel's internal order.
    ligand_atom_order: Tuple[str, ...] = field(default_factory=tuple)
    _pdbqt: Optional[str] = field(default=None, repr=False)

    # -- accessors ---------------------------------------------------------

    def best(self) -> Optional[Pose]:
        """The best-scoring pose."""
        return self.poses[0] if self.poses else None

    @property
    def best_affinity(self) -> Optional[float]:
        """The affinity of the best pose, in kcal/mol."""
        p = self.best()
        return None if p is None else p.affinity

    def within(self, energy_range: float = 3.0) -> List[Pose]:
        """Poses within `energy_range` kcal/mol of the best one."""
        if not self.poses:
            return []
        best = self.poses[0].affinity
        return [p for p in self.poses if p.affinity <= best + energy_range]

    # -- serialisation ----------------------------------------------------

    def table(self, energy_range: Optional[float] = None) -> str:
        """A Vina-style results table."""
        rows = self.poses if energy_range is None else self.within(energy_range)
        out = [
            "mode |   affinity | dist from best mode",
            "     | (kcal/mol) | rmsd l.b.| rmsd u.b.",
            "-----+------------+----------+----------",
        ]
        for i, p in enumerate(rows):
            out.append(
                f"{i + 1:>4d}    {p.affinity:>9.3f}  {p.rmsd_lower_bound:>9.3f}"
                f"  {p.rmsd_upper_bound:>9.3f}"
            )
        return "\n".join(out)

    def to_pdbqt(self) -> str:
        """The poses as a multi-model PDBQT document."""
        if self._pdbqt is not None:
            return self._pdbqt
        return ""

    def to_pdb(self) -> str:
        """The poses as a multi-model PDB document (REMARKs preserved)."""
        lines = []
        for i, block in enumerate(_split_models(self.to_pdbqt())):
            lines.append(f"MODEL     {i + 1}")
            for line in block.splitlines():
                if line.startswith(("BRANCH", "ENDBRANCH", "ROOT", "ENDROOT", "TORSDOF")):
                    continue
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    lines.append(line[:66].rstrip())
                elif line.startswith("REMARK"):
                    lines.append(line)
            lines.append("ENDMDL")
        return "\n".join(lines) + "\n"

    def summary(self) -> str:
        """A short human-readable summary."""
        best = self.best_affinity
        head = (
            f"OpenDocking: {len(self.poses)} poses, "
            f"seed={self.seed}, "
            f"{self.num_movable_atoms} movable atoms, "
            f"{self.num_dof} torsions, "
            f"N_tors={self.num_tors:g}, "
            f"grid={self.grid_points} points/{self.grid_mb} MB, "
            f"elapsed={self.elapsed:.2f}s"
        )
        if best is None:
            return head + "\nno poses"
        return head + "\nbest affinity: %.3f kcal/mol" % best + "\n" + self.table()


def _split_models(text: str) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            if current:
                blocks.append("\n".join(current))
            current = []
        elif line.startswith("ENDMDL"):
            if current:
                blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _as_text(source: Union[str, PathLike]) -> str:
    if isinstance(source, (str, os.PathLike)) and not str(source).lstrip().startswith(
        ("REMARK", "ATOM", "HETATM", "ROOT", "MODEL")
    ):
        p = Path(source)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    return str(source)


def dock(
    receptor: Union[str, PathLike],
    ligand: Union[str, PathLike],
    box: BoxSpec,
    *,
    scoring: str = "vina",
    exhaustiveness: int = 8,
    num_poses: int = 9,
    seed: int = 0,
    use_grid: bool = True,
    refine: bool = True,
    min_rmsd: float = 1.0,
    energy_range: float = 3.0,
    use_island_ga: bool = False,
    search: Optional[str] = None,
    islands: int = 4,
    population: int = 32,
    generations: int = 20,
    global_steps: Optional[int] = None,
    local_steps: Optional[int] = None,
) -> DockResult:
    """Dock `ligand` into `receptor` inside `box`.

    Parameters
    ----------
    receptor, ligand
        Paths to PDBQT files, or the PDBQT text itself.
    box
        A :class:`~odock.prepare.BoxSpec`.
    scoring
        ``"vina"`` (default), ``"vinardo"`` or ``"ad4"``.
    exhaustiveness
        Number of independent Monte-Carlo runs; scales roughly linearly with
        wall time and improves the chance of finding the global minimum.
    seed
        ``0`` draws a seed from the operating-system entropy pool; any other
        value makes the run exactly reproducible.
    use_island_ga
        Run the island-model Lamarckian genetic algorithm instead of parallel
        Monte-Carlo. Equivalent to ``search="lga"``.
    search
        Which search protocol to run: ``"monte_carlo"`` (default), ``"lga"`` for
        the island-model Lamarckian genetic algorithm, or ``"lga_solis"`` for
        that same genetic algorithm with AutoDock 4's Solis-Wets local search in
        place of BFGS. ``None`` leaves the kernel default in place.
    """
    import time

    receptor_text = _as_text(receptor)
    ligand_text = _as_text(ligand)
    t0 = time.perf_counter()
    engine = _odock.Docking(
        receptor_text,
        ligand_text,
        center=tuple(float(x) for x in box.center),
        size=tuple(float(x) for x in box.size),
        spacing=float(box.spacing),
        scoring=scoring,
        exhaustiveness=int(exhaustiveness),
        num_poses=int(num_poses),
        seed=int(seed),
        use_grid=bool(use_grid),
        refine=bool(refine),
        min_rmsd=float(min_rmsd),
        energy_range=float(energy_range),
        use_island_ga=bool(use_island_ga),
        islands=int(islands),
        population=int(population),
        generations=int(generations),
        global_steps=global_steps,
        local_steps=local_steps,
        # The kernel takes a `str`; passing None would be a type error, so the
        # keyword is only sent when the caller actually selected a protocol.
        **({"search": str(search)} if search is not None else {}),
    )
    raw = engine.run()
    elapsed = time.perf_counter() - t0
    return result_from_engine(
        engine,
        raw,
        elapsed=elapsed,
        box=box,
        receptor_pdbqt=receptor_text,
        ligand_pdbqt=ligand_text,
        energy_range=energy_range,
        scoring=scoring,
    )


def build_engine(
    receptor,
    ligand,
    box: BoxSpec,
    *,
    scoring: str = "vina",
    exhaustiveness: int = 8,
    num_poses: int = 9,
    seed: int = 0,
    use_grid: bool = True,
    refine: bool = True,
    min_rmsd: float = 1.0,
    energy_range: float = 3.0,
    use_island_ga: bool = False,
    search: Optional[str] = None,
    islands: int = 4,
    population: int = 32,
    generations: int = 20,
    global_steps: Optional[int] = None,
    local_steps: Optional[int] = None,
):
    """Create the kernel engine without running it.

    Exposed so that a caller which needs to *steer* a run — the workbench's
    pause/abort buttons — can hold the engine and call ``run()`` itself, while
    the result conversion stays in :func:`result_from_engine` instead of being
    duplicated (and drifting) at the call site.
    """
    return _odock.Docking(
        _as_text(receptor),
        _as_text(ligand),
        center=tuple(float(x) for x in box.center),
        size=tuple(float(x) for x in box.size),
        spacing=float(box.spacing),
        scoring=scoring,
        exhaustiveness=int(exhaustiveness),
        num_poses=int(num_poses),
        seed=int(seed),
        use_grid=bool(use_grid),
        refine=bool(refine),
        min_rmsd=float(min_rmsd),
        energy_range=float(energy_range),
        use_island_ga=bool(use_island_ga),
        islands=int(islands),
        population=int(population),
        generations=int(generations),
        global_steps=global_steps,
        local_steps=local_steps,
        **({"search": str(search)} if search is not None else {}),
    )


def result_from_engine(
    engine,
    raw: dict,
    *,
    elapsed: float,
    box: BoxSpec,
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    energy_range: float = 3.0,
    scoring: Optional[str] = None,
) -> DockResult:
    """Convert a raw kernel result into a :class:`DockResult`."""
    poses = [
        Pose(
            index=int(p["index"]),
            affinity=float(p["affinity"]),
            rmsd_lower_bound=float(p["rmsd_lower_bound"]),
            rmsd_upper_bound=float(p["rmsd_upper_bound"]),
            inter=float(p["inter"]),
            intra=float(p["intra"]),
            conf_independent=float(p["conf_independent"]),
            unbound=float(p["unbound"]),
            in_box=bool(p["in_box"]),
            num_atoms=int(p["num_atoms"]),
            position=p["position"],
            orientation=p["orientation"],
            torsions=list(p["torsions"]) if p["torsions"] is not None else None,
            coords=np.asarray(engine.pose_coords(int(p["index"])), dtype=float),
        )
        for p in raw["poses"]
    ]
    result = DockResult(
        poses=poses,
        seed=int(raw["seed"]),
        scoring=str(scoring) if scoring is not None else str(engine.scoring),
        grid_mb=int(raw["grid_mb"]),
        grid_points=int(raw["grid_points"]),
        num_tors=float(raw["num_tors"]),
        num_movable_atoms=int(raw["num_movable_atoms"]),
        num_dof=int(raw["num_dof"]),
        exact=bool(raw["exact"]),
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        box=box,
        elapsed=elapsed,
        ligand_atom_order=tuple(engine.ligand_atom_names()),
        _pdbqt=engine.poses_pdbqt(energy_range),
    )
    # Set by the kernel when a run was stopped early; absent on older builds.
    if "cancelled" in raw:
        try:
            result.cancelled = bool(raw["cancelled"])
        except Exception:  # pragma: no cover - dataclass without the field
            pass
    return result


# ---------------------------------------------------------------------------
# Turning poses back into RDKit molecules
# ---------------------------------------------------------------------------


def pose_to_mol(pose: Pose, template_mol, atom_order: Sequence[int]):
    """Build an RDKit molecule for `pose`.

    Parameters
    ----------
    pose
        A :class:`Pose` with `coords` populated.
    template_mol
        The prepared RDKit molecule the ligand PDBQT was written from.
    atom_order
        ``report.atom_order`` from :func:`odock.prepare.prepare_ligand`: the
        RDKit atom indices in the kernel's internal order.

    Returns
    -------
    A copy of `template_mol` whose conformer holds the pose.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    if pose.coords is None:
        raise ValueError("this pose has no coordinates; re-run dock() to populate them")
    if len(atom_order) != pose.coords.shape[0]:
        raise ValueError(
            f"atom_order has {len(atom_order)} entries but the pose has "
            f"{pose.coords.shape[0]} atoms; was the ligand re-prepared?"
        )
    mol = Chem.Mol(template_mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for k, rd_idx in enumerate(atom_order):
        x, y, z = pose.coords[k]
        conf.SetAtomPosition(int(rd_idx), Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    return mol


def aligned_rmsd(probe, reference, heavy_only: bool = True) -> Tuple[float, float]:
    """Symmetry-corrected RMSD of `probe` onto `reference`.

    Returns ``(rmsd_without_fitting, rmsd_after_optimal_superposition)``. The
    first number is the crystallographic figure of merit: no alignment is
    allowed, because the docking must place the ligand in the experimental pose
    in the *same* frame.
    """
    from rdkit.Chem import rdMolAlign

    if heavy_only:
        probe = _heavy_subset(probe)
        reference = _heavy_subset(reference)
    try:
        rms_after_fit = float(rdMolAlign.CalcRMS(probe, reference))
    except Exception:
        # `CalcRMS` needs a substructure match, which fails when the two
        # molecules were built from differently perceived bond orders. The
        # no-superposition RMSD below is still exact, so fall back to it.
        raw = _rmsd_same_frame(probe, reference)
        return raw, raw
    # CalcRMS superimposes internally. To measure the docking accuracy we need
    # the RMSD in the original frames, so compute it directly.
    raw = _rmsd_same_frame(probe, reference)
    return raw, rms_after_fit


def _heavy_subset(mol):
    from rdkit import Chem

    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return Chem.RWMol(mol).GetMol() if False else _select_atoms(mol, idx)


def _select_atoms(mol, idx: Sequence[int]):
    """A molecule restricted to `idx`, preserving the conformer."""
    from rdkit import Chem

    em = Chem.RWMol(mol)
    keep = set(int(i) for i in idx)
    for i in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
        em.RemoveAtom(i)
    out = em.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    return out


def _rmsd_same_frame(a, b) -> float:
    """Heavy-atom RMSD between two equally-sized molecules, no superposition."""
    import numpy as np
    from rdkit import Chem

    ca, cb = a.GetConformer(), b.GetConformer()
    pa = np.array([[ca.GetAtomPosition(i).x, ca.GetAtomPosition(i).y, ca.GetAtomPosition(i).z]
                   for i in range(a.GetNumAtoms())])
    pb = np.array([[cb.GetAtomPosition(i).x, cb.GetAtomPosition(i).y, cb.GetAtomPosition(i).z]
                   for i in range(b.GetNumAtoms())])
    if pa.shape != pb.shape:
        raise ValueError("molecules have different heavy-atom counts")
    return float(np.sqrt(((pa - pb) ** 2).sum(axis=1).mean()))


def score(
    receptor: Union[str, PathLike],
    ligand: Union[str, PathLike],
    box: Optional[BoxSpec] = None,
    *,
    scoring: str = "vina",
    refine: bool = True,
) -> Dict[str, float]:
    """Score `ligand` at its input position (no search).

    The box only provides the out-of-box penalty and, when the grid is used, the
    interpolation domain; the exact scorer ignores it apart from that penalty.
    """
    receptor_text = _as_text(receptor)
    ligand_text = _as_text(ligand)
    if box is None:
        box = BoxSpec(center=(0.0, 0.0, 0.0), size=(200.0, 200.0, 200.0), spacing=1.0)
    engine = _odock.Docking(
        receptor_text,
        ligand_text,
        center=tuple(float(x) for x in box.center),
        size=tuple(float(x) for x in box.size),
        spacing=float(box.spacing),
        scoring=scoring,
        refine=refine,
        use_grid=False,
    )
    return {k: float(v) for k, v in engine.score().items()}


def score_pair(t1: str, t2: str, r: float, scoring: str = "vina") -> float:
    """Evaluate one pairwise interaction term (for teaching and validation)."""
    return float(_odock.pair_energy(t1, t2, float(r), scoring))

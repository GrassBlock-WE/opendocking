# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared re-docking harness used by the validation scripts and the test-suite.

Re-docking a co-crystallised ligand is the standard end-to-end correctness check
for a docking engine: prepare both partners from the raw experimental structure,
search from random conformations, and compare the top-scoring pose with the
experimental binding mode.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import odock  # noqa: E402
from odock.cli import _use_utf8_streams  # noqa: E402

#: Reconfigure the console streams to UTF-8 before anything prints Å or Å³; a
#: non-UTF-8 code page (GBK, cp1252) would otherwise raise `UnicodeEncodeError`.
_use_utf8_streams()

#: Residue names that are solvent and never a ligand.
WATER_RESNAMES = {"HOH", "WAT", "DOD"}
#: Ions and common crystallisation additives that are part of the receptor.
RECEPTOR_HETERO = {
    "SO4", "PO4", "GOL", "EDO", "ACT", "ACE", "FMT", "MES", "TRS", "PEG",
    "NAG", "BMA", "MAN", "FUC", "GAL", "GLC", "NDG", "BGC",
    "MG", "MN", "ZN", "CA", "FE", "NA", "K", "CL", "BR", "IOD", "CS", "NI",
    "CD", "CU", "HG", "CO",
}


@dataclass
class RedockResult:
    """Everything a re-docking run produced."""

    name: str
    ligand_resname: str
    poses: List[Tuple[int, float, float, float]]  # (mode, affinity, rmsd, fitted)
    best_rmsd: float
    best_affinity: float
    crystal_affinity: float
    num_tors: float
    num_atoms: int
    elapsed: float
    grid_points: int
    receptor_atoms: int
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """The usual acceptance criterion: top pose within 2 Å of the crystal."""
        return self.best_rmsd <= 2.0

    def closest_within(self, window: float = 1.0) -> float:
        """Lowest RMSD among the poses inside an energy window of the best.

        Reporting only the top pose is harsh on a flexible ligand: when several
        poses sit within a few hundredths of a kcal/mol the *ranking* is not
        meaningful, even though the correct binding mode was found. This is the
        standard "top-N success" measure used to report docking accuracy.
        """
        if not self.poses:
            return float("inf")
        best = min(p[1] for p in self.poses)
        return min(p[2] for p in self.poses if p[1] <= best + window)

    def as_dict(self) -> dict:
        """A JSON-serialisable summary."""
        return {
            "name": self.name,
            "ligand": self.ligand_resname,
            "poses": self.poses,
            "best_rmsd": self.best_rmsd,
            "best_affinity": self.best_affinity,
            "crystal_affinity": self.crystal_affinity,
            "num_tors": self.num_tors,
            "num_atoms": self.num_atoms,
            "elapsed": self.elapsed,
            "grid_points": self.grid_points,
            "receptor_atoms": self.receptor_atoms,
            "passed": self.passed,
            "closest_within_1kcal": self.closest_within(1.0),
        }

    def table(self) -> str:
        """A formatted results table."""
        rows = [
            "mode |   affinity |  RMSD (no fit) |  RMSD (fitted)",
            "-----+------------+----------------+---------------",
        ]
        for mode, affinity, raw, fitted in self.poses:
            rows.append(f"{mode:>4d} | {affinity:>10.3f} | {raw:>14.3f} | {fitted:>14.3f}")
        return "\n".join(rows)

    def summary(self) -> str:
        """A one-paragraph report."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name} / {self.ligand_resname}: {verdict} — top-pose RMSD "
            f"{self.best_rmsd:.3f} A (affinity {self.best_affinity:.3f} kcal/mol, "
            f"crystal pose {self.crystal_affinity:.3f}), {self.num_atoms} ligand atoms, "
            f"N_tors={self.num_tors:g}, {self.receptor_atoms} receptor atoms, "
            f"grid {self.grid_points} points, {self.elapsed:.2f} s"
        )


def split_complex(pdb_path, ligand_resnames: Iterable[str]):
    """Split a crystal structure into ``(receptor_mol, ligand_mol)``.

    Waters are dropped. Residues named in `ligand_resnames` become the ligand;
    everything else (protein plus ions and additives) becomes the receptor.
    """
    from rdkit import Chem

    ligand_set = {r.strip().upper() for r in ligand_resnames}
    mol = Chem.MolFromPDBFile(
        str(pdb_path), removeHs=False, sanitize=False, proximityBonding=True
    )
    if mol is None:
        raise ValueError(f"RDKit could not read {pdb_path}")

    keep_rec: List[int] = []
    keep_lig: List[int] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip().upper() if info is not None else ""
        if name in WATER_RESNAMES:
            continue
        (keep_lig if name in ligand_set else keep_rec).append(atom.GetIdx())
    if not keep_lig:
        raise ValueError(
            f"no residue named {sorted(ligand_set)} in {pdb_path}; "
            "check the ligand residue name"
        )

    def subset(indices: Sequence[int]):
        em = Chem.RWMol(mol)
        total = set(range(mol.GetNumAtoms()))
        for i in sorted(total - set(indices), reverse=True):
            em.RemoveAtom(i)
        out = em.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            pass
        return out

    return subset(keep_rec), subset(keep_lig)


def redock(
    pdb_path,
    ligand_resname: str,
    *,
    smiles: Optional[str] = None,
    exhaustiveness: int = 8,
    seed: int = 42,
    buffer: float = 8.0,
    scoring: str = "vina",
    num_poses: int = 9,
    outdir: Optional[Path] = None,
    verbose: bool = True,
) -> RedockResult:
    """Prepare, dock and score one co-crystallised ligand."""
    from rdkit import Chem

    from odock.prepare import BoxSpec

    pdb_path = Path(pdb_path)
    receptor_mol, ligand_mol = split_complex(pdb_path, [ligand_resname])

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    say(f"[prepare] receptor from {pdb_path.name}")
    rec_mol, rec_pdbqt, rec_report = odock.prepare_receptor(
        receptor_mol, (outdir / "receptor.pdbqt") if outdir else None, keep_water=False
    )
    say(f"          {rec_report.summary()}")
    for w in rec_report.warnings:
        say(f"          warning: {w}")

    say("[prepare] ligand")
    lig_mol, lig_pdbqt, lig_report = odock.prepare_ligand(
        ligand_mol, (outdir / "ligand.pdbqt") if outdir else None,
        name=ligand_resname, optimize=False, smiles=smiles,
    )
    say(f"          {lig_report.summary()}")
    for w in lig_report.warnings:
        say(f"          warning: {w}")

    box = odock.box_from_ligand(ligand_mol, buffer=buffer)
    say(f"[box]     {box}")

    crystal = odock.score(rec_pdbqt, lig_pdbqt, box, scoring=scoring)
    say(
        f"[score]   crystal pose: {crystal['affinity']:.3f} kcal/mol "
        f"(inter {crystal['inter']:.3f})"
    )
    if crystal["inter"] > 0.0:
        say(
            "          WARNING: the experimental pose scores repulsively; the "
            "receptor preparation is probably wrong (spurious atoms in the site?)"
        )

    say(f"[dock]    {scoring}, exhaustiveness={exhaustiveness}, seed={seed}")
    result = odock.dock(
        rec_pdbqt,
        lig_pdbqt,
        box,
        scoring=scoring,
        exhaustiveness=exhaustiveness,
        num_poses=num_poses,
        seed=seed,
    )
    say(f"          {result.elapsed:.2f} s, N_tors={result.num_tors:g}, "
        f"grid {result.grid_points} points")
    say(result.table())

    # The reference must be the *prepared* molecule: it carries the crystal
    # conformer and the same bond orders as the poses, which the symmetry-aware
    # RMSD needs in order to match the two graphs.
    reference = Chem.Mol(lig_mol)
    rows: List[Tuple[int, float, float, float]] = []
    for pose in result.poses:
        mol = odock.pose_to_mol(pose, lig_mol, lig_report.atom_order)
        raw, fitted = odock.aligned_rmsd(mol, reference, heavy_only=True)
        rows.append((pose.index + 1, pose.affinity, raw, fitted))
    if rows:
        rows.sort(key=lambda r: r[0])

    best = rows[0][2] if rows else float("inf")
    out = RedockResult(
        name=pdb_path.stem,
        ligand_resname=ligand_resname,
        poses=rows,
        best_rmsd=best,
        best_affinity=result.best_affinity if result.poses else float("nan"),
        crystal_affinity=float(crystal["affinity"]),
        num_tors=float(result.num_tors),
        num_atoms=lig_mol.GetNumAtoms(),
        elapsed=float(result.elapsed),
        grid_points=int(result.grid_points),
        receptor_atoms=int(rec_mol.GetNumAtoms()),
        warnings=list(rec_report.warnings) + list(lig_report.warnings),
    )
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "poses.pdbqt").write_text(result.to_pdbqt(), encoding="utf-8")
    say("")
    say(out.summary())
    return out

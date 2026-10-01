"""End-to-end validation: re-dock benzamidine into bovine trypsin (PDB 3PTB).

The script prepares both partners from the crystal structure with the
OpenDocking preparation pipeline, docks the ligand back into the
crystallographic binding site and reports the RMSD of the top pose against the
experimental pose. A correct docking engine recovers the crystal pose to well
under 2 Å.

Usage::

    python tests/validate_3ptb.py [--exhaustiveness 16] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

import odock  # noqa: E402
from rdkit import Chem  # noqa: E402

#: Residues that are part of the receptor rather than the ligand.
LIGAND_RESNAMES = {"BEN", "LIG", "STI", "UNL"}
WATER_RESNAMES = {"HOH", "WAT", "DOD"}


def split_complex(pdb_path: Path):
    """Split a crystal structure into receptor, reference ligand and waters."""
    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False, proximityBonding=True)
    if mol is None:
        raise SystemExit(f"could not read {pdb_path}")

    receptor_idx, ligand_idx = [], []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip() if info is not None else ""
        if name in WATER_RESNAMES:
            continue
        if name in LIGAND_RESNAMES:
            ligand_idx.append(atom.GetIdx())
        else:
            receptor_idx.append(atom.GetIdx())

    def subset(indices):
        em = Chem.RWMol(mol)
        keep = set(indices)
        for i in sorted(set(range(mol.GetNumAtoms())) - keep, reverse=True):
            em.RemoveAtom(i)
        out = em.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:
            pass
        return out

    receptor = subset(receptor_idx)
    ligand = subset(ligand_idx)
    if receptor.GetNumAtoms() == 0 or ligand.GetNumAtoms() == 0:
        raise SystemExit("failed to split the complex into receptor and ligand")
    return receptor, ligand


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdb", nargs="?", default=str(ROOT / "reference" / "3PTB.pdb"))
    ap.add_argument("--exhaustiveness", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer", type=float, default=8.0)
    ap.add_argument("--scoring", default="vina", choices=["vina", "vinardo", "ad4"])
    ap.add_argument("--outdir", default=str(ROOT / "out"))
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdb = Path(args.pdb)
    if not pdb.exists():
        raise SystemExit(f"missing {pdb}; download it with `curl -O https://files.rcsb.org/download/3PTB.pdb`")

    print(f"[1/6] splitting {pdb.name}")
    receptor_mol, ligand_mol = split_complex(pdb)
    print(f"      receptor: {receptor_mol.GetNumAtoms()} atoms")
    print(f"      ligand  : {ligand_mol.GetNumAtoms()} atoms")

    print("[2/6] preparing the receptor")
    rec_mol, rec_pdbqt, rec_report = odock.prepare_receptor(
        receptor_mol, outdir / "3ptb_receptor.pdbqt", keep_water=False
    )
    print(f"      {rec_report.summary()}")
    for w in rec_report.warnings:
        print(f"      warning: {w}")

    print("[3/6] preparing the ligand")
    lig_mol, lig_pdbqt, lig_report = odock.prepare_ligand(
        ligand_mol, outdir / "3ptb_ligand.pdbqt", name="BEN", optimize=False
    )
    print(f"      {lig_report.summary()}")

    print("[4/6] building the search box")
    box = odock.box_from_ligand(ligand_mol, buffer=args.buffer)
    print(f"      {box}")

    print(f"[5/6] docking ({args.scoring}, exhaustiveness={args.exhaustiveness}, seed={args.seed})")
    result = odock.dock(
        rec_pdbqt,
        lig_pdbqt,
        box,
        scoring=args.scoring,
        exhaustiveness=args.exhaustiveness,
        num_poses=9,
        seed=args.seed,
    )
    print(result.table())
    print()
    print(f"      elapsed : {result.elapsed:.2f} s")
    print(f"      N_tors  : {result.num_tors:g}")
    print(f"      grid    : {result.grid_points} points, {result.grid_mb} MB")

    poses_path = outdir / "3ptb_poses.pdbqt"
    poses_path.write_text(result.to_pdbqt(), encoding="utf-8")
    print(f"      wrote {poses_path}")

    print("[6/6] scoring the poses against the crystal structure")
    reference = Chem.Mol(ligand_mol)
    rows = []
    for pose in result.poses:
        mol = odock.pose_to_mol(pose, lig_mol, lig_report.atom_order)
        raw, fitted = odock.aligned_rmsd(mol, reference, heavy_only=True)
        rows.append((pose.index + 1, pose.affinity, raw, fitted))
    print()
    print("mode |   affinity |  RMSD (no fit) |  RMSD (fitted)")
    print("-----+------------+----------------+---------------")
    for idx, affinity, raw, fitted in rows:
        print(f"{idx:>4d} | {affinity:>10.3f} | {raw:>14.3f} | {fitted:>14.3f}")

    best_raw = rows[0][2]
    print()
    print(f"best-pose RMSD to the crystal structure: {best_raw:.3f} Å")
    if best_raw <= 2.0:
        print("RESULT: PASS — the top pose reproduces the experimental binding mode")
        return 0
    print("RESULT: FAIL — the top pose is more than 2 Å from the crystal structure")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

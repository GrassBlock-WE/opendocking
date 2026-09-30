# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest fixtures for the OpenDocking test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory holding the bundled test structures."""
    return DATA


@pytest.fixture(scope="session")
def pdb_3ptb(data_dir: Path) -> Path:
    """Bovine trypsin with benzamidine (PDB 3PTB)."""
    p = data_dir / "3PTB.pdb"
    if not p.exists():
        pytest.skip(f"missing test fixture {p}")
    return p


@pytest.fixture(scope="session")
def complex_parts(pdb_3ptb: Path):
    """``(receptor_mol, ligand_mol)`` split out of 3PTB."""
    Chem = pytest.importorskip("rdkit.Chem")  # type: ignore[attr-defined]
    from rdkit import Chem as _Chem

    mol = _Chem.MolFromPDBFile(
        str(pdb_3ptb), removeHs=False, sanitize=False, proximityBonding=True
    )
    assert mol is not None
    lig_res = {"BEN"}
    water = {"HOH", "WAT", "DOD"}
    keep_rec, keep_lig = [], []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip() if info else ""
        if name in water:
            continue
        (keep_lig if name in lig_res else keep_rec).append(atom.GetIdx())

    def subset(indices):
        em = _Chem.RWMol(mol)
        total = set(range(mol.GetNumAtoms()))
        for i in sorted(total - set(indices), reverse=True):
            em.RemoveAtom(i)
        out = em.GetMol()
        try:
            _Chem.SanitizeMol(out)
        except Exception:
            pass
        return out

    return subset(keep_rec), subset(keep_lig)


@pytest.fixture(scope="session")
def prepared_3ptb(complex_parts, tmp_path_factory):
    """Prepared receptor and ligand for 3PTB, shared by the docking tests."""
    import odock

    outdir = tmp_path_factory.mktemp("3ptb")
    receptor_mol, ligand_mol = complex_parts
    _, rec_pdbqt, rec_report = odock.prepare_receptor(
        receptor_mol, outdir / "receptor.pdbqt", keep_water=False
    )
    lig_mol, lig_pdbqt, lig_report = odock.prepare_ligand(
        ligand_mol, outdir / "ligand.pdbqt", name="BEN", optimize=False
    )
    return {
        "receptor_mol": receptor_mol,
        "ligand_mol": ligand_mol,
        "ligand_prepared": lig_mol,
        "receptor_pdbqt": rec_pdbqt,
        "ligand_pdbqt": lig_pdbqt,
        "receptor_report": rec_report,
        "ligand_report": lig_report,
        "outdir": outdir,
    }

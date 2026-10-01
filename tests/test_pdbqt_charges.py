# SPDX-License-Identifier: GPL-3.0-or-later
"""The PDBQT charge guard: a non-finite Gasteiger charge must never be written.

RDKit's Gasteiger model does not converge for a few protein atoms and reports
``nan`` or ``inf`` for them.  Before this guard, an ``inf`` charge was written
verbatim, and the AD4 force field then returned a ``NaN`` total for any pose with
an electrostatic pair touching that atom -- a silent, unreproducible failure.
The guard writes ``0.000`` and *reports* it, because losing one atom's
electrostatic term is a real approximation and must be visible.
"""

from __future__ import annotations

import math
import types

import pytest

from odock import pdbqt as _pdbqt

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


def _molecule():
    """A tiny receptor-like molecule: three atoms with residue information."""
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol()
    for index, (symbol, name) in enumerate(
        (("C", "CB"), ("O", "OD1"), ("N", "N"))
    ):
        atom = Chem.Atom(symbol)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.ljust(4))
        info.SetResidueName("ALA")
        info.SetResidueNumber(7)
        info.SetChainId("A")
        atom.SetMonomerInfo(info)
        mol.AddAtom(atom)
    out = mol.GetMol()
    Chem.SanitizeMol(out)
    conformer = Chem.Conformer(out.GetNumAtoms())
    for index in range(out.GetNumAtoms()):
        conformer.SetAtomPosition(index, Point3D(float(index) * 1.5, 0.0, 0.0))
    out.AddConformer(conformer, assignId=True)
    return out


def _patch_charges(monkeypatch, values):
    """Force ``ComputeGasteigerCharges`` to write exactly `values`."""

    def fake(mol):
        for atom in mol.GetAtoms():
            atom.SetProp("_GasteigerCharge", values[atom.GetIdx()])

    monkeypatch.setattr(
        _pdbqt, "AllChem", types.SimpleNamespace(ComputeGasteigerCharges=fake)
    )


def _charges_in(text: str):
    out = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            out.append(float(line[70:76]))
    return out


def test_a_non_finite_charge_becomes_zero_and_is_reported(monkeypatch):
    _patch_charges(monkeypatch, ["inf", "0.250", "nan"])
    with pytest.warns(UserWarning) as caught:
        charges = _pdbqt.gasteiger_charges(_molecule())
    assert charges == [0.0, 0.25, 0.0]
    message = str(caught[0].message)
    assert "2 Gasteiger charge(s)" in message
    assert "ALA7:CB" in message
    assert "ALA7:N" in message
    assert "0.000" in message


def test_finite_charges_are_left_alone_and_do_not_warn(monkeypatch, recwarn):
    _patch_charges(monkeypatch, ["-0.310", "0.000", "0.1234"])
    charges = _pdbqt.gasteiger_charges(_molecule())
    assert charges == pytest.approx([-0.310, 0.000, 0.1234])
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_the_warning_can_be_suppressed_for_a_bulk_caller(monkeypatch, recwarn):
    _patch_charges(monkeypatch, ["inf", "inf", "inf"])
    charges = _pdbqt.gasteiger_charges(_molecule(), warn=False)
    assert charges == [0.0, 0.0, 0.0]
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_a_written_receptor_never_carries_a_non_finite_charge(monkeypatch):
    _patch_charges(monkeypatch, ["inf", "-inf", "nan"])
    mol = _molecule()
    with pytest.warns(UserWarning):
        text = _pdbqt.write_receptor_pdbqt(mol)
    lowered = text.lower()
    assert "inf" not in lowered and "nan" not in lowered
    charges = _charges_in(text)
    assert charges == [0.0, 0.0, 0.0]
    assert all(math.isfinite(value) for value in charges)
    # The atom lines keep their columns: a 300-character charge would shift the
    # AutoDock type out of the record and make the document unreadable.
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            assert len(line) <= 80


def test_a_written_ligand_never_carries_a_non_finite_charge(monkeypatch):
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    _patch_charges(monkeypatch, ["inf"] * mol.GetNumAtoms())
    with pytest.warns(UserWarning):
        text = _pdbqt.write_ligand_pdbqt(mol)
    charges = _charges_in(text)
    assert charges and all(math.isfinite(value) for value in charges)
    assert set(charges) == {0.0}


def test_a_real_receptor_preparation_reports_its_lost_charges(tmp_path):
    """The regression that started this: 3PTB's receptor used to carry `inf`."""
    root = __import__("pathlib").Path(__file__).resolve().parent.parent
    pdb = root / "tests" / "data" / "3PTB.pdb"
    if not pdb.exists():  # pragma: no cover - fixture missing
        pytest.skip("3PTB.pdb is not present")

    mol = Chem.MolFromPDBFile(
        str(pdb), removeHs=False, sanitize=False, proximityBonding=True
    )
    assert mol is not None
    text = _pdbqt.write_receptor_pdbqt(mol)
    charges = _charges_in(text)
    assert charges, "the receptor should have atom records"
    assert all(math.isfinite(value) for value in charges)
    lowered = text.lower()
    assert " inf " not in lowered and " nan " not in lowered

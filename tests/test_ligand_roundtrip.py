# SPDX-License-Identifier: GPL-3.0-or-later
"""A prepared ligand must survive an RDKit round trip with its X-H bonds intact.

RDKit's proximity bonding does not connect atoms across a change of residue
name.  `prepare_ligand` used to write a ligand's heavy atoms in the deposited
residue (``BEN``) and the polar hydrogens it added in ``LIG``, so reading the
PDBQT back with RDKit silently lost every X-H bond and MMFF94 could not type the
molecule at all -- which is what made ``odock.metrics.ligand_strain`` fall back
to UFF on the bundled demo.  These tests pin the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odock import prepare

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PDB_3PTB = ROOT / "tests" / "data" / "3PTB.pdb"


def _benzamidine():
    """The co-crystallised 3PTB ligand, as the PDB gives it (no bond orders)."""
    mol = Chem.MolFromPDBFile(
        str(PDB_3PTB), removeHs=False, sanitize=False, proximityBonding=True
    )
    keep = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetPDBResidueInfo() is not None
        and atom.GetPDBResidueInfo().GetResidueName().strip() == "BEN"
    ]
    editable = Chem.RWMol(mol)
    for index in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
        editable.RemoveAtom(index)
    out = editable.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    return out


requires_3ptb = pytest.mark.skipif(
    not PDB_3PTB.exists(), reason="tests/data/3PTB.pdb is not present"
)


def _residues_of(text: str):
    seen = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[17:20].strip(), line[21:22].strip(), line[22:26].strip())
            if key not in seen:
                seen.append(key)
    return seen


def _read_back(text: str):
    block = prepare.pdbqt_to_pdb_block(text)
    mol = Chem.MolFromPDBBlock(
        block, removeHs=False, sanitize=False, proximityBonding=True
    )
    assert mol is not None
    return mol


@requires_3ptb
def test_a_prepared_ligand_has_exactly_one_residue():
    mol, text, report = prepare.prepare_ligand(
        _benzamidine(), None, name="BEN", optimize=False
    )
    assert report.atom_order
    residues = _residues_of(text)
    assert residues == [("BEN", "A", "1")], residues


@requires_3ptb
def test_an_rdkit_round_trip_keeps_every_x_h_bond():
    """13 atoms, 13 bonds: the 9 covalent framework bonds plus all 4 N-H."""
    _, text, _ = prepare.prepare_ligand(
        _benzamidine(), None, name="BEN", optimize=False
    )
    mol = _read_back(text)
    assert mol.GetNumAtoms() == 13
    assert mol.GetNumBonds() == 13
    x_h = [
        bond
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() == 1
        or bond.GetEndAtom().GetAtomicNum() == 1
    ]
    assert len(x_h) == 4
    # Every hydrogen has exactly one heavy neighbour.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            assert atom.GetDegree() == 1


@requires_3ptb
def test_the_round_tripped_ligand_is_typeable_by_a_force_field():
    """The end the fix exists for: MMFF94 can now type the prepared ligand."""
    _, text, _ = prepare.prepare_ligand(
        _benzamidine(), None, name="BEN", optimize=False
    )
    mol = _read_back(text)
    Chem.SanitizeMol(mol)
    assert AllChem.MMFFHasAllMoleculeParams(Chem.AddHs(mol, addCoords=True))


@requires_3ptb
def test_hydrogen_names_stay_unique_within_the_residue():
    """Two hydrogens on one nitrogen must not both be called `N1`."""
    _, text, _ = prepare.prepare_ligand(
        _benzamidine(), None, name="BEN", optimize=False
    )
    names = [
        line[12:16].strip()
        for line in text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    assert len(names) == len(set(names))
    assert names[-4:] == ["H10", "H11", "H12", "H13"]


@requires_3ptb
def test_strain_no_longer_needs_the_residue_workaround():
    """The freshly written ligand types under MMFF94 without unifying residues."""
    from odock import metrics

    _, text, _ = prepare.prepare_ligand(
        _benzamidine(), None, name="BEN", optimize=False
    )
    mol, notes = metrics._mol_from_pdbqt(text)
    assert mol is not None
    assert AllChem.MMFFHasAllMoleculeParams(Chem.AddHs(mol, addCoords=True))
    # The note still records that a PDBQT carries no bond orders -- that
    # limitation is real and separate from the residue bug.
    assert any("no bond orders" in note for note in notes)


# ---------------------------------------------------------------------------
# The documented strain recipe: prepare with real chemistry, hand over the map
# ---------------------------------------------------------------------------

#: A receptor far from the ligand: the ligand's own intra term is what matters.
REMOTE_RECEPTOR = """\
ATOM      1  CB  ALA A   1     -40.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  CB  ALA A   2      40.000   0.000   0.000  1.00  0.00     0.000 C
TER
"""


def test_a_smiles_prepared_ligand_is_attested_and_gives_a_plausible_strain():
    """The recipe: prepare with real bond orders, pass mol + atom_order + report."""
    from odock import metrics

    mol, text, report = prepare.prepare_ligand("NC(=N)c1ccccc1", None, name="BEN")
    assert report.chemistry_trusted is True
    strain = metrics.ligand_strain(
        text,
        receptor=REMOTE_RECEPTOR,
        scoring="vina",
        mol=mol,
        atom_order=report.atom_order,
        preparation=report,
        require_reliable=True,
    )
    assert strain.reliable is True
    # Without the molecule the same document gives tens of kcal/mol of apparent
    # strain; with it, a benzamidine congener is a few kcal/mol at most.
    assert abs(strain.force_field_strain) < 20.0

    unreliably = metrics.ligand_strain(text, receptor=REMOTE_RECEPTOR, scoring="vina")
    assert unreliably.reliable is False


@requires_3ptb
def test_a_pdb_prepared_ligand_is_not_attested():
    """A bare PDB has no bond orders, and the report says so."""
    from odock import metrics

    mol, text, report = prepare.prepare_ligand(
        _benzamidine(), None, name="BEN", optimize=False
    )
    assert report.chemistry_trusted is False
    strain = metrics.ligand_strain(
        text,
        receptor=REMOTE_RECEPTOR,
        scoring="vina",
        mol=mol,
        atom_order=report.atom_order,
        preparation=report,
    )
    assert strain.reliable is False
    assert "chemistry_trusted is False" in strain.note



def test_the_metrics_workaround_still_repairs_an_old_file():
    """A document written before the fix is still read correctly."""
    from odock import metrics

    old = (
        "ROOT\n"
        "HETATM    1  N1  BEN A   1       0.000   0.000   0.000  1.00  0.00    -0.310 N\n"
        "HETATM    2 H10  LIG A   1       1.010   0.000   0.000  1.00  0.00     0.120 HD\n"
        "ENDROOT\nTORSDOF 0\n"
    )
    mol, _notes = metrics._mol_from_pdbqt(old)
    assert mol is not None
    assert mol.GetNumBonds() == 1


def test_the_inherit_name_flag_pins_both_conventions():
    """Receptor hydrogens keep the parent's name; ligand hydrogens do not."""
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol()
    atom = Chem.Atom("N")
    info = Chem.AtomPDBResidueInfo()
    info.SetName("N   ")
    info.SetResidueName("ALA")
    info.SetResidueNumber(12)
    info.SetChainId("B")
    atom.SetMonomerInfo(info)
    mol.AddAtom(atom)
    mol.AddAtom(Chem.Atom("H"))
    mol.AddBond(0, 1, Chem.BondType.SINGLE)
    editable = mol.GetMol()
    Chem.SanitizeMol(editable)
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, Point3D(1.0, 0.0, 0.0))
    editable.AddConformer(conformer, assignId=True)
    # The hydrogen has no residue information yet -- exactly an AddHs hydrogen.
    assert editable.GetAtomWithIdx(1).GetPDBResidueInfo() is None

    receptor_style = prepare._inherit_residue_info(editable)
    inherited = receptor_style.GetAtomWithIdx(1).GetPDBResidueInfo()
    assert inherited.GetName().strip() == "N"
    assert inherited.GetResidueName().strip() == "ALA"
    assert inherited.GetResidueNumber() == 12
    assert inherited.GetChainId().strip() == "B"

    ligand_style = prepare._inherit_residue_info(editable, inherit_name=False)
    named = ligand_style.GetAtomWithIdx(1).GetPDBResidueInfo()
    assert named.GetName().strip() == ""
    assert named.GetResidueName().strip() == "ALA"
    # Which is what makes the writer fall back to its own unique H naming.
    from odock.pdbqt import _atom_name

    assert _atom_name(ligand_style, 1) == "H2"

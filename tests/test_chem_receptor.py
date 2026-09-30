# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.chem.receptor` and :mod:`odock.chem.charges`.

The molecules are built explicitly (atom records, bonds and coordinates) rather
than read from a file so that every distance the algorithms look at is under the
test's control: a histidine tautomer test is only meaningful if the test knows
exactly how far the carboxylate is.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from odock.chem import charges, receptor  # noqa: E402


# ---------------------------------------------------------------------------
# Molecule builder
# ---------------------------------------------------------------------------


def _make_mol(residues, links=()):
    """Build a molecule from explicit residue specifications.

    Each entry of `residues` is a dict with ``res``, ``res_id``, ``atoms``
    (``[(name, element), ...]``), optional ``bonds`` (name pairs inside the
    residue), ``chain``, ``hetero``, ``coords`` (name -> (x, y, z)) and ``x0``
    (the default x of the first atom; successive atoms are 1.4 Å apart so that
    the geometry is at least chemically sane).
    """
    rows = []
    bonds = []
    index = {}
    for res in residues:
        res_id = res["res_id"]
        for i, (atom_name, element) in enumerate(res["atoms"]):
            index[(res_id, atom_name)] = len(rows)
            coords = res.get("coords") or {}
            xyz = coords.get(atom_name, (res.get("x0", 0.0) + 1.4 * i, 0.0, 0.0))
            rows.append(
                (
                    atom_name,
                    element,
                    res.get("res", "UNK"),
                    res.get("chain", "A"),
                    res_id,
                    xyz,
                    res.get("hetero", True),
                )
            )
        for bond in res.get("bonds", ()):
            a, b = bond[0], bond[1]
            order = float(bond[2]) if len(bond) > 2 else 1.0
            bonds.append((index[(res_id, a)], index[(res_id, b)], order))
    for (res_a, name_a), (res_b, name_b) in links:
        bonds.append((index[(res_a, name_a)], index[(res_b, name_b)], 1.0))

    em = Chem.RWMol()
    for atom_name, element, res_name, chain, res_id, _xyz, hetero in rows:
        atom = Chem.Atom(element)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(atom_name)
        info.SetResidueName(res_name)
        info.SetChainId(chain)
        info.SetResidueNumber(res_id)
        info.SetSerialNumber(em.GetNumAtoms() + 1)
        info.SetIsHeteroAtom(hetero)
        info.SetOccupancy(1.0)
        info.SetTempFactor(0.0)
        atom.SetMonomerInfo(info)
        em.AddAtom(atom)
    for i, j, order in bonds:
        types = {1.0: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC,
                 2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE}
        em.AddBond(i, j, types.get(order, Chem.BondType.SINGLE))

    mol = em.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.Set3D(True)
    for i, row in enumerate(rows):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in row[5]]))
    mol.AddConformer(conf, assignId=True)
    return mol


def _atom_index(mol, res_id, atom_name):
    """Index of an atom identified by residue number and atom name."""
    wanted = atom_name.strip().upper()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue
        if int(info.GetResidueNumber()) == res_id and info.GetName().strip().upper() == wanted:
            return atom.GetIdx()
    raise AssertionError(f"no atom {atom_name!r} in residue {res_id}")


def _h_count(mol, idx):
    return sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1)


def _formal_charge(mol, res_id, atom_name):
    return mol.GetAtomWithIdx(_atom_index(mol, res_id, atom_name)).GetFormalCharge()


# Standard residue templates used by the tests.
ASP = {
    "res": "ASP",
    "res_id": 1,
    "x0": 0.0,
    "hetero": False,
    "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"),
              ("CB", "C"), ("CG", "C"), ("OD1", "O"), ("OD2", "O")],
    "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
              ("CG", "OD1"), ("CG", "OD2")],
}
LYS = {
    "res": "LYS",
    "res_id": 2,
    "x0": 40.0,
    "hetero": False,
    "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("OXT", "O"),
              ("CB", "C"), ("CG", "C"), ("CD", "C"), ("CE", "C"), ("NZ", "N")],
    "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"), ("CA", "CB"),
              ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")],
}
HIS_RING = {
    "CG": (1.0936, 0.3554, 0.0),
    "ND1": (0.0, 1.15, 0.0),
    "CE1": (-1.0936, 0.3554, 0.0),
    "NE2": (-0.6758, -0.9303, 0.0),
    "CD2": (0.6758, -0.9303, 0.0),
    "CB": (-2.4, 1.0, 0.0),
    "CA": (-3.6, 1.8, 0.0),
    "N": (-4.9, 2.4, 0.0),
    "C": (-3.9, 3.3, 0.0),
    "O": (-3.5, 4.6, 0.0),
}
HIS = {
    "res": "HIS",
    "res_id": 10,
    "chain": "A",
    "hetero": False,
    "coords": HIS_RING,
    "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
              ("CG", "C"), ("ND1", "N"), ("CD2", "C"), ("CE1", "C"), ("NE2", "N")],
    "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
              ("CG", "ND1"), ("ND1", "CE1"), ("CE1", "NE2"), ("NE2", "CD2"),
              ("CD2", "CG")],
}


def _mini_protein():
    """ASP1-LYS2 dipeptide: an N-terminus, a peptide bond and a C-terminus."""
    return _make_mol(
        [ASP, LYS],
        links=[((1, "C"), (2, "N"))],
    )


def _carbonyl_partner(res_id, oxygen_xyz, chain="B"):
    """A minimal ALA whose backbone O sits exactly at `oxygen_xyz`."""
    return {
        "res": "ALA",
        "res_id": res_id,
        "chain": chain,
        "hetero": False,
        "coords": {
            "O": oxygen_xyz,
            "C": (oxygen_xyz[0] + 1.25, oxygen_xyz[1], oxygen_xyz[2]),
            "CA": (oxygen_xyz[0] + 2.5, oxygen_xyz[1], oxygen_xyz[2]),
            "N": (oxygen_xyz[0] + 3.8, oxygen_xyz[1], oxygen_xyz[2]),
            "CB": (oxygen_xyz[0] + 3.0, oxygen_xyz[1] + 1.2, oxygen_xyz[2]),
        },
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    }


def _carboxylate_partner(res_id, od2_xyz, chain="B"):
    """A minimal ASP whose OD2 oxygen sits exactly at `od2_xyz`."""
    x, y, z = od2_xyz
    return {
        "res": "ASP",
        "res_id": res_id,
        "chain": chain,
        "hetero": False,
        "coords": {
            "OD2": od2_xyz,
            "CG": (x + 1.25, y, z),
            "OD1": (x + 1.25, y + 1.30, z),
            "CB": (x + 2.6, y - 0.8, z),
            "CA": (x + 4.0, y - 1.6, z),
            "N": (x + 5.4, y - 2.4, z),
            "C": (x + 4.0, y - 0.2, z),
            "O": (x + 5.2, y + 0.4, z),
        },
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"),
                  ("CB", "C"), ("CG", "C"), ("OD1", "O"), ("OD2", "O")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
                  ("CG", "OD1"), ("CG", "OD2")],
    }


def _ligand_chain(res_id=50, count=14, x0=0.0, chain="A"):
    """An aliphatic `count`-carbon chain, heavy-atom mass 14 x 12.011 Da."""
    return {
        "res": "LIG",
        "res_id": res_id,
        "chain": chain,
        "hetero": True,
        "x0": x0,
        "atoms": [(f"C{i + 1}", "C") for i in range(count)],
        "bonds": [(f"C{i + 1}", f"C{i + 2}") for i in range(count - 1)],
    }


def _water(res_id, xyz, chain="A"):
    return {
        "res": "HOH",
        "res_id": res_id,
        "chain": chain,
        "hetero": True,
        "coords": {"O": xyz},
        "atoms": [("O", "O")],
    }


# ---------------------------------------------------------------------------
# HETATM classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["HOH", "WAT", "DOD", "TIP3", "H2O", "TIP4", "SPC", "SOL"])
def test_solvent_names_are_recognised(name):
    mol = _make_mol([_water(1, (0.0, 0.0, 0.0)) | {"res": name}])
    inv = receptor.classify_hetero(mol)
    assert inv.labels() == [f"{name} A1"]
    assert inv.of_kind(receptor.KIND_SOLVENT)[0].kind == receptor.KIND_SOLVENT


def test_solvent_prefixes_cover_water_models():
    mol = _make_mol([_water(1, (0.0, 0.0, 0.0)) | {"res": "TIP3P"}])
    assert receptor.classify_hetero(mol).summary()[receptor.KIND_SOLVENT] == 1


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ZN", receptor.KIND_ION), ("MG", receptor.KIND_ION), ("CA", receptor.KIND_ION),
        ("MN", receptor.KIND_ION), ("FE", receptor.KIND_ION), ("CU", receptor.KIND_ION),
        ("CO", receptor.KIND_ION), ("NI", receptor.KIND_ION), ("CD", receptor.KIND_ION),
        ("HG", receptor.KIND_ION), ("NA", receptor.KIND_ION), ("K", receptor.KIND_ION),
        ("CS", receptor.KIND_ION), ("RB", receptor.KIND_ION), ("SR", receptor.KIND_ION),
        ("BA", receptor.KIND_ION), ("CL", receptor.KIND_ION), ("BR", receptor.KIND_ION),
        ("IOD", receptor.KIND_ION), ("F", receptor.KIND_ION), ("SO4", receptor.KIND_ION),
        ("PO4", receptor.KIND_ION), ("NO3", receptor.KIND_ION), ("ACT", receptor.KIND_ION),
        ("EDO", receptor.KIND_ION), ("GOL", receptor.KIND_ION), ("FMT", receptor.KIND_ION),
        ("MES", receptor.KIND_ION), ("TRS", receptor.KIND_ION),
    ],
)
def test_ion_lists(name, expected):
    mol = _make_mol([{"res": name, "res_id": 1, "atoms": [("X", "Zn")]}])
    residues = receptor.classify_hetero(mol).residues
    assert len(residues) == 1
    assert residues[0].kind == expected


@pytest.mark.parametrize(
    "name",
    ["HEM", "HEC", "HEME", "NAD", "NAP", "NADH", "NAH", "FAD", "FMN", "PLP", "PMP",
     "ATP", "ADP", "AMP", "GTP", "GDP", "SAM", "SAH", "COA", "TPP", "THF", "UQ",
     "MQ", "PQQ", "B12", "CLA", "BCL", "HTH"],
)
def test_cofactor_list_is_protected(name):
    atoms = [("C1", "C"), ("C2", "C"), ("C3", "C"), ("C4", "C"), ("C5", "C"),
             ("C6", "C"), ("C7", "C"), ("C8", "C"), ("C9", "C"), ("C10", "C"),
             ("C11", "C"), ("C12", "C"), ("C13", "C"), ("O1", "O"), ("N1", "N")]
    mol = _make_mol([{"res": name, "res_id": 1, "atoms": atoms}])
    inv = receptor.classify_hetero(mol)
    assert inv.residues[0].kind == receptor.KIND_COFACTOR, name
    # The heavy-atom mass alone would have made it a ligand.
    assert inv.residues[0].mass > receptor.LIGAND_MIN_MASS


def test_standard_and_modified_residues_are_not_hetero():
    ala = {
        "res": "ALA", "res_id": 1, "hetero": False,
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    }
    mse = {
        "res": "MSE", "res_id": 2, "hetero": True,
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                  ("CG", "C"), ("SE", "Se"), ("CE", "C")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
                  ("CG", "SE"), ("SE", "CE")],
    }
    inv = receptor.classify_hetero(_make_mol([ala, mse]))
    assert inv.residues == [], "polymer residues must not appear in the inventory"


def test_ligand_detection_by_mass_only():
    """A 168 Da, 14-heavy-atom residue is a ligand; small ones are not."""
    big = _ligand_chain(res_id=1)
    small = {"res": "SM1", "res_id": 2, "atoms": [("C1", "C"), ("C2", "C")],
             "bonds": [("C1", "C2")]}
    single = {"res": "X1", "res_id": 3, "atoms": [("PB", "Pb")]}
    inv = receptor.classify_hetero(_make_mol([big, small, single]))
    kinds = {res.name: res.kind for res in inv.residues}
    assert kinds["LIG"] == receptor.KIND_LIGAND
    assert kinds["SM1"] == receptor.KIND_OTHER
    # More than one heavy atom is required, whatever the mass.
    assert kinds["X1"] == receptor.KIND_OTHER
    by_name = {res.name: res for res in inv.residues}
    assert by_name["LIG"].heavy_atoms == 14
    assert by_name["LIG"].mass == pytest.approx(14 * 12.011, abs=0.1)


def test_inventory_order_labels_and_summary():
    mol = _make_mol([
        _water(3, (60.0, 0.0, 0.0), chain="B"),
        _ligand_chain(res_id=5, chain="A"),
        {"res": "ZN", "res_id": 1, "chain": "A", "atoms": [("ZN", "Zn")]},
        {"res": "HEM", "res_id": 9, "chain": "A",
         "atoms": [(f"C{i}", "C") for i in range(16)] + [("FE", "Fe")]},
    ])
    inv = receptor.classify_hetero(mol)
    assert [res.label for res in inv.residues] == [
        "ZN A1", "LIG A5", "HEM A9", "HOH B3"
    ], "the inventory must be sorted by chain and residue id"
    assert inv.labels(receptor.KIND_ION) == ["ZN A1"]
    summary = inv.summary()
    assert set(summary) == set(receptor.KINDS)
    assert summary[receptor.KIND_ION] == 1
    assert summary[receptor.KIND_LIGAND] == 1
    assert summary[receptor.KIND_COFACTOR] == 1
    assert summary[receptor.KIND_SOLVENT] == 1
    assert summary[receptor.KIND_OTHER] == 0
    assert len(inv) == 4


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _clean_counts(before, cleaned):
    return before.GetNumAtoms() - cleaned.GetNumAtoms()


def test_clean_receptor_drops_solvent_ions_and_ligand():
    mol = _make_mol([
        _ligand_chain(res_id=1),
        {"res": "ZN", "res_id": 2, "x0": 80.0, "atoms": [("ZN", "Zn")]},
        _water(3, (60.0, 20.0, 0.0)),
        _water(4, (60.0, 40.0, 0.0)),
        LYS | {"res_id": 5, "x0": 100.0},
    ])
    cleaned, log, inv = receptor.clean_receptor(mol)
    assert _clean_counts(mol, cleaned) == 14 + 1 + 1 + 1
    text = "\n".join(log)
    assert "co-crystal ligand LIG A1" in text
    assert "free ion ZN A2" in text
    assert "solvent" in text
    # the input inventory is returned, not the cleaned one
    assert inv.summary()[receptor.KIND_LIGAND] == 1
    assert cleaned.GetNumAtoms() == 10  # the LYS


def test_clean_receptor_keeps_everything_when_no_hetero():
    mol = _make_mol([ASP | {"res_id": 1}])
    cleaned, log, inv = receptor.clean_receptor(mol)
    assert cleaned.GetNumAtoms() == mol.GetNumAtoms()
    assert inv.residues == []
    assert "unchanged" in log[0]


def test_structural_water_retention_uses_the_ligand_distance():
    """3.4 Å is kept, 3.6 Å is dropped, with a 3.5 Å cutoff."""
    waters = [_water(10, (7.0, 3.4, 0.0)), _water(11, (7.0, 3.6, 0.0))]
    mol = _make_mol([_ligand_chain(res_id=1), *waters])
    cleaned, log, _inv = receptor.clean_receptor(
        mol, receptor.CleanPolicy(keep_water_within=3.5, drop_ligands=False)
    )
    text = "\n".join(log)
    assert "kept 1 structural waters within 3.5 A of LIG A1, removed 1" in text
    # 14 ligand carbons + one water
    assert cleaned.GetNumAtoms() == 15
    kept = receptor.classify_hetero(cleaned).of_kind(receptor.KIND_SOLVENT)
    assert [res.res_id for res in kept] == [10]


def test_water_retention_without_a_ligand_drops_every_water():
    mol = _make_mol([_water(10, (0.0, 0.0, 0.0))])
    cleaned, log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(keep_water_within=3.5)
    )
    assert cleaned.GetNumAtoms() == 0
    assert "no ligand is present" in "\n".join(log)


def test_drop_solvent_false_keeps_the_waters():
    mol = _make_mol([_ligand_chain(res_id=1), _water(2, (100.0, 0.0, 0.0))])
    cleaned, _log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(drop_solvent=False, drop_ligands=False)
    )
    assert cleaned.GetNumAtoms() == 15


def test_keep_water_within_must_be_positive():
    mol = _make_mol([_water(1, (0.0, 0.0, 0.0))])
    with pytest.raises(ValueError):
        receptor.clean_receptor(mol, receptor.CleanPolicy(keep_water_within=0.0))


def test_cofactor_protection_and_removal():
    heme = {"res": "HEM", "res_id": 4, "x0": 50.0,
            "atoms": [(f"C{i}", "C") for i in range(16)] + [("FE", "Fe")],
            "bonds": [(f"C{i}", f"C{i + 1}") for i in range(15)]}
    mol = _make_mol([_ligand_chain(res_id=1), heme])
    kept, log, _ = receptor.clean_receptor(mol)
    assert "kept cofactor HEM A4" in "\n".join(log)
    assert kept.GetNumAtoms() == 17

    dropped, log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(keep_cofactors=False)
    )
    assert dropped.GetNumAtoms() == 0
    assert "removed cofactor HEM A4" in "\n".join(log)


def test_keep_residues_protects_an_ion_by_label_or_name():
    mol = _make_mol([
        {"res": "ZN", "res_id": 301, "atoms": [("ZN", "Zn")]},
        {"res": "ZN", "res_id": 302, "x0": 20.0, "atoms": [("ZN", "Zn")]},
    ])
    cleaned, log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(keep_residues=["ZN A301"])
    )
    assert cleaned.GetNumAtoms() == 1
    assert "kept ZN A301 (protected" in "\n".join(log)
    assert "removed free ion ZN A302" in "\n".join(log)

    both, _log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(keep_residues=["zn"])
    )
    assert both.GetNumAtoms() == 2


def test_drop_free_ions_false_keeps_ions():
    mol = _make_mol([{"res": "MG", "res_id": 1, "atoms": [("MG", "Mg")]}])
    cleaned, _log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(drop_free_ions=False)
    )
    assert cleaned.GetNumAtoms() == 1


def test_drop_ligands_false_keeps_the_ligand():
    mol = _make_mol([_ligand_chain(res_id=1)])
    cleaned, log, _ = receptor.clean_receptor(
        mol, receptor.CleanPolicy(drop_ligands=False)
    )
    assert cleaned.GetNumAtoms() == 14
    assert "kept co-crystal ligand LIG A1" in "\n".join(log)


def test_extract_ligands_returns_the_residue_with_its_coordinates():
    mol = _make_mol([_ligand_chain(res_id=1), _ligand_chain(res_id=2, x0=30.0)])
    extracted = receptor.extract_ligands(mol)
    assert [label for label, _ in extracted] == ["LIG A1", "LIG A2"]
    label, sub = extracted[0]
    assert label == "LIG A1"
    assert sub.GetNumAtoms() == 14
    assert sub.GetNumBonds() == 13
    conf = sub.GetConformer()
    first = conf.GetAtomPosition(0)
    assert (first.x, first.y, first.z) == pytest.approx((0.0, 0.0, 0.0))
    last = conf.GetAtomPosition(13)
    assert last.x == pytest.approx(13 * 1.4)
    # residue information survives the extraction
    info = sub.GetAtomWithIdx(0).GetPDBResidueInfo()
    assert info.GetResidueName().strip() == "LIG"
    assert int(info.GetResidueNumber()) == 1


def test_extract_ligands_accepts_a_supplied_inventory():
    mol = _make_mol([_ligand_chain(res_id=1)])
    inv = receptor.classify_hetero(mol)
    assert len(receptor.extract_ligands(mol, inv)) == 1
    empty = receptor.HeteroInventory()
    assert receptor.extract_ligands(mol, empty) == []


# ---------------------------------------------------------------------------
# Missing atoms
# ---------------------------------------------------------------------------


def test_missing_atoms_reports_incomplete_side_chains():
    broken = ASP | {"res_id": 1}
    broken = dict(broken)
    broken["atoms"] = [a for a in broken["atoms"] if a[0] != "OD2"]
    broken["bonds"] = [b for b in broken["bonds"] if "OD2" not in b]
    ala = {
        "res": "ALA", "res_id": 2, "x0": 40.0, "hetero": False,
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")],
    }
    lines = receptor.missing_atoms(_make_mol([broken, ala]))
    assert lines == ["ASP A1 missing OD2"]


def test_missing_atoms_accepts_backbone_oxygen_aliases():
    ala = {
        "res": "ALA", "res_id": 1, "hetero": False,
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("OT1", "O"), ("CB", "C")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "OT1"), ("CA", "CB")],
    }
    assert receptor.missing_atoms(_make_mol([ala])) == []


def test_missing_atoms_ignores_non_standard_residues():
    tiny = {"res": "SM1", "res_id": 1, "atoms": [("C1", "C")]}
    assert receptor.missing_atoms(_make_mol([tiny])) == []


def test_missing_atoms_reports_several_residues_in_order():
    gly = {
        "res": "GLY", "res_id": 2, "chain": "B", "hetero": False,
        "atoms": [("N", "N"), ("CA", "C")], "bonds": [("N", "CA")],
    }
    asp = dict(ASP)
    asp["atoms"] = [a for a in asp["atoms"] if a[0] not in ("CG", "OD1", "OD2")]
    asp["bonds"] = [b for b in asp["bonds"] if "CG" not in b and "OD1" not in b and "OD2" not in b]
    lines = receptor.missing_atoms(_make_mol([gly, asp]))
    assert lines == [
        "ASP A1 missing CG, OD1, OD2",
        "GLY B2 missing C, O",
    ]


# ---------------------------------------------------------------------------
# pH-aware protonation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ph,expected_charge,expected_h",
    [(2.0, 2, 9), (7.4, 0, 7), (12.0, -2, 5)],
)
def test_protonate_pH_states_of_the_dipeptide(ph, expected_charge, expected_h):
    mol = _mini_protein()
    out, log = receptor.protonate(mol, ph=ph)
    assert Chem.GetFormalCharge(out) == expected_charge, "\n".join(log)
    hydrogens = sum(1 for a in out.GetAtoms() if a.GetAtomicNum() == 1)
    assert hydrogens == expected_h
    # Only N/O/S-bound hydrogens are ever added.
    for atom in out.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        assert parent.GetAtomicNum() in (7, 8, 16)


def test_protonate_asp_and_lys_states():
    out, log = receptor.protonate(_mini_protein(), ph=7.4)
    # Aspartate: exactly one oxygen carries the -1, neither carries a hydrogen.
    od1 = _atom_index(out, 1, "OD1")
    od2 = _atom_index(out, 1, "OD2")
    assert sorted([out.GetAtomWithIdx(od1).GetFormalCharge(),
                   out.GetAtomWithIdx(od2).GetFormalCharge()]) == [-1, 0]
    assert _h_count(out, od1) == 0 and _h_count(out, od2) == 0
    # Lysine: NZ is an ammonium with three hydrogens.
    nz = _atom_index(out, 2, "NZ")
    assert out.GetAtomWithIdx(nz).GetFormalCharge() == 1
    assert _h_count(out, nz) == 3
    # The peptide nitrogen carries one amide hydrogen.
    assert _h_count(out, _atom_index(out, 2, "N")) == 1
    # The N-terminus is an ammonium, the C-terminus a carboxylate.
    assert out.GetAtomWithIdx(_atom_index(out, 1, "N")).GetFormalCharge() == 1
    assert _h_count(out, _atom_index(out, 1, "N")) == 3
    assert out.GetAtomWithIdx(_atom_index(out, 2, "OXT")).GetFormalCharge() == -1
    assert _h_count(out, _atom_index(out, 2, "OXT")) == 0
    assert "N-terminus +1" in "\n".join(log)
    assert "C-terminus -1" in "\n".join(log)


def test_protonate_low_pH_protonates_the_carboxylates():
    out, _log = receptor.protonate(_mini_protein(), ph=2.0)
    od1 = _atom_index(out, 1, "OD1")
    od2 = _atom_index(out, 1, "OD2")
    assert out.GetAtomWithIdx(od1).GetFormalCharge() == 0
    assert out.GetAtomWithIdx(od2).GetFormalCharge() == 0
    assert _h_count(out, od1) + _h_count(out, od2) == 1, "exactly one COOH hydrogen"
    assert _h_count(out, _atom_index(out, 2, "NZ")) == 3
    assert _h_count(out, _atom_index(out, 2, "OXT")) == 1


def test_protonate_high_pH_deprotonates_lysine():
    out, _log = receptor.protonate(_mini_protein(), ph=12.0)
    nz = _atom_index(out, 2, "NZ")
    assert out.GetAtomWithIdx(nz).GetFormalCharge() == 0
    assert _h_count(out, nz) == 2
    assert out.GetAtomWithIdx(_atom_index(out, 1, "N")).GetFormalCharge() == 0
    assert _h_count(out, _atom_index(out, 1, "N")) == 2


def test_protonate_arg_is_still_charged_at_pH_12():
    arg = {
        "res": "ARG", "res_id": 3, "hetero": False,
        "atoms": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                  ("CG", "C"), ("CD", "C"), ("NE", "N"), ("CZ", "C"),
                  ("NH1", "N"), ("NH2", "N")],
        "bonds": [("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
                  ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"),
                  ("CZ", "NH2")],
    }
    out, _log = receptor.protonate(_make_mol([arg]), ph=12.0)
    assert _formal_charge(out, 3, "CZ") == 1
    assert _h_count(out, _atom_index(out, 3, "NH1")) == 2
    assert _h_count(out, _atom_index(out, 3, "NH2")) == 2
    out_neutral, _log = receptor.protonate(_make_mol([arg]), ph=13.5)
    assert _formal_charge(out_neutral, 3, "CZ") == 0
    assert _h_count(out_neutral, _atom_index(out_neutral, 3, "NH1")) == 1


def _his_ring_hydrogens(mol, res_id=10):
    return {
        name: _h_count(mol, _atom_index(mol, res_id, name)) for name in ("ND1", "NE2")
    }


def test_protonate_his_defaults_to_hie():
    out, log = receptor.protonate(_make_mol([HIS]), ph=7.4)
    assert _his_ring_hydrogens(out) == {"ND1": 0, "NE2": 1}
    assert "HIE" in "\n".join(log)


def test_protonate_his_is_hid_when_only_nd1_has_a_partner():
    ne2 = HIS_RING["NE2"]
    nd1 = HIS_RING["ND1"]
    # Direction from ND1 away from NE2, so the carbonyl cannot also reach NE2.
    away = (0.3090, 0.9511)
    oxygen = (nd1[0] + 3.0 * away[0], nd1[1] + 3.0 * away[1], 0.0)
    mol = _make_mol([HIS, _carbonyl_partner(20, oxygen)])
    out, log = receptor.protonate(mol, ph=7.4)
    assert _his_ring_hydrogens(out) == {"ND1": 1, "NE2": 0}
    assert "HID" in "\n".join(log)


def test_protonate_his_is_hie_when_only_ne2_has_a_partner():
    ne2 = HIS_RING["NE2"]
    away = (-0.3090, -0.9511)
    oxygen = (ne2[0] + 3.0 * away[0], ne2[1] + 3.0 * away[1], 0.0)
    mol = _make_mol([HIS, _carbonyl_partner(21, oxygen)])
    out, log = receptor.protonate(mol, ph=7.4)
    assert _his_ring_hydrogens(out) == {"ND1": 0, "NE2": 1}
    assert "partner at NE2" in "\n".join(log)


def test_protonate_his_is_hip_next_to_a_carboxylate():
    ne2 = HIS_RING["NE2"]
    mol = _make_mol([HIS, _carboxylate_partner(22, (ne2[0] + 2.7, ne2[1], 0.0))])
    out, log = receptor.protonate(mol, ph=7.4)
    assert _his_ring_hydrogens(out) == {"ND1": 1, "NE2": 1}
    assert "HIP" in "\n".join(log)
    ring_charge = (
        _formal_charge(out, 10, "ND1") + _formal_charge(out, 10, "NE2")
    )
    assert ring_charge == 1


def test_protonate_his_hip_at_low_pH():
    out, log = receptor.protonate(_make_mol([HIS]), ph=2.0)
    assert _his_ring_hydrogens(out) == {"ND1": 1, "NE2": 1}
    assert "pH 2 < pKa 6" in "\n".join(log)


#: The kekule structure RDKit perceives for the hydrogen-less imidazole of a
#: crystal structure: the double bonds sit on ND1=CE1 and CG=CD2, which leaves
#: NE2 as the pyrrole-type nitrogen. That is exactly the HIE pattern, so a HID
#: assignment (a hydrogen on the double-bonded ND1) forces a re-kekulisation.
HIS_CRYSTAL_BONDS = [
    ("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "CG"),
    ("CG", "ND1"), ("ND1", "CE1", 2.0), ("CE1", "NE2"),
    ("NE2", "CD2"), ("CD2", "CG", 2.0),
]


def _over_valent_neutral(mol):
    """Neutral N/O atoms whose kekule valence exceeds the neutral maximum.

    Sulfur is deliberately excluded: it is legitimately hypervalent (a sulfoxide
    carries four bonds, a sulfone six), so a valence limit would produce false
    positives there.
    """
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    limits = {7: 3.0, 8: 2.0}
    out = []
    for atom in kek.GetAtoms():
        limit = limits.get(atom.GetAtomicNum())
        if limit is None or atom.GetFormalCharge() != 0:
            continue
        valence = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
        if valence > limit + 1e-6:
            out.append((atom.GetIdx(), atom.GetSymbol(), valence))
    return out


def test_protonate_repairs_the_kekule_structure_of_a_ring_nitrogen():
    """A hydrogen added to a double-bonded ring N must move the double bond.

    A crystal structure has no hydrogens, so RDKit's bond perception places the
    imidazole double bonds on ND1 and CG. Choosing HID then puts a hydrogen on
    ND1, which is four-valent unless the ring is re-kekulised with the hydrogen
    present — a neutral four-valent nitrogen does not exist.
    """
    his = dict(HIS)
    his["bonds"] = HIS_CRYSTAL_BONDS
    nd1 = HIS_RING["ND1"]
    away = (0.3090, 0.9511)
    oxygen = (nd1[0] + 3.0 * away[0], nd1[1] + 3.0 * away[1], 0.0)
    mol = _make_mol([his, _carbonyl_partner(20, oxygen)])
    # Sanity: the input really does carry the double bond on ND1.
    assert any(
        b.GetBondTypeAsDouble() == 2.0
        for b in mol.GetAtomWithIdx(_atom_index(mol, 10, "ND1")).GetBonds()
    )
    out, log = receptor.protonate(mol, ph=7.4)
    assert "HID" in "\n".join(log)
    assert _over_valent_neutral(out) == []
    assert _his_ring_hydrogens(out) == {"ND1": 1, "NE2": 0}
    assert _formal_charge(out, 10, "ND1") == 0
    assert _formal_charge(out, 10, "NE2") == 0
    # The ring is aromatic again for the downstream `A` typing.
    assert sum(1 for a in out.GetAtoms() if a.GetIsAromatic()) >= 5


def test_protonate_repairs_the_kekule_structure_of_an_imidazolium():
    his = dict(HIS)
    his["bonds"] = HIS_CRYSTAL_BONDS
    mol = _make_mol([his])
    out, _log = receptor.protonate(mol, ph=2.0)
    assert _over_valent_neutral(out) == []
    assert _his_ring_hydrogens(out) == {"ND1": 1, "NE2": 1}
    ring_charge = _formal_charge(out, 10, "ND1") + _formal_charge(out, 10, "NE2")
    assert ring_charge == 1, "an imidazolium carries exactly one positive charge"
    assert _formal_charge(out, 10, "ND1") == 1, "the iminium nitrogen carries the +1"


def test_protonate_removes_hydrogens_that_contradict_the_state():
    """A carboxylate that arrives protonated is deprotonated at pH 7.4."""
    asp = dict(ASP)
    asp["atoms"] = ASP["atoms"] + [("HD2", "H")]
    asp["bonds"] = ASP["bonds"] + [("OD2", "HD2")]
    asp["coords"] = {"OD2": (2.0, 0.0, 0.0), "HD2": (2.8, 0.0, 0.0)}
    out, log = receptor.protonate(_make_mol([asp]), ph=7.4)
    assert _h_count(out, _atom_index(out, 1, "OD2")) == 0
    assert "removed 1 hydrogens" in "\n".join(log)
    # The single aspartate is its own N-terminus (+1) and C-terminus (-1).
    assert Chem.GetFormalCharge(out) == 0


def test_protonate_rejects_a_pH_outside_the_range():
    mol = _make_mol([HIS])
    for bad in (-0.1, 14.1, float("nan"), 15):
        with pytest.raises(ValueError):
            receptor.protonate(mol, ph=bad)


def test_protonate_without_residue_information_uses_the_generic_rule():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    out, log = receptor.protonate(mol, ph=7.4)
    assert "no PDB residue information" in "\n".join(log)
    # The hydroxyl keeps its hydrogen; no new hydrogen is invented on carbon.
    oh = [a for a in out.GetAtoms() if a.GetAtomicNum() == 8][0]
    assert _h_count(out, oh.GetIdx()) == 1


# ---------------------------------------------------------------------------
# United-atom hydrogens
# ---------------------------------------------------------------------------


def test_strip_nonpolar_hydrogens_conserves_charge():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    before = float(charges.gasteiger_charges(mol).sum())
    stripped, removed = receptor.strip_nonpolar_hydrogens(mol)
    assert removed == sum(
        1
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1
        and a.GetNeighbors()[0].GetAtomicNum() == 6
    )
    assert stripped.GetNumAtoms() == mol.GetNumAtoms() - removed
    after_values = charges.gasteiger_charges(stripped)
    assert after_values.sum() == pytest.approx(before, abs=1e-9)
    # Every polar hydrogen survives.
    polar = [
        a.GetIdx()
        for a in stripped.GetAtoms()
        if a.GetAtomicNum() == 1
    ]
    assert all(
        stripped.GetAtomWithIdx(i).GetNeighbors()[0].GetAtomicNum() in (7, 8, 16)
        for i in polar
    )


def test_strip_nonpolar_hydrogens_moves_the_charge_onto_the_carbon():
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    original = charges.gasteiger_charges(mol)
    carbon = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6][0]
    methyl_h = [
        n.GetIdx() for n in mol.GetAtomWithIdx(carbon).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    expected = original[carbon] + sum(original[i] for i in methyl_h)
    stripped, removed = receptor.strip_nonpolar_hydrogens(mol)
    assert removed == 3
    assert charges.gasteiger_charges(stripped)[carbon] == pytest.approx(expected, abs=1e-9)
    # The value is published on the atom as well.
    assert stripped.GetAtomWithIdx(carbon).GetDoubleProp("_GasteigerCharge") == pytest.approx(
        expected, abs=1e-9
    )


def test_strip_nonpolar_hydrogens_without_collapse_leaves_charges_alone():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCC"))
    AllChem.EmbedMolecule(mol, randomSeed=5)
    stripped, removed = receptor.strip_nonpolar_hydrogens(mol, collapse_charges=False)
    assert removed == 8
    assert stripped.GetNumAtoms() == 3
    assert not stripped.GetAtomWithIdx(0).HasProp("_GasteigerCharge")
    # A molecule with no carbon-bound hydrogen is returned unchanged.
    copied, count = receptor.strip_nonpolar_hydrogens(Chem.AddHs(Chem.MolFromSmiles("O")))
    assert count == 0
    assert copied.GetNumAtoms() == 3


# ---------------------------------------------------------------------------
# Charges and AD4 typing
# ---------------------------------------------------------------------------


def _embedded(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    return mol


def test_ad4_types_contain_the_full_required_dictionary():
    required = {"C", "A", "N", "NA", "OA", "S", "SA", "HD", "F", "Cl", "Br", "I"}
    seen = set()
    for smiles in (
        "Clc1ccccc1S",   # halogen + sulfur in one molecule
        "CS(C)=O",       # sulfur that is not an acceptor -> S
        "CC(=O)N",       # amide N -> N, carbonyl -> OA
        "c1ccncc1",      # pyridine -> NA
        "FC", "ClC", "c1ccccc1Br", "c1ccccc1I",
    ):
        seen.update(charges.ad4_types(_embedded(smiles)))
    assert required <= seen
    assert all(t in charges.AD4_TYPES for t in seen)


def test_ad4_types_of_a_halogenated_thiophenol():
    mol = _embedded("Clc1ccccc1S")
    summary = charges.ad4_type_summary(mol)
    assert summary["A"] == 6
    assert summary["Cl"] == 1
    assert summary["SA"] == 1
    assert summary["HD"] == 1, "the S-H is the only polar hydrogen"
    assert summary["W"] == 4, "the four aromatic C-H carry no type"
    assert sum(summary.values()) == mol.GetNumAtoms()


def test_ad4_types_sentinel_for_untyped_hydrogens_and_elements():
    methane = Chem.AddHs(Chem.MolFromSmiles("C"))
    assert charges.ad4_types(methane) == ["C", "W", "W", "W", "W"]
    # Sodium has no AD4 type: the kernel expects the sentinel.
    sodium = Chem.MolFromSmiles("[Na+]")
    assert charges.ad4_types(sodium) == ["W"]
    # Selenium is docked as sulfur (Vina's atom_equivalence_data).
    assert charges.ad4_types(_embedded("C[SeH]"))[1] == "S"


def test_ad4_types_amide_nitrogen_is_not_an_acceptor():
    mol = _embedded("CC(=O)N")
    summary = charges.ad4_type_summary(mol)
    assert summary.get("N") == 1
    assert summary.get("NA") is None


def test_gasteiger_charges_are_finite_and_conserve_the_net_charge():
    for smiles in ("CCO", "CC(=O)[O-]", "C[N+](C)(C)C", "c1ccccc1"):
        mol = _embedded(smiles)
        values = charges.gasteiger_charges(mol)
        assert values.shape == (mol.GetNumAtoms(),)
        assert values.dtype == np.float64
        assert np.isfinite(values).all()
        assert values.sum() == pytest.approx(float(Chem.GetFormalCharge(mol)), abs=1e-6)


def test_gasteiger_charges_degrade_to_zero_on_a_protein(pdb_3ptb):
    """A model that cannot parametrise the molecule returns zeros, not noise.

    RDKit reports ``nan``/``inf`` for a fraction of a hydrogen-less protein and
    the surviving values no longer conserve the total charge; the documented
    behaviour is an all-zero set (charge-conserving, "nothing to say") rather
    than a partial set that would bias the AD4 electrostatics.
    """
    mol = Chem.MolFromPDBFile(
        str(pdb_3ptb), removeHs=False, sanitize=False, proximityBonding=True
    )
    values = charges.gasteiger_charges(mol)
    assert values.shape == (mol.GetNumAtoms(),)
    assert np.isfinite(values).all(), "NaN/inf must never propagate"
    assert not values.any(), "an unconverged model must yield zeros"
    # The united-atom model uses the same guarded base and is equally clean.
    kollman = charges.kollman_charges(mol)
    assert np.isfinite(kollman).all()
    assert float(kollman.sum()) == 0.0


def test_kollman_charges_are_united_atom_and_charge_conserving():
    mol = _embedded("CCO")
    values = charges.kollman_charges(mol)
    assert values.shape == (mol.GetNumAtoms(),)
    assert values.sum() == pytest.approx(0.0, abs=1e-9)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        if parent.GetAtomicNum() == 6:
            assert values[atom.GetIdx()] == 0.0, "non-polar H merged away"


def test_kollman_charges_use_mmff94_when_available():
    mol = _embedded("CCO")
    props = AllChem.MMFFGetMoleculeProperties(mol)
    mmff = np.array(
        [props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
    )
    expected = mmff.copy()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        hydrogen, heavy = (a, b) if a.GetAtomicNum() == 1 else (b, a)
        if hydrogen.GetAtomicNum() == 1 and heavy.GetAtomicNum() == 6:
            expected[heavy.GetIdx()] += mmff[hydrogen.GetIdx()]
            expected[hydrogen.GetIdx()] = 0.0
    assert charges.kollman_charges(mol) == pytest.approx(expected, abs=1e-9)


def test_assign_charges_dispatch():
    mol = _embedded("CCO")
    assert charges.assign_charges(mol, "gasteiger") == pytest.approx(
        charges.gasteiger_charges(mol)
    )
    assert charges.assign_charges(mol, "KOLLMAN") == pytest.approx(
        charges.kollman_charges(mol)
    )
    assert charges.assign_charges(mol, "mmff94").shape == (mol.GetNumAtoms(),)
    assert charges.assign_charges(mol, "none") == pytest.approx(np.zeros(mol.GetNumAtoms()))
    with pytest.raises(ValueError):
        charges.assign_charges(mol, "kollman-ish")


# ---------------------------------------------------------------------------
# Integration on the bundled structure
# ---------------------------------------------------------------------------


def test_3ptb_classification_pins_the_ligand_mass_threshold(pdb_3ptb):
    """The real fixture: 62 waters, one calcium, and benzamidine.

    Benzamidine is C7H8N2 — **112 Da of heavy atoms**, below the 150 Da ligand
    threshold the brief specifies. The test pins that behaviour deliberately:
    the classification follows the frozen rule, so a sub-threshold ligand lands
    in :data:`KIND_OTHER` and is kept by the default policy. Callers that must
    strip such a residue have to name it explicitly.
    """
    mol = Chem.MolFromPDBFile(
        str(pdb_3ptb), removeHs=False, sanitize=False, proximityBonding=True
    )
    inv = receptor.classify_hetero(mol)
    summary = inv.summary()
    assert summary[receptor.KIND_SOLVENT] == 62
    assert summary[receptor.KIND_ION] == 1
    assert inv.of_kind(receptor.KIND_ION)[0].label == "CA A480"
    ben = [res for res in inv.residues if res.name == "BEN"]
    assert len(ben) == 1
    assert ben[0].kind == receptor.KIND_OTHER
    assert ben[0].mass == pytest.approx(112.1, abs=0.5)
    assert ben[0].mass < receptor.LIGAND_MIN_MASS


def test_3ptb_cleaning_classification_and_protonation(complex_parts):
    receptor_mol, _ligand_mol = complex_parts
    inv = receptor.classify_hetero(receptor_mol)
    # `complex_parts` has already split the benzamidine out of the receptor.
    assert inv.summary()[receptor.KIND_SOLVENT] == 0
    assert all(res.name != "BEN" for res in inv.residues)

    cleaned, log, _inv = receptor.clean_receptor(receptor_mol)
    assert cleaned.GetNumAtoms() <= receptor_mol.GetNumAtoms()
    assert log

    missing = receptor.missing_atoms(receptor_mol)
    assert isinstance(missing, list)
    assert all(" missing " in line for line in missing)

    charges_low, _ = receptor.protonate(receptor_mol, ph=2.0)
    charges_mid, log_mid = receptor.protonate(receptor_mol, ph=7.4)
    charges_high, _ = receptor.protonate(receptor_mol, ph=12.0)
    low = Chem.GetFormalCharge(charges_low)
    mid = Chem.GetFormalCharge(charges_mid)
    high = Chem.GetFormalCharge(charges_high)
    assert low > mid > high, "lowering the pH must add positive charge"
    assert any("ASP" in line for line in log_mid)
    assert any("LYS" in line for line in log_mid)
    # Every added hydrogen sits on N, O or S.
    for atom in charges_mid.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        assert parent.GetAtomicNum() in (7, 8, 16)
    # The protonated receptor is chemically valid everywhere: no neutral atom is
    # over-valent, and the trypsin catalytic triad is recognised by name.
    assert _over_valent_neutral(charges_mid) == []
    his57 = [line for line in log_mid if line.startswith("HIS A57")]
    assert his57 and "HIP" in his57[0], "the Asp102-His57 pair must give HIP"
    # The other two histidines of trypsin are neutral and get opposite
    # tautomers, which is only possible with a kekule repair (HIS91 has its
    # H-bond partner on ND1, HIS40 has none).
    assert any(line.startswith("HIS A40") and "HIE" in line for line in log_mid)
    assert any(line.startswith("HIS A91") and "HID" in line for line in log_mid)
    charges_by_name = {}
    for atom in charges_mid.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or atom.GetAtomicNum() != 7 or atom.GetFormalCharge() == 0:
            continue
        charges_by_name.setdefault(info.GetResidueName().strip(), []).append(
            info.GetName().strip()
        )
    # Exactly one of the three histidines is the charged (HIP) one, and both
    # guanidinium charges sit on the double-bonded terminal nitrogen.
    assert charges_by_name.get("HIS") == ["ND1"]
    assert sorted(charges_by_name.get("ARG", [])) == ["NH2", "NH2"]
    # Tryptophan's indole N-H is neutral; the input's perceived kekule structure
    # had a double bond on NE1, so this pins the ring repair as well.
    assert "NE1" not in charges_by_name.get("TRP", [])

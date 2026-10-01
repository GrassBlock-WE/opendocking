# SPDX-License-Identifier: GPL-3.0-or-later
"""Ligand chemistry and the library pre-filters.

Covers the frozen interface:

* ``odock.chem.ligand`` — ``read_ligands`` (every supported format, batch
  aware), ``embed_3d``, ``minimize``, ``rotatable_bonds`` / ``rotation_reason``
  (one test per exclusion rule, each asserting the *reason* so a rule cannot be
  satisfied by accident) and ``torsion_tree``;
* ``odock.filters`` — ``lipinski``, ``veber``, ``pains``, ``drug_like`` and
  ``filter_library``, each with a pass and a fail case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from odock import filters  # noqa: E402
from odock.chem.ligand import (  # noqa: E402
    DEFAULT_SEED,
    FORCE_FIELDS,
    MAX_STEPS,
    MIN_STEPS,
    REASON_NOTES,
    RIGID_AMIDE,
    RIGID_AROMATIC,
    RIGID_BOND_ORDER,
    RIGID_HYDROGEN,
    RIGID_LOCKED,
    RIGID_RING,
    RIGID_SYMMETRIC,
    RIGID_TERMINAL,
    ROTATABLE,
    TorsionTree,
    embed_3d,
    minimize,
    read_ligands,
    rotatable_bonds,
    rotation_reason,
    torsion_tree,
)

SEED = DEFAULT_SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mol_from(smiles: str, *, embed: bool = False, add_hs: bool = True):
    """A molecule from SMILES, optionally with an ETKDGv3 conformer."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if embed:
        mol = Chem.AddHs(mol) if add_hs else mol
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        assert AllChem.EmbedMolecule(mol, params) == 0
    return mol


def coords_of(mol) -> np.ndarray:
    conf = mol.GetConformer()
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ]
    )


def write_sdf(path: Path, records, *, coords3d: bool) -> Path:
    """``records`` is a list of ``(smiles, title)``."""
    writer = Chem.SDWriter(str(path))
    for smiles, title in records:
        mol = mol_from(smiles, embed=False)
        mol.SetProp("_Name", title)
        if coords3d:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = SEED
            assert AllChem.EmbedMolecule(mol, params) == 0
            mol.SetProp("_Name", title)
        else:
            AllChem.Compute2DCoords(mol)
        writer.write(mol)
    writer.close()
    return path


BENZENE_MOL2 = """@<TRIPOS>MOLECULE
benzene
 6 6 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 C1        0.0000    1.3960    0.0000 C.ar    1  BENZENE   0.0000
      2 C2        1.2090    0.6980    0.0000 C.ar    1  BENZENE   0.0000
      3 C3        1.2090   -0.6980    0.0000 C.ar    1  BENZENE   0.0000
      4 C4        0.0000   -1.3960    0.0000 C.ar    1  BENZENE   0.0000
      5 C5       -1.2090   -0.6980    0.0000 C.ar    1  BENZENE   0.0000
      6 C6       -1.2090    0.6980    0.0000 C.ar    1  BENZENE   0.0000
@<TRIPOS>BOND
     1    1    2 ar
     2    2    3 ar
     3    3    4 ar
     4    4    5 ar
     5    5    6 ar
     6    6    1 ar
"""


# ---------------------------------------------------------------------------
# Reading ligands: formats and batches
# ---------------------------------------------------------------------------


def test_read_bare_smiles_embeds_a_3d_conformer():
    mols = read_ligands("CCO")
    assert len(mols) == 1
    mol = mols[0]
    assert mol.GetProp("_Name") == "ligand"
    assert mol.GetNumConformers() == 1
    assert mol.GetConformer().Is3D()
    # Embedding adds hydrogens, so the geometry has real C-H bonds.
    assert any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())


def test_read_multi_line_smiles_text():
    mols = read_ligands("CCO ethanol\nCCC propane\n# a comment\n\n")
    assert [m.GetProp("_Name") for m in mols] == ["ethanol", "propane"]
    assert all(m.GetConformer().Is3D() for m in mols)


def test_read_multi_line_smi_file(tmp_path: Path):
    path = tmp_path / "library.smi"
    path.write_text("CCO ethanol\nCCC propane\nc1ccccc1 benzene\n", encoding="utf-8")
    mols = read_ligands(path)
    assert [m.GetProp("_Name") for m in mols] == ["ethanol", "propane", "benzene"]
    assert all(m.GetNumConformers() == 1 for m in mols)


def test_read_batch_sdf_returns_every_record(tmp_path: Path):
    path = write_sdf(
        tmp_path / "batch.sdf",
        [("CCO", "ethanol"), ("c1ccccc1", "benzene"), ("CC(=O)O", "acetic")],
        coords3d=False,
    )
    mols = read_ligands(path)
    assert [m.GetProp("_Name") for m in mols] == ["ethanol", "benzene", "acetic"]
    # A 2-D depiction must arrive as a 3-D conformer.
    assert all(m.GetConformer().Is3D() for m in mols)


def test_read_sdf_keeps_the_input_3d_geometry(tmp_path: Path):
    path = write_sdf(tmp_path / "three_d.sdf", [("CCO", "ethanol")], coords3d=True)
    reference = Chem.SDMolSupplier(str(path), removeHs=False)[0]
    mol = read_ligands(path)[0]
    assert mol.GetNumAtoms() == reference.GetNumAtoms()
    # SDWriter rounds to four decimals; anything larger would mean a re-embed.
    assert np.allclose(coords_of(mol), coords_of(reference), atol=1e-3)


def test_read_sdf_skips_a_broken_record_with_a_warning(tmp_path: Path):
    path = write_sdf(
        tmp_path / "partial.sdf", [("CCO", "good"), ("CCC", "also_good")], coords3d=False
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("this is not a molfile\nnor is this\n$$$$\n")
    with pytest.warns(UserWarning, match="problem"):
        mols = read_ligands(path)
    assert [m.GetProp("_Name") for m in mols] == ["good", "also_good"]


def test_read_mol2_embeds_a_flat_record(tmp_path: Path):
    path = tmp_path / "benzene.mol2"
    path.write_text(BENZENE_MOL2, encoding="utf-8")
    mols = read_ligands(path)
    assert len(mols) == 1
    mol = mols[0]
    assert mol.GetNumAtoms() == 6
    assert all(a.GetIsAromatic() for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    # RDKit's MOL2 reader has no dimensionality field and always claims 3-D, so
    # the flat depiction is detected from the coordinates.  (`NoImplicit` is set
    # on MOL2 atoms, so no hydrogens are invented here.)
    assert mol.GetConformer().Is3D()
    assert mol.GetProp("odock_embed_method") == "ETKDGv3"
    spread = coords_of(mol).max(axis=0) - coords_of(mol).min(axis=0)
    assert spread.min() > 0.0


def test_read_mol2_batch(tmp_path: Path):
    path = tmp_path / "two.mol2"
    path.write_text(BENZENE_MOL2 + BENZENE_MOL2, encoding="utf-8")
    mols = read_ligands(path, name="lig")
    assert len(mols) == 2
    assert [m.GetProp("_Name") for m in mols] == ["benzene", "benzene"]


def test_read_mol_file(tmp_path: Path):
    mol = mol_from("CCO")
    AllChem.Compute2DCoords(mol)
    path = tmp_path / "ethanol.mol"
    Chem.MolToMolFile(mol, str(path))
    read = read_ligands(path)
    assert len(read) == 1
    assert read[0].GetNumConformers() == 1
    assert read[0].GetConformer().Is3D()


def test_read_pdb_file(tmp_path: Path):
    mol = mol_from("CCO", embed=True)
    path = tmp_path / "ethanol.pdb"
    Chem.MolToPDBFile(mol, str(path))
    read = read_ligands(path)
    assert len(read) == 1
    assert sum(1 for a in read[0].GetAtoms() if a.GetAtomicNum() > 1) == 3


def test_read_pdbqt_reads_every_pose(tmp_path: Path):
    import odock

    mol = Chem.AddHs(mol_from("CCO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    assert AllChem.EmbedMolecule(mol, params) == 0
    text = odock.write_ligand_pdbqt(mol, name="ethanol")
    path = tmp_path / "poses.pdbqt"
    path.write_text(
        "MODEL 1\n" + text + "ENDMDL\nMODEL 2\n" + text + "ENDMDL\n", encoding="utf-8"
    )
    mols = read_ligands(path)
    assert len(mols) == 2
    heavy = [sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1) for m in mols]
    assert heavy == [3, 3]
    assert all(m.GetNumConformers() == 1 for m in mols)


def test_read_accepts_an_iterable_of_mixed_sources(tmp_path: Path):
    path = write_sdf(tmp_path / "one.sdf", [("CCO", "from_file")], coords3d=False)
    mols = read_ligands([Chem.MolFromSmiles("CCC"), "CO methanol", path])
    assert len(mols) == 3
    assert [m.GetProp("_Name") for m in mols] == ["ligand_1", "methanol", "from_file"]


def test_read_rejects_garbage():
    with pytest.raises(ValueError):
        read_ligands("this is definitely not a smiles")
    with pytest.raises(ValueError):
        read_ligands("CCO", fmt="xyz")
    with pytest.raises(FileNotFoundError):
        read_ligands("/no/such/directory/ligand.sdf", fmt="sdf")


def test_read_empty_smiles_file_raises(tmp_path: Path):
    path = tmp_path / "empty.smi"
    path.write_text("\n# only a comment\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_ligands(path)


def test_read_skips_an_unparsable_smiles_line_with_a_warning(tmp_path: Path):
    path = tmp_path / "mixed.smi"
    path.write_text("CCO ethanol\nnot-a-smiles broken\nCCC propane\n", encoding="utf-8")
    with pytest.warns(UserWarning):
        mols = read_ligands(path)
    assert [m.GetProp("_Name") for m in mols] == ["ethanol", "propane"]


# ---------------------------------------------------------------------------
# 3-D embedding
# ---------------------------------------------------------------------------


def test_embed_3d_from_a_2d_sketch():
    mol = mol_from("CC(=O)Nc1ccccc1")
    AllChem.Compute2DCoords(mol)
    assert not mol.GetConformer().Is3D()
    embedded = embed_3d(mol, seed=SEED)
    assert embedded.GetConformer().Is3D()
    assert any(a.GetAtomicNum() == 1 for a in embedded.GetAtoms())
    spread = coords_of(embedded).max(axis=0) - coords_of(embedded).min(axis=0)
    assert spread.min() > 0.0


def test_embed_3d_keeps_an_existing_conformer_unless_forced():
    mol = mol_from("CCOCC", embed=True)
    before = coords_of(mol)
    kept = embed_3d(mol, seed=SEED)
    assert np.allclose(coords_of(kept), before)
    assert not kept.HasProp("odock_embed_method")  # nothing was regenerated
    forced = embed_3d(mol, seed=SEED + 1, force=True)
    assert forced.GetProp("odock_embed_seed") == str(SEED + 1)
    assert not np.allclose(coords_of(forced), before)


def test_embed_3d_is_reproducible_with_the_seed():
    mol = mol_from("CCOCC")
    first = embed_3d(mol, seed=SEED)
    second = embed_3d(mol, seed=SEED)
    assert np.allclose(coords_of(first), coords_of(second))


def test_embed_3d_rejects_an_empty_molecule():
    with pytest.raises(RuntimeError):
        embed_3d(Chem.Mol())


# ---------------------------------------------------------------------------
# Force-field minimisation
# ---------------------------------------------------------------------------


def strained(smiles: str = "CCOCC") -> object:
    """An embedded conformer pushed off its minimum by a seeded jitter."""
    mol = mol_from(smiles, embed=True)
    rng = np.random.default_rng(7)
    conf = mol.GetConformer()
    for index in range(mol.GetNumAtoms()):
        position = conf.GetAtomPosition(index)
        conf.SetAtomPosition(
            index,
            (
                position.x + float(rng.normal(0.0, 0.2)),
                position.y + float(rng.normal(0.0, 0.2)),
                position.z + float(rng.normal(0.0, 0.2)),
            ),
        )
    return mol


@pytest.mark.parametrize("force_field", FORCE_FIELDS)
def test_minimize_lowers_the_energy_for_every_force_field(force_field: str):
    mol = strained()
    before_coords = coords_of(mol)
    minimised = minimize(mol, force_field=force_field, steps=500)

    energy_before = float(minimised.GetProp("odock_energy_before"))
    energy_after = float(minimised.GetProp("odock_energy_after"))
    assert math.isfinite(energy_before) and math.isfinite(energy_after)
    assert energy_after <= energy_before + 1e-9
    assert energy_after < energy_before  # the jittered input had room to fall
    assert minimised.GetProp("odock_force_field") == force_field
    assert minimised.GetProp("odock_minimize_steps") == "500"
    assert minimised.GetProp("odock_minimize_converged") in ("0", "1")
    # The geometry really moved.
    assert np.abs(coords_of(minimised) - before_coords).max() > 1e-3


def test_minimize_clamps_the_step_count():
    assert minimize(mol_from("CCC", embed=True), steps=5).GetProp(
        "odock_minimize_steps"
    ) == str(MIN_STEPS)
    assert minimize(mol_from("CCC", embed=True), steps=10**6).GetProp(
        "odock_minimize_steps"
    ) == str(MAX_STEPS)
    for bad in ("500", True, None):
        with pytest.raises(ValueError):
            minimize(mol_from("CCC", embed=True), steps=bad)  # type: ignore[arg-type]


def test_minimize_rejects_an_unknown_force_field():
    with pytest.raises(ValueError, match="unknown force field"):
        minimize(mol_from("CCC", embed=True), force_field="AMBER")


def test_minimize_falls_back_when_mmff_cannot_type_the_molecule():
    # Boric acid: MMFF94 has no boron parameters, UFF does.
    mol = mol_from("B(O)(O)O")
    minimised = minimize(mol, force_field="MMFF94")
    assert minimised.GetProp("odock_force_field") == "UFF"
    note = minimised.GetProp("odock_minimize_note")
    assert "MMFF94" in note and "fell back to UFF" in note
    assert math.isfinite(float(minimised.GetProp("odock_energy_after")))


def test_minimize_never_raises_when_no_force_field_can_type_the_molecule():
    minimised = minimize(mol_from("*CC"))
    assert minimised.GetProp("odock_energy_before") == "nan"
    assert minimised.GetProp("odock_energy_after") == "nan"
    note = minimised.GetProp("odock_minimize_note")
    for force_field in FORCE_FIELDS:
        assert force_field in note


def test_minimize_leaves_the_input_molecule_alone():
    mol = mol_from("CCO")
    assert mol.GetNumConformers() == 0
    minimised = minimize(mol)
    assert mol.GetNumConformers() == 0
    assert minimised.GetNumConformers() == 1
    assert minimised is not mol


def test_minimize_embeds_a_2d_input_first():
    mol = mol_from("CCO")
    AllChem.Compute2DCoords(mol)
    minimised = minimize(mol)
    assert minimised.GetConformer().Is3D()
    assert "ETKDGv3" in minimised.GetProp("odock_minimize_note")


# ---------------------------------------------------------------------------
# Rotatable-bond perception: one case per exclusion rule
# ---------------------------------------------------------------------------

#: ``(smiles, atom pair, expected reason)``.
ROTATION_CASES = [
    # amide C-N: resonance gives the bond double-bond character
    ("CC(=O)NC", (1, 3), RIGID_AMIDE),
    ("CC(=O)Nc1ccccc1", (1, 3), RIGID_AMIDE),
    # alkyne: not a single bond
    ("CCC#CC", (2, 3), RIGID_BOND_ORDER),
    ("CC#Cc1ccccc1", (1, 2), RIGID_BOND_ORDER),
    # a double bond in a conjugated ring
    ("C1C=CC=C1", (1, 2), RIGID_BOND_ORDER),
    # aromatic ring bonds
    ("c1ccccc1", (0, 1), RIGID_AROMATIC),
    ("c1ccncc1", (0, 1), RIGID_AROMATIC),
    # a non-aromatic ring bond
    ("C1CCCCC1", (0, 1), RIGID_RING),
    ("C1CCNCC1", (1, 2), RIGID_RING),
    # terminal methyl / -OH / -NH2 / -CF3 / halide
    ("CCCC", (0, 1), RIGID_TERMINAL),
    ("CCO", (1, 2), RIGID_TERMINAL),
    ("CCN", (1, 2), RIGID_TERMINAL),
    ("CC(F)(F)F", (0, 1), RIGID_TERMINAL),
    ("CCCl", (1, 2), RIGID_TERMINAL),
    # tert-butyl and symmetric quaternary centres
    ("CC(C)(C)CCC", (1, 4), RIGID_SYMMETRIC),
    ("C[N+](C)(C)CC", (1, 4), RIGID_SYMMETRIC),
    # ... but two identical substituents are not enough
    ("CC(C)CC", (1, 3), ROTATABLE),
    ("CC(C)CCC", (1, 3), ROTATABLE),
    # genuine torsions
    ("CCCC", (1, 2), ROTATABLE),
    ("CC(=O)Nc1ccccc1", (3, 4), ROTATABLE),
    ("c1ccccc1-c1ccccc1", (5, 6), ROTATABLE),
    ("CCOCC", (1, 2), ROTATABLE),
]


@pytest.mark.parametrize("smiles,pair,expected", ROTATION_CASES)
def test_rotation_reason(smiles: str, pair, expected: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    assert rotation_reason(mol, pair) == expected, (
        f"{smiles} {pair}: expected {expected} ({REASON_NOTES[expected]}), "
        f"got {rotation_reason(mol, pair)}"
    )


def test_every_reason_has_an_explanation():
    assert set(REASON_NOTES) == {
        ROTATABLE,
        RIGID_HYDROGEN,
        RIGID_BOND_ORDER,
        RIGID_AROMATIC,
        RIGID_RING,
        RIGID_AMIDE,
        RIGID_TERMINAL,
        RIGID_SYMMETRIC,
        RIGID_LOCKED,
    }
    assert all(text for text in REASON_NOTES.values())


def test_amide_is_locked_even_without_implicit_hydrogens():
    """A MOL2 file marks every atom ``NoImplicit``; the amide rule must hold."""
    mol = Chem.MolFromSmiles("CC(=O)NC")
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    assert rotation_reason(mol, (1, 3)) == RIGID_AMIDE
    assert rotatable_bonds(mol) == []


def test_rotation_reason_accepts_a_bond_index():
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert rotation_reason(mol, 0) == RIGID_AROMATIC
    with pytest.raises(IndexError):
        rotation_reason(mol, 999)
    with pytest.raises(ValueError):
        rotation_reason(mol, (0, 3))


def test_hydrogens_are_never_torsions():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    h_bonds = [
        b
        for b in mol.GetBonds()
        if b.GetBeginAtom().GetAtomicNum() == 1 or b.GetEndAtom().GetAtomicNum() == 1
    ]
    assert h_bonds
    for bond in h_bonds:
        assert (
            rotation_reason(mol, (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            == RIGID_HYDROGEN
        )
    # The only heavy-atom bond left is the terminal C-O, so nothing turns.
    assert rotatable_bonds(mol) == []


def test_rotatable_bonds_are_normalised_atom_pairs():
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    pairs = rotatable_bonds(mol)
    assert pairs == [(3, 4)]
    assert all(i < j for i, j in pairs)
    for i, j in pairs:
        assert mol.GetBondBetweenAtoms(i, j) is not None


def test_locked_forces_a_bond_rigid():
    mol = Chem.MolFromSmiles("CCCC")
    assert rotatable_bonds(mol) == [(1, 2)]
    assert rotatable_bonds(mol, locked=[(1, 2)]) == []
    assert rotatable_bonds(mol, locked=[(2, 1)]) == []  # order does not matter
    assert rotation_reason(mol, (1, 2), locked=[(1, 2)]) == RIGID_LOCKED
    # Locking one bond leaves the others alone.
    hexane = Chem.MolFromSmiles("CCCCCC")
    assert rotatable_bonds(hexane, locked=[(1, 2)]) == [(2, 3), (3, 4)]


def test_locked_ignores_pairs_that_are_not_bonds():
    mol = Chem.MolFromSmiles("CCCC")
    assert rotatable_bonds(mol, locked=[(0, 3), (99, 100), (1, 1), ("x", "y")]) == [(1, 2)]


def test_rotatable_bonds_of_the_reference_ligand(complex_parts):
    """Benzamidine from 3PTB: perception must not crash on a PDB-derived ligand.

    The ligand comes straight out of the PDB, so the exact torsion count is not
    the point — what matters is that every reported bond really is an acyclic,
    non-aromatic single bond and that the kinematic tree stays consistent with
    the bond list.
    """
    _, ligand = complex_parts
    pairs = rotatable_bonds(ligand)
    for i, j in pairs:
        bond = ligand.GetBondBetweenAtoms(i, j)
        assert bond is not None
        assert bond.GetBondType() == Chem.BondType.SINGLE
        assert not bond.GetIsAromatic()
        assert not bond.IsInRing()
    tree = torsion_tree(ligand)
    assert tree.torsdof == tree.num_torsions
    assert tree.num_torsions <= len(pairs)
    assert sorted(tree.atoms()) == list(range(ligand.GetNumAtoms()))


# ---------------------------------------------------------------------------
# Kinematic torsion tree
# ---------------------------------------------------------------------------


def test_torsion_tree_of_a_chain():
    tree = torsion_tree(Chem.MolFromSmiles("CCCC"))
    assert isinstance(tree, TorsionTree)
    assert tree.num_torsions == 1
    assert tree.torsdof == 1
    assert tree.torsions == [(1, 2)]
    assert sorted(tree.root_atoms) == [0, 1]
    assert [attach for _, attach, _ in tree.children] == [2]


def test_torsion_tree_root_is_the_largest_rigid_cluster():
    # Ethylbenzene: the ring (6 atoms) beats the ethyl fragment (2 atoms).
    mol = Chem.MolFromSmiles("CCc1ccccc1")
    tree = torsion_tree(mol)
    assert len(tree.root_atoms) == 6
    assert tree.num_torsions == 1
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    assert set(tree.root_atoms) <= ring


def test_torsion_tree_covers_every_atom_exactly_once():
    # Diethyl ether puts a single bridging oxygen between two torsions.
    mol = Chem.MolFromSmiles("CCOCC")
    tree = torsion_tree(mol)
    assert tree.num_torsions == 2
    seen = []
    for node in tree.iter_nodes():
        seen.extend(node.root_atoms)
        seen.extend(attach for _, attach, _ in node.children)
    assert sorted(seen) == list(range(mol.GetNumAtoms()))
    assert len(seen) == len(set(seen))


def test_torsion_tree_of_a_rigid_molecule():
    mol = Chem.MolFromSmiles("c1ccccc1")
    tree = torsion_tree(mol)
    assert tree.num_torsions == 0
    assert sorted(tree.root_atoms) == list(range(mol.GetNumAtoms()))
    assert tree.atoms() == tuple(range(mol.GetNumAtoms()))


def test_torsion_tree_respects_locked():
    mol = Chem.MolFromSmiles("CCc1ccccc1")
    tree = torsion_tree(mol, locked=[tuple(rotatable_bonds(mol)[0])])
    assert tree.num_torsions == 0
    assert sorted(tree.atoms()) == list(range(mol.GetNumAtoms()))


def test_torsion_tree_is_json_serialisable():
    tree = torsion_tree(Chem.MolFromSmiles("CCOCC"))
    payload = json.dumps(tree.as_dict())
    assert "num_torsions" in payload
    assert tree.as_dict()["num_torsions"] == 2


# ---------------------------------------------------------------------------
# Filters: Lipinski
# ---------------------------------------------------------------------------


def test_lipinski_passes_a_drug_like_molecule():
    result = filters.lipinski(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin
    assert result.name == "Lipinski"
    assert result.passed
    assert result.violations == []
    assert result.properties["MW"] == pytest.approx(180.16, abs=0.1)
    assert result.properties["LogP"] == pytest.approx(1.31, abs=0.05)
    assert result.properties["HBD"] == 1
    assert result.properties["HBA"] == 3
    assert result.properties["n_violations"] == 0


@pytest.mark.parametrize(
    "smiles,rule",
    [
        ("C" * 40, "MW "),  # n-tetracontane, MW 563
        ("C" * 18, "LogP "),  # octadecane, cLogP 6.5
        ("OCC(O)C(O)C(O)C(O)CO", "HBD "),  # sorbitol, 6 donors
        ("CO" * 11 + "C", "HBA "),  # a polyether with 11 acceptors
    ],
)
def test_lipinski_flags_each_threshold(smiles: str, rule: str):
    result = filters.lipinski(Chem.MolFromSmiles(smiles))
    assert not result.passed
    assert any(v.startswith(rule) for v in result.violations), result.violations
    assert result.properties["n_violations"] >= 1


def test_lipinski_reports_every_breach_at_once():
    result = filters.lipinski(Chem.MolFromSmiles("C" * 40))
    assert not result.passed
    assert result.properties["n_violations"] == len(result.violations) == 2


# ---------------------------------------------------------------------------
# Filters: Veber
# ---------------------------------------------------------------------------


def test_veber_passes_a_drug_like_molecule():
    result = filters.veber(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))  # paracetamol
    assert result.name == "Veber"
    assert result.passed
    assert result.properties["RotB"] == 1
    assert result.properties["tPSA"] == pytest.approx(49.3, abs=0.2)


def test_veber_flags_too_many_rotatable_bonds():
    result = filters.veber(Chem.MolFromSmiles("C" * 16))
    assert not result.passed
    assert result.properties["RotB"] == 13
    assert result.violations and result.violations[0].startswith("RotB ")


def test_veber_flags_too_much_polar_surface():
    # Sucrose: tPSA 189.5, only 5 rotatable bonds.
    result = filters.veber(Chem.MolFromSmiles("OCC1OC(CO)(OC2OC(CO)C(O)C(O)C2O)C(O)C1O"))
    assert not result.passed
    assert result.properties["tPSA"] > filters.VEBER_MAX_TPSA
    assert result.properties["RotB"] <= filters.VEBER_MAX_ROTATABLE
    assert len(result.violations) == 1
    assert result.violations[0].startswith("tPSA ")


def test_veber_counts_amides_as_rigid():
    """The filter must use this package's torsion perception, not RDKit's."""
    result = filters.veber(Chem.MolFromSmiles("CC(=O)NC"))
    assert result.properties["RotB"] == 0


# ---------------------------------------------------------------------------
# Filters: PAINS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Nc1ccc(O)cc1", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"],
)
def test_pains_passes_clean_drugs(smiles: str):
    result = filters.pains(Chem.MolFromSmiles(smiles))
    assert result.passed
    assert result.violations == []
    assert result.properties["PAINS_hits"] == 0.0
    assert result.properties["PAINS_patterns_used"] >= 25


def test_pains_flags_a_quinone():
    result = filters.pains(Chem.MolFromSmiles("O=C1C=CC(=O)C=C1"))
    assert not result.passed
    assert result.properties["PAINS_hits"] >= 1
    assert any("quinone" in v for v in result.violations)


def test_pains_fallback_dictionary_is_a_real_pattern_set():
    assert len(filters.PAINS_SMARTS) >= 25
    for name, smarts in filters.PAINS_SMARTS.items():
        assert name and smarts
        assert Chem.MolFromSmarts(smarts) is not None, name
    hits = filters.match_pains_smarts(Chem.MolFromSmiles("O=C1C=CC(=O)C=C1"))
    assert "para_quinone" in hits
    assert filters.match_pains_smarts(Chem.MolFromSmiles("Oc1ccccc1O")) == ["catechol"]


@pytest.mark.parametrize(
    "smiles,name",
    [
        ("O=C1C=CC(=O)C=C1", "para_quinone"),
        ("Oc1ccccc1O", "catechol"),
        ("O=C1CSC(=S)N1", "rhodanine"),
        ("C1CO1", "epoxide"),
        ("O=C1C=CC(=O)N1", "maleimide"),
        ("CN=C=S", "isothiocyanate"),
        ("CC(=O)NO", "hydroxamic_acid"),
        ("Nc1ccc(cc1)-c1ccc(N)cc1", "benzidine"),
    ],
)
def test_pains_fallback_catches_known_frequent_hitters(smiles: str, name: str):
    assert name in filters.match_pains_smarts(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    "smiles",
    [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)Nc1ccc(O)cc1",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "OCC1OC(O)C(O)C(O)C1O",
        "NC(=N)c1ccccc1",
        "CN(C)C(=N)NC(=N)N",
    ],
)
def test_pains_fallback_leaves_clean_drugs_alone(smiles: str):
    assert filters.match_pains_smarts(Chem.MolFromSmiles(smiles)) == []


def test_pains_fallback_path_agrees_on_the_verdict():
    quinone = Chem.MolFromSmiles("O=C1C=CC(=O)C=C1")
    forced = filters.pains(quinone, use_catalog=False)
    assert forced.name == "PAINS"
    assert not forced.passed
    assert forced.properties["PAINS_patterns_used"] == len(filters.PAINS_SMARTS)
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    assert filters.pains(aspirin, use_catalog=False).passed


def test_pains_catalog_flag_matches_the_build():
    if filters.HAVE_FILTER_CATALOG:
        result = filters.pains(Chem.MolFromSmiles("O=C1C=CC(=O)C=C1"), use_catalog=True)
        assert not result.passed
        assert result.properties["PAINS_patterns_used"] >= 100
    else:  # pragma: no cover - depends on the RDKit build
        with pytest.raises(RuntimeError):
            filters.pains(Chem.MolFromSmiles("CCO"), use_catalog=True)


# ---------------------------------------------------------------------------
# Filters: the combined verdict
# ---------------------------------------------------------------------------


def test_drug_like_shape_and_pass():
    verdict = filters.drug_like(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert verdict["passed"] is True
    assert verdict["violations"] == []
    assert [r.name for r in verdict["results"]] == ["Lipinski", "Veber", "PAINS"]
    assert all(isinstance(r, filters.FilterResult) for r in verdict["results"])
    for key in ("MW", "LogP", "HBD", "HBA", "RotB", "tPSA", "PAINS_hits"):
        assert key in verdict["properties"]
    assert verdict["properties"]["PAINS_patterns_used"] >= 25
    assert verdict["properties"]["Lipinski_violations"] == 0
    assert verdict["properties"]["Veber_violations"] == 0


def test_drug_like_fails_and_prefixes_every_violation():
    verdict = filters.drug_like(Chem.MolFromSmiles("C" * 40))
    assert verdict["passed"] is False
    assert verdict["violations"]
    assert all(": " in v for v in verdict["violations"])
    assert any(v.startswith("Lipinski: ") for v in verdict["violations"])
    assert any(v.startswith("Veber: ") for v in verdict["violations"])


def test_drug_like_reports_a_pains_failure():
    verdict = filters.drug_like(Chem.MolFromSmiles("O=C1C=CC(=O)C=C1"))
    assert verdict["passed"] is False
    assert verdict["properties"]["PAINS_hits"] >= 1
    assert any(v.startswith("PAINS: ") for v in verdict["violations"])


def test_filter_library_keeps_the_order_and_names():
    library = read_ligands(
        "CC(=O)Oc1ccccc1C(=O)O aspirin\n" + "C" * 40 + " wax\n" + "O=C1C=CC(=O)C=C1 quinone",
        embed=False,
    )
    rows = filters.filter_library(library)
    assert [row["index"] for row in rows] == [0, 1, 2]
    assert [row["name"] for row in rows] == ["aspirin", "wax", "quinone"]
    assert [row["passed"] for row in rows] == [True, False, False]
    assert all("properties" in row and "violations" in row for row in rows)


def test_filter_library_names_an_unnamed_molecule():
    rows = filters.filter_library([Chem.MolFromSmiles("CCO")])
    assert rows[0]["name"] == "ligand_1"


def test_filters_need_no_conformer():
    mol = Chem.MolFromSmiles("CCO")
    assert mol.GetNumConformers() == 0
    assert filters.lipinski(mol).passed
    assert filters.veber(mol).passed
    assert filters.pains(mol).passed
    assert filters.drug_like(mol)["passed"] is True


def test_filter_result_summary_and_dict():
    result = filters.lipinski(Chem.MolFromSmiles("CCO"))
    assert "pass" in result.summary()
    payload = result.as_dict()
    assert payload["name"] == "Lipinski"
    assert payload["passed"] is True
    failed = filters.lipinski(Chem.MolFromSmiles("C" * 40))
    assert "fail" in failed.summary()
    json.dumps(failed.as_dict())


# ---------------------------------------------------------------------------
# Reading a library for the ligand-chemistry commands
# ---------------------------------------------------------------------------
#
# `odock similar/diverse/scaffolds/rgroups` fingerprint a whole library, so they
# read it with `embed=False`: a 100 000-compound triage must not generate 100 000
# three-dimensional structures.  These tests pin the two properties those
# commands rely on — no conformers are made, and the file order and titles
# survive — plus the SMILES round trip the `--diverse -o subset.smi` writer uses.


def test_read_ligands_without_embedding_leaves_a_flat_library_flat(tmp_path: Path):
    path = tmp_path / "library.smi"
    path.write_text("CCO ethanol\nc1ccccc1 benzene\n", encoding="utf-8")
    flat = read_ligands(path, embed=False)
    assert [mol.GetNumConformers() for mol in flat] == [0, 0]
    # The default still embeds, which is what the docking path needs.
    embedded = read_ligands(path)
    assert all(mol.GetConformer().Is3D() for mol in embedded)
    assert [mol.GetProp("_Name") for mol in embedded] == ["ethanol", "benzene"]


def test_read_ligands_keeps_library_order_and_titles_with_spaces(tmp_path: Path):
    """A name is everything after the first field, spaces included — the format
    the chemistry commands write a diverse subset in."""
    path = tmp_path / "library.smi"
    path.write_text(
        "CCO ethanol\nc1ccccc1 benzene, pure\n\n# a comment\nCCC propane\n",
        encoding="utf-8",
    )
    mols = read_ligands(path, embed=False)
    assert [mol.GetProp("_Name") for mol in mols] == [
        "ethanol",
        "benzene, pure",
        "propane",
    ]


def test_a_smiles_library_round_trips_through_the_writer_convention(tmp_path: Path):
    """`SMILES name` lines read back with the same molecules and titles, which is
    what makes a written subset a usable library again."""
    source = read_ligands("N=C(N)c1ccccc1 benzamidine\nCn1cnc2c1c(=O)n(C)c(=O)n2C caffeine", embed=False)
    text = "\n".join(
        f"{Chem.MolToSmiles(mol)} {mol.GetProp('_Name')}" for mol in source
    )
    path = tmp_path / "subset.smi"
    path.write_text(text + "\n", encoding="utf-8")
    again = read_ligands(path, embed=False)
    assert [mol.GetProp("_Name") for mol in again] == ["benzamidine", "caffeine"]
    assert [Chem.MolToSmiles(mol) for mol in again] == [
        Chem.MolToSmiles(mol) for mol in source
    ]

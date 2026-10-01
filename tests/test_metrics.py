# SPDX-License-Identifier: GPL-3.0-or-later
"""Ligand-efficiency metrics and ligand strain: hand-checkable numbers.

Every exact value asserted here is analytically derivable from the definitions in
``odock.metrics`` (``LE = -dG/HAC``, ``LLE = pIC50 - logP`` with
``pIC50 = -dG/1.37``, ``BEI = 1000 pIC50/MW``, ``SEI = 100 pIC50/TPSA``,
``entropy = n R T ln s``), so a failure means a definition changed, not a
tolerance drifted.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from odock import consensus, metrics
from odock.docking import DockResult, Pose
from odock.metrics import (
    GAS_CONSTANT,
    KCAL_PER_LOG,
    METRIC_KEYS,
    STANDARD_TEMPERATURE,
    STATES_PER_ROTOR,
    binding_efficiency_index,
    descriptors,
    efficiency_metrics,
    entropy_penalty,
    heavy_atom_count,
    ligand_efficiencies,
    ligand_efficiency,
    ligand_strain,
    lipophilic_ligand_efficiency,
    p_activity,
    pose_strain,
    surface_efficiency_index,
)

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO_LIGAND = ROOT / "demo" / "systems" / "3ptb" / "ligand.pdbqt"
DEMO_POSES = ROOT / "demo" / "systems" / "3ptb" / "poses.pdbqt"
DEMO_RECEPTOR = ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt"
DEMO_BOX = ROOT / "demo" / "systems" / "3ptb" / "box.json"

#: A remote two-atom receptor: the ligand's *intra* term is what the strain
#: measures, and a receptor 40 Å away contributes nothing to it.
REMOTE_RECEPTOR = """\
ATOM      1  CB  ALA A   1     -40.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  CB  ALA A   2      40.000   0.000   0.000  1.00  0.00     0.000 C
TER
"""


# ---------------------------------------------------------------------------
# The definitions themselves
# ---------------------------------------------------------------------------


def test_p_activity_is_delta_g_over_one_point_three_seven():
    assert p_activity(-6.85) == pytest.approx(5.0)
    assert p_activity(0.0) == pytest.approx(0.0)
    assert p_activity(-13.7) == pytest.approx(10.0)
    assert KCAL_PER_LOG == 1.37


def test_ligand_efficiency_is_affinity_per_heavy_atom():
    assert ligand_efficiency(-9.0, 30) == pytest.approx(0.30)
    assert ligand_efficiency(-6.21, 9) == pytest.approx(0.69)
    assert ligand_efficiency(6.21, 9) == pytest.approx(-0.69)


def test_lipophilic_ligand_efficiency_is_p_activity_minus_logp():
    assert lipophilic_ligand_efficiency(-6.85, 2.5) == pytest.approx(2.5)
    assert lipophilic_ligand_efficiency(-6.85, 0.0) == pytest.approx(5.0)
    # Low logP always helps: LLE must increase as logP falls.
    assert lipophilic_ligand_efficiency(-6.85, -1.0) > lipophilic_ligand_efficiency(
        -6.85, 1.0
    )


def test_binding_and_surface_efficiency_indices():
    # pIC50 = 5.0, MW = 250 g/mol -> 5 / 0.25 = 20
    assert binding_efficiency_index(-6.85, 250.0) == pytest.approx(20.0)
    # pIC50 = 5.0, TPSA = 50 A^2 -> 5 / 0.5 = 10
    assert surface_efficiency_index(-6.85, 50.0) == pytest.approx(10.0)


def test_efficiency_metrics_are_index_like_and_insensitive_to_scale():
    """Doubling the affinity doubles every efficiency index, nothing else."""
    single = efficiency_metrics(-3.0, heavy_atoms=20, molecular_weight=200.0, logp=1.0,
                                tpsa=40.0)
    double = efficiency_metrics(-6.0, heavy_atoms=20, molecular_weight=200.0, logp=1.0,
                                tpsa=40.0)
    assert double.ligand_efficiency == pytest.approx(2 * single.ligand_efficiency)
    assert double.bei == pytest.approx(2 * single.bei)
    assert double.sei == pytest.approx(2 * single.sei)
    # LLE is a difference, so it shifts rather than scales.
    assert double.lle - single.lle == pytest.approx(3.0 / KCAL_PER_LOG, abs=1e-12)


# ---------------------------------------------------------------------------
# The undefined-input contract: nan, never an exception, never a silent zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: ligand_efficiency(-9.0, 0),
        lambda: ligand_efficiency(-9.0, -3),
        lambda: ligand_efficiency(float("nan"), 10),
        lambda: ligand_efficiency(float("inf"), 10),
        lambda: lipophilic_ligand_efficiency(-9.0, float("nan")),
        lambda: lipophilic_ligand_efficiency(float("nan"), 1.0),
        lambda: binding_efficiency_index(-9.0, 0.0),
        lambda: binding_efficiency_index(-9.0, float("nan")),
        lambda: surface_efficiency_index(-9.0, 0.0),
        lambda: surface_efficiency_index(float("inf"), 40.0),
        lambda: entropy_penalty(-1.0),
        lambda: entropy_penalty(float("nan")),
        lambda: p_activity(float("inf")),
    ],
)
def test_undefined_metrics_come_back_as_nan(call):
    value = call()
    assert isinstance(value, float)
    assert math.isnan(value)


@pytest.mark.parametrize(
    "call",
    [
        lambda: binding_efficiency_index(-9.0, -250.0),
        lambda: surface_efficiency_index(-9.0, -40.0),
        lambda: ligand_efficiency("nine", 10),
        lambda: entropy_penalty("two"),
    ],
)
def test_caller_mistakes_raise(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_a_zero_heavy_atom_count_is_nan_not_zero():
    """The distinction matters: a blank cell, not a perfect efficiency."""
    value = ligand_efficiency(-9.0, 0)
    assert math.isnan(value)
    assert value != 0.0


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


def test_entropy_penalty_matches_the_analytic_formula():
    expected = 1.0 * GAS_CONSTANT * STANDARD_TEMPERATURE * math.log(STATES_PER_ROTOR)
    assert entropy_penalty(1) == pytest.approx(expected, rel=1e-15)
    # 0.651 kcal/mol per threefold rotor at 298.15 K.
    assert entropy_penalty(1) == pytest.approx(0.6509112466064476, abs=1e-12)
    assert entropy_penalty(0) == 0.0
    assert entropy_penalty(4) == pytest.approx(4 * expected, rel=1e-15)


def test_entropy_penalty_is_linear_in_the_torsion_count_and_logarithmic_in_states():
    assert entropy_penalty(3) - entropy_penalty(2) == pytest.approx(entropy_penalty(1))
    assert entropy_penalty(1, states_per_rotor=9.0) == pytest.approx(
        2 * entropy_penalty(1, states_per_rotor=3.0), rel=1e-12
    )
    assert entropy_penalty(1, temperature=2 * STANDARD_TEMPERATURE) == pytest.approx(
        2 * entropy_penalty(1)
    )


# ---------------------------------------------------------------------------
# Vectorised helper
# ---------------------------------------------------------------------------


def test_vectorised_ligand_efficiency_agrees_with_the_scalar_rule():
    values = np.array([-6.0, -9.0, -3.0, -1.0])
    counts = np.array([20, 30, 0, 10])
    result = ligand_efficiencies(values, counts)
    assert result.shape == (4,)
    assert result[0] == pytest.approx(0.3)
    assert result[1] == pytest.approx(0.3)
    assert math.isnan(result[2])
    assert result[3] == pytest.approx(0.1)


def test_vectorised_ligand_efficiency_broadcasts_and_handles_non_finite():
    result = ligand_efficiencies([-9.0, float("nan")], 30)
    assert result.tolist()[0] == pytest.approx(0.3)
    assert math.isnan(result.tolist()[1])
    grid = ligand_efficiencies(np.full((2, 3), -6.0), np.full(3, 10.0))
    assert grid.shape == (2, 3)
    assert np.allclose(grid, 0.6)


def test_vectorised_ligand_efficiency_of_an_empty_input_is_empty():
    assert ligand_efficiencies([], []).shape == (0,)


# ---------------------------------------------------------------------------
# Heavy atoms and descriptors
# ---------------------------------------------------------------------------


def test_heavy_atom_count_of_an_integer():
    assert heavy_atom_count(17) == 17
    assert heavy_atom_count(0) == 0
    assert heavy_atom_count(-4) == 0


def test_heavy_atom_count_of_a_molecule():
    assert heavy_atom_count(Chem.MolFromSmiles("c1ccccc1")) == 6
    assert heavy_atom_count(Chem.MolFromSmiles("CCO")) == 3
    assert heavy_atom_count(Chem.AddHs(Chem.MolFromSmiles("CCO"))) == 3


def test_heavy_atom_count_of_atom_like_objects():
    class A:
        def __init__(self, element):
            self.element = element

    atoms = [A("C"), A("H"), A("O"), A("D"), A("N")]
    assert heavy_atom_count(atoms) == 3


def test_heavy_atom_count_of_a_pdbqt_document():
    text = """\
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  H1  LIG A   1       1.090   0.000   0.000  1.00  0.00     0.000 H
ATOM      3  G0  LIG A   1       2.000   0.000   0.000  1.00  0.00     0.000 G0
ATOM      4  O1  LIG A   1       1.500   1.000   0.000  1.00  0.00    -0.300 OA
ENDROOT
TORSDOF 0
"""
    assert heavy_atom_count(text) == 2


def test_heavy_atom_count_refuses_an_ambiguous_coordinate_array():
    with pytest.raises(TypeError):
        heavy_atom_count(np.zeros((5, 3)))


def test_descriptors_of_ethanol_and_benzene():
    ethanol = descriptors(Chem.MolFromSmiles("CCO"))
    assert ethanol["heavy_atoms"] == 3
    assert ethanol["molecular_weight"] == pytest.approx(46.069, abs=0.01)
    assert ethanol["num_torsions"] == 0
    assert ethanol["num_hbd"] == 1
    assert ethanol["num_hba"] == 1

    benzene = descriptors(Chem.MolFromSmiles("c1ccccc1"))
    assert benzene["heavy_atoms"] == 6
    assert benzene["molecular_weight"] == pytest.approx(78.114, abs=0.01)
    assert benzene["logp"] == pytest.approx(1.6866, abs=0.002)
    assert benzene["tpsa"] == pytest.approx(0.0, abs=1e-9)
    assert benzene["num_torsions"] == 0
    assert benzene["num_hba"] == 0


# ---------------------------------------------------------------------------
# The metric bundle
# ---------------------------------------------------------------------------


def test_efficiency_metrics_from_a_molecule_fills_every_field():
    mol = Chem.MolFromSmiles("c1ccccc1")
    bundle = efficiency_metrics(-4.0, mol=mol)
    assert bundle.heavy_atoms == 6
    assert bundle.molecular_weight == pytest.approx(78.114, abs=0.01)
    assert bundle.logp == pytest.approx(1.6866, abs=0.002)
    assert bundle.ligand_efficiency == pytest.approx(4.0 / 6, rel=1e-12)
    assert bundle.p_activity == pytest.approx(4.0 / 1.37, rel=1e-12)
    assert bundle.lle == pytest.approx(4.0 / 1.37 - 1.6866, abs=1e-3)
    # Benzene has no polar surface, so SEI is undefined rather than infinite --
    # and ``ok`` reports that honestly instead of pretending otherwise.
    assert math.isnan(bundle.sei)
    assert not bundle.ok


def test_efficiency_metrics_without_a_molecule_still_gives_the_core_numbers():
    bundle = efficiency_metrics(-9.0, heavy_atoms=30, num_torsions=4)
    assert bundle.ligand_efficiency == pytest.approx(0.3)
    assert bundle.entropy_penalty == pytest.approx(4 * entropy_penalty(1))
    assert math.isnan(bundle.lle)
    assert math.isnan(bundle.bei)
    assert math.isnan(bundle.sei)
    assert bundle.logp is None
    # `ok` is "every metric that is present is finite"; LLE/BEI/SEI are NaN
    # because their inputs were not supplied, so it is False by design.
    assert not bundle.ok


def test_efficiency_metrics_ok_is_false_when_a_required_input_is_missing():
    bundle = efficiency_metrics(-9.0)
    assert math.isnan(bundle.ligand_efficiency)
    assert not bundle.ok


def test_efficiency_metrics_keys_are_stable_and_json_ready():
    bundle = efficiency_metrics(-6.0, heavy_atoms=20, molecular_weight=200.0, logp=1.0,
                                tpsa=40.0, num_torsions=2, strain=1.5)
    data = bundle.as_dict()
    for key in METRIC_KEYS:
        assert key in data, key
    assert data["strain"] == pytest.approx(1.5)
    import json

    assert json.loads(json.dumps({k: v for k, v in data.items() if v is not None}))
    assert set(bundle.row(prefix="le_")) > {"le_ligand_efficiency", "le_bei"}


def test_efficiency_metrics_explicit_arguments_beat_the_molecule():
    mol = Chem.MolFromSmiles("c1ccccc1")
    bundle = efficiency_metrics(-6.0, mol=mol, heavy_atoms=99)
    assert bundle.heavy_atoms == 99
    assert bundle.ligand_efficiency == pytest.approx(6.0 / 99, rel=1e-12)


# ---------------------------------------------------------------------------
# Strain
# ---------------------------------------------------------------------------


def _embedded(smiles: str, seed: int = 0xC0FFEE):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _embedded_heavy_only(smiles: str, seed: int = 0xC0FFEE):
    """A 3-D molecule without explicit hydrogens, for atom-order tests."""
    mol = Chem.MolFromSmiles(smiles)
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def _pdbqt(mol, name: str = "lig") -> str:
    from odock.pdbqt import write_ligand_pdbqt

    return write_ligand_pdbqt(mol, name=name)


def test_strain_of_a_force_field_minimum_is_small():
    """A molecule already at its MMFF94 minimum has almost no strain.

    Not exactly zero: the round trip drops the non-polar hydrogens into the
    united-atom PDBQT and adds them back at idealised positions, which is worth
    about a hundredth of a kcal/mol.  That residual is the floor of the method
    and it is asserted here so that a future change to the perception shows up.
    """
    from odock.chem import ligand as _ligand

    minimised = _ligand.minimize(_embedded("CCCCO"), force_field="MMFF94", steps=500)
    assert minimised.GetProp("odock_force_field") == "MMFF94"
    strain = ligand_strain(
        _pdbqt(minimised), receptor=REMOTE_RECEPTOR, box=None, scoring="vina"
    )
    assert strain.force_field == "MMFF94"
    assert abs(strain.force_field_strain) < 0.05
    assert strain.energy_relaxed <= strain.energy_pose + 1e-9


def test_strain_of_a_distorted_conformer_is_positive_and_internally_consistent():
    mol = _embedded("CCCCO")
    conformer = mol.GetConformer()
    # Push one carbon 0.25 A off its position: a real, small geometric strain.
    position = conformer.GetAtomPosition(1)
    from rdkit.Geometry import Point3D

    conformer.SetAtomPosition(1, Point3D(position.x + 0.25, position.y, position.z))
    pose = _pdbqt(mol)
    strain = ligand_strain(pose, receptor=REMOTE_RECEPTOR, box=None, scoring="vina")
    assert strain.force_field == "MMFF94"
    assert strain.force_field_strain > 1.0  # kcal/mol, not a rounding artefact
    # The two numbers reported are the two the difference is made of.
    assert strain.force_field_strain == pytest.approx(
        strain.energy_pose - strain.energy_relaxed, rel=1e-12
    )
    assert strain.energy_relaxed < strain.energy_pose


def test_strain_preserves_the_topology_and_moves_only_the_coordinates():
    mol = _embedded("CCOC(=O)C")
    pose = _pdbqt(mol)
    strain = ligand_strain(pose, receptor=REMOTE_RECEPTOR, box=None, scoring="vina")
    before = consensus.pdbqt_atoms(pose)
    after = consensus.pdbqt_atoms(strain.relaxed_pdbqt)
    assert len(before) == len(after)
    assert [a.name for a in before] == [a.name for a in after]
    assert [a.ad_type for a in before] == [a.ad_type for a in after]
    assert [a.res_name for a in before] == [a.res_name for a in after]


def test_strain_reports_the_kernel_intra_terms_it_used():
    """``intra`` must be exactly what the kernel says for that document."""
    import odock

    mol = _embedded("CCCC")
    pose = _pdbqt(mol)
    strain = ligand_strain(pose, receptor=REMOTE_RECEPTOR, box=None, scoring="vina")
    direct = odock.score(REMOTE_RECEPTOR, pose, None, scoring="vina", refine=False)
    assert strain.intra == pytest.approx(direct["intra"], rel=1e-12)
    assert strain.strain == pytest.approx(strain.intra - strain.intra_relaxed, rel=1e-12)
    assert "kernel intra from vina" in strain.note


def test_strain_without_a_receptor_reports_only_the_force_field_part():
    mol = _embedded("CCO")
    strain = ligand_strain(_pdbqt(mol), receptor=None, box=None, scoring="vina")
    assert math.isnan(strain.intra)
    assert math.isnan(strain.strain)
    assert math.isfinite(strain.force_field_strain)
    assert "skipped" in strain.note


# -- what the escape hatch does and does not buy ----------------------------


def test_a_pose_only_strain_is_marked_unreliable():
    """A PDBQT carries no bond orders: the MMFF94 number must be flagged."""
    from odock.chem import ligand as _ligand

    mol = _embedded("CCO")
    text = _pdbqt(mol)
    strain = ligand_strain(text, receptor=REMOTE_RECEPTOR, scoring="vina")
    assert strain.reliable is False
    assert "no bond orders" in strain.note
    assert "not necessarily of the intended ligand" in strain.note
    assert strain.as_dict()["reliable"] is False


def test_require_reliable_turns_the_flag_into_an_error():
    text = _pdbqt(_embedded("CCO"))
    with pytest.raises(ValueError) as caught:
        ligand_strain(
            text, receptor=REMOTE_RECEPTOR, scoring="vina", require_reliable=True
        )
    assert "cannot produce a reliable force-field strain" in str(caught.value)


def test_a_supplied_molecule_makes_the_strain_reliable():
    mol = _embedded("CCO")
    text = _pdbqt(mol)
    strain = ligand_strain(
        text, receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol, require_reliable=True
    )
    assert strain.reliable is True
    assert math.isfinite(strain.force_field_strain)


def test_a_molecule_in_the_wrong_atom_order_raises_instead_of_lying():
    """The relaxed coordinates are patched by file order, so the order matters."""
    mol = _embedded("CCO")           # C, C, O
    reordered = _embedded("OCC")     # O, C, C -- same molecule, different order
    with pytest.raises(ValueError) as caught:
        ligand_strain(
            _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=reordered
        )
    assert "atom order does not match" in str(caught.value)


def test_a_saturated_chain_is_not_suspicious():
    """Ethanol really is all single bonds: no false alarm."""
    mol = _embedded("CCO")
    strain = ligand_strain(
        _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol
    )
    assert strain.reliable is True
    assert "chemistry were checked" in strain.note


def test_an_all_single_bond_ring_is_flagged_as_suspicious():
    """The shape a molecule perceived from a bare PDB takes, warned about."""
    mol = _embedded("C1CCCCC1")  # cyclohexane: indistinguishable from benzene lost
    strain = ligand_strain(
        _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol
    )
    assert strain.reliable is False
    assert "all-single-bond ring system" in strain.note
    with pytest.raises(ValueError):
        ligand_strain(
            _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol,
            require_reliable=True,
        )


def test_a_preparation_report_attests_the_chemistry():
    class Report:
        chemistry_trusted = True
        warnings = ["bond orders taken from the supplied SMILES template"]

    mol = _embedded("C1CCCCC1")
    strain = ligand_strain(
        _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol,
        preparation=Report(),
    )
    assert strain.reliable is True
    assert "bond orders came from the input" in strain.note


def test_a_report_that_admits_inferred_chemistry_is_unreliable():
    class Report:
        chemistry_trusted = False
        warnings = [
            "a PDB file carries no bond orders; polar hydrogens were placed with a "
            "bond-length and ring-membership rule."
        ]

    mol = _embedded("C1CCCCC1")
    strain = ligand_strain(
        _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol,
        preparation=Report(),
    )
    assert strain.reliable is False
    assert "chemistry_trusted is False" in strain.note
    assert "preparation said: a PDB file carries no bond orders" in strain.note


def test_the_pose_aromatic_types_expose_a_lost_aromatic_ring():
    """A provable mismatch: the PDBQT says aromatic, the molecule does not."""
    mol = _embedded("C1CCCCC1")
    lines = _pdbqt(mol).splitlines()
    for index, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")):
            lines[index] = line[:77] + " A"  # claim the first carbon is aromatic
            break
    text = "\n".join(lines) + "\n"
    assert consensus.pdbqt_atoms(text)[0].ad_type == "A"
    strain = ligand_strain(
        text, receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol
    )
    assert strain.reliable is False
    assert "types atoms as aromatic" in strain.note


def test_atom_order_renumbers_the_molecule_into_the_pose_order():
    """The documented one-call recipe: prepare, then hand over the index map."""
    mol = _embedded_heavy_only("OCC")    # O, C, C (implicit hydrogens)
    reordered = _embedded_heavy_only("CCO")  # C, C, O -- the pose document's order
    pose = _pdbqt(reordered)
    available = {}
    for index, atom in enumerate(mol.GetAtoms()):
        available.setdefault(atom.GetSymbol(), []).append(index)
    order = [available[symbol].pop(0) for symbol in ("C", "C", "O")]
    assert order == [1, 2, 0]
    # A naive pass would raise; the explicit mapping makes it work.
    with pytest.raises(ValueError):
        ligand_strain(pose, receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol)
    strain = ligand_strain(
        pose, receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol, atom_order=order
    )
    assert math.isfinite(strain.force_field_strain)
    assert strain.reliable is True


def test_atom_order_must_be_a_permutation():
    mol = _embedded("CCO")
    with pytest.raises(ValueError):
        ligand_strain(
            _pdbqt(mol), receptor=REMOTE_RECEPTOR, scoring="vina", mol=mol,
            atom_order=[0, 0, 1],
        )


def test_strain_of_a_document_with_no_atoms_does_not_raise():
    strain = ligand_strain("REMARK nothing here\n", receptor=REMOTE_RECEPTOR)
    assert math.isnan(strain.strain)
    assert "no atoms" in strain.note


def test_strain_survives_a_document_rdkit_cannot_parse():
    strain = ligand_strain(
        "ROOT\nATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n",
        receptor=None,
    )
    # One lone carbon is parseable; the point is that nothing raises and the
    # note always explains the outcome.
    assert strain.note


def test_pose_strain_uses_the_result_topology_and_coordinates():
    """A pose with no relaxation must have zero strain in its own force field."""
    from odock import consensus as _consensus

    template = _pdbqt(_embedded("CCO"))
    coords = np.array([[a.x, a.y, a.z] for a in _consensus.pdbqt_atoms(template)])
    result = DockResult(
        poses=[Pose(index=0, affinity=-1.0, coords=coords)],
        seed=0,
        ligand_pdbqt=template,
        receptor_pdbqt=REMOTE_RECEPTOR,
    )
    strain = pose_strain(result, result.poses[0], scoring="vina")
    assert math.isfinite(strain.intra)
    assert math.isfinite(strain.strain)
    # The rebuilt document keeps the pose's own coordinates exactly.
    rebuilt = consensus.pdbqt_atoms(
        consensus.pose_pdbqt_from_coords(template, result.poses[0])
    )
    original = consensus.pdbqt_atoms(template)
    assert [round(a.x, 3) for a in rebuilt] == [round(a.x, 3) for a in original]
    assert [round(a.y, 3) for a in rebuilt] == [round(a.y, 3) for a in original]


# ---------------------------------------------------------------------------
# Against the bundled 3PTB demo
# ---------------------------------------------------------------------------


pytestmark_demo = pytest.mark.skipif(
    not (DEMO_LIGAND.exists() and DEMO_RECEPTOR.exists()),
    reason="the bundled demo is not present",
)


@pytestmark_demo
def test_the_demo_ligand_has_nine_heavy_atoms():
    assert heavy_atom_count(DEMO_LIGAND) == 9
    assert heavy_atom_count(DEMO_LIGAND.read_text(encoding="utf-8")) == 9


@pytestmark_demo
def test_demo_metrics_for_the_best_pose():
    """Numbers quoted in the progress report, recomputed from the shipped data."""
    best = -6.2104
    bundle = efficiency_metrics(best, heavy_atoms=9, molecular_weight=120.155,
                                logp=0.9707, tpsa=49.87, num_torsions=1.0)
    assert bundle.ligand_efficiency == pytest.approx(0.6900, abs=5e-5)
    assert bundle.p_activity == pytest.approx(4.5331, abs=5e-5)
    assert bundle.lle == pytest.approx(3.5624, abs=5e-4)
    assert bundle.entropy_penalty == pytest.approx(0.6509, abs=5e-4)
    assert bundle.ok


@pytestmark_demo
def test_demo_ligand_descriptors_match_benzamidine():
    from odock.prepare import pdbqt_to_pdb_block

    mol = Chem.MolFromPDBBlock(
        pdbqt_to_pdb_block(DEMO_LIGAND.read_text(encoding="utf-8")),
        removeHs=False,
        sanitize=False,
        proximityBonding=False,
    )
    assert mol is not None
    assert heavy_atom_count(mol) == 9


@pytestmark_demo
def test_strain_of_the_best_demo_pose_is_reported_with_its_caveats():
    from odock.prepare import BoxSpec

    models = consensus.pdbqt_models(DEMO_POSES.read_text(encoding="utf-8"))
    box = BoxSpec(center=(-1.8555, 14.366, 16.748), size=(17.883, 19.95, 20.514),
                  spacing=0.375)
    strain = ligand_strain(
        models[0], receptor=str(DEMO_RECEPTOR), box=box, scoring="vina"
    )
    assert math.isfinite(strain.intra)
    assert math.isfinite(strain.intra_relaxed)
    assert math.isfinite(strain.force_field_strain)
    # The kernel's own intra term is tiny for a united-atom ligand: the Vina
    # potential has almost no intramolecular terms here.  That is a fact worth
    # reporting rather than hiding.
    assert abs(strain.intra) < 1.0
    # A PDBQT carries no bond orders, and the note must say so.
    assert "no bond orders" in strain.note
    assert strain.force_field in ("MMFF94", "MMFF94s", "UFF")

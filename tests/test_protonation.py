# SPDX-License-Identifier: GPL-3.0-or-later
"""Protonation: what state the pipeline produced, and what it costs when it is wrong.

The worked example is benzamidine in trypsin: the bound species is the amidinium, and
the measured cost of getting that wrong is +22.09 kcal/mol of interaction energy where
the salt bridge should be favourable, 0.500 of score, and a different ranking of the
series.  The tests here pin the detector, the warning, the geometry and that cost.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

from odock import protonation as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "systems" / "3ptb"
RECEPTOR = DEMO / "receptor.pdbqt"

BENZAMIDINE = "N=C(N)c1ccccc1"


def mol(smiles: str, name: str = ""):
    out = Chem.MolFromSmiles(smiles)
    assert out is not None, smiles
    if name:
        out.SetProp("_Name", name)
    return out


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles, family, expected, observed",
    [
        (BENZAMIDINE, "amidinium", 1, 0),
        ("NC(=N)N", "amidinium", 1, 0),                     # guanidine
        ("NC(=[NH2+])N", "amidinium", 1, 1),                # guanidinium
        ("CC(=O)O", "carboxylate", -1, 0),
        ("CC(=O)[O-]", "carboxylate", -1, -1),
        ("NCc1ccccc1", "amine", 1, 0),
        ("C[NH3+]", "amine", 1, 1),
        ("OP(=O)(O)O", "phosphate", -1, 0),
        ("CS(=O)(=O)O", "sulfonate", -1, 0),
        ("CS", "thiolate", 0, 0),
        ("C[S-]", "thiolate", 0, -1),
        ("c1c[nH]cn1", "imidazole", 0, 0),
    ],
)
def test_the_detector_sees_both_states_of_every_family(smiles, family, expected, observed):
    """A detector that cannot see the *correct* state cannot compare against it.

    The charged rows matter as much as the neutral ones: the first version missed an
    amidinium (its imine nitrogen carries two hydrogens, so its connectivity is three,
    not two) and an ammonium (connectivity four), which meant the module could not
    recognise the very form it exists to recommend.
    """
    groups = [group for group in P.detect_groups(mol(smiles)) if group.family == family]
    assert groups, f"{smiles} should have a {family} group"
    assert groups[0].expected_charge == expected
    assert groups[0].observed_charge == observed


@pytest.mark.parametrize(
    "smiles", ["Oc1ccccc1", "Nc1ccccc1", "O=[N+]([O-])c1ccccc1", "c1ccccc1", "CCO"]
)
def test_no_false_positives_for_groups_that_are_neutral_at_seven_point_four(smiles):
    """Phenol, aniline, a nitro group and a plain alcohol are not charged groups.

    Aromatic amines are excluded from the amine pattern on purpose (aniline's pKa is
    4.6), imidazole and a thiol are perceived with an expected charge of 0 (pKa ≈ 6.0
    and ≈ 10), and a false warning is worse than a missed one.
    """
    report = P.protonation_report(mol(smiles))
    assert not report.is_ambiguous
    assert report.warnings == []


def test_one_group_per_central_atom_not_one_per_match():
    """Guanidinium has three N-C-N matches and a phosphate three P=O matches.

    Counting them separately would report a +2 guanidinium and a −3 phosphate, which
    no molecule has.
    """
    guanidinium = P.detect_groups(mol("NC(=[NH2+])N"))
    assert len([group for group in guanidinium if group.family == "amidinium"]) == 1
    assert sum(group.expected_charge for group in guanidinium) == 1
    phosphate = P.detect_groups(mol("OP(=O)(O)O"))
    assert len([group for group in phosphate if group.family == "phosphate"]) == 1
    # Lysine really does have three: two amines and a carboxylate.
    lysine = P.detect_groups(mol("NCCCC[C@H](N)C(=O)O"))
    assert len(lysine) == 3
    assert sum(group.expected_charge for group in lysine) == 1


# ---------------------------------------------------------------------------
# The report and the warning
# ---------------------------------------------------------------------------


def test_the_neutral_amidine_is_flagged_with_the_charges_that_were_assigned():
    report = P.protonation_report(mol(BENZAMIDINE, "benzamidine"), charges=None)
    assert report.net_charge == 0 and report.expected_net_charge == 1
    assert report.is_ambiguous is True
    assert len(report.warnings) == 1
    assert "NEUTRAL" in report.warnings[0] and "wrong species" in report.warnings[0]
    # The summary states both numbers, so a reader sees the contradiction itself.
    assert "net charge +0" in report.summary()
    assert "would be +1" in report.summary()
    assert report.as_dict()["ambiguous"] is True
    assert report.as_dict()["groups"][0]["state"] == "neutral (disagrees with the stated pH)"
    # With charges, the warning quotes what the model actually put on the group.
    with_charges = P.protonation_report(
        mol(BENZAMIDINE), charges=np.array([-0.30, 0.40, -0.32, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1])
    )
    assert "-0.30 e" in with_charges.warnings[0]


def test_the_charged_form_is_not_flagged():
    """``charged_copy`` produces a real amidinium, and the report says so."""
    charged = P.charged_copy(mol(BENZAMIDINE), "amidinium", name="benzamidinium")
    assert P.net_charge(charged) == 1
    assert rdMolDescriptors.CalcMolFormula(charged) == "C7H9N2+"
    report = P.protonation_report(charged)
    assert report.is_ambiguous is False
    assert report.warnings == []
    assert report.net_charge == report.expected_net_charge == 1
    # And the neutral molecule really has one fewer hydrogen.
    neutral = rdMolDescriptors.CalcMolFormula(mol(BENZAMIDINE))
    assert neutral == "C7H8N2"
    with pytest.raises(ValueError, match="no carboxylate group"):
        P.charged_copy(mol(BENZAMIDINE), "carboxylate")


def test_a_molecule_with_no_ionisable_group_says_so():
    report = P.protonation_report(mol("c1ccccc1", "benzene"))
    assert report.groups == [] and report.is_ambiguous is False
    assert "no amidinium" in report.notes[0]
    assert "no formally charged group" in report.summary()


# ---------------------------------------------------------------------------
# The geometric contradiction
# ---------------------------------------------------------------------------


def fake_receptor_atoms(residue: str, position, element: str = "O"):
    """One residue's worth of atom-like objects at a chosen place."""
    return [
        type(
            "Atom", (), {
                "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
                "element": element, "residue_name": residue, "chain": "A",
                "residue_number": "189",
            }
        )()
    ]


def embedded(smiles: str, name: str = ""):
    from rdkit.Chem import AllChem

    work = Chem.AddHs(mol(smiles, name))
    params = AllChem.ETKDGv3()
    params.randomSeed = 20240101
    params.numThreads = 1
    assert len(AllChem.EmbedMultipleConfs(work, numConfs=1, params=params)) == 1
    return work


def test_the_salt_bridge_warning_fires_on_a_neutral_cation_near_an_anion():
    """The check is geometry plus chemistry: a group, a residue, and 4 Å."""
    ligand = embedded(BENZAMIDINE, "benzamidine")
    conformer = ligand.GetConformer()
    nitrogens = [a.GetIdx() for a in ligand.GetAtoms() if a.GetAtomicNum() == 7]
    assert nitrogens
    anchor = conformer.GetAtomPosition(nitrogens[0])
    # An ASP oxygen 2.9 Å from the amidine nitrogen: a real salt bridge geometry.
    partners = fake_receptor_atoms("ASP", (anchor.x + 2.9, anchor.y, anchor.z))
    warnings = P.salt_bridge_warnings(ligand, partners, name="benzamidine")
    assert len(warnings) == 1
    assert "ASP" in warnings[0] and "NEUTRAL" in warnings[0] and "wrong sign" in warnings[0]
    # The same group, charged, is not a warning: it is the interaction working.
    charged = embedded("NC(=[NH2+])c1ccccc1", "benzamidinium")
    charged_conformer = charged.GetConformer()
    charged_n = [a.GetIdx() for a in charged.GetAtoms() if a.GetAtomicNum() == 7]
    charged_anchor = charged_conformer.GetAtomPosition(charged_n[0])
    near = fake_receptor_atoms(
        "ASP", (charged_anchor.x + 2.9, charged_anchor.y, charged_anchor.z)
    )
    assert P.salt_bridge_warnings(charged, near) == []


def test_the_warning_does_not_fire_when_the_residue_is_too_far_or_the_charge_agrees():
    ligand = embedded(BENZAMIDINE)
    conformer = ligand.GetConformer()
    nitrogen = [a.GetIdx() for a in ligand.GetAtoms() if a.GetAtomicNum() == 7][0]
    anchor = conformer.GetAtomPosition(nitrogen)
    # 6 Å away: outside the 4 Å salt-bridge ceiling.
    far = fake_receptor_atoms("ASP", (anchor.x + 6.0, anchor.y, anchor.z))
    assert P.salt_bridge_warnings(ligand, far) == []
    # A cationic residue next to the neutral *cation* is not a contradiction.
    lysine = fake_receptor_atoms("LYS", (anchor.x + 2.9, anchor.y, anchor.z), element="N")
    assert P.salt_bridge_warnings(ligand, lysine) == []
    # A neutral carboxylate next to a cationic residue *is* one, in the other direction.
    acid = embedded("CC(=O)O", "acetic acid")
    acid_conf = acid.GetConformer()
    oxygen = [a.GetIdx() for a in acid.GetAtoms() if a.GetAtomicNum() == 8][0]
    spot = acid_conf.GetAtomPosition(oxygen)
    warnings = P.salt_bridge_warnings(
        acid, fake_receptor_atoms("LYS", (spot.x + 2.9, spot.y, spot.z), element="N")
    )
    assert len(warnings) == 1 and "LYS" in warnings[0]
    # No receptor at all: nothing to check, and no warning.
    assert P.salt_bridge_warnings(ligand, []) == []


def test_the_warning_refuses_coordinates_in_another_atom_order():
    """Group indices index the molecule, so a mismatched array is a wrong answer.

    Measured: with the PDBQT's atom order instead of the SMILES', the real 2.87 Å
    amidine–Asp189 distance came out as 4.96 Å and the warning silently never fired.
    """
    ligand = embedded(BENZAMIDINE)
    with pytest.raises(ValueError, match="atom order"):
        P.salt_bridge_warnings(
            ligand, fake_receptor_atoms("ASP", (0.0, 0.0, 0.0)),
            ligand_coords=np.zeros((3, 3)),
        )


# ---------------------------------------------------------------------------
# The correction, and what it costs
# ---------------------------------------------------------------------------


def test_the_formal_charge_correction_moves_the_group_and_nothing_else():
    ligand = mol(BENZAMIDINE)
    original = np.array([-0.30, 0.40, -0.32, 0.10, -0.10, -0.10, -0.10, 0.10, 0.10])
    corrected, changed = P.correct_formal_charges(ligand, original)
    amidine = [index for group in P.detect_groups(ligand) for index in group.atoms]
    assert changed and "amidinium" in changed[0]
    assert corrected[amidine].sum() == pytest.approx(1.0)
    assert original[amidine].sum() == pytest.approx(-0.22, abs=0.01)
    # Every atom outside the group is untouched, and the total shift is the difference.
    outside = [index for index in range(len(original)) if index not in amidine]
    assert corrected[outside] == pytest.approx(original[outside])
    assert corrected.sum() - original.sum() == pytest.approx(1.0 - original[amidine].sum())
    # A molecule with nothing to correct is returned unchanged, with no descriptions.
    unchanged, notes = P.correct_formal_charges(mol("c1ccccc1"), np.zeros(6))
    assert notes == [] and unchanged == pytest.approx(np.zeros(6))
    # An already-correct group is left alone: the group sums to its formal +1.
    charged = mol("NC(=[NH2+])c1ccccc1")
    already = np.full(charged.GetNumAtoms(), 1.0 / 3.0)
    _, notes = P.correct_formal_charges(charged, already)
    assert notes == []


def test_the_capability_predicate_is_the_entry_point_consumers_use():
    """Can this charge set hold the formal charge the chemistry implies?

    Measured on the 3PTB ligand: the neutral file charges give the amidine −0.62 e
    where the formal charge is +1, and Gasteiger on the *correct* amidinium still gives
    a negative number.  Only a corrected set passes — which is why the predicate exists,
    and why a consumer should call it rather than reason about Gasteiger again.
    """
    neutral = mol(BENZAMIDINE)
    file_charges = np.array([-0.30, 0.40, -0.32, 0.10, -0.10, -0.10, -0.10, 0.10, 0.10])
    capability = P.can_represent_formal_charge(neutral, file_charges)
    assert capability.possible is False
    assert capability.groups and capability.groups[0][0] == "amidinium"
    assert capability.groups[0][1] == 1 and capability.groups[0][2] < 0.0
    assert "cannot represent that formal charge" in capability.statement()
    assert "PROTONATION" in capability.reasons[0]
    assert capability.as_dict()["possible"] is False
    # The corrected set passes, and says so.
    corrected, _ = P.correct_formal_charges(neutral, file_charges)
    fixed = P.can_represent_formal_charge(neutral, corrected)
    assert fixed.possible is True
    assert fixed.groups[0][2] == pytest.approx(1.0)
    assert "represents every formal charge" in fixed.statement()
    # A molecule with nothing to represent cannot fail the check.
    empty = P.can_represent_formal_charge(mol("c1ccccc1"), None)
    assert empty.possible is True and empty.groups == []
    assert "no formally charged group" in empty.statement()
    # A charge set that does not line up with the molecule is rejected, not mis-indexed.
    mismatched = P.can_represent_formal_charge(neutral, np.zeros(3))
    assert mismatched.possible is False
    assert "cannot be checked" in mismatched.reasons[0]
    # The molecule's own formal charges answer the protonation question from the same
    # call: a neutral amidine does not carry the +1 its pH implies.
    assert P.can_represent_formal_charge(neutral, None).possible is False
    assert P.can_represent_formal_charge(mol("NC(=[NH2+])c1ccccc1"), None).possible is True


@pytest.mark.slow
def test_the_real_complex_reproduces_the_documented_cost():
    """The (a)/(b)/(c) measurement of `docs/PROTONATION.md` §1, pinned.

    At the co-crystallised 3PTB pose: the prepared charges give an *unfavourable*
    interaction (+22 kcal/mol) so the electrostatic term is clamped to zero, Gasteiger
    on the amidinium barely moves it, and only the formal charge makes the term
    favourable and doubles the score.
    """
    from odock import pocket_score as PS

    if not RECEPTOR.exists():
        pytest.skip("the bundled 3PTB receptor is missing")
    import json

    box = json.loads((DEMO / "box.json").read_text(encoding="utf-8"))
    atoms = PS.read_pdbqt_atoms(RECEPTOR)
    ligand = PS.read_pdbqt_atoms(DEMO / "ligand.pdbqt")
    coords = np.asarray([a.coords for a in ligand])
    file_charges = np.asarray([a.charge for a in ligand])
    radii = np.asarray([PS._radius_of(a.element) for a in ligand])
    pocket = PS.PocketField.build(atoms, center=box["center"], size=box["size"])
    pocket.set_size_reference(PS.ligand_volume(radii))

    # The salt bridge this is all about is real, and short.
    nitrogens = [index for index, atom in enumerate(ligand) if atom.element == "N"]
    asp = [atom for atom in atoms if atom.residue_name == "ASP" and atom.residue_number == "189"]
    assert asp, "3PTB must have an Asp189"
    asp_coords = np.asarray([atom.coords for atom in asp])
    distance = float(
        np.sqrt(((coords[nitrogens][:, None, :] - asp_coords[None, :, :]) ** 2).sum(axis=2)).min()
    )
    assert distance == pytest.approx(2.87, abs=0.15)

    # (a) as prepared: unfavourable, clamped, shape only.
    prepared = PS.pocket_score(coords, file_charges, pocket, radii=radii)
    assert prepared.esp > 15.0
    assert prepared.esp_term_used is False and prepared.esp_term == 0.0
    assert prepared.is_shape_only is True
    assert "clamp" in prepared.esp_note and "PROTONATION" in prepared.esp_note
    assert prepared.combined == pytest.approx(0.5 * prepared.shape, abs=1e-9)

    # (c) the formal charge restored: favourable, and the score doubles.
    # Measured: −13.95 kcal/mol from the *file's* neutral charges (the probe in
    # docs/PROTONATION.md starts from Gasteiger-on-the-amidinium and reaches −28.75;
    # the sign and the conclusion are the same, the magnitude depends on the starting
    # distribution, which is why the document reports both).
    corrected, changed = P.correct_formal_charges(mol(BENZAMIDINE), file_charges)
    assert changed
    fixed = PS.pocket_score(coords, corrected, pocket, radii=radii)
    assert fixed.esp < -10.0, "the sign must flip: this is the salt bridge"
    assert fixed.esp_term_used is True and fixed.esp_term == 1.0
    assert fixed.combined > prepared.combined + 0.4
    assert "clamp" not in fixed.esp_note
    assert fixed.as_dict()["esp_term_used"] is True
    assert prepared.as_dict()["esp_term_used"] is False

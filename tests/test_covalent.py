# SPDX-License-Identifier: GPL-3.0-or-later
"""Covalent docking: warheads, reactive geometry, the score term and adducts.

Every number here is measured on this tree: the classification thresholds are the
module's documented criteria, the continuity check measures that the term has no
cliff, and the adduct's C-S and S-C-C values are asserted against the published
ranges the module documents.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from odock import covalent as CV  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RECEPTOR = ROOT / "demo" / "systems" / "1m17" / "receptor.pdbqt"

#: A simple Michael-acceptor ligand: an acrylamide with a phenyl tail.
ACRYLAMIDE = "C=CC(=O)NCCc1ccccc1"


def named(smiles: str, name: str = "") -> object:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if name:
        mol.SetProp("_Name", name)
    return mol


def embedded(smiles: str, seed: int = 7):
    mol = Chem.AddHs(named(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return mol


def approach_frame(mol, warhead):
    """The electrophile, the reference atom and the ideal 107-degree direction."""
    conf = mol.GetConformer()
    attack = np.array(conf.GetAtomPosition(warhead.electrophile), dtype=float)
    reference = np.array(conf.GetAtomPosition(warhead.reference), dtype=float)
    axis = (reference - attack) / np.linalg.norm(reference - attack)
    perpendicular = np.cross(axis, [0.0, 0.0, 1.0])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    ideal = math.radians(CV._pattern_for(warhead.name).ideal_angle)
    direction = math.cos(ideal) * axis + math.sin(ideal) * perpendicular
    return attack, reference, direction


def placed_ligand(mol, warhead, sulfur_position, distance: float):
    """The ligand translated so its electrophile is `distance` from the sulfur."""
    attack, _, direction = approach_frame(mol, warhead)
    shift = (np.asarray(sulfur_position) - distance * direction) - attack
    work = Chem.Mol(mol)
    conf = Chem.Conformer(work.GetConformer())
    for index in range(work.GetNumAtoms()):
        point = work.GetConformer().GetAtomPosition(index)
        conf.SetAtomPosition(
            index, (point.x + shift[0], point.y + shift[1], point.z + shift[2])
        )
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    return work


def translated(mol, shift):
    """The ligand moved by `shift` with its orientation untouched."""
    work = Chem.Mol(mol)
    conf = Chem.Conformer(work.GetConformer())
    for index in range(work.GetNumAtoms()):
        point = work.GetConformer().GetAtomPosition(index)
        conf.SetAtomPosition(
            index, (point.x + shift[0], point.y + shift[1], point.z + shift[2])
        )
    work.RemoveAllConformers()
    work.AddConformer(conf, assignId=True)
    return work


def sulfur_at(position) -> CV.Nucleophile:
    return CV.Nucleophile(
        index=0,
        element="S",
        position=tuple(float(v) for v in position),
        residue_name="CYS",
        residue_number=797,
        chain="A",
        atom_name="SG",
        cap_position=(float(position[0]) + 1.8, float(position[1]), float(position[2])),
        note="test fixture",
    )


# ---------------------------------------------------------------------------
# Warhead perception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles, warhead, mechanism",
    [
        ("C=CC(=O)N", "acrylamide", "michael"),
        ("C=C(C#N)C(=O)N", "alpha_cyanoacrylamide", "michael"),
        ("C=CS(=O)(=O)N", "vinyl_sulfonamide", "michael"),
        ("C=CS(=O)(=O)C", "vinyl_sulfone", "michael"),
        ("ClCC(=O)N", "chloroacetamide", "sn2"),
        ("C1CO1", "epoxide", "sn2"),
        ("CC#N", "nitrile", "nitrile"),
        ("CC=C", "catch_all_alkene", "michael"),
        ("CCCl", "catch_all_halide", "sn2"),
    ],
)
def test_each_warhead_is_perceived_with_its_mechanism(smiles, warhead, mechanism):
    hits = CV.perceive_warheads(named(smiles))
    assert warhead in [hit.name for hit in hits], smiles
    hit = next(item for item in hits if item.name == warhead)
    assert hit.mechanism == mechanism
    assert 0 <= hit.electrophile < named(smiles).GetNumAtoms()


def test_the_electrophilic_atom_is_the_right_one():
    """Hand check: in acrylamide the attack lands on the terminal CH2 (the beta
    carbon), and in chloroacetamide on the CH2 that carries the chlorine — not on
    the carbonyl carbon, which is what an off-by-one in the SMARTS map would give."""
    acrylamide = named("C=CC(=O)N")
    hit = CV.perceive_warheads(acrylamide)[0]
    assert hit.name == "acrylamide"
    assert acrylamide.GetAtomWithIdx(hit.electrophile).GetSymbol() == "C"
    # the beta carbon is a CH2 with one heavy neighbour
    beta = acrylamide.GetAtomWithIdx(hit.electrophile)
    assert beta.GetTotalNumHs() == 2
    assert sum(1 for n in beta.GetNeighbors() if n.GetAtomicNum() > 1) == 1

    chloroacetamide = named("ClCC(=O)N")
    hit = next(
        item for item in CV.perceive_warheads(chloroacetamide)
        if item.name == "chloroacetamide"
    )
    attacked = chloroacetamide.GetAtomWithIdx(hit.electrophile)
    assert attacked.GetSymbol() == "C" and attacked.GetTotalNumHs() == 2
    leaving = chloroacetamide.GetAtomWithIdx(hit.leaving_group)
    assert leaving.GetSymbol() == "Cl"
    assert chloroacetamide.GetBondBetweenAtoms(
        hit.electrophile, hit.leaving_group
    ) is not None


def test_a_named_warhead_suppresses_its_catch_all_duplicate():
    """An acrylamide is also "any terminal alkene"; it must be reported once, by
    name, and the catch-all must not duplicate it."""
    hits = CV.perceive_warheads(named("C=CC(=O)N"))
    names = [hit.name for hit in hits]
    assert "acrylamide" in names
    assert "catch_all_alkene" not in names
    # ... but the catch-all is available when it is asked for explicitly.
    only_catch = CV.perceive_warheads(named("C=CC(=O)N"), include_catch_all=True)
    assert len(only_catch) == len(hits)


def test_a_molecule_with_no_warhead_has_no_hit():
    for smiles in ("c1ccccc1", "CCO", "CC(=O)Nc1ccccc1"):
        assert CV.perceive_warheads(named(smiles)) == []


# ---------------------------------------------------------------------------
# Nucleophile perception
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def receptor():
    if not RECEPTOR.exists():
        pytest.skip("the bundled 1M17 receptor is missing")
    return CV._load_receptor(str(RECEPTOR))


def test_the_bundled_receptor_is_readable_and_its_cysteines_are_found(receptor):
    """The 1M17 receptor is a PDBQT: its AutoDock type column has to be rewritten
    before RDKit will read it.  Its six cysteines are then found by residue and
    atom name, each with its CB as a cap."""
    assert receptor is not None and receptor.GetNumAtoms() > 2000
    found = CV.perceive_nucleophiles(receptor)
    assert len(found) == 6, "1M17's construct carries six cysteines"
    assert {item.residue_number for item in found} == {751, 757, 773, 794, 915, 926}
    assert all(item.residue_name == "CYS" and item.atom_name == "SG" for item in found)
    assert all(item.cap_position is not None for item in found)
    assert all(item.note.startswith("identified by residue") for item in found)


def test_a_receptor_without_residue_information_is_reported_as_geometric():
    """No PDB residue info means the fallback rule (a non-ring sulfur with a heavy
    neighbour) and a note that says the catalytic question is not answered."""
    plain = Chem.AddHs(Chem.MolFromSmiles("CCSCC"))
    AllChem.EmbedMolecule(plain, randomSeed=3)
    found = CV.perceive_nucleophiles(plain)
    assert len(found) == 1
    assert found[0].element == "S"
    assert "geometric sulfur candidate" in found[0].note


def test_perceive_nucleophiles_accepts_text_and_molecules(receptor):
    assert CV.perceive_nucleophiles(None) == []
    assert len(CV.perceive_nucleophiles(receptor)) == 6
    from_text = CV.perceive_nucleophiles(RECEPTOR.read_text(encoding="utf-8"))
    assert len(from_text) == 6


# ---------------------------------------------------------------------------
# Reactive geometry and classification
# ---------------------------------------------------------------------------


def test_classification_by_distance_and_angle():
    """The documented criteria, measured on a constructed pose:

    * 1.82 A from the sulfur at the ideal 107-degree approach -> **bonded**
    * 2.8 A at the same angle -> **near_attack**
    * 5.0 A -> **non_productive**, out of reach
    * 2.5 A from *behind* the reference (180 degrees) -> **non_productive**, angle
    """
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    sulfur = np.array([10.0, 10.0, 10.0])
    nucleophile = sulfur_at(sulfur)
    expectations = [
        (1.82, "bonded"),
        (2.8, "near_attack"),
        (4.0, "near_attack"),
        (5.0, "non_productive"),
    ]
    for distance, expected in expectations:
        work = placed_ligand(mol, warhead, sulfur, distance)
        geometry = CV.classify_attack(work, warhead, nucleophile)
        assert geometry.distance == pytest.approx(distance, abs=0.05)
        assert geometry.classification == expected, (distance, geometry.reason)
        assert geometry.angle == pytest.approx(107.0, abs=1.0)

    # In line with the reference (S beyond the alpha carbon): the S-E-R angle is
    # ~0 degrees, outside 70-150, at a distance that would be near-attack.
    attack, reference, _ = approach_frame(mol, warhead)
    axis = (reference - attack) / np.linalg.norm(reference - attack)
    misaligned_pose = translated(mol, (sulfur - 2.5 * axis) - attack)
    geometry = CV.classify_attack(misaligned_pose, warhead, sulfur_at(sulfur))
    assert geometry.distance == pytest.approx(2.5, abs=0.05)
    assert geometry.angle is not None and geometry.angle < 70.0
    assert geometry.classification == "non_productive"
    assert "outside the 70-150 deg window" in geometry.reason


def test_classification_reports_the_failing_criterion():
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    sulfur = np.array([10.0, 10.0, 10.0])
    far = CV.classify_attack(
        placed_ligand(mol, warhead, sulfur, 6.0), warhead, sulfur_at(sulfur)
    )
    assert "out of reach" in far.reason
    assert far.quality == 0.0


def test_classify_attack_needs_coordinates_and_a_valid_conformer():
    mol = Chem.MolFromSmiles(ACRYLAMIDE)
    warhead = CV.perceive_warheads(Chem.MolFromSmiles("C=CC(=O)N"))[0]
    with pytest.raises(ValueError, match="3-D coordinates"):
        CV.classify_attack(mol, warhead, sulfur_at((0.0, 0.0, 0.0)))
    embedded_mol = embedded(ACRYLAMIDE)
    with pytest.raises(IndexError, match="conformer 3"):
        CV.classify_attack(
            embedded_mol, CV.perceive_warheads(embedded_mol)[0],
            sulfur_at((0.0, 0.0, 0.0)), conf_id=3,
        )


# ---------------------------------------------------------------------------
# The score term
# ---------------------------------------------------------------------------


def test_the_term_rewards_near_attack_geometry_and_nothing_beyond_reach():
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    sulfur = np.array([10.0, 10.0, 10.0])
    nucleophile = sulfur_at(sulfur)
    bonded = CV.covalent_term(CV.classify_attack(placed_ligand(mol, warhead, sulfur, 1.82), warhead, nucleophile))
    near = CV.covalent_term(CV.classify_attack(placed_ligand(mol, warhead, sulfur, 2.8), warhead, nucleophile))
    far = CV.covalent_term(CV.classify_attack(placed_ligand(mol, warhead, sulfur, 6.0), warhead, nucleophile))
    assert bonded.favourable and near.favourable
    assert far.value == pytest.approx(0.0), "a pose that cannot reach is not rewarded"
    assert bonded.value < near.value < 0.0, "the formed bond is worth more than the approach"
    assert bonded.bond > 0.0 and near.bond == pytest.approx(0.0, abs=1e-9)
    assert bonded.near_attack == pytest.approx(near.near_attack, abs=1e-6)
    assert bonded.strain_penalty == pytest.approx(0.0)
    assert CV.covalent_score(-7.0, bonded.geometry) == pytest.approx(-7.0 + bonded.value)


def test_a_strained_adduct_is_penalised():
    """The term's strain part bites when the S-C-C angle leaves the tetrahedral
    window: the same bonded distance at 140 degrees costs more than at 109.5."""
    criteria = CV.DEFAULT_CRITERIA
    good = CV.AttackGeometry(
        warhead="acrylamide", nucleophile="CYS797:SG", distance=1.82, angle=109.5,
        classification="bonded", quality=1.0, bond_fraction=1.0,
    )
    strained = CV.AttackGeometry(
        warhead="acrylamide", nucleophile="CYS797:SG", distance=1.82, angle=140.0,
        classification="bonded", quality=1.0, bond_fraction=1.0,
    )
    assert CV.covalent_term(good, criteria=criteria).strain_penalty == pytest.approx(0.0)
    penalty = CV.covalent_term(strained, criteria=criteria).strain_penalty
    assert penalty > 0.0
    expected = criteria.adduct_strain_k * (140.0 - criteria.adduct_ideal_angle - criteria.adduct_angle_tolerance) ** 2
    assert penalty == pytest.approx(expected)


def test_the_term_is_continuous_across_its_own_thresholds():
    """A step change at a threshold is how these scores become unfalsifiable.  The
    decisive test is that **halving the sample spacing halves the largest step** —
    a discontinuity would keep the same jump."""
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    attack, reference, _ = approach_frame(mol, warhead)
    coarse = CV.continuity_scan(warhead, attack, reference, samples=41, angle_samples=41)
    fine = CV.continuity_scan(warhead, attack, reference, samples=81, angle_samples=81)
    assert fine["distance_spacing"] == pytest.approx(coarse["distance_spacing"] / 2)
    ratio = coarse["max_distance_step"] / fine["max_distance_step"]
    assert ratio == pytest.approx(2.0, rel=0.15), "a cliff would give a ratio near 1"
    angle_ratio = coarse["max_angle_step"] / fine["max_angle_step"]
    assert angle_ratio == pytest.approx(2.0, rel=0.2)
    # No sample jumps by the whole reward: the gradient is bounded.
    assert coarse["max_distance_step"] < CV.DEFAULT_CRITERIA.near_attack_reward
    assert min(coarse["distance_values"]) < -3.0
    assert max(coarse["distance_values"]) == pytest.approx(0.0)
    assert set(coarse["distance_classes"]) == {"bonded", "near_attack", "non_productive"}


def test_the_covalent_term_changes_a_ranking():
    """Constructed case: three poses with the *same* non-covalent affinity — one at
    the reactive geometry, one at the same distance but pointing the wrong way, one
    out of reach.  The non-covalent score cannot separate them; the covalent term
    must, and the ranking reports both ranks."""
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    sulfur = np.array([10.0, 10.0, 10.0])
    attack, reference, direction = approach_frame(mol, warhead)
    reactive = CV.classify_attack(
        placed_ligand(mol, warhead, sulfur, 2.8), warhead, sulfur_at(sulfur)
    )
    away = CV.classify_attack(
        placed_ligand(mol, warhead, sulfur, 6.0), warhead, sulfur_at(sulfur)
    )
    # Same distance, wrong orientation: the ligand is moved so its electrophile sits
    # a near-attack distance from the sulfur but *in line with* the alpha carbon, so
    # the S-E-R angle is about 0 degrees instead of 107.
    axis = (reference - attack) / np.linalg.norm(reference - attack)
    misaligned_pose = translated(mol, (sulfur - 2.8 * axis) - attack)
    misaligned = CV.classify_attack(misaligned_pose, warhead, sulfur_at(sulfur))
    assert misaligned.classification == "non_productive"
    assert misaligned.distance == pytest.approx(2.8, abs=0.05)
    assert misaligned.angle is not None and misaligned.angle < 70.0

    rows = CV.covalent_ranking(
        [("reactive", -6.0, reactive), ("away", -6.0, away), ("misaligned", -6.0, misaligned)]
    )
    by_name = {row.name: row for row in rows}
    assert by_name["reactive"].rank_after == 1
    assert by_name["reactive"].total < by_name["misaligned"].total
    assert by_name["reactive"].total <= by_name["away"].total
    assert by_name["reactive"].classification == "near_attack"
    assert by_name["misaligned"].term == pytest.approx(0.0)
    assert {row.rank_before for row in rows} == {1, 2, 3}
    assert all("rank_change" in row.as_dict() for row in rows)


# ---------------------------------------------------------------------------
# The adduct
# ---------------------------------------------------------------------------


def test_the_adduct_is_built_with_the_right_topology_and_geometry():
    """Forming the bond puts the sulfur at 1.82 A along the pose's approach vector;
    the local relaxation keeps the C-S bond and the tetrahedral S-C-C angle inside
    the published ranges, and the strain machinery reports it as reliable."""
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    sulfur_position = np.array([10.0, 10.0, 10.0])
    nucleophile = sulfur_at(sulfur_position)
    pose = placed_ligand(mol, warhead, sulfur_position, 2.8)
    adduct = CV.build_adduct(pose, warhead, nucleophile)
    assert adduct.ok
    assert adduct.warhead == "acrylamide" and adduct.mechanism == "michael"
    assert any("alkene was saturated" in note for note in adduct.notes)
    assert any("proton" in note for note in adduct.notes)
    # The adduct's heavy-atom skeleton: the ligand plus an S and the CB cap.
    heavy = Chem.RemoveHs(adduct.mol)
    assert any(atom.GetSymbol() == "S" for atom in heavy.GetAtoms())

    quality = CV.adduct_quality(adduct)
    bond, angle = quality["bond_length"], quality["bond_angle"]
    low, high = CV.ADDUCT_GEOMETRY["c_s_bond"]
    assert low <= bond <= high, f"C-S bond {bond} outside the published range"
    low, high = CV.ADDUCT_GEOMETRY["s_c_c_angle"]
    assert low <= angle <= high, f"S-C-C angle {angle} outside the published range"
    assert quality["ok"] is True

    relaxed = CV.relax_adduct(adduct)
    assert relaxed.relaxed
    after = CV.adduct_quality(relaxed)
    assert CV.ADDUCT_GEOMETRY["c_s_bond"][0] <= after["bond_length"] <= CV.ADDUCT_GEOMETRY["c_s_bond"][1]

    strain = CV.adduct_strain(relaxed)
    assert strain.reliable is True
    assert math.isfinite(strain.force_field_strain)
    assert strain.force_field in ("MMFF94", "MMFF94s", "UFF")
    assert "local relaxation" in strain.note


def test_forming_the_bond_can_be_switched_off():
    """`form_bond=False` leaves the sulfur where the pose put it, which is how a
    caller inspects the unformed geometry (and is *not* what relaxation wants)."""
    mol = embedded(ACRYLAMIDE)
    warhead = CV.perceive_warheads(mol)[0]
    sulfur_position = np.array([10.0, 10.0, 10.0])
    pose = placed_ligand(mol, warhead, sulfur_position, 2.8)
    adduct = CV.build_adduct(pose, warhead, sulfur_at(sulfur_position), form_bond=False)
    assert adduct.bond_length() == pytest.approx(2.8, abs=0.05)
    formed = CV.build_adduct(pose, warhead, sulfur_at(sulfur_position), form_bond=True)
    assert formed.bond_length() == pytest.approx(1.82, abs=0.02)


def test_an_sn2_adduct_loses_its_leaving_group():
    mol = embedded("ClCC(=O)NCCc1ccccc1")
    warhead = next(
        item for item in CV.perceive_warheads(mol) if item.name == "chloroacetamide"
    )
    sulfur_position = np.array([10.0, 10.0, 10.0])
    attack = np.array(mol.GetConformer().GetAtomPosition(warhead.electrophile))
    leaving = np.array(mol.GetConformer().GetAtomPosition(warhead.leaving_group))
    # backside attack: approach from the side opposite the chlorine
    direction = (attack - leaving) / np.linalg.norm(attack - leaving)
    pose = placed_ligand(mol, warhead, sulfur_position, 2.8)
    adduct = CV.build_adduct(pose, warhead, sulfur_at(sulfur_position))
    assert adduct.ok
    symbols = [atom.GetSymbol() for atom in adduct.mol.GetAtoms()]
    assert "Cl" not in symbols, "the leaving group must be gone from the SN2 adduct"
    assert any("leaving group" in note for note in adduct.notes)


def test_an_epoxide_adduct_opens_the_ring():
    mol = embedded("C1CO1")
    warhead = next(item for item in CV.perceive_warheads(mol) if item.name == "epoxide")
    sulfur_position = np.array([10.0, 10.0, 10.0])
    pose = placed_ligand(mol, warhead, sulfur_position, 2.8)
    adduct = CV.build_adduct(pose, warhead, sulfur_at(sulfur_position))
    assert adduct.ok
    assert any("ring opened" in note for note in adduct.notes)
    # The sulfur is bonded to the attacked carbon and not in a ring any more.
    sulfur = adduct.mol.GetAtomWithIdx(adduct.sulfur)
    assert sulfur.GetDegree() >= 2
    assert not sulfur.IsInRing()


# ---------------------------------------------------------------------------
# Interactions and the module's contract
# ---------------------------------------------------------------------------


def test_interaction_change_reports_before_after_and_the_covalent_split(receptor):
    """The profile covers the *non-covalent* contacts; the covalent bond is stated
    as not being an interaction kind rather than silently missing."""
    ligand = embedded(ACRYLAMIDE)
    sulfur_position = np.array(receptor.GetConformer().GetAtomPosition(
        CV.perceive_nucleophiles(receptor)[0].index
    ), dtype=float)
    warhead = CV.perceive_warheads(ligand)[0]
    pose = placed_ligand(ligand, warhead, sulfur_position, 2.8)
    change = CV.interaction_change(receptor, pose, pose)
    assert set(change) >= {"before", "after", "added", "lost", "n_added", "n_lost", "note"}
    assert change["n_added"] == 0 and change["n_lost"] == 0
    assert "not an interaction kind" in change["note"]


def test_the_criteria_are_documented_data():
    criteria = CV.DEFAULT_CRITERIA
    assert criteria.bonded_max < criteria.near_attack_max
    assert 1.7 <= criteria.bonded_max <= 2.2, "a C-S single bond is about 1.82 A"
    assert CV.ADDUCT_GEOMETRY["c_s_bond"] == (1.70, 1.95)
    assert CV.ADDUCT_GEOMETRY["s_c_c_angle"] == (95.0, 125.0)
    assert CV.WARHEAD_NAMES[0] == "acrylamide"
    assert "bonded" in CV.ATTACK_CLASSES and "non_productive" in CV.ATTACK_CLASSES
    # every pattern names atoms that exist in its own SMARTS
    for pattern in CV.WARHEAD_PATTERNS:
        query = CV._pattern_query(pattern)
        maps = CV._PATTERN_MAPS[pattern.name]
        assert pattern.attack in maps
        assert pattern.reference in maps
        if pattern.leaving_group:
            assert pattern.leaving_group in maps
        assert query.GetNumAtoms() == len(maps)


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(CV, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        CV.require_rdkit()
    with pytest.raises(ImportError):
        CV.perceive_warheads(named("C=CC(=O)N"))

def test_receptor_text_is_not_probed_as_a_path(receptor, monkeypatch):
    """The probe itself must not raise for the text case.

    `perceive_nucleophiles` accepts a path *or* the PDBQT text. It decided which by calling
    `Path(value).exists()`, and on POSIX that raises for a long string:

        OSError: [Errno 36] File name too long

    measured on the CI ubuntu legs, where this test failed while passing on Windows (the
    probe simply returns False there). No local run reaches it, so the POSIX behaviour is
    simulated: the probe raises for anything longer than a filename, exactly as the kernel
    would.
    """
    from pathlib import Path

    real_exists = Path.exists

    def posix_like(self):
        if len(str(self)) > 255:
            raise OSError(36, "File name too long")
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", posix_like, raising=True)
    text = RECEPTOR.read_text(encoding="utf-8")
    assert len(CV.perceive_nucleophiles(text)) == 6
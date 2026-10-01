# SPDX-License-Identifier: GPL-3.0-or-later
"""Interaction fingerprints: encoding, similarity, pharmacophore, water bridges.

The synthetic geometries are chosen so the geometric criteria of
:func:`odock.analysis.profile_interactions` are satisfied by construction -- a
linear D-H...A hydrogen bond, a carboxylate salt bridge, a face-to-face pi
stack -- so the fingerprints that come out have hand-checkable counts rather
than "whatever the profiler happened to find".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from odock import analysis, consensus
from odock.analysis import (
    FingerprintKey,
    FingerprintSchema,
    Interaction,
    fingerprint_similarity,
    interaction_fingerprint,
    pharmacophore_summary,
    pose_fingerprints,
    profile_interactions,
    similarity_matrix,
    water_mediated_contacts,
)

ROOT = Path(__file__).resolve().parent.parent
PDB_3PTB = ROOT / "tests" / "data" / "3PTB.pdb"
DEMO_POSES = ROOT / "demo" / "3ptb" / "poses.pdbqt"


@dataclass
class Atom:
    """The subset of an atom-like object the analysis module reads."""

    name: str
    element: str
    res_name: str
    res_id: int
    x: float
    y: float
    z: float
    chain: str = "A"


def _key(label: str) -> FingerprintKey:
    residue, kind = label.split(":")
    return FingerprintKey(
        res_name=residue[:3], res_id=int(residue[3:]), chain="A", kind=kind
    )


# ---------------------------------------------------------------------------
# Keys and schemas
# ---------------------------------------------------------------------------


def test_fingerprint_key_labels():
    key = FingerprintKey(res_name="ASP", res_id=189, chain="A", kind="hbond")
    assert key.label == "ASP189:hbond"
    assert str(key) == "ASP189:hbond"
    assert FingerprintKey("", 7, "A", "clash").label == "#7:clash"


def test_schema_from_receptor_is_residues_times_kinds():
    receptor = [
        Atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        Atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
        Atom("CB", "C", "ALA", 55, 0.0, 0.0, 4.0),
    ]
    schema = FingerprintSchema.from_receptor(
        receptor, kinds=("hbond", "hydrophobic")
    )
    assert len(schema) == 4  # two residues x two kinds
    assert schema.column(_key("ASP189:hbond")) >= 0
    assert schema.column(_key("ALA55:hydrophobic")) >= 0
    assert schema.column(_key("ASP189:pi_pi")) == -1
    # deterministic order
    assert len(FingerprintSchema.from_receptor(receptor, kinds=("hbond",)).keys) == 2


def test_observed_schema_is_the_union_and_is_sorted():
    first = [Interaction("hbond", 0, 0, 2.8, residue=("ASP", 189, "A"))]
    second = [
        Interaction("hbond", 0, 0, 2.8, residue=("ASP", 189, "A")),
        Interaction("hydrophobic", 1, 1, 3.9, residue=("ALA", 55, "A")),
    ]
    schema = FingerprintSchema.observed([first, second])
    # Sorted by residue number, then interaction kind, so the schema -- and
    # therefore every fingerprint built from it -- is deterministic.
    assert [key.label for key in schema.keys] == ["ALA55:hydrophobic", "ASP189:hbond"]
    assert len(FingerprintSchema.observed([])) == 0


def test_observed_schema_drops_clashes_and_can_drop_waters():
    contacts = [
        Interaction("clash", 0, 0, 2.0, residue=("ASP", 189, "A")),
        Interaction("hbond", 0, 0, 2.8, residue=("ASP", 189, "A")),
        Interaction("water_bridge", 0, 0, 2.8, residue=("SER", 190, "A")),
    ]
    with_water = FingerprintSchema.observed(contacts)
    assert [key.label for key in with_water.keys] == ["ASP189:hbond", "SER190:water_bridge"]
    without = FingerprintSchema.observed(contacts, include_water=False)
    assert [key.label for key in without.keys] == ["ASP189:hbond"]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_interaction_fingerprint_counts_and_bits():
    schema = FingerprintSchema(
        keys=(_key("ASP189:hbond"), _key("ASP189:hydrophobic"), _key("ALA55:pi_pi"))
    )
    contacts = [
        Interaction("hbond", 0, 0, 2.8, residue=("ASP", 189, "A")),
        Interaction("hbond", 1, 0, 3.0, residue=("ASP", 189, "A")),
        Interaction("pi_pi", 2, 1, 3.8, residue=("ALA", 55, "A")),
        Interaction("clash", 3, 2, 2.0, residue=("ASP", 189, "A")),
    ]
    fp = interaction_fingerprint(contacts, schema)
    assert fp.counts.tolist() == [2.0, 0.0, 1.0]
    assert fp.bits.tolist() == [1, 0, 1]
    assert [key.label for key in fp.present()] == ["ASP189:hbond", "ALA55:pi_pi"]
    assert fp.to_dict() == {
        "labels": ["ASP189:hbond", "ALA55:pi_pi"],
        "counts": [2.0, 1.0],
        "bits": [1, 1],
    }
    assert len(fp.to_dict(include_empty=True)["labels"]) == 3
    assert len(fp) == 3


def test_interaction_fingerprint_rejects_a_mismatched_vector():
    schema = FingerprintSchema(keys=(_key("ASP189:hbond"),))
    with pytest.raises(ValueError):
        analysis.InteractionFingerprint(schema=schema, counts=np.zeros(2))


def test_interaction_fingerprint_encodes_one_pose_only():
    schema = FingerprintSchema(keys=(_key("ASP189:hbond"),))
    with pytest.raises(ValueError):
        interaction_fingerprint(
            [[Interaction("hbond", 0, 0, 1.0, residue=("ASP", 189, "A"))],
             [Interaction("hbond", 0, 0, 1.0, residue=("ASP", 189, "A"))]],
            schema,
        )


def test_an_interaction_without_a_residue_key_is_resolved_from_the_receptor():
    schema = FingerprintSchema(keys=(_key("ASP189:hbond"),))
    receptor = [
        Atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        Atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
    ]
    bare = [Interaction("hbond", 0, 0, 2.8)]  # no residue attached
    assert interaction_fingerprint(bare, schema).counts.tolist() == [0.0]
    resolved = interaction_fingerprint(bare, schema, receptor=receptor, ligand=None)
    assert resolved.counts.tolist() == [1.0]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _fingerprint(counts_by_label):
    keys = tuple(_key(label) for label in counts_by_label)
    schema = FingerprintSchema(keys=keys)
    return analysis.InteractionFingerprint(
        schema=schema, counts=np.array(list(counts_by_label.values()), dtype=float)
    )


def test_similarity_is_exact_on_hand_built_vectors():
    a = _fingerprint({"ASP189:hbond": 1, "ALA55:hydrophobic": 1, "GLY216:hbond": 0})
    b = _fingerprint({"ASP189:hbond": 1, "ALA55:hydrophobic": 0, "GLY216:hbond": 1})
    # bits: shared = 1 (ASP189), union = 3
    assert fingerprint_similarity(a, b) == pytest.approx(1 / 3)
    assert fingerprint_similarity(a, b, metric="dice") == pytest.approx(0.5)
    # counts [1,1,0] . [1,0,1] = 1, norms sqrt(2)*sqrt(2) = 2 -> 0.5
    assert fingerprint_similarity(a, b, metric="cosine") == pytest.approx(0.5)
    assert fingerprint_similarity(a, a) == pytest.approx(1.0)


def test_similarity_of_two_empty_fingerprints_is_undefined():
    empty_a = _fingerprint({"ASP189:hbond": 0})
    empty_b = _fingerprint({"ALA55:hbond": 0})
    assert math.isnan(fingerprint_similarity(empty_a, empty_b))
    assert math.isnan(fingerprint_similarity(empty_a, empty_b, metric="cosine"))


def test_similarity_rejects_unknown_metrics_and_shapes():
    a = _fingerprint({"ASP189:hbond": 1})
    b = _fingerprint({"ASP189:hbond": 1, "ALA55:hbond": 1})
    with pytest.raises(ValueError):
        fingerprint_similarity(a, a, metric="euclid")
    with pytest.raises(ValueError):
        fingerprint_similarity(a, b)


def test_similarity_matrix_is_symmetric_with_a_unit_diagonal():
    a = _fingerprint({"ASP189:hbond": 1, "ALA55:hydrophobic": 0})
    b = _fingerprint({"ASP189:hbond": 0, "ALA55:hydrophobic": 1})
    matrix = similarity_matrix([a, b, a])
    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 1.0 and matrix[2, 2] == 1.0
    assert matrix[0, 1] == matrix[1, 0] == pytest.approx(0.0)
    assert matrix[0, 2] == pytest.approx(1.0)
    # An empty fingerprint still has a unit diagonal.
    empty = _fingerprint({"ASP189:hbond": 0})
    empty_matrix = similarity_matrix([empty, empty])
    assert empty_matrix[0, 0] == 1.0
    assert math.isnan(empty_matrix[0, 1])


# ---------------------------------------------------------------------------
# Pharmacophore summary
# ---------------------------------------------------------------------------


def test_pharmacophore_summary_counts_recurrence_and_orders_by_it():
    schema = FingerprintSchema(
        keys=(_key("ASP189:hbond"), _key("ALA55:hydrophobic"), _key("GLY216:hbond"))
    )
    poses = [
        analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0, 1.0, 0.0])),
        analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0, 1.0, 0.0])),
        analysis.InteractionFingerprint(schema=schema, counts=np.array([0.0, 1.0, 1.0])),
        analysis.InteractionFingerprint(schema=schema, counts=np.array([0.0, 1.0, 0.0])),
    ]
    summary = pharmacophore_summary(poses)
    assert [key.label for key in summary.keys] == ["ALA55:hydrophobic", "ASP189:hbond"]
    assert summary.counts == [4, 2]
    assert summary.frequency == [1.0, 0.5]
    assert summary.n_poses == 4
    assert summary.interaction_types == {"hydrophobic": 1, "hbond": 1}
    assert "ALA55:hydrophobic" in summary.table()
    assert summary.as_dict()["features"][0]["poses"] == 4


def test_pharmacophore_summary_threshold_and_empty_cases():
    schema = FingerprintSchema(keys=(_key("ASP189:hbond"), _key("ALA55:hydrophobic")))
    # Four poses: each feature appears in one of them, i.e. a quarter.
    poses = [
        analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0, 0.0])),
        analysis.InteractionFingerprint(schema=schema, counts=np.array([0.0, 1.0])),
        analysis.InteractionFingerprint(schema=schema, counts=np.array([0.0, 0.0])),
        analysis.InteractionFingerprint(schema=schema, counts=np.array([0.0, 0.0])),
    ]
    assert pharmacophore_summary(poses).keys == []
    assert "no recurring interaction" in pharmacophore_summary(poses).table()
    assert pharmacophore_summary([]).n_poses == 0
    strict = pharmacophore_summary(poses, min_frequency=0.0)
    assert len(strict.keys) == 2
    assert strict.frequency == [0.25, 0.25]
    with pytest.raises(TypeError):
        pharmacophore_summary([np.zeros(2), np.zeros(2)])


# ---------------------------------------------------------------------------
# Synthetic geometry: water bridges
# ---------------------------------------------------------------------------


def _water_case(water_x: float):
    """A ligand N-H whose acceptor is an aspartate oxygen, plus one water.

    The ligand nitrogen at ``x = -2.8`` carries its hydrogen at ``x = -1.8`` and
    the receptor oxygen sits at ``x = +3.0``, so the D-H...A angle is exactly
    180 degrees and the ligand-water and water-receptor legs are 2.8 and 3.0 A.
    """
    receptor = [
        Atom("OD1", "O", "ASP", 189, 3.0, 0.0, 0.0),
        Atom("CG", "C", "ASP", 189, 3.0, 1.45, 0.0),
        Atom("O", "O", "HOH", 300, water_x, 0.0, 0.0),
    ]
    ligand = [
        Atom("N1", "N", "LIG", 1, -2.8, 0.0, 0.0),
        Atom("H1", "H", "LIG", 1, -1.8, 0.0, 0.0),
        Atom("C1", "C", "LIG", 1, -4.0, 0.5, 0.0),
    ]
    return receptor, ligand


def test_water_mediated_contact_is_detected_when_the_water_bridges():
    receptor, ligand = _water_case(water_x=0.0)
    bridges = water_mediated_contacts(receptor, ligand)
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge.kind == "water_bridge"
    assert bridge.detail == "ASP189:OD1...HOH300:O...LIG1:N1"
    assert bridge.distance == pytest.approx(2.8)
    assert bridge.residue == ("ASP", 189, "A")


def test_no_bridge_when_the_water_is_out_of_reach():
    receptor, ligand = _water_case(water_x=0.0)
    assert len(water_mediated_contacts(receptor, ligand, cutoff=2.5)) == 0
    receptor_far, ligand_far = _water_case(water_x=20.0)
    assert water_mediated_contacts(receptor_far, ligand_far) == []


def test_no_bridge_without_waters_or_without_a_polar_ligand():
    receptor, ligand = _water_case(water_x=0.0)
    dry = [atom for atom in receptor if atom.res_name != "HOH"]
    assert water_mediated_contacts(dry, ligand) == []
    # A pure hydrocarbon ligand has no donor and no acceptor.
    hydrocarbon = [
        Atom("C1", "C", "LIG", 1, -2.8, 0.0, 0.0),
        Atom("C2", "C", "LIG", 1, -4.0, 0.0, 0.0),
    ]
    assert water_mediated_contacts(receptor, hydrocarbon) == []
    with pytest.raises(ValueError):
        water_mediated_contacts(receptor, ligand, cutoff=0.0)


def test_water_bridges_are_shaped_like_an_interaction():
    receptor, ligand = _water_case(water_x=0.0)
    bridge = water_mediated_contacts(receptor, ligand)[0]
    schema = FingerprintSchema.observed([bridge])
    assert [key.label for key in schema.keys] == ["ASP189:water_bridge"]
    fp = interaction_fingerprint([bridge], schema)
    assert fp.counts.tolist() == [1.0]


# ---------------------------------------------------------------------------
# The whole pipeline on synthetic geometry
# ---------------------------------------------------------------------------


def _two_mode_case():
    """Two poses that make two *different* contacts with the same receptor.

    Pose 1 donates a hydrogen bond to ASP189:OD1 (D...A = 2.8 A, D-H...A = 180
    degrees).  Pose 2 makes a hydrophobic contact with ALA55:CB at 3.9 A.
    """
    receptor = [
        Atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        Atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
        Atom("CB", "C", "ALA", 55, 0.0, 6.0, 0.0),
    ]
    hbond_pose = [
        Atom("N1", "N", "LIG", 1, 0.0, 0.0, 0.0),
        Atom("H1", "H", "LIG", 1, 1.0, 0.0, 0.0),
        Atom("C1", "C", "LIG", 1, -1.2, 0.4, 0.0),
    ]
    hydrophobic_pose = [
        Atom("C1", "C", "LIG", 1, 0.0, 6.0, 3.9),
        Atom("C2", "C", "LIG", 1, 0.0, 6.0, 5.4),
    ]
    return receptor, hbond_pose, hydrophobic_pose


def test_pose_fingerprints_encodes_a_set_against_one_schema():
    receptor, hbond_pose, hydrophobic_pose = _two_mode_case()
    result = pose_fingerprints(receptor, [hbond_pose, hydrophobic_pose],
                               include_water=False)
    assert len(result) == 2
    assert result.labels() == ["ALA55:hydrophobic", "ASP189:hbond"]
    assert result.matrix.shape == (2, len(result.schema))
    assert result.matrix[0].tolist() == [0.0, 1.0]
    assert result.matrix[1].tolist() == [1.0, 0.0]
    assert result.bits.tolist() == [[0, 1], [1, 0]]
    # Nothing shared -> Tanimoto 0, and the matrix stays symmetric.
    assert result.similarity()[0, 1] == pytest.approx(0.0)
    assert result.pharmacophore(min_frequency=0.0).n_poses == 2
    shared = pose_fingerprints(receptor, [hbond_pose, hbond_pose], include_water=False)
    assert shared.similarity()[0, 1] == pytest.approx(1.0)


def test_pose_fingerprints_excludes_waters_from_the_direct_profile():
    """A water next to the ligand is a bridge, never a "HOH300" residue."""
    receptor, ligand = _water_case(water_x=0.0)
    with_water = pose_fingerprints(receptor, [ligand], include_water=True)
    labels = with_water.labels()
    assert not any(label.startswith("HOH") for label in labels)
    assert "ASP189:water_bridge" in labels
    assert len(with_water.water_bridges[0]) == 1
    # The direct ligand...water hydrogen bond the profiler would otherwise
    # report has been filtered out of the contacts entirely.
    assert all(item.a not in (2,) for item in with_water.interactions[0])
    without = pose_fingerprints(receptor, [ligand], include_water=False)
    assert labels != without.labels()
    assert without.water_bridges[0] == []


def test_pose_fingerprints_accepts_an_explicit_schema():
    receptor, ligand = _water_case(water_x=0.0)
    schema = FingerprintSchema.from_receptor(receptor, kinds=("hbond", "hydrophobic"))
    result = pose_fingerprints(receptor, [ligand], schema=schema, include_water=False)
    assert result.schema is schema
    assert result.matrix.shape == (1, len(schema))


def test_pose_fingerprints_never_records_a_clash():
    receptor = [
        Atom("CB", "C", "ALA", 1, 0.0, 0.0, 0.0),
        Atom("CB", "C", "ALA", 2, 10.0, 0.0, 0.0),
    ]
    overlapping = [
        Atom("C1", "C", "LIG", 1, 0.1, 0.0, 0.0),
        Atom("C2", "C", "LIG", 1, 1.3, 0.0, 0.0),
    ]
    contacts = profile_interactions(receptor, overlapping)
    assert any(item.kind == "clash" for item in contacts)
    result = pose_fingerprints(receptor, [overlapping], include_water=False)
    assert not any(key.kind == "clash" for key in result.schema.keys)


# ---------------------------------------------------------------------------
# Against the bundled 3PTB data
# ---------------------------------------------------------------------------


has_demo = pytest.mark.skipif(
    not (PDB_3PTB.exists() and DEMO_POSES.exists()),
    reason="the bundled 3PTB fixtures are not present",
)


def _3ptb_receptor_with_waters():
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(
        str(PDB_3PTB), removeHs=False, sanitize=False, proximityBonding=True
    )
    keep = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip().upper() if info else ""
        if name == "BEN":  # the co-crystallised ligand must not be in the receptor
            continue
        keep.append(atom.GetIdx())
    editable = Chem.RWMol(mol)
    for index in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
        editable.RemoveAtom(index)
    return editable.GetMol()


@has_demo
def test_demo_pose_fingerprints_against_the_real_receptor():
    receptor = _3ptb_receptor_with_waters()
    models = consensus.pdbqt_models(DEMO_POSES.read_text(encoding="utf-8"))
    ligands = [consensus.pdbqt_atoms(model) for model in models]
    result = pose_fingerprints(receptor, ligands, include_water=True)

    assert len(result) == 6
    assert result.matrix.shape[0] == 6
    assert len(result.schema) > 0

    # Poses 1 and 2 are the same binding mode (they differ by 0.05 A), so their
    # fingerprints must be identical.
    similarity = result.similarity()
    assert similarity[0, 1] == pytest.approx(1.0)
    assert np.allclose(similarity, similarity.T, equal_nan=True)
    assert all(similarity[i, i] == 1.0 for i in range(6))

    # The reproducible part of the binding mode across all six poses.
    summary = result.pharmacophore(min_frequency=0.5)
    assert summary.n_poses == 6
    labels = [key.label for key in summary.keys]
    assert "VAL213:hydrophobic" in labels
    assert summary.frequency == sorted(summary.frequency, reverse=True)
    assert any(key.kind == "water_bridge" for key in
               analysis.pharmacophore_summary(result.fingerprints, min_frequency=0.0).keys)

    # At least one pose is water-mediated: 3PTB keeps 62 crystallographic waters.
    assert any(len(bridges) > 0 for bridges in result.water_bridges)
    for bridges in result.water_bridges:
        for bridge in bridges:
            assert bridge.kind == "water_bridge"
            assert bridge.residue is not None
            assert "...HOH" in bridge.detail


@has_demo
def test_a_receptor_schema_lets_two_ligand_sets_share_one_vector_space():
    receptor = _3ptb_receptor_with_waters()
    models = consensus.pdbqt_models(DEMO_POSES.read_text(encoding="utf-8"))
    ligands = [consensus.pdbqt_atoms(model) for model in models]
    schema = FingerprintSchema.from_receptor(receptor, kinds=("hbond", "hydrophobic"))
    first = pose_fingerprints(receptor, ligands[:1], schema=schema, include_water=False)
    second = pose_fingerprints(receptor, ligands[1:2], schema=schema, include_water=False)
    assert first.schema is second.schema
    assert first.similarity().shape == (1, 1)
    # Identical binding modes must still be identical in the receptor-wide space.
    assert fingerprint_similarity(first.fingerprints[0], second.fingerprints[0]) == pytest.approx(1.0)
    assert len(first.schema) >= 2


# ---------------------------------------------------------------------------
# PDBQT text and files as input
# ---------------------------------------------------------------------------


def test_the_analyser_accepts_pdbqt_text_wherever_it_accepts_atoms():
    """A caller with a PDBQT on disk must not have to parse it first."""
    receptor_text = (
        "ATOM      1  OD1 ASP A 189       2.800   0.000   0.000  1.00  0.00    -0.500 OA\n"
        "ATOM      2  CG  ASP A 189       2.800   1.450   0.000  1.00  0.00     0.400 C\n"
        "TER\n"
    )
    ligand_text = (
        "ROOT\n"
        "ATOM      1  N1  LIG A   1       0.000   0.000   0.000  1.00  0.00    -0.300 N\n"
        "ATOM      2  H1  LIG A   1       1.000   0.000   0.000  1.00  0.00     0.200 HD\n"
        "ATOM      3  C1  LIG A   1      -1.200   0.400   0.000  1.00  0.00     0.100 C\n"
        "ENDROOT\nTORSDOF 0\n"
    )
    contacts = profile_interactions(receptor_text, ligand_text)
    assert [item.kind for item in contacts] == ["hbond"]
    assert contacts[0].residue == ("ASP", 189, "A")
    assert contacts[0].detail == "LIG1:N1->ASP189:OD1"  # the ligand is the donor

    result = pose_fingerprints(receptor_text, [ligand_text], include_water=False)
    assert result.labels() == ["ASP189:hbond"]
    assert result.matrix.tolist() == [[1.0]]

    schema = FingerprintSchema.from_receptor(receptor_text, kinds=("hbond",))
    assert [key.label for key in schema.keys] == ["ASP189:hbond"]


def test_a_pdbqt_path_is_read_from_disk(tmp_path):
    path = tmp_path / "receptor.pdbqt"
    path.write_text(
        "ATOM      1  OD1 ASP A 189       2.800   0.000   0.000  1.00  0.00    -0.500 OA\n"
        "ATOM      2  CG  ASP A 189       2.800   1.450   0.000  1.00  0.00     0.400 C\n",
        encoding="utf-8",
    )
    structure = analysis._structure(path)
    assert len(structure.atoms) == 2
    assert structure.atoms[0].res_name == "ASP"


def test_unparsable_text_is_rejected_with_a_clear_error():
    with pytest.raises(ValueError):
        analysis._structure("not a structure, and not a file that exists")
    with pytest.raises(ValueError):
        analysis._structure("REMARK this file has no atoms at all\n")

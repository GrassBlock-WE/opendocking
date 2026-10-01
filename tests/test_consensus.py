# SPDX-License-Identifier: GPL-3.0-or-later
"""Consensus scoring: rank aggregation, Spearman agreement, and rescoring.

The rank/correlation tests use hand-computed values (a four-element list has
ranks 1..4; ``rho = 1 - 6*sum(d^2)/(n(n^2-1))`` is exact for untied data).  The
kernel tests rescore the bundled 3PTB demo poses and anchor the result on the
``VINA RESULT`` values the run itself wrote into the file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from odock import consensus
from odock.consensus import (
    ConsensusPose,
    ConsensusResult,
    average_ranks,
    box_from_any,
    consensus_score,
    normalised_ranks,
    pdbqt_atoms,
    pdbqt_models,
    pose_pdbqt_from_coords,
    rescore_poses,
    resolve_poses,
    spearman,
    with_coordinates,
    z_scores,
)
from odock.docking import DockResult, Pose
from odock.prepare import BoxSpec

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "3ptb"
DEMO_RECEPTOR = DEMO / "receptor.pdbqt"
DEMO_LIGAND = DEMO / "ligand.pdbqt"
DEMO_POSES = DEMO / "poses.pdbqt"
DEMO_BOX = DEMO / "box.json"

has_demo = pytest.mark.skipif(
    not (DEMO_RECEPTOR.exists() and DEMO_POSES.exists() and DEMO_BOX.exists()),
    reason="the bundled 3PTB demo is not present",
)

SINGLE_LIGAND = """\
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.500   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
BRANCH    2   3
ATOM      3  O1  LIG A   1       2.500   0.000   0.000  1.00  0.00    -0.300 OA
ENDBRANCH    2   3
TORSDOF 1
"""

SMALL_RECEPTOR = """\
ATOM      1  CB  ALA A   1      -4.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  CB  ALA A   2       4.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      3  OG  SER A   3       0.000   4.000   0.000  1.00  0.00    -0.300 OA
TER
"""

SMALL_BOX = BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=0.5)


# ---------------------------------------------------------------------------
# Rank primitives
# ---------------------------------------------------------------------------


def test_average_ranks_are_the_competition_ranks():
    assert average_ranks([10, 20, 30, 40]).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert average_ranks([40, 30, 20, 10]).tolist() == [4.0, 3.0, 2.0, 1.0]
    # Ties share the average of the ranks they occupy.
    assert average_ranks([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert average_ranks([5, 5, 5]).tolist() == [2.0, 2.0, 2.0]


def test_average_ranks_ignore_non_finite_values():
    ranks = average_ranks([3.0, float("nan"), 1.0, 2.0])
    assert math.isnan(ranks[1])
    assert ranks[2] == 1.0 and ranks[3] == 2.0 and ranks[0] == 3.0


def test_normalised_ranks_span_zero_to_one():
    assert normalised_ranks([10, 20, 30, 40]).tolist() == [0.0, pytest.approx(1 / 3),
                                                          pytest.approx(2 / 3), 1.0]
    # Ties keep the shared position.
    assert normalised_ranks([10, 20, 20, 30]).tolist() == [0.0, 0.5, 0.5, 1.0]
    # A single usable value is trivially the best.
    assert normalised_ranks([7.0]).tolist() == [0.0]
    # Values with no usable entry stay nan rather than becoming a rank.
    empty = normalised_ranks([float("nan"), float("nan")]).tolist()
    assert all(math.isnan(value) for value in empty)


def test_z_scores_are_population_standard_scores():
    # mean 2, population sd = sqrt(2/3); (1-2)/sd = -1.224744871391589
    analytic = -1.0 / math.sqrt(2.0 / 3.0)
    assert z_scores([1, 2, 3]).tolist() == pytest.approx([analytic, 0.0, -analytic])
    assert z_scores([5, 5, 5]).tolist() == [0.0, 0.0, 0.0]
    assert math.isnan(z_scores([1.0, float("nan"), 3.0])[1])


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def test_spearman_is_exact_for_untied_lists():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # sum(d^2) = 2 -> rho = 1 - 6*2/(4*15) = 0.8
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    # Only the ordering matters, not the units: this list is increasing.
    assert spearman([1.0, 2.0, 3.0], [-4.0, 12.5, 1e6]) == pytest.approx(1.0)
    assert spearman([1.0, 2.0, 3.0], [1e6, -4.0, 12.5]) == pytest.approx(-0.5)


def test_spearman_handles_ties_by_average_ranks():
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1.0)


def test_spearman_of_a_constant_list_is_undefined():
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([1.0], [2.0]))
    assert math.isnan(spearman([1.0, float("nan")], [2.0, 3.0]))


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        spearman([1, 2, 3], [1, 2])


# ---------------------------------------------------------------------------
# PDBQT plumbing
# ---------------------------------------------------------------------------


def test_pdbqt_atoms_reads_the_documented_columns():
    atoms = pdbqt_atoms(SINGLE_LIGAND)
    assert len(atoms) == 3
    assert [a.name for a in atoms] == ["C1", "C2", "O1"]
    assert [a.element for a in atoms] == ["C", "C", "O"]
    assert [a.ad_type for a in atoms] == ["C", "C", "OA"]
    assert [a.is_heavy for a in atoms] == [True, True, True]
    assert atoms[2].charge == pytest.approx(-0.300)
    assert atoms[2].x == pytest.approx(2.5)
    assert atoms[0].label == "LIG1:C1"


def test_pdbqt_atoms_flags_hydrogens_and_closure_dummies_as_light():
    text = (
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ATOM      2  H1  LIG A   1       1.090   0.000   0.000  1.00  0.00     0.000 HD\n"
        "ATOM      3  G0  LIG A   1       2.000   0.000   0.000  1.00  0.00     0.000 G0\n"
    )
    assert [a.is_heavy for a in pdbqt_atoms(text)] == [True, False, False]


def test_pdbqt_models_splits_and_keeps_a_single_document_whole():
    text = "MODEL 1\nATOM\nENDMDL\nMODEL 2\nATOM\nENDMDL\n"
    models = pdbqt_models(text)
    assert len(models) == 2
    assert models[0].startswith("MODEL 1") and models[1].startswith("MODEL 2")
    assert pdbqt_models(SINGLE_LIGAND) == [SINGLE_LIGAND]
    assert pdbqt_models("   \n") == []


def test_with_coordinates_replaces_exactly_the_coordinates():
    moved = with_coordinates(SINGLE_LIGAND, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    atoms = pdbqt_atoms(moved)
    assert [(a.x, a.y, a.z) for a in atoms] == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    # Everything else is byte-identical outside the coordinate columns.
    assert moved.splitlines()[0] == SINGLE_LIGAND.splitlines()[0]
    assert moved.splitlines()[-1] == "TORSDOF 1"
    rebuilt = pdbqt_atoms(moved)
    original = pdbqt_atoms(SINGLE_LIGAND)
    assert [a.name for a in rebuilt] == [a.name for a in original]
    assert [a.charge for a in rebuilt] == [a.charge for a in original]


def test_with_coordinates_refuses_a_partial_patch():
    with pytest.raises(ValueError):
        with_coordinates(SINGLE_LIGAND, [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        with_coordinates(SINGLE_LIGAND, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        with_coordinates(SINGLE_LIGAND, np.zeros((3, 2)))


def test_pose_pdbqt_from_coords_round_trips_a_pose():
    coords = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    text = pose_pdbqt_from_coords(SINGLE_LIGAND, coords)
    assert [(a.x, a.y, a.z) for a in pdbqt_atoms(text)] == [
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        (3.0, 3.0, 3.0),
    ]
    # A pose whose atom count disagrees with the template is skipped, not
    # half-written.
    assert pose_pdbqt_from_coords(SINGLE_LIGAND, np.zeros((2, 3))) == ""
    assert pose_pdbqt_from_coords("", coords) == ""


def test_box_from_any_accepts_every_documented_spelling(tmp_path):
    reference = BoxSpec(center=(1.0, 2.0, 3.0), size=(4.0, 5.0, 6.0), spacing=0.375)
    assert box_from_any(reference) is reference
    assert box_from_any(
        {"center": [1, 2, 3], "size": [4, 5, 6], "spacing": 0.375}
    ) == reference
    assert box_from_any(
        {"center_x": 1, "center_y": 2, "center_z": 3, "size_x": 4, "size_y": 5,
         "size_z": 6, "spacing": 0.375}
    ) == reference
    assert box_from_any(((1, 2, 3), (4, 5, 6), 0.375)) == reference
    assert box_from_any(((1, 2, 3), (4, 5, 6))) == reference
    path = tmp_path / "box.json"
    path.write_text(json.dumps({"center": [1, 2, 3], "size": [4, 5, 6], "spacing": 0.375}))
    assert box_from_any(path) == reference
    with pytest.raises(TypeError):
        box_from_any(42)


def test_box_from_any_default_is_a_box_nothing_escapes():
    box = box_from_any(None)
    assert box.size == (200.0, 200.0, 200.0)


def test_resolve_poses_accepts_text_a_path_a_list_and_a_result(tmp_path):
    single = resolve_poses(SINGLE_LIGAND)
    assert len(single) == 1
    path = tmp_path / "one.pdbqt"
    path.write_text(SINGLE_LIGAND, encoding="utf-8")
    assert resolve_poses(path) == [SINGLE_LIGAND]
    assert len(resolve_poses([SINGLE_LIGAND, SINGLE_LIGAND])) == 2

    coords = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    result = DockResult(
        poses=[Pose(index=0, affinity=-1.0, coords=coords)],
        seed=0,
        ligand_pdbqt=SINGLE_LIGAND,
        _pdbqt="MODEL 1\n" + SINGLE_LIGAND + "ENDMDL\n",
    )
    models = resolve_poses(result)
    assert len(models) == 1
    assert pdbqt_atoms(models[0])[0].x == pytest.approx(1.0)
    # A list of Pose objects needs the template.
    assert len(resolve_poses([result.poses[0]], ligand_template=SINGLE_LIGAND)) == 1
    with pytest.raises(ValueError):
        resolve_poses([result.poses[0]])


# ---------------------------------------------------------------------------
# Rank aggregation, with injected scores so the arithmetic is exact
# ---------------------------------------------------------------------------


def _components(scores):
    """A ``rescore_poses``-shaped payload from raw affinity lists."""
    return {
        name: {
            "affinity": list(values),
            "inter": list(values),
            "intra": [0.0] * len(values),
            "unbound": [0.0] * len(values),
            "conf_independent": [0.0] * len(values),
            "total": list(values),
        }
        for name, values in scores.items()
    }


def test_consensus_rank_sum_is_exact():
    """Three poses, two fields, fully hand-computed.

    vina    = [-1, -2, -3]  best is index 2 -> ranks [3, 2, 1] -> norm [1.0, 0.5, 0.0]
    vinardo = [-3, -1, -2]  best is index 0 -> ranks [1, 3, 2] -> norm [0.0, 1.0, 0.5]
    consensus = mean -> [0.5, 0.75, 0.25] -> order 2, 0, 1
    """
    scores = {"vina": [-1.0, -2.0, -3.0], "vinardo": [-3.0, -1.0, -2.0]}
    result = consensus_score(
        None, None, components=_components(scores), scorings=("vina", "vinardo")
    )
    assert [p.index for p in result.poses] == [2, 0, 1]
    assert [p.consensus_rank for p in result.poses] == [1, 2, 3]
    by_index = {p.index: p for p in result.poses}
    assert by_index[0].consensus_score == pytest.approx(0.5)
    assert by_index[1].consensus_score == pytest.approx(0.75)
    assert by_index[2].consensus_score == pytest.approx(0.25)
    assert by_index[0].ranks == {"vina": 3.0, "vinardo": 1.0}
    assert by_index[0].scores == {"vina": -1.0, "vinardo": -3.0}
    # vina ranks [3, 2, 1], vinardo ranks [1, 3, 2]: sum(d^2) = 6,
    # rho = 1 - 6*6/(3*8) = -0.5
    assert result.correlation("vina", "vinardo") == pytest.approx(-0.5)
    assert result.agreement == pytest.approx(-0.5)


def test_consensus_weights_change_the_outcome():
    """A weight of 1 on vina and 0 on vinardo restores vina's own ordering."""
    scores = {"vina": [-1.0, -2.0, -3.0], "vinardo": [-3.0, -1.0, -2.0]}
    result = consensus_score(
        None,
        None,
        components=_components(scores),
        scorings=("vina", "vinardo"),
        weights={"vina": 1.0, "vinardo": 0.0},
    )
    assert [p.index for p in result.poses] == [2, 1, 0]
    assert result.weights == {"vina": 1.0, "vinardo": 0.0}
    by_index = {p.index: p for p in result.poses}
    assert by_index[0].consensus_score == pytest.approx(1.0)
    assert by_index[2].consensus_score == pytest.approx(0.0)


def test_borda_and_z_agree_with_rank_on_a_complete_untied_set():
    scores = {"vina": [-1.0, -2.0, -3.0], "vinardo": [-3.0, -1.0, -2.0]}
    rank = consensus_score(None, None, components=_components(scores),
                           scorings=("vina", "vinardo"), method="rank")
    borda = consensus_score(None, None, components=_components(scores),
                            scorings=("vina", "vinardo"), method="borda")
    assert [p.index for p in rank.poses] == [p.index for p in borda.poses]
    # Borda keeps the raw 1..n scale: means of [3,2,1] and [1,3,2] are
    # [2.0, 2.5, 1.5].
    assert sorted(p.consensus_score for p in borda.poses) == [1.5, 2.0, 2.5]


def test_z_method_is_scale_free_and_exactly_computable():
    """Multiplying a field by 10 must not change a z consensus at all."""
    agreed = {"a": [0.0, 1.0, 2.0], "b": [0.0, 10.0, 20.0]}
    result = consensus_score(None, None, components=_components(agreed),
                             scorings=("a", "b"), method="z")
    assert [p.index for p in result.poses] == [0, 1, 2]
    by_index = {p.index: p for p in result.poses}
    # mean 1, population sd sqrt(2/3); the two fields have identical z scores.
    analytic = -1.0 / math.sqrt(2.0 / 3.0)
    assert by_index[0].consensus_score == pytest.approx(analytic)
    assert by_index[1].consensus_score == pytest.approx(0.0)
    assert by_index[2].consensus_score == pytest.approx(-analytic)


def test_z_consensus_cancels_two_opposing_fields_exactly():
    """Two fields that disagree pose-for-pose cancel; no aggregation can help.

    'wide' and 'narrow' are the same ordering with a 1000-fold difference in
    spread, so z-scoring makes them contribute equally and *oppositely*: every
    pose ends up at zero.  That is the honest answer -- the consensus has no
    reason to prefer either field -- and it is the reason the Spearman
    correlation is reported next to the consensus.
    """
    scores = {"wide": [-10.0, -5.0, 0.0], "narrow": [-5.0, -5.01, -5.02]}
    result = consensus_score(None, None, components=_components(scores),
                             scorings=("wide", "narrow"), method="z")
    values = [p.consensus_score for p in result.poses]
    assert max(values) - min(values) < 1e-9
    assert all(abs(value) < 1e-9 for value in values)
    assert [p.consensus_rank for p in result.poses] == [1, 2, 3]
    # And the rank method agrees that there is no winner: every mean is 0.5.
    ranked = consensus_score(None, None, components=_components(scores),
                             scorings=("wide", "narrow"), method="rank")
    assert all(p.consensus_score == pytest.approx(0.5) for p in ranked.poses)
    assert ranked.correlation("wide", "narrow") == pytest.approx(-1.0)


def test_a_missing_score_is_the_worst_badness_and_is_named():
    scores = {
        "vina": [-1.0, -2.0, -3.0],
        "vinardo": [-1.0, float("nan"), -2.0],
    }
    result = consensus_score(None, None, components=_components(scores),
                             scorings=("vina", "vinardo"))
    by_index = {p.index: p for p in result.poses}
    assert by_index[1].missing == ("vinardo",)
    assert math.isnan(by_index[1].scores["vinardo"])
    assert by_index[1].badness["vinardo"] == 1.0
    # vina normalised ranks: [1.0, 0.5, 0.0].
    # vinardo has two usable values, so its ranks are 2 and 1 -> [1.0, nan, 0.0],
    # and the missing pose is charged the worst badness, 1.0.
    # means: [1.0, 0.75, 0.0]
    assert by_index[0].consensus_score == pytest.approx(1.0)
    assert by_index[1].consensus_score == pytest.approx(0.75)
    assert by_index[2].consensus_score == pytest.approx(0.0)
    assert [p.index for p in result.poses] == [2, 1, 0]


def test_consensus_rejects_bad_configuration():
    scores = {"vina": [1.0, 2.0], "vinardo": [1.0, 2.0]}
    with pytest.raises(ValueError):
        consensus_score(None, None, components=_components(scores), scorings=())
    with pytest.raises(ValueError):
        consensus_score(None, None, components=_components(scores),
                        scorings=("vina",), method="median")
    with pytest.raises(ValueError):
        consensus_score(None, None, components=_components(scores),
                        scorings=("vina", "vinardo"), weights=[1.0])
    with pytest.raises(ValueError):
        consensus_score(None, None, components=_components(scores),
                        scorings=("vina", "vinardo"), weights=[0.0, 0.0])
    with pytest.raises(ValueError):
        consensus_score(None, None,
                        components={"vina": {"affinity": [1.0]},
                                    "vinardo": {"affinity": [1.0, 2.0]}},
                        scorings=("vina", "vinardo"))


def test_consensus_result_matrix_and_rows():
    scores = {"vina": [-1.0, -2.0, -3.0], "vinardo": [-3.0, -1.0, -2.0]}
    result = consensus_score(None, None, components=_components(scores),
                             scorings=("vina", "vinardo"))
    matrix = result.matrix()
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0 and matrix[1, 1] == 1.0
    assert matrix[0, 1] == matrix[1, 0] == pytest.approx(-0.5)
    assert math.isnan(ConsensusResult(scorings=("vina",)).agreement)
    rows = result.rows()
    assert rows[0]["mode"] == 3  # pose index 2 is the best
    assert rows[0]["consensus_rank"] == 1
    assert rows[0]["vina_rank"] == 1.0
    assert "vina" in json.loads(json.dumps(result.as_dict()))["poses"][0]["scores"]


def test_consensus_table_is_a_text_table_with_a_row_per_pose():
    scores = {"vina": [-1.0, -2.0], "vinardo": [-2.0, -1.0]}
    result = consensus_score(None, None, components=_components(scores),
                             scorings=("vina", "vinardo"))
    table = result.table()
    lines = table.splitlines()
    assert len(lines) == 4  # header, rule, two poses
    assert "vina (rank)" in lines[0]
    # Pose 1 is best under vinardo (rank 1) and worst under vina (rank 2).
    assert "-1.000 (2.0)" in lines[2]
    assert "-2.000 (1.0)" in lines[2]


def test_consensus_pose_row_exposes_a_prefix():
    pose = ConsensusPose(index=4, scores={"vina": -6.0}, ranks={"vina": 1.0})
    row = pose.row(prefix="cs_")
    assert row["cs_mode"] == 5
    assert row["cs_vina"] == -6.0


# ---------------------------------------------------------------------------
# Against the real kernel and the bundled demo
# ---------------------------------------------------------------------------


def test_rescore_reproduces_a_plain_kernel_score():
    """The rescoring path must agree with `odock.score` to the last bit."""
    import odock

    raw = rescore_poses(SINGLE_LIGAND, SMALL_RECEPTOR, SMALL_BOX, ["vina"])
    direct = odock.score(SMALL_RECEPTOR, SINGLE_LIGAND, SMALL_BOX,
                         scoring="vina", refine=False)
    assert raw["vina"]["affinity"][0] == pytest.approx(direct["affinity"], rel=1e-15)
    assert raw["vina"]["inter"][0] == pytest.approx(direct["inter"], rel=1e-15)
    assert raw["vina"]["intra"][0] == pytest.approx(direct["intra"], rel=1e-15)


def test_rescore_returns_one_entry_per_pose_and_field():
    raw = rescore_poses(
        [SINGLE_LIGAND, with_coordinates(SINGLE_LIGAND, [[5.0, 0, 0], [6.5, 0, 0], [7.5, 0, 0]])],
        SMALL_RECEPTOR,
        SMALL_BOX,
        ["vina", "vinardo"],
    )
    assert set(raw) == {"vina", "vinardo"}
    for field in raw.values():
        assert len(field["affinity"]) == 2
        assert len(field["inter"]) == 2
        assert set(field) >= {"affinity", "inter", "intra", "unbound", "conf_independent"}


@has_demo
def test_rescoring_reproduces_the_stored_vina_results():
    """The strongest anchor available: the numbers the run itself wrote down."""
    models = pdbqt_models(DEMO_POSES.read_text(encoding="utf-8"))
    assert len(models) == 6
    stored = []
    for model in models:
        for line in model.splitlines():
            if line.startswith("REMARK VINA RESULT:"):
                stored.append(float(line.split()[3]))
                break
    assert len(stored) == 6

    raw = rescore_poses(models, DEMO_RECEPTOR, DEMO_BOX, ["vina"])
    deviations = []
    for got, expected in zip(raw["vina"]["affinity"], stored):
        # The stored value is the affinity of the *unrounded* pose, while the
        # PDBQT keeps three coordinate decimals; re-scoring the rounded geometry
        # moves the energy by a few millikcal/mol.  Measured on this data the
        # largest deviation is 0.0031 kcal/mol (-4.900 stored vs -4.8969
        # recomputed), so 5e-3 is the floor of the comparison -- and the other
        # five poses land within 1.1e-3 of the number the run wrote down.
        assert got == pytest.approx(expected, abs=5e-3)
        deviations.append(abs(got - expected))
    assert max(deviations) < 5e-3
    assert sum(1 for d in deviations if d < 2e-3) >= 4


@has_demo
def test_demo_consensus_finds_the_crystal_like_pose_first():
    result = consensus_score(DEMO_POSES, DEMO_RECEPTOR, DEMO_BOX)
    assert result.scorings == ("vina", "vinardo", "ad4")
    assert len(result.poses) == 6
    assert result.poses[0].index == 0
    assert result.poses[0].consensus_rank == 1
    # vina and ad4 both put pose 1 first; vinardo prefers pose 2.  That is the
    # disagreement the consensus exists to expose.
    first = result.poses[0]
    assert first.ranks["vina"] == 1.0
    assert first.ranks["ad4"] == 1.0
    assert first.ranks["vinardo"] == 2.0
    assert 0.9 <= result.agreement <= 1.0
    assert not any(pose.missing for pose in result.poses)


@has_demo
def test_demo_consensus_scores_are_monotone_and_ordered():
    result = consensus_score(DEMO_POSES, DEMO_RECEPTOR, DEMO_BOX)
    scores = [pose.consensus_score for pose in result.poses]
    assert scores == sorted(scores)
    assert scores[0] == pytest.approx(1 / 15, abs=1e-9)  # (0 + 0.5 + 0)/3
    assert scores[-1] == pytest.approx(1.0)


@has_demo
def test_demo_box_json_and_boxspec_agree():
    box = box_from_any(DEMO_BOX)
    assert box.center == pytest.approx((-1.8555, 14.366, 16.748))
    assert box.size == pytest.approx((17.883, 19.95, 20.514))
    assert box.spacing == pytest.approx(0.375)

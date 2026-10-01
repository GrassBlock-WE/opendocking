# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.endpoint`: the end-point estimate and its error bar.

The claims that matter are arithmetic and are pinned as such:

* the Coulomb term is closed-form -- two unit charges 4 A apart at ``eps = 4r`` give
  ``332.0637 / (4 * 16) = -5.188`` kcal/mol, so the constant and the convention are
  checkable by hand;
* the term identity holds: ``total == interaction + electrostatic + nonpolar``
  (plus the optional entropy and strain), so no term can be silently dropped;
* the interaction energy equals **the number the consensus layer reports for the
  same pose**, which is what stops this module drifting away from the docking
  pipeline;
* the interval behaves: one pose has no ensemble to resample and says so, several
  poses give a deterministic seeded interval that contains the mean;
* the ranking comparison refuses to resolve fewer than three ligands, returns
  Kendall's tau with its interval, and -- the point of the module -- says whether
  the two orderings are distinguishable at that n.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from odock import endpoint
from odock.consensus import pdbqt_atoms, pdbqt_models
from odock.prepare import BoxSpec

DATA = Path(__file__).resolve().parent / "data"
BOX_3PTB = Path(__file__).resolve().parent.parent / "demo" / "systems" / "3ptb" / "box.json"

#: Every test that lands a real pose computes solvent-accessible areas over ~2000
#: receptor atoms, so the file takes minutes rather than seconds.
pytestmark = pytest.mark.slow


class Point:
    """A minimal atom: position and charge, which is all the terms read."""

    def __init__(self, x, y, z, charge):
        self.x, self.y, self.z, self.charge = float(x), float(y), float(z), float(charge)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_the_coulomb_term_is_closed_form():
    """Two unit charges at 4 A with eps = 4r: 332.0637 / (4 * 4^2) = 5.188 kcal/mol."""
    left = [Point(0.0, 0.0, 0.0, +1.0)]
    right = [Point(4.0, 0.0, 0.0, -1.0)]
    value = endpoint.coulomb_energy(left, right, dielectric=4.0)
    expected = -endpoint.COULOMB / (4.0 * 16.0)
    assert value == pytest.approx(expected, rel=1e-12)
    assert value == pytest.approx(-5.188, abs=1e-3)
    # Like charges repel, and the magnitude falls as 1/r^2 because eps grows with r.
    assert endpoint.coulomb_energy(left, [Point(4.0, 0.0, 0.0, +1.0)]) > 0
    doubled = endpoint.coulomb_energy(left, [Point(8.0, 0.0, 0.0, -1.0)])
    assert doubled == pytest.approx(value / 4.0, rel=1e-12)
    # A different convention changes the answer, which is the point of naming it.
    assert endpoint.coulomb_energy(left, right, dielectric=1.0) == pytest.approx(
        value * 4.0, rel=1e-12
    )


def test_the_coulomb_term_is_zero_without_charges_or_atoms():
    assert endpoint.coulomb_energy([], []) == 0.0
    assert endpoint.coulomb_energy([Point(0, 0, 0, 0.0)], [Point(3, 0, 0, 0.0)]) == 0.0


def test_the_nonpolar_term_is_the_documented_coefficient():
    """dG_np = -gamma * (ligand buried + receptor buried), both from odock.sasa."""
    require_structure = DATA / "3PTB.pdb"
    if not require_structure.exists():
        pytest.skip("missing the bundled 3PTB structure")
    ligand = [Point(0.0, 0.0, 0.0, 0.0)]
    receptor = [Point(1.5, 0.0, 0.0, 0.0)]
    areas = endpoint.buried_area(ligand, receptor)
    assert set(areas) == {"ligand", "receptor", "total"}
    assert areas["total"] == pytest.approx(areas["ligand"] + areas["receptor"])
    assert areas["total"] > 0.0
    pose = endpoint.endpoint_score(
        label="x", pose_index=0, affinity=0.0, interaction=0.0,
        ligand_atoms=ligand, receptor_atoms=receptor,
    )
    assert pose.nonpolar == pytest.approx(-endpoint.DEFAULT_GAMMA * areas["total"], rel=1e-9)


def test_every_term_adds_up():
    ligand = [Point(0.0, 0.0, 0.0, 0.3)]
    receptor = [Point(2.0, 0.0, 0.0, -0.4)]
    pose = endpoint.endpoint_score(
        label="x", pose_index=0, affinity=-5.0, interaction=-4.0,
        ligand_atoms=ligand, receptor_atoms=receptor, n_torsions=2.0,
        with_entropy=True, strain=0.5,
    )
    assert pose.total == pytest.approx(
        pose.interaction + pose.electrostatic + pose.nonpolar + pose.entropy + pose.strain,
        rel=1e-12,
    )
    assert pose.strain == pytest.approx(0.5)
    assert pose.entropy != 0.0


def test_the_entropy_term_is_the_metrics_estimate():
    from odock import metrics

    ligand = [Point(0.0, 0.0, 0.0, 0.1)]
    receptor = [Point(3.0, 0.0, 0.0, -0.1)]
    pose = endpoint.endpoint_score(
        label="x", pose_index=0, affinity=0.0, interaction=0.0,
        ligand_atoms=ligand, receptor_atoms=receptor, n_torsions=4.0, with_entropy=True,
    )
    assert pose.entropy == pytest.approx(metrics.entropy_penalty(4.0), rel=1e-12)
    # Measured: 5 torsions cost 3.255 kcal/mol with this project's constants.
    assert metrics.entropy_penalty(5.0) == pytest.approx(3.255, abs=1e-3)


def test_torsions_are_read_from_the_model():
    assert endpoint._count_torsions("ROOT\nTORSDOF 5\nENDMDL\n") == 5.0
    assert endpoint._count_torsions("no torsion line") == 0.0


# ---------------------------------------------------------------------------
# The interval
# ---------------------------------------------------------------------------


def _result(totals):
    poses = [
        endpoint.EndpointPose(
            label="x", pose_index=index, affinity=-5.0, interaction=-5.0,
            electrostatic=0.0, nonpolar=0.0,
        )
        for index in range(len(totals))
    ]
    for pose, value in zip(poses, totals):
        pose.interaction = float(value)
    return endpoint.EndpointResult(ligand="x", receptor="r", poses=poses)


def test_one_pose_has_no_ensemble_and_says_so():
    interval = _result([-3.0]).bootstrap()
    assert interval["n"] == 1
    assert interval["low"] == interval["high"] == interval["mean"] == -3.0
    assert interval["samples"] == 0
    assert "no ensemble" in interval["note"]


def test_the_interval_is_seeded_and_contains_the_mean():
    result = _result([-3.0, -2.0, -2.5, -4.0, -1.5])
    first = result.bootstrap(samples=500, seed=7)
    second = result.bootstrap(samples=500, seed=7)
    assert first == second
    assert first["n"] == 5 and first["samples"] == 500
    assert first["low"] <= first["mean"] <= first["high"]
    assert first["mean"] == pytest.approx(-2.6)
    assert first["width"] > 0.0
    # A different seed gives a different interval of the same order.
    other = result.bootstrap(samples=500, seed=8)
    assert other["width"] == pytest.approx(first["width"], rel=0.5)


def test_no_poses_is_not_silently_zero():
    interval = _result([]).bootstrap()
    assert interval["n"] == 0
    assert math.isnan(interval["mean"])
    assert interval["note"] == "no poses"


# ---------------------------------------------------------------------------
# The ranking comparison
# ---------------------------------------------------------------------------


def test_kendall_tau_is_hand_checkable():
    assert endpoint.kendall_tau([3, 2, 1], [3, 2, 1]) == pytest.approx(1.0)
    assert endpoint.kendall_tau([3, 2, 1], [1, 2, 3]) == pytest.approx(-1.0)
    # One swapped adjacent pair out of six: (6 - 2)/6 = 0.667.
    assert endpoint.kendall_tau([3, 2, 1], [3, 1, 2]) == pytest.approx(1 / 3)
    assert math.isnan(endpoint.kendall_tau([1.0], [1.0]))


def test_two_ligands_cannot_be_ranked():
    results = [_result([-1.0]), _result([-2.0])]
    report = endpoint.compare_rankings(results, [-5.0, -6.0])
    assert report["n"] == 2
    assert report["distinguishable"] is None
    assert "not resolvable" in report["note"]


def test_identical_orderings_are_not_distinguishable():
    # Both channels favour the same ligand: the end-point total is most negative
    # for index 3 and the Vina affinity too (lower affinity is the better score).
    results = [_result([-1.0]), _result([-2.0]), _result([-3.0]), _result([-4.0])]
    report = endpoint.compare_rankings(results, [-1.0, -2.0, -3.0, -4.0], samples=200, seed=7)
    assert report["n"] == 4
    assert report["rho"] == pytest.approx(1.0)
    assert report["tau"] == pytest.approx(1.0)
    assert report["n_rank_changes"] == 0
    assert report["top1_agreement"] is True
    assert report["distinguishable"] is False
    assert "NOT distinguishable" in report["note"]


def test_a_reversed_ordering_is_distinguishable():
    results = [_result([-1.0]), _result([-2.0]), _result([-3.0]), _result([-4.0])]
    # The Vina channel now prefers the opposite end of the list.
    report = endpoint.compare_rankings(results, [-4.0, -3.0, -2.0, -1.0], samples=200, seed=7)
    assert report["rho"] == pytest.approx(-1.0)
    assert report["tau"] == pytest.approx(-1.0)
    assert report["top1_agreement"] is False
    assert report["distinguishable"] is True


# ---------------------------------------------------------------------------
# The real pose, and the equality with the pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def docked(tmp_path_factory):
    """Dock benzamidine into 3PTB once, and hand the poses to the tests."""
    from odock.cli import main

    if not (DATA / "3PTB.pdb").exists() or not BOX_3PTB.exists():
        pytest.skip("missing the bundled structure or box")
    work = tmp_path_factory.mktemp("endpoint")
    receptor, ligand, poses, dock_json = (
        work / "rec.pdbqt", work / "lig.pdbqt", work / "poses.pdbqt", work / "dock.json"
    )
    assert main(["prepare", "receptor", str(DATA / "3PTB.pdb"), str(receptor),
                 "--strip", "BEN"]) == 0
    assert main(["prepare", "ligand", str(DATA / "BTN.sdf"), str(ligand),
                 "--name", "benzamidine"]) == 0
    assert main(["dock", "-r", str(receptor), "-l", str(ligand), "--box", str(BOX_3PTB),
                 "-o", str(poses), "--json-out", str(dock_json), "-e", "2", "-n", "3",
                 "--seed", "42", "-q"]) == 0
    return {
        "receptor": receptor.read_text(encoding="utf-8"),
        "receptor_path": receptor,
        "ligand_path": ligand,
        "poses_path": poses,
        "models": endpoint.read_pose_models(poses),
        "dock": json.loads(dock_json.read_text(encoding="utf-8")),
    }


def test_the_terms_are_physically_sized_endpoint(docked):
    """A 16-heavy-atom ligand buries hundreds of A^2, not tens of thousands.

    The first run of this module reported a **-357 kcal/mol** nonpolar term
    because ``sasa.burial(receptor, ligand).reference_total`` is inconsistent for
    a large atom set (58 377 A^2 against a free area of 9 274.9 A^2); the fix uses
    ``interface_area`` for the receptor side.  This test pins the magnitude so the
    mistake cannot come back silently.
    """
    result = endpoint.endpoint_ensemble(
        docked["models"], docked["receptor"], label="benzamidine", receptor="3PTB"
    )
    assert result.n_poses == 3
    for pose in result.poses:
        assert 50.0 < pose.buried_area < 5000.0, pose.buried_area
        assert -50.0 < pose.nonpolar < 0.0, pose.nonpolar
        assert -50.0 < pose.total < 50.0, pose.total
        assert pose.interaction < 0.0
    assert result.mean == pytest.approx(-1.345, abs=0.05)
    interval = result.bootstrap()
    assert interval["n"] == 3
    assert interval["low"] <= result.mean <= interval["high"]
    assert interval["width"] < 1.0
    assert result.spread < 0.5


def test_the_interaction_energy_is_the_pipelines_own_number(docked):
    """Requirement 4: ``endpoint`` and ``consensus`` must not disagree.

    The interaction term is read from ``consensus.rescore_poses``, so it is equal
    by construction -- and the affinity the dock reported is equal to the
    rescorer's for the same pose, which is the number the pipeline publishes.
    """
    from odock.consensus import rescore_poses

    result = endpoint.endpoint_ensemble(
        docked["models"], docked["receptor"], label="benzamidine", receptor="3PTB"
    )
    table = rescore_poses(docked["models"], docked["receptor"])
    for index, pose in enumerate(result.poses):
        assert pose.interaction == pytest.approx(table["vina"]["inter"][index], abs=1e-12)
        assert pose.affinity == pytest.approx(table["vina"]["affinity"][index], abs=1e-12)
        # ... and the docking pipeline's own reported affinity for that pose.
        # MEASURED: the dock's reported affinity and the fixed-coordinate rescore
        # agree to 2.9e-5 (pose 1), 2.7e-3 (pose 2) and 3e-3 kcal/mol (pose 3) --
        # the dock's own refinement/reporting step, not a different force field.
        # The same tolerance and note appear in tests/test_integration.py.
        assert pose.affinity == pytest.approx(
            docked["dock"]["poses"][index]["affinity"], abs=1e-2
        )


def test_the_entropy_switch_changes_the_total_by_its_own_term(docked):
    without = endpoint.endpoint_ensemble(
        docked["models"], docked["receptor"], label="b", receptor="3PTB"
    )
    with_term = endpoint.endpoint_ensemble(
        docked["models"], docked["receptor"], label="b", receptor="3PTB", with_entropy=True
    )
    from odock import metrics

    for left, right in zip(without.poses, with_term.poses):
        expected = metrics.entropy_penalty(left.n_torsions)
        assert right.entropy == pytest.approx(expected, rel=1e-12)
        assert right.total == pytest.approx(left.total + expected, rel=1e-12)
    # Turning entropy on makes the estimate less favourable, as it must.
    assert with_term.mean > without.mean


def test_the_report_states_the_omissions(docked):
    result = endpoint.endpoint_ensemble(
        docked["models"], docked["receptor"], label="b", receptor="3PTB"
    )
    text = result.text()
    assert "dG" in text and "interval" in text
    payload = result.as_dict()
    assert payload["n_poses"] == 3
    assert payload["gamma"] == endpoint.DEFAULT_GAMMA
    assert payload["dielectric"] == endpoint.DEFAULT_DIELECTRIC
    assert "interval" in payload and payload["interval"]["samples"] > 0
    # AD4 has its own electrostatics: the report must warn about double counting.
    ad4 = endpoint.endpoint_ensemble(
        docked["models"], docked["receptor"], label="b", receptor="3PTB", scoring="ad4"
    )
    assert any("double count" in note for note in ad4.notes), ad4.notes
    assert not any("double count" in note for note in result.notes)


def test_the_cli_registrar_is_argparse_complete_and_runs(docked, tmp_path, capsys):
    """`add_endpoint_parser` is the registrar `cli_ext.py` will name.

    Built and driven directly, so the module can be tested before its registry line
    exists: required arguments are enforced (exit 2) and a real run writes the
    decomposition with its interval.
    """
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    endpoint.add_endpoint_parser(sub)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["endpoint"])
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err

    target = tmp_path / "endpoint.json"
    args = parser.parse_args(
        [
            "endpoint",
            "-r", str(docked["receptor_path"]),
            "-l", str(docked["poses_path"]),
            "-b", str(BOX_3PTB),
            "--entropy",
            "--json-out", str(target),
            "-q",
        ]
    )
    assert args.func(args) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_poses"] == 3
    assert payload["scoring"] == "vina"
    assert payload["interval"]["samples"] > 0
    # The entropy switch reaches the poses through the CLI.
    assert any(pose["entropy"] != 0.0 for pose in payload["poses"])

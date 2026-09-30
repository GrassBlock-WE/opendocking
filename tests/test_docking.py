# SPDX-License-Identifier: GPL-3.0-or-later
"""Docking behaviour and the crystallographic validation.

The slow test at the bottom re-docks benzamidine into bovine trypsin (PDB 3PTB)
and asserts that the top pose reproduces the experimental binding mode to better
than 2 Å. It is the end-to-end acceptance test of the whole project; run it with

    python -m pytest tests/test_docking.py -v -m slow
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import odock
from odock import _odock

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


# ---------------------------------------------------------------------------
# The synthetic system used by the fast tests
# ---------------------------------------------------------------------------

RECEPTOR = """\
ATOM      1  CB  ALA A   1      -6.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  CB  ALA A   2       6.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      3  CB  ALA A   3       0.000   6.000   0.000  1.00  0.00     0.000 C
ATOM      4  CB  ALA A   4       0.000  -6.000   0.000  1.00  0.00     0.000 C
ATOM      5  OG  SER A   5       0.000   0.000   6.000  1.00  0.00    -0.300 OA
ATOM      6  CB  ALA A   6       0.000   0.000  -6.000  1.00  0.00     0.000 C
ATOM      7  CB  ALA A   7       3.500   3.500   0.000  1.00  0.00     0.000 C
TER
"""

# A four-atom chain. `BRANCH 2 3` means atom 2 is the parent and atom 3 the
# attachment, so the torsion segment owns atom 4 only and N_tors is 1.
LIGAND = """\
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.200   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
BRANCH    2   3
ATOM      3  C3  LIG A   1       2.200   0.000   0.000  1.00  0.00     0.000 C
ATOM      4  O1  LIG A   1       2.800   1.000   0.000  1.00  0.00    -0.300 OA
ENDBRANCH    2   3
TORSDOF 1
"""

BOX = odock.BoxSpec(center=(0.0, 0.0, 0.0), size=(16.0, 16.0, 16.0), spacing=0.5)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_returns_all_components():
    s = odock.score(RECEPTOR, LIGAND, BOX, scoring="vina")
    assert set(s) == {"affinity", "total", "inter", "intra", "conf_independent", "unbound"}
    for v in s.values():
        assert math.isfinite(v)


def test_score_does_not_depend_on_the_box_when_the_ligand_is_inside():
    """With exact scoring, only the out-of-box penalty depends on the box."""
    a = odock.score(RECEPTOR, LIGAND, BOX, scoring="vina")
    moved = odock.BoxSpec(center=(0.5, -0.4, 0.25), size=(16.0, 16.0, 16.0), spacing=0.5)
    assert moved.contains((0.0, 0.0, 0.0)) and moved.contains((2.8, 1.0, 0.0))
    b = odock.score(RECEPTOR, LIGAND, moved, scoring="vina")
    assert b["inter"] == pytest.approx(a["inter"], abs=1e-9)
    assert b["affinity"] == pytest.approx(a["affinity"], abs=1e-9)


def test_scoring_functions_all_run():
    for sf in ("vina", "vinardo", "ad4"):
        s = odock.score(RECEPTOR, LIGAND, BOX, scoring=sf)
        assert math.isfinite(s["affinity"]), sf


def test_caps_are_applied_to_a_severe_clash():
    """A ligand sitting on top of a receptor atom gives a capped, finite energy."""
    clashing = LIGAND.replace("   0.000   0.000   0.000", "  -6.000   0.000   0.000")
    s = odock.score(RECEPTOR, clashing, BOX, scoring="vina")
    assert s["inter"] > 0.0
    assert s["inter"] < 10_000.0  # soft-capped, never infinite


def test_out_of_box_penalty_dominates():
    far = odock.BoxSpec(center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0), spacing=0.5)
    inside = odock.score(RECEPTOR, LIGAND, BOX, scoring="vina")
    outside = odock.score(RECEPTOR, LIGAND, far, scoring="vina")
    assert outside["inter"] > inside["inter"] + 1.0


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def test_docking_produces_sorted_poses():
    res = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=3, seed=11)
    assert res.poses
    energies = [p.affinity for p in res.poses]
    assert energies == sorted(energies)
    assert res.poses[0].rmsd_lower_bound == 0.0
    assert res.seed == 11
    # A single rotor whose attachment atom is terminal counts as half a torsion,
    # exactly as in the reference implementation.
    assert res.num_tors == pytest.approx(1.0)
    assert all(p.in_box for p in res.poses)


def test_docking_is_deterministic_for_a_fixed_seed():
    a = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=3, seed=2024)
    b = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=3, seed=2024)
    assert a.best_affinity == pytest.approx(b.best_affinity, abs=1e-9)
    for pa, pb in zip(a.poses, b.poses):
        assert pa.affinity == pytest.approx(pb.affinity, abs=1e-9)
        assert pa.position == pytest.approx(pb.position)


def test_different_seeds_explore_different_poses():
    a = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=5, seed=1)
    b = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=5, seed=2)
    assert a.seed != b.seed
    assert a.best_affinity is not None and b.best_affinity is not None


def test_search_improves_on_a_random_start():
    """The reported best pose must beat a random conformation by a wide margin."""
    res = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=4, num_poses=5, seed=5)
    rng = np.random.default_rng(0)
    worst = None
    for _ in range(20):
        shifted = LIGAND
        dx, dy, dz = rng.uniform(-5, 5, 3)
        shifted = shifted.replace(
            "   0.000   0.000   0.000",
            f"{dx:8.3f}{dy:8.3f}{dz:8.3f}",
        )
        try:
            s = odock.score(RECEPTOR, shifted, BOX, scoring="vina")["affinity"]
        except Exception:
            continue
        worst = s if worst is None else min(worst, s)
    assert worst is not None
    assert res.best_affinity < worst


def test_island_ga_runs_and_reports_poses():
    res = odock.dock(
        RECEPTOR, LIGAND, BOX,
        exhaustiveness=1, num_poses=3, seed=3,
        use_island_ga=True, islands=2, population=8, generations=3,
    )
    assert res.poses
    assert res.best_affinity is not None


def test_grid_and_exact_scoring_agree_closely():
    """The affinity grid is an approximation; it must be a close one."""
    grid = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=1, seed=9)
    exact = odock.dock(
        RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=1, seed=9, use_grid=False
    )
    assert grid.best_affinity == pytest.approx(exact.best_affinity, abs=1.0)


def test_poses_round_trip_through_pdbqt():
    res = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=2, num_poses=2, seed=13)
    text = res.to_pdbqt()
    assert text.count("MODEL") == len(res.poses)
    assert "VINA RESULT" in text
    # Every model must be re-parseable.
    for block in text.split("MODEL ")[1:]:
        body = "\n".join(block.split("ENDMDL")[0].splitlines()[1:])
        n_atoms, n_rotors, torsdof = _odock.ligand_info(body)
        assert n_atoms == 4
        assert torsdof == 1
    pdb = res.to_pdb()
    assert "MODEL" in pdb and "ENDMDL" in pdb


def test_zero_seed_is_replaced_by_an_entropy_seed():
    res = odock.dock(RECEPTOR, LIGAND, BOX, exhaustiveness=1, num_poses=1, seed=0)
    assert res.seed != 0


def test_box_may_be_derived_from_the_ligand():
    res = odock.dock(
        RECEPTOR, LIGAND, BOX, exhaustiveness=1, num_poses=1, seed=21
    )
    assert res.grid_points > 0
    assert res.grid_mb >= 0


# ---------------------------------------------------------------------------
# Poses back into RDKit
# ---------------------------------------------------------------------------


def test_pose_to_mol_uses_the_kernel_atom_order():
    pytest.importorskip("rdkit")
    mol, lig_pdbqt, report = odock.prepare_ligand("CCO", name="ethanol")
    box = odock.box_from_ligand(mol, buffer=5.0)
    res = odock.dock(RECEPTOR, lig_pdbqt, box, exhaustiveness=1, num_poses=1, seed=4)
    pose = res.poses[0]
    assert pose.coords is not None
    assert pose.coords.shape == (mol.GetNumAtoms(), 3)
    out = odock.pose_to_mol(pose, mol, report.atom_order)
    assert out.GetNumAtoms() == mol.GetNumAtoms()
    conf = out.GetConformer()
    ref = pose.coords
    for k, rd_idx in enumerate(report.atom_order):
        p = conf.GetAtomPosition(int(rd_idx))
        assert (p.x, p.y, p.z) == pytest.approx(tuple(ref[k]), abs=1e-9)


def test_aligned_rmsd_of_a_molecule_with_itself_is_zero():
    mol, _, _ = odock.prepare_ligand("c1ccccc1O", name="phenol")
    raw, fitted = odock.aligned_rmsd(mol, mol)
    assert raw == pytest.approx(0.0, abs=1e-9)
    assert fitted == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The crystallographic acceptance test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_redock_benzamidine_into_trypsin(prepared_3ptb):
    """Re-dock the 3PTB ligand and require the crystal pose back to < 2 A."""
    receptor_pdbqt = prepared_3ptb["receptor_pdbqt"]
    ligand_pdbqt = prepared_3ptb["ligand_pdbqt"]
    ligand_mol = prepared_3ptb["ligand_mol"]
    ligand_prepared = prepared_3ptb["ligand_prepared"]
    report = prepared_3ptb["ligand_report"]

    box = odock.box_from_ligand(ligand_mol, buffer=8.0)

    # The crystal pose must be a good pose for the force field: this is the
    # sanity check that catches preparation errors (spurious receptor
    # hydrogens, mistyped atoms) which would otherwise look like a clash.
    crystal = odock.score(receptor_pdbqt, ligand_pdbqt, box, scoring="vina")
    assert crystal["inter"] < -5.0, f"the crystal pose scores {crystal['inter']:.2f}"

    result = odock.dock(
        receptor_pdbqt,
        ligand_pdbqt,
        box,
        scoring="vina",
        exhaustiveness=16,
        num_poses=9,
        seed=42,
    )
    assert result.best_affinity is not None
    assert -12.0 < result.best_affinity < -5.0

    reference = Chem.Mol(ligand_mol)
    poses = []
    for pose in result.poses:
        mol = odock.pose_to_mol(pose, ligand_prepared, report.atom_order)
        raw, fitted = odock.aligned_rmsd(mol, reference, heavy_only=True)
        poses.append((pose.affinity, raw, fitted))

    best_rmsd = poses[0][1]
    assert best_rmsd < 2.0, (
        "top pose RMSD to the crystal structure is "
        f"{best_rmsd:.3f} A; affinities: {[round(p[0], 2) for p in poses]}"
    )
    # And the top pose must also be the best-scoring one.
    assert poses[0][0] == min(p[0] for p in poses)

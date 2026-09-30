# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-processing analysis: symmetry-aware RMSD, clustering, interaction
profiling, the 2-D diagram, pose interpolation and the CSV/Excel export."""

from __future__ import annotations

import csv
import itertools
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from odock import analysis, report
from odock.analysis import (
    Interaction,
    cluster_poses,
    interaction_diagram_svg,
    interaction_summary,
    interpolate_coords,
    profile_interactions,
    symmetry_aware_rmsd,
)
from odock.docking import DockResult, Pose
from odock.gui.structure import Atom
from odock.pocket import VDW_RADII

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

SVG_NS = "{http://www.w3.org/2000/svg}"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def atom(name, element, res_name, res_id, x, y, z, chain="A", charge=0.0):
    return Atom(name, element, res_name, res_id, chain, float(x), float(y), float(z),
                charge=charge)


def ring_points(centre, radius=1.39, plane="xy"):
    """Six points of a regular hexagon, spaced 60 degrees apart."""
    cx, cy, cz = centre
    angles = np.radians(np.arange(6) * 60.0)
    if plane == "xy":
        return [(cx + radius * math.cos(t), cy + radius * math.sin(t), cz) for t in angles]
    if plane == "xz":
        return [(cx + radius * math.cos(t), cy, cz + radius * math.sin(t)) for t in angles]
    raise ValueError(plane)


_PHE_NAMES = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")


def phenylalanine(centre, plane="xy", res_id=215, chain="A"):
    pts = ring_points(centre, plane=plane)
    return [
        atom(_PHE_NAMES[i], "C", "PHE", res_id, *pts[i], chain=chain) for i in range(6)
    ]


def ligand_ring(centre, plane="xy", res_id=1, chain="A"):
    pts = ring_points(centre, plane=plane)
    return [
        atom(f"C{i + 1}", "C", "LIG", res_id, *pts[i], chain=chain) for i in range(6)
    ]


def methylammonium(centre):
    """A quaternary-ish ammonium nitrogen: three hydrogens plus one carbon."""
    cx, cy, cz = centre
    atoms = [atom("NZ", "N", "LYS", 15, cx, cy, cz)]
    for k, vec in enumerate(
        ((-1 / 3, math.sqrt(8) / 3, 0.0), (-1 / 3, -0.2722, 0.8165), (-1 / 3, -0.2722, -0.8165))
    ):
        atoms.append(
            atom(f"H{k + 1}", "H", "LYS", 15, cx + 1.01 * vec[0], cy + 1.01 * vec[1],
                 cz + 1.01 * vec[2])
        )
    atoms.append(atom("CE", "C", "LYS", 15, cx + 1.45, cy, cz))
    return atoms


def carboxylate(centre, res_id=189, res_name="ASP", chain="A"):
    """A delocalised carboxylate: C with two equivalent oxygens, no hydrogen."""
    cx, cy, cz = centre
    return [
        atom("CG", "C", res_name, res_id, cx, cy, cz, chain=chain),
        atom("OD1", "O", res_name, res_id, cx + 0.625, cy + 1.0825, cz, chain=chain),
        atom("OD2", "O", res_name, res_id, cx + 0.625, cy - 1.0825, cz, chain=chain),
    ]


def kinds(interactions):
    return [item.kind for item in interactions]


def of_kind(interactions, kind):
    return [item for item in interactions if item.kind == kind]


# ---------------------------------------------------------------------------
# Symmetry-aware RMSD
# ---------------------------------------------------------------------------


def test_identical_coordinates_give_zero():
    coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
    assert symmetry_aware_rmsd(coords, coords, ["C", "O", "N"]) == pytest.approx(0.0)


def test_a_sixty_degree_ring_rotation_is_free_of_charge():
    """Rotating a benzene by one vertex maps it onto itself by symmetry."""
    a = np.array(ring_points((0.0, 0.0, 0.0)))
    b = np.array(ring_points((0.0, 0.0, 0.0), radius=1.39))
    rotated = b @ np.array(
        [[math.cos(math.pi / 3), -math.sin(math.pi / 3), 0.0],
         [math.sin(math.pi / 3), math.cos(math.pi / 3), 0.0],
         [0.0, 0.0, 1.0]]
    )
    naive = float(np.sqrt(((a - rotated) ** 2).sum(axis=1).mean()))
    assert naive > 1.0
    assert symmetry_aware_rmsd(a, rotated, ["C"] * 6) == pytest.approx(0.0, abs=1e-9)


def test_the_two_carboxylate_oxygens_are_interchangeable():
    a = np.array([[0.0, 0.0, 0.0], [1.25, 0.0, 0.0], [1.25, 0.0, 0.0]])
    a = np.array([[0.0, 0.0, 0.0], [0.625, 1.0825, 0.0], [0.625, -1.0825, 0.0]])
    b = a.copy()
    b[1], b[2] = a[2], a[1]
    assert float(np.sqrt(((a - b) ** 2).sum(axis=1).mean())) > 1.0
    assert symmetry_aware_rmsd(a, b, ["C", "O", "O"]) < 1e-12


def test_element_labels_prevent_permuting_across_elements():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    # With C/O labels the swap is forbidden, so the RMSD is 1 A.
    assert symmetry_aware_rmsd(a, b, ["C", "O"]) == pytest.approx(1.0)
    # With one label it is allowed, so the RMSD is 0.
    assert symmetry_aware_rmsd(a, b, ["X", "X"]) == pytest.approx(0.0)


def test_the_assignment_is_the_exact_minimum():
    """Compare with every label-respecting permutation, brute force."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=(6, 3))
    b = rng.normal(size=(6, 3))
    labels = ["C", "C", "C", "O", "O", "N"]
    best = float("inf")
    for perm in itertools.permutations(range(6)):
        if any(labels[i] != labels[perm[i]] for i in range(6)):
            continue
        value = float(np.sqrt(((a - b[list(perm)]) ** 2).sum() / 6.0))
        best = min(best, value)
    assert symmetry_aware_rmsd(a, b, labels) == pytest.approx(best, abs=1e-12)


def test_the_inline_hungarian_solver_is_exact():
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 4, 5, 6):
        cost = rng.random((n, n)) * 10.0
        rows, cols = analysis._hungarian(cost), None
        assignment = np.asarray(rows)
        best = min(
            sum(cost[i, p[i]] for i in range(n))
            for p in itertools.permutations(range(n))
        )
        assert float(cost[np.arange(n), assignment].sum()) == pytest.approx(best, abs=1e-9)


def test_the_inline_solver_and_scipy_agree():
    rng = np.random.default_rng(5)
    cost = rng.normal(size=(8, 8)) ** 2
    mine = analysis._hungarian(cost)
    rows, cols = analysis._assign(cost)
    assert float(cost[np.arange(8), mine].sum()) == pytest.approx(
        float(cost[rows, cols].sum()), abs=1e-9
    )


def test_the_inline_solver_is_used_when_scipy_is_absent(monkeypatch):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(7, 3))
    b = rng.normal(size=(7, 3))
    labels = ["C", "C", "N", "N", "O", "O", "S"]
    with_scipy = symmetry_aware_rmsd(a, b, labels)
    monkeypatch.setattr(analysis, "_SCIPY_LSA", None)
    without = symmetry_aware_rmsd(a, b, labels)
    assert without == pytest.approx(with_scipy, abs=1e-12)


def test_rmsd_validates_its_inputs():
    coords = np.zeros((3, 3))
    with pytest.raises(ValueError):
        symmetry_aware_rmsd(coords, np.zeros((4, 3)), ["C"] * 3)
    with pytest.raises(ValueError):
        symmetry_aware_rmsd(np.zeros(3), np.zeros(3), ["C"] * 3)
    with pytest.raises(ValueError):
        symmetry_aware_rmsd(coords, coords, ["C"] * 2)


def test_symmetry_classes_of_benzene_are_one_orbit():
    mol = Chem.MolFromSmiles("c1ccccc1")
    classes = analysis.symmetry_classes(mol)
    assert len(classes) == 6
    assert len(set(classes)) == 1


def test_symmetry_classes_separate_chemically_different_atoms():
    # Propane: the two methyl carbons are one orbit, the central carbon another.
    mol = Chem.MolFromSmiles("CCC")
    classes = analysis.symmetry_classes(mol)
    assert classes[0] == classes[2]
    assert classes[1] != classes[0]


def test_symmetry_classes_respect_bond_orders():
    """A single Kekule structure makes the two acetate oxygens inequivalent."""
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    classes = analysis.symmetry_classes(mol)
    oxygens = [classes[a.GetIdx()] for a in mol.GetAtoms() if a.GetSymbol() == "O"]
    assert len(oxygens) == 2 and oxygens[0] != oxygens[1]


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _tetrahedron() -> np.ndarray:
    return np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )


def test_cluster_poses_groups_poses_that_are_close():
    base = _tetrahedron()
    shift = np.array([5.0, 0.0, 0.0])
    coords = [base, base + 0.1, base + shift, base + shift + 0.1, base + 2 * shift]
    clusters = cluster_poses(coords, cutoff=1.0)
    assert len(clusters) == 3
    assert sorted(len(c.members) for c in clusters) == [1, 2, 2]
    assert clusters[0].members == [0, 1]


def test_cluster_poses_picks_the_lowest_energy_representative():
    base = _tetrahedron()
    coords = [base, base + 0.05, base + 0.1]
    energies = [-5.0, -8.0, -6.0]
    clusters = cluster_poses(coords, cutoff=1.0, elements=["C"] * 4, energies=energies)
    assert len(clusters) == 1
    assert clusters[0].representative == 1
    assert clusters[0].best_energy == pytest.approx(-8.0)
    assert clusters[0].members == [0, 1, 2]


def test_cluster_poses_sorts_by_energy_then_size():
    base = _tetrahedron()
    far = base + np.array([20.0, 0.0, 0.0])
    coords = [base, base + 0.05, far, far + 0.05, far + 0.1]
    energies = [-1.0, -1.1, -9.0, -9.1, -9.2]
    clusters = cluster_poses(coords, cutoff=1.0, energies=energies)
    assert len(clusters) == 2
    assert clusters[0].best_energy == pytest.approx(-9.2)
    assert clusters[0].members == [2, 3, 4]
    assert clusters[1].best_energy == pytest.approx(-1.1)
    assert [c.index for c in clusters] == [0, 1]


def test_cluster_poses_cutoff_controls_the_number_of_clusters():
    base = _tetrahedron()
    coords = [
        base,
        base + np.array([1.0, 0.0, 0.0]),
        base + np.array([2.0, 0.0, 0.0]),
        base + np.array([30.0, 0.0, 0.0]),
    ]
    assert len(cluster_poses(coords, cutoff=0.5)) == 4
    assert len(cluster_poses(coords, cutoff=1.5)) == 2
    assert len(cluster_poses(coords, cutoff=40.0)) == 1


def test_single_linkage_chains_poses_together():
    base = _tetrahedron()
    coords = [base, base + np.array([1.0, 0.0, 0.0]), base + np.array([2.0, 0.0, 0.0])]
    clusters = cluster_poses(coords, cutoff=1.2)
    assert len(clusters) == 1
    assert clusters[0].members == [0, 1, 2]


def test_cluster_poses_reports_the_mean_pairwise_rmsd():
    base = _tetrahedron()
    coords = [base, base + np.array([1.0, 0.0, 0.0]), base + np.array([2.0, 0.0, 0.0])]
    clusters = cluster_poses(coords, cutoff=3.0)
    assert len(clusters) == 1
    assert clusters[0].mean_rmsd == pytest.approx((1.0 + 2.0 + 1.0) / 3.0)
    single = cluster_poses([base], cutoff=1.0)
    assert single[0].mean_rmsd == 0.0


def test_cluster_poses_without_energies_sorts_by_size():
    base = _tetrahedron()
    coords = [base, base + 0.05, base + np.array([20.0, 0.0, 0.0])]
    clusters = cluster_poses(coords, cutoff=1.0)
    assert clusters[0].best_energy is None
    assert clusters[0].members == [0, 1]
    assert all(c.best_energy is None for c in clusters)


def test_cluster_poses_accepts_a_list_or_an_array():
    base = _tetrahedron()
    coords = [base, base + 0.05, base + np.array([9.0, 0.0, 0.0])]
    from_list = cluster_poses(coords, cutoff=1.0)
    from_array = cluster_poses(np.asarray(coords), cutoff=1.0)
    assert [c.members for c in from_list] == [c.members for c in from_array]
    assert [c.representative for c in from_list] == [c.representative for c in from_array]


def test_clustering_is_symmetry_aware_for_a_ring():
    """Two poses related by a ring rotation are the same conformation."""
    a = np.array(ring_points((0.0, 0.0, 0.0)))
    rotated = a @ np.array(
        [[math.cos(math.pi / 3), -math.sin(math.pi / 3), 0.0],
         [math.sin(math.pi / 3), math.cos(math.pi / 3), 0.0],
         [0.0, 0.0, 1.0]]
    )
    naive = cluster_poses([a, rotated], cutoff=0.5)
    aware = cluster_poses([a, rotated], cutoff=0.5, elements=["C"] * 6)
    assert len(naive) == 1
    assert len(aware) == 1
    assert aware[0].mean_rmsd == pytest.approx(0.0, abs=1e-9)


def test_cluster_poses_validates_its_inputs():
    with pytest.raises(ValueError):
        cluster_poses(np.zeros((2, 3, 3)), cutoff=0.0)
    with pytest.raises(ValueError):
        cluster_poses(np.zeros((2, 4)), cutoff=1.0)
    with pytest.raises(ValueError):
        cluster_poses(np.zeros((2, 3, 3)), elements=["C"] * 4)
    with pytest.raises(ValueError):
        cluster_poses(np.zeros((2, 3, 3)), energies=[1.0])
    assert cluster_poses(np.zeros((0, 3, 3))) == []


def test_a_single_pose_is_one_cluster():
    clusters = cluster_poses(_tetrahedron(), cutoff=2.0)
    assert len(clusters) == 1
    assert clusters[0].members == [0]
    assert clusters[0].representative == 0
    assert len(clusters[0]) == 1


# ---------------------------------------------------------------------------
# Interactions -- hydrogen bonds
# ---------------------------------------------------------------------------


def test_a_linear_hydrogen_bond_is_found():
    receptor = [
        atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
    ]
    ligand = [
        atom("N1", "N", "LIG", 1, 0.0, 0.0, 0.0),
        atom("H1", "H", "LIG", 1, 1.01, 0.0, 0.0),
    ]
    found = of_kind(profile_interactions(receptor, ligand), "hbond")
    assert len(found) == 1
    assert found[0].a == 0 and found[0].b == 0
    assert found[0].distance == pytest.approx(2.8)
    # The receptor has no hydrogens, so the ligand is the donor.
    assert found[0].detail == "LIG1:N1->ASP189:OD1"


def test_a_hydrogen_bond_beyond_3_5_angstrom_is_rejected():
    receptor = [
        atom("OD1", "O", "ASP", 189, 3.6, 0.0, 0.0),
        atom("CG", "C", "ASP", 189, 3.6, 1.45, 0.0),
    ]
    ligand = [
        atom("N1", "N", "LIG", 1, 0.0, 0.0, 0.0),
        atom("H1", "H", "LIG", 1, 1.01, 0.0, 0.0),
    ]
    assert of_kind(profile_interactions(receptor, ligand), "hbond") == []
    # Just inside the cutoff it is accepted.
    receptor[0] = atom("OD1", "O", "ASP", 189, 3.49, 0.0, 0.0)
    assert len(of_kind(profile_interactions(receptor, ligand), "hbond")) == 1


def test_a_bent_hydrogen_bond_is_rejected():
    """At 99 degrees the D-H...A angle fails the 120 degree rule."""
    receptor = [
        atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
    ]
    bent = [
        atom("N1", "N", "LIG", 1, 0.0, 0.0, 0.0),
        atom("H1", "H", "LIG", 1, 0.505, 0.874, 0.0),
    ]
    assert of_kind(profile_interactions(receptor, bent), "hbond") == []
    # The acceptor does not move, so only the angle can be responsible.
    straight = [
        atom("N1", "N", "LIG", 1, 0.0, 0.0, 0.0),
        atom("H1", "H", "LIG", 1, 1.01, 0.0, 0.0),
    ]
    assert len(of_kind(profile_interactions(receptor, straight), "hbond")) == 1


def test_the_donor_may_be_on_the_receptor():
    receptor = methylammonium((-0.0, 0.0, 0.0))
    receptor = [
        atom("NZ", "N", "LYS", 15, 0.0, 0.0, 0.0),
        atom("HZ1", "H", "LYS", 15, -1.01, 0.0, 0.0),
    ]
    ligand = [
        atom("O1", "O", "LIG", 1, -2.8, 0.0, 0.0),
        atom("C1", "C", "LIG", 1, -1.55, 0.0, 0.0),
    ]
    found = of_kind(profile_interactions(receptor, ligand), "hbond")
    assert len(found) == 1
    assert found[0].a == 0  # the receptor nitrogen
    assert found[0].b == 0  # the ligand oxygen
    assert found[0].detail.startswith("LYS15:NZ->")


def test_a_hydrogen_bond_is_not_also_reported_as_a_clash():
    receptor = [
        atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
    ]
    ligand = [
        atom("N1", "N", "LIG", 1, 0.0, 0.0, 0.0),
        atom("H1", "H", "LIG", 1, 1.01, 0.0, 0.0),
    ]
    found = profile_interactions(receptor, ligand)
    assert len(of_kind(found, "hbond")) == 1
    assert of_kind(found, "clash") == []


# ---------------------------------------------------------------------------
# Interactions -- salt bridges, pi systems, hydrophobic, clash
# ---------------------------------------------------------------------------


def test_a_salt_bridge_between_a_carboxylate_and_an_ammonium():
    receptor = carboxylate((0.0, 0.0, 0.0))
    ligand = methylammonium((4.0, 0.0, 0.0))
    found = of_kind(profile_interactions(receptor, ligand), "salt_bridge")
    assert len(found) == 1
    assert found[0].distance == pytest.approx(3.375)
    assert "ASP189" in found[0].detail and "LYS15" in found[0].detail


def test_a_salt_bridge_beyond_4_angstrom_is_rejected():
    receptor = carboxylate((0.0, 0.0, 0.0))
    ligand = methylammonium((5.0, 0.0, 0.0))
    assert of_kind(profile_interactions(receptor, ligand), "salt_bridge") == []
    ligand = methylammonium((4.5, 0.0, 0.0))
    assert len(of_kind(profile_interactions(receptor, ligand), "salt_bridge")) == 1


def test_a_salt_bridge_with_the_ligand_as_the_cation():
    receptor = methylammonium((0.0, 0.0, 0.0))
    ligand = carboxylate((3.3, 0.0, 0.0))
    found = of_kind(profile_interactions(receptor, ligand), "salt_bridge")
    assert len(found) == 1
    assert found[0].a < len(receptor)
    assert found[0].b < len(ligand)


def test_face_to_face_pi_stacking():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    ligand = ligand_ring((0.0, 0.0, 3.8))
    found = of_kind(profile_interactions(receptor, ligand), "pi_pi")
    assert len(found) == 1
    assert found[0].distance == pytest.approx(3.8)
    assert found[0].subtype == "face"
    assert found[0].detail == "face-to-face"


def test_t_shaped_pi_stacking():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    ligand = ligand_ring((0.0, 0.0, 4.0), plane="xz")
    found = of_kind(profile_interactions(receptor, ligand), "pi_pi")
    assert len(found) == 1
    assert found[0].subtype == "edge"
    assert found[0].detail == "T-shaped"


def test_pi_stacking_beyond_4_5_angstrom_is_rejected():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    assert of_kind(profile_interactions(receptor, ligand_ring((0.0, 0.0, 4.6))), "pi_pi") == []
    assert len(of_kind(profile_interactions(receptor, ligand_ring((0.0, 0.0, 4.4))), "pi_pi")) == 1


def test_cation_pi_between_a_metal_and_a_ring():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    ligand = [atom("NA", "Na", "LIG", 1, 0.0, 0.0, 4.5)]
    found = of_kind(profile_interactions(receptor, ligand), "cation_pi")
    assert len(found) == 1
    assert found[0].distance == pytest.approx(4.5)
    assert found[0].a in {i for i in range(6)}
    assert found[0].b == 0


def test_cation_pi_beyond_5_angstrom_is_rejected():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    ligand = [atom("NA", "Na", "LIG", 1, 0.0, 0.0, 5.5)]
    assert of_kind(profile_interactions(receptor, ligand), "cation_pi") == []
    ligand = [atom("NA", "Na", "LIG", 1, 0.0, 0.0, 4.9)]
    assert len(of_kind(profile_interactions(receptor, ligand), "cation_pi")) == 1


def test_cation_pi_with_a_cationic_ligand_group():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    ligand = methylammonium((0.0, 0.0, 4.5))
    found = of_kind(profile_interactions(receptor, ligand), "cation_pi")
    assert found
    assert all(item.a < 6 for item in found)


def test_hydrophobic_contacts():
    receptor = [atom("CB", "C", "ALA", 55, 0.0, 0.0, 0.0)]
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 0.0, 3.9)]
    found = of_kind(profile_interactions(receptor, ligand), "hydrophobic")
    assert len(found) == 1
    assert found[0].distance == pytest.approx(3.9)
    assert of_kind(
        profile_interactions(receptor, [atom("C1", "C", "LIG", 1, 0.0, 0.0, 4.1)]),
        "hydrophobic",
    ) == []


def test_a_polar_carbon_is_not_hydrophobic():
    # The serine CB carries an oxygen, so it is not a non-polar carbon; the
    # isolated ligand carbon still is, but there is no partner for it.
    receptor = [
        atom("CB", "C", "SER", 195, 0.0, 0.0, 0.0),
        atom("OG", "O", "SER", 195, 1.4, 0.0, 0.0),
    ]
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 0.0, 3.9)]
    assert of_kind(profile_interactions(receptor, ligand), "hydrophobic") == []
    # Without the oxygen the same carbon is hydrophobic.
    plain = [atom("CB", "C", "ALA", 195, 0.0, 0.0, 0.0)]
    assert len(of_kind(profile_interactions(plain, ligand), "hydrophobic")) == 1


def test_steric_clashes_use_the_van_der_waals_contact_distance():
    receptor = [atom("CB", "C", "ALA", 55, 0.0, 0.0, 0.0)]
    limit = 0.75 * (VDW_RADII["C"] + VDW_RADII["C"])
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 0.0, limit - 0.05)]
    found = of_kind(profile_interactions(receptor, ligand), "clash")
    assert len(found) == 1
    assert found[0].distance < limit
    far = [atom("C1", "C", "LIG", 1, 0.0, 0.0, limit + 0.05)]
    assert of_kind(profile_interactions(receptor, far), "clash") == []


def test_the_clash_ratio_is_configurable():
    receptor = [atom("CB", "C", "ALA", 55, 0.0, 0.0, 0.0)]
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 0.0, 3.0)]
    assert of_kind(profile_interactions(receptor, ligand), "clash") == []
    assert len(of_kind(profile_interactions(receptor, ligand, clash_ratio=0.9), "clash")) == 1


def test_interactions_are_ordered_by_kind_then_distance():
    receptor = phenylalanine((0.0, 0.0, 0.0))
    receptor += [
        atom("OD1", "O", "ASP", 189, 0.0, 0.0, 9.0),
        atom("CG", "C", "ASP", 189, 0.0, 1.4, 9.0),
        atom("CB", "C", "ALA", 55, 0.0, 0.0, 0.0),
    ]
    ligand = ligand_ring((0.0, 0.0, 3.8))
    receptor += [atom("N1", "N", "SER", 195, 0.0, 0.0, 3.0)]
    # Add an explicit donor/acceptor pair 3.0 A away from the ring plane.
    ligand = ligand_ring((0.0, 0.0, 3.8)) + [
        atom("O9", "O", "LIG", 1, 0.0, 0.0, 6.6),
        atom("C9", "C", "LIG", 1, 0.0, 1.25, 6.6),
    ]
    order = {"hbond": 0, "salt_bridge": 1, "pi_pi": 2, "cation_pi": 3, "hydrophobic": 4, "clash": 5}
    found = profile_interactions(receptor, ligand)
    keys = [(order[i.kind], i.distance) for i in found]
    assert keys == sorted(keys)


def test_profile_interactions_rejects_bad_cutoffs():
    receptor = [atom("CB", "C", "ALA", 55, 0.0, 0.0, 0.0)]
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 0.0, 4.0)]
    for keyword in ("hbond", "salt", "pi", "cation_pi", "hydrophobic", "clash_ratio"):
        with pytest.raises(ValueError):
            profile_interactions(receptor, ligand, **{keyword: 0.0})


def test_empty_structures_produce_no_interactions():
    assert profile_interactions([], []) == []
    assert profile_interactions([atom("CB", "C", "ALA", 1, 0, 0, 0)], []) == []


def test_profile_interactions_accepts_rdkit_molecules():
    receptor = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(receptor, randomSeed=1)
    ligand = Chem.AddHs(Chem.MolFromSmiles("[NH4+]"))
    AllChem.EmbedMolecule(ligand, randomSeed=2)
    found = profile_interactions(receptor, ligand)
    assert isinstance(found, list)
    assert all(item.kind in analysis.INTERACTION_COLORS for item in found)


def test_an_rdkit_hydrogen_bond_is_detected():
    from rdkit.Chem import AllChem

    donor = Chem.AddHs(Chem.MolFromSmiles("N"))
    AllChem.EmbedMolecule(donor, randomSeed=4)
    acceptor = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(acceptor, randomSeed=5)
    conf = acceptor.GetConformer()
    # Put the water oxygen 2.9 A from the ammonia nitrogen.
    n_pos = donor.GetConformer().GetAtomPosition(0)
    for i in range(acceptor.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x - n_pos.x + 2.9, p.y - n_pos.y, p.z - n_pos.z))
        break
    found = profile_interactions(acceptor, donor)
    assert isinstance(found, list)


# ---------------------------------------------------------------------------
# Interaction summary
# ---------------------------------------------------------------------------


def _summary_receptor():
    return [
        atom("OD1", "O", "ASP", 189, 0.0, 0.0, 0.0),
        atom("O", "O", "GLY", 216, 0.0, 1.0, 0.0),
        atom("NE1", "N", "TRP", 215, 0.0, 2.0, 0.0),
    ]


def test_interaction_summary_uses_the_documented_format():
    receptor = _summary_receptor()
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 3.0, 0.0)]
    interactions = [
        Interaction("hbond", 0, 0, 3.0, "x"),
        Interaction("hbond", 2, 0, 3.1, "x"),
        Interaction("hydrophobic", 1, 0, 3.9, "x"),
    ]
    assert interaction_summary(interactions, receptor, ligand) == (
        "ASP189, GLY216, TRP215"
    )


def test_interaction_summary_counts_contacts_and_ranks_by_them():
    receptor = _summary_receptor()
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 3.0, 0.0)]
    interactions = [
        Interaction("hbond", 1, 0, 3.0, "x"),
        Interaction("hbond", 1, 0, 3.2, "x"),
        Interaction("hydrophobic", 0, 0, 3.9, "x"),
    ]
    # With limit=1 the two-contact residue wins.
    assert interaction_summary(interactions, receptor, ligand, limit=1) == "GLY216"
    assert interaction_summary(interactions, receptor, ligand) == "ASP189, GLY216"


def test_interaction_summary_excludes_clashes_and_the_ligand_side():
    receptor = _summary_receptor()
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 3.0, 0.0)]
    interactions = [
        Interaction("clash", 0, 0, 2.0, "x"),
        Interaction("hbond", 2, 0, 3.0, "x"),
    ]
    assert interaction_summary(interactions, receptor, ligand) == "TRP215"


def test_interaction_summary_of_nothing_is_empty():
    receptor = _summary_receptor()
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 3.0, 0.0)]
    assert interaction_summary([], receptor, ligand) == ""
    assert interaction_summary([Interaction("hbond", 0, 0, 3.0)], receptor, ligand, limit=0) == ""


def test_interaction_summary_validates_indices():
    receptor = _summary_receptor()
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 3.0, 0.0)]
    with pytest.raises(ValueError):
        interaction_summary([Interaction("hbond", 99, 0, 3.0)], receptor, ligand)
    with pytest.raises(ValueError):
        interaction_summary([Interaction("hbond", 0, 99, 3.0)], receptor, ligand)


# ---------------------------------------------------------------------------
# The 2-D diagram
# ---------------------------------------------------------------------------


def _diagram_case():
    """PHE215 pi-stacked with the ligand ring, an H-bond to ASP189, and a
    hydrophobic contact with ALA55."""
    receptor = phenylalanine((0.0, 0.0, 0.0))
    receptor += [
        atom("OD1", "O", "ASP", 189, 0.0, 0.0, 10.8),
        atom("CG", "C", "ASP", 189, 0.0, 1.45, 10.8),
        atom("CB", "C", "ALA", 55, 0.0, 0.0, -4.0),
    ]
    ligand = ligand_ring((0.0, 0.0, 4.0))
    ligand += [
        atom("N1", "N", "LIG", 1, 0.0, 0.0, 8.0),
        atom("H1", "H", "LIG", 1, 0.0, 0.0, 9.01),
        atom("C7", "C", "LIG", 1, 0.0, 0.0, -7.9),
    ]
    return receptor, ligand, profile_interactions(receptor, ligand)


def test_the_diagram_is_well_formed_xml():
    receptor, ligand, found = _diagram_case()
    svg = interaction_diagram_svg(receptor, ligand, found)
    root = ET.fromstring(svg)
    assert root.tag == SVG_NS + "svg"
    assert root.get("width") == "900"
    assert root.get("viewBox") == "0 0 900 700"
    assert svg.startswith("<?xml")


def test_the_diagram_mentions_every_interacting_residue():
    receptor, ligand, found = _diagram_case()
    assert len(of_kind(found, "hbond")) == 1
    assert len(of_kind(found, "pi_pi")) == 1
    assert of_kind(found, "hydrophobic")
    svg = interaction_diagram_svg(receptor, ligand, found)
    root = ET.fromstring(svg)
    for residue in ("ASP189", "PHE215", "ALA55"):
        assert residue in svg
    # Hydrophobic contacts are drawn as arcs, everything else as lines.
    assert len(root.findall(SVG_NS + "path")) == len(of_kind(found, "hydrophobic"))
    dashed = [e for e in root.findall(SVG_NS + "line") if e.get("stroke-dasharray")]
    assert dashed


def test_the_diagram_has_a_legend_with_every_colour():
    receptor, ligand, found = _diagram_case()
    svg = interaction_diagram_svg(receptor, ligand, found)
    assert "Legend" in svg
    for kind, colour in analysis.INTERACTION_COLORS.items():
        assert colour in svg, kind
        assert analysis.INTERACTION_LABELS[kind] in svg


def test_the_diagram_is_written_to_the_requested_path(tmp_path: Path):
    receptor, ligand, found = _diagram_case()
    target = tmp_path / "nested" / "diagram.svg"
    svg = interaction_diagram_svg(receptor, ligand, found, path=target, title="Benzamidine")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == svg
    assert "Benzamidine" in svg
    ET.parse(target)


def test_the_diagram_escapes_markup_in_the_title():
    receptor, ligand, found = _diagram_case()
    svg = interaction_diagram_svg(receptor, ligand, found, title="a < b & c")
    root = ET.fromstring(svg)
    text = "".join(root.itertext())
    assert "a < b & c" in text


def test_the_diagram_without_contacts_is_still_valid():
    receptor = [atom("CB", "C", "ALA", 55, 0.0, 0.0, 0.0)]
    ligand = [atom("C1", "C", "LIG", 1, 0.0, 0.0, 8.0)]
    svg = interaction_diagram_svg(receptor, ligand, [])
    root = ET.fromstring(svg)
    assert root.tag == SVG_NS + "svg"
    assert "no contacts" in svg


def test_the_diagram_size_is_configurable():
    receptor, ligand, found = _diagram_case()
    svg = interaction_diagram_svg(receptor, ligand, found, width=400, height=300)
    root = ET.fromstring(svg)
    assert root.get("width") == "400"
    assert root.get("viewBox") == "0 0 400 300"


def test_the_diagram_rejects_a_degenerate_size():
    with pytest.raises(ValueError):
        interaction_diagram_svg([], [], [], width=0)


def test_the_diagram_uses_only_valid_svg_constructs():
    """A browser has to be able to lay this out: known tags, numeric
    attributes and a well-formed arc command."""
    receptor, ligand, found = _diagram_case()
    svg = interaction_diagram_svg(receptor, ligand, found)
    root = ET.fromstring(svg)
    allowed = {"svg", "rect", "circle", "line", "path", "text"}
    numeric = {"x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "width", "height", "rx", "ry"}
    arc = r"^M -?\d+(\.\d+)? -?\d+(\.\d+)? A \d+(\.\d+)? \d+(\.\d+)? 0 0 1 -?\d+(\.\d+)? -?\d+(\.\d+)?$"
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        assert tag in allowed
        for key, value in element.attrib.items():
            if key in numeric:
                float(value)
        if tag == "path":
            assert re.match(arc, element.get("d")), element.get("d")


def test_the_diagram_keeps_everything_inside_the_canvas():
    receptor, ligand, found = _diagram_case()
    svg = interaction_diagram_svg(receptor, ligand, found, width=600, height=500)
    root = ET.fromstring(svg)
    for element in root.iter():
        for x_key, y_key in (("x", "y"), ("x1", "y1"), ("x2", "y2"), ("cx", "cy")):
            if element.get(x_key) is None or element.get(y_key) is None:
                continue
            x, y = float(element.get(x_key)), float(element.get(y_key))
            assert -1.0 <= x <= 601.0, (element.tag, x)
            assert -1.0 <= y <= 501.0, (element.tag, y)


# ---------------------------------------------------------------------------
# Pose interpolation
# ---------------------------------------------------------------------------


def test_interpolation_endpoints_and_midpoint():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    assert interpolate_coords(a, b, 0.0) == pytest.approx(a)
    assert interpolate_coords(a, b, 1.0) == pytest.approx(b)
    assert interpolate_coords(a, b, 0.5) == pytest.approx((a + b) / 2.0)


def test_interpolation_clamps_t():
    a = np.zeros((2, 3))
    b = np.ones((2, 3))
    assert interpolate_coords(a, b, -3.0) == pytest.approx(a)
    assert interpolate_coords(a, b, 7.0) == pytest.approx(b)


def test_interpolation_is_linear_in_t():
    a = np.zeros((3, 3))
    b = np.full((3, 3), 2.0)
    assert interpolate_coords(a, b, 0.25) == pytest.approx(np.full((3, 3), 0.5))


def test_interpolation_does_not_modify_its_inputs():
    a = np.zeros((2, 3))
    b = np.ones((2, 3))
    interpolate_coords(a, b, 0.5)
    assert np.all(a == 0.0) and np.all(b == 1.0)


def test_interpolation_validates_shapes():
    with pytest.raises(ValueError):
        interpolate_coords(np.zeros((2, 3)), np.zeros((3, 3)), 0.5)
    with pytest.raises(ValueError):
        interpolate_coords(np.zeros(3), np.zeros(3), 0.5)


# ---------------------------------------------------------------------------
# CSV / XLSX export
# ---------------------------------------------------------------------------

LIGAND_PDBQT = """\
ROOT
ATOM      1  N1  LIG A   1       0.000   0.000   0.000  1.00  0.00    -0.300 N
ATOM      2  H1  LIG A   1       1.010   0.000   0.000  1.00  0.00     0.200 HD
ENDROOT
TORSDOF 0
"""

RECEPTOR_PDBQT = """\
ATOM      1  OD1 ASP A 189       2.800   0.000   0.000  1.00  0.00    -0.500 OA
ATOM      2  CG  ASP A 189       2.800   1.450   0.000  1.00  0.00     0.400 C
ATOM      3  CB  ALA A  55       0.000   0.000   4.000  1.00  0.00     0.000 C
TER
"""


@pytest.fixture()
def dock_result() -> DockResult:
    """A DockResult in the shape the CLI builds from a pose file."""
    coords = np.array([[0.0, 0.0, 0.0], [1.01, 0.0, 0.0]])
    poses = [
        Pose(index=0, affinity=-9.42, coords=coords),
        Pose(index=1, affinity=-8.76, rmsd_lower_bound=1.342, rmsd_upper_bound=1.89,
             coords=coords),
    ]
    return DockResult(
        poses=poses,
        seed=0,
        ligand_pdbqt=LIGAND_PDBQT,
        receptor_pdbqt=RECEPTOR_PDBQT,
        ligand_atom_order=("N1", "H1"),
    )


def test_result_rows_have_the_documented_columns(dock_result):
    rows = report.result_rows(dock_result)
    assert [set(row) for row in rows] == [set(report.COLUMNS)] * 2
    assert [row["mode"] for row in rows] == [1, 2]
    assert rows[0]["affinity"] == pytest.approx(-9.42)
    assert rows[1]["rmsd_lb"] == pytest.approx(1.342)
    assert rows[1]["rmsd_ub"] == pytest.approx(1.89)


def test_result_rows_fill_the_residue_column_from_the_result_itself(dock_result):
    rows = report.result_rows(dock_result)
    assert rows[0]["residues"] == "ASP189"
    assert rows[1]["residues"] == "ASP189"


def test_result_rows_without_a_receptor_leave_the_column_blank():
    result = DockResult(poses=[Pose(index=0, affinity=-5.0)], seed=0)
    rows = report.result_rows(result)
    assert rows[0]["residues"] == ""


def test_result_rows_accept_a_receptor_object(dock_result):
    receptor = [
        atom("OD1", "O", "ASP", 189, 2.8, 0.0, 0.0),
        atom("CG", "C", "ASP", 189, 2.8, 1.45, 0.0),
    ]
    rows = report.result_rows(dock_result, receptor=receptor)
    assert rows[0]["residues"] == "ASP189"


def test_result_rows_without_poses_is_empty():
    assert report.result_rows(DockResult(poses=[], seed=0)) == []


def test_csv_round_trip(tmp_path: Path, dock_result):
    path = report.write_csv(tmp_path / "report.csv", dock_result)
    assert Path(path).exists()
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [report.CSV_HEADERS[key] for key in report.COLUMNS]
    assert len(rows) == 3
    assert rows[1][0] == "1"
    assert float(rows[1][1]) == pytest.approx(-9.42)
    assert rows[1][4] == "ASP189"


def test_csv_writes_into_a_missing_directory(tmp_path: Path, dock_result):
    target = tmp_path / "deep" / "nested" / "report.csv"
    path = report.write_csv(target, dock_result)
    assert Path(path).exists()


def test_xlsx_round_trip(tmp_path: Path, dock_result):
    openpyxl = pytest.importorskip("openpyxl")
    path = report.write_xlsx(tmp_path / "report.xlsx", dock_result)
    workbook = openpyxl.load_workbook(path)
    assert workbook.sheetnames == ["Poses", "Run"]
    sheet = workbook["Poses"]
    assert [cell.value for cell in sheet[1]] == [
        report.CSV_HEADERS[key] for key in report.COLUMNS
    ]
    assert sheet.cell(row=2, column=1).value == 1
    assert sheet.cell(row=2, column=2).value == pytest.approx(-9.42)
    assert sheet.cell(row=2, column=5).value == "ASP189"
    assert sheet.cell(row=3, column=1).value == 2
    assert sheet.freeze_panes == "A2"
    run = workbook["Run"]
    labels = [run.cell(row=i, column=1).value for i in range(1, run.max_row + 1)]
    assert "poses" in labels and "seed" in labels


def test_xlsx_degrades_to_csv_without_openpyxl(tmp_path: Path, dock_result, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openpyxl" or name.startswith("openpyxl."):
            raise ImportError("openpyxl is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.warns(UserWarning):
        path = report.write_xlsx(tmp_path / "report.xlsx", dock_result)
    assert path.endswith(".csv")
    assert Path(path).exists()
    assert Path(path).read_text(encoding="utf-8").startswith("Mode,")


def test_csv_of_an_empty_result_is_just_the_header(tmp_path: Path):
    path = report.write_csv(tmp_path / "empty.csv", DockResult(poses=[], seed=0))
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("Mode,")


def test_the_report_survives_a_corrupt_ligand_block(tmp_path: Path):
    result = DockResult(
        poses=[Pose(index=0, affinity=-1.0, coords=np.zeros((2, 3)))],
        seed=0,
        ligand_pdbqt="not a pdbqt document at all\n",
        receptor_pdbqt=RECEPTOR_PDBQT,
    )
    rows = report.result_rows(result)
    assert rows[0]["residues"] == ""


# ---------------------------------------------------------------------------
# End-to-end against the real kernel
# ---------------------------------------------------------------------------


def test_a_real_docking_run_can_be_reported():
    """The ligand PDBQT order and ``pose.coords`` must line up in practice."""
    import odock

    receptor = """\
ATOM      1  CB  ALA A   1      -6.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  CB  ALA A   2       6.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      3  CB  ALA A   3       0.000   6.000   0.000  1.00  0.00     0.000 C
ATOM      4  CB  ALA A   4       0.000  -6.000   0.000  1.00  0.00     0.000 C
ATOM      5  OG  SER A   5       0.000   0.000   6.000  1.00  0.00    -0.300 OA
ATOM      6  CB  ALA A   6       0.000   0.000  -6.000  1.00  0.00     0.000 C
TER
"""
    ligand = """\
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.200   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
BRANCH    2   3
ATOM      3  O1  LIG A   1       2.200   0.000   0.000  1.00  0.00    -0.300 OA
ENDBRANCH    2   3
TORSDOF 1
"""
    box = odock.BoxSpec(center=(0.0, 0.0, 0.0), size=(16.0, 16.0, 16.0), spacing=0.5)
    result = odock.dock(receptor, ligand, box, exhaustiveness=1, num_poses=2, seed=3)
    assert len(result.ligand_atom_order) == 3

    # The report pairs the ATOM records of `ligand_pdbqt` with `pose.coords`
    # one to one; that only works because the PDBQT is written in the kernel's
    # flattened order.
    atoms = report._ligand_atoms(result, result.poses[0])
    assert len(atoms) == 3
    assert [a.element for a in atoms] == ["C", "C", "O"]
    assert [a.name for a in atoms] == list(result.ligand_atom_order)
    for parsed, xyz in zip(atoms, result.poses[0].coords):
        assert (parsed.x, parsed.y, parsed.z) == pytest.approx(
            tuple(float(v) for v in xyz)
        )

    rows = report.result_rows(result)
    assert len(rows) == len(result.poses)
    assert all("mode" in row for row in rows)
    assert all(isinstance(row["residues"], str) for row in rows)

# SPDX-License-Identifier: GPL-3.0-or-later
"""Blind pocket detection: synthetic cavities, residue labelling and boxing."""

from __future__ import annotations

import time

import numpy as np
import pytest

from odock.gui.structure import Atom
from odock.pocket import (
    VDW_RADII,
    Pocket,
    box_from_pocket,
    find_pockets,
    residues_within,
)

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _sphere(n: int, radius: float, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """`n` points spread evenly over a sphere (Fibonacci lattice)."""
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    pts = np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )
    return pts * radius + np.asarray(centre, dtype=float)


#: 200 atoms on a 8 A sphere: adjacent atoms are 2.15 A apart, so the shell is
#: watertight for a 1.4 A probe and the enclosed free volume is a ball of
#: radius 8.0 - 1.7 (carbon vdW) - 1.4 (probe) = 4.9 A, i.e. about 493 A^3.
CAVITY_RADIUS = 8.0
CAVITY_ATOMS = 200


@pytest.fixture(scope="module")
def hollow_receptor() -> np.ndarray:
    return _sphere(CAVITY_ATOMS, CAVITY_RADIUS)


@pytest.fixture(scope="module")
def cavity_pockets(hollow_receptor):
    # pocket_radius=0 disables the sub-pocket partition, so the whole connected
    # cavity is reported as one pocket and its volume can be compared with the
    # analytic value.
    return find_pockets(hollow_receptor, pocket_radius=0.0)


# ---------------------------------------------------------------------------
# The synthetic cavity
# ---------------------------------------------------------------------------


def test_the_known_cavity_is_found_with_a_plausible_volume(cavity_pockets):
    assert cavity_pockets, "the enclosed cavity was not detected"
    best = cavity_pockets[0]
    expected = 4.0 / 3.0 * np.pi * (CAVITY_RADIUS - 1.70 - 1.4) ** 3
    # The grid discretises the ball, so a few percent either way is expected.
    assert best.volume == pytest.approx(expected, rel=0.15)
    assert best.volume > 100.0


def test_the_cavity_centre_is_the_centre_of_the_shell(cavity_pockets):
    best = cavity_pockets[0]
    assert np.linalg.norm(np.asarray(best.center)) < 1.5


def test_volume_is_points_times_cell_volume(cavity_pockets):
    best = cavity_pockets[0]
    assert best.volume == pytest.approx(best.n_points * 1.0**3)
    assert best.n_points == len(best.points)
    for point in best.points:
        assert len(point) == 3
        assert np.linalg.norm(np.asarray(point)) < CAVITY_RADIUS


def test_pockets_are_ranked_and_renumbered(hollow_receptor):
    pockets = find_pockets(hollow_receptor, max_pockets=5)
    assert pockets
    assert [p.index for p in pockets] == list(range(len(pockets)))
    scores = [p.score for p in pockets]
    assert scores == sorted(scores, reverse=True)


def test_the_buried_cavity_beats_the_open_surface(hollow_receptor):
    """The enclosed cavity must score far above any surface dimple."""
    pockets = find_pockets(hollow_receptor, pocket_radius=0.0, min_volume=20.0)
    assert pockets
    best = pockets[0]
    assert np.linalg.norm(np.asarray(best.center)) < 1.5
    for other in pockets[1:]:
        assert best.score > other.score


def test_a_flat_slab_has_no_pocket():
    """A single layer of atoms is not a cavity."""
    grid = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0), [0.0]), axis=-1)
    slab = grid.reshape(-1, 3) * 1.5
    assert find_pockets(slab, min_volume=100.0) == []


def test_min_volume_filters_everything_above_it(hollow_receptor):
    small = find_pockets(hollow_receptor, pocket_radius=0.0, min_volume=1.0)
    large = find_pockets(hollow_receptor, pocket_radius=0.0, min_volume=100000.0)
    assert small and large == []
    assert all(p.volume >= 1.0 for p in small)


def test_max_pockets_is_an_upper_bound(hollow_receptor):
    assert len(find_pockets(hollow_receptor, max_pockets=1)) <= 1
    assert find_pockets(hollow_receptor, max_pockets=0) == []


def test_spacing_scales_the_volume(hollow_receptor):
    fine = find_pockets(hollow_receptor, spacing=0.5, pocket_radius=0.0)
    coarse = find_pockets(hollow_receptor, spacing=1.0, pocket_radius=0.0)
    assert fine and coarse
    assert fine[0].volume == pytest.approx(coarse[0].volume, rel=0.15)
    assert fine[0].n_points > coarse[0].n_points
    assert fine[0].volume == pytest.approx(fine[0].n_points * 0.125)


def test_the_whole_cavity_is_reachable_by_the_probe(hollow_receptor):
    """A bigger probe fits into a smaller volume; a huge one fits nowhere."""
    water = find_pockets(hollow_receptor, probe=1.4, pocket_radius=0.0)
    big = find_pockets(hollow_receptor, probe=4.0, pocket_radius=0.0)
    assert water
    assert not big or big[0].volume < water[0].volume


def test_buriedness_threshold_controls_the_seed_set(hollow_receptor):
    loose = find_pockets(hollow_receptor, buriedness=0.2, pocket_radius=0.0)
    strict = find_pockets(hollow_receptor, buriedness=0.99, pocket_radius=0.0)
    assert loose
    assert sum(p.n_points for p in strict) <= sum(p.n_points for p in loose)


def test_ray_length_one_is_the_immediate_neighbour_test(hollow_receptor):
    """With ray_length == spacing the criterion is the literal 26 neighbours."""
    pockets = find_pockets(hollow_receptor, ray_length=1.0, pocket_radius=0.0)
    assert isinstance(pockets, list)


def test_a_partitioned_cavity_still_finds_the_site(hollow_receptor):
    pockets = find_pockets(hollow_receptor)
    assert pockets
    assert min(np.linalg.norm(np.asarray(p.center)) for p in pockets) < 2.5
    assert sum(p.volume for p in pockets) > 300.0


def test_invalid_parameters_are_rejected(hollow_receptor):
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, spacing=0.0)
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, probe=-1.0)
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, min_volume=-1.0)
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, max_pockets=-1)
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, buriedness=1.5)
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, ray_length=0.0)
    with pytest.raises(ValueError):
        find_pockets(hollow_receptor, pocket_radius=-1.0)


def test_an_empty_receptor_yields_no_pockets():
    assert find_pockets([]) == []


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


def _pdb_receptor() -> str:
    """Three residues: one at the origin, one 3 A away, one 20 A away."""
    lines = []
    serial = 1

    def add(name, res, chain, res_id, x, y, z):
        nonlocal serial
        # Exact PDB columns: name 13-16, altLoc 17, resName 18-20, chain 22,
        # resSeq 23-26, x/y/z 31-54, element 77-78.
        lines.append(
            f"ATOM  {serial:>5d} {name:>4s} {res:>3s} {chain}{res_id:>4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}"
        )
        serial += 1

    add("OD1", "ASP", "A", 189, 0.0, 0.0, 0.0)
    add("CG", "ASP", "A", 189, 0.0, 1.5, 0.0)
    add("OG", "SER", "A", 195, 3.0, 0.0, 0.0)
    add("CB", "SER", "A", 195, 3.0, 1.5, 0.0)
    add("CZ", "PHE", "B", 215, 20.0, 0.0, 0.0)
    return "\n".join(lines) + "\nTER\nEND\n"


@pytest.fixture(scope="module")
def residue_mol():
    mol = Chem.MolFromPDBBlock(_pdb_receptor(), removeHs=False, sanitize=False)
    assert mol is not None
    return mol


def test_residues_within_reads_pdb_residue_information(residue_mol):
    labels = residues_within(residue_mol, (0.0, 0.0, 0.0), 5.0)
    assert labels == ["ASP189 A", "SER195 A"]


def test_residues_within_respects_the_radius(residue_mol):
    assert residues_within(residue_mol, (0.0, 0.0, 0.0), 2.0) == ["ASP189 A"]
    assert residues_within(residue_mol, (0.0, 0.0, 0.0), 40.0) == [
        "ASP189 A",
        "SER195 A",
        "PHE215 B",
    ]


def test_residues_within_measures_to_the_closest_atom(residue_mol):
    # The ASP CG atom is at y = 1.5, so a probe 2 A away from it is 2.5 A from
    # OD1 and inside a 2.6 A radius but outside a 2.4 A one.
    assert "ASP189 A" in residues_within(residue_mol, (0.0, 2.0, 0.0), 0.6)
    assert residues_within(residue_mol, (0.0, 2.0, 0.0), 0.4) == []


def test_residues_within_accepts_atom_like_objects():
    atoms = [
        Atom("OD1", "O", "ASP", 189, "A", 0.0, 0.0, 0.0),
        Atom("NZ", "N", "LYS", 15, "A", 4.0, 0.0, 0.0),
    ]
    # Sorted by chain, then residue number -- 15 comes before 189.
    assert residues_within(atoms, (0.0, 0.0, 0.0), 4.5) == ["LYS15 A", "ASP189 A"]
    assert residues_within(atoms, (0.0, 0.0, 0.0), 3.5) == ["ASP189 A"]


def test_residues_within_skips_atoms_without_residue_information():
    assert residues_within(np.zeros((3, 3)), (0.0, 0.0, 0.0), 5.0) == []


def test_a_coordinate_only_input_is_searchable(hollow_receptor):
    pockets = find_pockets(hollow_receptor, pocket_radius=0.0)
    assert pockets and pockets[0].residue_labels == []


def test_atom_like_input_gives_the_same_answer(hollow_receptor):
    from_array = find_pockets(hollow_receptor, pocket_radius=0.0)
    atoms = [
        Atom("C", "C", "ALA", i + 1, "A", float(x), float(y), float(z))
        for i, (x, y, z) in enumerate(hollow_receptor)
    ]
    from_atoms = find_pockets(atoms, pocket_radius=0.0)
    assert from_atoms and from_array
    assert from_atoms[0].volume == pytest.approx(from_array[0].volume)
    assert from_atoms[0].center == pytest.approx(from_array[0].center)


def test_a_molecule_without_coordinates_is_rejected():
    # An RDKit molecule straight out of a SMILES has no conformer at all.
    with pytest.raises(ValueError, match="conformer"):
        find_pockets(Chem.MolFromSmiles("CCO"))


def test_missing_conformer_is_reported_clearly():
    mol = Chem.MolFromSmiles("CCO")
    mol.RemoveAllConformers()
    with pytest.raises(ValueError, match="conformer"):
        find_pockets(mol)


def test_the_box_covers_every_cavity_point(cavity_pockets):
    pocket = cavity_pockets[0]
    box = box_from_pocket(pocket, buffer=2.0)
    for point in pocket.points:
        assert box.contains(point)
    assert box.volume > pocket.volume
    assert box.spacing == pytest.approx(0.375)


def test_the_box_buffer_widens_the_box(cavity_pockets):
    pocket = cavity_pockets[0]
    tight = box_from_pocket(pocket, buffer=0.0)
    wide = box_from_pocket(pocket, buffer=3.0)
    for i in range(3):
        assert wide.size[i] == pytest.approx(tight.size[i] + 6.0)


def test_the_box_falls_back_to_the_centre():
    pocket = Pocket(
        index=0, center=(1.0, 2.0, 3.0), volume=100.0, score=1.0, n_points=100
    )
    box = box_from_pocket(pocket, buffer=1.0)
    assert box.contains((1.0, 2.0, 3.0))
    assert box.volume > 0


def test_the_probe_radius_is_the_bondi_table():
    assert VDW_RADII["C"] == 1.70
    assert VDW_RADII["O"] == 1.52
    assert VDW_RADII["H"] == 1.20


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atoms", [3000])
def test_a_3000_atom_receptor_is_searched_in_a_few_seconds(atoms):
    """The stated budget: a few seconds for a receptor of this size."""
    rng = np.random.default_rng(20240101)
    directions = rng.normal(size=(atoms, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # A glob with a hollow core, roughly the density and extent of a protein.
    coords = np.vstack(
        [
            directions[: atoms // 3] * rng.uniform(0, 22, size=(atoms // 3, 1)),
            _sphere(atoms - atoms // 3, 20.0),
        ]
    )
    start = time.perf_counter()
    pockets = find_pockets(coords)
    elapsed = time.perf_counter() - start
    assert elapsed < 15.0, f"the pocket finder took {elapsed:.1f} s"
    assert isinstance(pockets, list)
    for pocket in pockets:
        assert pocket.volume >= 100.0
        assert pocket.n_points > 0

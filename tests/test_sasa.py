# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.sasa` — the numbers have to be hand-checkable.

Every test here either compares against a closed-form area (an isolated
sphere, a two-sphere cap intersection) or checks a property that must hold for
*any* correct SASA implementation (monotonicity under burial, invariance under
rigid motion, a buried fraction inside ``[0, 1]``). No test asserts a value
that was copied out of this implementation's output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from odock import sasa
from odock.gui.structure import Atom


def atom(element: str, x: float, y: float, z: float, **kwargs) -> Atom:
    """A minimal Atom-like record for the tests."""
    fields = {
        "res_name": "LIG",
        "res_id": 1,
        "chain": "A",
    }
    fields.update(kwargs)
    return Atom(name=f"{element}1", element=element, x=x, y=y, z=z, **fields)


# ---------------------------------------------------------------------------
# the sphere sampling and the radius table
# ---------------------------------------------------------------------------


def test_sphere_points_are_unit_vectors_and_reproducible():
    points = sasa.sphere_points(92)
    assert points.shape == (92, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    # Deterministic: the same call twice is bit-identical, which is what makes
    # a reported area reproducible.
    assert np.array_equal(points, sasa.sphere_points(92))
    assert sasa.sphere_points(1).tolist() == [[0.0, 0.0, 1.0]]
    with pytest.raises(ValueError):
        sasa.sphere_points(0)


def test_sphere_points_are_well_spread():
    """A near-uniform lattice: no point is closer than ~half the ideal spacing.

    The golden spiral exists to avoid the polar clustering of a lat/long grid,
    so the *minimum* separation is the property worth pinning.
    """
    points = sasa.sphere_points(400)
    # Pairwise minimum over a subsample would be O(n^2); the nearest neighbour
    # of each point is a fair proxy and 400 points is cheap.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    ideal = math.sqrt(4.0 * math.pi / 400)
    assert nearest.min() > 0.45 * ideal


def test_radius_of_uses_bondi_and_falls_back_to_carbon():
    assert sasa.radius_of("C") == pytest.approx(1.70)
    assert sasa.radius_of("o") == pytest.approx(1.52)
    assert sasa.radius_of("CL") == pytest.approx(1.75)
    assert sasa.radius_of("") == pytest.approx(sasa.DEFAULT_RADIUS)
    assert sasa.radius_of("Xx") == pytest.approx(sasa.DEFAULT_RADIUS)


# ---------------------------------------------------------------------------
# closed-form checks
# ---------------------------------------------------------------------------


def test_an_isolated_sphere_has_exactly_its_full_area():
    """A lone atom has every point accessible: A = 4 pi (r + probe)^2 exactly."""
    radius = 1.7
    area = sasa.sasa_per_atom([[0.0, 0.0, 0.0]], [radius])
    assert area[0] == pytest.approx(4.0 * math.pi * (radius + 1.4) ** 2, rel=1e-12)


def _two_sphere_exact(radius_a: float, radius_b: float, distance: float) -> tuple:
    """The analytic SASA of two spheres, in A^2.

    The probe spheres have radii ``R_i = r_i + probe``. They overlap when
    ``|R_a - R_b| < d < R_a + R_b``, and the part of sphere *a* that lies
    inside sphere *b* is a spherical cap of height

        h = R_a - (d^2 + R_a^2 - R_b^2) / (2 d)

    whose area is ``2 pi R_a h``. Subtracting it from the full sphere gives the
    accessible area — an exact closed form, so the numerical integration is
    being tested against mathematics rather than against itself.
    """
    probe = 1.4
    big_a = radius_a + probe
    big_b = radius_b + probe
    full_a = 4.0 * math.pi * big_a**2
    full_b = 4.0 * math.pi * big_b**2
    if distance >= big_a + big_b:
        return full_a, full_b
    if distance + big_a <= big_b:
        # Sphere a is entirely inside sphere b: none of a is reachable, and a
        # never reaches b's surface either.
        return 0.0, full_b
    if distance + big_b <= big_a:
        return full_a, 0.0  # pragma: no cover - symmetric case
    height_a = big_a - (distance**2 + big_a**2 - big_b**2) / (2.0 * distance)
    height_b = big_b - (distance**2 + big_b**2 - big_a**2) / (2.0 * distance)
    return (
        full_a - 2.0 * math.pi * big_a * height_a,
        full_b - 2.0 * math.pi * big_b * height_b,
    )


@pytest.mark.parametrize("distance", [2.6, 3.0, 3.6, 4.2])
def test_two_contacting_carbons_match_the_analytic_cap_area(distance):
    expected_a, expected_b = _two_sphere_exact(1.7, 1.7, distance)
    areas = sasa.sasa_per_atom(
        [[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        [1.7, 1.7],
        points=512,
    )
    # 512 points put the quadrature error of a cap boundary well under 1 %;
    # the tolerance is deliberately looser than the error measured here so a
    # legitimate numerical change does not make the test flaky.
    assert areas[0] == pytest.approx(expected_a, rel=0.01)
    assert areas[1] == pytest.approx(expected_b, rel=0.01)


def test_the_default_point_count_keeps_the_cap_error_small():
    """The default 92 points must be good enough for a real report.

    Measured against the same closed form, so raising or lowering
    :data:`odock.sasa.DEFAULT_POINTS` is a decision the test sees.
    """
    expected_a, _ = _two_sphere_exact(1.7, 1.7, 3.0)
    areas = sasa.sasa_per_atom([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], [1.7, 1.7])
    assert areas[0] == pytest.approx(expected_a, rel=0.03)


def test_a_swallowed_atom_has_no_accessibility():
    """A hydrogen inside a potassium ion is unreachable by solvent.

    The potassium's probe sphere (2.75 + 1.4 = 4.15 Å) contains the hydrogen's
    (1.20 + 1.4 = 2.60 Å) outright, so the hydrogen's area is exactly zero and
    the potassium keeps its full area — a sphere that swallows another does not
    lose any of its own surface.
    """
    areas = sasa.sasa_per_atom([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [1.20, 2.75])
    assert areas[0] == 0.0
    assert areas[1] == pytest.approx(4.0 * math.pi * (2.75 + 1.4) ** 2, rel=1e-12)


def test_two_atoms_almost_on_top_of_each_other_keep_the_analytic_lune():
    """A wrong implementation silently returns a cap instead of the lune.

    At 0.5 Å the two probe spheres overlap by almost nothing, so the
    accessible part of each is the *large* cap (height R + d/2) and the closed
    form is the one the test uses — a useful trap, because the naive
    ``4 pi R^2 - 2 pi R h`` with the height measured the wrong way round looks
    plausible and is off by a factor of two.
    """
    expected_a, expected_b = _two_sphere_exact(1.7, 1.7, 0.5)
    assert expected_a == pytest.approx(65.27, abs=0.05)
    areas = sasa.sasa_per_atom([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], [1.7, 1.7])
    assert areas[0] == pytest.approx(expected_a, rel=0.02)
    assert areas[1] == pytest.approx(expected_b, rel=0.02)


def test_different_radii_are_handled_per_atom():
    expected_a, expected_b = _two_sphere_exact(1.7, 1.9, 3.4)
    areas = sasa.sasa_per_atom(
        [[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]], [1.7, 1.9], points=512
    )
    assert areas[0] == pytest.approx(expected_a, rel=0.01)
    assert areas[1] == pytest.approx(expected_b, rel=0.01)


# ---------------------------------------------------------------------------
# properties that must hold for any correct implementation
# ---------------------------------------------------------------------------


def test_burying_an_atom_never_increases_its_sasa():
    """Monotonicity is what makes a buried fraction meaningful.

    An occluding atom can only remove accessible points, so the sequence of
    areas as a neighbour walks in from far away must be non-increasing.
    """
    centre = [0.0, 0.0, 0.0]
    previous = math.inf
    for distance in (12.0, 8.0, 6.0, 5.0, 4.0, 3.6, 3.2, 2.9, 2.6):
        areas = sasa.sasa_per_atom([centre, [distance, 0.0, 0.0]], [1.7, 1.7])
        assert areas[0] <= previous + 1e-9, distance
        previous = areas[0]
    assert previous < 4.0 * math.pi * (1.7 + 1.4) ** 2


def test_more_neighbours_never_add_area():
    base = [0.0, 0.0, 0.0]
    coords = [base]
    radii = [1.7]
    areas = [sasa.sasa_per_atom(coords, radii)[0]]
    for index in range(1, 7):
        angle = index * math.tau / 6.0
        coords.append([2.6 * math.cos(angle), 2.6 * math.sin(angle), 0.0])
        radii.append(1.7)
        areas.append(sasa.sasa_per_atom(coords, radii)[0])
    assert all(later <= earlier + 1e-9 for earlier, later in zip(areas, areas[1:]))
    # A fully coordinated carbon is tight around its neighbours but not zero.
    assert 0.0 < areas[-1] < areas[0]


def test_sasa_is_invariant_under_rigid_motion():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.2, 0.1], [2.9, -0.4, 0.9], [1.1, 1.6, -0.7]]
    )
    radii = [1.7, 1.52, 1.55, 1.8]
    reference = sasa.sasa_per_atom(coords, radii)

    # A rotation about an arbitrary axis plus a translation.
    axis = np.array([0.3, -0.7, 0.5])
    axis = axis / np.linalg.norm(axis)
    angle = 0.9
    rotation = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
    )
    moved = coords @ rotation.T + np.array([12.0, -4.0, 7.5])
    after = sasa.sasa_per_atom(moved, radii)
    # The sphere lattice is fixed in space while the atoms move, so a rigid
    # motion re-samples every cap boundary: the accessibility of a point that
    # sat exactly on a boundary can flip. The change is therefore bounded by
    # the area one point carries, which is the quadrature error the point
    # count controls — not by nothing, and not by a percent of the total.
    one_point = 4.0 * math.pi * (max(radii) + 1.4) ** 2 / sasa.DEFAULT_POINTS
    assert np.all(np.abs(after - reference) <= one_point + 1e-9)


def test_occlusion_can_come_from_a_second_set_that_is_not_scored():
    ligand = [[0.0, 0.0, 0.0]]
    receptor = [[0.0, 0.0, 2.6]]
    free = sasa.sasa_per_atom(ligand, [1.7])
    bound = sasa.sasa_per_atom(
        ligand, [1.7], extra_coords=np.asarray(receptor), extra_radii=[1.7]
    )
    assert bound[0] < free[0]
    assert bound.shape == (1,)


# ---------------------------------------------------------------------------
# residues and burial
# ---------------------------------------------------------------------------


def _peptide() -> list:
    """Three residues with different exposure: buried, middling, exposed."""
    atoms = []
    for index in range(6):
        atoms.append(
            atom("C", 3.8 * index, 0.0, 0.0, res_name="ALA", res_id=1)
        )
    # A lysine-like side chain sticking out into solvent.
    atoms.append(atom("N", 0.0, 3.0, 0.0, res_name="LYS", res_id=2))
    atoms.append(atom("C", 0.0, 4.5, 0.0, res_name="LYS", res_id=2))
    # A cysteine pair forming a disulfide, which buries both sulfurs.
    atoms.append(atom("S", 0.0, -1.0, 3.0, res_name="CYS", res_id=3))
    atoms.append(atom("S", 0.0, -1.0, 5.05, res_name="CYS", res_id=4))
    return atoms


def test_residue_sasa_groups_by_chain_number_and_name():
    atoms = _peptide()
    report = sasa.residue_sasa(atoms)
    keys = [item.key for item in report]
    assert keys == [
        ("A", 1, "ALA"),
        ("A", 2, "LYS"),
        ("A", 3, "CYS"),
        ("A", 4, "CYS"),
    ]
    assert sum(item.atoms for item in report) == len(atoms)
    per_atom = sasa.sasa_of_atoms(atoms)
    assert sum(item.area for item in report) == pytest.approx(float(per_atom.sum()))


def test_burial_is_a_fraction_between_zero_and_one():
    atoms = _peptide()
    report = sasa.burial(atoms, reference="free")
    assert report.reference == "free"
    assert report.total_area > 0.0
    assert report.reference_total >= report.total_area
    for item in report.residues:
        assert item.buried_fraction is not None
        assert -1e-9 <= item.buried_fraction <= 1.0 + 1e-9
        assert item.buried_area is not None and item.buried_area >= 0.0


def test_a_reference_of_free_residue_makes_a_buried_residue_buried():
    """The disulfide sulfur pair buries each other; the free reference sees it."""
    atoms = _peptide()
    report = sasa.burial(atoms, reference="free")
    by_label = {item.label: item for item in report.residues}
    # CYS3 and CYS4 are isolated two-atom residues whose sulfurs touch, so a
    # meaningful share of the residue's own surface is covered by its partner.
    for label in ("A/CYS3", "A/CYS4"):
        assert by_label[label].buried_fraction > 0.05


def test_unbound_reference_measures_what_the_partner_hides():
    receptor = _peptide()
    # A ligand parked right on top of the lysine side chain.
    ligand = [atom("C", 0.0, 4.5, 2.6, res_name="LIG")]
    report = sasa.burial(receptor, ligand, reference="unbound")
    assert report.reference == "unbound"
    assert report.partner_atoms == 1
    assert report.buried_area > 0.0
    lysine = next(item for item in report.residues if item.res_name == "LYS")
    assert lysine.buried_area is not None and lysine.buried_area > 0.0

    # With no partner at all the unbound reference equals the complex, so
    # nothing is buried — the reference cannot invent burial.
    alone = sasa.burial(receptor, reference="unbound")
    assert alone.buried_area == pytest.approx(0.0, abs=1e-9)


def test_most_buried_is_sorted_and_limited():
    receptor = _peptide()
    ligand = [atom("C", 0.0, 4.5, 2.6, res_name="LIG")]
    report = sasa.burial(receptor, ligand, reference="unbound")
    ranked = report.most_buried(2)
    assert len(ranked) <= 2
    values = [item.buried_area for item in ranked]
    assert values == sorted(values, reverse=True)
    assert "buried" in report.table().splitlines()[0]
    assert report.as_dict()["reference"] == "unbound"


def test_burial_rejects_an_unknown_reference():
    with pytest.raises(ValueError):
        sasa.burial(_peptide(), reference="sideways")


# ---------------------------------------------------------------------------
# the ligand's buried contact area
# ---------------------------------------------------------------------------


def _pocket() -> list:
    """A crude pocket: a ring of carbons with a hole in the middle."""
    atoms = []
    for index in range(10):
        angle = index * math.tau / 10.0
        atoms.append(atom("C", 5.0 * math.cos(angle), 5.0 * math.sin(angle), 0.0))
    return atoms


def test_a_ligand_in_a_pocket_buries_more_than_the_same_ligand_far_away():
    receptor = _pocket()          # a 5 A ring: the wall touches a central atom
    inside = [atom("C", 0.0, 0.0, 0.0, res_name="LIG")]
    outside = [atom("C", 60.0, 0.0, 0.0, res_name="LIG")]
    buried_inside = sasa.ligand_buried_contact_area(inside, receptor)
    buried_outside = sasa.ligand_buried_contact_area(outside, receptor)
    # 60 A away nothing can reach: the buried area is exactly zero, which is
    # the control that makes the positive number below mean something.
    assert buried_outside["buried_area"] == pytest.approx(0.0, abs=1e-9)
    # A central atom is surrounded by the wall, so most of its surface is
    # hidden: the ring atom centres are 5 A out and the probe spheres are
    # 3.1 A each, so they interpenetrate the ligand's sphere.
    assert buried_inside["buried_area"] > 0.0
    assert buried_inside["buried_fraction"] > 0.2
    assert 0.0 < buried_inside["exposed_fraction"] < 1.0

    touching = [atom("C", 3.0, 0.0, 0.0, res_name="LIG")]
    buried = sasa.ligand_buried_contact_area(touching, receptor)
    assert buried["buried_area"] > 0.0
    assert 0.0 < buried["buried_fraction"] < 1.0
    assert buried["free_area"] == pytest.approx(buried["complex_area"] + buried["buried_area"])
    assert buried["contact_atoms"] >= 1


def test_the_buried_contact_area_is_consistent_with_a_direct_difference():
    receptor = _pocket()
    ligand = [
        atom("C", 2.8, 0.0, 0.0, res_name="LIG"),
        atom("O", 4.0, 0.0, 0.0, res_name="LIG"),
    ]
    record = sasa.ligand_buried_contact_area(ligand, receptor)
    free = sasa.sasa_of_atoms(ligand).sum()
    bound = sasa.sasa_of_atoms(ligand, other=receptor).sum()
    assert record["free_area"] == pytest.approx(float(free))
    assert record["complex_area"] == pytest.approx(float(bound))
    assert record["buried_area"] == pytest.approx(float(free - bound))


def test_interface_area_matches_the_whole_protein_and_scales_with_a_shell():
    receptor = _pocket() + [atom("C", 40.0, 0.0, 0.0)]
    ligand = [atom("C", 2.8, 0.0, 0.0, res_name="LIG")]
    whole = sasa.interface_area(receptor, ligand)
    shell = sasa.interface_area(receptor, ligand, radius=8.0)
    assert whole["buried_area"] > 0.0
    # The far carbon contributes nothing, so a shell that contains the contact
    # gives the same answer — which is what makes the shell an exact
    # optimisation rather than an approximation.
    assert shell["buried_area"] == pytest.approx(whole["buried_area"], rel=1e-9)
    assert shell["receptor_atoms_used"] < shell["receptor_atoms_total"]


def test_buried_contact_per_pose_carries_the_index_and_the_score():
    receptor = _pocket()
    poses = [
        [atom("C", 2.8, 0.0, 0.0, res_name="LIG")],
        [atom("C", 30.0, 0.0, 0.0, res_name="LIG")],
    ]
    records = sasa.buried_contact_per_pose(poses, receptor, affinity=[-7.5, -6.0])
    assert [item["pose"] for item in records] == [0.0, 1.0]
    assert [item["affinity"] for item in records] == [-7.5, -6.0]
    assert records[0]["buried_area"] > records[1]["buried_area"]


def test_the_pose_burial_figure_is_valid_svg_with_one_bar_per_pose():
    """The figure is generated, so it is checked as XML rather than by eye."""
    import xml.etree.ElementTree as ElementTree

    receptor = _pocket()
    poses = [
        [atom("C", 2.8, 0.0, 0.0, res_name="LIG")],
        [atom("C", 3.4, 0.0, 0.0, res_name="LIG")],
        [atom("C", 30.0, 0.0, 0.0, res_name="LIG")],
    ]
    records = sasa.buried_contact_per_pose(poses, receptor, affinity=[-7.5, -7.0, -6.0])
    svg = sasa.pose_burial_svg(records, title="buried <per> pose & more")
    root = ElementTree.fromstring(svg)
    assert root.tag.endswith("svg")
    rectangles = [node for node in root.iter() if node.tag.endswith("rect")]
    # One background plus two per bar (the bar and its highlight cap).
    assert len(rectangles) == 1 + 2 * len(records)
    texts = [node.text or "" for node in root.iter() if node.tag.endswith("text")]
    assert "buried &lt;per&gt; pose &amp; more" not in texts  # escaped, not literal
    assert "buried <per> pose & more" in texts
    assert "1" in texts and "3" in texts          # the pose numbers
    assert any(text == "-7.5" for text in texts)  # the scores
    # The tallest bar belongs to the most buried pose, and the far-away ligand
    # gets no bar height at all.
    heights = [float(node.get("height", "0")) for node in rectangles[1::2]]
    assert max(heights) > 0.0
    assert min(heights) == pytest.approx(0.0)


def test_the_pose_burial_figure_handles_an_empty_pose_set():
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.fromstring(sasa.pose_burial_svg([]))
    assert root.tag.endswith("svg")
    # No bars, but the axes are still there: a caller never has to special-case
    # "nothing to plot".
    assert sum(1 for node in root.iter() if node.tag.endswith("line")) > 3


def test_sasa_summary_counts_exposed_and_buried_atoms():
    atoms = [atom("K", 0.0, 0.0, 0.0), atom("H", 0.3, 0.0, 0.0)]
    summary = sasa.sasa_summary(atoms)
    assert summary["atoms"] == 2
    assert summary["exposed_atoms"] == 1        # the ion keeps its surface
    assert summary["buried_atoms"] == 1         # the hydrogen is inside it
    assert summary["total"] == pytest.approx(
        4.0 * math.pi * (sasa.radius_of("K") + 1.4) ** 2
    )


def test_an_empty_structure_is_an_empty_answer():
    assert sasa.sasa_per_atom([], []).size == 0
    assert sasa.sasa_of_atoms([]).size == 0
    assert sasa.residue_sasa([]) == []
    assert sasa.ligand_buried_contact_area([], [])["buried_area"] == 0.0
    assert sasa.sasa_summary([])["atoms"] == 0


def test_a_mismatched_radius_list_is_loud():
    with pytest.raises(ValueError):
        sasa.sasa_per_atom([[0.0, 0.0, 0.0]], [1.7, 1.5])
    with pytest.raises(ValueError):
        sasa.sasa_per_atom(
            [[0.0, 0.0, 0.0]],
            [1.7],
            extra_coords=[[1.0, 0.0, 0.0]],
            extra_radii=[1.7, 1.5],
        )
    with pytest.raises(ValueError):
        sasa.sasa_per_atom([[0.0, 0.0, 0.0]], [1.7], probe=-1.0)


def test_atom_objects_are_read_through_their_element():
    """The Atom path and the raw-array path must agree exactly."""
    atoms = [atom("C", 0.0, 0.0, 0.0), atom("O", 2.6, 0.0, 0.0)]
    through_atoms = sasa.sasa_of_atoms(atoms)
    through_arrays = sasa.sasa_per_atom(
        [[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]], [1.70, 1.52]
    )
    assert np.allclose(through_atoms, through_arrays)

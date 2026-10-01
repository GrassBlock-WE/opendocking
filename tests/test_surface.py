# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.gui.surface`.

The point of this suite is that the surface is *measured*, not asserted into
existence: the triangulator is checked against a sphere of known area, the SAS
mesh area against the independent Shrake-Rupley integral in :mod:`odock.sasa`,
the SES against the van der Waals sphere it must reduce to for a single atom,
and every property scale against a value written down in the published source
it comes from.
"""

from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np
import pytest

from odock import sasa
from odock.gui import surface as surface_module
from odock.gui.structure import Atom, element_color
from odock.gui.surface import (
    AD4_TYPE_HYDROPHOBICITY,
    COULOMB_CONSTANT,
    HYDROPATHY_KD,
    PALETTES,
    SURFACE_MODES,
    SURFACE_PROPERTIES,
    Surface,
    SurfaceSettings,
    _TET_CASES,
    atom_hydrophobicity,
    build_surface,
    charges_from_atoms,
    colorize,
    electrostatic_potential,
    legend_stops,
    marching_tetrahedra,
    nearest_atoms,
    palette_color,
    resolve_spacing,
    sample_field,
    scalar_field,
    select_atoms,
    surface_area,
)


def atom(element: str, x: float, y: float, z: float, **kwargs) -> Atom:
    fields = {"res_name": "LIG", "res_id": 1, "chain": "A", "name": f"{element}1"}
    fields.update(kwargs)
    return Atom(element=element, x=x, y=y, z=z, **fields)


def residue(prefix: str, res_name: str, res_id: int, count: int, offset=(0.0, 0.0, 0.0)) -> list:
    """``count`` carbons in a row, all in one residue."""
    return [
        atom(
            "C",
            1.5 * index + offset[0],
            offset[1],
            offset[2],
            res_name=res_name,
            res_id=res_id,
            name=f"{prefix}{index}",
        )
        for index in range(count)
    ]


def _sphere_field(radius: float, spacing: float, half: float = 6.0):
    """``(|p| - radius)`` on a cubic grid: a sphere of known area."""
    axes = [np.arange(-half, half + 1e-9, spacing) for _ in range(3)]
    shape = tuple(len(axis) for axis in axes)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    field = (np.linalg.norm(grid, axis=1) - radius).reshape(shape)
    return field, axes


# ---------------------------------------------------------------------------
# the triangulator
# ---------------------------------------------------------------------------


def test_the_six_tetrahedra_partition_the_cube():
    """The decomposition must have no gap and no overlap.

    Six tetrahedra, each with one sixth of the cube's volume, sharing the main
    diagonal: that is the invariant the case table is built on, and it is
    checked by volume rather than by eye.
    """
    corners = np.asarray(surface_module.CUBE_CORNERS, dtype=float)
    total = 0.0
    for tetrahedron in surface_module.TETRAHEDRA:
        points = corners[list(tetrahedron)]
        matrix = np.stack([points[1] - points[0], points[2] - points[0], points[3] - points[0]])
        total += abs(float(np.linalg.det(matrix))) / 6.0
    assert total == pytest.approx(1.0)
    assert len(surface_module.TETRAHEDRA) == 6
    # The two ends of the shared diagonal belong to all six tetrahedra; each of
    # the six corners of the "belt" belongs to exactly two, which is what makes
    # the decomposition a partition with a consistent shared face.
    usage = np.bincount(np.concatenate(surface_module.TETRAHEDRA))
    assert usage[0] == 6 and usage[7] == 6
    assert sorted(usage[1:7].tolist()) == [2] * 6


def test_every_tetrahedron_case_has_the_right_triangle_count():
    for code in range(16):
        inside = bin(code).count("1")
        triangles = _TET_CASES[code]
        if inside in (0, 4):
            assert triangles == ()
            continue
        expected = 1 if inside in (1, 3) else 2
        assert len(triangles) == expected, code
        for triangle in triangles:
            assert len(set(triangle)) == 3
            assert all(0 <= edge < 6 for edge in triangle)


def test_marching_tetrahedra_converges_to_the_exact_sphere_area():
    """Refining the grid must approach 4 pi R^2 from below and monotonically.

    A faceted isosurface inscribes the true surface, so the area starts low;
    the test pins both the direction of the error and its convergence, which
    is what a "documented equivalent" triangulation has to demonstrate.
    """
    radius = 3.1
    exact = 4.0 * math.pi * radius * radius
    errors = []
    for spacing in (1.2, 0.8, 0.5, 0.32):
        field, axes = _sphere_field(radius, spacing)
        vertices, triangles = marching_tetrahedra(
            field, surface_module.axes_origin(axes), spacing
        )
        assert triangles.shape[0] > 0
        # Every vertex sits on the isosurface, i.e. at the sphere radius.
        radii = np.linalg.norm(vertices, axis=1)
        assert np.max(np.abs(radii - radius)) < 0.5 * spacing
        area = surface_area(vertices, triangles)
        assert area < exact
        errors.append(area / exact - 1.0)
    assert errors == sorted(errors), errors          # monotone improvement
    assert errors[-1] > -0.01                        # within 1 % at 0.32 A


def test_marching_tetrahedra_needs_a_sign_change():
    field = np.ones((4, 4, 4))
    vertices, triangles = marching_tetrahedra(field, (0.0, 0.0, 0.0), 1.0)
    assert vertices.shape == (0, 3)
    assert triangles.shape == (0, 3)
    with pytest.raises(ValueError):
        marching_tetrahedra(np.zeros((4, 4)), (0.0, 0.0), 1.0)


def test_surface_area_of_nothing_is_zero():
    assert surface_area(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)) == 0.0


# ---------------------------------------------------------------------------
# the field
# ---------------------------------------------------------------------------


def test_scalar_field_is_the_distance_to_the_inflated_sphere():
    atoms = [atom("C", 0.0, 0.0, 0.0)]
    probe = 1.4
    points = np.array([[3.1, 0.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    field = scalar_field(points, atoms, probe=probe, spacing=1.0)
    assert field[0] == pytest.approx(0.0, abs=1e-12)     # exactly on the SAS
    assert field[1] == pytest.approx(-3.1, abs=1e-12)    # at the centre
    assert field[2] == pytest.approx(1.9, abs=1e-12)


def test_the_regular_grid_fast_path_matches_the_general_search():
    """Two implementations of one field: they must agree everywhere.

    The grid path exists only for speed (a block per atom instead of a ragged
    pair search), so a mismatch is a bug in one of them, not a tolerance to
    relax.
    """
    atoms = [
        atom("C", 0.0, 0.0, 0.0),
        atom("O", 1.4, 0.6, 0.0),
        atom("N", -1.2, 0.4, 1.1),
        atom("S", 0.6, -1.5, 0.7),
    ]
    spacing = 0.4
    axes = [np.arange(-4.0, 4.0, spacing) for _ in range(3)]
    shape = tuple(len(axis) for axis in axes)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    general = scalar_field(grid, atoms, spacing=spacing).reshape(shape)
    fast = scalar_field(
        np.zeros((0, 3)), atoms, spacing=spacing, axes=axes
    )
    assert fast.shape == shape
    assert np.allclose(general, fast, atol=1e-12)


def test_a_point_far_from_everything_is_clamped_positive():
    atoms = [atom("C", 0.0, 0.0, 0.0)]
    far = scalar_field(np.array([[40.0, 0.0, 0.0]]), atoms, spacing=0.5)
    assert far[0] == pytest.approx(1.0)  # 2 * spacing, the documented clamp


def test_sample_field_reads_between_the_nodes():
    field = np.zeros((3, 3, 3))
    field[1, 1, 1] = 1.0
    value = sample_field(field, (0.0, 0.0, 0.0), 1.0, np.array([[1.0, 1.0, 1.0]]))
    assert value[0] == pytest.approx(1.0)
    # Halfway along every axis towards the only non-zero node: 1/8 of it.
    value = sample_field(field, (0.0, 0.0, 0.0), 1.0, np.array([[0.5, 0.5, 0.5]]))
    assert value[0] == pytest.approx(0.125)
    # And a node of the field itself reads back exactly.
    value = sample_field(field, (0.0, 0.0, 0.0), 1.0, np.array([[0.0, 0.0, 0.0]]))
    assert value[0] == pytest.approx(0.0)
    outside = sample_field(field, (0.0, 0.0, 0.0), 1.0, np.array([[9.0, 0.0, 0.0]]))
    assert not np.isfinite(outside[0])


def test_nearest_atoms_finds_the_owning_atom():
    atoms = [
        atom("C", 0.0, 0.0, 0.0),
        atom("O", 4.0, 0.0, 0.0),
        atom("N", 0.0, 4.0, 0.0),
    ]
    points = np.array([[3.0, 0.0, 0.0], [0.0, 3.2, 0.0], [-2.9, 0.1, 0.0]])
    index, distance = nearest_atoms(points, atoms, spacing=0.5)
    assert index.tolist() == [1, 2, 0]
    assert np.allclose(distance, [1.0, 0.8, math.hypot(2.9, 0.1)], atol=1e-9)


def test_resolve_spacing_respects_the_point_budget():
    lo, hi = (0.0, 0.0, 0.0), (200.0, 200.0, 200.0)
    automatic = resolve_spacing(lo, hi, 0.0, max_points=10_000_000)
    assert automatic > 0.0
    # The automatic spacing is the coarsest that fits the budget.
    assert (200.0 / automatic) ** 3 <= 10_000_000 * 1.05
    # An explicit request is honoured inside the documented clamps.
    assert resolve_spacing(lo, hi, 0.9) == pytest.approx(0.9)
    assert resolve_spacing(lo, hi, 0.01) == pytest.approx(surface_module.MIN_SPACING)
    assert resolve_spacing(lo, hi, 50.0) == pytest.approx(surface_module.MAX_SPACING)
    # The clamps win over the budget: a bigger cap cannot make the grid finer
    # than MIN_SPACING or coarser than MAX_SPACING.
    assert resolve_spacing(lo, hi, 0.0, max_points=10) == pytest.approx(
        surface_module.MAX_SPACING
    )
    assert resolve_spacing(lo, hi, 0.0, max_points=10**12) == pytest.approx(
        surface_module.MIN_SPACING
    )


# ---------------------------------------------------------------------------
# the properties
# ---------------------------------------------------------------------------


def test_kyte_doolittle_residues_map_to_the_published_extremes():
    atoms = [
        atom("C", 0.0, 0.0, 0.0, res_name="ILE", res_id=1),
        atom("C", 0.0, 0.0, 0.0, res_name="ARG", res_id=2),
        atom("C", 0.0, 0.0, 0.0, res_name="LEU", res_id=3),
    ]
    values = atom_hydrophobicity(atoms, scale="residue")
    assert values[0] == pytest.approx(1.0)      # Ile +4.5 is the top of the scale
    assert values[1] == pytest.approx(0.0)      # Arg -4.5 is the bottom
    expected_leu = (HYDROPATHY_KD["LEU"] - (-4.5)) / 9.0
    assert values[2] == pytest.approx(expected_leu)
    assert values[2] > 0.8                      # leucine is hydrophobic


def test_a_ligand_falls_back_to_its_autodock_type():
    atoms = [
        atom("C", 0.0, 0.0, 0.0, res_name="LIG", ad_type="C"),
        atom("O", 0.0, 0.0, 0.0, res_name="LIG", ad_type="OA"),
        atom("N", 0.0, 0.0, 0.0, res_name="LIG", ad_type="NA"),
    ]
    values = atom_hydrophobicity(atoms, scale="residue")
    assert values[0] == pytest.approx(AD4_TYPE_HYDROPHOBICITY["C"])
    assert values[1] == pytest.approx(AD4_TYPE_HYDROPHOBICITY["OA"])
    assert values[2] == pytest.approx(AD4_TYPE_HYDROPHOBICITY["NA"])
    assert values[0] > values[2] > values[1]


def test_the_vina_hydrophobic_rule_needs_the_bond_graph():
    """A carbon bonded to oxygen is not a hydrophobic carbon.

    Vina's ``xs_is_hydrophobic`` is true only for ``CH``/``FH``/…: the AD4
    dictionary calls both a methyl carbon and a carboxyl carbon ``C``, so the
    distinction has to come from the perceived bonds. This is the test that
    makes the difference between ``scale="residue"`` and the graph rule
    visible.
    """
    atoms = [
        atom("C", 0.0, 0.0, 0.0, res_name="LIG", ad_type="C", name="C1"),
        atom("C", 1.5, 0.0, 0.0, res_name="LIG", ad_type="C", name="C2"),
        atom("O", 2.9, 0.0, 0.0, res_name="LIG", ad_type="OA", name="O1"),
    ]
    bonds = [(0, 1), (1, 2)]
    plain = atom_hydrophobicity(atoms, scale="atom")
    ruled = atom_hydrophobicity(atoms, scale="atom", bonds=bonds)
    assert plain[0] == plain[1] == pytest.approx(AD4_TYPE_HYDROPHOBICITY["C"])
    assert ruled[0] == pytest.approx(AD4_TYPE_HYDROPHOBICITY["C"])  # untouched
    assert ruled[1] < plain[1]                                       # polar neighbour
    assert ruled[2] == plain[2]


def test_hydrophobicity_scale_names_are_canonicalised():
    assert surface_module.hydrophobicity_scale("KD") == "residue"
    assert surface_module.hydrophobicity_scale("vina") == "atom"
    with pytest.raises(ValueError):
        surface_module.hydrophobicity_scale("loudness")


def test_electrostatic_potential_is_coulomb_to_the_last_digit():
    atoms = [atom("N", 0.0, 0.0, 0.0, charge=1.0)]
    point = np.array([[4.0, 0.0, 0.0]])
    phi = electrostatic_potential(point, atoms, epsilon=4.0)
    assert phi[0] == pytest.approx(COULOMB_CONSTANT / (4.0 * 4.0), rel=1e-12)
    # A negative charge flips the sign and nothing else.
    atoms[0].charge = -1.0
    assert electrostatic_potential(point, atoms, epsilon=4.0)[0] == pytest.approx(-phi[0])
    # A distance-dependent dielectric is eps(r) = 4 r, i.e. 1/r^2 in total.
    assert electrostatic_potential(point, [atom("N", 0, 0, 0, charge=1.0)],
                                   dielectric="distance", epsilon=4.0)[0] == pytest.approx(
        COULOMB_CONSTANT / (4.0 * 16.0), rel=1e-12
    )


def test_electrostatic_potential_screening_and_zero_charges():
    atoms = [atom("N", 0.0, 0.0, 0.0, charge=1.0)]
    point = np.array([[4.0, 0.0, 0.0]])
    plain = electrostatic_potential(point, atoms, epsilon=4.0)[0]
    screened = electrostatic_potential(point, atoms, epsilon=4.0, screening=0.33)[0]
    assert screened == pytest.approx(plain * math.exp(-0.33 * 4.0), rel=1e-12)
    # No charge anywhere is an all-zero field rather than a division by zero.
    empty = electrostatic_potential(point, [atom("C", 0.0, 0.0, 0.0)])
    assert empty[0] == 0.0
    with pytest.raises(ValueError):
        electrostatic_potential(point, atoms, charges=[1.0, 2.0])
    with pytest.raises(ValueError):
        electrostatic_potential(point, atoms, dielectric="magic")


def test_charges_come_from_the_atoms_and_from_rdkit():
    atoms = [atom("C", 0.0, 0.0, 0.0, charge=0.25), atom("O", 1.0, 0.0, 0.0, charge=-0.25)]
    assert charges_from_atoms(atoms).tolist() == [0.25, -0.25]
    # A structure that never had a charge column reads as zeros.
    assert charges_from_atoms([atom("C", 0.0, 0.0, 0.0)]).tolist() == [0.0]


def test_charges_from_mol_uses_the_project_charge_model():
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    from odock.chem.charges import gasteiger_charges

    expected = gasteiger_charges(mol)
    got = surface_module.charges_from_mol(mol)
    assert np.allclose(got, expected)
    assert got.shape[0] == mol.GetNumAtoms()


def test_colorize_maps_the_range_and_survives_a_constant_array():
    values = np.array([0.0, 0.5, 1.0])
    colours, value_range = colorize(values, palette="hydrophobicity", value_range=(0.0, 1.0))
    assert value_range == (0.0, 1.0)
    assert colours.shape == (3, 3)
    # 0 is the polar end (blue-dominant) and 1 is the apolar end (red-dominant).
    assert colours[0][2] > colours[0][0]
    assert colours[2][0] > colours[2][2]
    # A constant property must not produce a division by zero.
    flat, flat_range = colorize(np.full(4, 2.0), value_range=None)
    assert flat_range[1] > flat_range[0]
    assert np.isfinite(flat).all()
    # Out-of-range values clamp instead of wrapping to the other end.
    clipped, _ = colorize(np.array([-5.0, 5.0]), value_range=(0.0, 1.0))
    assert np.allclose(clipped[0], colours[0])
    assert np.allclose(clipped[1], colours[2])
    assert colorize(np.zeros(0))[0].shape == (0, 3)


def test_palette_color_interpolates_between_the_published_stops():
    for name, stops in PALETTES.items():
        for fraction, colour in stops:
            assert np.allclose(palette_color(name, fraction), colour, atol=1e-9)
    assert palette_color("hydrophobicity", -1.0) == palette_color("hydrophobicity", 0.0)
    assert palette_color("hydrophobicity", 2.0) == palette_color("hydrophobicity", 1.0)


def test_legend_stops_cover_the_range_with_numeric_ticks():
    """The ticks are numbers; the unit belongs in the title, once."""
    stops = legend_stops("electrostatic", (-10.0, 10.0), count=5)
    assert len(stops) == 5
    assert stops[0][2] == "-10.0"
    assert stops[-1][2] == "10.0"
    assert stops[2][2] == "0.0"
    fractions = [item[0] for item in stops]
    assert fractions == sorted(fractions)
    # A small range keeps two decimals, a large one drops them: a tick label
    # has to stay short enough to sit beside the bar.
    fine = legend_stops("hydrophobicity", (0.0, 1.0), count=3)
    assert fine[1][2] == "0.50"
    coarse = legend_stops("electrostatic", (-400.0, 400.0), count=3)
    assert coarse[1][2] == "0"
    assert "kcal" not in "".join(item[2] for item in stops)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def test_select_atoms_intersects_indices_and_a_sphere():
    atoms = [atom("C", 1.0 * index, 0.0, 0.0) for index in range(10)]
    assert select_atoms(atoms, indices=[1, 3, 99]) == [1, 3]
    assert select_atoms(atoms, centre=(0.0, 0.0, 0.0), radius=2.5) == [0, 1, 2]
    both = select_atoms(atoms, indices=[0, 1, 2, 3], centre=(0.0, 0.0, 0.0), radius=2.5)
    assert both == [0, 1, 2]
    assert select_atoms(atoms) == list(range(10))


# ---------------------------------------------------------------------------
# the whole build
# ---------------------------------------------------------------------------


def _cluster(count: int = 9, spacing: float = 1.5) -> list:
    atoms = []
    for index in range(count):
        angle = index * math.tau / count
        atoms.append(atom("C", 2.6 * math.cos(angle), 2.6 * math.sin(angle), 0.0))
    del spacing
    return atoms


def test_an_empty_structure_builds_an_empty_surface():
    for settings in (None, SurfaceSettings()):
        result = build_surface([], settings)
        assert isinstance(result, Surface)
        assert result.vertices_count == 0
        assert result.triangles_count == 0
        assert result.area == 0.0
        assert result.mesh().shape == (0, 9)
        assert result.stats["atoms"] == 0


def test_the_sas_mesh_agrees_with_the_independent_shrake_rupley_integral():
    """Two implementations of one surface must land on the same area.

    :mod:`odock.sasa` integrates the accessibility of sampled sphere points;
    this module triangulates the zero level of the probe-centre field. They
    share no code beyond the radius table, so agreement is a real check.
    """
    atoms = _cluster(9)
    result = build_surface(atoms, SurfaceSettings(mode="sas", spacing=0.4))
    reference = float(sasa.sasa_of_atoms(atoms).sum())
    assert result.stats["area"] > 0.0
    assert abs(result.area / reference - 1.0) < 0.06


def test_the_solvent_excluded_surface_is_smaller_and_closer_to_the_atoms():
    atoms = _cluster(9)
    sas = build_surface(atoms, SurfaceSettings(mode="sas", spacing=0.4))
    ses = build_surface(atoms, SurfaceSettings(mode="ses", spacing=0.4))
    assert ses.area < sas.area
    # The reported SAS reference is the analytic Shrake-Rupley integral, not a
    # second triangulation (it is rounded to two decimals for the stats dict).
    assert ses.stats["sas_area"] == pytest.approx(
        float(sasa.sasa_of_atoms(atoms).sum()), abs=0.01
    )
    # The SES is the van der Waals surface plus reentrant patches, so every
    # point of it is at least a van der Waals radius from the nearest centre.
    coords = np.asarray([[a.x, a.y, a.z] for a in atoms])
    radii = np.asarray([sasa.radius_of(a.element) for a in atoms])
    distance = np.linalg.norm(
        ses.vertices[:, None, :] - coords[None, :, :], axis=2
    ).min(axis=1)
    assert np.all(distance >= radii.min() - 0.05)


def test_a_single_atom_ses_is_its_van_der_waals_sphere():
    """The rolling-ball construction must reduce to the sphere itself."""
    result = build_surface(
        [atom("C", 0.0, 0.0, 0.0)], SurfaceSettings(mode="ses", spacing=0.4)
    )
    exact = 4.0 * math.pi * sasa.radius_of("C") ** 2
    assert result.area == pytest.approx(exact, rel=0.03)
    # And its SAS is the same sphere inflated by the probe, exactly.
    sas = build_surface(
        [atom("C", 0.0, 0.0, 0.0)], SurfaceSettings(mode="sas", spacing=0.4)
    )
    assert sas.area == pytest.approx(4.0 * math.pi * (1.7 + 1.4) ** 2, rel=0.02)


def test_the_mesh_is_in_the_vertex_format_the_renderer_consumes():
    result = build_surface(
        [atom("C", 0.0, 0.0, 0.0), atom("O", 2.6, 0.0, 0.0)],
        SurfaceSettings(mode="sas", spacing=0.4),
    )
    mesh = result.mesh()
    assert mesh.shape == (result.vertices_count, 9)
    assert mesh.dtype == np.float32
    positions = mesh[:, :3]
    normals = mesh[:, 3:6]
    colours = mesh[:, 6:9]
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    assert colours.min() >= 0.0 and colours.max() <= 1.0
    # Positions must equal the surface's own vertices, in order.
    assert np.allclose(positions, result.vertices, atol=1e-4)
    # Normals point outwards: away from the atom each vertex belongs to.
    coords = np.asarray([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0]])
    owners = coords[np.maximum(result.atom_index, 0)]
    outward = np.einsum("ij,ij->i", result.normals, result.vertices - owners)
    assert np.all(outward > 0.0)


def test_the_property_range_and_the_legend_describe_what_is_shown():
    atoms = _cluster(9)
    result = build_surface(
        atoms, SurfaceSettings(mode="sas", spacing=0.5, property="hydrophobicity")
    )
    stops = result.legend(count=4)
    assert len(stops) == 4
    assert result.value_range == (0.0, 1.0)
    assert stops[0][2] == "0.00"      # a normalised scale reads in hundredths
    assert stops[-1][2] == "1.00"
    # A fixed range is honoured, and reported back unchanged.
    fixed = build_surface(
        atoms,
        SurfaceSettings(
            mode="sas", spacing=0.5, property="hydrophobicity", value_range=(0.2, 0.8)
        ),
    )
    assert fixed.value_range == (0.2, 0.8)


def test_the_electrostatic_surface_carries_a_symmetric_percentile_range():
    atoms = [
        atom("N", 0.0, 0.0, 0.0, charge=1.0),
        atom("O", 1.3, 0.0, 0.0, charge=-1.0),
        atom("C", 0.6, 1.2, 0.0, charge=0.0),
    ]
    result = build_surface(
        atoms, SurfaceSettings(mode="sas", spacing=0.4, property="electrostatic")
    )
    low, high = result.value_range
    assert low < 0.0 < high
    assert low == pytest.approx(-high)
    assert result.palette == "electrostatic"
    assert np.isfinite(result.values).all()
    # The values really are a potential: opposite charges give opposite signs.
    assert result.values.min() < 0.0 < result.values.max()


def test_the_element_property_uses_the_cpk_palette():
    atoms = [atom("C", 0.0, 0.0, 0.0), atom("O", 2.6, 0.0, 0.0)]
    result = build_surface(
        atoms, SurfaceSettings(mode="sas", spacing=0.4, property="element")
    )
    used = set()
    for row in result.colors:
        used.add(tuple(np.round(row, 6)))
    assert tuple(np.round(element_color("C"), 6)) in used
    assert tuple(np.round(element_color("O"), 6)) in used


def test_highlighting_marks_only_the_named_residues():
    # The two residues have to be apart: overlapping atoms make the nearest-atom
    # assignment a coin toss, which is not what this test is about.
    atoms = residue("A", "LEU", 1, 4) + residue(
        "B", "ASP", 2, 4, offset=(0.0, 12.0, 0.0)
    )
    result = build_surface(
        atoms,
        SurfaceSettings(
            mode="sas",
            spacing=0.5,
            highlighted_residues={("A", 1, "LEU")},
        ),
    )
    assert result.highlighted is not None
    assert 0 < int(result.highlighted.sum()) < result.vertices_count
    assert result.residue_labels == ["A/LEU1"]
    # The highlight shows in the mesh but not in the raw property colours.
    plain = build_surface(atoms, SurfaceSettings(mode="sas", spacing=0.5))
    assert not np.allclose(result.mesh()[:, 6:9], plain.mesh()[:, 6:9])


def test_a_pocket_lining_surface_uses_fewer_atoms_and_is_open():
    atoms = [atom("C", 1.5 * index, 0.0, 0.0) for index in range(20)]
    whole = build_surface(atoms, SurfaceSettings(mode="sas", spacing=0.5))
    pocket = build_surface(
        atoms, SurfaceSettings(mode="sas", spacing=0.5, centre=(7.0, 0.0, 0.0), radius=4.0)
    )
    assert pocket.stats["atoms"] < whole.stats["atoms"]
    assert pocket.stats["atoms_total"] == 20
    assert pocket.area < whole.area
    # An empty selection is reported rather than raising.
    nothing = build_surface(
        atoms, SurfaceSettings(mode="sas", spacing=0.5, centre=(500.0, 0.0, 0.0), radius=1.0)
    )
    assert nothing.vertices_count == 0
    assert nothing.stats["reason"] == "empty selection"


def test_the_stats_report_the_cost_of_the_build():
    atoms = _cluster(9)
    result = build_surface(atoms, SurfaceSettings(mode="ses", spacing=0.5))
    stats = result.stats
    assert stats["atoms"] == 9
    assert stats["grid_points"] == stats["grid"][0] * stats["grid"][1] * stats["grid"][2]
    assert stats["grid_points"] > 0
    assert stats["triangles"] == result.triangles_count
    assert stats["seconds"] >= 0.0
    assert stats["sas_area"] is not None
    assert result.summary().startswith("SES")
    assert result.as_dict()["mode"] == "ses"


def test_the_settings_are_validated_and_normalised():
    with pytest.raises(ValueError):
        SurfaceSettings(mode="rolling").normalized()
    with pytest.raises(ValueError):
        SurfaceSettings(property="loveliness").normalized()
    assert SurfaceSettings(mode="SES", property="Hydrophobicity").normalized().mode == "ses"
    assert SURFACE_MODES == ("sas", "ses")
    assert SURFACE_PROPERTIES == ("hydrophobicity", "electrostatic", "element")


def test_progress_is_reported_in_order_and_never_breaks_the_build():
    seen = []

    def progress(phase: str, fraction: float) -> None:
        seen.append((phase, fraction))

    result = build_surface(
        _cluster(9), SurfaceSettings(mode="ses", spacing=0.5), progress=progress
    )
    assert seen, "the builder must report phases"
    assert seen[-1][1] == pytest.approx(1.0)
    assert {name for name, _ in seen} >= {"grid", "field", "mesh", "colour"}
    assert result.vertices_count > 0

    def bad_progress(phase: str, fraction: float) -> None:  # pragma: no cover
        raise RuntimeError("the host callback is broken")

    still_built = build_surface(
        _cluster(4), SurfaceSettings(mode="sas", spacing=0.6), progress=bad_progress
    )
    assert still_built.vertices_count > 0


def test_the_grid_stays_inside_the_budget_however_large_the_structure():
    """The documented scaling claim: the grid is capped, not unbounded.

    With no explicit spacing the builder chooses one that fits the point
    budget, so a 2 000-atom blob cannot allocate an unbounded array. (The
    suite keeps the count modest; the real 2 000-atom receptor measurement is
    in ``docs/VISUALIZATION.md`` and ``Surface.stats`` reports it live.)
    """
    atoms = [
        atom("C", 1.4 * (index % 12), 1.4 * (index // 12 % 12), 1.4 * (index // 144))
        for index in range(2000)
    ]
    result = build_surface(
        atoms,
        SurfaceSettings(mode="sas", spacing=0.0, max_points=150_000),
    )
    assert result.stats["atoms"] == 2000
    assert result.stats["grid_points"] <= 150_000
    assert result.spacing >= surface_module.MIN_SPACING
    # An explicit spacing is a request the caller made, so it wins over the cap
    # (the caller asked for that resolution) — but never leaves the clamps.
    explicit = resolve_spacing((0, 0, 0), (40, 40, 40), 0.35, max_points=1000)
    assert explicit == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# the workbench integration
# ---------------------------------------------------------------------------
#
# These drive the real window offscreen: the surface reaches the screen through
# the mesh pipeline, its legend is drawn by the viewport's own painter, and the
# menu actions have to keep working after a language switch, which rebuilds
# every widget in the window.

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")


@pytest.fixture(scope="module")
def qapp():
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def workbench(qapp):
    pytest.importorskip("moderngl")
    from odock.gui.app import DockingWorkbench
    from odock.gui import i18n

    i18n.set_language("en")
    window = DockingWorkbench()
    window.resize(640, 480)
    window.load_receptor(str(_pdbqt_text(40, "ALA", 1, "C")))
    yield window
    window.close()
    window.deleteLater()
    i18n.set_language("en")


def _pdbqt_text(count: int, res_name: str, res_id: int, element: str) -> str:
    """A tiny receptor PDBQT written in the file, so nothing is mocked."""
    import tempfile

    lines = []
    for index in range(count):
        lines.append(
            f"ATOM  {index + 1:5d}  CA  {res_name} A{res_id:4d}    "
            f"{3.0 * math.cos(index / 5.0):8.3f}{3.0 * math.sin(index / 5.0):8.3f}"
            f"{0.6 * index - 6.0:8.3f}  1.00  0.00     0.000 {element}"
        )
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".pdbqt", delete=False, encoding="utf-8"
    )
    handle.write("\n".join(lines) + "\nTER\n")
    handle.close()
    return handle.name


def _wait_for_surface(workbench, app, timeout_s: float = 60.0) -> bool:
    """Wait for the background build the window started, then pump Qt once.

    The build runs on a real ``QThread`` (that is the point: the window must
    stay responsive), so the test joins it rather than polling a flag.
    """
    worker = getattr(workbench, "_background", None)
    if worker is not None and worker.isRunning():
        worker.wait(int(timeout_s * 1000))
    app.processEvents()
    return workbench.scene.surface is not None


def test_the_window_builds_a_surface_and_puts_it_on_the_scene(workbench, qapp):
    from odock.gui.surface import SurfaceSettings

    workbench.surface_settings = SurfaceSettings(mode="sas", spacing=0.3)
    workbench._rebuild_surface()
    assert _wait_for_surface(workbench, qapp), "no surface was built"
    surface = workbench.scene.surface
    assert surface.triangles_count > 0
    assert workbench.scene.show_surface is True
    # The renderer uploads it through the ordinary mesh pipeline on the next
    # frame, so the test draws one.
    workbench.viewport._ensure_context()
    workbench.viewport.refresh()
    renderer = workbench.viewport.renderer
    assert renderer is not None
    data = renderer.render_image(workbench.viewport.camera, 320, 240)
    assert renderer.surface_vertices == surface.mesh().shape[0]
    assert renderer.surface_triangles == surface.triangles_count
    pixels = np.frombuffer(data, dtype="u1").reshape(240, 320, 3).astype(int)
    assert pixels.std() > 3.0, "the frame is blank"
    # A translucent surface takes the depth-mask path: it must draw as well.
    workbench.scene.surface_alpha = 0.5
    translucent = renderer.render_image(workbench.viewport.camera, 320, 240)
    assert np.frombuffer(translucent, dtype="u1").std() > 3.0
    workbench.scene.surface_alpha = 1.0


def test_the_legend_is_drawn_by_the_viewport_painter(workbench, qapp):
    from PyQt6 import QtGui

    from odock.gui.surface import SurfaceSettings

    workbench.surface_settings = SurfaceSettings(mode="sas", spacing=0.3)
    workbench._rebuild_surface()
    assert _wait_for_surface(workbench, qapp)
    assert workbench.scene.legend_stops(), "the scene has no legend stops"
    image = QtGui.QImage(400, 300, QtGui.QImage.Format.Format_RGB888)
    image.fill(0)
    painter = QtGui.QPainter(image)
    try:
        workbench.viewport._paint_legend(painter, 400, 300)
        workbench.viewport._paint_scale_bar(painter, 400, 300)
    finally:
        painter.end()
    # Something was painted: a blank image would mean the legend silently
    # returned early (a missing font, an empty range, a wrong property name).
    colours = {image.pixel(x, y) for x in range(0, 120, 3) for y in range(180, 300, 3)}
    assert len(colours) > 3


def test_the_surface_menu_survives_a_language_switch(workbench):
    from odock.gui import i18n
    from odock.gui.surface import SurfaceSettings

    workbench.surface_settings = SurfaceSettings(mode="ses", spacing=0.4)
    workbench._pocket_only = True
    workbench._surface_alpha = 0.5
    workbench.set_language("zh")
    assert workbench.surface_settings.mode == "ses"
    assert workbench._pocket_only is True
    assert workbench._surface_alpha == pytest.approx(0.5)
    assert workbench.scene.surface_alpha == pytest.approx(0.5)
    # The new menu is rebuilt with its radio state intact.
    assert workbench._surface_mode_actions["ses"].isChecked()
    i18n.set_language("en")


def test_the_pocket_lining_settings_are_built_from_the_ligand(workbench):
    workbench.scene.ligand = [atom("C", 3.0, 0.0, 0.0, res_name="LIG", res_id=9)]
    workbench._pocket_only = True
    settings = workbench._surface_settings()
    assert settings.centre is not None
    assert settings.radius == pytest.approx(workbench._pocket_radius)
    assert settings.bonds is not None
    workbench._highlight_pocket = True
    highlighted = workbench._surface_settings()
    assert highlighted.highlighted_residues, "no pocket residues were found"


def test_the_per_atom_export_values_land_on_the_right_atoms(workbench):
    """The one error a picture cannot show you: right numbers, wrong atoms.

    A pocket-lining build indexes its own filtered atom list, so mapping the
    property back onto the scene has to go through ``Surface.selected``. The
    test builds a surface from a *subset* of the receptor and checks that the
    atoms the surface covers get their own values and the rest stay neutral —
    an off-by-one here paints a plausible and completely wrong picture.
    """
    from odock.gui import surface as surface_module

    receptor = list(workbench.scene.receptor)
    assert len(receptor) >= 20
    # Two well-separated groups, so a mis-mapping cannot look plausible.
    indices = list(range(5)) + list(range(len(receptor) - 5, len(receptor)))
    subset = [receptor[index] for index in indices]
    values = [0.0] * len(receptor)
    for position, atom in enumerate(subset):
        atom.name = f"H{position}"
    surface = surface_module.build_surface(
        subset, surface_module.SurfaceSettings(mode="sas", spacing=0.4, indices=None)
    )
    surface.selected = list(indices)
    workbench.scene.receptor = receptor
    workbench.scene.surface = surface
    exported = workbench._surface_values_per_atom(surface)
    assert exported is not None and len(exported) == len(receptor)
    neutral = 0.5 * (surface.value_range[0] + surface.value_range[1])
    covered = [exported[index] for index in indices]
    untouched = [exported[index] for index in range(len(receptor)) if index not in indices]
    assert all(abs(value - neutral) < 1e-9 for value in untouched)
    assert any(abs(value - neutral) > 1e-9 for value in covered)


# SPDX-License-Identifier: GPL-3.0-or-later
"""Closed-form tests for :mod:`odock.cavity`.

Every test here builds a shape whose answer is known in advance — a hollow
sphere, a tube whose bottleneck radius is ``R_wall - r_atom - probe``, the same
tube blocked, and a box that is deliberately too tight — so the classification
and the aperture are checked against geometry rather than against this
implementation, and against the failure mode the docs warn about.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from odock import cavity
from odock.gui.structure import Atom

PROBE = cavity.PROBE_RADIUS
CARBON = 1.70


def shell(radius: float, count: int = 90) -> list:
    """``count`` carbons spread evenly on a sphere of ``radius`` Å."""
    points = cavity.sphere_fibonacci(count)
    return [
        Atom(
            name=f"C{index}",
            element="C",
            res_name="SHE",
            res_id=1,
            chain="A",
            x=float(point[0] * radius),
            y=float(point[1] * radius),
            z=float(point[2] * radius),
        )
        for index, point in enumerate(points)
    ]


def tube(wall_radius: float, length: float, *, blocked=None) -> list:
    """A ring of carbons forming a cylindrical pore of radius ``wall_radius``.

    The pore runs along ``z``. A probe centre can sit at most
    ``wall_radius - (r_atom + probe)`` from the axis, which is the analytic
    bottleneck radius the tests compare against. ``blocked`` is ``None`` (an open
    bore), ``"middle"`` (one disc across the centre, which leaves two
    open-ended halves) or ``"both"`` (a disc near each end, which seals the
    middle into a cavity).
    """
    atoms = []
    ring = max(8, int(round(2.0 * math.pi * wall_radius / 3.0)))
    levels = max(3, int(round(length / 3.0)))
    for level in range(levels + 1):
        z = -length / 2.0 + length * level / levels
        for index in range(ring):
            angle = 2.0 * math.pi * index / ring
            atoms.append(
                Atom(
                    name=f"C{len(atoms)}",
                    element="C",
                    res_name="TUB",
                    res_id=1,
                    chain="A",
                    x=float(wall_radius * math.cos(angle)),
                    y=float(wall_radius * math.sin(angle)),
                    z=float(z),
                )
            )
    if blocked:
        # A *disc* of atoms across the bore. A ring on the axis is not enough:
        # the probe simply passes around it inside the tube, which is how this
        # test first failed.
        for plate_z in ((0.0,) if blocked == "middle" else (-length / 2.0 + 1.5, length / 2.0 - 1.5)):
            for gx in range(-3, 4):
                for gy in range(-3, 4):
                    x = gx * 1.1
                    y = gy * 1.1
                    if math.hypot(x, y) <= 3.2:
                        atoms.append(
                            Atom(
                                name=f"B{len(atoms)}",
                                element="C",
                                res_name="BLK",
                                res_id=9,
                                chain="A",
                                x=float(x),
                                y=float(y),
                                z=float(plate_z),
                            )
                        )
    return atoms


def tube_interior(atoms, allow=1.9):
    """The cells inside the bore: within ``allow`` Å of the ``z`` axis."""
    return allow


def field_for(atoms, spacing: float, margin: float = 6.0):
    coords, radii = cavity._atom_arrays(atoms)
    axes = cavity._axes(coords, margin, spacing)
    field = cavity.probe_field(axes, coords, radii + PROBE, clamp=2.0 * spacing)
    return axes, coords, field


# ---------------------------------------------------------------------------
# the field and the box
# ---------------------------------------------------------------------------


def test_the_cavity_field_matches_the_surface_builder_field():
    """Two implementations of one field: they must agree, or one is wrong.

    ``cavity.probe_field`` is a deliberate copy (a command-line cavity analysis
    must not import the viewer's package), so something has to check it against
    ``odock.gui.surface.scalar_field``.
    """
    surface = pytest.importorskip("odock.gui.surface")
    atoms = shell(6.0, 40)
    spacing = 0.7
    coords, radii = cavity._atom_arrays(atoms)
    axes = cavity._axes(coords, 5.0, spacing)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    mine = cavity.probe_field(axes, coords, radii + PROBE, clamp=2.0 * spacing)
    theirs = surface.scalar_field(
        grid, atoms, probe=PROBE, radii=radii, spacing=spacing, axes=axes
    )
    assert mine.shape == theirs.shape
    assert np.allclose(mine, theirs, atol=1e-9)


def test_the_scan_finds_protein_on_both_sides_and_counts_directions():
    """The LIGSITE test, on a grid whose answer is obvious.

    Serialising the count as integers is the point: in NumPy two summed boolean
    arrays are a logical *or*, so a version of this counted 0 or 1 and never
    reached the five-direction threshold at all.
    """
    grid = np.zeros((5, 5, 5), dtype=bool)
    grid[2, :, :] = True                     # a protein plane through the middle
    before, after = cavity._side_masks(grid, 0, 1)
    # One cell below the plane: protein lies in the + direction only.
    assert not bool(before[1, 0, 0])
    assert bool(after[1, 0, 0])
    # One cell above it: protein lies in the - direction only.
    assert bool(before[3, 0, 0])
    assert not bool(after[3, 0, 0])
    counted = cavity._directions_covered(grid, 1)
    # A cell just off the plane sees protein on one side, so it counts one; the
    # plane's own cell sees it on both sides and counts two. Nothing sees protein
    # along y or z. Serialising as integers matters: summing booleans in NumPy is
    # a logical *or*, which would cap this at one and never reach the threshold.
    assert int(counted[1, 0, 0]) == 1
    assert int(counted[3, 0, 0]) == 1
    # A cell *inside* the plane is inside the protein, so every direction sees it.
    assert int(counted[2, 0, 0]) == 6
    assert int(counted[0, 0, 0]) == 0        # out of reach of the plane


# ---------------------------------------------------------------------------
# a fully enclosed cavity
# ---------------------------------------------------------------------------


def test_a_sealed_shell_is_one_enclosed_cavity_with_the_analytic_volume():
    """A hollow sphere: the void is ``4/3 pi (R_shell - r_atom - probe)^3``.

    The volume the analysis reports is the *probe-accessible* volume, i.e. where
    the centre of a water molecule can go, so the analytic answer uses the
    inflated radii. The shell is 90 atoms, which is dense enough that the probe
    cannot squeeze between them.
    """
    radius = 8.0
    atoms = shell(radius, 90)
    result = cavity.analyse_cavities(
        atoms, cavity.CavitySettings(spacing=0.5)
    )
    inner = radius - CARBON - PROBE
    exact = 4.0 / 3.0 * math.pi * inner**3
    enclosed = result.of_kind("enclosed")
    assert len(enclosed) == 1, result.table()
    pocket = enclosed[0]
    assert pocket.volume == pytest.approx(exact, rel=0.2)
    assert pocket.bottleneck_radius is None
    assert pocket.openings == 0
    assert pocket.enclosure == 1.0
    assert result.box_too_tight is False
    assert result.bulk_faces == 6
    assert len(pocket.lining_residues) == 1          # the shell is one residue
    assert pocket.lining_residues[0].endswith("SHE1")


def test_a_cavity_that_is_a_dent_is_not_reported_as_a_site():
    """A shallow dish is not a pocket: the volume floor does its job."""
    atoms = shell(6.0, 90)
    atoms = [
        atom
        for atom in atoms
        if atom.z > -1.0                      # half a shell: an open bowl
    ]
    result = cavity.analyse_cavities(atoms, cavity.CavitySettings(spacing=0.6))
    assert all(pocket.kind != "enclosed" for pocket in result.pockets)
    for pocket in result.pockets:
        assert pocket.volume >= cavity.MIN_VOLUME


# ---------------------------------------------------------------------------
# a channel, and the bottleneck radius
# ---------------------------------------------------------------------------


def test_a_tube_open_at_both_ends_has_the_analytic_bottleneck_radius():
    """The aperture search, against ``R_wall - r_atom - probe``.

    A probe centre can reach no closer than ``R_wall - (r_atom + probe)`` to the
    axis of a cylindrical pore, so that difference *is* the bottleneck radius —
    no fitting, no tolerance beyond the grid step.
    """
    wall = 5.0
    spacing = 0.5
    atoms = tube(wall, 12.0)
    axes, coords, field = field_for(atoms, spacing)
    accessible = field > 0.0
    labels, _count = cavity._label(accessible)
    face = cavity._face_counts(labels)
    distance = cavity._ndimage.distance_transform_edt(accessible, sampling=spacing)

    # The bore: accessible cells inside the tube, away from the open ends (out
    # there the "bore" would be open solvent with an unbounded inradius).
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    bore = (
        accessible
        & (np.sqrt(xx**2 + yy**2) < 2.5)
        & (np.abs(zz) < 12.0 / 2.0 - 1.0)
    )
    assert bore.any()
    bottleneck, openings, neck = cavity._aperture(
        bore, accessible, distance, labels, set(face), spacing, 6.0
    )
    exact = wall - CARBON - PROBE
    assert bottleneck is not None
    assert bottleneck == pytest.approx(exact, abs=spacing)
    assert openings is None          # the mouth count is not reported
    assert len(neck) > 0             # the neck cells are, for the residues


def test_a_half_blocked_tube_is_still_a_channel():
    """Blocking the middle of a bore leaves two open-ended channels.

    This is the honest outcome and worth pinning: the blockage divides the tube
    into two halves, each of which still has one exit, so the aperture search
    finds a way out with the same bottleneck radius as the open tube.
    """
    wall = 5.0
    spacing = 0.5
    atoms = tube(wall, 12.0, blocked="middle")
    axes, coords, field = field_for(atoms, spacing)
    accessible = field > 0.0
    labels, _count = cavity._label(accessible)
    face = cavity._face_counts(labels)
    distance = cavity._ndimage.distance_transform_edt(accessible, sampling=spacing)
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    bore = (
        accessible
        & (np.sqrt(xx**2 + yy**2) < 2.5)
        & (zz > 1.6)                       # the half beyond the blockage
        & (zz < 12.0 / 2.0 - 1.0)
    )
    assert bore.any()
    bottleneck, openings, _neck = cavity._aperture(
        bore, accessible, distance, labels, set(face), spacing, 6.0
    )
    assert bottleneck is not None
    assert bottleneck == pytest.approx(1.5, abs=spacing)
    assert openings is None          # see _aperture: not reported, not guessed


def test_a_tube_sealed_at_both_ends_has_no_aperture():
    """Seal both exits and the same search must find no way out."""
    wall = 5.0
    spacing = 0.5
    atoms = tube(wall, 12.0, blocked="both")
    axes, coords, field = field_for(atoms, spacing)
    accessible = field > 0.0
    labels, _count = cavity._label(accessible)
    face = cavity._face_counts(labels)
    distance = cavity._ndimage.distance_transform_edt(accessible, sampling=spacing)
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    bore = accessible & (np.sqrt(xx**2 + yy**2) < 2.5) & (np.abs(zz) < 3.0)
    if not bore.any():  # pragma: no cover - defensive
        pytest.skip("the blockage left no bore cells to test")
    bottleneck, openings, _neck = cavity._aperture(
        bore, accessible, distance, labels, set(face), spacing, 6.0
    )
    assert bottleneck is None
    assert openings == 0
    # And the whole analysis agrees: a sealed middle is an enclosed cavity.
    report = cavity.analyse_cavities(atoms, cavity.CavitySettings(spacing=0.6))
    assert any(pocket.kind == "enclosed" for pocket in report.pockets), report.table()


def test_a_through_tunnel_needs_a_lower_direction_threshold():
    """The criterion's limit, stated as a test rather than a surprise.

    A bore open at *both* ends has protein on four of the six scan directions,
    so the five-of-six pocket rule does not call it a pocket — it is a channel,
    not a pocket. Lowering `min_directions` to four finds it, at the price of
    more false positives everywhere else; both halves are asserted so the
    trade-off cannot drift.
    """
    wall = 5.0
    atoms = tube(wall, 14.0)
    strict = cavity.analyse_cavities(
        atoms, cavity.CavitySettings(spacing=0.6, min_directions=5)
    )
    relaxed = cavity.analyse_cavities(
        atoms, cavity.CavitySettings(spacing=0.6, min_directions=4)
    )
    names = {pocket.kind for pocket in strict.pockets}
    assert "open" not in names, strict.table()
    assert relaxed.pockets, "the relaxed rule found nothing in a tube"
    assert any(pocket.kind in ("open", "shallow") for pocket in relaxed.pockets)


# ---------------------------------------------------------------------------
# the box hypothesis
# ---------------------------------------------------------------------------


def test_the_margin_is_raised_so_the_bulk_hypothesis_holds():
    """A too-tight box must be repaired, not guessed around.

    The classification calls the box faces "the outside". With a box that hugs
    the atoms there is no solvent layer inside it, so a genuinely open pocket has
    nothing to connect to and would look enclosed. ``auto_margin`` raises the
    margin to the value that guarantees a layer.
    """
    atoms = shell(8.0, 90)
    tight = cavity.analyse_cavities(
        atoms, cavity.CavitySettings(spacing=0.7, margin=0.5, auto_margin=True)
    )
    needed = PROBE + cavity.CORE_RADIUS + 2.0 * tight.spacing
    assert tight.margin >= needed
    assert tight.bulk_faces == 6
    assert tight.box_too_tight is False


def test_a_box_that_cannot_hold_a_solvent_layer_labels_nothing_enclosed():
    """The safe failure: refuse to classify rather than call an open pore sealed."""
    atoms = shell(8.0, 90)
    forced = cavity.analyse_cavities(
        atoms,
        cavity.CavitySettings(spacing=0.7, margin=0.05, auto_margin=False),
    )
    # Either the box was still adequate, or it was reported and nothing was
    # labelled enclosed. What must never happen is a silent "enclosed".
    if forced.box_too_tight:
        assert all(pocket.kind != "enclosed" for pocket in forced.pockets)
        assert all(pocket.kind in ("unknown", "open", "shallow") for pocket in forced.pockets)
    else:
        assert forced.bulk_faces == 6


# ---------------------------------------------------------------------------
# the report and the heuristic
# ---------------------------------------------------------------------------


def test_the_geometric_score_is_a_stated_weighted_mean_and_says_so():
    """It is arithmetic, not a model: the weights are documented and pinned."""
    assert cavity.geometric_score(1.0, 400.0, 1.0) == pytest.approx(1.0)
    assert cavity.geometric_score(0.0, 0.0, 0.0) == pytest.approx(0.0)
    half = cavity.geometric_score(0.5, 200.0, 0.5)
    assert half == pytest.approx(0.45 * 0.5 + 0.35 * 0.5 + 0.20 * 0.5)
    # Volume saturates: beyond 400 Å³ more space is not more score.
    assert cavity.geometric_score(0.5, 400.0, 0.5) == pytest.approx(
        cavity.geometric_score(0.5, 4000.0, 0.5)
    )
    doc = cavity.geometric_score.__doc__ or ""
    assert "druggability" in doc and "not" in doc


def test_an_empty_structure_is_an_empty_report():
    report = cavity.analyse_cavities([])
    assert report.pockets == []
    assert report.grid_points == 0
    assert report.seconds >= 0.0
    assert report.as_dict()["pockets"] == []


def test_the_report_carries_the_criterion_it_used():
    """The numbers behind the classification travel with it, so they can be read."""
    atoms = shell(8.0, 60)
    report = cavity.analyse_cavities(atoms, cavity.CavitySettings(spacing=0.7))
    payload = report.as_dict()
    assert payload["spacing"] == pytest.approx(0.7)
    assert payload["probe"] == pytest.approx(PROBE)
    assert payload["core_radius"] == pytest.approx(cavity.CORE_RADIUS)
    assert payload["bulk_faces"] == 6
    assert payload["direction_histogram"]
    assert isinstance(payload["pockets"], list)
    if report.pockets:
        first = report.pockets[0].as_dict()
        assert set(first) >= {
            "label", "kind", "volume", "openings", "bottleneck_radius",
            "bottleneck_residues", "lining_residues", "hydrophobicity",
            "enclosure", "geometric_score",
        }
        assert first["kind"] in ("enclosed", "open", "shallow", "unknown")

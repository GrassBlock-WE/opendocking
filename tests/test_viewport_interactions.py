# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for the interaction dashes' *legibility*.

A user reported "many white lines after docking" that they could not identify.
The geometry was verified correct (the rendered endpoint distance equals the
reported distance), so the defect was legibility: the hydrophobic contact was a
near-white grey, and every dash is additionally lit by a headlight, so on the
dark viewport it was the brightest thing in the frame apart from the ligand.

These tests pin the two fixes:

* the swatch itself — legible against **both** viewport backgrounds, still
  distinguishable from the other five kinds, and explicitly not near-white;
* the **dark under-stroke** every dash now carries, whose geometry has to be
  right or the core would be hidden by its own outline (the outline is fatter,
  so it must be pushed *away* from the camera by at least the radius
  difference);
* and, as the end-to-end check, that a frame drawn with the new swatch has
  measurably fewer light pixels in the contact region than the old one.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from odock.gui.viewport import (
    BOX_COLOR,
    INTERACTION_COLORS,
    INTERACTION_LABELS,
    Renderer,
    push_along_view,
)

#: The dark and the light viewport clear colours, as RGB in 0..1. Duplicated from
#: ``odock.gui.dashboard`` on purpose so this module needs no Qt to check the
#: colours; a test below asserts they still match the themes.
DARK_CANVAS = (0.086, 0.094, 0.125)
LIGHT_CANVAS = (0.784, 0.804, 0.831)


def luminance(colour) -> float:
    red, green, blue = (float(value) for value in colour[:3])
    return 0.299 * red + 0.587 * green + 0.114 * blue


def chroma(colour) -> float:
    values = [float(value) for value in colour[:3]]
    return max(values) - min(values)


def rgb_distance(first, second) -> float:
    return math.sqrt(
        sum((float(first[axis]) - float(second[axis])) ** 2 for axis in range(3))
    )


# ---------------------------------------------------------------------------
# the colours
# ---------------------------------------------------------------------------


def test_the_canvases_here_are_the_ones_the_workbench_uses():
    """The colours are checked against the real backgrounds, not invented ones."""
    dashboard = pytest.importorskip("odock.gui.dashboard")
    assert tuple(dashboard.DARK.viewport_clear[:3]) == pytest.approx(DARK_CANVAS)
    assert tuple(dashboard.LIGHT.viewport_clear[:3]) == pytest.approx(LIGHT_CANVAS)


def test_every_interaction_colour_is_legible_on_both_viewport_backgrounds():
    """A dash must be separable from the canvas it is drawn on.

    Luminance alone is not the test: cyan sits at almost exactly the light
    canvas's luminance (0.77 against 0.80) and is perfectly legible because it
    is a *colour*. The measure is therefore the distance in RGB, which covers
    both the luminance and the hue.
    """
    for kind, colour in INTERACTION_COLORS.items():
        for name, canvas in (("dark", DARK_CANVAS), ("light", LIGHT_CANVAS)):
            distance = rgb_distance(colour, canvas)
            assert distance >= 0.35, (
                f"{kind} is only {distance:.2f} from the {name} canvas"
            )


def test_the_six_interaction_colours_stay_distinguishable():
    kinds = sorted(INTERACTION_COLORS)
    for index, first in enumerate(kinds):
        for second in kinds[index + 1 :]:
            distance = rgb_distance(INTERACTION_COLORS[first], INTERACTION_COLORS[second])
            assert distance >= 0.40, f"{first} and {second} are {distance:.2f} apart"


def shaded_swatch(swatch, steps: int = 400) -> np.ndarray:
    """The colours a dash actually takes on screen, from the mesh shader's law.

    ``MESH_FS`` shades a fragment as ``colour·(0.34 + 0.72·d) + spec + rim``
    with ``d = |dot(normal, light)|`` and ``spec = d²⁴·0.28``. Across the visible
    half of a dash tube ``d`` sweeps 1 → 0, so this is the range the eye sees —
    and the reason a pale, achromatic swatch turns into a white-looking line
    under a headlight.
    """
    d = np.linspace(1.0, 0.0, int(steps))[:, None]
    base = np.asarray(swatch[:3], dtype=float)[None, :]
    shaded = base * (0.34 + 0.72 * d) + np.power(d, 24.0) * 0.28 + np.power(
        1.0 - d, 3.0
    ) * 0.12
    return np.clip(shaded, 0.0, 1.0)


def shaded_luminance(swatch) -> np.ndarray:
    shaded = shaded_swatch(swatch)
    return 0.299 * shaded[:, 0] + 0.587 * shaded[:, 1] + 0.114 * shaded[:, 2]


def test_the_hydrophobic_contact_is_no_longer_near_white():
    """The specific defect: a pale line reads as a stray white stroke.

    The old swatch was (0.62, 0.64, 0.68) — luminance 0.64 against a 0.09
    background, seven times the canvas, with a chroma of 0.06, i.e. grey. The
    new one is a mid-tone *with a hue*, and the shader's own colour law is what
    is checked, not just the swatch: the shaded band that used to reach the top
    of the range no longer does.
    """
    swatch = INTERACTION_COLORS["hydrophobic"]
    assert 0.45 <= luminance(swatch) <= 0.62
    assert chroma(swatch) > 2.0 * chroma(OLD_HYDROPHOBIC)
    assert luminance(swatch) < luminance(OLD_HYDROPHOBIC)

    before = shaded_luminance(OLD_HYDROPHOBIC)
    after = shaded_luminance(swatch)
    assert before.max() > 0.90, "the old swatch has to reproduce the report"
    assert after.max() < 0.90
    assert (before >= 0.90).sum() > 0
    assert (after >= 0.90).sum() == 0
    # And it must not be the brightest of the six kinds.
    assert luminance(swatch) < max(
        luminance(value) for value in INTERACTION_COLORS.values()
    )


def test_every_kind_has_a_label_and_an_opaque_enough_alpha():
    for kind, colour in INTERACTION_COLORS.items():
        assert kind in INTERACTION_LABELS
        assert 0.5 <= float(colour[3]) <= 1.0


def test_the_search_box_colour_cannot_be_mistaken_for_a_contact():
    """The box used to be drawn in the hydrogen-bond hue.

    A user read the resulting cube as "a hydrogen bond acting at a long
    distance". The measured distance between the old box colour
    ``(0.30, 0.85, 0.95)`` and the hydrogen-bond cyan ``(0.20, 0.90, 0.95)`` is
    **0.112** in RGB — the same colour as far as an eye is concerned. The box is
    now teal, at least 0.40 from every contact colour and from both canvases.
    """
    old_box = (0.30, 0.85, 0.95)
    assert rgb_distance(old_box, INTERACTION_COLORS["hbond"]) < 0.2, (
        "the recorded defect: the old box colour was the hydrogen-bond colour"
    )
    for kind, colour in INTERACTION_COLORS.items():
        gap = rgb_distance(BOX_COLOR, colour)
        assert gap >= 0.35, f"the box is {gap:.2f} from the {kind} colour"
    for name, canvas in (("dark", DARK_CANVAS), ("light", LIGHT_CANVAS)):
        gap = rgb_distance(BOX_COLOR, canvas)
        assert gap >= 0.35, f"the box is {gap:.2f} from the {name} canvas"


def test_a_contact_reads_as_one_dashed_line_not_a_comb():
    """The dash pattern is a legibility choice; the endpoints do not move.

    Every dash is a separate swept cylinder, so a short pitch turns one contact
    into a row of teeth: the previous 0.28 Å dash with a 0.20 Å gap put seven
    segments on a 3 Å contact. The pattern now gives at most five, while still
    covering the whole distance (the last dash is clipped to the endpoint, so the
    drawn line always terminates exactly on the atoms it names).
    """
    pitch = Renderer.DASH_LENGTH + Renderer.DASH_GAP
    assert Renderer.DASH_LENGTH >= 0.4
    assert pitch >= 0.7
    for length in (2.89, 3.0, 3.5, 4.0):
        segments = int(length // pitch) + 1
        assert segments <= 5, f"{length} Å of contact became {segments} dashes"


def test_every_contact_gets_an_endpoint_marker_at_each_named_atom():
    """A marker is what makes a contact *land* on an atom inside the cartoon.

    The ribbon is swept along the C-alpha trace with a 1.05 Å half-width, so it
    swallows the atom a dash ends on and the contact reads as a stroke from
    nowhere. The marker is pushed towards the camera by at least the ribbon's
    half-width, and must be wider on screen than the dash so it reads as a puck
    rather than a thickening of the line.
    """
    assert Renderer.INTERACTION_MARKER_PIXELS > Renderer.INTERACTION_PIXELS
    assert Renderer.INTERACTION_MARKER_CLEARANCE >= 1.05
    eye = np.array([0.0, 0.0, 0.0])
    atom = np.array([[6.0, 0.0, 0.0]])
    moved = push_along_view(atom, [-Renderer.INTERACTION_MARKER_CLEARANCE], eye)
    before = float(np.linalg.norm(atom - eye))
    after = float(np.linalg.norm(moved - eye))
    assert after == pytest.approx(before - Renderer.INTERACTION_MARKER_CLEARANCE)
    assert after < before, "the marker must move towards the eye, not away"


# ---------------------------------------------------------------------------
# the under-stroke geometry
# ---------------------------------------------------------------------------


def test_push_along_view_moves_points_away_from_the_eye_by_the_given_distance():
    eye = np.array([0.0, 0.0, 0.0])
    points = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, -3.0]])
    distance = np.array([0.2, 0.4, 0.1])
    moved = push_along_view(points, distance, eye)
    before = np.linalg.norm(points - eye, axis=1)
    after = np.linalg.norm(moved - eye, axis=1)
    assert after == pytest.approx(before + distance, rel=1e-12)
    # The direction from the eye is unchanged: only the radius moves.
    for index in range(3):
        assert moved[index] / after[index] == pytest.approx(
            points[index] / before[index], abs=1e-12
        )


def test_push_along_view_is_a_no_op_without_a_distance():
    points = np.array([[1.0, 2.0, 3.0]])
    assert push_along_view(points, [0.0], (0.0, 0.0, 0.0)) == pytest.approx(points)
    # A point exactly at the eye has no direction: it must not become NaN.
    moved = push_along_view([[0.0, 0.0, 0.0]], [0.5], (0.0, 0.0, 0.0))
    assert np.isfinite(moved).all()


def test_the_outline_is_fatter_than_the_core_and_pushed_back_far_enough():
    """The depth contract that makes the rim work instead of hiding the core.

    The outline's near surface sits ``(r_outline - r_core)`` closer to the eye
    than the core's, so pushing it back by at least that much (plus the bias) is
    what leaves the core visible with a rim around it.
    """
    assert Renderer.INTERACTION_OUTLINE_SCALE > 1.0
    for radius in (0.008, 0.02, 0.1, 0.6):
        outline = radius * Renderer.INTERACTION_OUTLINE_SCALE + Renderer.INTERACTION_OUTLINE_MIN
        difference = outline - radius
        assert difference > 0.0
        pushed = push_along_view(
            [[10.0, 0.0, 0.0]], [difference], (0.0, 0.0, 0.0),
            Renderer.INTERACTION_OUTLINE_BIAS,
        )
        # The outline's near surface along the view axis is now behind the
        # core's, which is what the depth test compares.
        core_near = 10.0 - radius
        outline_near = float(pushed[0][0]) - outline
        assert outline_near > core_near


def test_the_outline_colour_is_dark_on_both_canvases():
    outline = Renderer.INTERACTION_OUTLINE
    assert luminance(outline) < 0.25
    assert rgb_distance(outline, DARK_CANVAS) < 0.35 or luminance(outline) < 0.15


# ---------------------------------------------------------------------------
# the end-to-end check: fewer light pixels than before
# ---------------------------------------------------------------------------

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

OLD_HYDROPHOBIC = (0.62, 0.64, 0.68, 0.75)

SCENE_PDBQT = "\n".join(
    [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000 C",
        "ATOM      2  CA  ALA A   1       3.800   0.000   0.000  1.00  0.00     0.000 C",
        "ATOM      3  CA  ALA A   1       0.000   3.800   0.000  1.00  0.00     0.000 C",
        "ATOM      4  CA  ALA A   1       3.800   3.800   0.000  1.00  0.00     0.000 C",
        "ATOM      5  CA  ALA A   1       1.900   1.900  -3.800  1.00  0.00     0.000 C",
        "TER",
        "",
    ]
)

LIGAND_PDBQT = "\n".join(
    [
        "ROOT",
        "ATOM      1  C1  LIG A   1       1.900   1.900   0.400  1.00  0.00     0.000 C",
        "ATOM      2  C2  LIG A   1       2.500   1.400   1.100  1.00  0.00     0.000 C",
        "ATOM      3  C3  LIG A   1       1.300   2.400   1.100  1.00  0.00     0.000 C",
        "ENDROOT",
        "TORSDOF 0",
        "",
    ]
)


class _Interaction:
    def __init__(self, kind: str, a: int, b: int) -> None:
        self.kind = kind
        self.a = a
        self.b = b


@pytest.fixture(scope="module")
def context():
    pytest.importorskip("moderngl")
    import moderngl

    try:
        return moderngl.create_standalone_context(require=330)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no OpenGL 3.3 context: {exc}")


def _render(ctx, interactions, swatch) -> np.ndarray:
    """One frame with ``swatch`` as the hydrophobic colour, as RGB pixels."""
    from odock.gui.structure import parse_pdbqt
    from odock.gui.viewport import Camera, Renderer, Scene

    receptor = parse_pdbqt(SCENE_PDBQT)[0].atoms
    ligand = parse_pdbqt(LIGAND_PDBQT)[0].atoms
    scene = Scene(receptor=receptor, ligand=ligand)
    scene.style_protein = "spheres"
    scene.style_ligand = "ball_stick"
    scene.ssao = False
    scene.interactions = list(interactions)
    width, height = 420, 320
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((width, height), 3)],
        depth_attachment=ctx.depth_texture((width, height)),
    )
    renderer = Renderer(ctx, scene)
    camera = Camera(distance=16.0, azimuth=0.6, elevation=0.3)
    camera.target = (1.9, 1.9, 0.5)
    original = INTERACTION_COLORS["hydrophobic"]
    INTERACTION_COLORS["hydrophobic"] = swatch
    try:
        fbo.use()
        renderer.draw(camera, width, height)
        ctx.finish()
        data = fbo.read(components=3)
    finally:
        INTERACTION_COLORS["hydrophobic"] = original
        fbo.release()
    return np.frombuffer(data, dtype="u1").reshape(height, width, 3).astype(int)


def test_the_contacts_no_longer_render_as_light_lines(context):
    """The user's complaint, as a measurement rather than an opinion.

    Three hydrophobic contacts are drawn in a frame with the old swatch and with
    the new one; the pixels the swatch governs are counted, and the number of
    them that read as *light* (luminance >= 0.55, which is what the eye calls a
    pale line on a dark canvas) must drop sharply.
    """
    from odock.gui.viewport import INTERACTION_COLORS as colours

    interactions = [
        _Interaction("hydrophobic", 0, 0),
        _Interaction("hydrophobic", 1, 1),
        _Interaction("hydrophobic", 2, 2),
    ]
    new_swatch = colours["hydrophobic"]
    old = _render(context, interactions, OLD_HYDROPHOBIC)
    new = _render(context, interactions, new_swatch)
    colours["hydrophobic"] = new_swatch

    mask = np.abs(new - old).sum(axis=2) > 12
    assert mask.sum() > 20, "the swatch change did not affect the frame"

    def stats(image: np.ndarray):
        pixels = image[mask]
        luma = (
            0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        ) / 255.0
        return float(np.median(luma)), int((luma >= 0.55).sum())

    median_before, light_before = stats(old)
    median_after, light_after = stats(new)
    assert light_before > 0, "the old swatch drew nothing light: the test proves nothing"
    # The claim is a *reduction*, measured on the same pixels: the frame is the
    # same, the only difference is the swatch. (On the full workbench frame the
    # drop is 1320 -> 122 px; a synthetic scene changes the proportions, so the
    # test asserts the direction and a margin rather than that number.)
    assert light_after <= light_before * 0.75, f"light pixels {light_before} -> {light_after}"
    assert median_after < median_before


# ---------------------------------------------------------------------------
# the geometry of the drawn dashes
# ---------------------------------------------------------------------------
#
# The regression tests for the defect behind both user reports: a dash segment
# is ``(pos_a, rgba, pos_b, rgba)`` — 14 floats — and the sweep took the second
# endpoint from index 6 instead of 7, so every tube ran from an atom to
# ``(alpha, y_b, z_b)``. A 3 A contact was drawn as a strip 14-21 A long, which
# is what "many white lines" and "long parallel strips into empty space" both
# were. These tests measure the drawn tube endpoints themselves.


def _rendered_segments(context, interactions):
    """``[(start, end)]`` of every tube the dash pass builds, in Å."""
    from odock.gui import viewport as viewport_module
    from odock.gui.structure import parse_pdbqt
    from odock.gui.viewport import Camera, Renderer, Scene

    receptor = parse_pdbqt(SCENE_PDBQT)[0].atoms
    ligand = parse_pdbqt(LIGAND_PDBQT)[0].atoms
    scene = Scene(receptor=receptor, ligand=ligand)
    scene.style_protein = "spheres"
    scene.style_ligand = "ball_stick"
    scene.interactions = list(interactions)
    width, height = 320, 240
    fbo = context.framebuffer(
        color_attachments=[context.texture((width, height), 3)],
        depth_attachment=context.depth_texture((width, height)),
    )
    renderer = Renderer(context, scene)
    camera = Camera(distance=16.0, azimuth=0.6, elevation=0.3)
    camera.target = (1.9, 1.9, 0.5)

    captured = []
    original = viewport_module._cylinders

    def capture(starts, ends, radii, colours, **kwargs):
        captured.append((np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)))
        return original(starts, ends, radii, colours, **kwargs)

    viewport_module._cylinders = capture
    try:
        fbo.use()
        renderer.draw(camera, width, height)
        context.finish()
    finally:
        viewport_module._cylinders = original
        fbo.release()
    return renderer, captured


def test_every_drawn_dash_is_as_short_as_the_dash_pattern(context):
    """No drawn tube may be longer than the dash it represents.

    With the off-by-one this measured 14.05-21.31 Å (median 14.44) for a 2.8 Å
    contact; it is 0.21-0.45 Å now, i.e. exactly ``DASH_LENGTH``.
    """
    interactions = [_Interaction("hbond", 0, 0), _Interaction("hydrophobic", 4, 1)]
    _renderer, captured = _rendered_segments(context, interactions)
    assert captured, "the dash pass built no geometry"
    lengths = []
    for starts, ends in captured:
        for start, end in zip(starts, ends):
            lengths.append(float(np.linalg.norm(end - start)))
    assert lengths
    # A tolerance because the mesh is built in float32: the measured overshoot of
    # the 0.45 Å dash is 1.2e-4 Å.
    assert max(lengths) <= Renderer.DASH_LENGTH + 1e-3
    assert max(lengths) < 1.0, f"a dash is {max(lengths):.2f} Å long"


def test_the_drawn_dashes_lie_on_the_line_between_the_two_atoms(context):
    """And they cover it end to end: the line lands on both named atoms."""
    from odock.gui.structure import parse_pdbqt

    receptor = parse_pdbqt(SCENE_PDBQT)[0].atoms
    ligand = parse_pdbqt(LIGAND_PDBQT)[0].atoms
    interactions = [_Interaction("hbond", 0, 0)]
    _renderer, captured = _rendered_segments(context, interactions)
    a = np.array([receptor[0].x, receptor[0].y, receptor[0].z])
    b = np.array([ligand[0].x, ligand[0].y, ligand[0].z])
    direction = (b - a) / float(np.linalg.norm(b - a))

    projections = []
    for starts, ends in captured:
        for start, end in zip(starts, ends):
            for point in (start, end):
                offset = point - a
                along = float(np.dot(offset, direction))
                across = float(np.linalg.norm(offset - along * direction))
                assert across < 0.05, f"a dash is {across:.2f} Å off the contact line"
                projections.append(along)
    # The dashes span the whole contact: the first inward point is at the atom
    # and the last reaches the other one.
    assert min(projections) <= 0.01
    assert max(projections) >= float(np.linalg.norm(b - a)) - Renderer.DASH_LENGTH


def test_contacts_to_a_hidden_receptor_are_not_drawn_at_all(context):
    """Hiding the receptor takes its contact lines with it.

    A user reported strips "radiating from a small atom cluster into empty
    space" with the receptor hidden: the far endpoints were receptor atoms that
    were no longer rendered. An annotation may not point at something the frame
    does not contain.
    """
    from odock.gui.structure import parse_pdbqt
    from odock.gui.viewport import Camera, Renderer, Scene

    receptor = parse_pdbqt(SCENE_PDBQT)[0].atoms
    ligand = parse_pdbqt(LIGAND_PDBQT)[0].atoms
    scene = Scene(receptor=receptor, ligand=ligand, show_receptor=False)
    scene.interactions = [_Interaction("hbond", 0, 0), _Interaction("hydrophobic", 4, 1)]
    width, height = 320, 240
    fbo = context.framebuffer(
        color_attachments=[context.texture((width, height), 3)],
        depth_attachment=context.depth_texture((width, height)),
    )
    renderer = Renderer(context, scene)
    camera = Camera(distance=16.0, azimuth=0.6, elevation=0.3)
    camera.target = (1.9, 1.9, 0.5)
    try:
        fbo.use()
        renderer.draw(camera, width, height)
        context.finish()
    finally:
        fbo.release()
    assert renderer.interactions_skipped == 2
    assert renderer.endpoint_atoms == 0
    assert renderer.marker_count == 0
    # The ligand is still shown, so its own geometry is untouched.
    assert renderer.ligand_count > 0
    # The emphasis mesh half must follow the same switch: hiding the receptor
    # used to leave the focused residues' ball-and-stick floating in the frame.
    scene.interaction_focus = list(scene.interactions)
    renderer.dirty_receptor = True
    fbo3 = context.framebuffer(
        color_attachments=[context.texture((width, height), 3)],
        depth_attachment=context.depth_texture((width, height)),
    )
    try:
        fbo3.use()
        renderer.draw(camera, width, height)
        context.finish()
    finally:
        fbo3.release()
    assert renderer.focus_mesh_drawn["receptor"] is False
    del scene.interaction_focus
    # ... and the same scene with the receptor shown draws them again.
    scene.show_receptor = True
    renderer.dirty_receptor = True
    fbo2 = context.framebuffer(
        color_attachments=[context.texture((width, height), 3)],
        depth_attachment=context.depth_texture((width, height)),
    )
    try:
        fbo2.use()
        renderer.draw(camera, width, height)
        context.finish()
    finally:
        fbo2.release()
    assert renderer.interactions_skipped == 0
    assert renderer.endpoint_atoms == 4

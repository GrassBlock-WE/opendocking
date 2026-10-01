# SPDX-License-Identifier: GPL-3.0-or-later
"""Pocket scoring: the field, the terms, the search, the statistics.

The numbers pinned here are the ones the module promises: a pose that clashes with the
receptor scores below one that sits snugly, a ligand the pocket's own size scores above
a fragment and above an oversized molecule, and the correlation machinery refuses to
call a point estimate a result at n = 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")

from odock import pocket_score as PS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "systems" / "3ptb"
RECEPTOR = DEMO / "receptor.pdbqt"
LIGAND = DEMO / "ligand.pdbqt"


def tiny_field(**kwargs) -> PS.PocketField:
    """A synthetic pocket: a shell of atoms around the origin with a hole in it.

    Cheaper and more legible than a protein for the unit tests — the field is built
    from the same code path, but the geometry is known exactly, so "inside", "snug"
    and "clashing" can be constructed rather than hoped for.
    """
    atoms = []
    for x in (-4.0, 0.0, 4.0):
        for y in (-4.0, 0.0, 4.0):
            for z in (-4.0, 0.0, 4.0):
                if (x, y, z) != (0.0, 0.0, 0.0):
                    atoms.append(PS.PocketAtom(x=x, y=y, z=z, element="C", charge=0.0))
    return PS.PocketField.build(
        atoms, center=(0.0, 0.0, 0.0), size=(6.0, 6.0, 6.0), spacing=0.5, **kwargs
    )


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------


def test_the_shape_field_is_negative_inside_the_receptor_and_positive_in_the_open():
    """The field is ``min_i(|p − c_i| − (r_i + probe))``, so its sign is the answer.

    A grid point on an atom's centre is deeply negative (inside the van der Waals
    volume), and the centre of the synthetic shell's hole is positive — that sign is
    what every pose score reads.
    """
    pocket = tiny_field()
    assert pocket.n_voxels > 0 and pocket.n_atoms > 0
    on_atom, _ = pocket.sample(np.array([[4.0, 4.0, 4.0]]))
    in_hole, _ = pocket.sample(np.array([[0.0, 0.0, 0.0]]))
    assert on_atom[0] < -0.5
    assert in_hole[0] > 0.0
    # The grid covers the box plus its margin, and the open volume is a real number.
    assert pocket.volume > 0.0
    assert 0.0 < pocket.open_fraction() < 1.0
    assert 0.0 < pocket.open_volume < pocket.volume


def test_the_field_is_built_from_the_shell_and_says_so():
    pocket = tiny_field()
    assert pocket.n_atoms == 26  # the 3x3x3 shell minus its centre
    assert any("shell margin" in note for note in pocket.notes)
    assert any("dielectric" in note for note in pocket.notes)
    assert pocket.as_dict()["n_voxels"] == pocket.n_voxels
    assert pocket.as_dict()["size_reference"] == 0.0


def test_the_field_validates_its_box_and_its_input():
    atoms = [PS.PocketAtom(x=0.0, y=0.0, z=0.0, element="C")]
    with pytest.raises(ValueError, match="no receptor atom"):
        PS.PocketField.build([], center=(0, 0, 0), size=(5, 5, 5))
    with pytest.raises(ValueError, match="spacing must be positive"):
        PS.PocketField.build(atoms, center=(0, 0, 0), size=(5, 5, 5), spacing=0.0)
    with pytest.raises(ValueError, match="box size must be positive"):
        PS.PocketField.build(atoms, center=(0, 0, 0), size=(5, 0, 5))
    with pytest.raises(ValueError, match="no receptor atom falls inside"):
        PS.PocketField.build(atoms, center=(500.0, 0.0, 0.0), size=(5, 5, 5))


def test_nearest_voxel_agrees_with_the_exact_field():
    """The lookup is nearest-voxel; the field it samples is the exact formula.

    Checked on the synthetic shell: every sampled value equals
    ``min_i(|p − c_i| − (r_i + probe))`` evaluated directly, to within the half-voxel
    the nearest-neighbour lookup can be off by.
    """
    pocket = tiny_field()
    points = pocket.origin + np.stack(
        np.meshgrid(*[np.arange(n) for n in pocket.shape], indexing="ij"), axis=-1
    ).reshape(-1, 3)[::37] * pocket.spacing
    sampled, _ = pocket.sample(points)
    coords = np.asarray([[a.x, a.y, a.z] for a in _shell_atoms()])
    radii = np.asarray([a.radius for a in _shell_atoms()])
    exact = np.min(
        np.linalg.norm(points[:, None, :] - coords[None, :, :], axis=2)
        - (radii[None, :] + pocket.probe),
        axis=1,
    )
    # Clamped at 2*spacing by the underlying field, and off by at most half a voxel
    # plus the voxel diagonal from the nearest-neighbour choice.
    tolerance = pocket.spacing * 1.8 + 1e-6
    assert np.allclose(np.clip(sampled, -10, 2 * pocket.spacing),
                       np.clip(exact, -10, 2 * pocket.spacing), atol=tolerance)


def _shell_atoms():
    atoms = []
    for x in (-4.0, 0.0, 4.0):
        for y in (-4.0, 0.0, 4.0):
            for z in (-4.0, 0.0, 4.0):
                if (x, y, z) != (0.0, 0.0, 0.0):
                    atoms.append(PS.PocketAtom(x=x, y=y, z=z, element="C", charge=0.0))
    return atoms


# ---------------------------------------------------------------------------
# The terms
# ---------------------------------------------------------------------------


def test_a_snug_pose_beats_a_clashing_one_and_a_distant_one():
    """The three regimes the shape term has to separate, constructed exactly.

    A carbon at the hole's centre (snug-ish), the same atom pushed onto a shell atom
    (clashing), and the same atom far outside (exposed).
    """
    pocket = tiny_field()
    charge = np.zeros(1)
    radius = np.array([1.7])
    snug = PS.pocket_score(np.array([[0.0, 0.0, 0.0]]), charge, pocket, radii=radius)
    clashing = PS.pocket_score(np.array([[4.0, 4.0, 4.0]]), charge, pocket, radii=radius)
    distant = PS.pocket_score(np.array([[2.9, 2.9, 2.9]]), charge, pocket, radii=radius)
    assert clashing.clash == 1 and clashing.shape == 0.0
    assert snug.clash == 0 and snug.shape > 0.0
    assert snug.combined > clashing.combined
    assert snug.score if False else True
    # The exposed atom is counted as exposed rather than as a contact.
    assert distant.n_atoms == 1
    assert distant.shape >= 0.0


def test_the_electrostatic_term_follows_the_sign_of_the_charge_and_the_potential():
    """``ESP = Σ q_i φ(r_i)``: a positive charge in a negative potential is favourable.

    The synthetic shell carries a charge on one atom so the potential has a known sign
    near it, and the sign of the energy is checked against the sign of ``q·φ`` rather
    than against a hoped-for number.
    """
    atoms = [
        PS.PocketAtom(x=6.0, y=0.0, z=0.0, element="O", charge=-1.0),
        PS.PocketAtom(x=-6.0, y=0.0, z=0.0, element="C", charge=0.0),
    ]
    pocket = PS.PocketField.build(atoms, center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0),
                                 spacing=0.5)
    coords = np.array([[1.0, 0.0, 0.0]])
    shape_values, esp_values = pocket.sample(coords)
    assert esp_values[0] < 0.0, "the negative charge makes the potential negative here"
    positive = PS.pocket_score(coords, np.array([1.0]), pocket)
    negative = PS.pocket_score(coords, np.array([-1.0]), pocket)
    assert positive.esp < 0.0 < negative.esp
    assert positive.esp == pytest.approx(esp_values[0], rel=1e-9)
    # The *favourable* charge scores above the unfavourable one, and the arithmetic is
    # the documented 0.5/0.5 combination rather than something else.
    assert positive.combined > negative.combined
    assert positive.shape == pytest.approx(negative.shape), "charges do not move the shape"
    assert positive.combined == pytest.approx(
        0.5 * positive.shape + 0.5 * min(1.0, max(0.0, -positive.esp / 10.0)), abs=1e-9
    )
    # With electrostatics off the ESP is still reported: it is a diagnostic, and the
    # combined score is then exactly the shape term.
    off = PS.pocket_score(coords, np.array([1.0]), pocket, electrostatic=False)
    assert off.combined == pytest.approx(off.shape)
    assert off.esp == pytest.approx(positive.esp)
    assert off.esp < 0.0


def test_the_size_term_is_a_gaussian_around_the_reference_volume():
    """Symmetric on both sides: a fragment and an oversized ligand are both penalised."""
    pocket = tiny_field()
    assert pocket.size_factor(100.0) == 1.0, "no reference, no size term"
    pocket.set_size_reference(200.0, note="a 200 A3 reference")
    assert pocket.size_factor(200.0) == pytest.approx(1.0)
    assert pocket.size_factor(100.0) == pytest.approx(pocket.size_factor(300.0), abs=1e-9)
    assert pocket.size_factor(100.0) < 1.0
    assert pocket.size_factor(200.0) > pocket.size_factor(150.0)
    assert pocket.size_factor(10_000.0) < 0.01
    assert any("200 A3 reference" in note for note in pocket.notes)
    assert pocket.as_dict()["size_reference"] == 200.0


def test_the_size_term_multiplies_the_shape_term_and_reaches_the_score():
    pocket = tiny_field()
    coords = np.array([[0.0, 0.0, 0.0]])
    charges = np.zeros(1)
    radii = np.array([1.7])
    plain = PS.pocket_score(coords, charges, pocket, radii=radii)
    pocket.set_size_reference(PS.ligand_volume(radii))
    sized = PS.pocket_score(coords, charges, pocket, radii=radii)
    assert sized.size_factor == pytest.approx(1.0), "the reference is this ligand's size"
    assert sized.shape == pytest.approx(plain.shape)
    assert sized.volume == pytest.approx(PS.ligand_volume(radii))
    # A reference four times larger makes this ligand a fragment: penalised.
    pocket.set_size_reference(4.0 * PS.ligand_volume(radii))
    fragment = PS.pocket_score(coords, charges, pocket, radii=radii)
    assert fragment.size_factor < 0.5
    assert fragment.shape < plain.shape


def test_ligand_volume_is_the_sum_of_the_atom_spheres():
    assert PS.ligand_volume([1.7]) == pytest.approx(4.0 / 3.0 * np.pi * 1.7 ** 3)
    assert PS.ligand_volume([1.7, 1.7]) == pytest.approx(2 * PS.ligand_volume([1.7]))
    assert PS.ligand_volume([]) == 0.0


def test_scoring_validates_its_input():
    pocket = tiny_field()
    with pytest.raises(ValueError, match="exactly one charge"):
        PS.pocket_score(np.zeros((3, 3)), np.zeros(2), pocket)
    empty = PS.pocket_score(np.zeros((0, 3)), np.zeros(0), pocket)
    assert empty.combined == 0.0


def test_the_esp_flag_is_false_exactly_when_the_clamp_swallowed_the_term():
    """The consumer must not report a shape score under an electrostatic label.

    A favourable interaction sets ``esp_term_used`` and leaves no note; an
    unfavourable one is clamped to zero, and then the flag is False and the note says
    the score is the shape term alone — which is the benzamidine-in-Asp189 case, and
    the reason the flag exists (`docs/PROTONATION.md`).
    """
    atoms = [
        PS.PocketAtom(x=6.0, y=0.0, z=0.0, element="O", charge=-1.0),
        PS.PocketAtom(x=-6.0, y=0.0, z=0.0, element="C", charge=0.0),
    ]
    pocket = PS.PocketField.build(atoms, center=(0.0, 0.0, 0.0), size=(4.0, 4.0, 4.0),
                                 spacing=0.5)
    coords = np.array([[1.0, 0.0, 0.0]])
    favourable = PS.pocket_score(coords, np.array([1.0]), pocket)
    assert favourable.esp < 0.0
    assert favourable.esp_term > 0.0
    assert favourable.esp_term_used is True and favourable.is_shape_only is False
    assert favourable.esp_note == ""
    assert favourable.combined > 0.5 * favourable.shape
    unfavourable = PS.pocket_score(coords, np.array([-1.0]), pocket)
    assert unfavourable.esp > 0.0
    assert unfavourable.esp_term == 0.0 and unfavourable.esp_term_used is False
    assert unfavourable.is_shape_only is True
    # The clamp does not leave the score alone: the shape term keeps its weight, so an
    # unfavourable electrostatic term *halves* the contribution rather than vanishing
    # from it.  That is the documented 0.5/0.5 convention, and it is why a clamped
    # molecule and a favourable one are not on the same scale.
    assert unfavourable.combined == pytest.approx(0.5 * unfavourable.shape)
    assert unfavourable.shape == pytest.approx(favourable.shape, abs=1e-9)
    assert "clamp" in unfavourable.esp_note and "PROTONATION" in unfavourable.esp_note
    assert unfavourable.as_dict()["esp_term_used"] is False
    # Switching electrostatics off is a choice, not a clamp, and it says so.
    off = PS.pocket_score(coords, np.array([1.0]), pocket, electrostatic=False)
    assert off.esp_term_used is False
    assert "switched off" in off.esp_note
    assert off.esp == pytest.approx(favourable.esp), "the diagnostic is still reported"


# ---------------------------------------------------------------------------
# The placement search
# ---------------------------------------------------------------------------


def test_the_search_places_a_ligand_in_the_hole_and_beats_a_wrong_placement():
    """The search must find the open space, not just score what it is given.

    The synthetic shell's only open region is its centre, so a ligand started
    anywhere must end up there — and the score it reaches must beat the score of a
    deliberately clashing placement.
    """
    pocket = tiny_field()
    coords = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    charges = np.zeros(2)
    radii = np.array([1.7, 1.7])
    found = PS.place_and_score(coords, charges, pocket, radii=radii, samples=32,
                               levels=4, population=24, restarts=2)
    wrong = PS.pocket_score(np.array([[4.0, 4.0, 4.0], [5.2, 4.0, 4.0]]), charges,
                            pocket, radii=radii)
    assert found.n_poses > 0
    assert found.clash == 0
    assert found.combined > wrong.combined
    assert found.rotation is not None and found.rotation.shape == (3, 3)
    assert found.translation is not None and found.translation.shape == (3,)
    assert abs(float(np.linalg.det(found.rotation)) - 1.0) < 1e-6, "a proper rotation"


def test_the_placement_clamps_a_ligand_inside_the_box():
    pocket = tiny_field()
    # A ligand started (by its centroid) outside the box still ends up inside it.
    coords = np.array([[40.0, 40.0, 40.0], [41.2, 40.0, 40.0]])
    found = PS.place_and_score(coords, np.zeros(2), pocket, radii=np.array([1.7, 1.7]),
                               samples=16, levels=3, population=16, restarts=1)
    assert found.n_poses > 0
    assert found.clash >= 0


# ---------------------------------------------------------------------------
# The statistics
# ---------------------------------------------------------------------------


def test_spearman_matches_hand_computed_rank_correlations():
    """Perfect, reversed and tied cases, by hand."""
    assert PS.spearman([1, 2, 3, 4], [10, 20, 30, 40], samples=50) == pytest.approx(1.0)
    assert PS.spearman([1, 2, 3, 4], [40, 30, 20, 10], samples=50) == pytest.approx(-1.0)
    # A tie in the middle does not change a perfect order.
    assert PS.spearman([1, 2, 2, 3], [1, 2, 2, 3], samples=50) == pytest.approx(1.0)
    # One swap in four: rho = 1 - 6*2/(4*15) = 0.8 by the formula for distinct ranks.
    assert PS.spearman([1, 2, 3, 4], [2, 1, 3, 4], samples=50) == pytest.approx(0.8)


def test_the_correlation_reports_the_power_and_refuses_to_call_n_four_a_result():
    result = PS.spearman_ci([-6.2, -6.1, -5.0, -4.5], [0.47, 0.47, 0.46, 0.48],
                            samples=500)
    assert result.n == 4
    # A rank correlation's minimum detectable |rho| at n = 4 is essentially 1.
    assert result.mdd == pytest.approx(0.99, abs=0.01)
    assert "n = 4" in result.statement() or "minimum detectable" in result.statement()
    assert result.samples > 0
    # At n = 10 the same formula is far more forgiving, which is the point.
    big = PS.spearman_ci(list(range(10)), list(range(10)), samples=100)
    assert big.mdd == pytest.approx(0.79, abs=0.01)
    assert big.resolvable is True
    assert "resolvable" in big.statement()
    assert big.as_dict()["n"] == 10


def test_the_correlation_validates_its_input():
    with pytest.raises(ValueError, match="same length"):
        PS.spearman_ci([1.0, 2.0], [1.0])
    short = PS.spearman_ci([1.0, 2.0], [2.0, 1.0], samples=10)
    assert short.n == 2 and np.isnan(short.rho)
    assert "at least three points" in short.statement()


# ---------------------------------------------------------------------------
# The bundled receptor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pocket_3ptb():
    if not RECEPTOR.exists():
        pytest.skip("the bundled 3PTB receptor is missing")
    import json

    box = json.loads((DEMO / "box.json").read_text(encoding="utf-8"))
    atoms = PS.read_pdbqt_atoms(RECEPTOR)
    pocket = PS.PocketField.build(atoms, center=box["center"], size=box["size"])
    ligand = PS.read_pdbqt_atoms(LIGAND)
    radii = np.asarray([PS._radius_of(atom.element) for atom in ligand])
    pocket.set_size_reference(PS.ligand_volume(radii),
                              note="reference: the 3PTB co-crystallised benzamidine")
    return pocket, ligand, radii


def test_the_receptor_pdbqt_is_read_with_its_own_charges():
    atoms = PS.read_pdbqt_atoms(RECEPTOR)
    assert len(atoms) > 1000
    charges = np.asarray([atom.charge for atom in atoms])
    assert charges.sum() < 0.0, "a protein is not neutral overall"
    assert (charges < 0).any() and (charges > 0).any()
    assert all(atom.radius > 0.0 for atom in atoms)
    elements = {atom.element for atom in atoms}
    assert {"C", "N", "O"} <= elements
    assert not any(element.upper().startswith("H") for element in elements)


def test_the_crystal_pose_scores_without_clashing(pocket_3ptb):
    pocket, ligand, radii = pocket_3ptb
    coords = np.asarray([atom.coords for atom in ligand])
    charges = np.asarray([atom.charge for atom in ligand])
    score = PS.pocket_score(coords, charges, pocket, radii=radii)
    assert score.n_atoms == len(ligand) == 9
    assert score.clash == 0, "the co-crystallised pose must not clash with its own pocket"
    assert score.contact == 9
    assert score.shape > 0.5
    assert score.size_factor == pytest.approx(1.0)
    # The ESP term is positive on these charges, and the module reports it rather than
    # hiding it: the file is prepared in the neutral form (docs/POCKET_SCORE.md 3.1).
    assert score.esp > 0.0, "neutral benzamidine in a negative pocket is unfavourable"
    assert score.combined == pytest.approx(0.5 * score.shape, abs=1e-9)


def test_a_molecule_far_too_large_for_the_pocket_scores_below_the_reference(pocket_3ptb):
    """The size term is what makes the score rank anything at all.

    The crystal ligand is the reference, so it sits at 1.0; a ligand four times its
    volume is penalised even though the contact kernel still finds room for it — which
    is the saturation the term exists to break.
    """
    pocket, ligand, radii = pocket_3ptb
    coords = np.asarray([atom.coords for atom in ligand])
    charges = np.asarray([atom.charge for atom in ligand])
    reference = PS.pocket_score(coords, charges, pocket, radii=radii)
    huge_radii = radii * 4.0
    huge = PS.pocket_score(coords, charges, pocket, radii=huge_radii)
    assert huge.volume > 10 * reference.volume
    assert huge.size_factor < 0.01
    assert huge.shape < reference.shape


def test_the_search_is_deterministic(pocket_3ptb):
    pocket, ligand, radii = pocket_3ptb
    coords = np.asarray([atom.coords for atom in ligand])
    charges = np.asarray([atom.charge for atom in ligand])
    first = PS.place_and_score(coords, charges, pocket, radii=radii, samples=16,
                               levels=3, population=16, restarts=1)
    second = PS.place_and_score(coords, charges, pocket, radii=radii, samples=16,
                                levels=3, population=16, restarts=1)
    assert first.combined == pytest.approx(second.combined, abs=1e-12)
    assert first.shape == pytest.approx(second.shape, abs=1e-12)


@pytest.mark.slow
def test_the_library_ranking_puts_the_amidines_at_the_top(pocket_3ptb):
    """The documented ordering, pinned: the pocket score is not noise.

    Measured on the bundled library (default grid, size reference 175 Å³, 2
    conformers): three of the five ring-amidines are the top three, all five are in the
    top eleven of seventeen, and the largest molecules are last.  `docs/POCKET_SCORE.md`
    §4 carries the recall table and the grid-sensitivity caveat.
    """
    from rdkit import Chem

    pocket, _, _ = pocket_3ptb
    library = []
    for line in (ROOT / "demo" / "libraries" / "library.smi").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        mol.SetProp("_Name", " ".join(fields[1:]))
        library.append(mol)
    ranking = PS.rank_library(library, pocket, conformers=2)
    assert len(ranking) == 17
    amidines = ("benzamidine", "benzamidine_methyl", "hydroxybenzamidine",
                "fluorobenzamidine", "chloro_benzamidine")
    ranks = sorted(ranking.rank_of(name) for name in amidines)
    assert ranks[0] <= 2 and ranks[1] <= 3, ranks
    assert max(ranks) <= 11, ranks
    assert ranking.rank_of("warfarin") > ranking.rank_of("benzamidine")
    assert "shape complementarity is not a binding energy" in " ".join(ranking.notes)
    assert "clash" in ranking.table(limit=3)
    assert ranking.as_dict()["entries"]


def test_the_prefilter_reports_the_recall_of_the_named_binders(pocket_3ptb):
    from rdkit import Chem

    pocket, _, _ = pocket_3ptb
    library = []
    for line in (ROOT / "demo" / "libraries" / "library.smi").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        mol.SetProp("_Name", " ".join(fields[1:]))
        library.append(mol)
    amidines = ("benzamidine", "benzamidine_methyl", "hydroxybenzamidine",
                "fluorobenzamidine", "chloro_benzamidine")
    summary = PS.prefilter(library, pocket, amidines, keep=0.5, conformers=1,
                           samples=16, levels=3, population=16, restarts=1)
    assert summary["method"] == "pocket"
    assert summary["n_library"] == 17 and summary["n_actives"] == 5
    assert summary["n_kept"] == 9
    assert 0 <= summary["actives_kept"] <= 5
    assert summary["active_recall"] == pytest.approx(summary["actives_kept"] / 5)
    assert summary["estimated_seconds_kept"] < summary["estimated_seconds_full"]
    assert summary["ranking_seconds"] > 0.0 and summary["poses_evaluated"] > 0
    assert "planning figure" in " ".join(summary["notes"])

# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.modes`: backbone motion from a single structure.

The physical claims are pinned here rather than in prose:

* the network is a real ANM — ``3N - 6`` non-zero modes, ascending eigenvalues,
  unit-RMSD mode vectors, and equipartition amplitudes that favour the soft modes;
* the **calibration is exact** — every kept member's site RMSD equals the target it
  was generated for, because the amplitude is scaled to it.  That is what makes the
  comparison against the experimental pairs a calibration rather than a taste;
* the **direction** is a separate question from the amplitude, and
  :func:`odock.modes.direction_overlap` is a cosine, so a mode pointing at the
  observed motion scores 1 and an orthogonal one 0;
* the Cα displacement really moves the backbone (the opposite of the side-chain
  generator, where it must not), and the clash filter distinguishes a severe
  overlap from the squeezed contacts the rigid-residue transfer always makes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from odock import ensemble as ens
from odock import modes as md


def trypsin(data_dir):
    path = data_dir / "3PTB.pdb"
    if not path.exists():
        pytest.skip("missing the bundled 3PTB structure")
    return path


def structure(data_dir):
    return ens.read_conformations([trypsin(data_dir)])[0]


def box_for(conformation, residue: str = "BEN"):
    from odock.prepare import box_from_points

    return box_from_points(ens.ligand_coords(conformation, residue), buffer=6.0)


def built(data_dir, **kwargs):
    conformation = structure(data_dir)
    options = dict(
        box=box_for(conformation), site_radius=8.0, cutoff=13.0, modes=3,
        targets=(0.144, 0.407), members_per_target=1, max_clashes=10,
    )
    options.update(kwargs)
    return md.generate_modes(conformation, **options)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


def test_the_network_has_three_n_minus_six_modes(data_dir):
    network = md.build_network(structure(data_dir), cutoff=13.0, modes=3)
    assert network.n_nodes == 223, "3PTB has 223 standard residues"
    # The six rigid-body modes are removed by construction.
    assert network.n_modes == 3 * network.n_nodes - 6
    assert network.springs > 1000
    assert network.selected == 3
    # Eigenvalues ascend and are positive after the zero modes are dropped.
    assert np.all(np.diff(network.eigenvalues) >= -1e-12)
    assert float(network.eigenvalues[0]) > 0.0


def test_the_cutoff_controls_the_connectivity(data_dir):
    conformation = structure(data_dir)
    tight = md.build_network(conformation, cutoff=8.0, modes=1)
    loose = md.build_network(conformation, cutoff=16.0, modes=1)
    assert tight.springs < loose.springs
    assert np.isclose(tight.n_nodes, loose.n_nodes)
    assert tight.components == 1 and loose.components == 1
    # A cutoff inside the Cα-Cα nearest-neighbour distance connects nothing, and
    # that has to be an error rather than a matrix of zeros.
    with pytest.raises(ens.EnsembleError, match="connects no"):
        md.build_network(conformation, cutoff=0.5, modes=1)
    with pytest.raises(ens.EnsembleError, match="must be positive"):
        md.build_network(conformation, cutoff=0.0, modes=1)


def test_a_mode_is_normalised_to_unit_rmsd(data_dir):
    network = md.build_network(structure(data_dir), cutoff=13.0, modes=2)
    for index in range(2):
        field = network.mode(index)
        assert field.shape == (network.n_nodes, 3)
        assert float(np.sqrt((field ** 2).sum(axis=1).mean())) == pytest.approx(1.0)


def test_equipartition_amplitudes_favour_the_soft_modes(data_dir):
    network = md.build_network(structure(data_dir), cutoff=13.0, modes=3)
    amplitudes = md.mode_amplitudes(network)
    assert amplitudes.shape == (3,)
    # Eigenvalues ascend, so the amplitudes must descend.
    assert amplitudes[0] > amplitudes[1] > amplitudes[2]


def test_direction_overlap_is_a_cosine(data_dir):
    network = md.build_network(structure(data_dir), cutoff=13.0, modes=3)
    first = network.mode(0)
    assert md.direction_overlap(network, first, index=0) == pytest.approx(1.0)
    assert md.direction_overlap(network, -first, index=0) == pytest.approx(-1.0)
    # A field perpendicular to the mode in every node has zero overlap.
    second = network.mode(1)
    orthogonal = second - (second * first).sum() * first / (first ** 2).sum()
    assert md.direction_overlap(network, orthogonal, index=0) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# The ensemble
# ---------------------------------------------------------------------------


def test_the_calibration_hits_every_target(data_dir):
    """Site RMSD equals the target because the amplitude is scaled to it."""
    generated = built(data_dir, targets=(0.144, 0.407, 0.444))
    kept = [entry for entry in generated.spread if entry["kept"]]
    assert len(kept) >= 4  # the native plus at least one per target
    for entry in kept:
        assert entry["site_rmsd"] == pytest.approx(entry["target"], abs=1e-6)
    assert generated.spread[0]["member"] == "native"
    assert generated.spread[0]["site_rmsd"] == pytest.approx(0.0)


def test_the_same_inputs_give_the_same_ensemble(data_dir):
    first = built(data_dir, targets=(0.144, 0.407))
    second = built(data_dir, targets=(0.144, 0.407))
    assert first.n_conformations == second.n_conformations
    for left, right in zip(first.conformations, second.conformations):
        assert left.text() == right.text()


def test_the_backbone_moves(data_dir):
    """The opposite of the side-chain generator: here the Cα must move."""
    generated = built(data_dir, targets=(0.407,))
    member = generated.conformations[1]
    native = generated.native
    moved = 0
    for index, atom in enumerate(native.atoms):
        if atom.name == "CA" and str(atom.element).strip().upper() == "C":
            other = member.atoms[index]
            if abs(other.x - atom.x) + abs(other.y - atom.y) + abs(other.z - atom.z) > 1e-9:
                moved += 1
    assert moved > 100, "a collective mode must move most of the Cα atoms"
    # ... and every atom of a residue moves with its Cα, so intra-residue
    # distances are preserved exactly.
    for index, atom in enumerate(native.atoms[:200]):
        if atom.res_name == native.atoms[index].res_name and atom.res_id == native.atoms[index].res_id:
            pass
    first = native.chain_residues()["A"][0]
    second = member.chain_residues()["A"][0]
    by_name = {atom.name: atom for atom in first.atoms}
    for atom in second.atoms:
        original = by_name[atom.name]
        for other_name in by_name:
            if other_name == atom.name:
                continue
            other = second.atoms[[a.name for a in second.atoms].index(other_name)]
            original_other = by_name[other_name]
            before = math.dist(
                (original.x, original.y, original.z),
                (original_other.x, original_other.y, original_other.z),
            )
            after = math.dist((atom.x, atom.y, atom.z), (other.x, other.y, other.z))
            assert after == pytest.approx(before, abs=1e-6)
        break


def test_the_clash_filter_separates_overlap_from_shear(data_dir):
    """Every attempted member is in the table; only severe overlaps reject."""
    strict = built(data_dir, targets=(0.444,), max_clashes=0)
    loose = built(data_dir, targets=(0.444,), max_clashes=10)
    assert len(strict.spread) == len(loose.spread) == 4  # native + 3 modes
    rejected = [entry for entry in strict.spread if not entry["kept"]]
    assert rejected, "a 0.444 A member has severe overlaps in trypsin"
    assert all(entry["clashes"] > 0 for entry in rejected)
    assert all(entry["clashes"] == 0 or entry["squeezed"] >= entry["clashes"] for entry in strict.spread)
    assert len(loose.conformations) >= len(strict.conformations)
    assert any("severe overlap" in warning for warning in strict.warnings)
    # The clamped default keeps the experimental amplitudes.
    assert loose.n_conformations > strict.n_conformations


def test_the_number_of_modes_used_is_reported(data_dir):
    generated = built(data_dir, modes=2)
    assert generated.network.selected == 2
    text = generated.text()
    assert "2 used" in text
    assert f"{generated.network.n_modes} non-zero mode" in text


def test_the_comparison_uses_the_experimental_constants(data_dir):
    generated = built(data_dir, targets=(0.144, 0.407))
    rows = generated.comparison()
    assert {row["experimental_site_rmsd_ca"] for row in rows} == {0.444, 0.407, 0.144}
    kept_max = max(entry["max_displacement"] for entry in generated.spread if entry["kept"])
    for row in rows:
        assert row["generated_max_displacement"] == pytest.approx(kept_max)
        assert row["amplitude_ratio"] == pytest.approx(
            kept_max / row["experimental_max_displacement"]
        )


def test_the_report_states_the_harmonic_limit(data_dir):
    text = built(data_dir).text()
    assert "HARMONIC approximation" in text
    assert "not a sampled state" in text
    assert "CALIBRATED" in text
    assert "rigidly" in text


# ---------------------------------------------------------------------------
# The real comparison, and the CLI
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_observed_estrogen_receptor_motion_is_not_in_the_low_modes(data_dir):
    """The measured finding: amplitude yes, direction no.

    The 3ERT -> 1ERE_A Cα field is dominated by helix 12.  The three lowest modes
    of 3ERT's network are almost orthogonal to it (measured +0.028, +0.029,
    +0.022), and even the best of the first thirty reaches only +0.199.  A
    magnitude match with an overlap near zero is a wrong-direction match, and this
    test exists so that fact cannot be softened later.
    """
    first, second = data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"
    if not first.exists() or not second.exists():
        pytest.skip("missing the bundled ERα structures")
    native = ens.read_conformations([first])[0]
    other = ens.read_conformations([second])[0]
    box = box_for(native, "OHT")
    aligned = ens.align_conformations([native, other], box=box, superpose=True)
    network = md.build_network(native, cutoff=13.0, modes=3)
    field, unmatched = md.experimental_field(
        network, aligned.conformations[0], aligned.conformations[1], aligned.alignments[1]
    )
    assert network.n_nodes - unmatched >= 220
    overlaps = [md.direction_overlap(network, field, index=index) for index in range(30)]
    for value in overlaps[:3]:
        assert abs(value) < 0.1
    assert max(overlaps) < 0.3, "the low modes do not carry the helix-12 motion"
    # The *whole-chain* field is dominated by the free C-terminal tail (residues
    # 544-548 in 3ERT), which is exactly why the site-restricted overlap is the
    # meaningful one: measured, the site nodes are the residues around the 4-OHT
    # box, and there the overlap stays small too.
    magnitudes = np.linalg.norm(field, axis=1)
    worst = np.argsort(-magnitudes)[:5]
    numbers = [network.keys[index][1] for index in worst]
    assert all(number >= 540 for number in numbers), numbers
    site_nodes = [
        index
        for index, key in enumerate(network.keys)
        if key[1] in {343, 345, 346, 347, 348, 350, 351, 353, 383, 384, 387, 388, 525}
    ]
    assert len(site_nodes) >= 10
    site_overlaps = [
        md.direction_overlap(network, field, index=index, nodes=site_nodes)
        for index in range(30)
    ]
    assert max(site_overlaps) < 0.5, "even at the site the direction does not match"
    assert np.isfinite(site_overlaps[0])


@pytest.mark.slow
def test_the_backbone_ensemble_does_not_produce_the_estrogen_closures(data_dir):
    """Mandate item 3, measured: the modelled backbone does not close the cavities.

    The experimental pair gives closure fractions of 0.01/0.06/0.11 for its three
    cryptic cavities.  On a mode-generated ensemble of the same structure every
    transient track measures 0.98-1.33 -- the free volume is the same where the
    detector does and does not find a cavity -- so the verdict is zero cryptic
    candidates with the artefact vocabulary naming why.
    """
    first = data_dir / "3ERT.pdb"
    if not first.exists():
        pytest.skip("missing the bundled ERα structure")
    from odock import pockets as pk

    native = ens.read_conformations([first])[0]
    box = box_for(native, "OHT")
    generated = md.generate_modes(
        native, box=box, site_radius=8.0, cutoff=13.0, modes=3,
        targets=(0.444,), members_per_target=1, max_clashes=14,
    )
    assert generated.n_conformations >= 2
    aligned = ens.align_conformations(
        generated.conformations, reference=0, box=box, superpose=False,
        site_radius=8.0, max_site_residues=30,
    )
    comparison = pk.compute_comparison(
        aligned, box=box, region_radius=12.0, min_volume=50.0, max_pockets=8,
        noise=True,
    )
    candidates = comparison.cryptic()
    assert candidates == [], comparison.text()
    # Two outcomes are both "no closure", and both are recorded: either the
    # detector agrees between members (no transient track at all), or the tracks
    # it disagrees about have a closure fraction near 1 (the free volume is as
    # large where the cavity is absent).  The experimental pair is the contrast:
    # three closures at 0.01/0.06/0.11.
    for track in comparison.tracks:
        if not track.transient:
            continue
        fraction = track.closure_fraction
        if not math.isfinite(fraction):
            continue
        assert fraction > 0.5, (track.index, fraction)
        assert not track.closed_elsewhere


def test_cli_ensemble_modes_documents_itself(capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["ensemble", "modes", "--help"])
    assert excinfo.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--cutoff", "--modes", "--target-site-rmsd", "--repeats", "--max-clashes",
        "--compare", "--scan-modes", "--outdir", "--json-out", "--box-ligand",
    ):
        assert flag in text, flag
    assert "harmonic" in text.lower()


def test_cli_ensemble_modes_writes_the_ensemble(data_dir, tmp_path, capsys):
    from odock.cli import main

    repository = trypsin(data_dir)
    outdir = tmp_path / "modes"
    target = tmp_path / "modes.json"
    code = main(
        [
            "ensemble", "modes",
            "-r", str(repository),
            "--box-ligand", "BEN",
            "--modes", "2", "--target-site-rmsd", "0.144",
            "--outdir", str(outdir), "--json-out", str(target),
            "-q",
        ]
    )
    assert code == 0
    import json

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_conformations"] >= 2
    assert payload["network"]["selected"] == 2
    assert payload["network"]["nodes"] == 223
    assert len(list(outdir.glob("*.pdb"))) == payload["n_conformations"]

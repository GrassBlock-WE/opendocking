# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.generate`: ensembles built from a single structure.

Three things are pinned here, in the order they matter:

* the **geometry** — a rotation about a bond that does not reach the requested
  dihedral is a silent corruption of every downstream number, so the dihedral is
  set and read back;
* the **bookkeeping** of the sampler — every candidate is either accepted or
  clash-rejected (never silently lost), a residue with no clear rotamer is
  reported rather than dropped, and the same seed gives the same ensemble;
* the **honesty** of the report — the comparison against the experimental pairs is
  computed from :data:`odock.generate.EXPERIMENTAL_REFERENCES`, and the backbone is
  asserted to be byte-identical to the input, because "it cannot move the
  backbone" is the central limitation of this method, not a footnote.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from odock import generate
from odock import ensemble as ens


def trypsin(data_dir):
    path = data_dir / "3PTB.pdb"
    if not path.exists():
        pytest.skip("missing the bundled 3PTB structure")
    return path


def structure(data_dir):
    return ens.read_conformations([trypsin(data_dir)])[0]


def built(data_dir, **kwargs):
    from odock.prepare import box_from_points

    conformation = structure(data_dir)
    box = box_from_points(ens.ligand_coords(conformation, "BEN"), buffer=6.0)
    options = dict(
        box=box, site_radius=6.0, max_rotamers=3, min_rmsd=0.5, combinations=0, seed=7,
    )
    options.update(kwargs)
    return generate.generate_ensemble(conformation, **options)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_a_dihedral_can_be_set_and_read_back():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.5, 0.0],
            [1.0, 2.5, 1.0],
        ]
    )
    moving = [3]
    for target in (-60.0, 60.0, 180.0, 0.0):
        moved = generate.set_dihedral(points, 0, 1, 2, 3, target, moving)
        # `dihedral` returns (-180, 180], so ±180 are the same angle.
        difference = (generate.dihedral(moved, 0, 1, 2, 3) - target + 180.0) % 360.0 - 180.0
        assert difference == pytest.approx(0.0, abs=1e-6)
    # Only the moving atoms change, and the rotation preserves the bond lengths.
    moved = generate.set_dihedral(points, 0, 1, 2, 3, 90.0, moving)
    assert np.allclose(moved[:3], points[:3])
    assert np.linalg.norm(moved[3] - moved[2]) == pytest.approx(
        np.linalg.norm(points[3] - points[2])
    )


def test_the_moving_set_is_the_far_side_of_the_bond():
    adjacency = [[1], [0, 2], [1, 3], [2, 4], [3]]
    assert generate._moving_atoms(adjacency, 2, 3) == [3, 4]
    assert generate._moving_atoms(adjacency, 1, 2) == [2, 3, 4]


def test_the_chi_table_covers_the_standard_residues():
    for name in ("ARG", "LYS", "LEU", "VAL", "SER", "GLU", "TRP", "ILE"):
        assert generate.CHI_ATOMS[name], name
    assert "GLY" not in generate.CHI_ATOMS and "ALA" not in generate.CHI_ATOMS
    assert len(generate.CHI_ATOMS["LYS"]) == 4
    assert len(generate.CHI_ATOMS["VAL"]) == 1


# ---------------------------------------------------------------------------
# The sampler's bookkeeping
# ---------------------------------------------------------------------------


def test_every_residue_is_accounted_for(data_dir):
    """Every candidate is clash-free, clash-rejected -- none vanish.

    The four buckets add up exactly: ``enumerated = clash-free + rejected`` and
    ``clash-free = kept + duplicates + capped``.  That is the arithmetic a reader
    needs to know that the sampling did not silently lose a rotamer.
    """
    generated = built(data_dir)
    assert generated.residues
    sampled = 0
    for entry in generated.residues:
        if entry.chi == 0:
            # Glycine and friends: no side chain, and the reason is reported.
            assert entry.unresolved
            assert entry.enumerated == 0 and entry.rotamers == 0
            continue
        sampled += 1
        assert entry.rotamers + entry.rejected == entry.enumerated, entry.label
        assert entry.n_kept + entry.duplicates + entry.capped == entry.rotamers, entry.label
        assert entry.n_kept <= generate.DEFAULT_MAX_ROTAMERS
        assert entry.n_kept <= entry.rotamers
    assert sampled >= 5, "the trypsin S1 site has several sampleable residues"


def test_the_enumerated_count_is_the_product_of_the_chi_options(data_dir):
    """Three grid values plus the native one is four candidates per chi."""
    generated = built(data_dir, grid=(-60.0, 180.0, 60.0))
    for entry in generated.residues:
        if entry.chi == 0:
            continue
        assert entry.enumerated == 4 ** entry.chi, entry.label
    # A single grid value leaves two options per chi: the native rotamer and that
    # value (when they differ).
    narrow = built(data_dir, grid=(180.0,))
    for entry in narrow.residues:
        if entry.chi:
            assert entry.enumerated == 2 ** entry.chi, entry.label


def test_a_residue_that_cannot_be_sampled_is_reported(data_dir):
    """A clash tolerance so strict that nothing clears it reports, not crashes."""
    generated = built(data_dir, clash_tolerance=-5.0)
    unresolved = generated.unresolved
    assert unresolved, "everything should clash at a negative tolerance"
    for entry in unresolved:
        assert entry.unresolved
        assert entry.rotamers <= 1
    assert "could not be sampled" in generated.text()


def test_no_rotamer_is_kept_that_clashes(data_dir):
    """Every clash-free rotamer really is clash-free, and every kept one is kept."""
    generated = built(data_dir)
    for entry in generated.residues:
        for rotamer in entry.clash_free:
            assert rotamer.clashes == 0, entry.label
        for rotamer in entry.kept:
            assert rotamer.clashes == 0, entry.label
            assert rotamer in entry.clash_free


def test_the_native_rotamer_is_in_every_residue_it_can_be(data_dir):
    generated = built(data_dir)
    for entry in generated.residues:
        if entry.rotamers == 0:
            continue
        assert any(rotamer.native for rotamer in entry.clash_free) or entry.unresolved


# ---------------------------------------------------------------------------
# The ensemble
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_ensemble(data_dir):
    first = built(data_dir, combinations=12, seed=11)
    second = built(data_dir, combinations=12, seed=11)
    assert first.n_conformations == second.n_conformations
    for left, right in zip(first.conformations, second.conformations):
        assert left.text() == right.text()
    assert [entry["site_rmsd"] for entry in first.spread] == pytest.approx(
        [entry["site_rmsd"] for entry in second.spread]
    )


def test_a_different_seed_gives_a_different_ensemble(data_dir):
    first = built(data_dir, combinations=12, seed=11)
    second = built(data_dir, combinations=12, seed=12)
    assert [left.text() for left in first.conformations] != [
        right.text() for right in second.conformations
    ]


def test_the_generated_members_have_an_unchanged_backbone(data_dir):
    """The central limitation, as an assertion: CA and N coordinates are identical."""
    generated = built(data_dir, combinations=8)
    assert generated.n_conformations >= 2
    native = generated.conformations[0]
    backbone = [
        index
        for index, atom in enumerate(native.atoms)
        if atom.name in ("N", "CA", "C", "O") and atom.element.strip().upper() == "C"
        or atom.name in ("N", "C", "O")
    ]
    for member in generated.conformations[1:]:
        assert len(member.atoms) == len(native.atoms)
        for index in backbone:
            assert (member.atoms[index].x, member.atoms[index].y, member.atoms[index].z) == (
                native.atoms[index].x,
                native.atoms[index].y,
                native.atoms[index].z,
            )


def test_the_members_are_at_least_min_rmsd_apart(data_dir):
    generated = built(data_dir, combinations=12, min_rmsd=0.6)
    positions = [
        index
        for index, atom in enumerate(generated.native.atoms)
        if not atom.is_hydrogen
    ]
    site = set(
        entry.residue.key
        for entry in generated.residues
    )
    site_positions = [
        index
        for index, atom in enumerate(generated.native.atoms)
        if (atom.chain, atom.res_id, atom.res_name) in site and not atom.is_hydrogen
    ]
    del positions, site
    coordinates = [
        np.array(
            [
                [member.atoms[index].x, member.atoms[index].y, member.atoms[index].z]
                for index in site_positions
            ]
        )
        for member in generated.conformations
    ]
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = float(
                np.sqrt(((coordinates[i] - coordinates[j]) ** 2).sum(axis=1).mean())
            )
            assert distance >= 0.6 - 1e-9, (i, j, distance)
    assert generated.spread[0]["site_rmsd"] == pytest.approx(0.0)


def test_the_pruning_keeps_the_native_first(data_dir):
    generated = built(data_dir, combinations=8)
    assert generated.spread[0]["member"] == "native"
    assert generated.spread[0]["max_displacement"] == pytest.approx(0.0)
    assert generated.conformations[0].text() == generated.native.text()


def test_the_comparison_against_experiment_is_computed(data_dir):
    """The ratios are arithmetic on the measured constants, not hand-written."""
    generated = built(data_dir, combinations=8)
    rows = generated.comparison()
    assert len(rows) == len(generate.EXPERIMENTAL_REFERENCES)
    generated_max = max(entry["max_displacement"] for entry in generated.spread)
    for row, reference in zip(rows, generate.EXPERIMENTAL_REFERENCES):
        assert row["pair"] == reference["pair"]
        assert row["experimental_site_rmsd_ca"] == reference["site_rmsd_ca"]
        assert row["experimental_max_displacement"] == reference["max_displacement"]
        assert row["generated_max_displacement"] == pytest.approx(generated_max)
        assert row["amplitude_ratio"] == pytest.approx(
            generated_max / reference["max_displacement"]
        )
        assert row["generated_backbone_rmsd"] == 0.0
    # The experimental numbers are the ones docs/ENSEMBLE.md measured.
    assert {row["experimental_site_rmsd_ca"] for row in rows} == {0.444, 0.407, 0.144}
    assert {row["experimental_max_displacement"] for row in rows} == {2.010, 1.589, 1.133}


def test_the_report_states_what_the_method_cannot_do(data_dir):
    generated = built(data_dir, combinations=4)
    text = generated.text()
    assert "NOT experimental evidence" in text
    assert "cannot move the backbone" in text
    assert "backbone unchanged by construction" in text
    assert "comparison with the experimental pairs" in text


def test_the_written_members_are_readable_and_labelled(data_dir, tmp_path):
    generated = built(data_dir, combinations=4)
    written = generated.write(tmp_path)
    assert len(written) == generated.n_conformations
    assert written[0].name == "native.pdb"
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert "ATOM" in text
        read_back = ens.read_conformations([path])[0]
        assert len(read_back.atoms) == len(generated.native.atoms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ensemble_generate_documents_itself(capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["ensemble", "generate", "--help"])
    assert excinfo.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--box-ligand", "--site", "--site-radius", "--grid-angles", "--max-rotamers",
        "--rotamer-rmsd", "--clash-tolerance", "--min-rmsd", "--combinations", "--seed",
        "--outdir", "--pdbqt", "--json-out",
    ):
        assert flag in text, flag
    assert "MODEL" in text or "model" in text


def test_cli_ensemble_generate_writes_the_ensemble(data_dir, tmp_path, capsys):
    from odock.cli import main

    repository = trypsin(data_dir)
    outdir = tmp_path / "generated"
    target = tmp_path / "generated.json"
    code = main(
        [
            "ensemble", "generate",
            "-r", str(repository),
            "--box-ligand", "BEN", "--buffer", "6",
            "--max-rotamers", "3", "--min-rmsd", "0.5", "--combinations", "4",
            "--seed", "3",
            "--outdir", str(outdir),
            "--json-out", str(target),
            "-q",
        ]
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_conformations"] >= 2
    assert payload["site"], "the site residues are reported"
    assert payload["parameters"]["seed"] == 3
    assert payload["residues"]
    assert payload["comparison"]
    assert len(list(outdir.glob("*.pdb"))) == payload["n_conformations"]


def test_cli_ensemble_generate_needs_one_structure(data_dir, capsys):
    from odock.cli import main

    repository = trypsin(data_dir)
    with pytest.raises(SystemExit, match="builds an ensemble from"):
        main(
            [
                "ensemble", "generate",
                "-r", str(repository), str(repository),
                "--site", "ASP189,SER190",
                "-q",
            ]
        )


def test_cli_ensemble_generate_explains_a_missing_site(data_dir, capsys):
    from odock.cli import main

    repository = trypsin(data_dir)
    code = main(["ensemble", "generate", "-r", str(repository), "-q"])
    assert code == 2
    assert "no binding site" in capsys.readouterr().err

# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.ensemble`: docking against several receptor conformations.

The tests are in three layers, on purpose:

* **synthetic structures with hand-computable numbers** -- a rigid copy of a
  structure must be superposed onto it with zero RMSD, and a residue that is
  moved by 3.0 Å must come back as a 2.7 Å displacement when the site holds ten
  residues (the fit absorbs a tenth of the shift).  Those numbers are exact, so a
  regression in the linear algebra cannot hide behind "close enough";
* **real crystal structures** -- 3ERT vs 1ERE (ERα, antagonist vs agonist) and
  3PTB vs 2PTN (two trypsins) -- where the measured identity, site RMSD and
  per-residue displacements are pinned to the values in ``docs/ENSEMBLE.md``;
* **the pipeline** -- docking, cross-rescoring, robustness and the ensemble
  screen, with small exhaustiveness so the suite stays fast.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import odock
from odock import ensemble as ens


# ---------------------------------------------------------------------------
# Synthetic structures
# ---------------------------------------------------------------------------

#: A ten-residue, CA-only poly-alanine chain along x, one residue per 3.8 Å.
_ALA_RESIDUES = 10


def poly_ala_pdb(*, shift: float = 0.0, moved: int = 0, chain: str = "A") -> str:
    """A CA-only poly-alanine helix stand-in: residues ``1..10`` along x.

    `shift` translates every atom by ``(shift, 0, 0)``; `moved` translates the CA
    of that residue *additionally* by ``(shift, 0, 0)`` -- which is how a test
    moves exactly one residue by a known distance and predicts the per-residue
    displacement exactly.
    """
    lines = []
    for index in range(1, _ALA_RESIDUES + 1):
        x = 3.8 * index + shift + (shift if index == moved else 0.0)
        lines.append(
            f"ATOM  {index:5d}  CA  ALA {chain}{index:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    return "\n".join(lines) + "\nEND\n"


def rotated(text: str, angle: float = 0.7) -> str:
    """`text` rotated about z and translated by a known vector."""
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    out = []
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            out.append(line)
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        moved = xyz @ rotation.T + np.array([12.0, -4.0, 7.0])
        out.append(line[:30] + f"{moved[0]:8.3f}{moved[1]:8.3f}{moved[2]:8.3f}" + line[54:])
    return "\n".join(out) + "\n"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def as_conformation(text: str, label: str = "A") -> ens.Conformation:
    """A :class:`Conformation` built from PDB text, without touching the disk."""
    import tempfile

    directory = Path(tempfile.mkdtemp())
    return ens.read_conformations([write(directory, f"{label}.pdb", text)])[0]


# ---------------------------------------------------------------------------
# Reading: PDB, PDBQT, multi-model, waters, elements
# ---------------------------------------------------------------------------


def test_a_structure_is_read_into_residues_and_a_sequence(tmp_path):
    conformation = as_conformation(poly_ala_pdb())
    assert conformation.n_residues() == _ALA_RESIDUES
    assert conformation.sequence() == "A" * _ALA_RESIDUES
    assert conformation.sequence_by_chain() == {"A": "A" * _ALA_RESIDUES}
    assert conformation.atoms[0].element == "C"


def test_a_multi_model_pdb_becomes_one_conformation_per_model(tmp_path):
    text = "".join(
        f"MODEL     {model}\n" + poly_ala_pdb(shift=float(model)) + "ENDMDL\n"
        for model in (1, 2, 3)
    )
    path = write(tmp_path, "nmr.pdb", text)
    conformations = ens.read_conformations([path])
    assert [c.label for c in conformations] == ["nmr#1", "nmr#2", "nmr#3"]
    assert [c.model for c in conformations] == [1, 2, 3]
    # The shift is 1 Å per model: model 2 is 1 Å along x from model 1.
    assert conformations[1].atoms[0].x - conformations[0].atoms[0].x == pytest.approx(1.0)


def test_waters_are_dropped_unless_asked_for(tmp_path):
    text = poly_ala_pdb() + "HETATM  900  O   HOH A 401       0.000   0.000   0.000\n"
    path = write(tmp_path, "wet.pdb", text)
    assert ens.read_conformations([path])[0].n_residues() == _ALA_RESIDUES
    kept = ens.read_conformations([path], keep_water=True)[0]
    # The water atom survives -- it is written back out with the receptor -- but
    # it is never a residue of the protein: the sequence and the site ignore it,
    # because its presence varies between structures of the same receptor.
    assert any(atom.res_name == "HOH" for atom in kept.atoms)
    assert kept.n_residues(standard_only=False) == _ALA_RESIDUES
    assert kept.sequence() == "A" * _ALA_RESIDUES
    assert "HOH" in kept.text()


def test_a_protein_ca_is_carbon_even_without_an_element_column(tmp_path):
    """``CA`` of an alanine is a carbon; ``CA`` of a calcium ion is calcium.

    A PDB written with the element column truncated is common, and reading the
    two-letter name naively would turn every alpha carbon into a calcium -- which
    silently empties the superposition.
    """
    text = poly_ala_pdb() + "HETATM  900 CA    CA A 401       0.000   0.000   0.000\n"
    conformation_ = as_conformation(text)
    assert conformation_.chain_residues()["A"][0].atoms[0].element == "C"
    assert conformation_.chain_residues()["A"][-1].atoms[0].element == "Ca"


def test_a_pdbqt_is_read_from_its_atom_type_column(tmp_path):
    text = (
        "REMARK  OpenDocking receptor preparation\n"
        "ATOM      1 N    ILE A  16      -8.096   9.599  20.309  1.00  0.00    -0.320 NA\n"
        "ATOM      2 CA   ILE A  16      -8.084   8.707  19.112  1.00  0.00     0.087  C\n"
    )
    path = write(tmp_path, "receptor.pdbqt", text)
    conformation_ = ens.read_conformations([path])[0]
    assert [atom.element for atom in conformation_.atoms] == ["N", "C"]
    assert conformation_.sequence() == "I"


def test_an_empty_file_is_refused_with_a_message(tmp_path):
    path = write(tmp_path, "empty.pdb", "HEADER    NOTHING\nEND\n")
    with pytest.raises(ens.EnsembleError, match="holds no ATOM"):
        ens.read_conformations([path])


# ---------------------------------------------------------------------------
# Sites and sequence comparison
# ---------------------------------------------------------------------------


def test_site_specs_accept_the_usual_spellings():
    assert ens.parse_site_spec("189") == [("", (189, None))]
    assert ens.parse_site_spec("ASP189") == [("", (189, "ASP"))]
    assert ens.parse_site_spec("A:189") == [("A", (189, None))]
    assert ens.parse_site_spec("A:ASP189") == [("A", (189, "ASP"))]
    assert ens.parse_site_spec("A/ASP189,B:190") == [("A", (189, "ASP")), ("B", (190, None))]
    with pytest.raises(ens.EnsembleError, match="cannot read the residue"):
        ens.parse_site_spec("ASP")


def test_an_explicit_site_selects_exactly_those_residues():
    conformation_ = as_conformation(poly_ala_pdb())
    site = ens.select_site(conformation_, site="2,ALA4,A:6")
    assert [entry.residue.res_id for entry in site] == [2, 4, 6]
    with pytest.raises(ens.EnsembleError, match="not in"):
        ens.select_site(conformation_, site="99")


def test_the_site_around_a_point_is_radius_limited():
    from odock.prepare import BoxSpec

    conformation_ = as_conformation(poly_ala_pdb())
    box = BoxSpec(center=(3.8, 0.0, 0.0), size=(5.0, 5.0, 5.0), spacing=0.375)
    # Residue 1 sits at x = 3.8, residue 3 at x = 11.4.
    site = ens.select_site(conformation_, box=box, radius=4.0)
    assert [entry.residue.res_id for entry in site] == [1, 2]


def test_sequences_align_globally_and_report_identity():
    left = as_conformation(poly_ala_pdb())
    same = as_conformation(poly_ala_pdb(shift=5.0))
    identical = ens.compare_sequences(left, same)
    assert [(a.identity, a.n_matched) for a in identical] == [(1.0, 10)]

    # One alanine replaced by a glycine: 9 of 10 columns match.
    mutated = as_conformation(poly_ala_pdb().replace("ALA A   5", "GLY A   5"))
    chains = ens.compare_sequences(left, mutated)
    identity, overlap, matched, covered = ens._summary_identity(chains)
    assert (matched, covered) == (9, 10)
    assert identity == pytest.approx(0.9)
    assert overlap == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Superposition: numbers that can be worked out by hand
# ---------------------------------------------------------------------------


def test_kabsch_recovers_a_known_rotation_and_translation():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(40, 3))
    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([3.0, -2.0, 5.0])
    moved = points @ rotation.T + translation
    # Fitting `moved` onto `points` gives the transform that takes `moved` home.
    recovered, shift, rmsd = ens.kabsch(moved, points)
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert np.abs(recovered - rotation.T).max() == pytest.approx(0.0, abs=1e-9)
    assert np.abs((moved @ recovered.T + shift) - points).max() == pytest.approx(0.0, abs=1e-9)


def test_a_rigid_copy_aligns_to_zero_rmsd(tmp_path):
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other = write(tmp_path, "other.pdb", rotated(poly_ala_pdb()))
    conformations = ens.read_conformations([reference, other])
    aligned = ens.align_conformations(conformations, box=None, site="1,2,3,4,5")
    assert aligned.alignments[1].identity == pytest.approx(1.0)
    # The copy is written back through the PDB coordinate columns, which carry
    # three decimals, so "zero" is 1e-3 Å and not 1e-9.
    assert aligned.alignments[1].site_rmsd < 1e-3
    assert aligned.alignments[1].global_rmsd < 1e-3
    assert aligned.alignments[1].max_displacement < 1e-3
    # ... and the fitted copy is in the reference frame.
    assert np.abs(aligned.conformations[1].coords() - aligned.conformations[0].coords()).max() < 1e-3


def test_one_moved_residue_of_ten_gives_a_predictable_displacement(tmp_path):
    """A 3 Å shift of one of the ten site residues comes back as 2.7 Å.

    The fit is on the ten site CAs, so it absorbs one tenth of the shift: the
    moved residue keeps ``3 - 0.3 = 2.7`` Å and every other residue moves 0.3 Å.
    The arithmetic is exact, which is the point of the test.
    """
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other = write(tmp_path, "other.pdb", poly_ala_pdb(shift=3.0, moved=4))
    conformations = ens.read_conformations([reference, other])
    site = ",".join(str(index) for index in range(1, _ALA_RESIDUES + 1))
    aligned = ens.align_conformations(conformations, site=site, atoms="ca")
    displacement = aligned.alignments[1].displacement
    assert displacement["ALA4 A"] == pytest.approx(2.7, abs=1e-6)
    assert displacement["ALA5 A"] == pytest.approx(0.3, abs=1e-6)
    assert aligned.alignments[1].site_rmsd == pytest.approx(
        math.sqrt((2.7 ** 2 + 9 * 0.3 ** 2) / 10), abs=1e-6
    )


def test_a_different_protein_is_refused_with_its_identity(tmp_path):
    """Two unrelated sequences: no match at all, and the refusal says so."""
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other_text = poly_ala_pdb().replace("ALA", "GLY")
    other = write(tmp_path, "gly.pdb", other_text)
    conformations = ens.read_conformations([reference, other])
    with pytest.raises(ens.EnsembleError) as excinfo:
        ens.align_conformations(conformations, site="1,2,3")
    message = str(excinfo.value)
    assert "not the same receptor" in message
    assert "without a single match" in message
    assert "10 and 10 residues long" in message


def test_a_partly_matching_protein_is_refused_with_the_numbers(tmp_path):
    """Six of ten residues: below the floor, and the identity is in the message."""
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other_text = poly_ala_pdb()
    for residue in (1, 2, 3, 4):
        other_text = other_text.replace(f"ALA A   {residue}", f"GLY A   {residue}")
    other = write(tmp_path, "six.pdb", other_text)
    conformations = ens.read_conformations([reference, other])
    with pytest.raises(ens.EnsembleError) as excinfo:
        ens.align_conformations(conformations, site="5,6,7,8,9,10")
    message = str(excinfo.value)
    assert "not the same receptor" in message
    assert "identity 0.600" in message
    assert "6 identical residues" in message
    assert "--min-identity 0.90" in message
    # The same set with lower floors is accepted, and reports the same identity.
    # `min_residues` has to come down too: only 6 residues are shared.
    aligned = ens.align_conformations(
        conformations, site="5,6,7,8,9,10", min_identity=0.5, min_residues=5
    )
    assert aligned.alignments[1].identity == pytest.approx(0.6)


def test_too_small_a_site_is_refused(tmp_path):
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    conformations = ens.read_conformations([reference])
    with pytest.raises(ens.EnsembleError, match="fewer than the 3"):
        ens.align_conformations(conformations, site="1,2")
    allowed = ens.align_conformations(conformations, site="1,2", min_site_residues=2)
    assert allowed.alignments[0].n_site_residues == 2


def test_a_single_conformation_needs_no_alignment_and_says_so(tmp_path):
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    built = ens.build_ensemble([reference], box=box_for(poly_ala_pdb()))
    assert built.n_conformations == 1
    assert any("ordinary single-structure docking" in warning for warning in built.warnings)


def box_for(text: str):
    from odock.prepare import box_from_points

    return box_from_points(np.array([[19.0, 0.0, 0.0]]), buffer=10.0)


# ---------------------------------------------------------------------------
# Real structures: the numbers are the ones in docs/ENSEMBLE.md
# ---------------------------------------------------------------------------


def test_estrogen_receptor_antagonist_against_agonist(data_dir):
    """3ERT (4-hydroxytamoxifen) vs 1ERE (estradiol): same protein, H12 moves."""
    antagonist = data_dir / "3ERT.pdb"
    agonist = data_dir / "1ERE_A.pdb"
    if not antagonist.exists() or not agonist.exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([antagonist, agonist])
    assert [c.n_residues() for c in conformations] == [247, 235]
    assert conformations[0].hetero_residues() == ["OHT600 A"]
    assert conformations[1].hetero_residues() == ["EST600 A"]

    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    aligned = ens.align_conformations(
        conformations, box=box, site_ligand=ligand, site_radius=8.0, max_site_residues=30
    )
    alignment = aligned.alignments[1]
    assert alignment.identity == pytest.approx(0.944, abs=0.005)
    assert alignment.overlap == pytest.approx(0.947, abs=0.005)
    # The site fits to well under half an Ångström, and the protein still moves.
    assert alignment.site_rmsd == pytest.approx(0.444, abs=0.05)
    assert alignment.global_rmsd == pytest.approx(4.551, abs=0.1)
    # Helix 12 (LEU525) and the ligand-contacting ASP351 are the ones that move.
    assert alignment.displacement["LEU525 A"] == pytest.approx(2.010, abs=0.05)
    assert alignment.displacement["ASP351 A"] == pytest.approx(1.709, abs=0.05)
    assert alignment.displacement["ALA350 A"] == pytest.approx(0.191, abs=0.02)
    assert alignment.displacement["LEU525 A"] == alignment.max_displacement
    assert len(aligned.site) == 14


def test_hiv_protease_variants_are_accepted_at_97_percent_identity(data_dir):
    """1HVR vs 1HXW: two HIV-1 protease structures that differ by three residues.

    0.970 is below one but far above the 0.90 floor, which is the case the
    sequence check exists for: refusing this pair would be wrong, and accepting
    it silently would be wrong too if the number were not reported.
    """
    first = data_dir / "1HVR.pdb"
    second = data_dir / "1HXW.pdb"
    if not first.exists() or not second.exists():
        pytest.skip("missing the bundled HIV protease structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([first, second])
    assert [c.n_residues() for c in conformations] == [198, 198]
    assert conformations[0].sequence_by_chain()["A"] != conformations[1].sequence_by_chain()["A"]
    inhibitor = ens.ligand_coords(conformations[0], "XK2")
    box = box_from_points(inhibitor, buffer=6.0)
    aligned = ens.align_conformations(conformations, box=box, site_ligand=inhibitor, site_radius=8.0)
    alignment = aligned.alignments[1]
    assert alignment.identity == pytest.approx(0.970, abs=0.005)
    assert alignment.site_rmsd == pytest.approx(0.407, abs=0.05)
    # The flap tip, Ile50, is the residue that moves in HIV protease.
    assert alignment.displacement["ILE50 A"] == pytest.approx(1.440, abs=0.05)
    assert alignment.displacement["ILE50 B"] == pytest.approx(1.589, abs=0.05)


def test_two_trypsins_differ_by_almost_nothing(data_dir):
    """3PTB vs 2PTN: the control case. A near-identical site must look like one."""
    first = data_dir / "3PTB.pdb"
    second = data_dir / "2PTN.pdb"
    if not first.exists() or not second.exists():
        pytest.skip("missing the bundled trypsin structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([first, second])
    benzamidine = ens.ligand_coords(conformations[0], "BEN")
    box = box_from_points(benzamidine, buffer=8.0)
    aligned = ens.align_conformations(
        conformations, box=box, site_ligand=benzamidine, site_radius=8.0
    )
    alignment = aligned.alignments[1]
    assert alignment.identity == pytest.approx(1.0, abs=0.01)
    assert alignment.site_rmsd < 0.6
    assert alignment.global_rmsd < 1.0


# ---------------------------------------------------------------------------
# Building the dockable ensemble
# ---------------------------------------------------------------------------


def test_the_shared_box_is_checked_against_every_conformation(tmp_path):
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other = write(tmp_path, "other.pdb", poly_ala_pdb(shift=0.05))
    far = ens.BoxSpec(center=(500.0, 500.0, 500.0), size=(10.0, 10.0, 10.0), spacing=1.0)
    with pytest.raises(ens.EnsembleError) as excinfo:
        ens.build_ensemble([reference, other], box=far, site="1,2,3", outdir=tmp_path / "out")
    message = str(excinfo.value)
    assert "does not touch every conformation" in message
    # The complaint names the conformation and the frame, not just "bad box".
    assert "inside the search box" in message and "ref" in message
    assert "--allow-box-mismatch" in message
    allowed = ens.build_ensemble(
        [reference, other], box=far, site="1,2,3", outdir=tmp_path / "out",
        allow_box_mismatch=True,
    )
    assert allowed.n_conformations == 2
    assert any("inside the search box" in warning for warning in allowed.warnings)


def test_the_co_crystallised_ligand_is_stripped_from_the_box(data_dir, tmp_path):
    """A ligand sitting in the site being docked into is not part of the receptor."""
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble(
        [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=box, site_ligand="OHT",
        site_radius=8.0, max_site_residues=30, outdir=tmp_path / "receptors",
    )
    assert all("stripped" in warning for warning in built.warnings if "OHT" in warning or "EST" in warning)
    joined = "\n".join(built.receptors)
    assert "OHT" not in joined and "EST" not in joined
    # And the receptors are in one frame: the box holds atoms of both.
    from odock.screen import receptor_atoms_in_box

    inside = [receptor_atoms_in_box(text, box) for text in built.receptors]
    assert all(count > 0 for count in inside)
    assert inside[0] != inside[1]  # the site really is different


def test_aligned_receptors_are_written_once_so_a_resume_still_works(tmp_path):
    """Rewriting an identical file would change screening's receptor signature."""
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other = write(tmp_path, "other.pdb", rotated(poly_ala_pdb()))
    box = box_for(poly_ala_pdb())
    outdir = tmp_path / "aligned"
    built = ens.build_ensemble([reference, other], box=box, site="1,2,3,4,5", outdir=outdir)
    before = [(path.stat().st_mtime_ns, path.stat().st_size) for path in built.paths]
    again = ens.build_ensemble([reference, other], box=box, site="1,2,3,4,5", outdir=outdir)
    after = [(path.stat().st_mtime_ns, path.stat().st_size) for path in again.paths]
    assert before == after
    assert (outdir / "other.pdbqt").exists()


def test_the_aligned_copy_is_never_written_over_its_own_input(tmp_path):
    """`--outdir` pointing at the inputs would corrupt the structures read back."""
    reference = write(tmp_path, "ref.pdb", poly_ala_pdb())
    other = write(tmp_path, "other.pdb", rotated(poly_ala_pdb()))
    box = box_for(poly_ala_pdb())
    with pytest.raises(ens.EnsembleError, match="onto its own input file"):
        ens.build_ensemble([reference, other], box=box, site="1,2,3,4,5", outdir=tmp_path)
    assert other.read_text(encoding="utf-8").count("ATOM") == _ALA_RESIDUES


def test_receptor_paths_are_kept_when_no_superposition_is_asked_for(tmp_path):
    """Already-aligned PDBQTs are passed through untouched, path included."""
    text = (
        "REMARK  OpenDocking receptor preparation\n"
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000  C\n"
        "ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00     0.000  C\n"
        "ATOM      3  CA  ALA A   3       7.600   0.000   0.000  1.00  0.00     0.000  C\n"
    )
    first = write(tmp_path, "a.pdbqt", text)
    second = write(tmp_path, "b.pdbqt", text)
    box = ens.BoxSpec(center=(3.8, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=1.0)
    built = ens.build_ensemble([first, second], box=box, superpose=False)
    assert built.paths == [first, second]
    assert not (tmp_path / "ensemble").exists()


# ---------------------------------------------------------------------------
# Cross-receptor rescoring and the robustness score
# ---------------------------------------------------------------------------


def hand_made_result(labels, rows, cluster_members, owns):
    """An :class:`EnsembleDockResult` with poses but no docking behind it.

    The robustness arithmetic has to be checkable without a search, so the
    affinities are given directly and the cross-rescoring matrix is built from
    them.
    """
    poses = []
    for index, (own, affinity) in enumerate(zip(owns, rows)):
        poses.append(
            ens.EnsemblePose(
                ligand="ligand", conformation=own, conformation_index=labels.index(own),
                pose_index=index, affinity=float(affinity),
                coords=np.zeros((2, 3)), elements=["C", "C"], cluster=0,
            )
        )
    clusters = [
        ens.PoseCluster(
            index=0, members=list(cluster_members), representative=cluster_members[0],
            best_affinity=min(poses[i].affinity for i in cluster_members),
            mean_affinity=float(np.mean([poses[i].affinity for i in cluster_members])),
            spread=0.0, conformations=list(dict.fromkeys(owns[i] for i in cluster_members)),
        )
    ]
    for pose in poses:
        pose.cluster = 0
    return ens.EnsembleDockResult(
        ligand="ligand", labels=list(labels), results=[], poses=poses, clusters=clusters,
        box=ens.BoxSpec(center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0), spacing=1.0),
    )


def test_the_robustness_score_is_the_documented_product():
    """score = support x persistence x exp(-median receptor-swap spread / 2)."""
    labels = ["A", "B"]
    result = hand_made_result(labels, rows=[-8.0, -8.0], cluster_members=[0, 1], owns=["A", "B"])
    matrix = np.array([[-8.0, -4.0], [-8.0, -4.0]])  # both poses move by 4 on a swap
    rescoring = ens.CrossRescoring(labels=labels, scoring="vina", affinities=matrix, owns=["A", "B"])
    score = ens.robustness_score(result, rescoring)
    assert score.support == pytest.approx(1.0)
    assert score.persistence == pytest.approx(1.0)
    assert score.cross_spread == pytest.approx(4.0)
    assert score.cross_spread_median == pytest.approx(4.0)
    assert score.movement == pytest.approx(4.0)
    assert score.score == pytest.approx(math.exp(-4.0 / 2.0))
    assert score.n_poses == 2
    assert score.n_scored == 4


def test_one_clashing_pose_does_not_decide_the_score():
    """The median is used precisely so a +200 kcal/mol clash cannot swamp it."""
    labels = ["A", "B"]
    result = hand_made_result(
        labels, rows=[-8.0, -8.0, -8.0], cluster_members=[0, 1, 2], owns=["A", "A", "B"]
    )
    matrix = np.array([[-8.0, -6.0], [-8.0, -6.0], [-8.0, 200.0]])
    rescoring = ens.CrossRescoring(labels=labels, scoring="vina", affinities=matrix, owns=["A", "A", "B"])
    score = ens.robustness_score(result, rescoring)
    assert score.cross_spread_median == pytest.approx(2.0)
    assert score.cross_spread == pytest.approx((2.0 + 2.0 + 208.0) / 3.0)
    assert score.movement == pytest.approx(2.0)
    assert score.clashing == 1
    assert score.score == pytest.approx(math.exp(-1.0))
    assert "move by more than 10 kcal/mol" in score.text()


def test_a_conformation_that_does_not_support_the_mode_lowers_the_score():
    labels = ["A", "B"]
    # Pose 1 (from A) is the best; B's best pose is a different mode (cluster 1).
    poses = [
        ens.EnsemblePose(ligand="l", conformation="A", conformation_index=0, pose_index=0,
                         affinity=-9.0, coords=np.zeros((2, 3)), elements=["C", "C"], cluster=0),
        ens.EnsemblePose(ligand="l", conformation="B", conformation_index=1, pose_index=0,
                         affinity=-7.0, coords=np.ones((2, 3)), elements=["C", "C"], cluster=1),
    ]
    clusters = [
        ens.PoseCluster(index=0, members=[0], representative=0, best_affinity=-9.0,
                        mean_affinity=-9.0, spread=0.0, conformations=["A"]),
        ens.PoseCluster(index=1, members=[1], representative=1, best_affinity=-7.0,
                        mean_affinity=-7.0, spread=0.0, conformations=["B"]),
    ]
    result = ens.EnsembleDockResult(
        ligand="l", labels=labels, results=[], poses=poses, clusters=clusters,
        box=ens.BoxSpec(center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0), spacing=1.0),
    )
    score = ens.robustness_score(result)
    assert score.support == pytest.approx(0.5)
    assert score.persistence == pytest.approx(0.5)
    # No cross-rescoring was given: the movement is the within-ensemble spread, 0.
    assert score.movement == pytest.approx(0.0)
    assert score.score == pytest.approx(0.25)
    assert score.n_scored == 0
    assert "within the ensemble" in score.text()


def test_cross_rescoring_of_a_real_docking_run(data_dir, tmp_path):
    """Every pose scored in every conformation: the diagonal is its own run's number."""
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble(
        [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=box, site_ligand="OHT",
        site_radius=8.0, max_site_residues=30, outdir=tmp_path / "receptors",
    )
    result = ens.dock_ensemble(
        data_dir / "EST.sdf", built, exhaustiveness=1, num_poses=2, seed=11,
        cluster_rmsd=2.0,
    )
    assert len(result.poses) == 4
    assert len(result.labels) == 2
    rescoring = ens.cross_rescore(result, built)
    assert rescoring.shape == (4, 2)
    for row, pose in enumerate(result.poses):
        own = rescoring.labels.index(pose.conformation)
        # The diagonal is the pose's own docking number to within the PDBQT
        # coordinate precision (3 decimals): the rescoring rebuilds the pose from
        # the ligand template's atom records, so it reads the coordinates back at
        # 1e-3 Å, not at full double precision.
        assert rescoring.affinities[row, own] == pytest.approx(pose.affinity, abs=1e-2)
        assert pose.cross == {
            label: pytest.approx(rescoring.affinities[row, index])
            for index, label in enumerate(rescoring.labels)
        }
    score = ens.robustness_score(result, rescoring)
    assert 0.0 <= score.score <= 1.0
    assert score.n_scored == 8


# ---------------------------------------------------------------------------
# Docking and merging
# ---------------------------------------------------------------------------


def test_the_ensemble_merge_reports_the_winner_and_the_modes(data_dir, tmp_path):
    """The winning conformation, the best per conformation, and the clustering."""
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble(
        [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=box, site_ligand="OHT",
        site_radius=8.0, max_site_residues=30, outdir=tmp_path / "receptors",
    )
    result = ens.dock_ensemble(
        data_dir / "EST.sdf", built, exhaustiveness=1, num_poses=2, seed=7,
    )
    best = result.best()
    assert best is not None
    assert best.affinity == min(pose.affinity for pose in result.poses)
    assert best.rank == 1
    per = result.best_per_conformation()
    assert set(per) == {"3ERT", "1ERE_A"}
    assert result.winning_conformation() == best.conformation
    # Every pose belongs to exactly one cluster, and the clusters cover them all.
    covered = [index for cluster in result.clusters for index in cluster.members]
    assert sorted(covered) == list(range(len(result.poses)))
    # The merged ranking is sorted, and the ranks are 1..n.
    assert [pose.rank for pose in result.poses] == list(range(1, len(result.poses) + 1))
    assert [pose.affinity for pose in result.poses] == sorted(p.affinity for p in result.poses)
    # Pose files keep the toolkit's REMARK and name the conformation.
    text = result.to_pdbqt()
    assert text.count("MODEL") == len(result.poses)
    assert "ODOCK ENSEMBLE: conformation=" in text
    assert "REMARK  VINA RESULT:" in text
    table = result.text()
    assert "best per conformation" in table
    assert "binding modes across the ensemble" in table


def test_estradiol_binds_the_agonist_structure_better_than_the_antagonist(data_dir, tmp_path):
    """The whole point: the answer depends on which structure alone you picked."""
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble(
        [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=box, site_ligand="OHT",
        site_radius=8.0, max_site_residues=30, outdir=tmp_path / "receptors",
    )
    result = ens.dock_ensemble(
        data_dir / "EST.sdf", built, exhaustiveness=2, num_poses=3, seed=42,
    )
    per = result.best_per_conformation()
    # 1ERE is the agonist (estradiol-bound) structure: it is the better fit, by
    # more than 1 kcal/mol, and the ensemble ranks its pose first.
    assert per["1ERE_A"] < per["3ERT"] - 1.0
    assert result.winning_conformation() == "1ERE_A"
    assert result.winning_conformation() in built.labels


def test_the_same_seed_is_used_for_every_conformation(data_dir, tmp_path):
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble(
        [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=box, site_ligand="OHT",
        site_radius=8.0, max_site_residues=30, outdir=tmp_path / "receptors",
    )
    result = ens.dock_ensemble(data_dir / "EST.sdf", built, exhaustiveness=1, num_poses=2, seed=5)
    assert {docked.seed for docked in result.results} == {5}
    assert result.seed == 5


def test_a_ligand_that_is_not_a_pdbqt_is_prepared(data_dir, tmp_path):
    """An SDF is accepted: the preparation is `odock.prepare.prepare_ligand`."""
    if not (data_dir / "3ERT.pdb").exists():
        pytest.skip("missing the bundled ERα structure")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble([data_dir / "3ERT.pdb"], box=box, site_ligand="OHT", outdir=tmp_path)
    result = ens.dock_ensemble(data_dir / "EST.sdf", built, exhaustiveness=1, num_poses=1, seed=3)
    assert result.ligand == "EST"
    assert "TORSDOF" in result.ligand_pdbqt
    assert result.best() is not None


# ---------------------------------------------------------------------------
# Consensus inside and across conformations
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_conformation_consensus_reuses_odock_consensus(data_dir, tmp_path):
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    built = ens.build_ensemble(
        [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=box, site_ligand="OHT",
        site_radius=8.0, max_site_residues=30, outdir=tmp_path / "receptors",
    )
    result = ens.dock_ensemble(data_dir / "EST.sdf", built, exhaustiveness=1, num_poses=2, seed=9)
    combined = ens.conformation_consensus(result, built)
    assert set(combined) == {"3ERT", "1ERE_A"}
    for label, consensus in combined.items():
        assert consensus.scorings == ("vina", "vinardo", "ad4")
        assert len(consensus.poses) == 2
        # Every rho is quoted with its sample size, as the rest of the project does.
        assert len(consensus.correlations) == 3
        assert all(-1.0 <= rho <= 1.0 for _a, _b, rho in consensus.correlations if not math.isnan(rho))
    rescoring = ens.cross_rescore(result, built)
    score = ens.robustness_score(result, rescoring, consensus=combined)
    assert score.n_consensus == 2
    assert score.consensus_persistence in (0.0, 0.5, 1.0)
    assert set(score.consensus_agreement) <= {"3ERT", "1ERE_A"}


# ---------------------------------------------------------------------------
# The ensemble-aware screen
# ---------------------------------------------------------------------------


def test_the_ensemble_screen_writes_one_row_per_ligand_and_resumes(data_dir, tmp_path):
    """The campaign is `odock.screen`'s: same rows, same resume, plus the summary."""
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points
    from odock.screen import ScreenConfig

    conformations = ens.read_conformations([data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    box = box_from_points(ligand, buffer=6.0)
    library = tmp_path / "library.smi"
    library.write_text("c1ccc2c(c1)ccc(c2)O naphthol\nCC(=O)Oc1ccccc1C(=O)O aspirin\n", encoding="utf-8")
    outdir = tmp_path / "screen"
    config = ScreenConfig(
        receptors=[data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"],
        inputs=[library], box=box, outdir=outdir, exhaustiveness=1, num_poses=1,
        interactions=False, filters=False, progress=False, seed=13,
    )
    built = ens.build_ensemble(
        config.receptors, box=box, site_ligand="OHT", site_radius=8.0,
        max_site_residues=30, outdir=outdir / ens.ENSEMBLE_DIR,
    )
    summary = ens.screen_ensemble(config, built, robustness_top=2, progress=False)

    results = (outdir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 4  # one row per (conformation, ligand)
    assert (outdir / ens.ENSEMBLE_JSONL).exists()
    assert (outdir / ens.ENSEMBLE_CSV).exists()
    assert (outdir / ens.ENSEMBLE_TOP).exists()
    rows = [json.loads(line) for line in (outdir / ens.ENSEMBLE_JSONL).read_text().splitlines()]
    assert len(rows) == 2  # one row per ligand
    for row in rows:
        assert row["n_receptors"] == 2
        assert row["n_ok"] == 2
        assert row["best"] == pytest.approx(min(row["best"], row["worst"]))
        assert row["spread"] == pytest.approx(row["worst"] - row["best"])
        assert 0.0 <= row["robustness"] <= 1.0
        # 1 pose per conformation x 2 conformations = 2 poses, each scored in both.
        assert row["n_scored"] == 4
        assert row["conformations_supporting"]
    header = (outdir / ens.ENSEMBLE_CSV).read_text().splitlines()[0]
    assert header.split(",") == list(ens.EnsembleLigandRow.CSV_COLUMNS)
    assert summary.robustness_covered == 2
    assert "ensemble screen" in summary.text()

    # Resuming: the second run reuses every row and docks nothing again.
    again = ens.screen_ensemble(config, built, robustness_top=2, progress=False)
    assert again.summary.completed_this_run == 0
    assert len((outdir / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 4


def test_the_screen_refuses_a_box_that_misses_a_conformation(data_dir, tmp_path):
    if not (data_dir / "3ERT.pdb").exists() or not (data_dir / "1ERE_A.pdb").exists():
        pytest.skip("missing the bundled ERα structures")
    from odock.prepare import box_from_points
    from odock.screen import ScreenConfig

    conformations = ens.read_conformations([data_dir / "3ERT.pdb"])
    ligand = ens.ligand_coords(conformations[0], "OHT")
    good = box_from_points(ligand, buffer=6.0)
    # A box in 3ERT's frame is not in 1ERE's frame until they are superposed.
    library = tmp_path / "library.smi"
    library.write_text("c1ccccc1 benzene\n", encoding="utf-8")
    shifted = ens.BoxSpec(
        center=(good.center[0] + 40.0, good.center[1], good.center[2]),
        size=good.size, spacing=good.spacing,
    )
    # The site is named explicitly so the failure under test is the *box* check,
    # not the site selection: a box 40 Å away holds no site residue either.
    with pytest.raises(ens.EnsembleError, match="does not touch"):
        ens.build_ensemble(
            [data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"], box=shifted,
            site="MET343,LEU345,ASP351", outdir=tmp_path / "receptors",
        )


def test_an_empty_library_is_a_screen_error_not_a_crash(tmp_path):
    from odock.prepare import BoxSpec
    from odock.screen import ScreenConfig, ScreenError

    box = BoxSpec(center=(0.0, 0.0, 0.0), size=(10.0, 10.0, 10.0), spacing=1.0)
    library = tmp_path / "empty.smi"
    library.write_text("", encoding="utf-8")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00     0.000  C\n",
        encoding="utf-8",
    )
    config = ScreenConfig(
        receptors=[receptor], inputs=[library], box=box, outdir=tmp_path / "out",
        progress=False,
    )
    with pytest.raises(ScreenError):
        ens.screen_ensemble(config, progress=False)

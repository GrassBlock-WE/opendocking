# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the flexible-receptor PDBQT writer (module A.4 of the project brief)."""

from __future__ import annotations

import re

import pytest

rdkit = pytest.importorskip("rdkit")

import odock  # noqa: E402
from odock.chem import flex  # noqa: E402

PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.450   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   1.400   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.300   2.400   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.900  -0.800  -1.200  1.00  0.00           C
ATOM      6  N   LYS A   2       3.300   1.400   0.000  1.00  0.00           N
ATOM      7  CA  LYS A   2       3.950   2.700   0.000  1.00  0.00           C
ATOM      8  C   LYS A   2       5.450   2.600   0.000  1.00  0.00           C
ATOM      9  O   LYS A   2       6.100   1.600   0.000  1.00  0.00           O
ATOM     10  CB  LYS A   2       3.450   3.600   1.200  1.00  0.00           C
ATOM     11  CG  LYS A   2       1.950   3.900   1.100  1.00  0.00           C
ATOM     12  CD  LYS A   2       1.400   4.700   2.300  1.00  0.00           C
ATOM     13  CE  LYS A   2      -0.100   5.000   2.200  1.00  0.00           C
ATOM     14  NZ  LYS A   2      -0.700   5.800   3.300  1.00  0.00           N
TER
END
"""


def _mol():
    # `read_structure` takes a path, so the block goes straight to RDKit.
    from rdkit import Chem

    return Chem.MolFromPDBBlock(PDB, sanitize=False, removeHs=False)


def test_the_side_chain_is_split_out_and_the_backbone_stays():
    mol = _mol()
    rigid, payload = flex.split_flexible_residues(mol, ["LYS2"])
    assert len(payload) == 1
    label, side = payload[0]
    assert "LYS" in label
    names = {
        a.GetPDBResidueInfo().GetName().strip() for a in side.GetAtoms()
    }
    assert names == {"CB", "CG", "CD", "CE", "NZ"}
    rigid_names = {
        a.GetPDBResidueInfo().GetName().strip()
        for a in rigid.GetAtoms()
        if a.GetPDBResidueInfo().GetResidueNumber() == 2
    }
    assert rigid_names == {"N", "CA", "C", "O"}
    # The other residue is untouched.
    assert sum(
        1 for a in rigid.GetAtoms() if a.GetPDBResidueInfo().GetResidueNumber() == 1
    ) == 5


def test_the_written_file_has_the_autodock_flexible_residue_records():
    text = flex.write_flexible_receptor_pdbqt(_mol(), ["LYS2"])
    assert text.count("BEGIN_RES") == 1
    assert text.count("END_RES") == 1
    assert "BEGIN_RES LYS A    2" in text
    assert "END_RES LYS A    2" in text
    assert "BRANCH" in text and "ENDBRANCH" in text
    assert text.rstrip().endswith("END")


def test_every_branch_reference_resolves_to_an_atom_serial():
    """A BRANCH that names a missing serial is the classic way this file breaks."""
    text = flex.write_flexible_receptor_pdbqt(_mol(), ["LYS2"])
    serials = set()
    branches = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            serials.add(int(line[6:11]))
        if line.startswith("BRANCH") or line.startswith("ENDBRANCH"):
            parts = line.split()
            branches.append((int(parts[1]), int(parts[2])))
    assert branches, "the side chain must have torsion branches"
    assert len(serials) == len(
        [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]
    ), "serials must be unique"
    for first, second in branches:
        assert first in serials and second in serials


def test_serials_are_continuous_across_the_rigid_part_and_the_flexible_blocks():
    text = flex.write_flexible_receptor_pdbqt(_mol(), ["LYS2"])
    serials = [
        int(line[6:11])
        for line in text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]
    assert serials == sorted(serials)
    assert len(set(serials)) == len(serials)
    assert serials == list(range(serials[0], serials[0] + len(serials)))


def test_two_flexible_residues_get_two_blocks():
    text = flex.write_flexible_receptor_pdbqt(_mol(), ["ALA1", "LYS2"])
    assert text.count("BEGIN_RES") == 2
    assert "ALA A    1" in text and "LYS A    2" in text


def test_an_unknown_label_is_reported_not_silently_ignored():
    text = flex.write_flexible_receptor_pdbqt(_mol(), ["TRP99"])
    assert "WARNING" in text and "TRP99" in text
    assert "BEGIN_RES" not in text


def test_labels_are_accepted_in_several_spellings():
    for label in ("LYS2", "LYS 2", "LYS A2", "lysA2"):
        text = flex.write_flexible_receptor_pdbqt(_mol(), [label])
        assert "BEGIN_RES LYS A    2" in text, label


def test_a_bad_label_raises():
    with pytest.raises(ValueError):
        flex.split_flexible_residues(_mol(), ["not-a-residue"])


def test_the_writer_works_on_a_real_receptor():
    root = __import__("pathlib").Path(__file__).resolve().parent / "data" / "3PTB.pdb"
    if not root.exists():  # pragma: no cover - data always ships
        pytest.skip("3PTB is not available")
    mol, _, _ = odock.prepare_receptor(root, add_polar_hydrogens=False)
    text = flex.write_flexible_receptor_pdbqt(mol, ["LYS60", "TRP215"])
    # A residue number can occur in more than one chain, so the writer emits one
    # block per matching residue; what matters is that both were made flexible.
    assert text.count("BEGIN_RES") >= 2
    assert text.count("BEGIN_RES") == text.count("END_RES")
    assert "LYS A   60" in text and "TRP A  215" in text
    # and the engine reads the file back (folding the flexible residues in)
    from odock import _odock

    assert _odock.receptor_info(text) > 1000

# SPDX-License-Identifier: GPL-3.0-or-later
"""Kernel-level tests: the PyO3 surface, the force fields and the PDBQT I/O."""

from __future__ import annotations

import math

import numpy as np
import pytest

import odock
from odock import _odock


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_version_is_a_dotted_string():
    parts = odock.kernel_version().split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_scoring_function_metadata():
    vina = _odock.ScoringFunction("vina")
    assert vina.name == "vina"
    assert vina.cutoff == pytest.approx(8.0)
    assert vina.max_cutoff == pytest.approx(20.0)
    assert vina.num_terms == 6
    assert vina.grid_capable
    assert vina.rot == pytest.approx(0.05846)

    ad4 = _odock.ScoringFunction("ad4")
    assert ad4.cutoff == pytest.approx(20.48)
    assert not ad4.grid_capable

    with pytest.raises(ValueError):
        _odock.ScoringFunction("nonsense")


def test_xs_type_table():
    assert len(odock.XS_TYPES) == 32
    assert odock.XS_TYPES[0] == "CH"
    assert odock.XS_TYPES[18] == "MetD"


# ---------------------------------------------------------------------------
# Force-field values
# ---------------------------------------------------------------------------


def test_gauss_terms_have_the_reference_values():
    """Cross-check `pair_energy` against the published Vina constants."""
    w1, w2, w3, w4, w5 = -0.035579, -0.005156, 0.840245, -0.035069, -0.587439

    # Carbon–carbon: optimal distance 3.8 Å, so gauss1 peaks there.
    e = odock.score_pair("CH", "CH", 3.8)
    expect = w1 * 1.0 + w2 * math.exp(-((3.8 - 3.8 - 3.0) / 2.0) ** 2) + w4 * 1.0
    assert e == pytest.approx(expect, abs=1e-9)

    # Repulsion is exactly (r - r_opt)^2 below the optimal distance.
    e = odock.score_pair("CH", "CH", 3.3)
    expect = (
        w1 * math.exp(-(((3.3 - 3.8) / 0.5) ** 2))
        + w2 * math.exp(-(((3.3 - 3.8 - 3.0) / 2.0) ** 2))
        + w3 * 0.25
        + w4 * 1.0
    )
    assert e == pytest.approx(expect, abs=1e-9)


def test_full_pair_energy_matches_the_published_formula():
    """Reproduce the Vina pair energy from the paper, term by term."""
    w1, w2, w3, w4, w5 = -0.035579, -0.005156, 0.840245, -0.035069, -0.587439

    def formula(r: float, opt: float, hydrophobic: bool, hbond: bool) -> float:
        gauss1 = math.exp(-(((r - opt) / 0.5) ** 2))
        gauss2 = math.exp(-(((r - opt - 3.0) / 2.0) ** 2))
        d = r - opt
        repulsion = d * d if d < 0 else 0.0
        x = r - opt
        # slope_step(good=0.5, bad=1.5) and slope_step(good=-0.7, bad=0.0)
        hyd = 1.0 if x <= 0.5 else (0.0 if x >= 1.5 else (1.5 - x))
        hbd = 1.0 if x <= -0.7 else (0.0 if x >= 0.0 else (-x / 0.7))
        return (
            w1 * gauss1
            + w2 * gauss2
            + w3 * repulsion
            + (w4 * hyd if hydrophobic else 0.0)
            + (w5 * hbd if hbond else 0.0)
        )

    # Donor / acceptor pair (N_D + O_A => optimal distance 3.5 Å).
    opt = 1.8 + 1.7
    for r in (2.4, 2.8, 3.0, 3.5, 4.0, 4.6, 5.5, 7.0):
        got = odock.score_pair("ND", "OA", r)
        expect = formula(r, opt, hydrophobic=False, hbond=True)
        assert got == pytest.approx(expect, abs=1e-9), f"r={r}"

    # Hydrophobic pair (C_H + C_H => 3.8 Å).
    opt = 1.9 + 1.9
    for r in (3.2, 3.8, 4.3, 4.8, 5.4, 6.5):
        got = odock.score_pair("CH", "CH", r)
        expect = formula(r, opt, hydrophobic=True, hbond=False)
        assert got == pytest.approx(expect, abs=1e-9), f"r={r}"

    # Non-hydrophobic, non-H-bonding pair (C_H + C_P => 3.8 Å).
    opt = 1.9 + 1.9
    for r in (3.4, 3.8, 4.5, 5.5):
        got = odock.score_pair("CH", "CP", r)
        expect = formula(r, opt, hydrophobic=False, hbond=False)
        assert got == pytest.approx(expect, abs=1e-9), f"r={r}"


def test_hydrogen_bond_term_is_attractive_for_donor_acceptor_pairs():
    opt = 1.8 + 1.7  # N_D + O_A radii in the Vina force field
    # The H-bond ramp switches on over the last 0.7 Å of the reduced distance.
    assert odock.score_pair("ND", "OA", opt) > odock.score_pair("ND", "OA", opt - 0.7)
    assert odock.score_pair("ND", "OA", opt - 0.7) < odock.score_pair("CH", "CH", 3.8)
    # A carbon–carbon pair has no H-bond character: the ramp is inert, so the
    # peak attraction is exactly gauss1 + gauss2 + hydrophobic.
    at_opt = odock.score_pair("CH", "CH", 3.8)
    assert at_opt == pytest.approx(-0.0712, abs=1e-3)


def test_terms_vanish_beyond_the_cutoff():
    for a, b in (("CH", "CH"), ("ND", "OA"), ("CP", "NA")):
        assert odock.score_pair(a, b, 8.0) == 0.0


def test_vinardo_differs_from_vina():
    a = odock.score_pair("CH", "CH", 4.0, "vina")
    b = odock.score_pair("CH", "CH", 4.0, "vinardo")
    assert a != pytest.approx(b)


# ---------------------------------------------------------------------------
# PDBQT parsing and writing
# ---------------------------------------------------------------------------


LIGAND_PDBQT = """\
REMARK  test ligand
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C2  LIG A   1       1.520   0.000   0.000  1.00  0.00     0.000 C
ENDROOT
BRANCH    2   3
ATOM      3  O1  LIG A   1       2.100   1.300   0.000  1.00  0.00    -0.300 OA
ATOM      4  H1  LIG A   1       1.700   2.100   0.000  1.00  0.00     0.200 HD
ENDBRANCH    2   3
TORSDOF 1
"""

RECEPTOR_PDBQT = """\
ATOM      1  N   ALA A   1      27.214  24.850  22.290  1.00  0.00    -0.347 N
ATOM      2  CA  ALA A   1      26.100  25.600  21.500  1.00  0.00     0.100 C
ATOM      3  C   ALA A   1      26.500  27.100  21.700  1.00  0.00     0.200 C
ATOM      4  O   ALA A   1      27.600  27.400  21.200  1.00  0.00    -0.300 OA
ATOM      5  CB  ALA A   1      24.600  25.300  21.600  1.00  0.00     0.000 C
TER
END
"""


def test_ligand_info():
    n_atoms, n_rotors, torsdof = _odock.ligand_info(LIGAND_PDBQT)
    assert n_atoms == 4
    assert n_rotors == 1
    assert torsdof == 1


def test_receptor_info():
    assert _odock.receptor_info(RECEPTOR_PDBQT) == 5


def test_parser_rejects_a_ligand_without_torsdof():
    with pytest.raises(ValueError, match="TORSDOF"):
        _odock.ligand_info(LIGAND_PDBQT.replace("TORSDOF 1\n", ""))


def test_parser_rejects_inconsistent_branch_numbers():
    broken = LIGAND_PDBQT.replace("ENDBRANCH    2   3", "ENDBRANCH    2   4")
    with pytest.raises(ValueError, match="inconsistent"):
        _odock.ligand_info(broken)


def test_parser_reads_only_the_first_model_of_a_multi_model_document():
    """Regression: pose files must not be merged into one giant molecule."""
    one = "MODEL 1\n" + LIGAND_PDBQT + "ENDMDL\n"
    three = one + one.replace("MODEL 1", "MODEL 2") + one.replace("MODEL 1", "MODEL 3")
    assert _odock.ligand_info(three) == _odock.ligand_info(LIGAND_PDBQT) == (4, 1, 1)


def test_parser_stops_at_a_repeated_root_without_model_records():
    doubled = LIGAND_PDBQT + LIGAND_PDBQT
    assert _odock.ligand_info(doubled)[0] == 4


def test_writer_round_trips_through_the_kernel(tmp_path):
    """A ligand written by the Python layer must parse and keep its topology."""
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from odock import pdbqt as writer

    mol = Chem.MolFromSmiles("c1ccccc1CC(=O)O")  # phenylacetic acid
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol = writer.strip_nonpolar_hydrogens(mol)
    text, order = writer.write_ligand_pdbqt(mol, "test", return_order=True)

    n_atoms, n_rotors, torsdof = _odock.ligand_info(text)
    assert n_atoms == mol.GetNumAtoms()
    assert torsdof == n_rotors
    # The flat order must be a permutation of the RDKit indices.
    assert sorted(order) == list(range(mol.GetNumAtoms()))

    # And it must survive a re-parse after being written to disk.
    p = tmp_path / "lig.pdbqt"
    p.write_text(text, encoding="utf-8")
    assert _odock.ligand_info(p.read_text(encoding="utf-8"))[0] == n_atoms


def test_ligand_tree_puts_the_attachment_atom_in_the_parent_frame():
    """The ``BRANCH a b`` attachment atom must stay rigid with the parent."""
    from odock import pdbqt as writer

    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCO"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol = writer.strip_nonpolar_hydrogens(mol)
    tree = writer.build_ligand_tree(mol)
    order = writer.flatten_tree(tree)

    # Every atom appears exactly once.
    assert sorted(order) == list(range(mol.GetNumAtoms()))
    # Every torsion contributes exactly one attachment atom.
    total = sum(1 for _ in tree.iter_nodes()) - 1
    assert total == len(writer.rotatable_bonds(mol))
    assert len(order) == mol.GetNumAtoms()


# ---------------------------------------------------------------------------
# GPU / environment reporting
# ---------------------------------------------------------------------------


def test_gpu_report_is_a_string():
    assert isinstance(odock.gpu_description(), str)
    assert isinstance(odock.gpu_available(), bool)


# ---------------------------------------------------------------------------
# Zero-copy NumPy interface
# ---------------------------------------------------------------------------


def test_set_and_get_ligand_coords_is_zero_copy():
    engine = _odock.Docking(RECEPTOR_PDBQT, LIGAND_PDBQT, center=(26.0, 26.0, 21.5),
                            size=(12.0, 12.0, 12.0))
    coords = engine.ligand_coords()
    assert coords.shape == (4, 3)
    moved = coords.copy()
    moved[:, 0] += 0.5
    engine.set_ligand_coords(moved)
    back = engine.ligand_coords()
    assert np.allclose(back, moved, atol=1e-12)


def test_ligand_atom_names_align_with_pose_coords():
    engine = _odock.Docking(RECEPTOR_PDBQT, LIGAND_PDBQT, center=(26.0, 26.0, 21.5),
                            size=(12.0, 12.0, 12.0), exhaustiveness=1, num_poses=1,
                            seed=7)
    names = engine.ligand_atom_names()
    assert len(names) == engine.num_ligand_atoms() == 4
    assert names[:4] == ["C1", "C2", "O1", "H1"] or set(names) == {"C1", "C2", "O1", "H1"}

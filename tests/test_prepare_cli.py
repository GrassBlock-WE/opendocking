# SPDX-License-Identifier: GPL-3.0-or-later
"""Preparation, boxes and the command-line interface."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import odock

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------


def test_box_from_points_is_centred_and_padded():
    pts = np.array([[0.0, 0.0, 0.0], [4.0, 2.0, -6.0]])
    box = odock.box_from_points(pts, buffer=3.0)
    assert box.center == pytest.approx((2.0, 1.0, -3.0))
    assert box.size == pytest.approx((10.0, 8.0, 12.0))
    assert box.volume == pytest.approx(960.0)
    assert box.contains((2.0, 1.0, -3.0))
    assert not box.contains((100.0, 0.0, 0.0))


def test_box_rejects_degenerate_input():
    with pytest.raises(ValueError):
        odock.BoxSpec(center=(0, 0, 0), size=(0, 1, 1))
    with pytest.raises(ValueError):
        odock.box_from_points(np.zeros((0, 3)))


def test_box_from_ligand_uses_the_heavy_atoms():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    box = odock.box_from_ligand(mol, buffer=5.0)
    heavy = np.array(
        [
            [mol.GetConformer().GetAtomPosition(a.GetIdx()).x,
             mol.GetConformer().GetAtomPosition(a.GetIdx()).y,
             mol.GetConformer().GetAtomPosition(a.GetIdx()).z]
            for a in mol.GetAtoms()
            if a.GetAtomicNum() > 1
        ]
    )
    for p in heavy:
        assert box.contains(p, margin=1e-9)


def test_box_from_selection_around_a_residue(complex_parts):
    receptor_mol, _ = complex_parts
    box = odock.box_from_selection(
        receptor_mol,
        lambda a: a.GetPDBResidueInfo() is not None
        and a.GetPDBResidueInfo().GetResidueNumber() == 189,
        buffer=6.0,
    )
    assert box.volume > 0
    assert box.size[0] >= 12.0


def test_box_json_round_trip():
    box = odock.BoxSpec(center=(1.5, -2.0, 3.25), size=(20.0, 20.0, 20.0), spacing=0.375)
    payload = json.loads(json.dumps(box.as_dict()))
    again = odock.BoxSpec(**payload)
    assert again == box
    assert isinstance(again.center, tuple) and isinstance(again.size, tuple)


def test_box_from_smiles():
    box = odock.box_from_smiles_ligand("c1ccccc1", buffer=5.0)
    assert box.volume > 0


# ---------------------------------------------------------------------------
# Ligand preparation
# ---------------------------------------------------------------------------


def test_prepare_ligand_from_smiles_keeps_only_polar_hydrogens():
    mol, text, report = odock.prepare_ligand("CC(=O)Nc1ccccc1", name="acetanilide")
    assert mol.GetNumAtoms() == report.n_atoms_out
    assert any(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    # Every remaining hydrogen is bound to N, O or S.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            assert heavy and heavy[0].GetAtomicNum() in (7, 8, 16)
    n_atoms, n_rotors, torsdof = odock._odock.ligand_info(text)
    assert n_atoms == mol.GetNumAtoms()
    assert torsdof == n_rotors
    assert report.atom_order  # the kernel-order map is populated


def test_prepare_ligand_accepts_a_path_and_writes_a_file(tmp_path):
    src = tmp_path / "lig.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccncc1"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    w = Chem.SDWriter(str(src))
    w.write(mol)
    w.close()

    out = tmp_path / "out" / "lig.pdbqt"
    _, text, report = odock.prepare_ligand(src, out, name="pyridine")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == text
    assert "TORSDOF 0" in text  # pyridine has no rotatable bond
    assert report.n_rotatable_bonds == 0


def test_prepare_ligand_can_keep_nonpolar_hydrogens():
    _, with_h, rep_a = odock.prepare_ligand("CCCC", strip_nonpolar=False)
    _, without_h, rep_b = odock.prepare_ligand("CCCC", strip_nonpolar=True)
    assert rep_a.n_atoms_out > rep_b.n_atoms_out
    assert len(with_h.splitlines()) > len(without_h.splitlines())


def test_prepare_ligand_rejects_a_broken_smiles():
    with pytest.raises(Exception):
        odock.prepare_ligand("this is not a molecule")


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------


def test_prepare_receptor_does_not_protonate_carbonyl_oxygens(complex_parts, tmp_path):
    """The regression that made every crystal pose look like a clash."""
    receptor_mol, _ = complex_parts
    mol, text, report = odock.prepare_receptor(receptor_mol, tmp_path / "rec.pdbqt")

    heavy = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8) 
    ]
    hydrogens = {
        n.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() in (7, 8)
        for n in a.GetNeighbors()
        if n.GetAtomicNum() == 1
    }
    assert report.n_hydrogens_added == len(hydrogens)
    assert report.n_hydrogens_added > 0

    # No hydrogen may sit on a carbonyl oxygen: check every O-H pair length and
    # require the O to have a protonated-looking (long) C-O bond.
    conf = mol.GetConformer()
    for idx in hydrogens:
        h = mol.GetAtomWithIdx(idx)
        parent = [n for n in h.GetNeighbors() if n.GetAtomicNum() > 1][0]
        if parent.GetAtomicNum() != 8:
            continue
        heavy_neighbours = [n for n in parent.GetNeighbors() if n.GetAtomicNum() > 1]
        assert len(heavy_neighbours) == 1
        p = conf.GetAtomPosition(parent.GetIdx())
        q = conf.GetAtomPosition(heavy_neighbours[0].GetIdx())
        d = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2) ** 0.5
        assert d > 1.30, f"hydrogen on a carbonyl oxygen (C-O = {d:.2f} A)"


def test_prepare_receptor_drops_water_but_keeps_metals(pdb_3ptb, tmp_path):
    keep, _, _ = odock.prepare_receptor(pdb_3ptb, tmp_path / "keep.pdbqt", keep_water=True)
    drop, _, rep_drop = odock.prepare_receptor(pdb_3ptb, tmp_path / "drop.pdbqt", keep_water=False)
    assert keep.GetNumAtoms() > drop.GetNumAtoms()
    # Water must be gone, but the calcium ion of 3PTB must stay.
    assert not any(
        a.GetPDBResidueInfo() is not None
        and a.GetPDBResidueInfo().GetResidueName().strip() in ("HOH", "WAT")
        for a in drop.GetAtoms()
    )
    assert any(a.GetSymbol() == "Ca" for a in drop.GetAtoms())
    assert rep_drop.n_metal_atoms >= 1


def test_receptor_pdbqt_is_rigid(complex_parts):
    receptor_mol, _ = complex_parts
    _, text, _ = odock.prepare_receptor(receptor_mol)
    assert "ROOT" not in text and "BRANCH" not in text
    assert text.rstrip().endswith("TER")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(argv, capsys):
    from odock.cli import main

    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_info(capsys):
    code, out, err = run_cli(["info"], capsys)
    assert code == 0
    assert "OpenDocking" in out
    assert "vina" in out


# ---------------------------------------------------------------------------
# The workbench is the default entry point
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gui(monkeypatch):
    """Record calls to the workbench launcher instead of opening a window."""
    import odock.gui as gui

    calls = []

    def fake(receptor=None, ligand=None, poses=None):
        calls.append((receptor, ligand, poses))
        return 0

    monkeypatch.setattr(gui, "launch_gui", fake)
    return calls


def test_bare_odock_prints_the_cli_help(fake_gui, capsys):
    """A bare invocation must never surprise a script by opening a window."""
    from odock.cli import main

    assert main([]) == 0
    captured = capsys.readouterr()
    assert "usage: odock" in captured.out
    assert "odock gui" in captured.err
    assert fake_gui == []


def test_odock_gui_with_no_options_opens_the_workbench(fake_gui):
    from odock.cli import main

    assert main(["gui"]) == 0
    assert fake_gui == [(None, None, None)]


def test_odock_on_files_opens_the_workbench(tmp_path, fake_gui, prepared_3ptb):
    from odock.cli import main

    rec = tmp_path / "receptor.pdbqt"
    poses = tmp_path / "poses.pdbqt"
    rec.write_text(prepared_3ptb["receptor_pdbqt"], encoding="utf-8")
    poses.write_text(
        "MODEL 1\n" + prepared_3ptb["ligand_pdbqt"] + "ENDMDL\n"
        "MODEL 2\n" + prepared_3ptb["ligand_pdbqt"] + "ENDMDL\n",
        encoding="utf-8",
    )
    assert main([str(rec), str(poses)]) == 0
    assert fake_gui == [(str(rec), None, str(poses))]


def test_odock_on_a_single_ligand_file(tmp_path, fake_gui, prepared_3ptb):
    from odock.cli import main

    lig = tmp_path / "ligand.pdbqt"
    lig.write_text(prepared_3ptb["ligand_pdbqt"], encoding="utf-8")
    assert main([str(lig)]) == 0
    assert fake_gui == [(None, str(lig), None)]


def test_odock_gui_forwards_its_options(fake_gui):
    from odock.cli import main

    assert main(["gui", "-r", "r.pdbqt", "-l", "l.pdbqt", "-p", "p.pdbqt"]) == 0
    assert fake_gui == [("r.pdbqt", "l.pdbqt", "p.pdbqt")]


def test_subcommands_still_win_over_the_workbench(fake_gui, capsys):
    from odock.cli import main

    assert main(["info"]) == 0
    assert fake_gui == []
    assert "OpenDocking" in capsys.readouterr().out


def test_a_typo_is_still_rejected(fake_gui):
    from odock.cli import main

    with pytest.raises(SystemExit):
        main(["doking"])
    assert fake_gui == []


def test_missing_gui_extra_is_reported_not_raised(monkeypatch, capsys):
    """A bare `odock` without PyQt6 must explain itself, not traceback."""
    import odock.gui as gui
    from odock.cli import main

    def unavailable(*args, **kwargs):
        raise ImportError("no PyQt6")

    monkeypatch.setattr(gui, "launch_gui", unavailable)
    code = main(["gui"])
    captured = capsys.readouterr()
    assert code == 2
    assert "opendocking[gui]" in captured.err
    assert "usage: odock" in captured.err


def test_cli_prepare_and_box_and_score(tmp_path, capsys, complex_parts):
    receptor_mol, ligand_mol = complex_parts
    rec_pdb = tmp_path / "rec.pdb"
    rec_pdb.write_text(Chem.MolToPDBBlock(receptor_mol), encoding="utf-8")
    lig_sdf = tmp_path / "lig.sdf"
    w = Chem.SDWriter(str(lig_sdf))
    w.write(ligand_mol)
    w.close()

    rec_pdbqt = tmp_path / "rec.pdbqt"
    code, _, _ = run_cli(["prepare", "receptor", str(rec_pdb), str(rec_pdbqt)], capsys)
    assert code == 0 and rec_pdbqt.exists()

    lig_pdbqt = tmp_path / "lig.pdbqt"
    code, _, _ = run_cli(
        ["prepare", "ligand", str(lig_sdf), str(lig_pdbqt), "--no-optimize"], capsys
    )
    assert code == 0 and lig_pdbqt.exists()

    box_json = tmp_path / "box.json"
    code, out, _ = run_cli(
        ["box", "--ligand", str(lig_pdbqt), "--buffer", "8", "--out", str(box_json)], capsys
    )
    assert code == 0
    assert box_json.exists()
    payload = json.loads(box_json.read_text())
    assert len(payload["center"]) == 3 and len(payload["size"]) == 3

    code, out, _ = run_cli(
        ["score", "-r", str(rec_pdbqt), "-l", str(lig_pdbqt), "--box", str(box_json)],
        capsys,
    )
    assert code == 0
    assert "affinity" in out


def test_cli_dock_writes_poses(tmp_path, capsys, prepared_3ptb):
    box_json = tmp_path / "box.json"
    box = odock.box_from_ligand(prepared_3ptb["ligand_mol"], buffer=8.0)
    box_json.write_text(json.dumps(box.as_dict()), encoding="utf-8")

    rec = tmp_path / "receptor.pdbqt"
    lig = tmp_path / "ligand.pdbqt"
    rec.write_text(prepared_3ptb["receptor_pdbqt"], encoding="utf-8")
    lig.write_text(prepared_3ptb["ligand_pdbqt"], encoding="utf-8")

    poses = tmp_path / "poses.pdbqt"
    code, out, _ = run_cli(
        [
            "dock",
            "-r", str(rec),
            "-l", str(lig),
            "--box", str(box_json),
            "-o", str(poses),
            "-e", "2",
            "--seed", "7",
        ],
        capsys,
    )
    assert code == 0
    assert "affinity" in out
    assert poses.exists() and "MODEL" in poses.read_text()


def test_cli_split(tmp_path, capsys, prepared_3ptb):
    (tmp_path / "receptor.pdbqt").write_text(prepared_3ptb["receptor_pdbqt"], encoding="utf-8")
    (tmp_path / "ligand.pdbqt").write_text(prepared_3ptb["ligand_pdbqt"], encoding="utf-8")
    multi = tmp_path / "multi.pdbqt"
    multi.write_text(
        "MODEL 1\nROOT\nATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\nENDROOT\nTORSDOF 0\nENDMDL\n"
        "MODEL 2\nROOT\nATOM      1  C1  LIG A   1       1.000   0.000   0.000  1.00  0.00     0.000 C\nENDROOT\nTORSDOF 0\nENDMDL\n",
        encoding="utf-8",
    )
    code, _, err = run_cli(["split", str(multi), "--outdir", str(tmp_path / "split")], capsys)
    assert code == 0
    assert len(list((tmp_path / "split").glob("*.pdbqt"))) == 2

# ---------------------------------------------------------------------------
# Dimensionality guards
# ---------------------------------------------------------------------------


def test_a_two_dimensional_ligand_is_rejected_unless_embedding_is_on():
    """A flat depiction must never be docked silently."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    AllChem.Compute2DCoords(mol)
    flat = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
    assert flat.GetConformer().Is3D() is False

    with pytest.raises(ValueError, match="3-D"):
        odock.prepare_ligand(flat, embed=False)

    ready, _, report = odock.prepare_ligand(flat, name="flat")
    assert report.reembedded, "the 2-D input must have been re-embedded"
    assert any("2-D" in w for w in report.warnings)


def test_a_three_dimensional_crystal_ligand_is_left_alone():
    from rdkit import Chem

    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.AddHs(mol, addCoords=True)
    assert mol.GetConformer().Is3D() is True
    _, _, report = odock.prepare_ligand(mol, name="acetanilide", optimize=False)
    assert not report.reembedded


def test_the_planarity_report_separates_flat_from_three_dimensional():
    flat = odock.prepare_ligand("c1ccccc1", name="benzene")[2]
    voluminous = odock.prepare_ligand(
        "CCOC(=O)N1CCC(CC1)Oc1ccc(NC(=O)C)cc1", name="demo"
    )[2]
    assert flat.thickness < 0.15, "benzene is planar by chemistry"
    assert voluminous.thickness > 0.4, "a drug-like ligand is not planar"


# ---------------------------------------------------------------------------
# Pocket detection, library filtering and the exporters
# ---------------------------------------------------------------------------

DEMO_3PTB = Path(__file__).resolve().parent.parent / "demo" / "3ptb"


@pytest.fixture(scope="session")
def demo_3ptb():
    """The bundled 3PTB demo: a real receptor, ligand, pose file and box."""
    paths = {
        name: DEMO_3PTB / name
        for name in ("receptor.pdbqt", "ligand.pdbqt", "poses.pdbqt", "box.json")
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("the bundled 3PTB demo files are missing")
    return paths


def test_cli_pocket_reports_the_specificity_pocket(demo_3ptb, capsys):
    code, out, _ = run_cli(
        ["pocket", "-r", str(demo_3ptb["receptor.pdbqt"]), "--max", "12"], capsys
    )
    assert code == 0
    assert "volume" in out and "score" in out
    # The S1 specificity pocket of trypsin -- Ser195/Trp215/Gly216, where
    # benzamidine binds -- must be among the cavities a blind search reports.
    assert "TRP215" in out or "GLY216" in out


def test_cli_pocket_writes_json(demo_3ptb, tmp_path, capsys):
    target = tmp_path / "pockets.json"
    code, _, _ = run_cli(
        [
            "pocket",
            "-r", str(demo_3ptb["receptor.pdbqt"]),
            "--min-volume", "50",
            "--json-out", str(target),
        ],
        capsys,
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["pockets"], "3PTB has at least one detectable cavity"
    first = payload["pockets"][0]
    assert set(first) >= {"index", "center", "volume", "score", "residues"}
    assert len(first["center"]) == 3 and first["volume"] >= 50


def test_cli_filter_reports_every_rule(tmp_path, capsys):
    library = tmp_path / "library.smi"
    library.write_text(
        "c1ccccc1C(=O)O benzoic acid\n"
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine\n",
        encoding="utf-8",
    )
    code, out, _ = run_cli(["filter", "-i", str(library)], capsys)
    assert code == 0
    assert "benzoic acid" in out and "caffeine" in out
    for column in ("Lipinski", "Veber", "PAINS", "MW", "LogP", "tPSA"):
        assert column in out
    assert "2 of 2 molecule(s) pass every filter" in out


def test_cli_filter_accepts_several_inputs_and_writes_json(tmp_path, capsys):
    first = tmp_path / "first.smi"
    first.write_text("CCO ethanol\n", encoding="utf-8")
    second = tmp_path / "second.smi"
    second.write_text("CCN ethylamine\n", encoding="utf-8")
    target = tmp_path / "filter.json"
    code, _, _ = run_cli(
        ["filter", "-i", str(first), "-i", str(second), "--json-out", str(target)], capsys
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_molecules"] == 2
    assert [m["name"] for m in payload["molecules"]] == ["ethanol", "ethylamine"]
    assert len(payload["molecules"][0]["filters"]) == 3


def test_cli_filter_reports_an_unreadable_file(tmp_path, capsys):
    with pytest.raises(SystemExit, match="cannot read"):
        run_cli(["filter", "-i", str(tmp_path / "missing.sdf")], capsys)


def test_cli_cluster_groups_the_bundled_poses(demo_3ptb, tmp_path, capsys):
    pytest.importorskip("odock.analysis")
    target = tmp_path / "clusters.json"
    code, out, _ = run_cli(
        [
            "cluster",
            "-p", str(demo_3ptb["poses.pdbqt"]),
            "--cutoff", "2.0",
            "--json-out", str(target),
        ],
        capsys,
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_poses"] == 6
    assert sum(cluster["size"] for cluster in payload["clusters"]) == 6
    assert "cluster" in out and "mean RMSD" in out


def test_cli_cluster_can_compare_with_a_reference_ligand(demo_3ptb, capsys):
    pytest.importorskip("odock.analysis")
    code, out, _ = run_cli(
        [
            "cluster",
            "-p", str(demo_3ptb["poses.pdbqt"]),
            "-l", str(demo_3ptb["ligand.pdbqt"]),
        ],
        capsys,
    )
    assert code == 0
    assert "ref RMSD" in out


def test_cli_interactions_profiles_the_pose(demo_3ptb, tmp_path, capsys):
    pytest.importorskip("odock.analysis")
    target = tmp_path / "interactions.json"
    code, out, _ = run_cli(
        [
            "interactions",
            "-r", str(demo_3ptb["receptor.pdbqt"]),
            "-l", str(demo_3ptb["ligand.pdbqt"]),
            "--json-out", str(target),
        ],
        capsys,
    )
    assert code == 0
    assert "key residues" in out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["key_residues"] is not None
    for interaction in payload["interactions"]:
        assert interaction["kind"] and interaction["distance"] > 0
        assert interaction["receptor_label"] and interaction["ligand_label"]


def test_cli_diagram_writes_an_svg(demo_3ptb, tmp_path, capsys):
    pytest.importorskip("odock.analysis")
    target = tmp_path / "interaction.svg"
    code, _, err = run_cli(
        [
            "diagram",
            "-r", str(demo_3ptb["receptor.pdbqt"]),
            "-l", str(demo_3ptb["ligand.pdbqt"]),
            "-o", str(target),
            "--title", "benzamidine in trypsin",
        ],
        capsys,
    )
    assert code == 0
    assert target.exists()
    assert "<svg" in target.read_text(encoding="utf-8")
    assert "wrote" in err


def test_cli_report_writes_xlsx_and_csv(demo_3ptb, tmp_path, capsys):
    pytest.importorskip("odock.report")
    openpyxl = pytest.importorskip("openpyxl")

    xlsx = tmp_path / "report.xlsx"
    code, _, _ = run_cli(
        [
            "report",
            "-p", str(demo_3ptb["poses.pdbqt"]),
            "-r", str(demo_3ptb["receptor.pdbqt"]),
            "-o", str(xlsx),
        ],
        capsys,
    )
    assert code == 0 and xlsx.exists()
    rows = list(openpyxl.load_workbook(xlsx).active.values)
    assert len(rows) >= 7, "a header row plus one row per mode"
    assert all(len(row) >= 4 for row in rows)

    csv = tmp_path / "report.csv"
    code, _, _ = run_cli(
        ["report", "-p", str(demo_3ptb["poses.pdbqt"]), "-o", str(csv), "--csv"], capsys
    )
    assert code == 0 and csv.exists()
    assert csv.read_text(encoding="utf-8").count("\n") >= 7


def test_cli_dock_passes_the_search_selector_through(tmp_path, capsys, prepared_3ptb):
    rec = tmp_path / "receptor.pdbqt"
    lig = tmp_path / "ligand.pdbqt"
    rec.write_text(prepared_3ptb["receptor_pdbqt"], encoding="utf-8")
    lig.write_text(prepared_3ptb["ligand_pdbqt"], encoding="utf-8")
    box = tmp_path / "box.json"
    box.write_text(
        json.dumps(odock.box_from_ligand(prepared_3ptb["ligand_mol"], buffer=8.0).as_dict()),
        encoding="utf-8",
    )

    code, out, err = run_cli(
        [
            "dock",
            "-r", str(rec),
            "-l", str(lig),
            "--box", str(box),
            "-e", "1",
            "-n", "1",
            "--search", "lga",
        ],
        capsys,
    )
    assert code == 0
    assert "search: lga" in err
    assert "affinity" in out


def test_cli_dock_help_documents_the_search_algorithms(capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["dock", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for algorithm in ("monte_carlo", "lga", "lga_solis"):
        assert algorithm in help_text


# ---------------------------------------------------------------------------
# `odock screen`: the batch form of `odock dock`
# ---------------------------------------------------------------------------


def test_cli_screen_is_listed_and_documents_its_screening_flags(capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    top_level = capsys.readouterr().out
    assert "screen" in top_level
    assert "resumable" in top_level  # the one-line summary in the command list

    with pytest.raises(SystemExit) as excinfo:
        main(["screen", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--receptor", "--input", "--jobs", "--timeout", "--top", "--dry-run",
                 "--no-resume", "--csv", "--checkpoint-every", "--limit",
                 "--allow-box-mismatch", "--no-interactions", "--consensus",
                 "--consensus-top", "--consensus-method"):
        assert flag in help_text, flag
    # argparse wraps the description, so compare with the line breaks collapsed.
    assert "resumes where it stopped" in " ".join(help_text.split())


def test_cli_screen_requires_its_inputs(capsys):
    """A missing --receptor/-i/-o is argparse's job; it must not reach the pipeline."""
    from odock.cli import main

    for argv in (
        ["screen"],
        ["screen", "-r", "rec.pdbqt"],
        ["screen", "-r", "rec.pdbqt", "-i", "lib.smi"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        assert "required" in capsys.readouterr().err


def test_cli_screen_explains_a_missing_box(tmp_path, capsys):
    from odock.cli import main

    library = tmp_path / "library.smi"
    library.write_text("CCO ethanol\n", encoding="utf-8")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no search box"):
        main(
            [
                "screen",
                "-r", str(receptor),
                "-i", str(library),
                "-o", str(tmp_path / "out"),
            ]
        )


def test_cli_screen_reports_a_box_file_without_centre_or_size(tmp_path, capsys):
    from odock.cli import main

    library = tmp_path / "library.smi"
    library.write_text("CCO ethanol\n", encoding="utf-8")
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n", encoding="utf-8")
    box = tmp_path / "box.json"
    box.write_text('{"spacing": 0.375}', encoding="utf-8")

    with pytest.raises(SystemExit, match="not a box file"):
        main(
            [
                "screen",
                "-r", str(receptor),
                "-i", str(library),
                "-o", str(tmp_path / "out"),
                "--box", str(box),
            ]
        )


def test_cli_screen_reports_missing_files_before_docking(tmp_path, capsys):
    """A missing library is a message and exit 2, not a traceback."""
    from odock.cli import main

    code = main(
        [
            "screen",
            "-r", str(tmp_path / "missing-receptor.pdbqt"),
            "-i", str(tmp_path / "missing-library.smi"),
            "-o", str(tmp_path / "out"),
            "--center", "0", "0", "0",
            "--size", "20", "20", "20",
        ]
    )
    assert code == 2
    assert "no such file" in capsys.readouterr().err


def test_every_subcommand_documents_itself(capsys):
    from odock.cli import _subcommands, build_parser, main

    names = sorted(_subcommands(build_parser()))
    assert {
        "pocket", "filter", "cluster", "interactions", "diagram", "report",
        "fetch", "export",
    } <= set(names)
    for name in names:
        with pytest.raises(SystemExit) as excinfo:
            main([name, "--help"])
        assert excinfo.value.code == 0, f"`odock {name} --help` did not succeed"
        assert "usage: odock" in capsys.readouterr().out
    for name in ("gpf", "dpf", "config", "pdb"):
        with pytest.raises(SystemExit) as excinfo:
            main(["export", name, "--help"])
        assert excinfo.value.code == 0
        assert "usage: odock export" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The ligand-chemistry commands: similar, diverse, scaffolds, rgroups
# ---------------------------------------------------------------------------

#: A six-molecule library with three scaffolds: three benzamidines, two
#: pyridines and caffeine.  Small enough that every expected number below can be
#: checked by hand.
CHEM_LIBRARY = (
    "c1ccc(cc1)C(=N)N benzamidine\n"
    "N=C(N)c1ccc(O)cc1 hydroxybenzamidine\n"
    "N=C(N)c1ccc(F)cc1 fluorobenzamidine\n"
    "NC(=O)c1cccnc1 nicotinamide\n"
    "OC(=O)c1ccncc1 isonicotinic_acid\n"
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine\n"
)


@pytest.fixture
def chemistry_library(tmp_path):
    path = tmp_path / "chemistry.smi"
    path.write_text(CHEM_LIBRARY, encoding="utf-8")
    return path


@pytest.fixture
def chemistry_affinities(tmp_path):
    """A results file in the shape `odock screen` writes, with docked values."""
    path = tmp_path / "results.jsonl"
    values = {
        "benzamidine": -5.9,
        "hydroxybenzamidine": -6.42,
        "fluorobenzamidine": -6.23,
        "caffeine": -5.33,
    }
    with path.open("w", encoding="utf-8") as handle:
        for name, affinity in values.items():
            handle.write(
                json.dumps(
                    {
                        "receptor": "receptor",
                        "ligand": name,
                        "name": name,
                        "status": "ok",
                        "affinity": affinity,
                    }
                )
                + "\n"
            )
    return path


def test_cli_similar_ranks_the_analogues(chemistry_library, capsys):
    code, out, _ = run_cli(
        [
            "similar",
            "-q", "c1ccc(cc1)C(=N)N",
            "-i", str(chemistry_library),
            "--cutoff", "0.4",
        ],
        capsys,
    )
    assert code == 0
    # 4/7 = 0.5714 for the 4-hydroxy analogue, 6/11 = 0.5455 for the 4-fluoro one,
    # and nothing else clears 0.4 in this library.
    assert "hydroxybenzamidine" in out and "0.5714" in out
    assert "fluorobenzamidine" in out and "0.5455" in out
    assert "caffeine" not in out
    assert "3 of 6 molecule(s) at tanimoto >= 0.40 (morgan)" in out
    assert "6 comparison(s)" in out


def test_cli_similar_writes_json_and_the_matrix(chemistry_library, tmp_path, capsys):
    target = tmp_path / "similar.json"
    code, out, _ = run_cli(
        [
            "similar",
            "-q", "c1ccc(cc1)C(=N)N",
            "-i", str(chemistry_library),
            "--cutoff", "0.4",
            "--name", "benzamidine",
            "--matrix",
            "--json-out", str(target),
        ],
        capsys,
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["query"] == "benzamidine"
    assert payload["n_hits"] == 3 and payload["n_library"] == 6
    assert payload["hits"][0]["name"] == "benzamidine"
    assert payload["fingerprint"]["kind"] == "morgan"
    assert payload["matrix"]["names"][0] == "benzamidine"
    assert len(payload["matrix"]["values"]) == 6
    assert "off-diagonal" in out and "similarity matrix" in out


def test_cli_similar_3d_reports_what_it_costs(chemistry_library, capsys):
    code, out, _ = run_cli(
        [
            "similar",
            "-q", "c1ccc(cc1)C(=N)N",
            "-i", str(chemistry_library),
            "--3d",
            "--conformers", "2",
            "--cutoff", "0.0",
        ],
        capsys,
    )
    assert code == 0
    assert "pharmacophore" in out
    assert "best over 2 conformer(s) per molecule" in out
    # 2 query conformers x 2 library conformers x 6 molecules = 24 comparisons.
    assert "24 comparison(s)" in out


def test_cli_similar_rejects_a_bad_query(chemistry_library, capsys):
    with pytest.raises(SystemExit, match="neither an existing file nor a parsable SMILES"):
        run_cli(["similar", "-q", "not a molecule", "-i", str(chemistry_library)], capsys)


def test_cli_diverse_picks_a_representative_subset(chemistry_library, tmp_path, capsys):
    """Three MaxMin picks of six molecules: benzamidine, caffeine and
    isonicotinic acid cover all three scaffolds (100 %) for 50 % of the
    molecules, with a tightest internal similarity of 0.222."""
    target = tmp_path / "subset.sdf"
    payload_path = tmp_path / "diverse.json"
    code, out, err = run_cli(
        [
            "diverse",
            "-i", str(chemistry_library),
            "-n", "3",
            "-o", str(target),
            "--json-out", str(payload_path),
        ],
        capsys,
    )
    assert code == 0
    assert "diversity: 3 of 6 molecule(s) (50.0%)" in out
    assert "scaffold space: 3 of 3 scaffold(s) covered (100.0%)" in out
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["indices"] == [0, 5, 4]
    assert payload["names"] == ["benzamidine", "caffeine", "isonicotinic_acid"]
    assert payload["scaffold_coverage"]["n_library_scaffolds"] == 3.0
    assert payload["scaffold_coverage"]["coverage"] == pytest.approx(1.0)
    # The written subset is a real library the rest of the tool can read.
    from odock.chem.ligand import read_ligands

    subset = read_ligands(target, embed=False)
    assert [mol.GetProp("_Name") for mol in subset] == [
        "benzamidine",
        "caffeine",
        "isonicotinic_acid",
    ]
    assert "wrote" in err
    assert "odock_diverse_rank" in target.read_text(encoding="utf-8")


def test_cli_diverse_sphere_needs_a_cutoff(chemistry_library, capsys):
    with pytest.raises(SystemExit, match="needs --cutoff"):
        run_cli(
            ["diverse", "-i", str(chemistry_library), "-n", "3", "--method", "sphere"],
            capsys,
        )


def test_cli_scaffolds_counts_the_library(chemistry_library, chemistry_affinities, capsys):
    code, out, err = run_cli(
        [
            "scaffolds",
            "-i", str(chemistry_library),
            "--affinities", str(chemistry_affinities),
        ],
        capsys,
    )
    assert code == 0
    assert "scaffolds: 3 distinct Murcko scaffold(s) for 6 molecule(s)" in out
    assert "largest series: 3 molecule(s) on c1ccccc1" in out
    assert "best scaffold by affinity" in out
    assert "series: 3 scaffold family(ies) at tanimoto >= 0.65" in out
    assert "affinities: 4 molecule(s) from 4 result record(s)" in err


def test_cli_scaffolds_report_prints_the_chemistry_page(
    chemistry_library, chemistry_affinities, tmp_path, capsys
):
    target = tmp_path / "scaffolds.json"
    code, out, _ = run_cli(
        [
            "scaffolds",
            "-i", str(chemistry_library),
            "--affinities", str(chemistry_affinities),
            "--core", "N=C(N)c1ccccc1",
            "--report",
            "--json-out", str(target),
        ],
        capsys,
    )
    assert code == 0
    for heading in ("SCAFFOLDS", "SERIES", "R-GROUPS", "MATCHED PAIRS"):
        assert heading in out
    assert "LIGAND CHEMISTRY" in out
    assert "docking noise this project measured" in out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_molecules"] == 6
    assert payload["rgroups"]["core"] == "N=C(N)c1ccccc1"
    assert payload["matched_pairs"]["n_pairs"] >= 1
    assert payload["series"]["n_series"] == 3


def test_cli_rgroups_writes_the_matrix_and_the_matched_pairs(
    chemistry_library, chemistry_affinities, tmp_path, capsys
):
    target = tmp_path / "rgroups.csv"
    json_target = tmp_path / "rgroups.json"
    code, out, err = run_cli(
        [
            "rgroups",
            "-i", str(chemistry_library),
            "--core", "N=C(N)c1ccccc1",
            "--affinities", str(chemistry_affinities),
            "--mmp",
            "-o", str(target),
            "--json-out", str(json_target),
        ],
        capsys,
    )
    assert code == 0
    assert "core: N=C(N)c1ccccc1 (given)" in out
    # Only the para position is substituted anywhere in this library, so RDKit
    # finds one attachment point; the three benzamidines match it.
    assert "R-groups: 1 attachment point(s) (R1)" in out
    assert "3 of 6 molecule(s) matched (50%)" in out
    assert "matched pairs: 3 single-point substitution(s)" in out
    # The para H -> OH delta of the measured affinities: -6.42 - (-5.90) = -0.52.
    assert "-0.520" in out
    assert "+1" in out, "H -> OH adds one heavy atom, and the table says so"
    assert "wrote" in err
    rows = list(csv.reader(target.open(encoding="utf-8", newline="")))
    assert rows[0][:3] == ["name", "affinity", "matched"]
    assert len(rows) == 7
    payload = json.loads(json_target.read_text(encoding="utf-8"))
    assert payload["labels"] == rows[0][3:] == ["R1"]
    assert payload["matched_pairs"]["pairs"][0]["name_a"] == "benzamidine"
    assert payload["matched_pairs"]["pairs"][0]["delta"] == pytest.approx(-0.52)
    assert payload["matched_pairs"]["pairs"][0]["added_heavy_atoms"] == 1


def test_cli_rgroups_xlsx_has_a_series_sheet(
    chemistry_library, chemistry_affinities, tmp_path, capsys
):
    openpyxl = pytest.importorskip("openpyxl")
    target = tmp_path / "rgroups.xlsx"
    code, _, _ = run_cli(
        [
            "rgroups",
            "-i", str(chemistry_library),
            "--core", "N=C(N)c1ccccc1",
            "--affinities", str(chemistry_affinities),
            "-o", str(target),
        ],
        capsys,
    )
    assert code == 0
    workbook = openpyxl.load_workbook(target)
    assert "R-groups" in workbook.sheetnames
    assert "series" in workbook.sheetnames


def test_cli_rgroups_reports_an_unparsable_core(chemistry_library, capsys):
    with pytest.raises(SystemExit):
        run_cli(["rgroups", "-i", str(chemistry_library), "--core", "not a core"], capsys)


def test_cli_rgroups_reports_a_missing_results_file(chemistry_library, tmp_path, capsys):
    with pytest.raises(SystemExit, match="no such results file"):
        run_cli(
            [
                "rgroups",
                "-i", str(chemistry_library),
                "--affinities", str(tmp_path / "nope.jsonl"),
            ],
            capsys,
        )


def test_the_new_commands_are_advertised(capsys):
    from odock.cli import _subcommands, build_parser, main

    names = _subcommands(build_parser())
    assert {"similar", "diverse", "scaffolds", "rgroups"} <= set(names)
    with pytest.raises(SystemExit) as excinfo:
        main(["screen", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--diverse", "--diverse-method", "--diverse-cutoff", "--diverse-start"):
        assert flag in help_text, flag


def test_screen_diverse_replaces_the_library_with_the_subset(demo_3ptb, tmp_path, capsys):
    """`--diverse N` selects before the campaign starts, writes the subset in the
    output directory, and does not rewrite it on a second run — otherwise its
    mtime would change the library hash and break the resume."""
    from odock.cli import main

    outdir = tmp_path / "campaign"
    argv = [
        "screen",
        "-r", str(demo_3ptb["receptor.pdbqt"]),
        "-i", str(Path(__file__).resolve().parent.parent / "demo" / "library.smi"),
        "--box", str(demo_3ptb["box.json"]),
        "-o", str(outdir),
        "--diverse", "4",
        "--dry-run",
    ]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "--diverse:" in captured.err
    assert "scaffold space" in captured.err
    subset = outdir / "library_diverse.sdf"
    assert subset.exists()
    from odock.chem.ligand import read_ligands

    molecules = read_ligands(subset, embed=False)
    assert len(molecules) == 4
    payload = json.loads((outdir / "diverse.json").read_text(encoding="utf-8"))
    assert payload["n_selected"] == 4
    assert payload["n_library"] == 15, "the two molecules the filters remove are not eligible"
    assert payload["scaffold_coverage"]["n_library_scaffolds"] >= 1
    first_mtime = subset.stat().st_mtime_ns
    assert main(argv) == 0
    capsys.readouterr()
    assert subset.stat().st_mtime_ns == first_mtime, "an unchanged subset is left alone"


# ---------------------------------------------------------------------------
# `odock pharmacophore build|screen|show`
# ---------------------------------------------------------------------------

#: Five members of the benzamidine series, the evidence a model is built from.
PHARMACOPHORE_MEMBERS = (
    "c1ccc(cc1)C(=N)N benzamidine\n"
    "N=C(N)c1ccc(O)cc1 hydroxybenzamidine\n"
    "N=C(N)c1ccc(F)cc1 fluorobenzamidine\n"
    "N=C(N)c1ccccc1Cl chloro_benzamidine\n"
    "c1ccc(cc1)C(=N)NC benzamidine_methyl\n"
)
#: A library to screen: the five actives plus three molecules that should not fit.
PHARMACOPHORE_LIBRARY = PHARMACOPHORE_MEMBERS + (
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine\n"
    "CC(=O)Oc1ccccc1C(=O)O aspirin\n"
    "NC(=O)c1cccnc1 nicotinamide\n"
)


@pytest.fixture
def pharmacophore_inputs(tmp_path):
    members = tmp_path / "members.smi"
    members.write_text(PHARMACOPHORE_MEMBERS, encoding="utf-8")
    library = tmp_path / "library.smi"
    library.write_text(PHARMACOPHORE_LIBRARY, encoding="utf-8")
    actives = tmp_path / "actives.smi"
    actives.write_text(PHARMACOPHORE_MEMBERS, encoding="utf-8")
    return {"members": members, "library": library, "actives": actives}


def test_cli_pharmacophore_build_screen_and_show(pharmacophore_inputs, tmp_path, capsys):
    model_path = tmp_path / "model.json"
    code, out, err = run_cli(
        [
            "pharmacophore", "build",
            "-i", str(pharmacophore_inputs["members"]),
            "--core", "N=C(N)c1ccccc1",
            "-o", str(model_path),
        ],
        capsys,
    )
    assert code == 0
    assert "5 feature(s) from 5 of 5 member(s)" in out or "4 feature(s) from 5 of 5" in out
    assert "donor" in out and "aromatic" in out and "envelope" in out
    assert "meaningful" in out
    assert "wrote" in err
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["n_used"] == 5 and payload["meaningful"] is True
    assert payload["core"] == "N=C(N)c1ccccc1"

    scores = tmp_path / "scores.csv"
    screen_json = tmp_path / "screen.json"
    code, out, _ = run_cli(
        [
            "pharmacophore", "screen",
            "-m", str(model_path),
            "-i", str(pharmacophore_inputs["library"]),
            "--conformers", "2",
            "--actives", str(pharmacophore_inputs["actives"]),
            "--scores-out", str(scores),
            "--json-out", str(screen_json),
        ],
        capsys,
    )
    assert code == 0
    assert "screen: 8 molecule(s) scored" in out
    assert "enrichment:" in out
    assert "too small a labelled set" in out
    rows = list(csv.reader(scores.open(encoding="utf-8", newline="")))
    assert rows[0][:4] == ["rank", "name", "fit", "matched"]
    assert len(rows) == 9
    report = json.loads(screen_json.read_text(encoding="utf-8"))
    assert report["n_library"] == 8
    assert report["enrichment"]["n_actives"] == 5
    assert report["enrichment"]["enrichment_factor"] >= 1.0
    # The five members are the top five: they are the model's own evidence.
    assert sorted(hit["name"] for hit in report["hits"][:5]) == sorted(
        ["benzamidine", "benzamidine_methyl", "hydroxybenzamidine",
         "fluorobenzamidine", "chloro_benzamidine"]
    )
    assert report["hits"][0]["fit"] > report["hits"][-1]["fit"]

    code, out, _ = run_cli(
        ["pharmacophore", "show", "-m", str(model_path)], capsys
    )
    assert code == 0
    assert "pharmacophore model: 4 feature(s) from 5 of 5 member(s)" in out
    assert "meaningful: 5 of 5 member(s) contributed" in out


def test_cli_pharmacophore_flags_an_anecdote(tmp_path, capsys):
    """Two members are not evidence, and the command says so twice: in the table
    and on stderr."""
    members = tmp_path / "two.smi"
    members.write_text(
        "c1ccc(cc1)C(=N)N benzamidine\nN=C(N)c1ccc(O)cc1 hydroxybenzamidine\n",
        encoding="utf-8",
    )
    code, out, err = run_cli(
        ["pharmacophore", "build", "-i", str(members), "--core", "N=C(N)c1ccccc1"],
        capsys,
    )
    assert code == 0
    assert "ANECDOTE" in out
    assert "fewer than the 3" in out
    assert "not evidence" in err


def test_cli_pharmacophore_screen_reports_a_missing_model(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_cli(
            [
                "pharmacophore", "screen",
                "-m", str(tmp_path / "missing.json"),
                "-i", str(tmp_path / "library.smi"),
            ],
            capsys,
        )


def test_cli_pharmacophore_rejects_a_bad_core(tmp_path, capsys):
    members = tmp_path / "m.smi"
    members.write_text("c1ccc(cc1)C(=N)N benzamidine\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_cli(["pharmacophore", "build", "-i", str(members), "--core", "zzz"], capsys)


# ---------------------------------------------------------------------------
# `odock decoys` and `odock lbvs`: the ligand-based benchmark
# ---------------------------------------------------------------------------


def test_cli_decoys_selects_and_reports_the_matching(tmp_path, capsys):
    actives = tmp_path / "actives.smi"
    actives.write_text(
        "c1ccc(cc1)C(=N)N benzamidine\n"
        "N=C(N)c1ccc(O)cc1 hydroxybenzamidine\n"
        "N=C(N)c1ccc(F)cc1 fluorobenzamidine\n",
        encoding="utf-8",
    )
    pool = tmp_path / "pool.smi"
    pool.write_text(
        "Nc1ccccc1O 2_aminophenol\n"
        "NCc1ccccc1 benzylamine\n"
        "NCCc1ccccc1 phenethylamine\n"
        "Nc1ncccn1 2_aminopyrimidine\n"
        "c1ccc2ccccc2c1 naphthalene\n"
        "OC(=O)CCC(=O)O succinic_acid\n",
        encoding="utf-8",
    )
    target = tmp_path / "decoys.smi"
    payload = tmp_path / "decoys.json"
    code, out, err = run_cli(
        [
            "decoys",
            "-a", str(actives),
            "-p", str(pool),
            "-n", "1",
            "-o", str(target),
            "--json-out", str(payload),
        ],
        capsys,
    )
    assert code == 0
    assert "decoys:" in out and "selected from" in out
    assert "SMD" in out and "max |SMD|" in out
    assert "wrote" in err
    selected = json.loads(payload.read_text(encoding="utf-8"))
    assert selected["n_decoys"] >= 1
    assert selected["quality"]["properties"]["MW"]["actives_mean"] > 0
    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == selected["n_decoys"]
    assert all(line.split()[1] for line in lines), "every written decoy keeps its name"


def test_cli_lbvs_reports_the_metrics_with_their_controls(tmp_path, capsys):
    actives = tmp_path / "actives.smi"
    actives.write_text(
        "c1ccc(cc1)C(=N)N benzamidine\n"
        "N=C(N)c1ccc(O)cc1 hydroxybenzamidine\n"
        "N=C(N)c1ccc(F)cc1 fluorobenzamidine\n",
        encoding="utf-8",
    )
    decoys = tmp_path / "decoys.smi"
    decoys.write_text(
        "Nc1ccccc1O 2_aminophenol\n"
        "NCc1ccccc1 benzylamine\n"
        "Nc1ncccn1 2_aminopyrimidine\n"
        "NCc1ccccn1 2_aminomethylpyridine\n",
        encoding="utf-8",
    )
    payload = tmp_path / "lbvs.json"
    code, out, _ = run_cli(
        [
            "lbvs",
            "-a", str(actives),
            "-d", str(decoys),
            "--methods", "fingerprint",
            "--conformers", "1",
            "--bootstrap", "10",
            "--prefilter", "0.5",
            "--json-out", str(payload),
        ],
        capsys,
    )
    assert code == 0
    assert "EF1%" in out and "BEDROC" in out and "95% CI" in out
    assert "random" in out and "property_MW" in out
    assert "pre-filter at 50%" in out and "saved" in out
    report = json.loads(payload.read_text(encoding="utf-8"))
    assert report["n_actives"] == 3 and report["n_decoys"] == 4
    assert {result["method"] for result in report["results"]} >= {
        "fingerprint_morgan", "random", "property_MW"
    }
    assert report["prefilter"]["n_library"] == 7
    for result in report["results"]:
        assert 0.0 <= result["stats"]["auc"] <= 1.0
        assert len(result["intervals"]["auc"]) == 2


def test_cli_decoys_and_lbvs_are_advertised(capsys):
    from odock.cli import _subcommands, build_parser

    names = _subcommands(build_parser())
    assert {"decoys", "lbvs"} <= set(names)


# ---------------------------------------------------------------------------
# `odock ensemble`: docking against several receptor conformations
# ---------------------------------------------------------------------------


def ensemble_pair(data_dir):
    """``(3ERT, 1ERE chain A)``: the ERα antagonist and agonist structures.

    The same protein with a real binding-site difference (helix 12 moves), which
    is what an ensemble is for; both files ship in ``tests/data``.
    """
    antagonist = data_dir / "3ERT.pdb"
    agonist = data_dir / "1ERE_A.pdb"
    if not antagonist.exists() or not agonist.exists():
        pytest.skip("missing the bundled ERα structures")
    return antagonist, agonist


def test_cli_ensemble_is_listed_and_documents_itself(capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "ensemble" in capsys.readouterr().out

    with pytest.raises(SystemExit) as excinfo:
        main(["ensemble", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for kind in ("align", "dock", "screen"):
        assert kind in help_text

    for kind, flags in (
        ("align", ("--site", "--site-ligand", "--site-radius", "--min-identity",
                   "--max-site-residues", "--json-out")),
        ("dock", ("--box-ligand", "--cluster-rmsd", "--no-cross", "--consensus",
                  "--site-atoms", "--reference")),
        ("screen", ("--box-ligand", "--robustness-top", "--cluster-rmsd", "--no-cross",
                    "--top", "--jobs", "--no-resume", "--consensus")),
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(["ensemble", kind, "--help"])
        assert excinfo.value.code == 0
        text = capsys.readouterr().out
        for flag in flags:
            assert flag in text, f"`odock ensemble {kind} --help` does not document {flag}"


def test_cli_ensemble_align_reports_the_alignment(data_dir, tmp_path, capsys):
    """The refusal/acceptance numbers the docs quote come from this command."""
    from odock.cli import main

    antagonist, agonist = ensemble_pair(data_dir)
    target = tmp_path / "alignment.json"
    code = main(
        [
            "ensemble", "align",
            "-r", str(antagonist), str(agonist),
            "--box-ligand", "OHT",
            "--json-out", str(target),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "OpenDocking ensemble — 2 conformation(s)" in out
    # Sequence identity, site RMSD and the residue that moves most, all measured.
    assert "0.944" in out and "4.551" in out
    assert "LEU525 A" in out and "ASP351 A" in out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["site"] and len(payload["site"]) == 14
    assert payload["alignments"][1]["identity"] == pytest.approx(0.944, abs=0.005)
    assert payload["alignments"][1]["site_rmsd"] == pytest.approx(0.444, abs=0.05)
    assert payload["alignments"][1]["displacement"]["LEU525 A"] == pytest.approx(2.010, abs=0.05)
    assert payload["alignments"][0]["reference_frame"] is True


def test_cli_ensemble_align_writes_the_aligned_receptors(data_dir, tmp_path, capsys):
    from odock.cli import main

    antagonist, agonist = ensemble_pair(data_dir)
    outdir = tmp_path / "aligned"
    code = main(
        [
            "ensemble", "align",
            "-r", str(antagonist), str(agonist),
            "--box-ligand", "OHT",
            "--outdir", str(outdir),
            "--pdbqt",
            "-q",
        ]
    )
    assert code == 0
    capsys.readouterr()
    assert (outdir / "3ERT.pdb").exists() and (outdir / "1ERE_A.pdb").exists()
    for stem in ("3ERT", "1ERE_A"):
        text = (outdir / f"{stem}.pdbqt").read_text(encoding="utf-8")
        assert "ATOM" in text
        # The co-crystallised ligand is gone: it would sit in the site being docked into.
        assert "OHT" not in text and "EST" not in text


def test_cli_ensemble_align_refuses_a_different_protein(data_dir, capsys):
    from odock.cli import main

    antagonist, _agonist = ensemble_pair(data_dir)
    other = data_dir / "3PTB.pdb"
    if not other.exists():
        pytest.skip("missing 3PTB")
    code = main(
        [
            "ensemble", "align",
            "-r", str(antagonist), str(other),
            "--site", "MET343,LEU345,ASP351",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "not the same receptor" in err
    # The refusal carries the numbers, not just a verdict.
    assert "identity" in err and "aligned columns" in err


@pytest.mark.slow
def test_cli_ensemble_dock_merges_the_conformations(data_dir, tmp_path, capsys):
    """Two trypsins that differ by almost nothing: same answer, twice."""
    from odock.cli import main

    first = data_dir / "3PTB.pdb"
    second = data_dir / "2PTN.pdb"
    ligand = data_dir / "BTN.sdf"
    if not second.exists():
        pytest.skip("missing 2PTN")
    poses = tmp_path / "ensemble.pdbqt"
    report = tmp_path / "ensemble.json"
    code = main(
        [
            "ensemble", "dock",
            "-r", str(first), str(second),
            "--box-ligand", "BEN", "--buffer", "8",
            "-l", str(ligand),
            "-e", "1", "-n", "1", "--seed", "42",
            "-o", str(poses),
            "--json-out", str(report),
            "-q",
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["labels"] == ["3PTB", "2PTN"]
    assert payload["winning_conformation"] in {"3PTB", "2PTN"}
    assert set(payload["best_per_conformation"]) == {"3PTB", "2PTN"}
    # The two structures are nearly identical, so the affinities agree closely.
    values = list(payload["best_per_conformation"].values())
    assert max(values) - min(values) < 0.5
    assert payload["robustness"]["score"] > 0.0
    assert payload["rescoring"]["n_scores"] == 2 * 2
    # The merged pose file is a real multi-model PDBQT the toolkit can read back.
    text = poses.read_text(encoding="utf-8")
    assert text.count("MODEL") == len(payload["poses"])
    assert "ODOCK ENSEMBLE: conformation=" in text
    from odock.cli import _read_poses

    read_back = _read_poses(poses)
    assert len(read_back) == len(payload["poses"])
    assert all(pose["affinity"] is not None for pose in read_back)


def test_cli_ensemble_dock_explains_a_missing_box(data_dir, tmp_path, capsys):
    from odock.cli import main

    antagonist, agonist = ensemble_pair(data_dir)
    with pytest.raises(SystemExit, match="no search box"):
        main(
            [
                "ensemble", "dock",
                "-r", str(antagonist), str(agonist),
                "-l", str(data_dir / "EST.sdf"),
            ]
        )


def test_cli_ensemble_screen_requires_its_inputs(capsys):
    """`ensemble screen` is argparse-complete: no options, no crash, exit 2."""
    from odock.cli import main

    for argv in (
        ["ensemble", "screen"],
        ["ensemble", "screen", "-r", "rec.pdb"],
        ["ensemble", "screen", "-r", "rec.pdb", "-i", "lib.smi"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
        assert "required" in capsys.readouterr().err


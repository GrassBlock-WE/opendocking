# SPDX-License-Identifier: GPL-3.0-or-later
"""Preparation, boxes and the command-line interface."""

from __future__ import annotations

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

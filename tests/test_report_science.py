# SPDX-License-Identifier: GPL-3.0-or-later
"""The upgraded report: efficiency/consensus columns, sheets, notes and streams.

The synthetic :class:`DockResult` fixtures here reuse the ligand/receptor PDBQT
documents the existing report tests pin, so the residue column and the historical
column positions stay under the same assertions while the new columns are tested
on top of them.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from odock import analysis, consensus, metrics, report
from odock.docking import DockResult, Pose

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "3ptb"
DEMO_RECEPTOR = DEMO / "receptor.pdbqt"
DEMO_LIGAND = DEMO / "ligand.pdbqt"
DEMO_POSES = DEMO / "poses.pdbqt"
DEMO_BOX = DEMO / "box.json"

LIGAND_PDBQT = """\
ROOT
ATOM      1  N1  LIG A   1       0.000   0.000   0.000  1.00  0.00    -0.300 N
ATOM      2  H1  LIG A   1       1.010   0.000   0.000  1.00  0.00     0.200 HD
ATOM      3  C1  LIG A   1      -1.200   0.500   0.000  1.00  0.00     0.100 C
ATOM      4  C2  LIG A   1      -2.500   0.900   0.000  1.00  0.00     0.000 C
ENDROOT
TORSDOF 0
"""

RECEPTOR_PDBQT = """\
ATOM      1  OD1 ASP A 189       2.800   0.000   0.000  1.00  0.00    -0.500 OA
ATOM      2  CG  ASP A 189       2.800   1.450   0.000  1.00  0.00     0.400 C
ATOM      3  CB  ALA A  55       0.000   0.000   4.000  1.00  0.00     0.000 C
TER
"""


@pytest.fixture()
def dock_result() -> DockResult:
    """Two poses of a four-heavy-atom ligand, in the shape the CLI builds."""
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.01, 0.0, 0.0], [-1.2, 0.5, 0.0], [-2.5, 0.9, 0.0]]
    )
    # The second pose is displaced, so the two really do score differently: a
    # rank correlation between two identical poses is undefined, not 1.
    shifted = coords + np.array([[0.9, 0.7, 0.4]])
    poses = [
        Pose(index=0, affinity=-9.42, coords=coords),
        Pose(index=1, affinity=-8.76, rmsd_lower_bound=1.342, rmsd_upper_bound=1.89,
             coords=shifted),
    ]
    return DockResult(
        poses=poses,
        seed=3,
        ligand_pdbqt=LIGAND_PDBQT,
        receptor_pdbqt=RECEPTOR_PDBQT,
        ligand_atom_order=("N1", "H1", "C1", "C2"),
        num_tors=2.0,
        scoring="vina",
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def test_the_historical_columns_keep_their_positions(dock_result):
    rows = report.result_rows(dock_result)
    assert list(report.COLUMNS)[:5] == ["mode", "affinity", "rmsd_lb", "rmsd_ub", "residues"]
    for row in rows:
        assert set(row) == set(report.COLUMNS)
    assert rows[0]["mode"] == 1
    assert rows[0]["residues"] == "ASP189"


def test_efficiency_columns_are_filled_without_rdkit(dock_result):
    rows = report.result_rows(dock_result)
    first = rows[0]
    assert first["heavy_atoms"] == 3  # N1, C1, C2
    assert first["ligand_efficiency"] == pytest.approx(9.42 / 3)
    assert first["entropy_penalty"] == pytest.approx(metrics.entropy_penalty(2.0))
    # No descriptors were supplied, so these are blank rather than guessed.
    assert first["logp"] is None
    assert first["lle"] is None
    assert first["bei"] is None
    assert first["sei"] is None
    assert first["strain"] is None
    assert first["consensus_rank"] is None


def test_descriptors_columns_come_from_the_supplied_molecule(dock_result):
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene descriptors, deliberately
    rows = report.result_rows(dock_result, ligand_mol=mol)
    first = rows[0]
    assert first["logp"] == pytest.approx(1.6866, abs=0.002)
    assert first["lle"] == pytest.approx(9.42 / 1.37 - 1.6866, abs=1e-3)
    assert first["bei"] == pytest.approx(1000 * (9.42 / 1.37) / 78.114, rel=1e-3)
    # Benzene has no polar surface: SEI stays blank instead of infinite.
    assert first["sei"] is None


def test_explicit_descriptors_beat_the_molecule(dock_result):
    rows = report.result_rows(
        dock_result, descriptors={"logp": 2.0, "molecular_weight": 200.0, "tpsa": 50.0,
                                  "heavy_atoms": 10, "num_torsions": 1}
    )
    first = rows[0]
    assert first["heavy_atoms"] == 10
    assert first["ligand_efficiency"] == pytest.approx(0.942)
    assert first["logp"] == pytest.approx(2.0)
    assert first["sei"] == pytest.approx(100 * (9.42 / 1.37) / 50.0, rel=1e-9)


def test_consensus_columns_are_opt_in_and_appear_when_asked_for(dock_result):
    plain = report.result_rows(dock_result)
    assert all(row["consensus_score"] is None for row in plain)

    combined = consensus.consensus_score(
        dock_result,
        RECEPTOR_PDBQT,
        consensus.BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=0.5),
        scorings=("vina", "vinardo"),
    )
    rows = report.result_rows(dock_result, consensus=combined)
    for row in rows:
        assert row["consensus_score"] is not None
    # The best pose under both fields must be rank 1.
    ranks = {row["mode"]: row["consensus_rank"] for row in rows}
    assert sorted(ranks.values()) == [1, 2]
    assert "consensus_vina" in rows[0]
    assert "consensus_vinardo_rank" in rows[0]
    assert rows[0]["consensus_method"] == "rank"


def test_consensus_true_computes_it_from_the_result_itself(dock_result):
    combined = consensus.consensus_score(
        dock_result,
        RECEPTOR_PDBQT,
        consensus.BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=0.5),
        scorings=("vina",),
    )
    via_true = report.result_rows(
        dock_result, consensus=True, scorings=("vina",), box=combined_box()
    )
    via_object = report.result_rows(dock_result, consensus=combined)
    assert [row["consensus_rank"] for row in via_true] == [
        row["consensus_rank"] for row in via_object
    ]
    assert [round(row["consensus_score"], 12) for row in via_true] == [
        round(row["consensus_score"], 12) for row in via_object
    ]


def combined_box():
    return consensus.BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=0.5)


def test_strain_column_is_measured_when_the_result_has_a_receptor(dock_result):
    rows = report.result_rows(dock_result, strain=True, strain_options={"steps": 200})
    assert all(row["strain"] is not None for row in rows)


def test_strain_column_is_blank_and_warns_without_any_receptor():
    result = DockResult(
        poses=[Pose(index=0, affinity=-1.0, coords=np.zeros((2, 3)))],
        seed=0,
        ligand_pdbqt=LIGAND_PDBQT,
        receptor_pdbqt="",
    )
    with pytest.warns(UserWarning):
        rows = report.result_rows(result, strain=True)
    assert rows[0]["strain"] is None


# ---------------------------------------------------------------------------
# Column headers
# ---------------------------------------------------------------------------


def test_headers_are_human_readable_for_derived_columns():
    assert report._header("mode") == "Mode"
    assert report._header("residues") == "Key interacting residues"
    assert report._header("consensus_rank") == "Consensus rank"
    assert report._header("consensus_score") == "Consensus score"
    assert report._header("consensus_vina") == "Vina (kcal/mol)"
    assert report._header("consensus_ad4_rank") == "Ad4 rank"
    assert report._header("consensus_method") == "Consensus method"
    assert report._header("weird_key") == "Weird key"


def test_columns_for_keeps_the_base_order_and_appends_extras():
    rows = [{"mode": 1, "affinity": -1.0, "extra": 2}, {"mode": 2, "affinity": -2.0}]
    columns = report.columns_for(rows, base=("mode", "affinity"))
    assert columns == ["mode", "affinity", "extra"]
    assert report.columns_for([], base=("mode", "affinity")) == ["mode", "affinity"]


# ---------------------------------------------------------------------------
# Flat files
# ---------------------------------------------------------------------------


def test_csv_has_one_header_and_one_row_per_pose(tmp_path, dock_result):
    rows = report.result_rows(dock_result)
    path = report.write_csv(tmp_path / "r.csv", rows=rows)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(report._header(key) for key in report.columns_for(rows))
    assert len(lines) == 3
    parsed = list(csv.DictReader(lines))
    assert parsed[0]["Mode"] == "1"
    assert float(parsed[0]["Binding Energy (kcal/mol)"]) == pytest.approx(-9.42)
    assert parsed[0]["Heavy atoms"] == "3"


def test_csv_blanks_a_missing_metric_rather_than_writing_nan(tmp_path, dock_result):
    path = report.write_csv(tmp_path / "r.csv", rows=report.result_rows(dock_result))
    text = Path(path).read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    assert ",," in text  # the empty LogP/LLE/... cells


def test_jsonl_normalises_non_finite_numbers_to_null(tmp_path):
    rows = [
        {"mode": 1, "affinity": float("nan"), "le": 0.5},
        {"mode": 2, "affinity": -6.0, "le": float("inf")},
    ]
    path = report.write_jsonl(tmp_path / "r.jsonl", rows)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["affinity"] is None and first["le"] == 0.5
    second = json.loads(lines[1])
    assert second["affinity"] == -6.0 and second["le"] is None


def test_write_rows_picks_the_format_from_the_suffix(tmp_path):
    rows = [{"mode": 1, "affinity": -1.0}]
    csv_path = report.write_rows(tmp_path / "a.csv", rows)
    assert Path(csv_path).read_text(encoding="utf-8").startswith("Mode,")
    jsonl_path = report.write_rows(tmp_path / "a.jsonl", rows)
    assert json.loads(Path(jsonl_path).read_text(encoding="utf-8").splitlines()[0])
    tsv_path = report.write_rows(tmp_path / "a.tsv", rows)
    assert "\t" in Path(tsv_path).read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        report.write_rows(tmp_path / "a.bin", rows, fmt="parquet")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_a_jsonl_stream_writes_as_it_goes(tmp_path):
    path = tmp_path / "big.jsonl"
    stream = report.ResultStream(path, flush_every=2)
    assert isinstance(stream, report.ResultStream)
    assert stream.open() is stream
    assert stream.n_written == 0
    for index in range(5):
        stream.append({"mode": index + 1, "affinity": -float(index), "le": float("nan")})
    assert stream.n_written == 5
    # flush_every=2 means rows 2 and 4 are on disk; the rest are in the buffer
    # until close().  A crash here still leaves a complete prefix.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4
    stream.close()
    stream.close()  # idempotent
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["mode"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["le"] is None


def test_a_csv_stream_fixes_its_columns_on_the_first_row(tmp_path):
    path = tmp_path / "big.csv"
    with report.ResultStream(path, flush_every=1) as stream:
        stream.append({"mode": 1, "affinity": -1.0})
        with pytest.warns(UserWarning):
            stream.append({"mode": 2, "affinity": -2.0, "surprise": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Mode,Binding Energy (kcal/mol)"
    assert len(lines) == 3
    assert lines[2].startswith("2,-2.0")


def test_a_stream_can_be_given_its_columns_up_front(tmp_path):
    path = tmp_path / "big.csv"
    with report.ResultStream(path, columns=("mode", "affinity", "ligand_efficiency")) as stream:
        assert stream.extend([{"mode": 1, "affinity": -1.0, "ligand_efficiency": 0.25}]) == 1
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("Mode,Binding Energy (kcal/mol),LE")
    assert lines[1].startswith("1,-1.0,0.25")


def test_a_stream_rejects_an_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        report.ResultStream(tmp_path / "x.bin", fmt="parquet")


# ---------------------------------------------------------------------------
# Spreadsheets
# ---------------------------------------------------------------------------


def test_xlsx_has_the_historical_two_sheets_by_default(tmp_path, dock_result):
    openpyxl = pytest.importorskip("openpyxl")
    path = report.write_xlsx(tmp_path / "r.xlsx", dock_result)
    workbook = openpyxl.load_workbook(path)
    assert workbook.sheetnames == ["Poses", "Run"]


def test_xlsx_adds_fingerprint_and_pharmacophore_sheets(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    receptor = [
        analysis.Interaction("hbond", 0, 0, 2.8, residue=("ASP", 189, "A")),
    ]
    schema = analysis.FingerprintSchema(
        keys=(
            analysis.FingerprintKey("ASP", 189, "A", "hbond"),
            analysis.FingerprintKey("ALA", 55, "A", "hydrophobic"),
        )
    )
    fingerprints = analysis.FingerprintSet(
        schema=schema,
        fingerprints=[
            analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0, 0.0])),
            analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0, 1.0])),
        ],
        interactions=[[receptor[0]], []],
        water_bridges=[[], []],
    )
    path = report.write_xlsx(
        tmp_path / "r.xlsx",
        rows=[{"mode": 1, "affinity": -1.0}, {"mode": 2, "affinity": -2.0}],
        fingerprints=fingerprints,
    )
    workbook = openpyxl.load_workbook(path)
    assert workbook.sheetnames == ["Poses", "Run", "Fingerprints", "Pharmacophore"]
    sheet = workbook["Fingerprints"]
    assert [cell.value for cell in sheet[1]][:3] == ["Mode", "Present", "ASP189:hbond"]
    assert sheet.cell(row=2, column=2).value == 1
    assert sheet.cell(row=3, column=2).value == 2
    assert sheet.cell(row=3, column=4).value == 1
    pharm = workbook["Pharmacophore"]
    assert [cell.value for cell in pharm[1]] == [
        "Feature", "Poses", "Fraction", "Kind", "Residue"
    ]
    assert pharm.cell(row=2, column=1).value == "ASP189:hbond"
    assert pharm.cell(row=2, column=3).value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Notebook summary and energy decomposition
# ---------------------------------------------------------------------------


def test_notebook_summary_is_plain_text_with_the_headline_numbers(dock_result):
    text = report.notebook_summary(dock_result)
    assert "OpenDocking run summary" in text
    assert "2 pose(s)" in text
    assert "seed 3" in text
    assert "mode 1" in text
    assert "-9.420 kcal/mol" in text
    assert "LE 3.140" in text  # 9.42 / 3 heavy atoms
    assert "ASP189" in text
    assert "no ligand molecule was supplied" in text


def test_notebook_summary_reports_the_consensus_and_its_agreement(dock_result):
    combined = consensus.consensus_score(
        dock_result, RECEPTOR_PDBQT, combined_box(), scorings=("vina", "vinardo")
    )
    text = report.notebook_summary(dock_result, consensus=combined)
    assert "consensus    : vina, vinardo (rank aggregation)" in text
    assert "rank 1 = mode" in text
    assert "Spearman rho" in text
    assert "strain" not in text.split("note")[0].split("consensus")[1]


def test_notebook_summary_warns_when_the_force_fields_disagree(dock_result):
    raw = {
        "vina": _components([-1.0, -2.0]),
        "vinardo": _components([-2.0, -1.0]),
    }
    combined = consensus.consensus_score(
        None, None, components=raw, scorings=("vina", "vinardo")
    )
    text = report.notebook_summary(dock_result, consensus=combined)
    assert "only partly agree" in text
    assert "-1.000" in text


def _components(values):
    return {
        "affinity": list(values),
        "inter": list(values),
        "intra": [0.0] * len(values),
        "unbound": [0.0] * len(values),
        "conf_independent": [0.0] * len(values),
        "total": list(values),
    }


def test_notebook_summary_says_the_entropy_estimate_is_a_constant(dock_result):
    """A column that never changes within a run must be explained, not hidden."""
    text = report.notebook_summary(dock_result)
    assert "entropy est." in text
    assert "the same for every pose of this ligand" in text
    assert "not a pose discriminator" in text
    # 2 rotors on the fixture -> 2 x 0.6509
    assert "1.302" in text


def test_notebook_summary_reports_the_ligand_descriptors_when_given(dock_result):
    text = report.notebook_summary(
        dock_result, ligand_mol=Chem.MolFromSmiles("c1ccccc1")
    )
    assert "LLE" in text
    assert "no ligand molecule was supplied" not in text
    assert "LogP/LLE/BEI/SEI are blank" not in text


def test_notebook_summary_lists_the_pharmacophore():
    schema = analysis.FingerprintSchema(
        keys=(analysis.FingerprintKey("ASP", 189, "A", "hbond"),)
    )
    fingerprints = analysis.FingerprintSet(
        schema=schema,
        fingerprints=[
            analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0])),
            analysis.InteractionFingerprint(schema=schema, counts=np.array([1.0])),
        ],
        water_bridges=[[], []],
    )
    result = DockResult(poses=[Pose(index=0, affinity=-1.0)], seed=0)
    text = report.notebook_summary(result, fingerprints=fingerprints)
    assert "pharmacophore: recurring contacts among the top 2 pose(s)" in text
    assert "ASP189:hbond" in text
    assert "2/2 poses (100%)" in text


def test_decomposition_table_reproduces_the_torsion_identity(dock_result):
    table = report.decomposition_table_for(
        dock_result, scorings=("vina",), box=combined_box(), receptor=RECEPTOR_PDBQT
    )
    assert "affinity" in table and "torsion" in table
    rows = consensus.decomposition_rows(
        consensus.rescore_poses(
            dock_result, RECEPTOR_PDBQT, combined_box(), ("vina",)
        )
    )
    assert len(rows) == 2
    for row in rows:
        assert row.base == pytest.approx(row.inter + row.intra - row.unbound)
        assert row.torsion_term == pytest.approx(row.affinity - row.base)


# ---------------------------------------------------------------------------
# Reproducibility and the strain-corrected ranking
# ---------------------------------------------------------------------------


def _run(seed, shift, energies):
    base = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    poses = [
        Pose(index=index, affinity=energy, coords=base + np.asarray(shift) + 0.02 * index)
        for index, energy in enumerate(energies)
    ]
    return DockResult(poses=poses, seed=seed)


def test_reproducibility_counts_the_runs_that_found_the_winning_mode():
    runs = [
        _run(1, (0.0, 0.0, 0.0), [-8.0, -7.0]),
        _run(2, (0.1, 0.0, 0.0), [-7.5, -6.5]),
        _run(3, (6.0, 0.0, 0.0), [-7.8, -6.0]),
    ]
    report_card = consensus.reproducibility(runs, cutoff=2.0)
    assert report_card.n_runs == 3
    assert report_card.n_poses == 6
    assert report_card.n_clusters == 2
    assert report_card.cluster_sizes[0] == 4
    assert report_card.reproducibility == pytest.approx(2 / 3)
    assert report_card.top_cluster_runs == [0, 1]
    assert report_card.best_of_run_cluster == [0, 0, 1]
    assert "2/3 runs found the top mode" in report_card.table()
    assert report_card.as_dict()["reproducibility"] == pytest.approx(2 / 3)


def test_reproducibility_of_a_single_run_is_undefined():
    card = consensus.reproducibility([_run(1, (0, 0, 0), [-8.0])])
    assert math.isnan(card.reproducibility)
    assert card.n_runs == 1
    assert any("fewer than two" in note for note in card.notes)


def test_reproducibility_reports_a_run_without_coordinates():
    empty = DockResult(poses=[Pose(index=0, affinity=-1.0)], seed=1)
    card = consensus.reproducibility([_run(1, (0, 0, 0), [-8.0]), empty])
    assert any("no pose with coordinates" in note for note in card.notes)
    assert math.isnan(card.reproducibility)


def test_strain_corrected_ranking_reorders_and_keeps_both_ranks():
    rows = consensus.strain_corrected_ranking(
        [-8.0, -7.9, -7.8], [2.0, 0.1, 0.2], indices=[10, 11, 12]
    )
    # The strained best pose (index 10, -8.0 + 2.0 = -6.0) drops to last.
    assert [row.index for row in rows] == [11, 12, 10]
    assert [row.rank for row in rows] == [2, 3, 1]
    assert [row.corrected_rank for row in rows] == [1, 2, 3]
    assert rows[0].corrected == pytest.approx(-7.8)
    assert rows[-1].corrected == pytest.approx(-6.0)


def test_strain_corrected_ranking_keeps_an_unmeasured_pose_uncorrected():
    rows = consensus.strain_corrected_ranking([-8.0, -7.0], [float("nan"), 0.5])
    by_index = {row.index: row for row in rows}
    assert by_index[0].corrected == pytest.approx(-8.0)
    assert by_index[1].corrected == pytest.approx(-6.5)
    with pytest.raises(ValueError):
        consensus.strain_corrected_ranking([-8.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Against the bundled demo
# ---------------------------------------------------------------------------


has_demo = pytest.mark.skipif(
    not (DEMO_RECEPTOR.exists() and DEMO_POSES.exists() and DEMO_BOX.exists()),
    reason="the bundled 3PTB demo is not present",
)


@has_demo
def test_the_demo_poses_report_with_consensus_columns(tmp_path):
    models = consensus.pdbqt_models(DEMO_POSES.read_text(encoding="utf-8"))
    combined = consensus.consensus_score(models, DEMO_RECEPTOR, DEMO_BOX)
    rows = []
    for position, model in enumerate(models):
        rows.append(
            {
                "mode": position + 1,
                "affinity": combined.poses[position].scores["vina"]
                if combined.poses[position].index == position
                else None,
            }
        )
    # The consensus object is what the report needs; drive it through a real
    # DockResult-free path to make sure the columns land.
    result = DockResult(
        poses=[
            Pose(
                index=index,
                affinity=float(p.scores["vina"]),
                coords=np.array(
                    [[a.x, a.y, a.z] for a in consensus.pdbqt_atoms(models[index])]
                ),
            )
            for index, p in enumerate(combined.poses)
        ],
        seed=42,
        ligand_pdbqt=DEMO_LIGAND.read_text(encoding="utf-8"),
        receptor_pdbqt=DEMO_RECEPTOR.read_text(encoding="utf-8"),
    )
    table = report.result_rows(result, consensus=consensus.consensus_score(
        result, DEMO_RECEPTOR, DEMO_BOX), box=DEMO_BOX)
    assert len(table) == 6
    assert sorted(row["consensus_rank"] for row in table) == [1, 2, 3, 4, 5, 6]
    assert all(row["consensus_score"] is not None for row in table)
    assert all(row["heavy_atoms"] == 9 for row in table)
    assert table[0]["ligand_efficiency"] == pytest.approx(
        abs(table[0]["affinity"]) / 9, rel=1e-12
    )
    csv_path = report.write_csv(tmp_path / "demo.csv", rows=table)
    header = Path(csv_path).read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("Mode,Binding Energy")
    assert "Consensus rank" in header
    del rows

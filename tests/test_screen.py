# SPDX-License-Identifier: GPL-3.0-or-later
"""The screening pipeline: ``odock screen`` and :mod:`odock.screen`.

The end-to-end tests run against the bundled 3PTB demo, which is deliberately
small: two molecules at ``--exhaustiveness 1`` take a couple of seconds, so the
whole file stays inside the normal test budget.  The pipeline itself is exercised
on the important paths — dry run, a real campaign, resume, a broken molecule, a
timeout, a box that does not match the receptor, an empty library, CSV output,
a receptor panel and determinism.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

import pytest

import odock

pytest.importorskip("rdkit")

from odock.screen import (  # noqa: E402
    EXIT_ALL_FAILED,
    EXIT_INPUT,
    EXIT_OK,
    ScreenConfig,
    ScreenError,
    check_box_against_receptor,
    estimate_cost,
    grid_points,
    ligand_seed,
    read_records,
    receptor_atoms_in_box,
    receptor_bounds,
    screen_ligands,
)

ROOT = Path(__file__).resolve().parent.parent
DEMO_3PTB = ROOT / "demo" / "3ptb"
DEMO_LIBRARY = ROOT / "demo" / "library.smi"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def demo_3ptb():
    """The bundled 3PTB demo: receptor, ligand, poses and box."""
    paths = {
        name: DEMO_3PTB / name
        for name in ("receptor.pdbqt", "ligand.pdbqt", "poses.pdbqt", "box.json")
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("the bundled 3PTB demo files are missing")
    return paths


@pytest.fixture(scope="session")
def demo_box(demo_3ptb):
    """The 3PTB active-site box as a :class:`odock.BoxSpec`."""
    return odock.BoxSpec(**json.loads(demo_3ptb["box.json"].read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def screen_library(tmp_path_factory):
    """Four molecules: two drug-like, one PAINS, one over Lipinski's LogP.

    The bundled library is realistic but has 15 dockable molecules, which is more
    docking than a test needs; this one keeps the funnel just as visible with two
    molecules to dock.
    """
    path = tmp_path_factory.mktemp("library") / "small.smi"
    path.write_text(
        "c1ccc(cc1)C(=N)N benzamidine\n"
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine\n"
        "O=C1C=CC(=O)C=C1 benzoquinone\n"
        "c1ccc2c(c1)c1ccccc1c1ccccc21 triphenylene\n",
        encoding="utf-8",
    )
    return path


def run_cli(argv, capsys):
    from odock.cli import main

    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def screen_argv(demo_3ptb, library, outdir, *extra):
    """The canonical command line of a small campaign.

    ``-n 3`` keeps the pose refinement (the second-most expensive part after the
    search) small; the pipeline's behaviour does not depend on it.
    """
    return [
        "screen",
        "-r", str(demo_3ptb["receptor.pdbqt"]),
        "-i", str(library),
        "--box", str(demo_3ptb["box.json"]),
        "-o", str(outdir),
        "-e", "1",
        "-n", "3",
        "--seed", "42",
        *extra,
    ]


@pytest.fixture(scope="session")
def campaign(demo_3ptb, screen_library, tmp_path_factory):
    """One complete campaign, shared by the tests that only read its output.

    Session-scoped on purpose: a docking run is the expensive part of this file,
    and the interesting assertions are all about what the run *wrote*.
    """
    outdir = tmp_path_factory.mktemp("screen-campaign")
    summary = screen_ligands(
        ScreenConfig(
            receptors=[demo_3ptb["receptor.pdbqt"]],
            inputs=[screen_library],
            box=odock.BoxSpec(**json.loads(demo_3ptb["box.json"].read_text(encoding="utf-8"))),
            outdir=outdir,
            exhaustiveness=1,
            num_poses=3,
            seed=42,
            top=2,
            jobs=2,
            progress=False,
        )
    )
    return summary, outdir


# ---------------------------------------------------------------------------
# The pure parts: no docking, no files
# ---------------------------------------------------------------------------


def test_grid_points_reproduces_the_kernel_figure(demo_box):
    """The estimate must count grid samples exactly as `odock dock` does."""
    assert grid_points(demo_box) == 150_920  # what the 3PTB demo run reports
    egfr = odock.BoxSpec(**json.loads((ROOT / "demo" / "egfr" / "box.json").read_text()))
    assert grid_points(egfr) == 95_550


def test_ligand_seed_is_stable_and_per_molecule():
    first = ligand_seed(42, "trypsin", "library.smi#1")
    assert first == ligand_seed(42, "trypsin", "library.smi#1")
    assert first != ligand_seed(42, "trypsin", "library.smi#2")
    assert first != ligand_seed(42, "mutant", "library.smi#1")
    assert first != ligand_seed(43, "trypsin", "library.smi#1")
    assert 0 < first < 2**31


def test_estimate_cost_grows_with_exhaustiveness_and_flexibility():
    cheap = estimate_cost(n_ligands=10, n_atoms=10, n_torsions=0, exhaustiveness=1)
    stiff = estimate_cost(n_ligands=10, n_atoms=20, n_torsions=4, exhaustiveness=1)
    hard = estimate_cost(n_ligands=10, n_atoms=20, n_torsions=4, exhaustiveness=32)
    assert 0 < cheap < stiff < hard
    # More cores never means more time, and the model is linear in the library.
    assert estimate_cost(n_ligands=20, n_torsions=4) == pytest.approx(
        2 * estimate_cost(n_ligands=10, n_torsions=4)
    )
    assert estimate_cost(n_ligands=10, cores=32) < estimate_cost(n_ligands=10, cores=1)


def test_the_box_must_hold_a_receptor_atom(demo_3ptb, demo_box):
    text = demo_3ptb["receptor.pdbqt"].read_text(encoding="utf-8")
    assert receptor_atoms_in_box(text, demo_box) > 100
    lo, hi = receptor_bounds(text)
    assert all(lo[axis] < hi[axis] for axis in range(3))
    assert check_box_against_receptor(demo_box, text) is None

    far = odock.BoxSpec(center=(500.0, 500.0, 500.0), size=(20.0, 20.0, 20.0))
    complaint = check_box_against_receptor(far, text)
    assert complaint and "no receptor atom" in complaint
    assert receptor_atoms_in_box(text, far) == 0


def test_config_hashes_separate_library_from_docking(tmp_path, demo_3ptb, screen_library, demo_box):
    base = dict(
        receptors=[demo_3ptb["receptor.pdbqt"]],
        inputs=[screen_library],
        box=demo_box,
        outdir=tmp_path,
    )
    a = ScreenConfig(**base)
    # The campaign seed *is* part of the docking identity: every molecule's seed
    # is derived from it, so two seeds are two campaigns and must not share a
    # results file.  It does not touch the prepared library, though.
    assert ScreenConfig(**{**base, "seed": 7}).docking_hash() != a.docking_hash()
    assert ScreenConfig(**{**base, "seed": 7}).library_hash() == a.library_hash()
    # The box, the force field and the receptor do change it.
    other_box = odock.BoxSpec(center=demo_box.center, size=(30.0, 30.0, 30.0))
    assert ScreenConfig(**{**base, "box": other_box}).docking_hash() != a.docking_hash()
    assert ScreenConfig(**{**base, "scoring": "vinardo"}).docking_hash() != a.docking_hash()
    assert ScreenConfig(**{**base, "filters": False}).library_hash() != a.library_hash()
    assert ScreenConfig(**{**base, "optimize": False}).library_hash() != a.library_hash()


def test_config_validation_messages(tmp_path, demo_3ptb, screen_library, demo_box):
    base = dict(
        receptors=[demo_3ptb["receptor.pdbqt"]],
        inputs=[screen_library],
        box=demo_box,
        outdir=tmp_path,
    )
    with pytest.raises(ScreenError, match="no receptor"):
        ScreenConfig(**{**base, "receptors": []}).validated()
    with pytest.raises(ScreenError, match="no library"):
        ScreenConfig(**{**base, "inputs": []}).validated()
    with pytest.raises(ScreenError, match="no such file"):
        ScreenConfig(**{**base, "receptors": [tmp_path / "missing.pdbqt"]}).validated()
    with pytest.raises(ScreenError, match="exhaustiveness"):
        ScreenConfig(**{**base, "exhaustiveness": 0}).validated()
    with pytest.raises(ScreenError, match="format"):
        ScreenConfig(**{**base, "fmt": "xml"}).validated()


def test_read_records_tolerates_a_truncated_last_line(tmp_path):
    good = {"receptor": "rec", "ligand": "lib#1", "name": "one", "status": "ok", "affinity": -6.5}
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps(good) + "\n" + '{"receptor": "rec", "ligand": "lib#2"', encoding="utf-8")
    records = read_records(path)
    assert [r.name for r in records] == ["one"]
    assert records[0].affinity == pytest.approx(-6.5)
    assert read_records(tmp_path / "nothing.jsonl") == []


def test_ligand_efficiency_prefers_the_metrics_module():
    from odock.screen import _ligand_efficiency

    value, source = _ligand_efficiency(-7.0, 14)
    assert value == pytest.approx(0.5)
    try:
        import odock.metrics  # noqa: F401

        assert source == "odock.metrics"
    except ImportError:
        assert source == "builtin"
    assert _ligand_efficiency(None, 14) == (None, "")
    assert _ligand_efficiency(-7.0, 0) == (None, "")


# ---------------------------------------------------------------------------
# The campaign itself
# ---------------------------------------------------------------------------


def test_campaign_docks_every_survivor_and_ranks_them(campaign):
    summary, outdir = campaign
    assert summary.exit_code == EXIT_OK
    assert len(summary.members) == 2
    assert summary.n_ok() == 2
    assert summary.n_failed() == 0 and summary.n_timeout() == 0
    assert summary.library_source == "read"
    assert summary.completed_this_run == 2

    # The funnel: 4 read, 2 removed, one by each filter.
    assert summary.stats.n_read == 4
    assert summary.stats.n_kept == 2
    assert summary.stats.per_filter == {"Lipinski": 1, "PAINS": 1}
    assert summary.stats.exclusive == {"Lipinski": 1, "PAINS": 1}

    ranked = summary.ranked(summary.receptor_names[0])
    assert [r.affinity for r in ranked] == sorted(r.affinity for r in ranked)
    assert all(r.affinity is not None for r in ranked)
    assert all(r.in_box for r in ranked)
    assert all(r.ligand_efficiency and r.ligand_efficiency > 0 for r in ranked)
    assert all(r.le_source in ("builtin", "odock.metrics") for r in ranked)
    # The poses in the 3PTB site must contact the S1 residues.
    assert any(r.key_residues for r in ranked)
    assert (outdir / "results.jsonl").exists()


def test_campaign_writes_every_output(campaign):
    summary, outdir = campaign
    for name in ("results.jsonl", "summary.json", "summary.csv", "library.csv", "run.json"):
        assert (outdir / name).is_file(), name

    # library.csv: every molecule that was read, docked or not.
    with (outdir / "library.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["name"] for row in rows] == [
        "benzamidine", "caffeine", "benzoquinone", "triphenylene",
    ]
    removed = {row["name"]: row for row in rows if row["passed"] == "0"}
    assert set(removed) == {"benzoquinone", "triphenylene"}
    assert removed["benzoquinone"]["first_failed"] == "PAINS"
    assert removed["triphenylene"]["first_failed"] == "Lipinski"
    assert all(row["pdbqt_file"] for row in rows if row["passed"] == "1")
    assert not any(row["pdbqt_file"] for row in rows if row["passed"] == "0")

    # One pose file per successful molecule, referenced relatively.
    for record in summary.records:
        assert record.pose_file and (outdir / record.pose_file).is_file()
        assert "\\" not in record.pose_file

    # run.json records the campaign identity and how long it really took.
    manifest = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["docked"] == 2
    assert manifest["library"]["n_read"] == 4
    assert manifest["seconds_per_molecule"] > 0

    # summary.csv has exactly one header row and no duplicated column.
    first = (outdir / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    columns = first.split(",")
    assert len(columns) == len(set(columns))
    assert columns[:3] == ["receptor", "ligand", "name"]


def test_campaign_shortlist_is_a_multi_model_pdbqt(campaign):
    summary, outdir = campaign
    shortlist = summary.top_paths[summary.receptor_names[0]]
    assert shortlist.is_file() and shortlist.name.startswith("top_")
    text = shortlist.read_text(encoding="utf-8")
    assert text.count("MODEL") == 2
    assert "MODEL     1" in text and "MODEL     2" in text
    # Each model is labelled with the library molecule it came from.
    for record in summary.ranked(summary.receptor_names[0]):
        assert f"ligand {record.name}" in text
    # It is a real PDBQT: the splitter and the pose reader both accept it.
    from odock.cli import _read_poses

    poses = _read_poses(shortlist)
    assert len(poses) == 2
    assert all(pose["affinity"] is not None for pose in poses)


def test_campaign_report_mentions_the_funnel_and_the_ranking(campaign):
    summary, _outdir = campaign
    text = summary.text()
    assert "4 read -> 2 kept" in text
    assert "benzamidine" in text and "caffeine" in text
    assert "docked" in text
    assert "summary:" in text


def test_the_cli_runs_a_campaign_and_reports_it(
    demo_3ptb, screen_library, tmp_path, capsys
):
    outdir = tmp_path / "cli-campaign"
    code, out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--top", "1"), capsys
    )
    assert code == EXIT_OK
    assert "OpenDocking screen" in out
    assert "4 read -> 2 kept" in out
    assert "1 of 1 docking(s) succeeded" in out
    assert "best:" in out
    assert (outdir / "results.jsonl").is_file()
    # The progress line is on stderr and carries done/total.
    assert "1/1" in err


def test_the_bundled_library_is_the_documented_example(demo_3ptb, tmp_path, capsys):
    """`odock screen -r ... -i demo/library.smi --box ... -o ...` must work.

    docs/SCREENING.md quotes the funnel figures of this exact command, so they are
    pinned here: 17 molecules read, 15 kept, one removed by each of Lipinski and
    PAINS.
    """
    if not DEMO_LIBRARY.exists():
        pytest.skip("demo/library.smi is missing")
    code, out, _err = run_cli(
        screen_argv(demo_3ptb, DEMO_LIBRARY, tmp_path / "bundled", "--dry-run"), capsys
    )
    assert code == EXIT_OK
    assert "17 read -> 15 kept" in out
    assert "Lipinski" in out and "PAINS" in out
    assert "triphenylene" in out and "benzoquinone" in out
    assert "estimated cost" in out
    assert not (tmp_path / "bundled" / "results.jsonl").exists()
    assert (tmp_path / "bundled" / "library.csv").is_file()


def test_dry_run_docks_nothing_but_prepares_everything(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "dry"
    code, out, _err = run_cli(screen_argv(demo_3ptb, screen_library, outdir, "--dry-run"), capsys)
    assert code == EXIT_OK
    assert "dry run" in out
    assert "nothing was docked" in out
    assert not (outdir / "results.jsonl").exists()
    assert not (outdir / "poses").exists()
    # The prepared library is cached, so the real run afterwards is instant.
    assert (outdir / "library.csv").is_file()
    assert len(list((outdir / "library").glob("*.pdbqt"))) == 2
    assert json.loads((outdir / "run.json").read_text(encoding="utf-8"))["dry_run"] is True
    # A dry run at other settings is allowed to replace it (it docked nothing).
    code, _out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--dry-run", "-e", "2"), capsys
    )
    assert code == EXIT_OK


# ---------------------------------------------------------------------------
# Interruption, resume and never losing work
# ---------------------------------------------------------------------------


def test_resume_docks_only_what_is_missing(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "resume"
    code, _out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys
    )
    assert code == EXIT_OK
    first = read_records(outdir / "results.jsonl")
    assert len(first) == 1

    # The same command again: nothing to do, and nothing is rewritten.
    code, out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys
    )
    assert code == EXIT_OK
    assert "already done" in out
    assert read_records(outdir / "results.jsonl") == first

    # With a bigger limit, only the new molecule is docked.
    code, out, _err = run_cli(screen_argv(demo_3ptb, screen_library, outdir), capsys)
    assert code == EXIT_OK
    assert "1 to dock of 2 (1 already done)" in out
    after = read_records(outdir / "results.jsonl")
    assert len(after) == 2
    # The earlier record is byte-for-byte the one that was already on disk.
    assert after[0].ligand == first[0].ligand
    assert after[0].affinity == first[0].affinity
    assert after[0].seed == first[0].seed


def test_a_changed_campaign_refuses_to_mix_with_an_existing_one(
    demo_3ptb, screen_library, tmp_path, capsys
):
    outdir = tmp_path / "changed"
    assert run_cli(screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys)[0] == 0
    before = (outdir / "results.jsonl").read_text(encoding="utf-8")

    code, _out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "-e", "2"), capsys
    )
    assert code == EXIT_INPUT
    assert "docking settings changed" in err
    assert "--no-resume" in err
    # Nothing was touched.
    assert (outdir / "results.jsonl").read_text(encoding="utf-8") == before

    # A different library is refused as well.
    other = tmp_path / "other.smi"
    other.write_text("CCO ethanol\n", encoding="utf-8")
    code, _out, err = run_cli(screen_argv(demo_3ptb, other, outdir, "--limit", "1"), capsys)
    assert code == EXIT_INPUT
    assert "library inputs changed" in err

    # And --no-resume is the documented way out: it discards and starts over.
    code, _out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "-e", "2", "--no-resume"),
        capsys,
    )
    assert code == EXIT_OK
    assert "--no-resume removed" in err
    assert len(read_records(outdir / "results.jsonl")) == 1


def test_an_interrupted_run_keeps_the_completed_molecules(
    demo_3ptb, screen_library, tmp_path, capsys, monkeypatch
):
    """Simulate Ctrl-C after the first molecule: its record must be on disk."""
    from odock import screen as screen_module

    original = screen_module._make_record
    calls = {"n": 0}

    def failing(config, member, receptor_name, outcome, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return original(config, member, receptor_name, outcome, **kwargs)

    monkeypatch.setattr(screen_module, "_make_record", failing)
    outdir = tmp_path / "interrupted"
    code, _out, err = run_cli(screen_argv(demo_3ptb, screen_library, outdir, "--jobs", "1"), capsys)
    assert code == 130
    records = read_records(outdir / "results.jsonl")
    assert len(records) == 1
    assert records[0].status == "ok"
    assert "interrupted" in err and "continue" in err

    # The next run picks up where it stopped: only the missing molecule is docked.
    monkeypatch.undo()
    code, out, _err = run_cli(screen_argv(demo_3ptb, screen_library, outdir), capsys)
    assert code == EXIT_OK
    assert "1 to dock of 2 (1 already done)" in out
    assert len(read_records(outdir / "results.jsonl")) == 2


# ---------------------------------------------------------------------------
# Robustness: one bad molecule, a timeout, a bad box, an empty library
# ---------------------------------------------------------------------------


def test_one_bad_molecule_does_not_stop_the_run(demo_3ptb, tmp_path, capsys):
    ligand = (ROOT / "demo" / "3ptb" / "ligand.pdbqt").read_text(encoding="utf-8")
    broken = ligand.replace("TORSDOF 1", "REMARK deliberately broken")
    library = tmp_path / "mixed.pdbqt"
    library.write_text(
        "MODEL 1\n" + ligand + "ENDMDL\n"
        "MODEL 2\n" + broken + "ENDMDL\n"
        "MODEL 3\n" + ligand + "ENDMDL\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "mixed"
    code, out, _err = run_cli(screen_argv(demo_3ptb, library, outdir), capsys)
    assert code == EXIT_OK
    records = read_records(outdir / "results.jsonl")
    assert len(records) == 3
    statuses = sorted(record.status for record in records)
    assert statuses == ["failed", "ok", "ok"]
    failed = next(record for record in records if record.status == "failed")
    assert "TORSDOF" in failed.error
    assert failed.affinity is None
    assert "1 failed" in out and "2 of 3 docked" in out


def test_a_timeout_is_recorded_and_counted(demo_3ptb, screen_library, tmp_path, capsys):
    """A molecule that outruns its limit becomes one row, not a hung run.

    ``-e 256`` makes the search long enough that the cancel token is certainly
    observed while it is still running, whatever the machine's speed.
    """
    outdir = tmp_path / "timeout"
    code, out, err = run_cli(
        screen_argv(
            demo_3ptb, screen_library, outdir, "--limit", "1", "--timeout", "0.1",
            "-e", "256", "--quiet",
        ),
        capsys,
    )
    assert code == EXIT_ALL_FAILED
    records = read_records(outdir / "results.jsonl")
    assert len(records) == 1
    assert records[0].status == "timeout"
    assert "timeout" in records[0].error
    assert "no molecule docked successfully" in err
    # A quiet run still reports its counts on stderr.
    assert "0 of 1 docking(s) succeeded" in err


def test_a_box_that_misses_the_receptor_is_refused(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "mismatch"
    argv = screen_argv(demo_3ptb, screen_library, outdir, "--center", "500", "500", "500",
                       "--size", "20", "20", "20")
    argv = [a for a in argv if a not in ("--box", str(demo_3ptb["box.json"]))]
    code, _out, err = run_cli(argv, capsys)
    assert code == EXIT_INPUT
    assert "no receptor atom" in err
    assert "coordinate frame" in err
    assert not (outdir / "results.jsonl").exists()

    # --allow-box-mismatch is the documented escape hatch, and warns loudly.
    code, out, err = run_cli(argv + ["--dry-run", "--allow-box-mismatch"], capsys)
    assert code == EXIT_OK
    assert "warning" in err and "allow-box-mismatch" in err


def test_a_receptor_that_does_not_match_the_box_is_refused(demo_3ptb, screen_library, tmp_path, capsys):
    """The classic mistake: a box from another structure (here, EGFR's)."""
    egfr = ROOT / "demo" / "egfr"
    if not (egfr / "receptor.pdbqt").exists():
        pytest.skip("the bundled EGFR demo is missing")
    argv = [
        "screen",
        "-r", str(egfr / "receptor.pdbqt"),
        "-i", str(screen_library),
        "--box", str(demo_3ptb["box.json"]),
        "-o", str(tmp_path / "frame"),
        "-e", "1",
    ]
    code, _out, err = run_cli(argv, capsys)
    assert code == EXIT_INPUT
    assert "no receptor atom" in err


def test_an_empty_library_is_reported(tmp_path, demo_3ptb, capsys):
    empty = tmp_path / "empty.smi"
    empty.write_text("# nothing but a comment\n\n", encoding="utf-8")
    code, _out, err = run_cli(
        screen_argv(demo_3ptb, empty, tmp_path / "empty-out"), capsys
    )
    assert code == EXIT_INPUT
    assert "cannot read" in err or "no molecule" in err


def test_a_library_the_filters_empty_is_reported(tmp_path, demo_3ptb, capsys):
    library = tmp_path / "pains.smi"
    library.write_text("O=C1C=CC(=O)C=C1 benzoquinone\n", encoding="utf-8")
    code, _out, err = run_cli(screen_argv(demo_3ptb, library, tmp_path / "filtered"), capsys)
    assert code == EXIT_INPUT
    assert "removed all 1 molecule(s)" in err
    assert "--no-filter" in err
    # ... and --no-filter docks it anyway.
    code, _out, _err = run_cli(
        screen_argv(demo_3ptb, library, tmp_path / "filtered", "--no-filter", "--no-interactions"),
        capsys,
    )
    assert code == EXIT_OK


# ---------------------------------------------------------------------------
# Formats, panels and determinism
# ---------------------------------------------------------------------------


def test_csv_output_and_resume(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "csv"
    code, _out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--csv", "--limit", "1", "--top", "1"),
        capsys,
    )
    assert code == EXIT_OK
    assert (outdir / "results.csv").is_file()
    assert not (outdir / "results.jsonl").exists()
    with (outdir / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["name"] == "benzamidine"
    assert float(rows[0]["affinity"]) < 0
    assert rows[0]["status"] == "ok"

    # A CSV results file is resumable too.
    code, out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--csv", "--limit", "1"), capsys
    )
    assert code == EXIT_OK
    assert "already in results.csv" in out
    assert len(read_records(outdir / "results.csv")) == 1


def test_a_receptor_panel_produces_one_result_set_each(
    demo_3ptb, screen_library, tmp_path, capsys
):
    panel = tmp_path / "panel"
    for name in ("trypsin", "mutant"):
        (panel / name).mkdir(parents=True)
        (panel / name / "receptor.pdbqt").write_text(
            demo_3ptb["receptor.pdbqt"].read_text(encoding="utf-8"), encoding="utf-8"
        )
    outdir = tmp_path / "panel-out"
    code, out, _err = run_cli(
        [
            "screen",
            "-r", str(panel / "trypsin" / "receptor.pdbqt"),
            "-r", str(panel / "mutant" / "receptor.pdbqt"),
            "-i", str(screen_library),
            "--box", str(demo_3ptb["box.json"]),
            "-o", str(outdir),
            "-e", "1",
            "--seed", "42",
            "--limit", "1",
            "--top", "1",
            "--no-interactions",
        ],
        capsys,
    )
    assert code == EXIT_OK
    records = read_records(outdir / "results.jsonl")
    assert len(records) == 2
    assert {record.receptor for record in records} == {"trypsin_receptor", "mutant_receptor"}
    assert (outdir / "top_trypsin_receptor.pdbqt").is_file()
    assert (outdir / "top_mutant_receptor.pdbqt").is_file()
    assert "2 receptor(s)" in out
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["docked"] == 2


def test_the_same_seed_reproduces_the_campaign(demo_3ptb, screen_library, tmp_path, capsys):
    first, second = tmp_path / "seed-a", tmp_path / "seed-b"
    for outdir in (first, second):
        code, _out, _err = run_cli(
            screen_argv(
                demo_3ptb, screen_library, outdir, "--limit", "1", "--no-interactions", "--no-poses"
            ),
            capsys,
        )
        assert code == EXIT_OK
    a, b = read_records(first / "results.jsonl")[0], read_records(second / "results.jsonl")[0]
    assert a.seed == b.seed
    assert a.affinity == pytest.approx(b.affinity, abs=1e-9)
    assert a.name == b.name


def test_a_lost_manifest_recovers_the_campaign_seed(demo_3ptb, screen_library, tmp_path, capsys):
    """Without a manifest the rows themselves say which seed the campaign used."""
    outdir = tmp_path / "recover"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--seed", "7"), capsys
    )[0] == EXIT_OK
    first = read_records(outdir / "results.jsonl")[0]
    assert first.seed == ligand_seed(7, "receptor", first.ligand)
    (outdir / "run.json").unlink()

    # Resume with no --seed at all: guessing would re-dock the shared molecule
    # under a new seed, so the seed is recovered from the row instead.
    argv = screen_argv(demo_3ptb, screen_library, outdir)
    argv.remove("--seed")
    argv.remove("42")
    code, out, _err = run_cli(argv, capsys)
    assert code == EXIT_OK
    assert "campaign seed recovered from the results file: 7" in out
    rows = {record.ligand: record for record in read_records(outdir / "results.jsonl")}
    assert len(rows) == 2
    for record in rows.values():
        assert record.seed == ligand_seed(7, "receptor", record.ligand)
    assert rows[first.ligand].affinity == first.affinity


def test_a_lost_manifest_and_a_new_seed_is_refused(demo_3ptb, screen_library, tmp_path, capsys):
    """HOLE 2 without a manifest either: the rows prove the seed changed."""
    outdir = tmp_path / "recover-refused"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--seed", "7"), capsys
    )[0] == EXIT_OK
    before = (outdir / "results.jsonl").read_text(encoding="utf-8")
    (outdir / "run.json").unlink()

    code, _out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "2", "--seed", "9"), capsys
    )
    assert code == EXIT_INPUT
    assert "different campaign seed" in err
    assert "mixes two campaigns" in err
    assert (outdir / "results.jsonl").read_text(encoding="utf-8") == before


def test_no_resume_starts_over_even_without_a_manifest(demo_3ptb, screen_library, tmp_path, capsys):
    """--no-resume means discard, whether or not run.json survived."""
    outdir = tmp_path / "no-resume"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys
    )[0] == EXIT_OK
    (outdir / "run.json").unlink()          # a killed campaign left rows only

    code, _out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--no-resume"), capsys
    )
    assert code == EXIT_OK
    assert "--no-resume removed" in err
    assert len(read_records(outdir / "results.jsonl")) == 1   # replaced, not appended
    assert (outdir / "run.json").is_file()


def test_duplicate_rows_in_a_file_are_counted_once(demo_3ptb, screen_library, tmp_path, capsys):
    """A file left by the re-docking bug must not double-count its molecules."""
    outdir = tmp_path / "duplicates"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys
    )[0] == EXIT_OK
    results = outdir / "results.jsonl"
    first = results.read_text(encoding="utf-8")
    results.write_text(first + first, encoding="utf-8")   # the same row twice

    code, out, err = run_cli(screen_argv(demo_3ptb, screen_library, outdir), capsys)
    assert code == EXIT_OK
    assert "duplicate row(s)" in err
    assert "1 to dock of 2 (1 already done)" in out
    payload = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert payload["counts"]["docked"] == 2 == payload["counts"]["library"]
    assert len({row["ligand"] for row in payload["records"]}) == 2


def test_a_results_file_of_only_foreign_rows_is_refused(
    demo_3ptb, screen_library, tmp_path, capsys
):
    """HOLE 3: when *no* row matches the run, the file is still another campaign's.

    This is the case a check that filters the rows by receptor name cannot see:
    with rows for another receptor and no manifest, the filtered list is empty,
    the directory looked untouched, and a second campaign was appended to it.
    """
    import hashlib

    outdir = tmp_path / "foreign-only"
    library = tmp_path / "one.smi"
    library.write_text("c1ccc(cc1)C(=N)N benzamidine\n", encoding="utf-8")
    assert run_cli(screen_argv(demo_3ptb, library, outdir, "--limit", "1"), capsys)[0] == 0
    (outdir / "run.json").unlink()                       # what a kill leaves behind
    before = (outdir / "results.jsonl").read_bytes()
    digest = hashlib.sha256(before).hexdigest()

    # The same directory, a *different* receptor (a different file stem, which is
    # what names the rows): none of the rows is this run's.
    other = tmp_path / "other" / "egfr_receptor.pdbqt"
    other.parent.mkdir()
    other.write_text(demo_3ptb["receptor.pdbqt"].read_text(encoding="utf-8"), encoding="utf-8")
    argv = screen_argv(demo_3ptb, library, outdir, "--limit", "1", "--no-interactions")
    argv[argv.index("-r") + 1] = str(other)
    code, _out, err = run_cli(argv, capsys)

    assert code == EXIT_INPUT
    assert "belongs to another campaign" in err
    assert "'receptor'" in err and "egfr_receptor" in err
    assert hashlib.sha256((outdir / "results.jsonl").read_bytes()).hexdigest() == digest
    assert not (outdir / "run.json").exists()            # nothing was written either

    # ... and --no-resume is the documented way out.
    code, _out, err = run_cli(argv + ["--no-resume"], capsys)
    assert code == EXIT_OK
    assert "--no-resume removed" in err
    remaining = read_records(outdir / "results.jsonl")
    assert len(remaining) == 1
    assert {r.receptor for r in remaining} == {"egfr_receptor"}


def test_a_results_file_mixing_two_receptors_is_refused(demo_3ptb, screen_library, tmp_path, capsys):
    """Some of this run's rows plus some foreign ones is a mixture, not a resume."""
    outdir = tmp_path / "mixed-receptors"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys
    )[0] == EXIT_OK
    results = outdir / "results.jsonl"
    foreign = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    foreign["receptor"] = "someone_elses_receptor"
    results.write_text(
        results.read_text(encoding="utf-8") + json.dumps(foreign) + "\n", encoding="utf-8"
    )
    (outdir / "run.json").unlink()
    before = results.read_bytes()

    code, _out, err = run_cli(screen_argv(demo_3ptb, screen_library, outdir), capsys)
    assert code == EXIT_INPUT
    assert "belongs to another campaign" in err
    assert "someone_elses_receptor" in err
    assert results.read_bytes() == before


def test_a_killed_process_does_not_redock_anything(demo_3ptb, screen_library, tmp_path, capsys):
    """HOLE 1: a real SIGKILL mid-campaign must not duplicate the work.

    An in-process ``KeyboardInterrupt`` cannot cover this: it still runs the
    interpreter's cleanup and the end-of-run manifest write, which is exactly why
    the suite was green while a killed process left rows with no manifest and the
    restart re-docked the whole library.  This one starts the CLI as a **child
    process**, waits until a row is on disk, kills it hard, and continues the
    campaign the normal way.
    """
    import subprocess
    import sys

    outdir = tmp_path / "killed"
    argv = [
        sys.executable, "-m", "odock.cli", "screen",
        "-r", str(demo_3ptb["receptor.pdbqt"]),
        "-i", str(screen_library),
        "--box", str(demo_3ptb["box.json"]),
        "-o", str(outdir),
        "-e", "2", "-n", "2", "--seed", "42",
        "--jobs", "1",              # one at a time, so the kill lands mid-campaign
        "--no-interactions", "--quiet",
    ]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT / "python"))
    log = tmp_path / "child.log"
    with log.open("wb") as handle:
        process = subprocess.Popen(
            argv, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        results = outdir / "results.jsonl"
        try:
            deadline = time.monotonic() + 180.0
            while True:
                if results.exists() and len(read_records(results)) >= 1:
                    break
                if time.monotonic() > deadline:  # pragma: no cover - slow machine
                    pytest.fail("the child never wrote a record")
                time.sleep(0.05)
        finally:
            finished_early = process.poll() is not None and not (
                results.exists() and read_records(results)
            )
            process.kill()                      # SIGKILL / TerminateProcess
            process.wait(timeout=60)
    if finished_early:  # pragma: no cover - only if the child cannot start
        pytest.fail(
            "the child exited before writing a row (exit "
            f"{process.returncode}):\n{log.read_text(encoding='utf-8', errors='replace')[-2000:]}"
        )

    before = read_records(results)
    assert before, "the kill landed before the first record was written"
    assert (outdir / "run.json").is_file(), (
        "the manifest must exist before the first docking: a killed process that "
        "leaves rows without it cannot be resumed safely"
    )

    # Restart the *same* command: it must recognise the rows and dock only the rest.
    code, out, _err = run_cli(argv[3:], capsys)
    assert code == EXIT_OK
    assert f"{len(before)} molecule x receptor row(s) already in" in out
    assert "0 already done" not in out
    after = read_records(results)
    assert len(after) == 2                 # the two dockable molecules, once each
    assert len({record.ligand for record in after}) == len(after), "duplicate rows"
    # The rows that were already there are untouched: same ligand, same seed,
    # same affinity — nothing was re-docked.
    for old, new in zip(before, after):
        assert (old.ligand, old.seed, old.affinity) == (new.ligand, new.seed, new.affinity)


def test_rows_without_a_manifest_are_resumed_not_re_docked(
    demo_3ptb, screen_library, tmp_path, capsys
):
    """The same hole without needing a kill: delete run.json and restart."""
    outdir = tmp_path / "manifest-lost"
    assert run_cli(screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys)[0] == 0
    before = read_records(outdir / "results.jsonl")
    assert len(before) == 1
    (outdir / "run.json").unlink()

    code, out, err = run_cli(screen_argv(demo_3ptb, screen_library, outdir), capsys)
    assert code == EXIT_OK
    assert "no run.json" in err and "resuming from those rows anyway" in err
    assert "1 molecule x receptor row(s) already in" in out
    after = read_records(outdir / "results.jsonl")
    assert len(after) == 2 and len({r.ligand for r in after}) == 2
    assert (after[0].ligand, after[0].affinity, after[0].seed) == (
        before[0].ligand, before[0].affinity, before[0].seed
    )
    # ... and the manifest is back, so the next run is a normal resume.
    assert json.loads((outdir / "run.json").read_text(encoding="utf-8"))["phase"] == "done"


def test_a_results_file_from_another_campaign_is_refused(
    demo_3ptb, tmp_path, capsys
):
    """Without a manifest the rows are trusted — unless they are plainly someone else's."""
    outdir = tmp_path / "foreign"
    library = tmp_path / "one.smi"
    library.write_text("c1ccc(cc1)C(=N)N benzamidine\n", encoding="utf-8")
    assert run_cli(screen_argv(demo_3ptb, library, outdir, "--limit", "1"), capsys)[0] == 0
    (outdir / "run.json").unlink()

    # A different box means the rows cannot belong to this command.
    other = odock.BoxSpec(center=(0.0, 0.0, 0.0), size=(30.0, 30.0, 30.0))
    box_file = tmp_path / "other-box.json"
    box_file.write_text(json.dumps(other.as_dict()), encoding="utf-8")
    argv = screen_argv(demo_3ptb, library, outdir, "--limit", "1")
    argv[argv.index("--box") + 1] = str(box_file)
    code, _out, err = run_cli(argv, capsys)
    assert code == EXIT_INPUT
    assert "different grid" in err


def test_a_changed_seed_is_refused(demo_3ptb, screen_library, tmp_path, capsys):
    """HOLE 2: two campaigns with different seeds must not share a results file."""
    outdir = tmp_path / "seed-change"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--seed", "7"), capsys
    )[0] == EXIT_OK
    before = (outdir / "results.jsonl").read_text(encoding="utf-8")

    code, _out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--seed", "8"), capsys
    )
    assert code == EXIT_INPUT
    assert "docking settings changed" in err
    assert "seed: 7 -> 8" in err
    assert (outdir / "results.jsonl").read_text(encoding="utf-8") == before

    # The same seed still resumes, and docks only what is missing.
    code, out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--seed", "7"), capsys
    )
    assert code == EXIT_OK
    assert "1 to dock of 2 (1 already done)" in out


def test_unreadable_records_are_reported_in_the_funnel(demo_3ptb, tmp_path, capsys):
    """The funnel counts what parsed, and says so."""
    library = tmp_path / "broken.smi"
    library.write_text(
        "c1ccc(cc1)C(=N)N benzamidine\n"
        "C1CC[C@@H](C1 not_a_valid_smiles\n"
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C caffeine\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "broken-out"
    code, out, _err = run_cli(
        screen_argv(demo_3ptb, library, outdir, "--dry-run", "--no-filter"), capsys
    )
    assert code == EXIT_OK
    assert "2 read -> 2 kept" in out
    assert "could not be parsed by RDKit" in out
    assert "NOT counted above" in out
    payload = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert payload["library"]["n_unreadable"] == 1
    assert payload["library"]["unreadable"]


def test_no_filter_still_reports_the_descriptors(demo_3ptb, screen_library, tmp_path, capsys):
    """--no-filter skips the *verdict*, not the numbers: the table keeps its MW."""
    outdir = tmp_path / "no-filter"
    code, out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "2", "--no-filter"), capsys
    )
    assert code == EXIT_OK
    ranking = [line for line in out.splitlines() if line.strip().startswith("1 ")]
    assert ranking, out
    cells = ranking[0].split()
    assert float(cells[3]) > 0  # rank | name | affinity | MW | ...
    records = read_records(outdir / "results.jsonl")
    assert all(record.molecular_weight and record.molecular_weight > 0 for record in records)
    assert all(record.properties["MW"] == record.molecular_weight for record in records)
    assert not any(record.violations for record in records)  # nothing was judged


def test_box_wins_over_center_with_a_message(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "both-boxes"
    code, _out, err = run_cli(
        screen_argv(
            demo_3ptb, screen_library, outdir, "--dry-run",
            "--center", "0", "0", "0", "--size", "30", "30", "30",
        ),
        capsys,
    )
    # --box comes first in the argument list built by screen_argv, so it wins.
    assert code == EXIT_OK
    assert "wins over --center/--size" in err


def test_no_poses_means_no_pose_files_and_no_shortlist(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "no-poses"
    code, out, err = run_cli(
        screen_argv(
            demo_3ptb, screen_library, outdir, "--limit", "1", "--no-poses", "--top", "1",
            "--consensus",
        ),
        capsys,
    )
    assert code == EXIT_OK
    assert not (outdir / "poses").exists()
    assert not list(outdir.glob("top_*.pdbqt"))
    record = read_records(outdir / "results.jsonl")[0]
    assert record.pose_file == ""
    assert record.poses  # the affinities of the modes are still recorded
    # ... and asking for a consensus is reported, not silently ignored.
    assert "consensus scoring skipped" in err
    assert "unavailable" in out
    assert not list(outdir.glob("consensus_*.csv"))


def test_two_input_files_can_share_a_name_and_a_molecule_title(
    demo_3ptb, tmp_path, capsys
):
    """The library position is global: keys and prepared files never collide."""
    for sub, smiles in (("a", "c1ccc(cc1)C(=N)N"), ("b", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "library.smi").write_text(f"{smiles} same_title\n", encoding="utf-8")
    outdir = tmp_path / "two-files"
    code, _out, _err = run_cli(
        [
            "screen",
            "-r", str(demo_3ptb["receptor.pdbqt"]),
            "-i", str(tmp_path / "a" / "library.smi"),
            "-i", str(tmp_path / "b" / "library.smi"),
            "--box", str(demo_3ptb["box.json"]),
            "-o", str(outdir),
            "-e", "1", "-n", "2", "--seed", "42", "--no-interactions", "--quiet",
        ],
        capsys,
    )
    assert code == EXIT_OK
    records = read_records(outdir / "results.jsonl")
    assert len(records) == 2
    assert len({record.ligand for record in records}) == 2
    assert len({record.pose_file for record in records}) == 2
    assert all((outdir / record.pose_file).is_file() for record in records)
    prepared = sorted((outdir / "library").glob("*.pdbqt"))
    assert len(prepared) == 2
    # Different molecules must not have overwritten each other: the two prepared
    # files differ, and so do the two docked ligands.
    assert prepared[0].read_text(encoding="utf-8") != prepared[1].read_text(encoding="utf-8")


def test_the_library_cache_keeps_every_descriptor(demo_3ptb, screen_library, tmp_path, capsys):
    """A record rebuilt from library.csv must carry the same descriptors."""
    outdir = tmp_path / "descriptors"
    assert run_cli(screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys)[0] == 0
    fresh = read_records(outdir / "results.jsonl")[0]
    assert {"MW", "LogP", "tPSA", "RotB"} <= set(fresh.properties)

    # The second run restores the library from the cache and docks one more.
    code, _out, _err = run_cli(screen_argv(demo_3ptb, screen_library, outdir), capsys)
    assert code == EXIT_OK
    records = {record.name: record for record in read_records(outdir / "results.jsonl")}
    assert set(records["benzamidine"].properties) == set(fresh.properties)
    assert set(records["caffeine"].properties) == set(fresh.properties)


def test_a_missing_prepared_ligand_forces_a_rebuild(demo_3ptb, screen_library, tmp_path, capsys):
    """The library cache is only trusted while every PDBQT it names exists."""
    outdir = tmp_path / "cache"
    argv = screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1")
    assert run_cli(argv, capsys)[0] == EXIT_OK
    prepared = sorted((outdir / "library").glob("*.pdbqt"))
    assert prepared
    prepared[0].unlink()

    code, out, err = run_cli(argv, capsys)
    assert code == EXIT_OK
    assert "incomplete" in err
    assert "reading the library again" in err
    assert prepared[0].exists()
    assert len(read_records(outdir / "results.jsonl")) == 1  # still remembered


def test_json_out_writes_the_run_summary(demo_3ptb, screen_library, tmp_path, capsys):
    outdir = tmp_path / "json"
    target = tmp_path / "summary.json"
    code, _out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--json-out", str(target)),
        capsys,
    )
    assert code == EXIT_OK
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["counts"]["docked"] == 1
    assert payload["library"]["n_read"] == 4
    assert payload["records"][0]["name"] == "benzamidine"
    assert payload["records"][0]["affinity"] < 0
    assert payload["seed"] == 42


def test_the_writer_flushes_and_checkpoints_every_record(tmp_path):
    from odock.screen import LigandRecord, _ResultWriter

    path = tmp_path / "w.jsonl"
    with _ResultWriter(path, "jsonl", checkpoint_every=1) as writer:
        for i in range(3):
            writer.write(LigandRecord(receptor="r", ligand=f"lib#{i}", name=f"m{i}"))
        # A second reader sees the records before the writer is closed.
        assert len(read_records(path)) == 3
        assert writer.written == 3
    assert [r.name for r in read_records(path)] == ["m0", "m1", "m2"]

    # The CSV writer emits its header exactly once, even across two sessions.
    csv_path = tmp_path / "w.csv"
    with _ResultWriter(csv_path, "csv", checkpoint_every=2) as writer:
        writer.write(LigandRecord(receptor="r", ligand="lib#1", name="m1"))
    with _ResultWriter(csv_path, "csv", checkpoint_every=2) as writer:
        writer.write(LigandRecord(receptor="r", ligand="lib#2", name="m2"))
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("receptor,ligand,name")


# ---------------------------------------------------------------------------
# Consensus scoring of the shortlist
# ---------------------------------------------------------------------------


def test_consensus_ranks_the_hits_across_force_fields(demo_3ptb, screen_library, tmp_path, capsys):
    """The funnel ends with "which hits survive a change of force field"."""
    pytest.importorskip("odock.consensus")
    outdir = tmp_path / "consensus"
    code, out, err = run_cli(
        screen_argv(
            demo_3ptb, screen_library, outdir,
            "--consensus", "--top", "2", "--consensus-top", "2",
        ),
        capsys,
    )
    assert code == EXIT_OK
    assert "consensus" in out
    assert "force-field agreement" in out
    # Every force field gets a column, and the consensus names the same hits.
    for field in ("vina", "vinardo", "ad4"):
        assert field in out
    assert "benzamidine" in out and "caffeine" in out
    assert "rescoring the top 2 of 2 hit(s)" in out

    table = outdir / "consensus_receptor.csv"
    assert table.is_file()
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert sorted(int(row["consensus_rank"]) for row in rows) == [1, 2]
    for row in rows:
        assert row["name"] and row["pose_file"]
        # Ligand-level ranks are 1..N under every force field.
        for field in ("vina", "vinardo", "ad4"):
            assert float(row[f"{field}_rank"]) in (1.0, 2.0)
            assert float(row[field]) < 0
        assert float(row["consensus_score"]) >= 0
        assert row["docked_pose_agrees"] in ("0", "1")
    # The consensus order is a permutation of the screening order.
    assert int(rows[0]["consensus_rank"]) == 1

    payload = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    block = payload["consensus"]["receptor"]
    assert block["scorings"] == ["vina", "vinardo", "ad4"]
    assert block["method"] == "rank"
    assert block["n_ligands"] == 2
    assert -1.0 <= block["agreement"] <= 1.0
    assert len(block["correlations"]) == 3
    assert block["error"] == ""


def test_consensus_can_be_added_to_a_finished_campaign(demo_3ptb, screen_library, tmp_path, capsys):
    """It is post-processing: no docking is repeated to get it."""
    pytest.importorskip("odock.consensus")
    outdir = tmp_path / "after"
    assert run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1"), capsys
    )[0] == EXIT_OK
    before = read_records(outdir / "results.jsonl")[0]

    code, out, _err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--consensus"), capsys
    )
    assert code == EXIT_OK
    assert "already done" in out
    assert "force-field agreement" in out
    after = read_records(outdir / "results.jsonl")[0]
    assert (after.ligand, after.affinity, after.seed) == (
        before.ligand, before.affinity, before.seed
    )  # nothing was docked again
    assert (outdir / "consensus_receptor.csv").is_file()


def test_a_missing_consensus_module_does_not_fail_the_campaign(
    demo_3ptb, screen_library, tmp_path, capsys, monkeypatch
):
    from odock import screen as screen_module

    def unavailable():
        raise ImportError("no consensus in this build")

    monkeypatch.setattr(screen_module, "_load_consensus", unavailable)
    outdir = tmp_path / "no-consensus"
    code, out, err = run_cli(
        screen_argv(demo_3ptb, screen_library, outdir, "--limit", "1", "--consensus"), capsys
    )
    assert code == EXIT_OK  # the campaign itself is fine
    assert "consensus scoring failed" in err
    assert "consensus: unavailable" in out
    assert not list(outdir.glob("consensus_*.csv"))
    payload = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    assert payload["consensus"]["receptor"]["error"].startswith("consensus scoring is unavailable")
    assert len(read_records(outdir / "results.jsonl")) == 1

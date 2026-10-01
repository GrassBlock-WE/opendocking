# SPDX-License-Identifier: GPL-3.0-or-later
"""The re-docking benchmark: metrics, reporting, baseline and CLI.

The docking itself is exercised by one ``slow`` test (a single cheap system, one
seed); everything else here is about the *bookkeeping* -- the numbers that make a
benchmark comparable, the table, the JSON/CSV records, and the baseline check
that is supposed to catch a scoring regression.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

from odock import benchmark

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data"


# ---------------------------------------------------------------------------
# The system definitions
# ---------------------------------------------------------------------------


def test_every_system_is_defined_by_bundled_data_and_a_smiles():
    assert len(benchmark.SYSTEMS) >= 4
    for system in benchmark.SYSTEMS:
        assert (DATA / f"{system.pdb_id}.pdb").exists(), system.pdb_id
        assert system.ligand and system.ligand.isupper()
        assert system.smiles and len(system.smiles) > 3
        assert 0 < system.exhaustiveness
        assert system.note, system.pdb_id
    ids = [system.pdb_id for system in benchmark.SYSTEMS]
    assert len(ids) == len(set(ids))


def test_every_smiles_parses_and_matches_the_crystal_heavy_atom_count():
    """The template has to be the same molecule as the crystal residue."""
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem

    for system in benchmark.SYSTEMS:
        template = Chem.MolFromSmiles(system.smiles)
        assert template is not None, system.pdb_id
        crystal = Chem.MolFromPDBFile(
            str(DATA / f"{system.pdb_id}.pdb"), removeHs=False, sanitize=False,
            proximityBonding=True,
        )
        assert crystal is not None
        residue = [
            atom.GetIdx()
            for atom in crystal.GetAtoms()
            if atom.GetPDBResidueInfo() is not None
            and atom.GetPDBResidueInfo().GetResidueName().strip().upper()
            == system.ligand
        ]
        heavy = sum(
            1 for index in residue
            if crystal.GetAtomWithIdx(index).GetAtomicNum() > 1
        )
        assert template.GetNumAtoms() == heavy, (
            f"{system.pdb_id}: SMILES has {template.GetNumAtoms()} heavy atoms, "
            f"the {system.ligand} residue has {heavy}"
        )


# ---------------------------------------------------------------------------
# The metrics
# ---------------------------------------------------------------------------


def _metrics(**kwargs):
    return benchmark.RunMetrics(seed=kwargs.pop("seed", 1), **kwargs)


def _outcome(runs, system=None):
    return benchmark.SystemResult(
        system=system or benchmark.SYSTEMS[0], runs=list(runs), n_heavy=9,
        n_rotatable=1, receptor_atoms=1994,
    )


def test_the_aggregate_takes_the_best_seed_and_reports_the_spread():
    good = _metrics(seed=1, top_rmsd=1.0, best_within_1kcal=0.8, rank_of_correct=1,
                    best_affinity=-9.0, score_rmsd_rho=-0.5, agreement=0.9)
    bad = _metrics(seed=2, top_rmsd=2.5, best_within_1kcal=1.9, rank_of_correct=4,
                   best_affinity=-7.5, score_rmsd_rho=0.1, agreement=0.6)
    outcome = _outcome([good, bad])
    assert outcome.top_rmsd == pytest.approx(1.0)
    assert outcome.top_rmsd_worst == pytest.approx(2.5)
    assert outcome.top_rmsd_spread == pytest.approx(1.5)
    assert outcome.best_within_1kcal == pytest.approx(0.8)
    assert outcome.rank_of_correct == 1
    assert outcome.affinity_spread == pytest.approx(1.5)
    assert outcome.rho_range == pytest.approx((-0.5, 0.1))
    assert outcome.agreement_range == pytest.approx((0.6, 0.9))
    assert outcome.correct_in_all_runs is True  # rank 4 still counts as a hit
    assert outcome.poses == 0  # neither run reported a pose count


def test_correct_in_all_runs_needs_a_hit_in_every_seed():
    hit = _metrics(seed=1, rank_of_correct=2)
    miss = _metrics(seed=2, rank_of_correct=0)
    assert _outcome([hit]).correct_in_all_runs is True
    assert _outcome([hit, miss]).correct_in_all_runs is False
    assert _outcome([]).correct_in_all_runs is False


def test_the_weakest_pair_is_ordered_by_the_mean_over_seeds():
    first = _metrics(seed=1, correlations={"vina~vinardo": 0.9, "vina~ad4": 0.2,
                                           "vinardo~ad4": 0.5})
    second = _metrics(seed=2, correlations={"vina~vinardo": 0.8, "vina~ad4": 0.4,
                                            "vinardo~ad4": 0.6})
    outcome = _outcome([first, second])
    assert outcome.pair_order()[0] == "vina~ad4"
    assert outcome.pair_order()[-1] == "vina~vinardo"
    means = outcome.pair_means()
    assert means["vina~ad4"] == pytest.approx(0.3)
    assert means["vina~vinardo"] == pytest.approx(0.85)


def test_a_non_finite_metric_is_dropped_from_the_aggregate():
    """One undefined correlation must not poison the mean of the others."""
    undefined = _metrics(seed=1, top_rmsd=float("nan"), score_rmsd_rho=float("nan"),
                         agreement=float("nan"),
                         correlations={"vina~ad4": float("nan")})
    defined = _metrics(seed=2, top_rmsd=1.5, score_rmsd_rho=-0.2, agreement=0.7,
                       correlations={"vina~ad4": 0.3})
    outcome = _outcome([undefined, defined])
    assert outcome.top_rmsd == pytest.approx(1.5)
    assert outcome.rho_range == pytest.approx((-0.2, -0.2))
    assert outcome.pair_means() == {"vina~ad4": pytest.approx(0.3)}


# ---------------------------------------------------------------------------
# The table and the records
# ---------------------------------------------------------------------------


def test_the_table_carries_the_sample_size_next_to_every_rho():
    run = _metrics(seed=1, poses=11, top_rmsd=1.13, best_within_1kcal=1.13,
                   rank_of_correct=1, score_rmsd_rho=0.83, agreement=0.89,
                   correlations={"vina~vinardo": 0.9, "vina~ad4": 0.8,
                                 "vinardo~ad4": 0.7})
    text = benchmark.table([_outcome([run])])
    assert "rho (n)" in text
    assert "n=11" in text
    assert "+0.83" in text
    # The pair ordering is printed, weakest first.
    assert "vinardo~ad4" in text
    assert "no superposition" in text and "negative is good" in text


def test_the_table_says_so_when_a_metric_is_undefined():
    run = _metrics(seed=1, poses=1, top_rmsd=1.02, best_within_1kcal=1.02,
                   rank_of_correct=1, score_rmsd_rho=float("nan"),
                   agreement=float("nan"))
    text = benchmark.table([_outcome([run])])
    assert "--" in text


def test_the_table_reports_a_failed_system_instead_of_crashing():
    outcome = benchmark.SystemResult(system=benchmark.SYSTEMS[0],
                                     error="missing tests/data/9XYZ.pdb")
    text = benchmark.table([outcome])
    assert "FAILED" in text and "9XYZ" in text


def test_records_are_json_serialisable_and_nested_per_seed():
    run = _metrics(seed=5, poses=4, top_rmsd=1.2, rmsds=[1.2, 2.0],
                   affinities=[-7.0, -6.5], correlations={"vina~ad4": 0.5})
    outcome = _outcome([run])
    rows = benchmark.records([outcome])
    assert len(rows) == 1
    assert rows[0]["system"] == benchmark.SYSTEMS[0].pdb_id
    assert rows[0]["runs"][0]["seed"] == 5
    assert rows[0]["runs"][0]["rmsds"] == [1.2, 2.0]
    assert json.loads(json.dumps(rows))  # nothing exotic inside


def test_json_and_csv_writers_round_trip(tmp_path):
    rows = benchmark.records([_outcome([_metrics(seed=1, top_rmsd=1.0, poses=3)])])
    json_path = benchmark.write_json(tmp_path / "out" / "b.json", {"systems": rows})
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["systems"][0]["top_rmsd"] == 1.0

    csv_path = benchmark.write_csv(tmp_path / "out" / "b.csv", rows)
    text = Path(csv_path).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("system,ligand,n_heavy")
    assert "3PTB" in text
    # The nested fields survive as JSON strings (with the quotes doubled by the
    # CSV writer) rather than being silently dropped.
    assert '""seed"": 1' in text


# ---------------------------------------------------------------------------
# The baseline
# ---------------------------------------------------------------------------


def _baseline_and_run(top_rmsd: float, rho: float = 0.5, rank: int = 1,
                      correct_all: bool = True):
    """A two-seed result.  ``correct_all=False`` makes *both* seeds miss the hit."""
    outcome = _outcome([
        _metrics(seed=1, top_rmsd=top_rmsd, best_within_1kcal=top_rmsd,
                 rank_of_correct=rank if correct_all else 0, score_rmsd_rho=rho),
        _metrics(seed=2, top_rmsd=top_rmsd + 0.2, best_within_1kcal=top_rmsd + 0.2,
                 rank_of_correct=rank if correct_all else 0, score_rmsd_rho=rho + 0.1),
    ])
    return outcome


def test_the_baseline_tolerance_comes_from_repeatability_not_the_seed_spread():
    """The check re-runs the same command: its noise is repeat-run noise.

    A tolerance taken from the seed-to-seed spread (3.17 A on 1M17 here) would
    let a real regression through, so the two must not be confused.
    """
    outcome = _baseline_and_run(1.0)          # top_rmsd_spread = 0.2
    document = benchmark.baseline_from([outcome])
    tolerance = document["tolerance"]
    assert tolerance["top_rmsd"] == pytest.approx(benchmark.REPEATABILITY_FLOOR)
    assert tolerance["top_rmsd"] < 0.2, "the seed spread must not set the tolerance"
    assert "repeat-run noise" in tolerance["derivation"]
    assert "top_rmsd_spread" in tolerance["derivation"]
    assert document["systems"]["3PTB"]["top_rmsd"] == pytest.approx(1.0)
    assert document["systems"]["3PTB"]["top_rmsd_spread"] == pytest.approx(0.2)
    assert document["environment"]["kernel"]


def test_a_measured_repeatability_spread_widens_the_tolerance():
    document = benchmark.baseline_from(
        [_baseline_and_run(1.0)], repeatability={"top_rmsd": 0.4, "score_rmsd_rho": 0.3}
    )
    assert document["tolerance"]["top_rmsd"] == pytest.approx(0.4)
    assert document["tolerance"]["score_rmsd_rho"] == pytest.approx(0.3)


def test_a_matching_run_passes_the_baseline_check():
    document = benchmark.baseline_from([_baseline_and_run(1.0)])
    assert benchmark.check_baseline([_baseline_and_run(1.05)], document) == []


def test_a_regression_is_reported_with_the_numbers():
    document = benchmark.baseline_from([_baseline_and_run(1.0)])
    failures = benchmark.check_baseline([_baseline_and_run(2.4)], document)
    assert failures and "top_rmsd" in failures[0]
    assert "worse than the baseline" in failures[0]
    assert "2.40" in failures[0] and "1.00" in failures[0]


def test_a_change_within_the_tolerance_is_not_a_regression():
    document = benchmark.baseline_from([_baseline_and_run(1.0)])
    assert benchmark.check_baseline([_baseline_and_run(1.1)], document) == []


def test_a_lost_hit_and_a_rank_regression_are_reported():
    document = benchmark.baseline_from([_baseline_and_run(1.0)])
    failures = benchmark.check_baseline(
        [_baseline_and_run(1.0, rank=0, correct_all=False)], document
    )
    assert any("correct pose was found in every baseline run" in line
               for line in failures)


def test_a_worse_correlation_is_reported_on_the_worst_side():
    """rho is [worst, best] over seeds and negative is good, so compare worsts."""
    document = benchmark.baseline_from([_baseline_and_run(1.0, rho=0.4)])
    stored = document["systems"]["3PTB"]["score_rmsd_rho"]
    assert stored[0] < stored[1], stored  # [best, worst] as (min, max)

    worse = _outcome([
        _metrics(seed=1, top_rmsd=1.0, best_within_1kcal=1.0, rank_of_correct=1,
                 score_rmsd_rho=float(stored[1]) + 1.0),
    ])
    failures = benchmark.check_baseline([worse], document)
    assert any("score_rmsd_rho worst" in line for line in failures), failures

    same = _outcome([
        _metrics(seed=1, top_rmsd=1.0, best_within_1kcal=1.0, rank_of_correct=1,
                 score_rmsd_rho=float(stored[1]) - 0.1),
    ])
    assert benchmark.check_baseline([same], document) == []


def test_the_mean_correlation_and_agreement_are_real_means():
    outcome = _outcome([
        _metrics(seed=1, score_rmsd_rho=0.1, agreement=0.5),
        _metrics(seed=2, score_rmsd_rho=0.5, agreement=0.9),
    ])
    assert outcome.rho_mean == pytest.approx(0.3)
    assert outcome.agreement_mean == pytest.approx(0.7)
    assert math.isnan(_outcome([]).rho_mean)


def test_a_single_seed_falls_back_to_the_hit_fraction():
    """One seed cannot show cluster stability, but the field must not lie."""
    outcome = _outcome([_metrics(seed=1, rank_of_correct=1)])
    outcome.reproducibility = benchmark._top_cluster_stability(outcome.runs)
    assert math.isnan(outcome.reproducibility)  # one run is not a stability
    two = _outcome([_metrics(seed=1, rank_of_correct=1),
                    _metrics(seed=2, rank_of_correct=0)])
    assert benchmark._top_cluster_stability(two.runs) == pytest.approx(0.5)


def test_a_system_the_baseline_does_not_know_is_ignored():
    document = benchmark.baseline_from([_baseline_and_run(1.0)])
    document["systems"] = {}
    assert benchmark.check_baseline([_baseline_and_run(9.9)], document) == []


def test_the_recorded_command_is_portable():
    """A published baseline must not leak the machine it was measured on."""
    root = str(Path(benchmark.__file__).resolve().parent.parent.parent)
    # This checkout's root never appears...
    cleaned = benchmark.portable_command(
        "python -m odock.benchmark --json " + root + os.sep + "out" + os.sep + "b.json"
    )
    assert root not in cleaned
    assert cleaned == "python -m odock.benchmark --json out/b.json".replace("/", os.sep)
    # ...nor does any interpreter path, whoever's it is...
    for absolute in (
        "C:\\Users\\someone\\project\\.venv\\Scripts\\python.exe",
        "/opt/venv/bin/python",
    ):
        cleaned = benchmark.portable_command(f"{absolute} -m odock.benchmark --seeds 42")
        assert cleaned == "python -m odock.benchmark --seeds 42"
    # ...and this machine's own interpreter is named, not spelled out.
    assert benchmark.portable_command(
        f"{sys.executable} -m odock.benchmark"
    ) == "python -m odock.benchmark"
    # A caller's own relative argument is left exactly as it was.
    assert benchmark.portable_command(
        "python -m odock.benchmark --data-dir my/data"
    ) == "python -m odock.benchmark --data-dir my/data"
    # A run recorded as a script path (what `sys.argv[0]` is under `python -m`)
    # becomes the module form, so the recorded command is still runnable.
    assert benchmark.portable_command(
        root + os.sep + "python" + os.sep + "odock" + os.sep + "benchmark.py --seeds 42"
    ) == "python -m odock.benchmark --seeds 42"


def test_the_baseline_sanitises_an_inherited_command():
    """Re-cutting from an older run must not carry its absolute path along."""
    document = benchmark.baseline_from(
        [_baseline_and_run(1.0)],
        command="C:\\Users\\someone\\project\\python -m odock.benchmark --seeds 42",
    )
    assert "someone" not in document["command"]
    assert document["command"].endswith("-m odock.benchmark --seeds 42")


def test_the_cli_command_names_the_module_not_the_script():
    assert benchmark.cli_command(["--seeds", "42", "--fast"]) == (
        "python -m odock.benchmark --seeds 42 --fast"
    )


def test_a_failed_run_is_a_baseline_failure():
    document = benchmark.baseline_from([_baseline_and_run(1.0)])
    broken = benchmark.SystemResult(system=benchmark.SYSTEMS[0], error="boom")
    failures = benchmark.check_baseline([broken], document)
    assert failures and "the run failed" in failures[0]


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def test_the_cli_rejects_an_unknown_system(capsys):
    code = benchmark.main(["--systems", "ZZZZ", "--quiet"])
    assert code == 2
    assert "unknown system" in capsys.readouterr().err


def test_the_cli_reports_a_missing_baseline(tmp_path, capsys):
    code = benchmark.main([
        "--systems", "3PTB", "--check-baseline", "--quiet",
        "--baseline", str(tmp_path / "nope.json"),
    ])
    assert code == 2
    assert "no baseline" in capsys.readouterr().err


def test_the_cli_documents_itself(capsys):
    with pytest.raises(SystemExit) as caught:
        benchmark._parse_args(["--help"])
    assert caught.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--systems", "--seeds", "--fast", "--json", "--csv",
                 "--check-baseline", "--write-baseline"):
        assert flag in text, flag


# ---------------------------------------------------------------------------
# The real thing (slow): one cheap system, one seed
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_real_system_docks_and_measures_in_the_expected_range():
    """3PTB at low effort: the acceptance case has to stay reproducible."""
    outcome = benchmark.run_system(
        benchmark.SYSTEMS[0], data_dir=DATA, seeds=[42], exhaustiveness=4,
        verbose=False,
    )
    assert not outcome.error, outcome.error
    assert outcome.runs and outcome.runs[0].poses >= 1
    run = outcome.runs[0]
    assert run.top_rmsd <= 2.0, f"top pose is {run.top_rmsd:.2f} A from the crystal"
    assert run.rank_of_correct == 1
    assert run.best_within_1kcal <= 2.0
    assert run.best_affinity < -4.0
    assert outcome.n_heavy == 9 and outcome.n_rotatable == 1
    assert outcome.receptor_atoms > 1900
    # A ranking correlation needs at least three poses to mean anything.
    if run.poses >= 3:
        assert math.isfinite(run.score_rmsd_rho)
    else:
        assert math.isnan(run.score_rmsd_rho)

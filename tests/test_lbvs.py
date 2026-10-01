# SPDX-License-Identifier: GPL-3.0-or-later
"""The ligand-based benchmark: metrics, controls, methods and the pre-filter.

Every metric is pinned by a hand computation (the enrichment arithmetic, the AUC
as a Mann-Whitney statistic) or by its calibration property (a perfect ranking
scores 1.0 and a random one 0.5 in expectation, over 200 seeded shuffles).
"""

from __future__ import annotations

import math
import random
import statistics
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import lbvs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "library.smi"
POOL = ROOT / "demo" / "decoys.smi"

AMIDINES = (
    "benzamidine",
    "benzamidine_methyl",
    "hydroxybenzamidine",
    "fluorobenzamidine",
    "chloro_benzamidine",
)


def named(smiles: str, name: str):
    mol = Chem.MolFromSmiles(smiles)
    mol.SetProp("_Name", name)
    return mol


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_enrichment_factor_is_the_hand_computation():
    """Ten molecules, two actives at the top: at 20 % the top k = 2 molecules are
    both actives, the base rate is 0.2 and EF = (2/2)/0.2 = 5."""
    scores = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    labels = [True, True] + [False] * 8
    report = lbvs.ef_at(scores, labels, 0.2)
    assert report["k"] == 2
    assert report["actives_in_top_k"] == 2
    assert report["precision"] == 1.0 and report["recall"] == 1.0
    assert report["base_rate"] == pytest.approx(0.2)
    assert report["ef"] == pytest.approx(5.0)
    # 1 % of ten molecules is one molecule: EF = (1/1)/0.2 = 5 for an active first,
    # and 0 for a decoy first.
    assert lbvs.ef_at(scores, labels, 0.01)["ef"] == pytest.approx(5.0)
    assert lbvs.ef_at(scores, [False] * 8 + [True, True], 0.01)["ef"] == pytest.approx(0.0)


def test_auc_is_the_mann_whitney_statistic():
    perfect = lbvs.auc([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [True] * 5 + [False] * 5)
    reversed_ = lbvs.auc([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [True] * 5 + [False] * 5)
    assert perfect == 1.0 and reversed_ == 0.0
    # Hand check on a tie: one active and one decoy at the same score is a half.
    assert lbvs.auc([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    # One active level with two decoys and above a third: (1 + 0.5 + 0.5) / 3.
    assert lbvs.auc([0.5, 0.2, 0.5, 0.5], [True, False, False, False]) == pytest.approx(2.0 / 3)


def test_bedroc_is_calibrated_perfect_one_and_random_one_half():
    """The two properties BEDROC advertises, checked rather than assumed."""
    perfect = lbvs.bedroc([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [True] * 5 + [False] * 5)
    reversed_ = lbvs.bedroc([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [True] * 5 + [False] * 5)
    assert perfect == pytest.approx(1.0)
    assert reversed_ == pytest.approx(0.0)
    random_values = []
    for seed in range(200):
        rng = random.Random(seed)
        scores = [rng.random() for _ in range(30)]
        labels = [index < 6 for index in range(30)]
        random_values.append(lbvs.bedroc(scores, labels))
    mean = statistics.mean(random_values)
    assert mean == pytest.approx(0.5, abs=0.03), f"random BEDROC mean {mean}"
    # The weight really is early-recognition: concentrating the actives at the top
    # beats spreading them through the list.
    early = lbvs.bedroc([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [True] * 3 + [False] * 4 + [True] * 2 + [False])
    spread = lbvs.bedroc([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [False, True, False, True, False, True, False, True, False, True])
    assert early > spread


def test_metrics_reports_the_counts_and_both_fractions():
    scores = list(range(20, 0, -1))
    labels = [index < 4 for index in range(20)]
    report = lbvs.metrics(scores, labels)
    assert report["n"] == 20 and report["n_actives"] == 4
    assert report["ef1"] == pytest.approx(5.0)  # top 1 % of 20 = 1 molecule, an active
    assert report["ef5"] == pytest.approx(5.0)  # top 5 % = 1 molecule as well
    assert report["ef1_k"] == 1 and report["ef5_k"] == 1
    assert report["auc"] == 1.0 and report["bedroc"] == 1.0


def test_metrics_validate_their_input():
    with pytest.raises(ValueError, match="same length"):
        lbvs.metrics([1.0, 2.0], [True])
    with pytest.raises(ValueError, match="no molecule"):
        lbvs.metrics([], [])


def test_bootstrap_interval_brackets_the_estimate_and_widens_when_it_must():
    scores = list(range(30, 0, -1))
    labels = [index < 6 for index in range(30)]
    low, high = lbvs.bootstrap_ci(scores, labels, lbvs.auc, samples=200, seed=1)
    assert low <= lbvs.auc(scores, labels) <= high
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0), "a perfect ranking has no spread"
    # A mixed ranking has a real interval, and the two-sided percentile order holds.
    mixed = [0.5, 0.9, 0.1, 0.8, 0.4, 0.7, 0.2, 0.6, 0.3, 0.0]
    labels = [True, False, True, False, True, False, True, False, True, False]
    low, high = lbvs.bootstrap_ci(mixed, labels, lbvs.auc, samples=300, seed=2)
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_of_a_metric_with_no_positives_is_empty():
    low, high = lbvs.bootstrap_ci([1.0, 2.0], [True, True], lbvs.auc, samples=10)
    assert math.isnan(low) and math.isnan(high)


# ---------------------------------------------------------------------------
# Methods and controls
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def molecules():
    if not LIBRARY.exists():
        pytest.skip("the bundled demo library is missing")
    mols = []
    for line in LIBRARY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        mol.SetProp("_Name", " ".join(fields[1:]))
        mols.append(mol)
    return mols


@pytest.fixture(scope="module")
def actives(molecules):
    return [mol for mol in molecules if mol.GetProp("_Name") in AMIDINES]


def test_fingerprint_scores_leave_the_molecule_out_of_its_own_query(molecules, actives):
    """The leak that makes a benchmark meaningless, demonstrated: scoring an
    active against the full active set gives it 1.0 to itself, so the ranking is
    perfect for free.  Leave-one-out scores it against the other actives."""
    library = list(molecules)
    names = [mol.GetProp("_Name") for mol in library]
    leaky = lbvs.fingerprint_scores(library, actives, names=names, leave_one_out=False)
    honest = lbvs.fingerprint_scores(library, actives, names=names, leave_one_out=True)
    assert leaky.entries[0][1] == 1.0 and honest.entries[0][1] < 1.0
    assert leaky.notes[0].startswith("self-similarity")
    assert honest.notes[0] == "leave-one-out"
    # The amidines still come first: they are a congeneric series, so with the
    # self-match removed they still outrank every other molecule in the library.
    assert {
        "benzamidine",
        "benzamidine_methyl",
        "hydroxybenzamidine",
        "fluorobenzamidine",
    } <= set(honest.names[:5])


def test_scores_are_ranked_and_complete(molecules, actives):
    names = [mol.GetProp("_Name") for mol in molecules]
    ranking = lbvs.fingerprint_scores(molecules, actives, names=names)
    assert len(ranking) == len(molecules)
    assert set(ranking.names) == set(names)
    assert ranking.scores == sorted(ranking.scores, reverse=True)
    assert ranking.rank_of("caffeine") is not None
    assert ranking.rank_of("nothing") is None


def test_property_control_ranks_by_the_property_ascending(molecules):
    names = [mol.GetProp("_Name") for mol in molecules]
    ranking = lbvs.property_scores(molecules, by="MW", names=names)
    from odock.decoys import property_profile

    by_name = {mol.GetProp("_Name"): mol for mol in molecules}
    weights = [
        property_profile(by_name[ranking.names[0]]).MW,
        property_profile(by_name[ranking.names[-1]]).MW,
    ]
    assert weights[0] < weights[1], "ascending, so the ranking cannot favour big actives"
    assert ranking.method == "property_MW"
    with pytest.raises(ValueError, match="unknown property"):
        lbvs.property_scores(molecules, by="nonsense")


def test_random_control_is_seeded_and_reproducible(molecules):
    names = [mol.GetProp("_Name") for mol in molecules]
    first = lbvs.random_scores(molecules, seed=7, names=names)
    second = lbvs.random_scores(molecules, seed=7, names=names)
    other = lbvs.random_scores(molecules, seed=8, names=names)
    assert first.names == second.names
    assert first.names != other.names
    assert first.method == "random" and first.notes == ["seed 7"]


def test_prefilter_reports_what_it_keeps_and_what_it_saves(molecules, actives):
    """The realistic use of a ligand-based score: dock only the top fraction and
    accept the active recall it costs."""
    names = [mol.GetProp("_Name") for mol in molecules]
    summary = lbvs.prefilter(molecules, actives, keep=0.5)
    assert summary["n_library"] == len(molecules)
    assert summary["n_kept"] == math.ceil(0.5 * len(molecules))
    assert summary["n_actives"] == 5
    assert summary["actives_kept"] == 5, "half the library keeps all five amidines"
    assert summary["active_recall"] == 1.0
    assert summary["workload_saved_fraction"] > 0.0
    assert summary["estimated_seconds_kept"] < summary["estimated_seconds_full"]
    # A brutal cut loses actives, and says which.  6 % of 17 molecules rounds up
    # to two molecules, which keeps two of the five amidines.
    tight = lbvs.prefilter(molecules, actives, keep=0.06)
    assert tight["n_kept"] == 2
    assert tight["actives_kept"] == 2
    assert tight["actives_lost"], "the molecules thrown away are named"


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------


def test_benchmark_reports_every_method_with_its_controls(molecules, actives):
    """A tiny labelled set, so the table is fast to build: the point of this test
    is the *shape* of the report and the caveats it carries, not its numbers."""
    actives = [mol for mol in molecules if mol.GetProp("_Name") in AMIDINES[:3]]
    decoys = [
        named("Nc1ccccn1", "2_aminopyridine"),
        named("Nc1ncccn1", "2_aminopyrimidine"),
        named("Nc1nccs1", "2_aminothiazole"),
        named("NCc1ccccn1", "2_aminomethylpyridine"),
    ]
    report = lbvs.benchmark(
        actives,
        decoys,
        methods=("fingerprint", "pharmacophore"),
        conformers=1,
        bootstrap=20,
    )
    methods = {result.method for result in report.results}
    assert {"fingerprint_morgan", "pharmacophore", "random", "property_MW", "property_LogP"} <= methods
    assert report.n_actives == 3 and report.n_decoys == 4
    for result in report.results:
        assert 0.0 <= result.stats["auc"] <= 1.0
        assert 0.0 <= result.stats["bedroc"] <= 1.0
        assert result.stats["ef1"] >= 0.0
        assert "auc" in result.intervals and "bedroc" in result.intervals
    table = report.table()
    assert "EF1%" in table and "BEDROC" in table and "95% CI" in table
    assert any("case study" in note for note in report.notes)


def test_benchmark_flags_an_unmatched_decoy_set():
    report = lbvs.benchmark(
        [named("N=C(N)c1ccccc1", "benzamidine")],
        [named("c1ccccc1", "benzene")],
        methods=("fingerprint",),
        controls=(),
        bootstrap=5,
        decoy_quality={"matched": False, "max_abs_smd": 1.4},
    )
    assert any("NOT property-matched" in note for note in report.notes)
    assert report.as_dict()["decoy_quality"]["max_abs_smd"] == 1.4


def test_require_rdkit_and_argument_validation(monkeypatch):
    with pytest.raises(ValueError, match="no active"):
        lbvs.fingerprint_scores(["CCO"], [])
    with pytest.raises(ValueError, match="not a parsable molecule"):
        lbvs.fingerprint_scores(["not a molecule"], [named("CCO", "ethanol")])
    monkeypatch.setattr(lbvs, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        lbvs.require_rdkit()

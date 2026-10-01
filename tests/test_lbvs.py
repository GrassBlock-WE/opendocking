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
LIBRARY = ROOT / "demo" / "libraries" / "library.smi"
POOL = ROOT / "demo" / "libraries" / "decoys.smi"

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
# Power: the paired test, the minimum detectable difference, stratification
# ---------------------------------------------------------------------------


def test_minimum_detectable_difference_is_two_point_eight_standard_errors():
    """The constant is derived, not asserted: 1.96 + 0.84 = 2.80 at 80 % power."""
    assert lbvs.MDD_Z == pytest.approx(2.80, abs=0.01)
    assert lbvs.minimum_detectable_difference(0.0) == 0.0
    assert lbvs.minimum_detectable_difference(0.05) == pytest.approx(0.14, abs=0.005)
    # A 99 % level is stricter than 95 %, and 95 % power stricter than 80 %.
    assert lbvs.minimum_detectable_difference(0.05, level=0.99) > (
        lbvs.minimum_detectable_difference(0.05, level=0.95)
    )
    assert lbvs.minimum_detectable_difference(0.05, power=0.95) > (
        lbvs.minimum_detectable_difference(0.05, power=0.80)
    )
    assert math.isnan(lbvs.minimum_detectable_difference(float("nan")))


def test_the_paired_difference_cancels_the_noise_the_intervals_share():
    """Two rankings of the same molecules: the paired test is the sharper one.

    Both rankings are noisy (five actives and twenty-five decoys drawn from
    overlapping distributions), so each AUC has a real interval.  The *paired*
    standard error must be smaller than the unpaired one — ``sqrt(se_a² + se_b²)`` —
    because the shared molecule-to-molecule noise cancels in the difference, and
    that is the entire reason the comparison is paired.
    """
    rng = random.Random(11)
    labels = [True] * 5 + [False] * 25
    a = [rng.gauss(1.0 if flag else 0.0, 1.0) for flag in labels]
    b = [rng.gauss(1.0 if flag else 0.0, 1.0) for flag in labels]
    a_auc = lbvs.auc(a, labels)
    b_auc = lbvs.auc(b, labels)
    paired = lbvs.paired_difference(a, b, labels, samples=2000, seed=7)
    assert paired["difference"] == pytest.approx(a_auc - b_auc, abs=1e-9)
    low, high = paired["ci"]
    assert low <= paired["difference"] <= high
    se_a = lbvs.bootstrap_ci(a, labels, lbvs.auc, samples=400, seed=7)
    se_b = lbvs.bootstrap_ci(b, labels, lbvs.auc, samples=400, seed=7)
    unpaired_width = math.hypot(se_a[1] - se_a[0], se_b[1] - se_b[0])
    assert (high - low) < unpaired_width, (
        "the paired interval must be tighter than two independent ones"
    )
    assert paired["se"] > 0.0
    assert paired["mdd"] == pytest.approx(2.80 * paired["se"], rel=0.01)
    # Comparing a ranking with itself is not just unresolvable, it is exactly zero.
    same = lbvs.paired_difference(a, a, labels, samples=100)
    assert same["difference"] == 0.0
    assert same["ci"] == (0.0, 0.0) and same["resolvable"] is False
    assert same["n_actives"] == 5 and same["n_decoys"] == 25


def test_a_large_paired_difference_is_resolvable_and_a_small_one_is_not():
    """The test has to be able to say *yes* as well as *no*."""
    labels = [True] * 6 + [False] * 24
    perfect = [100.0 - index for index in range(30)]
    reversed_ = [index for index in range(30)]
    big = lbvs.paired_difference(perfect, reversed_, labels, samples=1000, seed=3)
    assert big["resolvable"] is True and big["ci"][0] > 0.0
    assert big["difference"] == pytest.approx(1.0, abs=1e-9)
    # Moving one decoy above one active is a real but tiny change: measured and
    # unresolvable, and the gap is smaller than the difference the set can detect.
    nearly = list(perfect)
    nearly[5], nearly[6] = nearly[6], nearly[5]  # last active / first decoy
    small = lbvs.paired_difference(perfect, nearly, labels, samples=1000, seed=3)
    assert small["difference"] > 0.0, "perfect is the better ranking"
    assert small["resolvable"] is False
    assert small["mdd"] > 0.0
    assert small["difference"] < small["mdd"], (
        "one discordant pair is far below what 6 actives and 24 decoys can detect"
    )


def test_the_report_compares_methods_in_library_order_not_ranking_order():
    """The bug the first paired test had: two rankings are in different orders.

    Method B is scored by *reversing* the scores within each label class, so its
    ranking lists the molecules in a different order from A's.  If the comparison
    paired B's scores with A's label vector, the labels would be misaligned and the
    difference would be wrong; aligned by name it is exactly the two AUCs' gap.
    """
    actives = [named("N=C(N)c1ccccc1", "benzamidine"),
               named("N=C(N)c1ccc(O)cc1", "hydroxybenzamidine")]
    decoys = [named("c1ccccc1", "benzene"), named("c1ccncc1", "pyridine"),
              named("CCO", "ethanol")]
    report = lbvs.benchmark(actives, decoys, methods=("fingerprint", "overlay"),
                            conformers=1, bootstrap=10, controls=())
    names, scores, labels = report._aligned("fingerprint_morgan")
    assert names == [name for name, _ in report.labels], "library order, not rank order"
    assert labels == [flag for _, flag in report.labels]
    assert lbvs.auc(scores, labels) == pytest.approx(
        report.by_method()["fingerprint_morgan"].stats["auc"], abs=5e-5
    )
    for method in ("fingerprint_morgan", "overlay_esp"):
        assert len(report._aligned(method)[1]) == len(report.labels)
    with pytest.raises(ValueError, match="no method"):
        report._aligned("nonsense")
    comparison = report.compare("fingerprint_morgan", "overlay_esp", samples=50)
    assert comparison["method_a"] == "fingerprint_morgan"
    assert comparison["n"] == len(report.labels)
    assert "MDD" in report.paired_table(samples=20)
    assert "resolvable" in report.paired_table(samples=20)
    power = report.power(samples=20)
    assert power["n_actives"] == 2 and power["pairs"] == 1
    assert power["min_mdd"] <= power["max_mdd"]
    assert report.as_dict()["power"]["pairs"] == 1


def test_read_targets_and_the_per_target_benchmark(molecules, actives):
    """The multi-target entry point, exercised on a small synthetic mapping.

    Two "targets" are defined over the bundled library so the stratification is
    real: the machinery must report an AUC per target, pool the targets with a
    mean-over-targets estimate, and **record the reason** it skipped the ones it
    could not score rather than dropping them silently.
    """
    pool = [
        named("Nc1ccccn1", "2_aminopyridine"),
        named("Nc1ncccn1", "2_aminopyrimidine"),
        named("Nc1nccs1", "2_aminothiazole"),
        named("NCc1ccccn1", "2_aminomethylpyridine"),
        named("Nc1ccccc1O", "2_aminophenol"),
        named("NCc1ccccc1", "benzylamine"),
        named("c1ccc2ccccc2c1", "naphthalene"),
        named("OC(=O)CCC(=O)O", "succinic_acid"),
        named("c1ccncc1", "pyridine"),
        named("Cc1ccccc1", "toluene"),
        named("OCC(O)CO", "glycerol"),
        named("NCCc1ccccc1", "phenethylamine"),
    ]
    targets = {
        "trypsin": actives[:3],
        "other": actives[3:5],
        "one_active": actives[:1],
    }
    report = lbvs.benchmark_per_target(
        targets, pool, per_active=2, methods=("fingerprint", "overlay"),
        conformers=1, bootstrap=10, min_decoys=2, controls=(),
    )
    assert set(report.reports) <= {"trypsin", "other"}
    assert report.targets, "at least one target must be scorable"
    assert "one_active" in report.skipped
    assert "no reference" in report.skipped["one_active"]
    for name in report.targets:
        assert report.reports[name].n_actives >= 2
        assert report.reports[name].n_decoys >= 2
    table = report.per_target_table()
    assert "pooled" in table and "not benchmarked" in table
    assert set(report.methods) == {"fingerprint_morgan", "overlay_esp"}
    comparison = report.compare("fingerprint_morgan", "overlay_esp", samples=20)
    assert set(comparison) >= {"difference", "ci", "se", "mdd", "resolvable"}
    assert "stratified" in report.paired_table(samples=20)
    power = report.power(samples=20)
    assert power["n_targets"] == len(report.targets)
    assert power["n_actives"] == sum(r.n_actives for r in report.reports.values())
    assert report.as_dict()["targets"]
    # A mapping with nothing scorable is an error that names the reasons.
    with pytest.raises(ValueError, match="no target could be benchmarked"):
        lbvs.benchmark_per_target(
            {"lonely": actives[:1]}, pool, per_active=2, methods=("fingerprint",),
            conformers=1, bootstrap=5,
        )


def test_the_stratified_difference_pools_over_targets_not_molecules():
    """A target with many molecules must not outvote a target with few.

    Two strata: one with 10 actives and 10 decoys where A is perfect and B is
    reversed, and one with 2 actives and 2 decoys where the two are identical.  The
    mean-over-targets difference is 0.5 (half the targets differ), which is *not*
    what pooling the molecules would give.
    """
    strata = [
        {
            "scores_a": [10.0 - i for i in range(20)],
            "scores_b": [float(i) for i in range(20)],
            "labels": [True] * 10 + [False] * 10,
        },
        {
            "scores_a": [1.0, 0.5, 0.4, 0.3],
            "scores_b": [1.0, 0.5, 0.4, 0.3],
            "labels": [True, True, False, False],
        },
    ]
    pooled = lbvs.stratified_difference(strata, samples=200, seed=11)
    assert pooled["auc_a"] == pytest.approx(1.0, abs=0.02)
    assert pooled["auc_b"] == pytest.approx(0.5, abs=0.02)
    assert pooled["difference"] == pytest.approx(0.5, abs=0.02)
    assert pooled["n_strata"] == 2
    assert pooled["mdd"] > 0.0
    assert pooled["resolvable"] is True, "a one-target-perfect split must resolve"
    with pytest.raises(ValueError, match="no stratum"):
        lbvs.stratified_difference([{"scores_a": [], "scores_b": [], "labels": []}])


def test_read_targets_reads_the_shipped_multi_target_file(tmp_path):
    path = ROOT / "demo" / "libraries" / "actives.smi"
    if not path.exists():
        pytest.skip("the bundled multi-target actives file is missing")
    targets = lbvs.read_targets(path)
    assert set(targets) == {"trypsin", "eralpha", "hiv_protease", "egfr", "streptavidin"}
    assert len(targets["trypsin"]) == 5, "the published ring-amidine set"
    assert len(targets["eralpha"]) == 2
    names = [mol.GetProp("_Name") for mol in targets["trypsin"]]
    assert names == ["benzamidine", "benzamidine_methyl", "hydroxybenzamidine",
                     "fluorobenzamidine", "chloro_benzamidine"]
    # The two ligands whose structure file is not bundled are checked against their
    # published formula and the heavy-atom count the PDB file actually contains.
    from rdkit.Chem import rdMolDescriptors

    erlotinib = targets["egfr"][0]
    assert rdMolDescriptors.CalcMolFormula(erlotinib) == "C22H23N3O4"
    assert erlotinib.GetNumHeavyAtoms() == 29
    assert targets["streptavidin"][0].GetNumHeavyAtoms() == 16
    with pytest.raises(ValueError, match="expected 'SMILES name target'"):
        lbvs.read_targets(_write_tmp(tmp_path, "broken.smi", "CCO ethanol\n"))


def _write_tmp(tmp_path, name, text):
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    return target


def test_read_targets_validates_its_input(tmp_path):
    with pytest.raises(ValueError, match="not a parsable SMILES"):
        lbvs.read_targets(_write_tmp(tmp_path, "bad.smi", "not a molecule x y\n"))
    with pytest.raises(ValueError, match="holds no actives"):
        lbvs.read_targets(_write_tmp(tmp_path, "empty.smi", "# nothing\n"))


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


# ---------------------------------------------------------------------------
# The two shape engines, and the pre-filter that scales the 3-D one
# ---------------------------------------------------------------------------


def test_the_shape_engine_flag_selects_between_the_two_overlays(molecules, actives):
    """Both engines stay available and are named apart, so the delta is visible.

    ``overlay`` is the default (analytic Gaussian densities, an optimised pose,
    both terms reported per molecule); ``crude`` is the original single-pose grid
    overlay, kept so the two can be measured on the same data.
    """
    names = [mol.GetProp("_Name") for mol in molecules]
    overlay = lbvs.shape_scores(molecules, actives, conformers=1, names=names)
    crude = lbvs.shape_scores(
        molecules, actives, engine="crude", conformers=1, names=names
    )
    assert overlay.method == "overlay_esp"
    assert crude.method == "crude_esp"
    assert len(overlay) == len(crude) == len(molecules)
    assert set(overlay.names) == set(crude.names) == set(names)
    # The overlay reports its terms per molecule; the grid scorer does not have
    # them to report.
    assert overlay.details and overlay.terms()["n"] == len(molecules)
    assert overlay.details["benzamidine"]["shape"] > 0.0
    assert crude.details == {}
    assert any("rigid grid overlay" in note for note in crude.notes)
    # Shape-only variants, and the deprecated alias still pointing at the chosen
    # engine.
    assert lbvs.shape_scores(
        molecules, actives, conformers=1, names=names, electrostatic=False
    ).method == "overlay_shape"
    assert lbvs.shape_scores(
        molecules, actives, engine="crude", conformers=1, names=names,
        electrostatic=False,
    ).method == "crude_shape"
    with pytest.raises(ValueError, match="unknown shape engine"):
        lbvs.shape_scores(molecules, actives, engine="nonsense")


def test_the_benchmark_can_run_both_shape_engines_side_by_side(molecules, actives):
    """The comparison the default documents, on one labelled set, one run."""
    small_actives = actives[:2]
    decoys = [
        named("Nc1ccccn1", "2_aminopyridine"),
        named("Nc1ncccn1", "2_aminopyrimidine"),
        named("Nc1nccs1", "2_aminothiazole"),
        named("NCc1ccccn1", "2_aminomethylpyridine"),
    ]
    report = lbvs.benchmark(
        small_actives, decoys, methods=("overlay", "overlay_only", "crude", "crude_only"),
        conformers=1, bootstrap=10, controls=(),
    )
    methods = [result.method for result in report.results]
    assert methods == ["overlay_esp", "overlay_shape", "crude_esp", "crude_shape"]
    for result in report.results:
        assert 0.0 <= result.stats["auc"] <= 1.0
    assert report.by_method()["overlay_esp"].ranking.details
    # `shape` follows the engine flag; the default engine is the overlay.
    default = lbvs.benchmark(
        small_actives, decoys, methods=("shape",), conformers=1, bootstrap=5, controls=()
    )
    assert default.results[0].method == "overlay_esp"
    flagged = lbvs.benchmark(
        small_actives, decoys, methods=("shape",), shape_engine="crude",
        conformers=1, bootstrap=5, controls=(),
    )
    assert flagged.results[0].method == "crude_esp"


def test_the_prefilter_switches_between_the_2d_and_the_3d_filter(molecules, actives):
    """``usr`` delegates to :func:`odock.overlay.usr_prefilter` and says more."""
    fingerprint = lbvs.prefilter(molecules, actives, keep=0.5)
    assert fingerprint["method"] == "fingerprint"
    assert "overlay_seconds_kept" not in fingerprint
    usr = lbvs.prefilter(molecules, actives, keep=0.5, method="usr", conformers=1)
    assert usr["method"] == "usr"
    assert usr["n_library"] == len(molecules)
    assert usr["n_kept"] == fingerprint["n_kept"]
    assert usr["self_match_recall"] == 1.0, "the leak is reported, not hidden"
    assert usr["overlay_seconds_kept"] > 0.0
    assert usr["overlay_seconds_full_estimate"] > usr["overlay_seconds_kept"]
    with pytest.raises(ValueError, match="unknown pre-filter method"):
        lbvs.prefilter(molecules, actives, method="nonsense")


@pytest.mark.slow
def test_the_duplicate_decoy_band_still_cannot_separate_the_two_engines(molecules, actives):
    """The load-bearing negative result, pinned so it cannot quietly be claimed away.

    Measured on the bundled easy band (5 actives, 25 decoys, 2 conformers,
    leave-one-out): the *crude* grid overlay scores AUC 1.000 and the analytic
    overlay 0.920, and the two bootstrap intervals overlap — [1.00, 1.00] against
    [0.79, 1.00] — so this set cannot rank the engines, and the overlay does **not**
    improve the enrichment numbers.  `docs/LBVS.md` says so next to the table.
    """
    from odock import decoys as D

    if not POOL.exists():
        pytest.skip("the bundled decoy pool is missing")
    pool = []
    for line in POOL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        mol.SetProp("_Name", " ".join(fields[1:]))
        pool.append(mol)
    selection = D.match_decoys(actives, pool, per_active=5)
    chosen = {name for name in selection.names()}
    decoys = [mol for mol in pool if mol.GetProp("_Name") in chosen]
    assert len(decoys) == 25
    report = lbvs.benchmark(
        actives, decoys, methods=("overlay", "crude"), conformers=2, bootstrap=50,
        controls=(), decoy_quality=selection.quality(),
    )
    overlay = report.by_method()["overlay_esp"]
    crude = report.by_method()["crude_esp"]
    assert crude.stats["auc"] >= overlay.stats["auc"]
    assert overlay.stats["auc"] >= 0.85, "the overlay is not broken, it is not better"
    # The interval is the reason the default did not change on this evidence.
    assert overlay.intervals["auc"][1] >= crude.intervals["auc"][0]
    assert report.n_actives == 5 and report.n_decoys == 25


@pytest.mark.slow
def test_the_usr_prefilter_costs_active_recall_on_the_whole_library(molecules, actives):
    """The 3-D pre-filter's measured price, on the 142-molecule library.

    Keeping 5 % of the 142 molecules keeps 2 of the 5 actives (40 % recall) and cuts
    the estimated docking workload 436 s -> 25 s (94 %).  That is the honest
    characterisation of the speedup: a 3-D shape descriptor that ignores element
    identity cannot tell the amidines from the pool's other flat aromatics, and the
    2-D fingerprint filter keeps all five.  The self-match row is 5/5 for free.
    """
    if not POOL.exists():
        pytest.skip("the bundled decoy pool is missing")
    pool = []
    for line in POOL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        mol.SetProp("_Name", " ".join(fields[1:]))
        pool.append(mol)
    everything = list(molecules) + pool
    assert len(everything) == 142
    summary = lbvs.prefilter(everything, actives, keep=0.05, method="usr", conformers=4)
    assert summary["n_library"] == 142
    assert summary["n_kept"] == 8
    assert summary["actives_kept"] == 2
    assert summary["active_recall"] == pytest.approx(0.4)
    assert summary["self_match_actives_kept"] == 5
    assert summary["estimated_seconds_full"] == pytest.approx(436.0, abs=1.0)
    assert summary["estimated_seconds_kept"] == pytest.approx(25.0, abs=1.0)
    assert summary["workload_saved_fraction"] > 0.9
    # The 2-D filter on the same library keeps every active at the same cut.
    fingerprint = lbvs.prefilter(everything, actives, keep=0.05)
    assert fingerprint["actives_kept"] == 5
    assert fingerprint["active_recall"] == 1.0

# SPDX-License-Identifier: GPL-3.0-or-later
"""Scaffolds, series, R-groups and matched molecular pairs.

The affinities used here are the ones the bundled demo campaign actually
measured::

    odock screen -r demo/3ptb/receptor.pdbqt -i demo/library.smi \\
        --box demo/3ptb/box.json -o out/3ptb-screen -e 8 --seed 42

They are written down as data rather than docked in the test, because a docking
run belongs in the benchmark, not in a unit test of the series arithmetic — and
the deltas this file pins are the deltas that run produced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import scaffold as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "library.smi"

#: The Vina affinity of each demo-library molecule, measured on this tree
#: (3PTB, exhaustiveness 8, seed 42).  Two library members were removed by the
#: drug-likeness filters, so they have no affinity.
MEASURED_AFFINITIES = {
    "benzamidine": -5.900566019865151,
    "benzamidine_methyl": -5.516103332073266,
    "benzylamidine": -5.5173023743972776,
    "hydroxybenzamidine": -6.420006909246549,
    "fluorobenzamidine": -6.233533617319526,
    "chloro_benzamidine": -6.180044060400122,
    "caffeine": -5.328932098158455,
    "aspirin": -5.719129996066307,
    "ibuprofen": -5.693167921669242,
    "salicylic_acid": -6.303422737064128,
    "acetanilide": -4.97236632932402,
    "paracetamol": -5.984092803721076,
    "nicotinamide": -5.82293419527753,
    "isonicotinic_acid": -5.587472469369731,
    "warfarin": -7.338133634804257,
}

AMIDINES = [
    "benzamidine",
    "benzamidine_methyl",
    "benzylamidine",
    "hydroxybenzamidine",
    "fluorobenzamidine",
    "chloro_benzamidine",
]
RING_AMIDINES = [name for name in AMIDINES if name != "benzylamidine"]


def _read_library(path=LIBRARY):
    mols = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        assert mol is not None
        mol.SetProp("_Name", " ".join(fields[1:]))
        mols.append(mol)
    return mols


@pytest.fixture(scope="module")
def demo_library():
    if not LIBRARY.exists():
        pytest.skip("the bundled demo library is missing")
    mols = _read_library()
    assert len(mols) == 17
    return mols


@pytest.fixture(scope="module")
def demo_names(demo_library):
    return [mol.GetProp("_Name") for mol in demo_library]


@pytest.fixture(scope="module")
def amidines(demo_library):
    return [mol for mol in demo_library if mol.GetProp("_Name") in AMIDINES]


# ---------------------------------------------------------------------------
# Murcko scaffolds
# ---------------------------------------------------------------------------


def test_murcko_scaffold_of_a_named_molecule():
    """Hand check: benzamidine's amidine is a terminal substituent, so the
    scaffold is the phenyl ring alone; pyridine survives as itself."""
    assert S.murcko_scaffold(Chem.MolFromSmiles("N=C(N)c1ccccc1")) == "c1ccccc1"
    assert S.murcko_scaffold(Chem.MolFromSmiles("NC(=O)c1cccnc1")) == "c1ccncc1"


def test_an_acyclic_molecule_has_no_scaffold():
    assert S.murcko_scaffold(Chem.MolFromSmiles("CCCC")) == ""
    assert S.murcko_scaffold(Chem.MolFromSmiles("CC(=O)N")) == ""
    assert S.scaffold_of(Chem.MolFromSmiles("CCO")) is None
    assert S.scaffold_key(Chem.MolFromSmiles("CCO")) == ""


def test_generic_scaffold_makes_every_ring_the_same():
    """Benzene, pyridine and cyclohexane collapse to one skeleton; nothing is
    aromatic any more, and a fused three-ring system keeps all three rings."""
    assert S.generic_scaffold(Chem.MolFromSmiles("c1ccccc1")) == "C1CCCCC1"
    assert S.generic_scaffold(Chem.MolFromSmiles("c1ccncc1")) == "C1CCCCC1"
    assert S.generic_scaffold(Chem.MolFromSmiles("C1CCCCC1")) == "C1CCCCC1"
    assert S.murcko_scaffold(Chem.MolFromSmiles("c1ccncc1")) == "c1ccncc1"
    triphenylene = Chem.MolFromSmiles("c1ccc2c(c1)c1ccccc1c1ccccc21")
    text = S.generic_scaffold(triphenylene)
    assert not any(char.islower() for char in text), "no aromatic atom survives"
    frame = Chem.MolFromSmiles(text)
    # Triphenylene has three fused rings (an SSSR of 4); genericising it changes
    # neither the ring count nor the atom count.
    assert frame.GetRingInfo().NumRings() == triphenylene.GetRingInfo().NumRings() == 4
    assert frame.GetNumAtoms() == triphenylene.GetNumAtoms()


def test_scaffold_of_returns_a_molecule_that_can_be_fingerprinted(demo_library):
    frame = S.scaffold_of(demo_library[0])
    assert frame is not None and frame.GetNumAtoms() == 6
    assert Chem.MolToSmiles(frame) == "c1ccccc1"


def test_most_common_scaffold_of_the_demo_library(demo_library):
    """Eleven of the seventeen molecules are benzenes, so benzene wins; the
    *generic* skeleton is the same ring (they are all carbocyclic)."""
    assert S.most_common_scaffold(demo_library) == "c1ccccc1"
    assert S.most_common_scaffold(demo_library, generic=True) == "C1CCCCC1"
    assert S.most_common_scaffold([Chem.MolFromSmiles("CCCC")]) == ""


def test_maximum_common_substructure_of_the_amidines(amidines):
    """The five ring-attached amidines share ``N=C(N)c1ccccc1`` exactly.

    Including benzylamidine (``N=C(N)Cc1ccccc1``) the amidine is no longer
    common to all of them, and the MCS collapses to toluene — the honest answer
    to "what do these five molecules have in common", and the reason the MCS of a
    diverse library is useless as a core.
    """
    ring = [mol for mol in amidines if mol.GetProp("_Name") in RING_AMIDINES]
    assert len(ring) == 5
    assert S.maximum_common_substructure(ring) == "N=C(N)c1ccccc1"
    assert S.maximum_common_substructure(amidines) == "Cc1ccccc1"
    assert S.maximum_common_substructure([Chem.MolFromSmiles("CCO")]) == ""


def test_maximum_common_substructure_of_a_diverse_library_is_empty(demo_library):
    """Caffeine and warfarin share no ring system, so a 17-molecule library that
    contains both has no common core at all."""
    assert S.maximum_common_substructure(demo_library, timeout=5) == ""


def test_maximum_common_substructure_can_return_a_generic_skeleton():
    ring = [mol for mol in _read_library() if mol.GetProp("_Name") in RING_AMIDINES]
    assert S.maximum_common_substructure(ring, generic=True) == "CC(C)C1CCCCC1"


# ---------------------------------------------------------------------------
# Scaffold grouping
# ---------------------------------------------------------------------------


def test_scaffold_groups_count_the_demo_library(demo_library, demo_names):
    groups = S.scaffold_groups(
        demo_library, names=demo_names, affinities=MEASURED_AFFINITIES
    )
    assert len(groups) == 6, "six distinct Murcko scaffolds"
    assert [group.count for group in groups] == [11, 2, 1, 1, 1, 1]
    largest = groups[0]
    assert largest.key == "c1ccccc1"
    assert largest.count == 11
    assert largest.label == "c1ccccc1"
    # The representative is the heaviest member of the group (ibuprofen, 13 heavy
    # atoms), which is the molecule that shows the decoracted scaffold best.
    assert largest.representative_name == "ibuprofen"
    assert largest.members[0] == 0 and len(largest.members) == 11
    # The best affinity of the benzene group belongs to the 4-hydroxy analogue.
    assert largest.best_affinity == pytest.approx(-6.420006909246549)
    assert json.loads(json.dumps(largest.as_dict()))["count"] == 11


def test_scaffold_groups_put_the_acyclic_molecules_in_their_own_group():
    mols = [
        Chem.MolFromSmiles("CCCC"),
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("c1ccccc1"),
    ]
    groups = S.scaffold_groups(mols)
    assert [group.label for group in groups] == [S.NO_SCAFFOLD_LABEL, "c1ccccc1"]
    assert groups[0].count == 2 and groups[0].key == ""


def test_scaffold_groups_attribute_an_affinity_by_name(demo_library, demo_names):
    groups = S.scaffold_groups(
        demo_library,
        names=demo_names,
        affinities={"warfarin": -7.3, "benzamidine": -5.9},
    )
    warfarin_group = next(group for group in groups if "warfarin" in group.names)
    assert warfarin_group.best_affinity == pytest.approx(-7.3)
    assert warfarin_group.affinities[warfarin_group.names.index("warfarin")] == -7.3
    # A group of eleven molecules has one value and ten blanks: what the mapping
    # does not name is unknown, never zero.
    benzene_group = next(group for group in groups if group.count == 11)
    assert benzene_group.best_affinity == pytest.approx(-5.9)
    assert benzene_group.affinities.count(None) == 10
    assert benzene_group.affinities[benzene_group.names.index("benzamidine")] == -5.9


def test_scaffold_clusters_group_similar_scaffolds(demo_library, demo_names):
    """At 0.65 no two scaffolds are similar enough to merge (six families); at
    0.30 benzene and pyridine (Tanimoto 1/3) become one family of 13 molecules."""
    tight = S.scaffold_clusters(
        demo_library, names=demo_names, affinities=MEASURED_AFFINITIES
    )
    assert tight.n_scaffolds == 6
    assert tight.n_clusters == 6
    assert tight.largest.n_molecules == 11
    assert tight.largest.representative_scaffold == "c1ccccc1"
    assert tight.clusters[0].representative_name == "ibuprofen"
    assert tight.seconds >= 0.0

    loose = S.scaffold_clusters(demo_library, names=demo_names, cutoff=0.30)
    assert loose.n_scaffolds == 6
    assert loose.n_clusters == 5, "benzene and pyridine merge at 1/3 similarity"
    merged = next(
        cluster for cluster in loose.clusters if cluster.n_scaffolds == 2
    )
    assert sorted(merged.scaffolds) == ["c1ccccc1", "c1ccncc1"]
    assert merged.n_molecules == 13
    assert json.loads(json.dumps(loose.as_dict()))["n_series"] == 5


def test_scaffold_clusters_separate_the_acyclic_molecules():
    mols = [Chem.MolFromSmiles("CCCC"), Chem.MolFromSmiles("CCO")]
    clustering = S.scaffold_clusters(mols)
    assert clustering.n_clusters == 0 and clustering.n_scaffolds == 0
    assert clustering.n_molecules == 2
    assert any("acyclic" in note for note in clustering.notes)
    assert clustering.table() == "no scaffolds"


# ---------------------------------------------------------------------------
# R-group decomposition
# ---------------------------------------------------------------------------


def test_rgroups_of_the_benzamidine_series_with_an_explicit_core(amidines):
    """Core ``N=C(N)c1ccccc1``: five of the six amidines match.

    * benzamidine is H at every attachment point;
    * the 4-hydroxy and 4-fluoro analogues change the *same* label (they are both
      para-substituted);
    * the 2-chloro analogue changes a different one (it is ortho);
    * the N-methyl analogue changes a third (the amidine nitrogen);
    * benzylamidine does not match at all — its amidine is one carbon away.
    """
    decomposition = S.rgroups(amidines, core="N=C(N)c1ccccc1")
    assert decomposition.core == "N=C(N)c1ccccc1"
    assert decomposition.core_source == "given"
    assert decomposition.core_with_labels != decomposition.core
    assert len(decomposition.labels) == 3
    rows = {row.name: row for row in decomposition.rows}
    assert decomposition.n_matched == 5 and decomposition.n_molecules == 6
    assert not rows["benzylamidine"].matched
    assert "core is not a substructure" in rows["benzylamidine"].note
    assert set(rows["benzamidine"].rgroups.values()) == {"H"}

    def changed(row):
        return [label for label, value in row.rgroups.items() if value != "H"]

    assert changed(rows["hydroxybenzamidine"]) == changed(rows["fluorobenzamidine"])
    assert len(changed(rows["hydroxybenzamidine"])) == 1
    assert rows["hydroxybenzamidine"].rgroups[changed(rows["hydroxybenzamidine"])[0]] == "*O"
    assert rows["fluorobenzamidine"].rgroups[changed(rows["fluorobenzamidine"])[0]] == "*F"
    assert changed(rows["chloro_benzamidine"]) != changed(rows["hydroxybenzamidine"])
    assert rows["chloro_benzamidine"].rgroups[changed(rows["chloro_benzamidine"])[0]] == "*Cl"
    assert rows["benzamidine_methyl"].rgroups[changed(rows["benzamidine_methyl"])[0]] == "*C"
    assert "yes" in decomposition.table()


def test_rgroups_derives_the_core_from_the_most_common_scaffold(demo_library, demo_names):
    """With no --core the default is the library's dominant scaffold: benzene,
    which 12 of the 17 molecules contain (over an *acyclic* substituent)."""
    decomposition = S.rgroups(demo_library, names=demo_names)
    assert decomposition.core == "c1ccccc1"
    assert decomposition.core_source == "scaffold"
    assert len(decomposition.labels) == 3
    assert decomposition.n_matched == 12
    assert decomposition.match_rate == pytest.approx(12 / 17)
    rows = {row.name: row for row in decomposition.rows}
    assert rows["benzamidine"].matched and not rows["caffeine"].matched
    # The amidine is a substituent of the ring here, so it shows up as an R-group.
    assert "*C(=N)N" in rows["benzamidine"].rgroups.values()
    assert any("do not contain the core" in note for note in decomposition.notes)


def test_rgroups_reports_a_fused_ring_as_unmatched_rather_than_wrong(demo_library, demo_names):
    """Triphenylene contains a benzene ring, but its 'substituent' would close a
    ring, so the core-plus-acyclic-R-groups model cannot express it."""
    decomposition = S.rgroups(demo_library, names=demo_names)
    triphenylene = next(row for row in decomposition.rows if row.name == "triphenylene")
    assert not triphenylene.matched
    assert "closes a ring" in triphenylene.note


def test_rgroups_accepts_the_mcs_core(amidines):
    decomposition = S.rgroups(amidines, core="mcs")
    # The MCS includes benzylamidine, so it is toluene and only that molecule
    # matches — a worked example of why a single outlier drives an MCS core.
    assert decomposition.core == "Cc1ccccc1"
    assert decomposition.core_source == "mcs"
    assert decomposition.n_matched == 1
    ring = [mol for mol in amidines if mol.GetProp("_Name") in RING_AMIDINES]
    better = S.rgroups(ring, core="mcs")
    assert better.core == "N=C(N)c1ccccc1"
    assert better.n_matched == 5


def test_rgroups_rejects_an_unparsable_core(demo_library):
    with pytest.raises(ValueError, match="not a parsable SMILES"):
        S.rgroups(demo_library, core="this is not a core")


def test_rgroups_of_a_core_that_matches_nothing(demo_library, demo_names):
    decomposition = S.rgroups(demo_library, core="c1ccoc1", names=demo_names)
    assert decomposition.n_matched == 0
    assert all(not row.matched for row in decomposition.rows)
    assert decomposition.match_rate == 0.0


def test_rgroup_matrix_and_the_csv_export(amidines, tmp_path):
    decomposition = S.rgroups(amidines, core="N=C(N)c1ccccc1", affinities=MEASURED_AFFINITIES)
    header, rows = decomposition.matrix()
    assert header == ["name", "affinity", "matched"] + decomposition.labels
    assert len(rows) == 6
    assert rows[0][0] == "benzamidine"
    assert rows[0][1] == pytest.approx(-5.9006, abs=1e-3)
    assert rows[0][2] == "1"
    target = S.write_rgroup_csv(tmp_path / "out" / "rgroups.csv", decomposition)
    with target.open(encoding="utf-8", newline="") as handle:
        read = list(csv.reader(handle))
    assert read[0] == header
    assert read[1][0] == "benzamidine"
    assert len(read) == 7


def test_rgroup_xlsx_export(amidines, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    decomposition = S.rgroups(amidines, core="N=C(N)c1ccccc1")
    series = S.series_table(amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1")
    target = S.write_rgroup_xlsx(tmp_path / "rgroups.xlsx", decomposition, series=series)
    workbook = openpyxl.load_workbook(target)
    assert set(workbook.sheetnames) == {"R-groups", "series", "core"}
    header = [cell.value for cell in workbook["R-groups"][1]]
    assert header == ["name", "affinity", "matched"] + decomposition.labels
    assert workbook["core"]["B1"].value == "N=C(N)c1ccccc1"


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def test_series_table_deltas_against_the_reference(amidines):
    """The measured deltas of the benzamidine series (3PTB, seed 42).

    ``hydroxybenzamidine - benzamidine = -6.420006909246549 - (-5.900566019865151)
    = -0.519440889381398`` kcal/mol — the para H to OH change, and the largest
    change in this series.
    """
    series = S.series_table(amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1")
    assert series.core == "N=C(N)c1ccccc1"
    assert series.reference == "benzamidine"
    assert series.n_members == 5
    rows = {row.name: row for row in series.rows}
    assert rows["benzamidine"].delta == pytest.approx(0.0)
    assert rows["benzamidine"].is_reference
    assert rows["hydroxybenzamidine"].delta == pytest.approx(
        -6.420006909246549 + 5.900566019865151
    )
    assert rows["hydroxybenzamidine"].delta == pytest.approx(-0.519440889381398)
    assert rows["fluorobenzamidine"].delta == pytest.approx(-0.332967597454375)
    assert rows["chloro_benzamidine"].delta == pytest.approx(-0.279478040534971)
    assert rows["benzamidine_methyl"].delta == pytest.approx(+0.384462687791885)
    assert series.best.name == "hydroxybenzamidine"
    assert series.span == pytest.approx(
        -5.516103332073266 + 6.420006909246549
    )
    assert "delta" in series.table()
    assert json.loads(json.dumps(series.as_dict()))["n_members"] == 5


def test_series_reference_can_be_chosen_by_name_or_index(amidines):
    by_name = S.series_table(
        amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1", reference="fluorobenzamidine"
    )
    assert by_name.reference == "fluorobenzamidine"
    rows = {row.name: row for row in by_name.rows}
    assert rows["fluorobenzamidine"].delta == pytest.approx(0.0)
    assert rows["benzamidine"].delta == pytest.approx(+0.332967597454375)
    by_index = S.series_table(
        amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1", reference=4
    )
    assert by_index.reference == "fluorobenzamidine"
    # An unknown reference falls back to the first matched member and says so.
    fallback = S.series_table(
        amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1", reference="nothing"
    )
    assert fallback.reference == "benzamidine"
    assert any("not a matched member" in note for note in fallback.notes)


def test_series_without_an_affinity_has_no_delta(amidines):
    partial = {"benzamidine": -5.9, "hydroxybenzamidine": -6.42}
    series = S.series_table(amidines, partial, core="N=C(N)c1ccccc1")
    rows = {row.name: row for row in series.rows}
    assert rows["hydroxybenzamidine"].delta == pytest.approx(-0.52)
    assert rows["fluorobenzamidine"].affinity is None
    assert rows["fluorobenzamidine"].delta is None, "no made-up zero"
    assert series.best.name == "hydroxybenzamidine"


def test_series_of_a_library_where_nothing_matches(demo_library):
    series = S.series_table(demo_library, core="c1ccoc1")
    assert series.rows == []
    assert any("no molecule" in note for note in series.notes)
    assert series.table() == "empty series"


# ---------------------------------------------------------------------------
# Matched molecular pairs
# ---------------------------------------------------------------------------


def test_matched_pairs_of_the_benzamidine_series(amidines):
    """Five single-point substitutions among the five matched members.

    ``C(5,2) = 10`` pairs are considered and exactly five differ in one
    attachment point: every pair with benzamidine, plus hydroxy->fluoro (both
    para).  The largest delta is the para H to OH change at -0.519 kcal/mol, and
    **none** of them reaches the 1 kcal/mol docking noise this project measured —
    the honest result on this system.
    """
    table = S.matched_pairs(amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1")
    assert table.n_matched == 5
    assert table.n_pairs_considered == 10
    assert table.n_pairs == 5
    assert table.n_significant == 0
    assert table.noise == S.DOCKING_NOISE_KCAL == 1.0
    largest = table.largest
    assert largest.name_a == "benzamidine" and largest.name_b == "hydroxybenzamidine"
    assert largest.delta == pytest.approx(-0.519440889381398)
    assert largest.rgroup_a == "H" and largest.rgroup_b == "*O"
    assert not largest.significant
    assert "dA -0.519 kcal/mol" in largest.transformation()
    assert largest.heavy_atoms_a == 0 and largest.heavy_atoms_b == 1
    assert largest.added_heavy_atoms == 1
    pair = next(
        p for p in table.pairs if p.name_a == "benzamidine" and p.name_b == "fluorobenzamidine"
    )
    assert pair.label == largest.label, "both are para substitutions"
    assert pair.delta == pytest.approx(-0.332967597454375)
    assert pair.added_heavy_atoms == 1, "H -> F adds the fluorine, one heavy atom"
    assert json.loads(json.dumps(table.as_dict()))["n_pairs"] == 5


def test_every_matched_pair_differs_in_exactly_one_substituent(amidines):
    """The defining property, re-derived from the decomposition."""
    decomposition = S.rgroups(
        amidines, core="N=C(N)c1ccccc1", affinities=MEASURED_AFFINITIES
    )
    rows = {row.index: row for row in decomposition.rows if row.matched}
    table = S.matched_pairs(amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1")
    assert table.pairs
    for pair in table.pairs:
        a, b = rows[pair.index_a], rows[pair.index_b]
        differing = [
            label
            for label in decomposition.labels
            if a.rgroups.get(label) != b.rgroups.get(label)
        ]
        assert differing == [pair.label]
        assert pair.rgroup_a == a.rgroups[pair.label]
        assert pair.rgroup_b == b.rgroups[pair.label]
        assert pair.delta == pytest.approx(b.affinity - a.affinity)
        assert pair.index_a < pair.index_b


def test_matched_pairs_can_keep_pairs_without_an_affinity(amidines):
    partial = {"benzamidine": -5.9, "hydroxybenzamidine": -6.42, "chloro_benzamidine": -6.18}
    with_affinity = S.matched_pairs(amidines, partial, core="N=C(N)c1ccccc1")
    assert with_affinity.n_matched == 3
    everything = S.matched_pairs(
        amidines, partial, core="N=C(N)c1ccccc1", require_affinity=False
    )
    assert everything.n_matched == 5
    assert everything.n_pairs > with_affinity.n_pairs
    assert any(pair.delta is None for pair in everything.pairs)
    # min_delta keeps only the changes that clear a threshold.
    filtered = S.matched_pairs(
        amidines, MEASURED_AFFINITIES, core="N=C(N)c1ccccc1", min_delta=0.4
    )
    assert [p.name_b for p in filtered.pairs] == ["hydroxybenzamidine"]


def test_matched_pairs_of_a_diverse_library_are_reported_with_their_size(demo_library):
    """With the benzene core the pairs are still single-point *substitutions*,
    but not small ones: acetanilide -> warfarin swaps an acetamido group for the
    whole coumarin-bearing chain, and the table says how many heavy atoms that
    adds rather than presenting a -2.4 kcal/mol "substituent effect"."""
    table = S.matched_pairs(demo_library, MEASURED_AFFINITIES)
    assert table.core == "c1ccccc1"
    assert table.n_matched == 12
    assert table.n_pairs == 17
    biggest = table.largest
    assert biggest.name_b == "warfarin"
    # The acetamido group is four heavy atoms, the coumarin-bearing chain
    # seventeen: the change adds thirteen heavy atoms, which is what makes a
    # -2.4 kcal/mol delta a change of many properties rather than a substituent
    # effect.
    assert (biggest.heavy_atoms_a, biggest.heavy_atoms_b) == (4, 17)
    assert biggest.added_heavy_atoms == 13
    assert biggest.significant, "the delta clears the 1 kcal/mol noise floor"
    assert any("more than two heavy atoms" in note for note in table.notes)
    # A change at two attachment points at once is not a matched pair at all:
    # paracetamol and salicylic acid differ at all three of this core's points.
    names = {(pair.name_a, pair.name_b) for pair in table.pairs} | {
        (pair.name_b, pair.name_a) for pair in table.pairs
    }
    assert ("paracetamol", "salicylic_acid") not in names
    assert "dheavy" in table.table()


# ---------------------------------------------------------------------------
# Affinities from a screening run
# ---------------------------------------------------------------------------


def test_affinities_from_records_accepts_a_mapping_and_pairs():
    assert S.affinities_from_records({"a": -5.0}) == {"a": -5.0}
    assert S.affinities_from_records([("a", -5.0), ("b", None)]) == {"a": -5.0}
    assert S.affinities_from_records([("a", float("nan"))]) == {}


def test_affinities_from_records_keeps_the_best_of_several_receptors():
    """A series table describes the best evidence for a molecule, so the most
    negative affinity of a multi-receptor campaign wins."""
    records = [
        {"name": "benzamidine", "affinity": -5.9},
        {"name": "benzamidine", "affinity": -7.2},
        {"name": "other", "affinity": -6.0},
    ]
    class Row:
        def __init__(self, name, affinity):
            self.name = name
            self.affinity = affinity

    assert S.affinities_from_records(records) == {"benzamidine": -7.2, "other": -6.0}
    assert S.affinities_from_records([Row("a", -1.0), Row("a", -2.0)]) == {"a": -2.0}


def test_affinities_from_records_reads_a_screening_results_file(tmp_path):
    pytest.importorskip("odock.screen")
    from odock.screen import LigandRecord

    target = tmp_path / "results.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        for name, affinity in (("benzamidine", -5.9), ("warfarin", -7.3)):
            handle.write(
                json.dumps(
                    LigandRecord(receptor="r", ligand=name, name=name, affinity=affinity).as_dict()
                )
                + "\n"
            )
    assert S.affinities_from_records(target) == {"benzamidine": -5.9, "warfarin": -7.3}


# ---------------------------------------------------------------------------
# The report section
# ---------------------------------------------------------------------------


def test_report_section_states_the_measured_numbers(demo_library, demo_names):
    text = S.report_section(
        demo_library,
        affinities=MEASURED_AFFINITIES,
        names=demo_names,
        core="N=C(N)c1ccccc1",
    )
    assert "SCAFFOLDS" in text and "SERIES" in text
    assert "R-GROUPS" in text and "MATCHED PAIRS" in text
    assert "6 distinct Murcko scaffold(s)" in text
    assert "11" in text and "c1ccccc1" in text
    assert "-0.519" in text, "the worked matched pair is in the report"
    assert "hydroxybenzamidine" in text
    assert "docking noise this project measured" in text
    assert "a delta below that is not a structure-activity result" in text


def test_report_section_without_affinities_says_so(demo_library):
    text = S.report_section(demo_library)
    assert "MATCHED PAIRS" in text
    assert "no affinity was supplied" in text
    assert "LIGAND CHEMISTRY" in text


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(S, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        S.require_rdkit()
    with pytest.raises(ImportError):
        S.murcko_scaffold(Chem.MolFromSmiles("CCO"))

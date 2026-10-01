# SPDX-License-Identifier: GPL-3.0-or-later
"""Fingerprints, similarity metrics, analogue search, diversity and clustering.

Every expected number in here is either derived by hand from the definition (the
toy bit sets below) or measured on the bundled library and written down with the
arithmetic that produced it.  A test that only says "the value is stable" would
not be a test of the science.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

import odock  # noqa: E402  (imports the chem layer lazily)
from odock import ligandsim as L  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "libraries" / "library.smi"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_library():
    """The bundled 17-molecule screening library, named, in file order."""
    if not LIBRARY.exists():
        pytest.skip("the bundled demo library is missing")
    mols = []
    for line in LIBRARY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        assert mol is not None, fields[0]
        mol.SetProp("_Name", " ".join(fields[1:]))
        mols.append(mol)
    assert len(mols) == 17
    return mols


@pytest.fixture(scope="module")
def demo_fingerprints(demo_library):
    names = [mol.GetProp("_Name") for mol in demo_library]
    return L.fingerprint_set(
        demo_library, names=names, smiles=True, source="demo/libraries/library.smi"
    )


def toy(indices, n_bits=8, name=""):
    """A hand-built fingerprint, so a hand-computed answer can be asserted."""
    return L.Fingerprint(kind="morgan", n_bits=n_bits, indices=tuple(indices), name=name)


def by_name(fingerprints, name):
    return fingerprints[fingerprints.names.index(name)]


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def test_morgan_fingerprint_has_the_documented_shape():
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("_Name", "ethanol")
    fp = L.fingerprint(mol, kind="morgan", radius=2, n_bits=2048)
    assert isinstance(fp, L.Fingerprint)
    assert fp.kind == "morgan"
    assert fp.n_bits == 2048 and len(fp) == 2048
    assert fp.radius == 2
    assert fp.name == "ethanol"
    assert 5 < fp.n_on < 60, "a three-heavy-atom molecule sets a handful of bits"
    assert list(fp.indices) == sorted(fp.indices)
    dense = fp.as_numpy()
    assert dense.shape == (2048,) and dense.dtype == bool
    assert int(dense.sum()) == fp.n_on
    assert not fp.is_empty
    # Nothing in the public object is an RDKit type: it is JSON-serialisable.
    assert json.loads(json.dumps(fp.as_dict()))["n_on"] == fp.n_on


def test_every_fingerprint_kind_produces_a_usable_fingerprint():
    mol = Chem.MolFromSmiles("c1ccc(cc1)C(=N)N")
    lengths = {}
    for kind in ("morgan", "rdkit", "atom_pair", "torsion", "maccs"):
        fp = L.fingerprint(mol, kind=kind)
        assert fp.kind == kind and fp.n_on > 0
        lengths[kind] = fp.n_bits
    assert lengths["maccs"] == 167, "the public MACCS keys are 166 keys in 167 bits"
    assert lengths["morgan"] == 2048 and lengths["rdkit"] == 2048


def test_fingerprint_kind_aliases_and_unknown_kinds():
    mol = Chem.MolFromSmiles("CCO")
    assert L.fingerprint(mol, kind="ECFP4").kind == "morgan"
    assert L.fingerprint(mol, kind="path").kind == "rdkit"
    with pytest.raises(ValueError, match="unknown fingerprint kind"):
        L.fingerprint(mol, kind="nope")


def test_fingerprint_parameters_change_the_bit_pattern():
    mol = Chem.MolFromSmiles("c1ccc(cc1)C(=N)N")
    assert (
        L.fingerprint(mol, radius=2).indices != L.fingerprint(mol, radius=3).indices
    ), "a larger radius sees larger environments"
    assert L.fingerprint(mol, n_bits=1024).n_bits == 1024
    assert (
        L.fingerprint(mol, use_features=True).indices
        != L.fingerprint(mol, use_features=False).indices
    ), "the feature (FCFP) variant is a different fingerprint"


def test_an_atomless_molecule_fingerprints_empty_rather_than_None():
    blank = Chem.RWMol().GetMol()
    fp = L.fingerprint(blank)
    assert fp.is_empty and fp.n_on == 0 and len(fp) == 2048


def test_pharmacophore_fingerprint_needs_a_conformer():
    mol = Chem.MolFromSmiles("c1ccc(cc1)C(=N)N")
    with pytest.raises(ValueError, match="3-D conformer"):
        L.fingerprint(mol, kind="pharmacophore")
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol, randomSeed=42)
    fp = L.fingerprint(mol, kind="pharmacophore")
    assert fp.kind == "pharmacophore" and fp.n_on > 0


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(L, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        L.require_rdkit()
    with pytest.raises(ImportError):
        L.fingerprint(Chem.MolFromSmiles("CCO"))


# ---------------------------------------------------------------------------
# The similarity coefficients, by hand
# ---------------------------------------------------------------------------


def test_tanimoto_dice_and_tversky_match_the_hand_computation():
    """A = {1,2}, B = {2,3,4}: one shared, so Tanimoto is 1/(1+1+2) = 0.25."""
    a = toy((1, 2))
    b = toy((2, 3, 4))
    assert L.tanimoto(a, b) == pytest.approx(1 / 4)
    assert L.dice(a, b) == pytest.approx(2 / 5)
    assert L.tversky(a, b, alpha=1.0, beta=1.0) == pytest.approx(1 / 4)
    assert L.tversky(a, b, alpha=0.5, beta=0.5) == pytest.approx(1 / 2.5)
    assert L.tversky(a, b, alpha=0.5, beta=1.0) == pytest.approx(1 / 3.5)
    # The three named metrics are the documented special cases.
    assert L.similarity(a, b, metric="tanimoto") == L.tanimoto(a, b)
    assert L.similarity(a, b, metric="dice") == L.dice(a, b)
    assert L.similarity(a, b, metric="tversky", alpha=0.5, beta=0.5) == L.dice(a, b)
    # Identical and disjoint sets are the extremes.
    assert L.tanimoto(a, a) == 1.0
    assert L.tanimoto(a, toy((5, 6))) == 0.0
    # Two empty fingerprints have no union: 0.0, never a perfect 1.0.
    assert L.tanimoto(toy(()), toy(())) == 0.0


def test_tanimoto_of_benzamidine_and_its_hydroxy_analogue_is_four_sevenths(demo_fingerprints):
    """Hand check with the real bit sets, measured on this tree.

    Morgan, radius 2, 2048 bits:

    * benzamidine ``N=C(N)c1ccccc1`` sets **16** bits;
    * 4-hydroxybenzamidine ``N=C(N)c1ccc(O)cc1`` sets **17**;
    * they share **12** (the amidine environments and the unsubstituted half of
      the ring);
    * so the union is ``16 + 17 - 12 = 21`` and Tanimoto is ``12 / 21 = 4/7``.
    """
    a = by_name(demo_fingerprints, "benzamidine")
    b = by_name(demo_fingerprints, "hydroxybenzamidine")
    assert (a.n_on, b.n_on) == (16, 17)
    shared = len(set(a.indices) & set(b.indices))
    assert shared == 12
    assert L.tanimoto(a, b) == pytest.approx(4 / 7)
    assert L.dice(a, b) == pytest.approx(2 * 12 / (16 + 17))


def test_similarity_refuses_incomparable_fingerprints():
    with pytest.raises(ValueError, match="cannot compare"):
        L.tanimoto(toy((1,)), L.Fingerprint(kind="maccs", n_bits=167, indices=(1,)))
    with pytest.raises(TypeError, match="Fingerprint"):
        L.tanimoto(toy((1,)), {1})
    with pytest.raises(ValueError, match="unknown similarity metric"):
        L.similarity(toy((1,)), toy((2,)), metric="cosine")


# ---------------------------------------------------------------------------
# Fingerprint sets and the similarity matrix
# ---------------------------------------------------------------------------


def test_fingerprint_set_keeps_order_names_and_smiles(demo_library, demo_fingerprints):
    assert len(demo_fingerprints) == 17
    assert demo_fingerprints.name_of(0) == "benzamidine"
    assert demo_fingerprints.smiles_of(0) == "N=C(N)c1ccccc1"
    assert demo_fingerprints.kind == "morgan"
    # A subset keeps the order it is asked for, not the library order.
    picked = demo_fingerprints.subset([3, 0])
    assert picked.names == ["hydroxybenzamidine", "benzamidine"]
    assert picked.kind == "morgan" and picked.n_bits == 2048
    assert len(picked) == 2


def test_fingerprint_set_accepts_smiles_and_names_unnamed_records():
    fset = L.fingerprint_set(["CCO", "c1ccccc1"])
    assert fset.names == ["ligand_1", "ligand_2"]
    named = L.fingerprints_from_smiles(["CCO", "CCN"], names=["ethanol", "ethylamine"])
    assert named.names == ["ethanol", "ethylamine"]
    with pytest.raises(ValueError, match="not a parsable SMILES"):
        L.fingerprints_from_smiles(["not a molecule"])


def test_similarity_matrix_agrees_with_the_pairwise_metric():
    fps = [toy((0, 1, 2)), toy((0, 1, 2)), toy((1, 2, 3)), toy((3, 4, 5))]
    matrix = L.similarity_matrix(fps)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    # Hand computed: |A|=3, |B|=3, |A n B| = 2 -> 2/(3+3-2) = 0.5.
    assert matrix[0, 2] == pytest.approx(0.5) == pytest.approx(L.tanimoto(fps[0], fps[2]))
    # C = {3,4,5} against A: a single shared bit, 1/(3+3-1) = 0.2.
    assert matrix[3, 2] == pytest.approx(L.tanimoto(fps[3], fps[2])) == pytest.approx(0.2)
    assert matrix[0, 1] == 1.0
    for metric in ("dice", "tversky"):
        other = L.similarity_matrix(fps, metric=metric)
        assert other[0, 2] == pytest.approx(
            L.similarity(fps[0], fps[2], metric=metric)
        )


def test_similarity_matrix_matches_every_pair_of_the_real_library(demo_fingerprints):
    matrix = L.similarity_matrix(demo_fingerprints)
    assert matrix.shape == (17, 17)
    for i in (0, 5, 16):
        for j in (0, 1, 8, 16):
            assert matrix[i, j] == pytest.approx(
                L.tanimoto(demo_fingerprints[i], demo_fingerprints[j]), abs=1e-9
            )
    off = matrix[~np.eye(17, dtype=bool)]
    assert 0.0 <= off.min() and off.max() < 1.0


def test_similarity_matrix_rejects_mixed_kinds():
    fps = [toy((1,)), L.Fingerprint(kind="maccs", n_bits=167, indices=(1,))]
    with pytest.raises(ValueError, match="mixed fingerprint kinds"):
        L.similarity_matrix(fps)


def test_similarity_matrix_is_independent_of_the_block_size(demo_fingerprints):
    """The matrix is computed in blocks to bound memory, so the blocking must not
    be visible in the result: a 2-row block and a 17-row block agree exactly."""
    whole = L.similarity_matrix(demo_fingerprints)
    for chunk in (1, 2, 5, 17, 64):
        blocked = L.similarity_matrix(demo_fingerprints, chunk=chunk)
        assert np.allclose(whole, blocked), f"chunk={chunk} differs"


def test_similarity_matrix_of_an_empty_library_is_empty():
    assert L.similarity_matrix([]).shape == (0, 0)


# ---------------------------------------------------------------------------
# Analogue search
# ---------------------------------------------------------------------------


def test_find_analogues_ranks_the_benzamidine_series(demo_library, demo_fingerprints):
    hits = L.find_analogues(
        demo_library[0], demo_fingerprints, cutoff=0.5, name="benzamidine"
    )
    assert hits.query_name == "benzamidine"
    assert hits.metric == "tanimoto" and hits.kind == "morgan"
    assert hits.n_library == 17 and hits.n_pairs == 17
    # At 0.5: benzamidine (1.0), the 4-hydroxy (0.571) and 4-fluoro (0.545)
    # analogues.  Nothing else reaches 0.5, so the cut-off does real work here.
    assert [hit.name for hit in hits.hits] == [
        "benzamidine",
        "hydroxybenzamidine",
        "fluorobenzamidine",
    ]
    assert [hit.rank for hit in hits.hits] == [1, 2, 3]
    assert hits.best.similarity == 1.0
    assert hits.hits[1].similarity == pytest.approx(4 / 7)
    assert hits.hit_rate() == pytest.approx(3 / 17)
    assert "hydroxybenzamidine" in hits.table()
    assert json.loads(json.dumps(hits.as_dict()))["n_hits"] == 3


def test_find_analogues_cutoff_and_top(demo_library, demo_fingerprints):
    # Every analogue of benzamidine sits in 0.44..1.0; a 1.0 cut-off leaves only
    # the molecule itself.
    only_self = L.find_analogues(demo_library[0], demo_fingerprints, cutoff=1.0)
    assert [hit.name for hit in only_self.hits] == ["benzamidine"]
    # The 0.44 analogue (the ortho-chloro) enters at 0.44 and not at 0.45.
    assert "chloro_benzamidine" not in [
        hit.name for hit in L.find_analogues(demo_library[0], demo_fingerprints, cutoff=0.45).hits
    ]
    assert "chloro_benzamidine" in [
        hit.name for hit in L.find_analogues(demo_library[0], demo_fingerprints, cutoff=0.44).hits
    ]
    topped = L.find_analogues(demo_library[0], demo_fingerprints, cutoff=0.3, top=2)
    assert topped.n_hits == 2 and topped.hits[0].rank == 1
    # Ties in similarity are broken by library order, so the list is stable.
    tied = L.find_analogues(demo_library[0], demo_fingerprints, cutoff=0.45)
    equal = [
        hit.name
        for hit in tied.hits
        if hit.similarity == pytest.approx(tied.hits[3].similarity)
    ]
    assert equal == ["benzamidine_methyl", "benzylamidine"]


def test_find_analogues_accepts_a_smiles_query_and_a_metric(demo_library, demo_fingerprints):
    from_smiles = L.find_analogues(
        "N=C(N)c1ccccc1", demo_fingerprints, cutoff=0.5, name="benzamidine"
    )
    from_mol = L.find_analogues(demo_library[0], demo_fingerprints, cutoff=0.5)
    assert [(h.name, round(h.similarity, 9)) for h in from_smiles.hits] == [
        (h.name, round(h.similarity, 9)) for h in from_mol.hits
    ]
    # Dice weighs the shared bits twice, so it is always >= Tanimoto for the same
    # pair; the ranking here happens to survive the change of coefficient.
    diced = L.find_analogues(demo_library[0], demo_fingerprints, cutoff=0.6, metric="dice")
    assert diced.metric == "dice"
    assert all(
        hit.similarity >= L.tanimoto(demo_fingerprints[0], demo_fingerprints[hit.index])
        for hit in diced.hits
    )
    with pytest.raises(ValueError, match="not a parsable SMILES"):
        L.find_analogues("not a molecule", demo_fingerprints)


def test_find_analogues_fingerprints_a_bare_molecule_list(demo_library):
    hits = L.find_analogues(demo_library[0], demo_library, cutoff=0.5)
    assert hits.n_library == 17
    assert [hit.name for hit in hits.hits] == [
        "benzamidine",
        "hydroxybenzamidine",
        "fluorobenzamidine",
    ]
    assert hits.hits[1].smiles == "N=C(N)c1ccc(O)cc1", "the library SMILES comes back"


def test_find_analogues_reports_an_empty_hit_list_readably(demo_library, demo_fingerprints):
    hits = L.find_analogues("c1ccncc1", demo_fingerprints, cutoff=0.99)
    assert hits.n_hits == 0 and hits.best is None
    assert "no analogue" in hits.table()


def test_find_analogues_can_drop_the_query_itself(demo_fingerprints):
    """`include_self=False` is what a "find me the others" call wants: the query
    is in the library, and reporting it at 1.000 is not an analogue."""
    hits = L.find_analogues(
        "N=C(N)c1ccccc1", demo_fingerprints, cutoff=0.0, include_self=False
    )
    assert hits.n_library == 17
    assert hits.n_hits == 16
    assert "benzamidine" not in [hit.name for hit in hits.hits]
    assert hits.hits[0].similarity == pytest.approx(4 / 7)


# ---------------------------------------------------------------------------
# Multi-conformer (3-D) similarity
# ---------------------------------------------------------------------------


def test_conformer_fingerprints_are_conformer_dependent():
    """A flexible molecule's conformers give different pharmacophore bits."""
    flexible = Chem.MolFromSmiles("c1ccc(cc1)CCCCCC1CCCCC1")
    group = L.conformer_fingerprints(flexible, n_confs=6, seed=7)
    assert len(group) == 6
    assert all(fp.kind == "pharmacophore" for fp in group)
    assert len({fp.indices for fp in group}) > 1, "6 conformers of a chain differ"
    # The 2-D kinds would all be identical: that is why they are not offered here.
    assert len({L.fingerprint(flexible).indices}) == 1


def test_best_over_conformers_is_the_maximum_pairwise_similarity():
    a = [toy((1, 2)), toy((1, 2, 3, 4))]
    b = [toy((9,)), toy((1, 2, 9))]
    expected = max(L.tanimoto(x, y) for x in a for y in b)
    assert L.best_over_conformers(a, b) == pytest.approx(expected)
    assert L.best_over_conformers([], b) == 0.0


def test_find_analogues_3d_reports_its_cost(demo_library):
    hits = L.find_analogues_3d(
        demo_library[0],
        demo_library[:4],
        cutoff=0.0,
        n_query_confs=2,
        n_library_confs=2,
        seed=42,
    )
    assert hits.kind == "pharmacophore" and hits.conformers == 2
    assert hits.n_library == 4
    # 2 query conformers x 2 library conformers x 4 molecules = 16 comparisons.
    assert hits.n_pairs == 2 * 2 * 4
    assert hits.n_hits == 4, "a 0.0 cut-off keeps everything"
    assert hits.hits[0].index == 0, "the query resembles itself most"
    assert hits.hits[0].similarity == pytest.approx(1.0)
    assert hits.seconds >= 0.0
    assert json.loads(json.dumps(hits.as_dict()))["n_pairs"] == 16


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------

#: Toy fingerprints with similarities computed by hand:
#:
#:   a = {0,1,2}  b = {0,1,2}  c = {1,2,3}  d = {3,4,5}
#:   sim(a,b)=1, sim(a,c)=2/4=0.5, sim(b,c)=0.5, sim(c,d)=1/5=0.2, sim(a,d)=0
TOY_DIVERSITY = [
    L.Fingerprint(kind="morgan", n_bits=6, indices=(0, 1, 2), name="a"),
    L.Fingerprint(kind="morgan", n_bits=6, indices=(0, 1, 2), name="b"),
    L.Fingerprint(kind="morgan", n_bits=6, indices=(1, 2, 3), name="c"),
    L.Fingerprint(kind="morgan", n_bits=6, indices=(3, 4, 5), name="d"),
]


def toy_set(fingerprints=TOY_DIVERSITY):
    return L.FingerprintSet(
        names=[fp.name for fp in fingerprints],
        fingerprints=list(fingerprints),
        kind="morgan",
        n_bits=6,
    )


def test_maxmin_picking_takes_the_farthest_point_each_time():
    """Hand computed on TOY_DIVERSITY.

    Start at a.  The nearest-neighbour similarity of every other molecule to
    ``{a}`` is b=1.0, c=0.5, d=0.0, so the *least* similar one, d, is picked
    second.  Against ``{a, d}`` the nearest distances are b=1.0, c=0.5, so c is
    third: ``[a, d, c]``.
    """
    assert L.maxmin_pick(TOY_DIVERSITY, 3) == [0, 3, 2]
    selection = L.diversity_subset(toy_set(), 3)
    assert selection.indices == [0, 3, 2]
    assert selection.min_similarity == pytest.approx([0.0, 0.0, 0.5])
    assert selection.worst_pairwise_similarity == pytest.approx(0.5)
    assert selection.n_library == 4 and selection.n_requested == 3
    assert selection.fraction == pytest.approx(0.75)
    assert selection.names == ["a", "d", "c"]
    assert json.loads(json.dumps(selection.as_dict()))["n_selected"] == 3


def test_maxmin_is_deterministic_and_start_dependent():
    assert L.maxmin_pick(TOY_DIVERSITY, 3) == L.maxmin_pick(TOY_DIVERSITY, 3)
    # Starting at c: the farthest point is d (0.2), then a and b tie at 0.5 and
    # the lowest index wins, so the third pick is a.
    assert L.maxmin_pick(TOY_DIVERSITY, 3, start=2) == [2, 3, 0]
    assert L.maxmin_pick(TOY_DIVERSITY, 0) == []
    assert len(L.maxmin_pick(TOY_DIVERSITY, 99)) == 4, "asking for more than exists"
    with pytest.raises(IndexError):
        L.maxmin_pick(TOY_DIVERSITY, 2, start=9)
    with pytest.raises(ValueError, match="unknown start"):
        L.maxmin_pick(TOY_DIVERSITY, 2, start="middle")
    centroid = L.maxmin_pick(TOY_DIVERSITY, 3, start="centroid")
    assert centroid[0] == 0, "a and b are the most typical molecules (tie -> index 0)"


def test_sphere_exclusion_keeps_the_first_molecule_of_each_sphere():
    """From a: b is at 1.0, c at 0.5 and d at 0.0.

    The acceptance test is ``similarity < cutoff``, so the *boundary* matters: at
    exactly 0.5 the half-similar c is rejected and d is the only molecule added;
    at 0.6 both c and d are below the cut-off and both are kept.
    """
    assert L.sphere_exclusion_pick(TOY_DIVERSITY, cutoff=0.5) == [0, 3]
    assert L.sphere_exclusion_pick(TOY_DIVERSITY, cutoff=0.6) == [0, 2, 3]
    assert L.sphere_exclusion_pick(TOY_DIVERSITY, cutoff=0.2) == [0, 3]
    assert L.sphere_exclusion_pick(TOY_DIVERSITY, cutoff=0.5, n=1) == [0]
    selection = L.diversity_subset(toy_set(), method="sphere", cutoff=0.5)
    assert selection.method == "sphere" and selection.cutoff == 0.5
    assert selection.indices == [0, 3]
    assert selection.min_similarity == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError, match="needs a cutoff"):
        L.diversity_subset(toy_set(), method="sphere")
    with pytest.raises(ValueError, match="unknown diversity method"):
        L.diversity_subset(toy_set(), 2, method="random")


def test_diversity_selection_on_the_demo_library_and_its_scaffold_coverage(
    demo_library, demo_fingerprints
):
    """Measured: 6 MaxMin picks cover 4 of the library's 6 Murcko scaffolds.

    The picks are benzamidine, caffeine, paracetamol, benzoquinone, ibuprofen and
    warfarin; their scaffolds are benzene (benzamidine, paracetamol, ibuprofen),
    the purine of caffeine, the quinone of benzoquinone and the coumarin of
    warfarin — 4 distinct scaffolds out of the library's 6 (benzene, pyridine,
    caffeine, warfarin, triphenylene, benzoquinone) for 35 % of the molecules.
    """
    selection = L.diversity_subset(demo_fingerprints, 6)
    assert selection.indices == [0, 6, 11, 16, 8, 14]
    assert selection.names == [
        "benzamidine",
        "caffeine",
        "paracetamol",
        "benzoquinone",
        "ibuprofen",
        "warfarin",
    ]
    # The nearest-pick similarity of each new molecule: caffeine is almost
    # unrelated to benzamidine (0.0513), the last pick (warfarin) is the closest
    # to anything already chosen at 0.2157 — that is the subset's tightest pair.
    assert selection.min_similarity == pytest.approx(
        [0.0, 0.051282, 0.125, 0.137931, 0.184211, 0.215686], abs=1e-6
    )
    assert selection.worst_pairwise_similarity == pytest.approx(0.215686, abs=1e-6)
    assert selection.n_pairs == 17 + 15 + 14 + 13 + 12 + 11, "one scan per pick"
    coverage = L.scaffold_coverage(selection, demo_library)
    assert coverage["n_library_scaffolds"] == 6
    assert coverage["n_subset_scaffolds"] == 4
    assert coverage["coverage"] == pytest.approx(4 / 6)
    assert coverage["subset_fraction"] == pytest.approx(6 / 17)
    # A subset that is *not* diverse must cover no more: the first six molecules
    # are the six amidines, i.e. ONE scaffold (benzene) — 17 % of scaffold space
    # for the same 35 % of the molecules.
    first_six = L.scaffold_coverage(list(range(6)), demo_library)
    assert first_six["n_subset_scaffolds"] == 1
    assert first_six["coverage"] == pytest.approx(1 / 6)
    assert first_six["coverage"] < coverage["coverage"]


def test_scaffold_coverage_refuses_a_set_without_smiles(demo_library):
    bare = L.fingerprint_set(demo_library, smiles=False)
    with pytest.raises(ValueError, match="no SMILES"):
        L.scaffold_coverage([0, 1], bare)
    # The molecules themselves work, and the scaffold space is the library's.
    coverage = L.scaffold_coverage([0, 1], demo_library)
    assert coverage["n_library_scaffolds"] == 6
    assert coverage["n_subset_scaffolds"] == 1
    assert L.scaffold_coverage([], demo_library)["coverage"] == 0.0


# ---------------------------------------------------------------------------
# Butina clustering
# ---------------------------------------------------------------------------


def test_butina_clusters_the_toy_set_by_hand():
    """At 0.6 only a and b see each other, so a+b is a cluster, c and d singletons.

    At 0.45 both a and b have two unassigned neighbours (c is at 0.5) and a wins
    the tie by index, so the cluster is {a, b, c} and d is a singleton.
    """
    clustering = L.butina_cluster(TOY_DIVERSITY, cutoff=0.6)
    assert [sorted(cluster.members) for cluster in clustering.clusters] == [[0, 1], [2], [3]]
    assert clustering.n_clusters == 3 and clustering.n_singletons == 2
    assert clustering.n_molecules == 4
    loose = L.butina_cluster(TOY_DIVERSITY, cutoff=0.45)
    assert [sorted(cluster.members) for cluster in loose.clusters] == [[0, 1, 2], [3]]
    assert loose.n_singletons == 1


def test_butina_representative_is_the_medoid():
    clustered = L.butina_cluster(toy_set(), cutoff=0.45)
    biggest = clustered.clusters[0]
    assert biggest.size == 3
    # Members are a, b, c, so the mean similarity *including the self-term of 1*
    # is (1 + 1 + 0.5)/3 = 0.8333 for both a and b, and (0.5 + 0.5 + 1)/3 for c.
    # a and b tie and the lowest index wins.
    assert biggest.representative == 0
    assert biggest.representative_name == "a"
    assert biggest.mean_similarity == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_butina_agrees_with_rdkits_implementation(demo_library, demo_fingerprints):
    """The same partition as RDKit's ``Butina.ClusterData``, at three cut-offs.

    Using RDKit's own implementation as the reference is the only way to know that
    "Butina-style" means Butina's algorithm and not a plausible-looking
    alternative; the two agree on the demo library at 0.50, 0.45 and 0.40.
    """
    Butina = pytest.importorskip("rdkit.ML.Cluster.Butina")
    matrix = L.similarity_matrix(demo_fingerprints)
    n = len(demo_library)
    distances = [
        1.0 - matrix[i, j] for i in range(1, n) for j in range(i)
    ]
    for cutoff in (0.50, 0.45, 0.40):
        reference = {
            frozenset(cluster)
            for cluster in Butina.ClusterData(distances, n, 1.0 - cutoff, isDistData=True)
        }
        mine = {
            frozenset(cluster.members)
            for cluster in L.butina_cluster(demo_fingerprints, cutoff=cutoff).clusters
        }
        assert mine == reference, f"partitions differ at {cutoff}"


def test_butina_finds_the_benzamidine_series_in_the_demo_library(demo_fingerprints):
    """At 0.45 the five ring-amidines cluster together and acetanilide joins
    paracetamol; every other molecule is its own cluster."""
    clustering = L.butina_cluster(demo_fingerprints, cutoff=0.45)
    assert clustering.n_clusters == 12
    assert clustering.n_molecules == 17
    assert sum(cluster.size for cluster in clustering.clusters) == 17
    assert [cluster.members for cluster in clustering.clusters[:2]] == [
        (0, 1, 2, 3, 4),
        (10, 11),
    ]
    assert clustering.largest.size == 5
    assert clustering.largest.representative_name == "benzamidine"
    assert clustering.table().count("\n") >= 5
    assert json.loads(json.dumps(clustering.as_dict()))["n_singletons"] == 10


def test_butina_of_an_empty_library():
    clustering = L.butina_cluster([])
    assert clustering.n_clusters == 0 and clustering.n_molecules == 0
    assert clustering.largest is None and clustering.table() == "no clusters"


def test_a_distance_cutoff_is_a_similarity_threshold():
    """Every cluster member is within ``1 - cutoff`` of its seed, by construction."""
    clustering = L.butina_cluster(TOY_DIVERSITY, cutoff=0.45)
    for cluster in clustering.clusters:
        seed = cluster.members[0]
        for member in cluster.members:
            assert L.tanimoto(TOY_DIVERSITY[seed], TOY_DIVERSITY[member]) >= 0.45 - 1e-12


def test_clustering_reports_its_cost(demo_fingerprints):
    clustering = L.butina_cluster(demo_fingerprints, cutoff=0.7)
    assert clustering.n_pairs == 17 * 17
    assert clustering.seconds >= 0.0
    assert math.isfinite(clustering.seconds)

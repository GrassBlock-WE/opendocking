# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragments: screening by efficiency, growing, linking — and the two failures.

The tests here pin the measured behaviour of `docs/FRAGMENTS.md`, and **the failures are
pinned as failures**: on 3PTB a fragment does not land where the ligand's substructure
sits (20.61 Å for the amidine after the formal-charge correction, 6.87 Å before it), and
the linker enumerator cannot span two fragments whose *placements* are 4.19 Å apart when
the true geometry wants a direct bond.  A test that let either of those pass as a success
would be worse than no test, so `test_the_fragment_placement_failure_is_pinned` asserts
the RMSD is **large**.

The one positive result is also pinned: growing from a three-heavy-atom amidine fragment,
ranked by incremental ligand efficiency, puts benzamidine — the actual 3PTB ligand —
first.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

from odock import fragments as F  # noqa: E402
from odock import pocket_score as PS  # noqa: E402
from odock import protonation as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SYS = ROOT / "demo" / "systems" / "3ptb"
BENZAMIDINE = "N=C(N)c1ccccc1"
#: The amidine–phenyl bond of benzamidine, which is the cut the validation uses.
AMIDINE_PHENYL_CUT = [(1, 3)]


def mol(smiles: str, name: str = ""):
    out = Chem.MolFromSmiles(smiles)
    assert out is not None, smiles
    if name:
        out.SetProp("_Name", name)
    return out


@lru_cache(maxsize=1)
def three_ptb():
    """The 3PTB pocket field, built once: the field build is ~1.2 s per call."""
    box = json.loads((SYS / "box.json").read_text(encoding="utf-8"))
    atoms = PS.read_pdbqt_atoms(SYS / "receptor.pdbqt")
    pocket = PS.PocketField.build(atoms, center=box["center"], size=box["size"])
    return pocket


has_demo = pytest.mark.skipif(
    not (SYS / "receptor.pdbqt").exists() or not (SYS / "ligand.pdbqt").exists(),
    reason="the bundled 3PTB demo is not present",
)


# ---------------------------------------------------------------------------
# The fragment window and the generator
# ---------------------------------------------------------------------------


def test_the_heavy_atom_floor_keeps_a_three_atom_binder_out():
    """The floor is the whole point: a 3-atom binder must not top an efficiency table."""
    library = [mol("N=C(N)", "amidine"),           # 3 heavy atoms
               mol("c1ccccc1", "benzene"),          # 6
               mol("N=C(N)c1ccccc1", "benzamidine")]  # 9
    # At the default window only benzamidine qualifies: the 3-atom binder and the
    # 6-atom ring are both below the floor.
    assert [F.heavy_atom_count(item) for item in F.fragment_library(library)] == [9]
    kept = F.fragment_library(library, min_heavy=6, max_heavy=9)
    assert [F.heavy_atom_count(item) for item in kept] == [6, 9]
    # The ceiling keeps leads out, and a molecule outside either end is dropped, never
    # quietly included.
    assert F.fragment_library(library, min_heavy=3, max_heavy=6)[0].GetNumAtoms() == 3
    assert F.heavy_atom_count(mol("c1ccccc1")) == 6


def test_brics_cleaving_produces_fragments_in_the_window_and_deduplicates():
    """A fragment set grown from what is in hand, not invented."""
    pool = [mol("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
            mol("CC(=O)Oc1ccccc1C(=O)O", "aspirin_again")]
    pieces = F.cleave_fragments(pool, min_heavy=1, max_heavy=20)
    assert pieces, "BRICS must produce something for aspirin"
    assert len({Chem.MolToSmiles(piece) for piece in pieces}) == len(pieces)
    assert all(1 <= F.heavy_atom_count(piece) <= 20 for piece in pieces)
    # Every piece records where it came from, so a table can be audited.
    assert all(piece.HasProp("_Parent") for piece in pieces)


def test_splitting_the_ligand_caps_each_cut_with_a_real_hydrogen():
    """Formamidine and benzene — the fragments *are* the molecules they are called.

    The cap matters: cutting without filling the valence leaves a fragment that RDKit
    refuses to sanitise, so a naive split produces nothing usable.  An explicit hydrogen
    is the cap, which is also why attaching a reagent has to **remove** it.
    """
    pieces = F.split_ligand(BENZAMIDINE, cuts=AMIDINE_PHENYL_CUT)
    formulas = sorted(rdMolDescriptors.CalcMolFormula(piece) for piece in pieces)
    assert formulas == ["C6H6", "CH4N2"], formulas
    amidine = [p for p in pieces if F.heavy_atom_count(p) == 3][0]
    phenyl = [p for p in pieces if F.heavy_atom_count(p) == 6][0]
    # The amidine carries an explicit cap hydrogen: that is the growth site.
    assert any(atom.GetAtomicNum() == 1 for atom in amidine.GetAtoms())
    assert Chem.MolToSmiles(phenyl) == "[H]c1ccccc1"


def test_the_docked_pose_is_mapped_onto_the_ligand_template():
    """RDKit cannot read PDBQT, so the pose is rebuilt from the parsed atoms."""
    if not (SYS / "ligand.pdbqt").exists():
        pytest.skip("the bundled 3PTB ligand is missing")
    template = Chem.MolFromSmiles(BENZAMIDINE)
    fixed = F.docked_pose_mol(BENZAMIDINE, SYS / "ligand.pdbqt")
    assert fixed.GetNumAtoms() == 9
    assert Chem.MolToSmiles(fixed) == "N=C(N)c1ccccc1"
    match = fixed.GetSubstructMatch(template)
    assert len(match) == 9
    # Template atoms 1 (the amidine carbon) and 3 (the ipso carbon) are bonded in the
    # docked pose, which is what makes the split's cut the right one.
    assert fixed.GetBondBetweenAtoms(int(match[1]), int(match[3])) is not None


# ---------------------------------------------------------------------------
# The dual failure: clamped is shape-only, corrected saturates
# ---------------------------------------------------------------------------


def anionic_field():
    """A small anionic pocket: the aspartate-like case, cheap enough for a unit test.

    The charge is -3 rather than -1 so that a formally cationic probe **saturates** the
    default 10 kcal/mol reference: measured, one -1 charge 5 Å away gives -5.19 kcal/mol
    for a +1 probe, which is a usable term but not a pinned one, and the pinned case is
    what the second test is about.
    """
    atoms = [
        PS.PocketAtom(x=5.0, y=0.0, z=0.0, element="O", charge=-3.0),
        PS.PocketAtom(x=-5.0, y=0.0, z=0.0, element="C", charge=0.0),
    ]
    return PS.PocketField.build(atoms, center=(0.0, 0.0, 0.0), size=(6.0, 6.0, 6.0),
                               spacing=0.5)


def test_a_clamped_electrostatic_term_makes_the_score_shape_only():
    """Failure mode 1: the clamp swallows the term and the score is the shape term."""
    pocket = anionic_field()
    coords = np.array([[1.0, 0.0, 0.0]])
    neutral = np.array([-0.30])          # a neutral amidine nitrogen: negative
    clamped = PS.pocket_score(coords, neutral, pocket)
    assert clamped.esp > 0.0
    assert clamped.esp_term_used is False
    assert clamped.combined == pytest.approx(0.5 * clamped.shape)
    # The capability predicate is what says the charge set cannot hold the formal +1.
    capability = P.can_represent_formal_charge(mol("N=C(N)c1ccccc1"),
                                              np.array([-0.30, 0.40, -0.32,
                                                        0.1, -0.1, -0.1, -0.1, 0.1, 0.1]))
    assert capability.possible is False
    assert "cannot represent that formal charge" in capability.statement()


def test_the_corrected_charge_saturates_the_term_and_that_is_not_an_improvement():
    """Failure mode 2, and the reason the correction is not a fix for placement.

    Restoring the formal charge un-clamps the term — and then it **saturates**: a buried
    ion sits in a field deep enough to drive ``-ESP/E_ref`` past 1.0, so the term pins at
    its maximum and the combined score is half shape plus half a constant.  A rank built
    on that constant is not a measurement, which is why `docs/FRAGMENTS.md` reports the
    LE jump 0.618 -> 1.243 as a hazard rather than an improvement.
    """
    pocket = anionic_field()
    coords = np.array([[1.0, 0.0, 0.0]])
    positive = np.array([1.0])           # the corrected, formally cationic nitrogen
    corrected = PS.pocket_score(coords, positive, pocket)
    assert corrected.esp < -PS.DEFAULT_ESP_REFERENCE, "deep enough to saturate"
    assert corrected.esp_term_used is True
    assert corrected.esp_term == 1.0, "the term is pinned at its maximum"
    assert corrected.combined == pytest.approx(0.5 * corrected.shape + 0.5, abs=1e-9)
    # And the capability predicate now passes, which is exactly the trap: the charge set
    # is 'correct' and the score is still not a ranking.
    fixed = P.correct_formal_charges(
        mol(BENZAMIDINE), np.array([-0.30, 0.40, -0.32, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1])
    )[0]
    assert P.can_represent_formal_charge(mol(BENZAMIDINE), fixed).possible is True


# ---------------------------------------------------------------------------
# Validation 4(a): the fragments do NOT reproduce the crystal pose
# ---------------------------------------------------------------------------


@has_demo
@pytest.mark.slow
def test_the_fragment_placement_failure_is_pinned():
    """**A failure, asserted as a failure.**  The fragments do not land where the
    ligand's substructures sit, before or after the formal-charge correction.

    If a future change makes fragment placement work, this test fails — and that is the
    point: it is the tripwire that says 'the underlying problem was fixed, rewrite the
    doc'.  The bounds are deliberately loose (``> 2.0`` Å) because the *claim* is that the
    placement is not trustworthy at 3-6 heavy atoms, not that the number is stable.
    """
    pocket = three_ptb()
    template = Chem.MolFromSmiles(BENZAMIDINE)
    fixed = F.docked_pose_mol(BENZAMIDINE, SYS / "ligand.pdbqt")
    match = fixed.GetSubstructMatch(template)
    conformer = fixed.GetConformer()
    docked = {
        index: np.array([conformer.GetAtomPosition(index).x,
                         conformer.GetAtomPosition(index).y,
                         conformer.GetAtomPosition(index).z])
        for index in range(fixed.GetNumAtoms())
    }
    results = {}
    for fragment in F.split_ligand(BENZAMIDINE, cuts=AMIDINE_PHENYL_CUT):
        heavy = F.heavy_atom_count(fragment)
        placed = F.place_fragment(fragment, pocket, name=f"f{heavy}", samples=48,
                                  levels=5, population=40, restarts=2)
        bare = Chem.RemoveHs(fragment)
        reference = np.asarray([docked[match[i]] for i in template.GetSubstructMatch(bare)])
        results[heavy] = (placed, F.placement_rmsd(placed, reference))
    assert set(results) == {3, 6}
    amidine, amidine_rmsd = results[3]
    phenyl, phenyl_rmsd = results[6]
    # The failure, both fragments, and the amidine's field minimum in the same statement.
    assert amidine_rmsd > 2.0, f"amidine placement unexpectedly good: {amidine_rmsd:.2f} A"
    assert phenyl_rmsd > 2.0, f"phenyl placement unexpectedly good: {phenyl_rmsd:.2f} A"
    assert amidine.score >= 0.99, (
        "the corrected amidine saturates the electrostatic term: "
        f"score {amidine.score:.3f}"
    )
    assert any("SHAPE ALONE" in note or "clamped" in note for note in amidine.notes), (
        "the placement must say the term was clamped or shape-decided"
    )


# ---------------------------------------------------------------------------
# Validation 4(b): the one positive result
# ---------------------------------------------------------------------------


@has_demo
@pytest.mark.slow
def test_growing_the_amidine_recovers_benzamidine_at_rank_one():
    """**The positive result.**  From a three-heavy-atom fragment, by enumeration and
    incremental ligand efficiency, the top growth is benzamidine — the actual 3PTB
    ligand.

    The assertion is on the *chemistry* (a benzamidine substructure in the leader), not on
    a reagent name, because a renamed reagent is not a different result.
    """
    pocket = three_ptb()
    amidine = [fragment for fragment in F.split_ligand(BENZAMIDINE, cuts=AMIDINE_PHENYL_CUT)
               if F.heavy_atom_count(fragment) == 3][0]
    placed = F.place_fragment(amidine, pocket, name="amidine", samples=48, levels=5,
                              population=40, restarts=2)
    report = F.grow_fragment(placed, pocket, seed=20240101)
    assert report.n_generated > 0
    survivors = report.survivors
    assert survivors, "some growth must survive the clash filter"
    assert report.n_clash_rejected + len(survivors) + report.n_unbuildable == report.n_generated
    # Ranked by incremental LE, the leader is benzamidine (the cap hydrogen is still on
    # the amidine carbon, so the substructure match is on the finished ligand).
    leader = survivors[0]
    # Ranked by incremental LE descending: the leader is the least negative, because a
    # more negative delta_le means the added atoms cost efficiency per atom.  Measured:
    # the phenyl addition (benzamidine) is -0.898 while a bare amino addition is -5.561,
    # so ranking by per-atom efficiency is what puts the whole aromatic ring first.
    assert leader.delta_le >= survivors[-1].delta_le
    assert leader.added_heavy >= 1
    # The leader's molecule, with the exploration cap hydrogen (the growth site the
    # reagent substituted) dropped, **is benzamidine** — asserted on its constitution
    # rather than on a canonical SMILES, because the amidine/imidine tautomers
    # (`N=C(N)c1ccccc1` and `N=CNc1ccccc1`) are the same molecule and RDKit's tautomer
    # canonicaliser only unifies them when the hydrogens are explicit.  A plain string
    # comparison would call a correct result wrong, so the claim is stated as: the right
    # formula, the right size, a phenyl ring, and the amidine N-C-N unit.
    leader_form = Chem.RemoveHs(Chem.MolFromSmiles(leader.smiles))
    assert rdMolDescriptors.CalcMolFormula(leader_form) == "C7H8N2"
    assert leader_form.GetNumHeavyAtoms() == 9
    assert leader_form.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1")), leader.smiles
    assert leader_form.HasSubstructMatch(Chem.MolFromSmarts("[NX3][CX3]=[NX2]")) or \
        leader_form.HasSubstructMatch(Chem.MolFromSmarts("[NX2]=[CX3][NX3]")), leader.smiles
    assert leader.added_heavy > 0 and math.isfinite(leader.delta_le)
    # The note says what the ranking means and that it is not synthesis-aware.
    joined = " ".join(report.notes)
    assert "efficiency per added atom" in joined
    assert "NOT synthesis-aware" in joined


@has_demo
@pytest.mark.slow
def test_the_incremental_le_spread_says_the_top_two_cannot_be_separated():
    """ΔLE is a heuristic, so its noise is estimated rather than assumed away.

    Measured spread is 0.0046 kcal/mol over three seeds with the top two reagents swapping
    — so the honest statement is that ΔLE **cannot separate the top two**, which is a
    different claim from "phenyl wins".
    """
    pocket = three_ptb()
    amidine = [fragment for fragment in F.split_ligand(BENZAMIDINE, cuts=AMIDINE_PHENYL_CUT)
               if F.heavy_atom_count(fragment) == 3][0]
    placed = F.place_fragment(amidine, pocket, name="amidine", samples=48, levels=5,
                              population=40, restarts=2)
    spread = F.growth_spread(placed, pocket, seeds=(20240101, 7, 99), max_growths=24)
    assert spread["n"] == 3, spread
    assert len(spread["values"]) == 3
    assert math.isfinite(spread["spread"])
    # A small spread is the finding: it is why the top two cannot be separated.
    assert spread["spread"] < 0.5, spread
    assert isinstance(spread["agree"], bool)


# ---------------------------------------------------------------------------
# Validation 4(c): the linker enumerator refuses, correctly
# ---------------------------------------------------------------------------


def test_the_linker_span_window_rejects_a_linker_that_cannot_reach():
    """The geometry filter is the checkable part, and it is asserted directly.

    A direct bond demands ~1.5 Å; two independently placed fragments in this pocket sit
    4.19 Å apart, so the zero-atom linker must be rejected **with the numbers named** —
    and that rejection is the correct behaviour, not a failure of the enumerator.
    """
    report = F.link_fragments(
        F.PlacedFragment(mol=mol("N=C(N)"), name="a",
                         coords=np.zeros((3, 3)),
                         vectors=[(1, np.array([0.0, 1.0, 0.0]), np.zeros(3), None)]),
        F.PlacedFragment(mol=mol("c1ccccc1"), name="b",
                         coords=np.zeros((6, 3)),
                         vectors=[(0, np.array([0.0, -1.0, 0.0]),
                                   np.array([4.19, 0.0, 0.0]), None)]),
        None,
    )
    assert report.demand == pytest.approx(4.19, abs=1e-6)
    assert report.n_generated == len(F.LINKERS)
    direct = [item for item in report.candidates if item.linker == "direct"][0]
    assert direct.survived is False
    assert "cannot cover" in direct.rejected
    assert direct.span == pytest.approx(1.5, abs=0.01)
    # Nothing survives, because no short linker in the set can reach 4.19 A.
    assert not report.survivors
    assert "4.19 A apart" in report.notes[0]


def test_the_linker_table_names_why_each_candidate_was_rejected():
    """A rejection with a reason is a result; a silent empty table is not."""
    report = F.link_fragments(
        F.PlacedFragment(mol=mol("N=C(N)"), name="a", coords=np.zeros((3, 3)),
                         vectors=[(1, np.array([0.0, 1.0, 0.0]), np.zeros(3), None)]),
        F.PlacedFragment(mol=mol("c1ccccc1"), name="b", coords=np.zeros((6, 3)),
                         vectors=[(0, np.array([0.0, -1.0, 0.0]),
                                   np.array([7.0, 0.0, 0.0]), None)]),
        None,
    )
    text = report.table()
    assert "linker" in text and "verdict" in text
    assert "cannot cover" in text
    # A fragment with no attachment vector cannot be linked, and says so.
    bare = F.link_fragments(F.PlacedFragment(mol=mol("c1ccccc1"), coords=np.zeros((6, 3))),
                            F.PlacedFragment(mol=mol("c1ccccc1"), coords=np.zeros((6, 3))),
                            None)
    assert bare.candidates == []
    assert "no attachment vector" in bare.notes[0]


def test_growing_something_that_was_never_placed_is_a_caller_error():
    """A growth needs a pose to grow from, and says so instead of returning nothing."""
    with pytest.raises(ValueError, match="no placed pose"):
        F.grow_fragment(F.PlacedFragment(mol=mol("N=C(N)")), None)
    # A fragment with nothing to grow from reports that rather than an empty table.
    flat = F.PlacedFragment(mol=mol("c1ccccc1"), coords=np.zeros((6, 3)), vectors=[])
    report = F.grow_fragment(flat, None)
    assert report.n_generated == 0 and report.growths == []
    assert "no attachment vector" in report.notes[0]

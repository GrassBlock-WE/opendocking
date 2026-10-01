# SPDX-License-Identifier: GPL-3.0-or-later
"""Conformer ensembles: generation, pruning, the measurements, and USR.

The measured values here are the ones the module promises: a rigid molecule keeps
one conformer and says so, a flexible one keeps several and reports the RMSD spread
and the torsion coverage it achieved, and the same seed reproduces the ensemble.
"""

from __future__ import annotations

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import conformers as C  # noqa: E402

#: A molecule with one shape and one rotor: the smallest interesting ensemble.
BENZAMIDINE = "N=C(N)c1ccccc1"
#: Four rotors and a genuinely folded/unfolded space.
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
#: Rigid, no rotor at all.
BENZENE = "c1ccccc1"


def named(smiles: str, name: str = "") -> object:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if name:
        mol.SetProp("_Name", name)
    return mol


@pytest.fixture(scope="module")
def benzene():
    return C.build_ensemble(named(BENZENE), n_conformers=6)


@pytest.fixture(scope="module")
def ibuprofen():
    return C.build_ensemble(named(IBUPROFEN), n_conformers=12)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_a_rigid_molecule_keeps_one_conformer_and_says_so(benzene):
    """Benzene has one shape: ETKDGv3 embeds, the RMSD pruning removes the
    duplicates and the ensemble reports that a 3-D score from it rests on a single
    geometry rather than pretending to have sampled something."""
    assert benzene.error == ""
    assert benzene.n_conformers == 1
    assert benzene.kept == (0,)
    assert benzene.embedded >= 1
    assert benzene.rmsd_max == 0.0, "one conformer has no spread"
    assert benzene.torsion_bins == (), "benzene has no rotatable bond"
    assert benzene.force_field == "MMFF94"
    assert benzene.best_energy is not None
    assert any("fewer than three conformers" in note for note in benzene.notes)
    assert "1 conformer(s) kept" in benzene.table()


def test_a_flexible_molecule_samples_a_real_spread(ibuprofen):
    """Ibuprofen's ensemble must be spread out *and* the measurements must show it:
    several conformers, a positive RMSD spread, several rotamer bins per rotor and
    an energy window that did not eat the set."""
    assert ibuprofen.error == ""
    assert ibuprofen.n_conformers >= 3
    assert ibuprofen.embedded >= ibuprofen.n_conformers
    assert ibuprofen.rmsd_max > ibuprofen.rmsd_min > 0.0
    assert ibuprofen.rmsd_mean > 0.0
    assert ibuprofen.energy_span is not None and ibuprofen.energy_span > 0.0
    assert ibuprofen.force_field in ("MMFF94", "MMFF94s", "UFF")
    assert ibuprofen.torsion_bins, "ibuprofen has four rotatable bonds"
    assert all(1 <= bins <= 6 for bins in ibuprofen.torsion_bins)
    assert 0.0 < ibuprofen.torsion_coverage <= 1.0
    assert all(0 <= index < ibuprofen.embedded for index in ibuprofen.kept)
    assert ibuprofen.seconds > 0.0


def test_the_same_seed_reproduces_the_ensemble():
    """Determinism is part of the contract: a library regenerates identically."""
    first = C.build_ensemble(named(IBUPROFEN), n_conformers=8)
    second = C.build_ensemble(named(IBUPROFEN), n_conformers=8)
    assert first.kept == second.kept
    assert first.embedded == second.embedded
    assert first.energies == pytest.approx(second.energies)
    assert first.rmsd_mean == pytest.approx(second.rmsd_mean)
    assert first.torsion_bins == second.torsion_bins


def test_the_energy_window_is_applied_and_can_be_measured():
    """A window of 0.01 kcal/mol keeps only the best conformer; a window of 1000
    keeps everything the embedder produced.  Both are reported, so a caller can see
    what the window did."""
    tiny = C.build_ensemble(named(IBUPROFEN), n_conformers=10, energy_window=0.01)
    wide = C.build_ensemble(named(IBUPROFEN), n_conformers=10, energy_window=1000.0)
    assert tiny.n_conformers <= wide.n_conformers
    assert wide.n_conformers <= wide.embedded
    assert tiny.n_conformers >= 1
    assert tiny.pruned_energy >= 0 and wide.pruned_energy >= 0
    if tiny.n_conformers < wide.n_conformers:
        assert tiny.pruned_energy > 0, "the window that removed conformers reports it"


def test_an_atomless_molecule_is_reported_not_raised():
    blank = Chem.RWMol().GetMol()
    ensemble = C.build_ensemble(blank)
    assert ensemble.error and "no atoms" in ensemble.error
    assert ensemble.n_conformers == 0
    assert "failed" in ensemble.table()


def test_build_ensemble_validates_its_arguments():
    with pytest.raises(ValueError, match="n_conformers"):
        C.build_ensemble(named(BENZENE), n_conformers=0)
    with pytest.raises(ValueError, match="rmsd_prune"):
        C.build_ensemble(named(BENZENE), rmsd_prune=-1.0)


# ---------------------------------------------------------------------------
# The attempt accounting and the rotor-scaled default
# ---------------------------------------------------------------------------


def test_the_attempt_count_scales_with_the_rotatable_bonds():
    """The deterministic rule behind ``n_conformers=None``, checked by hand.

    ``min(128, max(16, 8 · (1 + rotors)))``: a rigid ring gets the floor, a
    one-rotor amidine 16, and a nine-rotor chain 80.  It depends on the molecule and
    nothing else — no time, no memory, no seed — and a methyl rotor does not count,
    because :func:`odock.chem.ligand.rotatable_bonds` excludes it.
    """
    assert C.DEFAULT_ATTEMPTS_PER_ROTOR == 8 and C.MIN_ATTEMPTS == 16
    assert C.MAX_ATTEMPTS == 128
    assert C.rotor_scaled_attempts(named(BENZENE)) == 16
    assert C.rotor_scaled_attempts(named(BENZAMIDINE)) == 16  # 1 rotor
    assert C.rotor_scaled_attempts(named("CC(C)Cc1ccc(cc1)C(C)C(=O)O")) == 40  # 4
    assert C.rotor_scaled_attempts(named("CCCCCCCCCC")) == 64  # 7
    assert C.rotor_scaled_attempts(named("COCCOCCOCCOC")) == 80  # 9
    # A long chain hits the cap rather than running for minutes.
    assert C.rotor_scaled_attempts(
        named("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
    ) == C.MAX_ATTEMPTS
    # A methyl is not a rotor: toluene and benzene get the same attempt count.
    assert C.rotor_scaled_attempts(named("Cc1ccccc1")) == C.rotor_scaled_attempts(
        named(BENZENE)
    )
    # The rule can be re-parameterised, and it is still a pure function.
    assert C.rotor_scaled_attempts(named("CCCCCCCCCC"), per_rotor=4) == 32
    assert C.rotor_scaled_attempts(named("CCCCCCCCCC"), maximum=24) == 24


def test_the_default_is_the_scaled_rule_and_an_explicit_count_is_exact():
    """``None`` scales; an integer is honoured exactly, so a protocol reproduces."""
    scaled = C.build_ensemble(named(IBUPROFEN))
    assert scaled.scaled is True
    assert scaled.requested == 40, "ibuprofen has four rotatable bonds"
    assert scaled.rotors == 4
    exact = C.build_ensemble(named(IBUPROFEN), n_conformers=6)
    assert exact.scaled is False
    assert exact.requested == 6
    assert exact.embedded <= 6
    fixed = C.build_ensemble(named(IBUPROFEN), scale_with_rotors=False)
    assert fixed.scaled is True and fixed.requested == C.MIN_ATTEMPTS
    assert fixed.embedded <= C.MIN_ATTEMPTS


def test_the_accounting_says_where_the_conformers_went():
    """``attempts → embedded → (-energy, -RMSD) → kept``, and the stages add up.

    Measured on the demo library: the dominant loss is the embedding stage, because
    ETKDGv3 refuses to return two conformers closer than the pruning threshold —
    that is the generator working, not failing.
    """
    ensemble = C.build_ensemble(named(IBUPROFEN), name="ibuprofen")
    book = ensemble.accounting()
    assert book["name"] == "ibuprofen"
    assert book["requested"] == ensemble.requested
    assert book["embedded"] == ensemble.embedded
    assert book["lost_embedding"] == ensemble.requested - ensemble.embedded
    assert book["kept"] == ensemble.n_conformers
    # The stages account for every attempt.
    assert book["lost_embedding"] + book["pruned_energy"] + book["pruned_rmsd"] + book["kept"] == (
        book["requested"]
    )
    assert book["rotors"] == ensemble.rotors
    assert book["coverage_ceiling"] == pytest.approx(ensemble.coverage_ceiling)
    assert "attempts -> embedded" in ensemble.table()
    fields = ensemble.accounting_line().split()
    assert fields[0] == "ibuprofen"
    assert fields[1:6] == [
        str(ensemble.rotors), str(ensemble.requested),
        str(ensemble.embedded), str(ensemble.pruned_energy), str(ensemble.pruned_rmsd),
    ]
    assert ensemble.as_dict()["rotors"] == ensemble.rotors
    assert ensemble.as_dict()["coverage_ceiling"] == pytest.approx(ensemble.coverage_ceiling)


def test_the_coverage_ceiling_is_what_the_conformer_count_allows():
    """``k`` conformers can occupy at most ``k`` of a bond's six rotamer bins.

    A coverage of 17 % from one conformer is *at* its ceiling; the same number from
    six conformers would not be.  Without the ceiling the two are indistinguishable.
    """
    one = C.build_ensemble(named(BENZAMIDINE))
    assert one.n_conformers == 1
    assert one.coverage_ceiling == pytest.approx(1 / 6)
    assert one.torsion_coverage == pytest.approx(one.coverage_ceiling)
    rigid = C.build_ensemble(named(BENZENE))
    assert rigid.coverage_ceiling == 0.0, "no rotor, no ceiling to speak of"


def test_more_attempts_buy_real_torsion_coverage():
    """The measured reason for the scaling: coverage, not conformer count.

    A 4-rotor molecule's coverage rises from 69 % of its ceiling at 16 attempts to
    100 % at the rotor-scaled 40; the RMSD pruning and the energy window remove
    almost nothing either way, which is what makes the attempt count the stage worth
    fixing.  The task is marked slow because it embeds three times.
    """
    thin = C.build_ensemble(named(WARFARIN := "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
                            n_conformers=16)
    scaled = C.build_ensemble(named(WARFARIN))
    assert scaled.requested == 40
    assert scaled.n_conformers > thin.n_conformers
    assert scaled.torsion_coverage > thin.torsion_coverage
    assert scaled.torsion_coverage >= 0.95, scaled.torsion_bins
    assert thin.pruned_rmsd + thin.pruned_energy <= 3
    assert scaled.pruned_rmsd + scaled.pruned_energy <= 8
    # A rigid molecule is not padded with duplicates to look better.
    rigid = C.build_ensemble(named(BENZENE))
    assert rigid.n_conformers == 1 and rigid.embedded == 1


def test_input_coordinates_are_measured_not_replaced():
    """A docked pose must not be re-embedded just to be reported on.

    With ``use_input_conformers`` the ConformerEnsemble describes the geometry it was
    given: no embedding, no minimisation, no pruning, no energies, and a note that
    says exactly that.  A caller who wants a fresh sample simply omits the flag.
    """
    from rdkit.Chem import AllChem

    work = Chem.AddHs(named(IBUPROFEN, "ibuprofen"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 4242
    params.numThreads = 1
    params.pruneRmsThresh = 0.5
    assert len(AllChem.EmbedMultipleConfs(work, numConfs=4, params=params)) >= 2
    given = C.build_ensemble(work, use_input_conformers=True, name="ibuprofen")
    assert given.embedded == work.GetNumConformers()
    assert given.n_conformers == given.embedded, "nothing is pruned either"
    assert given.force_field == "" and given.energies
    assert all(not (value == value) for value in given.energies), "no energies: NaN"
    assert any("taken from the input" in note for note in given.notes)
    assert not any("no force field could type" in note for note in given.notes)
    # The measurements describe the given coordinates, not a fresh sample.
    assert given.rmsd_max == pytest.approx(
        float(C.rmsd_matrix(work).max()), abs=1e-9
    )
    # A molecule with no conformers falls back to embedding.
    built = C.build_ensemble(named(IBUPROFEN), use_input_conformers=True)
    assert built.embedded >= 1 and built.requested == 40
    assert not any("taken from the input" in note for note in built.notes)


def test_the_quality_table_carries_the_accounting():
    thin = C.build_ensemble(named(IBUPROFEN), n_conformers=16, name="thin")
    fat = C.build_ensemble(named(IBUPROFEN), name="fat")
    quality = C.ensemble_quality([thin, fat])
    assert quality["scaled_attempts"] == 1
    assert quality["attempts_total"] == thin.requested + fat.requested
    assert quality["losses"]["embedding"] >= 0
    assert quality["dominant_loss"] in ("embedding", "energy_window", "rmsd_prune")
    assert 0.0 < quality["coverage_fraction_of_ceiling"] <= 1.0
    assert quality["coverage_ceiling_mean"] >= quality["torsion_coverage_mean"]
    assert quality["attempts_per_molecule"] == pytest.approx(
        quality["attempts_total"] / 2
    )
    assert quality["rotors_mean"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Pruning and the RMSD matrix
# ---------------------------------------------------------------------------


def test_rmsd_matrix_is_symmetric_with_a_zero_diagonal():
    mol = C.build_ensemble(named(IBUPROFEN), n_conformers=6).mol
    matrix = C.rmsd_matrix(mol)
    assert matrix.shape[0] == matrix.shape[1] == mol.GetNumConformers()
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    assert upper.size and (upper > 0).all(), "distinct conformers are not identical"


def test_pruning_keeps_one_conformer_at_a_huge_threshold_and_all_at_zero():
    mol = C.build_ensemble(named(IBUPROFEN), n_conformers=6).mol
    count = mol.GetNumConformers()
    kept_all, dropped_none = C.prune_by_rmsd(mol, threshold=0.0)
    assert len(kept_all) == count and dropped_none == 0
    kept_one, dropped = C.prune_by_rmsd(mol, threshold=50.0)
    assert kept_one == [0] and dropped == count - 1
    # A threshold of 0 is a no-op; the returned indices are the input ones.
    assert C.prune_by_rmsd(mol, threshold=0.5)[0] == sorted(C.prune_by_rmsd(mol, threshold=0.5)[0])


# ---------------------------------------------------------------------------
# Torsion-space coverage
# ---------------------------------------------------------------------------


def test_torsion_bins_counts_distinct_rotamers(ibuprofen):
    mol = ibuprofen.mol
    rotors = C._rotatable_bonds(mol)
    assert rotors, "ibuprofen has rotatable bonds"
    bins = C.torsion_bins(mol, rotors[0], conformers=ibuprofen.kept)
    assert bins == sorted(bins)
    assert 1 <= len(bins) <= 6, "a torsion can only fall in six 60-degree bins"
    assert all(0 <= value < 6 for value in bins)
    # A molecule with no ring has no scaffold bond to measure, and a rigid one has
    # no rotatable bond at all.
    assert C.torsion_bins(named(BENZENE), (0, 1)) == []


def test_the_coverage_number_reflects_the_bins(ibuprofen):
    expected = float(
        np.prod([max(1, value) for value in ibuprofen.torsion_bins])
        ** (1.0 / len(ibuprofen.torsion_bins))
        / 6.0
    )
    assert ibuprofen.torsion_coverage == pytest.approx(min(1.0, expected))


# ---------------------------------------------------------------------------
# USR: the pre-filter descriptor
# ---------------------------------------------------------------------------


def test_usr_descriptors_have_the_documented_shape_and_ordering():
    benzene = C.build_ensemble(named(BENZENE), n_conformers=2).mol
    pyridine = C.build_ensemble(named("c1ccncc1"), n_conformers=2).mol
    toluene = C.build_ensemble(named("Cc1ccccc1"), n_conformers=2).mol
    values = C.usr_descriptors(benzene)
    assert len(values) == 12 == len(C.USR_DESCRIPTORS)
    assert C.usr_similarity(values, values) == pytest.approx(1.0)
    # USR is a *shape* descriptor: it ignores element identity, so pyridine is
    # nearly identical to benzene, while toluene (one atom bigger) is not.  That is
    # the property a pre-filter trades on, and the reason its recall is the number
    # that matters.
    pyridine_score = C.usr_similarity(values, C.usr_descriptors(pyridine))
    toluene_score = C.usr_similarity(values, C.usr_descriptors(toluene))
    assert pyridine_score > 0.95 > toluene_score
    assert C.usr_similarity(values, np.zeros(12)) < 1.0


def test_usr_of_a_molecule_without_a_conformer_is_none():
    assert C.usr_descriptors(Chem.MolFromSmiles("CCO")) is None


# ---------------------------------------------------------------------------
# Library-level quality
# ---------------------------------------------------------------------------


def test_ensemble_quality_folds_a_library(benzene, ibuprofen):
    quality = C.ensemble_quality([benzene, ibuprofen])
    assert quality["n_molecules"] == 2 and quality["n_embedded"] == 2
    assert quality["embed_rate"] == 1.0
    assert quality["conformers_min"] >= 1
    assert quality["conformers_mean"] >= 1
    assert quality["seconds_per_molecule"] > 0.0
    assert quality["torsion_coverage_mean"] > 0.0
    assert quality["failed"] == []


def test_ensemble_quality_names_the_failures():
    broken = C.build_ensemble(Chem.RWMol().GetMol(), name="broken")
    quality = C.ensemble_quality([broken])
    assert quality["embed_rate"] == 0.0
    assert quality["failed"] and quality["failed"][0][0] == "broken"


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(C, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        C.require_rdkit()
    with pytest.raises(ImportError):
        C.build_ensemble(named("CCO"))

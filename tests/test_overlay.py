# SPDX-License-Identifier: GPL-3.0-or-later
"""The shape and electrostatic overlay: the algebra, the search, the pre-filter.

Every number this module reports is a ratio of Gaussian overlaps, so the tests
here can hand-compute the small ones.  The three things that could silently be
wrong are pinned separately:

* **the algebra** — one atom against itself is 1.0, two atoms a distance ``d``
  apart give ``exp(−d²/4σ²)``, and the Tanimoto and Carbo indices are checked
  against values written out by hand;
* **the search** — a molecule overlaid on a rigid copy of itself is 1.000, and
  benzamidine against hydroxybenzamidine must reach the value the *chemically
  correct* superposition gives (0.923), which a rotation-only search does not;
* **the leak** — scoring an active against itself is a free 1.0, so the
  leave-one-out default is asserted in both directions.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import conformers as C  # noqa: E402
from odock import overlay as O  # noqa: E402
from odock import scaffold as S  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "libraries" / "library.smi"

BENZAMIDINE = "N=C(N)c1ccccc1"
HYDROXYBENZAMIDINE = "N=C(N)c1ccc(O)cc1"
FLUOROBENZAMIDINE = "N=C(N)c1ccc(F)cc1"
BENZENE = "c1ccccc1"
PYRIDINE = "c1ccncc1"
TOLUENE = "Cc1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
CYCLOHEXANE = "C1CCCCC1"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


def named(smiles: str, name: str = "") -> object:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if name:
        mol.SetProp("_Name", name)
    return mol


def placed(smiles: str, name: str = "", conformers: int = 1) -> O.PlacedMolecule:
    """One molecule through the same path the screen uses."""
    return O.prepare_molecule(named(smiles, name or smiles), conformers=conformers)


def chemical_shape(left, right, sigma: float = O.DEFAULT_SIGMA) -> float:
    """The shape Tanimoto at the *chemically correct* superposition (MCS + Kabsch).

    This is the reference value the search must reach: the two molecules are
    superposed through their maximum common substructure with a proper rotation,
    and the overlap is measured there.  It is a lower bound on the true optimum
    (the shape optimum can be better than the chemical one), so a search that
    finds less than this has failed.
    """
    from odock.pharmacophore import _kabsch

    core = Chem.MolFromSmiles(S.maximum_common_substructure([left, right]))
    assert core is not None
    heavy_left = [a.GetIdx() for a in left.GetAtoms() if a.GetAtomicNum() > 1]
    heavy_right = [a.GetIdx() for a in right.GetAtoms() if a.GetAtomicNum() > 1]
    match_left = left.GetSubstructMatch(core)
    target = O.heavy_coordinates(left)[[heavy_left.index(i) for i in match_left]]
    best = 0.0
    for match in right.GetSubstructMatches(core, uniquify=False):
        mobile = O.heavy_coordinates(right)[[heavy_right.index(i) for i in match]]
        rotation, translation, _ = _kabsch(mobile, target)
        moved = O.heavy_coordinates(right) @ rotation + translation
        best = max(
            best,
            O.shape_tanimoto(
                O.gaussian_overlap(O.heavy_coordinates(left), moved, sigma),
                O.gaussian_overlap(O.heavy_coordinates(left), O.heavy_coordinates(left), sigma),
                O.gaussian_overlap(moved, moved, sigma),
            ),
        )
    return best


# ---------------------------------------------------------------------------
# The algebra, by hand
# ---------------------------------------------------------------------------


def test_the_gaussian_overlap_is_the_closed_form():
    """``O_AB = ΣΣ exp(−|r_i − s_j|²/4σ²)``, which has a hand value in every case.

    One atom against itself is exactly 1.0; two single-atom molecules 1 Å apart at
    σ = 0.5 give ``exp(−1/1) = 1/e``; and two identical two-atom molecules give
    ``2 + 2·exp(−d²/4σ²)`` because each of the two atoms has a partner.
    """
    sigma = 0.5
    one = np.array([[0.0, 0.0, 0.0]])
    assert O.gaussian_overlap(one, one, sigma) == pytest.approx(1.0)
    assert O.gaussian_overlap(one, np.array([[1.0, 0.0, 0.0]]), sigma) == pytest.approx(
        math.exp(-1.0 / (4 * sigma ** 2))
    )
    assert O.gaussian_overlap(one, np.array([[1.0, 0.0, 0.0]]), sigma) == pytest.approx(
        1.0 / math.e
    )
    pair = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    expected = 2.0 + 2.0 * math.exp(-1.0 / (4 * sigma ** 2))
    assert O.gaussian_overlap(pair, pair, sigma) == pytest.approx(expected)
    # An empty point set has no overlap, and the indices never divide by zero.
    assert O.gaussian_overlap(np.zeros((0, 3)), pair, sigma) == 0.0
    assert O.shape_tanimoto(0.0, 0.0, 0.0) == 0.0
    assert O.carbo_index(0.0, 0.0, 0.0) == 0.0


def test_the_two_coefficients_are_the_hand_computed_ratios():
    """With hand-picked overlaps the two formulas are exact arithmetic."""
    # Tanimoto = O/(A + B − O) and Carbo = O/sqrt(A·B).
    assert O.shape_tanimoto(1.0, 2.0, 2.0) == pytest.approx(1.0 / 3.0)
    assert O.carbo_index(1.0, 2.0, 2.0) == pytest.approx(0.5)
    assert O.shape_tanimoto(2.0, 2.0, 2.0) == pytest.approx(1.0)
    assert O.carbo_index(3.0, 4.0, 9.0) == pytest.approx(0.5)  # 3/sqrt(36)
    # A Carbo index outside [0, 1] is possible for *signed* fields (the
    # electrostatic one); for shape densities both coefficients stay in [0, 1].
    for overlap in (0.0, 0.5, 1.0, 2.0):
        value = O.shape_tanimoto(overlap, 2.0, 3.0)
        assert 0.0 <= value <= 1.0
    # Anti-correlated fields give a negative Carbo index, and it is reported.
    assert O.carbo_index(-1.0, 1.0, 1.0) == pytest.approx(-1.0)


def test_identical_coordinates_score_exactly_one():
    """A molecule overlaid on itself is 1.000 on both terms.

    With its own Gasteiger charges the shape Tanimoto and the electrostatic Carbo
    index are both exactly 1.0, so the combined score is too.  With the charges
    zeroed the field is empty, the electrostatic term contributes nothing, and the
    combined score falls to the shape weight — which is the documented convention,
    not a bug: a molecule with no charge has no field to agree about.
    """
    item = placed(BENZENE)
    coords = item.conformers[0]
    score = O.rigid_overlay(coords, item.charges, coords, item.charges)
    assert score.shape == pytest.approx(1.0, abs=1e-6)
    assert score.esp == pytest.approx(1.0, abs=1e-6)
    assert score.combined == pytest.approx(1.0, abs=1e-6)
    neutral = np.zeros(coords.shape[0])
    empty = O.rigid_overlay(coords, neutral, coords, neutral)
    assert empty.shape == pytest.approx(1.0, abs=1e-6)
    assert empty.esp == 0.0
    assert empty.combined == pytest.approx(O.DEFAULT_SHAPE_WEIGHT, abs=1e-6)


def test_the_score_is_invariant_under_a_rigid_transformation():
    """Rotating and translating the probe must not change the best overlay."""
    reference = placed(BENZENE)
    theta = math.radians(37.0)
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moved = reference.conformers[0] @ rotation + np.array([4.0, -1.5, 2.25])
    plain = O.rigid_overlay(
        reference.conformers[0], reference.charges,
        reference.conformers[0], reference.charges,
    )
    shifted = O.rigid_overlay(
        reference.conformers[0], reference.charges, moved, reference.charges,
    )
    assert shifted.shape == pytest.approx(plain.shape, abs=1e-6)
    assert shifted.shape == pytest.approx(1.0, abs=1e-6)


def test_the_two_terms_combine_with_the_documented_weight():
    """``combined = w·shape + (1−w)·max(0, ESP)``, checked on the returned terms."""
    left = placed(BENZAMIDINE, "benzamidine")
    right = placed(PYRIDINE, "pyridine")
    for weight in (0.0, 0.25, 0.5, 1.0):
        score = O.rigid_overlay(
            left.conformers[0], left.charges, right.conformers[0], right.charges,
            shape_weight=weight,
        )
        assert score.combined == pytest.approx(
            weight * score.shape + (1.0 - weight) * max(0.0, score.esp), abs=1e-9
        )
    # With electrostatics off the score *is* the shape term, and the ESP index is
    # still reported — the point of reporting it is that a reader can see what the
    # term would have contributed without a second run.
    off = O.rigid_overlay(
        left.conformers[0], left.charges, right.conformers[0], right.charges,
        electrostatic=False,
    )
    assert off.combined == pytest.approx(off.shape)
    assert off.esp != 0.0


def test_the_shape_metric_can_be_switched_and_the_arguments_are_validated():
    left = placed(BENZENE)
    right = placed(TOLUENE)
    tanimoto = O.rigid_overlay(
        left.conformers[0], left.charges, right.conformers[0], right.charges,
        shape_metric="tanimoto",
    )
    carbo = O.rigid_overlay(
        left.conformers[0], left.charges, right.conformers[0], right.charges,
        shape_metric="carbo",
    )
    # Carbo is the more forgiving coefficient: it ignores the size difference that
    # the Tanimoto's union penalises.
    assert carbo.shape > tanimoto.shape > 0.5
    with pytest.raises(ValueError, match="unknown shape metric"):
        O.rigid_overlay(
            left.conformers[0], left.charges, right.conformers[0], right.charges,
            shape_metric="nonsense",
        )
    with pytest.raises(ValueError, match="unknown alignment objective"):
        O.rigid_overlay(
            left.conformers[0], left.charges, right.conformers[0], right.charges,
            align_on="nonsense",
        )
    with pytest.raises(ValueError, match="one charge per atom"):
        O.rigid_overlay(left.conformers[0], np.zeros(1), right.conformers[0], right.charges)


def test_an_empty_point_set_scores_zero_rather_than_raising():
    score = O.rigid_overlay(np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros(0))
    assert score.combined == 0.0 and score.shape == 0.0


# ---------------------------------------------------------------------------
# The search reaches the chemically correct superposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles, name",
    [
        (HYDROXYBENZAMIDINE, "hydroxybenzamidine"),
        (FLUOROBENZAMIDINE, "fluorobenzamidine"),
        ("N=C(N)c1ccccc1Cl", "chloro_benzamidine"),
        ("c1ccc(cc1)C(=N)NC", "benzamidine_methyl"),
        ("c1ccc(cc1)CC(=N)N", "benzylamidine"),
    ],
)
def test_the_search_reaches_the_chemical_superposition(smiles, name):
    """The pose the search finds must be at least as good as the chemical one.

    The reference is computed here, not pinned: the probe is superposed on the
    reference through its maximum common substructure with a proper rotation
    (:func:`chemical_shape`) and the shape Tanimoto is measured there.  A
    rotation-only search, and a search without the translation degree of freedom,
    both came in 0.06-0.09 *below* this line on the para-substituted amidines — so
    this is the test that fails if either is removed.
    """
    left = placed(BENZAMIDINE, "benzamidine")
    right = placed(smiles, name)
    measured = chemical_shape(left.mol, right.mol)
    found = O.rigid_overlay(
        left.conformers[0], left.charges, right.conformers[0], right.charges
    )
    assert found.shape >= measured - 0.02, (
        f"{name}: the search found {found.shape:.3f} where the chemical "
        f"superposition gives {measured:.3f}"
    )


def test_a_sigma_of_half_an_angstrom_separates_what_it_must():
    """The measured reason the default σ is 0.5 Å and not 0.8 Å.

    At σ = 0.8 Å the densities are so fat that a flat benzene ring scores 0.989
    against a puckered cyclohexane chair and 0.944 against toluene: the shape term
    stops saying anything about the third dimension.  At σ = 0.5 Å an extra methyl
    costs 0.11, a fused ring 0.40 and the chair 0.07 — measured through
    :func:`odock.overlay.prepare_molecule`, so the numbers here are the ones the
    module actually produces, and a molecule still scores exactly 1.0 against
    itself.
    """
    benzene = placed(BENZENE, "benzene")
    pairs = {
        "toluene": (placed(TOLUENE, "toluene"), 0.887),
        "naphthalene": (placed(NAPHTHALENE, "naphthalene"), 0.599),
        "cyclohexane": (placed(CYCLOHEXANE, "cyclohexane"), 0.926),
        "pyridine": (placed(PYRIDINE, "pyridine"), 0.997),
    }
    for name, (other, expected) in pairs.items():
        score = O.rigid_overlay(
            benzene.conformers[0], benzene.charges,
            other.conformers[0], other.charges, sigma=0.5,
        )
        assert score.shape == pytest.approx(expected, abs=0.02), name
    fat_cyclohexane = O.rigid_overlay(
        benzene.conformers[0], benzene.charges,
        pairs["cyclohexane"][0].conformers[0], pairs["cyclohexane"][0].charges,
        sigma=0.8,
    )
    fat_naphthalene = O.rigid_overlay(
        benzene.conformers[0], benzene.charges,
        pairs["naphthalene"][0].conformers[0], pairs["naphthalene"][0].charges,
        sigma=0.8,
    )
    assert fat_cyclohexane.shape > 0.98, "at σ = 0.8 the ring and the chair are one blob"
    assert fat_naphthalene.shape > 0.75 > pairs["naphthalene"][1]
    assert O.rigid_overlay(
        benzene.conformers[0], benzene.charges,
        benzene.conformers[0], benzene.charges, sigma=0.5,
    ).shape == pytest.approx(1.0, abs=1e-6)


def test_the_electrostatic_term_is_reported_and_can_be_turned_off():
    """Benzene and pyridine are nearly the same shape and a different field."""
    benzene = placed(BENZENE, "benzene")
    pyridine = placed(PYRIDINE, "pyridine")
    score = O.rigid_overlay(
        benzene.conformers[0], benzene.charges, pyridine.conformers[0], pyridine.charges
    )
    assert score.shape > 0.99, "USR is not the only descriptor that ignores elements"
    assert 0.0 < score.esp < 1.0, "the ring nitrogen changes the field"
    # Gasteiger on a hydrocarbon is a small alternating pattern, so the two
    # hydrocarbon fields correlate strongly; the reported value is what it is and
    # is not gated, which is why the raw index is always in `details`.
    same = O.rigid_overlay(
        benzene.conformers[0], benzene.charges,
        benzene.conformers[0], benzene.charges,
    )
    assert same.esp == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Conformers, leave-one-out and the library ranking
# ---------------------------------------------------------------------------


def test_best_over_conformers_is_the_maximum_over_the_ensemble():
    """A flexible probe is scored on its best conformer, and the winner is named."""
    reference = placed(BENZAMIDINE, "benzamidine")
    probe = placed(IBUPROFEN, "ibuprofen", conformers=8)
    assert probe.n_conformers >= 2, "ibuprofen must give the ensemble something to do"
    best = O.best_overlay(reference, probe, label="benzamidine")
    singles = [
        O.rigid_overlay(
            reference.conformers[0], reference.charges, conformer, probe.charges
        ).combined
        for conformer in probe.conformers
    ]
    assert best.combined == pytest.approx(max(singles), abs=1e-9)
    assert best.conformer == int(np.argmax(singles))
    assert best.reference == "benzamidine"
    assert best.n_rotations > 0


def test_the_overlay_removes_the_self_similarity_leak(molecules, actives):
    """Both directions of the leak, the same way the fingerprint test pins it."""
    names = [mol.GetProp("_Name") for mol in molecules]
    leaky = O.screen(molecules, actives, conformers=1, names=names, leave_one_out=False)
    honest = O.screen(molecules, actives, conformers=1, names=names, leave_one_out=True)
    assert leaky.details["benzamidine"].shape == pytest.approx(1.0, abs=1e-6)
    assert leaky.details["benzamidine"].reference == "benzamidine"
    assert honest.details["benzamidine"].shape < 1.0
    assert honest.details["benzamidine"].reference != "benzamidine"
    assert "leave-one-out" in " ".join(honest.notes)


def test_screen_ranks_every_molecule_and_reports_both_terms(molecules, actives):
    names = [mol.GetProp("_Name") for mol in molecules]
    result = O.screen(molecules, actives, conformers=1, names=names)
    assert len(result.entries) == len(molecules)
    assert set(result.names) == set(names)
    assert result.scores == sorted(result.scores, reverse=True)
    assert result.rank_of("caffeine") is not None and result.rank_of("nothing") is None
    for name, score in result.entries:
        detail = result.details[name]
        assert score == pytest.approx(detail.combined, abs=1e-9)
        assert 0.0 <= detail.shape <= 1.0
        assert -1.0 <= detail.esp <= 1.0
        assert result.ensembles[name]["n_conformers"] >= 1
    assert result.n_pairs > 0
    assert result.n_rotation_evaluations > result.n_pairs
    assert "table" in result.as_dict() or "entries" in result.as_dict()
    assert "combined" in result.table(limit=3)


def test_a_molecule_that_cannot_be_embedded_scores_zero_and_says_so():
    broken = Chem.RWMol().GetMol()
    result = O.screen([broken, named(BENZENE, "benzene")], [named(BENZAMINE := BENZAMIDINE, "benzamidine")],
                      conformers=1, names=["broken", "benzene"])
    assert result.entries[-1][1] == 0.0
    assert "could not be prepared" in " ".join(result.notes)
    assert result.rank_of("benzene") == 1


def test_prepare_molecule_keeps_the_coordinates_it_was_given():
    """Re-embedding a docked pose would silently replace what is being scored."""
    mol = named(BENZENE, "benzene")
    work = Chem.AddHs(mol)
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    params.numThreads = 1
    assert len(AllChem.EmbedMultipleConfs(work, numConfs=1, params=params)) == 1
    kept = O.prepare_molecule(work)
    assert kept.source == "given" and kept.n_conformers == 1
    assert np.allclose(kept.conformers[0], O.heavy_coordinates(work))
    built = O.prepare_molecule(work, conformers=4, rebuild=True)
    assert built.source == "built" and built.ensemble is not None


def test_screen_validates_its_reference_and_its_input():
    actives = [named(BENZAMIDINE, "benzamidine")]
    with pytest.raises(ValueError, match="not a parsable molecule"):
        O.screen(["not a molecule"], actives)
    with pytest.raises(ValueError, match="no active"):
        O.screen([named(BENZENE, "benzene")], [])
    with pytest.raises(ValueError, match="is not one of the actives"):
        O.screen([named(BENZENE, "benzene")], actives, reference="nope", conformers=1)


# ---------------------------------------------------------------------------
# The USR pre-filter
# ---------------------------------------------------------------------------


def test_usr_profile_is_the_descriptor_of_every_kept_conformer(molecules):
    by_name = {mol.GetProp("_Name"): mol for mol in molecules}
    item = O.prepare_molecule(by_name["ibuprofen"], conformers=6, name="ibuprofen")
    profiles = O.usr_profile(item)
    assert len(profiles) == item.n_conformers
    assert all(profile.shape == (12,) for profile in profiles)
    # A conformer set similar to itself is 1.0, and the best over the sets is the
    # maximum over pairs — not the first pair.
    assert O.usr_similarity_to(profiles, profiles) == pytest.approx(1.0)
    assert O.usr_similarity_to([], profiles) == 0.0


def test_usr_profile_works_on_coordinates_taken_as_given():
    mol = named(PYRIDINE, "pyridine")
    work = Chem.AddHs(mol)
    from rdkit.Chem import AllChem

    params = AllChem.ETKDGv3()
    params.randomSeed = 11
    params.numThreads = 1
    AllChem.EmbedMultipleConfs(work, numConfs=1, params=params)
    item = O.prepare_molecule(work)
    assert item.source == "given"
    profiles = O.usr_profile(item)
    assert len(profiles) == 1 and profiles[0].shape == (12,)


def test_the_pre_filter_reports_its_recall_and_what_it_saves(molecules, actives):
    """The reporting shape has to pair the speedup with the recall, always."""
    names = [mol.GetProp("_Name") for mol in molecules]
    summary = O.usr_prefilter(
        molecules, actives, keep=0.5, conformers=1, names=names
    )
    assert summary["method"] == "usr"
    assert summary["n_library"] == len(molecules)
    assert summary["n_kept"] == math.ceil(0.5 * len(molecules))
    assert summary["n_actives"] == len(actives)
    assert 0 <= summary["actives_kept"] <= len(actives)
    assert summary["active_recall"] == pytest.approx(
        summary["actives_kept"] / summary["n_actives"]
    )
    assert summary["actives_lost"] == sorted(
        set(actives and [mol.GetProp("_Name") for mol in actives])
        - set(summary["kept_names"])
    )
    # The leak is reported next to the recall: with self-match every active is
    # kept for free, which is why the leave-one-out number is the honest one.
    assert summary["self_match_actives_kept"] == len(actives)
    assert summary["self_match_recall"] == 1.0
    # The speedup is measured on the kept molecules and extrapolated, and the
    # docking figure comes from odock.screen's own cost model.
    assert summary["overlay_seconds_kept"] > 0.0
    assert summary["overlay_seconds_full_estimate"] > summary["overlay_seconds_kept"]
    assert 0.0 < summary["overlay_saved_fraction"] < 1.0
    assert summary["estimated_seconds_kept"] < summary["estimated_seconds_full"]
    assert summary["workload_saved_fraction"] > 0.0
    assert "not a measurement" in " ".join(summary["notes"])


def test_the_pre_filter_validates_its_fraction():
    with pytest.raises(ValueError, match="keep must be in"):
        O.usr_prefilter([named(BENZENE, "benzene")], [named(BENZAMIDINE, "benzamidine")], keep=0.0)
    with pytest.raises(ValueError, match="no active"):
        O.usr_prefilter([named(BENZENE, "benzene")], [])


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(O, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        O.require_rdkit()
    with pytest.raises(ImportError):
        O.prepare_molecule(named(BENZENE))


def test_the_cost_of_one_overlay_is_measured_not_asserted():
    """The per-overlay cost is what makes the pre-filter's saving meaningful.

    Pinned loosely (an order of magnitude) because it is a wall-clock number on a
    shared machine; `docs/CONFORMERS.md` carries the measured figure.
    """
    left = placed(BENZAMIDINE, "benzamidine")
    right = placed(IBUPROFEN, "ibuprofen")
    start = time.perf_counter()
    score = O.rigid_overlay(
        left.conformers[0], left.charges, right.conformers[0], right.charges
    )
    seconds = time.perf_counter() - start
    assert seconds < 1.0, f"one overlay took {seconds:.2f} s"
    assert score.n_rotations > 100, "the search evaluated a real candidate set"


# ---------------------------------------------------------------------------
# The bundled library
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
    wanted = (
        "benzamidine",
        "benzamidine_methyl",
        "hydroxybenzamidine",
        "fluorobenzamidine",
        "chloro_benzamidine",
    )
    return [mol for mol in molecules if mol.GetProp("_Name") in wanted]


@pytest.mark.slow
def test_the_overlay_enriches_the_amidine_series_in_the_bundled_library(molecules, actives):
    """A documented ordering: the series is enriched, and the exact ranks are pinned.

    Measured (σ = 0.5 Å, electrostatic term on, 4 conformers, leave-one-out): the
    three amidines that differ from the reference by a small substituent at the
    amidine-bearing carbon take ranks 1-3, and the two para-substituted ones take
    ranks 7 and 8 — below nicotinamide and salicylic acid, whose rings are the same
    shape.  The honest statement is therefore that the overlay recovers the series
    as an *enrichment* (mean amidine rank 4.2 of 17, against 9.0 for a random
    ranking), not the pharmacophore model's 1-5 sweep: a shape Tanimoto normalised
    by self-overlap is nearly blind to one extra atom on a shared ring.  With the
    electrostatic term off the same run puts one amidine in the top three instead
    of three, which is why the term is on by default.
    """
    names = [mol.GetProp("_Name") for mol in molecules]
    amidine_names = {mol.GetProp("_Name") for mol in actives}
    result = O.screen(molecules, actives, conformers=4, names=names)
    assert len(result) == len(molecules) == 17
    ranks = {name: result.rank_of(name) for name in names}
    assert ranks["benzamidine_methyl"] == 1
    assert ranks["benzamidine"] == 2
    assert ranks["chloro_benzamidine"] == 3
    amidine_ranks = [ranks[name] for name in amidine_names]
    assert sorted(amidine_ranks) == [1, 2, 3, 7, 8]
    assert np.mean(amidine_ranks) < 9.0, "better than the mean rank of a random ranking"
    assert result.details["hydroxybenzamidine"].shape > 0.8, (
        "the shape term sees the para-substituted amidines; it is the ESP term that "
        "separates them from the amidines sharing the ring"
    )
    shape_only = O.screen(
        molecules, actives, conformers=4, names=names, electrostatic=False
    )
    top_three = shape_only.names[:3]
    assert sum(1 for name in top_three if name in amidine_names) == 1
    assert sum(1 for name in result.names[:3] if name in amidine_names) == 3

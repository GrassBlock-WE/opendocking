# SPDX-License-Identifier: GPL-3.0-or-later
"""Pharmacophore models: feature perception, model building, fitting, enrichment.

The measured numbers here come from the bundled demo library
(`demo/libraries/library.smi`), embedded with ETKDGv3 at seed 20240101 when a model is
built and seed 42 when a library is screened — the same settings the CLI defaults
to, so `odock pharmacophore build/screen` reproduces them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import pharmacophore as P  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "libraries" / "library.smi"

#: The six amidines of the demo library: the labelled "actives" of the worked
#: enrichment example.  All six are trypsin S1-pocket binders (benzamidine is the
#: 3PTB crystal ligand); the other eleven molecules have no known trypsin affinity.
AMIDINES = (
    "benzamidine",
    "benzamidine_methyl",
    "benzylamidine",
    "hydroxybenzamidine",
    "fluorobenzamidine",
    "chloro_benzamidine",
)
RING_AMIDINES = tuple(name for name in AMIDINES if name != "benzylamidine")
#: The core the ring-amidines share; the model is superposed on it.
CORE = "N=C(N)c1ccccc1"


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
    return [mol for mol in demo_library if mol.GetProp("_Name") in RING_AMIDINES]


@pytest.fixture(scope="module")
def model(amidines):
    return P.build_model(amidines, frame="align", core=CORE)


@pytest.fixture(scope="module")
def hits(model, demo_library, demo_names):
    return P.screen(model, demo_library, names=demo_names, conformers=4, seed=20240101)


# ---------------------------------------------------------------------------
# Feature perception
# ---------------------------------------------------------------------------


def test_features_of_benzamidine_are_the_amidine_and_the_ring():
    """Benzamidine has two donor nitrogens (the =NH and the NH2), one aromatic
    ring and one lumped hydrophobe for that ring — and *no* per-atom hydrophobe,
    which is the deduplication that makes the model readable."""
    mol, _ = P._ensure_conformer(Chem.MolFromSmiles(CORE), seed=20240101)
    found = P.features_from_mol(mol)
    families = [family for family, _, _ in found]
    assert sorted(families) == ["aromatic", "donor", "donor", "hydrophobic"]
    aromatic = next(position for family, position, _ in found if family == "aromatic")
    hydrophobic = next(position for family, position, _ in found if family == "hydrophobic")
    assert aromatic == pytest.approx(hydrophobic), "the ring's two features share a centroid"
    # The two donors are the amidine nitrogens, ~2.2 Å apart.
    donors = [position for family, position, _ in found if family == "donor"]
    assert math.dist(donors[0], donors[1]) == pytest.approx(2.2, abs=0.4)


def test_feature_perception_needs_a_conformer():
    flat = Chem.MolFromSmiles("CCO")
    with pytest.raises(ValueError, match="3-D coordinates"):
        P.features_from_mol(flat)


def test_feature_perception_rejects_a_missing_conformer_index():
    mol, _ = P._ensure_conformer(Chem.MolFromSmiles("CCO"))
    with pytest.raises(IndexError, match="conformer 5"):
        P.features_from_mol(mol, conf_id=5)


def test_zinc_binders_are_dropped_by_default():
    """A zinc-binding site is a receptor property, so it cannot be part of a
    ligand-derived model unless the caller asks for it."""
    mol, _ = P._ensure_conformer(Chem.MolFromSmiles("c1ccncc1"))
    assert all(family in P.FEATURE_FAMILIES for family, _, _ in P.features_from_mol(mol))
    assert P.FAMILY_MAP["ZnBinder"] is None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_kabsch_recovers_a_known_rotation_and_refuses_a_mirror():
    rng = np.random.default_rng(0)
    mobile = rng.normal(size=(5, 3))
    angle = 0.7
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = mobile @ rotation + np.array([1.0, -2.0, 0.5])
    found, translation, rmsd = P._kabsch(mobile, target)
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(mobile @ found + translation, target)
    assert np.linalg.det(found) == pytest.approx(1.0)
    # A mirrored copy cannot be superposed by a rotation: the fit is worse and the
    # transform is still a proper rotation, so no molecule is ever reflected onto
    # a model.
    mirrored = mobile * np.array([1.0, 1.0, -1.0])
    _, _, mirrored_rmsd = P._kabsch(mobile, mirrored)
    assert mirrored_rmsd > 0.1


# ---------------------------------------------------------------------------
# Building a model
# ---------------------------------------------------------------------------


def test_model_from_five_ring_amidines(model):
    """Every feature recurs in **all five** members, so support is 5/5 and the
    model is backed by evidence rather than by one molecule."""
    assert model.core == CORE
    assert model.n_members == 5 and model.n_used == 5
    assert model.is_meaningful
    assert model.n_features == 4
    assert model.families() == {"aromatic": 1, "donor": 2, "hydrophobic": 1}
    assert model.support_profile() == [5, 5, 5, 5]
    assert all(feature.fraction == 1.0 for feature in model.features)
    assert model.shape_mode == "envelope" and model.has_shape
    assert len(model.envelope) == 49, "five members x their heavy atoms"
    assert "superposed on the core" in " ".join(model.notes)
    assert "4 feature(s) from 5 of 5 member(s)" in model.table()


def test_model_alignment_of_a_para_analogue_places_the_amidine_correctly(model):
    """The core alignment has to pick the *best* substructure match, not the first
    one: a symmetric core has several, and a ring-shifted match leaves a para
    analogue's amidine on the wrong side of the ring, which used to cost these
    molecules every donor match (both donors missed, fit 0.00).  With the best
    match the donors land within ~0.7 Å — a fraction of the 1.5 Å tolerance — even
    though the amidine plane is a free torsion the model cannot pin down.
    """
    mol = next(m for m in _read_library() if m.GetProp("_Name") == "fluorobenzamidine")
    work, _ = P._ensure_conformer(mol, seed=20240101, conformers=1)
    placed = P._align_on_core(work, model, conf_id=0)
    assert placed is not None
    P._set_positions(work, placed, 0)
    donors = [np.asarray(position) for family, position, _ in P.features_from_mol(work)
              if family == "donor"]
    model_donors = [np.asarray(f.position) for f in model.features if f.family == "donor"]
    distances = sorted(
        min(float(np.linalg.norm(a - b)) for b in donors) for a in model_donors
    )
    assert max(distances) < 1.0, f"the aligned amidine should reproduce the donors: {distances}"
    # Both donors match, so the molecule is scored and not rejected.
    result = P.fit_score(work, model, conformers=1, seed=20240101)
    assert result.n_missed == 0 and result.fit > 0.7


def test_threshold_decides_which_features_are_recurring():
    """A para substituent present in two of three members is evidence at a 0.5 or
    0.6 threshold and is not at 0.7: the acceptor that the para-F and the para-OH
    both present is supported by 2 members out of 3 (0.67)."""
    members = [
        m
        for m in _read_library()
        if m.GetProp("_Name") in ("benzamidine", "hydroxybenzamidine", "fluorobenzamidine")
    ]
    loose = P.build_model(members, frame="align", core=CORE, threshold=0.6)
    strict = P.build_model(members, frame="align", core=CORE, threshold=0.7)
    assert loose.families().get("acceptor") == 1
    assert next(f for f in loose.features if f.family == "acceptor").support == 2
    assert "acceptor" not in strict.families()
    assert strict.n_features == 4 and loose.n_features == 5
    assert all(feature.support == 3 for feature in strict.features)


def test_a_two_member_model_is_flagged_as_an_anecdote(amidines):
    tiny = P.build_model(amidines[:2], frame="align", core=CORE)
    assert tiny.n_used == 2
    assert not tiny.is_meaningful
    assert tiny.n_used < P.MIN_MEANINGFUL_MEMBERS == 3
    assert any("fewer than the 3" in note for note in tiny.notes)


def test_members_without_the_core_are_reported_not_silently_dropped(demo_library, demo_names):
    model = P.build_model(demo_library, frame="align", core=CORE, names=demo_names)
    assert model.n_used == 5, "only the ring-amidines contain that core"
    note = " ".join(model.notes)
    assert "dropped" in note and "caffeine" in note


def test_the_sphere_shape_constraint_is_honest_about_coming_out_empty(amidines):
    """A small rigid series' features cover its own atoms, so there is no scaffold
    atom left to put an exclusion sphere on — the model says so instead of
    pretending it has a shape constraint."""
    model = P.build_model(amidines, frame="align", core=CORE, shape="spheres")
    assert model.excluded == []
    assert not model.has_shape
    assert any("came out empty" in note for note in model.notes)
    assert "shape='envelope'" in " ".join(model.notes)


def test_build_model_validates_its_arguments(amidines):
    with pytest.raises(ValueError, match="unknown frame"):
        P.build_model(amidines, frame="floating")
    with pytest.raises(ValueError, match="unknown shape mode"):
        P.build_model(amidines, shape="blob")
    with pytest.raises(ValueError, match="threshold must be"):
        P.build_model(amidines, threshold=0.0)
    with pytest.raises(ValueError, match="radius must be positive"):
        P.build_model(amidines, radius=0.0)
    with pytest.raises(ValueError, match="no member"):
        P.build_model([])


def test_a_model_can_be_written_and_read_back(model, tmp_path):
    target = model.save(tmp_path / "model.json")
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["n_features"] == 4 and payload["meaningful"] is True
    again = P.PharmacophoreModel.load(target)
    assert again.core == model.core
    assert again.threshold == model.threshold and again.radius == model.radius
    assert [f.as_dict() for f in again.features] == [f.as_dict() for f in model.features]
    # The envelope is written with three decimals (it is a point cloud, not a
    # measurement), so the loaded copy agrees to a thousandth of an ångström.
    assert np.allclose(again.envelope, model.envelope, atol=1e-3)
    assert again.as_dict()["shape_mode"] == "envelope"
    assert again.table() == model.table()


def test_loading_something_that_is_not_a_model_is_refused(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"something": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not a pharmacophore model"):
        P.PharmacophoreModel.load(broken)


# ---------------------------------------------------------------------------
# Fitting and screening
# ---------------------------------------------------------------------------


def test_the_fit_is_the_documented_formula(model, demo_library):
    """Hand check of the score: ``fit = (sum of matched weights - missed) / n``."""
    benzamidine = demo_library[0]
    result = P.fit_score(benzamidine, model, conformers=1, seed=20240101)
    assert result.n_features == 4 and result.n_missed == 0
    hand = sum(weight for _, _, weight in result.matched) / 4
    assert result.fit == pytest.approx(hand)
    assert result.fit > 0.9, "the parent molecule fits the model built from its series"
    assert 0.0 <= result.coverage <= 1.0


def test_screening_ranks_the_five_actives_first(hits, demo_names):
    """Measured with the CLI's defaults (build and screen both seed 20240101,
    four conformers per molecule): the five ring-amidines take ranks 1-5 with fits
    0.918-0.944, the best decoy (aspirin) reaches 0.497, and three molecules cannot
    be aligned at all."""
    top5 = [hit.name for hit in hits.hits[:5]]
    assert sorted(top5) == sorted(RING_AMIDINES)
    assert hits.hits[0].fit == pytest.approx(0.944, abs=0.01)
    assert hits.hits[4].fit == pytest.approx(0.770, abs=0.02)
    assert hits.hits[5].name == "aspirin"
    assert hits.hits[5].fit == pytest.approx(0.497, abs=0.02)
    assert hits.n_library == 17
    assert hits.n_conformers == 68, "17 molecules x 4 conformers"
    assert hits.seconds >= 0.0


def test_screening_is_deterministic(model, demo_library, demo_names):
    """The same seed reproduces the same ranking, which is what makes the numbers
    in this file quotable."""
    first = P.screen(model, demo_library, names=demo_names, conformers=4, seed=20240101)
    second = P.screen(
        model, _read_library(), names=demo_names, conformers=4, seed=20240101
    )
    assert [(h.name, round(h.fit, 6)) for h in first.hits] == [
        (h.name, round(h.fit, 6)) for h in second.hits
    ]


def test_the_shape_constraint_and_unalignable_molecules_are_rejected(hits):
    """Four molecules do not get a score: three share no feature triple with the
    model, and warfarin matches three features but puts 18 of its 23 heavy atoms
    outside the members' envelope."""
    rejected = {hit.name: hit for hit in hits.hits if hit.rejected}
    assert hits.n_rejected == 4
    assert set(rejected) == {"caffeine", "triphenylene", "benzoquinone", "warfarin"}
    assert rejected["warfarin"].fit == 0.0
    assert "outside the members' envelope" in rejected["warfarin"].reason
    assert rejected["caffeine"].fit == 0.0
    assert "no rigid alignment" in rejected["caffeine"].reason
    assert all(hit.alignment for hit in hits.hits if not hit.rejected)


def test_the_shape_constraint_can_be_switched_off(model, demo_library, demo_names):
    relaxed = P.screen(model, demo_library, names=demo_names, conformers=1,
                       seed=20240101, enforce_shape=False)
    warfarin = next(hit for hit in relaxed.hits if hit.name == "warfarin")
    assert not warfarin.rejected
    assert warfarin.fit > 0.0
    assert warfarin.n_matched == 3


def test_a_molecule_that_cannot_be_aligned_is_not_given_a_score(model):
    caffeine = next(m for m in _read_library() if m.GetProp("_Name") == "caffeine")
    result = P.fit_score(caffeine, model, conformers=1, seed=20240101)
    assert result.rejected and result.fit == 0.0 and result.matched == []


def test_screen_reports_an_empty_model():
    empty = P.PharmacophoreModel(features=[])
    with pytest.raises(ValueError, match="no features"):
        P.screen(empty, [Chem.MolFromSmiles("CCO")])


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def test_enrichment_on_the_labelled_demo_set(hits):
    """Six actives in seventeen molecules, k = 6: five of the six best molecules
    are actives (83 % precision and recall), the base rate is 35 %, the enrichment
    factor is 2.36 and the AUC is 0.9545.  The result carries the caveat that this
    labelled set is far too small to be an enrichment claim."""
    report = P.enrichment(hits, AMIDINES)
    assert report["n_total"] == 17 and report["n_actives"] == 6
    assert report["k"] == 6
    assert report["n_actives_in_top_k"] == 5
    assert report["precision_at_k"] == pytest.approx(5 / 6, abs=1e-4)
    assert report["recall_at_k"] == pytest.approx(5 / 6, abs=1e-4)
    assert report["base_rate"] == pytest.approx(6 / 17, abs=1e-4)
    assert report["enrichment_factor"] == pytest.approx(2.36, abs=0.02)
    assert report["auc"] == pytest.approx(0.9545, abs=0.002)
    assert any("too small a labelled set" in note for note in report["notes"])
    # benzylamidine is the active the model misses (it does not contain the core).
    assert "benzylamidine" not in report["actives_in_top_k"]


def test_enrichment_accepts_a_ranked_list():
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.1), ("d", 0.05)]
    report = P.enrichment(ranked, {"a", "c"}, k=2)
    assert report["n_actives"] == 2 and report["k"] == 2
    assert report["precision_at_k"] == 0.5
    assert report["recall_at_k"] == 0.5
    assert report["enrichment_factor"] == pytest.approx(1.0)
    assert report["auc"] == pytest.approx(0.75)


def test_enrichment_reports_a_perfect_and_a_random_ranking():
    """With both actives in the top two of four molecules (base rate 0.5) the
    enrichment factor is 2.0 and the AUC 1.0; with both at the bottom it is 0.0."""
    perfect = P.enrichment([("a", 1.0), ("b", 0.6), ("d", 0.4), ("c", 0.2)], {"a", "b"}, k=2)
    assert perfect["precision_at_k"] == 1.0
    assert perfect["enrichment_factor"] == pytest.approx(2.0)
    assert perfect["auc"] == pytest.approx(1.0)
    random = P.enrichment([("d", 1.0), ("c", 0.6), ("b", 0.4), ("a", 0.2)], {"a", "b"}, k=2)
    assert random["enrichment_factor"] == pytest.approx(0.0)
    assert random["auc"] == pytest.approx(0.0)


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(P, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        P.require_rdkit()
    with pytest.raises(ImportError):
        P.build_model(["CCO"])

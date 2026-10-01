# SPDX-License-Identifier: GPL-3.0-or-later
"""Property-matched decoy selection, and the evidence that it matched."""

from __future__ import annotations

from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import decoys as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "libraries" / "library.smi"
POOL = ROOT / "demo" / "libraries" / "decoys.smi"

RING_AMIDINES = (
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


def _read(path):
    mols = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        mol = Chem.MolFromSmiles(fields[0])
        assert mol is not None, fields[0]
        mol.SetProp("_Name", " ".join(fields[1:]))
        mols.append(mol)
    return mols


@pytest.fixture(scope="module")
def actives():
    if not LIBRARY.exists():
        pytest.skip("the bundled demo library is missing")
    return [mol for mol in _read(LIBRARY) if mol.GetProp("_Name") in RING_AMIDINES]


@pytest.fixture(scope="module")
def pool():
    return _read(POOL)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_property_profile_of_hand_checked_molecules():
    """Ethanol is the arithmetic check: MW 46, one donor, one acceptor, no
    rotatable bond (the C-O bond is terminal by this project's definition)."""
    ethanol = D.property_profile(Chem.MolFromSmiles("CCO"))
    assert ethanol.MW == pytest.approx(46.07, abs=0.05)
    assert ethanol.HBD == 1 and ethanol.HBA == 1 and ethanol.RotB == 0
    assert ethanol.charge == 0
    # The rotatable-bond count is the project's own: pentane has two internal
    # C-C torsions, propane none (both its bonds end at a methyl).
    assert D.property_profile(Chem.MolFromSmiles("CCCCC")).RotB == 2
    assert D.property_profile(Chem.MolFromSmiles("CCC")).RotB == 0
    # A formal charge is a property, not a rounding error.
    assert D.property_profile(Chem.MolFromSmiles("[NH4+]")).charge == 1
    assert D.property_profile(Chem.MolFromSmiles("C(=O)[O-]")).charge == -1


def test_the_six_properties_are_the_documented_ones():
    assert D.PROPERTIES == ("MW", "LogP", "HBD", "HBA", "RotB", "charge")
    assert set(D.DEFAULT_TOLERANCES) == set(D.PROPERTIES)
    assert D.DEFAULT_TOLERANCES["charge"] == 0.0, "a charge must match exactly"


def test_property_distance_is_the_worst_tolerance_scaled_deviation():
    tolerances = {"MW": 10.0, "LogP": 1.0, "HBD": 1.0, "HBA": 1.0, "RotB": 1.0, "charge": 0.0}
    base = D.PropertyProfile(MW=100.0, LogP=1.0, HBD=2, HBA=2, RotB=1, charge=0)
    assert base.distance(base, tolerances) == 0.0
    # 5 Da off a 10 Da window is half a tolerance; 20 Da is twice it.
    assert base.distance(D.PropertyProfile(95.0, 1.0, 2, 2, 1, 0), tolerances) == pytest.approx(0.5)
    assert base.distance(D.PropertyProfile(120.0, 1.0, 2, 2, 1, 0), tolerances) == pytest.approx(2.0)
    # The *largest* deviation decides: a perfect MW does not excuse a bad LogP.
    assert base.distance(D.PropertyProfile(100.0, 3.0, 2, 2, 1, 0), tolerances) == pytest.approx(2.0)
    # charge is exact
    assert base.distance(D.PropertyProfile(100.0, 1.0, 2, 2, 1, 1), tolerances) == float("inf")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_an_analogue_is_rejected_and_an_unmatched_molecule_is_ignored():
    active = named("c1ccc(cc1)C(=N)N", "benzamidine")
    analogue = named("c1ccc(cc1)C(=N)NC", "benzamidine_methyl")   # Tanimoto 0.62
    too_big = named("C" * 40, "wax")                             # MW far outside
    good = named("Nc1ccccc1O", "2_aminophenol")                  # MW 109 vs 120
    selection = D.match_decoys([active], [analogue, too_big, good], per_active=2)
    assert selection.names() == ["2_aminophenol"]
    assert selection.rejected_similar == ["benzamidine_methyl"]
    assert selection.shortfalls == {"benzamidine": 1}
    assert any("could not supply every active" in note for note in selection.notes)


def test_selection_is_deterministic_and_seed_independent_in_its_quality(actives, pool):
    first = D.match_decoys(actives, pool, per_active=3)
    second = D.match_decoys(actives, pool, per_active=3)
    assert first.names() == second.names()
    assert first.quality()["max_abs_smd"] == second.quality()["max_abs_smd"]
    # Every selected decoy is inside the topology ceiling and inside the tolerances.
    assert all(decoy.max_similarity < 0.35 for decoy in first.decoys)
    assert all(decoy.distance <= 1.0 + 1e-9 for decoy in first.decoys)


def test_the_hard_band_requires_similarity_to_an_active(actives, pool):
    """`min_similarity` is the floor that turns 'anything unlike the actives' into
    'decoys that look like the actives but are not analogues' — the band a method
    has to be able to lose on."""
    easy = D.match_decoys(actives, pool, per_active=3)
    hard = D.match_decoys(actives, pool, per_active=3, min_similarity=0.3, max_similarity=0.6)
    assert hard.min_similarity == 0.3
    assert hard.n_decoys >= 1
    assert all(0.3 <= decoy.max_similarity < 0.6 for decoy in hard.decoys)
    assert hard.rejected_easy, "members below the floor are reported, not silently dropped"
    assert any("*hard* decoy band" in note for note in hard.notes)
    assert hard.n_decoys <= easy.n_decoys


def test_the_quality_report_shows_the_two_distributions(actives, pool):
    selection = D.match_decoys(actives, pool, per_active=5)
    quality = selection.quality()
    assert set(quality["properties"]) == set(D.PROPERTIES)
    for row in quality["properties"].values():
        assert set(row) == {"actives_mean", "actives_sd", "decoys_mean", "decoys_sd", "smd"}
    assert quality["max_abs_smd"] == pytest.approx(
        max(abs(row["smd"]) for row in quality["properties"].values()), abs=1e-4
    )
    table = selection.quality_table()
    for name in D.PROPERTIES:
        assert name in table
    assert "SMD" in table
    assert selection.quality()["matched"] == (quality["max_abs_smd"] <= 0.5)


def test_an_unmatched_decoy_set_is_reported_as_unmatched():
    """A pool that cannot match on HBD gives a large SMD, and the report says so
    instead of leaving the reader to assume the matching worked."""
    active = named("N=C(N)c1ccccc1", "benzamidine")
    poor = [named("c1ccc2ccccc2c1", "naphthalene"), named("Cc1ccccc1", "toluene")]
    selection = D.match_decoys(
        [active], poor, per_active=2, max_similarity=0.9, tolerances={"MW": 100.0, "HBD": 0.1}
    )
    # HBD is 3 for the active and 0 for both pool members: outside a 0.1 window.
    assert selection.n_decoys == 0
    assert any("no decoy at all" in note for note in selection.notes)


def test_the_bundled_pool_is_curated_and_unique(pool):
    assert len(pool) >= 100, "the pool is meant to be a real selection pool"
    canonical = {Chem.MolToSmiles(mol) for mol in pool}
    assert len(canonical) == len(pool), "no duplicate structures in the pool"
    names = [mol.GetProp("_Name") for mol in pool]
    assert len(set(names)) == len(names), "no duplicate names in the pool"
    assert all(name.strip() for name in names)
    # The pool must not contain the actives themselves.
    assert not (set(names) & set(RING_AMIDINES))


def test_decoy_set_is_json_serialisable(actives, pool):
    import json

    selection = D.match_decoys(actives, pool, per_active=2)
    payload = json.loads(json.dumps(selection.as_dict()))
    assert payload["n_decoys"] == selection.n_decoys
    assert payload["quality"]["max_abs_smd"] == selection.quality()["max_abs_smd"]
    assert payload["tolerances"]["MW"] == D.DEFAULT_TOLERANCES["MW"]


def test_match_decoys_validates_its_inputs(actives):
    with pytest.raises(ValueError, match="no active"):
        D.match_decoys([], actives)
    with pytest.raises(ValueError, match="pool is empty"):
        D.match_decoys(actives, [])
    with pytest.raises(ValueError, match="per_active"):
        D.match_decoys(actives, actives, per_active=0)
    with pytest.raises(ValueError, match="not a parsable molecule"):
        D.match_decoys(actives, ["not a molecule"])


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(D, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        D.require_rdkit()
    with pytest.raises(ImportError):
        D.property_profile(Chem.MolFromSmiles("CCO"))

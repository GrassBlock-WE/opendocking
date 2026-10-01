# SPDX-License-Identifier: GPL-3.0-or-later
"""Hit triage: structural alerts, their location, the property panel, the report.

Hand-checkable throughout: every SMARTS assertion names the molecule and the alert
it must fire, the location assertions are on molecules whose scaffold membership can
be read off the structure, and the catalogue assertions name the entry that must
match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from odock import triage as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIBRARY = ROOT / "demo" / "libraries" / "library.smi"


def named(smiles: str, name: str = ""):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if name:
        mol.SetProp("_Name", name)
    return mol


def alerts_of(smiles: str, **kwargs):
    return T.smarts_alerts(named(smiles), **kwargs)


def names_of(smiles: str, **kwargs):
    return sorted(alert.name for alert in alerts_of(smiles, **kwargs))


# ---------------------------------------------------------------------------
# SMARTS alerts, hand-checked
# ---------------------------------------------------------------------------


def test_nitrobenzene_matches_the_nitroaromatic_alert_on_the_right_atoms():
    """Nitrobenzene: the pattern is the nitro group plus the ring carbon it hangs
    from, and it is *mixed* — the nitro group is a substituent, the ring carbon is
    the scaffold."""
    alerts = alerts_of("O=[N+]([O-])c1ccccc1")
    nitro = [alert for alert in alerts if alert.name == "nitroaromatic"]
    assert len(nitro) == 1
    assert nitro[0].origin == "liability"
    assert len(nitro[0].atoms) == 4, "N, two O and the ring carbon"
    assert nitro[0].location == "mixed"


@pytest.mark.parametrize(
    "smiles, alert",
    [
        ("C=CC(=O)N", "michael_acceptor_enone"),          # acrylamide
        ("C=CC#N", "michael_acceptor_acrylonitrile"),     # acrylonitrile
        ("C=CS(=O)(=O)C", "michael_acceptor_vinyl_sulfone"),
        ("C1CO1", "epoxide"),
        ("C1CN1", "aziridine"),
        ("CC(=O)Cl", "acyl_halide"),
        ("CS(=O)(=O)Cl", "sulfonyl_halide"),
        ("CCBr", "alkyl_halide"),
        ("CC(=O)OC(C)=O", "anhydride"),
        ("O=C=Nc1ccccc1", "isocyanate"),
        ("CC=O", "aldehyde"),
        ("O=c1cc[nH]c(=O)[nH]1", "quinone") if False else ("O=C1C=CC(=O)C=C1", "quinone"),
        ("NNC", "hydrazine"),
        ("NN=Cc1ccccc1", "hydrazone"),
        ("CC(=O)NO", "hydroxamic_acid"),
        ("c1cc(O)c(O)cc1", "catechol"),
        ("Nc1ccccc1", "aniline"),
        ("CS", "thiol"),
        ("c1ccc(cc1)N=Nc1ccccc1", "azo"),
        ("C[N+](C)(C)C", "quaternary_nitrogen"),
        ("c1cc[n+]([O-])cc1", "n_oxide"),
    ],
)
def test_each_liability_class_fires_on_a_known_example(smiles, alert):
    assert alert in names_of(smiles)


def test_charged_centres_are_reported_without_flagging_a_nitro_group_twice():
    """A nitro group carries a formal N+/O- pair, but its liability is the nitro
    alert; reporting it again as a charged centre would be noise.  A genuine
    permanent charge (an ammonium, a carboxylate) is reported."""
    assert "charged_centre" not in names_of("O=[N+]([O-])c1ccccc1")
    assert "charged_centre" in names_of("C[N+](C)(C)C")
    assert "charged_centre" in names_of("CC(=O)[O-]")
    # An N-oxide is its own alert, not a bare charge.
    assert names_of("c1cc[n+]([O-])cc1") == ["n_oxide"]


def test_soft_spots_are_a_separate_set():
    """'Reactive' and 'gets metabolised quickly' are different decisions, so they
    are reported separately and the caller can ask for either."""
    anisole = "COc1ccc(CC(=O)O)cc1"
    liability = names_of(anisole, sets=["liability"])
    soft = names_of(anisole, sets=["soft_spot"])
    assert "para_methoxyphenyl" in soft
    assert "para_methoxyphenyl" not in liability
    assert names_of(anisole) == sorted(liability + soft)


def test_a_clean_molecule_reports_no_alert():
    for smiles in ("CCO", "c1ccc(cc1)CC(=O)O", "NC(=O)c1ccccc1"):
        assert alerts_of(smiles) == [], smiles


# ---------------------------------------------------------------------------
# Published catalogues
# ---------------------------------------------------------------------------


def test_a_pains_positive_molecule_is_flagged_by_the_right_catalogue():
    """Benzoquinone is the demo library's PAINS example: the PAINS catalogue must
    flag it, the entry must name a quinone, and the location must be the scaffold —
    the whole molecule is the ring system."""
    alerts = T.catalogue_alerts(named("O=C1C=CC(=O)C=C1"), catalogues=["PAINS", "NIH", "ZINC"])
    pains = [alert for alert in alerts if alert.origin == "PAINS"]
    assert pains, "the PAINS catalogue must flag a para-quinone"
    assert "quinone" in pains[0].name.lower()
    assert pains[0].location == "scaffold"
    assert len(pains[0].atoms) == 8, "the ring system"
    assert {alert.origin for alert in alerts} >= {"PAINS", "NIH"}


def test_a_clean_molecule_is_not_flagged_by_any_catalogue():
    alerts = T.catalogue_alerts(named("CCO"), catalogues=["PAINS", "BRENK", "NIH", "ZINC"])
    assert alerts == []


def test_the_catalogues_are_the_published_ones():
    assert T.CATALOGUES == ("PAINS", "PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC")
    if not T.HAVE_CATALOGUES:  # pragma: no cover - a trimmed RDKit build
        pytest.skip("this RDKit build has no FilterCatalog")
    for name in ("PAINS", "BRENK", "NIH", "ZINC"):
        alerts = T.catalogue_alerts(named("O=C1C=CC(=O)C=C1"), catalogues=[name])
        assert all(alert.origin == name for alert in alerts)


# ---------------------------------------------------------------------------
# Alert location
# ---------------------------------------------------------------------------


def test_alert_location_distinguishes_scaffold_from_substituent():
    """Three cases whose scaffold membership reads straight off the structure:

    * benzoquinone — the quinone *is* the ring system: ``scaffold``;
    * benzamidine — the amidine hangs off the benzene ring and is stripped by the
      Murcko framework, so its BRENK imine alert is on a ``substituent``;
    * warfarin — the coumarin is the fused scaffold: ``scaffold``.
    """
    quinone = T.catalogue_alerts(named("O=C1C=CC(=O)C=C1"), catalogues=["PAINS"])
    assert quinone[0].location == "scaffold"
    amidine = [alert for alert in T.catalogue_alerts(named("N=C(N)c1ccccc1"), catalogues=["BRENK"])
               if "imine" in alert.name.lower()]
    assert amidine and amidine[0].location == "substituent"
    warfarin = T.catalogue_alerts(
        named("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"), catalogues=["BRENK"]
    )
    assert any(alert.location == "scaffold" for alert in warfarin)


def test_alert_location_of_a_hand_built_pattern():
    """`alert_location` is a pure function of the matched atoms, so it can be
    checked directly: with the ring + the enone carbons as "scaffold", a pattern on
    the ring is scaffold, one on the carboxyl is substituent and one spanning both
    is mixed."""
    mol = named("OC(=O)/C=C/c1ccccc1")  # cinnamic acid
    scaffold = set(range(3, 9))  # the ring carbons of this SMILES order
    assert T.alert_location(mol, [3, 4], scaffold_atoms=scaffold) == "scaffold"
    assert T.alert_location(mol, [0, 1, 2], scaffold_atoms=scaffold) == "substituent"
    assert T.alert_location(mol, [1, 3], scaffold_atoms=scaffold) == "mixed"
    assert T.alert_location(mol, [], scaffold_atoms=scaffold) == "none"


def test_a_molecule_without_a_ring_has_only_substituents():
    """There is no core to protect in an acyclic molecule, so its alerts are
    `substituent` — the distinction is meaningless there and the tool says so
    rather than inventing a scaffold."""
    alerts = alerts_of("C=CC(=O)N")
    assert alerts and alerts[0].location == "substituent"


# ---------------------------------------------------------------------------
# Property panel
# ---------------------------------------------------------------------------


def test_panel_values_and_rules_on_hand_checked_molecules():
    panel = T.panel(named("CCO"))
    assert panel.values["MW"] == pytest.approx(46.07, abs=0.05)
    assert panel.values["HBD"] == 1 and panel.values["HBA"] == 1
    assert panel.values["TPSA"] == pytest.approx(20.23, abs=0.1)
    assert panel.values["aromatic_rings"] == 0
    assert panel.values["fraction_csp3"] == pytest.approx(1.0)
    assert 0.0 < panel.values["QED"] <= 1.0
    # Lipinski and Veber come from odock.filters, so the names are the same ones
    # `odock filter` prints, and Egan/Ghose are added here.
    assert set(panel.rules) >= {"Lipinski", "Veber", "PAINS", "Egan", "Ghose"}
    assert panel.passed is False, "ethanol fails Ghose's 160 Da lower bound"
    assert "Ghose" in panel.failing_rules()


def test_egan_rule_is_the_tpsa_and_logp_ellipsoid():
    """Hand check: the Egan verdict is exactly ``TPSA <= 131.6 and LogP <= 5.88``,
    so a very hydrophobic molecule fails it and the violation names the number."""
    panel = T.panel(named("OC(=O)CC(N)C(=O)O"))  # aspartic acid
    assert panel.rules["Egan"][0] == (
        panel.values["TPSA"] <= T.EGAN_TPSA and panel.values["LogP"] <= T.EGAN_LOGP
    )
    fatty = T.panel(named("CCCCCCCCCCCCCCCCCCCC"))  # icosane: LogP ~ 9
    assert fatty.values["LogP"] > T.EGAN_LOGP
    assert fatty.rules["Egan"][0] is False
    assert any("LogP" in breach for breach in fatty.rules["Egan"][1])


def test_ghose_bounds_are_the_published_ranges():
    assert T.GHOSE_BOUNDS["MW"] == (160.0, 480.0)
    assert T.GHOSE_BOUNDS["LogP"] == (-0.4, 5.6)
    panel = T.panel(named("CCO"))
    assert any("MW" in breach for breach in panel.rules["Ghose"][1])


def test_the_sa_score_is_reported_as_unavailable_rather_than_guessed():
    """The RDKit contrib scorer is not in most wheels.  This build either has it —
    and then the value is a float in the published 1-10 range — or it does not, and
    the panel reports None plus a note."""
    mol = named("c1ccccc1")
    value = T.sa_score(mol)
    if T.HAVE_SA_SCORE:
        assert 1.0 <= float(value) <= 10.0
    else:
        assert value is None
        report = T.triage_library([mol])
        assert any("synthetic-accessibility" in note for note in report.notes)
        assert report.molecules[0].properties.values["SA_score"] is None


# ---------------------------------------------------------------------------
# The triage report
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_molecules():
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


def test_the_report_groups_by_the_scaffold_series(demo_molecules):
    """The group column is the same grouping `odock scaffolds` computes, so a
    chemist can move between the two commands without re-deriving the series."""
    from odock.scaffold import scaffold_groups

    report = T.triage_library(demo_molecules, catalogues=["PAINS", "BRENK"])
    expected = [group.key for group in scaffold_groups(demo_molecules)]
    assert [scaffold for scaffold, _ in report.groups] == expected
    # The largest series is the eleven benzenes (the six amidines plus aspirin,
    # ibuprofen, salicylic acid, acetanilide and paracetamol — not caffeine or
    # nicotinamide, whose rings are different).
    assert report.groups[0][0] == "c1ccccc1"
    assert len(report.groups[0][1]) == 11
    assert 6 not in report.groups[0][1], "caffeine is not a benzene"
    assert report.n_molecules == 17
    summary = report.series_summary()
    assert summary[0]["n"] == 11
    assert summary[0]["n_flagged"] >= 1
    # The acyclic series does not exist here, but every index must be in a group.
    covered = {index for _, members in report.groups for index in members}
    assert covered == set(range(17))


def test_a_clean_molecule_is_kept_in_the_report_not_dropped(demo_molecules):
    report = T.triage_library(demo_molecules, catalogues=["PAINS", "BRENK"])
    ibuprofen = report.by_name("ibuprofen")
    assert ibuprofen is not None
    assert ibuprofen.clean and ibuprofen.alerts == []
    assert ibuprofen.summary() == "clean"
    assert report.n_clean == sum(1 for molecule in report.molecules if molecule.clean)
    assert report.n_clean + report.n_flagged == report.n_molecules
    assert report.n_clean >= 1
    # And the clean ones appear in the table with an empty alert cell.
    table = report.table()
    assert "clean" in table


def test_alert_rates_are_reported_and_the_breadth_of_a_catalogue_is_called_out(demo_molecules):
    """BRENK flags most of this 17-molecule library; the report must show the rate
    and say what a high rate means, instead of letting a ranking imply it."""
    report = T.triage_library(demo_molecules, catalogues=["PAINS", "BRENK", "NIH", "ZINC"])
    assert set(report.rates) == {"PAINS", "BRENK", "NIH", "ZINC"}
    assert report.rates["PAINS"]["n"] == 17
    assert report.rates["PAINS"]["fraction"] == pytest.approx(
        report.rates["PAINS"]["flagged"] / 17
    )
    assert 0.0 <= report.rates["PAINS"]["fraction"] <= 1.0
    if report.rates["BRENK"]["fraction"] > 0.25:
        assert any("more than a quarter" in note for note in report.notes)
    rates = T.alert_rates(demo_molecules, catalogues=["PAINS"])
    assert rates["PAINS"]["flagged"] == report.rates["PAINS"]["flagged"]
    assert "smarts:liability" in T.alert_rates(demo_molecules, catalogues=["PAINS"])


def test_the_report_carries_the_honesty_statement(demo_molecules):
    report = T.triage_library(demo_molecules)
    joined = " ".join(report.notes)
    assert "not predictions" in joined
    assert "absence of an alert is not safety" in joined
    assert "substitutes for an assay" in joined
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["n_flagged"] == report.n_flagged
    assert payload["molecules"][0]["alerts"] is not None
    assert "rates" in payload and "series" in payload


def test_triage_molecule_reports_affinity_and_alerts(demo_molecules):
    triaged = T.triage_molecule(
        demo_molecules[0], index=3, affinity=-5.9, catalogues=["PAINS", "BRENK"]
    )
    assert triaged.index == 3 and triaged.name == "benzamidine"
    assert triaged.affinity == -5.9
    assert triaged.scaffold == "c1ccccc1"
    assert triaged.alerts and not triaged.clean
    assert any(alert.location == "substituent" for alert in triaged.alerts)


def test_triage_library_rejects_an_unparsable_molecule():
    with pytest.raises(ValueError, match="not a parsable molecule"):
        T.triage_library(["CCO", "not a molecule"])


def test_require_rdkit_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setattr(T, "_HAVE_RDKIT", False)
    with pytest.raises(ImportError, match=r"pip install rdkit"):
        T.require_rdkit()
    with pytest.raises(ImportError):
        T.catalogue_alerts(named("CCO"))

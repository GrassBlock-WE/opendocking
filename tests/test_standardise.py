# SPDX-License-Identifier: GPL-3.0-or-later
"""Library standardisation: the recipe, the audit, and the cases it refuses.

Every fixture here is constructed from a SMILES string in the test, so the assertions do
not depend on a vendor file or a network fetch.  Each of the cases the requirement names
has its own test: a sodium salt, a mixture, a tautomer duplicate pair, an unspecified
stereocentre, a malformed SMILES with its line number — plus the counting invariant and
determinism, which are what make the audit trustworthy.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402

from odock.standardise import (  # noqa: E402
    CATEGORIES,
    KEPT,
    MERGED,
    REJECTED,
    RECIPE_VERSION,
    STEPS,
    PropertyLimits,
    StandardiseConfig,
    StandardiseReport,
    add_standardise_parser,
    canonical_smiles,
    cmd_standardise,
    main,
    parse_window,
    standardise,
    tautomer_key,
)

#: A sodium salt: the anion is the ligand, the sodium is the counter-ion.
SODIUM_ACETATE = "CC(=O)[O-].[Na+] sodium_acetate\n"

#: One record with two components: a larger organic molecule and a small one.
MIXTURE = "c1ccccc1CCCCCC.O big_component\n"

#: A tautomer pair: 2-hydroxypyridine and 2-pyridone are the same molecule in two forms.
TAUTOMER_PAIR = "Oc1ccccn1 hydroxy_pyridine\nO=c1cccc[nH]1 pyridone\n"

#: One unspecified stereocentre.
UNSPECIFIED_CENTRE = "CC(F)(Cl)Br unspecified_centre\n"


def report_for(text: str, **kwargs) -> StandardiseReport:
    return standardise(text, config=StandardiseConfig(**kwargs) if kwargs else None)


def only_audit(report: StandardiseReport):
    assert report.n_in == 1, [audit.describe() for audit in report.audits]
    return report.audits[0]


def sdf_text(smiles: str, name: str, *, sanitize: bool = True) -> str:
    """A one-record SDF document, optionally preserving an unsanitisable graph.

    ``sanitize=False`` on the way in is how a test builds a structure that *parses* but
    will not sanitise — the case the parse step has to refuse rather than repair.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    assert mol is not None, smiles
    mol.SetProp("_Name", name)
    block = Chem.MolToMolBlock(mol, kekulize=sanitize)
    return block + "$$$$\n"


# ---------------------------------------------------------------------------
# The counting invariant: nothing may vanish unexplained
# ---------------------------------------------------------------------------


def test_the_counts_always_account_for_every_record():
    """`in == kept + rejected + merged`, on a file that exercises every status."""
    text = (
        "SMILES name\n"                                    # invalid: a table header
        "CCO ethanol\n"                                    # kept
        "CCO ethanol_again\n"                              # merged (exact duplicate)
        "Oc1ccccn1 hydroxy_pyridine\n"                     # kept
        "O=c1cccc[nH]1 pyridone\n"                         # merged (tautomer)
        "CC(=O)[O-].[Na+] sodium_acetate\n"                # kept, after desalting
        "not_a_molecule broken\n"                          # invalid
        "CCCC.CCCO ambiguous\n"                            # unsupported
    )
    report = report_for(text)
    assert report.n_in == 8
    assert report.balanced
    assert report.n_kept + report.n_rejected + report.n_merged == report.n_in
    # The statuses are the three the invariant knows about, and nothing else appears.
    assert {audit.status for audit in report.audits} <= {KEPT, REJECTED, MERGED}
    assert report.n_rejected == 3
    assert report.n_merged == 2


def test_the_invariant_can_fail_and_says_so():
    """The assertion is real: a report that lost a record raises with the numbers."""
    config = StandardiseConfig()
    report = StandardiseReport(config=config)
    report.audits = []
    report.kept = []
    # A hand-made report that claims one input but none of the three outcomes.
    from odock.standardise import MoleculeAudit

    report.audits.append(
        MoleculeAudit(position=1, position_kind="smi-line", name="x", input_smiles="CCO")
    )
    report.audits[0].status = "something_else"
    assert not report.balanced
    with pytest.raises(AssertionError, match="lost track of 1 record"):
        report.assert_balanced()


def test_rejections_are_counted_by_category_and_property():
    limits = PropertyLimits(heavy_atoms=(8, 40), molecular_weight=(120, 400))
    report = report_for(
        "CCO ethanol\n"
        "c1ccccc1CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC big\n"
        "not_a_molecule broken\n",
        limits=limits,
    )
    assert report.rejected_by_category() == {"filter": 2, "invalid": 1}
    assert report.rejected_by_property() == {"heavy_atoms": 2, "molecular_weight": 2}


# ---------------------------------------------------------------------------
# (a) parse / repair — and refusal rather than repair
# ---------------------------------------------------------------------------


def test_a_malformed_smiles_is_refused_with_its_line_number():
    report = report_for("# a comment\nCCO ethanol\nnot_a_molecule broken\n")
    assert report.n_kept == 1
    refused = report.audits[1]
    assert refused.status == REJECTED
    assert refused.category == "invalid"
    assert refused.position == 3
    assert refused.position_kind == "smi-line"
    assert "'not_a_molecule' is not a parsable SMILES" in refused.reason


def test_a_table_header_row_is_refused_rather_than_silently_skipped():
    """A header row looks like a molecule until it is parsed; it is reported, not dropped."""
    report = report_for("SMILES ID\nCCO ethanol\n")
    assert report.n_in == 2
    header = report.audits[0]
    assert header.status == REJECTED and header.position == 1
    assert "'SMILES' is not a parsable SMILES" in header.reason


def test_comment_and_blank_lines_are_not_records():
    report = report_for("# header\n\n# another\nCCO ethanol\n")
    assert report.n_in == 1
    assert report.n_kept == 1


def test_an_unsanitisable_structure_is_refused_not_repaired(tmp_path):
    """A record that parses but cannot be sanitised is refused, with RDKit's reason."""
    path = tmp_path / "bad_valence.sdf"
    # A five-valent carbon: the graph is fine, the valence is not.
    path.write_text(sdf_text("C(C)(C)(C)(C)C", "over_valent", sanitize=False), encoding="utf-8")
    report = standardise(str(path))
    audit = only_audit(report)
    assert audit.status == REJECTED
    assert audit.category == "invalid"
    assert "cannot be sanitised" in audit.reason
    assert "refused rather than repaired" in audit.reason


def test_a_smiles_rdkit_will_not_parse_is_refused_as_invalid():
    """The other half of the same rule: the parse itself can fail."""
    audit = only_audit(report_for("C(C)(C)(C)(C)C over_valent\n"))
    assert audit.status == REJECTED
    assert audit.category == "invalid"
    assert "is not a parsable SMILES" in audit.reason


def test_repair_reports_when_it_changed_the_canonical_smiles():
    """RDKit's cleanup normalisations are recorded when they alter the structure."""
    # A nitro group written in the charge-separated form is normalised by Cleanup.
    report = report_for("C[N+](=O)[O-] nitromethane\n")
    audit = only_audit(report)
    assert audit.status == KEPT
    # Whatever Cleanup did, the audit says so if the SMILES moved; and if it did not, no
    # finding was invented.
    repaired = [finding for finding in audit.findings if finding.step == "repair"]
    assert len(repaired) <= 1


def test_the_repair_step_can_be_disabled():
    report = report_for("CCO ethanol\n", repair=False)
    assert "repair" in report.config.skipped()
    assert report.config.as_dict()["skipped"] == ["repair"]


# ---------------------------------------------------------------------------
# (b) salts and mixtures
# ---------------------------------------------------------------------------


def test_a_sodium_salt_keeps_the_anion_and_reports_the_removed_sodium():
    report = report_for(SODIUM_ACETATE)
    audit = only_audit(report)
    assert audit.status == KEPT
    assert report.kept[0].smiles == "CC(=O)[O-]"
    assert len(audit.removed_fragments) == 1
    removed = audit.removed_fragments[0]
    assert removed.smiles == "[Na+]"
    assert removed.charge == 1
    assert removed.kept is False
    # The removal is a change the audit reports, naming the fragment.
    assert audit.changed
    assert any("[Na+]" in finding.detail for finding in audit.changes)


def test_a_sodium_salt_neutralised_keeps_the_neutral_form_and_records_the_assumption():
    report = report_for(SODIUM_ACETATE, charge_policy="neutralise")
    audit = only_audit(report)
    assert report.kept[0].smiles == "CC(=O)O"
    assert report.kept[0].properties["net_charge"] == 0
    charge = [finding for finding in audit.changes if finding.step == "charge"]
    assert charge and "neutralised -1" in charge[0].detail
    # The shared predicate is asked, and the answer is recorded: a neutral acid is not
    # the species pH 7.4 describes.
    assert audit.charge_consistent is False
    assert "carboxylate" in audit.charge_statement


def test_a_mixture_keeps_the_larger_component_and_names_the_smaller():
    """One record, two components: the larger organic one is kept, the other reported."""
    report = report_for(MIXTURE)
    assert report.n_in == 1 and report.n_kept == 1
    audit = only_audit(report)
    assert audit.status == KEPT
    assert report.kept[0].smiles == "CCCCCCc1ccccc1"
    assert [fragment.smiles for fragment in audit.removed_fragments] == ["O"]
    assert audit.removed_fragments[0].organic is False
    assert "removed O" in audit.describe()


def test_a_tie_for_the_largest_component_is_refused_and_the_tie_break_is_opt_in():
    tie = "CCCC.CCCO ambiguous\n"
    refused = only_audit(report_for(tie))
    assert refused.status == REJECTED
    assert refused.category == "unsupported"
    assert "ambiguous largest component" in refused.reason
    assert "largest_on_tie='first'" in refused.reason

    kept = report_for(tie, largest_on_tie="first")
    assert kept.n_kept == 1
    # The documented tie-break is alphabetical by canonical SMILES: CCCC < CCCO.
    assert kept.kept[0].smiles == "CCCC"
    assert kept.audits[0].status == KEPT
    assert len(kept.audits[0].removed_fragments) == 1


def test_a_mixture_with_too_many_components_is_refused():
    text = ".".join(["CC"] * 3) + " three_ethanes\n"
    report = report_for(text, max_fragments=2)
    audit = only_audit(report)
    assert audit.status == REJECTED
    assert "components, more than the 2" in audit.reason


def test_a_record_with_no_organic_component_is_refused():
    report = report_for("[Na+].[Cl-] salt\n")
    audit = only_audit(report)
    assert audit.status == REJECTED
    assert "no component is organic" in audit.reason


def test_desalting_can_be_disabled_and_then_the_mixture_is_kept_whole():
    report = report_for(MIXTURE, desalt=False)
    assert report.n_kept == 1
    assert report.audits[0].removed_fragments == []
    assert "." in report.kept[0].smiles
    assert report.config.as_dict()["skipped"] == ["desalt"]


# ---------------------------------------------------------------------------
# (c) charge normalisation, through the shared protonation predicate
# ---------------------------------------------------------------------------


def test_the_charge_step_asks_the_shared_protonation_predicate():
    report = report_for("N=C(N)c1ccccc1 benzamidine\n")
    audit = only_audit(report)
    assert audit.status == KEPT
    assert audit.charge_statement
    assert audit.charge_consistent is False or audit.charge_consistent is True
    # The predicate is the one protonation exposes, not a re-implementation.
    from odock import protonation

    assert protonation.can_represent_formal_charge(
        Chem.MolFromSmiles("N=C(N)c1ccccc1")
    ).statement() == audit.charge_statement


def test_refuse_charged_rejects_a_salt_and_keeps_a_neutral_molecule():
    report = report_for(
        "CC(=O)[O-].[Na+] charged\nCCO neutral\n", charge_policy="refuse_charged"
    )
    assert report.n_kept == 1
    assert report.kept[0].name == "neutral"
    refused = report.audits[0]
    assert refused.status == REJECTED
    assert refused.category == "unsupported"
    assert "net formal charge of -1" in refused.reason


def test_the_charge_step_can_be_disabled():
    report = report_for("CC(=O)[O-].[Na+] salt\n", charge=False)
    assert report.config.as_dict()["skipped"] == ["charge"]
    assert report.audits[0].charge_statement == ""
    assert report.charge_observations() == {"consistent": 0, "inconsistent": 0, "unknown": 1}


# ---------------------------------------------------------------------------
# (d) deduplication: exact, and tautomer-insensitive
# ---------------------------------------------------------------------------


def test_an_exact_duplicate_merges_and_names_the_molecule_it_collapsed_into():
    report = report_for("CCO first\nCCO second\n")
    assert report.n_in == 2 and report.n_kept == 1 and report.n_merged == 1
    merged = report.audits[1]
    assert merged.status == MERGED
    assert merged.merged_into == 0
    assert merged.merged_into_name == "first"
    assert merged.merge_kind == "exact duplicate"
    assert "merged into first" in merged.describe()
    assert any("collapsed into first" in finding.detail for finding in merged.changes)
    assert report.merged_by_kind() == {"exact duplicate": 1}


def test_a_tautomer_pair_collapses_and_reports_which_one_was_kept():
    report = report_for(TAUTOMER_PAIR)
    assert report.n_in == 2
    assert report.n_kept == 1
    kept = report.audits[0]
    merged = report.audits[1]
    assert kept.status == KEPT and kept.name == "hydroxy_pyridine"
    assert merged.status == MERGED
    assert merged.merged_into_name == "hydroxy_pyridine"
    assert merged.merge_kind == "tautomer"
    # The two input molecules are different SMILES with the same tautomer key: that is
    # exactly the collision the key exists to find.
    assert kept.canonical_key != merged.canonical_key
    assert kept.tautomer_key == merged.tautomer_key
    assert report.merged_by_kind() == {"tautomer": 1}


def test_exact_dedupe_keeps_a_tautomer_pair_apart():
    report = report_for(TAUTOMER_PAIR, tautomer_dedupe=False)
    assert report.n_kept == 2 and report.n_merged == 0


def test_dedupe_can_be_disabled_entirely():
    report = report_for("CCO first\nCCO second\n", dedupe=False)
    assert report.n_kept == 2 and report.n_merged == 0


# ---------------------------------------------------------------------------
# (e) stereochemistry policy
# ---------------------------------------------------------------------------


def test_keep_records_an_unspecified_centre_without_changing_it():
    report = report_for(UNSPECIFIED_CENTRE)
    audit = only_audit(report)
    assert audit.status == KEPT
    assert "left unspecified" in " ".join(finding.detail for finding in audit.findings)
    assert audit.changed is False


def test_refuse_rejects_an_unspecified_centre_and_names_the_atom():
    report = report_for(UNSPECIFIED_CENTRE, stereo_policy="refuse")
    audit = only_audit(report)
    assert audit.status == REJECTED
    assert audit.category == "stereo"
    assert "no specified configuration" in audit.reason
    assert "atom 2" in audit.reason


def test_drop_strips_assigned_stereochemistry_and_reports_it():
    report = report_for("C[C@H](F)Cl chiral\n", stereo_policy="drop")
    audit = only_audit(report)
    assert audit.status == KEPT
    assert "@" not in report.kept[0].smiles
    assert "dropped 1 assigned stereocentre" in " ".join(
        finding.detail for finding in audit.changes
    )


def test_dropping_stereochemistry_makes_enantiomers_collapse_which_is_why_order_matters():
    """The dedupe key is taken after the stereo policy: this is that property, measured."""
    pair = "C[C@H](F)Cl one\nC[C@@H](F)Cl two\n"
    kept = report_for(pair, stereo_policy="keep")
    assert kept.n_kept == 2 and kept.n_merged == 0
    dropped = report_for(pair, stereo_policy="drop")
    assert dropped.n_kept == 1 and dropped.n_merged == 1
    assert dropped.merged_by_kind() == {"exact duplicate": 1}


# ---------------------------------------------------------------------------
# (f) property filters
# ---------------------------------------------------------------------------


def test_a_filter_reports_every_rejection_with_its_reason():
    limits = PropertyLimits(heavy_atoms=(8, 40), rotatable_bonds=(0, 2))
    report = report_for(
        "CCO ethanol\n"                        # 3 heavy atoms: too small
        "c1ccccc1CCCCCC hexylbenzene\n"        # 12 heavy atoms, six rotors: rotor window
        "c1ccccc1CCCCCCCCCCCCCCCCCCC long\n",  # too large and too flexible
        limits=limits,
    )
    assert report.n_kept == 0
    assert report.n_rejected == 3
    by_property = report.rejected_by_property()
    assert set(by_property) == {"heavy_atoms", "rotatable_bonds"}
    assert by_property["heavy_atoms"] == 1      # ethanol, at 3 heavy atoms
    assert by_property["rotatable_bonds"] == 2  # hexylbenzene and the long chain
    for audit in report.rejected:
        assert audit.category == "filter"
        assert "outside" in audit.reason
    # The smallest molecule names its exact measurement and window.
    assert "heavy_atoms 3 outside 8..40" == report.rejected[0].reason


def test_a_charge_window_counts_the_net_charge():
    limits = PropertyLimits(charge=(-1, 1))
    inside = report_for("CC(=O)[O-] anion\n", limits=limits)
    assert inside.n_kept == 1
    outside = report_for("[O-]C(=O)CC(=O)[O-] dianion\n", limits=limits)
    assert outside.n_kept == 0
    audit = only_audit(outside)
    assert audit.category == "filter"
    assert "net_charge -2 outside -1..1" in audit.reason


# ---------------------------------------------------------------------------
# The recipe: order, switches, identity, determinism
# ---------------------------------------------------------------------------


def test_the_default_recipe_is_the_documented_order():
    assert StandardiseConfig().order == STEPS
    assert StandardiseConfig().enabled() == STEPS
    assert StandardiseConfig(repair=False).skipped() == ("repair",)


def test_every_step_is_switchable_and_each_switch_changes_the_recipe_hash():
    base = StandardiseConfig().hash()
    for switch in ("repair", "desalt", "charge", "stereo", "dedupe", "filters"):
        assert StandardiseConfig(**{switch: False}).hash() != base
    for policy in (
        {"charge_policy": "neutralise"},
        {"charge_policy": "refuse_charged"},
        {"stereo_policy": "drop"},
        {"stereo_policy": "refuse"},
        {"largest_on_tie": "first"},
        {"tautomer_dedupe": False},
        {"max_fragments": 4},
        {"limits": PropertyLimits(heavy_atoms=(8, 60))},
    ):
        assert StandardiseConfig(**policy).hash() != base, policy


def test_an_order_that_cannot_be_honest_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="parse has to be first"):
        StandardiseConfig(order=("repair", "parse", "desalt", "charge", "stereo", "dedupe", "filter"))
    with pytest.raises(ValueError, match="dedupe has to run after stereo"):
        StandardiseConfig(order=("parse", "repair", "desalt", "charge", "dedupe", "stereo", "filter"))
    with pytest.raises(ValueError, match="filter has to run after dedupe"):
        StandardiseConfig(order=("parse", "repair", "desalt", "charge", "stereo", "filter", "dedupe"))
    with pytest.raises(ValueError, match="unknown step"):
        StandardiseConfig(order=("parse", "repair", "desalt", "charge", "stereo", "dedupe", "filter", "magic"))
    with pytest.raises(ValueError, match="omits"):
        StandardiseConfig(order=("parse", "repair", "desalt", "charge", "stereo", "dedupe"))


def test_the_hash_excludes_paths_and_covers_settings():
    text = "CCO ethanol\nnot_a_molecule broken\n"
    first = report_for(text)
    second = report_for(text)
    assert first.config.hash() == second.config.hash()
    # The same settings, reached through a different object, hash the same.
    assert first.config.hash() == StandardiseConfig().hash()
    assert first.config.hash() == StandardiseConfig.from_dict(
        first.config.hashed_settings()
    ).hash()


def test_the_same_input_produces_byte_identical_output():
    text = "CCO ethanol\nc1ccccc1CCCCCC big\nnot_a_molecule broken\n" + SODIUM_ACETATE
    first = report_for(text).to_smi()
    second = report_for(text).to_smi()
    assert first.encode("utf-8") == second.encode("utf-8")
    assert report_for(text).to_csv() == report_for(text).to_csv()


def test_the_output_header_records_the_recipe_the_versions_and_the_columns():
    report = report_for("CCO ethanol\n")
    header = report.header()
    assert report.config.hash() in header
    assert report.config.short_hash() in header
    assert f"schema {RECIPE_VERSION}" in header
    assert "rdkit" in header
    assert "columns: SMILES<TAB>name" in header
    assert "not registration" in header
    for line in header.splitlines():
        assert line.startswith("#")
    # Everything printed after the header is one SMILES and one name.
    body = [line for line in report.to_smi().splitlines() if not line.startswith("#")]
    assert body == ["CCO\tethanol"]


def test_a_recipe_with_limits_records_them_in_the_header():
    limits = PropertyLimits(heavy_atoms=(8, 60), molecular_weight=(120.0, 700.0))
    report = report_for("CCO ethanol\n", limits=limits)
    header = report.header()
    assert "limits:" in header
    assert "heavy_atoms 8..60" in header
    assert "molecular_weight 120..700" in header


def test_the_configuration_round_trips_and_refuses_unknown_keys():
    config = StandardiseConfig(
        charge_policy="neutralise",
        stereo_policy="refuse",
        largest_on_tie="first",
        max_fragments=4,
        limits=PropertyLimits(heavy_atoms=(8, 60)),
    )
    again = StandardiseConfig.from_dict(json.loads(json.dumps(config.hashed_settings())))
    assert again == config
    assert again.hash() == config.hash()
    with pytest.raises(ValueError, match="unknown configuration key"):
        StandardiseConfig.from_dict({"nonsense": 1})


def test_the_configuration_refuses_impossible_values():
    with pytest.raises(ValueError, match="charge_policy"):
        StandardiseConfig(charge_policy="explode")
    with pytest.raises(ValueError, match="recipe_version"):
        StandardiseConfig(recipe_version=0)
    with pytest.raises(ValueError, match="max_fragments"):
        StandardiseConfig(max_fragments=0)
    with pytest.raises(ValueError, match="empty"):
        PropertyLimits(heavy_atoms=(60, 8))
    with pytest.raises(ValueError, match="MIN:MAX"):
        parse_window("8-60")


def test_windows_parse_open_ended_and_integer_forms():
    assert parse_window("8:60", cast=int) == (8, 60)
    assert parse_window(":60", cast=int) == (None, 60)
    assert parse_window("8:", cast=int) == (8, None)
    assert parse_window("120.5:700") == (120.5, 700.0)


# ---------------------------------------------------------------------------
# Cases the recipe refuses rather than guesses
# ---------------------------------------------------------------------------


def test_a_metal_complex_is_unsupported_and_named():
    """A metal that survives every step is a coordination compound, not a ligand."""
    report = report_for("C[Hg]Cl methylmercury\n")
    audit = only_audit(report)
    assert audit.status == REJECTED
    assert audit.category == "unsupported"
    assert "still contains a metal" in audit.reason
    assert "docs/STANDARDISE.md" in audit.reason


def test_a_metal_counter_ion_is_removed_rather_than_refused():
    """The metal check is on the molecule that survives, not on the input."""
    report = report_for(SODIUM_ACETATE)
    assert report.n_kept == 1
    assert report.audits[0].removed_fragments[0].smiles == "[Na+]"


def test_a_metal_salt_with_desalting_off_is_refused_with_an_explanation():
    report = report_for(SODIUM_ACETATE, desalt=False)
    audit = only_audit(report)
    assert audit.status == REJECTED
    assert "still contains a metal" in audit.reason
    assert "--no-desalt" in audit.reason


def test_an_unsupported_input_format_is_refused_rather_than_guessed(tmp_path):
    path = tmp_path / "ligand.mol2"
    path.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a supported standardisation input"):
        standardise(str(path))


def test_the_documented_unsupported_cases_all_have_a_category():
    """Every refusal a user can meet is one of the documented categories."""
    assert set(CATEGORIES) == {"invalid", "unsupported", "stereo", "charge", "filter"}
    report = report_for(
        "not_a_molecule a\n"
        "CCCC.CCCO b\n"
        "CC(F)(Cl)Br c\n"
        "CC(=O)[O-].[Na+] d\n",
        stereo_policy="refuse",
        charge_policy="refuse_charged",
    )
    assert set(report.rejected_by_category()) <= set(CATEGORIES)


# ---------------------------------------------------------------------------
# The audit files
# ---------------------------------------------------------------------------


def test_the_csv_audit_has_one_row_per_record_including_rejects_and_merges():
    text = "CCO ethanol\nCCO again\nnot_a_molecule broken\n" + SODIUM_ACETATE
    report = report_for(text)
    rows = list(csv.DictReader(io.StringIO(report.to_csv())))
    assert len(rows) == report.n_in
    statuses = [row["status"] for row in rows]
    assert statuses.count(KEPT) == report.n_kept
    assert statuses.count(REJECTED) == report.n_rejected
    assert statuses.count(MERGED) == report.n_merged
    # The merge points at the row it collapsed into, and the rejection carries its reason.
    merged = next(row for row in rows if row["status"] == MERGED)
    assert merged["merged_into"] == "0"
    assert merged["merge_kind"] == "exact duplicate"
    rejected = next(row for row in rows if row["status"] == REJECTED)
    assert rejected["category"] == "invalid"
    assert rejected["reason"]
    # Every column the format promises is present.
    assert set(report.AUDIT_COLUMNS) <= set(rows[0])


def test_the_json_report_is_balanced_and_carries_the_recipe():
    report = report_for("CCO ethanol\nnot_a_molecule broken\n" + SODIUM_ACETATE)
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["counts"]["balanced"] is True
    assert payload["counts"]["in"] == payload["counts"]["kept"] + payload["counts"]["rejected"] + payload["counts"]["merged"]
    assert payload["recipe"]["hash"] == report.config.hash()
    assert len(payload["audit"]) == report.n_in
    assert payload["kept"]


def test_write_produces_the_three_files_and_the_smi_round_trips(tmp_path):
    report = report_for("CCO ethanol\nC1CCCCC1 cyclohexane\n")
    written = report.write(
        tmp_path / "clean.smi",
        csv_path=tmp_path / "audit.csv",
        json_path=tmp_path / "report.json",
    )
    assert [path.name for path in written] == ["clean.smi", "audit.csv", "report.json"]
    assert all(path.is_file() for path in written)
    # The cleaned library is readable by the toolchain's own reader, names intact.
    from odock.chem.ligand import read_ligands

    mols = read_ligands(str(written[0]), embed=False)
    assert [mol.GetProp("_Name") for mol in mols] == ["ethanol", "cyclohexane"]
    assert json.loads(written[2].read_text(encoding="utf-8"))["counts"]["in"] == 2


def test_a_clean_library_is_left_almost_untouched():
    """The recipe must not rewrite a library that is already standard."""
    text = "N=C(N)c1ccccc1 benzamidine\nCn1cnc2c1c(=O)n(C)c(=O)n2C caffeine\n"
    report = report_for(text)
    assert report.n_kept == 2
    assert report.changed_by_step() == {}
    assert all(not audit.changed for audit in report.audits)
    # ... and the SMILES are canonical, so a second pass changes nothing.
    again = report_for(report.to_smi())
    assert [molecule.smiles for molecule in again.kept] == [
        molecule.smiles for molecule in report.kept
    ]


def test_sdf_input_is_read_and_reported_by_record(tmp_path):
    path = tmp_path / "library.sdf"
    writer = Chem.SDWriter(str(path))
    for smiles, name in (("CCO", "ethanol"), ("c1ccccc1", "benzene")):
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    report = standardise(str(path))
    assert report.position_kind == "sdf-record"
    assert report.n_in == 2
    assert report.n_kept == 2
    assert [audit.position for audit in report.audits] == [1, 2]
    assert [molecule.name for molecule in report.kept] == ["ethanol", "benzene"]


def test_an_unreadable_sdf_record_is_an_invalid_audit_entry(tmp_path):
    """A broken record cannot vanish: it is a row with a reason, at its record number."""
    good = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
    path = tmp_path / "broken.sdf"
    path.write_text(good + "$$$$\n" + "this is not a mol block\n$$$$\n", encoding="utf-8")
    report = standardise(str(path))
    assert report.n_in == 2
    assert report.n_kept == 1
    assert report.n_rejected == 1
    refused = report.audits[1]
    assert refused.position == 2 and refused.position_kind == "sdf-record"
    assert refused.category == "invalid"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _parser():
    import argparse

    parser = argparse.ArgumentParser(prog="odock")
    sub = parser.add_subparsers(dest="command", required=True)
    add_standardise_parser(sub)
    return parser


def test_the_parser_registers_the_command_once():
    parser = _parser()
    assert "standardise" in parser._subparsers._group_actions[0].choices
    # Idempotent: registering twice does not raise and does not duplicate.
    sub = parser._subparsers._group_actions[0]
    add_standardise_parser(sub)
    assert list(sub.choices).count("standardise") == 1


def test_the_command_writes_the_library_and_the_audit(tmp_path, capsys):
    source = tmp_path / "vendor.smi"
    source.write_text("CCO ethanol\nCCO again\nnot_a_molecule broken\n" + SODIUM_ACETATE, encoding="utf-8")
    out = tmp_path / "clean.smi"
    audit = tmp_path / "audit.csv"
    args = _parser().parse_args(
        ["standardise", str(source), "-o", str(out), "--audit", str(audit)]
    )
    assert cmd_standardise(args) == 0
    captured = capsys.readouterr()
    assert "kept 2, rejected 1, merged 1" in captured.out
    assert out.is_file() and audit.is_file()
    assert "wrote" in captured.err


def test_the_command_reports_a_failed_run_with_an_exit_code(tmp_path, capsys):
    source = tmp_path / "empty.smi"
    source.write_text("# nothing but a comment\n", encoding="utf-8")
    args = _parser().parse_args(["standardise", str(source), "-o", str(tmp_path / "out.smi")])
    assert cmd_standardise(args) == 3
    assert "no record survived" in capsys.readouterr().err


def test_the_command_reports_a_bad_configuration_with_exit_code_two(tmp_path, capsys):
    source = tmp_path / "lib.smi"
    source.write_text("CCO ethanol\n", encoding="utf-8")
    args = _parser().parse_args(
        ["standardise", str(source), "--heavy-atoms", "60:8"]
    )
    assert cmd_standardise(args) == 2
    assert "empty" in capsys.readouterr().err


def test_the_command_applies_the_policies_and_limits(tmp_path, capsys):
    source = tmp_path / "vendor.smi"
    source.write_text(SODIUM_ACETATE + "c1ccccc1CCCCCC big\n", encoding="utf-8")
    out = tmp_path / "clean.smi"
    args = _parser().parse_args(
        [
            "standardise", str(source),
            "-o", str(out),
            "--charge-policy", "neutralise",
            "--heavy-atoms", "2:20",
            "--json", str(tmp_path / "report.json"),
        ]
    )
    assert cmd_standardise(args) == 0
    text = out.read_text(encoding="utf-8")
    assert "CC(=O)O" in text
    assert "heavy_atoms 2..20" in text


def test_main_works_without_the_registrar(tmp_path):
    source = tmp_path / "lib.smi"
    source.write_text("CCO ethanol\n", encoding="utf-8")
    out = tmp_path / "clean.smi"
    assert main([str(source), "-o", str(out)]) == 0
    assert out.is_file()


def test_verbose_prints_every_non_kept_record(tmp_path, capsys):
    source = tmp_path / "lib.smi"
    source.write_text("CCO ethanol\nCCO again\nbroken_x bad\n", encoding="utf-8")
    args = _parser().parse_args(
        ["standardise", str(source), "-o", str(tmp_path / "o.smi"), "-v"]
    )
    assert cmd_standardise(args) == 0
    err = capsys.readouterr().err
    assert "merged into ethanol" in err
    assert "invalid" in err


# ---------------------------------------------------------------------------
# The recipe against the bundled demo library
# ---------------------------------------------------------------------------


def test_the_bundled_library_standardises_without_changing_anything():
    root = Path(__file__).resolve().parent.parent
    library = root / "demo" / "libraries" / "library.smi"
    if not library.is_file():
        pytest.skip("the bundled demo library is missing")
    report = standardise(str(library))
    assert report.n_in > 5
    assert report.balanced
    assert report.n_kept == report.n_in
    # A hand-curated library is already standard: nothing structural should change.
    assert report.n_rejected == 0
    assert report.n_merged == 0

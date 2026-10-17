# SPDX-License-Identifier: GPL-3.0-or-later
"""The panel: refusal rules, selectivity arithmetic, and one real run.

Most of this file is deliberately *not* about docking.  The part of a panel
worth getting wrong silently is the arithmetic — the gap to the second best, the
count inside the window, and whether a ligand's best target is its own protein —
and those can be tested on hand-made cells in microseconds.  The one integration
test at the bottom proves that the orchestration over ``screen``,
``consensus``, ``analysis`` and ``metrics`` produces the same shape of data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import pytest

from odock import panel as panel_module
from odock.panel import (
    APO_MEMBER,
    BOX_POLICIES,
    BUNDLED_LIGANDS,
    BUNDLED_MEMBERS,
    PanelCell,
    PanelError,
    PanelMember,
    PanelProtocol,
    box_table,
    build_panel,
    bundled_ligands,
    bundled_panel,
    derive_boxes,
    hit_rate,
    matrix,
    selectivity_profile,
    split_crystal,
    table,
)

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"


# ---------------------------------------------------------------------------
# The bundled panel
# ---------------------------------------------------------------------------


def test_the_bundled_panel_is_seven_crystal_structures():
    panel = build_panel()
    assert panel.receptors == (
        "3PTB", "1STP", "3ERT", "1M17", "1HVR", "1ERE_A", "1HXW",
    )
    assert panel.protocol.box_policy == "reference"
    for member in panel.members:
        assert member.ligand, member.pdb_id
        assert (DATA / f"{member.pdb_id}.pdb").exists(), member.pdb_id


def test_the_bundled_ligands_each_have_a_home_in_the_panel():
    panel = build_panel()
    ligands = bundled_ligands()
    assert len(ligands) == 7
    for name, ligand in ligands.items():
        assert ligand.home, name
        for home in ligand.home:
            assert home in panel.receptors, (name, home)
        assert ligand.protein, name


def test_the_bundled_ligand_smiles_are_real_molecules():
    """Heavy-atom counts are the hand-checkable part of a SMILES."""
    from rdkit import Chem

    expected = {"BEN": 9, "BTN": 16, "EST": 20, "AQ4": 29, "OHT": 29, "XK2": 46, "RIT": 50}
    for name, ligand in BUNDLED_LIGANDS.items():
        mol = Chem.MolFromSmiles(ligand.smiles)
        assert mol is not None, name
        assert mol.GetNumHeavyAtoms() == expected[name], (name, mol.GetNumHeavyAtoms())


@pytest.mark.parametrize("name", ["BTN", "EST", "OHT", "XK2", "RIT"])
def test_the_bundled_smiles_match_the_ccd_files(name):
    """For every ligand with a CCD ideal file, the literal has to be that molecule.

    ``BEN`` and ``AQ4`` have no SDF in ``tests/data``; the rest do, so a typo in a
    hand-written SMILES cannot survive this test.
    """
    from rdkit import Chem

    sdf = DATA / f"{name}.sdf"
    assert sdf.exists(), sdf
    mol = next(iter(Chem.SDMolSupplier(str(sdf), removeHs=True)))
    assert mol is not None, name
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(
        Chem.MolFromSmiles(BUNDLED_LIGANDS[name].smiles)
    ), name


def test_each_bundled_ligand_knows_the_structure_it_came_from():
    """A family can hold two structures; the strict rank needs the right one."""
    ligands = bundled_ligands()
    assert ligands["OHT"].co_crystallised == "3ERT"
    assert ligands["EST"].co_crystallised == "1ERE_A"
    assert ligands["BEN"].co_crystallised == "3PTB"
    assert ligands["RIT"].co_crystallised == "1HXW"
    for name, ligand in ligands.items():
        assert ligand.co_crystallised in ligand.home, name


def test_a_subset_keeps_the_protocol():
    panel = build_panel().subset(["3ptb", "1STP"])
    assert panel.receptors == ("3PTB", "1STP")
    assert panel.protocol == build_panel().protocol


def test_an_unknown_receptor_in_a_subset_is_named():
    with pytest.raises(PanelError) as info:
        build_panel().subset(["3PTB", "9XYZ"])
    assert "9XYZ" in str(info.value)


# ---------------------------------------------------------------------------
# The refusals: a mixed panel fails with the field named
# ---------------------------------------------------------------------------


def test_an_empty_panel_is_refused():
    with pytest.raises(PanelError) as info:
        build_panel(members=[])
    assert "members" in str(info.value)


def test_a_duplicate_receptor_is_refused():
    member = BUNDLED_MEMBERS[0]
    with pytest.raises(PanelError) as info:
        build_panel(members=[member, member])
    assert "pdb_id" in str(info.value)
    assert "twice" in str(info.value)


def test_an_unknown_box_policy_is_refused():
    protocol = PanelProtocol(box_policy="whatever")
    with pytest.raises(PanelError) as info:
        build_panel(protocol=protocol)
    assert "box_policy" in str(info.value)


def test_a_non_positive_window_is_refused():
    with pytest.raises(PanelError) as info:
        build_panel(window=0.0)
    assert "window" in str(info.value)


def test_an_apo_member_is_refused_under_the_reference_policy():
    """The refusal is the point: no reference ligand, no comparable box."""
    with pytest.raises(PanelError) as info:
        build_panel(members=[BUNDLED_MEMBERS[0], APO_MEMBER])
    message = str(info.value)
    assert "2PTN" in message
    assert "ligand" in message
    assert "explicit" in message


def test_a_mixed_box_policy_is_refused_naming_the_field():
    """One member with an explicit box among derived boxes is not a panel."""
    from odock.prepare import BoxSpec

    box = BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0))
    member = PanelMember("1M17", "AQ4", box=box)
    with pytest.raises(PanelError) as info:
        build_panel(members=[BUNDLED_MEMBERS[0], member])
    message = str(info.value)
    assert "box_policy" in message
    assert "1M17" in message


def test_a_mixed_box_buffer_is_refused_naming_the_field():
    odd = PanelMember("1STP", "BTN", box_buffer=4.0)
    with pytest.raises(PanelError) as info:
        build_panel(members=[BUNDLED_MEMBERS[0], odd])
    message = str(info.value)
    assert "box_buffer" in message
    assert "1STP" in message


def test_the_explicit_policy_needs_a_box_for_every_member():
    protocol = PanelProtocol(box_policy="explicit")
    with pytest.raises(PanelError) as info:
        build_panel(members=[BUNDLED_MEMBERS[0]], protocol=protocol)
    assert "box" in str(info.value)


def test_the_explicit_policy_accepts_a_full_set_of_boxes():
    from odock.prepare import BoxSpec

    box = BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0))
    members = [
        PanelMember("A", "", box=box),
        PanelMember("B", "", box=box),
    ]
    panel = build_panel(members=members, protocol=PanelProtocol(box_policy="explicit"))
    assert panel.receptors == ("A", "B")
    boxes = derive_boxes(panel)
    assert boxes["A"].source == "explicit"
    assert boxes["A"].volume == pytest.approx(8000.0)


# ---------------------------------------------------------------------------
# Box derivation
# ---------------------------------------------------------------------------


def test_the_bundled_boxes_are_derived_from_the_reference_ligand():
    panel = build_panel().subset(["3PTB", "1STP"])
    boxes = derive_boxes(panel)
    assert set(boxes) == {"3PTB", "1STP"}
    for name, diagnostics in boxes.items():
        assert diagnostics.source == "reference"
        assert diagnostics.reference == panel.member(name).ligand
        volume = (
            diagnostics.box.size[0] * diagnostics.box.size[1] * diagnostics.box.size[2]
        )
        assert diagnostics.volume == pytest.approx(volume)
        assert diagnostics.volume > 4000  # a real pocket, not a point
        assert diagnostics.grid_points > 0


def test_a_wider_buffer_makes_a_bigger_box():
    protocol = PanelProtocol(box_buffer=5.0)
    narrow = derive_boxes(build_panel(protocol=protocol).subset(["3PTB"]))
    wide = derive_boxes(build_panel().subset(["3PTB"]))
    assert wide["3PTB"].volume > narrow["3PTB"].volume


def test_the_box_contains_the_reference_ligand():
    """The derived box must actually contain the atoms it was derived from."""
    from rdkit import Chem

    panel = build_panel().subset(["3PTB"])
    diagnostics = derive_boxes(panel)["3PTB"]
    _receptor, ligand = split_crystal(DATA / "3PTB.pdb", "BEN")
    conformer = ligand.GetConformer()
    center = diagnostics.box.center
    half = [size / 2.0 for size in diagnostics.box.size]
    for index in range(ligand.GetNumAtoms()):
        point = conformer.GetAtomPosition(index)
        for axis in range(3):
            offset = point[axis] - center[axis]
            assert abs(offset) <= half[axis] + 1e-6, (index, axis, offset)


def test_a_missing_structure_names_the_field():
    member = PanelMember("NOPE", "LIG")
    with pytest.raises(PanelError) as info:
        derive_boxes(build_panel(members=[member]))
    assert "path" in str(info.value)


def test_the_prepared_receptor_does_not_contain_its_own_crystal_ligand(tmp_path):
    """The single most expensive mistake a panel can make.

    A member is a holo structure: its ligand defines the box, and if that ligand
    is left in the receptor it blocks the site every query ligand is docked into.
    The first run of this module did exactly that and produced clash penalties
    (+17.7 kcal/mol for a 50-atom ligand in the estradiol site) that looked like
    selectivity. ``prepare_receptor`` warns about it; this test makes sure the
    panel keeps applying the remedy.
    """
    panel = build_panel()
    texts = panel_module.prepare_members(panel, outdir=tmp_path / "prepared")
    assert set(texts) == set(panel.receptors)
    for name, text in texts.items():
        residue = panel.member(name).ligand
        lines = [
            line
            for line in text.splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
        assert lines, name
        present = [line for line in lines if line[17:20].strip().upper() == residue]
        assert not present, f"{name} still contains its crystal ligand {residue}"
        # and the receptor really is the protein, not a stub
        assert len(lines) > 1000, name


def test_a_structure_without_its_reference_ligand_names_the_field():
    with pytest.raises(PanelError) as info:
        split_crystal(DATA / "3PTB.pdb", "XXX")
    assert "ligand" in str(info.value)


# ---------------------------------------------------------------------------
# The selectivity arithmetic — hand-made cells, no docking
# ---------------------------------------------------------------------------


def cell(receptor, affinity, **kwargs):
    return PanelCell(
        ligand=kwargs.pop("ligand", "LIG"),
        receptor=receptor,
        affinity=affinity,
        **kwargs,
    )


def test_the_gap_is_the_difference_to_the_second_best():
    profile = selectivity_profile(
        "LIG", [cell("A", -9.0), cell("B", -7.5), cell("C", -6.0)]
    )
    assert profile.best.receptor == "A"
    assert profile.second.receptor == "B"
    assert profile.gap == pytest.approx(1.5)


def test_the_window_counts_receptors_within_it():
    cells = [cell("A", -9.0), cell("B", -8.4), cell("C", -7.0)]
    tight = selectivity_profile("LIG", cells, window=0.5)
    loose = selectivity_profile("LIG", cells, window=1.0)
    assert [c.receptor for c in tight.within_window] == ["A"]
    assert [c.receptor for c in loose.within_window] == ["A", "B"]
    assert len(loose.within_window) == 2


def test_a_single_receptor_has_no_gap_and_reads_as_such():
    profile = selectivity_profile("LIG", [cell("A", -9.0)])
    assert profile.gap is None
    assert profile.outcome() == "only one receptor scored (A)"


def test_a_clear_winner_reads_as_selective():
    profile = selectivity_profile("LIG", [cell("A", -9.0), cell("B", -6.5)])
    assert profile.gap == pytest.approx(2.5)
    assert len(profile.within_window) == 1
    assert "selective for A" in profile.outcome()


def test_a_near_tie_reads_as_a_preference_not_selectivity():
    profile = selectivity_profile("LIG", [cell("A", -9.0), cell("B", -8.6)])
    assert len(profile.within_window) == 2
    assert "a preference, not selectivity" in profile.outcome()


def test_failed_cells_are_excluded_from_the_ranking_but_not_hidden():
    profile = selectivity_profile(
        "LIG",
        [
            cell("A", -9.0),
            PanelCell(ligand="LIG", receptor="B", status="failed", error="no pose"),
        ],
    )
    assert [c.receptor for c in profile.ranked] == ["A"]
    assert [c.receptor for c in profile.failures] == ["B"]
    assert profile.best.receptor == "A"


def test_ties_are_broken_by_receptor_id_so_the_table_is_stable():
    profile = selectivity_profile("LIG", [cell("B", -9.0), cell("A", -9.0)])
    assert [c.receptor for c in profile.ranked] == ["A", "B"]


def test_the_home_check_and_rank():
    cells = [cell("A", -9.0), cell("B", -8.0), cell("C", -7.0)]
    at_home = selectivity_profile("LIG", cells, home={"A"})
    away = selectivity_profile("LIG", cells, home={"B"})
    assert at_home.best_is_home is True
    assert at_home.home_rank == 1
    assert away.best_is_home is False
    assert away.home_rank == 2
    unknown = selectivity_profile("LIG", cells)
    assert unknown.best_is_home is None
    assert unknown.home_rank is None


def test_a_profile_without_any_usable_cell_says_so():
    profile = selectivity_profile(
        "LIG", [PanelCell(ligand="LIG", receptor="A", status="failed")]
    )
    assert profile.best is None
    assert profile.outcome() == "no receptor produced a pose"


# ---------------------------------------------------------------------------
# The hit rate and how a miss is explained
# ---------------------------------------------------------------------------


def _profile(name, home, affinities, boxes=None, window=1.0):
    boxes = boxes or {}
    cells = [
        cell(receptor, affinity, ligand=name, box_volume=boxes.get(receptor, 8000.0))
        for receptor, affinity in affinities.items()
    ]
    return selectivity_profile(name, cells, window=window, home=home, protein=name)


def test_the_hit_rate_counts_matches_and_reports_every_miss():
    profiles = [
        _profile("HIT", {"A"}, {"A": -9.0, "B": -6.0}),
        _profile("PROMISCUOUS", {"A"}, {"A": -8.9, "B": -9.0}),
        _profile("BOXED", {"A"}, {"B": -9.0, "A": -7.0}, boxes={"B": 20000.0, "A": 8000.0}),
        _profile("BEHIND", {"A"}, {"B": -9.0, "A": -6.0}, boxes={"B": 8000.0, "A": 8000.0}),
    ]
    report = hit_rate(profiles)
    assert report["n"] == 4
    assert report["hits"] == 1
    assert report["rate"] == pytest.approx(0.25)
    reasons = {miss["ligand"]: miss["reason"] for miss in report["misses"]}
    assert reasons == {
        "PROMISCUOUS": "promiscuous",
        "BOXED": "box",
        "BEHIND": "score",
    }


def test_a_ligand_without_a_home_is_not_counted():
    profiles = [
        _profile("KNOWN", {"A"}, {"A": -9.0}),
        selectivity_profile("UNKNOWN", [cell("A", -9.0)]),
    ]
    report = hit_rate(profiles)
    assert report["n"] == 1
    assert report["hits"] == 1


def test_a_failed_home_cell_is_reported_as_a_failure_not_a_miss_choice():
    profile = selectivity_profile(
        "LIG",
        [
            cell("B", -9.0),
            PanelCell(ligand="LIG", receptor="A", status="failed", error="no pose"),
        ],
        home={"A"},
    )
    report = hit_rate([profile])
    assert report["misses"][0]["reason"] == "failed"


# ---------------------------------------------------------------------------
# Rendering and writers
# ---------------------------------------------------------------------------


def _result(profiles, window=1.0, single_seed=True):
    boxes = {
        "A": panel_module.BoxDiagnostics(
            pdb_id="A",
            reference="L1",
            box=__import__("odock").BoxSpec(center=(0, 0, 0), size=(20, 20, 20)),
            volume=8000.0,
            receptor_atoms=350,
            grid_points=150000,
            source="reference",
        ),
        "B": panel_module.BoxDiagnostics(
            pdb_id="B",
            reference="L2",
            box=__import__("odock").BoxSpec(center=(0, 0, 0), size=(24, 24, 24)),
            volume=13824.0,
            receptor_atoms=500,
            grid_points=250000,
            source="reference",
        ),
    }
    panel = build_panel(
        members=[
            PanelMember("A", "L1", "protein A"),
            PanelMember("B", "L2", "protein B"),
        ],
        window=window,
    )
    cells = tuple(cell for profile in profiles for cell in profile.cells)
    return panel_module.PanelResult(
        panel=panel, profiles=tuple(profiles), boxes=boxes, cells=cells
    )


def test_the_matrix_contains_every_cell_and_marks_failures():
    profiles = [
        _profile("LIG", {"A"}, {"A": -9.0, "B": -7.0}),
    ]
    profiles[0].cells = profiles[0].cells + (
        PanelCell(ligand="LIG", receptor="B", status="failed"),
    )
    result = _result(profiles)
    text = matrix(result)
    assert "LIG" in text
    assert "-9.00" in text
    assert "--" in text
    assert "comparable only as far as the box table" in text


def test_the_table_reports_the_window_and_the_home_rank():
    result = _result([_profile("LIG", {"A"}, {"A": -9.0, "B": -7.0})])
    text = table(result)
    assert "LIG" in text
    assert "yes" in text  # best_is_home
    assert "ONE SEED" in text


def test_the_table_warns_when_there_is_more_than_one_seed():
    profiles = [_profile("LIG", {"A"}, {"A": -9.0, "B": -7.0})]
    for item in profiles[0].cells:
        item.n_seeds = 3
        item.seeds = (42, 7, 2024)
        item.spread = 0.12
    result = _result(profiles)
    text = table(result)
    assert "ONE SEED" not in text
    assert "±" in matrix(result)


def test_the_box_table_reports_volume_and_pocket_size():
    result = _result([_profile("LIG", {"A"}, {"A": -9.0, "B": -7.0})])
    text = box_table(result)
    assert "8000" in text and "13824" in text
    assert "350" in text and "500" in text
    assert "smaller box elsewhere" in text


def test_json_and_csv_round_trip(tmp_path):
    result = _result([_profile("LIG", {"A"}, {"A": -9.0, "B": -7.0})])
    json_path = result.write_json(tmp_path / "panel.json")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["panel"]["name"]
    assert document["panel"]["is_selectivity_assay"] is False
    assert document["profiles"][0]["ligand"] == "LIG"
    assert document["boxes"]["A"]["volume"] == pytest.approx(8000.0)
    assert document["hit_rate"]["n"] == 1

    csv_path = result.write_csv(tmp_path / "panel.csv")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["receptor"] for row in rows} == {"A", "B"}
    assert rows[0]["best"] == "A"
    assert rows[0]["n_within_window"] == "1"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_parser_registers_and_parses():
    parser = argparse.ArgumentParser(prog="odock")
    sub = parser.add_subparsers(dest="command")
    panel_module.add_panel_parser(sub)
    args = parser.parse_args(
        ["panel", "--out", "runs/panel", "--receptors", "3PTB,1STP", "--window", "0.5"]
    )
    assert args.command == "panel"
    assert args.out == "runs/panel"
    assert args.receptors == "3PTB,1STP"
    assert args.window == 0.5
    assert args.handler is panel_module._handle_panel


def test_the_cli_help_says_what_a_panel_is_not():
    from odock.panel import _build_parser

    text = _build_parser().format_help()
    assert "not a selectivity assay" in text
    assert "7" not in text.split("--receptors")[0]  # no accidental cliffhanger


def test_the_cli_seeds_are_parsed_into_a_tuple():
    args = panel_module._build_parser().parse_args(
        ["--out", "x", "--seeds", "42,7,2024"]
    )
    assert tuple(int(part) for part in args.seeds.split(",")) == (42, 7, 2024)


def test_a_missing_ligand_file_names_the_field(tmp_path):
    with pytest.raises(PanelError) as info:
        panel_module.read_ligands(tmp_path / "nope.smi")
    assert "ligands" in str(info.value)


def test_read_ligands_accepts_smiles_name_lines(tmp_path):
    path = tmp_path / "lib.smi"
    path.write_text("CCO ethanol\nc1ccccc1 benzene\n", encoding="utf-8")
    ligands = panel_module.read_ligands(path)
    assert ligands == {"ethanol": "CCO", "benzene": "c1ccccc1"}


def test_read_ligands_reads_an_sdf_without_explicit_hydrogens():
    """The library is SMILES; `screen` adds the hydrogens when it prepares."""
    from rdkit import Chem

    ligands = panel_module.read_ligands(DATA / "BTN.sdf")
    assert len(ligands) == 1
    name, smiles = next(iter(ligands.items()))
    assert name == "BTN", name
    assert "[H]" not in smiles
    assert Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() == 16
    assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(BUNDLED_LIGANDS["BTN"].smiles)
    )


# ---------------------------------------------------------------------------
# One real run: the orchestration, not the numbers
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_two_by_two_panel_is_produced_end_to_end(tmp_path):
    """The smallest real panel: does the orchestration return a usable result?

    The affinities are deliberately *not* asserted: at exhaustiveness 1 this is a
    screening estimate, and pinning a specific number here would turn a change of
    search effort into a test failure.  What must hold is the shape — every cell
    has an affinity, every ligand has a profile, the home rank is computed, and
    the tables and writers accept the result.
    """
    protocol = PanelProtocol(exhaustiveness=1, num_poses=5, agreement=False)
    panel = bundled_panel(protocol=protocol, receptors=["3PTB", "1STP"])
    ligands = bundled_ligands(["BEN", "BTN"])
    result = panel_module.profile(
        {name: spec.smiles for name, spec in ligands.items()},
        panel,
        tmp_path / "panel",
        progress=False,
        homes={name: spec.home for name, spec in ligands.items()},
        proteins={name: spec.protein for name, spec in ligands.items()},
    )

    assert result.receptors == ("3PTB", "1STP")
    assert result.ligands == ("BEN", "BTN")
    assert result.single_seed is True
    for ligand in result.ligands:
        profile = result.profile(ligand)
        assert len(profile.cells) == 2
        assert profile.best is not None
        assert profile.gap is not None
        assert profile.home_rank is not None
    for cell in result.cells:
        assert cell.affinity is not None
        assert cell.affinity < 0
        assert cell.spread is None  # one seed, so no spread to report
        assert cell.box_volume > 4000
        assert cell.pocket_atoms > 0
        assert cell.efficiency is not None
    assert "3PTB" in result.matrix()
    _assert_box_rows_match_the_members(result)
    document = result.as_dict()
    assert document["single_seed"] is True
    report = result.hit_rate()
    assert report["n"] == 2
    assert 0 <= report["hits"] <= 2


def _assert_box_rows_match_the_members(result):
    """The box table must carry, for every member, its reference, size and source.

    Written against the *structure* of the table rather than its wording: a word
    count of "reference" also counts the prose underneath, so it breaks on a
    copyedit and still misses a row that lost its source.
    """
    lines = result.box_table().splitlines()
    header = lines[0].split()
    assert "reference" in header, header
    assert "source" in header, header
    rows = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == 7 and parts[0] in result.receptors:
            rows[parts[0]] = parts
    assert set(rows) == set(result.receptors)
    for name, parts in rows.items():
        diagnostics = result.boxes[name]
        assert parts[1] == diagnostics.reference, name
        assert float(parts[3]) == pytest.approx(diagnostics.volume, abs=0.5), name
        assert int(parts[4]) == diagnostics.receptor_atoms, name
        assert parts[6], f"{name} lost its box source"
        assert parts[6] == diagnostics.source, name
        assert diagnostics.receptor_atoms > 0, name
        assert diagnostics.volume > 4000, name

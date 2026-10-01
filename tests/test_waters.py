# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.waters`: networks, conserved sites, displaced waters.

Three kinds of assertion, in order of how much they are worth:

* **constructed cases** for the conservation criterion — a water that is identical
  in two conformations must be *conserved*, one that moves 2 Å must be *moved*,
  and one that is present in the ligand-free structure and occupied by the ligand
  in the other must be *displaced*, with the ligand atom named.  Those are the
  three verdicts, and a classifier is only worth having if each can be produced on
  demand;
* **the network** — components and edges of a hand-built water cluster, where the
  arithmetic is exact;
* **the real pair** — 3PTB (benzamidine) against 2PTN (ligand-free), where the
  measured answer is chemically checkable: the displaced waters are the ones whose
  place benzamidine's amidine nitrogen takes, and they were hydrogen bonded to
  ASP189, the S1 specificity residue.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from odock import ensemble as ens
from odock import waters


def atom(name: str, element: str, res_name: str, res_id: int, x: float, y: float, z: float,
         chain: str = "A") -> ens.Atom:
    return ens.Atom(
        name=name, element=element, res_name=res_name, res_id=res_id, chain=chain,
        x=float(x), y=float(y), z=float(z),
    )


def conformation(label: str, atoms) -> ens.Conformation:
    return ens.Conformation(label=label, atoms=list(atoms), records=[])


def protein_at(x: float, y: float, z: float, res_id: int = 1, name: str = "OD1",
               res_name: str = "ASP") -> ens.Atom:
    return atom(name, "O", res_name, res_id, x, y, z)


def water(res_id: int, x: float, y: float, z: float, name: str = "O") -> ens.Atom:
    return atom(name, "O", "HOH", res_id, x, y, z)


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


def test_a_chain_of_waters_is_one_component():
    """Three waters 3 Å apart: two H-bonds, one component of three."""
    atoms = [water(1, 0.0, 0.0, 0.0), water(2, 3.0, 0.0, 0.0), water(3, 6.0, 0.0, 0.0)]
    network = waters.water_network(conformation("A", atoms))
    assert network.n_waters == 3
    assert network.n_edges == 2
    assert network.n_components == 1
    assert network.largest_component == 3
    assert network.bridges() == 1  # the middle water
    assert all(edge[2] == pytest.approx(3.0) for edge in network.edges)


def test_waters_further_apart_than_the_cutoff_are_separate_components():
    atoms = [water(1, 0.0, 0.0, 0.0), water(2, 5.0, 0.0, 0.0), water(3, 10.0, 0.0, 0.0)]
    network = waters.water_network(conformation("A", atoms), hbond_cutoff=3.5)
    assert network.n_edges == 0
    assert network.n_components == 3
    assert network.largest_component == 1


def test_the_network_records_protein_contacts():
    atoms = [water(1, 0.0, 0.0, 0.0), protein_at(3.0, 0.0, 0.0, res_id=189)]
    network = waters.water_network(conformation("A", atoms))
    assert network.n_waters == 1
    assert network.waters[0].contacts == ["ASP189 A"]
    assert network.protein_contacts == 1
    # A contact beyond the cutoff is not recorded.
    far = waters.water_network(conformation("A", [atoms[0], protein_at(9.0, 0.0, 0.0)]))
    assert far.waters[0].contacts == []


def test_a_water_without_an_oxygen_is_skipped():
    atoms = [water(1, 0.0, 0.0, 0.0), atom("H1", "H", "HOH", 2, 5.0, 0.0, 0.0)]
    network = waters.water_network(conformation("A", atoms))
    assert network.n_waters == 1


# ---------------------------------------------------------------------------
# The conservation criterion, on constructed cases
# ---------------------------------------------------------------------------


def site_from(points, *, label_a="A", label_b="B", ligand=None):
    """Two conformations, each with the water positions given for it."""
    first = conformation(label_a, [water(index + 1, *point) for index, point in enumerate(points["a"])])
    second = conformation(label_b, [water(index + 1, *point) for index, point in enumerate(points["b"])])
    return waters.compare_water_sites(
        [first, second], match_radius=1.5, occlusion_radius=2.5,
    ) if ligand is None else _with_ligand(points, ligand)


def _with_ligand(points, ligand):
    first = conformation("A", [water(index + 1, *point) for index, point in enumerate(points["a"])])
    second_atoms = [water(index + 1, *point) for index, point in enumerate(points["b"])]
    # The displacing ligand lives in conformation A (the holo one).
    second_atoms = [atom(f"C{k}", "C", "LIG", 99, *position) for k, position in enumerate([])]
    holo = conformation("A", second_atoms + [water(1, *points["a"][0])])
    apo_atoms = [water(1, *points["b"][0])]
    apo = conformation("B", apo_atoms)
    return waters.compare_water_sites([holo, apo], match_radius=1.5, occlusion_radius=2.5)


def test_an_identical_water_is_conserved():
    analysis = waters.compare_water_sites(
        [
            conformation("A", [water(1, 0.0, 0.0, 0.0)]),
            conformation("B", [water(1, 0.0, 0.0, 0.0)]),
        ]
    )
    assert len(analysis.sites) == 1
    site = analysis.sites[0]
    assert site.occupancy == pytest.approx(1.0)
    assert site.displacement == pytest.approx(0.0)
    assert site.classification == "conserved"


def test_a_water_that_moves_two_angstrom_is_moved():
    analysis = waters.compare_water_sites(
        [
            conformation("A", [water(1, 0.0, 0.0, 0.0)]),
            conformation("B", [water(1, 2.0, 0.0, 0.0)]),
        ],
        match_radius=2.5,
    )
    assert len(analysis.sites) == 1
    site = analysis.sites[0]
    assert site.occupancy == pytest.approx(1.0)
    assert site.displacement == pytest.approx(2.0)
    # Present everywhere but travelling: that is "moved", not "conserved".
    assert site.classification == "moved"
    assert analysis.by_class("conserved") == []


def test_a_water_displaced_by_a_ligand_is_displaced():
    """The ligand sits where the water was: the strongest evidence there is."""
    holo = conformation(
        "holo",
        [atom("C1", "C", "LIG", 99, 0.1, 0.0, 0.0), atom("N1", "N", "LIG", 99, 0.3, 0.0, 0.0)],
    )
    apo = conformation("apo", [water(1, 0.0, 0.0, 0.0)])
    analysis = waters.compare_water_sites([holo, apo], match_radius=1.5, occlusion_radius=2.5)
    assert len(analysis.sites) == 1
    site = analysis.sites[0]
    assert site.occupancy == pytest.approx(0.5)
    assert site.displaced_by
    assert "LIG99 A:C1" in sum(site.displaced_by.values(), [])
    assert site.classification == "displaced"


def test_a_water_lost_without_a_ligand_is_transient():
    """Present in one structure, gone in the other, nothing to blame: transient.

    With only two conformations an occupancy of 0.5 says "half of them", and
    calling that *moved* would claim the water travelled when in fact it is not
    there at all in one structure.
    """
    holo = conformation("A", [protein_at(0.0, 0.0, 0.0), protein_at(1.0, 0.0, 0.0)])
    apo = conformation("B", [water(1, 0.0, 0.0, 0.0)])
    analysis = waters.compare_water_sites([holo, apo], match_radius=1.5, occlusion_radius=2.5)
    assert analysis.sites
    assert analysis.sites[0].classification == "transient"
    assert analysis.sites[0].displaced_by == {}
    assert analysis.by_class("transient") == analysis.sites


def test_a_conserved_water_keeps_its_ligand_evidence():
    """Conserved wins the label, but the ligand overlap is still reported."""
    holo = conformation(
        "holo", [atom("C1", "C", "LIG", 99, 2.0, 0.0, 0.0), water(1, 0.0, 0.0, 0.0)]
    )
    apo = conformation("apo", [water(1, 0.0, 0.0, 0.0)])
    analysis = waters.compare_water_sites([holo, apo], match_radius=1.5, occlusion_radius=3.0)
    site = next(site for site in analysis.sites if site.occupancy == 1.0)
    assert site.classification == "conserved"
    assert site.displaced_by == {}  # the water is present everywhere


# ---------------------------------------------------------------------------
# Poses and the power discipline
# ---------------------------------------------------------------------------


def test_pose_displacement_counts_only_the_selected_classes():
    sites = [
        waters.WaterSite(index=0, center=(0.0, 0.0, 0.0)),
        waters.WaterSite(index=1, center=(10.0, 0.0, 0.0)),
    ]
    for site in sites:
        site.observations = {"A": waters.WaterObservation("A", "HOH1 A", "A", 1, site.center)}
    pose = type("Pose", (), {"coords": np.zeros((1, 3)), "affinity": -5.0, "conformation": "A"})()
    counted = waters.pose_displacement([pose], sites, radius=3.5)
    assert counted[0]["displaced"] == 1
    assert counted[0]["sites"] == [0]
    assert counted[0]["affinity"] == pytest.approx(-5.0)
    # Nothing is counted when the class filter excludes everything.
    assert waters.pose_displacement([pose], sites, classes=("displaced",))[0]["displaced"] == 0


def test_correlate_reports_n_and_says_when_n_is_too_small():
    result = waters.correlate([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0])
    assert result["n"] == 3
    assert result["samples"] == 0
    assert "not resolvable" in result["note"]
    assert result["rho"] == pytest.approx(-1.0)


def test_correlate_gives_a_bootstrap_interval_when_it_can():
    scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    affinities = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
    result = waters.correlate(scores, affinities, samples=200, seed=7)
    assert result["n"] == 10
    assert result["rho"] == pytest.approx(-1.0)
    assert result["samples"] == 200
    assert result["low"] <= result["rho"] <= result["high"]
    assert result["note"] == ""
    # A perfect monotone relation is perfect in every resample.
    assert result["low"] == pytest.approx(-1.0)


def test_correlate_is_deterministic(data_dir=None):
    scores = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    affinities = [-3.0, -1.0, -4.0, -2.0, -6.0, -5.0]
    first = waters.correlate(scores, affinities, samples=100, seed=3)
    second = waters.correlate(scores, affinities, samples=100, seed=3)
    assert first == second


# ---------------------------------------------------------------------------
# The real pair
# ---------------------------------------------------------------------------


def trypsin_pair(data_dir):
    first, second = data_dir / "3PTB.pdb", data_dir / "2PTN.pdb"
    if not first.exists() or not second.exists():
        pytest.skip("missing the bundled trypsin structures")
    return first, second


def test_the_trypsin_pair_networks_are_measured(data_dir):
    first, second = trypsin_pair(data_dir)
    conformations = ens.read_conformations([first, second], keep_water=True)
    assert [len(waters._water_atoms(c)) for c in conformations] == [62, 82]
    networks = [waters.water_network(c) for c in conformations]
    assert networks[0].n_waters == 62 and networks[1].n_waters == 82
    assert networks[0].n_edges == 20 and networks[1].n_edges == 35
    assert networks[0].largest_component == 6 and networks[1].largest_component == 7
    assert networks[0].n_components == 42 and networks[1].n_components == 48


@pytest.mark.slow
def test_benzamidine_displaces_two_waters_that_bonded_to_asp189(data_dir):
    """The honest-core check, and it is testable offline on this pair.

    3PTB carries benzamidine, 2PTN does not, so a water site present in 2PTN and
    occupied by the ligand in 3PTB is an experimentally observed ligand-displaced
    water.  Measured: two of them, both hydrogen bonded to ASP189 -- the S1
    specificity residue -- and one of them occupied by the ligand's amidine
    nitrogen.
    """
    from odock.prepare import box_from_points

    first, second = trypsin_pair(data_dir)
    conformations = ens.read_conformations([first, second], keep_water=True)
    box = box_from_points(ens.ligand_coords(conformations[0], "BEN"), buffer=6.0)
    ens.align_conformations(conformations, box=box, superpose=True)
    analysis = waters.compare_water_sites(conformations)
    summary = analysis.summary()
    assert summary["counts"]["conserved"] == 55
    assert summary["counts"]["displaced"] == 2
    displaced = analysis.displaced
    assert len(displaced) == 2
    for site in displaced:
        assert site.occupancy == pytest.approx(0.5)
        assert site.displaced_by
        labels = [label for label in site.displaced_by]
        assert labels == ["3PTB"]  # the ligand sits in 3PTB, the water in 2PTN
    bonded = [set(site.contacts) for site in displaced]
    assert all(any(contact.startswith("ASP189") for contact in contacts) for contacts in bonded)
    amidine = [
        atom_label
        for site in displaced
        for labels in site.displaced_by.values()
        for atom_label in labels
    ]
    assert any(label.endswith(":N1") for label in amidine), amidine
    # And the report names them.
    text = analysis.text()
    assert "waters displaced by the ligand" in text
    assert "ASP189" in text
    assert "the water bonds to" in text


def test_the_cli_reports_the_water_analysis(data_dir, tmp_path, capsys):
    from odock.cli import main

    first, second = trypsin_pair(data_dir)
    target = tmp_path / "waters.json"
    code = main(
        [
            "ensemble", "waters",
            "-r", str(first), str(second),
            "--box-ligand", "BEN",
            "--json-out", str(target),
            "-q",
        ]
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["labels"] == ["3PTB", "2PTN"]
    assert payload["summary"]["waters_per_conformation"] == {"3PTB": 62, "2PTN": 82}
    assert payload["summary"]["counts"]["conserved"] == 55
    assert payload["summary"]["counts"]["displaced"] == 2
    assert len(payload["sites"]) == 89
    assert payload["criteria"]["match_radius"] == 1.5

# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.coupling`: coupling, its truncation, and pathways.

The constructed cases are the acceptance criterion the mandate names, and they are
closed-form because the network is built by hand — eigenvectors and eigenvalues
set directly, so the expected correlation is arithmetic rather than a property of
some structure:

* one mode moving nodes 0 and 1 in phase and another moving nodes 2 and 3 in phase
  gives ``C[0,1] = C[2,3] = 1`` and ``C[0,2] = 0``: **two independently coupled
  blocks do not couple to each other**;
* a single mode moving every node in phase gives ``C = 1`` everywhere: **two
  rigidly linked blocks do couple strongly**;
* the path finder on a hand-built graph with a known shortest path returns that
  path, and a weak direct edge does not beat a two-step strong route.

The real-structure tests pin the two measured findings: the HIV protease route
between the catalytic aspartate and the flap tip passes through the flap **hinge**
(the known coupling, and a sanity signal), and the estrogen receptor's pocket-to-
helix-12 coupling is **at or below what any pair at that separation scores**, so it
is contact geometry rather than a functional coupling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from odock import coupling as cp
from odock import ensemble as ens
from odock.modes import ElasticNetwork


def hand_network(eigenvectors, eigenvalues, labels, coords=None, cutoff=13.0):
    """An ElasticNetwork with hand-set modes, so the correlations are closed form.

    Each "mode" is given as one number per node, and the displacements are placed
    along x: a mode is a ``3N`` vector, so the scalar pattern becomes
    ``(v[0], 0, 0, v[1], 0, 0, ...)``.  The correlation between two nodes then
    equals the product of their scalars, which is what makes these tests
    arithmetic instead of structural.
    """
    n = len(labels)
    columns = []
    for mode in eigenvectors:
        column = np.zeros(3 * n, dtype=float)
        for node, value in enumerate(mode):
            column[3 * node] = float(value)
        columns.append(column)
    vectors = np.column_stack(columns) if columns else np.zeros((3 * n, 0))
    return ElasticNetwork(
        labels=list(labels),
        coords=np.asarray(coords if coords is not None else np.zeros((n, 3)), dtype=float),
        cutoff=cutoff,
        eigenvalues=np.asarray(eigenvalues, dtype=float),
        eigenvectors=vectors,
        selected=len(eigenvalues),
    )


# ---------------------------------------------------------------------------
# Coupling, closed form
# ---------------------------------------------------------------------------


def test_two_independently_coupled_blocks_do_not_couple():
    """Mode 1 moves nodes 0,1; mode 2 moves nodes 2,3: no cross-coupling."""
    labels = ["A1", "A2", "B1", "B2"]
    vectors = [
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
    network = hand_network(vectors, [1.0, 1.0], labels)
    correlation = cp.cross_correlation(network)
    assert correlation[0, 1] == pytest.approx(1.0)
    assert correlation[2, 3] == pytest.approx(1.0)
    assert correlation[0, 2] == pytest.approx(0.0)
    assert correlation[0, 3] == pytest.approx(0.0)
    assert np.allclose(np.diag(correlation), 1.0)


def test_two_rigidly_linked_blocks_do_couple():
    """One mode moving every node in phase gives perfect correlation everywhere."""
    labels = ["A1", "A2", "B1", "B2"]
    network = hand_network([[1.0, 1.0, 1.0, 1.0]], [1.0], labels)
    correlation = cp.cross_correlation(network)
    assert np.allclose(correlation, 1.0)
    pairs = cp.strongest_couplings(correlation, labels, top=10, min_separation=1)
    assert len(pairs) == 6
    assert all(pair.correlation == pytest.approx(1.0) for pair in pairs)


def test_anti_correlated_blocks_couple_negatively_and_rank_by_magnitude():
    """Two halves swinging apart: |C| = 1, signed C = -1."""
    labels = ["A1", "A2", "B1", "B2"]
    network = hand_network([[1.0, 1.0, -1.0, -1.0]], [1.0], labels)
    correlation = cp.cross_correlation(network)
    assert correlation[0, 2] == pytest.approx(-1.0)
    # Ranking by |C| puts both the in-phase pair (0,1) and the anti-phase pair
    # (0,2) in the top two: a hinge is a coupling too.
    absolute = cp.strongest_couplings(correlation, labels, top=2, min_separation=1)
    assert {(pair.first, pair.second) for pair in absolute} == {(0, 1), (0, 2)}
    assert any(pair.correlation == pytest.approx(-1.0) for pair in absolute)
    # Ranking by the signed value sees only the in-phase pairs.
    signed = cp.strongest_couplings(
        correlation, labels, top=2, min_separation=1, absolute=False
    )
    assert all(pair.correlation == pytest.approx(1.0) for pair in signed)


def test_softer_modes_dominate_the_covariance():
    """Two identical modes, one softer: the soft one sets the correlation."""
    labels = ["A1", "A2", "B1", "B2"]
    # Mode 1 (soft, lambda = 0.1) moves A together; mode 2 (stiff, lambda = 10)
    # moves B against A. The soft mode must dominate.
    vectors = [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, -1.0],
    ]
    network = hand_network(vectors, [0.1, 10.0], labels)
    correlation = cp.cross_correlation(network)
    assert correlation[0, 1] > 0.9
    assert correlation[0, 2] < 0.0


def test_the_mode_count_changes_the_answer_and_the_report_says_so():
    """Each mode adds its own coupling pattern; the top list is truncation-dependent."""
    labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    vectors = [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # A only
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],   # B only
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],   # C only
    ]
    network = hand_network(vectors, [1.0, 1.0, 1.0], labels)
    report = cp.mode_count_sensitivity(
        network, labels, counts=(1, 2, 3), top=2, min_separation=1, reference=1
    )
    assert report["reference_modes"] == 1
    rows = {row["modes"]: row for row in report["rows"]}
    # With one mode only the A pair is coupled; with two, the B pair appears too,
    # so the top-2 list cannot be identical.
    assert rows[1]["jaccard_with_reference"] == pytest.approx(1.0)
    assert rows[2]["jaccard_with_reference"] < 0.6
    assert rows[3]["jaccard_with_reference"] < 0.6


def test_cross_correlation_normalisation_stays_in_range():
    """A random 3-D mode: every |C| <= 1 and the diagonal is exactly 1."""
    labels = [f"R{index}" for index in range(5)]
    rng = np.random.default_rng(11)
    mode = rng.normal(size=15)  # 5 nodes x 3 components
    network = ElasticNetwork(
        labels=labels, coords=np.zeros((5, 3)), eigenvalues=np.array([1.0]),
        eigenvectors=mode.reshape(15, 1), selected=1,
    )
    correlation = cp.cross_correlation(network)
    assert correlation.min() >= -1.0 - 1e-12
    assert correlation.max() <= 1.0 + 1e-12
    assert np.allclose(np.diag(correlation), 1.0)


# ---------------------------------------------------------------------------
# Pathways, on a hand-built graph
# ---------------------------------------------------------------------------


def graph_correlation(n: int, edges, labels=None):
    matrix = np.eye(n)
    for i, j, value in edges:
        matrix[i, j] = matrix[j, i] = value
    return matrix


def test_the_path_finder_returns_the_known_shortest_path():
    """A chain 0-1-2-3-4 with strong links: the route must follow the chain."""
    labels = [f"R{index}" for index in range(5)]
    chain = [(0, 1, 0.95), (1, 2, 0.95), (2, 3, 0.95), (3, 4, 0.95)]
    matrix = graph_correlation(5, chain)
    cheap, wide = cp.pathway(matrix, labels, 0, 4, edge_threshold=0.9)
    assert cheap.nodes == [0, 1, 2, 3, 4]
    assert cheap.length == 4
    assert cheap.bottleneck == pytest.approx(0.95)
    assert wide.nodes == [0, 1, 2, 3, 4]
    assert wide.bottleneck == pytest.approx(0.95)


def test_a_weak_direct_edge_does_not_beat_a_strong_multi_step_route():
    """The reason the threshold exists, with the arithmetic shown.

    A one-step edge of |C| 0.3 costs ``1/0.3 = 3.33``; a five-step route of |C|
    0.9 costs ``5/0.9 = 5.56``.  Without a threshold the **weak shortcut wins**,
    because the cost is a sum of steps and one weak step is still one step.  That
    is exactly what happened on 1HVR, where a one-step route with |C| 0.336 beat a
    five-step route of |C| 0.83.  Cutting edges below 0.5 leaves the strong route,
    which is the meaningful one.
    """
    labels = [f"R{index}" for index in range(6)]
    chain = [(index, index + 1, 0.9) for index in range(5)]
    matrix = graph_correlation(6, chain + [(0, 5, 0.3)])
    complete, _ = cp.pathway(matrix, labels, 0, 5, edge_threshold=0.0)
    assert complete.nodes == [0, 5], "the weak one-step shortcut is cheapest"
    assert complete.bottleneck == pytest.approx(0.3)
    cheap, wide = cp.pathway(matrix, labels, 0, 5, edge_threshold=0.5)
    assert cheap.nodes == [0, 1, 2, 3, 4, 5]
    assert cheap.bottleneck == pytest.approx(0.9)
    assert wide.bottleneck == pytest.approx(0.9)
    assert "edges >=" in cheap.kind and "0.500" in cheap.kind


def test_the_bottleneck_path_maximises_the_weakest_link():
    """A detour of strong steps beats a shortcut containing a weak one."""
    labels = ["R0", "R1", "R2", "R3"]
    edges = [(0, 3, 0.4), (0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9)]
    matrix = graph_correlation(4, edges)
    cheap, wide = cp.pathway(matrix, labels, 0, 3, edge_threshold=0.5)
    assert cheap.nodes == [0, 1, 2, 3]          # the weak shortcut is cut
    assert wide.bottleneck == pytest.approx(0.9)
    assert wide.nodes == [0, 1, 2, 3]


def test_unreachable_endpoints_are_reported_not_invented():
    labels = ["R0", "R1", "R2"]
    matrix = graph_correlation(3, [(0, 1, 0.99)])  # node 2 is isolated
    cheap, _wide = cp.pathway(matrix, labels, 0, 2, edge_threshold=0.5)
    assert cheap.nodes == []
    with pytest.raises(ens.EnsembleError):
        cp.pathway(matrix, labels, 0, 9)


def test_the_distance_profile_falls_off_with_separation():
    """Two coupled blocks 3 Å apart, 27 Å from each other: near pairs couple, far do not."""
    labels = [f"R{index}" for index in range(4)]
    coords = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (30.0, 0.0, 0.0), (33.0, 0.0, 0.0)]
    network = hand_network([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]], [1.0, 1.0],
                           labels, coords=coords)
    correlation = cp.cross_correlation(network)
    profile = cp.distance_profile(network, correlation, bins=(0.0, 6.0, 20.0, 1e9))
    near, middle, far = profile
    assert near["n_pairs"] == 2 and near["mean_correlation"] == pytest.approx(1.0)
    assert middle["n_pairs"] == 0
    assert far["n_pairs"] == 4 and far["mean_correlation"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The real structures
# ---------------------------------------------------------------------------


def bundled(data_dir, name: str):
    path = data_dir / name
    if not path.exists():
        pytest.skip(f"missing the bundled {name}")
    return ens.read_conformations([path])[0]


@pytest.mark.slow
def test_the_hiv_route_runs_through_the_flap_hinge(data_dir):
    """The sanity signal: Asp25 to the flap tip passes through the flap hinge.

    HIV-1 protease's flap tips and catalytic aspartates are a known coupled pair.
    Measured here: the lowest-cost route from ASP25 A to ILE50 A is
    ASP25 A -> ASP25 B -> ILE84 B -> VAL82 B -> THR80 B -> ILE50 A -- five steps,
    through the second monomer's catalytic aspartate and then the flap hinge
    (THR80/VAL82/ILE84).  The route does not go through space; it goes through the
    residues whose motion is known to be tied to the flaps.
    """
    conformation = bundled(data_dir, "1HVR.pdb")
    network_nodes = {"active": [("A", 25, "ASP")], "flaps": [("A", 50, "ILE"), ("A", 51, "GLY")]}
    analysis = cp.analyse_coupling(
        conformation, modes=10, top=5, sites=network_nodes,
        routes=[("active", "flaps")], sensitivity=False,
    )
    assert analysis.network.n_nodes == 198
    assert len(analysis.pathways) == 1
    _name, cheap, wide = analysis.pathways[0]
    assert cheap.length == 5
    labels = cheap.labels
    assert labels[0] == "ASP25 A"
    assert labels[-1] == "ILE50 A"
    assert "ASP25 B" in labels
    assert any(label in labels for label in ("THR80 B", "VAL82 B", "ILE84 B")), labels
    assert 0.8 < cheap.bottleneck < 0.9
    # The maximin route is stronger at every step but much longer: maximin
    # maximises the weakest link, it does not minimise the number of steps.
    assert wide.length > cheap.length
    assert wide.bottleneck > cheap.bottleneck
    block = analysis.route_coupling["active -> flaps"]
    assert block["n_pairs"] == 2  # ASP25 A against ILE50 A and GLY51 A
    assert 0.3 < block["max_direct"] < 0.4


@pytest.mark.slow
def test_the_estrogen_pocket_to_helix_12_coupling_is_contact_geometry(data_dir):
    """The honest test, and it refines the earlier falsification.

    The coupling analysis *does* flag helix 12 as coupled to the pocket, under both
    site definitions tried.  With the six-residue explicit site the best route from
    THR347 to LEU525 is four steps (THR347 -> MET343 -> GLU419 -> GLY420 ->
    LEU525, bottleneck |C| 0.869) and the direct block reaches |C| 0.851 with a mean
    of 0.592 over 30 pairs; with the fourteen-residue box-derived site the route
    runs through ASP351 (THR347 -> ASN348 -> ASP351 -> LEU540 -> LEU536) with a
    direct block of 0.885 max and 0.486 mean over 52 pairs.

    But the distance profile says what any pair at that separation scores: 0.928 on
    average below 6 A, 0.784 at 8-10 A, 0.708 at 10-12 A, 0.549 at 12-16 A.  A mean
    of 0.592 sits below every near-contact baseline, so the flag is contact
    geometry rather than a special coupling -- and that is consistent with the
    direction result (site overlap 0.03) instead of contradicting it: the modes
    couple the pocket to helix 12 because they are neighbours, in a direction that
    is not the experimental one.
    """
    conformation = bundled(data_dir, "3ERT.pdb")
    network = {"site": [
        ("A", 343, "MET"), ("A", 345, "LEU"), ("A", 346, "LEU"), ("A", 347, "THR"),
        ("A", 348, "ASN"), ("A", 349, "LEU"),
    ], "h12": [
        ("A", 525, "LEU"), ("A", 526, "TYR"), ("A", 528, "MET"), ("A", 529, "LYS"),
        ("A", 536, "LEU"),
    ]}
    analysis = cp.analyse_coupling(
        conformation, modes=10, top=5, sites=network, routes=[("site", "h12")],
        sensitivity=False,
    )
    _name, cheap, wide = analysis.pathways[0]
    # The endpoints are each site's residue nearest the other, which is measured,
    # not chosen: THR347 of the pocket and LEU525 at the start of helix 12.
    assert cheap.labels[0] == "THR347 A"
    assert cheap.labels[-1] == "LEU525 A"
    assert cheap.length == 4
    assert cheap.bottleneck == pytest.approx(0.869, abs=0.01)
    assert wide.length > cheap.length and wide.bottleneck > cheap.bottleneck
    block = analysis.route_coupling["site -> h12"]
    assert block["n_pairs"] == 30
    assert block["shared_residues"] == []
    assert block["max_direct"] > 0.8
    # The baseline: what any pair scores at short range, and at long range.
    profile = {row["from"]: row for row in analysis.profile}
    assert profile[0.0]["mean_correlation"] > 0.9
    assert profile[8.0]["mean_correlation"] > 0.75
    assert profile[30.0]["mean_correlation"] < 0.35
    # The pocket-to-helix-12 mean is below the near-contact baselines: it is not a
    # special coupling, it is what neighbours score.
    assert block["mean_direct"] < profile[0.0]["mean_correlation"]
    assert block["mean_direct"] < profile[8.0]["mean_correlation"]


@pytest.mark.slow
def test_trypsin_couples_the_catalytic_serine_to_the_pocket_wall(data_dir):
    """SER195 (catalytic) with VAL213 (S1 wall) at |C| 0.989, 14 apart in sequence."""
    conformation = bundled(data_dir, "3PTB.pdb")
    analysis = cp.analyse_coupling(conformation, modes=10, top=20, sensitivity=False)
    assert analysis.network.n_nodes == 223
    labels = {(pair.label_a, pair.label_b): pair for pair in analysis.couplings}
    assert ("SER195 A", "VAL213 A") in labels
    pair = labels[("SER195 A", "VAL213 A")]
    assert pair.correlation == pytest.approx(0.989, abs=0.002)
    assert pair.separation == 14
    # A long-range coupling survives: GLY43 to the oxyanion-hole loop, 151 apart.
    long_range = [
        pair for pair in analysis.couplings
        if abs(pair.separation) > 100 and pair.correlation > 0.95
    ]
    assert long_range, "the mode set couples distant regions too"


@pytest.mark.slow
def test_the_coupling_list_depends_on_the_mode_count(data_dir):
    """The finding item 1 asks for: the top list is truncation-dependent."""
    conformation = bundled(data_dir, "1HVR.pdb")
    analysis = cp.analyse_coupling(conformation, modes=10, top=20, sensitivity=True)
    rows = {row["modes"]: row for row in analysis.sensitivity["rows"]}
    assert set(rows) >= {3, 5, 10, 20, 50}
    assert rows[10]["jaccard_with_reference"] == pytest.approx(1.0)
    for count in (3, 5, 20, 50):
        assert rows[count]["jaccard_with_reference"] < 0.7, count
    # The matrix as a whole is much more stable than its top list: the coarse
    # coupling structure survives the truncation even where the ranking does not.
    for count in (5, 20, 50):
        assert rows[count]["spearman_with_reference"] > 0.8


def test_cli_ensemble_coupling_reports_json(data_dir, tmp_path):
    from odock.cli import main

    path = data_dir / "1HVR.pdb"
    if not path.exists():
        pytest.skip("missing the bundled 1HVR structure")
    target = tmp_path / "coupling.json"
    code = main(
        [
            "ensemble", "coupling", "-r", str(path),
            "--site", "active=ASP25", "--site", "flaps=ILE50,GLY51",
            "--route", "active=flaps", "--modes", "5",
            "--no-sensitivity", "--json-out", str(target), "-q",
        ]
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["network"]["nodes"] == 198
    assert payload["parameters"]["modes"] == 5
    assert payload["sites"]["active"] == [24]  # ASP25 A is node 24
    assert payload["pathways"][0]["lowest_cost"]["path"][0] == "ASP25 A"
    assert payload["pathways"][0]["bottleneck"]["bottleneck"] > 0.9
    assert payload["route_coupling"]["active -> flaps"]["n_pairs"] == 2
    assert any(row["from"] == 0.0 for row in payload["distance_profile"])


def test_cli_ensemble_coupling_needs_one_structure(data_dir):
    from odock.cli import main

    path = data_dir / "1HVR.pdb"
    if not path.exists():
        pytest.skip("missing the bundled 1HVR structure")
    with pytest.raises(SystemExit):
        main(["ensemble", "coupling", "-r", str(path), str(path), "-q"])

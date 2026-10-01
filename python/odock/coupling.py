# SPDX-License-Identifier: GPL-3.0-or-later
"""Residue coupling and pathways from the elastic network's modes.

:mod:`odock.modes` falsified one use of an ANM with numbers: the low-frequency
modes of 3ERT are essentially orthogonal to the antagonist-to-agonist difference
(site overlaps −0.002/−0.010/−0.010; best of the first thirty +0.296).  They are
not a conformational-change predictor.  But a mode set is also a **coupling**
description — which residues move together — and that is a different question, one
the same eigenvectors can answer:

* **coupling** (:func:`cross_correlation`): the ANM cross-correlation implied by
  the lowest modes, ``C_ij = <dR_i . dR_j> / sqrt(<dR_i^2><dR_j^2>)``, a number in
  ``[-1, 1]`` per residue pair;
* **the mode-count sensitivity** (:func:`mode_count_sensitivity`): the same
  analysis at several mode counts, because a coupling list that changes completely
  when the count changes is a fact about the truncation, not about the protein;
* **pathways** (:func:`pathway`): the best route between two sites through that
  coupling, reported both as the lowest-cost path and as the *bottleneck* path
  (the route that maximises its weakest link, which is what communication means).

Four statements belong in front of every number this module produces, and they are
in ``docs/COUPLING.md`` as a section rather than as a footnote:

1. a harmonic coupling is **not** a signal-transduction mechanism;
2. correlated motion is **not** causality — two residues moving together may share
   a hinge rather than a message;
3. a short path in a network is **not** a physical channel; it is a route through
   a graph built from one structure's contacts;
4. a coupling computed from one structure inherits every limitation already
   documented for one-structure ensembles (harmonic, one connectivity, no
   populations, no anharmonic transitions).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ensemble import Conformation, EnsembleError, read_conformations
from .modes import DEFAULT_CUTOFF, ElasticNetwork, build_network

__all__ = [
    "DEFAULT_MIN_SEPARATION",
    "CouplingAnalysis",
    "CouplingPair",
    "Pathway",
    "add_coupling_subparser",
    "cross_correlation",
    "mode_count_sensitivity",
    "pathway",
    "strongest_couplings",
]

#: Sequence separation (in residues) below which a pair is skipped when *ranking*
#: couplings: neighbours move together because they are bonded, which is not the
#: question.
DEFAULT_MIN_SEPARATION = 3

#: Mode counts the sensitivity analysis compares.
SENSITIVITY_COUNTS: Tuple[int, ...] = (3, 5, 10, 20, 50)


@dataclass
class CouplingPair:
    """One coupled residue pair."""

    first: int
    second: int
    label_a: str
    label_b: str
    correlation: float
    separation: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "first": int(self.first),
            "second": int(self.second),
            "label_a": self.label_a,
            "label_b": self.label_b,
            "correlation": self.correlation,
            "separation": int(self.separation),
        }

    def row(self) -> List[str]:
        return [
            self.label_a,
            self.label_b,
            f"{self.correlation:+.3f}",
            str(self.separation),
        ]


@dataclass
class Pathway:
    """A route through the coupling graph between two sites."""

    source: int
    target: int
    source_label: str
    target_label: str
    nodes: List[int] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    #: The weakest |correlation| along the route: the bottleneck.
    bottleneck: float = float("nan")
    #: The sum of the edge costs (``1 / |correlation|``) along the route.
    cost: float = float("nan")
    kind: str = "lowest cost"

    @property
    def length(self) -> int:
        return max(0, len(self.nodes) - 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source_label,
            "target": self.target_label,
            "length": self.length,
            "bottleneck": self.bottleneck,
            "cost": self.cost,
            "path": list(self.labels),
        }

    def text(self) -> str:
        return (
            f"{self.kind}: {self.source_label} -> {self.target_label}, "
            f"{self.length} step(s), bottleneck |C| {self.bottleneck:.3f}\n"
            f"  {' -> '.join(self.labels)}"
        )


def cross_correlation(
    network: ElasticNetwork,
    *,
    modes: Optional[int] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """The ANM cross-correlation matrix implied by the lowest `modes`.

    ``C_ij = <dR_i . dR_j> / sqrt(<dR_i^2> <dR_j^2>)`` with
    ``<dR_i . dR_j> = sum_k (T / lambda_k) (v_k,i . v_k,j)`` over the selected
    modes — the standard Gaussian-network result.  The diagonal is 1 and every
    entry lies in ``[-1, 1]``: ``+1`` means two residues move along the same
    direction with the same amplitude, ``-1`` means opposite.  The temperature
    cancels in the normalisation, so it never changes the answer; it is kept
    because the covariance it belongs to is only defined up to it.
    """
    count = network.selected if modes is None else int(modes)
    count = max(0, min(count, network.n_modes))
    n = network.n_nodes
    covariance = np.zeros((n, n), dtype=float)
    for index in range(count):
        eigenvalue = float(network.eigenvalues[index])
        if eigenvalue <= 1e-12:
            continue
        vector = np.asarray(network.eigenvectors[:, index], dtype=float).reshape(n, 3)
        covariance += (float(temperature) / eigenvalue) * (vector @ vector.T)
    diagonal = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(diagonal, diagonal)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.where(denominator > 0, covariance / denominator, 0.0)
    return np.clip(correlation, -1.0, 1.0)


def strongest_couplings(
    correlation: np.ndarray,
    labels: Sequence[str],
    *,
    top: int = 20,
    min_separation: int = DEFAULT_MIN_SEPARATION,
    absolute: bool = True,
) -> List[CouplingPair]:
    """The `top` strongest couplings, skipping near neighbours.

    `absolute` ranks by ``|C|``, so a strongly *anti*-correlated pair (two parts of
    a hinge swinging apart) counts as coupled, which is the physical statement.
    Set it false to rank by the signed value and see only the in-phase pairs.
    """
    count = correlation.shape[0]
    pairs: List[CouplingPair] = []
    for i in range(count):
        for j in range(i + 1, count):
            separation = j - i
            if separation < int(min_separation):
                continue
            value = float(correlation[i, j])
            key = abs(value) if absolute else value
            pairs.append(
                CouplingPair(
                    first=i, second=j, label_a=str(labels[i]), label_b=str(labels[j]),
                    correlation=value, separation=separation,
                )
            )
            pairs[-1]._key = key  # type: ignore[attr-defined]
    pairs.sort(key=lambda pair: -getattr(pair, "_key", 0.0))
    return pairs[: max(1, int(top))]


def mode_count_sensitivity(
    network: ElasticNetwork,
    labels: Sequence[str],
    *,
    counts: Sequence[int] = SENSITIVITY_COUNTS,
    top: int = 20,
    min_separation: int = DEFAULT_MIN_SEPARATION,
    reference: int = 3,
) -> Dict[str, Any]:
    """How much the coupling answer depends on how many modes are kept.

    For each mode count: the top pairs, the Jaccard overlap of that set with the
    count-`reference` set, and the Spearman correlation of the full off-diagonal
    coupling vectors between the two.  If the overlap collapses as the count
    grows, the coupling list is a property of the truncation and the report must
    say so — that is the finding, not a footnote.
    """
    from . import consensus

    usable = [int(value) for value in counts if 0 < int(value) <= network.n_modes]
    if reference not in usable:
        usable = [reference] + usable if 0 < reference <= network.n_modes else usable
    matrices = {count: cross_correlation(network, modes=count) for count in usable}
    masks: Dict[int, np.ndarray] = {}
    for count in usable:
        mask = np.zeros_like(matrices[count], dtype=bool)
        for i in range(mask.shape[0]):
            for j in range(i + int(min_separation), mask.shape[0]):
                mask[i, j] = True
        masks[count] = mask
    reference_pairs = {
        (pair.first, pair.second)
        for pair in strongest_couplings(
            matrices[reference], labels, top=top, min_separation=min_separation
        )
    }
    rows = []
    for count in usable:
        pairs = {
            (pair.first, pair.second)
            for pair in strongest_couplings(
                matrices[count], labels, top=top, min_separation=min_separation
            )
        }
        union = reference_pairs | pairs
        jaccard = len(reference_pairs & pairs) / len(union) if union else 0.0
        mask = masks[count] & masks[reference]
        rho = consensus.spearman(
            np.abs(matrices[count][mask]), np.abs(matrices[reference][mask])
        )
        rows.append(
            {
                "modes": count,
                "top_pairs": sorted(pairs),
                "jaccard_with_reference": jaccard,
                "spearman_with_reference": rho,
                "max_correlation": float(np.abs(matrices[count][mask]).max()) if mask.any() else 0.0,
            }
        )
    return {
        "reference_modes": int(reference),
        "counts": usable,
        "rows": rows,
        "n_pairs_compared": int(masks[reference].sum()),
    }


def pathway(
    correlation: np.ndarray,
    labels: Sequence[str],
    source: int,
    target: int,
    *,
    edge_threshold: Optional[float] = None,
) -> Tuple[Pathway, Pathway]:
    """Two routes between `source` and `target` through the coupling graph.

    * **lowest cost** — Dijkstra with edge cost ``1 / |C|`` over the edges that
      survive `edge_threshold`: the cheapest sum of *strong* steps;
    * **bottleneck** — the route that maximises its weakest link.  That is the
      right definition of "the strongest path" for communication, and it is
      computed exactly on the maximum spanning tree of ``|C|``: the maximin route
      between two nodes always lies on that tree.

    `edge_threshold`
        Edges weaker than this are removed before the lowest-cost search.  ``None``
        uses the 95th percentile of the off-diagonal ``|C|``, a data-driven cut
        that is reported: with the complete graph a single weak edge is a legal
        one-step "path", which is how a route of one step with ``|C| = 0.34`` won
        against a route of forty strong ones on the first run of this analysis.
        The bottleneck path uses the full graph, because the maximum spanning tree
        already prefers strong edges by construction.

    Both are graph constructs over one structure's correlation matrix.  A short
    route is not a channel and a strong coupling is not a signal; the docs say so
    before any of these numbers are read.
    """
    n = correlation.shape[0]
    if not 0 <= source < n or not 0 <= target < n:
        raise EnsembleError(f"pathway endpoints must be in [0, {n}), got {source}, {target}")
    strength = np.abs(np.asarray(correlation, dtype=float))
    np.fill_diagonal(strength, 0.0)
    off_diagonal = strength[np.triu_indices(n, k=1)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    if edge_threshold is None:
        edge_threshold = float(np.quantile(finite, 0.95)) if finite.size else 0.0
    threshold = float(edge_threshold)
    allowed = strength >= threshold
    np.fill_diagonal(allowed, False)

    # Dijkstra on 1/|C| over the allowed edges.
    cost = np.full(n, math.inf)
    cost[source] = 0.0
    previous = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    for _ in range(n):
        candidates = np.where(~visited, cost, math.inf)
        node = int(np.argmin(candidates))
        if not math.isfinite(candidates[node]):
            break
        visited[node] = True
        if node == target:
            break
        for neighbour in np.flatnonzero(allowed[node]):
            if visited[neighbour] or neighbour == node:
                continue
            step = cost[node] + 1.0 / max(strength[node, neighbour], 1e-12)
            if step < cost[neighbour]:
                cost[neighbour] = step
                previous[neighbour] = node
    cheap_nodes = _trace(previous, source, target)
    cheap_bottleneck = min(
        (strength[cheap_nodes[i], cheap_nodes[i + 1]] for i in range(len(cheap_nodes) - 1)),
        default=float("nan"),
    )
    cheap = Pathway(
        source=source, target=target, source_label=str(labels[source]),
        target_label=str(labels[target]), nodes=cheap_nodes,
        labels=[str(labels[node]) for node in cheap_nodes],
        bottleneck=cheap_bottleneck, cost=float(cost[target]),
        kind=f"lowest cost (edges >= {threshold:.3f})",
    )

    # Bottleneck (maximin) path on the maximum spanning tree of |C| (Prim).
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, -1.0)
    parent = np.full(n, -1, dtype=int)
    best[source] = math.inf
    for _ in range(n):
        candidates = np.where(~in_tree, best, -1.0)
        node = int(np.argmax(candidates))
        if candidates[node] < 0:
            break
        in_tree[node] = True
        for neighbour in range(n):
            if in_tree[neighbour] or neighbour == node:
                continue
            if strength[node, neighbour] > best[neighbour]:
                best[neighbour] = strength[node, neighbour]
                parent[neighbour] = node
    wide_nodes = _trace(parent, source, target)
    wide = Pathway(
        source=source, target=target, source_label=str(labels[source]),
        target_label=str(labels[target]), nodes=wide_nodes,
        labels=[str(labels[node]) for node in wide_nodes],
        bottleneck=float(best[target]),
        cost=float(sum(
            1.0 / max(strength[wide_nodes[i], wide_nodes[i + 1]], 1e-12)
            for i in range(len(wide_nodes) - 1)
        )) if len(wide_nodes) > 1 else 0.0,
        kind="bottleneck (maximin)",
    )
    return cheap, wide


def _trace(previous: np.ndarray, source: int, target: int) -> List[int]:
    """Walk the predecessor chain back from `target`; empty when unreachable."""
    if source == target:
        return [source]
    nodes = [target]
    node = target
    guard = 0
    while node != source:
        node = int(previous[node])
        if node < 0:
            return []
        nodes.append(node)
        guard += 1
        if guard > previous.size + 1:  # pragma: no cover - defensive
            return []
    nodes.reverse()
    return nodes


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


@dataclass
class CouplingAnalysis:
    """The coupling picture of one structure."""

    label: str
    network: ElasticNetwork
    correlation: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    couplings: List[CouplingPair] = field(default_factory=list)
    sensitivity: Dict[str, Any] = field(default_factory=dict)
    profile: List[Dict[str, Any]] = field(default_factory=list)
    sites: Dict[str, List[int]] = field(default_factory=dict)
    site_labels: Dict[str, List[str]] = field(default_factory=dict)
    pathways: List[Tuple[str, Pathway, Pathway]] = field(default_factory=list)
    #: Per route: the *direct* coupling between the two sites, which is the
    #: crispest answer to "are these two sites coupled at all?" -- a pathway can
    #: always be drawn through a connected graph, but a weak direct block means the
    #: route is doing the work, not the coupling.
    route_coupling: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def site_names(self) -> List[str]:
        return list(self.sites)

    def coupling_table(self, limit: int = 20) -> str:
        headers = ["residue A", "residue B", "correlation", "separation"]
        rows = [pair.row() for pair in self.couplings[: max(1, limit)]]
        return _table(headers, rows)

    def sensitivity_table(self) -> str:
        headers = ["modes", "top-pair Jaccard vs reference", "Spearman of |C| vs reference", "max |C|"]
        rows = [
            [
                str(row["modes"]),
                f"{row['jaccard_with_reference']:.2f}",
                f"{row['spearman_with_reference']:.3f}",
                f"{row['max_correlation']:.3f}",
            ]
            for row in self.sensitivity.get("rows", [])
        ]
        return _table(headers, rows)

    def site_table(self) -> str:
        headers = ["site", "residues", "nodes"]
        rows = [
            [name, ", ".join(self.site_labels.get(name, [])[:6]), str(len(nodes))]
            for name, nodes in self.sites.items()
        ]
        return _table(headers, rows)

    def text(self, *, limit: int = 20) -> str:
        lines = [
            f"OpenDocking coupling analysis — {self.label}",
            "",
            "network: "
            + ", ".join(f"{key}={value}" for key, value in self.parameters.items()),
            "",
            "strongest couplings (|C|, skipping pairs closer than "
            f"{self.parameters.get('min_separation')} in sequence):",
            self.coupling_table(limit),
            "",
            "does the answer depend on how many modes are kept?",
            self.sensitivity_table(),
            "",
            f"reference mode count: {self.sensitivity.get('reference_modes')}; "
            f"{self.sensitivity.get('n_pairs_compared')} residue pair(s) compared",
        ]
        if self.profile:
            lines += [
                "",
                "mean |C| against the C-alpha distance (the baseline a single "
                "coupling has to be read against: an elastic network couples what is "
                "near it):",
                _table(
                    ["from (A)", "to (A)", "pairs", "mean |C|", "max |C|"],
                    [
                        [
                            f"{row['from']:.0f}",
                            "--" if row["to"] is None else f"{row['to']:.0f}",
                            str(row["n_pairs"]),
                            f"{row['mean_correlation']:.3f}",
                            f"{row['max_correlation']:.3f}",
                        ]
                        for row in self.profile
                    ],
                ),
            ]
        if self.sites:
            lines += ["", "sites:", self.site_table()]
        for name, cheap, wide in self.pathways:
            lines += ["", f"pathway {name}:", cheap.text(), wide.text()]
            block = self.route_coupling.get(name)
            if block:
                lines.append(
                    f"  direct coupling between the two sites: max |C| "
                    f"{block['max_direct']:.3f}, mean {block['mean_direct']:.3f} over "
                    f"{block['n_pairs']} pair(s); endpoints "
                    f"{block['source_node']} and {block['target_node']}"
                )
                if block.get("shared_residues"):
                    lines.append(
                        "  the two sites share "
                        + ", ".join(block["shared_residues"])
                        + " (excluded from both ends: a residue cannot be the "
                        "start and the end of a pathway)"
                    )
                lines.append(
                    "  a pathway can always be drawn through a connected graph; the "
                    "direct block is the number that says whether the two sites are "
                    "coupled at all."
                )
        lines += [
            "",
            "what this is not: a harmonic coupling is not a signal-transduction "
            "mechanism, correlated motion is not causality (two residues can share a "
            "hinge rather than a message), a short path in a graph built from one "
            "structure's contacts is not a physical channel, and a coupling from one "
            "structure inherits every limitation of one-structure models.",
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "network": self.network.as_dict(),
            "parameters": dict(self.parameters),
            "couplings": [pair.as_dict() for pair in self.couplings],
            "distance_profile": list(self.profile),
            "sensitivity": {
                "reference_modes": self.sensitivity.get("reference_modes"),
                "counts": self.sensitivity.get("counts", []),
                "n_pairs_compared": self.sensitivity.get("n_pairs_compared"),
                "rows": [
                    {key: value for key, value in row.items() if key != "top_pairs"}
                    for row in self.sensitivity.get("rows", [])
                ],
            },
            "sites": {name: list(nodes) for name, nodes in self.sites.items()},
            "site_labels": {name: list(values) for name, values in self.site_labels.items()},
            "pathways": [
                {"name": name, "lowest_cost": cheap.as_dict(), "bottleneck": wide.as_dict()}
                for name, cheap, wide in self.pathways
            ],
            "route_coupling": {
                name: dict(values) for name, values in self.route_coupling.items()
            },
        }


def distance_profile(
    network: ElasticNetwork,
    correlation: np.ndarray,
    *,
    bins: Sequence[float] = (0.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 1e9),
) -> List[Dict[str, Any]]:
    """Mean ``|C|`` as a function of the Cα–Cα distance.

    This is the baseline every coupling number needs.  An elastic network couples
    residues that are *near each other* — they are joined by springs — so a strong
    correlation between two adjacent regions is guaranteed by construction and
    says nothing about a functional relationship.  Publishing the falloff lets a
    reader place any single coupling ("0.885 between the pocket and helix 12")
    against what any pair at that separation scores, which here is the difference
    between a finding and an artefact of contact geometry.
    """
    strength = np.abs(np.asarray(correlation, dtype=float))
    n = strength.shape[0]
    rows = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        values = []
        for i in range(n):
            for j in range(i + 1, n):
                distance = float(np.linalg.norm(network.coords[i] - network.coords[j]))
                if lower <= distance < upper:
                    values.append(strength[i, j])
        rows.append(
            {
                "from": float(lower),
                "to": None if upper >= 1e9 else float(upper),
                "n_pairs": len(values),
                "mean_correlation": float(np.mean(values)) if values else float("nan"),
                "max_correlation": float(np.max(values)) if values else float("nan"),
            }
        )
    return rows


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def render(cells):
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def _site_nodes(network: ElasticNetwork, keys: Sequence[Tuple[str, int, str]]) -> List[int]:
    wanted = set(keys)
    return [index for index, key in enumerate(network.keys) if key in wanted]


def analyse_coupling(
    conformation: Conformation,
    *,
    cutoff: float = DEFAULT_CUTOFF,
    modes: int = 10,
    top: int = 20,
    min_separation: int = DEFAULT_MIN_SEPARATION,
    sites: Optional[Dict[str, Sequence[Tuple[str, int, str]]]] = None,
    routes: Sequence[Tuple[str, str]] = (),
    sensitivity: bool = True,
) -> CouplingAnalysis:
    """Couplings, the mode-count sensitivity, and the requested pathways.

    `sites` maps a name to residue keys (``(chain, res_id, res_name)``), and
    `routes` names pairs of sites to connect.  The endpoint of a route inside a
    site is the node nearest the other site's centroid, which is deterministic and
    stated rather than optimised.
    """
    network = build_network(conformation, cutoff=cutoff, modes=modes)
    correlation = cross_correlation(network, modes=modes)
    couplings = strongest_couplings(
        correlation, network.labels, top=top, min_separation=min_separation
    )
    report = {
        "cutoff": float(cutoff),
        "modes": int(modes),
        "modes_available": int(network.n_modes),
        "top": int(top),
        "min_separation": int(min_separation),
        "nodes": int(network.n_nodes),
    }
    sensitivity_report = (
        mode_count_sensitivity(
            network, network.labels, top=top, min_separation=min_separation,
            reference=int(modes),
        )
        if sensitivity
        else {}
    )
    site_nodes: Dict[str, List[int]] = {}
    site_labels: Dict[str, List[str]] = {}
    for name, keys in (sites or {}).items():
        nodes = _site_nodes(network, keys)
        site_nodes[name] = nodes
        site_labels[name] = [network.labels[node] for node in nodes]
    connections: List[Tuple[str, Pathway, Pathway]] = []
    route_coupling: Dict[str, Dict[str, Any]] = {}
    for first, second in routes:
        if first not in site_nodes or second not in site_nodes:
            continue
        if not site_nodes[first] or not site_nodes[second]:
            continue
        # The two sites must be disjoint, or the "path" is a zero-step walk
        # between a residue and itself -- which is exactly what a box-derived site
        # and helix 12 do here, because LEU525 lines the pocket *and* is part of
        # helix 12.  The overlap is reported, and each endpoint is chosen from the
        # residues the two sites do not share.
        shared = set(site_nodes[first]) & set(site_nodes[second])
        own = [node for node in site_nodes[first] if node not in shared]
        other = [node for node in site_nodes[second] if node not in shared]
        if not own or not other:
            continue
        source = _nearest_node(network, own, other)
        target = _nearest_node(network, other, own)
        cheap, wide = pathway(correlation, network.labels, source, target)
        connections.append((f"{first} -> {second}", cheap, wide))
        block = np.abs(correlation[np.ix_(own, other)]).reshape(-1)
        route_coupling[f"{first} -> {second}"] = {
            "max_direct": float(block.max()) if block.size else float("nan"),
            "mean_direct": float(block.mean()) if block.size else float("nan"),
            "n_pairs": int(block.size),
            "shared_residues": [str(network.labels[node]) for node in sorted(shared)],
            "source_node": str(network.labels[source]),
            "target_node": str(network.labels[target]),
        }
    return CouplingAnalysis(
        label=conformation.label,
        network=network,
        correlation=correlation,
        couplings=couplings,
        sensitivity=sensitivity_report,
        profile=distance_profile(network, correlation),
        sites=site_nodes,
        site_labels=site_labels,
        pathways=connections,
        route_coupling=route_coupling,
        parameters=report,
    )


def _nearest_node(
    network: ElasticNetwork, own: Sequence[int], other: Sequence[int]
) -> int:
    """The node of `own` closest to the centroid of `other`."""
    centre = network.coords[list(other)].mean(axis=0)
    distances = [
        float(np.linalg.norm(network.coords[node] - centre)) for node in own
    ]
    return int(own[int(np.argmin(distances))])


# ---------------------------------------------------------------------------
# The command line: `odock ensemble coupling`
# ---------------------------------------------------------------------------


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def _parse_sites(specs: Sequence[str], conformation: Conformation) -> Dict[str, List[Tuple[str, int, str]]]:
    """``NAME=RES[,RES...]`` into residue keys, resolving names like ``A:189``."""
    from .ensemble import parse_site_spec, select_site

    out: Dict[str, List[Tuple[str, int, str]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise EnsembleError(
                f"cannot read the site {spec!r}: write it as NAME=ASP189,SER190"
            )
        name, _, body = spec.partition("=")
        selected = select_site(conformation, site=body)
        out[name.strip()] = [entry.residue.key for entry in selected]
    del parse_site_spec
    return out


def cmd_ensemble_coupling(args) -> int:
    """``odock ensemble coupling``: residue coupling and pathways from the modes."""
    from .ensemble import _cli_box, _cli_write_json, ligand_coords, select_site

    paths = [
        str(path)
        for entry in (getattr(args, "receptor", None) or [])
        for path in (entry if isinstance(entry, (list, tuple)) else [entry])
    ]
    if not paths:
        raise SystemExit("error: pass one -r/--receptor FILE")
    if len(paths) > 1:
        raise SystemExit(
            "error: `ensemble coupling` analyses *one* structure; run it per "
            "structure and compare"
        )
    try:
        conformation = read_conformations(paths, keep_water=False)[0]
        box = None
        if args.box_ligand:
            from .prepare import box_from_points

            points = ligand_coords(conformation, args.box_ligand)
            box = box_from_points(points, buffer=float(args.buffer), spacing=0.375)
            _cli_eprint(f"box: {box} (from residue {args.box_ligand})")
        elif getattr(args, "box", None) or (args.center and args.size):
            box = _cli_box(args)
        sites = _parse_sites(args.site or [], conformation)
        routes = []
        for route in args.route or []:
            first, _, second = route.partition("=")
            routes.append((first.strip(), second.strip()))
        if box is not None and "site" not in sites:
            # The box-derived site, so a command that names one site explicitly
            # still gets the pocket the box describes as the other end.
            site_residues = select_site(
                conformation, box=box, radius=float(args.site_radius),
                max_residues=int(args.max_site_residues),
            )
            sites["site"] = [entry.residue.key for entry in site_residues]
            centre = np.array(
                [
                    [atom.x, atom.y, atom.z]
                    for entry in site_residues
                    for atom in entry.residue.atoms
                    if not atom.is_hydrogen
                ],
                dtype=float,
            )
            if centre.size:
                centre = centre.mean(axis=0)
                candidates = []
                for chain, residues in conformation.chain_residues().items():
                    for residue in residues:
                        if not residue.is_standard:
                            continue
                        ca = next(
                            (
                                atom
                                for atom in residue.atoms
                                if atom.name == "CA"
                                and str(atom.element).strip().upper() == "C"
                            ),
                            None,
                        )
                        if ca is None:
                            continue
                        candidates.append(
                            (
                                float(
                                    np.linalg.norm(
                                        np.array([ca.x, ca.y, ca.z]) - centre
                                    )
                                ),
                                residue.key,
                            )
                        )
                if candidates:
                    candidates.sort(key=lambda item: -item[0])
                    sites["far"] = [candidates[0][1]]
            if not routes:
                routes = [("site", "far")]
        if not routes:
            for first, second in (("site", "far"),):
                if first in sites and second in sites:
                    routes = [(first, second)]
        missing = [
            name
            for route in routes
            for name in route
            if name not in sites
        ]
        if missing:
            _cli_eprint(
                "odock ensemble coupling: no pathway for "
                + ", ".join(sorted(set(missing)))
                + " (not a named site); pass --site NAME=RES[,RES...] for it"
            )
        analysis = analyse_coupling(
            conformation,
            cutoff=float(args.cutoff),
            modes=int(args.modes),
            top=int(args.top),
            min_separation=int(args.min_separation),
            sites=sites,
            routes=routes,
            sensitivity=not args.no_sensitivity,
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble coupling: error: {exc}")
        return int(exc.code)

    if not args.quiet:
        print(analysis.text(limit=int(args.top)))
        print()
    if args.json_out:
        _cli_write_json(args.json_out, analysis.as_dict())
    return 0


def add_coupling_subparser(ensub: Any) -> None:
    """Register ``odock ensemble coupling`` on the ensemble subparsers."""
    parser = ensub.add_parser(
        "coupling",
        help="residue coupling and pathways from the elastic network's modes",
        description=(
            "Compute the ANM cross-correlation implied by the lowest modes, list the "
            "strongest coupled residue pairs, measure how much that list depends on "
            "how many modes are kept, and find the best route between two sites "
            "through the coupling -- both the lowest-cost route and the bottleneck "
            "route.  A harmonic coupling is not a mechanism and a short path is not "
            "a channel; the report says so before the numbers."
        ),
    )
    parser.add_argument(
        "-r", "--receptor", action="append", nargs="+", required=True, metavar="FILE",
        help="the one structure to analyse",
    )
    parser.add_argument("--box", help="box JSON written by `odock box`")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--box-ligand", metavar="RESNAME", help="derive the site from this residue")
    parser.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    parser.add_argument(
        "--site", action="append", metavar="NAME=RES[,RES...]",
        help="a named site, repeatable, e.g. --site s1=ASP189,SER190 "
             "--site flaps=ILE50,GLY51; the default derives 'site' from the box and "
             "'far' as the residue farthest from it",
    )
    parser.add_argument(
        "--route", action="append", metavar="NAME=NAME",
        help="a pathway to report between two named sites (default site=far)",
    )
    parser.add_argument("--site-radius", type=float, default=8.0, help="site selection radius (Å)")
    parser.add_argument("--max-site-residues", type=int, default=40)
    parser.add_argument(
        "--cutoff", type=float, default=DEFAULT_CUTOFF,
        help="ANM Cα-Cα cutoff (Å)",
    )
    parser.add_argument(
        "--modes", type=int, default=10,
        help="how many low-frequency modes define the covariance",
    )
    parser.add_argument("--top", type=int, default=20, help="how many pairs to list")
    parser.add_argument(
        "--min-separation", type=int, default=DEFAULT_MIN_SEPARATION,
        help="skip pairs closer than this in sequence when ranking couplings",
    )
    parser.add_argument(
        "--no-sensitivity", action="store_true",
        help="skip the mode-count sensitivity analysis (faster)",
    )
    parser.add_argument("--json-out", help="write the whole analysis as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no report")
    parser.set_defaults(func=cmd_ensemble_coupling)

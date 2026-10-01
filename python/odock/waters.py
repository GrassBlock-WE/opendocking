# SPDX-License-Identifier: GPL-3.0-or-later
"""Water networks, conserved waters, and what a pose does to them.

Crystallographic waters are the first thing a docking script throws away and the
first thing a medicinal chemist asks about: a water that is *conserved* across
structures and bridges the ligand to the protein is a different object from one
that happens to be in the crystal, and a pose that displaces a conserved water is
paying for it.  This module measures both, on the ensemble machinery already in
the project:

1. **The network** (:func:`water_network`): every water of a conformation, the
   water–water hydrogen bonds, the water–protein contacts, and the connected
   components of that graph.
2. **Sites across an ensemble** (:func:`water_sites`): the waters of every
   conformation pooled and clustered into positions, each classified
   **conserved**, **moved** or **displaced** by a documented criterion, with the
   ligand atoms that occupy a displaced site and the protein contacts the water
   was making.
3. **A per-pose displacement count** (:func:`pose_displacement`) and its
   correlation with affinity, reported **with n and a bootstrap confidence
   interval** (:func:`correlate`) — and reported as unresolvable when n is too
   small, which for the bundled systems it is.

Everything is geometric and explicit.  There is no water-placement model here:
this measures the waters a structure *has*, not the ones it would have.
"""

from __future__ import annotations

import argparse
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ensemble import Conformation, EnsembleError, WATER_NAMES, read_conformations
from .prepare import BoxSpec

__all__ = [
    "DEFAULT_HBOND_CUTOFF",
    "DEFAULT_MATCH_RADIUS",
    "DEFAULT_OCCLUSION_RADIUS",
    "DEFAULT_OCCUPANCY",
    "WaterObservation",
    "WaterSite",
    "WaterNetwork",
    "WaterAnalysis",
    "add_waters_subparser",
    "compare_water_sites",
    "correlate",
    "pose_displacement",
    "water_network",
    "water_sites",
]

#: O···O distance (Å) below which two waters are hydrogen bonded.  3.5 Å is the
#: standard crystallographic criterion for a water–water H-bond; beyond it the
#: pair is a contact, not a bridge.
DEFAULT_HBOND_CUTOFF = 3.5

#: Distance (Å) within which a water of one conformation is the *same site* as a
#: water of another.  1.5 Å is deliberately tight: a water that has moved further
#: than that is a different observation, which is what makes "moved" measurable.
DEFAULT_MATCH_RADIUS = 1.5

#: Distance (Å) from a ligand heavy atom to a water site that counts as the ligand
#: occupying (and therefore displacing) that water.
DEFAULT_OCCLUSION_RADIUS = 2.5

#: Fraction of the conformations that must hold a water for the site to count as
#: conserved.
DEFAULT_OCCUPANCY = 0.8

#: Above this displacement (Å) a site that is still occupied is "moved" rather
#: than "conserved".
DEFAULT_MOVE_LIMIT = 1.0


@dataclass
class WaterObservation:
    """One water of one conformation, as it sits in that conformation."""

    conformation: str
    label: str
    chain: str
    res_id: int
    position: Tuple[float, float, float]
    #: Protein residues with an N/O within the H-bond cutoff of the oxygen.
    contacts: List[str] = field(default_factory=list)
    #: Waters within the H-bond cutoff.
    neighbours: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conformation": self.conformation,
            "label": self.label,
            "position": [float(value) for value in self.position],
            "contacts": list(self.contacts),
            "neighbours": int(self.neighbours),
        }


@dataclass
class WaterSite:
    """One water position followed across the conformations of an ensemble."""

    index: int
    center: Tuple[float, float, float]
    observations: Dict[str, WaterObservation] = field(default_factory=dict)
    #: Conformations where the site is empty and a ligand atom occupies it.
    displaced_by: Dict[str, List[str]] = field(default_factory=dict)
    #: Protein contacts of the water where it is present.
    contacts: List[str] = field(default_factory=list)

    @property
    def n_conformations(self) -> int:
        return len(self.observations)

    @property
    def n_present(self) -> int:
        return sum(1 for value in self.observations.values() if value is not None)

    @property
    def occupancy(self) -> float:
        return self.n_present / self.n_conformations if self.n_conformations else 0.0

    @property
    def displacement(self) -> float:
        """Largest distance between the matched positions, in Å."""
        points = [
            value.position for value in self.observations.values() if value is not None
        ]
        if len(points) < 2:
            return 0.0
        return max(
            float(np.linalg.norm(np.asarray(points[a]) - np.asarray(points[b])))
            for a in range(len(points))
            for b in range(a + 1, len(points))
        )

    @property
    def classification(self) -> str:
        """``conserved``, ``moved``, ``displaced`` or ``absent``.

        The criterion, applied in this order:

        * **conserved** — present in at least ``occupancy`` (0.8) of the
          conformations and never more than ``move_limit`` (1.0 Å) from its centre;
        * **displaced** — a non-water residue occupies the site in a conformation
          where the water is *not* there.  This is checked before "moved", because
          with two conformations a water present in one of them has an occupancy of
          exactly 0.5 and would otherwise be called moved -- and a ligand standing
          in its place is the strongest evidence there is;
        * **moved** — present in at least ``occupancy`` of the conformations but
          travelling further than the move limit: the water follows the structure;
        * **transient** — present in some conformations and not others, with no
          ligand to blame;
        * **absent** — seen in fewer than a fifth of them (a one-off water).
        """
        if self.occupancy >= DEFAULT_OCCUPANCY and self.displacement <= DEFAULT_MOVE_LIMIT:
            return "conserved"
        if self.displaced_by:
            return "displaced"
        if self.occupancy >= DEFAULT_OCCUPANCY:
            return "moved"
        if self.occupancy >= 0.2:
            return "transient"
        return "absent"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "site": int(self.index),
            "center": [float(value) for value in self.center],
            "occupancy": self.occupancy,
            "n_present": int(self.n_present),
            "n_conformations": int(self.n_conformations),
            "displacement": self.displacement,
            "classification": self.classification,
            "contacts": list(self.contacts),
            "displaced_by": {key: list(value) for key, value in self.displaced_by.items()},
            "conformations_present": [
                label for label, value in self.observations.items() if value is not None
            ],
        }


@dataclass
class WaterNetwork:
    """The water graph of one conformation."""

    conformation: str
    waters: List[WaterObservation] = field(default_factory=list)
    edges: List[Tuple[int, int, float]] = field(default_factory=list)
    components: List[List[int]] = field(default_factory=list)
    protein_contacts: int = 0

    @property
    def n_waters(self) -> int:
        return len(self.waters)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def largest_component(self) -> int:
        return max((len(component) for component in self.components), default=0)

    def bridges(self) -> int:
        """Waters with at least two neighbours: the network's branching points."""
        degree: Dict[int, int] = {}
        for first, second, _distance in self.edges:
            degree[first] = degree.get(first, 0) + 1
            degree[second] = degree.get(second, 0) + 1
        return sum(1 for value in degree.values() if value >= 2)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "conformation": self.conformation,
            "n_waters": self.n_waters,
            "n_edges": self.n_edges,
            "n_components": self.n_components,
            "largest_component": self.largest_component,
            "bridges": self.bridges(),
            "protein_contacts": int(self.protein_contacts),
            "waters": [water.as_dict() for water in self.waters],
        }

    def table(self, limit: int = 20) -> str:
        headers = ["water", "contacts", "neighbours", "chain"]
        rows = []
        for water in self.waters[: max(1, limit)]:
            rows.append([
                water.label,
                ",".join(water.contacts[:3]) or "-",
                str(water.neighbours),
                water.chain or "-",
            ])
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * width for width in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)


def _water_atoms(conformation: Conformation) -> "OrderedDict[Tuple[str, int, str], List[Any]]":
    """The waters of a conformation, grouped by residue key, in file order."""
    grouped: "OrderedDict[Tuple[str, int, str], List[Any]]" = OrderedDict()
    for atom in conformation.atoms:
        if str(atom.res_name).strip().upper() in WATER_NAMES:
            grouped.setdefault(atom.residue_key, []).append(atom)
    return grouped


def _heavy_protein_atoms(conformation: Conformation) -> List[Any]:
    return [
        atom
        for atom in conformation.atoms
        if not atom.is_hydrogen and str(atom.res_name).strip().upper() not in WATER_NAMES
    ]


def _hbond_acceptors(atoms: Sequence[Any]) -> List[Any]:
    return [atom for atom in atoms if str(atom.element).strip().upper() in ("N", "O")]


def water_network(
    conformation: Conformation,
    *,
    hbond_cutoff: float = DEFAULT_HBOND_CUTOFF,
) -> WaterNetwork:
    """Build the water–water and water–protein hydrogen-bond graph.

    Waters with no oxygen (a hydrogen-only water is not a thing, but a truncated
    file happens) are skipped, and every distance is O···O or O···N/O, which is
    all a crystal structure can support: crystallographic waters carry no
    hydrogens, so "hydrogen bond" here means "the heavy atoms are within the
    distance at which one is possible".
    """
    grouped = _water_atoms(conformation)
    waters: List[WaterObservation] = []
    positions: List[np.ndarray] = []
    protein = _hbond_acceptors(_heavy_protein_atoms(conformation))
    protein_points = np.array(
        [[atom.x, atom.y, atom.z] for atom in protein], dtype=float
    ).reshape(-1, 3)
    protein_labels = [
        f"{atom.res_name}{atom.res_id} {atom.chain}".strip() for atom in protein
    ]
    for (chain, res_id, res_name), atoms in grouped.items():
        oxygen = next(
            (atom for atom in atoms if str(atom.element).strip().upper() == "O"), None
        )
        if oxygen is None:
            continue
        point = np.array([oxygen.x, oxygen.y, oxygen.z], dtype=float)
        positions.append(point)
        contacts: List[str] = []
        if protein_points.size:
            distance = np.sqrt(((protein_points - point) ** 2).sum(axis=1))
            for index in np.flatnonzero(distance <= float(hbond_cutoff)):
                label = protein_labels[int(index)]
                if label not in contacts:
                    contacts.append(label)
        waters.append(
            WaterObservation(
                conformation=conformation.label,
                label=f"{res_name}{res_id} {chain}".strip(),
                chain=chain,
                res_id=int(res_id),
                position=(float(point[0]), float(point[1]), float(point[2])),
                contacts=contacts,
            )
        )
    coords = np.asarray(positions, dtype=float).reshape(-1, 3)
    edges: List[Tuple[int, int, float]] = []
    parent = list(range(len(waters)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(waters)):
        for j in range(i + 1, len(waters)):
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if distance <= float(hbond_cutoff):
                edges.append((i, j, distance))
                waters[i].neighbours += 1
                waters[j].neighbours += 1
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[max(root_i, root_j)] = min(root_i, root_j)
    groups: Dict[int, List[int]] = {}
    for index in range(len(waters)):
        groups.setdefault(find(index), []).append(index)
    components = sorted(groups.values(), key=lambda item: (-len(item), item[0]))
    return WaterNetwork(
        conformation=conformation.label,
        waters=waters,
        edges=edges,
        components=components,
        protein_contacts=sum(len(water.contacts) for water in waters),
    )


def water_sites(
    conformations: Sequence[Conformation],
    *,
    match_radius: float = DEFAULT_MATCH_RADIUS,
    hbond_cutoff: float = DEFAULT_HBOND_CUTOFF,
    occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS,
    ligands: Optional[Dict[str, Sequence[Tuple[str, Tuple[float, float, float]]]]] = None,
) -> List[WaterSite]:
    """Pool every water of every conformation into sites and classify them.

    A site is a leader cluster of water oxygens within `match_radius` (1.5 Å) of
    each other, seeded in conformation order so the result is deterministic.  A
    conformation is "present" at a site when one of its waters is in the cluster,
    which is what makes occupancy a statement about the ensemble rather than about
    one file.

    `ligands`
        Optional ``{conformation: [(atom label, position), ...]}`` — the ligand
        atoms of each conformation.  When given, a site that is empty in a
        conformation and has a ligand atom within `occlusion_radius` records the
        atoms that occupy it, which is the evidence for the ``displaced`` verdict.
    """
    if not conformations:
        return []
    hints = ligands or {}
    sites: List[WaterSite] = []
    centres: List[np.ndarray] = []
    for conformation in conformations:
        grouped = _water_atoms(conformation)
        protein = _hbond_acceptors(_heavy_protein_atoms(conformation))
        protein_points = np.array(
            [[atom.x, atom.y, atom.z] for atom in protein], dtype=float
        ).reshape(-1, 3)
        protein_labels = [
            f"{atom.res_name}{atom.res_id} {atom.chain}".strip() for atom in protein
        ]
        oxygens = []
        for (chain, res_id, res_name), atoms in grouped.items():
            oxygen = next(
                (atom for atom in atoms if str(atom.element).strip().upper() == "O"), None
            )
            if oxygen is not None:
                oxygens.append((f"{res_name}{res_id} {chain}".strip(), chain, res_id, oxygen))
        for label, chain, res_id, oxygen in oxygens:
            point = np.array([oxygen.x, oxygen.y, oxygen.z], dtype=float)
            site_index = None
            if centres:
                distances = [
                    float(np.linalg.norm(point - centre)) for centre in centres
                ]
                best = int(np.argmin(distances))
                if distances[best] <= float(match_radius):
                    site_index = best
            if site_index is None:
                sites.append(WaterSite(index=len(sites), center=(float(point[0]), float(point[1]), float(point[2]))))
                centres.append(point)
                site_index = len(sites) - 1
            contacts: List[str] = []
            if protein_points.size:
                distance = np.sqrt(((protein_points - point) ** 2).sum(axis=1))
                for index in np.flatnonzero(distance <= float(hbond_cutoff)):
                    name = protein_labels[int(index)]
                    if name not in contacts:
                        contacts.append(name)
            sites[site_index].observations[conformation.label] = WaterObservation(
                conformation=conformation.label,
                label=label,
                chain=chain,
                res_id=int(res_id),
                position=(float(point[0]), float(point[1]), float(point[2])),
                contacts=contacts,
            )
            for name in contacts:
                if name not in sites[site_index].contacts:
                    sites[site_index].contacts.append(name)
    # Every site must know every conformation, so occupancy has a denominator.
    labels = [conformation.label for conformation in conformations]
    for site in sites:
        for label in labels:
            site.observations.setdefault(label, None)  # type: ignore[arg-type]
    # Ligand occlusion: an empty site with a ligand atom on top of it.
    for site in sites:
        for label in labels:
            if site.observations.get(label) is not None:
                continue
            for atom_label, position in hints.get(label, ()):  # type: ignore[union-attr]
                if float(np.linalg.norm(np.asarray(position) - np.asarray(site.center))) <= float(occlusion_radius):
                    site.displaced_by.setdefault(label, []).append(atom_label)
    return sites


def compare_water_sites(
    conformations: Sequence[Conformation],
    *,
    match_radius: float = DEFAULT_MATCH_RADIUS,
    occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS,
) -> "WaterAnalysis":
    """The water networks and the classified sites of an ensemble."""
    networks = [water_network(conformation) for conformation in conformations]
    hints: Dict[str, List[Tuple[str, Tuple[float, float, float]]]] = {}
    for conformation in conformations:
        atoms: List[Tuple[str, Tuple[float, float, float]]] = []
        for residues in conformation.chain_residues().values():
            for residue in residues:
                if residue.is_standard or residue.is_water:
                    continue
                # Every non-protein residue is a candidate displacer, ions and
                # buffer components included: they occupy space the same way a
                # ligand does, and the report names what each one is so a reader
                # can tell a drug from a sulfate.
                for atom in residue.atoms:
                    if atom.is_hydrogen:
                        continue
                    atoms.append(
                        (
                            f"{residue.label}:{atom.name}",
                            (float(atom.x), float(atom.y), float(atom.z)),
                        )
                    )
        hints[conformation.label] = atoms
    sites = water_sites(
        conformations, match_radius=match_radius, occlusion_radius=occlusion_radius,
        ligands=hints,
    )
    return WaterAnalysis(
        labels=[conformation.label for conformation in conformations],
        networks=networks,
        sites=sites,
        match_radius=float(match_radius),
        occlusion_radius=float(occlusion_radius),
    )


@dataclass
class WaterAnalysis:
    """The water picture of an ensemble: networks plus classified sites."""

    labels: List[str]
    networks: List[WaterNetwork] = field(default_factory=list)
    sites: List[WaterSite] = field(default_factory=list)
    match_radius: float = DEFAULT_MATCH_RADIUS
    occlusion_radius: float = DEFAULT_OCCLUSION_RADIUS

    def by_class(self, name: str) -> List[WaterSite]:
        return [site for site in self.sites if site.classification == name]

    @property
    def conserved(self) -> List[WaterSite]:
        return self.by_class("conserved")

    @property
    def displaced(self) -> List[WaterSite]:
        return self.by_class("displaced")

    def summary(self) -> Dict[str, Any]:
        counts = {
            name: len(self.by_class(name))
            for name in ("conserved", "moved", "displaced", "transient", "absent")
        }
        return {
            "conformations": list(self.labels),
            "waters_per_conformation": {
                network.conformation: network.n_waters for network in self.networks
            },
            "edges_per_conformation": {
                network.conformation: network.n_edges for network in self.networks
            },
            "components_per_conformation": {
                network.conformation: network.n_components for network in self.networks
            },
            "largest_component": {
                network.conformation: network.largest_component for network in self.networks
            },
            "sites": len(self.sites),
            "counts": counts,
        }

    def table(self, limit: int = 20) -> str:
        headers = ["site", "class", "occupancy", "displacement (A)", "contacts", "displaced by"]
        rows = []
        order = {"conserved": 0, "moved": 1, "displaced": 2, "transient": 3, "absent": 4}
        ranked = sorted(
            self.sites, key=lambda site: (order.get(site.classification, 9), -site.occupancy, site.index)
        )
        for site in ranked[: max(1, limit)]:
            displaced = "; ".join(
                f"{label}:{','.join(atoms[:2])}" for label, atoms in site.displaced_by.items()
            )
            rows.append([
                str(site.index + 1),
                site.classification,
                f"{site.occupancy:.2f}",
                f"{site.displacement:.3f}",
                ", ".join(site.contacts[:3]) or "-",
                displaced or "-",
            ])
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines = [render(headers), "  ".join("-" * width for width in widths)]
        lines.extend(render(row) for row in rows)
        return "\n".join(lines)

    def text(self, *, limit: int = 20) -> str:
        summary = self.summary()
        lines = [
            f"OpenDocking water analysis — {len(self.labels)} conformation(s): "
            f"{', '.join(self.labels)}",
            "",
            f"water network: "
            + "; ".join(
                f"{network.conformation}: {network.n_waters} water(s), "
                f"{network.n_edges} H-bond(s), {network.n_components} component(s), "
                f"largest {network.largest_component}, {network.bridges()} branching"
                for network in self.networks
            ),
            "",
            f"sites across the ensemble ({len(self.sites)}): "
            + ", ".join(f"{name} {count}" for name, count in summary["counts"].items()),
            "",
            self.table(limit),
        ]
        displaced = self.displaced
        if displaced:
            lines += ["", "waters displaced by the ligand:"]
            for site in displaced:
                for label, atoms in site.displaced_by.items():
                    lines.append(
                        f"  site {site.index + 1} at ({site.center[0]:.2f}, "
                        f"{site.center[1]:.2f}, {site.center[2]:.2f}): empty in "
                        f"{label}, occupied by {', '.join(atoms[:4])}"
                        + (
                            f"; the water bonds to {', '.join(site.contacts[:3])}"
                            if site.contacts
                            else ""
                        )
                    )
        lines += [
            "",
            "criteria: a site is CONSERVED when it is present in at least "
            f"{DEFAULT_OCCUPANCY:.0%} of the conformations and never more than "
            f"{DEFAULT_MOVE_LIMIT:.1f} A from its centre; MOVED when it is present "
            f"in at least {DEFAULT_OCCUPANCY:.0%} but travels further; DISPLACED "
            "when a non-water residue occupies the empty site within "
            f"{self.occlusion_radius:.1f} A; TRANSIENT when it is present in some "
            "conformations and not others with no such residue; ABSENT below a "
            "fifth. Matching uses a "
            f"{self.match_radius:.1f} A radius, and 'hydrogen bond' means the heavy "
            "atoms are within the cutoff, because crystallographic waters carry no "
            "hydrogens.",
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "summary": self.summary(),
            "networks": [network.as_dict() for network in self.networks],
            "sites": [site.as_dict() for site in self.sites],
            "criteria": {
                "match_radius": self.match_radius,
                "occlusion_radius": self.occlusion_radius,
                "occupancy": DEFAULT_OCCUPANCY,
                "move_limit": DEFAULT_MOVE_LIMIT,
            },
        }


# ---------------------------------------------------------------------------
# Poses against conserved waters
# ---------------------------------------------------------------------------


def pose_displacement(
    poses: Sequence[Any],
    sites: Sequence[WaterSite],
    *,
    radius: float = 3.5,
    classes: Sequence[str] = ("conserved",),
) -> List[Dict[str, Any]]:
    """How many of the selected water sites each pose would displace.

    A site counts when a ligand heavy atom comes within `radius` of the water's
    centre: the pose occupies the space the water needs, so one of the two has to
    go.  `radius` defaults to 3.5 Å — the water's own first coordination shell —
    rather than a clash distance, because the question is "does the pose want that
    space", not "is it overlapping".

    `poses` accepts anything with ``coords``/``affinity`` — an
    :class:`odock.docking.Pose` or a dict from :func:`odock.cli._read_poses`.
    """
    wanted = [
        site
        for site in sites
        if site.classification in set(classes) and site.occupancy > 0.0
    ]
    out: List[Dict[str, Any]] = []
    for pose in poses:
        if isinstance(pose, dict):
            coords = pose.get("coords")
            affinity = pose.get("affinity", float("nan"))
            conformation = pose.get("conformation", "")
        else:
            coords = getattr(pose, "coords", None)
            affinity = getattr(pose, "affinity", float("nan"))
            conformation = getattr(pose, "conformation", "")
        if coords is None:
            continue
        points = np.asarray(coords, dtype=float).reshape(-1, 3)
        hits: List[int] = []
        for site in wanted:
            distance = np.sqrt(((points - np.asarray(site.center)) ** 2).sum(axis=1))
            if float(distance.min()) <= float(radius):
                hits.append(site.index)
        out.append(
            {
                "displaced": len(hits),
                "sites": hits,
                "affinity": float(affinity),
                "conformation": str(conformation),
            }
        )
    return out


def correlate(
    scores: Sequence[float],
    affinities: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 20240101,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Spearman rho between two lists, with n and a bootstrap confidence interval.

    The project's power discipline: a rho without its n and an interval is not a
    result.  The interval is a percentile bootstrap over the pairs, seeded so it
    is reproducible, and when fewer than five pairs are usable the function says
    so instead of returning a number that cannot be read.
    """
    from . import consensus

    first = np.asarray(list(scores), dtype=float)
    second = np.asarray(list(affinities), dtype=float)
    usable = np.isfinite(first) & np.isfinite(second)
    n = int(usable.sum())
    rho = consensus.spearman(first[usable], second[usable]) if n >= 2 else float("nan")
    if n < 5:
        return {
            "n": n,
            "rho": rho,
            "low": float("nan"),
            "high": float("nan"),
            "samples": 0,
            "note": (
                f"not resolvable: {n} usable pair(s); a rank correlation below "
                "five points is not a measurement, however large it looks"
            ),
        }
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(samples)):
        pick = rng.integers(0, n, n)
        value = consensus.spearman(first[usable][pick], second[usable][pick])
        if math.isfinite(value):
            values.append(value)
    if not values:
        low = high = float("nan")
    else:
        alpha = (1.0 - float(confidence)) / 2.0
        low = float(np.quantile(values, alpha))
        high = float(np.quantile(values, 1.0 - alpha))
    return {
        "n": n,
        "rho": rho,
        "low": low,
        "high": high,
        "samples": len(values),
        "note": "",
    }


# ---------------------------------------------------------------------------
# The command line: `odock ensemble waters`
# ---------------------------------------------------------------------------


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def cmd_ensemble_waters(args) -> int:
    """``odock ensemble waters``: the water network and the conserved sites."""
    from .ensemble import (
        _add_ensemble_arguments,  # noqa: F401  (documentation of the shared flags)
        _cli_write_json,
        _ensemble_box,
        _ensemble_inputs,
        align_conformations,
        ligand_coords,
    )

    paths = _ensemble_inputs(args)
    box = None
    if getattr(args, "box", None) or (args.center and args.size) or args.box_ligand:
        box = _ensemble_box(args, paths)
    try:
        # Waters are the subject here, so they must survive the reader.
        conformations = read_conformations(paths, keep_water=True)
        site_ligand = None
        if args.site_ligand:
            site_ligand = ligand_coords(
                conformations[int(args.reference)], args.site_ligand
            )
        if args.superpose and (box is not None or args.site or site_ligand is not None):
            align_conformations(
                conformations, reference=int(args.reference), box=box,
                site=args.site, site_ligand=site_ligand,
                site_radius=float(args.site_radius), superpose=True,
                min_identity=float(args.min_identity),
            )
        elif args.superpose:
            _cli_eprint(
                "note: --superpose needs a site to fit on; pass --box, --site or "
                "--site-ligand, or use --no-superpose if the structures already "
                "share a frame (water matching across frames is meaningless)"
            )
        analysis = compare_water_sites(
            conformations,
            match_radius=float(args.match_radius),
            occlusion_radius=float(args.occlusion_radius),
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble waters: error: {exc}")
        return int(exc.code)

    if not args.quiet:
        print(analysis.text(limit=int(args.top)))
        print()
    if args.outdir:
        target = Path(args.outdir)
        target.mkdir(parents=True, exist_ok=True)
        for network in analysis.networks:
            path = target / f"{network.conformation}_waters.pdb"
            lines = ["REMARK  ODOCK ENSEMBLE WATER SITES"]
            serial = 0
            for site in analysis.sites:
                observation = site.observations.get(network.conformation)
                if observation is None:
                    continue
                serial += 1
                x, y, z = observation.position
                lines.append(
                    f"HETATM{serial:5d}  O   HOH A{site.index % 9999 + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O"
                )
            lines.append("END")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            _cli_eprint(f"wrote {path}")
    if args.json_out:
        _cli_write_json(args.json_out, analysis.as_dict())
    return 0


def add_waters_subparser(ensub: Any) -> None:
    """Register ``odock ensemble waters`` on the ensemble subparsers."""
    from .ensemble import _add_ensemble_arguments

    parser = ensub.add_parser(
        "waters",
        help="the water network, conserved waters, and the ones a ligand displaces",
        description=(
            "Build the water-water and water-protein hydrogen-bond network of every "
            "conformation, pool the waters into sites, and classify each site as "
            "conserved, moved or displaced by a ligand -- with the ligand atoms "
            "that occupy a displaced site and the protein contacts the water was "
            "making.  Waters are read with the reader's water switch on, and the "
            "conformations are superposed first, because matching water positions "
            "across frames is meaningless."
        ),
    )
    _add_ensemble_arguments(parser, allow_box_mismatch=True)
    parser.add_argument("--box", help="box JSON written by `odock box`")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--box-ligand", metavar="RESNAME", help="take the site from this residue")
    parser.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    parser.add_argument("--spacing", type=float, default=0.375, help="box grid spacing (Å)")
    parser.add_argument(
        "--match-radius", type=float, default=DEFAULT_MATCH_RADIUS,
        help="distance within which two waters are the same site (Å)",
    )
    parser.add_argument(
        "--occlusion-radius", type=float, default=DEFAULT_OCCLUSION_RADIUS,
        help="how close a ligand heavy atom must come to an empty site to count as "
             "displacing it (Å)",
    )
    parser.add_argument(
        "--hbond", type=float, default=DEFAULT_HBOND_CUTOFF,
        help="O···O / O···N distance that counts as a hydrogen bond (Å)",
    )
    parser.add_argument(
        "-o", "--outdir",
        help="write each conformation's water sites as a PDB of HETATM oxygens",
    )
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    parser.add_argument("--json-out", help="write the whole analysis as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no report")
    parser.set_defaults(func=cmd_ensemble_waters)

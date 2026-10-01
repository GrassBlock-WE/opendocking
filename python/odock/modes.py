# SPDX-License-Identifier: GPL-3.0-or-later
"""Backbone motion from a single structure, by an elastic network model.

The side-chain generator in :mod:`odock.generate` proved its own limit with a
measurement: moving side chains by up to 2.9 Å produced **no** cryptic-pocket
closure (closure fractions 1.00/1.07), while the experimental ERα pair closes
three cavities with closure fractions of 0.01/0.06/0.11 — because closing them
needs the **backbone** (helix 12, 2.010 Å).  This module supplies that motion from
one structure: an anisotropic network model (ANM) over the Cα atoms, whose
low-frequency modes are the collective motions a protein of that connectivity can
make, sampled at a **calibrated amplitude**.

What it is, stated up front and repeated in ``docs/GENERATED_ENSEMBLES.md``:

* a **harmonic approximation** — normal modes of one structure's contact graph. It
  cannot reproduce a helix unravelling, a loop reorganising, or any anharmonic
  transition; it assumes the connectivity of the input structure and a harmonic
  energy surface around it;
* a displacement along a mode is **not a physically sampled state**: the modes are
  directions, and the amplitude is a choice. The default here is to *calibrate*
  that choice against the experimental pairs this project measured
  (:data:`odock.generate.EXPERIMENTAL_REFERENCES`), so the generated ensemble
  spans the experimental site RMSDs rather than an arbitrary number;
* the Cα displacement field is transferred to **all atoms of each residue
  rigidly**, the standard coarse-graining. Bond lengths inside a residue are
  exact; the peptide geometry between residues is not, and the report says so.

The output is the same shape as :mod:`odock.generate`'s: ordinary PDB members in
one frame, ready for ``odock ensemble dock/screen/pockets --no-superpose``.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .ensemble import Conformation, EnsembleError, read_conformations
from .generate import (
    EXPERIMENTAL_REFERENCES,
    GeneratedEnsemble,
    RotamerResidue,
    _radii,
)
from .pocket import VDW_RADII
from .prepare import BoxSpec

__all__ = [
    "DEFAULT_CUTOFF",
    "DEFAULT_MODES",
    "DEFAULT_TARGET_RMSD",
    "ElasticNetwork",
    "GeneratedModes",
    "add_modes_subparser",
    "build_network",
    "direction_overlap",
    "generate_modes",
    "mode_amplitudes",
]

#: Cα–Cα distance (Å) below which two residues are connected by a spring.  13 Å is
#: the standard ANM cutoff (Atilgan 2001; used by ANM web servers): it reproduces
#: the experimental B-factors and the low-frequency motions of a single chain
#: without making the network a rigid block, and it is what was measured here.
DEFAULT_CUTOFF = 13.0

#: Low-frequency modes used by default.  The first few dominate the
#: experimentally observed motion; beyond ~10 the modes are increasingly local.
DEFAULT_MODES = 3

#: Site RMSD targets the amplitude is calibrated to, in Å.  Default: the site
#: RMSDs of the three experimental pairs this project validated against (ERα,
#: HIV-1 protease, trypsin), so the generated ensemble *brackets* experiment.
DEFAULT_TARGET_RMSD: Tuple[float, ...] = tuple(
    float(reference["site_rmsd_ca"]) for reference in EXPERIMENTAL_REFERENCES
)

#: Heavy-atom pair distance below this fraction of the summed vdW radii counts as
#: a *severe* overlap, and is what the filter rejects.
CLASH_FACTOR = 0.6

#: The milder threshold (0.75) counts contacts the modelled displacement has
#: squeezed; at a 0.2–0.4 Å amplitude the rigid-per-residue transfer produces
#: dozens of them, so they are reported but do not reject a member.  Rejecting on
#: this number would reject every member of every ensemble.
SHEAR_FACTOR = 0.75


@dataclass
class ElasticNetwork:
    """An anisotropic network model over the Cα atoms of one conformation."""

    labels: List[str]
    coords: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 3)))
    #: Index of each node's Cα in ``conformation.atoms``.
    atom_index: List[int] = field(default_factory=list)
    #: ``(chain, res_id, res_name)`` of each node, in node order.
    keys: List[Tuple[str, int, str]] = field(default_factory=list)
    cutoff: float = DEFAULT_CUTOFF
    #: Number of springs (node pairs within the cutoff).
    springs: int = 0
    #: Connected components of the contact graph.  More than one means the modes
    #: are per-fragment motions of a split network, which the report warns about.
    components: int = 1
    eigenvalues: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    eigenvectors: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0)))
    selected: int = DEFAULT_MODES
    residue_index: List[int] = field(default_factory=list)

    @property
    def n_nodes(self) -> int:
        return len(self.labels)

    @property
    def n_modes(self) -> int:
        """Non-zero modes: ``3N - 6`` for a connected network."""
        return max(0, int(self.eigenvectors.shape[1]))

    @property
    def elastic_energy(self) -> float:
        """Mean-square fluctuation implied by the modes, from the eigenvalues."""
        values = [
            float(self.eigenvalues[index])
            for index in range(self.selected)
            if index < self.eigenvalues.size and self.eigenvalues[index] > 1e-9
        ]
        return float(sum(1.0 / value for value in values))

    def mode(self, index: int) -> np.ndarray:
        """Mode `index` as an ``(N, 3)`` displacement field, normalised to unit RMSD."""
        vector = np.asarray(self.eigenvectors[:, index], dtype=float).reshape(self.n_nodes, 3)
        scale = float(np.sqrt((vector ** 2).sum(axis=1).mean()))
        return vector / scale if scale > 0 else vector

    def amplitudes(self, *, temperature: float = 1.0) -> np.ndarray:
        """Equipartition amplitudes ``sqrt(T / lambda_k)`` for the selected modes.

        This is the standard result for a harmonic network: a softer mode
        (smaller eigenvalue) is excited more.  The overall scale is arbitrary, so
        :func:`generate_modes` rescales the whole field to a *measured* target
        rather than trusting the units.
        """
        values = []
        for index in range(self.selected):
            if index >= self.eigenvalues.size:
                break
            eigenvalue = float(self.eigenvalues[index])
            values.append(
                math.sqrt(float(temperature) / eigenvalue) if eigenvalue > 1e-9 else 0.0
            )
        return np.asarray(values, dtype=float)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.n_nodes,
            "cutoff": float(self.cutoff),
            "springs": int(self.springs),
            "n_modes": int(self.n_modes),
            "selected": int(self.selected),
            "eigenvalues": [
                float(value) for value in self.eigenvalues[: self.selected + 6]
            ],
            "amplitudes": [float(value) for value in self.amplitudes()],
        }


def build_network(
    conformation: Conformation,
    *,
    cutoff: float = DEFAULT_CUTOFF,
    modes: int = DEFAULT_MODES,
    residues: Optional[Sequence[Any]] = None,
) -> ElasticNetwork:
    """Build the ANM Hessian of `conformation` and diagonalise it.

    `residues` restricts the nodes (the default is every standard residue of the
    chain, which is what a collective motion needs: a helix swings *relative* to
    the rest of the domain, so a site-only network cannot produce it).  The
    Hessian is the standard ANM form, one 3×3 block per residue pair within
    `cutoff`:

    ``H_ij = -(gamma / r^2) * r r^T`` for ``i != j``, and the diagonal is minus
    the sum of a row's off-diagonal blocks, so the six rigid-body modes have zero
    eigenvalue.
    """
    if cutoff <= 0:
        raise EnsembleError(f"the ANM cutoff must be positive, got {cutoff}")
    if residues is None:
        residues = [
            residue
            for chain in conformation.chain_residues().values()
            for residue in chain
            if residue.is_standard
        ]
    labels: List[str] = []
    atom_index: List[int] = []
    keys: List[Tuple[str, int, str]] = []
    positions: List[Tuple[float, float, float]] = []
    position_of = {id(atom): index for index, atom in enumerate(conformation.atoms)}
    for residue in residues:
        ca = None
        for atom in residue.atoms:
            if atom.name == "CA" and str(atom.element).strip().upper() == "C":
                ca = atom
                break
        if ca is None:
            continue  # a Cα is required: the network's node is the Cα
        labels.append(residue.label)
        atom_index.append(position_of[id(ca)])
        keys.append(residue.key)
        positions.append((float(ca.x), float(ca.y), float(ca.z)))
    if len(labels) < 3:
        raise EnsembleError(
            f"an elastic network needs at least three Cα atoms, got {len(labels)}"
        )
    coordinates = np.asarray(positions, dtype=float)
    n = coordinates.shape[0]
    hessian = np.zeros((3 * n, 3 * n), dtype=float)
    springs = 0
    parent = list(range(n))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(n):
        for j in range(i + 1, n):
            delta = coordinates[i] - coordinates[j]
            distance = float(np.linalg.norm(delta))
            if distance <= 0.0 or distance > float(cutoff):
                continue
            springs += 1
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)
            block = np.outer(delta, delta) / (distance ** 2)
            hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = -block
            hessian[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = -block
            hessian[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += block
            hessian[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += block
    components = len({find(node) for node in range(n)})
    if springs == 0:
        raise EnsembleError(
            f"an ANM cutoff of {float(cutoff):g} A connects no Cα pair in this "
            "structure (the shortest Cα-Cα distance is about 3.8 A); the usual "
            "cutoff is 13 A"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    # Drop the six rigid-body modes (translation and rotation) and any numerically
    # zero eigenvalue that follows them.
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    sign = 6
    while sign < eigenvalues.size and eigenvalues[sign] <= 1e-8:
        sign += 1
    eigenvalues = eigenvalues[sign:]
    eigenvectors = eigenvectors[:, sign:]
    return ElasticNetwork(
        labels=labels,
        coords=coordinates,
        atom_index=atom_index,
        keys=keys,
        cutoff=float(cutoff),
        springs=springs,
        components=components,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        selected=min(int(modes), int(eigenvectors.shape[1])),
        residue_index=list(range(len(labels))),
    )


def mode_amplitudes(network: ElasticNetwork, *, temperature: float = 1.0) -> np.ndarray:
    """The equipartition amplitudes of the selected modes (see :meth:`amplitudes`)."""
    return network.amplitudes(temperature=temperature)


def experimental_field(
    network: ElasticNetwork,
    native: Conformation,
    other: Conformation,
    alignment: Any,
) -> Tuple[np.ndarray, int]:
    """The observed Cα displacement of `other` relative to `native`, per node.

    `alignment` is the :class:`odock.ensemble.Alignment` of `other` onto `native`,
    so the correspondence is the sequence alignment's, not the atom order: two
    crystal structures of one protein routinely have different numbers of atoms.
    Nodes with no counterpart in `other` get a zero displacement and are counted,
    so a partial overlap is reported rather than silently shrinking the field.
    """
    lookup: Dict[Tuple[str, int, str], Any] = {}
    for chain_alignment in alignment.chain_alignments:
        reference_residues = native.standard_residues(chain_alignment.chain)
        other_residues = other.standard_residues(chain_alignment.other_chain)
        for i, j in chain_alignment.pairs():
            if i < len(reference_residues) and j < len(other_residues):
                lookup[reference_residues[i].key] = other_residues[j]

    def ca_of(residue: Any) -> Optional[np.ndarray]:
        for atom in residue.atoms:
            if atom.name == "CA" and str(atom.element).strip().upper() == "C":
                return np.array([float(atom.x), float(atom.y), float(atom.z)])
        return None

    field = np.zeros((network.n_nodes, 3), dtype=float)
    unmatched = 0
    for node, key in enumerate(network.keys):
        partner = lookup.get(key)
        if partner is None:
            unmatched += 1
            continue
        native_ca = network.coords[node]
        other_ca = ca_of(partner)
        if other_ca is None:
            unmatched += 1
            continue
        field[node] = other_ca - native_ca
    return field, unmatched


def direction_overlap(
    network: ElasticNetwork,
    displacement: np.ndarray,
    *,
    index: int = 0,
    nodes: Optional[Sequence[int]] = None,
) -> float:
    """Cosine similarity between mode `index` and a displacement field.

    Both are ``(N, 3)`` Cα fields on the same nodes, so this is the fraction of
    the mode's direction that points along the observed motion: ``1.0`` is a
    perfect match, ``0`` is orthogonal, negative is opposite.  A magnitude match
    with an overlap near zero is a *wrong-direction* match, and the report prints
    the number so that cannot pass unnoticed.

    `nodes` restricts the comparison to a subset (the binding-site residues, say).
    That matters: an unfitted crystal structure's flexible termini can carry a
    larger displacement than the site itself, and a whole-chain cosine is then
    dominated by a floppy tail rather than by the motion of interest.
    """
    field = np.asarray(displacement, dtype=float).reshape(network.n_nodes, 3)
    mode = network.mode(index)
    if nodes is not None:
        selection = np.asarray(list(nodes), dtype=int)
        if selection.size == 0:
            return float("nan")
        field = field[selection]
        mode = mode[selection]
    denominator = float(np.linalg.norm(field) * np.linalg.norm(mode))
    if denominator <= 0.0:
        return float("nan")
    return float((field * mode).sum() / denominator)


@dataclass
class GeneratedModes:
    """A modelled ensemble built from one structure by elastic-network modes."""

    native: Conformation
    network: ElasticNetwork
    conformations: List[Conformation] = field(default_factory=list)
    spread: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)
    #: ``(pair, overlap)`` for each experimental reference the caller asked to
    #: compare a direction against.
    overlaps: List[Tuple[str, float]] = field(default_factory=list)
    #: The ``(chain, res_id, res_name)`` keys of the site residues the targets and
    #: the site-restricted direction overlap are measured on.
    site_keys: List[Tuple[str, int, str]] = field(default_factory=list)
    source: Optional[Path] = None

    @property
    def n_conformations(self) -> int:
        return len(self.conformations)

    def table(self) -> str:
        headers = [
            "member", "mode", "target RMSD (A)", "site RMSD (A)", "max disp (A)",
            "overlaps", "squeezed", "kept",
        ]
        rows = []
        for entry in self.spread:
            rows.append([
                entry["member"],
                str(entry.get("mode", "-")),
                f"{entry.get('target', 0.0):.3f}" if entry.get("target") else "-",
                f"{entry['site_rmsd']:.3f}",
                f"{entry['max_displacement']:.3f}",
                str(entry.get("clashes", 0)),
                str(entry.get("squeezed", 0)),
                "yes" if entry.get("kept") else "no",
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

    def comparison(self) -> List[Dict[str, Any]]:
        """The generated spread against the experimental pairs (computed)."""
        kept = [entry for entry in self.spread if entry.get("kept")]
        generated_site = [entry["site_rmsd"] for entry in kept[1:]] or [0.0]
        generated_max = max(
            (entry["max_displacement"] for entry in kept), default=0.0
        )
        out = []
        for reference in EXPERIMENTAL_REFERENCES:
            out.append(
                {
                    "pair": reference["pair"],
                    "experimental_site_rmsd_ca": reference["site_rmsd_ca"],
                    "experimental_max_displacement": reference["max_displacement"],
                    "generated_site_rmsd_mean": float(np.mean(generated_site)),
                    "generated_max_displacement": generated_max,
                    "amplitude_ratio": (
                        generated_max / reference["max_displacement"]
                        if reference["max_displacement"]
                        else float("nan")
                    ),
                }
            )
        return out

    def text(self, *, limit: int = 20) -> str:
        network = self.network
        lines = [
            f"OpenDocking elastic-network ensemble — {self.n_conformations} "
            f"conformation(s) from {self.native.label}",
            "",
            f"network: {network.n_nodes} Cα node(s), {network.springs} spring(s) at a "
            f"{network.cutoff:g} A cutoff, {network.n_modes} non-zero mode(s), "
            f"{network.selected} used",
            "parameters: " + ", ".join(f"{k}={v}" for k, v in self.parameters.items()),
            "lowest eigenvalues (arbitrary units): "
            + ", ".join(f"{value:.4g}" for value in network.eigenvalues[: network.selected]),
            "",
            self.table(),
            "",
            "comparison with the experimental pairs this project validates against "
            "(docs/ENSEMBLE.md section 1):",
        ]
        headers = ["pair", "exp site RMSD (A)", "exp max disp (A)", "generated max disp (A)", "ratio"]
        rows = []
        for entry in self.comparison():
            rows.append([
                entry["pair"],
                f"{entry['experimental_site_rmsd_ca']:.3f}",
                f"{entry['experimental_max_displacement']:.3f}",
                f"{entry['generated_max_displacement']:.3f}",
                f"{entry['amplitude_ratio']:.2f}",
            ])
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def render(cells):
            return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

        lines.append(render(headers))
        lines.append("  ".join("-" * width for width in widths))
        lines.extend(render(row) for row in rows)
        if self.overlaps:
            lines += ["", "direction of the motion (cosine overlap with the experimental field):"]
            lines += [
                f"  {pair}: {overlap:+.3f}" for pair, overlap in self.overlaps
            ]
            lines.append(
                "  an overlap near zero means the amplitude matches experiment but "
                "the direction does not: a magnitude match is not a mechanism."
            )
        lines += [
            "",
            "an elastic network is a HARMONIC approximation of one structure's "
            "contact graph: it cannot reproduce a helix unravelling, a loop "
            "reorganising or any anharmonic transition, and a displacement along a "
            "mode is a direction at a chosen amplitude, not a sampled state. The "
            "amplitude here is CALIBRATED to the experimental site RMSDs, so the "
            "spread brackets experiment by construction rather than by prediction.",
            "the Cα displacement is transferred to every atom of its residue "
            "rigidly (standard coarse-graining): intra-residue geometry is exact, "
            "inter-residue geometry is not.",
        ]
        if self.warnings:
            lines += [""] + [f"warning: {warning}" for warning in self.warnings]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": None if self.source is None else str(self.source),
            "native": self.native.label,
            "n_conformations": self.n_conformations,
            "network": self.network.as_dict(),
            "parameters": dict(self.parameters),
            "elapsed": round(float(self.elapsed), 4),
            "spread": list(self.spread),
            "comparison": self.comparison(),
            "overlaps": [
                {"pair": pair, "overlap": float(overlap)} for pair, overlap in self.overlaps
            ],
            "warnings": list(self.warnings),
        }

    def write(self, outdir: Union[str, Path]) -> List[Path]:
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for index, conformation in enumerate(self.conformations):
            name = "native" if index == 0 else f"mode_{index}"
            target = directory / f"{name}.pdb"
            target.write_text(conformation.text(), encoding="utf-8")
            written.append(target)
        return written


def _displace(
    conformation: Conformation,
    network: ElasticNetwork,
    field: np.ndarray,
) -> Conformation:
    """Move every atom of each node's residue rigidly by that node's displacement."""
    from .ensemble import Atom as _Atom
    from .ensemble import _rewrite_coords

    residue_atoms: Dict[int, List[int]] = {}
    for position, atom in enumerate(conformation.atoms):
        residue_atoms.setdefault(position, [])
    # Map each atom to its node through the residue key, so a residue whose Cα is a
    # node moves as a rigid body with it.
    key_to_node = {}
    for node, atom_position in enumerate(network.atom_index):
        ca = conformation.atoms[atom_position]
        key_to_node[(ca.chain, ca.res_id, ca.res_name)] = node
    atoms = []
    records = []
    for position, (atom, line) in enumerate(zip(conformation.atoms, conformation.records)):
        node = key_to_node.get((atom.chain, atom.res_id, atom.res_name))
        if node is None:
            atoms.append(atom)
            records.append(line)
            continue
        delta = field[node] if node is not None else np.zeros(3)
        x, y, z = float(atom.x + delta[0]), float(atom.y + delta[1]), float(atom.z + delta[2])
        atoms.append(
            _Atom(
                name=atom.name, element=atom.element, res_name=atom.res_name,
                res_id=atom.res_id, chain=atom.chain, x=x, y=y, z=z,
                altloc=atom.altloc, record=atom.record,
            )
        )
        records.append(_rewrite_coords(line, (x, y, z)) if line else line)
    return Conformation(
        label=conformation.label, atoms=atoms, records=records, path=conformation.path,
        model=conformation.model, source=conformation.source,
    )


def _count_clashes(
    native: Conformation, member: Conformation
) -> Tuple[int, int]:
    """``(severe, squeezed)`` heavy-atom contacts the displacement created.

    Comparing every pair against the van der Waals sum would count every chemical
    bond (a C–C bond is 1.5 Å, far inside the vdW sum) and every 1–3 pair
    (N···C across a peptide unit is 2.4 Å), so the criterion is *relative to the
    input structure*: a contact is counted when its distance was at or above the
    limit in the native conformation and has come below it in the member.

    Two limits are reported because they answer two questions.
    :data:`CLASH_FACTOR` (0.6) is a genuine overlap — 40 % interpenetration of the
    van der Waals shells — and is what the filter rejects.  :data:`SHEAR_FACTOR`
    (0.75) counts contacts the motion has squeezed, which the rigid-per-residue
    transfer of a collective mode produces by the dozen at a 0.2 Å amplitude; that
    is the coarse-graining's own artefact and it is *reported*, not used to reject
    a member.
    """
    native_heavy = [atom for atom in native.atoms if not atom.is_hydrogen]
    member_heavy = [atom for atom in member.atoms if not atom.is_hydrogen]
    if len(native_heavy) != len(member_heavy):
        raise EnsembleError("the member and its native structure have different atoms")
    before = np.array(
        [[atom.x, atom.y, atom.z] for atom in native_heavy], dtype=float
    ).reshape(-1, 3)
    after = np.array(
        [[atom.x, atom.y, atom.z] for atom in member_heavy], dtype=float
    ).reshape(-1, 3)
    radii = _radii(native_heavy)
    sums = radii[:, None] + radii[None, :]
    severe_limit = CLASH_FACTOR * sums
    shear_limit = SHEAR_FACTOR * sums
    severe = squeezed = 0
    chunk = 512
    for start in range(0, before.shape[0], chunk):
        stop = start + chunk
        native_distance = np.sqrt(
            ((before[start:stop, None, :] - before[None, :, :]) ** 2).sum(axis=2)
        )
        member_distance = np.sqrt(
            ((after[start:stop, None, :] - after[None, :, :]) ** 2).sum(axis=2)
        )
        grabbed = (member_distance < severe_limit[start:stop]) & (
            native_distance >= severe_limit[start:stop]
        )
        touched = (member_distance < shear_limit[start:stop]) & (
            native_distance >= shear_limit[start:stop]
        )
        for row in range(grabbed.shape[0]):
            index = start + row
            severe += int(grabbed[row, index + 1 :].sum())
            squeezed += int(touched[row, index + 1 :].sum())
    return severe, squeezed


def generate_modes(
    conformation: Conformation,
    *,
    site: Optional[str] = None,
    box: Optional[BoxSpec] = None,
    site_radius: float = 8.0,
    max_site_residues: int = 40,
    cutoff: float = DEFAULT_CUTOFF,
    modes: int = DEFAULT_MODES,
    targets: Sequence[float] = DEFAULT_TARGET_RMSD,
    members_per_target: int = 1,
    seed: int = 20240101,
    max_clashes: int = 10,
    source: Optional[Path] = None,
) -> GeneratedModes:
    """Generate backbone conformations by displacing along the low-frequency modes.

    For every ``(mode, target)`` pair one member is produced: the displacement
    field is the mode's Cα field (``sqrt(1/lambda)``-weighted so softer modes move
    more), rescaled so the member's **site RMSD equals the target** exactly.  With
    the default targets that means the ensemble brackets the experimental site
    RMSDs of the three validated pairs by construction -- a calibration, and
    labelled as one in the report.

    Members whose heavy atoms clash are counted and (by default) not kept; the
    table shows every attempted member with its clash count, so a rejected
    amplitude is visible rather than silently missing.
    """
    started = time.perf_counter()
    from .ensemble import select_site

    site_residues = select_site(
        conformation, site=site, box=box, radius=site_radius, max_residues=max_site_residues
    )
    site_keys = {(entry.residue.chain, entry.residue.res_id, entry.residue.res_name) for entry in site_residues}
    if not site_keys:
        raise EnsembleError("no site residues: pass --box, --site or --site-ligand")
    network = build_network(conformation, cutoff=cutoff, modes=modes)
    amplitudes = mode_amplitudes(network)
    if amplitudes.size == 0:
        raise EnsembleError("the elastic network has no non-zero mode to displace along")
    # The site atoms the target is measured on: the Cα and side-chain heavy atoms
    # of the site residues, exactly as the experimental site RMSD is defined.
    position_of = {id(atom): index for index, atom in enumerate(conformation.atoms)}
    site_positions = [
        position_of[id(atom)]
        for entry in site_residues
        for atom in entry.residue.atoms
        if not atom.is_hydrogen and id(atom) in position_of
    ]
    native_site = np.array(
        [
            [conformation.atoms[position].x, conformation.atoms[position].y,
             conformation.atoms[position].z]
            for position in site_positions
        ],
        dtype=float,
    ).reshape(-1, 3)

    rng = np.random.default_rng(int(seed))
    conformations: List[Conformation] = [conformation]
    spread: List[Dict[str, Any]] = [
        {
            "member": "native", "mode": None, "target": 0.0, "site_rmsd": 0.0,
            "max_displacement": 0.0, "clashes": 0, "kept": True,
        }
    ]
    warnings: List[str] = []
    for mode_index in range(min(int(modes), amplitudes.size)):
        for target in targets:
            for repeat in range(max(1, int(members_per_target))):
                # Deterministic sign pattern: the two directions of each mode are
                # different conformations, and both are tried.
                sign = 1.0 if (mode_index + repeat) % 2 == 0 else -1.0
                field = sign * amplitudes[mode_index] * network.mode(mode_index)
                member = _displace(conformation, network, field)
                achieved = float(
                    np.sqrt(
                        (
                            (
                                np.array(
                                    [
                                        [member.atoms[p].x, member.atoms[p].y, member.atoms[p].z]
                                        for p in site_positions
                                    ],
                                    dtype=float,
                                ).reshape(-1, 3)
                                - native_site
                            )
                            ** 2
                        ).sum(axis=1).mean()
                    )
                )
                if achieved > 0.0:
                    scale = float(target) / achieved
                    member = _displace(conformation, network, field * scale)
                per_residue = 0.0
                member_site = np.array(
                    [
                        [member.atoms[p].x, member.atoms[p].y, member.atoms[p].z]
                        for p in site_positions
                    ],
                    dtype=float,
                ).reshape(-1, 3)
                displacement_per_atom = np.sqrt(
                    ((member_site - native_site) ** 2).sum(axis=1)
                )
                per_residue = float(displacement_per_atom.max()) if displacement_per_atom.size else 0.0
                site_rmsd = float(
                    np.sqrt(((member_site - native_site) ** 2).sum(axis=1).mean())
                )
                clashes, squeezed = _count_clashes(conformation, member)
                keep = clashes <= int(max_clashes)
                label = f"mode_{mode_index + 1}_{target:.3f}_{repeat}"
                if not keep:
                    warnings.append(
                        f"{label}: {clashes} severe overlap(s) at a site RMSD of "
                        f"{site_rmsd:.3f} A; not kept (raise --max-clashes to keep it)"
                    )
                spread.append(
                    {
                        "member": label,
                        "mode": mode_index + 1,
                        "target": float(target),
                        "site_rmsd": site_rmsd,
                        "max_displacement": per_residue,
                        "clashes": clashes,
                        "squeezed": squeezed,
                        "kept": keep,
                    }
                )
                if keep:
                    conformations.append(member)
    del rng
    if len(conformations) == 1:
        warnings.append(
            "every displaced member clashed; raise --max-clashes to keep them, "
            "lower the targets, or use fewer modes"
        )
    return GeneratedModes(
        native=conformation,
        network=network,
        conformations=conformations,
        spread=spread,
        parameters={
            "cutoff": float(cutoff),
            "modes": int(modes),
            "targets": [float(value) for value in targets],
            "members_per_target": int(members_per_target),
            "max_clashes": int(max_clashes),
            "seed": int(seed),
            "site_radius": float(site_radius),
        },
        elapsed=time.perf_counter() - started,
        warnings=warnings,
        source=source,
        site_keys=[entry.residue.key for entry in site_residues],
    )


# ---------------------------------------------------------------------------
# The command line: `odock ensemble modes`
# ---------------------------------------------------------------------------


def _cli_eprint(*args: Any, **kwargs: Any) -> None:
    import sys

    print(*args, file=sys.stderr, **kwargs)


def cmd_ensemble_modes(args) -> int:
    """``odock ensemble modes``: backbone motion from one structure."""
    from .ensemble import _cli_box, _cli_write_json, ligand_coords
    from .prepare import box_from_points

    paths = [
        str(path)
        for entry in (getattr(args, "receptor", None) or [])
        for path in (entry if isinstance(entry, (list, tuple)) else [entry])
    ]
    if not paths:
        raise SystemExit("error: pass one -r/--receptor FILE")
    if len(paths) > 1:
        raise SystemExit(
            "error: `ensemble modes` builds an ensemble from *one* structure"
        )
    try:
        conformation = read_conformations(paths, keep_water=bool(args.keep_water))[0]
        box = None
        if args.box_ligand:
            points = ligand_coords(conformation, args.box_ligand)
            box = box_from_points(
                points, buffer=float(args.buffer), spacing=float(args.spacing)
            )
            _cli_eprint(f"box: {box} (from residue {args.box_ligand})")
        elif getattr(args, "box", None) or (args.center and args.size):
            box = _cli_box(args)
        targets = _parse_targets(args.target_site_rmsd)
        generated = generate_modes(
            conformation,
            site=args.site,
            box=box,
            site_radius=float(args.site_radius),
            max_site_residues=int(args.max_site_residues),
            cutoff=float(args.cutoff),
            modes=int(args.modes),
            targets=targets,
            members_per_target=int(args.repeats),
            seed=int(args.seed),
            max_clashes=int(args.max_clashes),
            source=Path(paths[0]),
        )
    except EnsembleError as exc:
        _cli_eprint(f"odock ensemble modes: error: {exc}")
        return int(exc.code)

    if args.compare:
        try:
            other = read_conformations([args.compare], keep_water=bool(args.keep_water))[0]
            from .ensemble import align_conformations

            aligned = align_conformations(
                [conformation, other], box=box, site=args.site,
                site_radius=float(args.site_radius), superpose=True,
                min_identity=float(args.min_identity),
            )
            field, unmatched = experimental_field(
                generated.network, aligned.conformations[0],
                aligned.conformations[1], aligned.alignments[1],
            )
            # Scan more modes than are sampled: if the observed motion is not in
            # the first few, the honest answer is "the direction appears at mode
            # k", not a bare near-zero overlap.  The site-restricted overlap is
            # reported next to the whole-chain one, because a flexible terminus can
            # dominate a whole-chain cosine.
            scan = min(int(args.scan_modes), generated.network.n_modes)
            site_nodes = [
                index
                for index, key in enumerate(generated.network.keys)
                if key in set(generated.site_keys)
            ]
            overlaps = [
                direction_overlap(generated.network, field, index=index)
                for index in range(max(1, scan))
            ]
            site_overlaps = [
                direction_overlap(generated.network, field, index=index, nodes=site_nodes)
                for index in range(max(1, scan))
            ]
            best = int(np.nanargmax(overlaps)) if overlaps else 0
            best_site = int(np.nanargmax(site_overlaps)) if site_overlaps else 0
            generated.overlaps = [
                (
                    f"{Path(paths[0]).stem} -> {Path(args.compare).stem} "
                    f"({generated.network.n_nodes - unmatched}/"
                    f"{generated.network.n_nodes} nodes matched; best of "
                    f"{len(overlaps)} scanned modes is mode {best + 1} at "
                    f"{overlaps[best]:+.3f} over all nodes and mode {best_site + 1} "
                    f"at {site_overlaps[best_site]:+.3f} over the "
                    f"{len(site_nodes)} site node(s); the "
                    f"{generated.network.selected} sampled mode(s) over the site: "
                    + ", ".join(f"{value:+.3f}" for value in site_overlaps[: generated.network.selected]),
                    float(site_overlaps[best_site]) if site_overlaps else float("nan"),
                )
            ]
        except EnsembleError as exc:
            _cli_eprint(f"odock ensemble modes: could not compare directions: {exc}")

    if not args.quiet:
        print(generated.text(limit=int(args.top)))
        print()
        _cli_eprint(
            f"built {generated.network.n_nodes} nodes and "
            f"{generated.network.n_modes} mode(s) in {generated.elapsed:.1f} s"
        )
    if args.outdir:
        written = generated.write(args.outdir)
        _cli_eprint(f"wrote {len(written)} structure(s) to {args.outdir}")
    if args.json_out:
        _cli_write_json(args.json_out, generated.as_dict())
    return 0


def _parse_targets(text: str) -> Tuple[float, ...]:
    if text is None:
        return DEFAULT_TARGET_RMSD
    values = []
    for part in str(text).replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise EnsembleError(
                f"cannot read the target site RMSDs {text!r}: {token!r} is not a number"
            ) from exc
    if not values:
        raise EnsembleError("the target list is empty; pass --target-site-rmsd 0.444")
    return tuple(values)


def add_modes_subparser(ensub: Any) -> None:
    """Register ``odock ensemble modes`` on the ensemble subparsers."""
    parser = ensub.add_parser(
        "modes",
        help="backbone motion from one structure by an elastic network model",
        description=(
            "Build an anisotropic network model over the Cα atoms, take its "
            "low-frequency modes -- the collective motions that connectivity "
            "supports -- and displace the structure along them.  The amplitude is "
            "calibrated so each member's binding-site RMSD hits a target, with the "
            "defaults set to the experimental site RMSDs this project measured, so "
            "the ensemble brackets experiment by construction rather than by "
            "prediction.  It is a harmonic model: no anharmonic transition, one "
            "structure's connectivity, and a displacement along a mode is a "
            "direction at a chosen amplitude, not a sampled state."
        ),
    )
    parser.add_argument(
        "-r", "--receptor", action="append", nargs="+", required=True, metavar="FILE",
        help="the one structure to model (PDB or PDBQT)",
    )
    parser.add_argument("--box", help="box JSON written by `odock box`")
    parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--box-ligand", metavar="RESNAME", help="take the site from this residue")
    parser.add_argument("--buffer", type=float, default=6.0, help="padding for --box-ligand (Å)")
    parser.add_argument("--spacing", type=float, default=0.375, help="box grid spacing (Å)")
    parser.add_argument("--site", metavar="RES[,RES...]", help="explicit site residues")
    parser.add_argument(
        "--site-radius", type=float, default=8.0,
        help="residues within this distance of the site centre define the target RMSD (Å)",
    )
    parser.add_argument("--max-site-residues", type=int, default=40)
    parser.add_argument(
        "--cutoff", type=float, default=DEFAULT_CUTOFF,
        help="Cα–Cα distance below which two residues are connected (Å)",
    )
    parser.add_argument(
        "--modes", type=int, default=DEFAULT_MODES,
        help="how many low-frequency modes to displace along",
    )
    parser.add_argument(
        "--target-site-rmsd", default=None,
        help="comma-separated target site RMSDs (Å); the default is the three "
             "experimental values this project measured (0.444, 0.407, 0.144)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="members per (mode, target) pair",
    )
    parser.add_argument(
        "--max-clashes", type=int, default=10,
        help="severe heavy-atom overlaps (below 0.6 x the vdW sum, relative to the "
             "input structure) a member may have and still be kept; the count is "
             "printed for every member. Measured on 3PTB, an experimental-scale "
             "site RMSD of 0.41-0.44 A costs 1-10 such contacts, so 10 keeps the "
             "experimental amplitudes and rejects only the strained ones. Pass 0 "
             "for a hard filter",
    )
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument(
        "--compare", metavar="FILE",
        help="report the cosine overlap between the modes and the difference to "
             "this structure (does the direction match, or only the magnitude?)",
    )
    parser.add_argument(
        "--scan-modes", type=int, default=30,
        help="how many modes --compare scans for the best overlap with the "
             "observed motion",
    )
    parser.add_argument("--min-identity", type=float, default=0.90, help="for --compare")
    parser.add_argument("--keep-water", action="store_true")
    parser.add_argument("-o", "--outdir", help="write native.pdb, mode_1.pdb, ...")
    parser.add_argument("--top", type=int, default=20, help="rows to print")
    parser.add_argument("--json-out", help="write the whole report as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no report")
    parser.set_defaults(func=cmd_ensemble_modes)

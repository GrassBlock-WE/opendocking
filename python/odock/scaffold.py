# SPDX-License-Identifier: GPL-3.0-or-later
"""Scaffolds, series, R-groups and matched molecular pairs.

This is how a chemist reads a screen.  A ranked hit list says *which* molecules
scored well; these functions say what the hits **have in common** and what to
make next:

* :func:`murcko_scaffold` / :func:`scaffold_of` — the Bemis-Murcko scaffold and
  its generic skeleton, as a molecule or as a canonical SMILES key;
* :func:`scaffold_groups` — the library grouped by scaffold, with counts and a
  representative molecule per group;
* :func:`scaffold_clusters` — those scaffolds clustered by similarity, which is
  what turns "1 000 scaffolds" into "40 series" (the clustering itself is
  :func:`odock.ligandsim.butina_cluster`);
* :func:`rgroups` — the R-group decomposition of a library against a core or a
  scaffold, as a molecule x R-group matrix that writes to CSV/XLSX;
* :func:`series_table` — a shared scaffold with its R-group variants and their
  measured or docked affinities, with a delta against a reference member;
* :func:`matched_pairs` — matched molecular pairs: two molecules that differ in
  exactly one substituent, with the affinity delta of that single change.  This
  is the smallest experiment a series contains, and the only one where a
  structure-activity statement is defensible at all.

Two honesty rules run through the module.  First, a scaffold is a *heuristic*:
a Murcko scaffold keeps rings and linkers and strips terminal substituents, so
benzamidine, aspirin and ibuprofen all share the scaffold ``c1ccccc1`` — the
scaffold is a grouping device, not a series, and the R-group table is what
distinguishes them.  Second, an affinity delta is only a structure-activity
statement when the affinity was *measured* (or docked consistently): the docking
noise this project measured is a ±3 Å seed-to-seed pose swing on a flexible
system and order-of-magnitude ~1 kcal/mol ranking uncertainty, so
:func:`matched_pairs` marks every delta against :data:`DOCKING_NOISE_KCAL` and
says plainly that a delta below it is not a result.  See
`docs/CHEMINFORMATICS.md`.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - RDKit is a hard dependency for chemistry
    Chem = None  # type: ignore[assignment]
    rdFMCS = None  # type: ignore[assignment]
    MurckoScaffold = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

try:
    from rdkit.Chem import rdRGroupDecomposition

    _HAVE_RGD = True
except Exception:  # pragma: no cover - older or trimmed RDKit
    rdRGroupDecomposition = None  # type: ignore[assignment]
    _HAVE_RGD = False

__all__ = [
    "NO_SCAFFOLD_LABEL",
    "DEFAULT_CLUSTER_CUTOFF",
    "DEFAULT_MCS_TIMEOUT",
    "DOCKING_NOISE_KCAL",
    "HAVE_RDKIT",
    "HAVE_RGROUP_DECOMPOSITION",
    "ScaffoldGroup",
    "ScaffoldCluster",
    "ScaffoldClustering",
    "RGroupRow",
    "RGroupDecomposition",
    "SeriesRow",
    "Series",
    "MatchedPair",
    "MatchedPairTable",
    "require_rdkit",
    "scaffold_of",
    "murcko_scaffold",
    "generic_scaffold",
    "scaffold_key",
    "most_common_scaffold",
    "maximum_common_substructure",
    "scaffold_groups",
    "scaffold_clusters",
    "rgroups",
    "series_table",
    "matched_pairs",
    "affinities_from_records",
    "write_rgroup_csv",
    "write_rgroup_xlsx",
    "report_section",
]

#: The key used for a molecule with no ring system, i.e. no Murcko scaffold.
NO_SCAFFOLD_LABEL = "(acyclic)"

#: Similarity at which two *scaffolds* are treated as one series.
DEFAULT_CLUSTER_CUTOFF = 0.65

#: Seconds :func:`maximum_common_substructure` may spend before giving up.
DEFAULT_MCS_TIMEOUT = 10

#: The docking noise this project has measured, in kcal/mol.  The benchmark
#: records a 3.2 Å seed-to-seed pose swing on the flexible 1M17 system and the
#: documentation warns against ordering two hits that differ by less than
#: ~1 kcal/mol.  A matched-pair delta below this is not distinguishable from the
#: noise of the search that produced it; :class:`MatchedPair` reports the
#: comparison rather than making the call for the reader.
DOCKING_NOISE_KCAL = 1.0

#: Whether this build has the RDKit pieces the module needs.
HAVE_RDKIT = _HAVE_RDKIT
#: Whether RDKit's R-group decomposition is available.
HAVE_RGROUP_DECOMPOSITION = _HAVE_RGD


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for scaffolds and R-groups. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


def _require_rgd() -> None:
    if not _HAVE_RGD:
        raise ImportError(
            "this RDKit build has no rdRGroupDecomposition, so the R-group "
            "decomposition is unavailable; scaffold grouping still works"
        )


# ---------------------------------------------------------------------------
# Scaffolds
# ---------------------------------------------------------------------------


def scaffold_of(mol, *, generic: bool = False, include_chirality: bool = False):
    """The Bemis-Murcko scaffold of ``mol`` as a molecule, or ``None``.

    The scaffold is the ring systems of the molecule plus the linkers between
    them (and their double-bonded neighbours), with every terminal substituent
    removed — the framework a medicinal chemist circles when they say "the
    series shares this core".  A molecule with no ring has no scaffold at all and
    returns ``None`` rather than a meaningless empty molecule.

    ``generic=True`` additionally makes every atom and bond generic
    (``MurckoScaffold.MakeScaffoldGeneric``): benzene, pyridine and cyclohexane
    all become ``C1CCCCC1``.  That is the skeleton to use when the question is
    "which shapes are in my library", not "which chemotype".
    """
    require_rdkit()
    if mol is None:
        return None
    try:
        frame = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:  # pragma: no cover - an unsanitisable molecule
        return None
    if frame is None or frame.GetNumAtoms() == 0:
        return None
    if generic:
        frame = MurckoScaffold.MakeScaffoldGeneric(frame)
    if not include_chirality:
        try:
            Chem.RemoveStereochemistry(frame)
        except Exception:  # pragma: no cover - defensive
            pass
    return frame


def murcko_scaffold(
    mol, *, generic: bool = False, include_chirality: bool = False
) -> str:
    """The canonical SMILES of the Murcko scaffold, or ``""`` when there is none.

    ``generic=True`` returns the generic skeleton instead (see
    :func:`scaffold_of`).  An acyclic molecule gives ``""`` — callers that need a
    printable key use :data:`NO_SCAFFOLD_LABEL`.
    """
    frame = scaffold_of(mol, generic=generic, include_chirality=include_chirality)
    if frame is None:
        return ""
    return Chem.MolToSmiles(frame)


def generic_scaffold(mol, *, include_chirality: bool = False) -> str:
    """The canonical SMILES of the **generic** Murcko skeleton, ``""`` if acyclic."""
    return murcko_scaffold(mol, generic=True, include_chirality=include_chirality)


def scaffold_key(mol, *, generic: bool = False) -> str:
    """The grouping key of ``mol``: its scaffold SMILES, or ``""`` when acyclic.

    This is the function :mod:`odock.ligandsim` calls for scaffold coverage, so
    it is deliberately cheap (one scaffold perception, one canonicalisation) and
    free of side effects.
    """
    return murcko_scaffold(mol, generic=generic)


def _mol_name(mol, fallback: str) -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def _to_molecules(items: Iterable[Any]) -> List[Any]:
    """Accept molecules and SMILES strings; return molecules."""
    require_rdkit()
    out: List[Any] = []
    for item in items:
        if isinstance(item, str):
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                raise ValueError(f"{item!r} is not a parsable SMILES")
            out.append(mol)
        else:
            out.append(item)
    return out


def _names_for(molecules: Sequence[Any], names: Optional[Sequence[str]]) -> List[str]:
    return [
        str(names[i]) if names is not None and i < len(names) else _mol_name(mol, f"ligand_{i + 1}")
        for i, mol in enumerate(molecules)
    ]


def _heavy_atoms(mol) -> int:
    try:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    except Exception:  # pragma: no cover - defensive
        return 0


def most_common_scaffold(
    molecules: Sequence[Any], *, generic: bool = False, names: Optional[Sequence[str]] = None
) -> str:
    """The scaffold shared by the most molecules of the library.

    Ties are broken by the scaffold with more heavy atoms and then by SMILES, so
    the answer is deterministic.  This is the default core of :func:`rgroups`:
    the chemotype the library is actually built around.  Returns ``""`` when no
    molecule has a ring.
    """
    counts: Dict[str, int] = {}
    mols = _to_molecules(molecules)
    for mol in mols:
        key = murcko_scaffold(mol, generic=generic)
        if key:
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda key: (counts[key], _heavy_key(key), key))


def _heavy_key(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return _heavy_atoms(mol) if mol is not None else 0


def maximum_common_substructure(
    molecules: Sequence[Any], *, timeout: int = DEFAULT_MCS_TIMEOUT, generic: bool = False
) -> str:
    """The maximum common substructure of the library, as a SMARTS/SMILES core.

    This is the honest "common core" of a *set* of molecules — the largest
    substructure every member contains — as opposed to a Murcko scaffold, which
    is derived from one molecule at a time.  It is also the expensive one: the
    MCS problem is NP-hard and :func:`rdFMCS.FindMCS` will happily take minutes
    on a large, diverse library, so the search is bounded by ``timeout`` seconds
    and returns ``""`` when it is cancelled or finds nothing.  For a 17-molecule
    demo library it is milliseconds; for 1 000 diverse drugs pass a bigger
    ``timeout`` and expect the answer to be small.
    """
    require_rdkit()
    mols = _to_molecules(molecules)
    mols = [mol for mol in mols if mol is not None and mol.GetNumAtoms() > 0]
    if len(mols) < 2:
        return ""
    try:
        result = rdFMCS.FindMCS(
            mols,
            timeout=int(timeout),
            completeRingsOnly=True,
            ringMatchesRingOnly=True,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        )
    except Exception:  # pragma: no cover - RDKit raises rarely
        return ""
    if getattr(result, "canceled", False) or not getattr(result, "numAtoms", 0):
        return ""
    smarts = str(getattr(result, "smartsString", "") or getattr(result, "smarts", ""))
    if not smarts:
        return ""
    core = Chem.MolFromSmarts(smarts)
    if core is None:  # pragma: no cover - defensive
        return ""
    # `FindMCS` returns a SMARTS whose aromatic bonds are flagged but whose atoms
    # are not aromatic, which canonicalises as the unreadable
    # `NC(=N)C1:C:C:C:C:C:1` and cannot be kekulised.  Round-tripping through
    # SMILES gives the string a chemist would write (`N=C(N)c1ccccc1`) and one
    # that reads back as an aromatic core, so the same string can be passed to
    # --core and used as a substructure.
    try:
        readable = Chem.MolFromSmiles(Chem.MolToSmiles(core))
    except Exception:  # pragma: no cover - defensive
        readable = None
    if readable is None:  # pragma: no cover - defensive
        readable = core
    if generic:
        readable = MurckoScaffold.MakeScaffoldGeneric(readable)
    try:
        return Chem.MolToSmiles(readable)
    except Exception:  # pragma: no cover - defensive
        return ""


# ---------------------------------------------------------------------------
# Scaffold grouping
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldGroup:
    """One scaffold and the library members that share it."""

    #: Canonical scaffold SMILES (``""`` for the acyclic group).
    key: str
    #: Canonical *generic* skeleton SMILES.
    generic_key: str
    #: Printable label: the key, or :data:`NO_SCAFFOLD_LABEL`.
    label: str
    #: Library indices of the members, ascending.
    members: List[int] = field(default_factory=list)
    #: Member names, in the same order as :attr:`members`.
    names: List[str] = field(default_factory=list)
    #: Affinity per member (``None`` when unknown), same order.
    affinities: List[Optional[float]] = field(default_factory=list)
    #: The member with the most heavy atoms (ties: lowest index) — the molecule
    #: that shows the decoracted scaffold best.
    representative: int = 0
    representative_name: str = ""

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def best_affinity(self) -> Optional[float]:
        values = [v for v in self.affinities if v is not None]
        return min(values) if values else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scaffold": self.key,
            "generic_scaffold": self.generic_key,
            "label": self.label,
            "count": int(self.count),
            "members": [int(i) for i in self.members],
            "names": list(self.names),
            "affinities": [None if v is None else round(float(v), 4) for v in self.affinities],
            "representative": int(self.representative),
            "representative_name": self.representative_name,
            "best_affinity": None if self.best_affinity is None else round(self.best_affinity, 4),
        }


def _affinity_lookup(
    affinities: Optional[Union[Dict[str, float], Dict[int, float], Sequence[Any]]]
) -> Dict[str, float]:
    """Normalise an affinity argument into a ``{name: kcal/mol}`` mapping."""
    if affinities is None:
        return {}
    if isinstance(affinities, dict):
        out: Dict[str, float] = {}
        for key, value in affinities.items():
            if value is None:
                continue
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out
    out = {}
    for item in affinities:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            try:
                out[str(item[0])] = float(item[1])
            except (TypeError, ValueError):
                continue
    return out


def _affinities_for(names: Sequence[str], lookup: Dict[str, float]) -> List[Optional[float]]:
    return [lookup.get(name) for name in names]


def scaffold_groups(
    molecules: Sequence[Any],
    *,
    generic: bool = False,
    names: Optional[Sequence[str]] = None,
    affinities: Optional[Union[Dict[str, float], Dict[int, float]]] = None,
) -> List[ScaffoldGroup]:
    """Group a library by Murcko scaffold, with counts and representatives.

    Groups come back sorted by decreasing member count, then by the scaffold
    SMILES, so the answer is stable and the *largest series is first* — the
    number a reader looks for.  The acyclic molecules (no ring at all) form their
    own group keyed ``""`` and labelled :data:`NO_SCAFFOLD_LABEL` rather than
    being dropped.

    ``affinities`` maps a molecule name to a kcal/mol value (or an index to one);
    each group then reports its members' values and the best of them, which is
    what makes this a *series* table rather than a counting exercise.
    """
    require_rdkit()
    mols = _to_molecules(molecules)
    labels = _names_for(mols, names)
    lookup = _affinity_lookup(affinities)
    buckets: Dict[str, ScaffoldGroup] = {}
    for index, mol in enumerate(mols):
        key = scaffold_key(mol, generic=generic)
        group = buckets.get(key)
        if group is None:
            group = ScaffoldGroup(
                key=key,
                generic_key=generic_scaffold(mol),
                label=key or NO_SCAFFOLD_LABEL,
            )
            buckets[key] = group
        group.members.append(index)
        group.names.append(labels[index])
        group.affinities.append(lookup.get(labels[index]))
    for group in buckets.values():
        heaviest = max(
            group.members,
            key=lambda i: (_heavy_atoms(mols[i]), -i),
        )
        group.representative = heaviest
        group.representative_name = labels[heaviest]
    return sorted(buckets.values(), key=lambda g: (-g.count, g.key))


@dataclass
class ScaffoldCluster:
    """A group of similar scaffolds: one chemotype family."""

    rank: int
    #: The scaffold SMILES in this family, largest member count first.
    scaffolds: List[str] = field(default_factory=list)
    #: Library indices of every molecule whose scaffold is in the family.
    members: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    #: The medoid scaffold of the family and one molecule that shows it.
    representative_scaffold: str = ""
    representative_name: str = ""
    representative_index: int = -1

    @property
    def n_scaffolds(self) -> int:
        return len(self.scaffolds)

    @property
    def n_molecules(self) -> int:
        return len(self.members)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "n_scaffolds": int(self.n_scaffolds),
            "n_molecules": int(self.n_molecules),
            "scaffolds": list(self.scaffolds),
            "members": [int(i) for i in self.members],
            "names": list(self.names),
            "representative_scaffold": self.representative_scaffold,
            "representative_name": self.representative_name,
            "representative_index": int(self.representative_index),
        }


@dataclass
class ScaffoldClustering:
    """A library's scaffolds clustered by similarity: the series view."""

    clusters: List[ScaffoldCluster] = field(default_factory=list)
    cutoff: float = DEFAULT_CLUSTER_CUTOFF
    metric: str = "tanimoto"
    n_scaffolds: int = 0
    n_molecules: int = 0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def largest(self) -> Optional[ScaffoldCluster]:
        return self.clusters[0] if self.clusters else None

    def table(self, limit: int = 20) -> str:
        """One line per scaffold family."""
        if not self.clusters:
            return "no scaffolds"
        rows = self.clusters if limit <= 0 else self.clusters[: int(limit)]
        lines = [
            f"{'series':<7}{'scaffolds':>9}{'molecules':>10}  representative",
            "-" * 7 + "-" * 9 + "-" * 10 + "  " + "-" * 40,
        ]
        for cluster in rows:
            lines.append(
                f"{cluster.rank:<7}{cluster.n_scaffolds:>9}{cluster.n_molecules:>10}  "
                f"{cluster.representative_scaffold[:40]} ({cluster.representative_name[:20]})"
            )
        if limit > 0 and len(self.clusters) > limit:
            lines.append(f"... and {len(self.clusters) - limit} more series")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cutoff": float(self.cutoff),
            "metric": self.metric,
            "n_scaffolds": int(self.n_scaffolds),
            "n_molecules": int(self.n_molecules),
            "n_series": int(self.n_clusters),
            "seconds": round(float(self.seconds), 4),
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "notes": list(self.notes),
        }


def scaffold_clusters(
    molecules: Sequence[Any],
    *,
    cutoff: float = DEFAULT_CLUSTER_CUTOFF,
    generic: bool = False,
    names: Optional[Sequence[str]] = None,
    affinities: Optional[Union[Dict[str, float], Dict[int, float]]] = None,
) -> ScaffoldClustering:
    """Cluster a library's *scaffolds*, not its molecules, at ``cutoff``.

    Group the library by scaffold (:func:`scaffold_groups`), fingerprint each
    distinct scaffold (Morgan, radius 2) and cluster those fingerprints with
    :func:`odock.ligandsim.butina_cluster`.  The result answers "how many
    chemotypes does this library really contain": a single-chemistry library
    answers 1, an assembled one answers tens, and a 20 % diverse subset that
    covers as many series as the full library has done its job.

    The representative of a family is the medoid scaffold; the reported
    representative *molecule* is the first library member carrying it.
    """
    require_rdkit()
    mols = _to_molecules(molecules)
    labels = _names_for(mols, names)
    lookup = _affinity_lookup(affinities)
    groups = scaffold_groups(mols, generic=generic, names=labels, affinities=lookup)
    start = time.perf_counter()
    # The acyclic group has no structure to fingerprint; it is reported
    # separately rather than silently folded into a family.
    keys = [group.key for group in groups if group.key]
    acyclic = [group for group in groups if not group.key]
    clusters: List[ScaffoldCluster] = []
    notes: List[str] = []
    if acyclic:
        total = sum(group.count for group in acyclic)
        notes.append(
            f"{total} acyclic molecule(s) have no scaffold and form no series"
        )
    if keys:
        from .ligandsim import butina_cluster, fingerprint

        fps = [fingerprint(Chem.MolFromSmiles(key), kind="morgan", radius=2) for key in keys]
        clustering = butina_cluster(fps, cutoff=float(cutoff))
        by_key = {group.key: group for group in groups}
        for cluster in clustering.clusters:
            family_keys = [keys[i] for i in cluster.members]
            family_keys.sort(key=lambda key: (-by_key[key].count, key))
            members: List[int] = []
            family_names: List[str] = []
            for key in family_keys:
                group = by_key[key]
                members.extend(group.members)
                family_names.extend(group.names)
            representative = keys[cluster.representative]
            clusters.append(
                ScaffoldCluster(
                    rank=0,
                    scaffolds=family_keys,
                    members=sorted(members),
                    names=family_names,
                    representative_scaffold=representative,
                    representative_name=by_key[representative].representative_name,
                    representative_index=by_key[representative].representative,
                )
            )
        clusters.sort(key=lambda c: (-c.n_molecules, c.representative_scaffold))
    for position, cluster in enumerate(clusters, start=1):
        cluster.rank = position
    return ScaffoldClustering(
        clusters=clusters,
        cutoff=float(cutoff),
        n_scaffolds=len(keys),
        n_molecules=len(mols),
        seconds=time.perf_counter() - start,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# R-group decomposition
# ---------------------------------------------------------------------------


@dataclass
class RGroupRow:
    """One molecule of an R-group decomposition."""

    index: int
    name: str
    #: Whether the core was found (and the decomposition succeeded).
    matched: bool
    #: ``label -> substituent SMILES``; ``"H"`` where the core atom carries a
    #: hydrogen instead of a substituent.
    rgroups: Dict[str, str] = field(default_factory=dict)
    #: The molecule's affinity, when one was supplied.
    affinity: Optional[float] = None
    #: The molecule's canonical SMILES.
    smiles: str = ""
    #: Why the decomposition failed, for an unmatched row.
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "name": self.name,
            "matched": bool(self.matched),
            "rgroups": dict(self.rgroups),
            "affinity": None if self.affinity is None else round(float(self.affinity), 4),
            "smiles": self.smiles,
            "note": self.note,
        }


@dataclass
class RGroupDecomposition:
    """A molecule x R-group table: the sheet a medicinal chemist asks for first."""

    #: The core as given, canonical SMILES (without attachment labels).
    core: str
    #: The core as RDKit labelled it, with ``[*:n]`` attachment points.
    core_with_labels: str
    #: R-group labels in positional order (see :func:`rgroups`).
    labels: List[str] = field(default_factory=list)
    rows: List[RGroupRow] = field(default_factory=list)
    n_molecules: int = 0
    #: How the core was chosen (``"given"``, ``"scaffold"`` or ``"mcs"``).
    core_source: str = "given"
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return sum(1 for row in self.rows if row.matched)

    @property
    def match_rate(self) -> float:
        return float(self.n_matched) / float(self.n_molecules) if self.n_molecules else 0.0

    def matrix(self) -> Tuple[List[str], List[List[Any]]]:
        """``(header, rows)`` of the molecule x R-group matrix, ready for a sheet.

        The header is ``["name", "affinity", "matched", *labels]``; every row is
        ``[name, affinity, "1"/"0", *values]`` with an empty string where a label
        does not apply (an unmatched molecule).
        """
        header = ["name", "affinity", "matched"] + list(self.labels)
        body: List[List[Any]] = []
        for row in self.rows:
            body.append(
                [
                    row.name,
                    "" if row.affinity is None else round(float(row.affinity), 4),
                    "1" if row.matched else "0",
                    *[row.rgroups.get(label, "") for label in self.labels],
                ]
            )
        return header, body

    def table(self, limit: int = 0) -> str:
        """A fixed-width text table of the decomposition."""
        if not self.rows:
            return "no molecules"
        rows = self.rows if limit <= 0 else self.rows[: int(limit)]
        header = ["name", "affinity", "matched"] + list(self.labels)
        widths = [
            max([len("name")] + [len(row.name) for row in rows]),
            len("affinity"),
            len("matched"),
        ]
        for label in self.labels:
            widths.append(
                max([len(label)] + [len(row.rgroups.get(label, "")) for row in rows])
            )
        lines = [
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(header)).rstrip(),
            "  ".join("-" * width for width in widths),
        ]
        for row in rows:
            cells = [
                row.name,
                "" if row.affinity is None else f"{row.affinity:.3f}",
                "yes" if row.matched else "no",
                *[row.rgroups.get(label, "") for label in self.labels],
            ]
            lines.append(
                "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()
            )
        if limit > 0 and len(self.rows) > limit:
            lines.append(f"... and {len(self.rows) - limit} more molecule(s)")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        header, rows = self.matrix()
        return {
            "core": self.core,
            "core_with_labels": self.core_with_labels,
            "core_source": self.core_source,
            "labels": list(self.labels),
            "n_molecules": int(self.n_molecules),
            "n_matched": int(self.n_matched),
            "match_rate": round(self.match_rate, 6),
            "seconds": round(float(self.seconds), 4),
            "header": header,
            "rows": rows,
            "notes": list(self.notes),
        }


def _resolve_core(
    molecules: Sequence[Any], core: Optional[Union[str, Any]], generic: bool
) -> Tuple[Any, str, str]:
    """Return ``(core mol, printable core SMILES, source name)``."""
    if core is None:
        key = most_common_scaffold(molecules, generic=generic)
        if not key:
            raise ValueError(
                "no molecule of the library has a ring, so no scaffold core could "
                "be derived; pass an explicit core=... (a SMILES)"
            )
        mol = Chem.MolFromSmiles(key)
        if mol is None:  # pragma: no cover - defensive
            raise ValueError(f"the derived scaffold {key!r} did not parse")
        return mol, key, "scaffold"
    if isinstance(core, str):
        text = core.strip()
        if text.lower() in ("mcs", "common"):
            key = maximum_common_substructure(molecules, generic=generic)
            if not key:
                raise ValueError(
                    "the maximum common substructure search found no core within "
                    "its time limit; pass an explicit core=... or a longer timeout"
                )
            mol = Chem.MolFromSmiles(key)
            if mol is None:  # pragma: no cover - defensive
                raise ValueError(f"the MCS {key!r} did not parse")
            return mol, key, "mcs"
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            raise ValueError(f"the core {core!r} is not a parsable SMILES")
        return mol, Chem.MolToSmiles(mol), "given"
    return core, Chem.MolToSmiles(core), "given"


def _label_plan(core_with_labels) -> Tuple[List[str], Dict[str, str]]:
    """Map RDKit's ``R<n>`` labels onto positional ``R1..Rk`` labels.

    RDKit numbers the attachment points in the order the core's dummy atoms were
    written, which is an accident of the implementation.  A table that a chemist
    reads (and diffs between runs) needs labels tied to the *position*: the
    attachment points are therefore renumbered by the canonical rank of the core
    atom they are attached to, with the original label breaking a tie (two groups
    on the same atom).  The returned mapping is ``{rdkit label: new label}``.
    """
    require_rdkit()
    dummy_indices = sorted(
        atom.GetIdx() for atom in core_with_labels.GetAtoms() if atom.GetAtomicNum() == 0
    )
    if not dummy_indices:
        return [], {}
    # Removing the dummies renumbers the remaining atoms; every attachment atom
    # moves down by the number of dummies that precede it.
    stripped = Chem.RWMol(core_with_labels)
    for index in reversed(dummy_indices):
        stripped.RemoveAtom(index)
    core_mol = stripped.GetMol()
    try:
        ranks = list(Chem.CanonicalRankAtoms(core_mol, breakTies=True))
    except Exception:  # pragma: no cover - defensive
        ranks = [0] * core_mol.GetNumAtoms()
    plan: List[Tuple[int, int, str]] = []
    for atom in core_with_labels.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        number = int(atom.GetAtomMapNum() or 0)
        neighbours = list(atom.GetNeighbors())
        if not neighbours:
            continue
        original = neighbours[0].GetIdx()
        shifted = original - sum(1 for index in dummy_indices if index < original)
        rank = int(ranks[shifted]) if 0 <= shifted < len(ranks) else shifted
        plan.append((rank, number, f"R{number}"))
    plan.sort()
    labels = [f"R{position}" for position in range(1, len(plan) + 1)]
    mapping = {original: new for new, (_, _, original) in zip(labels, plan)}
    return labels, mapping


def _substituent_smiles(group_mol, map_num: int) -> Tuple[str, str]:
    """``(substituent SMILES, problem)`` for one R-group of one molecule.

    The substituent is written with its attachment point as ``*`` — ``*C`` is a
    methyl, ``*C(=O)O`` a carboxylic acid, ``*O`` a hydroxyl — and a hydrogen is
    written ``H``.  Keeping the attachment marker is what makes the column
    unambiguous: a bare ``O`` would be water and a bare ``C(=O)O`` formic acid,
    neither of which is the group.

    A cell that still carries another attachment point after the labelled one is
    removed is not a substituent at all: it closes a ring (the fused-ring case),
    and the molecule cannot be written as a core plus acyclic R-groups.  The
    problem is returned as a sentence rather than a wrong SMILES.
    """
    matches = [
        atom.GetIdx()
        for atom in group_mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and int(atom.GetAtomMapNum() or 0) == int(map_num)
    ]
    if not matches:
        return "", f"RDKit returned no R-group for label {map_num}"
    work = Chem.RWMol(group_mol)
    for atom in work.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
    out = work.GetMol()
    dummies = [atom for atom in out.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) > 1:
        return (
            "",
            "an R-group of this molecule closes a ring, so the core plus acyclic "
            "substituents cannot express it (a fused-ring analogue)",
        )
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    heavy = [atom for atom in out.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy:
        return "H", ""
    try:
        return Chem.MolToSmiles(out), ""
    except Exception:  # pragma: no cover - defensive
        return "", "the R-group could not be canonicalised"


def _core_smiles_without_labels(text: str) -> str:
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    work = Chem.RWMol(mol)
    for index in sorted(
        [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0], reverse=True
    ):
        work.RemoveAtom(index)
    out = work.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        pass
    try:
        return Chem.MolToSmiles(out)
    except Exception:  # pragma: no cover - defensive
        return text


def rgroups(
    molecules: Sequence[Any],
    *,
    core: Optional[Union[str, Any]] = None,
    generic: bool = False,
    names: Optional[Sequence[str]] = None,
    affinities: Optional[Union[Dict[str, float], Dict[int, float]]] = None,
) -> RGroupDecomposition:
    """Decompose every molecule into its substituents per attachment point.

    Parameters
    ----------
    molecules
        The library (molecules or SMILES strings).
    core
        The common core.  ``None`` (default) uses the most common Murcko scaffold
        of the library; ``"mcs"`` uses the maximum common substructure
        (:func:`maximum_common_substructure`); a SMILES string or a molecule is
        used as given.  A core with explicit ``[*:n]`` attachment points is
        honoured; without them RDKit's decomposition finds the attachment points
        it needs.
    names, affinities
        Per-molecule names and kcal/mol values, as in :func:`scaffold_groups`.

    Returns
    -------
    :class:`RGroupDecomposition` with one row per input molecule — including the
    molecules the core does not match, which are reported as ``matched=False``
    with the reason instead of being dropped.  The labels are positional
    (``R1..Rk``, ordered by the canonical rank of the core atom they sit on), so
    the same position always gets the same column.

    Notes
    -----
    The decomposition itself is RDKit's ``rdRGroupDecomposition``, which solves
    the alignment problem globally: for a symmetric core it picks the assignment
    that fits *the whole set*, not each molecule independently.  A molecule whose
    substituent closes a ring (a fused-ring analogue) cannot be written as a
    core plus acyclic R-groups and comes back unmatched, which is a real limit of
    the method rather than a bug.
    """
    require_rdkit()
    _require_rgd()
    mols = _to_molecules(molecules)
    if not mols:
        raise ValueError("no molecules to decompose")
    labels_input = _names_for(mols, names)
    lookup = _affinity_lookup(affinities)
    core_mol, core_text, core_source = _resolve_core(mols, core, generic)
    start = time.perf_counter()
    params = rdRGroupDecomposition.RGroupDecompositionParameters()
    for attribute, value in (
        ("removeAllHydrogenRGroups", False),
        ("removeAllHydrogenRGroupsAndLabels", False),
        ("onlyMatchAtRGroups", False),
        ("allowMultipleRGroupsOnUncyclizedSimilarAtoms", True),
        ("alignCore", True),
    ):
        if hasattr(params, attribute):
            try:
                setattr(params, attribute, value)
            except Exception:  # pragma: no cover - parameter read-only in some builds
                pass
    try:
        decomposition = rdRGroupDecomposition.RGroupDecomposition(core_mol, params)
    except Exception as exc:
        raise ValueError(
            f"RDKit could not prepare the core {core_text!r} ({exc}); pass a "
            "smaller core or a different one"
        ) from None
    for mol in mols:
        decomposition.Add(mol)
    try:
        decomposition.Process()
    except Exception as exc:
        raise ValueError(
            f"RDKit could not decompose the library against the core {core_text!r} "
            f"({exc}); a core that matches only part of most molecules, or one with "
            "several attachment points on the same atom, can do this"
        ) from None
    columns = decomposition.GetRGroupsAsColumns()
    rgroup_labels = [label for label in decomposition.GetRGroupLabels() if label != "Core"]
    core_column = columns.get("Core") or []
    core_mol_labelled = core_column[0] if core_column else None
    label_map: Dict[str, str] = {}
    labels: List[str] = []
    if core_mol_labelled is not None:
        labels, label_map = _label_plan(core_mol_labelled)
    core_with_labels = (
        Chem.MolToSmiles(core_mol_labelled) if core_mol_labelled is not None else core_text
    )
    # Which molecules RDKit matched, in the order it appended them: the column
    # lists only carry the successful ones, and `GetMatchingCoreIdx` is the API
    # that says which those were.
    matched_flags: List[bool] = []
    for mol in mols:
        try:
            found = decomposition.GetMatchingCoreIdx(mol)
        except Exception:  # pragma: no cover - defensive
            found = -1
        matched_flags.append(bool(found is not None and int(found) >= 0))
    rows: List[RGroupRow] = []
    cursor = 0
    for index, mol in enumerate(mols):
        name = labels_input[index]
        smiles = _safe_smiles(mol)
        affinity = lookup.get(name)
        if not matched_flags[index]:
            rows.append(
                RGroupRow(
                    index=index,
                    name=name,
                    matched=False,
                    affinity=affinity,
                    smiles=smiles,
                    note="the core is not a substructure of this molecule",
                )
            )
            continue
        values: Dict[str, str] = {}
        failure = ""
        for original in rgroup_labels:
            column = columns.get(original) or []
            cell = column[cursor] if cursor < len(column) else None
            if cell is None:
                failure = "RDKit did not decompose this molecule"
                break
            map_num = int(original.lstrip("R") or 0) if original.startswith("R") else 0
            text, problem = _substituent_smiles(cell, map_num)
            if problem:
                failure = problem
                break
            values[label_map.get(original, original)] = text
        cursor += 1
        if failure:
            rows.append(
                RGroupRow(
                    index=index,
                    name=name,
                    matched=False,
                    affinity=affinity,
                    smiles=smiles,
                    note=failure,
                )
            )
            continue
        rows.append(
            RGroupRow(
                index=index,
                name=name,
                matched=True,
                rgroups={label: values.get(label, "H") for label in labels},
                affinity=affinity,
                smiles=smiles,
            )
        )
    notes: List[str] = []
    n_unmatched = sum(1 for row in rows if not row.matched)
    if n_unmatched:
        notes.append(
            f"{n_unmatched} of {len(rows)} molecule(s) do not contain the core and have "
            "no R-group row"
        )
    # The R-group columns only hold the molecules RDKit matched, in the order it
    # appended them, so a disagreement between `GetMatchingCoreIdx` and the column
    # length would shift every row by one.  That would be a wrong table rather
    # than a missing value, so it is reported loudly.
    if core_mol_labelled is not None and cursor != len(core_column):
        notes.append(
            "warning: RDKit matched a different number of molecules than its column "
            f"length ({cursor} vs {len(core_column)}); the R-group rows may be "
            "misaligned, please report this molecule set"
        )
    return RGroupDecomposition(
        core=_core_smiles_without_labels(core_with_labels) if core_mol_labelled is not None else core_text,
        core_with_labels=core_with_labels,
        labels=labels,
        rows=rows,
        n_molecules=len(mols),
        core_source=core_source,
        seconds=time.perf_counter() - start,
        notes=notes,
    )


def _safe_smiles(mol) -> str:
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - defensive
        return ""


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


@dataclass
class SeriesRow:
    """One member of a series: its substituents and its affinity."""

    index: int
    name: str
    rgroups: Dict[str, str] = field(default_factory=dict)
    affinity: Optional[float] = None
    #: ``affinity - reference affinity`` (negative = better than the reference).
    delta: Optional[float] = None
    smiles: str = ""
    is_reference: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "name": self.name,
            "rgroups": dict(self.rgroups),
            "affinity": None if self.affinity is None else round(float(self.affinity), 4),
            "delta": None if self.delta is None else round(float(self.delta), 4),
            "smiles": self.smiles,
            "is_reference": bool(self.is_reference),
        }


@dataclass
class Series:
    """A shared core, its R-group variants and their affinities."""

    core: str
    labels: List[str] = field(default_factory=list)
    rows: List[SeriesRow] = field(default_factory=list)
    #: Name of the row every delta is measured against.
    reference: str = ""
    core_source: str = "given"
    notes: List[str] = field(default_factory=list)

    @property
    def n_members(self) -> int:
        return len(self.rows)

    @property
    def with_affinity(self) -> List[SeriesRow]:
        return [row for row in self.rows if row.affinity is not None]

    @property
    def best(self) -> Optional[SeriesRow]:
        scored = self.with_affinity
        return min(scored, key=lambda row: row.affinity) if scored else None

    @property
    def span(self) -> Optional[float]:
        """The measured affinity range of the series (kcal/mol)."""
        scored = self.with_affinity
        if len(scored) < 2:
            return None
        values = [float(row.affinity) for row in scored]
        return max(values) - min(values)

    def table(self) -> str:
        if not self.rows:
            return "empty series"
        header = ["name", "affinity", "delta"] + list(self.labels)
        widths = [
            max([len("name")] + [len(row.name) for row in self.rows]),
            len("affinity"),
            len("delta"),
        ]
        for label in self.labels:
            widths.append(max([len(label)] + [len(row.rgroups.get(label, "")) for row in self.rows]))
        lines = [
            "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(header)).rstrip(),
            "  ".join("-" * width for width in widths),
        ]
        for row in self.rows:
            cells = [
                row.name,
                "" if row.affinity is None else f"{row.affinity:.3f}",
                "" if row.delta is None else f"{row.delta:+.3f}",
                *[row.rgroups.get(label, "") for label in self.labels],
            ]
            lines.append(
                "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()
            )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "core": self.core,
            "core_source": self.core_source,
            "labels": list(self.labels),
            "reference": self.reference,
            "n_members": int(self.n_members),
            "n_with_affinity": len(self.with_affinity),
            "span": None if self.span is None else round(self.span, 4),
            "best": self.best.name if self.best else "",
            "rows": [row.as_dict() for row in self.rows],
            "notes": list(self.notes),
        }


def series_table(
    molecules: Sequence[Any],
    affinities: Optional[Union[Dict[str, float], Dict[int, float]]] = None,
    *,
    core: Optional[Union[str, Any]] = None,
    generic: bool = False,
    names: Optional[Sequence[str]] = None,
    reference: Optional[Union[str, int]] = None,
) -> Series:
    """A shared core with its R-group variants and their affinities.

    The R-group decomposition (:func:`rgroups`) supplies the substituents; this
    function adds the affinity and a delta against the reference member, which is
    what turns a table of structures into a structure-activity series.

    ``reference`` selects the row every delta is measured from — a molecule name
    or a library index.  The default is the **first matched molecule in library
    order**, which is deterministic and, for the usual case of a library written
    parent-first, is the unsubstituted parent.  Rows without an affinity get no
    delta rather than a made-up zero.
    """
    decomposition = rgroups(
        molecules, core=core, generic=generic, names=names, affinities=affinities
    )
    matched = [row for row in decomposition.rows if row.matched]
    if not matched:
        return Series(
            core=decomposition.core,
            labels=decomposition.labels,
            rows=[],
            core_source=decomposition.core_source,
            notes=["no molecule of the library contains the core"],
        )
    chosen: Optional[RGroupRow] = None
    if reference is None:
        chosen = matched[0]
    elif isinstance(reference, int):
        for row in matched:
            if row.index == reference:
                chosen = row
                break
    else:
        for row in matched:
            if row.name == str(reference):
                chosen = row
                break
    notes: List[str] = []
    if chosen is None:
        notes.append(
            f"the reference {reference!r} is not a matched member of the series; "
            f"using {matched[0].name!r}"
        )
        chosen = matched[0]
    reference_value = chosen.affinity
    if reference_value is None:
        notes.append(
            f"the reference {chosen.name!r} has no affinity, so no delta could be "
            "computed; pass an affinity for it"
        )
    rows: List[SeriesRow] = []
    for row in matched:
        delta = None
        if reference_value is not None and row.affinity is not None:
            delta = float(row.affinity) - float(reference_value)
        rows.append(
            SeriesRow(
                index=row.index,
                name=row.name,
                rgroups=dict(row.rgroups),
                affinity=row.affinity,
                delta=delta,
                smiles=row.smiles,
                is_reference=row.index == chosen.index,
            )
        )
    return Series(
        core=decomposition.core,
        labels=decomposition.labels,
        rows=rows,
        reference=chosen.name,
        core_source=decomposition.core_source,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Matched molecular pairs
# ---------------------------------------------------------------------------


@dataclass
class MatchedPair:
    """Two molecules that differ in exactly one substituent, and their delta."""

    core: str
    #: The attachment point that changed.
    label: str
    #: The substituent on each side (``"H"`` for a hydrogen).
    rgroup_a: str
    rgroup_b: str
    index_a: int
    index_b: int
    name_a: str
    name_b: str
    affinity_a: Optional[float] = None
    affinity_b: Optional[float] = None
    #: ``affinity_b - affinity_a`` in kcal/mol (negative = b is better).
    delta: Optional[float] = None
    #: Heavy atoms of each substituent (``None`` when the SMILES did not parse).
    heavy_atoms_a: Optional[int] = None
    heavy_atoms_b: Optional[int] = None

    @property
    def added_heavy_atoms(self) -> Optional[int]:
        """How many heavy atoms the change adds (negative = it removes some).

        A single-point substitution is not automatically a single-*atom* change:
        ``H -> *C(=O)O`` adds three heavy atoms and a hydrogen-bond donor and
        acceptor.  Reporting the size of the change is what keeps a large delta
        from being read as a clean substituent effect.
        """
        if self.heavy_atoms_a is None or self.heavy_atoms_b is None:
            return None
        return int(self.heavy_atoms_b) - int(self.heavy_atoms_a)

    @property
    def significant(self) -> bool:
        """Whether the delta exceeds the docking noise this project measured.

        ``False`` does **not** mean the pair is wrong — it means the difference
        cannot be told apart from the search's own noise by docking alone.  A
        measured (assayed) affinity has its own error, which this flag says
        nothing about.
        """
        return self.delta is not None and abs(float(self.delta)) >= DOCKING_NOISE_KCAL

    def transformation(self) -> str:
        """``"name_a [R2=H] -> name_b [R2=*O]  dA -0.519 kcal/mol (+1 heavy)"``."""
        delta = "n/a" if self.delta is None else f"{self.delta:+.3f} kcal/mol"
        added = self.added_heavy_atoms
        size = "" if added is None else f" ({added:+d} heavy atom(s))"
        return (
            f"{self.name_a} [{self.label}={self.rgroup_a}] -> "
            f"{self.name_b} [{self.label}={self.rgroup_b}]  dA {delta}{size}"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "core": self.core,
            "label": self.label,
            "rgroup_a": self.rgroup_a,
            "rgroup_b": self.rgroup_b,
            "name_a": self.name_a,
            "name_b": self.name_b,
            "index_a": int(self.index_a),
            "index_b": int(self.index_b),
            "affinity_a": None if self.affinity_a is None else round(float(self.affinity_a), 4),
            "affinity_b": None if self.affinity_b is None else round(float(self.affinity_b), 4),
            "delta": None if self.delta is None else round(float(self.delta), 4),
            "heavy_atoms_a": self.heavy_atoms_a,
            "heavy_atoms_b": self.heavy_atoms_b,
            "added_heavy_atoms": self.added_heavy_atoms,
            "significant": bool(self.significant),
        }


@dataclass
class MatchedPairTable:
    """Every single-point substitution the library contains, with its delta."""

    pairs: List[MatchedPair] = field(default_factory=list)
    core: str = ""
    labels: List[str] = field(default_factory=list)
    n_molecules: int = 0
    n_matched: int = 0
    n_pairs_considered: int = 0
    noise: float = DOCKING_NOISE_KCAL
    notes: List[str] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_significant(self) -> int:
        return sum(1 for pair in self.pairs if pair.significant)

    @property
    def largest(self) -> Optional[MatchedPair]:
        scored = [pair for pair in self.pairs if pair.delta is not None]
        return max(scored, key=lambda pair: abs(pair.delta)) if scored else None

    def table(self, limit: int = 20) -> str:
        if not self.pairs:
            return "no matched molecular pair"
        rows = self.pairs if limit <= 0 else self.pairs[: int(limit)]
        lines = [
            f"{'label':<6}{'from':<22}{'rg':<10}{'to':<22}{'rg':<10}{'dA':>9}{'dheavy':>8}"
            "  significant",
            "-" * 6 + "-" * 22 + "-" * 10 + "-" * 22 + "-" * 10 + "-" * 9 + "-" * 8
            + "  " + "-" * 11,
        ]
        for pair in rows:
            delta = "n/a" if pair.delta is None else f"{pair.delta:+.3f}"
            added = pair.added_heavy_atoms
            heavy = "n/a" if added is None else f"{added:+d}"
            lines.append(
                f"{pair.label:<6}{pair.name_a[:21]:<22}{pair.rgroup_a[:9]:<10}"
                f"{pair.name_b[:21]:<22}{pair.rgroup_b[:9]:<10}{delta:>9}{heavy:>8}"
                f"  {'yes' if pair.significant else 'no'}"
            )
        if limit > 0 and len(self.pairs) > limit:
            lines.append(f"... and {len(self.pairs) - limit} more pair(s)")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "core": self.core,
            "labels": list(self.labels),
            "n_molecules": int(self.n_molecules),
            "n_matched": int(self.n_matched),
            "n_pairs": int(self.n_pairs),
            "n_significant": int(self.n_significant),
            "n_pairs_considered": int(self.n_pairs_considered),
            "noise": float(self.noise),
            "largest": self.largest.as_dict() if self.largest else None,
            "pairs": [pair.as_dict() for pair in self.pairs],
            "notes": list(self.notes),
        }


def matched_pairs(
    molecules: Sequence[Any],
    affinities: Optional[Union[Dict[str, float], Dict[int, float]]] = None,
    *,
    core: Optional[Union[str, Any]] = None,
    generic: bool = False,
    names: Optional[Sequence[str]] = None,
    require_affinity: bool = True,
    min_delta: Optional[float] = None,
) -> MatchedPairTable:
    """Matched molecular pairs: single-point substitutions with affinity deltas.

    A pair is *matched* when the two molecules decompose onto the same core and
    differ in exactly one attachment point — every other R-group is identical,
    hydrogen included.  That is the smallest controlled experiment a series
    contains ("replace the para hydrogen with a hydroxyl"), and the only place a
    structure-activity statement can be made from a handful of molecules.

    Notes on what this does **not** establish:

    * an R-group difference is not automatically a single-atom change (``H`` to
      ``C(=O)O`` adds five heavy atoms), so a large delta is not a clean
      substituent effect;
    * the delta is only as good as the affinity.  Docked affinities carry the
      search's own noise — this project measured a 3.2 Å seed-to-seed pose swing
      and ~1 kcal/mol ranking uncertainty on a flexible system — so a delta below
      :data:`DOCKING_NOISE_KCAL` is marked ``significant=False``.  A measured
      IC50/Ki is a different measurement with a different error, and is still an
      *assay* result, not a binding free energy;
    * the pairs share a core but not a conformation, a binding mode or a
      protonation state, and nothing here checks any of those.

    ``require_affinity`` (default) keeps only pairs where both affinities are
    known; ``min_delta`` keeps only pairs whose ``|delta|`` is at least that
    value, which is how a reader looks at the changes that survive the noise.
    """
    decomposition = rgroups(
        molecules, core=core, generic=generic, names=names, affinities=affinities
    )
    matched = [row for row in decomposition.rows if row.matched]
    notes: List[str] = []
    if not matched:
        return MatchedPairTable(
            core=decomposition.core,
            labels=decomposition.labels,
            n_molecules=decomposition.n_molecules,
            notes=["no molecule of the library contains the core"],
        )
    if require_affinity:
        scored = [row for row in matched if row.affinity is not None]
        if len(scored) < 2:
            notes.append(
                "fewer than two matched molecules carry an affinity, so no delta "
                "can be computed"
            )
        matched = scored
    pairs: List[MatchedPair] = []
    considered = 0
    for first in range(len(matched)):
        for second in range(first + 1, len(matched)):
            a, b = matched[first], matched[second]
            if a.index > b.index:
                a, b = b, a
            considered += 1
            changed = [
                label
                for label in decomposition.labels
                if a.rgroups.get(label, "") != b.rgroups.get(label, "")
            ]
            if len(changed) != 1:
                continue
            label = changed[0]
            delta = None
            if a.affinity is not None and b.affinity is not None:
                delta = float(b.affinity) - float(a.affinity)
            if min_delta is not None and (delta is None or abs(delta) < float(min_delta)):
                continue
            pairs.append(
                MatchedPair(
                    core=decomposition.core,
                    label=label,
                    rgroup_a=a.rgroups.get(label, ""),
                    rgroup_b=b.rgroups.get(label, ""),
                    index_a=a.index,
                    index_b=b.index,
                    name_a=a.name,
                    name_b=b.name,
                    affinity_a=a.affinity,
                    affinity_b=b.affinity,
                    delta=delta,
                    heavy_atoms_a=_rgroup_heavy_atoms(a.rgroups.get(label, "")),
                    heavy_atoms_b=_rgroup_heavy_atoms(b.rgroups.get(label, "")),
                )
            )
    pairs.sort(
        key=lambda pair: (
            -abs(pair.delta) if pair.delta is not None else 1e9,
            pair.index_a,
            pair.index_b,
        )
    )
    if len(pairs) > 1:
        notes.append(
            "the same pair of molecules can appear more than once when the core "
            "has symmetry-equivalent attachment points; each row names the label "
            "that changed"
        )
    big = [
        pair
        for pair in pairs
        if pair.added_heavy_atoms is not None and abs(pair.added_heavy_atoms) > 2
    ]
    if big:
        notes.append(
            f"{len(big)} of {len(pairs)} pair(s) change the size of the substituent "
            "by more than two heavy atoms; a large delta there is a change of "
            "several properties at once, not a clean substituent effect"
        )
    return MatchedPairTable(
        pairs=pairs,
        core=decomposition.core,
        labels=decomposition.labels,
        n_molecules=decomposition.n_molecules,
        n_matched=len(matched),
        n_pairs_considered=considered,
        notes=notes,
    )


def _rgroup_heavy_atoms(text: str) -> Optional[int]:
    """Heavy atoms of an R-group SMILES (``"H"`` is zero, ``""`` is unknown)."""
    if not text:
        return None
    if text == "H":
        return 0
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


# ---------------------------------------------------------------------------
# Affinities from a screening run
# ---------------------------------------------------------------------------


def affinities_from_records(records: Any) -> Dict[str, float]:
    """``{ligand name: kcal/mol}`` from a screening run's result records.

    Accepts

    * a path to ``results.jsonl``/``results.csv``/``summary.csv`` written by
      :func:`odock.screen.screen_ligands` (read with
      :func:`odock.screen.read_records`, imported lazily so this module stays
      usable without the screening pipeline);
    * an iterable of result records — objects with ``name`` and ``affinity``, or
      the plain dictionaries a decoded JSON line gives;
    * an iterable of ``(name, affinity)`` pairs, or a mapping.

    When a molecule was docked against several receptors, the **best** (most
    negative) affinity is kept — a series table describes the best evidence for
    the molecule, and the count of measurements is reported by the caller's own
    record count.  Rows without a finite affinity are skipped.
    """
    out: Dict[str, float] = {}
    items: Any = records
    if isinstance(records, (str, Path)):
        from .screen import read_records

        items = read_records(records)
    elif isinstance(records, dict):
        # A mapping may be the final answer ({"name": -7.1}) or *record-shaped*
        # ({"name": ..., "affinity": ...}), which is how one decoded JSON line
        # from a results file looks.
        if "name" in records and "affinity" in records:
            items = [records]
        else:
            return _affinity_lookup(records)
    for item in items:
        name = getattr(item, "name", None)
        affinity = getattr(item, "affinity", None)
        if name is None and isinstance(item, dict):
            name, affinity = item.get("name"), item.get("affinity")
        if name is None and isinstance(item, (tuple, list)) and len(item) >= 2:
            name, affinity = item[0], item[1]
        if name is None or affinity is None:
            continue
        try:
            value = float(affinity)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        key = str(name)
        if key not in out or value < out[key]:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Writing the R-group table
# ---------------------------------------------------------------------------


def write_rgroup_csv(path: Union[str, Path], decomposition: RGroupDecomposition) -> Path:
    """Write the molecule x R-group matrix as CSV; returns the path written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header, rows = decomposition.matrix()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return target


def write_rgroup_xlsx(
    path: Union[str, Path],
    decomposition: RGroupDecomposition,
    *,
    title: str = "R-groups",
    series: Optional[Series] = None,
) -> Path:
    """Write the R-group matrix (and optionally a series sheet) as XLSX.

    ``openpyxl`` is required.  When it is missing the call degrades to a CSV
    beside the requested name — with a warning, exactly as
    :func:`odock.report.write_xlsx` does — rather than failing a report that has
    already been computed.
    """
    import warnings

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except Exception:  # pragma: no cover - openpyxl is a test dependency
        warnings.warn(
            "openpyxl is not available; the R-group table was written as CSV instead",
            UserWarning,
            stacklevel=2,
        )
        return write_rgroup_csv(target.with_suffix(".csv"), decomposition)
    header, rows = decomposition.matrix()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = (title or "R-groups")[:31]
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    for column in range(1, len(header) + 1):
        sheet.cell(row=1, column=column).font = Font(bold=True)
        width = max(
            [len(str(header[column - 1]))]
            + [len(str(row[column - 1])) for row in rows[:200]]
        )
        sheet.column_dimensions[get_column_letter(column)].width = min(40, max(8, width + 2))
    if series is not None and series.rows:
        second = workbook.create_sheet("series")
        second.append(["name", "affinity", "delta"] + list(series.labels))
        for row in series.rows:
            second.append(
                [
                    row.name,
                    "" if row.affinity is None else round(float(row.affinity), 4),
                    "" if row.delta is None else round(float(row.delta), 4),
                    *[row.rgroups.get(label, "") for label in series.labels],
                ]
            )
        for column in range(1, len(series.labels) + 4):
            second.cell(row=1, column=column).font = Font(bold=True)
    info = workbook.create_sheet("core")
    info.append(["core", decomposition.core])
    info.append(["core with attachment labels", decomposition.core_with_labels])
    info.append(["core source", decomposition.core_source])
    info.append(["molecules", decomposition.n_molecules])
    info.append(["matched", decomposition.n_matched])
    info.append(["match rate", round(decomposition.match_rate, 4)])
    for note in decomposition.notes:
        info.append(["note", note])
    workbook.save(target)
    return target


# ---------------------------------------------------------------------------
# The report section
# ---------------------------------------------------------------------------


def report_section(
    molecules: Sequence[Any],
    *,
    affinities: Optional[Union[Dict[str, float], Dict[int, float]]] = None,
    names: Optional[Sequence[str]] = None,
    core: Optional[Union[str, Any]] = None,
    generic: bool = False,
    cutoff: float = DEFAULT_CLUSTER_CUTOFF,
    top_scaffolds: int = 10,
    top_rgroups: int = 20,
    top_pairs: int = 10,
) -> str:
    """A plain-text report section: scaffolds, series, the R-group table, MMPs.

    This is the "chemistry" page of a report: how many scaffolds the library has
    and which are largest, how many series those scaffolds fall into, the
    R-group matrix for the dominant core, and the matched pairs with their
    affinity deltas.  Every number is derived from the inputs; nothing is
    inferred from an affinity that was not supplied, and the deltas carry their
    noise caveat with them.
    """
    require_rdkit()
    mols = _to_molecules(molecules)
    labels = _names_for(mols, names)
    lookup = _affinity_lookup(affinities)
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("LIGAND CHEMISTRY — scaffolds, series, R-groups, matched pairs")
    lines.append("=" * 72)
    lines.append(f"library            : {len(mols)} molecule(s)")
    if lookup:
        lines.append(f"affinities         : {len(lookup)} value(s) supplied")
    lines.append("")

    # -- scaffolds --------------------------------------------------------
    groups = scaffold_groups(mols, generic=generic, names=labels, affinities=lookup)
    n_scaffolds = sum(1 for group in groups if group.key)
    lines.append(f"1. SCAFFOLDS — {n_scaffolds} distinct Murcko scaffold(s)")
    acyclic = [group for group in groups if not group.key]
    rows = [group for group in groups if group.key][: max(0, int(top_scaffolds))]
    lines.append(f"{'count':>6}  {'best dA':>8}  scaffold")
    lines.append("-" * 6 + "  " + "-" * 8 + "  " + "-" * 50)
    for group in rows:
        best = group.best_affinity
        lines.append(
            f"{group.count:>6}  {('' if best is None else f'{best:.3f}'):>8}  {group.key}"
            f"   (e.g. {group.representative_name})"
        )
    if len([g for g in groups if g.key]) > len(rows):
        lines.append(
            f"       ... and {len([g for g in groups if g.key]) - len(rows)} more scaffold(s)"
        )
    if acyclic:
        total = sum(group.count for group in acyclic)
        lines.append(f"{total:>6}  {'':>8}  {NO_SCAFFOLD_LABEL}")
    lines.append("")

    # -- series -----------------------------------------------------------
    clustering = scaffold_clusters(
        mols, cutoff=cutoff, generic=generic, names=labels, affinities=lookup
    )
    lines.append(
        f"2. SERIES — {clustering.n_clusters} scaffold family(ies) at "
        f"{clustering.metric} >= {cutoff:.2f}"
    )
    if clustering.clusters:
        lines.append(clustering.table(limit=10))
    for note in clustering.notes:
        lines.append(f"   note: {note}")
    lines.append("")

    # -- R-groups ---------------------------------------------------------
    decomposition = rgroups(mols, core=core, generic=generic, names=labels, affinities=lookup)
    lines.append(
        f"3. R-GROUPS — core {decomposition.core} "
        f"({decomposition.core_source}); {decomposition.n_matched} of "
        f"{decomposition.n_molecules} molecule(s) matched"
    )
    lines.append(f"   attachment labels: {', '.join(decomposition.labels) or 'none'}")
    lines.append(decomposition.table(limit=top_rgroups))
    for note in decomposition.notes:
        lines.append(f"   note: {note}")
    lines.append("")

    # -- matched pairs ----------------------------------------------------
    pairs = matched_pairs(
        mols, lookup, core=core, generic=generic, names=labels
    )
    lines.append(
        f"4. MATCHED PAIRS — {pairs.n_pairs} single-point substitution(s) from "
        f"{pairs.n_matched} matched molecule(s)"
    )
    if pairs.pairs:
        lines.append(pairs.table(limit=top_pairs))
        lines.append(
            f"   {pairs.n_significant} of {pairs.n_pairs} pair(s) have |dA| >= "
            f"{DOCKING_NOISE_KCAL:.1f} kcal/mol, the docking noise this project measured;"
        )
        lines.append(
            "   a delta below that is not a structure-activity result, and a docked "
            "affinity is not a measured one."
        )
    else:
        lines.append("   no two matched molecules differ in exactly one substituent")
        if not lookup:
            lines.append("   (no affinity was supplied, so no delta could be computed)")
    lines.append("")
    return "\n".join(lines)

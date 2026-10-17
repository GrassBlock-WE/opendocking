# SPDX-License-Identifier: GPL-3.0-or-later
"""Multi-target profiling: one ligand across a panel of receptors.

The question this module answers is the one every screening campaign asks next:
*is my hit selective, or does it bind anything?*  ``odock screen`` answers "which
of my molecules fit this protein"; a panel inverts that and asks "which of these
proteins does my molecule fit", which is a different experiment with a different
failure mode.

What a profile is
-----------------
For every (ligand, receptor) pair the panel records

* the **affinity** of the best pose, with the seed spread when more than one
  seed was run,
* the **ligand efficiency** (``odock.metrics``), so a small ligand is not
  punished for being small,
* the **interaction fingerprint** of the top pose (``odock.analysis``), so a
  reader can see whether the molecule is in the same pocket in the same way,
* the **force-field agreement** of that pose set (``odock.consensus``), i.e.
  whether the cell's own ranking is trustworthy at all,
* and the **box volume** and **pocket size** that produced the number.

From the cells it derives, per ligand, a *selectivity profile*: the best target,
the gap to the second best, and — the number that matters — **how many receptors
score within a stated window of the best**.  Three receptors inside 1 kcal/mol is
not selectivity; one receptor 2 kcal/mol clear of everything else is.

The caveat is data, not prose
-----------------------------
A cross-target affinity comparison is only meaningful if the boxes are
*comparable* and the protocol is *identical*.  Both are enforced rather than
recommended:

* a panel carries **one box derivation policy** for every member
  (``reference``: each member's box is its own co-crystallised ligand ± one
  shared buffer; ``explicit``: every member carries a caller-supplied box).
  Mixing them is refused with the offending field named (:class:`PanelError`);
* the box **volume** and the **number of receptor atoms inside it** are reported
  beside every affinity, because a bigger pocket scores better for reasons that
  have nothing to do with binding, and a "selective" ligand may simply have had a
  smaller box elsewhere;
* every cell comes from the same ``screen`` protocol — same force field, same
  exhaustiveness, same pose count — because a comparison across protocols is not
  a comparison.

**A panel of crystal structures is not a selectivity assay.**  It is a
hypothesis-generating comparison against a handful of rigid, mostly unrelated
proteins, each in one conformation, each with a pocket of its own size.  Seven
structures cannot establish selectivity; they can show that a ranking is
plausible, flag a ligand that scores equally well everywhere, and quantify how
much of a "preference" survives a change of seed.  See ``docs/PANEL.md``.

Orchestration, not new physics
------------------------------
Nothing here docks, scores, fingerprints or measures efficiency itself:

* docking and the per-receptor campaigns are :mod:`odock.screen`
  (:func:`screen.screen_ligands`, one campaign per receptor column);
* the three-field agreement is :func:`odock.consensus.consensus_score`;
* the fingerprints are :func:`odock.analysis.pose_fingerprints`;
* the efficiency comes from the ``screen`` records, which compute it with
  :func:`odock.metrics.efficiency_metrics` — and this module falls back to that
  same function for a record that carries an affinity but no efficiency.

The panel adds the matrix, the selectivity arithmetic, the box diagnostics, the
refusal rules and the hit rate — and nothing else.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import analysis, consensus, metrics, screen
from .prepare import BoxSpec, box_from_ligand, prepare_receptor

__all__ = [
    "BOX_POLICIES",
    "BUNDLED_LIGANDS",
    "BUNDLED_MEMBERS",
    "BundledLigand",
    "Panel",
    "PanelCell",
    "PanelError",
    "PanelMember",
    "PanelProfile",
    "PanelProtocol",
    "PanelResult",
    "add_panel_parser",
    "box_table",
    "build_panel",
    "bundled_ligands",
    "bundled_panel",
    "derive_boxes",
    "hit_rate",
    "main",
    "matrix",
    "profile",
    "selectivity_profile",
    "split_crystal",
    "table",
    "write_csv",
    "write_json",
]

#: The name of the panel bundled with the tests and the documentation.
PANEL_NAME = "bundled-crystal-panel"

#: A receptor is "within the window" when its affinity is at most this many
#: kcal/mol worse than the ligand's best.  1.0 is the conventional "poses within
#: 1 kcal/mol" threshold, and it is also the scale on which this project's three
#: force fields disagree with each other (see ``docs/BENCHMARK.md``): a
#: difference smaller than that is not a preference the tool can see.
DEFAULT_WINDOW = 1.0

#: Every member's box is its reference ligand plus this margin, in Å.  One value
#: for the whole panel: the boxes are *not* equal, but they are derived by one
#: rule, which is the most a cross-target comparison can honestly claim.
DEFAULT_BUFFER = 8.0

#: The only two box derivation policies.  See :class:`PanelProtocol`.
BOX_POLICIES = ("reference", "explicit")

#: Below this many poses a force-field correlation is not reported: with two
#: poses the value is +1 or -1 by construction and would read as certainty.
MIN_AGREEMENT_POSES = 3

#: Where the bundled crystal structures live.
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "data"

_WATER = frozenset({"HOH", "WAT", "H2O", "DOD"})


class PanelError(ValueError):
    """A panel cannot be built or run as specified.

    Every message names the offending field, because the point of the refusal is
    that the caller changes one thing and retries — not that they read a
    traceback.
    """


# ---------------------------------------------------------------------------
# The panel: members, protocol, refusal rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelMember:
    """One receptor in a panel, with the reference ligand its box comes from.

    ``ligand`` is the residue name of the co-crystallised ligand in ``path``.
    Under ``box_policy="reference"`` it is required, because the box is derived
    from it.  Under ``box_policy="explicit"`` it may be empty and ``box`` must be
    set.

    ``box_buffer`` exists only so that a *mixed* panel is refused by name instead
    of producing a number nobody can interpret: leave it ``None`` and the
    protocol's buffer applies to every member.
    """

    pdb_id: str
    ligand: str = ""
    protein: str = ""
    family: str = ""
    path: Optional[Path] = None
    box: Optional[BoxSpec] = None
    box_buffer: Optional[float] = None

    @property
    def label(self) -> str:
        return self.pdb_id

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "ligand": self.ligand,
            "protein": self.protein,
            "family": self.family,
            "path": str(self.path) if self.path else None,
            "box": (
                {
                    "center": list(self.box.center),
                    "size": list(self.box.size),
                    "spacing": self.box.spacing,
                }
                if self.box is not None
                else None
            ),
            "box_buffer": self.box_buffer,
        }


@dataclass(frozen=True)
class PanelProtocol:
    """The one protocol every cell of a panel is produced under.

    These are the settings that must not vary across a panel, or the affinities
    are not comparable.  They are passed through to
    :class:`odock.screen.ScreenConfig` unchanged.
    """

    box_policy: str = "reference"
    box_buffer: float = DEFAULT_BUFFER
    spacing: float = 0.375
    scoring: str = "vina"
    exhaustiveness: int = 1
    num_poses: int = 9
    min_rmsd: float = 1.0
    energy_range: float = 3.0
    search: Optional[str] = None
    #: Refinement of the top poses after the search.  Off by default: a panel is
    #: a screening comparison across receptors, and the per-pose BFGS pass costs
    #: time without changing the ranking it is used for.  Set ``refine=True``
    #: when one cell's absolute affinity is the point of the run.
    refine: bool = False
    filters: bool = False
    optimize: bool = False
    seeds: Tuple[int, ...] = (42,)
    agreement: bool = True
    fingerprints: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "box_policy": self.box_policy,
            "box_buffer": self.box_buffer,
            "spacing": self.spacing,
            "scoring": self.scoring,
            "exhaustiveness": self.exhaustiveness,
            "num_poses": self.num_poses,
            "min_rmsd": self.min_rmsd,
            "energy_range": self.energy_range,
            "search": self.search,
            "refine": self.refine,
            "filters": self.filters,
            "optimize": self.optimize,
            "seeds": list(self.seeds),
            "agreement": self.agreement,
            "fingerprints": self.fingerprints,
        }


@dataclass(frozen=True)
class Panel:
    """A set of receptors under one protocol, plus the window used to report.

    Construct it with :func:`build_panel`, which validates; :meth:`validate` is
    public so a caller that assembles members itself can ask for the same
    refusals.
    """

    members: Tuple[PanelMember, ...]
    protocol: PanelProtocol = field(default_factory=PanelProtocol)
    name: str = PANEL_NAME
    window: float = DEFAULT_WINDOW

    def validate(self) -> None:
        """Raise :class:`PanelError` naming the field when the panel is unsound."""
        protocol = self.protocol
        if not self.members:
            raise PanelError(
                "a panel needs at least one receptor, and this one is empty "
                "(field: members)"
            )
        if protocol.box_policy not in BOX_POLICIES:
            raise PanelError(
                f"unknown box policy {protocol.box_policy!r}; a panel derives "
                "every box the same way, so it must be one of "
                f"{', '.join(repr(p) for p in BOX_POLICIES)} (field: box_policy)"
            )
        if not math.isfinite(self.window) or self.window <= 0:
            raise PanelError(
                "the selectivity window must be a positive number of kcal/mol, "
                f"not {self.window!r} (field: window)"
            )
        if not protocol.seeds:
            raise PanelError("the protocol needs at least one seed (field: seeds)")
        if protocol.exhaustiveness < 1:
            raise PanelError(
                f"exhaustiveness must be at least 1, not {protocol.exhaustiveness} "
                "(field: exhaustiveness)"
            )

        seen: Dict[str, int] = {}
        for member in self.members:
            if not member.pdb_id:
                raise PanelError("a panel member has no pdb_id (field: pdb_id)")
            key = member.pdb_id.upper()
            if key in seen:
                raise PanelError(
                    f"the panel lists {member.pdb_id!r} twice; a receptor appears "
                    "once per panel (field: pdb_id)"
                )
            seen[key] = 1

        if protocol.box_policy == "reference":
            if not math.isfinite(protocol.box_buffer) or protocol.box_buffer <= 0:
                raise PanelError(
                    f"the box buffer must be positive, not "
                    f"{protocol.box_buffer!r} (field: box_buffer)"
                )
            for member in self.members:
                if member.box is not None:
                    raise PanelError(
                        "this panel mixes box policies (field: box_policy): the "
                        "protocol derives every box from its member's reference "
                        f"ligand ('reference'), but {member.pdb_id} carries an "
                        "explicit box. Give every member a box or none — a "
                        "comparison between a derived and a hand-placed box is "
                        "not a comparison."
                    )
                if not member.ligand:
                    raise PanelError(
                        f"{member.pdb_id} has no co-crystallised reference ligand, "
                        "so the 'reference' box policy cannot place its box "
                        f"(field: ligand). Either drop {member.pdb_id} from the "
                        "panel, or build it with box_policy='explicit' and give "
                        "every member its own box."
                    )
                if (
                    member.box_buffer is not None
                    and abs(member.box_buffer - protocol.box_buffer) > 1e-9
                ):
                    raise PanelError(
                        "the panel mixes box buffers (field: box_buffer): "
                        f"{member.pdb_id} declares {member.box_buffer} Å but the "
                        f"shared protocol says {protocol.box_buffer} Å. A "
                        "cross-target comparison needs one buffer everywhere."
                    )
        else:  # explicit
            for member in self.members:
                if member.box is None:
                    raise PanelError(
                        "the panel uses box_policy='explicit', so every member "
                        f"needs a box, and {member.pdb_id} has none "
                        "(field: box)"
                    )

    # -- convenience --------------------------------------------------------

    @property
    def receptors(self) -> Tuple[str, ...]:
        return tuple(member.pdb_id for member in self.members)

    def member(self, pdb_id: str) -> PanelMember:
        for member in self.members:
            if member.pdb_id.upper() == str(pdb_id).upper():
                return member
        raise KeyError(f"no member {pdb_id!r} in panel {self.name!r}")

    def subset(self, pdb_ids: Iterable[str]) -> "Panel":
        """A panel with only these receptors, same protocol and window."""
        wanted = {str(name).upper() for name in pdb_ids}
        unknown = wanted - {member.pdb_id.upper() for member in self.members}
        if unknown:
            raise PanelError(
                f"unknown receptor(s): {', '.join(sorted(unknown))} "
                f"(available: {', '.join(self.receptors)})"
            )
        return Panel(
            members=tuple(m for m in self.members if m.pdb_id.upper() in wanted),
            protocol=self.protocol,
            name=self.name,
            window=self.window,
        )

    def replaced(self, **changes: Any) -> "Panel":
        """A copy with some fields replaced (``window=``, ``protocol=``, ...)."""
        data = {
            "members": self.members,
            "protocol": self.protocol,
            "name": self.name,
            "window": self.window,
        }
        data.update(changes)
        return Panel(**data)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "window": self.window,
            "protocol": self.protocol.as_dict(),
            "members": [member.as_dict() for member in self.members],
            "is_selectivity_assay": False,
            "caveat": (
                "a panel of crystal structures is a comparison, not a "
                "selectivity assay: rigid receptors, one conformation each, "
                "pockets of different size, one scoring function"
            ),
        }


# ---------------------------------------------------------------------------
# The bundled panel
# ---------------------------------------------------------------------------

#: The seven bundled structures that carry a co-crystallised reference ligand,
#: every one of them already in ``tests/data`` — a panel run needs no network.
BUNDLED_MEMBERS: Tuple[PanelMember, ...] = (
    PanelMember("3PTB", "BEN", "bovine trypsin", "serine protease"),
    PanelMember("1STP", "BTN", "streptavidin", "biotin-binding"),
    PanelMember("3ERT", "OHT", "human ERalpha ligand-binding domain", "nuclear receptor"),
    PanelMember("1M17", "AQ4", "EGFR kinase domain", "protein kinase"),
    PanelMember("1HVR", "XK2", "HIV-1 protease", "aspartic protease"),
    PanelMember("1ERE_A", "EST", "human ERalpha ligand-binding domain", "nuclear receptor"),
    PanelMember("1HXW", "RIT", "HIV-1 protease", "aspartic protease"),
)

#: An apo structure in ``tests/data``: a legitimate panel member, but *not* under
#: the default box policy, because there is no ligand to derive a box from.  It
#: exists so the refusal can be demonstrated — and tested — on real data.
APO_MEMBER = PanelMember("2PTN", "", "bovine trypsin (apo)", "serine protease")


@dataclass(frozen=True)
class BundledLigand:
    """A co-crystallised ligand used as a query, with the target it belongs to.

    ``home`` is the *family* the ligand was crystallised against, as the set of
    panel members that represent it.  Estradiol and 4-hydroxytamoxifen are both
    ERalpha ligands and both ERalpha structures are in the panel, so a hit on
    either is a hit on the right protein — but the strict question ("is the best
    target the exact structure it was crystallised in?") is answered too, and
    reported separately.
    """

    name: str
    smiles: str
    home: FrozenSet[str]
    protein: str
    #: The exact structure this ligand was taken from.  Needed because a ligand's
    #: family can hold two members (two ERalpha structures, two HIV-1 protease
    #: structures), and "which one was it crystallised in" is not a set
    #: operation: tamoxifen is from 3ERT and estradiol from 1ERE_A, while sorting
    #: the family would silently give both the same answer.
    crystal: str = ""

    @property
    def co_crystallised(self) -> str:
        """The structure this ligand was taken from."""
        if self.crystal:
            return self.crystal
        return sorted(self.home)[0] if self.home else ""


#: The seven ligands of the bundled panel.  These are the same molecules as the
#: reference ligands, so the hit rate is checkable by hand: biotin should prefer
#: streptavidin, erlotinib EGFR, benzamidine trypsin, XK263 HIV-1 protease,
#: estradiol and 4-hydroxytamoxifen the two ERalpha structures.
BUNDLED_LIGANDS: Dict[str, BundledLigand] = {
    "BEN": BundledLigand("BEN", "NC(=N)c1ccccc1", frozenset({"3PTB"}), "benzamidine"),
    "BTN": BundledLigand(
        "BTN",
        "O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21",
        frozenset({"1STP"}),
        "biotin",
    ),
    "EST": BundledLigand(
        "EST",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CC[C@@H]2O",
        frozenset({"3ERT", "1ERE_A"}),
        "estradiol",
        crystal="1ERE_A",
    ),
    "AQ4": BundledLigand(
        "AQ4",
        "COCCOc1cc2c(cc1OCCOC)ncnc2Nc3cccc(c3)C#C",
        frozenset({"1M17"}),
        "erlotinib",
    ),
    "OHT": BundledLigand(
        "OHT",
        "CC/C(=C(\\c1ccc(O)cc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
        frozenset({"3ERT", "1ERE_A"}),
        "4-hydroxytamoxifen",
        crystal="3ERT",
    ),
    "XK2": BundledLigand(
        "XK2",
        "O=C1N(Cc2ccc3ccccc3c2)[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H]"
        "(Cc2ccccc2)N1Cc1ccc2ccccc2c1",
        frozenset({"1HVR"}),
        "XK263",
    ),
    "RIT": BundledLigand(
        "RIT",
        "CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)"
        "[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)C(C)C)cs1",
        frozenset({"1HXW"}),
        "ritonavir",
    ),
}


def build_panel(
    members: Optional[Sequence[PanelMember]] = None,
    protocol: Optional[PanelProtocol] = None,
    *,
    name: Optional[str] = None,
    window: float = DEFAULT_WINDOW,
) -> Panel:
    """A validated panel.

    ``members=None`` gives the seven bundled crystal structures.  The result is
    validated immediately, so a mixed panel fails here rather than after twenty
    minutes of docking.
    """
    panel = Panel(
        members=tuple(members) if members is not None else BUNDLED_MEMBERS,
        protocol=protocol or PanelProtocol(),
        name=name or PANEL_NAME,
        window=window,
    )
    panel.validate()
    return panel


def bundled_panel(
    *,
    protocol: Optional[PanelProtocol] = None,
    window: float = DEFAULT_WINDOW,
    receptors: Optional[Iterable[str]] = None,
) -> Panel:
    """The bundled panel, optionally restricted to some receptors."""
    panel = build_panel(None, protocol, window=window)
    return panel if receptors is None else panel.subset(receptors)


def bundled_ligands(
    names: Optional[Iterable[str]] = None,
) -> Dict[str, BundledLigand]:
    """The bundled query ligands, optionally restricted to some names."""
    if names is None:
        return dict(BUNDLED_LIGANDS)
    wanted = [str(name).upper() for name in names]
    unknown = [name for name in wanted if name not in BUNDLED_LIGANDS]
    if unknown:
        raise PanelError(
            f"unknown bundled ligand(s): {', '.join(unknown)} "
            f"(available: {', '.join(BUNDLED_LIGANDS)})"
        )
    return {name: BUNDLED_LIGANDS[name] for name in wanted}


# ---------------------------------------------------------------------------
# Crystal splitting and box derivation
# ---------------------------------------------------------------------------


def split_crystal(pdb: Any, ligand_resname: str) -> Tuple[Any, Any]:
    """``(receptor_mol, ligand_mol)`` out of one crystal file.

    Waters are dropped; the named residue is the ligand and everything else
    (ions and additives included) is the receptor — which is what the site
    actually contains.

    This is deliberately small and self-contained.  ``odock.benchmark`` has an
    equivalent private helper doing the same thing for the same reason; if a
    third caller appears it should be promoted to a public helper in
    ``odock.prepare`` and both modules pointed at it, rather than copied again.
    """
    from rdkit import Chem

    path = Path(pdb)
    mol = Chem.MolFromPDBFile(
        str(path), removeHs=False, sanitize=False, proximityBonding=True
    )
    if mol is None:
        raise PanelError(
            f"RDKit could not read {path}; a panel member must be a PDB file with "
            "parseable coordinates (field: path)"
        )
    keep_receptor: List[int] = []
    keep_ligand: List[int] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip().upper() if info is not None else ""
        if name in _WATER:
            continue
        target = keep_ligand if name == str(ligand_resname).strip().upper() else keep_receptor
        target.append(atom.GetIdx())
    if not keep_ligand:
        raise PanelError(
            f"no {ligand_resname!r} residue in {path.name}; the panel member's "
            "reference ligand name does not match the structure (field: ligand)"
        )

    def subset(indices: Sequence[int]):
        editable = Chem.RWMol(mol)
        for index in sorted(set(range(mol.GetNumAtoms())) - set(indices), reverse=True):
            editable.RemoveAtom(index)
        out = editable.GetMol()
        try:
            Chem.SanitizeMol(out)
        except Exception:  # pragma: no cover - crystal geometry is imperfect
            pass
        return out

    return subset(keep_receptor), subset(keep_ligand)


@dataclass
class BoxDiagnostics:
    """The box that produced a receptor's column, and what is inside it.

    A cross-target comparison lives or dies on this table: two receptors whose
    boxes differ by a third in volume are not comparing like with like, and a
    reader has to be able to see that without asking.
    """

    pdb_id: str
    reference: str
    box: BoxSpec
    volume: float
    receptor_atoms: int
    grid_points: int
    source: str  # "reference" or "explicit"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pdb_id": self.pdb_id,
            "reference": self.reference,
            "center": list(self.box.center),
            "size": list(self.box.size),
            "volume": round(self.volume, 1),
            "receptor_atoms": self.receptor_atoms,
            "grid_points": self.grid_points,
            "source": self.source,
        }


def derive_boxes(
    panel: Panel,
    *,
    data_dir: Optional[Any] = None,
    receptor_texts: Optional[Mapping[str, str]] = None,
) -> Dict[str, BoxDiagnostics]:
    """The box for every member, by the panel's one policy.

    ``receptor_texts`` (prepared PDBQT text per receptor) is only needed for the
    ``receptor_atoms`` column; the boxes themselves come from the crystal.
    """
    panel.validate()
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    out: Dict[str, BoxDiagnostics] = {}
    for member in panel.members:
        if panel.protocol.box_policy == "explicit":
            box = member.box
            assert box is not None  # validate() guarantees it
            source = "explicit"
        else:
            path = Path(member.path) if member.path else base / f"{member.pdb_id}.pdb"
            if not path.exists():
                raise PanelError(
                    f"no structure for {member.pdb_id} at {path} (field: path); "
                    "pass data_dir= or set member.path"
                )
            _receptor_mol, reference_mol = split_crystal(path, member.ligand)
            box = box_from_ligand(
                reference_mol,
                buffer=panel.protocol.box_buffer,
                spacing=panel.protocol.spacing,
            )
            source = "reference"
        volume = float(box.size[0]) * float(box.size[1]) * float(box.size[2])
        text = (receptor_texts or {}).get(member.pdb_id, "")
        atoms = screen.receptor_atoms_in_box(text, box) if text else 0
        points = screen.grid_points(box, panel.protocol.spacing)
        out[member.pdb_id] = BoxDiagnostics(
            pdb_id=member.pdb_id,
            reference=member.ligand,
            box=box,
            volume=volume,
            receptor_atoms=atoms,
            grid_points=points,
            source=source,
        )
    return out


# ---------------------------------------------------------------------------
# Cells and the selectivity arithmetic
# ---------------------------------------------------------------------------


@dataclass
class PanelCell:
    """One (ligand, receptor) measurement, aggregated over the seeds that ran."""

    ligand: str
    receptor: str
    affinity: Optional[float] = None
    spread: Optional[float] = None
    n_seeds: int = 1
    seeds: Tuple[int, ...] = ()
    efficiency: Optional[float] = None
    heavy_atoms: Optional[int] = None
    torsions: Optional[int] = None
    n_poses: int = 0
    n_interactions: int = 0
    key_residues: Tuple[str, ...] = ()
    agreement: Optional[float] = None
    box_volume: float = 0.0
    pocket_atoms: int = 0
    in_box: bool = True
    status: str = "ok"
    error: Optional[str] = None
    fingerprint: Tuple[str, ...] = ()
    pose_file: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.affinity is not None

    @property
    def score(self) -> float:
        """The number the ranking uses: the mean affinity over seeds."""
        return float(self.affinity) if self.affinity is not None else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ligand": self.ligand,
            "receptor": self.receptor,
            "affinity": _round(self.affinity),
            "spread": _round(self.spread),
            "n_seeds": self.n_seeds,
            "seeds": list(self.seeds),
            "efficiency": _round(self.efficiency, 4),
            "heavy_atoms": self.heavy_atoms,
            "torsions": self.torsions,
            "poses": self.n_poses,
            "interactions": self.n_interactions,
            "key_residues": list(self.key_residues),
            "agreement": _round(self.agreement),
            "box_volume": _round(self.box_volume, 1),
            "pocket_atoms": self.pocket_atoms,
            "in_box": self.in_box,
            "status": self.status,
            "error": self.error,
            "fingerprint": list(self.fingerprint),
            "pose_file": self.pose_file,
        }


@dataclass
class PanelProfile:
    """One ligand's result across the panel: the selectivity profile."""

    ligand: str
    cells: Tuple[PanelCell, ...]
    window: float
    home: FrozenSet[str] = frozenset()
    protein: str = ""

    @property
    def ranked(self) -> Tuple[PanelCell, ...]:
        """Cells that produced a number, best first.

        Ties are broken by receptor id so the table is stable across runs: the
        affinity is a mean over seeds, and two receptors tie on the rounded value
        often enough that an unstable order would look like noise.
        """
        usable = [cell for cell in self.cells if cell.ok]
        return tuple(sorted(usable, key=lambda c: (c.score, c.receptor.upper())))

    @property
    def failures(self) -> Tuple[PanelCell, ...]:
        return tuple(cell for cell in self.cells if not cell.ok)

    @property
    def best(self) -> Optional[PanelCell]:
        ranked = self.ranked
        return ranked[0] if ranked else None

    @property
    def second(self) -> Optional[PanelCell]:
        ranked = self.ranked
        return ranked[1] if len(ranked) > 1 else None

    @property
    def gap(self) -> Optional[float]:
        """How much better the best target is than the runner-up, in kcal/mol.

        ``None`` when fewer than two receptors produced a number — which is what
        a single-receptor panel is, and it has no selectivity to report.
        """
        best, second = self.best, self.second
        if best is None or second is None:
            return None
        return second.score - best.score

    @property
    def within_window(self) -> Tuple[PanelCell, ...]:
        """Receptors within ``window`` kcal/mol of the best, best first."""
        best = self.best
        if best is None:
            return ()
        cutoff = best.score + self.window
        return tuple(cell for cell in self.ranked if cell.score <= cutoff)

    @property
    def best_is_home(self) -> Optional[bool]:
        """Is the best target in the ligand's own family?  ``None`` when unknown."""
        best = self.best
        if best is None or not self.home:
            return None
        return best.receptor.upper() in {name.upper() for name in self.home}

    @property
    def co_crystallised(self) -> str:
        return sorted(self.home)[0] if self.home else ""

    @property
    def home_cell(self) -> Optional[PanelCell]:
        """The cell for the structure this ligand was crystallised in."""
        if not self.home:
            return None
        co = self.co_crystallised.upper()
        for cell in self.cells:
            if cell.receptor.upper() == co:
                return cell
        return None

    @property
    def home_rank(self) -> Optional[int]:
        """1-based rank of the co-crystallised receptor among the usable cells."""
        cell = self.home_cell
        if cell is None or not cell.ok:
            return None
        for position, ranked in enumerate(self.ranked, start=1):
            if ranked.receptor.upper() == cell.receptor.upper():
                return position
        return None

    @property
    def single_seed(self) -> bool:
        return all(cell.n_seeds <= 1 for cell in self.cells)

    def outcome(self) -> str:
        """A one-line reading of the profile, used by the report and the doc."""
        ranked = self.ranked
        if not ranked:
            return "no receptor produced a pose"
        best = ranked[0]
        gap = self.gap
        if gap is None:
            return f"only one receptor scored ({best.receptor})"
        count = len(self.within_window)
        if count == 1 and gap >= self.window:
            return f"selective for {best.receptor} by {gap:.2f} kcal/mol"
        if count >= 2:
            return (
                f"{count} of {len(ranked)} receptors within {self.window:g} "
                f"kcal/mol of {best.receptor}: a preference, not selectivity"
            )
        return f"best {best.receptor}, gap {gap:.2f} kcal/mol"

    def as_dict(self) -> Dict[str, Any]:
        best, second = self.best, self.second
        return {
            "ligand": self.ligand,
            "protein": self.protein,
            "home": sorted(self.home),
            "co_crystallised": self.co_crystallised,
            "window": self.window,
            "best": best.receptor if best else None,
            "best_affinity": _round(best.affinity) if best else None,
            "second": second.receptor if second else None,
            "second_affinity": _round(second.affinity) if second else None,
            "gap": _round(self.gap),
            "within_window": [cell.receptor for cell in self.within_window],
            "n_within_window": len(self.within_window),
            "n_scored": len(self.ranked),
            "best_is_home": self.best_is_home,
            "home_rank": self.home_rank,
            "single_seed": self.single_seed,
            "outcome": self.outcome(),
            "cells": [cell.as_dict() for cell in self.cells],
        }


def selectivity_profile(
    ligand: str,
    cells: Sequence[PanelCell],
    *,
    window: float = DEFAULT_WINDOW,
    home: Optional[Iterable[str]] = None,
    protein: str = "",
) -> PanelProfile:
    """Assemble a profile from cells that already exist.

    Public because it is the piece worth testing without docking: the ranking,
    the gap, the window count and the home check are arithmetic, and a test can
    hand it three cells and assert the answer.
    """
    return PanelProfile(
        ligand=ligand,
        cells=tuple(cells),
        window=float(window),
        home=frozenset(str(name).upper() for name in (home or ())),
        protein=protein,
    )


# ---------------------------------------------------------------------------
# Running a panel
# ---------------------------------------------------------------------------


@dataclass
class PanelResult:
    """The matrix, the profiles and the diagnostics of one panel run."""

    panel: Panel
    profiles: Tuple[PanelProfile, ...]
    boxes: Dict[str, BoxDiagnostics]
    cells: Tuple[PanelCell, ...]
    elapsed: float = 0.0
    data_dir: Optional[str] = None

    @property
    def ligands(self) -> Tuple[str, ...]:
        return tuple(profile.ligand for profile in self.profiles)

    @property
    def receptors(self) -> Tuple[str, ...]:
        return self.panel.receptors

    @property
    def window(self) -> float:
        return self.panel.window

    @property
    def single_seed(self) -> bool:
        return all(cell.n_seeds <= 1 for cell in self.cells)

    @property
    def seeds(self) -> Tuple[int, ...]:
        seen: List[int] = []
        for cell in self.cells:
            for seed in cell.seeds:
                if seed not in seen:
                    seen.append(seed)
        return tuple(seen)

    def profile(self, ligand: str) -> PanelProfile:
        for profile in self.profiles:
            if profile.ligand == ligand:
                return profile
        raise KeyError(f"no ligand {ligand!r} in this panel result")

    def cell(self, ligand: str, receptor: str) -> PanelCell:
        for cell in self.profile(ligand).cells:
            if cell.receptor.upper() == str(receptor).upper():
                return cell
        raise KeyError(f"no cell for {ligand!r} in {receptor!r}")

    # -- reports -----------------------------------------------------------

    def matrix(self) -> str:
        """The affinity matrix: ligands down, receptors across."""
        return matrix(self)

    def table(self) -> str:
        """The matrix with each ligand's selectivity profile beside it."""
        return table(self)

    def box_table(self) -> str:
        return box_table(self)

    def hit_rate(self) -> Dict[str, Any]:
        return hit_rate(self.profiles, window=self.window)

    def fingerprint_similarity(self, ligand: str) -> List[List[float]]:
        """Pairwise Tanimoto between the top-pose fingerprints across receptors."""
        cells = [cell for cell in self.profile(ligand).cells if cell.fingerprint]
        return [[_jaccard(a.fingerprint, b.fingerprint) for b in cells] for a in cells]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "panel": self.panel.as_dict(),
            "elapsed": round(self.elapsed, 3),
            "data_dir": self.data_dir,
            "single_seed": self.single_seed,
            "seeds": list(self.seeds),
            "boxes": {name: diag.as_dict() for name, diag in self.boxes.items()},
            "profiles": [profile.as_dict() for profile in self.profiles],
            "hit_rate": self.hit_rate(),
            "caveat": (
                "affinities from different receptors are comparable only as far "
                "as the shared protocol and the box table allow; a panel of "
                "crystal structures is not a selectivity assay"
            ),
        }

    def write_json(self, path: Any) -> Path:
        return write_json(path, self)

    def write_csv(self, path: Any) -> Path:
        return write_csv(path, self)


def _round(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Tanimoto over two label sets — the same metric ``analysis`` uses."""
    left, right = set(a), set(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _ligand_lines(ligands: Mapping[str, str]) -> str:
    """``SMILES name`` lines, which is the .smi convention ``screen`` reads."""
    lines = []
    for name, smiles in ligands.items():
        text = " ".join(str(smiles).split())
        if not text:
            raise PanelError(f"ligand {name!r} has an empty SMILES (field: ligands)")
        if " " in str(name) or not str(name).strip():
            raise PanelError(
                f"ligand name {name!r} is not a single token; names are the .smi "
                "title (field: ligands)"
            )
        lines.append(f"{text} {name}")
    return "\n".join(lines) + "\n"


def read_ligands(path: Any) -> Dict[str, str]:
    """Read ``SMILES name`` lines (or an SDF) into a name -> SMILES mapping."""
    source = Path(path)
    if not source.exists():
        raise PanelError(f"no ligand file at {source} (field: ligands)")
    if source.suffix.lower() in (".sdf", ".sd", ".mol"):
        from rdkit import Chem

        # Explicit hydrogens are dropped on purpose: this returns a library in
        # SMILES form, and `screen` re-adds hydrogens when it prepares each
        # molecule. Keeping them would put `[H]` in every line of the .smi.
        supplier = Chem.SDMolSupplier(str(source), removeHs=True)
        out: Dict[str, str] = {}
        for index, mol in enumerate(supplier):
            if mol is None:
                continue
            name = (mol.GetProp("_Name") if mol.HasProp("_Name") else "").strip()
            out[name or f"mol{index + 1}"] = Chem.MolToSmiles(mol)
        if not out:
            raise PanelError(f"no molecule could be read from {source} (field: ligands)")
        return out
    out = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise PanelError(
                f"{source.name}:{number}: expected 'SMILES name', got {line!r} "
                "(field: ligands)"
            )
        out[parts[1].strip()] = parts[0]
    if not out:
        raise PanelError(f"no ligand in {source} (field: ligands)")
    return out


def _fingerprint_labels(receptor_text: str, pose_text: str) -> Tuple[str, ...]:
    """The interaction fingerprint of the top pose, as ``RESIDUE:KIND`` labels."""
    models = consensus.pdbqt_models(pose_text)
    if not models:
        return ()
    try:
        fingerprint_set = analysis.pose_fingerprints(receptor_text, [models[0]])
        labels = list(fingerprint_set.labels())
        bits = fingerprint_set.bits
        row = bits[0] if getattr(bits, "ndim", 1) > 1 else bits
    except Exception:  # pragma: no cover - perception is best effort per cell
        return ()
    return tuple(sorted(str(label) for label, flag in zip(labels, row) if int(flag)))


def _agreement(
    pose_text: str,
    receptor_text: str,
    box: BoxSpec,
    scorings: Sequence[str] = ("vina", "vinardo", "ad4"),
) -> Optional[float]:
    """Mean pairwise Spearman agreement of the three force fields on this cell.

    This is the cell's own confidence: if the fields disagree about the ordering
    of the poses they produced, a 0.2 kcal/mol difference between two receptors
    is not something to read anything into.  ``None`` below
    :data:`MIN_AGREEMENT_POSES` poses, where the correlation is +1 or -1 by
    construction and would read as certainty.
    """
    if len(consensus.pdbqt_models(pose_text)) < MIN_AGREEMENT_POSES:
        return None
    try:
        result = consensus.consensus_score(pose_text, receptor_text, box, scorings)
    except Exception:  # pragma: no cover - AD4 cannot always type a receptor
        return None
    value = getattr(result, "agreement", None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    return number if math.isfinite(number) else None


def prepare_members(
    panel: Panel,
    *,
    outdir: Any,
    data_dir: Optional[Any] = None,
) -> Dict[str, str]:
    """Prepare every member's receptor and return the PDBQT text by receptor.

    **The co-crystallised ligand is stripped from its own receptor.** A panel
    member is a *holo* structure — that is why it was chosen, its ligand defines
    the box — and leaving that ligand in the receptor blocks the very site the
    query ligands are docked into. The first run of this module did exactly that
    and produced a matrix of clash penalties (a 50-atom ligand scored
    +17.7 kcal/mol into the estradiol site of 1ERE_A) that looked like
    selectivity. ``prepare_receptor`` documents the failure mode and its ``strip``
    parameter is the remedy; this function is where the panel applies it.

    Everything else is kept: ions, cofactors and modified residues are part of
    the site, and waters are dropped by ``prepare_receptor``'s default.
    """
    base = Path(outdir)
    receptor_dir = base / "receptors"
    receptor_dir.mkdir(parents=True, exist_ok=True)
    crystal_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    texts: Dict[str, str] = {}
    for member in panel.members:
        path = Path(member.path) if member.path else crystal_dir / f"{member.pdb_id}.pdb"
        if not path.exists():
            raise PanelError(
                f"no structure for {member.pdb_id} at {path} (field: path); "
                "pass data_dir= or set member.path"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mol, text, _report = prepare_receptor(
                path, strip=(member.ligand,) if member.ligand else None
            )
        (receptor_dir / f"{member.pdb_id}.pdbqt").write_text(text, encoding="utf-8")
        texts[member.pdb_id] = text
    return texts


def profile(
    ligands: Mapping[str, str] | Any,
    panel: Optional[Panel] = None,
    outdir: Any = None,
    *,
    data_dir: Optional[Any] = None,
    window: Optional[float] = None,
    seeds: Optional[Sequence[int]] = None,
    progress: bool = True,
    homes: Optional[Mapping[str, Iterable[str]]] = None,
    proteins: Optional[Mapping[str, str]] = None,
) -> PanelResult:
    """Profile every ligand across every receptor of ``panel``.

    One docking campaign per receptor (``screen``), one seed at a time, so the
    whole panel is produced under one protocol by one process.  ``outdir``
    defaults to a directory named after the panel in the current directory.

    ``homes`` maps a ligand name to the receptor(s) it is expected to prefer,
    which is what makes the hit rate computable; without it every profile has an
    empty home and the hit rate is ``None``.
    """
    panel = panel or build_panel()
    if window is not None:
        panel = panel.replaced(window=float(window))
    panel.validate()

    if isinstance(ligands, (str, Path)):
        ligand_map = read_ligands(ligands)
    else:
        ligand_map = {str(key): str(value) for key, value in dict(ligands).items()}
    if not ligand_map:
        raise PanelError("no ligands to profile (field: ligands)")

    seeds = tuple(int(seed) for seed in seeds) if seeds is not None else tuple(panel.protocol.seeds)
    if not seeds:
        raise PanelError("at least one seed is required (field: seeds)")
    protocol = panel.protocol
    base = Path(outdir) if outdir is not None else Path.cwd() / panel.name
    base.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ligands_file = base / "ligands.smi"
    ligands_file.write_text(_ligand_lines(ligand_map), encoding="utf-8")

    # -- prepare every receptor once, then derive its box -------------------
    receptor_texts = prepare_members(panel, outdir=base, data_dir=data_dir)
    receptor_dir = base / "receptors"

    boxes = derive_boxes(panel, data_dir=data_dir, receptor_texts=receptor_texts)

    # -- one campaign per (seed, receptor) ----------------------------------
    per_seed: List[PanelCell] = []
    for seed in seeds:
        for member in panel.members:
            diagnostics = boxes[member.pdb_id]
            run_dir = base / "runs" / f"seed{seed}" / member.pdb_id
            receptor_path = receptor_dir / f"{member.pdb_id}.pdbqt"
            if not receptor_path.exists():
                per_seed.extend(
                    PanelCell(
                        ligand=name,
                        receptor=member.pdb_id,
                        status="failed",
                        error="receptor could not be prepared",
                        seeds=(seed,),
                        box_volume=diagnostics.volume,
                        pocket_atoms=diagnostics.receptor_atoms,
                    )
                    for name in ligand_map
                )
                continue
            config = screen.ScreenConfig(
                receptors=[str(receptor_path)],
                inputs=[str(ligands_file)],
                box=diagnostics.box,
                outdir=str(run_dir),
                scoring=protocol.scoring,
                exhaustiveness=protocol.exhaustiveness,
                num_poses=protocol.num_poses,
                min_rmsd=protocol.min_rmsd,
                energy_range=protocol.energy_range,
                search=protocol.search,
                seed=int(seed),
                jobs=1,
                filters=protocol.filters,
                optimize=protocol.optimize,
                refine=protocol.refine,
                progress=False,
                interactions=True,
                write_poses=True,
                resume=False,
            )
            captured = io.StringIO()
            try:
                if progress:
                    summary = screen.screen_ligands(config)
                else:
                    with contextlib.redirect_stdout(captured):
                        summary = screen.screen_ligands(config)
            except screen.ScreenError as exc:
                per_seed.extend(
                    PanelCell(
                        ligand=name,
                        receptor=member.pdb_id,
                        status="failed",
                        error=f"campaign failed: {exc}",
                        seeds=(seed,),
                        box_volume=diagnostics.volume,
                        pocket_atoms=diagnostics.receptor_atoms,
                    )
                    for name in ligand_map
                )
                continue
            if progress:
                print(
                    f"  {member.pdb_id} seed {seed}: {len(summary.records)} cell(s)",
                    file=sys.stderr,
                )
            for record in summary.records:
                per_seed.append(
                    _cell_from_record(
                        record,
                        member=member,
                        diagnostics=diagnostics,
                        receptor_text=receptor_texts.get(member.pdb_id, ""),
                        run_dir=run_dir,
                        seed=int(seed),
                        protocol=protocol,
                    )
                )

    cells = _aggregate(per_seed, ligand_map, panel)
    profiles: List[PanelProfile] = []
    for name in ligand_map:
        bundled = BUNDLED_LIGANDS.get(name)
        if homes is not None:
            home: Iterable[str] = homes.get(name, ())
        else:
            home = bundled.home if bundled else ()
        if proteins is not None:
            protein = proteins.get(name, "")
        else:
            protein = bundled.protein if bundled else ""
        profiles.append(
            selectivity_profile(
                name,
                [cell for cell in cells if cell.ligand == name],
                window=panel.window,
                home=home,
                protein=protein,
            )
        )
    return PanelResult(
        panel=panel,
        profiles=tuple(profiles),
        boxes=boxes,
        cells=tuple(cells),
        elapsed=time.perf_counter() - started,
        data_dir=str(data_dir) if data_dir is not None else None,
    )


def _cell_from_record(
    record: Any,
    *,
    member: PanelMember,
    diagnostics: BoxDiagnostics,
    receptor_text: str,
    run_dir: Path,
    seed: int,
    protocol: PanelProtocol,
) -> PanelCell:
    """One ``LigandRecord`` from ``screen`` -> one raw panel cell."""
    status = str(getattr(record, "status", "failed"))
    affinity = getattr(record, "affinity", None)
    ok = status == "ok" and affinity is not None
    name = str(getattr(record, "name", "") or getattr(record, "ligand", ""))
    raw_residues = getattr(record, "key_residues", ()) or ()
    if isinstance(raw_residues, str):
        residues = tuple(
            part.strip() for part in raw_residues.split(",") if part.strip()
        )
    else:
        residues = tuple(str(part) for part in raw_residues)
    efficiency = getattr(record, "ligand_efficiency", None)
    if ok and efficiency is None:
        # The record already carries what `metrics` computes; this is the same
        # function called again only when a campaign did not fill it in.
        try:
            efficiency = metrics.efficiency_metrics(
                float(affinity),
                heavy_atoms=getattr(record, "n_heavy", None),
                molecular_weight=getattr(record, "molecular_weight", None),
                num_torsions=getattr(record, "n_torsions", None),
            ).ligand_efficiency
        except Exception:  # pragma: no cover - defensive
            efficiency = None
    cell = PanelCell(
        ligand=name,
        receptor=member.pdb_id,
        affinity=float(affinity) if ok else None,
        n_seeds=1,
        seeds=(seed,),
        efficiency=float(efficiency) if efficiency is not None else None,
        heavy_atoms=getattr(record, "n_heavy", None),
        torsions=getattr(record, "n_torsions", None),
        n_poses=len(getattr(record, "poses", ()) or ()),
        n_interactions=int(getattr(record, "n_interactions", 0) or 0),
        key_residues=residues,
        box_volume=diagnostics.volume,
        pocket_atoms=diagnostics.receptor_atoms,
        in_box=bool(getattr(record, "in_box", True)),
        status="ok" if ok else "failed",
        error=None if ok else (getattr(record, "error", None) or "no pose"),
    )
    relative = getattr(record, "pose_file", "") or ""
    if relative:
        pose_path = Path(relative)
        if not pose_path.is_absolute():
            pose_path = Path(run_dir) / pose_path
        cell.pose_file = str(pose_path)
    if ok and cell.pose_file:
        try:
            pose_text = Path(cell.pose_file).read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - screen writes the file
            pose_text = ""
        if pose_text:
            if protocol.fingerprints:
                cell.fingerprint = _fingerprint_labels(receptor_text, pose_text)
            if protocol.agreement:
                cell.agreement = _agreement(pose_text, receptor_text, diagnostics.box)
    return cell


def _aggregate(
    per_seed: Sequence[PanelCell],
    ligands: Mapping[str, str],
    panel: Panel,
) -> List[PanelCell]:
    """One cell per (ligand, receptor), with the mean affinity and its spread.

    The ranking uses the **mean** over seeds, not the best: a best-of-seeds
    ranking is optimistic by construction and hides exactly the instability the
    spread column exists to show.
    """
    grouped: Dict[Tuple[str, str], List[PanelCell]] = {}
    for cell in per_seed:
        grouped.setdefault((cell.ligand, cell.receptor), []).append(cell)
    out: List[PanelCell] = []
    for member in panel.members:
        for name in ligands:
            group = grouped.get((name, member.pdb_id), [])
            if not group:
                continue
            good = [cell for cell in group if cell.ok]
            seeds = tuple(sorted({seed for cell in group for seed in cell.seeds}))
            if not good:
                first = group[0]
                out.append(
                    PanelCell(
                        ligand=name,
                        receptor=member.pdb_id,
                        n_seeds=len(seeds),
                        seeds=seeds,
                        box_volume=first.box_volume,
                        pocket_atoms=first.pocket_atoms,
                        status="failed",
                        error=first.error,
                    )
                )
                continue
            affinities = [cell.score for cell in good]
            efficiencies = [cell.efficiency for cell in good if cell.efficiency is not None]
            agreements = [cell.agreement for cell in good if cell.agreement is not None]
            base = good[0]
            out.append(
                PanelCell(
                    ligand=name,
                    receptor=member.pdb_id,
                    affinity=statistics.fmean(affinities),
                    spread=(
                        max(affinities) - min(affinities)
                        if len(affinities) > 1
                        else None
                    ),
                    n_seeds=len(seeds),
                    seeds=seeds,
                    efficiency=statistics.fmean(efficiencies) if efficiencies else None,
                    heavy_atoms=base.heavy_atoms,
                    torsions=base.torsions,
                    n_poses=base.n_poses,
                    n_interactions=base.n_interactions,
                    key_residues=base.key_residues,
                    agreement=statistics.fmean(agreements) if agreements else None,
                    box_volume=base.box_volume,
                    pocket_atoms=base.pocket_atoms,
                    in_box=base.in_box,
                    status="ok",
                    fingerprint=base.fingerprint,
                    pose_file=base.pose_file,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cell_text(cell: Optional[PanelCell], width: int = 12) -> str:
    if cell is None or not cell.ok:
        return f"{'--':>{width}}"
    text = f"{cell.affinity:+.2f}"
    if cell.spread is not None and math.isfinite(cell.spread) and cell.spread > 0:
        text = f"{cell.affinity:+.2f}±{cell.spread:.2f}"
    return f"{text:>{width}}"


def matrix(result: PanelResult) -> str:
    """The affinity matrix, ligands down and receptors across."""
    receptors = result.receptors
    header = f"{'ligand':6s} " + " ".join(f"{name:>12s}" for name in receptors)
    lines = [header, "-" * len(header)]
    for profile in result.profiles:
        row = [f"{profile.ligand:<6s}"]
        for receptor in receptors:
            try:
                cell = result.cell(profile.ligand, receptor)
            except KeyError:
                cell = None
            row.append(_cell_text(cell))
        lines.append(" ".join(row))
    lines.append("")
    lines.append(
        "kcal/mol, the mean best pose of each campaign (± is the spread across "
        "seeds). More negative is better, and across receptors it is comparable "
        "only as far as the box table below allows."
    )
    return "\n".join(lines)


def table(result: PanelResult) -> str:
    """The matrix plus each ligand's selectivity profile."""
    header = (
        f"{'ligand':6s} {'best':8s} {'affinity':>9s} {'second':8s} {'gap':>7s} "
        f"{'<=':>3s} {'home':>5s} {'rank':>4s}  reading"
    )
    lines = [header, "-" * len(header)]
    for profile in result.profiles:
        best = profile.best
        gap = profile.gap
        home = (
            "yes"
            if profile.best_is_home
            else ("no" if profile.best_is_home is False else "?")
        )
        lines.append(
            " ".join(
                [
                    f"{profile.ligand:<6s}",
                    f"{best.receptor if best else '--':8s}",
                    f"{(f'{best.affinity:+.2f}' if best else '--'):>9s}",
                    f"{profile.second.receptor if profile.second else '--':8s}",
                    f"{(f'{gap:+.2f}' if gap is not None else '--'):>7s}",
                    f"{len(profile.within_window):>3d}",
                    f"{home:>5s}",
                    f"{(profile.home_rank if profile.home_rank is not None else '-'):>4}",
                    "  ",
                    profile.outcome(),
                ]
            )
        )
    lines.append("")
    lines.append(
        f"'<=' counts receptors within {result.window:g} kcal/mol of the best; "
        "'home' asks whether the best target is the protein family the ligand was "
        "crystallised against; 'rank' is that protein's position (1 = best)."
    )
    if result.single_seed:
        lines.append(
            "ONE SEED: these are point estimates. This project measured a 3.17 A "
            "top-pose swing between seeds on 1M17, so a ranking without a spread "
            "is a hypothesis, not a result."
        )
    return "\n".join(lines)


def box_table(result: PanelResult) -> str:
    """The boxes that produced the columns — the table that makes the caveat data."""
    header = (
        f"{'receptor':8s} {'reference':9s} {'size (A)':>20s} {'volume':>9s} "
        f"{'atoms':>6s} {'grid pts':>9s}  source"
    )
    lines = [header, "-" * len(header)]
    for name in result.receptors:
        diagnostics = result.boxes[name]
        size = "x".join(f"{value:.1f}" for value in diagnostics.box.size)
        lines.append(
            f"{name:8s} {diagnostics.reference or '-':9s} {size:>20s} "
            f"{diagnostics.volume:>9.0f} {diagnostics.receptor_atoms:>6d} "
            f"{diagnostics.grid_points:>9d}  {diagnostics.source}"
        )
    lines.append("")
    lines.append(
        "volume is the box in cubic angstrom; 'atoms' is how many receptor atoms "
        "it contains. Two receptors whose volumes differ by a third are not "
        "comparing like with like, which is why a 'selective' ligand may simply "
        "have had a smaller box elsewhere."
    )
    return "\n".join(lines)


def hit_rate(
    profiles: Sequence[PanelProfile],
    *,
    window: float = DEFAULT_WINDOW,
) -> Dict[str, Any]:
    """How many ligands' best target is their own co-crystallised protein.

    Every miss is classified, because "the tool got it wrong" is not a useful
    sentence: a miss is a *box* effect (the winning receptor's pocket is much
    larger), a *score* effect (the home receptor is clearly behind), or
    *promiscuity* (the home receptor is within the window, so the ligand simply
    does not distinguish them).
    """
    scored = [profile for profile in profiles if profile.home and profile.best is not None]
    hits = [profile for profile in scored if profile.best_is_home]
    misses: List[Dict[str, Any]] = []
    for profile in scored:
        if profile.best_is_home:
            continue
        best = profile.best
        home = profile.home_cell
        reason, detail = _miss_reason(profile, best, home, window)
        misses.append(
            {
                "ligand": profile.ligand,
                "protein": profile.protein,
                "home": sorted(profile.home),
                "best": best.receptor if best else None,
                "best_affinity": _round(best.affinity) if best else None,
                "home_affinity": _round(home.affinity) if home is not None else None,
                "home_rank": profile.home_rank,
                "gap": _round(profile.gap),
                "reason": reason,
                "detail": detail,
            }
        )
    return {
        "n": len(scored),
        "hits": len(hits),
        "rate": (len(hits) / len(scored)) if scored else None,
        "strict_hits": sum(
            1
            for profile in scored
            if profile.best is not None
            and profile.best.receptor.upper() == profile.co_crystallised.upper()
        ),
        "misses": misses,
        "window": window,
        "note": (
            "'home' is the ligand's target family: two ERalpha structures and two "
            "HIV-1 protease structures are in the panel, so a hit on either is a "
            "hit on the right protein; 'strict_hits' counts only the exact "
            "co-crystallised structure"
        ),
    }


def _miss_reason(
    profile: PanelProfile,
    best: Optional[PanelCell],
    home: Optional[PanelCell],
    window: float,
) -> Tuple[str, str]:
    """Why a ligand failed to prefer its own protein, in one of three buckets."""
    if best is None:
        return "failed", "no receptor produced a pose"
    if home is None:
        return "absent", "the ligand's own protein is not in this panel"
    if not home.ok:
        return "failed", f"{home.receptor} produced no pose ({home.error or 'no pose'})"
    if home.score <= best.score + window:
        return (
            "promiscuous",
            f"{home.receptor} is within {window:g} kcal/mol of {best.receptor}: "
            "the ligand does not distinguish them "
            f"({home.score - best.score:+.2f} kcal/mol)",
        )
    if home.box_volume and best.box_volume:
        ratio = best.box_volume / home.box_volume
        if math.isfinite(ratio) and ratio >= 1.25:
            return (
                "box",
                f"{best.receptor}'s box is {ratio:.2f}x the volume of "
                f"{home.receptor}'s ({best.box_volume:.0f} vs "
                f"{home.box_volume:.0f} A^3), so the comparison favours it for "
                "reasons unrelated to binding",
            )
    return (
        "score",
        f"{home.receptor} is {home.score - best.score:+.2f} kcal/mol behind "
        f"{best.receptor} with comparable boxes; a cross-target preference (or a "
        "force-field error — check the agreement column)",
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_json(path: Any, result: PanelResult) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
    return target


_CSV_COLUMNS = (
    "ligand",
    "receptor",
    "affinity",
    "spread",
    "n_seeds",
    "efficiency",
    "heavy_atoms",
    "torsions",
    "poses",
    "interactions",
    "key_residues",
    "agreement",
    "box_volume",
    "pocket_atoms",
    "best",
    "gap",
    "n_within_window",
    "home",
    "best_is_home",
    "home_rank",
    "status",
    "error",
)


def write_csv(path: Any, result: PanelResult) -> Path:
    """One row per cell, with the ligand's selectivity figures repeated on it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for profile in result.profiles:
            best = profile.best
            for cell in profile.cells:
                writer.writerow(
                    [
                        cell.ligand,
                        cell.receptor,
                        _round(cell.affinity),
                        _round(cell.spread),
                        cell.n_seeds,
                        _round(cell.efficiency, 4),
                        cell.heavy_atoms,
                        cell.torsions,
                        cell.n_poses,
                        cell.n_interactions,
                        ";".join(cell.key_residues),
                        _round(cell.agreement),
                        _round(cell.box_volume, 1),
                        cell.pocket_atoms,
                        best.receptor if best else "",
                        _round(profile.gap),
                        len(profile.within_window),
                        ";".join(sorted(profile.home)),
                        profile.best_is_home,
                        profile.home_rank,
                        cell.status,
                        cell.error,
                    ]
                )
    return target


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def add_panel_parser(sub: argparse._SubParsersAction) -> None:
    """Register ``odock panel`` (called from ``odock.cli_ext``)."""
    parser = sub.add_parser(
        "panel",
        help="profile ligands across a panel of receptors (a comparison, not an assay)",
        description=(
            "Dock every ligand of a library into every receptor of a panel under "
            "one shared protocol, then report the affinity matrix, each ligand's "
            "selectivity profile, and the box that produced every column. A panel "
            "of crystal structures is a comparison, not a selectivity assay: the "
            "receptors are rigid, in one conformation each, with pockets of "
            "different size, scored by one force field."
        ),
    )
    parser.add_argument(
        "--ligands",
        default=None,
        help="library file ('SMILES name' lines, or .sdf); default: the seven "
        "co-crystallised ligands of the bundled panel",
    )
    parser.add_argument(
        "--receptors",
        default="",
        help="comma-separated subset of the bundled panel "
        f"({', '.join(m.pdb_id for m in BUNDLED_MEMBERS)})",
    )
    parser.add_argument("--data-dir", default=None, help="where the PDB files live")
    parser.add_argument("--out", required=True, help="output directory for the campaigns")
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW,
        help=f"kcal/mol window for the 'within the best' count (default {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=DEFAULT_BUFFER,
        help=f"box margin around each reference ligand, A (default {DEFAULT_BUFFER})",
    )
    parser.add_argument(
        "-e",
        "--exhaustiveness",
        type=int,
        default=1,
        help="search effort per ligand (default 1: a panel is a screening tool, "
        "not a redocking test)",
    )
    parser.add_argument("-n", "--num-poses", type=int, default=9)
    parser.add_argument("--seeds", default="42", help="comma-separated seeds")
    parser.add_argument("--scoring", default="vina", help="force field (default vina)")
    parser.add_argument(
        "--no-agreement",
        action="store_true",
        help="skip the three-field rescoring of every cell (faster, less honest)",
    )
    parser.add_argument(
        "--no-fingerprints",
        action="store_true",
        help="skip the per-cell interaction fingerprint",
    )
    parser.add_argument("--json", default=None, help="write the full result here")
    parser.add_argument("--csv", default=None, help="write one row per cell here")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.set_defaults(handler=_handle_panel)


def _handle_panel(args: argparse.Namespace) -> int:
    protocol = PanelProtocol(
        box_buffer=float(args.buffer),
        scoring=str(args.scoring),
        exhaustiveness=int(args.exhaustiveness),
        num_poses=int(args.num_poses),
        seeds=tuple(int(x) for x in str(args.seeds).split(",") if x.strip()),
        agreement=not bool(getattr(args, "no_agreement", False)),
        fingerprints=not bool(getattr(args, "no_fingerprints", False)),
    )
    panel = bundled_panel(
        protocol=protocol,
        window=float(args.window),
        receptors=[x for x in str(args.receptors).split(",") if x.strip()] or None,
    )
    if args.ligands:
        ligands: Any = Path(args.ligands)
        homes: Optional[Mapping[str, Iterable[str]]] = {
            name: spec.home for name, spec in BUNDLED_LIGANDS.items()
        }
        proteins: Optional[Mapping[str, str]] = {
            name: spec.protein for name, spec in BUNDLED_LIGANDS.items()
        }
    else:
        bundled = bundled_ligands()
        ligands = {name: spec.smiles for name, spec in bundled.items()}
        homes = {name: spec.home for name, spec in bundled.items()}
        proteins = {name: spec.protein for name, spec in bundled.items()}

    result = profile(
        ligands,
        panel,
        args.out,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        seeds=protocol.seeds,
        progress=not args.quiet,
        homes=homes,
        proteins=proteins,
    )
    print()
    print(result.matrix())
    print()
    print(result.table())
    print()
    print(result.box_table())
    print()
    report = result.hit_rate()
    if report["rate"] is not None:
        print(
            f"hit rate: {report['hits']}/{report['n']} ligands prefer their own "
            f"protein family (strict, exact co-crystallised structure: "
            f"{report['strict_hits']}/{report['n']})"
        )
        for miss in report["misses"]:
            print(
                f"  miss {miss['ligand']} ({miss['protein']}): best "
                f"{miss['best']} — {miss['reason']}: {miss['detail']}"
            )
    else:
        print("hit rate: not available (no ligand in the library has a known home)")
    print()
    print(
        "A panel of crystal structures is not a selectivity assay: the receptors "
        "are rigid, in one conformation each, and their pockets are not the same "
        "size. Read the box table before the matrix."
    )
    if args.json:
        print(f"\nwrote {write_json(args.json, result)}")
    if args.csv:
        print(f"wrote {write_csv(args.csv, result)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m odock.panel",
        description=(
            "Profile ligands across a panel of receptors and report the "
            "selectivity matrix. A panel of crystal structures is a comparison, "
            "not a selectivity assay."
        ),
    )
    parser.add_argument("--ligands", default=None)
    parser.add_argument("--receptors", default="")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    parser.add_argument("-e", "--exhaustiveness", type=int, default=1)
    parser.add_argument("-n", "--num-poses", type=int, default=9)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--scoring", default="vina")
    parser.add_argument("--no-agreement", action="store_true")
    parser.add_argument("--no-fingerprints", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--csv", default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m odock.panel`` — the same code path as ``odock panel``."""
    return _handle_panel(_build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover - module entry point
    # A `-m` entry does not pass through `cli.main()`, so it does not get the UTF-8
    # stream setup, and on a non-UTF-8 console code page a pipe receives bytes the
    # parent cannot decode (0xa1, GBK, from an em dash).  Same call, same reason.
    from .cli import _use_utf8_streams

    _use_utf8_streams()
    raise SystemExit(main())

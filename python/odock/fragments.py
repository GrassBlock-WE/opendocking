# SPDX-License-Identifier: GPL-3.0-or-later
"""Fragment-based drug discovery: screen small, then **grow** and **link**.

Every other screening path in this project assumes a lead-sized molecule.  Fragments
are the opposite discipline, and the metrics are different too: a fragment hit is judged
by **ligand efficiency** — affinity per heavy atom — not by affinity, because a 12-atom
fragment with a −6 kcal/mol score is more interesting than a 30-atom one with −7.
This module is the missing workflow:

* :func:`fragment_library` — the fragment set: a heavy-atom window, plus
  :func:`cleave_fragments`, which uses RDKit's BRICS decomposition to cut a
  lead-sized pool into fragment-sized pieces rather than inventing them;
* :func:`screen_fragments` — rank by **LE with a stated heavy-atom floor**, so a
  three-atom binder cannot win on a technicality, and report the count beside every
  number;
* :func:`grow_fragment` — enumerate additions at the fragment's **own** attachment
  vectors from a documented reagent set, place each growth by aligning the fragment
  onto the pose it came from, filter clashes against the pocket field, and rank by
  **incremental LE** (ΔLE per added heavy atom) rather than by raw score;
* :func:`link_fragments` — enumerate short linkers from a documented set, filter by
  the **geometry** the two placed fragments demand (the linker must reach both
  attachment vectors with a sane span) and by the pocket's free volume, and report the
  candidates that survive.

**This is enumeration plus scoring, not chemistry.**  A suggested growth or linker has
no synthetic route, no yield, no protecting groups, no reagent availability and no
commercial catalogue behind it; a high-LE fragment scored by a docking-shaped function
is a *hypothesis to test*, not a hit.  The incremental LE is a heuristic, and
:func:`growth_spread` estimates its noise by re-running a growth at a different seed
and reporting the spread.  `docs/FRAGMENTS.md` states all of that beside the tables.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from .metrics import ligand_efficiency as _ligand_efficiency

__all__ = [
    "DEFAULT_MIN_HEAVY",
    "DEFAULT_MAX_HEAVY",
    "DEFAULT_ENERGY_REFERENCE",
    "GROWTH_REAGENTS",
    "LINKERS",
    "FragmentHit",
    "FragmentScreen",
    "PlacedFragment",
    "Growth",
    "GrowthReport",
    "LinkCandidate",
    "LinkReport",
    "require_rdkit",
    "heavy_atom_count",
    "fragment_library",
    "cleave_fragments",
    "screen_fragments",
    "place_fragment",
    "grow_fragment",
    "growth_spread",
    "link_fragments",
    "split_ligand",
    "placement_rmsd",
]

#: The heavy-atom floor for a fragment screen.  Eight is the usual cut: below it a
#: molecule is a solvent-sized `R` group, and a 3-atom binder would win every
#: efficiency table on a technicality rather than on chemistry.
DEFAULT_MIN_HEAVY = 8
#: The ceiling that makes a molecule a *fragment*: 20 heavy atoms is the common
#: definition (a fragment is roughly MW <= 300), and it keeps the library from
#: silently containing leads.
DEFAULT_MAX_HEAVY = 20
#: The energy the pocket score's 1.0 counts as (kcal/mol) when LE is computed from the
#: score rather than from a measurement.  It is the same reference
#: :data:`odock.pocket_score.DEFAULT_ESP_REFERENCE` uses, so the two modules cannot
#: drift apart, and it is a **convention**: a real LE needs a real ``delta_g``, which is
#: why :func:`screen_fragments` accepts one.
DEFAULT_ENERGY_REFERENCE = 10.0

#: The reagent set for growth, as ``(name, SMILES fragment)`` pairs.  Each is a small,
#: common medicinal-chemistry substituent, and the list is deliberately short: a
#: documented set that a reader can check beats a large one nobody can audit.  A growth
#: attaches the fragment's first atom to the attachment vector.
GROWTH_REAGENTS: Tuple[Tuple[str, str], ...] = (
    ("methyl", "C"),
    ("fluoro", "F"),
    ("hydroxyl", "O"),
    ("amino", "N"),
    ("cyano", "C#N"),
    ("carboxyl", "C(=O)O"),
    ("carboxamide", "C(=O)N"),
    ("methoxy", "OC"),
    ("dimethylamino", "N(C)C"),
    ("trifluoromethyl", "C(F)(F)F"),
    ("phenyl", "c1ccccc1"),
    ("pyridin-3-yl", "c1cccnc1"),
    ("amidine", "C(N)=N"),
    ("sulfonamide", "S(=O)(=O)N"),
)

#: The linker set for linking two placed fragments, as ``(name, SMILES with two ``*``)``.
#: ``direct`` is the zero-atom linker: two fragments that already sit a bond apart need
#: no atoms between them at all, and benzamidine is exactly that case (an amidine
#: bonded straight to a phenyl), which makes it the falsifiable test in
#: `docs/FRAGMENTS.md`.
LINKERS: Tuple[Tuple[str, str], ...] = (
    ("direct", "[*][*]"),
    ("methylene", "[*]C[*]"),
    ("ethylene", "[*]CC[*]"),
    ("propylene", "[*]CCC[*]"),
    ("amide", "[*]C(=O)N[*]"),
    ("reverse_amide", "[*]NC(=O)[*]"),
    ("methylamine", "[*]CNC[*]"),
    ("ether", "[*]COC[*]"),
    ("ketone", "[*]C(=O)[*]"),
    ("ethynyl", "[*]C#C[*]"),
    ("E_alkene", "[*]C=CC[*]"),
    ("sulfonamide", "[*]S(=O)(=O)N[*]"),
    ("piperazine", "[*]N1CCN([*])CC1"),
)

#: How much slack a linker has over the distance the two attachment atoms demand (Å).
#: A rotatable linker can absorb about this much by bending; beyond it the geometry is
#: strained and the candidate is rejected rather than scored.
LINKER_SLACK = 2.5

#: **A positional atom index is not an identity.**  Three separate bugs in this module
#: were the same mistake — a map that assumed "atom *i* of the child is atom *i* of the
#: parent".  It broke when a hydrogen was **removed** before a reagent bonded (every index
#: above it shifts), when a linker's dummy atoms were deleted, and when a fragment's atoms
#: were compared against a linked molecule's by position.  Those raised
#: `Range Error: atomId` or silently mis-scored.  Every atom correspondence here now comes
#: from `GetSubstructMatch`; if you add a path, do the same.  Three occurrences of one
#: mistake class is a pattern, not bad luck.
#: The lowest pocket field value a growth's or linker's atoms may sit at (Å, from
#: :func:`odock.pocket_score.PocketField.sample`).  Below it the new atoms overlap the
#: receptor, which is a clash and not a suggestion.
CLASH_LIMIT = -0.6


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for the fragment workflow. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


def heavy_atom_count(mol) -> int:
    """Heavy atoms, the denominator of every efficiency number here."""
    return int(mol.GetNumHeavyAtoms())


def _name_of(mol, fallback: str = "fragment") -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


# ---------------------------------------------------------------------------
# 1. The fragment library
# ---------------------------------------------------------------------------


def fragment_library(
    molecules: Iterable[Any],
    *,
    min_heavy: int = DEFAULT_MIN_HEAVY,
    max_heavy: int = DEFAULT_MAX_HEAVY,
) -> List[Any]:
    """Filter a molecule set to the fragment window ``[min_heavy, max_heavy]``.

    The window is the whole point of a fragment screen — the floor keeps
    solvent-sized molecules out of the efficiency table and the ceiling keeps leads
    out of it — so a molecule outside it is dropped and *counted* by the caller rather
    than quietly included.
    """
    require_rdkit()
    out: List[Any] = []
    for item in molecules:
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            continue
        count = heavy_atom_count(mol)
        if int(min_heavy) <= count <= int(max_heavy):
            out.append(mol)
    return out


def cleave_fragments(
    molecules: Iterable[Any],
    *,
    min_heavy: int = DEFAULT_MIN_HEAVY,
    max_heavy: int = DEFAULT_MAX_HEAVY,
    max_per_molecule: int = 8,
) -> List[Any]:
    """Cut lead-sized molecules into fragments with RDKit's BRICS decomposition.

    A fragment set grown from the pool in hand, rather than invented: BRICS
    decomposes on the bonds a medicinal chemist would break, which is why it is used
    instead of a hand-written cleavage rule.  Fragments outside the window are dropped,
    duplicates are removed by canonical SMILES, and the result is sorted so a run is
    reproducible.
    """
    require_rdkit()
    from rdkit.Chem import BRICS

    seen: Dict[str, Any] = {}
    for item in molecules:
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            continue
        try:
            pieces = sorted(BRICS.BRICSDecompose(mol))
        except Exception:  # pragma: no cover - BRICS refuses some inputs
            continue
        for offset, piece in enumerate(pieces[: int(max_per_molecule)]):
            fragment = Chem.MolFromSmiles(piece)
            if fragment is None:
                continue
            count = heavy_atom_count(fragment)
            if not (int(min_heavy) <= count <= int(max_heavy)):
                continue
            key = Chem.MolToSmiles(fragment)
            if key in seen:
                continue
            # A pool entry usually has no `_Name`, so the parent is named from its own
            # canonical SMILES — otherwise every fragment is called `frag_1` and the
            # report is unreadable (and two different parents collide).
            parent = _name_of(mol, "") or Chem.MolToSmiles(mol)
            fragment.SetProp("_Name", f"{parent[:18]}#{offset + 1}")
            fragment.SetProp("_Parent", parent)
            seen[key] = fragment
    return [seen[key] for key in sorted(seen)]


# ---------------------------------------------------------------------------
# 2. Screening by ligand efficiency
# ---------------------------------------------------------------------------


@dataclass
class FragmentHit:
    """One screened fragment and the three numbers that judge it."""

    name: str = ""
    smiles: str = ""
    heavy_atoms: int = 0
    #: The affinity used, kcal/mol (negative is favourable).  When the caller supplied
    #: none this is the pocket score mapped onto the energy reference, and
    #: :attr:`measured` is False.
    delta_g: float = 0.0
    #: ``-delta_g / heavy_atoms``, via :func:`odock.metrics.ligand_efficiency`.
    le: float = 0.0
    #: Whether ``delta_g`` is a measurement (a docking score, an assay) or a proxy.
    measured: bool = False
    #: The pocket score behind a proxy ``delta_g``, and its shape/electrostatic terms.
    score: float = 0.0
    shape: float = 0.0
    esp: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "smiles": self.smiles,
            "heavy_atoms": int(self.heavy_atoms),
            "delta_g": round(float(self.delta_g), 4),
            "le": round(float(self.le), 4),
            "measured": bool(self.measured),
            "score": round(float(self.score), 4),
            "shape": round(float(self.shape), 4),
            "esp": round(float(self.esp), 4),
        }


@dataclass
class FragmentScreen:
    """A fragment library ranked by ligand efficiency."""

    hits: List[FragmentHit] = field(default_factory=list)
    n_library: int = 0
    n_dropped_heavy: int = 0
    min_heavy: int = DEFAULT_MIN_HEAVY
    max_heavy: int = DEFAULT_MAX_HEAVY
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.hits)

    def top(self, limit: int = 10) -> List[FragmentHit]:
        return self.hits[: int(limit)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_library": int(self.n_library),
            "n_ranked": len(self.hits),
            "n_dropped_heavy": int(self.n_dropped_heavy),
            "min_heavy": int(self.min_heavy),
            "max_heavy": int(self.max_heavy),
            "seconds": round(float(self.seconds), 3),
            "hits": [hit.as_dict() for hit in self.hits],
            "notes": list(self.notes),
        }

    def table(self, limit: int = 10) -> str:
        rows = self.hits[: int(limit)] if limit > 0 else self.hits
        lines = [
            f"{'rank':<5}{'fragment':<26}{'HA':>4}{'LE':>8}{'deltaG':>9}{'score':>8}"
            f"{'shape':>7}{'ESP':>8}  source",
            "-" * 5 + "-" * 26 + "-" * 4 + "-" * 8 + "-" * 9 + "-" * 8 + "-" * 7 + "-" * 8
            + "  " + "-" * 8,
        ]
        for rank, hit in enumerate(rows, start=1):
            lines.append(
                f"{rank:<5}{hit.name[:25]:<26}{hit.heavy_atoms:>4}{hit.le:>8.3f}"
                f"{hit.delta_g:>9.2f}{hit.score:>8.3f}{hit.shape:>7.3f}{hit.esp:>8.2f}"
                f"  {'measured' if hit.measured else 'proxy'}"
            )
        if limit > 0 and len(self.hits) > limit:
            lines.append(f"... and {len(self.hits) - limit} more fragment(s)")
        return "\n".join(lines)


def screen_fragments(
    library: Sequence[Any],
    pocket,
    *,
    min_heavy: int = DEFAULT_MIN_HEAVY,
    max_heavy: int = DEFAULT_MAX_HEAVY,
    affinities: Optional[Dict[str, float]] = None,
    energy_reference: float = DEFAULT_ENERGY_REFERENCE,
    conformers: Optional[int] = None,
    seed: int = 20240101,
    samples: int = 64,
    levels: int = 6,
    population: int = 48,
    restarts: int = 2,
) -> FragmentScreen:
    """Rank fragments by **ligand efficiency** against a pocket field.

    ``affinities`` (name -> kcal/mol) is the honest input: when the caller has a real
    ``delta_g`` — a docking score, an assay — LE is computed from it.  Without one, LE
    is computed from the pocket score through
    :data:`DEFAULT_ENERGY_REFERENCE`, and every hit is marked ``measured=False`` in the
    table, because a score-derived efficiency is a *ranking device* and not an
    affinity.

    The heavy-atom window is reported rather than assumed: ``n_dropped_heavy`` counts
    what the window excluded, so a reader can see whether the table is a fragment table
    or a lead table with small entries.
    """
    require_rdkit()
    from . import pocket_score as _pocket_score

    start = time.perf_counter()
    ranked: List[FragmentHit] = []
    dropped = 0
    shape_only: List[str] = []
    measured_input = {str(key): float(value) for key, value in (affinities or {}).items()}
    for item in library:
        mol = Chem.MolFromSmiles(item) if isinstance(item, str) else item
        if mol is None:
            continue
        count = heavy_atom_count(mol)
        if not (int(min_heavy) <= count <= int(max_heavy)):
            dropped += 1
            continue
        name = _name_of(mol, f"fragment_{len(ranked) + 1}")
        placed = _pocket_score.rank_library(
            [mol], pocket, conformers=conformers, seed=seed, names=[name],
            samples=samples, levels=levels, population=population, restarts=restarts,
            formal_charge_correction=True,
        )
        score = placed.details.get(name, _pocket_score.PoseScore())
        if not score.esp_term_used:
            shape_only.append(name)
        if name in measured_input:
            delta_g = measured_input[name]
            measured = True
        else:
            delta_g = -float(energy_reference) * float(score.combined)
            measured = False
        ranked.append(
            FragmentHit(
                name=name,
                smiles=Chem.MolToSmiles(mol),
                heavy_atoms=count,
                delta_g=float(delta_g),
                le=float(_ligand_efficiency(float(delta_g), count)),
                measured=measured,
                score=float(score.combined),
                shape=float(score.shape),
                esp=float(score.esp),
            )
        )
    ranked.sort(key=lambda hit: (-hit.le, -hit.heavy_atoms, hit.name))
    notes = [
        f"ligand efficiency is -delta_g / heavy atoms, with a floor of {int(min_heavy)} "
        f"heavy atoms and a ceiling of {int(max_heavy)}: a 3-atom binder cannot win on "
        "a technicality",
        (
            "the affinities are the caller's measurements"
            if measured_input else
            f"NO measured affinity was supplied, so delta_g is the pocket score mapped "
            f"through {float(energy_reference):.1f} kcal/mol (the same reference the "
            "score's electrostatic clamp uses).  That makes LE a RANKING DEVICE, not an "
            "affinity: a high-LE fragment from a docking-shaped score is a hypothesis to "
            "test, not a hit"
        ),
        f"{dropped} molecule(s) fell outside the heavy-atom window and were excluded",
    ]
    if shape_only:
        notes.insert(
            1,
            f"BY SHAPE ONLY: the electrostatic term is clamped to zero for "
            f"{len(shape_only)} of {len(ranked)} fragment(s) even after the formal-charge "
            "correction, so their poses were decided by shape alone.  At 3-6 heavy atoms "
            "that is a weak basis for a ranking, and the LE column for those fragments "
            "should be read as a hypothesis rather than a score (docs/FRAGMENTS.md)",
        )
    return FragmentScreen(
        hits=ranked, n_library=len(list(library)), n_dropped_heavy=dropped,
        min_heavy=int(min_heavy), max_heavy=int(max_heavy),
        seconds=time.perf_counter() - start, notes=notes,
    )


# ---------------------------------------------------------------------------
# 3. Growing
# ---------------------------------------------------------------------------


@dataclass
class PlacedFragment:
    """A fragment with the pose it was placed in, ready to grow or link."""

    mol: Any = None
    name: str = ""
    #: Heavy-atom coordinates of the placed pose, in :func:`heavy_coordinates` order.
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    charges: np.ndarray = field(default_factory=lambda: np.zeros(0))
    score: float = 0.0
    le: float = 0.0
    #: Attachment vectors: ``(heavy atom index, outward unit vector, position)``, one per
    #: hydrogen the fragment carries (or per terminal atom when it carries none).
    vectors: List[Tuple[int, np.ndarray, np.ndarray, Optional[int]]] = field(default_factory=list)
    #: The desolvation and repulsion breakdown when they chose the pose, else ``None``.
    physics: Any = None
    notes: List[str] = field(default_factory=list)

    @property
    def heavy_atoms(self) -> int:
        return heavy_atom_count(self.mol) if self.mol is not None else 0


def _attachment_vectors(
    mol, coords_heavy: np.ndarray, heavy: Sequence[int]
) -> List[Tuple[int, np.ndarray, np.ndarray, Optional[int]]]:
    """Where a fragment can be grown, from the **placed** heavy-atom geometry.

    A growth point is a hydrogen: its direction is the outward vector from its heavy
    atom, computed from the heavy neighbours so it needs no hydrogen coordinates (a
    placement only ever produces heavy-atom positions), and its hydrogen is returned so
    the caller can **remove** it — a substitution, not an addition to a full valence.

    Four kinds of site, in priority order, each recorded rather than guessed:

    * an **explicit** hydrogen (``[*H]``) — returned with its atom index so attaching
      removes it.  A hydrogen added to cap a cut bond is exactly this case, and it is
      the site a chemist grows from;
    * an **implicit** hydrogen (the usual case) — nothing to remove, because RDKit's
      implicit count drops by itself when the bond is added;
    * a **terminal heavy atom** with no hydrogen at all — an outward extension, recorded
      as such;
    * nothing, for a saturated non-terminal atom: a fragment cannot be grown there.

    Returns ``(heavy atom index, outward unit vector, the hydrogen's position, the
    explicit hydrogen's index or None)``.
    """
    require_rdkit()
    positions = {
        int(index): np.asarray(coords_heavy[slot], dtype=float)
        for slot, index in enumerate(heavy)
    }
    vectors: List[Tuple[int, np.ndarray, np.ndarray, Optional[int]]] = []
    for atom in mol.GetAtoms():
        index = int(atom.GetIdx())
        if atom.GetAtomicNum() == 1 or index not in positions:
            continue
        heavy_neighbours = [
            int(neighbour.GetIdx()) for neighbour in atom.GetNeighbors()
            if neighbour.GetAtomicNum() > 1 and int(neighbour.GetIdx()) in positions
        ]
        if not heavy_neighbours:
            continue
        position = positions[index]
        centre = np.mean([positions[n] for n in heavy_neighbours], axis=0)
        direction = position - centre
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction = direction / norm
        explicit = [
            int(neighbour.GetIdx()) for neighbour in atom.GetNeighbors()
            if neighbour.GetAtomicNum() == 1
        ]
        if explicit:
            vectors.append(
                (index, direction, position + direction * 1.09, explicit[0])
            )
        elif atom.GetTotalNumHs() > 0:
            vectors.append((index, direction, position + direction * 1.09, None))
        elif atom.GetDegree() == 1:
            vectors.append((index, direction, position + direction * 1.4, None))
    return vectors


def place_fragment(
    mol,
    pocket,
    *,
    name: str = "",
    conformers: Optional[int] = None,
    seed: int = 20240101,
    samples: int = 64,
    levels: int = 6,
    population: int = 48,
    restarts: int = 2,
    physics_enabled: bool = False,
) -> PlacedFragment:
    """Place a fragment in the pocket and keep the pose, ready to grow or link.

    Growth and linking need *where* the fragment sits, not only how well it scores, so
    this returns the winning pose's coordinates and charges alongside the score.

    ``physics_enabled`` is **False by default, deliberately**.
    :mod:`odock.score_physics` supplies the desolvation and repulsion terms the fragment
    diagnosis called for, and they are correct as functions — but they are applied **to
    the winning pose, after the search**, and a term applied to the winner cannot change
    which pose is found.  Measured with the flag on: the amidine's score moves
    0.999 -> 0.749 while its placement RMSD stays 20.61 A, i.e. the numbers become
    *different without becoming better*.  Until the penalties are inside the search
    objective, the true default matters: with them in this path,
    `docs/FRAGMENTS.md` and the tripwire in `tests/test_fragments.py` would describe a
    regime the code no longer implements.  Turn the flag on only together with the
    objective hook, and re-measure the four rows.
    """
    require_rdkit()
    from . import overlay as _overlay
    from . import pocket_score as _pocket_score
    from . import protonation as _protonation
    from . import score_physics as _physics

    label = str(name) or _name_of(mol)
    prepared = _overlay.prepare_molecule(mol, conformers=conformers, seed=seed, name=label)
    if prepared.error or not prepared.conformers:
        return PlacedFragment(mol=mol, name=label, notes=[prepared.error or "no conformer"])
    radii = np.asarray(
        [
            _pocket_score._radius_of(atom.GetSymbol())
            for atom in prepared.mol.GetAtoms()
            if atom.GetAtomicNum() > 1
        ],
        dtype=float,
    )
    best = None
    # **Put the physics in the search, not after it.**  A penalty applied to the winning
    # pose cannot change which pose is found — measured, the amidine's score moved
    # 0.999 -> 0.749 while its placement stayed 20.61 A — so with `physics_enabled` the
    # objective the search maximises *is* the desolvation-plus-repulsion score.
    search = None
    if physics_enabled:
        elements_all = [
            atom.GetSymbol() for atom in prepared.mol.GetAtoms() if atom.GetAtomicNum() > 1
        ]
        polarity = _physics.polarity_per_atom(
            elements_all, np.asarray(prepared.charges, dtype=float)[: len(elements_all)]
        )
        search = _physics.search_objective(polarity)
    for conformer in prepared.conformers:
        score = _pocket_score.place_and_score(
            conformer, prepared.charges, pocket, radii=radii, samples=samples,
            levels=levels, population=population, restarts=restarts, seed=seed,
            objective=search,
        )
        if best is None or score.combined > best[0].combined:
            best = (score, conformer)
    score, coords = best
    # **The charges a small charged fragment actually needs.**  A 3-heavy-atom amidine
    # scored ~0.50, which is exactly `0.5 x shape`: the electrostatic term was clamped to
    # zero, so the placement was shape-only and a fragment carries almost no shape to
    # discriminate with.  `can_represent_formal_charge` is the predicate written for
    # exactly this case, and `correct_formal_charges` is the fix it points at — the same
    # pair task-32 built, reused here rather than a second path being invented.
    notes = list(prepared.notes)
    charges = np.asarray(prepared.charges, dtype=float)
    capability = _protonation.can_represent_formal_charge(prepared.mol, charges)
    if not capability.possible:
        corrected, shifted = _protonation.correct_formal_charges(prepared.mol, charges)
        if shifted:
            charges = corrected
            notes.append(
                "formal-charge correction applied for placement: "
                + "; ".join(shifted)
                + ".  Without it the electrostatic term is clamped to zero and the "
                "placement is shape-only (docs/PROTONATION.md)"
            )
    # Re-score the winning pose with the corrected charges, so the pose the caller keeps
    # is the pose those charges actually score.
    corrected_score = _pocket_score.pocket_score(
        _placed_heavy_coordinates(coords, score, prepared.mol), charges, pocket,
        radii=radii,
    )
    score = corrected_score
    placed_coords = _placed_heavy_coordinates(coords, score, prepared.mol)
    # **The two terms the plain score is missing.**  Without them this pose is either
    # decided by shape alone (term clamped) or drawn to the field minimum (term
    # saturated) — the dual failure `docs/FRAGMENTS.md` §1 measures.  Desolvation charges
    # for burying a polar atom; the repulsion ramp punishes overlap, so the combined value
    # can no longer pin at 1.000.
    elements = [
        atom.GetSymbol() for atom in prepared.mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    heavy_charges = np.asarray(charges, dtype=float)[: len(elements)]
    field_values = None
    try:
        field_values = pocket.sample(placed_coords)
    except Exception:  # pragma: no cover - a field without a sampler
        field_values = None
    physics = _physics.breakdown(
        placed_coords, elements, heavy_charges, radii, field_values,
        shape=score.shape, esp_term=score.esp_term,
    )
    score_used = float(physics.with_physics) if physics_enabled else float(score.combined)
    if physics_enabled:
        notes.append(
            f"the score uses the desolvation and repulsion terms: {score.combined:.3f} "
            f"without them -> {physics.with_physics:.3f} with them (desolvation "
            f"{physics.desolvation_penalty:.3f}, repulsion {physics.repulsion_penalty:.3f})"
        )
    if not score.esp_term_used:
        notes.append(
            f"the electrostatic term is clamped to zero for this fragment "
            f"(interaction energy {score.esp:+.2f} kcal/mol), so its placement was "
            "decided by SHAPE ALONE.  At 3-6 heavy atoms that is a weak basis for a "
            "pose and the numbers should be read as a hypothesis (docs/FRAGMENTS.md)"
        )
    work = Chem.Mol(prepared.mol)
    for existing in range(work.GetNumConformers() - 1, -1, -1):
        work.RemoveConformer(existing)
    conformer = Chem.Conformer(work.GetNumAtoms())
    heavy = [atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() > 1]
    for slot, index in enumerate(heavy):
        conformer.SetAtomPosition(
            index,
            (float(placed_coords[slot, 0]), float(placed_coords[slot, 1]),
             float(placed_coords[slot, 2])),
        )
    work.AddConformer(conformer, assignId=True)
    return PlacedFragment(
        mol=work, name=label, coords=placed_coords, charges=charges,
        score=float(score_used), physics=physics,
        le=float(_ligand_efficiency(-DEFAULT_ENERGY_REFERENCE * float(score_used),
                                    heavy_atom_count(work))),
        vectors=_attachment_vectors(work, placed_coords, heavy), notes=notes,
    )


def _placed_heavy_coordinates(coords: np.ndarray, score, mol) -> np.ndarray:
    """The placed heavy-atom coordinates, recomputed from the winning pose.

    ``place_and_score`` returns the winning rotation and translation, so the coordinates
    it scored are reproduced here exactly rather than re-searched — the growth and the
    linker must be aligned to the pose that was *scored*, not to a second guess at it.
    """
    centred = np.asarray(coords, dtype=float) - np.asarray(coords, dtype=float).mean(axis=0)
    if score.rotation is None or score.translation is None:
        return centred + np.asarray(coords, dtype=float).mean(axis=0)
    return centred @ np.asarray(score.rotation).T + np.asarray(score.translation)


@dataclass
class Growth:
    """One candidate growth, and the two efficiency numbers that judge it."""

    name: str = ""
    reagent: str = ""
    smiles: str = ""
    attachment_atom: int = -1
    added_heavy: int = 0
    heavy_atoms: int = 0
    delta_g: float = 0.0
    le: float = 0.0
    #: ``-(delta_g_child - delta_g_parent) / added_heavy``, kcal/mol per added atom.
    delta_le: float = 0.0
    score: float = 0.0
    clash: int = 0
    #: False when the growth was rejected: the reason is in :attr:`rejected`.
    survived: bool = True
    #: How well the growth's own atoms fitted the fragment's placed pose (Å).  A fit
    #: that is not tiny means the reagent's geometry cannot sit on the placement.
    fit_rmsd: float = 0.0
    rejected: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "reagent": self.reagent,
            "smiles": self.smiles,
            "attachment_atom": int(self.attachment_atom),
            "added_heavy": int(self.added_heavy),
            "heavy_atoms": int(self.heavy_atoms),
            "delta_g": round(float(self.delta_g), 4),
            "le": round(float(self.le), 4),
            "delta_le": round(float(self.delta_le), 4),
            "score": round(float(self.score), 4),
            "fit_rmsd": round(float(self.fit_rmsd), 4),
            "clash": int(self.clash),
            "survived": bool(self.survived),
            "rejected": self.rejected,
        }


@dataclass
class GrowthReport:
    """A ranked set of suggested analogues built on one fragment."""

    parent: str = ""
    parent_heavy: int = 0
    parent_score: float = 0.0
    growths: List[Growth] = field(default_factory=list)
    n_generated: int = 0
    n_clash_rejected: int = 0
    n_unbuildable: int = 0
    reagents: List[str] = field(default_factory=list)
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def survivors(self) -> List[Growth]:
        return [growth for growth in self.growths if growth.survived]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "parent": self.parent,
            "parent_heavy": int(self.parent_heavy),
            "parent_score": round(float(self.parent_score), 4),
            "n_generated": int(self.n_generated),
            "n_survived": len(self.survivors),
            "n_clash_rejected": int(self.n_clash_rejected),
            "n_unbuildable": int(self.n_unbuildable),
            "reagents": list(self.reagents),
            "seconds": round(float(self.seconds), 4),
            "growths": [growth.as_dict() for growth in self.growths],
            "notes": list(self.notes),
        }

    def table(self, limit: int = 10) -> str:
        rows = self.survivors[: int(limit)] if limit > 0 else self.survivors
        lines = [
            f"{'rank':<5}{'reagent':<18}{'site':>5}{'dHA':>5}{'HA':>4}{'deltaLE':>9}"
            f"{'LE':>7}{'score':>8}  smiles",
            "-" * 5 + "-" * 18 + "-" * 5 + "-" * 5 + "-" * 4 + "-" * 9 + "-" * 7 + "-" * 8
            + "  " + "-" * 28,
        ]
        for rank, growth in enumerate(rows, start=1):
            lines.append(
                f"{rank:<5}{growth.reagent[:17]:<18}{growth.attachment_atom:>5}"
                f"{growth.added_heavy:>5}{growth.heavy_atoms:>4}{growth.delta_le:>9.3f}"
                f"{growth.le:>7.3f}{growth.score:>8.3f}  {growth.smiles[:28]}"
            )
        rejected = [growth for growth in self.growths if not growth.survived]
        if rejected:
            lines.append("")
            lines.append(f"{len(rejected)} candidate(s) rejected:")
            for growth in rejected[:4]:
                lines.append(f"  {growth.reagent}@{growth.attachment_atom}: {growth.rejected}")
        return "\n".join(lines)


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation that best maps ``mobile`` onto ``target`` (Kabsch, no reflection).

    Computed here rather than taken from RDKit because the RDKit alignment was the one
    unverified part of the growth path: it returned a non-zero status for a perfectly
    good three-atom map.  The reflection is excluded by flipping the sign of the smallest
    singular value's contribution, which is what stops a mirror image scoring as a fit.
    """
    if mobile.shape != target.shape or mobile.shape[0] < 3:
        raise ValueError(
            f"a rigid fit needs matching coordinate sets of at least three points, got "
            f"{mobile.shape} and {target.shape}"
        )
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (target - target_centre)
    left, _singular, right = np.linalg.svd(covariance)
    sign = float(np.sign(np.linalg.det(right.T @ left.T))) or 1.0
    return right.T @ np.diag([1.0, 1.0, sign]) @ left.T


def _apply_rotation(coords: np.ndarray, rotation: np.ndarray, mobile, target) -> np.ndarray:
    """Rotate and translate ``coords`` so the fitted block lands on its target."""
    return ((coords - mobile.mean(axis=0)) @ rotation.T) + target.mean(axis=0)


def _growth_molecule(parent, attach_atom: int, hydrogen: Optional[int], reagent_smiles: str):
    """Substitute ``reagent_smiles`` for the hydrogen at ``attach_atom``.

    The hydrogen is **removed** first, and that is the whole point: an atom whose valence
    is already full cannot take another bond, so attaching without removing the hydrogen
    is refused by RDKit's valence model — which is what happened to all 14 growths of the
    first version of this function.  Building the combined molecule before the removal
    keeps the index bookkeeping honest: removing an atom shifts every index above it, so
    the attachment and the reagent are remapped by the same rule.
    """
    reagent = Chem.MolFromSmiles(reagent_smiles)
    if reagent is None:
        return None
    offset = parent.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(parent, reagent))
    remove = None if hydrogen is None else int(hydrogen)

    def shifted(index: int) -> int:
        if remove is None:
            return int(index)
        return int(index) - 1 if int(index) > remove else int(index)

    if remove is not None:
        combo.RemoveAtom(remove)
    combo.AddBond(shifted(attach_atom), shifted(offset), Chem.BondType.SINGLE)
    out = combo.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def grow_fragment(
    fragment: PlacedFragment,
    pocket,
    *,
    reagents: Sequence[Tuple[str, str]] = GROWTH_REAGENTS,
    max_growths: int = 0,
    seed: int = 20240101,
    energy_reference: float = DEFAULT_ENERGY_REFERENCE,
) -> GrowthReport:
    """Grow a placed fragment at its own attachment vectors, ranked by incremental LE.

    For every (attachment vector, reagent) pair the growth is built, **aligned on the
    fragment** (the new atoms go where the reagent's conformer puts them relative to the
    pose that was scored, not re-docked), filtered for clashes against the pocket field,
    and scored.  The ranking is ``delta_le = -(delta_g_child - delta_g_parent) /
    added_heavy`` — efficiency *per added atom* — because raw score is exactly the
    metric a fragment workflow exists to avoid.

    Rejected candidates are kept in the report with the reason, so "how many survived"
    is a number a reader can audit rather than a filter they have to trust.
    """
    require_rdkit()
    from . import pocket_score as _pocket_score

    start = time.perf_counter()
    if fragment.mol is None or fragment.coords.size == 0:
        raise ValueError("the fragment has no placed pose: place it first")
    report = GrowthReport(
        parent=fragment.name, parent_heavy=fragment.heavy_atoms,
        parent_score=float(fragment.score),
        reagents=[name for name, _ in reagents],
    )
    if not fragment.vectors:
        report.notes.append(
            "the fragment has no attachment vector (no hydrogen and no terminal heavy "
            "atom), so nothing can be grown from it"
        )
        return report
    parent_delta_g = -float(energy_reference) * float(fragment.score)
    parent_le = float(_ligand_efficiency(parent_delta_g, max(1, fragment.heavy_atoms)))
    heavy_parent = _heavy_indices(fragment.mol)
    for site, (attach_atom, direction, position, hydrogen) in enumerate(fragment.vectors):
        for reagent_name, reagent_smiles in reagents:
            if max_growths and report.n_generated >= int(max_growths):
                break
            report.n_generated += 1
            grown = _growth_molecule(fragment.mol, attach_atom, hydrogen, reagent_smiles)
            if grown is None:
                report.n_unbuildable += 1
                report.growths.append(
                    Growth(
                        name=f"{fragment.name}+{reagent_name}", reagent=reagent_name,
                        attachment_atom=int(attach_atom), survived=False,
                        rejected="RDKit refused the attachment (valence)",
                    )
                )
                continue
            candidate = _place_growth(
                grown, fragment, pocket, direction, position, energy_reference,
                reagent_name, attach_atom, parent_delta_g, parent_le,
                heavy_parent, seed=seed + site,
            )
            if not candidate.survived:
                report.n_clash_rejected += 1
            report.growths.append(candidate)
    survivors = [growth for growth in report.growths if growth.survived]
    survivors.sort(key=lambda growth: (-growth.delta_le, -growth.score))
    rejected = [growth for growth in report.growths if not growth.survived]
    report.growths = survivors + rejected
    report.notes = [
        f"{report.n_generated} growth(s) generated from {len(fragment.vectors)} "
        f"attachment vector(s) and {len(reagents)} reagent(s); "
        f"{len(survivors)} survived the clash filter, {report.n_clash_rejected} were "
        f"rejected for overlapping the receptor and {report.n_unbuildable} could not be "
        "built at all",
        "ranking is delta_le = -(delta_g_child - delta_g_parent) / added_heavy, i.e. "
        "efficiency per added atom; raw score would reward size, which is the mistake a "
        "fragment workflow exists to avoid",
        "the growth is enumerated and scored, NOT synthesis-aware: no route, no yield, "
        "no protecting groups, no reagent availability.  A suggested analogue is a "
        "hypothesis (docs/FRAGMENTS.md)",
    ]
    report.seconds = time.perf_counter() - start
    return report


def _heavy_indices(mol) -> List[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def _place_growth(
    grown,
    fragment: PlacedFragment,
    pocket,
    direction: np.ndarray,
    position: np.ndarray,
    energy_reference: float,
    reagent_name: str,
    attach_atom: int,
    parent_delta_g: float,
    parent_le: float,
    heavy_parent: Sequence[int],
    *,
    seed: int,
) -> Growth:
    """Embed a growth, align it on the scored fragment, clash-filter and score it."""
    from . import pocket_score as _pocket_score

    name = f"{fragment.name}+{reagent_name}"
    added = heavy_atom_count(grown) - len(list(heavy_parent))
    growth = Growth(
        name=name, reagent=reagent_name, smiles=Chem.MolToSmiles(grown),
        attachment_atom=int(attach_atom), added_heavy=int(max(0, added)),
        heavy_atoms=heavy_atom_count(grown),
    )
    work = Chem.AddHs(grown)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    if AllChem.EmbedMolecule(work, params) != 0:
        growth.survived = False
        growth.rejected = "ETKDG could not embed the growth"
        return growth
    # Place the growth by a **locally computed Kabsch fit** over the shared substructure.
    # The map comes from a substructure match, not from "atom i of the growth is atom i of
    # the fragment": the growth had a hydrogen removed when the reagent was attached, so
    # every index above it shifted, and a positional map is wrong the moment that happens.
    # RDKit's own `AlignMol` returned a non-zero status for a perfectly good three-atom
    # map, which is why the rotation is computed here (and checked: a fit whose RMSD is
    # not tiny is a rejected candidate with the number named).
    heavy_reference = Chem.RemoveHs(fragment.mol)
    match = grown.GetSubstructMatch(heavy_reference)
    if not match:
        growth.survived = False
        growth.rejected = "the growth does not contain the fragment it was grown from"
        return growth
    all_coords = np.asarray(work.GetConformer().GetPositions(), dtype=float)
    mobile = all_coords[[int(match[index]) for index in range(heavy_reference.GetNumAtoms())]]
    target = np.asarray(fragment.coords, dtype=float).reshape(-1, 3)
    if mobile.shape != target.shape:
        growth.survived = False
        growth.rejected = (
            f"the fragment has {target.shape[0]} placed atom(s) but the growth matches "
            f"{mobile.shape[0]}"
        )
        return growth
    try:
        rotation = _kabsch(mobile, target)
    except ValueError as exc:
        growth.survived = False
        growth.rejected = f"the growth could not be aligned ({exc})"
        return growth
    fitted = _apply_rotation(all_coords, rotation, mobile, target)
    growth.fit_rmsd = float(np.sqrt(((fitted[[int(match[index]) for index in
                                               range(heavy_reference.GetNumAtoms())]]
                                      - target) ** 2).sum(axis=1).mean()))
    if growth.fit_rmsd > 0.5:
        growth.survived = False
        growth.rejected = (
            f"the growth does not fit the fragment pose (RMSD {growth.fit_rmsd:.2f} A): "
            "the reagent's geometry cannot sit on the placement"
        )
        return growth
    placed_conformer = Chem.Conformer(work.GetNumAtoms())
    for index in range(work.GetNumAtoms()):
        placed_conformer.SetAtomPosition(
            index, (float(fitted[index, 0]), float(fitted[index, 1]), float(fitted[index, 2]))
        )
    work.RemoveAllConformers()
    work.AddConformer(placed_conformer, assignId=True)
    heavy = _heavy_indices(work)
    coords = np.asarray(
        [
            [work.GetConformer().GetAtomPosition(i).x,
             work.GetConformer().GetAtomPosition(i).y,
             work.GetConformer().GetAtomPosition(i).z]
            for i in heavy
        ],
        dtype=float,
    )
    radii = np.asarray([_pocket_score._radius_of(work.GetAtomWithIdx(i).GetSymbol())
                        for i in heavy], dtype=float)
    charges = _gasteiger_heavy(work, heavy)
    score = _pocket_score.pocket_score(coords, charges, pocket, radii=radii)
    growth.score = float(score.combined)
    growth.clash = int(score.clash)
    if score.clash:
        growth.survived = False
        growth.rejected = (
            f"{score.clash} atom(s) overlap the receptor (pocket field below "
            f"{CLASH_LIMIT} A) — a clash is not a suggestion"
        )
        return growth
    growth.delta_g = -float(energy_reference) * float(score.combined)
    growth.le = float(_ligand_efficiency(growth.delta_g, growth.heavy_atoms))
    growth.delta_le = (
        -(growth.delta_g - parent_delta_g) / growth.added_heavy
        if growth.added_heavy > 0 else 0.0
    )
    return growth


def _gasteiger_heavy(mol, heavy: Sequence[int]) -> np.ndarray:
    """Gasteiger charges of the heavy atoms of ``mol``, in ``heavy`` order."""
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:  # pragma: no cover - an untypeable molecule
        return np.zeros(len(heavy), dtype=float)
    values = []
    for index in heavy:
        try:
            values.append(float(mol.GetAtomWithIdx(int(index)).GetDoubleProp("_GasteigerCharge")))
        except Exception:
            values.append(0.0)
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def growth_spread(
    fragment: PlacedFragment,
    pocket,
    *,
    seeds: Sequence[int] = (20240101, 7, 99),
    **kwargs: Any,
) -> Dict[str, Any]:
    """Re-run a growth at several seeds and report the spread of ΔLE.

    Incremental LE is a heuristic computed from a conformer and a score, so its noise
    has to be *estimated* rather than assumed away.  This is the small experiment that
    does it: the top candidate's ΔLE at each seed, and the spread across seeds.
    """
    values: List[float] = []
    names: List[str] = []
    for seed in seeds:
        report = grow_fragment(fragment, pocket, seed=int(seed), **kwargs)
        survivors = report.survivors
        if not survivors:
            continue
        values.append(float(survivors[0].delta_le))
        names.append(survivors[0].name)
    if not values:
        return {"seeds": [int(seed) for seed in seeds], "n": 0, "values": [],
                "spread": float("nan"), "agree": False}
    return {
        "seeds": [int(seed) for seed in seeds],
        "n": len(values),
        "values": [round(value, 4) for value in values],
        "top_names": names,
        "mean": round(float(np.mean(values)), 4),
        "spread": round(float(max(values) - min(values)), 4),
        "agree": len(set(names)) == 1,
    }


# ---------------------------------------------------------------------------
# 4. Linking
# ---------------------------------------------------------------------------


@dataclass
class LinkCandidate:
    """One candidate linker between two placed fragments."""

    linker: str = ""
    smiles: str = ""
    #: The distance between the two attachment atoms in the placed fragments (Å), and
    #: the linker's own end-to-end span.
    demand: float = 0.0
    span: float = 0.0
    heavy_atoms: int = 0
    score: float = 0.0
    clash: int = 0
    #: How far each fragment moved from its placed pose when the linker was attached.
    rmsd_a: float = 0.0
    rmsd_b: float = 0.0
    survived: bool = True
    rejected: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "linker": self.linker,
            "smiles": self.smiles,
            "demand": round(float(self.demand), 3),
            "span": round(float(self.span), 3),
            "heavy_atoms": int(self.heavy_atoms),
            "score": round(float(self.score), 4),
            "clash": int(self.clash),
            "rmsd_a": round(float(self.rmsd_a), 3),
            "rmsd_b": round(float(self.rmsd_b), 3),
            "survived": bool(self.survived),
            "rejected": self.rejected,
        }


@dataclass
class LinkReport:
    """The linkers that span two placed fragments."""

    a: str = ""
    b: str = ""
    demand: float = 0.0
    candidates: List[LinkCandidate] = field(default_factory=list)
    n_generated: int = 0
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def survivors(self) -> List[LinkCandidate]:
        return [item for item in self.candidates if item.survived]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "demand": round(float(self.demand), 3),
            "n_generated": int(self.n_generated), "n_survived": len(self.survivors),
            "seconds": round(float(self.seconds), 4),
            "candidates": [item.as_dict() for item in self.candidates],
            "notes": list(self.notes),
        }

    def table(self) -> str:
        rows = self.survivors + [item for item in self.candidates if not item.survived]
        lines = [
            f"{'linker':<16}{'HA':>4}{'span':>7}{'demand':>8}{'RMSD a/b':>12}{'score':>8}"
            f"  verdict",
            "-" * 16 + "-" * 4 + "-" * 7 + "-" * 8 + "-" * 12 + "-" * 8 + "  " + "-" * 30,
        ]
        for item in rows:
            verdict = (
                "spans both" if item.survived else item.rejected[:30]
            )
            lines.append(
                f"{item.linker[:15]:<16}{item.heavy_atoms:>4}{item.span:>7.2f}"
                f"{item.demand:>8.2f}{item.rmsd_a:>6.2f}/{item.rmsd_b:<5.2f}"
                f"{item.score:>8.3f}  {verdict}"
            )
        return "\n".join(lines)


def _linker_span(reagent_smiles: str, *, seed: int = 20240101) -> float:
    """The end-to-end distance (Å) of a linker's two attachment atoms.

    Measured by building the linker with both attachment points capped by a methyl and
    embedding it, so the span is the geometry's answer rather than a bond count.  A
    ``direct`` linker has no atoms between the attachment points, so its span is a bond
    length (the Å a direct attachment demands).
    """
    capped = reagent_smiles.replace("[*]", "C")
    mol = Chem.MolFromSmiles(capped)
    if mol is None:  # pragma: no cover - a broken table entry
        return 0.0
    if "[*][*]" in reagent_smiles:
        return 1.5  # a C-C single bond: what "no linker" demands
    work = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    if AllChem.EmbedMolecule(work, params) != 0:
        return 0.0
    try:
        AllChem.MMFFOptimizeMolecule(work, maxIters=80)
    except Exception:  # pragma: no cover
        pass
    carbons = [atom.GetIdx() for atom in work.GetAtoms() if atom.GetAtomicNum() == 6]
    conformer = work.GetConformer()
    best = 0.0
    for i in carbons:
        position = conformer.GetAtomPosition(i)
        for j in carbons:
            if j <= i:
                continue
            other = conformer.GetAtomPosition(j)
            distance = math.dist((position.x, position.y, position.z),
                                 (other.x, other.y, other.z))
            best = max(best, distance)
    return float(best)


def link_fragments(
    a: PlacedFragment,
    b: PlacedFragment,
    pocket,
    *,
    linkers: Sequence[Tuple[str, str]] = LINKERS,
    tolerance: float = LINKER_SLACK,
    seed: int = 20240101,
) -> LinkReport:
    """Enumerate linkers that span two placed fragments, and report the candidates.

    The filter is geometry plus free volume, and it is deliberately strict because that
    is what makes the output checkable:

    1. **span** — the linker's own end-to-end length must cover the distance between the
       two attachment atoms, within ``tolerance`` Å of slack (what a rotatable linker can
       absorb by bending).  A linker that is too short cannot reach; one that is far too
       long is a different molecule, not a link;
    2. **both fragments stay put** — with the two fragments pinned, the embedded complex
       must place each fragment's atoms back within 0.5 Å of where they were placed;
    3. **free volume** — no atom of the complex may overlap the receptor (the same clash
       limit the growth filter uses).

    Only linkers with two attachment points are in the set, and both fragments need an
    attachment vector; a fragment with none cannot be linked and the report says so.
    """
    require_rdkit()
    from . import pocket_score as _pocket_score

    start = time.perf_counter()
    report = LinkReport(a=a.name, b=b.name)
    if not a.vectors or not b.vectors:
        report.notes.append(
            "one of the fragments has no attachment vector, so it cannot be linked"
        )
        return report
    index_a, _direction_a, position_a, hydrogen_a = a.vectors[0]
    index_b, _direction_b, position_b, hydrogen_b = b.vectors[0]
    report.demand = float(math.dist(tuple(position_a), tuple(position_b)))
    for name, reagent in linkers:
        report.n_generated += 1
        candidate = LinkCandidate(linker=name)
        try:
            candidate.span = _linker_span(reagent, seed=seed)
        except Exception:  # pragma: no cover - a broken table entry
            candidate.survived = False
            candidate.rejected = "the linker could not be embedded"
            report.candidates.append(candidate)
            continue
        if candidate.span <= 0.0:
            candidate.survived = False
            candidate.rejected = "the linker could not be measured"
            report.candidates.append(candidate)
            continue
        low, high = report.demand - 0.6, report.demand + float(tolerance)
        if not (low <= candidate.span <= high):
            candidate.survived = False
            candidate.rejected = (
                f"span {candidate.span:.2f} A cannot cover {report.demand:.2f} A "
                f"(needs {low:.2f}-{high:.2f} A)"
            )
            report.candidates.append(candidate)
            continue
        built = _link_molecules(
            a.mol, index_a, hydrogen_a, b.mol, index_b, hydrogen_b, reagent
        )
        if built is None:
            candidate.survived = False
            candidate.rejected = "the linked molecule could not be built"
            report.candidates.append(candidate)
            continue
        candidate.smiles = Chem.MolToSmiles(built)
        candidate.heavy_atoms = heavy_atom_count(built)
        candidate = _place_link(
            built, a, b, pocket, candidate, index_a, index_b, seed=seed
        )
        report.candidates.append(candidate)
    report.candidates.sort(
        key=lambda item: (not item.survived, -item.score)
    )
    report.notes = [
        f"the two attachment atoms are {report.demand:.2f} A apart, so a linker must "
        f"span {report.demand - 0.6:.2f}-{report.demand + float(tolerance):.2f} A",
        f"{report.n_generated} linker(s) enumerated from a documented set; "
        f"{len(report.survivors)} span both fragments without a clash",
        "linking is enumeration plus geometry, NOT synthesis-aware: a candidate has no "
        "route, no yield and no protecting-group analysis (docs/FRAGMENTS.md)",
    ]
    report.seconds = time.perf_counter() - start
    return report


def _link_molecules(
    mol_a, index_a: int, hydrogen_a: Optional[int],
    mol_b, index_b: int, hydrogen_b: Optional[int],
    reagent_smiles: str,
):
    """Bond two fragments through a linker, substituting both attachment hydrogens.

    ``[*][*]`` is the zero-atom linker: a direct bond between the two fragments, which
    is what benzamidine needs (its amidine is bonded straight to its phenyl).  Both cap
    hydrogens are removed — the same substitution the growth path performs, and for the
    same reason — and every index is remapped by counting the removals below it rather
    than by hoping the order works out.
    """
    direct = "[*][*]" in reagent_smiles
    linker = None if direct else Chem.MolFromSmiles(reagent_smiles)
    if not direct and linker is None:
        return None
    anchor_a = anchor_b = None
    if linker is not None:
        dummies = [a.GetIdx() for a in linker.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummies) != 2:
            return None
        # The attachment point is the dummy's neighbour, read before the dummy goes.
        anchor_a, anchor_b = (
            int(linker.GetAtomWithIdx(int(dummy)).GetNeighbors()[0].GetIdx())
            for dummy in dummies
        )

    n_a, n_b = mol_a.GetNumAtoms(), mol_b.GetNumAtoms()
    pieces = [mol_a, mol_b] if linker is None else [mol_a, mol_b, linker]
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = Chem.CombineMols(combined, piece)
    editable = Chem.RWMol(combined)
    base_linker = n_a + n_b

    removals = []
    if hydrogen_a is not None:
        removals.append(int(hydrogen_a))
    if hydrogen_b is not None:
        removals.append(n_a + int(hydrogen_b))
    if linker is not None:
        removals.extend(base_linker + int(dummy) for dummy in dummies)

    def mapped(index: int) -> int:
        return int(index) - sum(1 for removed in removals if removed < int(index))

    for target in sorted(set(removals), reverse=True):
        editable.RemoveAtom(int(target))
    editable.AddBond(mapped(index_a), mapped(n_a + int(index_b)), Chem.BondType.SINGLE)
    if linker is not None:
        pass  # the fragments are joined through the linker's own atoms below
    # A direct bond joins the fragments; a linker is inserted by replacing that bond.
    if linker is not None:
        editable.RemoveBond(mapped(index_a), mapped(n_a + int(index_b)))
        editable.AddBond(mapped(index_a), mapped(base_linker + int(anchor_a)),
                         Chem.BondType.SINGLE)
        editable.AddBond(mapped(n_a + int(index_b)), mapped(base_linker + int(anchor_b)),
                         Chem.BondType.SINGLE)
    out = editable.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _place_link(built, a: PlacedFragment, b: PlacedFragment, pocket, candidate, index_a, index_b, *, seed: int):
    """Embed a linked molecule with both fragments pinned, then clash-filter it."""
    from . import pocket_score as _pocket_score

    work = Chem.AddHs(built)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 1
    if AllChem.EmbedMolecule(work, params) != 0:
        candidate.survived = False
        candidate.rejected = "ETKDG could not embed the linked molecule"
        return candidate
    # Pin fragment A with a substructure-derived map (both cap hydrogens were removed when
    # the linker was attached, so a positional index is meaningless — the same lesson the
    # growth path taught, and it raises `Range Error: atomId` when ignored), then measure
    # how far each fragment sits from the pose it was placed in.
    try:
        reference_a = Chem.RemoveHs(a.mol)
        reference_b = Chem.RemoveHs(b.mol)
        match_a = work.GetSubstructMatch(reference_a)
        match_b = work.GetSubstructMatch(reference_b)
        if not match_a or not match_b:
            candidate.survived = False
            candidate.rejected = "the linked molecule lost one of the two fragments"
            return candidate
        all_coords = np.asarray(work.GetConformer().GetPositions(), dtype=float)
        target_a = np.asarray(a.coords, dtype=float).reshape(-1, 3)
        target_b = np.asarray(b.coords, dtype=float).reshape(-1, 3)
        mobile_a = all_coords[list(match_a)]
        rotation = _kabsch(mobile_a, target_a)
        fitted = _apply_rotation(all_coords, rotation, mobile_a, target_a)
        pinned = Chem.Conformer(work.GetNumAtoms())
        for index in range(work.GetNumAtoms()):
            pinned.SetAtomPosition(index, tuple(float(value) for value in fitted[index]))
        work.RemoveAllConformers()
        work.AddConformer(pinned, assignId=True)

        def block_rmsd(match, target) -> float:
            block = fitted[list(match)]
            count = min(block.shape[0], target.shape[0])
            if count == 0:
                return float("nan")
            return float(np.sqrt(((block[:count] - target[:count]) ** 2).sum(axis=1).mean()))

        candidate.rmsd_a = block_rmsd(match_a, target_a)
        candidate.rmsd_b = block_rmsd(match_b, target_b)
    except Exception as exc:  # a fit that does not exist is a rejected candidate
        candidate.survived = False
        candidate.rejected = f"the linked molecule could not be aligned ({exc})"
        return candidate
    if candidate.rmsd_a > 0.5 or candidate.rmsd_b > 0.5:
        candidate.survived = False
        candidate.rejected = (
            f"the linker cannot hold both fragments in place (RMSD "
            f"{candidate.rmsd_a:.2f}/{candidate.rmsd_b:.2f} A)"
        )
        return candidate
    heavy = _heavy_indices(work)
    coords = np.asarray(
        [
            [work.GetConformer().GetAtomPosition(i).x,
             work.GetConformer().GetAtomPosition(i).y,
             work.GetConformer().GetAtomPosition(i).z]
            for i in heavy
        ],
        dtype=float,
    )
    radii = np.asarray([_pocket_score._radius_of(work.GetAtomWithIdx(i).GetSymbol())
                        for i in heavy], dtype=float)
    charges = _gasteiger_heavy(work, heavy)
    score = _pocket_score.pocket_score(coords, charges, pocket, radii=radii)
    candidate.score = float(score.combined)
    candidate.clash = int(score.clash)
    if score.clash:
        candidate.survived = False
        candidate.rejected = f"{score.clash} atom(s) overlap the receptor"
    return candidate


def _fragment_rmsd(work, reference, start: int, stop: int) -> float:
    """Heavy-atom RMSD of one fragment's block, in ``work``'s current conformer.

    The mapping is positional because the linked molecule is built by concatenating the
    fragments in order, and the pinned alignment is on fragment A, so a small RMSD is
    the statement "the linker did not have to move either fragment".
    """
    conformer_w = work.GetConformer()
    conformer_r = reference.GetConformer()
    squared = []
    for index in range(start, stop):
        if work.GetAtomWithIdx(index).GetAtomicNum() <= 1:
            continue
        left = conformer_w.GetAtomPosition(index)
        right = conformer_r.GetAtomPosition(index)
        squared.append((left.x - right.x) ** 2 + (left.y - right.y) ** 2
                       + (left.z - right.z) ** 2)
    if not squared:
        return 0.0
    return float(np.sqrt(np.mean(squared)))


# ---------------------------------------------------------------------------
# 5. Validation helpers
# ---------------------------------------------------------------------------


def docked_pose_mol(smiles: str, pdbqt_path) -> Any:
    """An RDKit molecule of a **docked PDBQT pose**, with bond orders from a template.

    RDKit's PDB reader cannot parse PDBQT (the element column is not where it looks), so
    the atoms and coordinates come from :func:`odock.pocket_score.read_pdbqt_atoms` and
    the connectivity is inferred by distance; `AssignBondOrdersFromTemplate` then gives
    the bond orders the chemistry implies.  The result keeps the docked coordinates in
    its conformer and its atom order is the file's, so a template substructure match
    maps template atoms onto the pose — which is what makes the placement RMSD in
    `docs/FRAGMENTS.md` a measurement rather than an assumption.
    """
    require_rdkit()
    from .pocket_score import read_pdbqt_atoms

    atoms = read_pdbqt_atoms(pdbqt_path)
    editable = Chem.RWMol()
    for atom in atoms:
        editable.AddAtom(Chem.Atom(str(atom.element)))
    conformer = Chem.Conformer(len(atoms))
    for index, atom in enumerate(atoms):
        coords = np.asarray(atom.coords, dtype=float).reshape(3)
        conformer.SetAtomPosition(index, (float(coords[0]), float(coords[1]), float(coords[2])))
    positions = np.asarray([np.asarray(a.coords, dtype=float).reshape(3) for a in atoms])
    for left in range(len(atoms)):
        for right in range(left + 1, len(atoms)):
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            # 1.8 A covers every bond in a small molecule without inventing one across
            # a ring or between two fragments.
            if distance < 1.8:
                editable.AddBond(int(left), int(right), Chem.BondType.SINGLE)
    raw = editable.GetMol()
    raw.AddConformer(conformer, assignId=True)
    template = Chem.MolFromSmiles(str(smiles))
    if template is None:
        raise ValueError(f"{smiles!r} is not a parsable SMILES")
    return AllChem.AssignBondOrdersFromTemplate(template, raw)


def split_ligand(smiles: str, cuts: Sequence[Tuple[int, int]]) -> List[Any]:
    """Split a ligand into fragments by cutting the named bonds.

    ``cuts`` is a list of atom-index pairs; the bond between each pair is broken and the
    free valences are capped with hydrogens on a copy, so each fragment is a real
    molecule that can be placed.  The bond indices are the caller's, which keeps the
    split explicit rather than inferred: `docs/FRAGMENTS.md` uses benzamidine's
    amidine-phenyl bond for the validation and says which indices those are.
    """
    require_rdkit()
    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        raise ValueError(f"{smiles!r} is not a parsable SMILES")
    broken = Chem.RWMol(parent)
    for left, right in cuts:
        broken.RemoveBond(int(left), int(right))
        # Each cut leaves two dangling valences.  They are filled with a real hydrogen
        # atom rather than left open, because an open valence is not a molecule: RDKit
        # refuses to sanitise it and the fragment could not be embedded or scored.
        # Chemically this is the right cap — cutting benzamidine's amidine-phenyl bond
        # gives formamidine and benzene, which is what each fragment *is*.
        for anchor in (int(left), int(right)):
            hydrogen = broken.AddAtom(Chem.Atom(1))
            broken.AddBond(anchor, hydrogen, Chem.BondType.SINGLE)
    fragments = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    out = []
    for index, fragment in enumerate(fragments):
        # No `Chem.AddHs` here: the cap hydrogen already filled the broken valence, and
        # RDKit's cached implicit-H count is stale after the edit, so adding hydrogens
        # again pushed the cap atoms past their valence ("explicit valence for atom #1 C,
        # 5").  The remaining hydrogens stay implicit until a caller embeds the fragment.
        fragment.SetProp("_Name", f"fragment_{index + 1}")
        out.append(fragment)
    return out


def placement_rmsd(placed: PlacedFragment, reference_coords: np.ndarray) -> float:
    """Heavy-atom RMSD between a placed fragment and a reference position (Å).

    **No re-alignment**: the two coordinate sets are already in the same frame (the
    receptor's), so the number answers "did the fragment land where the reference says
    it should", which is the falsifiable question.  Aligning first would answer a much
    easier one.
    """
    left = np.asarray(placed.coords, dtype=float).reshape(-1, 3)
    right = np.asarray(reference_coords, dtype=float).reshape(-1, 3)
    if left.shape != right.shape:
        raise ValueError(
            f"the fragment has {left.shape[0]} heavy atom(s) and the reference "
            f"{right.shape[0]}: they must be the same atoms"
        )
    if left.size == 0:
        return float("nan")
    return float(np.sqrt(((left - right) ** 2).sum(axis=1).mean()))

# SPDX-License-Identifier: GPL-3.0-or-later
"""Score a ligand against the **pocket**, not against another ligand.

Everything 3-D in this project until now compared a ligand to a ligand.  This module
compares a ligand to the receptor's own surface and electrostatic field:

* the **pocket shape** is the probe-centre field of
  :func:`odock.gui.surface.scalar_field`, ``F(p) = min_i(|p − c_i| − (r_i + probe))``,
  sampled on a regular grid over the box: negative inside the receptor's van der
  Waals volume, zero at a contact, positive in the open space a ligand may occupy;
* the **pocket field** is :func:`odock.gui.surface.electrostatic_potential` on the
  same grid, ``φ(p) = 332.0637 · Σ q_i / (ε(r) · r)`` in kcal/(mol·e), using the
  receptor's per-atom charges;
* a **pose** is scored by looking both up at the ligand's heavy-atom centres, and the
  score is the pair of terms every other 3-D method in this project reports:

  ``shape = contact / (contact + clash + exposed)``
  ``ESP   = Σ_i q_i · φ(r_i)``                                  (kcal/mol, signed)
  ``score = w · shape + (1 − w) · min(1, max(0, −ESP / E_ref))``  (w = 0.5)

  with ``contact``, ``clash`` and ``exposed`` counting the ligand's heavy atoms whose
  grid value is in contact range, overlapping the receptor, or far from any receptor
  atom.  The ESP term is a *complementarity* — a positive ligand charge in a positive
  pocket potential is penalised — and ``E_ref`` is the interaction energy that counts
  as fully complementary, stated as a convention rather than fitted.

**It is fast because the field is built once.**  A pose is scored by two vectorised
grid lookups per heavy atom, so a whole library can be placed and ranked without any
docking: the placement is a rigid-body search over the same deterministic rotation set
:mod:`odock.overlay` uses, refined by the same multi-resolution perturbation, with the
translation clamped inside the box.  `docs/POCKET_SCORE.md` reports the milliseconds
per pose and how it scales with the pocket's voxel count.

**What this is not.**  Shape complementarity is not a binding energy: there is no
desolvation, no entropy, no induced fit, and the field comes from one rigid structure
with Gasteiger or file charges.  A pocket field inherits every limitation
`docs/CONFORMERS.md` documents for ensembles, plus its own: the receptor does not
move, and the box is a hypothesis about where the ligand goes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

from . import overlay as _overlay
from .sasa import radius_of as _radius_of

__all__ = [
    "DEFAULT_SPACING",
    "DEFAULT_PROBE",
    "DEFAULT_CONTACT_MARGIN",
    "DEFAULT_ESP_REFERENCE",
    "SHAPE_WEIGHT",
    "PocketAtom",
    "PocketField",
    "PoseScore",
    "PocketRanking",
    "SpearmanResult",
    "require_rdkit",
    "read_pdbqt_atoms",
    "spearman",
    "spearman_ci",
    "pocket_score",
]

#: Grid spacing of the pocket fields (Å).  Measured on the 3PTB box (17.9 x 20.0 x
#: 20.5 Å): 0.5 Å costs 12.8 s to build (178 k voxels, the electrostatic potential
#: dominating at 7.6 s), 0.8 Å about 1 s (52 k voxels) and 1.0 Å about 0.5 s.  The
#: score itself does not care — it is two lookups per atom — so the spacing is a
#: build-time choice, and 0.8 Å keeps the nearest-voxel error (±0.4 Å) well inside
#: the contact kernel's 1.5 Å width.
DEFAULT_SPACING = 0.8
#: Probe radius for the shape field (Å) — a water-sized probe, the SASA convention.
DEFAULT_PROBE = 1.4
#: Extra room the *grid* takes around the box (Å), so a ligand atom that strays just
#: outside it is still looked up rather than clamped to the boundary.
DEFAULT_GRID_MARGIN = 2.0
#: Extra room the *receptor shell* takes around the box (Å).  The field is local, so
#: atoms further out cannot change it inside the box; this is what keeps a 3 000-atom
#: protein cheap.  Measured: 700 of 3PTB's 1 630 atoms fall inside.
DEFAULT_SHELL_MARGIN = 4.0
#: An atom is "in contact" when its field value is at most this far above zero (Å).
#: ``F`` is the distance from the atom's surface to the nearest receptor surface, so
#: ``probe + margin`` is roughly a van der Waals contact plus one hydrogen bond.
DEFAULT_CONTACT_MARGIN = 2.5
#: The contact the shape term rewards (Å): ``F`` at which a ligand atom sits snugly
#: against the receptor surface.  A counted contact saturates — a small ligand in a
#: big pocket has *every* atom in contact at every pose — so the term is a Gaussian
#: kernel around this distance rather than a count, and it is continuous, which the
#: pose search needs.
CONTACT_OPTIMUM = 0.8
#: Width of that kernel (Å).  1.5 Å lets a contact 2.5 Å away still count about a
#: third, and an atom 4 Å from any receptor atom is effectively exposed.
CONTACT_WIDTH = 1.5
#: The **size complementarity** reference: the van der Waals volume (Å³) of the
#: ligand the pocket is known to bind.  A pose search saturates the contact kernel —
#: measured on 3PTB, every molecule from a 9-atom amidine to 25-atom warfarin reaches
#: 0.994-0.996, because a 20 Å pocket always has room to put all of a small ligand in
#: contact — so the score needs a term that says whether the ligand is the *right
#: size* for this pocket.  ``0.0`` disables the term (contact and clash only).
DEFAULT_SIZE_REFERENCE = 0.0
#: Width of the size term as a fraction of the reference volume: a ligand half or
#: one-and-a-half times the reference's volume keeps ``exp(-0.5) = 0.61``.
SIZE_WIDTH_FRACTION = 0.5
#: The ligand-receptor interaction energy that counts as fully complementary
#: (kcal/mol).  A convention, like the overlay's 0.5/0.5 weighting: it maps a signed
#: energy onto the [0, 1] scale the shape term lives on.
DEFAULT_ESP_REFERENCE = 10.0
#: Weight of the shape term in the combined score (the rest is the ESP term).
SHAPE_WEIGHT = 0.5

#: A ligand atom placed here is treated as clashing, whatever the field says.
_CLASH_TOLERANCE = 0.6


def ligand_volume(radii: Sequence[float]) -> float:
    """The ligand's van der Waals volume in Å³, as a sum over atoms.

    An **upper bound**: overlapping atoms are counted twice, which is the standard
    cheap estimate and is applied identically to the reference ligand, so the *ratio*
    the size term uses is unaffected by the approximation.
    """
    values = np.asarray(radii, dtype=float).reshape(-1)
    return float((4.0 / 3.0) * math.pi * (values ** 3).sum())


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for pocket scoring. Install it with `pip install "
            "rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# The pocket
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PocketAtom:
    """One receptor atom: where it is, how big it is, and what charge it carries."""

    x: float
    y: float
    z: float
    element: str = "C"
    charge: float = 0.0
    radius: float = 0.0
    #: Residue identity, kept so :func:`odock.protonation.salt_bridge_warnings` can
    #: ask whether a charged group in the ligand sits next to an oppositely charged
    #: residue — a question that needs the residue name, not just the coordinates.
    residue_name: str = ""
    chain: str = ""
    residue_number: str = ""

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            object.__setattr__(self, "radius", _radius_of(self.element))

    @property
    def coords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


def read_pdbqt_atoms(
    path: Path | str, *, keep_hydrogens: bool = False
) -> List[PocketAtom]:
    """Read a receptor PDBQT's atoms, with the file's own charges and residues.

    AutoDock PDBQT is a PDB with the partial charge in columns 71-76 and the AD4
    type in 78-79; both are read here, so the field uses the charges the docking
    itself uses rather than a second guess at them.  The residue name, chain and
    number come along for the protonation check, which is a question about residues.
    Hydrogens are dropped by default: they add voxels and no shape a ligand can feel.
    """
    atoms: List[PocketAtom] = []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip() if len(line) >= 78 else ""
        element = "".join(ch for ch in element if ch.isalpha()) or line[12:16].strip()[:1]
        if not keep_hydrogens and element.upper().startswith("H"):
            continue
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:  # pragma: no cover - a malformed line
            continue
        try:
            charge = float(line[70:76])
        except (ValueError, IndexError):
            charge = 0.0
        atoms.append(
            PocketAtom(
                x=x, y=y, z=z, element=element, charge=charge,
                residue_name=line[17:20].strip(),
                chain=line[21:22].strip(),
                residue_number=line[22:26].strip(),
            )
        )
    return atoms


@dataclass
class PocketField:
    """The pocket's shape and electrostatic fields on one regular grid."""

    #: ``(nx, ny, nz)`` voxel counts.
    shape: Tuple[int, int, int] = (0, 0, 0)
    #: Grid origin (the lowest corner) and spacing, in Å.
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    spacing: float = DEFAULT_SPACING
    #: Probe-centre field, negative inside the receptor (Å).
    shape_field: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    #: Electrostatic potential on the same grid, kcal/(mol·e).
    esp_field: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    probe: float = DEFAULT_PROBE
    n_atoms: int = 0
    #: Van der Waals volume (Å³) of the ligand this pocket is known to bind; ``0.0``
    #: disables the size term.  Set it with :meth:`set_size_reference`.
    size_reference: float = DEFAULT_SIZE_REFERENCE
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 0

    @property
    def volume(self) -> float:
        return self.n_voxels * float(self.spacing) ** 3

    @property
    def open_volume(self) -> float:
        """The box volume a ligand centre could occupy unclashed (Å³)."""
        if not self.n_voxels:
            return 0.0
        return float((self.shape_field > 0.0).sum()) * float(self.spacing) ** 3

    def set_size_reference(self, volume: float, *, note: str = "") -> "PocketField":
        """Calibrate the size term on a known ligand's van der Waals volume (Å³).

        The reference is what makes the term a statement about *this* pocket: the
        co-crystallised ligand's volume is the size the pocket is known to accept,
        and a fragment or an oversized molecule is penalised on both sides.
        """
        self.size_reference = max(0.0, float(volume))
        if note:
            self.notes.append(note)
        return self

    def size_factor(self, volume: float) -> float:
        """How close ``volume`` is to the pocket's reference size, in ``(0, 1]``."""
        reference = float(self.size_reference)
        if reference <= 0.0:
            return 1.0
        width = max(1e-9, SIZE_WIDTH_FRACTION * reference)
        return float(math.exp(-((float(volume) - reference) ** 2) / (2.0 * width ** 2)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": [int(value) for value in self.shape],
            "n_voxels": self.n_voxels,
            "spacing": round(float(self.spacing), 4),
            "probe": round(float(self.probe), 4),
            "volume": round(self.volume, 2),
            "open_volume": round(self.open_volume, 2),
            "size_reference": round(float(self.size_reference), 2),
            "n_atoms": int(self.n_atoms),
            "seconds": round(float(self.seconds), 4),
            "notes": list(self.notes),
        }

    # -- construction ------------------------------------------------------

    @classmethod
    def build(
        cls,
        atoms: Sequence[PocketAtom],
        *,
        center: Sequence[float],
        size: Sequence[float],
        spacing: float = DEFAULT_SPACING,
        probe: float = DEFAULT_PROBE,
        dielectric: str = "distance",
        epsilon: float = 4.0,
        grid_margin: float = DEFAULT_GRID_MARGIN,
        shell_margin: float = DEFAULT_SHELL_MARGIN,
    ) -> "PocketField":
        """Sample the shape and electrostatic fields over a box.

        ``center``/``size`` are the docking box (Å).  The grid covers the box plus
        ``grid_margin``; only the receptor atoms within the box plus ``shell_margin``
        are handed to the field builders — the field is local, so the shell is not an
        approximation of the answer, it *is* the answer inside the box, and it is what
        keeps a 3 000-atom protein cheap.
        """
        from .gui.surface import electrostatic_potential, scalar_field

        started = time.perf_counter()
        if not atoms:
            raise ValueError("no receptor atom: a pocket field needs a receptor")
        if float(spacing) <= 0.0:
            raise ValueError(f"spacing must be positive, got {spacing!r}")
        centre = np.asarray(center, dtype=float).reshape(3)
        extent = np.asarray(size, dtype=float).reshape(3)
        if np.any(extent <= 0.0):
            raise ValueError(f"box size must be positive, got {size!r}")
        low = centre - extent / 2.0 - float(grid_margin)
        high = centre + extent / 2.0 + float(grid_margin)
        axes = [
            np.arange(low[index], high[index] + float(spacing), float(spacing))
            for index in range(3)
        ]
        coords = np.asarray([atom.coords for atom in atoms], dtype=float)
        shell_low = centre - extent / 2.0 - float(shell_margin)
        shell_high = centre + extent / 2.0 + float(shell_margin)
        inside = np.all((coords >= shell_low) & (coords <= shell_high), axis=1)
        shell = [atom for atom, keep in zip(atoms, inside) if keep]
        if not shell:
            raise ValueError("no receptor atom falls inside the box plus its margin")

        shape_field = scalar_field(
            np.zeros((0, 3)), shell, probe=float(probe), spacing=float(spacing),
            axes=axes,
        )
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        esp = electrostatic_potential(
            grid, shell, dielectric=str(dielectric), epsilon=float(epsilon)
        )
        esp_field = np.asarray(esp, dtype=float).reshape(shape_field.shape)
        notes = [
            f"{len(shell)} of {len(atoms)} receptor atom(s) fall inside the box plus "
            f"its {float(shell_margin):.1f} A shell margin and were used to build the "
            "field; the field is local, so the shell is the answer inside the box, "
            "not an approximation of it",
            f"electrostatics: Coulomb with dielectric='{dielectric}' (epsilon "
            f"{float(epsilon):.1f}); the ligand term is Σ q_i φ(r_i), the same "
            "convention as odock.gui.surface's potential",
        ]
        if not np.any(np.abs(esp_field) > 1e-9):
            notes.append(
                "the receptor carries no charge in this file, so the electrostatic "
                "term is zero and the score is the shape term alone"
            )
        return cls(
            shape=tuple(int(value) for value in shape_field.shape),
            origin=low,
            spacing=float(spacing),
            shape_field=shape_field,
            esp_field=esp_field,
            probe=float(probe),
            n_atoms=len(shell),
            seconds=time.perf_counter() - started,
            notes=notes,
        )

    # -- lookup ------------------------------------------------------------

    def _indices(self, points: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(points, dtype=float) - self.origin) / float(self.spacing)
        return np.clip(np.rint(scaled).astype(np.int64), 0, np.asarray(self.shape) - 1)

    def sample(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The ``(shape, esp)`` field values at each point, by nearest voxel.

        Nearest-voxel rather than trilinear on purpose: a ligand atom moves by a
        fraction of a voxel during the pose search, and the interpolated field is
        smooth enough that the search's ranking is unchanged while the lookup stays a
        single fancy-index.  ``tests/test_pocket_score.py`` pins that the two agree.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if points.size == 0 or not self.n_voxels:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        index = self._indices(points)
        return (
            self.shape_field[index[:, 0], index[:, 1], index[:, 2]],
            self.esp_field[index[:, 0], index[:, 1], index[:, 2]],
        )

    def open_fraction(self, threshold: float = 0.0) -> float:
        """The share of the box's voxels a ligand centre could occupy unclashed."""
        if not self.n_voxels:
            return 0.0
        return float((self.shape_field > float(threshold)).mean())


# ---------------------------------------------------------------------------
# Scoring one pose
# ---------------------------------------------------------------------------


@dataclass
class PoseScore:
    """One placed ligand's complementarity with the pocket."""

    shape: float = 0.0
    esp: float = 0.0
    combined: float = 0.0
    #: Ligand heavy atoms in contact, clashing, or exposed.
    contact: int = 0
    clash: int = 0
    exposed: int = 0
    n_atoms: int = 0
    #: The ligand's van der Waals volume (Å³) and its size factor against the pocket's
    #: reference, for a report that has to explain a low score.
    volume: float = 0.0
    size_factor: float = 1.0
    #: ``min(1, max(0, −ESP / E_ref))``: the electrostatic term as it entered the
    #: combination.  **Zero means the clamp swallowed it** — the interaction energy is
    #: not favourable, so the combined score is the shape term alone.  A caller that
    #: reports ``combined`` without checking this is reporting a shape score under an
    #: electrostatic label, which is exactly the defect `docs/PROTONATION.md` is about.
    esp_term: float = 0.0
    esp_term_used: bool = False
    esp_note: str = ""
    #: The pose that produced it (a 3x3 rotation and a translation), for a report.
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    n_poses: int = 0

    @property
    def clash_fraction(self) -> float:
        return (self.clash / self.n_atoms) if self.n_atoms else 0.0

    @property
    def is_shape_only(self) -> bool:
        """Whether the reported score is effectively shape alone."""
        return not self.esp_term_used

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shape": round(float(self.shape), 4),
            "esp": round(float(self.esp), 4),
            "combined": round(float(self.combined), 4),
            "contact": int(self.contact),
            "clash": int(self.clash),
            "exposed": int(self.exposed),
            "n_atoms": int(self.n_atoms),
            "volume": round(float(self.volume), 2),
            "size_factor": round(float(self.size_factor), 4),
            "esp_term": round(float(self.esp_term), 4),
            "esp_term_used": bool(self.esp_term_used),
            "esp_note": self.esp_note,
            "n_poses": int(self.n_poses),
        }


def _esp_flags(
    esp: float, *, electrostatic: bool, esp_reference: float
) -> Tuple[float, bool, str]:
    """``(esp_term, used, note)`` — the flag that stops a shape score hiding as an ESP one."""
    if not electrostatic:
        return 0.0, False, (
            "electrostatics were switched off by the caller, so the score is the shape "
            "term alone by request, not by accident"
        )
    if float(esp_reference) <= 0.0:
        return 0.0, False, "no electrostatic reference energy was given"
    term = max(0.0, min(1.0, -float(esp) / float(esp_reference)))
    if term <= 0.0:
        return 0.0, False, (
            f"the ligand-receptor interaction energy is {float(esp):+.2f} kcal/mol, so "
            "the clamp sent the electrostatic term to zero and this score is the SHAPE "
            "term alone.  Check the protonation state and the charge model: a formally "
            "charged group left neutral, or a Gasteiger charge on a charged group, both "
            "produce a positive energy in an oppositely charged pocket (see "
            "docs/PROTONATION.md)"
        )
    return term, True, ""


def _score_arrays(
    shape_values: np.ndarray,
    esp_values: np.ndarray,
    charges: np.ndarray,
    *,
    contact_margin: float,
    esp_reference: float,
    shape_weight: float,
    electrostatic: bool,
    size_factor: float = 1.0,
    objective=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(combined, shape, esp, contact, clash, exposed)`` for a batch of poses.

    ``shape_values`` and ``esp_values`` are ``(k, n)``: one row per candidate pose,
    one column per ligand heavy atom.  Everything is a sum along the atom axis, so a
    whole batch costs two array operations.

    The shape term is a **Gaussian contact kernel** rather than a count of contacts.
    The count saturates — measured on 3PTB, every atom of every docked benzamidine
    pose is "in contact", so the count is 1.000 for all of them and discriminates
    nothing — while the kernel rewards an atom sitting *snugly* (``F`` near
    :data:`CONTACT_OPTIMUM`) and falls away for one floating in the open.  A clashing
    atom contributes nothing and is penalised again through the clash fraction.
    ``size_factor`` multiplies the result: it is 1.0 unless the pocket carries a
    :data:`DEFAULT_SIZE_REFERENCE`, in which case it is the Gaussian size
    complementarity of the ligand against the pocket's known binder.

    ``objective`` is an optional **vectorised** hook that replaces the final combination,
    so a caller can put extra physics *inside the search* instead of re-scoring the winner
    afterwards.  It is called as ``objective(shape, esp_term, combined, shape_values)``
    with ``(k,)`` arrays and the ``(k, n)`` field values, and must return a ``(k,)`` array
    — higher is better, matching ``combined``.  With ``objective=None`` (the default) the
    result is **bit-identical to the combination above**, which is why adding the hook
    cannot change any existing score.  See `docs/FRAGMENTS.md` §5 for why a term that is
    not in the search objective cannot affect the search.
    """
    clashing = shape_values < -_CLASH_TOLERANCE
    in_contact = (~clashing) & (shape_values <= float(contact_margin))
    exposed = shape_values > float(contact_margin)
    kernel = np.exp(
        -((np.clip(shape_values, 0.0, None) - CONTACT_OPTIMUM) ** 2)
        / (2.0 * CONTACT_WIDTH ** 2)
    )
    kernel = np.where(clashing, 0.0, kernel)
    shape = kernel.mean(axis=1) * (1.0 - clashing.mean(axis=1)) * float(size_factor)
    contact = in_contact.sum(axis=1).astype(float)
    clash = clashing.sum(axis=1).astype(float)
    exposed_count = exposed.sum(axis=1).astype(float)
    # The interaction energy is always computed and reported, even when the caller
    # asked for the shape term alone: on this system it is the term that says the most
    # about the charges, and hiding it would hide the finding (docs/POCKET_SCORE.md
    # 3.1).  `electrostatic=False` changes the *combination*, not the diagnostic.
    esp = (esp_values * charges[None, :]).sum(axis=1)
    if electrostatic and float(esp_reference) > 0.0:
        esp_term = np.clip(-esp / float(esp_reference), 0.0, 1.0)
        combined = float(shape_weight) * shape + (1.0 - float(shape_weight)) * esp_term
    else:
        esp_term = np.clip(-esp / float(esp_reference), 0.0, 1.0) \
            if float(esp_reference) > 0.0 else np.zeros_like(esp)
        combined = shape
    if objective is not None:
        # The hook sees the same field values the terms were built from, so a penalty for
        # a deep overlap or a buried polar atom can be computed per pose without a second
        # sampling pass over the pocket.
        combined = np.asarray(
            objective(shape, esp_term, combined, shape_values), dtype=float
        ).reshape(-1)
    return combined, shape, esp, contact, clash, exposed_count


def pocket_score(
    coords: np.ndarray,
    charges: np.ndarray,
    pocket: PocketField,
    *,
    radii: Optional[Sequence[float]] = None,
    contact_margin: float = DEFAULT_CONTACT_MARGIN,
    esp_reference: float = DEFAULT_ESP_REFERENCE,
    shape_weight: float = SHAPE_WEIGHT,
    electrostatic: bool = True,
) -> PoseScore:
    """Score **one already-placed** ligand against the pocket field.

    This is the primitive: the ligand's coordinates are assumed to be in the
    receptor's frame (a docked pose, a crystal pose, a pose the search placed).  See
    :func:`place_and_score` for the version that finds the pose itself.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.shape[0] != coords.shape[0]:
        raise ValueError("each ligand atom needs exactly one charge")
    if coords.shape[0] == 0:
        return PoseScore(n_poses=1)
    shape_values, esp_values = pocket.sample(coords)
    volume = ligand_volume(
        radii if radii is not None else np.full(coords.shape[0], _radius_of("C"))
    )
    combined, shape, esp, contact, clash, exposed = _score_arrays(
        shape_values[None, :], esp_values[None, :], charges,
        contact_margin=contact_margin, esp_reference=esp_reference,
        shape_weight=shape_weight, electrostatic=electrostatic,
        size_factor=pocket.size_factor(volume),
    )
    score = PoseScore(
        shape=float(shape[0]), esp=float(esp[0]), combined=float(combined[0]),
        contact=int(contact[0]), clash=int(clash[0]), exposed=int(exposed[0]),
        n_atoms=int(coords.shape[0]), n_poses=1,
    )
    score.volume = volume
    score.size_factor = pocket.size_factor(volume)
    term, used, note = _esp_flags(
        score.esp, electrostatic=electrostatic, esp_reference=esp_reference
    )
    score.esp_term, score.esp_term_used, score.esp_note = term, used, note
    return score


# ---------------------------------------------------------------------------
# Finding the pose: the rigid-body search against the field
# ---------------------------------------------------------------------------


def _placement_seeds(ligand: np.ndarray, pocket: PocketField, samples: int, seed: int):
    """Deterministic starting poses: rotations from the overlay, centroid in the box."""
    rotations = _overlay._seed_rotations(
        np.zeros((1, 3)), ligand - ligand.mean(axis=0), samples=samples, seed=seed
    )
    # Anchor the centroid at the pocket's open-space centroid, computed from the
    # field's own voxels: the box centre is a hypothesis, the open space is measured.
    open_voxels = np.argwhere(pocket.shape_field > 0.0)
    if open_voxels.size:
        centre_voxel = open_voxels.mean(axis=0)
    else:  # pragma: no cover - a box entirely inside the receptor
        centre_voxel = np.asarray(pocket.shape, dtype=float) / 2.0
    centre = pocket.origin + centre_voxel * float(pocket.spacing)
    translations = np.repeat(centre[None, :], rotations.shape[0], axis=0)
    return rotations, translations


def place_and_score(
    coords: np.ndarray,
    charges: np.ndarray,
    pocket: PocketField,
    *,
    radii: Optional[Sequence[float]] = None,
    samples: int = 64,
    levels: int = 6,
    population: int = 48,
    restarts: int = 2,
    rotation_step: float = 0.5,
    translation_step: float = 1.5,
    seed: int = _overlay.DEFAULT_ROTATION_SEED,
    contact_margin: float = DEFAULT_CONTACT_MARGIN,
    esp_reference: float = DEFAULT_ESP_REFERENCE,
    shape_weight: float = SHAPE_WEIGHT,
    electrostatic: bool = True,
    objective=None,
) -> PoseScore:
    """Place a conformer in the pocket and score the best pose found.

    The search is the overlay's, with the pocket field as the objective: a
    deterministic set of start rotations with the ligand's centroid at the pocket's
    open-space centroid, then a multi-resolution perturbation refinement (random
    rotations of ``rotation_step · 2^-level`` radians and Gaussian translations of
    ``translation_step · 2^-level`` Å, the centroid clamped inside the box).  It is a
    rigid-body search against a *fixed* receptor: no induced fit, no flexible
    sidechains, and the answer is ``max over the poses that were tried``.

    ``objective`` is the optional vectorised hook of :func:`_score_arrays`, and it is how
    a caller puts extra physics **into the search** rather than re-scoring the winner
    afterwards: the pose kept is the one that maximises the hook, and the returned
    ``combined`` is that maximised value while ``shape`` and ``esp`` stay the raw terms.
    Absent, the score is exactly what it was.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    charges = np.asarray(charges, dtype=float).reshape(-1)
    if coords.shape[0] == 0:
        return PoseScore(n_poses=0)
    volume = ligand_volume(
        radii if radii is not None else np.full(coords.shape[0], _radius_of("C"))
    )
    size_factor = pocket.size_factor(volume)
    centred = coords - coords.mean(axis=0)
    rotations, translations = _placement_seeds(centred, pocket, samples, seed)
    rng = np.random.default_rng(int(seed))
    low = pocket.origin + 0.5
    high = pocket.origin + (np.asarray(pocket.shape) - 0.5) * float(pocket.spacing)

    def evaluate(rot_batch: np.ndarray, trans_batch: np.ndarray):
        moved = np.einsum("kij,nj->kni", rot_batch, centred) + trans_batch[:, None, :]
        moved = np.clip(moved, low[None, None, :], high[None, None, :])
        index = pocket._indices(moved.reshape(-1, 3)).reshape(moved.shape)
        shape_values = pocket.shape_field[index[..., 0], index[..., 1], index[..., 2]]
        esp_values = pocket.esp_field[index[..., 0], index[..., 1], index[..., 2]]
        return _score_arrays(
            shape_values, esp_values, charges, contact_margin=contact_margin,
            esp_reference=esp_reference, shape_weight=shape_weight,
            electrostatic=electrostatic, size_factor=size_factor,
            objective=objective,
        )

    combined, shape, esp, contact, clash, exposed = evaluate(rotations, translations)
    evaluated = int(rotations.shape[0])
    order = np.argsort(-combined)
    best = (
        float(combined[order[0]]), rotations[order[0]], translations[order[0]],
        float(shape[order[0]]), float(esp[order[0]]),
        int(contact[order[0]]), int(clash[order[0]]), int(exposed[order[0]]),
    )
    for start in order[: max(1, int(restarts))]:
        rotation = rotations[start].copy()
        translation = translations[start].copy()
        value = float(combined[start])
        for level in range(max(1, int(levels))):
            scale = float(rotation_step) * (0.5 ** level)
            perturbations = _overlay._small_rotations(scale, int(population), rng)
            candidates = np.einsum("kij,jl->kil", perturbations, rotation)
            offsets = translation + rng.normal(
                scale=max(0.05, float(translation_step) * (0.5 ** level)),
                size=(int(population), 3),
            )
            batch = evaluate(candidates, offsets)
            evaluated += int(candidates.shape[0])
            pick = int(np.argmax(batch[0]))
            if float(batch[0][pick]) > value + 1e-12:
                value = float(batch[0][pick])
                rotation, translation = candidates[pick], offsets[pick]
                if value > best[0]:
                    best = (
                        value, rotation, translation, float(batch[1][pick]),
                        float(batch[2][pick]), int(batch[3][pick]),
                        int(batch[4][pick]), int(batch[5][pick]),
                    )
    score = PoseScore(
        shape=best[3], esp=best[4], combined=best[0], contact=best[5], clash=best[6],
        exposed=best[7], n_atoms=int(coords.shape[0]), rotation=best[1],
        translation=best[2], n_poses=evaluated, volume=volume, size_factor=size_factor,
    )
    term, used, note = _esp_flags(
        score.esp, electrostatic=electrostatic, esp_reference=esp_reference
    )
    score.esp_term, score.esp_term_used, score.esp_note = term, used, note
    return score


# ---------------------------------------------------------------------------
# Ranking a library
# ---------------------------------------------------------------------------


@dataclass
class PocketRanking:
    """A library ranked by pocket complementarity, best pose over conformers."""

    entries: List[Tuple[str, float]] = field(default_factory=list)
    details: Dict[str, PoseScore] = field(default_factory=dict)
    seconds: float = 0.0
    n_poses: int = 0
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def names(self) -> List[str]:
        return [name for name, _ in self.entries]

    @property
    def scores(self) -> List[float]:
        return [score for _, score in self.entries]

    def rank_of(self, name: str) -> Optional[int]:
        for position, (entry, _) in enumerate(self.entries, start=1):
            if entry == name:
                return position
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": len(self.entries),
            "seconds": round(float(self.seconds), 4),
            "n_poses": int(self.n_poses),
            "entries": [[name, round(float(score), 6)] for name, score in self.entries],
            "details": {name: value.as_dict() for name, value in self.details.items()},
            "notes": list(self.notes),
        }

    def table(self, limit: int = 0) -> str:
        rows = self.entries if limit <= 0 else self.entries[: int(limit)]
        lines = [
            f"{'rank':<5}{'ligand':<30}{'score':>8}{'shape':>8}{'ESP':>9}"
            f"{'clash':>7}{'poses':>8}",
            "-" * 5 + "-" * 30 + "-" * 8 + "-" * 8 + "-" * 9 + "-" * 7 + "-" * 8,
        ]
        for rank, (name, score) in enumerate(rows, start=1):
            detail = self.details.get(name)
            lines.append(
                f"{rank:<5}{name[:29]:<30}{score:>8.3f}"
                f"{(detail.shape if detail else float('nan')):>8.3f}"
                f"{(detail.esp if detail else float('nan')):>9.2f}"
                f"{(detail.clash if detail else 0):>7}{(detail.n_poses if detail else 0):>8}"
            )
        if limit > 0 and len(self.entries) > limit:
            lines.append(f"... and {len(self.entries) - limit} more ligand(s)")
        return "\n".join(lines)


def rank_library(
    library: Sequence[Any],
    pocket: PocketField,
    *,
    conformers: Optional[int] = None,
    seed: int = _overlay.DEFAULT_SEED,
    names: Optional[Sequence[str]] = None,
    samples: int = 64,
    levels: int = 6,
    population: int = 48,
    restarts: int = 2,
    contact_margin: float = DEFAULT_CONTACT_MARGIN,
    esp_reference: float = DEFAULT_ESP_REFERENCE,
    shape_weight: float = SHAPE_WEIGHT,
    electrostatic: bool = True,
    formal_charge_correction: bool = False,
) -> PocketRanking:
    """Rank a library by the best pocket complementarity over its conformers.

    Conformers come from :func:`odock.overlay.prepare_molecule`, so a molecule that
    already carries 3-D coordinates is used as it is and ``conformers=None`` scales
    the attempt count with its rotatable bonds.  Every conformer is placed by
    :func:`place_and_score`; the molecule keeps its best.

    ``formal_charge_correction`` shifts each formally charged group's partial charges
    so the group sums to its formal charge (:func:`odock.protonation.
    correct_formal_charges`).  It is off by default and **measured to matter**: on 3PTB
    the crystal pose's interaction energy goes from +22.09 kcal/mol to −28.75 when the
    amidinium's +1 is restored, and the score from 0.471 to 0.971 — Gasteiger alone,
    even on the correctly protonated species, keeps the nitrogen negative and gives the
    wrong sign.
    """
    require_rdkit()
    from .protonation import can_represent_formal_charge, correct_formal_charges

    start = time.perf_counter()
    entries: List[Tuple[str, float]] = []
    details: Dict[str, PoseScore] = {}
    corrections: List[str] = []
    incapable: List[str] = []
    n_poses = 0
    failed: List[str] = []
    for index, item in enumerate(library):
        label = (
            str(names[index]) if names is not None and index < len(names) else ""
        ) or _overlay._mol_name(
            item if not isinstance(item, str) else Chem.MolFromSmiles(item),
            f"ligand_{index + 1}",
        )
        placed = _overlay.prepare_molecule(
            item, conformers=conformers, seed=seed, name=label
        )
        if placed.error or not placed.conformers:
            failed.append(f"{label}: {placed.error or 'no conformer'}")
            entries.append((label, 0.0))
            details[label] = PoseScore()
            continue
        # The heavy-atom radii, in the same order as the conformer coordinates and the
        # charges, so the size term sees the real van der Waals volume.
        radii = np.asarray(
            [
                _radius_of(atom.GetSymbol())
                for atom in placed.mol.GetAtoms()
                if atom.GetAtomicNum() > 1
            ],
            dtype=float,
        )
        best = PoseScore()
        charges = np.asarray(placed.charges, dtype=float)
        if formal_charge_correction:
            charges, shifted = correct_formal_charges(placed.mol, charges)
            if shifted:
                corrections.append(f"{label}: " + "; ".join(shifted))
        elif electrostatic:
            # The capability check on the charges as they are: a molecule whose
            # charged groups cannot carry their formal charge gets a note naming it,
            # so the ESP term it produces is not read as chemistry.  This is the entry
            # point every consumer should use (docs/PROTONATION.md §3).
            capability = can_represent_formal_charge(placed.mol, charges)
            if not capability.possible:
                incapable.append(f"{label}: {capability.reasons[0]}")
        for conformer in placed.conformers:
            score = place_and_score(
                conformer, charges, pocket, radii=radii, samples=samples,
                levels=levels, population=population, restarts=restarts, seed=seed,
                contact_margin=contact_margin, esp_reference=esp_reference,
                shape_weight=shape_weight, electrostatic=electrostatic,
            )
            n_poses += score.n_poses
            if score.combined > best.combined:
                best = score
        entries.append((label, float(best.combined)))
        details[label] = best
    entries.sort(key=lambda item: -item[1])
    shape_only = [name for name, _ in entries if not details[name].esp_term_used]
    notes = [
        f"best pose over each molecule's conformers, rigid-body search in the box "
        f"({samples + restarts * levels * population} poses per conformer at most)",
        "shape complementarity is not a binding energy and the receptor does not move",
    ]
    if shape_only:
        notes.append(
            f"the electrostatic term was clamped to zero for {len(shape_only)} of "
            f"{len(entries)} molecule(s), so for those the reported score is the shape "
            "term alone: the interaction energy is not favourable, which is what a "
            "neutral formal charge in an oppositely charged pocket looks like.  See "
            "docs/PROTONATION.md and correct_formal_charges()"
        )
    if corrections:
        notes.append(
            f"formal-charge correction applied to {len(corrections)} molecule(s): "
            + corrections[0]
            + (" ..." if len(corrections) > 1 else "")
        )
    if incapable:
        notes.append(
            f"{len(incapable)} molecule(s) carry a formally charged group whose charges "
            f"cannot represent it, so the ESP term for them is computed for a different "
            f"species: {incapable[0]}"
            + (" ..." if len(incapable) > 1 else "")
            + "  (can_represent_formal_charge / correct_formal_charges, "
            "docs/PROTONATION.md)"
        )
    if failed:
        notes.append(
            f"{len(failed)} molecule(s) could not be prepared and score 0: "
            + "; ".join(failed[:3])
        )
    return PocketRanking(
        entries=entries, details=details, seconds=time.perf_counter() - start,
        n_poses=n_poses, notes=notes,
    )


def prefilter(
    library: Sequence[Any],
    pocket: PocketField,
    actives: Sequence[str],
    *,
    keep: float = 0.05,
    exhaustiveness: int = 8,
    cores: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """What a pocket-score pre-filter keeps, and what it saves the docking run.

    The mirror of :func:`odock.lbvs.prefilter`: how many molecules survive the cut,
    **how many of the known binders** survive with them, and the docking workload
    before and after from :func:`odock.screen.estimate_cost`.  A fast structure-based
    filter that loses the actives is worse than a cheap 2-D one that keeps them, so
    the recall is reported next to the speedup, always.
    """
    require_rdkit()
    from .screen import estimate_cost

    ranking = rank_library(library, pocket, **kwargs)
    n_total = len(ranking)
    cut = max(1, int(math.ceil(float(keep) * n_total)))
    kept = {name for name, _ in ranking.entries[:cut]}
    known = [name for name in ranking.names if name in set(str(a) for a in actives)]
    survived = [name for name in known if name in kept]
    workload = estimate_cost(n_ligands=n_total, exhaustiveness=exhaustiveness, cores=cores)
    workload_kept = estimate_cost(n_ligands=cut, exhaustiveness=exhaustiveness, cores=cores)
    return {
        "method": "pocket",
        "keep": float(keep),
        "n_library": n_total,
        "n_kept": cut,
        "fraction_kept": cut / n_total if n_total else 0.0,
        "n_actives": len(known),
        "actives_kept": len(survived),
        "actives_lost": sorted(set(known) - set(survived)),
        "active_recall": (len(survived) / len(known)) if known else 0.0,
        "kept_names": [name for name, _ in ranking.entries[:cut]],
        "estimated_seconds_full": round(workload, 1),
        "estimated_seconds_kept": round(workload_kept, 1),
        "workload_saved_fraction": (1.0 - workload_kept / workload) if workload > 0 else 0.0,
        "ranking_seconds": round(float(ranking.seconds), 3),
        "poses_evaluated": int(ranking.n_poses),
        "notes": [
            "the docking cost model is odock.screen's own estimate; it is a planning "
            "figure, not a measurement",
            "the recall is of the molecules the caller named as known binders — on the "
            "bundled library those are the documented amidines, not assay hits",
        ],
    }


# ---------------------------------------------------------------------------
# Validation: does the score correlate with the docked affinity?
# ---------------------------------------------------------------------------


@dataclass
class SpearmanResult:
    """A rank correlation with the interval and the power statement."""

    rho: float = 0.0
    n: int = 0
    ci: Tuple[float, float] = (float("nan"), float("nan"))
    #: The smallest |rho| this n can resolve at 80 % power, two-sided 5 %.
    mdd: float = float("nan")
    resolvable: bool = False
    samples: int = 0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rho": round(float(self.rho), 4),
            "n": int(self.n),
            "ci": [round(float(self.ci[0]), 4), round(float(self.ci[1]), 4)],
            "mdd": round(float(self.mdd), 4),
            "resolvable": bool(self.resolvable),
            "samples": int(self.samples),
            "seconds": round(float(self.seconds), 4),
        }

    def statement(self) -> str:
        """One line a reader can quote, including whether it means anything."""
        if self.n < 3:
            return f"n = {self.n}: a rank correlation needs at least three points"
        verdict = (
            "resolvable" if self.resolvable
            else f"NOT resolvable at this n (needs |rho| >= {self.mdd:.2f})"
        )
        return (
            f"Spearman rho = {self.rho:+.3f} (n = {self.n}, 95% CI "
            f"[{self.ci[0]:+.2f}, {self.ci[1]:+.2f}], minimum detectable |rho| = "
            f"{self.mdd:.2f}): {verdict}"
        )


def spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    *,
    samples: int = 2000,
    level: float = 0.95,
    seed: int = 20240101,
) -> SpearmanResult:
    """Spearman's rho with a bootstrap interval and the detectable effect size.

    The interval is a percentile bootstrap over **pairs**, and ``mdd`` is the
    smallest ``|rho|`` the sample size can distinguish from zero at 80 % power — for
    a rank correlation that is the Fisher-transformed criterion
    ``tanh((z_{1−α/2} + z_{power}) / sqrt(n − 3))``, computed rather than asserted.
    A point estimate whose interval spans zero is not a result, and this type says so
    in :meth:`SpearmanResult.statement`.
    """
    import random as _random
    from statistics import NormalDist

    started = time.perf_counter()
    left = np.asarray(x, dtype=float).reshape(-1)
    right = np.asarray(y, dtype=float).reshape(-1)
    if left.shape[0] != right.shape[0]:
        raise ValueError("x and y must have the same length")
    n = int(left.shape[0])
    if n < 3:
        return SpearmanResult(rho=float("nan"), n=n, seconds=time.perf_counter() - started)

    def rho_of(a: np.ndarray, b: np.ndarray) -> float:
        # Average ranks handle ties the way scipy's 'average' method does.
        ranks_a = _average_ranks(a)
        ranks_b = _average_ranks(b)
        return _pearson(ranks_a, ranks_b)

    observed = rho_of(left, right)
    rng = _random.Random(int(seed))
    values: List[float] = []
    for _ in range(max(1, int(samples))):
        picks = [rng.randrange(n) for _ in range(n)]
        value = rho_of(np.asarray([left[i] for i in picks]),
                       np.asarray([right[i] for i in picks]))
        if math.isfinite(value):
            values.append(value)
    tail = (1.0 - float(level)) / 2.0
    values.sort()
    low = values[max(0, int(math.floor(tail * len(values))))] if values else float("nan")
    high = values[min(len(values) - 1, int(math.ceil((1.0 - tail) * len(values))) - 1)] \
        if values else float("nan")
    if n > 3:
        z = NormalDist().inv_cdf(1.0 - tail) + NormalDist().inv_cdf(0.8)
        mdd = math.tanh(z / math.sqrt(n - 3))
    else:  # pragma: no cover - n == 3 has no residual degrees of freedom
        mdd = float("nan")
    return SpearmanResult(
        rho=float(observed), n=n, ci=(float(low), float(high)), mdd=float(mdd),
        resolvable=bool(low > 0.0 or high < 0.0), samples=len(values),
        seconds=time.perf_counter() - started,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    sorted_values = values[order]
    position = 0
    while position < values.shape[0]:
        end = position
        while end + 1 < values.shape[0] and sorted_values[end + 1] == sorted_values[position]:
            end += 1
        ranks[order[position:end + 1]] = (position + end) / 2.0 + 1.0
        position = end + 1
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = a - a.mean()
    right = b - b.mean()
    denominator = float(np.sqrt((left * left).sum() * (right * right).sum()))
    return float((left * right).sum() / denominator) if denominator > 0 else 0.0


def spearman(
    x: Sequence[float], y: Sequence[float], *, samples: int = 1, **kwargs: Any
) -> float:
    """Spearman's rho alone, for a caller that wants only the number."""
    return spearman_ci(x, y, samples=samples, **kwargs).rho

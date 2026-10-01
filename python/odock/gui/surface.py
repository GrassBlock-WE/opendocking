# SPDX-License-Identifier: GPL-3.0-or-later
"""A real molecular surface, and the properties it is painted with.

Why not a point cloud
---------------------
The workbench's ``dots`` style samples atom spheres and throws most of them
away, which is a *picture of atoms*, not a surface: it has no interior, no
normals, cannot be lit, cannot be clipped and cannot be measured. What a
docking tool actually needs is the boundary of the volume the protein presents
to solvent, with a property on it, because the picture that answers "why does
the ligand sit here?" is a **buried pocket coloured by hydrophobicity**.

Two surfaces, both derived from one field
----------------------------------------
Let the atoms carry Bondi van der Waals radii ``r_i`` and let the probe be a
sphere of radius :data:`odock.sasa.PROBE_RADIUS` (1.4 Å, water). Define the
**probe-centre field**

    F(p) = min_i ( |p - c_i| - (r_i + probe) )

which is negative exactly where the centre of a water molecule would overlap
an atom. Its zero level set is the **solvent-accessible surface (SAS)** —the
surface the probe centre traces, the same surface :mod:`odock.sasa` integrates
analytically with Shrake-Rupley. Triangulating ``F = 0`` with marching
tetrahedra therefore gives a SAS mesh whose area can be checked against the
Shrake-Rupley number, and the test suite does exactly that.

The **solvent-excluded surface (SES, the "molecular surface")** is the
boundary of the region a water molecule can never occupy: a point is outside
the SES when a probe sphere of radius ``probe`` can be placed so that it
contains the point while touching no atom. Written as an erosion of the
probe-centre field that is

    K(p) = max_{|u| = 1} F(p + probe · u)

—the point is outside the SES when *some* probe position that covers it is
legal. ``K = 0`` is triangulated the same way. This is an honest derivation,
but it is *numerical*: ``u`` is sampled on a golden-spiral lattice and the
shifted field is read from the grid by trilinear interpolation, so the
reentrant (concave) patches of the SES are resolved to roughly the grid
spacing. The contact patches —everything a chemist looks at —are exact,
and the SES area is always smaller than the SAS area, which the tests pin.

Grid, cost and the 2 000-atom case
----------------------------------
Both fields live on a uniform grid spanning the atoms plus a probe-width
margin. The spacing is chosen so a normal protein stays inside
:data:`DEFAULT_MAX_POINTS`; a caller may set it. Everything is NumPy over flat
arrays of (grid point, atom) pairs built from a uniform cell list, so the work
is linear in the atom count, and the caller can hand a ``progress`` callback
in. :func:`build_surface` reports the wall time, the grid size, the triangle
count and the area in ``Surface.stats`` —measured numbers rather than
reassurances.

Properties, and why *these* properties
-------------------------------------
* **Hydrophobicity** —per-atom, from a documented scale. An atom in a
  standard residue carries the **Kyte & Doolittle (1982)** hydropathy of its
  residue, normalised onto ``[0, 1]`` between the published extremes
  (Arg -4.5 →0.0, Ile +4.5 →1.0). An atom whose residue is not in that
  table —a ligand, a cofactor, an ion —carries the fragment value of its
  AutoDock 4 atom type (:data:`AD4_TYPE_HYDROPHOBICITY`), or, when the bonds
  are known, the **Vina ``xs_is_hydrophobic`` rule**: a carbon with no polar
  neighbour is apolar, one bonded to N/O/S is not.
* **Electrostatic potential** —sampled *on the surface* rather than carried
  from the atoms: the Coulomb potential of every partial charge at each
  surface vertex. The charges are the ones the project already computes —  ``odock.chem.charges.gasteiger_charges`` (or the merged Kollman scheme)
  written into the PDBQT, which the viewer reads back as ``Atom.charge``.
  This is a *picture* of the field, not the scoring function: the module
  documents the dielectric it uses and the caller can change it.
* **Element** —the CPK colours, so the surface can be checked against every
  other viewer.

A vertex takes the property of its **nearest atom**: on a molecular surface
each patch lies on exactly one atom's sphere, so the assignment is the same
Voronoi rule every viewer uses, and it is what makes a residue's patch a
single flat colour.

Honest limitations
------------------
* The SES reentrant patches are grid-resolved (above).
* The electrostatic map is a vacuum/continuum Coulomb picture; it is not
  solved on a grid with a Poisson-Boltzmann solver, and the module says so
  wherever it is shown.
* A surface built from a subset of atoms ("pocket lining only") is an open
  shell —the colours and the area are still right, but the mesh has a rim.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..sasa import (
    PROBE_RADIUS,
    radius_of,
    sasa_of_atoms,
    sphere_points,
)

__all__ = [
    "AD4_TYPE_HYDROPHOBICITY",
    "COULOMB_CONSTANT",
    "ELEMENT_HYDROPHOBICITY",
    "HYDROPATHY_KD",
    "PALETTES",
    "PROPERTY_LABELS",
    "SURFACE_MODES",
    "SURFACE_PROPERTIES",
    "Surface",
    "SurfaceSettings",
    "atom_hydrophobicity",
    "build_surface",
    "charges_from_atoms",
    "charges_from_mol",
    "colorize",
    "electrostatic_potential",
    "hydrophobicity_scale",
    "legend_stops",
    "marching_tetrahedra",
    "nearest_atoms",
    "palette_color",
    "resolve_spacing",
    "sample_field",
    "scalar_field",
    "select_atoms",
    "surface_area",
]


# ---------------------------------------------------------------------------
# the property scales
# ---------------------------------------------------------------------------

#: Kyte & Doolittle (1982) hydropathy, in the published units (a normalised
#: transfer free energy, octanol →water): *J. Mol. Biol.* **157**, 105.
#: Positive is hydrophobic. Every value here is the published one.
HYDROPATHY_KD: Dict[str, float] = {
    "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
    "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
    # The nucleic acids and the common modified residues, mapped from the
    # published scales of the same family so a DNA/RNA structure is not blank.
    "DA": -0.9, "DC": -3.5, "DG": -3.5, "DT": -0.9, "A": -0.9, "C": -3.5,
    "G": -3.5, "U": -0.9, "ADE": -0.9, "CYT": -3.5, "GUA": -3.5, "THY": -0.9,
    "SEP": -3.5, "TPO": -3.5, "PTR": -3.5, "MSE": 1.9, "CSO": 2.5,
    "HOH": 0.0, "WAT": 0.0, "DOD": 0.0,
}

#: The published extremes of :data:`HYDROPATHY_KD`, used to normalise it onto
#: ``[0, 1]``. They are the values of arginine and isoleucine, which is where
#: the scale is defined.
KD_MIN = -4.5
KD_MAX = 4.5

#: The fragment value of an AutoDock 4 atom type when the residue is unknown:
#: the apolar character of the atom as the AD4 dictionary sees it. The
#: classification is the one ``crates/dock-core/src/atom.rs`` implements
#: (``ad_type_property`` plus Vina's ``xs_is_hydrophobic``): aliphatic and
#: aromatic carbon and the halogens are the hydrophobic types, nitrogen and
#: oxygen the polar ones, sulfur and phosphorus in between, and every metal
#: or untyped atom is neutral because an ion is not a hydrophobe.
AD4_TYPE_HYDROPHOBICITY: Dict[str, float] = {
    "C": 0.85,   # aliphatic carbon
    "A": 0.85,   # aromatic carbon
    "CG0": 0.85, "CG1": 0.85, "CG2": 0.85, "CG3": 0.85,
    "S": 0.55,   # sulfur that is not an acceptor (thioether, disulfide)
    "SA": 0.35,  # H-bond-accepting sulfur (thiol)
    "P": 0.35,
    "F": 0.45, "Cl": 0.55, "Br": 0.55, "I": 0.55,
    "N": 0.20,   # amide / ammonium nitrogen
    "NA": 0.10,  # H-bond-accepting nitrogen
    "O": 0.05,   # carbonyl / hydroxyl oxygen
    "OA": 0.05,
    "HD": 0.05,  # polar hydrogen
    "H": 0.60,   # hydrogen with no AD4 type (non-polar, united-atom merged)
    "W": 0.50,   # untyped
    "Si": 0.45, "At": 0.55,
    "Mg": 0.00, "Mn": 0.00, "Zn": 0.00, "Ca": 0.00, "Fe": 0.00,
}

#: Per-element fallback for the *atom* scale, used when neither the residue
#: nor the AD4 type says anything. Values are the same fragment character as
#: :data:`AD4_TYPE_HYDROPHOBICITY`, indexed by element.
ELEMENT_HYDROPHOBICITY: Dict[str, float] = {
    "H": 0.60, "C": 0.85, "N": 0.15, "O": 0.05, "F": 0.45, "P": 0.35,
    "S": 0.50, "Cl": 0.55, "Br": 0.55, "I": 0.55, "Si": 0.45, "B": 0.30,
    "Se": 0.50, "As": 0.40,
    "Mg": 0.0, "Mn": 0.0, "Zn": 0.0, "Ca": 0.0, "Fe": 0.0, "Na": 0.0,
    "K": 0.0, "Cu": 0.0, "Ni": 0.0, "Co": 0.0,
}

#: Coulomb's constant in the units the rest of the project uses:
#: kcal·Å / (mol · e²).
COULOMB_CONSTANT = 332.0637

#: The surface flavours, and the properties a surface can be painted with.
SURFACE_MODES: Tuple[str, ...] = ("sas", "ses")
SURFACE_PROPERTIES: Tuple[str, ...] = ("hydrophobicity", "electrostatic", "element")

#: ``property -> (label, unit)`` for the legend and the exported files.
PROPERTY_LABELS: Dict[str, Tuple[str, str]] = {
    "hydrophobicity": ("hydrophobicity", "0 polar →1 apolar"),
    "electrostatic": ("electrostatic potential", "kcal/(mol·e)"),
    "element": ("element", ""),
}


def _residue_name(atom) -> str:
    return str(getattr(atom, "res_name", "") or "").strip().upper()


def _ad4_type(atom) -> str:
    return str(getattr(atom, "ad_type", "") or "").strip()


def _element(atom) -> str:
    text = str(getattr(atom, "element", "") or "").strip()
    if not text:
        return "C"
    return text[0].upper() + text[1:].lower() if len(text) > 1 else text.upper()


def hydrophobicity_scale(scale: str = "residue") -> str:
    """Canonical name of a hydrophobicity scale (``"residue"`` or ``"atom"``)."""
    key = str(scale).strip().lower()
    if key in ("residue", "kd", "kyte", "kyte-doolittle", "hydropathy"):
        return "residue"
    if key in ("atom", "element", "fragment", "vina"):
        return "atom"
    raise ValueError(f"unknown hydrophobicity scale {scale!r}; expected 'residue' or 'atom'")


def atom_hydrophobicity(
    atoms,
    *,
    scale: str = "residue",
    bonds: Optional[Sequence] = None,
) -> np.ndarray:
    """Per-atom hydrophobicity on ``[0, 1]``, from a documented scale.

    ``scale="residue"`` (default)
        Kyte & Doolittle hydropathy of the atom's residue, normalised so that
        Arg ``-4.5`` is ``0.0`` and Ile ``+4.5`` is ``1.0``. Every atom of a
        residue shares the value, which is what makes a residue's patch on the
        surface a single flat colour. A residue outside the table (a ligand, a
        cofactor, an ion, water) falls back to
        :data:`AD4_TYPE_HYDROPHOBICITY`.
    ``scale="atom"``
        The fragment value of the atom's AutoDock 4 type, with the element
        table behind it. When ``bonds`` are supplied the Vina rule is applied
        on top: a carbon with a polar neighbour (N, O or S) is *not* apolar —        which is exactly the ``CH``/``C`` split ``xs_is_hydrophobic`` makes —        so its value is averaged towards the polar end.
    """
    key = hydrophobicity_scale(scale)
    atoms = list(atoms)
    out = np.empty(len(atoms), dtype=float)
    for index, atom in enumerate(atoms):
        value = AD4_TYPE_HYDROPHOBICITY.get(_ad4_type(atom))
        if value is None:
            value = ELEMENT_HYDROPHOBICITY.get(_element(atom), 0.5)
        if key == "residue":
            raw = HYDROPATHY_KD.get(_residue_name(atom))
            if raw is not None:
                value = (raw - KD_MIN) / (KD_MAX - KD_MIN)
        out[index] = min(1.0, max(0.0, float(value)))

    if key == "atom" and bonds is not None:
        out = _apply_vina_hydrophobic_rule(atoms, out, bonds)
    return out


#: The elements that make a neighbouring carbon polar.
_POLAR_ELEMENTS = frozenset({"N", "O", "S"})


def _apply_vina_hydrophobic_rule(atoms, values: np.ndarray, bonds) -> np.ndarray:
    """Pull a carbon that is bonded to N/O/S away from the apolar end.

    Vina's ``xs_is_hydrophobic`` is true only for ``CH``/``FH``/``ClH``/``BrH``/
    ``IH``: a carbon counts as hydrophobic only when it carries no polar
    neighbour. The AD4 dictionary cannot express that (a carboxyl carbon and a
    methyl carbon are both ``C``), so the rule is applied here from the
    perceived bond graph.
    """
    neighbours: Dict[int, List[int]] = {}
    for bond in bonds or ():
        try:
            if hasattr(bond, "a") and hasattr(bond, "b"):
                first, second = int(bond.a), int(bond.b)
            else:
                first, second = int(bond[0]), int(bond[1])
        except (TypeError, IndexError, KeyError, ValueError):  # pragma: no cover
            continue
        if not (0 <= first < len(atoms) and 0 <= second < len(atoms)):
            continue
        neighbours.setdefault(first, []).append(second)
        neighbours.setdefault(second, []).append(first)

    out = np.array(values, dtype=float, copy=True)
    for index, atom in enumerate(atoms):
        if _element(atom) != "C":
            continue
        partners = neighbours.get(index, ())
        polar = [
            ELEMENT_HYDROPHOBICITY.get(_element(atoms[j]), 0.5)
            for j in partners
            if _element(atoms[j]) in _POLAR_ELEMENTS
        ]
        if not polar:
            continue
        # The Vina rule is a *binary* test; the picture reads better if the
        # change is graduated by how many polar neighbours there are, and the
        # limit is stated: one polar neighbour already makes it a non-CH carbon.
        pull = 0.55
        out[index] = min(out[index], (1.0 - pull) * out[index] + pull * min(polar))
    return out


# ---------------------------------------------------------------------------
# palettes
# ---------------------------------------------------------------------------

#: ``palette -> ((t, (r, g, b)), ...)`` with ``t`` running 0 →1 over the
#: value range. Interpolation is linear in sRGB, which is what every viewer
#: does for a colour ramp and what makes a legend stop readable.
PALETTES: Dict[str, Tuple[Tuple[float, Tuple[float, float, float]], ...]] = {
    # Hydrophilic blue →neutral →hydrophobic gold. The three stops are the
    # ones the field settled on decades ago (PyMOL's `color_hydrophobic` uses
    # the same blue/orange opposition) and they stay distinguishable at the
    # size a legend can be read.
    "hydrophobicity": (
        (0.00, (0.13, 0.31, 0.78)),
        (0.28, (0.28, 0.66, 0.90)),
        (0.50, (0.86, 0.86, 0.82)),
        (0.72, (0.94, 0.72, 0.24)),
        (1.00, (0.86, 0.36, 0.06)),
    ),
    # The electrostatic convention: positive blue, zero white, negative red.
    "electrostatic": (
        (0.00, (0.86, 0.16, 0.14)),
        (0.30, (0.94, 0.62, 0.50)),
        (0.50, (0.94, 0.94, 0.94)),
        (0.70, (0.42, 0.66, 0.92)),
        (1.00, (0.10, 0.25, 0.80)),
    ),
    # A grey ramp, for a caller that wants the property in the geometry only.
    "grey": (
        (0.00, (0.12, 0.12, 0.14)),
        (1.00, (0.95, 0.95, 0.95)),
    ),
}


def palette_color(palette: str, fraction: float) -> Tuple[float, float, float]:
    """One colour of a palette at ``fraction`` in ``[0, 1]``."""
    stops = PALETTES.get(str(palette)) or PALETTES["hydrophobicity"]
    t = min(1.0, max(0.0, float(fraction)))
    for index in range(len(stops) - 1):
        left_t, left = stops[index]
        right_t, right = stops[index + 1]
        if t <= right_t or index == len(stops) - 2:
            span = right_t - left_t
            local = 0.0 if span <= 0 else (t - left_t) / span
            local = min(1.0, max(0.0, local))
            return tuple(
                float(left[channel] + local * (right[channel] - left[channel]))
                for channel in range(3)
            )
    return tuple(float(c) for c in stops[-1][1])  # pragma: no cover - defensive


def colorize(
    values: np.ndarray,
    *,
    palette: str = "hydrophobicity",
    value_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Map values onto colours; returns ``((n, 3), (low, high))``.

    ``value_range`` is returned as well because a caller that did not supply
    one gets the range the data implies, and the legend has to show *that*
    range rather than the one the caller thought it asked for.
    """
    data = np.asarray(values, dtype=float).reshape(-1)
    if data.size == 0:
        return np.zeros((0, 3), dtype=float), (0.0, 1.0)
    if value_range is None:
        low = float(np.min(data))
        high = float(np.max(data))
    else:
        low, high = (float(value_range[0]), float(value_range[1]))
    if not math.isfinite(low) or not math.isfinite(high):
        low, high = 0.0, 1.0
    if high - low < 1e-12:
        # A constant property is a flat colour: pin the range so the caller
        # cannot then produce a division by zero when it scales the legend.
        low, high = low - 0.5, high + 0.5
    fraction = np.clip((data - low) / (high - low), 0.0, 1.0)

    stops = PALETTES.get(str(palette)) or PALETTES["hydrophobicity"]
    palette_t = np.array([stop[0] for stop in stops], dtype=float)
    palette_c = np.array([stop[1] for stop in stops], dtype=float)
    colors = np.empty((fraction.size, 3), dtype=float)
    for channel in range(3):
        colors[:, channel] = np.interp(fraction, palette_t, palette_c[:, channel])
    return colors, (low, high)


def legend_stops(
    palette: str,
    value_range: Tuple[float, float],
    *,
    count: int = 6,
    unit: str = "",
) -> List[Tuple[float, Tuple[float, float, float], str]]:
    """``(fraction, colour, label)`` entries for a legend drawn by the host.

    The labels are **numbers only**: a colour bar's job is to state the values
    of its ticks, and repeating the unit on every tick (``-36.08 kcal/(mol·e)``
    six times down the side of a figure) makes the one thing a reader needs —
    the shape of the scale — harder to see. The unit belongs in the title, which
    the host draws from :data:`PROPERTY_LABELS`, and it is passed here so the
    tick precision can follow it (a potential in kcal/mol needs one decimal, a
    normalised hydrophobicity two).
    """
    low, high = float(value_range[0]), float(value_range[1])
    magnitude = max(abs(low), abs(high))
    digits = 2 if magnitude < 10.0 else (1 if magnitude < 100.0 else 0)
    out = []
    for index in range(int(count)):
        fraction = index / max(1, count - 1)
        value = low + fraction * (high - low)
        out.append(
            (fraction, palette_color(palette, fraction), f"{value:.{digits}f}")
        )
    del unit
    return out


# ---------------------------------------------------------------------------
# per-vertex electrostatics
# ---------------------------------------------------------------------------


def charges_from_atoms(atoms) -> np.ndarray:
    """The partial charges the viewer already has, one per atom.

    A PDBQT carries the partial charge column, which is what
    ``odock.chem.charges.gasteiger_charges`` (or the Kollman united-atom
    scheme) wrote during preparation, and :class:`odock.gui.structure.Atom`
    keeps it as ``.charge``. Reading it back here is what makes the surface
    potential consistent with the score: the same charges the force field
    sees, not a second opinion.
    """
    out = np.zeros(len(atoms), dtype=float)
    for index, atom in enumerate(atoms):
        try:
            value = float(getattr(atom, "charge", 0.0) or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            value = 0.0
        out[index] = value if math.isfinite(value) else 0.0
    return out


def charges_from_mol(mol, model: str = "gasteiger") -> np.ndarray:
    """Charges for an RDKit molecule, via :mod:`odock.chem.charges`.

    Imported lazily so the viewer keeps working on an installation without
    RDKit: only a caller that has a molecule to charge reaches this.

    **Read :func:`charge_caveat` before painting a surface with these.** A
    Gasteiger column is a sigma-electronegativity model: it cannot hold a formal
    charge, so a formally cationic amidine still comes out with a *negative*
    nitrogen and a formally anionic carboxylate is only mildly negative. A map
    coloured by such a column has the right shape and the wrong sign in the
    places that matter most.
    """
    from ..chem.charges import assign_charges  # local: RDKit is optional

    return np.asarray(assign_charges(mol, model=model), dtype=float)


def charge_quality(atoms, charges=None) -> Dict[str, object]:
    """What a per-atom charge set can and cannot say, without RDKit.

    Two checks that need nothing but the numbers:

    * the **total**. A protein is not -48 electrons, and a charge model that
      sums to tens of electrons per molecule is a *relative* picture. The
      threshold is deliberately loose - more than one electron of net charge per
      fifty atoms is already impossible for a neutral-except-for-a-few-residues
      structure;
    * the **range**, so a column that is entirely zero (the documented failure
      mode of Gasteiger on an unsanitised protein) is visible as such.

    What it *cannot* do is know whether a formally charged group is missing its
    formal charge - that needs the chemistry, which is why
    :func:`charge_caveat` takes the molecule when one is available.
    """
    table = (
        charges_from_atoms(atoms)
        if charges is None
        else np.asarray(charges, dtype=float).reshape(-1)
    )
    count = len(table)
    total = float(table.sum()) if count else 0.0
    warnings: List[str] = []
    if count and not np.any(table):
        warnings.append("every charge is zero")
    elif count and abs(total) > max(1.0, count / 50.0):
        warnings.append(
            f"the charges sum to {total:+.1f} e over {count} atoms, "
            f"which is not a net charge this structure can have"
        )
    return {
        "atoms": count,
        "nonzero": int(np.count_nonzero(table)),
        "total": total,
        "minimum": float(table.min()) if count else 0.0,
        "maximum": float(table.max()) if count else 0.0,
        "absolute_mean": float(np.abs(table).mean()) if count else 0.0,
        "warnings": warnings,
    }


def charge_caveat(
    atoms,
    charges=None,
    mol=None,
    receptor_atoms=(),
    ligand_coords=None,
    *,
    ph: float = 7.4,
) -> Optional[str]:
    """One sentence a surface legend can carry, or ``None`` when all is well.

    This is the honest half of an electrostatic map. An ESP picture is only as
    good as the charge column behind it, and the two ways that column lies are
    both detectable:

    * it does not carry the **formal charge** of a group — a formally cationic
      amidine whose nitrogens are *negative* because Gasteiger spreads the +1
      over the whole ion. :func:`odock.protonation.detect_groups` names the
      family, and the charge array says whether the atoms hold the charge;
      :func:`odock.protonation.salt_bridge_warnings` adds the geometric
      contradiction (a neutral amidine 2.9 Å from an aspartate).
    * it does not **conserve** charge at all — the hydrogen-suppressed
      Gasteiger column on the bundled demo receptor sums to tens of electrons.

    Neither the detector nor the chemistry is re-implemented here: the
    protonation module owns them and is imported lazily, so a caller without
    RDKit still gets the conservation half. ``ligand_coords`` must be in the
    order the *molecule* expects; the protonation API raises on a mismatch
    rather than silently measuring a different geometry, which is why nothing
    here reshapes it.
    """
    notes: List[str] = []
    quality = charge_quality(atoms, charges)
    notes.extend(quality["warnings"])
    if mol is None:
        return "; ".join(notes) if notes else None
    try:
        from .. import protonation  # local: RDKit is optional
    except Exception:  # pragma: no cover - an installation without RDKit
        return "; ".join(notes) if notes else None

    table = (
        charges_from_atoms(atoms)
        if charges is None
        else np.asarray(charges, dtype=float).reshape(-1)
    )
    try:
        groups = protonation.detect_groups(mol)
    except Exception:  # pragma: no cover - defensive
        groups = []
    for group in groups:
        expected = int(getattr(group, "expected_charge", 0) or 0)
        if expected == 0:
            continue
        indices = [int(index) for index in getattr(group, "atoms", ())]
        if not indices or max(indices) >= table.size:
            continue
        held = float(table[indices].sum())
        if abs(held) < 0.5 * abs(expected) or (held * expected) < 0:
            notes.append(
                f"{getattr(group, 'label', group)} is a {getattr(group, 'family', '')} "
                f"that should carry {expected:+d} e but holds {held:+.2f} e "
                f"in this charge set"
            )
    if receptor_atoms:
        try:
            for warning in protonation.salt_bridge_warnings(
                mol,
                list(receptor_atoms),
                ligand_coords=ligand_coords,
                charges=table,
                ph=ph,
            ):
                notes.append(str(warning))
        except Exception as exc:  # pragma: no cover - the API raises on purpose
            notes.append(f"charge geometry could not be checked ({exc})")
    # The same family can match several atoms; say it once.
    unique = list(dict.fromkeys(text for text in notes if text))
    return "; ".join(unique) if unique else None


def electrostatic_potential(
    points,
    atoms,
    charges=None,
    *,
    dielectric: str = "uniform",
    epsilon: float = 4.0,
    screening: float = 0.0,
    minimum_distance: float = 1.0,
) -> np.ndarray:
    """The Coulomb potential of ``atoms`` at every point of ``points``.

    ``phi(p) = k · Σ_i q_i · exp(-screening · r_ip) / (eps(r_ip) · r_ip)`` with
    ``k = 332.0637 kcal·Å/(mol·e²)`` and ``r_ip`` the distance from the point
    to atom ``i``.

    ``dielectric``
        ``"uniform"`` (default): ``eps = epsilon``, the usual choice for a
        *picture* —it shows the shape of the field rather than the screening.
        ``"distance"``: ``eps(r) = epsilon · r``, which is the AutoDock 4
        convention the scoring kernel uses; the potential then falls off as
        ``1/r²`` and the picture is much more local. Both are documented here
        so the number on the legend can be read for what it is.
    ``screening``
        Debye-Hückel ``kappa`` in 1/Å for an ionic-strength-aware picture
        (``kappa = 0`` disables it). ``kappa ≈0.104 sqrt(I)`` at 298 K for an
        ionic strength ``I`` in mol/L.
    ``minimum_distance``
        Distances are floored at this value in Å. A surface vertex is never
        closer than a van der Waals contact to a *non-owning* atom, so this
        only ever guards a caller that samples somewhere pathological.

    The result is in kcal/(mol·e) —the same energy unit as the score —which
    is stated on the legend.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    count = len(atoms)
    values = np.zeros(pts.shape[0], dtype=float)
    if pts.size == 0 or count == 0:
        return values
    table = (
        charges_from_atoms(atoms)
        if charges is None
        else np.asarray(charges, dtype=float).reshape(-1)
    )
    if table.size != count:
        raise ValueError(f"charges has {table.size} entries for {count} atoms")
    if not np.any(table):
        return values

    coords = np.asarray(
        [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
    ).reshape(-1, 3)
    key = str(dielectric).strip().lower()
    if key not in ("uniform", "distance"):
        raise ValueError(f"dielectric must be 'uniform' or 'distance', got {dielectric!r}")

    # Chunked so the (points x atoms x 3) temporary stays bounded on a
    # 3 000-atom receptor.
    chunk = max(1, int(4_000_000 / max(count, 1)))
    for start in range(0, len(pts), chunk):
        block = pts[start : start + chunk]
        delta = block[:, None, :] - coords[None, :, :]
        distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        np.maximum(distance, float(minimum_distance), out=distance)
        scale = float(epsilon) * distance if key == "distance" else np.full_like(
            distance, float(epsilon)
        )
        contribution = (COULOMB_CONSTANT * table)[None, :] / (scale * distance)
        if screening:
            contribution *= np.exp(-float(screening) * distance)
        values[start : start + chunk] = contribution.sum(axis=1)
    return values


# ---------------------------------------------------------------------------
# the grid and the field
# ---------------------------------------------------------------------------

#: Defaults for the grid. The spacing is the resolution of the mesh; the point
#: cap bounds what an automatic spacing may allocate.
#:
#: **The default spacing is chosen from a measurement, not from convenience.**
#: ``tools/surface_convergence.py`` builds the bundled 1 994-atom receptor at
#: every setting and reports the grid, the triangle count, the mesh area, its
#: error against the analytic Shrake-Rupley integral (9 274.9 A2) and the wall
#: time:
#:
#: =========  ===========  ==========  ========  =======  ========
#: spacing A  grid points  triangles   area A2   error %  seconds
#: =========  ===========  ==========  ========  =======  ========
#: 1.20         105 092       52 152    8 473.2   -8.64     2.21
#: 1.00         173 600       76 600    8 588.5   -7.40     3.49
#: 0.80         315 248      121 592    8 705.5   -6.14     5.49
#: 0.65         565 064      186 652    8 804.5   -5.07     9.21
#: 0.50       1 188 260      319 624    8 902.0   -4.02    20.54
#: 0.40       2 237 742      503 848    8 974.2   -3.24    24.37
#: =========  ===========  ==========  ========  =======  ========
#:
#: The triangulation inscribes the true surface, so the area is always low and
#: closes slowly — the crevices between atoms are where it loses most. 0.65 A is
#: the knee of that table: the largest accuracy step short of the 0.5 A setting,
#: which costs 2.2x the time and 1.7x the triangles for one more point of error.
#: 0.8 A stays available as the quick-look setting, and ``spacing = 0`` asks the
#: builder to choose from the point budget instead (0.63 A for this receptor).
DEFAULT_SPACING = 0.65
MIN_SPACING = 0.32
MAX_SPACING = 1.6
DEFAULT_MAX_POINTS = 700_000
DEFAULT_DIRECTIONS = 42


@dataclass
class SurfaceSettings:
    """Everything that decides what a surface looks like."""

    #: A one-sentence caveat about the charge column, written by whoever knows
    #: the chemistry (see :func:`charge_caveat`). It is carried into
    #: ``Surface.charge_warning`` and into the stats, so the legend and the log
    #: can say "this map may be wrong-signed" instead of leaving a user to trust
    #: a picture that looks like an answer.
    charge_caveat: Optional[str] = None
    #: ``"sas"`` (probe-centre, Shrake-Rupley) or ``"ses"`` (molecular).
    mode: str = "sas"
    #: Grid spacing in Å; ``0`` (or less) picks one from the atom extent.
    spacing: float = DEFAULT_SPACING
    probe: float = PROBE_RADIUS
    #: ``"hydrophobicity"``, ``"electrostatic"`` or ``"element"``.
    property: str = "hydrophobicity"
    #: ``"residue"`` (Kyte-Doolittle) or ``"atom"`` (AD4 fragment + Vina rule).
    hydrophobicity_scale: str = "residue"
    #: Fixed colour range; ``None`` derives one from the sampled values.
    value_range: Optional[Tuple[float, float]] = None
    #: Dielectric model for the electrostatic property. ``"distance"`` is the
    #: AutoDock 4 convention (``eps = 4 r``) and therefore the screening the
    #: score itself uses; ``"uniform"`` is the local-field picture. The
    #: distance model is the default because it is both consistent with the
    #: force field and far better contrasted on a protein surface, where the
    #: dielectric constant rises with distance from the charge.
    dielectric: str = "distance"
    epsilon: float = 4.0
    screening: float = 0.0
    #: Atom subset to build from: explicit indices, and/or a sphere.
    indices: Optional[Sequence[int]] = None
    centre: Optional[Sequence[float]] = None
    radius: Optional[float] = None
    #: Bonds, used by the ``"atom"`` hydrophobicity scale.
    bonds: Optional[Sequence] = None
    #: Residue keys ``(chain, res_id, res_name)`` to highlight on the surface.
    highlighted_residues: Optional[Set[Tuple[str, int, str]]] = None
    #: Point cap and probe-sphere sampling count.
    max_points: int = DEFAULT_MAX_POINTS
    directions: int = DEFAULT_DIRECTIONS
    #: Close the rims of an open mesh with flat lids.
    #:
    #: **A surface built from a subset of atoms does not need this**: the zero
    #: level of the field over the selected atoms is the boundary of a union of
    #: balls, which is already closed (`tools/pocket_closure.py` measures zero
    #: boundary edges for the full receptor and for pocket-lining builds alike).
    #: It is needed when the mesh is *cut* — by a clipping plane, or by keeping
    #: only the triangles near a site — because a cut leaves a hole, a hole has
    #: no inside, and without an inside there is no volume to report. The lids
    #: are not molecular surface: they are drawn in a neutral grey, counted
    #: separately in ``Surface.stats`` and excluded from the property legend.
    close_rim: bool = False

    def normalized(self) -> "SurfaceSettings":
        mode = str(self.mode).strip().lower()
        if mode not in SURFACE_MODES:
            raise ValueError(f"mode must be one of {SURFACE_MODES}, got {self.mode!r}")
        prop = str(self.property).strip().lower()
        if prop not in SURFACE_PROPERTIES:
            raise ValueError(
                f"property must be one of {SURFACE_PROPERTIES}, got {self.property!r}"
            )
        return replace(
            self,
            mode=mode,
            property=prop,
            spacing=float(self.spacing),
            probe=float(self.probe),
            hydrophobicity_scale=hydrophobicity_scale(self.hydrophobicity_scale),
        )


def resolve_spacing(
    lo: Sequence[float],
    hi: Sequence[float],
    spacing: float = DEFAULT_SPACING,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> float:
    """The grid spacing actually used: the request, capped by the point budget.

    ``spacing <= 0`` asks for an automatic value, which is the largest
    resolution (the smallest spacing) that keeps the grid inside ``max_points``.
    The result is clamped to ``[MIN_SPACING, MAX_SPACING]``: finer than
    :data:`MIN_SPACING` costs time without showing anything at any zoom the
    workbench offers, coarser than :data:`MAX_SPACING` visibly facets a helix.
    """
    extent = [max(0.0, float(hi[i]) - float(lo[i])) for i in range(3)]
    volume = max(extent[0] * extent[1] * extent[2], 1e-6)
    wanted = float(spacing)
    if not (wanted > 0.0):
        wanted = (volume / max(1, int(max_points))) ** (1.0 / 3.0)
    return float(min(MAX_SPACING, max(MIN_SPACING, wanted)))


#: The 27 cell offsets of a face-, edge- and corner-adjacent neighbourhood.
_NEIGHBOUR_OFFSETS = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ],
    dtype=np.int64,
)


def _cell_pairs(
    points: np.ndarray,
    coords: np.ndarray,
    search: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(point index, atom index, distance)`` for every pair within ``search``.

    A uniform cell as wide as ``search`` guarantees that every atom within
    ``search`` of a point shares its cell or one of the 26 around it, so the
    candidate set is a 27-cell box. That box is ~6x the volume of the sphere
    it must contain, so the candidates are filtered by the real distance
    before they are returned, and the distance is handed back: the caller
    needs it anyway, and recomputing it would double the most expensive step
    in the build.

    Everything is vectorised: the atoms are bucketed once with a sort, and for
    each of the 27 offsets the matching points are fetched with two
    ``searchsorted`` calls over the *whole* point array. One offset at a time
    keeps the peak temporary at a few hundred thousand pairs instead of the
    tens of millions the unfiltered box would need.
    """
    empty = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=float),
    )
    if points.size == 0 or coords.size == 0 or search <= 0.0:
        return empty
    cell = float(search)
    atom_keys = np.floor(coords / cell).astype(np.int64)
    origin = atom_keys.min(axis=0) - 1
    dims = (atom_keys.max(axis=0) - origin) + 3

    def flatten(k: np.ndarray) -> np.ndarray:
        shifted = k - origin[None, :]
        return (shifted[:, 0] * dims[1] + shifted[:, 1]) * dims[2] + shifted[:, 2]

    atom_flat = flatten(atom_keys)
    order = np.argsort(atom_flat, kind="stable")
    sorted_flat = atom_flat[order]
    point_keys = np.floor(points / cell).astype(np.int64)
    limit_sq = cell * cell

    point_parts: List[np.ndarray] = []
    atom_parts: List[np.ndarray] = []
    distance_parts: List[np.ndarray] = []
    high = origin + dims - 1
    for offset in _NEIGHBOUR_OFFSETS:
        shifted = np.clip(point_keys + offset, origin[None, :], high[None, :])
        keys = flatten(shifted)
        left = np.searchsorted(sorted_flat, keys, side="left")
        right = np.searchsorted(sorted_flat, keys, side="right")
        counts = right - left
        total = int(counts.sum())
        if total == 0:
            continue
        # Ragged gather: `start` is where each point's slice begins in the
        # concatenated result, so the positions taken are start + 0, 1, ...
        start = np.cumsum(counts) - counts
        taken = left.repeat(counts) + (
            np.arange(total, dtype=np.int64) - start.repeat(counts)
        )
        point_ids = np.repeat(np.arange(points.shape[0], dtype=np.int64), counts)
        atom_ids = order[taken]
        delta = points[point_ids] - coords[atom_ids]
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        keep = distance_sq <= limit_sq
        if keep.any():
            point_parts.append(point_ids[keep])
            atom_parts.append(atom_ids[keep])
            distance_parts.append(np.sqrt(distance_sq[keep]))
    if not point_parts:
        return empty
    return (
        np.concatenate(point_parts),
        np.concatenate(atom_parts),
        np.concatenate(distance_parts),
    )


def _reduce_min_per_point(
    point_index: np.ndarray,
    values: np.ndarray,
    count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(min per point, has-entry)`` for a flat (point, value) pair list."""
    if point_index.size == 0:
        return np.zeros(count, dtype=float), np.zeros(count, dtype=bool)
    order = np.argsort(point_index, kind="stable")
    sorted_points = point_index[order]
    sorted_values = values[order]
    counts = np.bincount(sorted_points, minlength=count)
    # Only the groups that exist are reduced: ``reduceat`` needs every start
    # index inside the array, and an empty trailing group would hand it the
    # length itself.
    present = np.nonzero(counts)[0]
    starts = np.cumsum(counts)[present] - counts[present]
    out = np.zeros(count, dtype=float)
    has = np.zeros(count, dtype=bool)
    out[present] = np.minimum.reduceat(sorted_values, starts)
    has[present] = True
    return out, has


def scalar_field(
    points: np.ndarray,
    atoms,
    *,
    probe: float = PROBE_RADIUS,
    radii=None,
    spacing: float = DEFAULT_SPACING,
    axes: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """The probe-centre field ``F(p) = min_i(|p - c_i| - (r_i + probe))``.

    Exact wherever ``|F| < 2 · spacing`` — which is every value a marching
    tetrahedra crossing can interpolate — and a valid lower bound outside,
    because a point further than ``2 · spacing`` from every atom can only be
    positive and can only be adjacent to positive corners. See the module
    docstring for why that bound is what makes a bounded neighbour search
    correct rather than merely fast.

    ``axes`` names the three coordinate vectors the points came from, in
    ``numpy.meshgrid(..., indexing="ij")`` order. With it the field is filled
    from a rectangular block around each atom instead of through the general
    pair search: the same numbers, several times faster, because a regular
    grid turns the neighbour lookup into a slice. The result then carries the
    grid's shape.
    """
    coords, table = _atom_arrays(atoms, radii)
    if axes is not None:
        return _field_on_axes(
            axes, coords, table + float(probe), 2.0 * float(spacing)
        )

    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    count = len(pts)
    if count == 0 or coords.shape[0] == 0:
        return np.zeros(count, dtype=float)
    inflated = table + float(probe)
    clamp = 2.0 * float(spacing)
    search = float(np.max(inflated)) + clamp
    point_index, atom_index, distance = _cell_pairs(pts, coords, search)
    if point_index.size == 0:
        return np.full(count, clamp, dtype=float)
    values = distance - inflated[atom_index]
    reduced, has = _reduce_min_per_point(point_index, values, count)
    out = np.full(count, clamp, dtype=float)
    out[has] = np.minimum(reduced[has], clamp)
    return out


def _field_on_axes(
    axes: Sequence[np.ndarray],
    coords: np.ndarray,
    inflated: np.ndarray,
    clamp: float,
) -> np.ndarray:
    """The same field, filled one atom-sized block at a time.

    ``search`` is the largest centre distance at which a node can still see an
    atom, so a block of that half-width in every direction is guaranteed to
    contain every contributing atom — and each block is a slice, with no index
    arrays and no ragged gather.
    """
    shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    if shape[0] > 1:
        spacing = float(axes[0][1] - axes[0][0])
    elif shape[1] > 1:  # pragma: no cover - a degenerate grid
        spacing = float(axes[1][1] - axes[1][0])
    else:  # pragma: no cover - a degenerate grid
        spacing = 1.0
    out = np.full(shape, clamp, dtype=float)
    if coords.shape[0] == 0:
        return out
    search = float(np.max(inflated)) + clamp
    for atom in range(coords.shape[0]):
        centre = coords[atom]
        window = []
        starts = []
        for axis in range(3):
            values = axes[axis]
            low = int(np.searchsorted(values, centre[axis] - search, side="left"))
            high = int(np.searchsorted(values, centre[axis] + search, side="right"))
            low = max(0, min(low, shape[axis]))
            high = max(low, min(high, shape[axis]))
            starts.append(low)
            window.append(values[low:high])
        if any(part.size == 0 for part in window):
            continue
        block_shape = (len(window[0]), len(window[1]), len(window[2]))
        distance_sq = np.zeros(block_shape, dtype=float)
        for axis in range(3):
            offset = (window[axis] - centre[axis]).reshape(
                [-1 if index == axis else 1 for index in range(3)]
            )
            distance_sq += offset * offset
        np.sqrt(distance_sq, out=distance_sq)
        distance_sq -= float(inflated[atom])
        block = out[
            starts[0] : starts[0] + block_shape[0],
            starts[1] : starts[1] + block_shape[1],
            starts[2] : starts[2] + block_shape[2],
        ]
        np.minimum(block, distance_sq, out=block)
    return out


def _atom_arrays(atoms, radii=None) -> Tuple[np.ndarray, np.ndarray]:
    """``(coords, radii)`` for an atom sequence, with an explicit override."""
    atoms = list(atoms)
    coords = np.asarray(
        [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
    ).reshape(-1, 3)
    if radii is not None:
        table = np.asarray(radii, dtype=float).reshape(-1)
        if table.size != coords.shape[0]:
            raise ValueError(
                f"radii has {table.size} entries for {coords.shape[0]} atoms"
            )
        return coords, table
    return coords, np.asarray([radius_of(getattr(a, "element", None)) for a in atoms])


def nearest_atoms(
    points: np.ndarray,
    atoms,
    *,
    radii=None,
    probe: float = PROBE_RADIUS,
    spacing: float = DEFAULT_SPACING,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(index, distance)`` of the nearest atom for every point.

    Used to give every surface vertex the property of the atom whose sphere it
    lies on and the outward normal of that sphere. The search radius is the
    same bounded one :func:`scalar_field` uses.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    coords, table = _atom_arrays(atoms, radii)
    count = len(pts)
    index = np.full(count, -1, dtype=np.int64)
    distance = np.full(count, np.inf, dtype=float)
    if count == 0 or coords.shape[0] == 0:
        return index, distance
    # A surface vertex sits exactly on its owning atom's inflated sphere, so
    # the nearest atom centre is never further than the largest inflated
    # radius; the small allowance covers the sub-grid deviation of a marching
    # tetrahedra vertex from the true isosurface.
    search = float(np.max(table + probe)) + 0.5 * float(spacing)
    point_index, atom_index, point_distance = _cell_pairs(pts, coords, search)
    if point_index.size == 0:
        return index, distance
    # Sorted *descending* and written in that order: the smallest distance for
    # a point is therefore the last write it receives. It is one argsort
    # instead of a Python loop per point.
    order = np.argsort(-point_distance, kind="stable")
    index[point_index[order]] = atom_index[order]
    distance[point_index[order]] = point_distance[order]
    return index, distance


# ---------------------------------------------------------------------------
# marching tetrahedra
# ---------------------------------------------------------------------------

#: The eight cube corners, in the binary order used by the case codes.
CUBE_CORNERS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
)

#: The cube is split into six tetrahedra, all sharing the main diagonal 0—.
#: Each has volume 1/6 of the cube, so the decomposition is a partition with no
#: gap and no overlap, and —unlike the alternative 5-tetrahedra splits —it is
#: chosen consistently for every cube, so two neighbouring cubes agree on the
#: face they share and the mesh has no cracks.
TETRAHEDRA: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
    (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7),
)

#: The six edges of a tetrahedron, in a fixed order.
TET_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
)


def _tetrahedron_cases():
    """``{code: (triangle, ...)}`` where a triangle is three edge indices.

    ``code`` is the bit mask of the tetrahedron's corners that are *inside*
    (``value < level``). The table is derived here rather than transcribed
    from a paper: the classic 16-entry marching-tetrahedra table is exactly
    this enumeration, and deriving it removes the one part of the algorithm
    that is normally copied by hand and mis-copied.
    """
    edge_index = {edge: index for index, edge in enumerate(TET_EDGES)}

    def edge_of(first: int, second: int) -> int:
        return edge_index.get((first, second), edge_index.get((second, first), -1))

    table: Dict[int, Tuple[Tuple[int, int, int], ...]] = {}
    for code in range(16):
        inside = [corner for corner in range(4) if code >> corner & 1]
        outside = [corner for corner in range(4) if not code >> corner & 1]
        if len(inside) in (0, 4):
            table[code] = ()
        elif len(inside) == 1:
            corner = inside[0]
            table[code] = (
                tuple(edge_of(corner, other) for other in outside),
            )
        elif len(inside) == 3:
            corner = outside[0]
            table[code] = (
                tuple(edge_of(corner, other) for other in inside),
            )
        else:
            first, second = inside
            third, fourth = outside
            # The four crossings form a quad; walked in this order its edges
            # are shared between the two triangles, so the two halves cannot
            # be wound in opposite senses.
            table[code] = (
                (
                    edge_of(first, third),
                    edge_of(first, fourth),
                    edge_of(second, fourth),
                ),
                (
                    edge_of(first, third),
                    edge_of(second, fourth),
                    edge_of(second, third),
                ),
            )
    return table


_TET_CASES = _tetrahedron_cases()


def marching_tetrahedra(
    field: np.ndarray,
    origin: Sequence[float],
    spacing: float,
    *,
    level: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate the ``level`` isosurface of a uniform scalar field.

    Returns ``(vertices (n, 3) float64, triangles (m, 3) int32)``. Vertices are
    not welded: two adjacent tetrahedra each emit their own copy of a shared
    crossing point, which is what lets every vertex carry an independently
    computed normal and property, and costs nothing the GPU cares about.

    Only cubes whose eight corners straddle the level are touched, so the work
    scales with the *surface area* rather than the volume —which is why a
    400 000-point grid triangulates in well under a second.
    """
    data = np.asarray(field, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"field must be three-dimensional, got shape {data.shape}")
    shape = data.shape
    if min(shape) < 2:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int32)

    inside = data < float(level)
    outside = data > float(level)
    any_inside = np.zeros(tuple(size - 1 for size in shape), dtype=bool)
    any_outside = np.zeros_like(any_inside)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                sl = (
                    slice(dx, dx + shape[0] - 1),
                    slice(dy, dy + shape[1] - 1),
                    slice(dz, dz + shape[2] - 1),
                )
                any_inside |= inside[sl]
                any_outside |= outside[sl]
    cube_index = np.argwhere(any_inside & any_outside)
    if cube_index.shape[0] == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int32)

    # (cube, tetrahedron) corners, then the four corner values of each. The
    # offsets are gathered once per tetrahedron, so the (cubes, 6, 4) index
    # array is the only large temporary and no intermediate is duplicated.
    tet_offsets = np.asarray(CUBE_CORNERS, dtype=np.int64)[
        np.asarray(TETRAHEDRA, dtype=np.int64)
    ]                                                       # (6, 4, 3)
    positions = (
        cube_index[:, None, None, :] + tet_offsets[None, :, :, :]
    ).reshape(-1, 4, 3)                                     # (C*6, 4, 3)
    values = data[positions[..., 0], positions[..., 1], positions[..., 2]]

    grid_origin = np.asarray(origin, dtype=float).reshape(3)
    world = grid_origin[None, None, :] + positions * float(spacing)

    # The crossing of every one of the six edges, for every tetrahedron.
    edges = np.asarray(TET_EDGES, dtype=np.int64)
    first_value = values[:, edges[:, 0]]
    second_value = values[:, edges[:, 1]]
    denominator = second_value - first_value
    fraction = np.where(
        np.abs(denominator) > 1e-12,
        (float(level) - first_value) / np.where(denominator == 0.0, 1.0, denominator),
        0.5,
    )
    first_point = world[:, edges[:, 0], :]
    crossings = first_point + fraction[..., None] * (
        world[:, edges[:, 1], :] - first_point
    )

    codes = (
        (values < float(level))
        * np.array([1, 2, 4, 8], dtype=np.int64)[None, :]
    ).sum(axis=1)

    vertex_blocks: List[np.ndarray] = []
    triangle_blocks: List[np.ndarray] = []
    offset = 0
    for code in range(16):
        triangles = _TET_CASES.get(code, ())
        if not triangles:
            continue
        selected = np.nonzero(codes == code)[0]
        if selected.size == 0:
            continue
        points_here = crossings[selected]                 # (S, 6, 3)
        pieces = []
        for triangle in triangles:
            pieces.append(points_here[:, list(triangle), :])   # (S, 3, 3)
        block = np.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0]
        flat = block.reshape(-1, 3)
        vertex_blocks.append(flat)
        local = np.arange(flat.shape[0], dtype=np.int64).reshape(-1, 3) + offset
        triangle_blocks.append(local.astype(np.int32))
        offset += flat.shape[0]

    if not vertex_blocks:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.int32)
    return (
        np.concatenate(vertex_blocks, axis=0),
        np.concatenate(triangle_blocks, axis=0),
    )


def surface_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """The area of a triangulated surface, in Å²."""
    if len(triangles) == 0 or len(vertices) == 0:
        return 0.0
    corners = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def orient_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: Optional[np.ndarray] = None,
    reference: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Re-wind every triangle so the surface has one consistent outside.

    Marching tetrahedra emits each triangle from its own case, so the winding is
    locally arbitrary: fine for drawing (the shader lights back faces too), fatal
    for a volume, because the divergence theorem sums signed terms and
    inconsistent winding cancels them. The outward direction comes from the
    per-vertex normals the builder already computed from the field gradient, or
    — for a lid, which has no field — from the mesh centroid.
    """
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if faces.size == 0:
        return faces
    corners = points[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    if normals is not None:
        outward = np.asarray(normals, dtype=float).reshape(-1, 3)[faces].sum(axis=1)
    else:
        centre = (
            np.asarray(reference, dtype=float).reshape(3)
            if reference is not None
            else points.mean(axis=0)
        )
        outward = corners.mean(axis=1) - centre[None, :]
    flip = np.einsum("ij,ij->i", face_normals, outward) < 0.0
    if flip.any():
        faces = faces.copy()
        faces[flip] = faces[flip][:, ::-1]
    return faces


def surface_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    """The volume a **closed** mesh encloses, in Å³, by the divergence theorem.

    ``V = |Σ v0 · (v1 × v2)| / 6`` over every triangle. It is only meaningful
    for a closed surface, which is why :meth:`Surface.volume` refuses to report
    it otherwise — an open mesh would silently return the volume of a shape that
    has no inside, and a number nobody can check is worse than no number.
    """
    if len(triangles) == 0 or len(vertices) == 0:
        return 0.0
    corners = np.asarray(vertices, dtype=float)[np.asarray(triangles, dtype=np.int64)]
    triple = np.einsum("ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2]))
    return abs(float(triple.sum()) / 6.0)


#: Vertices closer than this are the same point when the mesh topology is
#: needed. The marching-tetrahedra output is deliberately unwelded (each triangle
#: carries its own copies so every vertex can have its own property), which is
#: fine for drawing and useless for finding a boundary.
WELD_TOLERANCE = 1e-4


def weld_mesh(vertices: np.ndarray, triangles: np.ndarray):
    """``(welded vertices, triangles, inverse)`` — coincident points merged.

    ``inverse[i]`` is the welded index of original vertex ``i``, so a per-vertex
    property can still be averaged back onto the welded mesh.
    """
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if points.size == 0:
        return points, faces, np.zeros(0, dtype=np.int64)
    keys = np.round(points / WELD_TOLERANCE).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    welded = unique.astype(float) * WELD_TOLERANCE
    return welded, inverse[faces], inverse.reshape(-1)


def mesh_boundary_loops(vertices: np.ndarray, triangles: np.ndarray) -> List[np.ndarray]:
    """The open rims of a mesh, as lists of welded vertex indices.

    An edge that belongs to exactly one triangle is on the boundary; in a
    manifold mesh every boundary vertex has exactly two such edges, so the
    boundary is a set of closed loops. The walk is defensive about a
    non-manifold input: it never visits a vertex twice and it stops rather than
    looping forever.
    """
    welded, faces, _inverse = weld_mesh(vertices, triangles)
    if faces.size == 0:
        return []
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if boundary.shape[0] == 0:
        return []

    neighbours: Dict[int, List[int]] = {}
    for first, second in boundary.tolist():
        neighbours.setdefault(first, []).append(second)
        neighbours.setdefault(second, []).append(first)

    loops: List[np.ndarray] = []
    unvisited = set(neighbours)
    while unvisited:
        start = min(unvisited)
        loop = [start]
        unvisited.discard(start)
        previous = None
        current = start
        while True:
            candidates = [
                node for node in neighbours.get(current, ()) if node != previous and node in unvisited
            ]
            if not candidates:
                # Close the loop if the start is a neighbour, otherwise stop.
                if start in neighbours.get(current, ()) and len(loop) > 2:
                    break
                break
            nxt = min(candidates)
            loop.append(nxt)
            unvisited.discard(nxt)
            previous, current = current, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def cap_open_mesh(
    vertices: np.ndarray, triangles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Close every rim with a flat fan, returning ``(vertices, triangles, added)``.

    Each boundary loop is capped by a triangle fan around its centroid. The cap
    is *not* a molecular surface — it is a lid over the hole a selection or a
    clip plane cut — so it is added as its own geometry with its own colour and
    its area is reported separately (see :attr:`Surface.cap_area`). What it
    buys is a **closed** mesh, and therefore a volume.
    """
    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    loops = mesh_boundary_loops(points, faces)
    if not loops:
        return points, faces, 0
    welded, _faces, inverse = weld_mesh(points, faces)
    # One drawn vertex per welded position: the fan is built in the mesh the
    # caller gets back, so the caps must reference *drawn* indices.
    representative: Dict[int, int] = {}
    for drawn, welded_index in enumerate(inverse.tolist()):
        representative.setdefault(int(welded_index), drawn)
    extra_points: List[np.ndarray] = []
    extra_faces: List[np.ndarray] = []
    offset = len(points)
    added = 0
    for loop in loops:
        centre = welded[loop].mean(axis=0)
        extra_points.append(centre)
        centre_index = offset + len(extra_points) - 1
        size = len(loop)
        for index in range(size):
            first = representative.get(int(loop[index]))
            second = representative.get(int(loop[(index + 1) % size]))
            if first is None or second is None:  # pragma: no cover - defensive
                continue
            extra_faces.append(np.array([centre_index, first, second], dtype=np.int64))
            added += 1
    if not extra_faces:
        return points, faces, 0
    return (
        np.concatenate([points, np.asarray(extra_points)], axis=0),
        np.concatenate([faces, np.asarray(extra_faces, dtype=np.int64)], axis=0),
        added,
    )


# ---------------------------------------------------------------------------
# the SES by grayscale erosion
# ---------------------------------------------------------------------------


def _erode_field(
    field: np.ndarray,
    spacing: float,
    probe: float,
    *,
    directions: int = DEFAULT_DIRECTIONS,
    clamp: float,
) -> np.ndarray:
    """``K(p) = max_u F(p + probe·u)`` on the same grid.

    Each sampled offset has the *same* fractional part for every grid point,
    so the trilinear read is eight fixed shifts and eight constant weights —    a handful of array adds per direction instead of an interpolation per
    point. The padding is filled with ``clamp``: outside the grid every atom
    is more than a probe away, so the true field there is positive and the
    maximum stays positive, which is what makes the surface closed at the
    grid boundary.
    """
    data = np.asarray(field, dtype=np.float32)
    shape = data.shape
    unit = sphere_points(int(directions)).astype(np.float64)
    pad = int(math.ceil(probe / spacing)) + 2
    padded = np.pad(data, pad, mode="constant", constant_values=np.float32(clamp))
    eroded = np.full(shape, -np.inf, dtype=np.float32)
    sample = np.empty(shape, dtype=np.float32)

    for direction in unit:
        offset = direction * float(probe) / float(spacing)
        base = np.floor(offset).astype(np.int64)
        weights = offset - base
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight = (
                        (weights[0] if dx else 1.0 - weights[0])
                        * (weights[1] if dy else 1.0 - weights[1])
                        * (weights[2] if dz else 1.0 - weights[2])
                    )
                    if weight <= 1e-9:
                        continue
                    start = (pad + base[0] + dx, pad + base[1] + dy, pad + base[2] + dz)
                    slab = padded[
                        start[0] : start[0] + shape[0],
                        start[1] : start[1] + shape[1],
                        start[2] : start[2] + shape[2],
                    ]
                    if dx == 0 and dy == 0 and dz == 0:
                        np.multiply(slab, np.float32(weight), out=sample)
                    else:
                        sample += np.float32(weight) * slab
        np.maximum(eroded, sample, out=eroded)
    return eroded.astype(np.float64)


# ---------------------------------------------------------------------------
# the potential, decomposed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PotentialGroup:
    """One residue's (or one atom's) share of the potential on a surface.

    The potential at a point is a *sum* over atoms, so it can be split exactly —
    no approximation is involved in asking which residue makes a pocket
    electropositive. ``mean`` is the group's contribution averaged over the
    sampled surface points and ``extreme`` its largest contribution at any one
    of them, which is what makes a single buried carboxylate visible in a sea of
    backbone carbonyls.
    """

    label: str
    kind: str
    atoms: int
    charge: float
    mean: float
    extreme: float
    share: float
    #: The group's contribution at one chosen point (the site centre), and its
    #: share of the total potential there. This is the discriminating number: a
    #: mean over a whole pocket surface dilutes every residue equally, while the
    #: value *at the site* is dominated by whatever is closest to it — which is
    #: what "which residue makes this pocket negative?" actually asks.
    at_focus: Optional[float] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "kind": self.kind,
            "atoms": int(self.atoms),
            "charge": round(float(self.charge), 4),
            "mean": round(float(self.mean), 3),
            "extreme": round(float(self.extreme), 3),
            "share": round(float(self.share), 4),
            "at_focus": None if self.at_focus is None else round(float(self.at_focus), 3),
        }


def electrostatic_decomposition(
    points,
    atoms,
    charges=None,
    *,
    group: str = "residue",
    dielectric: str = "distance",
    epsilon: float = 4.0,
    screening: float = 0.0,
    minimum_distance: float = 1.0,
    max_points: int = 2000,
    top: Optional[int] = None,
    focus: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Which residues make *this* surface electropositive or negative.

    The colour map says a pocket is negative; this says **why**. Every atom's
    contribution to the Coulomb potential is computed separately and summed per
    residue (or per atom) over a deterministic subsample of the surface points.

    ``max_points`` bounds the work: the mean over a few thousand surface
    vertices is the same number as the mean over a few hundred thousand to
    within the resolution of any legend, and it keeps the call fast enough to
    run from a menu. The subsample takes every ``n``-th point, so it is
    reproducible rather than random.

    This is still the same Coulomb model as :func:`electrostatic_potential` —
    point charges, a stated dielectric, no Poisson-Boltzmann solution and no
    ionic atmosphere beyond the optional Debye term. What it adds is an exact
    algebraic split of that model, which is a statement about the model and not
    about the real solvent.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    atoms = list(atoms)
    if pts.size == 0 or not atoms:
        return {
            "points": 0,
            "points_available": int(pts.shape[0]),
            "total_mean": 0.0,
            "groups": [],
            "dielectric": str(dielectric),
            "epsilon": float(epsilon),
        }
    stride = max(1, int(pts.shape[0] // max(1, int(max_points))))
    sample = pts[::stride]
    table = (
        charges_from_atoms(atoms)
        if charges is None
        else np.asarray(charges, dtype=float).reshape(-1)
    )
    if table.size != len(atoms):
        raise ValueError(f"charges has {table.size} entries for {len(atoms)} atoms")

    key = str(group).strip().lower()
    if key not in ("residue", "atom"):
        raise ValueError(f"group must be 'residue' or 'atom', got {group!r}")

    # Group the atoms, keeping their order of first appearance so the output is
    # stable and a residue's atoms stay together.
    order: List[str] = []
    members: Dict[str, List[int]] = {}
    for index, atom in enumerate(atoms):
        if key == "residue":
            label = (
                f"{str(getattr(atom, 'chain', '') or '')}/"
                f"{str(getattr(atom, 'res_name', '') or '')}"
                f"{int(getattr(atom, 'res_id', 0) or 0)}"
            )
        else:
            label = (
                f"{str(getattr(atom, 'res_name', '') or '')}"
                f"{int(getattr(atom, 'res_id', 0) or 0)}:"
                f"{str(getattr(atom, 'name', '') or '')}"
            )
        if label not in members:
            members[label] = []
            order.append(label)
        members[label].append(index)

    total = electrostatic_potential(
        sample,
        atoms,
        charges=table,
        dielectric=dielectric,
        epsilon=epsilon,
        screening=screening,
        minimum_distance=minimum_distance,
    )
    groups: List[PotentialGroup] = []
    for label in order:
        indices = members[label]
        subset = [atoms[index] for index in indices]
        contribution = electrostatic_potential(
            sample,
            subset,
            charges=table[indices],
            dielectric=dielectric,
            epsilon=epsilon,
            screening=screening,
            minimum_distance=minimum_distance,
        )
        mean = float(contribution.mean()) if contribution.size else 0.0
        strongest = (
            float(contribution[np.argmax(np.abs(contribution))])
            if contribution.size
            else 0.0
        )
        groups.append(
            PotentialGroup(
                label=label,
                kind=key,
                atoms=len(indices),
                charge=float(table[indices].sum()),
                mean=mean,
                extreme=strongest,
                share=0.0,
            )
        )

    mean_total = float(total.mean()) if total.size else 0.0
    denominator = abs(mean_total) if abs(mean_total) > 1e-12 else 1.0
    # The value at the focus point, if the caller gave one: the same split, read
    # where the ligand sits rather than averaged over the whole surface.
    focus_values: Dict[str, float] = {}
    focus_total = None
    if focus is not None:
        point = np.asarray(focus, dtype=float).reshape(1, 3)
        focus_total = float(
            electrostatic_potential(
                point,
                atoms,
                charges=table,
                dielectric=dielectric,
                epsilon=epsilon,
                screening=screening,
                minimum_distance=minimum_distance,
            )[0]
        )
        for label in order:
            indices = members[label]
            focus_values[label] = float(
                electrostatic_potential(
                    point,
                    [atoms[index] for index in indices],
                    charges=table[indices],
                    dielectric=dielectric,
                    epsilon=epsilon,
                    screening=screening,
                    minimum_distance=minimum_distance,
                )[0]
            )

    groups = [
        PotentialGroup(
            label=item.label,
            kind=item.kind,
            atoms=item.atoms,
            charge=item.charge,
            mean=item.mean,
            extreme=item.extreme,
            share=item.mean / denominator,
            at_focus=focus_values.get(item.label),
        )
        for item in groups
    ]
    if focus_total is not None:
        groups.sort(key=lambda item: (-abs(item.at_focus or 0.0), item.label))
    else:
        groups.sort(key=lambda item: (-abs(item.mean), item.label))
    if top is not None:
        groups = groups[: max(0, int(top))]
    return {
        "points": int(sample.shape[0]),
        "points_available": int(pts.shape[0]),
        "stride": int(stride),
        "total_mean": mean_total,
        "total_min": float(total.min()) if total.size else 0.0,
        "total_max": float(total.max()) if total.size else 0.0,
        "focus": None if focus is None else [float(v) for v in np.asarray(focus).reshape(3)],
        "focus_total": focus_total,
        "dielectric": str(dielectric),
        "epsilon": float(epsilon),
        "screening": float(screening),
        "total_charge": float(table.sum()),
        "charges": int(np.count_nonzero(table)),
        "atoms": len(atoms),
        "groups": groups,
    }


# ---------------------------------------------------------------------------
# atom selection
# ---------------------------------------------------------------------------


def select_atoms(
    atoms,
    *,
    indices: Optional[Sequence[int]] = None,
    centre: Optional[Sequence[float]] = None,
    radius: Optional[float] = None,
) -> List[int]:
    """The indices of the atoms a surface should be built from.

    ``indices`` picks atoms explicitly; ``centre``/``radius`` keeps everything
    within a sphere. Both may be given, and the result is their intersection.
    This is what "pocket lining only" is: the *same* surface, built from the
    atoms that line the site, so the picture shows one bowl instead of the
    whole protein and the buried pocket is suddenly in front of the camera
    rather than inside a shell.
    """
    total = len(atoms)
    keep = np.ones(total, dtype=bool)
    if indices is not None:
        keep[:] = False
        for index in indices:
            try:
                position = int(index)
            except (TypeError, ValueError):
                continue
            if 0 <= position < total:
                keep[position] = True
    if centre is not None and radius is not None:
        middle = np.asarray(centre, dtype=float).reshape(3)
        limit = float(radius)
        coords = np.asarray(
            [[float(a.x), float(a.y), float(a.z)] for a in atoms], dtype=float
        ).reshape(-1, 3)
        delta = coords - middle[None, :]
        keep &= np.einsum("ij,ij->i", delta, delta) <= limit * limit
    return [int(index) for index in np.nonzero(keep)[0]]


# ---------------------------------------------------------------------------
# the surface object
# ---------------------------------------------------------------------------


@dataclass
class Surface:
    """A triangulated molecular surface with a property on every vertex."""

    vertices: np.ndarray
    normals: np.ndarray
    colors: np.ndarray
    values: np.ndarray
    atom_index: np.ndarray
    triangles: np.ndarray
    mode: str = "sas"
    property_name: str = "hydrophobicity"
    palette: str = "hydrophobicity"
    value_range: Tuple[float, float] = (0.0, 1.0)
    probe: float = PROBE_RADIUS
    spacing: float = DEFAULT_SPACING
    stats: Dict[str, object] = field(default_factory=dict)
    residue_labels: List[str] = field(default_factory=list)
    highlighted: Optional[np.ndarray] = None
    #: The indices, into the atom sequence the build was handed, of the atoms
    #: this surface was built from. ``atom_index`` on a vertex indexes *that*
    #: filtered list, so a caller that wants the atom in its own scene must go
    #: through this mapping. It exists because getting it wrong is invisible:
    #: a surface painted with the properties of the wrong atoms is a beautiful
    #: picture of nothing (and a PDBQT round trip, which carries no bond
    #: orders, is exactly where an atom order silently changes).
    selected: Optional[List[int]] = None
    #: Whether the mesh is watertight, and — only then — the volume it encloses
    #: in Å³ (see :func:`surface_volume` and :meth:`volume`).
    closed: bool = False
    enclosed_volume: float = 0.0
    #: The charge-column caveat this surface was built with, or ``None``. A
    #: potential map that is wrong-signed looks exactly like one that is right,
    #: so the caveat travels with the surface rather than being printed once.
    charge_warning: Optional[str] = None
    #: Triangles and vertices that are rim lids rather than molecular surface.
    #: They are excluded from ``area`` and from the property legend, so a capped
    #: figure still reports the surface it is a figure *of*.
    cap_triangles: int = 0
    cap_vertices: int = 0

    def source_atom(self, vertex: int, atoms: Optional[Sequence] = None):
        """The atom a vertex belongs to, in the caller's own list.

        ``atoms`` is optional: without it the index into the original sequence
        is returned, which is what a caller that kept that sequence wants.
        """
        if self.selected is None or not (0 <= int(vertex) < len(self.atom_index)):
            return None
        local = int(self.atom_index[int(vertex)])
        if not (0 <= local < len(self.selected)):
            return None
        source = int(self.selected[local])
        if atoms is None:
            return source
        return atoms[source] if 0 <= source < len(atoms) else None

    # -- derived -----------------------------------------------------------

    @property
    def volume(self) -> Optional[float]:
        """The enclosed volume in Å³, or ``None`` when the mesh is open.

        A number nobody can check is worse than no number: an open mesh has no
        inside, so this refuses to answer for one instead of returning the
        volume of some imagined shape.
        """
        return float(self.enclosed_volume) if self.closed else None

    @property
    def cap_area(self) -> float:
        """The area of the rim lids alone, in Å² (0 without ``close_rim``)."""
        if not self.cap_triangles:
            return 0.0
        start = self.triangles_count - int(self.cap_triangles)
        return surface_area(self.vertices, self.triangles[start:])

    @property
    def area(self) -> float:
        """The mesh area in Å² (the sum of its triangles)."""
        return surface_area(self.vertices, self.triangles)

    @property
    def triangles_count(self) -> int:
        return int(len(self.triangles))

    @property
    def vertices_count(self) -> int:
        return int(len(self.vertices))

    def mesh(self) -> np.ndarray:
        """``(n, 9)`` interleaved ``position, normal, colour`` for the renderer.

        This is the vertex format :data:`odock.gui.viewport.MESH_VS` already
        consumes, which is the whole point: the surface is drawn through the
        existing mesh pipeline, with its lighting, its ambient occlusion and
        its depth interaction with the atoms, rather than through a second
        rendering path that would have to be kept in step.
        """
        if self.vertices_count == 0:
            return np.zeros((0, 9), dtype="f4")
        colours = np.clip(np.asarray(self.colors, dtype=np.float32), 0.0, 1.0)
        if self.highlighted is not None and len(self.highlighted):
            mask = np.asarray(self.highlighted, dtype=bool).reshape(-1)
            if mask.size == colours.shape[0] and bool(mask.any()):
                # A highlighted residue is not a different property: it is the
                # same colour pushed towards a bright cyan so a contact residue
                # can be *found* on the surface without hiding the map. Cyan is
                # chosen against both palettes: it is not the hydrophilic blue,
                # not the hydrophobic gold and not the electrostatic red.
                colours = colours.copy()
                colours[mask] = 0.40 * colours[mask] + 0.60 * np.array(
                    [0.30, 0.92, 0.95], dtype=np.float32
                )
        data = np.concatenate(
            [
                np.asarray(self.vertices, dtype=np.float32),
                np.asarray(self.normals, dtype=np.float32),
                colours,
            ],
            axis=1,
        )
        return np.ascontiguousarray(data, dtype="f4")

    def legend(self, count: int = 6) -> List[Tuple[float, Tuple[float, float, float], str]]:
        """``(fraction, colour, label)`` stops for the host's legend."""
        _label, unit = PROPERTY_LABELS.get(self.property_name, (self.property_name, ""))
        return legend_stops(self.palette, self.value_range, count=count, unit=unit)

    def summary(self) -> str:
        """One line for the log: the numbers a user can check."""
        volume = ""
        if self.closed:
            volume = f", {self.enclosed_volume:.0f} Å³ enclosed"
            if self.cap_triangles:
                volume += f" ({self.cap_triangles} lid triangles, {self.cap_area:.0f} Å²)"
        return (
            f"{self.mode.upper()} {self.vertices_count} vertices, "
            f"{self.triangles_count} triangles, {self.area:.0f} Å²{volume}, "
            f"grid {self.stats.get('spacing', self.spacing):.2f} Å, "
            f"{float(self.stats.get('seconds', 0.0)):.2f} s"
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "property": self.property_name,
            "palette": self.palette,
            "value_range": [float(self.value_range[0]), float(self.value_range[1])],
            "probe": float(self.probe),
            "spacing": float(self.spacing),
            "vertices": self.vertices_count,
            "triangles": self.triangles_count,
            "area": round(self.area, 2),
            "closed": bool(self.closed),
            "volume": None if self.volume is None else round(self.volume, 2),
            "cap_triangles": int(self.cap_triangles),
            "cap_area": round(self.cap_area, 2),
            "stats": dict(self.stats),
        }


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def _progress(progress, phase: str, fraction: float) -> None:
    if progress is None:
        return
    try:
        progress(phase, float(fraction))
    except Exception:  # pragma: no cover - a host callback must never break a build
        pass


def build_surface(
    atoms,
    settings: Optional[SurfaceSettings] = None,
    *,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Surface:
    """Build a surface from a sequence of Atom-like objects.

    The phases reported to ``progress`` are ``"grid"``, ``"field"``,
    ``"erode"`` (SES only), ``"mesh"`` and ``"colour"``, each with a fraction
    in ``[0, 1]``. The GUI runs this on a worker thread, so a 2 000-atom
    receptor never freezes the window while its surface is built.
    """
    options = (settings or SurfaceSettings()).normalized()
    started = time.perf_counter()
    atoms = list(atoms)
    if not atoms:
        return Surface(
            vertices=np.zeros((0, 3), dtype=float),
            normals=np.zeros((0, 3), dtype=float),
            colors=np.zeros((0, 3), dtype=float),
            values=np.zeros(0, dtype=float),
            atom_index=np.zeros(0, dtype=np.int64),
            triangles=np.zeros((0, 3), dtype=np.int32),
            mode=options.mode,
            property_name=options.property,
            palette=options.property,
            stats={"atoms": 0, "seconds": 0.0, "reason": "no atoms"},
        )

    chosen = select_atoms(
        atoms,
        indices=options.indices,
        centre=options.centre,
        radius=options.radius,
    )
    if not chosen:
        return Surface(
            vertices=np.zeros((0, 3), dtype=float),
            normals=np.zeros((0, 3), dtype=float),
            colors=np.zeros((0, 3), dtype=float),
            values=np.zeros(0, dtype=float),
            atom_index=np.zeros(0, dtype=np.int64),
            triangles=np.zeros((0, 3), dtype=np.int32),
            mode=options.mode,
            property_name=options.property,
            palette=options.property,
            stats={"atoms": 0, "seconds": 0.0, "reason": "empty selection"},
        )

    used = [atoms[index] for index in chosen]
    coords, radii = _atom_arrays(used)
    _progress(progress, "grid", 0.02)

    spacing = resolve_spacing(
        coords.min(axis=0) - (np.max(radii) + options.probe + 3.0),
        coords.max(axis=0) + (np.max(radii) + options.probe + 3.0),
        options.spacing,
        max_points=options.max_points,
    )
    margin = float(np.max(radii)) + options.probe + 2.5 * spacing
    low = coords.min(axis=0) - margin
    high = coords.max(axis=0) + margin
    axes: List[np.ndarray] = []
    for axis in range(3):
        values_axis = np.arange(float(low[axis]), float(high[axis]), spacing)
        if values_axis.size < 2:
            # A degenerate extent (every atom on one plane) still needs two
            # nodes so the tetrahedra have a volume to march through.
            values_axis = np.array([float(low[axis]), float(low[axis]) + spacing])
        axes.append(values_axis)
    shape = (len(axes[0]), len(axes[1]), len(axes[2]))
    points = int(shape[0] * shape[1] * shape[2])
    _progress(progress, "grid", 0.10)

    field = scalar_field(
        np.zeros((0, 3)),
        used,
        probe=options.probe,
        radii=radii,
        spacing=spacing,
        axes=axes,
    )
    _progress(progress, "field", 0.45)

    sas_area = None
    if options.mode == "sas":
        surface_field = field
    else:
        # The reference SAS area is the *analytic* Shrake-Rupley number from
        # odock.sasa, not a second triangulation: two independent methods
        # agreeing is a check on both, and the SES/SAS ratio it produces is
        # the number the tests pin.
        sas_area = float(
            sasa_of_atoms(used, probe=options.probe, radii=radii).sum()
        )
        _progress(progress, "erode", 0.55)
        surface_field = _erode_field(
            field,
            spacing,
            options.probe,
            directions=options.directions,
            clamp=2.0 * spacing,
        )
    _progress(progress, "mesh", 0.70)

    vertices, triangles = marching_tetrahedra(
        surface_field, axes_origin(axes), spacing
    )
    # Rim lids, for a mesh that was cut rather than built from a subset (see
    # SurfaceSettings.close_rim): a hole has no inside, so a volume needs one.
    cap_triangles = 0
    if options.close_rim:
        vertices, triangles, cap_triangles = cap_open_mesh(vertices, triangles)
    _progress(progress, "mesh", 0.82)
    if vertices.shape[0] == 0:
        return Surface(
            vertices=vertices,
            normals=np.zeros((0, 3), dtype=float),
            colors=np.zeros((0, 3), dtype=float),
            values=np.zeros(0, dtype=float),
            atom_index=np.zeros(0, dtype=np.int64),
            triangles=triangles,
            mode=options.mode,
            property_name=options.property,
            palette=options.property,
            probe=options.probe,
            spacing=spacing,
            stats={
                "atoms": len(used),
                "spacing": spacing,
                "grid": list(shape),
                "grid_points": int(points),
                "triangles": 0,
                "seconds": time.perf_counter() - started,
                "reason": "no isosurface on this grid",
            },
            selected=list(chosen),
        )

    # -- normals and the nearest atom -------------------------------------
    index, _distance = nearest_atoms(
        vertices, used, radii=radii, probe=options.probe, spacing=spacing
    )
    normals = _vertex_normals(
        vertices, surface_field, axes_origin(axes), spacing, used, radii, index
    )
    # One consistent outside. Without it the divergence theorem sums signed
    # terms with mixed signs and the volume cancels to nonsense, and an exported
    # mesh (the OBJ, a viewer that culls back faces) has no inside either.
    if cap_triangles:
        surface_faces = triangles[: len(triangles) - cap_triangles]
        cap_faces = triangles[len(triangles) - cap_triangles :]
        lid_centre = (
            vertices[: len(vertices) - cap_triangles].mean(axis=0)
            if len(vertices) > cap_triangles
            else np.zeros(3)
        )
        triangles = np.concatenate(
            [
                orient_mesh(vertices, surface_faces, normals),
                orient_mesh(vertices, cap_faces, None, reference=lid_centre),
            ],
            axis=0,
        )
    else:
        triangles = orient_mesh(vertices, triangles, normals)
    _progress(progress, "colour", 0.90)

    values = _vertex_property(vertices, used, index, options)
    palette = (
        "electrostatic" if options.property == "electrostatic" else (
            "grey" if options.property == "element" else "hydrophobicity"
        )
    )
    value_range = options.value_range
    if options.property == "element":
        # CPK colours come from the atom, not from a ramp: the property is
        # categorical, so there is no value axis and no legend.
        colours = np.asarray(
            [_element_color(used[i] if 0 <= i < len(used) else None) for i in index],
            dtype=float,
        )
        used_range = (0.0, 1.0)
    elif options.property == "hydrophobicity":
        colours, used_range = colorize(
            values,
            palette="hydrophobicity",
            value_range=value_range if value_range is not None else (0.0, 1.0),
        )
    else:
        if value_range is None:
            # A symmetric range keeps 0 white, which is what makes an
            # electrostatic map readable. The magnitude is a high percentile of
            # the values rather than their maximum: one buried charge owns the
            # extreme, and letting it set the scale turns the whole picture into
            # a single flat colour (a trypsin S1 pocket is mostly negative, so
            # the *variation* within the negative range is the information).
            magnitude = (
                float(np.percentile(np.abs(values), 85.0)) if values.size else 1.0
            )
            magnitude = magnitude if magnitude > 1e-6 else 1.0
            value_range = (-magnitude, magnitude)
        colours, used_range = colorize(
            values, palette="electrostatic", value_range=value_range
        )

    if cap_triangles:
        # A lid is not molecular surface: it is drawn as a neutral grey so the
        # figure does not claim a property it does not have, and its area is
        # reported separately.
        colours = np.asarray(colours, dtype=float).copy()
        colours[len(colours) - cap_triangles :] = (0.42, 0.42, 0.46)

    highlighted = None
    residue_labels: List[str] = []
    if options.highlighted_residues:
        mask = np.zeros(vertices.shape[0], dtype=bool)
        for slot, atom in enumerate(used):
            key = (
                str(getattr(atom, "chain", "") or ""),
                int(getattr(atom, "res_id", 0) or 0),
                str(getattr(atom, "res_name", "") or ""),
            )
            if key in options.highlighted_residues:
                mask |= index == slot
        highlighted = mask
        residue_labels = sorted(
            f"{chain}/{name}{res_id}" if chain else f"{name}{res_id}"
            for chain, res_id, name in options.highlighted_residues
        )

    elapsed = time.perf_counter() - started
    closed = not mesh_boundary_loops(vertices, triangles)
    lid_faces = triangles[len(triangles) - cap_triangles :] if cap_triangles else triangles[:0]
    # The caveat the caller prepared, plus whatever the charge column says on
    # its own. An electrostatic map carries it; the other properties have no
    # charge column to be wrong about.
    warning = options.charge_caveat
    if options.property == "electrostatic":
        own = charge_quality(used, charges_from_atoms(used))["warnings"]
        combined = [text for text in ([warning] if warning else []) + list(own) if text]
        warning = "; ".join(dict.fromkeys(combined)) if combined else None
    else:
        warning = None
    stats: Dict[str, object] = {
        "atoms": len(used),
        "atoms_total": len(atoms),
        "spacing": float(spacing),
        "grid": [int(size) for size in shape],
        "grid_points": int(points),
        "vertices": int(vertices.shape[0]),
        "triangles": int(triangles.shape[0]),
        "area": round(surface_area(vertices, triangles) - surface_area(vertices, lid_faces), 2),
        "closed": bool(closed),
        "volume": (
            round(surface_volume(vertices, triangles), 2) if closed else None
        ),
        "cap_triangles": int(cap_triangles),
        "cap_area": round(surface_area(vertices, lid_faces), 2),
        "sas_area": None if sas_area is None else round(sas_area, 2),
        "probe": float(options.probe),
        "directions": int(options.directions) if options.mode == "ses" else 0,
        "seconds": round(elapsed, 4),
        "highlighted_vertices": int(highlighted.sum()) if highlighted is not None else 0,
        "charge_warning": warning,
    }
    _progress(progress, "colour", 1.0)
    return Surface(
        vertices=vertices,
        normals=normals,
        colors=colours,
        values=values,
        atom_index=index,
        triangles=triangles,
        mode=options.mode,
        property_name=options.property,
        palette=palette,
        value_range=(float(used_range[0]), float(used_range[1])),
        probe=float(options.probe),
        spacing=float(spacing),
        stats=stats,
        residue_labels=residue_labels,
        highlighted=highlighted,
        selected=list(chosen),
        closed=bool(closed),
        enclosed_volume=float(surface_volume(vertices, triangles)) if closed else 0.0,
        cap_triangles=int(cap_triangles),
        cap_vertices=int(cap_triangles),
        charge_warning=warning,
    )


def axes_origin(axes: Sequence[np.ndarray]) -> np.ndarray:
    """The world coordinate of grid index ``(0, 0, 0)``."""
    return np.asarray([float(axis[0]) for axis in axes], dtype=float)


def _element_color(atom) -> Tuple[float, float, float]:
    from .structure import element_color  # local: keeps the import graph flat

    if atom is None:
        return (0.75, 0.55, 0.85)
    return element_color(str(getattr(atom, "element", "") or "C"))


def sample_field(
    field: np.ndarray,
    origin: Sequence[float],
    spacing: float,
    points: np.ndarray,
) -> np.ndarray:
    """Trilinear read of a uniform grid at arbitrary positions.

    Out-of-range positions read as ``+inf``: outside the grid every atom is
    further away than the margin the builder reserved, so the field there is
    positive and larger than any value the caller cares about. Returning
    ``+inf`` rather than a clamped node keeps a surface vertex that lands on
    the boundary from silently acquiring an interior value.
    """
    data = np.asarray(field, dtype=np.float64)
    shape = data.shape
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    origin = np.asarray(origin, dtype=float).reshape(3)
    local = (pts - origin[None, :]) / float(spacing)
    base = np.floor(local).astype(np.int64)
    frac = local - base
    out = np.full(pts.shape[0], np.inf, dtype=float)
    valid = np.ones(pts.shape[0], dtype=bool)
    for axis in range(3):
        valid &= (base[:, axis] >= 0) & (base[:, axis] < shape[axis] - 1)
    if not valid.any():
        return out
    b = base[valid]
    f = frac[valid]
    accumulated = np.zeros(b.shape[0], dtype=float)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (f[:, 0] if dx else 1.0 - f[:, 0])
                    * (f[:, 1] if dy else 1.0 - f[:, 1])
                    * (f[:, 2] if dz else 1.0 - f[:, 2])
                )
                accumulated += weight * data[
                    b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz
                ]
    out[valid] = accumulated
    return out


def _vertex_normals(
    vertices: np.ndarray,
    field: np.ndarray,
    origin: np.ndarray,
    spacing: float,
    used,
    radii: np.ndarray,
    index: np.ndarray,
) -> np.ndarray:
    """Outward normals, from the gradient of the field the mesh came from.

    The gradient of the probe-centre field points *out* of the molecule, so
    the interpolated gradient at a vertex is the true surface normal —for the
    SAS and for the SES alike, including the reentrant patches where the
    nearest atom is not the right reference and a "direction away from the
    owning atom" normal would be visibly wrong. The gradient is taken on the
    grid (three ``np.gradient`` passes, no per-vertex field evaluation) and
    read trilinearly at the vertices, which is the same construction
    ``skimage.measure.marching_cubes`` uses. The owning-atom direction is the
    fallback for a vertex that lands on the degenerate boundary of the grid.
    """
    data = np.asarray(field, dtype=np.float64)
    gradient = np.gradient(data, float(spacing), float(spacing), float(spacing))
    sampled = np.stack(
        [sample_field(component, origin, spacing, vertices) for component in gradient],
        axis=1,
    )
    del gradient, data
    length = np.linalg.norm(sampled, axis=1)
    normals = np.zeros_like(vertices, dtype=float)
    good = np.isfinite(length) & (length > 1e-9)
    normals[good] = sampled[good] / length[good, None]
    if not good.all():
        coords, _table = _atom_arrays(used, radii)
        fallback = vertices[~good] - coords[np.maximum(index[~good], 0)]
        length = np.linalg.norm(fallback, axis=1)
        normals[~good] = fallback / np.maximum(length, 1e-9)[:, None]
    return normals


def _vertex_property(
    vertices: np.ndarray,
    used,
    index: np.ndarray,
    options: "SurfaceSettings",
) -> np.ndarray:
    """The scalar on every vertex, from the selected property model."""
    if options.property == "hydrophobicity":
        per_atom = atom_hydrophobicity(
            used, scale=options.hydrophobicity_scale, bonds=options.bonds
        )
        safe = np.maximum(index, 0)
        return per_atom[safe] if per_atom.size else np.zeros(len(vertices), dtype=float)
    if options.property == "electrostatic":
        return electrostatic_potential(
            vertices,
            used,
            dielectric=options.dielectric,
            epsilon=options.epsilon,
            screening=options.screening,
        )
    return np.arange(len(vertices), dtype=float)


# ``SurfaceOptions`` is the older name of ``SurfaceSettings``; kept as an alias
# so a caller written against the first draft keeps working.
SurfaceOptions = SurfaceSettings

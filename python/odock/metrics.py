# SPDX-License-Identifier: GPL-3.0-or-later
"""Ligand-efficiency and strain metrics for docked poses.

A docking score on its own does not tell a modeller whether a hit is worth
following up: a large ligand almost always scores better than a small one, and a
lipophilic ligand is flattered by an empirical potential that rewards buried
surface.  This module computes the small set of derived numbers that are used to
make that judgement, plus the two corrections that decide whether a *pose* is
physically credible.

Efficiency metrics
------------------
With ``delta_g`` in kcal/mol and ``p = -delta_g / 1.37`` (the pIC50-equivalent
activity; ``RT ln 10 = 1.364`` kcal/mol at 298 K and **1.37** is the conventional
rounding used throughout the medicinal-chemistry literature, Hopkins 2004):

===========  ==========================================================
``LE``       ``-delta_g / HAC`` -- ligand efficiency per heavy atom
``LLE``      ``p - logP`` -- lipophilic ligand efficiency
``BEI``      ``1000 * p / MW`` -- binding efficiency index
``SEI``      ``100 * p / TPSA`` -- surface efficiency index
===========  ==========================================================

These are *definitions*, so the functions are exact given their inputs; the only
approximation is the 1.37 kcal/mol-per-log-unit conversion and the LogP/TPSA
models (RDKit's Crippen ``MolLogP`` and Ertl's fragment ``TPSA``), which are the
standard ones and are documented here rather than hidden.

Entropy
-------
:func:`entropy_penalty` is an **estimate**, not a measurement: freezing one
rotatable bond is assumed to cost ``R T ln(states_per_rotor)`` (650 cal/mol per
rotor at 298 K for a threefold rotor).  That is the crudest defensible torsional
entropy model and it is stated as such; the kernel's own Vina/AD4 torsional terms
are reported by :mod:`odock.consensus` alongside it, so the two can be compared
instead of confused.

Ligand strain
-------------
:func:`ligand_strain` reports both faces of the strain problem:

* ``strain`` -- the kernel's own ``intra`` term for the pose **minus** the same
  term for the same topology after a force-field relaxation.  Both numbers come
  from one potential, so the difference is on the docking energy scale and can be
  added to an affinity.  It is small in absolute terms for the united-atom
  potentials (Vina excludes hydrogens from the intra sum), which is itself worth
  knowing.
* ``force_field_strain`` -- the plain MMFF94 strain: the MMFF94 energy of the
  pose geometry minus that of the locally minimised conformer of the same
  molecule.  This is the number medicinal chemists mean by "strain", but it is on
  MMFF94's scale, not the docking potential's, so it is *not* added to an
  affinity.

Both are minimisations of the *same* molecule with the same atom order; the
relaxation is local (it starts from the pose geometry), so the result is a strain
relative to the nearest conformer basin, not a global conformational search.  The
relaxed geometry is returned as PDBQT so a caller can look at it.

Undefined inputs
----------------
A metric that is mathematically undefined -- zero heavy atoms, a non-finite
affinity, no LogP for LLE -- returns ``float("nan")`` rather than raising, so a
screening loop can put thousands of ligands through it and simply blank the
column.  :attr:`LigandMetrics.ok` is ``True`` only when every number that is
present is finite.  Type errors (a string where a number belongs) still raise.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "GAS_CONSTANT",
    "KCAL_PER_LOG",
    "STANDARD_TEMPERATURE",
    "STATES_PER_ROTOR",
    "METRIC_KEYS",
    "LigandMetrics",
    "StrainResult",
    "binding_efficiency_index",
    "descriptors",
    "efficiency_metrics",
    "entropy_penalty",
    "heavy_atom_count",
    "ligand_efficiencies",
    "ligand_efficiency",
    "ligand_strain",
    "lipophilic_ligand_efficiency",
    "p_activity",
    "pose_strain",
    "surface_efficiency_index",
]

#: kcal/mol per base-10 log unit of affinity at 298 K (``RT ln 10 = 1.364``;
#: 1.37 is the conventional rounding used by the efficiency literature).
KCAL_PER_LOG = 1.37

#: The gas constant in kcal/(mol K) -- ``8.31446261815324 / 4184``.
GAS_CONSTANT = 1.98720425864083e-3

#: Standard temperature for the entropy estimate, in K.
STANDARD_TEMPERATURE = 298.15

#: Effective number of torsion states a single bond loses on binding.  Three is
#: the classic threefold-rotor value; the model is linear in ``ln(states)``, so a
#: caller who prefers 6-fold counting can pass it through.
STATES_PER_ROTOR = 3.0

#: Field names of :class:`LigandMetrics`, in report order.  Stable: the CLI, the
#: GUI and the CSV writer all index by these.
METRIC_KEYS: Tuple[str, ...] = (
    "affinity",
    "heavy_atoms",
    "molecular_weight",
    "logp",
    "tpsa",
    "num_torsions",
    "p_activity",
    "ligand_efficiency",
    "lle",
    "bei",
    "sei",
    "entropy_penalty",
    "strain",
)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# Small numeric guards
# ---------------------------------------------------------------------------


def _number(value: Any, name: str) -> Optional[float]:
    """Coerce `value` to a finite float, or ``None`` when it is not usable.

    A non-numeric value is a programming error and raises; a NaN or an infinity
    is a data problem and yields ``None``, which every caller turns into
    ``float("nan")``.
    """
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a number, got {value!r}") from exc
    return number if math.isfinite(number) else None


def _count(value: Any, name: str) -> Optional[int]:
    """Coerce `value` to a non-negative integer count, or ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) if int(value) >= 0 else None
    number = _number(value, name)
    if number is None:
        return None
    if abs(number - round(number)) > 1e-9:
        raise TypeError(f"{name} must be a whole number, got {value!r}")
    return int(round(number))


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------


def p_activity(delta_g: float) -> float:
    """``pIC50``-equivalent activity: ``-delta_g / 1.37``.

    This is the conversion the efficiency literature uses: a binding free energy
    of ``-8.22`` kcal/mol is a ``pIC50`` of 6.0.  ``nan`` for a non-finite input.
    """
    value = _number(delta_g, "delta_g")
    return _NAN if value is None else -value / KCAL_PER_LOG


def ligand_efficiency(delta_g: float, heavy_atoms: int) -> float:
    """``LE = -delta_g / heavy_atoms`` in kcal/mol per heavy atom.

    ``nan`` when `delta_g` is not finite or `heavy_atoms` is not positive.
    A ligand with no heavy atom has no ligand efficiency; the value is not
    silently reported as zero.
    """
    value = _number(delta_g, "delta_g")
    count = _count(heavy_atoms, "heavy_atoms")
    if value is None or count is None or count <= 0:
        return _NAN
    return -value / count


def lipophilic_ligand_efficiency(delta_g: float, logp: float) -> float:
    """``LLE = pIC50 - logP`` with ``pIC50 = -delta_g / 1.37``.

    This is the standard definition (Leeson & Springthorpe 2007).  Note that it
    is *not* ``1.37 * LE - logP``: dividing LE by the heavy-atom count and
    subtracting logP gives a different (and much smaller) number.  ``nan`` when
    either input is not finite.
    """
    value = _number(delta_g, "delta_g")
    lipophilicity = _number(logp, "logp")
    if value is None or lipophilicity is None:
        return _NAN
    return -value / KCAL_PER_LOG - lipophilicity


def _positive(value: Any, name: str) -> Optional[float]:
    """A strictly positive finite float, ``None`` when the value is unusable.

    ``nan`` and ``inf`` are data problems and yield ``None``; a *negative* number
    is a caller mistake (a molecular weight or a polar surface area cannot be
    negative) and raises, because silently blanking it would hide a bug.
    """
    number = _number(value, name)
    if number is None:
        return None
    if number < 0:
        raise ValueError(f"{name} cannot be negative, got {value!r}")
    return number if number > 0 else None


def binding_efficiency_index(delta_g: float, molecular_weight: float) -> float:
    """``BEI = pIC50 / (MW / 1000)``.

    ``nan`` when `molecular_weight` is zero, NaN or infinite; a negative
    molecular weight raises.
    """
    value = _number(delta_g, "delta_g")
    weight = _positive(molecular_weight, "molecular_weight")
    if value is None or weight is None:
        return _NAN
    return 1000.0 * (-value / KCAL_PER_LOG) / weight


def surface_efficiency_index(delta_g: float, tpsa: float) -> float:
    """``SEI = pIC50 / (TPSA / 100)``.

    ``nan`` when `tpsa` is zero, NaN or infinite: a molecule with no polar
    surface has no surface efficiency, and dividing by zero would be a lie.
    """
    value = _number(delta_g, "delta_g")
    area = _positive(tpsa, "tpsa")
    if value is None or area is None:
        return _NAN
    return 100.0 * (-value / KCAL_PER_LOG) / area


def ligand_efficiencies(delta_g, heavy_atoms) -> np.ndarray:
    """Vectorised :func:`ligand_efficiency` over arrays of affinities and counts.

    Scalars or arrays are accepted; the result is always a ``float64`` array of
    the broadcast shape, with ``nan`` wherever the elementwise metric is
    undefined (which is exactly the scalar rule, applied elementwise).
    """
    values = np.asarray(delta_g, dtype=float)
    counts = np.asarray(heavy_atoms, dtype=float)
    shape = np.broadcast_shapes(values.shape, counts.shape)
    values = np.broadcast_to(values, shape).astype(float)
    counts = np.broadcast_to(counts, shape).astype(float)
    out = np.full(shape, _NAN, dtype=float)
    usable = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    np.divide(-values, counts, out=out, where=usable)
    return out


def entropy_penalty(
    num_torsions: float,
    *,
    temperature: float = STANDARD_TEMPERATURE,
    states_per_rotor: float = STATES_PER_ROTOR,
) -> float:
    """Torsional entropy cost of freezing `num_torsions` rotatable bonds.

    ``n * R * T * ln(states_per_rotor)`` in kcal/mol -- 0.6509 kcal/mol per rotor
    at 298.15 K with three states.  This is an **estimate**: it treats every
    rotor as independent and every bond as having the same number of states, and
    it ignores the residual entropy of the bound state.  It is meant to be
    compared with, not substituted for, the kernel's own torsional term.

    ``nan`` for a negative or non-finite torsion count.
    """
    count = _number(num_torsions, "num_torsions")
    temp = _number(temperature, "temperature")
    states = _number(states_per_rotor, "states_per_rotor")
    if count is None or temp is None or states is None:
        return _NAN
    if count < 0 or temp <= 0 or states <= 0:
        return _NAN
    return count * GAS_CONSTANT * temp * math.log(states)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def _is_mol(source) -> bool:
    return hasattr(source, "GetAtoms") and hasattr(source, "GetNumAtoms")


def _require_rdkit():
    from rdkit import Chem

    return Chem


def descriptors(mol) -> Dict[str, float]:
    """Molecular descriptors the efficiency metrics need, from an RDKit molecule.

    Returns ``heavy_atoms``, ``molecular_weight`` (average mass, RDKit
    ``MolWt``), ``logp`` (Crippen ``MolLogP``), ``tpsa`` (Ertl ``CalcTPSA``),
    ``num_torsions`` (RDKit's strict rotatable-bond count) and the donor/acceptor
    counts.  Every value is a plain float so the result is JSON-serialisable.

    ``num_torsions`` here is the *chemical* rotatable-bond count.  The kernel's
    ``num_tors`` is a different, half-weighted quantity (see ``docs/SCORING.md``)
    and is reported separately by the consensus module.
    """
    _require_rdkit()
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return {
        "heavy_atoms": float(heavy),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "num_torsions": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "num_hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "num_hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
        "num_heavy_atoms": float(heavy),
    }


def heavy_atom_count(source) -> int:
    """Number of non-hydrogen atoms in `source`.

    Accepts

    * an ``int`` (returned unchanged, negative values clamped to 0),
    * an RDKit molecule,
    * a sequence of atom-like objects exposing ``element``,
    * PDBQT text or a path to a PDBQT/PDB file.

    Hydrogen, deuterium, the ``W`` sentinel and the macrocycle closure dummies
    ``G0..G3`` are not heavy atoms.  For an ``(N, 3)`` coordinate array the
    element information simply is not there, so a :class:`TypeError` is raised
    rather than a guess being returned.
    """
    if isinstance(source, bool):
        raise TypeError("heavy_atom_count needs atoms, not a boolean")
    if isinstance(source, (int, np.integer)):
        return max(0, int(source))
    if _is_mol(source):
        return sum(1 for atom in source.GetAtoms() if atom.GetAtomicNum() > 1)
    if isinstance(source, (str, os.PathLike)):
        from .consensus import pdbqt_atoms

        text = str(source)
        if not text.lstrip().startswith(("ATOM", "HETATM", "REMARK", "ROOT", "MODEL")):
            path = os.fspath(source)
            if os.path.exists(path):
                from pathlib import Path

                text = Path(path).read_text(encoding="utf-8", errors="replace")
        return sum(1 for atom in pdbqt_atoms(text) if atom.is_heavy)
    if hasattr(source, "element") and not hasattr(source, "__len__"):
        return 1 if _is_heavy_element(str(getattr(source, "element"))) else 0
    if isinstance(source, np.ndarray):
        raise TypeError(
            "heavy_atom_count cannot tell heavy atoms from an (N, 3) coordinate "
            "array; pass a molecule, an atom sequence or a PDBQT document"
        )
    try:
        items = list(source)
    except TypeError:
        raise TypeError(
            f"cannot count heavy atoms in {type(source).__name__!r}"
        ) from None
    return sum(
        1
        for item in items
        if _is_heavy_element(str(getattr(item, "element", "") or ""))
    )


#: Elements and autoDock pseudo-types that are not heavy atoms.
_NOT_HEAVY = frozenset(("H", "D", "W", "G0", "G1", "G2", "G3", ""))


def _is_heavy_element(element: str) -> bool:
    return element.strip().upper() not in _NOT_HEAVY and bool(element.strip())


# ---------------------------------------------------------------------------
# The metric bundle
# ---------------------------------------------------------------------------


@dataclass
class LigandMetrics:
    """Every efficiency metric for one pose, plus the inputs they came from.

    A field is ``None`` when the caller did not supply the input it needs; a
    metric that is defined but not computable from the supplied inputs is
    ``nan``.  The distinction matters for a table: ``None`` is "not asked for",
    ``nan`` is "asked for and not available".
    """

    affinity: float
    heavy_atoms: Optional[int] = None
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    num_torsions: Optional[float] = None
    p_activity: float = _NAN
    ligand_efficiency: float = _NAN
    lle: float = _NAN
    bei: float = _NAN
    sei: float = _NAN
    entropy_penalty: float = _NAN
    strain: Optional[float] = None
    #: The kernel's ``intra`` term for the pose, when it was measured.
    intra: Optional[float] = None
    #: MMFF94 strain (pose minus locally minimised conformer), when measured.
    force_field_strain: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """A plain, JSON-serialisable view, in :data:`METRIC_KEYS` order.

        ``strain``/``intra``/``force_field_strain`` are appended after the keys
        in :data:`METRIC_KEYS`; ``nan`` is preserved (``json.dumps`` writes it as
        ``NaN``, which the JSONL writer is careful to normalise to ``null``).
        """
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float) and not math.isfinite(value):
                data[key] = _NAN
        return data

    @property
    def ok(self) -> bool:
        """``True`` when every number that is present is finite."""
        for key, value in self.as_dict().items():
            if value is None:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                return False
        return True

    def row(self, *, prefix: str = "") -> Dict[str, Any]:
        """The metric keys with a stable column prefix, for a results table."""
        return {f"{prefix}{key}": value for key, value in self.as_dict().items()}


def efficiency_metrics(
    affinity: float,
    *,
    mol=None,
    heavy_atoms: Optional[int] = None,
    molecular_weight: Optional[float] = None,
    logp: Optional[float] = None,
    tpsa: Optional[float] = None,
    num_torsions: Optional[float] = None,
    strain: Optional[float] = None,
) -> LigandMetrics:
    """Every efficiency metric for one pose.

    The explicit keyword arguments win over `mol`; anything missing from both is
    left ``None`` and the metrics that need it come back ``nan``.  Supplying no
    molecule at all is useful on purpose: ligand efficiency and the entropy
    penalty need nothing but the affinity and two counts, so a headless caller
    with no RDKit still gets the two most important numbers.

    Parameters
    ----------
    affinity
        The docking score in kcal/mol (the pose's reported affinity).
    mol
        An RDKit molecule, used to fill any descriptor left unset.
    strain
        A strain in kcal/mol, copied through for callers that measured it
        (:func:`ligand_strain`); it is never estimated here.
    """
    # Implicit conversion is deliberate: an integer affinity from a JSON round
    # trip must not raise.
    value = float(affinity)

    if mol is not None:
        found = descriptors(mol)
        heavy_atoms = found["heavy_atoms"] if heavy_atoms is None else heavy_atoms
        molecular_weight = (
            found["molecular_weight"] if molecular_weight is None else molecular_weight
        )
        logp = found["logp"] if logp is None else logp
        tpsa = found["tpsa"] if tpsa is None else tpsa
        num_torsions = found["num_torsions"] if num_torsions is None else num_torsions

    metrics = LigandMetrics(affinity=value)
    metrics.heavy_atoms = None if heavy_atoms is None else _count(heavy_atoms, "heavy_atoms")
    metrics.molecular_weight = (
        None if molecular_weight is None else _number(molecular_weight, "molecular_weight")
    )
    metrics.logp = None if logp is None else _number(logp, "logp")
    metrics.tpsa = None if tpsa is None else _number(tpsa, "tpsa")
    metrics.num_torsions = (
        None if num_torsions is None else _number(num_torsions, "num_torsions")
    )

    metrics.p_activity = p_activity(value)
    if metrics.heavy_atoms:
        metrics.ligand_efficiency = ligand_efficiency(value, metrics.heavy_atoms)
    if metrics.logp is not None:
        metrics.lle = lipophilic_ligand_efficiency(value, metrics.logp)
    if metrics.molecular_weight:
        metrics.bei = binding_efficiency_index(value, metrics.molecular_weight)
    if metrics.tpsa:
        metrics.sei = surface_efficiency_index(value, metrics.tpsa)
    if metrics.num_torsions is not None:
        metrics.entropy_penalty = entropy_penalty(metrics.num_torsions)
    metrics.strain = None if strain is None else _number(strain, "strain")
    return metrics


# ---------------------------------------------------------------------------
# Strain
# ---------------------------------------------------------------------------


@dataclass
class StrainResult:
    """The outcome of a ligand-strain calculation.

    Attributes
    ----------
    strain
        ``intra(pose) - intra(relaxed)`` from the **kernel's own** force field
        (``scoring=``), in kcal/mol.  Both terms come from one potential, so this
        number is on the docking energy scale.  It is ``nan`` when the kernel
        could not be reached or the pose could not be relaxed.
    intra, intra_relaxed
        The two kernel ``intra`` terms the difference is made of.
    force_field, force_field_strain, energy_pose, energy_relaxed
        The plain force-field strain (MMFF94 by default): the energy of the pose
        geometry minus that of the locally minimised conformer.  ``nan`` when no
        force field could type the molecule.
    relaxed_pdbqt
        The relaxed geometry in the **same topology and atom order** as the input
        pose, so a caller can diff or display it.
    reliable
        ``True`` only when the caller supplied the molecule (`mol=`), i.e. when
        the chemistry did not have to be inferred from a bond-order-free PDBQT.
        It is about the **force-field** strain: a molecule perceived from a PDBQT
        loses every multiple bond, and MMFF94 then relaxes a different molecule
        (measured: 68.6 kcal/mol of apparent strain for benzamidine, whose real
        value is about 2.5). The kernel ``strain`` is not affected in the same
        way, because both of its endpoints come from one potential. A caller who
        wants the unreliable case to be an error rather than a flag can pass
        ``require_reliable=True``.
    note
        Human-readable record of what was actually done (which force field ran,
        whether hydrogens had to be added, whether a 3-D conformer had to be
        embedded, and where the chemistry came from).  Always set; never a silent
        approximation.
    """

    strain: float = _NAN
    intra: float = _NAN
    intra_relaxed: float = _NAN
    force_field: str = ""
    force_field_strain: float = _NAN
    energy_pose: float = _NAN
    energy_relaxed: float = _NAN
    relaxed_pdbqt: str = ""
    note: str = ""
    reliable: bool = True

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable view without the (large) relaxed document."""
        return {
            "strain": self.strain,
            "intra": self.intra,
            "intra_relaxed": self.intra_relaxed,
            "force_field": self.force_field,
            "force_field_strain": self.force_field_strain,
            "energy_pose": self.energy_pose,
            "energy_relaxed": self.energy_relaxed,
            "reliable": self.reliable,
            "note": self.note,
        }


def _unify_residues(block: str, res_name: str = "LIG", chain: str = "A") -> str:
    """Rewrite every PDB residue name/chain in `block` to one value.

    RDKit's proximity bonding does **not** connect atoms across a change of
    residue name, and this module has to read PDBQT documents it did not write.
    :func:`odock.prepare.prepare_ligand` no longer produces a split-residue
    ligand (it inherits the parent residue onto the hydrogens it adds), but files
    written before that fix -- including the bundled ``demo/3tpb/ligand.pdbqt`` --
    and files from tools that label added hydrogens ``LIG`` still exist.

    Without this step such a document comes back with every X-H bond missing,
    which makes MMFF94 unable to type the molecule and silently degrades the
    strain to a UFF number that is dominated by the phantom hydrogens.  The
    rewrite is done on the *copy* handed to RDKit; the PDBQT itself is untouched,
    and unifying residues cannot change the chemistry of a single-residue ligand.
    """
    out: List[str] = []
    for line in block.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
            line = line[:17] + f"{res_name:>3}" + line[20:21] + f"{chain:1}" + line[22:]
        out.append(line)
    return "\n".join(out)


def _mol_from_pdbqt(text: str):
    """``(molecule, notes)`` for a PDBQT document, or ``(None, notes)``.

    The PDBQT is turned into a PDB block first (the format is a PDB dialect),
    its residues are unified so that RDKit's proximity bonding connects the polar
    hydrogens, and the molecule is sanitised best-effort.  A structure RDKit
    cannot fully sanitise still comes back: strain only needs a force field, and
    a molecule with the right atoms and bonds is enough for that.

    The notes record what RDKit had to be told, including the important
    limitation that a PDBQT carries no bond orders -- see :func:`ligand_strain`.
    """
    from rdkit import Chem

    from .prepare import pdbqt_to_pdb_block

    notes: List[str] = []
    block = _unify_residues(pdbqt_to_pdb_block(text))
    mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False, proximityBonding=True)
    if mol is None:
        try:
            mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        except Exception:  # pragma: no cover - RDKit raised rather than returning None
            return None, ["RDKit could not parse the pose PDBQT"]
    if mol is None:
        return None, ["RDKit could not parse the pose PDBQT"]
    notes.append(
        "the molecule was perceived from the PDBQT, which carries no bond orders "
        "and no aromaticity; the force-field strain is only as good as that "
        "perception -- pass mol=... with the prepared ligand for an exact one"
    )
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
            notes.append("the molecule needed a reduced RDKit sanitisation")
        except Exception:
            notes.append("RDKit could not sanitise the perceived molecule")
    return mol, notes


def _ring_is_bond_order_free(mol) -> bool:
    """Whether the molecule is an all-single-bond 5/6-ring system.

    A molecule perceived from a bare PDB (or from a PDBQT) has every bond typed
    single, so a benzene ring reads as cyclohexane.  A genuinely saturated
    ligand looks the same, which is why this can only ever *warn*: the shape is
    the fingerprint of lost bond orders, not proof of them.  A ring is required
    because a saturated chain (a fatty acid, say) is common and harmless.
    """
    from rdkit import Chem

    if any(bond.GetBondType() != Chem.BondType.SINGLE for bond in mol.GetBonds()):
        return False
    for ring in Chem.GetSymmSSSR(mol):
        atoms = list(ring)
        if len(atoms) not in (5, 6):
            continue
        if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in atoms):
            return True
    return False


def _strain_reliability(work, atoms, *, supplied: bool, preparation):
    """``(reliable, notes)`` for a molecule about to be relaxed.

    The evidence, strongest first:

    1. no caller-supplied molecule -> the chemistry was inferred from a
       bond-order-free PDBQT, so the force-field strain is not reliable;
    2. a ``PreparationReport`` whose ``chemistry_trusted`` is true -> the bond
       orders came from the input itself; reliable;
    3. a report that says they did not -> unreliable, and the report's own
       warning is repeated here;
    4. no report: the pose PDBQT's AutoDock types are compared with the molecule
       (a PDBQT that calls an atom aromatic while the molecule has no aromatic
       atom is proof that the molecule lost its bond orders), and an
       all-single-bond ring system is treated as suspicious.  Anything that fails
       is unreliable *with the reason in the note*, never silently.
    """
    from rdkit import Chem

    if not supplied:
        return False, [
            "the force-field strain is an energy difference of the molecule as "
            "perceived from the PDBQT, not necessarily of the intended ligand "
            "(a PDBQT carries no bond orders); pass mol= with the prepared "
            "molecule to make it reliable"
        ]

    notes: List[str] = []
    if preparation is not None:
        trusted = getattr(preparation, "chemistry_trusted", None)
        warnings = list(getattr(preparation, "warnings", []) or [])
        if trusted:
            notes.append("the supplied molecule's bond orders came from the input")
            return True, notes
        if trusted is False:
            notes.append(
                "the supplied molecule was prepared without bond orders "
                "(chemistry_trusted is False); its aromatic rings and carbonyls "
                "are guesses, so the force-field strain belongs to a different "
                "molecule than the intended ligand"
            )
            for warning in warnings:
                if "bond order" in warning:
                    notes.append(f"preparation said: {warning}")
            return False, notes

    aromatic_types = any(atom.ad_type.upper() == "A" for atom in atoms)
    has_aromatic = any(atom.GetIsAromatic() for atom in work.GetAtoms())
    if aromatic_types and not has_aromatic:
        notes.append(
            "the pose PDBQT types atoms as aromatic but the supplied molecule has "
            "no aromatic atom: its bond orders were lost between the two"
        )
        return False, notes
    if _ring_is_bond_order_free(work):
        notes.append(
            "the supplied molecule is an all-single-bond ring system, which is "
            "what a molecule perceived from a bare PDB looks like; pass the "
            "PreparationReport as preparation= to attest the chemistry, or ignore "
            "this if the ligand really is saturated"
        )
        return False, notes
    notes.append("the supplied molecule's atom order and chemistry were checked")
    del Chem
    return True, notes


def ligand_strain(
    pose_pdbqt: str,
    *,
    receptor=None,
    box=None,
    scoring: str = "vina",
    force_field: str = "MMFF94",
    steps: int = 500,
    mol=None,
    atom_order: Optional[Sequence[int]] = None,
    preparation=None,
    require_reliable: bool = False,
    topology_source: Optional[str] = None,
) -> StrainResult:
    """Ligand strain of one pose: kernel ``intra`` plus an MMFF94 relaxation.

    Parameters
    ----------
    pose_pdbqt
        The pose as a single-model PDBQT document (``ROOT``/``BRANCH`` tree
        included).  Its atom order defines the relaxed geometry's atom order.
    receptor, box, scoring
        Passed to :func:`odock.score` to evaluate the kernel's ``intra`` term.
        Without a receptor the kernel half is skipped and only the force-field
        strain is reported -- the ``note`` says so.
    force_field, steps
        The relaxation, driven by :func:`odock.chem.ligand.minimize`, which owns
        the documented fallback chain (MMFF94 -> MMFF94s -> UFF) and clamps the
        step count.  MMFF94 cannot type a united-atom structure that has had its
        non-polar hydrogens merged, so hydrogens are added to the *copy* used for
        the relaxation before the force field is asked to type it; the atom order
        of the original atoms is preserved, which is what makes the coordinates
        transferable back onto the pose.
    mol
        An RDKit molecule to relax instead of one perceived from `pose_pdbqt`.
        Its atom order must be the PDBQT's own order (the *k*-th atom of the
        molecule is the *k*-th ``ATOM`` record) -- that is checked, and a
        mismatch raises instead of quietly patching coordinates onto the wrong
        atoms.  Pass `atom_order` instead of renumbering by hand.
    atom_order
        ``report.atom_order`` from :func:`odock.prepare.prepare_ligand`: the RDKit
        atom indices in the kernel's order.  With `mol`, the molecule is
        renumbered into the pose's order before use, which makes the documented
        recipe a single call for a ligand prepared from a SMILES or an SDF.
    preparation
        ``report`` from :func:`odock.prepare.prepare_ligand`.  Its
        ``chemistry_trusted`` flag is what attests the molecule's chemistry: bond
        orders that came from the input (SDF, MOL2, MOL, SMILES) are real, while
        bond orders *inferred* from geometry belong to a molecule that may not be
        the intended ligand at all.
    require_reliable
        Raise :class:`ValueError` instead of returning an unreliable result.
        The kernel ``strain`` may still be usable in that case; the force-field
        one is the part that cannot be trusted without the chemistry.
    topology_source
        A PDBQT document whose topology should be kept when patching the relaxed
        coordinates back (for a caller whose `mol` came from a different file).

    Returns
    -------
    :class:`StrainResult`.  Nothing raises for a structure that cannot be typed
    or relaxed; the failure is reported in ``note`` with ``nan`` energies -- but
    a *mismatched* `mol` does raise, because that is a caller mistake that would
    otherwise produce a plausible-looking wrong number.

    Notes
    -----
    ``reliable`` is decided from evidence rather than from optimism:

    * a molecule perceived from the PDBQT is **never** reliable -- a PDBQT has no
      bond orders, and the measured error for benzamidine is 68.6 kcal/mol
      against a true 2.5;
    * a caller-supplied molecule is reliable when a `preparation` report attests
      that its bond orders came from the input *and* the atom order and element
      sequence match the pose;
    * a caller-supplied molecule **without** a report is reliable only when
      nothing looks wrong: the pose PDBQT does not call an atom aromatic while
      the molecule has no aromatic atom, and the molecule is not an
      all-single-bond ring system -- the shape a molecule perceived from a bare
      PDB takes, and the one case that cannot be distinguished from a genuinely
      saturated ligand by looking at the molecule alone.  A false positive costs
      a flag and a note; a false negative costs a strain that is wrong by two
      orders of magnitude.

    The documented recipe is therefore the only path that *guarantees* a reliable
    force-field strain::

        mol, pdbqt, prep = odock.prepare_ligand("ligand.sdf", smiles=...)   # or smiles=
        strain = odock.metrics.ligand_strain(
            pose_pdbqt, receptor=receptor, box=box,
            mol=mol, atom_order=prep.atom_order, preparation=prep,
        )
        assert strain.reliable
    """
    from .consensus import pdbqt_atoms, with_coordinates

    atoms = pdbqt_atoms(pose_pdbqt)
    note_parts: List[str] = []
    result = StrainResult()

    # -- the kernel's own intra term ---------------------------------------
    if receptor is not None:
        try:
            import odock

            components = odock.score(
                receptor, pose_pdbqt, box, scoring=scoring, refine=False
            )
            result.intra = float(components.get("intra", _NAN))
            note_parts.append(f"kernel intra from {scoring}")
        except Exception as exc:
            note_parts.append(f"the kernel could not score the pose: {exc}")
    else:
        note_parts.append("no receptor was given, so the kernel intra term was skipped")

    # -- the force-field relaxation ----------------------------------------
    if not atoms:
        result.note = "; ".join(note_parts + ["the pose PDBQT carries no atoms"])
        return result

    try:
        from .chem import ligand as _ligand

        _ligand.require_rdkit()
    except Exception as exc:
        result.note = "; ".join(note_parts + [f"RDKit is unavailable: {exc}"])
        return result

    from rdkit import Chem

    if mol is not None:
        work = Chem.Mol(mol)
        if atom_order is not None:
            # The documented recipe: the molecule as prepared, plus the index
            # mapping the kernel used.  Renumbering here is what makes the
            # coordinates transferable without the caller having to remember
            # which order the PDBQT is in.
            order = [int(index) for index in atom_order]
            if sorted(order) != list(range(work.GetNumAtoms())):
                raise ValueError(
                    f"atom_order must be a permutation of 0..{work.GetNumAtoms() - 1}; "
                    f"got {len(order)} entries covering "
                    f"{len(set(order))} distinct atom(s)"
                )
            work = Chem.RenumberAtoms(work, order)
        # A caller-supplied molecule is written back onto the pose document by
        # *file order*, so the orders have to agree.  A molecule prepared from a
        # SMILES, for instance, starts with whichever atom the SMILES did: using
        # it unchecked would patch the relaxed coordinates onto the wrong atoms
        # and still return a plausible number.
        expected = [atom.element.upper() for atom in atoms]
        got = [atom.GetSymbol().upper() for atom in work.GetAtoms()][: len(expected)]
        if got != expected:
            raise ValueError(
                "the supplied molecule's atom order does not match the pose PDBQT: "
                f"expected {len(expected)} atom(s) starting {expected[:8]}, got "
                f"{got[:8]}.  Pass the molecule in the PDBQT's own order, or give "
                "atom_order=report.atom_order from odock.prepare.prepare_ligand, or "
                "omit mol= and let the chemistry be perceived from the PDBQT (which "
                "makes the force-field strain unreliable rather than wrong in a way "
                "nobody can see)."
            )
    else:
        work, perception_notes = _mol_from_pdbqt(pose_pdbqt)
        note_parts.extend(perception_notes)
    if work is None:
        result.note = "; ".join(
            note_parts + ["RDKit could not perceive a molecule from the pose PDBQT"]
        )
        return result

    result.reliable, reliability_notes = _strain_reliability(
        work, atoms, supplied=mol is not None, preparation=preparation
    )
    note_parts.extend(reliability_notes)
    if require_reliable and not result.reliable:
        raise ValueError(
            "ligand_strain cannot produce a reliable force-field strain here: "
            + "; ".join(note_parts)
        )

    work = Chem.Mol(work)
    if sum(1 for a in work.GetAtoms() if a.GetAtomicNum() == 1) == 0:
        try:
            work = Chem.AddHs(work, addCoords=True)
            note_parts.append("hydrogens were added with coordinates for the relaxation")
        except Exception as exc:  # pragma: no cover - AddHs rarely fails
            note_parts.append(f"hydrogens could not be added ({exc})")

    try:
        relaxed = _ligand.minimize(work, force_field=force_field, steps=steps)
    except Exception as exc:  # pragma: no cover - minimize does not raise by design
        result.note = "; ".join(note_parts + [f"the relaxation failed: {exc}"])
        return result

    used = relaxed.GetProp("odock_force_field") if relaxed.HasProp("odock_force_field") else ""
    result.force_field = used
    detail = relaxed.GetProp("odock_minimize_note") if relaxed.HasProp("odock_minimize_note") else ""
    if detail:
        note_parts.append(detail)
    try:
        result.energy_pose = float(relaxed.GetProp("odock_energy_before"))
        result.energy_relaxed = float(relaxed.GetProp("odock_energy_after"))
    except (KeyError, ValueError):
        pass

    if not (math.isfinite(result.energy_pose) and math.isfinite(result.energy_relaxed)):
        result.note = "; ".join(note_parts + ["no force field could type the molecule"])
        return result

    result.force_field_strain = result.energy_pose - result.energy_relaxed

    conf = relaxed.GetConformer()
    relaxed_coords = np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(work.GetNumAtoms())
        ],
        dtype=float,
    )
    if relaxed_coords.shape[0] < len(atoms):
        result.note = "; ".join(note_parts + ["the relaxed molecule lost atoms"])
        return result

    source = topology_source if topology_source is not None else pose_pdbqt
    result.relaxed_pdbqt = with_coordinates(source, relaxed_coords[: len(atoms)])

    if receptor is not None and math.isfinite(result.intra):
        try:
            import odock

            relaxed_components = odock.score(
                receptor, result.relaxed_pdbqt, box, scoring=scoring, refine=False
            )
            result.intra_relaxed = float(relaxed_components.get("intra", _NAN))
            if math.isfinite(result.intra_relaxed):
                result.strain = result.intra - result.intra_relaxed
        except Exception as exc:
            note_parts.append(f"the relaxed geometry could not be scored: {exc}")

    note_parts.append(
        "strain is intra(pose) - intra(relaxed) in the same force field; the "
        "relaxation is local (from the pose geometry) and uses " + (used or force_field)
    )
    result.note = "; ".join(note_parts)
    return result


def pose_strain(
    result,
    pose,
    *,
    receptor=None,
    box=None,
    scoring: Optional[str] = None,
    **kwargs,
) -> StrainResult:
    """Ligand strain of `pose` inside a :class:`odock.DockResult`.

    The pose document is rebuilt from ``result.ligand_pdbqt`` (the topology
    template) and ``pose.coords`` (the kernel's atom order), so the strain is
    measured on exactly the coordinates the run produced -- including poses the
    run's own multi-model file dropped beyond its energy window.

    `scoring` defaults to whatever force field the result was produced with;
    `receptor` defaults to ``result.receptor_pdbqt`` and `box` to
    ``result.box``.
    """
    from .consensus import pose_pdbqt_from_coords

    template = getattr(result, "ligand_pdbqt", "") or ""
    if scoring is None:
        scoring = getattr(result, "scoring", None) or "vina"
    if receptor is None:
        receptor = getattr(result, "receptor_pdbqt", "") or None
    if box is None:
        box = getattr(result, "box", None)

    text = pose_pdbqt_from_coords(template, pose)
    return ligand_strain(
        text, receptor=receptor, box=box, scoring=scoring, topology_source=text, **kwargs
    )

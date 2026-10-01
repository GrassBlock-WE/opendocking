# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-atom partial charges and AutoDock 4 atom typing.

Two charge models are exposed, exactly as the project brief asks:

* :func:`gasteiger_charges` — the Gasteiger-Marsili (1980) electronegativity
  equalisation model, computed by RDKit (:func:`rdkit.Chem.AllChem.ComputeGasteigerCharges`).
* :func:`kollman_charges` — the *united-atom* charge convention used by the
  Kollman/AMBER force fields: every non-polar hydrogen is merged into its parent
  heavy atom, so a hydrogen-suppressed PDBQT carries the merged charge on the
  carbon instead of on a hydrogen that the file no longer contains.

Provenance of the Kollman model (read this before trusting the numbers)
----------------------------------------------------------------------
The Kollman all-atom parameter set (Weiner *et al.*, *J. Am. Chem. Soc.* **106**,
765 (1984); the ff94 revision, Cornell *et al.*, *J. Am. Chem. Soc.* **117**,
5179 (1995)) is a *tabulated* set of per-residue charges. RDKit does not ship
that table and the table itself is not available to this project (it is not in
RDKit, not in the repository and the build environment has no network access).
What :func:`kollman_charges` therefore implements is the **documented united-atom
scheme** — merge non-polar hydrogens onto their parent, keep the formal charge
of the molecule intact — applied on top of a *published, reproducible* RDKit
charge engine:

1. MMFF94 partial charges (Halgren, *J. Comput. Chem.* **17**, 490 (1996)) when
   RDKit has parameters for every atom, because MMFF94 is a bond-charge-increment
   model of the same physical character as the Kollman set and its charges
   already sum to the molecule's net charge;
2. otherwise Gasteiger-Marsili, which RDKit can compute for any molecule it can
   parametrise. When it cannot — and a protein read without explicit hydrogens
   is the standard case — every entry is ``0.0`` rather than a partial,
   charge-non-conserving set; see :func:`gasteiger_charges`.

This is an honest approximation, not a transcription of the Kollman parameter
table: the returned values have the right *character* (united-atom, charge
conserving, dipole-bearing) but they are **not** the published Kollman numbers.
Callers that need the literal Kollman set must supply their own per-residue
table. The name is kept because the frozen interface exposes it under that name.

Atom typing
-----------
:func:`ad4_types` maps every atom to the AutoDock 4 dictionary used by
``dock-core`` (see ``crates/dock-core/src/atom.rs``). Two details matter:

* every hydrogen the model does not type is written as the sentinel ``"W"`` —
  in the Rust kernel ``AdType::Unknown.name()`` and ``XsType::W`` both resolve
  to that string, and it is what makes the scorer skip the atom;
* elements that the AD4 table has no type for (Na, K, Cu, ...) are also ``"W"``,
  matching Vina's ``non_ad_metal_names`` behaviour.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "AD4_TYPES",
    "CHARGE_MODELS",
    "ad4_type_summary",
    "ad4_types",
    "assign_charges",
    "gasteiger_charges",
    "kollman_charges",
]

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import rdBase

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]
    _HAVE_RDKIT = False


class _quiet_rdkit:
    """Suppress RDKit's advisory logging for one call.

    ``ComputeGasteigerCharges`` writes "Molecule does not have explicit Hs.
    Consider calling AddHs()" to stderr for every hydrogen-less molecule. That
    is expected here — the caller may legitimately ask for charges on a crystal
    structure — and a library must not spam the host application's console, so
    the message is blocked for the duration of the call. Real failures are still
    visible: they arrive as exceptions or as non-finite values, both of which
    this module handles explicitly.
    """

    def __enter__(self):
        self._blocked = False
        if rdBase is not None:
            try:
                rdBase.BlockLogs()
                self._blocked = True
            except Exception:  # pragma: no cover - defensive
                self._blocked = False
        return self

    def __exit__(self, *exc_info):
        if self._blocked:
            try:
                rdBase.UnBlockLogs()
            except Exception:  # pragma: no cover - defensive
                pass
        return False


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for charge assignment and atom typing. Install it "
            "with `pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

#: Every AutoDock 4 atom type this module can emit, plus the ``"W"`` sentinel.
#: The order follows ``ATOM_KIND_NAMES`` in ``crates/dock-core/src/atom.rs``.
AD4_TYPES: Tuple[str, ...] = (
    "C", "A", "N", "NA", "OA", "O", "SA", "S", "P", "HD", "H",
    "F", "Cl", "Br", "I", "Si", "At",
    "Mg", "Mn", "Zn", "Ca", "Fe", "W",
)

#: Accepted values of the ``model`` argument of :func:`assign_charges`.
CHARGE_MODELS: Tuple[str, ...] = ("gasteiger", "kollman", "mmff94", "none")

#: Elements with a dedicated AD4 type.
_AD4_METAL_ELEMENTS = frozenset({"Mg", "Mn", "Zn", "Ca", "Fe"})

#: Elements that are metals but have **no** AD4 type. Vina's
#: ``non_ad_metal_names`` list maps them to the unknown sentinel; in this module
#: they become ``"W"`` (which is what ``AdType::Unknown.name()`` renders).
_UNTYPED_METAL_ELEMENTS = frozenset(
    {
        "Li", "Na", "K", "Rb", "Cs", "Be", "Sr", "Ba", "Sc", "Ti", "V", "Cr",
        "Co", "Ni", "Cu", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hg", "Pt", "Au",
        "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "U",
    }
)

#: Elements whose symbol is the AD4 type verbatim.
_AD4_SIMPLE_ELEMENTS = frozenset({"F", "Cl", "Br", "I", "Si", "At", "P"})

#: Vina's ``atom_equivalence_data``: selenium is docked as sulfur.
_AD4_ELEMENT_ALIASES = {"Se": "S", "D": "H", "T": "H"}

#: The H-bond **acceptor** definition, identical to the one in
#: :mod:`odock.pdbqt` (RDKit's own ``BaseFeatures.fdef`` pattern). Donors need
#: no SMARTS here: in the AD4 dictionary a hydrogen is ``HD`` exactly when it is
#: bound to N, O or S, which the bond graph already says.
_ACCEPTOR_SMARTS = (
    "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0])]"
)


def _smarts(pattern: str):
    """Compile a SMARTS once, tolerating a broken RDKit build."""
    if not _HAVE_RDKIT:  # pragma: no cover - guarded by require_rdkit
        return None
    try:
        return Chem.MolFromSmarts(pattern)
    except Exception:  # pragma: no cover - defensive
        return None


def _perception_copy(mol):
    """Return a sanitised copy of `mol` when that is possible.

    Receptors read from a PDB file are deliberately *not* sanitised by the
    readers (a protein often contains a valence the default model rejects), but
    aromaticity, ring perception and the MMFF94 parameter lookup all need the
    property cache. This helper gives the charge/typing code the best available
    version of the molecule and never raises: when sanitisation fails the
    original copy is used and the fallbacks inside the individual functions take
    over.
    """
    require_rdkit()
    work = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(work)
        return work
    except Exception:
        pass
    try:
        Chem.FastFindRings(work)
        work.UpdatePropertyCache(strict=False)
    except Exception:  # pragma: no cover - defensive
        pass
    return work


def _acceptors(mol) -> set:
    """Indices of H-bond acceptors, via SMARTS with a valence fallback."""
    out: set = set()
    pattern = _smarts(_ACCEPTOR_SMARTS)
    if pattern is not None:
        try:
            for match in mol.GetSubstructMatches(pattern):
                out.add(int(match[0]))
            return out
        except Exception:
            out.clear()
    # Fallback: no ring information or no SMARTS engine. The rule below is the
    # documented AD4 one -- a neutral N/O/S with a lone pair is an acceptor --
    # expressed through the valence that is available without ring perception.
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in (7, 8, 16) or atom.GetFormalCharge() > 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if z == 8:
            out.add(atom.GetIdx())
        elif z == 7:
            if len(heavy) <= 2:
                out.add(atom.GetIdx())
            elif len(heavy) == 3 and not _is_amide_nitrogen(atom):
                out.add(atom.GetIdx())
        else:  # sulfur
            if len(heavy) <= 1:
                out.add(atom.GetIdx())
    return out


def _is_amide_nitrogen(atom) -> bool:
    """`True` when a nitrogen is the N of an amide/peptide bond.

    A peptide nitrogen donates its hydrogen but keeps its lone pair in the
    carbonyl resonance, so AD4 types it ``N`` rather than ``NA``.
    """
    for n in atom.GetNeighbors():
        if n.GetAtomicNum() != 6:
            continue
        for nn in n.GetNeighbors():
            if nn.GetAtomicNum() != 8 or nn.GetIdx() == atom.GetIdx():
                continue
            bond = atom.GetOwningMol().GetBondBetweenAtoms(n.GetIdx(), nn.GetIdx())
            if bond is not None and bond.GetBondTypeAsDouble() >= 1.9:
                return True
    return False


def _is_aromatic(atom) -> bool:
    """``atom.GetIsAromatic()`` with the property-cache precondition handled."""
    try:
        return bool(atom.GetIsAromatic())
    except Exception:  # pragma: no cover - unsanitised molecule
        return False


# ---------------------------------------------------------------------------
# AD4 atom typing
# ---------------------------------------------------------------------------


def ad4_types(mol) -> List[str]:
    """The AutoDock 4 atom type of every atom, index-aligned with `mol`.

    The dictionary is the one ``dock-core`` parses (``atom.rs``):

    ============  ==========================================================
    type          assigned when
    ============  ==========================================================
    ``A`` / ``C``  aromatic / aliphatic carbon
    ``N``         nitrogen that is not an H-bond acceptor (amide, ammonium)
    ``NA``        H-bond-accepting nitrogen
    ``O``         oxygen that cannot accept an H-bond (e.g. oxonium)
    ``OA``        H-bond-accepting oxygen -- every neutral carbonyl,
                  hydroxyl, ether, carboxylate
    ``S``         sulfur that is not an acceptor
    ``SA``        H-bond-accepting sulfur (thiol, thioether, thiolate)
    ``P``         phosphorus
    ``HD``        hydrogen bound to N, O or S (the polar hydrogen)
    ``W``         **sentinel**: hydrogen with no AD4 type of its own and
                  every element the AD4 table does not cover
    ``F Cl Br I`` halogens, ``Si At`` metalloids
    ``Mg Mn Zn Ca Fe``
                  the metals that have an AD4 type
    ============  ==========================================================

    Why non-polar hydrogens become ``W`` and not ``H``
    --------------------------------------------------
    In the AutoDock united-atom convention a hydrogen bound to carbon does not
    exist in the docking model -- its charge and its van der Waals volume have
    been merged into the carbon. ``crates/dock-core/src/atom.rs`` expresses that
    with ``XsType::W``, the "no X-Score type" sentinel that every grid map,
    pair loop and gradient skips, and ``AdType::Unknown.name()`` renders to the
    same string. Emitting ``W`` therefore says exactly what is true: this atom
    carries no interaction type. (``odock.pdbqt`` writes ``H`` for the same
    case because it mirrors Vina's PDBQT writer byte for byte; the two are
    interchangeable on input, ``AdType::from_name`` accepts both.)

    Elements without an AD4 type (Na, K, Cu, Hg, ...) also become ``W``, which
    is Vina's ``non_ad_metal_names`` behaviour. Selenium is aliased to ``S``,
    matching Vina's ``atom_equivalence_data``.
    """
    require_rdkit()
    work = _perception_copy(mol)
    acceptors = _acceptors(work)

    types: List[str] = []
    for atom in work.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        symbol = _AD4_ELEMENT_ALIASES.get(symbol, symbol)
        z = atom.GetAtomicNum()
        if z == 1:
            heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            polar = bool(heavy) and heavy[0].GetAtomicNum() in (7, 8, 16)
            types.append("HD" if polar else "W")
        elif symbol == "C":
            types.append("A" if _is_aromatic(atom) else "C")
        elif symbol == "N":
            types.append("NA" if idx in acceptors else "N")
        elif symbol == "O":
            types.append("OA" if idx in acceptors else "O")
        elif symbol == "S":
            types.append("SA" if idx in acceptors else "S")
        elif symbol in _AD4_SIMPLE_ELEMENTS:
            types.append(symbol)
        elif symbol in _AD4_METAL_ELEMENTS:
            types.append(symbol)
        elif symbol in _UNTYPED_METAL_ELEMENTS:
            types.append("W")
        else:
            types.append("W")
    return types


def ad4_type_summary(mol) -> Dict[str, int]:
    """How many atoms carry each AD4 type, e.g. ``{"C": 12, "OA": 2, "W": 6}``.

    Types are reported in :data:`AD4_TYPES` order; an unknown type (which no
    code path should produce) would be appended at the end rather than dropped.
    """
    counts: Dict[str, int] = {}
    for name in ad4_types(mol):
        counts[name] = counts.get(name, 0) + 1
    ordered: Dict[str, int] = {t: counts.pop(t) for t in AD4_TYPES if t in counts}
    ordered.update(counts)
    return ordered


# ---------------------------------------------------------------------------
# Partial charges
# ---------------------------------------------------------------------------


def _stored_charges(mol, prop: str):
    """Read a complete per-atom charge set from an atom property, or ``None``.

    ``strip_nonpolar_hydrogens(collapse_charges=True)`` publishes the merged
    charges on the surviving atoms; honouring them here is what makes the
    collapse meaningful end to end (recomputing Gasteiger on a
    hydrogen-suppressed graph gives different, worse numbers). The property is
    only honoured when *every* atom carries it, so a partially populated
    molecule can never masquerade as a complete charge set.
    """
    values = []
    for atom in mol.GetAtoms():
        if not atom.HasProp(prop):
            return None
        try:
            value = float(atom.GetProp(prop))
        except Exception:
            return None
        values.append(0.0 if not math.isfinite(value) else value)
    return values if values else None


def _gasteiger_raw(mol):
    """``(values, converged)`` for the Gasteiger-Marsili model.

    RDKit's implementation is an iterative electronegativity equalisation that
    needs a molecule it can parametrise. On a large structure read without
    explicit hydrogens — an X-ray protein is the canonical case — it returns
    ``nan``/``inf`` for a fraction of the atoms, and the surviving values no
    longer sum to the molecule's net charge. ``converged`` reports exactly that:
    every value finite **and** the total equal to the net formal charge. A set
    that fails the test is a partial, charge-non-conserving one.
    """
    require_rdkit()
    work = _perception_copy(mol)
    count = work.GetNumAtoms()
    try:
        with _quiet_rdkit():
            AllChem.ComputeGasteigerCharges(work)
    except Exception:
        # A molecule RDKit cannot perceive (an unsanitised protein, an exotic
        # element) legitimately has no Gasteiger charges.
        return [0.0] * count, False
    values: List[float] = []
    converged = True
    for atom in work.GetAtoms():
        if not atom.HasProp("_GasteigerCharge"):
            converged = False
            values.append(0.0)
            continue
        try:
            value = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            converged = False
            value = 0.0
        if not math.isfinite(value):
            converged = False
            value = 0.0
        values.append(value)
    if len(values) != count:  # pragma: no cover - defensive
        return [0.0] * count, False
    try:
        net = float(Chem.GetFormalCharge(work))
    except Exception:  # pragma: no cover - defensive
        net = 0.0
    if abs(sum(values) - net) > 1e-3 * max(1.0, abs(net)):
        converged = False
    return values, converged


def gasteiger_charges(mol) -> np.ndarray:
    """Gasteiger-Marsili partial charges, one per atom, as ``(N,) float64``.

    RDKit computes the published Gasteiger-Marsili electronegativity
    equalisation; the charges already sum to the molecule's net formal charge
    (RDKit seeds the iteration with the formal charges), so no correction is
    applied.

    **Non-convergence.** RDKit reports ``nan``/``inf`` for atoms it cannot
    parametrise, which is common for a protein read without explicit hydrogens.
    The surviving values then no longer conserve the total charge, so the
    function returns an all-zero array instead of a partial set: zeros mean
    "this model has nothing to say here", while a charge-non-conserving set
    would silently bias the AD4 electrostatics. Finiteness *and* the total are
    both checked, so a set that merely drifts is caught as well.

    **Precedence.** When every atom carries a ``_GasteigerCharge`` property
    (which is what :func:`odock.chem.receptor.strip_nonpolar_hydrogens` leaves
    behind after merging non-polar hydrogens) those values are returned
    unchanged: they are the charges of the *original* molecule, and recomputing
    the model on the hydrogen-suppressed graph would silently change them.
    """
    require_rdkit()
    stored = _stored_charges(mol, "_GasteigerCharge")
    if stored is not None:
        return np.asarray(stored, dtype=float)
    values, converged = _gasteiger_raw(mol)
    if not converged:
        return np.zeros(mol.GetNumAtoms(), dtype=float)
    return np.asarray(values, dtype=float)


def _mmff94_raw(mol):
    """MMFF94 partial charges, or ``None`` when RDKit lacks the parameters."""
    require_rdkit()
    work = _perception_copy(mol)
    try:
        if not AllChem.MMFFHasAllMoleculeParams(work):
            return None
        props = AllChem.MMFFGetMoleculeProperties(work)
    except Exception:
        return None
    if props is None:
        return None
    values = []
    for idx in range(work.GetNumAtoms()):
        try:
            value = float(props.GetMMFFPartialCharge(idx))
        except Exception:
            return None
        values.append(value if math.isfinite(value) else 0.0)
    return values


def kollman_charges(mol) -> np.ndarray:
    """Kollman **united-atom** charges, one per atom, as ``(N,) float64``.

    The united-atom convention is realised in place so that the array stays
    index-aligned with `mol`: the charge of every hydrogen bound to carbon is
    added onto its parent carbon and the hydrogen's own entry becomes ``0.0``.
    A caller that then writes a hydrogen-merged PDBQT reads exactly the merged
    value off the carbon, and ``charges.sum()`` still equals the molecule's net
    formal charge.

    Which base model is used (and why) is documented in the module docstring:
    MMFF94 when RDKit has parameters for the whole molecule, the guarded
    Gasteiger-Marsili set otherwise. **This is the Kollman united-atom
    *scheme*, not a transcription of the Kollman/AMBER per-residue parameter
    table**, which is unavailable to this build. The values are united-atom and
    charge-conserving, but they are not the published Kollman numbers.

    When neither base model is usable — an unsanitised protein has no MMFF94
    parameters and makes the Gasteiger iteration diverge — every entry is
    ``0.0``. That is a deliberate, documented limitation: a receptor that needs
    AD4 electrostatics must be given charges from an explicit Kollman/AMBER
    parameter set by the caller.

    The per-atom values are also published on the atoms as the
    ``_KollmanCharge`` property so that a writer can pick them up without
    recomputing the model.
    """
    require_rdkit()
    base = _mmff94_raw(mol)
    if base is None:
        values, converged = _gasteiger_raw(mol)
        base = values if converged else [0.0] * mol.GetNumAtoms()
    charges = np.asarray(base, dtype=float)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        if not heavy or heavy[0].GetAtomicNum() != 6:
            continue
        parent = heavy[0].GetIdx()
        charges[parent] += charges[atom.GetIdx()]
        charges[atom.GetIdx()] = 0.0

    for atom in mol.GetAtoms():
        atom.SetDoubleProp("_KollmanCharge", float(charges[atom.GetIdx()]))
    return charges


def assign_charges(mol, model: str = "gasteiger") -> np.ndarray:
    """Per-atom charges from the named model.

    Parameters
    ----------
    model
        One of :data:`CHARGE_MODELS` (case-insensitive):

        * ``"gasteiger"`` (default) -- :func:`gasteiger_charges`;
        * ``"kollman"`` -- :func:`kollman_charges`, the united-atom scheme;
        * ``"mmff94"`` -- RDKit's MMFF94 partial charges, *without* the
          united-atom merge (useful when the caller wants to compare models);
        * ``"none"`` -- all zeros, for force fields that ignore charges
          (Vina and Vinardo do).

    Raises
    ------
    ValueError
        For an unknown model name, with the accepted names in the message.
    """
    require_rdkit()
    key = str(model).strip().lower()
    if key in ("gasteiger", "gasteiger-marsili", "gm"):
        return gasteiger_charges(mol)
    if key in ("kollman", "kollman_united_atom", "kollman-united-atom", "united_atom"):
        return kollman_charges(mol)
    if key in ("mmff94", "mmff", "mmff94s"):
        raw = _mmff94_raw(mol)
        if raw is None:
            raise ValueError(
                "MMFF94 charges are not available for this molecule (RDKit has no "
                "parameters for at least one atom); use model='gasteiger' or "
                "model='kollman' instead"
            )
        return np.asarray(raw, dtype=float)
    if key in ("none", "zero", "zeros"):
        return np.zeros(mol.GetNumAtoms(), dtype=float)
    raise ValueError(
        f"unknown charge model {model!r}; expected one of {list(CHARGE_MODELS)}"
    )

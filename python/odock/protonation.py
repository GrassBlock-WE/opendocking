# SPDX-License-Identifier: GPL-3.0-or-later
"""Protonation state: the assumption every electrostatic term in this project makes.

A charge model is not a chemical model.  Gasteiger, MMFF94 and a PDBQT written from a
neutral SMILES all answer "how are the electrons distributed in *this* molecule", and
none of them asks whether the molecule is the one that binds.  Benzamidine is the
worked example, measured rather than imagined: prepared in its neutral form it carries
**−0.30 e on each amidine nitrogen**, so in trypsin's Asp189 pocket — whose potential
at those atoms is −42 kcal/(mol·e) — the ligand-receptor Coulomb term comes out
**+12.9 and +14.2 kcal/mol, +21 kcal/mol in total and unfavourable** (see
`docs/POCKET_SCORE.md` §3.1).  The bound species is the *amidinium*, and its salt
bridge with Asp189 is the interaction that defines the pocket.  A term that says
"+21 kcal/mol" for that complex has the wrong sign, not a small error.

This module exists so that the assumption is **stated and checked** rather than
inherited:

* :func:`detect_groups` — the formally charged groups a molecule contains
  (amidinium/guanidinium, carboxylate, phosphate, sulfonate, amine, imidazole,
  thiolate), with the charge each carries at a stated pH;
* :func:`protonation_report` — what state the molecule is *in*, what the families in
  it *would* be at that pH, and the molecules where the two disagree;
* :func:`salt_bridge_warnings` — the geometric contradiction: a cationic group left
  **neutral** within salt-bridge distance of an anionic receptor residue (or the
  reverse), which is a checkable condition and is attached to the preparation report
  as a warning rather than left to a document;
* :func:`charged_copy` — the explicitly protonated/deprotonated form, so a caller can
  measure what the assumption costs.

**What this does not do.**  It does not predict a pKa, it does not titrate a
protein, and it does not decide which tautomer binds.  It reports the state the
pipeline produced and flags the cases where that state contradicts the pocket it is
about to be scored against.  `docs/PROTONATION.md` says so next to the measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

__all__ = [
    "DEFAULT_PH",
    "SALT_BRIDGE_DISTANCE",
    "ANIONIC_RESIDUES",
    "CATIONIC_RESIDUES",
    "CHARGED_FAMILIES",
    "ChargedGroup",
    "ProtonationReport",
    "require_rdkit",
    "detect_groups",
    "protonation_report",
    "salt_bridge_warnings",
    "charged_copy",
    "correct_formal_charges",
    "FormalChargeCapability",
    "can_represent_formal_charge",
    "net_charge",
]

#: The physiological pH the default state is reported against.  It is *reported*
#: against, not enforced: nothing in this repository titrates a molecule.
DEFAULT_PH = 7.4

#: The nitrogen-to-oxygen distance (Å) within which a charged pair is a salt bridge.
#: 4.0 Å is the usual ceiling for a hydrogen-bonded ion pair (N···O 2.7-3.2 Å in a
#: real one, up to ~4 Å for a water-mediated or slightly strained pair).
SALT_BRIDGE_DISTANCE = 4.0

#: Receptor residues whose side chain is anionic at physiological pH.
ANIONIC_RESIDUES: Tuple[str, ...] = ("ASP", "GLU")
#: Receptor residues whose side chain is cationic at physiological pH.
CATIONIC_RESIDUES: Tuple[str, ...] = ("LYS", "ARG", "HIS")

#: The families this module knows, with the SMARTS that perceives them, the formal
#: charge they carry at :data:`DEFAULT_PH`, and what the uncharged perception means.
#: The patterns are deliberately conservative: a missed group produces no warning,
#: which is the safe failure for a check whose output is a warning.
#:
#: **The expected charges are the ones pH 7.4 actually gives**, which is why imidazole
#: and thiol are here with a charge of 0: their pKa values (≈6.0 and ≈10) put the
#: neutral form in the majority at physiological pH, so flagging a neutral imidazole
#: or a thiol as "wrong" would be a false alarm.  Aromatic amines (aniline, pKa 4.6)
#: are excluded from the amine pattern for the same reason — the pattern requires the
#: nitrogen not to be attached to an aromatic carbon.
CHARGED_FAMILIES: Dict[str, Dict[str, Any]] = {
    "amidinium": {
        # `NX2` alone misses the charged form: an amidinium's imine nitrogen carries
        # two hydrogens ([NH2+]), so its connectivity is 3, and a pattern that cannot
        # see the *correct* state cannot compare a molecule against it.
        "smarts": "[CX3](=[NX2,NX3+])[NX3]",
        "charge": 1,
        "anionic": False,
        "label": "amidinium/guanidinium",
    },
    "carboxylate": {
        "smarts": "[CX3](=O)[OX2H1,OX1-]",
        "charge": -1,
        "anionic": True,
        "label": "carboxylate",
    },
    "phosphate": {
        "smarts": "[PX4](=O)([OX2,OX1-])[OX2,OX1-]",
        "charge": -1,
        "anionic": True,
        "label": "phosphate",
    },
    "sulfonate": {
        "smarts": "[SX4](=O)(=O)[OX2,OX1-]",
        "charge": -1,
        "anionic": True,
        "label": "sulfonate/sulfate",
    },
    "amine": {
        # H3 is included so that an ammonium ([NH3+], connectivity 3) is perceived as
        # the *charged* form of the amine family rather than missed; a quaternary
        # ammonium is NX4 and is deliberately not matched, because it has no neutral
        # form to disagree with.
        "smarts": "[$([NX3;H0,H1,H2;!$(N[#6]=[O,N,S]);!$(N=*);!$(N#*);!$(N[a])]),$([NX4+;H1,H2,H3;!$(N[a])])]",
        "charge": 1,
        "anionic": False,
        "label": "primary/secondary/tertiary amine",
    },
    "imidazole": {
        "smarts": "c1cnc[nH1]1",
        "charge": 0,
        "anionic": False,
        "label": "imidazole (neutral at pH 7.4)",
    },
    "thiolate": {
        "smarts": "[SX1-,SX2H1]",
        "charge": 0,
        "anionic": True,
        "label": "thiol (neutral at pH 7.4)",
    },
}


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for the protonation check. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


@dataclass(frozen=True)
class ChargedGroup:
    """One formally charged group the molecule contains, and what state it is in."""

    family: str
    label: str
    atoms: Tuple[int, ...]
    #: The charge the family carries at the stated pH.
    expected_charge: int
    #: The charge the atoms actually carry in the molecule as prepared.
    observed_charge: int

    @property
    def is_charged(self) -> bool:
        """Whether the molecule is *already* in the charged form of this family."""
        return int(self.observed_charge) == int(self.expected_charge) and (
            int(self.expected_charge) != 0
        )

    @property
    def is_neutral_disagreement(self) -> bool:
        """A family that should be charged at this pH and is not.

        This is the benzamidine case: an amidine that is a cation *and* neutral, which
        is possible — an amidine is a weak base — but is the state that makes every
        electrostatic term point the wrong way when the pocket is anionic.
        """
        return int(self.observed_charge) == 0 and int(self.expected_charge) != 0

    @property
    def is_charged_disagreement(self) -> bool:
        """A family that is charged although the stated pH does not require it."""
        return int(self.observed_charge) != 0 and int(self.observed_charge) != int(
            self.expected_charge
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "label": self.label,
            "atoms": [int(index) for index in self.atoms],
            "expected_charge": int(self.expected_charge),
            "observed_charge": int(self.observed_charge),
            "state": (
                "charged" if self.is_charged
                else "neutral (disagrees with the stated pH)"
                if self.is_neutral_disagreement
                else "charged (disagrees with the stated pH)"
                if self.is_charged_disagreement
                else "neutral"
            ),
        }


def detect_groups(mol) -> List[ChargedGroup]:
    """Every charged group of ``mol``, with its expected and observed charge.

    The **observed** charge is the formal charge on the matched atoms *in this
    molecule*, so a neutral amidine and an amidinium are the same family and different
    states — which is the whole point: the family is chemistry, the state is a
    decision someone made (or failed to make).
    """
    require_rdkit()
    if mol is None:
        return []
    groups: List[ChargedGroup] = []
    for family, spec in CHARGED_FAMILIES.items():
        query = Chem.MolFromSmarts(spec["smarts"])
        if query is None:  # pragma: no cover - a broken pattern would be a bug
            continue
        seen: set = set()
        seen_centres: set = set()
        for match in mol.GetSubstructMatches(query):
            key = tuple(sorted(int(index) for index in match))
            if key in seen:
                continue
            # One group per central atom: guanidinium has three N-C-N matches and a
            # phosphate three P=O matches, and counting them separately would report a
            # +2 guanidinium and a -3 phosphate that no molecule has.
            centre = int(match[0])
            if centre in seen_centres:
                continue
            seen.add(key)
            seen_centres.add(centre)
            observed = sum(int(mol.GetAtomWithIdx(index).GetFormalCharge()) for index in key)
            groups.append(
                ChargedGroup(
                    family=family,
                    label=str(spec["label"]),
                    atoms=key,
                    expected_charge=int(spec["charge"]),
                    observed_charge=int(observed),
                )
            )
    groups.sort(key=lambda group: (group.family, group.atoms))
    return groups


@dataclass
class ProtonationReport:
    """What protonation state the molecule is in, and where that contradicts the pH."""

    name: str = ""
    ph: float = DEFAULT_PH
    groups: List[ChargedGroup] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def net_charge(self) -> int:
        return int(sum(group.observed_charge for group in self.groups))

    @property
    def expected_net_charge(self) -> int:
        return int(sum(group.expected_charge for group in self.groups))

    @property
    def neutral_disagreements(self) -> List[ChargedGroup]:
        return [group for group in self.groups if group.is_neutral_disagreement]

    @property
    def is_ambiguous(self) -> bool:
        """Whether any family's state disagrees with the stated pH."""
        return bool(self.neutral_disagreements) or any(
            group.is_charged_disagreement for group in self.groups
        )

    def summary(self) -> str:
        """One line: the state, the net charge, and whether it is ambiguous."""
        if not self.groups:
            return f"{self.name}: no formally charged group at pH {self.ph}"
        families = ", ".join(
            f"{group.label} {'charged' if group.is_charged else 'neutral'}"
            for group in self.groups
        )
        return (
            f"{self.name}: net charge {self.net_charge:+d} "
            f"(at pH {self.ph} the families would be {self.expected_net_charge:+d}) — "
            f"{families}"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ph": float(self.ph),
            "net_charge": self.net_charge,
            "expected_net_charge": self.expected_net_charge,
            "ambiguous": bool(self.is_ambiguous),
            "groups": [group.as_dict() for group in self.groups],
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }


def protonation_report(
    mol, *, name: str = "", ph: float = DEFAULT_PH, charges: Optional[Sequence[float]] = None
) -> ProtonationReport:
    """The protonation state of ``mol`` at ``ph``, with the disagreements named.

    ``charges`` (one per atom) is optional and only used for the note that quotes the
    partial charge the model put on a group that is neutral — the benzamidine
    finding, where the amidine nitrogen carries −0.30 e.
    """
    require_rdkit()
    label = str(name) or _mol_name(mol)
    groups = detect_groups(mol)
    report = ProtonationReport(name=label, ph=float(ph), groups=groups)
    if not groups:
        report.notes.append(
            "no amidinium/guanidinium, carboxylate, phosphate, sulfonate, amine, "
            "imidazole or thiol group was perceived: the molecule carries no group "
            "whose protonation state a pH would decide"
        )
        return report
    for group in report.neutral_disagreements:
        detail = ""
        if charges is not None:
            values = np.asarray(charges, dtype=float).reshape(-1)
            if values.shape[0] > max(group.atoms):
                detail = (
                    " (the charge model puts "
                    + ", ".join(
                        f"{values[index]:+.2f} e on atom {index}" for index in group.atoms
                    )
                    + ")"
                )
        report.warnings.append(
            f"{group.label} is NEUTRAL although pH {float(ph):.1f} makes it "
            f"{group.expected_charge:+d}{detail}: every electrostatic term computed "
            "from this molecule is computed for the wrong species, and if the pocket "
            "is lined by an oppositely charged residue the sign of the interaction is "
            "wrong rather than merely inaccurate.  See docs/PROTONATION.md"
        )
    report.notes.append(
        "the state is reported, not chosen: this repository does not titrate, so the "
        "charge model is applied to whatever protonation state the input SMILES or "
        "structure carried"
    )
    return report


def _mol_name(mol, fallback: str = "ligand") -> str:
    try:
        if mol is not None and mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


# ---------------------------------------------------------------------------
# The geometric contradiction: a neutral group inside a charged pocket
# ---------------------------------------------------------------------------


def _residue_atoms(atoms: Sequence[Any], names: Iterable[str]) -> Dict[str, List[Any]]:
    wanted = {str(name).upper() for name in names}
    out: Dict[str, List[Any]] = {}
    for atom in atoms:
        residue = _residue_of(atom)
        if residue[0] in wanted:
            out.setdefault(residue[0], []).append(atom)
    return out


def _residue_of(atom) -> Tuple[str, str, str]:
    """``(residue name, chain, number)`` of an atom-like object, best effort."""
    for attribute in ("residue_name", "resname", "residue"):
        value = getattr(atom, attribute, None)
        if isinstance(value, str) and value.strip():
            chain = str(getattr(atom, "chain", "") or "")
            number = str(getattr(atom, "residue_number", "") or getattr(atom, "resid", "") or "")
            return (value.strip().upper(), chain, number)
    return ("", "", "")


def _coords_of(atoms: Sequence[Any]) -> np.ndarray:
    out = []
    for atom in atoms:
        if hasattr(atom, "coords"):
            out.append(list(atom.coords))
        else:
            out.append([float(atom.x), float(atom.y), float(atom.z)])
    return np.asarray(out, dtype=float).reshape(-1, 3)


def salt_bridge_warnings(
    mol,
    receptor_atoms: Sequence[Any],
    *,
    ligand_coords: Optional[np.ndarray] = None,
    charges: Optional[Sequence[float]] = None,
    distance: float = SALT_BRIDGE_DISTANCE,
    ph: float = DEFAULT_PH,
    name: str = "",
) -> List[str]:
    """Warnings for the charged-group/pocket contradictions, as checkable geometry.

    Two conditions, both of them a family plus a distance:

    * a **cationic** family (amidinium, amine, imidazole) that is *neutral* in the
      molecule as prepared, within ``distance`` Å of an anionic residue (ASP, GLU) —
      the benzamidine/Asp189 case, where the neutral form makes the interaction
      repulsive instead of attractive;
    * an **anionic** family (carboxylate, phosphate, sulfonate, thiolate) that is
      *neutral*, within ``distance`` Å of a cationic residue (LYS, ARG, HIS).

    The ligand atoms are ``ligand_coords`` when given (a pose, an ensemble member) and
    the molecule's own conformer otherwise.  A molecule with no 3-D coordinates or a
    receptor whose atoms carry no residue names returns no warnings rather than
    guessing.
    """
    require_rdkit()
    label = str(name) or _mol_name(mol)
    groups = detect_groups(mol)
    if not groups or not receptor_atoms:
        return []
    try:
        if ligand_coords is not None:
            coords = np.asarray(ligand_coords, dtype=float).reshape(-1, 3)
        else:
            conformer = mol.GetConformer()
            coords = np.asarray(
                [
                    [conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y,
                     conformer.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())
                ],
                dtype=float,
            )
    except Exception:  # pragma: no cover - a molecule without a conformer
        return []
    # The group indices index the *molecule*, so coordinates in another atom order
    # would silently measure the wrong distance.  Measured: a mismatched order turned
    # the 2.87 A amidine-Asp189 salt bridge into 4.96 A and the warning never fired.
    if coords.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"ligand_coords has {coords.shape[0]} atom(s) and the molecule has "
            f"{mol.GetNumAtoms()}: the charged-group indices index the molecule, so the "
            "coordinates must be given in its own atom order"
        )
    anionic = _residue_atoms(receptor_atoms, ANIONIC_RESIDUES)
    cationic = _residue_atoms(receptor_atoms, CATIONIC_RESIDUES)
    warnings: List[str] = []
    for group in groups:
        if group.is_charged or group.observed_charge != 0:
            continue
        if group.expected_charge > 0:
            partners, kind = anionic, "anionic"
        elif group.expected_charge < 0:
            partners, kind = cationic, "cationic"
        else:  # pragma: no cover - no family has zero expected charge
            continue
        for residue, atoms in partners.items():
            positions = _coords_of(atoms)
            if positions.size == 0 or coords.shape[0] <= max(group.atoms):
                continue
            centres = coords[list(group.atoms)]
            deltas = centres[:, None, :] - positions[None, :, :]
            nearest = float(np.sqrt((deltas ** 2).sum(axis=2)).min())
            if nearest <= float(distance):
                detail = ""
                if charges is not None:
                    values = np.asarray(charges, dtype=float).reshape(-1)
                    if values.shape[0] > max(group.atoms):
                        detail = (
                            " (the model gives it "
                            + ", ".join(f"{values[i]:+.2f} e" for i in group.atoms)
                            + ")"
                        )
                warnings.append(
                    f"{label}: {group.label} is NEUTRAL{detail} but sits "
                    f"{nearest:.2f} A from {residue}, which is {kind} at pH "
                    f"{float(ph):.1f} — a salt bridge needs the charged form, so every "
                    "electrostatic term computed for this complex has the wrong sign.  "
                    "Re-prepare from the charged SMILES or set the formal charge; see "
                    "docs/PROTONATION.md"
                )
                break
    return warnings


# ---------------------------------------------------------------------------
# The other state, so the cost of the assumption can be measured
# ---------------------------------------------------------------------------


def charged_copy(mol, family: str = "amidinium", *, name: Optional[str] = None):
    """The explicitly charged form of ``mol`` for one family.

    The hydrogen is added to (or removed from) the perceived group and the formal
    charge set, so the result is a real molecule a charge model can be run on rather
    than a hand-edited charge array.  For an amidine the proton goes on the imine
    nitrogen — the amidinium — which is the species that binds trypsin.

    Raises ``ValueError`` when the molecule has no group of that family, because
    silently returning the input would make the measurement this exists for a no-op.
    """
    require_rdkit()
    groups = [group for group in detect_groups(mol) if group.family == family]
    if not groups:
        raise ValueError(f"no {family} group in this molecule")
    target = groups[0]
    expected = int(CHARGED_FAMILIES[family]["charge"])
    work = Chem.RWMol(mol)
    if expected > 0:
        # Protonate: prefer the nitrogen (or the heteroatom) that carries no hydrogen.
        candidates = [
            index
            for index in target.atoms
            if work.GetAtomWithIdx(index).GetAtomicNum() in (7, 8, 16)
        ]
        if not candidates:  # pragma: no cover - a family without a heteroatom
            raise ValueError(f"cannot protonate {family}")
        index = min(
            candidates, key=lambda i: (work.GetAtomWithIdx(i).GetTotalNumHs(), i)
        )
        atom = work.GetAtomWithIdx(index)
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        atom.SetNoImplicit(True)
    else:
        index = max(
            target.atoms,
            key=lambda i: work.GetAtomWithIdx(i).GetTotalNumHs(),
        )
        atom = work.GetAtomWithIdx(index)
        if atom.GetTotalNumHs() == 0:  # pragma: no cover - already deprotonated
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        else:
            atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom.SetNoImplicit(True)
    out = work.GetMol()
    Chem.SanitizeMol(out)
    if name:
        out.SetProp("_Name", str(name))
    return out


def net_charge(mol) -> int:
    """The molecule's formal net charge, as the sum over its atoms."""
    require_rdkit()
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


@dataclass
class FormalChargeCapability:
    """Whether a charge set can represent the formal charges a molecule declares.

    The entry point for every consumer that has to say how much its electrostatics can
    be trusted.  A charge model that is a sigma-electronegativity model (Gasteiger,
    MMFF94) distributes a formal charge over a whole ion and can leave a *cationic*
    nitrogen negative; a set that cannot do that cannot be used to argue about a salt
    bridge, whatever the protonation state says.
    """

    #: True when every formally charged group's charges have the right sign and a
    #: magnitude worth calling a charge.  A molecule with no charged group is True:
    #: there is nothing to represent.
    possible: bool = True
    #: One line per group that fails, naming the group and what it sums to.
    reasons: List[str] = field(default_factory=list)
    #: ``(family, formal charge, the charge set's sum over the group)`` per group.
    groups: List[Tuple[str, int, float]] = field(default_factory=list)
    #: How large a group's total charge has to be to count as representing it.
    threshold: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "possible": bool(self.possible),
            "threshold": round(float(self.threshold), 4),
            "groups": [
                {"family": family, "formal_charge": int(charge), "assigned": round(float(total), 4)}
                for family, charge, total in self.groups
            ],
            "reasons": list(self.reasons),
        }

    def statement(self) -> str:
        """One line a report can quote."""
        if not self.groups:
            return "no formally charged group, so any charge set can represent it"
        if self.possible:
            return (
                f"the charge set represents every formal charge "
                f"({len(self.groups)} group(s), all within {self.threshold:g} e of their "
                "formal value)"
            )
        return "; ".join(self.reasons)


def can_represent_formal_charge(
    mol,
    charges: Optional[Sequence[float]] = None,
    *,
    threshold: float = 0.5,
    ph: float = DEFAULT_PH,
) -> FormalChargeCapability:
    """Whether ``charges`` can represent ``mol``'s formal charges.

    The question is asked against the charge the group **would carry at pH ``ph``**, so
    the answer means one thing for every consumer: *can this charge set hold the formal
    charge this chemistry implies?*  A group passes when the sum of its atoms' charges
    has the **same sign** as that formal charge and at least ``threshold`` of its
    magnitude.  Measured on the 3PTB ligand, every charge model this repository applies
    to a benzamidine fails: the neutral file charges give the amidine **−0.62 e** where
    the formal charge is **+1**, and Gasteiger on the *correct* amidinium still gives
    **−0.11 e**.  Only a set corrected to hold the formal charge passes — which is the
    whole reason `docs/PROTONATION.md` exists.

    With ``charges=None`` the molecule's own per-atom formal charges are used, so the
    predicate also answers "does this molecule itself carry the charges its pH implies"
    — the protonation question, from the same call.
    """
    require_rdkit()
    groups = detect_groups(mol)
    capability = FormalChargeCapability(threshold=float(threshold))
    if not groups:
        return capability
    if charges is None:
        values = np.asarray(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=float
        )
    else:
        values = np.asarray(charges, dtype=float).reshape(-1)
        # A charge set that does not describe *this* molecule cannot be checked: the
        # group indices index the molecule, so a shorter or longer array would be a
        # different molecule's charges.  It is rejected rather than mis-indexed.
        if values.shape[0] != mol.GetNumAtoms():
            capability.possible = False
            capability.reasons.append(
                f"the charge set has {values.shape[0]} entries for "
                f"{mol.GetNumAtoms()} atom(s), so it cannot be checked against this "
                "molecule (docs/PROTONATION.md)"
            )
            return capability
    for group in groups:
        if group.expected_charge == 0:
            continue
        total = float(values[list(group.atoms)].sum())
        capability.groups.append((group.family, group.expected_charge, total))
        if total == 0.0 or (total > 0) != (group.expected_charge > 0) or (
            abs(total) < float(threshold)
        ):
            capability.possible = False
            capability.reasons.append(
                f"{group.label} would carry {group.expected_charge:+d} at pH "
                f"{float(ph):.1f} but its charges sum to {total:+.2f} e: this charge set "
                "cannot represent that formal charge, so any electrostatic term computed "
                "from it is computed for a different species (docs/PROTONATION.md)"
            )
    return capability


def correct_formal_charges(
    mol, charges: Sequence[float], *, ph: float = DEFAULT_PH
) -> Tuple[np.ndarray, List[str]]:
    """Shift each charged group's partial charges so the group sums to its formal charge.

    **Measured, and the reason this function exists**: protonating the molecule is not
    enough.  Gasteiger is a sigma-electronegativity model, so an *amidinium* still
    comes out with **−0.11 e** on its nitrogens — the formal +1 is spread over the
    whole ion, and the nitrogen stays electronegative.  On 3PTB's crystal pose the
    interaction energy goes from +22.09 kcal/mol (neutral, as prepared) to +20.98
    (Gasteiger on the amidinium) and only to **−28.75** when the group's formal charge
    is restored.  So: no protonation-state change without a charge model that can hold
    a formal charge.

    The correction is deliberately minimal — the group's partial charges are shifted by
    a constant so they sum to the formal charge, leaving their *distribution* (and
    every other atom) untouched.  It is not a re-parameterisation and it does not
    re-run a charge model; it makes the formal charge the model already declares
    visible to a Coulomb sum.  Returns ``(charges, descriptions of what was shifted)``.
    """
    values = np.asarray(charges, dtype=float).reshape(-1).copy()
    changed: List[str] = []
    for group in detect_groups(mol):
        if group.expected_charge == 0:
            continue
        if values.shape[0] <= max(group.atoms):
            continue
        current = float(values[list(group.atoms)].sum())
        target = float(group.expected_charge)
        if abs(current - target) < 1e-9:
            continue
        shift = (target - current) / len(group.atoms)
        for index in group.atoms:
            values[index] += shift
        changed.append(
            f"{group.label} atoms {list(group.atoms)}: {current:+.2f} -> {target:+.2f} e "
            f"(each shifted {shift:+.2f} e)"
        )
    return values, changed

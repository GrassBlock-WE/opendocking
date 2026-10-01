# SPDX-License-Identifier: GPL-3.0-or-later
"""Hit triage: what is *wrong* with these molecules, and where.

A ranked hit list answers "what binds". Before anything is ordered, a project also
needs the liability view: which of these molecules carries a structural alert,
which one breaks a drug-likeness rule, and **where** the alert sits — on the shared
scaffold or on a decoration, because that is the difference between "this series
is not a series" and "make an analogue without the acrylate".

What this module adds on top of :mod:`odock.filters`
----------------------------------------------------
``filters.py`` owns the *verdicts* the screening pipeline already uses — Lipinski,
Veber and the PAINS catalogue — and they are the single source of truth for them;
this module **imports** them rather than re-deriving them.  On top of that it adds

* the published **alert catalogues** RDKit ships (PAINS A/B/C, Brenk, NIH, ZINC),
  each reporting the entry that matched and the atoms it matched;
* a documented **SMARTS liability set** for the classes a catalogue misses
  (Michael acceptors, epoxides, aziridines, nitroaromatics, anilines, hydrazines,
  acyl halides, charged/quaternary centres, aldehydes, thiols, catechols) plus a
  few metabolic **soft spots**;
* **where** every alert sits: ``scaffold``, ``substituent``, ``mixed``;
* an extended property panel (TPSA, aromatic rings, fraction sp³, QED, and the
  Egan and Ghose rules beside Lipinski and Veber);
* :func:`triage_library`, which folds all of it into one table **grouped by the
  scaffold series** :mod:`odock.scaffold` already computes, so the reader sees
  "this series is clean, that one carries a Michael acceptor".

These are **alerts, not predictions**
-------------------------------------
An alert is a literature-derived substructure pattern.  Matching one is not a
measurement, and the false-positive rate of an alert set is not a probability of
failure: it was estimated on some other set of molecules, at some other time.
Absence of an alert is **not** safety — it means nothing matched, which is a
statement about the pattern list and not about the molecule.  A molecule this
module flags can be a perfectly good drug, and a molecule it passes can fail in
the clinic.  Nothing here substitutes for an assay, and a catalogue that flags a
large fraction of a small set is telling you about the catalogue, not about the
set: :func:`alert_rates` and :class:`TriageReport` report those fractions so the
reader can see them, and say so in their notes.
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
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Crippen = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    QED = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

try:
    from rdkit.Chem import FilterCatalog as _FilterCatalog

    _HAVE_CATALOGUES = True
except Exception:  # pragma: no cover - trimmed RDKit build
    _FilterCatalog = None  # type: ignore[assignment]
    _HAVE_CATALOGUES = False

from .filters import drug_like as _drug_like

__all__ = [
    "CATALOGUES",
    "LIABILITY_SETS",
    "SMARTS_ALERTS",
    "ALERT_LOCATIONS",
    "EGAN_TPSA",
    "EGAN_LOGP",
    "GHOSE_BOUNDS",
    "HAVE_RDKIT",
    "HAVE_CATALOGUES",
    "HAVE_SA_SCORE",
    "AlertHit",
    "PropertyPanel",
    "MoleculeTriage",
    "TriageReport",
    "require_rdkit",
    "catalogue_alerts",
    "smarts_alerts",
    "alert_location",
    "panel",
    "triage_molecule",
    "triage_library",
    "alert_rates",
    "sa_score",
]

#: The alert catalogues RDKit ships, in report order.
CATALOGUES: Tuple[str, ...] = ("PAINS", "PAINS_A", "PAINS_B", "PAINS_C", "BRENK", "NIH", "ZINC")

#: The two SMARTS sets: genuine liabilities and metabolic soft spots.  They are
#: reported separately because "reactive" and "gets metabolised quickly" are
#: different decisions.
LIABILITY_SETS: Tuple[str, ...] = ("liability", "soft_spot")

#: Where an alert sits relative to the molecule's Murcko scaffold.
ALERT_LOCATIONS: Tuple[str, ...] = ("scaffold", "substituent", "mixed", "none")

#: Egan's "drug-like" ellipsoid cut-offs (TPSA and Crippen LogP).
EGAN_TPSA = 131.6
EGAN_LOGP = 5.88

#: Ghose's qualifying ranges: MW, LogP, heavy atoms, molar refractivity.
GHOSE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "MW": (160.0, 480.0),
    "LogP": (-0.4, 5.6),
    "heavy_atoms": (20.0, 70.0),
    "MolMR": (40.0, 130.0),
}

#: A documented SMARTS alert set for the liability classes a catalogue tends to
#: miss, plus a few metabolic soft spots.  Every entry names the pattern's intent,
#: so a report can explain *why* a molecule was flagged.
SMARTS_ALERTS: Dict[str, Dict[str, str]] = {
    "liability": {
        # -- electrophiles ---------------------------------------------------
        "michael_acceptor_enone": "[CX3]=[CX3][CX3]=[OX1]",
        "michael_acceptor_acrylonitrile": "[CX3]=[CX3][CX2]#[NX1]",
        "michael_acceptor_vinyl_sulfone": "[CX3]=[CX3][SX4](=O)=O",
        "michael_acceptor_nitroalkene": "[CX3]=[CX3][NX3+](=O)[O-]",
        "epoxide": "[OX2;r3]1[#6;r3][#6;r3]1",
        "aziridine": "[NX3;r3]1[#6;r3][#6;r3]1",
        "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
        "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
        "alkyl_halide": "[CX4][Cl,Br,I]",
        "anhydride": "[CX3](=O)[OX2][CX3]=O",
        "isocyanate": "[NX2]=[CX2]=[OX1]",
        "aldehyde": "[CX3H1](=O)[#6]",
        # -- redox-active and reactive aromatics -----------------------------
        "nitroaromatic": "[NX3](=[OX1])([OX1])[c]",
        "aniline": "[NX3;H2,H1;!$(NC=O)][c]",
        "catechol": "c1cc(O)c(O)cc1",
        "quinone": "O=C1C=CC(=O)C=C1",
        "azo": "[#6][NX2]=[NX2][#6]",
        "hydrazine": "[NX3][NX3]",
        "hydrazone": "[NX3][NX2]=[#6]",
        "hydroxamic_acid": "[CX3](=O)[NX3][OX2H]",
        # -- groups that are reactive for a different reason -----------------
        "thiol": "[SX2H]",
        "dithiocarbamate": "[NX3][CX3](=S)[SX2]",
        "n_oxide": "[#7+][OX1-]",
        # Permanent charges, excluding the charges of a nitro group (its own
        # alert) and of an N-oxide (also its own): a duplicate flag there is noise
        # rather than a second liability.
        "charged_centre": (
            "[$([NX4+;!$([NX4+][OX1-])]),$([NX3+;!$([NX3+]=[OX1])]),$([SX3+,PX4+]),"
            "$([CX3-,S-,NX2-]),$([O-;!$([O-][#7+])])]"
        ),
        "quaternary_nitrogen": "[NX4+]",
    },
    "soft_spot": {
        "para_methoxyphenyl": "COc1ccc([#6])cc1",
        "benzylic_ether": "[c][CH2][OX2]",
        "n_methyl_amide": "[NX3](C)C=O",
        "ester": "[CX3](=O)[OX2][#6]",
        "tertiary_amine": "[NX3]([#6])[#6]",
    },
}

#: Whether this RDKit build ships the alert catalogues.
HAVE_CATALOGUES = _HAVE_CATALOGUES

#: Whether the RDKit contrib synthetic-accessibility scorer is importable.  It is
#: an optional part of the RDKit source distribution; when it is missing the panel
#: reports ``None`` for the SA score and a note, rather than inventing a proxy.
HAVE_SA_SCORE = False
_sa_scorer = None
try:  # pragma: no cover - depends on the installed RDKit
    import sys as _sys

    from rdkit import RDConfig as _RDConfig

    _contrib = str(getattr(_RDConfig, "RDContribDir", ""))
    if _contrib and _contrib not in _sys.path:
        _sys.path.append(_contrib)
    import sascorer as _sascorer  # type: ignore

    _sa_scorer = _sascorer
    HAVE_SA_SCORE = True
except Exception:  # pragma: no cover - most wheels ship without it
    _sa_scorer = None
    HAVE_SA_SCORE = False

#: Whether this build has the RDKit pieces the module needs.
HAVE_RDKIT = _HAVE_RDKIT


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for hit triage. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


@dataclass
class AlertHit:
    """One alert that matched, with where it matched."""

    #: ``"catalogue"`` (a published RDKit set) or ``"smarts"`` (the set above).
    source: str
    #: The catalogue name (``"PAINS"``) or the SMARTS set (``"liability"``).
    origin: str
    #: The catalogue entry description or the SMARTS alert name.
    name: str
    #: Atom indices the pattern matched.
    atoms: Tuple[int, ...] = ()
    #: ``scaffold``, ``substituent``, ``mixed`` or ``none`` (no atoms reported).
    location: str = "none"
    #: One sentence on what the alert is about.
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "origin": self.origin,
            "name": self.name,
            "atoms": [int(i) for i in self.atoms],
            "location": self.location,
            "note": self.note,
        }

    def label(self) -> str:
        """``"PAINS:quinone_A(370) [scaffold]"`` for a one-line report cell."""
        return f"{self.origin}:{self.name} [{self.location}]"


def _scaffold_atoms(mol) -> Optional[set]:
    """The heavy-atom indices of the Murcko scaffold, or ``None`` when acyclic.

    Used to answer "is this alert on the core or on a decoration".  The scaffold is
    matched back onto the molecule, so the returned indices are the molecule's own.
    """
    from .scaffold import scaffold_of

    frame = scaffold_of(mol)
    if frame is None:
        return None
    match = mol.GetSubstructMatch(frame)
    if not match:
        return None
    return set(int(index) for index in match)


def alert_location(mol, atoms: Sequence[int], *, scaffold_atoms: Optional[set] = None) -> str:
    """Where an alert sits: ``scaffold``, ``substituent``, ``mixed`` or ``none``.

    ``scaffold`` means every matched atom is part of the Bemis-Murcko framework —
    the alert cannot be removed without changing the chemotype.  ``substituent``
    means none of them is: an analogue without the alert is a decoration away.
    ``mixed`` spans both, which is the case that needs a chemist's eye (a
    cinnamamide's enone includes the ring carbon).  A molecule with no ring has no
    scaffold, so its alerts are reported as ``substituent`` — everything is a
    decoration when there is no core.
    """
    indices = {int(index) for index in atoms}
    if not indices:
        return "none"
    if scaffold_atoms is None:
        scaffold_atoms = _scaffold_atoms(mol)
    if not scaffold_atoms:
        return "substituent"
    inside = indices & scaffold_atoms
    if not inside:
        return "substituent"
    if inside == indices:
        return "scaffold"
    return "mixed"


_CATALOGUE_CACHE: Dict[str, Any] = {}


def _catalogue(name: str):
    """The shared RDKit catalogue object for ``name``, or ``None``."""
    if not _HAVE_CATALOGUES:
        return None
    key = str(name).strip().upper()
    if key in _CATALOGUE_CACHE:
        return _CATALOGUE_CACHE[key]
    catalog = None
    try:
        params = _FilterCatalog.FilterCatalogParams()
        params.AddCatalog(getattr(_FilterCatalog.FilterCatalogParams.FilterCatalogs, key))
        catalog = _FilterCatalog.FilterCatalog(params)
    except Exception:  # pragma: no cover - unknown catalogue name
        catalog = None
    _CATALOGUE_CACHE[key] = catalog
    return catalog


def catalogue_alerts(
    mol,
    *,
    catalogues: Sequence[str] = ("PAINS", "BRENK", "NIH", "ZINC"),
) -> List[AlertHit]:
    """Every entry of the named published catalogues that ``mol`` matches.

    The catalogues are RDKit's: PAINS (480 entries), PAINS_A/B/C, Brenk (105), NIH
    (180) and ZINC (50).  Each hit reports the catalogue, the entry description and
    the matched atoms, so a report can say *which* alert fired and where.

    An unknown catalogue name or a build without ``FilterCatalog`` yields an empty
    list rather than an exception: a triage run must not die because one optional
    catalogue is missing.  :data:`HAVE_CATALOGUES` says whether any are available.
    """
    require_rdkit()
    hits: List[AlertHit] = []
    if not _HAVE_CATALOGUES:
        return hits
    scaffold_atoms = _scaffold_atoms(mol)
    for name in catalogues:
        catalog = _catalogue(name)
        if catalog is None:
            continue
        try:
            matches = catalog.GetMatches(mol)
        except Exception:  # pragma: no cover - an unsanitisable molecule
            continue
        seen = set()
        for entry in matches:
            description = str(entry.GetDescription())
            if (str(name).upper(), description) in seen:
                continue
            seen.add((str(name).upper(), description))
            atoms: Tuple[int, ...] = ()
            try:
                # `GetFilterMatches` returns `FilterMatch` objects whose
                # `atomPairs` are (pattern atom, molecule atom) pairs, so the
                # molecule's own indices are the second element of each pair.
                for match in entry.GetFilterMatches(mol):
                    pairs = getattr(match, "atomPairs", None)
                    if pairs:
                        atoms = tuple(int(pair[1]) for pair in pairs)
                        break
            except Exception:  # pragma: no cover - some entries expose no atoms
                atoms = ()
            hits.append(
                AlertHit(
                    source="catalogue",
                    origin=str(name).upper(),
                    name=description,
                    atoms=atoms,
                    location=alert_location(mol, atoms, scaffold_atoms=scaffold_atoms),
                    note="published alert catalogue entry",
                )
            )
    return hits


def smarts_alerts(
    mol,
    *,
    sets: Sequence[str] = LIABILITY_SETS,
    alerts: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[AlertHit]:
    """The embedded SMARTS liabilities and soft spots that ``mol`` matches.

    ``sets`` selects ``"liability"`` and/or ``"soft_spot"``.  Every hit carries the
    alert name, the matched atoms and the location, so "the Michael acceptor is on
    the substituent, the nitroaromatic is the scaffold" is one line of output.

    Hand-checkable examples: nitrobenzene matches ``nitroaromatic`` on the three
    atoms of the nitro group plus the ring carbon it is attached to, and an
    acrylamide matches ``michael_acceptor_enone`` (`tests/test_triage.py` pins
    both).
    """
    require_rdkit()
    table = alerts if alerts is not None else SMARTS_ALERTS
    hits: List[AlertHit] = []
    scaffold_atoms = _scaffold_atoms(mol)
    for family in sets:
        patterns = table.get(family)
        if not patterns:
            continue
        for name, smarts in patterns.items():
            query = Chem.MolFromSmarts(smarts)
            if query is None:  # pragma: no cover - a broken pattern in the table
                continue
            for match in mol.GetSubstructMatches(query):
                atoms = tuple(int(index) for index in match)
                hits.append(
                    AlertHit(
                        source="smarts",
                        origin=family,
                        name=name,
                        atoms=atoms,
                        location=alert_location(mol, atoms, scaffold_atoms=scaffold_atoms),
                        note="embedded SMARTS alert",
                    )
                )
    return hits


# ---------------------------------------------------------------------------
# Properties and rules
# ---------------------------------------------------------------------------


@dataclass
class PropertyPanel:
    """The descriptors that matter, and the rules they are judged against."""

    #: ``{name: value}``; ``None`` where the build cannot compute one (SA score).
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    #: ``{rule: (passed, [violations])}``.
    rules: Dict[str, Tuple[bool, List[str]]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(passed for passed, _ in self.rules.values())

    def failing_rules(self) -> List[str]:
        return [name for name, (passed, _) in self.rules.items() if not passed]

    def violations(self) -> List[str]:
        return [
            f"{rule}: {violation}"
            for rule, (_, breaches) in self.rules.items()
            for violation in breaches
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "values": {k: (None if v is None else round(float(v), 4)) for k, v in self.values.items()},
            "rules": {
                rule: {"passed": bool(passed), "violations": list(breaches)}
                for rule, (passed, breaches) in self.rules.items()
            },
            "passed": bool(self.passed),
        }


def sa_score(mol) -> Optional[float]:
    """The RDKit contrib synthetic-accessibility score (1 = easy, 10 = hard), or ``None``.

    The scorer lives in RDKit's ``Contrib`` directory, which most binary wheels do
    **not** ship; this build does not have it.  Rather than inventing a proxy the
    panel reports ``None`` and :data:`HAVE_SA_SCORE` is false, so a report can say
    "not available here" instead of quoting a number of unknown provenance.
    """
    require_rdkit()
    if _sa_scorer is None:
        return None
    try:
        return float(_sa_scorer.calculateScore(mol))
    except Exception:  # pragma: no cover - defensive
        return None


def panel(mol) -> PropertyPanel:
    """The descriptor panel and the four drug-likeness rules.

    Lipinski and Veber come from :mod:`odock.filters` (the layer that already owns
    them, and the one the screening pipeline uses), so a triage table and a screen
    cannot disagree.  The panel adds TPSA, aromatic rings, fraction sp³, QED, the
    SA score when the build has it, and the **Egan** and **Ghose** rules:

    ===========  ==========================================================
    Egan         TPSA <= 131.6 Å² **and** LogP <= 5.88 (the "drug-like"
                 ellipsoid; a molecule outside it is not necessarily bad)
    Ghose        MW 160-480, LogP −0.4-5.6, 20-70 heavy atoms, MolMR 40-130
    ===========  ==========================================================
    """
    require_rdkit()
    verdict = _drug_like(mol)
    props = verdict["properties"]
    values: Dict[str, Optional[float]] = {
        "MW": float(props.get("MW", float("nan"))),
        "LogP": float(props.get("LogP", float("nan"))),
        "TPSA": float(props.get("tPSA", float("nan"))),
        "HBD": float(props.get("HBD", 0)),
        "HBA": float(props.get("HBA", 0)),
        "RotB": float(props.get("RotB", 0)),
        "aromatic_rings": float(Descriptors.NumAromaticRings(mol)),
        "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "MolMR": float(Descriptors.MolMR(mol)),
        "QED": float(QED.qed(mol)),
        "SA_score": sa_score(mol),
    }
    rules: Dict[str, Tuple[bool, List[str]]] = {}
    for result in verdict["results"]:
        rules[result.name] = (bool(result.passed), list(result.violations))
    egan_breaches: List[str] = []
    if values["TPSA"] > EGAN_TPSA:
        egan_breaches.append(f"TPSA {values['TPSA']:.1f} > {EGAN_TPSA}")
    if values["LogP"] > EGAN_LOGP:
        egan_breaches.append(f"LogP {values['LogP']:.2f} > {EGAN_LOGP}")
    rules["Egan"] = (not egan_breaches, egan_breaches)
    ghose_breaches: List[str] = []
    for key, (low, high) in GHOSE_BOUNDS.items():
        value = values.get(key)
        if value is None or not math.isfinite(float(value)):
            continue
        if not (low <= float(value) <= high):
            ghose_breaches.append(f"{key} {float(value):.1f} outside {low:g}-{high:g}")
    rules["Ghose"] = (not ghose_breaches, ghose_breaches)
    return PropertyPanel(values=values, rules=rules)


# ---------------------------------------------------------------------------
# Per-molecule and library triage
# ---------------------------------------------------------------------------


@dataclass
class MoleculeTriage:
    """One molecule's alerts and panel."""

    index: int
    name: str
    smiles: str = ""
    alerts: List[AlertHit] = field(default_factory=list)
    properties: Optional[PropertyPanel] = None
    scaffold: str = ""
    affinity: Optional[float] = None

    @property
    def clean(self) -> bool:
        """No alert at all from any source.

        A clean molecule is still reported (with an empty alert cell) — a triage
        table that silently dropped the clean molecules would hide the fact that
        most of a real library has no alert.
        """
        return not self.alerts

    @property
    def failed_rules(self) -> List[str]:
        return self.properties.failing_rules() if self.properties else []

    def alert_names(self) -> List[str]:
        return [alert.name for alert in self.alerts]

    def summary(self) -> str:
        if self.clean:
            return "clean"
        return "; ".join(alert.label() for alert in self.alerts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "name": self.name,
            "smiles": self.smiles,
            "clean": bool(self.clean),
            "scaffold": self.scaffold,
            "affinity": None if self.affinity is None else round(float(self.affinity), 4),
            "alerts": [alert.as_dict() for alert in self.alerts],
            "failed_rules": self.failed_rules,
            "properties": self.properties.as_dict() if self.properties else None,
        }


@dataclass
class TriageReport:
    """A whole hit list, grouped by the scaffold series it falls into."""

    molecules: List[MoleculeTriage] = field(default_factory=list)
    #: ``{scaffold: [indices]}`` in the order :func:`odock.scaffold.scaffold_groups`
    #: returns them (largest series first).
    groups: List[Tuple[str, List[int]]] = field(default_factory=list)
    #: ``{catalogue: {"flagged": n, "fraction": f, "n": N}}``.
    rates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    catalogues: Tuple[str, ...] = ()
    seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def n_molecules(self) -> int:
        return len(self.molecules)

    @property
    def n_flagged(self) -> int:
        return sum(1 for molecule in self.molecules if not molecule.clean)

    @property
    def n_clean(self) -> int:
        return self.n_molecules - self.n_flagged

    def by_name(self, name: str) -> Optional[MoleculeTriage]:
        for molecule in self.molecules:
            if molecule.name == name:
                return molecule
        return None

    def series_summary(self) -> List[Dict[str, Any]]:
        """Per series: size, how many molecules carry an alert, which alerts.

        This is the line a chemist reads: "the benzamidine series is clean, the
        warfarin-like series carries a coumarin and an enone".
        """
        out: List[Dict[str, Any]] = []
        index = {molecule.index: molecule for molecule in self.molecules}
        for scaffold, members in self.groups:
            flagged = [index[i] for i in members if i in index and not index[i].clean]
            names: Dict[str, int] = {}
            for molecule in flagged:
                for alert in molecule.alerts:
                    names[alert.name] = names.get(alert.name, 0) + 1
            out.append(
                {
                    "scaffold": scaffold or "(acyclic)",
                    "n": len(members),
                    "n_flagged": len(flagged),
                    "alert_counts": names,
                    "molecules": [index[i].name for i in members if i in index],
                }
            )
        return out

    def table(self, limit: int = 0) -> str:
        """Per molecule: name, affinity, series size, alert count, and the detail."""
        rows = self.molecules if limit <= 0 else self.molecules[: int(limit)]
        if not rows:
            return "no molecule"
        lines = [
            f"{'name':<26}{'affinity':>9}{'series':>8}{'alerts':>7}  alerts / failed rules",
            "-" * 26 + "-" * 9 + "-" * 8 + "-" * 7 + "  " + "-" * 44,
        ]
        for molecule in rows:
            affinity = "" if molecule.affinity is None else f"{molecule.affinity:.3f}"
            detail = molecule.summary()
            if molecule.failed_rules:
                detail += "  |  rules: " + ", ".join(molecule.failed_rules)
            lines.append(
                f"{molecule.name[:25]:<26}{affinity:>9}"
                f"{self._group_size(molecule.index):>8}{len(molecule.alerts):>7}  {detail[:120]}"
            )
        if limit > 0 and len(self.molecules) > limit:
            lines.append(f"... and {len(self.molecules) - limit} more molecule(s)")
        return "\n".join(lines)

    def _group_size(self, index: int) -> int:
        for _, members in self.groups:
            if index in members:
                return len(members)
        return 0

    def series_table(self) -> str:
        lines = [
            f"{'series':<44}{'n':>5}{'flagged':>9}  alerts",
            "-" * 44 + "-" * 5 + "-" * 9 + "  " + "-" * 30,
        ]
        for row in self.series_summary():
            counts = ", ".join(f"{name} x{count}" for name, count in sorted(row["alert_counts"].items()))
            lines.append(
                f"{row['scaffold'][:43]:<44}{row['n']:>5}{row['n_flagged']:>9}  {counts[:60]}"
            )
        return "\n".join(lines)

    def rates_table(self) -> str:
        if not self.rates:
            return "no catalogue rate to report"
        lines = [f"{'catalogue':<12}{'flagged':>9}{'of':>6}{'fraction':>10}", "-" * 12 + "-" * 9 + "-" * 6 + "-" * 10]
        for name, row in self.rates.items():
            lines.append(
                f"{name:<12}{int(row['flagged']):>9}{int(row['n']):>6}{row['fraction']:>9.0%}"
            )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_molecules": int(self.n_molecules),
            "n_flagged": int(self.n_flagged),
            "n_clean": int(self.n_clean),
            "catalogues": list(self.catalogues),
            "rates": {name: dict(row) for name, row in self.rates.items()},
            "groups": [
                {"scaffold": scaffold or "(acyclic)", "members": [int(i) for i in members]}
                for scaffold, members in self.groups
            ],
            "series": self.series_summary(),
            "molecules": [molecule.as_dict() for molecule in self.molecules],
            "seconds": round(float(self.seconds), 4),
            "notes": list(self.notes),
        }


def triage_molecule(
    mol,
    *,
    index: int = 0,
    name: str = "",
    affinity: Optional[float] = None,
    catalogues: Sequence[str] = ("PAINS", "BRENK", "NIH", "ZINC"),
    smarts: bool = True,
    sets: Sequence[str] = LIABILITY_SETS,
) -> MoleculeTriage:
    """Triage one molecule: catalogue alerts, SMARTS alerts and the property panel.

    The alert list keeps every match (not just the first), each with its atoms and
    its location, because "flagged" without "where" is not actionable.
    """
    require_rdkit()
    label = name or _mol_name(mol)
    alerts: List[AlertHit] = []
    if catalogues:
        alerts.extend(catalogue_alerts(mol, catalogues=catalogues))
    if smarts:
        alerts.extend(smarts_alerts(mol, sets=sets))
    from .scaffold import scaffold_key

    return MoleculeTriage(
        index=int(index),
        name=label,
        smiles=_safe_smiles(mol),
        alerts=alerts,
        properties=panel(mol),
        scaffold=scaffold_key(mol),
        affinity=affinity,
    )


def triage_library(
    molecules: Sequence[Any],
    *,
    names: Optional[Sequence[str]] = None,
    affinities: Optional[Dict[str, float]] = None,
    catalogues: Sequence[str] = ("PAINS", "BRENK", "NIH", "ZINC"),
    smarts: bool = True,
    group_by_scaffold: bool = True,
) -> TriageReport:
    """Triage a whole hit list and group it by scaffold series.

    ``molecules`` may be molecules or SMILES; ``affinities`` maps a molecule name
    to a docked or measured value, so the table can be read in the order a chemist
    works (this is the series, this is its affinity, this is what is wrong with
    it).  The groups come from :func:`odock.scaffold.scaffold_groups`, so the
    series in this report are the same series ``odock scaffolds`` prints.

    The report's notes carry the honesty statement and, when a catalogue flags more
    than a quarter of the set, say so explicitly: on a small library an alert
    catalogue can flag a third of it, and a reader has to see that before reading
    any ranking.
    """
    require_rdkit()
    start = time.perf_counter()
    mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in molecules]
    for position, mol in enumerate(mols):
        if mol is None:
            raise ValueError(f"molecule {position + 1} is not a parsable molecule")
    lookup = {str(k): float(v) for k, v in (affinities or {}).items() if v is not None}
    triaged: List[MoleculeTriage] = []
    for index, mol in enumerate(mols):
        label = (
            str(names[index])
            if names is not None and index < len(names)
            else _mol_name(mol, f"ligand_{index + 1}")
        )
        triaged.append(
            triage_molecule(
                mol,
                index=index,
                name=label,
                affinity=lookup.get(label),
                catalogues=catalogues,
                smarts=smarts,
            )
        )

    groups: List[Tuple[str, List[int]]] = []
    if group_by_scaffold:
        from .scaffold import scaffold_groups

        for group in scaffold_groups(mols, names=[molecule.name for molecule in triaged]):
            groups.append((group.key, list(group.members)))
    else:
        groups.append(("", list(range(len(triaged)))))

    rates: Dict[str, Dict[str, float]] = {}
    for catalogue in catalogues:
        key = str(catalogue).upper()
        flagged = sum(
            1
            for molecule in triaged
            if any(alert.origin == key for alert in molecule.alerts)
        )
        rates[key] = {
            "flagged": float(flagged),
            "n": float(len(triaged)),
            "fraction": (flagged / len(triaged)) if triaged else 0.0,
        }

    notes = [
        "Alerts are literature-derived patterns, not predictions: a match is not a "
        "measurement, an alert set's false-positive rate is not a probability, and "
        "the absence of an alert is not safety.",
        "None of this substitutes for an assay.",
    ]
    if not _HAVE_CATALOGUES:
        notes.append(
            "this RDKit build has no FilterCatalog, so only the embedded SMARTS "
            "alerts ran"
        )
    if not HAVE_SA_SCORE:
        notes.append(
            "the RDKit contrib synthetic-accessibility scorer is not installed in "
            "this build, so SA_score is reported as unavailable rather than guessed"
        )
    heavy = [
        f"{name} {row['fraction']:.0%}"
        for name, row in rates.items()
        if row["n"] and row["fraction"] > 0.25
    ]
    if heavy:
        notes.append(
            "these catalogues flag more than a quarter of this set ("
            + ", ".join(heavy)
            + "): on a small or unusual library that is a statement about the "
            "catalogue's breadth, not about the molecules"
        )
    return TriageReport(
        molecules=triaged,
        groups=groups,
        rates=rates,
        catalogues=tuple(str(name).upper() for name in catalogues),
        seconds=time.perf_counter() - start,
        notes=notes,
    )


def alert_rates(
    molecules: Sequence[Any],
    *,
    catalogues: Sequence[str] = ("PAINS", "BRENK", "NIH", "ZINC"),
    smarts: bool = True,
) -> Dict[str, Dict[str, float]]:
    """How many molecules each alert source flags, so a reader can see the rates.

    A catalogue that flags a third of a 142-molecule set is telling you about the
    catalogue's breadth; a reader who is shown the rate can discount the ranking
    accordingly, and one who is not will read the ranking as a measurement.
    """
    require_rdkit()
    mols = [Chem.MolFromSmiles(item) if isinstance(item, str) else item for item in molecules]
    total = len(mols)
    out: Dict[str, Dict[str, float]] = {}
    for catalogue in catalogues:
        key = str(catalogue).upper()
        flagged = sum(1 for mol in mols if catalogue_alerts(mol, catalogues=[key]))
        out[key] = {
            "flagged": float(flagged),
            "n": float(total),
            "fraction": (flagged / total) if total else 0.0,
        }
    if smarts:
        for family in LIABILITY_SETS:
            flagged = sum(1 for mol in mols if smarts_alerts(mol, sets=[family]))
            out[f"smarts:{family}"] = {
                "flagged": float(flagged),
                "n": float(total),
                "fraction": (flagged / total) if total else 0.0,
            }
    return out


def _mol_name(mol, fallback: str = "ligand") -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def _safe_smiles(mol) -> str:
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - defensive
        return ""

# SPDX-License-Identifier: GPL-3.0-or-later
"""Library pre-filters for virtual screening: Lipinski, Veber and PAINS.

The project requirements ask for three pre-filters before an expensive docking
campaign.  All three are pure graph/descriptor chemistry, so a library can be
filtered *before* any 3-D conformer is generated:

* :func:`lipinski` — MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10;
* :func:`veber` — rotatable bonds <= 10, tPSA <= 140 A^2, with the rotatable
  bonds counted by :func:`odock.chem.ligand.rotatable_bonds` so that the filter
  sees exactly the torsions the docking engine will sample;
* :func:`pains` — pan-assay interference compounds.  The primary path is
  RDKit's :class:`~rdkit.Chem.FilterCatalog.FilterCatalog` with the published
  480-entry PAINS catalogue; when the RDKit build has no ``FilterCatalog`` the
  module falls back to :data:`PAINS_SMARTS`, a curated dictionary of 46 named
  SMARTS covering the published frequent-hitter motifs (quinones, catechols,
  rhodanines, Michael acceptors, azo/hydrazine/nitroso, epoxides, reactive
  halides, ...).

:func:`drug_like` combines the three verdicts into the single dictionary the
GUI table and the report writers consume.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    Crippen = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    Lipinski = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

try:  # The filter catalogues moved around between RDKit releases.
    from rdkit.Chem import FilterCatalog as _FilterCatalog

    _HAVE_FILTER_CATALOG = True
except Exception:  # pragma: no cover
    _FilterCatalog = None  # type: ignore[assignment]
    _HAVE_FILTER_CATALOG = False

from .chem.ligand import rotatable_bonds as _rotatable_bonds

__all__ = [
    "LIPINSKI_MAX_MW",
    "LIPINSKI_MAX_LOGP",
    "LIPINSKI_MAX_HBD",
    "LIPINSKI_MAX_HBA",
    "VEBER_MAX_ROTATABLE",
    "VEBER_MAX_TPSA",
    "PAINS_SMARTS",
    "HAVE_FILTER_CATALOG",
    "FilterResult",
    "lipinski",
    "veber",
    "pains",
    "match_pains_smarts",
    "drug_like",
    "filter_library",
]

# ---------------------------------------------------------------------------
# Thresholds (the drug-likeness rules the project specifies)
# ---------------------------------------------------------------------------

LIPINSKI_MAX_MW = 500.0
LIPINSKI_MAX_LOGP = 5.0
LIPINSKI_MAX_HBD = 5
LIPINSKI_MAX_HBA = 10

VEBER_MAX_ROTATABLE = 10
VEBER_MAX_TPSA = 140.0


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing."""
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for the drug-likeness filters. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """The verdict of one filter."""

    #: Filter name, e.g. ``"Lipinski"``.
    name: str
    #: ``True`` only when no rule was violated.
    passed: bool
    #: Human-readable rule breaches, e.g. ``"MW 563.1 > 500"``.
    violations: List[str] = field(default_factory=list)
    #: The numbers the verdict is based on.
    properties: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """One line for a log or a status bar."""
        if self.passed:
            return f"{self.name}: pass ({self._property_text()})"
        return f"{self.name}: fail - " + "; ".join(self.violations)

    def _property_text(self) -> str:
        return ", ".join(f"{k}={v:.4g}" for k, v in self.properties.items())

    def as_dict(self) -> Dict[str, object]:
        """A JSON-serialisable view."""
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "violations": list(self.violations),
            "properties": {k: float(v) for k, v in self.properties.items()},
        }


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def _crippen_logp(mol) -> float:
    """Crippen/Wildman LogP, computed on an explicit-H copy.

    The atom contributions are identical either way, but RDKit logs
    "Molecule does not have explicit Hs. Consider calling AddHs()" for an
    implicit-H molecule — noise in a screening loop, so the copy is used.
    """
    try:
        work = mol
        if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
            work = Chem.AddHs(mol)
        return float(Crippen.MolLogP(work))
    except Exception:
        try:
            return float(Crippen.MolLogP(mol))
        except Exception:
            return float("nan")


def _lipinski_values(mol) -> Dict[str, float]:
    """The four numbers Lipinski's rule uses."""
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "LogP": _crippen_logp(mol),
        "HBD": float(Lipinski.NumHDonors(mol)),
        "HBA": float(Lipinski.NumHAcceptors(mol)),
    }


def _veber_values(mol) -> Dict[str, float]:
    """The two numbers Veber's rule uses.

    Each filter computes only what it needs: triaging a 100k-compound library is
    the whole point of these filters, and the Crippen LogP plus the
    torsion perception are the expensive parts.
    """
    return {
        "RotB": float(len(_rotatable_bonds(mol))),
        "tPSA": float(Descriptors.TPSA(mol)),
    }


# ---------------------------------------------------------------------------
# Lipinski and Veber
# ---------------------------------------------------------------------------


def lipinski(mol) -> FilterResult:
    """Lipinski's rule of five: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10.

    The thresholds are applied strictly — a single breach fails the molecule.
    The classical formulation tolerates one violation; that leniency is a
    policy decision for the caller, so the raw count is reported as
    ``properties["n_violations"]`` and :func:`drug_like` exposes it too.

    ``LogP`` is the Crippen/Wildman contribution model; it is a calculated
    value, not an experiment, so a molecule sitting on the threshold should be
    treated as borderline rather than rejected.
    """
    require_rdkit()
    values = _lipinski_values(mol)
    violations: List[str] = []
    if values["MW"] > LIPINSKI_MAX_MW:
        violations.append(f"MW {values['MW']:.1f} > {LIPINSKI_MAX_MW:.0f}")
    if values["LogP"] > LIPINSKI_MAX_LOGP:
        violations.append(f"LogP {values['LogP']:.2f} > {LIPINSKI_MAX_LOGP:.0f}")
    if values["HBD"] > LIPINSKI_MAX_HBD:
        violations.append(f"HBD {values['HBD']:.0f} > {LIPINSKI_MAX_HBD}")
    if values["HBA"] > LIPINSKI_MAX_HBA:
        violations.append(f"HBA {values['HBA']:.0f} > {LIPINSKI_MAX_HBA}")
    properties = {
        "MW": values["MW"],
        "LogP": values["LogP"],
        "HBD": values["HBD"],
        "HBA": values["HBA"],
        "n_violations": float(len(violations)),
    }
    return FilterResult("Lipinski", not violations, violations, properties)


def veber(mol) -> FilterResult:
    """Veber's oral-bioavailability rules: RotB <= 10, tPSA <= 140 A^2.

    Rotatable bonds are counted by :func:`odock.chem.ligand.rotatable_bonds`, so
    an amide or a tert-butyl is *not* counted — the filter and the docking
    engine agree on what a torsion is.  RDKit's own
    :func:`~rdkit.Chem.Descriptors.NumRotatableBonds` uses a looser rule and
    would report a larger number.
    """
    require_rdkit()
    values = _veber_values(mol)
    violations: List[str] = []
    if values["RotB"] > VEBER_MAX_ROTATABLE:
        violations.append(
            f"RotB {values['RotB']:.0f} > {VEBER_MAX_ROTATABLE}"
        )
    if values["tPSA"] > VEBER_MAX_TPSA:
        violations.append(f"tPSA {values['tPSA']:.1f} > {VEBER_MAX_TPSA:.0f}")
    properties = {
        "RotB": values["RotB"],
        "tPSA": values["tPSA"],
        "n_violations": float(len(violations)),
    }
    return FilterResult("Veber", not violations, violations, properties)


# ---------------------------------------------------------------------------
# PAINS
# ---------------------------------------------------------------------------

#: A curated frequent-hitter / reactive-group SMARTS dictionary, used when the
#: RDKit build has no ``FilterCatalog``.  Every entry names the motif it looks
#: for, so a hit can be explained to the user.  The list is deliberately
#: conservative about *specific* motifs and never about whole compound classes:
#: aspirin, paracetamol, caffeine, glucose, ibuprofen, metformin, warfarin,
#: propranolol, sildenafil and imatinib all come out clean.
PAINS_SMARTS: Dict[str, str] = {
    # -- quinones and polyphenols -------------------------------------------
    "para_quinone": "O=C1C=CC(=O)C=C1",
    "ortho_quinone": "O=C1C(=O)C=CC=C1",
    "quinone_methide": "C=C1C=CC(=O)C=C1",
    "catechol": "c1cc(O)c(O)cc1",
    "pyrogallol": "c1cc(O)c(O)c(O)c1",
    "hydroquinone": "Oc1ccc(O)cc1",
    # -- rhodanines, thiazolidinediones and hydantoins ----------------------
    "rhodanine": "O=C1CSC(=S)N1",
    "thiazolidinedione": "O=C1CSC(=O)N1",
    "hydantoin": "O=C1CNC(=O)N1",
    "maleimide": "O=C1C=CC(=O)N1",
    # -- Michael acceptors and other alkylating groups ----------------------
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "chalcone": "[#6](=O)[CX3]=[CX3]c1ccccc1",
    "acrylate": "[CX3]=[CX3][CX3](=O)[OX2]",
    "cyano_acrylate": "N#C[CX3]=[CX3][CX3](=O)",
    "vinyl_sulfone": "[CX3]=[CX3][SX4](=O)=O",
    "alpha_halo_ketone": "[CX3](=O)[CX4][Cl,Br,I]",
    # -- strained and reactive heterocycles ---------------------------------
    "epoxide": "[OX2;r3]1[#6;r3][#6;r3]1",
    "aziridine": "[NX3;r3]1[#6;r3][#6;r3]1",
    # -- azo, azide, diazo, nitroso, nitro ----------------------------------
    "azo": "[#6][NX2]=[NX2][#6]",
    "azide": "[NX2]=[NX2+]=[NX1-]",
    "diazo": "[#6]=[NX2+]=[NX1-]",
    "nitroso": "[#6][NX2]=[OX1]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "n_oxide": "[#7+;!$([#7+]=[OX1])][OX1-]",
    # -- hydrazines, hydrazones, hydrazides ---------------------------------
    "hydrazine": "[NX3][NX3]",
    "hydrazone": "[NX3][NX2]=[#6]",
    "acylhydrazide": "[CX3](=O)[NX3][NX3]",
    "hydroxamic_acid": "[CX3](=O)[NX3][OX2H]",
    # -- isocyanates / isothiocyanates / thiocarbonyls ----------------------
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "isothiocyanate": "[NX2]=[CX2]=[SX1]",
    "thiocarbonyl": "[#6]=[SX1]",
    # -- halides and activated carbonyls ------------------------------------
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "acyl_halide": "[CX3](=O)[Cl,Br,I,F]",
    "sulfonyl_halide": "[SX4](=O)(=O)[Cl,Br,I,F]",
    "anhydride": "[CX3](=O)[OX2][CX3]=O",
    "aldehyde": "[CX3H1](=O)[#6,#1]",
    # -- peroxides, disulfides, thiols --------------------------------------
    "peroxide": "[OX2][OX2]",
    "disulfide": "[SX2][SX2]",
    "thiol": "[SX2H]",
    # -- conjugated and polymerisable systems -------------------------------
    "conjugated_diene": "[CX3]=[CX3][CX3]=[CX3]",
    "styrene": "c1ccccc1[CX3]=[CX3]",
    "vicinal_diketone": "[#6](=[OX1])[#6]=[OX1]",
    # -- anilines and benzidines --------------------------------------------
    "dialkyl_aniline": "c1ccccc1[NX3;H0]([#6])[#6]",
    "benzidine": "Nc1ccc(cc1)-c1ccc(N)cc1",
    "n_halo": "[NX3][Cl,Br,I]",
}

#: Whether this RDKit build ships the published filter catalogues.
HAVE_FILTER_CATALOG = _HAVE_FILTER_CATALOG

_CATALOG_LOCK = threading.Lock()
_PAINS_CATALOG = None
_compiled_cache: Optional[Dict[str, object]] = None


def _pains_catalog():
    """The shared RDKit PAINS catalogue, or ``None`` when unavailable."""
    global _PAINS_CATALOG
    if not _HAVE_FILTER_CATALOG:
        return None
    if _PAINS_CATALOG is None:
        with _CATALOG_LOCK:
            if _PAINS_CATALOG is None:
                try:
                    params = _FilterCatalog.FilterCatalogParams()
                    params.AddCatalog(
                        _FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS
                    )
                    _PAINS_CATALOG = _FilterCatalog.FilterCatalog(params)
                except Exception:  # pragma: no cover - broken RDKit build
                    _PAINS_CATALOG = False  # type: ignore[assignment]
    return _PAINS_CATALOG or None


def _compiled_patterns() -> Dict[str, object]:
    """Compile :data:`PAINS_SMARTS` once, dropping anything unparsable."""
    global _compiled_cache
    if _compiled_cache is None:
        compiled: Dict[str, object] = {}
        for name, smarts in PAINS_SMARTS.items():
            query = Chem.MolFromSmarts(smarts)
            if query is not None:
                compiled[name] = query
        _compiled_cache = compiled
    return _compiled_cache


def match_pains_smarts(mol, patterns: Optional[Dict[str, str]] = None) -> List[str]:
    """Names of the embedded PAINS SMARTS that ``mol`` matches.

    This is the fallback path of :func:`pains`, exposed so that the curated
    dictionary can be tested and inspected on a build that *does* ship
    ``FilterCatalog``.
    """
    require_rdkit()
    if patterns is None:
        compiled = _compiled_patterns()
    else:
        compiled = {}
        for name, smarts in patterns.items():
            query = Chem.MolFromSmarts(smarts)
            if query is not None:
                compiled[name] = query
    return [name for name, query in compiled.items() if mol.HasSubstructMatch(query)]


def pains(mol, *, use_catalog: Optional[bool] = None) -> FilterResult:
    """Flag pan-assay interference compounds.

    Parameters
    ----------
    mol
        The molecule to screen.  No conformer is needed.
    use_catalog
        ``None`` (default) uses RDKit's published PAINS catalogue when the build
        has one and :data:`PAINS_SMARTS` otherwise.  ``False`` forces the
        embedded dictionary, ``True`` forces the catalogue and raises when the
        build has none.

    The verdict is a failure as soon as one pattern matches; the matched
    pattern identifiers are the violations, and ``properties`` reports how many
    hits there were (``PAINS_hits``) and how many patterns were searched
    (``PAINS_patterns_used``) so a report can say "0 of 480".  If the matching
    itself fails (an unsanitised molecule, say) the molecule is reported as
    failing with ``PAINS_error`` set: silently passing a molecule whose
    interference profile could not be established would be the wrong default
    for a screening pipeline.
    """
    require_rdkit()
    want_catalog = use_catalog is not False and _HAVE_FILTER_CATALOG
    if use_catalog is True and not _HAVE_FILTER_CATALOG:
        raise RuntimeError("this RDKit build does not provide FilterCatalog")

    hits: List[str] = []
    n_patterns = 0
    try:
        if want_catalog:
            catalog = _pains_catalog()
            if catalog is not None:
                n_patterns = int(catalog.GetNumEntries())
                seen = set()
                for entry in catalog.GetMatches(mol):
                    description = entry.GetDescription()
                    if description not in seen:
                        seen.add(description)
                        hits.append(description)
        if not want_catalog or n_patterns == 0:
            compiled = _compiled_patterns()
            n_patterns = len(compiled)
            hits.extend(match_pains_smarts(mol))
    except Exception as exc:
        violations = [f"PAINS matching failed: {exc}"]
        return FilterResult(
            "PAINS",
            False,
            violations,
            {"PAINS_hits": 0.0, "PAINS_patterns_used": float(n_patterns), "PAINS_error": 1.0},
        )

    violations = [f"PAINS {hit}" for hit in hits]
    properties = {
        "PAINS_hits": float(len(hits)),
        "PAINS_patterns_used": float(n_patterns),
    }
    return FilterResult("PAINS", not hits, violations, properties)


# ---------------------------------------------------------------------------
# Combined verdicts
# ---------------------------------------------------------------------------


def drug_like(mol) -> Dict[str, object]:
    """Run every pre-filter on ``mol`` and combine the verdicts.

    Returns
    -------
    ``{"passed": bool, "results": [FilterResult, ...], "properties": {...},
    "violations": [...]}``

    ``properties`` merges the numeric descriptors the GUI table shows —
    ``MW``, ``LogP``, ``HBD``, ``HBA``, ``RotB``, ``tPSA``, ``PAINS_hits``,
    ``PAINS_patterns_used``, ``Lipinski_violations``, ``Veber_violations``.
    ``violations`` is the flattened list, each entry prefixed with the filter
    that raised it (``"Lipinski: MW 563.1 > 500"``).

    Only descriptors are computed, never coordinates, so a 100k-compound SDF
    can be triaged without generating a single conformer.
    """
    require_rdkit()
    results = [lipinski(mol), veber(mol), pains(mol)]
    by_name = {result.name: result for result in results}
    lipinski_props = by_name["Lipinski"].properties
    veber_props = by_name["Veber"].properties
    pains_props = by_name["PAINS"].properties
    properties: Dict[str, float] = {
        "MW": lipinski_props["MW"],
        "LogP": lipinski_props["LogP"],
        "HBD": lipinski_props["HBD"],
        "HBA": lipinski_props["HBA"],
        "RotB": veber_props["RotB"],
        "tPSA": veber_props["tPSA"],
        "PAINS_hits": pains_props["PAINS_hits"],
        "PAINS_patterns_used": pains_props["PAINS_patterns_used"],
        "Lipinski_violations": lipinski_props["n_violations"],
        "Veber_violations": veber_props["n_violations"],
    }
    violations = [
        f"{result.name}: {violation}"
        for result in results
        for violation in result.violations
    ]
    return {
        "passed": all(result.passed for result in results),
        "results": results,
        "properties": properties,
        "violations": violations,
    }


def filter_library(mols: Iterable[object]) -> List[Dict[str, object]]:
    """Run :func:`drug_like` over a library, keeping the input order.

    Each verdict gains ``index`` and ``name`` (the molecule's ``_Name``
    property, or ``ligand_<n>``), which is what a screening table needs to point
    back at the input record.
    """
    require_rdkit()
    out: List[Dict[str, object]] = []
    for index, mol in enumerate(mols):
        verdict = drug_like(mol)
        name = ""
        try:
            if mol.HasProp("_Name"):
                name = mol.GetProp("_Name").strip()
        except Exception:  # pragma: no cover - defensive
            name = ""
        row: Dict[str, object] = {
            "index": index,
            "name": name or f"ligand_{index + 1}",
        }
        row.update(verdict)
        out.append(row)
    return out

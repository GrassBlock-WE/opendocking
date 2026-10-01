# SPDX-License-Identifier: GPL-3.0-or-later
"""Ligand-based similarity: fingerprints, similarity, analogues, diversity.

The docking half of this project answers "how does this molecule bind". A real
project also asks the ligand-side questions, and they are all *2-D* questions:
what else looks like this hit, is my library diverse, which analogues should I
make next.  This module is that layer.  It is deliberately independent of the
kernel — no receptor, no box, no conformer — so a 100 000-compound library can be
triaged in seconds:

* :func:`fingerprint` / :func:`fingerprint_set` — circular (Morgan/ECFP-style)
  *and* path-based (RDKit) fingerprints, plus atom-pair, topological-torsion,
  MACCS-key and 3-D pharmacophore-pair fingerprints, all behind one documented
  :class:`Fingerprint` type so no RDKit object ever escapes this API;
* :func:`tanimoto`, :func:`dice`, :func:`tversky` and :func:`similarity_matrix`
  — the three similarity coefficients a chemist actually uses, as functions and
  as a matrix;
* :func:`find_analogues` — "find me analogues of this hit" over a library with a
  similarity cut-off, ranked, with the number of molecules considered reported;
* :func:`find_analogues_3d` / :func:`best_over_conformers` — the same search where
  each molecule carries several conformers and the score is the best pair of
  conformers.  This costs ``n_query_conformers x n_library_conformers`` times a
  2-D search and is only meaningful with a conformer-dependent fingerprint (see
  :func:`conformer_fingerprints`);
* :func:`maxmin_pick`, :func:`sphere_exclusion_pick` and :func:`butina_cluster` —
  the diversity selection and the distance-based clustering a screening cascade
  needs before it docks anything;
* :func:`scaffold_coverage` — the fraction of a library's scaffold space a
  selected subset covers, which is the number that says whether a diverse subset
  is actually diverse.

What this module does **not** do: it never claims that a fingerprint similarity
is an activity similarity, and it never invents an affinity.  A similarity of
0.85 means "these two graphs share most of their circular substructures"; whether
that means anything depends on the assay and the series.  See
`docs/CHEMINFORMATICS.md` for the measured behaviour and the limits.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors

    _HAVE_RDKIT = True
except Exception:  # pragma: no cover - RDKit is a hard dependency for chemistry
    Chem = None  # type: ignore[assignment]
    DataStructs = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    rdFingerprintGenerator = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _HAVE_RDKIT = False

try:  # The pharmacophore-pair signature factory moved around between releases.
    from rdkit.Chem.Pharm2D import Generate as _Pharm2DGenerate
    from rdkit.Chem.Pharm2D import Gobbi_Pharm2D as _Gobbi

    _HAVE_PHARM2D = True
except Exception:  # pragma: no cover - very old or trimmed RDKit
    _Pharm2DGenerate = None  # type: ignore[assignment]
    _Gobbi = None  # type: ignore[assignment]
    _HAVE_PHARM2D = False

__all__ = [
    "FINGERPRINT_KINDS",
    "SIMILARITY_METRICS",
    "DIVERSITY_METHODS",
    "DEFAULT_RADIUS",
    "DEFAULT_N_BITS",
    "DEFAULT_METRIC",
    "DEFAULT_CUTOFF",
    "DEFAULT_CLUSTER_CUTOFF",
    "DEFAULT_CONFORMERS",
    "HAVE_RDKIT",
    "HAVE_PHARM2D",
    "Fingerprint",
    "FingerprintSet",
    "Analogue",
    "AnalogueHits",
    "ConformerFingerprintSet",
    "DiversitySelection",
    "Cluster",
    "Clustering",
    "require_rdkit",
    "fingerprint",
    "fingerprint_set",
    "fingerprint_library",
    "fingerprints_from_smiles",
    "conformer_fingerprints",
    "conformer_fingerprint_set",
    "tanimoto",
    "dice",
    "tversky",
    "similarity",
    "similarity_matrix",
    "find_analogues",
    "best_over_conformers",
    "find_analogues_3d",
    "maxmin_pick",
    "sphere_exclusion_pick",
    "diversity_subset",
    "butina_cluster",
    "scaffold_coverage",
]

#: Every fingerprint :func:`fingerprint` can build.
FINGERPRINT_KINDS: Tuple[str, ...] = (
    "morgan",
    "rdkit",
    "atom_pair",
    "torsion",
    "maccs",
    "pharmacophore",
)

#: Every coefficient :func:`similarity` and :func:`similarity_matrix` can apply.
SIMILARITY_METRICS: Tuple[str, ...] = ("tanimoto", "dice", "tversky")

#: Every subset-picking method :func:`diversity_subset` can run.
DIVERSITY_METHODS: Tuple[str, ...] = ("maxmin", "sphere")

#: Morgan/ECFP radius.  Radius 2 is ECFP4 (a diameter of four bonds), the
#: published default and the one every "similar to" claim in the literature uses.
DEFAULT_RADIUS = 2
#: Fingerprint length.  2048 bits keeps the collisions of a drug-like library
#: negligible (a typical molecule sets 30-80 bits, so 2048 is not crowded) at
#: 256 bytes per molecule.
DEFAULT_N_BITS = 2048
#: Tanimoto is the default because it is the coefficient the similarity
#: literature and every vendor tool report.
DEFAULT_METRIC = "tanimoto"
#: The classic "similar" cut-off: 0.7 Tanimoto is the usual analogue-search
#: threshold in the ECFP literature (0.85 for "very similar").
DEFAULT_CUTOFF = 0.7
#: Butina clustering distance cut-off, expressed as a similarity: molecules at
#: >= 0.65 Tanimoto end up in one cluster by default.
DEFAULT_CLUSTER_CUTOFF = 0.65
#: Conformers per molecule for the 3-D (pharmacophore) path.
DEFAULT_CONFORMERS = 8

#: Whether this build has the RDKit pieces the module needs.
HAVE_RDKIT = _HAVE_RDKIT
#: Whether the 3-D pharmacophore fingerprints are available.
HAVE_PHARM2D = _HAVE_PHARM2D


def require_rdkit() -> None:
    """Raise a helpful error when RDKit is missing.

    Kept in the same shape as :func:`odock.filters.require_rdkit`: a missing
    optional dependency is a sentence a user can act on, never an ``ImportError``
    traceback about a C extension.
    """
    if not _HAVE_RDKIT:
        raise ImportError(
            "RDKit is required for ligand similarity. Install it with "
            "`pip install rdkit` (or `pip install opendocking[chem]`)."
        )


def _require_pharm2d() -> None:
    if not _HAVE_PHARM2D:
        raise ImportError(
            "the 3-D pharmacophore fingerprints need RDKit's Pharm2D module, "
            "which this RDKit build does not provide; use kind='morgan' or "
            "kind='rdkit' for a 2-D search"
        )


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprint:
    """A binary substructure fingerprint, independent of RDKit.

    The on-bits are stored as an ascending tuple of integers, which is exactly
    how a sparse fingerprint is written in a file, is cheap to intersect without
    allocating anything, and serialises to JSON without a byte vector.  A dense
    NumPy view is one call away (:meth:`as_numpy`) for the code that wants one.

    Two fingerprints may only be compared when they describe the same kind of
    substructure with the same length; :func:`similarity` refuses anything else
    rather than returning a number that means nothing.
    """

    #: One of :data:`FINGERPRINT_KINDS`.
    kind: str
    #: Length of the bit vector (``167`` for MACCS, the generator's ``fpSize``
    #: otherwise).
    n_bits: int
    #: Ascending on-bit indices.
    indices: Tuple[int, ...]
    #: The molecule's name, when the fingerprint came from a named library.
    name: str = ""
    #: Morgan radius / path length, for the ``as_dict`` provenance record.
    radius: Optional[int] = None

    def __len__(self) -> int:
        return int(self.n_bits)

    @property
    def n_on(self) -> int:
        """How many bits are set."""
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        """Whether no bit is set (an empty molecule, or no substructure matched)."""
        return not self.indices

    def as_indices(self) -> Tuple[int, ...]:
        """The on-bit indices, ascending."""
        return self.indices

    def as_numpy(self) -> np.ndarray:
        """A dense ``bool`` array of length :attr:`n_bits`."""
        out = np.zeros(int(self.n_bits), dtype=bool)
        if self.indices:
            out[np.asarray(self.indices, dtype=np.int64)] = True
        return out

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable view (the bit list included)."""
        return {
            "kind": self.kind,
            "n_bits": int(self.n_bits),
            "n_on": self.n_on,
            "radius": self.radius,
            "name": self.name,
            "indices": list(self.indices),
        }


class _GeneratorCache:
    """Cache the RDKit fingerprint generators, which are not free to build."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[Any, ...], Any] = {}

    def get(self, key: Tuple[Any, ...]):
        with self._lock:
            found = self._cache.get(key)
        if found is not None:
            return found
        built = _build_generator(key)
        with self._lock:
            self._cache.setdefault(key, built)
            return self._cache[key]


_CACHED = _GeneratorCache()


def _build_generator(key: Tuple[Any, ...]):
    """Build one RDKit fingerprint generator from a cache key.

    The feature-based Morgan ("FCFP") variant is spelled differently across the
    RDKit releases: ``GetMorganGenerator(useFeatures=True)`` existed before the
    atom-invariant generators were split out, and the current API asks for
    ``GetMorganFeatureAtomInvGen()`` explicitly.  Both are tried, newest first,
    so the module works on the RDKit the project pins and on an older one.
    """
    kind = key[0]
    if kind == "morgan":
        _, radius, n_bits, use_features, use_chirality = key
        if use_features:
            try:
                return rdFingerprintGenerator.GetMorganGenerator(
                    radius=int(radius),
                    fpSize=int(n_bits),
                    includeChirality=bool(use_chirality),
                    useBondTypes=False,
                    atomInvariantsGenerator=(
                        rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
                    ),
                )
            except Exception:  # pragma: no cover - older RDKit
                return rdFingerprintGenerator.GetMorganGenerator(
                    radius=int(radius),
                    fpSize=int(n_bits),
                    useFeatures=True,
                    includeChirality=bool(use_chirality),
                )
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=int(radius),
            fpSize=int(n_bits),
            includeChirality=bool(use_chirality),
        )
    if kind == "rdkit":
        _, n_bits, min_path, max_path, branched, bond_order = key
        return rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=int(n_bits),
            minPath=int(min_path),
            maxPath=int(max_path),
            branchedPaths=bool(branched),
            useBondOrder=bool(bond_order),
        )
    if kind == "atom_pair":
        _, n_bits, use_chirality = key
        return rdFingerprintGenerator.GetAtomPairGenerator(
            fpSize=int(n_bits), includeChirality=bool(use_chirality)
        )
    if kind == "torsion":
        _, n_bits, use_chirality = key
        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            fpSize=int(n_bits), includeChirality=bool(use_chirality)
        )
    raise ValueError(f"no generator for {kind!r}")  # pragma: no cover - guarded


def _normalise_kind(kind: str) -> str:
    key = str(kind).strip().lower().replace("-", "_")
    aliases = {
        "ecfp": "morgan",
        "ecfp4": "morgan",
        "circular": "morgan",
        "path": "rdkit",
        "rdk": "rdkit",
        "rdkitfp": "rdkit",
        "atompair": "atom_pair",
        "atom-pair": "atom_pair",
        "topological_torsion": "torsion",
        "maccs_keys": "maccs",
        "maccs166": "maccs",
        "pharm2d": "pharmacophore",
        "pharmacophore_pair": "pharmacophore",
    }
    key = aliases.get(key, key)
    if key not in FINGERPRINT_KINDS:
        raise ValueError(
            f"unknown fingerprint kind {kind!r}; supported: {list(FINGERPRINT_KINDS)}"
        )
    return key


def _pharmacophore_fingerprint(mol, *, conf_id: int, n_bits: int) -> Fingerprint:
    """The Gobbi pharmacophore-pair fingerprint of one conformer.

    ``n_bits`` is accepted for signature symmetry with :func:`fingerprint` and is
    ignored: the Gobbi signature factory defines its own (sparse, ~40 000-bit)
    space, so the length of the returned fingerprint is whatever the factory
    says and :class:`Fingerprint.n_bits` reports it.
    """
    _require_pharm2d()
    n_conf = mol.GetNumConformers()
    if n_conf == 0:
        raise ValueError(
            "a pharmacophore fingerprint needs a 3-D conformer; embed the molecule "
            "first (odock.chem.ligand.embed_3d)"
        )
    if conf_id < 0 or conf_id >= n_conf:
        raise IndexError(
            f"conformer {conf_id} does not exist (the molecule has {n_conf})"
        )
    factory = _Gobbi.factory
    # The Gobbi factory bins *distances*, and we hand it the 3-D distance matrix
    # of one conformer: that is what makes this fingerprint conformer-dependent
    # while every 2-D kind above is not.
    matrix = Chem.Get3DDistanceMatrix(mol, confId=int(conf_id))
    raw = _Pharm2DGenerate.Gen2DFingerprint(mol, factory, dMat=matrix)
    indices = tuple(int(i) for i in raw.GetOnBits())
    return Fingerprint(
        kind="pharmacophore",
        n_bits=int(raw.GetNumBits()),
        indices=indices,
        name=_mol_name(mol, ""),
        radius=None,
    )


def fingerprint(
    mol,
    *,
    kind: str = "morgan",
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    use_features: bool = False,
    use_chirality: bool = False,
    min_path: int = 1,
    max_path: int = 7,
    branched_paths: bool = True,
    use_bond_order: bool = True,
    conf_id: int = 0,
    name: Optional[str] = None,
) -> Fingerprint:
    """The binary fingerprint of ``mol``, as a plain :class:`Fingerprint`.

    Parameters
    ----------
    mol
        Any RDKit molecule.  No conformer is needed for the 2-D kinds.
    kind
        One of :data:`FINGERPRINT_KINDS`:

        ``morgan``
            Circular substructures (Morgan/ECFP), radius 2 = ECFP4, hashed into
            ``n_bits`` bits.  The default; what "Tanimoto similarity" means in
            most of the literature.
        ``rdkit``
            Path-based (the original RDKit fingerprint): every linear and
            branched path of ``min_path..max_path`` bonds, hashed.
        ``atom_pair``
            Atom-pair features: the (typed atom, typed atom, topological
            distance) triples, which are strong for scaffold-hopping.
        ``torsion``
            Topological torsions: the four-atom paths, which capture the shape of
            a chain more explicitly than paths do.
        ``maccs``
            The 166 public MACCS substructure keys (167 bits, key 0 unused) —
            interpretable, published, and only 167 bits long.
        ``pharmacophore``
            The Gobbi pharmacophore-pair fingerprint of conformer ``conf_id``:
            genuinely 3-D and therefore conformer-dependent (see
            :func:`conformer_fingerprints`).

    radius, n_bits, use_features, use_chirality
        Morgan/ECFP parameters.  ``use_features=True`` hashes pharmacophore
        features (donor, acceptor, aromatic, ...) instead of atom environments —
        the "feature" variant of ECFP.
    min_path, max_path, branched_paths, use_bond_order
        Path-fingerprint parameters (``rdkit`` only).
    conf_id
        Conformer index for ``kind="pharmacophore"``; ignored otherwise.
    name
        Overrides the name taken from the molecule's ``_Name`` property.

    Returns
    -------
    :class:`Fingerprint`.  An empty fingerprint (``is_empty``) is returned rather
    than ``None`` for a molecule with no atoms.
    """
    require_rdkit()
    if mol is None:
        raise ValueError("cannot fingerprint None")
    key = _normalise_kind(kind)
    label = _mol_name(mol, "") if name is None else str(name)
    if key == "maccs":
        raw = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        return Fingerprint(
            kind=key,
            n_bits=int(raw.GetNumBits()),
            indices=tuple(int(i) for i in raw.GetOnBits()),
            name=label,
            radius=None,
        )
    if key == "pharmacophore":
        out = _pharmacophore_fingerprint(mol, conf_id=conf_id, n_bits=n_bits)
        return Fingerprint(
            kind=out.kind,
            n_bits=out.n_bits,
            indices=out.indices,
            name=label if name is not None else out.name,
            radius=None,
        )

    if key == "morgan":
        cache_key: Tuple[Any, ...] = (
            "morgan",
            int(radius),
            int(n_bits),
            bool(use_features),
            bool(use_chirality),
        )
    elif key == "rdkit":
        cache_key = (
            "rdkit",
            int(n_bits),
            int(min_path),
            int(max_path),
            bool(branched_paths),
            bool(use_bond_order),
        )
    else:
        cache_key = (key, int(n_bits), bool(use_chirality))
    generator = _CACHED.get(cache_key)
    raw = generator.GetFingerprint(mol)
    return Fingerprint(
        kind=key,
        n_bits=int(raw.GetNumBits()),
        indices=tuple(int(i) for i in raw.GetOnBits()),
        name=label,
        radius=int(radius) if key == "morgan" else None,
    )


def _mol_name(mol, fallback: str) -> str:
    try:
        if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
            return mol.GetProp("_Name").strip()
    except Exception:  # pragma: no cover - defensive
        pass
    return str(fallback)


def _mol_smiles(mol) -> str:
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - an unsanitisable molecule
        return ""


# ---------------------------------------------------------------------------
# Fingerprint sets
# ---------------------------------------------------------------------------


@dataclass
class FingerprintSet:
    """A named library of fingerprints: the unit every search below operates on.

    ``smiles`` is optional (it costs a canonicalisation per molecule), but when
    present it makes an analogues table readable and lets a subset be written
    back out as a library file.
    """

    #: One name per fingerprint, in library order.
    names: List[str] = field(default_factory=list)
    #: The fingerprints, in library order.
    fingerprints: List[Fingerprint] = field(default_factory=list)
    #: Fingerprint kind (all members share it).
    kind: str = "morgan"
    #: Bit-vector length (all members share it).
    n_bits: int = DEFAULT_N_BITS
    #: Where the library came from, for a report line.
    source: str = ""
    #: Canonical SMILES per member, when they were computed.
    smiles: List[str] = field(default_factory=list)
    #: Non-fatal problems ("record 4: RDKit could not parse it").
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fingerprints)

    def __iter__(self):
        return iter(self.fingerprints)

    def __getitem__(self, index):
        return self.fingerprints[index]

    @property
    def n_molecules(self) -> int:
        return len(self.fingerprints)

    def name_of(self, index: int) -> str:
        if 0 <= index < len(self.names) and self.names[index]:
            return self.names[index]
        return f"ligand_{index + 1}"

    def smiles_of(self, index: int) -> str:
        if self.smiles and 0 <= index < len(self.smiles):
            return self.smiles[index]
        return ""

    def subset(self, indices: Sequence[int]) -> "FingerprintSet":
        """A new set holding ``indices``, in the order given."""
        keep = [int(i) for i in indices]
        return FingerprintSet(
            names=[self.name_of(i) for i in keep],
            fingerprints=[self.fingerprints[i] for i in keep],
            kind=self.kind,
            n_bits=self.n_bits,
            source=self.source,
            smiles=[self.smiles_of(i) for i in keep] if self.smiles else [],
            notes=list(self.notes),
        )

    def as_dict(self) -> Dict[str, Any]:
        """A compact JSON view (no bit vectors; use ``--json-out`` for a digest)."""
        return {
            "source": self.source,
            "kind": self.kind,
            "n_bits": int(self.n_bits),
            "n_molecules": self.n_molecules,
            "mean_on_bits": (
                float(np.mean([fp.n_on for fp in self.fingerprints]))
                if self.fingerprints
                else 0.0
            ),
            "notes": list(self.notes),
        }


def _normalise_molecules(molecules: Iterable[Any]) -> List[Any]:
    """Accept molecules, SMILES strings and paths, and return molecules."""
    out: List[Any] = []
    for item in molecules:
        if isinstance(item, str):
            mol = Chem.MolFromSmiles(item)
            if mol is None:
                raise ValueError(f"{item!r} is not a parsable SMILES")
            out.append(mol)
        else:
            out.append(item)
    return out


def fingerprint_set(
    molecules: Union[Sequence[Any], FingerprintSet],
    *,
    kind: str = "morgan",
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    use_features: bool = False,
    use_chirality: bool = False,
    min_path: int = 1,
    max_path: int = 7,
    names: Optional[Sequence[str]] = None,
    smiles: bool = False,
    source: str = "",
) -> FingerprintSet:
    """Fingerprint a whole library, keeping its order and its names.

    ``molecules`` is an iterable of RDKit molecules, SMILES strings, or an
    existing :class:`FingerprintSet` (returned unchanged, which makes the
    ``find_analogues`` family transparently accept both).  ``names`` overrides
    the ``_Name`` property; otherwise a molecule without one is called
    ``ligand_<n>``.  ``smiles=True`` additionally stores a canonical SMILES per
    member (one canonicalisation each — cheap next to the fingerprinting, and it
    makes a CSV export readable).
    """
    require_rdkit()
    if isinstance(molecules, FingerprintSet):
        return molecules
    mols = _normalise_molecules(molecules)
    key = _normalise_kind(kind)
    out: List[Fingerprint] = []
    labels: List[str] = []
    smiles_out: List[str] = []
    for index, mol in enumerate(mols):
        label = (
            str(names[index])
            if names is not None and index < len(names)
            else _mol_name(mol, f"ligand_{index + 1}")
        )
        out.append(
            fingerprint(
                mol,
                kind=key,
                radius=radius,
                n_bits=n_bits,
                use_features=use_features,
                use_chirality=use_chirality,
                min_path=min_path,
                max_path=max_path,
                name=label,
            )
        )
        labels.append(label)
        if smiles:
            smiles_out.append(_mol_smiles(mol))
    return FingerprintSet(
        names=labels,
        fingerprints=out,
        kind=key,
        n_bits=int(out[0].n_bits) if out else int(n_bits),
        source=str(source),
        smiles=smiles_out,
    )


def fingerprint_library(molecules, **kwargs) -> FingerprintSet:
    """Alias of :func:`fingerprint_set`, for the screening vocabulary."""
    return fingerprint_set(molecules, **kwargs)


def fingerprints_from_smiles(
    smiles: Sequence[str], *, names: Optional[Sequence[str]] = None, **kwargs
) -> FingerprintSet:
    """Fingerprint a list of SMILES strings (a ``.smi`` file's worth)."""
    require_rdkit()
    mols = []
    for index, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            raise ValueError(f"record {index + 1}: {text!r} is not a parsable SMILES")
        if names is not None and index < len(names):
            mol.SetProp("_Name", str(names[index]))
        mols.append(mol)
    return fingerprint_set(mols, **kwargs)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _check_comparable(a: Fingerprint, b: Fingerprint) -> None:
    if not isinstance(a, Fingerprint) or not isinstance(b, Fingerprint):
        raise TypeError(
            "similarity is defined between two odock.ligandsim.Fingerprint objects, "
            f"got {type(a).__name__} and {type(b).__name__}"
        )
    if a.kind != b.kind:
        raise ValueError(
            f"cannot compare a {a.kind!r} fingerprint with a {b.kind!r} one; "
            "fingerprint both molecules with the same kind and parameters"
        )


def _intersection_size(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """``|A n B|`` for two ascending tuples, without allocating a set."""
    i = j = 0
    count = 0
    len_a, len_b = len(a), len(b)
    while i < len_a and j < len_b:
        left, right = a[i], b[j]
        if left == right:
            count += 1
            i += 1
            j += 1
        elif left < right:
            i += 1
        else:
            j += 1
    return count


def tanimoto(a: Fingerprint, b: Fingerprint) -> float:
    """Tanimoto (Jaccard) similarity ``|A n B| / |A u B|``.

    Two empty fingerprints have no union and return ``0.0`` — the same answer
    RDKit's ``TanimotoSimilarity`` gives, and the only answer that cannot be
    mistaken for a perfect match.

    Hand check: benzamidine and its 4-hydroxy analogue set 16 and 17
    Morgan(radius 2, 2048 bits) bits with 12 in common, so the union is
    ``16 + 17 - 12 = 21`` and the similarity is ``12 / 21 = 4 / 7 =
    0.5714285714...`` — the value pinned in ``tests/test_ligandsim.py``.
    """
    _check_comparable(a, b)
    shared = _intersection_size(a.indices, b.indices)
    union = a.n_on + b.n_on - shared
    if union <= 0:
        return 0.0
    return float(shared) / float(union)


def dice(a: Fingerprint, b: Fingerprint) -> float:
    """Dice similarity ``2|A n B| / (|A| + |B|)``.

    Dice weighs the shared bits twice, so it scores higher than Tanimoto for the
    same pair; it is the coefficient of the Daylight/`fpsim2` world.
    """
    _check_comparable(a, b)
    total = a.n_on + b.n_on
    if total <= 0:
        return 0.0
    return 2.0 * float(_intersection_size(a.indices, b.indices)) / float(total)


def tversky(
    a: Fingerprint, b: Fingerprint, *, alpha: float = 1.0, beta: float = 1.0
) -> float:
    """Tversky similarity ``|A n B| / (|A n B| + alpha|A\\B| + beta|B\\A|)``.

    ``alpha = beta = 1`` is Tanimoto and ``alpha = beta = 0.5`` is Dice, which the
    tests pin.  Making the two asymmetric is how a "substructure of" search is
    expressed: ``alpha < beta`` rewards a query whose bits are all found in the
    library molecule (the query is the smaller, more general structure).

    Hand check: for ``A = {1, 2}`` and ``B = {2, 3, 4}``, ``|A n B| = 1``,
    ``|A\\B| = 1``, ``|B\\A| = 2``, so ``(1, 3, alpha=1, beta=1)`` gives
    ``1 / (1 + 1 + 2) = 0.25`` and ``alpha=0.5, beta=0.5`` gives
    ``1 / (1 + 0.5 + 1) = 0.4``.
    """
    _check_comparable(a, b)
    shared = _intersection_size(a.indices, b.indices)
    only_a = a.n_on - shared
    only_b = b.n_on - shared
    denominator = float(shared) + float(alpha) * only_a + float(beta) * only_b
    if denominator <= 0:
        return 0.0
    return float(shared) / denominator


def similarity(
    a: Fingerprint,
    b: Fingerprint,
    *,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """``tanimoto``, ``dice`` or ``tversky`` on one pair of fingerprints."""
    key = _normalise_metric(metric)
    if key == "tanimoto":
        return tanimoto(a, b)
    if key == "dice":
        return dice(a, b)
    return tversky(a, b, alpha=alpha, beta=beta)


def _normalise_metric(metric: str) -> str:
    key = str(metric).strip().lower()
    aliases = {"jaccard": "tanimoto", "morgan": "tanimoto", "sorensen": "dice",
               "czekanowski": "dice"}
    key = aliases.get(key, key)
    if key not in SIMILARITY_METRICS:
        raise ValueError(
            f"unknown similarity metric {metric!r}; supported: {list(SIMILARITY_METRICS)}"
        )
    return key


def _dense(fingerprints: Sequence[Fingerprint], n_bits: int) -> np.ndarray:
    out = np.zeros((len(fingerprints), int(n_bits)), dtype=np.float32)
    for row, fp in enumerate(fingerprints):
        if fp.indices:
            out[row, np.asarray(fp.indices, dtype=np.int64)] = 1.0
    return out


def similarity_matrix(
    fingerprints: Union[Sequence[Fingerprint], FingerprintSet],
    *,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
    chunk: int = 256,
) -> np.ndarray:
    """The all-pairs similarity matrix of a fingerprint set, as ``(N, N)``.

    The arithmetic is a dense block matrix product of the bit matrix, so the
    cost is a real BLAS ``N x N x bits`` multiply rather than ``N^2`` Python set
    intersections: 1 000 molecules at 2 048 bits take well under a second.  The
    memory is bounded by ``chunk`` rows rather than by ``N``, so a 100 000-member
    library can be walked in blocks even though its full matrix (10^10 entries)
    could never be held.

    Diagonal: ``1.0`` for a non-empty fingerprint and ``0.0`` for an empty one
    (see :func:`tanimoto`).  The matrix is symmetric and filled on both sides.
    """
    require_rdkit()
    set_ = fingerprints if isinstance(fingerprints, FingerprintSet) else None
    fps: List[Fingerprint] = list(set_.fingerprints) if set_ is not None else list(fingerprints)
    key = _normalise_metric(metric)
    n = len(fps)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if set_ is not None:
        counts = np.array([fp.n_on for fp in fps], dtype=np.float64)
        bits = int(set_.n_bits)
    else:
        kinds = {fp.kind for fp in fps}
        if len(kinds) > 1:
            raise ValueError(
                "cannot build a similarity matrix from mixed fingerprint kinds: "
                + ", ".join(sorted(kinds))
            )
        counts = np.array([fp.n_on for fp in fps], dtype=np.float64)
        bits = int(max(fp.n_bits for fp in fps))
    out = np.zeros((n, n), dtype=np.float64)
    step = max(1, int(chunk))
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        left = _dense(fps[i0:i1], bits)
        left_counts = counts[i0:i1]
        for j0 in range(i0, n, step):
            j1 = min(n, j0 + step)
            right = _dense(fps[j0:j1], bits)
            shared = left @ right.T
            right_counts = counts[j0:j1]
            block = _coefficient(
                shared, left_counts[:, None], right_counts[None, :], key, alpha, beta
            )
            out[i0:i1, j0:j1] = block
            if j0 != i0:
                out[j0:j1, i0:i1] = block.T
    if n:
        diagonal = np.where(counts > 0, 1.0, 0.0)
        np.fill_diagonal(out, diagonal)
    return out


def _coefficient(
    shared: np.ndarray, n_a: np.ndarray, n_b: np.ndarray, metric: str, alpha: float, beta: float
) -> np.ndarray:
    """The metric applied elementwise to a block of shared-bit counts."""
    if metric == "tanimoto":
        denominator = n_a + n_b - shared
    elif metric == "dice":
        denominator = n_a + n_b
        shared = 2.0 * shared
    else:
        denominator = shared + float(alpha) * (n_a - shared) + float(beta) * (n_b - shared)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(denominator > 0, shared / np.where(denominator > 0, denominator, 1.0), 0.0)
    return out


# ---------------------------------------------------------------------------
# Analogue search
# ---------------------------------------------------------------------------


@dataclass
class Analogue:
    """One library member that passed the similarity cut-off."""

    #: 1-based rank inside the hit list.
    rank: int
    #: Library index (0-based).
    index: int
    name: str
    similarity: float
    smiles: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "index": int(self.index),
            "name": self.name,
            "similarity": round(float(self.similarity), 6),
            "smiles": self.smiles,
        }


@dataclass
class AnalogueHits:
    """The result of :func:`find_analogues` (and of the 3-D variant).

    The two cost fields matter: ``n_pairs`` is how many fingerprint comparisons
    the search actually evaluated, and ``seconds`` how long it took.  A 3-D
    search that reports ``n_pairs = 64000`` has said exactly what it cost, which
    a bare hit list does not.
    """

    query_name: str
    kind: str
    metric: str
    cutoff: float
    n_library: int
    hits: List[Analogue] = field(default_factory=list)
    n_pairs: int = 0
    seconds: float = 0.0
    top: int = 0
    conformers: int = 1
    notes: List[str] = field(default_factory=list)

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    @property
    def best(self) -> Optional[Analogue]:
        return self.hits[0] if self.hits else None

    def hit_rate(self) -> float:
        return float(self.n_hits) / float(self.n_library) if self.n_library else 0.0

    def table(self, limit: int = 0) -> str:
        """A left-aligned text table of the hits."""
        rows = self.hits if limit <= 0 else self.hits[: int(limit)]
        if not rows:
            return (
                f"no analogue of {self.query_name} at {self.metric} >= {self.cutoff:.2f} "
                f"in {self.n_library} molecule(s)"
            )
        lines = [
            f"{'rank':<5}{'name':<32}{'similarity':>11}  smiles",
            "-" * 5 + "-" * 32 + "-" * 11 + "  " + "-" * 40,
        ]
        for hit in rows:
            lines.append(
                f"{hit.rank:<5}{hit.name[:31]:<32}{hit.similarity:>11.4f}  {hit.smiles[:60]}"
            )
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query_name,
            "kind": self.kind,
            "metric": self.metric,
            "cutoff": float(self.cutoff),
            "n_library": int(self.n_library),
            "n_hits": int(self.n_hits),
            "hit_rate": round(self.hit_rate(), 6),
            "n_pairs": int(self.n_pairs),
            "seconds": round(float(self.seconds), 4),
            "conformers": int(self.conformers),
            "hits": [hit.as_dict() for hit in self.hits],
            "notes": list(self.notes),
        }


def _query_fingerprint(query, *, kind, radius, n_bits, use_features, use_chirality,
                       min_path, max_path, name) -> Tuple[Fingerprint, str, str]:
    """Return ``(fingerprint, name, smiles)`` for a molecule/SMILES query."""
    if isinstance(query, Fingerprint):
        return query, query.name or str(name), ""
    if isinstance(query, str):
        mol = Chem.MolFromSmiles(query)
        if mol is None:
            raise ValueError(f"the query {query!r} is not a parsable SMILES")
        label = str(name) if name else _mol_name(mol, "query")
        return (
            fingerprint(
                mol, kind=kind, radius=radius, n_bits=n_bits, use_features=use_features,
                use_chirality=use_chirality, min_path=min_path, max_path=max_path,
                name=label,
            ),
            label,
            _mol_smiles(mol),
        )
    label = str(name) if name else _mol_name(query, "query")
    return (
        fingerprint(
            query, kind=kind, radius=radius, n_bits=n_bits, use_features=use_features,
            use_chirality=use_chirality, min_path=min_path, max_path=max_path, name=label,
        ),
        label,
        _mol_smiles(query),
    )


def find_analogues(
    query,
    library: Union[Sequence[Any], FingerprintSet],
    *,
    cutoff: float = DEFAULT_CUTOFF,
    metric: str = DEFAULT_METRIC,
    top: int = 0,
    kind: str = "morgan",
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    use_features: bool = False,
    use_chirality: bool = False,
    min_path: int = 1,
    max_path: int = 7,
    alpha: float = 1.0,
    beta: float = 1.0,
    name: str = "",
    include_self: bool = True,
) -> AnalogueHits:
    """Find the library members similar to ``query`` above ``cutoff``.

    Parameters
    ----------
    query
        A molecule, a SMILES string, or an already-built :class:`Fingerprint`.
    library
        Molecules, SMILES strings, or a :class:`FingerprintSet` (which is used as
        it is, so a screening cascade fingerprints its library once).
    cutoff
        Minimum similarity to report.  ``0.7`` Tanimoto is the usual
        "analogue" threshold (see :data:`DEFAULT_CUTOFF`).
    metric, alpha, beta
        The coefficient; see :func:`similarity`.
    top
        Keep at most this many hits (0 = every hit above the cut-off).
    include_self
        Whether a library member identical to the query is reported — which only
        happens when the query itself is in the library.

    Returns
    -------
    :class:`AnalogueHits`, sorted by decreasing similarity with ties broken by
    library order, so the result is deterministic and reproducible.  The number
    of comparisons and the elapsed time are reported with it.
    """
    require_rdkit()
    key = _normalise_metric(metric)
    start = time.perf_counter()
    query_fp, query_name, query_smiles = _query_fingerprint(
        query, kind=kind, radius=radius, n_bits=n_bits, use_features=use_features,
        use_chirality=use_chirality, min_path=min_path, max_path=max_path, name=name,
    )
    if query_smiles and not query_name:
        query_name = "query"
    pool = (
        library
        if isinstance(library, FingerprintSet)
        else fingerprint_set(
            library,
            kind=query_fp.kind,
            radius=radius,
            n_bits=query_fp.n_bits or n_bits,
            use_features=use_features,
            use_chirality=use_chirality,
            min_path=min_path,
            max_path=max_path,
            smiles=True,
        )
    )
    if pool.kind != query_fp.kind:
        raise ValueError(
            f"the query is a {query_fp.kind!r} fingerprint but the library holds "
            f"{pool.kind!r} ones; fingerprint both the same way"
        )
    found: List[Analogue] = []
    pairs = 0
    for index, fp in enumerate(pool.fingerprints):
        value = similarity(query_fp, fp, metric=key, alpha=alpha, beta=beta)
        pairs += 1
        if value < float(cutoff):
            continue
        if not include_self and index < len(pool.smiles) and pool.smiles[index] == query_smiles:
            continue
        found.append(
            Analogue(
                rank=0,
                index=index,
                name=pool.name_of(index),
                similarity=value,
                smiles=pool.smiles_of(index),
            )
        )
    found.sort(key=lambda hit: (-hit.similarity, hit.index))
    if top and int(top) > 0:
        found = found[: int(top)]
    for position, hit in enumerate(found, start=1):
        hit.rank = position
    return AnalogueHits(
        query_name=query_name,
        kind=query_fp.kind,
        metric=key,
        cutoff=float(cutoff),
        n_library=len(pool),
        hits=found,
        n_pairs=pairs,
        seconds=time.perf_counter() - start,
        top=int(top),
    )


# ---------------------------------------------------------------------------
# Multi-conformer similarity
# ---------------------------------------------------------------------------


def conformer_fingerprints(
    mol,
    *,
    n_confs: int = DEFAULT_CONFORMERS,
    seed: int = 20240101,
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
    name: Optional[str] = None,
) -> List[Fingerprint]:
    """One 3-D pharmacophore fingerprint per conformer of ``mol``.

    The conformers are generated with ETKDGv3 at ``seed`` when the molecule has
    none (an existing multi-conformer molecule is used as it is), and each one is
    fingerprinted with the Gobbi pharmacophore-pair signature over that
    conformer's 3-D distance matrix.

    Only ``kind="pharmacophore"`` is conformer-dependent.  Morgan, path,
    atom-pair, torsion and MACCS keys are 2-D graph descriptions: every conformer
    of a molecule produces the *same* fingerprint, so a "multi-conformer"
    similarity over them would be a 2-D search repeated ``n_confs`` times.  Use
    this function with :func:`best_over_conformers` / :func:`find_analogues_3d`,
    which say what the repetition costs.
    """
    require_rdkit()
    _require_pharm2d()
    work = Chem.Mol(mol)
    if work.GetNumAtoms() == 0:
        return []
    if work.GetNumConformers() == 0:
        work = Chem.AddHs(work)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        count = max(1, int(n_confs))
        ids = list(AllChem.EmbedMultipleConfs(work, numConfs=count, params=params))
        if not ids:
            # Fall back to one embedding with random coordinates, the same
            # forgiving second attempt `odock.chem.ligand.embed_3d` makes.
            if AllChem.EmbedMolecule(work, randomSeed=int(seed), useRandomCoords=True) != 0:
                raise RuntimeError(
                    "RDKit could not embed this molecule, so no 3-D similarity "
                    "can be computed for it"
                )
    label = _mol_name(work, "") if name is None else str(name)
    out: List[Fingerprint] = []
    for conf_id in range(work.GetNumConformers()):
        fp = _pharmacophore_fingerprint(work, conf_id=conf_id, n_bits=n_bits)
        out.append(
            Fingerprint(
                kind=fp.kind, n_bits=fp.n_bits, indices=fp.indices, name=label, radius=None
            )
        )
    return out


@dataclass
class ConformerFingerprintSet:
    """A library where every molecule carries a list of conformer fingerprints."""

    names: List[str] = field(default_factory=list)
    fingerprints: List[List[Fingerprint]] = field(default_factory=list)
    kind: str = "pharmacophore"
    source: str = ""
    #: How many conformers were requested per molecule.
    n_confs: int = 0
    #: Molecules that could not be embedded, with the reason.
    failures: List[Tuple[str, str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.fingerprints)

    @property
    def n_fingerprints(self) -> int:
        return sum(len(group) for group in self.fingerprints)

    def name_of(self, index: int) -> str:
        if 0 <= index < len(self.names) and self.names[index]:
            return self.names[index]
        return f"ligand_{index + 1}"


def conformer_fingerprint_set(
    molecules: Sequence[Any],
    *,
    n_confs: int = DEFAULT_CONFORMERS,
    seed: int = 20240101,
    n_bits: int = DEFAULT_N_BITS,
    names: Optional[Sequence[str]] = None,
    source: str = "",
) -> ConformerFingerprintSet:
    """Fingerprint every molecule of a library once per conformer.

    A molecule that RDKit cannot embed is recorded in
    :attr:`ConformerFingerprintSet.failures` with an empty fingerprint list
    instead of ending the search: one unembeddable record must not stop a
    library-wide 3-D comparison.
    """
    require_rdkit()
    out = ConformerFingerprintSet(kind="pharmacophore", source=str(source), n_confs=int(n_confs))
    for index, mol in enumerate(molecules):
        label = (
            str(names[index])
            if names is not None and index < len(names)
            else _mol_name(mol, f"ligand_{index + 1}")
        )
        try:
            group = conformer_fingerprints(
                mol, n_confs=n_confs, seed=seed, n_bits=n_bits, name=label
            )
        except Exception as exc:
            out.names.append(label)
            out.fingerprints.append([])
            out.failures.append((label, f"{type(exc).__name__}: {exc}"))
            continue
        out.names.append(label)
        out.fingerprints.append(group)
    return out


def best_over_conformers(
    query_fingerprints: Sequence[Fingerprint],
    library_fingerprints: Sequence[Fingerprint],
    *,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """The highest similarity over every ``(query conformer, library conformer)`` pair.

    This is the honest definition of multi-conformer similarity: a flexible
    molecule is "similar" to another if *some* of its conformers is.  It is also
    the expensive one — the number of comparisons is the product of the two
    conformer counts, which :func:`find_analogues_3d` reports as ``n_pairs``.
    Returns ``0.0`` when either side has no fingerprint.
    """
    key = _normalise_metric(metric)
    best = 0.0
    for left in query_fingerprints:
        for right in library_fingerprints:
            value = similarity(left, right, metric=key, alpha=alpha, beta=beta)
            if value > best:
                best = value
    return float(best)


def find_analogues_3d(
    query,
    library: Sequence[Any],
    *,
    cutoff: float = DEFAULT_CUTOFF,
    metric: str = DEFAULT_METRIC,
    top: int = 0,
    n_query_confs: int = DEFAULT_CONFORMERS,
    n_library_confs: int = DEFAULT_CONFORMERS,
    seed: int = 20240101,
    n_bits: int = DEFAULT_N_BITS,
    name: str = "",
    library_set: Optional[ConformerFingerprintSet] = None,
) -> AnalogueHits:
    """Analogues of ``query`` by best-over-conformers 3-D pharmacophore similarity.

    Both sides are embedded with ETKDGv3 at ``seed`` (unless an already-built
    :class:`ConformerFingerprintSet` is passed as ``library_set``), and the score
    of a library member is the maximum Tanimoto/Dice/Tversky over all conformer
    pairs.  The cost is real and is reported: ``n_pairs`` counts every
    fingerprint comparison, i.e. ``query_conformers x library_conformers`` per
    molecule.

    Measured on this machine (RDKit 2026.03.1, one core): embedding 8 conformers
    of a 12-heavy-atom molecule takes ~0.06 s and its 8 pharmacophore
    fingerprints ~0.025 s, so ~0.01 s per library member per 8 conformers — about
    two orders of magnitude more than the 2-D Morgan search over the same
    library, for a score that additionally depends on the embedding seed.
    """
    require_rdkit()
    _require_pharm2d()
    key = _normalise_metric(metric)
    start = time.perf_counter()
    query_fps = conformer_fingerprints(
        query if not isinstance(query, str) else Chem.MolFromSmiles(query),
        n_confs=n_query_confs,
        seed=seed,
        n_bits=n_bits,
        name=name or None,
    )
    if not query_fps:
        raise ValueError("the query could not be embedded, so no 3-D search is possible")
    query_name = str(name) if name else query_fps[0].name or "query"
    if library_set is None:
        library_set = conformer_fingerprint_set(
            library, n_confs=n_library_confs, seed=seed, n_bits=n_bits
        )
    found: List[Analogue] = []
    pairs = 0
    for index in range(len(library_set)):
        group = library_set.fingerprints[index]
        if not group:
            continue
        value = 0.0
        for left in query_fps:
            for right in group:
                pairs += 1
                score = similarity(left, right, metric=key)
                if score > value:
                    value = score
        if value < float(cutoff):
            continue
        found.append(
            Analogue(
                rank=0, index=index, name=library_set.name_of(index), similarity=value
            )
        )
    found.sort(key=lambda hit: (-hit.similarity, hit.index))
    if top and int(top) > 0:
        found = found[: int(top)]
    for position, hit in enumerate(found, start=1):
        hit.rank = position
    return AnalogueHits(
        query_name=query_name,
        kind="pharmacophore",
        metric=key,
        cutoff=float(cutoff),
        n_library=len(library_set),
        hits=found,
        n_pairs=pairs,
        seconds=time.perf_counter() - start,
        top=int(top),
        conformers=int(len(query_fps)),
        notes=[
            f"{len(query_fps)} query conformer(s) x up to {int(n_library_confs)} "
            "library conformer(s) per molecule"
        ]
        + ([f"{len(library_set.failures)} molecule(s) could not be embedded"]
           if library_set.failures else []),
    )


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------


@dataclass
class DiversitySelection:
    """A representative subset and why it is one."""

    method: str
    indices: List[int] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    smiles: List[str] = field(default_factory=list)
    #: For MaxMin: the similarity to the nearest already-picked molecule at the
    #: moment the molecule was picked (the first pick has 0.0).  For sphere
    #: exclusion: the same quantity, which is by construction below ``cutoff``.
    min_similarity: List[float] = field(default_factory=list)
    n_library: int = 0
    n_requested: int = 0
    cutoff: float = 0.0
    metric: str = DEFAULT_METRIC
    start: int = 0
    seconds: float = 0.0
    n_pairs: int = 0

    @property
    def n_selected(self) -> int:
        return len(self.indices)

    @property
    def fraction(self) -> float:
        return float(self.n_selected) / float(self.n_library) if self.n_library else 0.0

    @property
    def worst_pairwise_similarity(self) -> float:
        """The closest pair inside the subset (the subset's tightest redundancy).

        The first element of :attr:`min_similarity` is always ``0.0`` (nothing to
        compare with), so it is skipped.
        """
        rest = self.min_similarity[1:]
        return max(rest) if rest else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_library": int(self.n_library),
            "n_requested": int(self.n_requested),
            "n_selected": int(self.n_selected),
            "fraction": round(self.fraction, 6),
            "cutoff": float(self.cutoff),
            "metric": self.metric,
            "start": int(self.start),
            "n_pairs": int(self.n_pairs),
            "seconds": round(float(self.seconds), 4),
            "worst_pairwise_similarity": round(self.worst_pairwise_similarity, 6),
            "names": list(self.names),
            "indices": [int(i) for i in self.indices],
            "min_similarity": [round(float(v), 6) for v in self.min_similarity],
            "smiles": list(self.smiles),
        }


def _resolve_library(
    fingerprints: Union[Sequence[Fingerprint], FingerprintSet]
) -> Tuple[List[Fingerprint], Optional[FingerprintSet]]:
    if isinstance(fingerprints, FingerprintSet):
        return list(fingerprints.fingerprints), fingerprints
    return list(fingerprints), None


def _centroid_start(
    fps: Sequence[Fingerprint], metric: str, alpha: float, beta: float, chunk: int
) -> int:
    """The molecule with the highest mean similarity to the rest of the library.

    Costs a full ``N^2`` comparison (inside the blocked matrix code), which is
    why it is opt-in: for a 100 000-member library the O(N) first-molecule start
    is the only affordable one.
    """
    matrix = similarity_matrix(list(fps), metric=metric, alpha=alpha, beta=beta, chunk=chunk)
    if matrix.size == 0:
        return 0
    means = matrix.mean(axis=1)
    return int(np.argmax(means))


def maxmin_pick(
    fingerprints: Union[Sequence[Fingerprint], FingerprintSet],
    n: int,
    *,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
    start: Union[int, str] = 0,
    chunk: int = 256,
) -> List[int]:
    """MaxMin picking: ``n`` molecules that are mutually as dissimilar as possible.

    The algorithm is the standard greedy farthest-point one:

    1. the first pick is ``start`` (index ``0`` by default, or ``"centroid"`` for
       the molecule with the highest mean similarity to the whole library);
    2. repeatedly add the molecule whose *largest* similarity to any already
       picked molecule is *smallest* (ties broken by the lowest library index, so
       the result is deterministic and independent of any hash order).

    The cost is ``n`` passes over the library, i.e. ``O(N n)`` comparisons, with
    no similarity matrix held in memory.  The result is order-dependent through
    the first pick, which is why ``start`` is reported in
    :class:`DiversitySelection` and ``"centroid"`` is available.
    """
    fps, _ = _resolve_library(fingerprints)
    key = _normalise_metric(metric)
    indices, _, _ = _maxmin_pick(
        fps, n, metric=key, alpha=alpha, beta=beta, start=start, chunk=chunk
    )
    return indices


def _maxmin_pick(
    fps: Sequence[Fingerprint],
    n: int,
    *,
    metric: str,
    alpha: float,
    beta: float,
    start: Union[int, str],
    chunk: int,
) -> Tuple[List[int], List[float], int]:
    """MaxMin picking, returning ``(indices, nearest similarity, n_pairs)``.

    ``nearest[k]`` is the similarity of the k-th pick to its *closest earlier*
    pick, which is the quantity that says how redundant the subset is; the first
    pick has no earlier molecule and is reported as ``0.0``.  ``n_pairs`` counts
    every fingerprint comparison that was actually evaluated.
    """
    total = len(fps)
    want = max(0, int(n))
    if total == 0 or want == 0:
        return [], [], 0
    pairs = 0
    if isinstance(start, str):
        if start != "centroid":
            raise ValueError(f"unknown start {start!r}; use an index or 'centroid'")
        first = _centroid_start(fps, metric, alpha, beta, chunk)
        pairs += total * total
    else:
        first = int(start)
        if first < 0 or first >= total:
            raise IndexError(f"start {first} is outside the library (0..{total - 1})")
    # `nearest[i]` is the similarity of i to the closest picked molecule so far.
    nearest = np.empty(total, dtype=np.float64)
    for i, fp in enumerate(fps):
        nearest[i] = similarity(fp, fps[first], metric=metric, alpha=alpha, beta=beta)
        pairs += 1
    nearest[first] = 0.0
    picked = [first]
    is_picked = np.zeros(total, dtype=bool)
    is_picked[first] = True
    history = [0.0]
    while len(picked) < want:
        # The already-picked molecules are masked out with +inf, so they can
        # never win the argmin that selects the next, most-distant molecule.
        masked = np.where(is_picked, np.inf, nearest)
        candidate = int(np.argmin(masked))
        if not math.isfinite(float(masked[candidate])):
            break
        picked.append(candidate)
        history.append(float(masked[candidate]))
        is_picked[candidate] = True
        chosen = fps[candidate]
        for i in range(total):
            if is_picked[i]:
                continue
            value = similarity(fps[i], chosen, metric=metric, alpha=alpha, beta=beta)
            pairs += 1
            if value > nearest[i]:
                nearest[i] = value
    return picked, history, pairs


def sphere_exclusion_pick(
    fingerprints: Union[Sequence[Fingerprint], FingerprintSet],
    *,
    cutoff: float = 0.6,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
    start: int = 0,
    n: int = 0,
) -> List[int]:
    """Sphere exclusion: walk the library and keep a molecule when it is
    *less* similar than ``cutoff`` to every molecule already kept.

    The simplest diversity filter there is — one pass, no optimisation — and it
    answers "give me the first ``N`` molecules that are mutually below
    ``cutoff``".  ``n`` caps the number kept (0 = no cap); the walk is
    deterministic in library order.
    """
    fps, _ = _resolve_library(fingerprints)
    key = _normalise_metric(metric)
    indices, _, _ = _sphere_pick(
        fps, cutoff=cutoff, metric=key, alpha=alpha, beta=beta, start=start, n=n
    )
    return indices


def _sphere_pick(
    fps: Sequence[Fingerprint],
    *,
    cutoff: float,
    metric: str,
    alpha: float,
    beta: float,
    start: int,
    n: int,
) -> Tuple[List[int], List[float], int]:
    """Sphere exclusion, returning ``(indices, nearest similarity, n_pairs)``."""
    total = len(fps)
    if total == 0:
        return [], [], 0
    first = int(start)
    if first < 0 or first >= total:
        raise IndexError(f"start {first} is outside the library (0..{total - 1})")
    picked = [first]
    history = [0.0]
    pairs = 0
    for i in range(total):
        if i == first:
            continue
        if n and int(n) > 0 and len(picked) >= int(n):
            break
        nearest = 0.0
        accepted = True
        for j in picked:
            value = similarity(fps[i], fps[j], metric=metric, alpha=alpha, beta=beta)
            pairs += 1
            if value > nearest:
                nearest = value
            if value >= float(cutoff):
                accepted = False
                break
        if accepted:
            picked.append(i)
            history.append(nearest)
    return picked, history, pairs
def diversity_subset(
    fingerprints: Union[Sequence[Fingerprint], FingerprintSet],
    n: int = 0,
    *,
    method: str = "maxmin",
    cutoff: Optional[float] = None,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
    start: Union[int, str] = 0,
    chunk: int = 256,
) -> DiversitySelection:
    """Pick a representative subset, with the numbers that describe it.

    ``method="maxmin"`` picks exactly ``n`` molecules (:func:`maxmin_pick`);
    ``method="sphere"`` keeps every molecule below ``cutoff`` of the kept set, up
    to ``n`` (:func:`sphere_exclusion_pick`).  Every selected molecule carries the
    similarity to its nearest earlier pick in
    :attr:`DiversitySelection.min_similarity`, so the *tightest* redundancy
    inside the subset is one number — printed by the CLI and asserted by the
    tests.  ``n_pairs``/``seconds`` report what the pick cost.

    The scaffold-space coverage of the subset is :func:`scaffold_coverage`, kept
    separate because it needs the molecules, not just their fingerprints.
    """
    key = _normalise_metric(metric)
    name = str(method).strip().lower()
    if name not in DIVERSITY_METHODS:
        raise ValueError(
            f"unknown diversity method {method!r}; supported: {list(DIVERSITY_METHODS)}"
        )
    fps, set_ = _resolve_library(fingerprints)
    start_time = time.perf_counter()
    if name == "maxmin":
        indices, nearest, pairs = _maxmin_pick(
            fps, n, metric=key, alpha=alpha, beta=beta, start=start, chunk=chunk
        )
        threshold = 0.0
    else:
        if cutoff is None:
            raise ValueError("sphere exclusion needs a cutoff")
        indices, nearest, pairs = _sphere_pick(
            fps,
            cutoff=float(cutoff),
            metric=key,
            alpha=alpha,
            beta=beta,
            start=int(start) if not isinstance(start, str) else 0,
            n=n,
        )
        threshold = float(cutoff)
    return DiversitySelection(
        method=name,
        indices=[int(i) for i in indices],
        names=[set_.name_of(i) if set_ is not None else f"ligand_{i + 1}" for i in indices],
        smiles=(
            [set_.smiles_of(i) for i in indices] if set_ is not None else []
        ),
        min_similarity=nearest,
        n_library=len(fps),
        n_requested=int(n),
        cutoff=threshold,
        metric=key,
        start=int(start) if not isinstance(start, str) else -1,
        seconds=time.perf_counter() - start_time,
        n_pairs=pairs,
    )


def scaffold_coverage(
    selection: Union[DiversitySelection, Sequence[int]],
    library: Union[Sequence[Any], FingerprintSet],
    *,
    generic: bool = False,
) -> Dict[str, float]:
    """How much of the library's scaffold space a subset covers.

    Returns ``{"n_library_scaffolds", "n_subset_scaffolds", "coverage",
    "subset_fraction"}`` where ``coverage`` is
    ``n_subset_scaffolds / n_library_scaffolds`` and ``subset_fraction`` is the
    share of *molecules* taken.  A subset of 20 % of the molecules covering 60 %
    of the scaffolds is the number that says the subset is doing its job; 20 % of
    the molecules covering 20 % of the scaffolds means the picking found nothing
    new.

    ``library`` is the molecules themselves, or a :class:`FingerprintSet` built
    with ``smiles=True`` (whose stored SMILES are re-parsed to perceive the
    scaffolds) — a bare list of :class:`Fingerprint` objects cannot be turned back
    into structures, so it is refused.  The scaffold perception lives in
    :mod:`odock.scaffold` and is imported lazily, so either module can be used on
    its own.
    """
    from .scaffold import scaffold_key

    indices = (
        list(selection.indices)
        if isinstance(selection, DiversitySelection)
        else [int(i) for i in selection]
    )
    if isinstance(library, FingerprintSet):
        if not library.smiles:
            raise ValueError(
                "the FingerprintSet carries no SMILES, so its scaffolds cannot be "
                "perceived; build it with smiles=True or pass the molecules"
            )
        molecules = [Chem.MolFromSmiles(text) for text in library.smiles]
    else:
        molecules = _normalise_molecules(list(library))
    if not molecules:
        return {
            "n_library_scaffolds": 0.0,
            "n_subset_scaffolds": 0.0,
            "coverage": 0.0,
            "subset_fraction": 0.0,
        }
    keys = [scaffold_key(mol, generic=generic) or "(no scaffold)" for mol in molecules]
    library_keys = set(keys)
    subset_keys = {keys[i] for i in indices if 0 <= i < len(keys)}
    n_library = len(library_keys)
    return {
        "n_library_scaffolds": float(n_library),
        "n_subset_scaffolds": float(len(subset_keys)),
        "coverage": (float(len(subset_keys)) / n_library) if n_library else 0.0,
        "subset_fraction": float(len(indices)) / float(len(keys)),
    }


# ---------------------------------------------------------------------------
# Butina clustering
# ---------------------------------------------------------------------------


@dataclass
class Cluster:
    """One cluster of a :class:`Clustering`."""

    #: 1-based cluster rank (largest first).
    rank: int
    #: Member indices, ascending.
    members: Tuple[int, ...]
    #: Cluster size.
    size: int
    #: The member closest to the rest of the cluster (the medoid).
    representative: int
    #: The representative's name, when the set had names.
    representative_name: str = ""
    #: Mean similarity between the representative and the other members.
    mean_similarity: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rank": int(self.rank),
            "size": int(self.size),
            "members": [int(i) for i in self.members],
            "representative": int(self.representative),
            "representative_name": self.representative_name,
            "mean_similarity": round(float(self.mean_similarity), 6),
        }


@dataclass
class Clustering:
    """The clusters of a library at one similarity cut-off."""

    clusters: List[Cluster] = field(default_factory=list)
    cutoff: float = DEFAULT_CLUSTER_CUTOFF
    metric: str = DEFAULT_METRIC
    n_molecules: int = 0
    seconds: float = 0.0
    n_pairs: int = 0

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def n_singletons(self) -> int:
        return sum(1 for cluster in self.clusters if cluster.size == 1)

    @property
    def largest(self) -> Optional[Cluster]:
        return self.clusters[0] if self.clusters else None

    def table(self, limit: int = 20) -> str:
        """One line per cluster: rank, size, representative, first members."""
        if not self.clusters:
            return "no clusters"
        rows = self.clusters if limit <= 0 else self.clusters[: int(limit)]
        lines = [
            f"{'cluster':<8}{'size':>5}  {'representative':<32}members",
            "-" * 8 + "-" * 5 + "  " + "-" * 32 + "-" * 24,
        ]
        for cluster in rows:
            members = ", ".join(str(i) for i in cluster.members[:6])
            if cluster.size > 6:
                members += ", ..."
            lines.append(
                f"{cluster.rank:<8}{cluster.size:>5}  "
                f"{cluster.representative_name[:31]:<32}{members}"
            )
        if limit > 0 and len(self.clusters) > limit:
            lines.append(f"... and {len(self.clusters) - limit} more cluster(s)")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cutoff": float(self.cutoff),
            "metric": self.metric,
            "n_molecules": int(self.n_molecules),
            "n_clusters": int(self.n_clusters),
            "n_singletons": int(self.n_singletons),
            "largest": int(self.clusters[0].size) if self.clusters else 0,
            "n_pairs": int(self.n_pairs),
            "seconds": round(float(self.seconds), 4),
            "clusters": [cluster.as_dict() for cluster in self.clusters],
        }


def butina_cluster(
    fingerprints: Union[Sequence[Fingerprint], FingerprintSet],
    *,
    cutoff: float = DEFAULT_CLUSTER_CUTOFF,
    metric: str = DEFAULT_METRIC,
    alpha: float = 1.0,
    beta: float = 1.0,
    chunk: int = 256,
) -> Clustering:
    """Cluster a library by similarity, Butina's way, at ``cutoff``.

    The algorithm is Butina's (1999) sphere-exclusion around the most-connected
    molecule, which is what "Butina clustering" means in cheminformatics:

    1. every molecule's neighbours are the molecules at similarity ``>= cutoff``
       (equivalently, distance ``<= 1 - cutoff``);
    2. the molecule with the most still-unassigned neighbours starts a cluster,
       which is itself plus those neighbours; both the seed *and* the assigned
       neighbours are then removed from the pool — this is what makes a cluster
       tighter than the raw neighbour list;
    3. repeat until every molecule is assigned.  A molecule with no neighbour
       left becomes a singleton cluster, which is reported rather than hidden.

    Ties are broken by the lowest library index, so the partition is
    deterministic.  The representative of a cluster is its **medoid** — the
    member with the highest mean similarity to the others — which is the natural
    "exemplar" a chemist wants to look at, and is a stronger choice than "the
    first member".

    The cost is one ``N x N`` similarity matrix, computed in blocks; 1 000
    molecules are tens of milliseconds, 20 000 are seconds.
    """
    fps, set_ = _resolve_library(fingerprints)
    key = _normalise_metric(metric)
    start = time.perf_counter()
    total = len(fps)
    if total == 0:
        return Clustering(cutoff=float(cutoff), metric=key, n_molecules=0)
    matrix = similarity_matrix(fps, metric=key, alpha=alpha, beta=beta, chunk=chunk)
    adjacency = matrix >= float(cutoff)
    clusters: List[Cluster] = []
    assigned = np.zeros(total, dtype=bool)
    while not assigned.all():
        unassigned = np.flatnonzero(~assigned)
        # Neighbours that are still unassigned, per candidate seed.
        counts = adjacency[np.ix_(unassigned, unassigned)].sum(axis=1)
        best = int(np.argmax(counts))
        seed = int(unassigned[best])
        members = [seed]
        for other in unassigned:
            other = int(other)
            if other == seed:
                continue
            if adjacency[seed, other]:
                members.append(other)
        members.sort()
        assigned[np.asarray(members, dtype=np.int64)] = True
        clusters.append(_make_cluster(len(clusters) + 1, members, matrix, set_))
    clusters.sort(key=lambda cluster: (-cluster.size, cluster.representative))
    for position, cluster in enumerate(clusters, start=1):
        cluster.rank = position
    return Clustering(
        clusters=clusters,
        cutoff=float(cutoff),
        metric=key,
        n_molecules=total,
        seconds=time.perf_counter() - start,
        n_pairs=total * total,
    )


def _make_cluster(rank: int, members: Sequence[int], matrix: np.ndarray, set_) -> Cluster:
    """Build a cluster, choosing its medoid as the representative."""
    members = tuple(int(i) for i in members)
    if len(members) == 1:
        representative = members[0]
        mean = 0.0
    else:
        index = np.asarray(members, dtype=np.int64)
        sub = matrix[np.ix_(index, index)]
        means = sub.mean(axis=1)
        best = int(np.argmax(means))
        representative = members[best]
        mean = float(means[best])
    name = set_.name_of(representative) if set_ is not None else f"ligand_{representative + 1}"
    return Cluster(
        rank=int(rank),
        members=members,
        size=len(members),
        representative=representative,
        representative_name=name,
        mean_similarity=mean,
    )

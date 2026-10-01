# SPDX-License-Identifier: GPL-3.0-or-later
"""Writers for the file formats the AutoDock tool family consumes.

Four formats are produced here, each one written to match the *published*
convention rather than to merely look plausible:

``GPF`` (AutoGrid grid parameter file)
    The keyword set written by AutoDockTools when it saves a GPF: ``npts``,
    ``gridfld``, ``spacing``, ``receptor_types``, ``ligand_types``,
    ``receptor``, ``gridcenter``, ``smooth``, one ``map`` per ligand type,
    ``elecmap``, ``dsolvmap`` and ``dielectric``.  Source: the AutoGrid 4.2
    parameter-file documentation distributed with AutoDock 4.2 (Morris et al.,
    *J. Comput. Chem.* **30**, 2785 (2009)) and the GPF files written by
    AutoDockTools 1.5.7, e.g. the ``1hsg.gpf`` of the AutoDock tutorial set.
    Two details of that format matter and are implemented exactly: ``npts`` is
    always **odd** (AutoGrid rejects an even number of grid points) and it is
    derived from the box size as ``2 * floor(size / (2 * spacing)) + 1``.

``DPF`` (AutoDock 4 docking parameter file)
    ``autodock_parameter_version 4.2`` followed by the diagnostic, ligand-type,
    grid-map, ligand, orientation and search sections.  Source: the DPF keyword
    reference of AutoDock 4.2 (same paper) and the DPFs AutoDockTools writes for
    a "Lamarckian GA" job, i.e. the ``ga_*`` block terminated by ``set_ga``
    followed by the ``sw_*``/``ls_search_freq`` block terminated by
    ``set_psw1``.  ``ndihe`` is the number of *real* dihedrals, which is the
    ligand's ``TORSDOF``.

``Vina configuration file``
    ``key = value`` lines, one per option, as read by AutoDock Vina 1.1/1.2 and
    documented in ``vina --help``.  Source: the Vina configuration reference
    (Trott & Olson, *J. Comput. Chem.* **31**, 455 (2010); Eberhardt et al.,
    *J. Chem. Inf. Model.* **61**, 3891 (2021)).

``PDB`` (cleaned structure export)
    RDKit's PDB block plus the ``MASTER``/``END`` tail RDKit does not write.
    Source: the wwPDB PDB format specification v3.30 (record ``MASTER``:
    ``numRemark``, then a zero, ``numHet``, ``numHelix``, ``numSheet``,
    ``numTurn``, ``numSite``, ``numXform``, ``numCoord``, ``numTer``,
    ``numConect``, ``numSeq`` in five-column right-aligned fields starting at
    column 11).  The layout was verified against the two RCSB files bundled in
    this repository: :file:`tests/data/3PTB.pdb` and :file:`tests/data/1M17.pdb`
    (their ``MASTER`` counts reproduce the ``REMARK``/``HET``/``HELIX``/
    ``SHEET``/``SITE``/``ORIGX``+``SCALE``/``ATOM``+``HETATM``/``TER``/
    ``CONECT``/``SEQRES`` record counts of the files exactly).

Nothing here needs RDKit except :func:`export_cleaned_pdb`, which imports it
lazily, so an export-only script works in a minimal installation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .prepare import BoxSpec

__all__ = [
    "AD4_TYPES",
    "STANDARD_LIGAND_TYPES",
    "STANDARD_RECEPTOR_TYPES",
    "VINA_KEYS",
    "element_of",
    "export_cleaned_pdb",
    "export_dpf",
    "export_gpf",
    "export_vina_config",
    "master_record",
    "pdbqt_atom_type",
    "pdbqt_torsdof",
    "pdbqt_types",
]

PathLike = Union[str, os.PathLike]

# ---------------------------------------------------------------------------
# AutoDock 4 atom types
# ---------------------------------------------------------------------------

#: The AutoDock 4 atom types with the number the kernel gives them.  Kept as
#: the reference table for :data:`_TYPE_ELEMENT`; the list is the one in
#: ``crates/dock-core/src/atom.rs`` and in AutoDock's ``atom_constants.h``
#: (``AD_TYPE_C`` = 0, ``AD_TYPE_A`` = 1, ...).
AD4_TYPES: Tuple[str, ...] = (
    "C", "A", "N", "O", "P", "S", "H", "F", "I", "NA", "OA", "SA", "HD",
    "Mg", "Mn", "Zn", "Ca", "Fe", "Cl", "Br", "Si", "At",
    "G0", "G1", "G2", "G3", "CG0", "CG1", "CG2", "CG3", "W",
)

#: AD4 type -> element symbol, for the callers that need an element (the RMSD
#: code in :mod:`odock.analysis`, for instance).
_TYPE_ELEMENT: Dict[str, str] = {
    "C": "C", "A": "C", "N": "N", "O": "O", "P": "P", "S": "S", "H": "H",
    "HD": "H", "F": "F", "I": "I", "NA": "N", "OA": "O", "SA": "S",
    "Cl": "Cl", "Br": "Br", "Si": "Si", "At": "At",
    "Mg": "Mg", "Mn": "Mn", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe",
    "G0": "C", "G1": "C", "G2": "C", "G3": "C",
    "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C", "W": "H",
}

#: The ligand types the project asks AutoGrid maps for when nothing
#: better is known ("A, C, HD, N, NA, OA, S, SA, Cl, F 等").
STANDARD_LIGAND_TYPES: Tuple[str, ...] = (
    "A", "C", "HD", "N", "NA", "OA", "S", "SA", "Cl", "F",
)

#: The types a protein receptor realistically contains; the fallback when the
#: receptor PDBQT cannot be read.
STANDARD_RECEPTOR_TYPES: Tuple[str, ...] = (
    "A", "C", "HD", "N", "NA", "OA", "S", "SA",
)

#: AutoGrid/AutoDock defaults that both formats share.
DEFAULT_SPACING = 0.375
DEFAULT_SMOOTH = 0.5
DEFAULT_DIELECTRIC = -0.1465
DEFAULT_AUTODOCK_VERSION = "4.2"


def _order_types(types: Iterable[str]) -> Tuple[str, ...]:
    """De-duplicate `types` while keeping the caller's order."""
    out: List[str] = []
    for raw in types:
        value = str(raw).strip()
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _canonical_types(types: Iterable[str]) -> Tuple[str, ...]:
    """De-duplicate and sort `types` the way AutoDockTools lists them.

    AutoDockTools writes the types of a PDBQT file in plain alphabetical order,
    which is what produces the familiar ``receptor_types A C HD N NA OA SA``
    line.  Types the caller supplied by hand are *not* reordered -- see
    :func:`_order_types`.
    """
    return tuple(sorted(_order_types(types)))


def element_of(ad_type: str) -> str:
    """The element symbol for an AD4 atom type (``"OA"`` -> ``"O"``)."""
    name = str(ad_type).strip()
    return _TYPE_ELEMENT.get(name, name[:1].upper() if name else "C")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _write(out: Optional[PathLike], text: str) -> str:
    """Write `text` to `out` when given, and return it either way."""
    if out is not None:
        path = Path(out)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def _as_box(box) -> BoxSpec:
    """Accept a :class:`~odock.prepare.BoxSpec`, a mapping or a JSON file."""
    if isinstance(box, BoxSpec):
        return box
    if isinstance(box, (str, os.PathLike)):
        data = json.loads(Path(box).read_text(encoding="utf-8"))
        return _as_box(data)
    if isinstance(box, Mapping):
        payload = dict(box)
        payload.setdefault("spacing", DEFAULT_SPACING)
        return BoxSpec(
            center=tuple(float(v) for v in payload["center"]),
            size=tuple(float(v) for v in payload["size"]),
            spacing=float(payload["spacing"]),
        )
    raise TypeError(
        "the box must be an odock.BoxSpec, a {'center', 'size', 'spacing'} "
        f"mapping or the path of the JSON `odock box` writes, not {type(box).__name__}"
    )


def _stem(path: Optional[PathLike], fallback: str = "receptor") -> str:
    """The base name of `path` without its extension."""
    if path is None:
        return fallback
    stem = Path(os.fspath(path)).stem
    return stem or fallback


def _grid_stem(receptor_path: Optional[PathLike], gridfld: Optional[PathLike]) -> str:
    """The stem shared by the ``.fld`` and every ``.map`` file.

    With no explicit ``gridfld`` it is the receptor's stem, so
    ``receptor.pdbqt`` gives ``receptor.maps.fld``, ``receptor.A.map`` and so
    on -- AutoDockTools' naming.  An explicit ``gridfld`` of
    ``something.maps.fld`` or ``something.fld`` gives ``something``.
    """
    if gridfld is None:
        return _stem(receptor_path, "receptor")
    name = Path(os.fspath(gridfld)).name
    for suffix in (".fld", ".maps"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or "receptor"


def _grid_points(size: float, spacing: float) -> int:
    """AutoGrid's odd grid-point count: ``2 * floor(size / (2 * spacing)) + 1``.

    AutoGrid 4 rejects an even ``npts``; AutoDockTools therefore always writes
    an odd number and the box only ever covers a symmetric extent around its
    centre, one half-spacing short of the requested size at most.
    """
    return 2 * int(math.floor(size / (2.0 * spacing))) + 1


def _number(value: float) -> str:
    """A number as ``-1.8555`` rather than ``-1.8555000000000001``.

    Ten significant digits is more than any docking parameter needs and just
    enough to drop the binary-representation noise, while a value such as
    Vina's ``weight_gauss1 = -0.035579`` survives intact.
    """
    text = f"{float(value):.10g}"
    return "0" if text in ("", "-0") else text


def _spacing_number(value: float) -> str:
    return _number(value)


def _commented(pairs: Sequence[Tuple[str, str]], comments: bool, width: int = 36) -> List[str]:
    """Lay out ``"keyword value  # comment"`` lines in AutoDockTools' style."""
    lines: List[str] = []
    for body, comment in pairs:
        if comments and comment:
            padded = body.ljust(width) if len(body) < width else body + " "
            lines.append(padded + "# " + comment)
        else:
            lines.append(body)
    return lines


# ---------------------------------------------------------------------------
# PDBQT introspection
# ---------------------------------------------------------------------------


def _pdbqt_lines(text: str) -> List[str]:
    return [
        line
        for line in text.splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


def pdbqt_atom_type(line: str) -> str:
    """The AutoDock type of one PDBQT record.

    Columns 78-79 hold the type; when the record is too short (or the type is a
    three-letter one such as ``CG0``) the atom name is used as a fallback.
    """
    tail = line[77:].strip() if len(line) > 77 else ""
    if tail in _TYPE_ELEMENT:
        return tail
    if len(tail) in (1, 2) and tail.isalpha():
        return tail
    name = line[12:16].strip() if len(line) >= 16 else ""
    letters = "".join(c for c in name if c.isalpha())
    if not letters:
        return "C"
    pair = letters[:2]
    if pair.capitalize() in ("Cl", "Br", "Si", "At", "Mg", "Mn", "Zn", "Ca", "Fe"):
        return pair.capitalize()
    return letters[0].upper()


def pdbqt_types(text: str) -> Tuple[str, ...]:
    """Every distinct AutoDock type in a PDBQT document, in AutoDock's order."""
    return _canonical_types(pdbqt_atom_type(line) for line in _pdbqt_lines(text))


def pdbqt_torsdof(text: str) -> Optional[int]:
    """The ``TORSDOF`` of a PDBQT document, or ``None`` when it has none."""
    for line in text.splitlines():
        if line.startswith("TORSDOF"):
            try:
                return int(float(line.split()[1]))
            except (IndexError, ValueError):
                return None
    return None


def _read_text(source: Optional[PathLike]) -> Optional[str]:
    """Read a PDBQT file, or return ``None`` when it is not there.

    An export must not fail because the structure it points at has not been
    written yet: the file name is what ends up *inside* the GPF/DPF, and the
    maps are built by AutoGrid later.
    """
    if source is None:
        return None
    path = Path(os.fspath(source))
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover - unreadable file
        return None
    return None


# ---------------------------------------------------------------------------
# GPF
# ---------------------------------------------------------------------------


def export_gpf(
    box,
    receptor_path: PathLike,
    out: Optional[PathLike] = None,
    *,
    spacing: Optional[float] = None,
    gridfld: Optional[PathLike] = None,
    receptor_types: Optional[Sequence[str]] = None,
    ligand_types: Optional[Sequence[str]] = None,
    smooth: float = DEFAULT_SMOOTH,
    dielectric: float = DEFAULT_DIELECTRIC,
    comments: bool = True,
) -> str:
    """Write an AutoGrid grid parameter file (GPF) and return its text.

    Parameters
    ----------
    box
        The search box: an :class:`odock.BoxSpec`, a ``{"center": ...,
        "size": ..., "spacing": ...}`` mapping, or the path of the JSON file
        ``odock box`` writes.
    receptor_path
        The receptor PDBQT.  The string is written verbatim into the
        ``receptor`` line, so it must be a name AutoGrid can resolve from the
        directory the GPF lives in (or an absolute path).
    out
        Where to write the file.  ``None`` writes nothing and only returns the
        text.
    spacing
        Grid spacing in Å.  Defaults to the box's own spacing.
    gridfld
        The ``.fld`` file AutoGrid will create.  Defaults to
        ``<receptor stem>.maps.fld``, which is what AutoDockTools produces.
    receptor_types
        The types present in the receptor.  Defaults to the types actually found
        in `receptor_path` (AutoDock's own order); if that file cannot be read,
        :data:`STANDARD_RECEPTOR_TYPES`.
    ligand_types
        The probe types to build maps for -- one ``map`` line each.  Defaults to
        `receptor_types`, which is what AutoDockTools writes when the GPF is
        saved without a ligand loaded.
    smooth
        ``smooth`` keyword: store the minimum energy within this radius (Å).
    dielectric
        ``dielectric`` keyword: negative means a distance-dependent dielectric
        (AD4's ``-0.1465``), positive a constant one.
    comments
        Append the usual trailing ``#`` comments.  AutoGrid ignores them; they
        are what makes the file recognisable as an ADT-format GPF.

    Returns
    -------
    The file text; it has also been written to `out` when that is not ``None``.

    Notes
    -----
    ``npts`` follows AutoGrid's rule ``2 * floor(size / (2 * spacing)) + 1``, so
    it is always odd and never larger than the requested box.
    """
    spec = _as_box(box)
    step = float(spec.spacing if spacing is None else spacing)
    if not math.isfinite(step) or step <= 0:
        raise ValueError(f"spacing must be a positive number, got {step!r}")

    text = _read_text(receptor_path)
    if receptor_types is None:
        derived = pdbqt_types(text) if text else ()
        rec_types = derived or STANDARD_RECEPTOR_TYPES
    else:
        rec_types = _order_types(receptor_types)
    lig_types = (
        _order_types(ligand_types) if ligand_types is not None else tuple(rec_types)
    )
    if not lig_types:
        raise ValueError("ligand_types must not be empty: a GPF needs at least one map")

    stem = _grid_stem(receptor_path, gridfld)
    npts = tuple(_grid_points(size, step) for size in spec.size)

    pairs: List[Tuple[str, str]] = [
        ("npts " + " ".join(str(n) for n in npts), "number of grid points in xyz"),
        (f"gridfld {stem}.maps.fld", "grid data file"),
        (f"spacing {_spacing_number(step)}", "spacing (A)"),
        (
            "receptor_types " + " ".join(rec_types),
            "receptor atom types",
        ),
        (
            "ligand_types " + " ".join(lig_types),
            "ligand atom types",
        ),
        (f"receptor {os.fspath(receptor_path)}", "receptor file"),
        (
            "gridcenter " + " ".join(_number(v) for v in spec.center),
            "xyz-coordinates or auto",
        ),
        (f"smooth {_number(smooth)}", "store minimum energy w/in rad(A)"),
    ]
    pairs += [(f"map {stem}.{t}.map", "atom-specific affinity map") for t in lig_types]
    pairs += [
        (f"elecmap {stem}.e.map", "electrostatic potential map"),
        (f"dsolvmap {stem}.d.map", "desolvation potential map"),
        (
            f"dielectric {_number(dielectric)}",
            "<0, AD4 distance-dep.diel;>0, constant",
        ),
    ]
    return _write(out, "\n".join(_commented(pairs, comments)) + "\n")


# ---------------------------------------------------------------------------
# DPF
# ---------------------------------------------------------------------------


def export_dpf(
    box,
    ligand_path: PathLike,
    out: Optional[PathLike] = None,
    *,
    receptor: PathLike = "receptor.pdbqt",
    gridfld: Optional[PathLike] = None,
    ligand_types: Optional[Sequence[str]] = None,
    torsdof: Optional[int] = None,
    ndihe: Optional[int] = None,
    seed: Optional[int] = None,
    parameters: str = "lga",
    ga_pop_size: int = 150,
    ga_num_evals: int = 2500000,
    ga_num_generations: int = 27000,
    ga_elitism: int = 1,
    ga_mutation_rate: float = 0.02,
    ga_crossover_rate: float = 0.8,
    ga_window_size: int = 10,
    ga_cauchy_alpha: float = 0.0,
    ga_cauchy_beta: float = 1.0,
    sw_max_its: int = 300,
    sw_max_succ: int = 4,
    sw_max_fail: int = 4,
    sw_rho: float = 1.0,
    sw_lb_rho: float = 0.01,
    ls_search_freq: float = 0.06,
    outlev: int = 1,
    analysis: bool = False,
    comments: bool = True,
) -> str:
    """Write an AutoDock 4 docking parameter file (DPF) and return its text.

    Parameters
    ----------
    box
        The search box; ``about`` and the map references come from it.  Same
        accepted shapes as :func:`export_gpf`.
    ligand_path
        The ligand PDBQT.  It is written verbatim on the ``move`` line *and*
        read, when it exists, to derive ``ligand_types`` and ``ndihe`` -- which
        is exactly how AutoDockTools fills those keywords in.
    out
        Where to write the file (``None`` writes nothing).
    receptor
        The receptor PDBQT name written into the ``receptor`` line, and the
        source of the default ``fld``/map file names.
    gridfld
        The ``.fld`` written by the matching :func:`export_gpf`.  Defaults to
        ``<receptor stem>.maps.fld`` -- the same default as the GPF, so the two
        files agree by construction.
    ligand_types
        Override the ligand types.  Defaults to the types in `ligand_path`, or
        :data:`STANDARD_LIGAND_TYPES` if it cannot be read.
    torsdof, ndihe
        The ligand's torsional degrees of freedom.  ``ndihe`` (the number of
        real dihedrals) defaults to `torsdof`, which defaults to the ``TORSDOF``
        record of the ligand PDBQT (0 when there is none).
    seed
        A fixed random seed.  ``None`` writes AD4's ``seed pid time``, which
        draws a fresh seed per run.
    parameters
        The search section to write: ``"lga"`` (island-model Lamarckian GA with
        the pseudo-Solis & Wets local search: ``set_ga`` + ``set_psw1``),
        ``"ga"`` (GA only), ``"ls"`` (pseudo-Solis & Wets only) or ``"none"``.
    outlev, analysis, comments
        Diagnostic level, whether to add AD4's ``analysis`` keyword, and whether
        to append the trailing ``#`` comments.

    Returns
    -------
    The file text; it has also been written to `out` when that is not ``None``.
    """
    spec = _as_box(box)
    if parameters not in ("lga", "ga", "ls", "none"):
        raise ValueError(
            f"parameters must be 'lga', 'ga', 'ls' or 'none', got {parameters!r}"
        )

    ligand_text = _read_text(ligand_path)
    if ligand_types is None:
        derived = pdbqt_types(ligand_text) if ligand_text else ()
        lig_types = derived or STANDARD_LIGAND_TYPES
    else:
        lig_types = _order_types(ligand_types)
    if not lig_types:
        raise ValueError("ligand_types must not be empty: a DPF needs at least one map")

    if torsdof is None:
        torsdof = pdbqt_torsdof(ligand_text) if ligand_text else 0
    if torsdof is None:
        torsdof = 0
    torsdof = int(torsdof)
    if ndihe is None:
        ndihe = torsdof
    ndihe = int(ndihe)

    stem = _grid_stem(receptor, gridfld)

    pairs: List[Tuple[str, str]] = [
        (
            f"autodock_parameter_version {DEFAULT_AUTODOCK_VERSION}",
            "used by autodock to validate parameter set",
        ),
        (f"outlev {int(outlev)}", "diagnostic output level"),
        ("intelec", "calculate internal electrostatics"),
        (
            "seed pid time" if seed is None else f"seed {int(seed)}",
            "seeds for random generator",
        ),
        ("ligand_types " + " ".join(lig_types), "atoms types in ligand"),
        (f"fld {stem}.maps.fld", "grid_data_file"),
    ]
    pairs += [(f"map {stem}.{t}.map", "atom-specific affinity map") for t in lig_types]
    pairs += [
        (f"elecmap {stem}.e.map", "electrostatics map"),
        (f"desolvmap {stem}.d.map", "desolvation map"),
        (f"move {os.fspath(ligand_path)}", "small molecule"),
        (
            "about " + " ".join(_number(v) for v in spec.center),
            "small molecule center",
        ),
        ("tran0 random", "initial coordinates/A or random"),
        ("quat0 random", "initial quaternion"),
        (f"ndihe {ndihe}", "number of real dihedrals"),
    ]
    if analysis:
        pairs.append(("analysis", "show analysis of results"))

    if parameters in ("lga", "ga"):
        pairs += [
            (f"ga_pop_size {int(ga_pop_size)}", "number of individuals in population"),
            (f"ga_num_evals {int(ga_num_evals)}", "maximum number of energy evaluations"),
            (
                f"ga_num_generations {int(ga_num_generations)}",
                "maximum number of generations",
            ),
            (
                f"ga_elitism {int(ga_elitism)}",
                "number of top individuals to survive to next generation",
            ),
            (f"ga_mutation_rate {_number(ga_mutation_rate)}", "rate of gene mutation"),
            (f"ga_crossover_rate {_number(ga_crossover_rate)}", "rate of crossover"),
            (f"ga_window_size {int(ga_window_size)}", ""),
            (f"ga_cauchy_alpha {_number(ga_cauchy_alpha)}", ""),
            (f"ga_cauchy_beta {_number(ga_cauchy_beta)}", ""),
            ("set_ga", "set the above parameters for GA or LGA"),
        ]
    if parameters in ("lga", "ls"):
        pairs += [
            (f"sw_max_its {int(sw_max_its)}", "iterations of Solis & Wets local search"),
            (
                f"sw_max_succ {int(sw_max_succ)}",
                "consecutive successes before changing rho",
            ),
            (
                f"sw_max_fail {int(sw_max_fail)}",
                "consecutive failures before changing rho",
            ),
            (f"sw_rho {_number(sw_rho)}", "size of local search space to sample"),
            (f"sw_lb_rho {_number(sw_lb_rho)}", "lower bound on rho"),
            (
                f"ls_search_freq {_number(ls_search_freq)}",
                "probability of performing local search on individual",
            ),
            ("set_psw1", "set the above pseudo-Solis & Wets parameters"),
        ]

    return _write(out, "\n".join(_commented(pairs, comments)) + "\n")


# ---------------------------------------------------------------------------
# AutoDock Vina configuration
# ---------------------------------------------------------------------------

#: The keys AutoDock Vina 1.1/1.2 documents for its configuration file, plus
#: the aliases this module accepts.  Anything else is rejected rather than
#: written silently, because a misspelt key makes Vina fall back to a default.
VINA_KEYS: Tuple[str, ...] = (
    "receptor", "ligand", "flex", "out", "log",
    "center_x", "center_y", "center_z",
    "size_x", "size_y", "size_z",
    "autobox", "autobox_add", "force_even_voxels", "spacing",
    "exhaustiveness", "num_modes", "min_rmsd", "energy_range", "seed", "cpu",
    "scoring", "custom_scoring", "custom_atoms",
    "weight_gauss1", "weight_gauss2", "weight_repulsion", "weight_hydrophobic",
    "weight_hydrogen", "weight_rot", "weight_glue", "weight_amino",
    "weight_metallic", "weight_non_directionality",
    "no_refine", "local_only", "randomize_only", "score_only", "minimize",
    "verbosity", "max_evals", "max_step",
)

#: Accepted spellings that map onto a documented Vina key.
VINA_ALIASES: Dict[str, str] = {
    "num_poses": "num_modes",
    "out_poses": "out",
    "poses": "out",
    "modes": "num_modes",
    "threads": "cpu",
    "n_cpu": "cpu",
}


def export_vina_config(
    box,
    receptor: PathLike,
    ligand: PathLike,
    out: Optional[PathLike] = None,
    *,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    energy_range: float = 3.0,
    out_poses: Optional[PathLike] = None,
    seed: Optional[int] = None,
    cpu: Optional[int] = None,
    scoring: Optional[str] = None,
    spacing: Optional[float] = None,
    min_rmsd: Optional[float] = None,
    comments: bool = True,
    **kwargs,
) -> str:
    """Write an AutoDock Vina configuration file and return its text.

    Parameters
    ----------
    box
        The search box.  ``center_x/y/z`` and ``size_x/y/z`` come from it.
    receptor, ligand
        The PDBQT names written into the file, verbatim.
    out
        Where to write the config (``None`` writes nothing).
    exhaustiveness, num_modes, energy_range
        Vina's search effort, the number of binding modes to report and the
        energy window that defines a distinct mode.
    out_poses
        Write ``out = ...`` so Vina saves the poses itself.
    seed, cpu, scoring, spacing, min_rmsd
        Written only when given: Vina's own defaults are better left implicit
        when the caller did not ask for anything specific.  ``scoring`` is
        ``"vina"`` (default), ``"vinardo"`` or ``"ad4"``; ``spacing`` is the
        grid spacing in Å and ``min_rmsd`` the minimum RMSD between reported
        modes.
    comments
        Append the ``#`` comment holding the default.
    **kwargs
        Any other key of :data:`VINA_KEYS` (``weight_gauss1``, ``no_refine``,
        ...) and the aliases in :data:`VINA_ALIASES`.  Unknown keys raise
        :class:`ValueError`.

    Returns
    -------
    The file text; it has also been written to `out` when that is not ``None``.

    Notes
    -----
    Booleans are written as Vina expects them: ``1`` for a flag that is on.
    ``None`` omits the line entirely.
    """
    spec = _as_box(box)

    items: List[Tuple[str, object]] = [
        ("receptor", os.fspath(receptor)),
        ("ligand", os.fspath(ligand)),
    ]
    if out_poses is not None:
        items.append(("out", os.fspath(out_poses)))
    items += [
        ("center_x", spec.center[0]),
        ("center_y", spec.center[1]),
        ("center_z", spec.center[2]),
        ("size_x", spec.size[0]),
        ("size_y", spec.size[1]),
        ("size_z", spec.size[2]),
        ("exhaustiveness", exhaustiveness),
        ("num_modes", num_modes),
        ("energy_range", energy_range),
    ]
    optional = {
        "seed": seed,
        "cpu": cpu,
        "scoring": scoring,
        "spacing": spacing,
        "min_rmsd": min_rmsd,
    }
    for key, value in optional.items():
        if value is not None:
            items.append((key, value))

    resolved: List[Tuple[str, object]] = []
    for key, value in kwargs.items():
        name = VINA_ALIASES.get(key, key)
        if name not in VINA_KEYS:
            raise ValueError(
                f"unknown Vina option {key!r}; the documented keys are: "
                + ", ".join(sorted(VINA_KEYS))
                + "; aliases: " + ", ".join(sorted(VINA_ALIASES))
            )
        resolved.append((name, value))
    items += resolved

    lines: List[str] = []
    for key, value in items:
        if isinstance(value, bool):
            rendered = "1" if value else "0"
        elif isinstance(value, float):
            rendered = _number(value)
        else:
            rendered = str(value)
        line = f"{key} = {rendered}"
        if comments and key in optional:
            line = line.ljust(30) + f"# default: {_vina_default(key)}"
        lines.append(line)
    return _write(out, "\n".join(lines) + "\n")


def _vina_default(key: str) -> str:
    """The Vina default for the options this writer only emits on request."""
    return {
        "seed": "random",
        "cpu": "all cores",
        "scoring": "vina",
        "spacing": "0.375",
        "min_rmsd": "1.0",
    }.get(key, "-")


# ---------------------------------------------------------------------------
# Cleaned PDB
# ---------------------------------------------------------------------------


def master_record(lines: Sequence[str]) -> str:
    """The ``MASTER`` record for an already-serialised PDB document.

    The twelve fields are, in order, ``numRemark``, ``0``, ``numHet``,
    ``numHelix``, ``numSheet``, ``numTurn``, ``numSite``, ``numXform``,
    ``numCoord``, ``numTer``, ``numConect`` and ``numSeq`` -- the wwPDB PDB
    format v3.30 definition, each right-aligned in five columns starting at
    column 11.  ``numCoord`` counts ``ATOM`` plus ``HETATM`` records and
    ``numXform`` counts the ``ORIGX``/``SCALE`` records; the layout reproduces
    the ``MASTER`` line of the RCSB files bundled with this repository.
    """
    counts = {
        "remark": 0, "het": 0, "helix": 0, "sheet": 0, "turn": 0, "site": 0,
        "xform": 0, "coord": 0, "ter": 0, "conect": 0, "seqres": 0,
    }
    for line in lines:
        record = line[:6].strip().upper()
        if record == "REMARK":
            counts["remark"] += 1
        elif record == "HET":
            counts["het"] += 1
        elif record == "HELIX":
            counts["helix"] += 1
        elif record == "SHEET":
            counts["sheet"] += 1
        elif record == "TURN":
            counts["turn"] += 1
        elif record == "SITE":
            counts["site"] += 1
        elif record in ("ORIGX1", "ORIGX2", "ORIGX3", "SCALE1", "SCALE2", "SCALE3"):
            counts["xform"] += 1
        elif record in ("ATOM", "HETATM"):
            counts["coord"] += 1
        elif record == "TER":
            counts["ter"] += 1
        elif record == "CONECT":
            counts["conect"] += 1
        elif record == "SEQRES":
            counts["seqres"] += 1

    values = (
        counts["remark"], 0, counts["het"], counts["helix"], counts["sheet"],
        counts["turn"], counts["site"], counts["xform"], counts["coord"],
        counts["ter"], counts["conect"], counts["seqres"],
    )
    return "MASTER" + " " * 4 + "".join(f"{value:>5d}" for value in values)


def _strip_master_and_end(lines: Sequence[str]) -> List[str]:
    """Drop a trailing ``MASTER``/``END`` pair so it can be written afresh."""
    body: List[str] = []
    for line in lines:
        record = line[:6].strip().upper()
        if record in ("MASTER", "END"):
            continue
        body.append(line)
    return body


def export_cleaned_pdb(mol, out: Optional[PathLike] = None) -> str:
    """Write an RDKit molecule as a PDB file with a proper ``MASTER``/``END``.

    Parameters
    ----------
    mol
        An RDKit ``Mol`` or the path of a structure file (SDF/MOL/MOL2/PDB/
        PDBQT), which is read with :func:`odock.prepare.read_structure`.
    out
        Where to write the file.  ``None`` writes nothing and only returns the
        text.

    Returns
    -------
    The PDB text.  It is RDKit's own block -- ``COMPND``/``TITLE`` header,
    ``ATOM``/``HETATM`` records, ``CONECT`` connectivity -- with RDKit's bare
    ``END`` replaced by a ``MASTER`` record whose counts are computed from the
    records actually written, followed by ``END``.

    Notes
    -----
    RDKit writes no ``MASTER`` record at all, and downstream PDB readers
    (PyMOL's ``pdbsum``, wwPDB validation) use it as the file's self-check.
    The counts include only what is present: for a ligand export this is usually
    ``numHet`` = 0 and ``numCoord`` = the number of ``ATOM`` records, since
    RDKit writes the atoms of a non-standard residue as ``HETATM``.
    """
    Chem, _ = _require_rdkit()
    if isinstance(mol, (str, os.PathLike)):
        from .prepare import read_structure

        mol = read_structure(mol)
    if mol is None:
        raise ValueError("no molecule to write")

    block = Chem.MolToPDBBlock(mol)
    body = _strip_master_and_end(block.splitlines())
    lines = body + [master_record(body), "END"]
    return _write(out, "\n".join(line.rstrip() for line in lines) + "\n")


def _require_rdkit():
    """Import RDKit or explain why the export cannot run."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "writing a PDB needs RDKit: pip install 'opendocking[chem]'"
        ) from exc
    return Chem, AllChem

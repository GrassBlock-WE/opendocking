# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``odock`` command-line interface.

Every subcommand is a thin wrapper around the Python API, so anything the CLI
does can also be scripted::

    odock info
    odock prepare receptor receptor.pdb receptor.pdbqt
    odock prepare ligand ligand.sdf ligand.pdbqt --name donepezil
    odock box --ligand ligand.pdbqt --out box.json --buffer 8
    odock pocket -r receptor.pdbqt --json-out pockets.json
    odock filter -i library.sdf -i more.smi
    odock dock -r receptor.pdbqt -l ligand.pdbqt --box box.json -o poses.pdbqt -v
    odock score -r receptor.pdbqt -l poses.pdbqt
    odock cluster -p poses.pdbqt --cutoff 2.0
    odock interactions -r receptor.pdbqt -l ligand.pdbqt
    odock diagram -r receptor.pdbqt -l ligand.pdbqt -o interaction.svg
    odock report -p poses.pdbqt -o report.xlsx
    odock export gpf -r receptor.pdbqt -o receptor.gpf --box box.json
    odock fetch 3PTB -o 3PTB.pdb
    odock gui

The analysis subcommands (``pocket``, ``filter``, ``cluster``, ``interactions``,
``diagram``, ``report``) import the module that implements them only when they
run, so a partially installed package still gives a working ``odock``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .prepare import BoxSpec


def _eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def _lazy(module: str, feature: str):
    """Import an optional part of the package the first time it is needed.

    The analysis pipeline is written in parallel with this file, so a missing
    module must never break the whole command line: the command says what is
    unavailable and exits with status 2 instead of raising ``ImportError``.
    """
    import importlib

    try:
        return importlib.import_module(module, __package__)
    except ImportError as exc:
        _eprint(f"odock: {feature} is unavailable: cannot import {module} ({exc}).")
        _eprint("       this part of the installation is missing or out of date.")
        raise SystemExit(2)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A left-aligned, dash-underlined text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def render(cells: Sequence[str]) -> str:
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def _write_json(path, payload) -> None:
    """Write a machine-readable result and say so on stderr."""
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _eprint(f"wrote {path}")


def _load_box(args) -> BoxSpec:
    """Resolve the search box from the CLI arguments."""
    if getattr(args, "box", None):
        p = Path(args.box)
        data = json.loads(p.read_text(encoding="utf-8"))
        return BoxSpec(
            center=tuple(float(x) for x in data["center"]),
            size=tuple(float(x) for x in data["size"]),
            spacing=float(data.get("spacing", getattr(args, "spacing", 0.375))),
        )
    if getattr(args, "ligand", None) and getattr(args, "box_from_ligand", False):
        from .prepare import read_structure  # noqa: F401
        from .prepare import box_from_ligand

        mol = _read_any(args.ligand)
        return box_from_ligand(mol, buffer=args.buffer, spacing=args.spacing)
    if getattr(args, "center", None) and getattr(args, "size", None):
        return BoxSpec(
            center=tuple(float(x) for x in args.center),
            size=tuple(float(x) for x in args.size),
            spacing=float(args.spacing),
        )
    raise SystemExit(
        "error: no search box: pass --box FILE, or --center X Y Z --size X Y Z, "
        "or --box-from-ligand with --buffer"
    )


def _read_any(path: str):
    """Read a structure, accepting PDBQT pose files as well."""
    from .prepare import read_structure

    return read_structure(path)


def _export_box(args) -> BoxSpec:
    """The search box for the ``export`` subcommands.

    Unlike :func:`_load_box` this leaves the spacing alone when the caller did
    not ask for one, so ``--box box.json`` keeps the spacing recorded in the
    JSON instead of silently replacing it with the command's default.
    """
    default_spacing = 0.375
    if getattr(args, "box", None):
        data = json.loads(Path(args.box).read_text(encoding="utf-8"))
        spacing = args.spacing if args.spacing is not None else data.get("spacing", default_spacing)
        return BoxSpec(
            center=tuple(float(x) for x in data["center"]),
            size=tuple(float(x) for x in data["size"]),
            spacing=float(spacing),
        )
    if args.center and args.size:
        spacing = args.spacing if args.spacing is not None else default_spacing
        return BoxSpec(
            center=tuple(float(x) for x in args.center),
            size=tuple(float(x) for x in args.size),
            spacing=float(spacing),
        )
    if getattr(args, "ligand", None) and getattr(args, "box_from_ligand", False):
        from .prepare import box_from_ligand

        spacing = args.spacing if args.spacing is not None else default_spacing
        return box_from_ligand(_read_any(args.ligand), buffer=args.buffer, spacing=spacing)
    hint = " or --box-from-ligand with --ligand" if getattr(args, "ligand", None) else ""
    raise SystemExit(
        "error: no search box: pass --box FILE, or --center X Y Z --size X Y Z" + hint
    )


#: The ``REMARK VINA RESULT:`` line Vina and this kernel write per pose.
_VINA_RESULT = re.compile(
    r"^REMARK\s+VINA\s+RESULT:\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _split_models(text: str) -> List[str]:
    """The ``MODEL`` blocks of a PDBQT/PDB document, in file order."""
    lines = text.splitlines()
    if not any(line.startswith("MODEL") for line in lines):
        return [text] if text.strip() else []
    blocks: List[str] = []
    current: Optional[List[str]] = None
    for line in lines:
        if line.startswith("MODEL"):
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                blocks.append("\n".join(current) + "\n")
            current = None
            continue
        if current is not None:
            current.append(line)
    if current:
        blocks.append("\n".join(current) + "\n")
    return blocks or [text]


def _read_poses(path) -> List[dict]:
    """Read every pose of a PDBQT pose file.

    One dict per ``MODEL``: ``index``, ``affinity`` (from the
    ``REMARK VINA RESULT:`` line, or ``None``), ``rmsd_lower_bound``,
    ``rmsd_upper_bound``, ``coords`` (an ``(N, 3)`` float64 array), ``elements``
    (one element symbol per atom), ``atom_names`` and ``text`` (the block
    itself).  The coordinates stay in *file* order, which is the order the
    ligand PDBQT was written in and therefore the order
    :mod:`odock.analysis` and :mod:`odock.report` expect.
    """
    import numpy as np

    from .export import element_of, pdbqt_atom_type

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    poses: List[dict] = []
    for block in _split_models(text):
        coords: List[Tuple[float, float, float]] = []
        elements: List[str] = []
        names: List[str] = []
        affinity = lower = upper = None
        for line in block.splitlines():
            if line.startswith("REMARK"):
                match = _VINA_RESULT.match(line)
                if match:
                    affinity, lower, upper = (float(g) for g in match.groups())
                continue
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
            elements.append(element_of(pdbqt_atom_type(line)))
            names.append(line[12:16].strip())
        if not coords:
            continue
        poses.append(
            {
                "index": len(poses),
                "affinity": affinity,
                "rmsd_lower_bound": 0.0 if lower is None else lower,
                "rmsd_upper_bound": 0.0 if upper is None else upper,
                "coords": np.asarray(coords, dtype=float),
                "elements": elements,
                "atom_names": names,
                "text": block,
            }
        )
    return poses


def _atom_label(mol, index: int) -> str:
    """``"ASP189 A OD2"`` for one atom of an RDKit molecule."""
    try:
        atom = mol.GetAtomWithIdx(int(index))
    except Exception:  # pragma: no cover - defensive
        return f"atom {index}"
    symbol = atom.GetSymbol()
    info = atom.GetPDBResidueInfo()
    if info is None:
        return f"{symbol}{index}"
    name = info.GetName().strip() or symbol
    label = f"{info.GetResidueName().strip()}{info.GetResidueNumber()}"
    chain = info.GetChainId().strip()
    if chain:
        label += f" {chain}"
    return f"{label} {name}"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_info(args) -> int:
    """Report the kernel version, force fields and GPU status."""
    import odock

    print(f"OpenDocking {odock.__version__}")
    print(f"  python        : {sys.version.split()[0]} ({sys.executable})")
    try:
        import numpy

        print(f"  numpy         : {numpy.__version__}")
    except Exception:
        print("  numpy         : missing")
    try:
        import rdkit

        print(f"  rdkit         : {rdkit.__version__}")
    except Exception:
        print("  rdkit         : missing (preparation unavailable)")
    for name in ("vina", "vinardo", "ad4"):
        sf = odock._odock.ScoringFunction(name)
        print(
            f"  {name:<13} : cutoff {sf.cutoff:.2f} Å, {sf.num_terms} terms, "
            f"grid={'yes' if sf.grid_capable else 'no'}"
        )
    print(f"  gpu           : {odock.gpu_description()}")
    return 0


def cmd_prepare_receptor(args) -> int:
    """Prepare a rigid receptor PDBQT."""
    from .prepare import prepare_receptor

    mol, text, report = prepare_receptor(
        args.input,
        args.out,
        keep_water=args.keep_water,
        keep_hetero=not args.no_hetero,
        strip=args.strip,
        add_polar_hydrogens=not args.no_hydrogens,
        strict=args.strict,
    )
    _eprint(report.summary())
    for w in report.warnings:
        _eprint(f"  warning: {w}")
    if args.out:
        _eprint(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_prepare_ligand(args) -> int:
    """Prepare a ligand PDBQT, including its rotatable-bond tree."""
    from .prepare import prepare_ligand

    mol, text, report = prepare_ligand(
        args.input,
        args.out,
        name=args.name,
        smiles=args.smiles,
        add_hydrogens=not args.no_hydrogens,
        strip_nonpolar=not args.keep_nonpolar,
        rigid_amides=not args.flexible_amides,
        optimize=not args.no_optimize,
        seed=args.seed,
    )
    _eprint(report.summary())
    for w in report.warnings:
        _eprint(f"  warning: {w}")
    if args.out:
        _eprint(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_box(args) -> int:
    """Build a search box and write it as JSON."""
    from .prepare import box_from_ligand, box_from_selection, read_structure

    if args.ligand:
        mol = _read_any(args.ligand)
        box = box_from_ligand(mol, buffer=args.buffer, spacing=args.spacing)
    elif args.receptor and args.residue is not None:
        mol = _read_any(args.receptor)

        def keep(atom) -> bool:
            info = atom.GetPDBResidueInfo()
            if info is None:
                return False
            if info.GetResidueNumber() != args.residue:
                return False
            return not args.chain or info.GetChainId().strip() == args.chain

        box = box_from_selection(mol, keep, buffer=args.buffer, spacing=args.spacing)
    elif args.receptor:
        mol = _read_any(args.receptor)
        box = box_from_ligand(mol, buffer=args.buffer, spacing=args.spacing)
    else:
        raise SystemExit("error: pass --ligand or --receptor")

    payload = box.as_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _eprint(f"wrote {args.out}")
    print(json.dumps(payload, indent=2))
    return 0


def cmd_dock(args) -> int:
    """Dock a ligand."""
    import inspect

    from .docking import dock

    if args.receptor is None or args.ligand is None:
        raise SystemExit("error: --receptor and --ligand are required")
    box = _load_box(args)
    _eprint(f"box: {box}")

    kwargs = dict(
        scoring=args.scoring,
        exhaustiveness=args.exhaustiveness,
        num_poses=args.num_poses,
        seed=args.seed,
        use_grid=not args.no_grid,
        refine=not args.no_refine,
        min_rmsd=args.min_rmsd,
        energy_range=args.energy_range,
        islands=args.islands,
        population=args.population,
        generations=args.generations,
    )
    # `--search` selects the optimiser by name; a kernel that predates it only
    # knows the `use_island_ga` flag, so fall back to that instead of failing.
    parameters = inspect.signature(dock).parameters
    search = getattr(args, "search", None)
    if "search" in parameters:
        if search is not None:
            kwargs["search"] = search
            if args.ga and search != "lga":
                _eprint(f"note: --ga is implied by --search {search}; --search wins")
        elif args.ga:
            kwargs["search"] = "lga"
    else:
        if search is None:
            kwargs["use_island_ga"] = bool(args.ga)
        else:
            _eprint(
                f"note: this kernel has no search selector; --search {search} "
                "is mapped onto the island-model GA"
            )
            kwargs["use_island_ga"] = search != "monte_carlo"

    result = dock(args.receptor, args.ligand, box, **kwargs)
    chosen = kwargs.get("search") or (
        "lga" if kwargs.get("use_island_ga") else "monte_carlo"
    )
    _eprint(
        f"search: {chosen}, {result.num_movable_atoms} movable atoms, "
        f"{result.num_dof} torsions, N_tors={result.num_tors:g}, "
        f"grid {result.grid_points} points ({result.grid_mb} MB), "
        f"seed={result.seed}, {result.elapsed:.2f}s"
    )
    if args.out:
        Path(args.out).write_text(result.to_pdbqt(), encoding="utf-8")
        _eprint(f"wrote {args.out}")
    if args.json_out:
        payload = {
            "seed": result.seed,
            "elapsed": result.elapsed,
            "box": box.as_dict(),
            "grid_points": result.grid_points,
            "poses": [p.to_dict() for p in result.poses],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _eprint(f"wrote {args.json_out}")
    if args.quiet:
        _eprint(result.table())
    else:
        print(result.table())
        print()
        print(f"best affinity: {result.best_affinity:.3f} kcal/mol")
    return 0


def cmd_score(args) -> int:
    """Score a ligand (or the first model of a pose file) in place."""
    from .docking import score

    box = None
    if args.box or (args.center and args.size) or args.box_from_ligand:
        box = _load_box(args)
    result = score(args.receptor, args.ligand, box, scoring=args.scoring)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"affinity          : {result['affinity']:>10.3f} kcal/mol")
        print(f"  inter           : {result['inter']:>10.3f}")
        print(f"  intra           : {result['intra']:>10.3f}")
        print(f"  conf-independent: {result['conf_independent']:>10.3f}")
        print(f"  unbound         : {result['unbound']:>10.3f}")
    return 0


def cmd_split(args) -> int:
    """Split a multi-model PDBQT file into one file per model."""
    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    count = 0
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            if current:
                count += 1
                (outdir / f"{stem}_{count}.pdbqt").write_text(
                    "\n".join(current) + "\n", encoding="utf-8"
                )
            current = []
        elif line.startswith("ENDMDL"):
            continue
        else:
            current.append(line)
    if current:
        count += 1
        (outdir / f"{stem}_{count}.pdbqt").write_text(
            "\n".join(current) + "\n", encoding="utf-8"
        )
    _eprint(f"wrote {count} models to {outdir}")
    return 0


def cmd_pocket(args) -> int:
    """Detect binding pockets on a ligand-free receptor (blind docking)."""
    pocket = _lazy("odock.pocket", "pocket detection")
    from .prepare import read_structure

    mol = read_structure(args.receptor)
    pockets = pocket.find_pockets(
        mol,
        spacing=args.spacing,
        probe=args.probe,
        min_volume=args.min_volume,
        max_pockets=args.max_pockets,
        buriedness=args.buriedness,
    )
    if pockets:
        rows = []
        for found in pockets:
            centre = "({:8.2f},{:8.2f},{:8.2f})".format(*found.center)
            labels = list(found.residue_labels)
            residues = ", ".join(labels[:6]) or "-"
            if len(labels) > 6:
                residues += f", +{len(labels) - 6}"
            rows.append(
                [
                    str(found.index + 1),
                    centre,
                    f"{found.volume:.1f}",
                    f"{found.score:.2f}",
                    residues,
                ]
            )
        print(_table(["#", "center (x, y, z)", "volume/Å³", "score", "residues"], rows))
    else:
        _eprint(
            f"no cavity of at least {args.min_volume:g} Å³ was found "
            f"at {args.spacing:g} Å spacing; lower --min-volume or try another --spacing"
        )
    if args.json_out:
        _write_json(
            args.json_out,
            {
                "receptor": str(args.receptor),
                "spacing": args.spacing,
                "min_volume": args.min_volume,
                "max_pockets": args.max_pockets,
                "pockets": [
                    {
                        "index": found.index,
                        "center": list(found.center),
                        "volume": found.volume,
                        "score": found.score,
                        "n_points": found.n_points,
                        "residues": list(found.residue_labels),
                    }
                    for found in pockets
                ],
            },
        )
    return 0


def cmd_filter(args) -> int:
    """Report the Lipinski, Veber and PAINS verdicts of one or more files."""
    filters = _lazy("odock.filters", "the drug-likeness filters")
    ligand = _lazy("odock.chem.ligand", "ligand reading")

    mols = []
    for source in args.input:
        try:
            mols.extend(ligand.read_ligands(source, embed=False))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"error: cannot read {source}: {exc}")
    if not mols:
        raise SystemExit("error: no ligand was found in the given files")

    rows = filters.filter_library(mols)
    headers = [
        "name", "MW", "LogP", "HBD", "HBA", "RotB", "tPSA",
        "Lipinski", "Veber", "PAINS", "verdict",
    ]
    table: List[List[str]] = []
    for row in rows:
        props = row["properties"]
        verdicts = {result.name: result.passed for result in row["results"]}

        def mark(name: str) -> str:
            return "pass" if verdicts.get(name, True) else "fail"

        table.append(
            [
                str(row["name"])[:30],
                f"{props.get('MW', float('nan')):.1f}",
                f"{props.get('LogP', float('nan')):.2f}",
                f"{props.get('HBD', 0):.0f}",
                f"{props.get('HBA', 0):.0f}",
                f"{props.get('RotB', 0):.0f}",
                f"{props.get('tPSA', float('nan')):.1f}",
                mark("Lipinski"),
                mark("Veber"),
                mark("PAINS"),
                "PASS" if row["passed"] else "fail",
            ]
        )
    print(_table(headers, table))
    for row in rows:
        if row["violations"]:
            print(f"  {row['name']}: " + "; ".join(str(v) for v in row["violations"]))
    passing = sum(1 for row in rows if row["passed"])
    print()
    print(f"{passing} of {len(rows)} molecule(s) pass every filter")

    if args.json_out:
        _write_json(
            args.json_out,
            {
                "inputs": [str(source) for source in args.input],
                "n_molecules": len(rows),
                "n_passed": passing,
                "molecules": [
                    {
                        "index": row["index"],
                        "name": row["name"],
                        "passed": row["passed"],
                        "properties": row["properties"],
                        "violations": list(row["violations"]),
                        "filters": [result.as_dict() for result in row["results"]],
                    }
                    for row in rows
                ],
            },
        )
    return 0


def _dominant_pose_shape(poses: List[dict]) -> List[dict]:
    """Keep the poses that share the most common atom count.

    A pose file whose models disagree on the number of atoms cannot be
    superposed; rather than failing on the whole run, the odd models out are
    reported and dropped.
    """
    counts = {}
    for pose in poses:
        counts[pose["coords"].shape[0]] = counts.get(pose["coords"].shape[0], 0) + 1
    if len(counts) == 1:
        return poses
    wanted = max(counts, key=lambda n: (counts[n], -n))
    kept = [pose for pose in poses if pose["coords"].shape[0] == wanted]
    _eprint(
        f"warning: the models disagree on the atom count ({sorted(counts)}); "
        f"keeping the {len(kept)} with {wanted} atoms"
    )
    return kept


def cmd_cluster(args) -> int:
    """Cluster the poses of a docking run by symmetry-aware RMSD."""
    analysis = _lazy("odock.analysis", "RMSD clustering")

    poses = _dominant_pose_shape(_read_poses(args.poses))
    if not poses:
        raise SystemExit(f"error: no pose found in {args.poses}")
    for pose in poses[1:]:
        if pose["elements"] != poses[0]["elements"]:
            _eprint("warning: the models do not have identical atom types")
            break

    energies = [pose["affinity"] for pose in poses]
    if any(energy is None for energy in energies):
        energies = None  # type: ignore[assignment]
    clusters = analysis.cluster_poses(
        [pose["coords"] for pose in poses],
        cutoff=args.cutoff,
        elements=poses[0]["elements"],
        energies=energies,
    )

    reference = None
    if args.ligand:
        reference_poses = _read_poses(args.ligand)
        if not reference_poses:
            raise SystemExit(f"error: no structure found in {args.ligand}")
        reference = reference_poses[0]
        if reference["coords"].shape != poses[0]["coords"].shape:
            _eprint(
                "warning: the reference ligand has a different atom count; "
                "the reference RMSD column is omitted"
            )
            reference = None

    rows: List[List[str]] = []
    payload: List[dict] = []
    for cluster in clusters:
        representative = poses[int(cluster.representative)]
        entry = {
            "index": cluster.index,
            "size": len(cluster.members),
            "members": [int(m) + 1 for m in cluster.members],
            "representative": int(cluster.representative) + 1,
            "best_energy": cluster.best_energy,
            "mean_rmsd": cluster.mean_rmsd,
            "reference_rmsd": None,
        }
        cells = [
            str(cluster.index + 1),
            str(len(cluster.members)),
            f"mode {int(cluster.representative) + 1}",
            "-" if cluster.best_energy is None else f"{cluster.best_energy:.3f}",
            f"{cluster.mean_rmsd:.3f}",
        ]
        if reference is not None:
            try:
                value = analysis.symmetry_aware_rmsd(
                    representative["coords"], reference["coords"], poses[0]["elements"]
                )
            except ValueError as exc:
                _eprint(f"warning: cannot compare with the reference ligand: {exc}")
                reference = None
                value = None
            if value is not None:
                entry["reference_rmsd"] = float(value)
                cells.append(f"{value:.3f}")
        rows.append(cells)
        payload.append(entry)

    headers = ["cluster", "size", "representative", "affinity", "mean RMSD/Å"]
    if reference is not None:
        headers.append("ref RMSD/Å")
    print(_table(headers, rows))
    print()
    print(f"{len(poses)} poses in {len(clusters)} cluster(s) at {args.cutoff:g} Å cutoff")

    if args.json_out:
        _write_json(
            args.json_out,
            {
                "poses": str(args.poses),
                "reference": str(args.ligand) if args.ligand else None,
                "cutoff": args.cutoff,
                "n_poses": len(poses),
                "clusters": payload,
            },
        )
    return 0


def cmd_interactions(args) -> int:
    """Profile the non-covalent interactions between a receptor and a ligand."""
    analysis = _lazy("odock.analysis", "interaction profiling")
    from .prepare import read_structure

    receptor = read_structure(args.receptor)
    ligand = read_structure(args.ligand)
    interactions = analysis.profile_interactions(
        receptor,
        ligand,
        hbond=args.hbond,
        salt=args.salt,
        pi=args.pi,
        cation_pi=args.cation_pi,
        hydrophobic=args.hydrophobic,
        clash_ratio=args.clash_ratio,
    )

    counts: dict = {}
    rows: List[List[str]] = []
    for interaction in interactions:
        kind = str(getattr(interaction, "kind", "?"))
        counts[kind] = counts.get(kind, 0) + 1
        detail = str(getattr(interaction, "detail", "") or "")
        subtype = str(getattr(interaction, "subtype", "") or "")
        if subtype and subtype not in detail:
            detail = f"{detail} ({subtype})" if detail else subtype
        rows.append(
            [
                kind,
                _atom_label(receptor, getattr(interaction, "a", -1)),
                _atom_label(ligand, getattr(interaction, "b", -1)),
                f"{float(getattr(interaction, 'distance', float('nan'))):.2f}",
                detail,
            ]
        )
    if rows:
        print(_table(["kind", "receptor", "ligand", "d/Å", "detail"], rows))
        print()
        print("summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    else:
        print("no non-covalent interaction was found between these two structures")

    summary = analysis.interaction_summary(interactions, receptor, ligand)
    print(f"key residues: {summary or '-'}")

    if args.json_out:
        _write_json(
            args.json_out,
            {
                "receptor": str(args.receptor),
                "ligand": str(args.ligand),
                "counts": counts,
                "key_residues": summary,
                "interactions": [
                    {
                        "kind": getattr(interaction, "kind", ""),
                        "receptor_atom": int(getattr(interaction, "a", -1)),
                        "ligand_atom": int(getattr(interaction, "b", -1)),
                        "receptor_label": _atom_label(receptor, getattr(interaction, "a", -1)),
                        "ligand_label": _atom_label(ligand, getattr(interaction, "b", -1)),
                        "distance": float(getattr(interaction, "distance", float("nan"))),
                        "detail": getattr(interaction, "detail", ""),
                        "subtype": getattr(interaction, "subtype", ""),
                    }
                    for interaction in interactions
                ],
            },
        )
    return 0


def cmd_diagram(args) -> int:
    """Draw the 2-D interaction topology diagram as SVG."""
    analysis = _lazy("odock.analysis", "the interaction diagram")
    from .prepare import read_structure

    receptor = read_structure(args.receptor)
    ligand = read_structure(args.ligand)
    interactions = analysis.profile_interactions(
        receptor,
        ligand,
        hbond=args.hbond,
        salt=args.salt,
        pi=args.pi,
        cation_pi=args.cation_pi,
        hydrophobic=args.hydrophobic,
        clash_ratio=args.clash_ratio,
    )
    title = args.title or f"{Path(args.ligand).name} in {Path(args.receptor).name}"
    svg = analysis.interaction_diagram_svg(
        receptor,
        ligand,
        interactions,
        path=args.out,
        title=title,
        width=args.width,
        height=args.height,
    )
    if args.out:
        path = Path(args.out)
        if not path.exists() and isinstance(svg, str) and svg.strip():
            path.write_text(svg, encoding="utf-8")
        _eprint(f"wrote {args.out} ({len(interactions)} interaction(s))")
    elif isinstance(svg, str):
        sys.stdout.write(svg if svg.endswith("\n") else svg + "\n")
    return 0


def cmd_report(args) -> int:
    """Write the results table of a pose file as XLSX or CSV."""
    report = _lazy("odock.report", "the results table")
    from .docking import DockResult, Pose

    poses = _read_poses(args.poses)
    if not poses:
        raise SystemExit(f"error: no pose found in {args.poses}")

    receptor_text = ""
    if args.receptor:
        receptor_text = Path(args.receptor).read_text(encoding="utf-8", errors="replace")
    result = DockResult(
        poses=[
            Pose(
                index=pose["index"],
                affinity=0.0 if pose["affinity"] is None else float(pose["affinity"]),
                rmsd_lower_bound=float(pose["rmsd_lower_bound"]),
                rmsd_upper_bound=float(pose["rmsd_upper_bound"]),
                num_atoms=int(pose["coords"].shape[0]),
                coords=pose["coords"],
            )
            for pose in poses
        ],
        seed=0,
        receptor_pdbqt=receptor_text,
        ligand_pdbqt=poses[0]["text"],
        box=None,
        elapsed=0.0,
        ligand_atom_order=tuple(poses[0]["atom_names"]),
        _pdbqt="".join(pose["text"] for pose in poses),
    )

    fmt = "csv" if args.csv or Path(args.out).suffix.lower() == ".csv" else "xlsx"
    if fmt == "csv":
        returned = report.write_csv(args.out, result, receptor=None)
    else:
        returned = report.write_xlsx(args.out, result, receptor=None)
    if not Path(args.out).exists() and isinstance(returned, str) and returned.strip():
        # A writer that returns the document rather than saving it.
        Path(args.out).write_text(returned, encoding="utf-8")
    _eprint(f"wrote {args.out} ({len(result.poses)} mode(s), {fmt})")
    return 0


def cmd_fetch(args) -> int:
    """Download a PDB entry (or a chemical component) from the RCSB PDB."""
    from .fetch import fetch_ligand_sdf, fetch_pdb

    try:
        if args.ligand:
            text = fetch_ligand_sdf(args.identifier, args.out, timeout=args.timeout)
            suffix, kind = ".sdf", "ligand"
        else:
            text = fetch_pdb(args.identifier, args.out, timeout=args.timeout)
            suffix, kind = ".pdb", "PDB entry"
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")

    target = Path(args.out) if args.out else Path(args.identifier.strip().upper() + suffix)
    _eprint(f"wrote {target} ({len(text.splitlines())} lines, {len(text)} bytes, {kind})")
    return 0


def cmd_export_gpf(args) -> int:
    """Write an AutoGrid grid parameter file (GPF)."""
    from .export import export_gpf

    box = _export_box(args)
    text = export_gpf(
        box,
        args.receptor,
        args.out,
        gridfld=args.gridfld,
        receptor_types=args.receptor_types,
        ligand_types=args.ligand_types,
        smooth=args.smooth,
        dielectric=args.dielectric,
    )
    if args.out:
        _eprint(
            f"wrote {args.out} (spacing {box.spacing:g} Å, box "
            f"{box.size[0]:.1f}×{box.size[1]:.1f}×{box.size[2]:.1f} Å)"
        )
    else:
        sys.stdout.write(text)
    return 0


def cmd_export_dpf(args) -> int:
    """Write an AutoDock 4 docking parameter file (DPF)."""
    from .export import export_dpf

    box = _export_box(args)
    text = export_dpf(
        box,
        args.ligand,
        args.out,
        receptor=args.receptor,
        gridfld=args.gridfld,
        ligand_types=args.ligand_types,
        torsdof=args.torsdof,
        ndihe=args.ndihe,
        seed=args.seed,
        parameters=args.parameters,
        outlev=args.outlev,
    )
    if args.out:
        _eprint(f"wrote {args.out} (parameters {args.parameters})")
    else:
        sys.stdout.write(text)
    return 0


def cmd_export_config(args) -> int:
    """Write an AutoDock Vina configuration file."""
    from .export import export_vina_config

    box = _export_box(args)
    text = export_vina_config(
        box,
        args.receptor,
        args.ligand,
        args.out,
        exhaustiveness=args.exhaustiveness,
        num_modes=args.num_modes,
        energy_range=args.energy_range,
        out_poses=args.poses_out,
        seed=args.seed,
        cpu=args.cpu,
        scoring=args.scoring,
        min_rmsd=args.min_rmsd,
    )
    if args.out:
        _eprint(f"wrote {args.out} (Vina configuration)")
    else:
        sys.stdout.write(text)
    return 0


def cmd_export_pdb(args) -> int:
    """Write a cleaned structure as PDB, with a proper MASTER/END tail."""
    from .export import export_cleaned_pdb

    text = export_cleaned_pdb(args.input, args.out)
    if args.out:
        _eprint(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _launch_workbench(
    receptor: Optional[str] = None,
    ligand: Optional[str] = None,
    poses: Optional[str] = None,
    parser: Optional[argparse.ArgumentParser] = None,
) -> int:
    """Open the 3-D workbench, with a helpful message when it cannot start.

    Bare ``odock`` lands here: the workbench is the default entry point for
    interactive use, and the command-line subcommands are what you reach for
    when scripting.
    """
    try:
        from .gui import launch_gui
    except ImportError as exc:  # pragma: no cover - depends on the install
        _eprint(f"odock: the workbench is unavailable ({exc}).")
        _eprint("       install it with: pip install 'opendocking[gui]'")
        _eprint("")
        if parser is not None:
            parser.print_help(sys.stderr)
        return 2
    try:
        return int(launch_gui(receptor=receptor, ligand=ligand, poses=poses))
    except ImportError as exc:
        _eprint(f"odock: the workbench needs PyQt6 and ModernGL ({exc}).")
        _eprint("       install them with: pip install 'opendocking[gui]'")
        _eprint("")
        if parser is not None:
            parser.print_help(sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - depends on the display
        _eprint(f"odock: could not start the workbench: {exc}")
        _eprint("       run `odock --help` for the command-line interface.")
        return 2


def cmd_gui(args) -> int:
    """Launch the 3-D workbench."""
    return _launch_workbench(
        receptor=args.receptor,
        ligand=args.ligand,
        poses=args.poses,
        parser=args._parser if hasattr(args, "_parser") else None,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_interaction_thresholds(parser: argparse.ArgumentParser) -> None:
    """The geometric criteria shared by ``interactions`` and ``diagram``."""
    parser.add_argument(
        "--hbond", type=float, default=3.5, help="maximum D···A distance of a hydrogen bond (Å)"
    )
    parser.add_argument(
        "--salt", type=float, default=4.0, help="maximum charge-centre distance of a salt bridge (Å)"
    )
    parser.add_argument(
        "--pi", type=float, default=4.5, help="maximum ring-centre distance of a π-π stack (Å)"
    )
    parser.add_argument(
        "--cation-pi",
        dest="cation_pi",
        type=float,
        default=5.0,
        help="maximum cation···ring distance of a cation-π contact (Å)",
    )
    parser.add_argument(
        "--hydrophobic",
        type=float,
        default=4.0,
        help="maximum C···C distance of a hydrophobic contact (Å)",
    )
    parser.add_argument(
        "--clash-ratio",
        dest="clash_ratio",
        type=float,
        default=0.75,
        help="fraction of the summed van der Waals radii that counts as a clash",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser."""
    p = argparse.ArgumentParser(
        prog="odock",
        description="OpenDocking — a modern, open-source molecular docking toolchain",
        epilog=(
            "Run `odock gui` for the 3-D workbench (or `odock FILE...` to open "
            "it on some structures). Everything else below is the scripted "
            "interface; bare `odock` prints this help."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version="%(prog)s " + _version())
    sub = p.add_subparsers(dest="command", required=True)

    # -- info ---------------------------------------------------------------
    sp = sub.add_parser("info", help="show kernel, force-field and GPU information")
    sp.set_defaults(func=cmd_info)

    # -- prepare ------------------------------------------------------------
    pp = sub.add_parser("prepare", help="prepare a receptor or a ligand")
    psub = pp.add_subparsers(dest="kind", required=True)

    r = psub.add_parser("receptor", help="prepare a rigid receptor PDBQT")
    r.add_argument("input", help="input .pdb or .pdbqt file")
    r.add_argument("out", nargs="?", help="output .pdbqt (stdout when omitted)")
    r.add_argument("--keep-water", action="store_true", help="keep crystallographic waters")
    r.add_argument("--no-hetero", action="store_true", help="drop all non-standard residues")
    r.add_argument("--no-hydrogens", action="store_true", help="do not add polar hydrogens")
    r.add_argument(
        "--strip",
        nargs="+",
        metavar="RESNAME",
        help="residue names to delete, e.g. --strip BEN SO4 "
             "(use this for a holo structure, or the ligand blocks its own site)",
    )
    r.add_argument("--strict", action="store_true", help="fail instead of warning")
    r.set_defaults(func=cmd_prepare_receptor)

    l = psub.add_parser("ligand", help="prepare a ligand PDBQT")
    l.add_argument("input", help="input .sdf/.mol/.mol2/.pdb/.smi file")
    l.add_argument("out", nargs="?", help="output .pdbqt (stdout when omitted)")
    l.add_argument("--name", default="ligand", help="ligand name for the REMARK block")
    l.add_argument(
        "--smiles",
        help="the ligand's SMILES. Strongly recommended for a ligand taken from a "
             "PDB file: a PDB carries no bond orders, so without it the ring "
             "nitrogens are protonated by the valence model and the ligand gains "
             "phantom H-bond donors",
    )
    l.add_argument("--keep-nonpolar", action="store_true", help="keep non-polar hydrogens")
    l.add_argument("--no-hydrogens", action="store_true", help="do not add hydrogens")
    l.add_argument("--flexible-amides", action="store_true", help="treat amide bonds as rotors")
    l.add_argument("--no-optimize", action="store_true", help="skip the force-field pre-optimisation")
    l.add_argument("--seed", type=int, default=20240101, help="embedding seed")
    l.set_defaults(func=cmd_prepare_ligand)

    # -- box ----------------------------------------------------------------
    b = sub.add_parser("box", help="build a search box")
    b.add_argument("--ligand", help="centre the box on this ligand")
    b.add_argument("--receptor", help="receptor to derive the box from")
    b.add_argument("--residue", type=int, help="centre the box on this residue number")
    b.add_argument("--chain", help="restrict --residue to this chain")
    b.add_argument("--buffer", type=float, default=5.0, help="padding in Å")
    b.add_argument("--spacing", type=float, default=0.375, help="grid spacing in Å")
    b.add_argument("--out", help="write the box as JSON")
    b.set_defaults(func=cmd_box)

    # -- dock ---------------------------------------------------------------
    d = sub.add_parser("dock", help="dock a ligand")
    d.add_argument("-r", "--receptor", required=True, help="receptor PDBQT")
    d.add_argument("-l", "--ligand", required=True, help="ligand PDBQT")
    d.add_argument("-o", "--out", help="write the poses as multi-model PDBQT")
    d.add_argument("--json-out", help="write the poses as JSON")
    d.add_argument("--box", help="box JSON written by `odock box`")
    d.add_argument(
        "--box-from-ligand",
        action="store_true",
        help="derive the box from the ligand's input position",
    )
    d.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
    d.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
    d.add_argument("--buffer", type=float, default=5.0, help="padding for --box-from-ligand")
    d.add_argument("--spacing", type=float, default=0.375, help="grid spacing in Å")
    d.add_argument(
        "-s", "--scoring", default="vina", choices=["vina", "vinardo", "ad4"],
        help="force field",
    )
    d.add_argument("-e", "--exhaustiveness", type=int, default=8)
    d.add_argument("-n", "--num-poses", type=int, default=9)
    d.add_argument("--seed", type=int, default=0, help="0 draws a random seed")
    d.add_argument("--min-rmsd", type=float, default=1.0)
    d.add_argument("--energy-range", type=float, default=3.0)
    d.add_argument("--no-grid", action="store_true", help="use the exact scorer in the search too")
    d.add_argument("--no-refine", action="store_true", help="skip exact post-refinement")
    d.add_argument("--ga", action="store_true", help="use the island-model genetic algorithm")
    d.add_argument(
        "--search",
        choices=["monte_carlo", "lga", "lga_solis"],
        default=None,
        help="search algorithm: monte_carlo (parallel ILS), lga (island-model "
             "Lamarckian GA) or lga_solis (LGA with a Solis-Wets local search, "
             "which is what AutoDock 4 uses). Unset means monte_carlo, or lga "
             "with --ga",
    )
    d.add_argument("--islands", type=int, default=4)
    d.add_argument("--population", type=int, default=32)
    d.add_argument("--generations", type=int, default=20)
    d.add_argument("-q", "--quiet", action="store_true", help="do not print the table")
    d.set_defaults(func=cmd_dock)

    # -- score --------------------------------------------------------------
    s = sub.add_parser("score", help="score a ligand in place")
    s.add_argument("-r", "--receptor", required=True)
    s.add_argument("-l", "--ligand", required=True)
    s.add_argument("--box")
    s.add_argument("--box-from-ligand", action="store_true")
    s.add_argument("--center", nargs=3, type=float)
    s.add_argument("--size", nargs=3, type=float)
    s.add_argument("--buffer", type=float, default=5.0)
    s.add_argument("--spacing", type=float, default=0.375)
    s.add_argument("--scoring", default="vina", choices=["vina", "vinardo", "ad4"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_score)

    # -- split --------------------------------------------------------------
    sp2 = sub.add_parser("split", help="split a multi-model PDBQT file")
    sp2.add_argument("input")
    sp2.add_argument("--outdir", default="poses")
    sp2.set_defaults(func=cmd_split)

    # -- pocket -------------------------------------------------------------
    pk = sub.add_parser("pocket", help="detect binding pockets on a ligand-free receptor")
    pk.add_argument("-r", "--receptor", required=True, help="receptor .pdb or .pdbqt")
    pk.add_argument("--spacing", type=float, default=1.0, help="grid spacing in Å")
    pk.add_argument("--probe", type=float, default=1.4, help="radius of the rolling probe (Å)")
    pk.add_argument(
        "--min-volume", type=float, default=100.0, help="discard cavities smaller than this (Å³)"
    )
    pk.add_argument(
        "--max", dest="max_pockets", type=int, default=10, help="keep at most this many pockets"
    )
    pk.add_argument(
        "--buriedness",
        type=float,
        default=0.55,
        help="fraction of walled directions a cavity seed needs",
    )
    pk.add_argument("--json-out", help="write the pockets as JSON")
    pk.set_defaults(func=cmd_pocket)

    # -- filter -------------------------------------------------------------
    fl = sub.add_parser("filter", help="Lipinski/Veber/PAINS report for one or more files")
    fl.add_argument(
        "-i",
        "--input",
        action="append",
        required=True,
        metavar="FILE",
        help="ligand file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat for many files",
    )
    fl.add_argument("--json-out", help="write the verdicts as JSON")
    fl.set_defaults(func=cmd_filter)

    # -- cluster ------------------------------------------------------------
    cl = sub.add_parser("cluster", help="cluster the poses of a docking run by RMSD")
    cl.add_argument("-p", "--poses", required=True, help="multi-model pose PDBQT")
    cl.add_argument(
        "-l", "--ligand", help="reference ligand PDBQT: adds its RMSD to every cluster"
    )
    cl.add_argument("--cutoff", type=float, default=2.0, help="RMSD cluster cutoff in Å")
    cl.add_argument("--json-out", help="write the clusters as JSON")
    cl.set_defaults(func=cmd_cluster)

    # -- interactions -------------------------------------------------------
    it = sub.add_parser("interactions", help="profile the non-covalent interactions")
    it.add_argument("-r", "--receptor", required=True, help="receptor PDBQT")
    it.add_argument("-l", "--ligand", required=True, help="ligand PDBQT")
    _add_interaction_thresholds(it)
    it.add_argument("--json-out", help="write the interactions as JSON")
    it.set_defaults(func=cmd_interactions)

    # -- diagram ------------------------------------------------------------
    dg = sub.add_parser("diagram", help="draw the 2-D interaction diagram as SVG")
    dg.add_argument("-r", "--receptor", required=True, help="receptor PDBQT")
    dg.add_argument("-l", "--ligand", required=True, help="ligand PDBQT")
    dg.add_argument("-o", "--out", help="output .svg (stdout when omitted)")
    dg.add_argument("--title", help="diagram title")
    dg.add_argument("--width", type=int, default=900, help="image width in pixels")
    dg.add_argument("--height", type=int, default=700, help="image height in pixels")
    _add_interaction_thresholds(dg)
    dg.set_defaults(func=cmd_diagram)

    # -- report -------------------------------------------------------------
    rp = sub.add_parser("report", help="write the results table as XLSX or CSV")
    rp.add_argument("-p", "--poses", required=True, help="pose PDBQT written by `odock dock`")
    rp.add_argument("-r", "--receptor", help="receptor PDBQT, for the interaction column")
    rp.add_argument("-o", "--out", required=True, help="output .xlsx or .csv")
    rp.add_argument("--csv", action="store_true", help="write CSV, whatever the file name says")
    rp.set_defaults(func=cmd_report)

    # -- fetch --------------------------------------------------------------
    ft = sub.add_parser("fetch", help="download a structure from the RCSB PDB")
    ft.add_argument(
        "identifier",
        metavar="PDB_ID",
        help="four-character PDB ID (or a chemical-component ID together with --ligand)",
    )
    ft.add_argument("-o", "--out", help="output file (default: <ID>.pdb, or <ID>.sdf)")
    ft.add_argument(
        "-l",
        "--ligand",
        action="store_true",
        help="the identifier is a chemical component: fetch its SDF instead of a PDB entry",
    )
    ft.add_argument("--timeout", type=float, default=30.0, help="connection timeout in seconds")
    ft.set_defaults(func=cmd_fetch)

    # -- export -------------------------------------------------------------
    ex = sub.add_parser("export", help="write AutoGrid/AutoDock/Vina input files")
    exsub = ex.add_subparsers(dest="kind", required=True)

    def add_box(parser: argparse.ArgumentParser, *, ligand: bool = False) -> None:
        parser.add_argument("--box", help="box JSON written by `odock box`")
        parser.add_argument("--center", nargs=3, type=float, metavar=("X", "Y", "Z"))
        parser.add_argument("--size", nargs=3, type=float, metavar=("X", "Y", "Z"))
        parser.add_argument(
            "--spacing",
            type=float,
            default=None,
            help="grid spacing in Å (default: 0.375, or the spacing stored in --box)",
        )
        if ligand:
            parser.add_argument(
                "--box-from-ligand",
                action="store_true",
                help="derive the box from the ligand's input position",
            )
            parser.add_argument(
                "--buffer", type=float, default=5.0, help="padding for --box-from-ligand (Å)"
            )

    g = exsub.add_parser("gpf", help="AutoGrid grid parameter file")
    g.add_argument("-r", "--receptor", required=True, help="receptor PDBQT, written into the file")
    g.add_argument("-o", "--out", help="output .gpf (stdout when omitted)")
    add_box(g)
    g.add_argument("--gridfld", help=".fld name (default: <receptor stem>.maps.fld)")
    g.add_argument(
        "--receptor-types",
        nargs="+",
        metavar="TYPE",
        help="receptor atom types (default: read from the receptor PDBQT)",
    )
    g.add_argument(
        "--ligand-types",
        nargs="+",
        metavar="TYPE",
        help="types to build maps for (default: the receptor types)",
    )
    g.add_argument("--smooth", type=float, default=0.5, help="smooth radius in Å")
    g.add_argument(
        "--dielectric",
        type=float,
        default=-0.1465,
        help="dielectric constant; negative means AD4's distance-dependent one",
    )
    g.set_defaults(func=cmd_export_gpf)

    dp = exsub.add_parser("dpf", help="AutoDock 4 docking parameter file")
    dp.add_argument("-r", "--receptor", required=True, help="receptor PDBQT, written into the file")
    dp.add_argument("-l", "--ligand", required=True, help="ligand PDBQT (the `move` line)")
    dp.add_argument("-o", "--out", help="output .dpf (stdout when omitted)")
    add_box(dp, ligand=True)
    dp.add_argument("--gridfld", help=".fld name (default: <receptor stem>.maps.fld)")
    dp.add_argument("--ligand-types", nargs="+", metavar="TYPE", help="override the ligand types")
    dp.add_argument(
        "--torsdof", type=int, help="torsional degrees of freedom (default: the PDBQT's TORSDOF)"
    )
    dp.add_argument("--ndihe", type=int, help="number of real dihedrals (default: --torsdof)")
    dp.add_argument(
        "--parameters",
        choices=["lga", "ga", "ls", "none"],
        default="lga",
        help="search section to write",
    )
    dp.add_argument("--seed", type=int, help="fixed random seed (default: `seed pid time`)")
    dp.add_argument("--outlev", type=int, default=1, help="diagnostic output level")
    dp.set_defaults(func=cmd_export_dpf)

    cf = exsub.add_parser("config", help="AutoDock Vina configuration file")
    cf.add_argument("-r", "--receptor", required=True, help="receptor PDBQT, written into the file")
    cf.add_argument("-l", "--ligand", required=True, help="ligand PDBQT, written into the file")
    cf.add_argument("-o", "--out", help="output file (stdout when omitted)")
    add_box(cf, ligand=True)
    cf.add_argument("--poses-out", help="`out = ...`: where Vina writes the poses")
    cf.add_argument("-e", "--exhaustiveness", type=int, default=8, help="search effort")
    cf.add_argument("-n", "--num-modes", type=int, default=9, help="binding modes to report")
    cf.add_argument("--energy-range", type=float, default=3.0, help="energy window of a mode (kcal/mol)")
    cf.add_argument("--min-rmsd", type=float, help="minimum RMSD between reported modes (Å)")
    cf.add_argument("--seed", type=int, help="random seed (default: Vina draws one)")
    cf.add_argument("--cpu", type=int, help="number of CPUs (default: every core)")
    cf.add_argument(
        "--scoring",
        choices=["vina", "vinardo", "ad4"],
        help="scoring function (default: vina)",
    )
    cf.set_defaults(func=cmd_export_config)

    ep = exsub.add_parser("pdb", help="write a structure as PDB with a MASTER/END tail")
    ep.add_argument(
        "-i", "--input", required=True, help="structure to read (.sdf/.mol/.mol2/.pdb/.pdbqt)"
    )
    ep.add_argument("-o", "--out", help="output .pdb (stdout when omitted)")
    ep.set_defaults(func=cmd_export_pdb)

    # -- gui ----------------------------------------------------------------
    g = sub.add_parser("gui", help="launch the 3-D workbench")
    g.add_argument("-r", "--receptor", help="receptor PDBQT to load")
    g.add_argument("-l", "--ligand", help="ligand PDBQT to load")
    g.add_argument("-p", "--poses", help="pose PDBQT written by `odock dock`")
    g.set_defaults(func=cmd_gui)

    return p


def _version() -> str:
    try:
        from . import __version__

        return __version__
    except Exception:  # pragma: no cover
        return "unknown"


def _use_utf8_streams() -> None:
    """Make the console able to print the units this tool reports.

    The output contains `Å` and `Å³`. On a Windows console whose code page is not
    UTF-8 (a Chinese or Western European locale, for instance) Python encodes
    stdout with GBK or cp1252 and `print` raises `UnicodeEncodeError`, turning
    an informational command into a crash. Reconfiguring the stream is harmless
    where the console is already UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - stream replaced
            pass


def _subcommands(parser: argparse.ArgumentParser) -> set:
    """The subcommand names the parser knows about."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _classify_structure(path: Path) -> str:
    """Guess what a PDBQT file is: a receptor, a ligand or a set of poses."""
    try:
        head = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "receptor"
    if head.count("MODEL") > 1:
        return "poses"
    upper = head.upper()
    if "\nROOT" in upper or upper.lstrip().startswith("ROOT"):
        return "ligand"
    return "receptor"


def _workbench_arguments(paths: Sequence[str]):
    """Turn ``odock file [file ...]`` into workbench arguments.

    ``odock demo/3ptb/poses.pdbqt`` is the natural thing to type, so it opens
    the workbench on those files instead of failing with "invalid choice".
    """
    receptor = ligand = poses = None
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            return None
        kind = _classify_structure(path)
        if kind == "poses" and poses is None:
            poses = raw
        elif kind == "ligand" and ligand is None:
            ligand = raw
        elif receptor is None:
            receptor = raw
        else:
            return None
    return receptor, ligand, poses


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by the ``odock`` console script.

    ``odock`` on its own prints the command-line help; ``odock gui`` opens the
    3-D workbench. That split is deliberate: a bare invocation must never
    surprise a script by opening a window, and the workbench stays one explicit
    word away.
    """
    _use_utf8_streams()
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not raw:
        parser.print_help()
        _eprint("")
        _eprint("Run `odock gui` for the 3-D workbench.")
        return 0

    # `odock FILE...` opens the workbench on those structures: still explicit
    # enough to be safe, and it is how a structure is usually opened by hand.
    if (
        not raw[0].startswith("-")
        and raw[0] not in _subcommands(parser)
        and all(not a.startswith("-") for a in raw)
        and len(raw) <= 3
    ):
        targets = _workbench_arguments(raw)
        if targets is not None:
            return _launch_workbench(*targets, parser=parser)

    args = parser.parse_args(raw)
    args._parser = parser
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        _eprint("interrupted")
        return 130
    except BrokenPipeError:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

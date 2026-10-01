# SPDX-License-Identifier: GPL-3.0-or-later
"""The command-line surface of the ligand-chemistry layer.

Everything here belongs to the ligand side of the tool: the four
``odock similar`` / ``diverse`` / ``scaffolds`` / ``rgroups`` subcommands and the
``--diverse N`` option the screening pipeline uses to dock a representative
subset instead of the whole library.

The parsers live in this module rather than in :mod:`odock.cli` on purpose.
``cli.py`` is edited by several workstreams at once, so each one registers its
own subcommands through a single call::

    # in odock.cli.build_parser
    from .ligandcli import attach_ligand_chemistry
    ...
    attach_ligand_chemistry(sub)      # just before `return p`

The command bodies themselves stay in :mod:`odock.cli` with the other
subcommands; the imports below are deliberately deferred to call time so this
module never imports ``cli`` while ``cli`` is still importing it, and so
``odock.ligandsim`` stays free of ``argparse``.
"""

from __future__ import annotations

import argparse
import json

__all__ = [
    "FINGERPRINT_CHOICES",
    "add_fingerprint_options",
    "attach_ligand_chemistry",
    "add_screen_diversity_options",
    "command_functions",
]

#: The fingerprint kinds the CLI offers, in help order.
FINGERPRINT_CHOICES = (
    "morgan",
    "rdkit",
    "atom_pair",
    "torsion",
    "maccs",
    "pharmacophore",
)


def command_functions():
    """The four command functions, imported from :mod:`odock.cli` on demand.

    Deferred because ``cli`` calls :func:`attach_ligand_chemistry` while it is
    itself being imported: a module-level import here would be a cycle.
    """
    from .cli import cmd_diverse, cmd_rgroups, cmd_scaffolds, cmd_similar

    return {
        "similar": cmd_similar,
        "diverse": cmd_diverse,
        "scaffolds": cmd_scaffolds,
        "rgroups": cmd_rgroups,
    }


def add_fingerprint_options(parser: argparse.ArgumentParser) -> None:
    """The fingerprint options the ligand-chemistry commands share."""
    parser.add_argument(
        "--fingerprint",
        default="morgan",
        choices=list(FINGERPRINT_CHOICES),
        help="morgan (ECFP-style circular, the default), rdkit (path-based), "
             "atom_pair, torsion, maccs (166 public keys) or pharmacophore "
             "(3-D, needs --3d)",
    )
    parser.add_argument("--radius", type=int, default=2, help="Morgan radius (2 = ECFP4)")
    parser.add_argument("--bits", type=int, default=2048, help="fingerprint length in bits")
    parser.add_argument(
        "--features", action="store_true",
        help="hash pharmacophore features instead of atom environments (FCFP)",
    )
    parser.add_argument(
        "--chirality", action="store_true", help="include chirality in the fingerprint"
    )


def _add_similar(subs) -> None:
    commands = command_functions()
    si = subs.add_parser(
        "similar",
        help="find the library analogues of a query molecule",
        description=(
            "Rank every library molecule by fingerprint similarity to a query and "
            "report the ones above a cut-off.  2-D by default (Morgan/ECFP, the "
            "coefficient the similarity literature reports); --3d switches to a "
            "conformer-dependent pharmacophore search that is much more expensive "
            "and says so."
        ),
    )
    si.add_argument("-q", "--query", required=True, help="query SMILES or structure file")
    si.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    si.add_argument("--cutoff", type=float, default=0.7, help="minimum similarity to report")
    si.add_argument(
        "--metric", default="tanimoto", choices=["tanimoto", "dice", "tversky"],
        help="similarity coefficient",
    )
    si.add_argument("--top", type=int, default=0, help="keep at most N hits (0 = all)")
    si.add_argument("--name", help="the query's name in the output")
    si.add_argument(
        "--3d", dest="three_d", action="store_true",
        help="best-over-conformers 3-D pharmacophore similarity (slow)",
    )
    si.add_argument(
        "--conformers", type=int, default=8,
        help="conformers embedded per molecule by --3d",
    )
    si.add_argument("--seed", type=int, default=20240101, help="ETKDGv3 embedding seed for --3d")
    si.add_argument(
        "--matrix", action="store_true", help="print the whole library similarity matrix"
    )
    add_fingerprint_options(si)
    si.add_argument("--json-out", help="write the hits (and matrix) as JSON")
    si.set_defaults(func=commands["similar"])


def _add_diverse(subs) -> None:
    commands = command_functions()
    dv = subs.add_parser(
        "diverse",
        help="pick a representative subset of a library and report what it covers",
        description=(
            "MaxMin picking (or sphere exclusion at a --cutoff) chooses a subset of "
            "the library that is as mutually dissimilar as the fingerprints allow.  "
            "The command reports the tightest pair inside the subset and how much of "
            "the library's scaffold space the subset covers."
        ),
    )
    dv.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    dv.add_argument("-n", type=int, required=True, help="how many molecules to pick")
    dv.add_argument(
        "--method", default="maxmin", choices=["maxmin", "sphere"],
        help="maxmin (greedy farthest-point) or sphere (keep below --cutoff)",
    )
    dv.add_argument("--cutoff", type=float, default=None, help="sphere-exclusion similarity cut-off")
    dv.add_argument(
        "--metric", default="tanimoto", choices=["tanimoto", "dice", "tversky"],
        help="similarity coefficient",
    )
    dv.add_argument(
        "--start", default="0",
        help="first pick: a library index, or 'centroid' (the most typical molecule; "
             "that costs an N^2 comparison)",
    )
    dv.add_argument("--generic", action="store_true", help="count generic skeletons, not scaffolds")
    dv.add_argument("-o", "--out", help="write the subset as .sdf (properties kept) or .smi")
    add_fingerprint_options(dv)
    dv.add_argument("--json-out", help="write the selection and the coverage as JSON")
    dv.set_defaults(func=commands["diverse"])


def _add_scaffolds(subs) -> None:
    commands = command_functions()
    sf = subs.add_parser(
        "scaffolds",
        help="group a library by Murcko scaffold and cluster the scaffolds into series",
        description=(
            "The scaffold view of a library: how many molecules share each Murcko "
            "scaffold, which scaffold is largest, and how many scaffold families "
            "(series) the library falls into at a similarity cut-off.  --report "
            "prints the whole chemistry page."
        ),
    )
    sf.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    sf.add_argument(
        "--affinities",
        help="a screening results.jsonl/results.csv: adds the measured or docked "
             "affinity to every scaffold group and series",
    )
    sf.add_argument("--generic", action="store_true", help="use generic Murcko skeletons")
    sf.add_argument(
        "--series-cutoff", type=float, default=0.65,
        help="scaffold similarity at which two scaffolds are one series",
    )
    sf.add_argument("--top", type=int, default=0, help="show at most N rows per table")
    sf.add_argument(
        "--core", help="the core for the R-group part of --report (a SMILES, or 'mcs')"
    )
    sf.add_argument(
        "--report", action="store_true",
        help="print the full chemistry section: scaffolds, series, R-groups, matched pairs",
    )
    sf.add_argument("--json-out", help="write the groups and series as JSON")
    sf.set_defaults(func=commands["scaffolds"])


def _add_rgroups(subs) -> None:
    commands = command_functions()
    rg = subs.add_parser(
        "rgroups",
        help="decompose a library into R-groups around a common core",
        description=(
            "The molecule x R-group matrix a medicinal chemist asks for first: every "
            "molecule's substituent at every attachment point of a shared core, with "
            "an optional matched-molecular-pair analysis of the affinity deltas."
        ),
    )
    rg.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    rg.add_argument(
        "--core",
        help="the common core: a SMILES (with [*:n] attachment points if you have "
             "them), 'mcs' for the maximum common substructure, or omitted for the "
             "most common Murcko scaffold",
    )
    rg.add_argument("--generic", action="store_true", help="derive the core from generic skeletons")
    rg.add_argument(
        "--affinities",
        help="a screening results.jsonl/results.csv: adds affinity columns, the series "
             "sheet and the matched-pair deltas",
    )
    rg.add_argument("--top", type=int, default=0, help="show at most N molecules / pairs")
    rg.add_argument(
        "--mmp", action="store_true",
        help="also report the matched molecular pairs and their affinity deltas",
    )
    rg.add_argument(
        "-o", "--out", help="write the matrix as .xlsx (with a series sheet) or .csv"
    )
    rg.add_argument("--json-out", help="write the decomposition as JSON")
    rg.set_defaults(func=commands["rgroups"])


def add_screen_diversity_options(screen_parser: argparse.ArgumentParser) -> None:
    """Give an existing ``screen`` parser the ``--diverse`` family of options.

    Called with the already-built ``screen`` sub-parser so the screening pipeline
    gains the option without ``cli.py`` growing a block for it.  The parser's
    ``func`` is wrapped rather than replaced: the wrapper selects the subset and
    rewrites ``args.input`` before calling the original command, which therefore
    needs no knowledge of this feature at all.
    """
    screen_parser.add_argument(
        "--diverse", type=int, default=0, metavar="N",
        help="dock a representative subset of N molecules instead of the whole "
             "library (MaxMin over Morgan r=2/2048 fingerprints, chosen from the "
             "molecules that pass the drug-likeness filters). The subset is written "
             "as library_diverse.sdf and its scaffold coverage is reported in "
             "diverse.json",
    )
    screen_parser.add_argument(
        "--diverse-method", choices=["maxmin", "sphere"], default="maxmin",
        help="how --diverse picks the subset",
    )
    screen_parser.add_argument(
        "--diverse-cutoff", type=float, default=None,
        help="similarity cut-off for --diverse-method sphere",
    )
    screen_parser.add_argument(
        "--diverse-start", default="0",
        help="first pick of --diverse: a library index or 'centroid'",
    )
    original = screen_parser.get_default("func")
    if original is None or getattr(original, "_odock_diverse_wrapper", False):
        return
    screen_parser.set_defaults(func=_wrap_screen_command(original))


def _wrap_screen_command(original):
    """Wrap ``odock.cli.cmd_screen`` so ``--diverse`` selects before it runs."""

    def screen_with_diversity(args):
        if getattr(args, "diverse", 0):
            from pathlib import Path

            target = _diverse_screening_library(args, Path(args.out))
            args.input = [str(target)]
        return original(args)

    screen_with_diversity._odock_diverse_wrapper = True  # type: ignore[attr-defined]
    screen_with_diversity.__doc__ = original.__doc__
    screen_with_diversity.__name__ = getattr(original, "__name__", "cmd_screen")
    return screen_with_diversity


def _diverse_screening_library(args, outdir):
    """Write the ``--diverse N`` subset of a screening library and report it.

    The subset is selected from the molecules that survive the drug-likeness
    filters, so ``--diverse 100`` really is 100 dockable molecules, and it is
    written as an SDF beside the results directory (properties preserved).  The
    file is only rewritten when its content changes, which keeps a resumed run's
    input signature — and therefore its resume — stable.
    """
    from pathlib import Path

    from .cli import (
        _chemistry_guard,
        _fingerprint_options,
        _lazy,
        _read_json,
        _read_library,
        _write_subset,
        _eprint,
    )

    sim = _lazy("odock.ligandsim", "library diversity selection")
    scaffold = _lazy("odock.scaffold", "ligand chemistry")

    mols, names = _read_library(list(args.input))
    if not args.no_filter:
        filters = _lazy("odock.filters", "the drug-likeness filters")
        kept_mols, kept_names = [], []
        for mol, name in zip(mols, names):
            verdict = filters.drug_like(mol)
            if verdict.get("passed", True):
                kept_mols.append(mol)
                kept_names.append(name)
        _eprint(
            f"--diverse: {len(kept_mols)} of {len(mols)} molecule(s) survive the "
            "drug-likeness filters and are eligible for the subset"
        )
        mols, names = kept_mols, kept_names
    else:
        _eprint("--diverse: --no-filter, so every molecule is eligible")

    if not mols:
        raise SystemExit(
            "error: no molecule survived the filters, so --diverse has nothing to pick"
        )
    if int(args.diverse) >= len(mols):
        _eprint(
            f"--diverse {int(args.diverse)} >= the {len(mols)} eligible molecule(s); "
            "the whole library is used"
        )

    library = _chemistry_guard(
        sim.fingerprint_set,
        mols,
        names=names,
        smiles=True,
        source=", ".join(str(source) for source in args.input),
        **_fingerprint_options(args),
    )
    start = int(args.diverse_start) if str(args.diverse_start).isdigit() else args.diverse_start
    if args.diverse_method == "sphere" and args.diverse_cutoff is None:
        raise SystemExit("error: --diverse-method sphere needs --diverse-cutoff")
    selection = _chemistry_guard(
        sim.diversity_subset,
        library,
        int(args.diverse),
        method=args.diverse_method,
        cutoff=args.diverse_cutoff,
        metric="tanimoto",
        start=start,
    )
    coverage = sim.scaffold_coverage(selection, mols)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / "library_diverse.sdf"
    payload = selection.as_dict()
    payload["input"] = [str(source) for source in args.input]
    payload["scaffold_coverage"] = {k: round(v, 6) for k, v in coverage.items()}
    payload["subset_file"] = str(target)
    # The subset file's size and mtime ride in the campaign's library hash, so an
    # unchanged selection must not be rewritten: a resumed run has to find the
    # same input signature it wrote.  The sidecar `diverse.json` is the record of
    # what the file on disk holds, and it is not part of any hash.
    previous = _read_json(outdir / "diverse.json")
    unchanged = bool(
        previous
        and previous.get("indices") == payload["indices"]
        and previous.get("method") == payload["method"]
        and previous.get("cutoff") == payload["cutoff"]
        and previous.get("input") == payload["input"]
        and target.exists()
    )
    if not unchanged:
        _write_subset(target, library, selection)
    (outdir / "diverse.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    _eprint(
        f"--diverse: {selection.n_selected} of {selection.n_library} eligible molecule(s) "
        f"by {selection.method} (tanimoto); tightest pair inside the subset "
        f"{selection.worst_pairwise_similarity:.3f}"
    )
    _eprint(
        f"--diverse: scaffold space {coverage['n_subset_scaffolds']:.0f}/"
        f"{coverage['n_library_scaffolds']:.0f} = {coverage['coverage']:.1%} covered by "
        f"{coverage['subset_fraction']:.1%} of the molecules"
    )
    _eprint(
        f"--diverse: {'kept' if unchanged else 'wrote'} {target} and "
        f"{outdir / 'diverse.json'}"
    )
    return target


def _add_pharmacophore(subs) -> None:
    """``odock pharmacophore build|screen|show``.

    The three handlers live in this module (not in ``cli.py``): the extension
    point exists so a feature module can own its whole command-line surface.
    """
    ph = subs.add_parser(
        "pharmacophore",
        help="build a pharmacophore model from evidence and screen a library with it",
        description=(
            "Derive the chemical features that recur across a set of known ligands "
            "(a series, a scaffold group, the top hits of a screen), then rank a "
            "library by how well its molecules can present those features, with a "
            "shape constraint.  The model reports how many members support every "
            "feature: a model built from one or two molecules is an anecdote and "
            "says so."
        ),
    )
    phsub = ph.add_subparsers(dest="kind", required=True)

    build = phsub.add_parser("build", help="build a model from a set of members")
    build.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="the members: a library file (.sdf/.smi/.mol2/.pdb/.pdbqt) whose every "
             "record is one member; repeat to combine several",
    )
    build.add_argument(
        "--frame", choices=["shared", "align"], default="align",
        help="'align' (default) superposes every member on --core; 'shared' takes the "
             "input coordinates as they are (docked poses, one crystal frame)",
    )
    build.add_argument(
        "--core", help="the superposition core: a SMILES, or 'mcs' (default) for the "
                       "maximum common substructure of the members",
    )
    build.add_argument(
        "--threshold", type=float, default=0.5,
        help="minimum fraction of members that must support a feature",
    )
    build.add_argument("--radius", type=float, default=1.5, help="feature merge/tolerance radius (Å)")
    build.add_argument(
        "--min-members", type=int, default=3,
        help="below this many contributing members the model is flagged as an anecdote",
    )
    build.add_argument(
        "--shape", choices=["envelope", "spheres", "none"], default="envelope",
        help="shape constraint: the members' heavy-atom envelope (default), exclusion "
             "spheres on the recurring scaffold, or none",
    )
    build.add_argument("--shape-tolerance", type=float, default=1.2, help="envelope slack (Å)")
    build.add_argument(
        "--min-coverage", type=float, default=0.6,
        help="fraction of a candidate's heavy atoms that must sit inside the envelope",
    )
    build.add_argument("--seed", type=int, default=20240101, help="ETKDGv3 seed")
    build.add_argument("-o", "--out", help="write the model as JSON (required by screen)")
    build.add_argument("--json-out", help="write the model as JSON (same as --out)")
    build.set_defaults(func=cmd_pharmacophore_build)

    screen = phsub.add_parser("screen", help="rank a library against a model")
    screen.add_argument("-m", "--model", required=True, help="a model written by `build`")
    screen.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="the library to screen; repeat to combine several",
    )
    screen.add_argument("--tolerance", type=float, default=1.5, help="feature matching tolerance (Å)")
    screen.add_argument(
        "--miss-penalty", type=float, default=1.0,
        help="score cost of a missed feature, in units of one feature",
    )
    screen.add_argument("--conformers", type=int, default=8, help="conformers per molecule")
    screen.add_argument("--seed", type=int, default=20240101, help="ETKDGv3 seed")
    screen.add_argument("--top", type=int, default=0, help="keep at most N molecules (0 = all)")
    screen.add_argument(
        "--align", choices=["auto", "core", "features", "none"], default=None,
        help="how a molecule is placed on the model: the core, a matching feature "
             "triple, automatic (default) or none (the coordinates are already in "
             "the model's frame)",
    )
    screen.add_argument(
        "--no-shape", action="store_true",
        help="score the features only, ignoring the model's shape constraint",
    )
    screen.add_argument(
        "--actives",
        help="a file of known actives (.smi/.sdf): adds precision@k, recall@k, the "
             "enrichment factor and the AUC, with their sample size",
    )
    screen.add_argument(
        "--scores-out", help="write the ranked scores as CSV"
    )
    screen.add_argument("--json-out", help="write the ranked scores as JSON")
    screen.set_defaults(func=cmd_pharmacophore_screen)

    show = phsub.add_parser("show", help="print a model")
    show.add_argument("-m", "--model", required=True, help="a model written by `build`")
    show.add_argument("--json-out", help="write the model as JSON")
    show.set_defaults(func=cmd_pharmacophore_show)


# ---------------------------------------------------------------------------
# `odock pharmacophore build|screen|show`
# ---------------------------------------------------------------------------


def cmd_pharmacophore_build(args) -> int:
    """Derive a pharmacophore model from the members in ``-i`` and write it out."""
    from .cli import _chemistry_guard, _eprint, _lazy, _read_library, _write_json

    pharm = _lazy("odock.pharmacophore", "pharmacophore modelling")
    mols, names = _read_library(list(args.input))
    model = _chemistry_guard(
        pharm.build_model,
        mols,
        frame=args.frame,
        core=args.core,
        names=names,
        radius=float(args.radius),
        threshold=float(args.threshold),
        min_members=int(args.min_members),
        shape=args.shape,
        shape_tolerance=float(args.shape_tolerance),
        min_coverage=float(args.min_coverage),
        seed=int(args.seed),
        source=", ".join(str(source) for source in args.input),
    )
    print(model.table())
    print()
    print(
        f"built from {model.n_used} of {model.n_members} member(s); "
        f"{model.n_features} feature(s); shape {model.shape_mode}; "
        f"{'meaningful' if model.is_meaningful else 'ANECDOTE (too few members)'}"
    )
    if not model.is_meaningful:
        _eprint(
            f"odock pharmacophore: warning: only {model.n_used} member(s) contributed. "
            "A feature that recurs in two molecules is not evidence; treat the model "
            "as a description of those molecules."
        )
    target = args.out or args.json_out
    if target:
        written = model.save(target)
        _eprint(f"wrote {written}")
    return 0


def cmd_pharmacophore_screen(args) -> int:
    """Rank a library by its fit to a model, optionally against known actives."""
    import csv
    from pathlib import Path

    from .cli import _chemistry_guard, _eprint, _lazy, _read_library, _write_json

    pharm = _lazy("odock.pharmacophore", "pharmacophore modelling")
    if not Path(args.model).exists():
        raise SystemExit(f"error: no such model file: {args.model}")
    model = _chemistry_guard(pharm.PharmacophoreModel.load, args.model)
    mols, names = _read_library(list(args.input))
    hits = _chemistry_guard(
        pharm.screen,
        model,
        mols,
        names=names,
        tolerance=float(args.tolerance),
        miss_penalty=float(args.miss_penalty),
        conformers=int(args.conformers),
        seed=int(args.seed),
        top=int(args.top or 0),
        enforce_shape=not args.no_shape,
        align=args.align,
        source=", ".join(str(source) for source in args.input),
    )
    print(hits.model.table() if hits.model is not None else "no model")
    print()
    print(hits.table(limit=int(args.top) if args.top else 0))
    print()
    print(
        f"screen: {hits.n_hits} molecule(s) scored, {hits.n_rejected} rejected by the "
        f"shape constraint or an unalignable molecule, {hits.n_conformers} conformer(s) "
        f"in {hits.seconds:.2f} s"
    )

    report = None
    if args.actives:
        actives, _ = _read_library([args.actives])
        active_names = []
        for index, mol in enumerate(actives):
            try:
                label = mol.GetProp("_Name").strip()
            except Exception:  # pragma: no cover - defensive
                label = ""
            active_names.append(label or f"active_{index + 1}")
        report = pharm.enrichment(hits, active_names)
        print()
        print(
            f"enrichment: {report['n_actives_in_top_k']} of the {report['k']} best "
            f"molecule(s) are actives ({report['precision_at_k']:.0%} precision, "
            f"{report['recall_at_k']:.0%} recall), base rate {report['base_rate']:.1%}, "
            f"enrichment factor {report['enrichment_factor']:.2f}, AUC "
            f"{report['auc']}"
        )
        for note in report["notes"]:
            print(f"  note: {note}")

    if args.scores_out:
        target = Path(args.scores_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["rank", "name", "fit", "matched", "n_features", "missed", "clashes",
                 "shape_ok", "rejected", "alignment", "reason", "smiles"]
            )
            for rank, hit in enumerate(hits.hits, start=1):
                writer.writerow(
                    [
                        rank, hit.name, f"{hit.fit:.4f}", hit.n_matched, hit.n_features,
                        hit.n_missed, hit.clashes, "1" if hit.shape_ok else "0",
                        "1" if hit.rejected else "0", hit.alignment, hit.reason, hit.smiles,
                    ]
                )
        _eprint(f"wrote {target}")
    if args.json_out:
        payload = hits.as_dict()
        payload["input"] = [str(source) for source in args.input]
        if report is not None:
            payload["enrichment"] = report
        _write_json(args.json_out, payload)
    return 0


def cmd_pharmacophore_show(args) -> int:
    """Print a model file, as a table or as JSON."""
    from pathlib import Path

    from .cli import _chemistry_guard, _lazy, _write_json

    pharm = _lazy("odock.pharmacophore", "pharmacophore modelling")
    if not Path(args.model).exists():
        raise SystemExit(f"error: no such model file: {args.model}")
    model = _chemistry_guard(pharm.PharmacophoreModel.load, args.model)
    if args.json_out:
        _write_json(args.json_out, model.as_dict())
    print(model.table())
    print()
    print(
        f"{'meaningful' if model.is_meaningful else 'ANECDOTE'}: {model.n_used} of "
        f"{model.n_members} member(s) contributed"
    )
    return 0


def _add_lbvs(subs) -> None:
    """``odock decoys`` and ``odock lbvs``."""
    dec = subs.add_parser(
        "decoys",
        help="select property-matched, topologically dissimilar decoys",
        description=(
            "Pick a decoy set for a labelled benchmark: pool members matched to each "
            "active on MW, LogP, HBD, HBA, rotatable bonds and net charge within "
            "stated tolerances, and not an analogue of any active.  The command "
            "prints the property distributions side by side with their standardised "
            "mean differences, because an unmatched decoy set inflates every "
            "enrichment number."
        ),
    )
    dec.add_argument("-a", "--actives", action="append", required=True, metavar="FILE",
                     help="the actives (a .smi/.sdf/... file); repeat to combine")
    dec.add_argument("-p", "--pool", action="append", required=True, metavar="FILE",
                     help="the pool to select from (demo/libraries/decoys.smi is the bundled one)")
    dec.add_argument("-n", "--per-active", type=int, default=5, help="decoys per active")
    dec.add_argument("--max-similarity", type=float, default=0.35,
                     help="Tanimoto ceiling: at or above this a pool member is an analogue")
    dec.add_argument("--min-similarity", type=float, default=0.0,
                     help="Tanimoto floor: below this a pool member is too easy (the "
                          "hard band starts at ~0.35)")
    dec.add_argument("--mw", "--mw-tolerance", dest="mw_tolerance", type=float, default=25.0,
                     help="MW tolerance (Da)")
    dec.add_argument("--logp-tolerance", type=float, default=0.5)
    dec.add_argument("--hbd-tolerance", type=float, default=1.0)
    dec.add_argument("--hba-tolerance", type=float, default=1.0)
    dec.add_argument("--rotb-tolerance", type=float, default=1.0)
    dec.add_argument("-o", "--out", help="write the decoys as .smi or .sdf")
    dec.add_argument("--json-out", help="write the selection, the rejections and the quality")
    dec.set_defaults(func=cmd_decoys)

    lb = subs.add_parser(
        "lbvs",
        help="benchmark a ligand-based screen (EF1%%, EF5%%, AUC, BEDROC + intervals)",
        description=(
            "Score actives plus decoys with the 2-D fingerprint search, the 3-D "
            "pharmacophore fit and a shape/electrostatic overlay, and report EF1%, "
            "EF5%, AUC and BEDROC(20) with bootstrap intervals — beside a random "
            "ranking and a property-only ranking, so 'no signal' and 'trivial "
            "signal' can be read next to the methods.  Every method is scored "
            "leave-one-out: the actives are their own queries otherwise, and every "
            "metric is then a perfect 1.000 for free.  --shape picks between the two "
            "overlay engines; both are reported when both method names are asked for, "
            "so the delta is visible on the same actives and decoys."
        ),
    )
    lb.add_argument("-a", "--actives", action="append", required=True, metavar="FILE")
    lb.add_argument("-d", "--decoys", action="append", required=True, metavar="FILE",
                    help="the decoy files (`odock decoys -o` writes one)")
    lb.add_argument("--methods", default="fingerprint,pharmacophore,shape",
                    help="comma-separated: fingerprint, pharmacophore, shape, "
                         "shape_only, overlay, overlay_only, crude, crude_only, "
                         "overlay_esp, crude_esp, crude_shape")
    lb.add_argument(
        "--shape", dest="shape_engine", default="overlay", choices=["crude", "overlay"],
        help="the engine the 'shape' method uses: 'overlay' (default) is the "
             "Gaussian shape + electrostatic overlay with an optimised pose; 'crude' "
             "is the older single-pose grid overlay.  On the bundled decoy bands the "
             "crude engine measured the higher AUC — run both with "
             "--methods overlay,crude to see it on your own data",
    )
    lb.add_argument("--conformers", type=int, default=2, help="conformers per molecule")
    lb.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    lb.add_argument("--seed", type=int, default=20240101)
    lb.add_argument(
        "--shape-sigma", type=float, default=0.5,
        help="Gaussian width of the shape density used by the 'overlay' engine (Å)",
    )
    lb.add_argument(
        "--shape-weight", type=float, default=0.5,
        help="weight of the shape term in the overlay score; the rest is the "
             "electrostatic Carbo index (clamped at 0)",
    )
    lb.add_argument("--prefilter", type=float, default=None, metavar="FRACTION",
                    help="also report what a pre-filter at this fraction keeps "
                         "(active recall and the docking workload saved)")
    lb.add_argument(
        "--prefilter-method", default="fingerprint", choices=["fingerprint", "usr"],
        help="the pre-filter: 'fingerprint' (2-D, free) or 'usr' (3-D shape "
             "descriptors; needs conformers, and its active recall is measured and "
             "reported, because it is not free)",
    )
    lb.add_argument("--json-out", help="write the whole report as JSON")
    lb.set_defaults(func=cmd_lbvs)


def cmd_decoys(args) -> int:
    """Select property-matched decoys and report the matching evidence."""
    from .cli import _chemistry_guard, _eprint, _lazy, _read_library, _write_json

    decoys = _lazy("odock.decoys", "decoy selection")
    actives, active_names = _read_library(list(args.actives))
    pool, pool_names = _read_library(list(args.pool))
    selection = _chemistry_guard(
        decoys.match_decoys,
        actives,
        pool,
        per_active=int(args.per_active),
        tolerances={
            "MW": float(args.mw_tolerance),
            "LogP": float(args.logp_tolerance),
            "HBD": float(args.hbd_tolerance),
            "HBA": float(args.hba_tolerance),
            "RotB": float(args.rotb_tolerance),
        },
        max_similarity=float(args.max_similarity),
        min_similarity=float(args.min_similarity),
        names=active_names,
        pool_names=pool_names,
    )
    print(
        f"decoys: {selection.n_decoys} selected from {selection.n_pool} pool member(s) "
        f"for {len(selection.actives)} active(s), {int(args.per_active)} requested each"
    )
    print(selection.table(limit=12 if selection.n_decoys > 12 else 0))
    print()
    print(selection.quality_table())
    for note in selection.notes:
        print(f"  note: {note}")
    if selection.shortfalls:
        _eprint(
            "odock decoys: warning: the pool could not match every active; widen the "
            "pool rather than the tolerances"
        )
    if args.out:
        from pathlib import Path

        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in (".smi", ".smiles"):
            target.write_text(
                "\n".join(f"{decoy.smiles} {decoy.name}" for decoy in selection.decoys) + "\n",
                encoding="utf-8",
            )
        else:
            from rdkit import Chem

            writer = Chem.SDWriter(str(target))
            try:
                for decoy in selection.decoys:
                    mol = Chem.MolFromSmiles(decoy.smiles)
                    if mol is None:  # pragma: no cover - defensive
                        continue
                    mol.SetProp("_Name", decoy.name)
                    mol.SetProp("odock_matched_active", decoy.active)
                    mol.SetProp("odock_property_distance", f"{decoy.distance:.4f}")
                    writer.write(mol)
            finally:
                writer.close()
        _eprint(f"wrote {target}")
    if args.json_out:
        _write_json(args.json_out, selection.as_dict())
    return 0


def cmd_lbvs(args) -> int:
    """Benchmark the ligand-based methods against actives plus decoys."""
    from .cli import _chemistry_guard, _eprint, _lazy, _read_library, _write_json

    lbvs = _lazy("odock.lbvs", "the ligand-based benchmark")
    actives, active_names = _read_library(list(args.actives))
    decoy_mols: List[object] = []
    decoy_names: List[str] = []
    for source in args.decoys:
        found, labels = _read_library([source])
        decoy_mols.extend(found)
        decoy_names.extend(labels)
    methods = [part.strip() for part in str(args.methods).split(",") if part.strip()]
    report = _chemistry_guard(
        lbvs.benchmark,
        actives,
        decoy_mols,
        names=active_names,
        decoy_names=decoy_names,
        methods=methods,
        shape_engine=str(args.shape_engine),
        conformers=int(args.conformers),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    print(report.table())
    for note in report.notes:
        print(f"  note: {note}")
    for result in report.results:
        for note in result.notes:
            print(f"  {result.method}: {note}")
        if result.ranking.details:
            terms = result.ranking.terms()
            print(
                f"  {result.method}: mean shape {terms['mean_shape']:.3f}, mean ESP "
                f"{terms['mean_esp']:+.3f} over {terms['n']} molecule(s) — the terms "
                "are reported separately because the combined score hides which one "
                "did the ranking"
            )

    payload = report.as_dict()
    if args.prefilter is not None:
        from .cli import _chemistry_guard as guard

        summary = guard(
            lbvs.prefilter,
            list(actives) + list(decoy_mols),
            actives,
            keep=float(args.prefilter),
            method=str(args.prefilter_method),
            conformers=int(args.conformers),
            seed=int(args.seed),
        )
        print()
        print(
            f"pre-filter at {float(args.prefilter):.0%} ({summary['method']}): keeps "
            f"{summary['n_kept']} of {summary['n_library']} molecule(s) and "
            f"{summary['actives_kept']} of {summary['n_actives']} active(s) "
            f"({summary['active_recall']:.0%} recall); estimated docking "
            f"{summary['estimated_seconds_full']:.0f} s -> "
            f"{summary['estimated_seconds_kept']:.0f} s "
            f"({summary['workload_saved_fraction']:.0%} saved)"
        )
        if "overlay_seconds_kept" in summary:
            print(
                f"  overlay cost: {summary['overlay_seconds_kept']:.2f} s measured on "
                f"the kept molecule(s) against "
                f"{summary['overlay_seconds_full_estimate']:.1f} s extrapolated to the "
                f"whole library ({summary['overlay_saved_fraction']:.0%} saved); "
                f"{summary['conformers_per_molecule']:.2f} conformer(s) per molecule, "
                f"descriptors {summary['descriptor_seconds']:.2f} s"
            )
            print(
                f"  with self-match allowed the same filter keeps "
                f"{summary['self_match_actives_kept']} of {summary['n_actives']} "
                "active(s) for free — that is the leak, not the recall"
            )
        if summary["actives_lost"]:
            print(f"  actives lost: {', '.join(summary['actives_lost'])}")
        for note in summary["notes"]:
            print(f"  note: {note}")
        payload["prefilter"] = summary
    if args.json_out:
        _write_json(args.json_out, payload)
    else:
        _eprint(
            "odock lbvs: pass --json-out to keep the full rankings; the table above is "
            "the summary"
        )
    return 0


def _add_triage(subs) -> None:
    """``odock triage``: the liability view of a hit list."""
    tr = subs.add_parser(
        "triage",
        help="triage a hit list: structural alerts, drug-likeness, grouped by series",
        description=(
            "What is wrong with these molecules?  Runs the published alert "
            "catalogues (PAINS A/B/C, Brenk, NIH, ZINC) and a documented SMARTS "
            "liability set, reports WHERE each alert sits (scaffold, substituent or "
            "mixed), adds the property panel with the Lipinski, Veber, Egan and "
            "Ghose rules, and groups the whole table by the scaffold series the tool "
            "already computes.  Alerts are literature-derived patterns, not "
            "predictions: absence of an alert is not safety, and none of it "
            "substitutes for an assay."
        ),
    )
    tr.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    tr.add_argument(
        "--affinities",
        help="a screening results.jsonl/results.csv: adds the docked or measured "
             "value to each row and sorts nothing by it (triage is not a ranking)",
    )
    tr.add_argument(
        "--catalogues", default="PAINS,BRENK,NIH,ZINC",
        help="comma-separated catalogues: PAINS, PAINS_A, PAINS_B, PAINS_C, BRENK, "
             "NIH, ZINC (or 'none' for the SMARTS alerts alone)",
    )
    tr.add_argument("--no-smarts", action="store_true", help="skip the embedded SMARTS alerts")
    tr.add_argument(
        "--clean-only", action="store_true", help="print only the molecules with no alert"
    )
    tr.add_argument(
        "--flagged-only", action="store_true", help="print only the molecules with an alert"
    )
    tr.add_argument("--top", type=int, default=0, help="show at most N molecules (0 = all)")
    tr.add_argument("-o", "--out", help="write the per-molecule table as CSV or XLSX")
    tr.add_argument("--json-out", help="write the whole triage report as JSON")
    tr.set_defaults(func=cmd_triage)


def cmd_triage(args) -> int:
    """Run the triage and print the series view, the rates and the table."""
    import csv
    from pathlib import Path

    from .cli import _chemistry_guard, _eprint, _lazy, _load_affinities, _read_library, _write_json

    triage = _lazy("odock.triage", "hit triage")
    mols, names = _read_library(list(args.input))
    affinities = _load_affinities(args.affinities) if args.affinities else {}

    wanted = [part.strip() for part in str(args.catalogues).split(",") if part.strip()]
    catalogues = [] if wanted and wanted[0].lower() in ("none", "-") else wanted
    report = _chemistry_guard(
        triage.triage_library,
        mols,
        names=names,
        affinities=affinities,
        catalogues=catalogues,
        smarts=not args.no_smarts,
    )

    print(
        f"triage: {report.n_molecules} molecule(s), {report.n_flagged} with at least one "
        f"alert, {report.n_clean} clean"
    )
    print()
    print(
        "alert rates (a catalogue that flags a large fraction is telling you about "
        "the catalogue):"
    )
    print(report.rates_table())
    print()
    print(f"series: {len(report.groups)} group(s) by Murcko scaffold")
    print(report.series_table())

    rows = report.molecules
    if args.clean_only:
        rows = [molecule for molecule in rows if molecule.clean]
    if args.flagged_only:
        rows = [molecule for molecule in rows if not molecule.clean]
    if args.top and int(args.top) > 0:
        rows = rows[: int(args.top)]
    print()
    print(f"molecules ({len(rows)} shown):")
    subset = triage.TriageReport(
        molecules=rows,
        groups=report.groups,
        rates=report.rates,
        catalogues=report.catalogues,
        seconds=report.seconds,
        notes=report.notes,
    )
    print(subset.table())
    print()
    for note in report.notes:
        print(f"  note: {note}")

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "name", "affinity", "clean", "scaffold", "n_alerts", "alerts", "alert_locations",
            "failed_rules", "MW", "LogP", "TPSA", "HBD", "HBA", "RotB", "QED", "SA_score",
        ]
        rows_out = [_triage_row(molecule) for molecule in rows]
        if target.suffix.lower() in (".xlsx", ".xlsm"):
            try:
                from openpyxl import Workbook
            except Exception:  # pragma: no cover - openpyxl is a test dependency
                _eprint("odock triage: openpyxl is unavailable; writing CSV instead")
                target = target.with_suffix(".csv")
            else:
                workbook = Workbook()
                sheet = workbook.active
                sheet.title = "triage"
                sheet.append(header)
                for row in rows_out:
                    sheet.append(row)
                workbook.save(target)
                rows_out = None
        if rows_out is not None:
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows_out)
        _eprint(f"wrote {target}")
    if args.json_out:
        payload = report.as_dict()
        payload["shown"] = len(rows)
        _write_json(args.json_out, payload)
    return 0


def _triage_row(molecule) -> list:
    """One CSV/XLSX row for a triaged molecule."""
    values = (molecule.properties.values if molecule.properties else {}) or {}
    return [
        molecule.name,
        "" if molecule.affinity is None else round(float(molecule.affinity), 4),
        "1" if molecule.clean else "0",
        molecule.scaffold,
        len(molecule.alerts),
        "; ".join(alert.label() for alert in molecule.alerts),
        "; ".join(f"{alert.origin}:{alert.location}" for alert in molecule.alerts),
        "; ".join(molecule.failed_rules),
        _num_or_blank(values.get("MW")),
        _num_or_blank(values.get("LogP")),
        _num_or_blank(values.get("TPSA")),
        _num_or_blank(values.get("HBD")),
        _num_or_blank(values.get("HBA")),
        _num_or_blank(values.get("RotB")),
        _num_or_blank(values.get("QED")),
        _num_or_blank(values.get("SA_score")),
    ]


def _num_or_blank(value):
    if value is None:
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""


def _add_conformers(subs) -> None:
    """``odock conformers``: report the ensemble quality behind a 3-D score."""
    cf = subs.add_parser(
        "conformers",
        help="measure a library's conformer ensembles (RMSD, energy, torsion coverage)",
        description=(
            "Every 3-D ligand-based score in this tool — the pharmacophore fit, the "
            "shape/electrostatic overlay, the 3-D similarity search — rests on a "
            "conformer ensemble, and this command reports what that ensemble is: how "
            "many conformers survived the RMSD pruning and the energy window, how far "
            "apart they are, what fraction of the rotamer space they visit, and how "
            "long the embedding took.  A molecule that keeps one conformer scores "
            "1.000 against itself and says nothing about its flexibility; the report "
            "says which molecules those are."
        ),
    )
    cf.add_argument(
        "-i", "--input", action="append", required=True, metavar="FILE",
        help="library file (.sdf/.smi/.mol2/.pdb/.pdbqt); repeat to combine several",
    )
    cf.add_argument(
        "--n-conformers", type=int, default=None,
        help="embedding attempts per molecule; omit for the rotor-scaled default "
             "(8 per rotatable bond, at least 16, at most 128), or give an explicit "
             "count — which is honoured exactly, so a published protocol is "
             "reproducible",
    )
    cf.add_argument("--seed", type=int, default=20240101, help="ETKDGv3 embedding seed")
    cf.add_argument(
        "--rmsd-prune", type=float, default=0.5,
        help="heavy-atom RMSD above which two conformers are kept as different (Å)",
    )
    cf.add_argument(
        "--energy-window", type=float, default=10.0,
        help="keep conformers within this many kcal/mol of the best (MMFF94 or UFF)",
    )
    cf.add_argument(
        "--fixed-attempts", action="store_true",
        help="with the default attempt count, use the old fixed 16 attempts per "
             "molecule instead of scaling with the rotatable-bond count",
    )
    cf.add_argument(
        "--keep-input", action="store_true",
        help="report the input coordinates as they are instead of re-embedding; a "
             "docked pose or a crystal structure must not be silently replaced",
    )
    cf.add_argument("--top", type=int, default=0, help="show at most N molecules (0 = all)")
    cf.add_argument("--json-out", help="write the per-molecule measurements as JSON")
    cf.set_defaults(func=cmd_conformers)


def cmd_conformers(args) -> int:
    """Embed a library and print the ensemble measurements, per molecule."""
    from .cli import _chemistry_guard, _eprint, _lazy, _read_library, _write_json

    conformers = _lazy("odock.conformers", "conformer generation")
    mols, names = _read_library(list(args.input))
    with_coordinates = sum(1 for mol in mols if mol.GetNumConformers() > 0)
    if args.keep_input and with_coordinates < len(mols):
        _eprint(
            f"odock conformers: --keep-input, but only {with_coordinates} of "
            f"{len(mols)} molecule(s) carry 3-D coordinates; the rest are embedded "
            "because a SMILES has no pose to keep"
        )
    ensembles = []
    for mol, name in zip(mols, names):
        ensembles.append(
            _chemistry_guard(
                conformers.build_ensemble,
                mol,
                n_conformers=None if args.n_conformers is None else int(args.n_conformers),
                seed=int(args.seed),
                rmsd_prune=float(args.rmsd_prune),
                energy_window=float(args.energy_window),
                prune=True,
                minimize=True,
                name=name,
                scale_with_rotors=not args.fixed_attempts,
                use_input_conformers=bool(args.keep_input),
            )
        )
    quality = conformers.ensemble_quality(ensembles)
    print(
        f"conformers: {quality['n_embedded']} of {quality['n_molecules']} molecule(s) "
        f"embedded ({quality['embed_rate']:.0%}), "
        f"{quality['conformers_mean']:.2f} conformer(s) kept on average "
        f"(median {quality['conformers_median']:.0f}, min {quality['conformers_min']}) "
        f"from {quality['attempts_per_molecule']:.1f} attempt(s) per molecule, "
        f"{quality['seconds_per_molecule']:.3f} s per molecule"
    )
    print(
        "  accounting: attempts -> embedded -> (-energy, -RMSD) -> kept: "
        f"{quality['attempts_total']} -> {quality['embedded_total']} -> "
        f"(-{quality['losses']['energy_window']}, -{quality['losses']['rmsd_prune']}) "
        f"-> {sum(item.n_conformers for item in ensembles)}"
        f"   (dominant loss: {quality['dominant_loss']})"
    )
    print(
        f"  mean torsion coverage {quality['torsion_coverage_mean']:.0%} of its "
        f"{quality['coverage_ceiling_mean']:.0%} ceiling "
        f"({quality['coverage_fraction_of_ceiling']:.0%} of what the conformer count "
        "allows; k conformers can occupy at most k of a bond's six rotamer bins)"
    )
    print(
        f"  mean RMSD spread {quality['rmsd_spread_mean']:.2f} Å, "
        f"{quality['rotors_mean']:.1f} rotatable bond(s) per molecule, "
        f"{quality['scaled_attempts']} molecule(s) on the rotor-scaled rule"
    )
    print()
    rows = ensembles
    if args.top and int(args.top) > 0:
        rows = rows[: int(args.top)]
    if rows:
        print(
            f"{'molecule':<28}{'rot':>3}{'att':>6}{'emb':>6}{'Ecut':>7}{'Rcut':>7}"
            f"{'kept':>6}{'cov':>8}{'ceil':>8}{'s':>8}"
        )
    for ensemble in rows:
        print(ensemble.accounting_line())
        print(ensemble.table())
    if args.top and int(args.top) > 0 and len(ensembles) > int(args.top):
        print(f"... and {len(ensembles) - int(args.top)} more molecule(s)")
    if quality["failed"]:
        _eprint(
            f"odock conformers: {len(quality['failed'])} molecule(s) could not be "
            "embedded; every 3-D score for them is 0, not a low score"
        )
    print()
    print(
        "note: an ensemble is a sample of a modelled space at one seed, not the "
        "space itself — and a rigid molecule's ensemble is one geometry, which the "
        "torsion-coverage column shows as an empty one"
    )
    if args.json_out:
        payload = dict(quality)
        payload["input"] = [str(source) for source in args.input]
        payload["n_conformers_requested"] = (
            None if args.n_conformers is None else int(args.n_conformers)
        )
        payload["rotor_scaled"] = not args.fixed_attempts
        payload["seed"] = int(args.seed)
        payload["rmsd_prune"] = float(args.rmsd_prune)
        payload["energy_window"] = float(args.energy_window)
        payload["accounting"] = [item.accounting() for item in ensembles]
        payload["ensembles"] = [item.as_dict() for item in ensembles]
        _write_json(args.json_out, payload)
    return 0


def _add_pocket_score(subs) -> None:
    """``odock pocket-score``: rank a library against the receptor's pocket."""
    ps = subs.add_parser(
        "pocket-score",
        help="rank a library by shape and electrostatic complementarity with a pocket",
        description=(
            "Score each library molecule against the receptor itself rather than "
            "against another ligand: the pocket's shape field and its electrostatic "
            "potential are built once from the receptor and the docking box, and every "
            "conformer is placed in the pocket by a rigid-body search and scored by "
            "looking the field up at its atoms.  The score is shape complementarity "
            "(a snug-contact kernel, a clash penalty, and a size term against a "
            "reference ligand when one is given) plus the ligand-receptor Coulomb "
            "interaction.  This is a fast screen, not a docking: the receptor does not "
            "move and a complementarity is not a binding energy."
        ),
    )
    ps.add_argument("-r", "--receptor", required=True,
                    help="the receptor PDBQT (its own charges are used)")
    ps.add_argument("-i", "--input", action="append", required=True, metavar="FILE",
                    help="the library (.smi/.sdf/...); repeat to combine several")
    ps.add_argument("-b", "--box", help="a box.json with center/size (demo/systems/3ptb/box.json)")
    ps.add_argument("--center", help="box centre as x,y,z (an alternative to --box)")
    ps.add_argument("--size", help="box size as x,y,z (an alternative to --box)")
    ps.add_argument("--spacing", type=float, default=None,
                    help="pocket grid spacing in Å (default 0.8; 0.5 is finer and "
                         "about 4x slower to build)")
    ps.add_argument(
        "--reference-ligand",
        help="a pose to calibrate the size term on (the co-crystallised ligand): its "
             "van der Waals volume becomes the size the pocket is known to accept. "
             "Without it the size term is off and the score saturates",
    )
    ps.add_argument("--size-reference", type=float, default=None,
                    help="the size reference in Å³, instead of --reference-ligand")
    ps.add_argument("--conformers", type=int, default=None,
                    help="embedding attempts per molecule (default: rotor-scaled)")
    ps.add_argument("--poses", type=int, default=64,
                    help="start rotations per conformer in the placement search")
    ps.add_argument("--top", type=int, default=0, help="show at most N molecules")
    ps.add_argument("--keep", type=float, default=None, metavar="FRACTION",
                    help="also report what a pocket-score pre-filter at this fraction "
                         "keeps, with the recall of the actives named by --actives")
    ps.add_argument("--actives", action="append", metavar="NAME",
                    help="a known binder's name, for the --keep recall (repeatable)")
    ps.add_argument("--no-electrostatics", action="store_true",
                    help="shape only (the ESP term on these charges can mislead: see "
                         "docs/POCKET_SCORE.md)")
    ps.add_argument("--seed", type=int, default=20240101)
    ps.add_argument("--json-out", help="write the field, the ranking and the recall")
    ps.set_defaults(func=cmd_pocket_score)


def cmd_pocket_score(args) -> int:
    """Build the pocket field, rank the library, and report what the cut costs."""
    from .cli import _chemistry_guard, _eprint, _lazy, _read_library, _write_json

    surface = _lazy("odock.pocket_score", "pocket scoring")
    conformers = _lazy("odock.conformers", "conformer generation")
    import json as _json
    from pathlib import Path

    from . import sasa as _sasa

    if args.box:
        payload = _json.loads(Path(args.box).read_text(encoding="utf-8"))
        center = payload["center"]
        size = payload["size"]
        spacing = float(args.spacing or surface.DEFAULT_SPACING)
    else:
        if not args.center or not args.size:
            raise SystemExit("error: give --box, or both --center and --size")
        center = [float(part) for part in str(args.center).split(",")]
        size = [float(part) for part in str(args.size).split(",")]
        spacing = float(args.spacing or 0.8)

    atoms = surface.read_pdbqt_atoms(args.receptor)
    if not atoms:
        raise SystemExit(f"error: no receptor atom in {args.receptor}")
    pocket = _chemistry_guard(
        surface.PocketField.build, atoms, center=center, size=size, spacing=spacing
    )
    print(
        f"pocket: {pocket.n_voxels} voxel(s) at {pocket.spacing} A over "
        f"{pocket.volume:.0f} A3 ({pocket.open_volume:.0f} A3 open to a ligand "
        f"centre), built from {pocket.n_atoms} receptor atom(s) in {pocket.seconds:.2f} s"
    )
    for note in pocket.notes:
        print(f"  note: {note}")

    reference_volume = None
    if args.reference_ligand:
        reference_atoms = surface.read_pdbqt_atoms(args.reference_ligand)
        reference_volume = surface.ligand_volume(
            [_sasa.radius_of(atom.element) for atom in reference_atoms]
        )
    elif args.size_reference is not None:
        reference_volume = float(args.size_reference)
    if reference_volume:
        pocket.set_size_reference(
            reference_volume,
            note=f"size reference: {reference_volume:.0f} A3, the ligand this pocket "
                 "is known to bind",
        )
        print(f"  size reference {reference_volume:.0f} A3 "
              f"({reference_volume / max(pocket.open_volume, 1e-9):.1%} of the open volume)")
    else:
        _eprint(
            "odock pocket-score: no --reference-ligand and no --size-reference, so the "
            "size term is off; the contact term then saturates at ~1.0 for every "
            "molecule and the ranking carries almost no information"
        )

    mols, names = _read_library(list(args.input))
    ranking = _chemistry_guard(
        surface.rank_library, mols, pocket,
        conformers=None if args.conformers is None else int(args.conformers),
        names=names, samples=int(args.poses), electrostatic=not args.no_electrostatics,
        seed=int(args.seed),
    )
    print()
    print(ranking.table(limit=int(args.top or 0)))
    print()
    print(
        f"ranked {len(ranking)} molecule(s) in {ranking.seconds:.2f} s "
        f"({ranking.n_poses} pose(s) evaluated, "
        f"{ranking.seconds / max(1, ranking.n_poses) * 1e6:.0f} us/pose)"
    )
    shape_only = [
        name for name, _ in ranking.entries
        if name in ranking.details and ranking.details[name].is_shape_only
    ]
    if shape_only:
        _eprint(
            f"odock pocket-score: WARNING: the electrostatic term was clamped to zero "
            f"for {len(shape_only)} of {len(ranking)} molecule(s) "
            f"(e.g. {', '.join(shape_only[:3])}), so for those the reported score is the "
            "SHAPE term alone.  A formally charged group left neutral, or a Gasteiger "
            "charge on a charged group, both make the interaction energy positive in an "
            "oppositely charged pocket; re-run with --reference-ligand and check the "
            "protonation state.  See docs/PROTONATION.md"
        )
    for note in ranking.notes:
        print(f"  note: {note}")

    payload = {
        "pocket": pocket.as_dict(),
        "ranking": ranking.as_dict(),
        "reference_volume": reference_volume,
    }
    if args.keep is not None:
        actives = list(args.actives or [])
        summary = _chemistry_guard(
            surface.prefilter, mols, pocket, actives, keep=float(args.keep),
            conformers=None if args.conformers is None else int(args.conformers),
            names=names, samples=int(args.poses),
            electrostatic=not args.no_electrostatics, seed=int(args.seed),
        )
        print()
        print(
            f"pre-filter at {float(args.keep):.0%}: keeps {summary['n_kept']} of "
            f"{summary['n_library']} molecule(s) and {summary['actives_kept']} of "
            f"{summary['n_actives']} named binder(s) "
            f"({summary['active_recall']:.0%} recall); estimated docking "
            f"{summary['estimated_seconds_full']:.0f} s -> "
            f"{summary['estimated_seconds_kept']:.0f} s "
            f"({summary['workload_saved_fraction']:.0%} saved)"
        )
        if summary["actives_lost"]:
            print(f"  binders lost: {', '.join(summary['actives_lost'])}")
        for note in summary["notes"]:
            print(f"  note: {note}")
        payload["prefilter"] = summary
    if args.json_out:
        _write_json(args.json_out, payload)
    return 0


def attach_ligand_chemistry(subparsers) -> None:
    """Register every ligand-chemistry command on ``subparsers``.

    The one call :mod:`odock.cli_ext` makes: it adds the four similarity/diversity
    subcommands, the ``pharmacophore`` group and the screening pipeline's
    ``--diverse`` options.  It is idempotent — calling it twice replaces nothing,
    so a caller that re-builds a parser cannot end up with two ``similar``
    commands.
    """
    command_functions()
    existing = set(getattr(subparsers, "choices", {}) or {})
    for name, add in (
        ("similar", _add_similar),
        ("diverse", _add_diverse),
        ("scaffolds", _add_scaffolds),
        ("rgroups", _add_rgroups),
        ("pharmacophore", _add_pharmacophore),
        ("decoys", _add_lbvs),
        ("triage", _add_triage),
        ("conformers", _add_conformers),
        ("pocket-score", _add_pocket_score),
    ):
        if name in existing:
            continue
        add(subparsers)
    screen_parser = getattr(subparsers, "choices", {}).get("screen")
    if screen_parser is not None:
        add_screen_diversity_options(screen_parser)

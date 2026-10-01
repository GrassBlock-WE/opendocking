# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock tutorial``: the whole toolchain on the bundled data, measured.

A new user has to get from nothing to a result, and the failure mode of every
tutorial is drift: it says "you should see about nine poses" and the code now
returns seven.  So this is not prose with numbers pasted into it — it **runs** the
chain on the bundled 3PTB data and records what it measured, and the documentation
site renders that record.  The page cannot disagree with the code, because the page
*is* the run.

The chain, each step asserted for shape (not for a value, which is what the record
is for):

1. prepare the receptor from ``tests/data/3PTB.pdb`` → a PDBQT
2. build and prepare the ligand from ``demo/library.smi``
3. derive the search box from the ligand
4. dock (a small, seeded run)
5. rank the poses and cluster them
6. profile the receptor–ligand interactions
7. save a project and verify it reproduces
8. export the HTML report and check it is self-contained

Run it with ``odock tutorial`` (or ``python -m odock.tutorial``); ``--json`` gives
the record, ``--out DIR`` chooses the scratch directory.  The whole thing takes
about twenty seconds, and it writes only under ``out/``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["TUTORIAL_FORMAT", "TUTORIAL_VERSION", "TutorialError", "TutorialRecord",
           "main", "run_tutorial", "add_tutorial_parser", "cmd_tutorial"]

TUTORIAL_FORMAT = "odock-tutorial"
TUTORIAL_VERSION = 1

#: The small, seeded run the tutorial uses.  Exhaustiveness 2 and three poses keep
#: the tutorial at tutorial speed; the point is the chain, not the sampling.
EXHAUSTIVENESS = 2
NUM_POSES = 3
SEED = 42


class TutorialError(RuntimeError):
    """A step failed, or its output changed shape."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return super().__str__() + (f"\n  fix: {self.fix}" if self.fix else "")


@dataclass
class Step:
    """One narrated step and what it measured."""

    name: str
    detail: str
    numbers: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    ok: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "detail": self.detail,
            "numbers": self.numbers,
            "seconds": round(float(self.seconds), 2),
            "ok": bool(self.ok),
        }


@dataclass
class TutorialRecord:
    """The whole walkthrough, as data.  The site page is rendered from this."""

    steps: List[Step] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    seconds: float = 0.0
    demo: str = ""
    run_dir: str = ""

    def add(self, step: Step) -> Step:
        self.steps.append(step)
        return step

    @property
    def totals(self) -> Dict[str, Any]:
        return {
            "steps": len(self.steps),
            "failed": sum(1 for step in self.steps if not step.ok),
            "seconds": round(float(self.seconds), 2),
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": TUTORIAL_FORMAT,
            "version": TUTORIAL_VERSION,
            "ok": bool(self.ok),
            "error": self.error,
            "demo": self.demo,
            "run_dir": self.run_dir,
            "totals": self.totals,
            "steps": [step.as_dict() for step in self.steps],
        }

    def text(self) -> str:
        lines = [
            f"odock tutorial — the toolchain on the bundled {self.demo} data"
            + (" [FAILED]" if not self.ok else ""),
        ]
        for index, step in enumerate(self.steps, 1):
            mark = "ok  " if step.ok else "FAIL"
            lines.append(f"  [{mark}] {index}. {step.name} ({step.seconds:.1f} s)")
            lines.append(f"         {step.detail}")
            if step.numbers:
                lines.append(
                    "         "
                    + ", ".join(f"{key}={value}" for key, value in step.numbers.items())
                )
        lines.append(f"  total: {self.seconds:.1f} s, {len(self.steps)} step(s)")
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)


def _check(condition: bool, message: str, *, fix: str = "") -> None:
    if not condition:
        raise TutorialError(message, fix=fix)


def _report_numbers(report: Any, limit: int = 8) -> Dict[str, Any]:
    """The numeric fields a preparation report carries, whatever they are called.

    Read defensively on purpose: the preparation layer owns its report's shape, and
    the tutorial records what it finds rather than pinning a name that a refactor
    would break.
    """
    numbers: Dict[str, Any] = {}
    source = getattr(report, "__dict__", None) or {}
    for key, value in source.items():
        if key.startswith("_") or len(numbers) >= limit:
            continue
        if isinstance(value, bool):
            numbers[key] = value
        elif isinstance(value, int):
            numbers[key] = value
        elif isinstance(value, float):
            numbers[key] = round(value, 2)
        elif isinstance(value, (list, tuple)):
            numbers[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            numbers[f"{key}_count"] = len(value)
    return numbers


def run_tutorial(root=None, *, workdir=None, quiet: bool = True) -> TutorialRecord:
    """Run the chain on the bundled data and return the measured record.

    Every step asserts the **shape** of what it produced (a result with poses, a
    file that exists, a report that references nothing external).  It does not
    assert a score: the numbers are recorded so a reader — and the docs guard — can
    see when they move.
    """
    import odock
    from odock import analysis, htmlreport, prepare, project

    base = Path(root).resolve() if root is not None else _default_root()
    demo = base / "demo" / "3ptb"
    receptor_source = base / "tests" / "data" / "3PTB.pdb"
    library = base / "demo" / "library.smi"
    _check(receptor_source.is_file(), f"the bundled receptor is missing: {receptor_source}",
           fix="run `make demo` (the fixtures are generated, not committed)")
    _check(library.is_file(), f"the bundled ligand library is missing: {library}",
           fix="run `make demo`")

    run_dir = Path(workdir).resolve() if workdir is not None else base / "out" / "tutorial"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    record = TutorialRecord(demo="3PTB", run_dir=str(run_dir))
    started = time.perf_counter()

    # 1. the receptor -----------------------------------------------------------------
    step_started = time.perf_counter()
    receptor_pdbqt = run_dir / "receptor.pdbqt"
    # `prepare_*` return `(mol, text, report)`: the molecule is what the later steps
    # need (the interaction profile), and the file is written on the way.
    receptor_mol, _, receptor_report = prepare.prepare_receptor(
        receptor_source, receptor_pdbqt
    )
    _check(receptor_pdbqt.is_file(), "preparing the receptor produced no file")
    _check(receptor_mol is not None, "preparing the receptor returned no molecule")
    atoms = sum(1 for line in receptor_pdbqt.read_text(encoding="utf-8").splitlines()
                if line.startswith(("ATOM", "HETATM")))
    _check(atoms > 0, "the prepared receptor has no atoms")
    record.add(Step(
        "Prepare the receptor",
        f"{receptor_source.name} → {receptor_pdbqt.name}: waters dropped, hydrogens "
        "and charges added by the preparation layer.",
        numbers={"atoms": atoms, "bytes": receptor_pdbqt.stat().st_size,
                 **_report_numbers(receptor_report)},
        seconds=time.perf_counter() - step_started,
    ))

    # 2. the ligand -------------------------------------------------------------------
    step_started = time.perf_counter()
    # The library is `<smiles> <name>` per line (see its own header); the SMILES is
    # the first field.  Reading the last field is how this step failed the first
    # time — it fed the rdkit parser the word "benzamidine".
    smiles, ligand_name = "", ""
    for line in library.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        smiles = parts[0]
        ligand_name = parts[1] if len(parts) > 1 else "tutorial"
        break
    _check(bool(smiles), f"no SMILES could be read from {library}")
    ligand_mol, _, ligand_report = prepare.prepare_ligand(
        smiles, run_dir / "ligand.pdbqt", name=ligand_name
    )
    ligand_pdbqt = run_dir / "ligand.pdbqt"
    _check(ligand_pdbqt.is_file(), "preparing the ligand produced no file")
    _check(ligand_mol is not None, "preparing the ligand returned no molecule")
    record.add(Step(
        "Build the ligand",
        f"the first molecule in {library.name} ({ligand_name}: {smiles}) → "
        f"{ligand_pdbqt.name}, with hydrogens added and a torsion tree written.",
        numbers={"smiles": smiles, "name": ligand_name,
                 "bytes": ligand_pdbqt.stat().st_size,
                 **_report_numbers(ligand_report)},
        seconds=time.perf_counter() - step_started,
    ))

    # 3. the box ----------------------------------------------------------------------
    step_started = time.perf_counter()
    box = prepare.box_from_ligand(ligand_mol, buffer=5.0)
    centre = getattr(box, "center", None) or getattr(box, "centre", None)
    size = getattr(box, "size", None)
    _check(box is not None and size is not None, "the box has no size")
    record.add(Step(
        "Choose the search box",
        "derive it from the ligand: the coordinates a docking run may explore.",
        numbers={"buffer": 5.0,
                 "size": [round(float(value), 2) for value in (size or [])],
                 "spacing": round(float(getattr(box, "spacing", 0.375)), 3)},
        seconds=time.perf_counter() - step_started,
    ))

    # 4. dock -------------------------------------------------------------------------
    step_started = time.perf_counter()
    result = odock.dock(receptor_pdbqt, ligand_pdbqt, box,
                        exhaustiveness=EXHAUSTIVENESS, num_poses=NUM_POSES, seed=SEED)
    _check(len(result.poses) > 0, "the docking run returned no pose",
           fix="check that the box covers the ligand and the receptor parsed")
    best = result.best_affinity
    _check(best is not None and best == best, "the best affinity is not a number")
    affinities = [round(float(getattr(pose, "affinity", float("nan"))), 3)
                  for pose in result.poses]
    record.add(Step(
        "Dock",
        f"{result.scoring} scoring, exhaustiveness {EXHAUSTIVENESS}, seed {SEED}: the "
        "kernel fills a grid over the box and searches it.",
        numbers={"poses": len(result.poses), "best_affinity": round(float(best), 3),
                 "affinities": affinities, "grid_points": int(result.grid_points),
                 "grid_mb": round(float(result.grid_mb), 2),
                 "movable_atoms": int(result.num_movable_atoms),
                 "torsions": int(result.num_tors), "seconds_reported": round(float(result.elapsed), 2)},
        seconds=time.perf_counter() - step_started,
    ))

    # 5. rank and cluster -------------------------------------------------------------
    step_started = time.perf_counter()
    ranked = sorted(result.poses, key=lambda pose: float(pose.affinity))
    clusters = analysis.cluster_poses(
        [pose.coords for pose in result.poses], cutoff=2.0,
        energies=[float(pose.affinity) for pose in result.poses],
    )
    _check(len(clusters) > 0, "clustering returned nothing")
    populations = [
        len(getattr(cluster, "members", None) or getattr(cluster, "poses", None) or [])
        for cluster in clusters
    ]
    record.add(Step(
        "Read the ranking",
        "poses sorted by affinity, then clustered by symmetry-aware RMSD: one cluster "
        "per binding mode, which is what a reader actually compares.",
        numbers={"ranked_first": round(float(ranked[0].affinity), 3),
                 "ranked_last": round(float(ranked[-1].affinity), 3),
                 "clusters": len(clusters),
                 "populations": populations,
                 "atoms_per_pose": int(ranked[0].num_atoms)},
        seconds=time.perf_counter() - step_started,
    ))

    # 6. interactions -----------------------------------------------------------------
    step_started = time.perf_counter()
    # The docked pose, not the ligand's input geometry: profiling the input molecule
    # against the receptor reported **0 interactions** the first time this ran, which
    # is a number that looks like a result.
    docked_mol = odock.pose_to_mol(
        ranked[0], ligand_mol, getattr(ligand_report, "atom_order", None) or []
    )
    _check(docked_mol is not None, "the best pose could not be converted to a molecule")
    interactions = analysis.profile_interactions(receptor_mol, docked_mol)
    _check(isinstance(interactions, list), "the interaction profile is not a list")
    key_residues = analysis.interaction_summary(interactions, receptor_mol, docked_mol)
    counts: Dict[str, int] = {}
    for interaction in interactions:
        kind = str(getattr(interaction, "kind", "?"))
        counts[kind] = counts.get(kind, 0) + 1
    _check(isinstance(key_residues, str), "the interaction summary is not text")
    record.add(Step(
        "Look at the interactions",
        "every non-covalent contact the **best pose** makes, and the residues that "
        "carry them — the chemistry a score alone does not show.",
        numbers={"interactions": len(interactions), "kinds": len(counts),
                 "by_kind": counts,
                 "key_residues": key_residues or "(none within the cutoffs)"},
        seconds=time.perf_counter() - step_started,
    ))

    # 7. a project, and its reproduction ----------------------------------------------
    step_started = time.perf_counter()
    bundle = run_dir / "tutorial.odockproj"
    saved = project.save_project(
        bundle, result,
        receptor=receptor_pdbqt, ligand=ligand_pdbqt, box=box,
        original_inputs={"receptor": receptor_source, "ligand": library},
        command="odock tutorial",
        operator="odock tutorial",
    )
    _check(Path(bundle).exists(), "saving the project produced no file")
    verification = project.verify_project(bundle)
    _check(bool(getattr(verification, "ok", False)), "the project did not verify",
           fix="a project that does not verify is a defect in the bundle, not in the run")
    record.add(Step(
        "Save it as a project",
        "the inputs, the box, the poses and the analysis in one container, then read "
        "back and verified: this is what makes a result reproducible.",
        numbers={"bytes": Path(bundle).stat().st_size,
                 "entries": int(getattr(verification, "entries", 0) or 0),
                 "checked": int(getattr(verification, "checked", 0) or 0),
                 "schema_version": int(getattr(verification, "schema_version", 0) or 0),
                 "problems": len(getattr(verification, "problems", []) or []),
                 "verified": bool(getattr(verification, "ok", False)),
                 "operator": getattr(saved, "operator", "") or "odock tutorial"},
        seconds=time.perf_counter() - step_started,
    ))

    # 8. the report -------------------------------------------------------------------
    step_started = time.perf_counter()
    files = htmlreport.write_html_report(run_dir / "tutorial.html", project=bundle)
    html = Path(files.path).read_text(encoding="utf-8")
    external = htmlreport.find_external_references(html)
    _check(not external, f"the report references the network: {external[:2]}")
    record.add(Step(
        "Export the report",
        "one self-contained HTML file: the tables and figures are inside it, so it "
        "can be attached to an email or an issue.",
        numbers={"bytes": Path(files.path).stat().st_size,
                 "figures": html.count("<svg") + html.count("<img"),
                 "external_references": len(external),
                 "json_bytes": Path(files.json_path).stat().st_size
                 if getattr(files, "json_path", None) else 0},
        seconds=time.perf_counter() - step_started,
    ))

    record.seconds = time.perf_counter() - started
    return record


def _default_root() -> Path:
    from .doctor import _default_root as doctor_root

    found = doctor_root()
    return found if found is not None else Path.cwd()


def cmd_tutorial(args) -> int:
    try:
        record = run_tutorial(args.root or _default_root(), workdir=args.out)
    except TutorialError as exc:
        print(f"odock tutorial: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(record.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(record.text())
    return 0 if record.ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m odock.tutorial``: the same thing without the CLI."""
    import argparse

    parser = argparse.ArgumentParser(prog="odock tutorial", description=__doc__)
    parser.add_argument("--root", help="the repository root (default: this checkout)")
    parser.add_argument("--out", help="the scratch directory (default: out/tutorial)")
    parser.add_argument("--json", action="store_true")
    return cmd_tutorial(parser.parse_args(list(argv) if argv is not None else None))


def add_tutorial_parser(sub) -> None:
    """Register ``odock tutorial`` on `sub` (idempotent)."""
    choices = getattr(sub, "choices", None)
    if isinstance(choices, dict) and "tutorial" in choices:
        return
    parser = sub.add_parser(
        "tutorial",
        help="walk the whole toolchain on the bundled demo data, with measured numbers",
        description=(
            "Prepare a receptor and a ligand, choose a box, dock, read the ranking, "
            "profile the interactions, save a project, verify it reproduces and export "
            "a report — on the bundled 3PTB data, in about twenty seconds, printing "
            "what each step measured.  The documentation site renders this same record, "
            "so the tutorial cannot drift from the code."
        ),
    )
    parser.add_argument("--root", help="the repository root (default: this checkout)")
    parser.add_argument("--out", help="the scratch directory (default: out/tutorial)")
    parser.add_argument("--json", action="store_true", help="the record as JSON")
    parser.set_defaults(func=cmd_tutorial)


if __name__ == "__main__":  # pragma: no cover - a thin wrapper
    raise SystemExit(main())

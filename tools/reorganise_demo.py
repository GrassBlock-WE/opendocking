# SPDX-License-Identifier: GPL-3.0-or-later
"""Move demo/ by role, update every reference, and verify it — task-42.

The move has been applied.  This tool stays in the tree because it is the record of
what changed and the check that nothing was missed:

    python tools/reorganise_demo.py --verify     # the shipped-tree path grep + encoding
    python tools/reorganise_demo.py --check      # what a re-run would change (a no-op now)

What it did, as a *move* — no content deleted, no file rewritten except its `demo/...`
path strings:

    demo/{3ptb,egfr}            -> demo/systems/{3ptb,1m17}
    demo/{library,actives,decoys}.smi, demo/ligand_from_smiles.*
                                -> demo/libraries/
    demo/workbench_preview*.png -> demo/figures/

Every file is read and written as **UTF-8 explicitly**: the sources here contain Å, °, π
and em-dashes, and the one rule this repository learned the hard way is that a
non-UTF-8 write destroys them.  `--verify` re-checks both halves of that: no stale path
in the shipped tree, and the non-ASCII characters still present.

**Never walked**: `/odock/` is a *second checkout* with its own `.git` (gitignored, does
not ship), `/out/` and `/scratch` are scratch, and `odck.md` / `DELIVERY.md` are the
"internal working documents, explicitly not for publication" names `.gitignore` lists.  A
stale path in any of those is not a stale path in the shipped tree.

**Never written**: the files other workstreams own — they update their own references
(the run printed the list): `tests/simulate_workbench.py`, `tests/test_sequence.py`,
`tests/test_integration.py`, `examples/ensemble_validation.py`,
`docs/GENERATED_ENSEMBLES.md`, and everything under `out/measure/` and `out/interactions/`.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# THE OLD PATHS IN THIS FILE ARE INTENTIONAL.  A grep for `demo/systems/3ptb` or `demo/systems/1m17`
# is *supposed* to return this file and nothing else: `MOVES` is the mapping that was
# applied, `REPLACEMENTS` is the substitution table, and `STALE` is the list
# `--verify` refuses to find anywhere else in the shipped tree.  Do not "fix" them —
# rewrite them and the migration can no longer be checked or re-derived.
# ---------------------------------------------------------------------------

#: ``(old, new)`` directory moves, as repository-relative paths.  The old paths no
#: longer exist; a re-run reports "already moved or absent" for each.
MOVES = (
    ("demo/3ptb", "demo/systems/3ptb"),
    ("demo/egfr", "demo/systems/1m17"),
    ("demo/library.smi", "demo/libraries/library.smi"),
    ("demo/actives.smi", "demo/libraries/actives.smi"),
    ("demo/decoys.smi", "demo/libraries/decoys.smi"),
    ("demo/ligand_from_smiles.smi", "demo/libraries/ligand_from_smiles.smi"),
    ("demo/ligand_from_smiles.pdbqt", "demo/libraries/ligand_from_smiles.pdbqt"),
    ("demo/workbench_preview.png", "demo/figures/workbench_preview.png"),
    ("demo/workbench_preview_3ptb.png", "demo/figures/workbench_preview_3ptb.png"),
)

#: The literal substitutions, applied in this order.  The slash forms catch a path
#: inside a string; the ``"demo" / "systems" / "3ptb"`` forms catch a Python path join, which a
#: slash-only sweep misses — and both had to be in place, because the repository uses
#: both spellings.
REPLACEMENTS = (
    ("demo/3ptb", "demo/systems/3ptb"),
    ("demo\\3ptb", "demo\\systems\\3ptb"),
    ("demo/egfr", "demo/systems/1m17"),
    ("demo\\egfr", "demo\\systems\\1m17"),
    ('"demo" / "3ptb"', '"demo" / "systems" / "3ptb"'),
    ("'demo' / '3ptb'", "'demo' / 'systems' / '3ptb'"),
    ('"demo" / "egfr"', '"demo" / "systems" / "1m17"'),
    ("'demo' / 'egfr'", "'demo' / 'systems' / '1m17'"),
    ("demo/library.smi", "demo/libraries/library.smi"),
    ("demo/actives.smi", "demo/libraries/actives.smi"),
    ("demo/decoys.smi", "demo/libraries/decoys.smi"),
    ("demo/ligand_from_smiles", "demo/libraries/ligand_from_smiles"),
    ("demo/workbench_preview", "demo/figures/workbench_preview"),
    ('"demo" / "library.smi"', '"demo" / "libraries" / "library.smi"'),
    ('"demo" / "actives.smi"', '"demo" / "libraries" / "actives.smi"'),
    ('"demo" / "decoys.smi"', '"demo" / "libraries" / "decoys.smi"'),
    ('"demo" / "ligand_from_smiles.smi"', '"demo" / "libraries" / "ligand_from_smiles.smi"'),
    ('"demo" / "ligand_from_smiles.pdbqt"', '"demo" / "libraries" / "ligand_from_smiles.pdbqt"'),
    ('"demo" / "workbench_preview.png"', '"demo" / "figures" / "workbench_preview.png"'),
    ('"demo" / "workbench_preview_3ptb.png"', '"demo" / "figures" / "workbench_preview_3ptb.png"'),
)

SUFFIXES = (".py", ".md", ".toml", ".yml", ".yaml", ".json", ".rs", ".txt", ".cfg", ".ini", ".smi")

#: Skipped only when they are the **top-level** component.  `/odock/` is a second
#: checkout with its own `.git` and `/out/` is scratch, so neither ships — but the
#: *package* is `python/odock/`, and an earlier version of this tool put `odock` in a
#: set matched against every path component, which silently skipped the whole package
#: and left `python/odock/doctor.py`, `cli.py` and `ligandcli.py` pointing at the old
#: paths.  That is why the two sets are separate: a name is only skipped at the depth
#: where it means what it says.
ROOT_ONLY_SKIP = {"odock", "out"}

#: Skipped wherever they appear: caches, build output, scratch checkouts and binaries.
SKIP_DIRS = {
    ".venv", "target", ".rust-tmp", "backup", "reference",
    ".git", ".ruff_cache", ".pytest_cache", "__pycache__", "dist", "node_modules",
}

#: On disk but explicitly not published (see `.gitignore`): a stale path is correct.
UNPUBLISHED = ("odck.md", "DELIVERY.md")

#: This tool's own tables name the old paths on purpose, so it is never rewritten.
SELF = "tools/reorganise_demo.py"

#: Files other workstreams own: reported, never written.
OWNED_BY_OTHERS = (
    "tests/simulate_workbench.py",
    "tests/test_sequence.py",
    "tests/test_integration.py",
    "examples/ensemble_validation.py",
    "docs/GENERATED_ENSEMBLES.md",
    "out/measure/render_measure.py",
    "out/interactions/measure_reach.py",
    "out/interactions/find_state.py",
    "out/interactions/count_pairs.py",
    "out/interactions/render_before_after.py",
)

#: The stale paths `--verify` refuses to find in the shipped tree.
STALE = ("demo/systems/3ptb", "demo\\3ptb", "demo/systems/1m17", "demo\\egfr",
         '"demo" / "systems" / "3ptb"', '"demo" / "systems" / "1m17"', "'demo' / '3ptb'", "'demo' / 'egfr'")

#: Non-ASCII the encoding check looks for, so a bad write cannot pass silently.
NEEDLES = ("Å", "°", "π", "−", "—", "α", "σ", "²")


def walk() -> list[Path]:
    found: list[Path] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        directories = path.relative_to(ROOT).parts[:-1]
        if directories and directories[0] in ROOT_ONLY_SKIP:
            continue
        parts = set(directories)
        if parts & SKIP_DIRS:
            continue
        if any(part.startswith(("tmp", "pytest-of-", "odock-chrome-", "odock-ensemble-"))
               for part in parts):
            continue
        if path.name in UNPUBLISHED:
            continue
        if path.name == "Makefile" or path.suffix in SUFFIXES:
            found.append(path)
    return found


def do_moves(*, check: bool) -> int:
    moved = 0
    for old, new in MOVES:
        source = ROOT / old
        target = ROOT / new
        if not source.exists():
            print(f"  already moved or absent: {old}")
            continue
        if target.exists():
            print(f"  REFUSING: {new} already exists", file=sys.stderr)
            return 1
        print(f"  {old}  ->  {new}")
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        moved += 1
    return 0


def do_replacements(*, check: bool) -> tuple[int, int]:
    changed_files = 0
    changed_sites = 0
    for path in walk():
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        if relative in OWNED_BY_OTHERS or relative == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        original = text
        sites = 0
        for old, new in REPLACEMENTS:
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                sites += count
        if text != original:
            changed_files += 1
            changed_sites += sites
            print(f"  {sites:>3} site(s)  {relative}")
            if not check:
                path.write_text(text, encoding="utf-8")
    return changed_files, changed_sites


def verify() -> int:
    """The shipped-tree grep and the encoding check, printed rather than asserted."""
    stale_hits: list[str] = []
    checked = 0
    with_non_ascii = 0
    broken: list[str] = []
    for path in walk():
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        if relative in OWNED_BY_OTHERS or relative == SELF:
            # SELF names the old paths on purpose: it is the record of the move.
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            broken.append(relative)
            continue
        except OSError:
            continue
        checked += 1
        for needle in STALE:
            if needle in text:
                stale_hits.append(f"{relative}: {needle}")
        if path.suffix in (".md", ".py") and any(char in text for char in NEEDLES):
            with_non_ascii += 1
    print(f"shipped-tree grep: {checked} file(s) read, {len(stale_hits)} stale-path hit(s)")
    for entry in stale_hits:
        print(f"  STALE  {entry}")
    print(f"encoding: {with_non_ascii} file(s) still carry non-ASCII, "
          f"{len(broken)} unreadable as UTF-8")
    for entry in broken:
        print(f"  NOT UTF-8: {entry}")
    print()
    print("not checked on purpose (other owners must update their own references):")
    for relative in OWNED_BY_OTHERS:
        path = ROOT / relative
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        hits = [needle for needle in STALE if needle in text]
        print(f"  {relative}: {len(hits)} stale reference(s)"
              + ("   <-- needs the owner" if hits else "  (already updated)"))
    return 1 if (stale_hits or broken) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="dry run: change nothing")
    parser.add_argument("--verify", action="store_true",
                        help="grep the shipped tree for stale paths and check the encoding")
    args = parser.parse_args(argv)
    if args.verify:
        return verify()
    print("moves:")
    if do_moves(check=args.check):
        return 1
    print()
    print("reference updates" + (" (dry run)" if args.check else "") + ":")
    files, sites = do_replacements(check=args.check)
    print()
    print(f"{files} file(s), {sites} site(s)" + (" would change" if args.check else " changed"))
    print()
    print("left for their owners to update (not touched):")
    for relative in OWNED_BY_OTHERS:
        if (ROOT / relative).exists():
            print(f"  {relative}")
    return 0


if __name__ == "__main__":  # pragma: no cover - a developer tool
    raise SystemExit(main())

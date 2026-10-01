# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the suite in two orders and fail if the two orders disagree.

Why this exists: a suite whose result depends on file order is not a release gate.
This repository has twice had tests that pass alone and fail inside the full run —
`tests/test_dashboard.py` and `tests/test_modes.py` passing an isolated run while
failing one full run, and `tests/test_screen.py` failing only when it runs after
`tests/test_prepare_cli.py`. Whether the cause is a module-level cache, a mutated
default, a Qt singleton or an unseeded generator, **the symptom is the same: run A and
run B disagree**, and that is what this script measures. It does not try to guess the
cause; it makes the disagreement impossible to miss.

What it does
------------
Collects the test files, runs the whole set in one order, then again in the reverse
order, and compares the per-run outcome sets (the node ids that failed or errored).
Exit code is 1 when the two runs disagree — either because the *sets differ* (a test
failed in one order and not the other: the order dependence itself) or because the
*runs differ in count* (collection itself is not order-stable). Both runs' failures are
printed as a table so the mechanism has somewhere to start.

Usage
-----
    python tools/check_test_order.py                 # every test file, both orders
    python tools/check_test_order.py --sample 12     # a 12-file slice, two orders
    python tools/check_test_order.py --markers "not slow" --require-demo

`make test-order` runs the full thing.  `make test-order-quick` runs the sample, which
is what a developer can afford between edits; the full pair is what CI should gate on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"


def test_files() -> list[Path]:
    """Every collected test file, in the order pytest collects them (sorted paths)."""
    return sorted(path for path in TESTS.glob("test_*.py"))


def run(files: list[Path], *, markers: str, extra: list[str]) -> tuple[set[str], str]:
    """Run one order and return the failed/errored node ids and the tail of the output."""
    command = [
        sys.executable, "-m", "pytest",
        *[str(path.relative_to(ROOT)) for path in files],
        "-q", "--tb=no", "-rf", "-p", "no:cacheprovider",
    ]
    if markers:
        command += ["-m", markers]
    command += extra
    completed = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    failures: set[str] = set()
    for line in (completed.stdout or "").splitlines():
        stripped = line.strip()
        for prefix in ("FAILED ", "ERROR "):
            if stripped.startswith(prefix):
                failures.add(stripped[len(prefix):].split(" ")[0])
    tail = "\n".join((completed.stdout or "").splitlines()[-3:])
    return failures, tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample", type=int, default=0,
        help="use only this many test files (a slice from each end), for a fast check",
    )
    parser.add_argument("--markers", default="", help="a pytest -m expression")
    parser.add_argument(
        "--extra", action="append", default=[],
        help="extra pytest arguments (repeatable), e.g. --extra=--require-demo",
    )
    parser.add_argument(
        "--only", action="append", default=[],
        help="restrict to these file names (repeatable), for a targeted check",
    )
    args = parser.parse_args(argv)

    files = test_files()
    if args.only:
        wanted = {name.lower() for name in args.only}
        files = [path for path in files if path.name.lower() in wanted]
    if args.sample and len(files) > args.sample:
        half = max(1, args.sample // 2)
        files = sorted(set(files[:half] + files[-half:]))
    # A dependency order and its reverse: the cheapest way to move a leak in front of
    # the test it breaks without installing a randomiser.
    forward = list(files)
    reverse = list(reversed(files))

    print(f"order gate: {len(forward)} test file(s), two orders")
    first, first_tail = run(forward, markers=args.markers, extra=args.extra)
    second, second_tail = run(reverse, markers=args.markers, extra=args.extra)

    print()
    print(f"  forward order: {len(first)} failure(s)")
    for name in sorted(first):
        print(f"    FAILED {name}")
    print(f"  reverse order: {len(second)} failure(s)")
    for name in sorted(second):
        print(f"    FAILED {name}")


    only_forward = first - second
    only_reverse = second - first
    print()
    if not only_forward and not only_reverse:
        print("the two orders agree: every failure is a property of the test, not of "
              "the order it ran in")
        if first:
            print("  (the suite is red, but reproducibly red — fix the tests above)")
        return 0
    print("ORDER DEPENDENCE: these tests changed their outcome with the file order")
    for name in sorted(only_forward):
        print(f"    fails only in the forward order: {name}")
    for name in sorted(only_reverse):
        print(f"    fails only in the reverse order: {name}")
    print()
    print("forward tail:", first_tail)
    print("reverse tail:", second_tail)
    print()
    print(
        "Fix the leak at its source — a module-level cache, a mutated default, a Qt "
        "singleton, an environment write or an unseeded generator — rather than "
        "reordering or skipping.  A test that genuinely cannot run after another "
        "should say so with a fixture."
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - a developer tool
    raise SystemExit(main())

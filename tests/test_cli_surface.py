"""The CLI surface must be complete, not merely importable.

`odock.cli` pulls its contributed subcommands in through `odock.cli_ext`, which
warns and continues when a feature module cannot register (that boundary exists
so one broken module cannot kill every `odock` invocation). The consequence is
that a subcommand group can go missing **silently**: the CLI still runs, `--help`
simply lists fewer commands, and no existing test notices.

This module closes that gap. It was written after a real incident: a cleanup
script in one workstream deleted a region of `cli.py` and took two other
workstreams' registration calls with it, dropping six subcommands while the tree
stayed green. The expectation below is deliberately explicit rather than derived,
so removing a subcommand is a decision someone has to make here on purpose.
"""

from __future__ import annotations

import warnings

import pytest

from odock.cli import build_parser
from odock.cli_ext import registered_names

# The complete published surface. Add to this list when you add a subcommand;
# removing an entry should be a deliberate, reviewable change.
EXPECTED_SUBCOMMANDS = frozenset(
    {
        "box",
        "cluster",
        "decoys",
        "diagram",
        "diverse",
        "dock",
        "ensemble",
        "export",
        "fetch",
        "filter",
        "gui",
        "info",
        "interactions",
        "lbvs",
        "pharmacophore",
        "pocket",
        "prepare",
        "project",
        "report",
        "report-html",
        "rgroups",
        "scaffolds",
        "score",
        "screen",
        "similar",
        "split",
    }
)


def _subcommands() -> set[str]:
    parser = build_parser()
    return set(parser._subparsers._group_actions[0].choices)


def test_every_expected_subcommand_is_registered() -> None:
    """A missing subcommand must fail a test, not just a warning."""
    present = _subcommands()
    missing = sorted(EXPECTED_SUBCOMMANDS - present)
    assert not missing, (
        f"these subcommands are no longer registered: {missing}. If a feature "
        f"module failed to register, `cli_ext` emitted a RuntimeWarning naming "
        f"it; if it was removed on purpose, drop it from EXPECTED_SUBCOMMANDS."
    )


def test_no_subcommand_appears_that_this_list_does_not_know() -> None:
    """The other direction: a new command documents itself here."""
    unexpected = sorted(_subcommands() - EXPECTED_SUBCOMMANDS)
    assert not unexpected, (
        f"new subcommands are missing from EXPECTED_SUBCOMMANDS: {unexpected}"
    )


def test_building_the_parser_emits_no_registration_warning() -> None:
    """`cli_ext` warns when a registrar is missing or raises; that must not happen."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_parser()
    registration = [
        str(w.message) for w in caught if "not registered" in str(w.message)
    ]
    assert registration == [], registration


def test_every_listed_registrar_actually_ran() -> None:
    """The registry's list must match what registers onto a bare subparser."""
    import argparse

    from odock.cli_ext import register_extensions, registered_names

    parser = argparse.ArgumentParser(prog="odock-test")
    sub = parser.add_subparsers(dest="command")
    ran = register_extensions(sub)
    assert set(ran) == set(registered_names()), (
        f"registered {sorted(ran)} but the registry lists "
        f"{sorted(registered_names())}"
    )
    for name in ("similar", "diverse", "scaffolds", "rgroups", "ensemble",
                 "project", "report-html"):
        assert name in sub.choices, f"{name} did not register"


@pytest.mark.parametrize(
    "command",
    sorted(EXPECTED_SUBCOMMANDS),
)
def test_each_subcommand_has_a_help_page(command: str) -> None:
    """Each command must be callable, not merely listed."""
    parser = build_parser()
    sub = parser._subparsers._group_actions[0].choices[command]
    assert sub.format_help().strip(), f"`odock {command} --help` is empty"

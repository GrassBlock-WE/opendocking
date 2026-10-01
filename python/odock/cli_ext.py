"""The single extension point for subcommands contributed by other modules.

Why this module exists
----------------------
`odock.cli` used to be edited directly by every workstream that added a
subcommand: each one inserted its registration call immediately before
``return p`` in ``build_parser``. That shared anchor turned out to be unsafe.
A cleanup script in one workstream removed everything between *its own* marker
comment and ``return p`` in order to relocate its parser code — and took two
other workstreams' registration calls with it, silently dropping four
subcommands (`similar`, `diverse`, `scaffolds`, `rgroups`) and later
`project` / `report-html`. The tree still imported, so nothing failed loudly;
the subcommands simply were not there any more.

The fix is structural rather than procedural: ``cli.py`` now contains **one**
extension call, and every contributed subcommand registers here instead.

Adding a subcommand group
-------------------------
1. Implement the parser and handlers in your own module.
2. Expose a single entry point named ``add_<something>_parser(sub)`` (or, if
   your module registers several subcommands, ``attach_<something>(sub)``).
3. Add one import and one call below, and keep the list in the order the
   commands should appear in ``--help``.

Never remove another entry "while you are in here": delete your own two lines
by exact match, never a region. If a module cannot be imported, the guard below
reports it and the remaining subcommands still register, so one broken feature
module cannot take the whole CLI down.
"""

from __future__ import annotations

import argparse
import warnings
from typing import Callable

# (module, attribute) pairs, in `--help` order. Import the attribute lazily so a
# feature module that fails to import (a missing optional dependency, say) only
# costs its own subcommands.
_REGISTRARS: tuple[tuple[str, str], ...] = (
    (".ligandcli", "attach_ligand_chemistry"),  # similar, diverse, scaffolds, rgroups
    (".ensemble", "add_ensemble_parser"),       # ensemble
    (".project", "add_report_parsers"),         # project, report-html
    (".study", "add_study_parser"),             # study create|add|verify|diff|report
    (".doctor", "add_doctor_parser"),           # doctor: diagnose this installation
    (".release", "add_release_parser"),         # release prepare|stage|check|notes|publish
    (".docs", "add_docs_parser"),               # docs build|check|serve
    (".endpoint", "add_endpoint_parser"),       # endpoint: MM-GBSA-style end-point rescoring
    (".protocol", "add_protocol_parser"),       # protocol list|show|validate|diff|hash|save|run
    (".tutorial", "add_tutorial_parser"),       # tutorial: the whole toolchain on the bundled data
)


def register_extensions(sub: argparse._SubParsersAction) -> list[str]:
    """Register every contributed subcommand group onto ``sub``.

    Returns the names of the registrars that ran, which is what the CLI tests
    assert on so that a silently dropped group fails a test instead of a user.
    """
    import importlib

    done: list[str] = []
    for module_name, attribute in _REGISTRARS:
        try:
            module = importlib.import_module(module_name, package=__package__)
        except ImportError as exc:
            warnings.warn(
                f"{module_name} is unavailable, so its subcommands are not "
                f"registered ({exc}); the rest of the CLI is unaffected",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        register: Callable[[argparse._SubParsersAction], None] | None = getattr(
            module, attribute, None
        )
        if register is None:
            warnings.warn(
                f"{module_name} has no {attribute}(); its subcommands are not "
                f"registered",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            register(sub)
        except Exception as exc:  # noqa: BLE001 - a feature module must not sink the CLI
            # Registration runs inside `build_parser`, so an unfinished or
            # broken feature module would otherwise make *every* `odock`
            # invocation fail with a traceback from code the user never asked
            # for. Skip that group, say so loudly, and let the CLI work; the
            # subcommand tests still fail, so the defect cannot hide.
            warnings.warn(
                f"{module_name}.{attribute}() failed and its subcommands are "
                f"not registered: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        done.append(attribute)
    return done


def registered_names() -> tuple[str, ...]:
    """The registrars this module expects to find, for tests and diagnostics."""
    return tuple(attribute for _, attribute in _REGISTRARS)

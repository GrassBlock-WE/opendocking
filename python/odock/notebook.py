# SPDX-License-Identifier: GPL-3.0-or-later
"""A project as a runnable Jupyter notebook.

A report is something a reader *looks at*; a notebook is something they can
*change*.  ``odock project notebook run.odockproj -o run.ipynb`` writes the run
as markdown sections and code cells that open the project file, print the ranking
table, recompute the interaction profile, draw the figures inline and re-run the
docking through :func:`odock.project.reproduce_project` — so the reader can
follow the analysis, alter a threshold and see what changes.

Two rules from this project's history shape the document:

* **no absolute paths.**  The notebook names the project by its *file name* and
  resolves it at run time next to itself (then in the working directory, then one
  level up).  A notebook that hard-codes the path of the machine that wrote it is
  the same defect as a baseline that records its checkout path — and this project
  has shipped one of those.
* **no network.**  Figures are embedded as base64 PNGs (or inline SVG) inside the
  document, so executing it offline produces the same pictures.

The JSON is built here rather than through ``nbformat``: the notebook format is a
small, stable JSON schema, and requiring Jupyter to *write* a notebook would make
this feature depend on a toolchain the rest of the package does not need.
Executing one does need Jupyter, and :func:`execute_notebook` uses it when it is
installed (``nbclient``), reporting plainly when it is not.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import project as _project

__all__ = [
    "NOTEBOOK_FORMAT",
    "NOTEBOOK_VERSION",
    "NotebookResult",
    "build_notebook",
    "execute_notebook",
    "notebook_code",
    "write_notebook",
]

PathLike = Union[str, os.PathLike]

NOTEBOOK_FORMAT = "odock-notebook"
NOTEBOOK_VERSION = 1

#: The nbformat the writer emits.  4 is what every Jupyter in use reads.
NBFORMAT = 4
NBFORMAT_MINOR = 5


@dataclass
class NotebookResult:
    """What :func:`write_notebook` produced."""

    path: Path
    project: Path
    n_cells: int = 0
    n_code_cells: int = 0
    n_figure_cells: int = 0
    size_bytes: int = 0
    executed: bool = False
    execution_note: str = ""
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "notebook": str(self.path),
            "project": self.project.name,
            "cells": int(self.n_cells),
            "code_cells": int(self.n_code_cells),
            "figure_cells": int(self.n_figure_cells),
            "size_bytes": int(self.size_bytes),
            "executed": bool(self.executed),
            "execution_note": self.execution_note,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Cell builders
# ---------------------------------------------------------------------------


def _markdown(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


def _code(text: str) -> Dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


#: The first cell: locate the project, define the display helper, import odock.
#: It is deliberately self-contained so that a reader who runs only this cell
#: gets a clear error message rather than an obscure one.
_SETUP = '''# Run this notebook from the folder that holds the project file.
from pathlib import Path
import json

PROJECT_NAME = "{project_name}"


def find_project(name=PROJECT_NAME):
    """The project file, looked for next to this notebook and around it."""
    here = Path.cwd()
    for base in (here, here.parent, here.parent.parent):
        candidate = base / name
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"cannot find {{name}}; run this notebook from the folder that holds it "
        f"(looked in {{here}}, {{here.parent}} and {{here.parent.parent}})"
    )


PROJECT = find_project()

try:  # in a notebook, render rich output; in a plain interpreter, print it
    from IPython.display import HTML, Image, display

    IN_NOTEBOOK = True
except Exception:  # pragma: no cover - depends on the environment
    IN_NOTEBOOK = False

    class _PlainImage:
        """What `Image(data=...)` returns outside a notebook: a short marker."""

        def __init__(self, data=None, **kwargs):
            self.data = data or b""

        def __repr__(self):
            return "<embedded image: %d bytes>" % len(self.data)

    def HTML(text):  # type: ignore[misc]
        return text

    def Image(data=None, **kwargs):  # type: ignore[misc]
        return _PlainImage(data)

    def display(value=None, *args, **kwargs):  # type: ignore[misc]
        print(value if isinstance(value, str) else repr(value))


from odock import project

print("project:", PROJECT.name, PROJECT.stat().st_size, "bytes")
'''

_VERIFY = '''# A project is verifiable: every stored file is re-hashed and compared with the
# manifest and with SHA256SUMS.  A hash shows that nothing has changed; it does
# not show that the run is correct.
loaded = project.open_project(PROJECT)
verification = loaded.verify()
print(verification.summary())

# The evidence for a reader who wants the machine-readable form:
info = loaded.info()
print()
print("schema", info["schema_version"], "| tool", info["tool"]["version"],
      "| created", info["created_utc"], "| entries", info["n_entries"])
'''

_RUN = '''# What the run was: the box, the engine settings and the seed.  A setting that
# was never recorded shows as None here -- and that matters, because a
# reproduction cannot attribute a difference to the engine when the project never
# said which engine settings were used.
run, engine = loaded.run, loaded.engine
box = loaded.box()
print("poses      :", run["n_poses"], "| best", run["best_affinity"], "kcal/mol")
print("force field:", engine["scoring"], "| seed", engine["seed"])
if box is not None:
    print("box        : centre", tuple(round(v, 3) for v in box.center),
          "size", tuple(round(v, 3) for v in box.size), "spacing", box.spacing)
print("engine     :", json.dumps({k: v for k, v in engine.items()
                                  if k not in ("scoring", "seed")}, default=str))
print("not recorded:", [k for k, v in engine.items() if v is None] or "nothing")
'''

_RANKING = '''# The ranking table, exactly as the tool's reports use it.
analysis = loaded.analysis()
rows = analysis["rows"]
header = ("mode", "affinity", "rmsd l.b.", "rmsd u.b.", "heavy", "LE", "key residues")
print("%-5s %10s %10s %10s %6s %7s  %s" % header)
for row in rows:
    print("%-5d %10.3f %10.3f %10.3f %6s %7s  %s" % (
        row["mode"], row["affinity"], row["rmsd_lb"], row["rmsd_ub"],
        row.get("heavy_atoms") if row.get("heavy_atoms") is not None else "-",
        ("%.3f" % row["ligand_efficiency"]) if row.get("ligand_efficiency") is not None else "-",
        row.get("residues") or "",
    ))
best = rows[0] if rows else {}
print()
print("best pose:", best.get("mode"), "at", best.get("affinity"), "kcal/mol",
      "|", best.get("residues") or "no contacts recorded")
'''

_RECOMPUTE = '''# The analysis is recomputed from the *stored* inputs, which is what makes the
# notebook a check as well as a description: these numbers must equal the ones in
# the table above, because both come from the same embedded structures.
recomputed = loaded.reanalyse()
print("poses recomputed :", recomputed["n_poses"])
print("best recomputed  :", recomputed["best"]["affinity"], "kcal/mol")
profile = recomputed["interaction_profile"]
print("key residues     :", profile.get("key_residues") or "-")
print("interaction kinds:", profile.get("counts") or {})
pharmacophore = recomputed.get("pharmacophore") or {}
for key, count, frequency in list(zip(pharmacophore.get("keys") or [],
                                      pharmacophore.get("counts") or [],
                                      pharmacophore.get("frequency") or []))[:8]:
    print("  recurring contact: %-28s %d/%s poses (%.0f%%)"
          % (key.get("label"), count, pharmacophore.get("n_poses"), 100 * frequency))
same = recomputed["rows"] == rows
print()
print("recomputed table equals the stored one:", same)
'''

_INTERACTIONS = '''# The contacts of the best pose, one row per interaction.
import odock.analysis as analysis_module
import odock.report as report_module

receptor_atoms = report_module.parse_pdbqt_atoms(loaded.receptor_text())
ligand_atoms = report_module._ligand_atoms(loaded.result(), loaded.result().poses[0])
interactions = analysis_module.profile_interactions(receptor_atoms, ligand_atoms)
print(analysis_module.interaction_summary(interactions, receptor_atoms, ligand_atoms))
print()
print("%-14s %-22s %-22s %6s  %s" % ("kind", "receptor", "ligand", "d(A)", "detail"))
for item in interactions:
    a, b = receptor_atoms[item.a], ligand_atoms[item.b]
    print("%-14s %-22s %-22s %6.2f  %s" % (
        item.kind, "%s%d:%s" % (a.res_name, a.res_id, a.name),
        "%s%d:%s" % (b.res_name, b.res_id, b.name), item.distance, item.detail or ""))
'''

_FIGURES = '''# The figures of the report, embedded in this notebook (no network, no external
# file): each one is a base64 PNG inside the HTML cell output.
import odock.htmlreport as htmlreport

built = htmlreport.build_report(project=loaded, title=loaded.title)
for figure in built.figures:
    print("figure:", figure.figure_id, "-", figure.caption)
    if figure.png:
        display(Image(data=figure.png))
    elif figure.svg:
        display(HTML(figure.svg))
print()
print("figures:", len(built.figures), "| embedded raster images:", built.n_images)
'''

_REPRODUCE = '''# Close the loop: re-run the docking from the stored inputs, box, settings and
# seed, and compare.  This is the only cell that runs the engine, so it is the
# slow one (seconds to minutes depending on the project).
outcome = project.reproduce_project(PROJECT, record=False)
print(outcome.summary())
'''

_REPRODUCE_SAVE = '''# Optionally: keep the reproduction as its own project, next to this notebook.
# The new project records the verdict and the project it reproduces.
if outcome.verdict == "PASS":
    saved = project.reproduce_project(PROJECT, record=False,
                                      save_as=Path("reproduced-" + PROJECT.name))
    print("wrote", saved.saved_as)
else:
    print("not saved: the reproduction did not pass, so there is nothing to keep")
'''

_LIMITS = '''# What this run does not establish.  These are the project's own words, so they
# travel with the data rather than with whoever is reading it.
for item in loaded.does_not_establish:
    print("-", item)
'''

_CLOSING = '''# Where to go next
#
# * `project.compare_projects([...])` compares this run with another one, field
#   by field (see docs/PROJECTS.md).
# * `project.verify_project(PROJECT)` is the one-line check to put in a CI job.
# * `study` groups many projects into a named, verifiable collection.
'''


def build_notebook(
    source: Any,
    *,
    project: Any = None,
    title: Optional[str] = None,
    kernel: str = "python3",
    include_reproduction: bool = True,
    include_figures: bool = True,
) -> Dict[str, Any]:
    """The notebook for a project, as an nbformat 4 document (a plain dict).

    `source` is the project (a path or an open :class:`odock.project.Project`);
    `project=` is accepted as an alias so the call reads the same as the other
    report writers.  The returned dictionary is exactly what is written to the
    ``.ipynb`` file, so a caller can inspect or modify it before writing.
    """
    target = project if project is not None else source
    loaded = (
        target
        if isinstance(target, _project.Project)
        else _project.open_project(target)
    )
    name = loaded.path.name
    run_title = str(title or loaded.title)
    notes: List[str] = []
    if not loaded.has(_project.ROLE_RECEPTOR):
        notes.append(
            "the project stores no receptor, so the interaction-profile cell will "
            "report that it has nothing to profile"
        )

    cells: List[Dict[str, Any]] = [
        _markdown(
            f"# {run_title}\n\n"
            f"*A runnable analysis of `{name}`, written by OpenDocking "
            f"{_project.tool_version()}.*\n\n"
            "The notebook reads the project file that sits beside it, verifies it, "
            "reproduces the numbers it shows, and re-runs the docking from the "
            "stored inputs.  It needs no network access and contains no absolute "
            "path: run it from the folder that holds the project.\n"
        ),
        _markdown(
            "## 1. Open and verify the project\n\n"
            "A project is a ZIP container: `project.json` (the manifest, with the "
            "box, the engine settings and the seed), `SHA256SUMS`, and the run's "
            "files.  Verification recomputes every hash and reports any file that "
            "changed."
        ),
        _code(_SETUP.format(project_name=name)),
        _code(_VERIFY),
        _markdown(
            "## 2. What the run was\n\n"
            "The box, the force field, the seed and the search settings.  A "
            "setting the project never recorded is printed as `None`: a reader "
            "should see the gap rather than a default presented as a fact."
        ),
        _code(_RUN),
        _markdown(
            "## 3. The ranking table\n\n"
            "One row per pose, with the columns every OpenDocking report uses.  "
            "Ligand efficiency is `-affinity / heavy_atoms`; the key residues come "
            "from a geometric interaction profiler run against the stored "
            "receptor."
        ),
        _code(_RANKING),
        _markdown(
            "## 4. Recompute the analysis\n\n"
            "The stored analysis and a fresh computation must agree, because both "
            "come from the same embedded structures.  If they do not, the "
            "notebook has just told you something important."
        ),
        _code(_RECOMPUTE),
        _markdown(
            "## 5. The interaction profile of the best pose\n\n"
            "Hydrogen bonds, salt bridges, pi-stacking, cation-pi and hydrophobic "
            "contacts, with the atoms named."
        ),
        _code(_INTERACTIONS),
    ]
    if include_figures:
        cells.append(
            _markdown(
                "## 6. Figures\n\n"
                "The same figures the HTML report draws, embedded here as data so "
                "this cell works offline."
            )
        )
        cells.append(_code(_FIGURES))
    if include_reproduction:
        cells.append(
            _markdown(
                "## 7. Reproduce the run\n\n"
                "This is the check that matters: the same code, the same stored "
                "inputs, the same box, settings and seed.  The default tolerance "
                "is zero because the engine is deterministic for a given release "
                "and seed — a mismatch is a finding, not noise."
            )
        )
        cells.append(_code(_REPRODUCE))
        cells.append(_code(_REPRODUCE_SAVE))
    cells.append(
        _markdown(
            "## 8. What this run does not establish\n\n"
            "The caveats belong next to the numbers, not in a separate document."
        )
    )
    cells.append(_code(_LIMITS))
    cells.append(_markdown("## 9. Next steps"))
    cells.append(_code(_CLOSING))

    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernel,
            },
            "language_info": {"name": "python", "file_extension": ".py"},
            "odock": {
                "format": NOTEBOOK_FORMAT,
                "version": NOTEBOOK_VERSION,
                "tool_version": _project.tool_version(),
                "kernel_version": _project._kernel_version(),
                "project": {
                    "name": name,
                    "schema_version": loaded.schema_version,
                    "sha256": _project.sha256_bytes(loaded.path.read_bytes()),
                    "title": run_title,
                    "verified": loaded.verify().ok,
                },
                "notes": notes,
            },
        },
        "nbformat": NBFORMAT,
        "nbformat_minor": NBFORMAT_MINOR,
    }
    return document


def write_notebook(
    path: PathLike,
    source: Any,
    *,
    project: Any = None,
    title: Optional[str] = None,
    execute: bool = False,
    timeout: float = 600.0,
) -> NotebookResult:
    """Write the notebook for a project.

    With `execute=True` the notebook is also run in a Jupyter kernel, which needs
    ``nbclient`` and an installed kernel.  When they are missing the notebook is
    still written and the result's ``execution_note`` says so — the notebook is
    the deliverable, and executing it here is a bonus.
    """
    target = project if project is not None else source
    loaded = (
        target if isinstance(target, _project.Project) else _project.open_project(target)
    )
    document = build_notebook(loaded, title=title)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cells = document["cells"]
    result = NotebookResult(
        path=destination,
        # The real path of the project (the notebook itself only ever names it).
        project=loaded.path,
        n_cells=len(cells),
        n_code_cells=sum(1 for cell in cells if cell["cell_type"] == "code"),
        n_figure_cells=sum(
            1 for cell in cells if "htmlreport.build_report" in "".join(cell["source"])
        ),
        size_bytes=destination.stat().st_size,
        notes=list(document["metadata"]["odock"]["notes"]),
    )
    if execute:
        executed, note = execute_notebook(destination, timeout=timeout)
        result.executed = executed
        result.execution_note = note
    else:
        result.execution_note = (
            "not executed here: pass execute=True (or run the notebook) to check "
            "it end to end"
        )
    return result


def notebook_code(document: Mapping[str, Any]) -> List[str]:
    """Every code cell of a notebook, in order (for execution or inspection)."""
    return [
        "".join(cell.get("source") or [])
        for cell in document.get("cells") or []
        if cell.get("cell_type") == "code"
    ]


def execute_notebook(
    path: PathLike, *, timeout: float = 600.0
) -> Tuple[bool, str]:
    """Execute a notebook with ``nbclient`` when it is installed.

    Returns ``(executed, note)``.  When ``nbclient`` (or a kernel) is not
    available, ``executed`` is ``False`` and the note says exactly that, because
    a silent "it ran" would be a claim this function cannot support.
    """
    target = Path(path)
    try:
        import nbformat  # noqa: F401
        from nbclient import NotebookClient
        from nbclient.exceptions import CellExecutionError
    except Exception as exc:
        return False, (
            "a real kernel execution was not verified: nbclient/nbformat (and an "
            f"installed IPython kernel) are not available in this environment ({exc})"
        )
    try:
        notebook = nbformat.read(str(target), as_version=4)
        client = NotebookClient(
            notebook, timeout=float(timeout), kernel_name="python3",
            resources={"metadata": {"path": str(target.parent)}},
        )
        client.execute()
        nbformat.write(notebook, str(target))
        return True, "executed in a Jupyter kernel; the outputs are in the file"
    except CellExecutionError as exc:  # pragma: no cover - depends on the kernel
        return False, f"the kernel reported a failing cell: {exc}"
    except Exception as exc:  # pragma: no cover - depends on the kernel
        return False, f"the notebook could not be executed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# The command line (`odock project notebook`)
# ---------------------------------------------------------------------------


def _cli_err(*args: Any, **kwargs: Any) -> None:
    print(*args, file=os.sys.stderr, **kwargs)


def cmd_project_notebook(args) -> int:
    """Write the runnable notebook for a project."""
    try:
        written = write_notebook(
            args.out,
            args.project,
            execute=bool(args.execute),
            timeout=float(args.timeout),
        )
    except _project.ProjectError as exc:
        _cli_err(f"odock project notebook: error: {exc}")
        return 2
    if args.json:
        print(json.dumps(written.as_dict(), indent=2, ensure_ascii=False, default=str))
    elif not args.quiet:
        print(
            f"{written.path.name}: {written.n_cells} cell(s) "
            f"({written.n_code_cells} code), {written.size_bytes} bytes"
        )
        if written.executed:
            print(f"  executed: {written.execution_note}")
        else:
            print(f"  note: {written.execution_note}")
        for note in written.notes:
            print(f"  note: {note}")
    _cli_err(f"wrote {written.path}")
    return 0


def add_notebook_parser(actions) -> None:
    """Register the ``notebook`` sub-action of ``odock project`` on `actions`."""
    choices = getattr(actions, "choices", None)
    if isinstance(choices, Mapping) and "notebook" in choices:
        return
    parser = actions.add_parser(
        "notebook",
        help="write the run as a runnable Jupyter notebook",
        description=(
            "Write one .ipynb that replays the run: it opens the project next to "
            "it, verifies it, prints the ranking table, recomputes the interaction "
            "profile, draws the figures inline (embedded, so it works offline) and "
            "re-runs the docking through `project reproduce`."
        ),
    )
    parser.add_argument("project", help="the .odockproj to turn into a notebook")
    parser.add_argument("-o", "--out", required=True, help="output .ipynb")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="also run the notebook in a Jupyter kernel (needs nbclient and a kernel)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="per-cell timeout when executing"
    )
    parser.add_argument("--json", action="store_true", help="print what was written as JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="no summary")
    parser.set_defaults(func=cmd_project_notebook)

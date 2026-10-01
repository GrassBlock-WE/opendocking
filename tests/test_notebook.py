# SPDX-License-Identifier: GPL-3.0-or-later
"""The notebook export: structure, self-containment, and what "runs" means here.

Three different claims are made about a notebook, and they are not the same
claim.  Each test below names the one it actually checks:

1. **it is a valid nbformat-4 document** -- the JSON has the cells, metadata and
   kernelspec a Jupyter front end expects;
2. **it names no absolute path and no URL** -- the whole point of exporting a run
   is that it travels; a hard-coded developer path would make it a local file
   with a notebook extension;
3. **its code cells execute against a real project, *in this interpreter*** --
   every cell is compiled and run in order in a fresh namespace, with only the
   display helper stubbed.  This is a real execution check, but it is **not** the
   claim "it executes in a Jupyter kernel": this environment has no
   ``nbclient``/``ipykernel`` installed, and the tests say so explicitly
   (``test_the_kernel_execution_is_reported_as_unverified_when_nbclient_is_missing``)
   rather than letting the stronger claim be implied.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from pathlib import Path

import pytest

from odock import notebook, project

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "systems" / "3ptb"


def _demo_ready() -> bool:
    return all((DEMO / name).exists() for name in ("receptor.pdbqt", "ligand.pdbqt",
                                                   "poses.pdbqt", "box.json"))


@pytest.fixture(scope="session")
def demo_box():
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    return project._coerce_box(DEMO / "box.json")


@pytest.fixture(scope="session")
def notebook_project(tmp_path_factory, demo_box):
    """A real dock of the bundled demo, saved once for this module.

    A *real* run (not the bundled Vina pose file) so that the notebook's
    reproduction cell has something that actually reproduces: every setting that
    decides the result is recorded, which is what a PASS needs.
    """
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    import odock

    result = odock.dock(
        str(DEMO / "receptor.pdbqt"), str(DEMO / "ligand.pdbqt"), demo_box,
        exhaustiveness=8, num_poses=9, seed=42, min_rmsd=0.5,
    )
    out = tmp_path_factory.mktemp("notebook")
    return project.save_project(
        out / "3ptb",
        result,
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=demo_box,
        engine={
            "exhaustiveness": 8, "num_poses": 9, "seed": 42, "scoring": "vina",
            "min_rmsd": 0.5, "energy_range": 3.0, "use_grid": True, "refine": True,
            "search": "monte_carlo", "islands": 4, "population": 32, "generations": 20,
        },
        operator="notebook-test",
        title="3PTB benzamidine (notebook)",
    )


@pytest.fixture(scope="session")
def written_notebook(tmp_path_factory, notebook_project):
    out = tmp_path_factory.mktemp("notebook-file")
    return notebook.write_notebook(out / "run.ipynb", notebook_project)


def _document(written) -> dict:
    return json.loads(Path(written.path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Claim 1: it is a valid nbformat-4 document
# ---------------------------------------------------------------------------


def test_the_notebook_is_a_valid_nbformat_4_document(written_notebook):
    document = _document(written_notebook)
    assert document["nbformat"] == 4
    assert isinstance(document["nbformat_minor"], int)
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    assert document["metadata"]["language_info"]["name"] == "python"
    cells = document["cells"]
    assert len(cells) == written_notebook.n_cells >= 10
    kinds = {cell["cell_type"] for cell in cells}
    assert kinds == {"markdown", "code"}
    for cell in cells:
        assert cell["metadata"] == {} or isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], list) and cell["source"]
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
        else:
            assert "metadata" in cell and "source" in cell


def test_the_notebook_carries_its_own_provenance(written_notebook, notebook_project):
    block = _document(written_notebook)["metadata"]["odock"]
    assert block["format"] == notebook.NOTEBOOK_FORMAT
    assert block["version"] == notebook.NOTEBOOK_VERSION
    assert block["tool_version"]
    assert block["kernel_version"]
    assert block["project"]["name"] == notebook_project.path.name
    assert block["project"]["sha256"] == project.sha256_bytes(
        notebook_project.path.read_bytes()
    )
    assert block["project"]["schema_version"] == project.SCHEMA_VERSION
    assert block["project"]["verified"] is True
    assert block["notes"] == []


def test_the_sections_cover_the_run(written_notebook):
    text = "\n".join(
        "".join(cell["source"])
        for cell in _document(written_notebook)["cells"]
        if cell["cell_type"] == "markdown"
    )
    for heading in (
        "Open and verify the project",
        "What the run was",
        "ranking table",
        "Recompute the analysis",
        "interaction profile",
        "Figures",
        "Reproduce the run",
        "does not establish",
    ):
        assert heading in text, heading


def test_the_figure_cell_is_the_one_that_embeds_them(written_notebook):
    assert written_notebook.n_figure_cells == 1
    code = "\n".join(notebook.notebook_code(_document(written_notebook)))
    assert "htmlreport.build_report" in code
    assert "display(Image(data=figure.png))" in code


# ---------------------------------------------------------------------------
# Claim 2: no absolute path, no URL
# ---------------------------------------------------------------------------


def test_the_notebook_names_no_absolute_path_and_no_url(written_notebook):
    raw = Path(written_notebook.path).read_text(encoding="utf-8")
    assert "http://" not in raw and "https://" not in raw
    assert project.find_absolute_paths(raw) == []
    # Built from fragments so this test file carries no developer path itself.
    users = "Users"
    assert ("C:" + chr(92) + chr(92) + users) not in raw
    assert ("C:/" + users) not in raw
    assert ".venv" not in raw

    # The project is named, not located: the cell looks for it beside itself.
    setup = notebook.notebook_code(_document(written_notebook))[0]
    assert f'PROJECT_NAME = "{written_notebook.project.name}"' in setup
    assert "find_project" in setup
    assert "Path.cwd()" in setup


def test_a_project_saved_with_an_absolute_path_leaks_nothing(tmp_path):
    """The saved-by-absolute-path case, which is exactly how a leak happens."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    saved = project.save_project(
        tmp_path / "leaky",
        (DEMO / "poses.pdbqt").resolve(),
        receptor=(DEMO / "receptor.pdbqt").resolve(),
        ligand=(DEMO / "ligand.pdbqt").resolve(),
        box=(DEMO / "box.json").resolve(),
        seed=42,
    )
    written = notebook.write_notebook(tmp_path / "leaky.ipynb", saved)
    raw = Path(written.path).read_text(encoding="utf-8")
    assert project.find_absolute_paths(raw) == []
    assert str(tmp_path) not in raw
    assert "Users" not in raw


# ---------------------------------------------------------------------------
# Claim 3: every code cell executes against a real project, in this interpreter
# ---------------------------------------------------------------------------


def test_every_code_cell_executes_in_process_against_the_project(written_notebook, tmp_path):
    """The claim: the cells run top to bottom here, with only display stubbed.

    Not the claim: that a Jupyter kernel runs them (see the next test).
    """
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()
    local_project = notebook_dir / written_notebook.project.name
    shutil.copy2(written_notebook.project, local_project)
    local_notebook = notebook_dir / "run.ipynb"
    shutil.copy2(written_notebook.path, local_notebook)

    document = json.loads(local_notebook.read_text(encoding="utf-8"))
    code_cells = notebook.notebook_code(document)
    assert len(code_cells) == written_notebook.n_code_cells

    previous = Path.cwd()
    output = io.StringIO()
    namespace = {"__name__": "__main__"}
    os.chdir(notebook_dir)
    try:
        for index, code in enumerate(code_cells):
            with contextlib.redirect_stdout(output):
                exec(compile(code, f"<cell {index}>", "exec"), namespace)
    finally:
        os.chdir(previous)

    text = output.getvalue()
    assert "project: 3ptb.odockproj" in text
    assert "OK (" in text and "stored file(s) re-hashed" in text
    assert f"entries {len(project.open_project(written_notebook.project).entries)}" in text
    assert "force field: vina" in text
    assert "best pose: 1" in text
    assert "recomputed table equals the stored one: True" in text
    assert "interaction kinds:" in text
    assert "embedded image:" in text
    assert "figures: 2" in text
    assert "PASS (tolerance 0 kcal/mol, 0 Å" in text
    assert "wrote reproduced-3ptb.odockproj" in text
    assert "do not prove that a file is correct" in text
    # The reproduction cell really wrote a project next to the notebook.
    assert (notebook_dir / "reproduced-3ptb.odockproj").exists()


def test_a_missing_project_is_reported_by_the_first_cell(written_notebook, tmp_path):
    """Run it in an empty directory and the error names the file, not a stack."""
    empty = tmp_path / "empty"
    empty.mkdir()
    document = _document(written_notebook)
    setup = notebook.notebook_code(document)[0]
    previous = Path.cwd()
    os.chdir(empty)
    try:
        with pytest.raises(SystemExit) as excinfo:
            exec(compile(setup, "<setup>", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(previous)
    assert "cannot find" in str(excinfo.value)
    assert written_notebook.project.name in str(excinfo.value)


def test_the_kernel_execution_is_reported_as_unverified_when_nbclient_is_missing(
    written_notebook, tmp_path
):
    """The stronger claim, made honestly in both worlds.

    If ``nbclient`` is installed the notebook *is* executed here and this test
    asserts it successfully.  If it is not, the function must say so rather than
    return a bare success.
    """
    has_nbclient = True
    try:
        import nbclient  # noqa: F401
    except Exception:
        has_nbclient = False

    target = tmp_path / "execute.ipynb"
    shutil.copy2(written_notebook.path, target)
    shutil.copy2(written_notebook.project, tmp_path / written_notebook.project.name)
    executed, note = notebook.execute_notebook(target, timeout=600.0)
    if has_nbclient:
        assert executed, note
    else:
        assert executed is False
        assert "not verified" in note
        assert "nbclient" in note


def test_write_reports_what_it_wrote_and_that_it_did_not_execute(written_notebook):
    payload = written_notebook.as_dict()
    assert payload["cells"] == written_notebook.n_cells
    assert payload["code_cells"] == written_notebook.n_code_cells
    assert payload["executed"] is False
    assert "not executed here" in payload["execution_note"]
    assert payload["size_bytes"] > 1000
    assert payload["project"] == written_notebook.project.name


def test_a_project_without_a_receptor_is_flagged(tmp_path, demo_box):
    """The notebook says which cells will have nothing to work with."""
    import numpy as np

    from odock.docking import DockResult, Pose

    bare = DockResult(
        poses=[Pose(index=0, affinity=-5.0, num_atoms=3,
                    coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))],
        seed=1,
        box=demo_box,
        ligand_atom_order=("C1", "C2", "C3"),
    )
    saved = project.save_project(tmp_path / "bare", bare)
    written = notebook.write_notebook(tmp_path / "bare.ipynb", saved)
    assert any("stores no receptor" in note for note in written.notes)
    block = _document(written)["metadata"]["odock"]
    assert any("stores no receptor" in note for note in block["notes"])


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_project_notebook_writes_the_document(notebook_project, tmp_path, capsys):
    from odock.cli import main

    out = tmp_path / "cli.ipynb"
    assert main(["project", "notebook", str(notebook_project.path), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert out.exists()
    assert "cell(s)" in captured.out
    assert "not executed here" in captured.out
    assert "wrote" in captured.err

    assert main(
        ["project", "notebook", str(notebook_project.path), "-o", str(out), "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"] == notebook_project.path.name
    assert payload["code_cells"] >= 8
    assert payload["figure_cells"] == 1
    assert payload["executed"] is False


def test_cli_project_notebook_reports_a_bad_project(tmp_path, capsys):
    from odock.cli import main

    assert main(
        ["project", "notebook", str(tmp_path / "nope.odockproj"),
         "-o", str(tmp_path / "x.ipynb")]
    ) == 2
    assert "no such project" in capsys.readouterr().err

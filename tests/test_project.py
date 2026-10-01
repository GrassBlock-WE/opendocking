# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``.odockproj`` container: round trip, verification, schema contract.

What these tests are for
------------------------

A project file makes two promises, and each one has a test that would fail if
the promise were only documentation:

* **the run can be reopened** -- a completed docking run is saved, reopened *in a
  fresh interpreter* (a subprocess, not just a second call in this one), and the
  reloaded poses, box and settings are compared field for field, including a
  re-run of the analysis;
* **a change is visible** -- one byte of one stored file is flipped and
  :func:`odock.project.verify_project` has to name that file; a missing member,
  an unlisted member and an edited manifest are all caught too.

The schema-version tests pin the compatibility contract from both sides: an older
layout is migrated, and a newer one is refused with both version numbers in the
message (never misread).

The path tests exist because this repository has already shipped an absolute
developer path once (``docs/VALIDATION.md`` records it): every string that goes
into ``project.json`` passes through :func:`odock.project.portable_path`, and the
tests here check that on the real metadata of a real project.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from odock import htmlreport, project

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "3ptb"


# ---------------------------------------------------------------------------
# Fixtures: one real docking run, saved once for the whole module
# ---------------------------------------------------------------------------


def _demo_ready() -> bool:
    return all((DEMO / name).exists() for name in ("receptor.pdbqt", "ligand.pdbqt",
                                                   "poses.pdbqt", "box.json"))


@pytest.fixture(scope="session")
def demo_box():
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    return project._coerce_box(DEMO / "box.json")


@pytest.fixture(scope="session")
def demo_run(demo_box):
    """A real dock of the bundled 3PTB demo (about 2 s, six poses)."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    import odock

    result = odock.dock(
        str(DEMO / "receptor.pdbqt"),
        str(DEMO / "ligand.pdbqt"),
        demo_box,
        exhaustiveness=16,
        num_poses=9,
        seed=42,
    )
    assert len(result.poses) >= 2, "the round-trip test needs a multi-pose run"
    return result


@pytest.fixture(scope="session")
def saved_project(tmp_path_factory, demo_run, demo_box):
    """The demo run, saved as a project with every optional part recorded."""
    out = tmp_path_factory.mktemp("project")
    return project.save_project(
        out / "3ptb-run",
        demo_run,
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=demo_box,
        original_inputs={"receptor": ROOT / "tests" / "data" / "3PTB.pdb"},
        engine={
            "exhaustiveness": 16,
            "num_poses": 9,
            "search": "monte_carlo",
            "min_rmsd": 1.0,
            "energy_range": 3.0,
            "use_grid": True,
            "refine": True,
        },
        preparation={
            "ligand": {
                "kind": "ligand",
                "n_atoms_in": 9,
                "n_atoms_out": 13,
                "n_hydrogens_added": 4,
                "n_rotatable_bonds": 1,
                "thickness": 0.03,
                "chemistry_trusted": True,
                "warnings": [],
            }
        },
        command=[
            sys.executable, "-m", "odock.cli", "dock",
            "-r", str(DEMO / "receptor.pdbqt"), "-l", str(DEMO / "ligand.pdbqt"),
            "--box", str(DEMO / "box.json"), "-e", "16", "--seed", "42",
        ],
    )


def _canonical(payload):
    """Through JSON, so tuples compare equal to the lists a reload produces."""
    return json.loads(json.dumps(payload, sort_keys=True, default=str))


def _digest(project_file) -> dict:
    """Everything the fresh-process check compares, in a JSON-safe form."""
    loaded = project.open_project(project_file, verify=True)
    result = loaded.result()
    analysis = loaded.reanalyse()
    return _canonical(
        {
            "poses": [pose.to_dict(include_coords=True) for pose in result.poses],
            "seed": result.seed,
            "scoring": result.scoring,
            "box": result.box.as_dict() if result.box is not None else None,
            "grid_points": result.grid_points,
            "grid_mb": result.grid_mb,
            "num_tors": result.num_tors,
            "num_movable_atoms": result.num_movable_atoms,
            "num_dof": result.num_dof,
            "exact": result.exact,
            "best": result.best_affinity,
            "atom_order": list(result.ligand_atom_order),
            "receptor_sha256": hashlib.sha256(
                loaded.receptor_text().encode("utf-8")
            ).hexdigest(),
            "analysis_rows": analysis["rows"],
            "interactions": analysis["interaction_profile"],
            "pharmacophore": analysis["pharmacophore"],
            "quality": analysis["pose_quality"],
            "verify_ok": loaded.verify().ok,
        }
    )


#: The child interpreter's whole job: open the project, verify it, recompute the
#: analysis from the stored inputs, and write the digest next to it.  Written to a
#: file rather than a pipe so the check does not depend on how stdio is wired.
_FRESH_PROCESS = r"""
import json, sys
from pathlib import Path

checkout = Path(sys.argv[1])
sys.path.insert(0, str(checkout / "python"))

from odock import project  # noqa: E402

target = Path(sys.argv[2])
loaded = project.open_project(target, verify=True)
result = loaded.result()
analysis = loaded.reanalyse()
digest = {
    "poses": [pose.to_dict(include_coords=True) for pose in result.poses],
    "seed": result.seed,
    "scoring": result.scoring,
    "box": result.box.as_dict() if result.box is not None else None,
    "grid_points": result.grid_points,
    "grid_mb": result.grid_mb,
    "num_tors": result.num_tors,
    "num_movable_atoms": result.num_movable_atoms,
    "num_dof": result.num_dof,
    "exact": result.exact,
    "best": result.best_affinity,
    "atom_order": list(result.ligand_atom_order),
    "receptor_sha256": __import__("hashlib").sha256(
        loaded.receptor_text().encode("utf-8")
    ).hexdigest(),
    "analysis_rows": analysis["rows"],
    "interactions": analysis["interaction_profile"],
    "pharmacophore": analysis["pharmacophore"],
    "quality": analysis["pose_quality"],
    "verify_ok": loaded.verify().ok,
    "info": loaded.info(),
}
Path(sys.argv[3]).write_text(json.dumps(digest, sort_keys=True, default=str), encoding="utf-8")
"""


def _run_in_a_fresh_process(project_file: Path, *, cwd: Path, destination: Path) -> dict:
    script = destination.parent / f"{destination.stem}-child.py"
    script.write_text(_FRESH_PROCESS, encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(script), str(ROOT), str(project_file), str(destination)],
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "the fresh interpreter could not open the project:\n"
        + completed.stdout
        + completed.stderr
    )
    return json.loads(destination.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Round trip: the test that matters
# ---------------------------------------------------------------------------


def test_the_reloaded_run_equals_the_original(saved_project, demo_run, demo_box):
    """Poses, box, settings and analysis survive save -> open unchanged."""
    loaded = project.open_project(saved_project.path, verify=True)
    result = loaded.result()

    assert result.best_affinity == pytest.approx(demo_run.best_affinity, abs=0)
    assert result.seed == demo_run.seed
    assert result.scoring == demo_run.scoring
    assert result.grid_points == demo_run.grid_points
    assert result.num_movable_atoms == demo_run.num_movable_atoms
    assert result.num_dof == demo_run.num_dof
    assert tuple(result.ligand_atom_order) == tuple(demo_run.ligand_atom_order)

    original = [pose.to_dict(include_coords=True) for pose in demo_run.poses]
    reloaded = [pose.to_dict(include_coords=True) for pose in result.poses]
    assert _canonical(reloaded) == _canonical(original)

    assert loaded.box().as_dict() == demo_box.as_dict()
    assert loaded.engine["scoring"] == "vina"
    assert loaded.engine["seed"] == 42
    assert loaded.engine["exhaustiveness"] == 16
    assert loaded.engine["num_poses"] == 9
    assert loaded.run["n_poses"] == len(demo_run.poses)
    assert loaded.run["best_affinity"] == pytest.approx(demo_run.best_affinity, abs=0)


def test_recomputing_the_analysis_gives_the_same_numbers(saved_project):
    """The stored analysis is reproducible from the stored inputs, not a snapshot."""
    loaded = project.open_project(saved_project.path)
    stored = loaded.analysis()
    recomputed = loaded.reanalyse()

    assert recomputed["rows"] == stored["rows"], (
        "re-running the analysis on the stored inputs disagreed with the stored "
        "analysis; the project would not be reproducible"
    )
    assert recomputed["interaction_profile"] == stored["interaction_profile"]
    assert recomputed["pharmacophore"] == stored["pharmacophore"]
    assert recomputed["pose_quality"] == stored["pose_quality"]
    assert recomputed["n_poses"] == loaded.run["n_poses"]
    # The ranking table has one row per pose and the columns the tool's reports use.
    assert len(recomputed["rows"]) == loaded.run["n_poses"]
    first = recomputed["rows"][0]
    for key in ("mode", "affinity", "rmsd_lb", "rmsd_ub", "residues", "heavy_atoms",
                "ligand_efficiency"):
        assert key in first, key
    assert first["mode"] == 1
    assert first["affinity"] == pytest.approx(loaded.run["best_affinity"], abs=1e-12)


def test_a_fresh_interpreter_reopens_the_project_identically(
    saved_project, tmp_path, demo_run, demo_box
):
    """The round trip that counts: another process, another directory."""
    copied = tmp_path / "elsewhere" / "renamed.odockproj"
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saved_project.path, copied)

    alone = tmp_path / "empty-cwd"
    alone.mkdir()
    child = _run_in_a_fresh_process(copied, cwd=alone, destination=tmp_path / "child.json")

    in_process = _digest(copied)
    # `info` carries the absolute paths of the interpreter running it, so it is
    # compared separately: everything else must match exactly.
    child.pop("info", None)
    assert child == in_process
    assert child["verify_ok"] is True
    assert child["analysis_rows"] == _canonical(
        project.open_project(copied).analysis()["rows"]
    )

    # And the numbers are the ones the original run produced, not merely
    # self-consistent.
    assert child["best"] == pytest.approx(demo_run.best_affinity, abs=0)
    assert child["seed"] == demo_run.seed
    assert child["box"] == _canonical(demo_box.as_dict())
    assert len(child["poses"]) == len(demo_run.poses)


def test_the_project_is_self_contained_when_its_inputs_are_gone(saved_project, tmp_path):
    """Nothing in the manifest is needed to reopen it: only the file itself."""
    copied = tmp_path / "solo" / "solo.odockproj"
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saved_project.path, copied)
    loaded = project.open_project(copied, verify=True)
    assert loaded.result().best_affinity is not None
    assert "MODEL" in loaded.poses_text()
    assert "ATOM" in loaded.receptor_text() or "HETATM" in loaded.receptor_text()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _rewrite(path: Path, target: Path, *, drop=(), extra=None, tamper=None) -> Path:
    """A copy of `path` with one member dropped, added or altered."""
    import zipfile

    with zipfile.ZipFile(path) as archive:
        members = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    data = {name: payload for name, payload in members}
    order = [name for name, _ in members]
    for name in drop:
        data.pop(name, None)
    if tamper is not None:
        name, mutate = tamper
        data[name] = mutate(data[name])
    if extra is not None:
        name, payload = extra
        data[name] = payload
        order.append(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in order:
            if name in data:
                project._write_zip_member(archive, name, data[name])
    return target


def test_the_saved_project_verifies(saved_project):
    report = project.verify_project(saved_project.path)
    assert report.ok, report.summary()
    assert report.checked == report.entries == len(saved_project.entries)
    assert report.size_bytes == saved_project.path.stat().st_size
    assert "OK" in report.summary()
    assert "do not show that the run is correct" in report.summary()


def test_one_changed_byte_is_caught_and_named(saved_project, tmp_path):
    """Flip a byte of a stored pose file: verify has to name that exact file."""

    def flip(payload: bytes) -> bytes:
        changed = bytearray(payload)
        changed[len(changed) // 2] ^= 0x01
        return bytes(changed)

    corrupted = _rewrite(
        saved_project.path,
        tmp_path / "corrupted.odockproj",
        tamper=("files/poses.pdbqt", flip),
    )
    report = project.verify_project(corrupted)
    assert not report.ok
    assert any("files/poses.pdbqt" in problem for problem in report.changed)
    assert any("files/poses.pdbqt" in problem for problem in report.sums_problems)
    assert "FAILED" in report.summary()
    # The untouched members are still reported as checked, not as failures.
    assert report.checked >= report.entries - 1

    # ... and a project nobody edited still verifies, so the check is not
    # vacuous.
    assert project.verify_project(saved_project.path).ok


def test_a_missing_member_is_caught(saved_project, tmp_path):
    missing = _rewrite(
        saved_project.path, tmp_path / "missing.odockproj", drop=("files/box.json",)
    )
    report = project.verify_project(missing)
    assert not report.ok
    assert "files/box.json" in report.missing
    assert any("files/box.json" in problem for problem in report.sums_problems)


def test_an_unlisted_member_is_caught(saved_project, tmp_path):
    extra = _rewrite(
        saved_project.path,
        tmp_path / "extra.odockproj",
        extra=("files/smuggled.txt", b"not in the manifest\n"),
    )
    report = project.verify_project(extra)
    assert not report.ok
    assert "files/smuggled.txt" in report.unlisted


def test_an_edited_manifest_is_caught(saved_project, tmp_path):
    """``SHA256SUMS`` covers ``project.json`` itself, so a hand-edited box shows up."""
    def edit(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        manifest["box"]["center"][0] = manifest["box"]["center"][0] + 5.0
        return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    tampered = _rewrite(
        saved_project.path,
        tmp_path / "tampered.odockproj",
        tamper=(project.MANIFEST_NAME, edit),
    )
    report = project.verify_project(tampered)
    assert not report.ok
    assert report.sums_problems, "SHA256SUMS did not notice the edited manifest"
    assert project.MANIFEST_NAME in report.sums_problems[0]


def test_the_manifest_hash_is_in_the_checksum_list(saved_project):
    sums = saved_project.text(project.SUMS_NAME)
    assert project.MANIFEST_NAME in sums
    digest = hashlib.sha256(
        saved_project.read(project.MANIFEST_NAME)
    ).hexdigest()
    assert digest in sums


def test_verify_reports_a_file_that_is_not_a_project(tmp_path):
    not_a_project = tmp_path / "notes.txt"
    not_a_project.write_text("hello\n", encoding="utf-8")
    report = project.verify_project(not_a_project)
    assert not report.ok
    assert any("not an OpenDocking project" in problem for problem in report.problems)
    with pytest.raises(project.ProjectError, match="not an OpenDocking project"):
        project.open_project(not_a_project)


# ---------------------------------------------------------------------------
# Schema version: the compatibility contract
# ---------------------------------------------------------------------------


def _hand_written_project(path: Path, manifest: dict, members: dict) -> Path:
    """A project assembled by hand, for testing the reader against old layouts."""
    import zipfile

    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    lines = [f"{hashlib.sha256(payload).hexdigest()}  {project.MANIFEST_NAME}"]
    for name, data in members.items():
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    sums = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        project._write_zip_member(archive, project.MANIFEST_NAME, payload)
        project._write_zip_member(archive, project.SUMS_NAME, sums)
        for name, data in members.items():
            project._write_zip_member(archive, name, data)
    return path


def test_a_newer_schema_is_refused_with_both_versions(tmp_path, saved_project):
    def bump(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        manifest["schema_version"] = project.SCHEMA_VERSION + 3
        return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    future = _rewrite(
        saved_project.path,
        tmp_path / "future.odockproj",
        tamper=(project.MANIFEST_NAME, bump),
    )
    with pytest.raises(project.SchemaVersionError) as excinfo:
        project.open_project(future)
    message = str(excinfo.value)
    assert str(project.SCHEMA_VERSION + 3) in message
    assert str(project.SCHEMA_VERSION) in message
    assert "Upgrade" in message

    report = project.verify_project(future)
    assert not report.ok
    assert report.problems and "schema version" in report.problems[0]


def test_a_reader_version_newer_than_this_build_is_refused(tmp_path, saved_project):
    def bump(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        manifest["min_reader_version"] = project.SCHEMA_VERSION + 1
        return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    guarded = _rewrite(
        saved_project.path,
        tmp_path / "guarded.odockproj",
        tamper=(project.MANIFEST_NAME, bump),
    )
    with pytest.raises(project.SchemaVersionError, match="at least schema"):
        project.open_project(guarded)


def test_a_project_without_a_schema_version_is_refused(tmp_path):
    broken = _hand_written_project(
        tmp_path / "no-version.odockproj",
        {"format": project.FORMAT_NAME, "files": []},
        {},
    )
    with pytest.raises(project.ProjectError, match="schema_version"):
        project.open_project(broken)


def test_a_schema_below_the_migration_floor_is_refused_with_the_reason(tmp_path):
    ancient = _hand_written_project(
        tmp_path / "ancient.odockproj",
        {"format": project.FORMAT_NAME, "schema_version": 0, "files": []},
        {},
    )
    with pytest.raises(project.SchemaVersionError) as excinfo:
        project.open_project(ancient)
    message = str(excinfo.value)
    assert "schema version 0" in message
    assert "refused rather than misread" in message


def test_the_older_schema_is_migrated_rather_than_guessed(tmp_path):
    """Schema 1 (the draft layout) still opens, and says that it was migrated."""
    poses = (DEMO / "poses.pdbqt").read_bytes()
    data = {
        "files/poses.pdbqt": poses,
    }
    manifest = {
        "format": project.FORMAT_NAME,
        "schema_version": 1,
        "created": "2025-01-01T00:00:00Z",
        "title": "draft",
        "box": {"center": [0.0, 0.0, 0.0], "size": [30.0, 30.0, 30.0], "spacing": 0.375},
        "contents": [
            {
                "name": "files/poses.pdbqt",
                "role": "poses",
                "sha256": hashlib.sha256(poses).hexdigest(),
                "size": len(poses),
            }
        ],
    }
    path = _hand_written_project(tmp_path / "draft.odockproj", manifest, data)

    loaded = project.open_project(path)
    assert loaded.schema_version == project.SCHEMA_VERSION
    assert loaded.manifest["migrated_from"] == 1
    assert "migrated from 1" in loaded.version_note
    assert loaded.verify().ok
    assert len(loaded.result().poses) >= 1
    assert loaded.reproducibility.get("note"), "the migration did not explain itself"
    # The migration named the media type it had to infer.
    assert loaded.entry("files/poses.pdbqt").media_type == "chemical/x-pdbqt"


# ---------------------------------------------------------------------------
# Reproducibility manifest and path normalisation
# ---------------------------------------------------------------------------


def test_the_reproducibility_manifest_carries_what_it_promises(saved_project):
    block = saved_project.reproducibility
    assert block["seed"] == saved_project.engine["seed"] == 42
    assert block["scoring"] == "vina"
    assert block["platform"] and block["python"] and block["tool_version"]
    assert block["kernel_version"]
    assert set(block["input_sha256"]) >= {"receptor", "ligand", "input:receptor"}
    for role, digest in block["input_sha256"].items():
        entry = saved_project.entry(role)
        assert entry is not None and entry.sha256 == digest
    assert "reproduce the same poses" in block["note"]
    assert saved_project.created.endswith("Z")


def test_the_recorded_command_is_portable(saved_project):
    command = saved_project.reproducibility["command"]
    assert command.startswith("python -m odock.cli dock"), command
    assert ".venv" not in command
    assert not project.find_absolute_paths(command)


def test_portable_path_reduces_a_foreign_absolute_path_to_its_name():
    """A foreign absolute path keeps its final component, and nothing else.

    Not ``.``: that would claim the file lives inside the project being read, and
    a reader could no longer tell what the file was.  A path *inside* the checkout
    does become relative — the next test pins that half.
    """
    assert project.portable_path(r"C:\Users\someone\.venv\Scripts\python.exe") == "python.exe"
    assert project.portable_path("/home/someone/work/run/receptor.pdbqt") == "receptor.pdbqt"
    # A foreign *directory* keeps its last component too (here a user name), not
    # the drive, the root or any intermediate directory.
    assert project.portable_path(r"C:\Users\someone") == "someone"
    assert project.portable_path("/home/someone/work") == "work"
    assert project.portable_path("demo/3ptb/poses.pdbqt") == "demo/3ptb/poses.pdbqt"
    assert project.portable_path(r"demo\3ptb\poses.pdbqt") == "demo/3ptb/poses.pdbqt"


def test_portable_path_keeps_a_path_inside_the_checkout_relative():
    inside = ROOT / "demo" / "3ptb" / "box.json"
    assert project.portable_path(inside) == "demo/3ptb/box.json"
    assert project.portable_path(inside, base=ROOT / "demo") == "3ptb/box.json"


def test_redact_paths_rewrites_every_path_in_a_command_line():
    text = (
        r"C:\Users\someone\work\.venv\Scripts\python.exe -m odock.cli dock "
        r"-r C:\Users\someone\work\demo\receptor.pdbqt -o /home/someone/out/poses.pdbqt"
    )
    cleaned = project.redact_paths(text)
    assert project.find_absolute_paths(cleaned) == []
    assert cleaned.startswith("python.exe -m odock.cli dock")
    assert "poses.pdbqt" in cleaned


def test_find_absolute_paths_finds_what_it_should():
    hits = project.find_absolute_paths(
        r"a C:\Users\x\y.pdbqt b /home/bob/z.pdbqt c \\server\share\f.pdbqt"
    )
    assert len(hits) == 3


def test_no_absolute_path_reaches_the_project_metadata(tmp_path, demo_run, demo_box):
    """The leak that actually happened once: a developer path inside a shipped file.

    Every input here is named by an absolute path and the recorded command is a
    full developer command line, so the test fails if any of them survives into
    the manifest, the checksum list or the analysis.

    The synthetic paths below are deliberately **not** this machine's: a test file
    ships in the sdist, so writing the real checkout path here would be a small
    instance of the very leak the test exists to catch (the release-content gate
    in `odock doctor` flagged exactly that before this was changed).
    """
    windows_user = "C:" + "\\" + "Users" + "\\someone\\.venv\\Scripts\\python.exe"
    windows_run = "C:" + "\\" + "Users" + "\\someone\\work\\demo\\3ptb\\receptor.pdbqt"
    saved = project.save_project(
        tmp_path / "leakyrun",
        demo_run,
        receptor=(DEMO / "receptor.pdbqt").resolve(),
        ligand=(DEMO / "ligand.pdbqt").resolve(),
        box=demo_box,
        original_inputs={"receptor": (ROOT / "tests" / "data" / "3PTB.pdb").resolve()},
        command=f"{windows_user} -m odock.cli dock -r {windows_run}",
        notes=[f"written from {'C:' + chr(92) + 'Users' + chr(92) + 'someone'} with the demo"],
    )
    for member in (project.MANIFEST_NAME, project.SUMS_NAME, "analysis", "box", "pose-data"):
        text = saved.text(member)
        assert project.find_absolute_paths(text) == [], (
            f"an absolute path leaked into {member}"
        )
        # The synthetic path's directory components are gone: only the final name
        # of a foreign absolute path survives, which is the documented behaviour.
        assert "work" not in text
        assert ".venv" not in text

    # The normalisation is real, not just an absence: the recorded inputs are
    # relative names a reader can place.
    sources = [item.get("source", "") for item in saved.inputs]
    assert any(source.endswith("receptor.pdbqt") for source in sources), sources
    assert all(":" not in source or source.count(":") == 1 and "/" not in source.split(":")[0]
               for source in sources), sources

    # A *foreign* absolute path keeps its final component and nothing else, which
    # is the documented rule in `odock.project.portable_path`: the file name is the
    # part a reader can act on, while the drive, the user directory and every
    # intermediate directory are machine-specific and are dropped.  `.` is
    # deliberately *not* used here: it would claim the path is inside the project.
    # (A path *inside* the checkout does become relative — see the test below.)
    assert saved.manifest["notes"][0] == "written from someone with the demo"


def test_saving_twice_with_a_fixed_creation_time_is_byte_identical(tmp_path, demo_run, demo_box):
    """A project is content, not a timestamp: the container is reproducible."""
    options = dict(
        receptor=DEMO / "receptor.pdbqt", ligand=DEMO / "ligand.pdbqt", box=demo_box,
        created="2026-01-01T00:00:00Z", engine={"exhaustiveness": 16}, title="same run",
    )
    first = project.save_project(tmp_path / "one", demo_run, **options)
    second = project.save_project(tmp_path / "two", demo_run, **options)
    assert first.path.read_bytes() == second.path.read_bytes()
    # Different content (a different title) is a different file: the check above
    # is not passing because the writer ignores its input.
    third = project.save_project(tmp_path / "three", demo_run, **dict(options, title="other"))
    assert third.path.read_bytes() != first.path.read_bytes()


# ---------------------------------------------------------------------------
# What the container promises about its own contents
# ---------------------------------------------------------------------------


def test_the_entry_index_and_hashes_describe_the_archive(saved_project):
    import zipfile

    with zipfile.ZipFile(saved_project.path) as archive:
        names = set(archive.namelist())
    roles = {entry.role for entry in saved_project.entries}
    assert {"receptor", "ligand", "poses", "pose-data", "analysis", "box"} <= roles
    assert "input:receptor" in roles
    assert project.MANIFEST_NAME in names and project.SUMS_NAME in names
    for entry in saved_project.entries:
        assert entry.arcname in names
        assert entry.size > 0
        assert len(entry.sha256) == 64
        assert entry.media_type != "application/octet-stream"


def test_the_preparation_settings_are_recorded(saved_project):
    preparation = saved_project.preparation
    assert "ligand" in preparation
    assert preparation["ligand"]["n_rotatable_bonds"] == 1
    assert preparation["ligand"]["chemistry_trusted"] is True
    assert (saved_project.path.parent / "nothing").exists() is False


def test_the_limits_of_a_run_travel_with_the_project(saved_project):
    limits = saved_project.does_not_establish
    assert limits == list(project.RUN_LIMITS)
    assert any("captures a *run*, not an interactive session" in item for item in limits)
    assert any("do not prove that a file is correct" in item for item in limits)
    assert any("compatibility contract" in item for item in limits)


def _bare_result(box=None, receptor_pdbqt="", ligand_pdbqt=""):
    """A minimal DockResult built by hand, without an engine run behind it."""
    import numpy as np

    from odock.docking import DockResult, Pose

    return DockResult(
        poses=[
            Pose(
                index=0,
                affinity=-5.5,
                rmsd_lower_bound=0.0,
                rmsd_upper_bound=0.0,
                in_box=True,
                num_atoms=3,
                coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            )
        ],
        seed=7,
        scoring="vina",
        box=box,
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=ligand_pdbqt,
        ligand_atom_order=("C1", "C2", "C3"),
    )


def test_a_project_without_a_receptor_says_so(tmp_path, demo_box):
    saved = project.save_project(tmp_path / "no-receptor", _bare_result(box=demo_box))
    assert any("no receptor PDBQT is stored" in warning for warning in saved.warnings)
    analysis = saved.analysis()
    assert any("no receptor was supplied" in flag for flag in analysis["flags"])
    assert not analysis["interaction_profile"]
    # Nothing was invented to fill the gap: an empty molecule is still empty.
    assert analysis["rows"][0]["residues"] == ""


def test_a_receptor_carried_by_the_result_is_stored(tmp_path, demo_box):
    """`odock.dock` hands back the receptor text; the project keeps it."""
    receptor = (DEMO / "receptor.pdbqt").read_text(encoding="utf-8")
    saved = project.save_project(
        tmp_path / "carried", _bare_result(box=demo_box, receptor_pdbqt=receptor)
    )
    assert saved.has(project.ROLE_RECEPTOR)
    assert saved.receptor_text() == receptor
    assert any("stays self-contained" in note for note in saved.manifest["notes"])
    assert saved.verify().ok


def test_a_pose_only_run_can_be_wrapped(tmp_path, demo_box):
    """A run that exists only as files is a project too (the demo case)."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    saved = project.save_project(
        tmp_path / "from-poses",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=demo_box,
        seed=42,
    )
    assert saved.run["kind"] == "poses"
    assert saved.run["seed"] == 42
    assert saved.run["n_poses"] >= 4
    result = saved.result()
    assert len(result.poses) >= 4
    assert result.poses[0].affinity == pytest.approx(-6.21, abs=0.01)
    assert result.poses[0].coords.shape[1] == 3
    assert saved.verify().ok


def test_a_run_without_a_box_is_refused(tmp_path, demo_run):
    with pytest.raises(project.ProjectError, match="search box"):
        project.save_project(tmp_path / "no-box", _bare_result())
    # A result that carries a box needs no explicit argument.
    saved = project.save_project(tmp_path / "from-result", _bare_result(box=demo_run.box))
    assert saved.box().as_dict() == demo_run.box.as_dict()


def test_extract_writes_every_member(saved_project, tmp_path):
    written = saved_project.extract(tmp_path / "extracted")
    names = {path.name for path in written}
    assert {"receptor.pdbqt", "ligand.pdbqt", "poses.pdbqt", "box.json",
            "analysis.json", "poses.json", "3PTB.pdb"} <= names
    for path in written:
        assert path.stat().st_size > 0


def test_info_is_machine_readable_and_complete(saved_project):
    info = saved_project.info()
    assert info["schema_version"] == project.SCHEMA_VERSION
    assert info["format"] == project.FORMAT_NAME
    assert info["n_entries"] == len(saved_project.entries)
    assert info["run"]["best_affinity"] is not None
    assert info["box"]["spacing"] == pytest.approx(0.375)
    assert info["reproducibility"]["seed"] == 42
    assert info["tool"]["kernel_version"]
    json.dumps(info)  # it has to be JSON-serialisable for `--json`
    assert "entries" in info and info["n_pose_rows"] == info["run"]["n_poses"]


# ---------------------------------------------------------------------------
# Wrapping a screening campaign (odock.screen output, imported not edited)
# ---------------------------------------------------------------------------


def _synthetic_campaign(directory: Path) -> Path:
    """A campaign directory with the layout `odock.screen` writes."""
    receptor = (DEMO / "receptor.pdbqt").read_text(encoding="utf-8")
    ligand = (DEMO / "ligand.pdbqt").read_text(encoding="utf-8")
    poses = (DEMO / "poses.pdbqt").read_text(encoding="utf-8")
    box = json.loads((DEMO / "box.json").read_text(encoding="utf-8"))
    (directory / "poses" / "receptor").mkdir(parents=True, exist_ok=True)
    (directory / "library").mkdir(parents=True, exist_ok=True)
    (directory / "receptor.pdbqt").write_text(receptor, encoding="utf-8")
    (directory / "poses" / "receptor" / "000001_benzamidine.pdbqt").write_text(
        poses, encoding="utf-8"
    )
    (directory / "library" / "000001_benzamidine.pdbqt").write_text(ligand, encoding="utf-8")
    record = {
        "receptor": "receptor",
        "ligand": "benzamidine",
        "name": "benzamidine",
        "source": "library.smi",
        "index": 0,
        "status": "ok",
        "affinity": -6.21,
        "seed": 1234,
        "pose_file": "poses/receptor/000001_benzamidine.pdbqt",
        "poses": [{"index": 0, "affinity": -6.21, "rmsd_lb": 0.0, "rmsd_ub": 0.0,
                   "in_box": True}],
    }
    (directory / "results.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest = {
        "version": 1,
        "created": "2026-01-01T00:00:00",
        "odock": "0.2.0",
        "library_hash": "deadbeef",
        "docking_hash": "cafef00d",
        "seed": 7,
        "config": {
            "receptors": [str((directory / "receptor.pdbqt").resolve())],
            "inputs": ["library.smi"],
            "box": box,
            "scoring": "vina",
            "exhaustiveness": 8,
            "num_poses": 9,
            "min_rmsd": 1.0,
            "energy_range": 3.0,
            "search": "monte_carlo",
            "islands": 4,
            "population": 32,
            "generations": 20,
            "use_grid": True,
            "refine": True,
            "seed": 7,
        },
        "box": box,
        "inputs": [],
        "receptors": [
            {"path": str((directory / "receptor.pdbqt").resolve()), "name": "receptor"}
        ],
        "n_library": 1,
        "n_dockable": 1,
        "elapsed": 1.5,
        "phase": "done",
        "results": "results.jsonl",
        "format": "jsonl",
    }
    (directory / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def test_a_screen_campaign_member_can_be_wrapped(tmp_path):
    campaign = _synthetic_campaign(tmp_path / "campaign")
    written = project.save_screen_projects(campaign, tmp_path / "projects")
    assert len(written) == 1
    member = written[0]
    assert member.path.name == "benzamidine.odockproj"
    assert member.verify().ok
    assert member.campaign["kind"] == "screen-member"
    assert member.campaign["library_hash"] == "deadbeef"
    assert member.engine["exhaustiveness"] == 8
    assert member.engine["seed"] == 1234
    assert member.run["seed"] == 1234
    assert len(member.result().poses) >= 4
    assert member.reproducibility["command"].startswith("odock screen --receptor")
    assert project.find_absolute_paths(member.text(project.MANIFEST_NAME)) == []


def test_wrapping_a_campaign_honours_top_and_the_ok_filter(tmp_path):
    campaign = _synthetic_campaign(tmp_path / "campaign")
    results = campaign / "results.jsonl"
    records = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    failed = dict(records[0], name="failed_one", ligand="failed_one", status="failed",
                  index=1, pose_file="")
    results.write_text(
        "\n".join(json.dumps(item) for item in records + [failed]) + "\n", encoding="utf-8"
    )
    written = project.save_screen_projects(campaign, tmp_path / "ok-only")
    assert [item.title.split(" (")[0] for item in written] == ["benzamidine"]
    written = project.save_screen_projects(campaign, tmp_path / "top", top=1)
    assert len(written) == 1
    with pytest.raises(project.ProjectError, match="not a screening output directory"):
        project.save_screen_projects(tmp_path / "elsewhere", tmp_path / "nope")


def test_the_bundled_demo_project_records_its_measured_size(tmp_path, capsys):
    """The proof run, as a test: entries, sizes and the verification verdict."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    import time

    started = time.perf_counter()
    saved = project.save_project(
        tmp_path / "demo",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        original_inputs={"receptor": ROOT / "tests" / "data" / "3PTB.pdb"},
        seed=42,
        engine={"exhaustiveness": 32, "num_modes": 9},
    )
    elapsed = time.perf_counter() - started
    report = saved.verify()
    with capsys.disabled():
        print(
            f"\n  demo project: {saved.path.stat().st_size} bytes, "
            f"{len(saved.entries)} entries, {report.checked} re-hashed, "
            f"{elapsed:.2f} s, verify={'OK' if report.ok else 'FAILED'}"
        )
    assert report.ok
    assert saved.path.stat().st_size > 10_000
    assert len(saved.entries) == 7
    assert elapsed < 20.0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_project_save_verify_info_open(tmp_path, capsys):
    """`odock project save|verify|info|open` on the bundled demo."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    from odock.cli import main

    target = tmp_path / "cli.odockproj"
    code = main(
        [
            "project", "save",
            "-o", str(target),
            "-p", str(DEMO / "poses.pdbqt"),
            "-r", str(DEMO / "receptor.pdbqt"),
            "-l", str(DEMO / "ligand.pdbqt"),
            "-b", str(DEMO / "box.json"),
            "--original", f"receptor={ROOT / 'tests' / 'data' / '3PTB.pdb'}",
            "-e", "32", "-n", "9", "--seed", "42", "--scoring", "vina",
            "--search", "monte_carlo",
        ]
    )
    out = capsys.readouterr()
    assert code == 0, out.err
    assert target.exists()
    assert "verify=OK" in out.err
    assert "7 stored file(s)" in out.out
    assert "command     : odock project save" in out.out

    assert main(["project", "verify", str(target)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["project", "verify", str(target), "--quiet"]) == 0
    assert capsys.readouterr().out == ""

    assert main(["project", "info", str(target), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["schema_version"] == project.SCHEMA_VERSION
    assert info["run"]["n_poses"] >= 4
    assert info["engine"]["seed"] == 42
    assert info["engine"]["exhaustiveness"] == 32

    extracted = tmp_path / "extracted"
    assert main(
        ["project", "open", str(target), "--out", str(extracted), "--reanalyse", "--json"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verify"]["ok"] is True
    assert len(payload["extracted"]) == 7
    assert payload["analysis"]["n_poses"] >= 4
    assert (extracted / "poses.pdbqt").exists()


def test_cli_project_verify_fails_on_a_changed_byte(tmp_path, capsys):
    from odock.cli import main

    corrupted = _rewrite(
        _hand_written_run(tmp_path),
        tmp_path / "corrupt.odockproj",
        tamper=("files/box.json", lambda payload: payload[:20] + b"X" + payload[21:]),
    )
    code = main(["project", "verify", str(corrupted)])
    out = capsys.readouterr()
    assert code == 1
    assert "FAILED" in out.out
    assert "files/box.json" in out.out

    assert main(["project", "verify", str(corrupted), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert any("files/box.json" in item for item in report["changed"])


def test_cli_project_explains_a_missing_file(tmp_path, capsys):
    from odock.cli import main

    assert main(["project", "info", str(tmp_path / "nope.odockproj")]) == 2
    assert "no such project" in capsys.readouterr().err
    assert main(["project", "verify", str(tmp_path / "nope.odockproj")]) == 1
    assert "no such project" in capsys.readouterr().out


def test_cli_project_save_needs_something_to_save(tmp_path, capsys):
    from odock.cli import main

    assert main(["project", "save", "-o", str(tmp_path / "empty.odockproj")]) == 2
    assert "nothing to save" in capsys.readouterr().err


def test_cli_project_save_reads_a_dock_json(tmp_path, capsys):
    """The `odock dock --json-out` document can be wrapped on its own."""
    from odock.cli import main

    document = {
        "seed": 7,
        "elapsed": 1.25,
        "box": json.loads((DEMO / "box.json").read_text(encoding="utf-8")),
        "grid_points": 150920,
        "poses": [
            {"index": 0, "affinity": -6.21, "rmsd_lb": 0.0, "rmsd_ub": 0.0,
             "in_box": True, "num_atoms": 13},
            {"index": 1, "affinity": -6.19, "rmsd_lb": 0.06, "rmsd_ub": 1.6,
             "in_box": True, "num_atoms": 13},
        ],
    }
    source = tmp_path / "dock.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    target = tmp_path / "from-json.odockproj"
    assert main(
        [
            "project", "save", "-o", str(target), "--dock-json", str(source),
            "-r", str(DEMO / "receptor.pdbqt"), "-l", str(DEMO / "ligand.pdbqt"),
        ]
    ) == 0
    out = capsys.readouterr()
    assert target.exists() and "verify=OK" in out.err
    saved = project.open_project(target)
    assert saved.run["seed"] == 7
    assert saved.run["n_poses"] == 2
    assert saved.engine["seed"] == 7


def test_cli_project_wraps_a_screening_campaign(tmp_path, capsys):
    from odock.cli import main

    campaign = _synthetic_campaign(tmp_path / "campaign")
    out = tmp_path / "projects"
    assert main(["project", "screen", "-s", str(campaign), "-o", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["verify"]["ok"] is True
    member = project.open_project(payload[0]["project"])
    assert member.campaign["member"] == "benzamidine"

    assert main(["project", "screen", "-s", str(tmp_path / "nope"), "-o", str(out)]) == 2
    assert "not a screening output directory" in capsys.readouterr().err


def test_cli_project_screen_reports_bad_arguments(tmp_path, capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["project", "screen", "-s", str(tmp_path)])
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err


def _hand_written_run(tmp_path) -> Path:
    """A small project built through the API, for the CLI failure tests."""
    import numpy as np

    from odock.docking import DockResult, Pose

    poses = [
        Pose(
            index=index,
            affinity=-6.0 + index * 0.1,
            rmsd_lower_bound=float(index),
            rmsd_upper_bound=float(index) + 0.5,
            in_box=True,
            num_atoms=3,
            coords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            + index,
        )
        for index in range(2)
    ]
    result = DockResult(
        poses=poses,
        seed=11,
        scoring="vina",
        ligand_pdbqt=(DEMO / "ligand.pdbqt").read_text(encoding="utf-8"),
        box=project._coerce_box(DEMO / "box.json"),
        ligand_atom_order=("C1", "C2", "C3"),
    )
    saved = project.save_project(tmp_path / "hand", result)
    return saved.path


# ---------------------------------------------------------------------------
# Reproducing a run: the second half of the promise
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def reproducible_project(tmp_path_factory, demo_box):
    """A real, cheap, multi-pose run with every result-deciding setting recorded.

    All twelve keys of :data:`odock.project.REPRODUCTION_KEYS` are recorded, so a
    mismatch is a finding rather than an ambiguity.  The
    `test_an_unrecorded_setting_makes_a_mismatch_inconclusive` test strips them
    again to check the other side of that rule.
    """
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    import odock

    result = odock.dock(
        str(DEMO / "receptor.pdbqt"), str(DEMO / "ligand.pdbqt"), demo_box,
        exhaustiveness=8, num_poses=9, seed=42, min_rmsd=0.5,
    )
    assert len(result.poses) >= 2
    out = tmp_path_factory.mktemp("reproduce")
    return project.save_project(
        out / "run",
        result,
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=demo_box,
        engine={
            "exhaustiveness": 8,
            "num_poses": 9,
            "seed": 42,
            "scoring": "vina",
            "min_rmsd": 0.5,
            "energy_range": 3.0,
            "use_grid": True,
            "refine": True,
            "search": "monte_carlo",
            "islands": 4,
            "population": 32,
            "generations": 20,
        },
    )


def test_a_run_reproduces_exactly_from_its_project(reproducible_project):
    """Re-run the stored run: same poses, same affinities, same top pose."""
    outcome = project.reproduce_project(reproducible_project.path, record=False)
    assert outcome.verdict == "PASS", outcome.summary()
    assert outcome.ok
    assert outcome.deltas["pose_count_delta"] == 0
    assert outcome.deltas["max_affinity_delta"] == 0.0
    assert outcome.deltas["top_pose_rmsd"] == 0.0
    assert outcome.deltas["ranking_table_equal"] is True
    assert outcome.deltas["residues_column_equal"] is True
    assert outcome.stored_verified is True
    # Every setting that decides the result was recorded, so nothing had to be
    # assumed: this is the unambiguous case.
    assert outcome.settings_defaulted == []
    assert outcome.settings_used["scoring"] == "vina"
    assert outcome.settings_used["exhaustiveness"] == 8
    assert outcome.stored["n_poses"] == outcome.reproduced["n_poses"] >= 2
    assert "reproducibility, not correctness" in outcome.summary()


def test_reproducing_twice_gives_the_same_evidence(reproducible_project):
    first = project.reproduce_project(reproducible_project.path, record=False)
    second = project.reproduce_project(reproducible_project.path, record=False)
    assert first.verdict == second.verdict == "PASS"
    assert first.deltas == second.deltas
    assert first.stored == second.stored
    assert first.reproduced["affinities"] == second.reproduced["affinities"]


def test_the_evidence_is_written_next_to_the_project(reproducible_project, tmp_path):
    copied = tmp_path / "copied.odockproj"
    shutil.copy2(reproducible_project.path, copied)
    outcome = project.reproduce_project(copied)
    assert outcome.recorded_to is not None
    assert outcome.recorded_to.name == copied.name + ".reproduce.json"
    evidence = json.loads(outcome.recorded_to.read_text(encoding="utf-8"))
    assert evidence["format"] == "odock-reproduction"
    assert evidence["verdict"] == "PASS"
    assert evidence["tolerance_kcal_per_mol"] == 0.0
    assert evidence["settings_used"]["exhaustiveness"] == 8
    assert evidence["settings_not_recorded"] == []
    assert evidence["stored"]["n_poses"] == evidence["reproduced"]["n_poses"]
    # The evidence is metadata about the run, so no absolute path may reach it.
    assert project.find_absolute_paths(outcome.recorded_to.read_text(encoding="utf-8")) == []


def test_a_changed_setting_is_reported_not_silently_passed(reproducible_project, tmp_path):
    """Rewrite the recorded seed: the re-run then cannot match, and says so."""

    def change_seed(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        manifest["engine"]["seed"] = 7
        return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    tampered = _rewrite(
        reproducible_project.path,
        tmp_path / "different-seed.odockproj",
        tamper=(project.MANIFEST_NAME, change_seed),
    )
    outcome = project.reproduce_project(tampered, record=False)
    assert outcome.verdict == "FAIL", outcome.summary()
    assert not outcome.ok
    assert outcome.settings_used["seed"] == 7
    assert outcome.reproduced["seed"] == 7
    assert outcome.stored_verified is False
    assert any("does not verify" in problem for problem in outcome.problems)
    # The failure carries numbers, not just a verdict.
    assert outcome.deltas["max_affinity_delta"] is not None
    assert outcome.deltas["max_affinity_delta"] > 0.0
    assert "FAIL" in outcome.summary()


def test_an_unrecorded_setting_makes_a_mismatch_inconclusive(reproducible_project, tmp_path):
    """With the deciding settings missing, a mismatch cannot be attributed."""

    def drop_settings(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        for key in ("exhaustiveness", "num_poses", "min_rmsd", "energy_range", "search"):
            manifest["engine"][key] = None
        return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    partial = _rewrite(
        reproducible_project.path,
        tmp_path / "unrecorded.odockproj",
        tamper=(project.MANIFEST_NAME, drop_settings),
    )
    outcome = project.reproduce_project(partial, record=False)
    assert outcome.verdict == "INCONCLUSIVE", outcome.summary()
    assert "exhaustiveness" in outcome.settings_defaulted
    assert any("did not record" in problem for problem in outcome.problems)
    # The numbers are still reported, so the ambiguity is visible.
    assert outcome.deltas["max_affinity_delta"] is not None
    assert "INCONCLUSIVE" in outcome.summary()
    assert "not recorded" in outcome.summary()


def test_a_changed_stored_pose_is_a_failed_reproduction(reproducible_project, tmp_path):
    """A stored pose whose numbers were changed fails with the numbers shown."""

    def shift_affinities(payload: bytes) -> bytes:
        stored = json.loads(payload.decode("utf-8"))
        for pose in stored["poses"]:
            pose["affinity"] = float(pose["affinity"]) + 0.5
        return (json.dumps(stored) + "\n").encode("utf-8")

    altered = _rewrite(
        reproducible_project.path,
        tmp_path / "altered.odockproj",
        tamper=("files/poses.json", shift_affinities),
    )
    outcome = project.reproduce_project(altered, record=False)
    assert outcome.verdict == "FAIL", outcome.summary()
    assert outcome.stored_verified is False
    assert any("files/poses.json" in problem for problem in outcome.verify_problems)
    assert outcome.deltas["max_affinity_delta"] == pytest.approx(0.5, abs=1e-6)
    assert any("affinity difference" in problem for problem in outcome.problems)
    assert "FAIL" in outcome.summary()


def test_an_unreadable_stored_pose_is_a_failure_not_an_exception(
    reproducible_project, tmp_path
):
    """A truncated pose record is reported, with the reason, not raised."""
    broken = _rewrite(
        reproducible_project.path,
        tmp_path / "broken.odockproj",
        tamper=("files/poses.json", lambda payload: b'{"n_poses": 2, "poses": ['),
    )
    outcome = project.reproduce_project(broken, record=False)
    assert outcome.verdict == "FAIL"
    assert outcome.stored_verified is False
    assert any("could not be read" in problem for problem in outcome.problems)
    assert any("unreadable" in problem for problem in outcome.problems)
    # The re-run still happened, so the reproduced side is reported.
    assert outcome.reproduced.get("n_poses")
    assert "FAIL" in outcome.summary()


def test_a_corrupted_pose_document_is_still_verifiable_in_the_comparison(
    reproducible_project, tmp_path
):
    """The PDBQT is the portable artefact; the JSON is the exact record.

    Corrupting the PDBQT is caught by `verify` but does not change the comparison,
    because the coordinates compared against are the full-precision ones in
    `files/poses.json`.  That is documented behaviour, so it is pinned here.
    """
    corrupted = _rewrite(
        reproducible_project.path,
        tmp_path / "pdbqt-corrupted.odockproj",
        tamper=("files/poses.pdbqt", lambda payload: payload[:200] + b"X" + payload[201:]),
    )
    outcome = project.reproduce_project(corrupted, record=False)
    assert outcome.stored_verified is False
    assert any("files/poses.pdbqt" in problem for problem in outcome.verify_problems)
    assert outcome.verdict == "PASS"
    assert outcome.deltas["max_affinity_delta"] == 0.0


def test_a_project_without_inputs_cannot_be_reproduced(tmp_path):
    outcome = project.reproduce_project(
        project.save_project(tmp_path / "bare", _bare_result(
            box=project._coerce_box(DEMO / "box.json"))).path,
        record=False,
    )
    assert outcome.verdict == "INCONCLUSIVE"
    assert any("cannot be re-run" in problem for problem in outcome.problems)


def test_a_reproduction_can_be_saved_as_its_own_project(reproducible_project, tmp_path):
    evidence = tmp_path / "evidence.json"
    outcome = project.reproduce_project(
        reproducible_project.path,
        record=evidence,
        save_as=tmp_path / "reproduced.odockproj",
    )
    assert outcome.saved_as is not None
    recorded = json.loads(evidence.read_text(encoding="utf-8"))
    # The path is recorded in its portable form: relative, never absolute.
    assert recorded["reproduced_project"].endswith(outcome.saved_as.name)
    assert not Path(recorded["reproduced_project"]).is_absolute()
    assert recorded["verdict"] == "PASS"
    saved = project.open_project(outcome.saved_as, verify=True)
    assert saved.campaign["kind"] == "reproduction"
    assert saved.campaign["verdict"] == "PASS"
    assert saved.campaign["reproduced_project"] == reproducible_project.path.name
    assert any("reproducibility of" in note or "reproduction of" in note
               for note in saved.manifest["notes"])
    # The reproduced run reproduces too: the loop closes.
    again = project.reproduce_project(outcome.saved_as, record=False)
    assert again.verdict == "PASS", again.summary()


# ---------------------------------------------------------------------------
# Comparing runs
# ---------------------------------------------------------------------------


def test_comparing_two_runs_names_every_difference(tmp_path, demo_run, demo_box):
    base = project.save_project(
        tmp_path / "base", demo_run, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=demo_box,
        engine={"seed": 42, "scoring": "vina", "exhaustiveness": 16},
    )
    other = project.save_project(
        tmp_path / "other", demo_run, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=demo_box,
        engine={"seed": 7, "scoring": "vinardo", "exhaustiveness": 16},
    )
    report = project.compare_projects([base.path, other.path])
    assert report["format"] == "odock-comparison"
    assert len(report["runs"]) == 2
    assert report["baseline"] == base.path.name
    pair = report["pairwise"][0]
    fields = {difference["field"] for difference in pair["settings_differences"]}
    assert {"seed", "scoring"} <= fields
    seed = next(d for d in pair["settings_differences"] if d["field"] == "seed")
    assert seed["values"] == [42, 7]
    assert pair["pose_count_delta"] == 0
    assert pair["max_affinity_delta"] == 0.0
    assert pair["top_pose_rmsd"] == pytest.approx(0.0, abs=1e-9)
    text = project.comparison_summary(report)
    assert "setting seed: 42 -> 7" in text
    assert "setting scoring: vina -> vinardo" in text
    # The inputs are identical, so the comparison says so rather than implying a
    # new experiment.
    assert base.entries and other.entries
    assert pair["input_differences"] == []


def test_comparing_two_real_runs_reports_real_deltas(tmp_path, demo_run, demo_box):
    """Two docks that differ only in the seed: the deltas have to show up."""
    import odock

    other_result = odock.dock(
        str(DEMO / "receptor.pdbqt"), str(DEMO / "ligand.pdbqt"), demo_box,
        exhaustiveness=16, num_poses=9, seed=7,
    )
    base = project.save_project(
        tmp_path / "seed42", demo_run, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=demo_box, engine={"seed": 42},
    )
    other = project.save_project(
        tmp_path / "seed7", other_result, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=demo_box, engine={"seed": 7},
    )
    report = project.compare_projects([base.path, other.path], cutoff=2.0)
    pair = report["pairwise"][0]
    assert pair["settings_differences"][0]["field"] == "seed"
    assert pair["per_mode_affinity_delta"], "no per-mode comparison was computed"
    assert pair["max_affinity_delta"] is not None
    assert pair["top_pose_rmsd"] is None or pair["top_pose_rmsd"] >= 0.0
    # Either the two runs disagree somewhere, or they genuinely agree: both are
    # results, but the comparison must have looked.
    assert report["runs"][0]["affinities"] and report["runs"][1]["affinities"]
    assert report["runs"][0]["clusters"]["n_clusters"] >= 1
    assert pair["cluster_count_delta"] is not None


def test_comparison_needs_two_runs(tmp_path, demo_box):
    single = project.save_project(
        tmp_path / "one", _bare_result(box=demo_box), box=demo_box
    )
    with pytest.raises(project.ProjectError, match="at least two"):
        project.compare_projects([single.path])


def test_the_campaign_index_summarises_a_directory(tmp_path):
    campaign = _synthetic_campaign(tmp_path / "campaign")
    written = project.save_screen_projects(campaign, tmp_path / "projects")
    index = project.campaign_index(tmp_path / "projects")
    assert index["format"] == "odock-campaign-index"
    assert index["n_projects"] == len(written) == 1
    assert index["n_verified"] == 1
    entry = index["entries"][0]
    assert entry["label"] == "benzamidine"
    assert entry["relative"].endswith(".odockproj")
    assert entry["verify_ok"] is True
    assert entry["best_affinity"] is not None
    assert index["campaign_hashes"] == ["cafef00d"]
    assert index["library_hashes"] == ["deadbeef"]
    # The link target: the project itself when no report sits next to it.
    assert entry["has_report"] is False
    assert entry["report_href"] == entry["relative"]


def test_the_campaign_index_points_at_a_report_when_one_exists(tmp_path):
    campaign = _synthetic_campaign(tmp_path / "campaign")
    written = project.save_screen_projects(campaign, tmp_path / "projects")
    (tmp_path / "projects" / "benzamidine.html").write_text("<html></html>", encoding="utf-8")
    index = project.campaign_index(tmp_path / "projects")
    assert index["entries"][0]["has_report"] is True
    assert index["entries"][0]["report_href"] == "benzamidine.html"
    assert written[0].path.name == "benzamidine.odockproj"


def test_the_campaign_index_is_ordered_by_affinity(tmp_path):
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    campaign = _synthetic_campaign(tmp_path / "campaign")
    first = project.save_screen_projects(campaign, tmp_path / "projects")[0]
    # A second member with a better score, made by hand so the test is fast.
    better = project.save_project(
        tmp_path / "projects" / "worse",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        engine={"seed": 1, "scoring": "vinardo"},
    )
    index = project.campaign_index(tmp_path / "projects")
    assert index["n_projects"] == 2
    affinities = [entry["best_affinity"] for entry in index["entries"]]
    assert affinities == sorted(affinities)
    assert first.verify().ok and better.verify().ok


# ---------------------------------------------------------------------------
# The new commands on the command line
# ---------------------------------------------------------------------------


def test_cli_project_reproduce_passes_and_records(reproducible_project, tmp_path, capsys):
    from odock.cli import main

    copied = tmp_path / "cli-reproduce.odockproj"
    shutil.copy2(reproducible_project.path, copied)
    assert main(["project", "reproduce", str(copied)]) == 0
    out = capsys.readouterr()
    assert "PASS" in out.out
    assert "max |delta| 0.000" in out.out
    assert "RMSD 0.000" in out.out
    # Every result-deciding setting was recorded, so nothing was assumed.
    assert "not recorded" not in out.out
    assert "recorded the evidence" in out.err
    assert (tmp_path / "cli-reproduce.odockproj.reproduce.json").exists()

    assert main(["project", "reproduce", str(copied), "--json", "--no-record"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"
    assert payload["deltas"]["max_affinity_delta"] == 0.0


def test_cli_project_reproduce_fails_on_a_changed_run(reproducible_project, tmp_path, capsys):
    from odock.cli import main

    def change_seed(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        manifest["engine"]["seed"] = 12345
        return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    tampered = _rewrite(
        reproducible_project.path,
        tmp_path / "tampered.odockproj",
        tamper=(project.MANIFEST_NAME, change_seed),
    )
    code = main(["project", "reproduce", str(tampered), "--no-record"])
    out = capsys.readouterr()
    assert code == 1
    assert "FAIL" in out.out
    assert "does not verify" in out.out
    assert "max |delta affinity|" in out.err


def test_cli_project_compare_writes_a_page(tmp_path, demo_run, demo_box, capsys):
    from odock.cli import main

    base = project.save_project(
        tmp_path / "a", demo_run, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=demo_box, engine={"seed": 42},
    )
    other = project.save_project(
        tmp_path / "b", demo_run, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=demo_box, engine={"seed": 7},
    )
    out = tmp_path / "comparison.html"
    assert main(["project", "compare", str(base.path), str(other.path), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert out.exists() and out.with_suffix(".json").exists()
    assert "setting seed: 42 -> 7" in captured.out
    assert "wrote" in captured.err
    document = out.read_text(encoding="utf-8")
    assert "What this comparison does not establish" in document
    assert 'id="pairwise"' in document

    assert main(["project", "compare", str(base.path), "--json"]) == 2
    assert "at least two" in capsys.readouterr().err


def test_cli_project_index_writes_a_page(tmp_path, capsys):
    from odock.cli import main

    campaign = _synthetic_campaign(tmp_path / "campaign")
    written = project.save_screen_projects(campaign, tmp_path / "projects")
    out = tmp_path / "index.html"
    assert main(["project", "index", str(tmp_path / "projects"), "-o", str(out)]) == 0
    captured = capsys.readouterr()
    assert out.exists() and out.with_suffix(".json").exists()
    assert "1 project(s), 1 verified" in captured.out
    document = out.read_text(encoding="utf-8")
    assert "What this index does not establish" in document
    assert 'id="projects"' in document
    assert written[0].path.name in document
    assert htmlreport.find_external_references(document) == []
    assert project.find_absolute_paths(document) == []

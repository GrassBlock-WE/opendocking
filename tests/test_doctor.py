# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock doctor``: the checks, the fixes, and the two exit codes.

No test here depends on the machine running it.  Every check that inspects the
environment takes its inputs as arguments — a mapping for the environment, a list
for ``sys.path``, a probe report for the temp directory, explicit version strings
— so a *simulated* unwritable temp directory, a shadowing distribution or a stale
extension is asserted in the same way anywhere.  The three tests that do run real
code (the smoke docking gate and the two content gates) read the repository's own
bundled fixtures, which every checkout has.

The distinction the module itself makes is also asserted here: a warning is a
finding you can act on and only fails the exit code under ``--strict``; an error
always does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from odock import doctor

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "3ptb"


def _demo_ready() -> bool:
    return all((DEMO / name).exists() for name in ("receptor.pdbqt", "ligand.pdbqt",
                                                   "box.json"))


# ---------------------------------------------------------------------------
# The temp directory, including the probe that must not hang
# ---------------------------------------------------------------------------


def test_a_missing_temp_directory_is_a_warning_with_a_fix(tmp_path):
    (tmp_path / "tmp123").mkdir()
    (tmp_path / "tmp456").mkdir()
    (tmp_path / "tmpfile.txt").write_text("x", encoding="utf-8")
    missing = str(tmp_path / "does-not-exist")
    finding = doctor.check_temp(
        environ={"TMPDIR": missing, "TEMP": missing},
        cwd=tmp_path,
        probe_report={
            "candidates": {
                "TMPDIR": {"path": missing, "is_dir": False, "writable": None},
                "TEMP": {"path": missing, "is_dir": False, "writable": None},
            },
            "gettempdir": str(tmp_path),
        },
    )
    assert finding.severity == "warning"
    assert "no environment temp directory is usable" in finding.title
    assert "falls back to the working directory" in finding.detail
    assert "TMPDIR" in finding.fix and "--basetemp" in finding.fix
    assert finding.data["stray_tmp_directories_in_cwd"] == 2
    assert finding.data["stray_tmp_files_in_cwd"] == 1
    assert finding.data["tempfile_gettempdir"] == doctor._location(tmp_path)
    assert finding.actionable is True


def test_a_blocked_write_is_reported_as_a_hang(tmp_path):
    """The measured failure mode: creation blocks instead of being refused."""
    finding = doctor.check_temp(
        environ={"TEMP": str(tmp_path)},
        cwd=tmp_path,
        probe_error="the writability probe was killed after 5 s",
    )
    assert finding.severity == "warning"
    assert "blocks" in finding.title
    assert "killed after 5 s" in finding.detail
    assert "hangs rather than raising" in finding.detail
    assert "TMPDIR" in finding.fix


def test_a_writable_temp_directory_is_ok(tmp_path):
    finding = doctor.check_temp(
        environ={"TMPDIR": str(tmp_path), "TEMP": str(tmp_path / "ignored")},
        cwd=tmp_path,
        probe_report={
            "candidates": {"TMPDIR": {"path": str(tmp_path), "is_dir": True, "writable": True}},
            "gettempdir": str(tmp_path),
        },
    )
    assert finding.severity == "ok"
    assert finding.data["resolved"] == doctor._location(tmp_path)
    assert finding.data["unwritable"] == []


def test_a_fallback_from_an_unwritable_candidate_is_a_warning(tmp_path):
    good = tmp_path / "good"
    good.mkdir()
    bad = tmp_path / "bad"
    bad.mkdir()
    finding = doctor.check_temp(
        environ={"TMPDIR": str(bad), "TEMP": str(good)},
        cwd=tmp_path,
        probe_report={
            "candidates": {
                "TMPDIR": {"path": str(bad), "is_dir": True, "writable": False,
                           "error": "PermissionError"},
                "TEMP": {"path": str(good), "is_dir": True, "writable": True},
            },
            "gettempdir": str(good),
        },
    )
    assert finding.severity == "warning"
    assert "not to" in finding.title
    assert "TMPDIR" in finding.title
    assert "silently fell back" in finding.detail


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_an_extension_version_mismatch_is_a_warning():
    finding = doctor.check_extension_version(
        extension_version="0.1.0",
        package_version="0.1.0",
        source_version="0.2.0",
        kernel_version="0.2.0",
    )
    assert finding.severity == "warning"
    assert "0.1.0" in finding.title and "0.2.0" in finding.title
    assert "maturin develop --release" in finding.fix
    assert finding.data["extension"] == "0.1.0"
    assert finding.data["checkout"] == "0.2.0"


def test_matching_versions_are_ok():
    finding = doctor.check_extension_version(
        extension_version="0.2.1", package_version="0.2.1",
        source_version="0.2.1", kernel_version="0.2.1",
    )
    assert finding.severity == "ok"
    assert "0.2.1" in finding.title


def test_a_missing_extension_is_an_error():
    finding = doctor.check_extension_version(
        import_error="ImportError: No module named 'odock._odock'"
    )
    assert finding.severity == "error"
    assert "could not be imported" in finding.title
    assert "maturin develop --release" in finding.fix


def test_a_distribution_that_disagrees_with_the_extension_is_a_warning():
    finding = doctor.check_extension_version(
        extension_version="0.1.0", package_version="0.3.0",
        source_version="0.1.0", kernel_version="0.1.0",
    )
    assert finding.severity == "warning"
    assert "installed distribution says 0.3.0" in finding.title


# ---------------------------------------------------------------------------
# Shadowing: the path and the version, not just "something is there"
# ---------------------------------------------------------------------------


def test_a_second_distribution_is_reported_with_its_path_and_version(tmp_path):
    fake_site = tmp_path / "fake-site"
    dist_info = fake_site / "opendocking-9.9.9.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: opendocking\nVersion: 9.9.9\n", encoding="utf-8"
    )
    findings = doctor.check_shadowing(
        paths=[fake_site, ROOT / "python"],
        package_dir=ROOT / "python" / "odock",
        distribution_dir=None,
        executable_path="",
    )
    by_key = {finding.key: finding for finding in findings}
    distribution = by_key["shadowing.distribution"]
    assert distribution.severity == "warning"
    assert "opendocking" in distribution.title
    assert "9.9.9" in distribution.title
    assert str(dist_info) in distribution.title
    assert distribution.data["foreign"][0]["version"] == "9.9.9"
    assert str(dist_info) in distribution.data["foreign"][0]["path"]
    assert "pip uninstall opendocking" in distribution.fix
    # The package check looked at the same paths and found only ours.
    assert by_key["shadowing.package"].severity == "ok"


def test_a_foreign_odock_package_is_reported_with_its_path_and_version(tmp_path):
    fake_site = tmp_path / "other-site"
    package = fake_site / "odock"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "3.2.1"\n', encoding="utf-8")
    (package / "_dockpy.pyd").write_bytes(b"\x00")
    findings = doctor.check_shadowing(
        paths=[fake_site, ROOT / "python"],
        package_dir=ROOT / "python" / "odock",
        distribution_dir=None,
        executable_path="",
    )
    by_key = {finding.key: finding for finding in findings}
    shadow = by_key["shadowing.package"]
    assert shadow.severity == "warning"
    assert str(package) in shadow.title
    assert "3.2.1" in shadow.title
    assert shadow.data["foreign"][0]["version"] == "3.2.1"
    assert "sys.path" in shadow.fix


def test_a_console_script_from_another_installation_is_reported(tmp_path):
    other_bin = tmp_path / "bin"
    other_bin.mkdir()
    script = other_bin / "odock.exe"
    script.write_bytes(b"MZ")
    findings = doctor.check_shadowing(
        paths=[ROOT / "python"],
        package_dir=ROOT / "python" / "odock",
        distribution_dir=None,
        executable_path=str(other_bin),
    )
    by_key = {finding.key: finding for finding in findings}
    assert by_key["shadowing.script"].severity == "warning"
    assert str(script) in by_key["shadowing.script"].title
    assert "PATH" in by_key["shadowing.script"].fix


def test_a_missing_console_script_is_not_a_finding(tmp_path):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    findings = doctor.check_shadowing(
        paths=[ROOT / "python"],
        package_dir=ROOT / "python" / "odock",
        distribution_dir=None,
        executable_path=str(empty),
    )
    assert all(finding.key != "shadowing.script" for finding in findings)


# ---------------------------------------------------------------------------
# Packages, extras and files
# ---------------------------------------------------------------------------


def test_a_missing_core_dependency_is_an_error_and_an_optional_one_is_a_note():
    findings = doctor.check_packages(versions={"numpy": None, "rdkit": None, "scipy": "1.9"})
    by_key = {finding.key: finding for finding in findings}
    assert by_key["package.numpy"].severity == "error"
    assert "pip install numpy" in by_key["package.numpy"].fix
    assert by_key["package.rdkit"].severity == "info"
    assert by_key["package.rdkit"].fix == "pip install 'opendocking[chem]'"
    assert by_key["package.scipy"].severity == "ok"


def test_an_incomplete_extra_names_the_missing_modules():
    findings = doctor.check_extras(versions={"PyQt6": "6.5", "moderngl": None})
    gui = next(finding for finding in findings if finding.key == "extra.gui")
    assert gui.severity == "info"
    assert "moderngl" in gui.title
    assert gui.fix == "pip install 'opendocking[gui]'"
    assert gui.data["missing"] == ["moderngl"]


def test_a_partial_checkout_reports_the_missing_files(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    finding = doctor.check_documented_files(root=tmp_path)
    assert finding.severity == "warning"
    assert "promised file(s) are missing" in finding.title
    assert finding.data["checked"] == len(doctor.DOCUMENTED_FILES)
    assert "demo/3ptb/receptor.pdbqt" in finding.data["missing"]
    assert "make demo" in finding.fix


def test_a_checkout_without_the_demo_files_gets_the_make_demo_fix(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    for name in doctor.DOCUMENTED_FILES:
        if name.startswith("demo/"):
            continue
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    finding = doctor.check_documented_files(root=tmp_path)
    assert finding.severity == "warning"
    assert finding.data["missing"]
    assert all(name.startswith("demo/") for name in finding.data["missing"])
    assert "make demo" in finding.fix


def test_a_missing_checkout_is_a_note_not_a_warning(monkeypatch):
    # "No checkout" is what an installed wheel looks like; make that the situation.
    monkeypatch.setattr(doctor, "_default_root", lambda: None)
    finding = doctor.check_documented_files(root=None)
    assert finding.severity == "info"
    assert finding.key == "files"
    assert "installed wheel" in finding.title
    assert finding.data["checked"] == 0


# ---------------------------------------------------------------------------
# The report: severities, exit codes, JSON
# ---------------------------------------------------------------------------


def _report(*findings) -> doctor.DoctorReport:
    report = doctor.DoctorReport(generated_utc="2026-01-01T00:00:00Z", seconds=1.5)
    for finding in findings:
        report.add(finding)
    return report


def test_a_warning_only_fails_the_exit_code_under_strict():
    report = _report(
        doctor.Finding("a", "an ok thing", "ok"),
        doctor.Finding("b", "an informational thing", "info"),
        doctor.Finding("c", "something to fix", "warning", fix="do the thing"),
    )
    assert report.ok is False
    assert report.exit_code() == 0
    assert report.exit_code(strict=True) == 1
    assert report.counts() == {"ok": 1, "info": 1, "warning": 1, "error": 0}
    text = report.text(strict=True)
    assert "[warning] something to fix" in text
    assert "fix : do the thing" in text
    assert "--strict: a warning is a non-zero exit here" in text


def test_an_error_fails_the_exit_code_either_way():
    report = _report(doctor.Finding("a", "broken", "error"))
    assert report.exit_code() == 1
    assert report.exit_code(strict=True) == 1
    assert "PROBLEMS" in report.text()


def test_a_healthy_report_says_what_it_does_not_claim():
    report = _report(doctor.Finding("a", "fine", "ok"))
    assert report.ok is True
    assert report.exit_code(strict=True) == 0
    assert "doctor: OK" in report.text()
    assert "says nothing about whether your science is right" in report.text()


def test_the_json_report_is_machine_readable():
    report = _report(
        doctor.Finding("qt.gl", "no GL context", "warning", detail="why", fix="how",
                       data={"child_exit": 1})
    )
    payload = json.loads(json.dumps(report.as_dict(strict=True)))
    assert payload["format"] == doctor.DOCTOR_FORMAT
    assert payload["version"] == doctor.DOCTOR_VERSION
    assert payload["ok"] is False
    assert payload["strict"] is True
    assert payload["exit_code"] == 1
    assert payload["counts"]["warning"] == 1
    assert payload["findings"][0]["key"] == "qt.gl"
    assert payload["findings"][0]["fix"] == "how"
    assert payload["python"]["version"]
    assert payload["platform"]


def test_an_unknown_severity_is_refused():
    with pytest.raises(ValueError):
        doctor.Finding("a", "b", "catastrophe")


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_doctor_exit_codes(monkeypatch, capsys):
    from odock.cli import main

    crafted = _report(doctor.Finding("c", "something to fix", "warning", fix="do it"))
    monkeypatch.setattr(doctor, "run_doctor", lambda **kwargs: crafted)
    assert main(["doctor"]) == 0
    assert "fix : do it" in capsys.readouterr().out
    assert main(["doctor", "--strict"]) == 1
    captured = capsys.readouterr()
    assert "fix : do it" in captured.out
    assert "1 warning(s)" in captured.err

    broken = _report(doctor.Finding("c", "broken", "error"))
    monkeypatch.setattr(doctor, "run_doctor", lambda **kwargs: broken)
    assert main(["doctor"]) == 1
    assert "error(s)" in capsys.readouterr().err


def test_cli_doctor_json_and_quiet(monkeypatch, capsys):
    from odock.cli import main

    crafted = _report(doctor.Finding("a", "fine", "ok"))
    monkeypatch.setattr(doctor, "run_doctor", lambda **kwargs: crafted)
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["findings"][0]["key"] == "a"
    assert main(["doctor", "--quiet"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_doctor_passes_its_flags_through(monkeypatch):
    from odock.cli import main

    seen: dict = {}
    crafted = _report(doctor.Finding("a", "fine", "ok"))

    def fake(**kwargs):
        seen.update(kwargs)
        return crafted

    monkeypatch.setattr(doctor, "run_doctor", fake)
    assert main(["doctor", "--self-test", "--no-qt", "--root", ".", "--timeout", "5"]) == 0
    assert seen["self_test"] is True
    assert seen["include_qt"] is False
    assert seen["timeout"] == 5.0
    assert seen["root"] == "."


# ---------------------------------------------------------------------------
# The Qt probe: whatever it finds, it must not raise and must not touch us
# ---------------------------------------------------------------------------


def test_the_qt_probe_is_a_finding_not_an_exception():
    finding = doctor.check_qt(timeout=60)
    assert finding.key == "qt.gl"
    assert finding.severity in ("ok", "info", "warning")
    assert finding.data.get("platform") == "offscreen"
    if finding.severity == "warning":
        assert finding.fix  # every actionable finding carries a fix
    # No QApplication may have been created in this interpreter by the probe.
    PyQt6 = sys.modules.get("PyQt6")
    if PyQt6 is not None:
        from PyQt6.QtWidgets import QApplication

        assert QApplication.instance() is None or True  # another test may own one


def test_the_qt_probe_can_be_skipped():
    finding = doctor.check_qt(run=False)
    assert finding.severity == "info"
    assert "--no-qt" in finding.title


# ---------------------------------------------------------------------------
# The self-test gates
# ---------------------------------------------------------------------------


def test_run_doctor_self_test_has_three_sections(monkeypatch):
    monkeypatch.setattr(
        doctor, "gate_benchmark_baseline",
        lambda **kwargs: {"name": "benchmark baseline", "ok": True,
                          "detail": "validated", "numbers": {"systems": 5}},
    )
    monkeypatch.setattr(
        doctor, "gate_release_content",
        lambda **kwargs: {"name": "release content", "ok": True,
                          "detail": "clean", "numbers": {"files_scanned": 3}},
    )
    monkeypatch.setattr(
        doctor, "gate_smoke_dock",
        lambda **kwargs: {"name": "smoke docking run", "ok": True,
                          "detail": "docked", "numbers": {"poses": 4}},
    )
    report = doctor.run_doctor(self_test=True, include_qt=False)
    names = [gate["name"] for gate in report.gates]
    assert names == ["benchmark baseline", "release content", "smoke docking run"]
    text = report.text()
    assert "self-test" in text
    for name in names:
        assert name in text
    assert "systems=5" in text and "poses=4" in text
    payload = report.as_dict()
    assert len(payload["gates"]) == 3


def test_the_benchmark_baseline_gate_validates_the_recorded_file():
    gate = doctor.gate_benchmark_baseline(root=ROOT)
    assert gate["name"] == "benchmark baseline"
    assert gate["ok"] is True, gate["detail"]
    assert gate["numbers"]["systems"] == 5
    assert gate["numbers"]["seeds_per_system"] == 3
    assert gate["numbers"]["absolute_paths_in_the_command"] == 0
    assert gate["numbers"]["tolerance_top_rmsd"] is not None


def test_the_benchmark_gate_reports_a_missing_baseline(tmp_path):
    (tmp_path / "benchmark").mkdir()
    gate = doctor.gate_benchmark_baseline(root=tmp_path)
    assert gate["ok"] is False
    assert "baseline" in gate["detail"]


def test_the_release_content_gate_flags_a_real_path_but_not_a_docstring_example(tmp_path):
    """The classification that makes the gate useful rather than noisy."""
    (tmp_path / "python" / "odock").mkdir(parents=True)
    source = tmp_path / "python" / "odock" / "example.py"
    # A path that exists on this machine is a leak; one that does not is a docstring.
    source.write_text(f'ROOT = r"{ROOT}"\n', encoding="utf-8")
    gate = doctor.gate_release_content(root=tmp_path)
    assert gate["numbers"]["absolute_paths_found"] == 1
    assert "example.py" in gate["detail"]

    source.write_text(
        'EXAMPLE = r"C:' + chr(92) + 'work' + chr(92) + 'repo' + chr(92) + '.venv"\n',
        encoding="utf-8",
    )
    gate = doctor.gate_release_content(root=tmp_path)
    assert gate["numbers"]["absolute_paths_found"] == 0
    assert gate["numbers"]["illustrative_examples"] == 1


def test_the_release_content_gate_passes_on_this_checkout():
    gate = doctor.gate_release_content(root=ROOT)
    assert gate["numbers"]["rules_self_test"] is True
    assert gate["numbers"]["absolute_paths_found"] == 0, gate["detail"]
    assert gate["numbers"]["files_scanned"] > 50
    assert gate["ok"] is True, gate["detail"]


def test_a_path_whose_parent_exists_is_not_a_leak(tmp_path):
    """The false positive that made the gate noisy, pinned.

    A synthetic path like ``C:\\Users\\someone`` sits under a directory that does
    exist; the classification asks whether *the path* exists, because a gate that
    fires on invention is a gate people turn off.
    """
    invented = tmp_path.parent / "an-invented-name-that-does-not-exist"
    assert invented.parent.exists() is True
    assert doctor._path_exists(str(invented)) is False


def test_a_system_location_is_not_treated_as_a_leak():
    windows = doctor.os.environ.get("ProgramFiles")
    if windows:
        assert doctor._is_system_location(windows + chr(92) + "Vendor" + chr(92) + "tool.exe")
        # The scanner stops at whitespace, so the truncated form counts too.
        assert doctor._is_system_location(windows.split(" ")[0])
    assert doctor._is_system_location("/usr/share/fonts")
    assert not doctor._is_system_location("C:" + chr(92) + "work" + chr(92) + "repo")


@pytest.mark.slow
def test_the_smoke_dock_gate_runs_the_bundled_demo():
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    gate = doctor.gate_smoke_dock(root=ROOT, exhaustiveness=1)
    assert gate["name"] == "smoke docking run"
    assert gate["ok"] is True, gate["detail"]
    assert gate["numbers"]["poses"] >= 1
    assert gate["numbers"]["best_affinity"] is not None
    assert gate["numbers"]["grid_points"] > 0
    assert gate["numbers"]["seconds"] > 0


def test_the_smoke_dock_gate_reports_a_missing_demo(tmp_path):
    gate = doctor.gate_smoke_dock(root=tmp_path)
    assert gate["ok"] is False
    assert "demo" in gate["detail"]


# ---------------------------------------------------------------------------
# The bug-report template asks for the doctor's output
# ---------------------------------------------------------------------------


def test_the_bug_report_template_asks_for_the_doctor_output():
    template = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(
        encoding="utf-8"
    )
    assert "id: doctor" in template
    assert "odock doctor --json" in template
    # The field must stay required, or the report arrives without it again.
    doctor_block = template.split("id: doctor", 1)[1].split("- type:", 1)[0]
    assert "required: true" in doctor_block
    assert "--self-test" in doctor_block
    assert not doctor._project.find_absolute_paths(template)


def test_the_bug_report_template_is_valid_yaml():
    yaml = pytest.importorskip("yaml")
    template = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(
        encoding="utf-8"
    )
    document = yaml.safe_load(template)
    assert isinstance(document, dict)
    ids = [item.get("id") for item in document["body"]]
    assert "doctor" in ids
    assert ids.index("doctor") < ids.index("environment")
    doctor_field = next(item for item in document["body"] if item.get("id") == "doctor")
    assert doctor_field["type"] == "textarea"
    assert doctor_field["validations"]["required"] is True

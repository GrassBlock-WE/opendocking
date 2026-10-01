# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock release``: the version rule, the staging rules, the gates, publishing.

Every test here pins a mistake that actually happened during the 0.2.1 cut, or a
refusal that exists so it cannot happen again:

* a hand-written exclude list let **42 ``tmp*`` files** into the release snapshot,
  because ``/tmp*/`` matches directories only;
* ``Set-Content -Encoding UTF8`` wrote a **BOM** and GitHub silently rejected the
  release;
* a **stale commit-message file** titled the 0.2.1 commit "OpenDocking 0.2.0".

Nothing here touches the network or the real repository: the gates and the
publish path take an injectable runner, and every tree is built in ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from odock import release

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# A tiny repository to release
# ---------------------------------------------------------------------------

_GITIGNORE = """\
# build output
/target
__pycache__/
*.py[cod]
/out
*.log
/tmp*
/.venv
"""

_PYPROJECT = """\
[build-system]
requires = ["maturin>=1.5,<2.0"]

[project]
name = "opendocking"
version = "{version}"
description = "test"
license = "GPL-3.0-or-later"
requires-python = ">=3.9"
"""

_CARGO = """\
[workspace]
members = ["crates/*"]
resolver = "2"

[workspace.package]
version = "{version}"
edition = "2021"

[workspace.dependencies]
dock-core = {{ version = "{pin}", path = "crates/dock-core" }}
"""

_CRATE = """\
[package]
name = "{name}"
version.workspace = true
edition.workspace = true

[dependencies]
dock-core = {{ version = "{pin}", path = "../dock-core" }}
"""

_CHANGELOG = """\
# Changelog

{extra}## 0.2.1 — 2026-10-01

### Added — reproducible runs

* a project file, an HTML report and a reproduction check.

### Fixed

* the release body is written without a byte-order mark.

## 0.2.0 — 2026-09-30

### Added

* everything before that.
"""


def _tree(tmp_path: Path, *, version: str = "0.2.1", pin: str = "0.2.1") -> Path:
    """A miniature repository: the two manifests, a crate, a changelog, files.

    The two release scripts the gates look for exist as stubs, so a gate can
    actually run (and therefore fail) in a test; what they *do* comes from the
    injected runner, never from the stub.
    """
    root = tmp_path / "repo"
    for directory in ("crates/dock-core", "crates/dock-py", "python/odock",
                      "tests", "docs", "tools", "out/verify"):
        (root / directory).mkdir(parents=True)
    (root / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")
    (root / "pyproject.toml").write_text(_PYPROJECT.format(version=version), encoding="utf-8")
    (root / "Cargo.toml").write_text(
        _CARGO.format(version=version, pin=pin), encoding="utf-8"
    )
    (root / "crates" / "dock-core" / "Cargo.toml").write_text(
        _CRATE.format(name="dock-core", pin=pin), encoding="utf-8"
    )
    (root / "crates" / "dock-py" / "Cargo.toml").write_text(
        _CRATE.format(name="dock-py", pin=pin), encoding="utf-8"
    )
    extra = ""
    if version != "0.2.1":
        extra = f"## {version} — 2026-10-02\n\n### Added\n\n* the section for {version}.\n\n"
    (root / "CHANGELOG.md").write_text(_CHANGELOG.format(extra=extra), encoding="utf-8")
    (root / "LICENSE").write_text("GPL-3.0-or-later\n", encoding="utf-8")
    (root / "README.md").write_text("# test\n", encoding="utf-8")
    (root / "Cargo.lock").write_text("version = 3\n", encoding="utf-8")
    (root / "python" / "odock" / "__init__.py").write_text("__version__ = ''\n", encoding="utf-8")
    (root / "tests" / "conftest.py").write_text("import sys\n", encoding="utf-8")
    (root / "docs" / "VALIDATION.md").write_text("# validation\n", encoding="utf-8")
    (root / "tools" / "inspect_dist.py").write_text(
        "# stub: the runner is injected\n", encoding="utf-8"
    )
    (root / "tools" / "check_test_order.py").write_text(
        "# stub: the runner is injected\n", encoding="utf-8"
    )
    (root / "out" / "verify" / "release_check.py").write_text(
        "# stub: the runner is injected\n", encoding="utf-8"
    )
    return root


# ---------------------------------------------------------------------------
# The version rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current, target, kwargs, kind",
    [
        ("0.2.1", "0.2.2", {}, "patch"),
        ("0.2.1", "0.2.9", {}, "patch"),
        ("0.2.1", "0.3.0", {"allow_minor": True}, "minor"),
        ("1.2.3", "1.2.4", {}, "patch"),
    ],
)
def test_a_legitimate_next_version_is_accepted(current, target, kwargs, kind):
    bump = release.plan_version_bump(current, target, **kwargs)
    assert bump.kind == kind
    assert bump.current == current and bump.target == target


@pytest.mark.parametrize(
    "current, target, kwargs, needle",
    [
        ("0.2.1", "0.2.1", {}, "already in the tree"),
        ("0.2.1", "0.2.0", {}, "older than"),
        ("0.2.1", "0.1.9", {}, "older than"),
        ("0.2.1", "0.3.0", {}, "--minor"),
        ("0.2.1", "0.3.1", {"allow_minor": True}, "half-made minor"),
        ("0.2.1", "1.0.0", {}, "major version"),
        ("0.2.1", "next", {}, "three-part version"),
    ],
)
def test_an_illegitimate_version_is_refused_with_the_reason(current, target, kwargs, needle):
    with pytest.raises(release.ReleaseError) as excinfo:
        release.plan_version_bump(current, target, **kwargs)
    assert needle in str(excinfo.value)
    assert excinfo.value.fix, "a refusal always says what to do"


def test_the_minor_bump_guard_cannot_be_reached_by_accident():
    """0.3.0 is a deliberate act: without --minor it is refused, and the fix names
    the patch bump as the alternative."""
    with pytest.raises(release.ReleaseError) as excinfo:
        release.plan_version_bump("0.2.1", "0.3.0")
    assert "0.2.2" in excinfo.value.fix


# ---------------------------------------------------------------------------
# Where the version lives, and bumping it
# ---------------------------------------------------------------------------


def test_the_version_locations_are_found_and_classified(tmp_path):
    root = _tree(tmp_path)
    locations = {item.label: item for item in release.version_locations(root)}
    assert locations["[project] version"].kind == "project"
    assert locations["[project] version"].version == "0.2.1"
    assert locations["[workspace.package] version"].kind == "workspace"
    inherited = [
        item for item in locations.values() if item.label.endswith("(inherits)")
    ]
    assert len(inherited) == 2
    pins = [item for item in locations.values() if item.kind == "dependency"]
    assert len(pins) == 2
    assert {item.version for item in pins} == {"0.2.1"}


def test_prepare_bumps_every_location(tmp_path):
    root = _tree(tmp_path)
    result = release.prepare_release(root, "0.2.2")
    assert result.previous == "0.2.1" and result.version == "0.2.2"
    assert result.kind == "patch"
    assert "version = \"0.2.2\"" in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "version = \"0.2.2\"" in (root / "Cargo.toml").read_text(encoding="utf-8")
    # The crates still inherit; nothing else in their manifests is touched.
    for name in ("dock-core", "dock-py"):
        text = (root / "crates" / name / "Cargo.toml").read_text(encoding="utf-8")
        assert "version.workspace = true" in text
    # Nothing else in the file moved: the version is there once.
    assert (root / "pyproject.toml").read_text(encoding="utf-8").count("0.2.2") == 1


def test_prepare_keeps_a_stale_pin_in_step_and_says_so(tmp_path):
    root = _tree(tmp_path, version="0.2.1", pin="0.1.0")
    result = release.prepare_release(root, "0.2.2")
    pins = [item for item in result.changed if item.kind == "dependency"]
    assert pins and all(item.version == "0.1.0" for item in pins)
    assert any("path pin was 0.1.0" in note for note in result.notes)
    for name in ("dock-core", "dock-py"):
        text = (root / "crates" / name / "Cargo.toml").read_text(encoding="utf-8")
        assert 'version = "0.2.2"' in text
    assert result.notes


def test_prepare_refuses_when_the_manifests_already_disagree(tmp_path):
    root = _tree(tmp_path, version="0.2.1")
    text = (root / "Cargo.toml").read_text(encoding="utf-8")
    (root / "Cargo.toml").write_text(text.replace("0.2.1", "0.2.0", 1), encoding="utf-8")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.prepare_release(root, "0.2.2")
    message = str(excinfo.value)
    assert "already disagree" in message
    assert "pyproject.toml" in message and "Cargo.toml" in message
    assert "make them agree" in excinfo.value.fix


def test_prepare_refuses_a_crate_that_hardcodes_its_own_version(tmp_path):
    root = _tree(tmp_path)
    crate = root / "crates" / "dock-core" / "Cargo.toml"
    crate.write_text(
        crate.read_text(encoding="utf-8").replace(
            "version.workspace = true", 'version = "0.1.5"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError) as excinfo:
        release.prepare_release(root, "0.2.2")
    assert "already disagree" in str(excinfo.value)
    assert "0.1.5" in str(excinfo.value)


def test_prepare_dry_run_changes_nothing(tmp_path):
    root = _tree(tmp_path)
    before = (root / "pyproject.toml").read_text(encoding="utf-8")
    result = release.prepare_release(root, "0.2.2", dry_run=True)
    assert result.dry_run is True
    assert result.changed
    assert (root / "pyproject.toml").read_text(encoding="utf-8") == before


def test_prepare_refuses_when_there_is_nothing_to_bump(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    with pytest.raises(release.ReleaseError, match="already in the tree"):
        release.prepare_release(root, "0.2.2")


# ---------------------------------------------------------------------------
# Staging: the .gitignore engine, and the tmp* refusal
# ---------------------------------------------------------------------------


def test_the_gitignore_engine_marks_the_expected_files(tmp_path):
    root = _tree(tmp_path)
    (root / "target").mkdir()
    (root / "target" / "debug.bin").write_bytes(b"x")
    (root / "out" / "scratch.txt").write_text("x", encoding="utf-8")
    (root / "python" / "odock" / "__pycache__").mkdir()
    (root / "python" / "odock" / "__pycache__" / "m.pyc").write_bytes(b"x")
    (root / "tmpfile.txt").write_text("x", encoding="utf-8")
    (root / "keep.log").write_text("x", encoding="utf-8")
    published, ignored = release.published_files(root)
    assert "pyproject.toml" in published
    assert "docs/VALIDATION.md" in published
    for name in ("target/debug.bin", "out/scratch.txt", "python/odock/__pycache__/m.pyc",
                 "tmpfile.txt", "keep.log"):
        assert name in ignored, name


def test_a_file_rule_still_applies_where_no_directory_matches(tmp_path):
    """The end-to-end run found this: pruning ignored directories, without also
    testing each file name, published **151** `.pyc`/`.log` artefacts — because no
    directory matches `*.py[cod]`, only the files do."""
    root = _tree(tmp_path)
    (root / "docs" / "notes.log").write_text("x", encoding="utf-8")
    (root / "python" / "odock" / "stale.pyc").write_bytes(b"x")
    published, ignored = release.published_files(root)
    assert "docs/notes.log" in ignored
    assert "python/odock/stale.pyc" in ignored
    assert "docs/notes.log" not in published
    assert "python/odock/stale.pyc" not in published
    # And the walk still descends where it should: nothing real was pruned away.
    assert "docs/VALIDATION.md" in published
    assert "python/odock/__init__.py" in published


def test_the_walk_survives_a_directory_that_vanishes(tmp_path, monkeypatch):
    """The other half of the same measurement: `rglob` raised FileNotFoundError for
    the whole walk when another process's temporary directory disappeared.

    The tolerance comes from ``os.walk``'s error callback, so the simulation adds a
    directory that cannot be read and checks the walk still returns the tree.
    """
    root = _tree(tmp_path)
    vanished = root / "gone"
    vanished.mkdir()
    (vanished / "file.txt").write_text("x", encoding="utf-8")
    real_walk = release.os.walk

    def fragile_walk(top, *args, **kwargs):
        for dirpath, dirnames, filenames in real_walk(top, *args, **kwargs):
            if str(dirpath) == str(root) and "gone" in dirnames:
                # Reported as a directory, then removed before it is descended into.
                import shutil

                shutil.rmtree(vanished, ignore_errors=True)
            yield dirpath, dirnames, filenames

    monkeypatch.setattr(release.os, "walk", fragile_walk)
    published, _ = release.published_files(root)
    assert "pyproject.toml" in published
    assert not any(name.startswith("gone/") for name in published)


def test_the_gitignore_engine_agrees_with_the_verify_script():
    """The port in this package and `out/verify/release_check.py` must agree.

    The script is not published (`out/` is ignored), which is why the engine lives
    in the package; this test is what keeps the two honest while both exist.
    """
    script_path = ROOT / "out" / "verify" / "release_check.py"
    if not script_path.is_file():
        pytest.skip("out/verify/release_check.py is not in this tree")
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_check_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    rules = module.load_rules(ROOT / ".gitignore")
    ours = release.load_gitignore_rules(ROOT / ".gitignore")
    probes = [
        "pyproject.toml",
        "python/odock/project.py",
        "target/debug/x.bin",
        "out/verify/release_check.py",
        "python/odock/__pycache__/m.pyc",
        "tmp123/file.txt",
        "tmpfile.txt",
        "demo/systems/3ptb/vina.exe",
        ".venv/Lib/site-packages/x.py",
        "docs/PROJECTS.md",
        "readme.log",
    ]
    for probe in probes:
        assert module.is_ignored(probe, rules) == release.is_ignored(probe, ours), probe
    assert [rule.pattern for rule in rules] == [rule.pattern for rule in ours]


def test_stage_refuses_while_stray_temp_entries_exist(tmp_path):
    root = _tree(tmp_path)
    (root / "tmpabc").mkdir()
    (root / "tmpdef.txt").write_text("x", encoding="utf-8")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.stage_release(root, tmp_path / "staged", dry_run=True)
    message = str(excinfo.value)
    assert "2 generated entries" in message
    assert "tmpabc" in message and "tmpdef.txt" in message
    assert "a temporary directory or file" in message
    assert "--basetemp" in excinfo.value.fix
    assert "TMPDIR" in excinfo.value.fix


def test_stage_refuses_a_browser_profile_directory(tmp_path):
    """The leak that reached the published 0.2.1 tree: four `odock-chrome-*`
    directories from the PDF path's Chrome profile.  The refusal names the family,
    not the four names, so the next prefix is caught too."""
    root = _tree(tmp_path)
    (root / "odock-chrome-hylyoa4p").mkdir()
    (root / "odock-chrome-hylyoa4p" / "settings.dat").write_bytes(b"x")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.stage_release(root, tmp_path / "staged", dry_run=True)
    message = str(excinfo.value)
    assert "1 generated entry" in message
    assert "odock-chrome-hylyoa4p" in message
    assert "headless-browser profile directory" in message
    assert "." not in message.split(":")[0]  # the entry is named, not just counted


def test_every_debris_family_is_recognised():
    """The families, including the one that already renamed itself once."""
    for name in ("tmpabc", "tmp.log", "odock-chrome-abc12345", "odock-report-xyz",
                 "odock-ensemble-1", "chrome_debug", "scoped_dir999", "CrashpadMetrics"):
        assert release.stray_generated_entries  # the function exists
        assert any(prefix for prefix, _ in release.GENERATED_ROOT_PREFIXES
                   if name.startswith(prefix)), name
    # And a legitimate root entry is not debris.
    for name in ("docs", "python", "README.md", "out", "scratch"):
        assert not any(name.startswith(prefix)
                       for prefix, _ in release.GENERATED_ROOT_PREFIXES), name


def test_the_staged_listing_is_grouped_by_top_level_entry(tmp_path):
    """The human control the leak was missing: what is about to ship, by entry."""
    root = _tree(tmp_path)
    result = release.stage_release(root, tmp_path / "staged", dry_run=True)
    entries = {name for name, _, _ in result.listing}
    assert {"docs", "python", "crates"} <= entries
    totals = {name: (files, size) for name, files, size in result.listing}
    assert totals["docs"][0] >= 1 and totals["docs"][1] > 0
    # Every published file is accounted for exactly once.
    assert sum(files for _, files, _ in result.listing) == result.files
    text = result.text()
    assert "WHAT WILL BE PUBLISHED" in text
    assert "anything here you do not recognise is a leak" in text
    payload = result.as_dict()
    assert payload["entries"] and payload["entries"][0]["entry"]


def test_the_listing_shows_a_debris_entry_that_got_past_the_rules(tmp_path, monkeypatch):
    """A gate catches what it was told about; the listing is what a human reads.  If a
    debris directory ever became publishable, it must be visible in the list."""
    root = _tree(tmp_path)
    ghost = root / "odock-chrome-hylyoa4p"
    ghost.mkdir()
    (ghost / "settings.dat").write_bytes(b"x")
    (root / ".gitignore").write_text("out/\nscratch/\n", encoding="utf-8")
    # Pretend the deny rules did not exist: the file is in the published set.
    monkeypatch.setattr(release, "stray_generated_entries", lambda root: [])
    result = release.stage_release(root, tmp_path / "staged", dry_run=True)
    assert "odock-chrome-hylyoa4p" in {name for name, _, _ in result.listing}
    assert "odock-chrome-hylyoa4p" in result.text()


def test_stage_writes_exactly_the_published_set(tmp_path):
    root = _tree(tmp_path)
    (root / "out" / "ignored.txt").write_text("x", encoding="utf-8")
    result = release.stage_release(root, tmp_path / "staged")
    published, _ = release.published_files(root)
    assert result.files == len(published)
    staged = sorted(
        release._project._as_posix(str(path.relative_to(tmp_path / "staged")))
        for path in (tmp_path / "staged").rglob("*")
        if path.is_file()
    )
    assert staged == published
    assert not (tmp_path / "staged" / "out").exists()
    assert result.bytes > 0 and len(result.manifest_sha256) == 64
    assert result.ignored > 0


def test_stage_dry_run_writes_nothing(tmp_path):
    root = _tree(tmp_path)
    result = release.stage_release(root, tmp_path / "staged", dry_run=True)
    assert result.dry_run is True
    assert not (tmp_path / "staged").exists()
    assert result.files > 0


def test_stage_refuses_a_non_empty_destination(tmp_path):
    root = _tree(tmp_path)
    destination = tmp_path / "staged"
    destination.mkdir()
    (destination / "stale.txt").write_text("old\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.stage_release(root, destination)
    assert "is not empty" in str(excinfo.value)
    assert "--force" in excinfo.value.fix
    # With --force the stale file is gone and the set matches.
    result = release.stage_release(root, destination, force=True)
    assert not (destination / "stale.txt").exists()
    assert result.files > 0


def test_stage_refuses_a_generated_artefact_in_the_published_set(tmp_path):
    root = _tree(tmp_path)
    # A `.so` is not ignored by this miniature .gitignore: it would ship.
    (root / "python" / "odock" / "_odock.so").write_bytes(b"\x00")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.stage_release(root, tmp_path / "staged", dry_run=True)
    assert "generated artefact" in str(excinfo.value)
    assert "_odock.so" in str(excinfo.value)


def test_stage_refuses_a_required_file_excluded_by_a_rule(tmp_path):
    root = _tree(tmp_path)
    (root / ".gitignore").write_text(_GITIGNORE + "\ndocs/\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.stage_release(root, tmp_path / "staged", dry_run=True)
    assert "would not be published" in str(excinfo.value)
    assert "docs/VALIDATION.md" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The changelog and the release body
# ---------------------------------------------------------------------------


def test_the_changelog_section_is_extracted(tmp_path):
    root = _tree(tmp_path)
    section = release.changelog_section(root / "CHANGELOG.md", "0.2.1")
    assert section.version == "0.2.1"
    assert section.date == "2026-10-01"
    assert "a project file, an HTML report" in section.body
    assert "### Fixed" in section.body
    # The next version's section is not included.
    assert "everything before that" not in section.body
    assert section.start_line == 3


def test_a_missing_section_is_refused_with_the_versions_present(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(release.ReleaseError) as excinfo:
        release.changelog_section(root / "CHANGELOG.md", "0.2.2")
    message = str(excinfo.value)
    assert "no section for 0.2.2" in message
    assert "0.2.1" in excinfo.value.fix  # the sections that do exist


def test_a_newer_heading_than_the_release_is_refused(tmp_path):
    root = _tree(tmp_path)
    with pytest.raises(release.ReleaseError) as excinfo:
        release.changelog_section(root / "CHANGELOG.md", "0.2.0")
    assert "newer than the version being released" in str(excinfo.value)


def test_an_empty_section_is_refused(tmp_path):
    root = _tree(tmp_path)
    path = root / "CHANGELOG.md"
    path.write_text(
        "# Changelog\n\n## 0.2.2 — 2026-10-01\n\n## 0.2.1 — 2026-10-01\n\n* old\n",
        encoding="utf-8",
    )
    with pytest.raises(release.ReleaseError, match="is empty"):
        release.changelog_section(path, "0.2.2")


def test_a_missing_changelog_is_refused(tmp_path):
    with pytest.raises(release.ReleaseError, match="does not exist"):
        release.changelog_section(tmp_path / "CHANGELOG.md", "0.2.1")


def test_the_release_body_is_json_without_a_bom(tmp_path):
    """The incident: `Set-Content -Encoding UTF8` wrote a BOM and GitHub rejected
    the release.  The assertion is on the raw first byte."""
    root = _tree(tmp_path)
    target, payload = release.write_release_notes(
        root / "CHANGELOG.md", "0.2.1", root / "out" / "release" / "0.2.1.json"
    )
    raw = target.read_bytes()
    assert raw[:1] == b"{"
    assert raw[:3] != b"\xef\xbb\xbf"
    assert b"\xef\xbb\xbf" not in raw[:8]
    assert raw.decode("utf-8").startswith("{")
    parsed = json.loads(raw.decode("utf-8"))
    assert parsed["tag_name"] == "v0.2.1"
    assert parsed["name"] == "OpenDocking 0.2.1"
    assert "a project file, an HTML report" in parsed["body"]
    assert parsed["draft"] is False and parsed["prerelease"] is False
    assert payload["changelog"]["version"] == "0.2.1"


def test_the_body_honours_a_custom_tag_prefix_and_title(tmp_path):
    root = _tree(tmp_path)
    payload = release.release_notes_body(
        root / "CHANGELOG.md", "0.2.1", tag_prefix="release-", title="Trypsin 1"
    )
    assert payload["tag_name"] == "release-0.2.1"
    assert payload["name"] == "Trypsin 1"


# ---------------------------------------------------------------------------
# Publishing: refused preconditions, and a verification that reads the remote
# ---------------------------------------------------------------------------


class FakeRunner:
    """A recording transport: no network, no git, no gh."""

    def __init__(self, *, responses=None, fail=(), dirty="", tag_exists=False,
                 remote_sha=None, remote_message=None, release_tag=None,
                 assets=None, commit="a" * 40, message="OpenDocking 0.2.2\n",
                 base_tree="c" * 40, created_tree="d" * 40, created_commit="e" * 40,
                 read_back_tree=None, read_back_parents=None, remote_tag_exists=False,
                 tag_sha=None, head_frozen=False):
        self.calls = []
        self.stdins = []
        self.responses = dict(responses or {})
        self.fail = set(fail)
        self.dirty = dirty
        self.tag_exists = tag_exists
        self.remote_sha = remote_sha if remote_sha is not None else commit
        self.remote_message = remote_message if remote_message is not None else message
        self.release_tag = release_tag
        self.assets = list(assets or [])
        self.commit = commit
        self.message = message
        self.base_tree = base_tree
        self.created_tree = created_tree
        self.created_commit = created_commit
        # What the read-back of the new commit reports; the truth unless a test
        # wants to simulate a build that produced the wrong object.
        self.read_back_tree = read_back_tree if read_back_tree is not None else created_tree
        self.read_back_parents = (
            read_back_parents if read_back_parents is not None else [self.remote_sha]
        )
        self.remote_tag_exists = remote_tag_exists
        # The remote state the verification reads back: a ref reports its new value
        # only after the call that moves it, exactly as the API behaves.  A frozen
        # head models a protected branch that refused the update; `tag_sha` models a
        # tag ref that points somewhere else than asked.
        self.tag_sha = tag_sha
        self.head_frozen = head_frozen
        self.branch_head = self.remote_sha
        self.remote_tags: dict = {}

    def __call__(self, argv, *, timeout=None, cwd=None, stdin=None):
        argv = [str(item) for item in argv]
        self.calls.append(argv)
        self.stdins.append(stdin or "")
        joined = " ".join(argv)
        # A canned response matches by substring, so a test can fail a whole step
        # without reproducing the exact argv the implementation builds.
        for needle, (code, out) in self.responses.items():
            if needle in joined:
                return release.TransportResult(argv, code, out, "", 0.01)
        if joined.startswith("git status"):
            return release.TransportResult(argv, 0, self.dirty, "", 0.01)
        if joined.startswith("git rev-parse HEAD"):
            return release.TransportResult(argv, 0, self.commit + "\n", "", 0.01)
        if joined.startswith("git rev-parse -q --verify refs/tags/"):
            return release.TransportResult(argv, 0 if self.tag_exists else 1, "", "", 0.01)
        if joined.startswith("git log -1"):
            return release.TransportResult(argv, 0, self.message, "", 0.01)
        if joined.startswith("git ") and any(part in self.fail for part in argv):
            return release.TransportResult(argv, 1, "", "failed", 0.01)
        if joined.startswith("gh repo view"):
            return release.TransportResult(argv, 0, "GrassBlock-WE/opendocking\n", "", 0.01)

        # The refs move only when the call that moves them happens.
        if "/git/refs/heads/" in joined and "PATCH" in argv:
            sha = self._field(argv, "sha")
            if sha and not self.head_frozen:
                self.branch_head = sha
            return release.TransportResult(argv, 0, json.dumps({"object": {"sha": sha}}), "", 0.01)
        if "/git/refs" in joined and self._field(argv, "ref"):
            reference = self._field(argv, "ref")
            sha = self._field(argv, "sha")
            if reference.startswith("refs/tags/"):
                self.remote_tags[reference.split("refs/tags/", 1)[1]] = sha
            return release.TransportResult(argv, 0, json.dumps({"ref": reference}), "", 0.01)
        if "/git/ref/tags/" in joined:
            tag = joined.split("/git/ref/tags/", 1)[1].split()[0]
            if tag in self.remote_tags or self.remote_tag_exists:
                sha = self.remote_tags.get(tag, self.remote_sha)
                if self.tag_sha is not None:
                    sha = self.tag_sha
                return release.TransportResult(
                    argv, 0, self._answer({"ref": f"refs/tags/{tag}",
                                           "object": {"sha": sha}}, argv), "", 0.01)
            return release.TransportResult(argv, 1, "", "Not Found", 0.01)
        if "/git/commits/" in joined and joined.rstrip().endswith(".tree.sha"):
            return release.TransportResult(argv, 0, self.base_tree + "\n", "", 0.01)
        if "/git/commits/" in joined and not self._is_write(argv):
            if self.created_commit in joined:
                # The read-back of the commit this tool just created.
                return release.TransportResult(argv, 0, json.dumps({
                    "sha": self.created_commit,
                    "tree": {"sha": self.read_back_tree},
                    "parents": [{"sha": sha} for sha in self.read_back_parents],
                }), "", 0.01)
            return release.TransportResult(argv, 0, json.dumps({
                "sha": self.remote_sha, "tree": {"sha": self.base_tree},
            }), "", 0.01)
        if "/git/blobs" in joined:
            index = sum(1 for call in self.calls if "/git/blobs" in " ".join(call))
            return release.TransportResult(argv, 0, json.dumps({"sha": f"{index:040x}"}), "", 0.01)
        if "/git/trees" in joined and "recursive" in joined:
            return release.TransportResult(argv, 0, json.dumps({"tree": [
                {"path": "pyproject.toml", "type": "blob", "sha": "1" * 40},
                {"path": "removed.txt", "type": "blob", "sha": "2" * 40},
            ]}), "", 0.01)
        if "/git/trees" in joined:
            return release.TransportResult(argv, 0, json.dumps({"sha": self.created_tree}), "", 0.01)
        if "/git/commits" in joined and self._is_write(argv):
            return release.TransportResult(argv, 0, json.dumps({"sha": self.created_commit}), "", 0.01)
        if "/commits/" in joined and joined.endswith(".sha"):
            return release.TransportResult(argv, 0, self.branch_head + "\n", "", 0.01)
        if "/commits/" in joined and joined.endswith(".commit.message"):
            return release.TransportResult(argv, 0, self.remote_message, "", 0.01)
        if "/releases/tags/" in joined:
            tag = joined.split("/releases/tags/", 1)[1].split()[0]
            payload = {
                "tag_name": self.release_tag if self.release_tag is not None else tag,
                "assets": [{"name": name} for name in self.assets],
            }
            return release.TransportResult(argv, 0, json.dumps(payload), "", 0.01)
        return release.TransportResult(argv, 0, "", "", 0.01)

    @staticmethod
    def _field(argv, name):
        """The value of a `-f name=value` argument, if present."""
        for index, item in enumerate(argv):
            if item == "-f" and index + 1 < len(argv):
                candidate = argv[index + 1]
                if candidate.startswith(name + "="):
                    return candidate.split("=", 1)[1]
        return ""

    @staticmethod
    def _answer(payload, argv) -> str:
        """The response body, honouring `-q .a.b` the way `gh api` does.

        Without this the fake answered every read with raw JSON, and the
        verification compared JSON text against a sha — which would have made the
        tool look wrong for a transport that was wrong.
        """
        if "-q" not in argv:
            return json.dumps(payload)
        expression = argv[argv.index("-q") + 1]
        value = payload
        for part in expression.strip(".").split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return "" if value is None else str(value)

    @staticmethod
    def _is_write(argv) -> bool:
        return "--input" in argv or "POST" in argv


def _passing_check(**overrides) -> dict:
    payload = {
        "format": "odock-release-check",
        "ok": True,
        "gates": [{"name": "pytest suite", "ok": True}],
    }
    payload.update(overrides)
    return payload


def test_publish_verifies_its_own_result_against_the_remote(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner()
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner,
        message="OpenDocking 0.2.2\n\ncut mechanically",
    )
    assert result.ok, result.text()
    commands = [" ".join(call) for call in runner.calls]
    assert any(call.startswith("git tag -a v0.2.2") for call in commands)
    assert any("repos/GrassBlock-WE/opendocking/git/refs/heads/main" in call for call in commands)
    assert any("ref=refs/tags/v0.2.2" in call for call in commands)
    assert any(call.startswith("gh release create v0.2.2") for call in commands)
    # The verification read the remote back and compared four things.
    kinds = {item["what"] for item in result.verification}
    assert "the remote main head" in kinds
    assert "the remote commit message" in kinds
    assert "the tag v0.2.2 target" in kinds
    assert "the release tag" in kinds
    assert "the release file count" in kinds
    assert all(item["ok"] for item in result.verification)
    # The notes file was written, and it is BOM-free.
    notes = root / "out" / "release" / "0.2.2.json"
    assert notes.read_bytes()[:1] == b"{"
    assert any("release notes" == step["name"] and step["ok"] for step in result.steps)


def test_publish_reports_a_remote_that_disagrees(tmp_path):
    """A verification that cannot fail is not a verification.

    Three independent lies are modelled: a **frozen branch head** (a protected
    branch that refused the update), a **tag ref pointing elsewhere**, and a
    release whose tag is not the one asked for.  The branch-head lie is the reason
    the transport is stateful: after a push that *worked*, the head does match, and
    a transport that always reported the old head would have made the tool look
    wrong for a fake that was wrong.
    """
    root = _tree(tmp_path, version="0.2.2")
    # The remote starts somewhere else and refuses to move (a protected branch).
    runner = FakeRunner(remote_sha="b" * 40, release_tag="v0.2.1", tag_sha="f" * 40,
                        head_frozen=True)
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner
    )
    assert not result.ok
    failed = [item["what"] for item in result.verification if not item["ok"]]
    assert "the remote main head" in failed
    assert "the tag v0.2.2 target" in failed
    assert "the release tag" in failed
    assert any("does not match" in problem for problem in result.problems)
    assert "FAIL" in result.text()


def test_publish_succeeds_when_the_remote_agrees(tmp_path):
    """The other half: with a transport that behaves, the same code reports every
    check green — so the failures above are the lies, not a permanently unhappy
    verification."""
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner()
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner
    )
    assert result.ok, result.text()
    assert [item["ok"] for item in result.verification] == [True] * len(result.verification)
    assert runner.branch_head == result.commit


def test_publish_refuses_without_a_passing_check_report(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    with pytest.raises(release.ReleaseError, match="no `release check` report was given"):
        release.publish_release(root, "0.2.2", runner=FakeRunner())
    with pytest.raises(release.ReleaseError, match="did not pass"):
        release.publish_release(
            root, "0.2.2", runner=FakeRunner(),
            check_report={"ok": False, "gates": [{"name": "pytest suite", "ok": False}]},
        )
    report_file = tmp_path / "report.json"
    with pytest.raises(release.ReleaseError, match="no release check report at"):
        release.publish_release(root, "0.2.2", runner=FakeRunner(), check_report=report_file)


def test_publish_refuses_a_stale_commit_message(tmp_path):
    """The 0.2.1 commit was titled 0.2.0 by a reused message file."""
    root = _tree(tmp_path, version="0.2.2")
    message = tmp_path / "message.txt"
    message.write_text("OpenDocking 0.2.0\n\nwhat changed\n", encoding="utf-8")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=FakeRunner(),
            message_file=message,
        )
    assert "names 0.2.0" in str(excinfo.value)
    assert "stale message file" in excinfo.value.fix


def test_publish_derives_the_commit_message_from_the_version(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner(message="OpenDocking 0.2.2\n")
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner
    )
    assert result.ok
    tag_calls = [call for call in runner.calls if call[:3] == ["git", "tag", "-a"]]
    assert tag_calls, "the tag must be created with the derived message"
    # `git tag -a <tag> -m <message>`: the message follows the -m.
    assert tag_calls[0][3] == "v0.2.2"
    assert tag_calls[0][5] == "OpenDocking 0.2.2"


def test_publish_refuses_a_dirty_tree_with_unrelated_changes(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner(dirty=" M python/odock/project.py\n")
    with pytest.raises(release.ReleaseError) as excinfo:
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=runner
        )
    assert "not this release's version bump" in str(excinfo.value)
    assert "python/odock/project.py" in str(excinfo.value)


def test_publish_commits_only_the_version_bump(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner(dirty=" M pyproject.toml\n M Cargo.toml\n")
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner
    )
    assert result.ok, result.text()
    adds = [call for call in runner.calls if call[:2] == ["git", "add"]]
    assert adds and set(adds[0][3:]) == {"pyproject.toml", "Cargo.toml"}
    commits = [call for call in runner.calls if call[:2] == ["git", "commit"]]
    assert commits and "OpenDocking 0.2.2" in commits[0][-1]


def test_publish_refuses_an_existing_tag(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner(tag_exists=True)
    with pytest.raises(release.ReleaseError, match="already exists locally"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=runner
        )


def test_publish_refuses_a_version_without_a_changelog_section(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    with pytest.raises(release.ReleaseError, match="no section for 0.2.3"):
        release.publish_release(
            root, "0.2.3", check_report=_passing_check(), runner=FakeRunner()
        )


def test_publish_dry_run_changes_nothing(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner()
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner, dry_run=True
    )
    assert result.dry_run is True
    assert result.steps[-1]["name"] == "dry run"
    assert not any(call[:2] == ["git", "tag"] for call in runner.calls)
    assert not any(call[0] == "gh" for call in runner.calls)
    # The plan names every step it would take, including the body's provenance.
    names = [step["name"] for step in result.steps]
    assert "release body" in names
    assert "git tag" in names
    assert "gh release create" in names


def test_publish_dry_run_plans_without_a_git_worktree(tmp_path):
    """This project's own agent workspace has no `.git`, so a plan must still be
    possible there — saying what could not be checked rather than refusing."""
    root = _tree(tmp_path, version="0.2.2")

    def no_worktree(argv, *, timeout=None, cwd=None):
        argv = [str(item) for item in argv]
        if argv[:2] == ["git", "status"]:
            return release.TransportResult(
                argv, 128, "", "fatal: not a git repository (or any parent)", 0.01
            )
        return release.TransportResult(argv, 0, "", "", 0.01)

    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=no_worktree, dry_run=True
    )
    assert result.dry_run is True
    assert any("could not run" in note and "git status" in note for note in result.notes)
    details = {step["name"]: step["detail"] for step in result.steps}
    assert details["clean-tree check"] == "not verifiable here: this is not a git worktree"
    assert details["git tag"].startswith("would create the annotated tag v0.2.2")
    # The git error must not become the commit message (it did once, through a
    # shadowed local variable).
    commit_notes = [note for note in result.notes if note.startswith("commit message")]
    assert commit_notes and "OpenDocking 0.2.2" in commit_notes[0]
    assert "not a git repository" not in commit_notes[0]
    # Without a dry run the same situation is refused.
    with pytest.raises(release.ReleaseError, match="git worktree"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=no_worktree
        )


def test_publish_reads_the_repository_from_the_manifests_when_there_is_no_remote(tmp_path):
    """`gh repo view` needs a git remote; a worktree-less directory has none, so the
    slug comes from the manifests naming the repository."""
    root = _tree(tmp_path, version="0.2.2")
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    (root / "pyproject.toml").write_text(
        text + '\n[project.urls]\nRepository = "https://github.com/Owner-Name/opendocking"\n',
        encoding="utf-8",
    )
    sha = "b" * 40

    def no_remote(argv, *, timeout=None, cwd=None, stdin=None):
        argv = [str(item) for item in argv]
        if argv[:3] == ["gh", "repo", "view"]:
            return release.TransportResult(argv, 1, "", "no remote", 0.01)
        return FakeRunner(remote_sha=sha)(argv, timeout=timeout, cwd=cwd, stdin=stdin)

    seen: list = []

    def recording(argv, *, timeout=None, cwd=None, stdin=None):
        seen.append(" ".join(str(item) for item in argv))
        return no_remote(argv, timeout=timeout, cwd=cwd, stdin=stdin)

    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=recording, commit=sha,
        dry_run=True,
    )
    assert any("Owner-Name/opendocking" in call for call in seen), seen
    assert result.dry_run is True
    assert result.commit == sha


def test_publish_refuses_when_no_repository_can_be_determined(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"x\"\nversion = \"0.2.2\"\n", encoding="utf-8"
    )
    (root / "Cargo.toml").write_text(
        "[workspace]\n[workspace.package]\nversion = \"0.2.2\"\n", encoding="utf-8"
    )

    def no_remote(argv, *, timeout=None, cwd=None, stdin=None):
        argv = [str(item) for item in argv]
        if argv[:3] == ["gh", "repo", "view"]:
            return release.TransportResult(argv, 1, "", "no remote", 0.01)
        return release.TransportResult(argv, 0, "", "", 0.01)

    with pytest.raises(release.ReleaseError, match="could not work out the GitHub"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=no_remote,
            commit="c" * 40, dry_run=True,
        )


def test_publish_reports_a_failed_api_step(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    runner = FakeRunner()
    runner.responses["gh release create v0.2.2 --title OpenDocking 0.2.2"] = (1, "")
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner
    )
    # The step is recorded as failed even when the verification happens to agree.
    failed = [step for step in result.steps if not step["ok"]]
    assert failed, result.text()


# ---------------------------------------------------------------------------
# Publishing without a git worktree: --commit and --api-commit
# ---------------------------------------------------------------------------


def test_publish_accepts_a_commit_built_elsewhere(tmp_path):
    """This project's own workspace has no `.git`; the 0.2.1 commit was built
    through the API by hand.  `--commit` makes that a supported input: no worktree
    is touched, and the tag ref, the release and the verification are automated."""
    root = _tree(tmp_path, version="0.2.2")
    sha = "b" * 40
    runner = FakeRunner(remote_sha=sha)
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner, commit=sha,
    )
    assert result.ok, result.text()
    assert result.commit == sha
    commands = [" ".join(call) for call in runner.calls]
    assert not any(call.startswith("git status") for call in commands)
    assert not any(call.startswith("git tag") for call in commands)
    assert any("ref=refs/tags/v0.2.2" in call for call in commands)
    assert any("repos/GrassBlock-WE/opendocking/git/refs/heads/main" in call for call in commands)
    assert any(call.startswith("gh release create v0.2.2") for call in commands)
    assert all(item["ok"] for item in result.verification)
    assert any(step["name"] == "commit supplied" for step in result.steps)


def test_publish_refuses_a_commit_that_is_not_on_the_remote(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    sha = "b" * 40
    runner = FakeRunner()
    runner.responses[f"/git/commits/{sha}"] = (1, "")
    with pytest.raises(release.ReleaseError, match="does not exist on"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=runner, commit=sha,
        )


def test_publish_refuses_a_short_or_malformed_commit(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    for bad in ("abc123", "not-a-sha", "", "z" * 40):
        with pytest.raises(release.ReleaseError, match="40-character commit id"):
            release.publish_release(
                root, "0.2.2", check_report=_passing_check(), runner=FakeRunner(),
                commit=bad,
            )


def test_publish_refuses_a_tag_that_already_exists_on_the_remote(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    sha = "b" * 40
    runner = FakeRunner(remote_sha=sha, remote_tag_exists=True)
    with pytest.raises(release.ReleaseError, match="already exists on"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=runner, commit=sha,
        )


def test_api_commit_builds_the_object_graph_and_verifies_it_before_moving_a_ref(tmp_path):
    """The whole safety argument for `--api-commit`: blobs → tree (with deletions) →
    commit → **read back and compare** → only then the branch ref."""
    root = _tree(tmp_path, version="0.2.2")
    staged = tmp_path / "staged"
    release.stage_release(root, staged)
    parent, new_commit, new_tree, base_tree = "a" * 40, "e" * 40, "d" * 40, "c" * 40
    runner = FakeRunner(remote_sha=parent, created_commit=new_commit,
                        created_tree=new_tree, base_tree=base_tree)
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner,
        api_commit=True, tree_dir=staged,
    )
    assert result.ok, result.text()
    assert result.commit == new_commit

    commands = [" ".join(call) for call in runner.calls]

    def first(needle: str) -> int:
        for index, call in enumerate(commands):
            if needle in call:
                return index
        raise AssertionError(f"no call contained {needle!r}")

    head_read = first(f"git/commits/{parent}")
    base_tree_read = first(f"git/trees/{base_tree}?recursive=1")
    blob_calls = [index for index, call in enumerate(commands) if "/git/blobs" in call]
    tree_write = first("git/trees --input")
    commit_write = first("git/commits --input")
    read_back = first(f"git/commits/{new_commit}")
    ref_update = first("git/refs/heads/main")
    assert head_read < base_tree_read < blob_calls[0] < tree_write < commit_write
    assert read_back < ref_update, "the ref must only move after the read-back"
    assert any(step["name"] == "verify the new commit" for step in result.steps)
    assert any(step["name"] == "upload blobs" for step in result.steps)
    # One blob per staged file, each body on stdin (never in argv), and the
    # staging set is what was uploaded.
    published, _ = release.published_files(root)
    assert len(blob_calls) >= len(published) - 3  # the stubs differ slightly from the tree
    blob_stdins = [
        stdin for call, stdin in zip(runner.calls, runner.stdins)
        if "/git/blobs" in " ".join(call)
    ]
    assert blob_stdins and all('"encoding": "base64"' in stdin for stdin in blob_stdins)
    assert all(len(call) < 500 for call in commands)


def test_api_commit_asks_for_the_deletions(tmp_path):
    """The tree API with a base tree only adds and updates, so a file that left the
    staged set has to be deleted explicitly."""
    root = _tree(tmp_path, version="0.2.2")
    staged = tmp_path / "staged"
    release.stage_release(root, staged)
    runner = FakeRunner()
    release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner,
        api_commit=True, tree_dir=staged,
    )
    payloads = [
        json.loads(stdin) for call, stdin in zip(runner.calls, runner.stdins)
        if stdin and "git/trees" in " ".join(call)
    ]
    assert payloads, "the tree payload must be sent"
    entries = payloads[0]["tree"]
    # `removed.txt` is in the fake remote tree and not in the staged set.
    assert {"path": "removed.txt", "mode": "100644", "type": "blob", "sha": None} in entries
    assert payloads[0]["base_tree"] == "c" * 40
    assert all(entry["sha"] is not None for entry in entries if entry["path"] != "removed.txt")


def test_api_commit_refuses_when_the_read_back_disagrees(tmp_path):
    """A successful-looking response that built the wrong object must stop before
    any ref moves."""
    root = _tree(tmp_path, version="0.2.2")
    staged = tmp_path / "staged"
    release.stage_release(root, staged)
    runner = FakeRunner(read_back_tree="f" * 40)  # not the tree that was created
    with pytest.raises(release.ReleaseError) as excinfo:
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=runner,
            api_commit=True, tree_dir=staged,
        )
    message = str(excinfo.value)
    assert "does not match what was intended" in message
    assert "nothing has been moved" in excinfo.value.fix
    assert not any("git/refs/heads" in " ".join(call) for call in runner.calls)
    assert not any(call[:3] == ["gh", "release", "create"] for call in runner.calls)


def test_api_commit_refuses_a_wrong_parent(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    staged = tmp_path / "staged"
    release.stage_release(root, staged)
    runner = FakeRunner(read_back_parents=["9" * 40])
    with pytest.raises(release.ReleaseError, match="does not match what was intended"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=runner,
            api_commit=True, tree_dir=staged,
        )
    assert not any("git/refs/heads" in " ".join(call) for call in runner.calls)


def test_api_commit_needs_a_tree_directory(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    with pytest.raises(release.ReleaseError, match="--api-commit needs the directory"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=FakeRunner(),
            api_commit=True,
        )
    with pytest.raises(release.ReleaseError, match="is not a directory"):
        release.publish_release(
            root, "0.2.2", check_report=_passing_check(), runner=FakeRunner(),
            api_commit=True, tree_dir=tmp_path / "nope",
        )


def test_api_commit_dry_run_plans_without_building(tmp_path):
    root = _tree(tmp_path, version="0.2.2")
    staged = tmp_path / "staged"
    release.stage_release(root, staged)
    runner = FakeRunner()
    result = release.publish_release(
        root, "0.2.2", check_report=_passing_check(), runner=runner,
        api_commit=True, tree_dir=staged, dry_run=True,
    )
    assert result.dry_run is True
    names = [step["name"] for step in result.steps]
    assert "read the remote base" in names
    assert "build the commit (API)" in names
    assert not any("/git/blobs" in " ".join(call) for call in runner.calls)
    assert not any(call[:3] == ["gh", "release", "create"] for call in runner.calls)


# ---------------------------------------------------------------------------
# `release check` aggregates every gate
# ---------------------------------------------------------------------------


def _gate_runner(behaviour: dict):
    """A runner that fails or passes by the gate's command, and reports."""

    def runner(argv, *, timeout=None, cwd=None):
        argv = [str(item) for item in argv]
        joined = " ".join(argv)
        for needle, (code, out) in behaviour.items():
            if needle in joined:
                return release.TransportResult(argv, code, out, "", 0.5)
        return release.TransportResult(argv, 0, "", "", 0.5)

    return runner


def test_release_check_aggregates_a_failure_from_every_gate(tmp_path):
    root = _tree(tmp_path)
    failing = {
        "inspect_dist.py": (1, "[LEAK] one thing"),
        "release_check.py": (1, "=== RESULT: FAIL ==="),
        "docs check": (1, json.dumps({
            "ok": False,
            "counts": {"ok": 4, "warning": 1, "error": 2},
            "findings": [{"severity": "error", "title": "2 internal link(s) do not resolve"}],
        })),
        "odock.cli doctor": (1, json.dumps({
            "counts": {"warning": 2, "error": 0},
            "findings": [{"severity": "warning", "title": "the versions disagree"}],
        })),
        "odock.benchmark": (1, "3PTB: top_rmsd 2.00 A is worse than the baseline"),
        "pytest": (1, "1 failed, 10 passed in 5s"),
        "check_test_order.py": (1, "ORDER DEPENDENCE: these tests changed their outcome"),
    }
    report = release.check_release(root, runner=_gate_runner(failing), timeout=5)
    names = {gate.name for gate in report.gates}
    assert names == {
        "release content", "release_check.py", "inspect_dist self-test", "docs site",
        "doctor --strict", "benchmark --check-baseline", "pytest suite", "test order",
    }
    failed = {gate.name for gate in report.failures()}
    assert failed == {
        "release_check.py", "inspect_dist self-test", "docs site", "doctor --strict",
        "benchmark --check-baseline", "pytest suite", "test order",
    }
    docs_gate = next(gate for gate in report.gates if gate.name == "docs site")
    assert "2 error(s)" in docs_gate.detail
    assert "do not resolve" in docs_gate.detail
    assert report.ok is False
    assert "FAIL" in report.text()
    payload = report.as_dict()
    assert payload["ok"] is False
    assert len(payload["gates"]) == 8


def test_release_check_reports_each_gate_with_measured_numbers(tmp_path):
    root = _tree(tmp_path)
    passing = {
        "inspect_dist.py": (0, "[ok] rule one\n[ok] rule two\n"),
        "pytest": (0, "42 passed, 1 skipped in 12.3s"),
        "odock.cli doctor": (0, json.dumps({"counts": {"warning": 0, "error": 0}, "findings": []})),
        "odock.benchmark": (0, "all systems within tolerance"),
        "check_test_order.py": (0, ORDER_GREEN),
    }
    report = release.check_release(root, runner=_gate_runner(passing), timeout=5)
    by_name = {gate.name: gate for gate in report.gates}
    assert by_name["pytest suite"].numbers == {"passed": 42, "failed": 0, "skipped": 1}
    assert by_name["inspect_dist self-test"].numbers["ok_rules"] == 2
    assert by_name["doctor --strict"].ok is True
    assert by_name["benchmark --check-baseline"].ok is True
    assert by_name["test order"].numbers["failures"] == 0
    assert "both orders agree" in by_name["test order"].detail
    # The two content gates have no script runner in a synthetic tree.
    assert by_name["release content"].ok is True
    assert by_name["release content"].numbers["published"] > 0


ORDER_GREEN = """order gate: 40 test file(s), two orders

  forward order: 0 failure(s)
  reverse order: 0 failure(s)

the two orders agree: every failure is a property of the test, not of the order it ran in
"""

ORDER_REPRODUCIBLY_RED = """order gate: 40 test file(s), two orders

  forward order: 2 failure(s)
    FAILED tests/test_one.py::test_a
    FAILED tests/test_two.py::test_b
  reverse order: 2 failure(s)
    FAILED tests/test_one.py::test_a
    FAILED tests/test_two.py::test_b

the two orders agree: every failure is a property of the test, not of the order it ran in
  (the suite is red, but reproducibly red - fix the tests above)
"""

ORDER_DEPENDENT = """order gate: 40 test file(s), two orders

  forward order: 1 failure(s)
    FAILED tests/test_one.py::test_a
  reverse order: 0 failure(s)

ORDER DEPENDENCE: these tests changed their outcome with the file order
    fails only in the forward order: tests/test_one.py::test_a
"""


def test_the_order_gate_reads_all_three_verdicts(tmp_path):
    """The gate that distinguishes *broken* from *order-dependent* from *flaky*.

    Reproducibly red is exit 0 from the script — both orders agree — and the gate
    must say so rather than reporting a pass: a suite that fails the same way twice
    is a defect, not an ordering artefact.
    """
    root = _tree(tmp_path)

    green = release.check_release(
        root, runner=_gate_runner({"check_test_order.py": (0, ORDER_GREEN)}), timeout=5
    )
    gate = next(item for item in green.gates if item.name == "test order")
    assert gate.ok is True
    assert gate.numbers["failures"] == 0
    assert "both orders agree" in gate.detail

    red = release.check_release(
        root,
        runner=_gate_runner({
            "check_test_order.py": (0, ORDER_REPRODUCIBLY_RED),
            "pytest": (1, "2 failed, 10 passed in 5s"),
        }),
        timeout=5,
    )
    gate = next(item for item in red.gates if item.name == "test order")
    assert gate.ok is True
    assert gate.numbers["failures"] == 2
    assert "reproducibly red" in gate.detail

    dependent = release.check_release(
        root,
        runner=_gate_runner({
            "check_test_order.py": (1, ORDER_DEPENDENT),
            "pytest": (1, "1 failed, 10 passed in 5s"),
        }),
        timeout=5,
    )
    gate = next(item for item in dependent.gates if item.name == "test order")
    assert gate.ok is False
    assert gate.numbers["order_dependent"] is True
    assert gate.numbers["changed"] == ["tests/test_one.py::test_a"]
    assert "ORDER DEPENDENT" in gate.detail
    assert "tests/test_one.py::test_a" in gate.detail
    assert "test order" in {gate.name for gate in dependent.failures()}


def test_the_order_gate_names_the_flaky_signal(tmp_path):
    """The third state needs both gates together: the order check says both orders
    agree and neither failed, while the suite gate is red.  Neither "broken" nor
    "order-dependent" — hunt shared state."""
    root = _tree(tmp_path)
    report = release.check_release(
        root,
        runner=_gate_runner({
            "check_test_order.py": (0, ORDER_GREEN),
            "pytest": (1, "1 failed, 10 passed in 5s"),
        }),
        timeout=5,
    )
    gate = next(item for item in report.gates if item.name == "test order")
    assert gate.numbers.get("flaky_signal") is True
    assert "FLAKY SIGNAL" in gate.detail
    assert "shared state" in gate.detail
    # And it does not appear when the suite is green.
    clean = release.check_release(
        root,
        runner=_gate_runner({"check_test_order.py": (0, ORDER_GREEN), "pytest": (0, "42 passed")}),
        timeout=5,
    )
    assert "flaky_signal" not in next(
        item for item in clean.gates if item.name == "test order"
    ).numbers


def test_the_order_gate_is_skipped_when_the_script_is_absent(tmp_path):
    root = _tree(tmp_path)
    (root / "tools" / "check_test_order.py").unlink()
    report = release.check_release(
        root, only=["test order"], runner=_gate_runner({}), timeout=5
    )
    gate = report.gates[0]
    assert gate.skipped is True and gate.ok is True
    assert "not in this tree" in gate.detail


def test_release_check_names_every_failing_test(tmp_path):
    """The first end-to-end run reported "14 failed" and, in a 300-character tail,
    only three of the names — so the gate now lists them."""
    root = _tree(tmp_path)
    output = (
        "FAILED tests/test_one.py::test_a - AssertionError\n"
        "FAILED tests/test_two.py::test_b - RuntimeError: no GL context\n"
        "FAILED tests/test_three.py::test_c - AssertionError\n"
        "3 failed, 10 passed in 5s\n"
    )
    report = release.check_release(root, runner=_gate_runner({"pytest": (1, output)}), timeout=5)
    gate = next(item for item in report.gates if item.name == "pytest suite")
    assert gate.numbers["failing_total"] == 3
    assert gate.numbers["failing"] == [
        "tests/test_one.py::test_a",
        "tests/test_two.py::test_b",
        "tests/test_three.py::test_c",
    ]
    assert "3 test(s) failed" in gate.detail
    assert "tests/test_one.py::test_a" in gate.detail


def test_release_check_quick_validates_instead_of_redocking(tmp_path):
    root = _tree(tmp_path)
    runner = _gate_runner({})
    report = release.check_release(root, quick=True, runner=runner, timeout=5)
    by_name = {gate.name: gate for gate in report.gates}
    assert "benchmark baseline" in by_name
    assert "benchmark --check-baseline" not in by_name
    assert by_name["benchmark baseline"].numbers.get("seconds") is not None
    pytest_command = " ".join(by_name["pytest suite"].command)
    assert "-m not slow" in pytest_command


def test_release_check_can_run_one_gate(tmp_path):
    root = _tree(tmp_path)
    report = release.check_release(
        root, only=["release content"], runner=_gate_runner({}), timeout=5
    )
    assert [gate.name for gate in report.gates] == ["release content"]


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_release_prepare_stage_notes(tmp_path, capsys):
    from odock.cli import main

    root = _tree(tmp_path)
    out = tmp_path / "staged"
    assert main(["release", "prepare", "0.2.2", "--root", str(root), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "0.2.2" and payload["previous"] == "0.2.1"
    assert "version = \"0.2.2\"" in (root / "pyproject.toml").read_text(encoding="utf-8")

    assert main(["release", "prepare", "0.3.0", "--root", str(root)]) == 2
    assert "--minor" in capsys.readouterr().err

    assert main(["release", "stage", str(out), "--root", str(root), "--json"]) == 0
    staged = json.loads(capsys.readouterr().out)
    assert staged["files"] > 0 and not staged["dry_run"]

    notes = tmp_path / "body.json"
    assert main(["release", "notes", "0.2.2", "--root", str(root), "-o", str(notes)]) == 2
    assert "no section for 0.2.2" in capsys.readouterr().err

    assert main(["release", "notes", "0.2.1", "--root", str(root), "-o", str(notes)]) == 0
    captured = capsys.readouterr()
    assert "no BOM" in captured.out
    assert notes.read_bytes()[:1] == b"{"


def test_cli_release_stage_refuses_on_stray_temp(tmp_path, capsys):
    from odock.cli import main

    root = _tree(tmp_path)
    (root / "tmpstray").mkdir()
    assert main(["release", "stage", str(tmp_path / "out"), "--root", str(root)]) == 2
    assert "generated entry" in capsys.readouterr().err


def test_cli_release_stage_prints_what_will_be_published(tmp_path, capsys):
    """The review step: the grouped listing is printed, and it is what a maintainer
    reads before `publish` (`docs/RELEASE.md` makes that a required step)."""
    from odock.cli import main

    root = _tree(tmp_path)
    assert main(["release", "stage", str(tmp_path / "staged"), "--root", str(root)]) == 0
    out = capsys.readouterr().out
    assert "WHAT WILL BE PUBLISHED" in out
    assert "docs" in out and "python" in out
    assert "read this before `release publish`" in out
    # The JSON form carries the same grouping, so a script can review it too.
    assert main(["release", "stage", str(tmp_path / "staged2"), "--root", str(root),
                 "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["entry"] for entry in payload["entries"]} >= {"docs", "python"}


def test_cli_release_check_writes_a_report(tmp_path, capsys, monkeypatch):
    from odock import release as module
    from odock.cli import main

    report = module.CheckReport(root=tmp_path, gates=[
        module.CheckGate("release content", True, "clean", {"published": 3}),
    ], seconds=1.0, created_utc="2026-01-01T00:00:00Z")
    monkeypatch.setattr(module, "check_release", lambda *a, **k: report)
    target = tmp_path / "check.json"
    assert main(["release", "check", "-o", str(target)]) == 0
    assert "PASS" in capsys.readouterr().out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["ok"] is True and payload["gates"][0]["name"] == "release content"

    failing = module.CheckReport(root=tmp_path, gates=[
        module.CheckGate("pytest suite", False, "1 failed"),
    ], seconds=2.0)
    monkeypatch.setattr(module, "check_release", lambda *a, **k: failing)
    assert main(["release", "check", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_release_publish_reports_a_missing_report(tmp_path, capsys):
    from odock.cli import main

    root = _tree(tmp_path, version="0.2.2")
    assert main(["release", "publish", "0.2.2", "--root", str(root)]) == 2
    assert "no `release check` report was given" in capsys.readouterr().err


def test_cli_release_publish_dry_run_needs_no_network(tmp_path, capsys, monkeypatch):
    from odock import release as module
    from odock.cli import main

    root = _tree(tmp_path, version="0.2.2")
    report = tmp_path / "check.json"
    report.write_text(json.dumps(_passing_check()), encoding="utf-8")
    # The transport is injected as a module attribute so the CLI stays offline.
    fake = FakeRunner()
    monkeypatch.setattr(module, "run_command", fake)
    assert main([
        "release", "publish", "0.2.2", "--root", str(root),
        "--check-report", str(report), "--dry-run", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True and payload["tag"] == "v0.2.2"
    assert not any(call[:2] == ["git", "tag"] for call in fake.calls)

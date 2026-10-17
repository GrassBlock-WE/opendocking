# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock._fs``: removing a scratch tree that Windows has marked read-only.

Every test here pins a failure that actually happened, not a hypothetical one:

* a stale ``out/tutorial`` carrying the read-only attribute made ``odock tutorial``
  -- and the two documentation tests that run it -- exit 1 with
  ``PermissionError: [WinError 5]`` before doing any work;
* a stale ``out/docs-site/assets/images`` did the same to ``odock docs check``,
  which is the gate ``release check`` runs.

Both are the same platform quirk at two call sites, which is why the behaviour
lives in one helper rather than in two copies.
"""

from __future__ import annotations

import os
import stat

import pytest

from odock._fs import remove_tree


def _read_only_tree(root):
    """A tree shaped like the one an interrupted run leaves behind."""
    images = root / "assets" / "images"
    images.mkdir(parents=True)
    picture = images / "cutout.png"
    picture.write_bytes(b"\x89PNG" + b"x" * 24)
    os.chmod(picture, stat.S_IREAD)
    os.chmod(images, stat.S_IREAD | stat.S_IEXEC)
    return images, picture


def test_removes_a_tree_whose_files_and_directories_are_read_only(tmp_path):
    """The measured failure: ``shutil.rmtree`` raises ``WinError 5`` here."""
    target = tmp_path / "out" / "tutorial"
    target.mkdir(parents=True)
    images, picture = _read_only_tree(target)
    # The premise of the test: the attributes really are clear of the write bit.
    assert not picture.stat().st_mode & stat.S_IWRITE
    assert not images.stat().st_mode & stat.S_IWRITE

    remove_tree(target)

    assert not target.exists()


def test_a_missing_path_is_not_an_error(tmp_path):
    """Callers use this to clear a previous run, which usually did not happen."""
    remove_tree(tmp_path / "never-existed")


def test_it_removes_a_plain_tree_too(tmp_path):
    target = tmp_path / "scratch"
    (target / "a" / "b").mkdir(parents=True)
    (target / "a" / "b" / "file.txt").write_text("hello", encoding="utf-8")

    remove_tree(target)

    assert not target.exists()


def test_it_does_not_follow_a_path_out_of_the_tree(tmp_path):
    """A sibling must survive: this removes *one* directory, not a neighbourhood."""
    target = tmp_path / "scratch"
    sibling = tmp_path / "keep-me"
    target.mkdir()
    sibling.mkdir()
    (sibling / "important.txt").write_text("stay", encoding="utf-8")

    remove_tree(target)

    assert not target.exists()
    assert (sibling / "important.txt").read_text(encoding="utf-8") == "stay"


def test_a_directory_that_will_not_go_is_reported_with_its_path(tmp_path, monkeypatch):
    """The error must name the directory: "Access is denied" alone is not
    actionable, which is exactly what the traceback did before."""

    def stubborn(_target):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr("shutil.rmtree", stubborn)
    target = tmp_path / "held-open"
    target.mkdir()

    with pytest.raises(OSError) as raised:
        remove_tree(target)

    assert str(target) in str(raised.value)
    assert "holding it open" in str(raised.value)


def test_a_tree_that_survives_removal_is_reported_rather_than_assumed_gone(tmp_path, monkeypatch):
    """``rmtree`` returning is not evidence the tree is gone -- a caller that
    assumes it is will fail later, somewhere stranger."""
    target = tmp_path / "reappears"
    target.mkdir()

    monkeypatch.setattr("shutil.rmtree", lambda _path: None)  # "succeeds" silently

    with pytest.raises(OSError) as raised:
        remove_tree(target)

    assert str(target) in str(raised.value)
    assert "still exists" in str(raised.value)

def test_removes_an_empty_directory_that_is_itself_read_only(tmp_path):
    """The second measured failure, and the one my first fix missed.

    ``os.walk`` yields the root's *children*, so a read-only **empty** directory
    never had its own flag cleared -- ``os.rmdir`` then failed with ``WinError 5``
    on a directory that looked empty to the code.  This is exactly ``out/docs-site``
    after ``odock docs check``, which is the gate ``release check`` runs.
    """
    target = tmp_path / "out" / "docs-site"
    target.mkdir(parents=True)
    os.chmod(target, stat.S_IREAD | stat.S_IEXEC)
    assert not target.stat().st_mode & stat.S_IWRITE
    assert list(target.iterdir()) == []

    remove_tree(target)

    assert not target.exists()


def test_removes_a_read_only_directory_that_contains_a_read_only_subdirectory(tmp_path):
    """Both levels read-only: the attribute is cleared on every entry, not one."""
    target = tmp_path / "nested"
    inner = target / "assets" / "images"
    inner.mkdir(parents=True)
    os.chmod(inner, stat.S_IREAD | stat.S_IEXEC)
    os.chmod(target / "assets", stat.S_IREAD | stat.S_IEXEC)
    os.chmod(target, stat.S_IREAD | stat.S_IEXEC)

    remove_tree(target)

    assert not target.exists()
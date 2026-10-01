# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetching a chemical component from the RCSB.

No test here touches the network: `_download` is monkeypatched, so what is
pinned is the URL selection and the fallback, which is where the bug was.
``files.rcsb.org/ligands/download`` serves the *ideal* and *model* coordinate
variants of a component (``BTN_ideal.sdf``); the bare ``BTN.sdf`` path that the
service used to serve now answers 404, and the file variant is lower-case even
though the component ID is upper-case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odock import fetch


def test_a_bare_component_id_tries_the_ideal_variant_first():
    urls = fetch.ligand_urls("btn")
    assert urls[0] == "https://files.rcsb.org/ligands/download/BTN_ideal.sdf"
    assert urls[-1] == "https://files.rcsb.org/ligands/download/BTN.sdf"
    assert len(urls) == 3


def test_an_explicit_variant_is_requested_exactly_and_lower_cased():
    assert fetch.ligand_urls("BTN_ideal") == [
        "https://files.rcsb.org/ligands/download/BTN_ideal.sdf"
    ]
    # The component ID is upper-case on the server, the variant is not.
    assert fetch.ligand_urls("btn_model") == [
        "https://files.rcsb.org/ligands/download/BTN_model.sdf"
    ]


def test_a_bad_variant_is_rejected():
    with pytest.raises(ValueError):
        fetch.ligand_urls("BTN_wrong")
    with pytest.raises(ValueError):
        fetch.ligand_urls("TOOLONG")


def test_the_download_falls_back_when_the_ideal_file_is_missing(tmp_path, monkeypatch):
    seen = []

    def fake_download(url, what, timeout):
        seen.append(url)
        if url.endswith(("_ideal.sdf", "_model.sdf")):
            raise RuntimeError(f"HTTP 404 for {url}")
        return "BEN\n  test\n"

    monkeypatch.setattr(fetch, "_download", fake_download)
    target = tmp_path / "deep" / "BEN.sdf"
    text = fetch.fetch_ligand_sdf("BEN", target)
    assert text.startswith("BEN")
    assert target.read_text(encoding="utf-8") == text
    assert [url.rsplit("/", 1)[-1] for url in seen] == [
        "BEN_ideal.sdf", "BEN_model.sdf", "BEN.sdf"
    ]


def test_the_download_raises_when_every_variant_fails(tmp_path, monkeypatch):
    def always_fail(url, what, timeout):
        raise RuntimeError(f"HTTP 404 for {url}")

    monkeypatch.setattr(fetch, "_download", always_fail)
    with pytest.raises(RuntimeError) as caught:
        fetch.fetch_ligand_sdf("BEN", tmp_path / "BEN.sdf")
    assert "could not be downloaded" in str(caught.value)
    assert not (tmp_path / "BEN.sdf").exists()


def test_an_explicit_variant_does_not_fall_back(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, what, timeout):
        calls.append(url)
        raise RuntimeError("HTTP 404")

    monkeypatch.setattr(fetch, "_download", fake_download)
    with pytest.raises(RuntimeError):
        fetch.fetch_ligand_sdf("BEN_ideal", tmp_path / "BEN.sdf")
    assert len(calls) == 1, calls


def test_the_benchmark_ligands_are_bundled_so_no_run_needs_the_network():
    """The CCD files the benchmark's SMILES came from are in the repository."""
    root = Path(__file__).resolve().parent.parent
    for ligand in ("BTN", "OHT", "XK2"):
        path = root / "tests" / "data" / f"{ligand}.sdf"
        if not path.exists():  # pragma: no cover - optional provenance file
            pytest.skip(f"{path.name} is not bundled")
        assert path.stat().st_size > 100

# SPDX-License-Identifier: GPL-3.0-or-later
"""The AutoGrid/AutoDock/Vina writers, the PDB export and the RCSB fetcher.

The format tests parse the files back the way the target program does -- the
keywords AutoGrid and AutoDock read, the ``key = value`` pairs Vina reads, the
fixed columns of a ``MASTER`` record -- rather than merely checking that some
plausible-looking text came out.  Nothing here touches the network: every
download is mocked at :func:`urllib.request.urlopen`.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import odock
from odock import export, fetch

rdkit = pytest.importorskip("rdkit")

BOX = odock.BoxSpec(
    center=(-1.8555, 14.366, 16.748), size=(17.883, 19.95, 20.514), spacing=0.375
)

DEMO = Path(__file__).resolve().parent.parent / "demo" / "systems" / "3ptb"


@pytest.fixture(scope="session")
def demo_files():
    """The bundled 3PTB demo: receptor, ligand, poses and box."""
    paths = {
        name: DEMO / name
        for name in ("receptor.pdbqt", "ligand.pdbqt", "poses.pdbqt", "box.json")
    }
    if not all(path.exists() for path in paths.values()):
        pytest.skip("the bundled 3PTB demo files are missing")
    return paths


# ---------------------------------------------------------------------------
# Helpers that parse the files back
# ---------------------------------------------------------------------------


def keywords(text: str) -> dict:
    """``keyword -> [values]`` for an AutoDock-style file, comments stripped."""
    out: dict = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        key, _, value = body.partition(" ")
        out.setdefault(key, []).append(value.strip())
    return out


def settings(text: str) -> dict:
    """``key -> value`` for a Vina configuration file."""
    out = {}
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if not body or "=" not in body:
            continue
        key, _, value = body.partition("=")
        out[key.strip()] = value.strip()
    return out


def pdbqt_types_naive(text: str) -> set:
    """The AutoDock types in column 78-79, read the way AutoGrid reads them."""
    return {
        line[77:].strip()
        for line in text.splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[77:].strip()
    }


# ---------------------------------------------------------------------------
# GPF
# ---------------------------------------------------------------------------


def test_gpf_has_every_keyword_autogrid_needs(tmp_path):
    out = tmp_path / "receptor.gpf"
    text = export.export_gpf(
        BOX, "receptor.pdbqt", out, ligand_types=("A", "C", "HD", "N", "NA", "OA")
    )
    assert out.read_text(encoding="utf-8") == text

    parsed = keywords(text)
    for key in (
        "npts", "gridfld", "spacing", "receptor_types", "ligand_types",
        "receptor", "gridcenter", "smooth", "elecmap", "dsolvmap", "dielectric",
    ):
        assert key in parsed, f"the GPF has no {key} line"
    assert parsed["npts"][0].split() == ["47", "53", "55"]
    assert parsed["spacing"] == ["0.375"]
    assert parsed["gridcenter"][0].split() == ["-1.8555", "14.366", "16.748"]
    assert parsed["gridfld"] == ["receptor.maps.fld"]
    assert parsed["receptor"] == ["receptor.pdbqt"]
    assert parsed["elecmap"] == ["receptor.e.map"]
    assert parsed["dsolvmap"] == ["receptor.d.map"]
    assert parsed["dielectric"] == ["-0.1465"]
    # One map per ligand type, named after the receptor.
    assert parsed["map"] == [f"receptor.{t}.map" for t in ("A", "C", "HD", "N", "NA", "OA")]
    assert parsed["ligand_types"][0].split() == ["A", "C", "HD", "N", "NA", "OA"]


def test_gpf_npts_are_odd_and_follow_the_autogrid_rule():
    """AutoGrid rejects an even ``npts``; the rule is 2*floor(size/2s)+1."""
    text = export.export_gpf(BOX, "receptor.pdbqt", None)
    npts = [int(v) for v in keywords(text)["npts"][0].split()]
    expected = [
        2 * int(math.floor(size / (2 * BOX.spacing))) + 1 for size in BOX.size
    ]
    assert npts == expected
    assert all(n % 2 == 1 for n in npts)

    tighter = export.export_gpf(BOX, "receptor.pdbqt", None, spacing=0.5)
    parsed = keywords(tighter)
    assert parsed["spacing"] == ["0.5"]
    assert [int(v) for v in parsed["npts"][0].split()] == [35, 39, 41]


def test_gpf_reads_the_receptor_types_from_the_file(demo_files):
    text = demo_files["receptor.pdbqt"].read_text(encoding="utf-8")
    gpf = export.export_gpf(
        BOX, demo_files["receptor.pdbqt"], None, gridfld="trypsin.maps.fld"
    )
    parsed = keywords(gpf)
    assert set(parsed["receptor_types"][0].split()) == pdbqt_types_naive(text)
    # AutoDockTools copies the receptor types into ligand_types when no ligand
    # is loaded, and names the maps after the .fld stem.
    assert parsed["ligand_types"] == parsed["receptor_types"]
    assert parsed["gridfld"] == ["trypsin.maps.fld"]
    assert parsed["map"] == [f"trypsin.{t}.map" for t in parsed["ligand_types"][0].split()]


def test_gpf_falls_back_to_the_standard_map_list(tmp_path):
    """An unreadable receptor must not stop the file being written."""
    text = export.export_gpf(BOX, tmp_path / "not-there.pdbqt", None)
    parsed = keywords(text)
    assert parsed["ligand_types"][0].split() == list(export.STANDARD_RECEPTOR_TYPES)
    assert len(parsed["map"]) == len(export.STANDARD_RECEPTOR_TYPES)


def test_gpf_rejects_a_bad_spacing():
    with pytest.raises(ValueError, match="spacing"):
        export.export_gpf(BOX, "receptor.pdbqt", None, spacing=0.0)


def test_gpf_accepts_a_box_mapping_and_a_json_file(tmp_path):
    payload = BOX.as_dict()
    mapping = export.export_gpf(payload, "receptor.pdbqt", None)
    path = tmp_path / "box.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    from_json = export.export_gpf(path, "receptor.pdbqt", None)
    assert mapping == from_json


# ---------------------------------------------------------------------------
# DPF
# ---------------------------------------------------------------------------


def test_dpf_has_the_ligand_types_torsions_and_map_references(demo_files, tmp_path):
    ligand = demo_files["ligand.pdbqt"]
    out = tmp_path / "receptor.dpf"
    text = export.export_dpf(BOX, ligand, out, receptor="receptor.pdbqt")
    assert out.read_text(encoding="utf-8") == text

    parsed = keywords(text)
    assert parsed["autodock_parameter_version"] == ["4.2"]
    assert parsed["outlev"] == ["1"]
    assert "intelec" in parsed
    assert parsed["seed"] == ["pid time"]
    assert parsed["fld"] == ["receptor.maps.fld"]
    assert parsed["move"] == [str(ligand)]
    assert parsed["about"][0].split() == ["-1.8555", "14.366", "16.748"]
    assert parsed["tran0"] == ["random"]
    assert parsed["quat0"] == ["random"]

    ligand_text = ligand.read_text(encoding="utf-8")
    assert parsed["ligand_types"][0].split() == sorted(pdbqt_types_naive(ligand_text))
    assert parsed["map"] == [
        f"receptor.{t}.map" for t in parsed["ligand_types"][0].split()
    ]
    assert parsed["elecmap"] == ["receptor.e.map"]
    assert parsed["desolvmap"] == ["receptor.d.map"]

    torsdof = int(re.search(r"TORSDOF\s+(\d+)", ligand_text).group(1))
    assert parsed["ndihe"] == [str(torsdof)]


def test_dpf_writes_the_lamarckian_ga_and_solis_wets_blocks(demo_files):
    parsed = keywords(export.export_dpf(BOX, demo_files["ligand.pdbqt"], None))
    assert "set_ga" in parsed and "set_psw1" in parsed
    assert parsed["ga_pop_size"] == ["150"]
    assert parsed["ga_num_evals"] == ["2500000"]
    assert parsed["ga_num_generations"] == ["27000"]
    assert parsed["ga_elitism"] == ["1"]
    assert parsed["sw_max_its"] == ["300"]
    assert parsed["sw_rho"] == ["1"]
    assert parsed["ls_search_freq"] == ["0.06"]


def test_dpf_parameter_selector_drops_the_sections_it_should(demo_files):
    ligand = demo_files["ligand.pdbqt"]
    local = keywords(export.export_dpf(BOX, ligand, None, parameters="ls", seed=42))
    assert "set_psw1" in local and "set_ga" not in local
    assert local["seed"] == ["42"]

    ga_only = keywords(export.export_dpf(BOX, ligand, None, parameters="ga"))
    assert "set_ga" in ga_only and "set_psw1" not in ga_only

    bare = keywords(export.export_dpf(BOX, ligand, None, parameters="none"))
    assert "set_ga" not in bare and "set_psw1" not in bare

    with pytest.raises(ValueError, match="parameters"):
        export.export_dpf(BOX, ligand, None, parameters="annealing")


def test_dpf_torsdof_and_ndihe_can_be_overridden(demo_files):
    parsed = keywords(
        export.export_dpf(BOX, demo_files["ligand.pdbqt"], None, torsdof=4, ndihe=7)
    )
    assert parsed["ndihe"] == ["7"]


# ---------------------------------------------------------------------------
# Vina configuration
# ---------------------------------------------------------------------------


def test_vina_config_is_a_valid_key_value_file(tmp_path):
    out = tmp_path / "config.txt"
    text = export.export_vina_config(
        BOX,
        "receptor.pdbqt",
        "ligand.pdbqt",
        out,
        out_poses="result.pdbqt",
        exhaustiveness=32,
        num_modes=9,
    )
    assert out.read_text(encoding="utf-8") == text
    parsed = settings(text)
    assert parsed["receptor"] == "receptor.pdbqt"
    assert parsed["ligand"] == "ligand.pdbqt"
    assert parsed["out"] == "result.pdbqt"
    assert float(parsed["center_x"]) == pytest.approx(-1.8555)
    assert float(parsed["center_y"]) == pytest.approx(14.366)
    assert float(parsed["center_z"]) == pytest.approx(16.748)
    assert float(parsed["size_x"]) == pytest.approx(17.883)
    assert float(parsed["size_y"]) == pytest.approx(19.95)
    assert float(parsed["size_z"]) == pytest.approx(20.514)
    assert parsed["exhaustiveness"] == "32"
    assert parsed["num_modes"] == "9"
    assert parsed["energy_range"] == "3"


def test_vina_config_optional_keys_appear_only_when_asked_for():
    bare = settings(export.export_vina_config(BOX, "r.pdbqt", "l.pdbqt", None))
    assert "seed" not in bare and "cpu" not in bare and "scoring" not in bare

    full = settings(
        export.export_vina_config(
            BOX, "r.pdbqt", "l.pdbqt", None, seed=7, cpu=4, scoring="vinardo", min_rmsd=1.5
        )
    )
    assert full["seed"] == "7"
    assert full["cpu"] == "4"
    assert full["scoring"] == "vinardo"
    assert full["min_rmsd"] == "1.5"


def test_vina_config_accepts_documented_options_and_aliases():
    parsed = settings(
        export.export_vina_config(
            BOX, "r.pdbqt", "l.pdbqt", None, num_poses=12, weight_gauss1=-0.035579
        )
    )
    assert parsed["num_modes"] == "12"  # the `num_poses` alias
    assert parsed["weight_gauss1"] == "-0.035579"


def test_vina_config_rejects_a_misspelt_option():
    with pytest.raises(ValueError, match="unknown Vina option"):
        export.export_vina_config(BOX, "r.pdbqt", "l.pdbqt", None, exhaustivness=8)


# ---------------------------------------------------------------------------
# Cleaned PDB
# ---------------------------------------------------------------------------


def _master_of(text: str) -> str:
    masters = [line for line in text.splitlines() if line.startswith("MASTER")]
    assert len(masters) == 1, "an exported PDB must carry exactly one MASTER record"
    return masters[0]


def test_cleaned_pdb_has_a_master_and_end_tail(tmp_path):
    Chem = rdkit.Chem
    AllChem = rdkit.Chem.AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1N"))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol.SetProp("_Name", "aniline")

    out = tmp_path / "aniline.pdb"
    text = export.export_cleaned_pdb(mol, out)
    assert out.read_text(encoding="utf-8") == text

    lines = text.splitlines()
    assert lines[-1] == "END"
    master = _master_of(text)
    assert len(master) == 70
    assert master[:6] == "MASTER"

    n_atom = sum(1 for line in lines if line.startswith("ATOM"))
    n_hetatm = sum(1 for line in lines if line.startswith("HETATM"))
    n_conect = sum(1 for line in lines if line.startswith("CONECT"))
    assert int(master[10:15]) == 0  # numRemark
    assert int(master[15:20]) == 0  # the reserved zero
    assert int(master[20:25]) == 0  # numHet: HET records, of which there are none
    assert int(master[25:30]) == 0  # numHelix
    assert int(master[30:35]) == 0  # numSheet
    assert int(master[35:40]) == 0  # numTurn
    assert int(master[40:45]) == 0  # numSite
    assert int(master[45:50]) == 0  # numXform
    assert int(master[50:55]) == n_atom + n_hetatm  # numCoord
    assert int(master[55:60]) == 0  # numTer
    assert int(master[60:65]) == n_conect  # numConect
    assert int(master[65:70]) == 0  # numSeq

    again = Chem.MolFromPDBFile(str(out), removeHs=False)
    assert again is not None
    assert again.GetNumAtoms() == mol.GetNumAtoms()


def test_cleaned_pdb_accepts_a_structure_file(tmp_path, demo_files):
    out = tmp_path / "clean.pdb"
    text = export.export_cleaned_pdb(demo_files["ligand.pdbqt"], out)
    assert out.exists()
    assert text.rstrip().endswith("END")
    assert "MASTER" in text


def test_master_record_reproduces_the_rcsb_reference(data_dir):
    """The published MASTER layout, checked against a real RCSB file.

    ``tests/data/3PTB.pdb`` was not modified by this project, so its MASTER
    line is the ground truth for the record's field positions.
    """
    lines = (data_dir / "3PTB.pdb").read_text(encoding="utf-8").splitlines()
    reference = [line.rstrip() for line in lines if line.startswith("MASTER")]
    assert reference, "the fixture has no MASTER record"
    assert export.master_record(lines) == reference[0]


def test_cleaned_pdb_rejects_an_empty_molecule():
    with pytest.raises(ValueError):
        export.export_cleaned_pdb(None)


# ---------------------------------------------------------------------------
# Fetching from the RCSB
# ---------------------------------------------------------------------------

PDB_BODY = (
    b"HEADER    HYDROLASE                               01-JAN-90   3PTB\n"
    b"ATOM      1  N   ILE A   1      20.271  13.997  31.212  1.00 30.20           N\n"
    b"END\n"
)


class _FakeResponse:
    """The little bit of the ``urlopen`` protocol the fetcher uses."""

    def __init__(self, payload: bytes):
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def mock_urlopen(monkeypatch, *, payload: bytes = PDB_BODY, error: Exception | None = None):
    """Replace ``urllib.request.urlopen`` and record the calls it receives."""
    calls: list = []

    def urlopen(url, timeout=None):
        request = {"url": getattr(url, "full_url", str(url)), "timeout": timeout}
        headers = getattr(url, "headers", None)
        if headers:
            request["user_agent"] = headers.get("User-agent") or headers.get("User-Agent")
        calls.append(request)
        if error is not None:
            raise error
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def test_fetch_pdb_downloads_and_saves_the_entry(monkeypatch, tmp_path):
    calls = mock_urlopen(monkeypatch)
    out = tmp_path / "3ptb.pdb"
    text = fetch.fetch_pdb("3ptb", out)
    assert text.startswith("HEADER")
    assert out.read_text(encoding="utf-8") == text
    assert len(calls) == 1
    assert calls[0]["url"] == "https://files.rcsb.org/download/3PTB.pdb"
    assert calls[0]["timeout"] == 30.0
    assert "OpenDocking" in (calls[0].get("user_agent") or "")


def test_fetch_pdb_defaults_to_the_identifier_filename(monkeypatch, tmp_path):
    mock_urlopen(monkeypatch)
    monkeypatch.chdir(tmp_path)
    fetch.fetch_pdb("1M17")
    assert (tmp_path / "1M17.pdb").exists()


def test_fetch_pdb_honours_the_timeout(monkeypatch, tmp_path):
    calls = mock_urlopen(monkeypatch)
    fetch.fetch_pdb("3PTB", tmp_path / "e.pdb", timeout=2.5)
    assert calls[0]["timeout"] == 2.5


@pytest.mark.parametrize("bad", ["", "AB", "ABCDE", "3PT!", "3 PT", None, 1234])
def test_fetch_pdb_rejects_a_malformed_identifier(monkeypatch, bad):
    calls = mock_urlopen(monkeypatch)
    with pytest.raises(ValueError):
        fetch.fetch_pdb(bad)
    assert calls == [], "no request may be made for a malformed identifier"


def test_fetch_pdb_reports_a_404_cleanly(monkeypatch):
    mock_urlopen(
        monkeypatch,
        error=urllib.error.HTTPError(
            "https://files.rcsb.org/download/9ZZZ.pdb", 404, "Not Found", None, None
        ),
    )
    with pytest.raises(RuntimeError) as excinfo:
        fetch.fetch_pdb("9ZZZ")
    assert "404" in str(excinfo.value)
    assert "9ZZZ" in str(excinfo.value)


def test_fetch_pdb_reports_an_http_error_other_than_404(monkeypatch):
    mock_urlopen(
        monkeypatch,
        error=urllib.error.HTTPError("u", 503, "Service Unavailable", None, None),
    )
    with pytest.raises(RuntimeError, match="503"):
        fetch.fetch_pdb("3PTB")


def test_fetch_pdb_turns_a_network_failure_into_a_runtime_error(monkeypatch):
    mock_urlopen(
        monkeypatch,
        error=urllib.error.URLError(OSError("getaddrinfo failed")),
    )
    with pytest.raises(RuntimeError) as excinfo:
        fetch.fetch_pdb("3PTB")
    assert "could not be downloaded" in str(excinfo.value)
    assert "getaddrinfo failed" in str(excinfo.value)


def test_fetch_pdb_turns_a_timeout_into_a_runtime_error(monkeypatch):
    mock_urlopen(monkeypatch, error=TimeoutError("timed out"))
    with pytest.raises(RuntimeError, match="timed out"):
        fetch.fetch_pdb("3PTB")


def test_fetch_pdb_rejects_an_empty_response(monkeypatch):
    mock_urlopen(monkeypatch, payload=b"\n\n")
    with pytest.raises(RuntimeError, match="empty"):
        fetch.fetch_pdb("3PTB")


def test_fetch_ligand_sdf_uses_the_component_endpoint(monkeypatch, tmp_path):
    calls = mock_urlopen(monkeypatch, payload=b"BEN\n     RDKit          3D\n")
    out = tmp_path / "BEN.sdf"
    text = fetch.fetch_ligand_sdf("ben", out)
    assert text.startswith("BEN")
    assert out.read_text(encoding="utf-8") == text
    # The ideal coordinates are what the server serves; the bare ``BEN.sdf``
    # path is kept as a fallback and answers 404 on today's RCSB.
    assert calls[0]["url"] == "https://files.rcsb.org/ligands/download/BEN_ideal.sdf"


def test_fetch_ligand_sdf_falls_back_to_the_bare_endpoint(monkeypatch, tmp_path):
    """A 404 on the ideal file must not lose the component."""
    import urllib.error

    calls: list = []
    payload = b"BEN\n     RDKit          3D\n"

    def urlopen(url, timeout=None):
        full = getattr(url, "full_url", str(url))
        calls.append(full)
        if full.endswith(("_ideal.sdf", "_model.sdf")):
            raise urllib.error.HTTPError(full, 404, "Not Found", {}, None)
        return _FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    text = fetch.fetch_ligand_sdf("BEN", tmp_path / "BEN.sdf")
    assert text.startswith("BEN")
    assert calls == [
        "https://files.rcsb.org/ligands/download/BEN_ideal.sdf",
        "https://files.rcsb.org/ligands/download/BEN_model.sdf",
        "https://files.rcsb.org/ligands/download/BEN.sdf",
    ]


def test_fetch_ligand_sdf_defaults_to_the_identifier_filename(monkeypatch, tmp_path):
    mock_urlopen(monkeypatch, payload=b"ATP\n")
    monkeypatch.chdir(tmp_path)
    fetch.fetch_ligand_sdf("atp_model")
    assert (tmp_path / "ATP_MODEL.sdf").exists()


@pytest.mark.parametrize("bad", ["", "ABCD", "B3N!", "ATP_weird"])
def test_fetch_ligand_sdf_validates_the_identifier(monkeypatch, bad):
    calls = mock_urlopen(monkeypatch)
    with pytest.raises(ValueError):
        fetch.fetch_ligand_sdf(bad)
    assert calls == []


def test_fetch_rejects_a_bad_timeout(monkeypatch):
    mock_urlopen(monkeypatch)
    with pytest.raises(ValueError, match="timeout"):
        fetch.fetch_pdb("3PTB", None, timeout=0)


# ---------------------------------------------------------------------------
# The CLI counterparts
# ---------------------------------------------------------------------------


def run_cli(argv, capsys):
    from odock.cli import main

    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_fetch_saves_the_entry(monkeypatch, tmp_path, capsys):
    mock_urlopen(monkeypatch)
    out = tmp_path / "entry.pdb"
    code, stdout, stderr = run_cli(["fetch", "3ptb", "-o", str(out)], capsys)
    assert code == 0
    assert out.exists()
    assert "wrote" in stderr


def test_cli_fetch_reports_a_failure_without_a_traceback(monkeypatch, capsys):
    mock_urlopen(monkeypatch, error=urllib.error.URLError(OSError("offline")))
    with pytest.raises(SystemExit) as excinfo:
        run_cli(["fetch", "3PTB"], capsys)
    message = str(excinfo.value)
    assert "error:" in message and "offline" in message


def test_cli_fetch_ligand_writes_an_sdf(monkeypatch, tmp_path, capsys):
    calls = mock_urlopen(monkeypatch, payload=b"BEN\n")
    out = tmp_path / "ben.sdf"
    code, _, _ = run_cli(["fetch", "BEN", "--ligand", "-o", str(out)], capsys)
    assert code == 0
    assert calls[0]["url"].endswith("/ligands/download/BEN_ideal.sdf")
    assert out.read_text(encoding="utf-8") == "BEN\n"


def test_cli_export_gpf_dpf_and_config(tmp_path, capsys, demo_files):
    box = tmp_path / "box.json"
    box.write_text(json.dumps(BOX.as_dict()), encoding="utf-8")

    gpf = tmp_path / "receptor.gpf"
    code, _, _ = run_cli(
        [
            "export", "gpf",
            "-r", str(demo_files["receptor.pdbqt"]),
            "-o", str(gpf),
            "--box", str(box),
            "--ligand-types", "A", "C", "HD", "N", "NA", "OA",
        ],
        capsys,
    )
    assert code == 0
    parsed = keywords(gpf.read_text(encoding="utf-8"))
    assert parsed["spacing"] == ["0.375"]
    assert len(parsed["map"]) == 6
    assert parsed["gridcenter"][0].split() == ["-1.8555", "14.366", "16.748"]

    dpf = tmp_path / "receptor.dpf"
    code, _, _ = run_cli(
        [
            "export", "dpf",
            "-r", str(demo_files["receptor.pdbqt"]),
            "-l", str(demo_files["ligand.pdbqt"]),
            "-o", str(dpf),
            "--box", str(box),
            "--parameters", "ls",
        ],
        capsys,
    )
    assert code == 0
    parsed = keywords(dpf.read_text(encoding="utf-8"))
    assert "set_psw1" in parsed and "set_ga" not in parsed
    assert parsed["ndihe"] == ["1"]

    config = tmp_path / "vina.txt"
    code, _, _ = run_cli(
        [
            "export", "config",
            "-r", str(demo_files["receptor.pdbqt"]),
            "-l", str(demo_files["ligand.pdbqt"]),
            "-o", str(config),
            "--box", str(box),
            "-e", "24",
        ],
        capsys,
    )
    assert code == 0
    parsed = settings(config.read_text(encoding="utf-8"))
    assert parsed["exhaustiveness"] == "24"
    assert parsed["size_z"] == "20.514"


def test_cli_export_needs_a_box(tmp_path, capsys, demo_files):
    with pytest.raises(SystemExit, match="no search box"):
        run_cli(
            ["export", "gpf", "-r", str(demo_files["receptor.pdbqt"]),
             "-o", str(tmp_path / "x.gpf")],
            capsys,
        )


def test_cli_export_pdb_writes_a_master(tmp_path, capsys, demo_files):
    out = tmp_path / "clean.pdb"
    code, _, _ = run_cli(
        ["export", "pdb", "-i", str(demo_files["ligand.pdbqt"]), "-o", str(out)], capsys
    )
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "MASTER" in text and text.rstrip().endswith("END")


def test_cli_export_prints_to_stdout_without_out(demo_files, capsys):
    code, stdout, _ = run_cli(
        [
            "export", "config",
            "-r", "receptor.pdbqt",
            "-l", "ligand.pdbqt",
            "--center", "-1.8555", "14.366", "16.748",
            "--size", "17.883", "19.95", "20.514",
        ],
        capsys,
    )
    assert code == 0
    assert settings(stdout)["receptor"] == "receptor.pdbqt"

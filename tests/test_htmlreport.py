# SPDX-License-Identifier: GPL-3.0-or-later
"""The HTML report: self-contained, machine-readable, and honest.

Three properties are checked here, and each of them is the kind of thing that
looks obviously true until it is tested:

* **self-contained** -- the document references nothing on the network, and every
  base64 image in it decodes to a complete PNG.  The check is deliberately a raw
  string search for ``http://``/``https://`` as well: the inline SVG's namespace
  is stripped when it is embedded, precisely so that this assertion can be total.
* **the numbers are there** -- the ranking table has one row per pose with the
  affinities the run reported, and the JSON sibling carries the same numbers for a
  machine reader;
* **the caveats are there** -- the "what this run does not establish" section
  exists, and so does the note explaining anything the report could not draw.

The PNG validation is written with the standard library (signature, ``IHDR``,
and a real ``zlib`` inflate of the pixel data) so the suite does not grow a
Pillow dependency for it.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import struct
import zlib
from html.parser import HTMLParser
from pathlib import Path

import pytest

from odock import htmlreport, project

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "3ptb"


def _demo_ready() -> bool:
    return all((DEMO / name).exists() for name in ("receptor.pdbqt", "ligand.pdbqt",
                                                   "poses.pdbqt", "box.json"))


@pytest.fixture(scope="session")
def report_files(tmp_path_factory):
    """One report of the bundled 3PTB demo, built once for the module."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    out = tmp_path_factory.mktemp("htmlreport")
    return htmlreport.write_html_report(
        out / "3ptb.html",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        title="3PTB benzamidine re-docking",
        seed=42,
        scoring="vina",
        command="odock report-html demo/3ptb/poses.pdbqt -o 3ptb.html",
    )


@pytest.fixture(scope="session")
def report_html(report_files) -> str:
    return report_files.path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def report_payload(report_files) -> dict:
    return json.loads(report_files.json_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def demo_project(tmp_path_factory):
    """A project of the same demo, for the project -> report path."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    out = tmp_path_factory.mktemp("htmlreport-project")
    return project.save_project(
        out / "3ptb",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        original_inputs={"receptor": ROOT / "tests" / "data" / "3PTB.pdb"},
        seed=42,
        engine={"exhaustiveness": 32, "search": "monte_carlo"},
    )


# ---------------------------------------------------------------------------
# Helpers: a real PNG decode and a real table parse, standard library only
# ---------------------------------------------------------------------------


def _decode_png(raw: bytes) -> tuple:
    """Validate a PNG completely enough to prove it is not a truncated blob."""
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width, height = struct.unpack(">II", raw[16:24])
    assert width > 0 and height > 0
    position = 8
    header_seen = False
    idat = bytearray()
    while position < len(raw):
        (length,) = struct.unpack(">I", raw[position:position + 4])
        chunk_type = raw[position + 4:position + 8]
        payload = raw[position + 8:position + 8 + length]
        if chunk_type == b"IHDR":
            header_seen = True
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        elif chunk_type == b"IEND":
            break
        position += 12 + length
    assert header_seen, "the PNG has no IHDR chunk"
    assert idat, "the PNG has no IDAT chunk"
    pixels = zlib.decompress(bytes(idat))
    # RGBA (Qt's ARGB32 output) is 4 bytes per pixel plus one filter byte a row.
    assert len(pixels) == height * (1 + width * 4), (
        f"the pixel data does not decode to a {width}x{height} RGBA image"
    )
    return width, height


class _TableParser(HTMLParser):
    """Collect the header and body rows of one ``<table id="...">``."""

    def __init__(self, table_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.table_id = table_id
        self.headers: list = []
        self.rows: list = []
        self._inside = False
        self._row: list = []
        self._cell: list = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == self.table_id:
            self._inside = True
        elif self._inside and tag == "tr":
            self._row = []
        elif self._inside and tag in ("th", "td"):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag):
        if not self._inside:
            return
        if tag == "table":
            self._inside = False
        elif tag in ("th", "td") and self._in_cell:
            text = "".join(self._cell).strip()
            if tag == "th":
                self.headers.append(text)
            else:
                self._row.append(text)
            self._in_cell = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def _table(document: str, table_id: str) -> _TableParser:
    parser = _TableParser(table_id)
    parser.feed(document)
    return parser


def _images(document: str) -> list:
    return re.findall(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"', document)


# ---------------------------------------------------------------------------
# Self-containment
# ---------------------------------------------------------------------------


def test_the_report_references_nothing_on_the_network(report_html):
    assert "http://" not in report_html
    assert "https://" not in report_html
    assert htmlreport.find_external_references(report_html) == []
    assert "<link" not in report_html.lower()
    assert not re.search(r"<script[^>]*\bsrc\s*=", report_html, re.IGNORECASE)
    # The style is inline, so the document renders the same offline.
    assert "<style>" in report_html


def test_every_embedded_image_decodes(report_html):
    encoded = _images(report_html)
    assert encoded, "the report embedded no raster figure at all"
    for payload in encoded:
        raw = base64.b64decode(payload, validate=True)
        width, height = _decode_png(raw)
        assert (width, height) == (900, 700)


def test_the_embedded_images_are_the_figures_the_report_describes(report_files):
    assert report_files.n_figures >= 2
    assert report_files.n_images == len(_images(report_files.path.read_text(encoding="utf-8")))
    # The vector form is in the document too, so a reader without the raster
    # still sees the figure.
    assert "<svg" in report_files.path.read_text(encoding="utf-8")


def test_a_raster_preview_is_present_or_explained(report_files, report_html):
    """Either there is a raster figure, or the report says why there is not."""
    if report_files.n_images:
        assert "no raster preview was produced" not in report_html
    else:  # pragma: no cover - depends on how PyQt6/Qt is installed
        assert "no raster preview" in report_html or "raster preview was skipped" in report_html


def test_an_injected_raster_figure_is_embedded_and_decodes(tmp_path):
    """A caller can attach a figure (a screenshot), and it survives intact."""
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/"
        "q842iQAAAABJRU5ErkJggg=="
    )
    built = htmlreport.build_report(
        {"seed": 1, "box": {"center": [0, 0, 0], "size": [20, 20, 20], "spacing": 0.375},
         "poses": [{"index": 0, "affinity": -5.0, "rmsd_lb": 0.0, "rmsd_ub": 0.0,
                    "in_box": True, "num_atoms": 3}]},
        title="injected",
        extra_figures={"pose-preview": raw},
        rasterise=False,
    )
    encoded = _images(built.html)
    assert encoded and base64.b64decode(encoded[0]) == raw
    assert "figure-pose-preview" in built.html
    assert htmlreport.find_external_references(built.html) == []
    assert built.n_images == 1


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


def test_the_ranking_table_has_one_row_per_pose(report_html, report_payload):
    table = _table(report_html, "ranking")
    assert table.headers[0] == "mode"
    assert "affinity (kcal/mol)" in table.headers
    assert len(table.rows) == len(report_payload["ranking"]) > 1
    for index, row in enumerate(table.rows):
        assert int(row[0]) == index + 1
        assert row[1] == f"{report_payload['ranking'][index]['affinity']:.3f}"


def test_the_report_shows_the_inputs_with_their_hashes(report_html, report_payload):
    import hashlib

    table = _table(report_html, "inputs")
    assert table.headers == ["role", "file", "bytes", "SHA-256"]
    roles = {row[0] for row in table.rows}
    assert roles == {"receptor", "ligand"}
    by_role = {row[0]: row for row in table.rows}
    for role, filename in (("receptor", "receptor.pdbqt"), ("ligand", "ligand.pdbqt")):
        digest = hashlib.sha256((DEMO / filename).read_bytes()).hexdigest()
        assert by_role[role][1] == filename
        assert by_role[role][3] == digest[:32] + "…"
        assert by_role[role][2] == str((DEMO / filename).stat().st_size)
    assert set(report_payload["reproducibility"]["input_sha256"]) == {"receptor", "ligand"}


def test_the_engine_settings_and_the_seed_are_in_the_document(report_html, report_payload):
    assert report_payload["engine"]["seed"] == 42
    assert report_payload["engine"]["scoring"] == "vina"
    assert report_payload["engine"]["num_poses"] == len(report_payload["ranking"])
    # A setting nobody recorded is reported as such, never invented.  The pose
    # file carries no exhaustiveness, so the caller had to supply it -- and here
    # did not.
    assert report_payload["engine"].get("exhaustiveness") is None
    assert "not recorded" in report_html
    assert "odock-engine" in report_html


def test_the_engine_settings_supplied_by_the_caller_are_reported(demo_project, tmp_path):
    files = htmlreport.write_html_report(
        tmp_path / "engine.html", project=demo_project, title="engine settings"
    )
    payload = json.loads(files.json_path.read_text(encoding="utf-8"))
    assert payload["engine"]["exhaustiveness"] == 32
    assert payload["engine"]["search"] == "monte_carlo"
    assert payload["reproducibility"]["seed"] == 42


def test_the_box_is_reported(report_html, report_payload):
    assert report_payload["box"]["spacing"] == pytest.approx(0.375)
    assert "17.883" in report_html and "14.366" in report_html


def test_the_interaction_profile_and_its_diagram_are_present(report_html, report_payload):
    profile = report_payload["interaction_profile"]
    assert profile["key_residues"], "no key residues were reported for the demo run"
    assert profile["counts"].get("hbond")
    assert "figure-interaction-diagram" in report_html
    table = _table(report_html, "interactions")
    assert table.headers[0:4] == ["kind", "receptor atom", "ligand atom", "d (Å)"]
    assert len(table.rows) == len(profile["rows"]) > 0
    # The residue labels are readable, not bare indices.
    assert re.search(r"[A-Z]{3}\d+:", table.rows[0][1] + table.rows[0][2])


def test_the_pose_quality_metrics_are_reported(report_html, report_payload):
    quality = report_payload["pose_quality"]
    assert quality["n_poses"] == len(report_payload["ranking"])
    assert quality["best_affinity"] == pytest.approx(-6.210, abs=0.01)
    assert quality["spread"] is not None
    assert "Pose-quality metrics" in report_html
    assert re.search(r"poses within 1 kcal/mol of the best", report_html)


def test_the_pharmacophore_summary_is_reported(report_html, report_payload):
    pharmacophore = report_payload["pharmacophore"]
    assert pharmacophore.get("keys"), "the demo run has recurring contacts"
    table = _table(report_html, "pharmacophore")
    assert table.headers == ["feature", "kind", "poses", "frequency"]
    assert len(table.rows) == len(pharmacophore["keys"])
    assert any(row[1] in ("hbond", "hydrophobic") for row in table.rows)


# ---------------------------------------------------------------------------
# The caveats
# ---------------------------------------------------------------------------


def test_the_limits_section_is_present_and_specific(report_html, report_payload):
    assert 'id="limits"' in report_html
    assert "What this run does not establish" in report_html
    limits = report_payload["does_not_establish"]
    assert any("captures a *run*, not an interactive session" in item for item in limits)
    assert any("do not prove that a file is correct" in item for item in limits)
    assert any("compatibility contract" in item for item in limits)
    for item in limits:
        assert item.split(".")[0][:40] in report_html


def test_the_report_says_what_it_could_not_draw(tmp_path):
    """A run without a receptor cannot profile interactions, and says so."""
    payload = {
        "seed": 3,
        "poses": [
            {"index": 0, "affinity": -4.5, "rmsd_lb": 0.0, "rmsd_ub": 0.0,
             "in_box": True, "num_atoms": 3},
            {"index": 1, "affinity": -4.0, "rmsd_lb": 1.0, "rmsd_ub": 2.0,
             "in_box": True, "num_atoms": 3},
        ],
    }
    built = htmlreport.build_report(payload, title="no receptor", rasterise=False)
    assert "No interaction profile is available" in built.html
    assert any("coordinates" in note for note in built.notes)
    assert not built.payload["interaction_profile"]
    assert built.payload["ranking"][0]["affinity"] == pytest.approx(-4.5)


# ---------------------------------------------------------------------------
# The siblings and the project path
# ---------------------------------------------------------------------------


def test_the_json_sibling_matches_the_html(report_files, report_payload, report_html):
    assert report_payload["format"] == htmlreport.REPORT_FORMAT
    assert report_payload["version"] == htmlreport.REPORT_VERSION
    assert report_payload["title"] == "3PTB benzamidine re-docking"
    assert report_payload["reproducibility"]["seed"] == 42
    assert report_payload["reproducibility"]["command"].startswith("odock report-html")
    assert report_payload["reproducibility"]["input_sha256"]
    assert report_files.json_path.exists() and report_files.text_path.exists()

    text = report_files.text_path.read_text(encoding="utf-8")
    assert "What this run does not establish" in text
    assert "-6.210" in text
    assert "odock report-html" in text
    table = _table(report_html, "ranking")
    for row in table.rows:
        assert row[1] in text


def test_a_report_can_be_built_from_a_project(demo_project, tmp_path):
    files = htmlreport.write_html_report(
        tmp_path / "from-project.html", project=demo_project, title="project report"
    )
    payload = json.loads(files.json_path.read_text(encoding="utf-8"))
    assert payload["project"]["name"] == demo_project.path.name
    assert payload["project"]["schema_version"] == project.SCHEMA_VERSION
    assert payload["reproducibility"]["project_sha256"]
    assert len(payload["ranking"]) == demo_project.run["n_poses"]
    assert payload["inputs"], "the project's inputs and hashes did not reach the report"
    assert payload["box"]["spacing"] == pytest.approx(0.375)
    assert files.n_images >= 1
    assert htmlreport.find_external_references(files.path.read_text(encoding="utf-8")) == []

    # A project that has been altered is not reported as if nothing happened: the
    # report from a *verified* project still carries the limits section.
    assert payload["does_not_establish"]


def test_the_report_from_a_project_keeps_a_stored_figure(tmp_path):
    """A figure stored in the project is embedded again, byte for byte."""
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/"
        "q842iQAAAABJRU5ErkJggg=="
    )
    saved = project.save_project(
        tmp_path / "with-figure",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        figures={"workbench.png": png},
    )
    files = htmlreport.write_html_report(tmp_path / "fig.html", project=saved)
    assert png in [
        base64.b64decode(item) for item in _images(files.path.read_text(encoding="utf-8"))
    ]


def test_the_rdkit_depiction_is_embedded_when_a_molecule_is_given(tmp_path):
    Chem = pytest.importorskip("rdkit.Chem")
    mol = Chem.MolFromSmiles("NC(=N)c1ccccc1") or Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    built = htmlreport.build_report(
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        ligand_mol=mol,
        title="with depiction",
    )
    assert 'id="figure-ligand"' in built.html
    assert htmlreport.find_external_references(built.html) == []


# ---------------------------------------------------------------------------
# Path normalisation and escaping
# ---------------------------------------------------------------------------


def test_no_absolute_path_reaches_the_report(tmp_path, demo_project):
    """The published report must not name a directory from this machine."""
    files = htmlreport.write_html_report(
        tmp_path / "leaky.html",
        project=demo_project,
        command=(
            r"C:\Users\33654\Desktop\Python\OpenDocking\.venv\Scripts\python.exe "
            r"-m odock.cli report-html C:\Users\33654\Desktop\Python\OpenDocking\demo\3ptb\poses.pdbqt"
        ),
    )
    for path in (files.path, files.json_path, files.text_path):
        text = path.read_text(encoding="utf-8")
        assert project.find_absolute_paths(text) == [], f"a path leaked into {path.name}"
        assert "C:\\Users" not in text
        assert ".venv" not in text


def test_a_hostile_title_is_escaped_and_does_not_reach_the_network(tmp_path):
    files = htmlreport.write_html_report(
        tmp_path / "hostile.html",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json",
        title='<script>fetch("https://evil.example/x")</script> & "quotes"',
        rasterise=False,
    )
    document = files.path.read_text(encoding="utf-8")
    assert "<script>fetch" not in document
    assert "&lt;script&gt;" in document
    assert "&amp;" in document
    # The URL survives only as escaped *text*; nothing in the document fetches it.
    assert htmlreport.find_external_references(document) == []
    assert not re.search(r"(?:src|href)\s*=\s*[\"']?https?://", document)


def test_a_report_of_a_run_never_contains_a_url_at_all(report_html):
    """Stronger than the reference check, for a document built from run data."""
    assert "http" not in report_html


def test_the_report_is_one_file_with_no_side_car_needed(report_files, tmp_path):
    """Copying just the .html elsewhere keeps it readable."""
    import shutil

    alone = tmp_path / "alone.html"
    shutil.copy2(report_files.path, alone)
    document = alone.read_text(encoding="utf-8")
    assert _images(document), "the figures did not travel with the file"
    assert document.startswith("<!DOCTYPE html>")
    assert document.rstrip().endswith("</html>")


# ---------------------------------------------------------------------------
# The PDF: optional, discovered, never required
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_pdf_is_produced_only_when_a_converter_works(demo_project, tmp_path):
    """Either a real PDF, or a clear message -- never a silent half-document."""
    converter = htmlreport.pdf_converter()
    if converter is None:
        with pytest.raises(htmlreport.PdfUnavailable, match="HTML only"):
            htmlreport.write_pdf("<html><body>x</body></html>", tmp_path / "x.pdf")
        return
    try:
        written = htmlreport.write_pdf(
            "<html><body><h1>hello</h1></body></html>", tmp_path / "probe.pdf"
        )
    except htmlreport.PdfUnavailable as exc:
        assert "HTML" in str(exc)
        return
    assert written.exists()
    assert written.read_bytes()[:5] == b"%PDF-"


def test_the_pdf_flag_reports_itself(demo_project, tmp_path):
    """`pdf=True` never fails the report; it reports what happened."""
    files = htmlreport.write_html_report(
        tmp_path / "pdf-flag.html", project=demo_project, pdf=True, rasterise=False
    )
    assert files.path.exists()
    if files.pdf_path is None:  # pragma: no cover - depends on the converter
        assert any("PDF" in note for note in files.notes)
    else:
        assert files.pdf_path.read_bytes()[:5] == b"%PDF-"


def test_a_failed_converter_leaves_the_html_intact(tmp_path, monkeypatch):
    """A broken converter must not damage the deliverable."""
    monkeypatch.setenv("ODOCK_PDF_CONVERTER", str(tmp_path / "not-a-browser.exe"))
    files = htmlreport.write_html_report(
        tmp_path / "broken-pdf.html",
        DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt",
        box=DEMO / "box.json",
        pdf=True,
        rasterise=False,
    )
    assert files.path.exists() and files.path.stat().st_size > 1000
    assert files.pdf_path is None
    assert any("PDF" in note for note in files.notes)


# ---------------------------------------------------------------------------
# Comparing runs: one page over several projects
# ---------------------------------------------------------------------------


def _two_runs(tmp_path, *, second_seed=7):
    """Two projects of the demo pose file that differ only in the recorded seed."""
    first = project.save_project(
        tmp_path / "run-a", DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt", ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json", seed=42,
        engine={"seed": 42, "scoring": "vina", "exhaustiveness": 32},
        title="3PTB seed 42",
    )
    second = project.save_project(
        tmp_path / "run-b", DEMO / "poses.pdbqt",
        receptor=DEMO / "receptor.pdbqt", ligand=DEMO / "ligand.pdbqt",
        box=DEMO / "box.json", seed=second_seed,
        engine={"seed": second_seed, "scoring": "vina", "exhaustiveness": 32},
        title="3PTB other seed",
    )
    return first, second


def test_the_comparison_page_names_every_difference(tmp_path):
    first, second = _two_runs(tmp_path)
    comparison = project.compare_projects([first.path, second.path])
    files = htmlreport.write_comparison_report(
        tmp_path / "comparison.html", comparison, title="42 vs 7"
    )
    document = files.path.read_text(encoding="utf-8")
    assert htmlreport.find_external_references(document) == []
    assert "http://" not in document and "https://" not in document
    assert project.find_absolute_paths(document) == []

    for section in ("runs", "settings", "inputs", "deltas", "pairwise", "clusters", "limits"):
        assert f'id="{section}"' in document, section
    assert "What this comparison does not establish" in document

    settings = _table(document, "settings")
    fields = {row[0] for row in settings.rows}
    assert "seed" in fields
    seed_row = next(row for row in settings.rows if row[0] == "seed")
    assert seed_row[1:] == ["42", "7"]

    runs = _table(document, "runs")
    assert runs.headers[0] == "run"
    assert len(runs.rows) == 2
    assert {row[0] for row in runs.rows} == {first.path.name, second.path.name}

    deltas = _table(document, "deltas")
    assert deltas.headers[0] == "mode"
    assert len(deltas.rows) == len(comparison["runs"][0]["affinities"])

    pairwise = _table(document, "pairwise")
    assert pairwise.headers[2] == "Δposes"
    assert len(pairwise.rows) == 1

    # The machine-readable siblings carry the same comparison.
    payload = json.loads(files.json_path.read_text(encoding="utf-8"))
    assert payload["format"] == "odock-comparison"
    assert payload["pairwise"][0]["settings_differences"][0]["field"] == "seed"
    text = files.text_path.read_text(encoding="utf-8")
    assert "setting seed: 42 -> 7" in text
    assert "What this comparison does not establish" in text


def test_a_run_compared_with_itself_reports_no_difference(tmp_path):
    first, _ = _two_runs(tmp_path)
    copy = tmp_path / "copy.odockproj"
    copy.write_bytes(first.path.read_bytes())
    comparison = project.compare_projects([first.path, copy])
    files = htmlreport.write_comparison_report(tmp_path / "same.html", comparison)
    document = files.path.read_text(encoding="utf-8")
    # Identical runs: no settings differ, and the page says so explicitly rather
    # than leaving an empty table.
    assert "recorded identical engine settings" in document
    pair = comparison["pairwise"][0]
    assert pair["settings_differences"] == []
    assert pair["max_affinity_delta"] == 0.0
    assert pair["top_pose_rmsd"] == pytest.approx(0.0, abs=1e-9)
    assert pair["pose_count_delta"] == 0
    assert pair["input_differences"] == []


def test_the_comparison_rejects_an_empty_payload(tmp_path):
    with pytest.raises(ValueError, match="holds no run"):
        htmlreport.build_comparison_report({})
    with pytest.raises(ValueError, match="comparison must be"):
        htmlreport.build_comparison_report(3.5)


# ---------------------------------------------------------------------------
# A campaign index: one page over a directory of projects
# ---------------------------------------------------------------------------


def test_the_campaign_index_links_every_project(tmp_path):
    first, second = _two_runs(tmp_path)
    projects_dir = tmp_path / "campaign"
    projects_dir.mkdir()
    shutil.copy2(first.path, projects_dir / first.path.name)
    shutil.copy2(second.path, projects_dir / second.path.name)
    # A report next to a member is what the index links to.
    htmlreport.write_html_report(
        projects_dir / "run-a.html", project=project.open_project(projects_dir / "run-a.odockproj"),
        title="3PTB seed 42", rasterise=False,
    )
    payload = project.campaign_index(projects_dir)
    files = htmlreport.write_campaign_index(
        projects_dir / "index.html", payload, title="3PTB series"
    )
    document = files.path.read_text(encoding="utf-8")
    assert htmlreport.find_external_references(document) == []
    assert project.find_absolute_paths(document) == []
    assert "What this index does not establish" in document
    table = _table(document, "projects")
    assert table.headers[0] == "project"
    assert len(table.rows) == 2
    # Every link resolves to a file that exists next to the page.
    hrefs = re.findall(r'<a href="([^"]+)"', document)
    assert len(hrefs) == 2
    for href in hrefs:
        assert (files.path.parent / href).exists(), href
    assert any(href.endswith(".html") for href in hrefs)
    assert any(href.endswith(".odockproj") for href in hrefs)
    payload_json = json.loads(files.json_path.read_text(encoding="utf-8"))
    assert payload_json["n_projects"] == 2
    assert payload_json["n_verified"] == 2
    assert files.text_path.read_text(encoding="utf-8").startswith("3PTB series")


def test_the_index_rejects_an_empty_directory(tmp_path):
    with pytest.raises(project.ProjectError, match="no such directory"):
        project.campaign_index(tmp_path / "empty")
    empty = tmp_path / "exists-but-empty"
    empty.mkdir()
    with pytest.raises(project.ProjectError, match="no project matching"):
        project.campaign_index(empty)
    with pytest.raises(ValueError, match="holds no project"):
        htmlreport.build_campaign_index({"entries": []})

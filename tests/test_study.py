# SPDX-License-Identifier: GPL-3.0-or-later
"""Studies: a verifiable collection, and a diff that names what changed.

The interesting claims, each with the test that would catch it:

* **a study verifies its members' hashes** -- change one byte of one member copy
  and ``verify`` has to name that member (and the ``SHA256SUMS`` disagreement);
* **the diff is arithmetic, not prose** -- the protocol fields are named with both
  values, the members added/removed are listed, and the hit-list movement carries
  ranks and rank deltas that a hand-checked example can pin exactly;
* **the pages are self-contained** -- no ``http``/``https`` reference, no absolute
  path, and every member link resolves to a file that exists.

Most of the numbers here come from *fabricated* projects whose affinities the test
chooses, because that is the only way to pin a rank delta exactly; one test then
runs the same code over a real docked project so the fabricated case cannot drift
away from reality unnoticed.
"""

from __future__ import annotations

import json
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

import pytest

from odock import htmlreport, project, study

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "3ptb"


def _demo_ready() -> bool:
    return all((DEMO / name).exists() for name in ("receptor.pdbqt", "ligand.pdbqt",
                                                   "poses.pdbqt", "box.json"))


def _fabricated(tmp_path, name: str, affinities, *, seed: int = 1,
                engine=None, title=None):
    """A project with the affinities the test chooses (nothing is docked)."""
    import numpy as np

    from odock.docking import DockResult, Pose
    from odock.prepare import BoxSpec

    poses = [
        Pose(
            index=index,
            affinity=float(affinity),
            rmsd_lower_bound=float(index),
            rmsd_upper_bound=float(index) + 0.5,
            in_box=True,
            num_atoms=3,
            coords=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
            ) + (index + 1) * 0.25,
        )
        for index, affinity in enumerate(affinities)
    ]
    result = DockResult(
        poses=poses,
        seed=seed,
        scoring="vina",
        box=BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=0.375),
        ligand_atom_order=("C1", "C2", "C3"),
    )
    return project.save_project(
        tmp_path / name, result, engine=engine, title=title or name
    )


@pytest.fixture
def two_studies(tmp_path):
    """Two studies with the same molecules, different pools and protocols.

    Affinities (kcal/mol), chosen so every rank delta is checkable by hand::

        study A: alpha -9.0, beta -8.0, gamma -7.0, epsilon -6.0
        study B: alpha -7.5, beta -8.0, delta -9.5, epsilon -5.0

    With N = 2: alpha was rank 1 and becomes rank 3 (leaves the top 2), beta stays
    at rank 2, delta is added at rank 1 (enters the top 2), gamma is removed, and
    epsilon sits outside the top 2 on both sides, so its label is the affinity
    movement ("worsened", +1.0 kcal/mol).
    """
    projects = tmp_path / "projects"
    projects.mkdir()
    alpha_a = _fabricated(projects, "alpha-a", [-9.0, -8.5], seed=42)
    beta = _fabricated(projects, "beta", [-8.0, -7.6], seed=42)
    gamma = _fabricated(projects, "gamma", [-7.0, -6.5], seed=42)
    epsilon = _fabricated(projects, "epsilon", [-6.0, -5.8], seed=42)
    alpha_b = _fabricated(projects, "alpha-b", [-7.5, -7.4], seed=7)
    delta = _fabricated(projects, "delta", [-9.5, -9.0], seed=7)

    first = study.create_study(
        tmp_path / "study-a",
        name="protocol-a",
        title="protocol A",
        target="trypsin",
        library="library.smi",
        protocol={"scoring": "vina", "exhaustiveness": 8},
        operator="alice",
        notes=["first pass"],
        members=[alpha_a.path, beta.path, gamma.path, epsilon.path],
    )
    # The same molecule must carry the same member name on both sides, which is
    # what makes the diff able to pair them at all.
    _relabel(first, {"alpha-a": "alpha", "beta": "beta", "gamma": "gamma",
                     "epsilon": "epsilon"})
    epsilon_b = _fabricated(projects, "epsilon-b", [-5.0, -4.9], seed=7)
    second = study.create_study(
        tmp_path / "study-b",
        name="protocol-b",
        title="protocol B",
        target="trypsin",
        library="library-v2.smi",
        protocol={"scoring": "vinardo", "exhaustiveness": 32, "seed": 7},
        operator="bob",
        members=[alpha_b.path, beta.path, delta.path, epsilon_b.path],
    )
    _relabel(second, {"alpha-b": "alpha", "beta": "beta", "delta": "delta",
                      "epsilon-b": "epsilon"})
    return study.load_study(tmp_path / "study-a"), study.load_study(tmp_path / "study-b")


def _relabel(loaded: study.Study, mapping: dict) -> None:
    """Rename members in place (the label is what pairs two studies)."""
    for member in loaded.manifest["members"]:
        wanted = mapping.get(str(member.get("name")))
        if wanted:
            member["name"] = wanted
            member["label"] = wanted
    study._write_manifest(loaded)


def _rows(document: str, table_id: str):
    """The header and rows of one ``<table id="...">`` (stdlib HTML parser)."""
    parser = _TableParser(table_id)
    parser.feed(document)
    return parser


class _TableParser(HTMLParser):
    def __init__(self, table_id: str) -> None:
        super().__init__(convert_charrefs=True)
        self.want = table_id
        self.headers: list = []
        self.rows: list = []
        self.inside = False
        self._row: list = []
        self._cell: list = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == self.want:
            self.inside = True
        elif self.inside and tag == "tr":
            self._row = []
        elif self.inside and tag in ("th", "td"):
            self._in_cell, self._cell = True, []

    def handle_endtag(self, tag):
        if not self.inside:
            return
        if tag == "table":
            self.inside = False
        elif tag in ("th", "td") and self._in_cell:
            text = "".join(self._cell).strip()
            (self.headers if tag == "th" else self._row).append(text)
            self._in_cell = False
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = []

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


# ---------------------------------------------------------------------------
# Create, add, verify
# ---------------------------------------------------------------------------


def test_a_study_is_created_with_its_manifest_and_sums(tmp_path):
    saved = _fabricated(tmp_path, "run", [-6.0])
    created = study.create_study(
        tmp_path / "s",
        name="demo-study",
        title="demo study",
        target="trypsin",
        receptors=["receptor.pdbqt"],
        library="library.smi",
        protocol={"scoring": "vina", "exhaustiveness": 8},
        notes=["a note"],
        operator="alice",
        members=[saved.path],
    )
    manifest = json.loads((tmp_path / "s" / study.STUDY_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["format"] == study.STUDY_FORMAT
    assert manifest["schema_version"] == study.SCHEMA_VERSION
    assert manifest["name"] == "demo-study"
    assert manifest["metadata"]["target"] == "trypsin"
    assert manifest["metadata"]["receptors"] == ["receptor.pdbqt"]
    assert manifest["metadata"]["protocol"] == {"scoring": "vina", "exhaustiveness": 8}
    assert manifest["audit"]["operator"] == "alice"
    assert manifest["audit"]["tool_version"]
    assert manifest["audit"]["created_utc"].endswith("Z")
    assert len(created.members) == 1

    member = created.members[0]
    assert member["project"] == f"{study.MEMBERS_DIR}/run.odockproj"
    assert member["label"] == "run"
    assert member["sha256"] == project.sha256_bytes(
        (tmp_path / "s" / member["project"]).read_bytes()
    )
    assert member["best_affinity"] == pytest.approx(-6.0)
    assert member["added_utc"].endswith("Z")
    assert member["verified_at_add"] is True
    sums = (tmp_path / "s" / study.SUMS_NAME).read_text(encoding="utf-8")
    assert study.STUDY_MANIFEST in sums and member["project"] in sums

    report = created.verify()
    assert report.ok, report.summary()
    assert report.ok_members == ["run"]
    assert "does not show that the study is a good experiment" in report.summary()


def test_adding_a_project_keeps_the_study_self_contained(tmp_path):
    saved = _fabricated(tmp_path, "run", [-6.0])
    study.create_study(tmp_path / "s", name="s")
    loaded = study.add_projects(tmp_path / "s", [saved.path], label="the ligand")
    assert [member["name"] for member in loaded.members] == ["the_ligand"]
    assert loaded.members[0]["label"] == "the ligand"
    copied = tmp_path / "s" / loaded.members[0]["project"]
    assert copied.exists()
    # The copy is what the study verifies, so deleting the original is harmless.
    saved.path.unlink()
    assert study.verify_study(tmp_path / "s").ok


def test_a_linked_project_is_recorded_as_a_relative_path(tmp_path):
    saved = _fabricated(tmp_path, "run", [-6.0])
    study.create_study(tmp_path / "s", name="s")
    loaded = study.add_projects(tmp_path / "s", [saved.path], copy=False)
    member = loaded.members[0]
    assert member["copied"] is False
    assert not Path(member["project"]).is_absolute()
    assert (tmp_path / "s" / member["project"]).exists()
    assert loaded.verify().ok
    # Nothing may leak an absolute path into the manifest.
    assert project.find_absolute_paths(
        (tmp_path / "s" / study.STUDY_MANIFEST).read_text(encoding="utf-8")
    ) == []


def test_a_study_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(study.StudyError, match="is not a study"):
        study.load_study(tmp_path / "nope")
    report = study.verify_study(tmp_path / "nope")
    assert not report.ok and "is not a study" in report.problems[0]


def test_a_newer_study_schema_is_refused_with_both_versions(tmp_path):
    study.create_study(tmp_path / "s", name="s")
    manifest_path = tmp_path / "s" / study.STUDY_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = study.SCHEMA_VERSION + 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(study.StudyError) as excinfo:
        study.load_study(tmp_path / "s")
    message = str(excinfo.value)
    assert str(study.SCHEMA_VERSION) in message
    assert str(study.SCHEMA_VERSION + 2) in message
    assert "compatibility contract" in message


# ---------------------------------------------------------------------------
# Verify: a changed member is named
# ---------------------------------------------------------------------------


def test_verify_names_a_changed_member(two_studies, tmp_path):
    first, _ = two_studies
    assert first.verify().ok
    target = first.member_path(first.member("alpha"))
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    target.write_bytes(bytes(payload))

    report = first.verify()
    assert not report.ok
    assert any(item.startswith("alpha (") for item in report.changed), report.changed
    assert any(study.SUMS_NAME in item for item in report.sums_problems)
    assert "alpha" in report.summary()
    assert "FAILED" in report.summary()
    # The CLI reports the same thing and exits non-zero.
    from odock.cli import main

    assert main(["study", "verify", str(first.path)]) == 1


def test_verify_reports_a_missing_member_and_an_unlisted_one(two_studies):
    first, _ = two_studies
    first.member_path(first.member("beta")).unlink()
    (first.path / study.MEMBERS_DIR / "stray.odockproj").write_bytes(b"not a project")
    report = first.verify()
    assert not report.ok
    assert any("beta" in item for item in report.missing)
    assert f"{study.MEMBERS_DIR}/stray.odockproj" in report.unlisted


def test_verify_reports_a_member_whose_project_is_internally_broken(two_studies):
    """A member whose own archive no longer verifies is reported as a member."""
    import zipfile

    first, _ = two_studies
    target = first.member_path(first.member("gamma"))
    with zipfile.ZipFile(target) as archive:
        members = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
    order = [name for name, _ in members]
    data = dict(members)
    data["files/box.json"] = data["files/box.json"][:10] + b"X" + data["files/box.json"][11:]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in order:
            project._write_zip_member(archive, name, data[name])
    report = first.verify()
    assert not report.ok
    # The member hash changed first (the manifest records the file), which is the
    # point: the study notices before it has to interpret the project.
    assert any(item.startswith("gamma (") for item in report.changed)


# ---------------------------------------------------------------------------
# Diff: the numbers
# ---------------------------------------------------------------------------


def test_the_diff_names_the_protocol_fields_that_changed(two_studies):
    first, second = two_studies
    payload = study.study_diff(first.path, second.path, top=2)
    assert payload["format"] == "odock-study-diff"
    differences = {
        item["field"]: item["values"] for item in payload["metadata_differences"]
    }
    assert differences["protocol.exhaustiveness"] == [8, 32]
    assert differences["protocol.scoring"] == ["vina", "vinardo"]
    assert differences["protocol.seed"] == [None, 7]
    assert differences["library"] == ["library.smi", "library-v2.smi"]
    seed = next(
        item for item in payload["metadata_differences"] if item["field"] == "protocol.seed"
    )
    assert seed["note"] == "not recorded in one study"
    assert payload["studies"]["a"]["operator"] == "alice"
    assert payload["studies"]["b"]["operator"] == "bob"
    assert payload["studies"]["a"]["n_members"] == 4
    assert payload["studies"]["b"]["n_members"] == 4


def test_the_diff_names_the_members_added_and_removed(two_studies):
    first, second = two_studies
    payload = study.study_diff(first.path, second.path, top=2)
    members = payload["members"]
    assert [item["name"] for item in members["shared"]] == ["alpha", "beta", "epsilon"]
    assert [item["name"] for item in members["added"]] == ["delta"]
    assert [item["name"] for item in members["removed"]] == ["gamma"]
    added = members["added"][0]
    assert added["best_affinity"] == pytest.approx(-9.5)
    assert added["rank_b"] == 1
    removed = members["removed"][0]
    assert removed["best_affinity"] == pytest.approx(-7.0)
    assert removed["rank_a"] == 3


def test_the_hit_list_movement_carries_the_rank_deltas(two_studies):
    """Hand-checked ranks.

    Study A ranks alpha 1, beta 2, gamma 3, epsilon 4; study B ranks delta 1,
    beta 2, alpha 3, epsilon 4.  With N = 2, alpha leaves the top 2, delta enters
    it, and epsilon (outside on both sides) is judged by its affinity movement.
    """
    first, second = two_studies
    hit = study.study_diff(first.path, second.path, top=2)["hit_list"]
    assert hit["top_n"] == 2
    rows = {item["name"]: item for item in hit["members"]}
    assert set(rows) == {"alpha", "beta", "epsilon"}
    assert rows["alpha"]["rank_a"] == 1
    assert rows["alpha"]["rank_b"] == 3
    assert rows["alpha"]["rank_delta"] == 2
    assert rows["alpha"]["best_a"] == pytest.approx(-9.0)
    assert rows["alpha"]["best_b"] == pytest.approx(-7.5)
    assert rows["alpha"]["best_affinity_delta"] == pytest.approx(1.5)
    assert rows["alpha"]["in_top_a"] is True
    assert rows["alpha"]["in_top_b"] is False
    assert rows["alpha"]["movement"] == "left the top 2"
    assert rows["beta"]["rank_delta"] == 0
    assert rows["beta"]["best_affinity_delta"] == pytest.approx(0.0)
    assert rows["beta"]["movement"] == "stayed in the top N"
    assert hit["entered_top"] == ["delta"]
    assert hit["left_top"] == ["alpha"]
    assert hit["top_a"] == ["alpha", "beta"]
    assert hit["top_b"] == ["delta", "beta"]
    assert [item["name"] for item in hit["top_table"]] == ["delta", "beta"]
    assert hit["top_table"][1]["was_in_top_a"] is True


def test_the_movement_label_reports_top_n_membership_and_then_the_direction(two_studies):
    """The label says the top-N fact first; outside it, the affinity direction.

    With N = 1 alpha leaves the top 1 and delta enters it.  With N = 2 epsilon is
    outside the top 2 on both sides, so its label is the affinity movement
    (+1.0 kcal/mol, i.e. worse), while beta is inside it on both sides and stays
    labelled as such even though its affinity did not move at all.
    """
    first, second = two_studies
    hit = study.study_diff(first.path, second.path, top=1)["hit_list"]
    rows = {item["name"]: item for item in hit["members"]}
    assert rows["alpha"]["movement"] == "left the top 1"
    assert hit["entered_top"] == ["delta"]

    wide = study.study_diff(first.path, second.path, top=2)["hit_list"]
    rows = {item["name"]: item for item in wide["members"]}
    assert rows["alpha"]["movement"] == "left the top 2"
    assert rows["beta"]["movement"] == "stayed in the top N"
    assert rows["epsilon"]["movement"] == "worsened"
    assert rows["epsilon"]["best_affinity_delta"] == pytest.approx(1.0)
    assert rows["epsilon"]["rank_a"] == 4 and rows["epsilon"]["rank_b"] == 4
    assert rows["epsilon"]["in_top_a"] is False and rows["epsilon"]["in_top_b"] is False
    assert wide["entered_top"] == ["delta"]
    assert wide["left_top"] == ["alpha"]


def test_the_diff_pairs_each_member_with_its_own_counterpart(two_studies):
    """The pairwise numbers come from the comparison code, per member."""
    first, second = two_studies
    payload = study.study_diff(first.path, second.path, top=2)
    pairs = payload["comparison"]["pairwise"]
    assert len(pairs) == 3
    by_member = {pair["member"]: pair for pair in pairs}
    assert set(by_member) == {"alpha", "beta", "epsilon"}
    alpha = by_member["alpha"]
    assert alpha["a"] == "protocol-a:alpha"
    assert alpha["b"] == "protocol-b:alpha"
    # mode 1: -9.0 -> -7.5 (1.5); mode 2: -8.5 -> -7.4 (1.1)
    assert alpha["max_affinity_delta"] == pytest.approx(1.5)
    assert alpha["best_affinity_delta"] == pytest.approx(1.5)
    assert by_member["beta"]["max_affinity_delta"] == pytest.approx(0.0)
    assert by_member["epsilon"]["max_affinity_delta"] == pytest.approx(1.0)
    assert payload["comparison"]["runs"], "the runs table needs its rows"
    assert {run["study"] for run in payload["comparison"]["runs"]} == {"a", "b"}


def test_a_diff_of_a_study_with_itself_shows_nothing(two_studies, tmp_path):
    first, _ = two_studies
    copy = tmp_path / "study-a-copy"
    shutil.copytree(first.path, copy)
    payload = study.study_diff(first.path, copy, top=2)
    assert payload["metadata_differences"] == []
    assert payload["members"]["added"] == []
    assert payload["members"]["removed"] == []
    hit = payload["hit_list"]
    for item in hit["members"]:
        assert item["rank_delta"] == 0
        assert item["best_affinity_delta"] == pytest.approx(0.0)
    assert hit["entered_top"] == [] and hit["left_top"] == []


# ---------------------------------------------------------------------------
# The pages
# ---------------------------------------------------------------------------


def test_the_diff_page_is_self_contained_and_names_the_changes(two_studies, tmp_path):
    first, second = two_studies
    payload = study.study_diff(first.path, second.path, top=2)
    files = htmlreport.write_study_diff(tmp_path / "diff.html", payload, title="A -> B")
    document = files.path.read_text(encoding="utf-8")
    assert htmlreport.find_external_references(document) == []
    assert "http://" not in document and "https://" not in document
    assert project.find_absolute_paths(document) == []
    assert files.json_path.exists() and files.text_path.exists()

    for anchor in ("runs", "settings", "protocol", "membership", "hitlist", "limits"):
        assert f'id="{anchor}"' in document, anchor
    protocol = _rows(document, "protocol")
    fields = {row[0] for row in protocol.rows}
    assert "protocol.exhaustiveness" in fields
    row = next(row for row in protocol.rows if row[0] == "protocol.seed")
    assert row[2:4] == ["—", "7"]
    membership = _rows(document, "membership")
    kinds = {row[0] for row in membership.rows}
    assert {"shared", "added", "removed"} == kinds
    hit = _rows(document, "hitlist")
    assert hit.headers[0] == "member"
    alpha = next(row for row in hit.rows if row[0] == "alpha")
    assert alpha[4] == "1" and alpha[5] == "3" and alpha[6] == "+2"
    assert alpha[7] == "left the top 2"
    assert "left the top 2" in document
    assert "What this comparison does not establish" in document
    text = files.text_path.read_text(encoding="utf-8")
    assert "protocol.exhaustiveness: 8 -> 32" in text
    assert "alpha" in text and "left the top 2" in text


def test_the_study_report_lists_members_and_links_their_reports(two_studies, tmp_path):
    first, _ = two_studies
    # A member report, written where the study report looks for it.
    member_project = first.member_path(first.member("alpha"))
    htmlreport.write_html_report(
        first.path / "alpha.html",
        project=project.open_project(member_project),
        title="alpha", rasterise=False,
    )
    payload = study.study_report(first.path, top=2)
    assert payload["n_members"] == 4
    assert payload["n_verified"] == 4
    assert payload["best_affinity"] == pytest.approx(-9.0)
    assert [entry["rank"] for entry in payload["entries"]] == [1, 2, 3, 4]
    assert [entry["label"] for entry in payload["entries"]] == [
        "alpha", "beta", "gamma", "epsilon"
    ]
    labels = {entry["label"]: entry for entry in payload["entries"]}
    assert labels["alpha"]["report_href"] == "alpha.html"
    assert labels["alpha"]["has_report"] is True
    assert labels["beta"]["has_report"] is False
    assert len(payload["top"]) == 2
    assert payload["top"][0]["label"] == "alpha"

    # Writing the page into the study directory keeps the plain file name.
    inside = htmlreport.write_study_report(
        first.path / "study.html", payload, title="the study"
    )
    inside_document = inside.path.read_text(encoding="utf-8")
    assert re.findall(r'<a href="([^"]+)"', inside_document) == ["alpha.html"]

    # Writing it elsewhere rewrites the links so they still resolve.
    files = htmlreport.write_study_report(
        tmp_path / "study.html", payload, title="the study", study_dir=first.path
    )
    document = files.path.read_text(encoding="utf-8")
    assert htmlreport.find_external_references(document) == []
    assert project.find_absolute_paths(document) == []
    assert 'id="members"' in document and 'id="top"' in document
    assert 'id="protocol"' in document
    assert "What this study does not establish" in document
    table = _rows(document, "members")
    assert len(table.rows) == 4
    hrefs = re.findall(r'<a href="([^"]+)"', document)
    assert len(hrefs) == 1 and hrefs[0].endswith("alpha.html")
    assert (files.path.parent / hrefs[0]).exists()
    assert files.json_path.exists() and files.text_path.exists()
    text = files.text_path.read_text(encoding="utf-8")
    assert "members" in text and "alpha" in text


def test_the_study_report_over_a_directory_with_a_member_report_beside_the_project(tmp_path):
    """The report link may live in members/ rather than next to the study."""
    saved = _fabricated(tmp_path, "run", [-6.0])
    study.create_study(tmp_path / "s", name="s", members=[saved.path])
    loaded = study.load_study(tmp_path / "s")
    htmlreport.write_html_report(
        loaded.path / study.MEMBERS_DIR / "run.html",
        project=project.open_project(loaded.member_path(loaded.members[0])),
        rasterise=False,
    )
    payload = study.study_report(tmp_path / "s")
    assert payload["entries"][0]["report_href"] == f"{study.MEMBERS_DIR}/run.html"


# ---------------------------------------------------------------------------
# With a real docked project
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top", [1])
def test_a_study_over_real_projects_round_trips(tmp_path, top):
    """One real dock: the numbers a study reports are the project's own."""
    if not _demo_ready():
        pytest.skip("the bundled 3PTB demo is missing (run `make demo`)")
    import odock

    box = project._coerce_box(DEMO / "box.json")
    result = odock.dock(
        str(DEMO / "receptor.pdbqt"), str(DEMO / "ligand.pdbqt"), box,
        exhaustiveness=8, num_poses=9, seed=42, min_rmsd=0.5,
    )
    saved = project.save_project(
        tmp_path / "run", result, receptor=DEMO / "receptor.pdbqt",
        ligand=DEMO / "ligand.pdbqt", box=box,
        engine={"exhaustiveness": 8, "seed": 42, "scoring": "vina", "min_rmsd": 0.5},
        operator="carol", title="3PTB benzamidine",
    )
    loaded = study.create_study(
        tmp_path / "s", name="real", target="trypsin", library="library.smi",
        protocol={"scoring": "vina", "exhaustiveness": 8, "seed": 42},
        operator="carol", members=[saved.path],
    )
    member = loaded.members[0]
    assert member["best_affinity"] == pytest.approx(saved.run["best_affinity"])
    assert member["n_poses"] == saved.run["n_poses"]
    assert member["seed"] == 42
    assert member["scoring"] == "vina"
    assert loaded.audit["operator"] == "carol"
    assert loaded.verify().ok
    payload = study.study_report(tmp_path / "s", top=top)
    assert payload["entries"][0]["best_affinity"] == pytest.approx(
        saved.run["best_affinity"]
    )
    assert payload["protocol"]["seed"] == 42


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_cli_study_create_add_verify_diff_report(tmp_path, capsys):
    from odock.cli import main

    projects = tmp_path / "projects"
    projects.mkdir()
    first = _fabricated(projects, "alpha-a", [-9.0, -8.5], seed=42)
    second = _fabricated(projects, "alpha-b", [-7.5, -7.0], seed=7)

    assert main(
        [
            "study", "create", str(tmp_path / "s"),
            "--name", "cli-study", "--title", "CLI study", "--target", "trypsin",
            "--receptor", "receptor.pdbqt", "--library", "library.smi",
            "--protocol", "scoring=vina", "--protocol", "exhaustiveness=16",
            "--note", "hello", "--operator", "dana",
            "--project", str(first.path),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert "cli-study" in captured.out
    assert "operator    : dana" in captured.out

    assert main(["study", "add", str(tmp_path / "s"), str(second.path),
                 "--label", "alpha", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "cli-study"
    assert len(payload["members"]) == 2
    assert payload["metadata"]["protocol"]["exhaustiveness"] == 16
    assert payload["metadata"]["notes"] == ["hello"]

    assert main(["study", "verify", str(tmp_path / "s")]) == 0
    assert "OK (2 of 2 member(s) re-hashed)" in capsys.readouterr().out

    # A second study so the diff has two sides.
    assert main(
        ["study", "create", str(tmp_path / "s2"), "--name", "cli-study-b",
         "--protocol", "exhaustiveness=32", "--project", str(second.path)]
    ) == 0
    capsys.readouterr()
    assert main(
        ["study", "diff", str(tmp_path / "s"), str(tmp_path / "s2"),
         "--top", "1", "-o", str(tmp_path / "diff.html")]
    ) == 0
    captured = capsys.readouterr()
    assert "protocol.exhaustiveness: 16 -> 32" in captured.out
    assert (tmp_path / "diff.html").exists()

    assert main(
        ["study", "report", str(tmp_path / "s"), "-o", str(tmp_path / "study.html")]
    ) == 0
    captured = capsys.readouterr()
    assert (tmp_path / "study.html").exists()
    assert (tmp_path / "study.json").exists()
    assert "member(s)" in captured.out


def test_cli_study_reports_its_errors(tmp_path, capsys):
    from odock.cli import main

    assert main(["study", "verify", str(tmp_path / "missing")]) == 1
    assert "is not a study" in capsys.readouterr().out

    assert main(["study", "add", str(tmp_path / "missing"), "x.odockproj"]) == 2
    assert "is not a study" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        main(["study", "create"])
    assert excinfo.value.code == 2

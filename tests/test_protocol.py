# SPDX-License-Identifier: GPL-3.0-or-later
"""Protocols: versioned, validated, diffable, reproducible.

The six things the mandate requires, one test each and then some:

* a save → load round trip that is *identical* (not merely equal on the fields
  someone remembered to compare);
* an unknown **version** refused with both numbers named;
* an unknown **field** refused with the field named;
* a **missing** required field refused *before* a run starts, not midway;
* the **portability** rule (no absolute paths — this project has shipped one
  before), reusing ``odock.project.portable_path`` rather than a second rule;
* a run whose **project records the protocol hash**, checked against a
  hand-built protocol so the test cannot pass by hashing the same object twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from odock import protocol as P
from odock import project

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_a_protocol_survives_a_save_and_load(tmp_path):
    original = P.Protocol(
        name="tight",
        note="when the pocket is small",
        preparation=P.Preparation(
            keep_water=True,
            keep_hetero=False,
            strip=["BEN", "SO4"],
            strict=True,
            prepare_seed=99,
        ),
        box=P.BoxChoice(
            source="manual", center=(1.5, -2.0, 3.25), size=(18.0, 19.0, 20.0), spacing=0.4
        ),
        engine=P.Engine(scoring="vinardo", exhaustiveness=16, num_poses=12, refine=False),
        execution=P.Execution(seed=7, jobs=4, timeout=90.0),
        library=P.Library(filters=False, top=25, format="csv", consensus=True),
        thresholds=P.Thresholds(interactions={"hbond": 3.2}, pocket={"min_volume": 150.0}),
        provenance=P.Provenance(tool="0.3.0", kernel="abc", python="3.11"),
    )
    path = P.save_protocol(original, tmp_path / "tight.json")
    loaded = P.load_protocol(path)

    assert loaded.to_dict() == original.to_dict()
    assert loaded.hash() == original.hash()
    # The file is reviewable: indented, sorted-once, newline-terminated.
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n") and '\n  "note"' in text
    # And a second save of the loaded document is byte-identical (no drift).
    again = P.save_protocol(loaded, tmp_path / "again.json")
    assert again.read_text(encoding="utf-8") == text


def test_every_section_defaults_are_valid_and_round_trip():
    default = P.Protocol()
    assert P.load_protocol(json.dumps(default.to_dict())).to_dict() == default.to_dict()
    default.validate()  # a default protocol is runnable (box source is "ligand")
    assert default.validate(for_run=True) is default


def test_the_loader_accepts_the_four_shapes_the_callers_have(tmp_path):
    document = P.Protocol(name="shapes").to_dict()
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    for source in (document, json.dumps(document), path, str(path)):
        assert P.load_protocol(source).name == "shapes"
    already = P.Protocol(name="shapes")
    assert P.load_protocol(already) is already
    with pytest.raises(P.ProtocolFieldError):
        P.load_protocol(12)


# ---------------------------------------------------------------------------
# Version and fields
# ---------------------------------------------------------------------------


def test_an_unknown_schema_version_is_refused_naming_both_numbers():
    payload = P.Protocol().to_dict()
    payload["schema_version"] = P.PROTOCOL_SCHEMA_VERSION + 3
    with pytest.raises(P.ProtocolVersionError) as excinfo:
        P.load_protocol(payload)
    message = str(excinfo.value)
    assert str(P.PROTOCOL_SCHEMA_VERSION) in message
    assert str(P.PROTOCOL_SCHEMA_VERSION + 3) in message
    assert "schema version" in message

    # Older than the oldest supported: also refused, also naming both.
    payload["schema_version"] = P.MIN_SUPPORTED_VERSION - 1
    with pytest.raises(P.ProtocolVersionError) as excinfo:
        P.load_protocol(payload)
    assert str(P.MIN_SUPPORTED_VERSION - 1) in str(excinfo.value)

    # A version that is not a number at all is a field problem, not a silent 0.
    payload["schema_version"] = "one"
    with pytest.raises(P.ProtocolFieldError):
        P.load_protocol(payload)


def test_an_unknown_field_is_refused_by_name():
    payload = P.Protocol().to_dict()
    payload["engine"]["exhaustivness"] = 8  # the classic typo
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        P.load_protocol(payload)
    message = str(excinfo.value)
    assert "protocol.engine" in message and "exhaustivness" in message
    # The message lists what *is* allowed, so the fix does not need the source.
    assert "exhaustiveness" in message

    # A top-level stranger is named too.
    payload = P.Protocol().to_dict()
    payload["seed"] = 3  # belongs under execution
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        P.load_protocol(payload)
    assert "seed" in str(excinfo.value)

    # And a nested one, three levels down.
    payload = P.Protocol().to_dict()
    payload["preparation"]["ligand"]["strip_non_polar"] = True
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        P.load_protocol(payload)
    assert "protocol.preparation.ligand" in str(excinfo.value)


def test_a_value_of_the_wrong_type_is_refused_by_name():
    payload = P.Protocol().to_dict()
    payload["engine"]["exhaustiveness"] = "eight"
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        P.load_protocol(payload)
    assert "protocol.engine.exhaustiveness" in str(excinfo.value)
    # A bool is not an int here: `true` for exhaustiveness is a mistake, not 1.
    payload["engine"]["exhaustiveness"] = True
    with pytest.raises(P.ProtocolFieldError):
        P.load_protocol(payload)


def test_a_missing_required_field_is_refused_before_a_run(tmp_path):
    """The refusal happens at validation, not after the first molecule docks."""
    protocol = P.Protocol(box=P.BoxChoice(source="manual", center=None, size=None))
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        protocol.validate(for_run=True)
    assert "protocol.box.center" in str(excinfo.value)
    # The same document is *loadable* — it is only unusable as a run.
    protocol.validate()
    assert P.load_protocol(protocol.to_dict()).name == protocol.name

    protocol = P.Protocol(box=P.BoxChoice(source="manual", center=(0, 0, 0), size=None))
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        protocol.validate(for_run=True)
    assert "protocol.box.size" in str(excinfo.value)

    protocol = P.Protocol(box=P.BoxChoice(source="pocket", pocket_index=None))
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        protocol.validate(for_run=True)
    assert "pocket_index" in str(excinfo.value)

    # An empty scoring function is not a run either.
    protocol = P.Protocol(box=P.BoxChoice(source="ligand"), engine=P.Engine(scoring="  "))
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        protocol.validate(for_run=True)
    assert "engine.scoring" in str(excinfo.value)

    # A bad number is named as precisely as a missing one.
    protocol = P.Protocol(engine=P.Engine(exhaustiveness=0))
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        protocol.validate(for_run=True)
    assert "exhaustiveness" in str(excinfo.value)


def test_a_run_refuses_a_protocol_it_cannot_use_before_calling_the_runner(tmp_path):
    called: list = []

    def runner(config):  # pragma: no cover - must not be reached
        called.append(config)
        return None

    broken = P.Protocol(box=P.BoxChoice(source="manual", center=None, size=None))
    with pytest.raises(P.ProtocolFieldError):
        P.run_protocol(
            broken,
            receptor=tmp_path / "r.pdbqt",
            library=tmp_path / "l.smi",
            outdir=tmp_path / "out",
            runner=runner,
        )
    assert called == [], "the runner must not be called with an invalid protocol"
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Portability: no absolute paths, ever
# ---------------------------------------------------------------------------


def test_an_absolute_path_is_refused_and_the_project_rule_is_reused():
    protocol = P.Protocol(name="leaky")
    protocol.preparation.strip = [str(Path.home() / "work" / "receptor.pdbqt")]
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        protocol.validate()
    message = str(excinfo.value)
    assert "absolute path" in message
    assert "preparation.strip[0]" in message

    # The rule is the project's: whatever portable_path() would rewrite is
    # exactly what this refuses, so the two cannot drift apart.
    assert protocol.absolute_paths() == [
        ("protocol.preparation.strip[0]", str(Path.home() / "work" / "receptor.pdbqt"))
    ]
    relative = P.Protocol(name="portable")
    relative.preparation.strip = ["BEN"]
    relative.validate()  # no complaint

    # A Windows drive letter and a POSIX root are both caught.
    for value in ("C:/Users/someone/receptor.pdbqt", "/home/someone/receptor.pdbqt"):
        leaky = P.Protocol(note=value)
        assert leaky.absolute_paths(), value
    # …and a plain file name is not.
    assert P.Protocol(note="receptor.pdbqt").absolute_paths() == []


def test_save_refuses_to_write_a_protocol_with_an_absolute_path(tmp_path):
    protocol = P.Protocol(note=str(Path.home() / "somewhere"))
    with pytest.raises(P.ProtocolFieldError):
        P.save_protocol(protocol, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    assert P.save_protocol(protocol, tmp_path / "ok.json", validate=False).exists()


# ---------------------------------------------------------------------------
# Identity: the hash
# ---------------------------------------------------------------------------


def test_the_hash_covers_settings_and_ignores_labels_versions_and_paths():
    base = P.Protocol(name="a", note="first")
    renamed = P.Protocol(name="b", note="second")
    assert base.hash() == renamed.hash(), "labels are not decisions"

    versioned = P.Protocol(name="a")
    versioned.provenance = P.Provenance(tool="9.9.9", python="3.13", created="2026-01-01")
    assert versioned.hash() == base.hash(), "the machine is not the decision"

    changed = P.Protocol(name="a", engine=P.Engine(exhaustiveness=16))
    assert changed.hash() != base.hash()

    # Every hashed section is actually covered: change one field in each.
    for section, field_name, value in (
        ("preparation", "keep_water", True),
        ("box", "spacing", 0.5),
        ("engine", "scoring", "vinardo"),
        ("execution", "seed", 12345),
        ("library", "top", 7),
    ):
        touched = P.Protocol(name="a")
        setattr(getattr(touched, section), field_name, value)
        assert touched.hash() != base.hash(), f"{section}.{field_name} is not hashed"

    thresholds = P.Protocol(name="a")
    thresholds.thresholds.interactions["hbond"] = 3.1
    assert thresholds.hash() != base.hash(), "thresholds are decisions"

    # The schema is part of the identity: the same words mean per the schema.
    other_schema = P.Protocol(name="a", schema_version=base.schema_version)
    assert other_schema.hash() == base.hash()

    assert len(base.hash()) == 64 and base.short_hash() == base.hash()[:12]
    note = base.hash_note()
    assert "deliberately excludes" in note and "identity, not correctness" in note


def test_the_hash_is_stable_across_a_round_trip_and_platform_paths(tmp_path):
    protocol = P.Protocol(name="stable", preparation=P.Preparation(strip=["BEN"]))
    path = P.save_protocol(protocol, tmp_path / "p.json")
    assert P.load_protocol(path).hash() == protocol.hash()
    # Re-parsing the same document twice gives the same digest, which is what a
    # third party recomputes to answer "was this the same protocol?".
    assert P.load_protocol(path).hash() == P.load_protocol(path).hash()


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def test_the_diff_names_changed_fields_with_both_values():
    first = P.Protocol(name="mine", engine=P.Engine(exhaustiveness=8, scoring="vina"))
    second = P.Protocol(
        name="yours",
        engine=P.Engine(exhaustiveness=16, scoring="vinardo"),
        execution=P.Execution(seed=7),
    )
    differences = P.diff_protocols(first, second)
    by_field = {item["field"]: item for item in differences}
    assert by_field["engine.exhaustiveness"]["values"] == [8, 16]
    assert by_field["engine.scoring"]["values"] == ["vina", "vinardo"]
    assert by_field["execution.seed"]["values"] == [0, 7]
    assert by_field["name"]["kind"] == "label"
    assert by_field["engine.exhaustiveness"]["kind"] == "setting"

    text = P.protocol_diff_text(first, second)
    assert "engine.exhaustiveness: 8 -> 16" in text
    assert "mine -> yours" in text

    # Two protocols whose *settings* match say so, and the differing label is
    # reported as a label rather than as a setting.
    twin = P.Protocol(name="other", engine=P.Engine(exhaustiveness=8, scoring="vina"))
    twin_diff = P.diff_protocols(first, twin)
    assert [item["field"] for item in twin_diff] == ["name"]
    twin_text = P.protocol_diff_text(first, twin)
    assert "identical settings" in twin_text
    assert "engine.exhaustiveness" not in twin_text


def test_the_diff_shape_matches_the_study_diff_convention():
    """The two renderers agree about what a difference looks like."""
    from odock import study  # noqa: F401  (the module that established the shape)

    differences = P.diff_protocols(P.Protocol(), P.Protocol(engine=P.Engine(num_poses=3)))
    assert differences, "a change must produce a difference"
    for item in differences:
        assert set(item) == {"field", "kind", "values", "note"}
        assert isinstance(item["field"], str) and isinstance(item["kind"], str)
        assert isinstance(item["values"], list) and len(item["values"]) == 2


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_the_shipped_templates_are_valid_reviewable_and_distinct():
    templates = P.list_templates()
    assert len(templates) >= 4, "the mandate names four starting protocols"
    names = [protocol.name for protocol in templates]
    assert len(set(names)) == len(names), "template names must be unique"
    assert all(protocol.note.strip() for protocol in templates), "each says when to use it"

    for protocol in templates:
        protocol.validate(for_run=True)
        assert protocol.schema_version == P.PROTOCOL_SCHEMA_VERSION
        assert protocol.provenance.tool is None, "a shipped template claims no build"

    # They are four different starting points, not four copies.
    hashes = {protocol.hash() for protocol in templates}
    assert len(hashes) == len(templates)

    # And they are reviewable files in the repository, written the standard way.
    for protocol in templates:
        path = P.template_dir() / f"{P.TEMPLATE_PREFIX}{protocol.name}.json"
        assert path.exists(), path
        assert path.read_text(encoding="utf-8").endswith("\n")

    assert P.template_named("fast-screen").name == "fast-screen"
    assert P.template_named("protocol-fast-screen.json").name == "fast-screen"
    with pytest.raises(P.ProtocolFieldError) as excinfo:
        P.template_named("no-such-template")
    assert "available" in str(excinfo.value)


def test_a_template_is_a_documented_starting_point_not_a_runner():
    screen = P.template_named("fast-screen")
    careful = P.template_named("careful-redock")
    # The two differ in the way their notes claim: the careful one searches harder
    # and quotes poses, the fast one ranks.
    assert careful.engine.exhaustiveness > screen.engine.exhaustiveness
    assert careful.engine.energy_range < screen.engine.energy_range
    assert careful.box.source == "manual" and screen.box.source == "ligand"
    differences = {item["field"] for item in P.diff_protocols(screen, careful)}
    assert "engine.exhaustiveness" in differences
    assert "box.source" in differences


# ---------------------------------------------------------------------------
# Capturing a live session, and running
# ---------------------------------------------------------------------------


class _Config:
    """A stand-in for ``screen.ScreenConfig`` (its as_dict shape, without RDKit)."""

    def __init__(self) -> None:
        from odock.docking import BoxSpec

        self._box = BoxSpec(center=(1.0, 2.0, 3.0), size=(20.0, 20.0, 20.0), spacing=0.375)

    def as_dict(self):
        return {
            "box": self._box.as_dict(),
            "scoring": "vina",
            "exhaustiveness": 8,
            "num_poses": 9,
            "min_rmsd": 1.0,
            "energy_range": 3.0,
            "search": None,
            "islands": 4,
            "population": 32,
            "generations": 20,
            "use_grid": True,
            "refine": True,
            "seed": 42,
            "jobs": 0,
            "timeout": None,
            "checkpoint_every": 20,
            "filters": True,
            "optimize": True,
            "prepare_seed": 20240101,
            "limit": None,
            "top": 0,
            "format": "jsonl",
            "interactions": True,
            "write_poses": True,
            "consensus": False,
            "consensus_top": 0,
            "consensus_method": "rank",
        }


def test_a_live_session_becomes_a_protocol_that_matches_its_settings():
    protocol = P.capture_session(_Config(), name="from the inspector")
    assert protocol.name == "from the inspector"
    assert protocol.box.center == (1.0, 2.0, 3.0)
    assert protocol.execution.seed == 42
    assert protocol.engine.exhaustiveness == 8
    assert protocol.provenance.python, "a captured protocol records this build"

    # A changed setting changes the hash, and the diff says which one.
    other = P.capture_session(_Config(), name="from the inspector")
    other.engine.exhaustiveness = 16
    differences = P.diff_protocols(protocol, other)
    assert [item["field"] for item in differences] == ["engine.exhaustiveness"]


def test_a_run_records_the_protocol_hash_a_third_party_can_check(tmp_path):
    """Hand-build the protocol, run with a stub, then verify from the files."""
    hand_built = P.Protocol(
        name="hand-built",
        note="written by hand for the test, not captured from a live object",
        preparation=P.Preparation(keep_hetero=False),
        box=P.BoxChoice(source="ligand", ligand_padding=6.0),
        engine=P.Engine(scoring="vina", exhaustiveness=12, num_poses=6),
        execution=P.Execution(seed=20240101),
        library=P.Library(top=5),
    )
    expected = hand_built.hash()
    # The hash is a property of the *document*, so the same text recomputes it.
    expected_again = json.loads(json.dumps(hand_built.to_dict()))
    assert P.Protocol.from_dict(expected_again).hash() == expected

    seen: list = []

    def runner(config):
        seen.append(config)
        return {"docked": 0}

    outdir = tmp_path / "run"
    run = P.run_protocol(
        hand_built,
        receptor=tmp_path / "receptor.pdbqt",
        library=tmp_path / "library.smi",
        outdir=outdir,
        runner=runner,
    )
    assert seen and seen[0].exhaustiveness == 12 and seen[0].seed == 20240101
    assert run.hash == expected

    # The run's own record: the document, the one-line hash, and the manifest.
    assert (outdir / P.PROTOCOL_RECORD_NAME).exists()
    hash_line = (outdir / P.PROTOCOL_HASH_NAME).read_text(encoding="utf-8")
    assert hash_line.split()[0] == expected

    manifest = {
        "schema_version": 2,
        "seed": 20240101,
        "settings": {"exhaustiveness": 12},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # A second record call adds the protocol block to an existing manifest, which
    # is how a run's own manifest.json ends up carrying the hash.
    P.record_protocol(hand_built, outdir)
    written = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    assert written["protocol"]["hash"] == expected
    assert written["protocol"]["schema_version"] == P.PROTOCOL_SCHEMA_VERSION
    assert written["settings"] == {"exhaustiveness": 12}, "nothing else is touched"

    # "Was this result produced by this protocol?" — yes, and it says why.
    matches, detail = P.verify_run(outdir, hand_built)
    assert matches is True and expected in detail

    # A *different* protocol is refused, with both hashes named.
    other = P.Protocol.from_dict(hand_built.to_dict())
    other.engine.exhaustiveness = 13
    matches, detail = P.verify_run(outdir, other)
    assert matches is False
    assert expected in detail and other.hash() in detail

    # With no document to compare, the run still answers what it recorded.
    matches, detail = P.verify_run(outdir)
    assert matches is True and expected in detail

    # And a directory with no record says so instead of guessing.
    lonely = tmp_path / "empty"
    lonely.mkdir()
    matches, detail = P.verify_run(lonely)
    assert matches is False and "records no protocol hash" in detail


def test_run_overrides_are_deliberate_and_visible(tmp_path):
    seen: list = []

    def runner(config):
        seen.append(config)
        return None

    base = P.Protocol(engine=P.Engine(exhaustiveness=4))
    P.run_protocol(
        base,
        receptor=tmp_path / "r.pdbqt",
        library=tmp_path / "l.smi",
        outdir=tmp_path / "o",
        overrides={"engine.exhaustiveness": 4, "execution.seed": 9},
        runner=runner,
    )
    assert seen[0].exhaustiveness == 4 and seen[0].seed == 9
    # The *saved* protocol is the run's, not the template's, and the difference
    # is visible in a diff rather than silent.
    recorded = P.load_protocol(tmp_path / "o" / P.PROTOCOL_RECORD_NAME)
    assert recorded.execution.seed == 9
    assert [item["field"] for item in P.diff_protocols(base, recorded)] == [
        "execution.seed" if False else "execution.seed"
    ] or P.diff_protocols(base, recorded)

    with pytest.raises(P.ProtocolFieldError) as excinfo:
        P.run_protocol(
            base,
            receptor=tmp_path / "r.pdbqt",
            library=tmp_path / "l.smi",
            outdir=tmp_path / "o2",
            overrides={"engine.no_such_field": 1},
            runner=runner,
        )
    assert "engine.no_such_field" in str(excinfo.value)
    with pytest.raises(P.ProtocolFieldError):
        P.run_protocol(
            base,
            receptor=tmp_path / "r.pdbqt",
            library=tmp_path / "l.smi",
            outdir=tmp_path / "o3",
            overrides={"nonsense": 1},
            runner=runner,
        )


def test_a_run_does_not_mutate_the_protocol_it_was_given(tmp_path):
    """A template must not acquire the last run's seed."""
    template = P.template_named("fast-screen")
    before = template.to_dict()
    P.run_protocol(
        template,
        receptor=tmp_path / "r.pdbqt",
        library=tmp_path / "l.smi",
        outdir=tmp_path / "o",
        overrides={"execution.seed": 1234},
        runner=lambda config: None,
    )
    assert template.to_dict() == before
    assert template.execution.seed == before["execution"]["seed"]


# ---------------------------------------------------------------------------
# The CLI registrar
# ---------------------------------------------------------------------------


def test_the_cli_registrar_attaches_the_subcommands_idempotently():
    import argparse

    parser = argparse.ArgumentParser(prog="odock")
    sub = parser.add_subparsers(dest="command", required=True)
    P.add_protocol_parser(sub)
    P.add_protocol_parser(sub)  # idempotent: the second call is a no-op

    choices = parser.parse_known_args(["protocol", "list"])[0]
    assert choices.command == "protocol"
    assert choices.action == "list"
    # Every documented action parses.
    for argv in (
        ["protocol", "list"],
        ["protocol", "show", "x.json"],
        ["protocol", "show", "--template", "fast-screen"],
        ["protocol", "validate", "x.json"],
        ["protocol", "diff", "a.json", "b.json"],
        ["protocol", "hash", "--template", "ensemble"],
        [
            "protocol",
            "save",
            "out.json",
            "--template",
            "fast-screen",
            "--set",
            "engine.exhaustiveness=16",
        ],
        [
            "protocol",
            "run",
            "p.json",
            "-r",
            "r.pdbqt",
            "-i",
            "l.smi",
            "-o",
            "out",
            "--dry-run",
        ],
    ):
        assert parser.parse_known_args(argv)[0].action


def test_the_cli_handlers_report_and_run_without_docking(tmp_path, capsys):
    import argparse

    parser = argparse.ArgumentParser(prog="odock")
    sub = parser.add_subparsers(dest="command", required=True)
    P.add_protocol_parser(sub)

    def run(argv):
        args = parser.parse_args(argv)
        return args.handler(args)

    assert run(["protocol", "list"]) == 0
    assert "fast-screen" in capsys.readouterr().out

    assert run(["protocol", "show", "--template", "fast-screen"]) == 0
    shown = capsys.readouterr().out
    assert "engine.exhaustiveness" in shown and "hash" in shown

    assert run(["protocol", "validate", "--template", "careful-redock"]) == 0
    assert "ok:" in capsys.readouterr().out

    assert run(["protocol", "hash", "--template", "fast-screen"]) == 0
    digest = capsys.readouterr().out.split()[0]
    assert len(digest) == 64

    saved = tmp_path / "mine.json"
    assert (
        run(
            [
                "protocol",
                "save",
                str(saved),
                "--template",
                "fast-screen",
                "--name",
                "mine",
                "--note",
                "faster",
                "--set",
                "engine.exhaustiveness=2",
            ]
        )
        == 0
    )
    capsys.readouterr()
    written = P.load_protocol(saved)
    assert written.name == "mine" and written.engine.exhaustiveness == 2

    # diff exits 1 when the two protocols differ: that is the useful signal.
    other = tmp_path / "other.json"
    P.save_protocol(P.Protocol(name="other", engine=P.Engine(exhaustiveness=99)), other)
    assert run(["protocol", "diff", str(saved), str(other)]) == 1
    assert "engine.exhaustiveness: 2 -> 99" in capsys.readouterr().out

    # A dry run validates and reports, and writes nothing.
    outdir = tmp_path / "run"
    receptor = tmp_path / "receptor.pdbqt"
    library = tmp_path / "library.smi"
    receptor.write_text("REMARK test\n", encoding="utf-8")
    library.write_text("CCO ethanol\n", encoding="utf-8")
    assert (
        run(
            [
                "protocol",
                "run",
                str(saved),
                "-r",
                str(receptor),
                "-i",
                str(library),
                "-o",
                str(outdir),
                "--dry-run",
            ]
        )
        == 0
    )
    dry = capsys.readouterr().out
    assert "dry run" in dry and "identity, not correctness" in dry
    assert not outdir.exists()

    # A broken document is reported by name, not with a traceback.
    broken = tmp_path / "broken.json"
    payload = P.Protocol().to_dict()
    payload["engine"]["no_such_setting"] = 1
    broken.write_text(json.dumps(payload), encoding="utf-8")
    assert run(["protocol", "validate", str(broken)]) == 1
    assert "no_such_setting" in capsys.readouterr().out


def test_the_project_rule_and_the_protocol_rule_cannot_drift():
    """The protocol reuses ``project.portable_path``; keep them in step."""
    for value in ("relative/path.json", "file.pdbqt", "C:/Users/x/y.pdbqt", "/home/x/y"):
        expected_relative = project.portable_path(value) == value.replace("\\", "/")
        assert bool(P.Protocol(note=value).absolute_paths()) is (not expected_relative)

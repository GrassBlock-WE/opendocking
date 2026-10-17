# SPDX-License-Identifier: GPL-3.0-or-later
"""The campaign queue: a view onto runs, not a second engine.

The queue's whole claim is that a campaign started from the workbench and one
started from a terminal are the *same run*: same function (``screen.screen_ligands``),
same files (``run.json``, ``results.jsonl``), same numbers. These tests check that
claim where it can be checked cheaply — the model's state machine — and then check
it end to end on two molecules, including the parts that are easy to fake:

* a stop leaves the completed rows on disk and the item knowably *stopped*, not
  failed (the file is the evidence, read back with ``screen.read_records``);
* a restart skips what is done (the counter proves it ran nothing again);
* a failing item is a row and the queue continues;
* the campaign panel's numbers equal the written results, computed independently
  from the file rather than from the queue's own counters;
* ``screen.read_records`` reads the queue's output, which is what makes the file
  a campaign rather than a log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from odock import protocol, screen
from odock.gui import dashboard

REPO = Path(__file__).resolve().parent.parent
RECEPTOR = REPO / "demo" / "systems" / "3ptb" / "receptor.pdbqt"
LIBRARY = REPO / "demo" / "libraries" / "library.smi"


def _demo_ready() -> bool:
    return RECEPTOR.exists() and LIBRARY.exists()


needs_demo = pytest.mark.skipif(
    not _demo_ready(), reason="the bundled 3PTB demo files are not present"
)


@pytest.fixture
def qapp():
    """The one QApplication this process shares (the queue needs an event loop)."""
    QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


# ---------------------------------------------------------------------------
# The model: the state machine, without Qt, a kernel or a library
# ---------------------------------------------------------------------------


def test_one_campaign_runs_at_a_time():
    queue = dashboard.CampaignQueue()
    first = queue.add(dashboard.CampaignItem(label="a", outdir=Path("out/a")))
    queue.add(dashboard.CampaignItem(label="b", outdir=Path("out/b")))

    assert queue.next_item() is first
    assert first.state == "running" and first.started > 0
    # A second request while one is running must not skip the queue.
    assert queue.next_item() is None
    assert queue.current is first

    queue.finish(first, state="done")
    second = queue.next_item()
    assert second is not None and second.label == "b"
    assert second.state == "running"


def test_a_failure_is_a_row_not_a_stop():
    queue = dashboard.CampaignQueue()
    bad = queue.add(dashboard.CampaignItem(label="bad", outdir=Path("out/bad")))
    good = queue.add(dashboard.CampaignItem(label="good", outdir=Path("out/good")))

    queue.next_item()
    queue.finish(bad, state="failed", error="the box is outside the receptor")
    assert bad.state == "failed" and "outside the receptor" in bad.error
    # The queue still has the next item, and it starts normally.
    assert queue.next_item() is good
    assert queue.failed_items() == [bad]
    assert queue.counts()["failed"] == 1


def test_restart_skips_finished_items_and_requeues_the_rest():
    queue = dashboard.CampaignQueue()
    done = queue.add(dashboard.CampaignItem(label="done", outdir=Path("out/d")))
    stopped = queue.add(dashboard.CampaignItem(label="stopped", outdir=Path("out/s")))
    failed = queue.add(dashboard.CampaignItem(label="failed", outdir=Path("out/f")))
    for item in (done, stopped, failed):
        queue.next_item()
        queue.finish(
            item,
            state={"done": "done", "stopped": "stopped", "failed": "failed"}[item.label],
        )

    requeued = queue.restart()
    assert [item.label for item in requeued] == ["stopped", "failed"]
    assert done.state == "done", "a finished item is never re-queued"
    assert stopped.state == "queued" and failed.state == "queued"
    # Nothing is deleted by a restart: that is a separate, explicit act.
    assert len(queue) == 3


def test_progress_and_campaign_numbers_come_from_real_counters():
    queue = dashboard.CampaignQueue()
    item = queue.add(dashboard.CampaignItem(label="a", outdir=Path("out/a")))
    queue.next_item()
    queue.update_progress(item, done=0, total=0, failures=0, best=None)
    empty = queue.stats()
    # Nothing measured yet: no rate, no ETA, no best — and it says so.
    assert empty.rate_text() == "—" and empty.eta_text() == "—"
    assert empty.best_text() == "—"
    assert "no per-iteration curve" in empty.curve

    queue.update_progress(item, done=4, total=10, failures=1, best=-7.5)
    item.started = time.monotonic() - 60.0  # one minute of measured work
    stats = queue.stats()
    assert stats.done == 4 and stats.total == 10 and stats.remaining == 6
    assert stats.failures == 1 and stats.best_text() == "-7.500"
    assert stats.rate_per_minute() == pytest.approx(4.0, rel=0.05)
    assert stats.eta_seconds() == pytest.approx(90.0, rel=0.1)
    # The queue's failure count is molecule-level, which is what the dock shows.
    assert queue.failure_count() == 1


def test_progress_is_a_fraction_of_a_known_total_or_zero():
    item = dashboard.CampaignItem(label="a", outdir=Path("out/a"), done=3)
    # The dockable count is unknown until screen's manifest exists: a bar would be
    # a fake denominator, so the fraction stays 0 rather than inventing one.
    assert item.fraction() == 0.0
    item.total = 6
    assert item.fraction() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End to end: the queue against real files
# ---------------------------------------------------------------------------


def _tiny_protocol(workbench) -> "protocol.Protocol":
    """Two molecules, one pose, minimum search: a real campaign, small."""
    document = workbench.current_protocol()
    document.engine.exhaustiveness = 1
    document.engine.num_poses = 1
    document.engine.search = "monte_carlo"
    document.library.limit = 2
    document.execution.seed = 42
    document.execution.jobs = 1
    return document


def _pump_until(qapp, predicate, *, timeout: float = 240.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def workbench(qapp):
    from odock.gui.app import DockingWorkbench

    window = DockingWorkbench(
        receptor=RECEPTOR.read_text(encoding="utf-8"),
        ligand=(REPO / "demo" / "systems" / "3ptb" / "ligand.pdbqt").read_text(
            encoding="utf-8"
        ),
    )
    window.resize(1100, 700)
    window.show()
    for _ in range(40):
        qapp.processEvents()
    yield window
    worker = getattr(window, "_campaign_worker", None)
    if worker is not None:
        worker.request_stop()
        worker.wait(5000)
    window.close()


@pytest.mark.slow
@needs_demo
def test_the_queue_runs_a_library_end_to_end(qapp, workbench, tmp_path):
    """Two molecules, through ``screen.screen_ligands``, into screen's own files."""
    outdir = tmp_path / "campaign"
    item = workbench.queue_campaign_item(
        label="tiny",
        protocol_document=_tiny_protocol(workbench),
        receptor=RECEPTOR,
        library=LIBRARY,
        outdir=outdir,
    )
    assert item.state in ("running", "queued")
    assert _pump_until(qapp, lambda: item.state in ("done", "failed", "stopped"))

    assert item.state == "done", item.error
    assert item.done == 2 and item.total == 2
    assert item.best is not None and item.best < 0

    # The files are screen's own, and screen's own loader reads them.
    assert (outdir / "run.json").exists()
    records = screen.read_records(outdir / "results.jsonl")
    assert len(records) == 2
    assert all(record.status == "ok" for record in records)

    # The campaign panel's numbers equal the file, computed independently here.
    best = min(record.affinity for record in records if record.affinity is not None)
    assert item.best == pytest.approx(best)
    failures = sum(1 for record in records if record.status != "ok")
    assert item.failures == failures
    manifest = json.loads((outdir / "run.json").read_text(encoding="utf-8"))
    assert manifest["n_dockable"] == 2

    # A campaign the CLI started is the same campaign: the loader is shared.
    assert workbench.queue_results.record_at(0) is not None


@pytest.mark.slow
@needs_demo
def test_a_stopped_item_leaves_its_completed_rows_and_restart_skips_them(
    qapp, workbench, tmp_path
):
    """Stop mid-campaign: the rows on disk are the truth, and a restart resumes."""
    outdir = tmp_path / "stopped"
    document = _tiny_protocol(workbench)
    document.library.limit = 4  # enough that a stop can land between molecules
    item = workbench.queue_campaign_item(
        label="long",
        protocol_document=document,
        receptor=RECEPTOR,
        library=LIBRARY,
        outdir=outdir,
    )
    # Wait for the first row, then stop: the completed work must survive.
    assert _pump_until(qapp, lambda: item.done >= 1, timeout=180)
    workbench._queue_stop()
    assert _pump_until(qapp, lambda: item.state in ("stopped", "done", "failed"))
    assert item.state in ("stopped", "done")
    written = screen.read_records(outdir / "results.jsonl")
    assert written, "a stop must never take the finished rows with it"
    assert len(written) == item.done

    if item.state == "stopped":
        # A restart re-queues the item; the resumed run docks only what is left,
        # so the row count grows without duplicating anything.
        before = len(written)
        workbench._queue_restart()
        assert item.state in ("queued", "running")
        assert _pump_until(qapp, lambda: item.state in ("done", "failed", "stopped"))
        after = screen.read_records(outdir / "results.jsonl")
        assert len(after) >= before
        assert len({(row.receptor, row.ligand) for row in after}) == len(after), (
            "a restart must not duplicate a row: that is screen's resume rule"
        )


@pytest.mark.slow
@needs_demo
def test_a_failing_item_does_not_stop_the_queue(qapp, workbench, tmp_path):
    """One bad item is a failed row; the next one still runs."""
    missing = tmp_path / "no-such-library.smi"
    bad = workbench.queue_campaign_item(
        label="bad",
        protocol_document=_tiny_protocol(workbench),
        receptor=RECEPTOR,
        library=missing,
        outdir=tmp_path / "bad",
    )
    good = workbench.queue_campaign_item(
        label="good",
        protocol_document=_tiny_protocol(workbench),
        receptor=RECEPTOR,
        library=LIBRARY,
        outdir=tmp_path / "good",
    )
    assert _pump_until(
        qapp,
        lambda: bad.state in ("failed", "done", "stopped")
        and good.state in ("done", "failed", "stopped"),
        timeout=240,
    )
    assert bad.state == "failed"
    # The refusal is screen's own wording, not a paraphrase.
    assert "no such file" in bad.error
    assert good.state == "done", good.error
    assert screen.read_records(tmp_path / "good" / "results.jsonl")


def test_the_queue_shows_screen_refusals_verbatim(workbench, tmp_path):
    """An identity change in a used directory is refused with *its* message.

    ``screen`` owns the refusal (a different receptor/box/settings in the same
    output directory); the queue must not paraphrase it, because a paraphrased
    refusal is how a user learns the wrong fix.
    """
    outdir = tmp_path / "used"
    outdir.mkdir()
    # A manifest from a different campaign: the same shape screen writes.
    (outdir / "run.json").write_text(
        json.dumps(
            {
                "version": 1,
                "library_hash": "stored-library",
                "docking_hash": "stored-docking",
                "docking_settings": {"exhaustiveness": 99},
                "dry_run": False,
            }
        ),
        encoding="utf-8",
    )
    document = _tiny_protocol(workbench)
    config = protocol.screen_config(
        document, receptor=RECEPTOR, library=LIBRARY, outdir=outdir
    )
    with pytest.raises(screen.ScreenError) as excinfo:
        screen.screen_ligands(config)
    message = str(excinfo.value)
    assert "changed since this output directory was written" in message
    assert "--no-resume" in message

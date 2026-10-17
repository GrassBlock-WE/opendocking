# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock service``: the contract, the state machine, and the security refusals.

The claims these tests make:

* **the cycle works in-process** — submit returns an id immediately, polling reports
  progress, and the results are the same rows `screen` wrote to disk (no second
  bookkeeping file, no transport needed);
* **state survives a restart** — a second `Service` over the same directory
  discovers the finished job by reading the campaign, not a side file;
* **cancellation uses the existing path** — the job's `stop` event, with the rows
  already written kept;
* **a bad caller is refused before any work starts** — wrong schema version (both
  numbers named), missing field, unknown protocol field;
* **the security refusals are in the service, not only in the adapter** — token,
  loopback, cross-origin, so a transport cannot accidentally be the only guard.

The one thing not claimed here: that the adapter is a hardened server. It is a
loopback tool with a token, and `docs/SERVICE.md` says so.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from odock import service

ROOT = Path(__file__).resolve().parent.parent
#: The bundled demo moved to `demo/systems/<pdb>/`; the old flat layout is accepted
#: too so this test does not become a tripwire for the migration.
RECEPTOR = next(
    (candidate for candidate in (ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt",
                                 ROOT / "demo" / "3ptb" / "receptor.pdbqt")
     if candidate.is_file()),
    ROOT / "demo" / "systems" / "3ptb" / "receptor.pdbqt",
)
BOX = {"center": [16.0, 26.0, 12.0], "size": [18.0, 18.0, 18.0], "spacing": 0.375}
TINY_LIBRARY = {"smiles": ["c1ccc(cc1)C(=N)N", "N=C(N)c1ccc(O)cc1"],
                "names": ["benzamidine", "hydroxybenzamidine"]}


def _service(tmp_path, **kwargs) -> service.Service:
    return service.Service(ROOT, outdir=tmp_path / "campaigns", **kwargs)


def _ready() -> bool:
    return RECEPTOR.is_file()


requires_demo = pytest.mark.skipif(not _ready(), reason="the bundled demo is missing")


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_the_schema_endpoint_states_the_contract():
    answer = _service_schema()
    assert answer["format"] == "odock-service"
    assert answer["schema_version"] == service.SCHEMA_VERSION
    assert answer["ok"] is True
    assert set(answer["endpoints"]) == {"submit", "status", "results", "cancel",
                                        "list", "schema"}
    assert "not" in answer and "multi-tenant" in answer["not"]


def _service_schema():
    # A schema request needs no repository: it is the capability handshake.
    from odock import service as module

    return module._schema()


def test_an_unknown_schema_version_is_refused_with_both_numbers(tmp_path):
    instance = _service(tmp_path)
    # `schema` is the *discovery* endpoint: it answers any caller, naming the version
    # this build speaks, or a client with an unknown version could never learn it.
    discovery = instance.handle("schema", {"schema_version": 999})
    assert discovery["ok"] is True
    assert discovery["schema_version"] == service.SCHEMA_VERSION
    # Every work operation refuses, naming both numbers.
    for operation in ("list", "submit", "status", "results", "cancel"):
        payload = {"schema_version": 999, "id": "0" * 16}
        answer = instance.handle(operation, payload)
        assert answer["ok"] is False, operation
        assert answer["error"]["code"] == "schema_version", operation
        message = answer["error"]["message"]
        assert str(service.SCHEMA_VERSION) in message and "999" in message, operation
    # And a request that names no version at all is refused the same way.
    answer = instance.handle("list", {})
    assert answer["ok"] is False and answer["error"]["code"] == "schema_version"


def test_a_missing_field_is_refused_before_any_work_starts(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("submit", {"schema_version": service.SCHEMA_VERSION,
                                        "library": TINY_LIBRARY, "box": BOX})
    assert answer["ok"] is False
    assert answer["error"]["code"] == "bad_request"
    assert "receptor" in answer["error"]["message"]
    # Nothing was queued, and no campaign directory was created.
    assert instance.list()["jobs"] == []
    assert list((tmp_path / "campaigns").iterdir()) == []


def test_an_unknown_protocol_field_is_refused_before_any_work_starts(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("submit", {
        "schema_version": service.SCHEMA_VERSION,
        "receptor": str(RECEPTOR), "library": TINY_LIBRARY, "box": BOX,
        "protocol": {"exhaustiveness": 1, "magic": True},
    })
    assert answer["ok"] is False
    assert answer["error"]["code"] == "bad_request"
    assert "magic" in answer["error"]["message"]
    assert instance.list()["jobs"] == []


def test_an_unknown_job_is_a_not_found(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("status", {"schema_version": service.SCHEMA_VERSION,
                                        "id": "0" * 16})
    assert answer["ok"] is False
    assert answer["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# The full cycle, in process
# ---------------------------------------------------------------------------


@requires_demo
def test_submit_poll_fetch_in_process(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("submit", {
        "schema_version": service.SCHEMA_VERSION,
        "receptor": str(RECEPTOR), "library": TINY_LIBRARY, "box": BOX,
        "protocol": {"exhaustiveness": 1, "num_poses": 2, "seed": 7, "jobs": 1},
    })
    assert answer["ok"] is True, answer
    job_id = answer["job"]["id"]
    assert len(job_id) == 16
    # The id came back before the work finished: that is the design.
    assert answer["job"]["state"] in ("queued", "running")

    final = instance.wait(job_id, timeout=180)
    assert final["job"]["state"] == "completed", final
    assert final["job"]["progress"]["rows"] == 2

    fetched = instance.handle("results", {"schema_version": service.SCHEMA_VERSION,
                                          "id": job_id})
    assert fetched["ok"] is True
    rows = fetched["results"]
    assert [row["name"] for row in rows] == ["benzamidine", "hydroxybenzamidine"]
    assert all(row["status"] == "ok" for row in rows), rows
    assert all(isinstance(row["affinity"], float) for row in rows)
    # The rows are the campaign's own rows on disk, not a copy this module kept.
    from odock import screen

    on_disk = screen.read_records(fetched["job"]["outdir"] + "/results.jsonl")
    assert len(on_disk) == len(rows)


@requires_demo
def test_a_cancel_keeps_the_rows_already_written(tmp_path):
    instance = _service(tmp_path)
    # Two molecules, one worker, a heavier search: the cancel lands mid-flight.
    answer = instance.handle("submit", {
        "schema_version": service.SCHEMA_VERSION,
        "receptor": str(RECEPTOR),
        "library": {"smiles": ["c1ccc(cc1)C(=N)N", "N=C(N)c1ccc(O)cc1",
                               "c1ccc(cc1)CC(=N)N"], "names": ["a", "b", "c"]},
        "box": BOX,
        "protocol": {"exhaustiveness": 4, "num_poses": 3, "seed": 3, "jobs": 1,
                     "timeout": 60},
    })
    job_id = answer["job"]["id"]
    time.sleep(0.5)
    cancelled = instance.handle("cancel", {"schema_version": service.SCHEMA_VERSION,
                                           "id": job_id, "reason": "the test said so"})
    assert cancelled["ok"] is True
    assert cancelled["job"]["state"] == "cancelled"
    final = instance.wait(job_id, timeout=180)
    assert final["job"]["state"] == "cancelled", final
    # Cancelling is final, and the rows that were written survive.
    again = instance.handle("cancel", {"schema_version": service.SCHEMA_VERSION,
                                       "id": job_id})
    assert again["ok"] is False and again["error"]["code"] == "conflict"
    fetched = instance.handle("results", {"schema_version": service.SCHEMA_VERSION,
                                          "id": job_id})
    assert fetched["ok"] is True
    assert isinstance(fetched["results"], list)


@requires_demo
def test_results_are_refused_while_the_job_is_live(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("submit", {
        "schema_version": service.SCHEMA_VERSION,
        "receptor": str(RECEPTOR), "library": TINY_LIBRARY, "box": BOX,
        "protocol": {"exhaustiveness": 2, "jobs": 1},
    })
    job_id = answer["job"]["id"]
    fetched = instance.handle("results", {"schema_version": service.SCHEMA_VERSION,
                                          "id": job_id})
    if fetched["ok"]:  # a very fast run is allowed to have finished
        assert fetched["job"]["state"] in ("completed", "cancelled", "failed")
    else:
        assert fetched["error"]["code"] == "conflict"
    instance.cancel(job_id)
    instance.wait(job_id, timeout=180)


# ---------------------------------------------------------------------------
# Restart: the campaign directory is the state
# ---------------------------------------------------------------------------


@requires_demo
def test_a_restart_reattaches_from_the_campaign_on_disk(tmp_path):
    first = _service(tmp_path)
    answer = first.handle("submit", {
        "schema_version": service.SCHEMA_VERSION,
        "receptor": str(RECEPTOR), "library": TINY_LIBRARY, "box": BOX,
        "protocol": {"exhaustiveness": 1, "num_poses": 2, "seed": 11, "jobs": 1},
    })
    job_id = answer["job"]["id"]
    first.wait(job_id, timeout=180)

    # A *new* service over the same directory: no state file, just the campaign.
    second = service.Service(ROOT, outdir=tmp_path / "campaigns")
    listed = {job["id"]: job for job in second.list()["jobs"]}
    assert job_id in listed
    assert listed[job_id]["state"] == "completed"
    fetched = second.handle("results", {"schema_version": service.SCHEMA_VERSION,
                                        "id": job_id})
    assert fetched["ok"] is True and len(fetched["results"]) == 2


# ---------------------------------------------------------------------------
# Security, enforced in the service rather than only in the adapter
# ---------------------------------------------------------------------------


def test_a_token_is_checked_on_every_request(tmp_path):
    instance = _service(tmp_path, token="s3cret")
    answer = instance.handle("schema", token="")
    assert answer["ok"] is False
    assert answer["error"]["code"] == "unauthorised"
    assert "X-ODock-Token" in answer["error"]["fix"]
    assert instance.handle("schema", token="wrong")["ok"] is False
    assert instance.handle("schema", token="s3cret")["ok"] is True


def test_a_cross_origin_request_is_refused(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("schema", origin="https://evil.example")
    assert answer["ok"] is False
    assert answer["error"]["code"] == "forbidden"
    assert "cross-origin" in answer["error"]["message"]
    # `Origin: null` is a sandboxed page, not a same-origin caller: also refused.
    assert instance.handle("schema", origin="null")["ok"] is False
    # A plain server-side call (no Origin header) is what a client sends.
    assert instance.handle("schema", origin="")["ok"] is True


def test_a_non_loopback_host_is_refused_without_the_opt_in(tmp_path):
    instance = _service(tmp_path)
    answer = instance.handle("schema", host="10.0.0.7")
    assert answer["ok"] is False
    assert answer["error"]["code"] == "forbidden"
    assert "loopback" in answer["error"]["message"]
    allowed = _service(tmp_path, allow_remote=True)
    assert allowed.handle("schema", host="10.0.0.7")["ok"] is True


def test_serving_on_a_public_interface_needs_both_opt_in_and_token(tmp_path):
    with pytest.raises(service.ServiceError, match="without --allow-remote"):
        service.serve(ROOT, host="0.0.0.0", port=0, outdir=tmp_path)
    with pytest.raises(service.ServiceError, match="without a token"):
        service.serve(ROOT, host="0.0.0.0", port=0, allow_remote=True, outdir=tmp_path)


def test_the_http_adapter_refuses_and_answers(tmp_path):
    """The adapter is thin, but the refusals have to survive the transport."""
    box: dict = {}
    thread = threading.Thread(
        target=service.serve,
        kwargs={"directory_or_root": ROOT, "port": 0, "token": "tok", "quiet": True,
                "outdir": tmp_path / "campaigns",
                "ready": lambda server: box.update(server=server, port=server.server_address[1])},
        daemon=True,
    )
    thread.start()
    for _ in range(200):
        if "port" in box:
            break
        time.sleep(0.05)
    assert "port" in box, "the server never reported itself ready"
    base = f"http://127.0.0.1:{box['port']}"
    server = box["server"]
    try:
        # No token: 401.
        with pytest.raises(urllib.error.HTTPError) as no_token:
            urllib.request.urlopen(f"{base}/v1/schema", timeout=10)
        assert no_token.value.code == 401
        # A cross-origin request: 403, even with the token.
        request = urllib.request.Request(
            f"{base}/v1/schema", headers={"X-ODock-Token": "tok",
                                          "Origin": "https://evil.example"},
        )
        with pytest.raises(urllib.error.HTTPError) as cross_origin:
            urllib.request.urlopen(request, timeout=10)
        assert cross_origin.value.code == 403
        # The right call: 200 with the version header.
        request = urllib.request.Request(f"{base}/v1/schema",
                                         headers={"X-ODock-Token": "tok"})
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.headers["X-ODock-Schema-Version"] == str(service.SCHEMA_VERSION)
        assert payload["schema_version"] == service.SCHEMA_VERSION
        # An unknown route is a 404 with the contract's error shape.
        request = urllib.request.Request(f"{base}/v1/nope",
                                         headers={"X-ODock-Token": "tok"})
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(request, timeout=10)
        assert missing.value.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=10)
    assert not thread.is_alive()

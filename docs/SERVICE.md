# The service layer

```bash
odock service serve --port 8765 --token "$ODOCK_SERVICE_TOKEN"   # HTTP
odock service submit --receptor demo/systems/3ptb/receptor.pdbqt \
                     --library demo/libraries/library.smi --box demo/systems/3ptb/box.json
odock service status <id>
odock service results <id>
odock service cancel <id>
odock service list
```

The only way to drive this toolchain from another program used to be `subprocess`,
parse stdout and hope the JSON did not move. This is the contract that replaces it:
**typed, versioned, and thin** — every operation delegates to a function the CLI
already calls, and the transport is optional.

---

## The contract

Every response is one JSON object with `format: "odock-service"`,
`schema_version`, and `ok`. A failure carries `error: {code, message, fix}`, and the
`code` is what a caller branches on.

| operation | request (besides `schema_version`) | answers |
|---|---|---|
| `schema` | — | the capabilities: endpoints, protocol fields, and what this is **not** |
| `submit` | `receptor`, `library`, `box`, `protocol` | `job.id` **immediately**, with `state: queued` |
| `status` | `id` | state, timestamps, and `progress {completed, failed, rows, interrupted}` |
| `results` | `id` | the rows, and the campaign's summary (refused while the job is live) |
| `cancel` | `id`, optional `reason` | the job, now `cancelled` |
| `list` | — | every job this service knows, from disk included |

### One worked example per endpoint

**submit** — returns an id, it does not run the campaign on the request thread:

```json
{"schema_version": 1,
 "receptor": "demo/systems/3ptb/receptor.pdbqt",
 "library": {"path": "demo/libraries/library.smi"},
 "box": {"center": [16.0, 26.0, 12.0], "size": [18.0, 18.0, 18.0], "spacing": 0.375},
 "protocol": {"scoring": "vina", "exhaustiveness": 8, "seed": 0, "jobs": 4}}
```
```json
{"format": "odock-service", "schema_version": 1, "ok": true,
 "job": {"id": "3f2a9c0d41e8b771", "state": "queued", "poll": "3f2a9c0d41e8b771"}}
```

`library` is either `{"path": "..."}` (a `.smi`/`.sdf`) or
`{"smiles": ["c1ccc(cc1)C(=N)N", ...], "names": [...]}` — an inline set is written
into the job's own directory, because the job *is* its directory.

**status** — polling is the design, not a workaround: a campaign takes hours.

```json
{"job": {"id": "3f2a9c0d41e8b771", "state": "running",
         "progress": {"completed": 12, "failed": 0, "rows": 12, "interrupted": false},
         "outdir": "out/service/3f2a9c0d41e8b771", "error": ""}}
```

**results** — the same rows `screen` wrote to disk, not a copy this layer kept:

```json
{"ok": true, "job": {"state": "completed", "progress": {"completed": 2, "rows": 2}},
 "results": [{"name": "benzamidine", "status": "ok", "affinity": -5.9451,
              "ligand_efficiency": 0.4954, "n_heavy": 9, "key_residues": "GLY216, TRP215",
              "n_interactions": 8, "pose_file": "poses/benzamidine.pdbqt", "error": ""}],
 "summary": {"interrupted": false, "seed": 0}}
```

**cancel** — and this is the point of the whole design: cancellation is the
**existing** path. `screen_ligands(config, stop=event)`'s `stop` *is* the event
`screen._Runner.stop()` sets, so the kernel notices at its next Monte-Carlo step,
returns the poses it has, and the rows already written stay on disk. There is no
second cancellation mechanism in this layer.

```json
{"ok": true, "job": {"id": "3f2a9c0d41e8b771", "state": "cancelled",
                     "reason": "the caller changed its mind"}}
```

**list** — includes jobs discovered from disk at start-up:

```json
{"ok": true, "jobs": [{"id": "3f2a9c0d41e8b771", "state": "completed",
                       "created_utc": "2026-10-17T09:41:02Z",
                       "outdir": "out/service/3f2a9c0d41e8b771"}]}
```

**HTTP** is the same contract, one route per operation:

```
GET    /v1/schema              PUT-less: the capabilities
POST   /v1/jobs                submit
GET    /v1/jobs                list
GET    /v1/jobs/<id>           status
GET    /v1/jobs/<id>/results   results
DELETE /v1/jobs/<id>           cancel
```

The version travels in the body *and* in the `X-ODock-Schema-Version` response
header, so a client can check it before parsing anything.

## Versioning

The rule is the project container's rule: a request that names a version this build
does not speak is **refused, with both numbers named** — *"this service speaks
odock-service schema version 1, and the request names 2"* — and a request that names
no version at all is refused the same way. A caller never has to parse a log line to
discover it was talking to the wrong build.

`schema` is the one exception, deliberately: it answers **any** caller with the
version this build speaks, because a client that guessed wrong has to be able to
learn the right number. Everything that does work refuses.

Breaking a field's meaning is a new schema version. Adding an operation is also a
new version; adding a *protocol field* is not (the schema endpoint lists the ones
this build accepts, and an unknown one is refused by name before any work starts).

## Security — read this before exposing anything

**What it is:** a loopback HTTP endpoint with a shared token. **What it is not: a
hardened multi-tenant server.** There is no user model, no per-caller quota, no
audit trail beyond the campaign directories, no TLS, and no isolation between
callers. If you need those, put it behind a real gateway that terminates TLS and
authenticates users — do not invent them here.

The controls, and why each exists:

* **Loopback by default.** The service binds `127.0.0.1` and refuses to *answer* a
  request whose `Host` is not a loopback name. A non-loopback bind requires
  `--allow-remote` **and** a token; without both it refuses to start, because an
  unauthenticated docking service on a network interface is an open door with a
  queue in front of it.
* **A shared token, checked on every request** (`X-ODock-Token`, or `?token=` for a
  client that cannot set headers; `--token` or `ODOCK_SERVICE_TOKEN`). Missing or
  wrong is `401 unauthorised`.
* **Cross-origin requests are refused** (`403`): any `Origin`/`Referer`, and
  `Origin: null` — a sandboxed page — is not a server-side client. This is the
  control that matters most in practice: **a local HTTP endpoint that can run
  arbitrary docking is otherwise drivable by any page a browser loads**, and a page
  can reach `127.0.0.1` without asking.
* **No directory traversal in practice:** a request names paths on the *service's*
  machine, and a job writes only under `out/service/<id>/`. That is a consequence of
  the design, not a sandbox: **the caller can ask the service to read any structure
  the service's user can read**, which is why the token exists.

## What this is not

* **Not a scheduler.** There is no priority, no queue policy, no backpressure, no
  fair sharing: `submit` starts a campaign and returns. Run it under whatever
  supervisor you already have.
* **Not multi-tenant.** One token, one namespace of jobs, one working directory.
  Two callers with the token can see and cancel each other's jobs.
* **Not authenticated beyond a shared token.** No users, no scopes, no refresh, no
  per-caller identity in the results.
* **Not a replacement for `odock release`.** Releases are cut with `odock release`
  and its gates (`docs/RELEASE.md`); nothing here publishes anything.
* **Not a streaming API.** Progress is polled. A streaming variant would be a new
  transport for the same state and is deliberately not here; `progress` is a field
  you read, not a socket you hold.
* **Not a job store.** A job is its campaign directory: `screen`'s rows and summary
  are the state, which is why a restart reattaches by reading them — and why deleting
  the directory deletes the job.

## Restart behaviour

A restart reattaches by reading the campaign directories (`screen`'s
`results.jsonl` and `summary.json`), never a second bookkeeping file. A campaign
that was interrupted — rows on disk, no completed summary — is reported as
`cancelled`: its finished rows are kept, and submitting the same work again resumes
through `ScreenConfig(resume=True)` rather than re-docking what is done.

## Tests

`tests/test_service.py`: the full submit → poll → fetch cycle in process, a cancel
that leaves the completed rows, a restart that reattaches from disk, the unknown
version refused with both numbers, a missing field and an unknown protocol field
refused **before any work starts**, and the security refusals — token, cross-origin,
non-loopback host, and the HTTP adapter's `401`/`403`/`404` with its version header.

```bash
python -m pytest tests/test_service.py -q
```

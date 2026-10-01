# Releasing OpenDocking

```bash
odock release prepare 0.2.2                     # bump every version, patch by default
odock release stage ../opendocking-0.2.2        # build what a clone would contain
odock release check -o out/release-check.json   # every gate, with measured results
odock release notes 0.2.2                       # the release body, from CHANGELOG.md
odock release publish 0.2.2 \
      --check-report out/release-check.json     # commit, tag, push, verify
```

Each sub-action is one of the steps a human performed by hand for 0.2.1, and each
one exists because that step went wrong.  The command **refuses** rather than
guesses, and **verifies** rather than assumes.

---

## The three incidents this command was written from

| incident | what it cost | what now prevents it |
|---|---|---|
| A hand-written robocopy exclude list staged the tree; `/tmp*/` matches **directories only**, so **42 `tmp*` files** reached the release snapshot. | a snapshot nobody could vouch for | `release stage` decides the published set with the repository's own `.gitignore` engine and **refuses while any `tmp*` entry is in the root**, naming them and saying how to clear them |
| `Set-Content -Encoding UTF8` wrote a **byte-order mark** into the release body; GitHub silently rejected the release. | a debug cycle on a release that "just did not appear" | `release notes` writes UTF-8 with no BOM, validates the raw first byte at write time, and the test asserts `body.json[:1] == b"{"` |
| A **stale commit-message file** was reused: the 0.2.1 commit was titled "OpenDocking 0.2.0" until it was rewritten. | a rewritten commit on a tagged release | `release publish` derives the message from the version being released and **refuses a message that names another version** |
| (found by `prepare` on its first run) `crates/dock-py/Cargo.toml` pinned `dock-core = { version = "0.1.0" }` while the workspace was 0.2.1. | a stale pin that would return at every bump | a path pin on a workspace member is part of "every place the version lives"; it is normalised with a note |

---

## `prepare X.Y.Z` — the version rule

**Patch by default.**  A minor release is a deliberate act and needs `--minor`, so
`0.3.0` cannot be reached by a typo in a shell history.  A major bump, a move
sideways or backwards, and a half-made minor (`--minor 0.3.1`) are refused, each
naming the version that would be right.

The version lives in more places than is comfortable, and the tool knows all of
them:

| kind | where | how it is treated |
|---|---|---|
| **authoritative** | `pyproject.toml` `[project] version` | rewritten |
| **authoritative** | `Cargo.toml` `[workspace.package] version` | rewritten |
| **authoritative** | a crate's own `Cargo.toml` `version = "…"` | rewritten — and a crate that *should* inherit but hardcodes a different version is how a manifest drifts |
| **inherited** | `crates/*/Cargo.toml` `version.workspace = true` | verified, not touched |
| **derived** | `dock-core = { version = "…", path = "../dock-core" }` | rewritten with a note: cargo ignores it for a local build, so a lag does not block a release, but leaving it stale is how a release looks inconsistent |

The **authoritative** locations must already agree.  A disagreement is refused
with the file, the field and both values — a release must never be the thing that
discovers the manifests were inconsistent.

## `stage DIR` — the published set

The set is decided by `.gitignore`, using the **same engine** as
`out/verify/release_check.py`: anchored vs non-anchored patterns, a trailing `/`
for directories only, `**` across separators, `!` negation with git's
last-match-wins order, and the rule that an ignored *ancestor* ignores everything
below it.  The engine is ported into the package because `out/` is **not
published** — a release step that imported it would work only on the machine that
has `out/`.

Refusals: stray `tmp*` entries in the root; an empty published set; a required
file (LICENSE, both manifests, a crate manifest, the test conftest, the package
`__init__`, the validation report) that exists but would not be published;
generated artefacts (`.pyd`, `.so`, `.dll`, `.pyc`, `__pycache__`, `target/`,
`out/`) inside the set; a destination that already holds files, unless `--force`
(a stale snapshot mixed files into a release once).

The result reports the file count, the byte count, how many paths `.gitignore`
excluded, and a manifest SHA-256 over `(path, size, sha256)` rows — so two
stagings of the same tree can be compared without trusting either one.

## `check` — the gates

| gate | what it runs | what it catches |
|---|---|---|
| **release content** | the vendored content rules over the published set | a required file a rule excludes; an artefact that would ship; a family (the package, the tests, the demo, the docs) that lost files to a global rule |
| **release_check.py** | `out/verify/release_check.py <root>`, when that tree has it | the original reconciliation script; skipped with a note in a clone, because `out/` is not published |
| **inspect_dist self-test** | `tools/inspect_dist.py --self-test` | the leak rules themselves breaking |
| **doctor --strict** | `odock doctor --strict --json` | a stale compiled kernel, a shadowing distribution, a temp directory that is not writable, a Qt/GL platform that cannot render — the environment the release was cut in |
| **benchmark --check-baseline** | `python -m odock.benchmark --check-baseline` | a scoring regression; it re-docks five systems × three seeds and takes tens of minutes |
| **pytest suite** | `python -m pytest tests -q` | everything else |
| **test order** | `tools/check_test_order.py` — the suite twice, in two file orders | a suite whose result depends on file order is not a gate. It reads three verdicts: *both orders agree and pass*, *reproducibly red* (the same tests fail either way — a property of the tests), and *order-dependent candidate* (the outcome changed with the order) |

The order gate is an **instrument, not a fix**, and it says so: the two orders run
at different times, so a test whose outcome depends on machine state (a GUI test
under load, a timing-sensitive assertion) can change between them without file order
having anything to do with it. Measured here on a 12-file sample: the pair reported
8 forward / 7 reverse failures with one test changed — while repeating the *same*
order twice on a quiet machine gave **1 failure both times, identically**. The
signal is real when the same order repeats identically *and* the pair disagrees;
until then it is a candidate. That is why `release check` also names the third
state explicitly: an order gate that says "both orders agree" while the suite gate
is red is the **flaky signal** — hunt shared state (a module-level cache, a mutated
default, a Qt singleton, an environment write, an unseeded generator) rather than
reordering tests.

`--quick` replaces the benchmark re-docking with the baseline **validation**
(systems, seeds, tolerances) and the suite with `-m "not slow"`, and says so in
the gate detail rather than quietly passing.  `--only GATE` runs one gate;
`-o FILE` writes the JSON report that `publish` requires.

Each gate reports its measured numbers — file counts, the tolerance values,
`passed/failed/skipped`, the timing — and the command exits non-zero if **any**
gate fails.

## `notes X.Y.Z` — the release body

Extracted from `CHANGELOG.md`, with refusals for: no changelog; no version
headings at all; no section for this version (the sections that are present are
listed); an empty section; more than one section for the same version; and a
heading for a version **newer** than the one being released, which usually means
the version is wrong rather than the changelog.

The body is written as JSON (`tag_name`, `name`, `body`, `draft`, `prerelease`,
plus the section's line range) in UTF-8 with **no BOM**, and the first raw byte is
checked before the file is accepted.

## `publish X.Y.Z` — and verifying that it happened

### Three ways to supply the commit

A release is cut from more than one kind of place, so the commit can come from
three sources:

| source | what runs | when to use it |
|---|---|---|
| **default** (a git worktree) | `git status`, commit the version bump, `git tag -a`, then the remote refs through the `gh` API | you are in a clone |
| **`--commit SHA`** | no worktree is touched: the commit must already exist on the remote (checked), then the tag ref, the release and the verification | the commit was built or pushed by another tool |
| **`--api-commit --tree-dir DIR`** | the GitHub **git-data API**: one blob per staged file → a tree on the branch's current tree (with explicit deletions) → a commit → **read the commit back and compare its tree and parent with what was intended** → only then update the branch ref | there is no worktree at all — this project's agent workspace, which is why the 0.2.1 commit had to be built through the API by hand |

The read-back is the whole safety argument for `--api-commit`. A ref update is
destructive and irreversible in practice, so it must not depend on an HTTP 200:
the branch is only moved when the object graph that came back is the one that was
asked for, and the failure message says "nothing has been moved".

`--api-commit` sends every payload (blob bodies are base64 file contents) on
**stdin**, never in argv, because a command line runs out of room long before a
large file does. It reports the deletion count it is asking for, since the tree API
with a `base_tree` only adds and updates — a file that left the staged set has to
be deleted explicitly.

The repository slug comes from `gh repo view`, and when that cannot answer (it
infers the repository from the git **remote**, and a worktree-less directory has
none) it is read from `pyproject.toml`/`Cargo.toml`, which name it.

Measured on this workspace, where `gh` is authenticated and the directory has no
`.git`: `release publish 0.2.1 --api-commit --tree-dir out/project-proof/staged
--dry-run` read the real remote — `main` at `5d74d28717c9`, tree `21175301b69c` —
and planned **238 blobs** from the staged set. The write half of this mode is
exercised against an injected transport in the tests, not against the live API.

### Preconditions, all refused

* a **passing `release check` report from this run** (`--check-report FILE`);
* a clean working tree, or a tree whose only changes are the version files the
  tool itself bumps — anything else is somebody's work in progress and is named;
* a commit message that names **this** version (a stale message file is refused);
* a tag that does not already exist locally, or on the remote in the `--commit`
  and `--api-commit` modes;
* a `CHANGELOG.md` section for the version (the body has to come from somewhere).

In the default mode the steps are: commit the bump with the version-derived
message → create the annotated tag → write the BOM-free body → update `main` and
create the tag ref through the **`gh` API** (this environment cannot `git push`) →
`gh release create --verify-tag`.  In the other two modes the tag is a
**lightweight** ref created through the API, because an annotated tag needs a
worktree to write; the verification checks the target either way.

`--dry-run` validates the changelog section and prints the plan — including the
tag name, the refs that would move, and the body's provenance — without writing,
committing or calling out.  A dry run of the 0.2.2 cut on this tree refuses for
the right reason: there is no `## 0.2.2` section yet.

### A gate that retries is a gate that lies

The suite gate reports a failure; it does not re-run the failing tests and report
the second result. During the 0.2.2 preparation the suite failed twice under load
(14, then 2 tests, every one of which passed alone on a quiet machine), and the
temptation to add a retry is exactly the temptation to stop knowing whether the
release was tested. A release gate that retries is a release gate that lies: run
it on a quiet machine instead, and report the environmental failure and the
load-induced failure separately rather than collapsing them into one verdict.

Then it **re-reads the remote** and reports the comparison, not a success:

| verified | expected |
|---|---|
| the remote `main` head | the local commit |
| the remote commit message | the local message |
| the tag target | the local commit |
| the release tag | `v<version>` |
| the release asset names | the assets requested (sorted) |
| the release file count | how many assets were requested |

A mismatch is a failure with both values shown.  This is the part that turns
"the release command exited 0" into "the release exists and is the one I asked
for".

---

## What this command cannot do

* **It does not decide what to release.** The version rule is mechanical, but
  whether a change is a patch or a minor release is a judgement; `--minor` is how
  you state yours.
* **It does not write the changelog.** It refuses to release without a section,
  which is the part that can be mechanised.
* **It does not fix a dirty tree.** It commits *only* the version bump, and names
  anything else; hiding unrelated changes inside a release commit is how a
  release stops being reviewable.
* **It cannot verify what it did not read.** `publish` checks the branch head, the
  tag, the release and the assets; it does not read the release *page*, the wheel
  contents, or whether the artefacts were built from this commit.
* **`--quick` is a weaker claim.** It validates the recorded baseline instead of
  re-docking, and skips the slow tests; the gate says so, and a release cut on it
  has not run the scoring gate.

## Tests

`tests/test_release.py` builds a miniature repository in a temporary directory and
exercises every refusal and every success path.  Nothing touches the network or
the real repository: the gates and `publish` take an **injected runner**, and a
`FakeRunner` records the commands and answers them — so the tests assert the exact
commands issued (`git tag -a v0.2.2 -m …`, `gh api …/git/refs -f ref=refs/tags/v0.2.2`,
`gh release create v0.2.2 --notes-file …`), the verification that re-reads the
remote, and that a remote which disagrees is reported as a failure.

```bash
python -m pytest tests/test_release.py -q
```

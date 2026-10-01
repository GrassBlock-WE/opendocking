# Contributing to OpenDocking

OpenDocking is a Rust kernel with a Python workbench on top, so "it builds" means
two toolchains and one compiled extension. This file is the shortest path from a
clone to a green test run, and every command below is the command CI runs.

If you only read one section, read [Report a bug](#report-a-bug): the reports that
fixed this project's hardest defects contained an environment, an exact command
and the observed output.

---

## 1. What you need

| Tool | Version | Why |
|---|---|---|
| Rust (stable) | ≥ 1.75 | the kernel and the PyO3 bindings |
| Python | ≥ 3.9 (3.10 or newer recommended) | the workbench, the CLI and the tests |
| `maturin` | ≥ 1.5, < 2.0 | builds and installs the extension |
| `make` | GNU make | runs the same commands CI runs |
| RDKit | ≥ 2022.9 | receptor/ligand preparation and every chemistry test |
| `openpyxl` | any | the `odock report` spreadsheet tests |
| PyQt6 + ModernGL | ≥ 6.5 / ≥ 5.8 | the 3-D workbench (optional, `[gui]`) |
| `ruff` | any recent | the Python linter (`[dev]`) |

A GPU is not required: the CPU path is always available, and the optional `gpu`
feature falls back to it when no Vulkan/Metal/D3D12 adapter exists.

## 2. From clone to green

```bash
python -m venv .venv
# Windows (Git Bash): source .venv/Scripts/activate
# Linux / macOS:      source .venv/bin/activate

make venv        # the virtual environment + maturin
make install     # maturin develop --release --extras=dev
make test        # pytest tests -q -m "not slow" --require-demo
```

Working on the 3-D workbench as well? `make install-gui` adds PyQt6 and ModernGL
(`--extras=dev,gui`), which is what CI installs so the head-less Qt suite runs
there too.

`make` on its own prints every target. The important ones:

| Target | Runs |
|---|---|
| `make test` | `pytest tests -q -m "not slow" --require-demo` — the CI suite |
| `make test-ci` | the same plus `--require-gui`: what CI runs, where the Qt suite must not skip either || `make test-slow` | `pytest tests -q -m slow` — the crystallographic re-dockings |
| `make test-all` | `pytest tests -q` — everything |
| `make bench` | `python -m odock.benchmark --check-baseline` (writes `out/benchmark-results.*`) |
| `make lint` | `ruff check python tests tools examples` + `cargo clippy --workspace --all-targets -- -D warnings` |
| `make fmt` | `ruff format` + `cargo fmt --all` |
| `make rust-test` | `cargo test --workspace` |
| `make rust-gpu` | `cargo test -p dock-core --features gpu -- --nocapture` |
| `make wheel` | `maturin build --release --out dist` + `tools/inspect_dist.py dist` |
| `make sdist` | `maturin sdist --out dist` + `tools/inspect_dist.py dist` |
| `make dist-check` | inspects what is already in `dist/` without rebuilding |
| `make demo` | regenerates `demo/` offline from `tests/data/` |

Always build with `--release`: the kernel is numerically dense and a debug build
is an order of magnitude slower. `make test` uses the release extension that
`make install` put in the virtual environment.

Two of those flags exist because a skip is not a pass. `--require-demo` fails the
run when the generated demo fixtures are missing (a fresh clone has none of them),
and `--require-gui` does the same when PyQt6 or ModernGL is missing, which would
otherwise remove the whole head-less workbench suite — including the layout
assertion — from the run. CI passes both, so neither can disappear quietly.

### The two environment variables that matter

```bash
export QT_QPA_PLATFORM=offscreen    # run the workbench tests head-less
unset  QT_QPA_FONTDIR               # do NOT let Qt pick a font directory
```

`tests/test_sequence.py` asserts that the workbench window's `minimumSizeHint`
fits in 1000 px. Qt's floor depends on the font it loads, so with
`QT_QPA_FONTDIR` set the same commit passes on one machine and fails on another.
An *empty* value is not the same as an unset variable — unset it. The Makefile
does both for you; CI pins them in the workflow (and if that assertion ever fails
on a runner by a small margin, re-calibrate the test deliberately rather than
setting `QT_QPA_FONTDIR`, which would hide a real layout regression behind a font
choice).

### Windows: maturin and `%TEMP%`

`maturin develop` resolves a temporary directory before cargo runs, so a locked,
redirected or sandboxed `%TEMP%` fails the build with

```text
maturin failed: Failed to create temporary directory ... (os error 5)
```

Point the three variables at a directory inside the workspace first:

```powershell
$env:TEMP = "$PWD\.rust-tmp"; $env:TMP = $env:TEMP; $env:TMPDIR = $env:TEMP
make install
```

`.rust-tmp/` is already in `.gitignore`. The same note is in
[`.cargo/config.toml`](.cargo/config.toml).

## 3. The demo fixtures are generated, not committed

`demo/` is produced by `python examples/make_demo.py --fast` from the structures
in `tests/data/` — no network access. `.gitignore` keeps the generated files out
of the repository, so a fresh clone has none of them, and the tests that need
them would **skip**.

A skipped accuracy test is not a passing accuracy test, so both `make test` and
CI regenerate the fixtures when they are missing and then run pytest with
`--require-demo`, which turns "the bundled demo files are missing" into a
failure. If you see

```text
FAILED tests/... - the bundled demo fixtures are missing: run `make demo`
```

that is the guard doing its job.

`demo/library.smi` is the one hand-maintained file in that directory: it is the
screening library used by [`docs/SCREENING.md`](docs/SCREENING.md) and by the
screening tests, and it is explicitly tracked by `.gitignore`.

## 4. Layout

```text
crates/dock-core/     the kernel (pure Rust, #![deny(unsafe_code)])
crates/dock-py/       the PyO3 + NumPy bindings (odock._odock)
python/odock/         the Python package: API, CLI, chemistry, GUI, screening
tests/                the pytest suite (runs against the release extension)
benchmark/            the accuracy baseline and its gate
docs/                 architecture, scoring derivation, user guide, screening
examples/             the demo generator
tools/                maintenance scripts (e.g. tools/inspect_dist.py)
.github/workflows/    CI, the scheduled benchmark and the release wheels
```

## 5. Tests

* `tests/` runs against the **installed** extension, so `make install` first.
* Anything that takes longer than about 20 s belongs behind
  `@pytest.mark.slow`; `make test` skips that marker so the fast feedback loop
  stays fast, and `make test-slow` (and CI, on a schedule) runs it.
* A test that needs the generated demo fixtures must **skip** when they are
  absent (the fixtures are not committed) — and CI runs with `--require-demo` so
  that skip cannot hide a broken accuracy check.
* A regression test for a bug is expected to fail on the code *before* the fix.
  The suite has several that were written from a real failure (a `SIGKILL`
  mid-campaign, a results file from another receptor, a changed seed on resume);
  they run the real command, in a real subprocess where that is what it takes.

## 6. Style

* **Python** — `ruff check python tests tools examples` for the narrow
  bug-finding rule set configured in `pyproject.toml` (`E9`, `F63`, `F7`, `F82`:
  syntax errors and undefined names). The tree has never been linted with a
  broader selection, so broadening it is a deliberate future change, one rule
  family at a time, not a drive-by. `ruff format` is the formatter for **new or
  edited** files; it is not run over the whole tree, for the same reason.
* **Rust** — `cargo clippy --workspace --all-targets -- -D warnings` is a hard
  gate in CI and in `make lint`. The findings that already exist (33 with
  `--all-targets` on rustc 1.98's clippy, 15 of them in the library) are
  acknowledged **by name** in `[workspace.lints.clippy]` in `Cargo.toml`, each
  with a count and a reason; a new warning, or a new occurrence of a different
  lint, fails the build. Fixing those 33 is a deliberate follow-up — most are
  `needless_range_loop` (flat-buffer indexing the numeric kernels read best) and
  `neg_cmp_op_on_partial_ord` (comparisons whose NaN behaviour would have to be
  re-derived term by term to rewrite them safely), which is exactly the kind of
  change that needs its own review.
* `cargo fmt --all` is the formatter, and it is **not** a CI gate: the kernel
  keeps hand-aligned numeric tables (the AD4 parameter table in
  `crates/dock-core/src/atom.rs`) that rustfmt reflows into an unreadable block,
  and the tree differs from rustfmt in 77 hunks across 15 files. Format the code
  you touch; do not reformat the parameter tables.
* Comments explain *why*, not *what*: the existing code is the standard.

Run the hooks before a commit:

```bash
make lint
python -m pre_commit install      # optional, runs the same checks on commit
python -m pre_commit run --all-files
```

## 7. Packaging

```bash
make wheel       # maturin build --release --out dist, then inspect the contents
make sdist       # maturin sdist --out dist, then inspect the contents
python tools/inspect_dist.py --self-test   # prove the leak rules still fire
```

`tools/inspect_dist.py` reads the artefacts (`dist/*.whl` is a zip, `dist/*.tar.gz`
a tar) and fails if the Python package, the compiled extension, the metadata, the
console script, the licence, the documentation or the demo library is missing.

It fails in the other direction too, and that is the half that matters: an internal
document, a build directory, a release snapshot of the repository or stale
bytecode inside an artefact is a **leak**, and a check that only looks for missing
files cannot see one. The deny-list is anchored — a root-level file, a root-level
directory, a suffix, or something only legitimate under `tests/data/` — because a
substring rule once flagged `python/odock/__init__.py` while looking for a root
`odock/` snapshot. `--self-test` proves every rule still fires (a leak-catcher
that silently stopped matching is worse than none), and CI runs it before the
wheel is built.

Two files in the working tree are deliberately **not** publishable — the internal
requirements brief and the internal hand-over notes — and they are named in
`[tool.maturin] exclude` rather than merely omitted from `include`, so a future
broad include cannot resurrect them. Their substance is the published
[`docs/VALIDATION.md`](docs/VALIDATION.md). That property is measured: re-adding
them to `include` while they are excluded still leaves them out of the sdist.

One trap if you touch `exclude`: it is **not** format-scoped. Adding `odock/*` to
keep the root release snapshot out looked safe (all 32 `python/odock/*.py` stayed
in the sdist) and silently shrank the **wheel** from 39 files to 22, because in a
wheel `odock/` *is* the package. The inspect tool caught it; trust the artefact
over the reasoning.

## 8. Report a bug

Open an issue with the **Bug report** template and include:

1. **Environment** — OS and version, `python --version`, `odock info` (it prints
   the kernel version, the force fields and the GPU status), and whether the
   extension came from `maturin develop` or a wheel;
2. **The exact command** — copy-pasteable, including flags;
3. **The observed output** — the whole error, not a paraphrase, plus the exit
   code (`echo $LASTEXITCODE` on Windows, `echo $?` elsewhere);
4. **The input** — the smallest structure or PDBQT that reproduces it, if you can
   share one (`odock fetch 3PTB -o 3PTB.pdb` and `tests/data/` are good starting
   points);
5. **What you expected instead.**

A report of the form "it doesn't work" costs a round trip; "on Windows 11,
`odock screen -r rec.pdbqt -i lib.smi --box box.json -o out` exits 2 with
`results.jsonl belongs to another campaign`, but the directory was new" is a fix
in one step. Several of this project's real defects were found exactly that way.

For a **pull request**, fill in the template: the link to the issue, the command
you ran to verify, and what you measured. If a check could not run (no GPU, no
PyQt6, an operating system you do not have), say so in the description rather
than disabling the check.

## 9. Releases (maintainers)

```bash
# 1. Everything green on a clean checkout
make test && make test-slow && make lint && make rust-test

# 2. The accuracy gate
make bench

# 3. The artefacts, inspected
make wheel && make sdist

# 4. Tag; the Wheels workflow builds Windows, Linux and macOS and smoke-tests
#    the first two from a clean environment.
git tag -a v0.1.0 -m "OpenDocking 0.1.0"
git push origin v0.1.0
```

`maturin publish` is deliberately **not** wired into a workflow: uploading to
PyPI needs a token and a human decision. The Wheels workflow stops at "built,
verified, uploaded as an artefact".

## 10. Licence

OpenDocking is GPL-3.0-or-later. By contributing you agree your work is released
under the same terms. Upstream attribution (AutoDock Vina, AutoDock 4, Meeko) is
in the [README](README.md#upstream-attribution) and must be preserved.

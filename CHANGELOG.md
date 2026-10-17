# Changelog

All notable changes to OpenDocking are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.3.7 — 2026-10-18

Two defects that only the CI matrix could find, because each depends on something a
developer machine does not have: a **different interpreter version**, and a **missing
optional dependency**.

### Fixed

* **`odock standardise --audit` crashed on Python 3.9**, the declared floor.
  `standardise.py` wrote its CSV with

  ```python
  path.write_text(self.to_csv(), encoding="utf-8", newline="")
  ```

  and `Path.write_text(..., newline=...)` is **Python 3.10+**, so the CSV was never
  written there: `TypeError: write_text() got an unexpected keyword argument 'newline'`.
  The local interpreter is 3.10, which is exactly why no developer machine could see it —
  the 3.9 leg of the CI matrix is the only place `requires-python = ">=3.9"` is actually
  exercised. The file is opened explicitly now, which is the pattern the rest of the
  codebase already used for `newline=""`.

* **The `analysis` extra did not exist, but the error message told users to install it.**
  `cavity.py` raises

  > the cavity analysis needs scipy ... `pip install scipy` (or
  > `pip install opendocking[analysis]`)

  and `pyproject.toml` declared no `analysis` extra at all — so the one instruction a
  user got for fixing the ImportError was to install a name that is not in the metadata.
  The eleven cavity tests failed on every CI leg with that same message, which is how it
  surfaced: **the advice had never been executable.** `analysis = ["scipy>=1.10"]` is now
  declared, and the CI installs `--extras=dev,gui,analysis`.

### Added

* **`tests/test_python_floor.py`** — the declared floor has to be true in the code, not
  only in `pyproject.toml`. Four static checks: no shipped source calls
  `write_text(..., newline=)` (3.10+) or `read_text(..., newline=)` (3.13+), the floor is
  still `>=3.9` so relaxing it forces a re-read of the traps, every shipped source parses,
  and **every extra named in an error message is actually declared**. The first check is
  parsed with `ast` rather than searched with a regex, because the comment documenting the
  trap quotes the offending call and a text search flags its own explanation — which it
  did, on the first attempt.

### Verified

`test_python_floor` + `test_standardise` + `test_cavity` + `test_cli_surface` →
**122 passed**; the AST check was shown to catch an injected offender; full suite in a
clean environment → **2205 passed, 1 skipped, 0 failed**.

### Still open

The Windows legs still fail a `DID NOT WARN` in `test_read_sdf_skips_a_broken_record_with_a_warning`,
`found.shape >= measured - 0.02` in the overlay test, one numeric comparison that came out
equal where the test wants strictly smaller, and one count assertion; some viewport tests
still reach a `None` renderer through paths other than the guarded assertions
(`pick_atom`, `receptor_count`, `render_image`). The Linux legs are still blocked by
`tests/test_project.py::reproducible_project` (one distinct pose from `exhaustiveness=8,
seed=42` on Linux against at least two on Windows), which needs a Linux machine to
diagnose.

## 0.3.6 — 2026-10-18

### Fixed

* **A cancellation test asserted wall clock where the contract is work.**
  `search::tests::a_cancelled_run_returns_promptly_with_a_valid_pose` demanded
  `latency < 100 ms`, but one evaluation sleeps 200 µs *by construction* — so on a loaded
  runner the scheduler can stretch a single step far past 100 ms and the test fails for a
  reason that has nothing to do with cancellation. Measured on the CI ubuntu runner, where
  it was the only Rust failure.

  The assertion is now what the documentation actually promises: the search notices the
  request **at its next Monte-Carlo step**, so the number that must be small is the
  **evaluation count afterwards**, not the milliseconds. The bound is derived from the
  per-step work (`exhaustiveness = 2` workers each finishing a `local_steps = 20`
  minimisation, plus margin → 64), against a full run of ~2 × 400 × 21 evaluations, so it
  is two orders of magnitude short of running to completion. The remaining time bound is
  a documented **hang detector** (10 s), not a latency contract.

* **Three more viewport assertions were capability-critical.** `0.3.5` converted the nine
  that carried a `no GL context:` message; these three read
  `assert window.viewport.renderer is not None` (and one `assert renderer is not None`)
  and were missed by the exact-match replacement, so the windows-latest legs still failed
  on them. All twelve now go through `_require_gl()`, which skips with the driver's own
  error text and still escalates to a failure under `--require-gui`.

### Verified

`cargo test -p dock-core --lib` → **116 passed, 0 failed**;
`cargo clippy -p dock-core --all-targets -- -D warnings` → clean;
`test_gui_viewport` and the five GUI-adjacent modules → **256 passed, 1 skipped**; full
Python suite in a clean environment → **2201 passed, 1 skipped, 0 failed**.

### Still open

The Windows legs still fail `cmd_standardise` returning non-zero,
`found.shape >= measured - 0.02` in the overlay test, and one count assertion. The Linux
legs are still blocked by `tests/test_project.py::reproducible_project` (a single distinct
pose from `exhaustiveness=8, seed=42` on Linux against at least two on Windows), which
needs a Linux machine to diagnose — changing the assertion from here would be a guess.
Both are named in the CI log rather than absorbed into it.

## 0.3.5 — 2026-10-17

### Fixed

* **A missing OpenGL context is a capability, not a defect — and the tests now say so.**
  Nine assertions in `tests/test_gui_viewport.py` read

  ```python
  assert renderer is not None, f"no GL context: {viewport._gl_error}"
  ```

  which turned "this runner has no GL driver" into six failures per CI leg — measured on
  the windows-latest runner, where every viewport test failed with
  `wglCreateContextAttribsARB not found`. `pytest.importorskip("moderngl")` cannot catch
  it: the module imports fine and it is the *context* that cannot be created.

  They are now a single `_require_gl()` helper that **skips** with the driver's own error
  text. The skip message names `moderngl`/`PyQt6`, so `--require-gui` still escalates it
  to a failure — on a machine that declared the GUI extra a missing context *is* the
  defect, and that flag exists to surface it. This is the same distinction the rest of the
  suite already makes: a skip is a statement about the machine, a failure is a statement
  about the code, and conflating them is how a capability limit becomes an alarm nobody
  reads.

### Verified

`test_gui_viewport` + `test_viewport_interactions` + `test_sequence` + `test_dashboard` +
`test_output_encoding` + `test_cli_surface` → **256 passed, 1 skipped**. The change is
confined to one test module, so the Python package is untouched.

### Still open

The Windows legs still have failures that are **not** the GL class:
`test_the_command_writes_the_library_and_the_audit` (`cmd_standardise(args)` returns
non-zero), `test_the_overlay_enriches_the_amidine_series_in_the_bundled_library`
(`found.shape >= measured - 0.02`), one count assertion, and
`cargo test -p dock-core --lib` on the Windows runner. The Linux legs are still blocked by
`tests/test_project.py::reproducible_project` (a single distinct pose from
`exhaustiveness=8, seed=42` on Linux against at least two on Windows), which needs a Linux
machine to diagnose. Each is named in the CI log rather than absorbed.

## 0.3.4 — 2026-10-17

The three blockers the CI matrix exposed once 0.3.3 put the tests on PATH — and all three
were invisible from a developer machine, for the same reason: **they only appear in a
clean checkout or on a runner**.

### Fixed

* **Two documentation links pointed at unpublished files.** `CONTRIBUTING.md` linked
  `.cargo/config.toml` and `docs/VISUALIZATION.md` embedded four figures from
  `out/surface/`. Both exist on a working machine and neither is published — `.cargo/`
  holds local build settings and `out/` is generated scratch — so the links resolved
  locally and were dead in every clone. They are prose now, naming the file and the
  command that produces it instead of linking to something a reader will not have. The
  documentation gate is unchanged and still checks **117** local links; what changed is
  that the published set no longer contains links to files that are not in it.

* **`--require-gui` cannot be satisfied on a Windows runner.** Every viewport test failed
  with

  ```
  no GL context: wglCreateContextAttribsARB not found
  ```

  which turned a capability limit into six failures per leg. The flag is now passed only
  on Linux, where Mesa's software path provides a context; on Windows the workbench tests
  skip instead of failing, and the step prints why. This is the one place a requirement
  is genuinely platform-dependent, and stating it is better than a number nobody can act
  on — the alternative, dropping `--require-gui` everywhere, would let a missing Qt/GL
  dependency turn the workbench tests into silent skips, which is precisely what the flag
  exists to prevent.

### Verified

`tests/test_docs.py` + `tests/test_docs_site.py` + `tests/test_cli_surface.py` →
**93 passed**; `odock docs check` exit 0; both workflows parse and the `RUNNER_OS` split
is present with `--require-gui` retained on Linux. This patch changes **only** markdown
and workflow YAML, so the Python suite is unaffected by it — the last clean-environment
full run is the one reported with 0.3.3 (**2201 passed, 1 skipped, 0 failed**).

### Still open

The Linux legs still fail `tests/test_project.py::reproducible_project`, where
`exhaustiveness=8`, `seed=42` returns a single distinct pose on Linux against at least
two on Windows. That needs a Linux machine to diagnose; changing the assertion from here
would be a guess. And `cargo test -p dock-core --features gpu` fails on the Windows
runner, which asks for the CPU fallback — also runner-specific and not yet diagnosed.

## 0.3.3 — 2026-10-17

The CI matrix had **never been green** — not on 0.3.2, not on 0.3.1, not on any 0.2.x
commit. Turning it on exposed two long-standing defects and two workflow bugs, and this
patch fixes what could be verified locally.

### Fixed

* **A recorded command kept the other platform's interpreter path.**
  `benchmark.portable_command` normalised a leading interpreter only when it contained
  the *running* platform's separator, so `C:\Users\...\python.exe` survived on Linux
  and `/opt/venv/bin/python` survived on Windows — both of which a baseline is supposed
  to redact, because a baseline is a published document. It now checks **both**
  separators. Found by the CI matrix, which is the only place both platforms meet; the
  existing test already asserted both forms and had been failing on Linux all along.

* **The CI virtual environment was never on `PATH` on Windows.** The workflows wrote
  `"$PWD/.venv/Scripts"` into `GITHUB_PATH`, and under Git Bash `$PWD` is POSIX-style
  even on a Windows runner (`/d/a/repo/repo`); GitHub Actions does not translate those
  entries. The venv therefore never appeared on `PATH`, `maturin develop` and the test
  run used different interpreters, and every Windows leg failed with

  ```
  ModuleNotFoundError: No module named 'numpy'
  ```

  even though `numpy` is a declared runtime dependency and the same tree passed on
  Linux. The path is now produced by **Python** (`os.path.join` with `os.name`), which
  is native on both runners. The same bug in the Wheels workflow made the
  install-and-run job report `ModuleNotFoundError: No module named 'odock'` after a
  successful install.

* **The wheel shipped bytecode.** `[tool.maturin] exclude` filters the **sdist**; a
  wheel copies `python-source` as-is, so every `__pycache__` created while running the
  build job shipped inside it — the wheel asserted to contain no bytecode contained
  `odock/__pycache__/*.cpython-311.pyc` for **every** module. `tools/clean_bytecode.py`
  now removes the residue **before** packaging and does it through `_fs.remove_tree`,
  because a plain `shutil.rmtree` raised `PermissionError: [WinError 5]` on a read-only
  `__pycache__` on the first attempt — the same platform quirk this project already had
  one implementation of.

### Verified

`tests/test_benchmark.py` → **29 passed**; the interpreter normalisation now handles
Windows, POSIX and mixed-separator forms, and a baseline built from a Windows-style
command no longer carries `someone` in it. Full suite and CI status are reported with
the release.

## 0.3.2 — 2026-10-17

**The published source tree could not be built, and all eight CI jobs were red.**
That is the headline, and it is the reason this patch exists.

### Fixed

* **The workspace did not resolve.** `crates/dock-py/Cargo.toml` carried
  `dock-core = { version = "0.2.1", path = "../dock-core" }`. A version requirement on
  an *internal* crate means `^0.2.1` — `>=0.2.1, <0.3.0` — so the moment the workspace
  version crossed 0.3.0, `cargo metadata` rejected the whole thing:

  ```
  error: failed to select a version for the requirement `dock-core = "^0.2.1"`
  candidate versions found which didn't match: 0.3.1
  ```

  Everything downstream failed with it: `maturin develop`, `pip install` from the
  sdist, and **all eight CI jobs** (Rust and Python 3.9/3.10/3.12 across ubuntu and
  windows, plus Wheels and Benchmark) on **both the 0.3.0 and 0.3.1 commits**. The
  local Python suite stayed green throughout, because it runs against a *pre-built*
  extension — which is exactly why no release gate saw it. The dependency is now
  path-only (`dock-core` is never published to a registry, so the path *is* the
  requirement), and a **`cargo metadata` gate** runs in 30 ms before the expensive
  ones, because a release that cannot be built from its own sources is not a release.

* **The output boundary was wrong for every command on a non-UTF-8 console.** `odock`
  prints `Å`, `→`, `—`; on a Chinese Windows the console codepage is GBK and a
  *redirected* stdout is a pipe, so Python encoded with GBK. `Å` (U+00C5) is not in
  GBK, so the command raised `UnicodeEncodeError` — and a parent capturing with
  `encoding="utf-8"` saw **`returncode == 0` and `stdout is None` simultaneously**,
  because failing to decode leaves the attribute unset. That is how the documentation
  test failed for weeks while the same command "worked" by hand. Four modules with
  their own `__main__` block (`tutorial`, `benchmark`, `panel`, `standardise`) bypassed
  `cli._use_utf8_streams()` and now route through it, as do the two release-gate
  scripts `inspect_dist.py` and `check_test_order.py` — a gate that dies of its own
  redirected report fails for a plumbing reason, which is the worst kind. Pinned by
  `tests/test_output_encoding.py`, which forces `PYTHONIOENCODING=gbk` on the child:
  neither CI (a UTF-8 Linux runner) nor a developer shell (usually already set) can see
  this class, so the test has to create it.

* **The documentation gate read chemistry as markup.** A stereo SMILES written as
  `` `C[C@H](F)Cl` `` contains `[C@H](F)`, which the link checker matched as a link to a
  file named `F`, so `docs/STANDARDISE.md` reported **two dead links that are not
  links**. Fenced blocks and inline code spans are now stripped before scanning. The
  check stayed sharp rather than being loosened: it still catches a real dead link and
  still checks **117** local links.

* **The release gate's quick mode could not see a red slow test.** `release check
  --quick` runs `pytest -m "not slow"`, which **deselects** rather than skips — so a
  failing slow test appeared in neither the pass/fail line nor the skip count, which is
  how 0.3.0 shipped six failing tests. `deselected` is now parsed, a quick run declares
  how many tests it never executed, and the full run enforces `EXPECTED_SKIPS` (failing
  when the count moves in *either* direction) plus a 20 s collapse floor, because a
  suite that never ran finishes fast.

* **`odock tutorial` and `odock docs check` crashed** with `PermissionError:
  [WinError 5]` on a stale scratch directory carrying the read-only attribute. `os.walk`
  yields only a directory's *children*, so an empty read-only directory was missed —
  twice: by the original code and by the first attempt at the fix. `python/odock/_fs.py`
  now holds the one implementation (clear the flag on the root **and** every entry, then
  verify the tree is gone) used by `tutorial`, `docs`, `release` and `screen`.

### Added

* **`odock standardise`** — the ordered, switchable recipe that decides whether a
  screening run means anything: parse/refuse, repair, desalt, charge, stereochemistry,
  dedupe (canonical *and* tautomer-insensitive), property filters. The deliverable is an
  **audit**: one row per input record with `in == kept + rejected + merged` asserted
  inside the function (a report that cheats the invariant raises), every removed
  fragment named with its SMILES, heavy atoms and charge, and a recipe hash in the
  header. Determinism is measured: two runs give byte-identical output, and the bundled
  library is left untouched at 17/17 — with the honest line beside it that **7 records
  carry the charges pH 7.4 implies and 10 do not (recorded, not changed)**.
* **`odock panel`** — profile ligands across receptors, with the box sizes reported
  beside the affinities precisely so a "selective" ligand cannot be one that merely had
  a smaller box elsewhere.
* **`odock service`** — a typed, versioned contract (`odock-service`, schema 1) with an
  `http.server` adapter and no framework dependency. Security is enforced in the service
  rather than only in the adapter: loopback `Host` check, a per-request token,
  cross-origin and `Origin: null` refused, and a non-loopback bind requires an explicit
  opt-in *and* a token. The doc says plainly that it is not a hardened multi-tenant
  server.
* **`screen_ligands(config, *, stop=None, on_record=None)`** and
  `_Runner(jobs, stop_event=None)`: a caller-owned cancellation event, so
  `_Runner.stop()` remains the **single** cancellation path — the kernel notices at its
  next Monte-Carlo step, the rows already written stay on disk, and the summary comes
  back interrupted. Two workstreams independently needed it; neither was allowed to
  duplicate the submit/wait loop, and neither had to.
* **`odock covalent`** — warheads, reactive geometry and adducts, with continuity
  *measured* (halving the sample spacing halves the largest step: 2.00 distance, 1.97
  angle) rather than asserted.

### The CLI is now 40 subcommands

`benchmark`, `box`, `cluster`, `conformers`, `covalent`, `decoys`, `diagram`, `diverse`,
`dock`, `docs`, `doctor`, `endpoint`, `ensemble`, `export`, `fetch`, `filter`, `gui`,
`info`, `interactions`, `lbvs`, `panel`, `pharmacophore`, `pocket`, `pocket-score`,
`prepare`, `project`, `protocol`, `release`, `report`, `report-html`, `screen`, `score`,
`service`, `similar`, `split`, `standardise`, `study`, `triage`, `tutorial`.

## 0.3.1 — 2026-10-17

A patch release, and most of it is defects found by running the suite rather than
reading it: 0.3.0 shipped six failing *slow* tests because the release gate's
quick mode **deselects** slow tests, so they appeared in neither the pass/fail
line nor the skip count. That mechanism is now closed.

### Added

* **`odock covalent` — covalent docking**, the capability this toolchain lacked.
  Seven named warheads plus two documented catch-alls, each naming the
  electrophilic atom and its angle window (Michael 70-150 deg at 107, SN2 140-180
  at 180); nucleophile perception by residue/atom name with a geometric fallback
  that says so; classification as **bonded** (<= 2.0 A), **near_attack** (<= 4.0 A)
  or **non_productive** with the failing criterion named. The scoring term is
  continuous and that is measured, not claimed: halving the sample spacing halves
  the largest step (distance ratio **2.00**, angle **1.97**), so a pose cannot be
  rewarded for landing on a threshold. On three poses with identical non-covalent
  affinity the reactive pose moves **rank 3 to 1** while the same-distance,
  wrong-angle pose moves **2 to 3**. Adducts are built and relaxed through the
  project's own minimiser; the measured adduct has C-S **1.820 to 1.825 A** and
  S-C-C **107.0 to 112.2 deg**, both inside the documented ranges. The doc states
  the two limits plainly: strain of 263-389 kcal/mol beside a truncated cysteine is
  a **clash, not chemistry**, and the bundled 1M17 construct has six cysteines
  **none of which is C797**, so perception by name is not identification.
* **Cavity classification in the workbench and in the export** (`View > Surface >
  Cavities`). The surface is repainted by class through a nearest-neighbour query
  (geometry untouched), the legend becomes one swatch per class with pocket count
  and volume, and each bottleneck is a translucent sphere of the **measured** neck
  radius. Measured: 3PTB **9 pockets**, largest neck **3.28 A**; 1M17 **12
  pockets**, **4.48 A**. A through-tunnel that only qualified because
  `min_directions` was lowered is its own **channel** class, so a bore and an ATP
  site cannot look the same. PyMOL and ChimeraX get one marker per aperture at the
  measured radius, a selection of the bottleneck residues, and the criterion in the
  header - including "RELAXED to 4 of six" when it was.
* The release gate now enforces a **skip budget** (`EXPECTED_SKIPS`, failing when the
  count moves in *either* direction) and a **collapse floor** (`MIN_SUITE_SECONDS`,
  20 s, because a suite that never ran finishes fast), and a **quick run declares how
  many tests it never executed**.
* `python/odock/_fs.py` - one implementation of read-only-safe tree removal, used by
  `tutorial`, `docs`, `release` and `screen` instead of four copies of a platform
  quirk.
* Five documents are filed under an audience (`COVALENT`, `COUPLING`, `ENDPOINT`,
  `FRAGMENTS`, `PROTOCOLS`) rather than falling into "More documents".

### Fixed

* **`odock tutorial` and `odock docs check` crashed** with `PermissionError:
  [WinError 5]` on a stale scratch directory carrying the read-only attribute -
  which is exactly what an interrupted run leaves behind. Both failed before doing
  any work, and the traceback named neither the directory nor a remedy. `os.walk`
  also yields only the root's *children*, so an empty read-only directory was
  missed twice: once by the original code and once by the first attempt at the fix.
* **`project reproduce` now PASSES**, and the last defect was a pairing assumption
  in `report._ligand_atoms`: it zipped the ligand PDBQT's atom records with
  `pose.coords` **by index**, on a docstring claim that the orders always match -
  true for an engine-built result, false for an archive rebuilt from a pose file.
  The stored column turned out to be **right** and the fresh one was **dropping real
  contacts**: ASP189 OD1...ligand H17 at **2.10 A** (a hydrogen bond) and VAL213
  CG1...C4 at **3.72 A** (hydrophobic) were both missing from an index-paired
  analysis. Pairing is now by **name**, through one `report.atom_permutation()`, and
  a test reverses `coords` and `ligand_atom_order` together and requires identical
  contacts.
* **The unexplained 0.571 A `top_pose_rmsd` was not non-determinism.** On a re-run
  with bit-identical affinities the archive's `ligand_atom_order` differed from the
  fresh dock's (`C7 C6 C5 C4...` against `C7 C6 C8 C5...`) and the coordinate arrays
  were compared index-by-index; a symmetry-aware RMSD cannot undo an arbitrary
  permutation. After name alignment it is **5.1e-4 A** - the PDBQT's own three-decimal
  quantisation - and the default `--rmsd-tolerance` now inherits that precision,
  because a zero tolerance against a quantised archive cannot be satisfied.
* **`project save --poses` was silently ignored** when `--dock-json` was also given,
  so the archive was built from a source that stores no coordinates - and a PDBQT
  `REMARK VINA RESULT` is *rendered* text at three decimals. The exact JSON values
  now win: `max_affinity_delta` **4.37e-4 to 0.0**.
* **`pdbqt.write_ligand_pdbqt(return_order=True)` reported serial order, not file
  order.** A `BRANCH` record names its parent, so a serial is assigned before that
  atom's line is written; for a thioether the seventh atom in the file is `S` (RDKit
  index 4) while the reported order said `3` (`O C N H C C S H` against
  `O C N C C C S`). The documented `ligand_strain` recipe then raised *"atom order
  does not match the pose PDBQT"* on an ordinary ligand. The writer now records each
  index as it writes the atom, so the order is true **by construction**.
* **`sasa.burial` can no longer be misused the way it was.** `reference_total` for
  `reference="free"` is the **sum of each residue measured in isolation** (224
  residues, **58 377.1 A^2** against an intact receptor's **9 274.9 A^2**, a factor of
  **6.29**), so subtracting it produced a **-357 kcal/mol** nonpolar term. The name
  invited that, so `buried_area` and `buried_fraction` now return **`None`** for that
  reference - a caller gets a `TypeError` instead - `interface_area`'s docstring
  answers "how much area did the partner bury" outright, and it agrees with
  `burial(reference="unbound")` to 1e-12.
* **A document and its test agreed with each other and neither with the run.** The
  overlay docstring said the shape-only ranking puts *one* amidine in the top three
  and the assertion tested for exactly one; the measured count is **two** (against
  three with the electrostatic term). Both now come from the run, and the assertion
  pins the **pair** so a drift says which side moved.
* **Cavity bottleneck spheres were drawn but invisible** inside the opaque surface
  (4 markers, 0 pixels) - now drawn without the depth test, with the state restored
  in a `finally`. The **neck residue list was unbounded**: a 1.3 A neck reported
  **70 residues** straight into an exported selection, now within `neck + 3 A` and
  capped at 12 (measured 5-12).
* The notebook's expected reproduction line still read `tolerance 0 kcal/mol, 0 A`;
  it is now `0.001 A`, the archive precision.

## 0.3.0 — 2026-10-01

The largest step since the first release: five new workflows, a documentation
site, a release tool, and a long list of defects found by measuring rather than
by reading. 1398 -> 1930+ tests.

### Added — new workflows

* **Ensembles.** Dock against several receptor conformations, cluster poses
  across them, and score per-ligand *robustness* — which separates ligands the
  affinity ranking cannot (caffeine 0.814 vs benzamidine 0.004). Cryptic and
  transient pockets are classified across the ensemble with **closure /
  narrowed / detector-disagreement** verdicts, and the detector's own noise
  floor is reported beside every claim: three cavities close when the agonist
  binds (410->4, 352->22, 366->4 A^3, each 5.4-5.7x the local resolution) while
  the trypsin control yields **zero**.
* **One structure, many conformations.** Side-chain rotamers and an
  elastic-network model generate an ensemble from a single crystal. Measured:
  side-chain sampling moves the affinity **0.419** kcal/mol, *less* than the
  **0.558** kcal/mol between two crystal structures of the same protein.
* **Waters.** Networks, conserved/transient/displaced classification, and the
  offline-verified finding that benzamidine's amidine takes the place of a water
  hydrogen bonded to **ASP189**, the S1 specificity residue.
* **Coupling and pathways.** Cross-correlation from the low modes, with the
  distance baseline that makes a coupling readable, and the truncation finding:
  *a specific strong pair is a statement about the mode count; the overall
  pattern is a statement about the structure.*
* **Residue-level windows on the machinery**: the elastic network's low modes
  have site overlaps of -0.002/-0.010/-0.010 with the real antagonist->agonist
  motion, and a generated backbone ensemble produces **no** cryptic candidates —
  *an ANM of one structure is not a substitute for a second crystal structure
  when the difference is a ligand-driven rigid-body displacement of a helix.*

### Added — ligand side

* Pharmacophore models built from **recurring features with per-feature member
  support**, scored with a documented fit and an envelope constraint.
* Property-matched decoys and an enrichment benchmark with **EF1%, EF5%, AUC and
  BEDROC plus bootstrap intervals** and random/property-only controls — and a
  quality report that says when its own decoy set is not matched.
* Structure alerts (PAINS/Brenk/NIH/ZINC plus documented SMARTS), a property
  panel, and a triage table grouped by series, with the alert **rate per
  catalogue** published beside the flags.
* Fragments: screening by ligand efficiency with a heavy-atom floor, growth by
  **incremental LE**, and linking. One positive result stands out — growing from
  a three-atom amidine fragment recovered **benzamidine, the actual 3PTB
  ligand**, at rank 1 — and it ships with two falsified targets and an
  explanation (see below).
* `odock endpoint` — an MM-GBSA-style end-point estimate with a bootstrap
  interval. Against Vina on 17 ligands it is **not significantly correlated**
  (rho +0.297 [-0.278, +0.857]) while Kendall tau excludes perfect agreement, so
  it is a **disagreement detector, not a better ranking**.

### Added — surfaces and interop

* A real solvent-accessible surface, solvent-excluded surface, property
  colouring (Kyte-Doolittle hydropathy, electrostatic potential with a stated
  dielectric), per-residue SASA, and pyMol/ChimeraX/PDB/OBJ export generated
  from the live scene.
* Cavity classification (**enclosed / open / shallow / bulk**) with bottleneck
  radius by bisection, and two deliberate refusals: `openings` is `None` rather
  than a mouth count (a band at the critical radius returned **2 050** clusters
  on trypsin's P1), and a straight through-tunnel is not called a pocket.

### Added — engineering

* `odock docs` — a self-contained documentation site with a search index, a
  rendered-link guard, and an **API reference generated by reading the source
  with `ast`, never by importing it** (a documentation build must not execute
  what it documents).
* `odock tutorial` — the whole toolchain on the bundled data in one command.
* `odock protocol` — versioned, validated, diffable protocols with a hash
  recorded in the run, so a third party can ask "was this produced by that
  protocol?".
* `odock doctor` — installation diagnostics where every finding carries a *why*
  and a *fix*, `--strict` for CI, and probes in **bounded child processes**
  because a diagnostic must not risk what it diagnoses.
* `odock release` — prepare / stage / check / notes / publish, encoding the
  mistakes that were previously made by hand. `release stage` prints the staged
  set **grouped by top-level entry** and `docs/RELEASE.md` makes reading that
  list a required step; that control is what caught the leak below.

### Fixed

* **Interaction lines were swept to the wrong endpoint.** A dash segment is
  `(pos_a, rgba, pos_b, rgba)` — fourteen floats — so the second position starts
  at index **7**; the tube builder read **6**, i.e. `(alpha, y_b, z_b)`. Every
  contact was drawn **14-21 A long** for a 2.8 A contact and read as an
  interaction acting across the protein. The reported distances were always
  correct, which is why checking the interaction *rows* could not see it:
  **verifying the data is not verifying the pixels.**
* **Hiding the receptor left lines pointing at invisible atoms** (24 046 px of
  ink into empty space -> 0); the draw pass and the interaction-focus mesh now
  respect visibility.
* **The search box was the same colour as a hydrogen bond** — RGB distance
  **0.112**, indistinguishable by eye. It is now teal, >= 0.415 from every
  interaction colour, and every dash carries a dark under-stroke plus an
  endpoint marker.
* **Four Chrome profile directories reached the published repository**
  (`odock-chrome-*`), left by the Chrome-based PDF attempt when the temp
  directory was unwritable. They are removed, the whole family is denied in
  every rule that decides what ships, the source now **refuses** to write a
  profile next to the source rather than cleaning up after itself, and the
  `vina.exe` absence check — which had silently become a **no-op** because its
  regex matched one path segment — matches at any depth again.
* **`prepare --no-hetero` crashed** with `AttributeError:
  'AtomPDBResidueInfo' object has no attribute 'GetIsStandardResidue'` — an
  accessor that does not exist in RDKit 2026.03.1, while the sibling helper in
  the same file already documented the gap and tested by name. Chain
  modifications are now **kept and named** rather than dropped: a HETATM flag
  does not make a residue a ligand, and a silently wrong receptor is worse than
  a flag whose help text needs one clause.
* **AD4 produced NaN affinities** (128 non-finite Gasteiger charges), the
  **prepared ligand lost every X-H bond** on an RDKit round trip, **ligand
  strain could be silently wrong** (68.6 vs 2.5 kcal/mol, with coordinates
  matched to atoms **by file order**), and **a restart after a real process
  death re-docked the whole library** because the campaign manifest was written
  only at the end.
* **A symmetric core was ring-shifted** during pharmacophore alignment, costing
  para analogues both donor features; matches are now enumerated and the lowest
  RMSD kept.
* **Self-similarity leaked into the ligand-based benchmark** — with the actives
  as their own queries every method scored AUC and BEDROC 1.000 for free;
  leave-one-out is now the API default.
* **The published wheel metadata pointed at the wrong repository**, the wheel
  shipped 32 `__pycache__` files, the sdist included two internal documents, and
  `cargo fmt` failed outright because an alias shadowed `cargo-fmt`.
* The window could not be made narrower than **1134 px**, and a pre-existing
  unwrapped label pushed it to **1654 px** with poses loaded; both floors are gone
  (932x593) and the guard now asserts the mechanism rather than the number.

### Documented negative results

Six corrections in this release changed a number without improving the result,
and they are grouped in `docs/FRAGMENTS.md` because the shape repeats: **a
correct observation of a number, an incorrect expectation about a result.** The
defence, now written into the project's practice, is **to name the quantity a
change is supposed to move before making it**. Three of the six: the ESP term
saturating and moving a fragment from 6.87 to **20.61 A**; the same correction
raising a table's top ligand efficiency from 0.618 to **1.243** while the method
did not improve; and two penalties pinned at maximum for every atom of every
pose, so they cancelled and ordered candidates **identically**.

Two further results are recorded as falsified targets rather than successes: the
fragment placement does not land where the ligand's substructures sit, and the
linker enumerator correctly refuses to span a gap created by that failure. And
`docs/FRAGMENTS.md` §1.2 states the physical reason a pure desolvation term
cannot be an objective at all: **desolvation is a counterweight to the
interaction that burial buys, so maximising it alone means maximising solvent
exposure.**

### Known limitations

* `odock project reproduce` reports INCONCLUSIVE rather than PASS, because the
  archive stores affinities rounded to three decimals while the comparison
  defaults to zero tolerance. It is a **false negative** — it never claims a
  success it cannot verify — and the fix (store the unrounded value) is designed
  and recorded.
* The suite gate does not yet fail when the **skip count shifts**, so a moved
  dataset that turns assertions into skips can look green. The mechanism
  (`EXPECTED_SKIPS` plus a runtime floor) is designed and recorded.
* `odock.fragments` has no CLI entry: its placement is **not trustworthy at 3-6
  heavy atoms** in either regime, and §4 of its page is the case against
  exposing it.

## 0.2.2 — 2026-10-01

A patch release whose main purpose is to remove files that should never have
been published.

### Fixed

* **Interaction lines were drawn to the wrong endpoint.** A dash segment is
  `(pos_a, rgba, pos_b, rgba)` — fourteen floats — so the second position starts
  at index **7**; the tube builder read index **6**, i.e. `(alpha, y_b, z_b)`.
  Every contact was therefore swept from a real atom to a nonsense point, so a
  2.8 Å contact was drawn **14–21 Å long** and read as an interaction acting
  across the protein. Every swept tube is now at most `DASH_LENGTH` (0.45 Å) and
  lies on the line between the two named atoms to within 0.05 Å. The reported
  distances were always correct, which is why an earlier check of the
  interaction *rows* could not see this: **verifying the data is not verifying
  the pixels.**
* **Hiding the receptor left lines pointing at nothing.** `_draw_interactions`
  read `scene.receptor` without consulting `show_receptor`, and the
  interaction-focus pass drew its ball-and-stick mesh without the visibility
  check its sphere half already had. Both now respect it: a hidden receptor
  draws no contacts and no emphasis (24 046 px → 0 px).
* **The search box was the same colour as a hydrogen bond** — RGB distance
  **0.112**, indistinguishable by eye. The box is now teal (0.15, 0.55, 0.55),
  ≥ 0.415 from every interaction colour.
* **Four Chrome profile directories reached the published repository**
  (`odock-chrome-*`, each with `metadata`, `settings.dat`,
  `CrashpadMetrics-active.pma`). They were left behind by the Chrome-based PDF
  attempt, which correctly reported that no PDF was produced but wrote its
  profile into the working directory — because the environment temp directory is
  not writable, so it fell back to the checkout. They are deleted here and the
  family is now denied in every rule that decides what ships.
* Contact dashes are drawn longer (33 segments → 20 for the same five contacts)
  and every interaction now has an endpoint marker at each atom it names, so a
  line visibly lands on something instead of disappearing behind the ribbon.

### Added

* `odock docs` — a self-contained documentation site built from the published
  markdown, with a search index, a rendered-link guard, and a generator that
  reads the source with `ast` rather than importing it.
* `odock tutorial` — the whole toolchain on the bundled data in one runnable
  command, with the numbers it measured printed inline.
* `odock release` — prepare / stage / check / notes / publish, encoding the
  failures that were previously made by hand.

### Changed

* Every new documentation page is part of the site's set-equality check, so a
  module or document cannot be missing from it silently.
## 0.2.1 — 2026-10-01

The ligand side of the science, ensembles instead of one rigid structure, runs
that can be reopened and reproduced, and a CLI that cannot silently lose a
command. 1063 → 1398 tests.

### Added — ligand cheminformatics

* `odock.ligandsim` — six fingerprint kinds behind one RDKit-free type
  (Morgan/ECFP, RDKit path, atom pair, torsion, MACCS, 3-D pharmacophore) with
  Tanimoto/Dice/Tversky, a blocked similarity matrix, analogue search, and
  best-over-conformers 3-D search that reports its cost.
* `odock.scaffold` — Murcko and generic skeletons, scaffold groups and series,
  an MCS core, **R-group decomposition** into positional `R1..Rk` with
  unmatched rows reported, and **matched molecular pairs** with affinity deltas.
* Library diversity: MaxMin and sphere exclusion, Butina clustering (checked
  against RDKit's own implementation), and `odock screen --diverse N`.
* `odock.pharmacophore` — models built from **recurring features with per-feature
  member support**, scored against a library with a documented fit and an
  envelope constraint. Measured on the five ring-amidines: four features, every
  one supported 5/5, sweeping ranks 1–5 at 0.918–0.944 against a best decoy of
  0.497; warfarin is rejected with 18 of its 23 heavy atoms outside the envelope.
* `odock.decoys` + `odock.lbvs` — property-matched decoy generation with a
  Tanimoto ceiling, and an enrichment benchmark reporting **EF1 %, EF5 %, AUC and
  BEDROC with bootstrap intervals** alongside random and property-only controls.
  The quality report says when its own decoy set is not matched rather than
  hiding it.

### Added — ensembles and cryptic pockets

* `odock.ensemble` — several receptor conformations validated as the same
  protein (sequence identity, site superposition, a named refusal when they are
  not) and docked against a shared box, with **pose clustering across
  conformations** and a per-ligand **robustness score**. Measured: on ERα
  (3ERT/1ERE_A) the winning conformation is the receptor rather than the seed,
  and equal-affinity ligands separate on robustness (caffeine 0.814 vs
  benzamidine 0.004).
* `odock.pockets` across an ensemble — per-conformation cavities mapped into a
  common frame and tracked across structures, with **closure / narrowed /
  detector disagreement** verdicts and the detector's own noise floor reported
  beside every claim. Measured: three cavities close when the agonist binds
  (410→4, 352→22, 366→4 Å³, each 5.4–5.7× the local resolution) while the
  trypsin control yields **zero** candidates (largest apparent change 40 Å³
  against a 39 Å³ resolution).
* `odock ensemble {align,dock,screen,pockets}`; a ligand-free conformation is
  refused for box derivation with the reason named.

### Added — reproducible, presentable runs

* `odock.project` — a self-contained `.odockproj` (inputs with SHA-256, every
  setting, the seed, the poses, a schema version), with save / open / verify /
  info / extract. A flipped byte is caught and named.
* `odock project reproduce` — re-runs the docking from the recorded inputs and
  reports **PASS / FAIL / INCONCLUSIVE** with the numbers; the default tolerance
  is **exactly zero** because the engine is deterministic (measured
  `max |Δaffinity| = 0.000`, `top-pose RMSD = 0.000 Å`, two reproductions
  byte-identical).
* `odock project compare` and `odock project index` — several runs side by side
  with the differing settings named field by field, and a campaign index page.
* `odock report-html` — a single self-contained HTML report (inline SVG, base64
  figures, zero external references) with an explicit "what this run does not
  establish" section.

### Fixed

* **`odock <cmd> --help` crashed for every subcommand.** argparse `%`-formats
  help strings, and a new command's one-line help contained a literal `EF1%`.
  Every `format_help` raised `ValueError: unsupported format character ','`,
  which also broke `odock` with no arguments and `odock --help`.
* **A symmetric core was ring-shifted during pharmacophore alignment.** Taking
  RDKit's *first* core match left para analogues' amidines on the wrong side,
  costing them both donor features (fit 0.00, ranks 16–17 beside 0.94
  analogues); matches are now enumerated and the lowest RMSD kept.
* **Self-similarity leaked into the ligand-based benchmark.** With the actives
  as their own queries all three methods scored AUC and BEDROC 1.000 for free;
  leave-one-out is now the API default for fingerprint, pharmacophore and shape.
* **A verdict rule fired on its own counterexample** in the pocket tracker: a
  track absent where the free volume was *higher* is detector disagreement, not
  a cryptic site, and is now excluded from the candidate list.
* A test pinned a **suite count** rather than the shape of the result, so the
  documentation guard broke whenever any test landed anywhere.
* `docs/VALIDATION.md` and the README were brought back in step with the tree,
  and internal-document citations were removed from the shipped sources at the
  source rather than only in the staged copy.

### Added — engineering

* `tests/test_cli_surface.py` — the complete expected subcommand set asserted in
  **both** directions, that building the parser emits no registration warning,
  that every registrar ran, and that every command has a help page. It exists
  because a cleanup script once deleted a region of `cli.py` and took two other
  workstreams' registration calls with it, dropping six subcommands while the
  suite stayed green.
* `python/odock/cli_ext.py` — one extension entry point, so contributed
  subcommands register in their own modules and `cli.py` is edited once.

## 0.2.0 — 2026-10-01

Screening at library scale, the numbers needed to judge a pose, a real
molecular surface, a workbench that behaves like an instrument, and the
engineering that makes it reproducible by someone else. 605 → 1063 tests.

### Added — reaching a conclusion

* `odock.consensus` — rescore poses with **Vina, Vinardo and AD4** at the run's
  own coordinates and combine them by **rank aggregation** (`rank`/`borda`/`z`),
  with per-field scores and ranks and **pairwise Spearman** agreement. The force
  fields genuinely disagree: mean rho 0.96 on 3PTB but **0.68 on 1M17**.
* `odock.metrics` — ligand efficiency (LE, LLE, BEI, SEI), a torsion-entropy
  estimate, MMFF94 ligand strain, and `StrainResult.reliable` decided from
  evidence (`require_reliable=True` refuses a number that cannot be trusted).
* `odock.analysis` — interaction **fingerprints** over (residue, type) pairs
  with a similarity matrix and a recurring-feature summary, and **water-mediated
  contacts**.
* `odock.report` — consensus/metrics/strain columns, fingerprint and
  pharmacophore sheets, a notebook-ready summary, streaming JSONL/CSV writers.

### Added — `odock screen`

* Library screening against one receptor or a **panel** of receptors.
* **Resumable by construction**: results flushed per molecule, and the campaign
  manifest written *before* the first docking, so a killed process resumes
  instead of re-docking; a file belonging to another campaign is refused with a
  named reason rather than silently appended to.
* Funnel with per-filter removal counts and first-failure attribution;
  `--dry-run` cost extrapolation; `--consensus` rescoring that can be added to
  a finished campaign without repeating any docking.
* One bad molecule is one `failed` row, not a dead run; Ctrl-C keeps completed
  work on disk.

### Added — seeing the chemistry

* `odock.sasa` — Shrake–Rupley SASA per atom and residue, burial against a
  stated reference, interface area, and the ligand's **buried contact area**
  (benzamidine hides 267 Å² = 94 % of itself).
* `odock.gui.surface` — a real **solvent-accessible surface** from the
  probe-centre field via marching tetrahedra, a solvent-excluded surface from
  the same field, coloured by **Kyte–Doolittle hydropathy** or by the
  **electrostatic potential** sampled from the project's own charges, with
  legend, colour-range control, pocket-lining mode and a clipping plane.
* `odock.interop` — export the live scene to **PyMOL** `.pml`, **ChimeraX**
  `.cxc`, a pose **PDB** carrying the per-atom property in the B-factor column,
  and a surface **OBJ**.

### Added — the workbench

* A **run monitor** (real per-pose energy trace, phase chips, elapsed clock),
  **pose comparison** (symmetry-aware RMSD, contact-fingerprint diff, copyable
  report), a **Ctrl+K command palette** generated from the menu bar, **session
  persistence** with recent files, a **light theme** with a compact density,
  drag-and-drop loading routed by file content, layout presets, an atom-inspect
  card, a measurement history, copy-view-to-clipboard, and pose-contact marks
  on the sequence ruler.

### Added — proof and engineering

* `odock.benchmark` — five bundled complexes, several seeds, reporting top-pose
  RMSD, `best ≤1 kcal`, the crystal mode's rank, and force-field agreement each
  with its sample size and seed spread. Every seed of every system finds the
  crystal mode; on 1M17 the top pose moves 3.2 Å between seeds while
  `best ≤1 kcal` stays at 1.28 Å.
* `benchmark/baseline.json` + `--check-baseline`, a regression gate whose
  tolerance comes from measured repeat-run noise — not the seed spread, which is
  a property of the system (reasoning recorded in the file).
* GitHub Actions: CI (Linux + Windows × Python 3.9/3.10/3.12, with
  anti-silent-skip guards for the demo data and the GUI), a scheduled benchmark
  gate, wheel/sdist builds with a clean-venv install smoke test; `Makefile`
  targets identical to CI; issue/PR templates; `tools/inspect_dist.py`, which
  fails when an artefact contains anything internal and self-tests its rules.
* New docs: `BENCHMARK.md`, `SCIENCE.md`, `SCREENING.md`, `VISUALIZATION.md`,
  plus refreshed architecture, scoring, validation and user guides.

### Fixed

* **AD4 produced NaN affinities**: 128 Gasteiger charges were not finite (8
  written literally as `inf`) and the writer sanitised only NaN. Non-finite
  charges are now zeroed **and reported**.
* **Prepared ligands lost every X–H bond** on an RDKit round trip because the
  added hydrogens carried a different residue name from the heavy atoms — the
  reason MMFF94 could not type the demo ligand.
* **A restart after a real process death re-docked the whole library** and
  appended duplicates, because the manifest was written only at the end; and a
  **changed `--seed` on resume silently mixed two campaigns**.
* **Ligand strain could be silently wrong**: the documented escape hatch did not
  help (68.6 → 60.8 kcal/mol) and relaxed coordinates were matched to atoms by
  file order. The element sequence is now checked and a mismatch raises.
* **The published wheel metadata pointed at the wrong repository**, the wheel
  shipped 32 `__pycache__` files, and the sdist included two internal documents.
* **`cargo fmt` failed outright** (an alias shadowed `cargo-fmt`), and 33 clippy
  findings made `-D warnings` unpassable — now acknowledged by name so the gate
  catches anything new.
* The window could not be made narrower than **1134 px**, and a pre-existing
  unwrapped label pushed it to **1654 px** with poses loaded; both floors are
  gone (932 × 593) and the guard now asserts the mechanism.
* The run monitor attributed a real duration to the wrong phase while running;
  `make_demo.py` and two scripts crashed on a GBK console printing Å;
  `fetch_ligand_sdf` 404'd after RCSB moved to `_ideal.sdf`; the light theme
  washed out the 3-D legend ink; the measurement hint was hidden behind the
  floating tool strip.

## 0.1.0 — 2026-09-30

The first public release: a complete AutoDock-family docking toolchain built on
a pure-Rust kernel, with a Python API, a command line and a 3-D workbench.

### Kernel (`dock-core`, `dock-py`)

* Rigid-cluster / torsion-tree kinematics with forward kinematics and a
  projected degree-of-freedom gradient.
* Three force fields — AutoDock **Vina** (default), **Vinardo** and **AutoDock
  4.2** — with an analytic first derivative for every term, each verified
  against central finite differences.
* Trilinearly interpolated affinity grids with an analytic gradient and an
  out-of-box penalty; every reported pose is refined on the exact pairwise
  surface.
* Three searches: Metropolis Monte-Carlo iterated local search, the island-model
  Lamarckian GA, and the LGA with a **Solis-Wets** adaptive local search.
* `rayon` data parallelism, deterministic seeding, and pause / resume / abort
  (`Docking.pause`, `resume`, `cancel`, `is_running`).
* Batch docking of independent ligands from released-GIL threads
  (`_odock.dock_batch`).
* Optional **wgpu** compute backend (`--features gpu`) for grid construction,
  batch forward kinematics and batch atom-pair evaluation, numerically equal to
  the CPU path.
* PDBQT reader/writer, including the nested `ROOT`/`BRANCH`/`TORSDOF` topology
  language.

### Python layer

* Receptor preparation: HETATM classification and cleaning, structural-water
  retention by distance, co-crystallised-ligand extraction, cofactor and ion
  policies, missing-atom detection, pH-aware protonation (HID/HIE/HIP),
  polar-hydrogen-only conversion, Gasteiger/Kollman charges, AD4 typing, and
  flexible-residue PDBQT export.
* Ligand preparation: `.smi`/`.sdf`/`.mol`/`.mol2`/`.pdb`/`.pdbqt` input in
  batches, ETKDGv3 embedding, MMFF94/MMFF94s/UFF minimisation,
  rotatable-bond perception with an interactive bond locker, and the torsion
  tree writer.
* Library pre-filters: Lipinski, Veber and PAINS.
* Blind pocket detection with volume, buriedness and residue labelling.
* Analysis: symmetry-aware RMSD, RMSD clustering, the six non-covalent
  interaction types with adjustable cut-offs, 2-D interaction diagrams (SVG),
  pose interpolation, and XLSX/CSV reports.
* Exporters: AutoGrid **GPF**, AutoDock 4 **DPF**, Vina **config**, and cleaned
  **PDB** with a correct `MASTER`/`END` tail; RCSB downloads by PDB ID or
  chemical-component ID.
* CLI: `info`, `prepare receptor|ligand`, `box`, `dock`, `score`, `split`,
  `pocket`, `filter`, `cluster`, `interactions`, `diagram`, `report`, `fetch`,
  `export gpf|dpf|config|pdb`, `gui`; bare `odock` opens the workbench.
* Workbench (PyQt6 + ModernGL): cartoon / ribbon / tube / space-filling /
  ball-and-stick / sticks / dots protein styles and four ligand styles,
  valence-aware bond perception, an interactive draggable grid box, the residue
  ruler with click-to-select, the atom panel, interaction highlighting with
  adjustable cut-offs, pose playback, and an English / Simplified Chinese
  interface.

### Documentation

* `docs/ARCHITECTURE.md`, `docs/DATA_STRUCTURES.md`, `docs/SCORING.md`,
  `docs/USER_GUIDE.md` and `docs/VALIDATION.md`; `demo/` ships two complete,
  regenerable example systems (3PTB and 1M17).

### Known limitations

Recorded in full in [`docs/VALIDATION.md`](docs/VALIDATION.md#7-what-is-not-validated).
In short: the search is rigid-receptor (a flexible-residue PDBQT can be written,
but the kernel's receptor reader folds it back to rigid and says so); the
Kollman charge model is the united-atom scheme, not the AMBER residue tables;
GPU acceleration covers grid construction and batch kernels only; SSAO is a
screen-space approximation; and macrocycle closure sampling is not specially
optimised.

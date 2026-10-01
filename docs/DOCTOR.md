# `odock doctor`

```bash
odock doctor                 # diagnose this installation, with a fix per finding
odock doctor --json          # the same, machine-readable (this is what an issue needs)
odock doctor --strict        # exit non-zero on a warning as well as on an error
odock doctor --self-test     # also run the three real gates, with measured numbers
odock doctor --self-test --full   # ... including the full benchmark re-docking gate
odock doctor --no-qt         # skip the Qt/GL probe (a machine with no display)
```

Every check in this file exists because the corresponding failure cost real time
in this project and none of them announced itself. The doctor's job is to turn
each of those into one line with a severity, a reason and a command.

---

## What it checks, and why each check exists

| check | what it reports | why it exists (the incident) |
|---|---|---|
| **interpreter** | Python version, implementation, executable, prefix, whether it is a virtual environment | a report without the interpreter is not reproducible |
| **compiled kernel vs versions** | the `_odock.__version__`, the installed distribution's version, the checkout's `pyproject.toml`, the kernel manifest | a wheel shipped where the local `.pyd` said **0.1.0** while the tree said **0.2.0**: every version a user read came from the stale extension, and an already-fixed bug looked present |
| **dependencies and extras** | numpy (required) and rdkit, PyQt6, moderngl, openpyxl, scipy, pytest, ruff — each with the extra that provides it and the install command | "it imports on my machine" is a difference in extras, not in code |
| **shadowing** | another `odock` *package* on `sys.path`, a second `opendocking`/`odock` *distribution*, and any `odock` console script on `PATH` that is not this installation's — each with its **resolved path and version** | a completely different package (`_dockpy.pyd`, its own `workbench` module) was installed ahead on `PATH`, and a global `opendocking` sat in another interpreter's site-packages |
| **temp directory** | whether `TMPDIR`/`TEMP`/`TMP` is actually writable, **where `tempfile` really lands**, and how many stray `tmp*` entries are in the working directory | `%TEMP%` was not writable, so `tempfile` fell back to the working directory and left **165 `tmp*` directories and 48 files** in the repository root; a release snapshot swallowed 42 of them |
| **Qt/GL** | whether an offscreen `QApplication` can create a **GL context**, the renderer, and how many font families the platform has | 14 GUI tests failed with `no GL context: CreateWindow failed` under load and the cause was invisible; a platform with no fonts rasterises figure labels as boxes |
| **promised files** | README, LICENSE, the docs set, the release tooling, the benchmark baseline, the bundled demo, `tests/data/3PTB.pdb` | a missing demo file turns the quick start into a first-command failure; in an installed wheel the same check is information, not a warning |

Three design decisions worth knowing when you read the output:

* **The Qt and temp probes run in bounded child processes.** A missing GL driver
  *aborts* rather than raises, and creating a `QApplication` in the caller would
  change the state the doctor is inspecting — a diagnostic must not risk what it
  diagnoses. The temp probe is in a child for a sharper reason: creating a
  temporary file **hangs** where a policy blocks writes instead of refusing them
  (measured here — `tempfile.NamedTemporaryFile` never returned, while a plain
  `open()` failed immediately). A probe that hangs is reported as a finding, with
  a fix.
* **The output names absolute locations on purpose.** Everywhere else in this
  project a recorded path is normalised so it cannot leak, but "another
  distribution shadows this one" is useless without *which one*, and the fix is a
  `pip uninstall` of a specific installation. The doctor never writes a file; its
  output is for the console and for an issue. (The *tree* is still protected: the
  `--self-test` release gate scans the published text for any absolute path that
  exists on this machine — which caught this project's own test files.)
* **A finding that cannot be acted on is information, not a warning.** A missing
  optional extra in a headless install is `info`; a missing required dependency
  is `error`; a stale extension is `warning`. `--strict` turns any warning into a
  non-zero exit so CI can gate on it, and an error always exits non-zero.

### Severities and exit codes

| severity | meaning | exit code (plain / `--strict`) |
|---|---|---|
| `ok` | checked and healthy | 0 / 0 |
| `info` | a fact worth recording, or an optional piece that is absent | 0 / 0 |
| `warning` | something you can and should fix | 0 / 1 |
| `error` | this installation cannot work (the extension is missing) | 1 / 1 |

---

## `--self-test`: the three real gates

These are the "is my installation actually working end to end" answers a bug
report needs, each reported with its measured numbers rather than a verdict:

| gate | what it runs | numbers it reports |
|---|---|---|
| **benchmark baseline** | validates `benchmark/baseline.json`: the recorded systems, seeds per system, the tolerances, and that the recorded command carries no absolute path. With `--full` it runs the real `python -m odock.benchmark --check-baseline`, which re-docks five systems × three seeds and takes tens of minutes | systems, seeds, both tolerances, the recorded command, paths in it |
| **release content** | `tools/inspect_dist.py --self-test` (the leak rules still work) **and** a scan of the published text for an absolute path that exists on this machine | files scanned, absolute paths found, illustrative docstring examples |
| **smoke docking run** | a small real docking run on the bundled 3PTB demo (two Monte-Carlo runs by default) | poses, best affinity, grid points, movable atoms, seconds |

The smoke run is deliberately small. The doctor is a diagnostic, not a
benchmark: `python -m odock.benchmark` is the benchmark.

---

## What the doctor cannot tell you

* **It cannot validate your science.** A green doctor means the plumbing works:
  the kernel imports, the inputs parse, the search runs, the files are where the
  documentation says. It says nothing about whether a force field suits your
  system, whether your box covers the site, whether a pose is right, or whether a
  preparation step did what you meant.
* **It reports *this* machine.** A green doctor on a laptop says nothing about a
  cluster, and a warning about a missing GL context is a fact about the session,
  not about the code (14 GUI failures in this project were load, not a defect).
* **The temp probe can be inconclusive.** If it is killed, the finding says so
  rather than guessing "writable".
* **The release-content gate cannot flag a path that no longer exists.** It asks
  whether the path named in the text *exists on this machine*, which is what makes
  it quiet about illustrative examples (`C:\work\repo\.venv\...` in a docstring)
  without being quiet about a real checkout path. A leaked path to a file that has
  since been deleted is therefore not caught.
* **It does not benchmark, and it does not compare runs.** Use
  `odock project reproduce`, `odock project compare` and the benchmark for that.
* **It does not fix anything.** It prints the command; running it is your call
  (uninstalling a distribution is destructive, and the doctor will not do it for
  you).

## In a bug report

`.github/ISSUE_TEMPLATE/bug_report.yml` asks for `odock doctor --json`. The JSON
carries the interpreter, the platform, the version comparison, the dependency
table, the findings with their fixes and the gate numbers — everything a
maintainer needs to reproduce the situation, in one paste. `odock info` (the
kernel, the force fields, the GPU status) is still wanted as well.

## Tests

`tests/test_doctor.py` asserts the checks and, more importantly, the two
properties that make the output trustworthy: **every actionable finding carries a
fix**, and the exit codes follow the severity table. No test depends on the
machine it runs on — each check takes its inputs as arguments (an environment
mapping, a `sys.path` list, a probe report, explicit version strings), so a
simulated unwritable temp directory, a shadowing distribution or a stale
extension is asserted identically anywhere. The three tests that run real code
use the repository's own fixtures.

```bash
python -m pytest tests/test_doctor.py -q
```

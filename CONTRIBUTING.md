# Contributing to OpenDocking

Thanks for taking the time to look at the project. Bug reports, force-field
checks, documentation fixes and new tests are all welcome. This file covers the
practical part: how to build, how to test, and the conventions the tree follows.

---

## Prerequisites

* **A Rust toolchain** — stable, 1.75 or newer (`rustup` is the easy way).
* **Python 3.9 or newer**, with `pip`.
* **`maturin`** — `pip install maturin`.
* Optional, for the parts of the project that need them:
  * `rdkit` for all receptor/ligand preparation and the chemistry tests;
  * `PyQt6` and `moderngl` for the 3-D workbench;
  * `openpyxl` for XLSX reports;
  * `numpy` (a hard dependency, installed with the package).

> **The Python package cannot be imported before the Rust extension is built.**
> `odock/__init__.py` imports the compiled `odock._odock` module, and that
> `.pyd`/`.so` is a build artefact — it is deliberately *not* in the repository.
> A fresh clone therefore has to run `maturin develop --release` **before** any
> `import odock` or `pytest` invocation works. Running the test suite on a clone
> that has never been built fails with
> `ModuleNotFoundError: No module named 'odock._odock'`.

---

## Building and testing

```bash
# 1. A virtual environment with the Python-side dependencies.
python -m venv .venv
.venv/Scripts/activate                  # Windows
source .venv/bin/activate               # Linux / macOS
pip install maturin rdkit numpy

# 2. Build the Rust extension into that environment. This step is mandatory
#    before anything below; it installs `odock._odock` and the `odock` script.
maturin develop --release

# 3. Optional extras: the workbench and the Excel reports.
pip install PyQt6 moderngl openpyxl
```

```bash
# The Rust test suite (unit tests, integration tests and the doc test).
cargo test --workspace

# The same tests with the optional wgpu backend compiled in.
cargo test -p dock-core --features gpu --lib

# The Python test-suite. The GUI tests run on Qt's `offscreen` platform, so a
# display is not needed.
python -m pytest tests -q

# The slow end-to-end crystallographic validation.
python -m pytest tests -m slow -v
python tests/validate_3ptb.py --exhaustiveness 16 --seed 42

# The workbench driven through a complete user session (48 steps, screenshots
# and an illustrated report under out/).
python tests/simulate_workbench.py
```

On a Windows console whose code page is not UTF-8, prefix the reporting scripts
with `set PYTHONIOENCODING=utf-8` (or `chcp 65001`): they print Å and Å³. The
`odock` command line already reconfigures its own streams.

Every number the documentation quotes was measured with the commands above, and
the measurements are collected in [`docs/VALIDATION.md`](docs/VALIDATION.md).
When you change something that those numbers depend on, re-measure and update
that file in the same change.

---

## Code layout

The tree is described file by file in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and the core data structures are
documented field by field in [`docs/DATA_STRUCTURES.md`](docs/DATA_STRUCTURES.md).
In short:

| where | what goes there |
|---|---|
| `crates/dock-core` | the numerical kernel: typing, kinematics, the three force fields, grids, the searches, PDBQT I/O, orchestration. Pure Rust, `#![deny(unsafe_code)]`. |
| `crates/dock-py` | the PyO3 bindings. No numerics of its own — every entry point forwards to `dock-core`. |
| `python/odock` | the user-facing layer: preparation, the docking API, pockets, analysis, reports, exports, the CLI. |
| `python/odock/chem` | chemical perception with RDKit; no docking engine code. |
| `python/odock/gui` | the PyQt6 + ModernGL workbench. |
| `tests/` | the Python test-suite and the validation scripts. |
| `docs/` | architecture, data structures, the scoring derivation, the user guide, validation. |

---

## Conventions

**Rust.** No `unsafe` (the crates are `#![deny(unsafe_code)]`). Every constant
taken from an upstream implementation is documented at its definition with the
name it has there. Every public item needs a doc comment and a test; analytic
derivatives additionally need a finite-difference check.

**Python.** Type hints and a docstring on every public function and class.
Import optional heavy dependencies *lazily*, inside the function that needs them,
and fail with a message that names the `pip` command that fixes it. NumPy arrays
are `(N, 3)` float64 for coordinates. Keep the CLI thin: a subcommand is a
wrapper around the Python API, never a second implementation of it.

**Tests.** One behaviour per test, named as the sentence it asserts. Tests that
need bundled data skip themselves (`pytest.skip`) when that data is absent
rather than erroring. Tests that download anything must mock the network. Add a
test with every behavioural change; a bug fix without a regression test will be
asked for one in review.

**Documentation.** Markdown, English, and the numbers in it must be reproducible
from a command stated next to them. If you cannot measure a number, do not write
it.

---

## Contributing a change

1. Fork, branch, and keep the change focused.
2. Run the Rust and Python suites, plus the validation script if you touched
   scoring, kinematics, preparation or the search.
3. Update the docs that your change makes stale — including
   [`docs/VALIDATION.md`](docs/VALIDATION.md) and [`CHANGELOG.md`](CHANGELOG.md).
4. Open a pull request describing *what was measured*, not only what changed.

Keep the SPDX header (`SPDX-License-Identifier: GPL-3.0-or-later`) on every new
source file.

### Licensing of contributions

OpenDocking is released under the **GNU General Public License, version 3 or
later**. By submitting a contribution you agree that it is licensed to the
project under the same terms (GPL-3.0-or-later, "inbound = outbound"), and that
you have the right to submit it.

Two rules follow from how this project relates to its ancestors:

* **Do not copy code from AutoDockTools (ADT / MGLTools).** It is deliberately
  out of bounds, and nothing in this tree may depend on it. PDBQT parsing,
  preparation and the torsion tree are written from the format specification and
  RDKit.
* **Do not paste code from a project whose licence is incompatible with the
  GPL.** The freely licensed implementations this project *learns from*
  (AutoDock Vina, Apache-2.0; AutoDock 4, GPL; Meeko, LGPL-2.1; AutoDock-GPU,
  LGPL-2.1) may be read for algorithms, constants and behaviour — attribute what
  you take, in the file that takes it, and write the implementation yourself.

---

## Reporting a bug

Please include: what you ran (the exact command), what you expected, what
happened, the output of `odock info`, your OS and Python version, and — for a
docking problem — the input structures or a minimal reproduction of them. For
anything numerical, state the force field, the box, the seed and the
exhaustiveness; without them a result cannot be reproduced.

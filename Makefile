# OpenDocking — the commands CI runs, for a human.
#
# Every target here is a thin wrapper around one command, and the same command
# appears in .github/workflows/.  If CI fails, `make <target>` reproduces it.
# Nothing in this file is clever on purpose: a contributor should be able to read
# a target and type the command it runs by hand.
#
# GNU make, so it works in Git Bash, WSL, Linux and macOS (on Windows, install
# make with `winget install GnuWin32.Make` or use the Git Bash one).  The virtual
# environment lives in .venv/Scripts on Windows, .venv/bin everywhere else; the
# detection below is the only platform-specific line in the file.

ifeq ($(OS),Windows_NT)
  PYTHON ?= .venv/Scripts/python.exe
else
  PYTHON ?= .venv/bin/python
endif

MATURIN ?= $(PYTHON) -m maturin
PYTEST  ?= $(PYTHON) -m pytest
DIST    ?= dist

# The environment the Python suite needs.  QT_QPA_FONTDIR is deliberately
# *unset*: with it set, Qt picks a different font and the layout assertion in
# tests/test_sequence.py (minimumSizeHint <= 1000 px) fails on a machine that
# otherwise "works".  An empty value is not the same as an unset variable.
export QT_QPA_PLATFORM := offscreen
export PYTHONIOENCODING := utf-8
unexport QT_QPA_FONTDIR

.DEFAULT_GOAL := help
.PHONY: help venv install install-gui test test-ci test-order test-order-quick test-slow test-all bench \
        bench-baseline demo lint fmt rust-test rust-gpu wheel sdist dist-check \
        leak-check clean

help: ## Show this list of targets
	@echo "OpenDocking — make targets (the same commands CI runs)"
	@echo
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "The Python used is: $(PYTHON)"

venv: ## Create the virtual environment and install the build tooling
	$(PYTHON) -m venv .venv 2>/dev/null || python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install "maturin>=1.5,<2.0"

install: ## Build the extension and install the dev extras (one build)
	$(MATURIN) develop --release --extras=dev

install-gui: ## The same, plus PyQt6 + ModernGL for the workbench
	$(MATURIN) develop --release --extras=dev,gui

demo: demo/systems/3ptb/receptor.pdbqt ## Generate the bundled demo fixtures (offline)

demo/systems/3ptb/receptor.pdbqt:
	$(PYTHON) examples/make_demo.py --fast

test: demo ## The CI test command: fast suite, refusing to skip the demo fixtures
	$(PYTEST) tests -q -m "not slow" --require-demo

test-ci: demo ## Exactly what CI runs, including the refusal to skip the Qt suite
	$(PYTEST) tests -q -m "not slow" --require-demo --require-gui

test-order: demo ## The suite in two orders, failing if the two disagree (the gate)
	$(PYTHON) tools/check_test_order.py --markers "not slow" --extra=--require-demo

test-order-quick: demo ## The order gate on a slice of files, for between edits
	$(PYTHON) tools/check_test_order.py --sample 12 --markers "not slow"

test-slow: demo ## The slow suite alone (crystallographic validation)
	$(PYTEST) tests -q -m slow

test-all: demo ## Everything, including the slow suite
	$(PYTEST) tests -q

bench: ## Run the accuracy benchmark and compare it with benchmark/baseline.json
	@mkdir -p out
	$(PYTHON) -m odock.benchmark --json out/benchmark-results.json \
		--csv out/benchmark-results.csv --check-baseline

bench-baseline: ## Re-record benchmark/baseline.json from a fresh run (review it!)
	@mkdir -p out
	$(PYTHON) -m odock.benchmark --json out/benchmark-updated.json \
		--csv out/benchmark-updated.csv --write-baseline

lint: ## Static checks: ruff (narrow, bug-finding rules) + clippy
	$(PYTHON) -m ruff check python tests tools examples
	cargo clippy --workspace --all-targets -- -D warnings

fmt: ## Format Python (ruff format) and Rust (cargo fmt --all)
	$(PYTHON) -m ruff format python tests tools examples
	cargo fmt --all

rust-test: ## The Rust kernel's own test suite
	cargo test --workspace

rust-gpu: ## The kernel with the optional GPU feature (a head-less machine falls back)
	cargo test -p dock-core --features gpu -- --nocapture

wheel: ## Build a release wheel into dist/ and check what is inside it
	$(MATURIN) build --release --out $(DIST)
	$(PYTHON) tools/inspect_dist.py $(DIST)

sdist: ## Build the source distribution into dist/ and check what is inside it
	$(MATURIN) sdist --out $(DIST)
	$(PYTHON) tools/inspect_dist.py $(DIST)

dist-check: ## Inspect the artefacts already in dist/ (does not rebuild them)
	$(PYTHON) tools/inspect_dist.py $(DIST)

leak-check: ## Prove the packaging deny-list still fires (no artefact needed)
	$(PYTHON) tools/inspect_dist.py --self-test

clean: ## Remove build output (the Python caches, dist/ and the Rust target dir)
	rm -rf $(DIST) build *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	cargo clean

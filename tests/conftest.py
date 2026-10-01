# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest fixtures for the OpenDocking test-suite.

This file also carries two guards against a green run that proves nothing:

* ``--require-demo`` — the demo fixtures (``demo/systems/3ptb/...``, ``demo/systems/1m17/...``)
  are generated rather than committed, so a fresh checkout has none of them and
  every test that needs them *skips*.  A skipped accuracy test is not a passing
  accuracy test, so CI runs pytest with ``--require-demo``, which turns those
  skips into a failure with the command that fixes it.
* ``--require-gui`` — the workbench tests call
  ``pytest.importorskip("PyQt6...")``, so a runner without the GUI extra skips
  the whole head-less Qt suite, including the layout assertion that has to hold
  with ``QT_QPA_PLATFORM=offscreen`` and ``QT_QPA_FONTDIR`` **unset**.  CI installs
  the extra and passes this flag so that cannot happen silently.

Locally both flags are off and a missing dependency skips as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

DATA = Path(__file__).resolve().parent / "data"

#: What a skip message must contain to count as "the demo fixtures are missing".
#: Every fixture that depends on generated data says "demo" in its reason.
_DEMO_MARKERS = ("demo", "bundled")

#: What a skip message must contain to count as "the workbench could not run".
_GUI_MARKERS = ("pyqt6", "moderngl", "qtwidgets", "qtgui")

#: Skips recorded by :func:`pytest_runtest_logreport`, reported by
#: :func:`pytest_sessionfinish` when the matching flag is given.
_DEMO_SKIPS: list = []
_GUI_SKIPS: list = []


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--require-demo",
        action="store_true",
        default=False,
        help=(
            "fail instead of skipping when the generated demo fixtures are "
            "missing (CI uses this so a green run cannot hide skipped accuracy "
            "tests); run `make demo` to generate them"
        ),
    )
    parser.addoption(
        "--require-gui",
        action="store_true",
        default=False,
        help=(
            "fail instead of skipping when PyQt6 or ModernGL is missing (CI "
            "installs the [gui] extra and uses this so the head-less workbench "
            "suite, including the layout assertion, cannot silently disappear)"
        ),
    )


def pytest_configure(config) -> None:
    _DEMO_SKIPS.clear()
    _GUI_SKIPS.clear()


def pytest_runtest_logreport(report) -> None:
    """Remember every skip whose reason mentions generated data or the workbench.

    A module-level list rather than something stashed on the config: a
    :class:`TestReport` carries no reference to the config, and this hook is the
    one that runs for every phase of every test.
    """
    if not report.skipped:
        return
    reason = ""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        reason = str(longrepr[2])
    reason = reason or str(longrepr or "")
    lowered = reason.lower()
    entry = f"{report.nodeid}: {reason}"
    if any(marker in lowered for marker in _DEMO_MARKERS):
        _DEMO_SKIPS.append(entry)
    elif any(marker in lowered for marker in _GUI_MARKERS):
        _GUI_SKIPS.append(entry)


def _report(option: str, recorded: list, what: str, fix: str) -> bool:
    if not recorded:
        return False
    lines = ["", f"{len(recorded)} test(s) needed {what} and skipped instead:"]
    lines += [f"  {entry}" for entry in recorded[:10]]
    if len(recorded) > 10:
        lines.append(f"  ... and {len(recorded) - 10} more")
    lines += ["", fix]
    print("\n".join(lines))
    return True


# ---------------------------------------------------------------------------
# Process-global state, restored around every test
# ---------------------------------------------------------------------------

#: Module-level containers that live for the whole process and are written by the code
#: under test or by a test's own fixture.  A test that appends to one changes what a
#: *later* test sees, which is how a suite starts depending on file order.  The list is
#: explicit rather than clever: each entry is ``"module:attribute"`` and is restored
#: (mutated in place, so existing references keep working) after every test.
_GLOBAL_REGISTRIES = (
    # `ensemble._scratch_dir` registers every scratch directory it hands out for
    # teardown at exit; a test that measures its *own* scratch usage would otherwise
    # see the directories every earlier test left behind.
    "odock.ensemble:_SCRATCH",
)


@pytest.fixture(autouse=True)
def _isolate_process_globals():
    """Restore the process-global state a test may have written.

    Two failures in this repository have been *"passes alone, fails in the full run"*,
    and the general shape of that bug is a piece of process-global state — an
    environment variable, the working directory, a module-level registry, a seeded
    generator the previous test reset.  This fixture makes the *known* state explicit
    and restores it, so a leak of that kind is silenced at the source rather than
    reordered around.  It is deliberately cheap: three snapshots and a restore per
    test, and no behaviour of its own.

    It does not hide an order dependence that lives in a *library* global (a Qt
    singleton, an RDKit logger); `tools/check_test_order.py` is the gate that catches
    those, and the fix for them is in the test that starts them.
    """
    import importlib
    import os
    import random

    environment = dict(os.environ)
    cwd = os.getcwd()
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    saved: list = []
    for entry in _GLOBAL_REGISTRIES:
        module_name, _, attribute = entry.partition(":")
        module = sys.modules.get(module_name)
        if module is None:
            continue
        value = getattr(module, attribute, None)
        if isinstance(value, list):
            saved.append((value, list(value)))
        elif isinstance(value, dict):
            saved.append((value, dict(value)))
        elif isinstance(value, set):
            saved.append((value, set(value)))

    yield

    os.environ.clear()
    os.environ.update(environment)
    try:
        os.chdir(cwd)
    except OSError:  # pragma: no cover - a test removed its own cwd
        pass
    random.setstate(python_state)
    if numpy is not None and numpy_state is not None:
        numpy.random.set_state(numpy_state)
    for container, contents in saved:
        if isinstance(container, list):
            container[:] = contents
        elif isinstance(container, dict):
            container.clear()
            container.update(contents)
        else:
            container.clear()
            container.update(contents)


def pytest_sessionfinish(session, exitstatus) -> None:
    """With the matching flag, a silently skipped suite fails the session."""
    failed = False
    if session.config.getoption("--require-demo", default=False):
        failed |= _report(
            "--require-demo",
            list(_DEMO_SKIPS),
            "the generated demo fixtures",
            "Generate them offline with `make demo` (or "
            "`python examples/make_demo.py --fast`), or drop --require-demo to "
            "accept the skips.",
        )
    if session.config.getoption("--require-gui", default=False):
        failed |= _report(
            "--require-gui",
            list(_GUI_SKIPS),
            "PyQt6 and ModernGL",
            "Install them with `pip install '.[gui]'` (and on a bare Linux image "
            "the Qt/GL libraries), or drop --require-gui to accept the skips.",
        )
    if failed:
        session.exitstatus = 1


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory holding the bundled test structures."""
    return DATA


@pytest.fixture(scope="session")
def pdb_3ptb(data_dir: Path) -> Path:
    """Bovine trypsin with benzamidine (PDB 3PTB)."""
    p = data_dir / "3PTB.pdb"
    if not p.exists():
        pytest.skip(f"missing test fixture {p}")
    return p


@pytest.fixture(scope="session")
def complex_parts(pdb_3ptb: Path):
    """``(receptor_mol, ligand_mol)`` split out of 3PTB."""
    Chem = pytest.importorskip("rdkit.Chem")  # type: ignore[attr-defined]
    from rdkit import Chem as _Chem

    mol = _Chem.MolFromPDBFile(
        str(pdb_3ptb), removeHs=False, sanitize=False, proximityBonding=True
    )
    assert mol is not None
    lig_res = {"BEN"}
    water = {"HOH", "WAT", "DOD"}
    keep_rec, keep_lig = [], []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        name = info.GetResidueName().strip() if info else ""
        if name in water:
            continue
        (keep_lig if name in lig_res else keep_rec).append(atom.GetIdx())

    def subset(indices):
        em = _Chem.RWMol(mol)
        total = set(range(mol.GetNumAtoms()))
        for i in sorted(total - set(indices), reverse=True):
            em.RemoveAtom(i)
        out = em.GetMol()
        try:
            _Chem.SanitizeMol(out)
        except Exception:
            pass
        return out

    return subset(keep_rec), subset(keep_lig)


@pytest.fixture(scope="session")
def prepared_3ptb(complex_parts, tmp_path_factory):
    """Prepared receptor and ligand for 3PTB, shared by the docking tests."""
    import odock

    outdir = tmp_path_factory.mktemp("3ptb")
    receptor_mol, ligand_mol = complex_parts
    _, rec_pdbqt, rec_report = odock.prepare_receptor(
        receptor_mol, outdir / "receptor.pdbqt", keep_water=False
    )
    lig_mol, lig_pdbqt, lig_report = odock.prepare_ligand(
        ligand_mol, outdir / "ligand.pdbqt", name="BEN", optimize=False
    )
    return {
        "receptor_mol": receptor_mol,
        "ligand_mol": ligand_mol,
        "ligand_prepared": lig_mol,
        "receptor_pdbqt": rec_pdbqt,
        "ligand_pdbqt": lig_pdbqt,
        "receptor_report": rec_report,
        "ligand_report": lig_report,
        "outdir": outdir,
    }

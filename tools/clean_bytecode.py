# SPDX-License-Identifier: GPL-3.0-or-later
"""Remove bytecode residue before packaging, portably and read-only-safely.

Why this exists
---------------
``pyproject.toml``'s ``[tool.maturin] exclude`` filters the **sdist**; a wheel copies
``python-source`` as-is, so any ``__pycache__`` created while running the tooling ships
inside the wheel. Measured on the 0.3.2 release run, the wheel asserted to carry no
bytecode carried ``odock/__pycache__/*.cpython-311.pyc`` for **every** module.

A wheel is importable code, not the build residue of the machine that built it. There
are two ways to prevent that and only one of them is reliable:

* *exclude it afterwards* — ``maturin`` has no wheel-side exclude, and a deny-check
  that runs after the artefact exists only tells you the artefact is wrong;
* *do not have it at build time* — this script, which is what the workflow runs.

``shutil.rmtree`` is not used here on purpose: a directory carrying the read-only
attribute cannot be unlinked on Windows, and the first draft of this cleanup raised
``PermissionError: [WinError 5]`` on exactly that. ``odock._fs.remove_tree`` is the
project's one implementation of that (clear the flag on the root *and* every entry,
then verify the tree is gone), and it is loaded from its file so this runs **before**
the package is installed or importable.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import List

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_remove_tree():
    """Import ``odock/_fs.py`` without importing the package (no kernel needed)."""
    path = ROOT / "python" / "odock" / "_fs.py"
    spec = importlib.util.spec_from_file_location("_odock_fs", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.remove_tree


def bytecode_directories(root: pathlib.Path) -> List[pathlib.Path]:
    """Every ``__pycache__`` under ``root``, deepest first so removal is safe."""
    found = [path for path in root.rglob("__pycache__") if path.is_dir()]
    return sorted(found, key=lambda path: len(path.parts), reverse=True)


def main(argv: List[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    roots = [pathlib.Path(item) for item in arguments] or [ROOT / "python"]
    remove_tree = _load_remove_tree()
    removed = 0
    for root in roots:
        if not root.exists():
            print(f"  {root}: absent, nothing to do")
            continue
        for directory in bytecode_directories(root):
            remove_tree(directory)
            removed += 1
        # A stray `.pyc` outside a `__pycache__` can only exist if something wrote it
        # by hand, but it would ship just the same.
        for stray in root.rglob("*.py[co]"):
            try:
                stray.unlink()
                removed += 1
            except OSError as exc:  # pragma: no cover - defensive
                raise SystemExit(f"cannot remove {stray}: {exc}")
    print(f"  removed {removed} bytecode path(s) from {', '.join(str(r) for r in roots)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

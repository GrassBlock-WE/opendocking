# SPDX-License-Identifier: GPL-3.0-or-later
"""OpenDocking — a modern, open-source molecular docking toolchain.

The package exposes the Rust kernel (`odock._odock`) through a small,
pythonic API::

    import odock

    receptor, receptor_pdbqt, _ = odock.prepare_receptor("receptor.pdb")
    ligand, ligand_pdbqt, _ = odock.prepare_ligand("ligand.sdf")
    box = odock.box_from_ligand(ligand, buffer=8.0)
    result = odock.dock(receptor_pdbqt, ligand_pdbqt, box, exhaustiveness=16, seed=42)
    print(result.table())
"""

from __future__ import annotations

__all__ = [
    "BoxSpec",
    "DockResult",
    "Pose",
    "PreparationReport",
    "XS_TYPES",
    "aligned_rmsd",
    "box_from_ligand",
    "box_from_points",
    "box_from_selection",
    "box_from_smiles_ligand",
    "dock",
    "dock_text",
    "gpu_available",
    "gpu_description",
    "kernel_version",
    "pdbqt_to_pdb_block",
    "planarity",
    "pose_to_mol",
    "prepare_ligand",
    "prepare_receptor",
    "read_structure",
    "score",
    "score_pair",
    "write_ligand_pdbqt",
    "write_receptor_pdbqt",
    "__version__",
]

# The compiled kernel. Import it first so that a broken installation fails with
# a clear message rather than a confusing attribute error later.
from . import _odock  # noqa: F401  (also reachable as odock._odock)

from .docking import (  # noqa: E402
    XS_TYPES,
    DockResult,
    Pose,
    aligned_rmsd,
    dock,
    gpu_available,
    gpu_description,
    kernel_version,
    pose_to_mol,
    score,
    score_pair,
)
from .pdbqt import write_ligand_pdbqt, write_receptor_pdbqt  # noqa: E402
from .prepare import (  # noqa: E402
    BoxSpec,
    PreparationReport,
    box_from_ligand,
    box_from_points,
    box_from_selection,
    box_from_smiles_ligand,
    pdbqt_to_pdb_block,
    planarity,
    prepare_ligand,
    prepare_receptor,
    read_structure,
)

__version__ = str(_odock.__version__)


def dock_text(receptor_pdbqt: str, ligand_pdbqt: str, box: BoxSpec, **kwargs) -> DockResult:
    """Dock directly from PDBQT *text* rather than from files."""
    return dock(receptor_pdbqt, ligand_pdbqt, box, **kwargs)


#: Submodules that are cheap to import and useful to reach as ``odock.<name>``.
#: The heavy ones (``gui``, the chemistry, the analysis) stay lazy so that
#: ``import odock`` never needs RDKit, Qt or a GPU.
_EAGER_SUBMODULES = ("analysis", "chem", "export", "fetch", "filters",
                     "pocket", "report")


def __getattr__(name: str):  # pragma: no cover - lazy submodule access
    """Expose submodules lazily so a headless install still imports cleanly."""
    if name in ("gui", "DockingWorkbench", "launch_gui"):
        from . import gui

        return gui if name == "gui" else getattr(gui, name)
    if name in _EAGER_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_EAGER_SUBMODULES) | {"gui"})


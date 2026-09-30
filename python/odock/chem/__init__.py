# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry perception: receptor and ligand preparation.

This subpackage holds the deep chemistry that the top-level :mod:`odock` API
exposes:

* :mod:`odock.chem.receptor` — HETATM classification, cleaning, missing-atom
  detection, pH-aware protonation.
* :mod:`odock.chem.charges` — Gasteiger and Kollman charges, AD4 atom typing.
* :mod:`odock.chem.ligand` — multi-format ligand input, 3-D embedding, force-field
  minimisation, rotatable-bond perception and the kinematic torsion tree.

Everything here needs RDKit, which is imported lazily by the callers so that
``import odock`` stays cheap.
"""

from __future__ import annotations

__all__ = [
    "receptor",
    "charges",
    "ligand",
]

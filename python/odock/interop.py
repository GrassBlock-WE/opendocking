# SPDX-License-Identifier: GPL-3.0-or-later
"""Take the workbench's view into the tools a modeller already uses.

A docking result is not finished when it is on screen in one program. It has to
survive being handed to a collaborator, dropped into a figure, or opened next to
a paper's coordinates — and the cheapest, most durable container for that is a
**plain PDB with the pose in it** plus a short script for the viewer the reader
actually has. This module writes three things, all of them *generated from the
live scene* rather than from a frozen template, so they cannot drift from what
the workbench is showing:

* a **PyMOL ``.pml``** that loads the structures, applies the current styles,
  draws every interaction annotation as a real ``distance`` object, shows the
  surface and colours it by the property that is on screen, and restores the
  camera with ``set_view``;
* a **ChimeraX ``.cxc``** that does the same with ChimeraX's own vocabulary;
* a **PDB** containing the receptor and the posed ligand, with the per-atom
  property written into the **B-factor column** — which is what makes
  ``spectrum b`` in PyMOL and ``color byattribute bfactor`` in ChimeraX
  reproduce the workbench's surface colouring exactly.

Where a viewer has no equivalent, the script says so in a comment instead of
inventing a command: ChimeraX has no ``set_view``, so the camera is not
reproduced there (a ``zoom`` onto the site is emitted instead), and PyMOL has
no tube representation, so the ``tube`` style falls back to a cartoon.

The module is deliberately free of Qt and ModernGL: it takes plain Atom-like
objects and duck-typed interactions, so it can be used from a script or a
server as well as from the GUI. It never imports RDKit itself — although
``import odock`` may, through the public API, so nothing here depends on it.
The few tables it needs that also live in the GUI (the interaction colours, the
style names) are duplicated here on purpose and a test asserts the two copies
agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "INTERACTION_COLORS",
    "INTERACTION_LABELS",
    "PDB_REMARKS",
    "STYLE_LIGAND",
    "STYLE_PROTEIN",
    "SceneState",
    "camera_vectors",
    "chimerax_script",
    "export_bundle",
    "pdb_atom_line",
    "pose_pdb_text",
    "pymol_script",
    "surface_obj_text",
    "write_chimerax_script",
    "write_pose_pdb",
    "write_pymol_script",
]


# ---------------------------------------------------------------------------
# the small tables
# ---------------------------------------------------------------------------

#: Colour per interaction type. A copy of
#: :data:`odock.gui.viewport.INTERACTION_COLORS` so this module can be imported
#: without Qt; ``tests/test_interop.py`` asserts the two stay identical.
INTERACTION_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "hbond": (0.20, 0.90, 0.95, 0.95),
    "salt_bridge": (0.95, 0.25, 0.85, 0.95),
    "pi_pi": (0.25, 0.90, 0.35, 0.95),
    "cation_pi": (0.98, 0.60, 0.15, 0.95),
    "hydrophobic": (0.56, 0.58, 0.44, 0.90),
    "clash": (1.00, 0.15, 0.15, 1.00),
}

#: Human-readable label per interaction type, for the script comments and the
#: distance object names.
INTERACTION_LABELS: Dict[str, str] = {
    "hbond": "hydrogen bond",
    "salt_bridge": "salt bridge",
    "pi_pi": "pi-pi stacking",
    "cation_pi": "cation-pi",
    "hydrophobic": "hydrophobic",
    "clash": "steric clash",
}

#: PyMOL commands per protein style, with the compound name placeholder.
#: Every entry is a real PyMOL command; the ones without an exact equivalent
#: carry the closest honest approximation and a comment in the script.
STYLE_PROTEIN: Dict[str, Tuple[str, ...]] = {
    "cartoon": ("show cartoon, {obj}",),
    "ribbon": ("show cartoon, {obj}", "set cartoon_flat_sheets, 1, {obj}"),
    # PyMOL has no tube: a smooth cartoon is the nearest representation, and
    # the script is emitted with a comment saying exactly that.
    "tube": ("show cartoon, {obj}", "set cartoon_flat_sheets, 0, {obj}"),
    "spheres": ("show spheres, {obj}", "set sphere_scale, {scale}, {obj}"),
    "spacefill": ("show spheres, {obj}", "set sphere_scale, 1.0, {obj}"),
    "sticks": ("show sticks, {obj}", "set stick_radius, 0.15, {obj}"),
    "ball_stick": (
        "show sticks, {obj}",
        "set stick_radius, 0.12, {obj}",
        "show spheres, {obj}",
        "set sphere_scale, 0.13, {obj}",
    ),
    "wireframe": ("show lines, {obj}", "set line_width, 1.4, {obj}"),
    "dots": ("show dots, {obj}",),
}

#: PyMOL commands per ligand style.
STYLE_LIGAND: Dict[str, Tuple[str, ...]] = {
    "ball_stick": (
        "show sticks, {obj}",
        "set stick_radius, 0.12, {obj}",
        "show spheres, {obj}",
        "set sphere_scale, {scale}, {obj}",
    ),
    "sticks": ("show sticks, {obj}", "set stick_radius, 0.15, {obj}"),
    "wireframe": ("show lines, {obj}", "set line_width, 1.4, {obj}"),
    "spheres": ("show spheres, {obj}", "set sphere_scale, {scale}, {obj}"),
    "spacefill": ("show spheres, {obj}", "set sphere_scale, 1.0, {obj}"),
}

#: ChimeraX commands per protein style. ChimeraX puts the atom specification
#: *before* the keyword options (``size #1 sphereRadius 0.2``), which is the
#: order these strings are written in.
CHIMERAX_PROTEIN: Dict[str, Tuple[str, ...]] = {
    "cartoon": ("cartoon {obj}",),
    "ribbon": ("cartoon {obj}", "cartoon style flat"),
    "tube": ("cartoon {obj}", "cartoon style tube"),
    "spheres": ("style sphere {obj}", "size {obj} sphereRadius {scale}"),
    "spacefill": ("style sphere {obj}", "size {obj} sphereRadius 1.0"),
    "sticks": ("style stick {obj}", "size {obj} stickRadius 0.15"),
    "ball_stick": (
        "style stick {obj}",
        "size {obj} stickRadius 0.12",
        "style sphere {obj}",
        "size {obj} sphereRadius 0.13",
    ),
    "wireframe": ("style wire {obj}",),
    "dots": ("style sphere {obj}", "size {obj} sphereRadius 0.3"),
}

#: ChimeraX commands per ligand style.
CHIMERAX_LIGAND: Dict[str, Tuple[str, ...]] = {
    "ball_stick": (
        "style stick {obj}",
        "size {obj} stickRadius 0.12",
        "style sphere {obj}",
        "size {obj} sphereRadius {scale}",
    ),
    "sticks": ("style stick {obj}", "size {obj} stickRadius 0.15"),
    "wireframe": ("style wire {obj}",),
    "spheres": ("style sphere {obj}", "size {obj} sphereRadius {scale}"),
    "spacefill": ("style sphere {obj}", "size {obj} sphereRadius 1.0"),
}

#: The header every generated PDB carries, so the file explains itself.
PDB_REMARKS: Tuple[str, ...] = (
    "GENERATED BY OPENDOCKING (odock.interop)",
    "COORDINATES ARE THE LIVE SCENE STATE OF THE WORKBENCH",
)

#: PyMOL's camera matrix is an orthonormal frame followed by positions; this is
#: the near/far clip it is given when the camera distance is known.
_VIEW_NEAR = 1.0
_VIEW_FAR = 1000.0


def _column(text: object, width: int, *, align: str = "<") -> str:
    """A fixed-width PDB field, truncated rather than allowed to overflow."""
    value = str(text if text is not None else "")
    value = value[:width]
    if align == ">":
        return value.rjust(width)
    return value.ljust(width)


# ---------------------------------------------------------------------------
# the scene description
# ---------------------------------------------------------------------------


@dataclass
class SceneState:
    """Everything a script needs to reproduce what the workbench is showing.

    The atoms are duck-typed: anything with ``name``, ``element``, ``res_name``,
    ``res_id``, ``chain``, ``x``, ``y``, ``z`` (and optionally ``charge``,
    ``ad_type``, ``serial``) works — which is exactly
    :class:`odock.gui.structure.Atom`. Bonds may be objects with ``.a``/``.b``
    (and ``.order``) or plain ``(a, b)`` pairs; interactions are objects with
    ``.kind``, ``.a`` (receptor atom index) and ``.b`` (ligand atom index).
    """

    receptor: Sequence = ()
    ligand: Sequence = ()
    receptor_bonds: Sequence = ()
    ligand_bonds: Sequence = ()
    interactions: Sequence = ()
    measurements: Sequence = ()
    box: Optional[Tuple[Sequence[float], Sequence[float]]] = None
    style_protein: str = "spheres"
    style_ligand: str = "ball_stick"
    show_receptor: bool = True
    show_ligand: bool = True
    receptor_scale: float = 0.20
    ball_scale: float = 0.55
    #: Per-atom property of the receptor (hydrophobicity, potential, ...). It
    #: goes into the B-factor column, which is how ``spectrum b`` and
    #: ``color byattribute bfactor`` reproduce the surface colouring.
    receptor_values: Optional[Sequence[float]] = None
    #: Same for the ligand, so a ligand property survives the trip too.
    ligand_values: Optional[Sequence[float]] = None
    #: Viewer annotations, in the shape ``odock.gui.app`` hands them over:
    #: ``{"text": str, "anchor": ("receptor"|"ligand", index) |
    #: {"residue": (chain, res_id, res_name)} | {"measurement": int},
    #: "color": (r, g, b), "visible": bool}``. An annotation anchored to a
    #: residue is placed at that residue's centroid; an invisible one is
    #: skipped, because that is what the viewer shows.
    annotations: Sequence = ()
    #: Atom-referenced measurements (the richer kind the measurement tool
    #: records): ``{"kind": "distance"|"angle"|"dihedral"|"centroid"|"plane"|
    #: "plane_angle"|"plane_bond", "refs": [("receptor"|"ligand", index), ...],
    #: "split": int | None, "value": float, "unit": "Å"|"°"}``. This supplements
    #: :attr:`measurements`, which are world-space point pairs.
    viewer_measurements: Sequence = ()
    property_name: str = ""
    property_unit: str = ""
    property_range: Optional[Tuple[float, float]] = None
    #: Surface display state.
    show_surface: bool = False
    surface_mode: str = "sas"
    surface_property: str = "hydrophobicity"
    surface_palette: str = "hydrophobicity"
    surface_alpha: float = 1.0
    surface: object = None
    #: ``{"target": (3,), "distance": f, "azimuth": f, "elevation": f, "fov": f}``
    camera: Optional[dict] = None
    title: str = "odock"
    receptor_source: Optional[str] = None
    ligand_source: Optional[str] = None
    #: Names of the files the scripts should load, filled in by
    #: :func:`export_bundle` so a standalone script points at what was written.
    receptor_file: str = "receptor.pdb"
    ligand_file: str = "ligand.pdb"
    pose_file: str = "pose.pdb"

    def receptor_atom(self, index: int):
        return self.receptor[index] if 0 <= index < len(self.receptor) else None

    def ligand_atom(self, index: int):
        return self.ligand[index] if 0 <= index < len(self.ligand) else None

    def all_values(self) -> Optional[List[float]]:
        """The per-atom B-factor column for the combined pose, or ``None``."""
        if self.receptor_values is None and self.ligand_values is None:
            return None
        receptor = list(self.receptor_values or [0.0] * len(self.receptor))
        ligand = list(self.ligand_values or [0.0] * len(self.ligand))
        return [float(value) for value in receptor] + [float(value) for value in ligand]


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------


def _atom_name_field(name: str, element: str) -> str:
    """The 4-character atom-name field, following the PDB convention.

    A one-letter element's name starts in column 14 (so ``CA`` reads as a
    carbon alpha), a two-letter element's in column 13 (so ``CA`` reads as
    calcium). Getting this wrong is the classic way to have a viewer bond a
    calcium to its neighbours.
    """
    text = str(name or "").strip()[:4]
    if not text:
        text = str(element or "C")
    if len(element or "") == 1 and len(text) < 4:
        return (" " + text).ljust(4)
    return text.ljust(4)


def pdb_atom_line(
    atom,
    serial: int,
    *,
    record: str = "ATOM",
    bfactor: float = 0.0,
    occupancy: float = 1.0,
) -> str:
    """One column-correct PDB ATOM/HETATM record.

    The columns are the ones the PDB format specification fixes, and they are
    what every reader depends on: a shifted coordinate column silently produces
    a structure whose atoms are in the wrong place, or one that does not parse
    at all.
    """
    name = str(getattr(atom, "name", "") or "")
    element = str(getattr(atom, "element", "") or "")
    res_name = _column(getattr(atom, "res_name", "") or "LIG", 3, align=">")
    chain = _column(getattr(atom, "chain", "") or " ", 1)
    try:
        res_id = int(getattr(atom, "res_id", 0) or 0) % 10000
    except (TypeError, ValueError):  # pragma: no cover - defensive
        res_id = 0
    try:
        x = float(getattr(atom, "x", 0.0))
        y = float(getattr(atom, "y", 0.0))
        z = float(getattr(atom, "z", 0.0))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        x = y = z = 0.0
    # A non-finite coordinate is written as 0.0: a viewer that meets "nan" in a
    # coordinate column refuses the whole file, and a wrecked atom in the wrong
    # place is easier to spot than a structure that will not open at all.
    x = x if math.isfinite(x) else 0.0
    y = y if math.isfinite(y) else 0.0
    z = z if math.isfinite(z) else 0.0
    value = float(bfactor)
    if not math.isfinite(value):
        value = 0.0
    element_field = _column(element, 2, align=">")
    return (
        f"{record:<6}{int(serial) % 100000:>5} "
        f"{_atom_name_field(name, element)} "
        f"{res_name} {chain}{res_id:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{max(0.0, min(1.0, float(occupancy))):6.2f}{value:6.2f}"
        f"          {element_field}"
    )


def _structure_lines(
    atoms,
    *,
    first_serial: int,
    record: str,
    values: Optional[Sequence[float]],
    remarks: Sequence[str] = (),
) -> List[str]:
    lines = [f"REMARK  {text}" for text in remarks]
    serial = first_serial
    for index, atom in enumerate(atoms):
        bfactor = 0.0
        if values is not None and index < len(values):
            bfactor = float(values[index])
        lines.append(
            pdb_atom_line(atom, serial, record=record, bfactor=bfactor)
        )
        serial += 1
    return lines


def pose_pdb_text(
    state: SceneState,
    *,
    include_receptor: bool = True,
    per_atom_values: Optional[Sequence[float]] = None,
) -> str:
    """The scene as a plain PDB: receptor as ``ATOM``, ligand as ``HETATM``.

    Serials are renumbered from 1 in file order (an atom's ``serial`` field in
    a PDBQT is not trustworthy after editing, and a duplicate serial makes a
    file many readers reject outright). The two structures are separated by
    ``TER``.

    ``per_atom_values`` fills the B-factor column; it must be index-aligned with
    ``receptor + ligand`` in the order they are written, which is what
    :meth:`SceneState.all_values` produces.
    """
    values = per_atom_values
    if values is None:
        values = state.all_values()
    header = list(PDB_REMARKS)
    header.append(f"TITLE     {state.title}")
    if state.property_name:
        header.append(f"PROPERTY  {state.property_name}")
    if state.property_unit:
        header.append(f"UNITS     {state.property_unit}")
    if state.property_range:
        header.append(
            "RANGE     "
            f"{float(state.property_range[0]):.4f} {float(state.property_range[1]):.4f}"
        )
    if state.surface:
        mode = str(getattr(state.surface, "mode", state.surface_mode)).upper()
        header.append(f"SURFACE   {mode}")

    lines = [f"REMARK  {text}" for text in header]
    serial = 1
    written = 0
    if include_receptor and len(state.receptor):
        receptor_values = None
        if values is not None:
            receptor_values = list(values[: len(state.receptor)])
        lines.extend(
            _structure_lines(
                state.receptor,
                first_serial=serial,
                record="ATOM",
                values=receptor_values,
            )
        )
        serial += len(state.receptor)
        written += len(state.receptor)
        lines.append("TER")
    if len(state.ligand):
        ligand_values = None
        if values is not None:
            ligand_values = list(values[len(state.receptor) : len(state.receptor) + len(state.ligand)])
        lines.extend(
            _structure_lines(
                state.ligand,
                first_serial=serial,
                record="HETATM",
                values=ligand_values,
            )
        )
        written += len(state.ligand)
    if written == 0:
        lines.append("REMARK  THE SCENE HAS NO ATOMS")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pose_pdb(path, state: SceneState, **kwargs) -> Path:
    """Write :func:`pose_pdb_text` to ``path`` and return the path."""
    target = Path(path)
    target.write_text(pose_pdb_text(state, **kwargs), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# the camera
# ---------------------------------------------------------------------------


def camera_vectors(camera: Optional[dict]):
    """``(right, up, forward, eye, target, distance)`` from a camera dict.

    The convention is :meth:`odock.gui.viewport.Camera.view`'s: the world up
    vector is ``+Z``, ``forward = normalize(target - eye)``,
    ``right = normalize(cross(forward, up))`` and ``up = cross(right, forward)``.
    Reproducing that here (rather than importing the viewport, which would drag
    Qt into a command-line export) is what lets the PyMOL script restore the
    same view.
    """
    if not camera:
        return None
    target = tuple(float(v) for v in camera.get("target", (0.0, 0.0, 0.0)))[:3]
    distance = float(camera.get("distance", 40.0))
    azimuth = float(camera.get("azimuth", 0.6))
    elevation = float(camera.get("elevation", 0.35))
    cos_e = math.cos(elevation)
    eye = (
        target[0] + distance * cos_e * math.cos(azimuth),
        target[1] + distance * cos_e * math.sin(azimuth),
        target[2] + distance * math.sin(elevation),
    )
    forward = [target[i] - eye[i] for i in range(3)]
    norm = math.sqrt(sum(component * component for component in forward))
    if norm < 1e-9:  # pragma: no cover - degenerate camera
        forward = [0.0, 0.0, -1.0]
    else:
        forward = [component / norm for component in forward]
    world_up = (0.0, 0.0, 1.0)
    right = [
        forward[1] * world_up[2] - forward[2] * world_up[1],
        forward[2] * world_up[0] - forward[0] * world_up[2],
        forward[0] * world_up[1] - forward[1] * world_up[0],
    ]
    norm = math.sqrt(sum(component * component for component in right))
    if norm < 1e-6:  # pragma: no cover - camera looking straight down
        right = [1.0, 0.0, 0.0]
    else:
        right = [component / norm for component in right]
    up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return right, up, forward, eye, target, distance


def _pymol_set_view(camera: Optional[dict]) -> Optional[str]:
    """A PyMOL ``set_view`` line for a camera dict.

    PyMOL's matrix is the camera-to-model transform: rows 0-2 are the camera's
    right/up/forward axes, 3-5 the camera position in model space, 6-8 the
    rotation origin, 9/10 the near and far clip and 11 the perspective flag.
    """
    vectors = camera_vectors(camera)
    if vectors is None:
        return None
    right, up, forward, eye, target, distance = vectors
    near = max(0.1, distance * 0.05)
    far = distance * 20.0 + 500.0
    numbers = [
        *right,
        *up,
        *forward,
        *eye,
        *target,
        near,
        far,
        -1.0,
    ]
    return "set_view (" + ", ".join(f"{value:.6f}" for value in numbers) + ")"


# ---------------------------------------------------------------------------
# PyMOL
# ---------------------------------------------------------------------------


def _value_text(value, unit: str) -> str:
    """``"3.21 A"`` for a measurement value, or ``""`` when there is none."""
    if value is None:
        return ""
    try:
        number = f"{float(value):.2f}"
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""
    return f"{number} {unit}".strip()


def _escape_pymol(text: str) -> str:
    """Make a label safe inside a PyMOL double-quoted string."""
    return str(text).replace("\\", "/").replace('"', "'").replace("\n", " ")


def _pymol_color(kind: str) -> str:
    red, green, blue, _alpha = INTERACTION_COLORS.get(kind, (0.8, 0.8, 0.8, 1.0))
    return f"set_color {kind}_color, [{red:.3f}, {green:.3f}, {blue:.3f}]"


def _anchor_point(state: "SceneState", anchor):
    """``(point, atom or None)`` for an annotation anchor, or ``(None, None)``.

    ``anchor`` is the viewer's own shape: a ``("receptor"|"ligand", index)``
    pair, a ``{"residue": (chain, res_id, res_name)}`` mapping (placed at the
    residue's centroid) or a ``{"measurement": index}`` reference (placed at the
    first atom that measurement names).
    """
    if isinstance(anchor, dict):
        if "residue" in anchor:
            key = tuple(anchor["residue"])[:3]
            members = [
                atom
                for atom in state.receptor
                if (
                    str(getattr(atom, "chain", "") or ""),
                    int(getattr(atom, "res_id", 0) or 0),
                    str(getattr(atom, "res_name", "") or ""),
                )
                == (str(key[0]), int(key[1]), str(key[2]))
            ]
            if not members:
                return None, None
            point = tuple(
                sum(float(getattr(atom, axis)) for atom in members) / len(members)
                for axis in ("x", "y", "z")
            )
            return point, None
        if "measurement" in anchor:
            try:
                index = int(anchor["measurement"])
                measurement = state.viewer_measurements[index]
                refs = list(measurement.get("refs") or [])
                if refs:
                    return _anchor_point(state, tuple(refs[0]))
            except Exception:
                return None, None
            return None, None
        return None, None
    try:
        which, index = str(anchor[0]), int(anchor[1])
    except (TypeError, IndexError, ValueError, KeyError):
        return None, None
    atom = state.receptor_atom(index) if which == "receptor" else state.ligand_atom(index)
    if atom is None:
        return None, None
    return (float(atom.x), float(atom.y), float(atom.z)), atom


def _measurement_ref_atoms(state: "SceneState", measurement):
    """``(which, index, atom)`` triples for a viewer measurement, or ``[]``."""
    out = []
    for ref in list(measurement.get("refs") or []):
        try:
            which, index = str(ref[0]), int(ref[1])
        except (TypeError, IndexError, ValueError):
            continue
        atom = state.receptor_atom(index) if which == "receptor" else state.ligand_atom(index)
        if atom is not None:
            out.append((which, index, atom))
    return out


def pymol_script(state: SceneState) -> str:
    """A ``.pml`` script reproducing the current view.

    Structure of the script, in order: load, hide, styles, interaction
    distances, measurements, search box, surface, camera. Everything is
    generated from ``state``; nothing is a fixed snippet.
    """
    lines: List[str] = [
        "# OpenDocking PyMOL script -- generated from the live scene.",
        f"# scene: {state.title}",
        f"# receptor: {len(state.receptor)} atoms"
        + (f" (from {state.receptor_source})" if state.receptor_source else ""),
        f"# ligand:   {len(state.ligand)} atoms"
        + (f" (from {state.ligand_source})" if state.ligand_source else ""),
        f"# protein style: {state.style_protein}    ligand style: {state.style_ligand}",
    ]
    if state.show_surface:
        lines.append(
            f"# surface: {str(state.surface_mode).upper()} coloured by "
            f"{state.surface_property}"
            + (
                f" over [{state.property_range[0]:.3f}, {state.property_range[1]:.3f}]"
                if state.property_range
                else ""
            )
        )
    lines.append("")

    # Load. The bundle writes these files next to the script, so the paths are
    # relative and the pair travels together.
    lines.append(f'load {state.receptor_file}, receptor')
    if len(state.ligand):
        lines.append(f'load {state.ligand_file}, ligand')
    lines.append("hide everything")
    lines.append("")

    def style_block(obj: str, style: str, table: Dict[str, Tuple[str, ...]], scale: float):
        lines.append(f"# {obj}: {style}")
        for command in table.get(style, table.get("spheres" if obj == "receptor" else "ball_stick", ())):
            lines.append(command.format(obj=obj, scale=f"{scale:g}"))
    if state.show_receptor and len(state.receptor):
        style_block("receptor", state.style_protein, STYLE_PROTEIN, state.receptor_scale)
    if state.show_ligand and len(state.ligand):
        style_block("ligand", state.style_ligand, STYLE_LIGAND, state.ball_scale)
    lines.append("")

    # Interaction distances: real PyMOL distance objects, coloured per type.
    seen_colors = set()
    for index, item in enumerate(state.interactions or (), start=1):
        kind = str(getattr(item, "kind", "") or "contact")
        first = getattr(item, "a", None)
        second = getattr(item, "b", None)
        if first is None or second is None:
            try:
                first, second = int(item[0]), int(item[1])
            except Exception:  # pragma: no cover - defensive
                continue
        try:
            first = int(first)
            second = int(second)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if not (0 <= first < len(state.receptor) and 0 <= second < len(state.ligand)):
            lines.append(
                f"# interaction {index} ({kind}) skipped: atom index out of range"
            )
            continue
        if kind not in seen_colors:
            lines.append(_pymol_color(kind))
            seen_colors.add(kind)
        name = f"{kind}_{index}"
        lines.append(
            f"distance {name}, receptor and index {first + 1}, "
            f"ligand and index {second + 1}"
        )
        lines.append(f"color {kind}_color, {name}")
        lines.append(f"set dash_gap, 0.35, {name}")
        lines.append(f"set dash_radius, 0.08, {name}")
        label = INTERACTION_LABELS.get(kind, kind)
        lines.append(f"# {label}: receptor atom {first}, ligand atom {second}")
    if state.interactions:
        lines.append("")

    # Measurements are world-space point pairs: pseudoatoms are the honest way
    # to draw them in PyMOL, which measures between selections.
    for index, measurement in enumerate(state.measurements or (), start=1):
        try:
            first, second = measurement[0], measurement[1]
            label = measurement[2] if len(measurement) > 2 else ""
        except Exception:  # pragma: no cover - defensive
            continue
        lines.append(
            f"pseudoatom measure_{index}a, pos=[{first[0]:.3f}, {first[1]:.3f}, {first[2]:.3f}]"
        )
        lines.append(
            f"pseudoatom measure_{index}b, pos=[{second[0]:.3f}, {second[1]:.3f}, {second[2]:.3f}]"
        )
        lines.append(f"distance m{index}, measure_{index}a, measure_{index}b")
        lines.append("set dash_gap, 0, m%d" % index)
        if label:
            lines.append(f'label m{index}, "{label}"')
    if state.measurements:
        lines.append("")

    # -- the measurement tool's richer measurements ------------------------
    # A distance, an angle and a dihedral are commands both viewers have; a
    # centroid, a plane and the plane/plane or plane/bond angles are not, so
    # those are emitted as a labelled comment plus the pseudoatoms of the atoms
    # involved rather than as an invented command (the viewer's own reported
    # value is carried in the comment).
    simple = {"distance": "distance", "angle": "angle", "dihedral": "dihedral"}
    unsupported: List[str] = []
    for index, measurement in enumerate(state.viewer_measurements or (), start=1):
        kind = str(measurement.get("kind", "") or "").lower()
        refs = _measurement_ref_atoms(state, measurement)
        if not refs:
            lines.append(f"# measurement {index} ({kind}) skipped: no usable atoms")
            continue
        value = measurement.get("value")
        unit = str(measurement.get("unit", "") or "")
        name = f"viewer_{kind}_{index}"
        selections = [
            f"{which} and index {atom_index + 1}" for which, atom_index, _atom in refs
        ]
        if kind in simple and len(selections) == (3 if kind in ("angle", "dihedral") else 2):
            lines.append("# %s %s" % (kind, _value_text(value, unit)))
            lines.append(f"{simple[kind]} {name}, " + ", ".join(selections))
            lines.append(f"color yellow, {name}")
        else:
            unsupported.append(name)
            lines.append(
                "# %s %s -- %s has no single-verb equivalent in PyMOL; the atoms it"
                % (kind, _value_text(value, unit), kind)
            )
            lines.append(
                "# names are marked with pseudoatoms instead of guessing a command."
            )
            for position, (which, atom_index, atom) in enumerate(refs, start=1):
                lines.append(
                    "pseudoatom %s_%d, pos=[%.3f, %.3f, %.3f]"
                    % (
                        name,
                        position,
                        float(getattr(atom, "x", 0.0)),
                        float(getattr(atom, "y", 0.0)),
                        float(getattr(atom, "z", 0.0)),
                    )
                )
            lines.append(f"show spheres, {name}_*")
            lines.append(f"color orange, {name}_*")
            lines.append(f'label {name}_1, "{kind} {_value_text(value, unit)}"')
    if state.viewer_measurements:
        lines.append("")

    # -- viewer annotations ------------------------------------------------
    labels: List[str] = []
    for index, note in enumerate(state.annotations or (), start=1):
        if not note.get("visible", True):
            continue
        text = str(note.get("text", "") or "").strip()
        if not text:
            continue
        point, atom = _anchor_point(state, note.get("anchor"))
        if point is None:
            lines.append(f"# annotation {index} skipped: its anchor is not in the scene")
            continue
        name = f"note_{index}"
        red, green, blue = (float(v) for v in (list(note.get("color") or (1.0, 1.0, 0.5)) + [0.5])[:3])
        lines.append(f'pseudoatom {name}, pos=[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]')
        lines.append(f"set_color {name}_color, [{red:.3f}, {green:.3f}, {blue:.3f}]")
        lines.append(f"color {name}_color, {name}")
        lines.append(f"show spheres, {name}")
        lines.append(f"set sphere_scale, 0.25, {name}")
        lines.append(f'label {name}, "{_escape_pymol(text)}"')
        lines.append(f"set label_color, {name}_color, {name}")
        labels.append(name)
        del atom
    if labels:
        lines.append("")

    # The search box, as eight corner markers: cheap, and unmistakable.
    if state.box is not None:
        center, size = state.box
        lines.append(f"# search box: centre {tuple(round(float(v), 3) for v in center)}")
        for corner in range(8):
            position = [
                float(center[axis]) + (0.5 if corner >> axis & 1 else -0.5) * float(size[axis])
                for axis in range(3)
            ]
            lines.append(
                "pseudoatom box_%d, pos=[%.3f, %.3f, %.3f]" % (corner, *position)
            )
        lines.append("show dots, box_*")
        lines.append("color cyan, box_*")
        lines.append("")

    if state.show_surface and len(state.receptor):
        lines.append("# surface")
        lines.append("show surface, receptor")
        lines.append("set surface_quality, 1")
        lines.append(f"set transparency, {max(0.0, min(1.0, 1.0 - float(state.surface_alpha))):.2f}, receptor")
        if state.property_range:
            low, high = float(state.property_range[0]), float(state.property_range[1])
            palette = "bluered" if state.surface_property == "electrostatic" else "yellow_white_blue"
            lines.append(
                f"spectrum b, {palette}, receptor, minimum={low:.4f}, maximum={high:.4f}"
            )
            if state.surface_property == "hydrophobicity":
                lines.append(
                    "# b encodes hydrophobicity: blue = hydrophilic (0), "
                    "yellow = hydrophobic (1); the PDB B-factor column carries it"
                )
        else:
            lines.append("spectrum b, rainbow, receptor")
        lines.append("")

    view = _pymol_set_view(state.camera)
    if view:
        lines.append("# camera, from the workbench's orbit camera")
        lines.append(view)
    lines.append("")
    return "\n".join(lines)


def write_pymol_script(path, state: SceneState) -> Path:
    target = Path(path)
    target.write_text(pymol_script(state), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# ChimeraX
# ---------------------------------------------------------------------------


def chimerax_script(state: SceneState) -> str:
    """A ``.cxc`` script reproducing the current view in ChimeraX.

    ChimeraX selects by residue and atom *name* rather than by index, and it
    has no ``set_view``, so the camera is not reproduced — the script says so in
    a comment and zooms to the binding site instead.
    """
    lines: List[str] = [
        "# OpenDocking ChimeraX script -- generated from the live scene.",
        f"# scene: {state.title}",
        "# NOTE: ChimeraX has no equivalent of PyMOL's set_view, so the camera is",
        "#       not reproduced; the model is zoomed to the ligand instead.",
        "",
    ]
    if len(state.receptor):
        lines.append(f"open {state.receptor_file}")
    if len(state.ligand):
        lines.append(f"open {state.ligand_file}")
    lines.append("hide atoms")
    lines.append("")

    if state.show_receptor and len(state.receptor):
        lines.append(f"# protein: {state.style_protein}")
        for command in CHIMERAX_PROTEIN.get(state.style_protein, ()):
            lines.append(command.format(obj="#1", scale=f"{state.receptor_scale:g}"))
    if state.show_ligand and len(state.ligand):
        lines.append(f"# ligand: {state.style_ligand}")
        for command in CHIMERAX_LIGAND.get(state.style_ligand, ()):
            lines.append(command.format(obj="#2", scale=f"{state.ball_scale:g}"))
    lines.append("")

    for index, item in enumerate(state.interactions or (), start=1):
        kind = str(getattr(item, "kind", "") or "contact")
        first = getattr(item, "a", None)
        second = getattr(item, "b", None)
        if first is None or second is None:
            continue
        receptor_atom = state.receptor_atom(int(first))
        ligand_atom = state.ligand_atom(int(second))
        if receptor_atom is None or ligand_atom is None:
            lines.append(f"# {kind} {index}: atom index out of range, skipped")
            continue
        colour = INTERACTION_COLORS.get(kind, (0.8, 0.8, 0.8, 1.0))
        lines.append("# %s %d" % (INTERACTION_LABELS.get(kind, kind), index))
        lines.append(
            "distance %s_%d %s %s"
            % (
                kind,
                index,
                _chimerax_atom_spec("#1", receptor_atom),
                _chimerax_atom_spec("#2", ligand_atom),
            )
        )
        # ChimeraX colour specs accept #rrggbb, so the dash carries exactly the
        # workbench's colour for that interaction type.
        lines.append(
            "color %s_%d #%02x%02x%02x"
            % (
                kind,
                index,
                int(round(colour[0] * 255)),
                int(round(colour[1] * 255)),
                int(round(colour[2] * 255)),
            )
        )
    if state.interactions:
        lines.append("")

    for index, measurement in enumerate(state.measurements or (), start=1):
        try:
            first, second = measurement[0], measurement[1]
        except Exception:  # pragma: no cover - defensive
            continue
        lines.append(
            "marker #90%d position %.3f,%.3f,%.3f radius 0.3 color gold"
            % (index, float(first[0]), float(first[1]), float(first[2]))
        )
        lines.append(
            "marker #91%d position %.3f,%.3f,%.3f radius 0.3 color gold"
            % (index, float(second[0]), float(second[1]), float(second[2]))
        )
        lines.append("distance m%d #90%d #91%d" % (index, index, index))

    # The measurement tool's atom-referenced measurements: distance, angle and
    # dihedral map onto ChimeraX commands; centroid/plane/plane-angle have no
    # single-verb form, so those become labelled markers plus the value.
    simple = {"distance": "distance", "angle": "angle", "dihedral": "dihedral"}
    for index, measurement in enumerate(state.viewer_measurements or (), start=1):
        kind = str(measurement.get("kind", "") or "").lower()
        refs = _measurement_ref_atoms(state, measurement)
        value = measurement.get("value")
        unit = str(measurement.get("unit", "") or "")
        if not refs:
            lines.append(f"# measurement {index} ({kind}) skipped: no usable atoms")
            continue
        specs = [
            _chimerax_atom_spec("#1" if which == "receptor" else "#2", atom)
            for which, _atom_index, atom in refs
        ]
        if kind in simple and len(specs) == (3 if kind in ("angle", "dihedral") else 2):
            lines.append("# %s %s" % (kind, _value_text(value, unit)))
            lines.append(f"{simple[kind]} {kind}_{index} " + " ".join(specs))
            lines.append(f"color {kind}_{index} goldenrod")
        else:
            lines.append(
                "# %s %s -- ChimeraX has no single-verb equivalent for %s; the"
                % (kind, _value_text(value, unit), kind)
            )
            lines.append("# atoms it names are marked instead of guessing a command.")
            for position, (which, _atom_index, atom) in enumerate(refs, start=1):
                lines.append(
                    "marker #80%d%d position %.3f,%.3f,%.3f radius 0.25 color orange"
                    % (
                        index % 10,
                        position,
                        float(getattr(atom, "x", 0.0)),
                        float(getattr(atom, "y", 0.0)),
                        float(getattr(atom, "z", 0.0)),
                    )
                )

    # Viewer annotations: a named marker at the anchor, with the label text.
    for index, note in enumerate(state.annotations or (), start=1):
        if not note.get("visible", True):
            continue
        text = str(note.get("text", "") or "").strip()
        if not text:
            continue
        point, _atom = _anchor_point(state, note.get("anchor"))
        if point is None:
            lines.append(f"# annotation {index} skipped: its anchor is not in the scene")
            continue
        red, green, blue = (
            float(v) for v in (list(note.get("color") or (1.0, 1.0, 0.5)) + [0.5])[:3]
        )
        lines.append(
            "marker #70%d position %.3f,%.3f,%.3f radius 0.3 color #%02x%02x%02x"
            % (
                index % 10,
                point[0],
                point[1],
                point[2],
                int(round(red * 255)),
                int(round(green * 255)),
                int(round(blue * 255)),
            )
        )
        lines.append('label #70%d text "%s"' % (index % 10, str(text).replace('"', "'")))

    if state.box is not None:
        center, size = state.box
        lines.append(
            "# search box: centre %.3f,%.3f,%.3f size %.3f,%.3f,%.3f"
            % (*[float(v) for v in center], *[float(v) for v in size])
        )
        for corner in range(8):
            position = [
                float(center[axis]) + (0.5 if corner >> axis & 1 else -0.5) * float(size[axis])
                for axis in range(3)
            ]
            lines.append(
                "marker #200%d position %.3f,%.3f,%.3f radius 0.4 color cyan"
                % (corner, *position)
            )
    lines.append("")

    if state.show_surface and len(state.receptor):
        lines.append("# surface")
        lines.append("surface #1")
        lines.append(f"transparency {max(0.0, min(1.0, 1.0 - float(state.surface_alpha))) * 100:.0f}")
        if state.property_range:
            low, high = float(state.property_range[0]), float(state.property_range[1])
            palette = "bluered" if state.surface_property == "electrostatic" else "yellow:white:blue"
            lines.append(
                f"color #1 byattribute bfactor palette {palette} "
                f"range {low:.4f},{high:.4f}"
            )
            lines.append(
                "# bfactor carries " + (state.property_name or state.surface_property)
            )
    lines.append("")

    if len(state.ligand):
        lines.append("zoom #2")
    elif len(state.receptor):
        lines.append("zoom #1")
    lines.append("")
    return "\n".join(lines)


def _chimerax_atom_spec(model: str, atom) -> str:
    """``#1/A:102@OD1`` — ChimeraX selects residues and atoms by name."""
    chain = str(getattr(atom, "chain", "") or "")
    try:
        res_id = int(getattr(atom, "res_id", 0) or 0)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        res_id = 0
    name = str(getattr(atom, "name", "") or "").strip()
    prefix = f"{model}/{chain}:{res_id}" if chain.strip() else f"{model}:{res_id}"
    return f"{prefix}@{name}" if name else prefix


def write_chimerax_script(path, state: SceneState) -> Path:
    target = Path(path)
    target.write_text(chimerax_script(state), encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# the surface mesh as OBJ
# ---------------------------------------------------------------------------


def surface_obj_text(surface) -> str:
    """The surface mesh as a Wavefront OBJ with per-vertex colour.

    ``v x y z r g b`` is the widely supported extension (MeshLab, Blender,
    ChimeraX's obj reader) that carries the property colouring with the
    geometry, so the picture survives without the PDB or this program.
    """
    if surface is None:
        return "# OpenDocking surface\n# no surface\n"
    vertices = getattr(surface, "vertices", None)
    triangles = getattr(surface, "triangles", None)
    if vertices is None or len(vertices) == 0:
        return "# OpenDocking surface\n# no surface\n"
    colours = getattr(surface, "colors", None)
    lines = [
        "# OpenDocking surface",
        f"# mode {getattr(surface, 'mode', 'sas')}, property {getattr(surface, 'property_name', '')}",
        f"# {len(vertices)} vertices, {0 if triangles is None else len(triangles)} triangles",
    ]
    if colours is None or len(colours) != len(vertices):
        colours = [[0.8, 0.8, 0.8]] * len(vertices)
    for index in range(len(vertices)):
        x, y, z = (float(value) for value in vertices[index][:3])
        red, green, blue = (float(value) for value in colours[index][:3])
        lines.append(f"v {x:.4f} {y:.4f} {z:.4f} {red:.4f} {green:.4f} {blue:.4f}")
    if triangles is not None:
        for triangle in triangles:
            a, b, c = (int(value) + 1 for value in triangle[:3])
            lines.append(f"f {a} {b} {c}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------


def export_bundle(directory, state: SceneState, *, stem: str = "odock") -> Dict[str, object]:
    """Write the poses, the scripts and the surface into one directory.

    Returns a manifest with the paths, so a caller (the workbench's log, a test,
    a script) can report exactly what was produced without guessing the naming.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    named = SceneState(**{**state.__dict__})
    named.receptor_file = f"{stem}_receptor.pdb"
    named.ligand_file = f"{stem}_ligand.pdb"
    named.pose_file = f"{stem}_pose.pdb"

    files: Dict[str, Path] = {}
    files["receptor"] = _write_side(named, named.receptor_file, target, named.receptor, "ATOM")
    files["ligand"] = _write_side(named, named.ligand_file, target, named.ligand, "HETATM")
    pose = target / named.pose_file
    pose.write_text(pose_pdb_text(named), encoding="utf-8")
    files["pose"] = pose

    pymol = target / f"{stem}.pml"
    pymol.write_text(pymol_script(named), encoding="utf-8")
    files["pymol"] = pymol
    chimerax = target / f"{stem}.cxc"
    chimerax.write_text(chimerax_script(named), encoding="utf-8")
    files["chimerax"] = chimerax

    surface_path = None
    if named.surface is not None:
        surface_path = target / f"{stem}_surface.obj"
        surface_path.write_text(surface_obj_text(named.surface), encoding="utf-8")
        files["surface"] = surface_path

    return {
        "directory": target,
        "stem": stem,
        "files": files,
        "pymol": pymol,
        "chimerax": chimerax,
        "pose": pose,
        "surface": surface_path,
        "receptor": files["receptor"],
        "ligand": files["ligand"],
    }


def _write_side(state: SceneState, name: str, directory: Path, atoms, record: str) -> Path:
    """One structure file of the bundle, carrying the property in the B-factor."""
    values = None
    if record == "ATOM":
        values = state.receptor_values
    else:
        values = state.ligand_values
    text = pose_pdb_text(
        SceneState(**{**state.__dict__, "receptor": atoms if record == "ATOM" else (), "ligand": atoms if record == "HETATM" else ()}),
        per_atom_values=list(values) if values is not None else None,
    )
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path

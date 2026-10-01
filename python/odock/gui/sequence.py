# SPDX-License-Identifier: GPL-3.0-or-later
"""The residue ruler that sits under the 3-D viewport.

What it is
----------
A linear sequence ruler, one horizontal row per chain: residues are drawn left
to right **in residue order** at a fixed cell width, and the row is divided into
**intervals of five residues**. A tick mark sits at the right edge of every
interval with the residue number above it, so the reader can count along the
chain the way they would count along a ruler.

Numbering rule (the one thing that needs stating)
-------------------------------------------------
Intervals are counted **from the first residue of the row**, not from residue 1,
and a boundary tick is labelled with the **real residue number of the last
residue in its interval**. For a chain numbered 1..12 the ticks therefore read
``1  5  10  12``; for 3PTB, whose first residue is 16, they read
``16  20  25  30 … 238`` — the first residue number and the last residue number
of a row are always labelled as well.

Residues are *counted*, numbers are *reported*: when the file has gaps (a
disordered loop, residues 31-36 missing) the interval still holds five residues
and the next tick simply carries the next real number that exists, so the labels
may step by more than five (``… 30  37 …``) while the five-code grouping stays
exact.

* an amino acid is one cell wide and shows its **one-letter code**, coloured by
  residue class (hydrophobic / polar / acidic / basic / Gly / Pro / Cys),
* anything that is not an amino acid keeps its **real residue name** (ligand
  ``BEN``, ion ``CA``/``ZN``, water ``HOH``, cofactor ``NAD``) and gets a wider
  cell so the name fits. Those entities are **not** given a row of their own:
  they ride in one tail at the end of the last chain, after a small gap, and
  **no tick numbers are drawn over the tail** — the numbering belongs to the
  polymer, and numbering BEN or HOH as well would only cost height and make the
  ruler harder to read. The tail is ordered by chain then residue number, so it
  is stable, and it never moves the residue codes aside: it simply extends the
  last row,
* a structure with no amino acids at all (a bare ligand) gets a single row named
  after its first entity,
* a receptor row is labelled with its chain id.

Selection
---------
Click selects a residue (clicking the only selected one clears it), shift-click
extends from the previous click, ctrl-click adds or removes one, and
double-click asks the window to centre the camera on that residue.
``selectionChanged`` carries ``[(chain, res_id, res_name), ...]`` and
``residueActivated`` carries ``[("receptor"|"ligand", atom_index), ...]``.

Why a plain ``QWidget`` with ``QPainter``
-----------------------------------------
The ruler has to work head-lessly (the test-suite and the simulation run on the
``offscreen`` Qt platform) and it has to stay cheap for a 30 000-atom receptor:
painting a few thousand small rectangles is nothing, so there is no widget per
residue and no OpenGL involved. Only :meth:`SequenceTrack.set_structure` does
real work, and it is ``O(n_atoms)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from .i18n import tr

__all__ = [
    "AMINO_ACIDS",
    "AMINO_ACID_ALIASES",
    "CLASS_COLORS",
    "CONTACT_MARKS",
    "DARK_RULER",
    "LIGHT_RULER",
    "STANDARD_AMINO_ACIDS",
    "ResidueBlock",
    "RulerPalette",
    "SequenceTrack",
    "class_color",
    "one_letter",
    "residue_class",
]

# ---------------------------------------------------------------------------
# residue nomenclature
# ---------------------------------------------------------------------------

#: The 20 standard amino acids.
STANDARD_AMINO_ACIDS: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

#: Spellings that are still that amino acid: the histidine and cysteine
#: protonation states a prepared receptor carries, the neutralised variants, and
#: selenomethionine (a very common HETATM in a crystal structure).
AMINO_ACID_ALIASES: Dict[str, str] = {
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HID": "H",
    "HIE": "H",
    "HIP": "H",
    "HSD": "H",
    "HSE": "H",
    "HSP": "H",
    "CYX": "C",
    "CYM": "C",
    "ASH": "D",
    "GLH": "E",
    "LYN": "K",
    "ARN": "R",
}

#: Every residue name that is drawn as a one-letter code.
AMINO_ACIDS: Dict[str, str] = {**STANDARD_AMINO_ACIDS, **AMINO_ACID_ALIASES}

#: Water, kept apart only so it can be drawn more quietly than a ligand.
WATER_NAMES = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3"})

#: The colour class of every one-letter code.
_CLASS_OF: Dict[str, str] = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "L": "hydrophobic",
    "I": "hydrophobic",
    "M": "hydrophobic",
    "F": "hydrophobic",
    "W": "hydrophobic",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "Y": "polar",
    "D": "acidic",
    "E": "acidic",
    "K": "basic",
    "R": "basic",
    "H": "basic",
    "G": "gly",
    "P": "pro",
    "C": "cys",
    "U": "cys",
    "O": "basic",
}

#: Cell colours per class, tuned for the dark viewport. ``other`` is the
#: non-polymer colour (a ligand, an ion, a cofactor) and ``water`` the quiet one.
CLASS_COLORS: Dict[str, Tuple[float, float, float]] = {
    "hydrophobic": (0.26, 0.44, 0.70),
    "polar": (0.16, 0.55, 0.45),
    "acidic": (0.71, 0.31, 0.29),
    "basic": (0.48, 0.35, 0.72),
    "gly": (0.45, 0.49, 0.55),
    "pro": (0.66, 0.52, 0.22),
    "cys": (0.74, 0.60, 0.18),
    "other": (0.62, 0.40, 0.66),
    "water": (0.20, 0.28, 0.38),
}

#: Legend order, the one the tests quote.
CLASS_ORDER = (
    "hydrophobic",
    "polar",
    "acidic",
    "basic",
    "gly",
    "pro",
    "cys",
    "other",
)


def one_letter(res_name: str) -> Optional[str]:
    """The one-letter code of an amino acid, or ``None`` for anything else."""
    return AMINO_ACIDS.get((res_name or "").strip().upper())


def residue_class(res_name: str, code: Optional[str] = None) -> str:
    """The colour class of a residue name (or of an already-known code)."""
    if code is None:
        code = one_letter(res_name)
    if code is not None:
        return _CLASS_OF.get(code, "other")
    name = (res_name or "").strip().upper()
    return "water" if name in WATER_NAMES else "other"


def class_color(name: str) -> Tuple[float, float, float]:
    """The RGB of a colour class, as floats in 0..1."""
    return CLASS_COLORS.get(name, CLASS_COLORS["other"])


def _foreground(rgb: Sequence[float]) -> QtGui.QColor:
    """Black or white text on ``rgb``, whichever is legible."""
    r, g, b = (float(c) for c in rgb)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance > 0.60:
        return QtGui.QColor(14, 18, 24)
    return QtGui.QColor(238, 242, 248)


# ---------------------------------------------------------------------------
# the cells
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RulerPalette:
    """The colours the ruler paints with.

    The ruler is a ``QWidget`` that fills its own background, so it cannot
    inherit a theme from the style sheet: the window hands it the palette that
    matches the active theme (see ``DARK_RULER``/``LIGHT_RULER``).
    """

    background: str = "#0d1016"
    bar: str = "#181d26"
    border: str = "#262e3c"
    tick: str = "#687484"
    tick_line: str = "#343e4e"
    tick_text: str = "#929eb0"
    label: str = "#96a2b4"
    cell_border: str = "#12161e"
    cell_text: str = "#ced6e2"
    tail_line: str = "#4a5668"
    empty_text: str = "#768294"


#: The default ruler palette: the dark workbench.
DARK_RULER = RulerPalette()

#: The light workbench: a light strip with dark codes, so the ruler does not
#: glare out of a light window the way a near-black strip does.
LIGHT_RULER = RulerPalette(
    background="#e9edf3",
    bar="#dee3eb",
    border="#c3cddc",
    tick="#5b6b7d",
    tick_line="#c8d1de",
    tick_text="#4a5a6b",
    label="#3c4a5b",
    cell_border="#ffffff",
    cell_text="#18222e",
    tail_line="#8fa0b3",
    empty_text="#5b6b7d",
)

#: ``(colour, band row)`` of every contact mark the ruler can draw. The bands are
#: stacked from the bottom of the cell upwards, so two marks on one residue are
#: both visible. Chosen to be unmistakable against the muted class colours and to
#: not clash with the gold selection outline.
CONTACT_MARKS: Dict[str, Tuple[Tuple[float, float, float], int]] = {
    # The displayed pose touches this residue.
    "contact": ((1.00, 0.67, 0.27), 0),
    # Both compared poses touch it …
    "shared": ((0.36, 0.86, 0.55), 0),
    # … and one of them touches it alone.
    "unique": ((0.95, 0.45, 0.42), 1),
}

#: The order marks are painted in, so a shared mark is never covered by a unique
#: one on the same cell.
MARK_ORDER = ("contact", "shared", "unique")

#: Height of one contact band, in pixels.
MARK_HEIGHT = 3


@dataclass
class ResidueBlock:
    """One residue of the ruler: what it is called and which atoms it holds."""

    chain: str
    res_id: int
    res_name: str
    code: str
    kind: str
    color_class: str
    group: str
    atom_refs: List[Tuple[str, int]] = field(default_factory=list)
    index: int = 0

    @property
    def key(self) -> Tuple[str, int, str]:
        """The selection key: ``(chain, res_id, res_name)``."""
        return (self.chain, self.res_id, self.res_name)

    @property
    def label(self) -> str:
        """What the cell writes: a one-letter code, or the real name."""
        return self.code


# ---------------------------------------------------------------------------
# the ruler
# ---------------------------------------------------------------------------


class SequenceTrack(QtWidgets.QWidget):
    """A horizontal, scrollable residue ruler that doubles as a picker.

    ``selectionChanged`` carries ``[(chain, res_id, res_name), ...]`` in ruler
    order, and ``residueActivated`` carries the atom references
    ``[("receptor"|"ligand", atom_index), ...]`` of the double-clicked residue.
    """

    selectionChanged = QtCore.pyqtSignal(list)
    residueActivated = QtCore.pyqtSignal(list)

    #: Residues per interval: a tick mark and a number every five residues.
    INTERVAL = 5
    #: One residue cell. 13 px keeps ~2000 residues scrollable and the letters
    #: legible; entity cells are wider because a residue name has to fit.
    CELL_WIDTH = 13
    ENTITY_WIDTH = 27
    #: A tick row plus one code row, as tight as the letters allow: the
    #: non-polymer entities ride in a tail on the last row instead of in a row of
    #: their own, which is what saves the height.
    CODE_HEIGHT = 14
    TICK_HEIGHT = 13
    ROW_GAP = 4
    LABEL_WIDTH = 24
    PADDING = 5
    #: The visual break between the last residue and the entity tail.
    TAIL_GAP = 7

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._groups: Dict[str, List[ResidueBlock]] = {}
        self._blocks: List[ResidueBlock] = []
        self._rows_data: List[Tuple[str, str, List[ResidueBlock]]] = []
        self._rects: List[QtCore.QRect] = []
        self._ticks: List[Tuple[str, int, int]] = []
        self._rows: List[Tuple[str, int, int, int]] = []
        self._selected: set = set()
        self._anchor: Optional[int] = None
        self._tail_marks: List[Tuple[int, int]] = []
        #: Residue keys the current pose (and, while comparing, the two compared
        #: poses) touch, by mark name. Stored as *keys*, not cell indices, so a
        #: reload of the same structure — switching pose reloads the ligand — keeps
        #: the marks on the residues they were computed for.
        self._contact_keys: Dict[str, set] = {}
        self.palette = DARK_RULER
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setAutoFillBackground(False)
        self.setToolTip(tr("tip.sequence"))
        self.setMinimumHeight(self.row_height())

    # -- appearance ---------------------------------------------------------

    def set_palette(self, palette: RulerPalette) -> None:
        """Use ``palette`` (the dark or the light ruler colours)."""
        self.palette = palette if palette is not None else DARK_RULER
        self.update()

    # -- contact marks ------------------------------------------------------

    def set_contact_marks(self, marks: Dict[str, Sequence[Tuple[str, int, str]]]) -> None:
        """Highlight the residues a pose touches, by mark name.

        ``marks`` maps a name from :data:`CONTACT_MARKS` ("contact", "shared",
        "unique") to the residue keys the mark covers. Names that are not in
        :data:`CONTACT_MARKS` are ignored; an empty mapping clears the marks.
        """
        cleaned: Dict[str, set] = {}
        for name, keys in (marks or {}).items():
            if name not in CONTACT_MARKS:
                continue
            wanted = {tuple(key) for key in keys or ()}
            if wanted:
                cleaned[name] = wanted
        self._contact_keys = cleaned
        self.update()

    def clear_contact_marks(self) -> None:
        self._contact_keys = {}
        self.update()

    def contact_marks(self) -> Dict[str, List[Tuple[str, int, str]]]:
        """The marks, resolved to the residues the ruler actually has."""
        present = {block.key for block in self._blocks}
        return {
            name: [block.key for block in self._blocks if block.key in keys]
            for name, keys in self._contact_keys.items()
            if keys & present
        }

    def marked_keys(self, name: str) -> List[Tuple[str, int, str]]:
        """The keys of one mark, in ruler order."""
        return list(self.contact_marks().get(name, []))

    def _marks_of(self, block: ResidueBlock) -> List[str]:
        return [
            name
            for name in MARK_ORDER
            if block.key in self._contact_keys.get(name, ())
        ]

    # -- geometry constants -------------------------------------------------

    def row_height(self) -> int:
        """The height of one ruler row: the tick header plus the code strip."""
        return self.TICK_HEIGHT + self.CODE_HEIGHT

    def track_height(self) -> int:
        """The height the widget wants for the rows it currently has."""
        rows = max(1, len(self._rows_data))
        return rows * (self.row_height() + self.ROW_GAP) + 2 * self.PADDING

    def _cell_width(self, block: ResidueBlock) -> int:
        return self.CELL_WIDTH if block.kind == "amino" else self.ENTITY_WIDTH

    # -- content ------------------------------------------------------------

    def set_structure(self, atoms: Sequence, group: str = "receptor") -> None:
        """Replace the cells of ``group``.

        ``group="receptor"`` is the "a new structure was loaded" call: it drops
        every other group too, because the ligand and the ions belonged to the
        old one. ``group="ligand"`` only replaces the ligand row, which is how a
        ligand is added to (or removed from) an open receptor.

        The marks survive for every residue that still exists: switching pose
        (which reloads the ligand, atom indices and all) and loading a structure
        that shares residue numbering must not throw away what the user picked.
        """
        previous = self.selected_keys()
        blocks = self._blocks_for(atoms, group)
        if group == "receptor":
            self._groups = {"receptor": blocks} if blocks else {}
        elif blocks:
            self._groups[group] = blocks
        else:
            self._groups.pop(group, None)
        self._rebuild()
        self._selected = set()
        self._anchor = None
        self.select_keys(previous, emit=False)
        self._emit()

    def clear(self) -> None:
        """Forget the whole structure and the selection."""
        self._groups.clear()
        self._rebuild()
        self._selected.clear()
        self._anchor = None
        self._emit()

    def blocks(self) -> List[ResidueBlock]:
        """Every cell, in ruler order."""
        return list(self._blocks)

    def block_keys(self) -> List[Tuple[str, int, str]]:
        """Every residue key, in ruler order."""
        return [block.key for block in self._blocks]

    def rows(self) -> List[Tuple[str, List[ResidueBlock]]]:
        """``(row label, cells)`` per ruler row, in drawing order."""
        return [(label, list(blocks)) for label, _group, blocks in self._rows_data]

    def _blocks_for(self, atoms: Sequence, group: str) -> List[ResidueBlock]:
        """Group ``atoms`` into residues, chains and numeric order.

        Atoms keep the order they appear in; the *residues* are then sorted by
        chain (first appearance) and residue number, which is what makes the
        ruler read left to right like the sequence does.
        """
        order: List[Tuple[str, int, str]] = []
        slots: Dict[Tuple[str, int, str], List[int]] = {}
        for index, atom in enumerate(atoms or ()):
            key = (
                str(getattr(atom, "chain", "") or ""),
                int(getattr(atom, "res_id", 0) or 0),
                str(getattr(atom, "res_name", "") or ""),
            )
            if key not in slots:
                order.append(key)
                slots[key] = []
            slots[key].append(index)

        chains: List[str] = []
        for chain, _res_id, _res_name in order:
            if chain not in chains:
                chains.append(chain)

        blocks: List[ResidueBlock] = []
        for chain in chains:
            keys = [key for key in order if key[0] == chain]
            keys.sort(key=lambda key: (key[1], key[2]))
            for key in keys:
                code = one_letter(key[2])
                colour_class = residue_class(key[2], code)
                kind = (
                    "amino"
                    if code
                    else ("water" if colour_class == "water" else "other")
                )
                blocks.append(
                    ResidueBlock(
                        chain=key[0],
                        res_id=key[1],
                        res_name=key[2],
                        code=code or key[2],
                        kind=kind,
                        color_class=colour_class,
                        group=group,
                        atom_refs=[(group, index) for index in slots[key]],
                    )
                )
        return blocks

    def _rebuild(self) -> None:
        """Recompute the rows, the cell order and the geometry.

        One row per polymer chain, and every non-polymer entity (ion, water,
        cofactor, ligand) becomes a cell of a **single tail appended to the last
        row**, ordered by chain and residue number. That keeps a structure with a
        ligand to exactly one row per chain and saves the height a separate
        entity row used to take.
        """
        self._blocks = []
        self._rows_data = []
        chain_order: List[str] = []
        polymer: Dict[str, List[ResidueBlock]] = {}
        tail: List[ResidueBlock] = []
        for block in self._all_cells():
            if block.kind == "amino":
                if block.chain not in chain_order:
                    chain_order.append(block.chain)
                polymer.setdefault(block.chain, []).append(block)
            else:
                tail.append(block)
        tail.sort(key=lambda item: (item.chain, item.res_id, item.res_name))

        if chain_order:
            for chain in chain_order:
                cells = list(polymer[chain])
                self._rows_data.append((chain or "-", "polymer", cells))
                self._blocks.extend(cells)
            if tail:
                label, kind, cells = self._rows_data[-1]
                self._rows_data[-1] = (label, kind, cells + tail)
                self._blocks.extend(tail)
        elif tail:
            self._rows_data.append((tail[0].res_name or "-", "tail", list(tail)))
            self._blocks.extend(tail)

        for index, block in enumerate(self._blocks):
            block.index = index
        # A stale selection index would point at a different residue.
        self._selected = {
            index for index in self._selected if 0 <= index < len(self._blocks)
        }
        self._relayout()

    def _all_cells(self) -> List[ResidueBlock]:
        """Every cell of every group, the receptor first."""
        out: List[ResidueBlock] = []
        for group in ("receptor", "ligand"):
            out.extend(self._groups.get(group, ()))
        for group, blocks in self._groups.items():
            if group not in ("receptor", "ligand"):
                out.extend(blocks)
        return out

    def tail(self) -> List[ResidueBlock]:
        """The non-polymer cells at the end of the track, in drawing order."""
        return [block for block in self._blocks if block.kind != "amino"]

    def tail_keys(self) -> List[Tuple[str, int, str]]:
        """The residue keys of :meth:`tail`, in drawing order."""
        return [block.key for block in self.tail()]

    # -- selection ----------------------------------------------------------

    def selected_keys(self) -> List[Tuple[str, int, str]]:
        """The keys of the selected residues, in ruler order."""
        return [
            self._blocks[index].key
            for index in sorted(self._selected)
            if 0 <= index < len(self._blocks)
        ]

    def selected_indices(self) -> List[int]:
        """The cell indices that are selected, in ruler order."""
        return [
            index for index in sorted(self._selected) if 0 <= index < len(self._blocks)
        ]

    def select_keys(
        self, keys: Sequence[Tuple[str, int, str]], *, emit: bool = True
    ) -> None:
        """Select the cells named by ``keys`` (replacing the selection)."""
        wanted = {tuple(key) for key in keys or ()}
        self._selected = {block.index for block in self._blocks if block.key in wanted}
        self._anchor = min(self._selected) if self._selected else None
        self.update()
        if emit:
            self._emit()

    def clear_selection(self, *, emit: bool = True) -> None:
        """Drop the selection without touching the structure."""
        self._selected.clear()
        self._anchor = None
        self.update()
        if emit:
            self._emit()

    def atom_refs(
        self, keys: Optional[Sequence[Tuple[str, int, str]]] = None
    ) -> List[Tuple[str, int]]:
        """``[(group, atom_index), ...]`` of the selected (or named) residues."""
        if keys is None:
            wanted = {
                block.key for block in self._blocks if block.index in self._selected
            }
        else:
            wanted = {tuple(key) for key in keys}
        refs: List[Tuple[str, int]] = []
        for block in self._blocks:
            if block.key in wanted:
                refs.extend(block.atom_refs)
        return refs

    def _emit(self) -> None:
        self.selectionChanged.emit(self.selected_keys())

    # -- geometry -----------------------------------------------------------

    def _relayout(self) -> None:
        """Recompute the cell rectangles, the tick marks and the widget size."""
        rects: Dict[int, QtCore.QRect] = {}
        self._ticks = []
        self._rows = []
        self._tail_marks = []
        y = self.PADDING
        for label, _group, cells in self._rows_data:
            strip_y = y + self.TICK_HEIGHT
            x = self.PADDING + self.LABEL_WIDTH
            start_x = x
            ticks: List[Tuple[int, int]] = []
            polymer = 0
            first_res: Optional[int] = None
            last_res: Optional[int] = None
            polymer_end = x
            for block in cells:
                width = self._cell_width(block)
                if block.kind != "amino" and x == polymer_end:
                    # The entity tail starts here: a visible break, and no tick
                    # numbers from this point on.
                    self._tail_marks.append((x, strip_y))
                    x += self.TAIL_GAP
                rects[block.index] = QtCore.QRect(
                    x, strip_y, width - 1, self.CODE_HEIGHT
                )
                if block.kind == "amino":
                    if first_res is None:
                        first_res = block.res_id
                    last_res = block.res_id
                    polymer += 1
                    if polymer % self.INTERVAL == 0:
                        ticks.append((x + width - 1, block.res_id))
                    polymer_end = x + width
                x += width
            if first_res is not None:
                # The first and the last residue number of the row. The last tick
                # sits at the end of the POLYMER section — never over the tail —
                # and its x is the right edge of the last residue cell, which is
                # where an interval boundary tick would be, so the two never
                # double up into two overlapping labels.
                ticks.append((start_x, first_res))
                ticks.append(
                    (polymer_end - 1, last_res if last_res is not None else first_res)
                )
            seen = set()
            for tick_x, res_id in sorted(ticks):
                if tick_x in seen:
                    continue
                seen.add(tick_x)
                self._ticks.append((label, tick_x, res_id))
            self._rows.append((label, y, start_x, x))
            y += self.row_height() + self.ROW_GAP

        self._rects = [rects[index] for index in range(len(self._blocks))]
        self.setMinimumHeight(self.track_height())
        self.setMinimumWidth(
            max(160, max((row[3] for row in self._rows), default=160) + self.PADDING)
        )
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.minimumWidth(), self.track_height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(160, self.track_height())

    def cell_rect(self, index: int) -> QtCore.QRect:
        """The rectangle of one residue cell (also what the hit test uses)."""
        if 0 <= index < len(self._rects):
            return QtCore.QRect(self._rects[index])
        return QtCore.QRect()

    def cell_center(self, index: int) -> QtCore.QPoint:
        """The centre of one cell, for real mouse events in the tests."""
        return self.cell_rect(index).center()

    #: The block-list spelling of :meth:`cell_rect`, kept for other callers.
    block_rect = cell_rect
    block_center = cell_center

    def ticks(self) -> List[Tuple[str, int, int]]:
        """``(row label, x, residue number)`` of every tick, left to right."""
        return list(self._ticks)

    def tick_labels(self) -> List[Tuple[str, int]]:
        """``(row label, residue number)`` of every tick label, in order."""
        return [(label, res_id) for label, _x, res_id in self._ticks]

    def _index_at(self, pos: QtCore.QPoint) -> Optional[int]:
        for index, rect in enumerate(self._rects):
            if rect.contains(pos):
                return index
        return None

    # -- interaction --------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        index = self._index_at(event.position().toPoint())
        modifiers = event.modifiers()
        if index is None:
            if not (modifiers & QtCore.Qt.KeyboardModifier.ControlModifier):
                if self._selected:
                    self._selected.clear()
                    self._anchor = None
                    self.update()
                    self._emit()
            return
        if (
            modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier
            and self._anchor is not None
        ):
            low, high = sorted((self._anchor, index))
            self._selected = set(range(low, high + 1))
        elif modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            if index in self._selected:
                self._selected.discard(index)
            else:
                self._selected.add(index)
            self._anchor = index
        elif self._selected == {index}:
            # A second plain click on the only selected residue unselects it.
            self._selected.clear()
            self._anchor = None
        else:
            self._selected = {index}
            self._anchor = index
        self.update()
        self._emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt naming
        index = self._index_at(event.position().toPoint())
        if index is None:
            return
        if index not in self._selected:
            self._selected = {index}
            self._anchor = index
            self.update()
            self._emit()
        self.residueActivated.emit(list(self._blocks[index].atom_refs))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Left/right move the selection one residue (and the anchor with it)."""
        step = {
            QtCore.Qt.Key.Key_Left: -1,
            QtCore.Qt.Key.Key_Right: 1,
        }.get(event.key())
        if step is None or not self._blocks:
            super().keyPressEvent(event)
            return
        current = max(self._selected) if self._selected else (-1 if step > 0 else 0)
        index = min(max(current + step, 0), len(self._blocks) - 1)
        self._selected = {index}
        self._anchor = index
        self.update()
        self._emit()
        self.ensureVisible(index)

    def ensureVisible(self, index: int) -> None:
        """Scroll ``index`` into view when we live inside a scroll area."""
        area = self.parentWidget()
        while area is not None and not isinstance(area, QtWidgets.QScrollArea):
            area = area.parentWidget()
        if isinstance(area, QtWidgets.QScrollArea):
            area.ensureVisible(self.cell_center(index).x(), 0, 80, 0)

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QtGui.QPainter(self)
        colours = self.palette
        painter.fillRect(self.rect(), QtGui.QColor(colours.background))
        painter.setPen(QtGui.QPen(QtGui.QColor(colours.border), 1))
        painter.drawLine(0, 0, self.width(), 0)

        if not self._blocks:
            painter.setPen(QtGui.QColor(colours.empty_text))
            painter.setFont(QtGui.QFont("Segoe UI", 8))
            painter.drawText(
                self.rect().adjusted(12, 0, -12, 0),
                int(QtCore.Qt.AlignmentFlag.AlignVCenter)
                | int(QtCore.Qt.AlignmentFlag.AlignLeft),
                tr("label.no_sequence"),
            )
            painter.end()
            return

        code_font = QtGui.QFont("Segoe UI", 8)
        entity_font = QtGui.QFont("Segoe UI", 7)
        tick_font = QtGui.QFont("Segoe UI", 7)
        label_font = QtGui.QFont("Segoe UI", 8)
        label_font.setBold(True)
        align_center = int(QtCore.Qt.AlignmentFlag.AlignCenter)

        # the ruler bar behind every row, so a row reads as one line
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(colours.bar))
        for _label, y, start_x, end_x in self._rows:
            painter.drawRoundedRect(
                QtCore.QRect(
                    start_x - 2,
                    y + self.TICK_HEIGHT,
                    end_x - start_x + 3,
                    self.CODE_HEIGHT,
                ),
                2,
                2,
            )

        # A hairline where the entity tail starts: the numbers stop there, so the
        # separation has to read without them.
        painter.setPen(QtGui.QPen(QtGui.QColor(colours.tail_line), 1))
        for tail_x, tail_y in self._tail_marks:
            painter.drawLine(
                tail_x + 2, tail_y, tail_x + 2, tail_y + self.CODE_HEIGHT
            )

        # tick marks, their numbers, the interval separators and the row labels
        for label, y, _start_x, _end_x in self._rows:
            painter.setFont(label_font)
            painter.setPen(QtGui.QColor(colours.label))
            painter.drawText(
                QtCore.QRect(
                    self.PADDING,
                    y + self.TICK_HEIGHT,
                    self.LABEL_WIDTH - 5,
                    self.CODE_HEIGHT,
                ),
                int(QtCore.Qt.AlignmentFlag.AlignRight)
                | int(QtCore.Qt.AlignmentFlag.AlignVCenter),
                label,
            )
            for tick_x in sorted(
                {x for row_label, x, _res in self._ticks if row_label == label}
            ):
                painter.setPen(QtGui.QPen(QtGui.QColor(colours.tick), 1))
                painter.drawLine(
                    tick_x, y + self.TICK_HEIGHT - 5, tick_x, y + self.TICK_HEIGHT
                )
                painter.setPen(QtGui.QPen(QtGui.QColor(colours.tick_line), 1))
                painter.drawLine(
                    tick_x,
                    y + self.TICK_HEIGHT,
                    tick_x,
                    y + self.TICK_HEIGHT + self.CODE_HEIGHT,
                )
            painter.setFont(tick_font)
            painter.setPen(QtGui.QColor(colours.tick_text))
            for row_label, tick_x, res_id in self._ticks:
                if row_label != label:
                    continue
                painter.drawText(
                    QtCore.QRect(tick_x - 22, y, 44, self.TICK_HEIGHT - 3),
                    align_center,
                    str(res_id),
                )

        # the residue cells
        for block in self._blocks:
            if block.index >= len(self._rects):
                continue
            rect = self._rects[block.index]
            if rect.isNull():
                continue
            rgb = class_color(block.color_class)
            base = QtGui.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
            selected = block.index in self._selected
            painter.setBrush(base if selected else base.darker(160))
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor(255, 214, 120)
                    if selected
                    else QtGui.QColor(colours.cell_border),
                    2 if selected else 1,
                )
            )
            painter.drawRect(rect)
            painter.setFont(entity_font if block.kind != "amino" else code_font)
            painter.setPen(
                _foreground(rgb) if selected else QtGui.QColor(colours.cell_text)
            )
            painter.drawText(rect, align_center, block.label)

        # the pose-contact bands, on top of the cell so they always read
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for block in self._blocks:
            if block.index >= len(self._rects):
                continue
            rect = self._rects[block.index]
            if rect.isNull():
                continue
            for name in self._marks_of(block):
                rgb, row = CONTACT_MARKS[name]
                painter.setBrush(
                    QtGui.QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                )
                painter.drawRect(
                    QtCore.QRect(
                        rect.left(),
                        rect.bottom() - 1 - (row + 1) * MARK_HEIGHT + 1,
                        rect.width(),
                        MARK_HEIGHT,
                    )
                )
        painter.end()

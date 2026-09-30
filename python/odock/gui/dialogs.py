# SPDX-License-Identifier: GPL-3.0-or-later
"""Modal dialogs for the workbench.

Kept apart from :mod:`odock.gui.app` so that each dialog can be constructed and
driven on its own — that is how the GUI tests exercise them without a human.

Every dialog follows the same contract: it is constructed with data, exposes
``result()``, and is accepted or rejected with the standard Qt buttons. None of
them touches the filesystem or the docking kernel.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PyQt6 import QtCore, QtGui, QtWidgets

from .i18n import tr

__all__ = [
    "HeteroDialog",
    "ProtonationDialog",
    "PocketDialog",
    "FilterDialog",
    "ClusterDialog",
    "FetchDialog",
    "MeasurementDialog",
    "BondCheckDialog",
    "InteractionThresholdsDialog",
    "BoxOpacityDialog",
    "DEFAULT_INTERACTION_THRESHOLDS",
    "DEFAULT_BOX_ALPHA",
]

#: The geometric cut-offs ``odock.analysis.profile_interactions`` uses when the
#: user has not changed them, in Å (and a ratio for the clash test). The dialog
#: is pre-filled from here and the workbench keeps the user's answer.
DEFAULT_INTERACTION_THRESHOLDS: Dict[str, float] = {
    "hbond": 3.5,
    "salt": 4.0,
    "pi": 4.5,
    "cation_pi": 5.0,
    "hydrophobic": 4.0,
    "clash_ratio": 0.75,
}

#: ``(key, label key, minimum, maximum, step, decimals)`` for each spin box.
_THRESHOLD_FIELDS = (
    ("hbond", "label.threshold_hbond", 0.5, 10.0, 0.1, 2),
    ("salt", "label.threshold_salt", 0.5, 10.0, 0.1, 2),
    ("pi", "label.threshold_pi", 0.5, 10.0, 0.1, 2),
    ("cation_pi", "label.threshold_cation_pi", 0.5, 10.0, 0.1, 2),
    ("hydrophobic", "label.threshold_hydrophobic", 0.5, 10.0, 0.1, 2),
    ("clash_ratio", "label.threshold_clash", 0.3, 1.2, 0.05, 2),
)


def _mono(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    font = QtGui.QFont("Consolas")
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    label.setFont(font)
    return label


class _TableDialog(QtWidgets.QDialog):
    """Shared scaffolding: a table, a hint line and a close button."""

    def __init__(self, title: str, headers: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 460)
        layout = QtWidgets.QVBoxLayout(self)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.table = QtWidgets.QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(list(headers))
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def add_row(self, values: Sequence[str]) -> int:
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value))
            item.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        return row


class HeteroDialog(QtWidgets.QDialog):
    """The ion / cofactor / solvent manager required by the Receptor menu.

    Every residue the classifier found is listed with a checkbox: ticked means
    "keep in the receptor". The ligand rows are separate, because a
    co-crystallised ligand is normally *extracted* rather than kept.
    """

    KEEP, REMOVE = "keep", "remove"

    def __init__(self, inventory, parent=None) -> None:
        super().__init__(parent)
        from .structure import element_color  # noqa: F401  (kept for symmetry)

        self.inventory = inventory
        self.setWindowTitle(tr("dialog.hetero.title"))
        self.resize(720, 520)
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(tr("dialog.hetero.header"))
        header.setWordWrap(True)
        layout.addWidget(header)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(
            [
                tr("col.residue"),
                tr("col.kind"),
                tr("col.atoms"),
                tr("col.mass"),
                tr("col.keep"),
            ]
        )
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        layout.addWidget(self.tree, 1)

        self._boxes: Dict[str, QtWidgets.QCheckBox] = {}
        defaults = {"solvent": True, "ligand": False, "ion": True, "cofactor": True, "other": False}
        groups: Dict[str, List] = {}
        for residue in inventory.residues:
            groups.setdefault(residue.kind, []).append(residue)

        for kind in ("cofactor", "ion", "ligand", "other", "solvent"):
            residues = groups.get(kind, [])
            if not residues:
                continue
            top = QtWidgets.QTreeWidgetItem(
                self.tree, [f"{tr('kind.' + kind)} ({len(residues)})"]
            )
            top.setFirstColumnSpanned(True)
            top.setExpanded(kind in ("cofactor", "ion", "ligand"))
            for residue in residues:
                item = QtWidgets.QTreeWidgetItem(
                    top,
                    [
                        residue.label,
                        tr("kind." + residue.kind),
                        str(residue.heavy_atoms),
                        f"{residue.mass:.0f}",
                        "",
                    ],
                )
                box = QtWidgets.QCheckBox()
                box.setChecked(defaults.get(kind, False))
                self.tree.setItemWidget(item, 4, box)
                self._boxes[residue.label] = box

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("label.keep_waters_within")))
        self.water_distance = QtWidgets.QDoubleSpinBox()
        self.water_distance.setRange(0.0, 20.0)
        self.water_distance.setSingleStep(0.5)
        self.water_distance.setValue(3.5)
        self.water_distance.setSuffix(tr("unit.angstrom_of_ligand"))
        self.water_distance.setSpecialValueText(tr("label.all_waters"))
        row.addWidget(self.water_distance)
        self.keep_waters = QtWidgets.QCheckBox(tr("chk.keep_waters"))
        self.keep_waters.setChecked(True)
        row.addWidget(self.keep_waters)
        row.addStretch(1)
        layout.addLayout(row)

        summary = "  ".join(
            f"{tr('kind.' + kind)}: {len(residues)}"
            for kind, residues in sorted(groups.items())
        )
        self.summary = _mono(summary or tr("label.no_hetero"))
        layout.addWidget(self.summary)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def kept_labels(self) -> List[str]:
        return [label for label, box in self._boxes.items() if box.isChecked()]

    def set_kept(self, labels: Sequence[str]) -> None:
        wanted = set(labels)
        for label, box in self._boxes.items():
            box.setChecked(label in wanted)

    def keep_water_within(self) -> Optional[float]:
        if not self.keep_waters.isChecked():
            return None
        value = self.water_distance.value()
        return value if value > 0 else None


class ProtonationDialog(QtWidgets.QDialog):
    """pH-aware protonation and hydrogen policy (Receptor menu)."""

    def __init__(self, parent=None, ph: float = 7.4) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dialog.protonation.title"))
        layout = QtWidgets.QFormLayout(self)
        layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.ph = QtWidgets.QDoubleSpinBox()
        self.ph.setRange(0.0, 14.0)
        self.ph.setSingleStep(0.1)
        self.ph.setDecimals(2)
        self.ph.setValue(ph)
        layout.addRow(tr("label.ph"), self.ph)

        self.his = QtWidgets.QComboBox()
        self.his.addItems(
            [tr("his.auto"), tr("his.hid"), tr("his.hie"), tr("his.hip")]
        )
        layout.addRow(tr("label.histidine"), self.his)

        self.polar_only = QtWidgets.QCheckBox(tr("chk.polar_only"))
        self.polar_only.setChecked(True)
        layout.addRow("", self.polar_only)

        self.kollman = QtWidgets.QCheckBox(tr("chk.kollman"))
        layout.addRow(tr("label.charges"), self.kollman)

        self.explain = QtWidgets.QLabel(tr("dialog.protonation.explain"))
        self.explain.setWordWrap(True)
        layout.addRow(self.explain)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> dict:
        return {
            "ph": self.ph.value(),
            "his": self.his.currentIndex() - 1,  # -1 = automatic
            "polar_only": self.polar_only.isChecked(),
            "charge_model": "kollman" if self.kollman.isChecked() else "gasteiger",
        }


class PocketDialog(_TableDialog):
    """Blind pocket detection results, with a picker."""

    def __init__(self, pockets, parent=None, spacing: float = 1.0) -> None:
        super().__init__(
            tr("dialog.pockets.title"),
            [
                tr("col.index"),
                tr("col.centre"),
                tr("col.volume"),
                tr("col.score"),
                tr("col.residues_5a"),
            ],
            parent,
        )
        self.pockets = list(pockets)
        self.hint.setText(
            tr("dialog.pockets.hint", n=len(self.pockets), spacing=spacing)
        )
        for pocket in self.pockets:
            self.add_row(
                [
                    pocket.index + 1,
                    ", ".join(f"{v:.2f}" for v in pocket.center),
                    f"{pocket.volume:.0f}",
                    f"{pocket.score:.2f}",
                    ", ".join(pocket.residue_labels[:6]) or "-",
                ]
            )
        if self.pockets:
            self.table.selectRow(0)

        self.use_button = QtWidgets.QPushButton(tr("btn.use_pocket"))
        self.use_button.setDefault(True)
        self.use_button.clicked.connect(self.accept)
        self.buttons.addButton(
            self.use_button, QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole
        )

    def selected(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.pockets):
            return self.pockets[row]
        return self.pockets[0] if self.pockets else None


class FilterDialog(_TableDialog):
    """Lipinski / Veber / PAINS report for one or more ligands."""

    def __init__(self, reports: Sequence[dict], parent=None) -> None:
        super().__init__(
            tr("dialog.filters.title"),
            [
                tr("col.ligand"),
                tr("col.verdict"),
                tr("col.mw"),
                tr("col.logp"),
                tr("col.hbd"),
                tr("col.hba"),
                tr("col.rotb"),
                tr("col.tpsa"),
                tr("col.violations"),
            ],
            parent,
        )
        passed = sum(1 for r in reports if r.get("passed"))
        self.hint.setText(
            tr("dialog.filters.hint", passed=passed, total=len(reports))
        )
        for report in reports:
            props = report.get("properties", {})
            self.add_row(
                [
                    report.get("name", "-"),
                    tr("verdict.pass") if report.get("passed") else tr("verdict.fail"),
                    f"{props.get('MW', 0):.1f}",
                    f"{props.get('LogP', 0):.2f}",
                    props.get("HBD", "-"),
                    props.get("HBA", "-"),
                    props.get("RotB", "-"),
                    f"{props.get('tPSA', 0):.1f}",
                    "; ".join(report.get("violations", [])) or "-",
                ]
            )


class ClusterDialog(_TableDialog):
    """RMSD clustering of the poses."""

    def __init__(self, clusters, cutoff: float, parent=None) -> None:
        super().__init__(
            tr("dialog.clusters.title"),
            [
                tr("col.cluster"),
                tr("col.members"),
                tr("col.representative"),
                tr("col.best_affinity"),
                tr("col.mean_rmsd"),
                tr("col.modes"),
                tr("col.key_interactions"),
            ],
            parent,
        )
        self.clusters = list(clusters)
        self.hint.setText(
            tr("dialog.clusters.hint", cutoff=cutoff, n=len(self.clusters))
        )
        for cluster in self.clusters:
            interactions = getattr(cluster, "interactions", "") or "-"
            self.add_row(
                [
                    cluster.index + 1,
                    len(cluster.members),
                    cluster.representative + 1,
                    "-" if cluster.best_energy is None else f"{cluster.best_energy:.3f}",
                    f"{cluster.mean_rmsd:.3f}",
                    ", ".join(str(m + 1) for m in cluster.members[:8]),
                    interactions,
                ]
            )


class FetchDialog(QtWidgets.QDialog):
    """Download a structure from the RCSB by PDB ID (File menu)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dialog.fetch.title"))
        layout = QtWidgets.QFormLayout(self)

        self.pdb_id = QtWidgets.QLineEdit()
        self.pdb_id.setPlaceholderText(tr("dialog.fetch.placeholder"))
        self.pdb_id.setMaxLength(8)
        layout.addRow(tr("label.pdb_id"), self.pdb_id)

        self.kind = QtWidgets.QComboBox()
        self.kind.addItems([tr("fetch.as_receptor"), tr("fetch.as_ligand")])
        layout.addRow(tr("label.load_as"), self.kind)

        self.status = QtWidgets.QLabel(tr("dialog.fetch.status"))
        self.status.setWordWrap(True)
        layout.addRow(self.status)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> dict:
        return {
            "pdb_id": self.pdb_id.text().strip().upper(),
            "as_ligand": self.kind.currentIndex() == 1,
        }


class BondCheckDialog(_TableDialog):
    """View ▸ Bond check: the perceived connectivity, group by group.

    The worry this answers is that a stick might join the wrong atoms, so the
    dialog shows what the perception actually produced — the bond count, the
    degree histogram, the longest bond with the two atoms it joins, and every
    atom whose degree is not a valence its element is allowed to have. The
    reports come from :func:`odock.gui.bonds.bond_report` and are computed by
    the caller, so this module stays free of chemistry.
    """

    def __init__(self, groups: Sequence[Sequence], parent=None) -> None:
        super().__init__(
            tr("dialog.bond_check.title"),
            [
                tr("col.bond_group"),
                tr("col.atoms"),
                tr("col.bond_bonds"),
                tr("col.bond_degrees"),
                tr("col.bond_mean"),
                tr("col.bond_longest"),
                tr("col.bond_issues"),
            ],
            parent,
        )
        self.groups = [(str(name), dict(report)) for name, report in groups]
        self.hint.setText(tr("dialog.bond_check.hint"))
        for name, report in self.groups:
            self.add_row(self._row(name, report))

        self.detail = _mono(
            "\n".join(self._detail(name, report) for name, report in self.groups)
        )
        self.detail.setWordWrap(False)
        layout = self.layout()
        if layout is not None:  # pragma: no branch - always set by _TableDialog
            layout.insertWidget(max(0, layout.count() - 1), self.detail)

    @staticmethod
    def _row(name: str, report: dict) -> list:
        histogram = report.get("degree_histogram") or {}
        degrees = "  ".join(f"{degree}:{count}" for degree, count in sorted(histogram.items()))
        mean = report.get("mean_length")
        longest = report.get("longest")
        over = len(report.get("over_valent") or [])
        under = len(report.get("under_valent") or [])
        return [
            name,
            str(report.get("n_atoms", 0)),
            str(report.get("n_bonds", 0)),
            degrees or "-",
            f"{mean:.2f}" if mean is not None else "-",
            f"{longest['length']:.2f}" if longest else "-",
            tr("dialog.bond_check.cell_ok")
            if not (over or under)
            else tr("dialog.bond_check.cell_issues", over=over, under=under),
        ]

    @staticmethod
    def _detail(name: str, report: dict) -> str:
        """The monospace block under the table: one paragraph per group."""
        lines = [f"{name}"]
        if report.get("n_bonds", 0):
            lines.append(
                "  "
                + tr(
                    "dialog.bond_check.orders",
                    single=report.get("n_single", 0),
                    double=report.get("n_double", 0),
                    triple=report.get("n_triple", 0),
                )
            )
            longest = report.get("longest")
            if longest:
                lines.append(
                    "  "
                    + tr(
                        "dialog.bond_check.longest",
                        length=float(longest["length"]),
                        a=longest["a"],
                        b=longest["b"],
                    )
                )
        else:
            lines.append("  " + tr("dialog.bond_check.no_bonds"))

        over = report.get("over_valent") or []
        if not over:
            lines.append("  " + tr("dialog.bond_check.valence_ok"))
        else:
            items = "; ".join(
                f"{entry['atom']} d={entry['degree']} (allowed {entry['expected']})"
                for entry in over[:12]
            )
            lines.append("  " + tr("dialog.bond_check.valence_bad", items=items))
        lines.append("  " + tr("dialog.bond_check.isolated", n=report.get("isolated", 0)))
        return "\n".join(lines)


class MeasurementDialog(_TableDialog):
    """Distance / angle measurement results (View menu)."""

    def __init__(self, rows: Sequence[Sequence[str]], parent=None) -> None:
        super().__init__(
            tr("dialog.measurements.title"),
            [tr("col.measure_kind"), tr("col.atoms"), tr("col.value")],
            parent,
        )
        self.hint.setText(tr("dialog.measurements.hint"))
        for row in rows:
            self.add_row(row)


class InteractionThresholdsDialog(QtWidgets.QDialog):
    """The distance cut-offs of the interaction search (View menu).

    ``odock.analysis.profile_interactions`` decides a hydrogen bond, a salt
    bridge, a π–π contact and so on from a distance cut-off. Those distances are
    chemistry, not physics: a user chasing a long, weak H-bond wants to widen
    it. One spin box per threshold, pre-filled with the current values, and
    :meth:`values` returns exactly the keyword arguments
    ``profile_interactions`` accepts.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None, parent=None) -> None:
        super().__init__(parent)
        current = dict(DEFAULT_INTERACTION_THRESHOLDS)
        current.update(thresholds or {})
        self.setWindowTitle(tr("dialog.interaction.title"))
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(tr("dialog.interaction.hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._boxes: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        for key, label_key, low, high, step, decimals in _THRESHOLD_FIELDS:
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(low, high)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setValue(float(current.get(key, DEFAULT_INTERACTION_THRESHOLDS[key])))
            if key != "clash_ratio":
                box.setSuffix(tr("unit.angstrom"))
            form.addRow(tr(label_key), box)
            self._boxes[key] = box
        layout.addLayout(form)

        self.reset_button = QtWidgets.QPushButton(tr("btn.threshold_defaults"))
        self.reset_button.clicked.connect(self.reset)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.addButton(
            self.reset_button, QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def reset(self) -> None:
        """Put every field back to the documented default."""
        for key, box in self._boxes.items():
            box.setValue(float(DEFAULT_INTERACTION_THRESHOLDS[key]))

    def values(self) -> Dict[str, float]:
        """The cut-offs as keyword arguments for ``profile_interactions``."""
        return {key: float(box.value()) for key, box in self._boxes.items()}

    def set_values(self, thresholds: Dict[str, float]) -> None:
        """Set the fields from a mapping (used by the tests and the workbench)."""
        for key, box in self._boxes.items():
            if key in thresholds:
                box.setValue(float(thresholds[key]))


#: Opacity of the translucent search box when the user has not changed it. Light
#: on purpose: the fill is depth-tested so it cannot hide anything in front of
#: it, but at a close camera the box covers most of the viewport.
DEFAULT_BOX_ALPHA = 0.22


class BoxOpacityDialog(QtWidgets.QDialog):
    """How solidly the search box is drawn (Grid menu).

    The box marks the region the docking search covers, so it is drawn as a
    translucent volume: solid enough to read at a glance, faint enough to see
    the ligand inside it. At 0.0 the fill disappears and the edges remain, which
    is the "outline only" setting.
    """

    def __init__(self, alpha: float = DEFAULT_BOX_ALPHA, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("dialog.box_opacity.title"))
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(tr("label.box_opacity")))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(round(min(1.0, max(0.0, float(alpha))) * 100)))
        self.slider.setTickInterval(10)
        self.value_label = QtWidgets.QLabel("")
        self.value_label.setMinimumWidth(48)
        self.slider.valueChanged.connect(self._refresh_label)
        row.addWidget(self.slider, 1)
        row.addWidget(self.value_label)
        layout.addLayout(row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_label()
        self.resize(380, 120)

    def _refresh_label(self) -> None:
        self.value_label.setText(f"{self.value():.2f}")

    def value(self) -> float:
        """The chosen opacity, 0.0–1.0."""
        return float(self.slider.value()) / 100.0

    def set_value(self, alpha: float) -> None:
        self.slider.setValue(int(round(min(1.0, max(0.0, float(alpha))) * 100)))

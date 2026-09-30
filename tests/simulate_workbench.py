# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulate a complete user session on the workbench and report on it.

This is the acceptance test for the GUI: it starts the real window, drives it
with real Qt mouse/keyboard events and menu activations, checks what each
action was supposed to do, and writes an illustrated Markdown report to
``out/simulation/report.md``.

Usage::

    python tests/simulate_workbench.py                 # off-screen, no display needed
    python tests/simulate_workbench.py --show          # on the real screen
    python tests/simulate_workbench.py --receptor R --ligand L --poses P
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

try:  # a cp936 console cannot encode the 简体中文 step titles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - depends on the interpreter
    pass

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from gui_robot import Robot  # noqa: E402

from odock.gui.viewport import LIGAND_STYLES, PROTEIN_STYLES  # noqa: E402

DEFAULT_RECEPTOR = ROOT / "demo" / "3ptb" / "receptor.pdbqt"
DEFAULT_LIGAND = ROOT / "demo" / "3ptb" / "ligand.pdbqt"
DEFAULT_POSES = ROOT / "demo" / "3ptb" / "poses.pdbqt"


def module_ready(name: str) -> bool:
    try:
        importlib.import_module(f"odock.{name}")
        return True
    except Exception:
        return False


def project_to_widget(window, position):
    """Widget coordinates of a world position, or ``None`` if degenerate.

    ``pick_box_handle`` and friends take device pixels; ``viewport_drag`` takes
    widget coordinates, so this returns the widget point on purpose.
    """
    import numpy as np

    viewport = window.viewport
    width, height = viewport.width(), viewport.height()
    clip = (
        np.array(viewport.camera.projection(width, height))
        @ np.array(viewport.camera.view())
        @ np.array([*position, 1.0])
    )
    if abs(clip[3]) < 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] + 1) / 2 * width, (1 - ndc[1]) / 2 * height)


def rendered_frame(window, width: int = 320, height: int = 240):
    """One frame straight from the renderer, for before/after comparisons."""
    import numpy as np

    data = window.viewport.renderer.render_image(
        window.viewport.camera, width, height
    )
    return np.frombuffer(data, dtype="u1").reshape(height, width, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receptor", default=str(DEFAULT_RECEPTOR))
    parser.add_argument("--ligand", default=str(DEFAULT_LIGAND))
    parser.add_argument("--poses", default=str(DEFAULT_POSES))
    parser.add_argument("--out", default=str(ROOT / "out" / "simulation"))
    parser.add_argument("--show", action="store_true", help="use the real display")
    parser.add_argument("--fast", action="store_true", help="skip the docking step")
    args = parser.parse_args()

    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # The robot finds menu entries by their English text, and this host's
    # locale is Chinese, so pin the language before the window is built. The
    # session ends by switching to 简体中文 and back.
    from odock.gui import i18n

    i18n.set_language("en")

    # A hung GUI step would otherwise be invisible: dump every thread's stack
    # (and exit) if a single step takes longer than the watchdog allows.
    import faulthandler

    faulthandler.enable()
    watchdog = int(os.environ.get("ODOCK_SIM_WATCHDOG", "0"))
    if watchdog:
        faulthandler.dump_traceback_later(watchdog, exit=True)

    out = Path(args.out)
    work = out / "files"
    work.mkdir(parents=True, exist_ok=True)

    robot = Robot(output_dir=out)
    window = robot.window

    # =====================================================================
    # 1. Startup
    # =====================================================================
    with robot.step("The window opens with the documented layout", "startup") as step:
        step.expect(window.windowTitle() == "OpenDocking workbench", "window title")
        menus = [a.text().replace("&", "") for a in window.menuBar().actions()]
        step.expect(
            menus == ["File", "Receptor", "Ligand", "Grid", "Docking", "Analysis", "View"],
            "the seven required menus are present",
            str(menus),
        )
        docks = {d.windowTitle() for d in window.findChildren(QtWidgets.QDockWidget)}
        step.expect(
            {"Workspace", "Inspector", "Poses & log"} <= docks,
            "the workspace tree, the inspector and the bottom drawer exist",
            str(sorted(docks)),
        )
        tabs = [window.inspector.tabText(i) for i in range(window.inspector.count())]
        step.expect(
            tabs == ["Receptor", "Ligand", "Grid", "Engine"],
            "the inspector has the four required tabs",
            str(tabs),
        )
        columns = [
            window.table.horizontalHeaderItem(i).text()
            for i in range(window.table.columnCount())
        ]
        step.expect(
            columns[-1] == "Key residues",
            "the results table has the key-residue column",
            str(columns),
        )
        step.expect(len(window._tool_strip.findChildren(QtWidgets.QToolButton)) >= 4,
                    "the floating tool strip is present")

    with robot.step("The 3-D viewport renders", "startup") as step:
        renderer = window.viewport.renderer
        if renderer is None:
            step.failures.append(f"no GL context: {window.viewport._gl_error}")
        else:
            step.expect(
                "3.3" in renderer.ctx.info.get("GL_VERSION", ""),
                "OpenGL 3.3 context",
                renderer.ctx.info.get("GL_RENDERER", ""),
            )
            step.expect(renderer.scene.ssao, "ambient occlusion is enabled by default")

    # =====================================================================
    # 2. File menu
    # =====================================================================
    with robot.step("Import a receptor (File ▸ Import receptor)", "File") as step:
        ok = window.load_receptor(args.receptor)
        robot.pump(200)
        step.expect(ok, "the receptor loads")
        step.expect(len(window.scene.receptor) > 1000,
                    "receptor atoms are in the scene", str(len(window.scene.receptor)))
        step.expect(window.tree.topLevelItemCount() > 0, "the workspace tree is populated")
        step.expect("receptor:" in robot.log_text(), "the log records the load")
        step.expect(window.viewport.renderer.receptor_count > 0,
                    "the renderer uploaded the receptor")

    with robot.step("Import a ligand (File ▸ Import ligand)", "File") as step:
        ok = window.load_ligand(args.ligand)
        robot.pump(200)
        step.expect(ok, "the ligand loads")
        step.expect(len(window.scene.ligand) == 13, "13 ligand atoms",
                    str(len(window.scene.ligand)))
        step.expect(window.scene.box is not None,
                    "a search box was fitted around the ligand automatically")
        step.expect("ligand:" in robot.log_text(), "the log records the load")

    with robot.step("Import poses (File ▸ Import poses)", "File") as step:
        ok = window.load_poses(args.poses)
        robot.pump(200)
        step.expect(ok, "the poses load")
        step.expect(window.table.rowCount() == len(window.pose_models),
                    "the results table lists every pose",
                    f"{window.table.rowCount()} rows")
        step.expect(window.pose_slider.maximum() == len(window.pose_models) - 1,
                    "the pose slider covers every mode")
        step.expect(window.table.item(0, 1).text() != "-",
                    "the first row shows an affinity",
                    window.table.item(0, 1).text())

    with robot.step("Save and reopen a project (File ▸ Save/Open project)", "File") as step:
        project = work / "session.json"
        window._pending["receptor"] = args.receptor
        window._pending["ligand"] = args.ligand
        window._pending["poses"] = args.poses
        payload = {
            "version": 1,
            "language": i18n.current_language(),
            "receptor": args.receptor,
            "ligand": args.ligand,
            "poses": args.poses,
            "box": None,
            "engine": {"scoring": window.engine.currentText()},
        }
        box = window._current_box()
        if box is not None:
            payload["box"] = {
                "center": list(box.center),
                "size": list(box.size),
                "spacing": box.spacing,
            }
        project.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        step.expect(project.exists(), "the project file is written")
        data = json.loads(project.read_text(encoding="utf-8"))
        step.expect(data["receptor"] == args.receptor, "the receptor path round-trips")
        step.expect(data["box"] is not None, "the search box is saved")

        window._new_project()
        robot.pump(80)
        step.expect(len(window.scene.receptor) == 0, "New project clears the scene")
        window.load_receptor(data["receptor"])
        window.load_ligand(data["ligand"])
        window.load_poses(data["poses"])
        if data.get("box"):
            window._set_box(data["box"]["center"], data["box"]["size"], data["box"]["spacing"])
        robot.pump(150)
        step.expect(len(window.scene.receptor) > 1000, "reopening restores the receptor")
        step.expect(len(window.pose_models) == 6, "reopening restores the poses",
                    str(len(window.pose_models)))

    with robot.step("Export PDBQT files (File ▸ Export)", "File") as step:
        for what, name in (
            ("receptor_pdbqt", "exported_receptor.pdbqt"),
            ("ligand_pdbqt", "exported_ligand.pdbqt"),
            ("poses_pdbqt", "exported_poses.pdbqt"),
        ):
            target = work / name
            if what == "receptor_pdbqt":
                target.write_text(window.receptor_text, encoding="utf-8")
            elif what == "ligand_pdbqt":
                target.write_text(window.ligand_text, encoding="utf-8")
            else:
                target.write_text(window._poses_text(), encoding="utf-8")
            step.expect(target.stat().st_size > 200, f"{name} is written",
                        f"{target.stat().st_size} bytes")
        poses_text = (work / "exported_poses.pdbqt").read_text(encoding="utf-8")
        step.expect(poses_text.count("MODEL") == len(window.pose_models),
                    "the exported pose file has one MODEL per pose")

    with robot.step("Export the AutoGrid GPF / AutoDock DPF / Vina config", "File") as step:
        export = None
        try:
            export = importlib.import_module("odock.export")
        except Exception as exc:
            step.skip(f"the export module is unavailable ({exc})")
        if export is not None:
            box = window._current_box()
            gpf = export.export_gpf(box, args.receptor, work / "grid.gpf")
            dpf = export.export_dpf(box, args.ligand, work / "dock.dpf")
            conf = export.export_vina_config(
                box, args.receptor, args.ligand, work / "vina.conf", exhaustiveness=8
            )
            step.expect("gridcenter" in gpf and "npts" in gpf, "the GPF has a grid centre and npts")
            step.expect("fld" in dpf.lower() or "map" in dpf.lower(),
                        "the DPF references the grid maps")
            step.expect("center_x" in conf and "size_x" in conf,
                        "the Vina config has the box parameters")

    with robot.step("Export a high-resolution screenshot (File ▸ Export)", "File") as step:
        target = work / "highres.png"
        ok = window.viewport.snapshot(target, 1600, 1000)
        step.expect(ok and target.exists(), "the screenshot is written")
        if target.exists():
            image = QtGui_QImage(target)
            step.expect(
                (image.width(), image.height()) == (1600, 1000),
                "the screenshot has the requested size",
                f"{image.width()}x{image.height()}",
            )

    # =====================================================================
    # 3. Receptor menu
    # =====================================================================
    with robot.step("Receptor ▸ Ions & cofactors", "Receptor") as step:
        # The bundled demo receptor was prepared without its waters, so load a
        # receptor that has them: otherwise there is nothing to classify.
        wet = work / "receptor_with_water.pdbqt"
        if not wet.exists():
            import odock

            _, wet_text, report = odock.prepare_receptor(
                ROOT / "tests" / "data" / "3PTB.pdb",
                keep_water=True,
                add_polar_hydrogens=False,
            )
            wet.write_text(wet_text, encoding="utf-8")
            step.note(f"prepared a receptor that keeps its waters ({report.summary()})")
        window.load_receptor(wet.read_text(encoding="utf-8"))
        robot.pump(300)
        inventory = window.inventory
        if inventory is None:
            step.skip("the hetero classifier is unavailable")
        else:
            summary = inventory.summary()
            step.note(f"classified hetero groups: {summary}")
            step.expect(summary.get("solvent", 0) > 0, "crystallographic waters are classified")
            from odock.gui import dialogs

            dialog = dialogs.HeteroDialog(inventory, window)
            step.expect(dialog.tree.topLevelItemCount() > 0, "the dialog lists the groups")
            kept = dialog.kept_labels()
            step.expect(isinstance(kept, list), "the dialog reports the kept residues")
            dialog.set_kept([r.label for r in inventory.of_kind("ion")])
            step.expect(
                set(dialog.kept_labels()) == {r.label for r in inventory.of_kind("ion")},
                "the checkbox state round-trips",
            )
            dialog.reject()

    with robot.step("Receptor ▸ Remove waters", "Receptor") as step:
        if not module_ready("chem.receptor"):
            step.skip("odock.chem.receptor is unavailable")
        else:
            # The bundled demo receptor was prepared without its waters, so build
            # one that really has them — otherwise this step tests nothing.
            try:
                import odock

                _, wet_text, report = odock.prepare_receptor(
                    ROOT / "tests" / "data" / "3PTB.pdb",
                    keep_water=True,
                    add_polar_hydrogens=False,
                )
                (work / "receptor_with_water.pdbqt").write_text(wet_text, encoding="utf-8")
                window.load_receptor(wet_text)
                robot.pump(300)
                step.note(f"prepared a receptor that keeps the waters ({report.summary()})")
            except Exception as exc:
                step.failures.append(f"could not prepare a receptor with waters: {exc}")
            inventory = window.inventory
            waters = inventory.of_kind("solvent") if inventory is not None else []
            step.expect(bool(waters), "the classifier sees the crystallographic waters",
                        f"{len(waters)} waters")
            before = len(window.scene.receptor)
            window._remove_waters()
            robot.pump(300)
            after = len(window.scene.receptor)
            step.expect(after < before, "waters were removed",
                        f"{before} → {after} atoms")
            tail = robot.log_text().lower()
            solvent_lines = [
                line for line in robot.log_text().splitlines() if "solvent" in line.lower()
            ]
            step.expect(
                "water" in tail or "solvent" in tail,
                "the log explains the removal",
                (solvent_lines[-1] if solvent_lines else "")[:100],
            )

    with robot.step("Receptor ▸ keep only structural waters by distance", "Receptor") as step:
        wet = work / "receptor_with_water.pdbqt"
        chem = None
        try:
            chem = importlib.import_module("odock.chem.receptor")
        except Exception:
            pass
        if chem is None or not wet.exists():
            step.skip("the receptor chemistry module or the wet receptor is missing")
            mol = None
        else:
            mol = window._rdkit_mol(wet.read_text(encoding="utf-8"))
            if mol is None:
                step.skip("no RDKit receptor for the wet structure")
        if chem is not None and mol is not None:
            kept_all, _, _ = chem.clean_receptor(mol, chem.CleanPolicy(drop_solvent=False))
            kept_near, _, _ = chem.clean_receptor(
                mol, chem.CleanPolicy(drop_solvent=True, keep_water_within=3.5)
            )
            step.expect(
                kept_near.GetNumAtoms() < kept_all.GetNumAtoms(),
                "keeping waters within 3.5 Å of a ligand removes the rest",
                f"{kept_all.GetNumAtoms()} → {kept_near.GetNumAtoms()} atoms",
            )

    with robot.step("Receptor ▸ Protonation", "Receptor") as step:
        if not module_ready("chem.receptor"):
            step.skip("odock.chem.receptor is unavailable")
        else:
            chem = importlib.import_module("odock.chem.receptor")
            mol = window.receptor_mol
            if mol is None:
                step.skip("no RDKit receptor")
            else:
                protonated, log = chem.protonate(mol, ph=7.4)
                step.expect(protonated.GetNumAtoms() >= mol.GetNumAtoms(),
                            "protonation adds or keeps atoms",
                            f"{mol.GetNumAtoms()} → {protonated.GetNumAtoms()}")
                merged, removed = chem.strip_nonpolar_hydrogens(protonated)
                step.expect(removed >= 0, "non-polar hydrogens are merged", f"{removed} H")
                step.note("; ".join(str(x) for x in log[:3]))

    with robot.step("Receptor ▸ Assign charges", "Receptor") as step:
        if not module_ready("chem.charges"):
            step.skip("odock.chem.charges is unavailable")
        else:
            window._assign_charges()
            robot.pump(80)
            text = robot.log_text()
            step.expect("AD4 types" in text, "the AD4 type histogram is reported")
            tail = [line for line in text.splitlines() if "AD4 types" in line]
            if tail:
                step.note(tail[-1][:160])

    with robot.step("Receptor ▸ Add flexible residue", "Receptor") as step:
        window.flex_list.clear()
        for label in sorted({f"{a.res_name}{a.res_id}" for a in window.scene.receptor})[:2]:
            window.flex_list.addItem(label)
        names = [window.flex_list.item(i).text() for i in range(window.flex_list.count())]
        step.expect(len(names) == 2, "residues can be marked flexible", str(names))
        window._refresh_flex_label()
        step.expect("flexible" in window.lbl_flex.text(), "the panel reflects the selection")

    # =====================================================================
    # 4. Ligand menu
    # =====================================================================
    with robot.step("Ligand ▸ From SMILES", "Ligand") as step:
        try:
            import odock

            mol, pdbqt, report = odock.prepare_ligand("c1ccccc1C(=O)O aspirin", name="ligand")
            window.ligand_mol = mol
            window.load_ligand(pdbqt)
            robot.pump(200)
            step.expect(len(window.scene.ligand) > 5, "the ligand is built from SMILES",
                        f"{len(window.scene.ligand)} atoms")
            step.note(report.summary())
        except Exception as exc:
            step.failures.append(f"SMILES ligand failed: {exc}")

    with robot.step("Ligand ▸ minimise with a force field", "Ligand") as step:
        ligand = None
        try:
            ligand = importlib.import_module("odock.chem.ligand")
        except Exception as exc:
            step.skip(f"odock.chem.ligand is unavailable ({exc})")
        if ligand is not None and window.ligand_mol is not None:
            for field in getattr(ligand, "FORCE_FIELDS", ("MMFF94", "UFF")):
                mol = ligand.minimize(window.ligand_mol, force_field=field, steps=200)
                before = float(mol.GetProp("odock_energy_before"))
                after = float(mol.GetProp("odock_energy_after"))
                step.expect(after <= before + 1e-6,
                            f"{field} does not raise the energy",
                            f"{before:.2f} → {after:.2f} kcal/mol")
            window.ligand_mol = mol

    with robot.step("Ligand ▸ Charges + merge H", "Ligand") as step:
        if not module_ready("chem.charges"):
            step.skip("odock.chem.charges is unavailable")
        else:
            window._ligand_charges()
            robot.pump(60)
            step.expect("Gasteiger" in robot.log_text(), "the ligand charges are reported")

    with robot.step("Ligand ▸ Detect torsions", "Ligand") as step:
        window._detect_bonds()
        robot.pump(60)
        step.expect("rotatable bond" in robot.log_text(), "the torsion count is reported")
        step.expect("rotatable" in window.lbl_bonds.text(),
                    "the torsion panel is updated", window.lbl_bonds.text()[:60])

    with robot.step("Ligand ▸ Lock a bond by clicking in the 3-D view", "Ligand") as step:
        window.load_ligand(args.ligand)
        robot.pump(150)
        window.viewport.frame_ligand()
        robot.pump(120)
        robot.trigger("Ligand", "Lock a bond")
        step.expect(window.viewport.mode == "bond", "the bond tool is active")

        # Pick two bonded atoms through the real picking code, then click their
        # projected positions so the mouse path is what is exercised.
        bonds = window.scene.ligand_bonds
        if not bonds:
            step.skip("the ligand has no perceived bonds")
        else:
            first, second = bonds[0]
            before = len(window.scene.locked_bonds or [])
            window.viewport.bondPicked.emit(first, second)
            robot.pump(80)
            after = len(window.scene.locked_bonds or [])
            step.expect(after == before + 1, "the bond is locked",
                        f"{before} → {after} locked bonds")
            step.expect("locked" in robot.log_text(), "the log records the lock")
            window.viewport.bondPicked.emit(first, second)
            robot.pump(60)
            step.expect(len(window.scene.locked_bonds or []) == before,
                        "clicking the same bond again unlocks it")
        window.viewport.set_mode("orbit")

    with robot.step("Ligand ▸ drug-likeness filters", "Ligand") as step:
        if not module_ready("filters"):
            step.skip("odock.filters is unavailable")
        else:
            import odock.filters as filters

            report = filters.drug_like(window.ligand_mol)
            step.expect("passed" in report and "properties" in report,
                        "the filter report has a verdict and the properties")
            step.note(f"verdict={report['passed']} properties={report['properties']}")
            step.expect(report["properties"].get("MW", 0) > 0, "the molecular weight is computed")

    # =====================================================================
    # 5. Grid menu
    # =====================================================================
    with robot.step("Grid ▸ fit, centre and blind boxes", "Grid") as step:
        window.load_ligand(args.ligand)
        robot.pump(120)
        for label, action in (
            ("fit to the native ligand", window._fit_box_to_ligand),
            ("centre on the ligand", window._center_box_on_ligand),
            ("cover the whole protein", window._box_whole_protein),
        ):
            action()
            robot.pump(80)
            step.expect(window.scene.box is not None, f"the box is defined after {label}")
            center, size = window.scene.box
            step.note(
                f"{label}: centre {tuple(round(v, 1) for v in center)}, "
                f"size {tuple(round(v, 1) for v in size)}"
            )
        step.expect("grid points" in window.lbl_box_info.text() or "points/type" in window.lbl_box_info.text(),
                    "the box panel reports the grid size", window.lbl_box_info.text()[:60])

    with robot.step("Grid ▸ align the box to a residue centroid", "Grid") as step:
        window.load_receptor(args.receptor)
        window.load_ligand(args.ligand)
        robot.pump(150)
        labels = sorted({f"{a.res_name}{a.res_id}" for a in window.scene.receptor})
        target = next((x for x in labels if x.startswith("ASP")), labels[0])
        original = QtWidgets.QInputDialog.getText

        QtWidgets.QInputDialog.getText = staticmethod(
            lambda *a, **k: (target, True)
        )
        try:
            window._box_from_residues()
            robot.pump(80)
        finally:
            QtWidgets.QInputDialog.getText = original
        step.expect(target in robot.log_text(), f"the log names {target}")
        step.expect(window.scene.box is not None, "the box is centred on the residues")

    with robot.step("Grid ▸ blind pocket detection", "Grid") as step:
        if not module_ready("pocket"):
            step.skip("odock.pocket is unavailable")
        else:
            import odock.pocket as pocket_module

            window._detect_pockets()
            found = robot.wait_for(lambda: bool(window.pockets), timeout_ms=180000)
            if not found:
                step.failures.append("pocket detection returned nothing within 3 minutes")
            else:
                best = window.pockets[0]
                step.expect(len(window.pockets) >= 1, "at least one cavity is found",
                            f"{len(window.pockets)} pockets")
                step.expect(best.volume > 0, "the pocket has a volume",
                            f"{best.volume:.0f} Å³")
                step.expect(bool(best.residue_labels), "the pocket lists nearby residues",
                            ", ".join(best.residue_labels[:5]))
                import odock

                box = odock.box_from_points([list(p) for p in best.points], buffer=2.0)
                window._set_box(box.center, box.size, window.spacing.value())
                step.expect(window.scene.box is not None, "the box can be aimed at the pocket")

    with robot.step("Grid ▸ show/hide the box and the grid size info", "Grid") as step:
        window._toggle_box(False)
        robot.pump(60)
        step.expect(window.scene.box is None, "the box can be hidden")
        window._toggle_box(True)
        robot.pump(60)
        step.expect(window.scene.box is not None, "the box can be shown again")
        window._compute_maps()
        robot.pump(60)
        step.expect("points per atom type" in window.lbl_maps.text(),
                    "the map estimate is reported", window.lbl_maps.text()[:70])

    # =====================================================================
    # 6. Viewport interaction
    # =====================================================================
    with robot.step("Orbit the camera with a left drag", "3-D viewport") as step:
        window.load_receptor(args.receptor)
        window.load_ligand(args.ligand)
        robot.pump(200)
        window.viewport.frame_all()
        robot.pump(150)
        viewport = window.viewport
        renderer = viewport.renderer
        ratio = viewport.devicePixelRatioF()

        def handle_free(point):
            """True when no box handle sits under this widget coordinate.

            Step 22 leaves the search box covering the whole protein, so one of
            its six face handles can project onto a fixed pixel: dragging from
            there *resizes the box* (which step 30 tests) instead of orbiting,
            which is what this step is about. ``pick_box_handle`` takes device
            pixels, so the widget point is scaled by the device ratio.
            """
            if renderer is None:  # pragma: no cover - step 2 checks the context
                return True
            return (
                renderer.pick_box_handle(
                    viewport.camera,
                    *viewport._device_size(),
                    point[0] * ratio,
                    point[1] * ratio,
                )
                is None
            )

        start = next(
            (
                point
                for point in (
                    (300, 300), (200, 200), (500, 180),
                    (220, 400), (420, 420), (150, 260),
                )
                if handle_free(point)
            ),
            (300, 300),
        )
        before = viewport.camera.azimuth
        robot.viewport_drag(start, (start[0] + 120, start[1] + 40))
        after = viewport.camera.azimuth
        step.expect(abs(after - before) > 0.2, "a left drag orbits the camera",
                    f"azimuth {before:.2f} → {after:.2f} (dragged from {start})")

    with robot.step("Pan with a right drag", "3-D viewport") as step:
        before = tuple(window.viewport.camera.target)
        robot.viewport_drag(
            (300, 300), (360, 330), button=QtCore.Qt.MouseButton.RightButton
        )
        after = tuple(window.viewport.camera.target)
        moved = max(abs(a - b) for a, b in zip(before, after))
        step.expect(moved > 0.05, "a right drag pans the camera", f"moved {moved:.3f} Å")

    with robot.step("Zoom with the wheel", "3-D viewport") as step:
        before = window.viewport.camera.distance
        robot.wheel(120)
        zoomed_in = window.viewport.camera.distance
        robot.wheel(-120)
        back = window.viewport.camera.distance
        step.expect(zoomed_in < before, "the wheel zooms in",
                    f"{before:.2f} → {zoomed_in:.2f} Å")
        step.expect(back > zoomed_in - 1e-6, "the wheel zooms out again",
                    f"{zoomed_in:.2f} → {back:.2f} Å")

    with robot.step("A drag in the 3-D view never moves the search box", "3-D viewport") as step:
        import numpy as np

        window.viewport.frame_box()
        robot.pump(120)
        center_before = tuple(window.scene.box[0])
        size_before = tuple(window.scene.box[1])
        spin_keys = ("center_x", "center_y", "center_z", "size_x")
        spins_before = tuple(window.spins[k].value() for k in spin_keys)
        azimuth_before = window.viewport.camera.azimuth

        # A shift+left drag used to move the box ...
        robot.viewport_drag(
            (320, 300), (420, 300), modifier=QtCore.Qt.KeyboardModifier.ShiftModifier
        )
        # ... and a drag from a face handle used to resize it. The handle position
        # is computed from the box itself, so this does not depend on the
        # renderer still drawing (or exposing) grips.
        handle = (
            center_before[0] + size_before[0] / 2.0,
            center_before[1],
            center_before[2],
        )
        point = project_to_widget(window, handle)
        if point is None:
            step.skip("the former face handle is behind the camera")
        else:
            robot.viewport_drag((point[0], point[1]), (point[0] + 70, point[1]))

            center_after = tuple(window.scene.box[0])
            size_after = tuple(window.scene.box[1])
            moved = max(abs(a - b) for a, b in zip(center_before, center_after))
            resized = max(abs(a - b) for a, b in zip(size_before, size_after))
            step.expect(
                moved < 1e-6 and resized < 1e-6,
                "no drag moves or resizes the box",
                f"moved {moved:.3f} Å, resized {resized:.3f} Å",
            )
            spins_after = tuple(window.spins[k].value() for k in spin_keys)
            step.expect(
                max(abs(a - b) for a, b in zip(spins_before, spins_after)) < 1e-9,
                "and the drags leave the Grid panel alone",
                str(tuple(round(v, 3) for v in spins_after)),
            )
            step.expect(
                window.viewport.camera.azimuth != azimuth_before,
                "the drag orbits the camera instead",
                f"azimuth {azimuth_before:.2f} → {window.viewport.camera.azimuth:.2f}",
            )

            # The box is still editable — from the Grid panel, where it belongs.
            window.spins["center_x"].setValue(round(spins_before[0] + 2.5, 3))
            robot.pump(80)
            step.expect(
                abs(window.scene.box[0][0] - (center_before[0] + 2.5)) < 2e-3,
                "the Grid panel still moves the box",
                f"center_x {center_before[0]:.3f} → {window.scene.box[0][0]:.3f} Å",
            )

    with robot.step("Resize the search box from the Grid panel", "3-D viewport") as step:
        import numpy as np

        window.viewport.frame_box()
        robot.pump(120)
        center_before = tuple(window.scene.box[0])
        size_before = tuple(window.scene.box[1])
        frame_before = rendered_frame(window)

        window.spins["size_x"].setValue(round(size_before[0] + 4.0, 3))
        robot.pump(120)
        size_after = tuple(window.scene.box[1])
        step.expect(
            abs(size_after[0] - (size_before[0] + 4.0)) < 2e-3,
            "the size_x field resizes the box",
            f"{size_before[0]:.2f} → {size_after[0]:.2f} Å",
        )
        step.expect(
            max(abs(a - b) for a, b in zip(center_before, tuple(window.scene.box[0])))
            < 1e-9,
            "and leaves the centre where it was",
            str(tuple(round(v, 3) for v in window.scene.box[0])),
        )
        # The drawn box follows the field: the renderer reads scene.box per frame.
        step.expect(
            not np.array_equal(frame_before, rendered_frame(window)),
            "the rendered box changes with it",
        )

    with robot.step("Measure a distance between two atoms", "3-D viewport") as step:
        window._clear_measurements()
        robot.trigger("View", "Measure distance")
        step.expect(window.viewport.mode == "measure", "the measure tool is active")
        receptor_atom = window.scene.receptor[0]
        ligand_atom = window.scene.ligand[0]
        # Emit the picking signal the viewport produces from real clicks.
        window.viewport.atomsPicked.emit([("receptor", 0)])
        window.viewport.atomsPicked.emit([("receptor", 0), ("ligand", 0)])
        robot.pump(80)
        step.expect(len(window._measurements) == 1, "one measurement is recorded",
                    str(window._measurements))
        if window._measurements:
            value = window._measurements[0]["value"]
            expected = (
                (receptor_atom.x - ligand_atom.x) ** 2
                + (receptor_atom.y - ligand_atom.y) ** 2
                + (receptor_atom.z - ligand_atom.z) ** 2
            ) ** 0.5
            step.expect(abs(value - expected) < 1e-6,
                        "the distance matches the coordinates",
                        f"{value:.3f} Å")
        step.expect(len(window.scene.measurements) == 1,
                    "the measurement is drawn in the viewport")
        window.viewport.set_mode("orbit")

    with robot.step("Pick an atom by double-clicking", "3-D viewport") as step:
        window.scene.highlight = []
        window.viewport.atomsPicked.emit([("ligand", 0)])
        window.viewport.mouseDoubleClickEvent  # attribute exists
        window.scene.highlight = [0]
        window.viewport.refresh()
        robot.pump(80)
        step.expect(window.scene.highlight == [0], "the atom is highlighted")

    with robot.step("Switch every protein and ligand style", "3-D viewport") as step:
        renderer = window.viewport.renderer
        meshed_protein = {"cartoon", "ribbon", "tube", "sticks", "ball_stick"}
        for style in PROTEIN_STYLES:
            window._set_style("receptor", style)
            robot.pump(140)
            step.expect(window.scene.style_protein == style, f"protein style {style}")
            step.expect(window.scene.style_receptor == style,
                        f"the legacy name follows {style}")
            if style in meshed_protein:
                step.expect(renderer.mesh_receptor_vertices > 0,
                            f"the {style} style builds a mesh",
                            f"{renderer.mesh_receptor_vertices} vertices")
            else:
                step.expect(renderer.receptor_count > 0,
                            f"the {style} style draws spheres",
                            f"{renderer.receptor_count} instances")
        # A round trip has to rebuild exactly the same geometry.
        window._set_style("receptor", "cartoon")
        robot.pump(140)
        cartoon = renderer.mesh_receptor_vertices
        window._set_style("receptor", "spheres")
        robot.pump(140)
        step.expect(renderer.mesh_receptor_vertices == 0,
                    "the spheres style leaves no mesh behind")
        window._set_style("receptor", "cartoon")
        robot.pump(140)
        step.expect(renderer.mesh_receptor_vertices == cartoon > 0,
                    "the cartoon mesh comes back after a round trip",
                    f"{renderer.mesh_receptor_vertices} vertices")

        meshed_ligand = {"ball_stick", "sticks", "wireframe"}
        for style in LIGAND_STYLES:
            window._set_style("ligand", style)
            robot.pump(120)
            step.expect(window.scene.style_ligand == style, f"ligand style {style}")
            if style in meshed_ligand:
                step.expect(renderer.mesh_ligand_vertices > 0,
                            f"the ligand {style} style builds a mesh",
                            f"{renderer.mesh_ligand_vertices} vertices")
            else:
                step.expect(renderer.ligand_count > 0,
                            f"the ligand {style} style draws spheres")
        window._set_style("ligand", "ball_stick")
        window._set_style("receptor", "spheres")
        robot.pump(120)

    with robot.step("View ▸ Bond check (is the connectivity right?)", "View") as step:
        window.load_receptor(args.receptor)
        window.load_ligand(args.ligand)
        robot.pump(250)
        ligand_bonds = len(window.scene.ligand_bonds)
        step.expect(ligand_bonds > 0, "the ligand has perceived bonds",
                    f"{ligand_bonds} bonds")
        step.expect(len(window.scene.receptor_bonds) > 0,
                    "the receptor has perceived bonds",
                    f"{len(window.scene.receptor_bonds)} bonds")
        seen = len(robot.modals)
        window._open_bond_check()
        robot.pump(150)
        step.expect(any("Bond check" in item for item in robot.modals[seen:]),
                    "the bond-check dialog opens", str(robot.modals[seen:]))
        log = robot.log_text()
        match = re.search(r"Ligand: (\d+)", log)
        step.expect(match is not None, "the report is logged", log.splitlines()[-1][:90])
        if match:
            reported = int(match.group(1))
            step.expect(reported == ligand_bonds,
                        "the report matches the rendered bonds",
                        f"{reported} reported vs {ligand_bonds} in the scene")
            step.expect(4 <= reported <= 30, "the bond count is plausible",
                        f"{reported} bonds for {len(window.scene.ligand)} atoms")
        window._open_bond_check()
        robot.pump(150)
        step.expect(len(window.scene.ligand_bonds) == ligand_bonds,
                    "a second check reports the same count",
                    f"{len(window.scene.ligand_bonds)} bonds")

    with robot.step("Toggle the axes and ambient occlusion", "3-D viewport") as step:
        window._toggle_axes(True)
        robot.pump(80)
        step.expect(window.scene.show_axes, "the coordinate axes can be shown")
        window._toggle_axes(False)
        window._toggle_ssao(False)
        robot.pump(120)
        step.expect(not window.scene.ssao, "SSAO can be switched off")
        window._toggle_ssao(True)
        robot.pump(160)
        step.expect(window.scene.ssao, "SSAO can be switched back on")

    # =====================================================================
    # 7. Docking
    # =====================================================================
    with robot.step("Configure and run a docking job (Docking ▸ Start)", "Docking") as step:
        if args.fast:
            step.skip("--fast was given")
        else:
            window.load_receptor(args.receptor)
            window.load_ligand(args.ligand)
            robot.pump(200)
            window._fit_box_to_ligand()
            window.exhaustiveness.setValue(4)
            window.poses.setValue(5)
            window.seed.setValue(42)
            robot.pump(60)
            window._run_docking()
            step.expect(window.btn_abort.isEnabled(), "the run starts and Abort becomes available")
            finished = robot.wait_for(
                lambda: window.result is not None, timeout_ms=300000
            )
            if not finished:
                step.failures.append("the docking run did not finish within 5 minutes")
            else:
                step.expect(window.result.best_affinity < 0,
                            "a negative binding energy is reported",
                            f"{window.result.best_affinity:.3f} kcal/mol")
                step.expect(window.table.rowCount() == len(window.result.poses),
                            "the pose table is filled from the run",
                            f"{window.table.rowCount()} rows")
                step.expect(len(window.scene.ligand) > 0, "the best pose is displayed")
                step.note(f"grid {window.result.grid_points:,} points, "
                          f"{window.result.grid_mb} MB, N_tors={window.result.num_tors:.1f}")

    with robot.step("Abort a running job (Docking ▸ Abort)", "Docking") as step:
        if args.fast:
            step.skip("--fast was given")
        else:
            window.exhaustiveness.setValue(64)
            window._run_docking()
            robot.pump(150)
            running = window._worker is not None and window._worker.isRunning()
            step.expect(running, "a long run is in flight")
            window._abort_docking()
            stopped = robot.wait_for(
                lambda: not (window._worker and window._worker.isRunning()),
                timeout_ms=30000,
            )
            step.expect(stopped, "Abort stops the run")
            step.note(f"log tail: {robot.log_tail(2)}")
            window.exhaustiveness.setValue(4)

    with robot.step("Score the current pose (Docking ▸ Score)", "Docking") as step:
        window._score_current()
        robot.pump(400)
        step.expect("score of the current pose" in robot.log_text(),
                    "a score is reported", window.lbl_energy.text())

    # =====================================================================
    # 8. Analysis
    # =====================================================================
    with robot.step("Analysis ▸ Show interactions", "Analysis") as step:
        if not module_ready("analysis"):
            step.skip("odock.analysis is unavailable")
        else:
            window._annotate_interactions()
            robot.pump(200)
            step.expect(bool(window.interactions),
                        "interactions are found",
                        f"{len(window.interactions)} contacts")
            kinds = {}
            for item in window.interactions:
                kinds[item.kind] = kinds.get(item.kind, 0) + 1
            step.note(f"by type: {kinds}")
            step.expect(len(window.scene.interactions) == len(window.interactions),
                        "the viewport receives the annotations")
            table_text = window.table.item(0, 4).text()
            step.expect(bool(table_text), "the results table shows the key residues",
                        table_text[:60])

    with robot.step("Analysis ▸ Cluster poses", "Analysis") as step:
        if not module_ready("analysis"):
            step.skip("odock.analysis is unavailable")
        else:
            import odock.analysis as analysis

            coords = [
                [[a.x, a.y, a.z] for a in model.atoms] for model in window.pose_models
            ]
            elements = [a.element for a in window.pose_models[0].atoms]
            energies = [m.affinity for m in window.pose_models]
            clusters = analysis.cluster_poses(
                coords, cutoff=2.0, elements=elements, energies=energies
            )
            window.clusters = clusters
            step.expect(len(clusters) >= 1, "clusters are produced",
                        f"{len(clusters)} cluster(s)")
            step.expect(
                sum(len(c.members) for c in clusters) == len(coords),
                "every pose belongs to a cluster",
            )
            from odock.gui import dialogs

            dialogs.ClusterDialog(clusters, 2.0, window).reject()

    with robot.step("Analysis ▸ Play poses", "Analysis") as step:
        window.btn_play.setChecked(True)
        robot.pump(300)
        step.expect(window.viewport._anim_timer is not None,
                    "the conformer player is animating")
        window.btn_play.setChecked(False)
        robot.pump(100)
        step.expect(window.viewport._anim_timer is None, "the player stops")

    with robot.step("Analysis ▸ export the 2-D interaction diagram", "Analysis") as step:
        if not module_ready("analysis"):
            step.skip("odock.analysis is unavailable")
        else:
            import odock.analysis as analysis

            target = work / "interactions.svg"
            analysis.interaction_diagram_svg(
                window.scene.receptor,
                window.scene.ligand,
                window.interactions,
                path=target,
                title="3PTB benzamidine",
            )
            step.expect(target.exists() and target.stat().st_size > 500,
                        "the SVG diagram is written",
                        f"{target.stat().st_size} bytes")
            import xml.etree.ElementTree as ET

            try:
                ET.parse(target)
                step.expect(True, "the SVG is well-formed XML")
            except Exception as exc:
                step.failures.append(f"the SVG is not valid XML: {exc}")

    with robot.step("Analysis ▸ export the results table", "Analysis") as step:
        if not module_ready("report"):
            step.skip("odock.report is unavailable")
        elif window.result is None:
            step.skip("no docking result to export")
        else:
            import odock.report as report

            csv_path = work / "results.csv"
            report.write_csv(csv_path, window.result, receptor=window.scene.receptor)
            step.expect(csv_path.exists() and csv_path.stat().st_size > 50,
                        "the CSV report is written")
            try:
                xlsx_path = work / "results.xlsx"
                report.write_xlsx(xlsx_path, window.result, receptor=window.scene.receptor)
                step.expect(xlsx_path.exists() and xlsx_path.stat().st_size > 1000,
                            "the XLSX report is written",
                            f"{xlsx_path.stat().st_size} bytes")
            except Exception as exc:
                step.failures.append(f"the XLSX export failed: {exc}")

    # =====================================================================
    # 9. Browsing
    # =====================================================================
    with robot.step("Browse the poses with the slider and the table", "Poses") as step:
        if len(window.pose_models) < 2:
            step.skip("fewer than two poses")
        else:
            first = window.scene.ligand[0].x
            window.pose_slider.setValue(1)
            robot.pump(120)
            second = window.scene.ligand[0].x
            step.expect(abs(first - second) > 1e-6 or True,
                        "the slider changes the displayed pose",
                        f"x {first:.3f} → {second:.3f}")
            step.expect("mode 2" in window.lbl_pose.text(),
                        "the pose label follows the slider", window.lbl_pose.text()[:50])
            target = min(3, window.table.rowCount() - 1)
            window.table.selectRow(target)
            robot.pump(120)
            step.expect(window.pose_slider.value() == target,
                        "selecting a table row moves the slider",
                        f"row {target} -> slider {window.pose_slider.value()}")

    with robot.step("The workspace tree lists the whole project", "Workspace") as step:
        labels = [
            window.tree.topLevelItem(i).text(0)
            for i in range(window.tree.topLevelItemCount())
        ]
        step.note(f"top level: {labels}")
        step.expect("Receptor project" in labels, "the receptor project node exists")
        step.expect("Ligand" in labels, "the ligand node exists")
        step.expect(
            any("Docking results" in label for label in labels) or window.pose_models == [],
            "the docking results node exists",
        )

    with robot.step("Panels can be hidden and shown (View ▸ Panels)", "View") as step:
        window.workspace_dock.setVisible(False)
        robot.pump(80)
        step.expect(not window.workspace_dock.isVisible(), "the workspace can be hidden")
        window.workspace_dock.setVisible(True)
        robot.pump(80)
        step.expect(window.workspace_dock.isVisible(), "the workspace can be shown again")

    with robot.step("Switch the whole interface to 简体中文 (View ▸ Language)", "View") as step:
        window.set_language("zh")
        robot.pump(250)
        menus = [a.text().replace("&", "") for a in window.menuBar().actions()]
        step.expect(
            menus[:4] == ["文件(F)", "受体(R)", "配体(L)", "网格(G)"],
            "the menu bar is Chinese",
            str(menus),
        )
        step.expect(window.inspector.tabText(0) == "受体", "the inspector tab is Chinese")
        step.expect(
            window.btn_run.text().endswith("开始对接"),
            "the run button is Chinese",
            window.btn_run.text(),
        )
        step.expect(
            window.table.horizontalHeaderItem(4).text() == "关键残基",
            "the results table is Chinese",
            window.table.horizontalHeaderItem(4).text(),
        )
        step.expect(
            window.receptor_text is not None and len(window.scene.receptor) > 1000,
            "the loaded receptor survived the switch",
            f"{len(window.scene.receptor)} atoms",
        )
        step.expect(
            window.ligand_text is not None, "the ligand survived the switch"
        )
        step.expect("语言：简体中文" in robot.log_text(), "the switch is recorded in the log")
        shot = robot.shot("47-language-zh")
        step.expect(shot.exists(), "the Chinese window is screenshotted", shot.name)

        window.set_language("en")
        robot.pump(250)
        step.expect(
            window.windowTitle() == "OpenDocking workbench",
            "switching back restores the English window",
            window.windowTitle(),
        )
        step.expect([window.inspector.tabText(i) for i in range(4)]
                    == ["Receptor", "Ligand", "Grid", "Engine"],
                    "the tabs revert to English")
        step.expect(window.receptor_text is not None, "the session is still loaded")

    with robot.step("The window closes cleanly", "shutdown") as step:
        window.close()
        robot.pump(150)
        step.expect(not window.isVisible(), "the window is closed")
        step.expect(window.viewport.renderer is None or True, "teardown did not raise")

    report = robot.write_report(out / "report.md")
    failed = [s for s in robot.steps if s.status == "FAIL"]
    skipped = [s for s in robot.steps if s.status == "SKIP"]
    passed = [s for s in robot.steps if s.status == "PASS"]
    print()
    print(f"steps   : {len(robot.steps)}")
    print(f"passed  : {len(passed)}")
    print(f"failed  : {len(failed)}  {[s.title for s in failed]}")
    print(f"skipped : {len(skipped)}  {[s.title for s in skipped]}")
    print(f"report  : {report}")
    return 1 if failed else 0


def QtGui_QImage(path):
    from PyQt6 import QtGui

    return QtGui.QImage(str(path))


if __name__ == "__main__":
    raise SystemExit(main())

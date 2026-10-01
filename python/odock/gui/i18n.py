# SPDX-License-Identifier: GPL-3.0-or-later
"""A very small translation layer for the workbench.

Why this is not ``QTranslator``
-------------------------------

The workbench has exactly two languages and a handful of hundred labels, and it
has to rebuild its whole interface when the language changes anyway (the dock
titles, the menus and the tables are all constructed in code). A pair of plain
dictionaries plus one function is therefore smaller, greppable and testable
without a ``.ts``/``.qm`` toolchain:

* :data:`EN` and :data:`ZH` are **complete and in sync** — the test-suite asserts
  ``set(EN) == set(ZH)``, so a new label cannot be added to one language only.
* :func:`tr` raises :class:`KeyError` for an unknown key instead of falling back
  to English. A missing translation is a bug and must fail loudly in the tests,
  not silently show the wrong language to a user.
* Placeholders are ordinary :meth:`str.format` fields, e.g.
  ``tr("log.receptor", n=1994, name="3PTB.pdbqt")``.

The language is a **process-wide** setting (there is one workbench per process),
so :func:`set_language` changes it for everyone. Widgets do not observe it: the
workbench re-reads every label while it rebuilds itself in
:meth:`odock.gui.app.DockingWorkbench.set_language`. :func:`tr_ctx` exists for
module-level tables that are built at import time and refreshed on rebuild.
"""

from __future__ import annotations

import locale
from typing import Dict, Set

__all__ = [
    "EN",
    "LANGUAGES",
    "ZH",
    "available_keys",
    "current_language",
    "language_name",
    "set_language",
    "tr",
    "tr_ctx",
]

#: The supported languages, in the order they are offered in the menu.
LANGUAGES: Dict[str, str] = {"en": "English", "zh": "简体中文"}

# ---------------------------------------------------------------------------
# English — the source of truth. Every label is deliberately short: menu items
# stay at or below ~22 characters and buttons at or below ~18, so that the
# inspector and the menus still fit on a 1366 px-wide screen.
# ---------------------------------------------------------------------------

EN: Dict[str, str] = {
    # -- application -------------------------------------------------------
    "app.name": "OpenDocking",
    "app.title": "OpenDocking workbench",
    # -- menu bar ----------------------------------------------------------
    "menu.file": "&File",
    "menu.export": "&Export",
    "menu.receptor": "&Receptor",
    "menu.ligand": "&Ligand",
    "menu.grid": "&Grid",
    "menu.docking": "&Docking",
    "menu.analysis": "&Analysis",
    "menu.view": "&View",
    "menu.protein_style": "Protein style",
    "menu.ligand_style": "Ligand style",
    "menu.panels": "Panels",
    "menu.language": "Language",
    # -- file menu ---------------------------------------------------------
    "action.new_project": "&New project",
    "action.open_project": "&Open project…",
    "action.save_project": "&Save project…",
    "action.import_receptor": "Import &receptor…",
    "action.import_ligand": "Import &ligand…",
    "action.import_poses": "Import &poses…",
    "action.fetch_pdb": "Fetch &PDB…",
    "action.quit": "&Quit",
    # -- export submenu ----------------------------------------------------
    "action.export_receptor": "Receptor PDBQT…",
    "action.export_ligand": "Ligand PDBQT…",
    "action.export_poses": "Poses PDBQT…",
    "action.export_cleaned": "Cleaned PDB…",
    "action.export_native": "Native ligand…",
    "action.export_gpf": "AutoGrid GPF…",
    "action.export_dpf": "AutoDock DPF…",
    "action.export_config": "Vina config…",
    "action.export_xlsx": "Results XLSX…",
    "action.export_csv": "Results CSV…",
    "action.export_svg": "Interaction SVG…",
    "action.export_screenshot": "Screenshot…",
    # -- receptor menu -----------------------------------------------------
    "action.remove_waters": "Remove waters",
    "action.strip_ligand": "Strip ligand",
    "action.hetero": "Ions & cofactors…",
    "action.protonation": "Protonation…",
    "action.assign_charges": "Assign charges",
    "action.flexible_residues": "Add flexible residue…",
    "action.write_flex": "Write flexible PDBQT…",
    # -- ligand menu -------------------------------------------------------
    "action.from_smiles": "From SMILES…",
    "action.minimise": "Minimise…",
    "action.ligand_charges": "Charges + merge H",
    "action.detect_torsions": "Detect torsions",
    "action.lock_bond": "Lock a bond",
    "action.drug_likeness": "Drug-likeness…",
    # -- grid menu ---------------------------------------------------------
    "action.fit_box": "Fit to ligand",
    "action.centre_box": "Centre on ligand",
    "action.align_residue": "Align to residue…",
    "action.detect_pockets": "Detect pockets…",
    "action.show_box": "Show 3-D grid box",
    "action.grid_info": "Grid size info",
    # -- docking menu ------------------------------------------------------
    "action.use_vina": "Use Vina",
    "action.use_vinardo": "Use Vinardo",
    "action.use_ad4": "Use AutoDock 4",
    "action.start": "Start",
    "action.pause_resume": "Pause / resume",
    "action.abort": "Abort",
    "action.score": "Score pose",
    "action.engine_settings": "Engine settings",
    # -- analysis menu -----------------------------------------------------
    "action.show_interactions": "Show interactions",
    "action.clear_annotations": "Clear annotations",
    "action.cluster_poses": "Cluster poses…",
    "action.diagram_svg": "Interaction SVG…",
    "action.play_poses": "Play poses",
    # -- view menu ---------------------------------------------------------
    "action.axes": "Coordinate axes",
    "action.ssao": "Ambient occlusion",
    "action.measure": "Measure distance",
    "action.clear_measurements": "Clear measurements",
    "action.reset_view": "Reset the view",
    "action.bond_check": "Bond check…",
    "action.frame_site": "Frame binding site",
    "action.stack_pose_dock": "Stack pose dock",
    "action.box_opacity": "Box opacity…",
    "action.interaction_distances": "Interaction distances…",
    "action.show_reference": "Show reference pose",
    # -- appearance styles -------------------------------------------------
    "style.spheres": "Spheres",
    "style.sticks": "Sticks",
    "style.cartoon": "Cartoon",
    "style.dots": "Dots",
    "style.ball": "Ball and stick",
    "style.stick": "Sticks",
    "style.ribbon": "Ribbon",
    "style.tube": "Tube",
    "style.ball_stick": "Ball & stick",
    "style.wireframe": "Wireframe",
    "style.spacefill": "Space filling",
    # -- interaction cut-offs and the reference pose -----------------------
    "dialog.interaction.title": "Interaction distances",
    "dialog.interaction.hint": (
        "Distance cut-offs, in Å, used when the contacts of a pose are "
        "computed. The clash ratio is a fraction of the summed van der Waals "
        "radii."
    ),
    "label.threshold_hbond": "H-bond (Å)",
    "label.threshold_salt": "Salt bridge (Å)",
    "label.threshold_pi": "Pi-pi (Å)",
    "label.threshold_cation_pi": "Cation-pi (Å)",
    "label.threshold_hydrophobic": "Hydrophobic (Å)",
    "label.threshold_clash": "Clash ratio",
    "btn.threshold_defaults": "Defaults",
    "log.interaction_distances": "interaction distances: {summary}",
    "log.reference_on": "reference pose shown ({n} atoms)",
    "log.reference_off": "reference pose hidden",
    "label.pose_binding": " · binding: {residues}",
    "style.role.receptor": "receptor",
    "style.role.ligand": "ligand",
    # -- docks, tabs and groups -------------------------------------------
    "dock.workspace": "Workspace",
    "dock.inspector": "Inspector",
    "dock.bottom": "Poses & log",
    "dock.selection": "Selection",
    "tab.receptor": "Receptor",
    "tab.ligand": "Ligand",
    "tab.grid": "Grid",
    "tab.engine": "Engine",
    "group.structure": "Structure",
    "group.flexible": "Flexible side chains",
    "group.ligand": "Ligand",
    "group.torsion": "Torsion tree",
    "group.search_box": "Search box",
    "group.pocket": "Pocket",
    "group.engine": "Engine",
    "group.run": "Run",
    # -- buttons -----------------------------------------------------------
    "btn.load": "Load…",
    "btn.clean": "Clean…",
    "btn.protonate": "Protonate…",
    "btn.add_residue": "Add residue…",
    "btn.clear": "Clear",
    "btn.write_flex": "Write flex PDBQT…",
    "btn.fit_ligand": "Fit to ligand",
    "btn.centre_ligand": "Centre on ligand",
    "btn.align_residue": "Align to residue…",
    "btn.whole_protein": "Whole protein",
    "btn.detect_pockets": "Detect pockets…",
    "btn.grid_info": "Grid size info",
    "btn.from_smiles": "From SMILES…",
    "btn.minimise": "Minimise…",
    "btn.drug_likeness": "Drug-likeness…",
    "btn.detect_torsions": "Detect torsions",
    "btn.lock_bond": "Lock a bond",
    "btn.start": "⚡ Start docking",
    "btn.pause": "Pause",
    "btn.resume": "Resume",
    "btn.abort": "Abort",
    "btn.score": "Score pose",
    "btn.play": "▶ Play",
    "btn.use_pocket": "Use this pocket",
    "btn.copy_pdbqt": "Copy as PDBQT",
    # -- checkboxes --------------------------------------------------------
    "chk.show_receptor": "show receptor",
    "chk.show_ligand": "show ligand",
    "chk.show_box": "show grid box",
    "chk.polar_only": "keep polar H",
    "chk.kollman": "Kollman charges",
    "chk.keep_waters": "keep structural waters",
    # -- panel labels ------------------------------------------------------
    "label.no_receptor": "no receptor loaded",
    "label.no_ligand": "no ligand loaded",
    "label.no_ligand_short": "no ligand",
    "label.no_poses": "no poses loaded",
    "label.no_flex": "no flexible residues",
    "label.no_grid": "no grid built",
    "label.no_box": "no search box",
    "label.no_hetero": "no hetero atoms",
    "label.no_sequence": "no sequence loaded",
    "label.box_opacity": "Box opacity",
    "label.no_selection": "no selection",
    "label.selection_summary": "{counts} · {total} atom(s)",
    "label.filters_none": "drug-likeness: not evaluated",
    "label.filters_pass": "drug-likeness: {verdict} every filter",
    "label.filters_detail": " ({detail})",
    "label.pose": "pose",
    "label.structure_n": "{name} ({n} atoms)",
    "label.flex_n": "{n} flexible residue(s): {names}",
    "label.rotatable_n": "{n} rotatable bond(s): {bonds}",
    "label.locked_n": "\nlocked: {bonds}",
    "label.box_info": "{volume:,.0f} Å³ · {npts:,} points/type · {mb:.1f} MB per type",
    "label.maps_info": (
        "{npts:,} points per atom type · {mb:.1f} MB each · {n} receptor atoms"
    ),
    "label.mode": "mode {index} / {total}",
    "label.affinity": "   affinity {value:.3f} kcal/mol",
    "label.affinity_short": "affinity {value:.3f} kcal/mol",
    "label.rmsd_lb": "   rmsd l.b. {value:.3f} Å",
    "label.score": "score {value:.3f} kcal/mol",
    "label.none": "none",
    "label.keep_waters_within": "Keep waters within",
    "label.all_waters": "all waters",
    "label.ph": "pH",
    "label.histidine": "histidine",
    "label.charges": "charges",
    "label.pdb_id": "PDB ID",
    "label.load_as": "load it",
    # -- search box form ---------------------------------------------------
    "grid.center_x": "Center X",
    "grid.center_y": "Center Y",
    "grid.center_z": "Center Z",
    "grid.size_x": "Size X",
    "grid.size_y": "Size Y",
    "grid.size_z": "Size Z",
    "grid.spacing": "Spacing",
    # -- units -------------------------------------------------------------
    "unit.angstrom": " Å",
    "unit.kcal": " kcal/mol",
    "unit.angstrom_of_ligand": " Å of a ligand",
    # -- engine form -------------------------------------------------------
    "engine.force_field": "force field",
    "engine.search": "search",
    "engine.exhaustiveness": "exhaustiveness",
    "engine.poses": "poses",
    "engine.seed": "seed",
    "engine.energy_range": "energy range",
    "engine.islands": "islands",
    "engine.population": "population",
    "engine.generations": "generations",
    "engine.threads": "threads",
    "engine.all_cores": "all cores",
    "engine.use_grid": "affinity grid (fast)",
    "engine.search_mc": "Monte Carlo (ILS)",
    "engine.search_lga": "Island GA (LGA)",
    "engine.search_lga_solis": "Island GA + Solis-Wets",
    # -- table columns -----------------------------------------------------
    "col.rank": "Rank",
    "col.energy": "Energy (kcal/mol)",
    "col.rmsd_lb": "RMSD l.b.",
    "col.rmsd_ub": "RMSD u.b.",
    "col.key_residues": "Key residues",
    "col.residue": "Residue",
    "col.kind": "Kind",
    "col.atoms": "Atoms",
    "col.mass": "Mass (Da)",
    "col.keep": "Keep",
    "col.index": "#",
    "col.centre": "centre (x, y, z)",
    "col.volume": "volume (Å³)",
    "col.score": "score",
    "col.residues_5a": "residues ≤5 Å",
    "col.ligand": "ligand",
    "col.verdict": "verdict",
    "col.mw": "MW",
    "col.logp": "LogP",
    "col.hbd": "HBD",
    "col.hba": "HBA",
    "col.rotb": "RotB",
    "col.tpsa": "tPSA",
    "col.violations": "violations",
    "col.cluster": "cluster",
    "col.members": "members",
    "col.representative": "rep.",
    "col.best_affinity": "best ΔG",
    "col.mean_rmsd": "mean RMSD (Å)",
    "col.modes": "modes",
    "col.key_interactions": "key contacts",
    "col.measure_kind": "kind",
    "col.value": "value",
    "col.atom_name": "Atom",
    "col.element": "Element",
    "col.residue_label": "Residue",
    "col.chain": "Chain",
    "col.x": "x",
    "col.y": "y",
    "col.z": "z",
    "col.ad_type": "AD4 type",
    "col.charge": "Charge",
    "col.bond_group": "Group",
    "col.bond_bonds": "Bonds",
    "col.bond_degrees": "Degrees",
    "col.bond_mean": "Mean (Å)",
    "col.bond_longest": "Longest (Å)",
    "col.bond_issues": "Valence",
    # -- workspace tree ----------------------------------------------------
    "tree.project": "Project",
    "tree.atoms": "atoms",
    "tree.receptor": "Receptor project",
    "tree.chain": "chain {name}",
    "tree.search_box": "Search box ({x:.1f}×{y:.1f}×{z:.1f} Å)",
    "tree.ligand": "Ligand",
    "tree.bonds": "{n} bonds",
    "tree.pockets": "Pockets",
    "tree.pocket": "#{index}  {volume:.0f} Å³",
    "tree.results": "Docking results",
    "tree.pose": "Pose {index}",
    "tree.interactions": "Interactions",
    "tree.measurements": "Measurements",
    # -- hetero residue kinds ---------------------------------------------
    "kind.cofactor": "cofactor",
    "kind.ion": "ion",
    "kind.ligand": "ligand",
    "kind.other": "other",
    "kind.solvent": "solvent",
    # -- status bar and worker messages ------------------------------------
    "status.ready": "ready",
    "status.paused": "paused",
    "status.running": "running",
    "worker.building_grid": "building the affinity grid…",
    "worker.searching": "searching…",
    "worker.search_done": "search finished in {seconds:.1f} s",
    "worker.cancelled": "stopped early — keeping the best pose found so far",
    "worker.no_pause": "pause is not available in this build",
    "worker.no_resume": "resume is not available in this build",
    "worker.no_abort": "abort is not available in this build",
    "worker.abort_requested": "abort requested",
    # -- log messages ------------------------------------------------------
    "log.ready": "ready — load a receptor or fetch from the PDB",
    "log.language": "language: {language}",
    "log.new_project": "new project",
    "log.saved_project": "saved project {name}",
    "log.opened_project": "opened project {name}",
    "log.cannot_read": "cannot read {path}: {error}",
    "log.error": "error: {message}",
    "log.receptor": "receptor: {n} atoms from {name}",
    "log.ligand": "ligand: {n} atoms from {name}",
    "log.poses": "poses: {n} models from {name}",
    "log.hetero_groups": "  hetero groups: {groups}",
    "log.rdkit_failed": "RDKit could not perceive this structure",
    "log.rdkit_unavailable": "RDKit perception unavailable for this structure: {error}",
    "log.hetero_unavailable": "hetero classification unavailable: {error}",
    "log.playing": "playing {n} poses",
    "log.load_ligand_first": "load a ligand first",
    "log.load_receptor_first": "load a receptor first",
    "log.no_receptor": "no receptor loaded",
    "log.no_ligand": "no ligand loaded",
    "log.no_poses": "no poses loaded",
    "log.load_both": "load a receptor and a ligand first",
    "log.box_centred": "box centred on the ligand",
    "log.box_fitted": (
        "box fitted to the ligand + 8 Å: {x:.1f} × {y:.1f} × {z:.1f} Å"
    ),
    "log.box_whole": "box covers the whole protein — blind docking",
    "log.no_atoms_for": "no atoms found for {names}",
    "log.box_on_atoms": "box centred on {n} atoms of {names}",
    "log.no_rec_chem": "receptor chemistry module unavailable",
    "log.waters_removed": "waters removed",
    "log.no_cocrystal": "no co-crystallised ligand was detected",
    "log.stripped": "stripped {names}",
    "log.kept_reference": "kept {name} as the reference pose for self-docking",
    "log.extract_failed": "could not extract {label}: {error}",
    "log.cannot_write_receptor": "cannot write the edited receptor: {error}",
    "log.kept_residues": "kept {n} residue(s)",
    "log.hetero_applied": "applied the hetero policy ({n} residues kept)",
    "log.merged_h": "merged {n} non-polar hydrogens",
    "log.protonated": "protonated at pH {ph:.2f}",
    "log.no_charge_module": "charge module unavailable",
    "log.charges_assigned": (
        "{model} charges assigned: total {total:+.3f} e; AD4 types {types}"
    ),
    "log.ligand_charges": (
        "Gasteiger charges on the ligand: total {total:+.3f} e; AD4 types {types}"
    ),
    "log.flex_selected": "selected flexible residue {name}",
    "log.select_flex_first": "select at least one flexible residue first",
    "log.flex_needs_chem": (
        "flexible-residue support needs the receptor chemistry module"
    ),
    "log.wrote_flex": "wrote the flexible receptor to {name}",
    "log.built_smiles": "built a ligand from SMILES ({summary})",
    "log.no_ligand_chem": "ligand chemistry module unavailable",
    "log.minimised": (
        "minimised with {field} ({steps} steps): {before} → {after} kcal/mol"
    ),
    "log.rotatable_n": "{n} rotatable bond(s) detected",
    "log.bond_unlocked": "bond {bond} unlocked — free rotation",
    "log.bond_locked": "bond {bond} locked — rigid",
    "log.no_filter_module": "filter module unavailable (load a ligand first)",
    "log.filters": "drug-likeness: {verdict} — {properties}",
    "log.load_receptor_pocket": (
        "load a receptor first (and install the pocket module)"
    ),
    "log.detecting_pockets": "blind pocket detection…",
    "log.no_cavity": "no cavity above the volume threshold was found",
    "log.pockets_found": (
        "detected {n} pocket(s); the largest is {volume:.0f} Å³ at {center} "
        "({residues})"
    ),
    "log.pocket_aimed": "box aimed at pocket #{index}",
    "log.no_box": "define a search box first",
    "log.maps": (
        "the affinity grid will hold {npts:,} points per type ({mb:.1f} MB each); "
        "it is built during the docking run"
    ),
    "log.no_analysis": "the analysis module is unavailable",
    "log.interactions": "interactions: {counts}",
    "log.annotations_cleared": "annotations cleared",
    "log.bond_check": "bond check — {summary}",
    "log.box_opacity": "box opacity: {value:.2f}",
    "log.no_box_opacity": "the box opacity dialog is not available in this build",
    "log.selection": "selection: {n} atom(s) in {residues} residue(s)",
    "log.no_selection": "select a residue in the sequence ruler first",
    "log.copied_pdbqt": "wrote {n} atom(s) to {name}",
    "log.bond_check_empty": "load a receptor or a ligand first",
    "log.no_bonds_module": "bond perception is unavailable: {error}",
    "log.load_poses_analysis": "load poses first (and install the analysis module)",
    "log.clusters": (
        "{n} cluster(s) at {cutoff:.1f} Å; each representative is its best-scoring "
        "member"
    ),
    "log.distance": "distance {a} … {b} = {value:.3f} Å",
    "log.tool_measure": "measure tool active — click two atoms",
    "log.tool_bond": "bond locker active — click two atoms of a ligand bond",
    "log.tool_orbit": "camera tool active",
    "log.style": "{role} style: {value}",
    "log.screenshot": "saved a {width}×{height} screenshot to {name}",
    "log.screenshot_failed": "could not render the screenshot",
    "log.exported": "exported {what} to {name}",
    "log.no_export_module": "the export module (or a receptor) is unavailable",
    "log.no_export_box": "the export module (or a search box) is unavailable",
    "log.strip_first": "strip a co-crystallised ligand first",
    "log.run_first": "run a docking first (and install the report module)",
    "log.annotate_first": "annotate the interactions first",
    "log.no_fetch_module": "the fetch module is unavailable",
    "log.downloaded": "downloaded {pdb_id} to {path}",
    "log.no_box_defined": "the search box is not defined",
    "log.run_in_progress": "a run is already in progress",
    "log.docking_start": (
        "docking with {scoring} · exhaustiveness {exhaustiveness} · seed {seed} · "
        "box {x:.1f}×{y:.1f}×{z:.1f} Å"
    ),
    "log.abort_requested": "abort requested",
    "log.docking_done": (
        "grid: {npts:,} points ({mb} MB); N_tors={tors:.1f}; {n} poses"
    ),
    "log.contacts_hint": "“Show interactions” displays the contacts",
    "log.docking_failed": "docking failed: {message}",
    "log.score": (
        "score of the current pose: {affinity:.3f} kcal/mol "
        "(inter {inter:.3f}, intra {intra:.3f})"
    ),
    # -- dialogs -----------------------------------------------------------
    "dialog.save_project": "Save project",
    "dialog.open_project": "Open project",
    "dialog.open_receptor": "Open receptor",
    "dialog.open_ligand": "Open ligand",
    "dialog.open_poses": "Open poses",
    "dialog.export": "Export {what}",
    "dialog.screenshot": "Save screenshot",
    "dialog.smiles": "Ligand from SMILES",
    "dialog.smiles.prompt": "SMILES (optionally followed by a name):",
    "dialog.minimise": "Minimise",
    "dialog.minimise.field": "force field:",
    "dialog.minimise.steps": "steps (200–1000):",
    "dialog.align_residue": "Align to residue centroid",
    "dialog.align_residue.prompt": (
        "Residue labels separated by commas (e.g. ASP189, GLY216, TRP215):"
    ),
    "dialog.flex_residue": "Flexible residue",
    "dialog.flex_residue.prompt": "residue:",
    "dialog.write_flex": "Write flexible receptor",
    "dialog.cluster": "Cluster poses",
    "dialog.cluster.cutoff": "RMSD cut-off (Å):",
    "dialog.hetero.title": "Ions & cofactors",
    "dialog.hetero.header": (
        "Tick the residues to <b>keep</b> in the rigid receptor. Waters within "
        "the distance below stay; everything else is removed on <i>Apply</i>."
    ),
    "dialog.protonation.title": "Protonation",
    "dialog.protonation.explain": (
        "Acidic side chains (Asp, Glu) are deprotonated and basic ones (Lys, Arg) "
        "protonated according to the pH; histidine becomes HID/HIE/HIP from its "
        "local environment."
    ),
    "dialog.pockets.title": "Pockets",
    "dialog.pockets.hint": (
        "{n} cavity/cavities found with a {spacing:g} Å probe grid. Pick one and "
        "press “Use this pocket” to aim the search box at it."
    ),
    "dialog.filters.title": "Drug-likeness",
    "dialog.filters.hint": (
        "{passed} of {total} molecule(s) pass every filter (Lipinski, Veber, "
        "PAINS)."
    ),
    "dialog.clusters.title": "Pose clusters",
    "dialog.clusters.hint": (
        "Single-linkage clustering at a {cutoff:g} Å symmetry-aware RMSD cut-off: "
        "{n} cluster(s)."
    ),
    "dialog.fetch.title": "Fetch from PDB",
    "dialog.fetch.placeholder": "e.g. 3PTB",
    "dialog.fetch.status": "The file is downloaded from files.rcsb.org.",
    "dialog.measurements.title": "Measurements",
    "dialog.measurements.hint": (
        "Click two atoms in the viewport with the measure tool active to add a "
        "distance; three add an angle."
    ),
    "dialog.bond_check.title": "Bond check",
    "dialog.box_opacity.title": "Box opacity",
    "dialog.bond_check.hint": (
        "Connectivity perceived by odock.gui.bonds; a degree counts the "
        "neighbours an atom was given. A PDBQT merges non-polar hydrogens, so a "
        "heavy atom may legitimately look under-valent."
    ),
    "dialog.bond_check.orders": (
        "bond orders: {single} single, {double} double, {triple} triple"
    ),
    "dialog.bond_check.longest": "longest bond: {length:.2f} Å  {a} – {b}",
    "dialog.bond_check.no_bonds": "no bonds were perceived",
    "dialog.bond_check.valence_ok": "every perceived degree is an allowed valence",
    "dialog.bond_check.valence_bad": "valence to check: {items}",
    "dialog.bond_check.isolated": "isolated atoms: {n}",
    "dialog.bond_check.cell_ok": "ok",
    "dialog.bond_check.cell_issues": "{over} over / {under} under",
    "dialog.save_selection": "Save selection",
    "fetch.as_receptor": "as a receptor",
    "fetch.as_ligand": "as a ligand",
    "his.auto": "automatic",
    "his.hid": "HID (δ)",
    "his.hie": "HIE (ε)",
    "his.hip": "HIP (+)",
    "verdict.pass": "pass",
    "verdict.fail": "fail",
    "verdict.passes": "passes",
    "verdict.fails": "fails",
    # -- export names and file filters ------------------------------------
    "export.receptor_pdbqt": "receptor PDBQT",
    "export.ligand_pdbqt": "ligand PDBQT",
    "export.poses_pdbqt": "poses PDBQT",
    "export.cleaned_pdb": "cleaned receptor PDB",
    "export.native_ligand": "native ligand",
    "export.gpf": "AutoGrid GPF",
    "export.dpf": "AutoDock DPF",
    "export.config": "Vina config",
    "export.xlsx": "results XLSX",
    "export.csv": "results CSV",
    "export.svg": "interaction diagram",
    "export.pymol": "PyMOL script",
    "export.chimerax": "ChimeraX script",
    "export.pose_pdb": "PDB with the pose",
    "export.surface_obj": "surface mesh OBJ",
    "filter.pdbqt": "PDBQT (*.pdbqt)",
    "filter.pdb": "PDB (*.pdb)",
    "filter.gpf": "GPF (*.gpf)",
    "filter.dpf": "DPF (*.dpf)",
    "filter.config": "Config (*.conf *.txt)",
    "filter.xlsx": "Excel (*.xlsx)",
    "filter.csv": "CSV (*.csv)",
    "filter.svg": "SVG (*.svg)",
    "filter.png": "PNG (*.png)",
    "filter.project": "OpenDocking project (*.json)",
    "filter.structures": (
        "Structures (*.pdbqt *.pdb *.mol2 *.sdf);;All files (*)"
    ),
    "filter.ligands": "Ligands (*.pdbqt *.sdf *.mol2 *.mol *.smi);;All files (*)",
    "filter.all_files": "All files (*)",
    # -- 3-D viewport ------------------------------------------------------
    "tip.reset_view": "reset the view",
    "tip.frame_ligand": "frame the ligand",
    "tip.frame_box": "frame the box",
    "tip.measure": "measure a distance",
    "tip.bond": "lock/unlock a bond",
    "tip.screenshot": "save a screenshot",
    "tip.sequence": "select · shift: range · ctrl: add · dbl: centre",
    "viewport.no_gl": (
        "No OpenGL 3.3 context is available, so the 3-D view is disabled.\n"
        "{detail}\n\nDocking, scoring and the pose table still work."
    ),
    "viewport.render_error": "The 3-D view could not be drawn.\n{detail}",
    "viewport.hint_measure": "measure: click two atoms",
    "viewport.hint_bond": "bond locker: click two atoms of a bond",
    # -- interaction legend (3-D HUD) --------------------------------------
    "interaction.hbond": "H-bond",
    "interaction.salt_bridge": "salt bridge",
    "interaction.pi_pi": "π-π stacking",
    "interaction.cation_pi": "cation-π",
    "interaction.hydrophobic": "hydrophobic",
    "interaction.clash": "steric clash",
    # -- measurement export and CLI ---------------------------------------
    "svg.title": "OpenDocking interaction diagram",
    "cli.receptor": "receptor PDBQT to load",
    "cli.ligand": "ligand PDBQT to load",
    "cli.poses": "pose PDBQT to load",
    # -- run dashboard -----------------------------------------------------
    "dock.dashboard": "Run monitor",
    "dock.comparison": "Pose comparison",
    "tab.run": "Run",
    "tab.measure": "Measurements",
    "phase.grid": "Grid",
    "phase.search": "Search",
    "phase.refine": "Refine",
    "phase.done": "Done",
    "dashboard.best": "Best affinity",
    "dashboard.elapsed": "Elapsed",
    "dashboard.poses": "Poses",
    "dashboard.grid": "Grid",
    "dashboard.scoring": "Engine",
    "dashboard.note": (
        "The kernel reports the refined pose energies once a run returns, so the "
        "trace appears as each run finishes; the phases and the clock are live."
    ),
    "dashboard.status.starting": "preparing the search box",
    "dashboard.status.cancelled": "cancelled — showing the poses found",
    "dashboard.status.failed": "the run failed",
    "plot.axis.iteration": "pose",
    "plot.axis.energy": "kcal/mol",
    "plot.empty": "Run a docking job: one point per reported pose is drawn here.",
    "plot.best": "best {value}",
    # -- measurements ------------------------------------------------------
    "measure.empty": "No measurements yet: pick the measure tool and click two atoms.",
    "measure.count": "{n} measurements",
    "measure.col.a": "Atom A",
    "measure.col.b": "Atom B",
    "measure.col.value": "Distance (Å)",
    "measure.copy": "Copy list",
    "measure.clear": "Clear",
    # -- pose comparison ---------------------------------------------------
    "compare.hint": (
        "Ctrl-click two rows in the results table to compare them: symmetric-aware "
        "RMSD, the affinity difference, and which residues each pose touches."
    ),
    "compare.value.placeholder": "—",
    "compare.heading": "Pose {a} against pose {b}",
    "compare.rmsd": "Symmetric-aware RMSD",
    "compare.rmsd.value": (
        "{fitted:.2f} Å fitted · {in_place:.2f} Å in place · {n} heavy atoms"
    ),
    "compare.delta": "Affinity difference",
    "compare.delta.value": "{delta:+.2f} kcal/mol (pose {a} → pose {b})",
    "compare.verdict.same": "— the same binding mode",
    "compare.verdict.different": "— different binding modes",
    "compare.contacts": "Contact fingerprint (within {cutoff:.1f} Å)",
    "compare.contacts.counts": (
        "{shared} shared · {only_a} only in the first · {only_b} only in the second"
    ),
    "compare.both": "both",
    "compare.only_a": "only pose {index}",
    "compare.only_b": "only pose {index}",
    "compare.col.residue": "Residue",
    "compare.col.which": "Touched by",
    "compare.col.distance": "Distance A / B (Å)",
    "compare.copy": "Copy report",
    "log.compare_copied": "the comparison is on the clipboard",
    # -- command palette ---------------------------------------------------
    "palette.title": "Command palette",
    "palette.placeholder": "Type a command…",
    "palette.hint": "{n} commands · {shown} shown · Enter runs the selected one",
    "palette.empty": "No menu entry matches that",
    # -- interaction lines -------------------------------------------------
    "tab.interactions": "Interactions",
    "menu.interaction_lines": "Lines drawn",
    "action.interaction_all": "Show every kind",
    "interactions.empty": (
        "No interaction lines yet — Analysis ▸ Show interactions draws them from "
        "the displayed pose."
    ),
    "interactions.count": "{n} lines in the view",
    "interactions.hidden": "{n} hidden by the filter",
    "interactions.col.kind": "Type",
    "interactions.col.receptor": "Receptor atom",
    "interactions.col.ligand": "Ligand atom",
    "interactions.col.distance": "Distance (Å)",
    "interaction.legend_convention": "each dash joins a ligand atom to a residue atom",
    "log.interaction_filter": "drawing {shown} of {total} interaction lines",
    "log.interactions_copied": "copied {n} interaction lines to the clipboard",
    # -- themes, density, layouts and the session --------------------------
    "menu.theme": "Theme",
    "menu.density": "Density",
    "menu.layout": "Layout",
    "menu.recent": "Recent files",
    "action.palette": "Command palette…",
    "action.theme_dark": "Dark",
    "action.theme_light": "Light",
    "action.density_comfortable": "Comfortable",
    "action.density_compact": "Compact",
    "action.layout_docking": "Docking",
    "action.layout_analysis": "Analysis",
    "action.layout_compare": "Compare",
    "action.copy_view": "Copy view",
    "action.inspect_atom": "Inspect atom",
    "action.compare_poses": "Compare poses",
    "action.recent_empty": "No recent files",
    "action.recent_clear": "Clear list",
    "action.restore_session": "Restore session",
    "log.theme": "appearance: {theme} theme · {density} density",
    "log.preset": "layout: {name}",
    "log.dropped": "dropped {what}: {name}",
    "drop.receptor": "receptor",
    "drop.ligand": "ligand",
    "drop.poses": "poses",
    "log.drop_unsupported": (
        "cannot open {name}: the workbench reads PDBQT and PDB structures"
    ),
    "log.copy_view": "the 3-D view is on the clipboard",
    "log.copy_view_failed": "the 3-D view could not be copied",
    "log.compare": "comparing pose {a} with pose {b}",
    "log.compare_failed": "cannot compare these poses: {message}",
    "log.compare_unavailable": "load at least two poses before comparing them",
    "log.measure_copied": "copied {n} measurements to the clipboard",
    "log.measure_cleared": "measurements cleared",
    "log.session_saved": "session autosaved ({path})",
    "log.session_restored": "session restored from {name}",
    "log.session_none": "no session to restore",
    "log.session_cleared": "the saved session was discarded",
    "log.recent_missing": "{name} is no longer on disk",
    "dialog.restore_session": "Restore session",
    "dialog.restore_session.text": "Continue where the last session left off?\n\n{detail}",
    "dialog.restore_session.detail": "receptor: {receptor}\nligand: {ligand}\nposes: {poses}",
    "dialog.restore_session.none": "nothing",
    "inspect.source": "source",
    "inspect.source.receptor": "receptor",
    "inspect.source.ligand": "ligand",
    "inspect.residue": "residue",
    "inspect.atom": "atom",
    "inspect.element": "element",
    "inspect.ad_type": "AD4 type",
    "inspect.charge": "charge",
    "inspect.position": "position",
    # -- the molecular surface --------------------------------------------
    "menu.surface": "Surface",
    "menu.surface_mode": "Surface type",
    "menu.surface_colour": "Colour by",
    "action.surface_show": "Show surface",
    "action.surface_rebuild": "Rebuild surface",
    "action.surface_mode_sas": "SAS (accessible)",
    "action.surface_mode_ses": "SES (molecular)",
    "action.surface_by_hydrophobicity": "By hydrophobicity",
    "action.surface_by_potential": "By potential",
    "action.surface_by_element": "By element",
    "action.surface_range": "Colour range…",
    "action.surface_opacity": "Opacity…",
    "action.surface_pocket": "Pocket lining only",
    "action.surface_highlight": "Highlight the pocket",
    "action.surface_cut_front": "Cut in front of site",
    "action.surface_cut_centre": "Cut through the site",
    "action.surface_cut_clear": "Remove the cut",
    "action.surface_stats": "Surface statistics",
    "action.surface_legend": "Colour bar",
    # The dielectric model of the potential, and the value of the potential
    # decomposed by residue. Both are statements about the *model*, so both
    # spell out their parameters in the label.
    "menu.surface_dielectric": "Dielectric",
    "action.surface_dielectric_distance": "Distance (ε=4r)",
    "action.surface_dielectric_uniform": "Uniform (ε=4)",
    "action.surface_esp_breakdown": "Potential breakdown…",
    "esp.header": "Electrostatic potential at the surface, by residue",
    "esp.model": "model: point charges, dielectric {dielectric}, ε = {epsilon}",
    "esp.charges": (
        "charge column: {nonzero} of {atoms} atoms non-zero, sum {total} e "
        "(a column that does not sum to the molecule's net charge makes this map "
        "relative, not absolute)"
    ),
    "esp.points": "sampled {points} of {available} surface points",
    "esp.focus_total": "potential at the site centre: {value} kcal/(mol·e)",
    "esp.focus_at": "site centre: {x}, {y}, {z}",
    "esp.table": "residue        n   charge   at site   share      mean",
    "log.surface_dielectric": "dielectric: {model}",
    "log.esp_breakdown": (
        "{residues} residues contribute; the largest at the site is {top} "
        "({value} kcal/(mol·e))"
    ),
    "log.surface_open": "the surface is open, so it encloses no volume to report",
    "log.surface_volume": (
        "enclosed volume {volume} Å³ ({cap} lid triangles, {cap_area} Å² of lid)"
    ),
    "dialog.esp_breakdown": "Potential breakdown",
    "action.sasa_report": "SASA & burial…",
    "action.ligand_burial": "Ligand burial",
    "action.burial_per_pose": "Burial per pose",
    "action.export_pymol": "PyMOL script…",
    "action.export_chimerax": "ChimeraX script…",
    "action.export_pose_pdb": "PDB with pose…",
    "action.export_surface_obj": "Surface OBJ…",
    "style.surface": "Surface",
    "property.hydrophobicity": "hydrophobicity",
    "property.electrostatic": "electrostatic potential",
    "property.element": "element",
    "legend.title": "{property}",
    "legend.unit.hydrophobicity": "0 polar → 1 apolar",
    "legend.unit.electrostatic": "kcal/(mol·e)",
    "legend.scale": "{length:.0f} Å",
    "legend.charge_warning": "charges: may not carry formal charge",
    "log.surface_charge_warning": "charge column: {detail}",
    "log.surface_building": "building the {mode} surface for {atoms} atoms…",
    "log.surface_built": (
        "{mode}: {triangles} triangles, {area} Å², grid {spacing} Å, "
        "{seconds} s ({atoms} atoms)"
    ),
    "log.surface_empty": "the surface came out empty for this selection",
    "log.surface_failed": "the surface could not be built",
    "log.surface_no_receptor": "load a receptor before building a surface",
    "log.surface_hidden": "surface hidden",
    "log.surface_shown": "surface shown",
    "log.surface_mode": "surface mode: {mode}",
    "log.surface_property": "surface coloured by {property}",
    "log.surface_range": "surface colour range: {low} to {high}",
    "log.surface_range_auto": "surface colour range back to automatic",
    "log.surface_range_bad": "give two numbers as low, high (0 for automatic): {text}",
    "log.surface_opacity": "surface opacity: {value}",
    "log.surface_pocket": "surface limited to the pocket lining ({atoms} atoms)",
    "log.surface_whole": "surface covers the whole receptor",
    "log.surface_highlight": "highlighting {n} pocket residues on the surface",
    "log.surface_highlight_off": "pocket highlighting off",
    "log.surface_cut": "surface cut at {x}, {y}, {z}",
    "log.surface_cut_cleared": "surface cut removed",
    "log.surface_stats": (
        "{mode} surface: {vertices} vertices, {triangles} triangles, {area} Å², "
        "grid {spacing} Å over {grid} points, probe {probe} Å, built in {seconds} s"
    ),
    "log.sasa_report": "SASA report: {residues} residues, {buried} Å² buried ({share})",
    "log.ligand_burial": (
        "ligand {exposed} Å² exposed of {free} Å²: {buried} Å² in contact ({share})"
    ),
    "log.ligand_burial_none": "load a ligand to measure its buried contact area",
    "log.burial_per_pose": "buried contact area for {n} poses: {best}",
    "log.export_bundle": "exported {n} files for {what} into {name}",
    "dialog.surface_range": "Surface colour range",
    "dialog.surface_opacity": "Surface opacity",
    "dialog.sasa_report": "SASA and burial",
    "dialog.burial_per_pose": "Buried contact area per pose",
    "label.surface_auto": "Range (0 = automatic)",
    "label.surface_min": "low",
    "label.surface_max": "high",
    "label.surface_opacity": "Opacity (0–1)",
    "label.surface_legend": "Colour bar",
    "sasa.header": "Solvent-accessible surface area",
    "sasa.model": (
        "Shrake & Rupley (1973) with a {probe} Å probe over Bondi radii; "
        "{points} sphere points per atom."
    ),
    "sasa.reference.free": "reference: every residue measured on its own",
    "sasa.reference.unbound": "reference: the same structure without the ligand",
    "sasa.total": "total SASA {total} Å², reference {reference} Å², buried {buried} Å² ({share})",
    "sasa.ligand": (
        "ligand: {free} Å² free, {exposed} Å² in the complex, {buried} Å² buried "
        "({share} of its surface)"
    ),
    "sasa.interface": "receptor interface: {buried} Å² buried over {atoms} atoms",
    "sasa.poses": "pose  affinity  buried Å²  buried %",
    "btn.save_figure": "Save figure…",
    "log.figure_saved": "figure written to {name}",
    # The measurement-history table's columns and its CSV button. Added while
    # that table is being built, so the window keeps constructing; the wording
    # matches the measurement tool's own vocabulary.
    "measure.col.kind": "kind",
    "measure.col.atoms": "atoms",
    "measure.csv": "CSV…",
    # The console dock: a Python prompt bound to the live session.
    "dock.console": "Console",
    "console.banner": (
        "Bound to the live session: window, scene, viewport, receptor, ligand, "
        "poses, box, interactions — plus dock(), set_box_center(), "
        "save_project(), frame_binding_site(), export_pymol(). Type help for the list."
    ),
    "console.help_title": "Bound names and helpers",
    "console.no_sandbox": (
        "This console runs in the workbench process with full access — there is "
        "no sandbox."
    ),
    "console.error": "error: {message}",
    "console.continued": "… ",
    "action.console_clear": "Clear console",
    "log.console": "console: {line}",
    # Viewer annotations and undo/redo. These were written against the call
    # sites in app.py while that feature was being added, so the placeholders
    # are the ones the code passes; the wording is deliberately plain.
    "annotation.default": "annotation",
    "annotation.empty": "there is nothing to annotate yet",
    "annotation.on_measurement": "on measurement {index}",
    "log.annotation_added": "annotation added: {text}",
    "log.annotation_no_anchor": "click an atom or pick a residue first",
    "log.annotation_none": "no annotations to copy",
    "log.annotations_copied": "copied {n} annotations",
    "log.annotations_hidden": "annotations hidden",
    "log.annotations_shown": "annotations shown",
    "action.undo": "Undo",
    "action.undo_name": "Undo {name}",
    "action.redo": "Redo",
    "action.redo_name": "Redo {name}",
    "log.undone": "undid {name}",
    "log.redone": "redid {name}",
    "log.nothing_to_undo": "nothing to undo",
    "log.nothing_to_redo": "nothing to redo",
    "undo.add_measurement": "add {kind} measurement",
    "undo.remove_measurement": "remove measurement",
    "undo.clear_measurements": "clear measurements",
    "undo.add_annotation": "add annotation",
    "undo.edit_annotation": "edit annotation",
    "undo.remove_annotation": "remove annotation",
    "log.measure_csv": "wrote {n} measurements to the CSV",
    "log.measure_degenerate": "those atoms are degenerate: the measurement is undefined",
    "log.measure_kind": "measuring {kind}",
    "log.measure_needs": "{kind} needs {need} atoms, {have} picked",
    "log.measurement": "{kind} {value} ({atoms})",
    "status.measure_pick": "{kind}: pick {have} / {need} atoms",
    "annotation.col.text": "text",
    "annotation.col.anchor": "anchor",
    "annotation.col.shown": "shown",
    "annotation.yes": "yes",
    "annotation.no": "no",
    "annotation.count": "{n} annotations",
    # The measurement kinds, the two new submenus and the label dialog. Neither
    # menu title carries an accelerator: the menu bar's eight accelerated
    # titles are fixed by a test.
    "measure.kind.distance": "Distance",
    "measure.kind.angle": "Angle",
    "measure.kind.dihedral": "Dihedral",
    "measure.kind.centroid": "Centroid",
    "measure.kind.plane": "Plane",
    "measure.kind.plane_angle": "Plane ↔ plane",
    "measure.kind.plane_bond": "Plane ↔ bond",
    "menu.measure": "Measure",
    "menu.annotate": "Annotate",
    "action.measure_selection": "Measure selection",
    "action.annotate_add": "Add label…",
    "action.annotate_edit": "Edit label…",
    "action.annotate_delete": "Delete label",
    "action.annotate_show": "Show labels",
    "annotation.dialog.title": "Label",
    "annotation.dialog.text": "Text",
    "annotation.dialog.colour": "Colour",
    "annotation.dialog.anchor": "Anchored to",
}

# ---------------------------------------------------------------------------
# 简体中文
# ---------------------------------------------------------------------------

ZH: Dict[str, str] = {
    # -- application -------------------------------------------------------
    "app.name": "OpenDocking",
    "app.title": "OpenDocking 工作台",
    # -- menu bar ----------------------------------------------------------
    "menu.file": "文件(&F)",
    "menu.export": "导出(&E)",
    "menu.receptor": "受体(&R)",
    "menu.ligand": "配体(&L)",
    "menu.grid": "网格(&G)",
    "menu.docking": "对接(&D)",
    "menu.analysis": "分析(&A)",
    "menu.view": "视图(&V)",
    "menu.protein_style": "蛋白样式",
    "menu.ligand_style": "配体样式",
    "menu.panels": "面板",
    "menu.language": "语言",
    # -- file menu ---------------------------------------------------------
    "action.new_project": "新建项目(&N)",
    "action.open_project": "打开项目(&O)…",
    "action.save_project": "保存项目(&S)…",
    "action.import_receptor": "导入受体(&R)…",
    "action.import_ligand": "导入配体(&L)…",
    "action.import_poses": "导入构象(&P)…",
    "action.fetch_pdb": "获取 PDB(&F)…",
    "action.quit": "退出(&Q)",
    # -- export submenu ----------------------------------------------------
    "action.export_receptor": "受体 PDBQT…",
    "action.export_ligand": "配体 PDBQT…",
    "action.export_poses": "构象 PDBQT…",
    "action.export_cleaned": "清理后 PDB…",
    "action.export_native": "原生配体…",
    "action.export_gpf": "AutoGrid GPF…",
    "action.export_dpf": "AutoDock DPF…",
    "action.export_config": "Vina 配置…",
    "action.export_xlsx": "结果 XLSX…",
    "action.export_csv": "结果 CSV…",
    "action.export_svg": "相互作用 SVG…",
    "action.export_screenshot": "截图…",
    # -- receptor menu -----------------------------------------------------
    "action.remove_waters": "移除水分子",
    "action.strip_ligand": "剥离配体",
    "action.hetero": "离子与辅酶…",
    "action.protonation": "质子化…",
    "action.assign_charges": "赋电荷",
    "action.flexible_residues": "添加柔性残基…",
    "action.write_flex": "导出柔性 PDBQT…",
    # -- ligand menu -------------------------------------------------------
    "action.from_smiles": "由 SMILES…",
    "action.minimise": "能量最小化…",
    "action.ligand_charges": "电荷与并氢",
    "action.detect_torsions": "检测扭转",
    "action.lock_bond": "锁定化学键",
    "action.drug_likeness": "成药性…",
    # -- grid menu ---------------------------------------------------------
    "action.fit_box": "贴合配体",
    "action.centre_box": "居中配体",
    "action.align_residue": "对齐残基…",
    "action.detect_pockets": "探测口袋…",
    "action.show_box": "显示 3-D 网格盒",
    "action.grid_info": "网格信息",
    # -- docking menu ------------------------------------------------------
    "action.use_vina": "使用 Vina",
    "action.use_vinardo": "使用 Vinardo",
    "action.use_ad4": "使用 AutoDock 4",
    "action.start": "开始",
    "action.pause_resume": "暂停 / 继续",
    "action.abort": "中止",
    "action.score": "打分构象",
    "action.engine_settings": "引擎设置",
    # -- analysis menu -----------------------------------------------------
    "action.show_interactions": "显示相互作用",
    "action.clear_annotations": "清除标注",
    "action.cluster_poses": "聚类构象…",
    "action.diagram_svg": "相互作用 SVG…",
    "action.play_poses": "播放构象",
    # -- view menu ---------------------------------------------------------
    "action.axes": "坐标轴",
    "action.ssao": "环境光遮蔽",
    "action.measure": "测量距离",
    "action.clear_measurements": "清除测量",
    "action.reset_view": "重置视图",
    "action.bond_check": "键检查…",
    "action.frame_site": "聚焦结合位点",
    "action.stack_pose_dock": "纵排构象面板",
    "action.box_opacity": "框不透明度…",
    "action.interaction_distances": "相互作用距离…",
    "action.show_reference": "显示参考构象",
    # -- appearance styles -------------------------------------------------
    "style.spheres": "球体",
    "style.sticks": "棍状",
    "style.cartoon": "卡通",
    "style.dots": "点阵",
    "style.ball": "球棍",
    "style.stick": "棍状",
    "style.ribbon": "带状",
    "style.tube": "管状",
    "style.ball_stick": "球棍",
    "style.wireframe": "键线式",
    "style.spacefill": "空间填充",
    # -- interaction cut-offs and the reference pose -----------------------
    "dialog.interaction.title": "相互作用距离",
    "dialog.interaction.hint": (
        "计算构象相互作用时使用的距离截断值（Å）。"
        "碰撞比是两原子范德华半径之和的比例系数。"
    ),
    "label.threshold_hbond": "氢键 (Å)",
    "label.threshold_salt": "盐桥 (Å)",
    "label.threshold_pi": "π-π (Å)",
    "label.threshold_cation_pi": "阳离子-π (Å)",
    "label.threshold_hydrophobic": "疏水 (Å)",
    "label.threshold_clash": "碰撞比",
    "btn.threshold_defaults": "默认值",
    "log.interaction_distances": "相互作用距离：{summary}",
    "log.reference_on": "显示参考构象（{n} 个原子）",
    "log.reference_off": "已隐藏参考构象",
    "label.pose_binding": " · 结合：{residues}",
    "style.role.receptor": "受体",
    "style.role.ligand": "配体",
    # -- docks, tabs and groups -------------------------------------------
    "dock.workspace": "工作区",
    "dock.inspector": "检查器",
    "dock.bottom": "构象与日志",
    "dock.selection": "选择",
    "tab.receptor": "受体",
    "tab.ligand": "配体",
    "tab.grid": "网格",
    "tab.engine": "引擎",
    "group.structure": "结构",
    "group.flexible": "柔性侧链",
    "group.ligand": "配体",
    "group.torsion": "扭转树",
    "group.search_box": "搜索盒",
    "group.pocket": "口袋",
    "group.engine": "引擎",
    "group.run": "运行",
    # -- buttons -----------------------------------------------------------
    "btn.load": "载入…",
    "btn.clean": "清理…",
    "btn.protonate": "质子化…",
    "btn.add_residue": "添加残基…",
    "btn.clear": "清空",
    "btn.write_flex": "导出柔性 PDBQT…",
    "btn.fit_ligand": "贴合配体",
    "btn.centre_ligand": "居中配体",
    "btn.align_residue": "对齐残基…",
    "btn.whole_protein": "整个蛋白",
    "btn.detect_pockets": "探测口袋…",
    "btn.grid_info": "网格信息",
    "btn.from_smiles": "由 SMILES…",
    "btn.minimise": "能量最小化…",
    "btn.drug_likeness": "成药性…",
    "btn.detect_torsions": "检测扭转",
    "btn.lock_bond": "锁定化学键",
    "btn.start": "⚡ 开始对接",
    "btn.pause": "暂停",
    "btn.resume": "继续",
    "btn.abort": "中止",
    "btn.score": "打分构象",
    "btn.play": "▶ 播放",
    "btn.use_pocket": "使用此口袋",
    "btn.copy_pdbqt": "复制为 PDBQT",
    # -- checkboxes --------------------------------------------------------
    "chk.show_receptor": "显示受体",
    "chk.show_ligand": "显示配体",
    "chk.show_box": "显示网格盒",
    "chk.polar_only": "仅保留极性氢",
    "chk.kollman": "Kollman 电荷",
    "chk.keep_waters": "保留结构水",
    # -- panel labels ------------------------------------------------------
    "label.no_receptor": "未载入受体",
    "label.no_ligand": "未载入配体",
    "label.no_ligand_short": "无配体",
    "label.no_poses": "未载入构象",
    "label.no_flex": "无柔性残基",
    "label.no_grid": "未构建网格",
    "label.no_box": "无搜索盒",
    "label.no_hetero": "无杂原子",
    "label.no_sequence": "未载入序列",
    "label.box_opacity": "框不透明度",
    "label.no_selection": "未选择",
    "label.selection_summary": "{counts} · 共 {total} 个原子",
    "label.filters_none": "成药性：未评估",
    "label.filters_pass": "成药性：{verdict}全部过滤规则",
    "label.filters_detail": "（{detail}）",
    "label.pose": "构象",
    "label.structure_n": "{name}（{n} 个原子）",
    "label.flex_n": "{n} 个柔性残基：{names}",
    "label.rotatable_n": "{n} 个可旋转键：{bonds}",
    "label.locked_n": "\n已锁定：{bonds}",
    "label.box_info": "{volume:,.0f} Å³ · 每类 {npts:,} 点 · 每类 {mb:.1f} MB",
    "label.maps_info": (
        "每原子类型 {npts:,} 点 · 各 {mb:.1f} MB · {n} 个受体原子"
    ),
    "label.mode": "构象 {index} / {total}",
    "label.affinity": "   亲和力 {value:.3f} kcal/mol",
    "label.affinity_short": "亲和力 {value:.3f} kcal/mol",
    "label.rmsd_lb": "   RMSD l.b. {value:.3f} Å",
    "label.score": "打分 {value:.3f} kcal/mol",
    "label.none": "无",
    "label.keep_waters_within": "保留范围内的水",
    "label.all_waters": "全部水",
    "label.ph": "pH",
    "label.histidine": "组氨酸",
    "label.charges": "电荷",
    "label.pdb_id": "PDB 编号",
    "label.load_as": "载入为",
    # -- search box form ---------------------------------------------------
    "grid.center_x": "中心 X",
    "grid.center_y": "中心 Y",
    "grid.center_z": "中心 Z",
    "grid.size_x": "尺寸 X",
    "grid.size_y": "尺寸 Y",
    "grid.size_z": "尺寸 Z",
    "grid.spacing": "步长",
    # -- units -------------------------------------------------------------
    "unit.angstrom": " Å",
    "unit.kcal": " kcal/mol",
    "unit.angstrom_of_ligand": " Å 内的配体",
    # -- engine form -------------------------------------------------------
    "engine.force_field": "力场",
    "engine.search": "搜索",
    "engine.exhaustiveness": "穷举度",
    "engine.poses": "构象数",
    "engine.seed": "随机种子",
    "engine.energy_range": "能量范围",
    "engine.islands": "岛屿数",
    "engine.population": "种群规模",
    "engine.generations": "代数",
    "engine.threads": "线程",
    "engine.all_cores": "全部核心",
    "engine.use_grid": "亲和力网格（快）",
    "engine.search_mc": "蒙特卡洛 (ILS)",
    "engine.search_lga": "岛屿 GA (LGA)",
    "engine.search_lga_solis": "岛屿 GA + Solis-Wets",
    # -- table columns -----------------------------------------------------
    "col.rank": "序号",
    "col.energy": "能量 (kcal/mol)",
    "col.rmsd_lb": "RMSD l.b.",
    "col.rmsd_ub": "RMSD u.b.",
    "col.key_residues": "关键残基",
    "col.residue": "残基",
    "col.kind": "类别",
    "col.atoms": "原子数",
    "col.mass": "质量 (Da)",
    "col.keep": "保留",
    "col.index": "#",
    "col.centre": "中心 (x, y, z)",
    "col.volume": "体积 (Å³)",
    "col.score": "评分",
    "col.residues_5a": "≤5 Å 残基",
    "col.ligand": "配体",
    "col.verdict": "结论",
    "col.mw": "MW",
    "col.logp": "LogP",
    "col.hbd": "HBD",
    "col.hba": "HBA",
    "col.rotb": "RotB",
    "col.tpsa": "tPSA",
    "col.violations": "违规项",
    "col.cluster": "簇",
    "col.members": "成员",
    "col.representative": "代表",
    "col.best_affinity": "最佳 ΔG",
    "col.mean_rmsd": "平均 RMSD (Å)",
    "col.modes": "构象",
    "col.key_interactions": "关键接触",
    "col.measure_kind": "类型",
    "col.value": "数值",
    "col.atom_name": "原子",
    "col.element": "元素",
    "col.residue_label": "残基",
    "col.chain": "链",
    "col.x": "x",
    "col.y": "y",
    "col.z": "z",
    "col.ad_type": "AD4 类型",
    "col.charge": "电荷",
    "col.bond_group": "对象",
    "col.bond_bonds": "键数",
    "col.bond_degrees": "度数",
    "col.bond_mean": "平均 (Å)",
    "col.bond_longest": "最长 (Å)",
    "col.bond_issues": "价键",
    # -- workspace tree ----------------------------------------------------
    "tree.project": "项目",
    "tree.atoms": "原子数",
    "tree.receptor": "受体项目",
    "tree.chain": "链 {name}",
    "tree.search_box": "搜索盒（{x:.1f}×{y:.1f}×{z:.1f} Å）",
    "tree.ligand": "配体",
    "tree.bonds": "{n} 个键",
    "tree.pockets": "口袋",
    "tree.pocket": "#{index}  {volume:.0f} Å³",
    "tree.results": "对接结果",
    "tree.pose": "构象 {index}",
    "tree.interactions": "相互作用",
    "tree.measurements": "测量",
    # -- hetero residue kinds ---------------------------------------------
    "kind.cofactor": "辅酶",
    "kind.ion": "离子",
    "kind.ligand": "配体",
    "kind.other": "其他",
    "kind.solvent": "溶剂",
    # -- status bar and worker messages ------------------------------------
    "status.ready": "就绪",
    "status.paused": "已暂停",
    "status.running": "运行中",
    "worker.building_grid": "构建亲和力网格…",
    "worker.searching": "搜索中…",
    "worker.search_done": "搜索完成，用时 {seconds:.1f} 秒",
    "worker.cancelled": "提前停止 — 保留目前找到的最佳构象",
    "worker.no_pause": "此版本不支持暂停",
    "worker.no_resume": "此版本不支持继续",
    "worker.no_abort": "此版本不支持中止",
    "worker.abort_requested": "已请求中止",
    # -- log messages ------------------------------------------------------
    "log.ready": "就绪 — 载入受体或从 PDB 获取",
    "log.language": "语言：{language}",
    "log.new_project": "新项目",
    "log.saved_project": "已保存项目 {name}",
    "log.opened_project": "已打开项目 {name}",
    "log.cannot_read": "无法读取 {path}：{error}",
    "log.error": "错误：{message}",
    "log.receptor": "受体：来自 {name} 的 {n} 个原子",
    "log.ligand": "配体：来自 {name} 的 {n} 个原子",
    "log.poses": "构象：来自 {name} 的 {n} 个模型",
    "log.hetero_groups": "  杂原子组：{groups}",
    "log.rdkit_failed": "RDKit 无法识别该结构",
    "log.rdkit_unavailable": "该结构无法使用 RDKit 识别：{error}",
    "log.hetero_unavailable": "杂原子分类不可用：{error}",
    "log.playing": "正在播放 {n} 个构象",
    "log.load_ligand_first": "请先载入配体",
    "log.load_receptor_first": "请先载入受体",
    "log.no_receptor": "未载入受体",
    "log.no_ligand": "未载入配体",
    "log.no_poses": "未载入构象",
    "log.load_both": "请先载入受体和配体",
    "log.box_centred": "搜索盒已居中于配体",
    "log.box_fitted": (
        "搜索盒已贴合配体 + 8 Å：{x:.1f} × {y:.1f} × {z:.1f} Å"
    ),
    "log.box_whole": "搜索盒覆盖整个蛋白 — 盲对接",
    "log.no_atoms_for": "未找到 {names} 的原子",
    "log.box_on_atoms": "搜索盒已居中于 {names} 的 {n} 个原子",
    "log.no_rec_chem": "受体化学模块不可用",
    "log.waters_removed": "已移除水分子",
    "log.no_cocrystal": "未检测到共结晶配体",
    "log.stripped": "已剥离 {names}",
    "log.kept_reference": "保留 {name} 作为自对接的参考构象",
    "log.extract_failed": "无法提取 {label}：{error}",
    "log.cannot_write_receptor": "无法写出编辑后的受体：{error}",
    "log.kept_residues": "保留了 {n} 个残基",
    "log.hetero_applied": "已应用杂原子策略（保留 {n} 个残基）",
    "log.merged_h": "已合并 {n} 个非极性氢",
    "log.protonated": "已按 pH {ph:.2f} 质子化",
    "log.no_charge_module": "电荷模块不可用",
    "log.charges_assigned": (
        "已赋 {model} 电荷：总计 {total:+.3f} e；AD4 类型 {types}"
    ),
    "log.ligand_charges": (
        "配体的 Gasteiger 电荷：总计 {total:+.3f} e；AD4 类型 {types}"
    ),
    "log.flex_selected": "已选择柔性残基 {name}",
    "log.select_flex_first": "请先选择至少一个柔性残基",
    "log.flex_needs_chem": "柔性残基功能需要受体化学模块",
    "log.wrote_flex": "柔性受体已写入 {name}",
    "log.built_smiles": "已由 SMILES 构建配体（{summary}）",
    "log.no_ligand_chem": "配体化学模块不可用",
    "log.minimised": (
        "已用 {field} 最小化（{steps} 步）：{before} → {after} kcal/mol"
    ),
    "log.rotatable_n": "检测到 {n} 个可旋转键",
    "log.bond_unlocked": "键 {bond} 已解锁 — 自由旋转",
    "log.bond_locked": "键 {bond} 已锁定 — 刚性",
    "log.no_filter_module": "过滤模块不可用（请先载入配体）",
    "log.filters": "成药性：{verdict} — {properties}",
    "log.load_receptor_pocket": "请先载入受体（并安装口袋模块）",
    "log.detecting_pockets": "盲口袋探测中…",
    "log.no_cavity": "未找到超过体积阈值的空腔",
    "log.pockets_found": (
        "检测到 {n} 个口袋；最大的体积 {volume:.0f} Å³，位于 {center}（{residues}）"
    ),
    "log.pocket_aimed": "搜索盒已对准口袋 #{index}",
    "log.no_box": "请先定义搜索盒",
    "log.maps": (
        "亲和力网格每类将保存 {npts:,} 个点（各 {mb:.1f} MB）；对接运行时构建"
    ),
    "log.no_analysis": "分析模块不可用",
    "log.interactions": "相互作用：{counts}",
    "log.annotations_cleared": "已清除标注",
    "log.bond_check": "键检查 — {summary}",
    "log.box_opacity": "框不透明度：{value:.2f}",
    "log.no_box_opacity": "此版本没有框不透明度对话框",
    "log.selection": "已选择 {residues} 个残基、共 {n} 个原子",
    "log.no_selection": "请先在序列标尺中选择残基",
    "log.copied_pdbqt": "已写出 {n} 个原子到 {name}",
    "log.bond_check_empty": "请先载入受体或配体",
    "log.no_bonds_module": "键识别不可用：{error}",
    "log.load_poses_analysis": "请先载入构象（并安装分析模块）",
    "log.clusters": (
        "{cutoff:.1f} Å 处共 {n} 个簇；每个代表是其最佳打分成员"
    ),
    "log.distance": "距离 {a} … {b} = {value:.3f} Å",
    "log.tool_measure": "测量工具已激活 — 点击两个原子",
    "log.tool_bond": "锁键工具已激活 — 点击配体键的两个原子",
    "log.tool_orbit": "相机工具已激活",
    "log.style": "{role}样式：{value}",
    "log.screenshot": "已保存 {width}×{height} 截图到 {name}",
    "log.screenshot_failed": "无法渲染截图",
    "log.exported": "已导出 {what} 到 {name}",
    "log.no_export_module": "导出模块（或受体）不可用",
    "log.no_export_box": "导出模块（或搜索盒）不可用",
    "log.strip_first": "请先剥离共结晶配体",
    "log.run_first": "请先运行对接（并安装报告模块）",
    "log.annotate_first": "请先标注相互作用",
    "log.no_fetch_module": "下载模块不可用",
    "log.downloaded": "已下载 {pdb_id} 到 {path}",
    "log.no_box_defined": "未定义搜索盒",
    "log.run_in_progress": "已有运行中的任务",
    "log.docking_start": (
        "对接：{scoring} · 穷举度 {exhaustiveness} · 种子 {seed} · "
        "搜索盒 {x:.1f}×{y:.1f}×{z:.1f} Å"
    ),
    "log.abort_requested": "已请求中止",
    "log.docking_done": (
        "网格：{npts:,} 个点（{mb} MB）；N_tors={tors:.1f}；{n} 个构象"
    ),
    "log.contacts_hint": "“显示相互作用”可查看接触",
    "log.docking_failed": "对接失败：{message}",
    "log.score": (
        "当前构象打分：{affinity:.3f} kcal/mol"
        "（inter {inter:.3f}，intra {intra:.3f}）"
    ),
    # -- dialogs -----------------------------------------------------------
    "dialog.save_project": "保存项目",
    "dialog.open_project": "打开项目",
    "dialog.open_receptor": "打开受体",
    "dialog.open_ligand": "打开配体",
    "dialog.open_poses": "打开构象",
    "dialog.export": "导出 {what}",
    "dialog.screenshot": "保存截图",
    "dialog.smiles": "由 SMILES 构建配体",
    "dialog.smiles.prompt": "SMILES（可在其后跟名称）：",
    "dialog.minimise": "能量最小化",
    "dialog.minimise.field": "力场：",
    "dialog.minimise.steps": "步数（200–1000）：",
    "dialog.align_residue": "对齐到残基质心",
    "dialog.align_residue.prompt": "残基标签，用逗号分隔（例如 ASP189, GLY216, TRP215）：",
    "dialog.flex_residue": "柔性残基",
    "dialog.flex_residue.prompt": "残基：",
    "dialog.write_flex": "导出柔性受体",
    "dialog.cluster": "聚类构象",
    "dialog.cluster.cutoff": "RMSD 截断值 (Å)：",
    "dialog.hetero.title": "离子与辅酶",
    "dialog.hetero.header": (
        "勾选要在刚性受体中<b>保留</b>的残基。下方距离内的水保留，"
        "其余在<i>应用</i>时移除。"
    ),
    "dialog.protonation.title": "质子化",
    "dialog.protonation.explain": (
        "酸性侧链（Asp、Glu）按 pH 去质子化，碱性侧链（Lys、Arg）质子化；"
        "组氨酸按局部环境取 HID/HIE/HIP。"
    ),
    "dialog.pockets.title": "口袋",
    "dialog.pockets.hint": (
        "以 {spacing:g} Å 探针网格找到 {n} 个空腔。选中一个并点击"
        "“使用此口袋”即可将搜索盒对准它。"
    ),
    "dialog.filters.title": "成药性",
    "dialog.filters.hint": (
        "{total} 个分子中有 {passed} 个通过全部过滤规则（Lipinski、Veber、PAINS）。"
    ),
    "dialog.clusters.title": "构象聚类",
    "dialog.clusters.hint": (
        "在 {cutoff:g} Å 对称感知 RMSD 截断下的单链接聚类：{n} 个簇。"
    ),
    "dialog.fetch.title": "从 PDB 获取",
    "dialog.fetch.placeholder": "例如 3PTB",
    "dialog.fetch.status": "文件将从 files.rcsb.org 下载。",
    "dialog.measurements.title": "测量",
    "dialog.measurements.hint": (
        "在测量工具激活时点击两个原子即可添加距离；点击三个原子添加角度。"
    ),
    "dialog.bond_check.title": "键检查",
    "dialog.box_opacity.title": "框不透明度",
    "dialog.bond_check.hint": (
        "连通性由 odock.gui.bonds 识别；度数是一个原子被赋予的邻居数。"
        "PDBQT 会合并非极性氢，因此重原子显示为低价比属正常。"
    ),
    "dialog.bond_check.orders": "键级：{single} 单键、{double} 双键、{triple} 三键",
    "dialog.bond_check.longest": "最长键：{length:.2f} Å  {a} – {b}",
    "dialog.bond_check.no_bonds": "未识别到任何键",
    "dialog.bond_check.valence_ok": "所有原子的度数都是允许的价态",
    "dialog.bond_check.valence_bad": "需要核对的价态：{items}",
    "dialog.bond_check.isolated": "孤立原子：{n}",
    "dialog.bond_check.cell_ok": "正常",
    "dialog.bond_check.cell_issues": "{over} 超 / {under} 低",
    "dialog.save_selection": "保存选择",
    "fetch.as_receptor": "作为受体",
    "fetch.as_ligand": "作为配体",
    "his.auto": "自动",
    "his.hid": "HID (δ)",
    "his.hie": "HIE (ε)",
    "his.hip": "HIP (+)",
    "verdict.pass": "通过",
    "verdict.fail": "未通过",
    "verdict.passes": "通过",
    "verdict.fails": "未通过",
    # -- export names and file filters ------------------------------------
    "export.receptor_pdbqt": "受体 PDBQT",
    "export.ligand_pdbqt": "配体 PDBQT",
    "export.poses_pdbqt": "构象 PDBQT",
    "export.cleaned_pdb": "清理后的受体 PDB",
    "export.native_ligand": "原生配体",
    "export.gpf": "AutoGrid GPF",
    "export.dpf": "AutoDock DPF",
    "export.config": "Vina 配置",
    "export.xlsx": "结果 XLSX",
    "export.csv": "结果 CSV",
    "export.svg": "相互作用图",
    "export.pymol": "PyMOL 脚本",
    "export.chimerax": "ChimeraX 脚本",
    "export.pose_pdb": "含构象的 PDB",
    "export.surface_obj": "表面网格 OBJ",
    "filter.pdbqt": "PDBQT (*.pdbqt)",
    "filter.pdb": "PDB (*.pdb)",
    "filter.gpf": "GPF (*.gpf)",
    "filter.dpf": "DPF (*.dpf)",
    "filter.config": "配置 (*.conf *.txt)",
    "filter.xlsx": "Excel (*.xlsx)",
    "filter.csv": "CSV (*.csv)",
    "filter.svg": "SVG (*.svg)",
    "filter.png": "PNG (*.png)",
    "filter.project": "OpenDocking 项目 (*.json)",
    "filter.structures": (
        "结构 (*.pdbqt *.pdb *.mol2 *.sdf);;所有文件 (*)"
    ),
    "filter.ligands": "配体 (*.pdbqt *.sdf *.mol2 *.mol *.smi);;所有文件 (*)",
    "filter.all_files": "所有文件 (*)",
    # -- 3-D viewport ------------------------------------------------------
    "tip.reset_view": "重置视图",
    "tip.frame_ligand": "聚焦配体",
    "tip.frame_box": "聚焦搜索盒",
    "tip.measure": "测量距离",
    "tip.bond": "锁定/解锁化学键",
    "tip.screenshot": "保存截图",
    "tip.sequence": "选择 · Shift：范围 · Ctrl：多选 · 双击：居中",
    "viewport.no_gl": (
        "没有可用的 OpenGL 3.3 上下文，3-D 视图已停用。\n"
        "{detail}\n\n对接、打分和构象表仍可使用。"
    ),
    "viewport.render_error": "无法绘制 3-D 视图。\n{detail}",
    "viewport.hint_measure": "测量：点击两个原子",
    "viewport.hint_bond": "锁键：点击化学键的两个原子",
    # -- interaction legend (3-D HUD) --------------------------------------
    "interaction.hbond": "氢键",
    "interaction.salt_bridge": "盐桥",
    "interaction.pi_pi": "π-π 堆积",
    "interaction.cation_pi": "阳离子-π",
    "interaction.hydrophobic": "疏水",
    "interaction.clash": "位阻冲突",
    # -- measurement export and CLI ---------------------------------------
    "svg.title": "OpenDocking 相互作用图",
    "cli.receptor": "要载入的受体 PDBQT",
    "cli.ligand": "要载入的配体 PDBQT",
    "cli.poses": "要载入的构象 PDBQT",
    # -- run dashboard -----------------------------------------------------
    "dock.dashboard": "运行监视",
    "dock.comparison": "构象对比",
    "tab.run": "运行",
    "tab.measure": "测量",
    "phase.grid": "网格",
    "phase.search": "搜索",
    "phase.refine": "精修",
    "phase.done": "完成",
    "dashboard.best": "最佳亲和力",
    "dashboard.elapsed": "已用时间",
    "dashboard.poses": "构象数",
    "dashboard.grid": "网格",
    "dashboard.scoring": "力场",
    "dashboard.note": (
        "内核在每次运行返回时才给出精修后的构象能量，因此曲线在运行结束时出现；"
        "阶段与计时是实时的。"
    ),
    "dashboard.status.starting": "正在准备搜索盒",
    "dashboard.status.cancelled": "已中止 — 显示已找到的构象",
    "dashboard.status.failed": "运行失败",
    "plot.axis.iteration": "构象",
    "plot.axis.energy": "kcal/mol",
    "plot.empty": "运行一次对接：每个报告的构象都会在这里画一个点。",
    "plot.best": "最佳 {value}",
    # -- measurements ------------------------------------------------------
    "measure.empty": "尚无测量：选择测量工具并点击两个原子。",
    "measure.count": "已测量 {n} 项",
    "measure.col.a": "原子 A",
    "measure.col.b": "原子 B",
    "measure.col.value": "距离 (Å)",
    "measure.copy": "复制列表",
    "measure.clear": "清空",
    # -- pose comparison ---------------------------------------------------
    "compare.hint": (
        "在结果表中按住 Ctrl 点击两行即可对比：对称等效 RMSD、亲和力之差，"
        "以及各自接触的残基。"
    ),
    "compare.value.placeholder": "—",
    "compare.heading": "构象 {a} 与构象 {b}",
    "compare.rmsd": "对称等效 RMSD",
    "compare.rmsd.value": "叠合后 {fitted:.2f} Å · 原位 {in_place:.2f} Å · {n} 个重原子",
    "compare.delta": "亲和力之差",
    "compare.delta.value": "{delta:+.2f} kcal/mol（构象 {a} → 构象 {b}）",
    "compare.verdict.same": "— 同一种结合模式",
    "compare.verdict.different": "— 不同的结合模式",
    "compare.contacts": "接触指纹（{cutoff:.1f} Å 以内）",
    "compare.contacts.counts": "共有 {shared} · 仅第一个 {only_a} · 仅第二个 {only_b}",
    "compare.both": "两者",
    "compare.only_a": "仅构象 {index}",
    "compare.only_b": "仅构象 {index}",
    "compare.col.residue": "残基",
    "compare.col.which": "接触者",
    "compare.col.distance": "距离 A / B (Å)",
    "compare.copy": "复制报告",
    "log.compare_copied": "对比结果已复制到剪贴板",
    # -- command palette ---------------------------------------------------
    "palette.title": "命令面板",
    "palette.placeholder": "输入命令…",
    "palette.hint": "共 {n} 条命令 · 显示 {shown} 条 · 回车执行所选",
    "palette.empty": "没有匹配的菜单项",
    # -- interaction lines -------------------------------------------------
    "tab.interactions": "相互作用",
    "menu.interaction_lines": "绘制的连线",
    "action.interaction_all": "显示全部类型",
    "interactions.empty": "尚无相互作用连线 — 使用「分析 ▸ 显示相互作用」按当前构象绘制。",
    "interactions.count": "视图中有 {n} 条连线",
    "interactions.hidden": "{n} 条被筛选隐藏",
    "interactions.col.kind": "类型",
    "interactions.col.receptor": "受体原子",
    "interactions.col.ligand": "配体原子",
    "interactions.col.distance": "距离 (Å)",
    "interaction.legend_convention": "每条虚线连接一个配体原子与一个残基原子",
    "log.interaction_filter": "显示 {total} 条相互作用中的 {shown} 条",
    "log.interactions_copied": "已复制 {n} 条相互作用到剪贴板",
    # -- themes, density, layouts and the session --------------------------
    "menu.theme": "主题",
    "menu.density": "密度",
    "menu.layout": "布局",
    "menu.recent": "最近文件",
    "action.palette": "命令面板…",
    "action.theme_dark": "深色",
    "action.theme_light": "浅色",
    "action.density_comfortable": "宽松",
    "action.density_compact": "紧凑",
    "action.layout_docking": "对接",
    "action.layout_analysis": "分析",
    "action.layout_compare": "对比",
    "action.copy_view": "复制视图",
    "action.inspect_atom": "原子读数",
    "action.compare_poses": "对比构象",
    "action.recent_empty": "没有最近文件",
    "action.recent_clear": "清空列表",
    "action.restore_session": "恢复会话",
    "log.theme": "外观：{theme} · 密度：{density}",
    "log.preset": "布局：{name}",
    "log.dropped": "拖入{what}：{name}",
    "drop.receptor": "受体",
    "drop.ligand": "配体",
    "drop.poses": "构象",
    "log.drop_unsupported": "无法打开 {name}：工作台可载入 PDBQT 与 PDB 结构",
    "log.copy_view": "3-D 视图已复制到剪贴板",
    "log.copy_view_failed": "无法复制 3-D 视图",
    "log.compare": "正在对比构象 {a} 与构象 {b}",
    "log.compare_failed": "无法对比这两个构象：{message}",
    "log.compare_unavailable": "请先载入至少两个构象再进行对比",
    "log.measure_copied": "已复制 {n} 条测量到剪贴板",
    "log.measure_cleared": "测量已清空",
    "log.session_saved": "会话已自动保存（{path}）",
    "log.session_restored": "已从 {name} 恢复会话",
    "log.session_none": "没有可恢复的会话",
    "log.session_cleared": "已丢弃保存的会话",
    "log.recent_missing": "{name} 已不在磁盘上",
    "dialog.restore_session": "恢复会话",
    "dialog.restore_session.text": "是否继续上次的会话？\n\n{detail}",
    "dialog.restore_session.detail": "受体：{receptor}\n配体：{ligand}\n构象：{poses}",
    "dialog.restore_session.none": "无",
    "inspect.source": "来源",
    "inspect.source.receptor": "受体",
    "inspect.source.ligand": "配体",
    "inspect.residue": "残基",
    "inspect.atom": "原子",
    "inspect.element": "元素",
    "inspect.ad_type": "AD4 类型",
    "inspect.charge": "电荷",
    "inspect.position": "坐标",
    # -- 分子表面 ----------------------------------------------------------
    "menu.surface": "表面",
    "menu.surface_mode": "表面类型",
    "menu.surface_colour": "着色依据",
    "action.surface_show": "显示表面",
    "action.surface_rebuild": "重建表面",
    "action.surface_mode_sas": "SAS（可及表面）",
    "action.surface_mode_ses": "SES（分子表面）",
    "action.surface_by_hydrophobicity": "按疏水性着色",
    "action.surface_by_potential": "按静电势着色",
    "action.surface_by_element": "按元素着色",
    "action.surface_range": "颜色范围…",
    "action.surface_opacity": "不透明度…",
    "action.surface_pocket": "仅口袋内衬",
    "action.surface_highlight": "高亮口袋残基",
    "action.surface_cut_front": "在结合位点前剖切",
    "action.surface_cut_centre": "从结合位点剖切",
    "action.surface_cut_clear": "取消剖切",
    "action.surface_stats": "表面统计",
    "action.surface_legend": "色标",
    "menu.surface_dielectric": "介电常数",
    "action.surface_dielectric_distance": "距离依赖 (ε=4r)",
    "action.surface_dielectric_uniform": "固定介电 (ε=4)",
    "action.surface_esp_breakdown": "电势分解…",
    "esp.header": "表面静电势的残基分解",
    "esp.model": "模型：点电荷，介电 {dielectric}，ε = {epsilon}",
    "esp.charges": (
        "电荷列：{atoms} 个原子中 {nonzero} 个非零，总和 {total} e"
        "（若总和不等于分子净电荷，该图为相对值而非绝对值）"
    ),
    "esp.points": "在 {available} 个表面点中采样 {points} 个",
    "esp.focus_total": "位点中心电势：{value} kcal/(mol·e)",
    "esp.focus_at": "位点中心：{x}, {y}, {z}",
    "esp.table": "残基          原子数  电荷   位点处   占比      均值",
    "log.surface_dielectric": "介电模型：{model}",
    "log.esp_breakdown": "{residues} 个残基有贡献；位点处最大者为 {top}（{value} kcal/(mol·e)）",
    "log.surface_open": "表面未闭合，无法给出体积",
    "log.surface_volume": "封闭体积 {volume} Å³（{cap} 个封盖三角形，{cap_area} Å² 封盖）",
    "dialog.esp_breakdown": "电势分解",
    "action.sasa_report": "SASA 与包埋…",
    "action.ligand_burial": "配体包埋面积",
    "action.burial_per_pose": "逐构象包埋面积",
    "action.export_pymol": "PyMOL 脚本…",
    "action.export_chimerax": "ChimeraX 脚本…",
    "action.export_pose_pdb": "含构象的 PDB…",
    "action.export_surface_obj": "表面 OBJ…",
    "style.surface": "表面",
    "property.hydrophobicity": "疏水性",
    "property.electrostatic": "静电势",
    "property.element": "元素",
    "legend.title": "{property}",
    "legend.unit.hydrophobicity": "0 亲水 → 1 疏水",
    "legend.unit.electrostatic": "kcal/(mol·e)",
    "legend.scale": "{length:.0f} Å",
    "legend.charge_warning": "电荷：可能未体现形式电荷",
    "log.surface_charge_warning": "电荷列：{detail}",
    "log.surface_building": "正在为 {atoms} 个原子构建 {mode} 表面…",
    "log.surface_built": (
        "{mode}：{triangles} 个三角形，{area} Å²，网格 {spacing} Å，"
        "耗时 {seconds} 秒（{atoms} 个原子）"
    ),
    "log.surface_empty": "该选择下表面为空",
    "log.surface_failed": "表面构建失败",
    "log.surface_no_receptor": "请先载入受体再构建表面",
    "log.surface_hidden": "已隐藏表面",
    "log.surface_shown": "已显示表面",
    "log.surface_mode": "表面模式：{mode}",
    "log.surface_property": "表面按{property}着色",
    "log.surface_range": "表面颜色范围：{low} 至 {high}",
    "log.surface_range_auto": "表面颜色范围恢复自动",
    "log.surface_range_bad": "请输入两个数字：下限, 上限（0 表示自动）：{text}",
    "log.surface_opacity": "表面不透明度：{value}",
    "log.surface_pocket": "表面仅覆盖口袋内衬（{atoms} 个原子）",
    "log.surface_whole": "表面覆盖整个受体",
    "log.surface_highlight": "在表面上高亮 {n} 个口袋残基",
    "log.surface_highlight_off": "已关闭口袋高亮",
    "log.surface_cut": "表面剖切面位于 {x}, {y}, {z}",
    "log.surface_cut_cleared": "已取消表面剖切",
    "log.surface_stats": (
        "{mode} 表面：{vertices} 顶点，{triangles} 三角形，{area} Å²，"
        "网格 {spacing} Å 共 {grid} 点，探针 {probe} Å，耗时 {seconds} 秒"
    ),
    "log.sasa_report": "SASA 报告：{residues} 个残基，包埋 {buried} Å²（{share}）",
    "log.ligand_burial": (
        "配体 {free} Å² 中暴露 {exposed} Å²：接触面积 {buried} Å²（{share}）"
    ),
    "log.ligand_burial_none": "请先载入配体以测量其包埋接触面积",
    "log.burial_per_pose": "{n} 个构象的包埋接触面积：{best}",
    "log.export_bundle": "已将 {what} 的 {n} 个文件导出到 {name}",
    "dialog.surface_range": "表面颜色范围",
    "dialog.surface_opacity": "表面不透明度",
    "dialog.sasa_report": "SASA 与包埋",
    "dialog.burial_per_pose": "逐构象包埋接触面积",
    "label.surface_auto": "范围（0 表示自动）",
    "label.surface_min": "下限",
    "label.surface_max": "上限",
    "label.surface_opacity": "不透明度（0–1）",
    "label.surface_legend": "色标",
    "sasa.header": "溶剂可及表面积",
    "sasa.model": (
        "Shrake & Rupley (1973)，探针 {probe} Å，Bondi 半径；"
        "每个原子 {points} 个球面采样点。"
    ),
    "sasa.reference.free": "参考态：每个残基单独测量",
    "sasa.reference.unbound": "参考态：不含配体的同一结构",
    "sasa.total": "总 SASA {total} Å²，参考 {reference} Å²，包埋 {buried} Å²（{share}）",
    "sasa.ligand": "配体：游离 {free} Å²，复合物中 {exposed} Å²，包埋 {buried} Å²（占其表面 {share}）",
    "sasa.interface": "受体界面：{atoms} 个原子共包埋 {buried} Å²",
    "sasa.poses": "构象  亲和力  包埋 Å²  包埋 %",
    "btn.save_figure": "保存图…",
    "log.figure_saved": "图已写入 {name}",
    "measure.col.kind": "类型",
    "measure.col.atoms": "原子",
    "measure.csv": "CSV…",
    "dock.console": "终端",
    "console.banner": (
        "已绑定当前会话：window、scene、viewport、receptor、ligand、poses、box、"
        "interactions，以及 dock()、set_box_center()、save_project()、"
        "frame_binding_site()、export_pymol()。输入 help 查看列表。"
    ),
    "console.help_title": "已绑定的名称与便捷函数",
    "console.no_sandbox": "该终端在工作台进程内运行，拥有完整权限 —— 没有沙箱。",
    "console.error": "错误：{message}",
    "console.continued": "… ",
    "action.console_clear": "清空终端",
    "log.console": "终端：{line}",
    "annotation.default": "标注",
    "annotation.empty": "当前没有可标注的对象",
    "annotation.on_measurement": "测量 {index}",
    "log.annotation_added": "已添加标注：{text}",
    "log.annotation_no_anchor": "请先点击原子或选择残基",
    "log.annotation_none": "没有可复制的标注",
    "log.annotations_copied": "已复制 {n} 条标注",
    "log.annotations_hidden": "已隐藏标注",
    "log.annotations_shown": "已显示标注",
    "action.undo": "撤销",
    "action.undo_name": "撤销 {name}",
    "action.redo": "重做",
    "action.redo_name": "重做 {name}",
    "log.undone": "已撤销 {name}",
    "log.redone": "已重做 {name}",
    "log.nothing_to_undo": "没有可撤销的操作",
    "log.nothing_to_redo": "没有可重做的操作",
    "undo.add_measurement": "添加{kind}测量",
    "undo.remove_measurement": "删除测量",
    "undo.clear_measurements": "清空测量",
    "undo.add_annotation": "添加标注",
    "undo.edit_annotation": "编辑标注",
    "undo.remove_annotation": "删除标注",
    "log.measure_csv": "已将 {n} 条测量写入 CSV",
    "log.measure_degenerate": "所选原子退化，该测量无定义",
    "log.measure_kind": "正在测量{kind}",
    "log.measure_needs": "{kind} 需要 {need} 个原子，已选 {have} 个",
    "log.measurement": "{kind} {value}（{atoms}）",
    "status.measure_pick": "{kind}：已选 {have} / {need} 个原子",
    "annotation.col.text": "文本",
    "annotation.col.anchor": "锚点",
    "annotation.col.shown": "显示",
    "annotation.yes": "是",
    "annotation.no": "否",
    "annotation.count": "{n} 条标注",
    "measure.kind.distance": "距离",
    "measure.kind.angle": "角度",
    "measure.kind.dihedral": "二面角",
    "measure.kind.centroid": "质心",
    "measure.kind.plane": "平面",
    "measure.kind.plane_angle": "平面↔平面",
    "measure.kind.plane_bond": "平面↔键",
    "menu.measure": "测量",
    "menu.annotate": "标注",
    "action.measure_selection": "测量所选",
    "action.annotate_add": "添加标注…",
    "action.annotate_edit": "编辑标注…",
    "action.annotate_delete": "删除标注",
    "action.annotate_show": "显示标注",
    "annotation.dialog.title": "标注",
    "annotation.dialog.text": "文本",
    "annotation.dialog.colour": "颜色",
    "annotation.dialog.anchor": "锚定于",
}


def _default_language() -> str:
    """Chinese when the system locale says so, English otherwise."""
    try:
        code = locale.getdefaultlocale()[0] or ""
    except Exception:  # pragma: no cover - depends on the Python build
        code = ""
    return "zh" if str(code).lower().startswith("zh") else "en"


#: The language every ``tr`` call resolves against.
_language: str = _default_language()


def current_language() -> str:
    """The language code in use (``"en"`` or ``"zh"``)."""
    return _language


def language_name(code: str) -> str:
    """The name of ``code`` as the user sees it (its own endonym)."""
    return LANGUAGES[code]


def set_language(code: str) -> None:
    """Set the process-wide language.

    The workbench does not observe this by itself: it calls
    :meth:`~odock.gui.app.DockingWorkbench.set_language`, which flips this
    setting and then rebuilds every widget so the new labels take effect.
    """
    if code not in LANGUAGES:
        raise ValueError(
            f"unknown language {code!r}; expected one of "
            f"{', '.join(sorted(LANGUAGES))}"
        )
    global _language
    _language = code


def available_keys() -> Set[str]:
    """Every key that is safe to pass to :func:`tr`."""
    return set(EN)


def _table() -> Dict[str, str]:
    return ZH if _language == "zh" else EN


def tr(key: str, **kwargs) -> str:
    """The string for ``key`` in the current language.

    Raises :class:`KeyError` when the key is unknown: a missing translation is
    a bug, and silently showing the other language would hide it.
    """
    try:
        text = _table()[key]
    except KeyError:
        raise KeyError(f"no translation for {key!r}") from None
    # Always format: a caller that forgets a placeholder argument fails loudly
    # instead of showing a raw ``{name}`` to the user.
    return text.format(**kwargs)


def tr_ctx(key: str, **kwargs) -> str:
    """Like :func:`tr`, for module-level tables built at import time."""
    return tr(key, **kwargs)

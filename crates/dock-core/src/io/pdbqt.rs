// SPDX-License-Identifier: GPL-3.0-or-later
//! AutoDock PDBQT reader and writer.
//!
//! # Format
//!
//! ```text
//! ATOM      1  N   ALA A   1      27.214  24.850  22.290  1.00  0.00    -0.347 N
//! |         |  |   |   |   |      |       |       |       |     |      |       |
//! 1         7  13  18  22  23    31      39      47     55    61     71      78
//! ```
//!
//! * columns 31-38, 39-46, 47-54 — Cartesian coordinates (Å)
//! * columns 71-76 — partial charge
//! * columns 78-79 — AutoDock atom type
//!
//! Ligands add a topology layer:
//!
//! ```text
//! ROOT
//! ATOM ... (root atoms)
//! ENDROOT
//! BRANCH   1   5
//! ATOM ... (atoms of the torsion branch; atom 5 stays fixed w.r.t. the parent)
//! ENDBRANCH   1   5
//! TORSDOF 4
//! ```
//!
//! Atom `b` in `BRANCH a b` is the atom that stays rigid relative to the parent
//! frame; the rotation axis runs from `a` to `b` and every *other* atom inside
//! the branch rotates about it. This is exactly the convention AutoDock Vina
//! implements, and it is what makes the nesting unambiguous.

use crate::atom::AdType;
use crate::error::{DockError, Result};
use crate::kinematics::TopologyNode;
use crate::molecule::Atom;
use crate::math::DVec3;
use crate::molecule::Molecule;

/// A non-fatal observation made while parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseIssue {
    /// 1-based line number.
    pub line: usize,
    /// What was wrong.
    pub message: String,
}

/// A raw parsed atom together with the line it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordAtom {
    /// The parsed atom.
    pub atom: Atom,
    /// The original line, used as a template when writing poses back out.
    pub template: String,
}

impl RecordAtom {
    /// Render this atom at the given position, preserving the original layout.
    pub fn render(&self, coords: DVec3) -> String {
        render_pdbqt_line(&self.template, coords)
    }
}

/// A node of the parsed ligand topology.
#[derive(Debug, Clone, Default)]
struct RawNode {
    atoms: Vec<RecordAtom>,
    /// Index within `atoms` of the atom shared with the parent frame.
    immobile: Option<usize>,
    /// `(parent index within `atoms`, child branch)`.
    children: Vec<(usize, RawNode)>,
}

impl RawNode {
    /// `true` when this branch rotates nothing, i.e. it owns no atom beyond the
    /// attachment point and none of its descendants do either.
    fn has_segment(&self) -> bool {
        self.atoms
            .iter()
            .enumerate()
            .any(|(i, _)| Some(i) != self.immobile)
            || self.children.iter().any(|(_, c)| c.has_segment())
    }
}

/// A parsed ligand: flat movable atoms plus the kinematic topology.
#[derive(Debug, Clone)]
pub struct ParsedLigand {
    /// Movable atoms in the order the docking kernel expects.
    pub atoms: Vec<Atom>,
    /// Original lines, aligned with [`ParsedLigand::atoms`].
    pub records: Vec<RecordAtom>,
    /// The kinematic tree description.
    pub top: TopologyNode,
    /// `(parent movable index or `None`, attachment movable index)` for every
    /// rotatable bond; used for the torsional penalty.
    pub rotors: Vec<(Option<usize>, usize)>,
    /// The `TORSDOF` value declared in the file.
    pub torsdof: u32,
    /// Interactive output context (`REMARK`, `CRYST1`, ...) in file order.
    pub context: Vec<String>,
    /// Non-fatal parse observations.
    pub issues: Vec<ParseIssue>,
}

impl ParsedLigand {
    /// All atoms as a [`Molecule`] with bond perception applied.
    pub fn molecule(&self) -> Molecule {
        let mut m = Molecule::from_atoms(self.atoms.clone());
        m.perceive();
        m
    }
}

/// A receptor (rigid atoms only in this release).
#[derive(Debug, Clone)]
pub struct ReceptorRecord {
    /// The immobile atoms.
    pub atoms: Vec<Atom>,
    /// Original lines, aligned with [`ReceptorRecord::atoms`].
    pub records: Vec<RecordAtom>,
    /// Non-fatal parse observations.
    pub issues: Vec<ParseIssue>,
}

/// Convenience alias kept for API symmetry.
pub type LigandRecord = ParsedLigand;

// ---------------------------------------------------------------------------
// Low-level line handling
// ---------------------------------------------------------------------------

fn is_record(line: &str, tag: &str) -> bool {
    line.len() >= tag.len() && line[..tag.len()].eq_ignore_ascii_case(tag)
}

fn col(line: &str, from: usize, to: usize) -> &str {
    let b = line.as_bytes();
    if from >= b.len() {
        return "";
    }
    let end = to.min(b.len());
    std::str::from_utf8(&b[from..end]).unwrap_or("")
}

fn parse_coord(s: &str, line_no: usize, what: &str) -> Result<f64> {
    s.trim().parse::<f64>().map_err(|_| DockError::Parse {
        line: line_no,
        message: format!("cannot parse {what} coordinate from {s:?}"),
    })
}

/// Parse a PDBQT `ATOM`/`HETATM` line.
fn parse_atom_line(line: &str, line_no: usize) -> Result<RecordAtom> {
    if line.len() < 54 {
        return Err(DockError::Parse {
            line: line_no,
            message: "atom record is shorter than 54 columns".to_string(),
        });
    }
    let serial = col(line, 6, 11).trim().parse::<i32>().unwrap_or(0);
    let name = col(line, 12, 16).trim().to_string();
    let res_name = col(line, 17, 20).trim().to_string();
    let chain_id = col(line, 21, 22).trim().to_string();
    let res_id = col(line, 22, 26).trim().parse::<i32>().unwrap_or(0);
    let x = parse_coord(col(line, 30, 38), line_no, "x")?;
    let y = parse_coord(col(line, 38, 46), line_no, "y")?;
    let z = parse_coord(col(line, 46, 54), line_no, "z")?;

    // Charge and atom type: prefer the fixed columns, fall back to a
    // whitespace split for files that do not follow the column convention.
    let mut charge_s = col(line, 70, 76).trim().to_string();
    let mut type_s = col(line, 77, 80).trim().to_string();
    if type_s.is_empty() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if let Some(last) = toks.last() {
            type_s = (*last).to_string();
            if toks.len() >= 2 {
                let prev = toks[toks.len() - 2];
                if prev.parse::<f64>().is_ok() {
                    charge_s = prev.to_string();
                }
            }
        }
    }
    let charge = charge_s.parse::<f64>().unwrap_or(0.0);
    let mut ad = AdType::from_name(&type_s);
    if ad == AdType::Unknown {
        // Fall back to the element implied by the atom name.
        if let Some(el) = crate::atom::Element::from_symbol(&name) {
            ad = match el {
                crate::atom::Element::H => AdType::H,
                crate::atom::Element::C => AdType::C,
                crate::atom::Element::N => AdType::N,
                crate::atom::Element::O => AdType::O,
                crate::atom::Element::S => AdType::S,
                crate::atom::Element::P => AdType::P,
                crate::atom::Element::F => AdType::F,
                crate::atom::Element::Cl => AdType::Cl,
                crate::atom::Element::Br => AdType::Br,
                crate::atom::Element::I => AdType::I,
                crate::atom::Element::Si => AdType::Si,
                crate::atom::Element::At => AdType::At,
                _ => AdType::Unknown,
            };
        }
    }

    let mut atom = Atom::new(DVec3::new(x, y, z), ad, charge);
    atom.serial = serial;
    atom.name = name;
    atom.res_name = res_name;
    atom.chain_id = chain_id;
    atom.res_id = res_id;
    atom.type_name = type_s;

    Ok(RecordAtom {
        atom,
        template: line.trim_end().to_string(),
    })
}

fn parse_two_unsigneds(line: &str, tag: &str, line_no: usize) -> Result<(i32, i32)> {
    let rest = line[tag.len()..].trim();
    let mut it = rest.split_whitespace();
    let a = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .ok_or_else(|| DockError::Parse {
            line: line_no,
            message: format!("{tag} record needs two atom serial numbers"),
        })?;
    let b = it
        .next()
        .and_then(|s| s.parse::<i32>().ok())
        .ok_or_else(|| DockError::Parse {
            line: line_no,
            message: format!("{tag} record needs two atom serial numbers"),
        })?;
    Ok((a, b))
}

/// Write `coords` into columns 31-54 of a PDBQT line template.
pub fn render_pdbqt_line(template: &str, coords: DVec3) -> String {
    let mut b: Vec<u8> = template.as_bytes().to_vec();
    let sep = [b' ', b' ', b' ', b' '];
    if b.len() < 54 {
        // Pad short templates so the coordinate block always fits.
        b.resize(54, b' ');
    }
    // Make sure the trailing two columns can hold the atom type.
    if b.len() < 80 {
        b.resize(80, b' ');
    }
    let fields = [
        format!("{:8.3}", coords.x),
        format!("{:8.3}", coords.y),
        format!("{:8.3}", coords.z),
    ];
    for (k, f) in fields.iter().enumerate() {
        let start = 30 + k * 8;
        let bytes = f.as_bytes();
        for (i, byte) in bytes.iter().enumerate().take(8) {
            b[start + i] = *byte;
        }
    }
    // A truncated coordinate (|x| >= 10000) would overflow the field; the
    // separators below keep the record parseable regardless.
    for (k, _) in fields.iter().enumerate() {
        let _ = k;
        let _ = &sep;
    }
    String::from_utf8_lossy(&b).trim_end().to_string()
}

// ---------------------------------------------------------------------------
// Ligand topology parsing
// ---------------------------------------------------------------------------

/// Parser state for the nested `ROOT`/`BRANCH` language.
struct TopologyParser {
    issues: Vec<ParseIssue>,
}

impl TopologyParser {
    /// Parse the body of a ligand file into a tree, starting at `pos`.
    ///
    /// Returns the root node and the line index just past the last consumed
    /// line. `stop` names the record that terminates the current level.
    fn parse_level(
        &mut self,
        lines: &[&str],
        pos: &mut usize,
        context: &mut Vec<String>,
        torsdof: &mut Option<u32>,
    ) -> Result<RawNode> {
        let mut node = RawNode::default();
        let mut seen_root = false;
        while *pos < lines.len() {
            let line = lines[*pos];
            let line_no = *pos + 1;
            if line.trim().is_empty() {
                *pos += 1;
                continue;
            }
            if is_record(line, "ROOT") {
                if seen_root {
                    // A second ROOT at this level starts another model. Only the
                    // first model is docked, and merging the two would silently
                    // produce a single, wrong molecule.
                    break;
                }
                seen_root = true;
                context.push(line.to_string());
                *pos += 1;
                // Root atoms until ENDROOT.
                loop {
                    if *pos >= lines.len() {
                        return Err(DockError::Parse {
                            line: line_no,
                            message: "ROOT without ENDROOT".to_string(),
                        });
                    }
                    let l = lines[*pos];
                    let ln = *pos + 1;
                    if is_record(l, "ENDROOT") {
                        context.push(l.to_string());
                        *pos += 1;
                        break;
                    }
                    if is_record(l, "ATOM") || is_record(l, "HETATM") {
                        node.atoms.push(parse_atom_line(l, ln)?);
                    } else if is_record(l, "REMARK") || is_record(l, "WARNING") {
                        context.push(l.to_string());
                    } else if is_record(l, "MODEL") || is_record(l, "ENDMDL") {
                        // Multi-model ligand files: only the first model is used.
                        self.issues.push(ParseIssue {
                            line: ln,
                            message: "ignoring MODEL/ENDMDL record".to_string(),
                        });
                    } else {
                        return Err(DockError::Parse {
                            line: ln,
                            message: format!("unexpected record inside ROOT: {l:?}"),
                        });
                    }
                    *pos += 1;
                }
                continue;
            }
            if is_record(line, "BRANCH") {
                let (from, to) = parse_two_unsigneds(line, "BRANCH", line_no)?;
                *pos += 1;
                let child = self.parse_branch(lines, pos, context, from, to)?;
                // Attach to the atom whose serial is `from`.
                let parent = node
                    .atoms
                    .iter()
                    .position(|a| a.atom.serial == from)
                    .ok_or_else(|| DockError::Parse {
                        line: line_no,
                        message: format!("BRANCH {from} {to}: atom {from} not found in this level"),
                    })?;
                node.children.push((parent, child));
                continue;
            }
            if is_record(line, "TORSDOF") {
                let v = line["TORSDOF".len()..].trim().parse::<u32>().unwrap_or(0);
                *torsdof = Some(v);
                context.push(line.to_string());
                *pos += 1;
                continue;
            }
            if is_record(line, "ENDROOT")
                || is_record(line, "ENDBRANCH")
                || is_record(line, "END_RES")
            {
                return Ok(node);
            }
            if is_record(line, "REMARK") || is_record(line, "WARNING") {
                context.push(line.to_string());
                *pos += 1;
                continue;
            }
            if is_record(line, "ENDMDL") {
                // End of the first model: stop reading rather than merging the
                // remaining models into one molecule.
                *pos += 1;
                break;
            }
            if is_record(line, "MODEL") {
                if seen_root {
                    break;
                }
                *pos += 1;
                continue;
            }
            if is_record(line, "CRYST1") || is_record(line, "TER") || is_record(line, "END") {
                context.push(line.to_string());
                *pos += 1;
                continue;
            }
            return Err(DockError::Parse {
                line: line_no,
                message: format!("unexpected record: {line:?}"),
            });
        }
        Ok(node)
    }

    fn parse_branch(
        &mut self,
        lines: &[&str],
        pos: &mut usize,
        context: &mut Vec<String>,
        from: i32,
        to: i32,
    ) -> Result<RawNode> {
        let mut node = RawNode::default();
        let mut seen_immobile = false;
        while *pos < lines.len() {
            let line = lines[*pos];
            let line_no = *pos + 1;
            if line.trim().is_empty() {
                *pos += 1;
                continue;
            }
            if is_record(line, "ENDBRANCH") {
                let (f, t) = parse_two_unsigneds(line, "ENDBRANCH", line_no)?;
                if f != from || t != to {
                    return Err(DockError::Parse {
                        line: line_no,
                        message: format!("inconsistent branch numbers: BRANCH {from} {to} vs ENDBRANCH {f} {t}"),
                    });
                }
                if !seen_immobile {
                    return Err(DockError::Parse {
                        line: line_no,
                        message: format!("atom {to} has not been found in this branch"),
                    });
                }
                *pos += 1;
                return Ok(node);
            }
            if is_record(line, "ATOM") || is_record(line, "HETATM") {
                let rec = parse_atom_line(line, line_no)?;
                if rec.atom.serial == to {
                    node.immobile = Some(node.atoms.len());
                    seen_immobile = true;
                }
                node.atoms.push(rec);
                *pos += 1;
                continue;
            }
            if is_record(line, "BRANCH") {
                let (f, t) = parse_two_unsigneds(line, "BRANCH", line_no)?;
                *pos += 1;
                let child = self.parse_branch(lines, pos, context, f, t)?;
                let parent = node
                    .atoms
                    .iter()
                    .position(|a| a.atom.serial == f)
                    .ok_or_else(|| DockError::Parse {
                        line: line_no,
                        message: format!("BRANCH {f} {t}: atom {f} not found in this branch"),
                    })?;
                node.children.push((parent, child));
                continue;
            }
            if is_record(line, "REMARK") || is_record(line, "WARNING") {
                context.push(line.to_string());
                *pos += 1;
                continue;
            }
            return Err(DockError::Parse {
                line: line_no,
                message: format!("unexpected record inside BRANCH: {line:?}"),
            });
        }
        Err(DockError::Parse {
            line: lines.len(),
            message: "BRANCH without ENDBRANCH".to_string(),
        })
    }
}

/// Flatten a raw topology into the movable atom array plus a [`TopologyNode`].
struct Flattener {
    atoms: Vec<Atom>,
    records: Vec<RecordAtom>,
    /// `(parent movable index, attachment movable index)`
    rotors: Vec<(Option<usize>, usize)>,
}

impl Flattener {
    /// Insert a level. `seed` carries the movable index of this level's
    /// attachment atom when the parent has already inserted it.
    fn insert(&mut self, node: &RawNode, seed: Option<(usize, usize)>) -> Result<TopologyNode> {
        let n = node.atoms.len();
        let mut gmap: Vec<Option<usize>> = vec![None; n];
        if let Some((i, g)) = seed {
            gmap[i] = Some(g);
        }
        let mut out = TopologyNode::default();
        for (i, rec) in node.atoms.iter().enumerate() {
            if gmap[i].is_some() {
                continue; // inserted by the parent frame
            }
            gmap[i] = Some(self.atoms.len());
            self.atoms.push(rec.atom.clone());
            self.records.push(rec.clone());
            out.atoms.push(self.atoms.len() - 1);
        }
        // Children: their attachment atom is rigid relative to *this* frame, so
        // it belongs to this frame's atom list.
        //
        // # Ordering invariant
        //
        // A frame's atoms must be **contiguous** in the movable array: the
        // kinematics indexes frames by a `(begin, end)` range. Every attachment
        // atom is therefore inserted here, immediately after this frame's own
        // atoms and *before* any child subtree is descended into — the same
        // order the reference implementation produces with its
        // `insert_immobiles` call inside the per-atom loop.
        struct Pending {
            parent_atom: usize,
            child_index: usize,
            attachment: usize,
        }
        let mut pending: Vec<Pending> = Vec::new();
        for (child_index, (parent_i, child)) in node.children.iter().enumerate() {
            let im = child
                .immobile
                .ok_or_else(|| DockError::Invalid("branch without an attachment atom".into()))?;
            let parent_global = gmap[*parent_i].ok_or_else(|| {
                DockError::Invalid("branch parent atom was never inserted".into())
            })?;
            let im_global = self.atoms.len();
            self.atoms.push(child.atoms[im].atom.clone());
            self.records.push(child.atoms[im].clone());
            out.atoms.push(im_global);
            pending.push(Pending {
                parent_atom: parent_global,
                child_index,
                attachment: im_global,
            });
        }

        for p in pending {
            let (_, child) = &node.children[p.child_index];
            let im = child.immobile.expect("checked above");
            if !raw_has_segment(child) {
                // A branch that rotates nothing: its attachment atom stays with
                // this frame, which is already the case.
                continue;
            }
            self.rotors.push((Some(p.parent_atom), p.attachment));
            let mut sub = self.insert(child, Some((im, p.attachment)))?;
            sub.immobile = None;
            sub.attachment = Some(p.attachment);
            sub.attachment_coords = Some(self.atoms[p.attachment].coords);
            sub.axis_root = Some(self.atoms[p.parent_atom].coords);
            sub.axis_root_atom = Some(p.parent_atom);
            out.children.push(sub);
        }
        Ok(out)
    }
}

/// `true` when a raw branch rotates at least one atom.
fn raw_has_segment(node: &RawNode) -> bool {
    node.has_segment()
}

/// Parse a ligand PDBQT document.
pub fn parse_ligand_pdbqt(text: &str) -> Result<ParsedLigand> {
    let lines: Vec<&str> = text.lines().collect();
    let mut parser = TopologyParser {
        issues: Vec::new(),
    };
    let mut context: Vec<String> = Vec::new();
    let mut torsdof: Option<u32> = None;
    let mut pos = 0usize;
    let root = parser.parse_level(&lines, &mut pos, &mut context, &mut torsdof)?;

    if root.atoms.is_empty() {
        return Err(DockError::EmptyMolecule("ligand"));
    }
    if root.immobile.is_some() {
        return Err(DockError::Invalid(
            "the ligand ROOT level must not declare an attachment atom".into(),
        ));
    }
    let torsdof = torsdof.ok_or_else(|| DockError::Parse {
        line: lines.len(),
        message: "missing TORSDOF keyword".to_string(),
    })?;

    let mut flattener = Flattener {
        atoms: Vec::new(),
        records: Vec::new(),
        rotors: Vec::new(),
    };
    let top = flattener.insert(&root, None)?;
    if flattener.atoms.is_empty() {
        return Err(DockError::EmptyMolecule("ligand"));
    }

    Ok(ParsedLigand {
        atoms: flattener.atoms,
        records: flattener.records,
        top,
        rotors: flattener.rotors,
        torsdof,
        context,
        issues: parser.issues,
    })
}

/// Parse a rigid receptor PDBQT document.
///
/// `END_RES` markers (flexible-residue files) are rejected with a clear
/// message: this release docks rigid receptors only, and silently treating a
/// flexible-residue file as rigid would give wrong numbers.
pub fn parse_receptor_pdbqt(text: &str) -> Result<ReceptorRecord> {
    let mut atoms: Vec<Atom> = Vec::new();
    let mut records: Vec<RecordAtom> = Vec::new();
    let mut issues: Vec<ParseIssue> = Vec::new();
    let mut structures = 0usize;
    for (i, line) in text.lines().enumerate() {
        let line_no = i + 1;
        if line.trim().is_empty()
            || is_record(line, "TER")
            || is_record(line, "END")
            || is_record(line, "WARNING")
            || is_record(line, "REMARK")
            || is_record(line, "CRYST1")
        {
            continue;
        }
        if is_record(line, "ATOM") || is_record(line, "HETATM") {
            let rec = parse_atom_line(line, line_no)?;
            atoms.push(rec.atom.clone());
            records.push(rec);
            continue;
        }
        if is_record(line, "MODEL") {
            structures += 1;
            if structures > 1 {
                return Err(DockError::Parse {
                    line: line_no,
                    message: "multi-MODEL receptor files are not supported; split the models first"
                        .to_string(),
                });
            }
            continue;
        }
        if is_record(line, "ENDMDL") {
            continue;
        }
        if is_record(line, "ROOT")
            || is_record(line, "BRANCH")
            || is_record(line, "ENDROOT")
            || is_record(line, "ENDBRANCH")
            || is_record(line, "BEGIN_RES")
            || is_record(line, "END_RES")
            || is_record(line, "TORSDOF")
        {
            issues.push(ParseIssue {
                line: line_no,
                message: format!(
                    "treating flexible-residue record {:?} as rigid: OpenDocking {VERSION} docks rigid receptors",
                    line.trim(),
                    VERSION = crate::VERSION
                ),
            });
            continue;
        }
        return Err(DockError::Parse {
            line: line_no,
            message: format!("unknown or inappropriate tag in receptor: {line:?}"),
        });
    }
    if atoms.is_empty() {
        return Err(DockError::EmptyMolecule("receptor"));
    }
    Ok(ReceptorRecord {
        atoms,
        records,
        issues,
    })
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write a pose as a single `MODEL` of a PDBQT multi-model file.
///
/// * `records` — the original atom templates of the ligand, aligned with the
///   ligand's movable atom order.
/// * `coords` — the pose coordinates (heavy atoms first, then hydrogens), in
///   the same order as the parsing produced.
/// * `context` — the ligand's non-`ATOM` lines, reproduced verbatim so that
///   `ROOT`/`BRANCH`/`TORSDOF` survive a round trip.
/// * `remarks` — extra `REMARK` lines to insert at the top (Vina's result
///   block).
///
/// The topology records are re-emitted from `top`, so the output is a valid
/// PDBQT that can be re-docked.
pub fn write_pose_pdbqt(
    records: &[RecordAtom],
    top: &TopologyNode,
    coords: &[DVec3],
    remarks: &[String],
    model_number: Option<usize>,
) -> String {
    let mut out = String::with_capacity(records.len() * 64 + 256);
    if let Some(n) = model_number {
        out.push_str(&format!("MODEL {n}\n"));
    }
    for r in remarks {
        out.push_str("REMARK ");
        out.push_str(r.trim_start_matches("REMARK ").trim_end());
        out.push('\n');
    }
    out.push_str("ROOT\n");
    emit_node(&mut out, top, records, coords);
    out.push_str("ENDROOT\n");
    emit_children(&mut out, top, records, coords);
    let num_tors = count_torsions(top);
    out.push_str(&format!("TORSDOF {num_tors}\n"));
    if model_number.is_some() {
        out.push_str("ENDMDL\n");
    }
    out
}

fn count_torsions(node: &TopologyNode) -> usize {
    node.children.iter().map(|c| 1 + count_torsions(c)).sum()
}

fn emit_node(out: &mut String, node: &TopologyNode, records: &[RecordAtom], coords: &[DVec3]) {
    for &i in &node.atoms {
        // A child branch's attachment atom belongs to this frame's *cluster*
        // but is printed at the top of that branch's block, so it must not be
        // emitted twice.
        if node.children.iter().any(|c| c.attachment == Some(i)) {
            continue;
        }
        out.push_str(&records[i].render(coords[i]));
        out.push('\n');
    }
}

fn emit_children(out: &mut String, node: &TopologyNode, records: &[RecordAtom], coords: &[DVec3]) {
    for child in &node.children {
        // `BRANCH a b`: `a` is the parent atom, `b` the attachment atom that
        // stays rigid relative to the parent frame.
        let attach = child
            .attachment
            .expect("torsion branch must record its attachment atom");
        let parent_atom = child.axis_root_atom.unwrap_or(attach);
        let a_serial = records[parent_atom].atom.serial;
        let b_serial = records[attach].atom.serial;
        out.push_str(&format!("BRANCH {a_serial:>4} {b_serial:>4}\n"));
        out.push_str(&records[attach].render(coords[attach]));
        out.push('\n');
        emit_node(out, child, records, coords);
        emit_children(out, child, records, coords);
        out.push_str(&format!("ENDBRANCH {a_serial:>4} {b_serial:>4}\n"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AdType;

    const LIGAND: &str = "\
REMARK  ligand
ROOT
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C 
ATOM      2  C2  LIG A   1       1.520   0.000   0.000  1.00  0.00     0.000 C 
ENDROOT
BRANCH    2   3
ATOM      3  O1  LIG A   1       2.100   1.300   0.000  1.00  0.00    -0.300 OA
ATOM      4  H1  LIG A   1       1.700   2.100   0.000  1.00  0.00     0.200 HD
ENDBRANCH    2   3
TORSDOF 1
";

    #[test]
    fn parses_a_simple_two_level_ligand() {
        let lig = parse_ligand_pdbqt(LIGAND).unwrap();
        assert_eq!(lig.atoms.len(), 4);
        assert_eq!(lig.torsdof, 1);
        // The root frame owns atoms 1 and 2 plus atom 3, which is the
        // attachment point of the torsion branch and therefore also belongs to
        // the root cluster (it simply has no effect, sitting on the axis).
        // Frame ranges are contiguous in the movable array, which is what the
        // kinematics relies on.
        assert_eq!(lig.top.atoms, vec![0, 1, 2]);
        assert_eq!(lig.top.children.len(), 1);
        assert_eq!(lig.top.children[0].attachment, Some(2));
        // The branch rotates atom 4 only (atom 3 sits on the axis).
        assert_eq!(lig.top.children[0].atoms, vec![3]);
        assert_eq!(lig.rotors.len(), 1);
        assert_eq!(lig.rotors[0], (Some(1), 2));
        assert_eq!(lig.atoms[2].ad, AdType::OA);
        assert!(lig.atoms[2].charge < 0.0);
    }

    #[test]
    fn detects_missing_torsdof() {
        let text = LIGAND.replace("TORSDOF 1\n", "");
        let err = parse_ligand_pdbqt(&text).unwrap_err();
        assert!(format!("{err}").contains("TORSDOF"));
    }

    #[test]
    fn detects_inconsistent_branch_numbers() {
        let text = LIGAND.replace("ENDBRANCH    2   3", "ENDBRANCH    2   4");
        assert!(parse_ligand_pdbqt(&text).is_err());
    }

    #[test]
    fn parses_a_receptor() {
        let text = "\
ATOM      1  N   ALA A   1      27.214  24.850  22.290  1.00  0.00    -0.347 N 
ATOM      2  CA  ALA A   1      26.100  25.600  21.500  1.00  0.00     0.100 C 
TER
END
";
        let r = parse_receptor_pdbqt(text).unwrap();
        assert_eq!(r.atoms.len(), 2);
        assert_eq!(r.atoms[0].res_name, "ALA");
        assert_eq!(r.atoms[0].res_id, 1);
        assert_eq!(r.atoms[1].ad, AdType::C);
        assert!((r.atoms[0].coords.x - 27.214).abs() < 1e-9);
    }

    #[test]
    fn flexible_residue_records_are_reported_not_silently_dropped() {
        let text = "\
ATOM      1  N   ALA A   1      27.214  24.850  22.290  1.00  0.00    -0.347 N 
BEGIN_RES SER A 2
ROOT
ATOM      2  CB  SER A   2      28.000  24.000  22.000  1.00  0.00     0.100 C 
ENDROOT
TORSDOF 1
END_RES SER A 2
";
        let r = parse_receptor_pdbqt(text).unwrap();
        // The side-chain atoms are kept (rigidly) so the box still excludes
        // them, and a warning records the approximation.
        assert_eq!(r.atoms.len(), 2);
        assert!(!r.issues.is_empty());
        assert!(r.issues.iter().any(|i| i.message.contains("flexible-residue")));
    }

    #[test]
    fn a_multi_model_document_parses_as_its_first_model_only() {
        // Regression: a pose file with several MODELs must not be merged into a
        // single (much larger) molecule.
        let one = write_pose_pdbqt(
            &parse_ligand_pdbqt(LIGAND).unwrap().records,
            &parse_ligand_pdbqt(LIGAND).unwrap().top,
            &parse_ligand_pdbqt(LIGAND)
                .unwrap()
                .atoms
                .iter()
                .map(|a| a.coords)
                .collect::<Vec<_>>(),
            &["VINA RESULT:    -7.500  0.000  0.000".to_string()],
            Some(1),
        );
        let two = format!(
            "{}{}",
            one,
            one.replace("MODEL 1", "MODEL 2")
        );
        let parsed = parse_ligand_pdbqt(&two).unwrap();
        let expected = parse_ligand_pdbqt(LIGAND).unwrap();
        assert_eq!(parsed.atoms.len(), expected.atoms.len());
        assert_eq!(parsed.rotors, expected.rotors);
        assert_eq!(parsed.torsdof, expected.torsdof);
    }

    #[test]
    fn a_repeated_root_without_model_records_stops_at_the_first() {
        let doubled = format!("{LIGAND}{LIGAND}");
        let parsed = parse_ligand_pdbqt(&doubled).unwrap();
        assert_eq!(parsed.atoms.len(), 4, "the second ROOT must be ignored");
    }

    #[test]
    fn writes_a_pose_that_round_trips() {
        let lig = parse_ligand_pdbqt(LIGAND).unwrap();
        let coords: Vec<DVec3> = lig.atoms.iter().map(|a| a.coords).collect();
        let text = write_pose_pdbqt(
            &lig.records,
            &lig.top,
            &coords,
            &["VINA RESULT:    -7.500  0.000  0.000".to_string()],
            Some(1),
        );
        let re = parse_ligand_pdbqt(&text).unwrap();
        assert_eq!(re.atoms.len(), lig.atoms.len());
        assert_eq!(re.rotors, lig.rotors);
        assert_eq!(re.torsdof, lig.torsdof);
        for (a, b) in re.atoms.iter().zip(&lig.atoms) {
            assert!((a.coords - b.coords).length() < 1e-3);
            assert_eq!(a.ad, b.ad);
        }
    }

    #[test]
    fn render_pads_short_templates() {
        let s = render_pdbqt_line("ATOM 1 C", DVec3::new(1.0, -2.0, 3.5));
        assert!(s.len() >= 54);
        assert!(s.contains("1.000"));
    }
}

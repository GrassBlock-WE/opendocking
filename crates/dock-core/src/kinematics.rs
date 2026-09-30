// SPDX-License-Identifier: GPL-3.0-or-later
//! Rigid clusters, the rotatable-bond tree, and forward kinematics.
//!
//! # The model
//!
//! A docked ligand is a **rigid body** (3 translation + 3 rotation degrees of
//! freedom) carrying an arbitrary number of **torsion segments** arranged in a
//! tree. Every torsion segment owns a contiguous block of atoms and rotates
//! them about an axis defined by two atoms:
//!
//! ```text
//! axis       = normalize(coords[attachment] - coords[parent])
//! origin     = coords[attachment]
//! orientation = angle_to_quaternion(axis, torsion) * parent_orientation
//! coords[i]  = origin + orientation * local[i]        for i in this segment
//! ```
//!
//! This is precisely AutoDock Vina's `tree.h` / `conf.h` formulation
//! (Apache-2.0, The Scripps Research Institute), re-expressed with `glam`
//! quaternions. Flexible receptor side chains reuse the same machinery with an
//! *absolute* root segment whose origin and axis are pinned to the (immobile)
//! protein frame.
//!
//! # Why a tree, not a graph
//!
//! A rigid body plus a spanning tree of rotatable bonds spans exactly the same
//! set of conformations as the full molecular graph, but each conformation can
//! be generated in a single `O(n_atoms)` forward pass with no constraint
//! solving. Ring systems are handled by treating the ring as part of a single
//! rigid cluster (the spanning tree simply never crosses a ring), which is the
//! same approximation Vina and AutoDock 4 make.
//!
//! # Gradients
//!
//! The Cartesian energy gradient `鈭侲/鈭倄_i` is projected onto the tree's degrees
//! of freedom by the standard rigid-body chain rule:
//!
//! ```text
//! 鈭侲/鈭倀ranslation = 危_i 鈭侲/鈭倄_i
//! 鈭侲/鈭傁?          = 危_i (x_i - origin) 脳 鈭侲/鈭倄_i
//! 鈭侲/鈭倀orsion     = axis 路 危_i (x_i - origin) 脳 鈭侲/鈭倄_i
//! ```
//!
//! where `蠅` is the rotation-vector (exponential-map) increment used by the
//! BFGS step and `origin` is the rotation centre of the corresponding frame.

use crate::math::{angle_to_quaternion, normalize_angle, quaternion_increment, DQuat, DVec3, EPSILON};
use crate::molecule::{Atom, Pair};
use crate::rng::Rng;

/// A local coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// Frame origin in the laboratory frame.
    pub origin: DVec3,
    /// Frame orientation (unit quaternion).
    pub orientation: DQuat,
}

impl Frame {
    /// Express a point of the local frame in the laboratory frame.
    #[inline]
    pub fn local_to_lab(&self, v: DVec3) -> DVec3 {
        self.origin + self.orientation * v
    }

    /// Express a direction of the local frame in the laboratory frame.
    #[inline]
    pub fn local_to_lab_direction(&self, v: DVec3) -> DVec3 {
        self.orientation * v
    }
}

/// Which kind of degree of freedom a node carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// 6 DOF: translation + rotation of a whole rigid cluster.
    Rigid,
    /// 1 DOF: rotation about the node's axis.
    Torsion,
}

/// One node of the kinematic tree.
#[derive(Debug, Clone)]
pub struct Node {
    /// Rigid cluster or torsion segment.
    pub kind: FrameKind,
    /// Current laboratory-frame origin (rotation centre).
    pub origin: DVec3,
    /// Current laboratory-frame orientation.
    pub orientation: DQuat,
    /// Current laboratory-frame unit rotation axis.
    pub axis: DVec3,
    /// Origin expressed in the parent frame (torsion nodes).
    pub rel_origin: DVec3,
    /// Axis expressed in the parent frame (torsion nodes).
    pub rel_axis: DVec3,
    /// `true` when origin/axis are pinned in the laboratory frame
    /// (a flexible-residue root segment).
    pub absolute: bool,
    /// Half-open range of movable atom indices owned by this frame.
    pub atoms: (usize, usize),
    /// Index of the driving torsion, `None` for a rigid root.
    pub torsion: Option<usize>,
    /// Child segments.
    pub children: Vec<Node>,
}

impl Node {
    /// A rigid root with the given atom range.
    pub fn rigid_root(atoms: (usize, usize)) -> Node {
        Node {
            kind: FrameKind::Rigid,
            origin: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            axis: DVec3::Z,
            rel_origin: DVec3::ZERO,
            rel_axis: DVec3::Z,
            absolute: false,
            atoms,
            torsion: None,
            children: Vec::new(),
        }
    }

    /// Write this frame's atoms into the laboratory coordinate array.
    #[inline]
    fn write_coords(&self, local: &[DVec3], coords: &mut [DVec3]) {
        for i in self.atoms.0..self.atoms.1 {
            coords[i] = self.origin + self.orientation * local[i];
        }
    }

    /// Recursively apply a torsion value to this segment and its children.
    fn apply_segment(
        &mut self,
        parent: &Frame,
        torsions: &[f64],
        cursor: &mut usize,
        local: &[DVec3],
        coords: &mut [DVec3],
    ) {
        let t = torsions.get(*cursor).copied().unwrap_or(0.0);
        *cursor += 1;
        if !self.absolute {
            self.origin = parent.local_to_lab(self.rel_origin);
            self.axis = parent.local_to_lab_direction(self.rel_axis);
            let q = angle_to_quaternion(self.axis, t) * parent.orientation;
            let n = q.length();
            self.orientation = if n > 0.0 {
                DQuat::from_xyzw(q.x / n, q.y / n, q.z / n, q.w / n)
            } else {
                DQuat::IDENTITY
            };
        } else {
            self.orientation = angle_to_quaternion(self.axis, t);
        }
        self.write_coords(local, coords);
        let frame = Frame {
            origin: self.origin,
            orientation: self.orientation,
        };
        for child in &mut self.children {
            child.apply_segment(&frame, torsions, cursor, local, coords);
        }
    }

    /// Recursively accumulate the degrees-of-freedom gradient.
    ///
    /// Returns `(危 force, 危 torque about this node's origin)`.
    fn accumulate_gradient(
        &self,
        coords: &[DVec3],
        forces: &[DVec3],
        g: &mut [f64],
        rigid_slots: Option<(usize, usize)>,
        torsion_base: usize,
    ) -> (DVec3, DVec3) {
        let mut force = DVec3::ZERO;
        let mut torque = DVec3::ZERO;
        for i in self.atoms.0..self.atoms.1 {
            let f = forces[i];
            force += f;
            torque += (coords[i] - self.origin).cross(f);
        }
        for child in &self.children {
            let (cf, ct) = child.accumulate_gradient(coords, forces, g, None, torsion_base);
            force += cf;
            torque += (child.origin - self.origin).cross(cf) + ct;
        }
        match rigid_slots {
            Some((p, o)) => {
                g[p] = force.x;
                g[p + 1] = force.y;
                g[p + 2] = force.z;
                g[o] = torque.x;
                g[o + 1] = torque.y;
                g[o + 2] = torque.z;
            }
            None => {
                if let Some(t) = self.torsion {
                    g[torsion_base + t] = torque.dot(self.axis);
                }
            }
        }
        (force, torque)
    }
}

/// Whether a [`KinematicTree`] describes a ligand (free rigid body) or a
/// flexible receptor side chain (pinned root).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeKind {
    /// A ligand: the root is a [`FrameKind::Rigid`] node.
    Ligand,
    /// A flexible residue: the root is an absolute torsion segment.
    Flex,
}

/// A rigid body plus a tree of torsion segments.
#[derive(Debug, Clone)]
pub struct KinematicTree {
    /// Root node (rigid body, or the pinned first segment of a flex residue).
    pub root: Node,
    /// Ligand or flexible residue.
    pub kind: TreeKind,
    /// Number of torsion degrees of freedom.
    pub num_torsions: usize,
    /// Reference (frame-local) coordinates, indexed by movable atom index.
    pub local: Vec<DVec3>,
    /// Half-open range of movable atoms spanned by this tree.
    pub atoms: (usize, usize),
}

impl KinematicTree {
    /// Total number of atoms spanned by the tree.
    #[inline]
    pub fn num_atoms(&self) -> usize {
        self.atoms.1 - self.atoms.0
    }

    /// Forward kinematics for a ligand: apply `conf` and refresh `coords`.
    pub fn apply_ligand(
        &mut self,
        conf: &LigandConf,
        coords: &mut [DVec3],
    ) {
        debug_assert_eq!(self.kind, TreeKind::Ligand);
        self.root.origin = conf.position;
        self.root.orientation = conf.orientation;
        self.root.write_coords(&self.local, coords);
        let frame = Frame {
            origin: self.root.origin,
            orientation: self.root.orientation,
        };
        let mut cursor = 0usize;
        for child in &mut self.root.children {
            child.apply_segment(&frame, &conf.torsions, &mut cursor, &self.local, coords);
        }
        debug_assert_eq!(cursor, self.num_torsions);
    }

    /// Forward kinematics for a flexible residue.
    pub fn apply_flex(&mut self, torsions: &[f64], coords: &mut [DVec3]) {
        debug_assert_eq!(self.kind, TreeKind::Flex);
        let mut cursor = 0usize;
        self.root
            .apply_segment_root_absolute(torsions, &mut cursor, &self.local, coords);
        debug_assert_eq!(cursor, self.num_torsions);
    }

    /// Accumulate this tree's DOF gradient.
    ///
    /// `g` is the flat degree-of-freedom gradient buffer. For a ligand,
    /// `rigid_slots` gives the offsets of the translation and rotation blocks.
    /// For a flex residue the root's torsion slot has already been accounted
    /// for by `torsion_base`.
    pub fn accumulate_gradient(
        &self,
        coords: &[DVec3],
        forces: &[DVec3],
        g: &mut [f64],
        rigid_slots: Option<(usize, usize)>,
        torsion_base: usize,
    ) {
        match self.kind {
            TreeKind::Ligand => {
                let _ =
                    self.root
                        .accumulate_gradient(coords, forces, g, rigid_slots, torsion_base);
            }
            TreeKind::Flex => {
                // The pinned root segment contributes only to its own torsion.
                let mut force = DVec3::ZERO;
                let mut torque = DVec3::ZERO;
                for i in self.root.atoms.0..self.root.atoms.1 {
                    let f = forces[i];
                    force += f;
                    torque += (coords[i] - self.root.origin).cross(f);
                }
                for child in &self.root.children {
                    let (cf, ct) =
                        child.accumulate_gradient(coords, forces, g, None, torsion_base);
                    force += cf;
                    torque += (child.origin - self.root.origin).cross(cf) + ct;
                }
                if let Some(t) = self.root.torsion {
                    g[torsion_base + t] = torque.dot(self.root.axis);
                }
            }
        }
    }
}

impl Node {
    /// Apply a pinned (absolute) root segment, i.e. a flexible-residue root.
    fn apply_segment_root_absolute(
        &mut self,
        torsions: &[f64],
        cursor: &mut usize,
        local: &[DVec3],
        coords: &mut [DVec3],
    ) {
        let t = torsions.get(*cursor).copied().unwrap_or(0.0);
        *cursor += 1;
        self.orientation = angle_to_quaternion(self.axis, t);
        self.write_coords(local, coords);
        let frame = Frame {
            origin: self.origin,
            orientation: self.orientation,
        };
        for child in &mut self.children {
            child.apply_segment(&frame, torsions, cursor, local, coords);
        }
    }
}

/// Input topology needed to build a [`KinematicTree`].
///
/// Produced by the PDBQT reader; atom entries are *global movable atom indices*.
/// The parent atom of a branch may be immobile (a receptor atom), so its
/// position is carried as a coordinate rather than as an index.
#[derive(Debug, Clone, Default)]
pub struct TopologyNode {
    /// Atoms owned by this frame, in file order.
    pub atoms: Vec<usize>,
    /// Position within [`TopologyNode::atoms`] of the atom that remains fixed
    /// relative to the parent frame (the `BRANCH a b` "b" atom). Only set by
    /// the raw PDBQT parser; a flattened topology records `attachment` instead.
    pub immobile: Option<usize>,
    /// Global movable index of this branch attachment atom. The atom itself
    /// belongs to the *parent* frame: it lies on the rotation axis and therefore
    /// never moves, which is exactly AutoDock Vina mobility convention.
    pub attachment: Option<usize>,
    /// Laboratory coordinates of the attachment atom; set even when it is an
    /// immobile receptor atom, as in a flexible-residue root.
    pub attachment_coords: Option<DVec3>,
    /// Laboratory position of the parent atom (`BRANCH a b` "a" atom).
    pub axis_root: Option<DVec3>,
    /// Global movable index of the parent atom, when it is movable.
    pub axis_root_atom: Option<usize>,
    /// Child torsion branches.
    pub children: Vec<TopologyNode>,
}

impl TopologyNode {
    /// `true` when this branch actually rotates atoms, i.e. it owns at least one
    /// atom beyond the attachment point or has a descendant that does.
    pub fn has_segment(&self) -> bool {
        !self.atoms.is_empty() || self.children.iter().any(|c| c.has_segment())
    }

    /// Vina's `essentially_empty`: a branch that carries no atoms of its own
    /// beyond the attachment atom and has no non-empty sub-branches.
    pub fn is_empty_branch(&self) -> bool {
        !self.has_segment()
    }
}

/// Shared scratch used while building trees.
struct BuildCtx<'a> {
    coords: &'a [DVec3],
    /// Flat atom index assignment, in the order Vina's parser would produce.
    order: &'a mut Vec<usize>,
}

impl BuildCtx<'_> {
    /// Build one frame.
    ///
    /// `own_immobile` is `true` only for a flexible-residue root segment: its
    /// attachment atom is itself a movable atom (it sits at the rotation centre
    /// and therefore never moves) rather than belonging to a parent cluster.
    fn build(
        &mut self,
        top: &TopologyNode,
        frame_origin: DVec3,
        local: &mut [DVec3],
        own_immobile: bool,
    ) -> (usize, usize, Vec<Node>, usize) {
        // The frame's atom range is expressed in *global movable atom indices*,
        // because that is what `write_coords` indexes. The parser guarantees
        // that a frame's atoms are contiguous, which the debug assertion below
        // re-checks: computing the range from a traversal counter instead — as
        // an earlier revision did — silently mis-assigns frames whenever the
        // traversal order differs from the array order, which is exactly what
        // happens for a ligand with nested torsion branches.
        let mut begin = usize::MAX;
        let mut end = 0usize;
        let mut pushed = 0usize;

        // 1. This frame's own atoms (the parent owns the attachment atom,
        //    unless this frame is the pinned root of a flexible residue).
        for (i, &idx) in top.atoms.iter().enumerate() {
            if Some(i) == top.immobile && !own_immobile {
                continue;
            }
            local[idx] = self.coords[idx] - frame_origin;
            self.order.push(idx);
            begin = begin.min(idx);
            end = end.max(idx + 1);
            pushed += 1;
        }
        // 2. The attachment atoms of the child branches are rigid relative to
        //    *this* frame, so they belong to this frame's cluster. The PDBQT
        //    reader already lists them among `atoms`; this branch only covers
        //    callers that build a topology by hand (and the flexible-residue
        //    path, where the attachment is an immobile receptor atom).
        for c in &top.children {
            let Some(idx) = c.attachment else { continue };
            if top.atoms.contains(&idx) {
                continue;
            }
            local[idx] = self.coords[idx] - frame_origin;
            self.order.push(idx);
            begin = begin.min(idx);
            end = end.max(idx + 1);
            pushed += 1;
        }
        if pushed == 0 {
            begin = 0;
            end = 0;
        }
        debug_assert_eq!(
            end.saturating_sub(begin),
            pushed,
            "a frame's atoms must be contiguous in the movable atom array"
        );

        // 3. Build the child segments.
        let mut children = Vec::new();
        let mut n_tors = 0usize;
        for c in &top.children {
            if !c.has_segment() {
                continue;
            }
            let im = c.attachment.expect("branch must record its attachment atom");
            let seg_origin = self.coords[im];
            let axis_root = c.axis_root.expect("branch must record its parent atom");
            let axis_raw = seg_origin - axis_root;
            let n = axis_raw.length();
            let axis = if n > EPSILON { axis_raw / n } else { DVec3::Z };
            let (cb, ce, cc, cn) = self.build(c, seg_origin, local, false);
            children.push(Node {
                kind: FrameKind::Torsion,
                origin: seg_origin,
                orientation: DQuat::IDENTITY,
                axis,
                rel_origin: seg_origin - frame_origin,
                rel_axis: axis,
                absolute: false,
                atoms: (cb, ce),
                torsion: None,
                children: cc,
            });
            n_tors += 1 + cn;
        }
        (begin, end, children, n_tors)
    }
}

/// Build a ligand kinematic tree from its parsed topology.
///
/// * `coords` 鈥?laboratory coordinates of the movable atoms in *file* order of
///   the flat movable array; only the entries referenced by `top` are read.
/// * `n_movable` 鈥?length of the movable atom array.
///
/// Returns the tree and the atom index of the root (`coords[root]` is the rigid
/// body's initial position).
pub fn build_ligand_tree(
    top: &TopologyNode,
    coords: &[DVec3],
    n_movable: usize,
) -> (KinematicTree, usize) {
    assert!(!top.atoms.is_empty(), "ligand topology must have a root atom");
    let root_origin = coords[top.atoms[0]];
    let mut order: Vec<usize> = Vec::with_capacity(n_movable);
    let mut local = vec![DVec3::ZERO; n_movable];
    let (begin, end, children, n_tors) = {
        let mut ctx = BuildCtx {
            coords,
            order: &mut order,
        };
        ctx.build(top, root_origin, &mut local, false)
    };
    // Sanity: the topology must reference every movable atom exactly once.
    debug_assert_eq!(order.len(), n_movable, "topology/atom count mismatch");
    debug_assert_eq!(local.len(), n_movable);
    let mut root = Node::rigid_root((begin, end));
    root.origin = root_origin;
    root.orientation = DQuat::IDENTITY;
    root.children = children;
    let atoms = tree_atom_range(&root, n_movable);
    let mut tree = KinematicTree {
        root,
        kind: TreeKind::Ligand,
        num_torsions: n_tors,
        local,
        atoms,
    };
    assign_torsions(&mut tree.root, &mut 0usize);
    (tree, top.atoms[0])
}

/// Build a flexible-residue (or flexible-side-chain) kinematic tree.
///
/// `top` describes the first segment: its `immobile` atom is the side-chain
/// attachment (a movable atom that sits on the rotation axis and therefore
/// never moves) and its `axis_root` is the (immobile) parent atom. The root
/// segment is pinned in the laboratory frame.
pub fn build_flex_tree(top: &TopologyNode, coords: &[DVec3], n_movable: usize) -> KinematicTree {
    let root_origin = top
        .attachment_coords
        .expect("flexible residue branch must record its attachment coordinates");
    let axis_root = top
        .axis_root
        .expect("flexible residue must record its parent atom");
    let axis_raw = root_origin - axis_root;
    let n = axis_raw.length();
    let axis = if n > EPSILON { axis_raw / n } else { DVec3::Z };

    let mut order: Vec<usize> = Vec::with_capacity(n_movable);
    let mut local = vec![DVec3::ZERO; n_movable];
    let (begin, end, children, n_tors) = {
        let mut ctx = BuildCtx {
            coords,
            order: &mut order,
        };
        ctx.build(top, root_origin, &mut local, true)
    };
    debug_assert_eq!(order.len(), n_movable, "topology/atom count mismatch");
    let root = Node {
        kind: FrameKind::Torsion,
        origin: root_origin,
        orientation: DQuat::IDENTITY,
        axis,
        rel_origin: DVec3::ZERO,
        rel_axis: axis,
        absolute: true,
        atoms: (begin, end),
        torsion: None,
        children,
    };
    let mut tree = KinematicTree {
        root,
        kind: TreeKind::Flex,
        num_torsions: n_tors + 1,
        local,
        atoms: (0, n_movable),
    };
    assign_torsions(&mut tree.root, &mut 0usize);
    tree
}

/// Union atom range over all nodes of a tree.
fn tree_atom_range(root: &Node, n_movable: usize) -> (usize, usize) {
    fn walk(n: &Node, lo: &mut usize, hi: &mut usize) {
        *lo = (*lo).min(n.atoms.0);
        *hi = (*hi).max(n.atoms.1);
        for c in &n.children {
            walk(c, lo, hi);
        }
    }
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    walk(root, &mut lo, &mut hi);
    if lo == usize::MAX {
        (0, n_movable)
    } else {
        (lo, hi)
    }
}

/// Assign pre-order torsion indices, matching Vina's `flv::iterator` traversal.
fn assign_torsions(node: &mut Node, next: &mut usize) {
    if node.kind == FrameKind::Torsion {
        node.torsion = Some(*next);
        *next += 1;
    }
    for c in &mut node.children {
        assign_torsions(c, next);
    }
}

// ---------------------------------------------------------------------------
// Conformations
// ---------------------------------------------------------------------------

/// A ligand conformation.
#[derive(Debug, Clone, PartialEq)]
pub struct LigandConf {
    /// Position of the rigid body's root atom.
    pub position: DVec3,
    /// Orientation of the rigid body.
    pub orientation: DQuat,
    /// Torsion angles (radians).
    pub torsions: Vec<f64>,
}

impl LigandConf {
    /// The "identity" conformation: current position, identity rotation, all
    /// torsions at zero.
    pub fn null(position: DVec3, num_torsions: usize) -> LigandConf {
        LigandConf {
            position,
            orientation: DQuat::IDENTITY,
            torsions: vec![0.0; num_torsions],
        }
    }
}

/// A ligand degree-of-freedom *change* (translation, rotation vector, torsions).
#[derive(Debug, Clone, PartialEq)]
pub struct LigandChange {
    /// Translation increment (脜).
    pub position: DVec3,
    /// Rotation-vector increment (radians).
    pub orientation: DVec3,
    /// Torsion increments (radians).
    pub torsions: Vec<f64>,
}

impl LigandChange {
    /// Zero change with room for `num_torsions` torsions.
    pub fn zeros(num_torsions: usize) -> LigandChange {
        LigandChange {
            position: DVec3::ZERO,
            orientation: DVec3::ZERO,
            torsions: vec![0.0; num_torsions],
        }
    }
}

/// The full conformational state of a docking system.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Conf {
    /// One entry per ligand (OpenDocking currently docks one ligand per system).
    pub ligands: Vec<LigandConf>,
    /// Torsion vectors of the flexible receptor residues.
    pub flex: Vec<Vec<f64>>,
}

impl Conf {
    /// All-zero torsions, identity rotation, positions taken from `positions`.
    pub fn null(positions: &[DVec3], layout: &DofLayout) -> Conf {
        Conf {
            ligands: positions
                .iter()
                .zip(&layout.ligands)
                .map(|(p, &n)| LigandConf::null(*p, n))
                .collect(),
            flex: layout.flex.iter().map(|&n| vec![0.0; n]).collect(),
        }
    }

    /// Apply a flat degree-of-freedom delta (`delta`) scaled by `factor`.
    pub fn increment_flat(&mut self, layout: &DofLayout, delta: &[f64], factor: f64) {
        let mut off = 0usize;
        for (li, lig) in self.ligands.iter_mut().enumerate() {
            lig.position += factor * DVec3::new(delta[off], delta[off + 1], delta[off + 2]);
            off += 3;
            let rot = DVec3::new(delta[off], delta[off + 1], delta[off + 2]) * factor;
            off += 3;
            lig.orientation = quaternion_increment(lig.orientation, rot);
            for t in lig.torsions.iter_mut() {
                *t += normalize_angle(factor * delta[off]);
                *t = normalize_angle(*t);
                off += 1;
            }
            let _ = li;
        }
        for flex in self.flex.iter_mut() {
            for t in flex.iter_mut() {
                *t += normalize_angle(factor * delta[off]);
                *t = normalize_angle(*t);
                off += 1;
            }
        }
        debug_assert_eq!(off, layout.num_floats());
    }

    /// Randomise every degree of freedom inside the search box.
    pub fn randomize(&mut self, corner1: DVec3, corner2: DVec3, rng: &mut Rng) {
        for lig in self.ligands.iter_mut() {
            lig.position = rng.in_box(corner1, corner2);
            lig.orientation = rng.orientation();
            for t in lig.torsions.iter_mut() {
                *t = rng.angle();
            }
        }
        for flex in self.flex.iter_mut() {
            for t in flex.iter_mut() {
                *t = rng.angle();
            }
        }
    }

    /// Reset torsions and rotation to the identity, keeping positions.
    pub fn set_to_null(&mut self) {
        for lig in self.ligands.iter_mut() {
            lig.orientation = DQuat::IDENTITY;
            for t in lig.torsions.iter_mut() {
                *t = 0.0;
            }
        }
        for flex in self.flex.iter_mut() {
            for t in flex.iter_mut() {
                *t = 0.0;
            }
        }
    }

    /// True when two conformations share their torsions within `cutoff` and
    /// their rigid bodies within the position/orientation cut-offs.
    pub fn too_close(&self, other: &Conf, torsion_cutoff: f64, position_cutoff: f64, orientation_cutoff: f64) -> bool {
        if self.ligands.len() != other.ligands.len() {
            return false;
        }
        for (a, b) in self.ligands.iter().zip(&other.ligands) {
            if a.torsions.len() != b.torsions.len() {
                return false;
            }
            for (x, y) in a.torsions.iter().zip(&b.torsions) {
                if normalize_angle(x - y).abs() > torsion_cutoff {
                    return false;
                }
            }
            if (a.position - b.position).length_squared() > position_cutoff * position_cutoff {
                return false;
            }
            if crate::math::quaternion_difference(a.orientation, b.orientation).length_squared()
                > orientation_cutoff * orientation_cutoff
            {
                return false;
            }
        }
        for (a, b) in self.flex.iter().zip(&other.flex) {
            for (x, y) in a.iter().zip(b) {
                if normalize_angle(x - y).abs() > torsion_cutoff {
                    return false;
                }
            }
        }
        true
    }
}

/// The degree-of-freedom layout of a docking system.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DofLayout {
    /// Number of torsions per ligand.
    pub ligands: Vec<usize>,
    /// Number of torsions per flexible residue.
    pub flex: Vec<usize>,
}

impl DofLayout {
    /// Total number of floating-point degrees of freedom.
    pub fn num_floats(&self) -> usize {
        let lig: usize = self.ligands.iter().map(|n| 6 + n).sum();
        let flex: usize = self.flex.iter().sum();
        lig + flex
    }

    /// Offset of the ligand torsion block.
    pub fn ligand_torsion_offset(&self, ligand: usize) -> usize {
        self.ligands[..ligand].iter().map(|n| 6 + n).sum::<usize>() + 6
    }

    /// Offset of the first ligand's translation block.
    pub fn ligand_position_offset(&self, ligand: usize) -> usize {
        self.ligands[..ligand].iter().map(|n| 6 + n).sum::<usize>()
    }

    /// Total number of torsion degrees of freedom (excluding the rigid body).
    pub fn num_torsions_total(&self) -> usize {
        self.ligands.iter().sum::<usize>() + self.flex.iter().sum::<usize>()
    }

    /// Offset of the flexible-residue torsion block.
    pub fn flex_torsion_offset(&self, flex: usize) -> usize {
        let lig: usize = self.ligands.iter().map(|n| 6 + n).sum();
        lig + self.flex[..flex].iter().sum::<usize>()
    }
}

/// The movable part of a docking system: ligand plus flexible residues.
#[derive(Debug, Clone)]
pub struct MovableModel {
    /// Movable atoms (ligand first, then the flexible residues).
    pub atoms: Vec<Atom>,
    /// Current laboratory coordinates.
    pub coords: Vec<DVec3>,
    /// Cartesian energy gradient `鈭侲/鈭倄`, refreshed by
    /// [`MovableModel::accumulate_gradient`].
    pub minus_forces: Vec<DVec3>,
    /// The ligand's kinematic tree.
    pub ligand: KinematicTree,
    /// Intra-ligand interaction pairs.
    pub ligand_pairs: Vec<Pair>,
    /// Flexible receptor residues.
    pub flex: Vec<KinematicTree>,
    /// Intra-flex interaction pairs.
    pub flex_pairs: Vec<Pair>,
    /// Degree-of-freedom layout.
    pub layout: DofLayout,
}

impl MovableModel {
    /// Number of movables atoms.
    #[inline]
    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// `true` when `i` belongs to the ligand.
    #[inline]
    pub fn is_atom_in_ligand(&self, i: usize) -> bool {
        i >= self.ligand.atoms.0 && i < self.ligand.atoms.1
    }

    /// Ligand index owning atom `i`, if any.
    #[inline]
    pub fn find_ligand(&self, i: usize) -> Option<usize> {
        if self.is_atom_in_ligand(i) {
            Some(0)
        } else {
            None
        }
    }

    /// Apply a conformation, i.e. run the whole forward kinematics.
    pub fn apply(&mut self, conf: &Conf) {
        if let Some(lig) = conf.ligands.first() {
            // Temporarily take the tree out so that `coords` can be borrowed
            // mutably at the same time.
            let mut tree = std::mem::replace(&mut self.ligand, dummy_tree());
            tree.apply_ligand(lig, &mut self.coords);
            self.ligand = tree;
        }
        for (k, tree_slot) in self.flex.iter_mut().enumerate() {
            if let Some(t) = conf.flex.get(k) {
                let mut tree = std::mem::replace(tree_slot, dummy_tree());
                tree.apply_flex(t, &mut self.coords);
                *tree_slot = tree;
            }
        }
    }

    /// The identity conformation for the *current* coordinates.
    pub fn initial_conf(&self) -> Conf {
        Conf {
            ligands: self
                .layout
                .ligands
                .iter()
                .map(|&n| LigandConf::null(self.ligand.root.origin, n))
                .collect(),
            flex: self.layout.flex.iter().map(|&n| vec![0.0; n]).collect(),
        }
    }

    /// Project the Cartesian gradient onto the degrees of freedom.
    ///
    /// Returns the flat DOF gradient (length [`DofLayout::num_floats`]).
    pub fn dof_gradient(&self) -> Vec<f64> {
        let mut g = vec![0.0f64; self.layout.num_floats()];
        if !self.layout.ligands.is_empty() {
            let p = self.layout.ligand_position_offset(0);
            let o = p + 3;
            let tb = self.layout.ligand_torsion_offset(0);
            self.ligand.accumulate_gradient(
                &self.coords,
                &self.minus_forces,
                &mut g,
                Some((p, o)),
                tb,
            );
        }
        for (k, tree) in self.flex.iter().enumerate() {
            let tb = self.layout.flex_torsion_offset(k);
            tree.accumulate_gradient(&self.coords, &self.minus_forces, &mut g, None, tb);
        }
        g
    }

    /// Gyration radius of the ligand (used to scale rotational mutations).
    pub fn ligand_gyration_radius(&self) -> f64 {
        let range = self.ligand.atoms;
        crate::molecule::gyration_radius(&self.atoms, &(range.0..range.1).collect::<Vec<_>>())
    }

    /// Heavy-atom coordinates of the movable atoms (used for RMSD).
    pub fn heavy_atom_coords(&self) -> Vec<DVec3> {
        self.atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| !a.is_hydrogen())
            .map(|(i, _)| self.coords[i])
            .collect()
    }

    /// Clone of the ligand atoms with their current coordinates.
    pub fn ligand_atoms(&self) -> Vec<Atom> {
        let mut out = Vec::with_capacity(self.ligand.atoms.1 - self.ligand.atoms.0);
        for i in self.ligand.atoms.0..self.ligand.atoms.1 {
            let mut a = self.atoms[i].clone();
            a.coords = self.coords[i];
            out.push(a);
        }
        out
    }
}

/// Placeholder used by [`std::mem::replace`] during forward kinematics.
fn dummy_tree() -> KinematicTree {
    KinematicTree {
        root: Node::rigid_root((0, 0)),
        kind: TreeKind::Ligand,
        num_torsions: 0,
        local: Vec::new(),
        atoms: (0, 0),
    }
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

/// How many mutable entities (translation, rotation, torsions) the system has.
pub fn count_mutable_entities(conf: &Conf) -> usize {
    let mut n = 0usize;
    for l in &conf.ligands {
        n += 2 + l.torsions.len();
    }
    for f in &conf.flex {
        n += f.len();
    }
    n
}

/// Apply exactly one random mutation to `conf`, in place.
///
/// This is Vina's `mutate_conf` (`mutate.cpp`, Apache-2.0): pick one of the
/// mutable entities uniformly at random and perturb it 鈥?///
/// * translation: `position += amplitude * U(sphere)`
/// * rotation: `orientation = quaternion_increment(orientation, amplitude / Rg * U(sphere))`
/// * torsion: re-drawn uniformly from `[-蟺, 蟺]`
///
/// where `Rg` is the ligand's gyration radius, which makes the rotational
/// perturbation scale-free with respect to molecular size.
pub fn mutate_conf(conf: &mut Conf, amplitude: f64, gyration_radii: &[f64], rng: &mut Rng) {
    let n = count_mutable_entities(conf);
    if n == 0 {
        return;
    }
    let mut which = rng.int(0, n as i64 - 1) as usize;

    for (li, lig) in conf.ligands.iter_mut().enumerate() {
        if which == 0 {
            lig.position += amplitude * rng.inside_unit_sphere();
            return;
        }
        which -= 1;
        if which == 0 {
            let gr = gyration_radii.get(li).copied().unwrap_or(0.0);
            if gr > EPSILON {
                let rot = (amplitude / gr) * rng.inside_unit_sphere();
                lig.orientation = quaternion_increment(lig.orientation, rot);
            }
            return;
        }
        which -= 1;
        if which < lig.torsions.len() {
            lig.torsions[which] = rng.angle();
            return;
        }
        which -= lig.torsions.len();
    }
    for f in conf.flex.iter_mut() {
        if which < f.len() {
            f[which] = rng.angle();
            return;
        }
        which -= f.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AdType;

    /// Build a 4-atom chain A-B-C-D with a single torsion about the B-C bond.
    ///
    /// PDBQT would express this as `BRANCH 2 3` (parent B, attachment C): the
    /// root frame owns A, B and the attachment atom C, and the torsion segment
    /// owns D. Only D moves when the torsion is applied.
    fn chain_topology() -> (TopologyNode, Vec<DVec3>) {
        let coords = vec![
            DVec3::new(-1.0, 0.0, 0.0), // 0 = A
            DVec3::new(0.0, 0.0, 0.0),  // 1 = B (parent atom of the rotor)
            DVec3::new(1.0, 0.0, 0.0),  // 2 = C (attachment atom, on the axis)
            DVec3::new(1.0, 1.0, 0.0),  // 3 = D (rotates about +x)
        ];
        let segment = TopologyNode {
            atoms: vec![3],
            immobile: None,
            attachment: Some(2),
            attachment_coords: Some(coords[2]),
            axis_root: Some(coords[1]),
            axis_root_atom: Some(1),
            children: vec![],
        };
        let root = TopologyNode {
            atoms: vec![0, 1],
            immobile: None,
            attachment: None,
            attachment_coords: None,
            axis_root: None,
            axis_root_atom: None,
            children: vec![segment],
        };
        (root, coords)
    }

    #[test]
    fn forward_kinematics_at_identity_reproduces_input() {
        let (top, coords) = chain_topology();
        let (mut tree, root_idx) = build_ligand_tree(&top, &coords, 4);
        assert_eq!(tree.num_torsions, 1);
        let mut out = vec![DVec3::ZERO; 4];
        let conf = LigandConf {
            position: coords[root_idx],
            orientation: DQuat::IDENTITY,
            torsions: vec![0.0],
        };
        tree.apply_ligand(&conf, &mut out);
        for i in 0..4 {
            assert!((out[i] - coords[i]).length() < 1e-12, "{i}: {:?}", out[i]);
        }
    }

    #[test]
    fn torsion_rotates_only_the_moving_side() {
        let (top, coords) = chain_topology();
        let (mut tree, root_idx) = build_ligand_tree(&top, &coords, 4);
        let mut out = vec![DVec3::ZERO; 4];
        let conf = LigandConf {
            position: coords[root_idx],
            orientation: DQuat::IDENTITY,
            torsions: vec![std::f64::consts::FRAC_PI_2],
        };
        tree.apply_ligand(&conf, &mut out);
        // A, B and the attachment atom C all lie on the rotation axis and must
        // not move.
        for i in 0..3 {
            assert!((out[i] - coords[i]).length() < 1e-12, "{i} moved");
        }
        // D is 90 degrees around +x: (0, 1, 0) -> (0, 0, 1).
        let expect = DVec3::new(1.0, 0.0, 1.0);
        assert!((out[3] - expect).length() < 1e-12, "{:?}", out[3]);
        // Bond lengths are preserved.
        assert!(((out[3] - out[2]).length() - (coords[3] - coords[2]).length()).abs() < 1e-12);
    }

    #[test]
    fn rigid_body_transform_is_an_isometry() {
        let (top, coords) = chain_topology();
        let (mut tree, root_idx) = build_ligand_tree(&top, &coords, 4);
        let mut out = vec![DVec3::ZERO; 4];
        let conf = LigandConf {
            position: coords[root_idx] + DVec3::new(3.0, -1.0, 2.0),
            orientation: DQuat::from_axis_angle(DVec3::new(1.0, 2.0, 3.0).normalize(), 1.1),
            torsions: vec![0.4],
        };
        tree.apply_ligand(&conf, &mut out);
        // All pairwise distances must be unchanged.
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d0 = (coords[i] - coords[j]).length();
                let d1 = (out[i] - out[j]).length();
                assert!((d0 - d1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn gradient_matches_finite_differences_for_the_rigid_body() {
        let (top, coords) = chain_topology();
        let (mut tree, root_idx) = build_ligand_tree(&top, &coords, 4);
        let layout = DofLayout {
            ligands: vec![1],
            flex: vec![],
        };
        // A smooth toy energy: sum of squared distances to a fixed point.
        let target = DVec3::new(4.0, 2.0, -1.0);
        let energy = |coords: &[DVec3]| -> f64 {
            coords
                .iter()
                .map(|c| (*c - target).length_squared())
                .sum::<f64>()
        };
        let base = LigandConf {
            position: coords[root_idx],
            orientation: DQuat::IDENTITY,
            torsions: vec![0.3],
        };
        let mut out = coords.clone();
        tree.apply_ligand(&base, &mut out);
        let mut forces = vec![DVec3::ZERO; 4];
        for i in 0..4 {
            forces[i] = 2.0 * (out[i] - target);
        }
        let mut g = vec![0.0; layout.num_floats()];
        tree.accumulate_gradient(&out, &forces, &mut g, Some((0, 3)), 6);

        let mut conf = Conf {
            ligands: vec![base.clone()],
            flex: vec![],
        };
        let eps = 1e-6;
        let mut numeric = vec![0.0; layout.num_floats()];
        for k in 0..layout.num_floats() {
            let mut dplus = vec![0.0; layout.num_floats()];
            let mut dminus = vec![0.0; layout.num_floats()];
            dplus[k] = eps;
            dminus[k] = -eps;
            let mut c1 = conf.clone();
            c1.increment_flat(&layout, &dplus, 1.0);
            let mut o1 = coords.clone();
            tree.apply_ligand(&c1.ligands[0], &mut o1);
            let mut c2 = conf.clone();
            c2.increment_flat(&layout, &dminus, 1.0);
            let mut o2 = coords.clone();
            tree.apply_ligand(&c2.ligands[0], &mut o2);
            numeric[k] = (energy(&o1) - energy(&o2)) / (2.0 * eps);
        }
        let mut d0 = vec![0.0; layout.num_floats()];
        d0[0] = 0.0;
        conf.increment_flat(&layout, &d0, 0.0);
        for k in 0..layout.num_floats() {
            assert!(
                (g[k] - numeric[k]).abs() < 1e-5,
                "dof {k}: analytic {} vs numeric {}",
                g[k],
                numeric[k]
            );
        }
    }

    #[test]
    fn change_flat_layout_round_trips() {
        let layout = DofLayout {
            ligands: vec![2],
            flex: vec![1],
        };
        assert_eq!(layout.num_floats(), 6 + 2 + 1);
        let mut conf = Conf {
            ligands: vec![LigandConf::null(DVec3::ZERO, 2)],
            flex: vec![vec![0.0]],
        };
        let mut delta = vec![0.0; layout.num_floats()];
        delta[0] = 1.0;
        delta[layout.ligand_torsion_offset(0)] = 0.5;
        delta[layout.flex_torsion_offset(0)] = -0.5;
        conf.increment_flat(&layout, &delta, 1.0);
        assert!((conf.ligands[0].position.x - 1.0).abs() < 1e-15);
        assert!((conf.ligands[0].torsions[0] - 0.5).abs() < 1e-15);
        assert!((conf.flex[0][0] + 0.5).abs() < 1e-15);
    }

    #[test]
    fn mutation_changes_exactly_one_entity() {
        let layout = DofLayout {
            ligands: vec![2],
            flex: vec![],
        };
        let conf = Conf {
            ligands: vec![LigandConf::null(DVec3::new(1.0, 2.0, 3.0), 2)],
            flex: vec![],
        };
        let _ = layout;
        let mut rng = Rng::new(17);
        for _ in 0..200 {
            let mut c = conf.clone();
            mutate_conf(&mut c, 2.0, &[1.5], &mut rng);
            let mut diffs = 0;
            if c.ligands[0].position != conf.ligands[0].position {
                diffs += 1;
            }
            if c.ligands[0].orientation != conf.ligands[0].orientation {
                diffs += 1;
            }
            if c.ligands[0].torsions != conf.ligands[0].torsions {
                diffs += 1;
            }
            assert_eq!(diffs, 1, "mutation touched {diffs} entities");
        }
    }

    #[test]
    fn flex_tree_pins_the_root_origin() {
        // Flexible side chain. The parent atom P and the attachment atom C are
        // both immobile receptor atoms; only D and E are movable.
        let p = DVec3::new(0.0, 0.0, 0.0);
        let c = DVec3::new(1.0, 0.0, 0.0);
        let coords = vec![DVec3::new(1.0, 1.0, 0.0), DVec3::new(1.0, 1.0, 1.0)];
        let branch = TopologyNode {
            atoms: vec![0, 1],
            immobile: None,
            attachment: None,
            attachment_coords: Some(c),
            axis_root: Some(p),
            axis_root_atom: None,
            children: vec![],
        };
        let mut out = vec![DVec3::ZERO; 2];
        let mut tree = build_flex_tree(&branch, &coords, 2);
        assert_eq!(tree.num_torsions, 1);
        tree.apply_flex(&[0.0], &mut out);
        assert!((out[0] - coords[0]).length() < 1e-12);
        assert!((out[1] - coords[1]).length() < 1e-12);
        // 90 degrees about +x: (y, z) -> (-z, y).
        tree.apply_flex(&[std::f64::consts::FRAC_PI_2], &mut out);
        let expect_d = DVec3::new(1.0, 0.0, 1.0);
        let expect_e = DVec3::new(1.0, -1.0, 1.0);
        assert!((out[0] - expect_d).length() < 1e-12, "{:?}", out[0]);
        assert!((out[1] - expect_e).length() < 1e-12, "{:?}", out[1]);
    }

    #[test]
    fn atom_types_exist_for_movable_model_lifecycle() {
        let (top, coords) = chain_topology();
        let (mut tree, root_idx) = build_ligand_tree(&top, &coords, 4);
        let mut out = vec![DVec3::ZERO; 4];
        let conf = LigandConf::null(coords[root_idx], 1);
        tree.apply_ligand(&conf, &mut out);
        let atoms = vec![
            Atom::new(out[0], AdType::C, 0.0),
            Atom::new(out[1], AdType::C, 0.0),
            Atom::new(out[2], AdType::O, 0.0),
        ];
        assert_eq!(atoms.len(), 3);
    }
}

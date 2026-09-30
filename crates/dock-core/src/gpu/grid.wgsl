// SPDX-License-Identifier: GPL-3.0-or-later
//
// OpenDocking compute kernels (WGSL).
//
// 1. `main`           -- affinity-grid construction: one invocation per
//                        (x, y, z, atom_type) sample, summing the interaction
//                        energy of a probe of that type against every receptor
//                        atom within the force-field cutoff.
// 2. `expand_batch`   -- batch forward kinematics: expands the rigid-body +
//                        torsion tree of every conformation in a batch and
//                        writes every movable atom coordinate.
// 3. `evaluate_batch` -- batch pair evaluation: the non-cached interaction
//                        energy of every ligand atom of every conformation
//                        against an explicit receptor atom list.
//
// Layout rules that matter
// ------------------------
// * Uniform arrays of `f32` have a 16-byte element stride in WGSL, so the
//   force-field tables live in *storage* buffers (bindings 3 / 24) with the
//   natural 4-byte stride; see the comment at `Params`.
// * Every `@group(0) @binding(N)` pair may appear once per module, so the three
//   kernels use disjoint binding ranges (0.., 10.., 20..) and each pipeline is
//   created with its own bind group layout listing only the bindings it uses.

// ---------------------------------------------------------------------------
// Shared parameter tables and geometry helpers
// ---------------------------------------------------------------------------

struct GridParams {
    // (n_x, n_y, n_z, n_types) -- number of *sample points* per axis.
    shape: vec4<u32>,
    // Grid origin in Angstrom.
    origin: vec4<f32>,
    // 1 / spacing, padded.
    inv_spacing: vec4<f32>,
    // Force-field cutoff in Angstrom.
    cutoff: f32,
    // Number of receptor atoms.
    n_atoms: u32,
    // Number of atom types.
    n_types: u32,
    // 0 = Vina force field, 1 = Vinardo.
    variant: u32,
};

struct GpuAtom {
    // Position in Angstrom (.xyz used).
    position: vec4<f32>,
    // X-Score type index in .x.
    type_index: vec4<u32>,
};

struct Params {
    // Per-type van der Waals radii for the selected force field.
    radii: array<f32, 32>,
    // 1.0 when the type is hydrophobic, 0.0 otherwise.
    hydrophobic: array<f32, 32>,
    // 1.0 when the type is an H-bond donor, 0.0 otherwise.
    donor: array<f32, 32>,
    // 1.0 when the type is an H-bond acceptor, 0.0 otherwise.
    acceptor: array<f32, 32>,
    // Term weights: gauss1, gauss2, repulsion, hydrophobic, hydrogen, glue.
    weights: array<f32, 8>,
    // Gaussian offsets, widths and cutoffs for the (up to) two gaussians.
    gauss_offset: array<f32, 4>,
    gauss_width: array<f32, 4>,
    // Slope-step bounds: (good, bad) for the hydrophobic and H-bond terms.
    ramp: array<f32, 4>,
    // Maps an output slot back onto its X-Score type index (slot -> type).
    slot_to_type: array<u32, 32>,
};

// X-Score type 31 is the "no type" sentinel (a hydrogen: invisible to the
// force field); 21, 24, 27 and 30 are the macrocycle closure dummies.
const XS_UNTYPED: u32 = 31u;

fn is_glue_type(t: u32) -> bool {
    return t == 21u || t == 24u || t == 27u || t == 30u;
}

fn slope_step(x_bad: f32, x_good: f32, x: f32) -> f32 {
    if (x_bad < x_good) {
        if (x <= x_bad) { return 0.0; }
        if (x >= x_good) { return 1.0; }
    } else {
        if (x >= x_bad) { return 0.0; }
        if (x <= x_good) { return 1.0; }
    }
    return (x - x_bad) / (x_good - x_bad);
}

fn optimal_distance(t1: u32, t2: u32) -> f32 {
    // Closure dummy atoms (types 21, 24, 27, 30) have no radius.
    if (is_glue_type(t1) || is_glue_type(t2)) {
        return 0.0;
    }
    return fparams.radii[t1] + fparams.radii[t2];
}

// Full pairwise potential for one (probe, receptor atom) pair at distance r.
fn pair_energy(t1: u32, t2: u32, r: f32, cutoff: f32) -> f32 {
    let opt = optimal_distance(t1, t2);
    var e = 0.0;

    // --- gaussian terms ---
    if (r < cutoff) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let u = r - (opt + fparams.gauss_offset[k]);
            let w = fparams.gauss_width[k];
            e = e + fparams.weights[k] * exp(-(u / w) * (u / w));
        }
        // --- soft-sphere repulsion ---
        let d = r - opt;
        if (d < 0.0) {
            e = e + fparams.weights[2] * d * d;
        }
        // --- hydrophobic contact ---
        let hb = fparams.hydrophobic[t1] * fparams.hydrophobic[t2];
        e = e + fparams.weights[3] * hb *
            slope_step(fparams.ramp[1], fparams.ramp[0], r - opt);
        // --- non-directional hydrogen bond ---
        let donor_acceptor = fparams.donor[t1] * fparams.acceptor[t2] +
                             fparams.donor[t2] * fparams.acceptor[t1];
        let hbv = clamp(donor_acceptor, 0.0, 1.0);
        e = e + fparams.weights[4] * hbv *
            slope_step(fparams.ramp[3], fparams.ramp[2], r - opt);
    }
    return e;
}

// Vina's `curl`: a soft saturation of a positive per-atom energy, applied to
// the *summed* per-atom term exactly as `scoring::noncache::NonCache::eval`
// does on the CPU.
fn curl_scalar(e: f32, v: f32) -> f32 {
    if (e > 0.0 && v < 0.1 * 3.402823466e+38) {
        var tmp = 0.0;
        if (v >= 1.192092896e-7) {
            tmp = v / (v + e);
        }
        return e * tmp;
    }
    return e;
}

// ---------------------------------------------------------------------------
// 1. Affinity grid
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> gparams: GridParams;
@group(0) @binding(1) var<storage, read> atoms: array<GpuAtom>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
// The parameter tables live in a *storage* buffer: a uniform array of f32 has a
// 16-byte element stride in WGSL, which would waste three quarters of the
// buffer, while a storage array keeps the natural 4-byte stride.
@group(0) @binding(3) var<storage, read> fparams: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    let n_total = gparams.shape.x * gparams.shape.y * gparams.shape.z * gparams.shape.w;
    if (id >= n_total) {
        return;
    }

    let n_types = gparams.shape.w;
    let n_z = gparams.shape.z;
    let n_y = gparams.shape.y;

    var rest = id;
    let t = rest % n_types;
    rest = rest / n_types;
    let iz = rest % n_z;
    rest = rest / n_z;
    let iy = rest % n_y;
    let ix = rest / n_y;

    let spacing = 1.0 / gparams.inv_spacing.xyz;
    let probe = gparams.origin.xyz + spacing * vec3<f32>(f32(ix), f32(iy), f32(iz));

    var acc = 0.0;
    let cutoff_sq = gparams.cutoff * gparams.cutoff;
    for (var j = 0u; j < gparams.n_atoms; j = j + 1u) {
        let delta = probe - atoms[j].position.xyz;
        let r2 = dot(delta, delta);
        if (r2 < cutoff_sq) {
            acc = acc + pair_energy(atoms[j].type_index.x, fparams.slot_to_type[t], sqrt(r2), gparams.cutoff);
        }
    }
    out[id] = acc;
}

// ---------------------------------------------------------------------------
// 2. Batch forward kinematics
// ---------------------------------------------------------------------------
//
// One invocation per (conformation, node, atom slot) = (z, y, x) of the
// dispatch grid. Each invocation walks the kinematic chain from the atom's
// owning frame up to the root, then applies the frames downwards, which needs
// no inter-invocation synchronisation at all: every atom is written by exactly
// one invocation and frames are pure functions of the conformation.

struct BatchParams {
    // (n_conformations, n_nodes, max_atoms_per_node, n_torsions)
    shape: vec4<u32>,
    // (n_atoms, atom_base, conf_stride, reserved)
    info: vec4<u32>,
};

struct KinNode {
    // Parent node index; 0xFFFFFFFF for a root.
    parent: u32,
    // 0 = rigid root, 1 = torsion segment, 2 = pinned (absolute) root.
    kind: u32,
    // Torsion slot inside the conformation, or 0xFFFFFFFF.
    torsion: u32,
    // First movable atom owned by this frame.
    atom_begin: u32,
    // Number of atoms owned by this frame.
    atom_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Frame origin expressed in the parent frame.
    rel_origin: vec4<f32>,
    // Rotation axis expressed in the parent frame.
    rel_axis: vec4<f32>,
    // Pinned (laboratory) axis, used by `kind == 2`.
    abs_axis: vec4<f32>,
    // Pinned (laboratory) origin, used by `kind == 2`.
    abs_origin: vec4<f32>,
};

@group(0) @binding(10) var<uniform> bparams: BatchParams;
@group(0) @binding(11) var<storage, read> nodes: array<KinNode>;
// Frame-local coordinates, one `vec4` per movable atom of the tree.
@group(0) @binding(12) var<storage, read> locals: array<vec4<f32>>;
// Packed conformations: [x y z | qx qy qz qw | torsion_0 ... torsion_n].
@group(0) @binding(13) var<storage, read> confs: array<f32>;
// Output coordinates, one `vec4` per (conformation, atom).
@group(0) @binding(14) var<storage, read_write> coords: array<vec4<f32>>;

fn conf_scalar(conf: u32, index: u32) -> f32 {
    return confs[conf * bparams.info.z + index];
}

fn conf_position(conf: u32) -> vec3<f32> {
    return vec3<f32>(conf_scalar(conf, 0u), conf_scalar(conf, 1u), conf_scalar(conf, 2u));
}

fn conf_orientation(conf: u32) -> vec4<f32> {
    return vec4<f32>(
        conf_scalar(conf, 3u),
        conf_scalar(conf, 4u),
        conf_scalar(conf, 5u),
        conf_scalar(conf, 6u),
    );
}

fn conf_torsion(conf: u32, slot: u32) -> f32 {
    if (slot == 0xFFFFFFFFu) {
        return 0.0;
    }
    return conf_scalar(conf, 7u + slot);
}

// Hamilton product, identical to `glam`'s `Quat` multiplication (which is what
// `kinematics::Node::apply_segment` uses).
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let n = length(q);
    if (n > 0.0) {
        return q / n;
    }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// `math::normalize_angle`: wrap into (-pi, pi].
fn normalize_angle(x: f32) -> f32 {
    let pi = 3.141592653589793;
    if (x > -pi && x <= pi) {
        return x;
    }
    var a = x % (2.0 * pi);
    if (a > pi) {
        a = a - 2.0 * pi;
    } else if (a <= -pi) {
        a = a + 2.0 * pi;
    }
    return a;
}

// `math::angle_to_quaternion`: the axis is used as given (it is unit length by
// construction) and the angle is normalised.
fn angle_to_quat(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let a = normalize_angle(angle);
    let s = sin(a * 0.5);
    return vec4<f32>(axis * s, cos(a * 0.5));
}

const NO_PARENT: u32 = 0xFFFFFFFFu;
// Maximum supported tree depth. Every real ligand is far below this (the
// deepest fixture in the repository has depth 5); a deeper tree falls back to
// the CPU implementation, which the host checks before dispatching.
const MAX_DEPTH: u32 = 32u;

@compute @workgroup_size(64)
fn expand_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let conf = gid.z;
    if (conf >= bparams.shape.x || gid.y >= bparams.shape.y) {
        return;
    }
    let node_id = gid.y;
    let node = nodes[node_id];
    if (gid.x >= node.atom_count) {
        return;
    }
    let atom = node.atom_begin + gid.x;

    // 1. Walk from this atom's frame up to the root, remembering the chain.
    var chain: array<u32, 32>;
    var depth = 0u;
    var cur = node_id;
    loop {
        chain[depth] = cur;
        depth = depth + 1u;
        let parent = nodes[cur].parent;
        if (parent == NO_PARENT || depth >= MAX_DEPTH) {
            break;
        }
        cur = parent;
    }

    // 2. Walk back down, composing the frames.
    var origin = vec3<f32>(0.0, 0.0, 0.0);
    var orientation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    var i = depth;
    loop {
        if (i == 0u) {
            break;
        }
        i = i - 1u;
        let n = nodes[chain[i]];
        if (n.kind == 0u) {
            // Rigid body: the pose comes straight from the conformation.
            origin = conf_position(conf);
            orientation = conf_orientation(conf);
        } else if (n.kind == 2u) {
            // Pinned root of a flexible residue: independent of the parent.
            origin = n.abs_origin.xyz;
            let axis = normalize(n.abs_axis.xyz);
            orientation = angle_to_quat(axis, conf_torsion(conf, n.torsion));
        } else {
            origin = origin + quat_rotate(orientation, n.rel_origin.xyz);
            let axis = quat_rotate(orientation, n.rel_axis.xyz);
            let qt = angle_to_quat(axis, conf_torsion(conf, n.torsion));
            orientation = quat_normalize(quat_mul(qt, orientation));
        }
    }

    let local = locals[atom - bparams.info.y];
    let world = origin + quat_rotate(orientation, local.xyz);
    coords[conf * bparams.info.x + atom - bparams.info.y] = vec4<f32>(world, 0.0);
}

// ---------------------------------------------------------------------------
// 3. Batch pair evaluation
// ---------------------------------------------------------------------------
//
// One invocation per (conformation, ligand atom). `atoms` holds the ligand
// atom templates first (their `position` field is unused: the coordinates come
// from the expanded `coords` buffer) followed by the receptor atoms, so the
// whole kernel needs four storage buffers - the limit of
// `Limits::downlevel_defaults()`.

struct EvalParams {
    // (n_conformations, n_ligand_atoms, n_receptor_atoms, receptor_offset)
    shape: vec4<u32>,
    // Force-field cutoff in Angstrom.
    cutoff: f32,
    // Vina's `curl` cap (`v`).
    curl_cap: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(20) var<uniform> eparams: EvalParams;
@group(0) @binding(21) var<storage, read> eatoms: array<GpuAtom>;
@group(0) @binding(22) var<storage, read> ecoords: array<vec4<f32>>;
@group(0) @binding(23) var<storage, read_write> eout: array<f32>;
// The force-field tables are the *same* binding as the grid kernel uses
// (`fparams`, binding 3) because `pair_energy` reads them directly; the bind
// group layout of this pipeline therefore contains binding 3 as well. Sharing
// one declaration is also what keeps every (group, binding) pair unique in the
// module, which WGSL requires.

@compute @workgroup_size(64)
fn evaluate_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let conf = gid.y;
    if (conf >= eparams.shape.x || gid.x >= eparams.shape.y) {
        return;
    }
    let i = gid.x;
    let t1 = eatoms[i].type_index.x;

    var e = 0.0;
    // Untyped hydrogens and closure dummies carry no interaction, exactly as
    // in `scoring::noncache::NonCache::eval`.
    if (t1 != XS_UNTYPED && !is_glue_type(t1)) {
        let pos = ecoords[conf * eparams.shape.y + i].xyz;
        let cutoff_sq = eparams.cutoff * eparams.cutoff;
        for (var j = 0u; j < eparams.shape.z; j = j + 1u) {
            let rec = eatoms[eparams.shape.w + j];
            let t2 = rec.type_index.x;
            if (t2 == XS_UNTYPED) {
                continue;
            }
            let delta = pos - rec.position.xyz;
            let r2 = dot(delta, delta);
            if (r2 < cutoff_sq) {
                e = e + pair_energy(t1, t2, sqrt(r2), eparams.cutoff);
            }
        }
    }
    eout[conf * eparams.shape.y + i] = curl_scalar(e, eparams.curl_cap);
}

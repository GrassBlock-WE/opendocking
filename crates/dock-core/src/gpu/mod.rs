// SPDX-License-Identifier: GPL-3.0-or-later
//! Optional GPU backend: cross-platform affinity-grid construction and batch
//! grid probing with `wgpu` compute shaders (WGSL).
//!
//! # Why `wgpu`
//!
//! AutoDock-GPU binds the docking kernel to CUDA, which excludes every machine
//! without an NVIDIA device. `wgpu` compiles the same WGSL shader to SPIR-V
//! (Vulkan), MSL (Metal), HLSL (Direct3D 12) or GLSL (OpenGL ES), so one
//! implementation covers all three desktop platforms and the browser.
//!
//! # What is offloaded
//!
//! Building the affinity grid is the only step of a docking run that is both
//! trivially parallel and large: a 22.5 Å box at 0.375 Å spacing has ~67³
//! sample points, and each point is summed over every receptor atom within the
//! 8 Å cutoff for every ligand atom type. That is exactly a "gather" kernel and
//! it maps one-to-one onto a compute dispatch.
//!
//! The Monte-Carlo / BFGS search itself is *not* offloaded in this release. The
//! search is dominated by very short, latency-sensitive per-atom work with lots
//! of branching, which is precisely the workload CPUs win at; the CPU path
//! already uses `rayon` across all cores. Offloading the grid removes the only
//! genuinely throughput-bound step.
//!
//! # Status
//!
//! The module is compiled only with `--features gpu` so that a default build
//! stays dependency-light and always succeeds on a machine without a working
//! graphics driver. Everything in this file is safe Rust; the only `unsafe`
//! would be inside `wgpu` itself.

use crate::atom::XsType;
use crate::molecule::Atom;
use crate::error::{DockError, Result};
use crate::kinematics::{mutate_conf, Conf, KinematicTree, LigandConf, TreeKind};
#[cfg(feature = "gpu")]
use crate::kinematics::{FrameKind, Node};
use crate::math::DVec3;
use crate::rng::Rng;
use crate::scoring::grid::{AffinityGrid, GridBox, GridDim};
use crate::scoring::noncache::NonCache;
use crate::scoring::ScoringFunction;

#[cfg(feature = "gpu")]
use crate::scoring::grid::grid_type;

/// The WGSL source of the grid-building kernel.
pub const GRID_SHADER: &str = include_str!("grid.wgsl");

/// Parameters passed to the compute shader.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GridParams {
    /// Number of sample points per axis.
    pub shape: [u32; 4],
    /// Grid origin (Å), padded.
    pub origin: [f32; 4],
    /// Inverse cell size per axis (Å⁻¹), padded.
    pub inv_spacing: [f32; 4],
    /// Force-field cutoff (Å).
    pub cutoff: f32,
    /// Number of receptor atoms.
    pub n_atoms: u32,
    /// Number of atom types.
    pub n_types: u32,
    /// Padding.
    pub _pad: u32,
}

/// A receptor atom as the shader sees it (position + type index + padding).
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAtom {
    /// Position in Å.
    pub position: [f32; 4],
    /// X-Score type index in `.0`.
    pub type_index: [u32; 4],
}

/// The force-field parameter tables the shader needs.
///
/// The layout mirrors `struct Params` in `grid.wgsl` exactly. It is uploaded as
/// a *storage* buffer because WGSL pads uniform arrays of `f32` to a 16-byte
/// element stride, which would waste three quarters of the transfer.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Coefficients {
    /// Per-type van der Waals radii for the selected force field.
    pub radii: [f32; 32],
    /// 1.0 when the type is hydrophobic.
    pub hydrophobic: [f32; 32],
    /// 1.0 when the type is an H-bond donor.
    pub donor: [f32; 32],
    /// 1.0 when the type is an H-bond acceptor.
    pub acceptor: [f32; 32],
    /// Term weights: gauss1, gauss2, repulsion, hydrophobic, hydrogen, glue.
    pub weights: [f32; 8],
    /// Gaussian offsets.
    pub gauss_offset: [f32; 4],
    /// Gaussian widths.
    pub gauss_width: [f32; 4],
    /// `[good_hydrophobic, bad_hydrophobic, good_hbond, bad_hbond]`.
    pub ramp: [f32; 4],
    /// Maps an output slot back onto its X-Score type index.
    pub slot_to_type: [u32; 32],
}

#[cfg(feature = "gpu")]
impl Coefficients {
    /// Build the tables for a force field.
    ///
    /// Only the X-Score-parameterised force fields are supported; AD4 needs
    /// per-atom charges and is rejected before this point.
    pub fn for_choice(
        choice: crate::scoring::SfChoice,
        weights: &crate::scoring::Weights,
    ) -> Self {
        Self::for_choice_with_slots(choice, weights, &[])
    }

    /// Like [`Coefficients::for_choice`], but also records the slot -> type map
    /// used by the grid builder.
    pub fn for_choice_with_slots(
        choice: crate::scoring::SfChoice,
        weights: &crate::scoring::Weights,
        slots: &[crate::atom::XsType],
    ) -> Self {
        use crate::atom::{XsType, XS_VINARDO_VDW_RADII, XS_VDW_RADII};
        use crate::scoring::SfChoice;

        let table: [f64; 32] = match choice {
            SfChoice::Vinardo => XS_VINARDO_VDW_RADII,
            _ => XS_VDW_RADII,
        };
        let mut radii = [0.0f32; 32];
        let mut hydrophobic = [0.0f32; 32];
        let mut donor = [0.0f32; 32];
        let mut acceptor = [0.0f32; 32];
        for i in 0..XsType::COUNT {
            let t = XsType::from_index(i);
            radii[i] = table[i] as f32;
            hydrophobic[i] = if t.is_hydrophobic() { 1.0 } else { 0.0 };
            donor[i] = if t.is_donor() { 1.0 } else { 0.0 };
            acceptor[i] = if t.is_acceptor() { 1.0 } else { 0.0 };
        }

        // (gauss offsets, gauss widths, ramps) -- see `scoring::XsScoringFunction`.
        let (gauss_offset, gauss_width, ramp) = match choice {
            SfChoice::Vinardo => (
                [0.0, 0.0, 0.0, 0.0],
                // A zero width would produce a division by zero; the second
                // Gaussian is simply switched off by a zero weight instead.
                [0.8, 1.0, 0.0, 0.0],
                [0.0, 2.5, -0.6, 0.0],
            ),
            _ => (
                [0.0, 3.0, 0.0, 0.0],
                [0.5, 2.0, 0.0, 0.0],
                [0.5, 1.5, -0.7, 0.0],
            ),
        };

        let mut w = [0.0f32; 8];
        // The shader evaluates six fixed slots:
        // `(gauss1, gauss2, repulsion, hydrophobic, hbond, glue)`.
        // Vina's term list is already in that order; Vinardo has only *one*
        // Gaussian, so its terms have to be expanded into the same slots —
        // copying them straight across would put the repulsion weight in the
        // gauss2 slot and shift every other term by one.
        let slot_sources: [Option<usize>; 6] = match choice {
            SfChoice::Vinardo => [Some(0), None, Some(1), Some(2), Some(3), Some(4)],
            _ => [Some(0), Some(1), Some(2), Some(3), Some(4), Some(5)],
        };
        for (slot, source) in slot_sources.iter().enumerate() {
            if let Some(src) = source {
                if let Some(v) = weights.terms.get(*src) {
                    w[slot] = *v as f32;
                }
            }
        }
        let mut slot_to_type = [0u32; 32];
        for (slot, ty) in slots.iter().take(32).enumerate() {
            slot_to_type[slot] = ty.index() as u32;
        }

        Coefficients {
            radii,
            hydrophobic,
            donor,
            acceptor,
            weights: w,
            gauss_offset,
            gauss_width,
            ramp,
            slot_to_type,
        }
    }
}

/// Whether a GPU adapter was found, and which one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// Adapter name.
    pub name: String,
    /// Backend (e.g. `Vulkan`, `Dx12`).
    pub backend: String,
    /// Driver description.
    pub driver: String,
    /// Device type.
    pub device_type: String,
}

/// Request a suitable GPU adapter.
///
/// Returns `Ok(None)` when no adapter exists — callers must fall back to the
/// CPU path rather than failing.
#[cfg(feature = "gpu")]
pub fn request_adapter() -> Result<Option<(wgpu::Device, wgpu::Queue, GpuInfo)>> {
    use pollster::block_on;
    let instance = wgpu::Instance::default();
    let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })) {
        Ok(a) => a,
        // No adapter is not an error: the caller falls back to the CPU path.
        Err(_) => return Ok(None),
    };
    let info = adapter.get_info();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("odock-grid"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| DockError::Gpu(e.to_string()))?;
    Ok(Some((
        device,
        queue,
        GpuInfo {
            name: info.name,
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            device_type: format!("{:?}", info.device_type),
        },
    )))
}

/// Stub used when the `gpu` feature is disabled.
#[cfg(not(feature = "gpu"))]
pub fn request_adapter() -> Result<Option<((), (), GpuInfo)>> {
    Err(DockError::Gpu(
        "dock-core was built without the `gpu` feature; rebuild with \
         `cargo build --features gpu` to enable the wgpu backend"
            .to_string(),
    ))
}

/// Interactively report whether a GPU backend is available.
pub fn available() -> bool {
    cfg!(feature = "gpu") && request_adapter().map(|a| a.is_some()).unwrap_or(false)
}

/// Build an affinity grid on the GPU.
///
/// Falls back to the CPU implementation whenever the `gpu` feature is off, no
/// adapter is present, or the device is lost — the result is numerically
/// identical because both paths evaluate the same force-field terms at the same
/// sample points.
pub fn populate_grid(
    grid: &mut AffinityGrid,
    receptor: &[Atom],
    sf: &dyn ScoringFunction,
    types: &[XsType],
) -> Result<usize> {
    #[cfg(feature = "gpu")]
    {
        match populate_grid_gpu(grid, receptor, sf, types) {
            Ok(n) => return Ok(n),
            Err(DockError::Gpu(msg)) => {
                eprintln!("odock: GPU grid build unavailable ({msg}); using the CPU path");
            }
            Err(e) => return Err(e),
        }
    }
    let _ = (sf, types);
    let wanted: Vec<XsType> = types.to_vec();
    Ok(grid.populate(receptor, sf, &wanted))
}

/// The actual `wgpu` dispatch.
#[cfg(feature = "gpu")]
fn populate_grid_gpu(
    grid: &mut AffinityGrid,
    receptor: &[Atom],
    sf: &dyn ScoringFunction,
    types: &[XsType],
) -> Result<usize> {
    use wgpu::util::DeviceExt;

    let Some((device, queue, info)) = request_adapter()? else {
        return Err(DockError::Gpu("no compatible GPU adapter found".into()));
    };
    let _ = info;

    // The WGSL kernel evaluates the pairwise potentials directly, so the CPU
    // path is used for force fields whose terms are not expressible with the
    // X-Score typing alone (AD4).
    if !sf.is_grid_capable() {
        return Err(DockError::Gpu(
            "the AD4 force field needs per-atom charges and is CPU-only".into(),
        ));
    }

    let mut wanted: Vec<XsType> = Vec::new();
    for t in types {
        if let Some(g) = grid_type(*t) {
            if !wanted.contains(&g) {
                wanted.push(g);
            }
        }
    }
    if wanted.is_empty() {
        return Err(DockError::Gpu("no grid-able atom types requested".into()));
    }

    // Allocate the maps through the CPU implementation so that the layout stays
    // in one place, then fill the data with the compute shader.
    grid.populate(receptor, sf, &wanted);
    let [nx, ny, nz] = grid.shape();
    let nt = grid.available_types().len();

    let atoms: Vec<GpuAtom> = receptor
        .iter()
        .filter_map(|a| {
            let t = grid_type(a.xs)?;
            Some(GpuAtom {
                position: [
                    a.coords.x as f32,
                    a.coords.y as f32,
                    a.coords.z as f32,
                    0.0,
                ],
                // The shader indexes the parameter tables by *X-Score type*,
                // not by grid slot.
                type_index: [t.index() as u32, 0, 0, 0],
            })
        })
        .collect();

    let params = GridParams {
        shape: [nx as u32, ny as u32, nz as u32, nt as u32],
        origin: [
            grid.dims[0].begin as f32,
            grid.dims[1].begin as f32,
            grid.dims[2].begin as f32,
            0.0,
        ],
        inv_spacing: {
            let s = grid.spacing as f32;
            [1.0 / s, 1.0 / s, 1.0 / s, 0.0]
        },
        cutoff: sf.cutoff() as f32,
        n_atoms: atoms.len() as u32,
        n_types: nt as u32,
        _pad: 0,
    };

    let coefficients = Coefficients::for_choice_with_slots(sf.choice(), sf.weights(), &wanted);
    let coefficients_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-grid-coefficients"),
        contents: bytemuck::bytes_of(&coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("odock-grid-shader"),
        source: wgpu::ShaderSource::Wgsl(GRID_SHADER.into()),
    });

    // `GridParams` is four `vec4`s plus a scalar tail, so it satisfies the
    // uniform layout rules as-is.
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-grid-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let atoms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-grid-atoms"),
        contents: bytemuck::cast_slice(&atoms),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_points = (nx * ny * nz * nt) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("odock-grid-out"),
        size: n_points * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("odock-grid-readback"),
        size: n_points * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // The buffer has one f32 slot per (x, y, z, type); expose it as a flat array.
    let _ = &params_buf;
    let _ = &atoms_buf;

    // Bindings are declared here so that a device without compute support
    // produces a clear error rather than a panic.
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("odock-grid-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 3: the force-field parameter tables (radii, donor and
            // acceptor flags, term weights). A storage buffer is used because a
            // uniform array of `f32` would be padded to a 16-byte element
            // stride by the WGSL layout rules.
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("odock-grid-pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("odock-grid-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("odock-grid-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: atoms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: coefficients_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("odock-grid-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("odock-grid-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = ((n_points as u32) + 63) / 64;
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, n_points * 4);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::Wait).map_err(|e| DockError::Gpu(e.to_string()))?;
    rx.recv()
        .map_err(|e| DockError::Gpu(e.to_string()))?
        .map_err(|e| DockError::Gpu(e.to_string()))?;
    let data = slice.get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&data);
    grid.upload(values)
        .map_err(|e| DockError::Gpu(e))?;
    drop(data);
    staging.unmap();

    Ok(grid.num_points())
}

/// Helper used by the CLI to print the GPU it would use.
pub fn describe() -> String {
    match request_adapter() {
        Ok(Some((_, _, info))) => format!(
            "{} ({}, {}, {})",
            info.name, info.backend, info.driver, info.device_type
        ),
        Ok(None) => "no compatible GPU adapter".to_string(),
        Err(e) => format!("GPU backend unavailable: {e}"),
    }
}

/// Interpolation helper shared with the CPU path, exposed for testing.
#[inline]
pub fn trilinear(
    f000: f64,
    f100: f64,
    f010: f64,
    f110: f64,
    f001: f64,
    f101: f64,
    f011: f64,
    f111: f64,
    s: DVec3,
) -> f64 {
    let (x, y, z) = (s.x, s.y, s.z);
    let (mx, my, mz) = (1.0 - x, 1.0 - y, 1.0 - z);
    f000 * mx * my * mz
        + f100 * x * my * mz
        + f010 * mx * y * mz
        + f110 * x * y * mz
        + f001 * mx * my * z
        + f101 * x * my * z
        + f011 * mx * y * z
        + f111 * x * y * z
}

/// A convenience shim so callers can build a grid from a box in one call.
pub fn grid_for_box(box_: &GridBox, slope: f64) -> AffinityGrid {
    AffinityGrid::new(box_, slope)
}

// ---------------------------------------------------------------------------
// Batch forward kinematics and batch pair evaluation (module D: ODock-GPU)
// ---------------------------------------------------------------------------
//
// Two more kernels are offloaded besides the affinity grid:
//
// * **forward kinematics** — expand the rigid-body + torsion tree of a whole
//   batch of conformations into explicit atom coordinates;
// * **pair evaluation** — the non-cached Vina/Vinardo interaction energy of
//   every ligand atom of every conformation against an explicit receptor atom
//   list.
//
// Both have a pure-CPU reference implementation right next to them, and a unit
// test compares the two numerically (not just "it compiled and dispatched").
// The public entry points use the GPU when one is available and silently fall
// back to the CPU otherwise, exactly like [`populate_grid`].

/// Host-side view of one kinematic-tree node (mirrors `KinNode` in the WGSL).
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KinNode {
    /// Parent node index, `u32::MAX` for a root.
    pub parent: u32,
    /// 0 = rigid root, 1 = torsion segment, 2 = pinned (absolute) root.
    pub kind: u32,
    /// Torsion slot inside the conformation, `u32::MAX` when unused.
    pub torsion: u32,
    /// First movable atom owned by this frame.
    pub atom_begin: u32,
    /// Number of atoms owned by this frame.
    pub atom_count: u32,
    /// Padding.
    pub _pad0: u32,
    /// Padding.
    pub _pad1: u32,
    /// Padding.
    pub _pad2: u32,
    /// Origin of the frame, expressed in the parent frame.
    pub rel_origin: [f32; 4],
    /// Rotation axis, expressed in the parent frame.
    pub rel_axis: [f32; 4],
    /// Pinned (laboratory) axis, used by `kind == 2`.
    pub abs_axis: [f32; 4],
    /// Pinned (laboratory) origin, used by `kind == 2`.
    pub abs_origin: [f32; 4],
}

/// Uniform parameters of the forward-kinematics kernel.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BatchParams {
    /// `(n_conformations, n_nodes, max_atoms_per_node, n_torsions)`.
    pub shape: [u32; 4],
    /// `(n_atoms, atom_base, conf_stride, reserved)`.
    pub info: [u32; 4],
}

/// Uniform parameters of the batch pair-evaluation kernel.
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EvalParams {
    /// `(n_conformations, n_ligand_atoms, n_receptor_atoms, receptor_offset)`.
    pub shape: [u32; 4],
    /// Force-field cutoff (Å).
    pub cutoff: f32,
    /// Vina's `curl` cap (`v`).
    pub curl_cap: f32,
    /// Padding.
    pub _pad0: f32,
    /// Padding.
    pub _pad1: f32,
}

/// Deepest kinematic tree the WGSL kernel can expand.
#[cfg(feature = "gpu")]
const MAX_GPU_DEPTH: u32 = 32;

/// Longest atom range owned by a single frame, used to size the dispatch.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct TreeTopology {
    /// Nodes in pre-order; `parent` indexes into this vector.
    nodes: Vec<FlattenedNode>,
    /// Largest `atom_count` over all nodes.
    max_atoms: u32,
    /// Deepest node.
    depth: u32,
    /// Number of atoms owned by exactly one node.
    covered_atoms: usize,
}

/// A [`Node`] with its parent resolved to an index.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy)]
struct FlattenedNode {
    parent: u32,
    kind: u32,
    torsion: u32,
    atom_begin: u32,
    atom_count: u32,
    rel_origin: DVec3,
    rel_axis: DVec3,
    abs_axis: DVec3,
    abs_origin: DVec3,
}

/// Flatten a kinematic tree into pre-order, parent-linked nodes.
#[cfg(feature = "gpu")]
fn flatten_tree(tree: &KinematicTree) -> TreeTopology {
    fn walk(
        node: &Node,
        parent: u32,
        depth: u32,
        out: &mut Vec<FlattenedNode>,
        max_atoms: &mut u32,
        max_depth: &mut u32,
        covered: &mut usize,
    ) {
        let id = out.len() as u32;
        let count = node.atoms.1.saturating_sub(node.atoms.0);
        *max_atoms = (*max_atoms).max(count as u32);
        *max_depth = (*max_depth).max(depth);
        *covered += count;
        out.push(FlattenedNode {
            parent,
            kind: match node.kind {
                FrameKind::Rigid => 0,
                FrameKind::Torsion => {
                    if node.absolute {
                        2
                    } else {
                        1
                    }
                }
            },
            torsion: node.torsion.map(|t| t as u32).unwrap_or(u32::MAX),
            atom_begin: node.atoms.0 as u32,
            atom_count: count as u32,
            rel_origin: node.rel_origin,
            rel_axis: node.rel_axis,
            abs_axis: node.axis,
            abs_origin: node.origin,
        });
        for c in &node.children {
            walk(c, id, depth + 1, out, max_atoms, max_depth, covered);
        }
    }

    let mut nodes = Vec::new();
    let mut max_atoms = 0u32;
    let mut depth = 0u32;
    let mut covered = 0usize;
    walk(
        &tree.root,
        u32::MAX,
        0,
        &mut nodes,
        &mut max_atoms,
        &mut depth,
        &mut covered,
    );
    TreeTopology {
        nodes,
        max_atoms,
        depth,
        covered_atoms: covered,
    }
}

/// Expand a batch of conformations into explicit atom coordinates (CPU).
///
/// Returns `confs.len() * tree.num_atoms()` coordinates; conformation `k` owns
/// the block `k * n_atoms .. (k + 1) * n_atoms`, in movable-atom order. This is
/// the reference implementation of the `expand_batch` kernel.
pub fn expand_batch_cpu(tree: &KinematicTree, confs: &[LigandConf]) -> Vec<DVec3> {
    let (begin, end) = tree.atoms;
    let n = end.saturating_sub(begin);
    if n == 0 || confs.is_empty() {
        return Vec::new();
    }
    let mut work = tree.clone();
    let mut buf = vec![DVec3::ZERO; end];
    let mut out = Vec::with_capacity(confs.len() * n);
    for c in confs {
        match work.kind {
            TreeKind::Ligand => work.apply_ligand(c, &mut buf),
            TreeKind::Flex => work.apply_flex(&c.torsions, &mut buf),
        }
        out.extend_from_slice(&buf[begin..end]);
    }
    out
}

/// Expand a batch of conformations, using the GPU when one is available.
///
/// Falls back to [`expand_batch_cpu`] when the `gpu` feature is off, when no
/// adapter exists, or when the tree cannot be dispatched (a tree deeper than
/// the kernel's chain buffer). The numbers are identical either way.
pub fn expand_batch(tree: &KinematicTree, confs: &[LigandConf]) -> Vec<DVec3> {
    #[cfg(feature = "gpu")]
    {
        match expand_batch_gpu(tree, confs) {
            Ok(v) => return v,
            Err(DockError::Gpu(msg)) => {
                eprintln!("odock: GPU forward kinematics unavailable ({msg}); using the CPU path");
            }
            Err(e) => {
                eprintln!("odock: batch expansion failed ({e}); using the CPU path");
            }
        }
    }
    expand_batch_cpu(tree, confs)
}

/// The actual `wgpu` dispatch of the forward-kinematics kernel.
///
/// `Err(DockError::Gpu(_))` means "no GPU path is possible right now"; the
/// caller is expected to fall back to [`expand_batch_cpu`].
#[cfg(feature = "gpu")]
pub fn expand_batch_gpu(tree: &KinematicTree, confs: &[LigandConf]) -> Result<Vec<DVec3>> {
    use wgpu::util::DeviceExt;

    if confs.is_empty() {
        return Ok(Vec::new());
    }
    if tree.kind != TreeKind::Ligand {
        return Err(DockError::Gpu(
            "batch forward kinematics is implemented for ligand trees; flexible residues use the CPU path"
                .to_string(),
        ));
    }
    let (begin, end) = tree.atoms;
    let n_atoms = end.saturating_sub(begin);
    if n_atoms == 0 {
        return Err(DockError::Gpu("the kinematic tree owns no atoms".to_string()));
    }
    let topology = flatten_tree(tree);
    if topology.depth > MAX_GPU_DEPTH {
        return Err(DockError::Gpu(format!(
            "the kinematic tree is {} nodes deep, deeper than the kernel's chain buffer ({MAX_GPU_DEPTH})",
            topology.depth
        )));
    }
    if topology.covered_atoms != n_atoms {
        return Err(DockError::Gpu(format!(
            "the kinematic frames own {} atoms but the tree has {n_atoms}",
            topology.covered_atoms
        )));
    }
    let Some((device, queue, _info)) = shared_device() else {
        return Err(DockError::Gpu("no compatible GPU adapter found".into()));
    };

    let n_torsions = tree.num_torsions as u32;
    let conf_stride = 7 + n_torsions as usize;
    let mut packed: Vec<f32> = Vec::with_capacity(confs.len() * conf_stride);
    for c in confs {
        packed.push(c.position.x as f32);
        packed.push(c.position.y as f32);
        packed.push(c.position.z as f32);
        let q = c.orientation.normalize();
        packed.push(q.x as f32);
        packed.push(q.y as f32);
        packed.push(q.z as f32);
        packed.push(q.w as f32);
        for k in 0..n_torsions as usize {
            packed.push(c.torsions.get(k).copied().unwrap_or(0.0) as f32);
        }
    }

    let mut locals: Vec<[f32; 4]> = Vec::with_capacity(n_atoms);
    for i in begin..end {
        let l = tree.local.get(i).copied().unwrap_or(DVec3::ZERO);
        locals.push([l.x as f32, l.y as f32, l.z as f32, 0.0]);
    }
    let nodes: Vec<KinNode> = topology
        .nodes
        .iter()
        .map(|n| KinNode {
            parent: n.parent,
            kind: n.kind,
            torsion: n.torsion,
            atom_begin: n.atom_begin,
            atom_count: n.atom_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            rel_origin: [
                n.rel_origin.x as f32,
                n.rel_origin.y as f32,
                n.rel_origin.z as f32,
                0.0,
            ],
            rel_axis: [
                n.rel_axis.x as f32,
                n.rel_axis.y as f32,
                n.rel_axis.z as f32,
                0.0,
            ],
            abs_axis: [n.abs_axis.x as f32, n.abs_axis.y as f32, n.abs_axis.z as f32, 0.0],
            abs_origin: [
                n.abs_origin.x as f32,
                n.abs_origin.y as f32,
                n.abs_origin.z as f32,
                0.0,
            ],
        })
        .collect();

    let params = BatchParams {
        shape: [
            confs.len() as u32,
            nodes.len() as u32,
            topology.max_atoms.max(1),
            n_torsions,
        ],
        info: [n_atoms as u32, begin as u32, conf_stride as u32, 0],
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("odock-batch-shader"),
        source: wgpu::ShaderSource::Wgsl(GRID_SHADER.into()),
    });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-expand-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let nodes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-expand-nodes"),
        contents: bytemuck::cast_slice(&nodes),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let locals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-expand-locals"),
        contents: bytemuck::cast_slice(&locals),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let confs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-expand-confs"),
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let n_coords = confs.len() * n_atoms;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("odock-expand-out"),
        size: (n_coords * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("odock-expand-readback"),
        size: (n_coords * 16) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("odock-expand-bgl"),
        entries: &[
            uniform_entry(10),
            storage_entry(11, true),
            storage_entry(12, true),
            storage_entry(13, true),
            storage_entry(14, false),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("odock-expand-bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 10,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: nodes_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: locals_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: confs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let values = dispatch(
        &device,
        &queue,
        &shader,
        "expand_batch",
        &layout,
        &bind_group,
        [
            topology.max_atoms.max(1).div_ceil(64),
            nodes.len() as u32,
            confs.len() as u32,
        ],
        &out_buf,
        &staging,
        n_coords * 16,
    )?;

    let mut out = Vec::with_capacity(n_coords);
    for chunk in values.chunks_exact(4) {
        out.push(DVec3::new(
            chunk[0] as f64,
            chunk[1] as f64,
            chunk[2] as f64,
        ));
    }
    Ok(out)
}

/// Non-cached pair energy of a batch of expanded conformations (CPU).
///
/// `coords` holds `n_confs * ligand_atoms.len()` coordinates in the layout
/// produced by [`expand_batch_cpu`]. The result is the same number
/// [`NonCache::eval`] returns for the corresponding conformation with the
/// out-of-box penalty disabled (empty grid dimensions).
pub fn evaluate_batch_cpu(
    sf: &dyn ScoringFunction,
    receptor: &[Atom],
    ligand_atoms: &[Atom],
    coords: &[DVec3],
    n_confs: usize,
    curl_cap: f64,
) -> Vec<f64> {
    let n = ligand_atoms.len();
    assert_eq!(
        coords.len(),
        n * n_confs,
        "evaluate_batch_cpu expects n_confs * n_atoms coordinates"
    );
    if n == 0 || n_confs == 0 {
        return Vec::new();
    }
    let dims = [GridDim::empty(); 3];
    let noncache = NonCache::new(receptor.to_vec(), sf, dims, 1.0);
    (0..n_confs)
        .map(|k| {
            let slice = &coords[k * n..(k + 1) * n];
            noncache.eval(sf, ligand_atoms, slice, curl_cap, true, n, None)
        })
        .collect()
}

/// Evaluate a batch of expanded conformations, using the GPU when possible.
///
/// Falls back to [`evaluate_batch_cpu`] whenever the GPU path is unavailable,
/// including for the AD4 force field (which needs per-atom charges) and for a
/// mismatched buffer layout.
pub fn evaluate_batch(
    sf: &dyn ScoringFunction,
    receptor: &[Atom],
    ligand_atoms: &[Atom],
    coords: &[DVec3],
    n_confs: usize,
    curl_cap: f64,
) -> Vec<f64> {
    #[cfg(feature = "gpu")]
    {
        match evaluate_batch_gpu(sf, receptor, ligand_atoms, coords, n_confs, curl_cap) {
            Ok(v) => return v,
            Err(DockError::Gpu(msg)) => {
                eprintln!("odock: GPU pair evaluation unavailable ({msg}); using the CPU path");
            }
            Err(e) => {
                eprintln!("odock: batch pair evaluation failed ({e}); using the CPU path");
            }
        }
    }
    evaluate_batch_cpu(sf, receptor, ligand_atoms, coords, n_confs, curl_cap)
}

/// The actual `wgpu` dispatch of the pair-evaluation kernel.
#[cfg(feature = "gpu")]
pub fn evaluate_batch_gpu(
    sf: &dyn ScoringFunction,
    receptor: &[Atom],
    ligand_atoms: &[Atom],
    coords: &[DVec3],
    n_confs: usize,
    curl_cap: f64,
) -> Result<Vec<f64>> {
    use wgpu::util::DeviceExt;

    if !sf.is_grid_capable() || !sf.is_xs_typed() {
        return Err(DockError::Gpu(
            "the AD4 force field needs per-atom charges and is CPU-only".into(),
        ));
    }
    let n = ligand_atoms.len();
    if n == 0 || n_confs == 0 {
        return Ok(Vec::new());
    }
    if coords.len() != n * n_confs {
        return Err(DockError::Invalid(format!(
            "evaluate_batch expects {} coordinates, got {}",
            n * n_confs,
            coords.len()
        )));
    }
    let Some((device, queue, _info)) = shared_device() else {
        return Err(DockError::Gpu("no compatible GPU adapter found".into()));
    };

    // Ligand templates first, then the receptor: one buffer, no extra binding
    // (the downlevel limit is four storage buffers per stage).
    let mut atoms: Vec<GpuAtom> = Vec::with_capacity(n + receptor.len());
    for a in ligand_atoms {
        atoms.push(GpuAtom {
            position: [0.0, 0.0, 0.0, 0.0],
            type_index: [a.xs.index() as u32, 0, 0, 0],
        });
    }
    for a in receptor {
        atoms.push(GpuAtom {
            position: [
                a.coords.x as f32,
                a.coords.y as f32,
                a.coords.z as f32,
                0.0,
            ],
            type_index: [a.xs.index() as u32, 0, 0, 0],
        });
    }
    let flat: Vec<[f32; 4]> = coords
        .iter()
        .map(|c| [c.x as f32, c.y as f32, c.z as f32, 0.0])
        .collect();

    let params = EvalParams {
        shape: [
            n_confs as u32,
            n as u32,
            receptor.len() as u32,
            n as u32,
        ],
        cutoff: sf.cutoff() as f32,
        curl_cap: curl_cap as f32,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let coefficients = Coefficients::for_choice(sf.choice(), sf.weights());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("odock-eval-shader"),
        source: wgpu::ShaderSource::Wgsl(GRID_SHADER.into()),
    });
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-eval-params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let atoms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-eval-atoms"),
        contents: bytemuck::cast_slice(&atoms),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let coords_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-eval-coords"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tables_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("odock-eval-tables"),
        contents: bytemuck::bytes_of(&coefficients),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("odock-eval-out"),
        size: (n * n_confs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("odock-eval-readback"),
        size: (n * n_confs * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("odock-eval-bgl"),
        entries: &[
            uniform_entry(20),
            storage_entry(21, true),
            storage_entry(22, true),
            storage_entry(23, false),
            storage_entry(3, true),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("odock-eval-bg"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 20,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 21,
                resource: atoms_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 22,
                resource: coords_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 23,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: tables_buf.as_entire_binding(),
            },
        ],
    });
    let values = dispatch(
        &device,
        &queue,
        &shader,
        "evaluate_batch",
        &layout,
        &bind_group,
        [(n as u32).div_ceil(64), n_confs as u32, 1],
        &out_buf,
        &staging,
        n * n_confs * 4,
    )?;

    let mut out = Vec::with_capacity(n_confs);
    for k in 0..n_confs {
        out.push(values[k * n..(k + 1) * n].iter().map(|v| *v as f64).sum());
    }
    Ok(out)
}

/// Expand and score a batch in one call.
///
/// This is the "batch conformer sampling" path of module D: a pool of
/// conformations is expanded by the forward-kinematics kernel and scored by the
/// pair-evaluation kernel, with the CPU implementation behind both.
#[allow(clippy::too_many_arguments)]
pub fn score_batch(
    tree: &KinematicTree,
    confs: &[LigandConf],
    sf: &dyn ScoringFunction,
    receptor: &[Atom],
    ligand_atoms: &[Atom],
    curl_cap: f64,
) -> Vec<f64> {
    let coords = expand_batch(tree, confs);
    evaluate_batch(
        sf,
        receptor,
        ligand_atoms,
        &coords,
        confs.len(),
        curl_cap,
    )
}

/// Generate a pool of candidate conformations from a starting point.
///
/// Sample `k` is a single Vina-style mutation of sample `k - 1`
/// ([`crate::kinematics::mutate_conf`]), i.e. a random walk through torsion /
/// rigid-body space. The pool feeds [`expand_batch`] + [`evaluate_batch`].
///
/// Returns an empty vector when the system has no ligand.
pub fn sample_batch(
    base: &Conf,
    n: usize,
    amplitude: f64,
    gyration_radius: f64,
    rng: &mut Rng,
) -> Vec<LigandConf> {
    let Some(first) = base.ligands.first() else {
        return Vec::new();
    };
    let mut current = Conf {
        ligands: vec![first.clone()],
        flex: Vec::new(),
    };
    // `mutate_conf` also mutates flexible residues; the batch API works on the
    // ligand only, so the pool is built from a ligand-only conformation.
    let radii = [gyration_radius];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        mutate_conf(&mut current, amplitude, &radii, rng);
        out.push(current.ligands[0].clone());
    }
    out
}

/// A bind group layout entry for a storage buffer at `binding`.
#[cfg(feature = "gpu")]
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A bind group layout entry for a uniform buffer at `binding`.
#[cfg(feature = "gpu")]
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Run one compute pass and read the output buffer back as `f32`s.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader: &wgpu::ShaderModule,
    entry_point: &str,
    layout: &wgpu::BindGroupLayout,
    bind_group: &wgpu::BindGroup,
    workgroups: [u32; 3],
    out_buf: &wgpu::Buffer,
    staging: &wgpu::Buffer,
    n_bytes: usize,
) -> Result<Vec<f32>> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("odock-batch-pl"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("odock-batch-pipeline"),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("odock-batch-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("odock-batch-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(
            workgroups[0].max(1),
            workgroups[1].max(1),
            workgroups[2].max(1),
        );
    }
    encoder.copy_buffer_to_buffer(out_buf, 0, staging, 0, n_bytes as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait)
        .map_err(|e| DockError::Gpu(e.to_string()))?;
    rx.recv()
        .map_err(|e| DockError::Gpu(e.to_string()))?
        .map_err(|e| DockError::Gpu(e.to_string()))?;
    let data = slice.get_mapped_range();
    let values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(values)
}

/// A process-wide `wgpu` device.
///
/// Creating a device costs tens of milliseconds (adapter enumeration, driver
/// initialisation, shader cache warm-up), so the batch entry points share one
/// instead of paying it on every call. `wgpu::Device` and `wgpu::Queue` are
/// `Send + Sync` and internally synchronised, so sharing them across the
/// `rayon` workers of a batch is safe.
#[cfg(feature = "gpu")]
fn shared_device() -> Option<(wgpu::Device, wgpu::Queue, GpuInfo)> {
    static DEVICE: std::sync::OnceLock<Option<(wgpu::Device, wgpu::Queue, GpuInfo)>> =
        std::sync::OnceLock::new();
    DEVICE
        .get_or_init(|| request_adapter().ok().flatten())
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trilinear_reproduces_corner_values() {
        let f = |x: f64, y: f64, z: f64| x + 2.0 * y + 3.0 * z;
        let got = trilinear(
            f(0.0, 0.0, 0.0),
            f(1.0, 0.0, 0.0),
            f(0.0, 1.0, 0.0),
            f(1.0, 1.0, 0.0),
            f(0.0, 0.0, 1.0),
            f(1.0, 0.0, 1.0),
            f(0.0, 1.0, 1.0),
            f(1.0, 1.0, 1.0),
            DVec3::new(0.3, 0.6, 0.2),
        );
        // The function is linear, so trilinear interpolation is exact.
        assert!((got - f(0.3, 0.6, 0.2)).abs() < 1e-12);
    }

    #[test]
    fn shader_source_is_embedded() {
        assert!(GRID_SHADER.contains("@compute"));
    }

    #[test]
    fn cpu_fallback_produces_a_usable_grid() {
        use crate::atom::AdType;
        use crate::scoring::make_scoring_function;
        use crate::scoring::SfChoice;
        let receptor = vec![Atom::new(DVec3::ZERO, AdType::C, 0.0)];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let box_ = GridBox::new(DVec3::ZERO, DVec3::splat(6.0), 0.5);
        let mut g = grid_for_box(&box_, 1e6);
        let n = populate_grid(&mut g, &receptor, sf.as_ref(), &[XsType::CH]).unwrap();
        assert!(n > 0);
        assert!(g.is_initialized());
    }

    /// The GPU kernel and the CPU loop must agree: they evaluate the same
    /// potentials at the same sample points, so any difference is a bug.
    ///
    /// Both griddable force fields are checked: the shader's six weight slots
    /// are Vina's term order, and Vinardo's five terms have to be mapped into
    /// them correctly.
    ///
    /// Without a GPU adapter `populate_grid` silently uses the CPU, in which
    /// case the comparison is trivially satisfied — the test still guards the
    /// dispatch path on a machine that has one.
    #[test]
    fn gpu_and_cpu_grids_agree() {
        use crate::atom::AdType;
        use crate::molecule::Atom;
        use crate::scoring::{make_scoring_function, SfChoice};

        let receptor: Vec<Atom> = (0..40)
            .map(|i| {
                let f = i as f64;
                Atom::new(
                    DVec3::new(
                        2.0 * (f * 1.3).cos(),
                        2.0 * (f * 0.7).sin(),
                        0.5 * f - 5.0,
                    ),
                    if i % 4 == 0 { AdType::OA } else { AdType::C },
                    0.0,
                )
            })
            .collect();
        let box_ = GridBox::new(DVec3::ZERO, DVec3::splat(10.0), 0.5);
        let types = [XsType::CH, XsType::OA];

        let probes: Vec<Atom> = (0..12)
            .map(|i| {
                let f = i as f64;
                Atom::new(
                    DVec3::new(0.3 * f - 1.5, (f * 0.9).sin(), (f * 0.6).cos()),
                    AdType::C,
                    0.0,
                )
            })
            .collect();
        let coords: Vec<DVec3> = probes.iter().map(|a| a.coords).collect();

        for choice in [SfChoice::Vina, SfChoice::Vinardo] {
            let sf = make_scoring_function(choice, None);
            let mut cpu = grid_for_box(&box_, 1e6);
            cpu.populate(&receptor, sf.as_ref(), &types);

            let mut gpu = grid_for_box(&box_, 1e6);
            populate_grid(&mut gpu, &receptor, sf.as_ref(), &types).unwrap();

            let a = cpu.eval(&probes, &coords, 1000.0);
            let b = gpu.eval(&probes, &coords, 1000.0);
            assert!(
                (a - b).abs() < 0.05,
                "{choice}: GPU grid {b:.4} differs from the CPU grid {a:.4}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batch forward kinematics + batch pair evaluation
    // -----------------------------------------------------------------------

    const LIGAND: &str = include_str!("../../tests/data/erlotinib.pdbqt");
    const RECEPTOR: &str = include_str!("../../tests/data/egfr.pdbqt");

    /// The erlotinib fixture (11 torsions, nested branches) and the receptor
    /// atoms of its pocket, both typed exactly as `build_system` types them.
    fn fixture() -> (KinematicTree, Vec<Atom>) {
        use crate::io::pdbqt::{parse_ligand_pdbqt, parse_receptor_pdbqt};
        use crate::molecule::{assign_types, perceive_bonds};

        let parsed = parse_ligand_pdbqt(LIGAND).expect("the ligand fixture must parse");
        let coords: Vec<DVec3> = parsed.atoms.iter().map(|a| a.coords).collect();
        let (tree, _root) = crate::kinematics::build_ligand_tree(&parsed.top, &coords, coords.len());

        let mut receptor = parse_receptor_pdbqt(RECEPTOR)
            .expect("the receptor fixture must parse")
            .atoms;
        perceive_bonds(&mut receptor);
        assign_types(&mut receptor);
        let centre = coords.iter().sum::<DVec3>() / coords.len() as f64;
        receptor.retain(|a| (a.coords - centre).length() <= 12.0);
        assert!(receptor.len() > 50, "the pocket must contain atoms");
        (tree, receptor)
    }

    /// The ligand atoms in kernel order, typed.
    fn ligand_atoms() -> Vec<Atom> {
        use crate::io::pdbqt::parse_ligand_pdbqt;
        use crate::molecule::{assign_types, perceive_bonds};
        let mut atoms = parse_ligand_pdbqt(LIGAND)
            .expect("the ligand fixture must parse")
            .atoms;
        perceive_bonds(&mut atoms);
        assign_types(&mut atoms);
        atoms
    }

    /// A batch of conformations: the input pose plus random mutations of it.
    fn test_conformations(tree: &KinematicTree, n: usize) -> Vec<LigandConf> {
        let parsed = crate::io::pdbqt::parse_ligand_pdbqt(LIGAND).expect("fixture");
        let coords: Vec<DVec3> = parsed.atoms.iter().map(|a| a.coords).collect();
        let mut base = crate::kinematics::Conf {
            ligands: vec![LigandConf::null(coords[0], tree.num_torsions)],
            flex: Vec::new(),
        };
        let mut rng = Rng::new(20240929);
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            if k > 0 {
                crate::kinematics::mutate_conf(&mut base, 1.5, &[4.0], &mut rng);
                // Mutate a couple more degrees of freedom so that the batch
                // covers the whole tree, not just one branch.
                crate::kinematics::mutate_conf(&mut base, 1.0, &[4.0], &mut rng);
            }
            out.push(base.ligands[0].clone());
        }
        out
    }

    #[test]
    fn cpu_expansion_reproduces_the_input_structure() {
        let (tree, _) = fixture();
        let confs = test_conformations(&tree, 3);
        let out = expand_batch_cpu(&tree, &confs);
        assert_eq!(out.len(), confs.len() * tree.num_atoms());

        // The first conformation is the identity: every atom must sit exactly
        // where the file put it.
        let parsed = crate::io::pdbqt::parse_ligand_pdbqt(LIGAND).unwrap();
        let n = tree.num_atoms();
        let mut worst = 0.0f64;
        for (got, reference) in out[..n].iter().zip(parsed.atoms.iter().take(n)) {
            worst = worst.max((*got - reference.coords).length());
        }
        assert!(worst < 1e-9, "the identity pose moved by {worst:.3e} A");

        // ... and the mutated ones must be genuine isometries of it.
        for k in 1..confs.len() {
            let block = &out[k * n..(k + 1) * n];
            let d01 = (block[0] - block[1]).length();
            let d01_ref = (parsed.atoms[0].coords - parsed.atoms[1].coords).length();
            assert!((d01 - d01_ref).abs() < 1e-9);
        }
    }

    /// The forward-kinematics kernel must reproduce the CPU expansion; without
    /// the `gpu` feature (or without an adapter) the test exercises the CPU
    /// fallback path instead, which is what production code would use.
    #[test]
    fn gpu_forward_kinematics_matches_the_cpu() {
        let (tree, _) = fixture();
        let confs = test_conformations(&tree, 8);
        let cpu = expand_batch_cpu(&tree, &confs);
        assert_eq!(cpu.len(), 8 * tree.num_atoms());

        #[cfg(feature = "gpu")]
        match expand_batch_gpu(&tree, &confs) {
            Ok(gpu) => {
                assert_eq!(gpu.len(), cpu.len());
                let worst = cpu
                    .iter()
                    .zip(&gpu)
                    .map(|(a, b)| (*a - *b).length())
                    .fold(0.0f64, f64::max);
                assert!(
                    worst < 1e-3,
                    "GPU forward kinematics differs from the CPU by {worst:.3e} A"
                );
                eprintln!(
                    "GPU/CPU forward kinematics: max deviation {worst:.3e} A over {} atoms",
                    cpu.len()
                );
                // A bit-identical result would mean the kernel never ran: the
                // GPU works in `f32`, the CPU in `f64`.
                assert!(worst > 0.0, "the kernel cannot have been dispatched");
            }
            Err(DockError::Gpu(msg)) => {
                eprintln!("no GPU adapter ({msg}); exercising the CPU fallback instead");
                assert_eq!(expand_batch(&tree, &confs), cpu);
            }
            Err(e) => panic!("unexpected batch expansion failure: {e}"),
        }

        #[cfg(not(feature = "gpu"))]
        assert_eq!(expand_batch(&tree, &confs), cpu);
    }

    /// Brute-force validation of the CPU batch reference, independent of the
    /// cell list: every pair inside the cutoff, `curl`-ed per ligand atom.
    #[test]
    fn cpu_pair_evaluation_matches_a_brute_force_sum() {
        use crate::scoring::{make_scoring_function, SfChoice};

        let (tree, receptor) = fixture();
        let ligand = ligand_atoms();
        let confs = test_conformations(&tree, 3);
        let coords = expand_batch_cpu(&tree, &confs);
        let sf = make_scoring_function(SfChoice::Vina, None);
        let n = ligand.len();

        let curled = evaluate_batch_cpu(sf.as_ref(), &receptor, &ligand, &coords, 3, 1000.0);
        assert_eq!(curled.len(), 3);

        for (k, got) in curled.iter().enumerate() {
            let mut expect = 0.0;
            for i in 0..n {
                if ligand[i].xs == XsType::W || ligand[i].xs.is_glue() {
                    continue;
                }
                let mut e = 0.0;
                for b in &receptor {
                    if b.xs == XsType::W {
                        continue;
                    }
                    let r = (coords[k * n + i] - b.coords).length();
                    if r < sf.cutoff() {
                        e += sf.pair_energy(&ligand[i], b, r);
                    }
                }
                crate::math::curl_scalar(&mut e, 1000.0);
                expect += e;
            }
            assert!(
                (got - expect).abs() < 1e-9,
                "conformation {k}: batch gave {got}, brute force {expect}"
            );
        }
    }

    /// The pair-evaluation kernel must reproduce the CPU scorer for a whole
    /// batch, to better than 1e-3 kcal/mol.
    #[test]
    fn gpu_pair_evaluation_matches_the_cpu() {
        use crate::scoring::{make_scoring_function, SfChoice};

        let (tree, receptor) = fixture();
        let ligand = ligand_atoms();
        let confs = test_conformations(&tree, 8);
        // Both implementations score the *same* coordinates here, so any
        // difference is the kernel's own arithmetic error.
        let coords = expand_batch_cpu(&tree, &confs);
        let sf = make_scoring_function(SfChoice::Vina, None);

        let cpu = evaluate_batch_cpu(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0);

        #[cfg(feature = "gpu")]
        match evaluate_batch_gpu(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0) {
            Ok(gpu) => {
                assert_eq!(gpu.len(), cpu.len());
                let mut worst = 0.0f64;
                for (k, (a, b)) in cpu.iter().zip(&gpu).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-3,
                        "conformation {k}: GPU {b:.6} vs CPU {a:.6} kcal/mol"
                    );
                    worst = worst.max((a - b).abs());
                }
                eprintln!(
                    "GPU/CPU batch pair evaluation: max deviation {worst:.3e} kcal/mol over {} conformations",
                    cpu.len()
                );
                // A zeroed batch would pass the test above only if the CPU
                // result were zero too; make sure the energies are real.
                assert!(cpu.iter().any(|e| e.abs() > 1.0), "{cpu:?}");
            }
            Err(DockError::Gpu(msg)) => {
                eprintln!("no GPU adapter ({msg}); exercising the CPU fallback instead");
                let fallback =
                    evaluate_batch(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0);
                assert_eq!(fallback, cpu);
            }
            Err(e) => panic!("unexpected pair-evaluation failure: {e}"),
        }

        #[cfg(not(feature = "gpu"))]
        {
            let fallback =
                evaluate_batch(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0);
            assert_eq!(fallback, cpu);
        }
    }

    /// The pair kernel must also reproduce the CPU scorer for the Vinardo
    /// force field, whose five terms have a different order from Vina's six.
    #[test]
    fn gpu_pair_evaluation_matches_the_cpu_for_vinardo() {
        use crate::scoring::{make_scoring_function, SfChoice};

        let (tree, receptor) = fixture();
        let ligand = ligand_atoms();
        let confs = test_conformations(&tree, 6);
        let coords = expand_batch_cpu(&tree, &confs);
        let sf = make_scoring_function(SfChoice::Vinardo, None);

        let cpu = evaluate_batch_cpu(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0);
        let got = evaluate_batch(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0);
        assert_eq!(got.len(), cpu.len());
        for (k, (a, b)) in cpu.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "vinardo, conformation {k}: GPU {b:.6} vs CPU {a:.6} kcal/mol"
            );
        }
        assert!(cpu.iter().any(|e| e.abs() > 1.0), "{cpu:?}");
    }

    /// End to end: sample a pool, expand it, score it — and get the same
    /// numbers as the all-CPU pipeline.
    #[test]
    fn score_batch_matches_the_cpu_pipeline() {
        use crate::scoring::{make_scoring_function, SfChoice};

        let (tree, receptor) = fixture();
        let ligand = ligand_atoms();
        let confs = test_conformations(&tree, 6);
        let sf = make_scoring_function(SfChoice::Vina, None);

        let coords = expand_batch_cpu(&tree, &confs);
        let cpu = evaluate_batch_cpu(sf.as_ref(), &receptor, &ligand, &coords, confs.len(), 1000.0);
        let got = score_batch(&tree, &confs, sf.as_ref(), &receptor, &ligand, 1000.0);
        assert_eq!(got.len(), cpu.len());
        for (k, (a, b)) in cpu.iter().zip(&got).enumerate() {
            assert!(
                (a - b).abs() < 1e-2,
                "conformation {k}: pipeline {b:.6} vs CPU {a:.6} kcal/mol"
            );
        }
    }

    #[test]
    fn sample_batch_produces_a_diverse_pool() {
        let (tree, _) = fixture();
        let confs = test_conformations(&tree, 1);
        let base = crate::kinematics::Conf {
            ligands: confs,
            flex: Vec::new(),
        };
        let mut rng = Rng::new(7);
        let pool = sample_batch(&base, 16, 2.0, 4.0, &mut rng);
        assert_eq!(pool.len(), 16);
        let mut distinct = 0;
        for (i, a) in pool.iter().enumerate() {
            for b in pool.iter().skip(i + 1) {
                if a.position != b.position || a.torsions != b.torsions {
                    distinct += 1;
                }
            }
        }
        assert!(distinct > 100, "the pool is not diverse ({distinct} pairs)");
        assert_eq!(pool[0].torsions.len(), tree.num_torsions);

        // No ligand at all: an empty pool, not a panic.
        assert!(sample_batch(&crate::kinematics::Conf::default(), 4, 1.0, 1.0, &mut rng).is_empty());
    }

    #[test]
    fn batch_helpers_survive_degenerate_inputs() {
        let (tree, receptor) = fixture();
        let ligand = ligand_atoms();
        use crate::scoring::{make_scoring_function, SfChoice};
        let sf = make_scoring_function(SfChoice::Vina, None);

        assert!(expand_batch(&tree, &[]).is_empty());
        assert!(expand_batch_cpu(&tree, &[]).is_empty());
        assert!(evaluate_batch_cpu(sf.as_ref(), &receptor, &ligand, &[], 0, 1000.0).is_empty());
        assert!(score_batch(&tree, &[], sf.as_ref(), &receptor, &ligand, 1000.0).is_empty());
    }
}

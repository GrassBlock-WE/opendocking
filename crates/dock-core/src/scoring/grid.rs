// SPDX-License-Identifier: GPL-3.0-or-later
//! The affinity grid: a trilinearly interpolated 3-D lookup of the receptor
//! interaction energy.
//!
//! # Why a grid
//!
//! The docking search evaluates the scoring function millions of times. The
//! exact pairwise evaluation costs `O(n_ligand 路 n_neighbours)` `exp()` calls
//! per conformation; a pre-computed grid reduces that to eight array reads and
//! one trilinear blend per ligand atom, roughly two orders of magnitude faster.
//! Local refinement then re-scores with the exact
//! [`crate::scoring::noncache::NonCache`] so the reported energies are not
//! interpolation artefacts.
//!
//! # Interpolation
//!
//! For a probe at fractional cell coordinates `(x, y, z) 鈭?[0, 1]鲁`, the eight
//! corner values `f_ijk` give
//!
//! ```text
//! f = 危_ijk f_ijk 路 w_i(x) 路 w_j(y) 路 w_k(z),   w_0(t) = 1 - t,  w_1(t) = t
//! ```
//!
//! and the analytic gradient of that blend is
//!
//! ```text
//! 鈭俧/鈭倄 = (危_ijk f_ijk 路 (2i-1) 路 w_j(y) 路 w_k(z)) 路 factor_x
//! ```
//!
//! which is exactly AutoDock Vina's `grid::evaluate_aux` (`grid.cpp`,
//! Apache-2.0). The gradient is therefore the exact derivative of the
//! interpolated energy, keeping the BFGS step consistent.
//!
//! # Out-of-box penalty
//!
//! A probe outside the grid is clamped to the boundary cell and a linear
//! penalty `slope 路 distance` is added, with `slope = 1e6` by default. The
//! gradient gets the corresponding constant `slope 路 region`. This keeps the
//! search inside the user's box without constraining the optimiser.

use std::fmt;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::atom::XsType;
use crate::molecule::Atom;
use crate::math::{curl, curl_scalar, DVec3};
use crate::scoring::noncache::CellList;
use crate::scoring::ScoringFunction;

/// One axis of the search box.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GridDim {
    /// World coordinate of the first sample point.
    pub begin: f64,
    /// World coordinate of the last sample point.
    pub end: f64,
    /// Number of *voxels*; the number of sample points is `n_voxels + 1`.
    pub n_voxels: usize,
}

impl GridDim {
    /// A disabled axis (no voxels, no penalty).
    pub fn empty() -> GridDim {
        GridDim {
            begin: 0.0,
            end: 0.0,
            n_voxels: 0,
        }
    }

    /// Length of the axis in 脜.
    #[inline]
    pub fn span(&self) -> f64 {
        self.end - self.begin
    }

    /// Number of sample points along the axis.
    #[inline]
    pub fn n_points(&self) -> usize {
        self.n_voxels + 1
    }
}

/// The axis-aligned search box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridBox {
    /// Box centre (脜).
    pub center: DVec3,
    /// Requested edge lengths (脜).
    pub size: DVec3,
    /// Grid spacing (脜); 0.375 is the AutoDock convention.
    pub spacing: f64,
    /// Round voxel counts up to even numbers, as required by `.map` files.
    pub force_even_voxels: bool,
}

impl Default for GridBox {
    fn default() -> Self {
        GridBox {
            center: DVec3::ZERO,
            size: DVec3::splat(22.5),
            spacing: 0.375,
            force_even_voxels: false,
        }
    }
}

impl GridBox {
    /// A box with the given centre, edge lengths and spacing.
    pub fn new(center: DVec3, size: DVec3, spacing: f64) -> GridBox {
        GridBox {
            center,
            size,
            spacing,
            force_even_voxels: false,
        }
    }

    /// Request even voxel counts on every axis (required by `.map` output).
    pub fn with_even_voxels(mut self, even: bool) -> GridBox {
        self.force_even_voxels = even;
        self
    }

    /// Derive a box that contains a molecule of the given half-extents plus
    /// `buffer` 脜 of padding (Vina's `grid_dimensions_from_ligand`).
    pub fn around(center: DVec3, half_extents: DVec3, buffer: f64, spacing: f64) -> GridBox {
        let size = (half_extents + DVec3::splat(buffer)) * 2.0;
        GridBox {
            center,
            size: DVec3::new(size.x.ceil(), size.y.ceil(), size.z.ceil()),
            spacing,
            force_even_voxels: false,
        }
    }

    /// The three per-axis descriptions.
    pub fn dims(&self) -> [GridDim; 3] {
        let mut out = [GridDim::empty(); 3];
        for i in 0..3 {
            let span = self.size[i].max(1e-9);
            let mut n = (span / self.spacing).ceil() as usize;
            if n == 0 {
                n = 1;
            }
            if self.force_even_voxels && n % 2 == 1 {
                n += 1;
            }
            let real_span = self.spacing * n as f64;
            let begin = self.center[i] - real_span / 2.0;
            out[i] = GridDim {
                begin,
                end: begin + real_span,
                n_voxels: n,
            };
        }
        out
    }

    /// Lower corner of the (rounded) box.
    pub fn corner1(&self) -> DVec3 {
        let d = self.dims();
        DVec3::new(d[0].begin, d[1].begin, d[2].begin)
    }

    /// Upper corner of the (rounded) box.
    pub fn corner2(&self) -> DVec3 {
        let d = self.dims();
        DVec3::new(d[0].end, d[1].end, d[2].end)
    }

    /// Volume of the rounded box in 脜鲁.
    pub fn volume(&self) -> f64 {
        let d = self.dims();
        d[0].span() * d[1].span() * d[2].span()
    }
}

/// Map an atom type onto the grid slot it uses.
///
/// Macrocycle closure types collapse onto their parent carbon type and closure
/// dummy atoms have no grid at all 鈥?both rules are Vina's.
#[inline]
pub fn grid_type(t: XsType) -> Option<XsType> {
    match t {
        XsType::G0 | XsType::G1 | XsType::G2 | XsType::G3 | XsType::W => None,
        XsType::CHCg0 | XsType::CHCg1 | XsType::CHCg2 | XsType::CHCg3 => Some(XsType::CH),
        XsType::CPCg0 | XsType::CPCg1 | XsType::CPCg2 | XsType::CPCg3 => Some(XsType::CP),
        other => Some(other),
    }
}

/// A pre-computed receptor affinity grid.
#[derive(Clone)]
pub struct AffinityGrid {
    /// Per-axis geometry.
    pub dims: [GridDim; 3],
    /// Grid spacing actually used (脜).
    pub spacing: f64,
    /// Out-of-box penalty slope.
    pub slope: f64,
    /// Atom types stored, in slot order.
    pub types: Vec<XsType>,
    /// `slot[t]` is the storage slot of X-Score type `t`, or `usize::MAX`.
    slot: [usize; XsType::COUNT],
    /// Flat `[x][y][z][slot]` energy storage.
    data: Vec<f64>,
}

impl fmt::Debug for AffinityGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffinityGrid")
            .field(
                "voxels",
                &self.dims.iter().map(|d| d.n_voxels).collect::<Vec<_>>(),
            )
            .field("spacing", &self.spacing)
            .field("types", &self.types)
            .field("memory_mb", &self.memory_mb())
            .finish()
    }
}

impl AffinityGrid {
    /// Allocate an empty grid for the given box.
    pub fn new(box_: &GridBox, slope: f64) -> AffinityGrid {
        AffinityGrid::from_dims(box_.dims(), box_.spacing, slope)
    }

    /// Allocate an empty grid from explicit dimensions.
    pub fn from_dims(dims: [GridDim; 3], spacing: f64, slope: f64) -> AffinityGrid {
        AffinityGrid {
            dims,
            spacing,
            slope,
            types: Vec::new(),
            slot: [usize::MAX; XsType::COUNT],
            data: Vec::new(),
        }
    }

    /// `true` when at least one atom-type map has been built.
    pub fn is_initialized(&self) -> bool {
        !self.types.is_empty()
    }

    /// Number of sample points per axis.
    pub fn shape(&self) -> [usize; 3] {
        [
            self.dims[0].n_points(),
            self.dims[1].n_points(),
            self.dims[2].n_points(),
        ]
    }

    /// Total number of sample points.
    pub fn num_points(&self) -> usize {
        let s = self.shape();
        s[0] * s[1] * s[2]
    }

    /// Approximate memory footprint of the maps, in megabytes.
    pub fn memory_mb(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>() / 1_048_576
    }

    /// `true` when every requested type (after closure collapsing) is present.
    pub fn has_types(&self, types: &[XsType]) -> bool {
        types.iter().all(|t| match grid_type(*t) {
            Some(g) => self.slot[g.index()] != usize::MAX,
            None => true,
        })
    }

    /// The atom types present in the grid.
    pub fn available_types(&self) -> &[XsType] {
        &self.types
    }

    /// Build the maps for `types` from the receptor atoms.
    ///
    /// The work is parallelised over the first grid axis with `rayon`.
    /// Returns the number of sample points written.
    pub fn populate(
        &mut self,
        receptor: &[Atom],
        sf: &dyn ScoringFunction,
        types: &[XsType],
    ) -> usize {
        let mut wanted: Vec<XsType> = Vec::new();
        for t in types {
            if let Some(g) = grid_type(*t) {
                if !wanted.contains(&g) {
                    wanted.push(g);
                }
            }
        }
        if wanted.is_empty() {
            return 0;
        }

        self.slot = [usize::MAX; XsType::COUNT];
        for (i, t) in wanted.iter().enumerate() {
            self.slot[t.index()] = i;
        }
        self.types = wanted;

        let [nx, ny, nz] = self.shape();
        let nt = self.types.len();
        self.data = vec![0.0; nx * ny * nz * nt];

        let positions: Vec<DVec3> = receptor.iter().map(|a| a.coords).collect();
        let receptor_xs: Vec<XsType> = receptor.iter().map(|a| a.xs).collect();
        let cells = CellList::new(&positions, sf.cutoff().clamp(3.0, 8.0));
        let cutoff = sf.cutoff();
        let cutoff_sqr = cutoff * cutoff;

        let init = DVec3::new(self.dims[0].begin, self.dims[1].begin, self.dims[2].begin);
        let factors = DVec3::new(
            (nx - 1) as f64 / self.dims[0].span(),
            (ny - 1) as f64 / self.dims[1].span(),
            (nz - 1) as f64 / self.dims[2].span(),
        );
        let inv_factors = DVec3::new(1.0 / factors.x, 1.0 / factors.y, 1.0 / factors.z);

        let slab = ny * nz * nt;
        let types_ref: &[XsType] = &self.types;

        self.data
            .par_chunks_mut(slab)
            .enumerate()
            .for_each(|(x, chunk)| {
                let px = init.x + inv_factors.x * x as f64;
                for y in 0..ny {
                    let py = init.y + inv_factors.y * y as f64;
                    for z in 0..nz {
                        let pz = init.z + inv_factors.z * z as f64;
                        let probe = DVec3::new(px, py, pz);
                        let base = (y * nz + z) * nt;
                        for k in 0..nt {
                            chunk[base + k] = 0.0;
                        }
                        cells.for_each_in_radius(probe, cutoff, |j| {
                            let rt = receptor_xs[j];
                            // Hydrogens have no X-Score type; the reference
                            // implementation skips them on both sides.
                            if rt == XsType::W {
                                return;
                            }
                            let r2 = (probe - positions[j]).length_squared();
                            if r2 > cutoff_sqr {
                                return;
                            }
                            let r = r2.sqrt();
                            for (k, t2) in types_ref.iter().enumerate() {
                                chunk[base + k] += sf.pair_energy_xs(rt, *t2, r);
                            }
                        });
                    }
                }
            });

        self.num_points()
    }

    /// Upload a raw `[x][y][z][slot]` f32 map built elsewhere (the GPU backend).
    ///
    /// `values` must have exactly `num_points() * types.len()` entries; the
    /// layout is identical to the CPU path, so a grid built on the GPU behaves
    /// bit-for-bit like one built here modulo f32 rounding.
    pub fn upload(&mut self, values: &[f32]) -> std::result::Result<(), String> {
        let expected = self.data.len();
        if values.len() != expected {
            return Err(format!(
                "grid upload size mismatch: expected {expected} values, got {}",
                values.len()
            ));
        }
        for (slot, v) in self.data.iter_mut().zip(values) {
            *slot = f64::from(*v);
        }
        Ok(())
    }

    /// Affinity of a probe of type `t` at `p`, plus the out-of-box penalty.
    #[inline]
    fn evaluate(&self, p: DVec3, t: XsType, v: f64, deriv: Option<&mut DVec3>) -> f64 {
        let slot = self.slot[t.index()];
        if slot == usize::MAX {
            if let Some(d) = deriv {
                *d = DVec3::ZERO;
            }
            return 0.0;
        }
        let [nx, ny, nz] = self.shape();
        let nt = self.types.len();
        let init = DVec3::new(self.dims[0].begin, self.dims[1].begin, self.dims[2].begin);
        let nf = [nx - 1, ny - 1, nz - 1];
        let spans = [self.dims[0].span(), self.dims[1].span(), self.dims[2].span()];
        let factors = [
            nf[0] as f64 / spans[0],
            nf[1] as f64 / spans[1],
            nf[2] as f64 / spans[2],
        ];
        let inv_factors = [1.0 / factors[0], 1.0 / factors[1], 1.0 / factors[2]];

        let mut miss = 0.0f64;
        let mut region = [0i32; 3];
        let mut a = [0usize; 3];
        let mut s = [0.0f64; 3];
        for i in 0..3 {
            let si = (p[i] - init[i]) * factors[i];
            if si < 0.0 {
                miss += -si * inv_factors[i];
                region[i] = -1;
                a[i] = 0;
                s[i] = 0.0;
            } else if si >= nf[i] as f64 {
                miss += (si - nf[i] as f64) * inv_factors[i];
                region[i] = 1;
                a[i] = nf[i] - 1;
                s[i] = 1.0;
            } else {
                region[i] = 0;
                a[i] = si as usize;
                s[i] = si - a[i] as f64;
            }
        }
        let penalty = self.slope * miss;

        let (x0, y0, z0) = (a[0], a[1], a[2]);
        let idx = |x: usize, y: usize, z: usize| -> usize { ((x * ny + y) * nz + z) * nt + slot };
        let d = &self.data;
        let (x, y, z) = (s[0], s[1], s[2]);
        let (mx, my, mz) = (1.0 - x, 1.0 - y, 1.0 - z);

        let f000 = d[idx(x0, y0, z0)];
        let f100 = d[idx(x0 + 1, y0, z0)];
        let f010 = d[idx(x0, y0 + 1, z0)];
        let f110 = d[idx(x0 + 1, y0 + 1, z0)];
        let f001 = d[idx(x0, y0, z0 + 1)];
        let f101 = d[idx(x0 + 1, y0, z0 + 1)];
        let f011 = d[idx(x0, y0 + 1, z0 + 1)];
        let f111 = d[idx(x0 + 1, y0 + 1, z0 + 1)];

        let mut f = f000 * mx * my * mz
            + f100 * x * my * mz
            + f010 * mx * y * mz
            + f110 * x * y * mz
            + f001 * mx * my * z
            + f101 * x * my * z
            + f011 * mx * y * z
            + f111 * x * y * z;

        match deriv {
            Some(out) => {
                let gx = (f100 - f000) * my * mz
                    + (f110 - f010) * y * mz
                    + (f101 - f001) * my * z
                    + (f111 - f011) * y * z;
                let gy = (f010 - f000) * mx * mz
                    + (f110 - f100) * x * mz
                    + (f011 - f001) * mx * z
                    + (f111 - f101) * x * z;
                let gz = (f001 - f000) * mx * my
                    + (f101 - f100) * x * my
                    + (f011 - f010) * mx * y
                    + (f111 - f110) * x * y;
                let mut gradient = DVec3::new(gx, gy, gz);
                curl(&mut f, &mut gradient, v);
                for i in 0..3 {
                    out[i] = if region[i] == 0 {
                        factors[i] * gradient[i]
                    } else {
                        self.slope * region[i] as f64
                    };
                }
            }
            None => curl_scalar(&mut f, v),
        }
        f + penalty
    }

    /// Ligand + flexible-residue energy against the receptor grid.
    pub fn eval(&self, atoms: &[Atom], coords: &[DVec3], v: f64) -> f64 {
        let mut e = 0.0;
        for (i, a) in atoms.iter().enumerate() {
            if let Some(t) = grid_type(a.xs) {
                e += self.evaluate(coords[i], t, v, None);
            }
        }
        e
    }

    /// Energy of the atoms at index `>= from` only (i.e. flexible residues).
    pub fn eval_from(&self, atoms: &[Atom], coords: &[DVec3], from: usize, v: f64) -> f64 {
        let mut e = 0.0;
        for i in from..atoms.len() {
            if let Some(t) = grid_type(atoms[i].xs) {
                e += self.evaluate(coords[i], t, v, None);
            }
        }
        e
    }

    /// Grid energy of every atom plus the Cartesian gradient in `forces`.
    pub fn eval_deriv(
        &self,
        atoms: &[Atom],
        coords: &[DVec3],
        v: f64,
        forces: &mut [DVec3],
    ) -> f64 {
        let mut e = 0.0;
        for i in 0..atoms.len() {
            let mut d = DVec3::ZERO;
            if let Some(t) = grid_type(atoms[i].xs) {
                e += self.evaluate(coords[i], t, v, Some(&mut d));
            }
            forces[i] = d;
        }
        e
    }

    /// `true` when every heavy atom lies inside the box (within `margin`).
    pub fn is_in_grid(&self, atoms: &[Atom], coords: &[DVec3], margin: f64) -> bool {
        for (i, a) in atoms.iter().enumerate() {
            if a.is_hydrogen() {
                continue;
            }
            let p = coords[i];
            for j in 0..3 {
                if self.dims[j].n_voxels == 0 {
                    continue;
                }
                if p[j] < self.dims[j].begin - margin || p[j] > self.dims[j].end + margin {
                    return false;
                }
            }
        }
        true
    }

    /// Write the maps in AutoDock `.map` format, one file per atom type.
    ///
    /// The layout matches what AutoDock Vina writes (`cache::write`), so the
    /// files can be consumed by other AutoDock-family tools.
    pub fn write_maps(&self, prefix: &Path) -> std::io::Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        let [nx, ny, nz] = self.shape();
        if (nx - 1) % 2 == 1 || (ny - 1) % 2 == 1 || (nz - 1) % 2 == 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot write .map files with an odd number of voxels; \
                 rebuild the grid with force_even_voxels",
            ));
        }
        let nt = self.types.len();
        for (k, t) in self.types.iter().enumerate() {
            let path = PathBuf::from(format!("{}.{}.map", prefix.display(), map_suffix(*t)));
            let mut s = String::with_capacity(nx * ny * nz * 8);
            s.push_str("GRID_PARAMETER_FILE odock.gpf\n");
            s.push_str("GRID_DATA_FILE odock.fld\n");
            s.push_str("MACROMOLECULE receptor.pdbqt\n");
            s.push_str(&format!("SPACING {}\n", self.spacing));
            s.push_str(&format!("NELEMENTS {} {} {}\n", nx - 1, ny - 1, nz - 1));
            let cx = self.dims[0].begin + self.dims[0].span() * 0.5;
            let cy = self.dims[1].begin + self.dims[1].span() * 0.5;
            let cz = self.dims[2].begin + self.dims[2].span() * 0.5;
            s.push_str(&format!("CENTER {cx} {cy} {cz}\n"));
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let idx = ((x * ny + y) * nz + z) * nt + k;
                        s.push_str(&format!("{:.4}\n", self.data[idx]));
                    }
                }
            }
            std::fs::write(&path, s)?;
            out.push(path);
        }
        Ok(out)
    }
}

/// The AutoDock map-file suffix for an X-Score type.
pub fn map_suffix(t: XsType) -> &'static str {
    match t {
        XsType::CH => "C_H",
        XsType::CP => "C_P",
        XsType::NP => "N_P",
        XsType::ND => "N_D",
        XsType::NA => "N_A",
        XsType::NDA => "N_DA",
        XsType::OP => "O_P",
        XsType::OD => "O_D",
        XsType::OA => "O_A",
        XsType::ODA => "O_DA",
        XsType::SP => "S_P",
        XsType::PP => "P_P",
        XsType::FH => "F_H",
        XsType::ClH => "Cl_H",
        XsType::BrH => "Br_H",
        XsType::IH => "I_H",
        XsType::Si => "Si",
        XsType::At => "At",
        XsType::MetD => "Met_D",
        _ => "C_H",
    }
}

/// The X-Score types that can be stored in a grid, in canonical order.
pub const GRID_TYPES: [XsType; 19] = [
    XsType::CH,
    XsType::CP,
    XsType::NP,
    XsType::ND,
    XsType::NA,
    XsType::NDA,
    XsType::OP,
    XsType::OD,
    XsType::OA,
    XsType::ODA,
    XsType::SP,
    XsType::PP,
    XsType::FH,
    XsType::ClH,
    XsType::BrH,
    XsType::IH,
    XsType::Si,
    XsType::At,
    XsType::MetD,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AdType;
    use crate::scoring::{make_scoring_function, SfChoice};

    fn rc(x: f64, y: f64, z: f64, ad: AdType) -> Atom {
        Atom::new(DVec3::new(x, y, z), ad, 0.0)
    }

    #[test]
    fn box_dimensions_are_rounded_to_the_spacing() {
        let b = GridBox::new(DVec3::ZERO, DVec3::splat(10.0), 0.375);
        let d = b.dims();
        for dim in d {
            assert!(dim.span() >= 10.0 - 1e-9);
            assert!((dim.span() - 0.375 * dim.n_voxels as f64).abs() < 1e-9);
        }
        assert!((b.volume() - d[0].span() * d[1].span() * d[2].span()).abs() < 1e-9);
    }

    #[test]
    fn grid_energy_tracks_the_exact_energy() {
        let receptor = vec![rc(0.0, 0.0, 0.0, AdType::C)];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let gbox = GridBox::new(DVec3::ZERO, DVec3::splat(8.0), 0.25);
        let mut g = AffinityGrid::new(&gbox, 1e6);
        g.populate(&receptor, sf.as_ref(), &[XsType::CH, XsType::OA]);
        assert!(g.is_initialized());
        assert!(g.has_types(&[XsType::CH]));

        let probe = vec![rc(1.7, 0.9, -0.4, AdType::C)];
        let coords = vec![DVec3::new(1.7, 0.9, -0.4)];
        let approx = g.eval(&probe, &coords, 1000.0);
        let mut exact = sf.pair_energy(&probe[0], &receptor[0], (probe[0].coords).length());
        curl_scalar(&mut exact, 1000.0);
        assert!(
            (approx - exact).abs() < 0.05,
            "grid {approx} vs exact {exact}"
        );
    }

    #[test]
    fn grid_gradient_matches_finite_differences() {
        let receptor = vec![
            rc(0.0, 0.0, 0.0, AdType::C),
            rc(1.4, 1.4, 0.0, AdType::A),
            rc(0.0, 2.0, 1.0, AdType::OA),
        ];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let gbox = GridBox::new(DVec3::new(1.0, 1.0, 0.5), DVec3::splat(9.0), 0.375);
        let mut g = AffinityGrid::new(&gbox, 1e6);
        g.populate(&receptor, sf.as_ref(), &[XsType::CH, XsType::CP, XsType::OA]);

        let probe = vec![rc(1.2, 0.8, 0.3, AdType::C)];
        let base = vec![DVec3::new(1.2, 0.8, 0.3)];
        let mut forces = vec![DVec3::ZERO; 1];
        let _ = g.eval_deriv(&probe, &base, 1000.0, &mut forces);
        let energy = |c: &[DVec3]| g.eval(&probe, c, 1000.0);
        let h = 1e-6;
        for j in 0..3 {
            let mut plus = base.clone();
            plus[0][j] += h;
            let mut minus = base.clone();
            minus[0][j] -= h;
            let numeric = (energy(&plus) - energy(&minus)) / (2.0 * h);
            assert!(
                (forces[0][j] - numeric).abs() < 5e-3 * (1.0 + numeric.abs()),
                "axis {j}: analytic {} vs numeric {numeric}",
                forces[0][j]
            );
        }
    }

    #[test]
    fn outside_the_box_is_penalised_linearly() {
        let receptor = vec![rc(0.0, 0.0, 0.0, AdType::C)];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let gbox = GridBox::new(DVec3::ZERO, DVec3::splat(6.0), 0.375);
        let mut g = AffinityGrid::new(&gbox, 1e6);
        g.populate(&receptor, sf.as_ref(), &[XsType::CH]);
        let probe = vec![rc(10.0, 0.0, 0.0, AdType::C)];
        let coords = vec![DVec3::new(10.0, 0.0, 0.0)];
        let e = g.eval(&probe, &coords, 1000.0);
        assert!(e > 1e6, "{e}");
        assert!(!g.is_in_grid(&probe, &coords, 0.0));
        assert!(g.is_in_grid(&probe, &coords, 100.0));
    }

    #[test]
    fn closure_types_collapse() {
        assert_eq!(grid_type(XsType::CHCg0), Some(XsType::CH));
        assert_eq!(grid_type(XsType::CPCg3), Some(XsType::CP));
        assert_eq!(grid_type(XsType::G0), None);
        assert_eq!(grid_type(XsType::W), None);
        assert_eq!(grid_type(XsType::OA), Some(XsType::OA));
    }
}

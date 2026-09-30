// SPDX-License-Identifier: GPL-3.0-or-later
//! Exact (non-grid) scoring against the explicit receptor atoms.
//!
//! This is the OpenDocking equivalent of AutoDock Vina's `non_cache`
//! (`non_cache.cpp`, Apache-2.0): every movable atom is paired with the
//! receptor atoms that lie within the force-field cutoff, using a uniform cell
//! list (Vina's `szv_grid`) to make the neighbour search `O(1)` per atom.
//!
//! The module provides
//!
//! * [`CellList`] 鈥?a uniform spatial hash used by both this module and the
//!   affinity-grid builder,
//! * [`NonCache`] 鈥?receptor atoms plus the exact energy and analytic-gradient
//!   evaluation used for local refinement and final scoring.

use crate::atom::{AdType, XsType};
use crate::molecule::Atom;
use crate::math::{curl, DVec3};
use crate::scoring::{grid::GridDim, ScoringFunction};

/// Maximum number of cells scanned along one axis.
const MAX_CELLS_PER_AXIS: usize = 64;

/// A uniform, axis-aligned spatial hash over a set of points.
#[derive(Debug, Clone)]
pub struct CellList {
    origin: DVec3,
    inv_cell: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    /// Flat `x + nx*(y + ny*z)` bucket index.
    buckets: Vec<Vec<usize>>,
    positions: Vec<DVec3>,
}

impl CellList {
    /// Build a cell list over `positions`.
    ///
    /// `cell` is the side length of a cell in 脜; it is clamped to a sane range
    /// so that neither degenerate single-atom cells nor huge buckets appear.
    pub fn new(positions: &[DVec3], cell: f64) -> CellList {
        let cell = cell.clamp(1.5, 12.0);
        let (mut lo, mut hi) = (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN));
        for p in positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        if positions.is_empty() {
            lo = DVec3::ZERO;
            hi = DVec3::ZERO;
        }
        let span = (hi - lo).max(DVec3::splat(cell));
        let d = |s: f64| -> usize {
            (((s / cell).ceil() as usize) + 1).clamp(1, MAX_CELLS_PER_AXIS)
        };
        let nx = d(span.x);
        let ny = d(span.y);
        let nz = d(span.z);
        let mut buckets = vec![Vec::new(); nx * ny * nz];
        let len = cell * 1.0;
        let idx = |p: DVec3, nx: usize, ny: usize, nz: usize| -> usize {
            let ix = (((p.x - lo.x) / len).floor() as isize).clamp(0, nx as isize - 1) as usize;
            let iy = (((p.y - lo.y) / len).floor() as isize).clamp(0, ny as isize - 1) as usize;
            let iz = (((p.z - lo.z) / len).floor() as isize).clamp(0, nz as isize - 1) as usize;
            ix + nx * (iy + ny * iz)
        };
        for (i, p) in positions.iter().enumerate() {
            let b = idx(*p, nx, ny, nz);
            buckets[b].push(i);
        }
        CellList {
            origin: lo,
            inv_cell: 1.0 / len,
            cell: len,
            nx,
            ny,
            nz,
            buckets,
            positions: positions.to_vec(),
        }
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// `true` when nothing is indexed.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Invoke `f` for every indexed point within `radius` of `p`.
    ///
    /// The callback receives the point index; the caller is responsible for the
    /// final exact distance test (the cell-list test is conservative).
    pub fn for_each_in_radius<F: FnMut(usize)>(&self, p: DVec3, radius: f64, mut f: F) {
        if self.buckets.is_empty() {
            return;
        }
        let span = (radius * self.inv_cell).ceil() as isize;
        let base = (p - self.origin) * self.inv_cell;
        let cx = base.x.floor() as isize;
        let cy = base.y.floor() as isize;
        let cz = base.z.floor() as isize;
        for dz in -span..=span {
            let iz = cz + dz;
            if iz < 0 || iz >= self.nz as isize {
                continue;
            }
            for dy in -span..=span {
                let iy = cy + dy;
                if iy < 0 || iy >= self.ny as isize {
                    continue;
                }
                for dx in -span..=span {
                    let ix = cx + dx;
                    if ix < 0 || ix >= self.nx as isize {
                        continue;
                    }
                    let b = ix as usize
                        + self.nx * (iy as usize + self.ny * (iz as usize));
                    for &i in &self.buckets[b] {
                        f(i);
                    }
                }
            }
        }
    }

    /// Cell side length in 脜.
    pub fn cell_size(&self) -> f64 {
        self.cell
    }
}

/// Exact ligand鈥搑eceptor scoring with analytic gradients.
#[derive(Debug, Clone)]
pub struct NonCache {
    /// Receptor atoms (immobile).
    pub receptor: Vec<Atom>,
    /// Spatial index over [`NonCache::receptor`].
    pub cells: CellList,
    /// The force-field cutoff in 脜.
    pub cutoff: f64,
    /// The largest force-field cutoff in 脜.
    pub max_cutoff: f64,
    /// Out-of-box penalty slope (Vina uses `1e6`).
    pub slope: f64,
    /// Search-box dimensions; `n_voxels == 0` disables the out-of-box penalty.
    pub dims: [GridDim; 3],
}

impl NonCache {
    /// Build the exact scorer over a receptor.
    pub fn new(
        receptor: Vec<Atom>,
        sf: &dyn ScoringFunction,
        dims: [GridDim; 3],
        slope: f64,
    ) -> NonCache {
        let positions: Vec<DVec3> = receptor.iter().map(|a| a.coords).collect();
        let cells = CellList::new(&positions, sf.cutoff().clamp(3.0, 8.0));
        NonCache {
            receptor,
            cells,
            cutoff: sf.cutoff(),
            max_cutoff: sf.max_cutoff(),
            slope,
            dims,
        }
    }

    /// Distance penalty (and its gradient) incurred by an atom lying outside
    /// the search box, plus the clamped position used for the interaction.
    #[inline]
    fn clamp_to_box(&self, p: DVec3) -> (DVec3, f64, DVec3) {
        let mut adj = p;
        let mut penalty = 0.0;
        let mut deriv = DVec3::ZERO;
        for j in 0..3 {
            if self.dims[j].n_voxels == 0 {
                continue;
            }
            let v = p[j];
            if v < self.dims[j].begin {
                adj[j] = self.dims[j].begin;
                penalty += (v - self.dims[j].begin).abs();
                deriv[j] = -1.0;
            } else if v > self.dims[j].end {
                adj[j] = self.dims[j].end;
                penalty += (v - self.dims[j].end).abs();
                deriv[j] = 1.0;
            }
        }
        (adj, penalty * self.slope, deriv * self.slope)
    }

    /// Energy of every movable atom against the receptor, optionally
    /// accumulating the Cartesian gradient in `forces`.
    ///
    /// * `movable` 鈥?movable atoms; indices `0..n_ligand` belong to the ligand,
    ///   the remainder to flexible receptor residues.
    /// * `coords` 鈥?laboratory coordinates of the movable atoms.
    /// * `v` 鈥?the soft cap applied to each atom's energy (Vina's `curl`).
    /// * `ligand_only` 鈥?when true, flexible residues are skipped (used to
    ///   obtain the pure ligand鈥搑eceptor term).
    pub fn eval(
        &self,
        sf: &dyn ScoringFunction,
        movable: &[Atom],
        coords: &[DVec3],
        v: f64,
        ligand_only: bool,
        n_ligand: usize,
        mut forces: Option<&mut [DVec3]>,
    ) -> f64 {
        let cutoff_sqr = sf.cutoff() * sf.cutoff();
        let xs_typed = sf.is_xs_typed();
        let mut total = 0.0;
        for i in 0..movable.len() {
            if ligand_only && i >= n_ligand {
                continue;
            }
            let a = &movable[i];
            let t = a.xs;
            if sf.is_xs_typed() && t == XsType::W {
                // Untyped hydrogen: the X-Score force field ignores it.
                if let Some(f) = forces.as_deref_mut() {
                    f[i] = DVec3::ZERO;
                }
                continue;
            }
            if matches!(t, XsType::G0 | XsType::G1 | XsType::G2 | XsType::G3) {
                if let Some(f) = forces.as_deref_mut() {
                    f[i] = DVec3::ZERO;
                }
                continue;
            }
            let (adj, penalty, pen_deriv) = self.clamp_to_box(coords[i]);
            let mut e = 0.0;
            let mut deriv = DVec3::ZERO;
            self.cells.for_each_in_radius(adj, self.cutoff, |j| {
                let b = &self.receptor[j];
                // Hydrogens carry no X-Score type and are skipped on *both*
                // sides of the pair, exactly as the reference implementation
                // does (`non_cache.cpp` checks `b.get(atom_type::XS) >= n`).
                // Leaving the receptor's hydrogens in adds a small, spurious
                // attraction that biases every pose.
                if xs_typed && b.xs == XsType::W {
                    return;
                }
                let r_ba = adj - b.coords;
                let r2 = r_ba.length_squared();
                if r2 < cutoff_sqr {
                    let r = r2.sqrt();
                    if forces.is_some() {
                        let (pair_e, de) = sf.pair_energy_deriv(a, b, r);
                        e += pair_e;
                        // dE/dx_a = (dE/dr) * (x_a - x_b)/r
                        if r > 0.0 {
                            deriv += r_ba * (de / r);
                        }
                    } else {
                        e += sf.pair_energy(a, b, r);
                    }
                }
            });
            curl(&mut e, &mut deriv, v);
            if let Some(f) = forces.as_deref_mut() {
                f[i] = deriv + pen_deriv;
            }
            total += e + penalty;
        }
        total
    }

    /// `true` when every heavy movable atom lies inside the search box.
    pub fn within(&self, movable: &[Atom], coords: &[DVec3], margin: f64) -> bool {
        for (i, a) in movable.iter().enumerate() {
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
}

/// Pairwise energy over an explicit pair list, optionally with gradient.
///
/// Used for the intra-molecular terms (ligand鈥搇igand, flex鈥揻lex) and for the
/// macrocycle glue pairs. `coords` are the movable-atom coordinates and
/// `forces` receives `鈭侲/鈭倄` contributions when `with_deriv` is set.
///
/// The Vina semantics are preserved exactly:
///
/// * each pair energy is soft-capped with `curl(路, v)`,
/// * the pair must satisfy `r虏 < cutoff虏` (where `cutoff` is the *maximum*
///   cutoff for glue pairs, so that the long-range linear attraction applies).
pub fn eval_pairs(
    sf: &dyn ScoringFunction,
    atoms: &[Atom],
    coords: &[DVec3],
    pairs: &[crate::molecule::Pair],
    v: f64,
    cutoff: f64,
    mut forces: Option<&mut [DVec3]>,
) -> f64 {
    let cutoff_sqr = cutoff * cutoff;
    let mut total = 0.0;
    let xs_typed = sf.is_xs_typed();
    for p in pairs {
        if xs_typed && (atoms[p.a].xs == XsType::W || atoms[p.b].xs == XsType::W) {
            continue;
        }
        let r_ba = coords[p.a] - coords[p.b];
        let r2 = r_ba.length_squared();
        if r2 >= cutoff_sqr {
            continue;
        }
        let r = r2.sqrt();
        let (mut e, de) = if forces.is_some() {
            sf.pair_energy_deriv(&atoms[p.a], &atoms[p.b], r)
        } else {
            (sf.pair_energy(&atoms[p.a], &atoms[p.b], r), 0.0)
        };
        if let Some(f) = forces.as_deref_mut() {
            let mut grad = if r > 0.0 { r_ba * (de / r) } else { DVec3::ZERO };
            curl(&mut e, &mut grad, v);
            f[p.a] += grad;
            f[p.b] -= grad;
        } else {
            crate::math::curl_scalar(&mut e, v);
        }
        total += e;
    }
    total
}

/// Convenience: the AD4/X-Score type of an atom, for dispatch.
#[inline]
pub fn atom_type_of(a: &Atom) -> (AdType, XsType) {
    (a.ad, a.xs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AdType;
    use crate::scoring::{make_scoring_function, SfChoice};

    fn rc(x: f64, y: f64, z: f64, ad: AdType) -> Atom {
        Atom::new(DVec3::new(x, y, z), ad, 0.0)
    }

    #[test]
    fn cell_list_finds_all_neighbours() {
        let positions: Vec<DVec3> = (0..200)
            .map(|i| {
                DVec3::new(
                    (i % 10) as f64 * 2.0,
                    ((i / 10) % 10) as f64 * 2.0,
                    (i / 100) as f64 * 2.0,
                )
            })
            .collect();
        let cl = CellList::new(&positions, 4.0);
        assert_eq!(cl.len(), 200);
        let probe = DVec3::new(9.0, 9.0, 1.0);
        let mut found = Vec::new();
        cl.for_each_in_radius(probe, 3.0, |i| found.push(i));
        found.sort_unstable();
        let reach = 3.0 + 4.0 * 3f64.sqrt();
        for i in 0..200 {
            let d = (positions[i] - probe).length();
            if d <= 3.0 {
                assert!(found.contains(&i), "missed {i} at {d}");
            }
        }
        for &i in &found {
            let d = (positions[i] - probe).length();
            assert!(d <= reach, "spurious {i} at {d}");
        }
        assert!(!found.is_empty());
    }

    #[test]
    fn exact_eval_matches_a_brute_force_sum() {
        let receptor: Vec<Atom> = vec![
            rc(0.0, 0.0, 0.0, AdType::C),
            rc(3.0, 0.0, 0.0, AdType::A),
            rc(0.0, 3.0, 0.0, AdType::OA),
            rc(0.0, 0.0, 3.0, AdType::N),
            rc(12.0, 0.0, 0.0, AdType::C), // far away
        ];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let dims = [GridDim::empty(); 3];
        let nc = NonCache::new(receptor.clone(), sf.as_ref(), dims, 1e6);
        let movable = vec![rc(1.5, 0.0, 0.0, AdType::C), rc(0.0, 1.5, 0.0, AdType::O)];
        let coords: Vec<DVec3> = movable.iter().map(|a| a.coords).collect();

        let got = nc.eval(sf.as_ref(), &movable, &coords, 1000.0, true, 2, None);

        // The reference implementation accumulates every receptor pair of one
        // movable atom and soft-caps the *sum* (non_cache.cpp).
        let mut expect = 0.0;
        for a in &movable {
            let mut this_e = 0.0;
            for b in &receptor {
                let r = (a.coords - b.coords).length();
                if r < sf.cutoff() {
                    this_e += sf.pair_energy(a, b, r);
                }
            }
            crate::math::curl_scalar(&mut this_e, 1000.0);
            expect += this_e;
        }
        assert!((got - expect).abs() < 1e-10, "{got} vs {expect}");
    }

    /// The cell list must find *exactly* the same partners as a brute-force
    /// scan: a single missed neighbour silently weakens every energy.
    #[test]
    fn cell_list_matches_a_brute_force_neighbour_scan() {
        // A large, sparse receptor: the worst case for a cell list.
        let mut rng = crate::rng::Rng::new(20240929);
        let n = 3000;
        let receptor: Vec<Atom> = (0..n)
            .map(|i| {
                let p = DVec3::new(
                    rng.uniform(-35.0, 35.0),
                    rng.uniform(-30.0, 30.0),
                    rng.uniform(-25.0, 25.0),
                );
                let ad = match i % 5 {
                    0 => AdType::OA,
                    1 => AdType::N,
                    2 => AdType::HD,
                    _ => AdType::C,
                };
                let mut a = Atom::new(p, ad, 0.0);
                a.bonds.push(crate::molecule::Bond {
                    other: 0,
                    length: 1.0,
                    rotatable: false,
                });
                a
            })
            .collect();
        let mut receptor = receptor;
        crate::molecule::assign_types(&mut receptor);

        let sf = make_scoring_function(SfChoice::Vina, None);
        let dims = [GridDim::empty(); 3];
        let nc = NonCache::new(receptor.clone(), sf.as_ref(), dims, 1e6);

        // Probes inside, near the edge, and well outside the box.
        let probes = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(30.0, -28.0, 22.0),
            DVec3::new(-60.0, 0.0, 0.0),
            DVec3::new(34.5, 29.5, 24.5),
        ];
        for (k, p) in probes.iter().enumerate() {
            let movable = vec![{
                let mut a = Atom::new(*p, AdType::C, 0.0);
                a.bonds.push(crate::molecule::Bond {
                    other: 0,
                    length: 1.0,
                    rotatable: false,
                });
                a
            }];
            crate::molecule::assign_types(&mut movable.clone());
            let mut a = movable[0].clone();
            crate::molecule::assign_types(std::slice::from_mut(&mut a));
            let movable = vec![a];
            let coords = vec![*p];
            let got = nc.eval(sf.as_ref(), &movable, &coords, 1000.0, false, 1, None);

            let mut expect = 0.0;
            for b in &receptor {
                if b.xs == crate::atom::XsType::W {
                    continue;
                }
                let r = (movable[0].coords - b.coords).length();
                if r < sf.cutoff() {
                    expect += sf.pair_energy(&movable[0], b, r);
                }
            }
            crate::math::curl_scalar(&mut expect, 1000.0);
            assert!(
                (got - expect).abs() < 1e-9,
                "probe {k}: cell list gave {got}, brute force {expect}"
            );
        }
    }

    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let receptor: Vec<Atom> = vec![
            rc(0.0, 0.0, 0.0, AdType::C),
            rc(3.0, 0.4, 0.0, AdType::A),
            rc(0.3, 3.0, 0.2, AdType::OA),
            rc(0.1, 0.0, 3.0, AdType::N),
        ];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let dims = [GridDim::empty(); 3];
        let nc = NonCache::new(receptor, sf.as_ref(), dims, 1e6);
        let movable = vec![
            rc(1.6, 0.1, 0.2, AdType::C),
            rc(0.2, 1.7, -0.3, AdType::O),
        ];

        let energy = |coords: &[DVec3]| -> f64 {
            nc.eval(sf.as_ref(), &movable, coords, 1000.0, true, 2, None)
        };

        let base: Vec<DVec3> = movable.iter().map(|a| a.coords).collect();
        let mut forces = vec![DVec3::ZERO; movable.len()];
        let _ = nc.eval(
            sf.as_ref(),
            &movable,
            &base,
            1000.0,
            true,
            2,
            Some(&mut forces),
        );

        let h = 1e-6;
        for i in 0..movable.len() {
            for j in 0..3 {
                let mut plus = base.clone();
                plus[i][j] += h;
                let mut minus = base.clone();
                minus[i][j] -= h;
                let numeric = (energy(&plus) - energy(&minus)) / (2.0 * h);
                assert!(
                    (forces[i][j] - numeric).abs() < 1e-5 * (1.0 + numeric.abs()),
                    "atom {i} axis {j}: analytic {} vs numeric {numeric}",
                    forces[i][j]
                );
            }
        }
    }

    #[test]
    fn out_of_box_penalty_is_a_linear_ramp() {
        let receptor: Vec<Atom> = vec![rc(-20.0, 0.0, 0.0, AdType::C)];
        let sf = make_scoring_function(SfChoice::Vina, None);
        let dims = [
            GridDim {
                begin: -5.0,
                end: 5.0,
                n_voxels: 10,
            },
            GridDim {
                begin: -5.0,
                end: 5.0,
                n_voxels: 10,
            },
            GridDim {
                begin: -5.0,
                end: 5.0,
                n_voxels: 10,
            },
        ];
        let nc = NonCache::new(receptor, sf.as_ref(), dims, 1e6);
        let movable = vec![rc(8.0, 0.0, 0.0, AdType::C)];
        let coords = vec![DVec3::new(8.0, 0.0, 0.0)];
        let e = nc.eval(sf.as_ref(), &movable, &coords, 1000.0, true, 1, None);
        // 3 脜 outside the box, slope 1e6, plus the (curled) interaction energy.
        assert!(e >= 3.0e6 - 1e-6, "{e}");
        let mut forces = vec![DVec3::ZERO; 1];
        let _ = nc.eval(
            sf.as_ref(),
            &movable,
            &coords,
            1000.0,
            true,
            1,
            Some(&mut forces),
        );
        assert!((forces[0].x - 1.0e6).abs() < 1.0, "{:?}", forces[0]);
    }
}

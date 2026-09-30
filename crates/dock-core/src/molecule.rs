// SPDX-License-Identifier: GPL-3.0-or-later
//! Molecules: atoms, bonds, graph perception and intra-molecular pair lists.
//!
//! # Bond perception
//!
//! PDBQT files carry no explicit connectivity, so — exactly like AutoDock Vina
//! — connectivity is perceived from the geometry. An atom pair `(i, j)` is
//! bonded when
//!
//! ```text
//! d(i, j) < factor * (rcov(i) + rcov(j))          factor = 1.1
//! ```
//!
//! and no third atom `k` sits in the "lens" between them, i.e. there is no `k`
//! with
//!
//! ```text
//! d(i,k) < d(i,j)  and  d(j,k) < d(i,j)
//! d(i,k) < factor * (rcov(i) + rcov(k))   and   d(j,k) < factor * (rcov(j) + rcov(k))
//! ```
//!
//! Vina uses the same geometric test but gates the third-atom check on its
//! *mobility matrix* (whether `k` is immobile relative to `i` or `j`). The
//! mobility-gated form is an approximation of the purely geometric statement
//! ("k is genuinely between i and j and bonded to both"), and the two agree on
//! every well-formed structure; the geometric form additionally works when no
//! mobility information is available, e.g. for a standalone receptor.

use crate::atom::{ad_type_property, covalent_radius, AdType, Element, XsType};
use crate::math::{distance_sqr, DVec3};

/// Bond-length tolerance factor (Vina's `bond_length_allowance_factor`).
pub const BOND_LENGTH_ALLOWANCE_FACTOR: f64 = 1.1;

/// A pair of atoms that contributes an explicit interaction term.
///
/// `a` is always strictly less than `b` in the arrays produced by this crate,
/// which lets the scoring code skip the symmetric half of the pair matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pair {
    /// Lower atom index.
    pub a: usize,
    /// Higher atom index.
    pub b: usize,
}

impl Pair {
    /// Create a pair, normalising the order.
    #[inline]
    pub fn new(a: usize, b: usize) -> Pair {
        if a < b {
            Pair { a, b }
        } else {
            Pair { a: b, b: a }
        }
    }
}

/// A perceived covalent bond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    /// Index of the partner atom inside the same [`Molecule`].
    pub other: usize,
    /// Inter-atomic distance at perception time (Å).
    pub length: f64,
    /// Whether this bond is a rotatable torsion in the input topology.
    pub rotatable: bool,
}

/// A single atom with its perceived connectivity and force-field typing.
#[derive(Debug, Clone, PartialEq)]
pub struct Atom {
    /// PDBQT serial number (0 when unknown).
    pub serial: i32,
    /// Atom name (PDB columns 13-16).
    pub name: String,
    /// Residue name.
    pub res_name: String,
    /// Chain identifier.
    pub chain_id: String,
    /// Residue sequence number.
    pub res_id: i32,
    /// Current Cartesian position in the laboratory frame (Å).
    pub coords: DVec3,
    /// AutoDock 4 atom type.
    pub ad: AdType,
    /// X-Score atom type (Vina / Vinardo force fields).
    pub xs: XsType,
    /// Chemical element.
    pub el: Element,
    /// Partial charge (used by the AD4 force field).
    pub charge: f64,
    /// AutoDock "atom type" string as it appeared in the file (kept for output).
    pub type_name: String,
    /// Perceived covalent bonds.
    pub bonds: Vec<Bond>,
}

impl Atom {
    /// Create an atom with no bonds; typing is filled in by [`perceive_bonds`].
    pub fn new(coords: DVec3, ad: AdType, charge: f64) -> Atom {
        let el = ad.element();
        Atom {
            serial: 0,
            name: String::new(),
            res_name: String::new(),
            chain_id: String::new(),
            res_id: 0,
            coords,
            ad,
            xs: XsType::from_element(el, ad, false, false),
            el,
            charge,
            type_name: ad.name().to_string(),
            bonds: Vec::new(),
        }
    }

    /// `true` for a non-polar (AD4 `H`) or polar (`HD`) hydrogen.
    #[inline]
    pub fn is_hydrogen(&self) -> bool {
        self.el == Element::H
    }

    /// Vina's `is_heteroatom`.
    #[inline]
    pub fn is_heteroatom(&self) -> bool {
        self.ad.is_heteroatom()
    }

    /// Sum of the two covalent radii (Vina's `optimal_covalent_bond_length`).
    #[inline]
    pub fn optimal_covalent_bond_length(&self, other: &Atom) -> f64 {
        covalent_radius(self.ad) + covalent_radius(other.ad)
    }

    /// `true` when this atom is covalently bound to a polar hydrogen.
    fn bonded_to_hd(&self, atoms: &[Atom]) -> bool {
        self.bonds
            .iter()
            .any(|b| atoms[b.other].ad == AdType::HD)
    }

    /// `true` when this atom is covalently bound to a non-C/non-H atom.
    fn bonded_to_heteroatom(&self, atoms: &[Atom]) -> bool {
        self.bonds
            .iter()
            .any(|b| atoms[b.other].is_heteroatom())
    }
}

/// A molecule: a flat atom list plus perceived connectivity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Molecule {
    /// The atoms.
    pub atoms: Vec<Atom>,
}

impl Molecule {
    /// Empty molecule.
    pub fn new() -> Molecule {
        Molecule { atoms: Vec::new() }
    }

    /// Build from a plain atom list (bond perception is *not* run).
    pub fn from_atoms(atoms: Vec<Atom>) -> Molecule {
        Molecule { atoms }
    }

    /// Number of atoms.
    #[inline]
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// `true` when the molecule has no atoms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Arithmetic mean of all atom positions (Vina's `model::center`).
    pub fn center(&self) -> DVec3 {
        if self.atoms.is_empty() {
            return DVec3::ZERO;
        }
        let mut c = DVec3::ZERO;
        for a in &self.atoms {
            c += a.coords;
        }
        c / self.atoms.len() as f64
    }

    /// Bounding box as `(min, max)`.
    pub fn bounding_box(&self) -> (DVec3, DVec3) {
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        for a in &self.atoms {
            lo = lo.min(a.coords);
            hi = hi.max(a.coords);
        }
        (lo, hi)
    }

    /// Look up an atom index by PDBQT serial number.
    pub fn index_of_serial(&self, serial: i32) -> Option<usize> {
        self.atoms.iter().position(|a| a.serial == serial)
    }

    /// Perceive bonds and (re)assign X-Score atom types.
    pub fn perceive(&mut self) {
        perceive_bonds(&mut self.atoms);
        assign_types(&mut self.atoms);
    }

    /// Append `other` to this molecule, shifting bond indices, then re-perceive.
    pub fn append(&mut self, other: &Molecule) {
        let offset = self.atoms.len();
        for a in &other.atoms {
            let mut a = a.clone();
            for b in &mut a.bonds {
                b.other += offset;
            }
            self.atoms.push(a);
        }
    }

    /// Remove every atom for which `keep` returns `false`, and remap bonds.
    ///
    /// Bond indices that point at removed atoms are dropped; the molecule is
    /// re-perceived afterwards by the caller when needed.
    pub fn retain<F: Fn(&Atom) -> bool>(&mut self, keep: F) {
        let mut new_index = vec![usize::MAX; self.atoms.len()];
        let mut kept = Vec::new();
        for (i, a) in self.atoms.iter().enumerate() {
            if keep(a) {
                new_index[i] = kept.len();
                kept.push(a.clone());
            }
        }
        for a in &mut kept {
            a.bonds.clear();
        }
        for (old, a) in self.atoms.iter().enumerate() {
            let Some(ni) = (new_index[old] != usize::MAX).then_some(new_index[old]) else {
                continue;
            };
            for b in &a.bonds {
                if new_index[b.other] != usize::MAX {
                    kept[ni].bonds.push(Bond {
                        other: new_index[b.other],
                        length: b.length,
                        rotatable: b.rotatable,
                    });
                }
            }
            kept[ni].bonds.sort_by_key(|b| b.other);
        }
        self.atoms = kept;
    }
}

/// Perceive covalent bonds for `atoms` using distance + covalent radii.
///
/// Any pre-existing bond list is discarded. The perception is deterministic.
///
/// # Complexity
///
/// A uniform cell list makes this `O(n · k)` with `k` the number of atoms inside
/// one cell (tens, for a protein), instead of the naive `O(n³)` of a
/// three-atom "lens" test. A 5 000-atom receptor is perceived in a few
/// milliseconds, which matters because perception now runs on the receptor as
/// well as on the ligand.
pub fn perceive_bonds(atoms: &mut [Atom]) {
    let n = atoms.len();
    for a in atoms.iter_mut() {
        a.bonds.clear();
    }
    if n < 2 {
        return;
    }

    let cov: Vec<f64> = atoms.iter().map(|a| covalent_radius(a.ad)).collect();
    let max_pair_cutoff = 2.0 * BOND_LENGTH_ALLOWANCE_FACTOR * cov.iter().cloned().fold(0.0, f64::max);
    let cell = (max_pair_cutoff.max(1.5)).min(6.0);

    // --- uniform grid ------------------------------------------------------
    let mut lo = DVec3::splat(f64::MAX);
    let mut hi = DVec3::splat(f64::MIN);
    for a in atoms.iter() {
        lo = lo.min(a.coords);
        hi = hi.max(a.coords);
    }
    let axis = |span: f64| -> usize { (((span / cell).ceil() as usize) + 1).clamp(1, 256) };
    let nx = axis(hi.x - lo.x);
    let ny = axis(hi.y - lo.y);
    let nz = axis(hi.z - lo.z);
    let bucket_of = |p: DVec3| -> usize {
        let ix = (((p.x - lo.x) / cell).floor() as isize).clamp(0, nx as isize - 1) as usize;
        let iy = (((p.y - lo.y) / cell).floor() as isize).clamp(0, ny as isize - 1) as usize;
        let iz = (((p.z - lo.z) / cell).floor() as isize).clamp(0, nz as isize - 1) as usize;
        ix + nx * (iy + ny * iz)
    };
    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); nx * ny * nz];
    for (i, a) in atoms.iter().enumerate() {
        let b = bucket_of(a.coords);
        buckets[b].push(i);
    }
    // Flat neighbour lists within one cell plus its 26 neighbours.
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let b = ix + nx * (iy + ny * iz);
                if buckets[b].is_empty() {
                    continue;
                }
                let mut pool: Vec<usize> = Vec::new();
                for dx in -1i64..=1 {
                    for dy in -1i64..=1 {
                        for dz in -1i64..=1 {
                            let jx = ix as i64 + dx;
                            let jy = iy as i64 + dy;
                            let jz = iz as i64 + dz;
                            if jx < 0 || jy < 0 || jz < 0 {
                                continue;
                            }
                            let (jx, jy, jz) = (jx as usize, jy as usize, jz as usize);
                            if jx >= nx || jy >= ny || jz >= nz {
                                continue;
                            }
                            pool.extend_from_slice(&buckets[jx + nx * (jy + ny * jz)]);
                        }
                    }
                }
                for &i in &buckets[b] {
                    neighbours[i] = pool.clone();
                }
            }
        }
    }

    // --- candidate bonds ---------------------------------------------------
    let mut bonds: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for &j in &neighbours[i] {
            if j <= i {
                continue;
            }
            let cutoff = BOND_LENGTH_ALLOWANCE_FACTOR * (cov[i] + cov[j]);
            let d2 = distance_sqr(atoms[i].coords, atoms[j].coords);
            if d2 >= cutoff * cutoff {
                continue;
            }
            let d = d2.sqrt();
            // Lens test: reject when a third atom is bonded-close to both and
            // lies strictly between them. Only `i`'s neighbours can be in that
            // lens, so the search space is tiny.
            let mut blocked = false;
            for &k in &neighbours[i] {
                if k == i || k == j {
                    continue;
                }
                let dik = distance_sqr(atoms[i].coords, atoms[k].coords);
                if dik >= d2 {
                    continue;
                }
                let djk = distance_sqr(atoms[j].coords, atoms[k].coords);
                if djk >= d2 {
                    continue;
                }
                let cik = BOND_LENGTH_ALLOWANCE_FACTOR * (cov[i] + cov[k]);
                if dik >= cik * cik {
                    continue;
                }
                let cjk = BOND_LENGTH_ALLOWANCE_FACTOR * (cov[j] + cov[k]);
                if djk >= cjk * cjk {
                    continue;
                }
                blocked = true;
                break;
            }
            if !blocked {
                bonds.push((i, j, d));
            }
        }
    }

    for (i, j, d) in bonds {
        atoms[i].bonds.push(Bond {
            other: j,
            length: d,
            rotatable: false,
        });
        atoms[j].bonds.push(Bond {
            other: i,
            length: d,
            rotatable: false,
        });
    }
    for a in atoms.iter_mut() {
        a.bonds.sort_by_key(|b| b.other);
    }
}

/// Re-derive [`Atom::xs`] from the element, the AD4 type and the bond graph.
pub fn assign_types(atoms: &mut [Atom]) {
    // Two passes: the query only reads bond lists, which are not modified here.
    let flags: Vec<(bool, bool)> = atoms
        .iter()
        .map(|a| (a.bonded_to_hd(atoms), a.bonded_to_heteroatom(atoms)))
        .collect();
    for (a, (hd, het)) in atoms.iter_mut().zip(flags) {
        a.el = a.ad.element();
        a.xs = XsType::from_element(a.el, a.ad, hd, het);
    }
}

/// The set of atoms reachable from `root` within `depth` bonds, including
/// `root` itself.
///
/// Vina's `bonded_to(a, 3)` therefore yields `root`, its 1-2 and its 1-3
/// neighbours, i.e. everything that must be excluded from the intra-molecular
/// interaction list ("up to 1-4").
pub fn bonded_within(atoms: &[Atom], root: usize, depth: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::with_capacity(8);
    let mut seen = vec![false; atoms.len()];
    fn walk(
        atoms: &[Atom],
        seen: &mut [bool],
        out: &mut Vec<usize>,
        idx: usize,
        depth: usize,
    ) {
        if seen[idx] {
            return;
        }
        seen[idx] = true;
        out.push(idx);
        if depth > 0 {
            for b in &atoms[idx].bonds {
                walk(atoms, seen, out, b.other, depth - 1);
            }
        }
    }
    walk(atoms, &mut seen, &mut out, root, depth);
    out
}

/// Count heavy-atom rotatable bonds defined by the AD4/PDBQT topology.
///
/// Used when a ligand carries no explicit `TORSDOF`; Vina's own torsion count
/// is derived from the kinematic tree instead, so this is only a fallback.
pub fn count_rotatable_bonds(atoms: &[Atom]) -> usize {
    let mut n = 0;
    for i in 0..atoms.len() {
        if atoms[i].is_hydrogen() {
            continue;
        }
        for b in &atoms[i].bonds {
            let j = b.other;
            if j <= i || atoms[j].is_hydrogen() {
                continue;
            }
            // A bond is a rotor when both endpoints have >1 heavy neighbour,
            // which excludes terminal groups such as -CH3 and -OH.
            let heavy_i = heavy_degree(atoms, i);
            let heavy_j = heavy_degree(atoms, j);
            if heavy_i > 1 && heavy_j > 1 {
                n += 1;
            }
        }
    }
    n
}

/// Number of heavy-atom neighbours of `idx`.
pub fn heavy_degree(atoms: &[Atom], idx: usize) -> usize {
    atoms[idx]
        .bonds
        .iter()
        .filter(|b| !atoms[b.other].is_hydrogen())
        .count()
}

/// Gyration radius of a subset of atoms about their centroid.
pub fn gyration_radius(atoms: &[Atom], subset: &[usize]) -> f64 {
    let mut heavy: Vec<usize> = subset
        .iter()
        .copied()
        .filter(|&i| !atoms[i].is_hydrogen())
        .collect();
    if heavy.is_empty() {
        heavy = subset.to_vec();
    }
    if heavy.is_empty() {
        return 0.0;
    }
    let mut c = DVec3::ZERO;
    for &i in &heavy {
        c += atoms[i].coords;
    }
    c /= heavy.len() as f64;
    let sum: f64 = heavy
        .iter()
        .map(|&i| distance_sqr(atoms[i].coords, c))
        .sum();
    (sum / heavy.len() as f64).sqrt()
}

/// Heavy-atom RMSD between two coordinate sets, computed over the best
/// one-to-one *element-wise* assignment.
///
/// This is Vina's `rmsd_lower_bound`, a permutation-invariant lower bound on
/// the true symmetry-corrected RMSD. It is what the "rmsd l.b." column reports.
pub fn rmsd_lower_bound(a: &[Atom], b: &[Atom]) -> f64 {
    fn asymmetric(x: &[Atom], y: &[Atom]) -> f64 {
        let mut sum = 0.0;
        let mut counter = 0usize;
        for ai in x {
            if ai.el == Element::H {
                continue;
            }
            let mut best = f64::MAX;
            for bj in y {
                if bj.el == ai.el && bj.el != Element::H {
                    let d = distance_sqr(ai.coords, bj.coords);
                    if d < best {
                        best = d;
                    }
                }
            }
            if best < f64::MAX {
                sum += best;
                counter += 1;
            }
        }
        if counter == 0 {
            0.0
        } else {
            (sum / counter as f64).sqrt()
        }
    }
    asymmetric(a, b).max(asymmetric(b, a))
}

/// Heavy-atom RMSD with a fixed atom-to-atom correspondence (Vina's
/// `rmsd_upper_bound`).
pub fn rmsd_upper_bound(a: &[Atom], b: &[Atom]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    let mut counter = 0usize;
    for i in 0..n {
        if a[i].el == Element::H {
            continue;
        }
        sum += distance_sqr(a[i].coords, b[i].coords);
        counter += 1;
    }
    if counter == 0 {
        0.0
    } else {
        (sum / counter as f64).sqrt()
    }
}

/// Squared-distance helper used by callers that already hold coordinates.
#[inline]
pub fn coords_distance_sqr(a: DVec3, b: DVec3) -> f64 {
    distance_sqr(a, b)
}

/// Convenience: AD4 van der Waals radius of an atom.
#[inline]
pub fn vdw_radius(t: AdType) -> f64 {
    ad_type_property(t).radius
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(name: &str, el: AdType, x: f64, y: f64, z: f64) -> Atom {
        let mut a = Atom::new(DVec3::new(x, y, z), el, 0.0);
        a.name = name.to_string();
        a
    }

    #[test]
    fn perceives_water_and_methane_bonds() {
        // Water: O-H 0.96 Å, H-H 1.51 Å.
        let mut mol = Molecule::from_atoms(vec![
            atom("O", AdType::OA, 0.0, 0.0, 0.0),
            atom("H1", AdType::HD, 0.96, 0.0, 0.0),
            atom("H2", AdType::HD, -0.24, 0.93, 0.0),
        ]);
        mol.perceive();
        assert_eq!(mol.atoms[0].bonds.len(), 2, "oxygen must have two bonds");
        assert_eq!(mol.atoms[1].bonds.len(), 1);
        assert_eq!(mol.atoms[2].bonds.len(), 1);
        // The two hydrogens must NOT be bonded to each other (lens test).
        assert!(mol.atoms[1].bonds.iter().all(|b| b.other != 2));
        // Oxygen bonded to polar H => donor + acceptor.
        assert_eq!(mol.atoms[0].xs, XsType::ODA);
    }

    #[test]
    fn perceives_carbonyl_without_spurious_bonds() {
        // Formaldehyde-like fragment: C=O 1.21, C-H 1.10.
        let mut mol = Molecule::from_atoms(vec![
            atom("C", AdType::C, 0.0, 0.0, 0.0),
            atom("O", AdType::O, 1.21, 0.0, 0.0),
            atom("H1", AdType::H, -0.60, 0.94, 0.0),
            atom("H2", AdType::H, -0.60, -0.94, 0.0),
        ]);
        mol.perceive();
        assert_eq!(mol.atoms[0].bonds.len(), 3);
        assert!(mol.atoms[1].bonds.iter().all(|b| b.other == 0));
        assert_eq!(mol.atoms[1].xs, XsType::OP); // not an acceptor in PDBQT terms
        assert_eq!(mol.atoms[0].xs, XsType::CP); // bonded to a heteroatom
    }

    #[test]
    fn bonded_within_is_up_to_three_bonds() {
        // Linear chain C0-C1-C2-C3.
        let mut mol = Molecule::from_atoms(vec![
            atom("C0", AdType::C, 0.0, 0.0, 0.0),
            atom("C1", AdType::C, 1.5, 0.0, 0.0),
            atom("C2", AdType::C, 3.0, 0.0, 0.0),
            atom("C3", AdType::C, 4.5, 0.0, 0.0),
        ]);
        mol.perceive();
        let v = bonded_within(&mol.atoms, 0, 3);
        assert!(v.contains(&0) && v.contains(&1) && v.contains(&2) && v.contains(&3));
        let v = bonded_within(&mol.atoms, 0, 2);
        assert!(v.contains(&0) && v.contains(&1) && v.contains(&2));
    }

    #[test]
    fn rotatable_bond_count_excludes_terminal_groups() {
        // CH3-CH2-CH2-OH: only the central C-C bond has two non-terminal
        // heavy-atom endpoints, so exactly one rotor is counted.
        let mut mol = Molecule::from_atoms(vec![
            atom("C1", AdType::C, 0.00, 0.0, 0.0),
            atom("C2", AdType::C, 1.52, 0.0, 0.0),
            atom("C3", AdType::C, 3.04, 0.0, 0.0),
            atom("O", AdType::OA, 3.62, 1.30, 0.0),
            atom("H1", AdType::H, -0.55, 0.90, 0.0),
            atom("H2", AdType::H, -0.55, -0.50, 0.90),
            atom("H3", AdType::H, -0.55, -0.50, -0.90),
            atom("HO", AdType::HD, 3.22, 2.10, 0.0),
        ]);
        mol.perceive();
        assert_eq!(count_rotatable_bonds(&mol.atoms), 1);
    }

    #[test]
    fn rmsd_helpers_behave() {
        let a = vec![
            atom("C", AdType::C, 0.0, 0.0, 0.0),
            atom("O", AdType::O, 1.2, 0.0, 0.0),
        ];
        let mut b = a.clone();
        b[0].coords += DVec3::new(0.3, 0.0, 0.0);
        // One of the two heavy atoms moved by 0.3 Å => RMSD = 0.3 / sqrt(2).
        let expect = (0.09f64 / 2.0).sqrt();
        let r = rmsd_upper_bound(&a, &b);
        assert!((r - expect).abs() < 1e-12, "{r} vs {expect}");
        assert!(rmsd_lower_bound(&a, &b) <= expect + 1e-12);
    }

    #[test]
    fn retain_remaps_bonds() {
        let mut mol = Molecule::from_atoms(vec![
            atom("C", AdType::C, 0.0, 0.0, 0.0),
            atom("O", AdType::O, 1.2, 0.0, 0.0),
            atom("H", AdType::H, -0.6, 0.9, 0.0),
        ]);
        mol.perceive();
        mol.retain(|a| a.el != Element::H);
        assert_eq!(mol.len(), 2);
        assert_eq!(mol.atoms[0].bonds.len(), 1);
        assert_eq!(mol.atoms[0].bonds[0].other, 1);
    }
}

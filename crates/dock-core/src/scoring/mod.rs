// SPDX-License-Identifier: GPL-3.0-or-later
//! Scoring functions: the empirical potentials and their analytic derivatives.
//!
//! Three force fields are implemented, all matching their reference
//! implementations term for term:
//!
//! | choice | terms | reference |
//! |---|---|---|
//! | [`SfChoice::Vina`] | gauss1, gauss2, repulsion, hydrophobic, non-directional H-bond, linear attraction | Trott & Olson, *J. Comput. Chem.* **31**, 455 (2010); Vina `potentials.h` |
//! | [`SfChoice::Vinardo`] | gauss, repulsion, hydrophobic, non-directional H-bond, linear attraction | Quiroga & Villarreal, *PLoS ONE* **11**, e0155183 (2016); Vina `potentials.h` |
//! | [`SfChoice::Ad4`] | 12-6 vdW, 12-10 H-bond, screened electrostatics, desolvation | Morris et al., *J. Comput. Chem.* **30**, 2785 (2009); Vina `potentials.h` |
//!
//! ## Derivation of the Vina terms
//!
//! With `d_ij` the interatomic distance, `R_i` the X-Score van der Waals radius
//! of atom `i` and `r_ij = d_ij - (R_i + R_j)` the *reduced* distance:
//!
//! ```text
//! gauss1(r)      = exp(-(r / 0.5)^2)
//! gauss2(r)      = exp(-((r - 3.0) / 2.0)^2)
//! repulsion(r)   = r^2                       if r < 0 else 0
//! hydrophobic(r) = slope_step(1.5, 0.5, r)   only if both atoms are hydrophobic
//! hbond(r)       = slope_step(0.0, -0.7, r)  only if a donor/acceptor pair exists
//! ```
//!
//! The pair energy is the weighted sum
//!
//! ```text
//! E_ij = w1路gauss1(r) + w2路gauss2(r) + w3路repulsion(r)
//!      + w4路hydrophobic(r) + w5路hbond(r)                 for r < 8 脜, else 0
//! ```
//!
//! and the conformation dependence enters through the torsional penalty
//!
//! ```text
//! E_total = (E_inter + E_intra - E_unbound) / (1 + w_rot 路 N_tors)
//! ```
//!
//! ## Analytic gradients
//!
//! Each term is differentiated exactly with respect to `r` and converted to a
//! Cartesian gradient through
//!
//! ```text
//! 鈭侲_ij/鈭倄_i = (dE_ij/dr_ij) 路 (x_i - x_j) / d_ij
//! ```
//!
//! Every term is `C鹿` in `r` on `(0, cutoff)`. The only discontinuity is the
//! (tiny) jump of `gauss2` at the 8 脜 cutoff, whose magnitude is
//! `w2路exp(-1/4) 鈮?-0.004` kcal/mol.

use std::fmt;
use std::sync::Arc;

use crate::atom::{
    ad_type_property, is_glued, optimal_distance, optimal_distance_vinardo, AdType, XsType,
    ATOM_KIND_DATA, METAL_SOLVATION_PARAMETER,
};
use crate::math::{sqr, slope_step, EPSILON, MAX_F, PI};
use crate::molecule::{Atom, Bond, Molecule};

pub mod grid;
pub mod noncache;

// ---------------------------------------------------------------------------
// Scoring-function choice and weights
// ---------------------------------------------------------------------------

/// Which force field to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SfChoice {
    /// AutoDock Vina's empirical scoring function (default).
    Vina,
    /// The Vinardo re-parameterisation of the Vina terms.
    Vinardo,
    /// The AutoDock 4.2 force field.
    Ad4,
}

impl SfChoice {
    /// Lower-case identifier used by the CLI / Python layer.
    pub fn name(self) -> &'static str {
        match self {
            SfChoice::Vina => "vina",
            SfChoice::Vinardo => "vinardo",
            SfChoice::Ad4 => "ad4",
        }
    }

    /// Parse the identifier.
    pub fn parse(s: &str) -> Option<SfChoice> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vina" => Some(SfChoice::Vina),
            "vinardo" => Some(SfChoice::Vinardo),
            "ad4" | "ad42" | "autodock4" => Some(SfChoice::Ad4),
            _ => None,
        }
    }
}

impl fmt::Display for SfChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Force-field term weights.
///
/// `terms` holds one weight per distance-dependent potential, in the order the
/// potentials are declared. `rot` is the *user-facing* torsional weight, i.e.
/// the value quoted in the papers: `E_final = E / (1 + rot 路 N_tors)` for
/// Vina/Vinardo and `E_final = E + rot 路 N_tors` for AD4.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    /// One weight per distance-dependent term.
    pub terms: Vec<f64>,
    /// Torsional weight.
    pub rot: f64,
}

impl Weights {
    /// AutoDock Vina defaults (`vina.h`, Apache-2.0).
    pub fn vina_default() -> Weights {
        Weights {
            terms: vec![-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0],
            rot: 0.05846,
        }
    }

    /// Vinardo defaults (`vina.h`, Apache-2.0).
    pub fn vinardo_default() -> Weights {
        Weights {
            terms: vec![-0.045, 0.8, -0.035, -0.600, 50.0],
            rot: 0.05846,
        }
    }

    /// AutoDock 4.2 defaults (`vina.h`, Apache-2.0).
    pub fn ad4_default() -> Weights {
        Weights {
            terms: vec![0.1662, 0.1209, 0.1406, 0.1322, 50.0],
            rot: 0.2983,
        }
    }

    /// The default weights for a force field.
    pub fn default_for(choice: SfChoice) -> Weights {
        match choice {
            SfChoice::Vina => Weights::vina_default(),
            SfChoice::Vinardo => Weights::vinardo_default(),
            SfChoice::Ad4 => Weights::ad4_default(),
        }
    }

    /// Number of distance-dependent terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Set one term weight by index.
    pub fn set_term(&mut self, index: usize, value: f64) -> bool {
        match self.terms.get_mut(index) {
            Some(slot) => {
                *slot = value;
                true
            }
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Individual potentials
// ---------------------------------------------------------------------------

/// Which radii table a [`Term::Gauss`] should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Radii {
    /// Vina van der Waals radii.
    Vina,
    /// Vinardo van der Waals radii.
    Vinardo,
}

/// A single distance-dependent potential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Term {
    /// `exp(-(r/width)^2)` evaluated at the reduced distance plus an offset.
    Gauss {
        /// Additive offset applied to the optimal distance.
        offset: f64,
        /// Gaussian width (脜).
        width: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// `r^2` for `r < 0` (a soft-sphere repulsion).
    Repulsion {
        /// Offset added to the optimal distance.
        offset: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// Smooth hydrophobic contact term.
    Hydrophobic {
        /// Reduced distance where the term reaches 1.
        good: f64,
        /// Reduced distance where the term reaches 0.
        bad: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// Non-directional hydrogen bond.
    HBond {
        /// Reduced distance where the term reaches 1.
        good: f64,
        /// Reduced distance where the term reaches 0.
        bad: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// Linear attraction used to "glue" macrocycle closure pseudo-atoms.
    LinearAttraction {
        /// Interaction cutoff (脜); deliberately longer than the other terms.
        cutoff: f64,
    },
    /// AD4 Lennard-Jones 12-6 van der Waals term.
    Ad4Vdw {
        /// Distance smoothing full width.
        smoothing: f64,
        /// Energy cap.
        cap: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// AD4 12-10 hydrogen-bond term.
    Ad4Hb {
        /// Distance smoothing full width.
        smoothing: f64,
        /// Energy cap.
        cap: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// AD4 distance-dependent-dielectric Coulomb term.
    Ad4Elec {
        /// Energy cap.
        cap: f64,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
    /// AD4 desolvation term.
    Ad4Solv {
        /// Gaussian width of the desolvation shell (脜).
        sigma: f64,
        /// Charge-dependent solvation coefficient.
        solvation_q: f64,
        /// Whether the charge-dependent part is active.
        charge_dependent: bool,
        /// Interaction cutoff (脜).
        cutoff: f64,
    },
}

impl Term {
    /// The term's interaction cutoff in 脜.
    pub fn cutoff(&self) -> f64 {
        match *self {
            Term::Gauss { cutoff, .. }
            | Term::Repulsion { cutoff, .. }
            | Term::Hydrophobic { cutoff, .. }
            | Term::HBond { cutoff, .. }
            | Term::LinearAttraction { cutoff }
            | Term::Ad4Vdw { cutoff, .. }
            | Term::Ad4Hb { cutoff, .. }
            | Term::Ad4Elec { cutoff, .. }
            | Term::Ad4Solv { cutoff, .. } => cutoff,
        }
    }

    #[inline]
    fn optimal(&self, t1: XsType, t2: XsType, radii: Radii) -> f64 {
        match radii {
            Radii::Vina => optimal_distance(t1, t2),
            Radii::Vinardo => optimal_distance_vinardo(t1, t2),
        }
    }

    /// Evaluate an X-Score-based term.
    pub fn eval_xs(&self, t1: XsType, t2: XsType, r: f64, radii: Radii) -> f64 {
        self.eval_xs_deriv(t1, t2, r, radii).0
    }

    /// Evaluate an X-Score-based term and its derivative `dE/dr`.
    pub fn eval_xs_deriv(&self, t1: XsType, t2: XsType, r: f64, radii: Radii) -> (f64, f64) {
        match *self {
            Term::Gauss {
                offset,
                width,
                cutoff,
            } => {
                if r >= cutoff {
                    return (0.0, 0.0);
                }
                let u = r - (self.optimal(t1, t2, radii) + offset);
                let e = (-sqr(u / width)).exp();
                (e, e * (-2.0 * u / sqr(width)))
            }
            Term::Repulsion { offset, cutoff } => {
                if r >= cutoff {
                    return (0.0, 0.0);
                }
                let d = r - (self.optimal(t1, t2, radii) + offset);
                if d > 0.0 {
                    (0.0, 0.0)
                } else {
                    (d * d, 2.0 * d)
                }
            }
            Term::Hydrophobic { good, bad, cutoff } => {
                if r >= cutoff || !(t1.is_hydrophobic() && t2.is_hydrophobic()) {
                    return (0.0, 0.0);
                }
                let x = r - self.optimal(t1, t2, radii);
                (slope_step(bad, good, x), slope_step_deriv(bad, good, x))
            }
            Term::HBond { good, bad, cutoff } => {
                if r >= cutoff || !t1.h_bond_possible(t2) {
                    return (0.0, 0.0);
                }
                let x = r - self.optimal(t1, t2, radii);
                (slope_step(bad, good, x), slope_step_deriv(bad, good, x))
            }
            Term::LinearAttraction { cutoff } => {
                if r >= cutoff || !is_glued(t1, t2) {
                    (0.0, 0.0)
                } else {
                    (r, 1.0)
                }
            }
            _ => (0.0, 0.0),
        }
    }

    /// Evaluate an AD4 term. `a` and `b` supply the AD4 type and charge.
    pub fn eval_ad4(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        self.eval_ad4_deriv(a, b, r).0
    }

    /// Evaluate an AD4 term and its derivative `dE/dr`.
    pub fn eval_ad4_deriv(&self, a: &Atom, b: &Atom, r: f64) -> (f64, f64) {
        match *self {
            Term::Ad4Vdw {
                smoothing,
                cap,
                cutoff,
            } => {
                if r >= cutoff {
                    return (0.0, 0.0);
                }
                let pa = ad_type_property(a.ad);
                let pb = ad_type_property(b.ad);
                let hb_depth = pa.hb_depth * pb.hb_depth;
                if hb_depth < 0.0 {
                    return (0.0, 0.0); // this pair is an H-bond, not a vdW contact
                }
                let vdw_rij = pa.radius + pb.radius;
                let vdw_depth = (pa.depth * pb.depth).sqrt();
                let (r_s, dp) = smoothen(r, vdw_rij, smoothing);
                let c12 = vdw_rij.powi(12) * vdw_depth;
                let c6 = vdw_rij.powi(6) * vdw_depth * 2.0;
                if r_s > EPSILON {
                    let r6 = r_s.powi(6);
                    let r12 = r_s.powi(12);
                    let e = c12 / r12 - c6 / r6;
                    if e >= cap {
                        (cap, 0.0)
                    } else {
                        let de = (-12.0 * c12 / (r12 * r_s) + 6.0 * c6 / (r6 * r_s)) * dp;
                        (e, de)
                    }
                } else {
                    (cap, 0.0)
                }
            }
            Term::Ad4Hb {
                smoothing,
                cap,
                cutoff,
            } => {
                if r >= cutoff {
                    return (0.0, 0.0);
                }
                let pa = ad_type_property(a.ad);
                let pb = ad_type_property(b.ad);
                let hb_rij = pa.hb_radius + pb.hb_radius;
                let hb_depth = pa.hb_depth * pb.hb_depth;
                if hb_depth >= 0.0 {
                    return (0.0, 0.0); // this pair is a vdW contact, not an H-bond
                }
                let (r_s, dp) = smoothen(r, hb_rij, smoothing);
                let d = -hb_depth;
                let c12 = hb_rij.powi(12) * d * 10.0 / 2.0;
                let c10 = hb_rij.powi(10) * d * 12.0 / 2.0;
                if r_s > EPSILON {
                    let r10 = r_s.powi(10);
                    let r12 = r_s.powi(12);
                    let e = c12 / r12 - c10 / r10;
                    if e >= cap {
                        (cap, 0.0)
                    } else {
                        let de = (-12.0 * c12 / (r12 * r_s) + 10.0 * c10 / (r10 * r_s)) * dp;
                        (e, de)
                    }
                } else {
                    (cap, 0.0)
                }
            }
            Term::Ad4Elec { cap, cutoff } => {
                if r >= cutoff {
                    return (0.0, 0.0);
                }
                let q1q2 = a.charge * b.charge * 332.0;
                let (diel, ddiel) = dielectric(r);
                if r < EPSILON {
                    let e = if diel.abs() > EPSILON {
                        q1q2 * cap / diel
                    } else {
                        0.0
                    };
                    (e, 0.0)
                } else {
                    let raw = 1.0 / (r * diel);
                    if raw >= cap {
                        (q1q2 * cap, 0.0)
                    } else {
                        // E = q1q2 / (r路diel)
                        let e = q1q2 * raw;
                        let de = q1q2 * (-raw / r - ddiel / (r * diel * diel));
                        (e, de)
                    }
                }
            }
            Term::Ad4Solv {
                sigma,
                solvation_q,
                charge_dependent,
                cutoff,
            } => {
                if r >= cutoff {
                    return (0.0, 0.0);
                }
                let s1 = solvation_parameter(a);
                let s2 = solvation_parameter(b);
                let v1 = atomic_volume(a);
                let v2 = atomic_volume(b);
                let mq = if charge_dependent { solvation_q } else { 0.0 };
                let c = (s1 + mq * a.charge.abs()) * v2 + (s2 + mq * b.charge.abs()) * v1;
                let x = r / sigma;
                let g = (-0.5 * x * x).exp();
                let e = c * g;
                let de = e * (-r / sqr(sigma));
                (e, de)
            }
            Term::LinearAttraction { cutoff } => {
                if r >= cutoff || !is_glued(a.xs, b.xs) {
                    (0.0, 0.0)
                } else {
                    (r, 1.0)
                }
            }
            _ => (0.0, 0.0),
        }
    }
}

/// Analytic derivative of `slope_step` with respect to `x`.
#[inline]
fn slope_step_deriv(x_bad: f64, x_good: f64, x: f64) -> f64 {
    let (lo, hi) = if x_bad < x_good {
        (x_bad, x_good)
    } else {
        (x_good, x_bad)
    };
    if x <= lo || x >= hi {
        0.0
    } else {
        1.0 / (x_good - x_bad)
    }
}

/// Vina's `smoothen`: shift `r` away from a plateau of full width
/// `smoothing`. Returns the shifted radius and `d(shifted)/dr`.
#[inline]
fn smoothen(r: f64, rij: f64, smoothing: f64) -> (f64, f64) {
    let half = smoothing * 0.5;
    if half <= 0.0 {
        return (r, 1.0);
    }
    if r > rij + half {
        (r - half, 1.0)
    } else if r < rij - half {
        (r + half, 1.0)
    } else {
        (rij, 0.0)
    }
}

/// AutoDock 4's distance-dependent dielectric and its derivative.
#[inline]
fn dielectric(r: f64) -> (f64, f64) {
    const B: f64 = 78.4 + 8.5525;
    const LB: f64 = -B * 0.003627;
    let e = 7.7839 * (LB * r).exp();
    let s = 1.0 + e;
    let diel = -8.5525 + B / s;
    let ddiel = -B / (s * s) * e * LB;
    (diel, ddiel)
}

/// AD4 solvation parameter of an atom.
#[inline]
pub fn solvation_parameter(a: &Atom) -> f64 {
    if a.ad != AdType::Unknown {
        ad_type_property(a.ad).solvation
    } else if a.xs == XsType::MetD {
        METAL_SOLVATION_PARAMETER
    } else {
        0.0
    }
}

/// AD4 atomic volume of an atom (脜鲁).
#[inline]
pub fn atomic_volume(a: &Atom) -> f64 {
    if a.ad != AdType::Unknown {
        ad_type_property(a.ad).volume
    } else {
        4.0 * PI / 3.0 * a.xs.radius().powi(3)
    }
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// Summed energy components of a pose, in kcal/mol.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScoreComponents {
    /// The value the search minimises, including the torsional penalty.
    pub total: f64,
    /// Ligand鈥搑eceptor energy.
    pub inter: f64,
    /// Ligand internal energy.
    pub intra: f64,
    /// Torsional (conformation-independent) penalty.
    pub conf_independent: f64,
    /// Intramolecular energy of the unbound ligand (Vina subtracts it).
    pub unbound: f64,
}

impl ScoreComponents {
    /// The reported binding affinity.
    pub fn affinity(&self) -> f64 {
        self.total
    }
}

/// The common interface implemented by every force field.
///
/// The interface is *term-oriented*: callers can ask for one potential at a
/// time (which is what the affinity-grid builder needs) or for the summed pair
/// energy, always together with an exact first derivative.
pub trait ScoringFunction: Send + Sync + fmt::Debug {
    /// Which force field this is.
    fn choice(&self) -> SfChoice;

    /// The term weights in use.
    fn weights(&self) -> &Weights;

    /// Number of distance-dependent terms.
    fn num_terms(&self) -> usize;

    /// The interaction cutoff below which the terms are non-zero (脜).
    fn cutoff(&self) -> f64;

    /// The largest cutoff of any term, including the macrocycle glue term (脜).
    fn max_cutoff(&self) -> f64;

    /// Evaluate term `k` for an X-Score-typed pair.
    fn eval_term(&self, k: usize, t1: XsType, t2: XsType, r: f64) -> f64;

    /// Evaluate term `k` and its radial derivative for an X-Score-typed pair.
    fn eval_term_deriv(&self, k: usize, t1: XsType, t2: XsType, r: f64) -> (f64, f64);

    /// Evaluate term `k` for a pair of typed atoms (AD4 needs charges).
    fn eval_term_atoms(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> f64;

    /// Evaluate term `k` and its radial derivative for a pair of typed atoms.
    fn eval_term_atoms_deriv(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> (f64, f64);

    /// The conformation-independent (torsional) term.
    fn conf_independent(&self, e: f64, num_tors: f64) -> f64;

    /// True when the force field can be evaluated through an affinity grid.
    fn is_grid_capable(&self) -> bool {
        true
    }

    /// True when the force field is parameterised by X-Score atom types.
    ///
    /// For such a force field hydrogens carry no type and are skipped entirely:
    /// they influence the energy only indirectly, by making their heavy
    /// neighbour an H-bond donor.
    fn is_xs_typed(&self) -> bool {
        true
    }

    /// Summed pair energy.
    fn pair_energy(&self, a: &Atom, b: &Atom, r: f64) -> f64 {
        let w = &self.weights().terms;
        let mut e = 0.0;
        for k in 0..self.num_terms() {
            e += w[k] * self.eval_term_atoms(k, a, b, r);
        }
        e
    }

    /// Summed pair energy and its radial derivative `dE/dr`.
    ///
    /// The derivative is the exact analytic derivative of the returned energy
    /// for every `r < cutoff`. At the cutoff itself each term is discontinuous
    /// by construction 鈥?the same situation as in the reference
    /// implementations, which work around it by tabulating the potential and
    /// differentiating the table. The residual step in the Vina force field is
    /// the `gauss2` term, `w2路exp(-1/4) 鈮?-0.004` kcal/mol at 8 脜.
    fn pair_energy_deriv(&self, a: &Atom, b: &Atom, r: f64) -> (f64, f64) {
        let w = &self.weights().terms;
        let mut e = 0.0;
        let mut de = 0.0;
        for k in 0..self.num_terms() {
            let (ek, dek) = self.eval_term_atoms_deriv(k, a, b, r);
            e += w[k] * ek;
            de += w[k] * dek;
        }
        (e, de)
    }

    /// Summed pair energy for X-Score types only.
    fn pair_energy_xs(&self, t1: XsType, t2: XsType, r: f64) -> f64 {
        let w = &self.weights().terms;
        let mut e = 0.0;
        for k in 0..self.num_terms() {
            e += w[k] * self.eval_term(k, t1, t2, r);
        }
        e
    }
}

// ---------------------------------------------------------------------------
// Concrete force fields
// ---------------------------------------------------------------------------

/// X-Score-based force field driver shared by Vina and Vinardo.
#[derive(Debug, Clone)]
pub struct XsScoringFunction {
    choice: SfChoice,
    weights: Weights,
    terms: Vec<Term>,
    radii: Radii,
    cutoff: f64,
    max_cutoff: f64,
}

impl XsScoringFunction {
    /// AutoDock Vina.
    pub fn vina(weights: Weights) -> XsScoringFunction {
        XsScoringFunction {
            choice: SfChoice::Vina,
            weights,
            terms: vec![
                Term::Gauss {
                    offset: 0.0,
                    width: 0.5,
                    cutoff: 8.0,
                },
                Term::Gauss {
                    offset: 3.0,
                    width: 2.0,
                    cutoff: 8.0,
                },
                Term::Repulsion {
                    offset: 0.0,
                    cutoff: 8.0,
                },
                Term::Hydrophobic {
                    good: 0.5,
                    bad: 1.5,
                    cutoff: 8.0,
                },
                Term::HBond {
                    good: -0.7,
                    bad: 0.0,
                    cutoff: 8.0,
                },
                Term::LinearAttraction { cutoff: 20.0 },
            ],
            radii: Radii::Vina,
            cutoff: 8.0,
            max_cutoff: 20.0,
        }
    }

    /// Vinardo.
    pub fn vinardo(weights: Weights) -> XsScoringFunction {
        XsScoringFunction {
            choice: SfChoice::Vinardo,
            weights,
            terms: vec![
                Term::Gauss {
                    offset: 0.0,
                    width: 0.8,
                    cutoff: 8.0,
                },
                Term::Repulsion {
                    offset: 0.0,
                    cutoff: 8.0,
                },
                Term::Hydrophobic {
                    good: 0.0,
                    bad: 2.5,
                    cutoff: 8.0,
                },
                Term::HBond {
                    good: -0.6,
                    bad: 0.0,
                    cutoff: 8.0,
                },
                Term::LinearAttraction { cutoff: 20.0 },
            ],
            radii: Radii::Vinardo,
            cutoff: 8.0,
            max_cutoff: 20.0,
        }
    }

    /// The terms in evaluation order.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// The radii table in use.
    pub fn radii(&self) -> Radii {
        self.radii
    }
}

impl ScoringFunction for XsScoringFunction {
    fn choice(&self) -> SfChoice {
        self.choice
    }

    fn weights(&self) -> &Weights {
        &self.weights
    }

    fn num_terms(&self) -> usize {
        self.terms.len()
    }

    fn cutoff(&self) -> f64 {
        self.cutoff
    }

    fn max_cutoff(&self) -> f64 {
        self.max_cutoff
    }

    fn eval_term(&self, k: usize, t1: XsType, t2: XsType, r: f64) -> f64 {
        match self.terms.get(k) {
            Some(t) => t.eval_xs(t1, t2, r, self.radii),
            None => 0.0,
        }
    }

    fn eval_term_deriv(&self, k: usize, t1: XsType, t2: XsType, r: f64) -> (f64, f64) {
        match self.terms.get(k) {
            Some(t) => t.eval_xs_deriv(t1, t2, r, self.radii),
            None => (0.0, 0.0),
        }
    }

    fn eval_term_atoms(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> f64 {
        self.eval_term(k, a.xs, b.xs, r)
    }

    fn eval_term_atoms_deriv(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> (f64, f64) {
        self.eval_term_deriv(k, a.xs, b.xs, r)
    }

    fn conf_independent(&self, e: f64, num_tors: f64) -> f64 {
        // Vina's `num_tors_div`: E / (1 + weight 路 num_tors / 5) with
        // weight = 0.1 路 (5路rot/0.1) = 5路rot, hence E / (1 + rot 路 num_tors).
        if num_tors.abs() < EPSILON {
            return e;
        }
        e / (1.0 + self.weights.rot * num_tors)
    }
}

/// The AutoDock 4.2 force field.
#[derive(Debug, Clone)]
pub struct Ad4ScoringFunction {
    weights: Weights,
    terms: Vec<Term>,
}

impl Ad4ScoringFunction {
    /// AutoDock 4.2 with the given weights.
    pub fn new(weights: Weights) -> Ad4ScoringFunction {
        Ad4ScoringFunction {
            weights,
            terms: vec![
                Term::Ad4Vdw {
                    smoothing: 0.5,
                    cap: 100_000.0,
                    cutoff: 8.0,
                },
                Term::Ad4Hb {
                    smoothing: 0.5,
                    cap: 100_000.0,
                    cutoff: 8.0,
                },
                Term::Ad4Elec {
                    cap: 100.0,
                    cutoff: 20.48,
                },
                Term::Ad4Solv {
                    sigma: 3.6,
                    solvation_q: 0.01097,
                    charge_dependent: true,
                    cutoff: 20.48,
                },
                Term::LinearAttraction { cutoff: 20.0 },
            ],
        }
    }

    /// The terms in evaluation order.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }
}

impl ScoringFunction for Ad4ScoringFunction {
    fn choice(&self) -> SfChoice {
        SfChoice::Ad4
    }

    fn weights(&self) -> &Weights {
        &self.weights
    }

    fn num_terms(&self) -> usize {
        self.terms.len()
    }

    fn cutoff(&self) -> f64 {
        20.48
    }

    fn max_cutoff(&self) -> f64 {
        20.48
    }

    fn eval_term(&self, k: usize, _t1: XsType, _t2: XsType, _r: f64) -> f64 {
        // AD4 is not expressible in terms of X-Score types alone; callers must
        // use `eval_term_atoms`.
        let _ = k;
        0.0
    }

    fn eval_term_deriv(&self, k: usize, _t1: XsType, _t2: XsType, _r: f64) -> (f64, f64) {
        let _ = k;
        (0.0, 0.0)
    }

    fn eval_term_atoms(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> f64 {
        match self.terms.get(k) {
            Some(t) => t.eval_ad4(a, b, r),
            None => 0.0,
        }
    }

    fn eval_term_atoms_deriv(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> (f64, f64) {
        match self.terms.get(k) {
            Some(t) => t.eval_ad4_deriv(a, b, r),
            None => (0.0, 0.0),
        }
    }

    fn conf_independent(&self, e: f64, num_tors: f64) -> f64 {
        // Vina's `ad4_tors_add`: E + weight 路 torsdof.
        e + self.weights.rot * num_tors
    }

    fn is_grid_capable(&self) -> bool {
        false
    }

    fn is_xs_typed(&self) -> bool {
        false
    }
}

/// Construct a force field from a choice and optional weights.
pub fn make_scoring_function(
    choice: SfChoice,
    weights: Option<Weights>,
) -> Arc<dyn ScoringFunction> {
    let w = weights.unwrap_or_else(|| Weights::default_for(choice));
    match choice {
        SfChoice::Vina => Arc::new(XsScoringFunction::vina(w)),
        SfChoice::Vinardo => Arc::new(XsScoringFunction::vinardo(w)),
        SfChoice::Ad4 => Arc::new(Ad4ScoringFunction::new(w)),
    }
}

// ---------------------------------------------------------------------------
// Macrocycle closure bookkeeping
// ---------------------------------------------------------------------------

/// Vina's `is_glue_pair`: a closure dummy (`G0`..`G3`) paired with its own
/// closure carbon (`CG0`..`CG3`).
pub fn is_glue_pair(a: &Atom, b: &Atom) -> bool {
    matches!(
        (a.ad, b.ad),
        (AdType::Cg0, AdType::G0)
            | (AdType::G0, AdType::Cg0)
            | (AdType::Cg1, AdType::G1)
            | (AdType::G1, AdType::Cg1)
            | (AdType::Cg2, AdType::G2)
            | (AdType::G2, AdType::Cg2)
            | (AdType::Cg3, AdType::G3)
            | (AdType::G3, AdType::Cg3)
    )
}

/// Vina's `is_unmatched_closure_dummy`: a closure dummy paired with something
/// other than its own closure carbon.
pub fn is_unmatched_closure_dummy(a: &Atom, b: &Atom) -> bool {
    let pair = |x: AdType, y: AdType| match x {
        AdType::G0 => y != AdType::Cg0,
        AdType::G1 => y != AdType::Cg1,
        AdType::G2 => y != AdType::Cg2,
        AdType::G3 => y != AdType::Cg3,
        _ => false,
    };
    pair(a.ad, b.ad) || pair(b.ad, a.ad)
}

/// Vina's `is_closure_clash`: two atoms that belong to the same macrocycle
/// closure ring but are not the glued pair itself.
pub fn is_closure_clash(atoms: &[Atom], i: usize, j: usize) -> bool {
    if is_glue_pair(&atoms[i], &atoms[j]) {
        return false;
    }
    let has_cg = |a: &Atom, which: AdType| -> bool {
        a.bonds.iter().any(|b| atoms[b.other].ad == which)
    };
    let i_cg: [bool; 4] = [
        has_cg(&atoms[i], AdType::Cg0),
        has_cg(&atoms[i], AdType::Cg1),
        has_cg(&atoms[i], AdType::Cg2),
        has_cg(&atoms[i], AdType::Cg3),
    ];
    for b in &atoms[j].bonds {
        let t = atoms[b.other].ad;
        let hit = match t {
            AdType::G0 => i_cg[0],
            AdType::G1 => i_cg[1],
            AdType::G2 => i_cg[2],
            AdType::G3 => i_cg[3],
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Torsion counting
// ---------------------------------------------------------------------------

/// Vina's `num_tors`: the effective torsion count used by the torsional
/// penalty. Counts `0.5` for every rotatable-bond endpoint whose partner has
/// more than one heavy neighbour, which excludes terminal groups such as
/// `-CH3` from contributing a full rotor.
///
/// `rotors` holds `(parent atom index, attachment atom index)` pairs; the
/// parent index is `None` when the rotatable bond points into an immobile
/// receptor atom.
pub fn num_tors(ligand_atoms: &[Atom], rotors: &[(Option<usize>, usize)]) -> f64 {
    let heavy_degree = |i: usize| -> usize {
        ligand_atoms[i]
            .bonds
            .iter()
            .filter(|b: &&Bond| !ligand_atoms[b.other].is_hydrogen())
            .count()
    };
    let mut n = 0.0;
    for &(parent, attach) in rotors {
        let attach_heavy = !ligand_atoms[attach].is_hydrogen();
        if attach_heavy && heavy_degree(attach) > 1 {
            n += 0.5;
        }
        if let Some(p) = parent {
            if !ligand_atoms[p].is_hydrogen() && attach_heavy && heavy_degree(p) > 1 {
                n += 0.5;
            }
        }
    }
    n
}

/// Count the heavy atoms of a molecule.
pub fn num_heavy_atoms(mol: &Molecule) -> usize {
    mol.atoms.iter().filter(|a| !a.is_hydrogen()).count()
}

/// Sentinel used by the search to disable an energy cap.
pub const NO_CAP: f64 = MAX_F;

/// Access the AD4 parameter table by index, clamped to the valid range.
#[inline]
pub fn ad_kind(index: usize) -> crate::atom::AtomKind {
    ATOM_KIND_DATA[index.min(ATOM_KIND_DATA.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::DVec3;

    fn atom(ad: AdType, charge: f64) -> Atom {
        Atom::new(DVec3::ZERO, ad, charge)
    }

    #[test]
    fn vina_defaults_match_the_reference() {
        let w = Weights::vina_default();
        assert_eq!(w.terms.len(), 6);
        assert!((w.terms[0] - -0.035579).abs() < 1e-12);
        assert!((w.terms[4] - -0.587439).abs() < 1e-12);
        assert!((w.rot - 0.05846).abs() < 1e-12);
    }

    #[test]
    fn gauss_terms_have_known_values() {
        let sf = XsScoringFunction::vina(Weights::vina_default());
        // Carbon鈥揷arbon: optimal distance 3.8 脜.
        assert!((sf.eval_term(0, XsType::CH, XsType::CH, 3.8) - 1.0).abs() < 1e-12);
        assert!((sf.eval_term(1, XsType::CH, XsType::CH, 6.8) - 1.0).abs() < 1e-12);
        assert_eq!(sf.eval_term(2, XsType::CH, XsType::CH, 3.9), 0.0);
        assert!((sf.eval_term(2, XsType::CH, XsType::CH, 3.3) - 0.25).abs() < 1e-12);
        for k in 0..6 {
            assert_eq!(sf.eval_term(k, XsType::CH, XsType::CH, 8.5), 0.0);
        }
    }

    #[test]
    fn hydrophobic_and_hbond_ramps() {
        let sf = XsScoringFunction::vina(Weights::vina_default());
        assert!((sf.eval_term(3, XsType::CH, XsType::CH, 3.8) - 1.0).abs() < 1e-12);
        assert!((sf.eval_term(3, XsType::CH, XsType::CH, 4.3) - 1.0).abs() < 1e-12);
        assert!((sf.eval_term(3, XsType::CH, XsType::CH, 4.8) - 0.5).abs() < 1e-12);
        assert_eq!(sf.eval_term(3, XsType::CH, XsType::CH, 5.3), 0.0);
        assert_eq!(sf.eval_term(3, XsType::CH, XsType::OA, 3.8), 0.0);
        let opt = optimal_distance(XsType::ND, XsType::OA);
        assert!((sf.eval_term(4, XsType::ND, XsType::OA, opt)).abs() < 1e-12);
        assert!((sf.eval_term(4, XsType::ND, XsType::OA, opt - 0.7) - 1.0).abs() < 1e-12);
        assert!((sf.eval_term(4, XsType::ND, XsType::OA, opt - 0.35) - 0.5).abs() < 1e-12);
    }

    /// Central-difference validation of every analytic radial derivative.
    #[test]
    fn analytic_radial_derivatives_match_finite_differences() {
        let cases: Vec<(&str, Box<dyn ScoringFunction>)> = vec![
            (
                "vina",
                Box::new(XsScoringFunction::vina(Weights::vina_default())),
            ),
            (
                "vinardo",
                Box::new(XsScoringFunction::vinardo(Weights::vinardo_default())),
            ),
            (
                "ad4",
                Box::new(Ad4ScoringFunction::new(Weights::ad4_default())),
            ),
        ];
        let pairs = [
            (XsType::CH, XsType::CH),
            (XsType::CH, XsType::OA),
            (XsType::ND, XsType::OA),
            (XsType::CP, XsType::NA),
            (XsType::SP, XsType::CH),
        ];
        let ad_pairs = [
            (AdType::C, AdType::C, 0.0, 0.0),
            (AdType::C, AdType::OA, 0.2, -0.4),
            (AdType::NA, AdType::HD, -0.3, 0.2),
            (AdType::A, AdType::SA, 0.1, -0.1),
        ];
        for (name, sf) in &cases {
            if sf.choice() != SfChoice::Ad4 {
                for (t1, t2) in pairs {
                    for step in 0..=200 {
                        let r = 0.5 + step as f64 * 0.035;
                        if r >= sf.cutoff() - 1e-6 {
                            break;
                        }
                        let mut a1 = atom(AdType::C, 0.0);
                        a1.xs = t1;
                        let mut b1 = atom(AdType::C, 0.0);
                        b1.xs = t2;
                        let (_, de) = sf.pair_energy_deriv(&a1, &b1, r);
                        let h = 1e-6;
                        let e0 = sf.pair_energy(&a1, &b1, r);
                        let e1 = sf.pair_energy(&a1, &b1, r + h);
                        let e2 = sf.pair_energy(&a1, &b1, r - h);
                        let fwd = (e1 - e0) / h;
                        let bwd = (e0 - e2) / h;
                        // `slope_step` creates genuine derivative kinks at
                        // r = optimal + good/bad. Skip only those isolated
                        // points; everywhere else the analytic derivative must
                        // agree with the centred difference.
                        if (fwd - bwd).abs() > 1e-3 {
                            continue;
                        }
                        let numeric = (e1 - e2) / (2.0 * h);
                        assert!(
                            (de - numeric).abs() < 2e-5 * (1.0 + numeric.abs()),
                            "{name} {t1:?}/{t2:?} r={r}: analytic {de} vs numeric {numeric}"
                        );
                    }
                }
            } else {
                for (ad1, ad2, q1, q2) in ad_pairs {
                    for step in 1..=200 {
                        let r = 0.5 + step as f64 * 0.08;
                        if r >= sf.cutoff() - 1e-6 {
                            break;
                        }
                        let a1 = atom(ad1, q1);
                        let b1 = atom(ad2, q2);
                        let (_, de) = sf.pair_energy_deriv(&a1, &b1, r);
                        let h = 1e-6;
                        let e1 = sf.pair_energy(&a1, &b1, r + h);
                        let e2 = sf.pair_energy(&a1, &b1, r - h);
                        let numeric = (e1 - e2) / (2.0 * h);
                        if numeric.abs() > 1e-3 || de.abs() > 1e-3 {
                            assert!(
                                (de - numeric).abs() < 5e-3 * (1.0 + numeric.abs()),
                                "ad4 {ad1:?}/{ad2:?} r={r}: analytic {de} vs numeric {numeric}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn conf_independent_reproduces_the_published_penalty() {
        let sf = XsScoringFunction::vina(Weights::vina_default());
        let e = sf.conf_independent(-10.0, 4.0);
        let expect = -10.0 / (1.0 + 0.05846 * 4.0);
        assert!((e - expect).abs() < 1e-12);
        assert!((e - -8.1048).abs() < 1e-4, "{e}");
    }

    #[test]
    fn ad4_torsion_term_is_additive() {
        let sf = Ad4ScoringFunction::new(Weights::ad4_default());
        assert!((sf.conf_independent(1.5, 4.0) - (1.5 + 0.2983 * 4.0)).abs() < 1e-12);
    }

    #[test]
    fn dielectric_limits() {
        let (d0, _) = dielectric(0.0);
        assert!((d0 - 1.3465).abs() < 1e-3, "{d0}");
        let (dinf, _) = dielectric(1000.0);
        assert!((dinf - 78.4).abs() < 1e-3, "{dinf}");
    }

    #[test]
    fn num_tors_matches_vina_semantics() {
        let mut mol = Molecule::from_atoms(vec![
            Atom::new(DVec3::new(0.0, 0.0, 0.0), AdType::C, 0.0),
            Atom::new(DVec3::new(1.52, 0.0, 0.0), AdType::C, 0.0),
            Atom::new(DVec3::new(3.04, 0.0, 0.0), AdType::C, 0.0),
            Atom::new(DVec3::new(4.56, 0.0, 0.0), AdType::C, 0.0),
            Atom::new(DVec3::new(-0.55, 0.9, 0.0), AdType::H, 0.0),
            Atom::new(DVec3::new(-0.55, -0.5, 0.85), AdType::H, 0.0),
            Atom::new(DVec3::new(-0.55, -0.5, -0.85), AdType::H, 0.0),
            Atom::new(DVec3::new(5.11, 0.9, 0.0), AdType::H, 0.0),
            Atom::new(DVec3::new(5.11, -0.5, 0.85), AdType::H, 0.0),
            Atom::new(DVec3::new(5.11, -0.5, -0.85), AdType::H, 0.0),
        ]);
        mol.perceive();
        let n = num_tors(&mol.atoms, &[(Some(1), 2)]);
        assert!((n - 1.0).abs() < 1e-12, "{n}");
    }

    #[test]
    fn glue_pairs_are_recognised() {
        let a = atom(AdType::Cg0, 0.0);
        let g = atom(AdType::G0, 0.0);
        let c = atom(AdType::C, 0.0);
        assert!(is_glue_pair(&a, &g));
        assert!(!is_glue_pair(&a, &c));
        assert!(is_unmatched_closure_dummy(&g, &c));
        assert!(!is_unmatched_closure_dummy(&g, &a));
    }

    #[test]
    fn smoothen_plateau() {
        assert_eq!(smoothen(3.0, 3.0, 0.5), (3.0, 0.0));
        assert_eq!(smoothen(4.0, 3.0, 0.5), (3.75, 1.0));
        assert_eq!(smoothen(2.0, 3.0, 0.5), (2.25, 1.0));
    }
}

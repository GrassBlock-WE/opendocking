// SPDX-License-Identifier: GPL-3.0-or-later
//! Atom typing and force-field parameters.
//!
//! Two typing schemes are supported, exactly as in AutoDock Vina:
//!
//! * [`AdType`] — the AutoDock 4 typing used by the PDBQT `partial charge` /
//!   atom-name column. Required for the AD4.2 force field and for parsing PDBQT.
//! * [`XsType`] — the X-Score typing used by the Vina / Vinardo force fields.
//!   It refines `AdType` with explicit donor/acceptor flags that are *derived
//!   from the bonding graph*, not from the file.
//!
//! The numeric tables below are ported from AutoDock Vina's `atom_constants.h`
//! (Copyright (c) 2006-2010, The Scripps Research Institute, Apache-2.0), which
//! is itself "generated from edited AD4_parameters.data using a script".
//!
//! ## Index stability warning
//!
//! The discriminants of [`AdType`] and [`XsType`] are part of the PDBQT wire
//! format and of the lookup-table layout. **Never reorder them.**

use std::fmt;

// ---------------------------------------------------------------------------
// Element
// ---------------------------------------------------------------------------

/// Chemical element as far as the force fields care.
///
/// Discriminants follow Vina's `EL_TYPE_*` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Element {
    /// Hydrogen.
    H = 0,
    /// Carbon.
    C = 1,
    /// Nitrogen.
    N = 2,
    /// Oxygen.
    O = 3,
    /// Sulfur.
    S = 4,
    /// Phosphorus.
    P = 5,
    /// Fluorine.
    F = 6,
    /// Chlorine.
    Cl = 7,
    /// Bromine.
    Br = 8,
    /// Iodine.
    I = 9,
    /// Silicon.
    Si = 10,
    /// Astatine.
    At = 11,
    /// Any metal.
    Met = 12,
    /// Pseudo-atom (macrocycle glue / dummy).
    Dummy = 13,
    /// Out-of-range sentinel.
    Size = 14,
}

impl Element {
    /// Number of element kinds (including the sentinel).
    pub const COUNT: usize = 15;

    /// Parse an element symbol, returning `None` when unknown.
    pub fn from_symbol(s: &str) -> Option<Element> {
        let t = s.trim();
        // PDBQT/PDB names carry the element in the first columns; accept both
        // "CA" (alpha carbon) style and bare element symbols.
        let cleaned: String = t.chars().take_while(|c| !c.is_ascii_digit()).collect();
        let up = cleaned.to_ascii_uppercase();
        let two = if up.len() >= 2 { &up[..2] } else { "" };
        let one = if up.is_empty() { "" } else { &up[..1] };
        match two {
            "CL" => Some(Element::Cl),
            "BR" => Some(Element::Br),
            "SI" => Some(Element::Si),
            "AT" => Some(Element::At),
            _ => match one {
                "H" | "D" => Some(Element::H),
                "C" => Some(Element::C),
                "N" => Some(Element::N),
                "O" => Some(Element::O),
                "S" => Some(Element::S),
                "P" => Some(Element::P),
                "F" => Some(Element::F),
                "I" => Some(Element::I),
                _ => None,
            },
        }
    }

    /// `true` for elements that are not C and not H.
    #[inline]
    pub fn is_heteroatom(self) -> bool {
        !matches!(self, Element::C | Element::H)
    }

    /// Single-letter-ish symbol for output.
    pub fn symbol(self) -> &'static str {
        match self {
            Element::H => "H",
            Element::C => "C",
            Element::N => "N",
            Element::O => "O",
            Element::S => "S",
            Element::P => "P",
            Element::F => "F",
            Element::Cl => "Cl",
            Element::Br => "Br",
            Element::I => "I",
            Element::Si => "Si",
            Element::At => "At",
            Element::Met => "Met",
            Element::Dummy => "Du",
            Element::Size => "X",
        }
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

// ---------------------------------------------------------------------------
// AD4 atom types
// ---------------------------------------------------------------------------

/// AutoDock 4 atom type (the PDBQT atom-type column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum AdType {
    /// Aliphatic carbon (used in PDBQT as `C`).
    C = 0,
    /// Aromatic carbon (`A`).
    A = 1,
    /// Nitrogen (`N`).
    N = 2,
    /// Oxygen (`O`).
    O = 3,
    /// Phosphorus (`P`).
    P = 4,
    /// Sulfur (`S`).
    S = 5,
    /// Non-polar hydrogen (`H`).
    H = 6,
    /// Fluorine (`F`).
    F = 7,
    /// Iodine (`I`).
    I = 8,
    /// H-bond-accepting nitrogen (`NA`).
    NA = 9,
    /// H-bond-accepting oxygen (`OA`).
    OA = 10,
    /// H-bond-accepting sulfur (`SA`).
    SA = 11,
    /// Polar hydrogen (`HD`).
    HD = 12,
    /// Magnesium (`Mg`).
    Mg = 13,
    /// Manganese (`Mn`).
    Mn = 14,
    /// Zinc (`Zn`).
    Zn = 15,
    /// Calcium (`Ca`).
    Ca = 16,
    /// Iron (`Fe`).
    Fe = 17,
    /// Chlorine (`Cl`).
    Cl = 18,
    /// Bromine (`Br`).
    Br = 19,
    /// Silicon (`Si`).
    Si = 20,
    /// Astatine (`At`).
    At = 21,
    /// Macrocycle closure dummy 0 (`G0`).
    G0 = 22,
    /// Macrocycle closure dummy 1 (`G1`).
    G1 = 23,
    /// Macrocycle closure dummy 2 (`G2`).
    G2 = 24,
    /// Macrocycle closure dummy 3 (`G3`).
    G3 = 25,
    /// Closure carbon 0 (`CG0`).
    Cg0 = 26,
    /// Closure carbon 1 (`CG1`).
    Cg1 = 27,
    /// Closure carbon 2 (`CG2`).
    Cg2 = 28,
    /// Closure carbon 3 (`CG3`).
    Cg3 = 29,
    /// Hydrated-ligand pseudo atom (`W`).
    W = 30,
    /// Unknown / unsupported (Vina's `AD_TYPE_SIZE`).
    Unknown = 31,
}

impl AdType {
    /// Number of *known* AD4 types (excludes [`AdType::Unknown`]).
    pub const COUNT: usize = 31;

    /// All known types, in table order.
    pub const ALL: [AdType; 31] = {
        use AdType::*;
        [
            C, A, N, O, P, S, H, F, I, NA, OA, SA, HD, Mg, Mn, Zn, Ca, Fe, Cl, Br, Si, At, G0, G1,
            G2, G3, Cg0, Cg1, Cg2, Cg3, W,
        ]
    };

    /// Discriminant as `usize`, clamped to the `Unknown` sentinel.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Build from a raw discriminant.
    #[inline]
    pub fn from_index(i: usize) -> AdType {
        AdType::ALL.get(i).copied().unwrap_or(AdType::Unknown)
    }

    /// Two-letter PDBQT name.
    pub fn name(self) -> &'static str {
        ATOM_KIND_NAMES[self.index().min(AdType::COUNT - 1)]
    }

    /// Parse the PDBQT atom-type column.
    ///
    /// Matches Vina's `string_to_ad_type`, including the `Se -> S` equivalence
    /// and the "unknown metal" fallback (`AD_TYPE_SIZE`).
    pub fn from_name(name: &str) -> AdType {
        let t = name.trim();
        for (i, n) in ATOM_KIND_NAMES.iter().enumerate() {
            if *n == t {
                return AdType::from_index(i);
            }
        }
        // Vina's atom_equivalence_data: {"Se" -> "S"}
        if t == "Se" {
            return AdType::S;
        }
        // Vina's non_ad_metal_names: metals unknown to AD4 are left untyped.
        const NON_AD_METALS: [&str; 9] = ["Cu", "Fe", "Na", "K", "Hg", "Co", "U", "Cd", "Ni"];
        if NON_AD_METALS.contains(&t) {
            return AdType::Unknown;
        }
        // A bare element symbol may still be present.
        match t.to_ascii_uppercase().as_str() {
            "MG" => AdType::Mg,
            "MN" => AdType::Mn,
            "ZN" => AdType::Zn,
            "CA" => AdType::Ca,
            _ => AdType::Unknown,
        }
    }

    /// Element implied by this AD4 type.
    pub fn element(self) -> Element {
        use AdType::*;
        match self {
            C | A | Cg0 | Cg1 | Cg2 | Cg3 => Element::C,
            N | NA => Element::N,
            O | OA => Element::O,
            S | SA => Element::S,
            P => Element::P,
            H | HD => Element::H,
            F => Element::F,
            I => Element::I,
            Cl => Element::Cl,
            Br => Element::Br,
            Si => Element::Si,
            At => Element::At,
            Mg | Mn | Zn | Ca | Fe => Element::Met,
            G0 | G1 | G2 | G3 | W => Element::Dummy,
            Unknown => Element::Size,
        }
    }

    /// `true` for non-polar and polar hydrogens.
    #[inline]
    pub fn is_hydrogen(self) -> bool {
        matches!(self, AdType::H | AdType::HD)
    }

    /// Vina's `ad_is_heteroatom`.
    #[inline]
    pub fn is_heteroatom(self) -> bool {
        !matches!(
            self,
            AdType::A | AdType::C | AdType::H | AdType::HD | AdType::Unknown
        )
    }

    /// `true` for macrocycle closure dummy atoms.
    #[inline]
    pub fn is_glue(self) -> bool {
        matches!(self, AdType::G0 | AdType::G1 | AdType::G2 | AdType::G3)
    }
}

/// `atom_kind_data[].name`, in table order (Vina `atom_constants.h`).
pub const ATOM_KIND_NAMES: [&str; 31] = [
    "C", "A", "N", "O", "P", "S", "H", "F", "I", "NA", "OA", "SA", "HD", "Mg", "Mn", "Zn", "Ca",
    "Fe", "Cl", "Br", "Si", "At", "G0", "G1", "G2", "G3", "CG0", "CG1", "CG2", "CG3", "W",
];

/// AutoDock 4 atom-kind properties.
///
/// Ported verbatim from `atom_kind_data[]` in Vina's `atom_constants.h`
/// (originally generated from `AD4_parameters.data`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtomKind {
    /// van der Waals radius (Å).
    pub radius: f64,
    /// Lennard-Jones well depth (kcal/mol).
    pub depth: f64,
    /// H-bond well depth; a pair `(i, j)` is an H-bond when
    /// `hb_depth[i] * hb_depth[j] < 0`.
    pub hb_depth: f64,
    /// H-bond radius (Å).
    pub hb_radius: f64,
    /// Desolvation parameter.
    pub solvation: f64,
    /// Atomic volume (Å³).
    pub volume: f64,
    /// Covalent radius (Å).
    pub covalent_radius: f64,
}

/// The AD4 parameter table (31 entries, index == [`AdType`] discriminant).
pub const ATOM_KIND_DATA: [AtomKind; 31] = [
    // name, radius, depth, hb_depth, hb_r, solvation, volume, covalent radius
    AtomKind { radius: 2.00000, depth: 0.15000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 }, //  0 C
    AtomKind { radius: 2.00000, depth: 0.15000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00052, volume: 33.51030, covalent_radius: 0.77 }, //  1 A
    AtomKind { radius: 1.75000, depth: 0.16000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00162, volume: 22.44930, covalent_radius: 0.75 }, //  2 N
    AtomKind { radius: 1.60000, depth: 0.20000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00251, volume: 17.15730, covalent_radius: 0.73 }, //  3 O
    AtomKind { radius: 2.10000, depth: 0.20000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 38.79240, covalent_radius: 1.06 }, //  4 P
    AtomKind { radius: 2.00000, depth: 0.20000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00214, volume: 33.51030, covalent_radius: 1.02 }, //  5 S
    AtomKind { radius: 1.00000, depth: 0.02000, hb_depth: 0.0, hb_radius: 0.0, solvation: 0.00051, volume: 0.00000, covalent_radius: 0.37 }, //  6 H
    AtomKind { radius: 1.54500, depth: 0.08000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 15.44800, covalent_radius: 0.71 }, //  7 F
    AtomKind { radius: 2.36000, depth: 0.55000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 55.05850, covalent_radius: 1.33 }, //  8 I
    AtomKind { radius: 1.75000, depth: 0.16000, hb_depth: -5.0, hb_radius: 1.9, solvation: -0.00162, volume: 22.44930, covalent_radius: 0.75 }, //  9 NA
    AtomKind { radius: 1.60000, depth: 0.20000, hb_depth: -5.0, hb_radius: 1.9, solvation: -0.00251, volume: 17.15730, covalent_radius: 0.73 }, // 10 OA
    AtomKind { radius: 2.00000, depth: 0.20000, hb_depth: -1.0, hb_radius: 2.5, solvation: -0.00214, volume: 33.51030, covalent_radius: 1.02 }, // 11 SA
    AtomKind { radius: 1.00000, depth: 0.02000, hb_depth: 1.0, hb_radius: 0.0, solvation: 0.00051, volume: 0.00000, covalent_radius: 0.37 }, // 12 HD
    AtomKind { radius: 0.65000, depth: 0.87500, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 1.56000, covalent_radius: 1.30 }, // 13 Mg
    AtomKind { radius: 0.65000, depth: 0.87500, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 2.14000, covalent_radius: 1.39 }, // 14 Mn
    AtomKind { radius: 0.74000, depth: 0.55000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 1.70000, covalent_radius: 1.31 }, // 15 Zn
    AtomKind { radius: 0.99000, depth: 0.55000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 2.77000, covalent_radius: 1.74 }, // 16 Ca
    AtomKind { radius: 0.65000, depth: 0.01000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 1.84000, covalent_radius: 1.25 }, // 17 Fe
    AtomKind { radius: 2.04500, depth: 0.27600, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 35.82350, covalent_radius: 0.99 }, // 18 Cl
    AtomKind { radius: 2.16500, depth: 0.38900, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 42.56610, covalent_radius: 1.14 }, // 19 Br
    AtomKind { radius: 2.30000, depth: 0.20000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00143, volume: 50.96500, covalent_radius: 1.11 }, // 20 Si
    AtomKind { radius: 2.40000, depth: 0.55000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00110, volume: 57.90580, covalent_radius: 1.44 }, // 21 At
    AtomKind { radius: 0.00000, depth: 0.00000, hb_depth: 0.0, hb_radius: 0.0, solvation: 0.00000, volume: 0.00000, covalent_radius: 0.77 }, // 22 G0
    AtomKind { radius: 0.00000, depth: 0.00000, hb_depth: 0.0, hb_radius: 0.0, solvation: 0.00000, volume: 0.00000, covalent_radius: 0.77 }, // 23 G1
    AtomKind { radius: 0.00000, depth: 0.00000, hb_depth: 0.0, hb_radius: 0.0, solvation: 0.00000, volume: 0.00000, covalent_radius: 0.77 }, // 24 G2
    AtomKind { radius: 0.00000, depth: 0.00000, hb_depth: 0.0, hb_radius: 0.0, solvation: 0.00000, volume: 0.00000, covalent_radius: 0.77 }, // 25 G3
    AtomKind { radius: 2.00000, depth: 0.15000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 }, // 26 CG0
    AtomKind { radius: 2.00000, depth: 0.15000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 }, // 27 CG1
    AtomKind { radius: 2.00000, depth: 0.15000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 }, // 28 CG2
    AtomKind { radius: 2.00000, depth: 0.15000, hb_depth: 0.0, hb_radius: 0.0, solvation: -0.00143, volume: 33.51030, covalent_radius: 0.77 }, // 29 CG3
    AtomKind { radius: 0.00000, depth: 0.00000, hb_depth: 0.0, hb_radius: 0.0, solvation: 0.00000, volume: 0.00000, covalent_radius: 0.00 }, // 30 W
];

/// Properties of an AD4 type (panics only for the `Unknown` sentinel).
#[inline]
pub fn ad_type_property(t: AdType) -> AtomKind {
    ATOM_KIND_DATA[t.index().min(AdType::COUNT - 1)]
}

/// Solvation parameter assigned to metals without an AD4 type.
pub const METAL_SOLVATION_PARAMETER: f64 = -0.00110;

/// Largest covalent radius in the AD4 table (Vina's `max_covalent_radius`).
pub fn max_covalent_radius() -> f64 {
    ATOM_KIND_DATA
        .iter()
        .fold(0.0_f64, |acc, k| acc.max(k.covalent_radius))
}

/// Covalent radius for an arbitrary AD4 type, falling back to the AD4 maximum
/// for types outside the table (Vina's `assign_bonds` behaviour).
#[inline]
pub fn covalent_radius(t: AdType) -> f64 {
    if t == AdType::Unknown {
        max_covalent_radius()
    } else {
        ad_type_property(t).covalent_radius
    }
}

// ---------------------------------------------------------------------------
// X-Score atom types
// ---------------------------------------------------------------------------

/// X-Score atom type used by the Vina and Vinardo force fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum XsType {
    /// Hydrophobic (aliphatic) carbon.
    CH = 0,
    /// Polar carbon (bonded to a heteroatom).
    CP = 1,
    /// Non-polar nitrogen.
    NP = 2,
    /// H-bond donating nitrogen.
    ND = 3,
    /// H-bond accepting nitrogen.
    NA = 4,
    /// Donating + accepting nitrogen.
    NDA = 5,
    /// Non-polar oxygen.
    OP = 6,
    /// H-bond donating oxygen.
    OD = 7,
    /// H-bond accepting oxygen.
    OA = 8,
    /// Donating + accepting oxygen.
    ODA = 9,
    /// Sulfur.
    SP = 10,
    /// Phosphorus.
    PP = 11,
    /// Fluorine.
    FH = 12,
    /// Chlorine.
    ClH = 13,
    /// Bromine.
    BrH = 14,
    /// Iodine.
    IH = 15,
    /// Silicon.
    Si = 16,
    /// Astatine.
    At = 17,
    /// Metal (donor).
    MetD = 18,
    /// Macrocycle closure carbon 0 (hydrophobic).
    CHCg0 = 19,
    /// Macrocycle closure carbon 0 (polar).
    CPCg0 = 20,
    /// Macrocycle closure dummy 0.
    G0 = 21,
    /// Macrocycle closure carbon 1 (hydrophobic).
    CHCg1 = 22,
    /// Macrocycle closure carbon 1 (polar).
    CPCg1 = 23,
    /// Macrocycle closure dummy 1.
    G1 = 24,
    /// Macrocycle closure carbon 2 (hydrophobic).
    CHCg2 = 25,
    /// Macrocycle closure carbon 2 (polar).
    CPCg2 = 26,
    /// Macrocycle closure dummy 2.
    G2 = 27,
    /// Macrocycle closure carbon 3 (hydrophobic).
    CHCg3 = 28,
    /// Macrocycle closure carbon 3 (polar).
    CPCg3 = 29,
    /// Macrocycle closure dummy 3.
    G3 = 30,
    /// Hydrated-ligand pseudo atom; never produced by [`XsType::from_element`].
    W = 31,
}

impl XsType {
    /// Number of X-Score types.
    pub const COUNT: usize = 32;

    /// Discriminant as `usize`.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Build from a raw discriminant, clamping invalid values to [`XsType::W`]
    /// (Vina maps "no W atoms in XS types" to `XS_TYPE_SIZE`, here `W`).
    #[inline]
    pub fn from_index(i: usize) -> XsType {
        if i >= XsType::COUNT {
            XsType::W
        } else {
            XS_ALL[i]
        }
    }

    /// van der Waals radius in the Vina force field (Å).
    #[inline]
    pub fn radius(self) -> f64 {
        XS_VDW_RADII[self.index()]
    }

    /// van der Waals radius in the Vinardo force field (Å).
    #[inline]
    pub fn vinardo_radius(self) -> f64 {
        XS_VINARDO_VDW_RADII[self.index()]
    }

    /// Vina's `xs_is_hydrophobic`.
    #[inline]
    pub fn is_hydrophobic(self) -> bool {
        matches!(
            self,
            XsType::CH | XsType::FH | XsType::ClH | XsType::BrH | XsType::IH
        )
    }

    /// Vina's `xs_is_acceptor`.
    #[inline]
    pub fn is_acceptor(self) -> bool {
        matches!(self, XsType::NA | XsType::NDA | XsType::OA | XsType::ODA)
    }

    /// Vina's `xs_is_donor`.
    #[inline]
    pub fn is_donor(self) -> bool {
        matches!(
            self,
            XsType::ND | XsType::NDA | XsType::OD | XsType::ODA | XsType::MetD
        )
    }

    /// `true` when a donor/acceptor pair exists in either order.
    #[inline]
    pub fn h_bond_possible(self, other: XsType) -> bool {
        (self.is_donor() && other.is_acceptor()) || (other.is_donor() && self.is_acceptor())
    }

    /// Vina's `is_glue_type`.
    #[inline]
    pub fn is_glue(self) -> bool {
        matches!(self, XsType::G0 | XsType::G1 | XsType::G2 | XsType::G3)
    }

    /// Compute the X-Score type from the element, the AD4 type and the
    /// *bonding graph*.
    ///
    /// Ported from `model::assign_types` (Vina, Apache-2.0). `bonded_to_hd`
    /// must be `true` when the atom is covalently bound to a polar hydrogen
    /// (`HD`), `bonded_to_heteroatom` when it is bound to a non-carbon,
    /// non-hydrogen atom.
    #[inline]
    pub fn from_element(
        el: Element,
        ad: AdType,
        bonded_to_hd: bool,
        bonded_to_heteroatom: bool,
    ) -> XsType {
        let acceptor = matches!(ad, AdType::OA | AdType::NA);
        let donor = el == Element::Met || bonded_to_hd;
        match el {
            // Hydrogens have no place in the X-Score force field: AutoDock Vina
            // leaves them at the `XS_TYPE_SIZE` sentinel so that every grid and
            // pair loop skips them. They still matter, because the donor flag of
            // their heavy neighbour is derived from their presence
            // (`bonded_to_hd`). `XsType::W` is the no-X-Score-type sentinel.
            Element::H => XsType::W,
            Element::C => match ad {
                AdType::Cg0 => {
                    if bonded_to_heteroatom {
                        XsType::CPCg0
                    } else {
                        XsType::CHCg0
                    }
                }
                AdType::Cg1 => {
                    if bonded_to_heteroatom {
                        XsType::CPCg1
                    } else {
                        XsType::CHCg1
                    }
                }
                AdType::Cg2 => {
                    if bonded_to_heteroatom {
                        XsType::CPCg2
                    } else {
                        XsType::CHCg2
                    }
                }
                AdType::Cg3 => {
                    if bonded_to_heteroatom {
                        XsType::CPCg3
                    } else {
                        XsType::CHCg3
                    }
                }
                _ => {
                    if bonded_to_heteroatom {
                        XsType::CP
                    } else {
                        XsType::CH
                    }
                }
            },
            Element::N => match (acceptor, donor) {
                (true, true) => XsType::NDA,
                (true, false) => XsType::NA,
                (false, true) => XsType::ND,
                (false, false) => XsType::NP,
            },
            Element::O => match (acceptor, donor) {
                (true, true) => XsType::ODA,
                (true, false) => XsType::OA,
                (false, true) => XsType::OD,
                (false, false) => XsType::OP,
            },
            Element::S => XsType::SP,
            Element::P => XsType::PP,
            Element::F => XsType::FH,
            Element::Cl => XsType::ClH,
            Element::Br => XsType::BrH,
            Element::I => XsType::IH,
            Element::Si => XsType::Si,
            Element::At => XsType::At,
            Element::Met => XsType::MetD,
            Element::Dummy => match ad {
                AdType::G0 => XsType::G0,
                AdType::G1 => XsType::G1,
                AdType::G2 => XsType::G2,
                AdType::G3 => XsType::G3,
                _ => XsType::W,
            },
            Element::Size => XsType::W,
        }
    }
}

impl fmt::Display for XsType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// All X-Score types in discriminant order.
pub const XS_ALL: [XsType; 32] = {
    use XsType::*;
    [
        CH, CP, NP, ND, NA, NDA, OP, OD, OA, ODA, SP, PP, FH, ClH, BrH, IH, Si, At, MetD, CHCg0,
        CPCg0, G0, CHCg1, CPCg1, G1, CHCg2, CPCg2, G2, CHCg3, CPCg3, G3, W,
    ]
};

/// Vina force-field van der Waals radii, `xs_vdw_radii[]`.
pub const XS_VDW_RADII: [f64; 32] = [
    1.9, // CH
    1.9, // CP
    1.8, // NP
    1.8, // ND
    1.8, // NA
    1.8, // NDA
    1.7, // OP
    1.7, // OD
    1.7, // OA
    1.7, // ODA
    2.0, // SP
    2.1, // PP
    1.5, // FH
    1.8, // ClH
    2.0, // BrH
    2.2, // IH
    2.2, // Si
    2.3, // At
    1.2, // MetD
    1.9, // CHCg0
    1.9, // CPCg0
    1.9, // G0
    1.9, // CHCg1
    1.9, // CPCg1
    1.9, // G1
    1.9, // CHCg2
    1.9, // CPCg2
    1.9, // G2
    1.9, // CHCg3
    1.9, // CPCg3
    1.9, // G3
    0.0, // W
];

/// Vinardo force-field van der Waals radii, `xs_vinardo_vdw_radii[]`.
pub const XS_VINARDO_VDW_RADII: [f64; 32] = [
    2.0, 2.0, 1.7, 1.7, 1.7, 1.7, 1.6, 1.6, 1.6, 1.6, 2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 2.2, 2.3,
    1.2, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0,
];

/// Vina's `optimal_distance`: the sum of X-Score radii, or `0` for glue types.
#[inline]
pub fn optimal_distance(t1: XsType, t2: XsType) -> f64 {
    if t1.is_glue() || t2.is_glue() {
        return 0.0;
    }
    t1.radius() + t2.radius()
}

/// Vina's `optimal_distance_vinardo`.
#[inline]
pub fn optimal_distance_vinardo(t1: XsType, t2: XsType) -> f64 {
    if t1.is_glue() || t2.is_glue() {
        return 0.0;
    }
    t1.vinardo_radius() + t2.vinardo_radius()
}

/// Vina's `is_glued`: which (dummy, closure-carbon) pairs are glued together.
#[inline]
pub fn is_glued(t1: XsType, t2: XsType) -> bool {
    matches!(
        (t1, t2),
        (XsType::G0, XsType::CHCg0)
            | (XsType::G0, XsType::CPCg0)
            | (XsType::CHCg0, XsType::G0)
            | (XsType::CPCg0, XsType::G0)
            | (XsType::G1, XsType::CHCg1)
            | (XsType::G1, XsType::CPCg1)
            | (XsType::CHCg1, XsType::G1)
            | (XsType::CPCg1, XsType::G1)
            | (XsType::G2, XsType::CHCg2)
            | (XsType::G2, XsType::CPCg2)
            | (XsType::CHCg2, XsType::G2)
            | (XsType::CPCg2, XsType::G2)
            | (XsType::G3, XsType::CHCg3)
            | (XsType::G3, XsType::CPCg3)
            | (XsType::CHCg3, XsType::G3)
            | (XsType::CPCg3, XsType::G3)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ad_type_table_is_consistent() {
        assert_eq!(ATOM_KIND_DATA.len(), AdType::COUNT);
        assert_eq!(ATOM_KIND_NAMES.len(), AdType::COUNT);
        // Round-trip every name.
        for (i, n) in ATOM_KIND_NAMES.iter().enumerate() {
            assert_eq!(AdType::from_name(n).index(), i, "name {n}");
        }
        assert_eq!(AdType::from_name("Se"), AdType::S);
        assert_eq!(AdType::from_name("Cu"), AdType::Unknown);
        assert_eq!(AdType::from_name("Zn"), AdType::Zn);
    }

    #[test]
    fn xs_radii_are_ordered_with_the_enum() {
        // Spot-check the indices whose order is easy to get wrong.
        assert_eq!(XsType::CH.radius(), 1.9);
        assert_eq!(XsType::MetD.radius(), 1.2);
        assert_eq!(XsType::G0.radius(), 1.9);
        assert_eq!(XsType::G2.radius(), 1.9);
        assert_eq!(XsType::W.radius(), 0.0);
        assert_eq!(XsType::CH.vinardo_radius(), 2.0);
        assert_eq!(XsType::OD.vinardo_radius(), 1.6);
        assert_eq!(XsType::W.vinardo_radius(), 0.0);
    }

    #[test]
    fn optimal_distances_ignore_glue() {
        assert_eq!(optimal_distance(XsType::G0, XsType::CH), 0.0);
        assert_eq!(optimal_distance(XsType::CH, XsType::CH), 3.8);
        assert_eq!(optimal_distance_vinardo(XsType::CH, XsType::CH), 4.0);
    }

    #[test]
    fn h_bond_flags_are_symmetric() {
        assert!(XsType::ND.h_bond_possible(XsType::OA));
        assert!(XsType::OA.h_bond_possible(XsType::ND));
        assert!(!XsType::CH.h_bond_possible(XsType::CH));
        assert!(!XsType::ND.h_bond_possible(XsType::ND));
        assert!(XsType::MetD.h_bond_possible(XsType::OA));
    }

    #[test]
    fn donor_acceptor_typing_matches_vina() {
        // Carbonyl oxygen bonded to carbon: acceptor, not a donor.
        assert_eq!(
            XsType::from_element(Element::O, AdType::OA, false, false),
            XsType::OA
        );
        // Hydroxyl oxygen: acceptor and donor.
        assert_eq!(
            XsType::from_element(Element::O, AdType::OA, true, false),
            XsType::ODA
        );
        // Backbone nitrogen with an amide hydrogen: donor only.
        assert_eq!(
            XsType::from_element(Element::N, AdType::N, true, false),
            XsType::ND
        );
        // Aromatic carbon attached to nitrogen is polar.
        assert_eq!(
            XsType::from_element(Element::C, AdType::A, false, true),
            XsType::CP
        );
        // Aromatic carbon in a pure hydrocarbon ring is hydrophobic.
        assert_eq!(
            XsType::from_element(Element::C, AdType::A, false, false),
            XsType::CH
        );
        assert_eq!(
            XsType::from_element(Element::Met, AdType::Zn, false, false),
            XsType::MetD
        );
    }

    #[test]
    fn glue_pairs() {
        assert!(is_glued(XsType::G0, XsType::CHCg0));
        assert!(!is_glued(XsType::G0, XsType::CHCg1));
        assert!(!is_glued(XsType::CH, XsType::CH));
    }

    #[test]
    fn max_covalent_radius_is_ca() {
        assert!((max_covalent_radius() - 1.74).abs() < 1e-12);
    }
}
